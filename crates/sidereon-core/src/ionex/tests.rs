//! 0-ULP parity tests for the IONEX slant ionospheric delay pipeline.
//!
//! These assert the Rust port reproduces the canonical reference recipe
//! `parity/generator/ionex.py` bit-for-bit, using the committed golden fixture
//! `parity/fixtures/ionex_golden.json` and the synthetic IONEX product the
//! fixture was generated from. Values are serialised as hex-float (Python
//! `float.hex()`) so there is no decimal-parse ambiguity, and parity is measured
//! as ULP distance via the integer reinterpretation of the IEEE-754 bit pattern.
//!
//! Two things are checked. First the Rust IONEX parser is run on the committed
//! synthetic product and its grid (latitude/longitude node axes, every per-map
//! TEC value, and the derived J2000-second epoch view) is asserted bit-for-bit
//! against the parser substrate recorded in the golden, so the byte/record
//! reader is pinned. Second the float pipeline is run per case and every intermediate
//! (pierce point, per-map bilinear VTEC, time-blended VTEC, STEC, meters) is
//! checked per component, so a divergence is localised to a single algorithm
//! step. The cases span the branch matrix: longitude wrap at both seams,
//! descending-latitude bracket, EXPONENT scaling, pierce-point latitude clamp at
//! the grid edge, epoch coincident with a map versus interior, endpoint hold
//! before the first and after the last map, and the L1/L2/L5 frequency scaling.

use std::path::PathBuf;

use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};
use crate::astro::time::split_julian_date;
use serde_json::Value;

use super::grid::Ionex;
use super::slant::{slant_delay_components, PierceLineOfSight, SlantComponents, VtecGridView};
use super::{
    galileo_nequick_g_native, ionosphere_delay, GalileoNequickCoeffs, GalileoNequickEval, IonoModel,
};

/// Parse a C99 / Python `float.hex()` hex-float string into the exact `f64`.
///
/// Every hex frac digit is 4 mantissa bits and f64 has 52, so the reconstruction
/// of a 13-digit fraction is exact (no rounding).
fn parse_hex_float(s: &str) -> f64 {
    let s = s.trim();
    let (neg, rest) = match s.strip_prefix('-') {
        Some(r) => (true, r),
        None => (false, s),
    };
    let rest = rest
        .strip_prefix("0x")
        .or_else(|| rest.strip_prefix("0X"))
        .unwrap_or_else(|| panic!("not a hex float (missing 0x): {s:?}"));

    let (mantissa, exp_str) = rest
        .split_once(['p', 'P'])
        .unwrap_or_else(|| panic!("not a hex float (missing p exponent): {s:?}"));
    let exp2: i32 = exp_str
        .parse()
        .unwrap_or_else(|_| panic!("bad binary exponent in {s:?}"));

    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mantissa, ""),
    };

    let int_val: f64 = i64::from_str_radix(int_part, 16)
        .unwrap_or_else(|_| panic!("bad integer hex digits in {s:?}"))
        as f64;

    let mut frac_val = 0.0f64;
    let mut scale = 1.0f64 / 16.0;
    for c in frac_part.chars() {
        let d = c
            .to_digit(16)
            .unwrap_or_else(|| panic!("bad hex frac digit {c:?} in {s:?}"));
        frac_val += (d as f64) * scale;
        scale /= 16.0;
    }

    let significand = int_val + frac_val;
    let val = significand * 2.0f64.powi(exp2);
    if neg {
        -val
    } else {
        val
    }
}

/// ULP distance between two `f64`, using the monotone signed-integer mapping of
/// the IEEE-754 bit pattern. Returns `u64::MAX` for any NaN so a NaN never
/// silently reads as 0 ULP.
fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    ordered_i64(a).abs_diff(ordered_i64(b))
}

/// Map an `f64` to a sign-magnitude-ordered `i64` so adjacent floats differ by 1.
fn ordered_i64(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    if bits < 0 {
        i64::MIN - bits
    } else {
        bits
    }
}

/// Render an `f64` as a Python-`float.hex()`-style string for diagnostics.
fn float_hex(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0x0.0p+0".into()
        } else {
            "0x0.0p+0".into()
        };
    }
    let bits = x.to_bits();
    let sign = if (bits >> 63) & 1 == 1 { "-" } else { "" };
    let exp = ((bits >> 52) & 0x7ff) as i64;
    let mantissa = bits & 0x000f_ffff_ffff_ffff;
    let unbiased = exp - 1023;
    if unbiased >= 0 {
        format!("{sign}0x1.{mantissa:013x}p+{unbiased}")
    } else {
        format!("{sign}0x1.{mantissa:013x}p{unbiased}")
    }
}

fn fixtures_dir() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .join("tests/fixtures")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!(
                "cannot locate tests/fixtures from {}: {e}",
                crate_dir.display()
            )
        })
}

fn midnight_epoch(year: i32, month: i32, day: i32) -> Instant {
    let (jd_whole, fraction) = split_julian_date(year, month, day, 0, 0, 0.0);
    Instant::from_julian_date(
        TimeScale::Gst,
        JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date"),
    )
}

fn hexf(v: &Value, key: &str) -> f64 {
    parse_hex_float(
        v[key]
            .as_str()
            .unwrap_or_else(|| panic!("missing/non-string {key}")),
    )
}

/// Compare a parsed `f64` against a golden hex-float, recording any nonzero ULP.
fn check(failures: &mut Vec<String>, label: String, got: f64, want_hex: &str) {
    let want = parse_hex_float(want_hex);
    let ulp = ulp_distance(got, want);
    if ulp != 0 {
        failures.push(format!(
            "{label}: {ulp} ULP (rust={} ref={want_hex})",
            float_hex(got)
        ));
    }
}

#[test]
fn ionex_slant_zero_ulp_full_branch_matrix() {
    let fx = fixtures_dir();
    let golden_path = fx.join("ionex_golden.json");
    let raw = std::fs::read_to_string(&golden_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", golden_path.display()));
    let doc: Value = serde_json::from_str(&raw).expect("parse ionex_golden.json");

    // Self-check the hex-float parser/serialiser round-trips a known bit pattern,
    // so a parser bug can never masquerade as parity.
    let probe = "0x1.921fb54442d18p+1"; // math.pi
    assert_eq!(
        float_hex(parse_hex_float(probe)),
        probe,
        "hex-float parser/serialiser round-trip is broken"
    );

    // Parse the committed synthetic IONEX product with the Rust parser.
    let file_meta = &doc["ionex_file"];
    let ionex_name = file_meta["name"].as_str().expect("ionex file name");
    let ionex_path = fx.join("ionex").join(ionex_name);
    let ionex_bytes =
        std::fs::read(&ionex_path).unwrap_or_else(|e| panic!("read {}: {e}", ionex_path.display()));
    let ionex = Ionex::parse(&ionex_bytes).expect("parse synthetic IONEX product");

    let mut failures: Vec<String> = Vec::new();

    // ---- Parser parity: node axes, epochs, and every TEC value ----
    let lat_ref = file_meta["lat_arr"].as_array().expect("lat_arr");
    let lon_ref = file_meta["lon_arr"].as_array().expect("lon_arr");
    assert_eq!(
        ionex.lat_nodes_deg().len(),
        lat_ref.len(),
        "parsed latitude node count"
    );
    assert_eq!(
        ionex.lon_nodes_deg().len(),
        lon_ref.len(),
        "parsed longitude node count"
    );
    for (i, want) in lat_ref.iter().enumerate() {
        check(
            &mut failures,
            format!("lat_arr[{i}]"),
            ionex.lat_nodes_deg()[i],
            want.as_str().unwrap(),
        );
    }
    for (j, want) in lon_ref.iter().enumerate() {
        check(
            &mut failures,
            format!("lon_arr[{j}]"),
            ionex.lon_nodes_deg()[j],
            want.as_str().unwrap(),
        );
    }

    let exp_ref = file_meta["exponent"].as_f64().expect("exponent") as i32;
    assert_eq!(ionex.exponent(), exp_ref, "parsed EXPONENT");

    assert_eq!(
        ionex.map_epochs().len(),
        ionex.map_epochs_s().len(),
        "instant and J2000-second epoch views differ in count"
    );
    let epochs_ref = file_meta["map_epochs_s"].as_array().expect("map_epochs_s");
    assert_eq!(
        ionex.map_epochs_s().len(),
        epochs_ref.len(),
        "parsed map count"
    );
    for (m, want) in epochs_ref.iter().enumerate() {
        assert_eq!(
            ionex.map_epochs_s()[m],
            want.as_i64().expect("epoch int"),
            "parsed map epoch[{m}] (J2000 seconds)"
        );
    }

    let maps_ref = file_meta["maps_vtec"].as_array().expect("maps_vtec");
    assert_eq!(
        ionex.tec_maps().len(),
        maps_ref.len(),
        "parsed TEC map count"
    );
    for (m, map_ref) in maps_ref.iter().enumerate() {
        let rows = map_ref.as_array().unwrap();
        for (i, row_ref) in rows.iter().enumerate() {
            let cols = row_ref.as_array().unwrap();
            for (j, want) in cols.iter().enumerate() {
                check(
                    &mut failures,
                    format!("maps_vtec[{m}][{i}][{j}]"),
                    ionex.tec_maps()[m][i][j],
                    want.as_str().unwrap(),
                );
            }
        }
    }

    // ---- Pipeline parity: every intermediate, per case ----
    let cases = doc["cases"].as_array().expect("cases array");
    assert!(
        cases.len() >= 12,
        "expected the full branch matrix (>= 12 cases), found {}",
        cases.len()
    );

    let lat_arr = ionex.lat_nodes_deg();
    let lon_arr = ionex.lon_nodes_deg();
    let dlat = ionex.dlat_deg();
    let dlon = ionex.dlon_deg();
    let re = ionex.base_radius_km();
    let h = ionex.shell_height_km();
    let epochs = ionex.map_epochs();
    let maps = ionex.tec_maps();

    let mut checks = 0usize;

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let exp = &case["expect"];

        let epoch_s = inp["epoch_s"].as_i64().expect("epoch_s int");

        let got = slant_delay_components(
            PierceLineOfSight {
                lat_rad: hexf(inp, "lat_rad"),
                lon_rad: hexf(inp, "lon_rad"),
                az_rad: hexf(inp, "az_rad"),
                el_rad: hexf(inp, "el_rad"),
            },
            hexf(inp, "frequency_hz"),
            re,
            h,
            epoch_s,
            VtecGridView {
                map_epochs: epochs,
                maps,
                lat_arr,
                lon_arr,
                dlat,
                dlon,
            },
        );

        // The temporal-bracket index is a discrete branch outcome, not a float.
        assert_eq!(
            got.map_index as i64,
            case["map_index"].as_i64().expect("map_index"),
            "case {name}: temporal bracket index"
        );

        let components: &[(&str, f64)] = &[
            ("s", got.s),
            ("psi", got.psi),
            ("phi_ipp_deg", got.phi_ipp_deg),
            ("lambda_ipp_deg_raw", got.lambda_ipp_deg_raw),
            ("lambda_ipp_deg", got.lambda_ipp_deg),
            ("w", got.w),
            ("vtec0", got.vtec0),
            ("vtec1", got.vtec1),
            ("p0", got.p0),
            ("q0", got.q0),
            ("vtec", got.vtec),
            ("m", got.m),
            ("stec", got.stec),
            ("delay_m", got.delay_m),
        ];

        for &(c, value) in components {
            let want_hex = exp[c]
                .as_str()
                .unwrap_or_else(|| panic!("case {name}: missing expected component {c}"));
            check(&mut failures, format!("{name}.{c}"), value, want_hex);
            checks += 1;
        }
    }

    assert!(checks > 0, "no components were checked - fixture empty?");
    assert!(
        failures.is_empty(),
        "IONEX Rust port diverged from the reference recipe on {} components:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn ionex_map_epochs_are_utc_instants_with_exact_j2000_seconds_view() {
    let fx = fixtures_dir();
    let path = fx.join("ionex/synthetic_2map_7x7.20i");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let ionex = Ionex::parse(&bytes).expect("parse synthetic IONEX product");
    let epoch_seconds = ionex.map_epochs_s();

    assert_eq!(epoch_seconds, vec![646_228_800, 646_236_000]);
    assert_eq!(ionex.map_epochs().len(), epoch_seconds.len());
    for (epoch, seconds) in ionex.map_epochs().iter().zip(epoch_seconds) {
        assert_eq!(epoch.scale, TimeScale::Utc);
        assert_eq!(
            super::j2000_seconds_from_instant(*epoch),
            Some(seconds),
            "IONEX UTC instant must recover the integer J2000-second map epoch"
        );
        assert!(
            matches!(epoch.repr, InstantRepr::JulianDate(_)),
            "IONEX epoch should use the split-Julian-date instant representation"
        );
    }
}

#[test]
fn galileo_nequick_wrapper_uses_epoch_day_of_year() {
    let receiver =
        crate::frame::Wgs84Geodetic::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 0.0)
            .expect("valid WGS84 geodetic position");
    let elevation_rad = 37.0_f64.to_radians();
    let azimuth_rad = 122.0_f64.to_radians();
    let frequency_hz = crate::frequencies::frequency_hz(
        crate::GnssSystem::Galileo,
        crate::frequencies::CarrierBand::E1,
    )
    .expect("canonical Galileo E1 carrier exists");
    let coeffs = GalileoNequickCoeffs {
        ai0: 65.0,
        ai1: 0.25,
        ai2: -0.02,
    };
    let model = IonoModel::GalileoNequickG(coeffs);

    let spring = midnight_epoch(2021, 3, 21);
    let autumn = midnight_epoch(2021, 9, 22);
    let mid_year = midnight_epoch(2021, 7, 2);

    let spring_delay = ionosphere_delay(
        receiver,
        elevation_rad,
        azimuth_rad,
        spring,
        frequency_hz,
        &model,
    )
    .expect("valid Galileo ionosphere inputs");
    let autumn_delay = ionosphere_delay(
        receiver,
        elevation_rad,
        azimuth_rad,
        autumn,
        frequency_hz,
        &model,
    )
    .expect("valid Galileo ionosphere inputs");

    assert_ne!(
        spring_delay.to_bits(),
        autumn_delay.to_bits(),
        "different seasons should not collapse to the same Galileo delay"
    );

    for (epoch, day_of_year) in [(spring, 80.0), (autumn, 265.0), (mid_year, 183.0)] {
        let wrapper = ionosphere_delay(
            receiver,
            elevation_rad,
            azimuth_rad,
            epoch,
            frequency_hz,
            &model,
        )
        .expect("valid Galileo ionosphere wrapper inputs");
        let native = galileo_nequick_g_native(
            &coeffs,
            GalileoNequickEval {
                lat_deg: receiver.lat_rad * crate::constants::RAD_TO_DEG,
                lon_deg: receiver.lon_rad * crate::constants::RAD_TO_DEG,
                el_deg: elevation_rad * crate::constants::RAD_TO_DEG,
                t_gal_s: super::gps_second_of_day(epoch),
                day_of_year,
                frequency_hz,
            },
        )
        .expect("valid Galileo native inputs");
        assert_eq!(
            wrapper.to_bits(),
            native.to_bits(),
            "wrapper should pass day-of-year {day_of_year} into the native Galileo kernel"
        );
    }
}

fn valid_klobuchar_model() -> IonoModel {
    IonoModel::Klobuchar(super::KlobucharParams {
        alpha: [0.0, 0.0, 0.0, 0.0],
        beta: [90_000.0, 0.0, 0.0, 0.0],
    })
}

fn valid_ionosphere_epoch() -> Instant {
    Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(2_451_544.5, 0.25).expect("valid split Julian date"),
    )
}

fn valid_ionosphere_receiver() -> crate::frame::Wgs84Geodetic {
    crate::frame::Wgs84Geodetic::new(45.0_f64.to_radians(), 8.0_f64.to_radians(), 400.0)
        .expect("valid WGS84 geodetic position")
}

fn assert_invalid_input<T: core::fmt::Debug>(result: crate::Result<T>) {
    let err = result.expect_err("invalid ionosphere input must be rejected");
    assert!(
        matches!(err, crate::error::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn ionosphere_delay_rejects_invalid_public_inputs() {
    let receiver = valid_ionosphere_receiver();
    let epoch = valid_ionosphere_epoch();
    let model = valid_klobuchar_model();
    let frequency_hz = crate::frequencies::frequency_hz(
        crate::GnssSystem::Gps,
        crate::frequencies::CarrierBand::L1,
    )
    .expect("canonical GPS L1 carrier exists");

    let bad_receiver = crate::frame::Wgs84Geodetic {
        lat_rad: f64::NAN,
        lon_rad: receiver.lon_rad,
        height_m: receiver.height_m,
    };
    assert_invalid_input(ionosphere_delay(
        bad_receiver,
        30.0_f64.to_radians(),
        10.0_f64.to_radians(),
        epoch,
        frequency_hz,
        &model,
    ));

    assert_invalid_input(ionosphere_delay(
        receiver,
        f64::NAN,
        10.0_f64.to_radians(),
        epoch,
        frequency_hz,
        &model,
    ));
    assert_invalid_input(ionosphere_delay(
        receiver,
        -1.0e-6,
        10.0_f64.to_radians(),
        epoch,
        frequency_hz,
        &model,
    ));
    assert_invalid_input(ionosphere_delay(
        receiver,
        30.0_f64.to_radians(),
        f64::INFINITY,
        epoch,
        frequency_hz,
        &model,
    ));

    let bad_epoch = Instant {
        scale: TimeScale::Gpst,
        repr: InstantRepr::JulianDate(JulianDateSplit {
            jd_whole: f64::NAN,
            fraction: 0.0,
        }),
    };
    assert_invalid_input(ionosphere_delay(
        receiver,
        30.0_f64.to_radians(),
        10.0_f64.to_radians(),
        bad_epoch,
        frequency_hz,
        &model,
    ));

    assert_invalid_input(ionosphere_delay(
        receiver,
        30.0_f64.to_radians(),
        10.0_f64.to_radians(),
        epoch,
        f64::INFINITY,
        &model,
    ));

    assert_invalid_input(ionosphere_delay(
        receiver,
        30.0_f64.to_radians(),
        10.0_f64.to_radians(),
        epoch,
        f64::MIN_POSITIVE,
        &model,
    ));

    let bad_model = IonoModel::GalileoNequickG(GalileoNequickCoeffs {
        ai0: 63.7,
        ai1: f64::NAN,
        ai2: 0.0,
    });
    assert_invalid_input(ionosphere_delay(
        receiver,
        30.0_f64.to_radians(),
        10.0_f64.to_radians(),
        epoch,
        frequency_hz,
        &bad_model,
    ));
}

#[test]
fn ionosphere_delay_accepts_west_antimeridian_receiver() {
    let receiver = crate::frame::Wgs84Geodetic {
        lat_rad: 0.0,
        lon_rad: -core::f64::consts::PI,
        height_m: 0.0,
    };
    let epoch = valid_ionosphere_epoch();
    let params = super::KlobucharParams {
        alpha: [0.0, 0.0, 0.0, 0.0],
        beta: [90_000.0, 0.0, 0.0, 0.0],
    };
    let model = IonoModel::Klobuchar(params);
    let frequency_hz = crate::frequencies::frequency_hz(
        crate::GnssSystem::Gps,
        crate::frequencies::CarrierBand::L1,
    )
    .expect("canonical GPS L1 carrier exists");

    let wrapped = ionosphere_delay(
        receiver,
        30.0_f64.to_radians(),
        10.0_f64.to_radians(),
        epoch,
        frequency_hz,
        &model,
    )
    .expect("west antimeridian ionosphere receiver is valid");
    assert!(wrapped.is_finite() && wrapped > 0.0);

    let direct = super::klobuchar(
        &params,
        receiver,
        30.0_f64.to_radians(),
        10.0_f64.to_radians(),
        epoch,
        frequency_hz,
    )
    .expect("west antimeridian Klobuchar receiver is valid");
    assert!(direct.is_finite() && direct > 0.0);
}

#[test]
fn ionosphere_native_helpers_reject_invalid_domains() {
    let params = super::KlobucharParams {
        alpha: [0.0, 0.0, 0.0, 0.0],
        beta: [90_000.0, 0.0, 0.0, 0.0],
    };
    assert_invalid_input(super::klobuchar_native(
        &params,
        91.0,
        0.0,
        0.0,
        30.0,
        12_000.0,
        1_575_420_000.0,
    ));
    assert_invalid_input(super::klobuchar_native(
        &super::KlobucharParams {
            alpha: [0.0, f64::NAN, 0.0, 0.0],
            beta: [90_000.0, 0.0, 0.0, 0.0],
        },
        45.0,
        0.0,
        0.0,
        30.0,
        12_000.0,
        1_575_420_000.0,
    ));

    assert_invalid_input(galileo_nequick_g_native(
        &GalileoNequickCoeffs {
            ai0: 63.7,
            ai1: 0.0,
            ai2: 0.0,
        },
        GalileoNequickEval {
            lat_deg: 45.0,
            lon_deg: 8.0,
            el_deg: 30.0,
            t_gal_s: 86_400.0,
            day_of_year: 80.0,
            frequency_hz: 1_575_420_000.0,
        },
    ));

    let receiver = crate::frame::Wgs84Geodetic {
        lat_rad: 0.0,
        lon_rad: -core::f64::consts::PI,
        height_m: 0.0,
    };
    let ionex = Ionex::parse(
        &std::fs::read(fixtures_dir().join("ionex/synthetic_2map_7x7.20i"))
            .expect("read IONEX fixture"),
    )
    .expect("parse IONEX fixture");
    super::ionex_slant_delay(
        &ionex,
        receiver,
        30.0_f64.to_radians(),
        0.0,
        ionex.map_epochs_s()[0],
        1_575_420_000.0,
    )
    .expect("west antimeridian receiver is valid");

    let bad_receiver = crate::frame::Wgs84Geodetic {
        lat_rad: 0.0,
        lon_rad: -core::f64::consts::PI - 1.0e-12,
        height_m: 0.0,
    };
    assert_invalid_input(super::ionex_slant_delay(
        &ionex,
        bad_receiver,
        30.0_f64.to_radians(),
        0.0,
        ionex.map_epochs_s()[0],
        1_575_420_000.0,
    ));
}

/// Regression: a single-map IONEX product must not panic in the temporal
/// bracket. The multi-map path computes `nmaps - 2`, which underflows (usize)
/// for one map and then indexes a second, non-existent map. A one-map product
/// has no interval to interpolate, so it holds that single map; querying it must
/// return the same value the equivalent two-map product returns at its first
/// epoch (where the temporal weight is 0).
#[test]
fn ionex_single_map_does_not_panic_and_holds_the_map() {
    let fx = fixtures_dir();
    let two_map_path = fx.join("ionex/synthetic_2map_7x7.20i");
    let full = std::fs::read_to_string(&two_map_path)
        .unwrap_or_else(|e| panic!("read {}: {e}", two_map_path.display()));
    let lines: Vec<&str> = full.lines().collect();
    let hdr_end = lines
        .iter()
        .position(|l| l.contains("END OF HEADER"))
        .expect("END OF HEADER");
    let first_map_end = lines
        .iter()
        .position(|l| l.contains("END OF TEC MAP"))
        .expect("END OF TEC MAP");

    // Build a one-map product by reusing the real file's exact formatting:
    // header (with the maps-count digit forced to 1) + the first map block only.
    let mut single = String::new();
    for l in &lines[..=hdr_end] {
        if l.contains("# OF MAPS IN FILE") {
            single.push_str(&l.replacen('2', "1", 1));
        } else {
            single.push_str(l);
        }
        single.push('\n');
    }
    for l in &lines[(hdr_end + 1)..=first_map_end] {
        single.push_str(l);
        single.push('\n');
    }

    let one = super::Ionex::parse(single.as_bytes()).expect("parse single-map IONEX");
    assert_eq!(one.map_epochs_s().len(), 1, "expected exactly one map");
    let two = super::Ionex::parse(full.as_bytes()).expect("parse two-map IONEX");

    let receiver = crate::frame::Wgs84Geodetic::new(30.0_f64.to_radians(), 0.0, 0.0)
        .expect("valid WGS84 geodetic position");
    let el = 45.0_f64.to_radians();
    let az = 90.0_f64.to_radians();
    let f_l1 = crate::frequencies::frequency_hz(
        crate::GnssSystem::Gps,
        crate::frequencies::CarrierBand::L1,
    )
    .expect("canonical GPS L1 carrier exists");
    let epoch0 = one.map_epochs_s()[0];

    // Must not panic, and must be a finite positive delay.
    let d_one =
        super::ionex_slant_delay(&one, receiver, el, az, epoch0, f_l1).expect("valid IONEX delay");
    assert!(
        d_one.is_finite() && d_one > 0.0,
        "single-map delay not finite/positive: {d_one}"
    );

    // At its first epoch the two-map product weights the first map only (w == 0),
    // so the single-map hold must reproduce it bit-for-bit.
    let d_two = super::ionex_slant_delay(&two, receiver, el, az, epoch0, f_l1)
        .expect("valid two-map IONEX delay");
    assert_eq!(
        d_one.to_bits(),
        d_two.to_bits(),
        "single-map delay {d_one} != two-map-at-first-epoch {d_two}"
    );
}

fn equatorial_zenith_components(
    lon_deg: f64,
    lon_arr: &[f64],
    maps: &[Vec<Vec<f64>>],
) -> SlantComponents {
    let epochs = [super::ionex_epoch_from_j2000_seconds(0)];
    let lat_arr = [0.0, -1.0];
    slant_delay_components(
        PierceLineOfSight {
            lat_rad: 0.0,
            lon_rad: lon_deg.to_radians(),
            az_rad: 0.0,
            el_rad: 90.0_f64.to_radians(),
        },
        1_575_420_000.0,
        6371.0,
        450.0,
        0,
        VtecGridView {
            map_epochs: &epochs,
            maps,
            lat_arr: &lat_arr,
            lon_arr,
            dlat: -1.0,
            dlon: lon_arr[1] - lon_arr[0],
        },
    )
}

fn assert_close(got: f64, want: f64) {
    assert!((got - want).abs() <= 1.0e-12, "got {got}, want {want}");
}

#[test]
fn ionex_regional_longitudes_hold_edges_without_extrapolation() {
    let lon_arr = [0.0, 90.0, 180.0];
    let maps = vec![vec![vec![0.0, 90.0, 180.0], vec![0.0, 90.0, 180.0]]];

    let inside = equatorial_zenith_components(45.0, &lon_arr, &maps);
    assert_close(inside.lambda_ipp_deg, 45.0);
    assert_close(inside.p0, 0.5);
    assert_close(inside.vtec, 45.0);

    let east = equatorial_zenith_components(240.0, &lon_arr, &maps);
    assert_close(east.lambda_ipp_deg, 180.0);
    assert_close(east.p0, 1.0);
    assert_close(east.vtec, 180.0);

    let west = equatorial_zenith_components(-30.0, &lon_arr, &maps);
    assert_close(west.lambda_ipp_deg, 0.0);
    assert_close(west.p0, 0.0);
    assert_close(west.vtec, 0.0);
}

#[test]
fn ionex_global_longitudes_still_wrap() {
    let lon_arr = [0.0, 90.0, 180.0, 270.0, 360.0];
    let maps = vec![vec![
        vec![0.0, 90.0, 180.0, 270.0, 360.0],
        vec![0.0, 90.0, 180.0, 270.0, 360.0],
    ]];

    let wrapped = equatorial_zenith_components(-90.0, &lon_arr, &maps);
    let direct = equatorial_zenith_components(270.0, &lon_arr, &maps);
    assert_close(wrapped.lambda_ipp_deg, 270.0);
    assert_close(wrapped.p0, direct.p0);
    assert_close(wrapped.vtec, direct.vtec);

    let over = equatorial_zenith_components(450.0, &lon_arr, &maps);
    assert_close(over.lambda_ipp_deg, 90.0);
    assert_close(over.vtec, 90.0);
}

/// A degenerate grid with fewer than two nodes on an axis must be rejected at
/// parse time, not accepted and then panicked on at evaluation (bilinear
/// interpolation brackets a cell with `node[i+1]`). Build a one-latitude-node
/// product by forcing `LAT2 == LAT1` and keeping a single band, reusing the real
/// fixture's column layout otherwise.
#[test]
fn ionex_degenerate_single_node_axis_is_rejected_at_parse() {
    let fx = fixtures_dir();
    let path = fx.join("ionex/synthetic_2map_7x7.20i");
    let full =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    let lines: Vec<&str> = full.lines().collect();
    let hdr_end = lines
        .iter()
        .position(|l| l.contains("END OF HEADER"))
        .unwrap();
    let map_start = hdr_end + 1; // START OF TEC MAP
    let map_end = lines
        .iter()
        .position(|l| l.contains("END OF TEC MAP"))
        .unwrap();

    let mut s = String::new();
    for l in &lines[..=hdr_end] {
        if l.contains("LAT1 / LAT2 / DLAT") {
            // 60.0 / -60.0 / -20.0 -> 60.0 / 60.0 / -20.0 == a single latitude node.
            s.push_str(&l.replacen("-60.0", " 60.0", 1));
        } else if l.contains("# OF MAPS IN FILE") {
            s.push_str(&l.replacen('2', "1", 1));
        } else {
            s.push_str(l);
        }
        s.push('\n');
    }
    // One map, first band only (latitude 60.0 with its seven longitude values).
    for l in &[
        lines[map_start],
        lines[map_start + 1],
        lines[map_start + 2],
        lines[map_start + 3],
        lines[map_end],
    ] {
        s.push_str(l);
        s.push('\n');
    }

    let parsed = super::Ionex::parse(s.as_bytes());
    assert!(
        parsed.is_err(),
        "degenerate single-node grid should be rejected"
    );
    let msg = format!("{:?}", parsed.err().unwrap()).to_lowercase();
    assert!(
        msg.contains("node"),
        "expected a node-count parse error, got: {msg}"
    );
}

#[test]
fn ionex_malformed_axis_header_returns_parse_error_not_panic() {
    let fx = fixtures_dir();
    let path = fx.join("ionex/synthetic_2map_7x7.20i");
    let full =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let cases = [
        (
            "non-finite latitude start",
            "LAT1 / LAT2 / DLAT",
            "inf -60.0 -20.0",
        ),
        (
            "out-of-domain latitude start",
            "LAT1 / LAT2 / DLAT",
            "1.0e9 -60.0 -20.0",
        ),
        (
            "huge longitude node count",
            "LON1 / LON2 / DLON",
            "-180.0 180.0 1.0e-9",
        ),
    ];

    for (name, label, data) in cases {
        let malformed = replace_ionex_record_data(&full, label, data);
        let parsed = std::panic::catch_unwind(|| Ionex::parse_str(&malformed))
            .unwrap_or_else(|_| panic!("{name} panicked"));
        let err = parsed.unwrap_err();
        assert!(
            matches!(err, crate::error::Error::Parse(_)),
            "{name}: expected Parse, got {err:?}"
        );
    }
}

#[test]
fn ionex_fine_axis_header_parses_expected_grid() {
    let mut text = String::new();
    text.push_str(&ionex_record("1.0", "IONEX VERSION / TYPE"));
    text.push_str(&ionex_record("1", "# OF MAPS IN FILE"));
    text.push_str(&ionex_record("1.0 0.0 -0.1", "LAT1 / LAT2 / DLAT"));
    text.push_str(&ionex_record("0.0 1.0 0.1", "LON1 / LON2 / DLON"));
    text.push_str(&ionex_record("450.0 450.0 0.0", "HGT1 / HGT2 / DHGT"));
    text.push_str(&ionex_record("6371.0", "BASE RADIUS"));
    text.push_str(&ionex_record("0", "EXPONENT"));
    text.push_str(&ionex_record("", "END OF HEADER"));
    text.push_str(&ionex_record("1", "START OF TEC MAP"));
    text.push_str(&ionex_record("2020 1 1 0 0 0", "EPOCH OF CURRENT MAP"));
    for lat_idx in 0..11 {
        let lat = 1.0 - (lat_idx as f64) * 0.1;
        text.push_str(&ionex_record(
            &format!("{lat:.1} 0.0 1.0 0.1 450.0"),
            "LAT/LON1/LON2/DLON/H",
        ));
        text.push_str("0 1 2 3 4 5 6 7 8 9 10\n");
    }
    text.push_str(&ionex_record("1", "END OF TEC MAP"));

    let ionex = Ionex::parse_str(&text).expect("valid fine-axis IONEX grid parses");
    assert_eq!(ionex.lat_nodes_deg().len(), 11);
    assert_eq!(ionex.lon_nodes_deg().len(), 11);
    assert_eq!(ionex.lat_nodes_deg()[0], 1.0);
    assert_eq!(ionex.lat_nodes_deg()[10], 0.0);
    assert_eq!(ionex.lon_nodes_deg()[0], 0.0);
    assert_eq!(ionex.lon_nodes_deg()[10], 1.0);
    assert_eq!(ionex.tec_maps()[0][0].len(), 11);
}

#[test]
fn ionex_valid_axis_header_parses_expected_grid() {
    let fx = fixtures_dir();
    let path = fx.join("ionex/synthetic_2map_7x7.20i");
    let full =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let ionex = Ionex::parse_str(&full).expect("valid synthetic IONEX grid parses");
    assert_eq!(
        ionex.lat_nodes_deg(),
        &[60.0, 40.0, 20.0, 0.0, -20.0, -40.0, -60.0]
    );
    assert_eq!(
        ionex.lon_nodes_deg(),
        &[-180.0, -120.0, -60.0, 0.0, 60.0, 120.0, 180.0]
    );
}

#[test]
fn ionex_round_trips_through_the_serializer() {
    let fx = fixtures_dir();
    let path = fx.join("ionex/synthetic_2map_7x7.20i");
    let full =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    let original = Ionex::parse_str(&full).expect("parse synthetic IONEX product");
    assert_eq!(original.skipped_records(), 0, "clean product has no skips");

    // Encode -> parse must reproduce the canonical IR bit-for-bit, and a second
    // encode of the reparsed product must be byte-identical (deterministic).
    let encoded = original.to_ionex_string();
    let reparsed = Ionex::parse_str(&encoded).expect("serialized IONEX reparses");
    assert_eq!(reparsed, original, "round-trip preserves the IONEX IR");
    assert_eq!(
        reparsed.to_ionex_string(),
        encoded,
        "serializer is deterministic"
    );
}

#[test]
fn ionex_round_trips_a_product_with_rms_maps() {
    // EXPONENT 0 keeps the scaled-integer fields exact, so the round-trip checks
    // the band/serializer plumbing (including the RMS branch) directly.
    let mut text = String::new();
    text.push_str(&ionex_record("1.0", "IONEX VERSION / TYPE"));
    text.push_str(&ionex_record("1", "# OF MAPS IN FILE"));
    text.push_str(&ionex_record("1.0 0.0 -1.0", "LAT1 / LAT2 / DLAT"));
    text.push_str(&ionex_record("0.0 1.0 1.0", "LON1 / LON2 / DLON"));
    text.push_str(&ionex_record("450.0 450.0 0.0", "HGT1 / HGT2 / DHGT"));
    text.push_str(&ionex_record("6371.0", "BASE RADIUS"));
    text.push_str(&ionex_record("0", "EXPONENT"));
    text.push_str(&ionex_record("", "END OF HEADER"));
    text.push_str(&ionex_record("1", "START OF TEC MAP"));
    text.push_str(&ionex_record("2020 1 1 0 0 0", "EPOCH OF CURRENT MAP"));
    text.push_str(&ionex_record(
        "1.0 0.0 1.0 1.0 450.0",
        "LAT/LON1/LON2/DLON/H",
    ));
    text.push_str("10 11\n");
    text.push_str(&ionex_record(
        "0.0 0.0 1.0 1.0 450.0",
        "LAT/LON1/LON2/DLON/H",
    ));
    text.push_str("12 13\n");
    text.push_str(&ionex_record("1", "END OF TEC MAP"));
    text.push_str(&ionex_record("1", "START OF RMS MAP"));
    text.push_str(&ionex_record("2020 1 1 0 0 0", "EPOCH OF CURRENT MAP"));
    text.push_str(&ionex_record(
        "1.0 0.0 1.0 1.0 450.0",
        "LAT/LON1/LON2/DLON/H",
    ));
    text.push_str("1 2\n");
    text.push_str(&ionex_record(
        "0.0 0.0 1.0 1.0 450.0",
        "LAT/LON1/LON2/DLON/H",
    ));
    text.push_str("3 4\n");
    text.push_str(&ionex_record("1", "END OF RMS MAP"));

    let original = Ionex::parse_str(&text).expect("parse IONEX with RMS map");
    assert_eq!(original.rms_maps().len(), 1, "one RMS map present");

    let encoded = original.to_ionex_string();
    let reparsed = Ionex::parse_str(&encoded).expect("serialized IONEX reparses");
    assert_eq!(reparsed, original, "round-trip preserves TEC and RMS grids");
    assert_eq!(
        reparsed.to_ionex_string(),
        encoded,
        "serializer is deterministic"
    );
}

#[test]
fn ionex_aux_data_block_is_skipped_with_a_diagnostic() {
    let fx = fixtures_dir();
    let path = fx.join("ionex/synthetic_2map_7x7.20i");
    let full =
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));

    // Inject an unsupported auxiliary-data block after the header. It must be
    // skipped (counted, never silently dropped) and must not perturb the grid.
    let mut aux = String::new();
    aux.push_str(&ionex_record("", "START OF AUX DATA"));
    aux.push_str(&ionex_record("G01  -1.234  0.567", "PRN / BIAS / RMS"));
    aux.push_str(&ionex_record("", "END OF AUX DATA"));
    let header_pos = full.find("END OF HEADER").expect("END OF HEADER present");
    let line_end = full[header_pos..]
        .find('\n')
        .map(|offset| header_pos + offset + 1)
        .expect("newline after END OF HEADER");
    let mut injected = String::new();
    injected.push_str(&full[..line_end]);
    injected.push_str(&aux);
    injected.push_str(&full[line_end..]);
    assert_ne!(injected, full, "aux block was injected");

    let clean = Ionex::parse_str(&full).expect("parse clean product");
    let with_aux = Ionex::parse_str(&injected).expect("parse product with aux block");
    assert_eq!(with_aux.skipped_records(), 1, "aux block recorded one skip");
    assert_eq!(
        with_aux.tec_maps(),
        clean.tec_maps(),
        "aux block does not perturb the grid"
    );
    assert_eq!(with_aux.map_epochs_s(), clean.map_epochs_s());
}

#[test]
fn ionex_truncated_second_tec_map_data_row_errors() {
    let mut text = String::new();
    text.push_str(&ionex_record("1.0", "IONEX VERSION / TYPE"));
    text.push_str(&ionex_record("2", "# OF MAPS IN FILE"));
    text.push_str(&ionex_record("1.0 0.0 -1.0", "LAT1 / LAT2 / DLAT"));
    text.push_str(&ionex_record("0.0 2.0 1.0", "LON1 / LON2 / DLON"));
    text.push_str(&ionex_record("450.0 450.0 0.0", "HGT1 / HGT2 / DHGT"));
    text.push_str(&ionex_record("6371.0", "BASE RADIUS"));
    text.push_str(&ionex_record("0", "EXPONENT"));
    text.push_str(&ionex_record("", "END OF HEADER"));

    text.push_str(&ionex_record("1", "START OF TEC MAP"));
    text.push_str(&ionex_record("2020 1 1 0 0 0", "EPOCH OF CURRENT MAP"));
    text.push_str(&ionex_record(
        "1.0 0.0 2.0 1.0 450.0",
        "LAT/LON1/LON2/DLON/H",
    ));
    text.push_str("1 2 3\n");
    text.push_str(&ionex_record(
        "0.0 0.0 2.0 1.0 450.0",
        "LAT/LON1/LON2/DLON/H",
    ));
    text.push_str("4 5 6\n");
    text.push_str(&ionex_record("1", "END OF TEC MAP"));

    text.push_str(&ionex_record("2", "START OF TEC MAP"));
    text.push_str(&ionex_record("2020 1 1 2 0 0", "EPOCH OF CURRENT MAP"));
    text.push_str(&ionex_record(
        "1.0 0.0 2.0 1.0 450.0",
        "LAT/LON1/LON2/DLON/H",
    ));
    text.push_str("7 8\n");

    let err = Ionex::parse_str(&text).expect_err("truncated second TEC map must error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("latitude band") && msg.contains("expected 3"),
        "expected short-row parse error, got: {msg}"
    );
}

#[test]
fn ionex_unsorted_map_epochs_are_rejected_at_parse() {
    let mut text = String::new();
    text.push_str(&ionex_record("1.0", "IONEX VERSION / TYPE"));
    text.push_str(&ionex_record("2", "# OF MAPS IN FILE"));
    text.push_str(&ionex_record("1.0 0.0 -1.0", "LAT1 / LAT2 / DLAT"));
    text.push_str(&ionex_record("0.0 1.0 1.0", "LON1 / LON2 / DLON"));
    text.push_str(&ionex_record("450.0 450.0 0.0", "HGT1 / HGT2 / DHGT"));
    text.push_str(&ionex_record("6371.0", "BASE RADIUS"));
    text.push_str(&ionex_record("0", "EXPONENT"));
    text.push_str(&ionex_record("", "END OF HEADER"));

    for (map_index, epoch) in [("1", "2020 1 1 2 0 0"), ("2", "2020 1 1 0 0 0")] {
        text.push_str(&ionex_record(map_index, "START OF TEC MAP"));
        text.push_str(&ionex_record(epoch, "EPOCH OF CURRENT MAP"));
        text.push_str(&ionex_record(
            "1.0 0.0 1.0 1.0 450.0",
            "LAT/LON1/LON2/DLON/H",
        ));
        text.push_str("1 2\n");
        text.push_str(&ionex_record(
            "0.0 0.0 1.0 1.0 450.0",
            "LAT/LON1/LON2/DLON/H",
        ));
        text.push_str("3 4\n");
        text.push_str(&ionex_record(map_index, "END OF TEC MAP"));
    }

    let err = Ionex::parse_str(&text).expect_err("unsorted map epochs must error");
    let msg = format!("{err:?}");
    assert!(
        msg.contains("strictly increasing"),
        "expected epoch-order parse error, got: {msg}"
    );
}

fn ionex_record(data: &str, label: &str) -> String {
    format!("{data:<60}{label}\n")
}

fn replace_ionex_record_data(text: &str, label: &str, data: &str) -> String {
    let mut replaced = false;
    let mut out = String::new();
    for line in text.lines() {
        if line.contains(label) {
            out.push_str(&format!("{data:<60}{label}"));
            replaced = true;
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    assert!(replaced, "missing IONEX record {label}");
    out
}

/// A record line carrying a multibyte character straddling column 60 must not
/// panic the fixed-width reader. The label/data window helpers raw-sliced `&str`
/// by byte offset, so a non-ASCII byte before the offset (byte 60 here lands in
/// the middle of `\u{20ac}`) used to panic on `line[60..]`/`&line[..60]`. The
/// char-boundary-safe parse helpers now floor to a boundary instead, so the
/// reader returns a typed [`Error::Parse`] (the unrecognized label is ignored and
/// the required header records never appear) rather than aborting.
#[test]
fn ionex_multibyte_line_returns_parse_error_not_panic() {
    let bad = format!("{}\u{20ac}xxxxxxx", "0".repeat(59));
    assert!(
        !bad.is_char_boundary(60),
        "test input must straddle byte 60"
    );

    let err = super::Ionex::parse_str(&bad).expect_err("multibyte line must not parse");
    assert!(
        matches!(err, crate::error::Error::Parse(_)),
        "expected a typed Parse error, got: {err:?}"
    );
}

fn fixture_path_named(parts: &[&str]) -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests");
    path.push("fixtures");
    for part in parts {
        path.push(part);
    }
    path
}

fn bits_value(v: &Value) -> f64 {
    crate::test_parity::f64_from_hex(v.as_str().expect("hex-bit string")).expect("valid f64 bits")
}

fn bits_vec(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("hex-bit array")
        .iter()
        .map(bits_value)
        .collect()
}

// Fixture provenance: `tests/fixtures/tec_grid/tec_grid.json` is generated by the
// committed script `crates/sidereon-core/fixtures-generators/generate_tec_grid.py`.
// The generator downloads the public IONEX product
// `ftp://gssc.esa.int/gnss/products/ionex/2024/001/IGS0OPSFIN_20240010000_01D_02H_GIM.INX.gz`,
// parses the TEC maps, builds a regular epoch/lat/lon grid, and records probe
// values from the reference oracle SciPy 1.11.3
// `scipy.interpolate.RegularGridInterpolator`. Generated with Python 3.11.15,
// NumPy 1.26.0, SciPy 1.11.3 on macOS-26.5.1-arm64. All floating-point values are
// serialized as f64 hex-bit strings and must be compared with `f64::to_bits`,
// never tolerances.
#[test]
fn regular_tec_grid_matches_scipy_regular_grid_bits() {
    let raw = std::fs::read_to_string(fixture_path_named(&["tec_grid", "tec_grid.json"]))
        .expect("read tec_grid fixture");
    let doc: Value = serde_json::from_str(&raw).expect("parse tec_grid fixture");
    assert_eq!(doc["schema"], "gnss-tec-grid-v1");

    let grid = super::tec_grid::TecGrid::new(
        bits_vec(&doc["epochs_bits"]),
        bits_vec(&doc["lats_bits"]),
        bits_vec(&doc["lons_bits"]),
        bits_vec(&doc["values_bits"]),
    )
    .expect("regular TEC grid");

    let mut checked = 0usize;
    for probe in doc["regular_grid_probes"].as_array().expect("probes") {
        let name = probe["name"].as_str().expect("name");
        let point = bits_vec(&probe["point_bits"]);
        let got = grid
            .interpolate_vtec(point[0], point[1], point[2])
            .expect("interpolate vtec");
        let want = bits_value(&probe["value_bits"]);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{name} TEC grid bits: got=0x{:016x} want=0x{:016x}",
            got.to_bits(),
            want.to_bits()
        );
        checked += 1;
    }
    assert!(checked > 0, "empty TEC grid probes");
}

#[test]
fn regular_tec_grid_shell_geometry_is_configurable() {
    use super::tec_grid::{
        iono_delay_xyz, pierce_point_with_shell_radius, tec_xyz, TecGrid, TecGridEpoch,
        TecGridEvalOptions, TecGridShellGeometry, EARTH_RADIUS_M, IONOSPHERE_HEIGHT_M,
    };
    fn spherical_lonlatalt(xyz: &[f64; 3]) -> [f64; 3] {
        let p = (xyz[0] * xyz[0] + xyz[1] * xyz[1]).sqrt();
        let r = (p * p + xyz[2] * xyz[2]).sqrt();
        [
            xyz[1].atan2(xyz[0]) * crate::constants::RAD_TO_DEG,
            xyz[2].atan2(p) * crate::constants::RAD_TO_DEG,
            r - super::tec_grid::EARTH_RADIUS_M,
        ]
    }

    let grid = TecGrid::new(
        vec![0.0, 1_000_000_000.0],
        vec![-90.0, 90.0],
        vec![-180.0, 180.0],
        vec![10.0; 8],
    )
    .expect("constant TEC grid");
    let epoch = TecGridEpoch::new(0, 1);
    let options = TecGridEvalOptions::l1(epoch);
    let receiver = [EARTH_RADIUS_M, 0.0, 0.0];
    let satellite = [26_000_000.0, 9_000_000.0, 7_000_000.0];

    let (vtec_default, stec_default) =
        tec_xyz(&grid, options, &satellite, &receiver, spherical_lonlatalt)
            .expect("default geometry TEC");
    let delay_default = iono_delay_xyz(&grid, options, &satellite, &receiver, spherical_lonlatalt)
        .expect("default geometry delay");

    let default_shell_radius_m = EARTH_RADIUS_M + IONOSPHERE_HEIGHT_M;
    let (_, _, mut elevation_rad) = pierce_point_with_shell_radius(
        &satellite,
        &receiver,
        default_shell_radius_m,
        spherical_lonlatalt,
    );
    if elevation_rad < options.min_elevation_rad {
        elevation_rad = options.min_elevation_rad;
    }
    let default_arg = EARTH_RADIUS_M * elevation_rad.cos() / default_shell_radius_m;
    let expected_default_stec = vtec_default / (1.0 - default_arg * default_arg).sqrt();
    assert_eq!(
        stec_default.to_bits(),
        expected_default_stec.to_bits(),
        "default shell geometry must match the historical obliquity mapping"
    );

    let custom_shell = TecGridShellGeometry::new(EARTH_RADIUS_M, IONOSPHERE_HEIGHT_M + 250_000.0);
    let custom_options = options.with_shell_geometry(custom_shell);
    let (vtec_custom, stec_custom) = tec_xyz(
        &grid,
        custom_options,
        &satellite,
        &receiver,
        spherical_lonlatalt,
    )
    .expect("custom geometry TEC");
    let delay_custom = iono_delay_xyz(
        &grid,
        custom_options,
        &satellite,
        &receiver,
        spherical_lonlatalt,
    )
    .expect("custom geometry delay");

    assert!(
        (vtec_custom - vtec_default).abs() < 1.0e-12,
        "constant grid VTEC should not depend on shell geometry"
    );
    let custom_arg =
        custom_shell.earth_radius_m * elevation_rad.cos() / custom_shell.shell_radius_m();
    let expected_custom_stec = vtec_custom / (1.0 - custom_arg * custom_arg).sqrt();
    assert_eq!(
        stec_custom.to_bits(),
        expected_custom_stec.to_bits(),
        "custom shell geometry must drive the obliquity mapping"
    );
    assert_ne!(
        stec_custom.to_bits(),
        stec_default.to_bits(),
        "non-default shell height should change slant TEC"
    );
    assert_ne!(
        delay_custom.to_bits(),
        delay_default.to_bits(),
        "non-default shell height should change the mapped delay"
    );
}
