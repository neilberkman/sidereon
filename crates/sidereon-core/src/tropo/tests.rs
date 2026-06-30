//! 0-ULP parity tests for the Saastamoinen + Niell tropospheric delay recipe.
//!
//! These assert the Rust port reproduces the canonical reference recipe
//! `parity/generator/troposphere.py` bit-for-bit, using the committed golden
//! fixture `parity/fixtures/troposphere_golden.json`. Values are serialised as
//! hex-float (Python `float.hex()`) so there is no decimal-parse ambiguity, and
//! parity is measured as ULP distance via the integer reinterpretation of the
//! IEEE-754 bit pattern, per the `skyfield_parity_test.exs` discipline.
//!
//! Every intermediate quantity (the water-vapour partial pressure, both zenith
//! delays, both mapping factors, the height correction, the seasonal phase) and
//! the final slant delay is checked per component, so a divergence is localised
//! to a single algorithm step rather than only seen at the output. The cases
//! span the full branch matrix: elevations from zenith down past the Niell low
//! range and the at/below-horizon gate; equator / mid / high / polar / southern
//! latitudes exercising the latitude-interpolation nodes, clamps, and the
//! southern seasonal-phase offset; day-of-year at the cosine-phase extremes and
//! a fractional mid-year value; sea level, high altitude, below sea level, the
//! Met-path height-gate edges; and humidity, temperature, and pressure extremes.
//! A separate standard-atmosphere family covers the pressure/temperature
//! synthesis and its height clamp.

use std::path::PathBuf;

use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use serde_json::Value;

use super::saastamoinen::{slant_components, standard_atmosphere};

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

fn fixture_path() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .join("tests/fixtures/troposphere_golden.json")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!(
                "cannot locate tests/fixtures/troposphere_golden.json from {}: {e}",
                crate_dir.display()
            )
        })
}

fn hexf(v: &Value, key: &str) -> f64 {
    parse_hex_float(
        v[key]
            .as_str()
            .unwrap_or_else(|| panic!("missing/non-string {key}")),
    )
}

#[test]
fn troposphere_zero_ulp_full_branch_matrix() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read troposphere_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse troposphere_golden.json");

    // Self-check the hex-float parser/serialiser round-trips a known bit pattern,
    // so a parser bug can never masquerade as parity.
    let probe = "0x1.921fb54442d18p+1"; // math.pi
    assert_eq!(
        float_hex(parse_hex_float(probe)),
        probe,
        "hex-float parser/serialiser round-trip is broken"
    );

    let mut failures: Vec<String> = Vec::new();
    let mut checks = 0usize;

    // --- Slant cases (zenith + mapping + composed delay). ---
    let slant_cases = doc["slant_cases"].as_array().expect("slant_cases array");
    assert!(
        slant_cases.len() >= 22,
        "expected the full slant branch matrix (>= 22 cases), found {}",
        slant_cases.len()
    );

    let slant_components_keys: &[&str] = &[
        "e", "zhd_m", "zwd_m", "mh", "mw", "dm", "cosy", "y", "slant_m",
    ];

    for case in slant_cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let exp = &case["expect"];

        // Radian angles and the fractional day-of-year are taken straight from
        // the fixture, exactly as the reference recipe consumes them.
        let got = slant_components(
            hexf(inp, "el_rad"),
            crate::frame::Wgs84Geodetic::new(
                hexf(inp, "lat_rad"),
                hexf(inp, "lon_rad"),
                hexf(inp, "height_m"),
            )
            .expect("valid WGS84 geodetic position"),
            hexf(inp, "pressure_hpa"),
            hexf(inp, "temperature_k"),
            hexf(inp, "relative_humidity"),
            hexf(inp, "doy"),
        );

        let actual = |c: &str| -> f64 {
            match c {
                "e" => got.e,
                "zhd_m" => got.zhd_m,
                "zwd_m" => got.zwd_m,
                "mh" => got.mh,
                "mw" => got.mw,
                "dm" => got.dm,
                "cosy" => got.cosy,
                "y" => got.y,
                "slant_m" => got.slant_m,
                other => panic!("unknown component {other}"),
            }
        };

        for &c in slant_components_keys {
            let want = parse_hex_float(
                exp[c]
                    .as_str()
                    .unwrap_or_else(|| panic!("case {name}: missing expected component {c}")),
            );
            let a = actual(c);
            let ulp = ulp_distance(a, want);
            checks += 1;
            if ulp != 0 {
                failures.push(format!(
                    "slant {name}.{c}: {ulp} ULP (rust={} ref={})",
                    float_hex(a),
                    exp[c].as_str().unwrap()
                ));
            }
        }
    }

    // --- Standard-atmosphere helper family (P/T synthesis + height clamp). ---
    let std_cases = doc["standard_atmosphere_cases"]
        .as_array()
        .expect("standard_atmosphere_cases array");
    assert!(
        std_cases.len() >= 4,
        "expected the standard-atmosphere family (>= 4 cases), found {}",
        std_cases.len()
    );

    let std_keys: &[&str] = &["pressure_hpa", "temperature_k", "relative_humidity"];

    for case in std_cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let exp = &case["expect"];

        let got = standard_atmosphere(hexf(inp, "height_m"), hexf(inp, "relative_humidity"));

        let actual = |c: &str| -> f64 {
            match c {
                "pressure_hpa" => got.pressure_hpa,
                "temperature_k" => got.temperature_k,
                "relative_humidity" => got.relative_humidity,
                other => panic!("unknown component {other}"),
            }
        };

        for &c in std_keys {
            let want = parse_hex_float(
                exp[c]
                    .as_str()
                    .unwrap_or_else(|| panic!("case {name}: missing expected component {c}")),
            );
            let a = actual(c);
            let ulp = ulp_distance(a, want);
            checks += 1;
            if ulp != 0 {
                failures.push(format!(
                    "std {name}.{c}: {ulp} ULP (rust={} ref={})",
                    float_hex(a),
                    exp[c].as_str().unwrap()
                ));
            }
        }
    }

    assert!(checks > 0, "no components were checked - fixture empty?");
    assert!(
        failures.is_empty(),
        "Troposphere Rust port diverged from the reference recipe on {} of {checks} components:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Coverage for the PUBLIC `tropo_slant(epoch, ...)` wrapper, which the kernel
/// test above does not exercise (it feeds `doy` straight into
/// `slant_components`). The wrapper derives the day-of-year from `epoch`. Unlike
/// the ionosphere path there is no angle conversion (latitude stays in radians),
/// so when the epoch reconstructs the day-of-year exactly the public path is
/// bit-for-bit identical to the kernel/golden. This drives the `zenith_midlat`
/// case (day-of-year 28 -> 2000-01-28 00:00, reconstructed exactly) through the
/// public API and asserts 0 ULP against the golden, closing the coverage gap.
#[test]
fn tropo_public_wrapper_is_0_ulp_when_epoch_reconstructs_doy() {
    use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};

    let raw = std::fs::read_to_string(fixture_path()).expect("read troposphere_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse troposphere_golden.json");
    let cases = doc["slant_cases"].as_array().expect("slant_cases array");

    let case = cases
        .iter()
        .find(|c| c["name"].as_str() == Some("zenith_midlat"))
        .expect("zenith_midlat case present");
    let inp = &case["inputs"];

    // This reconstruction is only exact for an integer day-of-year at midnight;
    // assert that precondition so a fixture change can't silently invalidate it.
    assert_eq!(
        hexf(inp, "doy"),
        28.0,
        "test assumes zenith_midlat is day-of-year 28"
    );

    // 2000-01-28 00:00:00: JDN(noon) = 2451572, midnight boundary = 2451571.5.
    // fractional_day_of_year() maps this to exactly 28.0.
    let epoch = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(2_451_571.5, 0.0).expect("valid split Julian date"),
    );

    let receiver = crate::frame::Wgs84Geodetic::new(
        hexf(inp, "lat_rad"),
        hexf(inp, "lon_rad"),
        hexf(inp, "height_m"),
    )
    .expect("valid WGS84 geodetic position");
    let met = super::Met::new(
        hexf(inp, "pressure_hpa"),
        hexf(inp, "temperature_k"),
        hexf(inp, "relative_humidity"),
    )
    .expect("valid troposphere met");

    let got = super::tropo_slant(hexf(inp, "el_rad"), receiver, met, epoch)
        .expect("valid public troposphere inputs");
    let want = parse_hex_float(case["expect"]["slant_m"].as_str().unwrap());

    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "public tropo_slant not 0-ULP: got={} want={}",
        float_hex(got),
        case["expect"]["slant_m"].as_str().unwrap()
    );
}

fn valid_tropo_receiver() -> crate::frame::Wgs84Geodetic {
    crate::frame::Wgs84Geodetic::new(45.0_f64.to_radians(), 8.0_f64.to_radians(), 400.0)
        .expect("valid WGS84 geodetic position")
}

fn valid_tropo_epoch() -> Instant {
    Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(2_451_571.5, 0.0).expect("valid split Julian date"),
    )
}

fn valid_tropo_met() -> super::Met {
    super::Met::new(1013.25, 288.15, 0.5).expect("valid troposphere met")
}

fn assert_invalid_input<T: core::fmt::Debug>(result: crate::Result<T>) {
    let err = result.expect_err("invalid troposphere input must be rejected");
    assert!(
        matches!(err, crate::error::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn tropo_public_helpers_reject_invalid_domains() {
    let receiver = valid_tropo_receiver();
    let met = valid_tropo_met();
    let epoch = valid_tropo_epoch();
    let elevation = 30.0_f64.to_radians();

    assert_invalid_input(super::Met::new(f64::NAN, 288.15, 0.5));
    assert_invalid_input(super::Met::new(1013.25, 0.0, 0.5));
    assert_invalid_input(super::Met::new(1013.25, 288.15, 1.5));
    assert_invalid_input(super::Met::standard(f64::INFINITY, 0.5));
    assert_invalid_input(super::Met::standard(100.0, f64::NAN));
    assert_invalid_input(super::Met::standard(100_000.0, 0.5));

    assert_invalid_input(super::tropo_slant(f64::NAN, receiver, met, epoch));
    assert_invalid_input(super::tropo_slant(-1.0e-6, receiver, met, epoch));
    assert_invalid_input(super::tropo_mapping(
        super::MappingModel::Niell,
        0.0,
        receiver,
        epoch,
    ));

    let bad_receiver = crate::frame::Wgs84Geodetic {
        lat_rad: f64::NAN,
        lon_rad: receiver.lon_rad,
        height_m: receiver.height_m,
    };
    assert_invalid_input(super::tropo_slant(elevation, bad_receiver, met, epoch));

    let west_antimeridian = crate::frame::Wgs84Geodetic {
        lat_rad: receiver.lat_rad,
        lon_rad: -core::f64::consts::PI,
        height_m: receiver.height_m,
    };
    super::tropo_slant(elevation, west_antimeridian, met, epoch)
        .expect("west antimeridian receiver is valid");

    let bad_lon_receiver = crate::frame::Wgs84Geodetic {
        lat_rad: receiver.lat_rad,
        lon_rad: -core::f64::consts::PI - 1.0e-12,
        height_m: receiver.height_m,
    };
    assert_invalid_input(super::tropo_slant(elevation, bad_lon_receiver, met, epoch));

    let high_receiver = crate::frame::Wgs84Geodetic {
        lat_rad: receiver.lat_rad,
        lon_rad: receiver.lon_rad,
        height_m: 20_000.0,
    };
    assert_invalid_input(super::tropo_mapping(
        super::MappingModel::Niell,
        elevation,
        high_receiver,
        epoch,
    ));

    let bad_met = super::Met::new_unchecked(1013.25, f64::INFINITY, 0.5);
    assert_invalid_input(super::tropo_zenith(
        super::TropoModel::Saastamoinen,
        receiver,
        bad_met,
    ));

    let bad_epoch = Instant {
        scale: TimeScale::Gpst,
        repr: crate::astro::time::model::InstantRepr::JulianDate(JulianDateSplit {
            jd_whole: f64::NAN,
            fraction: 0.0,
        }),
    };
    assert_invalid_input(super::tropo_mapping(
        super::MappingModel::Niell,
        elevation,
        receiver,
        bad_epoch,
    ));

    let huge_epoch = Instant {
        scale: TimeScale::Gpst,
        repr: crate::astro::time::model::InstantRepr::JulianDate(JulianDateSplit {
            jd_whole: 1.0e20,
            fraction: 0.0,
        }),
    };
    assert_invalid_input(super::tropo_mapping(
        super::MappingModel::Niell,
        elevation,
        receiver,
        huge_epoch,
    ));

    let ancient_epoch = Instant {
        scale: TimeScale::Gpst,
        repr: crate::astro::time::model::InstantRepr::JulianDate(JulianDateSplit {
            jd_whole: 213.0,
            fraction: 0.0,
        }),
    };
    assert_invalid_input(super::tropo_mapping(
        super::MappingModel::Niell,
        elevation,
        receiver,
        ancient_epoch,
    ));
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

fn bits_array3(v: &Value) -> [f64; 3] {
    let values: Vec<f64> = v
        .as_array()
        .expect("hex-bit array")
        .iter()
        .map(bits_value)
        .collect();
    [values[0], values[1], values[2]]
}

// Fixture provenance: `tests/fixtures/tropo_zwd/tropo_zwd.json` is generated by the
// committed script `crates/sidereon-core/fixtures-generators/generate_tropo_zwd.py`.
// Reference oracle is the committed public-formula generator itself: standard-pressure
// hydrostatic delay, exponential-scale-height ZWD wet delay, and Niell-style mapping
// coefficients encoded in the script. Inputs are deterministic receiver/satellite
// geometries from public WGS84 CRS transforms and public formulas. Generated with
// Python 3.11.15, PyProj 3.6.1, PROJ 9.3.0 on macOS-26.5.1-arm64. All floating-point
// values are serialized as f64 hex-bit strings and must be compared with
// `f64::to_bits`, never tolerances.
#[test]
fn zwd_xyz_variant_matches_generated_fixture_bits() {
    let raw = std::fs::read_to_string(fixture_path_named(&["tropo_zwd", "tropo_zwd.json"]))
        .expect("read tropo_zwd fixture");
    let doc: Value = serde_json::from_str(&raw).expect("parse tropo_zwd fixture");
    assert_eq!(doc["schema"], "gnss-tropo-zwd-v1");
    assert_eq!(doc["pyproj_version"], "3.6.1");

    let mut checked = 0usize;
    for case in doc["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let day = case["day_of_year"].as_u64().expect("day") as u16;
        let sat = bits_array3(&case["sat_xyz_bits"]);
        let rx = bits_array3(&case["receiver_xyz_bits"]);
        let receiver_lonlatalt = bits_array3(&case["receiver_lonlatalt_bits"]);
        let options = super::ZwdSlantOptions::new(
            super::ZwdEpoch::new(0, day).expect("valid ZWD epoch"),
            super::ZwdProfile::default(),
        )
        .expect("valid ZWD options");
        let got = super::tropo_zwd_delay_xyz(options, &sat, &rx, |_| receiver_lonlatalt)
            .expect("valid ZWD XYZ inputs");
        let want = bits_value(&case["delay_bits"]);
        assert_eq!(
            got.to_bits(),
            want.to_bits(),
            "{name} ZWD delay bits: got=0x{:016x} want=0x{:016x}",
            got.to_bits(),
            want.to_bits()
        );
        checked += 1;
    }
    assert!(checked > 0, "empty tropo_zwd fixture");
}

#[test]
fn zwd_public_helpers_reject_invalid_inputs() {
    assert_invalid_input(super::ZwdEpoch::new(0, 0));
    assert_invalid_input(super::ZwdEpoch::new(0, 367));

    let profile = super::ZwdProfile::default();
    assert_invalid_input(super::ZwdSlantOptions::new(
        super::ZwdEpoch {
            unix_nanos: 0,
            day_of_year: 120,
        },
        super::ZwdProfile {
            wet_scale_height_m: f64::NAN,
            ..profile
        },
    ));
    assert_invalid_input(super::zwd_zenith_wet_delay(
        super::ZwdProfile {
            altitude_clamp: super::AltitudeClamp {
                min_m: 10.0,
                max_m: -10.0,
            },
            ..profile
        },
        100.0,
    ));
    assert_invalid_input(super::zwd_zenith_wet_delay(profile, f64::INFINITY));

    assert_invalid_input(super::zwd::niell_mapping_function(
        30.0_f64.to_radians(),
        91.0,
        120,
        100.0,
    ));

    let options = super::ZwdSlantOptions::new(
        super::ZwdEpoch::new(0, 120).expect("valid ZWD epoch"),
        profile,
    )
    .expect("valid ZWD options");
    let sat = [20_000_000.0, 1_000_000.0, 2_000_000.0];
    let rx = [6_300_000.0, 0.0, 0.0];
    assert_invalid_input(super::tropo_zwd_delay_xyz(
        options,
        &[f64::NAN, sat[1], sat[2]],
        &rx,
        |_| [0.0, 0.0, 0.0],
    ));
    assert_invalid_input(super::tropo_zwd_delay_xyz(options, &rx, &rx, |_| {
        [0.0, 0.0, 0.0]
    }));
    assert_invalid_input(super::tropo_zwd_delay_xyz(options, &sat, &rx, |_| {
        [0.0, f64::NAN, 0.0]
    }));
}

/// VMF1 (Vienna Mapping Function 1) site-wise mapping factors match the TU Wien
/// reference implementation `vmf1.f`.
///
/// Oracle: `https://vmf.geo.tuwien.ac.at/codes/vmf1.f` (Böhm 2006, site-wise, no
/// height correction), compiled with gfortran and evaluated at the ZIM2
/// site-wise `a` coefficients for 2026-05-13 00 UT (MJD 61173.00,
/// ah = 0.00121738, aw = 0.00058796; VMF1_OP GNSS daily product) and ZIM2's
/// ellipsoidal latitude 46.8771 deg, over elevations 5..90 deg. The reference
/// values below are that program's output. The only difference between the port
/// and the reference is the value of pi: the reference uses the truncated
/// constant 3.14159265359 while the port uses the correctly rounded
/// `std::f64::consts::PI`; the resulting agreement is reported by the assert and
/// is ~3e-9 in the mapping factor at 5 deg (under a nanometre of slant delay),
/// shrinking to exactly zero at the zenith.
#[test]
fn vmf1_site_wise_matches_tuwien_reference() {
    // (elevation_deg, reference mfh, reference mfw) from vmf1.f.
    const REF: [(f64, f64, f64); 18] = [
        (5.0, 10.155_463_643_141_664, 10.743_138_939_798_179),
        (10.0, 5.556_234_634_556_705, 5.655_970_679_266_376),
        (15.0, 3.801_631_023_430_789_7, 3.832_944_785_704_285),
        (20.0, 2.897_815_243_816_23, 2.911_049_475_573_457_3),
        (25.0, 2.353_258_031_813_431_3, 2.359_879_405_799_712_3),
        (30.0, 1.992_823_009_969_912_5, 1.996_503_893_078_469_5),
        (35.0, 1.739_178_595_491_464_5, 1.741_371_095_952_693_3),
        (40.0, 1.553_065_702_048_180_4, 1.554_432_547_444_540_2),
        (45.0, 1.412_509_611_408_922_4, 1.413_386_429_553_977_5),
        (50.0, 1.304_298_645_257_952_1, 1.304_869_418_596_862_1),
        (55.0, 1.220_052_060_683_842_3, 1.220_424_182_951_631_1),
        (60.0, 1.154_235_627_795_293, 1.154_475_134_679_011_2),
        (65.0, 1.103_087_997_669_529_1, 1.103_237_385_731_385_7),
        (70.0, 1.064_007_362_157_323_8, 1.064_095_182_638_310_5),
        (75.0, 1.035_186_311_241_028, 1.035_232_629_962_372_2),
        (80.0, 1.015_388_434_814_004_5, 1.015_408_112_739_035_5),
        (85.0, 1.003_810_545_431_223_2, 1.003_815_335_137_829_8),
        (90.0, 1.0, 1.0),
    ];

    // MJD 61173.00 -> JD 2461173.5.
    let epoch = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(2_461_173.5, 0.0).expect("valid VMF epoch split"),
    );
    let receiver =
        crate::frame::Wgs84Geodetic::new(46.8771_f64.to_radians(), 7.4650_f64.to_radians(), 956.40)
            .expect("valid ZIM2 receiver");
    let model = super::MappingModel::Vmf1 {
        ah: 0.00121738,
        aw: 0.00058796,
    };

    let mut max_dry = 0.0_f64;
    let mut max_wet = 0.0_f64;
    for (el_deg, ref_mfh, ref_mfw) in REF {
        let m = super::tropo_mapping(model, el_deg.to_radians(), receiver, epoch)
            .expect("VMF1 mapping resolves");
        max_dry = max_dry.max((m.dry - ref_mfh).abs());
        max_wet = max_wet.max((m.wet - ref_mfw).abs());
    }

    eprintln!("VMF1 vs vmf1.f: max |dry| = {max_dry:e}, max |wet| = {max_wet:e}");
    // Agreement bound: the residual is the reference's pi truncation only
    // (achieved ~1.3e-9 hydrostatic, ~2.8e-9 wet at 5 deg).
    assert!(
        max_dry < 5.0e-9,
        "VMF1 hydrostatic mapping vs vmf1.f off by {max_dry:e}"
    );
    assert!(
        max_wet < 5.0e-9,
        "VMF1 wet mapping vs vmf1.f off by {max_wet:e}"
    );
}

/// The VMF1 continued fraction is exactly 1 at the zenith and decreases
/// monotonically toward the horizon, independent of the data coefficients.
#[test]
fn vmf1_zenith_is_unity_and_monotone() {
    let epoch = Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(2_461_173.5, 0.0).expect("valid VMF epoch split"),
    );
    let receiver =
        crate::frame::Wgs84Geodetic::new(46.8771_f64.to_radians(), 7.4650_f64.to_radians(), 956.40)
            .expect("valid receiver");
    let model = super::MappingModel::Vmf1 {
        ah: 0.00121738,
        aw: 0.00058796,
    };

    let zenith = super::tropo_mapping(model, core::f64::consts::FRAC_PI_2, receiver, epoch)
        .expect("zenith mapping");
    assert_eq!(zenith.dry, 1.0);
    assert_eq!(zenith.wet, 1.0);

    let mut prev_dry = f64::INFINITY;
    let mut prev_wet = f64::INFINITY;
    for el_deg in (5..=90).step_by(5) {
        let m = super::tropo_mapping(model, (el_deg as f64).to_radians(), receiver, epoch)
            .expect("mapping resolves");
        assert!(m.dry >= 1.0 && m.wet >= 1.0);
        assert!(
            m.dry <= prev_dry && m.wet <= prev_wet,
            "mapping not monotone at {el_deg} deg"
        );
        prev_dry = m.dry;
        prev_wet = m.wet;
    }
}

/// VMF1 with invalid (`<= 0` / non-finite) `a` coefficients is rejected by the
/// checked public entry.
#[test]
fn vmf1_rejects_invalid_a_coefficients() {
    let receiver = valid_tropo_receiver();
    let epoch = valid_tropo_epoch();
    assert_invalid_input(super::tropo_mapping(
        super::MappingModel::Vmf1 {
            ah: 0.0,
            aw: 0.0006,
        },
        45.0_f64.to_radians(),
        receiver,
        epoch,
    ));
    assert_invalid_input(super::tropo_mapping(
        super::MappingModel::Vmf1 {
            ah: 0.0012,
            aw: f64::NAN,
        },
        45.0_f64.to_radians(),
        receiver,
        epoch,
    ));
}
