//! 0-ULP parity tests for the IONEX slant ionospheric delay pipeline.
//!
//! These assert the Rust port reproduces, bit for bit, the Python reference
//! values in the committed golden fixture `tests/fixtures/ionex_golden.json`,
//! using the synthetic IONEX product the fixture was generated from. Values are serialised as hex-float (Python
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
use crate::constants::SECONDS_PER_DAY;
use serde_json::Value;

use super::grid::Ionex;
use super::slant::{slant_delay_components, PierceLineOfSight, SlantComponents, VtecGridView};
use super::{
    galileo_nequick_g_native, ionex_slant_delay_results, ionex_slant_delay_with_policy,
    ionex_slant_delays, ionosphere_delay, GalileoNequickCoeffs, GalileoNequickEval,
    IonexAssumedMapping, IonexCoverageError, IonexCoveragePolicy, IonexHeader,
    IonexMappingDeclaration, IonexMappingFunction, IonexMappingPolicy, IonexMissingNodePolicy,
    IonexMissingNodes, IonexNodeGap, IonexSlantDelayStatus, IonexSlantPolicy, IonexSlantRefusal,
    IonexSlantRequest, IonexWarning, IonoModel, TecGridSamples, TecSample, TecSamplesError,
};

const REAL_IONEX_EXCERPT: &[u8] =
    include_bytes!("../../tests/fixtures/ionex/esa_2024176_first_map_2row.inx");

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
    let sample_built =
        Ionex::from_samples(ionex.tec_grid_samples()).expect("sample-built IONEX product");
    assert_eq!(
        sample_built, ionex,
        "sample-built IONEX must preserve the parsed IR"
    );

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
                    ionex.tec_maps()[m][i][j].expect("synthetic node holds a value"),
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

    let mut checks = 0usize;

    for (source, product) in [("parsed", &ionex), ("sample-built", &sample_built)] {
        let lat_arr = product.lat_nodes_deg();
        let lon_arr = product.lon_nodes_deg();
        let dlat = product.dlat_deg();
        let dlon = product.dlon_deg();
        let re = product.base_radius_km();
        let h = product.shell_height_km();
        let epochs = product.map_epochs();
        let maps = product.tec_maps();

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
            )
            .expect("synthetic grid holds every node");

            // The temporal-bracket index is a discrete branch outcome, not a float.
            assert_eq!(
                got.map_index as i64,
                case["map_index"].as_i64().expect("map_index"),
                "case {source}.{name}: temporal bracket index"
            );

            let components: &[(&str, f64)] = &[
                ("s", got.s),
                ("psi", got.psi),
                ("phi_ipp_deg", got.phi_ipp_deg),
                ("lambda_ipp_deg_raw", got.lambda_ipp_deg_raw),
                ("lambda_ipp_deg", got.lambda_ipp_deg),
                ("w", got.w),
                ("vtec0", got.vtec0.expect("lower map value")),
                ("vtec1", got.vtec1.expect("upper map value")),
                ("p0", got.p0),
                ("q0", got.q0),
                ("vtec", got.vtec),
                ("m", got.m),
                ("stec", got.stec),
                ("delay_m", got.delay_m),
            ];

            for &(c, value) in components {
                let want_hex = exp[c].as_str().unwrap_or_else(|| {
                    panic!("case {source}.{name}: missing expected component {c}")
                });
                check(
                    &mut failures,
                    format!("{source}.{name}.{c}"),
                    value,
                    want_hex,
                );
                checks += 1;
            }
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
fn ionex_real_grid_nodes_evaluate_to_parsed_map_values() {
    let ionex = Ionex::parse(REAL_IONEX_EXCERPT).expect("real IONEX excerpt");
    assert_eq!(ionex.lat_nodes_deg(), &[0.0, -2.5]);
    assert_eq!(ionex.lon_nodes_deg().len(), 73);

    let map = &ionex.tec_maps()[0];
    let cases = [
        (0usize, 0usize, 0x4050_4666_6666_6667u64),
        (0, 36, 0x4035_0000_0000_0000),
        (0, 72, 0x4050_4666_6666_6667),
        (1, 0, 0x4050_6000_0000_0000),
        (1, 36, 0x4034_3333_3333_3334),
        (1, 72, 0x4050_6000_0000_0000),
    ];

    for (lat_index, lon_index, expected_bits) in cases {
        let lat_deg = ionex.lat_nodes_deg()[lat_index];
        let lon_deg = ionex.lon_nodes_deg()[lon_index];
        assert_eq!(
            map[lat_index][lon_index]
                .expect("ESA node holds a value")
                .to_bits(),
            expected_bits,
            "IONEX parsed source node lat={lat_deg} lon={lon_deg}"
        );
        let evaluated = super::slant::bilinear_vtec(
            map,
            ionex.lat_nodes_deg(),
            ionex.lon_nodes_deg(),
            ionex.dlat_deg(),
            ionex.dlon_deg(),
            lat_deg,
            lon_deg,
        );
        assert_eq!(
            evaluated.vtec.expect("bilinear value").to_bits(),
            expected_bits,
            "IONEX node lat={lat_deg} lon={lon_deg}"
        );
    }
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
            super::exact_j2000_second(*epoch),
            Some(seconds),
            "IONEX UTC instant must recover the integer J2000-second map epoch"
        );
        assert!(
            matches!(epoch.repr, InstantRepr::JulianDate(_)),
            "IONEX epoch should use the split-Julian-date instant representation"
        );
    }
}

fn synthetic_ionex() -> Ionex {
    let path = fixtures_dir().join("ionex/synthetic_2map_7x7.20i");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    Ionex::parse(&bytes).expect("parse synthetic IONEX product")
}

fn valid_tec_grid_samples() -> TecGridSamples {
    TecGridSamples {
        map_epochs: vec![super::ionex_epoch_from_j2000_seconds(0)],
        lat_nodes_deg: vec![1.0, 0.0],
        lon_nodes_deg: vec![0.0, 1.0],
        dlat_deg: -1.0,
        dlon_deg: 1.0,
        shell_height_km: 450.0,
        base_radius_km: 6371.0,
        exponent: 0,
        tec_maps: vec![vec![
            vec![Some(10.0), Some(11.0)],
            vec![Some(12.0), Some(13.0)],
        ]],
        rms_maps: Vec::new(),
        height_maps: Vec::new(),
        header: IonexHeader::new(IonexMappingFunction::CosZ),
    }
}

fn single_map_ionex_text(epoch: &str) -> String {
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
    text.push_str(&ionex_record(epoch, "EPOCH OF CURRENT MAP"));
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
    text
}

#[test]
fn ionex_from_samples_rejects_empty() {
    let err = Ionex::from_samples(TecGridSamples {
        map_epochs: Vec::new(),
        lat_nodes_deg: Vec::new(),
        lon_nodes_deg: Vec::new(),
        dlat_deg: -1.0,
        dlon_deg: 1.0,
        shell_height_km: 450.0,
        base_radius_km: 6371.0,
        exponent: 0,
        tec_maps: Vec::new(),
        rms_maps: Vec::new(),
        height_maps: Vec::new(),
        header: IonexHeader::new(IonexMappingFunction::CosZ),
    })
    .expect_err("empty IONEX samples must fail");
    assert_eq!(err, TecSamplesError::Empty);
}

#[test]
fn ionex_from_samples_rejects_too_few_nodes() {
    let mut samples = valid_tec_grid_samples();
    samples.lat_nodes_deg = vec![1.0];
    samples.tec_maps = vec![vec![vec![Some(10.0), Some(11.0)]]];
    let err = Ionex::from_samples(samples).expect_err("single latitude node must fail");
    assert_eq!(err, TecSamplesError::TooFewNodes(1));
}

#[test]
fn ionex_from_samples_rejects_non_monotonic_latitudes() {
    let mut samples = valid_tec_grid_samples();
    samples.lat_nodes_deg = vec![0.0, 1.0];
    let err = Ionex::from_samples(samples).expect_err("ascending latitude nodes must fail");
    assert_eq!(err, TecSamplesError::NonMonotonicLat);
}

#[test]
fn ionex_from_samples_rejects_non_monotonic_longitudes() {
    let mut samples = valid_tec_grid_samples();
    samples.lon_nodes_deg = vec![1.0, 0.0];
    let err = Ionex::from_samples(samples).expect_err("descending longitude nodes must fail");
    assert_eq!(err, TecSamplesError::NonMonotonicLon);
}

#[test]
fn ionex_from_samples_rejects_non_monotonic_epochs() {
    let mut samples = valid_tec_grid_samples();
    samples.map_epochs = vec![
        super::ionex_epoch_from_j2000_seconds(10),
        super::ionex_epoch_from_j2000_seconds(0),
    ];
    samples.tec_maps.push(samples.tec_maps[0].clone());
    let err = Ionex::from_samples(samples).expect_err("descending map epochs must fail");
    assert_eq!(err, TecSamplesError::NonMonotonicEpochs);
}

#[test]
fn ionex_from_samples_rejects_epoch_not_representable() {
    let mut samples = valid_tec_grid_samples();
    samples.map_epochs = vec![Instant::from_nanos(TimeScale::Utc, 1)];
    let err = Ionex::from_samples(samples).expect_err("fractional-second epoch must fail");
    assert_eq!(err, TecSamplesError::EpochNotRepresentable);
}

#[test]
fn ionex_from_samples_rejects_dimension_mismatch() {
    let mut samples = valid_tec_grid_samples();
    samples.tec_maps = vec![vec![vec![Some(10.0), Some(11.0)]]];
    let err = Ionex::from_samples(samples).expect_err("short TEC grid must fail");
    assert_eq!(err, TecSamplesError::ShapeMismatch);
}

#[test]
fn ionex_from_samples_rejects_rms_count_mismatch() {
    let mut samples = valid_tec_grid_samples();
    samples.rms_maps = vec![
        vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0), Some(4.0)]],
        vec![vec![Some(5.0), Some(6.0)], vec![Some(7.0), Some(8.0)]],
    ];
    let err = Ionex::from_samples(samples).expect_err("extra RMS map must fail");
    assert_eq!(err, TecSamplesError::RmsCountMismatch);
}

#[test]
fn ionex_from_samples_rejects_non_finite_values() {
    let mut samples = valid_tec_grid_samples();
    samples.tec_maps[0][0][0] = Some(f64::NAN);
    let err = Ionex::from_samples(samples).expect_err("non-finite TEC value must fail");
    assert_eq!(err, TecSamplesError::NonFiniteValue);
}

#[test]
fn ionex_from_samples_rejects_non_positive_step() {
    let mut samples = valid_tec_grid_samples();
    samples.dlon_deg = 0.0;
    let err = Ionex::from_samples(samples).expect_err("zero longitude step must fail");
    assert_eq!(err, TecSamplesError::NonPositiveStep);
}

#[test]
fn ionex_from_samples_rejects_axis_out_of_range() {
    let mut samples = valid_tec_grid_samples();
    samples.lon_nodes_deg = vec![0.0, 361.0];
    let err = Ionex::from_samples(samples).expect_err("longitude node out of range must fail");
    assert_eq!(err, TecSamplesError::AxisOutOfRange(361.0));
}

#[test]
fn ionex_from_samples_round_trips_parsed_epoch_eleven_seconds_after_j2000() {
    let text = single_map_ionex_text("2000 1 1 12 0 11");
    let parsed = Ionex::parse_str(&text).expect("valid J2000+11s IONEX parses");
    assert_eq!(parsed.map_epochs_s(), vec![11]);

    let rebuilt = Ionex::from_samples(parsed.tec_grid_samples())
        .expect("parsed J2000+11s samples rebuild cleanly");
    assert_eq!(
        rebuilt, parsed,
        "sample round-trip preserves a parsed J2000+11s epoch"
    );
}

#[test]
fn ionex_from_samples_rebuilds_synthetic_ir_byte_identically() {
    // No real IGS `.YYi` fixture is added in this change. This synthetic
    // reconstruction plus the golden branch matrix covers the current target;
    // a future real fixture can add another byte-identical reconstruction guard.
    let original = synthetic_ionex();
    let samples = original.tec_grid_samples();
    assert_eq!(samples.dlat_deg.to_bits(), original.dlat_deg().to_bits());
    assert_eq!(samples.dlon_deg.to_bits(), original.dlon_deg().to_bits());
    let rebuilt = Ionex::from_samples(samples).expect("sample-built IONEX");
    assert_eq!(
        rebuilt, original,
        "sample round-trip preserves the IONEX IR"
    );
}

#[test]
fn ionex_from_node_samples_treats_signed_zero_as_one_axis_node() {
    let epoch = super::ionex_epoch_from_j2000_seconds(0);
    let samples = [
        TecSample {
            epoch,
            lat_deg: 1.0,
            lon_deg: -0.0,
            vtec_tecu: Some(10.0),
            rms_tecu: None,
            height_offset_km: None,
        },
        TecSample {
            epoch,
            lat_deg: 1.0,
            lon_deg: 1.0,
            vtec_tecu: Some(11.0),
            rms_tecu: None,
            height_offset_km: None,
        },
        TecSample {
            epoch,
            lat_deg: 0.0,
            lon_deg: 0.0,
            vtec_tecu: Some(12.0),
            rms_tecu: None,
            height_offset_km: None,
        },
        TecSample {
            epoch,
            lat_deg: 0.0,
            lon_deg: 1.0,
            vtec_tecu: Some(13.0),
            rms_tecu: None,
            height_offset_km: None,
        },
    ];

    let ionex = Ionex::from_node_samples(
        samples,
        450.0,
        6371.0,
        0,
        IonexHeader::new(IonexMappingFunction::CosZ),
    )
    .expect("signed-zero nodes rebuild");
    assert_eq!(ionex.lon_nodes_deg().len(), 2);
    assert_eq!(ionex.lon_nodes_deg()[0].to_bits(), (-0.0_f64).to_bits());
    assert_eq!(
        ionex.tec_maps()[0],
        vec![vec![Some(10.0), Some(11.0)], vec![Some(12.0), Some(13.0)]]
    );
}

#[test]
fn ionex_from_node_samples_rebuilds_synthetic_ir_byte_identically() {
    let original = synthetic_ionex();
    let rebuilt = Ionex::from_node_samples(
        original.tec_samples(),
        original.shell_height_km(),
        original.base_radius_km(),
        original.exponent(),
        original.header().clone(),
    )
    .expect("flat node samples rebuild IONEX");
    assert_eq!(
        rebuilt, original,
        "flat node samples preserve the synthetic IONEX IR"
    );
}

#[test]
fn ionex_from_samples_text_round_trip_reparses_to_same_ir() {
    let original = synthetic_ionex();
    let sample_built =
        Ionex::from_samples(original.tec_grid_samples()).expect("sample-built IONEX");
    let encoded = sample_built.to_ionex_string().expect("writable IONEX");
    let reparsed = Ionex::parse_str(&encoded).expect("serialized sample-built IONEX reparses");
    assert_eq!(
        reparsed, sample_built,
        "serialized sample-built IONEX preserves lattice-aligned VTEC"
    );
}

#[test]
fn ionex_slant_delays_batch_matches_scalar_bits() {
    let parsed = synthetic_ionex();
    let sample_built = Ionex::from_samples(parsed.tec_grid_samples()).expect("sample-built IONEX");
    let f_l1 = crate::frequencies::frequency_hz(
        crate::GnssSystem::Gps,
        crate::frequencies::CarrierBand::L1,
    )
    .expect("canonical GPS L1 carrier exists");
    let f_l2 = crate::frequencies::frequency_hz(
        crate::GnssSystem::Gps,
        crate::frequencies::CarrierBand::L2,
    )
    .expect("canonical GPS L2 carrier exists");
    let epochs = parsed.map_epochs_s();
    let requests = vec![
        IonexSlantRequest {
            receiver: crate::frame::Wgs84Geodetic::new(
                30.0_f64.to_radians(),
                0.0_f64.to_radians(),
                0.0,
            )
            .expect("valid receiver"),
            elevation_rad: 45.0_f64.to_radians(),
            azimuth_rad: 90.0_f64.to_radians(),
            epoch_j2000_s: epochs[0],
            frequency_hz: f_l1,
        },
        IonexSlantRequest {
            receiver: crate::frame::Wgs84Geodetic::new(
                -15.0_f64.to_radians(),
                170.0_f64.to_radians(),
                250.0,
            )
            .expect("valid receiver"),
            elevation_rad: 20.0_f64.to_radians(),
            azimuth_rad: 250.0_f64.to_radians(),
            epoch_j2000_s: (epochs[0] + epochs[1]) / 2,
            frequency_hz: f_l2,
        },
        IonexSlantRequest {
            receiver: crate::frame::Wgs84Geodetic::new(
                0.0_f64.to_radians(),
                0.0_f64.to_radians(),
                10.0,
            )
            .expect("valid receiver"),
            elevation_rad: 90.0_f64.to_radians(),
            azimuth_rad: 0.0_f64.to_radians(),
            epoch_j2000_s: epochs[1],
            frequency_hz: f_l1,
        },
    ];

    for product in [&parsed, &sample_built] {
        let mut out = vec![f64::NAN; requests.len()];
        product
            .slant_delays_batch(&requests, &mut out)
            .expect("valid method batch");
        let vec_out = product
            .slant_delays_batch_vec(&requests)
            .expect("valid vector batch");
        assert_eq!(vec_out, out, "vector batch matches slice-output batch");
        let mut free_out = vec![f64::NAN; requests.len()];
        ionex_slant_delays(product, &requests, &mut free_out).expect("valid function batch");
        assert_eq!(free_out, out, "function batch matches method batch");
        for (request, got) in requests.iter().zip(out) {
            let scalar = super::ionex_slant_delay(
                product,
                request.receiver,
                request.elevation_rad,
                request.azimuth_rad,
                request.epoch_j2000_s,
                request.frequency_hz,
            )
            .expect("valid scalar");
            assert_eq!(
                got.to_bits(),
                scalar.to_bits(),
                "batch and scalar IONEX slant delay must match bit-for-bit"
            );
        }
    }

    let mut short = vec![0.0; requests.len() - 1];
    let err = ionex_slant_delays(&parsed, &requests, &mut short)
        .expect_err("short output slice must fail");
    assert!(
        matches!(err, crate::error::Error::InvalidInput(_)),
        "length mismatch should be InvalidInput, got {err:?}"
    );

    for bad_request in [
        IonexSlantRequest {
            receiver: crate::frame::Wgs84Geodetic {
                lat_rad: f64::NAN,
                lon_rad: 0.0,
                height_m: 0.0,
            },
            ..requests[0]
        },
        IonexSlantRequest {
            elevation_rad: -1.0e-6,
            ..requests[0]
        },
        IonexSlantRequest {
            azimuth_rad: f64::INFINITY,
            ..requests[0]
        },
        IonexSlantRequest {
            frequency_hz: 0.0,
            ..requests[0]
        },
    ] {
        let scalar_err = super::ionex_slant_delay(
            &parsed,
            bad_request.receiver,
            bad_request.elevation_rad,
            bad_request.azimuth_rad,
            bad_request.epoch_j2000_s,
            bad_request.frequency_hz,
        )
        .expect_err("bad scalar request must fail");
        let mut out = [0.0];
        let batch_err = ionex_slant_delays(&parsed, &[bad_request], &mut out)
            .expect_err("bad batch request must fail");
        assert_eq!(
            format!("{batch_err:?}"),
            format!("{scalar_err:?}"),
            "batch validation error must match scalar validation error"
        );
    }
}

fn coverage_ionex(epoch_s: &[i64]) -> Ionex {
    let mut samples = valid_tec_grid_samples();
    // Forces a zenith pierce point to land exactly on the receiver coordinate.
    samples.base_radius_km = 0.0;
    samples.map_epochs = epoch_s
        .iter()
        .map(|&seconds| super::ionex_epoch_from_j2000_seconds(seconds))
        .collect();
    let map = vec![vec![Some(10.0), Some(11.0)], vec![Some(12.0), Some(13.0)]];
    samples.tec_maps = epoch_s.iter().map(|_| map.clone()).collect();
    Ionex::from_samples(samples).expect("coverage-test IONEX")
}

fn coverage_request(lat_deg: f64, lon_deg: f64, epoch_j2000_s: i64) -> IonexSlantRequest {
    IonexSlantRequest {
        receiver: crate::frame::Wgs84Geodetic::new(lat_deg.to_radians(), lon_deg.to_radians(), 0.0)
            .expect("valid receiver"),
        elevation_rad: core::f64::consts::FRAC_PI_2,
        azimuth_rad: 0.0,
        epoch_j2000_s,
        frequency_hz: 1_575_420_000.0,
    }
}

fn assert_ionex_coverage_error<T: core::fmt::Debug>(
    result: crate::Result<T>,
    expected: IonexCoverageError,
) {
    assert_eq!(
        result.expect_err("request must fail coverage"),
        crate::error::Error::IonexOutOfCoverage(expected)
    );
}

#[test]
fn ionex_strict_rejects_epoch_outside_coverage_and_hold_marks_status() {
    let ionex = coverage_ionex(&[0, 10]);
    let first = coverage_request(0.5, 0.5, 0);
    let last = coverage_request(0.5, 0.5, 10);
    let before = coverage_request(0.5, 0.5, -1);
    let after = coverage_request(0.5, 0.5, 11);

    for request in [first, last] {
        let strict = ionex_slant_delay_with_policy(
            &ionex,
            request.receiver,
            request.elevation_rad,
            request.azimuth_rad,
            request.epoch_j2000_s,
            request.frequency_hz,
            IonexCoveragePolicy::Strict.into(),
        )
        .expect("boundary epoch is covered");
        assert_eq!(strict.status, IonexSlantDelayStatus::VALID);
        let scalar = super::ionex_slant_delay(
            &ionex,
            request.receiver,
            request.elevation_rad,
            request.azimuth_rad,
            request.epoch_j2000_s,
            request.frequency_hz,
        )
        .expect("strict scalar boundary epoch is covered");
        assert_eq!(scalar.to_bits(), strict.delay_m.to_bits());
    }

    assert_ionex_coverage_error(
        super::ionex_slant_delay(
            &ionex,
            before.receiver,
            before.elevation_rad,
            before.azimuth_rad,
            before.epoch_j2000_s,
            before.frequency_hz,
        ),
        IonexCoverageError::EpochBeforeFirstMap,
    );
    assert_ionex_coverage_error(
        super::ionex_slant_delay(
            &ionex,
            after.receiver,
            after.elevation_rad,
            after.azimuth_rad,
            after.epoch_j2000_s,
            after.frequency_hz,
        ),
        IonexCoverageError::EpochAfterLastMap,
    );

    let held_before = ionex_slant_delay_with_policy(
        &ionex,
        before.receiver,
        before.elevation_rad,
        before.azimuth_rad,
        before.epoch_j2000_s,
        before.frequency_hz,
        IonexCoveragePolicy::Hold.into(),
    )
    .expect("hold policy returns before-coverage value");
    assert_eq!(
        held_before.status,
        IonexSlantDelayStatus {
            held: Some(IonexCoverageError::EpochBeforeFirstMap),
            degraded: None,
            assumed_mapping: None,
        }
    );

    let held_after = ionex_slant_delay_with_policy(
        &ionex,
        after.receiver,
        after.elevation_rad,
        after.azimuth_rad,
        after.epoch_j2000_s,
        after.frequency_hz,
        IonexCoveragePolicy::Hold.into(),
    )
    .expect("hold policy returns after-coverage value");
    assert_eq!(
        held_after.status,
        IonexSlantDelayStatus {
            held: Some(IonexCoverageError::EpochAfterLastMap),
            degraded: None,
            assumed_mapping: None,
        }
    );
}

#[test]
fn ionex_strict_rejects_spatial_outside_coverage_and_includes_boundaries() {
    let ionex = coverage_ionex(&[0]);
    for request in [coverage_request(1.0, 0.0, 0), coverage_request(0.0, 1.0, 0)] {
        let eval = ionex_slant_delay_with_policy(
            &ionex,
            request.receiver,
            request.elevation_rad,
            request.azimuth_rad,
            request.epoch_j2000_s,
            request.frequency_hz,
            IonexCoveragePolicy::Strict.into(),
        )
        .expect("grid boundary is covered");
        assert_eq!(eval.status, IonexSlantDelayStatus::VALID);
    }

    let north = coverage_request(1.25, 0.5, 0);
    let south = coverage_request(-0.25, 0.5, 0);
    let west = coverage_request(0.5, -0.25, 0);
    let east = coverage_request(0.5, 1.25, 0);

    for request in [north, south] {
        assert_ionex_coverage_error(
            super::ionex_slant_delay(
                &ionex,
                request.receiver,
                request.elevation_rad,
                request.azimuth_rad,
                request.epoch_j2000_s,
                request.frequency_hz,
            ),
            IonexCoverageError::LatitudeOutOfRange,
        );
    }
    for request in [west, east] {
        assert_ionex_coverage_error(
            super::ionex_slant_delay(
                &ionex,
                request.receiver,
                request.elevation_rad,
                request.azimuth_rad,
                request.epoch_j2000_s,
                request.frequency_hz,
            ),
            IonexCoverageError::LongitudeOutOfRange,
        );
    }

    let held = ionex_slant_delay_with_policy(
        &ionex,
        east.receiver,
        east.elevation_rad,
        east.azimuth_rad,
        east.epoch_j2000_s,
        east.frequency_hz,
        IonexCoveragePolicy::Hold.into(),
    )
    .expect("hold policy returns spatial edge value");
    assert_eq!(
        held.status,
        IonexSlantDelayStatus {
            held: Some(IonexCoverageError::LongitudeOutOfRange),
            degraded: None,
            assumed_mapping: None,
        }
    );
}

#[test]
fn ionex_batch_results_report_coverage_per_element() {
    let ionex = coverage_ionex(&[0, 10]);
    let requests = [
        coverage_request(0.5, 0.5, 0),
        coverage_request(0.5, 0.5, -1),
        coverage_request(0.5, 1.25, 0),
        IonexSlantRequest {
            frequency_hz: 0.0,
            ..coverage_request(0.5, 0.5, 0)
        },
    ];

    let strict = ionex_slant_delay_results(&ionex, &requests, IonexCoveragePolicy::Strict.into());
    assert_eq!(strict.len(), requests.len());
    assert_eq!(
        strict[0].as_ref().expect("first element covered").status,
        IonexSlantDelayStatus::VALID
    );
    assert_eq!(
        strict[1]
            .as_ref()
            .expect_err("second element before coverage"),
        &crate::error::Error::IonexOutOfCoverage(IonexCoverageError::EpochBeforeFirstMap)
    );
    assert_eq!(
        strict[2]
            .as_ref()
            .expect_err("third element longitude coverage"),
        &crate::error::Error::IonexOutOfCoverage(IonexCoverageError::LongitudeOutOfRange)
    );
    assert!(
        matches!(strict[3], Err(crate::error::Error::InvalidInput(_))),
        "invalid request should stay element-local"
    );

    let held = ionex.slant_delays_batch_results(&requests[..3], IonexCoveragePolicy::Hold.into());
    assert_eq!(
        held[0].as_ref().expect("first element covered").status,
        IonexSlantDelayStatus::VALID
    );
    assert_eq!(
        held[1].as_ref().expect("second element held").status,
        IonexSlantDelayStatus {
            held: Some(IonexCoverageError::EpochBeforeFirstMap),
            degraded: None,
            assumed_mapping: None,
        }
    );
    assert_eq!(
        held[2].as_ref().expect("third element held").status,
        IonexSlantDelayStatus {
            held: Some(IonexCoverageError::LongitudeOutOfRange),
            degraded: None,
            assumed_mapping: None,
        }
    );

    let mut out = [0.0; 3];
    assert_ionex_coverage_error(
        ionex_slant_delays(&ionex, &requests[..3], &mut out),
        IonexCoverageError::EpochBeforeFirstMap,
    );
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
            t_gal_s: SECONDS_PER_DAY,
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
    let maps: Vec<Vec<Vec<Option<f64>>>> = maps
        .iter()
        .map(|map| {
            map.iter()
                .map(|row| row.iter().copied().map(Some).collect())
                .collect()
        })
        .collect();
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
            maps: &maps,
            lat_arr: &lat_arr,
            lon_arr,
            dlat: -1.0,
            dlon: lon_arr[1] - lon_arr[0],
        },
    )
    .expect("every node holds a value")
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
    let encoded = original.to_ionex_string().expect("writable IONEX");
    let reparsed = Ionex::parse_str(&encoded).expect("serialized IONEX reparses");
    assert_eq!(reparsed, original, "round-trip preserves the IONEX IR");
    assert_eq!(
        reparsed.to_ionex_string().expect("writable IONEX"),
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

    let encoded = original.to_ionex_string().expect("writable IONEX");
    let reparsed = Ionex::parse_str(&encoded).expect("serialized IONEX reparses");
    assert_eq!(reparsed, original, "round-trip preserves TEC and RMS grids");
    assert_eq!(
        reparsed.to_ionex_string().expect("writable IONEX"),
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
        bits_vec(&doc["values_bits"])
            .into_iter()
            .map(Some)
            .collect(),
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
            libm::atan2(xyz[1], xyz[0]) * crate::constants::RAD_TO_DEG,
            libm::atan2(xyz[2], p) * crate::constants::RAD_TO_DEG,
            r - super::tec_grid::EARTH_RADIUS_M,
        ]
    }

    let grid = TecGrid::new(
        vec![0.0, 1_000_000_000.0],
        vec![-90.0, 90.0],
        vec![-180.0, 180.0],
        vec![Some(10.0); 8],
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
    let default_arg = EARTH_RADIUS_M * libm::cos(elevation_rad) / default_shell_radius_m;
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
        custom_shell.earth_radius_m * libm::cos(elevation_rad) / custom_shell.shell_radius_m();
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

// ---------------------------------------------------------------------------
// IONEX 1 layout: the spec's examples, real products, and RTKLIB's reading.
// ---------------------------------------------------------------------------

fn fixture_text(name: &str) -> String {
    let path = fixtures_dir().join("ionex").join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()))
}

fn scaled(raw: i64, exponent: i32) -> f64 {
    super::grid::scale_value(raw, exponent)
}

fn node_bits(value: Option<f64>) -> Option<u64> {
    value.map(f64::to_bits)
}

const REAL_TRIMS: [&str; 5] = [
    "COD0OPSFIN_20240010000_01D_01H_GIM_trim",
    "EMR0OPSFIN_20240010000_01D_01H_GIM_trim",
    "IGS0OPSFIN_20240010000_01D_02H_GIM_trim",
    "uqrg0010.24i_trim",
    "uqrg0010.24i_nan_trim",
];

/// The warnings and skipped records a trimmed real product reads with: the
/// `uqrg` products have no `OBSERVABLES USED` or `END OF FILE` record and seven
/// `AUX DATA` blocks, and each other product one `AUX DATA` block. The `uqrg`
/// trim around map 55 keeps the one RMS field that product gives as `nan`.
fn real_trim_findings(name: &str) -> (Vec<IonexWarning>, usize) {
    if name.starts_with("uqrg") {
        // The header findings come first, then the ones the maps gave.
        let mut warnings = vec![
            IonexWarning::MissingRecord("OBSERVABLES USED"),
            IonexWarning::MissingRecord("END OF FILE"),
        ];
        if name.ends_with("nan_trim") {
            warnings.push(IonexWarning::NotANumberValue {
                kind: "RMS",
                map_number: 1,
                line: 227,
                lat_deg: 40.0,
                lon_deg: -125.0,
            });
        }
        (warnings, 7)
    } else {
        (Vec::new(), 1)
    }
}

#[test]
fn ionex_spec_example_2d_reads_every_map_by_its_own_records() {
    let (ionex, warnings) =
        Ionex::parse_str_with_warnings(&fixture_text("spec_example_2d.inx")).expect("example");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(ionex.skipped_records(), 0);
    assert_eq!(ionex.lat_nodes_deg(), &[85.0, 80.0, 75.0]);
    assert_eq!(ionex.lon_nodes_deg().len(), 72);
    assert_eq!(ionex.map_epochs_s().len(), 2);
    assert_eq!(ionex.exponent(), -1);
    assert_eq!(ionex.shell_height_km(), 400.0);

    let header = ionex.header();
    assert_eq!(header.version, 1.0);
    assert_eq!(header.satellite_system, "GPS");
    assert_eq!(header.program, "ionpgm v1.0");
    assert_eq!(header.run_by, "aiub");
    assert_eq!(header.date, "29-jan-96 17:29");
    assert_eq!(header.interval_s, 21600);
    assert_eq!(header.mapping_function, Some(IonexMappingFunction::CosZ));
    assert_eq!(header.elevation_cutoff_deg, 20.0);
    assert_eq!(header.observables_used, "double-difference carrier phase");
    assert_eq!(header.station_count, Some(80));
    assert_eq!(header.satellite_count, Some(24));
    assert_eq!(header.descriptions.len(), 2);
    assert_eq!(header.comments.len(), 6);

    for i in 0..3usize {
        for j in 0..72usize {
            let (ri, rj) = (i as i64, j as i64);
            let tec1 = (i, j) != (1, 1);
            assert_eq!(
                node_bits(ionex.tec_maps()[0][i][j]),
                node_bits(tec1.then(|| scaled(1000 + 100 * ri + rj, -1))),
                "TEC map 1 [{i}][{j}]"
            );
            let tec2 = match i {
                0 => Some(scaled(10000 + 10 * rj, -2)),
                1 => Some(scaled(100 + rj, -1)),
                _ => (j != 71).then(|| scaled(12000 + 10 * rj, -2)),
            };
            assert_eq!(
                node_bits(ionex.tec_maps()[1][i][j]),
                node_bits(tec2),
                "TEC map 2 [{i}][{j}]"
            );
            assert_eq!(
                node_bits(ionex.rms_maps()[0][i][j]),
                node_bits(((i, j) != (0, 0)).then(|| scaled(10 + ri + rj, -1))),
                "RMS map 1 [{i}][{j}]"
            );
            assert_eq!(
                node_bits(ionex.rms_maps()[1][i][j]),
                node_bits(Some(scaled(20 + ri + rj, -1))),
                "RMS map 2 [{i}][{j}]"
            );
            assert_eq!(
                node_bits(ionex.height_maps()[0][i][j]),
                node_bits(((i, j) != (2, 5)).then(|| scaled(10 * ri + rj, -1))),
                "height map 1 [{i}][{j}]"
            );
            assert_eq!(
                node_bits(ionex.height_maps()[1][i][j]),
                node_bits(Some(scaled(5 + rj, -1))),
                "height map 2 [{i}][{j}]"
            );
        }
    }
}

#[test]
fn ionex_spec_example_3d_is_refused_by_name() {
    let text = fixture_text("spec_example_3d.inx");
    let err = Ionex::parse_str(&text).expect_err("3-D maps");
    assert!(err.to_string().contains("MAP DIMENSION 3"), "{err}");

    let without_dimension: String = text
        .lines()
        .filter(|line| !line.ends_with("MAP DIMENSION"))
        .map(|line| format!("{line}\n"))
        .collect();
    let err = Ionex::parse_str(&without_dimension).expect_err("3-D heights");
    assert!(err.to_string().contains("more than one height"), "{err}");
}

#[test]
fn ionex_real_products_match_rtklib_at_every_node() {
    for name in REAL_TRIMS {
        let (ionex, warnings) =
            Ionex::parse_str_with_warnings(&fixture_text(&format!("{name}.INX"))).expect(name);
        let (expected_warnings, expected_skips) = real_trim_findings(name);
        assert_eq!(warnings, expected_warnings, "{name}");
        assert_eq!(
            ionex.skipped_records(),
            expected_skips,
            "{name}: AUX DATA blocks"
        );

        let listing = fixture_text(&format!("rtklib/{name}.nodes"));
        let mut map = 0usize;
        let mut nodes = 0usize;
        let mut not_available = 0usize;
        let mut rms_not_available = 0usize;
        for line in listing.lines() {
            let fields: Vec<&str> = line.split_whitespace().collect();
            match fields[0] {
                "file" => continue,
                "map" => {
                    map = fields[1].parse::<usize>().expect("map number") - 1;
                    assert_eq!(
                        ionex.map_epochs_s()[map],
                        crate::astro::time::civil::j2000_seconds(
                            fields[2][0..4].parse().expect("year"),
                            fields[2][5..7].parse().expect("month"),
                            fields[2][8..10].parse().expect("day"),
                            fields[3][0..2].parse().expect("hour"),
                            fields[3][3..5].parse().expect("minute"),
                            0.0,
                        ) as i64,
                        "{name}: epoch of map {}",
                        map + 1
                    );
                    continue;
                }
                _ => {}
            }
            let [k, i, j] =
                [fields[0], fields[1], fields[2]].map(|f| f.parse::<usize>().expect("index"));
            assert_eq!(k, 0, "{name}: one height layer");
            // RTKLIB prints the node it read from a `nan` field as `nan`, which
            // is no hex float.
            let rtklib_value = |field: &str| {
                if field.eq_ignore_ascii_case("nan") {
                    f64::NAN
                } else {
                    parse_hex_float(field)
                }
            };
            let rtklib_tec = rtklib_value(fields[3]);
            let rtklib_rms = rtklib_value(fields[4]);
            match ionex.tec_maps()[map][i][j] {
                // RTKLIB forms a value as `raw * 10^k`, this reader as the
                // decimal `raw / 10^-k`, which IEEE division rounds correctly.
                // The two are the same number to one unit in the last place,
                // and differ by that much wherever the product is not the
                // decimal: RTKLIB reads `73` at EXPONENT -1 as
                // 7.300000000000001, this reader as 7.3.
                Some(tec) => assert!(
                    ulp_distance(tec, rtklib_tec) <= 1,
                    "{name}: TEC map {} [{i}][{j}] sidereon {tec} rtklib {rtklib_tec}",
                    map + 1
                ),
                None => {
                    assert_eq!(rtklib_tec, 0.0, "{name}: RTKLIB leaves a 9999 node unset");
                    not_available += 1;
                }
            }
            match ionex.rms_maps()[map][i][j] {
                Some(rms) => assert_eq!(
                    (rms as f32).to_bits(),
                    (rtklib_rms as f32).to_bits(),
                    "{name}: RMS map {} [{i}][{j}] (RTKLIB keeps RMS as a float)",
                    map + 1
                ),
                // RTKLIB reads the `nan` field of uqrg through `sscanf`, which
                // gives it a NaN; this reader gives the node no value.
                None => {
                    assert!(
                        rtklib_rms.is_nan(),
                        "{name}: RMS map {} [{i}][{j}] is non-available here, RTKLIB has \
                         {rtklib_rms}",
                        map + 1
                    );
                    rms_not_available += 1;
                }
            }
            nodes += 1;
        }
        assert_eq!(
            nodes,
            ionex.tec_maps().len() * ionex.lat_nodes_deg().len() * ionex.lon_nodes_deg().len(),
            "{name}: every node compared"
        );
        let expected_not_available = usize::from(name.starts_with("EMR"));
        assert_eq!(not_available, expected_not_available, "{name}");
        assert_eq!(
            rms_not_available,
            usize::from(name.ends_with("nan_trim")),
            "{name}: RMS nodes without a value"
        );
    }
}

#[test]
fn ionex_hour_24_reads_as_midnight_of_the_next_day() {
    let text = fixture_text("uqrg0010.24i_trim.INX");
    let ionex = Ionex::parse_str(&text).expect("uqrg trim");
    let epochs = ionex.map_epochs_s();
    let midnight = crate::astro::time::civil::j2000_seconds(2024, 1, 2, 0, 0, 0.0) as i64;
    assert_eq!(epochs, vec![midnight - 1800, midnight - 900, midnight]);
    assert_eq!(ionex.header().interval_s, 900);

    let refused = text.replacen(
        "  2024     1     1    24     0     0                        EPOCH OF CURRENT MAP",
        "  2024     1     1    24     0     1                        EPOCH OF CURRENT MAP",
        1,
    );
    assert_ne!(refused, text);
    let message = Ionex::parse_str(&refused)
        .expect_err("24:00:01")
        .to_string();
    assert!(message.contains("IONEX epoch"), "{message}");
}

#[test]
fn ionex_real_non_available_node_reads_as_none() {
    let ionex = Ionex::parse_str(&fixture_text("EMR0OPSFIN_20240010000_01D_01H_GIM_trim.INX"))
        .expect("EMR trim");
    assert_eq!(ionex.lat_nodes_deg()[0], 10.0);
    assert_eq!(ionex.lon_nodes_deg()[65], 145.0);
    assert_eq!(ionex.tec_maps()[1][0][65], None);
    assert_eq!(
        ionex
            .tec_maps()
            .iter()
            .flatten()
            .flatten()
            .filter(|v| v.is_none())
            .count(),
        1
    );
    assert_eq!(ionex.rms_maps()[1][0][65], Some(scaled(44, -1)));
    assert_eq!(
        ionex.header().mapping_function,
        Some(IonexMappingFunction::Other("MOD".into()))
    );

    let code = Ionex::parse_str(&fixture_text("COD0OPSFIN_20240010000_01D_01H_GIM_trim.INX"))
        .expect("CODE trim");
    assert_eq!(
        code.header().mapping_function,
        Some(IonexMappingFunction::NoMapping)
    );
    assert_eq!(code.header().program, "ADDNEQ2 V5.5");
    assert_eq!(code.header().satellite_system, "GNSS");
    let igs = Ionex::parse_str(&fixture_text("IGS0OPSFIN_20240010000_01D_02H_GIM_trim.INX"))
        .expect("IGS trim");
    assert_eq!(
        igs.header().mapping_function,
        Some(IonexMappingFunction::CosZ)
    );
}

#[test]
fn ionex_real_products_round_trip_through_the_serializer() {
    for name in REAL_TRIMS {
        let original = Ionex::parse_str(&fixture_text(&format!("{name}.INX"))).expect(name);
        let reparsed = Ionex::parse_str(&original.to_ionex_string().expect("writable IONEX"))
            .expect("reparse");
        let without_skips =
            Ionex::from_samples(original.tec_grid_samples()).expect("sample-built copy");
        assert_eq!(reparsed, without_skips, "{name}");
    }
}

// ---------------------------------------------------------------------------
// Records built one at a time.
// ---------------------------------------------------------------------------

const LAYOUT_LAT: &str = "     1.0   0.0  -1.0";
const LAYOUT_LON: &str = "     0.0   1.0   1.0";
const LAYOUT_EPOCH_0: &str = "  2020     1     1     0     0     0";
const LAYOUT_EPOCH_1: &str = "  2020     1     1     1     0     0";

fn layout_header(maps: usize, last_epoch: &str, extra: &str) -> String {
    let mut t = String::new();
    t.push_str(&ionex_record(
        "     1.0            IONOSPHERE MAPS     GPS",
        "IONEX VERSION / TYPE",
    ));
    t.push_str(&ionex_record("layout test", "PGM / RUN BY / DATE"));
    t.push_str(&ionex_record(LAYOUT_EPOCH_0, "EPOCH OF FIRST MAP"));
    t.push_str(&ionex_record(last_epoch, "EPOCH OF LAST MAP"));
    t.push_str(&ionex_record("  3600", "INTERVAL"));
    t.push_str(&ionex_record(&format!("{maps:6}"), "# OF MAPS IN FILE"));
    t.push_str(&ionex_record("  COSZ", "MAPPING FUNCTION"));
    t.push_str(&ionex_record("     0.0", "ELEVATION CUTOFF"));
    t.push_str(&ionex_record("", "OBSERVABLES USED"));
    t.push_str(&ionex_record("  6371.0", "BASE RADIUS"));
    t.push_str(&ionex_record("     2", "MAP DIMENSION"));
    t.push_str(&ionex_record("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"));
    t.push_str(&ionex_record(LAYOUT_LAT, "LAT1 / LAT2 / DLAT"));
    t.push_str(&ionex_record(LAYOUT_LON, "LON1 / LON2 / DLON"));
    t.push_str(extra);
    t.push_str(&ionex_record("", "END OF HEADER"));
    t
}

fn layout_band(lat: f64, lon1: f64, lon2: f64, dlon: f64, h: f64, values: &str) -> String {
    format!(
        "{}{values}\n",
        ionex_record(
            &format!("  {lat:6.1}{lon1:6.1}{lon2:6.1}{dlon:6.1}{h:6.1}"),
            "LAT/LON1/LON2/DLON/H"
        )
    )
}

fn layout_bands(north: &str, south: &str) -> String {
    layout_band(1.0, 0.0, 1.0, 1.0, 450.0, north) + &layout_band(0.0, 0.0, 1.0, 1.0, 450.0, south)
}

fn layout_map(kind: &str, index: usize, epoch: Option<&str>, body: &str) -> String {
    let mut t = ionex_record(&format!("{index:6}"), &format!("START OF {kind} MAP"));
    if let Some(epoch) = epoch {
        t.push_str(&ionex_record(epoch, "EPOCH OF CURRENT MAP"));
    }
    t.push_str(body);
    t.push_str(&ionex_record(
        &format!("{index:6}"),
        &format!("END OF {kind} MAP"),
    ));
    t
}

fn layout_end() -> String {
    ionex_record("", "END OF FILE")
}

fn exponent_record(exponent: i32) -> String {
    ionex_record(&format!("{exponent:6}"), "EXPONENT")
}

fn one_map_file(extra_header: &str, body: &str) -> String {
    layout_header(1, LAYOUT_EPOCH_0, extra_header)
        + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), body)
        + &layout_end()
}

fn parse_error(text: &str) -> String {
    Ionex::parse_str(text).expect_err("refused").to_string()
}

#[test]
fn ionex_bands_are_placed_by_their_own_latitude_and_longitudes() {
    // South band first, its longitudes east to west; north band split in two.
    let body = layout_band(0.0, 1.0, 0.0, -1.0, 450.0, "    4    3")
        + &layout_band(1.0, 1.0, 1.0, 1.0, 450.0, "    2")
        + &layout_band(1.0, 0.0, 0.0, 1.0, 450.0, "    1");
    let text = one_map_file(&exponent_record(0), &body);
    let (ionex, warnings) = Ionex::parse_str_with_warnings(&text).expect("placed bands");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        ionex.tec_maps()[0],
        vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0), Some(4.0)]]
    );
}

#[test]
fn ionex_band_off_the_header_grid_is_refused_by_name() {
    let exponent = exponent_record(0);
    let north = layout_band(1.0, 0.0, 1.0, 1.0, 450.0, "    1    2");
    for (band, expected) in [
        (
            layout_band(0.5, 0.0, 1.0, 1.0, 450.0, "    3    4"),
            "latitude 0.5",
        ),
        (
            layout_band(0.0, 5.0, 6.0, 1.0, 450.0, "    3    4"),
            "longitude 5",
        ),
        (
            layout_band(0.0, 0.0, 1.0, 1.0, 350.0, "    3    4"),
            "height 350",
        ),
        (
            layout_band(0.0, 0.0, 1.0, 0.0, 450.0, "    3    4"),
            "not a range",
        ),
    ] {
        let message = parse_error(&one_map_file(&exponent, &(north.clone() + &band)));
        assert!(message.contains(expected), "{expected}: {message}");
    }
}

#[test]
fn ionex_node_given_twice_or_left_without_a_value_is_refused() {
    let exponent = exponent_record(0);
    let north = layout_band(1.0, 0.0, 1.0, 1.0, 450.0, "    1    2");
    let twice = parse_error(&one_map_file(&exponent, &(north.clone() + &north)));
    assert!(twice.contains("latitude 1 longitude 0 twice"), "{twice}");

    let missing_band = parse_error(&one_map_file(&exponent, &north));
    assert!(
        missing_band.contains("no LAT/LON1/LON2/DLON/H band for latitude 0"),
        "{missing_band}"
    );

    let partial = north + &layout_band(0.0, 0.0, 0.0, 1.0, 450.0, "    3");
    let partial = parse_error(&one_map_file(&exponent, &partial));
    assert!(
        partial.contains("gives no value for latitude 0 longitude 1"),
        "{partial}"
    );
}

#[test]
fn ionex_in_map_exponent_sets_the_unit_of_the_blocks_after_it() {
    let before_first = exponent_record(-2) + &layout_bands("  100  200", "  300  400");
    let ionex = Ionex::parse_str(&one_map_file("", &before_first)).expect("exponent in map");
    assert_eq!(
        ionex.tec_maps()[0],
        vec![
            vec![Some(scaled(100, -2)), Some(scaled(200, -2))],
            vec![Some(scaled(300, -2)), Some(scaled(400, -2))]
        ]
    );
    assert_eq!(
        ionex.exponent(),
        -1,
        "the header exponent stays the default"
    );

    let between = layout_band(1.0, 0.0, 1.0, 1.0, 450.0, "   10   20")
        + &exponent_record(-2)
        + &layout_band(0.0, 0.0, 1.0, 1.0, 450.0, "  300  400");
    let ionex = Ionex::parse_str(&one_map_file("", &between)).expect("exponent between bands");
    assert_eq!(
        ionex.tec_maps()[0],
        vec![
            vec![Some(scaled(10, -1)), Some(scaled(20, -1))],
            vec![Some(scaled(300, -2)), Some(scaled(400, -2))]
        ]
    );
}

#[test]
fn ionex_exponent_a_map_leaves_changed_carries_into_the_next() {
    let changed = exponent_record(-2) + &layout_bands("  100  200", "  300  400");
    let unstated = layout_bands("  100  200", "  300  400");
    // IONEX 1: "Each value remains valid until changed by an additional header
    // record", so map 2 reads at the -2 map 1 set, and is reported for it.
    let text = layout_header(2, LAYOUT_EPOCH_1, "")
        + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &changed)
        + &layout_map("TEC", 2, Some(LAYOUT_EPOCH_1), &unstated)
        + &layout_end();
    let (inherited, warnings) = Ionex::parse_str_with_warnings(&text).expect("carried exponent");
    assert_eq!(inherited.tec_maps()[1][0][0], Some(scaled(100, -2)));
    assert_eq!(
        warnings,
        vec![IonexWarning::ExponentCarriedIntoMap {
            kind: "TEC",
            map_number: 2,
            line: 26,
            exponent: -2,
            set_by_line: 16,
        }]
    );
    // The map is named once, however many bands it carries.
    assert_eq!(
        warnings
            .iter()
            .filter(|w| matches!(w, IonexWarning::ExponentCarriedIntoMap { .. }))
            .count(),
        1
    );

    let restated = exponent_record(-1) + &unstated;
    let text = layout_header(2, LAYOUT_EPOCH_1, "")
        + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &changed)
        + &layout_map("TEC", 2, Some(LAYOUT_EPOCH_1), &restated)
        + &layout_end();
    let ionex = Ionex::parse_str(&text).expect("restated exponent");
    assert_eq!(ionex.tec_maps()[1][0][0], Some(scaled(100, -1)));

    let set_between_maps = layout_header(2, LAYOUT_EPOCH_1, "")
        + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &changed)
        + &exponent_record(-2)
        + &layout_map("TEC", 2, Some(LAYOUT_EPOCH_1), &unstated)
        + &layout_end();
    let ionex = Ionex::parse_str(&set_between_maps).expect("exponent between maps");
    assert_eq!(ionex.tec_maps()[1][0][0], Some(scaled(100, -2)));
}

#[test]
fn ionex_values_are_read_in_their_i5_columns() {
    let exponent = exponent_record(0);
    let merged = Ionex::parse_str(&one_map_file(
        &exponent,
        &layout_bands("1000010000", " 9999   -5"),
    ))
    .expect("adjacent five-digit values");
    assert_eq!(
        merged.tec_maps()[0],
        vec![vec![Some(10000.0), Some(10000.0)], vec![None, Some(-5.0)]]
    );

    let whitespace = Ionex::parse_str(&one_map_file(&exponent, &layout_bands("10 11", "12 13")))
        .expect("values not in columns");
    assert_eq!(whitespace.tec_maps()[0][1], vec![Some(12.0), Some(13.0)]);

    let empty = parse_error(&one_map_file(
        &exponent,
        &layout_bands("    1         2", "3 4"),
    ));
    assert!(empty.contains("empty field"), "{empty}");

    let letters = parse_error(&one_map_file(&exponent, &layout_bands("    1    x", "3 4")));
    assert!(letters.contains("not integer values"), "{letters}");

    let short = parse_error(&one_map_file(
        &exponent,
        &layout_bands("    1", "    3    4"),
    ));
    assert!(
        short.contains("latitude band 1 has 1 values, expected 2"),
        "{short}"
    );

    let long = parse_error(&one_map_file(
        &exponent,
        &layout_bands("    1    2    3", "3 4"),
    ));
    assert!(long.contains("more than 2 values"), "{long}");

    let stray = one_map_file(&exponent, &(layout_bands("1 2", "3 4") + "    5    6\n"));
    let stray = parse_error(&stray);
    assert!(
        stray.contains("outside a LAT/LON1/LON2/DLON/H block"),
        "{stray}"
    );
}

#[test]
fn ionex_nan_value_reads_as_non_available_with_a_warning() {
    // uqrg0010.24i gives one RMS value as `  nan`.
    let text = one_map_file(
        &exponent_record(0),
        &layout_bands("    1  nan", "    3    4"),
    );
    let (ionex, warnings) = Ionex::parse_str_with_warnings(&text).expect("nan field");
    assert_eq!(
        ionex.tec_maps()[0],
        vec![vec![Some(1.0), None], vec![Some(3.0), Some(4.0)]]
    );
    assert_eq!(
        warnings,
        vec![IonexWarning::NotANumberValue {
            kind: "TEC",
            map_number: 1,
            line: 20,
            lat_deg: 1.0,
            lon_deg: 1.0,
        }]
    );
}

#[test]
fn ionex_map_numbers_and_related_maps_are_checked() {
    let exponent = exponent_record(0);
    let bands = layout_bands("    1    2", "    3    4");
    let two_maps = |second: usize, end: usize| {
        let mut text = layout_header(2, LAYOUT_EPOCH_1, &exponent)
            + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &bands);
        text.push_str(&ionex_record(&format!("{second:6}"), "START OF TEC MAP"));
        text.push_str(&ionex_record(LAYOUT_EPOCH_1, "EPOCH OF CURRENT MAP"));
        text.push_str(&bands);
        text.push_str(&ionex_record(&format!("{end:6}"), "END OF TEC MAP"));
        text + &layout_end()
    };
    let skipped = parse_error(&two_maps(3, 3));
    assert!(
        skipped.contains("numbered 3, but TEC map 2 comes next"),
        "{skipped}"
    );
    let end = parse_error(&two_maps(2, 1));
    assert!(
        end.contains("does not give the number of TEC map 2"),
        "{end}"
    );
    Ionex::parse_str(&two_maps(2, 2)).expect("numbered in sequence");

    let with_related = |related: &str| {
        layout_header(1, LAYOUT_EPOCH_0, &exponent)
            + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &bands)
            + related
            + &layout_end()
    };
    let epoch = parse_error(&with_related(&layout_map(
        "RMS",
        1,
        Some(LAYOUT_EPOCH_1),
        &bands,
    )));
    assert!(
        epoch.contains("RMS map 1 gives epoch 2020-01-01 01:00:00"),
        "{epoch}"
    );
    let orphan = parse_error(&with_related(&layout_map("RMS", 2, None, &bands)));
    assert!(orphan.contains("RMS map 2 at line"), "{orphan}");
    let twice = parse_error(&with_related(
        &(layout_map("HEIGHT", 1, None, &bands) + &layout_map("HEIGHT", 1, None, &bands)),
    ));
    assert!(twice.contains("HEIGHT map 1 appears at lines"), "{twice}");

    let rms_for_second_only = layout_header(2, LAYOUT_EPOCH_1, &exponent)
        + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &bands)
        + &layout_map("TEC", 2, Some(LAYOUT_EPOCH_1), &bands)
        + &layout_map("RMS", 2, Some(LAYOUT_EPOCH_1), &bands)
        + &layout_end();
    let ionex = Ionex::parse_str(&rms_for_second_only).expect("RMS for one map");
    assert_eq!(
        ionex.rms_maps()[0],
        vec![vec![None, None], vec![None, None]]
    );
    assert_eq!(ionex.rms_maps()[1], ionex.tec_maps()[1]);

    let all_non_available = layout_bands(" 9999 9999", " 9999 9999");
    let ionex = Ionex::parse_str(&with_related(&layout_map(
        "RMS",
        1,
        None,
        &all_non_available,
    )))
    .expect("RMS map without values");
    // The file states an RMS map for TEC map 1 and gives every value as
    // non-available. That is not the same as stating no RMS map, so the stack
    // stays, holding no value.
    assert_eq!(
        ionex.rms_maps(),
        &[vec![vec![None, None], vec![None, None]]]
    );
}

#[test]
fn ionex_height_maps_are_kept_apart_from_the_tec_maps() {
    let exponent = exponent_record(0);
    let text = layout_header(1, LAYOUT_EPOCH_0, &exponent)
        + &layout_map(
            "TEC",
            1,
            Some(LAYOUT_EPOCH_0),
            &layout_bands("    1    2", "    3    4"),
        )
        + &layout_map("HEIGHT", 1, None, &layout_bands("    7    8", " 9999    6"))
        + &layout_end();
    let ionex = Ionex::parse_str(&text).expect("height map");
    assert_eq!(
        ionex.tec_maps()[0],
        vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0), Some(4.0)]]
    );
    assert_eq!(
        ionex.height_maps()[0],
        vec![vec![Some(7.0), Some(8.0)], vec![None, Some(6.0)]]
    );
    assert!(ionex.rms_maps().is_empty());

    let without_values = layout_header(1, LAYOUT_EPOCH_0, &exponent)
        + &layout_map(
            "TEC",
            1,
            Some(LAYOUT_EPOCH_0),
            &layout_bands("    1    2", "    3    4"),
        )
        + &layout_map("HEIGHT", 1, None, &layout_bands(" 9999 9999", " 9999 9999"))
        + &layout_end();
    let ionex = Ionex::parse_str(&without_values).expect("height map without values");
    assert_eq!(
        ionex.height_maps(),
        &[vec![vec![None, None], vec![None, None]]]
    );
}

/// Two files whose TEC maps, header records, axes and epochs are the same but
/// which differ in whether they state RMS maps at all stay apart: the one
/// giving every RMS value as non-available declares the maps, the other
/// declares none. Both give no RMS number anywhere; only the first says the
/// maps exist, and only the first writes RMS map records back.
#[test]
fn ionex_keeps_a_parsed_rms_stack_that_states_no_value() {
    let exponent = exponent_record(0);
    let tec = layout_map(
        "TEC",
        1,
        Some(LAYOUT_EPOCH_0),
        &layout_bands("    1    2", "    3    4"),
    );
    let without_rms = layout_header(1, LAYOUT_EPOCH_0, &exponent) + &tec + &layout_end();
    let with_valueless_rms = layout_header(1, LAYOUT_EPOCH_0, &exponent)
        + &tec
        + &layout_map(
            "RMS",
            1,
            Some(LAYOUT_EPOCH_0),
            &layout_bands(" 9999 9999", " 9999 9999"),
        )
        + &layout_end();

    let absent = Ionex::parse_str(&without_rms).expect("product stating no RMS map");
    let present = Ionex::parse_str(&with_valueless_rms).expect("product stating an RMS map");

    assert!(
        absent.rms_maps().is_empty(),
        "a file with no RMS map record has no RMS map"
    );
    assert_eq!(
        present.rms_maps(),
        &[vec![vec![None, None], vec![None, None]]],
        "a stated RMS map survives every value being non-available"
    );
    assert_ne!(absent, present, "the two products are not the same product");
    assert_eq!(
        absent.tec_maps(),
        present.tec_maps(),
        "the RMS stack changes no TEC value"
    );
    assert_eq!(absent.map_epochs(), present.map_epochs());
    assert_eq!(absent.lat_nodes_deg(), present.lat_nodes_deg());
    assert_eq!(absent.lon_nodes_deg(), present.lon_nodes_deg());

    // Whole-grid samples carry each product's own RMS authority, so rebuilding
    // from them gives the same product back.
    for (ionex, label) in [(&absent, "absent"), (&present, "stated")] {
        let samples = ionex.tec_grid_samples();
        assert_eq!(
            samples.rms_maps,
            ionex.rms_maps().to_vec(),
            "{label} RMS stack reaches the samples"
        );
        let rebuilt = Ionex::from_samples(samples).expect("sample-built copy");
        assert_eq!(
            rebuilt, *ionex,
            "{label} RMS stack survives the sample rebuild"
        );
    }

    // Writing states the difference, and reading the written text back keeps it.
    let absent_text = absent.to_ionex_string().expect("writable product");
    let present_text = present.to_ionex_string().expect("writable product");
    assert!(
        !absent_text.contains("START OF RMS MAP"),
        "no RMS map is written for a product stating none:\n{absent_text}"
    );
    assert!(
        present_text.contains("START OF RMS MAP"),
        "the stated RMS map is written:\n{present_text}"
    );
    assert_eq!(
        Ionex::parse_str(&absent_text)
            .expect("written text reparses")
            .rms_maps(),
        absent.rms_maps()
    );
    let reparsed = Ionex::parse_str(&present_text).expect("written text reparses");
    assert_eq!(reparsed.rms_maps(), present.rms_maps());
    assert!(
        reparsed
            .rms_maps()
            .iter()
            .flatten()
            .flatten()
            .all(Option::is_none),
        "the stated stack still gives no RMS number at any node"
    );
}

/// An RMS stack where one map states values and the other states none keeps
/// both maps at their own index, and the values it does state are unchanged.
#[test]
fn ionex_rms_stack_keeps_each_map_at_its_tec_map_index() {
    let exponent = exponent_record(0);
    let text = layout_header(2, LAYOUT_EPOCH_1, &exponent)
        + &layout_map(
            "TEC",
            1,
            Some(LAYOUT_EPOCH_0),
            &layout_bands("    1    2", "    3    4"),
        )
        + &layout_map(
            "TEC",
            2,
            Some(LAYOUT_EPOCH_1),
            &layout_bands("    5    6", "    7    8"),
        )
        + &layout_map(
            "RMS",
            1,
            Some(LAYOUT_EPOCH_0),
            &layout_bands(" 9999 9999", " 9999 9999"),
        )
        + &layout_map(
            "RMS",
            2,
            Some(LAYOUT_EPOCH_1),
            &layout_bands("    0    9", " 9999   11"),
        )
        + &layout_end();

    let ionex = Ionex::parse_str(&text).expect("two TEC maps with one valued RMS map");
    assert_eq!(ionex.rms_maps().len(), 2, "one RMS map per TEC map");
    assert_eq!(
        ionex.rms_maps()[0],
        vec![vec![None, None], vec![None, None]],
        "the first RMS map states no value"
    );
    assert_eq!(
        ionex.rms_maps()[1],
        vec![vec![Some(0.0), Some(9.0)], vec![None, Some(11.0)]],
        "the second RMS map states its values, a stated zero among them"
    );
    assert_eq!(
        ionex.rms_maps()[1][0][0].map(f64::to_bits),
        Some(0.0_f64.to_bits()),
        "a stated zero RMS is a value, not a missing node"
    );
    assert_eq!(
        ionex.tec_maps()[1],
        vec![vec![Some(5.0), Some(6.0)], vec![Some(7.0), Some(8.0)]],
        "the RMS stack changes no TEC value"
    );

    // Node samples see the stack by map index; the second map's values are
    // there and the first map's nodes are empty.
    let samples = ionex.tec_samples();
    assert_eq!(samples.len(), 8, "one sample per node of two maps");
    assert!(
        samples[..4].iter().all(|sample| sample.rms_tecu.is_none()),
        "the first map states no RMS value"
    );
    assert_eq!(
        samples[4..]
            .iter()
            .map(|sample| sample.rms_tecu)
            .collect::<Vec<_>>(),
        vec![Some(0.0), Some(9.0), None, Some(11.0)]
    );

    let rebuilt = Ionex::from_samples(ionex.tec_grid_samples()).expect("sample-built copy");
    assert_eq!(rebuilt, ionex, "the mixed stack survives a sample rebuild");
    let text = ionex.to_ionex_string().expect("writable product");
    assert_eq!(
        Ionex::parse_str(&text).expect("written text reparses"),
        ionex,
        "the mixed stack survives writing and reading back"
    );
}

/// Whole-grid construction carries the same distinction as a parsed file: an
/// empty `rms_maps` stack states no RMS map, and a stack whose every node is
/// `None` states the maps with no value in them. Flat node samples have no
/// field for that, which this pins as the input model's limit.
#[test]
fn ionex_from_samples_keeps_an_rms_stack_that_states_no_value() {
    let mut absent_samples = valid_tec_grid_samples();
    absent_samples.rms_maps = Vec::new();
    let mut present_samples = valid_tec_grid_samples();
    present_samples.rms_maps = vec![vec![vec![None, None], vec![None, None]]];

    let absent = Ionex::from_samples(absent_samples).expect("product stating no RMS map");
    let present = Ionex::from_samples(present_samples).expect("product stating an RMS map");

    assert!(absent.rms_maps().is_empty());
    assert_eq!(
        present.rms_maps(),
        &[vec![vec![None, None], vec![None, None]]],
        "a stated RMS stack with no value at any node is kept"
    );
    assert_ne!(absent, present);
    assert_eq!(absent.tec_maps(), present.tec_maps());
    assert_eq!(
        present.tec_grid_samples().rms_maps,
        vec![vec![vec![None, None], vec![None, None]]],
        "the stack reaches the samples the product is rebuilt from"
    );

    let absent_text = absent.to_ionex_string().expect("writable product");
    let present_text = present.to_ionex_string().expect("writable product");
    assert!(!absent_text.contains("START OF RMS MAP"), "{absent_text}");
    assert!(present_text.contains("START OF RMS MAP"), "{present_text}");
    assert_eq!(
        Ionex::parse_str(&present_text)
            .expect("written text reparses")
            .rms_maps(),
        present.rms_maps(),
        "the written RMS map reads back as a stated stack with no value"
    );

    // A flat node stream has no map-presence field: with no RMS value anywhere
    // it can only build the product that states no RMS map.
    let from_nodes = Ionex::from_node_samples(
        present.tec_samples(),
        present.shell_height_km(),
        present.base_radius_km(),
        present.exponent(),
        present.header().clone(),
    )
    .expect("node-built product");
    assert!(
        from_nodes.rms_maps().is_empty(),
        "flat node samples cannot state an RMS stack with no value at any node"
    );
    assert_eq!(from_nodes.tec_maps(), present.tec_maps());
}

#[test]
fn ionex_summary_records_are_checked_against_the_maps() {
    let exponent = exponent_record(0);
    let bands = layout_bands("    1    2", "    3    4");
    let two_maps = |header: String| {
        header
            + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &bands)
            + &layout_map("TEC", 2, Some(LAYOUT_EPOCH_1), &bands)
            + &layout_end()
    };
    let clean = two_maps(layout_header(2, LAYOUT_EPOCH_1, &exponent));
    let (_, warnings) = Ionex::parse_str_with_warnings(&clean).expect("clean");
    assert!(warnings.is_empty(), "{warnings:?}");

    let wrong_last = two_maps(layout_header(2, LAYOUT_EPOCH_0, &exponent));
    let (_, warnings) = Ionex::parse_str_with_warnings(&wrong_last).expect("wrong last epoch");
    assert!(
        matches!(
            warnings.as_slice(),
            [IonexWarning::EpochMismatch {
                label: "EPOCH OF LAST MAP",
                line: 4,
                ..
            }]
        ),
        "{warnings:?}"
    );

    let wrong_count = two_maps(layout_header(3, LAYOUT_EPOCH_1, &exponent));
    let (_, warnings) = Ionex::parse_str_with_warnings(&wrong_count).expect("wrong count");
    assert_eq!(
        warnings,
        vec![IonexWarning::MapCountMismatch {
            line: 6,
            declared: 3,
            tec_maps: 2,
            all_maps: 2
        }]
    );
    let counting_rms = layout_header(2, LAYOUT_EPOCH_0, &exponent)
        + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &bands)
        + &layout_map("RMS", 1, None, &bands)
        + &layout_end();
    let (_, warnings) = Ionex::parse_str_with_warnings(&counting_rms).expect("TEC and RMS count");
    assert!(warnings.is_empty(), "{warnings:?}");

    let wrong_interval =
        two_maps(layout_header(2, LAYOUT_EPOCH_1, &exponent).replace("  3600", "  1800"));
    let (_, warnings) = Ionex::parse_str_with_warnings(&wrong_interval).expect("interval");
    assert_eq!(
        warnings,
        vec![IonexWarning::IntervalMismatch {
            line: 5,
            declared_s: 1800,
            map_number: 2,
            spacing_s: 3600
        }]
    );
    let variable =
        two_maps(layout_header(2, LAYOUT_EPOCH_1, &exponent).replace("  3600", "     0"));
    let (ionex, warnings) = Ionex::parse_str_with_warnings(&variable).expect("variable");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(ionex.header().interval_s, 0);

    let decimals = two_maps(
        layout_header(2, LAYOUT_EPOCH_1, &exponent)
            .replace("  3600", "  3600.0")
            .replace(LAYOUT_EPOCH_1, "  2020     1     1     1     0  0.00"),
    );
    let (ionex, warnings) = Ionex::parse_str_with_warnings(&decimals).expect("whole decimals");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(ionex.header().interval_s, 3600);

    let unreadable =
        two_maps(layout_header(2, LAYOUT_EPOCH_1, &exponent).replace("  3600", "  36.5"));
    let (ionex, warnings) = Ionex::parse_str_with_warnings(&unreadable).expect("unreadable");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(ionex.skipped_records(), 1);

    let no_end = clean.replace(&layout_end(), "");
    let (_, warnings) = Ionex::parse_str_with_warnings(&no_end).expect("no END OF FILE");
    assert_eq!(warnings, vec![IonexWarning::MissingRecord("END OF FILE")]);

    let comment_first = ionex_record("a comment first", "COMMENT") + &clean;
    let (_, warnings) = Ionex::parse_str_with_warnings(&comment_first).expect("comment first");
    assert_eq!(
        warnings,
        vec![IonexWarning::VersionRecordNotFirst { line: 2 }]
    );

    let mut bare = String::new();
    bare.push_str(&ionex_record("  6371.0", "BASE RADIUS"));
    bare.push_str(&ionex_record("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"));
    bare.push_str(&ionex_record(LAYOUT_LAT, "LAT1 / LAT2 / DLAT"));
    bare.push_str(&ionex_record(LAYOUT_LON, "LON1 / LON2 / DLON"));
    bare.push_str(&exponent);
    bare.push_str(&ionex_record("", "END OF HEADER"));
    bare.push_str(&layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &bands));
    let (ionex, warnings) = Ionex::parse_str_with_warnings(&bare).expect("bare header");
    assert_eq!(
        warnings,
        [
            "IONEX VERSION / TYPE",
            "PGM / RUN BY / DATE",
            "EPOCH OF FIRST MAP",
            "EPOCH OF LAST MAP",
            "INTERVAL",
            "# OF MAPS IN FILE",
            "MAPPING FUNCTION",
            "ELEVATION CUTOFF",
            "OBSERVABLES USED",
            "MAP DIMENSION",
            "END OF FILE",
        ]
        .map(IonexWarning::MissingRecord)
        .to_vec()
    );
    assert_eq!(ionex.header().mapping_function, None);
    assert_eq!(ionex.header().elevation_cutoff_deg, 0.0);
}

#[test]
fn ionex_header_records_that_disagree_or_do_not_describe_ionosphere_maps_are_refused() {
    let exponent = exponent_record(0);
    let bands = layout_bands("    1    2", "    3    4");
    let twice = one_map_file(
        &(exponent.clone() + &ionex_record("     1.0   0.0  -0.5", "LAT1 / LAT2 / DLAT")),
        &bands,
    );
    let message = parse_error(&twice);
    assert!(
        message.contains("LAT1 / LAT2 / DLAT at lines 13 and 16 with different values"),
        "{message}"
    );
    let same_twice = one_map_file(
        &(exponent.clone() + &ionex_record(LAYOUT_LAT, "LAT1 / LAT2 / DLAT")),
        &bands,
    );
    Ionex::parse_str(&same_twice).expect("identical duplicate");

    let other_type = one_map_file(&exponent, &bands).replacen(
        "     1.0            IONOSPHERE MAPS",
        "     1.0            OBSERVATION DATA",
        1,
    );
    let message = parse_error(&other_type);
    assert!(message.contains("file type 'O'"), "{message}");

    let unclosed = one_map_file(&(exponent + &ionex_record("", "START OF AUX DATA")), &bands);
    let message = parse_error(&unclosed);
    assert!(
        message.contains("not closed before END OF HEADER"),
        "{message}"
    );
}

#[test]
fn ionex_sample_round_trips_keep_non_available_nodes() {
    let parsed = Ionex::parse_str(&fixture_text("spec_example_2d.inx")).expect("example");
    let rebuilt = Ionex::from_samples(parsed.tec_grid_samples()).expect("grid samples");
    assert_eq!(rebuilt, parsed);
    let from_nodes = Ionex::from_node_samples(
        parsed.tec_samples(),
        parsed.shell_height_km(),
        parsed.base_radius_km(),
        parsed.exponent(),
        parsed.header().clone(),
    )
    .expect("node samples");
    assert_eq!(from_nodes, parsed);
}

// ---------------------------------------------------------------------------
// Slant delay over non-available nodes.
// ---------------------------------------------------------------------------

fn zenith_product(ionex: &Ionex) -> Ionex {
    let mut samples = ionex.tec_grid_samples();
    // A zero base radius puts a zenith pierce point on the receiver coordinate.
    samples.base_radius_km = 0.0;
    samples.height_maps.clear();
    Ionex::from_samples(samples).expect("zenith product")
}

fn zenith_delay(
    ionex: &Ionex,
    lat_deg: f64,
    lon_deg: f64,
    epoch_j2000_s: i64,
) -> crate::Result<f64> {
    let request = coverage_request(lat_deg, lon_deg, epoch_j2000_s);
    super::ionex_slant_delay(
        ionex,
        request.receiver,
        request.elevation_rad,
        request.azimuth_rad,
        request.epoch_j2000_s,
        request.frequency_hz,
    )
}

#[test]
fn ionex_slant_delay_refuses_a_weighted_non_available_node() {
    let ionex =
        zenith_product(&Ionex::parse_str(&fixture_text("spec_example_2d.inx")).expect("example"));
    let epochs = ionex.map_epochs_s();

    // TEC map 1 gives latitude 80 longitude 5 as 9999: the cell from
    // [0][0] to [1][1] weights it.
    let err = zenith_delay(&ionex, 82.5, 2.5, epochs[0]).expect_err("weighted node");
    assert_eq!(
        err,
        crate::error::Error::IonexNodesNotAvailable(Box::new(IonexNodeGap {
            earlier: Some(IonexMissingNodes {
                map_number: 1,
                lat_index: 0,
                lon_index: 0,
                lon_index_next: 1,
                missing: [false, false, false, true],
            }),
            later: None,
        }))
    );
    assert_eq!(
        err.to_string(),
        "IONEX nodes not available: map 1 cell [0][0] missing [1][1]"
    );

    // Between the maps both carry weight; TEC map 2 holds every node of the cell.
    let err = zenith_delay(&ionex, 82.5, 2.5, (epochs[0] + epochs[1]) / 2).expect_err("between");
    assert!(
        matches!(
            err,
            crate::error::Error::IonexNodesNotAvailable(ref gap)
                if gap.earlier.is_some() && gap.later.is_none()
        ),
        "{err:?}"
    );

    // A node without weight is not used: on the available node at latitude
    // 80 longitude 0, and at the epoch of the second map.
    zenith_delay(&ionex, 80.0, 0.0, epochs[0]).expect("query on an available node");
    zenith_delay(&ionex, 82.5, 2.5, epochs[1]).expect("epoch of the other map");

    let requests = [
        coverage_request(82.5, 2.5, epochs[0]),
        coverage_request(82.5, 12.5, epochs[0]),
    ];
    let results = ionex_slant_delay_results(&ionex, &requests, IonexCoveragePolicy::Hold.into());
    assert!(matches!(
        results[0],
        Err(crate::error::Error::IonexNodesNotAvailable(_))
    ));
    assert!(results[1].is_ok(), "{:?}", results[1]);
}

#[test]
fn ionex_axes_read_in_either_direction() {
    // IONEX 1 gives an axis as "'LAT1' to 'LAT2' with increment 'DLAT'", which
    // says nothing about the direction. This file runs its latitudes south to
    // north and its longitudes east to west.
    let mut text = String::new();
    text.push_str(&ionex_record(
        "     1.0            IONOSPHERE MAPS     GPS",
        "IONEX VERSION / TYPE",
    ));
    text.push_str(&ionex_record("     0.0   1.0   1.0", "LAT1 / LAT2 / DLAT"));
    text.push_str(&ionex_record("     1.0   0.0  -1.0", "LON1 / LON2 / DLON"));
    text.push_str(&ionex_record("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"));
    text.push_str(&ionex_record("  6371.0", "BASE RADIUS"));
    text.push_str(&ionex_record("     0", "EXPONENT"));
    text.push_str(&ionex_record("  COSZ", "MAPPING FUNCTION"));
    text.push_str(&ionex_record("     2", "MAP DIMENSION"));
    text.push_str(&ionex_record("", "END OF HEADER"));
    text.push_str(&ionex_record("     1", "START OF TEC MAP"));
    text.push_str(&ionex_record(LAYOUT_EPOCH_0, "EPOCH OF CURRENT MAP"));
    text.push_str(&layout_band(0.0, 1.0, 0.0, -1.0, 450.0, "   11   10"));
    text.push_str(&layout_band(1.0, 1.0, 0.0, -1.0, 450.0, "   13   12"));
    text.push_str(&ionex_record("     1", "END OF TEC MAP"));
    text.push_str(&layout_end());

    let ionex = Ionex::parse_str(&text).expect("axes in either direction");
    assert_eq!(ionex.lat_nodes_deg(), &[0.0, 1.0]);
    assert_eq!(ionex.lon_nodes_deg(), &[1.0, 0.0]);
    assert_eq!(ionex.dlat_deg(), 1.0);
    assert_eq!(ionex.dlon_deg(), -1.0);
    // Each band is placed by its own latitude and longitudes.
    assert_eq!(
        ionex.tec_maps()[0],
        vec![vec![Some(11.0), Some(10.0)], vec![Some(13.0), Some(12.0)]]
    );

    // A step whose sign contradicts its bounds is refused.
    let wrong = text.replace("     0.0   1.0   1.0", "     0.0   1.0  -1.0");
    assert!(
        parse_error(&wrong).contains("IONEX"),
        "a contradicting step is refused"
    );
}

#[test]
fn ionex_data_record_outside_ascii_is_refused_by_name() {
    // A value field is an ASCII number in its columns. A record holding any
    // other character cannot be cut into them, and splitting it on whitespace
    // would place its values at the wrong nodes where a field is also blank.
    let body = layout_bands("    1    2", "    3    \u{e9}");
    let message = parse_error(&one_map_file(&exponent_record(0), &body));
    assert!(message.contains("outside ASCII"), "{message}");
}

#[test]
fn ionex_exponent_record_between_maps_sets_the_unit_of_the_maps_after_it() {
    // A record between maps states the unit at file level, which is what the
    // carry-over refusal asks a map to state, so the map after it carries no
    // EXPONENT record of its own and nothing is inherited silently.
    let bands = layout_bands("    1    2", "    3    4");
    let text = layout_header(2, LAYOUT_EPOCH_1, &exponent_record(0))
        + &layout_map("TEC", 1, Some(LAYOUT_EPOCH_0), &bands)
        + &exponent_record(-1)
        + &layout_map("TEC", 2, Some(LAYOUT_EPOCH_1), &bands)
        + &layout_end();
    let (ionex, warnings) = Ionex::parse_str_with_warnings(&text).expect("exponent between maps");
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(
        ionex.tec_maps()[0],
        vec![vec![Some(1.0), Some(2.0)], vec![Some(3.0), Some(4.0)]]
    );
    assert_eq!(
        ionex.tec_maps()[1],
        vec![
            vec![Some(scaled(1, -1)), Some(scaled(2, -1))],
            vec![Some(scaled(3, -1)), Some(scaled(4, -1))]
        ]
    );
    // `EXPONENT` stays the header's own value.
    assert_eq!(ionex.exponent(), 0);
}

// ---------------------------------------------------------------------------
// Slant-delay policies: renormalizing fallback, mapping function, height maps.
// ---------------------------------------------------------------------------

fn renormalizing() -> IonexSlantPolicy {
    IonexSlantPolicy::default().with_missing_nodes(IonexMissingNodePolicy::Renormalize)
}

fn delay_with(
    ionex: &Ionex,
    request: IonexSlantRequest,
    policy: IonexSlantPolicy,
) -> crate::Result<super::IonexSlantDelayEvaluation> {
    ionex_slant_delay_with_policy(
        ionex,
        request.receiver,
        request.elevation_rad,
        request.azimuth_rad,
        request.epoch_j2000_s,
        request.frequency_hz,
        policy,
    )
}

/// Meters of L1 group delay per TECU of slant TEC.
fn l1_delay_per_tecu() -> f64 {
    super::slant::K_IONO / (1_575_420_000.0 * 1_575_420_000.0)
}

fn assert_relative(got: f64, want: f64) {
    assert!(
        (got - want).abs() <= 1.0e-9 * want.abs(),
        "got {got}, want {want}"
    );
}

#[test]
fn ionex_renormalize_interpolates_around_missing_nodes_and_marks_the_value_degraded() {
    let ionex =
        zenith_product(&Ionex::parse_str(&fixture_text("spec_example_2d.inx")).expect("example"));
    let epochs = ionex.map_epochs_s();
    let gap = IonexNodeGap {
        earlier: Some(IonexMissingNodes {
            map_number: 1,
            lat_index: 0,
            lon_index: 0,
            lon_index_next: 1,
            missing: [false, false, false, true],
        }),
        later: None,
    };

    let evaluation = delay_with(
        &ionex,
        coverage_request(82.5, 2.5, epochs[0]),
        renormalizing(),
    )
    .expect("renormalized value");
    assert_eq!(
        evaluation.status,
        IonexSlantDelayStatus {
            held: None,
            degraded: Some(gap),
            assumed_mapping: None,
        }
    );
    assert!(!evaluation.status.is_valid());
    // The three nodes left carry equal weight: latitude 85 longitudes 0 and 5,
    // and latitude 80 longitude 0. A zenith pierce point maps with factor 1.
    let vtec = (scaled(1000, -1) + scaled(1001, -1) + scaled(1100, -1)) / 3.0;
    assert_relative(evaluation.delay_m, l1_delay_per_tecu() * vtec);

    // A cell with every node present gives the strict value, not degraded.
    let request = coverage_request(82.5, 12.5, epochs[0]);
    let renormalized = delay_with(&ionex, request, renormalizing()).expect("full cell");
    let strict = delay_with(&ionex, request, IonexSlantPolicy::default()).expect("strict");
    assert_eq!(renormalized.status, IonexSlantDelayStatus::VALID);
    assert_eq!(renormalized.delay_m.to_bits(), strict.delay_m.to_bits());
}

#[test]
fn ionex_renormalize_blends_only_the_maps_that_give_a_value() {
    let mut samples = valid_tec_grid_samples();
    samples.base_radius_km = 0.0;
    samples.map_epochs = vec![
        super::ionex_epoch_from_j2000_seconds(0),
        super::ionex_epoch_from_j2000_seconds(10),
    ];
    samples.tec_maps = vec![
        vec![vec![None, None], vec![None, None]],
        vec![vec![Some(10.0), Some(11.0)], vec![Some(12.0), Some(13.0)]],
    ];
    let ionex = Ionex::from_samples(samples).expect("one map without values");
    let gap = IonexNodeGap {
        earlier: Some(IonexMissingNodes {
            map_number: 1,
            lat_index: 0,
            lon_index: 0,
            lon_index_next: 1,
            missing: [true; 4],
        }),
        later: None,
    };

    let between = coverage_request(0.5, 0.5, 5);
    assert_eq!(
        delay_with(&ionex, between, IonexSlantPolicy::default()).expect_err("strict"),
        crate::error::Error::IonexNodesNotAvailable(Box::new(gap))
    );
    let evaluation = delay_with(&ionex, between, renormalizing()).expect("later map");
    assert_eq!(evaluation.status.degraded, Some(gap));
    assert_relative(evaluation.delay_m, l1_delay_per_tecu() * 11.5);

    // At the epoch of the map without values only that map carries weight.
    assert_eq!(
        delay_with(&ionex, coverage_request(0.5, 0.5, 0), renormalizing()).expect_err("no value"),
        crate::error::Error::IonexNodesNotAvailable(Box::new(gap))
    );
}

#[test]
fn ionex_real_non_available_node_is_refused_or_renormalized() {
    let ionex = Ionex::parse_str(&fixture_text("EMR0OPSFIN_20240010000_01D_01H_GIM_trim.INX"))
        .expect("EMR trim");
    // TEC map 2 gives latitude 10 longitude 145 as 9999.
    let request = coverage_request(9.0, 147.0, ionex.map_epochs_s()[1]);
    let gap = IonexNodeGap {
        earlier: Some(IonexMissingNodes {
            map_number: 2,
            lat_index: 0,
            lon_index: 65,
            lon_index_next: 66,
            missing: [true, false, false, false],
        }),
        later: None,
    };
    assert_eq!(
        delay_with(&ionex, request, IonexSlantPolicy::default()).expect_err("strict"),
        crate::error::Error::IonexNodesNotAvailable(Box::new(gap))
    );
    let evaluation = delay_with(&ionex, request, renormalizing()).expect("renormalized");
    assert_eq!(evaluation.status.degraded, Some(gap));
    assert!(evaluation.delay_m.is_finite() && evaluation.delay_m > 0.0);
}

#[test]
fn ionex_slant_delay_maps_single_layer_by_default_and_names_what_the_product_declares() {
    let request_at = |ionex: &Ionex| coverage_request(0.0, 0.0, ionex.map_epochs_s()[0]);
    let declared_only = IonexSlantPolicy::default().with_mapping(IonexMappingPolicy::Declared);
    for (name, assumed, declared) in [
        (
            "COD0OPSFIN_20240010000_01D_01H_GIM_trim.INX",
            IonexAssumedMapping::NoMapping,
            IonexMappingDeclaration::Declared(IonexMappingFunction::NoMapping),
        ),
        (
            "EMR0OPSFIN_20240010000_01D_01H_GIM_trim.INX",
            IonexAssumedMapping::Other,
            IonexMappingDeclaration::Declared(IonexMappingFunction::Other("MOD".into())),
        ),
        (
            "esa_2024176_first_map_2row.inx",
            IonexAssumedMapping::Absent,
            IonexMappingDeclaration::Absent,
        ),
    ] {
        let ionex = Ionex::parse_str(&fixture_text(name)).expect(name);
        let request = request_at(&ionex);

        // The default maps these with 1/cos(z') and names what they declare.
        let chosen = delay_with(&ionex, request, IonexSlantPolicy::default()).expect(name);
        assert_eq!(chosen.status.assumed_mapping, Some(assumed), "{name}");
        // Nothing was held and nothing was degraded, so the value is the one
        // the grid gives at the request. The assumed mapping is its own field,
        // for a caller that will not take a single-layer factor on a product
        // determined some other way.
        assert!(chosen.status.is_valid(), "{name}");

        assert_eq!(
            delay_with(&ionex, request, declared_only).expect_err(name),
            crate::error::Error::IonexSlantUnavailable(IonexSlantRefusal::MappingFunction(
                declared
            )),
            "{name}"
        );

        let mut samples = ionex.tec_grid_samples();
        samples.header.mapping_function = Some(IonexMappingFunction::CosZ);
        let cosz = Ionex::from_samples(samples).expect("COSZ copy");
        let declared_cosz = delay_with(&cosz, request, declared_only).expect(name);
        assert_eq!(
            chosen.delay_m.to_bits(),
            declared_cosz.delay_m.to_bits(),
            "{name}: the default applies the COSZ factor"
        );
        assert_eq!(declared_cosz.status, IonexSlantDelayStatus::VALID, "{name}");
    }
    let message = crate::error::Error::IonexSlantUnavailable(IonexSlantRefusal::MappingFunction(
        IonexMappingDeclaration::Declared(IonexMappingFunction::NoMapping),
    ))
    .to_string();
    assert!(message.contains("MAPPING FUNCTION NONE"), "{message}");

    // A product declaring COSZ is mapped with the factor it declares, so its
    // status names none.
    let igs = Ionex::parse_str(&fixture_text("IGS0OPSFIN_20240010000_01D_02H_GIM_trim.INX"))
        .expect("IGS trim");
    let cosz = delay_with(&igs, request_at(&igs), IonexSlantPolicy::default()).expect("COSZ trim");
    assert_eq!(cosz.status, IonexSlantDelayStatus::VALID);

    // QFAC names a factor the spec gives no formula for, and reads as NONE does.
    let mut samples = valid_tec_grid_samples();
    samples.header.mapping_function = Some(IonexMappingFunction::QFactor);
    let qfac = Ionex::from_samples(samples).expect("QFAC product");
    let request = coverage_request(0.5, 0.5, 0);
    let mut out = [0.0];
    ionex_slant_delays(&qfac, &[request], &mut out).expect("QFAC batch");
    assert!(out[0].is_finite() && out[0] > 0.0);
    assert_eq!(
        delay_with(&qfac, request, declared_only).expect_err("QFAC declared"),
        crate::error::Error::IonexSlantUnavailable(IonexSlantRefusal::MappingFunction(
            IonexMappingDeclaration::Declared(IonexMappingFunction::QFactor)
        ))
    );

    // Batch results carry the flag element by element.
    let results = ionex_slant_delay_results(&qfac, &[request], IonexSlantPolicy::default());
    assert_eq!(
        results[0]
            .as_ref()
            .expect("QFAC element")
            .status
            .assumed_mapping,
        Some(IonexAssumedMapping::QFactor)
    );
}

#[test]
fn ionex_slant_delay_uses_the_one_height_height_maps_give_and_refuses_others() {
    let parsed = Ionex::parse_str(&fixture_text("spec_example_2d.inx")).expect("example");
    let request = IonexSlantRequest::new(
        crate::frame::Wgs84Geodetic::new(80.0_f64.to_radians(), 180.0_f64.to_radians(), 0.0)
            .expect("receiver"),
        45.0_f64.to_radians(),
        90.0_f64.to_radians(),
        parsed.map_epochs_s()[0],
        1_575_420_000.0,
    );
    let with_heights = |heights: Vec<Vec<Vec<Option<f64>>>>, shell_height_km: f64| {
        let mut samples = parsed.tec_grid_samples();
        samples.height_maps = heights;
        samples.shell_height_km = shell_height_km;
        Ionex::from_samples(samples).expect("product")
    };
    let uniform = |height: f64| vec![vec![vec![Some(height); 72]; 3]; 2];
    let refusal = |map_number, lat_index, lon_index, varying: bool| {
        crate::error::Error::IonexSlantUnavailable(if varying {
            IonexSlantRefusal::VaryingHeights {
                map_number,
                lat_index,
                lon_index,
            }
        } else {
            IonexSlantRefusal::HeightNotAvailable {
                map_number,
                lat_index,
                lon_index,
            }
        })
    };

    for mapping in [
        IonexMappingPolicy::Declared,
        IonexMappingPolicy::SingleLayer,
    ] {
        let policy = IonexSlantPolicy::default().with_mapping(mapping);
        let at_400 = delay_with(&with_heights(Vec::new(), 400.0), request, policy).expect("400 km");
        let at_450 = delay_with(&with_heights(Vec::new(), 450.0), request, policy).expect("450 km");
        assert_ne!(at_400.delay_m.to_bits(), at_450.delay_m.to_bits());

        // The spec's example 1 gives every height as 0, which leaves the shell at
        // HGT1.
        let zero = delay_with(&with_heights(uniform(0.0), 400.0), request, policy)
            .expect("every height 0");
        assert_eq!(zero.delay_m.to_bits(), at_400.delay_m.to_bits());
        // Every height 50 km puts the shell at HGT1 + 50 km.
        let fifty = delay_with(&with_heights(uniform(50.0), 400.0), request, policy)
            .expect("every height 50");
        assert_eq!(fifty.delay_m.to_bits(), at_450.delay_m.to_bits());

        // The fixture's heights vary from the second node on.
        assert_eq!(
            delay_with(&parsed, request, policy).expect_err("varying heights"),
            refusal(1, 0, 1, true)
        );
        let mut one_missing = uniform(0.0);
        one_missing[1][2][3] = None;
        assert_eq!(
            delay_with(&with_heights(one_missing, 400.0), request, policy)
                .expect_err("a non-available height"),
            refusal(2, 2, 3, false)
        );
        assert_eq!(
            delay_with(
                &with_heights(vec![vec![vec![None; 72]; 3]; 2], 400.0),
                request,
                policy
            )
            .expect_err("no height available"),
            refusal(1, 0, 0, false)
        );
    }
}

#[test]
fn regular_tec_grid_refuses_or_renormalizes_missing_nodes() {
    use super::tec_grid::{TecGrid, TecGridError};

    let mut values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(Some).to_vec();
    values[3] = None;
    let grid = TecGrid::new(vec![0.0, 10.0], vec![0.0, 10.0], vec![20.0, 30.0], values)
        .expect("grid with one missing node");
    let gap = IonexNodeGap {
        earlier: Some(IonexMissingNodes {
            map_number: 1,
            lat_index: 0,
            lon_index: 0,
            lon_index_next: 1,
            missing: [false, false, false, true],
        }),
        later: None,
    };
    assert_eq!(
        grid.interpolate_vtec(5.0, 5.0, 25.0).expect_err("strict"),
        TecGridError::NodesNotAvailable(gap)
    );
    let renormalized = grid
        .interpolate_vtec_with_policy(5.0, 5.0, 25.0, IonexMissingNodePolicy::Renormalize)
        .expect("renormalized");
    // Epoch 0 gives (1 + 2 + 3) / 3 = 2 and epoch 1 gives 6.5, blended evenly.
    assert_eq!(renormalized.value.to_bits(), 4.25f64.to_bits());
    assert_eq!(renormalized.degraded, Some(gap));
    assert_eq!(
        grid.interpolate_vtec(0.0, 0.0, 20.0)
            .expect("a query on a node uses no neighbour")
            .to_bits(),
        1.0f64.to_bits()
    );

    let mut values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(Some).to_vec();
    values[..4].fill(None);
    let grid = TecGrid::new(vec![0.0, 10.0], vec![0.0, 10.0], vec![20.0, 30.0], values)
        .expect("grid with an epoch without values");
    let later_only = grid
        .interpolate_vtec_with_policy(5.0, 5.0, 25.0, IonexMissingNodePolicy::Renormalize)
        .expect("later epoch");
    assert_eq!(later_only.value.to_bits(), 6.5f64.to_bits());
    assert!(matches!(
        grid.interpolate_vtec_with_policy(0.0, 5.0, 25.0, IonexMissingNodePolicy::Renormalize),
        Err(TecGridError::NodesNotAvailable(_))
    ));
}

#[test]
fn ionex_longitude_seam_interpolates_between_the_last_node_and_the_first() {
    // IONEX 1's example 1 runs its longitudes 0 to 355 by 5, which closes the
    // circle without naming the seam twice, so 357.5 lies in the cell between
    // the node at 355 and the node at 0.
    let ionex =
        zenith_product(&Ionex::parse_str(&fixture_text("spec_example_2d.inx")).expect("example"));
    let epochs = ionex.map_epochs_s();
    // Map 1 holds 1000 + 100 * lat_index + lon_index at EXPONENT -1.
    let node = |lon_index: i64| scaled(1000 + lon_index, -1);

    // A receiver longitude is given in [-180, 180]; -2.5 is the 357.5 of the
    // grid, which the query normalizes onto the seam cell.
    for (lon_deg, p) in [(-2.5, 0.5), (-0.1, (359.9 - 355.0) / 5.0)] {
        let got = zenith_delay(&ionex, 85.0, lon_deg, epochs[0]).expect("the seam is covered");
        let vtec = (1.0 - p) * node(71) + p * node(0);
        assert_relative(got, l1_delay_per_tecu() * vtec);
    }

    // A node of the grid still reads as that node: -5 is its 355.
    assert_relative(
        zenith_delay(&ionex, 85.0, -5.0, epochs[0]).expect("last node"),
        l1_delay_per_tecu() * node(71),
    );
}

#[test]
fn ionex_seam_cell_reads_the_same_from_any_turn() {
    // The cell that closes the circle runs past the turn, and a receiver
    // longitude arrives in [-180, 180], any number of turns from the node that
    // opens that cell. The offset is taken in whole turns of the axis, so one
    // point reads the same however it is named.
    let lat_arr = [1.0_f64, 0.0];
    let map = vec![
        vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)],
        vec![Some(50.0), Some(60.0), Some(70.0), Some(80.0)],
    ];
    // Latitude 1 is the first node row, so the value is the longitude
    // interpolation alone.
    let at = |lon_arr: &[f64], dlon: f64, lam: f64| {
        super::slant::bilinear_vtec(&map, &lat_arr, lon_arr, -1.0, dlon, 1.0, lam)
    };

    // West to east, where the cell from the node at 270 runs to 360, which is
    // the node at 0. Every name below is 315.
    let west_to_east = [0.0_f64, 90.0, 180.0, 270.0];
    for lam in [315.0_f64, -45.0, -405.0, 675.0] {
        let got = at(&west_to_east, 90.0, lam);
        assert_eq!(got.lon_index, 3, "{lam}");
        assert_eq!(got.p, 0.5, "{lam}");
        assert_eq!(got.vtec, Some(0.5 * 40.0 + 0.5 * 10.0), "{lam}");
    }
    // A query at the node that closes the cell reads that node, not a turn
    // past it. Every name below is the node at 0.
    for lam in [0.0_f64, 360.0, -360.0, 720.0] {
        let got = at(&west_to_east, 90.0, lam);
        assert_eq!(got.p, 0.0, "{lam}");
        assert_eq!(got.vtec, Some(10.0), "{lam}");
    }

    // East to west, where a turn is the same count of cells the other way.
    // Its node at 270 is index 0 and its node at 0 is index 3.
    let east_to_west = [270.0_f64, 180.0, 90.0, 0.0];
    for lam in [315.0_f64, -45.0, -405.0, 675.0] {
        let got = at(&east_to_west, -90.0, lam);
        assert_eq!(got.lon_index, 3, "{lam}");
        assert_eq!(got.p, 0.5, "{lam}");
        assert_eq!(got.vtec, Some(0.5 * 40.0 + 0.5 * 10.0), "{lam}");
    }
    for lam in [270.0_f64, -90.0, -450.0, 630.0] {
        let got = at(&east_to_west, -90.0, lam);
        assert_eq!(got.p, 0.0, "{lam}");
        assert_eq!(got.vtec, Some(10.0), "{lam}");
    }
}

#[test]
fn ionex_seam_cell_names_the_column_that_closes_the_circle() {
    // The example's longitudes run 0 to 355 by 5, so the cell that closes the
    // circle opens at column 71 and reaches the node at column 0. The grid has
    // no column 72, and a caller who indexes the axes with what a refusal
    // reports must land on a node the product has.
    let parsed = Ionex::parse_str(&fixture_text("spec_example_2d.inx")).expect("example");
    let mut samples = parsed.tec_grid_samples();
    // A zero base radius puts a zenith pierce point on the receiver coordinate.
    samples.base_radius_km = 0.0;
    samples.height_maps.clear();
    // The node the seam cell reaches over the turn.
    samples.tec_maps[0][0][0] = None;
    let ionex = Ionex::from_samples(samples).expect("a non-available node across the seam");
    let epochs = ionex.map_epochs_s();

    // Latitude 85 is the first node row, so only the two nodes of that row
    // carry weight, and the one across the seam is non-available.
    let err = zenith_delay(&ionex, 85.0, -2.5, epochs[0]).expect_err("the seam node is weighted");
    let nodes = IonexMissingNodes {
        map_number: 1,
        lat_index: 0,
        lon_index: 71,
        lon_index_next: 0,
        missing: [false, true, false, false],
    };
    assert_eq!(
        err,
        crate::error::Error::IonexNodesNotAvailable(Box::new(IonexNodeGap {
            earlier: Some(nodes),
            later: None,
        }))
    );
    // The message names column 0, not the column 72 the product has not got.
    assert_eq!(
        err.to_string(),
        "IONEX nodes not available: map 1 cell [0][71] missing [0][0]"
    );

    // Indexing the axes with what the refusal reports lands on the node the
    // product gives as non-available.
    assert_eq!(ionex.lon_nodes_deg().len(), 72);
    assert_eq!(ionex.lat_nodes_deg()[nodes.lat_index], 85.0);
    assert_eq!(ionex.lon_nodes_deg()[nodes.lon_index], 355.0);
    assert_eq!(ionex.lon_nodes_deg()[nodes.lon_index_next], 0.0);
    assert_eq!(
        ionex.tec_maps()[0][nodes.lat_index][nodes.lon_index_next],
        None
    );
}

#[test]
fn ionex_slant_delay_reads_an_ascending_product_as_a_descending_one() {
    // One grid holds 10 at latitude 0 longitude 0, 11 at 0 and 1, 12 at 1 and
    // 0, and 13 at 1 and 1. Written south to north with its longitudes east to
    // west, and north to south with them west to east, it is that same grid,
    // so a pierce point in the middle of its one cell reads the same.
    let product = |lat: &str, lon: &str, bands: String| {
        let mut text = String::new();
        text.push_str(&ionex_record(
            "     1.0            IONOSPHERE MAPS     GPS",
            "IONEX VERSION / TYPE",
        ));
        text.push_str(&ionex_record(lat, "LAT1 / LAT2 / DLAT"));
        text.push_str(&ionex_record(lon, "LON1 / LON2 / DLON"));
        text.push_str(&ionex_record("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"));
        text.push_str(&ionex_record("  6371.0", "BASE RADIUS"));
        text.push_str(&ionex_record("     0", "EXPONENT"));
        text.push_str(&ionex_record("  COSZ", "MAPPING FUNCTION"));
        text.push_str(&ionex_record("     2", "MAP DIMENSION"));
        text.push_str(&ionex_record("", "END OF HEADER"));
        text.push_str(&ionex_record("     1", "START OF TEC MAP"));
        text.push_str(&ionex_record(LAYOUT_EPOCH_0, "EPOCH OF CURRENT MAP"));
        text.push_str(&bands);
        text.push_str(&ionex_record("     1", "END OF TEC MAP"));
        text.push_str(&layout_end());
        zenith_product(&Ionex::parse_str(&text).expect("product"))
    };

    let ascending = product(
        "     0.0   1.0   1.0",
        "     1.0   0.0  -1.0",
        layout_band(0.0, 1.0, 0.0, -1.0, 450.0, "   11   10")
            + &layout_band(1.0, 1.0, 0.0, -1.0, 450.0, "   13   12"),
    );
    let descending = product(
        "     1.0   0.0  -1.0",
        "     0.0   1.0   1.0",
        layout_band(1.0, 0.0, 1.0, 1.0, 450.0, "   12   13")
            + &layout_band(0.0, 0.0, 1.0, 1.0, 450.0, "   10   11"),
    );
    assert_eq!(ascending.lat_nodes_deg(), &[0.0, 1.0]);
    assert_eq!(descending.lat_nodes_deg(), &[1.0, 0.0]);

    // The centre of the cell weighs the four nodes equally.
    let epoch = ascending.map_epochs_s()[0];
    assert_eq!(descending.map_epochs_s()[0], epoch);
    let up = zenith_delay(&ascending, 0.5, 0.5, epoch).expect("ascending");
    let down = zenith_delay(&descending, 0.5, 0.5, epoch).expect("descending");
    assert_relative(up, l1_delay_per_tecu() * 11.5);
    assert_eq!(up.to_bits(), down.to_bits());
}

#[test]
fn ionex_pierce_point_through_a_pole_gives_a_value() {
    // The quotient that names the pierce-point longitude divides by the cosine
    // of its latitude, which is zero at a pole, and the spherical-trig
    // quotients can round past 1, where asin gives NaN.
    for az_deg in [0.0_f64, 45.0, 90.0, 179.0, 180.0, 270.0, 359.0] {
        for el_deg in [1.0_f64, 5.0, 30.0, 89.0, 90.0] {
            for lat_deg in [89.999_999_f64, 90.0, -90.0] {
                let geom = super::slant::pierce_point(
                    lat_deg.to_radians(),
                    0.0,
                    az_deg.to_radians(),
                    el_deg.to_radians(),
                    6371.0,
                    450.0,
                );
                assert!(
                    geom.phi_ipp_deg.is_finite() && geom.lambda_ipp_deg.is_finite(),
                    "lat {lat_deg} az {az_deg} el {el_deg}: {geom:?}"
                );
            }
        }
    }

    // Straight up from a pole the pierce point stays on the pole, and takes the
    // receiver's meridian. The quotient the other branch forms divides a sine
    // of about -5.7e-17 by a cosine of about 6.1e-17, which gives about -0.93
    // and an arcsine 69 degrees round the pole, so a meridian read back
    // unchanged is this branch's own answer and not the general one.
    for lat_deg in [90.0_f64, -90.0] {
        for lon_deg in [0.0_f64, 40.0, -125.0] {
            for az_deg in [0.0_f64, 90.0, 180.0, 270.0] {
                let geom = super::slant::pierce_point(
                    lat_deg.to_radians(),
                    lon_deg.to_radians(),
                    az_deg.to_radians(),
                    core::f64::consts::FRAC_PI_2,
                    6371.0,
                    450.0,
                );
                let at = format!("lat {lat_deg} lon {lon_deg} az {az_deg}");
                assert!(
                    (geom.phi_ipp_deg - lat_deg).abs() <= 1.0e-9,
                    "{at}: {geom:?}"
                );
                assert!(
                    (geom.lambda_ipp_deg - lon_deg).abs() <= 1.0e-9,
                    "{at}: {geom:?}"
                );
            }
        }
    }

    // A line of sight straight up from the pole reaches the grid through the
    // hold policy, which carries the latitude to the grid edge. The example's
    // height maps give nodes different heights, which the slant delay refuses
    // whatever the geometry, so this product keeps only its TEC maps.
    let parsed = Ionex::parse_str(&fixture_text("spec_example_2d.inx")).expect("example");
    let mut samples = parsed.tec_grid_samples();
    samples.height_maps.clear();
    let ionex = Ionex::from_samples(samples).expect("product without height maps");
    let evaluation = ionex_slant_delay_with_policy(
        &ionex,
        crate::frame::Wgs84Geodetic::new(core::f64::consts::FRAC_PI_2, 0.0, 0.0).expect("pole"),
        core::f64::consts::FRAC_PI_2,
        0.0,
        ionex.map_epochs_s()[0],
        1_575_420_000.0,
        IonexSlantPolicy::from(IonexCoveragePolicy::Hold),
    )
    .expect("a value at the pole");
    assert!(
        evaluation.delay_m.is_finite() && evaluation.delay_m > 0.0,
        "{evaluation:?}"
    );
}

// ---------------------------------------------------------------------------
// Writer: IONEX 1 columns, exact values, refusals.
// ---------------------------------------------------------------------------

fn written(ionex: &Ionex) -> String {
    ionex.to_ionex_string().expect("writable product")
}

#[test]
fn ionex_writer_keeps_the_axis_direction_it_read() {
    // The axis records carry the direction the nodes run in, so a product whose
    // latitudes run south to north and whose longitudes run east to west writes
    // back the axes it holds.
    let mut samples = valid_tec_grid_samples();
    samples.lat_nodes_deg = vec![0.0, 1.0];
    samples.dlat_deg = 1.0;
    samples.lon_nodes_deg = vec![1.0, 0.0];
    samples.dlon_deg = -1.0;
    let ionex = Ionex::from_samples(samples).expect("axes in either direction");
    let text = written(&ionex);
    assert!(text.contains("     0.0   1.0   1.0"), "{text}");
    assert!(text.contains("     1.0   0.0  -1.0"), "{text}");
    assert_rereads_as(&ionex, &text);
}

#[test]
fn ionex_writer_states_a_value_no_field_reads_back() {
    // No field states 0.7 for the reader: 7 * 1e-1 through 70000 * 1e-5, every
    // one that fits I5, give 0x1.6666666666667p-1, while 0.7 is
    // 0x1.6666666666666p-1, and 700000 leaves the five columns. The file states
    // it as the decimal the field names, and the value read back is the product
    // that field gives, one unit in the last place away.
    let mut samples = valid_tec_grid_samples();
    samples.tec_maps = vec![vec![vec![Some(0.7), Some(0.7)], vec![Some(0.7), Some(0.7)]]];
    let ionex = Ionex::from_samples(samples).expect("0.7 product");
    let text = written(&ionex);
    assert_rereads_within_one_ulp(&ionex, &text);

    let reparsed = Ionex::parse_str(&text).expect("written 0.7 reparses");
    let got = reparsed.tec_maps()[0][0][0].expect("node holds a value");
    assert_ne!(
        got.to_bits(),
        0.7_f64.to_bits(),
        "no field reads back as 0.7, so the round trip is not exact here"
    );
    assert_eq!(ulp_distance(got, 0.7), 1, "read back as {got}");
}

#[test]
fn ionex_writer_writes_a_product_built_from_decimal_values() {
    // A product built from samples holds decimals that no whole number of
    // tenths reaches. Hundredths hold a field the reader takes back for three
    // of these, and state the fourth: 0.7 is written as the decimal 70, which
    // reads back one unit in the last place away. One unit writes them all, so
    // the file carries that exponent in its header and no record inside a map.
    let mut samples = valid_tec_grid_samples();
    samples.tec_maps = vec![vec![
        vec![Some(0.7), Some(12.3)],
        vec![Some(-0.25), Some(100.0)],
    ]];
    let ionex = Ionex::from_samples(samples).expect("decimal product");
    let out = written(&ionex);
    assert_rereads_within_one_ulp(&ionex, &out);

    let reparsed = Ionex::parse_str(&out).expect("written decimals reparse");
    // Hundredths hold a field the reader takes back for three of these, so they
    // read back as themselves. 0.7 has no such field at any exponent, so the
    // file states the decimal 70 and the reader gives back the product of it.
    for (want, got) in [
        (12.3_f64, reparsed.tec_maps()[0][0][1]),
        (-0.25, reparsed.tec_maps()[0][1][0]),
        (100.0, reparsed.tec_maps()[0][1][1]),
    ] {
        let got = got.expect("node holds a value");
        assert_eq!(got.to_bits(), want.to_bits(), "{want} read back as {got}");
    }
    let seven = reparsed.tec_maps()[0][0][0].expect("node holds a value");
    assert_eq!(ulp_distance(seven, 0.7), 1, "0.7 read back as {seven}");
    // Hundredths state every one of them, so that is the unit chosen.
    assert_eq!(reparsed.exponent(), -2);
}

/// Reparse `text` and compare it with `original` rebuilt from its samples at the
/// exponent the text declares, and check the writer reproduces the text.
fn assert_rereads_as(original: &Ionex, text: &str) {
    let reparsed = Ionex::parse_str(text).expect("written text reparses");
    let mut expected = original.tec_grid_samples();
    expected.exponent = reparsed.exponent();
    // A product built from samples states no `# OF MAPS IN FILE`; the file the
    // writer wrote states one, so the product read back from it carries that
    // record.
    expected.header.maps_in_file = reparsed.header().maps_in_file;
    assert_eq!(
        reparsed,
        Ionex::from_samples(expected).expect("samples rebuild")
    );
    assert_eq!(written(&reparsed), text, "the writer is deterministic");
}

/// As [`assert_rereads_as`], for a product built from samples rather than read
/// from a file: every node within one unit in the last place.
///
/// The writer states a value with the field the file can hold it in, and the
/// reader forms the product the reference readers form, so a value that never
/// came from a field can come back one unit away. A product parsed from text
/// has no such gap.
fn assert_rereads_within_one_ulp(original: &Ionex, text: &str) {
    let reparsed = Ionex::parse_str(text).expect("written text reparses");
    assert_eq!(reparsed.map_epochs_s(), original.map_epochs_s());
    for (maps, rebuilt) in [
        (original.tec_maps(), reparsed.tec_maps()),
        (original.rms_maps(), reparsed.rms_maps()),
        (original.height_maps(), reparsed.height_maps()),
    ] {
        assert_eq!(maps.len(), rebuilt.len(), "map count");
        for (map, remap) in maps.iter().zip(rebuilt) {
            for (row, rerow) in map.iter().zip(remap) {
                for (node, renode) in row.iter().zip(rerow) {
                    match (node, renode) {
                        (Some(want), Some(got)) => assert!(
                            ulp_distance(*want, *got) <= 1,
                            "node {want} read back as {got}"
                        ),
                        (None, None) => {}
                        _ => panic!("node availability changed: {node:?} became {renode:?}"),
                    }
                }
            }
        }
    }
    assert_eq!(written(&reparsed), text, "the writer is deterministic");
}

fn data_records(text: &str) -> impl Iterator<Item = &str> {
    text.lines().filter(|line| {
        !line.is_empty()
            && line
                .bytes()
                .all(|byte| byte == b' ' || byte == b'-' || byte.is_ascii_digit())
    })
}

#[test]
fn ionex_writer_lays_out_every_record_as_ionex_1_defines() {
    let original = synthetic_ionex();
    let text = written(&original);
    assert!(text.lines().all(|line| line.len() <= 80), "{text}");

    let header: Vec<&str> = text
        .lines()
        .take_while(|line| !line.ends_with("END OF HEADER"))
        .map(|line| line[60..].trim())
        .collect();
    assert_eq!(
        header,
        [
            "IONEX VERSION / TYPE",
            "PGM / RUN BY / DATE",
            "EPOCH OF FIRST MAP",
            "EPOCH OF LAST MAP",
            "INTERVAL",
            "# OF MAPS IN FILE",
            "MAPPING FUNCTION",
            "ELEVATION CUTOFF",
            "OBSERVABLES USED",
            "BASE RADIUS",
            "MAP DIMENSION",
            "HGT1 / HGT2 / DHGT",
            "LAT1 / LAT2 / DLAT",
            "LON1 / LON2 / DLON",
        ]
    );
    for (data, label) in [
        (
            "     1.1            IONOSPHERE MAPS",
            "IONEX VERSION / TYPE",
        ),
        ("  2020     6    24     0     0     0", "EPOCH OF FIRST MAP"),
        ("  2020     6    24     2     0     0", "EPOCH OF LAST MAP"),
        ("  COSZ", "MAPPING FUNCTION"),
        ("  6371.0", "BASE RADIUS"),
        ("   450.0 450.0   0.0", "HGT1 / HGT2 / DHGT"),
        ("    60.0 -60.0 -20.0", "LAT1 / LAT2 / DLAT"),
        ("  -180.0 180.0  60.0", "LON1 / LON2 / DLON"),
        ("    60.0-180.0 180.0  60.0 450.0", "LAT/LON1/LON2/DLON/H"),
        ("   -60.0-180.0 180.0  60.0 450.0", "LAT/LON1/LON2/DLON/H"),
    ] {
        assert!(
            text.contains(&format!("{data:<60}{label}\n")),
            "{label} {data:?}:\n{text}"
        );
    }
    assert!(
        text.contains("\n  100  101  102  103  104  105  106\n"),
        "{text}"
    );
    assert!(
        !text.contains("EXPONENT"),
        "the default exponent is not written"
    );
    assert!(text.ends_with(&format!("{:60}END OF FILE\n", "")));
    assert!(data_records(&text).all(|line| line.len() % 5 == 0));
    assert_rereads_as(&original, &text);
}

#[test]
fn ionex_writer_writes_non_available_values_as_9999_and_rereads_exactly() {
    let mut products = Vec::new();
    for name in [
        "spec_example_2d.inx",
        "COD0OPSFIN_20240010000_01D_01H_GIM_trim.INX",
        "EMR0OPSFIN_20240010000_01D_01H_GIM_trim.INX",
        "IGS0OPSFIN_20240010000_01D_02H_GIM_trim.INX",
        "uqrg0010.24i_trim.INX",
    ] {
        products.push((name, Ionex::parse_str(&fixture_text(name)).expect(name)));
    }
    for (name, original) in products {
        let text = written(&original);
        assert_rereads_as(&original, &text);

        let non_available = [
            original.tec_maps(),
            original.rms_maps(),
            original.height_maps(),
        ]
        .iter()
        .flat_map(|maps| maps.iter().flatten().flatten())
        .filter(|value| value.is_none())
        .count();
        let written_9999 = data_records(&text)
            .flat_map(|line| line.as_bytes().chunks(5))
            .filter(|field| *field == b" 9999")
            .count();
        assert_eq!(written_9999, non_available, "{name}");
        assert!(
            data_records(&text).all(|line| line.len() <= 80 && line.len() % 5 == 0),
            "{name}: values in 16I5"
        );
    }
}

#[test]
fn ionex_writer_chooses_an_exponent_that_writes_every_value_exactly() {
    // Hundredths of a TECU that tenths cannot write.
    let mut samples = valid_tec_grid_samples();
    samples.exponent = -1;
    samples.tec_maps = vec![vec![
        vec![Some(scaled(123, -2)), Some(scaled(450, -2))],
        vec![Some(scaled(-5, -2)), Some(scaled(99999, -2))],
    ]];
    let original = Ionex::from_samples(samples).expect("hundredths");
    let text = written(&original);
    assert!(
        text.contains(&format!("{:<60}EXPONENT\n", "    -2")),
        "{text}"
    );
    assert_rereads_as(&original, &text);

    // 9999 units marks a non-available value, so 9999 TECU is written in tenths.
    let mut samples = valid_tec_grid_samples();
    samples.tec_maps = vec![vec![
        vec![Some(9999.0), Some(1.0)],
        vec![Some(2.0), Some(3.0)],
    ]];
    let original = Ionex::from_samples(samples).expect("9999 TECU");
    let text = written(&original);
    // Tenths are the default unit, so no EXPONENT record is written.
    assert!(!text.contains("EXPONENT"), "{text}");
    assert_rereads_as(&original, &text);
    let reparsed = Ionex::parse_str(&text).expect("reparse");
    assert_eq!(reparsed.exponent(), -1);
    assert_eq!(reparsed.tec_maps()[0][0][0], Some(9999.0));

    // No one exponent writes 1.23 and 1234500 TECU, so the band at latitude 0
    // splits into two band records, each after its own EXPONENT record, and the
    // second map restates the header exponent the first map left changed.
    let mut samples = valid_tec_grid_samples();
    samples.exponent = 0;
    samples.map_epochs = vec![
        super::ionex_epoch_from_j2000_seconds(0),
        super::ionex_epoch_from_j2000_seconds(10),
    ];
    samples.tec_maps = vec![
        vec![
            vec![Some(12.0), Some(13.0)],
            vec![Some(scaled(123, -2)), Some(scaled(12345, 2))],
        ],
        vec![vec![Some(10.0), Some(11.0)], vec![Some(12.0), Some(13.0)]],
    ];
    let original = Ionex::from_samples(samples).expect("mixed units");
    let text = written(&original);
    let records: Vec<&str> = text
        .lines()
        .filter(|line| line.ends_with("EXPONENT") || line.ends_with("LAT/LON1/LON2/DLON/H"))
        .map(|line| line[..60].trim_end())
        .collect();
    assert_eq!(
        records,
        [
            "     0",
            "     1.0   0.0   1.0   1.0 450.0",
            "    -2",
            "     0.0   0.0   0.0   1.0 450.0",
            "     2",
            "     0.0   1.0   1.0   1.0 450.0",
            "     0",
            "     1.0   0.0   1.0   1.0 450.0",
            "     0.0   0.0   1.0   1.0 450.0",
        ],
        "{text}"
    );
    let reparsed = Ionex::parse_str(&text).expect("per-block exponents read back");
    let mut expected = original.tec_grid_samples();
    // The written file states its map count, which the product built from
    // samples did not.
    expected.header.maps_in_file = reparsed.header().maps_in_file;
    assert_eq!(
        reparsed,
        Ionex::from_samples(expected).expect("samples rebuild"),
        "the header keeps the product's exponent"
    );
    assert_eq!(written(&reparsed), text);

    // IONEX 1 example 1, as the fixture gives it, mixes units across its bands.
    let spec = Ionex::parse_str(&fixture_text("spec_example_2d.inx")).expect("example");
    let text = written(&spec);
    assert!(
        text.lines()
            .skip_while(|line| !line.ends_with("END OF HEADER"))
            .any(|line| line.ends_with("EXPONENT")),
        "{text}"
    );
    assert_eq!(Ionex::parse_str(&text).expect("reparse"), spec);

    // No exponent writes 12345.6 TECU within I5.
    let mut samples = valid_tec_grid_samples();
    samples.exponent = -1;
    samples.tec_maps = vec![vec![
        vec![Some(scaled(123_456, -1)), Some(1.0)],
        vec![Some(2.0), Some(3.0)],
    ]];
    let message = Ionex::from_samples(samples)
        .expect("wide value")
        .to_ionex_string()
        .expect_err("no exponent writes it")
        .to_string();
    assert!(
        message.contains("TEC map 1 value 12345.6")
            && message.contains("latitude 1 longitude 0")
            && message.contains("within I5"),
        "{message}"
    );
}

/// A change to a header and the record label its refusal names.
type HeaderEdit = (fn(&mut IonexHeader), &'static str);

#[test]
fn ionex_writer_refuses_fields_its_columns_cannot_hold() {
    let mut samples = valid_tec_grid_samples();
    samples.lat_nodes_deg = vec![0.5, 0.25];
    samples.lon_nodes_deg = vec![0.0, 0.25];
    samples.dlat_deg = -0.25;
    samples.dlon_deg = 0.25;
    let message = Ionex::from_samples(samples)
        .expect("quarter-degree grid")
        .to_ionex_string()
        .expect_err("F6.1 axis")
        .to_string();
    assert!(
        message.contains("LAT1 / LAT2 / DLAT nodes 0.5 to 0.25 by -0.25"),
        "{message}"
    );

    let edits: [HeaderEdit; 6] = [
        (
            |header| header.program = "a program name over twenty".into(),
            "PGM / RUN BY / DATE",
        ),
        (
            |header| header.descriptions.push("a trailing blank ".into()),
            "DESCRIPTION",
        ),
        (
            |header| header.observables_used = "carrier\tphase".into(),
            "OBSERVABLES USED",
        ),
        (
            |header| header.elevation_cutoff_deg = 10.25,
            "ELEVATION CUTOFF",
        ),
        (
            |header| header.mapping_function = Some(IonexMappingFunction::Other("COSZ".into())),
            "MAPPING FUNCTION",
        ),
        (
            |header| header.mapping_function = Some(IonexMappingFunction::Other("MSLM1".into())),
            "MAPPING FUNCTION",
        ),
    ];
    for (edit, label) in edits {
        let mut samples = valid_tec_grid_samples();
        edit(&mut samples.header);
        let message = Ionex::from_samples(samples)
            .expect("product")
            .to_ionex_string()
            .expect_err(label)
            .to_string();
        assert!(message.contains(label), "{label}: {message}");
    }
}

#[test]
fn ionex_writer_keeps_utf8_header_text_in_byte_columns() {
    // uhrg0010.24i gives its contact address in UTF-8, with an accented name.
    let mut samples = valid_tec_grid_samples();
    samples.header.descriptions = vec![
        "Contact address: Manuel Hernández-Pajares".into(),
        format!("{:<58}á", "sixty bytes in fifty-nine characters:"),
    ];
    samples.header.run_by = "Hernández UPC".into();
    let original = Ionex::from_samples(samples).expect("product");
    let text = written(&original);
    let accented: Vec<&str> = text.lines().filter(|line| line.contains('á')).collect();
    assert_eq!(accented.len(), 3, "{text}");
    for line in accented {
        assert!(line.is_char_boundary(60), "{line:?}");
        assert!(
            matches!(&line[60..], "DESCRIPTION" | "PGM / RUN BY / DATE"),
            "{line:?}"
        );
    }
    assert_rereads_as(&original, &text);
}

#[test]
fn ionex_writer_writes_the_spec_values_for_unstated_header_records() {
    let original = Ionex::from_samples(valid_tec_grid_samples()).expect("samples");
    let text = written(&original);
    for (data, label) in [
        (
            "     1.0            IONOSPHERE MAPS",
            "IONEX VERSION / TYPE",
        ),
        ("", "PGM / RUN BY / DATE"),
        ("     0", "INTERVAL"),
        ("     1", "# OF MAPS IN FILE"),
        ("  COSZ", "MAPPING FUNCTION"),
        ("     0.0", "ELEVATION CUTOFF"),
        ("", "OBSERVABLES USED"),
        ("     2", "MAP DIMENSION"),
        ("     0", "EXPONENT"),
        ("", "END OF FILE"),
    ] {
        assert!(
            text.contains(&format!("{data:<60}{label}\n")),
            "{label}:\n{text}"
        );
    }
    assert_rereads_as(&original, &text);
    let (_, warnings) = Ionex::parse_str_with_warnings(&text).expect("reparse");
    assert!(warnings.is_empty(), "{warnings:?}");
}

// ---------------------------------------------------------------------------
// Exact whole-second IONEX epoch conversion
// ---------------------------------------------------------------------------

/// A split-Julian-date instant built from its two parts, bypassing the
/// reader's encoder so a test can state a representation directly.
fn split_epoch(jd_whole: f64, fraction: f64) -> Instant {
    Instant::from_julian_date(
        TimeScale::Utc,
        JulianDateSplit::new(jd_whole, fraction).expect("split within one residual day"),
    )
}

/// The `*.0` noon day boundary the IONEX reader's encoder places a whole
/// J2000 second on.
fn noon_boundary(days: i64) -> f64 {
    crate::constants::J2000_JD + days as f64
}

fn nanos_epoch(nanos: i128) -> Instant {
    Instant {
        scale: TimeScale::Utc,
        repr: InstantRepr::Nanos(nanos),
    }
}

/// The double one representable step above a positive `value`, with the exact
/// difference between the two.
///
/// The two doubles are within a factor of two of each other, so their
/// difference is exact (Sterbenz) and the pair `(stepped, -step)` sums back to
/// `value` with nothing dropped.
fn next_double_up(value: f64) -> (f64, f64) {
    assert!(value > 0.0, "the bit increment walks away from zero");
    let stepped = f64::from_bits(value.to_bits() + 1);
    (stepped, stepped - value)
}

#[test]
fn exact_j2000_second_round_trips_reader_epochs_across_calendar_boundaries() {
    // J2000 noon itself, both civil midnights either side of it, a day and a
    // year rollover, a pre-J2000 date, and the seconds either side of each, so
    // the encoding is exercised on and off every boundary it can land on.
    for base in [
        0_i64,            // 2000-01-01 12:00:00, the J2000 origin
        -43_200,          // 2000-01-01 00:00:00, civil midnight
        43_200,           // 2000-01-02 00:00:00, the next civil midnight
        646_315_200,      // 2020-06-25 00:00:00
        -43_200 - 86_400, // 1999-12-31 00:00:00, before J2000
        -1_041_465_600,   // 12_054 whole days before the J2000 origin
        31_579_200,       // 2001-01-01 00:00:00, a year rollover
    ] {
        for offset in [-1_i64, 0, 1] {
            let seconds = base + offset;
            let epoch = super::ionex_epoch_from_j2000_seconds(seconds);
            assert_eq!(
                super::exact_j2000_second(epoch),
                Some(seconds),
                "reader-encoded epoch {seconds} must read back as itself"
            );
        }
    }
}

#[test]
fn exact_j2000_second_reads_the_civil_midnight_boundary_convention() {
    // `split_julian_date` places a civil date on the `*.5` midnight boundary
    // rather than the reader's `*.0` noon boundary. 2020-06-25 00:00:00 UTC is
    // 646_315_200 J2000 seconds, the value the writer's civil round trip pins.
    let (jd_whole, fraction) = split_julian_date(2020, 6, 25, 0, 0, 0.0);
    assert_eq!(
        super::exact_j2000_second(split_epoch(jd_whole, fraction)),
        Some(646_315_200)
    );

    let (jd_whole, fraction) = split_julian_date(2020, 6, 25, 2, 2, 3.0);
    assert_eq!(
        super::exact_j2000_second(split_epoch(jd_whole, fraction)),
        Some(646_315_200 + 7_323)
    );
}

#[test]
fn exact_j2000_second_reads_any_split_whose_boundary_names_a_whole_second() {
    // The two boundaries the crate's encoders emit are not the only ones a
    // `JulianDateSplit` may carry: its constructor checks that both parts are
    // finite and that the residual is within one day, and says nothing about
    // where the boundary sits. What a split names is the value of its two
    // parts, so every boundary below that is a whole number of seconds from
    // J2000 is read as the second it names, with or without a residual.
    for (offset_days, residual_s, seconds) in [
        (0.25, 0, 21_600_i64), // six hours past the origin
        (-0.25, 0, -21_600),   // and six hours before it
        (0.75, 0, 64_800),
        (-0.75, 0, -64_800),
        (0.125, 0, 10_800),    // three hours
        (1.0 / 128.0, 0, 675), // the finest boundary that names a second
        (-1.0 / 128.0, 0, -675),
        (0.25, 11, 21_611), // a quarter day carrying a residual
        (-0.25, 11, -21_589),
        (12_345.25, 11, 12_345 * 86_400 + 21_600 + 11),
        (-12_345.75, 43_199, -12_345 * 86_400 - 64_800 + 43_199),
    ] {
        let jd_whole = crate::constants::J2000_JD + offset_days;
        let fraction = residual_s as f64 / SECONDS_PER_DAY;
        assert_eq!(
            super::exact_j2000_second(split_epoch(jd_whole, fraction)),
            Some(seconds),
            "({jd_whole}, {residual_s} s) names J2000 + {seconds} s"
        );
    }
}

#[test]
fn exact_j2000_second_reads_split_parts_that_cancel_onto_a_whole_second() {
    // A split names the value of its two exact binary parts, whatever each part
    // looks like alone. A boundary one representable step above the origin with
    // a residual of exactly the negative of that step sums to the origin: it is
    // second 0, and refusing it because the encoder would have written the pair
    // differently would refuse an epoch that sits exactly on the axis.
    let (stepped, step) = next_double_up(crate::constants::J2000_JD);
    assert!(step > 0.0);
    assert_eq!(
        super::exact_j2000_second(split_epoch(stepped, -step)),
        Some(0)
    );

    // The same cancellation 36_525 whole days before the origin, and again with
    // a half-day residual riding on top of it. `0.5 - step` is representable
    // exactly, so the pair still sums to a whole second.
    let earlier = crate::constants::J2000_JD - 36_525.0;
    let (stepped, step) = next_double_up(earlier);
    assert_eq!(
        super::exact_j2000_second(split_epoch(stepped, -step)),
        Some(-36_525 * 86_400)
    );
    assert_eq!(
        super::exact_j2000_second(split_epoch(stepped, 0.5 - step)),
        Some(-36_525 * 86_400 + 43_200)
    );
}

#[test]
fn exact_j2000_second_refuses_a_fractional_split_julian_date() {
    // Either side of the rounding boundary and on it: the rounded conversion
    // took 11.4 s as second 11 and 11.6 s as second 12, moving the epoch. None
    // of these name a whole second, so all are refused.
    for second_of_day in [11.4_f64, 11.5, 11.6, 0.5, 0.25, 86_399.5] {
        let fraction = second_of_day / SECONDS_PER_DAY;
        assert_eq!(
            super::exact_j2000_second(split_epoch(noon_boundary(0), fraction)),
            None,
            "{second_of_day} s past the boundary names no whole second"
        );
    }
    // The whole seconds bracketing those fractions are still read exactly, so
    // the refusal is of the fraction and not of the neighbourhood.
    for second_of_day in [11_i64, 12, 0, 86_399] {
        assert_eq!(
            super::exact_j2000_second(split_epoch(
                noon_boundary(0),
                second_of_day as f64 / SECONDS_PER_DAY,
            )),
            Some(second_of_day)
        );
    }
}

#[test]
fn exact_j2000_second_refuses_a_boundary_finer_than_a_whole_second() {
    // `86_400 = 2^7 * 675`, so a day offset is a whole number of seconds
    // exactly when it is a whole multiple of `2^-7` days. A 256th of a day is
    // 337.5 s: it names half a second, which the epoch axis does not hold, and
    // it is refused on its own and carrying an otherwise valid residual.
    let half_second_boundary = crate::constants::J2000_JD + 1.0 / 256.0;
    assert_eq!(
        super::exact_j2000_second(split_epoch(half_second_boundary, 0.0)),
        None
    );
    assert_eq!(
        super::exact_j2000_second(split_epoch(half_second_boundary, 11.0 / SECONDS_PER_DAY)),
        None
    );
    // One step coarser is 675 s, which the axis does hold.
    assert_eq!(
        super::exact_j2000_second(split_epoch(
            crate::constants::J2000_JD + 1.0 / 128.0,
            11.0 / SECONDS_PER_DAY
        )),
        Some(675 + 11)
    );
}

#[test]
fn exact_j2000_second_refuses_the_doubles_either_side_of_an_encoded_fraction() {
    // `11 / 86_400` is not a dyadic rational, so the reader's J2000 + 11 s
    // epoch has no whole-second value in exact arithmetic and is read as the
    // encoding of second 11. That reading is an equality against the one double
    // the encoder produces: the representable neighbour on either side is a
    // different epoch and is refused, so nothing near a whole second is swept
    // onto it.
    let encoded = 11.0 / SECONDS_PER_DAY;
    assert_eq!(
        super::exact_j2000_second(split_epoch(noon_boundary(0), encoded)),
        Some(11)
    );
    for neighbour in [
        f64::from_bits(encoded.to_bits() + 1),
        f64::from_bits(encoded.to_bits() - 1),
    ] {
        assert_ne!(neighbour, encoded);
        assert_eq!(
            super::exact_j2000_second(split_epoch(noon_boundary(0), neighbour)),
            None,
            "a neighbouring double is a different epoch"
        );
    }
}

#[test]
fn exact_j2000_second_refuses_a_non_finite_split() {
    for (jd_whole, fraction) in [
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (f64::NEG_INFINITY, 0.0),
        (crate::constants::J2000_JD, f64::NAN),
        (crate::constants::J2000_JD, f64::INFINITY),
    ] {
        assert_eq!(
            super::exact_j2000_second(Instant::from_julian_date(
                TimeScale::Utc,
                JulianDateSplit { jd_whole, fraction },
            )),
            None
        );
    }
}

#[test]
fn exact_j2000_second_refuses_a_boundary_beyond_the_whole_second_axis() {
    // `2^47` days from the origin is already more than `i64::MAX` seconds away
    // in either direction, so a boundary at or past it names no second the axis
    // holds however the residual is stated.
    for jd_whole in [
        crate::constants::J2000_JD + 140_737_488_355_328.0,
        crate::constants::J2000_JD - 140_737_488_355_328.0,
        f64::MAX,
        f64::MIN,
    ] {
        for fraction in [0.0, 0.5, -0.5] {
            assert_eq!(
                super::exact_j2000_second(split_epoch(jd_whole, fraction)),
                None,
                "{jd_whole} is past the whole-second axis"
            );
        }
    }
}

#[test]
fn exact_j2000_second_refuses_nanos_that_are_not_whole_seconds() {
    for nanos in [1_i128, -1, 500_000_000, 1_000_000_001, -1_000_000_001] {
        assert_eq!(
            super::exact_j2000_second(nanos_epoch(nanos)),
            None,
            "{nanos} ns is not a whole second"
        );
    }
    for seconds in [0_i64, 1, -1, 86_400] {
        assert_eq!(
            super::exact_j2000_second(nanos_epoch(i128::from(seconds) * 1_000_000_000)),
            Some(seconds)
        );
    }
}

#[test]
fn exact_j2000_second_keeps_nanos_seconds_past_the_f64_integers() {
    // 2^53 and its neighbours: projecting the nanosecond count through `f64`
    // collapsed these onto the same second. Integer division keeps them apart.
    for seconds in [
        9_007_199_254_740_991_i64,
        9_007_199_254_740_992,
        9_007_199_254_740_993,
        -9_007_199_254_740_993,
    ] {
        assert_eq!(
            super::exact_j2000_second(nanos_epoch(i128::from(seconds) * 1_000_000_000)),
            Some(seconds)
        );
    }
    assert_ne!(
        super::exact_j2000_second(nanos_epoch(9_007_199_254_740_993_i128 * 1_000_000_000)),
        super::exact_j2000_second(nanos_epoch(9_007_199_254_740_992_i128 * 1_000_000_000)),
    );
}

#[test]
fn exact_j2000_second_keeps_adjacent_split_epochs_past_the_f64_integers_distinct() {
    // The same neighbourhood through the reader's split encoding: the day count
    // and the residual second are each exact, so adjacent seconds stay
    // adjacent where the rounded projection merged them.
    let mut previous = None;
    for seconds in [
        9_007_199_254_740_992_i64,
        9_007_199_254_740_993,
        9_007_199_254_740_994,
    ] {
        let read = super::exact_j2000_second(super::ionex_epoch_from_j2000_seconds(seconds));
        assert_eq!(read, Some(seconds));
        assert_ne!(read, previous, "adjacent large epochs stay distinct");
        previous = read;
    }
}

#[test]
fn exact_j2000_second_converts_the_i64_bounds_through_both_representations() {
    for seconds in [i64::MIN, i64::MIN + 1, i64::MAX - 1, i64::MAX] {
        assert_eq!(
            super::exact_j2000_second(super::ionex_epoch_from_j2000_seconds(seconds)),
            Some(seconds),
            "the reader encoding of {seconds} must read back exactly"
        );
        assert_eq!(
            super::exact_j2000_second(nanos_epoch(i128::from(seconds) * 1_000_000_000)),
            Some(seconds)
        );
    }
}

#[test]
fn exact_j2000_second_refuses_epochs_just_outside_the_i64_bounds() {
    // One second past each end, stated on the same day boundary as the bound
    // itself, so only the residual differs. The two parts are summed in `i128`,
    // so neither wraps into a plausible second.
    let (max_whole, max_fraction) = match super::ionex_epoch_from_j2000_seconds(i64::MAX).repr {
        InstantRepr::JulianDate(split) => (split.jd_whole, split.fraction),
        InstantRepr::Nanos(_) => panic!("the IONEX encoder produces a split Julian date"),
    };
    let max_residual = (max_fraction * SECONDS_PER_DAY).round();
    assert_eq!(
        super::exact_j2000_second(split_epoch(
            max_whole,
            (max_residual + 1.0) / SECONDS_PER_DAY
        )),
        None,
        "one second past i64::MAX is refused, not saturated"
    );
    assert_eq!(
        super::exact_j2000_second(split_epoch(max_whole, max_fraction)),
        Some(i64::MAX)
    );

    let (min_whole, min_fraction) = match super::ionex_epoch_from_j2000_seconds(i64::MIN).repr {
        InstantRepr::JulianDate(split) => (split.jd_whole, split.fraction),
        InstantRepr::Nanos(_) => panic!("the IONEX encoder produces a split Julian date"),
    };
    let min_residual = (min_fraction * SECONDS_PER_DAY).round();
    assert_eq!(
        super::exact_j2000_second(split_epoch(
            min_whole,
            (min_residual - 1.0) / SECONDS_PER_DAY
        )),
        None,
        "one second before i64::MIN is refused, not saturated"
    );
    assert_eq!(
        super::exact_j2000_second(split_epoch(min_whole, min_fraction)),
        Some(i64::MIN)
    );
}

#[test]
fn exact_j2000_second_refuses_the_nanos_count_the_f64_bound_saturated() {
    // `i64::MAX as f64` rounds up to 2^63, so an inclusive comparison against
    // it admitted counts an `i64` cannot hold and the cast saturated them onto
    // `i64::MAX`. Both of these were taken as `i64::MAX`; neither is.
    let past_max = i128::from(i64::MAX) * 1_000_000_000 + 999_999_999;
    assert_eq!(super::exact_j2000_second(nanos_epoch(past_max)), None);
    let two_pow_63 = 9_223_372_036_854_775_808_i128;
    assert_eq!(
        super::exact_j2000_second(nanos_epoch(two_pow_63 * 1_000_000_000)),
        None
    );
    assert_eq!(
        super::exact_j2000_second(nanos_epoch(-two_pow_63 * 1_000_000_000 - 1_000_000_000)),
        None
    );
}

#[test]
fn ionex_from_samples_keeps_a_quarter_day_map_epoch_through_writing() {
    // A boundary neither of the crate's encoders emits still names a whole
    // second, and public construction, the J2000-second view, the sample round
    // trip and the writer must all agree on which one. Six hours past the J2000
    // origin is 2000-01-01 18:00:00.
    let mut samples = valid_tec_grid_samples();
    samples.map_epochs = vec![split_epoch(crate::constants::J2000_JD + 0.25, 0.0)];
    let ionex = Ionex::from_samples(samples).expect("a quarter-day map epoch names 21_600 s");
    assert_eq!(ionex.map_epochs_s(), vec![21_600]);

    let rebuilt =
        Ionex::from_samples(ionex.tec_grid_samples()).expect("the quarter-day epoch rebuilds");
    assert_eq!(
        rebuilt, ionex,
        "the sample round trip keeps the epoch as given"
    );

    let encoded = ionex.to_ionex_string().expect("writable IONEX");
    assert!(
        encoded.contains("  2000     1     1    18     0     0"),
        "the written epoch record must state the second the product holds:\n{encoded}"
    );
    let reparsed = Ionex::parse_str(&encoded).expect("the written product reparses");
    assert_eq!(reparsed.map_epochs_s(), vec![21_600]);
}

#[test]
fn ionex_from_samples_refuses_a_fractional_map_epoch() {
    // The public sample path takes the epoch as given: a half-second epoch is
    // refused by name rather than stored as the whole second next to it.
    let mut samples = valid_tec_grid_samples();
    samples.map_epochs = vec![split_epoch(noon_boundary(0), 0.5 / SECONDS_PER_DAY)];
    assert_eq!(
        Ionex::from_samples(samples).expect_err("a fractional map epoch must be refused"),
        TecSamplesError::EpochNotRepresentable
    );
}

#[test]
fn ionex_from_node_samples_refuses_a_fractional_map_epoch() {
    let epoch = split_epoch(noon_boundary(0), 0.5 / SECONDS_PER_DAY);
    let samples = [
        TecSample {
            epoch,
            lat_deg: 1.0,
            lon_deg: 0.0,
            vtec_tecu: Some(10.0),
            rms_tecu: None,
            height_offset_km: None,
        },
        TecSample {
            epoch,
            lat_deg: 0.0,
            lon_deg: 1.0,
            vtec_tecu: Some(11.0),
            rms_tecu: None,
            height_offset_km: None,
        },
    ];
    assert_eq!(
        Ionex::from_node_samples(
            samples,
            450.0,
            6371.0,
            0,
            IonexHeader::new(IonexMappingFunction::CosZ),
        )
        .expect_err("a fractional node-sample epoch must be refused"),
        TecSamplesError::EpochNotRepresentable
    );
}

#[test]
fn ionex_from_samples_refuses_a_non_whole_nanosecond_map_epoch() {
    let mut samples = valid_tec_grid_samples();
    samples.map_epochs = vec![nanos_epoch(500_000_000)];
    assert_eq!(
        Ionex::from_samples(samples).expect_err("a sub-second nanosecond epoch must be refused"),
        TecSamplesError::EpochNotRepresentable
    );
}

// ---------------------------------------------------------------------------
// Slant evaluation on map times past the 53-bit integers
// ---------------------------------------------------------------------------

/// A two-map product whose maps sit at `map_times_s`, each holding one uniform
/// VTEC value at every node so the temporal blend is readable off the result.
fn two_map_samples_at(map_times_s: [i64; 2], vtec: [f64; 2]) -> TecGridSamples {
    TecGridSamples {
        map_epochs: map_times_s
            .iter()
            .map(|&seconds| super::ionex_epoch_from_j2000_seconds(seconds))
            .collect(),
        lat_nodes_deg: vec![1.0, 0.0],
        lon_nodes_deg: vec![0.0, 1.0],
        dlat_deg: -1.0,
        dlon_deg: 1.0,
        shell_height_km: 450.0,
        base_radius_km: 6371.0,
        exponent: 0,
        tec_maps: vtec
            .iter()
            .map(|&value| {
                vec![
                    vec![Some(value), Some(value)],
                    vec![Some(value), Some(value)],
                ]
            })
            .collect(),
        rms_maps: Vec::new(),
        height_maps: Vec::new(),
        header: IonexHeader::new(IonexMappingFunction::CosZ),
    }
}

/// One zenith evaluation on a two-map grid whose maps hold uniform VTEC, at the
/// grid's first node so the bilinear step is the node value itself.
fn two_map_components_at(map_times_s: [i64; 2], vtec: [f64; 2], epoch_s: i64) -> SlantComponents {
    let maps: Vec<Vec<Vec<Option<f64>>>> = vtec
        .iter()
        .map(|&value| {
            vec![
                vec![Some(value), Some(value)],
                vec![Some(value), Some(value)],
            ]
        })
        .collect();
    let epochs = map_times_s.map(super::ionex_epoch_from_j2000_seconds);
    let lat_arr = [0.0, -1.0];
    let lon_arr = [0.0, 1.0];
    slant_delay_components(
        PierceLineOfSight {
            lat_rad: 0.0,
            lon_rad: 0.0,
            az_rad: 0.0,
            el_rad: 90.0_f64.to_radians(),
        },
        1_575_420_000.0,
        6371.0,
        450.0,
        epoch_s,
        VtecGridView {
            map_epochs: &epochs,
            maps: &maps,
            lat_arr: &lat_arr,
            lon_arr: &lon_arr,
            dlat: -1.0,
            dlon: 1.0,
        },
    )
    .expect("every node holds a value")
}

#[test]
fn slant_temporal_weight_is_exact_on_map_times_past_the_f64_integers() {
    // Both map times and the query were projected through `f64` before the
    // weight was formed. At 2^53 and 2^53 + 1 the two projections were equal,
    // so the span was zero and the weight was not a number; at 2^53 and
    // 2^53 + 2 the midpoint rounded onto the first endpoint, so the later map
    // carried no weight anywhere inside the interval. The difference is now
    // taken in whole seconds, so each of these is the weight the axis states.
    let two_pow_53 = 9_007_199_254_740_992_i64;
    let vtec = [10.0, 30.0];

    let adjacent = [two_pow_53, two_pow_53 + 1];
    for (epoch_s, want_w, want_index) in [
        (adjacent[0], 0.0, 0_usize),
        (adjacent[1], 1.0, 0),
        (adjacent[0] - 86_400, 0.0, 0), // held before the first map
        (adjacent[1] + 86_400, 1.0, 0), // held after the last map
    ] {
        let components = two_map_components_at(adjacent, vtec, epoch_s);
        assert_eq!(components.w, want_w, "weight at {epoch_s}");
        assert_eq!(components.map_index, want_index);
        assert!(components.delay_m.is_finite() && components.delay_m > 0.0);
    }

    let spaced = [two_pow_53, two_pow_53 + 2];
    for (epoch_s, want_w) in [
        (spaced[0], 0.0),
        (spaced[0] + 1, 0.5), // the midpoint the collapsed axis could not see
        (spaced[1], 1.0),
    ] {
        let components = two_map_components_at(spaced, vtec, epoch_s);
        assert_eq!(components.w, want_w, "weight at {epoch_s}");
        assert!(components.delay_m.is_finite() && components.delay_m > 0.0);
    }

    // The blend itself follows the weight, so the midpoint sits strictly
    // between the two maps rather than on one of them.
    let low = two_map_components_at(spaced, vtec, spaced[0]);
    let mid = two_map_components_at(spaced, vtec, spaced[0] + 1);
    let high = two_map_components_at(spaced, vtec, spaced[1]);
    assert!(
        low.vtec < mid.vtec && mid.vtec < high.vtec,
        "{} {} {}",
        low.vtec,
        mid.vtec,
        high.vtec
    );
    assert_close(mid.vtec, 0.5 * (vtec[0] + vtec[1]));
}

#[test]
fn slant_temporal_weight_spans_the_whole_second_axis_without_overflow() {
    // A bracket that runs from one end of the axis to the other is wider than
    // an `i64` difference holds, and a query may sit at either end of it or
    // outside it altogether. The differences are formed in `i128`, so the span
    // stays positive and every weight stays inside `[0, 1]`.
    let ends = [i64::MIN, i64::MAX];
    let vtec = [10.0, 30.0];
    // The two map times themselves weight their own map exactly.
    for (epoch_s, want_w) in [(i64::MIN, 0.0), (i64::MAX, 1.0)] {
        let components = two_map_components_at(ends, vtec, epoch_s);
        assert_eq!(components.w, want_w, "weight at {epoch_s}");
        assert!(components.delay_m.is_finite());
    }
    // The seconds beside them, and a query outside the bracket altogether, stay
    // inside the unit interval instead of overflowing into a wild weight.
    for epoch_s in [i64::MIN + 1, i64::MAX - 1, 0, -86_400, 86_400] {
        let components = two_map_components_at(ends, vtec, epoch_s);
        assert!(
            (0.0..=1.0).contains(&components.w),
            "weight at {epoch_s} is {}",
            components.w
        );
        assert!(components.delay_m.is_finite() && components.delay_m > 0.0);
    }

    // A query in the middle of that span is an ordinary interior blend.
    let middle = two_map_components_at(ends, vtec, 0);
    assert!(
        middle.w > 0.0 && middle.w < 1.0,
        "interior weight {}",
        middle.w
    );
    assert!(middle.vtec > vtec[0] && middle.vtec < vtec[1]);
    assert!(middle.delay_m.is_finite() && middle.delay_m > 0.0);
}

#[test]
fn ionex_slant_delay_interpolates_across_map_times_past_the_f64_integers() {
    // The same correction through the public product entry: the map axis is
    // built, validated and evaluated from whole seconds end to end.
    let two_pow_53 = 9_007_199_254_740_992_i64;
    let map_times = [two_pow_53, two_pow_53 + 2];
    let ionex = Ionex::from_samples(two_map_samples_at(map_times, [10.0, 30.0]))
        .expect("map times past the f64 integers build a product");
    assert_eq!(ionex.map_epochs_s(), map_times.to_vec());

    let receiver =
        crate::frame::Wgs84Geodetic::new(0.5_f64.to_radians(), 0.5_f64.to_radians(), 0.0)
            .expect("valid WGS84 geodetic position");
    let el = 90.0_f64.to_radians();
    let az = 0.0;
    let f_l1 = crate::frequencies::frequency_hz(
        crate::GnssSystem::Gps,
        crate::frequencies::CarrierBand::L1,
    )
    .expect("canonical GPS L1 carrier exists");
    let delay_at = |epoch_s: i64| {
        super::ionex_slant_delay(&ionex, receiver, el, az, epoch_s, f_l1)
            .expect("a whole-second epoch on a valid product evaluates")
    };

    let low = delay_at(map_times[0]);
    let mid = delay_at(map_times[0] + 1);
    let high = delay_at(map_times[1]);
    for delay in [low, mid, high] {
        assert!(delay.is_finite() && delay > 0.0, "delay {delay}");
    }
    assert!(low < mid && mid < high, "{low} {mid} {high}");

    // Outside the bracket, a holding policy holds each endpoint map bit for
    // bit rather than extrapolating past a map time this large.
    let held = IonexSlantPolicy::default().with_coverage(IonexCoveragePolicy::Hold);
    let held_delay_at = |epoch_s: i64, want: IonexCoverageError| {
        let evaluation =
            ionex_slant_delay_with_policy(&ionex, receiver, el, az, epoch_s, f_l1, held)
                .expect("a held query outside coverage still evaluates");
        assert_eq!(evaluation.status.held, Some(want));
        evaluation.delay_m
    };
    assert_eq!(
        held_delay_at(
            map_times[0] - 86_400,
            IonexCoverageError::EpochBeforeFirstMap
        )
        .to_bits(),
        low.to_bits()
    );
    assert_eq!(
        held_delay_at(map_times[1] + 86_400, IonexCoverageError::EpochAfterLastMap).to_bits(),
        high.to_bits()
    );

    // Adjacent map times leave no interior second at all; the endpoints still
    // evaluate to their own maps rather than collapsing onto one value.
    let adjacent = Ionex::from_samples(two_map_samples_at(
        [two_pow_53, two_pow_53 + 1],
        [10.0, 30.0],
    ))
    .expect("adjacent map times build a product");
    let first = super::ionex_slant_delay(&adjacent, receiver, el, az, two_pow_53, f_l1)
        .expect("the first map time evaluates");
    let second = super::ionex_slant_delay(&adjacent, receiver, el, az, two_pow_53 + 1, f_l1)
        .expect("the second map time evaluates");
    assert!(first.is_finite() && second.is_finite());
    assert!(first < second, "{first} {second}");
}
