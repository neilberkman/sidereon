//! 0-ULP parity tests for the dilution-of-precision recipe.
//!
//! These assert the Rust port reproduces the canonical reference recipe
//! `parity/generator/dop.py` bit-for-bit, using the committed golden fixture
//! `parity/fixtures/dop_golden.json` (vendored at
//! `tests/fixtures/dop_golden.json`). Values are serialised as hex-float
//! (Python `float.hex()`) so there is no decimal-parse ambiguity, and parity is
//! measured as ULP distance via the integer reinterpretation of the IEEE-754
//! bit pattern, per the `skyfield_parity_test.exs` discipline.
//!
//! Unlike the BLAS-bound `trf` solver step, the DOP inverse is an explicit
//! small-matrix cofactor expansion with a pinned operation order, so it is a
//! genuine libm/arithmetic-bound 0-ULP target (not a tolerance/agreement
//! check): the determinant, the cofactor matrix, the ENU-rotated position
//! block, and every DOP scalar are asserted component-by-component to 0 ULP.
//! A singular family verifies the documented failure mode.

use std::path::PathBuf;

use serde_json::Value;

use super::{dop, dop_multi, line_of_sight_from_az_el_deg, test_support, DopError, LineOfSight};
use crate::frame::Wgs84Geodetic;
use crate::id::GnssSystem;

/// A fixed, distinct GNSS for each clock column so the geometry-only multi-system
/// tests can supply the `systems` mapping `dop_multi` tags its TDOPs with. The
/// geometry is independent of which constellations these are; only the tag
/// identity is exercised.
fn systems_for(n_clocks: usize) -> Vec<GnssSystem> {
    const ORDER: [GnssSystem; 7] = [
        GnssSystem::Gps,
        GnssSystem::Galileo,
        GnssSystem::Glonass,
        GnssSystem::BeiDou,
        GnssSystem::Qzss,
        GnssSystem::Navic,
        GnssSystem::Sbas,
    ];
    ORDER[..n_clocks].to_vec()
}

/// Parse a C99 / Python `float.hex()` hex-float string into the exact `f64`.
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

/// ULP distance between two `f64`, NaN -> `u64::MAX` so it never reads as 0 ULP.
fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    ordered_i64(a).abs_diff(ordered_i64(b))
}

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
        .join("tests/fixtures/dop_golden.json")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!(
                "cannot locate tests/fixtures/dop_golden.json from {}: {e}",
                crate_dir.display()
            )
        })
}

fn multi_fixture_path() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .join("tests/fixtures/dop_multi_golden.json")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!(
                "cannot locate tests/fixtures/dop_multi_golden.json from {}: {e}",
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

/// Reconstruct the line-of-sight directions and weights from a fixture case.
///
/// The golden stores the ECEF LOS unit vectors directly; the design rows are
/// derived from them by the Rust port (so the row form is itself under test).
fn los_and_weights(inp: &Value) -> (Vec<LineOfSight>, Vec<f64>) {
    let los_arr = inp["los_ecef"].as_array().expect("los_ecef array");
    let w_arr = inp["weights"].as_array().expect("weights array");
    assert_eq!(los_arr.len(), w_arr.len(), "los/weight length mismatch");
    let los = los_arr
        .iter()
        .map(|e| {
            let c = e.as_array().expect("los entry array");
            LineOfSight::new(
                parse_hex_float(c[0].as_str().unwrap()),
                parse_hex_float(c[1].as_str().unwrap()),
                parse_hex_float(c[2].as_str().unwrap()),
            )
        })
        .collect();
    let weights = w_arr
        .iter()
        .map(|w| parse_hex_float(w.as_str().unwrap()))
        .collect();
    (los, weights)
}

#[test]
fn dop_zero_ulp_full_branch_matrix() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read dop_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse dop_golden.json");

    // Self-check the hex-float parser/serialiser round-trips a known bit
    // pattern, so a parser bug cannot masquerade as parity.
    let probe = "0x1.921fb54442d18p+1"; // math.pi
    assert_eq!(
        float_hex(parse_hex_float(probe)),
        probe,
        "hex-float parser/serialiser round-trip is broken"
    );

    let mut failures: Vec<String> = Vec::new();
    let mut checks = 0usize;

    let cases = doc["cases"].as_array().expect("cases array");
    assert!(
        cases.len() >= 6,
        "expected the full DOP branch matrix (>= 6 cases), found {}",
        cases.len()
    );

    let scalar_keys: &[&str] = &[
        "qe", "qn", "qu", "qt", "gdop", "pdop", "hdop", "vdop", "tdop",
    ];

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let exp = &case["expect"];

        let (los, weights) = los_and_weights(inp);
        let lat_rad = hexf(inp, "lat_rad");
        let lon_rad = hexf(inp, "lon_rad");
        let receiver =
            Wgs84Geodetic::new(lat_rad, lon_rad, 0.0).expect("valid WGS84 geodetic position");

        let mut check = |label: String, a: f64, want_hex: &str| {
            let want = parse_hex_float(want_hex);
            let ulp = ulp_distance(a, want);
            checks += 1;
            if ulp != 0 {
                failures.push(format!(
                    "{label}: {ulp} ULP (rust={} ref={})",
                    float_hex(a),
                    want_hex
                ));
            }
        };

        // --- Normal matrix H^T W H (intermediate). ---
        let a = test_support::normal_matrix_for(&los, &weights);
        let nm = exp["normal_matrix"].as_array().expect("normal_matrix");
        for i in 0..4 {
            let row = nm[i].as_array().unwrap();
            for j in 0..4 {
                check(
                    format!("{name}.A[{i}][{j}]"),
                    a[i][j],
                    row[j].as_str().unwrap(),
                );
            }
        }

        // --- Cofactor matrix Q = A^-1 (intermediate). ---
        let q = test_support::inv4_for(&a).expect("non-singular nominal geometry");
        let cm = exp["cofactor_matrix"].as_array().expect("cofactor_matrix");
        for i in 0..4 {
            let row = cm[i].as_array().unwrap();
            for j in 0..4 {
                check(
                    format!("{name}.Q[{i}][{j}]"),
                    q[i][j],
                    row[j].as_str().unwrap(),
                );
            }
        }

        // --- Determinant (intermediate). ---
        check(
            format!("{name}.det"),
            test_support::det4_for(&a),
            exp["det"].as_str().unwrap(),
        );

        // --- ENU position block (intermediate). ---
        let enu = test_support::enu_block_for(&q, lat_rad, lon_rad);
        let eb = exp["enu_pos_block"].as_array().expect("enu_pos_block");
        for i in 0..3 {
            let row = eb[i].as_array().unwrap();
            for j in 0..3 {
                check(
                    format!("{name}.ENU[{i}][{j}]"),
                    enu[i][j],
                    row[j].as_str().unwrap(),
                );
            }
        }

        // --- The public DOP scalars. ---
        let got = dop(&los, &weights, receiver).expect("nominal geometry yields DOP");
        let scalar = |k: &str| -> f64 {
            match k {
                // qe/qn/qu/qt are the ENU/clock variances the scalars derive
                // from; assert them via the recomputed block and Q so the
                // intermediate split is itself checked against the golden.
                "qe" => enu[0][0],
                "qn" => enu[1][1],
                "qu" => enu[2][2],
                "qt" => q[3][3],
                "gdop" => got.gdop,
                "pdop" => got.pdop,
                "hdop" => got.hdop,
                "vdop" => got.vdop,
                "tdop" => got.tdop,
                other => panic!("unknown scalar {other}"),
            }
        };
        for &k in scalar_keys {
            check(format!("{name}.{k}"), scalar(k), exp[k].as_str().unwrap());
        }
    }

    assert!(checks > 0, "no components were checked - fixture empty?");
    assert!(
        failures.is_empty(),
        "DOP Rust port diverged from the reference recipe on {} of {checks} components:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn dop_singular_geometries_are_rejected() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read dop_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse dop_golden.json");

    let cases = doc["singular_cases"]
        .as_array()
        .expect("singular_cases array");
    assert!(
        cases.len() >= 2,
        "expected the singular family (>= 2 cases), found {}",
        cases.len()
    );

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let (los, weights) = los_and_weights(inp);
        let receiver = Wgs84Geodetic::new(hexf(inp, "lat_rad"), hexf(inp, "lon_rad"), 0.0)
            .expect("valid WGS84 geodetic position");

        // The normal matrix is still computed bit-for-bit; only the inverse /
        // variance predicate flags the geometry as having no finite DOP.
        let a = test_support::normal_matrix_for(&los, &weights);
        let nm = case["expect"]["normal_matrix"]
            .as_array()
            .expect("normal_matrix");
        for i in 0..4 {
            let row = nm[i].as_array().unwrap();
            for j in 0..4 {
                let want = parse_hex_float(row[j].as_str().unwrap());
                assert_eq!(
                    a[i][j].to_bits(),
                    want.to_bits(),
                    "singular {name}.A[{i}][{j}] not 0-ULP: rust={} ref={}",
                    float_hex(a[i][j]),
                    row[j].as_str().unwrap()
                );
            }
        }

        let res = dop(&los, &weights, receiver);
        match res {
            Err(DopError::Singular) | Err(DopError::TooFewSatellites) => {}
            other => panic!("singular case {name} expected a DopError, got {other:?}"),
        }
    }
}

/// A geometry with fewer than four satellites is rejected before any inverse.
#[test]
fn dop_too_few_satellites() {
    let los = [
        LineOfSight::new(0.1, 0.2, 0.9746794344808963),
        LineOfSight::new(-0.3, 0.4, 0.8660254037844387),
        LineOfSight::new(0.5, -0.1, 0.8602325267042627),
    ];
    let weights = [1.0, 1.0, 1.0];
    let receiver = Wgs84Geodetic::new(0.5, 0.2, 0.0).expect("valid WGS84 geodetic position");
    assert_eq!(
        dop(&los, &weights, receiver),
        Err(DopError::TooFewSatellites)
    );
}

#[test]
fn dop_rejects_non_unit_los_and_out_of_range_receiver() {
    let los = [
        LineOfSight::new(0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
        LineOfSight::new(0.5773502691896258, -0.5773502691896258, -0.5773502691896258),
        LineOfSight::new(-0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
        LineOfSight::new(-0.5773502691896258, -0.5773502691896258, 0.5773502691896258),
    ];
    let weights = [1.0, 1.0, 1.0, 1.0];
    let receiver = Wgs84Geodetic::new(0.5, 0.2, 0.0).expect("valid WGS84 geodetic position");

    let mut bad_los = los;
    bad_los[0] = LineOfSight::new(2.0, 0.0, 0.0);
    assert_eq!(
        dop(&bad_los, &weights, receiver),
        Err(DopError::InvalidInput {
            field: "los",
            reason: "not unit length"
        })
    );
    assert_eq!(
        dop_multi(
            &bad_los,
            &[0usize; 4],
            &systems_for(1),
            1,
            &weights,
            receiver
        ),
        Err(DopError::InvalidInput {
            field: "los",
            reason: "not unit length"
        })
    );

    let bad_lat = Wgs84Geodetic {
        lat_rad: std::f64::consts::FRAC_PI_2 + 0.1,
        lon_rad: 0.2,
        height_m: 0.0,
    };
    assert_eq!(
        dop(&los, &weights, bad_lat),
        Err(DopError::InvalidInput {
            field: "receiver.lat_rad",
            reason: "out of range"
        })
    );

    let west_antimeridian = Wgs84Geodetic {
        lat_rad: 0.5,
        lon_rad: -std::f64::consts::PI,
        height_m: 0.0,
    };
    dop(&los, &weights, west_antimeridian).expect("west antimeridian receiver is valid");
    dop_multi(
        &los,
        &[0usize; 4],
        &systems_for(1),
        1,
        &weights,
        west_antimeridian,
    )
    .expect("west antimeridian receiver is valid");

    let bad_lon = Wgs84Geodetic {
        lat_rad: 0.5,
        lon_rad: -std::f64::consts::PI - 0.1,
        height_m: 0.0,
    };
    assert_eq!(
        dop(&los, &weights, bad_lon),
        Err(DopError::InvalidInput {
            field: "receiver.lon_rad",
            reason: "out of range"
        })
    );
}

#[test]
fn dop_mismatched_los_and_weight_lengths_return_invalid_input() {
    let los = [
        LineOfSight::new(0.0, 0.34202014332566877, 0.9396926207859084),
        LineOfSight::new(0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(-0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(0.8137976813493737, 0.46984631039295416, 0.3420201433256687),
    ];
    let weights = [1.0, 1.0, 1.0];
    let receiver = Wgs84Geodetic::new(0.5, 0.2, 0.0).expect("valid WGS84 geodetic position");

    assert_eq!(
        dop(&los, &weights, receiver),
        Err(DopError::InvalidInput {
            field: "weights",
            reason: "length must match los"
        })
    );
}

#[test]
fn dop_non_finite_inputs_return_invalid_input() {
    let los = [
        LineOfSight::new(0.0, 0.34202014332566877, 0.9396926207859084),
        LineOfSight::new(0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(-0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(0.8137976813493737, 0.46984631039295416, 0.3420201433256687),
    ];
    let weights = [1.0, 1.0, 1.0, 1.0];
    let receiver = Wgs84Geodetic::new(0.5, 0.2, 0.0).expect("valid WGS84 geodetic position");

    let mut bad_los = los;
    bad_los[0].e_x = f64::NAN;
    assert_eq!(
        dop(&bad_los, &weights, receiver),
        Err(DopError::InvalidInput {
            field: "los",
            reason: "not finite"
        })
    );

    let mut bad_weights = weights;
    bad_weights[1] = f64::INFINITY;
    assert_eq!(
        dop(&los, &bad_weights, receiver),
        Err(DopError::InvalidInput {
            field: "weights",
            reason: "not finite"
        })
    );

    let bad_receiver = Wgs84Geodetic {
        lat_rad: f64::NAN,
        lon_rad: 0.2,
        height_m: 0.0,
    };
    assert_eq!(
        dop(&los, &weights, bad_receiver),
        Err(DopError::InvalidInput {
            field: "receiver",
            reason: "not finite"
        })
    );
}

#[test]
fn dop_negative_weight_returns_invalid_input() {
    let los = [
        LineOfSight::new(0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
        LineOfSight::new(0.5773502691896258, -0.5773502691896258, -0.5773502691896258),
        LineOfSight::new(-0.5773502691896258, 0.5773502691896258, -0.5773502691896258),
        LineOfSight::new(-0.5773502691896258, -0.5773502691896258, 0.5773502691896258),
    ];
    let weights = [1.0, 1.0, 1.0, -1.0];
    let receiver = Wgs84Geodetic::new(0.5, 0.2, 0.0).expect("valid WGS84 geodetic position");

    assert_eq!(
        dop(&los, &weights, receiver),
        Err(DopError::InvalidInput {
            field: "weights",
            reason: "negative"
        })
    );
}

#[test]
fn az_el_constructor_matches_symmetric_geometry() {
    let receiver = Wgs84Geodetic::new(0.0, 0.0, 0.0).expect("valid WGS84 geodetic position");
    let az_el = [
        (45.0, 35.264389682754654),
        (225.0, 35.264389682754654),
        (135.0, -35.264389682754654),
        (315.0, -35.264389682754654),
    ];
    let los: Vec<LineOfSight> = az_el
        .iter()
        .map(|&(az, el)| {
            line_of_sight_from_az_el_deg(az, el, receiver).expect("valid az/el line of sight")
        })
        .collect();
    let weights = [1.0, 1.0, 1.0, 1.0];
    let d = dop(&los, &weights, receiver).expect("symmetric DOP");

    assert_eq!(d.gdop.to_bits(), 0x3ff94c583ada5b53);
    assert_eq!(d.pdop.to_bits(), 0x3ff8000000000000);
    assert_eq!(d.hdop.to_bits(), 0x3ff3988e1409212e);
    assert_eq!(d.vdop.to_bits(), 0x3febb67ae8584caa);
    assert_eq!(d.tdop.to_bits(), 0x3fe0000000000000);
}

/// GDOP^2 = PDOP^2 + TDOP^2 and PDOP^2 = HDOP^2 + VDOP^2 are identities of the
/// definition (the ENU rotation is orthogonal and preserves the position
/// trace). This is a sanity relation, not a parity assertion, so it is checked
/// to a tight tolerance rather than to the bit.
#[test]
fn dop_consistency_relations() {
    let los = [
        LineOfSight::new(0.0, 0.34202014332566877, 0.9396926207859084),
        LineOfSight::new(0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(-0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(0.8137976813493737, 0.46984631039295416, 0.3420201433256687),
    ];
    let weights = [1.0, 1.0, 1.0, 1.0];
    let receiver = Wgs84Geodetic::new(std::f64::consts::FRAC_PI_4, 0.17453292519943295, 0.0)
        .expect("valid WGS84 geodetic position");
    let d = dop(&los, &weights, receiver).expect("DOP");

    let lhs = d.gdop * d.gdop;
    let rhs = d.pdop * d.pdop + d.tdop * d.tdop;
    assert!(
        (lhs - rhs).abs() <= 1e-9 * lhs.max(1.0),
        "GDOP^2 != PDOP^2 + TDOP^2"
    );

    let plhs = d.pdop * d.pdop;
    let prhs = d.hdop * d.hdop + d.vdop * d.vdop;
    assert!(
        (plhs - prhs).abs() <= 1e-9 * plhs.max(1.0),
        "PDOP^2 != HDOP^2 + VDOP^2"
    );
}

// With a single clock column, the general multi-system inverse and the 4x4
// cofactor inverse describe the same normal matrix, so the two DOP routines must
// agree to within the Cholesky-vs-cofactor floating-point gap (~1e-9 relative).
// (The single-system `dop` keeps the 0-ULP golden; `dop_multi` is a
// deterministic diagnostic and is not bit-pinned, hence the tolerance.)
#[test]
fn dop_multi_matches_single_for_one_clock() {
    let los = [
        LineOfSight::new(0.0, 0.34202014332566877, 0.9396926207859084),
        LineOfSight::new(0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(-0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(0.8137976813493737, 0.46984631039295416, 0.3420201433256687),
    ];
    let weights = [1.0, 0.8, 1.2, 0.5];
    let receiver = Wgs84Geodetic::new(std::f64::consts::FRAC_PI_4, 0.17453292519943295, 0.0)
        .expect("valid WGS84 geodetic position");

    let single = dop(&los, &weights, receiver).expect("single-system DOP");
    let clock_index = [0usize; 4];
    let multi = dop_multi(&los, &clock_index, &systems_for(1), 1, &weights, receiver)
        .expect("multi-system DOP");

    for (a, b, name) in [
        (single.gdop, multi.gdop, "GDOP"),
        (single.pdop, multi.pdop, "PDOP"),
        (single.hdop, multi.hdop, "HDOP"),
        (single.vdop, multi.vdop, "VDOP"),
        (single.tdop, multi.tdop, "TDOP"),
    ] {
        assert!(
            (a - b).abs() <= 1e-9 * a.abs().max(1.0),
            "{name}: single {a} != multi {b}"
        );
    }
}

// A two-system geometry (one extra clock column) must produce a finite,
// positive DOP, satisfy PDOP^2 = HDOP^2 + VDOP^2 (a pure position-block
// relation, unaffected by the clock split), and have GDOP^2 >= PDOP^2 + TDOP^2
// because the trace now carries a second clock variance on top of the reference.
#[test]
fn dop_multi_two_systems_is_finite_and_consistent() {
    // Six lines of sight; the first three carry clock 0, the last three clock 1
    // (e.g. GPS and Galileo). Six satellites >= 3 position + 2 clock parameters.
    let los = [
        LineOfSight::new(0.0, 0.34202014332566877, 0.9396926207859084),
        LineOfSight::new(0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(-0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(0.8137976813493737, 0.46984631039295416, 0.3420201433256687),
        LineOfSight::new(-0.6427876096865393, 0.0, 0.766044443118978),
        LineOfSight::new(
            -0.49240387650610395,
            -0.6040227735550537,
            0.6269790924435587,
        ),
    ];
    let weights = [1.0, 0.9, 1.1, 0.7, 1.3, 0.8];
    let clock_index = [0usize, 0, 0, 1, 1, 1];
    let receiver = Wgs84Geodetic::new(std::f64::consts::FRAC_PI_4, 0.17453292519943295, 0.0)
        .expect("valid WGS84 geodetic position");

    let d = dop_multi(&los, &clock_index, &systems_for(2), 2, &weights, receiver)
        .expect("multi-system DOP");

    // The per-system TDOPs are tagged with the supplied clock-column systems,
    // in clock-column order, and entry 0 (reference clock) equals the scalar
    // TDOP bit-for-bit.
    assert_eq!(
        d.system_tdops.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
        systems_for(2),
        "system_tdops must be tagged in clock-column order"
    );
    assert_eq!(
        d.system_tdops[0].1.to_bits(),
        d.tdop.to_bits(),
        "system_tdops[0] must equal the scalar TDOP bit-for-bit"
    );

    for (v, name) in [
        (d.gdop, "GDOP"),
        (d.pdop, "PDOP"),
        (d.hdop, "HDOP"),
        (d.vdop, "VDOP"),
        (d.tdop, "TDOP"),
    ] {
        assert!(v.is_finite() && v > 0.0, "{name} not finite/positive: {v}");
    }

    let plhs = d.pdop * d.pdop;
    let prhs = d.hdop * d.hdop + d.vdop * d.vdop;
    assert!(
        (plhs - prhs).abs() <= 1e-9 * plhs.max(1.0),
        "PDOP^2 != HDOP^2 + VDOP^2"
    );

    // GDOP^2 is the full trace: position block + both clock variances. TDOP^2 is
    // only the reference clock, so GDOP^2 must exceed PDOP^2 + TDOP^2 by the
    // second system's clock variance.
    let gsq = d.gdop * d.gdop;
    assert!(
        gsq > plhs + d.tdop * d.tdop,
        "GDOP^2 ({gsq}) should exceed PDOP^2 + TDOP^2 with a second clock"
    );
}

// Fewer satellites than parameters (3 position + n_clocks) is rejected before
// any factorisation.
#[test]
fn dop_multi_too_few_satellites() {
    let los = [
        LineOfSight::new(0.1, 0.2, 0.9746794344808963),
        LineOfSight::new(-0.3, 0.4, 0.8660254037844387),
        LineOfSight::new(0.5, -0.1, 0.8602325267042627),
        LineOfSight::new(0.0, 0.0, 1.0),
    ];
    let weights = [1.0, 1.0, 1.0, 1.0];
    let clock_index = [0usize, 0, 1, 1];
    let receiver = Wgs84Geodetic::new(0.5, 0.2, 0.0).expect("valid WGS84 geodetic position");
    // 4 satellites but 3 + 2 = 5 parameters.
    let err = dop_multi(&los, &clock_index, &systems_for(2), 2, &weights, receiver).unwrap_err();
    assert!(matches!(err, DopError::TooFewSatellites));
}

// Multi-system per-constellation TDOP against the numpy dense-inverse reference
// (`dop_multi_golden.json`, generated by `generate_dop_multi.py`). Unlike the
// single-system `dop` 0-ULP golden, `dop_multi` forms Q with a dense symmetric
// (Cholesky) inverse and the reference uses an LU inverse, so this is a
// tight-tolerance agreement check, not 0 ULP. The single-system golden above is
// untouched and still asserted to 0 ULP.
#[test]
fn dop_multi_per_system_tdop_matches_numpy_reference() {
    // Worst observed relative agreement is ~1.6e-15 (sub-ULP of the values);
    // gate well above that yet far below any geometry-meaningful scale, so a
    // real regression in the dense inverse or the diagonal mapping trips it.
    const REL_TOL: f64 = 1.0e-9;

    let raw = std::fs::read_to_string(multi_fixture_path()).expect("read dop_multi_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse dop_multi_golden.json");
    let cases = doc["cases"].as_array().expect("cases array");
    assert!(cases.len() >= 2, "expected >= 2 multi-system cases");

    let mut worst_rel = 0.0_f64;
    let mut checks = 0usize;
    let mut failures: Vec<String> = Vec::new();

    let mut check = |label: String, got: f64, want: f64, worst: &mut f64, n: &mut usize| {
        *n += 1;
        let rel = (got - want).abs() / want.abs().max(1.0);
        if rel > *worst {
            *worst = rel;
        }
        if rel > REL_TOL {
            failures.push(format!(
                "{label}: rel {rel:.3e} (rust={} ref={})",
                float_hex(got),
                float_hex(want)
            ));
        }
    };

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let exp = &case["expect"];

        let (los, weights) = los_and_weights(inp);
        let n_clocks = inp["n_clocks"].as_u64().expect("n_clocks") as usize;
        let clock_index: Vec<usize> = inp["clock_index"]
            .as_array()
            .expect("clock_index array")
            .iter()
            .map(|c| c.as_u64().expect("clock index") as usize)
            .collect();
        let receiver = Wgs84Geodetic::new(hexf(inp, "lat_rad"), hexf(inp, "lon_rad"), 0.0)
            .expect("valid WGS84 geodetic position");

        let systems = systems_for(n_clocks);
        let d = dop_multi(&los, &clock_index, &systems, n_clocks, &weights, receiver)
            .expect("nominal multi-system geometry yields DOP");

        // The per-system TDOP vector is GNSS-tagged in clock-column order and one
        // entry per clock; entry 0 is the reference clock and must equal the
        // scalar TDOP exactly (same diagonal, same sqrt) - a 0-ULP
        // internal-consistency bar. The tags must be the supplied systems.
        assert_eq!(
            d.system_tdops.len(),
            n_clocks,
            "{name}: system_tdops length must equal n_clocks"
        );
        assert_eq!(
            d.system_tdops.iter().map(|(s, _)| *s).collect::<Vec<_>>(),
            systems,
            "{name}: system_tdops must be tagged in clock-column order"
        );
        assert_eq!(
            d.system_tdops[0].1.to_bits(),
            d.tdop.to_bits(),
            "{name}: system_tdops[0] must equal the scalar TDOP bit-for-bit"
        );

        let per = exp["per_system_tdop"].as_array().expect("per_system_tdop");
        assert_eq!(
            per.len(),
            n_clocks,
            "{name}: reference per_system_tdop count"
        );
        for (i, want) in per.iter().enumerate() {
            check(
                format!("{name}.system_tdop[{i}]"),
                d.system_tdops[i].1,
                parse_hex_float(want.as_str().unwrap()),
                &mut worst_rel,
                &mut checks,
            );
        }
        for (k, got) in [
            ("gdop", d.gdop),
            ("pdop", d.pdop),
            ("hdop", d.hdop),
            ("vdop", d.vdop),
            ("tdop", d.tdop),
        ] {
            check(
                format!("{name}.{k}"),
                got,
                hexf(exp, k),
                &mut worst_rel,
                &mut checks,
            );
        }
    }

    assert!(checks > 0, "no components were checked - fixture empty?");
    assert!(
        failures.is_empty(),
        "multi-system DOP diverged from the numpy reference on {} of {checks} components \
         (worst rel {worst_rel:.3e}):\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
    // Surface the achieved agreement so a tightening/loosening is a deliberate edit.
    assert!(
        worst_rel < REL_TOL,
        "worst relative agreement {worst_rel:.3e} exceeded tolerance"
    );
}
