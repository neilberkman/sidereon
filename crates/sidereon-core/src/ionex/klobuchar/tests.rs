//! 0-ULP parity tests for the Klobuchar L1 ionospheric delay recipe.
//!
//! These assert the Rust port reproduces the canonical reference recipe
//! `parity/generator/klobuchar.py` bit-for-bit, using the committed golden
//! fixture `parity/fixtures/klobuchar_golden.json`. Values are serialised as
//! hex-float (Python `float.hex()`) so there is no decimal-parse ambiguity, and
//! parity is measured as ULP distance via the integer reinterpretation of the
//! IEEE-754 bit pattern, per the `skyfield_parity_test.exs` discipline.
//!
//! Every intermediate quantity (psi, phi_i, lambda_i, phi_m, t, F, AMP, PER, x,
//! t_iono) and the final delay is checked per component, so a divergence is
//! localised to a single algorithm step rather than only seen at the output.
//! The cases span the full branch matrix: cosine-series and night-time floor,
//! latitude clamping at both poles, local-time wrap above 86400 and below zero,
//! amplitude flooring at zero, and low elevation near the mask.

use std::path::PathBuf;

use serde_json::Value;

use super::klobuchar_l1_components;

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
        .join("tests/fixtures/klobuchar_golden.json")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!(
                "cannot locate tests/fixtures/klobuchar_golden.json from {}: {e}",
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

fn hexf_array4(v: &Value, key: &str) -> [f64; 4] {
    let arr = v[key]
        .as_array()
        .unwrap_or_else(|| panic!("{key} not an array"));
    assert_eq!(arr.len(), 4, "{key} must have 4 elements");
    [
        parse_hex_float(arr[0].as_str().unwrap()),
        parse_hex_float(arr[1].as_str().unwrap()),
        parse_hex_float(arr[2].as_str().unwrap()),
        parse_hex_float(arr[3].as_str().unwrap()),
    ]
}

#[test]
fn klobuchar_l1_zero_ulp_full_branch_matrix() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read klobuchar_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse klobuchar_golden.json");

    // Self-check the hex-float parser/serialiser round-trips a known bit pattern,
    // so a parser bug can never masquerade as parity.
    let probe = "0x1.921fb54442d18p+1"; // math.pi
    assert_eq!(
        float_hex(parse_hex_float(probe)),
        probe,
        "hex-float parser/serialiser round-trip is broken"
    );

    let cases = doc["cases"].as_array().expect("cases array");
    assert!(
        cases.len() >= 11,
        "expected the full branch matrix (>= 11 cases), found {}",
        cases.len()
    );

    // Each intermediate is checked, so a divergence is localised to one step.
    let components: &[&str] = &[
        "psi",
        "phi_i",
        "lambda_i",
        "phi_m",
        "t",
        "F",
        "AMP",
        "PER",
        "x",
        "t_iono",
        "delay_l1_m",
    ];

    let mut failures: Vec<String> = Vec::new();
    let mut checks = 0usize;

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let exp = &case["expect"];

        let got = klobuchar_l1_components(
            hexf(inp, "lat_deg"),
            hexf(inp, "lon_deg"),
            hexf(inp, "az_deg"),
            hexf(inp, "el_deg"),
            hexf(inp, "t_gps_s"),
            hexf_array4(inp, "alpha"),
            hexf_array4(inp, "beta"),
        );

        let actual = |c: &str| -> f64 {
            match c {
                "psi" => got.psi,
                "phi_i" => got.phi_i,
                "lambda_i" => got.lambda_i,
                "phi_m" => got.phi_m,
                "t" => got.t,
                "F" => got.f,
                "AMP" => got.amp,
                "PER" => got.per,
                "x" => got.x,
                "t_iono" => got.t_iono,
                "delay_l1_m" => got.delay_l1_m,
                other => panic!("unknown component {other}"),
            }
        };

        for &c in components {
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
                    "{name}.{c}: {ulp} ULP (rust={} ref={})",
                    float_hex(a),
                    exp[c].as_str().unwrap()
                ));
            }
        }

        // Dispersive scaling to BeiDou B1I: `klobuchar_native` applies the
        // `(f_L1 / f)^2` factor to the L1 delay, so it must reproduce the golden's
        // `delay_b1i_m` bit-for-bit (the frequency is taken from the golden so the
        // test and recipe share the exact carrier f64).
        let f_b1i = parse_hex_float(doc["constants"]["f_b1i"].as_str().expect("constants.f_b1i"));
        let params = super::super::KlobucharParams {
            alpha: hexf_array4(inp, "alpha"),
            beta: hexf_array4(inp, "beta"),
        };
        let got_b1i = super::super::klobuchar_native(
            &params,
            hexf(inp, "lat_deg"),
            hexf(inp, "lon_deg"),
            hexf(inp, "az_deg"),
            hexf(inp, "el_deg"),
            hexf(inp, "t_gps_s"),
            f_b1i,
        )
        .expect("valid Klobuchar golden inputs");
        let want_b1i = parse_hex_float(
            exp["delay_b1i_m"]
                .as_str()
                .unwrap_or_else(|| panic!("case {name}: missing delay_b1i_m")),
        );
        let ulp = ulp_distance(got_b1i, want_b1i);
        checks += 1;
        if ulp != 0 {
            failures.push(format!(
                "{name}.delay_b1i_m: {ulp} ULP (rust={} ref={})",
                float_hex(got_b1i),
                exp["delay_b1i_m"].as_str().unwrap()
            ));
        }
    }

    assert!(checks > 0, "no components were checked - fixture empty?");
    assert!(
        failures.is_empty(),
        "Klobuchar Rust port diverged from the reference recipe on {} of {checks} components:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Coverage for the PUBLIC `klobuchar(epoch, radians...)` wrapper, which the
/// 0-ULP kernel test above does not exercise (it calls the kernel directly with
/// degrees + an exact second-of-day). The wrapper converts angle (rad->deg) and
/// time (split Julian date -> second-of-day) at its boundary; both are
/// representation-bound, so it is NOT bit-exact to the golden. This pins that
/// the only divergence is that documented conversion bound: the public result
/// must agree with the golden delay to well under a micrometre (observed at the
/// nanometre level). A regression that broke the wiring or units would blow far
/// past this and fail.
#[test]
fn klobuchar_public_wrapper_matches_golden_within_conversion_bound() {
    use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};
    use core::f64::consts::PI;

    let raw = std::fs::read_to_string(fixture_path()).expect("read klobuchar_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse klobuchar_golden.json");
    let cases = doc["cases"].as_array().expect("cases array");

    // The conversion bound. Observed worst case across the matrix is ~1e-9 m;
    // 1e-6 m is a tight ceiling that still catches any wiring/unit regression.
    const BOUND_M: f64 = 1.0e-6;

    let mut checked = 0usize;
    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let exp = &case["expect"];

        let receiver = crate::frame::Wgs84Geodetic::new(
            hexf(inp, "lat_deg") * PI / 180.0,
            hexf(inp, "lon_deg") * PI / 180.0,
            0.0,
        )
        .expect("valid WGS84 geodetic position");
        let el_rad = hexf(inp, "el_deg") * PI / 180.0;
        let az_rad = hexf(inp, "az_deg") * PI / 180.0;
        let params = super::super::KlobucharParams {
            alpha: hexf_array4(inp, "alpha"),
            beta: hexf_array4(inp, "beta"),
        };

        // A split Julian date whose second-of-day equals the case's t_gps_s:
        // jd_whole on a .5 boundary (midnight), fraction = t_gps_s / 86400.
        let t_gps_s = hexf(inp, "t_gps_s");
        let epoch = Instant::from_julian_date(
            TimeScale::Gpst,
            JulianDateSplit::new(2_451_544.5, t_gps_s / 86_400.0).expect("valid split Julian date"),
        );

        // Report on L1 so the dispersive factor is exactly 1
        // and this isolates the angle/time conversion boundary.
        let f_l1_hz = crate::frequencies::frequency_hz(
            crate::GnssSystem::Gps,
            crate::frequencies::CarrierBand::L1,
        )
        .expect("canonical GPS L1 carrier exists");
        let got = super::super::klobuchar(&params, receiver, el_rad, az_rad, epoch, f_l1_hz)
            .expect("valid Klobuchar public wrapper inputs");
        let want = parse_hex_float(exp["delay_l1_m"].as_str().unwrap());

        let diff = (got - want).abs();
        assert!(
            diff < BOUND_M,
            "{name}: public klobuchar wrapper off by {diff:e} m (> {BOUND_M:e}); got={}, want={}",
            float_hex(got),
            exp["delay_l1_m"].as_str().unwrap()
        );
        checked += 1;
    }
    assert!(
        checked >= 11,
        "expected the full case matrix, checked {checked}"
    );
}
