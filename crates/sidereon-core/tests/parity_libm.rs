#![cfg(sidereon_repo_tests)]
//! Tiny libm-bound 0-ULP parity test (parity build order step 2).
//!
//! Proves the parity HARNESS + MATH CONTRACT end-to-end before any GNSS
//! numerics: read the hex-float golden vector emitted by the pinned numpy
//! reference stack (`parity/fixtures/libm_tiny.json`), recompute each value
//! with Rust `std` (`f64::sin` etc.), and assert 0 ULP (bit-identical) per
//! component.
//!
//! This mirrors the discipline of `repos/sidereon/test/skyfield_parity_test.exs`:
//! values are serialized as hex-float (Python `float.hex()`) so there is no
//! decimal-parse ambiguity, and parity is measured as ULP distance via the
//! integer reinterpretation of the IEEE-754 bit pattern.
//!
//! If Rust `std` libm diverges from numpy's (Apple libsystem_m on this pinned
//! target) by even one ULP, this test FAILS and prints the exact delta rather
//! than hiding it.

use std::path::PathBuf;

use serde_json::Value;

/// Parse a C99 / Python `float.hex()` hex-float string into the exact `f64`.
///
/// Rust's `str::parse::<f64>()` does not accept hex-float syntax, so we parse
/// it by hand. Format examples produced by Python `float.hex()`:
///   `0x1.921fb54442d18p-1`, `-0x1.a000000000000p+1`, `0x0.0p+0`,
///   `0x1.0c6f7a0b5ed8dp-20`. Mantissa is `<int>.<frac-hex>`, exponent is a
///   base-2 power after `p`. The leading integer digit is 1 for normals, 0 for
///   zero/subnormal. This reconstruction is exact (no rounding): every hex frac
///   digit is 4 mantissa bits and f64 has 52, so 13 hex digits fit losslessly.
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

    // Significand value = int_part + frac_part(hex) * 16^-k, scaled by 2^exp2.
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
    let val = significand * exp2_pow(exp2);
    if neg {
        -val
    } else {
        val
    }
}

/// Exact `2^n` for an `i32` exponent, via `f64::powi` on the radix. `powi(2, n)`
/// is exact for the range used here (the libm fixture stays well inside f64).
fn exp2_pow(n: i32) -> f64 {
    2.0f64.powi(n)
}

/// ULP distance between two `f64`, using the monotone signed-integer mapping of
/// the IEEE-754 bit pattern (same scheme as `skyfield_parity_test.exs`). Returns
/// `u64::MAX` for any NaN, so a NaN never silently reads as 0 ULP.
fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    let ia = ordered_i64(a);
    let ib = ordered_i64(b);
    ia.abs_diff(ib)
}

/// Map an `f64` to a sign-magnitude-ordered `i64` so that adjacent floats differ
/// by exactly 1 (handles the +0/-0 and negative-side ordering correctly).
fn ordered_i64(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    if bits < 0 {
        // Negative floats: flip to a continuous descending order below zero.
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
    let s = if unbiased >= 0 {
        format!("{sign}0x1.{mantissa:013x}p+{unbiased}")
    } else {
        format!("{sign}0x1.{mantissa:013x}p{unbiased}")
    };
    s
}

fn fixture_path() -> PathBuf {
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    crate_dir
        .join("tests/fixtures/libm_tiny.json")
        .canonicalize()
        .unwrap_or_else(|e| {
            panic!(
                "cannot locate tests/fixtures/libm_tiny.json from {}: {e}",
                crate_dir.display()
            )
        })
}

#[test]
fn libm_tiny_zero_ulp() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read libm_tiny.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse libm_tiny.json");

    let mut failures: Vec<String> = Vec::new();
    let mut checks = 0usize;

    // ---- self-check the hex-float parser/serializer round-trips a known bit
    // pattern, so a parser bug can never masquerade as parity. ----
    let probe = "0x1.921fb54442d18p-1"; // pi/4
    let pv = parse_hex_float(probe);
    assert_eq!(
        float_hex(pv),
        probe,
        "hex-float parser/serializer round-trip is broken"
    );

    let cases = &doc["cases"];

    // ---- unary transcendentals ----
    for case in cases["unary"].as_array().expect("unary array") {
        let name = case["name"].as_str().unwrap();
        let x = parse_hex_float(case["x"].as_str().unwrap());
        let expect = &case["expect"];

        let got = [
            ("sin", x.sin()),
            ("cos", x.cos()),
            ("sqrt", x.abs().sqrt()),
            // Mirror the generator exactly: exp(x / 256.0). 256 is a power of
            // two so the divide is exact and injects no extra rounding.
            ("exp", (x / 256.0).exp()),
            ("log", (x.abs() + 1.0).ln()),
        ];

        for (fname, actual) in got {
            let exp_hex = expect[fname].as_str().unwrap();
            let expected = parse_hex_float(exp_hex);
            let ulp = ulp_distance(actual, expected);
            checks += 1;
            if ulp != 0 {
                failures.push(format!(
                    "unary {name}.{fname}: {ulp} ULP (rust={} numpy={})",
                    float_hex(actual),
                    exp_hex
                ));
            }
        }
    }

    // ---- binary + composite ----
    for case in cases["binary"].as_array().expect("binary array") {
        let name = case["name"].as_str().unwrap();
        let y = parse_hex_float(case["y"].as_str().unwrap());
        let x = parse_hex_float(case["x"].as_str().unwrap());
        let expect = &case["expect"];

        let got = [
            ("atan2", y.atan2(x)),
            // composite: sqrt(x*x + y*y), plain ops (no FMA), matching the
            // generator and the sidereon "no FMA except mat3_vec3_mul" rule.
            ("norm2", (x * x + y * y).sqrt()),
        ];

        for (fname, actual) in got {
            let exp_hex = expect[fname].as_str().unwrap();
            let expected = parse_hex_float(exp_hex);
            let ulp = ulp_distance(actual, expected);
            checks += 1;
            if ulp != 0 {
                failures.push(format!(
                    "binary {name}.{fname}: {ulp} ULP (rust={} numpy={})",
                    float_hex(actual),
                    exp_hex
                ));
            }
        }
    }

    assert!(checks > 0, "no components were checked - fixture empty?");

    assert!(
        failures.is_empty(),
        "Rust std libm diverged from the numpy reference on {} of {checks} components:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}
