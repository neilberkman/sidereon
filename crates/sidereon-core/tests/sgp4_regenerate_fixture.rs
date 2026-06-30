#![cfg(sidereon_repo_tests)]
//! One-shot fixture regeneration. Walks every (satellite, propagation row)
//! in the existing JSON fixture, calls our **non-FMA** C++ Vallado SGP4
//! reference, and overwrites the px/py/pz/vx/vy/vz hex bit-patterns with
//! the non-FMA results. Everything else (TLE lines, element bit-patterns,
//! tsince, error tags) is left untouched.
//!
//! After this runs, the fixture is canonically *non-FMA Vallado* - bit-exact
//! to our pure-Rust port and reproducible by anyone with `-ffp-contract=off`.
//!
//! Run with:
//!
//!   cargo test --features sgp4-debug-oracle --test sgp4_regenerate_fixture \
//!       regenerate_json_fixture_in_place -- --nocapture --ignored

#![cfg(feature = "sgp4-debug-oracle")]

use serde_json::Value;
use sidereon_core::astro::sgp4_cpp_oracle::{cpp_sgp4_step, force_link_oracle};
use std::os::raw::{c_char, c_int};

fn hex_to_f64(s: &str) -> f64 {
    let (neg, rest) = if let Some(r) = s.strip_prefix("-0x") {
        (true, r)
    } else if let Some(r) = s.strip_prefix("0x") {
        (false, r)
    } else {
        panic!("bad hex float: {s}");
    };
    let (mant_str, exp_str) = rest.split_once('p').unwrap();
    let (int_part, frac_part) = mant_str.split_once('.').unwrap();
    let exp: i32 = exp_str.parse().unwrap();
    let full = format!("{int_part}{frac_part}");
    let mant = u64::from_str_radix(&full, 16).unwrap();
    let frac_bits = frac_part.len() as i32 * 4;
    let val = mant as f64 * (2.0_f64).powi(exp - frac_bits);
    if neg {
        -val
    } else {
        val
    }
}

/// Format an f64 as a C99-style hex float literal, matching the existing
/// fixture format: `[-]0x1.HHHHHHHHHHHHHpEEE` (13 hex digits of mantissa,
/// signed exponent), or `0x0.0p+0` for zero, or the subnormal/inf/nan form.
fn f64_to_hex(x: f64) -> String {
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0x0.0p+0".to_string()
        } else {
            "0x0.0p+0".to_string()
        };
    }
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x.is_sign_negative() {
            "-inf".to_string()
        } else {
            "inf".to_string()
        };
    }

    let bits = x.to_bits();
    let sign = bits >> 63;
    let exp_bits = ((bits >> 52) & 0x7ff) as i32;
    let mant_bits = bits & ((1u64 << 52) - 1);
    let sign_str = if sign == 1 { "-" } else { "" };

    if exp_bits == 0 {
        // Subnormal: leading 0., exponent is the smallest normal exponent.
        format!("{}0x0.{:013x}p{:+}", sign_str, mant_bits, -1022)
    } else {
        let exp_unbiased = exp_bits - 1023;
        format!("{}0x1.{:013x}p{:+}", sign_str, mant_bits, exp_unbiased)
    }
}

#[test]
fn hex_format_roundtrip() {
    // Self-check the formatter on values pulled directly from the fixture.
    // We can't use Rust hex float literals (they're nightly-only), so we
    // round-trip via the parser.
    let samples = [
        "0x1.b6e771d6b8739p+12",
        "-0x1.5e054f5724496p+10",
        "0x1.47487b5210918p-5",
        "0x0.0p+0",
        "0x1.2b48540000000p+21",
    ];
    for s in samples {
        let parsed = hex_to_f64(s);
        let reformatted = f64_to_hex(parsed);
        let reparsed = hex_to_f64(&reformatted);
        assert_eq!(
            reparsed.to_bits(),
            parsed.to_bits(),
            "roundtrip mismatch: {s} → parsed bits=0x{:016x} → reformatted={reformatted} → reparsed bits=0x{:016x}",
            parsed.to_bits(),
            reparsed.to_bits()
        );
    }
}

#[test]
#[ignore = "rewrites tests/sgp4_verification.json - run explicitly"]
fn regenerate_json_fixture_in_place() {
    let _keep = force_link_oracle();

    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let path = std::path::PathBuf::from(&manifest).join("tests/sgp4_verification.json");
    eprintln!("loading {}", path.display());

    let raw = std::fs::read_to_string(&path).expect("read fixture");
    let mut data: Value = serde_json::from_str(&raw).expect("parse fixture");

    let mut sat_count = 0;
    let mut row_count = 0;
    let mut error_row_count = 0;
    let mut bits_changed = 0;

    let satellites = data["satellites"].as_array_mut().expect("satellites array");
    for sat in satellites.iter_mut() {
        sat_count += 1;
        let norad = sat["norad"].as_str().unwrap().to_string();

        // Pull the JSON's exact element bits as inputs to the C++.
        let jdsatepoch = hex_to_f64(sat["jdsatepoch"].as_str().unwrap());
        let jdsatepoch_f = hex_to_f64(sat["jdsatepochF"].as_str().unwrap());
        let bstar = hex_to_f64(sat["bstar"].as_str().unwrap());
        let ndot = hex_to_f64(sat["ndot"].as_str().unwrap());
        let nddot = hex_to_f64(sat["nddot"].as_str().unwrap());
        let ecco = hex_to_f64(sat["ecco"].as_str().unwrap());
        let argpo = hex_to_f64(sat["argpo"].as_str().unwrap());
        let inclo = hex_to_f64(sat["inclo"].as_str().unwrap());
        let mo = hex_to_f64(sat["mo"].as_str().unwrap());
        let no_kozai = hex_to_f64(sat["no_kozai"].as_str().unwrap());
        let nodeo = hex_to_f64(sat["nodeo"].as_str().unwrap());

        let epoch_sgp4 = jdsatepoch + jdsatepoch_f - 2433281.5;

        let line1 = sat["line1"].as_str().unwrap().to_string();
        let satnum_str: String = line1.chars().skip(2).take(5).collect();
        let two_digit_year: i32 = line1[18..20].trim().parse().unwrap();
        let epochdays: f64 = line1[20..32].trim().parse().unwrap();
        let satnum_c = std::ffi::CString::new(satnum_str.as_str()).unwrap();

        let props = sat["propagations"].as_array_mut().expect("propagations");
        for prop in props.iter_mut() {
            row_count += 1;
            if prop.get("error").is_some() {
                error_row_count += 1;
                continue;
            }
            let tsince = prop["tsince"].as_f64().unwrap();
            let mut r = [0.0_f64; 3];
            let mut v = [0.0_f64; 3];
            let err = unsafe {
                cpp_sgp4_step(
                    satnum_c.as_ptr(),
                    b'i' as c_char,
                    epoch_sgp4,
                    bstar,
                    ndot,
                    nddot,
                    ecco,
                    argpo,
                    inclo,
                    mo,
                    no_kozai,
                    nodeo,
                    two_digit_year as c_int,
                    epochdays,
                    jdsatepoch,
                    jdsatepoch_f,
                    tsince,
                    r.as_mut_ptr(),
                    v.as_mut_ptr(),
                )
            };
            assert_eq!(err, 0, "C++ sgp4 returned err {err} for {norad} t={tsince}");

            for (i, label) in ["px", "py", "pz", "vx", "vy", "vz"].iter().enumerate() {
                let new_val = if i < 3 { r[i] } else { v[i - 3] };
                let new_str = f64_to_hex(new_val);
                let old_str = prop[*label].as_str().unwrap().to_string();
                if old_str != new_str {
                    bits_changed += 1;
                }
                prop[*label] = Value::String(new_str);
            }
        }
    }

    eprintln!(
        "satellites={sat_count}  prop rows={row_count}  error rows={error_row_count}  bit-pattern fields rewritten={bits_changed}"
    );

    // Pretty-print with the same 2-space indent as the original.
    let mut out = serde_json::to_string_pretty(&data).expect("serialize");
    out.push('\n');
    std::fs::write(&path, out).expect("write fixture");

    eprintln!("wrote regenerated fixture to {}", path.display());
}
