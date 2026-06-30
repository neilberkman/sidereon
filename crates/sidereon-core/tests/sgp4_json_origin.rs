//! Determines whether the JSON fixture's propagation bit-patterns came from
//! an FMA-enabled or non-FMA C++ build. We compute one satellite's t=0
//! position with our non-FMA C++ build and compare to the JSON bits.
//!
//!   match  → JSON is non-FMA, the divergence is from another bug.
//!   nomatch → JSON is FMA-based, and our pure-Rust port (which has the
//!             "do not use mul_add" rule) cannot bit-match it. We must
//!             regenerate the fixture from our non-FMA reference.

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

fn ulp(a: f64, b: f64) -> u64 {
    let ia = a.to_bits() as i64;
    let ib = b.to_bits() as i64;
    (ia - ib).unsigned_abs()
}

#[test]
fn nonfma_cpp_vs_json_fixture() {
    let _keep = force_link_oracle();
    let data: Value = serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();

    let mut total_checks = 0usize;
    let mut total_failures = 0usize;
    let mut worst: Vec<(String, f64, &'static str, u64)> = Vec::new();

    for sat in data["satellites"].as_array().unwrap() {
        let norad = sat["norad"].as_str().unwrap().to_string();

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

        let line1 = sat["line1"].as_str().unwrap();
        let satnum_str = &line1[2..7];
        let two_digit_year: i32 = line1[18..20].trim().parse().unwrap();
        let epochdays: f64 = line1[20..32].trim().parse().unwrap();
        let satnum_c = std::ffi::CString::new(satnum_str).unwrap();

        for prop in sat["propagations"].as_array().unwrap() {
            if prop.get("error").is_some() {
                continue;
            }
            let tsince = prop["tsince"].as_f64().unwrap();
            let mut r = [0.0f64; 3];
            let mut v = [0.0f64; 3];
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
            if err != 0 {
                continue;
            }

            let labels = ["px", "py", "pz", "vx", "vy", "vz"];
            let actual = [r[0], r[1], r[2], v[0], v[1], v[2]];
            for (i, label) in labels.iter().enumerate() {
                total_checks += 1;
                let expected = hex_to_f64(prop[*label].as_str().unwrap());
                let d = ulp(actual[i], expected);
                if d > 0 {
                    total_failures += 1;
                    if worst.len() < 10 || d > worst.last().unwrap().3 {
                        worst.push((norad.clone(), tsince, *label, d));
                        worst.sort_by_key(|entry| std::cmp::Reverse(entry.3));
                        worst.truncate(10);
                    }
                }
            }
        }
    }

    eprintln!(
        "non-FMA C++ vs JSON fixture: {}/{} checks failed",
        total_failures, total_checks
    );
    for (norad, tsince, label, d) in &worst {
        eprintln!("  {norad} t={tsince} {label}: {d} ULP");
    }
    if total_failures == 0 {
        eprintln!("\nVERDICT: JSON is non-FMA. Our Rust failure is from a different bug.");
    } else {
        eprintln!(
            "\nVERDICT: JSON is FMA-based. Even our non-FMA C++ doesn't match it. We must regenerate the fixture from a non-FMA reference."
        );
    }
}
