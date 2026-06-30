//! Diagnostic for the SGP4 oracle failure.
//!
//! The JSON fixture stores both the *parsed elements* (jdsatepoch, bstar,
//! ndot, etc.) and the *propagation results* (px, py, pz, vx, vy, vz) as
//! exact f64 hex bit-patterns. This test splits the validation into two
//! independent stages so we can tell whether the bug is in TLE parsing or
//! in the SGP4 algorithm itself:
//!
//!   stage A - parse the TLE in pure Rust (via twoline2rv_propagate's parser)
//!             and compare against the JSON's element bit-patterns.
//!   stage B - feed the JSON's element bit-patterns directly into
//!             vallado::sgp4_propagate (bypassing TLE parsing entirely) and
//!             compare against the position/velocity bit-patterns.
//!
//! Only stage B can tell us whether the algorithm port itself is bit-exact.

#[path = "../src/astro/sgp4/vallado.rs"]
#[allow(
    dead_code,
    unused_variables,
    unused_assignments,
    unused_mut,
    non_snake_case,
    non_camel_case_types,
    clippy::approx_constant,
    clippy::excessive_precision,
    clippy::too_many_arguments,
    clippy::needless_return,
    clippy::assign_op_pattern,
    clippy::manual_range_contains,
    clippy::collapsible_if,
    clippy::collapsible_else_if,
    clippy::float_cmp,
    clippy::needless_late_init,
    clippy::field_reassign_with_default
)]
mod vallado;

use serde_json::Value;

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
fn stage_b_direct_sgp4init_with_exact_elements() {
    // Bypass all unit conversion and TLE parsing. Pull the JSON's exact f64
    // bit-patterns for the parsed elements, call sgp4init directly with them,
    // then sgp4. If this still fails, the bug is unambiguously in the
    // sgp4init / initl / dscom / dsinit / dpper / sgp4 code path.

    let data: Value = serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();

    let mut total_checks = 0usize;
    let mut total_failures = 0usize;
    let mut worst: Vec<(String, f64, &'static str, u64)> = Vec::new();

    for sat in data["satellites"].as_array().unwrap() {
        let norad = sat["norad"].as_str().unwrap().to_string();

        // Exact element bit-patterns from the JSON.
        let jdsatepoch = hex_to_f64(sat["jdsatepoch"].as_str().unwrap());
        let jdsatepoch_f = hex_to_f64(sat["jdsatepochF"].as_str().unwrap());
        let bstar = hex_to_f64(sat["bstar"].as_str().unwrap());
        let ndot = hex_to_f64(sat["ndot"].as_str().unwrap());
        let nddot = hex_to_f64(sat["nddot"].as_str().unwrap());
        let ecco = hex_to_f64(sat["ecco"].as_str().unwrap());
        let argpo = hex_to_f64(sat["argpo"].as_str().unwrap()); // radians
        let inclo = hex_to_f64(sat["inclo"].as_str().unwrap()); // radians
        let mo = hex_to_f64(sat["mo"].as_str().unwrap()); // radians
        let no_kozai = hex_to_f64(sat["no_kozai"].as_str().unwrap()); // rad/min
        let nodeo = hex_to_f64(sat["nodeo"].as_str().unwrap()); // radians

        // SGP4 epoch as used inside sgp4init.
        let epoch_sgp4 = jdsatepoch + jdsatepoch_f - 2433281.5;

        let line1 = sat["line1"].as_str().unwrap();
        let satnum = &line1[2..7];

        let mut satrec = vallado::ElsetRec::default();
        // Mirror twoline2rv_propagate's pre-init field assignments.
        let two_digit_year: i32 = line1[18..20].trim().parse().unwrap();
        let epochdays: f64 = line1[20..32].trim().parse().unwrap();
        satrec.epochyr = two_digit_year;
        satrec.epochdays = epochdays;
        satrec.jdsatepoch = jdsatepoch;
        satrec.jdsatepochF = jdsatepoch_f;

        vallado::sgp4init(
            vallado::GravConstType::Wgs72,
            'i',
            satnum,
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
            &mut satrec,
        );

        if satrec.error != 0 {
            // Skip - fixture's prop rows for this sat will have error tags too.
            continue;
        }

        // Restore the split epoch (sgp4init/initl may have rewritten it).
        satrec.jdsatepoch = jdsatepoch;
        satrec.jdsatepochF = jdsatepoch_f;

        for prop in sat["propagations"].as_array().unwrap() {
            if prop.get("error").is_some() {
                continue;
            }
            let tsince = prop["tsince"].as_f64().unwrap();

            let mut r = [0.0_f64; 3];
            let mut v = [0.0_f64; 3];
            let ok = vallado::sgp4(&mut satrec, tsince, &mut r, &mut v);
            if !ok || satrec.error != 0 {
                satrec.error = 0;
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
                    if worst.len() < 25 || d > worst.last().unwrap().3 {
                        worst.push((norad.clone(), tsince, *label, d));
                        worst.sort_by_key(|w| std::cmp::Reverse(w.3));
                        worst.truncate(25);
                    }
                }
            }
        }
    }

    eprintln!(
        "STAGE B (direct sgp4init w/ exact element bits): {}/{} checks failed",
        total_failures, total_checks
    );
    if !worst.is_empty() {
        eprintln!("worst 25 ULP failures:");
        for (norad, tsince, label, d) in &worst {
            eprintln!("  {norad} t={tsince} {label}: {d} ULP");
        }
    }
    if total_failures == 0 {
        eprintln!(
            "\nDIAGNOSTIC: stage B CLEAN. sgp4init+sgp4 are bit-exact when fed exact element bits."
        );
    } else {
        eprintln!(
            "\nDIAGNOSTIC: stage B failed. Bug is in sgp4init / initl / dscom / dsinit / dpper / sgp4."
        );
    }
}

#[test]
fn stage_a_tle_parser_against_json_elements_all_sats() {
    // For every satellite in the fixture, walk the Rust TLE parser the same
    // way `twoline2rv_propagate` does internally and compare every parsed
    // element to the JSON's bit-patterns. Reports any divergence and which
    // field caused it.
    let data: Value = serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();

    let mut total_failures = 0usize;
    let mut sats_with_failures = 0usize;

    for sat in data["satellites"].as_array().unwrap() {
        let norad = sat["norad"].as_str().unwrap().to_string();
        let line1 = sat["line1"].as_str().unwrap();
        let line2 = sat["line2"].as_str().unwrap();

        // ── jdsatepoch / jdsatepochF (from line1) ─────────────────────
        let two_digit_year: i32 = line1[18..20].trim().parse().unwrap();
        let epochdays: f64 = line1[20..32].trim().parse().unwrap();
        let year_full = if two_digit_year < 57 {
            two_digit_year + 2000
        } else {
            two_digit_year + 1900
        };
        let (mon, day, hr, minute, sec) = vallado::days2mdhms_SGP4(year_full, epochdays);
        let (jd, jdfrac_raw) = vallado::jday_SGP4(year_full, mon, day, hr, minute, sec);
        let jdfrac = (jdfrac_raw * 100_000_000.0).round() / 100_000_000.0;

        // ── element bits - replicate twoline2rv_propagate's parse path ─
        let deg2rad = std::f64::consts::PI / 180.0;
        let xpdotp = 1440.0 / (2.0 * std::f64::consts::PI);

        let ndot_raw: f64 = line1[33..43].trim().parse().unwrap();
        let nddot_str = format!("{}.{}", &line1[44..45], &line1[45..50]);
        let nddot_mantissa: f64 = nddot_str.trim().parse().unwrap_or(0.0);
        let nexp: i32 = line1[50..52].trim().parse().unwrap_or(0);
        let bstar_str = format!("{}.{}", &line1[53..54], &line1[54..59]);
        let bstar_mantissa: f64 = bstar_str.trim().parse().unwrap_or(0.0);
        let ibexp: i32 = line1[59..61].trim().parse().unwrap_or(0);

        let inclo_deg: f64 = line2[8..16].trim().parse().unwrap();
        let nodeo_deg: f64 = line2[17..25].trim().parse().unwrap();
        let ecco_str = format!("0.{}", line2[26..33].replace(' ', "0"));
        let ecco: f64 = ecco_str.parse().unwrap();
        let argpo_deg: f64 = line2[34..42].trim().parse().unwrap();
        let mo_deg: f64 = line2[43..51].trim().parse().unwrap();
        let no_kozai_revday: f64 = line2[52..63].trim().parse().unwrap();

        let no_kozai = no_kozai_revday / xpdotp;
        let nddot = nddot_mantissa * 10.0_f64.powi(nexp) / (xpdotp * 1440.0 * 1440.0);
        let bstar = bstar_mantissa * 10.0_f64.powi(ibexp);
        let ndot = ndot_raw / (xpdotp * 1440.0);
        let inclo = inclo_deg * deg2rad;
        let nodeo = nodeo_deg * deg2rad;
        let argpo = argpo_deg * deg2rad;
        let mo = mo_deg * deg2rad;

        let parsed = [
            (
                "jdsatepoch",
                jd,
                hex_to_f64(sat["jdsatepoch"].as_str().unwrap()),
            ),
            (
                "jdsatepochF",
                jdfrac,
                hex_to_f64(sat["jdsatepochF"].as_str().unwrap()),
            ),
            ("bstar", bstar, hex_to_f64(sat["bstar"].as_str().unwrap())),
            ("ndot", ndot, hex_to_f64(sat["ndot"].as_str().unwrap())),
            ("nddot", nddot, hex_to_f64(sat["nddot"].as_str().unwrap())),
            ("ecco", ecco, hex_to_f64(sat["ecco"].as_str().unwrap())),
            ("argpo", argpo, hex_to_f64(sat["argpo"].as_str().unwrap())),
            ("inclo", inclo, hex_to_f64(sat["inclo"].as_str().unwrap())),
            ("mo", mo, hex_to_f64(sat["mo"].as_str().unwrap())),
            (
                "no_kozai",
                no_kozai,
                hex_to_f64(sat["no_kozai"].as_str().unwrap()),
            ),
            ("nodeo", nodeo, hex_to_f64(sat["nodeo"].as_str().unwrap())),
        ];

        let mut sat_failures = 0;
        for (name, ours, expected) in parsed {
            let d = ulp(ours, expected);
            if d > 0 {
                if sat_failures == 0 {
                    eprintln!("[{norad}]");
                }
                eprintln!(
                    "  DIFF {name:<12} ours=0x{:016x}  expected=0x{:016x}  ulp={d}",
                    ours.to_bits(),
                    expected.to_bits()
                );
                sat_failures += 1;
                total_failures += 1;
            }
        }
        if sat_failures > 0 {
            sats_with_failures += 1;
        }
    }

    eprintln!(
        "\nSTAGE A: {total_failures} parser-element divergences across {sats_with_failures} satellites"
    );
}
