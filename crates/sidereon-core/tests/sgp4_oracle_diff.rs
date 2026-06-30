//! Field-level diff between our pure-Rust `vallado::sgp4init` output and the
//! Vallado C++ reference. Gated behind the `sgp4-debug-oracle` feature so the
//! C++ build never runs in normal `cargo test`.
//!
//! Run with:
//!
//!   cargo test --features sgp4-debug-oracle --test sgp4_oracle_diff -- --nocapture
//!
//! The first satellite checked is NORAD 04632 (the worst Stage-B offender).
//! When a divergence is found, every subsequent field that differs is also
//! reported so we can see whether the bug is localized to one variable or
//! cascades from a single root.

#![cfg(feature = "sgp4-debug-oracle")]

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
use sidereon_core::astro::sgp4_cpp_oracle::{
    cpp_sgp4_step, cpp_sgp4init_dump, force_link_oracle, CPP_DUMP_DOUBLE_COUNT, CPP_DUMP_INT_COUNT,
};
use std::os::raw::{c_char, c_int};

/// Field-name list. MUST match the order of stores in `cpp_sgp4init_dump`
/// inside `tests/cpp/sgp4_dump_wrapper.cpp`.
const FIELDS_F64: &[&str] = &[
    // Near-Earth (25)
    "aycof",
    "con41",
    "cc1",
    "cc4",
    "cc5",
    "d2",
    "d3",
    "d4",
    "delmo",
    "eta",
    "argpdot",
    "omgcof",
    "sinmao",
    "t",
    "t2cof",
    "t3cof",
    "t4cof",
    "t5cof",
    "x1mth2",
    "x7thm1",
    "mdot",
    "nodedot",
    "xlcof",
    "xmcof",
    "nodecf",
    // Deep-space (55)
    "d2201",
    "d2211",
    "d3210",
    "d3222",
    "d4410",
    "d4422",
    "d5220",
    "d5232",
    "d5421",
    "d5433",
    "dedt",
    "del1",
    "del2",
    "del3",
    "didt",
    "dmdt",
    "dnodt",
    "domdt",
    "e3",
    "ee2",
    "peo",
    "pgho",
    "pho",
    "pinco",
    "plo",
    "se2",
    "se3",
    "sgh2",
    "sgh3",
    "sgh4",
    "sh2",
    "sh3",
    "si2",
    "si3",
    "sl2",
    "sl3",
    "sl4",
    "gsto",
    "xfact",
    "xgh2",
    "xgh3",
    "xgh4",
    "xh2",
    "xh3",
    "xi2",
    "xi3",
    "xl2",
    "xl3",
    "xl4",
    "xlamo",
    "zmol",
    "zmos",
    "atime",
    "xli",
    "xni",
    // Element / state (16)
    "a",
    "altp",
    "alta",
    "epochdays",
    "jdsatepoch",
    "jdsatepochF",
    "nddot",
    "ndot",
    "bstar",
    "rcse",
    "inclo",
    "nodeo",
    "ecco",
    "argpo",
    "mo",
    "no_kozai",
    // Singly-averaged + unkozai (8)
    "no_unkozai",
    "am",
    "em",
    "im",
    "Om",
    "om",
    "mm",
    "nm",
    // Constants (8)
    "tumin",
    "mus",
    "radiusearthkm",
    "xke",
    "j2",
    "j3",
    "j4",
    "j3oj2",
];

const FIELDS_I32: &[&str] = &["epochyr", "epochtynumrev", "error", "isimp", "irez"];

fn dump_rust(
    satrec: &vallado::ElsetRec,
) -> ([f64; CPP_DUMP_DOUBLE_COUNT], [i32; CPP_DUMP_INT_COUNT]) {
    let f = [
        // Near-Earth
        satrec.aycof,
        satrec.con41,
        satrec.cc1,
        satrec.cc4,
        satrec.cc5,
        satrec.d2,
        satrec.d3,
        satrec.d4,
        satrec.delmo,
        satrec.eta,
        satrec.argpdot,
        satrec.omgcof,
        satrec.sinmao,
        satrec.t,
        satrec.t2cof,
        satrec.t3cof,
        satrec.t4cof,
        satrec.t5cof,
        satrec.x1mth2,
        satrec.x7thm1,
        satrec.mdot,
        satrec.nodedot,
        satrec.xlcof,
        satrec.xmcof,
        satrec.nodecf,
        // Deep-space
        satrec.d2201,
        satrec.d2211,
        satrec.d3210,
        satrec.d3222,
        satrec.d4410,
        satrec.d4422,
        satrec.d5220,
        satrec.d5232,
        satrec.d5421,
        satrec.d5433,
        satrec.dedt,
        satrec.del1,
        satrec.del2,
        satrec.del3,
        satrec.didt,
        satrec.dmdt,
        satrec.dnodt,
        satrec.domdt,
        satrec.e3,
        satrec.ee2,
        satrec.peo,
        satrec.pgho,
        satrec.pho,
        satrec.pinco,
        satrec.plo,
        satrec.se2,
        satrec.se3,
        satrec.sgh2,
        satrec.sgh3,
        satrec.sgh4,
        satrec.sh2,
        satrec.sh3,
        satrec.si2,
        satrec.si3,
        satrec.sl2,
        satrec.sl3,
        satrec.sl4,
        satrec.gsto,
        satrec.xfact,
        satrec.xgh2,
        satrec.xgh3,
        satrec.xgh4,
        satrec.xh2,
        satrec.xh3,
        satrec.xi2,
        satrec.xi3,
        satrec.xl2,
        satrec.xl3,
        satrec.xl4,
        satrec.xlamo,
        satrec.zmol,
        satrec.zmos,
        satrec.atime,
        satrec.xli,
        satrec.xni,
        // Element / state
        satrec.a,
        satrec.altp,
        satrec.alta,
        satrec.epochdays,
        satrec.jdsatepoch,
        satrec.jdsatepochF,
        satrec.nddot,
        satrec.ndot,
        satrec.bstar,
        satrec.rcse,
        satrec.inclo,
        satrec.nodeo,
        satrec.ecco,
        satrec.argpo,
        satrec.mo,
        satrec.no_kozai,
        // Singly-averaged + unkozai
        satrec.no_unkozai,
        satrec.am,
        satrec.em,
        satrec.im,
        satrec.Om,
        satrec.om,
        satrec.mm,
        satrec.nm,
        // Constants
        satrec.tumin,
        satrec.mus,
        satrec.radiusearthkm,
        satrec.xke,
        satrec.j2,
        satrec.j3,
        satrec.j4,
        satrec.j3oj2,
    ];
    let i = [
        satrec.epochyr,
        satrec.epochtynumrev,
        satrec.error,
        satrec.isimp,
        satrec.irez,
    ];
    (f, i)
}

fn ulp(a: f64, b: f64) -> u64 {
    let ia = a.to_bits() as i64;
    let ib = b.to_bits() as i64;
    (ia - ib).unsigned_abs()
}

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

fn run_satellite(norad_target: &str, opsmode: char) -> usize {
    // Touch the oracle symbols so the linker keeps the C++ static lib.
    let _keep = force_link_oracle();
    assert_eq!(FIELDS_F64.len(), CPP_DUMP_DOUBLE_COUNT);
    assert_eq!(FIELDS_I32.len(), CPP_DUMP_INT_COUNT);

    let data: Value = serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();
    let sat = data["satellites"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["norad"].as_str() == Some(norad_target))
        .unwrap_or_else(|| panic!("{norad_target} not in fixture"));

    // Pull exact element bits from JSON.
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

    // ── Run the C++ ────────────────────────────────────────────────
    let mut cpp_doubles = [0.0_f64; CPP_DUMP_DOUBLE_COUNT];
    let mut cpp_ints = [0_i32; CPP_DUMP_INT_COUNT];
    let satnum_c = std::ffi::CString::new(satnum_str).unwrap();
    let cpp_err = unsafe {
        cpp_sgp4init_dump(
            satnum_c.as_ptr(),
            opsmode as u8 as c_char,
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
            cpp_doubles.as_mut_ptr(),
            cpp_ints.as_mut_ptr(),
        )
    };
    eprintln!("[{norad_target}] (opsmode '{opsmode}') cpp sgp4init returned error code {cpp_err}");

    // ── Run our Rust port ─────────────────────────────────────────
    let mut satrec = vallado::ElsetRec {
        epochyr: two_digit_year,
        epochdays,
        jdsatepoch,
        jdsatepochF: jdsatepoch_f,
        ..vallado::ElsetRec::default()
    };
    vallado::sgp4init(
        vallado::GravConstType::Wgs72,
        opsmode,
        satnum_str,
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
    satrec.jdsatepoch = jdsatepoch;
    satrec.jdsatepochF = jdsatepoch_f;
    eprintln!(
        "[{norad_target}] rust sgp4init returned error code {}",
        satrec.error
    );

    let (rust_doubles, rust_ints) = dump_rust(&satrec);

    // ── Diff ──────────────────────────────────────────────────────
    let mut first_div: Option<usize> = None;
    let mut diff_count = 0usize;

    eprintln!("\n── i32 fields ──");
    for (i, name) in FIELDS_I32.iter().enumerate() {
        let r = rust_ints[i];
        let c = cpp_ints[i];
        if r != c {
            diff_count += 1;
            eprintln!("  DIFF {name}: rust={r}  cpp={c}");
        }
    }

    eprintln!("\n── f64 fields (showing only divergent + 5 surrounding before first diff) ──");
    for (i, name) in FIELDS_F64.iter().enumerate() {
        let r = rust_doubles[i];
        let c = cpp_doubles[i];
        let d = ulp(r, c);
        if d != 0 {
            if first_div.is_none() {
                first_div = Some(i);
                let lo = i.saturating_sub(5);
                eprintln!("  ── context: 5 fields preceding first divergence ──");
                for j in lo..i {
                    eprintln!(
                        "    OK   [{:>3}] {:<14} = 0x{:016x}",
                        j,
                        FIELDS_F64[j],
                        rust_doubles[j].to_bits()
                    );
                }
                eprintln!("  ── divergences ──");
            }
            diff_count += 1;
            eprintln!(
                "    DIFF [{:>3}] {:<14} rust=0x{:016x} ({:+.17e})  cpp=0x{:016x} ({:+.17e})  ulp={}",
                i,
                name,
                r.to_bits(),
                r,
                c.to_bits(),
                c,
                d
            );
        }
    }

    eprintln!(
        "\n[{norad_target}] total divergent fields: {diff_count} / {}",
        FIELDS_F64.len() + FIELDS_I32.len()
    );
    if let Some(idx) = first_div {
        eprintln!(
            "[{norad_target}] FIRST f64 divergence at field index {idx}: {}",
            FIELDS_F64[idx]
        );
    } else if diff_count == 0 {
        eprintln!("[{norad_target}] CLEAN - no satrec divergence after sgp4init.");
    }

    diff_count
}

#[test]
fn diff_norad_04632_worst_offender() {
    run_satellite("04632", 'i');
}

#[test]
fn diff_norad_28129() {
    run_satellite("28129", 'i');
}

#[test]
fn diff_norad_88888_analyst() {
    run_satellite("88888", 'i');
}

/// Run the full sgp4init+sgp4 r/v step diff (Rust vs C++ reference) over every
/// satellite/propagation in the fixture, for a given `opsmode`. Returns
/// `(total_failures, total_checks)`. The opsmode is passed identically to both
/// the Rust port and the C++ reference, so a 0-ULP result proves the Rust port
/// is bit-exact to the reference *for that mode*.
fn step_diff_for_opsmode(opsmode: char) -> (usize, usize) {
    let _keep = force_link_oracle();
    let data: Value = serde_json::from_str(include_str!("sgp4_verification.json")).unwrap();

    let mut total_checks = 0usize;
    let mut total_failures = 0usize;
    let mut worst: Vec<(String, f64, &'static str, u64)> = Vec::new();

    for sat in data["satellites"].as_array().unwrap() {
        let norad = sat["norad"].as_str().unwrap().to_string();
        let line1 = sat["line1"].as_str().unwrap();

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
        let satnum_str = &line1[2..7];
        let two_digit_year: i32 = line1[18..20].trim().parse().unwrap();
        let epochdays: f64 = line1[20..32].trim().parse().unwrap();
        let satnum_c = std::ffi::CString::new(satnum_str).unwrap();

        // ── Rust path: init + step ─────────────────────────────────────
        let mut satrec = vallado::ElsetRec {
            epochyr: two_digit_year,
            epochdays,
            jdsatepoch,
            jdsatepochF: jdsatepoch_f,
            ..vallado::ElsetRec::default()
        };
        vallado::sgp4init(
            vallado::GravConstType::Wgs72,
            opsmode,
            satnum_str,
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
            continue;
        }
        satrec.jdsatepoch = jdsatepoch;
        satrec.jdsatepochF = jdsatepoch_f;

        for prop in sat["propagations"].as_array().unwrap() {
            if prop.get("error").is_some() {
                continue;
            }
            let tsince = prop["tsince"].as_f64().unwrap();

            // Rust step
            let mut rust_satrec = satrec.clone();
            let mut rr = [0.0_f64; 3];
            let mut rv = [0.0_f64; 3];
            vallado::sgp4(&mut rust_satrec, tsince, &mut rr, &mut rv);
            if rust_satrec.error != 0 {
                continue;
            }

            // C++ step (independent init+step)
            let mut cr = [0.0_f64; 3];
            let mut cv = [0.0_f64; 3];
            let cerr = unsafe {
                cpp_sgp4_step(
                    satnum_c.as_ptr(),
                    opsmode as u8 as c_char,
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
                    cr.as_mut_ptr(),
                    cv.as_mut_ptr(),
                )
            };
            if cerr != 0 {
                continue;
            }

            let labels = ["px", "py", "pz", "vx", "vy", "vz"];
            let actual = [rr[0], rr[1], rr[2], rv[0], rv[1], rv[2]];
            let expected = [cr[0], cr[1], cr[2], cv[0], cv[1], cv[2]];

            for (i, label) in labels.iter().enumerate() {
                total_checks += 1;
                let d = ulp(actual[i], expected[i]);
                if d > 0 {
                    total_failures += 1;
                    if worst.len() < 15 || d > worst.last().unwrap().3 {
                        worst.push((norad.clone(), tsince, *label, d));
                        worst.sort_by_key(|entry| std::cmp::Reverse(entry.3));
                        worst.truncate(15);
                    }
                }
            }
        }
    }

    eprintln!(
        "[opsmode '{opsmode}'] Rust sgp4 step vs non-FMA C++ sgp4 step: {total_failures}/{total_checks} failures"
    );
    for (norad, tsince, label, d) in &worst {
        eprintln!("  {norad} t={tsince} {label}: {d} ULP");
    }
    if total_failures == 0 {
        eprintln!(
            "\n[opsmode '{opsmode}'] CLEAN - Rust sgp4 step is bit-exact to non-FMA C++ sgp4 step."
        );
    } else {
        eprintln!(
            "\n[opsmode '{opsmode}'] DIVERGENCE - bug is inside the sgp4() propagation kernel."
        );
    }

    (total_failures, total_checks)
}

#[test]
fn diff_sgp4_step_at_nonzero_tsince() {
    // Both Rust sgp4init+sgp4 and C++ sgp4init+sgp4 should produce identical
    // r/v vectors when called with bit-identical inputs (after the FMA fix
    // on the C++ side). If they don't, the divergence is in the sgp4 step
    // itself, not in sgp4init.  Existing opsmode 'i' (improved) contract.
    let (failures, checks) = step_diff_for_opsmode('i');
    assert!(checks > 0, "no propagations were checked");
    assert_eq!(
        failures, 0,
        "opsmode 'i' Rust sgp4 step is not bit-exact (0 ULP) to the C++ reference"
    );
}

#[test]
fn diff_sgp4_step_at_nonzero_tsince_afspc() {
    // AFSPC opsmode 'a' oracle pass: Rust sgp4init+sgp4 with opsmode 'a' must be
    // bit-exact (0 ULP) to the SAME Vallado C++ reference also driven with
    // opsmode 'a'. The deep-space / low-inclination sats in the fixture exercise
    // the dpper nodep-wrap branch that distinguishes 'a' from 'i', so this is a
    // meaningful AFSPC fidelity check, not a trivial near-Earth pass-through.
    let (failures, checks) = step_diff_for_opsmode('a');
    assert!(checks > 0, "no propagations were checked");
    assert_eq!(
        failures, 0,
        "AFSPC opsmode 'a' Rust sgp4 step is not bit-exact (0 ULP) to the C++ reference"
    );
}

#[test]
fn diff_norad_25544_iss_clean_baseline() {
    // ISS-class LEO. Should be clean since near-Earth sats had few failures.
    // If it's also clean field-by-field, that confirms the bug is in
    // deep-space init, not initl.
    run_satellite("00005", 'i');
}

/// AFSPC opsmode 'a' sgp4init field-level oracle diff. For the deep-space,
/// low-inclination satellites that actually exercise the opsmode-'a' nodep wrap
/// in dpper, every initialized satrec field must match the C++ reference (also
/// run with 'a') at 0 ULP. This proves the AFSPC init path field-by-field, the
/// same way the 'i' diagnostics do for improved mode.
#[test]
fn afspc_opsmode_a_sgp4init_field_diff() {
    // Deep-space, low-inclination sats from the fixture — these are where
    // opsmode 'a' diverges from 'i' (geosync / Molniya-class resonance sats).
    let deep_low_incl = [
        "04632", "09998", "14128", "20413", "23177", "23599", "24208", "25954", "26900", "28626",
        "33335",
    ];
    let mut total_field_diffs = 0usize;
    for norad in deep_low_incl {
        let d = run_satellite(norad, 'a');
        total_field_diffs += d;
    }
    assert_eq!(
        total_field_diffs, 0,
        "AFSPC opsmode 'a' sgp4init satrec fields are not bit-exact (0 ULP) to the C++ reference"
    );
}
