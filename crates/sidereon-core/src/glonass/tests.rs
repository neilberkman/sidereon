//! 0-ULP parity tests for the GLONASS RK4 state-vector propagator.
//!
//! These assert the Rust port reproduces the canonical reference recipe
//! `parity/generator/glonass_eval.py` bit-for-bit, using the committed golden
//! `parity/fixtures/glonass_golden.json` (vendored at `tests/fixtures/`). Values
//! are hex-float (Python `float.hex()`) and parity is measured as ULP distance on
//! the IEEE-754 bit pattern. The golden carries the state after every RK4 step,
//! so a divergence is localised to a single integration step, not just the
//! output. The integration direction/step branches (tk = 0, one step, partial
//! step, multi-step, backward, the +15 min edge) are all exercised.

use std::path::PathBuf;

use serde_json::Value;

use super::{clock_offset_s, deq, glorbit, propagate};

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
        .unwrap_or_else(|_| panic!("bad exponent in {s:?}"));
    let (int_part, frac_part) = match mantissa.split_once('.') {
        Some((i, f)) => (i, f),
        None => (mantissa, ""),
    };
    let int_val = i64::from_str_radix(int_part, 16)
        .unwrap_or_else(|_| panic!("bad integer hex digits in {s:?}")) as f64;
    let mut frac_val = 0.0f64;
    let mut scale = 1.0f64 / 16.0;
    for c in frac_part.chars() {
        let d = c
            .to_digit(16)
            .unwrap_or_else(|| panic!("bad hex frac digit {c:?} in {s:?}"));
        frac_val += (d as f64) * scale;
        scale /= 16.0;
    }
    let val = (int_val + frac_val) * 2.0f64.powi(exp2);
    if neg {
        -val
    } else {
        val
    }
}

fn ordered_i64(x: f64) -> i64 {
    let bits = x.to_bits() as i64;
    if bits < 0 {
        i64::MIN - bits
    } else {
        bits
    }
}

fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    ordered_i64(a).abs_diff(ordered_i64(b))
}

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/glonass_golden.json")
        .canonicalize()
        .unwrap_or_else(|e| panic!("cannot locate tests/fixtures/glonass_golden.json: {e}"))
}

fn hexf(v: &Value, key: &str) -> f64 {
    parse_hex_float(
        v[key]
            .as_str()
            .unwrap_or_else(|| panic!("missing/non-string {key}")),
    )
}

fn hex_array(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|e| parse_hex_float(e.as_str().unwrap()))
        .collect()
}

#[test]
fn glonass_propagation_zero_ulp() {
    let raw = std::fs::read_to_string(fixture_path()).expect("read glonass_golden.json");
    let doc: Value = serde_json::from_str(&raw).expect("parse glonass_golden.json");

    // The recipe's constants must match the port's constants bit-for-bit (so the
    // integrator the golden was produced with is the one under test).
    let c = &doc["constants"];
    assert_eq!(parse_hex_float(c["mu"].as_str().unwrap()), super::MU, "MU");
    assert_eq!(parse_hex_float(c["j2"].as_str().unwrap()), super::J2, "J2");
    assert_eq!(
        parse_hex_float(c["omega_e"].as_str().unwrap()),
        super::OMEGA_E,
        "OMEGA_E"
    );
    assert_eq!(
        parse_hex_float(c["r_e"].as_str().unwrap()),
        super::R_E,
        "R_E"
    );
    assert_eq!(
        parse_hex_float(c["tstep_s"].as_str().unwrap()),
        super::TSTEP_S,
        "TSTEP_S"
    );

    let cases = doc["cases"].as_array().expect("cases array");
    assert!(
        cases.len() >= 6,
        "expected the full branch matrix, found {}",
        cases.len()
    );

    let mut failures: Vec<String> = Vec::new();
    let mut checks = 0usize;

    for case in cases {
        let name = case["name"].as_str().unwrap_or("<unnamed>");
        let inp = &case["inputs"];
        let exp = &case["expect"];

        let pos = hex_array(&inp["pos_m"]);
        let vel = hex_array(&inp["vel_m_s"]);
        let acc_v = hex_array(&inp["acc_m_s2"]);
        let acc = [acc_v[0], acc_v[1], acc_v[2]];
        let state0 = [pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]];
        let tk = hexf(inp, "tk_s");

        let mut check = |label: String, got: f64, want: f64| {
            let ulp = ulp_distance(got, want);
            checks += 1;
            if ulp != 0 {
                failures.push(format!("{label}: {ulp} ULP (rust={got:e} ref={want:e})"));
            }
        };

        // Replay each RK4 step with the golden's step size, asserting the state
        // after every step (localises a divergence to one step).
        let mut state = state0;
        for (si, step_val) in exp["steps"].as_array().expect("steps").iter().enumerate() {
            let step = parse_hex_float(step_val["step_s"].as_str().unwrap());
            let want = hex_array(&step_val["state"]);
            state = glorbit(step, &state, &acc);
            for k in 0..6 {
                check(format!("{name}.step{si}.s{k}"), state[k], want[k]);
            }
        }

        // The full propagation (including the step-size policy) lands on the
        // final state.
        let final_want = hex_array(&exp["final_state"]);
        let final_got = propagate(state0, acc, tk).expect("valid GLONASS propagation");
        for k in 0..6 {
            check(format!("{name}.final.s{k}"), final_got[k], final_want[k]);
        }

        // The clock offset.
        let clk = clock_offset_s(hexf(inp, "clk_bias"), hexf(inp, "gamma_n"), tk);
        check(format!("{name}.clock"), clk, hexf(exp, "clock_offset_s"));
    }

    assert!(checks > 0, "no components checked - fixture empty?");
    assert!(
        failures.is_empty(),
        "GLONASS port diverged from the reference recipe on {} of {checks} components:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

#[test]
fn deq_is_a_pure_derivative() {
    // A sanity check independent of the golden: at a circular-orbit-ish state the
    // derivative's first three components are the velocity, and the acceleration
    // points roughly inward (negative radial), so the integrator is integrating
    // the right thing.
    let s = [
        10_908_942.0,
        -2_885_726.0,
        22_883_539.0,
        1407.8,
        2795.9,
        -317.0,
    ];
    let d = deq(&s, &[0.0, 0.0, 0.0]);
    assert_eq!(
        [d[0], d[1], d[2]],
        [s[3], s[4], s[5]],
        "first half is velocity"
    );
    let radial = d[3] * s[0] + d[4] * s[1] + d[5] * s[2];
    assert!(
        radial < 0.0,
        "acceleration has an inward radial component: {radial}"
    );
}
