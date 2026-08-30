//! 0-ULP parity for broadcast-ephemeris orbit/clock evaluation.
//!
//! The committed fixture (`tests/fixtures/broadcast_golden.json`) records, per
//! case, the broadcast elements and clock terms plus every load-bearing
//! intermediate, all as raw IEEE-754 bit patterns. Each `f64` is rebuilt from
//! its bits and the Rust evaluation is asserted bit-for-bit (ULP distance 0), so
//! a miss localizes to a single operation rather than the final coordinate.
//!
//! The fixture's canonical evaluator is the portable Rust `libm` crate. The
//! mpmath implementation in `fixtures-generators/broadcast_eval_portable.py`
//! is retained as an independent high-precision audit, not as the engine
//! oracle; its one-ULP disagreements with `libm` are intentional audit data.

use super::*;
use serde_json::Value;
use std::path::PathBuf;

/// Parse a `0x...` raw-bits literal to f64.
fn bits(s: &str) -> f64 {
    let s = s.trim();
    let hex = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or_else(|| panic!("not a 0x bits literal: {s:?}"));
    let u = u64::from_str_radix(hex, 16).unwrap_or_else(|_| panic!("bad hex bits in {s:?}"));
    f64::from_bits(u)
}

/// ULP distance between two f64; NaN on either side reads as `u64::MAX`.
fn ulp_distance(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        return u64::MAX;
    }
    ordered(a).abs_diff(ordered(b))
}

fn ordered(x: f64) -> i64 {
    let b = x.to_bits() as i64;
    if b < 0 {
        i64::MIN - b
    } else {
        b
    }
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn read_fixture(name: &str) -> Value {
    let raw =
        std::fs::read_to_string(fixture_path(name)).unwrap_or_else(|e| panic!("read {name}: {e}"));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {name}: {e}"))
}

fn b(v: &Value, key: &str) -> f64 {
    bits(
        v[key]
            .as_str()
            .unwrap_or_else(|| panic!("missing/!str {key}")),
    )
}

fn elements_from(v: &Value) -> KeplerianElements {
    KeplerianElements {
        sqrt_a: b(v, "sqrt_a"),
        e: b(v, "e"),
        m0: b(v, "m0"),
        delta_n: b(v, "delta_n"),
        omega0: b(v, "omega0"),
        i0: b(v, "i0"),
        omega: b(v, "omega"),
        omega_dot: b(v, "omega_dot"),
        idot: b(v, "idot"),
        cuc: b(v, "cuc"),
        cus: b(v, "cus"),
        crc: b(v, "crc"),
        crs: b(v, "crs"),
        cic: b(v, "cic"),
        cis: b(v, "cis"),
        toe_sow: b(v, "toe_sow"),
    }
}

fn clock_from(v: &Value) -> ClockPolynomial {
    ClockPolynomial {
        af0: b(v, "af0"),
        af1: b(v, "af1"),
        af2: b(v, "af2"),
        toc_sow: b(v, "toc_sow"),
    }
}

fn cnav_rates_from(v: &Value) -> CnavRates {
    CnavRates {
        adot_m_s: b(v, "adot_m_s"),
        delta_n0_dot_rad_s2: b(v, "delta_n0_dot_rad_s2"),
    }
}

fn consts_for(system: &str) -> ConstellationConstants {
    match system {
        "GPS" => ConstellationConstants::GPS,
        "QZS" => ConstellationConstants::GPS,
        "GAL" => ConstellationConstants::GALILEO,
        "BDS" => ConstellationConstants::BEIDOU,
        other => panic!("unknown system {other}"),
    }
}

fn legacy_satellite_position_ecef_reference(
    elements: &KeplerianElements,
    consts: &ConstellationConstants,
    t_sow_s: f64,
    is_geo: bool,
) -> OrbitState {
    let sqrt_a = elements.sqrt_a;
    let e = elements.e;
    let gm = consts.gm_m3_s2;
    let omega_e = consts.omega_e_rad_s;

    let a = sqrt_a * sqrt_a;
    let n0 = (gm / (a * a * a)).sqrt();
    let n = n0 + elements.delta_n;

    let tk = time_from_reference_s(t_sow_s, elements.toe_sow);

    let mk = elements.m0 + n * tk;
    let kepler = eccentric_anomaly_unchecked(mk, e);
    let ecc_anom = kepler.value;
    let sin_e = libm::sin(ecc_anom);
    let cos_e = libm::cos(ecc_anom);

    let e2 = e * e;
    let nu = libm::atan2((1.0 - e2).sqrt() * sin_e, cos_e - e);
    let phi = nu + elements.omega;

    let two_phi = 2.0 * phi;
    let s2 = libm::sin(two_phi);
    let c2 = libm::cos(two_phi);
    let du = elements.cus * s2 + elements.cuc * c2;
    let dr = elements.crs * s2 + elements.crc * c2;
    let di = elements.cis * s2 + elements.cic * c2;

    let u = phi + du;
    let r = a * (1.0 - e * cos_e) + dr;
    let i = elements.i0 + di + elements.idot * tk;

    let xp = r * libm::cos(u);
    let yp = r * libm::sin(u);

    let omega_k = if is_geo {
        elements.omega0 + elements.omega_dot * tk - omega_e * elements.toe_sow
    } else {
        elements.omega0 + (elements.omega_dot - omega_e) * tk - omega_e * elements.toe_sow
    };

    let sin_o = libm::sin(omega_k);
    let cos_o = libm::cos(omega_k);
    let sin_i = libm::sin(i);
    let cos_i = libm::cos(i);
    let xg = xp * cos_o - yp * cos_i * sin_o;
    let yg = xp * sin_o + yp * cos_i * cos_o;
    let zg = yp * sin_i;

    let (x, y, z) = if is_geo {
        let deg5 = 5.0_f64.to_radians();
        let cos_phi = libm::cos(deg5);
        let sin_phi = -libm::sin(deg5);
        let z_ang = omega_e * tk;
        let cos_z = libm::cos(z_ang);
        let sin_z = libm::sin(z_ang);
        let yr = yg * cos_phi + zg * sin_phi;
        let zr = -yg * sin_phi + zg * cos_phi;
        (xg * cos_z + yr * sin_z, -xg * sin_z + yr * cos_z, zr)
    } else {
        (xg, yg, zg)
    };

    OrbitState {
        a,
        n0,
        n,
        tk,
        mk,
        eccentric_anomaly: ecc_anom,
        kepler_iterations: kepler.iterations,
        sin_e,
        cos_e,
        nu,
        phi,
        s2,
        c2,
        du,
        dr,
        di,
        u,
        r,
        i,
        xp,
        yp,
        omega_k,
        x_m: x,
        y_m: y,
        z_m: z,
    }
}

/// The pinned controls and per-constellation constants in the Rust module must
/// bit-match the recipe that produced the goldens, or every downstream value is
/// trivially "0 ULP against the wrong input".
#[test]
fn pinned_constants_match_the_recipe() {
    let doc = read_fixture("broadcast_golden.json");

    assert_eq!(
        ulp_distance(bits(doc["kepler_tol_hex"].as_str().unwrap()), KEPLER_TOL),
        0
    );
    assert_eq!(
        doc["kepler_max_iter"].as_u64().unwrap() as usize,
        KEPLER_MAX_ITER
    );
    assert_eq!(
        doc["clock_max_iter"].as_u64().unwrap() as usize,
        CLOCK_MAX_ITER
    );
    assert_eq!(
        ulp_distance(
            bits(doc["seconds_per_week_hex"].as_str().unwrap()),
            SECONDS_PER_WEEK
        ),
        0
    );
    assert_eq!(
        ulp_distance(bits(doc["half_week_s_hex"].as_str().unwrap()), HALF_WEEK_S),
        0
    );

    for (name, c) in [
        ("GPS", ConstellationConstants::GPS),
        ("GAL", ConstellationConstants::GALILEO),
        ("BDS", ConstellationConstants::BEIDOU),
    ] {
        let f = &doc["constellations"][name];
        assert_eq!(
            ulp_distance(bits(f["gm_m3_s2_hex"].as_str().unwrap()), c.gm_m3_s2),
            0,
            "{name} GM"
        );
        assert_eq!(
            ulp_distance(
                bits(f["omega_e_rad_s_hex"].as_str().unwrap()),
                c.omega_e_rad_s
            ),
            0,
            "{name} omega_e"
        );
        assert_eq!(
            ulp_distance(bits(f["dtr_f_hex"].as_str().unwrap()), c.dtr_f),
            0,
            "{name} dtr_f"
        );
    }
}

#[test]
fn lnav_position_refactor_preserves_legacy_bits() {
    let doc = read_fixture("broadcast_golden.json");
    let cases = doc["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "fixture has no cases");

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let system = case["system"].as_str().unwrap();
        let consts = consts_for(system);
        let elems = elements_from(&case["elements_hex"]);
        let t_sow = bits(case["t_sow_hex"].as_str().unwrap());
        let is_geo = case["is_geo"].as_bool().unwrap_or(false);

        let before = legacy_satellite_position_ecef_reference(&elems, &consts, t_sow, is_geo);
        let after = satellite_position_ecef_unchecked(&elems, &consts, t_sow, is_geo);
        assert_eq!(after, before, "{name}: refactored legacy orbit changed");
    }
}

#[test]
fn broadcast_eval_is_zero_ulp_against_recipe() {
    let doc = read_fixture("broadcast_golden.json");
    let cases = doc["cases"].as_array().expect("cases array");
    assert!(!cases.is_empty(), "fixture has no cases");

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let system = case["system"].as_str().unwrap();
        let consts = consts_for(system);
        let elems = elements_from(&case["elements_hex"]);
        let clock = clock_from(&case["clock_hex"]);
        let t_sow = bits(case["t_sow_hex"].as_str().unwrap());
        let tgd = bits(case["tgd_s_hex"].as_str().unwrap());
        let is_geo = case["is_geo"].as_bool().unwrap_or(false);

        let orbit = satellite_position_ecef(&elems, &consts, t_sow, is_geo)
            .expect("valid broadcast orbit inputs");
        let clk = satellite_clock_offset_s(&clock, &consts, &elems, orbit.sin_e, t_sow, tgd)
            .expect("valid broadcast clock inputs");

        let ex = &case["expect_hex"];
        let chk = |key: &str, got: f64| {
            let want = bits(
                ex[key]
                    .as_str()
                    .unwrap_or_else(|| panic!("{name}: missing {key}")),
            );
            let d = ulp_distance(want, got);
            assert_eq!(
                d, 0,
                "{name}: {key} diverged: got={got:?} want={want:?} ({d} ULP)"
            );
        };

        chk("a", orbit.a);
        chk("n0", orbit.n0);
        chk("n", orbit.n);
        chk("tk", orbit.tk);
        chk("mk", orbit.mk);
        chk("eccentric_anomaly", orbit.eccentric_anomaly);
        chk("sin_e", orbit.sin_e);
        chk("cos_e", orbit.cos_e);
        chk("nu", orbit.nu);
        chk("phi", orbit.phi);
        chk("s2", orbit.s2);
        chk("c2", orbit.c2);
        chk("du", orbit.du);
        chk("dr", orbit.dr);
        chk("di", orbit.di);
        chk("u", orbit.u);
        chk("r", orbit.r);
        chk("i", orbit.i);
        chk("xp", orbit.xp);
        chk("yp", orbit.yp);
        chk("omega_k", orbit.omega_k);
        chk("x_m", orbit.x_m);
        chk("y_m", orbit.y_m);
        chk("z_m", orbit.z_m);

        assert_eq!(
            orbit.kepler_iterations,
            case["kepler_iterations"].as_u64().unwrap() as usize,
            "{name}: kepler iteration count diverged"
        );

        chk("dt_clock_poly_s", clk.dt_clock_poly_s);
        chk("dt_rel_s", clk.dt_rel_s);
        chk("tgd_s", clk.tgd_s);
        chk("dt_clock_total_s", clk.dt_clock_total_s);
    }
}

/// The combined [`satellite_state`] seam must reproduce the component
/// [`satellite_position_ecef`] / [`satellite_clock_offset_s`] results exactly
/// (it feeds the orbit's own `sin(E)` into the clock), and every Galileo case
/// must be the explicitly-selected I/NAV message.
#[test]
fn satellite_state_combines_orbit_and_clock_consistently() {
    let doc = read_fixture("broadcast_golden.json");
    for case in doc["cases"].as_array().expect("cases array") {
        let name = case["name"].as_str().unwrap();
        let system = case["system"].as_str().unwrap();
        if system == "GAL" {
            assert_eq!(
                case["message"].as_str().unwrap(),
                "GAL_INAV",
                "{name}: Galileo cases must be the selected I/NAV message"
            );
        }

        let consts = consts_for(system);
        let elems = elements_from(&case["elements_hex"]);
        let clock = clock_from(&case["clock_hex"]);
        let t_sow = bits(case["t_sow_hex"].as_str().unwrap());
        let tgd = bits(case["tgd_s_hex"].as_str().unwrap());
        let is_geo = case["is_geo"].as_bool().unwrap_or(false);

        let split_orbit = satellite_position_ecef(&elems, &consts, t_sow, is_geo)
            .expect("valid broadcast orbit inputs");
        let split_clock =
            satellite_clock_offset_s(&clock, &consts, &elems, split_orbit.sin_e, t_sow, tgd)
                .expect("valid broadcast clock inputs");
        let combined = satellite_state(&elems, &clock, &consts, t_sow, tgd, is_geo)
            .expect("valid broadcast state inputs");

        assert_eq!(
            combined.orbit, split_orbit,
            "{name}: combined orbit != split"
        );
        assert_eq!(
            combined.clock, split_clock,
            "{name}: combined clock != split"
        );
    }
}

#[test]
fn cnav_rates_drive_time_dependent_axis_and_mean_motion() {
    let elems = KeplerianElements {
        sqrt_a: 5153.707128525,
        e: 0.01000394229777,
        m0: 0.6342094507864,
        delta_n: 4.304822170265e-9,
        omega0: 2.572838528869,
        i0: 0.9806518601091,
        omega: 0.7941703015008,
        omega_dot: -8.384634967987e-9,
        idot: -5.714523747137e-11,
        cuc: -2.177432179451e-6,
        cus: 1.9371509552e-6,
        crc: 353.96875,
        crs: -39.6875,
        cic: -1.508742570877e-7,
        cis: 1.359730958939e-7,
        toe_sow: 360_000.0,
    };
    let rates = CnavRates {
        adot_m_s: 0.125,
        delta_n0_dot_rad_s2: 1.0e-18,
    };
    let t_sow = elems.toe_sow + 1800.0;
    let orbit = satellite_position_ecef_cnav(&elems, &rates, &ConstellationConstants::GPS, t_sow)
        .expect("valid CNAV orbit");

    let tk = time_from_reference_s(t_sow, elems.toe_sow);
    let a0 = elems.sqrt_a * elems.sqrt_a;
    let n0 = (ConstellationConstants::GPS.gm_m3_s2 / (a0 * a0 * a0)).sqrt();
    let expected_a = a0 + rates.adot_m_s * tk;
    let expected_delta_n = elems.delta_n + 0.5 * rates.delta_n0_dot_rad_s2 * tk;
    let expected_n = n0 + expected_delta_n;

    assert_eq!(orbit.tk.to_bits(), tk.to_bits());
    assert_eq!(orbit.a.to_bits(), expected_a.to_bits());
    assert_eq!(orbit.n.to_bits(), expected_n.to_bits());
    assert!(
        orbit.position().is_ok(),
        "CNAV evaluation should produce a finite ECEF position"
    );
}

#[test]
fn cnav_broadcast_eval_is_zero_ulp_against_recipe() {
    let doc = read_fixture("cnav_broadcast_golden.json");

    assert_eq!(
        ulp_distance(bits(doc["kepler_tol_hex"].as_str().unwrap()), KEPLER_TOL),
        0
    );
    assert_eq!(
        doc["kepler_max_iter"].as_u64().unwrap() as usize,
        KEPLER_MAX_ITER
    );
    assert_eq!(
        doc["clock_max_iter"].as_u64().unwrap() as usize,
        CLOCK_MAX_ITER
    );
    assert_eq!(
        ulp_distance(
            bits(doc["seconds_per_week_hex"].as_str().unwrap()),
            SECONDS_PER_WEEK
        ),
        0
    );
    assert_eq!(
        ulp_distance(bits(doc["half_week_s_hex"].as_str().unwrap()), HALF_WEEK_S),
        0
    );

    for (name, c) in [
        ("GPS", ConstellationConstants::GPS),
        ("QZS", ConstellationConstants::GPS),
    ] {
        let f = &doc["constellations"][name];
        assert_eq!(
            ulp_distance(bits(f["gm_m3_s2_hex"].as_str().unwrap()), c.gm_m3_s2),
            0,
            "{name} GM"
        );
        assert_eq!(
            ulp_distance(
                bits(f["omega_e_rad_s_hex"].as_str().unwrap()),
                c.omega_e_rad_s
            ),
            0,
            "{name} omega_e"
        );
        assert_eq!(
            ulp_distance(bits(f["dtr_f_hex"].as_str().unwrap()), c.dtr_f),
            0,
            "{name} dtr_f"
        );
    }

    let cases = doc["cases"].as_array().expect("cases array");
    assert_eq!(
        cases.len(),
        20,
        "fixture should cover four records at five epochs"
    );

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let system = case["system"].as_str().unwrap();
        let consts = consts_for(system);
        let elems = elements_from(&case["elements_hex"]);
        let rates = cnav_rates_from(&case["rates_hex"]);
        let clock = clock_from(&case["clock_hex"]);
        let t_sow = bits(case["t_sow_hex"].as_str().unwrap());
        let tgd = bits(case["tgd_s_hex"].as_str().unwrap());

        let orbit = satellite_position_ecef_cnav(&elems, &rates, &consts, t_sow)
            .expect("valid CNAV orbit inputs");
        let state = satellite_state_cnav(&elems, &rates, &clock, &consts, t_sow, tgd)
            .expect("valid CNAV state inputs");
        assert_eq!(state.orbit, orbit, "{name}: combined orbit != split");

        let ex = &case["expect_hex"];
        let chk = |key: &str, got: f64| {
            let want = bits(
                ex[key]
                    .as_str()
                    .unwrap_or_else(|| panic!("{name}: missing {key}")),
            );
            let d = ulp_distance(want, got);
            assert_eq!(
                d, 0,
                "{name}: {key} diverged: got={got:?} want={want:?} ({d} ULP)"
            );
        };

        chk("a", orbit.a);
        chk("n0", orbit.n0);
        chk("n", orbit.n);
        chk("tk", orbit.tk);
        chk("mk", orbit.mk);
        chk("eccentric_anomaly", orbit.eccentric_anomaly);
        chk("sin_e", orbit.sin_e);
        chk("cos_e", orbit.cos_e);
        chk("nu", orbit.nu);
        chk("phi", orbit.phi);
        chk("s2", orbit.s2);
        chk("c2", orbit.c2);
        chk("du", orbit.du);
        chk("dr", orbit.dr);
        chk("di", orbit.di);
        chk("u", orbit.u);
        chk("r", orbit.r);
        chk("i", orbit.i);
        chk("xp", orbit.xp);
        chk("yp", orbit.yp);
        chk("omega_k", orbit.omega_k);
        chk("x_m", orbit.x_m);
        chk("y_m", orbit.y_m);
        chk("z_m", orbit.z_m);

        assert_eq!(
            orbit.kepler_iterations,
            case["kepler_iterations"].as_u64().unwrap() as usize,
            "{name}: kepler iteration count diverged"
        );

        chk("dt_clock_poly_s", state.clock.dt_clock_poly_s);
        chk("dt_rel_s", state.clock.dt_rel_s);
        chk("tgd_s", state.clock.tgd_s);
        chk("dt_clock_total_s", state.clock.dt_clock_total_s);
    }
}

#[test]
fn orbit_state_position_rejects_invalid_coordinates() {
    let doc = read_fixture("broadcast_golden.json");
    let case = &doc["cases"].as_array().expect("cases array")[0];
    let consts = consts_for(case["system"].as_str().unwrap());
    let elems = elements_from(&case["elements_hex"]);
    let t_sow = bits(case["t_sow_hex"].as_str().unwrap());
    let is_geo = case["is_geo"].as_bool().unwrap_or(false);

    let mut orbit = satellite_position_ecef(&elems, &consts, t_sow, is_geo)
        .expect("valid broadcast orbit inputs");
    orbit.x_m = f64::NAN;

    assert_eq!(
        orbit.position(),
        Err(crate::frame::FrameValueError::InvalidInput {
            field: "x_m",
            reason: "must be finite",
        })
    );
}

fn assert_invalid_input<T: core::fmt::Debug>(result: crate::Result<T>) {
    let err = result.expect_err("invalid broadcast input must be rejected");
    assert!(
        matches!(err, crate::error::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn broadcast_helpers_reject_invalid_public_inputs() {
    let doc = read_fixture("broadcast_golden.json");
    let case = &doc["cases"].as_array().expect("cases array")[0];
    let mut elems = elements_from(&case["elements_hex"]);
    let mut clock = clock_from(&case["clock_hex"]);
    let consts = consts_for(case["system"].as_str().unwrap());
    let t_sow = bits(case["t_sow_hex"].as_str().unwrap());
    let tgd = bits(case["tgd_s_hex"].as_str().unwrap());

    assert_invalid_input(eccentric_anomaly(f64::NAN, 0.1));
    assert_invalid_input(eccentric_anomaly(0.0, 1.0));

    assert_invalid_input(satellite_position_ecef(
        &elems,
        &consts,
        f64::INFINITY,
        false,
    ));
    elems.sqrt_a = f64::NAN;
    assert_invalid_input(satellite_position_ecef(&elems, &consts, t_sow, false));

    elems = elements_from(&case["elements_hex"]);
    elems.e = 1.0;
    assert_invalid_input(satellite_state(&elems, &clock, &consts, t_sow, tgd, false));

    elems = elements_from(&case["elements_hex"]);
    clock.af0 = f64::NAN;
    assert_invalid_input(satellite_clock_offset_s(
        &clock, &consts, &elems, 0.0, t_sow, tgd,
    ));

    clock = clock_from(&case["clock_hex"]);
    assert_invalid_input(satellite_clock_offset_s(
        &clock,
        &consts,
        &elems,
        f64::NAN,
        t_sow,
        tgd,
    ));
    assert_invalid_input(satellite_state(
        &elems,
        &clock,
        &ConstellationConstants {
            gm_m3_s2: -1.0,
            ..consts
        },
        t_sow,
        tgd,
        false,
    ));
}
