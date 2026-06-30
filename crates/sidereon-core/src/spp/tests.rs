//! Two-track parity for the GPS L1 single-point-positioning pipeline.
//!
//! The reference recipe is `parity/generator/spp.py`; the committed fixtures
//! (vendored at `tests/fixtures/spp_trace_*.json`) record the synthesized
//! observations, the frozen-branch choices, the effective scipy options, and
//! the full iteration trace. Float values are serialized as the raw IEEE-754
//! bit pattern (`f64::to_bits`, a 16-hex-digit `0x...` literal) so there is no
//! decimal-parse ambiguity; parity is measured as ULP distance via the integer
//! reinterpretation of the bit pattern, per the existing SP3/DOP/atmosphere
//! parity discipline.
//!
//! Track 1 (0 ULP, libm/arithmetic-bound) is the trace-replay: for each
//! recorded state `x`, the per-satellite intermediates, the weighted residual
//! vector, and its 2-point finite-difference Jacobian are recomputed by the
//! Rust SPP substrate AT THAT x and asserted bit-for-bit. The Rust solver is
//! not run for this track. The ladder is built up by correction level (L0
//! geometry+clock+Sagnac, L1 +ionosphere, L2 +troposphere, L3 relativistic
//! no-op) so a miss localizes to the term added.
//!
//! Track 2 (sub-micron, BLAS-bound) is the independent-solve agreement: the
//! crate trust-region solver is run from the same inputs and the converged
//! position/clock is asserted to agree with both the recorded scipy solution
//! and the synthesized truth to a documented sub-micron bound. This is a
//! solver-agreement check, explicitly NOT a 0-ULP physics claim, because the
//! trust-region linear-algebra step is not bit-reproducible across BLAS builds.

use std::path::PathBuf;

use serde_json::Value;

use super::test_support;
use super::{
    solve, solve_spp_batch_parallel, solve_spp_batch_serial, solve_with_policy, Corrections,
    KlobucharCoeffs, Observation, RejectionReason, RobustConfig, SatModelEnv, SolveInputs,
    SolvePolicy, SolvePolicyError, SppError, SppInputErrorKind, SppIonosphere, SppModelRecipe,
    SurfaceMet,
};
use crate::astro::math::least_squares::{jacobian_2point, FD_REL_STEP_2POINT};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::ionex::GalileoNequickCoeffs;
use crate::quality::{SolutionValidationError, SolutionValidationOptions};
use crate::rinex_nav::BroadcastStore;
use crate::rinex_obs::{pseudoranges, RinexObs, SignalPolicy};
use nalgebra::DVector;

// ---------------------------------------------------------------------------
// Hex (f64::to_bits) helpers, ULP distance, NaN guard, parser self-check.
// ---------------------------------------------------------------------------

/// Parse a `0x...` raw-bits literal (Python `hexf` / Rust `f64::to_bits`) to f64.
fn bits(s: &str) -> f64 {
    let s = s.trim();
    let hex = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or_else(|| panic!("not a 0x bits literal: {s:?}"));
    let u = u64::from_str_radix(hex, 16).unwrap_or_else(|_| panic!("bad hex bits in {s:?}"));
    f64::from_bits(u)
}

/// ULP distance between two f64; NaN on either side reads as `u64::MAX` so it can
/// never masquerade as 0 ULP.
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

fn sp3() -> crate::sp3::Sp3 {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    );
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
    crate::sp3::Sp3::parse(&bytes).expect("parse real IGS SP3")
}

fn esbc_broadcast_store() -> BroadcastStore {
    let nav = std::fs::read_to_string(fixture_path("nav/ESBC00DNK_R_20201770000_01D_MN.rnx"))
        .expect("read ESBC broadcast NAV fixture");
    BroadcastStore::from_nav(&nav).expect("parse ESBC broadcast NAV")
}

fn esbc_first_epoch_inputs(initial_guess: [f64; 4]) -> (SolveInputs, [f64; 3]) {
    let obs_text = std::fs::read_to_string(fixture_path(
        "obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx",
    ))
    .expect("read ESBC OBS fixture");
    let obs = RinexObs::parse(&obs_text).expect("parse ESBC OBS fixture");
    let truth = obs
        .header()
        .approx_position_m
        .expect("ESBC OBS carries APPROX POSITION XYZ");
    let policy = SignalPolicy {
        codes: [(GnssSystem::Gps, vec!["C1C".to_string()])]
            .into_iter()
            .collect(),
    };
    let observations = pseudoranges(&obs, &obs.epochs()[0], &policy)
        .expect("valid pseudoranges")
        .into_iter()
        .map(|(satellite_id, pseudorange_m)| Observation {
            satellite_id,
            pseudorange_m,
        })
        .collect();

    (
        SolveInputs {
            observations,
            t_rx_j2000_s: 646_315_200.0,
            t_rx_second_of_day_s: 0.0,
            day_of_year: 177.0,
            initial_guess,
            corrections: Corrections {
                ionosphere: false,
                troposphere: true,
            },
            klobuchar: KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            },
            beidou_klobuchar: None,
            galileo_nequick: None,
            glonass_channels: std::collections::BTreeMap::new(),
            met: SurfaceMet {
                pressure_hpa: 1013.25,
                temperature_k: 288.15,
                relative_humidity: 0.5,
            },
            robust: None,
        },
        truth,
    )
}

fn position_error_m(solution: &super::ReceiverSolution, truth: [f64; 3]) -> f64 {
    let p = solution.position.as_array();
    ((p[0] - truth[0]).powi(2) + (p[1] - truth[1]).powi(2) + (p[2] - truth[2]).powi(2)).sqrt()
}

fn assert_solution_bits_eq(left: &super::ReceiverSolution, right: &super::ReceiverSolution) {
    assert_eq!(left.position.x_m.to_bits(), right.position.x_m.to_bits());
    assert_eq!(left.position.y_m.to_bits(), right.position.y_m.to_bits());
    assert_eq!(left.position.z_m.to_bits(), right.position.z_m.to_bits());
    assert_eq!(left.geodetic, right.geodetic);
    assert_eq!(left.rx_clock_s.to_bits(), right.rx_clock_s.to_bits());
    assert_eq!(left.system_clocks_s.len(), right.system_clocks_s.len());
    for ((left_system, left_clock), (right_system, right_clock)) in left
        .system_clocks_s
        .iter()
        .zip(right.system_clocks_s.iter())
    {
        assert_eq!(left_system, right_system);
        assert_eq!(left_clock.to_bits(), right_clock.to_bits());
    }
    assert_eq!(left.dop, right.dop);
    assert_eq!(
        left.residuals_m
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        right
            .residuals_m
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(left.used_sats, right.used_sats);
    assert_eq!(left.rejected_sats, right.rejected_sats);
    assert_eq!(left.metadata, right.metadata);
}

fn parse_prn(token: &str) -> GnssSatelliteId {
    let sys = GnssSystem::from_letter(token.chars().next().unwrap()).expect("known system letter");
    let prn: u8 = token[1..].parse().expect("prn digits");
    GnssSatelliteId::new(sys, prn).expect("valid satellite id")
}

fn arr3(v: &Value) -> [f64; 3] {
    let a = v.as_array().expect("array");
    [
        bits(a[0].as_str().unwrap()),
        bits(a[1].as_str().unwrap()),
        bits(a[2].as_str().unwrap()),
    ]
}

/// Read the level-independent inputs shared by every solve from a fixture.
struct Inputs {
    observations: Vec<Observation>,
    t_rx_j2000_s: f64,
    sod_s: f64,
    doy: f64,
    x0: [f64; 4],
    klobuchar: KlobucharCoeffs,
    met: SurfaceMet,
    corrections: Corrections,
}

fn corrections_for(level: &str) -> Corrections {
    match level {
        "L0_minimal" => Corrections::NONE,
        "L1_iono" => Corrections::IONO,
        "L2_tropo" | "L3_relativistic" => Corrections::IONO_TROPO,
        other => panic!("unknown level {other}"),
    }
}

fn load_inputs(doc: &Value, level: &str) -> Inputs {
    let f = &doc["fixture"];
    let inp = &f["inputs"];

    let observations = inp["observations"]
        .as_array()
        .expect("observations array")
        .iter()
        .map(|o| Observation {
            satellite_id: parse_prn(o["sat_id"].as_str().unwrap()),
            pseudorange_m: bits(o["p_meas_m"].as_str().unwrap()),
        })
        .collect();

    let alpha_v = inp["klobuchar_alpha"].as_array().unwrap();
    let beta_v = inp["klobuchar_beta"].as_array().unwrap();
    let klobuchar = KlobucharCoeffs {
        alpha: [
            bits(alpha_v[0].as_str().unwrap()),
            bits(alpha_v[1].as_str().unwrap()),
            bits(alpha_v[2].as_str().unwrap()),
            bits(alpha_v[3].as_str().unwrap()),
        ],
        beta: [
            bits(beta_v[0].as_str().unwrap()),
            bits(beta_v[1].as_str().unwrap()),
            bits(beta_v[2].as_str().unwrap()),
            bits(beta_v[3].as_str().unwrap()),
        ],
    };

    let met = SurfaceMet {
        pressure_hpa: bits(inp["met"]["pressure_hpa"].as_str().unwrap()),
        temperature_k: bits(inp["met"]["temperature_k"].as_str().unwrap()),
        relative_humidity: bits(inp["met"]["relative_humidity"].as_str().unwrap()),
    };

    let x0v = f["frozen"]["initial_guess_x0"].as_array().unwrap();
    let x0 = [
        bits(x0v[0].as_str().unwrap()),
        bits(x0v[1].as_str().unwrap()),
        bits(x0v[2].as_str().unwrap()),
        bits(x0v[3].as_str().unwrap()),
    ];

    Inputs {
        observations,
        t_rx_j2000_s: bits(inp["t_rx_j2000_s"].as_str().unwrap()),
        sod_s: bits(inp["t_rx_sod_s"].as_str().unwrap()),
        doy: bits(inp["doy"].as_str().unwrap()),
        x0,
        klobuchar,
        met,
        corrections: corrections_for(level),
    }
}

fn solve_inputs(i: &Inputs) -> SolveInputs {
    SolveInputs {
        observations: i.observations.clone(),
        t_rx_j2000_s: i.t_rx_j2000_s,
        t_rx_second_of_day_s: i.sod_s,
        day_of_year: i.doy,
        initial_guess: i.x0,
        corrections: i.corrections,
        klobuchar: i.klobuchar,
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met: i.met,
        robust: None,
    }
}

const LEVELS: &[&str] = &["L0_minimal", "L1_iono", "L2_tropo", "L3_relativistic"];

fn fixture_name(level: &str) -> String {
    format!("spp_trace_{level}.json")
}

// ---------------------------------------------------------------------------
// Parser self-check: a known bit pattern round-trips. A parser bug must not be
// able to masquerade as parity.
// ---------------------------------------------------------------------------
#[test]
fn hex_bits_parser_round_trips() {
    let pi = std::f64::consts::PI;
    let hexed = format!("0x{:016x}", pi.to_bits());
    assert_eq!(
        bits(&hexed).to_bits(),
        pi.to_bits(),
        "bits parser round-trip broken"
    );
    // ULP distance of a value to itself is zero; to its neighbour is one.
    assert_eq!(ulp_distance(pi, pi), 0);
    let nxt = f64::from_bits(pi.to_bits() + 1);
    assert_eq!(ulp_distance(pi, nxt), 1);
    assert_eq!(ulp_distance(f64::NAN, 1.0), u64::MAX);
}

// ---------------------------------------------------------------------------
// Boundary cross-check: the meters/radians-native geodetic helper vs the core
// km/deg itrs_to_geodetic_compute. The 0-ULP claim is on the meters-native
// radians vs the recipe's recorded meters_native radians (both replicate the
// same Skyfield AU-internal algorithm); the core public deg API is checked
// against the recipe's recorded core_api_deg.
// ---------------------------------------------------------------------------
#[test]
fn geodetic_meters_native_zero_ulp() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let cases = doc["geodetic_crosscheck"]["cases"]
        .as_array()
        .expect("cases");
    assert!(cases.len() >= 3, "expected >= 3 cross-check cases");

    let mut failures = Vec::new();
    let mut checks = 0usize;

    for case in cases {
        let km = arr3(&case["ecef_km"]);
        let g = test_support::geodetic_from_ecef_m_for_test(
            km[0] * 1000.0,
            km[1] * 1000.0,
            km[2] * 1000.0,
        );

        let mn = &case["meters_native"];
        let want_lat = bits(mn["lat_rad"].as_str().unwrap());
        let want_lon = bits(mn["lon_rad"].as_str().unwrap());
        let want_h = bits(mn["height_m"].as_str().unwrap());
        for (label, got, want) in [
            ("lat_rad", g.lat_rad, want_lat),
            ("lon_rad", g.lon_rad, want_lon),
            ("height_m", g.height_m, want_h),
        ] {
            checks += 1;
            let u = ulp_distance(got, want);
            if u != 0 {
                failures.push(format!("meters_native.{label}: {u} ULP"));
            }
        }

        // Core public deg API vs the recipe's recorded core_api_deg.
        let (lat_deg, lon_deg, alt_km) =
            test_support::itrs_to_geodetic_core_km(km[0], km[1], km[2]);
        let want_lat_deg = bits(case["core_api_deg"]["lat_deg"].as_str().unwrap());
        let want_lon_deg = bits(case["core_api_deg"]["lon_deg"].as_str().unwrap());
        let want_alt_km = bits(case["core_internal"]["alt_km"].as_str().unwrap());
        for (label, got, want) in [
            ("core.lat_deg", lat_deg, want_lat_deg),
            ("core.lon_deg", lon_deg, want_lon_deg),
            ("core.alt_km", alt_km, want_alt_km),
        ] {
            checks += 1;
            let u = ulp_distance(got, want);
            if u != 0 {
                failures.push(format!("{label}: {u} ULP"));
            }
        }
    }

    assert!(checks > 0, "no cross-check components asserted");
    assert!(
        failures.is_empty(),
        "geodetic boundary cross-check diverged on {} of {checks} components:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// TRACK 1 (0 ULP): trace-replay of the per-satellite intermediates, the
// weighted residual vector, and the 2-point FD Jacobian at each recorded state.
// ---------------------------------------------------------------------------

fn used_sats(doc: &Value) -> Vec<GnssSatelliteId> {
    doc["fixture"]["used_sats"]
        .as_array()
        .expect("used_sats")
        .iter()
        .map(|s| parse_prn(s.as_str().unwrap()))
        .collect()
}

/// The weighted residual closure (`sqrt(w) * (P_meas - P_hat)`) used by the FD
/// Jacobian, mirroring what scipy differences and what `LeastSquaresProblem::
/// with_weights` scales.
fn weighted_residual_at(
    sp3: &crate::sp3::Sp3,
    used: &[GnssSatelliteId],
    obs_by_id: &[(GnssSatelliteId, f64)],
    sqrt_w: &[f64],
    inputs: &Inputs,
    x: &[f64; 4],
) -> DVector<f64> {
    let rx = [x[0], x[1], x[2]];
    let b = x[3];
    let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
    let env = SatModelEnv {
        eph: sp3,
        t_rx_j2000_s: inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.sod_s,
        day_of_year: inputs.doy,
        corrections: inputs.corrections,
        met: &inputs.met,
        glonass_channels: &glonass_channels,
        model: SppModelRecipe::reference(),
    };
    let r: Vec<f64> = used
        .iter()
        .enumerate()
        .map(|(i, &sat)| {
            let p_meas = obs_by_id
                .iter()
                .find(|(id, _)| *id == sat)
                .map(|(_, p)| *p)
                .unwrap();
            let m = test_support::sat_model_for_test(&env, sat, rx, b, p_meas, &inputs.klobuchar)
                .expect("ephemeris present at trace state");
            sqrt_w[i] * (p_meas - m.p_hat_m)
        })
        .collect();
    DVector::from_vec(r)
}

fn trace_replay_level(level: &str) {
    let name = fixture_name(level);
    let doc = read_fixture(&name);
    let f = &doc["fixture"];
    let inputs = load_inputs(&doc, level);
    let sp3 = sp3();
    let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
    let env = SatModelEnv {
        eph: &sp3,
        t_rx_j2000_s: inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.sod_s,
        day_of_year: inputs.doy,
        corrections: inputs.corrections,
        met: &inputs.met,
        glonass_channels: &glonass_channels,
        model: SppModelRecipe::reference(),
    };

    let used = used_sats(&doc);
    let obs_by_id: Vec<(GnssSatelliteId, f64)> = inputs
        .observations
        .iter()
        .map(|o| (o.satellite_id, o.pseudorange_m))
        .collect();

    // sqrt(weight) per used satellite, from the recorded frozen geometry.
    let geom = &f["used_sat_geometry"];
    let sqrt_w: Vec<f64> = used
        .iter()
        .map(|id| bits(geom[id.to_string()]["sqrt_weight"].as_str().unwrap()))
        .collect();

    let mut failures = Vec::new();
    let mut checks = 0usize;
    let mut check = |label: String, got: f64, want: f64, failures: &mut Vec<String>| {
        checks += 1;
        let u = ulp_distance(got, want);
        if u != 0 {
            failures.push(format!(
                "{label}: {u} ULP (rust=0x{:016x} ref=0x{:016x})",
                got.to_bits(),
                want.to_bits()
            ));
        }
    };

    let states = f["trace_states"].as_array().expect("trace_states");
    assert!(!states.is_empty(), "{level}: no trace states");

    for st in states {
        let ti = st["trace_index"].as_i64().unwrap();
        let xv = st["x"].as_array().unwrap();
        let x = [
            bits(xv[0].as_str().unwrap()),
            bits(xv[1].as_str().unwrap()),
            bits(xv[2].as_str().unwrap()),
            bits(xv[3].as_str().unwrap()),
        ];
        let rx = [x[0], x[1], x[2]];
        let b = x[3];

        // (1) Per-satellite named intermediates, bit-for-bit.
        let per_sat = st["per_sat"].as_array().unwrap();
        for (i, &sat) in used.iter().enumerate() {
            let ps = &per_sat[i];
            assert_eq!(
                ps["prn"].as_str().unwrap(),
                sat.to_string(),
                "{level}.state{ti}: per_sat order"
            );
            let p_meas = obs_by_id
                .iter()
                .find(|(id, _)| *id == sat)
                .map(|(_, p)| *p)
                .unwrap();
            let m = test_support::sat_model_for_test(&env, sat, rx, b, p_meas, &inputs.klobuchar)
                .expect("ephemeris present");

            let pfx = format!("{level}.state{ti}.{sat}");
            check(
                format!("{pfx}.tau_s"),
                m.tau_s,
                bits(ps["tau_s"].as_str().unwrap()),
                &mut failures,
            );
            check(
                format!("{pfx}.t_tx_j2000_s"),
                m.t_tx_j2000_s,
                bits(ps["t_tx_j2000_s"].as_str().unwrap()),
                &mut failures,
            );
            let se = arr3(&ps["sat_ecef_m"]);
            check(
                format!("{pfx}.sat_ecef_x"),
                m.sat_ecef_m[0],
                se[0],
                &mut failures,
            );
            check(
                format!("{pfx}.sat_ecef_y"),
                m.sat_ecef_m[1],
                se[1],
                &mut failures,
            );
            check(
                format!("{pfx}.sat_ecef_z"),
                m.sat_ecef_m[2],
                se[2],
                &mut failures,
            );
            check(
                format!("{pfx}.dt_sat_s"),
                m.dt_sat_s,
                bits(ps["dt_sat_s"].as_str().unwrap()),
                &mut failures,
            );
            check(
                format!("{pfx}.theta_rad"),
                m.theta_rad,
                bits(ps["theta_rad"].as_str().unwrap()),
                &mut failures,
            );
            let sr = arr3(&ps["sat_rot_ecef_m"]);
            check(
                format!("{pfx}.sat_rot_x"),
                m.sat_rot_ecef_m[0],
                sr[0],
                &mut failures,
            );
            check(
                format!("{pfx}.sat_rot_y"),
                m.sat_rot_ecef_m[1],
                sr[1],
                &mut failures,
            );
            check(
                format!("{pfx}.sat_rot_z"),
                m.sat_rot_ecef_m[2],
                sr[2],
                &mut failures,
            );
            check(
                format!("{pfx}.rho_m"),
                m.rho_m,
                bits(ps["rho_m"].as_str().unwrap()),
                &mut failures,
            );
            check(
                format!("{pfx}.az_rad"),
                m.az_rad,
                bits(ps["az_rad"].as_str().unwrap()),
                &mut failures,
            );
            check(
                format!("{pfx}.el_rad"),
                m.el_rad,
                bits(ps["el_rad"].as_str().unwrap()),
                &mut failures,
            );
            check(
                format!("{pfx}.iono_m"),
                m.iono_m,
                bits(ps["iono_m"].as_str().unwrap()),
                &mut failures,
            );
            check(
                format!("{pfx}.tropo_m"),
                m.tropo_m,
                bits(ps["tropo_m"].as_str().unwrap()),
                &mut failures,
            );
            check(
                format!("{pfx}.p_hat_m"),
                m.p_hat_m,
                bits(ps["p_hat_m"].as_str().unwrap()),
                &mut failures,
            );
            // The weighted residual the solver sees.
            let r_w = sqrt_w[i] * (p_meas - m.p_hat_m);
            check(
                format!("{pfx}.residual_m"),
                r_w,
                bits(ps["residual_m"].as_str().unwrap()),
                &mut failures,
            );
        }

        // (2) Weighted residual vector (used-sat order).
        let r = weighted_residual_at(&sp3, &used, &obs_by_id, &sqrt_w, &inputs, &x);
        let res_v = st["residual"].as_array().unwrap();
        for (i, want) in res_v.iter().enumerate() {
            check(
                format!("{level}.state{ti}.residual[{i}]"),
                r[i],
                bits(want.as_str().unwrap()),
                &mut failures,
            );
        }

        // (3) 2-point FD Jacobian of the weighted residual at x. The fixture
        // records the FD rel_step; assert it equals the crate constant so the
        // step itself is pinned, then compare the assembled Jacobian.
        let fd = &st["fd_2point"];
        check(
            format!("{level}.state{ti}.fd_rel_step"),
            FD_REL_STEP_2POINT,
            bits(fd["rel_step"].as_str().unwrap()),
            &mut failures,
        );

        let f0 = r.clone();
        let x_vec = DVector::from_row_slice(&x);
        let resid_closure = |p: &DVector<f64>| -> DVector<f64> {
            let pa = [p[0], p[1], p[2], p[3]];
            weighted_residual_at(&sp3, &used, &obs_by_id, &sqrt_w, &inputs, &pa)
        };
        let jac = jacobian_2point(resid_closure, &x_vec, &f0).expect("valid SPP jacobian");

        let jac_v = fd["jac"].as_array().unwrap();
        for (row, want_row) in jac_v.iter().enumerate() {
            let cols = want_row.as_array().unwrap();
            for (col, want) in cols.iter().enumerate() {
                check(
                    format!("{level}.state{ti}.jac[{row}][{col}]"),
                    jac[(row, col)],
                    bits(want.as_str().unwrap()),
                    &mut failures,
                );
            }
        }
    }

    assert!(checks > 0, "{level}: no components checked");
    assert!(
        failures.is_empty(),
        "{level}: SPP substrate diverged from the reference recipe on {} of {checks} components:\n  {}",
        failures.len(),
        failures.join("\n  ")
    );
}

/// Hex bits literal for a regenerated golden value (Rust `f64::to_bits`).
fn hexbits(v: f64) -> String {
    format!("0x{:016x}", v.to_bits())
}

/// One-shot regeneration of the satellite-position-dependent golden fields in
/// the SPP trace fixtures, gated behind `REGEN_SPP_TRACE=1`.
///
/// The SP3 position interpolation moved from the (wrong) global scipy cubic
/// spline to the IGS/RTKLIB sliding-window Lagrange recipe (see
/// `sp3::interp`). That shifts every satellite ECEF position by sub-millimetre
/// to sub-centimetre, which cascades through the deterministic SPP geometry /
/// atmosphere into `tau`, `theta`, `sat_rot`, `rho`, `az`, `el`, `p_hat`, the
/// residual vector, and the FD Jacobian. The satellite-position leg is now
/// externally certified against RTKLIB in `sp3::interp::interp_tests`; this
/// routine recomputes the downstream-of-position trace fields through the
/// production substrate so the trace-replay remains a faithful 0-ULP pin of the
/// corrected pipeline. Inputs (x, observations, weights, the recorded Klobuchar
/// / met coefficients) are left untouched.
fn regen_trace_level(level: &str) {
    let name = fixture_name(level);
    let mut doc = read_fixture(&name);
    let sp3 = sp3();

    // (0) Re-synthesize the noise-free observations so the fixture's synthetic
    // world is self-consistent with the corrected (RTKLIB) interpolation: each
    // pseudorange is the production forward model `p_hat` evaluated at the fixed
    // receiver truth and truth clock bias. With scipy-recipe observations the
    // solver would land ~0.6 mm off truth purely from the recipe change; with
    // RTKLIB-recipe observations it recovers truth to sub-nm, as the original
    // fixture intended. Truth (rx ECEF, clock bias) is left untouched.
    {
        let inputs0 = load_inputs(&doc, level);
        let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
        let env0 = SatModelEnv {
            eph: &sp3,
            t_rx_j2000_s: inputs0.t_rx_j2000_s,
            t_rx_second_of_day_s: inputs0.sod_s,
            day_of_year: inputs0.doy,
            corrections: inputs0.corrections,
            met: &inputs0.met,
            glonass_channels: &glonass_channels,
            model: SppModelRecipe::reference(),
        };
        let tr = doc["fixture"]["inputs"]["rx_truth_ecef_m"]
            .as_array()
            .unwrap()
            .clone();
        let rx_truth = [
            bits(tr[0].as_str().unwrap()),
            bits(tr[1].as_str().unwrap()),
            bits(tr[2].as_str().unwrap()),
        ];
        let b_truth = bits(doc["fixture"]["inputs"]["b_truth_m"].as_str().unwrap());
        let n_obs = doc["fixture"]["inputs"]["observations"]
            .as_array()
            .unwrap()
            .len();
        for oi in 0..n_obs {
            let sat = parse_prn(
                doc["fixture"]["inputs"]["observations"][oi]["sat_id"]
                    .as_str()
                    .unwrap(),
            );
            // p_meas placeholder for tau seed: use the existing value so the
            // transmit-time iteration starts from the same tau (it converges to
            // the same fixed point regardless, the seed only sets iteration count).
            let p_seed = bits(
                doc["fixture"]["inputs"]["observations"][oi]["p_meas_m"]
                    .as_str()
                    .unwrap(),
            );
            let m = test_support::sat_model_for_test(
                &env0,
                sat,
                rx_truth,
                b_truth,
                p_seed,
                &inputs0.klobuchar,
            )
            .expect("ephemeris present at truth");
            doc["fixture"]["inputs"]["observations"][oi]["p_meas_m"] = hexbits(m.p_hat_m).into();
        }
    }

    // Reload inputs with the regenerated observations.
    let inputs = load_inputs(&doc, level);
    let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
    let env = SatModelEnv {
        eph: &sp3,
        t_rx_j2000_s: inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.sod_s,
        day_of_year: inputs.doy,
        corrections: inputs.corrections,
        met: &inputs.met,
        glonass_channels: &glonass_channels,
        model: SppModelRecipe::reference(),
    };
    let used = used_sats(&doc);
    let obs_by_id: Vec<(GnssSatelliteId, f64)> = inputs
        .observations
        .iter()
        .map(|o| (o.satellite_id, o.pseudorange_m))
        .collect();
    let geom = doc["fixture"]["used_sat_geometry"].clone();
    let sqrt_w: Vec<f64> = used
        .iter()
        .map(|id| bits(geom[id.to_string()]["sqrt_weight"].as_str().unwrap()))
        .collect();

    let n_states = doc["fixture"]["trace_states"].as_array().unwrap().len();
    for si in 0..n_states {
        let xv = doc["fixture"]["trace_states"][si]["x"]
            .as_array()
            .unwrap()
            .clone();
        let x = [
            bits(xv[0].as_str().unwrap()),
            bits(xv[1].as_str().unwrap()),
            bits(xv[2].as_str().unwrap()),
            bits(xv[3].as_str().unwrap()),
        ];
        let rx = [x[0], x[1], x[2]];
        let b = x[3];

        for (i, &sat) in used.iter().enumerate() {
            let p_meas = obs_by_id.iter().find(|(id, _)| *id == sat).unwrap().1;
            let m = test_support::sat_model_for_test(&env, sat, rx, b, p_meas, &inputs.klobuchar)
                .expect("ephemeris present");
            let r_w = sqrt_w[i] * (p_meas - m.p_hat_m);
            let ps = &mut doc["fixture"]["trace_states"][si]["per_sat"][i];
            ps["tau_s"] = hexbits(m.tau_s).into();
            ps["t_tx_j2000_s"] = hexbits(m.t_tx_j2000_s).into();
            ps["sat_ecef_m"] = serde_json::json!([
                hexbits(m.sat_ecef_m[0]),
                hexbits(m.sat_ecef_m[1]),
                hexbits(m.sat_ecef_m[2])
            ]);
            ps["dt_sat_s"] = hexbits(m.dt_sat_s).into();
            ps["theta_rad"] = hexbits(m.theta_rad).into();
            ps["sat_rot_ecef_m"] = serde_json::json!([
                hexbits(m.sat_rot_ecef_m[0]),
                hexbits(m.sat_rot_ecef_m[1]),
                hexbits(m.sat_rot_ecef_m[2])
            ]);
            ps["rho_m"] = hexbits(m.rho_m).into();
            ps["az_rad"] = hexbits(m.az_rad).into();
            ps["el_rad"] = hexbits(m.el_rad).into();
            ps["iono_m"] = hexbits(m.iono_m).into();
            ps["tropo_m"] = hexbits(m.tropo_m).into();
            ps["p_hat_m"] = hexbits(m.p_hat_m).into();
            ps["residual_m"] = hexbits(r_w).into();
        }

        let r = weighted_residual_at(&sp3, &used, &obs_by_id, &sqrt_w, &inputs, &x);
        let res_arr: Vec<Value> = (0..r.len()).map(|i| hexbits(r[i]).into()).collect();
        doc["fixture"]["trace_states"][si]["residual"] = Value::Array(res_arr);

        let f0 = r.clone();
        let x_vec = DVector::from_row_slice(&x);
        let resid_closure = |p: &DVector<f64>| -> DVector<f64> {
            let pa = [p[0], p[1], p[2], p[3]];
            weighted_residual_at(&sp3, &used, &obs_by_id, &sqrt_w, &inputs, &pa)
        };
        let jac = jacobian_2point(resid_closure, &x_vec, &f0).expect("valid SPP jacobian");
        let jac_rows: Vec<Value> = (0..jac.nrows())
            .map(|row| {
                Value::Array(
                    (0..jac.ncols())
                        .map(|col| hexbits(jac[(row, col)]).into())
                        .collect(),
                )
            })
            .collect();
        doc["fixture"]["trace_states"][si]["fd_2point"]["jac"] = Value::Array(jac_rows);
    }

    // (4) Re-solve and record the converged solution as the new agreement
    // target. With the regenerated RTKLIB-consistent observations the solver
    // recovers truth to sub-nm; the recorded `final_solution.x` becomes the
    // corrected solver's converged value.
    {
        let sol = solve(&sp3, &solve_inputs(&inputs), true).expect("solve converges");
        let x = [
            sol.position.x_m,
            sol.position.y_m,
            sol.position.z_m,
            sol.rx_clock_s * super::C_M_S,
        ];
        doc["fixture"]["final_solution"]["x"] =
            serde_json::json!([hexbits(x[0]), hexbits(x[1]), hexbits(x[2]), hexbits(x[3])]);
        doc["fixture"]["final_solution"]["rx_clock_s"] = hexbits(sol.rx_clock_s).into();
        // Absolute error vs the (unchanged) truth.
        let tr = doc["fixture"]["inputs"]["rx_truth_ecef_m"]
            .as_array()
            .unwrap()
            .clone();
        let truth = [
            bits(tr[0].as_str().unwrap()),
            bits(tr[1].as_str().unwrap()),
            bits(tr[2].as_str().unwrap()),
        ];
        let b_truth = bits(doc["fixture"]["inputs"]["b_truth_m"].as_str().unwrap());
        doc["fixture"]["final_solution"]["abs_err_x_m"] = serde_json::json!([
            hexbits((x[0] - truth[0]).abs()),
            hexbits((x[1] - truth[1]).abs()),
            hexbits((x[2] - truth[2]).abs())
        ]);
        doc["fixture"]["final_solution"]["abs_err_clock_m"] =
            hexbits((x[3] - b_truth).abs()).into();
    }

    // Record the position-leg reference change in the fixture provenance.
    doc["env_ref"]["sp3_interp_reference"] = serde_json::json!(
        "position: RTKLIB preceph.c interppol/pephpos (sliding-window degree-10 \
         Lagrange + OMGE per-node rotation), certified in sp3::interp::interp_tests; \
         these trace fields recomputed through the corrected production substrate. \
         clock: scipy.interpolate.CubicSpline (unchanged)."
    );

    let out = serde_json::to_string_pretty(&doc).expect("serialize fixture");
    std::fs::write(fixture_path(&name), out + "\n").expect("write fixture");
    eprintln!("regenerated {name}");
}

/// Gated regeneration entry point. Run with:
/// `REGEN_SPP_TRACE=1 cargo test -p sidereon-core regen_spp_trace_fixtures -- --ignored --nocapture`
#[test]
#[ignore = "regeneration helper; run explicitly with REGEN_SPP_TRACE=1"]
fn regen_spp_trace_fixtures() {
    if std::env::var("REGEN_SPP_TRACE").as_deref() != Ok("1") {
        panic!("set REGEN_SPP_TRACE=1 to regenerate the SPP trace fixtures");
    }
    for level in ["L0_minimal", "L1_iono", "L2_tropo", "L3_relativistic"] {
        regen_trace_level(level);
    }
}

#[test]
fn trace_replay_l0_minimal_zero_ulp() {
    trace_replay_level("L0_minimal");
}

#[test]
fn trace_replay_l1_iono_zero_ulp() {
    trace_replay_level("L1_iono");
}

#[test]
fn trace_replay_l2_tropo_zero_ulp() {
    trace_replay_level("L2_tropo");
}

#[test]
fn trace_replay_l3_relativistic_zero_ulp() {
    trace_replay_level("L3_relativistic");
}

/// The relativistic level is a documented no-op: SP3 precise clocks are used
/// as-is with no separate periodic term, so L3 must reproduce L2 bit-for-bit at
/// every recorded per-satellite predicted range.
#[test]
fn relativistic_level_equals_tropo_level() {
    let l2 = read_fixture("spp_trace_L2_tropo.json");
    let l3 = read_fixture("spp_trace_L3_relativistic.json");
    let p2 = &l2["fixture"]["trace_states"][0]["per_sat"];
    let p3 = &l3["fixture"]["trace_states"][0]["per_sat"];
    let a2 = p2.as_array().unwrap();
    let a3 = p3.as_array().unwrap();
    assert_eq!(a2.len(), a3.len(), "L2/L3 used-sat count differs");
    for (s2, s3) in a2.iter().zip(a3) {
        assert_eq!(
            s2["p_hat_m"].as_str().unwrap(),
            s3["p_hat_m"].as_str().unwrap(),
            "relativistic no-op changed p_hat for {}",
            s2["prn"].as_str().unwrap()
        );
    }
}

// ---------------------------------------------------------------------------
// TRACK 2 (sub-micron, BLAS-bound): independent-solve agreement. The crate
// solver runs from the same inputs; the converged position/clock is asserted to
// agree with both the recorded scipy solution and the synthesized truth to a
// documented sub-micron bound. This is a SOLVER-AGREEMENT check, never a 0-ULP
// physics claim.
// ---------------------------------------------------------------------------

/// Documented agreement bound: the converged receiver coordinates and the clock
/// length agree to better than one micron with both the recorded scipy solution
/// and the synthesized truth. The trust-region linear-algebra step is owned by
/// the platform BLAS, so the last bits are not independently reproducible;
/// sub-micron is the non-obtrusive bar from the spec's Solver-reality section.
///
/// In practice this implementation agrees to a few nanometres (observed
/// ~5e-9 m, ~7e-16 relative on the ~6.3e6 m position magnitude); the bound is
/// held at the spec's sub-micron figure with comfortable margin, and is a
/// SOLVER-AGREEMENT check, explicitly not a 0-ULP physics-parity claim.
const AGREEMENT_BOUND_M: f64 = 1.0e-6;

fn independent_solve_level(level: &str) {
    let name = fixture_name(level);
    let doc = read_fixture(&name);
    let f = &doc["fixture"];
    let inputs = load_inputs(&doc, level);
    let sp3 = sp3();

    let sol = solve(&sp3, &solve_inputs(&inputs), true).expect("solve converges");

    // used_sats / rejected_sats match the fixture exactly (deterministic order).
    let want_used = used_sats(&doc);
    assert_eq!(sol.used_sats, want_used, "{level}: used_sats order/content");

    let want_rej = f["rejected_sats"].as_array().unwrap();
    assert_eq!(
        sol.rejected_sats.len(),
        want_rej.len(),
        "{level}: rejected count"
    );
    for (got, want) in sol.rejected_sats.iter().zip(want_rej) {
        assert_eq!(
            got.satellite_id,
            parse_prn(want["id"].as_str().unwrap()),
            "{level}: rejected id"
        );
        let want_reason = match want["reason"].as_str().unwrap() {
            "no_ephemeris" => RejectionReason::NoEphemeris,
            "low_elevation" => RejectionReason::LowElevation,
            other => panic!("unexpected rejection reason {other}"),
        };
        assert_eq!(
            got.reason, want_reason,
            "{level}: rejected reason for {}",
            got.satellite_id
        );
    }

    // Converged position/clock vs the recorded scipy solution.
    let fs = &f["final_solution"];
    let scipy_x = fs["x"].as_array().unwrap();
    let sx = [
        bits(scipy_x[0].as_str().unwrap()),
        bits(scipy_x[1].as_str().unwrap()),
        bits(scipy_x[2].as_str().unwrap()),
        bits(scipy_x[3].as_str().unwrap()),
    ];
    let got = [
        sol.position.x_m,
        sol.position.y_m,
        sol.position.z_m,
        sol.rx_clock_s * super::C_M_S,
    ];
    for (k, (g, s)) in got.iter().zip(sx.iter()).enumerate() {
        assert!(
            (g - s).abs() <= AGREEMENT_BOUND_M,
            "{level}: component {k} disagrees with scipy: |{g} - {s}| = {} > {AGREEMENT_BOUND_M} m",
            (g - s).abs()
        );
    }

    // Converged position/clock vs the synthesized truth.
    let tx = fs["truth_x"].as_array().unwrap();
    let truth = [
        bits(tx[0].as_str().unwrap()),
        bits(tx[1].as_str().unwrap()),
        bits(tx[2].as_str().unwrap()),
        bits(tx[3].as_str().unwrap()),
    ];
    for (k, (g, t)) in got.iter().zip(truth.iter()).enumerate() {
        assert!(
            (g - t).abs() <= AGREEMENT_BOUND_M,
            "{level}: component {k} disagrees with truth: |{g} - {t}| = {} > {AGREEMENT_BOUND_M} m",
            (g - t).abs()
        );
    }

    // The clock-second boundary: rx_clock_s == b_m / c.
    let want_clock_s = bits(fs["truth_rx_clock_s"].as_str().unwrap());
    assert!(
        (sol.rx_clock_s - want_clock_s).abs() <= AGREEMENT_BOUND_M / super::C_M_S,
        "{level}: rx_clock_s off by {}",
        (sol.rx_clock_s - want_clock_s).abs()
    );

    assert!(sol.metadata.converged, "{level}: solver did not converge");
    assert!(sol.dop.is_some(), "{level}: DOP missing");
}

#[test]
fn independent_solve_l0_agreement() {
    independent_solve_level("L0_minimal");
}

#[test]
fn independent_solve_l1_agreement() {
    independent_solve_level("L1_iono");
}

#[test]
fn independent_solve_l2_agreement() {
    independent_solve_level("L2_tropo");
}

#[test]
fn independent_solve_l3_agreement() {
    independent_solve_level("L3_relativistic");
}

// ---------------------------------------------------------------------------
// DOP from the converged geometry agrees with the recipe's recorded DOP.
// (BLAS-agreement track: the DOP recipe is 0-ULP at a fixed geometry, but here
// it is recomputed at the independently-converged position, so it is a
// sub-micron-geometry agreement, not a 0-ULP claim.)
// ---------------------------------------------------------------------------
#[test]
fn dop_from_converged_geometry_agrees() {
    for &level in LEVELS {
        let doc = read_fixture(&fixture_name(level));
        let inputs = load_inputs(&doc, level);
        let sp3 = sp3();
        let sol = solve(&sp3, &solve_inputs(&inputs), false).expect("solve");
        let dop = sol.dop.expect("dop present");
        let want = &doc["fixture"]["dop"];
        for (label, got) in [
            ("gdop", dop.gdop),
            ("pdop", dop.pdop),
            ("hdop", dop.hdop),
            ("vdop", dop.vdop),
            ("tdop", dop.tdop),
        ] {
            let w = bits(want[label].as_str().unwrap());
            let rel = (got - w).abs() / w.max(1.0);
            assert!(
                rel <= 1e-9,
                "{level}: {label} disagrees: rust={got} ref={w} (rel {rel})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Failure/rejection behavior.
// ---------------------------------------------------------------------------

fn assert_invalid_spp_input(inputs: SolveInputs, field: &'static str, kind: SppInputErrorKind) {
    let sp3 = sp3();
    match solve(&sp3, &inputs, false) {
        Err(SppError::InvalidInput {
            field: got_field,
            kind: got_kind,
        }) => {
            assert_eq!(got_field, field);
            assert_eq!(got_kind, kind);
        }
        other => panic!("expected InvalidInput({field}, {kind:?}), got {other:?}"),
    }
}

#[test]
fn invalid_spp_scalar_inputs_are_rejected_at_entry() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let base = solve_inputs(&load_inputs(&doc, "L0_minimal"));

    let mut inputs = base.clone();
    inputs.t_rx_j2000_s = f64::NAN;
    assert_invalid_spp_input(inputs, "t_rx_j2000_s", SppInputErrorKind::NonFinite);

    let mut inputs = base.clone();
    inputs.t_rx_second_of_day_s = -1.0;
    assert_invalid_spp_input(
        inputs,
        "t_rx_second_of_day_s",
        SppInputErrorKind::OutOfRange,
    );

    let mut inputs = base.clone();
    inputs.t_rx_second_of_day_s = 300_000.0;
    assert_invalid_spp_input(
        inputs,
        "t_rx_second_of_day_s",
        SppInputErrorKind::OutOfRange,
    );

    let mut inputs = base.clone();
    inputs.day_of_year = f64::INFINITY;
    assert_invalid_spp_input(inputs, "day_of_year", SppInputErrorKind::NonFinite);

    let mut inputs = base.clone();
    inputs.day_of_year = 367.0;
    assert_invalid_spp_input(inputs, "day_of_year", SppInputErrorKind::OutOfRange);

    let mut inputs = base;
    inputs.initial_guess[1] = f64::NAN;
    assert_invalid_spp_input(inputs, "initial_guess", SppInputErrorKind::NonFinite);
}

#[test]
fn invalid_spp_model_inputs_are_rejected_at_entry() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let base = solve_inputs(&load_inputs(&doc, "L0_minimal"));

    let mut inputs = base.clone();
    inputs.observations[0].pseudorange_m = f64::NAN;
    assert_invalid_spp_input(
        inputs,
        "observation.pseudorange_m",
        SppInputErrorKind::NonFinite,
    );

    let mut inputs = base.clone();
    inputs.klobuchar.alpha[2] = f64::NAN;
    assert_invalid_spp_input(inputs, "klobuchar", SppInputErrorKind::NonFinite);

    let mut inputs = base.clone();
    inputs.beidou_klobuchar = Some(KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0, f64::INFINITY, 0.0, 0.0],
    });
    assert_invalid_spp_input(inputs, "beidou_klobuchar", SppInputErrorKind::NonFinite);

    let mut inputs = base.clone();
    inputs.galileo_nequick = Some(GalileoNequickCoeffs {
        ai0: 0.0,
        ai1: f64::NAN,
        ai2: 0.0,
    });
    assert_invalid_spp_input(inputs, "galileo_nequick", SppInputErrorKind::NonFinite);

    let doc = read_fixture("spp_trace_L2_tropo.json");
    let mut inputs = solve_inputs(&load_inputs(&doc, "L2_tropo"));
    inputs.met.pressure_hpa = 0.0;
    assert_invalid_spp_input(inputs, "met.pressure_hpa", SppInputErrorKind::NotPositive);

    let mut inputs = solve_inputs(&load_inputs(&doc, "L2_tropo"));
    inputs.met.relative_humidity = 50.0;
    assert_invalid_spp_input(
        inputs,
        "met.relative_humidity",
        SppInputErrorKind::OutOfRange,
    );

    let mut inputs = base.clone();
    inputs.robust = Some(RobustConfig {
        max_outer: 0,
        ..RobustConfig::default()
    });
    assert_invalid_spp_input(inputs, "robust.max_outer", SppInputErrorKind::NotPositive);

    let mut inputs = base;
    inputs.robust = Some(RobustConfig {
        huber_k: f64::NAN,
        ..RobustConfig::default()
    });
    assert_invalid_spp_input(inputs, "robust.huber_k", SppInputErrorKind::NonFinite);
}

#[test]
fn bounded_spp_inputs_accept_valid_upper_edges() {
    let doc = read_fixture("spp_trace_L2_tropo.json");
    let mut inputs = solve_inputs(&load_inputs(&doc, "L2_tropo"));
    inputs.t_rx_second_of_day_s = 86_399.999;
    inputs.day_of_year = 366.999;
    inputs.met.relative_humidity = 1.0;

    solve(&sp3(), &inputs, false).expect("valid bounded SPP inputs");
}

#[test]
fn galileo_nequick_coeffs_are_publicly_nameable_for_solve_inputs() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let mut inputs: crate::positioning::SolveInputs =
        solve_inputs(&load_inputs(&doc, "L0_minimal"));
    let atmosphere_coeffs = crate::atmosphere::ionosphere::GalileoNequickCoeffs {
        ai0: 66.25,
        ai1: -0.16406,
        ai2: -0.0024719,
    };
    let rinex_coeffs: crate::rinex::nav::GalileoNequickCoeffs = atmosphere_coeffs;
    let positioning_coeffs: crate::positioning::GalileoNequickCoeffs = rinex_coeffs;

    inputs.galileo_nequick = Some(positioning_coeffs);

    assert_eq!(inputs.galileo_nequick, Some(positioning_coeffs));
}

#[test]
fn galileo_ionosphere_uses_nequick_coefficients_and_gps_stays_klobuchar() {
    let doc = read_fixture("spp_trace_L1_iono.json");
    let fixture_inputs = load_inputs(&doc, "L1_iono");
    let mut solve_inputs = solve_inputs(&fixture_inputs);
    solve_inputs.corrections = Corrections::IONO;

    let sp3 = sp3();
    let glonass_channels = std::collections::BTreeMap::<u8, i8>::new();
    let env = SatModelEnv {
        eph: &sp3,
        t_rx_j2000_s: fixture_inputs.t_rx_j2000_s,
        t_rx_second_of_day_s: fixture_inputs.sod_s,
        day_of_year: fixture_inputs.doy,
        corrections: Corrections::IONO,
        met: &fixture_inputs.met,
        glonass_channels: &glonass_channels,
        model: SppModelRecipe::reference(),
    };
    let tr = doc["fixture"]["inputs"]["rx_truth_ecef_m"]
        .as_array()
        .unwrap();
    let rx = [
        bits(tr[0].as_str().unwrap()),
        bits(tr[1].as_str().unwrap()),
        bits(tr[2].as_str().unwrap()),
    ];
    let state = [rx[0], rx[1], rx[2], 0.0];
    let p_seed = 22_000_000.0;
    let gal_coeffs = GalileoNequickCoeffs {
        ai0: 66.25,
        ai1: -0.16406,
        ai2: -0.0024719,
    };

    let find_sat = |system, ionosphere| {
        (1..=64).find_map(|prn| {
            let sat = GnssSatelliteId::new(system, prn).ok()?;
            test_support::sat_model_with_ionosphere_for_test(&env, sat, rx, 0.0, p_seed, ionosphere)
                .map(|model| (sat, model))
        })
    };

    let (gal_sat, gal_model) = find_sat(
        GnssSystem::Galileo,
        SppIonosphere::GalileoNequick(gal_coeffs),
    )
    .expect("SP3 fixture has a Galileo satellite");
    let gal_klobuchar = test_support::sat_model_with_ionosphere_for_test(
        &env,
        gal_sat,
        rx,
        0.0,
        p_seed,
        SppIonosphere::Klobuchar(fixture_inputs.klobuchar),
    )
    .expect("same Galileo satellite is modeled with Klobuchar");
    assert_ne!(
        gal_model.iono_m.to_bits(),
        gal_klobuchar.iono_m.to_bits(),
        "Galileo NeQuick-G path must be distinct from GPS Klobuchar"
    );

    solve_inputs.observations = vec![Observation {
        satellite_id: gal_sat,
        pseudorange_m: p_seed,
    }];
    solve_inputs.galileo_nequick = Some(gal_coeffs);
    let got = super::residual_unweighted(
        &sp3,
        &[gal_sat],
        &[(gal_sat, p_seed)],
        &state,
        &solve_inputs,
        SppModelRecipe::reference(),
    )
    .expect("Galileo residual evaluates");
    assert_eq!(
        got[0].to_bits(),
        (p_seed - gal_model.p_hat_m).to_bits(),
        "SolveInputs with GAL coefficients must dispatch Galileo to NeQuick-G"
    );
    assert_ne!(
        got[0].to_bits(),
        (p_seed - gal_klobuchar.p_hat_m).to_bits(),
        "Galileo residual must not use the GPS Klobuchar result"
    );

    let (gps_sat, gps_model) = find_sat(
        GnssSystem::Gps,
        SppIonosphere::Klobuchar(fixture_inputs.klobuchar),
    )
    .expect("SP3 fixture has a GPS satellite");
    solve_inputs.observations = vec![Observation {
        satellite_id: gps_sat,
        pseudorange_m: p_seed,
    }];
    let gps = super::residual_unweighted(
        &sp3,
        &[gps_sat],
        &[(gps_sat, p_seed)],
        &state,
        &solve_inputs,
        SppModelRecipe::reference(),
    )
    .expect("GPS residual evaluates");
    assert_eq!(
        gps[0].to_bits(),
        (p_seed - gps_model.p_hat_m).to_bits(),
        "GPS must remain on Klobuchar even when Galileo coefficients are present"
    );
}

#[test]
fn unused_met_is_ignored_without_troposphere_correction() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let base = solve_inputs(&load_inputs(&doc, "L0_minimal"));
    assert_eq!(base.corrections, Corrections::NONE);

    let sp3 = sp3();
    let standard = solve(&sp3, &base, false).expect("solve with standard met");

    let mut zero_met = base;
    zero_met.met = SurfaceMet {
        pressure_hpa: 0.0,
        temperature_k: 0.0,
        relative_humidity: 0.0,
    };
    let placeholder = solve(&sp3, &zero_met, false).expect("solve with unused zero met");

    assert_eq!(
        placeholder.position.x_m.to_bits(),
        standard.position.x_m.to_bits()
    );
    assert_eq!(
        placeholder.position.y_m.to_bits(),
        standard.position.y_m.to_bits()
    );
    assert_eq!(
        placeholder.position.z_m.to_bits(),
        standard.position.z_m.to_bits()
    );
    assert_eq!(
        placeholder.rx_clock_s.to_bits(),
        standard.rx_clock_s.to_bits()
    );
    assert_eq!(
        placeholder.system_clocks_s.len(),
        standard.system_clocks_s.len()
    );
    for ((got_system, got_clock), (want_system, want_clock)) in placeholder
        .system_clocks_s
        .iter()
        .zip(&standard.system_clocks_s)
    {
        assert_eq!(got_system, want_system);
        assert_eq!(got_clock.to_bits(), want_clock.to_bits());
    }
    assert_eq!(placeholder.dop, standard.dop);
    assert_eq!(placeholder.residuals_m.len(), standard.residuals_m.len());
    for (got, want) in placeholder.residuals_m.iter().zip(&standard.residuals_m) {
        assert_eq!(got.to_bits(), want.to_bits());
    }
    assert_eq!(placeholder.used_sats, standard.used_sats);
    assert_eq!(placeholder.rejected_sats, standard.rejected_sats);
    assert_eq!(placeholder.metadata, standard.metadata);
}

fn degenerate_geometry_case() -> (crate::sp3::Sp3, SolveInputs) {
    let bytes = std::fs::read(fixture_path("sp3/degenerate_coincident_5sat.sp3"))
        .expect("read degenerate SP3");
    let sp3 = crate::sp3::Sp3::parse(&bytes).expect("parse degenerate SP3");

    // Identical pseudorange for every satellite, so the Sagnac rotation is the
    // same for all of them and the rows stay coincident (rank-deficient).
    let p = 20_181_863.0;
    let observations = (1..=5)
        .map(|prn| Observation {
            satellite_id: GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id"),
            pseudorange_m: p,
        })
        .collect();

    (
        sp3,
        SolveInputs {
            observations,
            // A receive epoch inside the product's [00:00, 00:15] window.
            t_rx_j2000_s: 646_229_000.0,
            t_rx_second_of_day_s: 200.0,
            day_of_year: 176.0,
            initial_guess: [6_378_137.0, 0.0, 0.0, 0.0],
            corrections: Corrections::NONE,
            klobuchar: KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            },
            beidou_klobuchar: None,
            galileo_nequick: None,
            glonass_channels: std::collections::BTreeMap::new(),
            met: SurfaceMet {
                pressure_hpa: 1013.25,
                temperature_k: 288.15,
                relative_humidity: 0.5,
            },
            robust: None,
        },
    )
}

/// Fewer than four usable satellites is the documented underdetermined failure.
#[test]
fn too_few_satellites_rejected() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let mut inputs = load_inputs(&doc, "L0_minimal");
    // Keep only the first three used satellites' observations.
    let used = used_sats(&doc);
    let keep: Vec<_> = used.iter().take(3).copied().collect();
    inputs
        .observations
        .retain(|o| keep.contains(&o.satellite_id));
    let sp3 = sp3();
    match solve(&sp3, &solve_inputs(&inputs), false) {
        Err(SppError::TooFewSatellites { used, required }) => {
            // GPS-only here, so the requirement is the classic four.
            assert!(used < 4, "expected <4 usable, got {used}");
            assert_eq!(required, 4, "single-system solve requires 4 satellites");
        }
        other => panic!("expected TooFewSatellites, got {other:?}"),
    }
}

/// A satellite present in the observations but absent from the SP3 product is
/// rejected with `no_ephemeris` (the residual path must error/skip it, never
/// panic), and the solve still succeeds on the remaining satellites.
#[test]
fn no_ephemeris_satellite_is_rejected() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let mut inputs = load_inputs(&doc, "L0_minimal");
    // GPS PRN 99 is not in the product, so it has no ephemeris at any epoch.
    let ghost = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 99,
    };
    inputs.observations.push(Observation {
        satellite_id: ghost,
        pseudorange_m: 2.2e7,
    });

    let sp3 = sp3();
    let sol =
        solve(&sp3, &solve_inputs(&inputs), false).expect("solve succeeds on the real satellites");
    assert!(
        sol.rejected_sats
            .iter()
            .any(|r| r.satellite_id == ghost && r.reason == RejectionReason::NoEphemeris),
        "ghost satellite should be rejected with no_ephemeris; rejected = {:?}",
        sol.rejected_sats
    );
}

/// Duplicate observations for the same satellite are rejected deterministically
/// (by the smallest repeated id), so the result can never depend on input order.
#[test]
fn duplicate_observation_is_rejected() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let mut inputs = load_inputs(&doc, "L0_minimal");
    let dup = inputs.observations[0];
    // Push a second observation for the same satellite with a different range.
    inputs.observations.push(Observation {
        satellite_id: dup.satellite_id,
        pseudorange_m: dup.pseudorange_m + 1234.5,
    });

    let sp3 = sp3();
    match solve(&sp3, &solve_inputs(&inputs), false) {
        Err(SppError::DuplicateObservation { satellite }) => {
            assert_eq!(satellite, dup.satellite_id)
        }
        other => panic!("expected DuplicateObservation, got {other:?}"),
    }
}

/// The residual path returns `Err(satellite)` instead of panicking when a used
/// satellite cannot be modeled at the query state. This is the condition
/// `solve()` records in its closure and surfaces as `SppError::EphemerisLost`
/// if it occurs during a solver probe (the harder case where a satellite
/// survives selection and then drops). With real SP3 coverage a
/// selection-surviving satellite does not actually drop across the bounded
/// transmit-time probes, so the `Err` path itself is exercised directly here to
/// lock in the no-panic guarantee the solver relies on.
#[test]
fn residual_errs_instead_of_panicking_on_unmodelable_satellite() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let si = solve_inputs(&load_inputs(&doc, "L0_minimal"));
    let sp3 = sp3();

    // A satellite with no ephemeris in the product, placed in the `used` set as
    // if it had survived selection; the residual must error on it, not panic.
    let ghost = GnssSatelliteId {
        system: GnssSystem::Gps,
        prn: 99,
    };
    let used = [ghost];
    let obs_by_id = [(ghost, 2.2e7)];
    let r = super::residual_unweighted(
        &sp3,
        &used,
        &obs_by_id,
        &si.initial_guess,
        &si,
        SppModelRecipe::reference(),
    );
    assert_eq!(
        r,
        Err(ghost),
        "residual must return Err for an unmodelable used satellite, never panic"
    );
}

/// The solve's rejected set, reasons, and order match the fixture exactly. This
/// L0 geometry exercises the `low_elevation` mask; the `no_ephemeris` branch is
/// covered separately by `no_ephemeris_satellite_is_rejected`.
#[test]
fn rejection_reasons_match_fixture() {
    let doc = read_fixture("spp_trace_L0_minimal.json");
    let inputs = load_inputs(&doc, "L0_minimal");
    let sp3 = sp3();
    let sol = solve(&sp3, &solve_inputs(&inputs), false).expect("solve");

    let want: Vec<(GnssSatelliteId, RejectionReason)> = doc["fixture"]["rejected_sats"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            let id = parse_prn(r["id"].as_str().unwrap());
            let reason = match r["reason"].as_str().unwrap() {
                "no_ephemeris" => RejectionReason::NoEphemeris,
                "low_elevation" => RejectionReason::LowElevation,
                other => panic!("unexpected reason {other}"),
            };
            (id, reason)
        })
        .collect();

    let got: Vec<(GnssSatelliteId, RejectionReason)> = sol
        .rejected_sats
        .iter()
        .map(|r| (r.satellite_id, r.reason))
        .collect();
    assert_eq!(
        got, want,
        "rejected set/reasons/order diverged from fixture"
    );
    // At least one low-elevation rejection exists in this geometry.
    assert!(
        want.iter()
            .any(|(_, r)| *r == RejectionReason::LowElevation),
        "fixture should exercise the low-elevation mask"
    );
}

/// A degenerate geometry - several satellites at coincident positions, so every
/// line-of-sight row is identical and the design matrix is rank-deficient - must
/// be handled gracefully, never a panic. The Levenberg-damped solver regularizes
/// the rank deficiency and converges, but the DOP cofactor inverse detects the
/// singular geometry and reports no DOP. (`SppError::Singular` is the defensive
/// path for a subproblem the damping cannot step through, and is not reachable
/// from this robustly-regularized geometry.)
#[test]
fn degenerate_geometry_is_handled_gracefully() {
    let (sp3, inputs) = degenerate_geometry_case();

    match solve(&sp3, &inputs, false) {
        // Damped solver converges, but the rank-deficient geometry yields no DOP.
        Ok(sol) => assert!(
            sol.dop.is_none(),
            "a rank-deficient geometry must not report a DOP, got {:?}",
            sol.dop
        ),
        // Also acceptable if the subproblem genuinely cannot be stepped.
        Err(SppError::Singular(_)) => {}
        other => {
            panic!("degenerate geometry mishandled (expected Ok/no-DOP or Singular): {other:?}")
        }
    }
}

#[test]
fn policy_validation_rejects_rank_deficient_geometry() {
    let (sp3, inputs) = degenerate_geometry_case();

    match solve_with_policy(&sp3, &inputs, false, SolvePolicy::default()) {
        Err(SolvePolicyError::Validation(
            SolutionValidationError::DegenerateGeometryRankDeficient,
        )) => {}
        other => panic!("expected rank-deficient validation error, got {other:?}"),
    }
}

#[test]
fn policy_validation_applies_max_pdop() {
    let store = esbc_broadcast_store();
    let (inputs, _) = esbc_first_epoch_inputs([3_582_135.0, 532_569.0, 5_232_779.0, 0.0]);
    let policy = SolvePolicy {
        validation: SolutionValidationOptions {
            max_pdop: Some(0.1),
            ..SolutionValidationOptions::default()
        },
        coarse_search_seeds: None,
    };

    match solve_with_policy(&store, &inputs, false, policy) {
        Err(SolvePolicyError::Validation(SolutionValidationError::DegenerateGeometryPdop(
            pdop,
        ))) => assert!(pdop > 0.1, "PDOP ceiling should report the actual PDOP"),
        other => panic!("expected PDOP validation error, got {other:?}"),
    }
}

#[test]
fn policy_coarse_search_recovers_esbc_cold_start() {
    let store = esbc_broadcast_store();
    let (inputs, truth) = esbc_first_epoch_inputs([0.0, 0.0, 0.0, 0.0]);
    let policy = SolvePolicy {
        coarse_search_seeds: Some(24),
        ..SolvePolicy::default()
    };

    let sol = solve_with_policy(&store, &inputs, true, policy).expect("coarse search solves");
    assert_eq!(sol.position.x_m.to_bits(), 0x414b544d32219a0d);
    assert_eq!(sol.position.y_m.to_bits(), 0x412040dc182a9b20);
    assert_eq!(sol.position.z_m.to_bits(), 0x4153f61dfc670caa);
    assert_eq!(sol.rx_clock_s.to_bits(), 0x3f3f84f505aa3883);
    assert!(sol.metadata.converged);
    assert!(sol.metadata.redundancy >= 1);
    assert!(sol.metadata.raim_checkable);
    assert_eq!(sol.metadata.used_count, sol.used_sats.len());
    assert_eq!(sol.metadata.systems, vec![GnssSystem::Gps]);
    assert!(
        position_error_m(&sol, truth) < 6.0,
        "ESBC cold-start error was {} m",
        position_error_m(&sol, truth)
    );
    assert!(sol.geodetic.is_some(), "geodetic output was requested");
}

/// The P4 runtime selector dispatches the SPP reference strategy to the same
/// `solve_with_policy` entry point, so `estimate` produces a bit-identical
/// `ReceiverSolution`. This is the behavior-preserving proof for the facade: no
/// new numerics, only selection plus a verbatim forward.
#[test]
fn estimate_spp_reference_matches_solve_with_policy_bit_for_bit() {
    use crate::estimation::{
        estimate, EstimateError, EstimateInput, EstimateOptions, EstimateOutput, StrategyId,
        Technique,
    };

    let store = esbc_broadcast_store();
    let (inputs, _) = esbc_first_epoch_inputs([3_582_135.0, 532_569.0, 5_232_779.0, 0.0]);
    let policy = SolvePolicy::default();

    let direct = solve_with_policy(&store, &inputs, true, policy).expect("direct solve");
    let via = estimate(
        EstimateInput::Spp {
            eph: &store,
            inputs: &inputs,
            with_geodetic: true,
            policy,
        },
        EstimateOptions::default(),
    )
    .expect("estimate solve");
    let EstimateOutput::Spp(via) = via else {
        panic!("SPP input must dispatch to an SPP output, got {via:?}");
    };

    assert_eq!(via.position.x_m.to_bits(), direct.position.x_m.to_bits());
    assert_eq!(via.position.y_m.to_bits(), direct.position.y_m.to_bits());
    assert_eq!(via.position.z_m.to_bits(), direct.position.z_m.to_bits());
    assert_eq!(via.rx_clock_s.to_bits(), direct.rx_clock_s.to_bits());
    assert_eq!(via.residuals_m.len(), direct.residuals_m.len());
    for (v, d) in via.residuals_m.iter().zip(&direct.residuals_m) {
        assert_eq!(v.to_bits(), d.to_bits());
    }
    assert_eq!(via.used_sats, direct.used_sats);
    assert_eq!(
        via.system_clocks_s
            .iter()
            .map(|(s, c)| (*s, c.to_bits()))
            .collect::<Vec<_>>(),
        direct
            .system_clocks_s
            .iter()
            .map(|(s, c)| (*s, c.to_bits()))
            .collect::<Vec<_>>(),
    );

    // Selecting a strategy whose technique does not match the input is a
    // selection error, not a silent wrong-strategy solve.
    let mismatch = estimate(
        EstimateInput::Spp {
            eph: &store,
            inputs: &inputs,
            with_geodetic: false,
            policy,
        },
        EstimateOptions::new(StrategyId::rtk_reference()),
    )
    .expect_err("rtk strategy on spp input must error");
    assert!(matches!(
        mismatch,
        EstimateError::TechniqueMismatch {
            strategy: Technique::Rtk,
            input: Technique::Spp,
        }
    ));
}

/// P5 owned deterministic solver. `solve_with_solver` selecting the legacy
/// recipe must be bit-identical to `solve` (the additive guarantee: the new
/// dispatch leaves the reference SPP path untouched), and the owned
/// deterministic trust-region kernel produces its OWN frozen-bits solution on
/// the ESBC first-epoch fixture. The owned factorization is a different
/// reduction order than the legacy nalgebra LU, so it carries its own pinned
/// bits rather than reusing the legacy goldens. Determinism scope: the owned
/// kernel owns the dense subproblem factorization (no nalgebra LU, no black-box
/// BLAS in that solve); the surrounding normal-matrix / gradient / norm
/// reductions still go through nalgebra, so these pinned bits are this build's
/// reproducible output (asserted run-to-run below), with the cross-platform bit
/// guarantee scoped to the factorization.
#[test]
fn owned_deterministic_solver_frozen_bits() {
    use super::solve_with_solver;
    use crate::estimation::recipe::SolverRecipe;

    let store = esbc_broadcast_store();
    let (inputs, truth) = esbc_first_epoch_inputs([3_582_135.0, 532_569.0, 5_232_779.0, 0.0]);

    // The legacy recipe arm is bit-identical to the reference `solve`.
    let reference = solve(&store, &inputs, false).expect("reference solve");
    let legacy =
        solve_with_solver(&store, &inputs, false, SolverRecipe::NalgebraTrfLegacy).expect("legacy");
    assert_eq!(
        legacy.position.x_m.to_bits(),
        reference.position.x_m.to_bits()
    );
    assert_eq!(
        legacy.position.y_m.to_bits(),
        reference.position.y_m.to_bits()
    );
    assert_eq!(
        legacy.position.z_m.to_bits(),
        reference.position.z_m.to_bits()
    );
    assert_eq!(legacy.rx_clock_s.to_bits(), reference.rx_clock_s.to_bits());

    // Owned deterministic kernel: its own frozen-bits golden.
    let owned = solve_with_solver(&store, &inputs, true, SolverRecipe::OwnedDeterministicTrf)
        .expect("owned deterministic solve");
    assert_eq!(owned.position.x_m.to_bits(), 0x414b544cd339d204);
    assert_eq!(owned.position.y_m.to_bits(), 0x412040dc030556d9);
    assert_eq!(owned.position.z_m.to_bits(), 0x4153f61de1d76fa6);
    assert_eq!(owned.rx_clock_s.to_bits(), 0x3f3f84ebef5aa1b8);
    assert_eq!(owned.used_sats, reference.used_sats);
    assert_eq!(owned.residuals_m.len(), reference.residuals_m.len());

    // Determinism: a second owned solve is bit-identical.
    let owned_again = solve_with_solver(&store, &inputs, true, SolverRecipe::OwnedDeterministicTrf)
        .expect("owned deterministic solve again");
    assert_eq!(
        owned.position.x_m.to_bits(),
        owned_again.position.x_m.to_bits()
    );
    assert_eq!(
        owned.position.y_m.to_bits(),
        owned_again.position.y_m.to_bits()
    );
    assert_eq!(
        owned.position.z_m.to_bits(),
        owned_again.position.z_m.to_bits()
    );

    // Selectable via the runtime strategy selector, not only the opt-in helper:
    // driving `estimate` with the owned deterministic SPP strategy reaches the
    // same owned solver and yields bit-identical output.
    use crate::estimation::strategies::{estimate, EstimateInput, EstimateOptions, EstimateOutput};
    use crate::estimation::StrategyId;
    let via_strategy = match estimate(
        EstimateInput::Spp {
            eph: &store,
            inputs: &inputs,
            with_geodetic: true,
            policy: SolvePolicy::default(),
        },
        EstimateOptions::new(StrategyId::spp_owned_deterministic()),
    )
    .expect("owned deterministic solve via estimate")
    {
        EstimateOutput::Spp(solution) => *solution,
        _ => unreachable!("the SPP strategy yields an SPP solution"),
    };
    assert_eq!(
        via_strategy.position.x_m.to_bits(),
        owned.position.x_m.to_bits()
    );
    assert_eq!(
        via_strategy.position.y_m.to_bits(),
        owned.position.y_m.to_bits()
    );
    assert_eq!(
        via_strategy.position.z_m.to_bits(),
        owned.position.z_m.to_bits()
    );
    assert_eq!(
        via_strategy.rx_clock_s.to_bits(),
        owned.rx_clock_s.to_bits()
    );

    // The owned solution remains physically close to truth (a sanity bound,
    // not the bit-exact gate).
    assert!(
        position_error_m(&owned, truth) < 6.0,
        "owned solver error was {} m",
        position_error_m(&owned, truth)
    );
}

/// The satellite ordering the solve pins (`GnssSatelliteId` `Ord`) matches a
/// zero-padded PRN string sort for GPS, so the Rust ordering agrees with the
/// fixture generator's string-keyed sort. This is the GPS-only v1 assumption;
/// multi-GNSS would need the same property to hold across the system letters.
#[test]
fn gnss_satellite_id_orders_like_zero_padded_prn_strings() {
    let mut ids: Vec<GnssSatelliteId> = (1..=12u8)
        .rev()
        .map(|prn| GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id"))
        .collect();
    ids.sort();
    let by_ord: Vec<String> = ids.iter().map(|s| s.to_string()).collect();

    let mut by_string = by_ord.clone();
    by_string.sort();

    assert_eq!(
        by_ord, by_string,
        "GnssSatelliteId Ord must match the zero-padded PRN string order"
    );
    assert_eq!(by_ord.first().map(String::as_str), Some("G01"));
    assert_eq!(by_ord.last().map(String::as_str), Some("G12"));
}

/// The opt-in Huber path is additive and default-off: with `robust = None` the
/// solve is byte-identical to today, and the new metadata reports no outer
/// iterations. With `robust = Some`, a large injected pseudorange outlier is
/// down-weighted, so the converged position moves AWAY from the corrupted-solve
/// fix and TOWARD the clean fix, and the outer loop reports having run. This
/// exercises the engage path against the real SP3 substrate without a new
/// fixture.
#[test]
fn huber_engages_on_outlier_and_is_off_by_default() {
    let doc = read_fixture(&fixture_name("L0_minimal"));
    let inputs = load_inputs(&doc, "L0_minimal");
    let sp3 = sp3();

    // Clean baseline (no outlier), static weighting.
    let base = solve_inputs(&inputs);
    let clean = solve(&sp3, &base, false).expect("clean solve");

    // robust=None must be byte-identical to today: no outer iterations, no scale.
    assert_eq!(clean.metadata.outer_iterations, 0);
    assert!(clean.metadata.final_robust_scale_m.is_none());

    // Inject a large bias into a USED satellite's pseudorange (the first
    // observation may be a masked/rejected satellite, so target one the solve
    // actually weights).
    let used0 = clean.used_sats[0];
    let mut corrupt = base.clone();
    let idx = corrupt
        .observations
        .iter()
        .position(|o| o.satellite_id == used0)
        .expect("used satellite has an observation");
    corrupt.observations[idx].pseudorange_m += 75.0;

    let corrupt_static = solve(&sp3, &corrupt, false).expect("corrupt static solve");
    // Byte-identical-off invariant on the corrupted inputs too.
    assert_eq!(corrupt_static.metadata.outer_iterations, 0);

    // Same corrupted inputs, Huber on.
    let mut corrupt_robust = corrupt.clone();
    corrupt_robust.robust = Some(RobustConfig {
        huber_k: 1.345,
        scale_floor_m: 1.0,
        max_outer: 5,
        outer_tol_m: 1e-4,
    });
    let robust = solve(&sp3, &corrupt_robust, false).expect("robust solve");

    // The outer loop ran and recorded a scale.
    assert!(
        robust.metadata.outer_iterations >= 1,
        "Huber outer loop did not run (outer_iterations={})",
        robust.metadata.outer_iterations
    );
    assert!(robust.metadata.final_robust_scale_m.is_some());

    // Down-weighting the outlier pulls the Huber fix closer to the clean fix
    // than the static-weighted corrupted fix is.
    let cp = clean.position.as_array();
    let sp = corrupt_static.position.as_array();
    let rp = robust.position.as_array();
    let d = |a: [f64; 3], b: [f64; 3]| {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    };
    let static_err = d(sp, cp);
    let robust_err = d(rp, cp);
    assert!(
        robust_err < static_err,
        "Huber did not move the outlier-corrupted fix toward the clean fix \
         (robust_err={robust_err:.3} m, static_err={static_err:.3} m)"
    );
}

#[test]
fn robust_max_outer_counts_total_solves_and_preserves_early_convergence() {
    let doc = read_fixture(&fixture_name("L0_minimal"));
    let inputs = load_inputs(&doc, "L0_minimal");
    let sp3 = sp3();
    let base = solve_inputs(&inputs);
    let clean = solve(&sp3, &base, false).expect("clean solve");

    let used0 = clean.used_sats[0];
    let mut corrupt = base.clone();
    let idx = corrupt
        .observations
        .iter()
        .position(|o| o.satellite_id == used0)
        .expect("used satellite has an observation");
    corrupt.observations[idx].pseudorange_m += 75.0;
    let corrupt_static = solve(&sp3, &corrupt, false).expect("corrupt static solve");

    for (max_outer, expected_reweighted_solves) in [(1, 0), (2, 1), (5, 4)] {
        let mut capped = corrupt.clone();
        capped.robust = Some(RobustConfig {
            huber_k: 1.345,
            scale_floor_m: 1.0,
            max_outer,
            outer_tol_m: f64::MIN_POSITIVE,
        });
        let solution = solve(&sp3, &capped, false).expect("capped robust solve");
        assert_eq!(
            solution.metadata.outer_iterations, expected_reweighted_solves,
            "max_outer={max_outer} should leave room for only \
             {expected_reweighted_solves} reweighted solves"
        );
        assert_eq!(
            solution.metadata.outer_iterations + 1,
            max_outer,
            "max_outer={max_outer} must count the warm-start solve"
        );

        if max_outer == 1 {
            assert!(
                solution.metadata.final_robust_scale_m.is_none(),
                "warm-start-only robust solve must not record a reweighting scale"
            );
            assert_solution_bits_eq(&solution, &corrupt_static);
        } else {
            assert!(
                solution.metadata.final_robust_scale_m.is_some(),
                "reweighted robust solve should record a scale"
            );
        }
    }

    let mut early_stop_two = corrupt.clone();
    early_stop_two.robust = Some(RobustConfig {
        huber_k: 1.345,
        scale_floor_m: 1.0,
        max_outer: 2,
        outer_tol_m: f64::MAX,
    });
    let two = solve(&sp3, &early_stop_two, false).expect("early stop robust solve");
    assert_eq!(two.metadata.outer_iterations, 1);

    let mut early_stop_five = early_stop_two.clone();
    early_stop_five.robust = Some(RobustConfig {
        max_outer: 5,
        ..early_stop_two.robust.expect("robust config")
    });
    let five = solve(&sp3, &early_stop_five, false).expect("higher-cap robust solve");
    assert_eq!(five.metadata.outer_iterations, 1);
    assert_solution_bits_eq(&two, &five);
}

/// Bounded-tolerance band for canonical SPP vs the Skyfield-faithful reference
/// SPP on a shared case. Canonical and reference implement the same physics and
/// differ only in op-order: canonical iterates the light-time loop to convergence
/// (vs the reference's fixed two-iteration truncation) and uses a meters-native
/// WGS84 geodetic basis (vs the Skyfield AU-scaled three-iteration solve). Both
/// refinements perturb only the atmospheric-correction az/el geometry, whose
/// geodetic basis agrees to ~13 microarcseconds (~0.4 mm on the ground; see
/// `frames::tests::canonical_and_skyfield_geodetic_agree_to_sub_milliarcsecond`),
/// so the converged position can only cluster well inside a millimetre. The band
/// is held at 1 mm; a divergence beyond it is a canonical bug to root-cause, not
/// a tolerance to widen.
const CANONICAL_VS_REFERENCE_SPP_TOL_M: f64 = 1.0e-3;

/// Surveyed-truth sanity bound (m): the canonical converged position vs the ESBC
/// RINEX `APPROX POSITION XYZ`. This is the same physical-truth bound the
/// reference/owned SPP solves hold on this fixture (a broadcast single-frequency
/// solve), not a bit-exact gate.
const CANONICAL_SPP_TRUTH_BOUND_M: f64 = 6.0;

/// P6 increment 1: the canonical SPP strategy, an ADDITIVE selectable strategy
/// implementing the IERS-rigorous SPP op-order (full iterative light-time with
/// the closed-form Sagnac, a meters-native WGS84 geodetic basis, on the owned
/// deterministic solver). It does not touch the reference SPP path. Both
/// canonical bars are checked here against the real ESBC broadcast first epoch:
///
///   1. DETERMINISM: canonical is bit-reproducible run-to-run on this build (the
///      frozen-bits golden below, re-asserted on a second solve). Scope caveat,
///      exactly as the owned-solver claim: the owned kernel owns only the dense
///      subproblem factorization; the surrounding normal-matrix / gradient / norm
///      reductions still ride nalgebra's CPU-dispatched dense algebra, so these
///      pinned bits are THIS build's reproducible output, with the cross-platform
///      bit guarantee scoped to the factorization, not a portable constant.
///   2. BOUNDED-TOLERANCE + TRUTH: canonical lands within
///      [`CANONICAL_VS_REFERENCE_SPP_TOL_M`] of the Skyfield-faithful reference
///      SPP on the shared case (same used satellites), and within
///      [`CANONICAL_SPP_TRUTH_BOUND_M`] of the surveyed RINEX truth.
#[test]
fn canonical_spp_is_deterministic_bounded_and_truthful() {
    use crate::estimation::strategies::{estimate, EstimateInput, EstimateOptions, EstimateOutput};
    use crate::estimation::{StrategyId, Technique};

    let store = esbc_broadcast_store();
    let (inputs, truth) = esbc_first_epoch_inputs([3_582_135.0, 532_569.0, 5_232_779.0, 0.0]);
    let policy = SolvePolicy::default();

    let run_canonical = || -> super::ReceiverSolution {
        match estimate(
            EstimateInput::Spp {
                eph: &store,
                inputs: &inputs,
                with_geodetic: true,
                policy,
            },
            EstimateOptions::new(StrategyId::Canonical {
                technique: Technique::Spp,
            }),
        )
        .expect("canonical SPP solves")
        {
            EstimateOutput::Spp(solution) => *solution,
            other => panic!("canonical SPP must yield an SPP solution, got {other:?}"),
        }
    };

    let canonical = run_canonical();

    // Reference SPP (Skyfield-faithful) for the bounded-tolerance comparison;
    // this is the unchanged reference path, proving canonical is additive.
    let reference = solve_with_policy(&store, &inputs, true, policy).expect("reference SPP");

    // The bounded-tolerance bar only compares like with like: canonical and the
    // reference must select the same satellites (the geodetic basis difference is
    // far from the elevation-mask boundary, so the frozen selection is identical).
    assert_eq!(
        canonical.used_sats, reference.used_sats,
        "canonical and reference SPP must select the same satellites on the shared case"
    );

    // BAR 2a: bounded tolerance vs the reference.
    let dpos = position_error_m(&canonical, reference.position.as_array());
    let dclock = (canonical.rx_clock_s - reference.rx_clock_s).abs() * super::C_M_S;
    assert!(
        dclock < CANONICAL_VS_REFERENCE_SPP_TOL_M,
        "canonical SPP clock diverged from reference by {dclock} m (> {CANONICAL_VS_REFERENCE_SPP_TOL_M} m)"
    );
    assert!(
        dpos < CANONICAL_VS_REFERENCE_SPP_TOL_M,
        "canonical SPP diverged from reference by {dpos} m (> {CANONICAL_VS_REFERENCE_SPP_TOL_M} m); root-cause, do not widen"
    );

    // BAR 2b: surveyed-truth sanity bound.
    let terr = position_error_m(&canonical, truth);
    assert!(
        terr < CANONICAL_SPP_TRUTH_BOUND_M,
        "canonical SPP truth error was {terr} m (> {CANONICAL_SPP_TRUTH_BOUND_M} m)"
    );

    // BAR 1: frozen-bits determinism golden (this build's reproducible output).
    assert_eq!(canonical.position.x_m.to_bits(), 0x414b544cd339bab6);
    assert_eq!(canonical.position.y_m.to_bits(), 0x412040dc03055a75);
    assert_eq!(canonical.position.z_m.to_bits(), 0x4153f61de1d7513a);
    assert_eq!(canonical.rx_clock_s.to_bits(), 0x3f3f84ebef550522);

    // Determinism: a second canonical solve is bit-identical.
    let again = run_canonical();
    assert_eq!(
        canonical.position.x_m.to_bits(),
        again.position.x_m.to_bits()
    );
    assert_eq!(
        canonical.position.y_m.to_bits(),
        again.position.y_m.to_bits()
    );
    assert_eq!(
        canonical.position.z_m.to_bits(),
        again.position.z_m.to_bits()
    );
    assert_eq!(canonical.rx_clock_s.to_bits(), again.rx_clock_s.to_bits());
}

// ---------------------------------------------------------------------------
// Batch SPP: the parallel fan-out must be bit-identical to the serial path and
// to per-epoch solve_with_policy, since epochs are independent.
// ---------------------------------------------------------------------------

#[test]
fn spp_batch_parallel_is_bit_identical_to_serial_and_per_epoch() {
    let sp3 = sp3();
    let policy = SolvePolicy::default();
    let with_geodetic = true;

    // A handful of independent receive epochs (distinct valid input sets), each
    // a self-contained SolveInputs the serial solver already converges on.
    let epochs: Vec<SolveInputs> = ["L0_minimal", "L1_iono", "L2_tropo"]
        .iter()
        .map(|level| {
            let doc = read_fixture(&format!("spp_trace_{level}.json"));
            solve_inputs(&load_inputs(&doc, level))
        })
        .collect();

    let serial = solve_spp_batch_serial(&sp3, &epochs, with_geodetic, policy);
    let parallel = solve_spp_batch_parallel(&sp3, &epochs, with_geodetic, policy);

    assert_eq!(serial.len(), epochs.len());
    assert_eq!(parallel.len(), epochs.len());

    for (i, inputs) in epochs.iter().enumerate() {
        // Reference: the single-epoch path the batch must reproduce element-wise.
        let reference = solve_with_policy(&sp3, inputs, with_geodetic, policy);

        let s = serial[i].as_ref().expect("serial epoch solves");
        let p = parallel[i].as_ref().expect("parallel epoch solves");
        let r = reference.as_ref().expect("reference epoch solves");

        for (axis, ((sa, pa), ra)) in s
            .position
            .as_array()
            .iter()
            .zip(p.position.as_array().iter())
            .zip(r.position.as_array().iter())
            .enumerate()
        {
            assert_eq!(sa.to_bits(), ra.to_bits(), "epoch {i} axis {axis} serial");
            assert_eq!(pa.to_bits(), ra.to_bits(), "epoch {i} axis {axis} parallel");
        }
        assert_eq!(
            s.rx_clock_s.to_bits(),
            r.rx_clock_s.to_bits(),
            "epoch {i} clock serial"
        );
        assert_eq!(
            p.rx_clock_s.to_bits(),
            r.rx_clock_s.to_bits(),
            "epoch {i} clock parallel"
        );

        // Whole-solution equality, not just position/clock: the Debug repr
        // renders every field (geodetic, per-system clocks, used/rejected sats,
        // residuals, metadata, DOP) with round-trip-exact float formatting, so
        // matching Debug strings means byte-for-byte identical solutions. This
        // backs the "bit-identical" contract across the full ReceiverSolution.
        let s_dbg = format!("{s:?}");
        let p_dbg = format!("{p:?}");
        let r_dbg = format!("{r:?}");
        assert_eq!(s_dbg, r_dbg, "epoch {i} full solution serial");
        assert_eq!(p_dbg, r_dbg, "epoch {i} full solution parallel");
    }
}

// ---------------------------------------------------------------------------
// GLONASS FDMA ionosphere-scaling: deterministic, position-solve-free unit
// tests of the measurement-model ionosphere term itself.
//
// The position-level RTKLIB oracle cannot see the FDMA scaling: its effect
// (~3% of the slant iono delay, ~10-16 cm) is absorbed by the free per-system
// GLONASS receiver clock and sits far below the meter-level cross-implementation
// agreement floor. These tests instead evaluate `sat_model` (the path that
// calls `klobuchar_native_unchecked` via `spp_iono_frequency_hz`) directly and
// assert the GLONASS iono delay is EXACTLY `(f_L1 / f_k)^2` times the GPS iono
// delay computed at the identical geometry, with no solver or BLAS in the loop.
// They fail if the scaling is removed or if GLONASS were to use the GPS L1
// carrier.
// ---------------------------------------------------------------------------

/// Ephemeris stub that returns the SAME satellite ECEF position and zero clock
/// for every satellite and every epoch. This forces a GPS and a GLONASS
/// satellite through bit-identical light-time / Sagnac / az-el geometry, so the
/// only difference in their modeled ionosphere delay is the carrier-frequency
/// scaling under test.
struct FixedSat {
    pos_ecef_m: [f64; 3],
}

impl super::EphemerisSource for FixedSat {
    fn position_clock_at_j2000_s(
        &self,
        _sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        Some((self.pos_ecef_m, 0.0))
    }
}

/// Nonzero broadcast Klobuchar coefficients (the ESBC00DNK nav header GPSA/GPSB
/// set) so the modeled delay is well clear of zero.
fn nonzero_klobuchar() -> KlobucharCoeffs {
    KlobucharCoeffs {
        alpha: [4.6566e-09, 1.4901e-08, -5.9605e-08, -1.1921e-07],
        beta: [8.1920e+04, 9.8304e+04, -6.5536e+04, -5.2429e+05],
    }
}

/// Receiver near the Earth's surface, and a satellite placed along the receiver
/// radial at GNSS altitude so it is high in the sky (a real, positive-elevation
/// geometry that produces a physically meaningful, nonzero Klobuchar delay).
fn fdma_geometry() -> ([f64; 3], FixedSat) {
    let rx = [3_582_110.0_f64, 532_590.0, 5_232_765.0];
    let rx_norm = (rx[0] * rx[0] + rx[1] * rx[1] + rx[2] * rx[2]).sqrt();
    let r_sat = 25_500_000.0_f64;
    let sat = FixedSat {
        pos_ecef_m: [
            rx[0] / rx_norm * r_sat,
            rx[1] / rx_norm * r_sat,
            rx[2] / rx_norm * r_sat,
        ],
    };
    (rx, sat)
}

/// Evaluate the SPP measurement model's ionosphere term for one satellite at the
/// fixed geometry, with the broadcast Klobuchar correction enabled.
fn iono_term_m(
    eph: &FixedSat,
    rx: [f64; 3],
    sat: GnssSatelliteId,
    klobuchar: KlobucharCoeffs,
    glonass_channels: &std::collections::BTreeMap<u8, i8>,
) -> f64 {
    let met = SurfaceMet {
        pressure_hpa: 1013.25,
        temperature_k: 288.15,
        relative_humidity: 0.5,
    };
    let env = SatModelEnv {
        eph,
        t_rx_j2000_s: 0.0,
        t_rx_second_of_day_s: 43_200.0,
        day_of_year: 177.0,
        corrections: Corrections::IONO,
        met: &met,
        glonass_channels,
        model: SppModelRecipe::reference(),
    };
    test_support::sat_model_with_ionosphere_for_test(
        &env,
        sat,
        rx,
        0.0,
        22_000_000.0,
        SppIonosphere::Klobuchar(klobuchar),
    )
    .expect("fixed-position ephemeris always models the satellite")
    .iono_m
}

#[test]
fn glonass_iono_is_exactly_fdma_scaled_gps_l1_delay() {
    let (rx, eph) = fdma_geometry();
    let klobuchar = nonzero_klobuchar();
    let empty = std::collections::BTreeMap::<u8, i8>::new();

    // GPS reference satellite: its carrier is L1, so the (f_L1 / f)^2 factor is
    // exactly 1 and its modeled iono delay is the base L1 slant delay.
    let gps = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid GPS id");
    let gps_iono = iono_term_m(&eph, rx, gps, klobuchar, &empty);
    assert!(
        gps_iono > 0.5,
        "expected a clearly nonzero GPS L1 Klobuchar delay, got {gps_iono} m"
    );

    let f_l1 =
        crate::frequencies::frequency_hz(GnssSystem::Gps, crate::frequencies::CarrierBand::L1)
            .expect("canonical GPS L1 carrier");

    // For every valid FDMA channel, the GLONASS iono delay must equal the GPS L1
    // delay scaled by exactly (f_L1 / f_k)^2 -- bit for bit, because the only
    // difference along the identical geometry is this dispersive factor.
    for k in [-7_i8, -4, -1, 0, 3, 6] {
        let glonass = GnssSatelliteId::new(GnssSystem::Glonass, 7).expect("valid GLONASS id");
        let mut channels = std::collections::BTreeMap::new();
        channels.insert(7u8, k);
        let glo_iono = iono_term_m(&eph, rx, glonass, klobuchar, &channels);

        let f_k = crate::frequencies::glonass_g1_frequency_hz(k);
        let ratio = f_l1 / f_k;
        let expected = gps_iono * (ratio * ratio);
        assert_eq!(
            glo_iono.to_bits(),
            expected.to_bits(),
            "k={k}: GLONASS iono must be exactly (f_L1/f_k)^2 * GPS L1 iono \
             (got {glo_iono}, expected {expected})"
        );

        // Every GLONASS G1 carrier is above L1, so the scaled delay is strictly
        // smaller than the GPS L1 delay. If the scaling were removed or used the
        // GPS L1 carrier, these would be equal.
        assert!(
            glo_iono < gps_iono,
            "k={k}: GLONASS G1 (> L1) must scale the delay DOWN, got {glo_iono} >= {gps_iono}"
        );
    }
}

#[test]
fn glonass_iono_changes_monotonically_with_channel() {
    let (rx, eph) = fdma_geometry();
    let klobuchar = nonzero_klobuchar();

    let delay_for_channel = |k: i8| {
        let glonass = GnssSatelliteId::new(GnssSystem::Glonass, 7).expect("valid GLONASS id");
        let mut channels = std::collections::BTreeMap::new();
        channels.insert(7u8, k);
        iono_term_m(&eph, rx, glonass, klobuchar, &channels)
    };

    // A lower channel number means a lower G1 carrier, hence a LARGER (f_L1/f_k)^2
    // factor and a larger iono delay. The term must move in that direction and by
    // the channel-frequency ratio, not stay constant (which a non-FDMA model
    // would).
    let low = delay_for_channel(-7);
    let mid = delay_for_channel(0);
    let high = delay_for_channel(6);
    assert!(
        low > mid && mid > high,
        "iono delay must strictly decrease with channel: k=-7 {low}, k=0 {mid}, k=6 {high}"
    );

    // The ratio between two channels' delays must equal the ratio of their
    // squared carriers (the base L1 delay cancels).
    let f_low = crate::frequencies::glonass_g1_frequency_hz(-7);
    let f_high = crate::frequencies::glonass_g1_frequency_hz(6);
    // delay scales as 1/f^2, so delay(k=6)/delay(k=-7) = (f_low/f_high)^2.
    let expected_ratio = (f_low / f_high) * (f_low / f_high);
    let got_ratio = high / low;
    assert!(
        (got_ratio - expected_ratio).abs() < 1e-12,
        "delay ratio {got_ratio} must match carrier-squared ratio {expected_ratio}"
    );
}

// ---------------------------------------------------------------------------
// GLONASS channel validation at the SPP boundary: an observed GLONASS satellite
// whose FDMA channel is missing OR outside the valid [-7, +6] range is rejected
// with a typed `IonosphereUnsupported` error when the ionosphere is enabled,
// rather than silently scaling against a bogus-but-positive carrier frequency.
// ---------------------------------------------------------------------------

fn glonass_validation_inputs(channels: std::collections::BTreeMap<u8, i8>) -> SolveInputs {
    SolveInputs {
        observations: vec![Observation {
            satellite_id: GnssSatelliteId::new(GnssSystem::Glonass, 7).expect("valid GLONASS id"),
            pseudorange_m: 22_000_000.0,
        }],
        t_rx_j2000_s: 0.0,
        t_rx_second_of_day_s: 43_200.0,
        day_of_year: 177.0,
        initial_guess: [3_582_110.0, 532_590.0, 5_232_765.0, 0.0],
        corrections: Corrections::IONO,
        klobuchar: nonzero_klobuchar(),
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: channels,
        met: SurfaceMet {
            pressure_hpa: 1013.25,
            temperature_k: 288.15,
            relative_humidity: 0.5,
        },
        robust: None,
    }
}

#[test]
fn glonass_out_of_range_channel_is_rejected() {
    let (_rx, eph) = fdma_geometry();
    let glonass = GnssSatelliteId::new(GnssSystem::Glonass, 7).expect("valid GLONASS id");

    // Channel 99 is outside the valid [-7, +6] FDMA range. glonass_g1_frequency_hz
    // would return a bogus-but-positive carrier for it, so the boundary must
    // reject the satellite instead.
    let mut channels = std::collections::BTreeMap::new();
    channels.insert(7u8, 99i8);
    let err = super::solve(&eph, &glonass_validation_inputs(channels), false)
        .expect_err("out-of-range GLONASS channel must be rejected");
    assert!(
        matches!(err, SppError::IonosphereUnsupported { satellite } if satellite == glonass),
        "expected IonosphereUnsupported for the out-of-range GLONASS sat, got {err:?}"
    );

    // The low end is rejected the same way.
    let mut channels = std::collections::BTreeMap::new();
    channels.insert(7u8, -8i8);
    let err = super::solve(&eph, &glonass_validation_inputs(channels), false)
        .expect_err("channel -8 is below the valid range");
    assert!(
        matches!(err, SppError::IonosphereUnsupported { satellite } if satellite == glonass),
        "expected IonosphereUnsupported for channel -8, got {err:?}"
    );
}

#[test]
fn glonass_missing_channel_is_rejected() {
    let (_rx, eph) = fdma_geometry();
    let glonass = GnssSatelliteId::new(GnssSystem::Glonass, 7).expect("valid GLONASS id");

    // No channel entry for the observed GLONASS satellite: still unresolvable.
    let err = super::solve(
        &eph,
        &glonass_validation_inputs(std::collections::BTreeMap::new()),
        false,
    )
    .expect_err("missing GLONASS channel must be rejected");
    assert!(
        matches!(err, SppError::IonosphereUnsupported { satellite } if satellite == glonass),
        "expected IonosphereUnsupported for the channel-less GLONASS sat, got {err:?}"
    );
}

#[test]
fn glonass_boundary_channels_are_accepted_for_iono_scaling() {
    // The extremes of the valid range must resolve (no false rejection). A
    // single-satellite solve is underdetermined, so the solve fails LATER with
    // TooFewSatellites, never with IonosphereUnsupported.
    let (_rx, eph) = fdma_geometry();
    for k in [-7i8, 6i8] {
        let mut channels = std::collections::BTreeMap::new();
        channels.insert(7u8, k);
        let err = super::solve(&eph, &glonass_validation_inputs(channels), false)
            .expect_err("one satellite cannot determine a position");
        assert!(
            matches!(err, SppError::TooFewSatellites { .. }),
            "valid channel k={k} must pass the ionosphere gate (got {err:?})"
        );
    }
}
