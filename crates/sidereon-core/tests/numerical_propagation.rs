#![cfg(sidereon_repo_tests)]
//! Validates the high-level numerical state-vector propagator
//! ([`sidereon_core::astro::propagator::StatePropagator`]) against an
//! INDEPENDENT analytic oracle and against the integrators called directly.
//!
//! There is no external numerical-propagator oracle (Skyfield has none), so the
//! truth anchors here are, per the feature brief:
//!
//! 1. a pure two-body (no J2) case vs the closed-form Kepler solution, computed
//!    independently with the universal-variable f/g formulation (Vallado,
//!    "Fundamentals of Astrodynamics and Applications", Algorithm 8), to a tight
//!    bound;
//! 2. an energy / angular-momentum conservation and forward-then-back round-trip
//!    check;
//! 3. bit-for-bit agreement of the entry point with the underlying integrator
//!    invoked directly (also covered by the crate-internal unit test).
//!
//! The universal-variable propagator below is written from the textbook
//! algorithm and shares no code with the integration engine, so it is a genuine
//! cross-check rather than a tautology.

use std::path::PathBuf;

use nalgebra::Vector3;
use sidereon_core::astro::constants::MU_EARTH;
use sidereon_core::astro::error::PropagationError;
use sidereon_core::astro::forces::TwoBodyGravity;
use sidereon_core::astro::integrators::{DynamicsModel, Integrator, DP54};
use sidereon_core::astro::propagator::api::{IntegratorOptions, PropagationContext};
use sidereon_core::astro::propagator::{
    ForceModelKind, IntegratorKind, OrbitalDynamics, StatePropagator,
};
use sidereon_core::astro::state::{CartesianState, StateDerivative};

/// Stumpff functions c2(psi), c3(psi) for the universal-variable Kepler solve.
fn stumpff(psi: f64) -> (f64, f64) {
    if psi > 1.0e-6 {
        let s = psi.sqrt();
        let c2 = (1.0 - s.cos()) / psi;
        let c3 = (s - s.sin()) / (psi * s);
        (c2, c3)
    } else if psi < -1.0e-6 {
        let s = (-psi).sqrt();
        let c2 = (s.cosh() - 1.0) / (-psi);
        let c3 = (s.sinh() - s) / ((-psi) * s);
        (c2, c3)
    } else {
        // Series expansion near psi = 0.
        let c2 = 0.5 - psi / 24.0 + psi * psi / 720.0;
        let c3 = 1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0;
        (c2, c3)
    }
}

/// Analytic two-body propagation via the universal-variable f/g functions.
/// Returns (position_km, velocity_km_s) at `dt` seconds after the given state.
fn kepler_universal(
    r0: Vector3<f64>,
    v0: Vector3<f64>,
    dt: f64,
    mu: f64,
) -> (Vector3<f64>, Vector3<f64>) {
    let sqrt_mu = mu.sqrt();
    let r0n = r0.norm();
    let v0n = v0.norm();
    let rdotv = r0.dot(&v0);
    let alpha = -v0n * v0n / mu + 2.0 / r0n; // 1/a

    // Initial guess for the universal anomaly chi (elliptic / hyperbolic).
    let mut chi = if alpha > 1.0e-9 {
        sqrt_mu * dt * alpha
    } else {
        // Hyperbolic guess (Vallado). Not exercised by the elliptic test cases,
        // but kept so the oracle is general.
        let a = 1.0 / alpha;
        dt.signum()
            * (-a).sqrt()
            * ((-2.0 * mu * alpha * dt)
                / (rdotv + dt.signum() * (-mu * a).sqrt() * (1.0 - r0n * alpha)))
                .ln()
    };

    let mut psi;
    let mut c2;
    let mut c3;
    let mut r;
    for _ in 0..200 {
        psi = chi * chi * alpha;
        let (cc2, cc3) = stumpff(psi);
        c2 = cc2;
        c3 = cc3;
        r = chi * chi * c2 + rdotv / sqrt_mu * chi * (1.0 - psi * c3) + r0n * (1.0 - psi * c2);
        let chi_next = chi
            + (sqrt_mu * dt
                - chi * chi * chi * c3
                - rdotv / sqrt_mu * chi * chi * c2
                - r0n * chi * (1.0 - psi * c3))
                / r;
        if (chi_next - chi).abs() < 1.0e-12 {
            chi = chi_next;
            break;
        }
        chi = chi_next;
    }

    psi = chi * chi * alpha;
    let (cc2, cc3) = stumpff(psi);
    c2 = cc2;
    c3 = cc3;
    r = chi * chi * c2 + rdotv / sqrt_mu * chi * (1.0 - psi * c3) + r0n * (1.0 - psi * c2);

    let f = 1.0 - chi * chi / r0n * c2;
    let g = dt - chi * chi * chi / sqrt_mu * c3;
    let r_vec = r0 * f + v0 * g;

    let fdot = sqrt_mu / (r * r0n) * chi * (psi * c3 - 1.0);
    let gdot = 1.0 - chi * chi / r * c2;
    let v_vec = r0 * fdot + v0 * gdot;

    (r_vec, v_vec)
}

/// An eccentric LEO start state used across the oracle checks.
fn elliptic_start() -> ([f64; 3], [f64; 3]) {
    // r = 7000 km on +x, slightly above circular speed in +y plus a small +z
    // out-of-plane component -> an inclined eccentric orbit.
    ([7000.0, 0.0, 0.0], [0.5, 8.2, 1.1])
}

struct ConstantVelocityDynamics;

impl DynamicsModel for ConstantVelocityDynamics {
    fn derivative(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<StateDerivative, PropagationError> {
        Ok(StateDerivative::new(state.velocity_km_s, Vector3::zeros()))
    }
}

fn constant_velocity_state() -> CartesianState {
    CartesianState::new(0.0, [1000.0, -20.0, 7.0], [1.0, 0.25, -0.5])
}

fn dense_step_seconds(
    states: &[sidereon_core::astro::propagator::result::PropagationPoint],
) -> Vec<f64> {
    states
        .windows(2)
        .map(|window| window[1].epoch_tdb_seconds - window[0].epoch_tdb_seconds)
        .collect()
}

#[test]
fn dp54_max_step_limits_controller_growth_and_initial_step() {
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        initial_step: 5.0,
        max_step: 2.0,
        dense_output: true,
        ..IntegratorOptions::default()
    };

    let result = DP54
        .propagate(
            constant_velocity_state(),
            10.0,
            &ConstantVelocityDynamics,
            &ctx,
            &opts,
        )
        .expect("propagate");

    let steps = dense_step_seconds(&result.points);
    assert_eq!(steps.len() as u32, result.stats.accepted_steps);
    assert_eq!(result.stats.accepted_steps, 5);
    assert!(
        steps.iter().all(|step| step.abs() <= opts.max_step),
        "accepted steps exceeded max_step: {steps:?}"
    );
    assert!(steps.iter().all(|step| step.is_sign_positive()));
}

#[test]
fn dp54_max_step_preserves_backward_step_sign() {
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        initial_step: 5.0,
        max_step: 2.0,
        dense_output: true,
        ..IntegratorOptions::default()
    };

    let result = DP54
        .propagate(
            constant_velocity_state(),
            -10.0,
            &ConstantVelocityDynamics,
            &ctx,
            &opts,
        )
        .expect("propagate");

    let steps = dense_step_seconds(&result.points);
    assert_eq!(steps.len() as u32, result.stats.accepted_steps);
    assert_eq!(result.stats.accepted_steps, 5);
    assert!(
        steps.iter().all(|step| step.abs() <= opts.max_step),
        "accepted steps exceeded max_step: {steps:?}"
    );
    assert!(steps.iter().all(|step| step.is_sign_negative()));
}

#[test]
fn dp54_unbound_max_step_keeps_controller_growth() {
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        initial_step: 1.0,
        max_step: 60.0,
        dense_output: true,
        ..IntegratorOptions::default()
    };

    let result = DP54
        .propagate(
            constant_velocity_state(),
            10.0,
            &ConstantVelocityDynamics,
            &ctx,
            &opts,
        )
        .expect("propagate");

    let steps = dense_step_seconds(&result.points);
    assert_eq!(steps, vec![1.0, 6.0, 3.0]);
    assert_eq!(result.stats.accepted_steps, 3);
}

#[test]
fn two_body_matches_analytic_kepler_to_tight_bound() {
    let (pos, vel) = elliptic_start();
    let r0 = Vector3::from_column_slice(&pos);
    let v0 = Vector3::from_column_slice(&vel);

    let propagator = StatePropagator::new(
        0.0,
        pos,
        vel,
        ForceModelKind::two_body(),
        IntegratorKind::Dp54,
    )
    .with_options(IntegratorOptions {
        abs_tol: 1.0e-13,
        rel_tol: 1.0e-13,
        ..IntegratorOptions::default()
    });

    // Sample across roughly one orbital period.
    let a = 1.0 / (-v0.norm_squared() / MU_EARTH + 2.0 / r0.norm());
    let period = 2.0 * std::f64::consts::PI * (a.powi(3) / MU_EARTH).sqrt();
    let sample_dts: Vec<f64> = (1..=8).map(|i| period * i as f64 / 8.0).collect();
    let states = propagator.ephemeris(&sample_dts).expect("ephemeris");

    let mut worst_pos = 0.0_f64;
    let mut worst_vel = 0.0_f64;
    for (state, &dt) in states.iter().zip(sample_dts.iter()) {
        let (r_ref, v_ref) = kepler_universal(r0, v0, dt, MU_EARTH);
        let dpos = (state.position_km - r_ref).norm();
        let dvel = (state.velocity_km_s - v_ref).norm();
        worst_pos = worst_pos.max(dpos);
        worst_vel = worst_vel.max(dvel);
    }

    // Numerical DP54 (1e-13 tolerances) vs the closed-form solution over a full
    // period. The bound is a tight physical agreement, not a loose sanity check.
    assert!(
        worst_pos < 1.0e-6,
        "worst position error vs analytic Kepler: {worst_pos} km"
    );
    assert!(
        worst_vel < 1.0e-9,
        "worst velocity error vs analytic Kepler: {worst_vel} km/s"
    );
}

#[test]
fn two_body_conserves_energy_and_angular_momentum() {
    let (pos, vel) = elliptic_start();
    let r0 = Vector3::from_column_slice(&pos);
    let v0 = Vector3::from_column_slice(&vel);
    let e0 = v0.norm_squared() / 2.0 - MU_EARTH / r0.norm();
    let h0 = r0.cross(&v0);

    let propagator = StatePropagator::new(
        0.0,
        pos,
        vel,
        ForceModelKind::two_body(),
        IntegratorKind::Dp54,
    )
    .with_options(IntegratorOptions {
        abs_tol: 1.0e-13,
        rel_tol: 1.0e-13,
        ..IntegratorOptions::default()
    });

    let final_state = propagator
        .propagate_to(20_000.0)
        .expect("propagate")
        .final_state;
    let r = final_state.position_km;
    let v = final_state.velocity_km_s;
    let e = v.norm_squared() / 2.0 - MU_EARTH / r.norm();
    let h = r.cross(&v);

    assert!((e - e0).abs() < 1.0e-9, "energy drift: {}", (e - e0).abs());
    assert!(
        (h - h0).norm() < 1.0e-7,
        "angular-momentum drift: {}",
        (h - h0).norm()
    );
}

#[test]
fn forward_then_back_round_trips_to_start() {
    let (pos, vel) = elliptic_start();

    let forward = StatePropagator::new(
        0.0,
        pos,
        vel,
        ForceModelKind::two_body_j2(),
        IntegratorKind::Dp54,
    )
    .with_options(IntegratorOptions {
        abs_tol: 1.0e-13,
        rel_tol: 1.0e-13,
        ..IntegratorOptions::default()
    });

    let mid = forward.propagate_to(5400.0).expect("forward").final_state;

    let back = StatePropagator {
        initial: mid,
        force_model: ForceModelKind::two_body_j2(),
        integrator: IntegratorKind::Dp54,
        options: IntegratorOptions {
            abs_tol: 1.0e-13,
            rel_tol: 1.0e-13,
            ..IntegratorOptions::default()
        },
    };
    let home = back.propagate_to(0.0).expect("backward").final_state;

    let dpos = (home.position_km - Vector3::from_column_slice(&pos)).norm();
    let dvel = (home.velocity_km_s - Vector3::from_column_slice(&vel)).norm();
    assert!(dpos < 1.0e-6, "round-trip position error: {dpos} km");
    assert!(dvel < 1.0e-9, "round-trip velocity error: {dvel} km/s");
}

/// Reference parameters for the Python-binding cross-check fixture: a two-body +
/// J2 DP54 propagation of an inclined eccentric LEO, sampled every 600 s.
const DUMP_EPOCH_S: f64 = 0.0;
const DUMP_POS_KM: [f64; 3] = [7000.0, 0.0, 0.0];
const DUMP_VEL_KM_S: [f64; 3] = [0.5, 8.2, 1.1];
const DUMP_ABS_TOL: f64 = 1.0e-12;
const DUMP_REL_TOL: f64 = 1.0e-12;
const DUMP_INITIAL_STEP_S: f64 = 60.0;
const DUMP_MIN_STEP_S: f64 = 1.0e-6;
const DUMP_MAX_STEP_S: f64 = 3600.0;
const DUMP_MAX_STEPS: u32 = 1_000_000;

fn dump_propagator() -> StatePropagator {
    StatePropagator {
        initial: CartesianState::new(DUMP_EPOCH_S, DUMP_POS_KM, DUMP_VEL_KM_S),
        force_model: ForceModelKind::two_body_j2(),
        integrator: IntegratorKind::Dp54,
        options: IntegratorOptions {
            abs_tol: DUMP_ABS_TOL,
            rel_tol: DUMP_REL_TOL,
            initial_step: DUMP_INITIAL_STEP_S,
            min_step: DUMP_MIN_STEP_S,
            max_step: DUMP_MAX_STEP_S,
            max_steps: DUMP_MAX_STEPS,
            dense_output: false,
        },
    }
}

fn dump_times() -> Vec<f64> {
    (0..10).map(|i| (i as f64) * 600.0).collect()
}

#[test]
fn reference_arc_is_deterministic_and_frozen() {
    let propagator = dump_propagator();
    let times = dump_times();
    let states = propagator.ephemeris(&times).expect("ephemeris");
    assert_eq!(states.len(), 10);

    // Determinism: a second run is bit-for-bit identical.
    let again = dump_propagator().ephemeris(&times).expect("ephemeris");
    for (a, b) in states.iter().zip(again.iter()) {
        for axis in 0..3 {
            assert_eq!(a.position_km[axis].to_bits(), b.position_km[axis].to_bits());
            assert_eq!(
                a.velocity_km_s[axis].to_bits(),
                b.velocity_km_s[axis].to_bits()
            );
        }
    }

    // Frozen-bits regression lock on the last sample (full arc cross-checked
    // Python-side against the dumped fixture).
    let last = states.last().unwrap();
    assert_eq!(last.position_km.x.to_bits(), 0xc0c0_c53c_4169_956d);
    assert_eq!(last.position_km.y.to_bits(), 0xc0b2_3e93_900c_70b7);
    assert_eq!(last.position_km.z.to_bits(), 0xc083_abd6_a837_88d8);
    assert_eq!(last.velocity_km_s.x.to_bits(), 0x400e_7a9b_324f_c2f4);
    assert_eq!(last.velocity_km_s.y.to_bits(), 0xc012_7346_7da9_1578);
    assert_eq!(last.velocity_km_s.z.to_bits(), 0xbfe3_beff_5750_5a62);

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture(&times, &states);
    }
}

/// Env-gated emitter (`SIDEREON_DUMP_FIXTURES=1`) that serializes the reference
/// initial state, options, output epochs, and the engine's ephemeris (raw f64
/// plus IEEE-754 hex bits) to the JSON fixture consumed by the Python binding's
/// pytest. Never runs in a normal `cargo test`.
fn dump_fixture(times: &[f64], states: &[CartesianState]) {
    use serde_json::{json, Value};

    let hex = |v: f64| -> String { format!("0x{:016x}", v.to_bits()) };
    let hex3 = |v: [f64; 3]| -> Vec<String> { v.iter().map(|&x| hex(x)).collect() };

    let samples: Vec<Value> = times
        .iter()
        .zip(states.iter())
        .map(|(&t, s)| {
            json!({
                "time_s": t,
                "time_s_hex": hex(t),
                "position_km_hex": hex3(s.position_array()),
                "velocity_km_s_hex": hex3(s.velocity_array()),
            })
        })
        .collect();

    let doc = json!({
        "source": "reference_arc_is_deterministic_and_frozen",
        "force_model": "two_body_j2",
        "integrator": "dp54",
        "options": {
            "abs_tol": DUMP_ABS_TOL,
            "rel_tol": DUMP_REL_TOL,
            "initial_step_s": DUMP_INITIAL_STEP_S,
            "min_step_s": DUMP_MIN_STEP_S,
            "max_step_s": DUMP_MAX_STEP_S,
            "max_steps": DUMP_MAX_STEPS,
        },
        "epoch_s_hex": hex(DUMP_EPOCH_S),
        "position_km_hex": hex3(DUMP_POS_KM),
        "velocity_km_s_hex": hex3(DUMP_VEL_KM_S),
        "samples": samples,
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/numerical_propagation.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped numerical propagation fixture to {out:?}");
}

#[test]
fn entry_point_is_bit_identical_to_direct_integration() {
    let (pos, vel) = elliptic_start();
    let opts = || IntegratorOptions {
        abs_tol: 1.0e-12,
        rel_tol: 1.0e-12,
        ..IntegratorOptions::default()
    };

    let via_entry = StatePropagator::new(
        0.0,
        pos,
        vel,
        ForceModelKind::two_body(),
        IntegratorKind::Dp54,
    )
    .with_options(opts())
    .propagate_to(4000.0)
    .expect("entry")
    .final_state;

    let force = TwoBodyGravity::default();
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let ctx = PropagationContext::default();
    let via_direct = DP54
        .propagate(
            CartesianState::new(0.0, pos, vel),
            4000.0,
            &dynamics,
            &ctx,
            &opts(),
        )
        .expect("direct")
        .final_state;

    for axis in 0..3 {
        assert_eq!(
            via_entry.position_km[axis].to_bits(),
            via_direct.position_km[axis].to_bits()
        );
        assert_eq!(
            via_entry.velocity_km_s[axis].to_bits(),
            via_direct.velocity_km_s[axis].to_bits()
        );
    }
}
