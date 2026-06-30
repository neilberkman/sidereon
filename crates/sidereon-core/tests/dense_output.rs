use nalgebra::Vector3;
use sidereon_core::astro::forces::TwoBodyGravity;
use sidereon_core::astro::integrators::{Integrator, DP54};
use sidereon_core::astro::propagator::{
    api::IntegratorOptions, OrbitalDynamics, PropagationContext,
};
use sidereon_core::astro::state::CartesianState;

#[test]
fn test_dense_output_endpoint_exactness() {
    let mu: f64 = 398600.4418;
    let r_mag: f64 = 7000.0;
    let v_mag: f64 = (mu / r_mag).sqrt();

    let initial_state = CartesianState {
        epoch_tdb_seconds: 0.0,
        position_km: Vector3::new(r_mag, 0.0, 0.0),
        velocity_km_s: Vector3::new(0.0, v_mag, 0.0),
    };

    let force = TwoBodyGravity::default();
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let integrator = DP54;
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        abs_tol: 1e-12,
        rel_tol: 1e-12,
        dense_output: true,
        ..IntegratorOptions::default()
    };

    let result = integrator
        .propagate(initial_state, 3600.0, &dynamics, &ctx, &opts)
        .expect("Propagation failed");
    let dense = result.dense.expect("Dense output missing");

    for (i, seg) in dense.segments.iter().enumerate() {
        // Evaluate at start
        let y_start_eval = seg.eval(seg.t_start).expect("Eval failed");
        assert_eq!(
            y_start_eval.position_km, seg.y_start.position_km,
            "Start position bit-exactly fail at step {}",
            i
        );
        assert_eq!(
            y_start_eval.velocity_km_s, seg.y_start.velocity_km_s,
            "Start velocity bit-exactly fail at step {}",
            i
        );

        // Evaluate at end
        let y_end_eval = seg.eval(seg.t_end()).expect("Eval failed");
        let next_point = &result.points[i + 1];

        // Check if bit-exact with the point recorded at step end
        assert_eq!(
            y_end_eval.position_km.x, next_point.position_km[0],
            "End position X bit-exactly fail at step {}",
            i
        );
        assert_eq!(
            y_end_eval.position_km.y, next_point.position_km[1],
            "End position Y bit-exactly fail at step {}",
            i
        );
        assert_eq!(
            y_end_eval.position_km.z, next_point.position_km[2],
            "End position Z bit-exactly fail at step {}",
            i
        );
    }
}

#[test]
fn test_dense_output_circular_orbit_parity() {
    let mu: f64 = 398600.4418;
    let r_mag: f64 = 7000.0;
    let v_mag: f64 = (mu / r_mag).sqrt();
    let n = (mu / r_mag.powi(3)).sqrt();

    let initial_state = CartesianState {
        epoch_tdb_seconds: 0.0,
        position_km: Vector3::new(r_mag, 0.0, 0.0),
        velocity_km_s: Vector3::new(0.0, v_mag, 0.0),
    };

    let force = TwoBodyGravity::default();
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let integrator = DP54;
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        abs_tol: 1e-12,
        rel_tol: 1e-12,
        dense_output: true,
        ..IntegratorOptions::default()
    };

    let t_end = 3600.0;
    let result = integrator
        .propagate(initial_state, t_end, &dynamics, &ctx, &opts)
        .expect("Propagation failed");
    let dense = result.dense.expect("Dense output missing");

    // Evaluate at 50 intermediate times
    for i in 0..=50 {
        let t = (i as f64) / 50.0 * t_end;
        let interpolated = dense.eval(t).expect("Interpolation failed");

        // Analytical circular orbit
        let theta = n * t;
        let expected_pos = Vector3::new(r_mag * theta.cos(), r_mag * theta.sin(), 0.0);
        let expected_vel = Vector3::new(-v_mag * theta.sin(), v_mag * theta.cos(), 0.0);

        let pos_err = (interpolated.position_km - expected_pos).norm();
        let vel_err = (interpolated.velocity_km_s - expected_vel).norm();

        // Interpolation error should be bounded by tolerance (approx)
        // Actually, DP54 with 1e-12 should be very accurate.
        assert!(
            pos_err < 1e-7,
            "Position error too large at t={}: {}",
            t,
            pos_err
        );
        assert!(
            vel_err < 1e-10,
            "Velocity error too large at t={}: {}",
            t,
            vel_err
        );
    }
}

#[test]
fn test_dense_output_elliptic_orbit_invariants() {
    let mu: f64 = 398600.4418;
    let initial_state = CartesianState {
        epoch_tdb_seconds: 0.0,
        position_km: Vector3::new(7000.0, 0.0, 0.0),
        velocity_km_s: Vector3::new(0.0, 8.5, 0.0),
    };

    let initial_energy =
        initial_state.velocity_km_s.norm_squared() / 2.0 - mu / initial_state.position_km.norm();

    let force = TwoBodyGravity::default();
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let integrator = DP54;
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        abs_tol: 1e-12,
        rel_tol: 1e-12,
        dense_output: true,
        ..IntegratorOptions::default()
    };

    let t_end = 5000.0;
    let result = integrator
        .propagate(initial_state, t_end, &dynamics, &ctx, &opts)
        .expect("Propagation failed");
    let dense = result.dense.expect("Dense output missing");

    // Evaluate at many times and check energy conservation
    for i in 0..=100 {
        let t = (i as f64) / 100.0 * t_end;
        let y = dense.eval(t).expect("Interpolation failed");

        let energy = y.velocity_km_s.norm_squared() / 2.0 - mu / y.position_km.norm();
        let energy_err = (energy - initial_energy).abs();

        // Energy drift should be small even at interpolated points
        assert!(
            energy_err < 1e-8,
            "Energy drift too large at t={}: {}",
            t,
            energy_err
        );
    }
}

#[test]
fn test_dense_output_monotonic_continuity() {
    let mu: f64 = MU_EARTH_DEFAULT;
    let r_mag: f64 = 7000.0;
    let v_mag: f64 = (mu / r_mag).sqrt();

    let initial_state = CartesianState {
        epoch_tdb_seconds: 0.0,
        position_km: Vector3::new(r_mag, 0.0, 0.0),
        velocity_km_s: Vector3::new(0.0, v_mag, 0.0),
    };

    let force = TwoBodyGravity::default();
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let integrator = DP54;
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        abs_tol: 1e-10,
        rel_tol: 1e-10,
        dense_output: true,
        ..IntegratorOptions::default()
    };

    let t_end = 10000.0;
    let result = integrator
        .propagate(initial_state, t_end, &dynamics, &ctx, &opts)
        .expect("Propagation failed");
    let dense = result.dense.expect("Dense output missing");

    for i in 0..=1000 {
        let t = (i as f64) / 1000.0 * t_end;
        let _y = dense.eval(t).expect("Interpolation failed");
    }

    // Check boundaries specifically
    for i in 0..dense.segments.len() - 1 {
        let t_boundary = dense.segments[i].t_end();
        let y_left = dense.segments[i].eval(t_boundary).unwrap();
        let y_right = dense.segments[i + 1].eval(t_boundary).unwrap();

        let jump_pos = (y_left.position_km - y_right.position_km).norm();
        let jump_vel = (y_left.velocity_km_s - y_right.velocity_km_s).norm();

        assert!(
            jump_pos < 1e-11,
            "Boundary jump pos too large at t={}: {}",
            t_boundary,
            jump_pos
        );
        assert!(
            jump_vel < 1e-13,
            "Boundary jump vel too large at t={}: {}",
            t_boundary,
            jump_vel
        );
    }
}

#[test]
fn test_dense_output_range_rejection() {
    let initial_state = CartesianState {
        epoch_tdb_seconds: 100.0,
        position_km: Vector3::new(7000.0, 0.0, 0.0),
        velocity_km_s: Vector3::new(0.0, 7.5, 0.0),
    };

    let force = TwoBodyGravity::default();
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let integrator = DP54;
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        dense_output: true,
        ..IntegratorOptions::default()
    };

    let result = integrator
        .propagate(initial_state, 200.0, &dynamics, &ctx, &opts)
        .expect("Propagation failed");
    let dense = result.dense.expect("Dense output missing");

    assert!(dense.eval(99.0).is_err(), "Should reject time before start");
    assert!(dense.eval(201.0).is_err(), "Should reject time after end");
    assert!(dense.eval(150.0).is_ok(), "Should accept time in range");
}

const MU_EARTH_DEFAULT: f64 = 398600.4418;
