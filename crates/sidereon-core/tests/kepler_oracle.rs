use nalgebra::Vector3;
use sidereon_core::astro::forces::TwoBodyGravity;
use sidereon_core::astro::integrators::{Integrator, DP54};
use sidereon_core::astro::propagator::{
    api::IntegratorOptions, OrbitalDynamics, PropagationContext,
};
use sidereon_core::astro::state::CartesianState;

#[test]
fn test_kepler_circular_orbit_full_period() {
    // Reference values for a 7000km circular orbit
    let r_mag: f64 = 7000.0;
    let mu: f64 = 398600.4418;
    let v_mag: f64 = (mu / r_mag).sqrt();
    let period = 2.0 * std::f64::consts::PI * (r_mag.powi(3) / mu).sqrt();

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
        ..IntegratorOptions::default()
    };

    let result = integrator
        .propagate(initial_state, period, &dynamics, &ctx, &opts)
        .expect("Propagation failed");

    let final_pos = result.final_state.position_km;
    let final_vel = result.final_state.velocity_km_s;

    // Position return parity (within 1e-7 km)
    assert!(
        (final_pos.x - r_mag).abs() < 1e-7,
        "X position error: {}",
        (final_pos.x - r_mag).abs()
    );
    assert!(
        final_pos.y.abs() < 1e-7,
        "Y position error: {}",
        final_pos.y.abs()
    );
    assert!(
        final_pos.z.abs() < 1e-15,
        "Z position error: {}",
        final_pos.z.abs()
    );

    // Velocity return parity
    assert!(
        final_vel.x.abs() < 1e-7,
        "X velocity error: {}",
        final_vel.x.abs()
    );
    assert!(
        (final_vel.y - v_mag).abs() < 1e-7,
        "Y velocity error: {}",
        (final_vel.y - v_mag).abs()
    );
    assert!(
        final_vel.z.abs() < 1e-15,
        "Z velocity error: {}",
        final_vel.z.abs()
    );
}

#[test]
fn test_kepler_elliptic_orbit_invariants() {
    // Eccentric orbit
    let mu: f64 = 398600.4418;
    let r0 = Vector3::new(7000.0, 0.0, 0.0);
    let v0 = Vector3::new(0.0, 8.5, 0.0); // Slightly above circular velocity

    let initial_state = CartesianState {
        epoch_tdb_seconds: 0.0,
        position_km: r0,
        velocity_km_s: v0,
    };

    // Invariants
    let initial_energy = v0.norm_squared() / 2.0 - mu / r0.norm();
    let initial_angular_momentum = r0.cross(&v0);

    let force = TwoBodyGravity::default();
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let integrator = DP54;
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        abs_tol: 1e-13,
        rel_tol: 1e-13,
        ..IntegratorOptions::default()
    };

    // Propagate half an orbit (approx)
    let t_end = 3600.0;
    let result = integrator
        .propagate(initial_state, t_end, &dynamics, &ctx, &opts)
        .expect("Propagation failed");

    let final_pos = result.final_state.position_km;
    let final_vel = result.final_state.velocity_km_s;

    let final_energy = final_vel.norm_squared() / 2.0 - mu / final_pos.norm();
    let final_angular_momentum = final_pos.cross(&final_vel);

    // Energy conservation (integral of motion)
    assert!(
        (final_energy - initial_energy).abs() < 1e-10,
        "Energy drift: {}",
        (final_energy - initial_energy).abs()
    );

    // Angular momentum conservation (integral of motion)
    let am_diff = (final_angular_momentum - initial_angular_momentum).norm();
    assert!(am_diff < 1e-8, "Angular momentum drift: {}", am_diff);
}

#[test]
fn test_j2_secular_drift_oracle() {
    use sidereon_core::astro::constants::{J2_EARTH, MU_EARTH, RE_EARTH};
    use sidereon_core::astro::forces::J2Gravity;

    // Sun-synchronous-like orbit
    let r_mag: f64 = 7000.0;
    let inc_deg: f64 = 98.0;
    let inc_rad = inc_deg.to_radians();
    let mu = MU_EARTH;
    let re = RE_EARTH;
    let j2 = J2_EARTH;

    let v_mag = (mu / r_mag).sqrt();
    let initial_state = CartesianState {
        epoch_tdb_seconds: 0.0,
        position_km: Vector3::new(r_mag, 0.0, 0.0),
        velocity_km_s: Vector3::new(0.0, v_mag * inc_rad.cos(), v_mag * inc_rad.sin()),
    };

    let mut forces = sidereon_core::astro::forces::CompositeForceModel::new();
    forces.add(Box::new(TwoBodyGravity::default()));
    forces.add(Box::new(J2Gravity::default()));

    let dynamics = OrbitalDynamics {
        force_model: &forces,
    };
    let integrator = DP54;
    let ctx = PropagationContext::default();
    let opts = IntegratorOptions {
        abs_tol: 1e-12,
        rel_tol: 1e-12,
        ..IntegratorOptions::default()
    };

    // Propagate for one day
    let t_end = 86400.0;
    let result = integrator
        .propagate(initial_state, t_end, &dynamics, &ctx, &opts)
        .expect("Propagation failed");

    let final_pos = result.final_state.position_km;
    let final_vel = result.final_state.velocity_km_s;

    // Analytical J2 drift for circular orbit (e=0, p=a=r)
    let n = (mu / r_mag.powi(3)).sqrt();
    let p = r_mag;
    let raan_drift_rate = -1.5 * j2 * (re / p).powi(2) * n * inc_rad.cos();

    let expected_raan_drift = raan_drift_rate * t_end;

    // Calculate actual RAAN drift from final state
    // Initial RAAN is 0 (pos is on X axis, velocity has Y and Z components)
    // Wait, if pos is [r, 0, 0], and vel is [0, vy, vz],
    // angular momentum h = r x v = [0, -r*vz, r*vy]
    // node vector n = K x h = [-r*vy, -r*vz, 0]
    // RAAN = atan2(n.y, n.x)
    let h_vec = initial_state
        .position_km
        .cross(&initial_state.velocity_km_s);
    let n_vec = Vector3::new(0.0, 0.0, 1.0).cross(&h_vec);
    let initial_raan = n_vec.y.atan2(n_vec.x);

    let h_final = final_pos.cross(&final_vel);
    let n_final = Vector3::new(0.0, 0.0, 1.0).cross(&h_final);
    let final_raan = n_final.y.atan2(n_final.x);

    let mut actual_raan_drift = final_raan - initial_raan;
    while actual_raan_drift > std::f64::consts::PI {
        actual_raan_drift -= 2.0 * std::f64::consts::PI;
    }
    while actual_raan_drift < -std::f64::consts::PI {
        actual_raan_drift += 2.0 * std::f64::consts::PI;
    }

    // J2 drift should be within 1% of analytical for this simple case over one day
    // (Analytical formula is a first-order approximation)
    let drift_diff = (actual_raan_drift - expected_raan_drift).abs();
    assert!(
        drift_diff < expected_raan_drift.abs() * 0.01,
        "RAAN drift error too large: actual={}, expected={}, diff={}",
        actual_raan_drift,
        expected_raan_drift,
        drift_diff
    );
}
