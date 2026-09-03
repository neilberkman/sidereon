use sidereon_core::astro::forces::TwoBodyGravity;
use sidereon_core::astro::integrators::{Integrator, DP54};
use sidereon_core::astro::propagator::{IntegratorOptions, OrbitalDynamics, PropagationContext};
use sidereon_core::astro::{
    coe2rv, propagate_kepler, AnomalyError, CartesianState, ClassicalElements, OrbitType,
};

const MU: f64 = 398600.4418;
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const DEG: f64 = std::f64::consts::PI / 180.0;

fn angle_diff(a: f64, b: f64) -> f64 {
    let wrapped = (a - b + std::f64::consts::PI).rem_euclid(TWO_PI) - std::f64::consts::PI;
    if wrapped <= -std::f64::consts::PI {
        std::f64::consts::PI
    } else {
        wrapped
    }
}

fn assert_angle_close(got: f64, want: f64, tol: f64, label: &str) {
    let diff = angle_diff(got, want).abs();
    assert!(diff <= tol, "{label}: got {got}, want {want}, diff {diff}");
}

fn assert_vec_close(got: [f64; 3], want: [f64; 3], tol: f64, label: &str) {
    for i in 0..3 {
        assert!(
            (got[i] - want[i]).abs() <= tol,
            "{label}[{i}]: got {}, want {}, diff {}",
            got[i],
            want[i],
            (got[i] - want[i]).abs()
        );
    }
}

fn assert_bits_same(got: f64, want: f64, label: &str) {
    assert_eq!(got.to_bits(), want.to_bits(), "{label}");
}

fn eccentric_inclined() -> ClassicalElements {
    let ecc = 0.12;
    let a = 7000.0;
    let p = a * (1.0 - ecc * ecc);
    ClassicalElements::new(p, ecc, 51.0 * DEG, 40.0 * DEG, 25.0 * DEG, 10.0 * DEG)
}

#[test]
fn propagate_kepler_matches_dp54_two_body() {
    let coe = eccentric_inclined();
    let (r0, v0) = coe2rv(&coe, MU).unwrap();
    let initial = CartesianState::new(0.0, r0, v0);
    let dt = 900.0;

    let force = TwoBodyGravity { mu: MU };
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let mut opts = IntegratorOptions::default();
    opts.abs_tol = 1.0e-12;
    opts.rel_tol = 1.0e-12;
    opts.initial_step = 10.0;
    opts.max_step = 60.0;
    opts.min_step = 1.0e-12;
    let numerical = DP54
        .propagate(
            initial,
            dt,
            &dynamics,
            &PropagationContext::default(),
            &opts,
        )
        .unwrap();

    let propagated = propagate_kepler(&coe, MU, dt).unwrap();
    let (r1, v1) = coe2rv(&propagated, MU).unwrap();

    assert_vec_close(r1, numerical.final_state.position_array(), 1.0e-6, "r");
    assert_vec_close(v1, numerical.final_state.velocity_array(), 1.0e-9, "v");
}

#[test]
fn elliptic_zero_backward_and_full_period_are_consistent() {
    let coe = eccentric_inclined();
    let dt = 1200.0;

    let zero = propagate_kepler(&coe, MU, 0.0).unwrap();
    assert_bits_same(zero.p, coe.p, "p");
    assert_bits_same(zero.a, coe.a, "a");
    assert_bits_same(zero.ecc, coe.ecc, "ecc");
    assert_bits_same(zero.incl, coe.incl, "incl");
    assert_bits_same(zero.raan, coe.raan, "raan");
    assert_bits_same(zero.argp, coe.argp, "argp");
    assert_angle_close(zero.nu, coe.nu, 1.0e-13, "zero nu");
    assert_bits_same(zero.arglat, coe.arglat, "arglat");
    assert_bits_same(zero.truelon, coe.truelon, "truelon");
    assert_bits_same(zero.lonper, coe.lonper, "lonper");
    assert_eq!(zero.orbit_type, coe.orbit_type);

    let forward = propagate_kepler(&coe, MU, dt).unwrap();
    let backward = propagate_kepler(&forward, MU, -dt).unwrap();
    assert_angle_close(backward.nu, coe.nu, 1.0e-10, "forward backward nu");

    let period = TWO_PI / (MU / coe.a.powi(3)).sqrt();
    let after_period = propagate_kepler(&coe, MU, period).unwrap();
    assert_angle_close(after_period.nu, coe.nu, 1.0e-9, "period nu");
}

#[test]
fn circular_degenerate_orbits_advance_auxiliary_angles() {
    let mut inclined = ClassicalElements::new(7000.0, 0.0, 51.6 * DEG, 80.0 * DEG, 0.0, 0.0);
    inclined.orbit_type = OrbitType::CircularInclined;
    inclined.argp = f64::NAN;
    inclined.nu = f64::NAN;
    inclined.arglat = 135.0 * DEG;

    let dt = 600.0;
    let inclined_out = propagate_kepler(&inclined, MU, dt).unwrap();
    let inclined_expected =
        (inclined.arglat + (MU / inclined.a.powi(3)).sqrt() * dt).rem_euclid(TWO_PI);
    assert_angle_close(inclined_out.arglat, inclined_expected, 1.0e-13, "arglat");
    assert!(inclined_out.nu.is_nan());
    assert_bits_same(inclined_out.nu, inclined.nu, "inclined nu");
    assert_bits_same(inclined_out.argp, inclined.argp, "inclined argp");
    coe2rv(&inclined_out, MU).unwrap();

    let mut equatorial = ClassicalElements::new(8000.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    equatorial.orbit_type = OrbitType::CircularEquatorial;
    equatorial.raan = f64::NAN;
    equatorial.argp = f64::NAN;
    equatorial.nu = f64::NAN;
    equatorial.truelon = 35.0 * DEG;

    let equatorial_out = propagate_kepler(&equatorial, MU, dt).unwrap();
    let equatorial_expected =
        (equatorial.truelon + (MU / equatorial.a.powi(3)).sqrt() * dt).rem_euclid(TWO_PI);
    assert_angle_close(
        equatorial_out.truelon,
        equatorial_expected,
        1.0e-13,
        "truelon",
    );
    assert!(equatorial_out.nu.is_nan());
    assert_bits_same(equatorial_out.nu, equatorial.nu, "equatorial nu");
    assert_bits_same(equatorial_out.raan, equatorial.raan, "equatorial raan");
    assert_bits_same(equatorial_out.argp, equatorial.argp, "equatorial argp");
    coe2rv(&equatorial_out, MU).unwrap();
}

#[test]
fn open_orbit_propagation_round_trips_forward_and_backward() {
    let hyperbolic = ClassicalElements::new(10000.0, 1.5, 35.0 * DEG, 10.0 * DEG, 20.0 * DEG, 0.4);
    assert!(hyperbolic.a < 0.0);
    let hyper_forward = propagate_kepler(&hyperbolic, MU, 500.0).unwrap();
    let hyper_back = propagate_kepler(&hyper_forward, MU, -500.0).unwrap();
    assert_angle_close(hyper_back.nu, hyperbolic.nu, 1.0e-10, "hyperbolic nu");
    coe2rv(&hyper_forward, MU).unwrap();

    let parabolic = ClassicalElements::new(12000.0, 1.0, 45.0 * DEG, 30.0 * DEG, 40.0 * DEG, 0.3);
    assert!(parabolic.a.is_infinite());
    let para_forward = propagate_kepler(&parabolic, MU, 500.0).unwrap();
    let para_back = propagate_kepler(&para_forward, MU, -500.0).unwrap();
    assert_angle_close(para_back.nu, parabolic.nu, 1.0e-10, "parabolic nu");
    coe2rv(&para_forward, MU).unwrap();
}

#[test]
fn propagate_kepler_rejects_invalid_inputs() {
    let coe = eccentric_inclined();

    assert_eq!(
        propagate_kepler(&coe, f64::NAN, 0.0),
        Err(AnomalyError::NonFinite { field: "mu" })
    );
    assert_eq!(
        propagate_kepler(&coe, 0.0, 0.0),
        Err(AnomalyError::NonPositiveMu)
    );
    assert_eq!(
        propagate_kepler(&coe, MU, f64::INFINITY),
        Err(AnomalyError::NonFinite { field: "dt" })
    );

    let mut bad_p = coe;
    bad_p.p = 0.0;
    assert_eq!(
        propagate_kepler(&bad_p, MU, 0.0),
        Err(AnomalyError::NonPositiveSemiLatus)
    );

    let mut bad_ecc = coe;
    bad_ecc.ecc = -0.1;
    assert_eq!(
        propagate_kepler(&bad_ecc, MU, 0.0),
        Err(AnomalyError::NegativeEccentricity)
    );

    let mut bad_angle = coe;
    bad_angle.nu = f64::NAN;
    assert_eq!(
        propagate_kepler(&bad_angle, MU, 0.0),
        Err(AnomalyError::NonFinite { field: "nu" })
    );

    let mut bad_hyper =
        ClassicalElements::new(10000.0, 2.0, 35.0 * DEG, 10.0 * DEG, 20.0 * DEG, 0.4);
    bad_hyper.a = 10000.0;
    assert_eq!(
        propagate_kepler(&bad_hyper, MU, 0.0),
        Err(AnomalyError::InconsistentElements { field: "a" })
    );

    let mut bad_type = ClassicalElements::new(7000.0, 0.3, 51.6 * DEG, 80.0 * DEG, 0.0, 0.0);
    bad_type.orbit_type = OrbitType::CircularInclined;
    bad_type.argp = f64::NAN;
    bad_type.nu = f64::NAN;
    bad_type.arglat = 120.0 * DEG;
    assert_eq!(
        propagate_kepler(&bad_type, MU, 0.0),
        Err(AnomalyError::InconsistentElements { field: "ecc" })
    );
}
