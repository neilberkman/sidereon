//! Relative-frame public API tests.
//!
//! Clohessy-Wiltshire provenance for the analytic unit tests in
//! `astro::relative`: W. H. Clohessy and R. S. Wiltshire, "Terminal Guidance
//! System for Satellite Rendezvous," Journal of the Aerospace Sciences, 27(9),
//! pp. 653-658, 1960, DOI 10.2514/8.8704. D. A. Vallado, "Fundamentals of
//! Astrodynamics and Applications," 4th ed., 2013, Section 6.7
//! "Clohessy-Wiltshire".

use sidereon_core::astro::constants::{earth::WGS84_A_KM, MU_EARTH};
use sidereon_core::astro::covariance::RtnFrameError;
use sidereon_core::astro::math::mat3::{self, Mat3};
use sidereon_core::astro::relative::{
    absolute_from_relative, lvlh_to_inertial_rotation, mean_motion_circular,
    mean_motion_from_state, relative_state, ric_to_inertial_rotation, rsw_to_inertial_rotation,
    rtn_to_inertial_rotation,
};
use sidereon_core::astro::state::CartesianState;

#[test]
fn rotations_are_orthonormal_right_handed_and_aliased() {
    for chief in chief_states() {
        let r = rsw_to_inertial_rotation(&chief).expect("valid chief frame");
        let rt = mat3::inline_tr(&r);
        let should_be_identity = mat3::inline_rxr(&rt, &r);

        for (i, row) in should_be_identity.iter().enumerate() {
            for (j, actual) in row.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (*actual - expected).abs() < 1.0e-12,
                    "entry ({i},{j}): actual {}, expected {}",
                    actual,
                    expected
                );
            }
        }

        assert!((det3(&r) - 1.0).abs() < 1.0e-12);
        assert_eq!(rtn_to_inertial_rotation(&chief).unwrap(), r);
        assert_eq!(ric_to_inertial_rotation(&chief).unwrap(), r);
        assert_eq!(lvlh_to_inertial_rotation(&chief).unwrap(), r);
    }
}

#[test]
fn absolute_relative_round_trip_reproduces_deputy() {
    for (idx, chief) in chief_states().iter().enumerate() {
        let scale = idx as f64 + 1.0;
        let rel = CartesianState::new(
            chief.epoch_tdb_seconds,
            [0.05 * scale, -0.03 * scale, 0.01 * scale],
            [1.0e-5 * scale, -2.0e-5 * scale, 0.5e-5 * scale],
        );
        let deputy = absolute_from_relative(chief, &rel).expect("valid deputy");
        let recovered_rel = relative_state(chief, &deputy).expect("valid relative state");
        let rebuilt = absolute_from_relative(chief, &recovered_rel).expect("valid deputy");

        assert_vec_close(recovered_rel.position_array(), rel.position_array(), 1.0e-9);
        assert_vec_close(
            recovered_rel.velocity_array(),
            rel.velocity_array(),
            1.0e-12,
        );
        assert_vec_close(rebuilt.position_array(), deputy.position_array(), 1.0e-9);
        assert_vec_close(rebuilt.velocity_array(), deputy.velocity_array(), 1.0e-12);
    }
}

#[test]
fn relative_state_of_chief_with_itself_is_zero() {
    for chief in chief_states() {
        let rel = relative_state(&chief, &chief).expect("valid relative state");

        assert_vec_close(rel.position_array(), [0.0, 0.0, 0.0], 1.0e-12);
        assert_vec_close(rel.velocity_array(), [0.0, 0.0, 0.0], 1.0e-12);
    }
}

#[test]
fn degenerate_geometry_surfaces_existing_frame_errors() {
    let zero_position = CartesianState::new(0.0, [0.0, 0.0, 0.0], [0.0, 7.5, 0.0]);
    assert_eq!(
        rsw_to_inertial_rotation(&zero_position).unwrap_err(),
        RtnFrameError::ZeroPosition
    );

    let parallel = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    assert_eq!(
        rsw_to_inertial_rotation(&parallel).unwrap_err(),
        RtnFrameError::ParallelPositionVelocity
    );

    let chief = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]);
    let non_finite_deputy = CartesianState::new(0.0, [7001.0, f64::NAN, 0.0], [0.0, 7.5, 0.0]);
    assert!(matches!(
        relative_state(&chief, &non_finite_deputy),
        Err(RtnFrameError::InvalidInput {
            field: "deputy.position",
            ..
        })
    ));
}

#[test]
fn mean_motion_helpers_agree_for_known_circular_altitude() {
    let radius_km = WGS84_A_KM + 500.0;
    let speed_km_s = (MU_EARTH / radius_km).sqrt();
    let chief = CartesianState::new(0.0, [radius_km, 0.0, 0.0], [0.0, speed_km_s, 0.0]);

    let circular = mean_motion_circular(radius_km).expect("valid circular radius");
    let from_state = mean_motion_from_state(&chief).expect("valid circular state");
    let expected_period = 2.0 * std::f64::consts::PI * (radius_km.powi(3) / MU_EARTH).sqrt();

    assert!((circular - from_state).abs() < 1.0e-12);
    assert!((circular - 2.0 * std::f64::consts::PI / expected_period).abs() < 1.0e-12);
}

fn chief_states() -> Vec<CartesianState> {
    let radii = [WGS84_A_KM + 500.0, 26_560.0, 42_164.0];
    let inclinations = [
        1.0e-9,
        0.05,
        std::f64::consts::FRAC_PI_2 - 1.0e-9,
        std::f64::consts::FRAC_PI_2,
        1.7,
    ];
    let mut rng = Lcg::new(0x9e37_79b9_7f4a_7c15);
    let mut states = Vec::new();

    for radius in radii {
        for inclination in inclinations {
            let raan = rng.next_unit() * std::f64::consts::TAU;
            let argument_latitude = rng.next_unit() * std::f64::consts::TAU;
            states.push(circular_state(radius, inclination, raan, argument_latitude));
        }
    }

    states
}

fn circular_state(
    radius_km: f64,
    inclination_rad: f64,
    raan_rad: f64,
    argument_latitude_rad: f64,
) -> CartesianState {
    let speed = (MU_EARTH / radius_km).sqrt();
    let (sin_u, cos_u) = libm::sincos(argument_latitude_rad);
    let (sin_i, cos_i) = libm::sincos(inclination_rad);
    let (sin_raan, cos_raan) = libm::sincos(raan_rad);

    let x_orb = radius_km * cos_u;
    let y_orb = radius_km * sin_u;
    let vx_orb = -speed * sin_u;
    let vy_orb = speed * cos_u;

    CartesianState::new(
        0.0,
        [
            cos_raan * x_orb - sin_raan * cos_i * y_orb,
            sin_raan * x_orb + cos_raan * cos_i * y_orb,
            sin_i * y_orb,
        ],
        [
            cos_raan * vx_orb - sin_raan * cos_i * vy_orb,
            sin_raan * vx_orb + cos_raan * cos_i * vy_orb,
            sin_i * vy_orb,
        ],
    )
}

fn det3(m: &Mat3) -> f64 {
    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][0], m[1][1], m[1][2]);
    let (g, h, i) = (m[2][0], m[2][1], m[2][2]);
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

fn assert_vec_close(actual: [f64; 3], expected: [f64; 3], tolerance: f64) {
    for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (*actual - *expected).abs() <= tolerance,
            "component {idx}: actual {}, expected {}",
            actual,
            expected
        );
    }
}

struct Lcg {
    state: u64,
}

impl Lcg {
    const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_unit(&mut self) -> f64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        ((self.state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
    }
}
