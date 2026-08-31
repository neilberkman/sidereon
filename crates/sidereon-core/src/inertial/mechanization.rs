//! ECEF strapdown inertial mechanization.

use crate::astro::constants::earth::OMEGA_E_DOT_RAD_S;
use crate::astro::math::mat3::Mat3;
use crate::astro::math::vec3::{add3, dot3, scale3};

use super::config::MechanizationConfig;
use super::frames::gravity_ecef_mps2;
use super::imu::{CorrectedImuIncrement, ImuErrorModel, ImuSample};
use super::state::{
    mat3_add, mat3_identity, mat3_mul, mat3_mul_vec, mat3_scale, reorthonormalize_dcm, skew,
    NavState,
};
use super::{validate_finite, validate_vec3, InertialError};

/// Streaming ECEF strapdown mechanizer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StrapdownMechanizer {
    state: NavState,
    imu_model: ImuErrorModel,
    config: MechanizationConfig,
}

impl StrapdownMechanizer {
    /// Build a mechanizer from an initial navigation state.
    pub fn new(state: NavState) -> Result<Self, InertialError> {
        state.validate()?;
        Ok(Self {
            state,
            imu_model: ImuErrorModel::default(),
            config: MechanizationConfig::default(),
        })
    }

    /// Return a copy with a different IMU error model.
    pub fn with_imu_model(mut self, imu_model: ImuErrorModel) -> Result<Self, InertialError> {
        imu_model.bias.validate()?;
        imu_model.calibration.validate()?;
        self.imu_model = imu_model;
        Ok(self)
    }

    /// Return a copy with a different mechanization config.
    pub const fn with_config(mut self, config: MechanizationConfig) -> Self {
        self.config = config;
        self
    }

    /// Current navigation state.
    pub const fn state(&self) -> &NavState {
        &self.state
    }

    /// Propagate the state with one raw IMU sample.
    pub fn propagate(&mut self, sample: ImuSample) -> Result<&NavState, InertialError> {
        let increment = self
            .imu_model
            .correct_sample(&sample, self.state.t_j2000_s)?;
        self.state = mechanize_ecef(&self.state, &increment, self.config)?;
        Ok(&self.state)
    }
}

/// Propagate an ECEF navigation state by one corrected IMU increment.
pub fn mechanize_ecef(
    state: &NavState,
    increment: &CorrectedImuIncrement,
    _config: MechanizationConfig,
) -> Result<NavState, InertialError> {
    state.validate()?;
    validate_increment(increment)?;
    if increment.t_j2000_s <= state.t_j2000_s {
        return Err(InertialError::NonMonotonicSample);
    }

    let dt_s = increment.dt_s;
    let body_delta = rodrigues_delta_dcm(increment.delta_theta_rad)?;
    let earth_delta = earth_rotation_first_order(dt_s);
    let attitude_raw = mat3_mul(
        &mat3_mul(&earth_delta, &state.attitude_body_to_ecef),
        &body_delta,
    );
    let attitude_body_to_ecef = reorthonormalize_dcm(&attitude_raw)?;

    let c_avg = mid_interval_dcm(
        &state.attitude_body_to_ecef,
        increment.delta_theta_rad,
        dt_s,
    )?;
    let delta_v_ecef = mat3_mul_vec(&c_avg, increment.delta_velocity_mps);
    let gravity = gravity_ecef_mps2(state.position_ecef_m)?;
    let coriolis = scale3(earth_rate_cross(state.velocity_ecef_mps), -2.0);
    let acceleration = add3(gravity, coriolis);
    let velocity_ecef_mps = add3(
        add3(state.velocity_ecef_mps, delta_v_ecef),
        scale3(acceleration, dt_s),
    );
    let avg_velocity = scale3(add3(state.velocity_ecef_mps, velocity_ecef_mps), 0.5);
    let position_ecef_m = add3(state.position_ecef_m, scale3(avg_velocity, dt_s));

    NavState {
        t_j2000_s: increment.t_j2000_s,
        position_ecef_m,
        velocity_ecef_mps,
        attitude_body_to_ecef,
        accel_bias_mps2: state.accel_bias_mps2,
        gyro_bias_rps: state.gyro_bias_rps,
    }
    .with_biases(state.accel_bias_mps2, state.gyro_bias_rps)
}

/// Exact Rodrigues direction-cosine matrix for a body delta angle.
pub fn rodrigues_delta_dcm(delta_theta_rad: [f64; 3]) -> Result<Mat3, InertialError> {
    validate_vec3(delta_theta_rad, "delta_theta_rad")?;
    let phi2 = dot3(delta_theta_rad, delta_theta_rad);
    let phi = phi2.sqrt();
    let phi4 = phi2 * phi2;
    let (a, b) = if phi < 1.0e-8 {
        (
            1.0 - phi2 / 6.0 + phi4 / 120.0,
            0.5 - phi2 / 24.0 + phi4 / 720.0,
        )
    } else {
        (libm::sin(phi) / phi, (1.0 - libm::cos(phi)) / phi2)
    };
    let k = skew(delta_theta_rad);
    let k2 = mat3_mul(&k, &k);
    Ok(mat3_add(
        &mat3_add(&mat3_identity(), &mat3_scale(&k, a)),
        &mat3_scale(&k2, b),
    ))
}

pub(crate) fn validate_increment(increment: &CorrectedImuIncrement) -> Result<(), InertialError> {
    validate_finite(increment.t_j2000_s, "increment.t_j2000_s")?;
    validate_vec3(increment.delta_velocity_mps, "increment.delta_velocity_mps")?;
    validate_vec3(increment.delta_theta_rad, "increment.delta_theta_rad")?;
    validate_finite(increment.dt_s, "increment.dt_s")?;
    if increment.dt_s > 0.0 {
        Ok(())
    } else {
        Err(super::invalid_input("increment.dt_s", "must be positive"))
    }
}

pub(crate) fn mid_interval_dcm(
    attitude_body_to_ecef: &Mat3,
    delta_theta_rad: [f64; 3],
    dt_s: f64,
) -> Result<Mat3, InertialError> {
    let earth_half = earth_rotation_first_order(0.5 * dt_s);
    let body_half = rodrigues_delta_dcm(scale3(delta_theta_rad, 0.5))?;
    reorthonormalize_dcm(&mat3_mul(
        &mat3_mul(&earth_half, attitude_body_to_ecef),
        &body_half,
    ))
}

pub(crate) fn earth_rotation_first_order(dt_s: f64) -> Mat3 {
    [
        [1.0, OMEGA_E_DOT_RAD_S * dt_s, 0.0],
        [-OMEGA_E_DOT_RAD_S * dt_s, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

pub(crate) fn earth_rate_cross(v: [f64; 3]) -> [f64; 3] {
    [-OMEGA_E_DOT_RAD_S * v[1], OMEGA_E_DOT_RAD_S * v[0], 0.0]
}

#[cfg(test)]
mod tests {
    //! Provenance: mechanization tests follow Groves, Principles of GNSS,
    //! Inertial, and Multisensor Integrated Navigation Systems, 2nd ed.,
    //! Chapter 5.4. The synthetic trajectory tests invert the implemented ECEF
    //! mechanization equations for the exact body-frame increments that produce
    //! the analytic target state.

    use super::*;
    use crate::astro::constants::earth::{WGS84_A_M, WGS84_F};
    use crate::astro::math::mat3::inline_tr;
    use crate::astro::math::vec3::sub3;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.17e}"
        );
    }

    fn assert_vec_close(actual: [f64; 3], expected: [f64; 3], tolerance: f64) {
        for i in 0..3 {
            assert_close(actual[i], expected[i], tolerance);
        }
    }

    fn inverse_delta_velocity(
        state: &NavState,
        target_velocity_ecef_mps: [f64; 3],
        delta_theta_rad: [f64; 3],
        dt_s: f64,
    ) -> [f64; 3] {
        let c_avg = mid_interval_dcm(&state.attitude_body_to_ecef, delta_theta_rad, dt_s)
            .expect("mid-interval dcm");
        let c_avg_t = inline_tr(&c_avg);
        let gravity = gravity_ecef_mps2(state.position_ecef_m).expect("gravity");
        let coriolis = scale3(earth_rate_cross(state.velocity_ecef_mps), -2.0);
        let acceleration = add3(gravity, coriolis);
        let required_ecef = sub3(
            sub3(target_velocity_ecef_mps, state.velocity_ecef_mps),
            scale3(acceleration, dt_s),
        );
        mat3_mul_vec(&c_avg_t, required_ecef)
    }

    #[test]
    fn rodrigues_hits_right_angle_rotation() {
        let dcm = rodrigues_delta_dcm([0.0, 0.0, core::f64::consts::FRAC_PI_2]).expect("rodrigues");
        assert_close(dcm[0][0], 0.0, 1.2e-16);
        assert_close(dcm[0][1], -1.0, 0.0);
        assert_close(dcm[1][0], 1.0, 0.0);
        assert_close(dcm[1][1], 0.0, 1.2e-16);
        assert_close(dcm[2][2], 1.0, 0.0);
    }

    #[test]
    fn attitude_mechanization_matches_closed_form_update() {
        let state =
            NavState::new(0.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity()).expect("state");
        let dt_s = 0.25;
        let delta_theta_rad = [0.0, 0.0, 0.9];
        let delta_velocity_mps = inverse_delta_velocity(&state, [0.0; 3], delta_theta_rad, dt_s);
        let increment = CorrectedImuIncrement {
            t_j2000_s: dt_s,
            delta_velocity_mps,
            delta_theta_rad,
            dt_s,
        };
        let propagated =
            mechanize_ecef(&state, &increment, MechanizationConfig::default()).expect("step");

        let expected = reorthonormalize_dcm(&mat3_mul(
            &earth_rotation_first_order(dt_s),
            &rodrigues_delta_dcm(delta_theta_rad).expect("delta dcm"),
        ))
        .expect("expected");
        for (i, row) in expected.iter().enumerate() {
            for (j, expected_value) in row.iter().enumerate() {
                assert_close(
                    propagated.attitude_body_to_ecef[i][j],
                    *expected_value,
                    3.0e-16,
                );
            }
        }
    }

    #[test]
    fn inverted_static_trajectory_holds_velocity_and_position() {
        let mut state =
            NavState::new(0.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity()).expect("state");
        let dt_s = 0.01;
        for step in 1..=200 {
            let delta_theta_rad = [0.0; 3];
            let delta_velocity_mps =
                inverse_delta_velocity(&state, [0.0; 3], delta_theta_rad, dt_s);
            let increment = CorrectedImuIncrement {
                t_j2000_s: step as f64 * dt_s,
                delta_velocity_mps,
                delta_theta_rad,
                dt_s,
            };
            state =
                mechanize_ecef(&state, &increment, MechanizationConfig::default()).expect("step");
        }
        assert_vec_close(state.velocity_ecef_mps, [0.0; 3], 2.0e-14);
        assert_vec_close(state.position_ecef_m, [WGS84_A_M, 0.0, 0.0], 2.0e-13);
    }

    #[test]
    fn inverted_constant_acceleration_trajectory_hits_position() {
        let b_m = WGS84_A_M * (1.0 - WGS84_F);
        let state = NavState::new(
            100.0,
            [4_510_000.0, 120_000.0, b_m / 2.0],
            [12.0, -4.0, 1.5],
            mat3_identity(),
        )
        .expect("state");
        let dt_s = 0.2;
        let target_velocity = [12.3, -4.1, 1.55];
        let delta_theta_rad = [0.01, -0.02, 0.03];
        let delta_velocity_mps =
            inverse_delta_velocity(&state, target_velocity, delta_theta_rad, dt_s);
        let increment = CorrectedImuIncrement {
            t_j2000_s: 100.2,
            delta_velocity_mps,
            delta_theta_rad,
            dt_s,
        };
        let propagated =
            mechanize_ecef(&state, &increment, MechanizationConfig::default()).expect("step");
        let expected_position = add3(
            state.position_ecef_m,
            scale3(add3(state.velocity_ecef_mps, target_velocity), 0.5 * dt_s),
        );
        assert_vec_close(propagated.velocity_ecef_mps, target_velocity, 2.0e-14);
        assert_vec_close(propagated.position_ecef_m, expected_position, 3.0e-10);
    }
}
