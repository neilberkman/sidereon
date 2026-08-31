//! ECEF indirect error-state system model and covariance prediction.

use crate::astro::constants::earth::{GM_EARTH_M3_S2, OMEGA_E_DOT_RAD_S};
use crate::astro::math::mat3::{inline_tr, mul_vec3, Mat3};
use crate::astro::math::vec3::norm3;
use crate::inertial::config::RANDOM_WALK_BIAS_TAU_S;
use crate::inertial::state::{mat3_identity, skew, validate_dcm_orthonormal};
use crate::inertial::{validate_vec3, ImuSpec, NavState};

use super::state::{
    identity, invalid_input, matmul, matrix_add, reproject_covariance_psd, symmetrize_in_place,
    validate_covariance_matrix, validate_nonnegative, validate_positive, validate_square_matrix,
    ErrorStateLayout, FusionError, ERROR_ACCEL_BIAS_INDEX, ERROR_ACCEL_SCALE_INDEX,
    ERROR_ATTITUDE_INDEX, ERROR_GYRO_BIAS_INDEX, ERROR_GYRO_SCALE_INDEX, ERROR_POSITION_INDEX,
    ERROR_VELOCITY_INDEX,
};

/// Body-frame IMU kinematics used to linearize the error-state model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorStateImuKinematics {
    /// Specific force resolved in body axes, in m/s^2.
    pub specific_force_body_mps2: [f64; 3],
    /// Angular rate resolved in body axes, in rad/s.
    pub angular_rate_body_rps: [f64; 3],
}

impl ErrorStateImuKinematics {
    /// Build kinematics from body-frame specific force and angular rate.
    pub fn new(
        specific_force_body_mps2: [f64; 3],
        angular_rate_body_rps: [f64; 3],
    ) -> Result<Self, FusionError> {
        validate_vec3(specific_force_body_mps2, "specific_force_body_mps2")
            .map_err(FusionError::from)?;
        validate_vec3(angular_rate_body_rps, "angular_rate_body_rps").map_err(FusionError::from)?;
        Ok(Self {
            specific_force_body_mps2,
            angular_rate_body_rps,
        })
    }
}

/// Linearized continuous and discrete error-state model for one predict step.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorStateLinearization {
    /// Continuous-time ECEF error-state system matrix.
    pub f: Vec<Vec<f64>>,
    /// Second-order state transition matrix.
    pub phi: Vec<Vec<f64>>,
    /// Discrete process-noise covariance.
    pub q_d: Vec<Vec<f64>>,
    /// Time step used for `phi` and `q_d`, in seconds.
    pub dt_s: f64,
    /// Specific force transformed into ECEF axes, in m/s^2.
    pub specific_force_ecef_mps2: [f64; 3],
}

/// Build the ECEF error-state system matrix.
pub fn error_state_system_matrix_ecef(
    state: &NavState,
    kinematics: ErrorStateImuKinematics,
    imu_spec: &ImuSpec,
    layout: ErrorStateLayout,
) -> Result<Vec<Vec<f64>>, FusionError> {
    error_state_system_matrix_ecef_with_imu_to_body(
        state,
        kinematics,
        imu_spec,
        layout,
        mat3_identity(),
    )
}

/// Build the ECEF error-state system matrix with IMU axes rotated into body axes.
pub fn error_state_system_matrix_ecef_with_imu_to_body(
    state: &NavState,
    kinematics: ErrorStateImuKinematics,
    imu_spec: &ImuSpec,
    layout: ErrorStateLayout,
    imu_to_body_dcm: Mat3,
) -> Result<Vec<Vec<f64>>, FusionError> {
    state.validate()?;
    imu_spec.validate()?;
    validate_dcm_orthonormal(&imu_to_body_dcm, "imu_to_body_dcm").map_err(FusionError::from)?;
    let dimension = layout.dimension();
    let mut f = vec![vec![0.0; dimension]; dimension];
    let c_b_e = state.attitude_body_to_ecef;
    let c_imu_e = crate::astro::math::mat3::inline_rxr(&c_b_e, &imu_to_body_dcm);
    let specific_force_ecef = mul_vec3(&c_b_e, kinematics.specific_force_body_mps2);
    let body_to_imu_dcm = inline_tr(&imu_to_body_dcm);
    let specific_force_imu = mul_vec3(&body_to_imu_dcm, kinematics.specific_force_body_mps2);
    let angular_rate_imu = mul_vec3(&body_to_imu_dcm, kinematics.angular_rate_body_rps);

    for axis in 0..3 {
        f[ERROR_POSITION_INDEX + axis][ERROR_VELOCITY_INDEX + axis] = 1.0;
    }

    let gravity_gradient = gravity_gradient_prompt_ecef(state.position_ecef_m)?;
    add_mat3_block(
        &mut f,
        ERROR_VELOCITY_INDEX,
        ERROR_POSITION_INDEX,
        &gravity_gradient,
    );

    let omega = skew([0.0, 0.0, OMEGA_E_DOT_RAD_S]);
    for row in 0..3 {
        for col in 0..3 {
            f[ERROR_VELOCITY_INDEX + row][ERROR_VELOCITY_INDEX + col] = -2.0 * omega[row][col];
            f[ERROR_ATTITUDE_INDEX + row][ERROR_ATTITUDE_INDEX + col] = -omega[row][col];
        }
    }

    let specific_force_skew = skew(specific_force_ecef);
    for row in 0..3 {
        for col in 0..3 {
            f[ERROR_VELOCITY_INDEX + row][ERROR_ATTITUDE_INDEX + col] =
                -specific_force_skew[row][col];
            f[ERROR_VELOCITY_INDEX + row][ERROR_ACCEL_BIAS_INDEX + col] = c_imu_e[row][col];
            f[ERROR_ATTITUDE_INDEX + row][ERROR_GYRO_BIAS_INDEX + col] = -c_imu_e[row][col];
        }
    }

    fill_bias_decay(&mut f, ERROR_ACCEL_BIAS_INDEX, imu_spec.accel_bias_tau_s);
    fill_bias_decay(&mut f, ERROR_GYRO_BIAS_INDEX, imu_spec.gyro_bias_tau_s);

    if layout.includes_scale_factors() {
        for row in 0..3 {
            for col in 0..3 {
                f[ERROR_VELOCITY_INDEX + row][ERROR_ACCEL_SCALE_INDEX + col] =
                    c_imu_e[row][col] * specific_force_imu[col];
                f[ERROR_ATTITUDE_INDEX + row][ERROR_GYRO_SCALE_INDEX + col] =
                    -c_imu_e[row][col] * angular_rate_imu[col];
            }
        }
    }

    Ok(f)
}

/// Discretize `F` as `I + F dt + 0.5 (F dt)^2`.
pub fn error_state_transition_matrix(
    f: &[Vec<f64>],
    dt_s: f64,
) -> Result<Vec<Vec<f64>>, FusionError> {
    validate_nonnegative(dt_s, "dt_s")?;
    let dimension = f.len();
    if dimension != ErrorStateLayout::Fifteen.dimension()
        && dimension != ErrorStateLayout::TwentyOne.dimension()
    {
        return Err(invalid_input("f", "dimension must be 15 or 21"));
    }
    validate_square_matrix(f, dimension, "f")?;

    let mut fdt = vec![vec![0.0; dimension]; dimension];
    for row in 0..dimension {
        for col in 0..dimension {
            fdt[row][col] = f[row][col] * dt_s;
        }
    }
    let fdt2 = matmul(&fdt, &fdt)?;
    let mut phi = identity(dimension);
    for row in 0..dimension {
        for col in 0..dimension {
            phi[row][col] += fdt[row][col] + 0.5 * fdt2[row][col];
        }
    }
    Ok(phi)
}

/// Build the discrete process-noise covariance from an [`ImuSpec`].
pub fn error_state_process_noise_discrete(
    imu_spec: &ImuSpec,
    dt_s: f64,
    layout: ErrorStateLayout,
) -> Result<Vec<Vec<f64>>, FusionError> {
    imu_spec.validate()?;
    validate_nonnegative(dt_s, "dt_s")?;
    let dimension = layout.dimension();
    let mut q_d = vec![vec![0.0; dimension]; dimension];
    let q_accel = imu_spec.accel_vrw_mps_sqrt_s * imu_spec.accel_vrw_mps_sqrt_s;
    let q_gyro = imu_spec.gyro_arw_rad_sqrt_s * imu_spec.gyro_arw_rad_sqrt_s;
    let dt = dt_s.abs();
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;

    for axis in 0..3 {
        q_d[ERROR_POSITION_INDEX + axis][ERROR_POSITION_INDEX + axis] = q_accel * dt3 / 3.0;
        q_d[ERROR_POSITION_INDEX + axis][ERROR_VELOCITY_INDEX + axis] = q_accel * dt2 / 2.0;
        q_d[ERROR_VELOCITY_INDEX + axis][ERROR_POSITION_INDEX + axis] = q_accel * dt2 / 2.0;
        q_d[ERROR_VELOCITY_INDEX + axis][ERROR_VELOCITY_INDEX + axis] = q_accel * dt;
        q_d[ERROR_ATTITUDE_INDEX + axis][ERROR_ATTITUDE_INDEX + axis] = q_gyro * dt;
        q_d[ERROR_ACCEL_BIAS_INDEX + axis][ERROR_ACCEL_BIAS_INDEX + axis] =
            imu_spec.accel_bias_variance_increment(dt_s)?;
        q_d[ERROR_GYRO_BIAS_INDEX + axis][ERROR_GYRO_BIAS_INDEX + axis] =
            imu_spec.gyro_bias_variance_increment(dt_s)?;
    }

    if layout.includes_scale_factors() {
        let accel_scale = imu_spec.accel_scale_instab_ppm.unwrap_or(0.0) * 1.0e-6;
        let gyro_scale = imu_spec.gyro_scale_instab_ppm.unwrap_or(0.0) * 1.0e-6;
        let accel_scale_q = accel_scale * accel_scale;
        let gyro_scale_q = gyro_scale * gyro_scale;
        for axis in 0..3 {
            q_d[ERROR_ACCEL_SCALE_INDEX + axis][ERROR_ACCEL_SCALE_INDEX + axis] =
                accel_scale_q * dt;
            q_d[ERROR_GYRO_SCALE_INDEX + axis][ERROR_GYRO_SCALE_INDEX + axis] = gyro_scale_q * dt;
        }
    }

    reproject_covariance_psd(&mut q_d, "process_noise")?;
    Ok(q_d)
}

/// Build `F`, `Phi`, and `Q_d` for one error-state predict step.
pub fn linearize_error_state_ecef(
    state: &NavState,
    kinematics: ErrorStateImuKinematics,
    imu_spec: &ImuSpec,
    dt_s: f64,
    layout: ErrorStateLayout,
) -> Result<ErrorStateLinearization, FusionError> {
    linearize_error_state_ecef_with_imu_to_body(
        state,
        kinematics,
        imu_spec,
        dt_s,
        layout,
        mat3_identity(),
    )
}

/// Build `F`, `Phi`, and `Q_d` for one predict step with IMU axes rotated into body axes.
pub fn linearize_error_state_ecef_with_imu_to_body(
    state: &NavState,
    kinematics: ErrorStateImuKinematics,
    imu_spec: &ImuSpec,
    dt_s: f64,
    layout: ErrorStateLayout,
    imu_to_body_dcm: Mat3,
) -> Result<ErrorStateLinearization, FusionError> {
    let f = error_state_system_matrix_ecef_with_imu_to_body(
        state,
        kinematics,
        imu_spec,
        layout,
        imu_to_body_dcm,
    )?;
    let phi = error_state_transition_matrix(&f, dt_s)?;
    let q_d = error_state_process_noise_discrete(imu_spec, dt_s, layout)?;
    Ok(ErrorStateLinearization {
        f,
        phi,
        q_d,
        dt_s,
        specific_force_ecef_mps2: mul_vec3(
            &state.attitude_body_to_ecef,
            kinematics.specific_force_body_mps2,
        ),
    })
}

/// Predict covariance as `P = Phi P Phi^T + Q_d`.
pub fn predict_error_state_covariance(
    covariance: &mut Vec<Vec<f64>>,
    phi: &[Vec<f64>],
    q_d: &[Vec<f64>],
) -> Result<(), FusionError> {
    let dimension = covariance.len();
    if dimension != ErrorStateLayout::Fifteen.dimension()
        && dimension != ErrorStateLayout::TwentyOne.dimension()
    {
        return Err(invalid_input("covariance", "dimension must be 15 or 21"));
    }
    validate_covariance_matrix(covariance, dimension, "covariance")?;
    validate_square_matrix(phi, dimension, "phi")?;
    validate_covariance_matrix(q_d, dimension, "q_d")?;

    let mut temp = vec![vec![0.0; dimension]; dimension];
    for i in 0..dimension {
        for j in 0..dimension {
            for k in 0..dimension {
                temp[i][j] += phi[i][k] * covariance[k][j];
            }
        }
    }

    let mut propagated = vec![vec![0.0; dimension]; dimension];
    for i in 0..dimension {
        for j in 0..dimension {
            for k in 0..dimension {
                propagated[i][j] += temp[i][k] * phi[j][k];
            }
        }
    }
    let propagated = matrix_add(&propagated, q_d)?;
    *covariance = propagated;
    symmetrize_in_place(covariance);
    reproject_covariance_psd(covariance, "covariance")
}

pub(crate) fn gravity_gradient_prompt_ecef(position_ecef_m: [f64; 3]) -> Result<Mat3, FusionError> {
    validate_vec3(position_ecef_m, "position_ecef_m").map_err(FusionError::from)?;
    let radius_m = norm3(position_ecef_m);
    validate_positive(radius_m, "position_radius_m")?;
    let radius3 = radius_m * radius_m * radius_m;
    let scale = -GM_EARTH_M3_S2 / radius3;
    let r_hat = [
        position_ecef_m[0] / radius_m,
        position_ecef_m[1] / radius_m,
        position_ecef_m[2] / radius_m,
    ];
    let omega = skew([0.0, 0.0, OMEGA_E_DOT_RAD_S]);
    let omega2 = [
        [
            omega[0][0] * omega[0][0] + omega[0][1] * omega[1][0] + omega[0][2] * omega[2][0],
            omega[0][0] * omega[0][1] + omega[0][1] * omega[1][1] + omega[0][2] * omega[2][1],
            omega[0][0] * omega[0][2] + omega[0][1] * omega[1][2] + omega[0][2] * omega[2][2],
        ],
        [
            omega[1][0] * omega[0][0] + omega[1][1] * omega[1][0] + omega[1][2] * omega[2][0],
            omega[1][0] * omega[0][1] + omega[1][1] * omega[1][1] + omega[1][2] * omega[2][1],
            omega[1][0] * omega[0][2] + omega[1][1] * omega[1][2] + omega[1][2] * omega[2][2],
        ],
        [
            omega[2][0] * omega[0][0] + omega[2][1] * omega[1][0] + omega[2][2] * omega[2][0],
            omega[2][0] * omega[0][1] + omega[2][1] * omega[1][1] + omega[2][2] * omega[2][1],
            omega[2][0] * omega[0][2] + omega[2][1] * omega[1][2] + omega[2][2] * omega[2][2],
        ],
    ];
    let mut gradient = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            let identity = if row == col { 1.0 } else { 0.0 };
            gradient[row][col] =
                scale * (identity - 3.0 * r_hat[row] * r_hat[col]) - omega2[row][col];
        }
    }
    Ok(gradient)
}

fn fill_bias_decay(f: &mut [Vec<f64>], index: usize, tau_s: f64) {
    if tau_s != RANDOM_WALK_BIAS_TAU_S && tau_s.is_finite() {
        for axis in 0..3 {
            f[index + axis][index + axis] = -1.0 / tau_s;
        }
    }
}

fn add_mat3_block(matrix: &mut [Vec<f64>], row0: usize, col0: usize, block: &Mat3) {
    for row in 0..3 {
        for col in 0..3 {
            matrix[row0 + row][col0 + col] += block[row][col];
        }
    }
}

#[cfg(test)]
mod tests {
    //! Provenance: these tests use the ECEF indirect error-state equations from
    //! Groves, Principles of GNSS, Inertial, and Multisensor Integrated
    //! Navigation Systems, 2nd ed., Section 14.2.4. The ECEF gravity-gradient
    //! sign and centrifugal derivative are cross-checked against the public
    //! Sandia ESKF derivation, OSTI report 2516824, equations 7.58 and 7.65.
    //! The process-noise position/velocity block follows the white-acceleration integral
    //! `q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]`, with the same operation order
    //! used by the covariance propagation process-noise increment.

    use super::*;
    use crate::astro::constants::earth::WGS84_A_M;
    use crate::astro::math::mat3::{inline_rxr, inline_tr, mul_vec3};
    use crate::inertial::mechanization::mechanize_ecef;
    use crate::inertial::state::{mat3_identity, reorthonormalize_dcm};
    use crate::inertial::{CorrectedImuIncrement, MechanizationConfig};
    use nalgebra::DMatrix;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.17e}"
        );
    }

    fn reference_state() -> NavState {
        NavState::new(
            0.0,
            [WGS84_A_M + 1000.0, 25.0, -40.0],
            [3.0, -2.0, 1.0],
            mat3_identity(),
        )
        .expect("reference state")
    }

    fn reference_imu() -> ErrorStateImuKinematics {
        ErrorStateImuKinematics::new([0.12, -0.05, 9.72], [0.004, -0.002, 0.001])
            .expect("imu kinematics")
    }

    fn reference_spec() -> ImuSpec {
        ImuSpec::datasheet(
            0.02,
            0.003,
            0.004,
            0.0002,
            3600.0,
            7200.0,
            Some(25.0),
            Some(30.0),
        )
    }

    fn reference_increment(imu: ErrorStateImuKinematics, dt_s: f64) -> CorrectedImuIncrement {
        CorrectedImuIncrement {
            t_j2000_s: dt_s,
            delta_velocity_mps: [
                imu.specific_force_body_mps2[0] * dt_s,
                imu.specific_force_body_mps2[1] * dt_s,
                imu.specific_force_body_mps2[2] * dt_s,
            ],
            delta_theta_rad: [
                imu.angular_rate_body_rps[0] * dt_s,
                imu.angular_rate_body_rps[1] * dt_s,
                imu.angular_rate_body_rps[2] * dt_s,
            ],
            dt_s,
        }
    }

    fn attitude_error_ecef(perturbed: &Mat3, base: &Mat3) -> [f64; 3] {
        let delta = inline_rxr(perturbed, &inline_tr(base));
        [
            0.5 * (delta[1][2] - delta[2][1]),
            0.5 * (delta[2][0] - delta[0][2]),
            0.5 * (delta[0][1] - delta[1][0]),
        ]
    }

    #[test]
    fn gravity_gradient_matches_published_ecef_point_mass_plus_centrifugal_bits() {
        let radius_m = WGS84_A_M + 1000.0;
        let gradient = gravity_gradient_prompt_ecef([radius_m, 0.0, 0.0]).expect("gradient");
        let radius3 = radius_m * radius_m * radius_m;
        let point = GM_EARTH_M3_S2 / radius3;
        let omega2 = OMEGA_E_DOT_RAD_S * OMEGA_E_DOT_RAD_S;

        assert_eq!(gradient[0][0].to_bits(), (2.0 * point + omega2).to_bits());
        assert_eq!(gradient[1][1].to_bits(), (-point + omega2).to_bits());
        assert_eq!(gradient[2][2].to_bits(), (-point).to_bits());
        for (row, values) in gradient.iter().enumerate() {
            for (col, value) in values.iter().enumerate() {
                if row != col {
                    assert_eq!(*value, 0.0);
                }
            }
        }
    }

    #[test]
    fn phi_matches_mechanization_finite_difference_for_kinematic_block() {
        let state = reference_state();
        let imu = reference_imu();
        let spec = reference_spec();
        let dt_s = 1.0e-5;
        let epsilon = 1.0;
        let linear =
            linearize_error_state_ecef(&state, imu, &spec, dt_s, ErrorStateLayout::Fifteen)
                .expect("linearization");
        let increment = reference_increment(imu, dt_s);
        let base_next =
            mechanize_ecef(&state, &increment, MechanizationConfig::default()).expect("base step");

        for col in 0..6 {
            let mut perturbed = state;
            if col < 3 {
                perturbed.position_ecef_m[col] += epsilon;
            } else {
                perturbed.velocity_ecef_mps[col - 3] += epsilon;
            }
            let next = mechanize_ecef(&perturbed, &increment, MechanizationConfig::default())
                .expect("perturbed step");
            for row in 0..3 {
                let fd = (next.position_ecef_m[row] - base_next.position_ecef_m[row]) / epsilon;
                assert_close(fd, linear.phi[row][col], 2.0e-8);
            }
            for row in 0..3 {
                let fd = (next.velocity_ecef_mps[row] - base_next.velocity_ecef_mps[row]) / epsilon;
                assert_close(fd, linear.phi[ERROR_VELOCITY_INDEX + row][col], 5.0e-10);
            }
        }
    }

    #[test]
    fn phi_matches_mechanization_finite_difference_for_attitude_error_block() {
        let state = reference_state();
        let imu = reference_imu();
        let spec = reference_spec();
        let dt_s = 1.0e-4;
        let epsilon = 1.0e-6;
        let linear =
            linearize_error_state_ecef(&state, imu, &spec, dt_s, ErrorStateLayout::Fifteen)
                .expect("linearization");
        let increment = reference_increment(imu, dt_s);
        let base_next =
            mechanize_ecef(&state, &increment, MechanizationConfig::default()).expect("base step");

        for col in 0..3 {
            let mut psi = [0.0; 3];
            psi[col] = epsilon;
            let psi_skew = skew(psi);
            let mut correction = mat3_identity();
            for row in 0..3 {
                for col in 0..3 {
                    correction[row][col] += psi_skew[row][col];
                }
            }
            let mut perturbed = state;
            perturbed.attitude_body_to_ecef =
                reorthonormalize_dcm(&inline_rxr(&correction, &state.attitude_body_to_ecef))
                    .expect("perturbed attitude");
            let next = mechanize_ecef(&perturbed, &increment, MechanizationConfig::default())
                .expect("perturbed step");
            let raw_attitude_error = attitude_error_ecef(
                &next.attitude_body_to_ecef,
                &base_next.attitude_body_to_ecef,
            );
            let psi_next = [
                -raw_attitude_error[0],
                -raw_attitude_error[1],
                -raw_attitude_error[2],
            ];

            for (row, value) in psi_next.iter().enumerate() {
                let fd = *value / epsilon;
                assert_close(
                    fd,
                    linear.phi[ERROR_ATTITUDE_INDEX + row][ERROR_ATTITUDE_INDEX + col],
                    2.0e-9,
                );
            }
            for row in 0..3 {
                let fd = (next.velocity_ecef_mps[row] - base_next.velocity_ecef_mps[row]) / epsilon;
                assert_close(
                    fd,
                    linear.phi[ERROR_VELOCITY_INDEX + row][ERROR_ATTITUDE_INDEX + col],
                    5.0e-7,
                );
            }
        }
    }

    #[test]
    fn qd_position_velocity_block_matches_closed_form_bits() {
        let dt_s = 0.125_f64;
        let q_accel = 0.02_f64 * 0.02_f64;
        let spec = ImuSpec::datasheet(
            0.02,
            0.0,
            0.0,
            0.0,
            RANDOM_WALK_BIAS_TAU_S,
            RANDOM_WALK_BIAS_TAU_S,
            None,
            None,
        );
        let q_d = error_state_process_noise_discrete(&spec, dt_s, ErrorStateLayout::Fifteen)
            .expect("process noise");
        let dt = dt_s.abs();
        let dt2 = dt * dt;
        let dt3 = dt2 * dt;
        for axis in 0..3 {
            assert_eq!(q_d[axis][axis].to_bits(), (q_accel * dt3 / 3.0).to_bits());
            assert_eq!(
                q_d[axis][ERROR_VELOCITY_INDEX + axis].to_bits(),
                (q_accel * dt2 / 2.0).to_bits()
            );
            assert_eq!(
                q_d[ERROR_VELOCITY_INDEX + axis][axis].to_bits(),
                (q_accel * dt2 / 2.0).to_bits()
            );
            assert_eq!(
                q_d[ERROR_VELOCITY_INDEX + axis][ERROR_VELOCITY_INDEX + axis].to_bits(),
                (q_accel * dt).to_bits()
            );
        }
    }

    #[test]
    fn imu_to_body_dcm_rotates_bias_and_scale_jacobians() {
        let state = reference_state();
        let imu = ErrorStateImuKinematics::new([2.0, 3.0, 4.0], [0.1, 0.2, 0.3]).expect("imu");
        let imu_to_body = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let body_to_imu = inline_tr(&imu_to_body);
        let specific_force_imu = mul_vec3(&body_to_imu, imu.specific_force_body_mps2);
        let angular_rate_imu = mul_vec3(&body_to_imu, imu.angular_rate_body_rps);
        let f = error_state_system_matrix_ecef_with_imu_to_body(
            &state,
            imu,
            &reference_spec(),
            ErrorStateLayout::TwentyOne,
            imu_to_body,
        )
        .expect("system matrix");

        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(
                    f[ERROR_VELOCITY_INDEX + row][ERROR_ACCEL_BIAS_INDEX + col].to_bits(),
                    imu_to_body[row][col].to_bits()
                );
                assert_eq!(
                    f[ERROR_ATTITUDE_INDEX + row][ERROR_GYRO_BIAS_INDEX + col].to_bits(),
                    (-imu_to_body[row][col]).to_bits()
                );
                assert_eq!(
                    f[ERROR_VELOCITY_INDEX + row][ERROR_ACCEL_SCALE_INDEX + col].to_bits(),
                    (imu_to_body[row][col] * specific_force_imu[col]).to_bits()
                );
                assert_eq!(
                    f[ERROR_ATTITUDE_INDEX + row][ERROR_GYRO_SCALE_INDEX + col].to_bits(),
                    (-imu_to_body[row][col] * angular_rate_imu[col]).to_bits()
                );
            }
        }
    }

    #[test]
    fn imu_to_body_dcm_matches_identity_mount_in_imu_frame_with_nonidentity_attitude() {
        let body_to_ecef = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let imu_to_body = [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let imu_to_ecef = inline_rxr(&body_to_ecef, &imu_to_body);
        let body_state = NavState::new(
            25.0,
            [WGS84_A_M + 750.0, -125.0, 80.0],
            [4.0, -3.0, 2.0],
            body_to_ecef,
        )
        .expect("body-frame state");
        let imu_frame_state = NavState::new(
            body_state.t_j2000_s,
            body_state.position_ecef_m,
            body_state.velocity_ecef_mps,
            imu_to_ecef,
        )
        .expect("imu-frame state");
        let body_kinematics = ErrorStateImuKinematics::new([1.5, -0.25, 9.6], [0.03, -0.02, 0.01])
            .expect("body kinematics");
        let body_to_imu = inline_tr(&imu_to_body);
        let imu_frame_kinematics = ErrorStateImuKinematics::new(
            mul_vec3(&body_to_imu, body_kinematics.specific_force_body_mps2),
            mul_vec3(&body_to_imu, body_kinematics.angular_rate_body_rps),
        )
        .expect("imu-frame kinematics");

        let mounted = error_state_system_matrix_ecef_with_imu_to_body(
            &body_state,
            body_kinematics,
            &reference_spec(),
            ErrorStateLayout::TwentyOne,
            imu_to_body,
        )
        .expect("mounted system matrix");
        let identity_in_imu_frame = error_state_system_matrix_ecef(
            &imu_frame_state,
            imu_frame_kinematics,
            &reference_spec(),
            ErrorStateLayout::TwentyOne,
        )
        .expect("identity system matrix");

        for row in 0..mounted.len() {
            for col in 0..mounted[row].len() {
                assert_close(mounted[row][col], identity_in_imu_frame[row][col], 1.0e-15);
            }
        }
    }

    #[test]
    fn propagate_only_logdet_grows_monotonically_with_process_noise() {
        let state = reference_state();
        let imu = reference_imu();
        let spec = ImuSpec::datasheet(0.2, 0.03, 0.02, 0.002, 600.0, 900.0, None, None);
        let mut covariance = vec![vec![0.0; 15]; 15];
        for (idx, row) in covariance.iter_mut().enumerate() {
            row[idx] = 1.0e-3;
        }
        let mut previous = logdet(&covariance);
        for _ in 0..12 {
            let linear =
                linearize_error_state_ecef(&state, imu, &spec, 0.02, ErrorStateLayout::Fifteen)
                    .expect("linearization");
            predict_error_state_covariance(&mut covariance, &linear.phi, &linear.q_d)
                .expect("predict covariance");
            let current = logdet(&covariance);
            assert!(
                current > previous,
                "current {current:.17e}, previous {previous:.17e}"
            );
            previous = current;
        }
    }

    fn logdet(covariance: &[Vec<f64>]) -> f64 {
        let flat = covariance
            .iter()
            .flat_map(|row| row.iter().copied())
            .collect::<Vec<_>>();
        let matrix = DMatrix::from_row_slice(covariance.len(), covariance.len(), &flat);
        let cholesky = matrix.cholesky().expect("test covariance is SPD");
        2.0 * (0..covariance.len())
            .map(|idx| libm::log(cholesky.l()[(idx, idx)]))
            .sum::<f64>()
    }
}
