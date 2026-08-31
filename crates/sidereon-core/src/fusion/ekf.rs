//! Generic EKF correction and closed-loop reset for the indirect INS state.

use crate::astro::math::mat3::inline_rxr;
use crate::astro::math::portable;
use crate::inertial::state::{mat3_identity, reorthonormalize_dcm, skew};

use super::state::{
    dmatrix_from_rows, identity, invalid_input, matmul, matrix_add, matrix_sub, matvec,
    reproject_covariance_psd, solve_spd, transpose, validate_covariance_matrix,
    validate_finite_slice, validate_matrix_cols, validate_positive, validate_square_matrix,
    ErrorStateLayout, FusionError, InsFilterState, ERROR_ACCEL_BIAS_INDEX, ERROR_ACCEL_SCALE_INDEX,
    ERROR_ATTITUDE_INDEX, ERROR_GYRO_BIAS_INDEX, ERROR_GYRO_SCALE_INDEX, ERROR_POSITION_INDEX,
    ERROR_VELOCITY_INDEX,
};

/// Generic linearized EKF measurement correction.
#[derive(Debug, Clone, PartialEq)]
pub struct EkfCorrection {
    /// Innovation vector `y = z - h(x)`.
    pub innovation: Vec<f64>,
    /// Measurement design matrix `H`, with one row per innovation.
    pub design: Vec<Vec<f64>>,
    /// Measurement covariance matrix `R`.
    pub measurement_covariance: Vec<Vec<f64>>,
}

impl EkfCorrection {
    /// Build and validate a generic correction.
    pub fn new(
        innovation: Vec<f64>,
        design: Vec<Vec<f64>>,
        measurement_covariance: Vec<Vec<f64>>,
    ) -> Result<Self, FusionError> {
        if innovation.is_empty() {
            return Err(invalid_input("innovation", "must not be empty"));
        }
        if design.len() != innovation.len() {
            return Err(FusionError::DimensionMismatch {
                field: "design",
                expected: innovation.len(),
                actual: design.len(),
            });
        }
        validate_finite_slice(&innovation, "innovation")?;
        validate_measurement_covariance(&measurement_covariance, innovation.len())?;
        Ok(Self {
            innovation,
            design,
            measurement_covariance,
        })
    }

    /// Return the number of measurement rows.
    pub fn row_count(&self) -> usize {
        self.innovation.len()
    }

    /// Validate this correction for a state dimension.
    pub fn validate_for_dimension(&self, dimension: usize) -> Result<(), FusionError> {
        if self.innovation.is_empty() {
            return Err(invalid_input("innovation", "must not be empty"));
        }
        validate_finite_slice(&self.innovation, "innovation")?;
        if self.design.len() != self.innovation.len() {
            return Err(FusionError::DimensionMismatch {
                field: "design",
                expected: self.innovation.len(),
                actual: self.design.len(),
            });
        }
        validate_matrix_cols(&self.design, dimension, "design")?;
        validate_measurement_covariance(&self.measurement_covariance, self.innovation.len())
    }
}

/// Innovation screening options.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InnovationGate {
    /// Rejection threshold in absolute normalized-innovation sigma.
    pub threshold_sigma: f64,
    /// Minimum accepted rows required to apply an update.
    pub min_rows: usize,
}

impl InnovationGate {
    /// Validate gate options.
    pub fn validate(&self) -> Result<(), FusionError> {
        validate_positive(self.threshold_sigma, "threshold_sigma")
    }
}

/// EKF correction options.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EkfUpdateOptions {
    /// Optional normalized-innovation screen applied before correction.
    pub innovation_gate: Option<InnovationGate>,
}

/// Diagnostics from an innovation screen.
#[derive(Debug, Clone, PartialEq)]
pub struct InnovationGateReport {
    /// Rejection threshold in sigma.
    pub threshold_sigma: f64,
    /// Minimum accepted rows requested by the gate.
    pub min_rows: usize,
    /// Number of input measurement rows.
    pub input_rows: usize,
    /// Number of accepted rows.
    pub accepted_rows: usize,
    /// Number of rejected rows.
    pub rejected_rows: usize,
    /// Largest absolute normalized innovation across all rows.
    pub max_abs_normalized_innovation: Option<f64>,
    /// Largest absolute normalized innovation among rejected rows.
    pub max_rejected_abs_normalized_innovation: Option<f64>,
    /// Whether too few rows remained to apply the update.
    pub coasted: bool,
}

/// Diagnostics from one EKF correction attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct EkfCorrectionReport {
    /// Whether the correction was applied to the nominal state and covariance.
    pub applied: bool,
    /// Normalized innovation squared for the rows used by the report.
    pub normalized_innovation_squared: f64,
    /// Number of rows accepted by screening or used without screening.
    pub accepted_rows: usize,
    /// Number of rows rejected by screening.
    pub rejected_rows: usize,
    /// Optional innovation gate diagnostics.
    pub innovation_gate: Option<InnovationGateReport>,
    /// Innovation covariance `S`.
    pub innovation_covariance: Vec<Vec<f64>>,
    /// Kalman gain `K`.
    pub kalman_gain: Vec<Vec<f64>>,
    /// Error-state estimate applied to the closed-loop nominal state.
    pub dx: Vec<f64>,
}

/// Apply one EKF correction, then close the loop and reset the error vector.
pub fn ekf_correct_closed_loop(
    state: &mut InsFilterState,
    correction: &EkfCorrection,
    options: EkfUpdateOptions,
) -> Result<EkfCorrectionReport, FusionError> {
    state.validate()?;
    correction.validate_for_dimension(state.dimension())?;

    if let Some(gate) = options.innovation_gate {
        gate.validate()?;
        let full_s = innovation_covariance(&state.covariance, correction)?;
        let (screened, report) = screen_correction(correction, &full_s, gate)?;
        let full_nis = normalized_innovation_squared(&full_s, &correction.innovation)?;
        if report.coasted {
            return Ok(EkfCorrectionReport {
                applied: false,
                normalized_innovation_squared: full_nis,
                accepted_rows: report.accepted_rows,
                rejected_rows: report.rejected_rows,
                innovation_gate: Some(report),
                innovation_covariance: full_s,
                kalman_gain: vec![vec![0.0; correction.row_count()]; state.dimension()],
                dx: vec![0.0; state.dimension()],
            });
        }
        let accepted_rows = report.accepted_rows;
        let rejected_rows = report.rejected_rows;
        let mut applied = apply_correction(state, &screened)?;
        applied.accepted_rows = accepted_rows;
        applied.rejected_rows = rejected_rows;
        applied.innovation_gate = Some(report);
        return Ok(applied);
    }

    apply_correction(state, correction)
}

/// Apply one EKF correction using an inflated predicted covariance.
///
/// The scale is used for the innovation covariance, Kalman gain, and Joseph
/// covariance update. A scale of `1.0` is intentionally routed through
/// [`ekf_correct_closed_loop`] by callers that need bit-exact default behavior.
pub(super) fn ekf_correct_closed_loop_with_predicted_covariance_scale(
    state: &mut InsFilterState,
    correction: &EkfCorrection,
    options: EkfUpdateOptions,
    predicted_covariance_scale: f64,
) -> Result<EkfCorrectionReport, FusionError> {
    state.validate()?;
    correction.validate_for_dimension(state.dimension())?;
    validate_positive(predicted_covariance_scale, "predicted_covariance_scale")?;

    let predicted_covariance = scaled_covariance(&state.covariance, predicted_covariance_scale);
    validate_covariance_matrix(
        &predicted_covariance,
        state.dimension(),
        "scaled_covariance",
    )?;

    if let Some(gate) = options.innovation_gate {
        gate.validate()?;
        let full_s = innovation_covariance(&predicted_covariance, correction)?;
        let (screened, report) = screen_correction(correction, &full_s, gate)?;
        let full_nis = normalized_innovation_squared(&full_s, &correction.innovation)?;
        if report.coasted {
            return Ok(EkfCorrectionReport {
                applied: false,
                normalized_innovation_squared: full_nis,
                accepted_rows: report.accepted_rows,
                rejected_rows: report.rejected_rows,
                innovation_gate: Some(report),
                innovation_covariance: full_s,
                kalman_gain: vec![vec![0.0; correction.row_count()]; state.dimension()],
                dx: vec![0.0; state.dimension()],
            });
        }
        let accepted_rows = report.accepted_rows;
        let rejected_rows = report.rejected_rows;
        let mut applied =
            apply_correction_with_predicted_covariance(state, &screened, &predicted_covariance)?;
        applied.accepted_rows = accepted_rows;
        applied.rejected_rows = rejected_rows;
        applied.innovation_gate = Some(report);
        return Ok(applied);
    }

    apply_correction_with_predicted_covariance(state, correction, &predicted_covariance)
}

/// Compute Joseph-form covariance update.
pub fn joseph_covariance_update(
    covariance: &[Vec<f64>],
    design: &[Vec<f64>],
    kalman_gain: &[Vec<f64>],
    measurement_covariance: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, FusionError> {
    let dimension = covariance.len();
    validate_covariance_matrix(covariance, dimension, "covariance")?;
    if design.is_empty() {
        return Err(invalid_input("design", "must not be empty"));
    }
    validate_matrix_cols(design, dimension, "design")?;
    if kalman_gain.len() != dimension {
        return Err(FusionError::DimensionMismatch {
            field: "kalman_gain",
            expected: dimension,
            actual: kalman_gain.len(),
        });
    }
    validate_matrix_cols(kalman_gain, design.len(), "kalman_gain")?;
    validate_measurement_covariance(measurement_covariance, design.len())?;

    let kh = matmul(kalman_gain, design)?;
    let identity_minus_kh = matrix_sub(&identity(dimension), &kh)?;
    let left = matmul(&identity_minus_kh, covariance)?;
    let right = transpose(&identity_minus_kh)?;
    let stabilized = matmul(&left, &right)?;
    let kr = matmul(kalman_gain, measurement_covariance)?;
    let k_t = transpose(kalman_gain)?;
    let noise = matmul(&kr, &k_t)?;
    let mut updated = matrix_add(&stabilized, &noise)?;
    reproject_covariance_psd(&mut updated, "joseph_covariance")?;
    Ok(updated)
}

/// Apply an indirect error estimate to the nominal INS state.
pub fn apply_closed_loop_error(
    state: &mut crate::inertial::NavState,
    dx: &[f64],
    layout: ErrorStateLayout,
) -> Result<(), FusionError> {
    layout.validate_len(dx.len(), "dx")?;
    validate_finite_slice(dx, "dx")?;
    if layout.includes_scale_factors()
        && dx[ERROR_ACCEL_SCALE_INDEX..ERROR_GYRO_SCALE_INDEX + 3]
            .iter()
            .any(|value| *value != 0.0)
    {
        return Err(invalid_input(
            "dx",
            "scale-factor errors require filter state",
        ));
    }
    apply_closed_loop_navigation_error(state, dx)
}

pub(super) fn apply_closed_loop_navigation_error(
    state: &mut crate::inertial::NavState,
    dx: &[f64],
) -> Result<(), FusionError> {
    for axis in 0..3 {
        state.position_ecef_m[axis] -= dx[ERROR_POSITION_INDEX + axis];
        state.velocity_ecef_mps[axis] -= dx[ERROR_VELOCITY_INDEX + axis];
    }

    let psi = [
        dx[ERROR_ATTITUDE_INDEX],
        dx[ERROR_ATTITUDE_INDEX + 1],
        dx[ERROR_ATTITUDE_INDEX + 2],
    ];
    let psi_skew = skew(psi);
    let mut correction = mat3_identity();
    for row in 0..3 {
        for col in 0..3 {
            correction[row][col] -= psi_skew[row][col];
        }
    }
    let attitude = inline_rxr(&correction, &state.attitude_body_to_ecef);
    state.attitude_body_to_ecef = reorthonormalize_dcm(&attitude)?;

    for axis in 0..3 {
        state.accel_bias_mps2[axis] += dx[ERROR_ACCEL_BIAS_INDEX + axis];
        state.gyro_bias_rps[axis] += dx[ERROR_GYRO_BIAS_INDEX + axis];
    }
    state.validate()?;
    Ok(())
}

pub(super) fn apply_closed_loop_scale_error(state: &mut InsFilterState, dx: &[f64]) {
    if state.layout().includes_scale_factors() {
        for axis in 0..3 {
            state.accel_scale_factor[axis] += dx[ERROR_ACCEL_SCALE_INDEX + axis];
            state.gyro_scale_factor[axis] += dx[ERROR_GYRO_SCALE_INDEX + axis];
        }
    }
}

fn apply_correction(
    state: &mut InsFilterState,
    correction: &EkfCorrection,
) -> Result<EkfCorrectionReport, FusionError> {
    let covariance = state.covariance.clone();
    apply_correction_with_predicted_covariance(state, correction, &covariance)
}

fn apply_correction_with_predicted_covariance(
    state: &mut InsFilterState,
    correction: &EkfCorrection,
    predicted_covariance: &[Vec<f64>],
) -> Result<EkfCorrectionReport, FusionError> {
    let dimension = state.dimension();
    validate_covariance_matrix(predicted_covariance, dimension, "predicted_covariance")?;
    let s = innovation_covariance(predicted_covariance, correction)?;
    let h_t = transpose(&correction.design)?;
    let p_h_t = matmul(predicted_covariance, &h_t)?;
    let mut kalman_gain = vec![vec![0.0; correction.row_count()]; dimension];
    let mut scratch = crate::astro::math::linear::FlatCholeskySolveScratch::default();
    for row in 0..dimension {
        kalman_gain[row] = solve_spd(&s, &p_h_t[row], &mut scratch)?;
    }
    let dx = matvec(&kalman_gain, &correction.innovation)?;
    let nis = normalized_innovation_squared(&s, &correction.innovation)?;
    let covariance = joseph_covariance_update(
        predicted_covariance,
        &correction.design,
        &kalman_gain,
        &correction.measurement_covariance,
    )?;

    apply_closed_loop_navigation_error(&mut state.nominal, &dx)?;
    apply_closed_loop_scale_error(state, &dx);
    state.covariance = covariance;
    state.reset_error_state();
    state.validate()?;

    Ok(EkfCorrectionReport {
        applied: true,
        normalized_innovation_squared: nis,
        accepted_rows: correction.row_count(),
        rejected_rows: 0,
        innovation_gate: None,
        innovation_covariance: s,
        kalman_gain,
        dx,
    })
}

fn scaled_covariance(covariance: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    covariance
        .iter()
        .map(|row| row.iter().map(|value| value * scale).collect())
        .collect()
}

pub(super) fn innovation_covariance(
    covariance: &[Vec<f64>],
    correction: &EkfCorrection,
) -> Result<Vec<Vec<f64>>, FusionError> {
    let hp = matmul(&correction.design, covariance)?;
    let h_t = transpose(&correction.design)?;
    let hph_t = matmul(&hp, &h_t)?;
    matrix_add(&hph_t, &correction.measurement_covariance)
}

fn validate_measurement_covariance(
    measurement_covariance: &[Vec<f64>],
    dimension: usize,
) -> Result<(), FusionError> {
    if dimension == 0 {
        return Err(invalid_input("measurement_covariance", "must not be empty"));
    }
    validate_covariance_matrix(measurement_covariance, dimension, "measurement_covariance")?;
    let matrix = dmatrix_from_rows(measurement_covariance);
    if portable::cholesky_lower_dynamic(&matrix).is_some() {
        Ok(())
    } else {
        Err(FusionError::NonPositiveDefinite {
            field: "measurement_covariance",
        })
    }
}

pub(super) fn normalized_innovation_squared(
    innovation_covariance: &[Vec<f64>],
    innovation: &[f64],
) -> Result<f64, FusionError> {
    validate_square_matrix(
        innovation_covariance,
        innovation.len(),
        "innovation_covariance",
    )?;
    validate_finite_slice(innovation, "innovation")?;
    let mut scratch = crate::astro::math::linear::FlatCholeskySolveScratch::default();
    let solved = solve_spd(innovation_covariance, innovation, &mut scratch)?;
    Ok(innovation
        .iter()
        .zip(solved.iter())
        .map(|(a, b)| a * b)
        .sum())
}

pub(super) fn screen_correction(
    correction: &EkfCorrection,
    innovation_covariance: &[Vec<f64>],
    gate: InnovationGate,
) -> Result<(EkfCorrection, InnovationGateReport), FusionError> {
    let mut accepted_indices = Vec::with_capacity(correction.row_count());
    let mut rejected_rows = 0usize;
    let mut max_abs_normalized_innovation = None;
    let mut max_rejected_abs_normalized_innovation = None;

    for (row, s_row) in innovation_covariance
        .iter()
        .enumerate()
        .take(correction.row_count())
    {
        let variance = s_row[row];
        validate_positive(variance, "innovation_covariance_diagonal")?;
        let normalized = (correction.innovation[row] / variance.sqrt()).abs();
        max_abs_normalized_innovation = Some(
            max_abs_normalized_innovation
                .map_or(normalized, |current: f64| current.max(normalized)),
        );
        if normalized <= gate.threshold_sigma {
            accepted_indices.push(row);
        } else {
            rejected_rows += 1;
            max_rejected_abs_normalized_innovation = Some(
                max_rejected_abs_normalized_innovation
                    .map_or(normalized, |current: f64| current.max(normalized)),
            );
        }
    }

    let coasted = accepted_indices.len() < gate.min_rows;
    let report = InnovationGateReport {
        threshold_sigma: gate.threshold_sigma,
        min_rows: gate.min_rows,
        input_rows: correction.row_count(),
        accepted_rows: accepted_indices.len(),
        rejected_rows,
        max_abs_normalized_innovation,
        max_rejected_abs_normalized_innovation,
        coasted,
    };

    if coasted {
        return Ok((correction.clone(), report));
    }

    let innovation = accepted_indices
        .iter()
        .map(|idx| correction.innovation[*idx])
        .collect::<Vec<_>>();
    let design = accepted_indices
        .iter()
        .map(|idx| correction.design[*idx].clone())
        .collect::<Vec<_>>();
    let mut measurement_covariance =
        vec![vec![0.0; accepted_indices.len()]; accepted_indices.len()];
    for (row_out, row_in) in accepted_indices.iter().enumerate() {
        for (col_out, col_in) in accepted_indices.iter().enumerate() {
            measurement_covariance[row_out][col_out] =
                correction.measurement_covariance[*row_in][*col_in];
        }
    }
    let screened = EkfCorrection::new(innovation, design, measurement_covariance)?;
    Ok((screened, report))
}

#[cfg(test)]
mod tests {
    //! Provenance: EKF correction tests use the standard Kalman innovation
    //! equations and Joseph stabilized covariance identity. The closed-loop
    //! reset follows the indirect INS convention in Groves, Principles of GNSS,
    //! Inertial, and Multisensor Integrated Navigation Systems, 2nd ed.,
    //! Chapter 14.1.

    use super::*;
    use crate::astro::constants::earth::WGS84_A_M;
    use crate::inertial::state::mat3_identity;
    use crate::inertial::NavState;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.17e}"
        );
    }

    fn nominal_state() -> NavState {
        NavState::new(10.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity())
            .expect("nominal state")
    }

    #[test]
    fn closed_loop_reset_subtracts_navigation_errors_and_adds_biases() {
        let mut state = nominal_state();
        let mut dx = vec![0.0; 15];
        dx[0] = 2.0;
        dx[4] = -3.0;
        dx[9] = 0.01;
        dx[14] = -0.02;
        apply_closed_loop_error(&mut state, &dx, ErrorStateLayout::Fifteen)
            .expect("closed-loop reset");
        assert_eq!(
            state.position_ecef_m[0].to_bits(),
            (WGS84_A_M - 2.0).to_bits()
        );
        assert_eq!(state.velocity_ecef_mps[1].to_bits(), 3.0_f64.to_bits());
        assert_eq!(state.accel_bias_mps2[0].to_bits(), 0.01_f64.to_bits());
        assert_eq!(state.gyro_bias_rps[2].to_bits(), (-0.02_f64).to_bits());
    }

    #[test]
    fn closed_loop_nav_helper_rejects_nonzero_scale_errors() {
        let mut state = nominal_state();
        let mut dx = vec![0.0; 21];
        dx[ERROR_ACCEL_SCALE_INDEX] = 0.25;
        let err = apply_closed_loop_error(&mut state, &dx, ErrorStateLayout::TwentyOne)
            .expect_err("scale errors require filter state");
        assert!(matches!(
            err,
            FusionError::InvalidInput {
                field: "dx",
                reason: "scale-factor errors require filter state"
            }
        ));
    }

    #[test]
    fn ekf_correction_applies_21_state_scale_errors_before_reset() {
        let mut covariance = vec![vec![0.0; 21]; 21];
        for (idx, row) in covariance.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        let mut state =
            InsFilterState::new(nominal_state(), ErrorStateLayout::TwentyOne, covariance)
                .expect("filter state");
        let mut design = vec![vec![0.0; 21]; 6];
        for axis in 0..3 {
            design[axis][ERROR_ACCEL_SCALE_INDEX + axis] = 1.0;
            design[axis + 3][ERROR_GYRO_SCALE_INDEX + axis] = 1.0;
        }
        let correction = EkfCorrection::new(
            vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0],
            design,
            vec![
                vec![3.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 3.0, 0.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 3.0, 0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
                vec![0.0, 0.0, 0.0, 0.0, 0.0, 3.0],
            ],
        )
        .expect("correction");

        let report = ekf_correct_closed_loop(&mut state, &correction, EkfUpdateOptions::default())
            .expect("ekf correction");

        assert!(report.applied);
        assert_eq!(state.error_state.as_slice(), &[0.0; 21]);
        assert_eq!(state.accel_scale_factor[0].to_bits(), 0.25_f64.to_bits());
        assert_eq!(state.accel_scale_factor[1].to_bits(), (-0.5_f64).to_bits());
        assert_eq!(state.accel_scale_factor[2].to_bits(), 0.75_f64.to_bits());
        assert_eq!(state.gyro_scale_factor[0].to_bits(), (-1.0_f64).to_bits());
        assert_eq!(state.gyro_scale_factor[1].to_bits(), 1.25_f64.to_bits());
        assert_eq!(state.gyro_scale_factor[2].to_bits(), (-1.5_f64).to_bits());
    }

    #[test]
    fn joseph_matches_naive_well_conditioned_to_bits() {
        let covariance = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let design = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let kalman_gain = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        let measurement_covariance = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let joseph =
            joseph_covariance_update(&covariance, &design, &kalman_gain, &measurement_covariance)
                .expect("joseph covariance");
        let naive = naive_covariance_update(&covariance, &design, &kalman_gain).expect("naive");
        for row in 0..2 {
            for col in 0..2 {
                assert_eq!(joseph[row][col].to_bits(), naive[row][col].to_bits());
            }
        }
    }

    #[test]
    fn joseph_stays_psd_for_ill_conditioned_update_where_naive_fails() {
        let covariance = vec![
            vec![1.0e6, 1.0e6 * (1.0 - 2.0e-15)],
            vec![1.0e6 * (1.0 - 2.0e-15), 1.0e6],
        ];
        let design = vec![vec![1.0, 0.0]];
        let measurement_covariance = vec![vec![1.0e-30]];
        let correction = EkfCorrection::new(vec![0.0], design.clone(), measurement_covariance)
            .expect("correction");
        let s = innovation_covariance(&covariance, &correction).expect("innovation covariance");
        let h_t = transpose(&design).expect("transpose");
        let p_h_t = matmul(&covariance, &h_t).expect("pht");
        let mut scratch = crate::astro::math::linear::FlatCholeskySolveScratch::default();
        let kalman_gain = vec![
            solve_spd(&s, &p_h_t[0], &mut scratch).expect("gain row 0"),
            solve_spd(&s, &p_h_t[1], &mut scratch).expect("gain row 1"),
        ];
        let joseph = joseph_covariance_update(
            &covariance,
            &design,
            &kalman_gain,
            &correction.measurement_covariance,
        )
        .expect("joseph covariance");
        let naive = naive_covariance_update(&covariance, &design, &kalman_gain).expect("naive");

        assert!(
            super::super::state::covariance_is_positive_semidefinite(&joseph).expect("joseph psd")
        );
        assert!(
            !super::super::state::covariance_is_positive_semidefinite(&naive).expect("naive psd"),
            "naive covariance unexpectedly remained PSD: {naive:?}"
        );
    }

    #[test]
    fn ekf_correction_applies_closed_loop_and_resets_dx() {
        let mut covariance = vec![vec![0.0; 15]; 15];
        for (idx, row) in covariance.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        let mut state = InsFilterState::new(nominal_state(), ErrorStateLayout::Fifteen, covariance)
            .expect("filter state");
        let mut design = vec![vec![0.0; 15]; 3];
        for (axis, row) in design.iter_mut().enumerate().take(3) {
            row[axis] = 1.0;
        }
        let correction = EkfCorrection::new(
            vec![1.0, 0.0, 0.0],
            design,
            vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ],
        )
        .expect("correction");
        let report = ekf_correct_closed_loop(
            &mut state,
            &correction,
            EkfUpdateOptions {
                innovation_gate: Some(InnovationGate {
                    threshold_sigma: 3.0,
                    min_rows: 3,
                }),
            },
        )
        .expect("ekf correction");
        assert!(report.applied);
        assert_close(report.normalized_innovation_squared, 0.5, 1.0e-16);
        assert_eq!(state.error_state.as_slice(), &[0.0; 15]);
        assert_close(state.nominal.position_ecef_m[0], WGS84_A_M - 0.5, 0.0);
    }

    #[test]
    fn ekf_correction_rejects_singular_measurement_covariance() {
        let mut design = vec![vec![0.0; 15]; 1];
        design[0][0] = 1.0;
        let err = EkfCorrection::new(vec![1.0], design, vec![vec![0.0]])
            .expect_err("singular covariance must be rejected");
        assert!(matches!(
            err,
            FusionError::NonPositiveDefinite {
                field: "measurement_covariance"
            }
        ));
    }

    #[test]
    fn innovation_gate_reports_rejected_rows_when_update_still_applies() {
        let mut covariance = vec![vec![0.0; 15]; 15];
        for (idx, row) in covariance.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        let mut state = InsFilterState::new(nominal_state(), ErrorStateLayout::Fifteen, covariance)
            .expect("filter state");
        let mut design = vec![vec![0.0; 15]; 2];
        design[0][0] = 1.0;
        design[1][1] = 1.0;
        let correction = EkfCorrection::new(
            vec![1.0, 10.0],
            design,
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        )
        .expect("correction");
        let report = ekf_correct_closed_loop(
            &mut state,
            &correction,
            EkfUpdateOptions {
                innovation_gate: Some(InnovationGate {
                    threshold_sigma: 3.0,
                    min_rows: 1,
                }),
            },
        )
        .expect("ekf correction");

        assert!(report.applied);
        assert_eq!(report.accepted_rows, 1);
        assert_eq!(report.rejected_rows, 1);
        let gate = report.innovation_gate.expect("gate report");
        assert_eq!(gate.accepted_rows, 1);
        assert_eq!(gate.rejected_rows, 1);
    }

    #[test]
    fn innovation_gate_rejects_large_row_and_coasts_below_minimum() {
        let mut covariance = vec![vec![0.0; 15]; 15];
        for (idx, row) in covariance.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        let mut state = InsFilterState::new(nominal_state(), ErrorStateLayout::Fifteen, covariance)
            .expect("filter state");
        let mut design = vec![vec![0.0; 15]; 1];
        design[0][0] = 1.0;
        let correction =
            EkfCorrection::new(vec![10.0], design, vec![vec![1.0]]).expect("correction");
        let report = ekf_correct_closed_loop(
            &mut state,
            &correction,
            EkfUpdateOptions {
                innovation_gate: Some(InnovationGate {
                    threshold_sigma: 3.0,
                    min_rows: 1,
                }),
            },
        )
        .expect("ekf correction");
        assert!(!report.applied);
        assert_eq!(report.accepted_rows, 0);
        assert_eq!(report.rejected_rows, 1);
        assert_eq!(
            state.nominal.position_ecef_m[0].to_bits(),
            WGS84_A_M.to_bits()
        );
    }

    fn naive_covariance_update(
        covariance: &[Vec<f64>],
        design: &[Vec<f64>],
        kalman_gain: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, FusionError> {
        let kh = matmul(kalman_gain, design)?;
        let identity_minus_kh = matrix_sub(&identity(covariance.len()), &kh)?;
        matmul(&identity_minus_kh, covariance)
    }
}
