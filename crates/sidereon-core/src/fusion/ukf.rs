//! Scaled-sigma-point unscented correction for the fusion error state.

use nalgebra::DMatrix;

use crate::astro::math::portable;

use super::ekf::{
    apply_closed_loop_navigation_error, apply_closed_loop_scale_error,
    normalized_innovation_squared, EkfCorrection, EkfCorrectionReport, InnovationGate,
    InnovationGateReport,
};
use super::state::{
    covariance_eigenvalue_tolerance, dmatrix_from_rows, invalid_input, matmul, matrix_sub,
    reproject_covariance_psd, solve_spd, symmetrize_in_place, transpose,
    validate_covariance_matrix, validate_finite_slice, validate_matrix_cols, validate_nonnegative,
    validate_positive, FusionError, InsFilterState,
};

/// Scaled unscented-transform parameters.
///
/// `alpha`, `beta`, and `kappa` produce Wan/van der Merwe sigma-point weights
/// with `lambda = alpha^2 * (n + kappa) - n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnscentedTransformOptions {
    /// Sigma-point spread around the mean.
    pub alpha: f64,
    /// Prior-distribution shape parameter. `2.0` is the Gaussian choice.
    pub beta: f64,
    /// Secondary sigma-point scaling parameter.
    pub kappa: f64,
}

impl Default for UnscentedTransformOptions {
    fn default() -> Self {
        Self {
            alpha: 0.5,
            beta: 2.0,
            kappa: 0.0,
        }
    }
}

impl UnscentedTransformOptions {
    /// Validate scaling parameters for a state dimension.
    pub fn validate_for_dimension(&self, dimension: usize) -> Result<(), FusionError> {
        if dimension == 0 {
            return Err(invalid_input("dimension", "must be positive"));
        }
        validate_positive(self.alpha, "ukf_alpha")?;
        validate_nonnegative(self.beta, "ukf_beta")?;
        validate_finite_slice(&[self.kappa], "ukf_kappa")?;
        let scale = self.scale(dimension);
        if scale.is_finite() && scale > 0.0 {
            Ok(())
        } else {
            Err(invalid_input("ukf_scale", "must be positive"))
        }
    }

    fn lambda(self, dimension: usize) -> f64 {
        self.alpha * self.alpha * (dimension as f64 + self.kappa) - dimension as f64
    }

    fn scale(self, dimension: usize) -> f64 {
        dimension as f64 + self.lambda(dimension)
    }
}

/// UKF measurement-correction options.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct UkfUpdateOptions {
    /// Scaled unscented-transform parameters.
    pub transform: UnscentedTransformOptions,
    /// Optional normalized-innovation screen applied before correction.
    pub innovation_gate: Option<InnovationGate>,
}

impl UkfUpdateOptions {
    /// Validate transform and gate options for a state dimension.
    pub fn validate_for_dimension(&self, dimension: usize) -> Result<(), FusionError> {
        self.transform.validate_for_dimension(dimension)?;
        if let Some(gate) = self.innovation_gate {
            gate.validate()?;
        }
        Ok(())
    }
}

/// Apply a linear UKF correction, then close the loop and reset the error vector.
///
/// This uses the same [`EkfCorrection`] measurement struct as the EKF path. The
/// supplied design matrix is evaluated as a linear measurement function at each
/// sigma point.
pub fn ukf_correct_closed_loop(
    state: &mut InsFilterState,
    correction: &EkfCorrection,
    options: UkfUpdateOptions,
) -> Result<EkfCorrectionReport, FusionError> {
    state.validate()?;
    correction.validate_for_dimension(state.dimension())?;
    options.validate_for_dimension(state.dimension())?;

    let report = ukf_measurement_update(
        &state.covariance,
        &correction.innovation,
        &correction.measurement_covariance,
        options,
        |sigma| super::state::matvec(&correction.design, sigma),
    )?;
    if !report.applied {
        return Ok(report.into_public_report());
    }

    apply_closed_loop_navigation_error(&mut state.nominal, &report.dx)?;
    apply_closed_loop_scale_error(state, &report.dx);
    state.covariance = report.posterior_covariance.clone();
    state.reset_error_state();
    state.validate()?;
    Ok(report.into_public_report())
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct InternalUkfReport {
    pub(crate) applied: bool,
    pub(crate) normalized_innovation_squared: f64,
    pub(crate) accepted_rows: usize,
    pub(crate) rejected_rows: usize,
    pub(crate) innovation_gate: Option<InnovationGateReport>,
    pub(crate) innovation_covariance: Vec<Vec<f64>>,
    pub(crate) kalman_gain: Vec<Vec<f64>>,
    pub(crate) dx: Vec<f64>,
    pub(crate) posterior_covariance: Vec<Vec<f64>>,
}

impl InternalUkfReport {
    pub(crate) fn into_public_report(self) -> EkfCorrectionReport {
        EkfCorrectionReport {
            applied: self.applied,
            normalized_innovation_squared: self.normalized_innovation_squared,
            accepted_rows: self.accepted_rows,
            rejected_rows: self.rejected_rows,
            innovation_gate: self.innovation_gate,
            innovation_covariance: self.innovation_covariance,
            kalman_gain: self.kalman_gain,
            dx: self.dx,
        }
    }
}

pub(crate) fn ukf_measurement_update<F>(
    covariance: &[Vec<f64>],
    innovation: &[f64],
    measurement_covariance: &[Vec<f64>],
    options: UkfUpdateOptions,
    measurement_model: F,
) -> Result<InternalUkfReport, FusionError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, FusionError>,
{
    let dimension = covariance.len();
    validate_covariance_matrix(covariance, dimension, "covariance")?;
    validate_finite_slice(innovation, "innovation")?;
    validate_covariance_matrix(
        measurement_covariance,
        innovation.len(),
        "measurement_covariance",
    )?;
    options.validate_for_dimension(dimension)?;

    let sigma = sigma_points(covariance, options.transform)?;
    let prediction = measurement_statistics(&sigma, innovation.len(), &measurement_model)?;
    let full = predicted_update(
        covariance,
        innovation,
        measurement_covariance,
        &prediction,
        None,
    )?;

    let Some(gate) = options.innovation_gate else {
        return Ok(full);
    };

    let (accepted, gate_report) = screen_rows(
        innovation,
        &prediction.mean,
        &full.innovation_covariance,
        gate,
    )?;
    if gate_report.coasted {
        let full_nis = normalized_innovation_squared(
            &full.innovation_covariance,
            &innovation_residual(innovation, &prediction.mean)?,
        )?;
        return Ok(InternalUkfReport {
            applied: false,
            normalized_innovation_squared: full_nis,
            accepted_rows: gate_report.accepted_rows,
            rejected_rows: gate_report.rejected_rows,
            innovation_gate: Some(gate_report),
            innovation_covariance: full.innovation_covariance,
            kalman_gain: vec![vec![0.0; innovation.len()]; dimension],
            dx: vec![0.0; dimension],
            posterior_covariance: covariance.to_vec(),
        });
    }

    let mut screened = predicted_update(
        covariance,
        innovation,
        measurement_covariance,
        &prediction,
        Some(&accepted),
    )?;
    screened.accepted_rows = gate_report.accepted_rows;
    screened.rejected_rows = gate_report.rejected_rows;
    screened.innovation_gate = Some(gate_report);
    Ok(screened)
}

#[derive(Debug, Clone, PartialEq)]
struct SigmaSet {
    points: Vec<Vec<f64>>,
    mean_weights: Vec<f64>,
    covariance_weights: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
struct MeasurementPrediction {
    values: Vec<Vec<f64>>,
    mean: Vec<f64>,
    cross_covariance: Vec<Vec<f64>>,
    covariance_weights: Vec<f64>,
}

fn sigma_points(
    covariance: &[Vec<f64>],
    options: UnscentedTransformOptions,
) -> Result<SigmaSet, FusionError> {
    let dimension = covariance.len();
    options.validate_for_dimension(dimension)?;
    let scale = options.scale(dimension);
    let lambda = options.lambda(dimension);
    let gamma = scale.sqrt();
    let sqrt = covariance_square_root(covariance)?;

    let point_count = 2 * dimension + 1;
    let mut points = Vec::with_capacity(point_count);
    points.push(vec![0.0; dimension]);
    for col in 0..dimension {
        let mut point = vec![0.0; dimension];
        for row in 0..dimension {
            point[row] = gamma * sqrt[(row, col)];
        }
        points.push(point);
    }
    for col in 0..dimension {
        let mut point = vec![0.0; dimension];
        for row in 0..dimension {
            point[row] = -gamma * sqrt[(row, col)];
        }
        points.push(point);
    }

    let mut mean_weights = vec![0.5 / scale; point_count];
    let mut covariance_weights = mean_weights.clone();
    mean_weights[0] = lambda / scale;
    covariance_weights[0] = mean_weights[0] + (1.0 - options.alpha * options.alpha + options.beta);

    Ok(SigmaSet {
        points,
        mean_weights,
        covariance_weights,
    })
}

fn covariance_square_root(covariance: &[Vec<f64>]) -> Result<DMatrix<f64>, FusionError> {
    let dimension = covariance.len();
    validate_covariance_matrix(covariance, dimension, "covariance")?;
    let matrix = dmatrix_from_rows(covariance);
    if let Some(cholesky) = portable::cholesky_lower_dynamic(&matrix) {
        return Ok(cholesky);
    }

    let (eigenvectors, eigenvalues) = portable::symmetric_eigen_dynamic(&matrix);
    let mut diagonal = DMatrix::<f64>::zeros(dimension, dimension);
    for idx in 0..dimension {
        let eigenvalue = eigenvalues[idx];
        if eigenvalue < 0.0 {
            let tolerance = covariance_eigenvalue_tolerance(covariance, &eigenvectors, idx);
            if eigenvalue < -tolerance {
                return Err(FusionError::NonPositiveSemidefinite {
                    field: "covariance",
                });
            }
            diagonal[(idx, idx)] = 0.0;
        } else {
            diagonal[(idx, idx)] = eigenvalue.sqrt();
        }
    }
    Ok(portable::product(&eigenvectors, &diagonal))
}

fn measurement_statistics<F>(
    sigma: &SigmaSet,
    measurement_dimension: usize,
    measurement_model: &F,
) -> Result<MeasurementPrediction, FusionError>
where
    F: Fn(&[f64]) -> Result<Vec<f64>, FusionError>,
{
    let mut values = Vec::with_capacity(sigma.points.len());
    for point in &sigma.points {
        let value = measurement_model(point)?;
        if value.len() != measurement_dimension {
            return Err(FusionError::DimensionMismatch {
                field: "ukf_measurement",
                expected: measurement_dimension,
                actual: value.len(),
            });
        }
        validate_finite_slice(&value, "ukf_measurement")?;
        values.push(value);
    }

    let mut mean = vec![0.0; measurement_dimension];
    for (weight, value) in sigma.mean_weights.iter().zip(values.iter()) {
        for col in 0..measurement_dimension {
            mean[col] += weight * value[col];
        }
    }

    let state_dimension = sigma.points[0].len();
    let mut cross_covariance = vec![vec![0.0; measurement_dimension]; state_dimension];
    for (idx, point) in sigma.points.iter().enumerate() {
        let weight = sigma.covariance_weights[idx];
        for row in 0..state_dimension {
            for col in 0..measurement_dimension {
                cross_covariance[row][col] += weight * point[row] * (values[idx][col] - mean[col]);
            }
        }
    }

    Ok(MeasurementPrediction {
        values,
        mean,
        cross_covariance,
        covariance_weights: sigma.covariance_weights.clone(),
    })
}

fn predicted_update(
    covariance: &[Vec<f64>],
    innovation: &[f64],
    measurement_covariance: &[Vec<f64>],
    prediction: &MeasurementPrediction,
    accepted: Option<&[usize]>,
) -> Result<InternalUkfReport, FusionError> {
    let selected = accepted
        .map(<[usize]>::to_vec)
        .unwrap_or_else(|| (0..innovation.len()).collect());
    let innovation = select_vector(innovation, &selected)?;
    let mean = select_vector(&prediction.mean, &selected)?;
    let measurement_covariance = select_matrix(measurement_covariance, &selected)?;
    let values = prediction
        .values
        .iter()
        .map(|value| select_vector(value, &selected))
        .collect::<Result<Vec<_>, _>>()?;
    let cross_covariance = select_columns(&prediction.cross_covariance, &selected)?;

    let residual = innovation_residual(&innovation, &mean)?;
    let mut innovation_covariance = measurement_covariance;
    for (idx, value) in values.iter().enumerate() {
        let weight = prediction.covariance_weights[idx];
        for row in 0..selected.len() {
            let dy_row = value[row] - mean[row];
            for col in 0..selected.len() {
                innovation_covariance[row][col] += weight * dy_row * (value[col] - mean[col]);
            }
        }
    }
    symmetrize_in_place(&mut innovation_covariance);
    validate_covariance_matrix(
        &innovation_covariance,
        selected.len(),
        "innovation_covariance",
    )?;

    let mut kalman_gain = vec![vec![0.0; selected.len()]; covariance.len()];
    let mut scratch = crate::astro::math::linear::FlatCholeskySolveScratch::default();
    for row in 0..covariance.len() {
        kalman_gain[row] = solve_spd(&innovation_covariance, &cross_covariance[row], &mut scratch)?;
    }

    let dx = super::state::matvec(&kalman_gain, &residual)?;
    let nis = normalized_innovation_squared(&innovation_covariance, &residual)?;
    let ks = matmul(&kalman_gain, &innovation_covariance)?;
    let k_t = transpose(&kalman_gain)?;
    let ksk_t = matmul(&ks, &k_t)?;
    let mut posterior_covariance = matrix_sub(covariance, &ksk_t)?;
    symmetrize_in_place(&mut posterior_covariance);
    reproject_covariance_psd(&mut posterior_covariance, "ukf_covariance")?;

    Ok(InternalUkfReport {
        applied: true,
        normalized_innovation_squared: nis,
        accepted_rows: selected.len(),
        rejected_rows: innovation.len().saturating_sub(selected.len()),
        innovation_gate: None,
        innovation_covariance,
        kalman_gain,
        dx,
        posterior_covariance,
    })
}

fn innovation_residual(innovation: &[f64], mean: &[f64]) -> Result<Vec<f64>, FusionError> {
    if innovation.len() != mean.len() {
        return Err(FusionError::DimensionMismatch {
            field: "innovation_mean",
            expected: innovation.len(),
            actual: mean.len(),
        });
    }
    Ok(innovation
        .iter()
        .zip(mean.iter())
        .map(|(actual, predicted)| actual - predicted)
        .collect())
}

fn screen_rows(
    innovation: &[f64],
    mean: &[f64],
    innovation_covariance: &[Vec<f64>],
    gate: InnovationGate,
) -> Result<(Vec<usize>, InnovationGateReport), FusionError> {
    gate.validate()?;
    let residual = innovation_residual(innovation, mean)?;
    let mut accepted = Vec::with_capacity(innovation.len());
    let mut rejected_rows = 0usize;
    let mut max_abs_normalized_innovation = None;
    let mut max_rejected_abs_normalized_innovation = None;

    for (row, value) in residual.iter().enumerate() {
        let variance = innovation_covariance[row][row];
        validate_positive(variance, "innovation_covariance_diagonal")?;
        let normalized = (value / variance.sqrt()).abs();
        max_abs_normalized_innovation = Some(
            max_abs_normalized_innovation
                .map_or(normalized, |current: f64| current.max(normalized)),
        );
        if normalized <= gate.threshold_sigma {
            accepted.push(row);
        } else {
            rejected_rows += 1;
            max_rejected_abs_normalized_innovation = Some(
                max_rejected_abs_normalized_innovation
                    .map_or(normalized, |current: f64| current.max(normalized)),
            );
        }
    }

    let coasted = accepted.len() < gate.min_rows;
    let report = InnovationGateReport {
        threshold_sigma: gate.threshold_sigma,
        min_rows: gate.min_rows,
        input_rows: innovation.len(),
        accepted_rows: accepted.len(),
        rejected_rows,
        max_abs_normalized_innovation,
        max_rejected_abs_normalized_innovation,
        coasted,
    };
    Ok((accepted, report))
}

fn select_vector(values: &[f64], indices: &[usize]) -> Result<Vec<f64>, FusionError> {
    let mut selected = Vec::with_capacity(indices.len());
    for idx in indices {
        let Some(value) = values.get(*idx) else {
            return Err(FusionError::DimensionMismatch {
                field: "selected_measurement",
                expected: values.len(),
                actual: *idx,
            });
        };
        selected.push(*value);
    }
    Ok(selected)
}

fn select_matrix(matrix: &[Vec<f64>], indices: &[usize]) -> Result<Vec<Vec<f64>>, FusionError> {
    let mut out = vec![vec![0.0; indices.len()]; indices.len()];
    for (row_out, row_in) in indices.iter().enumerate() {
        for (col_out, col_in) in indices.iter().enumerate() {
            out[row_out][col_out] = matrix[*row_in][*col_in];
        }
    }
    Ok(out)
}

fn select_columns(matrix: &[Vec<f64>], indices: &[usize]) -> Result<Vec<Vec<f64>>, FusionError> {
    if matrix.is_empty() {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    validate_matrix_cols(matrix, matrix[0].len(), "matrix")?;
    let mut out = vec![vec![0.0; indices.len()]; matrix.len()];
    for (row_out, row) in matrix.iter().enumerate() {
        for (col_out, col_in) in indices.iter().enumerate() {
            out[row_out][col_out] = row[*col_in];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! Provenance: UKF weights and correction equations follow Wan and van der
    //! Merwe, The Unscented Kalman Filter for Nonlinear Estimation, 2000, and
    //! van der Merwe, Sigma-Point Kalman Filters for Probabilistic Inference in
    //! Dynamic State-Space Models, 2004, Section 3.2.3. The linear-measurement
    //! oracle is the closed-form scalar Kalman update `K = P H' / (H P H' + R)`.

    use super::*;
    use crate::astro::constants::earth::WGS84_A_M;
    use crate::fusion::ekf::{ekf_correct_closed_loop, EkfUpdateOptions};
    use crate::fusion::state::{ErrorStateLayout, ERROR_STATE_DIMENSION_15};
    use crate::inertial::state::mat3_identity;
    use crate::inertial::NavState;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.17e}"
        );
    }

    fn linear_test_state() -> InsFilterState {
        let nominal =
            NavState::new(0.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity()).expect("nominal");
        let mut covariance = vec![vec![0.0; ERROR_STATE_DIMENSION_15]; ERROR_STATE_DIMENSION_15];
        for (idx, row) in covariance.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        covariance[0][0] = 4.0;
        covariance[0][1] = 1.0;
        covariance[1][0] = 1.0;
        covariance[1][1] = 9.0;
        InsFilterState::new(nominal, ErrorStateLayout::Fifteen, covariance).expect("state")
    }

    #[test]
    fn linear_measurement_matches_closed_form_and_ekf() {
        let mut design = vec![vec![0.0; ERROR_STATE_DIMENSION_15]];
        design[0][0] = 0.5;
        design[0][1] = -2.0;
        let correction =
            EkfCorrection::new(vec![1.25], design, vec![vec![0.25]]).expect("correction");
        let mut ekf_state = linear_test_state();
        let mut ukf_state = linear_test_state();

        let ekf = ekf_correct_closed_loop(&mut ekf_state, &correction, EkfUpdateOptions::default())
            .expect("ekf");
        let ukf = ukf_correct_closed_loop(
            &mut ukf_state,
            &correction,
            UkfUpdateOptions {
                transform: UnscentedTransformOptions {
                    alpha: 1.0,
                    beta: 2.0,
                    kappa: 0.0,
                },
                innovation_gate: None,
            },
        )
        .expect("ukf");

        let expected_s = 35.25_f64;
        let expected_k0 = 0.0_f64;
        let expected_k1 = -17.5 / expected_s;
        let expected_dx1 = expected_k1 * 1.25;
        assert_close(ukf.innovation_covariance[0][0], expected_s, 1.0e-13);
        assert_close(ukf.kalman_gain[0][0], expected_k0, 1.0e-14);
        assert_close(ukf.kalman_gain[1][0], expected_k1, 1.0e-14);
        assert_close(ukf.dx[1], expected_dx1, 1.0e-14);

        for row in 0..ERROR_STATE_DIMENSION_15 {
            assert_close(ukf.kalman_gain[row][0], ekf.kalman_gain[row][0], 1.0e-15);
            assert_close(ukf.dx[row], ekf.dx[row], 1.0e-15);
            for col in 0..ERROR_STATE_DIMENSION_15 {
                assert_close(
                    ukf_state.covariance[row][col],
                    ekf_state.covariance[row][col],
                    1.0e-15,
                );
            }
        }
        assert_close(
            ukf_state.nominal.position_ecef_m[1],
            ekf_state.nominal.position_ecef_m[1],
            3.0e-13,
        );
    }
}
