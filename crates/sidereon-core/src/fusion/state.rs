//! Error-state layout, filter state, and covariance validation.

use nalgebra::DMatrix;

use crate::astro::math::portable;
use crate::inertial::{validate_finite, NavState};

/// Number of states in the position, velocity, attitude, and bias layout.
pub const ERROR_STATE_DIMENSION_15: usize = 15;
/// Number of states after adding accelerometer and gyroscope scale factors.
pub const ERROR_STATE_DIMENSION_21: usize = 21;
/// Start index of ECEF position error states.
pub const ERROR_POSITION_INDEX: usize = 0;
/// Start index of ECEF velocity error states.
pub const ERROR_VELOCITY_INDEX: usize = 3;
/// Start index of ECEF attitude error states.
pub const ERROR_ATTITUDE_INDEX: usize = 6;
/// Start index of accelerometer bias error states.
pub const ERROR_ACCEL_BIAS_INDEX: usize = 9;
/// Start index of gyroscope bias error states.
pub const ERROR_GYRO_BIAS_INDEX: usize = 12;
/// Start index of accelerometer scale-factor error states in the 21-state layout.
pub const ERROR_ACCEL_SCALE_INDEX: usize = 15;
/// Start index of gyroscope scale-factor error states in the 21-state layout.
pub const ERROR_GYRO_SCALE_INDEX: usize = 18;
/// Reserved start index for future mounting-misalignment error states.
pub const ERROR_MOUNTING_MISALIGNMENT_INDEX: usize = 21;
/// Reserved mounting-misalignment state count for a later layout.
pub const ERROR_MOUNTING_MISALIGNMENT_STATE_COUNT: usize = 3;

const PSD_REL_TOLERANCE: f64 = 128.0 * f64::EPSILON;

/// Error returned by GNSS/INS fusion primitives.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FusionError {
    /// A public input was non-finite or outside its documented domain.
    #[error("invalid fusion input {field}: {reason}")]
    InvalidInput {
        /// Name of the invalid field.
        field: &'static str,
        /// Short reason suitable for logs and tests.
        reason: &'static str,
    },
    /// A matrix or vector dimension did not match the selected state layout.
    #[error("invalid fusion dimension {field}: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Name of the invalid field.
        field: &'static str,
        /// Expected element, row, or column count.
        expected: usize,
        /// Actual element, row, or column count.
        actual: usize,
    },
    /// An innovation covariance could not be factored as positive definite.
    #[error("fusion innovation covariance is singular")]
    SingularInnovation,
    /// A covariance was not positive semidefinite under numerical bounds.
    #[error("fusion covariance {field} is not positive semidefinite")]
    NonPositiveSemidefinite {
        /// Name of the covariance field.
        field: &'static str,
    },
    /// A covariance was not positive definite under numerical bounds.
    #[error("fusion covariance {field} is not positive definite")]
    NonPositiveDefinite {
        /// Name of the covariance field.
        field: &'static str,
    },
    /// A nominal inertial state failed validation.
    #[error("invalid nominal inertial state")]
    NominalState,
}

impl From<crate::inertial::InertialError> for FusionError {
    fn from(_: crate::inertial::InertialError) -> Self {
        Self::NominalState
    }
}

/// Fusion filter family selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionFilterKind {
    /// Extended Kalman filter using the linearized error-state model.
    Ekf,
    /// Unscented Kalman filter using scaled sigma points.
    Ukf,
}

/// Error-state covariance layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorStateLayout {
    /// Fifteen-state layout `dr, dv, psi, b_a, b_g`.
    Fifteen,
    /// Twenty-one-state layout adding `s_a, s_g` after the first 15 states.
    TwentyOne,
}

impl ErrorStateLayout {
    /// Return the state dimension for this layout.
    pub const fn dimension(self) -> usize {
        match self {
            Self::Fifteen => ERROR_STATE_DIMENSION_15,
            Self::TwentyOne => ERROR_STATE_DIMENSION_21,
        }
    }

    /// Return whether this layout carries IMU scale-factor error states.
    pub const fn includes_scale_factors(self) -> bool {
        matches!(self, Self::TwentyOne)
    }

    /// Validate a vector length against this layout.
    pub fn validate_len(self, len: usize, field: &'static str) -> Result<(), FusionError> {
        let expected = self.dimension();
        if len == expected {
            Ok(())
        } else {
            Err(FusionError::DimensionMismatch {
                field,
                expected,
                actual: len,
            })
        }
    }
}

/// Indirect filter error vector.
#[derive(Debug, Clone, PartialEq)]
pub struct ErrorStateVector {
    layout: ErrorStateLayout,
    values: Vec<f64>,
}

impl ErrorStateVector {
    /// Build a zero error vector for `layout`.
    pub fn zeros(layout: ErrorStateLayout) -> Self {
        Self {
            layout,
            values: vec![0.0; layout.dimension()],
        }
    }

    /// Build an error vector from owned values.
    pub fn from_vec(layout: ErrorStateLayout, values: Vec<f64>) -> Result<Self, FusionError> {
        layout.validate_len(values.len(), "error_state")?;
        validate_finite_slice(&values, "error_state")?;
        Ok(Self { layout, values })
    }

    /// Return the state layout.
    pub const fn layout(&self) -> ErrorStateLayout {
        self.layout
    }

    /// Return the vector dimension.
    pub fn dimension(&self) -> usize {
        self.values.len()
    }

    /// Borrow the error-state values.
    pub fn as_slice(&self) -> &[f64] {
        &self.values
    }

    /// Mutably borrow the error-state values.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Reset all error-state values to zero.
    pub fn reset(&mut self) {
        self.values.fill(0.0);
    }

    /// Validate the vector shape and finite values.
    pub fn validate(&self) -> Result<(), FusionError> {
        self.layout.validate_len(self.values.len(), "error_state")?;
        validate_finite_slice(&self.values, "error_state")
    }
}

/// Closed-loop indirect INS filter state.
#[derive(Debug, Clone, PartialEq)]
pub struct InsFilterState {
    /// Nonlinear mechanized navigation state.
    pub nominal: NavState,
    /// Small error-state estimate, reset to zero after closed-loop correction.
    pub error_state: ErrorStateVector,
    /// Error-state covariance matrix in row-major nested-vector form.
    pub covariance: Vec<Vec<f64>>,
    /// Fractional accelerometer scale-factor estimates for the 21-state layout.
    pub accel_scale_factor: [f64; 3],
    /// Fractional gyroscope scale-factor estimates for the 21-state layout.
    pub gyro_scale_factor: [f64; 3],
}

impl InsFilterState {
    /// Build a filter state from a nominal state and covariance.
    pub fn new(
        nominal: NavState,
        layout: ErrorStateLayout,
        covariance: Vec<Vec<f64>>,
    ) -> Result<Self, FusionError> {
        nominal.validate()?;
        validate_covariance_matrix(&covariance, layout.dimension(), "covariance")?;
        Ok(Self {
            nominal,
            error_state: ErrorStateVector::zeros(layout),
            covariance,
            accel_scale_factor: [0.0; 3],
            gyro_scale_factor: [0.0; 3],
        })
    }

    /// Build a filter state from diagonal covariance entries.
    pub fn from_diagonal(
        nominal: NavState,
        layout: ErrorStateLayout,
        diagonal: &[f64],
    ) -> Result<Self, FusionError> {
        layout.validate_len(diagonal.len(), "covariance_diagonal")?;
        let mut covariance = vec![vec![0.0; layout.dimension()]; layout.dimension()];
        for (idx, value) in diagonal.iter().enumerate() {
            validate_finite(*value, "covariance_diagonal").map_err(FusionError::from)?;
            if *value < 0.0 {
                return Err(FusionError::InvalidInput {
                    field: "covariance_diagonal",
                    reason: "must be non-negative",
                });
            }
            covariance[idx][idx] = *value;
        }
        Self::new(nominal, layout, covariance)
    }

    /// Return the selected error-state layout.
    pub const fn layout(&self) -> ErrorStateLayout {
        self.error_state.layout()
    }

    /// Return the state dimension.
    pub fn dimension(&self) -> usize {
        self.error_state.dimension()
    }

    /// Reset the indirect error estimate to zero.
    pub fn reset_error_state(&mut self) {
        self.error_state.reset();
    }

    /// Validate nominal state, error vector, and covariance.
    pub fn validate(&self) -> Result<(), FusionError> {
        self.nominal.validate()?;
        self.error_state.validate()?;
        validate_scale_factors(
            self.layout(),
            self.accel_scale_factor,
            self.gyro_scale_factor,
        )?;
        validate_covariance_matrix(&self.covariance, self.dimension(), "covariance")
    }
}

/// Validate covariance shape, finiteness, symmetry, and PSD.
pub fn validate_covariance_matrix(
    covariance: &[Vec<f64>],
    dimension: usize,
    field: &'static str,
) -> Result<(), FusionError> {
    validate_square_matrix(covariance, dimension, field)?;
    if covariance_is_positive_semidefinite(covariance)? {
        Ok(())
    } else {
        Err(FusionError::NonPositiveSemidefinite { field })
    }
}

/// Test whether a covariance is positive semidefinite under numerical bounds.
#[allow(clippy::needless_range_loop)]
pub fn covariance_is_positive_semidefinite(covariance: &[Vec<f64>]) -> Result<bool, FusionError> {
    let dimension = covariance.len();
    validate_square_matrix(covariance, dimension, "covariance")?;
    let scale = covariance
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let symmetry_tolerance = psd_tolerance(dimension, scale);
    for row in 0..dimension {
        for col in (row + 1)..dimension {
            if (covariance[row][col] - covariance[col][row]).abs() > symmetry_tolerance {
                return Ok(false);
            }
        }
    }
    let matrix = dmatrix_from_rows(covariance);
    let (eigenvectors, eigenvalues) = portable::symmetric_eigen_dynamic(&matrix);
    for (idx, value) in eigenvalues.iter().enumerate() {
        if !value.is_finite() {
            return Ok(false);
        }
        let tolerance = covariance_eigenvalue_tolerance(covariance, &eigenvectors, idx);
        if *value < -tolerance {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Symmetrize and reproject a covariance onto the PSD cone under numerical bounds.
pub fn reproject_covariance_psd(
    covariance: &mut [Vec<f64>],
    field: &'static str,
) -> Result<(), FusionError> {
    let dimension = covariance.len();
    validate_square_matrix(covariance, dimension, field)?;
    symmetrize_in_place(covariance);
    let matrix = dmatrix_from_rows(covariance);
    let (eigenvectors, eigenvalues) = portable::symmetric_eigen_dynamic(&matrix);
    let mut needs_repair = false;
    for (idx, value) in eigenvalues.iter().enumerate() {
        if !value.is_finite() {
            return Err(FusionError::NonPositiveSemidefinite { field });
        }
        let tolerance = covariance_eigenvalue_tolerance(covariance, &eigenvectors, idx);
        if *value < -tolerance {
            return Err(FusionError::NonPositiveSemidefinite { field });
        }
        needs_repair |= *value < 0.0;
    }

    if needs_repair {
        let mut diagonal = DMatrix::<f64>::zeros(dimension, dimension);
        for idx in 0..dimension {
            diagonal[(idx, idx)] = eigenvalues[idx].max(0.0);
        }
        let repaired = portable::product(
            &portable::product(&eigenvectors, &diagonal),
            &eigenvectors.transpose(),
        );
        for row in 0..dimension {
            for col in 0..dimension {
                covariance[row][col] = repaired[(row, col)];
            }
        }
        symmetrize_in_place(covariance);
    }
    validate_covariance_matrix(covariance, dimension, field)
}

pub(crate) fn invalid_input(field: &'static str, reason: &'static str) -> FusionError {
    FusionError::InvalidInput { field, reason }
}

pub(crate) fn validate_positive(value: f64, field: &'static str) -> Result<(), FusionError> {
    validate_finite(value, field).map_err(FusionError::from)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be positive"))
    }
}

pub(crate) fn validate_nonnegative(value: f64, field: &'static str) -> Result<(), FusionError> {
    validate_finite(value, field).map_err(FusionError::from)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be non-negative"))
    }
}

pub(crate) fn validate_finite_slice(
    values: &[f64],
    field: &'static str,
) -> Result<(), FusionError> {
    for value in values {
        validate_finite(*value, field).map_err(FusionError::from)?;
    }
    Ok(())
}

pub(crate) fn validate_scale_factors(
    layout: ErrorStateLayout,
    accel_scale_factor: [f64; 3],
    gyro_scale_factor: [f64; 3],
) -> Result<(), FusionError> {
    for value in accel_scale_factor {
        validate_finite(value, "accel_scale_factor").map_err(FusionError::from)?;
    }
    for value in gyro_scale_factor {
        validate_finite(value, "gyro_scale_factor").map_err(FusionError::from)?;
    }
    if !layout.includes_scale_factors()
        && (accel_scale_factor.iter().any(|value| *value != 0.0)
            || gyro_scale_factor.iter().any(|value| *value != 0.0))
    {
        return Err(invalid_input(
            "scale_factor",
            "requires the 21-state layout",
        ));
    }
    Ok(())
}

pub(crate) fn validate_square_matrix(
    matrix: &[Vec<f64>],
    dimension: usize,
    field: &'static str,
) -> Result<(), FusionError> {
    if matrix.len() != dimension {
        return Err(FusionError::DimensionMismatch {
            field,
            expected: dimension,
            actual: matrix.len(),
        });
    }
    for row in matrix {
        if row.len() != dimension {
            return Err(FusionError::DimensionMismatch {
                field,
                expected: dimension,
                actual: row.len(),
            });
        }
        validate_finite_slice(row, field)?;
    }
    Ok(())
}

pub(crate) fn validate_matrix_cols(
    matrix: &[Vec<f64>],
    cols: usize,
    field: &'static str,
) -> Result<(), FusionError> {
    for row in matrix {
        if row.len() != cols {
            return Err(FusionError::DimensionMismatch {
                field,
                expected: cols,
                actual: row.len(),
            });
        }
        validate_finite_slice(row, field)?;
    }
    Ok(())
}

pub(crate) fn identity(dimension: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; dimension]; dimension];
    for (idx, row) in matrix.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    matrix
}

pub(crate) fn transpose(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FusionError> {
    let rows = matrix.len();
    if rows == 0 {
        return Ok(Vec::new());
    }
    let cols = matrix[0].len();
    validate_matrix_cols(matrix, cols, "matrix")?;
    let mut out = vec![vec![0.0; rows]; cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col][row] = matrix[row][col];
        }
    }
    Ok(out)
}

pub(crate) fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FusionError> {
    if a.is_empty() || b.is_empty() {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    let inner = a[0].len();
    validate_matrix_cols(a, inner, "matrix_a")?;
    if b.len() != inner {
        return Err(FusionError::DimensionMismatch {
            field: "matrix_b",
            expected: inner,
            actual: b.len(),
        });
    }
    let cols = b[0].len();
    validate_matrix_cols(b, cols, "matrix_b")?;
    let mut out = vec![vec![0.0; cols]; a.len()];
    for row in 0..a.len() {
        for col in 0..cols {
            let mut sum = 0.0;
            for k in 0..inner {
                sum += a[row][k] * b[k][col];
            }
            out[row][col] = sum;
        }
    }
    Ok(out)
}

pub(crate) fn matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>, FusionError> {
    if matrix.is_empty() {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    let cols = vector.len();
    validate_matrix_cols(matrix, cols, "matrix")?;
    validate_finite_slice(vector, "vector")?;
    let mut out = vec![0.0; matrix.len()];
    for row in 0..matrix.len() {
        let mut sum = 0.0;
        for (col, value) in vector.iter().enumerate() {
            sum += matrix[row][col] * value;
        }
        out[row] = sum;
    }
    Ok(out)
}

pub(crate) fn matrix_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FusionError> {
    same_shape(a, b, "matrix_add")?;
    let mut out = vec![vec![0.0; a[0].len()]; a.len()];
    for row in 0..a.len() {
        for col in 0..a[0].len() {
            out[row][col] = a[row][col] + b[row][col];
        }
    }
    Ok(out)
}

pub(crate) fn matrix_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, FusionError> {
    same_shape(a, b, "matrix_sub")?;
    let mut out = vec![vec![0.0; a[0].len()]; a.len()];
    for row in 0..a.len() {
        for col in 0..a[0].len() {
            out[row][col] = a[row][col] - b[row][col];
        }
    }
    Ok(out)
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn symmetrize_in_place(matrix: &mut [Vec<f64>]) {
    let dimension = matrix.len();
    for row in 0..dimension {
        for col in (row + 1)..dimension {
            let value = 0.5 * (matrix[row][col] + matrix[col][row]);
            matrix[row][col] = value;
            matrix[col][row] = value;
        }
    }
}

pub(crate) fn solve_spd(
    matrix: &[Vec<f64>],
    rhs: &[f64],
    scratch: &mut crate::astro::math::linear::FlatCholeskySolveScratch,
) -> Result<Vec<f64>, FusionError> {
    validate_square_matrix(matrix, rhs.len(), "spd_matrix")?;
    validate_finite_slice(rhs, "spd_rhs")?;
    let flat = flatten(matrix);
    crate::astro::math::linear::solve_flat_normal_square_root_into(&flat, rhs, scratch)
        .map(<[f64]>::to_vec)
        .ok_or(FusionError::SingularInnovation)
}

pub(crate) fn flatten(matrix: &[Vec<f64>]) -> Vec<f64> {
    let rows = matrix.len();
    let cols = if rows == 0 { 0 } else { matrix[0].len() };
    let mut out = Vec::with_capacity(rows * cols);
    for row in matrix {
        out.extend(row);
    }
    out
}

pub(crate) fn dmatrix_from_rows(rows: &[Vec<f64>]) -> DMatrix<f64> {
    let nrows = rows.len();
    let ncols = if nrows == 0 { 0 } else { rows[0].len() };
    DMatrix::from_row_slice(nrows, ncols, &flatten(rows))
}

fn same_shape(a: &[Vec<f64>], b: &[Vec<f64>], field: &'static str) -> Result<(), FusionError> {
    if a.is_empty() || b.is_empty() {
        return Err(invalid_input(field, "must not be empty"));
    }
    validate_matrix_cols(a, a[0].len(), field)?;
    validate_matrix_cols(b, b[0].len(), field)?;
    if a.len() != b.len() {
        return Err(FusionError::DimensionMismatch {
            field,
            expected: a.len(),
            actual: b.len(),
        });
    }
    if a[0].len() != b[0].len() {
        return Err(FusionError::DimensionMismatch {
            field,
            expected: a[0].len(),
            actual: b[0].len(),
        });
    }
    Ok(())
}

fn psd_tolerance(dimension: usize, scale: f64) -> f64 {
    let dimension_scale = dimension.max(1) as f64;
    PSD_REL_TOLERANCE * dimension_scale * scale
}

pub(crate) fn covariance_eigenvalue_tolerance(
    covariance: &[Vec<f64>],
    eigenvectors: &DMatrix<f64>,
    mode: usize,
) -> f64 {
    let dimension = covariance.len();
    let mut scale = 0.0_f64;
    for row in 0..dimension {
        let row_weight = eigenvectors[(row, mode)].abs();
        for col in 0..dimension {
            scale += row_weight * covariance[row][col].abs() * eigenvectors[(col, mode)].abs();
        }
    }
    psd_tolerance(dimension, scale)
}

#[cfg(test)]
mod tests {
    //! Provenance: PSD validation tests lock the fusion safety contract that
    //! invalid covariance input remains flagged instead of being repaired into a
    //! false covariance.

    use super::*;

    #[test]
    fn symmetry_tolerance_is_matrix_relative_for_tiny_off_diagonals() {
        // A rotated block-diagonal covariance: off-diagonal pairs are float dust
        // around zero whose absolute difference dwarfs their own magnitude but is
        // negligible against the matrix scale. Element-relative symmetry
        // tolerances reject this PSD matrix; the tolerance must be
        // matrix-scale-relative.
        let mut covariance = vec![vec![0.0; 6]; 6];
        for (idx, variance) in [2.25, 2.25, 9.0, 0.0025, 0.0025, 0.0025].iter().enumerate() {
            covariance[idx][idx] = *variance;
        }
        covariance[0][3] = 1.0e-19;
        covariance[3][0] = -3.0e-19;
        assert!(covariance_is_positive_semidefinite(&covariance).expect("validate"));
    }

    #[test]
    fn zero_covariance_is_psd() {
        let covariance = vec![vec![0.0]];
        assert!(covariance_is_positive_semidefinite(&covariance).expect("psd"));
    }

    #[test]
    fn tiny_negative_variance_is_rejected_not_repaired() {
        let covariance = vec![vec![-1.0e-15]];
        assert!(!covariance_is_positive_semidefinite(&covariance).expect("psd"));

        let mut covariance = covariance;
        let err = reproject_covariance_psd(&mut covariance, "covariance")
            .expect_err("negative variance must remain flagged");
        assert!(matches!(
            err,
            FusionError::NonPositiveSemidefinite {
                field: "covariance"
            }
        ));
        assert_eq!(covariance[0][0].to_bits(), (-1.0e-15_f64).to_bits());
    }

    #[test]
    fn unrelated_large_variance_does_not_hide_negative_mode() {
        let covariance = vec![vec![1.0e16, 0.0], vec![0.0, -100.0]];
        assert!(!covariance_is_positive_semidefinite(&covariance).expect("psd"));

        let mut covariance = covariance;
        let err = reproject_covariance_psd(&mut covariance, "covariance")
            .expect_err("negative mode must remain flagged");
        assert!(matches!(
            err,
            FusionError::NonPositiveSemidefinite {
                field: "covariance"
            }
        ));
        assert_eq!(covariance[1][1].to_bits(), (-100.0_f64).to_bits());
    }
}
