//! Position-covariance modeling for conjunction and orbit analysis.
//!
//! Owns the authoritative RTN->ECI frame transform of a 3x3 position
//! covariance, typed 6x6 state covariance propagation, and the symmetric
//! positive-semidefinite (PSD) validation used to reject ill-formed
//! covariances. The sidereon Elixir binding is a thin marshaling and
//! structural-validation layer over this module; no frame or PSD formula lives
//! there.
//!
//! The covariance is transformed but never rescaled here, so it carries the
//! squared units of whatever position vectors it was formed from.

use crate::astro::math::mat3::{self, Mat3};
use crate::astro::math::portable;
use crate::astro::math::vec3;
use crate::astro::state::CartesianState;
use crate::validate;
use nalgebra::SMatrix;

/// Position magnitudes below this are treated as a degenerate (zero) position
/// vector, for which the RTN frame is undefined.
const ZERO_POSITION_EPS: f64 = 1.0e-30;
/// Orbit-normal magnitudes below this mean position and velocity are parallel,
/// so the RTN frame normal (and thus the frame) is undefined.
const PARALLEL_RV_EPS: f64 = 1.0e-30;
/// Diagonal covariance entries are allowed to dip to this (negative) bound
/// before the PSD check rejects them, absorbing float round-off.
const PSD_DIAGONAL_EPS: f64 = 1.0e-15;
/// Second- and third-order principal minors are allowed to dip to this
/// (negative) bound before the PSD check rejects them.
const PSD_MINOR_EPS: f64 = 1.0e-12;
/// Off-diagonal pairs differing by more than this are treated as asymmetric.
const SYMMETRY_EPS: f64 = 1.0e-12;
/// Relative off-diagonal tolerance for 6x6 covariance symmetry checks.
const SYMMETRY_REL_EPS6: f64 = 1.0e-12;
/// Eigenvalues below this relative bound are treated as negative for 6x6 PSD.
const PSD6_EIGEN_REL_EPS: f64 = 1.0e-10;
/// Relative eigenvalue floor used only before interpolation Cholesky factoring.
const INTERPOLATION_EIGEN_REL_FLOOR: f64 = 1.0e-9;

/// Row-major 6x6 covariance for state vector `[r_x, r_y, r_z, v_x, v_y, v_z]`.
pub type Mat6 = [[f64; 6]; 6];

/// Typed 6x6 state covariance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Covariance6 {
    matrix: Mat6,
}

/// Reason a 6x6 state covariance was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Covariance6Error {
    /// At least one matrix entry was NaN or infinite.
    NonFinite,
    /// The matrix was not symmetric within the covariance tolerance.
    Asymmetric,
    /// The symmetric matrix was not positive semidefinite.
    NotPositiveSemidefinite,
    /// A PSD interpolation endpoint could not be Cholesky-factorized.
    NotFactorizable,
    /// The interpolation parameter was non-finite or outside `[0, 1]`.
    InvalidInterpolationParameter,
}

impl Covariance6 {
    /// Validate and wrap a row-major 6x6 state covariance.
    pub fn try_from_matrix(matrix: Mat6) -> Result<Self, Covariance6Error> {
        if !finite6(&matrix) {
            return Err(Covariance6Error::NonFinite);
        }
        if !symmetric6(&matrix) {
            return Err(Covariance6Error::Asymmetric);
        }
        if !positive_semidefinite6(&matrix) {
            return Err(Covariance6Error::NotPositiveSemidefinite);
        }
        Ok(Self { matrix })
    }

    /// Build a diagonal state covariance from six variances.
    pub fn from_diagonal(diagonal: [f64; 6]) -> Result<Self, Covariance6Error> {
        let mut matrix = [[0.0_f64; 6]; 6];
        for (idx, value) in diagonal.into_iter().enumerate() {
            matrix[idx][idx] = value;
        }
        Self::try_from_matrix(matrix)
    }

    /// Wrap a matrix without validation.
    ///
    /// Intended for trusted fixtures; prefer [`Self::try_from_matrix`] for
    /// caller data.
    pub const fn from_matrix_unchecked(matrix: Mat6) -> Self {
        Self { matrix }
    }

    /// Borrow the row-major 6x6 matrix.
    pub const fn as_matrix(&self) -> &Mat6 {
        &self.matrix
    }

    /// Consume this covariance and return its row-major 6x6 matrix.
    pub const fn into_matrix(self) -> Mat6 {
        self.matrix
    }

    /// Extract the 3x3 position covariance block.
    pub fn position_covariance_km2(&self) -> Mat3 {
        [
            [self.matrix[0][0], self.matrix[0][1], self.matrix[0][2]],
            [self.matrix[1][0], self.matrix[1][1], self.matrix[1][2]],
            [self.matrix[2][0], self.matrix[2][1], self.matrix[2][2]],
        ]
    }

    /// Whether this covariance is symmetric within the covariance tolerance.
    pub fn is_symmetric(&self) -> bool {
        symmetric6(&self.matrix)
    }

    /// Whether this covariance is positive semidefinite within tolerance.
    pub fn is_positive_semidefinite(&self) -> bool {
        positive_semidefinite6(&self.matrix)
    }

    /// Propagate this covariance through a state-transition matrix:
    /// `P_f = Phi * P_0 * Phi^T`.
    #[allow(clippy::needless_range_loop)]
    pub fn propagate_with_stm(&self, stm: &Mat6) -> Result<Self, Covariance6Error> {
        if !finite6(stm) {
            return Err(Covariance6Error::NonFinite);
        }

        let mut temp = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    temp[i][j] += stm[i][k] * self.matrix[k][j];
                }
            }
        }

        let mut propagated = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    propagated[i][j] += temp[i][k] * stm[j][k];
                }
            }
        }
        symmetrize6(&mut propagated);

        Self::try_from_matrix(propagated)
    }
}

/// Transform a 6x6 inertial state covariance to RTN at `state`.
///
/// This uses the kinematic covariance convention: the same instantaneous RTN
/// rotation is applied to position and velocity rows, without rotating-frame
/// velocity terms.
pub fn eci_to_rtn_covariance6(
    covariance: &Covariance6,
    state: &CartesianState,
) -> Result<Covariance6, RtnFrameError> {
    let rot = rtn_to_eci_rotation(state.position_array(), state.velocity_array())?;
    let rot_t = mat3::inline_tr(&rot);
    covariance_congruence6(covariance, &rot_t)
}

/// Transform a 6x6 RTN state covariance to inertial axes at `state`.
///
/// This is the inverse of [`eci_to_rtn_covariance6`] under the same kinematic
/// covariance convention.
pub fn rtn_to_eci_covariance6(
    covariance: &Covariance6,
    state: &CartesianState,
) -> Result<Covariance6, RtnFrameError> {
    let rot = rtn_to_eci_rotation(state.position_array(), state.velocity_array())?;
    covariance_congruence6(covariance, &rot)
}

/// Convert a km-based 6x6 state covariance to m-based covariance units.
///
/// Every entry scales by 1e6 because position and velocity components both
/// scale by 1e3 and covariance is quadratic in the state.
pub fn covariance6_km_to_m(covariance: &Covariance6) -> Result<Covariance6, Covariance6Error> {
    scale_covariance6(covariance, 1.0e6)
}

/// Convert an m-based 6x6 state covariance to km-based covariance units.
pub fn covariance6_m_to_km(covariance: &Covariance6) -> Result<Covariance6, Covariance6Error> {
    scale_covariance6(covariance, 1.0e-6)
}

/// PSD-safe interpolation between two same-frame 6x6 covariances.
///
/// The interpolation follows the Log-Cholesky geodesic: strictly lower
/// Cholesky entries are linearly blended, diagonal entries are blended in log
/// space, and the covariance is reconstructed as `L * L^T`. Endpoints are
/// returned bit-for-bit. Singular but non-zero validated endpoints are nudged
/// through `eigen_floor6` before factorization; an all-zero endpoint is
/// rejected because the logarithmic diagonal is undefined.
#[allow(clippy::needless_range_loop)]
pub fn interpolate_covariance_psd(
    a: &Covariance6,
    b: &Covariance6,
    u: f64,
) -> Result<Covariance6, Covariance6Error> {
    if !u.is_finite() || !(0.0..=1.0).contains(&u) {
        return Err(Covariance6Error::InvalidInterpolationParameter);
    }
    if u == 0.0 {
        return Ok(*a);
    }
    if u == 1.0 {
        return Ok(*b);
    }
    if is_all_zero6(a.as_matrix()) || is_all_zero6(b.as_matrix()) {
        return Err(Covariance6Error::NotFactorizable);
    }

    let la = cholesky_lower_with_floor(a.as_matrix())?;
    let lb = cholesky_lower_with_floor(b.as_matrix())?;
    let mut l = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            l[i][j] = if i == j {
                libm::exp(libm::log(la[i][j]) * (1.0 - u) + libm::log(lb[i][j]) * u)
            } else {
                la[i][j] * (1.0 - u) + lb[i][j] * u
            };
        }
    }

    let mut interpolated = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let mut value = 0.0_f64;
            for k in 0..=j {
                value += l[i][k] * l[j][k];
            }
            interpolated[i][j] = value;
            interpolated[j][i] = value;
        }
    }
    Covariance6::try_from_matrix(interpolated)
}

/// Reason an RTN->ECI transform could not be built from an orbit state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtnFrameError {
    /// A numeric input was non-finite.
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    /// The position vector is effectively zero.
    ZeroPosition,
    /// Position and velocity are parallel, leaving the orbit normal undefined.
    ParallelPositionVelocity,
}

impl RtnFrameError {
    /// Message string matching the historical sidereon error verbatim, so the
    /// thin Elixir binding preserves its public `{:error, reason}` shapes.
    pub fn message(self) -> &'static str {
        match self {
            RtnFrameError::InvalidInput { .. } => "invalid input",
            RtnFrameError::ZeroPosition => "zero position vector",
            RtnFrameError::ParallelPositionVelocity => "position and velocity are parallel",
        }
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> RtnFrameError {
    RtnFrameError::InvalidInput { field, reason }
}

fn validate_vec3(field: &'static str, values: [f64; 3]) -> Result<(), RtnFrameError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_input(field, "components must be finite"))
    }
}

fn validate_covariance(field: &'static str, values: &Mat3) -> Result<(), RtnFrameError> {
    validate::validate_covariance_psd(values, field).map_err(|error| match error {
        validate::FieldError::NonFinite { field } => {
            invalid_input(field, "components must be finite")
        }
        validate::FieldError::NotPositive { field } => invalid_input(field, "not positive"),
        validate::FieldError::Negative { field } => invalid_input(field, "negative"),
        validate::FieldError::OutOfRange { field, .. } => invalid_input(field, "out of range"),
        validate::FieldError::Missing { field }
        | validate::FieldError::FloatParse { field, .. }
        | validate::FieldError::IntParse { field, .. }
        | validate::FieldError::InvalidCivilDate { field, .. }
        | validate::FieldError::InvalidCivilTime { field, .. } => invalid_input(field, "invalid"),
    })
}

fn validate_mat3_finite(field: &'static str, values: &Mat3) -> Result<(), RtnFrameError> {
    for row in values {
        validate_vec3(field, *row)?;
    }
    Ok(())
}

/// Build the RTN->ECI rotation whose columns are the radial, transverse, and
/// normal unit vectors of the orbit state `(r, v)`.
///
/// Operation order (magnitude before normalize, division not reciprocal
/// multiply, cross-product component order) is fixed to reproduce the prior
/// Elixir reference bit-for-bit.
pub fn rtn_to_eci_rotation(r: [f64; 3], v: [f64; 3]) -> Result<Mat3, RtnFrameError> {
    validate_vec3("position", r)?;
    validate_vec3("velocity", v)?;
    if vec3::norm3(r) < ZERO_POSITION_EPS {
        return Err(RtnFrameError::ZeroPosition);
    }
    let r_hat = vec3::unit3_ref_unchecked(&r);
    let h = vec3::cross3(r, v);
    if vec3::norm3(h) < PARALLEL_RV_EPS {
        return Err(RtnFrameError::ParallelPositionVelocity);
    }
    let n_hat = vec3::unit3_ref_unchecked(&h);
    let t_hat = vec3::cross3(n_hat, r_hat);
    Ok([
        [r_hat[0], t_hat[0], n_hat[0]],
        [r_hat[1], t_hat[1], n_hat[1]],
        [r_hat[2], t_hat[2], n_hat[2]],
    ])
}

/// Transform a 3x3 RTN position covariance to ECI: `C_eci = R * C_rtn * R^T`.
///
/// The triple product materialises the intermediate `R * C_rtn` and applies
/// `R^T` in a second multiply (left-to-right `k` summation), matching the
/// chained Elixir `mat_mul` reduction order rather than a fused Kahan product.
pub fn rtn_to_eci(cov_rtn: &Mat3, r: [f64; 3], v: [f64; 3]) -> Result<Mat3, RtnFrameError> {
    validate_covariance("cov_rtn", cov_rtn)?;
    let rot = rtn_to_eci_rotation(r, v)?;
    let rot_t = mat3::inline_tr(&rot);
    let cov_eci = mat3::inline_rxr(&mat3::inline_rxr(&rot, cov_rtn), &rot_t);
    validate_mat3_finite("cov_eci", &cov_eci)?;
    Ok(cov_eci)
}

/// Whether a 3x3 matrix is symmetric within `SYMMETRY_EPS`.
pub fn symmetric(m: &Mat3) -> bool {
    (m[0][1] - m[1][0]).abs() < SYMMETRY_EPS
        && (m[0][2] - m[2][0]).abs() < SYMMETRY_EPS
        && (m[1][2] - m[2][1]).abs() < SYMMETRY_EPS
}

/// Determinant of a 3x3 matrix via cofactor expansion along the first row,
/// matching the Elixir reference operation order.
fn det3x3(m: &Mat3) -> f64 {
    let (a, b, c) = (m[0][0], m[0][1], m[0][2]);
    let (d, e, f) = (m[1][0], m[1][1], m[1][2]);
    let (g, h, i) = (m[2][0], m[2][1], m[2][2]);
    a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
}

/// Whether a symmetric 3x3 matrix is positive semidefinite by Sylvester's
/// criterion: every leading-and-trailing principal minor is non-negative
/// within tolerance. A non-symmetric matrix is rejected.
pub fn positive_semidefinite(m: &Mat3) -> bool {
    if !symmetric(m) {
        return false;
    }

    let m11 = m[0][0];
    let m22 = m[1][1];
    let m33 = m[2][2];
    let m12 = m[0][1];
    let m13 = m[0][2];
    let m23 = m[1][2];

    let det12 = m11 * m22 - m12 * m12;
    let det13 = m11 * m33 - m13 * m13;
    let det23 = m22 * m33 - m23 * m23;
    let det123 = det3x3(m);

    m11 >= -PSD_DIAGONAL_EPS
        && m22 >= -PSD_DIAGONAL_EPS
        && m33 >= -PSD_DIAGONAL_EPS
        && det12 >= -PSD_MINOR_EPS
        && det13 >= -PSD_MINOR_EPS
        && det23 >= -PSD_MINOR_EPS
        && det123 >= -PSD_MINOR_EPS
}

pub(crate) fn finite6(m: &Mat6) -> bool {
    m.iter().flatten().all(|value| value.is_finite())
}

fn covariance_scale6(m: &Mat6) -> f64 {
    (0..6).fold(0.0_f64, |scale, idx| scale.max(m[idx][idx].abs()))
}

#[allow(clippy::needless_range_loop)]
fn symmetric6(m: &Mat6) -> bool {
    let tolerance = SYMMETRY_REL_EPS6 * covariance_scale6(m);
    for i in 0..6 {
        for j in (i + 1)..6 {
            if (m[i][j] - m[j][i]).abs() > tolerance {
                return false;
            }
        }
    }
    true
}

fn positive_semidefinite6(m: &Mat6) -> bool {
    if !finite6(m) || !symmetric6(m) {
        return false;
    }

    let matrix = SMatrix::<f64, 6, 6>::from_fn(|i, j| m[i][j]);
    let (_, eigenvalues) = portable::symmetric_eigen6(&matrix);
    let scale = covariance_scale6(m);
    let floor = -PSD6_EIGEN_REL_EPS * scale;
    eigenvalues.iter().all(|&lambda| lambda >= floor)
}

/// Clamp small eigenvalues of a symmetric 6x6 matrix to a relative floor.
///
/// This is used only to make marginal PSD interpolation endpoints strictly
/// factorizable. It is not a propagation repair path.
pub(crate) fn eigen_floor6(matrix: &Mat6, rel_floor: f64) -> Mat6 {
    let m = SMatrix::<f64, 6, 6>::from_fn(|i, j| matrix[i][j]);
    let (eigenvectors, eigenvalues) = portable::symmetric_eigen6(&m);
    let scale = covariance_scale6(matrix);
    let floor = rel_floor.max(0.0) * scale;
    let mut diagonal = SMatrix::<f64, 6, 6>::zeros();
    for i in 0..6 {
        diagonal[(i, i)] = eigenvalues[i].max(floor);
    }
    let floored = portable::product_fixed(
        &portable::product_fixed(&eigenvectors, &diagonal),
        &eigenvectors.transpose(),
    );
    let mut out = mat6_from_smatrix(&floored);
    symmetrize6(&mut out);
    out
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn symmetrize6(m: &mut Mat6) {
    for i in 0..6 {
        for j in (i + 1)..6 {
            let value = 0.5 * (m[i][j] + m[j][i]);
            m[i][j] = value;
            m[j][i] = value;
        }
    }
}

fn is_all_zero6(m: &Mat6) -> bool {
    m.iter().flatten().all(|value| *value == 0.0)
}

fn mat6_from_smatrix(matrix: &SMatrix<f64, 6, 6>) -> Mat6 {
    let mut out = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            out[i][j] = matrix[(i, j)];
        }
    }
    out
}

fn cholesky_lower(matrix: &Mat6) -> Option<Mat6> {
    let m = SMatrix::<f64, 6, 6>::from_fn(|i, j| matrix[i][j]);
    portable::cholesky_lower(&m).map(|lower| mat6_from_smatrix(&lower))
}

fn cholesky_lower_with_floor(matrix: &Mat6) -> Result<Mat6, Covariance6Error> {
    if let Some(lower) = cholesky_lower(matrix) {
        return Ok(lower);
    }
    let floored = eigen_floor6(matrix, INTERPOLATION_EIGEN_REL_FLOOR);
    cholesky_lower(&floored).ok_or(Covariance6Error::NotFactorizable)
}

#[allow(clippy::needless_range_loop)]
pub(crate) fn covariance_congruence6_checked(
    covariance: &Covariance6,
    rotation: &Mat3,
) -> Result<Covariance6, Covariance6Error> {
    let matrix = covariance.as_matrix();
    let mut block_rotation = [[0.0_f64; 6]; 6];
    for i in 0..3 {
        for j in 0..3 {
            block_rotation[i][j] = rotation[i][j];
            block_rotation[i + 3][j + 3] = rotation[i][j];
        }
    }

    let mut temp = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                temp[i][j] += block_rotation[i][k] * matrix[k][j];
            }
        }
    }

    let mut transformed = [[0.0_f64; 6]; 6];
    for i in 0..6 {
        for j in 0..6 {
            for k in 0..6 {
                transformed[i][j] += temp[i][k] * block_rotation[j][k];
            }
        }
    }
    symmetrize6(&mut transformed);
    Covariance6::try_from_matrix(transformed)
}

fn covariance_congruence6(
    covariance: &Covariance6,
    rotation: &Mat3,
) -> Result<Covariance6, RtnFrameError> {
    covariance_congruence6_checked(covariance, rotation).map_err(covariance_error_to_rtn_error)
}

fn covariance_error_to_rtn_error(error: Covariance6Error) -> RtnFrameError {
    match error {
        Covariance6Error::NonFinite => invalid_input("covariance", "components must be finite"),
        Covariance6Error::Asymmetric => invalid_input("covariance", "not symmetric"),
        Covariance6Error::NotPositiveSemidefinite
        | Covariance6Error::NotFactorizable
        | Covariance6Error::InvalidInterpolationParameter => {
            invalid_input("covariance", "not positive")
        }
    }
}

fn scale_covariance6(
    covariance: &Covariance6,
    scale: f64,
) -> Result<Covariance6, Covariance6Error> {
    let mut scaled = *covariance.as_matrix();
    for row in &mut scaled {
        for value in row {
            *value *= scale;
        }
    }
    Covariance6::try_from_matrix(scaled)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Frozen ECI bits from the prior Elixir `Sidereon.Covariance.rtn_to_eci`
    /// reference for `r = (7000.123, 1234.5, -250.7)`,
    /// `v = (1.2, 7.4, 0.3)`, and the non-diagonal RTN covariance below.
    /// Row-major; proves cross-language 0-ULP parity, including the last-ULP
    /// off-diagonal asymmetry the chained multiply produces.
    const RTN_TO_ECI_GOLDEN_BITS: [u64; 9] = [
        0x4010077f74cce7ac,
        0xbfd92b0043adb450,
        0x3fe26dc422b0767a,
        0xbfd92b0043adb44a,
        0x402207fb1ad4c218,
        0xbfb9ef5fd1874930,
        0x3fe26dc422b0767a,
        0xbfb9ef5fd1874930,
        0x402ff4452ac4ca0f,
    ];

    #[test]
    fn rtn_to_eci_matches_frozen_elixir_bits() {
        let r = [7000.123, 1234.5, -250.7];
        let v = [1.2, 7.4, 0.3];
        let cov_rtn = [[4.0, 0.5, 0.1], [0.5, 9.0, 0.2], [0.1, 0.2, 16.0]];

        let eci = rtn_to_eci(&cov_rtn, r, v).expect("non-degenerate state");

        let mut flat = [0u64; 9];
        for (idx, slot) in flat.iter_mut().enumerate() {
            *slot = eci[idx / 3][idx % 3].to_bits();
        }
        assert_eq!(flat, RTN_TO_ECI_GOLDEN_BITS);
    }

    #[test]
    fn rtn_to_eci_aligned_state_is_exactly_the_rtn_diagonal() {
        // r along +X, v along +Y -> RTN axes coincide with ECI, so the
        // transform is the identity and the diagonal is reproduced exactly.
        let r = [7000.0, 0.0, 0.0];
        let v = [0.0, 7.5, 0.0];
        let cov_rtn = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]];

        let eci = rtn_to_eci(&cov_rtn, r, v).expect("non-degenerate state");

        assert_eq!(eci[0][0].to_bits(), 1.0_f64.to_bits());
        assert_eq!(eci[1][1].to_bits(), 2.0_f64.to_bits());
        assert_eq!(eci[2][2].to_bits(), 3.0_f64.to_bits());
    }

    #[test]
    fn rtn_to_eci_rejects_zero_position() {
        let err = rtn_to_eci(&identity(), [0.0, 0.0, 0.0], [0.0, 7.5, 0.0]).unwrap_err();
        assert_eq!(err, RtnFrameError::ZeroPosition);
        assert_eq!(err.message(), "zero position vector");
    }

    #[test]
    fn rtn_to_eci_rejects_parallel_position_velocity() {
        let err = rtn_to_eci(&identity(), [7000.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap_err();
        assert_eq!(err, RtnFrameError::ParallelPositionVelocity);
        assert_eq!(err.message(), "position and velocity are parallel");
    }

    #[test]
    fn rtn_to_eci_rejects_nonfinite_geometry_and_covariance() {
        let err = rtn_to_eci(&identity(), [7000.0, f64::NAN, 0.0], [0.0, 7.5, 0.0]).unwrap_err();
        assert_eq!(
            err,
            RtnFrameError::InvalidInput {
                field: "position",
                reason: "components must be finite",
            }
        );

        let err =
            rtn_to_eci(&identity(), [7000.0, 0.0, 0.0], [0.0, f64::INFINITY, 0.0]).unwrap_err();
        assert_eq!(
            err,
            RtnFrameError::InvalidInput {
                field: "velocity",
                reason: "components must be finite",
            }
        );

        let mut cov = identity();
        cov[2][1] = f64::NEG_INFINITY;
        let err = rtn_to_eci(&cov, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]).unwrap_err();
        assert_eq!(
            err,
            RtnFrameError::InvalidInput {
                field: "cov_rtn",
                reason: "components must be finite",
            }
        );
    }

    #[test]
    fn rtn_to_eci_rejects_invalid_covariance_geometry() {
        let r = [7000.0, 0.0, 0.0];
        let v = [0.0, 7.5, 0.0];

        let mut negative_variance = identity();
        negative_variance[0][0] = -1.0;
        let err = rtn_to_eci(&negative_variance, r, v).unwrap_err();
        assert_eq!(
            err,
            RtnFrameError::InvalidInput {
                field: "cov_rtn",
                reason: "not positive",
            }
        );

        let asymmetric = [[1.0, 0.5, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let err = rtn_to_eci(&asymmetric, r, v).unwrap_err();
        assert_eq!(
            err,
            RtnFrameError::InvalidInput {
                field: "cov_rtn",
                reason: "not positive",
            }
        );

        let indefinite = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let err = rtn_to_eci(&indefinite, r, v).unwrap_err();
        assert_eq!(
            err,
            RtnFrameError::InvalidInput {
                field: "cov_rtn",
                reason: "not positive",
            }
        );
    }

    #[test]
    fn positive_semidefinite_accepts_identity_rejects_negative_and_asymmetric() {
        assert!(positive_semidefinite(&identity()));

        let negative_diag = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(!positive_semidefinite(&negative_diag));

        let asymmetric = [[1.0, 0.5, 0.0], [0.4, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(!symmetric(&asymmetric));
        assert!(!positive_semidefinite(&asymmetric));
    }

    #[test]
    fn positive_semidefinite_rejects_symmetric_indefinite_matrix() {
        // Symmetric but the 2x2 minor m11*m22 - m12^2 = 1 - 4 < 0.
        let indefinite = [[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(symmetric(&indefinite));
        assert!(!positive_semidefinite(&indefinite));
    }

    #[test]
    fn covariance6_accepts_diagonal_and_rejects_bad_matrices() {
        let covariance =
            Covariance6::from_diagonal([1.0, 2.0, 3.0, 1.0e-6, 2.0e-6, 3.0e-6]).unwrap();
        assert!(covariance.is_symmetric());
        assert!(covariance.is_positive_semidefinite());

        let mut asymmetric = *covariance.as_matrix();
        asymmetric[0][1] = 1.0e-3;
        assert_eq!(
            Covariance6::try_from_matrix(asymmetric),
            Err(Covariance6Error::Asymmetric)
        );

        let mut indefinite = *covariance.as_matrix();
        indefinite[5][5] = -1.0;
        assert_eq!(
            Covariance6::try_from_matrix(indefinite),
            Err(Covariance6Error::NotPositiveSemidefinite)
        );
    }

    #[test]
    fn covariance6_scales_psd_tolerance_to_covariance_magnitude() {
        let mut large = [[0.0_f64; 6]; 6];
        for (idx, row) in large.iter_mut().enumerate() {
            row[idx] = 1.0e18;
        }
        large[0][1] = 2.5e17;
        large[1][0] = 2.5e17 + 1.0e3;

        let covariance = Covariance6::try_from_matrix(large).expect("large PSD covariance");
        assert!(covariance.is_symmetric());
        assert!(covariance.is_positive_semidefinite());

        let mut indefinite = large;
        indefinite[2][2] = -1.0e9;
        assert_eq!(
            Covariance6::try_from_matrix(indefinite),
            Err(Covariance6Error::NotPositiveSemidefinite)
        );

        let small =
            Covariance6::from_diagonal([1.0e-18, 2.0e-18, 3.0e-18, 4.0e-18, 5.0e-18, 6.0e-18])
                .expect("small PSD covariance");
        assert!(small.is_symmetric());
        assert!(small.is_positive_semidefinite());

        let mut small_indefinite = *small.as_matrix();
        small_indefinite[0][0] = -1.0e-20;
        assert_eq!(
            Covariance6::try_from_matrix(small_indefinite),
            Err(Covariance6Error::NotPositiveSemidefinite)
        );
    }

    #[test]
    fn covariance6_rtn_round_trip_recovers_input() {
        let state = CartesianState::new(100.0, [7000.0, 100.0, 20.0], [-0.1, 7.5, 0.3]);
        let covariance = Covariance6::try_from_matrix([
            [4.0, 0.2, 0.1, 1.0e-5, 2.0e-5, 3.0e-5],
            [0.2, 9.0, 0.3, 4.0e-5, 5.0e-5, 6.0e-5],
            [0.1, 0.3, 16.0, 7.0e-5, 8.0e-5, 9.0e-5],
            [1.0e-5, 4.0e-5, 7.0e-5, 1.0e-4, 1.0e-5, 2.0e-5],
            [2.0e-5, 5.0e-5, 8.0e-5, 1.0e-5, 2.0e-4, 3.0e-5],
            [3.0e-5, 6.0e-5, 9.0e-5, 2.0e-5, 3.0e-5, 3.0e-4],
        ])
        .expect("SPD covariance");

        let rtn = eci_to_rtn_covariance6(&covariance, &state).expect("ECI to RTN");
        let eci = rtn_to_eci_covariance6(&rtn, &state).expect("RTN to ECI");

        for i in 0..6 {
            for j in 0..6 {
                let expected = covariance.as_matrix()[i][j];
                let actual = eci.as_matrix()[i][j];
                let tolerance = 1.0e-12 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "entry [{i}][{j}] expected {expected}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn covariance6_position_block_matches_existing_rtn_to_eci() {
        let state = CartesianState::new(0.0, [7000.123, 1234.5, -250.7], [1.2, 7.4, 0.3]);
        let cov_rtn = [[4.0, 0.5, 0.1], [0.5, 9.0, 0.2], [0.1, 0.2, 16.0]];
        let full = Covariance6::from_diagonal([4.0, 9.0, 16.0, 1.0, 1.0, 1.0]).unwrap();
        let mut matrix = *full.as_matrix();
        for i in 0..3 {
            for j in 0..3 {
                matrix[i][j] = cov_rtn[i][j];
            }
        }
        let full = Covariance6::try_from_matrix(matrix).unwrap();

        let eci3 = rtn_to_eci(&cov_rtn, state.position_array(), state.velocity_array()).unwrap();
        let eci6 = rtn_to_eci_covariance6(&full, &state).unwrap();

        // The 6x6 path symmetrizes by spec after congruence, while the legacy
        // 3x3 helper preserves its frozen multiply asymmetry for binding
        // parity. Pin the deviation explicitly instead of widening silently.
        for (i, row) in eci3.iter().enumerate() {
            for (j, expected) in row.iter().enumerate() {
                assert!((eci6.as_matrix()[i][j] - expected).abs() <= 1.0e-14);
            }
        }
    }

    #[test]
    fn covariance6_unit_scaling_round_trips() {
        let covariance =
            Covariance6::from_diagonal([1.0, 2.0, 3.0, 1.0e-6, 2.0e-6, 3.0e-6]).unwrap();

        let meters = covariance6_km_to_m(&covariance).expect("km to m");
        assert_eq!(meters.as_matrix()[0][0].to_bits(), 1.0e6_f64.to_bits());
        assert_eq!(meters.as_matrix()[3][3].to_bits(), 1.0_f64.to_bits());

        let kilometers = covariance6_m_to_km(&meters).expect("m to km");
        assert_eq!(kilometers, covariance);
    }

    #[test]
    fn covariance6_interpolation_rejects_invalid_parameters_and_zero_endpoint() {
        let a = Covariance6::from_diagonal([1.0, 2.0, 3.0, 1.0e-6, 2.0e-6, 3.0e-6]).unwrap();
        let b = Covariance6::from_diagonal([4.0, 5.0, 6.0, 4.0e-6, 5.0e-6, 6.0e-6]).unwrap();

        assert_eq!(interpolate_covariance_psd(&a, &b, 0.0).unwrap(), a);
        assert_eq!(interpolate_covariance_psd(&a, &b, 1.0).unwrap(), b);
        for u in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                interpolate_covariance_psd(&a, &b, u),
                Err(Covariance6Error::InvalidInterpolationParameter)
            );
        }

        let zero = Covariance6::from_diagonal([0.0; 6]).unwrap();
        assert_eq!(
            interpolate_covariance_psd(&zero, &b, 0.5),
            Err(Covariance6Error::NotFactorizable)
        );
    }

    #[test]
    fn covariance6_interpolation_floors_singular_endpoint() {
        let singular = Covariance6::from_diagonal([1.0, 2.0, 3.0, 0.0, 5.0e-6, 6.0e-6]).unwrap();
        let full_rank =
            Covariance6::from_diagonal([1.5, 2.5, 3.5, 1.0e-6, 5.5e-6, 6.5e-6]).unwrap();

        let interpolated = interpolate_covariance_psd(&singular, &full_rank, 0.5)
            .expect("floored singular endpoint interpolates");

        assert!(interpolated.is_symmetric());
        assert!(interpolated.is_positive_semidefinite());
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn eigen_floor6_clamps_only_values_below_floor() {
        let mut marginal = [[0.0_f64; 6]; 6];
        for (idx, row) in marginal.iter_mut().enumerate() {
            row[idx] = (idx + 1) as f64;
        }
        marginal[5][5] = -1.0e-15;

        let floored = eigen_floor6(&marginal, 1.0e-9);
        assert!(cholesky_lower(&floored).is_some());
        assert!(Covariance6::try_from_matrix(floored).is_ok());

        let healthy = Covariance6::from_diagonal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let healthy_floored = eigen_floor6(healthy.as_matrix(), 1.0e-12);
        for i in 0..6 {
            for j in 0..6 {
                assert!((healthy_floored[i][j] - healthy.as_matrix()[i][j]).abs() <= 1.0e-12);
            }
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn covariance6_cdm_lower_triangle_unit_bridge_is_pinned() {
        let mut matrix = [[0.0_f64; 6]; 6];
        let mut value = 1.0_f64;
        for i in 0..6 {
            for j in 0..=i {
                matrix[i][j] = value;
                matrix[j][i] = value;
                value += 1.0;
            }
        }
        for i in 0..6 {
            matrix[i][i] += 30.0;
        }
        let covariance = Covariance6::try_from_matrix(matrix).unwrap();
        let meters = covariance6_km_to_m(&covariance).unwrap();
        let lower_triangle = [
            meters.as_matrix()[0][0],
            meters.as_matrix()[1][0],
            meters.as_matrix()[1][1],
            meters.as_matrix()[2][0],
            meters.as_matrix()[2][1],
            meters.as_matrix()[2][2],
            meters.as_matrix()[3][0],
            meters.as_matrix()[3][1],
            meters.as_matrix()[3][2],
            meters.as_matrix()[3][3],
            meters.as_matrix()[4][0],
            meters.as_matrix()[4][1],
            meters.as_matrix()[4][2],
            meters.as_matrix()[4][3],
            meters.as_matrix()[4][4],
            meters.as_matrix()[5][0],
            meters.as_matrix()[5][1],
            meters.as_matrix()[5][2],
            meters.as_matrix()[5][3],
            meters.as_matrix()[5][4],
            meters.as_matrix()[5][5],
        ];

        assert_eq!(
            lower_triangle,
            [
                31.0e6, 2.0e6, 33.0e6, 4.0e6, 5.0e6, 36.0e6, 7.0e6, 8.0e6, 9.0e6, 40.0e6, 11.0e6,
                12.0e6, 13.0e6, 14.0e6, 45.0e6, 16.0e6, 17.0e6, 18.0e6, 19.0e6, 20.0e6, 51.0e6,
            ]
        );
        assert_eq!(covariance6_m_to_km(&meters).unwrap(), covariance);
    }

    fn identity() -> Mat3 {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    }
}
