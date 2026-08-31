//! Shared covariance and protection-metric primitives.
//!
//! This module holds the small numerical kernels used by protection-level and
//! position-error code: a symmetric 2x2 covariance eigensolve, row gain metrics,
//! and public special-function entry points.

use crate::astro::math::linear::invert_symmetric_pd;
pub use crate::astro::math::special::{erfc_inv, normal_q, normal_q_inv};
use crate::dop::ecef_to_enu_rotation;
use crate::frame::Wgs84Geodetic;

/// A confidence ellipse from an arbitrary 2x2 covariance block.
///
/// The semi-axes carry the square root unit of the covariance entries.
/// `orientation_rad` is measured from axis 0 toward axis 1.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ErrorEllipse2 {
    /// Confidence probability associated with `chi_square_scale`.
    pub confidence: f64,
    /// Two-degree-of-freedom chi-square scale applied to the covariance
    /// eigenvalues.
    pub chi_square_scale: f64,
    /// Semi-major axis length.
    pub semi_major: f64,
    /// Semi-minor axis length.
    pub semi_minor: f64,
    /// Semi-major-axis orientation in radians.
    pub orientation_rad: f64,
}

/// Error returned by shared integrity primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegrityError {
    /// At least one numeric input was NaN or infinite.
    NonFinite,
    /// The covariance has a negative eigenvalue outside round-off tolerance.
    NotPositiveSemidefinite,
    /// A probability argument was outside `(0, 1)`.
    InvalidProbability {
        /// Reason the probability failed validation.
        reason: &'static str,
    },
    /// A design or weight input was malformed.
    InvalidInput {
        /// Name of the malformed field.
        field: &'static str,
        /// Validation failure reason.
        reason: &'static str,
    },
    /// The weighted design matrix was singular.
    Singular,
}

/// Gain rows in local ENU coordinates.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct GainMatrix {
    pub(crate) enu_rows: [Vec<f64>; 3],
}

/// Confidence ellipse for a symmetric 2x2 covariance block.
///
/// The covariance is symmetrized by averaging the off-diagonal entries. The
/// eigenvalues are the closed-form symmetric-2x2 solution
/// `lambda = center +/- sqrt(half_delta^2 + b^2)`, then scaled by the
/// chi-square contour factor `-2 ln(1 - confidence)`.
pub fn error_ellipse_2x2(
    covariance: [[f64; 2]; 2],
    confidence: f64,
) -> Result<ErrorEllipse2, IntegrityError> {
    validate_probability(confidence)?;
    let scale = -2.0 * libm::log(1.0 - confidence);
    error_ellipse_2x2_scaled(covariance, scale, confidence)
}

/// One-sigma ellipse for a symmetric 2x2 covariance block.
///
/// This uses the same eigensolve as [`error_ellipse_2x2`] with a unit scale.
/// The stored confidence is the two-degree-of-freedom probability associated
/// with a unit chi-square contour.
pub fn error_ellipse_2x2_unit(covariance: [[f64; 2]; 2]) -> Result<ErrorEllipse2, IntegrityError> {
    error_ellipse_2x2_scaled(covariance, 1.0, 1.0 - libm::exp(-0.5_f64))
}

/// Standard deviation for one gain row and per-measurement sigmas.
pub fn metric_sigma(gain_row: &[f64], sigmas_m: &[f64]) -> f64 {
    metric_cross(gain_row, gain_row, sigmas_m).sqrt()
}

/// Covariance cross term for two gain rows and per-measurement sigmas.
pub fn metric_cross(gain_row_a: &[f64], gain_row_b: &[f64], sigmas_m: &[f64]) -> f64 {
    gain_row_a
        .iter()
        .zip(gain_row_b)
        .zip(sigmas_m)
        .map(|((&a, &b), &sigma)| a * b * sigma * sigma)
        .sum()
}

pub(crate) fn gain_matrix_enu_from_design_rows(
    design_rows: &[Vec<f64>],
    weights: &[f64],
    receiver: Wgs84Geodetic,
    n_state: usize,
) -> Result<GainMatrix, IntegrityError> {
    if weights.len() != design_rows.len() {
        return Err(invalid_input("weights", "length must match rows"));
    }
    if n_state < 3 {
        return Err(invalid_input("n_state", "must include position columns"));
    }
    if weights.iter().filter(|&&weight| weight > 0.0).count() < n_state {
        return Err(IntegrityError::Singular);
    }

    let mut normal = vec![vec![0.0_f64; n_state]; n_state];
    for (row, &weight) in design_rows.iter().zip(weights) {
        if row.len() != n_state {
            return Err(invalid_input("rows", "length must match state dimension"));
        }
        if !weight.is_finite() || weight < 0.0 {
            return Err(invalid_input("weights", "not finite non-negative"));
        }
        if row.iter().any(|value| !value.is_finite()) {
            return Err(invalid_input("rows", "not finite"));
        }
        if weight > 0.0 {
            for i in 0..n_state {
                for j in 0..n_state {
                    normal[i][j] += row[i] * weight * row[j];
                }
            }
        }
    }

    let inverse = invert_symmetric_pd(&normal).ok_or(IntegrityError::Singular)?;
    let mut ecef_rows = [
        vec![0.0; design_rows.len()],
        vec![0.0; design_rows.len()],
        vec![0.0; design_rows.len()],
    ];
    for (measurement_idx, (design, &weight)) in design_rows.iter().zip(weights).enumerate() {
        if weight == 0.0 {
            continue;
        }
        for state_idx in 0..3 {
            let mut value = 0.0;
            for col in 0..n_state {
                value += inverse[state_idx][col] * design[col] * weight;
            }
            ecef_rows[state_idx][measurement_idx] = value;
        }
    }

    let r = ecef_to_enu_rotation(receiver.lat_rad, receiver.lon_rad);
    let mut enu_rows = [
        vec![0.0; design_rows.len()],
        vec![0.0; design_rows.len()],
        vec![0.0; design_rows.len()],
    ];
    for coord in 0..3 {
        for measurement_idx in 0..design_rows.len() {
            enu_rows[coord][measurement_idx] = r[coord][0] * ecef_rows[0][measurement_idx]
                + r[coord][1] * ecef_rows[1][measurement_idx]
                + r[coord][2] * ecef_rows[2][measurement_idx];
        }
    }

    Ok(GainMatrix { enu_rows })
}

fn error_ellipse_2x2_scaled(
    covariance: [[f64; 2]; 2],
    chi_square_scale: f64,
    confidence: f64,
) -> Result<ErrorEllipse2, IntegrityError> {
    for row in covariance {
        if row.iter().any(|value| !value.is_finite()) {
            return Err(IntegrityError::NonFinite);
        }
    }

    let a = covariance[0][0];
    let b = 0.5 * (covariance[0][1] + covariance[1][0]);
    let c = covariance[1][1];
    let half_delta = 0.5 * (a - c);
    let center = 0.5 * (a + c);
    let root = (half_delta * half_delta + b * b).sqrt();
    let lambda_major = center + root;
    let lambda_minor = center - root;
    if !lambda_major.is_finite() || !lambda_minor.is_finite() || lambda_minor < -1.0e-12 {
        return Err(IntegrityError::NotPositiveSemidefinite);
    }

    let semi_major = (lambda_major.max(0.0) * chi_square_scale).sqrt();
    let semi_minor = (lambda_minor.max(0.0) * chi_square_scale).sqrt();
    let orientation_rad = if root == 0.0 {
        0.0
    } else {
        0.5 * libm::atan2(2.0 * b, a - c)
    };
    Ok(ErrorEllipse2 {
        confidence,
        chi_square_scale,
        semi_major,
        semi_minor,
        orientation_rad,
    })
}

fn validate_probability(value: f64) -> Result<(), IntegrityError> {
    if !value.is_finite() {
        return Err(IntegrityError::InvalidProbability {
            reason: "not finite",
        });
    }
    if !(0.0..1.0).contains(&value) {
        return Err(IntegrityError::InvalidProbability {
            reason: "out of range",
        });
    }
    Ok(())
}

fn invalid_input(field: &'static str, reason: &'static str) -> IntegrityError {
    IntegrityError::InvalidInput { field, reason }
}
