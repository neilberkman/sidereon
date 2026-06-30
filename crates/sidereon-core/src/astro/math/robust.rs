//! Robust M-estimation primitives for iteratively reweighted least squares.
//!
//! These are the per-outer-iteration reweighting pieces of a Huber IRLS loop:
//! a median-absolute-deviation scale estimate and the Huber weight function.
//! They are deliberately pure `f64` arithmetic (abs, compare, divide, sort by
//! [`f64::total_cmp`]) with no fused-multiply-add and no contraction, so the
//! per-iteration weight vector is bit-reproducible against an explicit
//! outer-loop reference recipe. The trust-region linear-algebra step that
//! consumes the weights is BLAS-bound and is NOT a 0-ULP target.

/// The default Huber tuning constant. Residuals scaled below this (in units of
/// the robust scale) keep full weight; larger ones are down-weighted as
/// `k / |u|`. `1.345` gives ~95% efficiency at the Gaussian model.
pub const HUBER_K: f64 = 1.345;

/// The MAD-to-sigma consistency constant for a normal distribution,
/// `1 / Phi^-1(3/4)`. Multiplying the median absolute deviation by this makes
/// it a consistent estimator of the standard deviation under normality.
pub const MAD_NORMAL_CONST: f64 = 1.4826;

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum RobustError {
    #[error("invalid robust statistic {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

impl RobustError {
    pub const fn field(&self) -> &'static str {
        match self {
            Self::InvalidInput { field, .. } => field,
        }
    }

    pub const fn reason(&self) -> &'static str {
        match self {
            Self::InvalidInput { reason, .. } => reason,
        }
    }
}

/// The median of `values`, computed on a `total_cmp` sort so the order (and
/// thus the result for an even count, which averages the two central values)
/// is deterministic. An empty slice yields `0.0`. The averaging of the two
/// central elements is a single `(a + b) / 2.0`, no FMA.
pub fn median(values: &[f64]) -> Result<f64, RobustError> {
    validate_finite_slice(values, "values")?;
    if values.is_empty() {
        return Ok(0.0);
    }
    let mut v: Vec<f64> = values.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        Ok(v[n / 2])
    } else {
        Ok((v[n / 2 - 1] + v[n / 2]) / 2.0)
    }
}

/// The median-absolute-deviation scale of `residuals`, scaled to a normal-sigma
/// estimate and floored at `scale_floor`.
///
/// `s = max(scale_floor, MAD_NORMAL_CONST * median(|r_i - median(r)|))`. The
/// floor prevents a near-perfect fit (MAD approaching zero) from blowing up the
/// scaled residuals `u_i = r_i / s` and spuriously down-weighting every
/// observation. Both medians use [`median`]'s `total_cmp` sort.
pub fn mad_scale(residuals: &[f64], scale_floor: f64) -> Result<f64, RobustError> {
    validate_finite_positive(scale_floor, "scale_floor")?;
    let med = median(residuals)?;
    let abs_dev: Vec<f64> = residuals.iter().map(|r| (r - med).abs()).collect();
    let mad = median(&abs_dev)?;
    let scaled = MAD_NORMAL_CONST * mad;
    if scaled > scale_floor {
        Ok(scaled)
    } else {
        Ok(scale_floor)
    }
}

/// The Huber weight for a scaled residual `u = r / s`.
///
/// `w(u) = 1` for `|u| <= k` and `w(u) = k / |u|` otherwise (the Huber
/// `psi(u) / u` form, always in `(0, 1]`). At `u == 0` the weight is `1`. This
/// is the multiplier applied on top of any base (elevation) weight to obtain the
/// effective per-observation weight of the current outer iteration.
pub fn huber_weight(u: f64, k: f64) -> f64 {
    let a = u.abs();
    if a <= k {
        1.0
    } else {
        k / a
    }
}

fn validate_finite_slice(values: &[f64], field: &'static str) -> Result<(), RobustError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn validate_finite_positive(value: f64, field: &'static str) -> Result<(), RobustError> {
    if !value.is_finite() {
        Err(invalid_input(field, "not finite"))
    } else if value <= 0.0 {
        Err(invalid_input(field, "not positive"))
    } else {
        Ok(())
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> RobustError {
    RobustError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]).unwrap(), 2.0);
        assert_eq!(median(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 2.5);
        assert_eq!(median(&[]).unwrap(), 0.0);
    }

    #[test]
    fn median_rejects_nonfinite_sample() {
        assert_eq!(
            median(&[1.0, f64::NAN]),
            Err(RobustError::InvalidInput {
                field: "values",
                reason: "not finite"
            })
        );
    }

    #[test]
    fn huber_weight_breaks_at_k() {
        assert_eq!(huber_weight(0.0, HUBER_K), 1.0);
        assert_eq!(huber_weight(HUBER_K, HUBER_K), 1.0);
        let w = huber_weight(2.0 * HUBER_K, HUBER_K);
        assert!((w - 0.5).abs() < 1e-15);
    }

    #[test]
    fn mad_scale_floored() {
        // All-equal residuals give MAD 0, so the floor governs.
        assert_eq!(mad_scale(&[5.0, 5.0, 5.0], 0.25).unwrap(), 0.25);
    }

    #[test]
    fn mad_scale_rejects_nonfinite_sample() {
        assert_eq!(
            mad_scale(&[5.0, f64::INFINITY], 0.25),
            Err(RobustError::InvalidInput {
                field: "values",
                reason: "not finite"
            })
        );
    }
}
