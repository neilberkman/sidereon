//! Shared residual-screening / quality-control substrate (Phase-2 P3).
//!
//! Every reference strategy screens its post-fit (or, for the sequential filter,
//! its predicted) residuals before trusting a solution, but the screens were
//! written as independent code with their own inline normalization. The four
//! screen families are named as DATA by
//! [`crate::estimation::recipe::ScreenKind`]; the chi-square RAIM aggregate keeps
//! its own kernel in [`crate::quality`], while the per-residual screens share one
//! decision (*normalize a residual by its weight, compare to a sigma threshold*)
//! and differ in the normalization op-order AND in what the weight means
//! (inverse variance for the sequential filter, inverse sigma for the static
//! baselines and PPP).
//!
//! Both are parity-sensitive choices, so each lives here as one named-recipe arm
//! ([`normalized_residual`], keyed by
//! [`crate::estimation::recipe::ResidualNormRecipe`]) instead of being copied
//! inline at each screen. Every recipe commits its `weight` argument to a single
//! unambiguous meaning, and each arm reproduces the exact arithmetic the original
//! call site used, so every frozen-bits golden is unchanged.

use crate::estimation::recipe::ResidualNormRecipe;

/// Normalize one residual `value` by its `weight` under the named recipe. Each
/// recipe commits `weight` to a single meaning so the argument is unambiguous:
/// `RtkInverseVarianceInnovation` takes an inverse *variance* (`1/sigma^2`),
/// `RtkInverseSigmaResidual` and `PppInverseSigmaMagnitude` take an inverse
/// *sigma* (`1/sigma`), so the normalized residual is the plain product
/// `value * weight` (the studentized residual in sigmas). Callers that screen on
/// the magnitude apply `.abs()` to the result (RTK stores the signed product and
/// takes the absolute value at comparison time); the PPP variant folds the
/// absolute value of the residual in directly.
#[inline]
pub(crate) fn normalized_residual(recipe: ResidualNormRecipe, value: f64, weight: f64) -> f64 {
    match recipe {
        // Same product op-order; the two recipes differ only in what `weight`
        // means (inverse variance for the sequential filter, inverse sigma for
        // the static baselines), which the caller commits to by variant.
        ResidualNormRecipe::RtkInverseVarianceInnovation
        | ResidualNormRecipe::RtkInverseSigmaResidual => value * weight,
        // PPP weights are inverse sigma (the least-squares stage whitens each
        // observation by `weight = 1/sigma`), so the studentized residual is the
        // plain product with the magnitude folded in: `|value| * weight`.
        ResidualNormRecipe::PppInverseSigmaMagnitude => value.abs() * weight,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rtk_inverse_variance_innovation_is_a_product() {
        // value · weight (weight an inverse variance), sign preserved (the caller
        // abs()es at compare time).
        let value = -2.0_f64;
        let weight = 0.25_f64;
        let normalized = normalized_residual(
            ResidualNormRecipe::RtkInverseVarianceInnovation,
            value,
            weight,
        );
        assert_eq!(normalized.to_bits(), (value * weight).to_bits());
        assert_eq!(normalized.abs().to_bits(), (value * weight).abs().to_bits());
    }

    #[test]
    fn rtk_inverse_sigma_residual_is_a_product() {
        // value · weight (weight an inverse sigma): same product op-order as the
        // innovation recipe, distinct only in the committed weight meaning.
        let value = -2.0_f64;
        let weight = 0.5_f64;
        let normalized =
            normalized_residual(ResidualNormRecipe::RtkInverseSigmaResidual, value, weight);
        assert_eq!(normalized.to_bits(), (value * weight).to_bits());
    }

    #[test]
    fn ppp_inverse_sigma_magnitude_studentizes_the_residual() {
        // |r| · weight (weight an inverse sigma) = |residual| / sigma, folding the
        // residual magnitude in directly. A 5-sigma residual normalizes to 5.0.
        let value = -3.0_f64;
        let weight = 0.5_f64; // inverse sigma -> sigma = 2.0
        let normalized =
            normalized_residual(ResidualNormRecipe::PppInverseSigmaMagnitude, value, weight);
        assert_eq!(normalized.to_bits(), (value.abs() * weight).to_bits());
        // 3.0 / 2.0 = 1.5 sigma.
        assert_eq!(normalized, 1.5);
    }
}
