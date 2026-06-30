//! Shared integer-ambiguity resolution substrate (Phase-2 P3).
//!
//! Before P3 each stack invoked the LAMBDA integer least-squares kernel
//! ([`crate::ils::lambda_ils_search`]) from its own call site: the RTK static
//! fixed solver through `rtk_filter::search`, the sequential filter inline in
//! `rtk_filter::update`, the PPP fixed solver inline in
//! `precise_positioning::fixed`. The kernel was already shared, but the
//! *resolution step* (turn float cycles + a cycle covariance into the
//! ratio-tested integer fix) had no single named home. This module is that
//! home: [`resolve_integer_lattice`] is the one substrate entry every strategy
//! routes its LAMBDA call through. It is a thin, behavior-preserving
//! wrapper - same kernel, same arguments, same op-order - so every reference
//! golden is unchanged.
//!
//! The RTK-vs-PPP *difference* is named as DATA, not as two algorithm trees, by
//! [`crate::estimation::recipe::AmbiguityIdPolicy`]: the two stacks run the same
//! LAMBDA kernel and differ only in how they form ambiguity identifiers, whether
//! they gate float-only constellations, and whether they attempt partial
//! resolution. The runtime strategy selector (P4) consumes that policy.

use crate::ils::{lambda_ils_search, IlsError, IlsResult};

/// Resolve one integer-ambiguity lattice: the single substrate home for the
/// LAMBDA integer least-squares kernel.
///
/// `float_cycles` are the float ambiguities in cycles and `covariance` their
/// cycle covariance (row-major, symmetric). This delegates verbatim to
/// [`crate::ils::lambda_ils_search`] with the supplied ratio threshold, so the
/// returned [`IlsResult`] - fixed integers, ratio, scores, symmetrized
/// covariance - is bit-identical to the previous inline call at every site.
#[inline]
pub(crate) fn resolve_integer_lattice(
    float_cycles: &[f64],
    covariance: &[Vec<f64>],
    ratio_threshold: f64,
) -> Result<IlsResult, IlsError> {
    if let Some(covariance) = symmetrized_square(covariance) {
        lambda_ils_search(float_cycles, &covariance, ratio_threshold)
    } else {
        lambda_ils_search(float_cycles, covariance, ratio_threshold)
    }
}

fn symmetrized_square(covariance: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = covariance.len();
    if n == 0 || covariance.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut out = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        out[i][i] = covariance[i][i];
        for j in (i + 1)..n {
            let value = 0.5 * (covariance[i][j] + covariance[j][i]);
            out[i][j] = value;
            out[j][i] = value;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolver_matches_direct_lambda_call() {
        let floats = [0.2_f64, -0.9, 1.1];
        let covariance = vec![
            vec![0.04_f64, 0.001, 0.0],
            vec![0.001, 0.05, 0.002],
            vec![0.0, 0.002, 0.03],
        ];
        let ratio = 3.0;
        let direct = lambda_ils_search(&floats, &covariance, ratio).unwrap();
        let routed = resolve_integer_lattice(&floats, &covariance, ratio).unwrap();
        assert_eq!(direct.fixed, routed.fixed);
        assert_eq!(direct.fixed_status, routed.fixed_status);
        assert_eq!(direct.ratio.to_bits(), routed.ratio.to_bits());
        assert_eq!(direct.best_score.to_bits(), routed.best_score.to_bits());
        assert_eq!(direct.candidates_evaluated, routed.candidates_evaluated);
    }
}
