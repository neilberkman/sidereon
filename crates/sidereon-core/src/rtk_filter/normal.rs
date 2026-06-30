//! Information-form normal-equation adapters for the sequential RTK filter.
//!
//! The correlated double-difference block covariance `R = D + σ_ref²·11ᵀ`, its
//! inverse, and the block fold (`Λ += Hᵀ R⁻¹ H`, `η += Hᵀ R⁻¹ y`) all live in the
//! shared substrate ([`CovarianceBlock`]); this module adapts the RTK row types
//! (`DdRow`, `DdRowScratch` from `rows`) onto that closure-parameterized fold and
//! owns only the RTK-specific arithmetic the substrate has no row type for: the
//! single-row reference fold the unit tests check, the fix-and-hold
//! pseudo-measurement fold, and the flat Gaussian solve. (`FilterState` from
//! `state` and `Hold` in the parent reach this module through the parent's
//! imports.)

#[cfg(test)]
use crate::astro::math::linear::FlatLinearScratch;

use crate::estimation::substrate::normal::{BlockFoldScratch, CorrelatedBlock, CovarianceBlock};

#[cfg(test)]
use super::DdRow;
use super::{DdRowScratch, FilterState, Hold};

/// Fold one weighted measurement into an information system in place:
/// `Λ += weight·hhᵀ`, `η += weight·h·y`. `lambda` is the row-major `n x n`
/// information matrix, `eta` the length-`n` information vector, `h` the length-`n`
/// measurement row. Single-row reference accumulation the unit tests check the
/// block covariance fold path against; production folds rows through the block
/// covariance path so shared-reference covariance is preserved.
#[cfg(test)]
pub(super) fn fold_measurement(
    lambda: &mut [f64],
    eta: &mut [f64],
    h: &[f64],
    weight: f64,
    y: f64,
) {
    let n = eta.len();
    for i in 0..n {
        let whi = weight * h[i];
        eta[i] += whi * y;
        let row = i * n;
        for j in 0..n {
            lambda[row + j] += whi * h[j];
        }
    }
}

// The shared-reference double-difference covariance inverse lives in the
// substrate ([`CovarianceBlock::inverse_into`]). This test wrapper materializes
// the inverse for one block of `DdRow` rows so the unit tests can assert `R⁻¹`
// directly.
#[cfg(test)]
pub(super) fn double_difference_inverse_covariance(rows: &[&DdRow]) -> Option<Vec<f64>> {
    let mut r_inv = Vec::new();
    let mut cov = Vec::new();
    let mut invert = FlatLinearScratch::default();
    CovarianceBlock::SharedReferenceDoubleDifference.inverse_into(
        rows.len(),
        |k| rows[k].sd_variance_m2,
        |k| rows[k].ref_sd_variance_m2,
        &mut r_inv,
        &mut cov,
        &mut invert,
    )?;
    Some(r_inv)
}

/// Adapts an `indices`-selected slice of pooled `DdRowScratch` rows onto the
/// substrate fold's [`CorrelatedBlock`] view (production hot path).
struct ScratchRowBlock<'a> {
    rows: &'a [DdRowScratch],
    indices: &'a [usize],
}

impl CorrelatedBlock for ScratchRowBlock<'_> {
    fn len(&self) -> usize {
        self.indices.len()
    }
    fn sd_variance(&self, k: usize) -> f64 {
        self.rows[self.indices[k]].sd_variance_m2
    }
    fn ref_variance(&self, k: usize) -> f64 {
        self.rows[self.indices[k]].ref_sd_variance_m2
    }
    fn design(&self, k: usize) -> &[f64] {
        &self.rows[self.indices[k]].h
    }
    fn value(&self, k: usize) -> f64 {
        self.rows[self.indices[k]].y
    }
}

/// Adapts a slice of borrowed `DdRow` rows onto the substrate fold's
/// [`CorrelatedBlock`] view (unit-test path).
#[cfg(test)]
struct DdRowBlock<'a>(&'a [&'a DdRow]);

#[cfg(test)]
impl CorrelatedBlock for DdRowBlock<'_> {
    fn len(&self) -> usize {
        self.0.len()
    }
    fn sd_variance(&self, k: usize) -> f64 {
        self.0[k].sd_variance_m2
    }
    fn ref_variance(&self, k: usize) -> f64 {
        self.0[k].ref_sd_variance_m2
    }
    fn design(&self, k: usize) -> &[f64] {
        &self.0[k].h
    }
    fn value(&self, k: usize) -> f64 {
        self.0[k].y
    }
}

/// Fold one epoch/kind block of `DdRow` double-difference rows through the shared
/// correlated-block fold ([`CovarianceBlock::fold_block_into`]). Rows in a block
/// share the reference satellite's single-difference variance, so the covariance
/// is `R = D + σ_ref²·11ᵀ`, not diagonal. Test adapter over the substrate fold.
#[cfg(test)]
pub(super) fn fold_measurement_block(
    lambda: &mut [f64],
    eta: &mut [f64],
    rows: &[&DdRow],
) -> Option<()> {
    let mut scratch = BlockFoldScratch::default();
    CovarianceBlock::SharedReferenceDoubleDifference.fold_block_into(
        &DdRowBlock(rows),
        lambda,
        eta,
        &mut scratch,
    )
}

/// Production hot path: fold the `indices`-selected block of `DdRowScratch` rows
/// through the shared correlated-block fold ([`CovarianceBlock::fold_block_into`]),
/// reusing `scratch`. Adapts the pooled scratch rows onto the substrate fold's
/// row view so the fold arithmetic lives once in the substrate.
pub(super) fn fold_measurement_block_indices(
    lambda: &mut [f64],
    eta: &mut [f64],
    rows: &[DdRowScratch],
    indices: &[usize],
    scratch: &mut BlockFoldScratch,
) -> Option<()> {
    CovarianceBlock::SharedReferenceDoubleDifference.fold_block_into(
        &ScratchRowBlock { rows, indices },
        lambda,
        eta,
        scratch,
    )
}

pub(super) fn fold_hold_block_with_ambiguities(
    lambda: &mut [f64],
    eta: &mut [f64],
    state: &FilterState,
    sd_ambiguities_m: &[f64],
    held: &[Hold],
    hold_weight: f64,
) -> Option<()> {
    let n = state.dim();

    // Op-for-op with Elixir `sequential_hold_normal_equations/5`: build the hold
    // normal-equation block separately, using `weighted_h = h * weight`, then
    // accumulate `hi * weighted_hj`. Do not call `fold_measurement`: its
    // `(weight * hi) * hj` grouping is algebraically equivalent but not 0-ULP
    // equivalent, and Elixir adds the hold block after `(prior + measurement)`.
    for h in held {
        let sp = state.ambiguity_pos(&h.sat_sd_id)?;
        let rp = state.ambiguity_pos(&h.ref_sd_id)?;
        let sp_col = 3 + sp;
        let rp_col = 3 + rp;
        let current_dd = sd_ambiguities_m[sp] - sd_ambiguities_m[rp];
        let residual = h.fixed_m - current_dd;

        for i in 0..n {
            let hi = hold_design_value(i, sp_col, rp_col);
            let offset = i * n;
            for j in 0..n {
                let weighted_hj = hold_design_value(j, sp_col, rp_col) * hold_weight;
                lambda[offset + j] += hi * weighted_hj;
            }
        }

        for (i, eta_i) in eta.iter_mut().enumerate().take(n) {
            let weighted_hi = hold_design_value(i, sp_col, rp_col) * hold_weight;
            *eta_i += residual * weighted_hi;
        }
    }

    Some(())
}

fn hold_design_value(i: usize, sat_col: usize, ref_col: usize) -> f64 {
    if i == sat_col {
        1.0
    } else if i == ref_col {
        -1.0
    } else {
        0.0
    }
}

/// Solve the information system `Λ x = η` for the state correction, using flat
/// row-major Gaussian elimination (partial pivoting, singular guard). Returns
/// `None` if `Λ` is singular. `lambda` is row-major `n x n`. Test-only reference
/// wrapper over the shared first-tie kernel.
#[cfg(test)]
pub(super) fn solve_normal(lambda: &[f64], eta: &[f64]) -> Option<Vec<f64>> {
    crate::astro::math::linear::solve_flat_normal_first_tie(lambda, eta)
}
