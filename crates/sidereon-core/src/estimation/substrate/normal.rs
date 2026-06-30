//! Shared normal-equation assembly for the estimation substrate.
//!
//! The three reference strategies reduce their weighted measurement rows to a
//! normal system in three named ways (see [`NormalRecipe`]):
//! - SPP feeds `sqrt(w)·(P_meas - P_hat)` residual rows to the trust-region
//!   solver, so the `AᵀWA` assembly is internal to the factorization
//!   ([`NormalRecipe::SppWeightedResidualFiniteDifference`], realized by
//!   [`crate::estimation::recipe::SolverRecipe::NalgebraTrfLegacy`]); nothing is
//!   assembled here for SPP.
//! - RTK folds correlated double-difference blocks into a flat information system
//!   `Λ x = η` and solves it first-tie
//!   ([`NormalRecipe::RtkDoubleDifferenceBlockFirstTie`]).
//! - PPP assembles a dense `AᵀWA x = AᵀWy` system from independent undifferenced
//!   rows and solves it last-tie ([`NormalRecipe::PppDenseLastTie`]).
//!
//! [`NormalAssembler`] is the recipe-keyed home for the RTK and PPP reductions;
//! each arm delegates to the exact operation-order kernel in
//! [`crate::astro::math::linear`] so selecting a recipe preserves every 0-ULP
//! reference golden. [`CovarianceBlock`] names the measurement covariance the
//! assembler folds: a diagonal (independent) block for SPP/PPP rows and the
//! correlated shared-reference double-difference block `R = D + σ_ref²·11ᵀ` for
//! RTK.

use crate::astro::math::linear::{
    invert_flat_first_tie_into, normal_equations_weighted, solve_flat_normal_first_tie_into,
    solve_flat_normal_square_root_into, solve_linear_last_tie, FlatCholeskySolveScratch,
    FlatLinearScratch, FlatNormalSolveScratch,
};

use crate::estimation::recipe::NormalRecipe;

/// The measurement covariance a [`NormalAssembler`] block folds.
///
/// SPP and PPP rows are independent: their covariance is diagonal and the fold
/// uses each row's scalar inverse-variance weight directly inside
/// [`crate::astro::math::linear::normal_equations_weighted`], so the independent
/// case needs no block object. RTK double-difference rows that share a reference
/// single difference are correlated, and that correlated block is what this type
/// names: `R = D + σ_ref²·11ᵀ` (each diagonal the row's own single-difference
/// variance plus the shared reference variance, every off-diagonal the shared
/// reference variance). Its inverse is materialized by
/// [`CovarianceBlock::inverse_into`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CovarianceBlock {
    /// Correlated double differences sharing one reference single difference:
    /// `R = D + σ_ref²·11ᵀ`.
    SharedReferenceDoubleDifference,
}

/// Reusable buffers for one correlated-block fold ([`CovarianceBlock::fold_block_into`]):
/// the materialized inverse covariance, the dense covariance staging, the `R⁻¹·y`
/// work vector, and the flat-invert scratch. Holding them across folds keeps a
/// steady-state filter step allocation-free. Opaque to callers; only the fold owns
/// the field layout.
#[derive(Debug, Default)]
pub(crate) struct BlockFoldScratch {
    r_inv: Vec<f64>,
    cov: Vec<f64>,
    r_inv_y: Vec<f64>,
    invert: FlatLinearScratch,
}

/// One block of correlated measurement rows, as seen by [`CovarianceBlock::fold_block_into`].
///
/// The fold reads each row's single-difference variance, shared reference
/// single-difference variance, design vector (length `n`), and measurement value
/// through this trait, so the substrate fold names no caller row type. Callers
/// implement it over their own pooled/borrowed row storage.
pub(crate) trait CorrelatedBlock {
    /// Number of rows in the block.
    fn len(&self) -> usize;
    /// The `k`-th row's single-difference variance (covariance diagonal term).
    fn sd_variance(&self, k: usize) -> f64;
    /// The `k`-th row's shared reference single-difference variance.
    fn ref_variance(&self, k: usize) -> f64;
    /// The `k`-th row's design vector (length `n`, the information-system dimension).
    fn design(&self, k: usize) -> &[f64];
    /// The `k`-th row's prefit measurement value.
    fn value(&self, k: usize) -> f64;
}

impl CovarianceBlock {
    /// Invert the shared-reference double-difference covariance for `m` rows into
    /// `r_inv` (row-major `m × m`). `sd_variance(k)` and `ref_variance(k)` return
    /// the k-th row's single-difference and reference single-difference variance.
    ///
    /// Op-for-op with the Elixir `double_difference_inverse_covariance`: the
    /// equal-variance closed form when every row shares the first row's variance,
    /// otherwise the dense `R = D + σ_ref²·11ᵀ` inverted through the first-tie
    /// flat kernel ([`crate::astro::math::linear::invert_flat_first_tie_into`]).
    /// `cov` and `invert` are reused scratch buffers, so steady-state folds do not
    /// allocate. NOT a Sherman-Morrison structured inverse: the kernel reproduces
    /// the reference's exact floating-point order for the 0-ULP trace gate.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn inverse_into(
        &self,
        m: usize,
        sd_variance: impl Fn(usize) -> f64,
        ref_variance: impl Fn(usize) -> f64,
        r_inv: &mut Vec<f64>,
        cov: &mut Vec<f64>,
        invert: &mut FlatLinearScratch,
    ) -> Option<()> {
        if m == 0 {
            r_inv.clear();
            return Some(());
        }

        let first = sd_variance(0);
        let constant = (0..m).all(|k| sd_variance(k) == first && ref_variance(k) == first);

        if constant {
            // Elixir `equal_double_difference_inverse_covariance/2`, exact op order:
            //   diagonal_scale = 1.0 / sd_var * (1.0 - 1.0 / (m + 1.0))
            //   off_diagonal   = -1.0 / (sd_var * (m + 1.0))
            let mf = m as f64;
            let diagonal_scale = 1.0 / first * (1.0 - 1.0 / (mf + 1.0));
            let off_diagonal = -1.0 / (first * (mf + 1.0));
            r_inv.resize(m * m, 0.0);
            for i in 0..m {
                for j in 0..m {
                    r_inv[i * m + j] = if i == j { diagonal_scale } else { off_diagonal };
                }
            }
            Some(())
        } else {
            // Elixir `double_difference_covariance/1` + `invert_matrix/1`, flat and
            // allocation-light. The off-diagonal and diagonal additions both use
            // the first row's reference variance; the Gaussian-elimination order
            // mirrors the shared first-tie flat linear kernel.
            let ref_v = ref_variance(0);
            cov.resize(m * m, 0.0);
            for i in 0..m {
                for j in 0..m {
                    cov[i * m + j] = if i == j {
                        sd_variance(i) + ref_v
                    } else {
                        ref_v
                    };
                }
            }
            invert_flat_first_tie_into(cov, m, r_inv, invert)
        }
    }

    /// Fold one block of correlated measurement rows into an information system
    /// in place: `Λ += Hᵀ R⁻¹ H`, `η += Hᵀ R⁻¹ y`, with `R⁻¹` materialized by
    /// [`CovarianceBlock::inverse_into`] from the block's variances. The rows are
    /// read through the [`CorrelatedBlock`] trait, so the fold names no caller row
    /// type. `lambda` is the row-major `n × n` information matrix, `eta` the
    /// length-`n` information vector (`n` = each row's design length); `scratch` is
    /// reused so steady-state folds do not allocate. Returns `None` if the block
    /// covariance is singular.
    ///
    /// Op-for-op with the Elixir `block_normal_equations`:
    ///   r_inv_y[a] = Σ_b R⁻¹[a][b]·y_b
    ///   Λ[i][j]   += Σ_a h_a[i]·(Σ_b R⁻¹[a][b]·h_b[j])   (b-sum grouped first)
    ///   η[i]      += Σ_a h_a[i]·r_inv_y[a]
    /// The grouping (inner b-sum, then ·h_a[i], then accumulate over a) must match
    /// exactly - a per-(a,b) form rounds differently and breaks the 0-ULP trace gate.
    #[allow(clippy::needless_range_loop)]
    pub(crate) fn fold_block_into(
        &self,
        block: &impl CorrelatedBlock,
        lambda: &mut [f64],
        eta: &mut [f64],
        scratch: &mut BlockFoldScratch,
    ) -> Option<()> {
        let m = block.len();
        if m == 0 {
            return Some(());
        }

        let n = eta.len();
        self.inverse_into(
            m,
            |k| block.sd_variance(k),
            |k| block.ref_variance(k),
            &mut scratch.r_inv,
            &mut scratch.cov,
            &mut scratch.invert,
        )?;
        let r_inv = &scratch.r_inv;

        scratch.r_inv_y.resize(m, 0.0);
        for (a, rinvy_a) in scratch.r_inv_y.iter_mut().enumerate() {
            let mut s = 0.0;
            for b in 0..m {
                s += r_inv[a * m + b] * block.value(b);
            }
            *rinvy_a = s;
        }

        for i in 0..n {
            let row = i * n;
            for j in 0..n {
                let mut acc = 0.0;
                for a in 0..m {
                    let hi = block.design(a)[i];
                    let mut row_sum = 0.0;
                    for b in 0..m {
                        row_sum += r_inv[a * m + b] * block.design(b)[j];
                    }
                    acc += hi * row_sum;
                }
                lambda[row + j] += acc;
            }
        }

        for (i, e) in eta.iter_mut().enumerate() {
            let mut acc = 0.0;
            for a in 0..m {
                acc += block.design(a)[i] * scratch.r_inv_y[a];
            }
            *e += acc;
        }

        Some(())
    }
}

/// Recipe-keyed reduction of weighted measurement rows to a normal system and its
/// solution. Each arm delegates to the exact [`crate::astro::math::linear`]
/// operation order the named [`NormalRecipe`] selects, so the reduction preserves
/// the strategy's 0-ULP reference golden. SPP
/// ([`NormalRecipe::SppWeightedResidualFiniteDifference`]) assembles no normal
/// equations here: its trust-region solver owns the factorization.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NormalAssembler {
    recipe: NormalRecipe,
}

impl NormalAssembler {
    /// A normal assembler for one named reduction order.
    pub(crate) const fn new(recipe: NormalRecipe) -> Self {
        Self { recipe }
    }

    /// PPP dense path: assemble `AᵀWA x = AᵀWy` from independent weighted rows
    /// (a diagonal-covariance fold, so each row contributes its scalar
    /// inverse-variance weight) and solve it with last-tie Gaussian elimination
    /// ([`NormalRecipe::PppDenseLastTie`]). Returns `None` on a singular system.
    pub(crate) fn solve_dense_last_tie<'a, I>(&self, rows: I, n: usize) -> Option<Vec<f64>>
    where
        I: IntoIterator<Item = (&'a [f64], f64, f64)>,
    {
        debug_assert_eq!(self.recipe, NormalRecipe::PppDenseLastTie);
        let (ata, aty) = normal_equations_weighted(rows, n)?;
        solve_linear_last_tie(ata, aty)
    }

    /// PPP dense assembly only (no solve): the `AᵀWA` / `AᵀWy` pair the fixed
    /// solver reduces with a Schur complement to seed the ambiguity covariance.
    pub(crate) fn assemble_dense<'a, I>(
        &self,
        rows: I,
        n: usize,
    ) -> Option<(Vec<Vec<f64>>, Vec<f64>)>
    where
        I: IntoIterator<Item = (&'a [f64], f64, f64)>,
    {
        debug_assert_eq!(self.recipe, NormalRecipe::PppDenseLastTie);
        normal_equations_weighted(rows, n)
    }

    /// Canonical PPP square-root path: assemble the SAME dense weighted normal
    /// system `AᵀWA x = AᵀWy` the PPP reference assembles from independent
    /// undifferenced rows (a diagonal-covariance fold), but solve it by the owned
    /// deterministic Cholesky (square-root) factorization `AᵀWA = L Lᵀ` plus
    /// forward/back substitution ([`NormalRecipe::CanonicalSquareRoot`], driven by
    /// the owned [`crate::estimation::recipe::SolverRecipe::OwnedDeterministicCholesky`]
    /// kernel). The Cholesky factor `L` is the information-matrix square root, so
    /// this is the square-root-information solve: the numerically rigorous op-order
    /// for the SPD normal matrix (no pivoting; exploits symmetry), distinct from
    /// the reference's dense last-tie Gaussian elimination
    /// ([`NormalRecipe::PppDenseLastTie`]). Assembly plus solve is owned scalar
    /// arithmetic with no nalgebra and no BLAS, and f64 sqrt is IEEE-754 correctly
    /// rounded, so this solve is bit-portable; the surrounding PPP measurement
    /// model that builds the weighted rows uses platform transcendentals, so the
    /// end-to-end canonical PPP bits are this-build reproducible, not portable.
    /// Returns `None` if `AᵀWA` is not positive definite (rank-deficient geometry).
    pub(crate) fn solve_dense_square_root<'a, I>(&self, rows: I, n: usize) -> Option<Vec<f64>>
    where
        I: IntoIterator<Item = (&'a [f64], f64, f64)>,
    {
        debug_assert_eq!(self.recipe, NormalRecipe::CanonicalSquareRoot);
        let (ata, aty) = normal_equations_weighted(rows, n)?;
        let mut lambda = Vec::with_capacity(n * n);
        for row in &ata {
            lambda.extend_from_slice(row);
        }
        let mut scratch = FlatCholeskySolveScratch::default();
        solve_flat_normal_square_root_into(&lambda, &aty, &mut scratch).map(<[f64]>::to_vec)
    }

    /// RTK flat path: solve the accumulated information system `Λ x = η` with
    /// first-tie flat Gaussian elimination, reusing `scratch`
    /// ([`NormalRecipe::RtkDoubleDifferenceBlockFirstTie`]). `lambda` is the
    /// row-major `n × n` information matrix, `eta` the length-`n` information
    /// vector. Returns `None` if `Λ` is singular.
    pub(crate) fn solve_flat_first_tie<'s>(
        &self,
        lambda: &[f64],
        eta: &[f64],
        scratch: &'s mut FlatNormalSolveScratch,
    ) -> Option<&'s [f64]> {
        debug_assert_eq!(self.recipe, NormalRecipe::RtkDoubleDifferenceBlockFirstTie);
        solve_flat_normal_first_tie_into(lambda, eta, scratch)
    }

    /// Canonical RTK square-root path: solve the SAME accumulated double-difference
    /// information system `Λ x = η` the reference assembles, but by the owned
    /// deterministic Cholesky (square-root) factorization `Λ = L Lᵀ` plus
    /// forward/back substitution ([`NormalRecipe::CanonicalSquareRoot`], driven by
    /// the owned [`crate::estimation::recipe::SolverRecipe::OwnedDeterministicCholesky`]
    /// kernel), reusing `scratch`. The Cholesky factor `L` is the information-matrix
    /// square root, so this is the square-root-information solve. It is the
    /// numerically rigorous op-order for the SPD normal matrix: it needs no
    /// pivoting and exploits the symmetry the reference's general first-tie
    /// Gaussian elimination
    /// ([`NormalRecipe::RtkDoubleDifferenceBlockFirstTie`]) does not. The whole
    /// path (block fold plus this solve) is owned scalar arithmetic with no
    /// nalgebra or BLAS, and f64 sqrt is IEEE-754 correctly rounded, so the
    /// canonical RTK bits are reproducible across platforms. Returns `None` if `Λ`
    /// is not positive definite (rank-deficient geometry).
    pub(crate) fn solve_square_root<'s>(
        &self,
        lambda: &[f64],
        eta: &[f64],
        scratch: &'s mut FlatCholeskySolveScratch,
    ) -> Option<&'s [f64]> {
        debug_assert_eq!(self.recipe, NormalRecipe::CanonicalSquareRoot);
        solve_flat_normal_square_root_into(lambda, eta, scratch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_reference_dd_inverse_matches_dense_inverse() {
        // Non-constant single-difference variances take the dense branch:
        // R = [[1.5, 0.5], [0.5, 2.5]] with ref variance 0.5.
        let sd = [1.0_f64, 2.0];
        let mut r_inv = Vec::new();
        let mut cov = Vec::new();
        let mut invert = FlatLinearScratch::default();
        CovarianceBlock::SharedReferenceDoubleDifference
            .inverse_into(2, |k| sd[k], |_| 0.5, &mut r_inv, &mut cov, &mut invert)
            .unwrap();
        // R · R⁻¹ ≈ I.
        let r = [[1.5_f64, 0.5], [0.5, 2.5]];
        for (i, r_row) in r.iter().enumerate() {
            for j in 0..2 {
                let prod = r_row[0] * r_inv[j] + r_row[1] * r_inv[2 + j];
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((prod - expect).abs() < 1.0e-12);
            }
        }
    }

    #[test]
    fn shared_reference_dd_inverse_uses_equal_variance_closed_form() {
        // Equal variances take the closed-form branch: a single double difference
        // (m = 1) sharing one reference single difference has inverse 1/(2·var).
        let mut r_inv = Vec::new();
        let mut cov = Vec::new();
        let mut invert = FlatLinearScratch::default();
        CovarianceBlock::SharedReferenceDoubleDifference
            .inverse_into(1, |_| 4.0, |_| 4.0, &mut r_inv, &mut cov, &mut invert)
            .unwrap();
        // diagonal_scale = 1/4 · (1 - 1/2) = 1/8.
        assert_eq!(r_inv, vec![0.125]);
    }

    #[test]
    fn empty_block_clears_inverse() {
        let mut r_inv = vec![9.0, 9.0];
        let mut cov = Vec::new();
        let mut invert = FlatLinearScratch::default();
        CovarianceBlock::SharedReferenceDoubleDifference
            .inverse_into(0, |_| 1.0, |_| 1.0, &mut r_inv, &mut cov, &mut invert)
            .unwrap();
        assert!(r_inv.is_empty());
    }

    #[test]
    fn dense_last_tie_solves_diagonal_system() {
        // Two independent rows pinning each unknown directly.
        let r0: Vec<f64> = vec![1.0, 0.0];
        let r1: Vec<f64> = vec![0.0, 1.0];
        let rows = [(r0.as_slice(), 1.0, 1.0), (r1.as_slice(), 2.0, 1.0)];
        let x = NormalAssembler::new(NormalRecipe::PppDenseLastTie)
            .solve_dense_last_tie(rows.iter().copied(), 2)
            .unwrap();
        assert_eq!(x, vec![1.0, 2.0]);
    }
}
