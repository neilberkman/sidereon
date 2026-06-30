//! Dense normal-equation kernels for the static PPP solvers.
//!
//! These are the least-squares primitives shared by the float and fixed solve
//! clusters: the weighted measurement [`Row`], the `AᵀWA x = AᵀWy` assembly and
//! its Gaussian solve, and the Schur-complement reduction that extracts the
//! ambiguity-block covariance used to seed the LAMBDA search. The arithmetic
//! delegates to the parity-aware kernels in [`crate::astro::math::linear`];
//! this module owns only the PPP-specific assembly and the error mapping
//! ([`FloatSolveError`], [`FixedSolveError`]) onto those kernels' optional
//! results.

use crate::astro::math::linear::{
    invert_matrix_last_tie, matmul, matrix_sub, solve_matrix_last_tie, transpose,
};

use crate::estimation::recipe::NormalRecipe;
use crate::estimation::substrate::normal::NormalAssembler;
use crate::estimation::substrate::rows::ResidualRow;

use super::{FixedSolveError, FloatSolveError};

/// One weighted measurement row: design coefficients `h`, prefit residual `y`,
/// and the diagonal weight (inverse variance). The PPP solver row is the shared
/// substrate [`ResidualRow`].
pub(super) type Row = ResidualRow;

/// The PPP reductions select the dense last-tie assembler.
const PPP_ASSEMBLER: NormalAssembler = NormalAssembler::new(NormalRecipe::PppDenseLastTie);

/// Solve the PPP normal equations under the resolved [`NormalRecipe`]. The
/// validated runners flow the resolved recipe through here. For the PPP reference
/// recipe (`NormalRecipe::PppDenseLastTie`) this is bit-identical to assembling
/// through [`PPP_ASSEMBLER`] (which the covariance path [`normal_equations`] still
/// uses) and solving last-tie. For the canonical recipe
/// (`NormalRecipe::CanonicalSquareRoot`) the SAME dense weighted normal system is
/// instead solved by the owned deterministic Cholesky square-root factorization;
/// canonical PPP is the only non-reference recipe that reaches this seam. Any
/// other recipe is a wiring error (no PPP strategy selects it).
pub(super) fn solve_normal_equations(
    rows: &[Row],
    n: usize,
    normal: NormalRecipe,
) -> Result<Vec<f64>, FloatSolveError> {
    let assembler = NormalAssembler::new(normal);
    let weighted = || rows.iter().map(Row::as_weighted);
    let solution = match normal {
        NormalRecipe::CanonicalSquareRoot => assembler.solve_dense_square_root(weighted(), n),
        _ => assembler.solve_dense_last_tie(weighted(), n),
    };
    solution.ok_or(FloatSolveError::SingularGeometry)
}

pub(super) fn normal_equations(
    rows: &[Row],
    n: usize,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), FloatSolveError> {
    PPP_ASSEMBLER
        .assemble_dense(rows.iter().map(Row::as_weighted), n)
        .ok_or(FloatSolveError::SingularGeometry)
}

fn submatrix(
    matrix: &[Vec<f64>],
    row_start: usize,
    row_count: usize,
    col_start: usize,
    col_count: usize,
) -> Vec<Vec<f64>> {
    matrix[row_start..row_start + row_count]
        .iter()
        .map(|row| row[col_start..col_start + col_count].to_vec())
        .collect()
}

pub(super) fn ambiguity_covariance_from_normal(
    normal: &[Vec<f64>],
    start: usize,
    n_ambiguities: usize,
) -> Result<Vec<Vec<f64>>, FixedSolveError> {
    let a = submatrix(normal, 0, start, 0, start);
    let b = submatrix(normal, 0, start, start, n_ambiguities);
    let c = submatrix(normal, start, n_ambiguities, start, n_ambiguities);
    let a_inv_b = solve_matrix_last_tie(&a, &b)
        .ok_or(FixedSolveError::Float(FloatSolveError::SingularGeometry))?;
    let b_t = transpose(&b).ok_or(FixedSolveError::Float(FloatSolveError::SingularGeometry))?;
    let bt_a_inv_b =
        matmul(&b_t, &a_inv_b).ok_or(FixedSolveError::Float(FloatSolveError::SingularGeometry))?;
    let schur = matrix_sub(&c, &bt_a_inv_b)
        .ok_or(FixedSolveError::Float(FloatSolveError::SingularGeometry))?;
    invert_matrix_last_tie(&schur).ok_or(FixedSolveError::Float(FloatSolveError::SingularGeometry))
}
