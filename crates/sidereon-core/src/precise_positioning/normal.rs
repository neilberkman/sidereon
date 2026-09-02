//! Dense normal-equation kernels for the static PPP solvers.
//!
//! These are the least-squares primitives shared by the float and fixed solve
//! clusters: the weighted measurement [`Row`], the `AᵀWA x = AᵀWy` reductions,
//! the per-epoch receiver-clock Schur elimination used by the static solve, and
//! the Schur-complement reduction that extracts the ambiguity-block covariance
//! used to seed the LAMBDA search. The arithmetic
//! delegates to the parity-aware kernels in [`crate::astro::math::linear`];
//! this module owns only the PPP-specific assembly and the error mapping
//! ([`FloatSolveError`], [`FixedSolveError`]) onto those kernels' optional
//! results.

use crate::astro::math::linear::{
    invert_matrix_last_tie, invert_symmetric_pd, matmul, matrix_sub,
    solve_flat_normal_square_root_into, solve_linear_last_tie, solve_matrix_last_tie, transpose,
    FlatCholeskySolveScratch,
};

use crate::estimation::recipe::NormalRecipe;
use crate::estimation::substrate::normal::NormalAssembler;
use crate::estimation::substrate::rows::ResidualRow;
use crate::frame::{itrf_to_geodetic, ItrfPositionM};

use super::{FixedSolveError, FloatSolveError};

/// One weighted measurement row: design coefficients `h`, prefit residual `y`,
/// and the diagonal weight as an inverse sigma. The PPP solver row is the shared
/// substrate [`ResidualRow`].
pub(super) type Row = ResidualRow;

#[derive(Debug, Clone, PartialEq)]
pub(super) struct PppPositionCovariance {
    pub(super) scaled: crate::dop::PositionCovariance,
    pub(super) formal: crate::dop::PositionCovariance,
    pub(super) tropo_gradient_scaled_m2: Option<[[f64; 2]; 2]>,
    pub(super) tropo_gradient_formal_m2: Option<[[f64; 2]; 2]>,
    pub(super) posterior_variance_factor: f64,
    pub(super) covariance_scale_factor: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PppNormalLayout {
    n_epochs: usize,
    n_ztd: usize,
    n_tropo_gradients: usize,
    n_residual_ionosphere: usize,
    n_ambiguities: usize,
}

impl PppNormalLayout {
    pub(super) const fn new(
        n_epochs: usize,
        n_ztd: usize,
        n_tropo_gradients: usize,
        n_residual_ionosphere: usize,
        n_ambiguities: usize,
    ) -> Self {
        Self {
            n_epochs,
            n_ztd,
            n_tropo_gradients,
            n_residual_ionosphere,
            n_ambiguities,
        }
    }

    pub(super) const fn full_dim(&self) -> usize {
        3 + self.n_epochs
            + self.n_ztd
            + self.n_tropo_gradients
            + self.n_residual_ionosphere
            + self.n_ambiguities
    }

    pub(super) const fn reduced_dim(&self) -> usize {
        3 + self.n_ztd + self.n_tropo_gradients + self.n_residual_ionosphere + self.n_ambiguities
    }

    pub(super) const fn reduced_ambiguity_offset(&self) -> usize {
        3 + self.n_ztd + self.n_tropo_gradients + self.n_residual_ionosphere
    }

    const fn reduced_tropo_gradient_offset(&self) -> usize {
        3 + self.n_ztd
    }

    const fn clock_start(&self) -> usize {
        3
    }

    const fn clock_end(&self) -> usize {
        3 + self.n_epochs
    }

    const fn reduced_index(&self, full_idx: usize) -> Option<usize> {
        if full_idx < self.clock_start() {
            Some(full_idx)
        } else if full_idx < self.clock_end() {
            None
        } else {
            Some(full_idx - self.n_epochs)
        }
    }
}

/// Solve the PPP normal equations under the resolved [`NormalRecipe`]. The
/// validated runners flow the resolved recipe through here. For the PPP reference
/// recipe (`NormalRecipe::PppDenseLastTie`) each epoch-local receiver clock is
/// eliminated onto the static position, optional ZTD, and ambiguity states, then
/// the reduced system is solved last-tie and clocks are recovered by
/// back-substitution. This is algebraically equivalent to the unreduced dense
/// normal system and keeps day-length static arcs tractable. For the canonical
/// recipe (`NormalRecipe::CanonicalSquareRoot`) the same reduced system is
/// solved by the owned deterministic Cholesky square-root factorization;
/// canonical PPP is the only non-reference recipe that reaches this seam. Any
/// other recipe is a wiring error because no PPP strategy selects it.
pub(super) fn solve_normal_equations(
    rows: &[Row],
    layout: PppNormalLayout,
    normal: NormalRecipe,
) -> Result<Vec<f64>, FloatSolveError> {
    let assembler = NormalAssembler::new(normal);
    let weighted = || rows.iter().map(Row::as_weighted);
    let solution = match normal {
        NormalRecipe::PppDenseLastTie => solve_clock_eliminated_last_tie(rows, layout),
        NormalRecipe::CanonicalSquareRoot => solve_clock_eliminated_square_root(rows, layout),
        _ => assembler.solve_dense_last_tie(weighted(), layout.full_dim()),
    };
    solution.ok_or(FloatSolveError::SingularGeometry)
}

pub(super) fn clock_eliminated_normal_equations(
    rows: &[Row],
    layout: PppNormalLayout,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), FloatSolveError> {
    reduced_normal_equations(rows, layout).ok_or(FloatSolveError::SingularGeometry)
}

pub(super) fn ppp_position_covariance(
    rows: &[Row],
    layout: PppNormalLayout,
    position_m: [f64; 3],
) -> Result<PppPositionCovariance, FloatSolveError> {
    let (normal, _) = clock_eliminated_normal_equations(rows, layout)?;
    let ecef_m2 = position_covariance_ecef_m2(&normal)?;
    let tropo_gradient_formal_m2 = tropo_gradient_covariance_m2(&normal, layout)?;
    let receiver = ItrfPositionM::new(position_m[0], position_m[1], position_m[2])
        .map_err(|_| FloatSolveError::SingularGeometry)?;
    let geodetic = itrf_to_geodetic(receiver).map_err(|_| FloatSolveError::SingularGeometry)?;
    let mut enu_m2 = crate::dop::rotate_covariance_ecef_to_enu_m2(ecef_m2, geodetic)
        .map_err(|_| FloatSolveError::SingularGeometry)?;
    symmetrize_3x3(&mut enu_m2);
    let formal = crate::dop::PositionCovariance { ecef_m2, enu_m2 };
    let posterior_variance_factor = posterior_variance_factor(rows, layout)?;
    let covariance_scale_factor = posterior_variance_factor;
    let scaled = scale_position_covariance(formal, covariance_scale_factor);
    let tropo_gradient_scaled_m2 =
        tropo_gradient_formal_m2.map(|covariance| scale_2x2(covariance, covariance_scale_factor));
    Ok(PppPositionCovariance {
        scaled,
        formal,
        tropo_gradient_scaled_m2,
        tropo_gradient_formal_m2,
        posterior_variance_factor,
        covariance_scale_factor,
    })
}

pub(super) fn posterior_variance_factor(
    rows: &[Row],
    layout: PppNormalLayout,
) -> Result<f64, FloatSolveError> {
    let dof = rows
        .len()
        .checked_sub(layout.full_dim())
        .ok_or(FloatSolveError::SingularGeometry)?;
    if dof == 0 {
        return Err(FloatSolveError::SingularGeometry);
    }
    let weighted_ssr = rows.iter().try_fold(0.0_f64, |sum, row| {
        let weighted = row.y * row.weight;
        let contribution = weighted * weighted;
        if contribution.is_finite() {
            Some(sum + contribution)
        } else {
            None
        }
    });
    let Some(weighted_ssr) = weighted_ssr else {
        return Err(FloatSolveError::SingularGeometry);
    };
    let factor = weighted_ssr / dof as f64;
    if factor.is_finite() && factor >= 0.0 {
        Ok(factor)
    } else {
        Err(FloatSolveError::SingularGeometry)
    }
}

fn scale_position_covariance(
    covariance: crate::dop::PositionCovariance,
    factor: f64,
) -> crate::dop::PositionCovariance {
    crate::dop::PositionCovariance {
        ecef_m2: scale_3x3(covariance.ecef_m2, factor),
        enu_m2: scale_3x3(covariance.enu_m2, factor),
    }
}

fn scale_3x3(mut matrix: [[f64; 3]; 3], factor: f64) -> [[f64; 3]; 3] {
    for row in &mut matrix {
        for value in row {
            *value *= factor;
        }
    }
    matrix
}

fn scale_2x2(mut matrix: [[f64; 2]; 2], factor: f64) -> [[f64; 2]; 2] {
    for row in &mut matrix {
        for value in row {
            *value *= factor;
        }
    }
    matrix
}

fn position_covariance_ecef_m2(normal: &[Vec<f64>]) -> Result<[[f64; 3]; 3], FloatSolveError> {
    if normal.len() < 3 {
        return Err(FloatSolveError::SingularGeometry);
    }
    let mut position_normal = if normal.len() == 3 {
        submatrix(normal, 0, 3, 0, 3)
    } else {
        let nuisance = normal.len() - 3;
        let a = submatrix(normal, 0, 3, 0, 3);
        let b = submatrix(normal, 0, 3, 3, nuisance);
        let c = submatrix(normal, 3, nuisance, 3, nuisance);
        let b_t = transpose(&b).ok_or(FloatSolveError::SingularGeometry)?;
        let c_inv_b_t = solve_matrix_last_tie(&c, &b_t).ok_or(FloatSolveError::SingularGeometry)?;
        let b_c_inv_b_t = matmul(&b, &c_inv_b_t).ok_or(FloatSolveError::SingularGeometry)?;
        matrix_sub(&a, &b_c_inv_b_t).ok_or(FloatSolveError::SingularGeometry)?
    };
    symmetrize_square(&mut position_normal);
    let mut covariance =
        invert_symmetric_pd(&position_normal).ok_or(FloatSolveError::SingularGeometry)?;
    symmetrize_square(&mut covariance);
    let mut ecef_m2 = [
        [covariance[0][0], covariance[0][1], covariance[0][2]],
        [covariance[1][0], covariance[1][1], covariance[1][2]],
        [covariance[2][0], covariance[2][1], covariance[2][2]],
    ];
    symmetrize_3x3(&mut ecef_m2);
    Ok(ecef_m2)
}

fn tropo_gradient_covariance_m2(
    normal: &[Vec<f64>],
    layout: PppNormalLayout,
) -> Result<Option<[[f64; 2]; 2]>, FloatSolveError> {
    if layout.n_tropo_gradients == 0 {
        return Ok(None);
    }
    if layout.n_tropo_gradients != 2 {
        return Err(FloatSolveError::SingularGeometry);
    }
    let start = layout.reduced_tropo_gradient_offset();
    let covariance = covariance_block_from_normal(normal, start, 2)
        .map_err(|_| FloatSolveError::SingularGeometry)?;
    let mut out = [
        [covariance[0][0], covariance[0][1]],
        [covariance[1][0], covariance[1][1]],
    ];
    symmetrize_2x2(&mut out);
    Ok(Some(out))
}

fn symmetrize_square(matrix: &mut [Vec<f64>]) {
    for i in 0..matrix.len() {
        for j in (i + 1)..matrix.len() {
            let symmetric = 0.5 * (matrix[i][j] + matrix[j][i]);
            matrix[i][j] = symmetric;
            matrix[j][i] = symmetric;
        }
    }
}

fn symmetrize_3x3(matrix: &mut [[f64; 3]; 3]) {
    for i in 0..3 {
        for j in (i + 1)..3 {
            let symmetric = 0.5 * (matrix[i][j] + matrix[j][i]);
            matrix[i][j] = symmetric;
            matrix[j][i] = symmetric;
        }
    }
}

fn symmetrize_2x2(matrix: &mut [[f64; 2]; 2]) {
    let symmetric = 0.5 * (matrix[0][1] + matrix[1][0]);
    matrix[0][1] = symmetric;
    matrix[1][0] = symmetric;
}

fn solve_clock_eliminated_last_tie(rows: &[Row], layout: PppNormalLayout) -> Option<Vec<f64>> {
    let (normal, rhs) = reduced_normal_equations(rows, layout)?;
    let reduced = solve_linear_last_tie(normal, rhs)?;
    recover_clock_states(rows, layout, &reduced)
}

fn solve_clock_eliminated_square_root(rows: &[Row], layout: PppNormalLayout) -> Option<Vec<f64>> {
    let (normal, rhs) = reduced_normal_equations(rows, layout)?;
    let mut flat = Vec::with_capacity(layout.reduced_dim() * layout.reduced_dim());
    for row in &normal {
        flat.extend_from_slice(row);
    }
    let mut scratch = FlatCholeskySolveScratch::default();
    let reduced = solve_flat_normal_square_root_into(&flat, &rhs, &mut scratch)?;
    recover_clock_states(rows, layout, reduced)
}

fn reduced_normal_equations(
    rows: &[Row],
    layout: PppNormalLayout,
) -> Option<(Vec<Vec<f64>>, Vec<f64>)> {
    let full_dim = layout.full_dim();
    let reduced_dim = layout.reduced_dim();
    let mut normal = vec![vec![0.0_f64; reduced_dim]; reduced_dim];
    let mut rhs = vec![0.0_f64; reduced_dim];
    let mut clock_normal = vec![0.0_f64; layout.n_epochs];
    let mut clock_rhs = vec![0.0_f64; layout.n_epochs];
    let mut clock_cross = vec![vec![0.0_f64; reduced_dim]; layout.n_epochs];

    for row in rows {
        if row.h.len() != full_dim {
            return None;
        }
        let weighted_row = weighted_reduced_row(row, layout)?;
        let y = row.y * row.weight;

        for (i, reduced_i) in weighted_row.reduced.iter().copied().enumerate() {
            rhs[i] += reduced_i * y;
            for (cell, reduced_j) in normal[i].iter_mut().zip(&weighted_row.reduced) {
                *cell += reduced_i * *reduced_j;
            }
        }

        if let Some(clock_idx) = weighted_row.clock_idx {
            let clock_value = weighted_row.clock_value;
            clock_normal[clock_idx] += clock_value * clock_value;
            clock_rhs[clock_idx] += clock_value * y;
            for (cross, reduced_i) in clock_cross[clock_idx].iter_mut().zip(&weighted_row.reduced) {
                *cross += clock_value * *reduced_i;
            }
        }
    }

    for epoch_idx in 0..layout.n_epochs {
        let clock_inv = 1.0 / clock_normal[epoch_idx];
        if !clock_inv.is_finite() || clock_inv <= 0.0 {
            return None;
        }
        let cross = &clock_cross[epoch_idx];
        for (rhs_i, cross_i) in rhs.iter_mut().zip(cross) {
            *rhs_i -= *cross_i * clock_rhs[epoch_idx] * clock_inv;
        }
        for (normal_row, cross_i) in normal.iter_mut().zip(cross) {
            for (cell, cross_j) in normal_row.iter_mut().zip(cross) {
                *cell -= *cross_i * *cross_j * clock_inv;
            }
        }
    }

    if normal
        .iter()
        .flatten()
        .chain(rhs.iter())
        .any(|value| !value.is_finite())
    {
        return None;
    }
    Some((normal, rhs))
}

fn recover_clock_states(
    rows: &[Row],
    layout: PppNormalLayout,
    reduced: &[f64],
) -> Option<Vec<f64>> {
    if reduced.len() != layout.reduced_dim() {
        return None;
    }
    let mut clock_normal = vec![0.0_f64; layout.n_epochs];
    let mut clock_rhs = vec![0.0_f64; layout.n_epochs];
    let mut clock_cross = vec![vec![0.0_f64; layout.reduced_dim()]; layout.n_epochs];

    for row in rows {
        let weighted_row = weighted_reduced_row(row, layout)?;
        if let Some(clock_idx) = weighted_row.clock_idx {
            let clock_value = weighted_row.clock_value;
            clock_normal[clock_idx] += clock_value * clock_value;
            clock_rhs[clock_idx] += clock_value * row.y * row.weight;
            for (cross, reduced_i) in clock_cross[clock_idx].iter_mut().zip(&weighted_row.reduced) {
                *cross += clock_value * *reduced_i;
            }
        }
    }

    let mut clocks = vec![0.0_f64; layout.n_epochs];
    for epoch_idx in 0..layout.n_epochs {
        let mut projected = 0.0;
        for (value, delta) in clock_cross[epoch_idx].iter().zip(reduced) {
            projected += value * delta;
        }
        clocks[epoch_idx] = (clock_rhs[epoch_idx] - projected) / clock_normal[epoch_idx];
        if !clocks[epoch_idx].is_finite() {
            return None;
        }
    }

    let mut full = vec![0.0_f64; layout.full_dim()];
    full[..3].copy_from_slice(&reduced[..3]);
    full[3..3 + layout.n_epochs].copy_from_slice(&clocks);
    let reduced_tail_start = 3;
    let full_tail_start = 3 + layout.n_epochs;
    full[full_tail_start..].copy_from_slice(&reduced[reduced_tail_start..]);
    Some(full)
}

struct WeightedReducedRow {
    reduced: Vec<f64>,
    clock_idx: Option<usize>,
    clock_value: f64,
}

fn weighted_reduced_row(row: &Row, layout: PppNormalLayout) -> Option<WeightedReducedRow> {
    if row.h.len() != layout.full_dim() || !row.y.is_finite() || !row.weight.is_finite() {
        return None;
    }
    let mut reduced = vec![0.0_f64; layout.reduced_dim()];
    let mut clock_idx = None;
    let mut clock_value = 0.0_f64;
    for (full_idx, value) in row.h.iter().copied().enumerate() {
        if !value.is_finite() {
            return None;
        }
        let weighted = value * row.weight;
        if let Some(reduced_idx) = layout.reduced_index(full_idx) {
            reduced[reduced_idx] = weighted;
        } else if weighted != 0.0 {
            if clock_idx.is_some() {
                return None;
            }
            clock_idx = Some(full_idx - layout.clock_start());
            clock_value = weighted;
        }
    }
    Some(WeightedReducedRow {
        reduced,
        clock_idx,
        clock_value,
    })
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
    covariance_block_from_normal(normal, start, n_ambiguities)
        .map_err(|_| FixedSolveError::Float(FloatSolveError::SingularGeometry))
}

fn covariance_block_from_normal(
    normal: &[Vec<f64>],
    start: usize,
    count: usize,
) -> Result<Vec<Vec<f64>>, ()> {
    if count == 0 || start + count > normal.len() {
        return Err(());
    }
    for row in normal {
        if row.len() != normal.len() {
            return Err(());
        }
    }
    let other = (0..normal.len())
        .filter(|idx| *idx < start || *idx >= start + count)
        .collect::<Vec<_>>();
    let block = (start..start + count).collect::<Vec<_>>();
    if other.is_empty() {
        return invert_matrix_last_tie(&indexed_submatrix(normal, &block, &block)).ok_or(());
    }
    let a = indexed_submatrix(normal, &other, &other);
    let b = indexed_submatrix(normal, &other, &block);
    let c = indexed_submatrix(normal, &block, &block);
    let a_inv_b = solve_matrix_last_tie(&a, &b).ok_or(())?;
    let b_t = transpose(&b).ok_or(())?;
    let bt_a_inv_b = matmul(&b_t, &a_inv_b).ok_or(())?;
    let mut schur = matrix_sub(&c, &bt_a_inv_b).ok_or(())?;
    symmetrize_square(&mut schur);
    invert_matrix_last_tie(&schur).ok_or(())
}

fn indexed_submatrix(matrix: &[Vec<f64>], rows: &[usize], cols: &[usize]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|&row| cols.iter().map(|&col| matrix[row][col]).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ppp_clock_elimination_matches_unreduced_dense_solve() {
        let n_epochs = 10;
        let n_ambiguities = 6;
        let layout = PppNormalLayout::new(n_epochs, 1, 0, 0, n_ambiguities);
        let rows = ppp_schur_rows(n_epochs, n_ambiguities);
        let dense = NormalAssembler::new(NormalRecipe::PppDenseLastTie)
            .solve_dense_last_tie(rows.iter().map(Row::as_weighted), layout.full_dim())
            .expect("unreduced dense PPP normal solves");
        let reduced = solve_normal_equations(&rows, layout, NormalRecipe::PppDenseLastTie)
            .expect("clock-eliminated PPP normal solves");
        let max_abs = dense
            .iter()
            .zip(&reduced)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("PPP clock elimination equivalence max_abs={max_abs:.3e} m");
        assert!(
            max_abs < 2.0e-8,
            "clock-eliminated solution differed from unreduced dense by {max_abs} m"
        );
    }

    #[test]
    fn ppp_position_covariance_matches_unreduced_dense_inverse() {
        let n_epochs = 10;
        let n_ambiguities = 6;
        let layout = PppNormalLayout::new(n_epochs, 1, 0, 0, n_ambiguities);
        let rows = ppp_schur_rows(n_epochs, n_ambiguities);

        // Dense oracle: assemble the full unreduced normal matrix, invert it,
        // and read the position block straight out of the inverse. The reduced
        // path must reproduce this scale, not merely stay positive definite.
        // Row weights are inverse SIGMAS (see MeasurementWeights), so the
        // information contribution of a row is weight^2 * h * h^T.
        let dim = layout.full_dim();
        let mut normal = vec![vec![0.0_f64; dim]; dim];
        for row in &rows {
            for i in 0..dim {
                for j in 0..dim {
                    normal[i][j] += row.weight * row.weight * row.h[i] * row.h[j];
                }
            }
        }
        let dense_inverse = invert_symmetric_pd(&normal).expect("dense normal inverts");

        let position_m = [4_075_580.0, 931_854.0, 4_801_568.0];
        let covariance =
            ppp_position_covariance(&rows, layout, position_m).expect("reduced covariance");
        assert_eq!(
            covariance.covariance_scale_factor.to_bits(),
            covariance.posterior_variance_factor.to_bits()
        );

        for i in 0..3 {
            for j in 0..3 {
                let oracle = dense_inverse[i][j];
                let got = covariance.formal.ecef_m2[i][j];
                let scale = oracle.abs().max(1.0e-12);
                assert!(
                    ((got - oracle) / scale).abs() < 1.0e-6,
                    "covariance [{i}][{j}] = {got} differs from dense inverse {oracle}"
                );
                let scaled = covariance.scaled.ecef_m2[i][j];
                let expected_scaled = got * covariance.covariance_scale_factor;
                let scaled_scale = expected_scaled.abs().max(1.0e-12);
                assert!(
                    ((scaled - expected_scaled) / scaled_scale).abs() < 1.0e-12,
                    "scaled covariance [{i}][{j}] = {scaled} differs from formal * factor {expected_scaled}"
                );
            }
        }
    }

    #[test]
    fn ppp_posterior_variance_factor_counts_eliminated_clocks() {
        let n_epochs = 10;
        let n_ambiguities = 6;
        let layout = PppNormalLayout::new(n_epochs, 1, 0, 0, n_ambiguities);
        let rows = ppp_schur_rows(n_epochs, n_ambiguities);
        let weighted_ssr = rows
            .iter()
            .map(|row| {
                let weighted = row.y * row.weight;
                weighted * weighted
            })
            .sum::<f64>();
        let dof = rows.len() - layout.full_dim();
        let factor = posterior_variance_factor(&rows, layout).expect("posterior variance factor");
        assert_eq!(dof, rows.len() - (3 + n_epochs + 1 + n_ambiguities));
        assert_eq!(factor.to_bits(), (weighted_ssr / dof as f64).to_bits());
    }

    fn ppp_schur_rows(n_epochs: usize, n_ambiguities: usize) -> Vec<Row> {
        let los = [
            [-0.72, 0.11, -0.68],
            [0.48, -0.74, -0.47],
            [-0.24, -0.61, 0.75],
            [0.66, 0.38, -0.64],
            [-0.53, 0.79, 0.30],
            [0.18, 0.54, 0.82],
        ];
        let mut rows = Vec::new();
        for epoch_idx in 0..n_epochs {
            for amb_idx in 0..n_ambiguities {
                let mut code = vec![0.0_f64; 3 + n_epochs + 1 + n_ambiguities];
                code[..3].copy_from_slice(&los[amb_idx % los.len()]);
                code[3 + epoch_idx] = 1.0;
                code[3 + n_epochs] = 1.1 + 0.03 * amb_idx as f64;
                rows.push(Row {
                    h: code.clone(),
                    y: 0.2 * epoch_idx as f64 - 0.05 * amb_idx as f64,
                    weight: 1.0 + 0.01 * amb_idx as f64,
                });

                let mut phase = code;
                phase[3 + n_epochs + 1 + amb_idx] = 1.0;
                rows.push(Row {
                    h: phase,
                    y: -0.1 * epoch_idx as f64 + 0.07 * amb_idx as f64,
                    weight: 30.0 + amb_idx as f64,
                });
            }
        }
        rows
    }
}
