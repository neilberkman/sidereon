//! Static batch float RTK baseline solver.
//!
//! The iterated double-difference least-squares solve that recovers the float
//! baseline and single-difference float ambiguities from a set of normalized
//! epochs, plus its Schur-reduced ambiguity covariance and post-fit residuals.
//! This is the static counterpart of the sequential `update` filter: the
//! measurement model is the shared [`super::rows::dd_epoch_rows_into`] builder
//! (driven with [`DdRowRecipe::FloatBaseline`]), but the solve accumulates every
//! epoch into one normal system rather than folding epoch-to-epoch.
//!
//! The fixed-baseline solver in the parent reuses this module's scratch buffer,
//! normal-equation assembly, and residual reconstruction (`FloatSolveScratch`,
//! `float_normal_equations`, `float_residuals`), so those are `pub(super)`; the
//! iterated solve itself is `pub` because the Sidereon NIF and the validated fixed
//! solve call it directly.
//! Operation order reproduces the Elixir reference for the frozen-bits golden.

use crate::astro::math::linear::{
    invert_flat_first_tie_into, solve_matrix_flat_first_tie_into, FlatCholeskySolveScratch,
    FlatLinearScratch as InvertFlatScratch, FlatNormalSolveScratch as SolveNormalScratch,
};
use crate::astro::math::vec3;
use crate::estimation::recipe::{EstimationRecipe, NormalRecipe, ResidualNormRecipe, SolverRecipe};
use crate::estimation::substrate::normal::NormalAssembler;
use crate::estimation::substrate::parameters::ParameterLayout;
use crate::estimation::substrate::qc::normalized_residual;
use crate::validate;

use super::{
    dd_epoch_rows_into, fold_measurement_block_indices, rms, BlockFoldScratch, DdRowError,
    DdRowRecipe, DdRowScratch, Epoch, EpochRowsScratch, MeasContext, MeasModel,
    ReceiverAntennaCorrections, ReceiverAntennaError, RowKind,
};

/// Terminal status of the static batch float RTK solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FloatSolveStatus {
    StateTolerance,
    MaxIterations,
}

/// Controls for the static batch float RTK solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FloatSolveOpts {
    pub position_tol_m: f64,
    pub ambiguity_tol_m: f64,
    pub max_iterations: usize,
}

impl Default for FloatSolveOpts {
    /// Canonical float-solve controls, read from [`super::defaults`]. This is the
    /// single source of truth bindings construct from instead of hardcoding
    /// literals; it does not change any solve, which still reads the caller's
    /// opts.
    fn default() -> Self {
        Self {
            position_tol_m: super::defaults::POSITION_TOL_M,
            ambiguity_tol_m: super::defaults::AMBIGUITY_TOL_M,
            max_iterations: super::defaults::MAX_ITERATIONS,
        }
    }
}

/// One public residual row from the static batch float solve.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatResidual {
    pub epoch_index: usize,
    pub satellite_id: String,
    pub reference_satellite_id: String,
    pub ambiguity_id: String,
    pub code_m: f64,
    pub phase_m: f64,
    pub code_sigma_m: f64,
    pub phase_sigma_m: f64,
    pub code_normalized: f64,
    pub phase_normalized: f64,
}

/// Static batch float RTK solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatBaselineSolution {
    pub baseline_m: [f64; 3],
    pub ambiguities_m: Vec<(String, f64)>,
    pub ambiguity_covariance_m: Vec<f64>,
    pub ambiguity_covariance_inverse_m: Vec<f64>,
    pub residuals: Vec<FloatResidual>,
    pub iterations: usize,
    pub converged: bool,
    pub status: FloatSolveStatus,
    pub code_rms_m: f64,
    pub phase_rms_m: f64,
    pub weighted_rms_m: f64,
    pub n_observations: usize,
}

/// Why the static batch float RTK solve could not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FloatSolveError {
    MissingSystemReference(String),
    MissingAmbiguityColumn(String),
    InvalidInput {
        field: &'static str,
        kind: super::RtkInputErrorKind,
    },
    SingularGeometry,
    IncompleteResidualPair,
    ReceiverAntenna(ReceiverAntennaError),
}

impl core::fmt::Display for FloatSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MissingSystemReference(system) => write!(
                f,
                "missing RTK reference satellite for constellation {system}"
            ),
            Self::MissingAmbiguityColumn(id) => {
                write!(f, "missing RTK ambiguity column {id}")
            }
            Self::InvalidInput { field, kind } => {
                write!(f, "invalid RTK float input {field}: {kind}")
            }
            Self::SingularGeometry => write!(f, "RTK float geometry is singular"),
            Self::IncompleteResidualPair => {
                write!(
                    f,
                    "RTK float residual rows are not complete code/phase pairs"
                )
            }
            Self::ReceiverAntenna(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FloatSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReceiverAntenna(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct FloatState {
    baseline_m: [f64; 3],
    ambiguities_m: Vec<f64>,
}

/// The converged solver state plus the iteration bookkeeping that the static
/// `float` and `fixed` `finalize_*` builders turn into a solution. Bundled so
/// the finalize argument lists stay small; generic over the solver's state type
/// (`FloatState`/`FixedState`).
pub(super) struct SolveOutcome<S> {
    pub(super) state: S,
    pub(super) iterations: usize,
    pub(super) converged: bool,
    pub(super) status: FloatSolveStatus,
}

/// Reusable buffers for the iterated batch solve, shared with the fixed solver.
#[derive(Debug, Default)]
pub(super) struct FloatSolveScratch {
    pub(super) epoch_rows: EpochRowsScratch,
    pub(super) rows: Vec<DdRowScratch>,
    block_indices: Vec<usize>,
    block_rows: Vec<usize>,
    fold_block: BlockFoldScratch,
    pub(super) solve: SolveNormalScratch,
    sqrt_solve: FlatCholeskySolveScratch,
    pub(super) lambda: Vec<f64>,
    pub(super) eta: Vec<f64>,
    solve_matrix: InvertFlatScratch,
    cov_invert: InvertFlatScratch,
    cov_inv_invert: InvertFlatScratch,
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    a_inv_b: Vec<f64>,
    bt_a_inv_b: Vec<f64>,
    schur: Vec<f64>,
    covariance: Vec<f64>,
}

/// Solve a static multi-epoch float RTK baseline from normalized
/// double-difference epochs. Public Elixir callers own input normalization and
/// metadata shape; the numerical model, iterated least squares, covariance, and
/// residuals live here.
pub fn solve_float_baseline(
    epochs: &[Epoch],
    base: [f64; 3],
    ambiguity_ids: &[String],
    initial_baseline_m: [f64; 3],
    model: &MeasModel,
    opts: FloatSolveOpts,
    receiver_antenna_corrections: Option<&ReceiverAntennaCorrections>,
) -> Result<FloatBaselineSolution, FloatSolveError> {
    use crate::estimation::recipe::StrategyId;
    use crate::estimation::strategies::{
        estimate, EstimateError, EstimateInput, EstimateOptions, EstimateOutput,
    };
    match estimate(
        EstimateInput::RtkFloat {
            epochs,
            base,
            ambiguity_ids,
            initial_baseline_m,
            model,
            opts,
            receiver_antenna_corrections,
        },
        EstimateOptions::new(StrategyId::rtk_reference()),
    ) {
        Ok(EstimateOutput::RtkFloat(solution)) => Ok(*solution),
        Err(EstimateError::RtkFloat(error)) => Err(error),
        Ok(_) | Err(_) => {
            unreachable!(
                "the RTK reference strategy yields an RTK float solution or an RTK float error"
            )
        }
    }
}

/// Drive the static float RTK baseline from a resolved [`EstimationRecipe`]: the
/// shared per-technique implementation that
/// [`crate::estimation::strategies::estimate`] dispatches to. The recipe's
/// [`NormalRecipe`] selects the normal-equation assembler used in the solve loop;
/// for the RTK reference recipe (`recipe.normal ==
/// NormalRecipe::RtkDoubleDifferenceBlockFirstTie`) this is bit-identical to the
/// legacy path.
pub(crate) fn run_float(
    recipe: &EstimationRecipe,
    ctx: MeasContext<'_>,
    epochs: &[Epoch],
    ambiguity_ids: &[String],
    initial_baseline_m: [f64; 3],
    opts: FloatSolveOpts,
) -> Result<FloatBaselineSolution, FloatSolveError> {
    let mut scratch = FloatSolveScratch::default();
    solve_float_baseline_with_scratch(
        ctx,
        epochs,
        ambiguity_ids,
        initial_baseline_m,
        opts,
        recipe,
        &mut scratch,
    )
}

fn float_input_error(error: validate::FieldError) -> FloatSolveError {
    FloatSolveError::InvalidInput {
        field: error.field(),
        kind: super::RtkInputErrorKind::from(&error),
    }
}

fn invalid_float_option(field: &'static str, kind: super::RtkInputErrorKind) -> FloatSolveError {
    FloatSolveError::InvalidInput { field, kind }
}

pub(super) fn validate_float_solve_opts(opts: FloatSolveOpts) -> Result<(), FloatSolveError> {
    validate::finite_positive(opts.position_tol_m, "rtk.float.position_tol_m")
        .map_err(float_input_error)?;
    validate::finite_positive(opts.ambiguity_tol_m, "rtk.float.ambiguity_tol_m")
        .map_err(float_input_error)?;
    if opts.max_iterations == 0 {
        return Err(invalid_float_option(
            "rtk.float.max_iterations",
            super::RtkInputErrorKind::NotPositive,
        ));
    }
    Ok(())
}

/// Solve the accumulated double-difference information system `Λ x = η` under the
/// resolved `(normal, solver)` recipe stages, copying the solution into `dx`.
/// The RTK reference recipe
/// ([`NormalRecipe::RtkDoubleDifferenceBlockFirstTie`]) solves it by first-tie
/// Gaussian elimination; canonical RTK
/// ([`NormalRecipe::CanonicalSquareRoot`] with
/// [`SolverRecipe::OwnedDeterministicCholesky`]) solves the SAME system by the
/// owned deterministic Cholesky square-root factorization. Both arms own the
/// solve, so selecting the recipe never touches the row assembly.
pub(super) fn solve_dd_normal_into(
    normal: NormalRecipe,
    solver: SolverRecipe,
    scratch: &mut FloatSolveScratch,
    dx: &mut Vec<f64>,
) -> Option<()> {
    let solution = match (normal, solver) {
        (NormalRecipe::CanonicalSquareRoot, SolverRecipe::OwnedDeterministicCholesky) => {
            NormalAssembler::new(normal).solve_square_root(
                &scratch.lambda,
                &scratch.eta,
                &mut scratch.sqrt_solve,
            )?
        }
        _ => NormalAssembler::new(normal).solve_flat_first_tie(
            &scratch.lambda,
            &scratch.eta,
            &mut scratch.solve,
        )?,
    };
    dx.clear();
    dx.extend_from_slice(solution);
    Some(())
}

fn solve_float_baseline_with_scratch(
    ctx: MeasContext,
    epochs: &[Epoch],
    ambiguity_ids: &[String],
    initial_baseline_m: [f64; 3],
    opts: FloatSolveOpts,
    recipe: &EstimationRecipe,
    scratch: &mut FloatSolveScratch,
) -> Result<FloatBaselineSolution, FloatSolveError> {
    validate_float_solve_opts(opts)?;
    let mut state = FloatState {
        baseline_m: initial_baseline_m,
        ambiguities_m: vec![0.0; ambiguity_ids.len()],
    };

    let mut dx: Vec<f64> = Vec::new();
    for iter in 1..=opts.max_iterations.max(1) {
        build_float_rows(ctx, epochs, ambiguity_ids, &state, scratch)?;
        float_normal_equations(ParameterLayout::rtk(ambiguity_ids.len()).dim(), scratch)
            .ok_or(FloatSolveError::SingularGeometry)?;
        solve_dd_normal_into(recipe.normal, recipe.solver, scratch, &mut dx)
            .ok_or(FloatSolveError::SingularGeometry)?;

        for (k, value) in state.baseline_m.iter_mut().enumerate() {
            *value += dx[k];
        }
        for (idx, value) in state.ambiguities_m.iter_mut().enumerate() {
            *value += dx[3 + idx];
        }

        let baseline_step = vec3::norm3([dx[0], dx[1], dx[2]]);
        let ambiguity_step = dx[3..].iter().fold(0.0_f64, |m, &v| m.max(v.abs()));

        if baseline_step <= opts.position_tol_m && ambiguity_step <= opts.ambiguity_tol_m {
            return finalize_float_baseline(
                ctx,
                epochs,
                ambiguity_ids,
                SolveOutcome {
                    state,
                    iterations: iter,
                    converged: true,
                    status: FloatSolveStatus::StateTolerance,
                },
                scratch,
            );
        }

        if iter >= opts.max_iterations.max(1) {
            return finalize_float_baseline(
                ctx,
                epochs,
                ambiguity_ids,
                SolveOutcome {
                    state,
                    iterations: iter,
                    converged: false,
                    status: FloatSolveStatus::MaxIterations,
                },
                scratch,
            );
        }
    }

    Err(FloatSolveError::SingularGeometry)
}

fn finalize_float_baseline(
    ctx: MeasContext,
    epochs: &[Epoch],
    ambiguity_ids: &[String],
    outcome: SolveOutcome<FloatState>,
    scratch: &mut FloatSolveScratch,
) -> Result<FloatBaselineSolution, FloatSolveError> {
    let SolveOutcome {
        state,
        iterations,
        converged,
        status,
    } = outcome;
    build_float_rows(ctx, epochs, ambiguity_ids, &state, scratch)?;
    float_normal_equations(ParameterLayout::rtk(ambiguity_ids.len()).dim(), scratch)
        .ok_or(FloatSolveError::SingularGeometry)?;
    let normal = scratch.lambda.clone();
    let ambiguity_covariance_m =
        ambiguity_covariance_from_normal_flat(&normal, ambiguity_ids.len(), scratch)
            .ok_or(FloatSolveError::SingularGeometry)?;
    let mut ambiguity_covariance_inverse_m = Vec::new();
    invert_flat_first_tie_into(
        &ambiguity_covariance_m,
        ambiguity_ids.len(),
        &mut ambiguity_covariance_inverse_m,
        &mut scratch.cov_inv_invert,
    )
    .ok_or(FloatSolveError::SingularGeometry)?;
    let residuals = float_residuals(&scratch.rows)?;
    let code_rms_m = rms(residuals.iter().map(|r| r.code_m));
    let phase_rms_m = rms(residuals.iter().map(|r| r.phase_m));
    let weighted_rms_m = rms(scratch.rows.iter().map(|r| r.y * r.weight));

    Ok(FloatBaselineSolution {
        baseline_m: state.baseline_m,
        ambiguities_m: ambiguity_ids
            .iter()
            .cloned()
            .zip(state.ambiguities_m)
            .collect(),
        ambiguity_covariance_m,
        ambiguity_covariance_inverse_m,
        residuals,
        iterations,
        converged,
        status,
        code_rms_m,
        phase_rms_m,
        weighted_rms_m,
        n_observations: scratch.rows.len(),
    })
}

/// Map a shared-builder error onto the float solve's error surface.
fn float_row_error(error: DdRowError) -> FloatSolveError {
    match error {
        DdRowError::MissingReference(system) => FloatSolveError::MissingSystemReference(system),
        DdRowError::MissingAmbiguity(id) => FloatSolveError::MissingAmbiguityColumn(id),
        DdRowError::ReceiverAntenna(error) => FloatSolveError::ReceiverAntenna(error),
        DdRowError::InvalidInput { field, kind } => FloatSolveError::InvalidInput { field, kind },
    }
}

/// Owned snapshot of the float double-difference rows for one epoch, linearized
/// at `baseline_m`/`ambiguities_m`. Test-only wrapper over the shared
/// [`super::rows::dd_epoch_rows_into`] builder that drives the float row-level
/// golden trace; `FloatState`'s fields are private to this module, so the
/// construction lives here rather than in the test module.
#[cfg(test)]
pub(super) fn float_epoch_rows(
    base: [f64; 3],
    epoch: &Epoch,
    ambiguity_ids: &[String],
    baseline_m: [f64; 3],
    ambiguities_m: Vec<f64>,
    model: &MeasModel,
) -> Result<Vec<super::rows::DdRow>, FloatSolveError> {
    let state = FloatState {
        baseline_m,
        ambiguities_m,
    };
    let ctx = MeasContext {
        base,
        model,
        antenna: None,
    };
    let mut scratch = EpochRowsScratch::default();
    let rows = dd_epoch_rows_into(
        ctx,
        epoch,
        0,
        state.baseline_m,
        DdRowRecipe::FloatBaseline {
            ambiguity_ids,
            ambiguities_m: &state.ambiguities_m,
        },
        &mut scratch,
    )
    .map_err(float_row_error)?;
    Ok(rows.iter().map(super::rows::DdRow::from_scratch).collect())
}

fn build_float_rows(
    ctx: MeasContext,
    epochs: &[Epoch],
    ambiguity_ids: &[String],
    state: &FloatState,
    scratch: &mut FloatSolveScratch,
) -> Result<(), FloatSolveError> {
    scratch.rows.clear();
    for (epoch_index, epoch) in epochs.iter().enumerate() {
        let rows = dd_epoch_rows_into(
            ctx,
            epoch,
            epoch_index,
            state.baseline_m,
            DdRowRecipe::FloatBaseline {
                ambiguity_ids,
                ambiguities_m: &state.ambiguities_m,
            },
            &mut scratch.epoch_rows,
        )
        .map_err(float_row_error)?;
        scratch.rows.extend(rows.iter().cloned());
    }
    Ok(())
}

pub(super) fn float_normal_equations(n: usize, scratch: &mut FloatSolveScratch) -> Option<()> {
    scratch.lambda.resize(n * n, 0.0);
    scratch.lambda.fill(0.0);
    scratch.eta.resize(n, 0.0);
    scratch.eta.fill(0.0);

    scratch.block_indices.clear();
    scratch.block_indices.extend(0..scratch.rows.len());
    scratch.block_indices.sort_by(|&a, &b| {
        (
            scratch.rows[a].epoch_index,
            scratch.rows[a].kind,
            scratch.rows[a].ref_sat.as_str(),
        )
            .cmp(&(
                scratch.rows[b].epoch_index,
                scratch.rows[b].kind,
                scratch.rows[b].ref_sat.as_str(),
            ))
    });

    let mut start = 0;
    while start < scratch.block_indices.len() {
        let first = scratch.block_indices[start];
        let epoch_index = scratch.rows[first].epoch_index;
        let kind = scratch.rows[first].kind;
        let ref_sat = scratch.rows[first].ref_sat.as_str();
        let mut end = start + 1;
        while end < scratch.block_indices.len() {
            let idx = scratch.block_indices[end];
            if scratch.rows[idx].epoch_index != epoch_index
                || scratch.rows[idx].kind != kind
                || scratch.rows[idx].ref_sat != ref_sat
            {
                break;
            }
            end += 1;
        }
        scratch.block_rows.clear();
        scratch
            .block_rows
            .extend_from_slice(&scratch.block_indices[start..end]);
        scratch
            .block_rows
            .sort_by(|&a, &b| scratch.rows[a].sat.cmp(&scratch.rows[b].sat));
        fold_measurement_block_indices(
            &mut scratch.lambda,
            &mut scratch.eta,
            &scratch.rows,
            &scratch.block_rows,
            &mut scratch.fold_block,
        )?;
        start = end;
    }
    Some(())
}

fn ambiguity_covariance_from_normal_flat(
    normal: &[f64],
    n_ambiguities: usize,
    scratch: &mut FloatSolveScratch,
) -> Option<Vec<f64>> {
    let n = 3 + n_ambiguities;
    scratch.a.resize(9, 0.0);
    scratch.b.resize(3 * n_ambiguities, 0.0);
    scratch.c.resize(n_ambiguities * n_ambiguities, 0.0);

    for i in 0..3 {
        for j in 0..3 {
            scratch.a[i * 3 + j] = normal[i * n + j];
        }
        for j in 0..n_ambiguities {
            scratch.b[i * n_ambiguities + j] = normal[i * n + 3 + j];
        }
    }
    for i in 0..n_ambiguities {
        for j in 0..n_ambiguities {
            scratch.c[i * n_ambiguities + j] = normal[(3 + i) * n + 3 + j];
        }
    }

    solve_matrix_flat_first_tie_into(
        &scratch.a,
        3,
        &scratch.b,
        n_ambiguities,
        &mut scratch.a_inv_b,
        &mut scratch.solve_matrix,
    )?;

    scratch
        .bt_a_inv_b
        .resize(n_ambiguities * n_ambiguities, 0.0);
    for i in 0..n_ambiguities {
        for j in 0..n_ambiguities {
            let mut acc = 0.0;
            for k in 0..3 {
                acc += scratch.b[k * n_ambiguities + i] * scratch.a_inv_b[k * n_ambiguities + j];
            }
            scratch.bt_a_inv_b[i * n_ambiguities + j] = acc;
        }
    }

    scratch.schur.resize(n_ambiguities * n_ambiguities, 0.0);
    for i in 0..(n_ambiguities * n_ambiguities) {
        scratch.schur[i] = scratch.c[i] - scratch.bt_a_inv_b[i];
    }

    invert_flat_first_tie_into(
        &scratch.schur,
        n_ambiguities,
        &mut scratch.covariance,
        &mut scratch.cov_invert,
    )?;
    Some(scratch.covariance.clone())
}

pub(super) fn float_residuals(
    rows: &[DdRowScratch],
) -> Result<Vec<FloatResidual>, FloatSolveError> {
    let mut residuals = Vec::with_capacity(rows.len() / 2);
    let mut idx = 0;
    while idx < rows.len() {
        let code = rows
            .get(idx)
            .ok_or(FloatSolveError::IncompleteResidualPair)?;
        let phase = rows
            .get(idx + 1)
            .ok_or(FloatSolveError::IncompleteResidualPair)?;
        if code.kind != RowKind::Code
            || phase.kind != RowKind::Phase
            || code.epoch_index != phase.epoch_index
            || code.sat != phase.sat
            || code.ref_sat != phase.ref_sat
            || code.ambiguity_id != phase.ambiguity_id
        {
            return Err(FloatSolveError::IncompleteResidualPair);
        }
        residuals.push(FloatResidual {
            epoch_index: code.epoch_index,
            satellite_id: code.sat.clone(),
            reference_satellite_id: code.ref_sat.clone(),
            ambiguity_id: code.ambiguity_id.as_str().to_string(),
            code_m: code.y,
            phase_m: phase.y,
            code_sigma_m: 1.0 / code.weight,
            phase_sigma_m: 1.0 / phase.weight,
            code_normalized: normalized_residual(
                ResidualNormRecipe::RtkInverseSigmaResidual,
                code.y,
                code.weight,
            ),
            phase_normalized: normalized_residual(
                ResidualNormRecipe::RtkInverseSigmaResidual,
                phase.y,
                phase.weight,
            ),
        });
        idx += 2;
    }
    Ok(residuals)
}
