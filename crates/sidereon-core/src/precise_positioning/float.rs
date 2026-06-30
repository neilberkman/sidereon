//! Static multi-epoch float PPP solve and the iterated Gauss-Newton update.
//!
//! This leaf owns the float-only orchestration: the public multi-epoch and
//! single-epoch entry points, the iterated normal-equation solve, the design
//! rows and state delta, the post-fit residual rows, and the leave-one-out
//! residual screening loop. The shared measurement model lives in
//! [`super::model`], the dense normal-equation kernel in [`super::normal`], and
//! the row staging / shared scalar helpers in [`super`].

use std::collections::{BTreeMap, BTreeSet};

use crate::ambiguity::AmbiguityId;
use crate::astro::math::vec3;
use crate::estimation::recipe::{EstimationRecipe, NormalRecipe, ResidualNormRecipe};
use crate::estimation::substrate::parameters::ParameterLayout;
use crate::estimation::substrate::qc::normalized_residual;
use crate::observables::ObservableEphemerisSource;

use super::normal::solve_normal_equations;
use super::rows::{build_rows, residual_rows, AmbiguityBinding, PppRowError};
use super::{
    estimates_ztd, max_abs, rms, state_from_solution, validate_float_solution_output,
    validate_float_solve_boundary, weighted_rms, ztd_unknown_count, FloatEpoch, FloatSolution,
    FloatSolveConfig, FloatSolveError, FloatSolveOptions, FloatState, FloatStatus, ModelContext,
    TroposphereOptions,
};

const RESIDUAL_SCREEN_THRESHOLD: f64 = 4.0;
const RESIDUAL_SCREEN_MAX_PASSES: usize = 8;
const RESIDUAL_SCREEN_ACCEPT_FACTOR: f64 = 2.0;
const SINGLE_EPOCH_AMBIGUITY_TOLERANCE_M: f64 = f64::MAX;

/// Solve a static multi-epoch float PPP arc.
pub fn solve_float_epochs(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    initial_state: FloatState,
    config: FloatSolveConfig,
) -> Result<FloatSolution, FloatSolveError> {
    validate_float_solve_boundary(epochs, &initial_state, &config)?;
    use crate::estimation::recipe::StrategyId;
    use crate::estimation::strategies::{
        estimate, EstimateError, EstimateInput, EstimateOptions, EstimateOutput,
    };
    match estimate(
        EstimateInput::PppFloat {
            source,
            epochs,
            initial_state,
            config,
        },
        EstimateOptions::new(StrategyId::ppp_reference()),
    ) {
        Ok(EstimateOutput::PppFloat(solution)) => Ok(*solution),
        Err(EstimateError::PppFloat(error)) => Err(error),
        Ok(_) | Err(_) => {
            unreachable!(
                "the PPP reference strategy yields a PPP float solution or a PPP float error"
            )
        }
    }
}

/// Drive the static float PPP arc from a resolved [`EstimationRecipe`]: the shared
/// per-technique implementation that
/// [`crate::estimation::strategies::estimate`] dispatches to. The recipe's
/// [`NormalRecipe`] reaches the solve seam through [`ModelContext::normal`]; for
/// the PPP reference recipe (`NormalRecipe::PppDenseLastTie`) this is
/// bit-identical to the legacy path.
pub(crate) fn run_float_epochs(
    recipe: &EstimationRecipe,
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    initial_state: FloatState,
    config: FloatSolveConfig,
) -> Result<FloatSolution, FloatSolveError> {
    solve_float_multi_screened(source, epochs, initial_state, config, recipe.normal)
}

/// Solve one float PPP epoch with the same state shape as Sidereon' historical
/// single-epoch API: receiver position, one receiver clock, and one ambiguity
/// per observation.
pub fn solve_float_epoch(
    source: &dyn ObservableEphemerisSource,
    epoch: FloatEpoch,
    initial_state: FloatState,
    mut config: FloatSolveConfig,
) -> Result<FloatSolution, FloatSolveError> {
    let epochs = [epoch];
    validate_float_solve_boundary(&epochs, &initial_state, &config)?;
    let ambiguity_ids = epochs[0]
        .observations
        .iter()
        .map(|obs| AmbiguityId::new(obs.ambiguity_id.clone()))
        .collect::<Vec<_>>();
    config.opts.ambiguity_tolerance_m = SINGLE_EPOCH_AMBIGUITY_TOLERANCE_M;
    let ctx = ModelContext {
        source,
        weights: config.weights,
        tropo: config.tropo,
        corrections: &config.corrections,
        normal: NormalRecipe::PppDenseLastTie,
    };
    iterate_multi(ctx, &epochs, &ambiguity_ids, initial_state, config.opts, 1)
}

fn solve_float_multi_screened(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    state: FloatState,
    config: FloatSolveConfig,
    normal: NormalRecipe,
) -> Result<FloatSolution, FloatSolveError> {
    validate_float_solve_boundary(epochs, &state, &config)?;
    let FloatSolveConfig {
        weights,
        tropo,
        corrections,
        opts,
        residual_screen,
    } = config;
    let ctx = ModelContext {
        source,
        weights,
        tropo,
        corrections: &corrections,
        normal,
    };
    let ambiguity_ids = multi_ambiguity_ids(epochs);
    let solution = iterate_multi(ctx, epochs, &ambiguity_ids, state.clone(), opts, 1)?;

    if !residual_screen {
        return Ok(solution);
    }

    let unscreened_wrms = solution_weighted_rms(ctx, epochs, &solution, &state);
    match run_residual_screen(ctx, epochs.to_vec(), state, opts, solution.clone(), 1)? {
        ScreenResult::Clean => Ok(solution),
        ScreenResult::Screened {
            solution: screened,
            epochs: retained,
        } => {
            let screened_wrms = solution_weighted_rms(
                ctx,
                &retained,
                &screened,
                &state_from_solution(&screened, &FloatState::default_for_epochs(&retained)),
            );
            if screened_wrms.is_finite()
                && unscreened_wrms.is_finite()
                && screened_wrms * RESIDUAL_SCREEN_ACCEPT_FACTOR < unscreened_wrms
            {
                Ok(screened)
            } else {
                Ok(solution)
            }
        }
    }
}

enum ScreenResult {
    Clean,
    Screened {
        solution: FloatSolution,
        epochs: Vec<FloatEpoch>,
    },
}

fn run_residual_screen(
    ctx: ModelContext,
    epochs: Vec<FloatEpoch>,
    seed_state: FloatState,
    opts: FloatSolveOptions,
    solution: FloatSolution,
    pass: usize,
) -> Result<ScreenResult, FloatSolveError> {
    if pass > RESIDUAL_SCREEN_MAX_PASSES {
        return Ok(ScreenResult::Screened { solution, epochs });
    }

    let candidate_state = state_from_solution(&solution, &seed_state);
    match worst_multi_residual(ctx, &epochs, &candidate_state)? {
        Some((epoch_idx, sat)) => {
            let pruned = exclude_observation(&epochs, epoch_idx, &sat);
            if !multi_enough_after_prune(&pruned, ctx.tropo) {
                return Ok(ScreenResult::Screened { solution, epochs });
            }
            let ambiguity_ids = multi_ambiguity_ids(&pruned);
            let candidate = iterate_multi(
                ctx,
                &pruned,
                &ambiguity_ids,
                reseed_state(&seed_state, &pruned),
                opts,
                1,
            )?;
            run_residual_screen(ctx, pruned, seed_state, opts, candidate, pass + 1)
        }
        None => {
            if pass == 1 {
                Ok(ScreenResult::Clean)
            } else {
                Ok(ScreenResult::Screened { solution, epochs })
            }
        }
    }
}

fn iterate_multi(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    ambiguity_ids: &[AmbiguityId],
    state: FloatState,
    opts: FloatSolveOptions,
    iter: usize,
) -> Result<FloatSolution, FloatSolveError> {
    let n = ParameterLayout::ppp(
        epochs.len(),
        ztd_unknown_count(ctx.tropo),
        ambiguity_ids.len(),
    )
    .dim();
    let mut current = state;
    let mut iteration = iter;
    let max_iterations = opts.max_iterations;

    loop {
        let binding = AmbiguityBinding::Estimated {
            ids: ambiguity_ids,
            values: &current.ambiguities_m,
        };
        let rows = build_rows(ctx, epochs, &binding, &current).map_err(PppRowError::into_float)?;
        let dx = solve_normal_equations(&rows, n, ctx.normal)?;
        let next = apply_multi_delta(&current, epochs.len(), ambiguity_ids, &dx, ctx.tropo)?;
        let (pos_step, clock_step, ztd_step, ambiguity_step) =
            multi_step_norms(&dx, epochs.len(), ctx.tropo);

        if pos_step <= opts.position_tolerance_m
            && clock_step <= opts.clock_tolerance_m
            && ztd_step <= opts.ztd_tolerance_m
            && ambiguity_step <= opts.ambiguity_tolerance_m
        {
            return finalize_multi(
                ctx,
                epochs,
                ambiguity_ids,
                next,
                iteration,
                true,
                FloatStatus::StateTolerance,
            );
        }

        if iteration >= max_iterations {
            return finalize_multi(
                ctx,
                epochs,
                ambiguity_ids,
                next,
                iteration,
                false,
                FloatStatus::MaxIterations,
            );
        }

        current = next;
        iteration += 1;
    }
}

fn apply_multi_delta(
    state: &FloatState,
    n_epochs: usize,
    ambiguity_ids: &[AmbiguityId],
    dx: &[f64],
    tropo: TroposphereOptions,
) -> Result<FloatState, FloatSolveError> {
    let mut idx = 3;
    let clock_deltas = &dx[idx..idx + n_epochs];
    idx += n_epochs;
    let ztd_delta = if estimates_ztd(tropo) {
        let v = dx[idx];
        idx += 1;
        v
    } else {
        0.0
    };
    let ambiguity_deltas = &dx[idx..];
    let clocks_m = state
        .clocks_m
        .iter()
        .zip(clock_deltas)
        .map(|(clock, delta)| clock + delta)
        .collect();
    let mut ambiguities_m = BTreeMap::new();
    for (id, delta) in ambiguity_ids.iter().zip(ambiguity_deltas) {
        let prior = state
            .ambiguities_m
            .get(id.as_str())
            .copied()
            .ok_or_else(|| FloatSolveError::MissingAmbiguity(id.as_str().to_string()))?;
        ambiguities_m.insert(id.as_str().to_string(), prior + delta);
    }
    Ok(FloatState {
        position_m: [
            state.position_m[0] + dx[0],
            state.position_m[1] + dx[1],
            state.position_m[2] + dx[2],
        ],
        clocks_m,
        ambiguities_m,
        ztd_m: state.ztd_m + ztd_delta,
    })
}

fn multi_step_norms(
    dx: &[f64],
    n_epochs: usize,
    tropo: TroposphereOptions,
) -> (f64, f64, f64, f64) {
    let pos = vec3::norm3([dx[0], dx[1], dx[2]]);
    let mut idx = 3;
    let clock = max_abs(&dx[idx..idx + n_epochs]);
    idx += n_epochs;
    let ztd = if estimates_ztd(tropo) {
        let v = dx[idx].abs();
        idx += 1;
        v
    } else {
        0.0
    };
    let ambiguity = max_abs(&dx[idx..]);
    (pos, clock, ztd, ambiguity)
}

fn finalize_multi(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    ambiguity_ids: &[AmbiguityId],
    state: FloatState,
    iterations: usize,
    converged: bool,
    status: FloatStatus,
) -> Result<FloatSolution, FloatSolveError> {
    let residuals = residual_rows(ctx, epochs, &state.ambiguities_m, &state)
        .map_err(PppRowError::into_float)?;
    let code: Vec<f64> = residuals.iter().map(|r| r.code_m).collect();
    let phase: Vec<f64> = residuals.iter().map(|r| r.phase_m).collect();
    let solution = FloatSolution {
        position_m: state.position_m,
        epoch_clocks_m: state.clocks_m,
        ambiguities_m: state.ambiguities_m,
        ztd_residual_m: if estimates_ztd(ctx.tropo) {
            Some(state.ztd_m)
        } else {
            None
        },
        residuals_m: residuals.clone(),
        used_sats: ambiguity_ids
            .iter()
            .map(|id| id.as_str().to_string())
            .collect(),
        iterations,
        converged,
        status,
        code_rms_m: rms(&code),
        phase_rms_m: rms(&phase),
        weighted_rms_m: weighted_rms(&residuals, ctx.weights),
    };
    validate_float_solution_output(&solution, epochs.len())?;
    Ok(solution)
}

fn solution_weighted_rms(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    solution: &FloatSolution,
    seed_state: &FloatState,
) -> f64 {
    let state = state_from_solution(solution, seed_state);
    match residual_rows(ctx, epochs, &state.ambiguities_m, &state) {
        Ok(rows) => weighted_rms(&rows, ctx.weights),
        Err(_) => f64::INFINITY,
    }
}

fn worst_multi_residual(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    state: &FloatState,
) -> Result<Option<(usize, String)>, FloatSolveError> {
    let rows =
        residual_rows(ctx, epochs, &state.ambiguities_m, state).map_err(PppRowError::into_float)?;
    let candidate = rows
        .iter()
        .flat_map(|r| {
            [
                (
                    normalized_residual(
                        ResidualNormRecipe::PppInverseSigmaMagnitude,
                        r.code_m,
                        r.code_weight,
                    ),
                    r.epoch_index,
                    r.satellite_id.clone(),
                ),
                (
                    normalized_residual(
                        ResidualNormRecipe::PppInverseSigmaMagnitude,
                        r.phase_m,
                        r.phase_weight,
                    ),
                    r.epoch_index,
                    r.satellite_id.clone(),
                ),
            ]
        })
        .max_by(|a, b| a.0.total_cmp(&b.0));
    Ok(match candidate {
        Some((normalized, epoch_idx, sat)) if normalized > RESIDUAL_SCREEN_THRESHOLD => {
            Some((epoch_idx, sat))
        }
        _ => None,
    })
}

fn exclude_observation(
    epochs: &[FloatEpoch],
    drop_epoch_idx: usize,
    drop_sat: &str,
) -> Vec<FloatEpoch> {
    epochs
        .iter()
        .enumerate()
        .filter_map(|(epoch_idx, epoch)| {
            let mut epoch = epoch.clone();
            if epoch_idx == drop_epoch_idx {
                epoch
                    .observations
                    .retain(|obs| obs.satellite_id != drop_sat);
            }
            if epoch.observations.is_empty() {
                None
            } else {
                Some(epoch)
            }
        })
        .collect()
}

fn multi_enough_after_prune(epochs: &[FloatEpoch], tropo: TroposphereOptions) -> bool {
    if epochs.len() < 2 {
        return false;
    }
    let n_sats = multi_ambiguity_ids(epochs).len();
    let n_obs: usize = epochs.iter().map(|e| e.observations.len()).sum();
    let equations = 2 * n_obs;
    let unknowns = ParameterLayout::ppp(epochs.len(), ztd_unknown_count(tropo), n_sats).dim();
    n_sats >= 4 && equations >= unknowns
}

fn reseed_state(state: &FloatState, epochs: &[FloatEpoch]) -> FloatState {
    FloatState {
        position_m: state.position_m,
        clocks_m: vec![state.clocks_m[0]; epochs.len()],
        ambiguities_m: initial_ambiguities(epochs),
        ztd_m: state.ztd_m,
    }
}

pub(super) fn initial_ambiguities(epochs: &[FloatEpoch]) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for obs in epochs.iter().flat_map(|e| e.observations.iter()) {
        out.entry(obs.ambiguity_id.clone())
            .or_insert(obs.phase_m - obs.code_m);
    }
    out
}

fn multi_ambiguity_ids(epochs: &[FloatEpoch]) -> Vec<AmbiguityId> {
    epochs
        .iter()
        .flat_map(|e| {
            e.observations
                .iter()
                .map(|o| AmbiguityId::new(o.ambiguity_id.clone()))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}
