//! Static integer-fixed PPP solve.
//!
//! This leaf owns the fixed-ambiguity orchestration: the LAMBDA integer search
//! from a float solution, the ambiguity-conditioned multi-epoch re-solve, the
//! post-fit residual rows, and the cycle/metre ambiguity conversions. The shared
//! measurement model lives in [`super::model`], the dense normal-equation kernel
//! in [`super::normal`], and the row staging / shared scalar helpers in [`super`].

use std::collections::{BTreeMap, BTreeSet};

use crate::ambiguity::AmbiguityId;
use crate::astro::math::vec3;
use crate::estimation::recipe::{EstimationRecipe, NormalRecipe};
use crate::estimation::substrate::ambiguity::resolve_integer_lattice;
use crate::observables::ObservableEphemerisSource;

use super::float::{
    next_pass, prepare_arc, screened_float_solution, solve_to_ssr_fixed_point, ssr_bias_holds_at,
    ArcSettings, FixedPointStart, LeftOut, PreparedArc, StepError,
};
use super::normal::{
    ambiguity_covariance_from_normal, clock_eliminated_normal_equations, ppp_position_covariance,
    solve_normal_equations, PppNormalLayout,
};
use super::rows::exclude_unresolved_ssr_bias_observations;
use super::rows::{build_rows, residual_rows, AmbiguityBinding, PppRowError};
use super::temporal::{estimate_temporal_correlation, temporal_position_covariance};
use super::{
    estimates_tropo_gradients, estimates_ztd, max_abs, residual_ionosphere_unknown_count, rms,
    state_from_solution, tropo_gradient_unknown_count, validate_fixed_solve_boundary, weighted_rms,
    ztd_unknown_count, AmbiguitySearch, FixedIntegerMetadata, FixedSolution, FixedSolveConfig,
    FixedSolveError, FloatEpoch, FloatSolution, FloatSolveError, FloatSolveOptions, FloatState,
    FloatStatus, IntegerStatus, ModelContext, SsrBiasExclusion, SsrBiasExclusionStage,
    TroposphereOptions,
};

/// Search integer ambiguities from an existing float PPP solution and re-solve
/// position/clocks with those ambiguities held fixed.
pub fn solve_fixed_from_float(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    float_solution: FloatSolution,
    config: FixedSolveConfig,
) -> Result<FixedSolution, FixedSolveError> {
    validate_fixed_solve_boundary(epochs, &float_solution, &config)?;
    use crate::estimation::recipe::StrategyId;
    use crate::estimation::strategies::{
        estimate, EstimateError, EstimateInput, EstimateOptions, EstimateOutput,
    };
    match estimate(
        EstimateInput::PppFixed {
            source,
            epochs,
            float_solution,
            config,
        },
        EstimateOptions::new(StrategyId::ppp_reference()),
    ) {
        Ok(EstimateOutput::PppFixed(solution)) => Ok(*solution),
        Err(EstimateError::PppFixed(error)) => Err(error),
        Ok(_) | Err(_) => {
            unreachable!(
                "the PPP reference strategy yields a PPP fixed solution or a PPP fixed error"
            )
        }
    }
}

/// Drive the integer-fixed PPP re-solve from a resolved [`EstimationRecipe`]: the
/// shared per-technique implementation that
/// [`crate::estimation::strategies::estimate`] dispatches to. The recipe's
/// [`NormalRecipe`] reaches the solve seam through [`ModelContext::normal`]; for
/// the PPP reference recipe (`NormalRecipe::PppDenseLastTie`) this is
/// bit-identical to the legacy path.
pub(crate) fn run_fixed_from_float(
    recipe: &EstimationRecipe,
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    float_solution: FloatSolution,
    config: FixedSolveConfig,
) -> Result<FixedSolution, FixedSolveError> {
    validate_fixed_solve_boundary(epochs, &float_solution, &config)?;
    let arc = ArcSettings {
        source,
        weights: config.weights,
        tropo: config.tropo,
        corrections: &config.corrections,
        normal: recipe.normal,
        estimate_residual_ionosphere: config.estimate_residual_ionosphere,
        elevation_cutoff_deg: config.elevation_cutoff_deg,
    };
    let observation_count = epochs.iter().map(|e| e.observations.len()).sum::<usize>();
    let mut float_solution = float_solution;
    let mut pinned: Vec<(usize, String)> = Vec::new();
    let mut readmitted_after_fix: Vec<(usize, String)> = Vec::new();
    let mut solved_for_fixed_arc: BTreeSet<String> = BTreeSet::new();
    loop {
        debug_assert!(
            readmitted_after_fix.len() <= observation_count,
            "each observation is admitted again after a fix at most once"
        );
        let deferred = float_solution
            .ssr_bias_readmissions
            .iter()
            .chain(&readmitted_after_fix)
            .cloned()
            .collect::<Vec<_>>();
        let exclusion = match fix_once(&arc, recipe, epochs, &float_solution, &config, &deferred) {
            Ok(solution) => {
                // Every transmit-time exclusion was judged at a state other than the fixed
                // position: check each at the fixed position and admit those that hold,
                // once each after a fix.
                let pass = next_pass(&float_solution, &solution.ssr_bias_exclusions)
                    .map_err(FixedSolveError::Float)?;
                let holding = solution
                    .ssr_bias_exclusions
                    .iter()
                    .filter(|exclusion| exclusion.transmit_time_failure.is_some())
                    .map(|exclusion| (exclusion.epoch_index, exclusion.ambiguity_id.clone()))
                    .filter(|key| !readmitted_after_fix.contains(key))
                    .filter(|key| ssr_bias_holds_at(&arc, epochs, key, solution.position_m, pass))
                    .collect::<Vec<_>>();
                if holding.is_empty() {
                    return Ok(solution);
                }
                pinned.retain(|key| !holding.contains(key));
                let mut exclusions = float_solution.ssr_bias_exclusions.clone();
                exclusions.retain(|exclusion| {
                    !holding.iter().any(|(epoch_index, ambiguity_id)| {
                        *epoch_index == exclusion.epoch_index
                            && *ambiguity_id == exclusion.ambiguity_id
                    })
                });
                readmitted_after_fix.extend(holding);
                float_solution = resolve_float(
                    &arc,
                    epochs,
                    &float_solution,
                    exclusions,
                    &pinned,
                    &readmitted_after_fix,
                )?;
                continue;
            }
            Err(FixedStep::Fixed(error)) => return Err(error),
            Err(FixedStep::Unsolved(ids)) => {
                // The fixed arc observes ambiguities the float solution never solved, such
                // as a satellite the float solve's elevation cutoff removed and the fixed
                // configuration keeps. Solve the float arc again over the fixed arc, once
                // for each such ambiguity, and fix again.
                if let Some(id) = ids.iter().find(|id| solved_for_fixed_arc.contains(*id)) {
                    return Err(FixedSolveError::Float(FloatSolveError::MissingAmbiguity(
                        id.clone(),
                    )));
                }
                solved_for_fixed_arc.extend(ids);
                let exclusions = float_solution.ssr_bias_exclusions.clone();
                float_solution = resolve_float(
                    &arc,
                    epochs,
                    &float_solution,
                    exclusions,
                    &pinned,
                    &readmitted_after_fix,
                )?;
                continue;
            }
            Err(FixedStep::Exclude(exclusion)) => *exclusion,
        };
        // The fixed re-solve crossed a bias record boundary the float solve did not.
        // Exclude that observation, solve the float arc again to its fixed point from the
        // float state, and fix again.
        pinned.push((exclusion.epoch_index, exclusion.ambiguity_id.clone()));
        let mut exclusions = float_solution.ssr_bias_exclusions.clone();
        exclusions.push(exclusion);
        float_solution = resolve_float(
            &arc,
            epochs,
            &float_solution,
            exclusions,
            &pinned,
            &readmitted_after_fix,
        )?;
    }
}

/// Solve the float arc again to its fixed point from `float_solution`'s state, starting
/// from `exclusions`, with the residual screen setting and options it was solved with.
/// `pinned` observations are never admitted again; the float solution's own readmissions
/// and `readmitted_after_fix` are judged only at convergence and not admitted again.
fn resolve_float(
    arc: &ArcSettings<'_>,
    epochs: &[FloatEpoch],
    float_solution: &FloatSolution,
    exclusions: Vec<SsrBiasExclusion>,
    pinned: &[(usize, String)],
    readmitted_after_fix: &[(usize, String)],
) -> Result<FloatSolution, FixedSolveError> {
    let opts = float_solution.solve_options;
    let residual_screen = float_solution.residual_screen;
    let first_pass = next_pass(float_solution, &exclusions).map_err(FixedSolveError::Float)?;
    let start = FixedPointStart {
        exclusions,
        pinned: pinned.to_vec(),
        readmitted: float_solution
            .ssr_bias_readmissions
            .iter()
            .chain(readmitted_after_fix)
            .cloned()
            .collect(),
        judged_after_fix: readmitted_after_fix.to_vec(),
        first_pass,
    };
    let mut solution = solve_to_ssr_fixed_point(
        arc,
        epochs,
        input_float_state(float_solution, epochs),
        start,
        |ctx, solve_epochs, state| {
            screened_float_solution(ctx, solve_epochs, state, opts, residual_screen)
        },
    )
    .map_err(FixedSolveError::Float)?;
    solution.residual_screen = residual_screen;
    solution.solve_options = opts;
    Ok(solution)
}

/// The float solution's state over all `n_epochs` input epochs: its position, troposphere,
/// ambiguities and ionosphere, and a clock for every input epoch seeded as the fixed solve
/// seeds it.
fn input_float_state(float_solution: &FloatSolution, epochs: &[FloatEpoch]) -> FloatState {
    let mut state = state_from_solution(float_solution, &FloatState::default_for_epochs(epochs));
    state.clocks_m = fixed_seed_clocks(float_solution, &(0..epochs.len()).collect::<Vec<_>>());
    state
}

/// A fixed solve step failure: SSR/HAS bias records that stopped holding during the fixed
/// re-solve, ambiguities the fixed arc observes that the float solution did not solve, or
/// any other error.
enum FixedStep {
    Exclude(Box<SsrBiasExclusion>),
    Unsolved(Vec<String>),
    Fixed(FixedSolveError),
}

impl From<FixedSolveError> for FixedStep {
    fn from(error: FixedSolveError) -> Self {
        Self::Fixed(error)
    }
}

impl From<FloatSolveError> for FixedStep {
    fn from(error: FloatSolveError) -> Self {
        Self::Fixed(FixedSolveError::Float(error))
    }
}

impl FixedStep {
    fn from_rows(
        error: PppRowError,
        ctx: ModelContext,
        epochs: &[FloatEpoch],
        state: &FloatState,
    ) -> Self {
        match error {
            flip @ PppRowError::SsrBiasFlip { .. } => {
                match StepError::from_rows(flip, ctx, epochs, state) {
                    StepError::Flip(flip) => Self::Exclude(Box::new(flip.exclusion)),
                    StepError::Float(error) => Self::Fixed(FixedSolveError::Float(error)),
                }
            }
            other => Self::Fixed(other.into_fixed()),
        }
    }
}

/// One integer search and fixed re-solve from `float_solution`, starting from its
/// exclusions.
fn fix_once(
    arc: &ArcSettings<'_>,
    recipe: &EstimationRecipe,
    epochs: &[FloatEpoch],
    float_solution: &FloatSolution,
    config: &FixedSolveConfig,
    deferred: &[(usize, String)],
) -> Result<FixedSolution, FixedStep> {
    let source = arc.source;
    // Seed every input epoch's clock from the float solution, then prepare the arc as the
    // float solve does, starting from the float solve's exclusions.
    let mut input_state = fixed_state_from_float(float_solution);
    input_state.clocks_m =
        fixed_seed_clocks(float_solution, &(0..epochs.len()).collect::<Vec<_>>());
    let float_exclusions = float_solution.ssr_bias_exclusions.clone();
    let pass = next_pass(float_solution, &float_exclusions)?;
    let (prepared, new_exclusions) = prepare_arc(
        arc,
        epochs,
        &LeftOut {
            excluded: &float_exclusions,
            screened: &float_solution.residual_screen_removals,
            deferred,
            seed_ambiguities: &BTreeMap::new(),
        },
        &input_state,
        pass,
    )?;
    let mut ssr_bias_exclusions = float_exclusions;
    ssr_bias_exclusions.extend(new_exclusions);
    let PreparedArc {
        epochs: solved_epochs,
        correction_epoch_indices,
        state: initial_state,
    } = prepared;
    let solve_epochs = solved_epochs.as_slice();
    let unsolved = active_ambiguity_ids(solve_epochs)
        .into_iter()
        .map(|id| id.as_str().to_string())
        .filter(|id| !float_solution.used_sats.contains(id))
        .collect::<Vec<_>>();
    if !unsolved.is_empty() {
        return Err(FixedStep::Unsolved(unsolved));
    }
    let active_order;
    let search_order = if config.elevation_cutoff_deg.is_some()
        || !ssr_bias_exclusions.is_empty()
        || !float_solution.residual_screen_removals.is_empty()
    {
        active_order = active_ambiguity_ids(solve_epochs);
        Some(active_order.as_slice())
    } else {
        None
    };
    let fixed_meta = search_integer_ambiguities(
        source,
        solve_epochs,
        &FixedArc {
            correction_epoch_indices: &correction_epoch_indices,
            seed_clocks_m: &initial_state.clocks_m,
            deferred,
            pass,
        },
        float_solution,
        config,
        search_order,
    )?;
    let fixed_m = fixed_ambiguities_m(
        &fixed_meta.fixed_cycles,
        &config.ambiguity.wavelengths_m,
        &config.ambiguity.offsets_m,
    )?;
    let ctx = ModelContext {
        source,
        weights: config.weights,
        tropo: config.tropo,
        corrections: &config.corrections,
        normal: recipe.normal,
        estimate_residual_ionosphere: config.estimate_residual_ionosphere,
        correction_epoch_indices: Some(&correction_epoch_indices),
        ssr_bias_pass: pass,
        ssr_bias_stage: SsrBiasExclusionStage::FixedResolve,
        ssr_bias_deferred: deferred,
    };
    let resolve = iterate_fixed_multi(ctx, solve_epochs, &fixed_m, initial_state, config.opts, 1)?;
    let mut solution = finalize_fixed_multi(
        ctx,
        solve_epochs,
        fixed_meta,
        fixed_m,
        float_solution.clone(),
        resolve,
    )?;
    // The observations admitted again after an exclusion were not checked at intermediate
    // states; the fixed position decides.
    if let Some(exclusion) = deferred
        .iter()
        .filter(|key| {
            solution
                .residuals_m
                .iter()
                .any(|residual| residual.epoch_index == key.0 && residual.ambiguity_id == key.1)
        })
        .find_map(|key| {
            let epoch = epochs.get(key.0)?;
            let mut single = epoch.clone();
            single.observations.retain(|obs| obs.ambiguity_id == key.1);
            exclude_unresolved_ssr_bias_observations(
                source,
                std::slice::from_ref(&single),
                key.0,
                solution.position_m,
                &config.corrections.ppp,
                pass,
                SsrBiasExclusionStage::AtConvergence,
            )
            .1
            .into_iter()
            .next()
        })
    {
        return Err(FixedStep::Exclude(Box::new(exclusion)));
    }
    solution.ssr_bias_exclusions = ssr_bias_exclusions;
    Ok(solution)
}

struct FixedSearchResult {
    order: Vec<AmbiguityId>,
    fixed_cycles: BTreeMap<String, i64>,
    integer: FixedIntegerMetadata,
}

/// Converged state from the ambiguity-conditioned re-solve, carried from
/// [`iterate_fixed_multi`] into [`finalize_fixed_multi`].
struct FixedResolve {
    state: FloatState,
    iterations: usize,
    converged: bool,
    status: FloatStatus,
}

impl From<FloatSolveError> for FixedSolveError {
    fn from(value: FloatSolveError) -> Self {
        Self::Float(value)
    }
}

/// Receiver clock seeds for the fixed re-solve: for each solved input epoch, the float
/// solution's clock of that input epoch, or of the nearest input epoch the float solve
/// solved when it left that one out.
fn fixed_seed_clocks(
    float_solution: &FloatSolution,
    correction_epoch_indices: &[usize],
) -> Vec<f64> {
    correction_epoch_indices
        .iter()
        .map(|&epoch_index| {
            float_solution
                .solved_epoch_indices
                .iter()
                .zip(&float_solution.epoch_clocks_m)
                .min_by_key(|(solved, _)| solved.abs_diff(epoch_index))
                .map_or(0.0, |(_, clock_m)| *clock_m)
        })
        .collect()
}

/// The solved epochs of a fixed solve: the input index of each, the receiver clock seeds,
/// the observations admitted again after an exclusion, and the pass number.
struct FixedArc<'a> {
    correction_epoch_indices: &'a [usize],
    seed_clocks_m: &'a [f64],
    deferred: &'a [(usize, String)],
    pass: usize,
}

fn search_integer_ambiguities(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    fixed_arc: &FixedArc<'_>,
    float_solution: &FloatSolution,
    config: &FixedSolveConfig,
    active_order: Option<&[AmbiguityId]>,
) -> Result<FixedSearchResult, FixedStep> {
    let order: Vec<AmbiguityId> = active_order.map_or_else(
        || {
            float_solution
                .used_sats
                .iter()
                .map(|sat| AmbiguityId::new(sat.clone()))
                .collect()
        },
        |order| order.to_vec(),
    );
    let covariance_cycles =
        ambiguity_covariance_cycles(source, epochs, fixed_arc, &order, float_solution, config)?;
    let float_cycles = float_ambiguities_cycles(
        float_solution,
        &config.ambiguity.wavelengths_m,
        &config.ambiguity.offsets_m,
    )?;
    let floats = order
        .iter()
        .map(|id| {
            float_cycles.get(id.as_str()).copied().ok_or_else(|| {
                FixedSolveError::Float(FloatSolveError::MissingAmbiguity(id.as_str().to_string()))
            })
        })
        .collect::<Result<Vec<f64>, _>>()?;
    let result = resolve_integer_lattice(
        &floats,
        &covariance_cycles,
        config.ambiguity.ratio_threshold,
    )
    .map_err(FixedSolveError::Integer)?;
    let fixed_cycles = order
        .iter()
        .map(|id| id.as_str().to_string())
        .zip(result.fixed.iter().copied())
        .collect::<BTreeMap<_, _>>();
    let search_order: Vec<String> = order.iter().map(|id| id.as_str().to_string()).collect();
    Ok(FixedSearchResult {
        order,
        fixed_cycles,
        integer: FixedIntegerMetadata {
            integer_status: if result.fixed_status {
                IntegerStatus::Fixed
            } else {
                IntegerStatus::NotFixed
            },
            integer_ratio: result.ratio,
            integer_best_score: result.best_score,
            integer_second_best_score: result.second_best_score,
            integer_candidates: result.candidates_evaluated,
            ambiguity_search: AmbiguitySearch {
                order: search_order,
                float_cycles,
                covariance_cycles: result.covariance,
                covariance_inverse_cycles: result.covariance_inverse,
            },
        },
    })
}

fn active_ambiguity_ids(epochs: &[FloatEpoch]) -> Vec<AmbiguityId> {
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

fn iterate_fixed_multi(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    fixed_m: &BTreeMap<String, f64>,
    state: FloatState,
    opts: FloatSolveOptions,
    iter: usize,
) -> Result<FixedResolve, FixedStep> {
    let mut current = state;
    let mut iteration = iter;
    let max_iterations = opts.max_iterations;

    loop {
        let binding = AmbiguityBinding::Held { values: fixed_m };
        let rows = build_rows(ctx, epochs, &binding, &current)
            .map_err(|error| FixedStep::from_rows(error, ctx, epochs, &current))?;
        let layout = PppNormalLayout::new(
            epochs.len(),
            ztd_unknown_count(ctx.tropo),
            tropo_gradient_unknown_count(ctx.tropo),
            residual_ionosphere_unknown_count(ctx.estimate_residual_ionosphere, fixed_m.len()),
            0,
        );
        let dx = solve_normal_equations(&rows, layout, ctx.normal)?;
        let next = apply_fixed_multi_delta(
            &current,
            epochs.len(),
            fixed_m,
            &dx,
            ctx.tropo,
            ctx.estimate_residual_ionosphere,
        );
        let (pos_step, clock_step, ztd_step, gradient_step) = fixed_multi_step_norms(
            &dx,
            ctx.tropo,
            ctx.estimate_residual_ionosphere,
            fixed_m.len(),
        );

        if pos_step <= opts.position_tolerance_m
            && clock_step <= opts.clock_tolerance_m
            && ztd_step <= opts.ztd_tolerance_m
            && gradient_step <= opts.ztd_tolerance_m
        {
            return Ok(FixedResolve {
                state: next,
                iterations: iteration,
                converged: true,
                status: FloatStatus::StateTolerance,
            });
        }

        if iteration >= max_iterations {
            return Ok(FixedResolve {
                state: next,
                iterations: iteration,
                converged: false,
                status: FloatStatus::MaxIterations,
            });
        }

        current = next;
        iteration += 1;
    }
}

fn apply_fixed_multi_delta(
    state: &FloatState,
    n_epochs: usize,
    fixed_m: &BTreeMap<String, f64>,
    dx: &[f64],
    tropo: TroposphereOptions,
    estimate_residual_ionosphere: bool,
) -> FloatState {
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
    let (tropo_gradient_north_delta, tropo_gradient_east_delta) =
        if estimates_tropo_gradients(tropo) {
            let north = dx[idx];
            let east = dx[idx + 1];
            idx += 2;
            (north, east)
        } else {
            (0.0, 0.0)
        };
    let mut residual_ionosphere_m = BTreeMap::new();
    if estimate_residual_ionosphere {
        let ionosphere_deltas = &dx[idx..idx + fixed_m.len()];
        for ((id, _), delta) in fixed_m.iter().zip(ionosphere_deltas) {
            let prior = state.residual_ionosphere_m.get(id).copied().unwrap_or(0.0);
            residual_ionosphere_m.insert(id.clone(), prior + delta);
        }
    }
    let clocks_m = state
        .clocks_m
        .iter()
        .zip(clock_deltas)
        .map(|(clock, delta)| clock + delta)
        .collect();
    FloatState {
        position_m: [
            state.position_m[0] + dx[0],
            state.position_m[1] + dx[1],
            state.position_m[2] + dx[2],
        ],
        clocks_m,
        ambiguities_m: BTreeMap::new(),
        ztd_m: state.ztd_m + ztd_delta,
        tropo_gradient_north_m: state.tropo_gradient_north_m + tropo_gradient_north_delta,
        tropo_gradient_east_m: state.tropo_gradient_east_m + tropo_gradient_east_delta,
        residual_ionosphere_m,
    }
}

fn fixed_multi_step_norms(
    dx: &[f64],
    tropo: TroposphereOptions,
    estimate_residual_ionosphere: bool,
    n_residual_ionosphere: usize,
) -> (f64, f64, f64, f64) {
    let pos = vec3::norm3([dx[0], dx[1], dx[2]]);
    let n_ztd = ztd_unknown_count(tropo);
    let n_gradients = tropo_gradient_unknown_count(tropo);
    let n_ionosphere =
        residual_ionosphere_unknown_count(estimate_residual_ionosphere, n_residual_ionosphere);
    let n_clocks = dx.len() - 3 - n_ztd - n_gradients - n_ionosphere;
    let clock = max_abs(&dx[3..3 + n_clocks]);
    let mut idx = 3 + n_clocks;
    let ztd = if estimates_ztd(tropo) {
        let v = dx[idx].abs();
        idx += 1;
        v
    } else {
        0.0
    };
    let gradient = if estimates_tropo_gradients(tropo) {
        let v = max_abs(&dx[idx..idx + 2]);
        idx += 2;
        v
    } else {
        0.0
    };
    let ionosphere = if estimate_residual_ionosphere {
        max_abs(&dx[idx..idx + n_residual_ionosphere])
    } else {
        0.0
    };
    (pos, clock, ztd, gradient.max(ionosphere))
}

fn finalize_fixed_multi(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    search: FixedSearchResult,
    fixed_m: BTreeMap<String, f64>,
    float_solution: FloatSolution,
    resolve: FixedResolve,
) -> Result<FixedSolution, FixedStep> {
    // A flip the rows find while the solution is assembled is found at convergence.
    let ctx = ModelContext {
        ssr_bias_stage: SsrBiasExclusionStage::AtConvergence,
        ..ctx
    };
    let FixedResolve {
        state,
        iterations,
        converged,
        status,
    } = resolve;
    let residuals = residual_rows(ctx, epochs, &fixed_m, &state)
        .map_err(|error| FixedStep::from_rows(error, ctx, epochs, &state))?;
    let binding = AmbiguityBinding::Held { values: &fixed_m };
    let rows = build_rows(ctx, epochs, &binding, &state)
        .map_err(|error| FixedStep::from_rows(error, ctx, epochs, &state))?;
    let covariance = ppp_position_covariance(
        &rows,
        PppNormalLayout::new(
            epochs.len(),
            ztd_unknown_count(ctx.tropo),
            tropo_gradient_unknown_count(ctx.tropo),
            residual_ionosphere_unknown_count(ctx.estimate_residual_ionosphere, fixed_m.len()),
            0,
        ),
        state.position_m,
    )?;
    let code: Vec<f64> = residuals.iter().map(|r| r.code_m).collect();
    let phase: Vec<f64> = residuals.iter().map(|r| r.phase_m).collect();
    let solved_epoch_indices: Vec<usize> = (0..epochs.len())
        .map(|epoch_idx| ctx.correction_epoch_index(epoch_idx))
        .collect();
    let temporal_correlation =
        estimate_temporal_correlation(&residuals, epochs, &solved_epoch_indices);
    let (temporal_position_covariance, temporal_position_covariance_scale_factor) =
        temporal_position_covariance(
            covariance.formal,
            covariance.posterior_variance_factor,
            temporal_correlation,
        );
    Ok(FixedSolution {
        position_m: state.position_m,
        position_covariance: covariance.scaled,
        formal_position_covariance: covariance.formal,
        posterior_variance_factor: covariance.posterior_variance_factor,
        position_covariance_scale_factor: covariance.covariance_scale_factor,
        temporal_position_covariance,
        temporal_position_covariance_scale_factor,
        temporal_correlation,
        epoch_clocks_m: state.clocks_m,
        fixed_ambiguities_cycles: search.fixed_cycles,
        fixed_ambiguities_m: fixed_m,
        residual_ionosphere_m: if ctx.estimate_residual_ionosphere {
            state.residual_ionosphere_m
        } else {
            BTreeMap::new()
        },
        ztd_residual_m: if estimates_ztd(ctx.tropo) {
            Some(state.ztd_m)
        } else {
            None
        },
        tropo_gradient_north_m: if estimates_tropo_gradients(ctx.tropo) {
            Some(state.tropo_gradient_north_m)
        } else {
            None
        },
        tropo_gradient_east_m: if estimates_tropo_gradients(ctx.tropo) {
            Some(state.tropo_gradient_east_m)
        } else {
            None
        },
        tropo_gradient_covariance_m2: covariance.tropo_gradient_scaled_m2,
        formal_tropo_gradient_covariance_m2: covariance.tropo_gradient_formal_m2,
        float_solution,
        residuals_m: residuals.clone(),
        used_sats: search
            .order
            .into_iter()
            .map(AmbiguityId::into_string)
            .collect(),
        iterations,
        converged,
        status,
        code_rms_m: rms(&code),
        phase_rms_m: rms(&phase),
        weighted_rms_m: weighted_rms(&residuals, ctx.weights),
        integer: search.integer,
        ssr_bias_exclusions: Vec::new(),
        solved_epoch_indices,
    })
}

fn fixed_state_from_float(solution: &FloatSolution) -> FloatState {
    FloatState {
        position_m: solution.position_m,
        clocks_m: solution.epoch_clocks_m.clone(),
        ambiguities_m: BTreeMap::new(),
        ztd_m: solution.ztd_residual_m.unwrap_or(0.0),
        tropo_gradient_north_m: solution.tropo_gradient_north_m.unwrap_or(0.0),
        tropo_gradient_east_m: solution.tropo_gradient_east_m.unwrap_or(0.0),
        residual_ionosphere_m: solution.residual_ionosphere_m.clone(),
    }
}

fn float_ambiguities_cycles(
    solution: &FloatSolution,
    wavelengths_m: &BTreeMap<String, f64>,
    offsets_m: &BTreeMap<String, f64>,
) -> Result<BTreeMap<String, f64>, FixedSolveError> {
    let mut out = BTreeMap::new();
    for sat in &solution.used_sats {
        let wavelength = wavelengths_m
            .get(sat)
            .copied()
            .ok_or_else(|| FixedSolveError::MissingWavelength(sat.clone()))?;
        let offset = offsets_m
            .get(sat)
            .copied()
            .ok_or_else(|| FixedSolveError::MissingOffset(sat.clone()))?;
        let ambiguity_m = solution.ambiguities_m.get(sat).copied().ok_or_else(|| {
            FixedSolveError::Float(FloatSolveError::MissingAmbiguity(sat.clone()))
        })?;
        out.insert(sat.clone(), (ambiguity_m - offset) / wavelength);
    }
    Ok(out)
}

fn fixed_ambiguities_m(
    fixed_cycles: &BTreeMap<String, i64>,
    wavelengths_m: &BTreeMap<String, f64>,
    offsets_m: &BTreeMap<String, f64>,
) -> Result<BTreeMap<String, f64>, FixedSolveError> {
    let mut out = BTreeMap::new();
    for (sat, cycles) in fixed_cycles {
        let wavelength = wavelengths_m
            .get(sat)
            .copied()
            .ok_or_else(|| FixedSolveError::MissingWavelength(sat.clone()))?;
        let offset = offsets_m
            .get(sat)
            .copied()
            .ok_or_else(|| FixedSolveError::MissingOffset(sat.clone()))?;
        out.insert(sat.clone(), offset + *cycles as f64 * wavelength);
    }
    Ok(out)
}

fn ambiguity_covariance_cycles(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    fixed_arc: &FixedArc<'_>,
    ambiguity_ids: &[AmbiguityId],
    float_solution: &FloatSolution,
    config: &FixedSolveConfig,
) -> Result<Vec<Vec<f64>>, FixedStep> {
    let mut state = state_from_solution(float_solution, &FloatState::default_for_epochs(epochs));
    state.clocks_m = fixed_arc.seed_clocks_m.to_vec();
    let layout = PppNormalLayout::new(
        epochs.len(),
        ztd_unknown_count(config.tropo),
        tropo_gradient_unknown_count(config.tropo),
        residual_ionosphere_unknown_count(config.estimate_residual_ionosphere, ambiguity_ids.len()),
        ambiguity_ids.len(),
    );
    let start = layout.reduced_ambiguity_offset();
    let ctx = ModelContext {
        source,
        weights: config.weights,
        tropo: config.tropo,
        corrections: &config.corrections,
        // Covariance assembly uses the const last-tie assembler directly; the
        // recipe field is the PPP reference and unused on this path.
        normal: NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: config.estimate_residual_ionosphere,
        correction_epoch_indices: Some(fixed_arc.correction_epoch_indices),
        ssr_bias_pass: fixed_arc.pass,
        ssr_bias_stage: SsrBiasExclusionStage::FixedResolve,
        ssr_bias_deferred: fixed_arc.deferred,
    };
    let binding = AmbiguityBinding::Estimated {
        ids: ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let rows = build_rows(ctx, epochs, &binding, &state)
        .map_err(|error| FixedStep::from_rows(error, ctx, epochs, &state))?;
    let (normal, _rhs) = clock_eliminated_normal_equations(&rows, layout)?;
    let covariance_m = ambiguity_covariance_from_normal(&normal, start, ambiguity_ids.len())?;
    let mut covariance_cycles = vec![vec![0.0; ambiguity_ids.len()]; ambiguity_ids.len()];
    for i in 0..ambiguity_ids.len() {
        let lambda_i = config
            .ambiguity
            .wavelengths_m
            .get(ambiguity_ids[i].as_str())
            .copied()
            .ok_or_else(|| {
                FixedSolveError::MissingWavelength(ambiguity_ids[i].as_str().to_string())
            })?;
        for j in 0..ambiguity_ids.len() {
            let lambda_j = config
                .ambiguity
                .wavelengths_m
                .get(ambiguity_ids[j].as_str())
                .copied()
                .ok_or_else(|| {
                    FixedSolveError::MissingWavelength(ambiguity_ids[j].as_str().to_string())
                })?;
            covariance_cycles[i][j] = covariance_m[i][j] / (lambda_i * lambda_j);
        }
    }
    Ok(covariance_cycles)
}
