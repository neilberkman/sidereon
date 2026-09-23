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

use super::normal::{ppp_position_covariance, solve_normal_equations, PppNormalLayout};
use super::rows::{
    build_rows, drop_empty_epochs, exclude_unresolved_ssr_bias_observations, residual_rows,
    AmbiguityBinding, PppRowError,
};
use super::temporal::{estimate_temporal_correlation, temporal_position_covariance};
use super::{
    apply_elevation_cutoff, estimates_tropo_gradients, estimates_ztd, max_abs,
    residual_ionosphere_unknown_count, rms, state_from_solution, tropo_gradient_unknown_count,
    validate_float_solution_output, validate_float_solve_boundary,
    validate_ssr_bias_exclusion_retained, weighted_rms, ztd_unknown_count, FloatEpoch,
    FloatSolution, FloatSolveConfig, FloatSolveError, FloatSolveOptions, FloatState, FloatStatus,
    MeasurementWeights, ModelContext, RangeCorrections, SsrBiasExclusion, SsrBiasExclusionStage,
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
/// the PPP reference recipe (`NormalRecipe::PppDenseLastTie`) the static path
/// uses clock-eliminated reduced normals, pinned equivalent to the legacy dense
/// solve (solution and covariance oracles in `precise_positioning::normal`).
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
    config.opts.ambiguity_tolerance_m = SINGLE_EPOCH_AMBIGUITY_TOLERANCE_M;
    let arc = ArcSettings {
        source,
        weights: config.weights,
        tropo: config.tropo,
        corrections: &config.corrections,
        normal: NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: config.estimate_residual_ionosphere,
        elevation_cutoff_deg: config.elevation_cutoff_deg,
    };
    let opts = config.opts;
    let mut solution = solve_to_ssr_fixed_point(
        &arc,
        &epochs,
        initial_state,
        FixedPointStart::default(),
        |ctx, solve_epochs, state| {
            let ambiguity_ids = solve_epochs[0]
                .observations
                .iter()
                .map(|obs| AmbiguityId::new(obs.ambiguity_id.clone()))
                .collect::<Vec<_>>();
            iterate_multi_steps(ctx, solve_epochs, &ambiguity_ids, state, opts, 1)
        },
    )?;
    solution.residual_screen = false;
    solution.solve_options = opts;
    Ok(solution)
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
        elevation_cutoff_deg,
        residual_screen,
        estimate_residual_ionosphere,
    } = config;
    let arc = ArcSettings {
        source,
        weights,
        tropo,
        corrections: &corrections,
        normal,
        estimate_residual_ionosphere,
        elevation_cutoff_deg,
    };
    let mut solution = solve_to_ssr_fixed_point(
        &arc,
        epochs,
        state,
        FixedPointStart::default(),
        |ctx, solve_epochs, state| {
            screened_float_solution(ctx, solve_epochs, state, opts, residual_screen)
        },
    )?;
    solution.residual_screen = residual_screen;
    solution.solve_options = opts;
    Ok(solution)
}

/// The settings of a static float solve that stay fixed across its passes.
pub(super) struct ArcSettings<'a> {
    pub(super) source: &'a dyn ObservableEphemerisSource,
    pub(super) weights: MeasurementWeights,
    pub(super) tropo: TroposphereOptions,
    pub(super) corrections: &'a RangeCorrections,
    pub(super) normal: NormalRecipe,
    pub(super) estimate_residual_ionosphere: bool,
    pub(super) elevation_cutoff_deg: Option<f64>,
}

/// The epochs one solve pass solves: the input epochs without the observations left out
/// so far, and the input index of each.
pub(super) struct PreparedArc {
    pub(super) epochs: Vec<FloatEpoch>,
    pub(super) correction_epoch_indices: Vec<usize>,
    pub(super) state: FloatState,
}

/// Prepare one solve pass over the input `epochs`, in the order the solve refuses them:
/// the observations in `excluded` and every observation whose required SSR/HAS bias does
/// not hold at `state`'s position are left out first, labelled `pass`, and the arc checked
/// for sufficiency after that step; then the elevation cutoff, with its own check; then
/// the epochs left with no observations are dropped, with their receiver clocks.
///
/// SSR exclusion runs before the cutoff so that a satellite the source cannot place, and
/// whose biases therefore do not resolve, is left out instead of failing the cutoff's
/// geometry. An arc given with no observation at all is refused as invalid input.
pub(super) fn prepare_arc(
    arc: &ArcSettings<'_>,
    epochs: &[FloatEpoch],
    left_out: &LeftOut<'_>,
    state: &FloatState,
    pass: usize,
) -> Result<(PreparedArc, Vec<SsrBiasExclusion>), FloatSolveError> {
    let remaining = without_keys(
        &without_exclusions(epochs, left_out.excluded),
        left_out.screened,
    );
    // Observations admitted again after an exclusion are judged only at convergence, so
    // the check from the starting state leaves them in.
    let (_, new_exclusions) = exclude_unresolved_ssr_bias_observations(
        arc.source,
        &without_keys(&remaining, left_out.deferred),
        0,
        state.position_m,
        &arc.corrections.ppp,
        pass,
        SsrBiasExclusionStage::BeforeSolve,
    )?;
    let new_keys = new_exclusions
        .iter()
        .map(|exclusion| (exclusion.epoch_index, exclusion.ambiguity_id.clone()))
        .collect::<Vec<_>>();
    let bias_filtered = without_keys(&remaining, &new_keys);
    let excluded = left_out.excluded;
    let seed_ambiguities = left_out.seed_ambiguities;
    let excluded_count = excluded.len() + new_exclusions.len();
    if excluded_count > 0 {
        validate_ssr_bias_exclusion_retained(
            &bias_filtered,
            excluded_count,
            arc.tropo,
            arc.estimate_residual_ionosphere,
        )?;
    }
    let cut = match arc.elevation_cutoff_deg {
        Some(cutoff_deg) => apply_elevation_cutoff(
            arc.source,
            &bias_filtered,
            state,
            cutoff_deg,
            arc.tropo,
            arc.estimate_residual_ionosphere,
        )?,
        None => bias_filtered,
    };
    let (solved, indices) = drop_empty_epochs(cut);
    if solved.is_empty() {
        return Err(FloatSolveError::InvalidInput {
            field: "ppp epochs",
            reason: "no epoch has an observation",
        });
    }
    let mut solved_state = state.clone();
    solved_state.clocks_m = indices.iter().map(|&index| state.clocks_m[index]).collect();
    // An ambiguity the state lacks, such as one whose only observations were excluded on
    // the pass the state comes from and are now admitted again, starts from the caller's
    // seed, or from phase minus code of its first observation as `initial_ambiguities`
    // seeds it.
    for obs in solved.iter().flat_map(|epoch| epoch.observations.iter()) {
        if !solved_state.ambiguities_m.contains_key(&obs.ambiguity_id) {
            let seed = seed_ambiguities
                .get(&obs.ambiguity_id)
                .copied()
                .unwrap_or(obs.phase_m - obs.code_m);
            solved_state
                .ambiguities_m
                .insert(obs.ambiguity_id.clone(), seed);
        }
    }
    Ok((
        PreparedArc {
            epochs: solved,
            correction_epoch_indices: indices,
            state: solved_state,
        },
        new_exclusions,
    ))
}

/// What a solve pass leaves out of the input epochs, and the ambiguity seeds for the
/// observations it keeps.
pub(super) struct LeftOut<'a> {
    /// Exclusions already made.
    pub(super) excluded: &'a [SsrBiasExclusion],
    /// Observations the residual screen removed, as (input epoch index, ambiguity id).
    pub(super) screened: &'a [(usize, String)],
    /// Observations admitted again after an exclusion, which the check from the starting
    /// state leaves in.
    pub(super) deferred: &'a [(usize, String)],
    /// Ambiguity values for observations the state has none for.
    pub(super) seed_ambiguities: &'a BTreeMap<String, f64>,
}

/// `epochs` without the observations whose (input epoch index, ambiguity id) is in `keys`,
/// epochs kept in place.
pub(super) fn without_keys(epochs: &[FloatEpoch], keys: &[(usize, String)]) -> Vec<FloatEpoch> {
    let keys = keys
        .iter()
        .map(|(epoch_index, ambiguity_id)| (*epoch_index, ambiguity_id.as_str()))
        .collect::<BTreeSet<_>>();
    epochs
        .iter()
        .enumerate()
        .map(|(epoch_index, epoch)| {
            let mut epoch = epoch.clone();
            epoch
                .observations
                .retain(|obs| !keys.contains(&(epoch_index, obs.ambiguity_id.as_str())));
            epoch
        })
        .collect()
}

/// `epochs` without the observations in `excluded`, epochs kept in place.
pub(super) fn without_exclusions(
    epochs: &[FloatEpoch],
    excluded: &[SsrBiasExclusion],
) -> Vec<FloatEpoch> {
    let keys = excluded
        .iter()
        .map(|exclusion| (exclusion.epoch_index, exclusion.ambiguity_id.as_str()))
        .collect::<BTreeSet<_>>();
    epochs
        .iter()
        .enumerate()
        .map(|(epoch_index, epoch)| {
            let mut epoch = epoch.clone();
            epoch
                .observations
                .retain(|obs| !keys.contains(&(epoch_index, obs.ambiguity_id.as_str())));
            epoch
        })
        .collect()
}

/// `prior`, an input-indexed state, updated from a state over the solved epochs whose
/// input indices are `indices`: position, troposphere, ambiguities and ionosphere from
/// `solved`, and the receiver clocks of the solved epochs.
fn input_state(solved: &FloatState, indices: &[usize], prior: &FloatState) -> FloatState {
    let mut state = solved.clone();
    state.clocks_m = prior.clocks_m.clone();
    for (clock, &index) in solved.clocks_m.iter().zip(indices) {
        state.clocks_m[index] = *clock;
    }
    state
}

/// Where a fixed-point solve starts: the exclusions already made, the observations that
/// are never admitted again, those already admitted again once, and the first pass number.
#[derive(Default)]
pub(super) struct FixedPointStart {
    pub(super) exclusions: Vec<SsrBiasExclusion>,
    /// Excluded for good.
    pub(super) pinned: Vec<(usize, String)>,
    /// Admitted again once already: their records are judged only at convergence, and
    /// they are not admitted again.
    pub(super) readmitted: Vec<(usize, String)>,
    /// Admitted again after a fixed solve: judged at the fixed position, not at this
    /// solve's convergence.
    pub(super) judged_after_fix: Vec<(usize, String)>,
    pub(super) first_pass: usize,
}

/// Solve a static float arc to a fixed point of the SSR/HAS bias exclusion.
///
/// Each pass prepares the arc ([`prepare_arc`]) and solves it with `solve`. When a row's
/// recorded biases stop holding at the transmission time of an iteration, including one of
/// the residual screen's re-solves, or no longer hold at the position the pass converged
/// to for an observation the solution kept, that observation joins the exclusions and the
/// solve starts again from the state it reached.
///
/// Those exclusions depend on a position the solve had not converged to. Once a pass
/// converges with nothing new to exclude, every exclusion made for a transmit-time reason
/// is checked again at the converged position; those that hold are admitted again, once
/// each, and the solve continues. The rows do not check an admitted observation's records
/// at intermediate states; the check at convergence decides, and an observation that
/// fails it stays excluded, so one that keeps crossing a boundary is excluded
/// deterministically. Each pass excludes or admits at least one observation, so there are
/// at most three passes per observation.
pub(super) fn solve_to_ssr_fixed_point<F>(
    arc: &ArcSettings<'_>,
    epochs: &[FloatEpoch],
    initial_state: FloatState,
    start: FixedPointStart,
    mut solve: F,
) -> Result<FloatSolution, FloatSolveError>
where
    F: FnMut(ModelContext, &[FloatEpoch], FloatState) -> Result<FloatSolution, StepError>,
{
    let observation_count = epochs.iter().map(|e| e.observations.len()).sum::<usize>();
    let seed_ambiguities = initial_state.ambiguities_m.clone();
    let FixedPointStart {
        mut exclusions,
        pinned,
        mut readmitted,
        judged_after_fix,
        first_pass,
    } = start;
    let mut state = initial_state;
    let mut pass = first_pass;
    let pass_limit = observation_count
        .checked_mul(3)
        .and_then(|count| count.checked_add(1))
        .and_then(|count| first_pass.checked_add(count))
        .ok_or_else(pass_overflow)?;
    loop {
        // Each pass excludes or admits at least one observation; an observation is
        // excluded at most twice and admitted at most once.
        debug_assert!(
            pass <= pass_limit,
            "each pass excludes or admits an observation"
        );
        let (prepared, new_exclusions) = prepare_arc(
            arc,
            epochs,
            &LeftOut {
                excluded: &exclusions,
                screened: &[],
                deferred: &readmitted,
                seed_ambiguities: &seed_ambiguities,
            },
            &state,
            pass,
        )?;
        exclusions.extend(new_exclusions);
        pass = pass.checked_add(1).ok_or_else(pass_overflow)?;
        let ctx = ModelContext {
            source: arc.source,
            weights: arc.weights,
            tropo: arc.tropo,
            corrections: arc.corrections,
            normal: arc.normal,
            estimate_residual_ionosphere: arc.estimate_residual_ionosphere,
            correction_epoch_indices: Some(&prepared.correction_epoch_indices),
            ssr_bias_pass: pass,
            ssr_bias_stage: SsrBiasExclusionStage::DuringIteration,
            ssr_bias_deferred: &readmitted,
        };
        let mut solution = match solve(ctx, &prepared.epochs, prepared.state.clone()) {
            Ok(solution) => solution,
            Err(StepError::Flip(flip)) => {
                let StepFlip {
                    exclusion,
                    state: reached,
                    state_epoch_indices,
                } = *flip;
                exclusions.push(exclusion);
                state = input_state(&reached, &state_epoch_indices, &state);
                continue;
            }
            Err(StepError::Float(error)) => return Err(error),
        };
        let solved_state = input_state(
            &state_from_solution(&solution, &state),
            &solution.solved_epoch_indices,
            &state,
        );

        // Re-check the observations the solution kept at the converged position,
        // including those the rows did not check because they were admitted again.
        let late = kept_ssr_bias_failures(arc, &prepared, &solution, pass)?
            .into_iter()
            .filter(|exclusion| {
                !judged_after_fix.iter().any(|(epoch_index, ambiguity_id)| {
                    *epoch_index == exclusion.epoch_index && *ambiguity_id == exclusion.ambiguity_id
                })
            })
            .collect::<Vec<_>>();
        if !late.is_empty() {
            exclusions.extend(late);
            state = solved_state;
            continue;
        }

        // Admit again, once each, the transmit-time exclusions that hold at convergence.
        let before = exclusions.len();
        let mut still_excluded = Vec::with_capacity(before);
        for exclusion in exclusions.drain(..) {
            let key = (exclusion.epoch_index, exclusion.ambiguity_id.clone());
            if exclusion.transmit_time_failure.is_none()
                || pinned.contains(&key)
                || readmitted.contains(&key)
            {
                still_excluded.push(exclusion);
            } else if ssr_bias_holds_at(arc, epochs, &key, solution.position_m, pass)? {
                readmitted.push(key);
            } else {
                still_excluded.push(exclusion);
            }
        }
        exclusions = still_excluded;
        if exclusions.len() < before {
            state = solved_state;
            continue;
        }
        solution.ssr_bias_exclusions = exclusions;
        solution.ssr_bias_readmissions = readmitted;
        solution.ssr_bias_last_pass = pass;
        return Ok(solution);
    }
}

/// The observations `solution` kept whose SSR/HAS bias records do not hold at its
/// position, as exclusions found at convergence on pass `pass`.
fn kept_ssr_bias_failures(
    arc: &ArcSettings<'_>,
    prepared: &PreparedArc,
    solution: &FloatSolution,
    pass: usize,
) -> Result<Vec<SsrBiasExclusion>, FloatSolveError> {
    let kept = solution
        .residuals_m
        .iter()
        .map(|residual| (residual.epoch_index, residual.ambiguity_id.as_str()))
        .collect::<BTreeSet<_>>();
    let failures = prepared
        .epochs
        .iter()
        .zip(&prepared.correction_epoch_indices)
        .map(|(epoch, &epoch_index)| {
            let mut kept_epoch = epoch.clone();
            kept_epoch
                .observations
                .retain(|obs| kept.contains(&(epoch_index, obs.ambiguity_id.as_str())));
            exclude_unresolved_ssr_bias_observations(
                arc.source,
                std::slice::from_ref(&kept_epoch),
                epoch_index,
                solution.position_m,
                &arc.corrections.ppp,
                pass,
                SsrBiasExclusionStage::AtConvergence,
            )
            .map(|(_, exclusions)| exclusions)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(failures.into_iter().flatten().collect())
}

/// Whether the SSR/HAS bias records of the observation `key` names in `epochs` hold at the
/// transmission time from `position_m`, or the UT1 refusal met checking them.
pub(super) fn ssr_bias_holds_at(
    arc: &ArcSettings<'_>,
    epochs: &[FloatEpoch],
    key: &(usize, String),
    position_m: [f64; 3],
    pass: usize,
) -> Result<bool, FloatSolveError> {
    let Some(epoch) = epochs.get(key.0) else {
        return Ok(false);
    };
    let mut single = epoch.clone();
    single.observations.retain(|obs| obs.ambiguity_id == key.1);
    if single.observations.is_empty() {
        return Ok(false);
    }
    let (_, failures) = exclude_unresolved_ssr_bias_observations(
        arc.source,
        std::slice::from_ref(&single),
        key.0,
        position_m,
        &arc.corrections.ppp,
        pass,
        SsrBiasExclusionStage::AtConvergence,
    )?;
    Ok(failures.is_empty())
}

/// The pass a solve continuing from `float_solution` and `exclusions` numbers first: one
/// past the float solve's last pass and the highest pass the exclusions were found on.
pub(super) fn next_pass(
    float_solution: &FloatSolution,
    exclusions: &[SsrBiasExclusion],
) -> Result<usize, FloatSolveError> {
    exclusions
        .iter()
        .map(|exclusion| exclusion.pass)
        .chain(std::iter::once(float_solution.ssr_bias_last_pass))
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(pass_overflow)
}

/// The SSR/HAS bias pass counter would pass `usize::MAX`.
fn pass_overflow() -> FloatSolveError {
    FloatSolveError::InvalidInput {
        field: "ppp ssr_bias pass",
        reason: "overflows the pass counter",
    }
}

/// A solve step failure: SSR/HAS bias records that stopped holding at the state the rows
/// were built at, which a fixed-point solve turns into an exclusion, or any other error.
pub(super) enum StepError {
    Flip(Box<StepFlip>),
    Float(FloatSolveError),
}

/// The observation whose SSR/HAS bias records stopped holding, and the state the rows were
/// built at with the input epoch index of each of its receiver clocks.
pub(super) struct StepFlip {
    pub(super) exclusion: SsrBiasExclusion,
    pub(super) state: FloatState,
    pub(super) state_epoch_indices: Vec<usize>,
}

impl From<FloatSolveError> for StepError {
    fn from(error: FloatSolveError) -> Self {
        Self::Float(error)
    }
}

impl StepError {
    /// A row error at `state`, the state over `epochs` in `ctx`.
    pub(super) fn from_rows(
        error: PppRowError,
        ctx: ModelContext,
        epochs: &[FloatEpoch],
        state: &FloatState,
    ) -> Self {
        match error {
            PppRowError::SsrBiasFlip { exclusion, .. } => Self::Flip(Box::new(StepFlip {
                exclusion: *exclusion,
                state: state.clone(),
                state_epoch_indices: (0..epochs.len())
                    .map(|epoch_idx| ctx.correction_epoch_index(epoch_idx))
                    .collect(),
            })),
            other => Self::Float(other.into_float()),
        }
    }
}

pub(super) fn screened_float_solution(
    ctx: ModelContext,
    solve_epochs: &[FloatEpoch],
    state: FloatState,
    opts: FloatSolveOptions,
    residual_screen: bool,
) -> Result<FloatSolution, StepError> {
    let ambiguity_ids = multi_ambiguity_ids(solve_epochs);
    let solution = iterate_multi_steps(ctx, solve_epochs, &ambiguity_ids, state.clone(), opts, 1)?;

    if !residual_screen {
        return Ok(solution);
    }

    let unscreened_wrms = solution_weighted_rms(ctx, solve_epochs, &solution, &state);
    let screen = ScreenEpochs {
        epochs: solve_epochs.to_vec(),
        correction_indices: (0..solve_epochs.len())
            .map(|epoch_idx| ctx.correction_epoch_index(epoch_idx))
            .collect(),
        removed: Vec::new(),
    };
    match run_residual_screen(ctx, screen, state, opts, solution.clone(), 1)? {
        ScreenResult::Clean => Ok(solution),
        ScreenResult::Screened {
            solution: screened,
            epochs:
                ScreenEpochs {
                    epochs: retained,
                    correction_indices,
                    removed,
                },
        } => {
            let screened_wrms = solution_weighted_rms(
                ModelContext {
                    correction_epoch_indices: Some(&correction_indices),
                    ..ctx
                },
                &retained,
                screened.as_ref(),
                &state_from_solution(&screened, &FloatState::default_for_epochs(&retained)),
            );
            if screened_wrms.is_finite()
                && unscreened_wrms.is_finite()
                && screened_wrms * RESIDUAL_SCREEN_ACCEPT_FACTOR < unscreened_wrms
            {
                let mut screened = *screened;
                screened.residual_screen_removals = removed;
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
        solution: Box<FloatSolution>,
        epochs: ScreenEpochs,
    },
}

/// Epochs still in a residual screen and, for each, the caller epoch index that keys its
/// corrections. An epoch whose last observation the screen removes is dropped from
/// `epochs`, since its receiver clock would have no rows, and the later epochs keep their
/// own corrections through `correction_indices`.
struct ScreenEpochs {
    epochs: Vec<FloatEpoch>,
    correction_indices: Vec<usize>,
    /// Observations the screen removed, as (input epoch index, ambiguity id).
    removed: Vec<(usize, String)>,
}

/// `ctx` keys corrections by the caller's epochs, as the unscreened solve does; each pass
/// solves `screen.epochs` against the corrections of `screen.correction_indices`.
fn run_residual_screen(
    ctx: ModelContext,
    screen: ScreenEpochs,
    seed_state: FloatState,
    opts: FloatSolveOptions,
    solution: FloatSolution,
    pass: usize,
) -> Result<ScreenResult, StepError> {
    if pass > RESIDUAL_SCREEN_MAX_PASSES {
        return Ok(ScreenResult::Screened {
            solution: Box::new(solution),
            epochs: screen,
        });
    }

    let candidate_state = state_from_solution(&solution, &seed_state);
    let screen_ctx = ModelContext {
        correction_epoch_indices: Some(&screen.correction_indices),
        ..ctx
    };
    match worst_multi_residual(screen_ctx, &screen.epochs, &candidate_state)? {
        Some((epoch_index, ambiguity_id)) => {
            let pruned = exclude_observation(&screen, epoch_index, &ambiguity_id);
            if !multi_enough_after_prune(
                &pruned.epochs,
                ctx.tropo,
                ctx.estimate_residual_ionosphere,
            ) {
                return Ok(ScreenResult::Screened {
                    solution: Box::new(solution),
                    epochs: screen,
                });
            }
            let ambiguity_ids = multi_ambiguity_ids(&pruned.epochs);
            let candidate = iterate_multi_steps(
                ModelContext {
                    correction_epoch_indices: Some(&pruned.correction_indices),
                    ..ctx
                },
                &pruned.epochs,
                &ambiguity_ids,
                reseed_state(&seed_state, &pruned.epochs),
                opts,
                1,
            )?;
            run_residual_screen(ctx, pruned, seed_state, opts, candidate, pass + 1)
        }
        None => {
            if pass == 1 {
                Ok(ScreenResult::Clean)
            } else {
                Ok(ScreenResult::Screened {
                    solution: Box::new(solution),
                    epochs: screen,
                })
            }
        }
    }
}

fn iterate_multi_steps(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    ambiguity_ids: &[AmbiguityId],
    state: FloatState,
    opts: FloatSolveOptions,
    iter: usize,
) -> Result<FloatSolution, StepError> {
    let mut current = state;
    let mut iteration = iter;
    let max_iterations = opts.max_iterations;

    loop {
        let binding = AmbiguityBinding::Estimated {
            ids: ambiguity_ids,
            values: &current.ambiguities_m,
        };
        let rows = build_rows(ctx, epochs, &binding, &current)
            .map_err(|error| StepError::from_rows(error, ctx, epochs, &current))?;
        let layout = PppNormalLayout::new(
            epochs.len(),
            ztd_unknown_count(ctx.tropo),
            tropo_gradient_unknown_count(ctx.tropo),
            residual_ionosphere_unknown_count(
                ctx.estimate_residual_ionosphere,
                ambiguity_ids.len(),
            ),
            ambiguity_ids.len(),
        );
        let dx = solve_normal_equations(&rows, layout, ctx.normal)?;
        let next = apply_multi_delta(
            &current,
            epochs.len(),
            ambiguity_ids,
            &dx,
            ctx.tropo,
            ctx.estimate_residual_ionosphere,
        )?;
        let (pos_step, clock_step, ztd_step, gradient_step, ambiguity_step) = multi_step_norms(
            &dx,
            epochs.len(),
            ctx.tropo,
            ctx.estimate_residual_ionosphere,
            ambiguity_ids.len(),
        );

        if pos_step <= opts.position_tolerance_m
            && clock_step <= opts.clock_tolerance_m
            && ztd_step <= opts.ztd_tolerance_m
            && gradient_step <= opts.ztd_tolerance_m
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
    estimate_residual_ionosphere: bool,
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
        let ionosphere_deltas = &dx[idx..idx + ambiguity_ids.len()];
        idx += ambiguity_ids.len();
        for (id, delta) in ambiguity_ids.iter().zip(ionosphere_deltas) {
            let prior = state
                .residual_ionosphere_m
                .get(id.as_str())
                .copied()
                .unwrap_or(0.0);
            residual_ionosphere_m.insert(id.as_str().to_string(), prior + delta);
        }
    }
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
        tropo_gradient_north_m: state.tropo_gradient_north_m + tropo_gradient_north_delta,
        tropo_gradient_east_m: state.tropo_gradient_east_m + tropo_gradient_east_delta,
        residual_ionosphere_m,
    })
}

fn multi_step_norms(
    dx: &[f64],
    n_epochs: usize,
    tropo: TroposphereOptions,
    estimate_residual_ionosphere: bool,
    n_ambiguities: usize,
) -> (f64, f64, f64, f64, f64) {
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
    let gradient = if estimates_tropo_gradients(tropo) {
        let v = max_abs(&dx[idx..idx + 2]);
        idx += 2;
        v
    } else {
        0.0
    };
    let ionosphere = if estimate_residual_ionosphere {
        let v = max_abs(&dx[idx..idx + n_ambiguities]);
        idx += n_ambiguities;
        v
    } else {
        0.0
    };
    let ambiguity = max_abs(&dx[idx..]);
    (pos, clock, ztd, gradient, ambiguity.max(ionosphere))
}

fn finalize_multi(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    ambiguity_ids: &[AmbiguityId],
    state: FloatState,
    iterations: usize,
    converged: bool,
    status: FloatStatus,
) -> Result<FloatSolution, StepError> {
    // A flip the rows find while the solution is assembled is found at convergence.
    let ctx = ModelContext {
        ssr_bias_stage: SsrBiasExclusionStage::AtConvergence,
        ..ctx
    };
    let residuals = residual_rows(ctx, epochs, &state.ambiguities_m, &state)
        .map_err(|error| StepError::from_rows(error, ctx, epochs, &state))?;
    let binding = AmbiguityBinding::Estimated {
        ids: ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let rows = build_rows(ctx, epochs, &binding, &state)
        .map_err(|error| StepError::from_rows(error, ctx, epochs, &state))?;
    let covariance = ppp_position_covariance(
        &rows,
        PppNormalLayout::new(
            epochs.len(),
            ztd_unknown_count(ctx.tropo),
            tropo_gradient_unknown_count(ctx.tropo),
            residual_ionosphere_unknown_count(
                ctx.estimate_residual_ionosphere,
                ambiguity_ids.len(),
            ),
            ambiguity_ids.len(),
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
    let solution = FloatSolution {
        position_m: state.position_m,
        position_covariance: covariance.scaled,
        formal_position_covariance: covariance.formal,
        posterior_variance_factor: covariance.posterior_variance_factor,
        position_covariance_scale_factor: covariance.covariance_scale_factor,
        temporal_position_covariance,
        temporal_position_covariance_scale_factor,
        temporal_correlation,
        epoch_clocks_m: state.clocks_m,
        ambiguities_m: state.ambiguities_m,
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
        solved_epoch_indices,
        ssr_bias_exclusions: Vec::new(),
        ssr_bias_readmissions: Vec::new(),
        ssr_bias_last_pass: 0,
        residual_screen: false,
        solve_options: FloatSolveOptions::default(),
        residual_screen_removals: Vec::new(),
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
) -> Result<Option<(usize, String)>, StepError> {
    let rows = residual_rows(ctx, epochs, &state.ambiguities_m, state)
        .map_err(|error| StepError::from_rows(error, ctx, epochs, state))?;
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
                    r.ambiguity_id.clone(),
                ),
                (
                    normalized_residual(
                        ResidualNormRecipe::PppInverseSigmaMagnitude,
                        r.phase_m,
                        r.phase_weight,
                    ),
                    r.epoch_index,
                    r.ambiguity_id.clone(),
                ),
            ]
        })
        .max_by(|a, b| a.0.total_cmp(&b.0));
    Ok(match candidate {
        Some((normalized, epoch_index, ambiguity_id)) if normalized > RESIDUAL_SCREEN_THRESHOLD => {
            Some((epoch_index, ambiguity_id))
        }
        _ => None,
    })
}

/// Remove the observation with ambiguity id `drop_ambiguity_id` from the screened epoch
/// whose input epoch index is `drop_epoch_index`; other observations of the same satellite
/// stay. An epoch left with no observations is dropped together with its correction index,
/// so every remaining epoch keeps the corrections of its own input epoch.
fn exclude_observation(
    screen: &ScreenEpochs,
    drop_epoch_index: usize,
    drop_ambiguity_id: &str,
) -> ScreenEpochs {
    let mut pruned = ScreenEpochs {
        epochs: Vec::with_capacity(screen.epochs.len()),
        correction_indices: Vec::with_capacity(screen.epochs.len()),
        removed: screen.removed.clone(),
    };
    pruned
        .removed
        .push((drop_epoch_index, drop_ambiguity_id.to_string()));
    for (epoch, correction_index) in screen.epochs.iter().zip(&screen.correction_indices) {
        let mut epoch = epoch.clone();
        if *correction_index == drop_epoch_index {
            epoch
                .observations
                .retain(|obs| obs.ambiguity_id != drop_ambiguity_id);
        }
        if !epoch.observations.is_empty() {
            pruned.epochs.push(epoch);
            pruned.correction_indices.push(*correction_index);
        }
    }
    pruned
}

fn multi_enough_after_prune(
    epochs: &[FloatEpoch],
    tropo: TroposphereOptions,
    estimate_residual_ionosphere: bool,
) -> bool {
    if epochs.len() < 2 {
        return false;
    }
    let n_sats = multi_ambiguity_ids(epochs).len();
    let n_obs: usize = epochs.iter().map(|e| e.observations.len()).sum();
    let equations = 2 * n_obs;
    let unknowns = ParameterLayout::ppp(
        epochs.len(),
        ztd_unknown_count(tropo),
        tropo_gradient_unknown_count(tropo),
        residual_ionosphere_unknown_count(estimate_residual_ionosphere, n_sats),
        n_sats,
    )
    .dim();
    n_sats >= 4 && equations >= unknowns
}

fn reseed_state(state: &FloatState, epochs: &[FloatEpoch]) -> FloatState {
    FloatState {
        position_m: state.position_m,
        clocks_m: vec![state.clocks_m[0]; epochs.len()],
        ambiguities_m: initial_ambiguities(epochs),
        ztd_m: state.ztd_m,
        tropo_gradient_north_m: state.tropo_gradient_north_m,
        tropo_gradient_east_m: state.tropo_gradient_east_m,
        residual_ionosphere_m: BTreeMap::new(),
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
