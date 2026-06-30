//! Static integer-fixed PPP solve.
//!
//! This leaf owns the fixed-ambiguity orchestration: the LAMBDA integer search
//! from a float solution, the ambiguity-conditioned multi-epoch re-solve, the
//! post-fit residual rows, and the cycle/metre ambiguity conversions. The shared
//! measurement model lives in [`super::model`], the dense normal-equation kernel
//! in [`super::normal`], and the row staging / shared scalar helpers in [`super`].

use std::collections::BTreeMap;

use crate::ambiguity::AmbiguityId;
use crate::astro::math::vec3;
use crate::estimation::recipe::{EstimationRecipe, NormalRecipe};
use crate::estimation::substrate::ambiguity::resolve_integer_lattice;
use crate::estimation::substrate::parameters::ParameterLayout;
use crate::observables::ObservableEphemerisSource;

use super::normal::{ambiguity_covariance_from_normal, normal_equations, solve_normal_equations};
use super::rows::{build_rows, residual_rows, AmbiguityBinding, PppRowError};
use super::{
    estimates_ztd, max_abs, rms, state_from_solution, validate_fixed_solve_boundary, weighted_rms,
    ztd_unknown_count, AmbiguitySearch, FixedIntegerMetadata, FixedSolution, FixedSolveConfig,
    FixedSolveError, FloatEpoch, FloatSolution, FloatSolveError, FloatSolveOptions, FloatState,
    FloatStatus, IntegerStatus, ModelContext, TroposphereOptions,
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
    let fixed_meta = search_integer_ambiguities(source, epochs, &float_solution, &config)?;
    let fixed_m = fixed_ambiguities_m(
        &fixed_meta.fixed_cycles,
        &config.ambiguity.wavelengths_m,
        &config.ambiguity.offsets_m,
    )?;
    let initial_state = fixed_state_from_float(&float_solution);
    let ctx = ModelContext {
        source,
        weights: config.weights,
        tropo: config.tropo,
        corrections: &config.corrections,
        normal: recipe.normal,
    };
    let resolve = iterate_fixed_multi(ctx, epochs, &fixed_m, initial_state, config.opts, 1)?;
    finalize_fixed_multi(ctx, epochs, fixed_meta, fixed_m, float_solution, resolve)
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

fn search_integer_ambiguities(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    float_solution: &FloatSolution,
    config: &FixedSolveConfig,
) -> Result<FixedSearchResult, FixedSolveError> {
    let order: Vec<AmbiguityId> = float_solution
        .used_sats
        .iter()
        .map(|sat| AmbiguityId::new(sat.clone()))
        .collect();
    let covariance_cycles =
        ambiguity_covariance_cycles(source, epochs, &order, float_solution, config)?;
    let float_cycles = float_ambiguities_cycles(
        float_solution,
        &config.ambiguity.wavelengths_m,
        &config.ambiguity.offsets_m,
    )?;
    let floats: Vec<f64> = order
        .iter()
        .map(|id| float_cycles.get(id.as_str()).copied().unwrap())
        .collect();
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

fn iterate_fixed_multi(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    fixed_m: &BTreeMap<String, f64>,
    state: FloatState,
    opts: FloatSolveOptions,
    iter: usize,
) -> Result<FixedResolve, FixedSolveError> {
    let n = ParameterLayout::ppp(epochs.len(), ztd_unknown_count(ctx.tropo), 0).dim();
    let mut current = state;
    let mut iteration = iter;
    let max_iterations = opts.max_iterations;

    loop {
        let binding = AmbiguityBinding::Held { values: fixed_m };
        let rows = build_rows(ctx, epochs, &binding, &current).map_err(PppRowError::into_fixed)?;
        let dx = solve_normal_equations(&rows, n, ctx.normal)?;
        let next = apply_fixed_multi_delta(&current, epochs.len(), &dx, ctx.tropo);
        let (pos_step, clock_step, ztd_step) = fixed_multi_step_norms(&dx, ctx.tropo);

        if pos_step <= opts.position_tolerance_m
            && clock_step <= opts.clock_tolerance_m
            && ztd_step <= opts.ztd_tolerance_m
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
    dx: &[f64],
    tropo: TroposphereOptions,
) -> FloatState {
    let mut idx = 3;
    let clock_deltas = &dx[idx..idx + n_epochs];
    idx += n_epochs;
    let ztd_delta = if estimates_ztd(tropo) { dx[idx] } else { 0.0 };
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
    }
}

fn fixed_multi_step_norms(dx: &[f64], tropo: TroposphereOptions) -> (f64, f64, f64) {
    let pos = vec3::norm3([dx[0], dx[1], dx[2]]);
    let n_ztd = ztd_unknown_count(tropo);
    let n_clocks = dx.len() - 3 - n_ztd;
    let clock = max_abs(&dx[3..3 + n_clocks]);
    let ztd = if estimates_ztd(tropo) {
        dx[3 + n_clocks].abs()
    } else {
        0.0
    };
    (pos, clock, ztd)
}

fn finalize_fixed_multi(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    search: FixedSearchResult,
    fixed_m: BTreeMap<String, f64>,
    float_solution: FloatSolution,
    resolve: FixedResolve,
) -> Result<FixedSolution, FixedSolveError> {
    let FixedResolve {
        state,
        iterations,
        converged,
        status,
    } = resolve;
    let residuals =
        residual_rows(ctx, epochs, &fixed_m, &state).map_err(PppRowError::into_fixed)?;
    let code: Vec<f64> = residuals.iter().map(|r| r.code_m).collect();
    let phase: Vec<f64> = residuals.iter().map(|r| r.phase_m).collect();
    Ok(FixedSolution {
        position_m: state.position_m,
        epoch_clocks_m: state.clocks_m,
        fixed_ambiguities_cycles: search.fixed_cycles,
        fixed_ambiguities_m: fixed_m,
        ztd_residual_m: if estimates_ztd(ctx.tropo) {
            Some(state.ztd_m)
        } else {
            None
        },
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
    })
}

fn fixed_state_from_float(solution: &FloatSolution) -> FloatState {
    FloatState {
        position_m: solution.position_m,
        clocks_m: solution.epoch_clocks_m.clone(),
        ambiguities_m: BTreeMap::new(),
        ztd_m: solution.ztd_residual_m.unwrap_or(0.0),
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
    ambiguity_ids: &[AmbiguityId],
    float_solution: &FloatSolution,
    config: &FixedSolveConfig,
) -> Result<Vec<Vec<f64>>, FixedSolveError> {
    let state = state_from_solution(float_solution, &FloatState::default_for_epochs(epochs));
    let layout = ParameterLayout::ppp(
        epochs.len(),
        ztd_unknown_count(config.tropo),
        ambiguity_ids.len(),
    );
    let n = layout.dim();
    let start = layout.ambiguity_offset();
    let ctx = ModelContext {
        source,
        weights: config.weights,
        tropo: config.tropo,
        corrections: &config.corrections,
        // Covariance assembly uses the const last-tie assembler directly; the
        // recipe field is the PPP reference and unused on this path.
        normal: NormalRecipe::PppDenseLastTie,
    };
    let binding = AmbiguityBinding::Estimated {
        ids: ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let rows = build_rows(ctx, epochs, &binding, &state)
        .map_err(|e| FixedSolveError::from(e.into_float()))?;
    let (normal, _rhs) = normal_equations(&rows, n)?;
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
