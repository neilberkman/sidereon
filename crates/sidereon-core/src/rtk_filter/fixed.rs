//! Static batch integer-fixed RTK baseline solver.
//!
//! The LAMBDA-fixed counterpart of the static `float` solve: given a prior float
//! baseline and its ambiguity covariance, run the integer ambiguity search
//! (`search` submodule), then re-solve the double-difference least squares with
//! the fixed ambiguities held and the free ones estimated. Also hosts the
//! residual-validation / FDE orchestration (`solve_fixed_baseline_validated`)
//! that the public Sidereon fixed-baseline API drives. The arithmetic and operation
//! order are unchanged from the original in-module implementation, so the
//! frozen-bits goldens and real-arc LAMBDA ratios are identical.

use std::collections::BTreeMap;

use crate::astro::math::vec3;
use crate::estimation::recipe::{EstimationRecipe, NormalRecipe, SolverRecipe};
use crate::estimation::substrate::parameters::ParameterLayout;

use super::float::{
    float_normal_equations, float_residuals, run_float, solve_dd_normal_into,
    validate_float_solve_opts, FloatBaselineSolution, FloatResidual, FloatSolveError,
    FloatSolveOpts, FloatSolveScratch, FloatSolveStatus, SolveOutcome,
};
use super::model::{float_only_set, satellite_system};
use super::search::{
    covariance_m_to_cycles, covariance_submatrix, empty_integer_search_meta, float_cycles_for_ids,
    float_only_ambiguity_ids, partial_meta, search_ambiguity_ids, search_partial_fixed_ambiguities,
    FixedSearchResult, IntegerSearchMeta, IntegerStatus, PartialSearchInputs,
};
#[cfg(test)]
use super::EpochRowsScratch;
use super::{
    assign_double_difference_ambiguity_id, dd_epoch_rows_into, rms, AmbiguityScale, DdRowError,
    DdRowRecipe, Epoch, MeasContext, MeasModel, ReceiverAntennaCorrections, ReceiverAntennaError,
};
use crate::ambiguity::AmbiguityId;
use crate::validate;

/// Controls for the static fixed RTK solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FixedSolveOpts {
    pub position_tol_m: f64,
    pub ambiguity_tol_m: f64,
    pub max_iterations: usize,
    pub ratio_threshold: f64,
    pub partial_ambiguity_resolution: bool,
    pub partial_min_ambiguities: usize,
}

impl Default for FixedSolveOpts {
    /// Canonical fixed-solve controls, read from [`super::defaults`]. Partial
    /// ambiguity resolution is off by default (the core goldens and bindings run
    /// full-set resolution). Exposing this changes no solve; the runners still
    /// read the caller's opts.
    fn default() -> Self {
        Self {
            position_tol_m: super::defaults::POSITION_TOL_M,
            ambiguity_tol_m: super::defaults::AMBIGUITY_TOL_M,
            max_iterations: super::defaults::MAX_ITERATIONS,
            ratio_threshold: super::defaults::RATIO_THRESHOLD,
            partial_ambiguity_resolution: false,
            partial_min_ambiguities: super::defaults::PARTIAL_MIN_AMBIGUITIES,
        }
    }
}

/// Static batch integer-fixed RTK solution.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedBaselineSolution {
    pub baseline_m: [f64; 3],
    pub free_ambiguities_m: Vec<(String, f64)>,
    pub fixed_ambiguities_cycles: Vec<(String, i64)>,
    pub fixed_ambiguities_m: Vec<(String, f64)>,
    pub residuals: Vec<FloatResidual>,
    pub search: IntegerSearchMeta,
    pub iterations: usize,
    pub converged: bool,
    pub status: FloatSolveStatus,
    pub code_rms_m: f64,
    pub phase_rms_m: f64,
    pub weighted_rms_m: f64,
    pub n_observations: usize,
}

/// Code or phase component selected by the RTK residual-validation gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidualComponentKind {
    Code,
    Phase,
}

/// Worst normalized residual selected by the RTK validation gate.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualValidationOutlier {
    pub epoch_index: usize,
    pub satellite_id: String,
    pub reference_satellite_id: String,
    pub ambiguity_id: String,
    pub kind: ResidualComponentKind,
    pub residual_m: f64,
    pub sigma_m: f64,
    pub normalized_residual: f64,
    pub threshold_sigma: f64,
}

/// Optional residual-validation controls for fixed RTK baseline solving.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResidualValidationOpts {
    pub threshold_sigma: Option<f64>,
    pub max_exclusions: usize,
}

/// Residual-validation metadata for the accepted fixed RTK solution.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualValidationMeta {
    pub threshold_sigma: f64,
    pub max_exclusions: usize,
    pub excluded_sats: Vec<String>,
    pub exclusions: Vec<ResidualValidationOutlier>,
}

/// Fixed RTK solution plus the final float solve used by integer fixing.
#[derive(Debug, Clone, PartialEq)]
pub struct ValidatedFixedBaselineSolution {
    pub float_solution: FloatBaselineSolution,
    pub fixed_solution: FixedBaselineSolution,
    pub residual_validation: Option<ResidualValidationMeta>,
    pub ambiguity_ids: Vec<String>,
    pub ambiguity_satellites: BTreeMap<String, String>,
}

/// Why the residual-validated fixed RTK solve could not complete.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidatedFixedSolveError {
    Fixed(FixedSolveError),
    ResidualValidationFailed {
        outlier: Box<ResidualValidationOutlier>,
        exclusions: Vec<ResidualValidationOutlier>,
    },
    DuplicateAmbiguityId {
        ambiguity_id: String,
        first_satellite_id: String,
        second_satellite_id: String,
    },
    Underdetermined {
        row_count: usize,
        unknown_count: usize,
    },
}

impl core::fmt::Display for ValidatedFixedSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Fixed(error) => write!(f, "fixed RTK solve failed: {error}"),
            Self::ResidualValidationFailed {
                outlier,
                exclusions,
            } => write!(
                f,
                "fixed RTK residual validation failed on {} {} residual for satellite {} against reference {} with normalized residual {}; {} exclusions were already tried",
                outlier.ambiguity_id,
                residual_component_name(outlier.kind),
                outlier.satellite_id,
                outlier.reference_satellite_id,
                outlier.normalized_residual,
                exclusions.len()
            ),
            Self::DuplicateAmbiguityId {
                ambiguity_id,
                first_satellite_id,
                second_satellite_id,
            } => write!(
                f,
                "duplicate RTK ambiguity id {ambiguity_id} for satellites {first_satellite_id} and {second_satellite_id}"
            ),
            Self::Underdetermined {
                row_count,
                unknown_count,
            } => write!(
                f,
                "fixed RTK system is underdetermined: {row_count} rows for {unknown_count} unknowns"
            ),
        }
    }
}

impl std::error::Error for ValidatedFixedSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Fixed(error) => Some(error),
            _ => None,
        }
    }
}

fn residual_component_name(kind: ResidualComponentKind) -> &'static str {
    match kind {
        ResidualComponentKind::Code => "code",
        ResidualComponentKind::Phase => "phase",
    }
}

impl From<FixedSolveError> for ValidatedFixedSolveError {
    fn from(err: FixedSolveError) -> Self {
        Self::Fixed(err)
    }
}

/// Why a static fixed RTK solve could not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FixedSolveError {
    Float(FloatSolveError),
    Ils(crate::ils::IlsError),
    MissingAmbiguity(String),
    MissingWavelength(String),
    MissingOffset(String),
    InvalidCovarianceDimensions,
    InvalidInput {
        field: &'static str,
        kind: super::RtkInputErrorKind,
    },
    SingularGeometry,
    IncompleteResidualPair,
    ReceiverAntenna(ReceiverAntennaError),
}

impl core::fmt::Display for FixedSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Float(error) => write!(f, "RTK float prerequisite failed: {error}"),
            Self::Ils(error) => write!(f, "RTK integer ambiguity search failed: {error}"),
            Self::MissingAmbiguity(id) => write!(f, "missing RTK ambiguity {id}"),
            Self::MissingWavelength(id) => write!(f, "missing RTK wavelength for ambiguity {id}"),
            Self::MissingOffset(id) => write!(f, "missing RTK offset for ambiguity {id}"),
            Self::InvalidCovarianceDimensions => {
                write!(f, "RTK ambiguity covariance dimensions are invalid")
            }
            Self::InvalidInput { field, kind } => {
                write!(f, "invalid RTK fixed input {field}: {kind}")
            }
            Self::SingularGeometry => write!(f, "RTK fixed geometry is singular"),
            Self::IncompleteResidualPair => {
                write!(
                    f,
                    "RTK fixed residual rows are not complete code/phase pairs"
                )
            }
            Self::ReceiverAntenna(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for FixedSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Float(error) => Some(error),
            Self::Ils(error) => Some(error),
            Self::ReceiverAntenna(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FloatSolveError> for FixedSolveError {
    fn from(err: FloatSolveError) -> Self {
        match err {
            FloatSolveError::SingularGeometry => Self::SingularGeometry,
            FloatSolveError::IncompleteResidualPair => Self::IncompleteResidualPair,
            FloatSolveError::ReceiverAntenna(error) => Self::ReceiverAntenna(error),
            FloatSolveError::InvalidInput { field, kind } => Self::InvalidInput { field, kind },
            other => Self::Float(other),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct FixedState {
    baseline_m: [f64; 3],
    ambiguities_m: Vec<f64>,
}

/// The ambiguity partition the fixed-baseline row builders consume: the free
/// (still-estimated) ambiguity ids and the ambiguities already fixed to metres.
/// Carried together so the row/finalize argument lists stay small. Both are keyed
/// by the typed [`AmbiguityId`] so the row builder compares them against the
/// double-difference id it composes without confusing an ambiguity id with a raw
/// satellite token; they convert to `String` only when the public solution is
/// built.
#[derive(Clone, Copy)]
struct FixedAmbiguities<'a> {
    free_ids: &'a [AmbiguityId],
    fixed_m: &'a BTreeMap<AmbiguityId, f64>,
}

/// The double-difference ambiguity set a fixed RTK solve operates on: the ordered
/// ambiguity ids, the id -> rover-satellite map, the cycle/metre scaling, and the
/// systems held float-only. Carried together so the public fixed-baseline entry
/// points stay within a small argument list instead of threading the four pieces
/// separately.
#[derive(Clone, Copy)]
pub struct AmbiguitySet<'a> {
    pub ids: &'a [String],
    pub satellites: &'a BTreeMap<String, String>,
    pub scale: AmbiguityScale<'a>,
    pub float_only_systems: &'a [String],
}

/// The prior float baseline the integer-fixed re-solve conditions on: the float
/// baseline vector, the float ambiguity estimates (metres), and their covariance
/// (row-major, metres squared). Carried together so the fixed-baseline entry point
/// stays within a small argument list.
#[derive(Clone, Copy)]
pub struct FloatPrior<'a> {
    pub baseline_m: [f64; 3],
    pub ambiguities_m: &'a [(String, f64)],
    pub covariance_m: &'a [f64],
}

/// The three option groups the residual-validated fixed RTK solve drives: the
/// inner float solve, the integer-fixed solve, and the residual-validation gate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidatedFixedSolveOpts {
    pub float: FloatSolveOpts,
    pub fixed: FixedSolveOpts,
    pub residual: ResidualValidationOpts,
}

/// Solve a static multi-epoch integer-fixed RTK baseline from a prior float
/// solution. The caller supplies normalized epochs and the float ambiguity
/// covariance; this function owns the LAMBDA decision, optional partial AR, and
/// the ambiguity-conditioned re-solve.
///
/// Thin wrapper over [`solve_fixed_baseline_with_normal`] pinned to the RTK
/// reference normal-equation recipe
/// ([`NormalRecipe::RtkDoubleDifferenceBlockFirstTie`]), so this public entry
/// point is bit-identical to the legacy fixed re-solve. The runtime selector
/// drives the private runner with the resolved recipe's normal stage instead.
pub fn solve_fixed_baseline(
    epochs: &[Epoch],
    base: [f64; 3],
    ambiguities: AmbiguitySet,
    float_prior: FloatPrior,
    model: &MeasModel,
    opts: FixedSolveOpts,
    receiver_antenna_corrections: Option<&ReceiverAntennaCorrections>,
) -> Result<FixedBaselineSolution, FixedSolveError> {
    let ctx = MeasContext {
        base,
        model,
        antenna: receiver_antenna_corrections,
    };
    solve_fixed_baseline_with_normal(
        ctx,
        epochs,
        ambiguities,
        float_prior,
        opts,
        NormalRecipe::RtkDoubleDifferenceBlockFirstTie,
        SolverRecipe::FlatGaussianFirstTie,
    )
}

/// The integer-conditioned fixed re-solve under a selectable normal-equation
/// recipe: the held integers fold into the prefit, the free ambiguities and the
/// baseline are re-estimated, and each iteration's reduced system is solved by
/// [`NormalAssembler::new(normal)`] rather than a hard-coded assembler. The RTK
/// reference recipe ([`NormalRecipe::RtkDoubleDifferenceBlockFirstTie`]) selects
/// the same first-tie block fold the legacy path used, so the goldens are
/// unchanged; canonical RTK ([`NormalRecipe::CanonicalSquareRoot`] with
/// [`SolverRecipe::OwnedDeterministicCholesky`]) solves the same reduced system
/// by the owned Cholesky square-root factorization. `ctx` carries the
/// base/model/antenna measurement environment.
fn solve_fixed_baseline_with_normal(
    ctx: MeasContext,
    epochs: &[Epoch],
    ambiguities: AmbiguitySet,
    float_prior: FloatPrior,
    opts: FixedSolveOpts,
    normal: NormalRecipe,
    solver: SolverRecipe,
) -> Result<FixedBaselineSolution, FixedSolveError> {
    validate_fixed_solve_opts(opts)?;
    let AmbiguitySet {
        ids: ambiguity_ids,
        satellites: ambiguity_satellites,
        scale,
        float_only_systems,
    } = ambiguities;
    let FloatPrior {
        baseline_m: float_baseline_m,
        ambiguities_m: float_ambiguities_m,
        covariance_m: float_covariance_m,
    } = float_prior;

    let n_ambiguities = ambiguity_ids.len();
    if float_covariance_m.len() != n_ambiguities * n_ambiguities {
        return Err(FixedSolveError::InvalidCovarianceDimensions);
    }
    validate_float_prior_covariance(float_covariance_m, n_ambiguities)?;

    let float_ambiguities = float_ambiguities_m
        .iter()
        .cloned()
        .collect::<BTreeMap<_, _>>();
    for id in ambiguity_ids {
        if !float_ambiguities.contains_key(id) {
            return Err(FixedSolveError::MissingAmbiguity(id.clone()));
        }
    }

    let search = search_fixed_ambiguities(
        ambiguity_ids,
        &float_ambiguities,
        float_covariance_m,
        scale,
        ambiguity_satellites,
        float_only_systems,
        opts,
    )?;
    let fixed_m = fixed_ambiguities_m(&search.fixed_cycles, scale)?;
    let free_ambiguity_ids = free_ambiguity_ids(ambiguity_ids, &search.fixed_cycles);
    let mut state = FixedState {
        baseline_m: float_baseline_m,
        ambiguities_m: free_ambiguity_ids
            .iter()
            .map(|id| float_ambiguities[id.as_str()])
            .collect(),
    };
    let ambiguities = FixedAmbiguities {
        free_ids: &free_ambiguity_ids,
        fixed_m: &fixed_m,
    };
    let mut scratch = FloatSolveScratch::default();
    let mut dx: Vec<f64> = Vec::new();

    for iter in 1..=opts.max_iterations.max(1) {
        build_fixed_rows(ctx, epochs, ambiguities, &state, &mut scratch)?;
        float_normal_equations(
            ParameterLayout::rtk(free_ambiguity_ids.len()).dim(),
            &mut scratch,
        )
        .ok_or(FixedSolveError::SingularGeometry)?;
        solve_dd_normal_into(normal, solver, &mut scratch, &mut dx)
            .ok_or(FixedSolveError::SingularGeometry)?;

        for (k, value) in state.baseline_m.iter_mut().enumerate() {
            *value += dx[k];
        }
        for (idx, value) in state.ambiguities_m.iter_mut().enumerate() {
            *value += dx[3 + idx];
        }

        let baseline_step = vec3::norm3([dx[0], dx[1], dx[2]]);
        let ambiguity_step = dx[3..].iter().fold(0.0_f64, |m, &v| m.max(v.abs()));

        if baseline_step <= opts.position_tol_m && ambiguity_step <= opts.ambiguity_tol_m {
            return finalize_fixed_baseline(
                ctx,
                epochs,
                ambiguities,
                search,
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
            return finalize_fixed_baseline(
                ctx,
                epochs,
                ambiguities,
                search,
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

    Err(FixedSolveError::SingularGeometry)
}

/// Solve a static fixed RTK baseline with optional residual-validation/FDE.
///
/// This is the core-owned orchestration used by the public Sidereon fixed-baseline
/// API: run the float solve, reject the worst satellite residual while the gate
/// allows it, then run the integer fixed solve against the accepted float state.
/// Static fixed RTK baseline with residual validation/FDE.
///
/// Thin compatibility wrapper over the runtime strategy selector
/// ([`crate::estimation::strategies::estimate`]): it drives the shared
/// per-technique implementation [`run_fixed_validated`] under the RTK reference
/// strategy. The reference strategy always yields an RTK validated-fixed solution
/// or an RTK validated-fixed error, so the result is bit-identical to the recipe
/// driving [`run_fixed_validated`] directly.
pub fn solve_fixed_baseline_validated(
    epochs: &[Epoch],
    base: [f64; 3],
    initial_ambiguities: AmbiguitySet,
    initial_baseline_m: [f64; 3],
    model: &MeasModel,
    opts: ValidatedFixedSolveOpts,
    receiver_antenna_corrections: Option<&ReceiverAntennaCorrections>,
) -> Result<ValidatedFixedBaselineSolution, ValidatedFixedSolveError> {
    use crate::estimation::recipe::StrategyId;
    use crate::estimation::strategies::{
        estimate, EstimateError, EstimateInput, EstimateOptions, EstimateOutput,
    };
    match estimate(
        EstimateInput::RtkFixed {
            epochs,
            base,
            initial_ambiguities,
            initial_baseline_m,
            model,
            opts,
            receiver_antenna_corrections,
        },
        EstimateOptions::new(StrategyId::rtk_reference()),
    ) {
        Ok(EstimateOutput::RtkFixed(solution)) => Ok(*solution),
        Err(EstimateError::RtkFixed(error)) => Err(error),
        Ok(_) | Err(_) => unreachable!(
            "the RTK reference strategy yields an RTK validated-fixed solution or error"
        ),
    }
}

/// Drive the static validated-fixed RTK baseline from a resolved
/// [`EstimationRecipe`]: the shared per-technique implementation that
/// [`crate::estimation::strategies::estimate`] dispatches to. The recipe's
/// [`NormalRecipe`] flows into BOTH the float prerequisite ([`run_float`]) and
/// the integer-conditioned fixed re-solve ([`solve_fixed_baseline_with_normal`]),
/// so the whole validated-fixed path is driven by the resolved recipe rather than
/// a hard-coded assembler. For the RTK reference recipe every selected order
/// equals the legacy value, so this is bit-identical to it. `ctx` carries the
/// base/model/antenna measurement environment.
pub(crate) fn run_fixed_validated(
    recipe: &EstimationRecipe,
    ctx: MeasContext<'_>,
    epochs: &[Epoch],
    initial_ambiguities: AmbiguitySet,
    initial_baseline_m: [f64; 3],
    opts: ValidatedFixedSolveOpts,
) -> Result<ValidatedFixedBaselineSolution, ValidatedFixedSolveError> {
    let AmbiguitySet {
        ids: initial_ambiguity_ids,
        satellites: initial_ambiguity_satellites,
        scale,
        float_only_systems,
    } = initial_ambiguities;
    let ValidatedFixedSolveOpts {
        float: float_opts,
        fixed: fixed_opts,
        residual: residual_opts,
    } = opts;
    validate_residual_validation_opts(residual_opts)?;
    validate_float_solve_opts(float_opts).map_err(FixedSolveError::from)?;
    validate_fixed_solve_opts(fixed_opts)?;

    let mut working_epochs = epochs.to_vec();
    let mut exclusions = Vec::new();

    loop {
        let (ambiguity_ids, ambiguity_satellites) = if exclusions.is_empty() {
            (
                initial_ambiguity_ids.to_vec(),
                initial_ambiguity_satellites.clone(),
            )
        } else {
            baseline_ambiguity_index_core(&working_epochs)?
        };

        let row_count = baseline_row_count_core(&working_epochs);
        let unknown_count = 3 + ambiguity_ids.len();
        if row_count < unknown_count {
            return Err(ValidatedFixedSolveError::Underdetermined {
                row_count,
                unknown_count,
            });
        }

        let float_solution = run_float(
            recipe,
            ctx,
            &working_epochs,
            &ambiguity_ids,
            initial_baseline_m,
            float_opts,
        )
        .map_err(FixedSolveError::from)?;

        match residual_validation_outlier_core(&float_solution.residuals, residual_opts) {
            Some(outlier) if exclusions.len() < residual_opts.max_exclusions => {
                drop_baseline_satellite_core(&mut working_epochs, &outlier.satellite_id);
                exclusions.push(outlier);
            }
            Some(outlier) => {
                return Err(ValidatedFixedSolveError::ResidualValidationFailed {
                    outlier: Box::new(outlier),
                    exclusions,
                });
            }
            None => {
                // The integer-conditioned fixed re-solve consumes the resolved
                // recipe's normal-equation stage, not a hard-coded assembler.
                let fixed_solution = solve_fixed_baseline_with_normal(
                    ctx,
                    &working_epochs,
                    AmbiguitySet {
                        ids: &ambiguity_ids,
                        satellites: &ambiguity_satellites,
                        scale,
                        float_only_systems,
                    },
                    FloatPrior {
                        baseline_m: float_solution.baseline_m,
                        ambiguities_m: &float_solution.ambiguities_m,
                        covariance_m: &float_solution.ambiguity_covariance_m,
                    },
                    fixed_opts,
                    recipe.normal,
                    recipe.solver,
                )?;
                let residual_validation =
                    residual_validation_meta_core(residual_opts, exclusions.clone());

                return Ok(ValidatedFixedBaselineSolution {
                    float_solution,
                    fixed_solution,
                    residual_validation,
                    ambiguity_ids,
                    ambiguity_satellites,
                });
            }
        }
    }
}

fn validate_residual_validation_opts(
    opts: ResidualValidationOpts,
) -> Result<(), ValidatedFixedSolveError> {
    if let Some(threshold_sigma) = opts.threshold_sigma {
        validate::finite_positive(threshold_sigma, "rtk.residual_threshold_sigma").map_err(
            |error| {
                ValidatedFixedSolveError::Fixed(FixedSolveError::InvalidInput {
                    field: error.field(),
                    kind: super::RtkInputErrorKind::from(&error),
                })
            },
        )?;
    }
    Ok(())
}

fn baseline_row_count_core(epochs: &[Epoch]) -> usize {
    epochs.iter().map(|epoch| epoch.nonref.len()).sum::<usize>() * 2
}

pub(crate) fn baseline_ambiguity_index_core(
    epochs: &[Epoch],
) -> Result<(Vec<String>, BTreeMap<String, String>), ValidatedFixedSolveError> {
    let mut ambiguity_satellites = BTreeMap::<String, String>::new();

    for epoch in epochs {
        for sat in &epoch.nonref {
            let system = satellite_system(&sat.sat);
            let Some(reference) = epoch
                .references
                .iter()
                .find(|reference| satellite_system(&reference.sat) == system)
            else {
                return Err(ValidatedFixedSolveError::Fixed(FixedSolveError::Float(
                    FloatSolveError::MissingSystemReference(system.to_string()),
                )));
            };

            let mut ambiguity_id = AmbiguityId::default();
            assign_double_difference_ambiguity_id(
                &mut ambiguity_id,
                &sat.sat,
                &sat.sd_ambiguity_id,
                &reference.sat,
                &reference.sd_ambiguity_id,
            );

            match ambiguity_satellites.get(ambiguity_id.as_str()) {
                Some(existing) if existing == &sat.sat => {}
                Some(existing) => {
                    return Err(ValidatedFixedSolveError::DuplicateAmbiguityId {
                        ambiguity_id: ambiguity_id.into_string(),
                        first_satellite_id: existing.clone(),
                        second_satellite_id: sat.sat.clone(),
                    });
                }
                None => {
                    ambiguity_satellites.insert(ambiguity_id.into_string(), sat.sat.clone());
                }
            }
        }
    }

    let mut ordered = ambiguity_satellites
        .iter()
        .map(|(ambiguity_id, sat)| (ambiguity_id.clone(), sat.clone()))
        .collect::<Vec<_>>();
    ordered.sort_by(|(a_id, a_sat), (b_id, b_sat)| (a_sat, a_id).cmp(&(b_sat, b_id)));
    let ambiguity_ids = ordered
        .into_iter()
        .map(|(ambiguity_id, _sat)| ambiguity_id)
        .collect();

    Ok((ambiguity_ids, ambiguity_satellites))
}

fn drop_baseline_satellite_core(epochs: &mut [Epoch], satellite_id: &str) {
    for epoch in epochs {
        epoch.nonref.retain(|sat| sat.sat != satellite_id);
    }
}

fn residual_validation_outlier_core(
    residuals: &[FloatResidual],
    opts: ResidualValidationOpts,
) -> Option<ResidualValidationOutlier> {
    let threshold = opts.threshold_sigma?;
    let mut best: Option<(ResidualValidationOutlier, f64)> = None;

    for residual in residuals {
        for (kind, residual_m, sigma_m, normalized_residual) in [
            (
                ResidualComponentKind::Code,
                residual.code_m,
                residual.code_sigma_m,
                residual.code_normalized,
            ),
            (
                ResidualComponentKind::Phase,
                residual.phase_m,
                residual.phase_sigma_m,
                residual.phase_normalized,
            ),
        ] {
            let abs_normalized = normalized_residual.abs();
            if best
                .as_ref()
                .is_none_or(|(_outlier, best_abs)| abs_normalized > *best_abs)
            {
                best = Some((
                    ResidualValidationOutlier {
                        epoch_index: residual.epoch_index,
                        satellite_id: residual.satellite_id.clone(),
                        reference_satellite_id: residual.reference_satellite_id.clone(),
                        ambiguity_id: residual.ambiguity_id.clone(),
                        kind,
                        residual_m,
                        sigma_m,
                        normalized_residual,
                        threshold_sigma: threshold,
                    },
                    abs_normalized,
                ));
            }
        }
    }

    best.and_then(|(outlier, abs_normalized)| {
        if abs_normalized > threshold {
            Some(outlier)
        } else {
            None
        }
    })
}

fn residual_validation_meta_core(
    opts: ResidualValidationOpts,
    exclusions: Vec<ResidualValidationOutlier>,
) -> Option<ResidualValidationMeta> {
    let threshold_sigma = opts.threshold_sigma?;
    let excluded_sats = exclusions
        .iter()
        .map(|outlier| outlier.satellite_id.clone())
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    Some(ResidualValidationMeta {
        threshold_sigma,
        max_exclusions: opts.max_exclusions,
        excluded_sats,
        exclusions,
    })
}

fn finalize_fixed_baseline(
    ctx: MeasContext,
    epochs: &[Epoch],
    ambiguities: FixedAmbiguities,
    search: FixedSearchResult,
    outcome: SolveOutcome<FixedState>,
    mut scratch: FloatSolveScratch,
) -> Result<FixedBaselineSolution, FixedSolveError> {
    let SolveOutcome {
        state,
        iterations,
        converged,
        status,
    } = outcome;
    let FixedAmbiguities {
        free_ids: free_ambiguity_ids,
        fixed_m,
    } = ambiguities;
    build_fixed_rows(ctx, epochs, ambiguities, &state, &mut scratch)?;
    let residuals = float_residuals(&scratch.rows).map_err(FixedSolveError::from)?;
    let code_rms_m = rms(residuals.iter().map(|r| r.code_m));
    let phase_rms_m = rms(residuals.iter().map(|r| r.phase_m));
    let weighted_rms_m = rms(scratch.rows.iter().map(|r| r.y * r.weight));

    Ok(FixedBaselineSolution {
        baseline_m: state.baseline_m,
        free_ambiguities_m: free_ambiguity_ids
            .iter()
            .cloned()
            .zip(state.ambiguities_m)
            .map(|(id, value)| (id.into_string(), value))
            .collect(),
        fixed_ambiguities_cycles: search
            .fixed_cycles
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect(),
        fixed_ambiguities_m: fixed_m
            .iter()
            .map(|(k, &v)| (k.as_str().to_string(), v))
            .collect(),
        residuals,
        search: search.meta,
        iterations,
        converged,
        status,
        code_rms_m,
        phase_rms_m,
        weighted_rms_m,
        n_observations: scratch.rows.len(),
    })
}

/// Map a shared-builder error onto the fixed solve's error surface.
fn fixed_row_error(error: DdRowError) -> FixedSolveError {
    match error {
        DdRowError::MissingReference(system) => {
            FixedSolveError::Float(FloatSolveError::MissingSystemReference(system))
        }
        DdRowError::MissingAmbiguity(id) => FixedSolveError::MissingAmbiguity(id),
        DdRowError::ReceiverAntenna(error) => FixedSolveError::ReceiverAntenna(error),
        DdRowError::InvalidInput { field, kind } => FixedSolveError::InvalidInput { field, kind },
    }
}

fn fixed_input_error(error: validate::FieldError) -> FixedSolveError {
    FixedSolveError::InvalidInput {
        field: error.field(),
        kind: super::RtkInputErrorKind::from(&error),
    }
}

fn invalid_fixed_option(field: &'static str, kind: super::RtkInputErrorKind) -> FixedSolveError {
    FixedSolveError::InvalidInput { field, kind }
}

fn validate_fixed_solve_opts(opts: FixedSolveOpts) -> Result<(), FixedSolveError> {
    validate::finite_positive(opts.position_tol_m, "rtk.fixed.position_tol_m")
        .map_err(fixed_input_error)?;
    validate::finite_positive(opts.ambiguity_tol_m, "rtk.fixed.ambiguity_tol_m")
        .map_err(fixed_input_error)?;
    if opts.max_iterations == 0 {
        return Err(invalid_fixed_option(
            "rtk.fixed.max_iterations",
            super::RtkInputErrorKind::NotPositive,
        ));
    }
    validate::finite_positive(opts.ratio_threshold, "rtk.fixed.ratio_threshold")
        .map_err(fixed_input_error)?;
    if opts.partial_ambiguity_resolution && opts.partial_min_ambiguities == 0 {
        return Err(invalid_fixed_option(
            "rtk.fixed.partial_min_ambiguities",
            super::RtkInputErrorKind::NotPositive,
        ));
    }
    Ok(())
}

fn validate_float_prior_covariance(covariance_m: &[f64], n: usize) -> Result<(), FixedSolveError> {
    if n == 0 {
        return validate::validate_covariance_psd_rows(&[], "rtk.fixed.float_covariance_m")
            .map_err(fixed_input_error);
    }
    let rows = covariance_m.chunks_exact(n).collect::<Vec<_>>();
    validate::validate_covariance_psd_rows(&rows, "rtk.fixed.float_covariance_m")
        .map_err(fixed_input_error)
}

/// The [`DdRowRecipe::FixedBaseline`] for one fixed solve: held integers leave
/// their column out of the design (their metres value folds into the prefit) and
/// still-free ambiguities carry `+1` at their free-column index.
fn fixed_recipe<'a>(ambiguities: &FixedAmbiguities<'a>, state: &'a FixedState) -> DdRowRecipe<'a> {
    DdRowRecipe::FixedBaseline {
        free_ids: ambiguities.free_ids,
        fixed_m: ambiguities.fixed_m,
        ambiguities_m: &state.ambiguities_m,
    }
}

/// Owned snapshot of the fixed double-difference rows for one epoch, linearized
/// at `baseline_m`/`ambiguities_m` with `fixed_m` ambiguities held and `free_ids`
/// still estimated. Test-only wrapper over the shared
/// [`super::rows::dd_epoch_rows_into`] builder that drives the fixed row-level
/// golden trace; `FixedState`/`FixedAmbiguities` are private to this module, so
/// the construction lives here.
#[cfg(test)]
pub(super) fn fixed_epoch_rows(
    base: [f64; 3],
    epoch: &Epoch,
    free_ids: &[AmbiguityId],
    fixed_m: &BTreeMap<AmbiguityId, f64>,
    baseline_m: [f64; 3],
    ambiguities_m: Vec<f64>,
    model: &MeasModel,
) -> Result<Vec<super::rows::DdRow>, FixedSolveError> {
    let state = FixedState {
        baseline_m,
        ambiguities_m,
    };
    let ambiguities = FixedAmbiguities { free_ids, fixed_m };
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
        fixed_recipe(&ambiguities, &state),
        &mut scratch,
    )
    .map_err(fixed_row_error)?;
    Ok(rows.iter().map(super::rows::DdRow::from_scratch).collect())
}

fn build_fixed_rows(
    ctx: MeasContext,
    epochs: &[Epoch],
    ambiguities: FixedAmbiguities,
    state: &FixedState,
    scratch: &mut FloatSolveScratch,
) -> Result<(), FixedSolveError> {
    scratch.rows.clear();
    for (epoch_index, epoch) in epochs.iter().enumerate() {
        let rows = dd_epoch_rows_into(
            ctx,
            epoch,
            epoch_index,
            state.baseline_m,
            fixed_recipe(&ambiguities, state),
            &mut scratch.epoch_rows,
        )
        .map_err(fixed_row_error)?;
        scratch.rows.extend(rows.iter().cloned());
    }
    Ok(())
}

fn search_fixed_ambiguities(
    ambiguity_ids: &[String],
    float_ambiguities_m: &BTreeMap<String, f64>,
    covariance_m: &[f64],
    scale: AmbiguityScale,
    ambiguity_satellites: &BTreeMap<String, String>,
    float_only_systems: &[String],
    opts: FixedSolveOpts,
) -> Result<FixedSearchResult, FixedSolveError> {
    let float_only_ids =
        float_only_ambiguity_ids(ambiguity_satellites, &float_only_set(float_only_systems));
    let search_ids = ambiguity_ids
        .iter()
        .filter(|id| !float_only_ids.contains(id.as_str()))
        .cloned()
        .collect::<Vec<_>>();

    if search_ids.is_empty() {
        return Ok(FixedSearchResult {
            fixed_cycles: BTreeMap::new(),
            meta: empty_integer_search_meta(scale.offsets_m),
        });
    }

    let covariance_cycles =
        covariance_m_to_cycles(covariance_m, ambiguity_ids, scale.wavelengths_m)?;
    let search_covariance = if search_ids == ambiguity_ids {
        covariance_cycles.clone()
    } else {
        covariance_submatrix(ambiguity_ids, &covariance_cycles, &search_ids)
    };
    let float_cycles = float_cycles_for_ids(
        &search_ids,
        float_ambiguities_m,
        scale.wavelengths_m,
        scale.offsets_m,
    )?;
    let full = search_ambiguity_ids(&search_ids, &float_cycles, &search_covariance, opts)?;
    let mut full_meta = full.meta;
    full_meta.ambiguity_offsets_m = scale
        .offsets_m
        .iter()
        .map(|(k, &v)| (k.clone(), v))
        .collect();
    let full_fixed = full
        .fixed_cycles
        .iter()
        .map(|(k, &v)| (k.clone(), v))
        .collect::<BTreeMap<_, _>>();

    if full_meta.integer_status == IntegerStatus::Fixed || !opts.partial_ambiguity_resolution {
        full_meta.partial = partial_meta(false, false, &full_fixed, &[], None, None);
        return Ok(FixedSearchResult {
            fixed_cycles: full_fixed,
            meta: full_meta,
        });
    }

    search_partial_fixed_ambiguities(
        PartialSearchInputs {
            all_ids: &search_ids,
            float_cycles: &float_cycles,
            covariance_cycles: &search_covariance,
            offsets_m: scale.offsets_m,
            opts,
        },
        full_fixed,
        full_meta,
    )
}

fn free_ambiguity_ids(
    ambiguity_ids: &[String],
    fixed_cycles: &BTreeMap<String, i64>,
) -> Vec<AmbiguityId> {
    ambiguity_ids
        .iter()
        .filter(|id| !fixed_cycles.contains_key(*id))
        .map(|id| AmbiguityId::new(id.clone()))
        .collect()
}

fn fixed_ambiguities_m(
    fixed_cycles: &BTreeMap<String, i64>,
    scale: AmbiguityScale,
) -> Result<BTreeMap<AmbiguityId, f64>, FixedSolveError> {
    fixed_cycles
        .iter()
        .map(|(id, &cycles)| {
            let offset = *scale
                .offsets_m
                .get(id)
                .ok_or_else(|| FixedSolveError::MissingOffset(id.clone()))?;
            let wavelength = *scale
                .wavelengths_m
                .get(id)
                .ok_or_else(|| FixedSolveError::MissingWavelength(id.clone()))?;
            Ok((
                AmbiguityId::new(id.clone()),
                offset + cycles as f64 * wavelength,
            ))
        })
        .collect()
}
