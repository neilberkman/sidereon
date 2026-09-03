//! Runtime-selectable estimation strategies (Phase-2 P4, driving in 2b).
//!
//! P0-P3 named the operation-order recipes ([`super::recipe`]) and routed the
//! frame/range/normal/ambiguity/qc kernels of the three reference stacks through
//! the shared `super::substrate`. This module is the runtime selector that ties
//! those names together: [`estimate`] takes an [`EstimateInput`] plus an
//! [`EstimateOptions`] carrying a [`StrategyId`], resolves the strategy into its
//! [`EstimationRecipe`] and screen/ambiguity policy DATA, and DRIVES the shared
//! per-technique implementation with that recipe.
//!
//! [`estimate`] is the driver, not a facade: each branch passes `resolved.recipe`
//! into the technique's shared runner (`spp::run`, `rtk_filter::run_float` /
//! `run_fixed_validated`, `precise_positioning::run_float_epochs` /
//! `run_fixed_from_float`), which consumes the recipe to select its operation
//! order (the SPP trust-region [`crate::estimation::recipe::SolverRecipe`] via
//! `spp::solve_with_solver`, the RTK/PPP normal-equation
//! [`crate::estimation::recipe::NormalRecipe`] via the shared
//! `super::substrate::normal::NormalAssembler`). The old public entry points
//! (`spp::solve_with_policy`, `rtk_filter::solve_float_baseline` /
//! `solve_fixed_baseline_validated`, `precise_positioning::solve_float_epochs` /
//! `solve_fixed_from_float`) are now thin compatibility wrappers that call
//! [`estimate`] under their reference strategy. For a reference recipe every
//! selected operation order equals the value the legacy path hard-coded, so
//! results are bit-identical and existing 0-ULP goldens are unchanged, with one
//! exception: the static PPP reference path now eliminates per-epoch receiver
//! clocks from the normal equations (pinned equivalent to the unreduced dense
//! solve and its inverse in `precise_positioning::normal` tests), so PPP
//! goldens were re-frozen at the reduced path's bits.
//!
//! `Canonical` strategies (the bounded-tolerance "best" model) are the P6
//! additive strategy, and all three techniques are now wired. Resolving
//! [`StrategyId::Canonical`] with [`Technique::Spp`] drives `spp::run` under the
//! [`EstimationRecipe::canonical_spp`] recipe (the IERS-rigorous light-time /
//! WGS84-geodetic op-order on the owned deterministic solver); with
//! [`Technique::Rtk`] drives the RTK runners under
//! [`EstimationRecipe::canonical_rtk`] (the owned Cholesky square-root-information
//! solve); and with [`Technique::Ppp`] drives the PPP runners under
//! [`EstimationRecipe::canonical_ppp`] (the same owned Cholesky
//! square-root-information solve on the dense weighted PPP normal system).
//! [`EstimateError::CanonicalUnavailable`] is retained as the resolver's
//! not-yet-implemented surface but no technique currently produces it.

use super::recipe::{
    AmbiguityIdPolicy, EstimationRecipe, ReferenceTarget, ScreenKind, StrategyId, Technique,
};
use crate::observables::ObservableEphemerisSource;
use crate::precise_positioning::{
    FixedSolution, FixedSolveConfig, FixedSolveError, FloatEpoch, FloatSolution, FloatSolveConfig,
    FloatSolveError as PppFloatSolveError, FloatState,
};
use crate::rtk_filter::{
    AmbiguitySet, Epoch, FloatBaselineSolution, FloatSolveError as RtkFloatSolveError,
    FloatSolveOpts, MeasModel, ReceiverAntennaCorrections, ValidatedFixedBaselineSolution,
    ValidatedFixedSolveError, ValidatedFixedSolveOpts,
};
use crate::spp::{EphemerisSource, ReceiverSolution, SolveInputs, SolvePolicy, SolvePolicyError};

/// Runtime selection options for [`estimate`]. Defaults to the SPP reference
/// strategy ([`StrategyId::default`]), matching the per-stage recipe defaults.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct EstimateOptions {
    /// Strategy identity resolved by [`estimate`] before it checks the input
    /// technique. The default is the SPP/Skyfield reference strategy.
    pub strategy: StrategyId,
}

impl EstimateOptions {
    /// Options selecting `strategy`.
    pub const fn new(strategy: StrategyId) -> Self {
        Self { strategy }
    }
}

/// The unified input to [`estimate`], one variant per technique entry. Each
/// variant carries exactly the arguments the shared per-technique runner needs;
/// [`estimate`] drives that runner with the resolved recipe. RTK and PPP expose a
/// float and a fixed entry; both map to the same [`Technique`].
#[allow(clippy::large_enum_variant)]
pub enum EstimateInput<'a> {
    /// SPP under the public validation/orchestration policy
    /// (`spp::solve_with_policy`).
    Spp {
        /// Ephemeris source queried for selected satellites' transmit-time
        /// positions and clocks.
        eph: &'a dyn EphemerisSource,
        /// SPP epoch bundle whose observations, receive-time arguments, initial
        /// state, and corrections are validated before the solve.
        inputs: &'a SolveInputs,
        /// Whether the returned receiver solution includes its geodetic field;
        /// `false` leaves that field as `None`.
        with_geodetic: bool,
        /// SPP orchestration policy: coarse seeded candidates when configured,
        /// or one solve followed by validation otherwise.
        policy: SolvePolicy,
    },
    /// Static multi-epoch float RTK baseline (`rtk_filter::solve_float_baseline`).
    RtkFloat {
        /// Normalized static epochs whose double-difference code/phase rows are
        /// accumulated into the batch normal system and residuals.
        epochs: &'a [Epoch],
        /// Base-station ECEF position used by RTK geometry and weighting rows.
        base: [f64; 3],
        /// Ordered ambiguity column keys paired with the float estimates and
        /// covariance in the returned solution.
        ambiguity_ids: &'a [String],
        /// Initial ECEF baseline estimate in meters for the iterated float state.
        initial_baseline_m: [f64; 3],
        /// Code/phase sigmas, Sagnac choice, and stochastic model read while
        /// constructing RTK rows.
        model: &'a MeasModel,
        /// Position/ambiguity tolerances and maximum iteration count for the
        /// float loop.
        opts: FloatSolveOpts,
        /// Optional receiver antenna calibration applied while building RTK
        /// rows; `None` supplies no antenna correction.
        receiver_antenna_corrections: Option<&'a ReceiverAntennaCorrections>,
    },
    /// Static fixed RTK baseline with residual validation/FDE
    /// (`rtk_filter::solve_fixed_baseline_validated`).
    RtkFixed {
        /// Working static epoch set used by the float prerequisite, residual
        /// validation exclusions, and integer-conditioned fixed re-solve.
        epochs: &'a [Epoch],
        /// Base-station ECEF position shared by the validated float and fixed
        /// RTK stages.
        base: [f64; 3],
        /// Initial ambiguity ids, satellite map, cycle/meter scale, and
        /// float-only systems for the fixed solve.
        initial_ambiguities: AmbiguitySet<'a>,
        /// Initial ECEF baseline estimate used to seed the float prerequisite.
        initial_baseline_m: [f64; 3],
        /// RTK measurement sigmas, Sagnac choice, and stochastic model shared by
        /// both validated stages.
        model: &'a MeasModel,
        /// Float, fixed, and residual-validation controls for the full fixed
        /// workflow.
        opts: ValidatedFixedSolveOpts,
        /// Optional receiver antenna calibration forwarded to both RTK stages;
        /// `None` supplies no antenna correction.
        receiver_antenna_corrections: Option<&'a ReceiverAntennaCorrections>,
    },
    /// Static multi-epoch float PPP arc
    /// (`precise_positioning::solve_float_epochs`).
    PppFloat {
        /// Observable ephemeris source used to predict satellite state while
        /// PPP rows are built.
        source: &'a dyn ObservableEphemerisSource,
        /// Static PPP arc; configured elevation filtering occurs before active
        /// ambiguity ids and normal rows are built.
        epochs: &'a [FloatEpoch],
        /// Initial receiver position, clocks, ambiguities, and enabled
        /// atmospheric or residual-ionosphere states.
        initial_state: FloatState,
        /// PPP weights, corrections, atmospheric settings, iteration controls,
        /// and optional screening/state-estimation flags.
        config: FloatSolveConfig,
    },
    /// Integer-fixed PPP from an existing float solution
    /// (`precise_positioning::solve_fixed_from_float`).
    PppFixed {
        /// Observable ephemeris source used for cutoff prediction, ambiguity
        /// search rows, and the fixed re-solve.
        source: &'a dyn ObservableEphemerisSource,
        /// Static arc used for integer search and the fixed re-solve, after any
        /// configured elevation filtering.
        epochs: &'a [FloatEpoch],
        /// Validated float result converted into the initial fixed state and
        /// used with its ambiguity covariance for integer search.
        float_solution: FloatSolution,
        /// PPP measurement/correction settings, integer-search controls,
        /// convergence controls, and optional filtering/state estimation.
        config: FixedSolveConfig,
    },
}

impl EstimateInput<'_> {
    /// The estimation technique this input runs.
    pub fn technique(&self) -> Technique {
        match self {
            Self::Spp { .. } => Technique::Spp,
            Self::RtkFloat { .. } | Self::RtkFixed { .. } => Technique::Rtk,
            Self::PppFloat { .. } | Self::PppFixed { .. } => Technique::Ppp,
        }
    }
}

/// The unified result of [`estimate`], wrapping each reference entry point's
/// existing return type unchanged. The payloads are heterogeneously sized
/// (RTK/PPP solutions are large), so each is boxed to keep the enum
/// pointer-sized regardless of which technique ran.
#[derive(Debug, Clone)]
pub enum EstimateOutput {
    /// Successful SPP result produced by `spp::run` and boxed for the unified
    /// output enum; the SPP compatibility wrapper unwraps this variant.
    Spp(Box<ReceiverSolution>),
    /// Successful static RTK float result produced by `rtk_filter::run_float`.
    RtkFloat(Box<FloatBaselineSolution>),
    /// Successful validated fixed RTK result produced by
    /// `rtk_filter::run_fixed_validated`.
    RtkFixed(Box<ValidatedFixedBaselineSolution>),
    /// Successful static PPP float result produced by
    /// `precise_positioning::run_float_epochs`.
    PppFloat(Box<FloatSolution>),
    /// Successful integer-fixed PPP result produced by
    /// `precise_positioning::run_fixed_from_float`.
    PppFixed(Box<FixedSolution>),
}

/// Failure of [`estimate`]: a selection error, or the wrapped error of the
/// dispatched reference entry point.
#[derive(Debug)]
pub enum EstimateError {
    /// The selected strategy's technique does not match the input's technique
    /// (e.g. an RTK strategy with an SPP input).
    TechniqueMismatch {
        /// Technique selected by the resolved strategy.
        strategy: Technique,
        /// Technique returned by [`EstimateInput::technique`] for the input.
        input: Technique,
    },
    /// A `Reference` strategy named a `target` that is not a supported reference
    /// for its `technique` (e.g. an RTK technique against the Skyfield SPP
    /// oracle, or the owned deterministic solver for a non-SPP technique). The
    /// supported pairs are enumerated by [`EstimationRecipe::for_reference`].
    IncompatibleTarget {
        /// Technique passed to [`EstimationRecipe::for_reference`].
        technique: Technique,
        /// Reference target passed to [`EstimationRecipe::for_reference`].
        target: ReferenceTarget,
    },
    /// A `Canonical` strategy was selected for a technique whose canonical model
    /// is not yet implemented. Canonical SPP, RTK, and PPP are all wired, so no
    /// technique currently produces this; it is retained as the resolver's stable
    /// not-yet-implemented surface for any future technique.
    CanonicalUnavailable {
        /// Technique for which [`EstimationRecipe::for_canonical`] returned no
        /// recipe.
        technique: Technique,
    },
    /// Error returned by the SPP runner and preserved for the compatibility
    /// wrapper.
    Spp(SolvePolicyError),
    /// Error returned by the static RTK float runner and preserved for the
    /// compatibility wrapper.
    RtkFloat(RtkFloatSolveError),
    /// Error returned by the validated RTK fixed runner and preserved for the
    /// compatibility wrapper.
    RtkFixed(ValidatedFixedSolveError),
    /// Error returned by the static PPP float runner and preserved for the
    /// compatibility wrapper.
    PppFloat(PppFloatSolveError),
    /// Error returned by the integer-fixed PPP runner and preserved for the
    /// compatibility wrapper.
    PppFixed(FixedSolveError),
}

/// A [`StrategyId`] resolved into the selection DATA it runs under: the
/// operation-order [`EstimationRecipe`] (P0-P2) and the residual-screen families
/// (P3). The recipe is the current reference recipe for the technique, so a
/// resolved reference strategy dispatches bit-identically to the existing path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedStrategy {
    /// Requested strategy identity copied into the resolved record.
    pub id: StrategyId,
    /// Technique selected by `id`, checked against the input before dispatch.
    pub technique: Technique,
    /// Operation-order recipe selected by `id` and passed to the technique
    /// runner.
    pub recipe: EstimationRecipe,
    /// The residual-screen families this technique applies (P3 `ScreenKind`).
    pub screens: &'static [ScreenKind],
}

impl ResolvedStrategy {
    /// Resolve a runtime [`StrategyId`] into its recipe and screen policy.
    /// `Reference` strategies resolve to the recipe for their `(technique,
    /// target)` pair, rejecting an unsupported pair with
    /// [`EstimateError::IncompatibleTarget`]; `Canonical` strategies resolve to
    /// their canonical recipe ([`EstimationRecipe::for_canonical`]), rejecting a
    /// technique whose canonical model is not yet implemented with
    /// [`EstimateError::CanonicalUnavailable`].
    pub fn resolve(id: StrategyId) -> Result<Self, EstimateError> {
        match id {
            StrategyId::Reference { technique, target } => {
                let recipe = EstimationRecipe::for_reference(technique, target)
                    .ok_or(EstimateError::IncompatibleTarget { technique, target })?;
                Ok(Self {
                    id,
                    technique,
                    recipe,
                    screens: screens_for(technique),
                })
            }
            StrategyId::Canonical { technique } => {
                let recipe = EstimationRecipe::for_canonical(technique)
                    .ok_or(EstimateError::CanonicalUnavailable { technique })?;
                Ok(Self {
                    id,
                    technique,
                    recipe,
                    screens: screens_for(technique),
                })
            }
        }
    }

    /// The integer-ambiguity identity policy (P3) this strategy resolves under,
    /// parameterized by the runtime ratio threshold and (RTK only) partial-set
    /// floor. `None` for SPP, which carries no integer ambiguities.
    pub fn ambiguity_id_policy(
        &self,
        ratio_threshold: f64,
        partial_min_ambiguities: usize,
    ) -> Option<AmbiguityIdPolicy> {
        match self.technique {
            Technique::Spp => None,
            Technique::Rtk => Some(AmbiguityIdPolicy::rtk_static(
                ratio_threshold,
                partial_min_ambiguities,
            )),
            Technique::Ppp => Some(AmbiguityIdPolicy::ppp(ratio_threshold)),
        }
    }
}

/// The residual-screen families a technique applies (P3 `ScreenKind`).
const fn screens_for(technique: Technique) -> &'static [ScreenKind] {
    match technique {
        Technique::Spp => &[ScreenKind::RaimChiSquare],
        Technique::Rtk => &[ScreenKind::RtkFixedResidualValidation],
        Technique::Ppp => &[ScreenKind::PppFloatLeaveOneOut],
    }
}

/// Run estimation under a runtime-selected [`StrategyId`].
///
/// Resolves `options.strategy` into its recipe/screen policy, checks that the
/// strategy's technique matches `input`, then drives the technique's shared
/// runner with `resolved.recipe`. The runner consumes the recipe to select its
/// operation order; for a reference recipe every selected order equals the value
/// the legacy path hard-coded, so the result is bit-identical and every existing
/// 0-ULP golden is preserved.
pub fn estimate(
    input: EstimateInput<'_>,
    options: EstimateOptions,
) -> Result<EstimateOutput, EstimateError> {
    let resolved = ResolvedStrategy::resolve(options.strategy)?;
    let input_technique = input.technique();
    if resolved.technique != input_technique {
        return Err(EstimateError::TechniqueMismatch {
            strategy: resolved.technique,
            input: input_technique,
        });
    }

    match input {
        EstimateInput::Spp {
            eph,
            inputs,
            with_geodetic,
            policy,
        } => crate::spp::run(&resolved.recipe, eph, inputs, with_geodetic, policy)
            .map(|s| EstimateOutput::Spp(Box::new(s)))
            .map_err(EstimateError::Spp),
        EstimateInput::RtkFloat {
            epochs,
            base,
            ambiguity_ids,
            initial_baseline_m,
            model,
            opts,
            receiver_antenna_corrections,
        } => crate::rtk_filter::run_float(
            &resolved.recipe,
            crate::rtk_filter::MeasContext::new(base, model, receiver_antenna_corrections),
            epochs,
            ambiguity_ids,
            initial_baseline_m,
            opts,
        )
        .map(|s| EstimateOutput::RtkFloat(Box::new(s)))
        .map_err(EstimateError::RtkFloat),
        EstimateInput::RtkFixed {
            epochs,
            base,
            initial_ambiguities,
            initial_baseline_m,
            model,
            opts,
            receiver_antenna_corrections,
        } => crate::rtk_filter::run_fixed_validated(
            &resolved.recipe,
            crate::rtk_filter::MeasContext::new(base, model, receiver_antenna_corrections),
            epochs,
            initial_ambiguities,
            initial_baseline_m,
            opts,
        )
        .map(|s| EstimateOutput::RtkFixed(Box::new(s)))
        .map_err(EstimateError::RtkFixed),
        EstimateInput::PppFloat {
            source,
            epochs,
            initial_state,
            config,
        } => crate::precise_positioning::run_float_epochs(
            &resolved.recipe,
            source,
            epochs,
            initial_state,
            config,
        )
        .map(|s| EstimateOutput::PppFloat(Box::new(s)))
        .map_err(EstimateError::PppFloat),
        EstimateInput::PppFixed {
            source,
            epochs,
            float_solution,
            config,
        } => crate::precise_positioning::run_fixed_from_float(
            &resolved.recipe,
            source,
            epochs,
            float_solution,
            config,
        )
        .map(|s| EstimateOutput::PppFixed(Box::new(s)))
        .map_err(EstimateError::PppFixed),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::estimation::recipe::{ReferenceTarget, ResidualNormRecipe};

    #[test]
    fn input_technique_matches_each_variant() {
        // Compile-time-ish guard that the float/fixed entries share a technique.
        assert_eq!(
            screens_for(Technique::Rtk),
            &[ScreenKind::RtkFixedResidualValidation]
        );
        assert_eq!(screens_for(Technique::Spp), &[ScreenKind::RaimChiSquare]);
        assert_eq!(
            screens_for(Technique::Ppp),
            &[ScreenKind::PppFloatLeaveOneOut]
        );
    }

    #[test]
    fn resolve_reference_strategies_to_their_recipe_and_screens() {
        let spp = ResolvedStrategy::resolve(StrategyId::spp_reference()).unwrap();
        assert_eq!(spp.technique, Technique::Spp);
        assert_eq!(spp.recipe, EstimationRecipe::spp());
        assert_eq!(spp.screens, &[ScreenKind::RaimChiSquare]);
        assert!(spp.ambiguity_id_policy(3.0, 1).is_none());

        let rtk = ResolvedStrategy::resolve(StrategyId::rtk_reference()).unwrap();
        assert_eq!(rtk.technique, Technique::Rtk);
        assert_eq!(rtk.recipe, EstimationRecipe::rtk());
        let rtk_policy = rtk.ambiguity_id_policy(3.0, 4).unwrap();
        assert_eq!(rtk_policy, AmbiguityIdPolicy::rtk_static(3.0, 4));

        let ppp = ResolvedStrategy::resolve(StrategyId::ppp_reference()).unwrap();
        assert_eq!(ppp.technique, Technique::Ppp);
        assert_eq!(ppp.recipe, EstimationRecipe::ppp());
        assert_eq!(ppp.screens, &[ScreenKind::PppFloatLeaveOneOut]);
        let ppp_policy = ppp.ambiguity_id_policy(2.5, 0).unwrap();
        assert_eq!(ppp_policy, AmbiguityIdPolicy::ppp(2.5));
    }

    #[test]
    fn each_resolved_strategy_screen_uses_its_own_residual_norm() {
        // Each resolved screen maps to its committed normalization recipe: the
        // RTK static baseline to the inverse-sigma residual and PPP to the
        // inverse-sigma root. SPP's aggregate RAIM screen has no per-residual
        // recipe.
        let rtk = ResolvedStrategy::resolve(StrategyId::rtk_reference()).unwrap();
        assert_eq!(
            rtk.screens
                .iter()
                .map(|screen| screen.residual_norm())
                .collect::<Vec<_>>(),
            vec![Some(ResidualNormRecipe::RtkInverseSigmaResidual)]
        );
        let ppp = ResolvedStrategy::resolve(StrategyId::ppp_reference()).unwrap();
        assert_eq!(
            ppp.screens[0].residual_norm(),
            Some(ResidualNormRecipe::PppInverseSigmaMagnitude)
        );
        let spp = ResolvedStrategy::resolve(StrategyId::spp_reference()).unwrap();
        assert_eq!(spp.screens[0].residual_norm(), None);
    }

    #[test]
    fn resolve_owned_deterministic_spp_selects_the_owned_solver() {
        use crate::estimation::recipe::SolverRecipe;

        let owned = ResolvedStrategy::resolve(StrategyId::spp_owned_deterministic()).unwrap();
        assert_eq!(owned.technique, Technique::Spp);
        assert_eq!(owned.recipe.solver, SolverRecipe::OwnedDeterministicTrf);
        assert_eq!(owned.recipe, EstimationRecipe::spp_owned_deterministic());
        // Same SPP screen policy as the Skyfield reference strategy.
        assert_eq!(owned.screens, &[ScreenKind::RaimChiSquare]);
    }

    #[test]
    fn resolve_rejects_incompatible_technique_target_pairs() {
        for (technique, target) in [
            (Technique::Spp, ReferenceTarget::Rtklib),
            (Technique::Spp, ReferenceTarget::Scipy),
            (Technique::Rtk, ReferenceTarget::OwnedDeterministic),
            (Technique::Ppp, ReferenceTarget::Skyfield),
        ] {
            let err =
                ResolvedStrategy::resolve(StrategyId::Reference { technique, target }).unwrap_err();
            match err {
                EstimateError::IncompatibleTarget {
                    technique: t,
                    target: g,
                } => {
                    assert_eq!(t, technique);
                    assert_eq!(g, target);
                }
                other => {
                    panic!("{technique:?} + {target:?} should be IncompatibleTarget, got {other:?}")
                }
            }
        }
    }

    #[test]
    fn canonical_spp_resolves_to_the_canonical_recipe() {
        let resolved = ResolvedStrategy::resolve(StrategyId::Canonical {
            technique: Technique::Spp,
        })
        .expect("canonical SPP resolves");
        assert_eq!(resolved.technique, Technique::Spp);
        assert_eq!(resolved.recipe, EstimationRecipe::canonical_spp());
        // Canonical SPP carries the SPP screen policy (no integer ambiguities).
        assert_eq!(resolved.screens, &[ScreenKind::RaimChiSquare]);
        assert!(resolved.ambiguity_id_policy(3.0, 1).is_none());
    }

    #[test]
    fn canonical_rtk_resolves_to_the_canonical_recipe() {
        let resolved = ResolvedStrategy::resolve(StrategyId::Canonical {
            technique: Technique::Rtk,
        })
        .expect("canonical RTK resolves");
        assert_eq!(resolved.technique, Technique::Rtk);
        assert_eq!(resolved.recipe, EstimationRecipe::canonical_rtk());
        // The owned Cholesky square-root information solve, not the reference
        // first-tie Gaussian elimination.
        assert_eq!(
            resolved.recipe.normal,
            crate::estimation::recipe::NormalRecipe::CanonicalSquareRoot
        );
        assert_eq!(
            resolved.recipe.solver,
            crate::estimation::recipe::SolverRecipe::OwnedDeterministicCholesky
        );
    }

    #[test]
    fn canonical_ppp_resolves_to_the_canonical_recipe() {
        let resolved = ResolvedStrategy::resolve(StrategyId::Canonical {
            technique: Technique::Ppp,
        })
        .expect("canonical PPP resolves");
        assert_eq!(resolved.technique, Technique::Ppp);
        assert_eq!(resolved.recipe, EstimationRecipe::canonical_ppp());
        // The owned Cholesky square-root information solve on the dense PPP normal
        // system, not the reference dense last-tie Gaussian elimination.
        assert_eq!(
            resolved.recipe.normal,
            crate::estimation::recipe::NormalRecipe::CanonicalSquareRoot
        );
        assert_eq!(
            resolved.recipe.solver,
            crate::estimation::recipe::SolverRecipe::OwnedDeterministicCholesky
        );
        // Canonical PPP carries the PPP screen policy.
        assert_eq!(resolved.screens, &[ScreenKind::PppFloatLeaveOneOut]);
        let policy = resolved.ambiguity_id_policy(2.5, 0).unwrap();
        assert_eq!(policy, AmbiguityIdPolicy::ppp(2.5));
    }

    #[test]
    fn default_options_select_spp_reference() {
        let resolved = ResolvedStrategy::resolve(EstimateOptions::default().strategy).unwrap();
        assert_eq!(
            resolved.id,
            StrategyId::Reference {
                technique: Technique::Spp,
                target: ReferenceTarget::Skyfield,
            }
        );
        assert_eq!(resolved.recipe, EstimationRecipe::spp());
    }
}
