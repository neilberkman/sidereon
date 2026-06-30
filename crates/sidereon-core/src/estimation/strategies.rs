//! Runtime-selectable estimation strategies (Phase-2 P4, driving in 2b).
//!
//! P0-P3 named the operation-order recipes ([`super::recipe`]) and routed the
//! frame/range/normal/ambiguity/qc kernels of the three reference stacks through
//! the shared [`super::substrate`]. This module is the runtime selector that ties
//! those names together: [`estimate`] takes an [`EstimateInput`] plus an
//! [`EstimateOptions`] carrying a [`StrategyId`], resolves the strategy into its
//! [`EstimationRecipe`] and screen/ambiguity policy DATA, and DRIVES the shared
//! per-technique implementation with that recipe.
//!
//! [`estimate`] is the driver, not a facade: each branch passes `resolved.recipe`
//! into the technique's shared runner (`spp::run`, `rtk_filter::run_float` /
//! `run_fixed_validated`, `precise_positioning::run_float_epochs` /
//! `run_fixed_from_float`), which consumes the recipe to select its operation
//! order (the SPP trust-region [`SolverRecipe`] via `spp::solve_with_solver`, the
//! RTK/PPP normal-equation [`NormalRecipe`] via the shared
//! [`super::substrate::normal::NormalAssembler`]). The old public entry points
//! (`spp::solve_with_policy`, `rtk_filter::solve_float_baseline` /
//! `solve_fixed_baseline_validated`, `precise_positioning::solve_float_epochs` /
//! `solve_fixed_from_float`) are now thin compatibility wrappers that call
//! [`estimate`] under their reference strategy. For a reference recipe every
//! selected operation order equals the value the legacy path hard-coded, so the
//! result is bit-identical and every existing 0-ULP golden is unchanged.
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
pub struct EstimateOptions {
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
pub enum EstimateInput<'a> {
    /// SPP under the public validation/orchestration policy
    /// (`spp::solve_with_policy`).
    Spp {
        eph: &'a dyn EphemerisSource,
        inputs: &'a SolveInputs,
        with_geodetic: bool,
        policy: SolvePolicy,
    },
    /// Static multi-epoch float RTK baseline (`rtk_filter::solve_float_baseline`).
    RtkFloat {
        epochs: &'a [Epoch],
        base: [f64; 3],
        ambiguity_ids: &'a [String],
        initial_baseline_m: [f64; 3],
        model: &'a MeasModel,
        opts: FloatSolveOpts,
        receiver_antenna_corrections: Option<&'a ReceiverAntennaCorrections>,
    },
    /// Static fixed RTK baseline with residual validation/FDE
    /// (`rtk_filter::solve_fixed_baseline_validated`).
    RtkFixed {
        epochs: &'a [Epoch],
        base: [f64; 3],
        initial_ambiguities: AmbiguitySet<'a>,
        initial_baseline_m: [f64; 3],
        model: &'a MeasModel,
        opts: ValidatedFixedSolveOpts,
        receiver_antenna_corrections: Option<&'a ReceiverAntennaCorrections>,
    },
    /// Static multi-epoch float PPP arc
    /// (`precise_positioning::solve_float_epochs`).
    PppFloat {
        source: &'a dyn ObservableEphemerisSource,
        epochs: &'a [FloatEpoch],
        initial_state: FloatState,
        config: FloatSolveConfig,
    },
    /// Integer-fixed PPP from an existing float solution
    /// (`precise_positioning::solve_fixed_from_float`).
    PppFixed {
        source: &'a dyn ObservableEphemerisSource,
        epochs: &'a [FloatEpoch],
        float_solution: FloatSolution,
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
    Spp(Box<ReceiverSolution>),
    RtkFloat(Box<FloatBaselineSolution>),
    RtkFixed(Box<ValidatedFixedBaselineSolution>),
    PppFloat(Box<FloatSolution>),
    PppFixed(Box<FixedSolution>),
}

/// Failure of [`estimate`]: a selection error, or the wrapped error of the
/// dispatched reference entry point.
#[derive(Debug)]
pub enum EstimateError {
    /// The selected strategy's technique does not match the input's technique
    /// (e.g. an RTK strategy with an SPP input).
    TechniqueMismatch {
        strategy: Technique,
        input: Technique,
    },
    /// A `Reference` strategy named a `target` that is not a supported reference
    /// for its `technique` (e.g. an RTK technique against the Skyfield SPP
    /// oracle, or the owned deterministic solver for a non-SPP technique). The
    /// supported pairs are enumerated by [`EstimationRecipe::for_reference`].
    IncompatibleTarget {
        technique: Technique,
        target: ReferenceTarget,
    },
    /// A `Canonical` strategy was selected for a technique whose canonical model
    /// is not yet implemented. Canonical SPP, RTK, and PPP are all wired, so no
    /// technique currently produces this; it is retained as the resolver's stable
    /// not-yet-implemented surface for any future technique.
    CanonicalUnavailable {
        technique: Technique,
    },
    Spp(SolvePolicyError),
    RtkFloat(RtkFloatSolveError),
    RtkFixed(ValidatedFixedSolveError),
    PppFloat(PppFloatSolveError),
    PppFixed(FixedSolveError),
}

/// A [`StrategyId`] resolved into the selection DATA it runs under: the
/// operation-order [`EstimationRecipe`] (P0-P2) and the residual-screen families
/// (P3). The recipe is the current reference recipe for the technique, so a
/// resolved reference strategy dispatches bit-identically to the existing path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResolvedStrategy {
    pub id: StrategyId,
    pub technique: Technique,
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

/// The residual-screen families a technique applies (P3 `ScreenKind`). RTK lists
/// both the static fixed-baseline validation and the sequential-filter innovation
/// screen, the two members of its screen family.
const fn screens_for(technique: Technique) -> &'static [ScreenKind] {
    match technique {
        Technique::Spp => &[ScreenKind::RaimChiSquare],
        Technique::Rtk => &[
            ScreenKind::RtkFixedResidualValidation,
            ScreenKind::RtkSequentialInnovation,
        ],
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
            &[
                ScreenKind::RtkFixedResidualValidation,
                ScreenKind::RtkSequentialInnovation,
            ]
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
        // RTK static baseline to the inverse-sigma residual, the RTK sequential
        // filter to the inverse-variance innovation, PPP to the inverse-sigma
        // root. SPP's aggregate RAIM screen has no per-residual recipe.
        let rtk = ResolvedStrategy::resolve(StrategyId::rtk_reference()).unwrap();
        assert_eq!(
            rtk.screens
                .iter()
                .map(|screen| screen.residual_norm())
                .collect::<Vec<_>>(),
            vec![
                Some(ResidualNormRecipe::RtkInverseSigmaResidual),
                Some(ResidualNormRecipe::RtkInverseVarianceInnovation),
            ]
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
