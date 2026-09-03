//! Named operation-order recipes for the GNSS estimation substrate.
//!
//! Phase-2 collapses the three thick estimator stacks (`spp`, `rtk`/`rtk_filter`,
//! `precise_positioning`) onto one shared substrate plus thin, runtime-selectable
//! strategies. The single hard constraint is that each external reference's
//! bit-exactness (Skyfield for SPP, RTKLIB for RTK, the PPP oracle for PPP) must
//! be preserved to 0 ULP. Different references need different floating-point
//! operation orders for the *same* physical quantity, so the substrate never
//! "simplifies" a parity-sensitive formula into one shared form. Instead every
//! such choice is a NAMED variant: a strategy selects the op-order it needs by
//! enum value rather than by owning a copy of the helper.
//!
//! This module *names* the recipes; the substrate and strategies route every
//! caller through them. Each reference-faithful strategy resolves to the single
//! op-order it was already using, so threading the recipe through the shared
//! spine reproduces the prior code path bit-for-bit and leaves every existing
//! 0-ULP golden unchanged.
//!
//! The `Canonical*` variants belong to the single consistent IERS-rigorous
//! model (the bounded-tolerance canonical strategy, P6). They are NOT used by
//! any reference-faithful strategy; canonical is an additional selectable
//! strategy that changes nothing about the references. The SPP canonical range
//! and frame variants ([`RangeRecipe::CanonicalLightTimeClosedFormSagnac`],
//! [`FrameRecipe::CanonicalWgs84`]) are implemented and driven by
//! [`EstimationRecipe::canonical_spp`]; the RTK and PPP canonical square-root
//! solve ([`NormalRecipe::CanonicalSquareRoot`] on
//! [`SolverRecipe::OwnedDeterministicCholesky`]) by
//! [`EstimationRecipe::canonical_rtk`] and [`EstimationRecipe::canonical_ppp`].
//! Canonical SPP, RTK, and PPP are all wired.

/// Estimation technique: which physical observation model and parameter set a
/// strategy estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Technique {
    /// Single-point positioning: undifferenced pseudorange PVT.
    #[default]
    Spp,
    /// Real-time kinematic: double-differenced code/phase baseline.
    Rtk,
    /// Precise point positioning: undifferenced ionosphere-free code/phase.
    Ppp,
}

/// The reference a reference-faithful strategy is bit-exact against. The
/// external oracles (Skyfield, RTKLIB, the PPP oracle) are CI validation targets
/// whose goldens stay 0-ULP unchanged through P0-P5; [`Self::OwnedDeterministic`]
/// is instead pinned to the owned solver's own frozen-bits golden (P5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ReferenceTarget {
    /// Skyfield (the SPP geometry/clock/Sagnac reference).
    #[default]
    Skyfield,
    /// RTKLIB (the RTK double-difference baseline reference).
    Rtklib,
    /// scipy least-squares host solve (the SPP solver-agreement reference).
    /// Named for the `trust-region-least-squares` host-LAPACK fingerprint study;
    /// not a runtime
    /// estimation strategy (it is not wired into the SPP solve path), so it is
    /// not a valid [`StrategyId`] target.
    Scipy,
    /// The PPP float/fixed oracle arc.
    PppOracle,
    /// The SPP owned deterministic trust-region solver
    /// ([`SolverRecipe::OwnedDeterministicTrf`]): a fixed-reduction-order dense
    /// subproblem factorization with no nalgebra LU and no black-box BLAS in that
    /// solve. It is pinned to its own frozen-bits golden rather than to an
    /// external library, and is selectable only for [`Technique::Spp`]. The owned
    /// kernel uses fixed-order scalar arithmetic for the complete trust-region
    /// assembly and factorization.
    OwnedDeterministic,
}

/// Runtime-selectable strategy identity. `Reference` strategies are 0-ULP
/// bit-exact to an external reference and remain the validation oracles;
/// `Canonical` is the single bounded-tolerance "best" model (P6). Canonical SPP,
/// RTK, and PPP are all wired.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StrategyId {
    /// A reference-faithful strategy: 0-ULP to `target` for `technique`.
    Reference {
        /// Technique passed to [`EstimationRecipe::for_reference`] and checked
        /// against the input technique before runner dispatch.
        technique: Technique,
        /// Reference target paired with `technique` when resolving the recipe;
        /// unsupported pairs are rejected as incompatible targets.
        target: ReferenceTarget,
    },
    /// The canonical strategy for `technique` (bounded-tolerance, truth-gated).
    Canonical {
        /// Technique passed to [`EstimationRecipe::for_canonical`] and used to
        /// select the corresponding canonical runner and screen family.
        technique: Technique,
    },
}

impl Default for StrategyId {
    fn default() -> Self {
        Self::Reference {
            technique: Technique::Spp,
            target: ReferenceTarget::Skyfield,
        }
    }
}

impl StrategyId {
    /// SPP, bit-exact to Skyfield (`spp::solve`).
    pub const fn spp_reference() -> Self {
        Self::Reference {
            technique: Technique::Spp,
            target: ReferenceTarget::Skyfield,
        }
    }

    /// RTK, bit-exact to RTKLIB (`rtk` / `rtk_filter`).
    pub const fn rtk_reference() -> Self {
        Self::Reference {
            technique: Technique::Rtk,
            target: ReferenceTarget::Rtklib,
        }
    }

    /// PPP, bit-exact to the PPP oracle arc (`precise_positioning`).
    pub const fn ppp_reference() -> Self {
        Self::Reference {
            technique: Technique::Ppp,
            target: ReferenceTarget::PppOracle,
        }
    }

    /// SPP via the owned deterministic trust-region solver
    /// ([`SolverRecipe::OwnedDeterministicTrf`]): the owned trust-region
    /// assembly and dense subproblem factorization, pinned to its own
    /// frozen-bits golden. Selecting this through
    /// [`crate::estimation::strategies::estimate`] drives the owned solver
    /// rather than the legacy nalgebra LU path.
    pub const fn spp_owned_deterministic() -> Self {
        Self::Reference {
            technique: Technique::Spp,
            target: ReferenceTarget::OwnedDeterministic,
        }
    }
}

/// Geometric range / light-time / transmit-time operation order. Each variant
/// names an existing range model; the substrate selects the op-order rather
/// than copying the helper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RangeRecipe {
    /// SPP closed-form light-time with a fixed transmit-time iteration count and
    /// a measured-pseudorange seed (`spp/mod.rs` `sat_model`).
    #[default]
    SppMeasuredPseudorangeFixedIter,
    /// `observables::predict` rounded-microsecond transmit time with a fixed
    /// light-time iteration count (PPP / forward-prediction model).
    ObservableRoundedMicrosecondFixedIter,
    /// RTK provided-transmit-position range with the RTKLIB first-order Sagnac
    /// scalar (`rtk_filter::model` line-of-sight / geometric range).
    RtkProvidedTxFirstOrderSagnac,
    /// Canonical: full iterative light-time (iterated to convergence, not a
    /// fixed truncation) with the closed-form Sagnac Z-rotation, never a
    /// first-order scalar Sagnac. Driven by [`EstimationRecipe::canonical_spp`]
    /// in the SPP measurement model; not used by any reference strategy.
    CanonicalLightTimeClosedFormSagnac,
}

/// Earth-rotation (Sagnac) correction operation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SagnacRecipe {
    /// Closed-form z-axis rotation of the satellite ECEF position by
    /// `OMEGA_E_DOT * tau` (SPP / observables).
    #[default]
    ClosedFormZRotation,
    /// RTKLIB first-order scalar Sagnac term added to the range
    /// (`rtk_filter::model`).
    RtklibFirstOrderScalar,
    /// No Sagnac correction (synthetic / ECI-consistent inputs).
    Off,
}

/// Local-frame / ENU / az-el basis construction operation order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrameRecipe {
    /// SPP Skyfield-parity ECEF->geodetic with the three-iteration AU-scaled
    /// latitude solve (`spp` geodetic conversion).
    #[default]
    SppSkyfieldAuThreeIter,
    /// Geocentric-up local frame used by the RTK elevation reference
    /// (`rtk_filter` elevation mask / antenna projection).
    GeocentricUpRtkReference,
    /// Geodetic NEU basis built from the cross-product convention
    /// (`precise_positioning::model` troposphere geometry).
    GeodeticNeuCrossProduct,
    /// DOP ENU rotation basis (`dop`).
    DopEnuRotation,
    /// Canonical: one consistent meters-native WGS84/ITRF geodetic basis under
    /// IERS conventions (the core PROJ-pinned closed-form solve), never a
    /// reference-specific AU-scaled path. Driven by
    /// [`EstimationRecipe::canonical_spp`]; not used by any reference strategy.
    CanonicalWgs84,
}

/// Normal-equation assembly tie-breaking / fold order. The tie order is the
/// pivot/elimination convention that fixes the bit pattern of the reduced
/// system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum NormalRecipe {
    /// SPP weighted-residual rows with a finite-difference design matrix
    /// (`spp` least-squares).
    #[default]
    SppWeightedResidualFiniteDifference,
    /// RTK double-difference block assembly with the first-tie covariance fold
    /// (`rtk_filter::normal` first-tie block).
    RtkDoubleDifferenceBlockFirstTie,
    /// PPP weighted normal equations with epoch-local receiver clocks eliminated
    /// and the reduced static system solved last-tie.
    PppDenseLastTie,
    /// Canonical square-root-information solve, shared by canonical RTK and
    /// canonical PPP: the SPD normal system is solved by the owned deterministic
    /// Cholesky factorization `Λ = L Lᵀ` plus forward/back substitution, where
    /// `L` is the information-matrix square root. For RTK this is the
    /// double-difference information system `Λ x = η` assembled by the same shared
    /// block fold the RTK reference uses; for PPP it is the weighted normal
    /// system assembled from the same undifferenced rows the PPP reference uses,
    /// with epoch-local receiver clocks eliminated before factorization. This is
    /// the numerically rigorous op-order for an SPD normal
    /// matrix (no pivoting; exploits symmetry), distinct from the reference RTK
    /// general first-tie Gaussian elimination
    /// ([`Self::RtkDoubleDifferenceBlockFirstTie`]) and the reference PPP last-tie
    /// Gaussian elimination ([`Self::PppDenseLastTie`]). Driven by
    /// [`EstimationRecipe::canonical_rtk`] and [`EstimationRecipe::canonical_ppp`]
    /// on the owned [`SolverRecipe::OwnedDeterministicCholesky`] kernel; not used
    /// by any reference strategy.
    CanonicalSquareRoot,
}

/// Linear-solve / factorization operation order. Determinism note: the legacy
/// SPP path is nalgebra LU (not bit-portable end-to-end), preserved as a named
/// variant; the owned deterministic kernel (P5) owns the complete dense
/// trust-region assembly and factorization with its own goldens -- see
/// [`Self::OwnedDeterministicTrf`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SolverRecipe {
    /// nalgebra trust-region least squares, the current SPP solver
    /// (`spp` / `crate::astro::math::least_squares`). Existing SPP goldens use
    /// this; kept unchanged.
    #[default]
    NalgebraTrfLegacy,
    /// Flat first-tie Gaussian elimination (RTK baseline/filter solve).
    FlatGaussianFirstTie,
    /// Dense last-tie Gaussian elimination (PPP solve,
    /// `crate::astro::math::linear::solve_linear_last_tie`).
    DenseGaussianLastTie,
    /// scipy host LAPACK reference solve (machine-dependent; only as a
    /// fingerprinted CI reference, never canonical).
    ScipyHostLapackReference,
    /// Owned deterministic Cholesky (square-root) linear solve, the canonical RTK
    /// (P6 increment 2) and canonical PPP (P6 increment 3) solver: the SPD normal
    /// system is factored `Λ = L Lᵀ` and solved by forward/back substitution
    /// through the owned
    /// [`crate::astro::math::linear::solve_flat_normal_square_root_into`] kernel,
    /// with no nalgebra LU and no black-box BLAS. Paired with
    /// [`NormalRecipe::CanonicalSquareRoot`]. Both the RTK and PPP canonical paths
    /// are owned scalar arithmetic and f64 sqrt is IEEE-754 correctly rounded, so
    /// like [`Self::OwnedDeterministicTrf`], its bit guarantee covers the full
    /// solve and is portable across platforms.
    OwnedDeterministicCholesky,
    /// Owned deterministic trust-region subproblem solve added in P5: a
    /// fixed-reduction-order dense Gaussian elimination (the
    /// `OwnedGaussianFirstTie` kernel) with no nalgebra LU or black-box BLAS.
    /// Its normal-matrix, gradient, cost, norm, and optimality reductions are
    /// fixed-order scalar operations too, and its frozen bits are portable
    /// across CPU targets.
    OwnedDeterministicTrf,
}

/// The full operation-order recipe a strategy composes: one variant per stage.
/// `Default` and the named constructors reproduce the CURRENT behavior of each
/// existing strategy, so selecting a recipe never changes a reference golden
/// (PPP goldens were re-frozen once when static PPP moved to clock-eliminated
/// reduced normals; see `estimation::strategies`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct EstimationRecipe {
    /// SPP transmit-time branch copied into `SppModelRecipe` and consumed by
    /// `sat_model`: the reference uses a measured-pseudorange seed with a fixed
    /// iteration count, while the canonical branch iterates to convergence.
    pub range: RangeRecipe,
    /// Earth-rotation operation passed by SPP to both satellite rotation and
    /// geometric-range calculation in the shared range substrate.
    pub sagnac: SagnacRecipe,
    /// Receiver-frame selection passed by SPP to the shared azimuth/elevation
    /// substrate; SPP accepts the Skyfield and canonical WGS84 frame variants.
    pub frame: FrameRecipe,
    /// Normal-equation reduction forwarded to the RTK and PPP solve seams; the
    /// reference arms retain their tie orders and the canonical arm selects the
    /// square-root reduction.
    pub normal: NormalRecipe,
    /// Factorization selection forwarded to the SPP trust-region and RTK solve
    /// seams; SPP recognizes its owned trust-region variant, while RTK uses the
    /// owned Cholesky variant only with [`NormalRecipe::CanonicalSquareRoot`].
    pub solver: SolverRecipe,
}

impl EstimationRecipe {
    /// The current SPP reference recipe (`spp::solve`, Skyfield-parity).
    pub const fn spp() -> Self {
        Self {
            range: RangeRecipe::SppMeasuredPseudorangeFixedIter,
            sagnac: SagnacRecipe::ClosedFormZRotation,
            frame: FrameRecipe::SppSkyfieldAuThreeIter,
            normal: NormalRecipe::SppWeightedResidualFiniteDifference,
            solver: SolverRecipe::NalgebraTrfLegacy,
        }
    }

    /// The current RTK reference recipe (`rtk` / `rtk_filter`, RTKLIB-parity).
    pub const fn rtk() -> Self {
        Self {
            range: RangeRecipe::RtkProvidedTxFirstOrderSagnac,
            sagnac: SagnacRecipe::RtklibFirstOrderScalar,
            frame: FrameRecipe::GeocentricUpRtkReference,
            normal: NormalRecipe::RtkDoubleDifferenceBlockFirstTie,
            solver: SolverRecipe::FlatGaussianFirstTie,
        }
    }

    /// The current PPP reference recipe (`precise_positioning`, oracle-parity).
    pub const fn ppp() -> Self {
        Self {
            range: RangeRecipe::ObservableRoundedMicrosecondFixedIter,
            sagnac: SagnacRecipe::ClosedFormZRotation,
            frame: FrameRecipe::GeodeticNeuCrossProduct,
            normal: NormalRecipe::PppDenseLastTie,
            solver: SolverRecipe::DenseGaussianLastTie,
        }
    }

    /// The SPP recipe driving the owned deterministic trust-region solver: the
    /// SPP reference model with [`SolverRecipe::OwnedDeterministicTrf`] swapped
    /// in for the legacy nalgebra LU linear-solve stage. Every other stage is the
    /// SPP reference op-order; the owned solver's assembly and factorization
    /// use fixed-order scalar arithmetic.
    pub const fn spp_owned_deterministic() -> Self {
        let mut recipe = Self::spp();
        recipe.solver = SolverRecipe::OwnedDeterministicTrf;
        recipe
    }

    /// The canonical SPP recipe: the single consistent IERS-rigorous SPP
    /// measurement model. It diverges from [`Self::spp`] (the Skyfield-faithful
    /// reference) only where the physics says to:
    /// - range: [`RangeRecipe::CanonicalLightTimeClosedFormSagnac`] iterates the
    ///   light-time loop to convergence (vs the reference's fixed
    ///   transmit-time truncation), with the closed-form Sagnac Z-rotation (never
    ///   a first-order scalar Sagnac).
    /// - frame: [`FrameRecipe::CanonicalWgs84`] solves ECEF->geodetic directly in
    ///   meters on the WGS84 ellipsoid (vs the reference's Skyfield AU-scaled
    ///   three-iteration latitude loop).
    /// - solver: [`SolverRecipe::OwnedDeterministicTrf`] owns the trust-region
    ///   assembly and subproblem factorization so canonical is deterministic
    ///   run-to-run across CPU targets.
    ///
    /// The Sagnac stage is the closed-form Z-rotation the SPP reference already
    /// uses (the rigorous form), and the normal stage is the SPP
    /// weighted-residual finite-difference assembly the trust-region solver
    /// consumes; neither needs a separate canonical variant for SPP.
    pub const fn canonical_spp() -> Self {
        Self {
            range: RangeRecipe::CanonicalLightTimeClosedFormSagnac,
            sagnac: SagnacRecipe::ClosedFormZRotation,
            frame: FrameRecipe::CanonicalWgs84,
            normal: NormalRecipe::SppWeightedResidualFiniteDifference,
            solver: SolverRecipe::OwnedDeterministicTrf,
        }
    }

    /// The canonical RTK recipe: the double-difference baseline under the
    /// numerically rigorous square-root-information solve. It keeps the RTK
    /// reference's double-difference measurement physics (the provided-transmit
    /// range with the RTKLIB first-order Sagnac scalar, the geocentric-up
    /// elevation frame), because the canonical RTK divergence the physics calls
    /// for is in the linear algebra, not the observation model: the same SPD
    /// information system the reference assembles is solved by the owned
    /// deterministic Cholesky square-root factorization
    /// ([`NormalRecipe::CanonicalSquareRoot`] on
    /// [`SolverRecipe::OwnedDeterministicCholesky`]) instead of the reference's
    /// general first-tie Gaussian elimination. The square-root solve needs no
    /// pivoting, exploits the symmetry of the SPD normal matrix, and is entirely
    /// owned scalar arithmetic (no nalgebra, no BLAS), so canonical RTK is
    /// well-conditioned and bit-reproducible across platforms.
    pub const fn canonical_rtk() -> Self {
        Self {
            range: RangeRecipe::RtkProvidedTxFirstOrderSagnac,
            sagnac: SagnacRecipe::RtklibFirstOrderScalar,
            frame: FrameRecipe::GeocentricUpRtkReference,
            normal: NormalRecipe::CanonicalSquareRoot,
            solver: SolverRecipe::OwnedDeterministicCholesky,
        }
    }

    /// The canonical PPP recipe: the undifferenced ionosphere-free PPP arc under
    /// the numerically rigorous square-root-information solve. Like
    /// [`Self::canonical_rtk`] it keeps the PPP reference's measurement physics
    /// (the rounded-microsecond fixed-iteration light-time with the rigorous
    /// closed-form Sagnac Z-rotation, and the geodetic NEU antenna frame), because
    /// the canonical PPP divergence the physics calls for is in the linear
    /// algebra, not the observation model: the same weighted normal equations
    /// the reference assembles from the undifferenced rows are reduced by
    /// eliminating epoch-local receiver clocks, then solved by the owned
    /// deterministic Cholesky square-root factorization
    /// ([`NormalRecipe::CanonicalSquareRoot`] on
    /// [`SolverRecipe::OwnedDeterministicCholesky`]) instead of the reference's
    /// dense last-tie Gaussian elimination ([`SolverRecipe::DenseGaussianLastTie`]).
    /// The square-root solve needs no pivoting, exploits the symmetry of the SPD
    /// normal matrix, and is entirely owned scalar arithmetic (no nalgebra, no
    /// BLAS), so it is well-conditioned and the solve itself is bit-portable.
    /// Determinism scope (calibrated, not overstated): unlike canonical RTK, the PPP
    /// measurement model that builds the rows evaluates troposphere mapping,
    /// antenna, and geodetic-frame transcendentals through the platform math
    /// library, so canonical PPP's overall output is bit-reproducible run-to-run on
    /// a pinned build but is not claimed bit-portable across platforms; only the
    /// owned Cholesky solve carries the cross-platform guarantee.
    pub const fn canonical_ppp() -> Self {
        Self {
            range: RangeRecipe::ObservableRoundedMicrosecondFixedIter,
            sagnac: SagnacRecipe::ClosedFormZRotation,
            frame: FrameRecipe::GeodeticNeuCrossProduct,
            normal: NormalRecipe::CanonicalSquareRoot,
            solver: SolverRecipe::OwnedDeterministicCholesky,
        }
    }

    /// The canonical recipe for a `technique`. Canonical SPP (P6 increment 1),
    /// canonical RTK (P6 increment 2), and canonical PPP (P6 increment 3) are all
    /// wired, so every technique has a canonical strategy. Returns `Option` to keep
    /// the resolver's "not yet implemented" surface stable.
    pub const fn for_canonical(technique: Technique) -> Option<Self> {
        match technique {
            Technique::Spp => Some(Self::canonical_spp()),
            Technique::Rtk => Some(Self::canonical_rtk()),
            Technique::Ppp => Some(Self::canonical_ppp()),
        }
    }

    /// The reference recipe for an explicit `(technique, target)` pair, or `None`
    /// if the pair is not a supported reference strategy. This is the single
    /// source of truth for which targets each technique can run: only the wired
    /// reference oracles (Skyfield for SPP, RTKLIB for RTK, the PPP oracle for
    /// PPP) and the SPP owned deterministic solver are valid. Every other pair
    /// (a cross-technique oracle, or the unwired scipy host-LAPACK reference) is
    /// rejected so an impossible strategy can never silently run a mismatched
    /// recipe.
    pub const fn for_reference(technique: Technique, target: ReferenceTarget) -> Option<Self> {
        match (technique, target) {
            (Technique::Spp, ReferenceTarget::Skyfield) => Some(Self::spp()),
            (Technique::Spp, ReferenceTarget::OwnedDeterministic) => {
                Some(Self::spp_owned_deterministic())
            }
            (Technique::Rtk, ReferenceTarget::Rtklib) => Some(Self::rtk()),
            (Technique::Ppp, ReferenceTarget::PppOracle) => Some(Self::ppp()),
            _ => None,
        }
    }
}

/// How a strategy forms its integer-ambiguity identifiers, and against what they
/// are referenced. Naming this lets the RTK and PPP fixed solvers share one
/// LAMBDA resolution kernel
/// (`crate::estimation::substrate::ambiguity::resolve_integer_lattice`) and
/// differ only in DATA rather than in separate algorithm trees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DifferencingMode {
    /// Double-differenced ambiguities, one reference satellite per constellation
    /// (the RTK baseline / sequential-filter convention: each non-reference
    /// satellite is differenced against its own system's reference).
    DoubleDifferencePerSystemReference,
    /// Undifferenced ambiguities, one per satellite per receiver (the PPP
    /// convention: no reference satellite, all satellites carry their own
    /// ionosphere-free ambiguity).
    Undifferenced,
}

/// Whether partial ambiguity resolution is attempted when the full-set integer
/// fix fails its ratio test, and with what floor on the retained subset size.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PartialResolution {
    /// Full-set only: a failed ratio test means "not fixed" (PPP, and the RTK
    /// sequential filter, both take the full set or nothing).
    Disabled,
    /// Confidence-ranked then exhaustive subset fallback down to
    /// `min_ambiguities` retained (the RTK static fixed solver,
    /// `rtk_filter::search::search_partial_fixed_ambiguities`).
    Exhaustive {
        /// Retained-subset floor copied from `partial_min_ambiguities` by
        /// [`AmbiguityIdPolicy::rtk_static`].
        min_ambiguities: usize,
    },
}

/// The integer-ambiguity identity/eligibility policy a strategy resolves under:
/// the strategy DATA that replaces the RTK-vs-PPP algorithm-tree split. The
/// LAMBDA resolution kernel is common; only these fields differ between the
/// reference strategies. Named in P3; consumed by the runtime selector in P4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbiguityIdPolicy {
    /// Ambiguity identity convention: the RTK constructors store per-system
    /// double differences, while [`AmbiguityIdPolicy::ppp`] stores undifferenced
    /// per-satellite ambiguities.
    pub differencing: DifferencingMode,
    /// Exclude float-only constellations from the integer search set.
    pub float_only_gating: bool,
    /// Resolution mode returned by the policy constructors: static RTK stores
    /// an exhaustive fallback, while sequential RTK and PPP store `Disabled`.
    pub partial: PartialResolution,
    /// Ratio-test acceptance threshold passed to the LAMBDA kernel.
    pub ratio_threshold: f64,
}

impl AmbiguityIdPolicy {
    /// The static RTK fixed-baseline policy (`rtk_filter::fixed`): per-system
    /// double differences, float-only constellations excluded from the search,
    /// partial resolution down to `partial_min_ambiguities`.
    pub const fn rtk_static(ratio_threshold: f64, partial_min_ambiguities: usize) -> Self {
        Self {
            differencing: DifferencingMode::DoubleDifferencePerSystemReference,
            float_only_gating: true,
            partial: PartialResolution::Exhaustive {
                min_ambiguities: partial_min_ambiguities,
            },
            ratio_threshold,
        }
    }

    /// The sequential RTK filter policy (`rtk_filter::update`): per-system double
    /// differences, float-only constellations excluded, full set or nothing.
    pub const fn rtk_sequential(ratio_threshold: f64) -> Self {
        Self {
            differencing: DifferencingMode::DoubleDifferencePerSystemReference,
            float_only_gating: true,
            partial: PartialResolution::Disabled,
            ratio_threshold,
        }
    }

    /// The static PPP fixed policy (`precise_positioning::fixed`): undifferenced
    /// per-satellite ambiguities, no constellation gating, full set or nothing.
    pub const fn ppp(ratio_threshold: f64) -> Self {
        Self {
            differencing: DifferencingMode::Undifferenced,
            float_only_gating: false,
            partial: PartialResolution::Disabled,
            ratio_threshold,
        }
    }
}

/// The operation order used to normalize one residual against its weight before
/// the sigma comparison in a per-residual screen. Naming the order keeps each
/// screen bit-identical while the formula lives in exactly one place
/// (`crate::estimation::substrate::qc::normalized_residual`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResidualNormRecipe {
    /// `value · weight` where `weight` is an inverse *sigma*
    /// (`1/sqrt(sigma_sat^2 + sigma_ref^2)`), so the normalized residual is
    /// `value / sigma`. The RTK static float/fixed least-squares baselines, whose
    /// DD rows weight by inverse sigma, screen their post-fit residuals this way.
    RtkInverseSigmaResidual,
    /// `|value| · sqrt(weight)` where `weight` is an inverse *sigma* (`1/sigma`):
    /// the residual magnitude times the square root of the inverse-sigma weight.
    /// The PPP float leave-one-out screen (PPP rows weight by inverse sigma, as
    /// `MeasurementWeights` documents).
    PppInverseSigmaMagnitude,
}

/// The residual-screen family a strategy applies after a solve. Strategy DATA
/// for the P4 selector; the chi-square variant is
/// the SPP RAIM aggregate test, the rest are per-residual sigma screens that
/// share `crate::estimation::substrate::qc::normalized_residual`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScreenKind {
    /// SPP RAIM: aggregate chi-square on the weighted residual sum, then FDE
    /// leave-one-out exclusion (`quality::raim` / `quality::fde`).
    RaimChiSquare,
    /// RTK static fixed: worst information-weighted residual vs a sigma gate,
    /// excluding the worst satellite within a budget
    /// (`rtk_filter::fixed::solve_fixed_baseline_validated`).
    RtkFixedResidualValidation,
    /// PPP float: worst studentized residual vs a sigma gate, leave-one-out prune
    /// and re-solve while WRMS improves (`precise_positioning::float`).
    PppFloatLeaveOneOut,
}

impl ScreenKind {
    /// The per-residual normalization op-order this screen uses, or `None` for
    /// the aggregate chi-square RAIM screen (which scores the weighted residual
    /// sum, not individual residuals).
    pub const fn residual_norm(self) -> Option<ResidualNormRecipe> {
        match self {
            Self::RaimChiSquare => None,
            Self::RtkFixedResidualValidation => Some(ResidualNormRecipe::RtkInverseSigmaResidual),
            Self::PppFloatLeaveOneOut => Some(ResidualNormRecipe::PppInverseSigmaMagnitude),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_name_current_spp_behavior() {
        // The per-stage defaults are the SPP reference op-orders, so an
        // unspecified recipe reproduces the current SPP path.
        assert_eq!(EstimationRecipe::default(), EstimationRecipe::spp());
        assert_eq!(
            RangeRecipe::default(),
            RangeRecipe::SppMeasuredPseudorangeFixedIter
        );
        assert_eq!(SagnacRecipe::default(), SagnacRecipe::ClosedFormZRotation);
        assert_eq!(FrameRecipe::default(), FrameRecipe::SppSkyfieldAuThreeIter);
        assert_eq!(
            NormalRecipe::default(),
            NormalRecipe::SppWeightedResidualFiniteDifference
        );
        assert_eq!(SolverRecipe::default(), SolverRecipe::NalgebraTrfLegacy);
        assert_eq!(StrategyId::default(), StrategyId::spp_reference());
    }

    #[test]
    fn strategy_constructors_match_reference_targets() {
        assert_eq!(
            StrategyId::spp_reference(),
            StrategyId::Reference {
                technique: Technique::Spp,
                target: ReferenceTarget::Skyfield,
            }
        );
        assert_eq!(
            StrategyId::rtk_reference(),
            StrategyId::Reference {
                technique: Technique::Rtk,
                target: ReferenceTarget::Rtklib,
            }
        );
        assert_eq!(
            StrategyId::ppp_reference(),
            StrategyId::Reference {
                technique: Technique::Ppp,
                target: ReferenceTarget::PppOracle,
            }
        );
    }

    #[test]
    fn for_reference_selects_each_supported_pairs_recipe() {
        assert_eq!(
            EstimationRecipe::for_reference(Technique::Spp, ReferenceTarget::Skyfield),
            Some(EstimationRecipe::spp())
        );
        assert_eq!(
            EstimationRecipe::for_reference(Technique::Rtk, ReferenceTarget::Rtklib),
            Some(EstimationRecipe::rtk())
        );
        assert_eq!(
            EstimationRecipe::for_reference(Technique::Ppp, ReferenceTarget::PppOracle),
            Some(EstimationRecipe::ppp())
        );
    }

    #[test]
    fn owned_deterministic_recipe_swaps_only_the_solver() {
        let owned = EstimationRecipe::spp_owned_deterministic();
        assert_eq!(owned.solver, SolverRecipe::OwnedDeterministicTrf);
        // Every non-solver stage is the SPP reference op-order.
        assert_eq!(
            EstimationRecipe {
                solver: SolverRecipe::NalgebraTrfLegacy,
                ..owned
            },
            EstimationRecipe::spp()
        );
        assert_eq!(
            EstimationRecipe::for_reference(Technique::Spp, ReferenceTarget::OwnedDeterministic),
            Some(owned)
        );
    }

    #[test]
    fn canonical_spp_recipe_uses_the_rigorous_op_orders() {
        let canonical = EstimationRecipe::canonical_spp();
        // Range: full iterative light-time with closed-form Sagnac, not the SPP
        // reference's fixed-iteration measured-pseudorange recipe.
        assert_eq!(
            canonical.range,
            RangeRecipe::CanonicalLightTimeClosedFormSagnac
        );
        assert_ne!(canonical.range, EstimationRecipe::spp().range);
        // Frame: one consistent meters-native WGS84 basis, not the Skyfield AU
        // path.
        assert_eq!(canonical.frame, FrameRecipe::CanonicalWgs84);
        assert_ne!(canonical.frame, EstimationRecipe::spp().frame);
        // Sagnac stays the closed-form Z-rotation (the rigorous form the SPP
        // reference already uses); the canonical divergence is never a
        // first-order scalar Sagnac.
        assert_eq!(canonical.sagnac, SagnacRecipe::ClosedFormZRotation);
        assert_ne!(canonical.sagnac, SagnacRecipe::RtklibFirstOrderScalar);
        // Solver: the owned deterministic factorization, for run-to-run
        // determinism on a pinned build.
        assert_eq!(canonical.solver, SolverRecipe::OwnedDeterministicTrf);
        assert_eq!(
            EstimationRecipe::for_canonical(Technique::Spp),
            Some(canonical)
        );
    }

    #[test]
    fn canonical_rtk_recipe_uses_the_square_root_solve() {
        let canonical = EstimationRecipe::canonical_rtk();
        // Normal + solver: the owned Cholesky square-root information solve, not
        // the reference RTK first-tie Gaussian elimination.
        assert_eq!(canonical.normal, NormalRecipe::CanonicalSquareRoot);
        assert_eq!(canonical.solver, SolverRecipe::OwnedDeterministicCholesky);
        assert_ne!(canonical.normal, EstimationRecipe::rtk().normal);
        assert_ne!(canonical.solver, EstimationRecipe::rtk().solver);
        // Measurement physics stays the RTK reference double-difference model: the
        // canonical RTK divergence is in the linear algebra, not the observation
        // model, so range/sagnac/frame match the reference.
        assert_eq!(canonical.range, EstimationRecipe::rtk().range);
        assert_eq!(canonical.sagnac, EstimationRecipe::rtk().sagnac);
        assert_eq!(canonical.frame, EstimationRecipe::rtk().frame);
        assert_eq!(
            EstimationRecipe::for_canonical(Technique::Rtk),
            Some(canonical)
        );
    }

    #[test]
    fn canonical_ppp_recipe_uses_the_square_root_solve() {
        let canonical = EstimationRecipe::canonical_ppp();
        // Normal + solver: the owned Cholesky square-root information solve, not
        // the reference PPP dense last-tie Gaussian elimination.
        assert_eq!(canonical.normal, NormalRecipe::CanonicalSquareRoot);
        assert_eq!(canonical.solver, SolverRecipe::OwnedDeterministicCholesky);
        assert_ne!(canonical.normal, EstimationRecipe::ppp().normal);
        assert_ne!(canonical.solver, EstimationRecipe::ppp().solver);
        // Measurement physics stays the PPP reference undifferenced model: the
        // canonical PPP divergence is in the linear algebra, not the observation
        // model, so range/sagnac/frame match the reference.
        assert_eq!(canonical.range, EstimationRecipe::ppp().range);
        assert_eq!(canonical.sagnac, EstimationRecipe::ppp().sagnac);
        assert_eq!(canonical.frame, EstimationRecipe::ppp().frame);
        // Canonical RTK and PPP share the square-root normal + owned Cholesky
        // solver (the same numerically rigorous SPD op-order).
        assert_eq!(canonical.normal, EstimationRecipe::canonical_rtk().normal);
        assert_eq!(canonical.solver, EstimationRecipe::canonical_rtk().solver);
        assert_eq!(
            EstimationRecipe::for_canonical(Technique::Ppp),
            Some(canonical)
        );
    }

    #[test]
    fn for_canonical_wires_all_three_techniques() {
        assert_eq!(
            EstimationRecipe::for_canonical(Technique::Spp),
            Some(EstimationRecipe::canonical_spp())
        );
        assert_eq!(
            EstimationRecipe::for_canonical(Technique::Rtk),
            Some(EstimationRecipe::canonical_rtk())
        );
        assert_eq!(
            EstimationRecipe::for_canonical(Technique::Ppp),
            Some(EstimationRecipe::canonical_ppp())
        );
    }

    #[test]
    fn for_reference_rejects_impossible_pairs() {
        // Cross-technique oracles and the unwired scipy reference are not
        // supported reference strategies.
        for (technique, target) in [
            (Technique::Spp, ReferenceTarget::Rtklib),
            (Technique::Spp, ReferenceTarget::PppOracle),
            (Technique::Spp, ReferenceTarget::Scipy),
            (Technique::Rtk, ReferenceTarget::Skyfield),
            (Technique::Rtk, ReferenceTarget::OwnedDeterministic),
            (Technique::Rtk, ReferenceTarget::PppOracle),
            (Technique::Ppp, ReferenceTarget::Skyfield),
            (Technique::Ppp, ReferenceTarget::OwnedDeterministic),
        ] {
            assert_eq!(
                EstimationRecipe::for_reference(technique, target),
                None,
                "{technique:?} + {target:?} must be rejected"
            );
        }
    }

    #[test]
    fn reference_ambiguity_policies_name_current_behavior() {
        let rtk_static = AmbiguityIdPolicy::rtk_static(3.0, 4);
        assert_eq!(
            rtk_static.differencing,
            DifferencingMode::DoubleDifferencePerSystemReference
        );
        assert!(rtk_static.float_only_gating);
        assert_eq!(
            rtk_static.partial,
            PartialResolution::Exhaustive { min_ambiguities: 4 }
        );

        let rtk_seq = AmbiguityIdPolicy::rtk_sequential(3.0);
        assert_eq!(
            rtk_seq.differencing,
            DifferencingMode::DoubleDifferencePerSystemReference
        );
        assert!(rtk_seq.float_only_gating);
        assert_eq!(rtk_seq.partial, PartialResolution::Disabled);

        let ppp = AmbiguityIdPolicy::ppp(2.5);
        assert_eq!(ppp.differencing, DifferencingMode::Undifferenced);
        assert!(!ppp.float_only_gating);
        assert_eq!(ppp.partial, PartialResolution::Disabled);
    }

    #[test]
    fn rtk_and_ppp_id_policies_differ_only_in_data() {
        // Same LAMBDA kernel, different identity/eligibility data: the two stacks
        // are no longer separate algorithm trees, only different policy values.
        let rtk = AmbiguityIdPolicy::rtk_static(3.0, 1);
        let ppp = AmbiguityIdPolicy::ppp(3.0);
        assert_ne!(rtk.differencing, ppp.differencing);
        assert_ne!(rtk.float_only_gating, ppp.float_only_gating);
        assert_ne!(rtk.partial, ppp.partial);
    }

    #[test]
    fn screen_kinds_select_their_normalization_order() {
        assert_eq!(ScreenKind::RaimChiSquare.residual_norm(), None);
        assert_eq!(
            ScreenKind::RtkFixedResidualValidation.residual_norm(),
            Some(ResidualNormRecipe::RtkInverseSigmaResidual)
        );
        assert_eq!(
            ScreenKind::PppFloatLeaveOneOut.residual_norm(),
            Some(ResidualNormRecipe::PppInverseSigmaMagnitude)
        );
    }

    #[test]
    fn each_strategy_selects_a_distinct_solver_order() {
        // The three reference strategies must not collapse onto one solver
        // op-order; that distinction is what preserves their separate goldens.
        assert_ne!(
            EstimationRecipe::spp().solver,
            EstimationRecipe::rtk().solver
        );
        assert_ne!(
            EstimationRecipe::rtk().solver,
            EstimationRecipe::ppp().solver
        );
        assert_ne!(
            EstimationRecipe::spp().solver,
            EstimationRecipe::ppp().solver
        );
    }
}
