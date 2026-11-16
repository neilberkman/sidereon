//! Sequential RTK baseline arc driver.
//!
//! This leaf owns the high-level "raw rover+base epochs in, per-epoch baseline
//! solutions out" orchestration that previously lived only in the Elixir binding
//! (`solve_filter_baseline_epochs/3`). It performs epoch
//! normalization, common-satellite selection, per-system reference selection,
//! single-/double-difference epoch construction, sequential filter
//! initialization with phase-minus-code ambiguity seeding, and the per-epoch
//! [`update_epoch`] loop.
//!
//! It re-implements none of the numerics: reference scoring comes from
//! [`crate::rtk::baseline_reference_satellites`], the per-epoch Kalman
//! predict/update/search/hold from [`update_epoch`], and the SD/DD ambiguity-id
//! naming from [`crate::rtk`]. Only the marshalling and ordering policy moves
//! here, reproducing the Elixir reference so a later binding delegation is
//! bit-for-bit:
//!
//! - **Satellite availability** per epoch is the intersection of the base
//!   observation, rover observation, shared-position, base-position, and
//!   rover-position keys; the arc satellite set is their union, requiring at
//!   least four. (The per-receiver position maps default to the shared map when
//!   left empty.)
//! - **References** are selected once for the whole arc (per constellation) via
//!   the geometry-based [`crate::rtk::baseline_reference_satellites`].
//! - **Column order** of the filter state is the globally sorted
//!   `(satellite, ambiguity_id)` SD-ambiguity set, each column seeded on first
//!   sighting with the single-difference phase-minus-code value, so the
//!   information matrix is column-identical to the reference.
//! - **Prediction deltas** between consecutive epochs come from the optional
//!   per-epoch `prediction_epoch` (exact) or `prediction_time_s`: the exact
//!   difference, rounded once, when both epochs carry an exact epoch, otherwise
//!   the difference of the seconds; the first epoch is zero, a missing time is
//!   zero under [`DynamicsModel::ConstantPosition`] (lenient) and an error under
//!   [`DynamicsModel::VelocityPropagated`] (strict).

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::math::linear::invert_matrix_first_tie;
use crate::astro::time::ExactEpoch;
use crate::carrier_phase::CycleSlipOptions;
use crate::geometry_quality::{classify, GeometryQuality, GeometryQualityThresholds};
use crate::id::constellation_letter;
use crate::rtk::{
    apply_elevation_mask, baseline_reference_satellites, dd_ambiguity_token,
    estimate_wide_lane_ambiguities, hatch_smooth_baseline_code_epochs,
    prepare_cycle_slip_baseline_epochs, prepare_dual_cycle_slip_baseline_epochs,
    prepare_ionosphere_free_baseline_epochs, sd_ambiguity_token, BaselineReferenceEpoch,
    BaselineReferenceSelection, CodeSmoothingEpoch, CodeSmoothingError, CodeSmoothingObservation,
    CycleSlipPolicy, CycleSlipPrepError, CycleSlipSplitArc, DoubleDifferenceError,
    DualCycleSlipEpoch, DualCycleSlipObservation, DualEpoch, DualIonosphereFreeSetupEpoch,
    DualObservation, DualSatelliteObservation, ElevationMaskEpoch, IonosphereFreeBaselineEpoch,
    IonosphereFreeBaselineError, Observation as CoreRtkObservation, WideLaneError, WideLaneOptions,
};

use super::{
    baseline_ambiguity_index_core, solve_fixed_baseline_validated, solve_float_baseline,
    update_epoch, AmbiguityScale, AmbiguitySet, DynamicsModel, Epoch, FilterState,
    FilterStateValidationError, FloatBaselineSolution, FloatResidual, FloatSolveError,
    IntegerSearchMeta, MeasModel, SatMeas, UpdateError, UpdateOpts, ValidatedFixedBaselineSolution,
    ValidatedFixedSolveError, ValidatedFixedSolveOpts,
};

/// Minimum number of arc satellites required to attempt the baseline solve.
const MINIMUM_ARC_SATELLITES: usize = 4;

/// One raw single-frequency code/carrier observation at a receiver.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkArcObservation {
    /// Physical satellite id token, e.g. `"G05"`.
    pub satellite_id: String,
    /// Ambiguity-arc id. A clean arc uses the satellite id; a cycle-slip split
    /// carries a distinct id (e.g. `"G05#2"`) so the single-difference key resets.
    pub ambiguity_id: String,
    /// Pseudorange from the selected code observable, in meters. The arc solve
    /// uses it in the code measurement and in the first-sighting ambiguity seed.
    pub code_m: f64,
    /// Carrier phase from the selected phase observable, converted to meters.
    /// The arc solve uses it in the phase measurement and in the first-sighting
    /// ambiguity seed.
    pub phase_m: f64,
    /// Optional loss-of-lock indicator. Only consumed by the optional cycle-slip
    /// preprocessing ([`RtkArcPreprocessing::cycle_slip`]): bit 0 set marks a slip
    /// on this satellite at this epoch. `None` (default) is no-LLI, which never
    /// triggers a slip and leaves the solve unchanged.
    pub lli: Option<i64>,
}

/// One raw RTK arc epoch: paired base/rover observations and the satellite
/// positions needed to form double differences.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkArcEpoch {
    /// Base observations retained for satellites that also have a rover record
    /// and all required shared and transmit-time position entries.
    pub base: Vec<RtkArcObservation>,
    /// Rover observations retained for satellites that also have a base record
    /// and all required shared and transmit-time position entries.
    pub rover: Vec<RtkArcObservation>,
    /// Shared receive-time satellite ECEF positions (metres), used for the
    /// elevation-dependent variance model and as the reference geometry.
    pub satellite_positions_m: BTreeMap<String, [f64; 3]>,
    /// Transmit-time satellite ECEF positions for the base receiver. Empty
    /// defaults to [`Self::satellite_positions_m`].
    pub base_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    /// Transmit-time satellite ECEF positions for the rover receiver. Empty
    /// defaults to [`Self::satellite_positions_m`].
    pub rover_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    /// Optional rover ECEF velocity (metres/second) for the velocity-propagated
    /// prediction branch.
    pub velocity_mps: Option<[f64; 3]>,
    /// Optional epoch time coordinate (seconds) for prediction-delta computation.
    pub prediction_time_s: Option<f64>,
    /// Optional epoch held exactly. When this epoch and the timed epoch before
    /// it both carry one, the prediction delta is their exact difference,
    /// rounded once ([`ExactEpoch::seconds_since`]), and `prediction_time_s`
    /// is not used for it.
    pub prediction_epoch: Option<ExactEpoch>,
}

/// Optional preprocessing chained ahead of the core arc solve.
///
/// Each stage is opt-in; an all-`None`/all-default value (the [`Default`]) makes
/// [`solve_rtk_arc`] behave exactly as the bare core solve. When set, the stages
/// run in this fixed order before the sequential filter: cycle-slip handling,
/// then Hatch code smoothing, then elevation masking. Every stage delegates
/// verbatim to the standalone `crate::rtk` preprocessing function of the same
/// name, so the driver result equals manually composing prepare-then-solve.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RtkArcPreprocessing {
    /// Cycle-slip handling policy applied via
    /// [`crate::rtk::prepare_cycle_slip_baseline_epochs`]. `None` skips
    /// cycle-slip detection, arc splitting, and reacquisition segmentation. Reads
    /// [`RtkArcObservation::lli`].
    pub cycle_slip: Option<CycleSlipPolicy>,
    /// Hatch code-smoothing window cap applied via
    /// [`crate::rtk::hatch_smooth_baseline_code_epochs`]. `None` skips smoothing.
    pub hatch_window_cap: Option<usize>,
    /// Elevation mask (degrees). Preprocessing applies the base-side mask at the
    /// fixed base position. Each sequential filter epoch applies the rover-side
    /// mask at that epoch's predicted filter position and repeats the screen at
    /// the updated position. [`RtkArcEpochSolution::elevation_mask_converged`]
    /// reports whether the screen settled before its iteration cap. RTKLIB `zdres` uses its
    /// current state too; in kinematic mode without dynamics RTKLIB first seeds
    /// that state from its full single-point solver, while this library retains
    /// and predicts its existing state because an [`RtkArcEpoch`] does not carry
    /// the corrections required to reproduce that single-point solve. `None`
    /// skips masking. The sequential screen and its inner solve share
    /// [`UpdateOpts::max_iterations`]; the static screen and inner solve share
    /// [`super::FloatSolveOpts::max_iterations`].
    pub elevation_mask_deg: Option<f64>,
}

impl RtkArcPreprocessing {
    /// True when at least one preprocessing stage is enabled.
    fn is_active(&self) -> bool {
        self.cycle_slip.is_some()
            || self.hatch_window_cap.is_some()
            || self.elevation_mask_deg.is_some()
    }
}

/// Sequential RTK arc driver configuration.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RtkArcConfig {
    /// Base-station ECEF position (metres).
    pub base_m: [f64; 3],
    /// Reference-satellite selection policy (default geometry-based per system).
    /// A reference change transports the existing single-difference gauge.
    /// Held integer differences are re-expressed when the new reference is
    /// already held and carrier wavelengths agree. Otherwise that
    /// constellation's holds are released with per-epoch typed records;
    /// pending constraints first enter the carried information. Mixed-wavelength
    /// constellations remain float-only while their reference differs from the
    /// arc reference. Returning to the arc reference permits fixing again.
    pub reference: BaselineReferenceSelection,
    /// Measurement model (sigmas, Sagnac, stochastic model).
    pub model: MeasModel,
    /// Baseline prior sigma (metres) for the initial information matrix.
    pub baseline_prior_sigma_m: f64,
    /// Ambiguity prior sigma (metres) for each new SD ambiguity column.
    pub ambiguity_prior_sigma_m: f64,
    /// Initial baseline guess (metres, ECEF rover - base).
    pub initial_baseline_m: [f64; 3],
    /// Per-ambiguity carrier wavelengths (metres) for the integer search.
    /// Maps describe double differences against the selected arc reference.
    /// A missing arc-reference wavelength is inferred only when all supplied
    /// wavelengths in that constellation agree; otherwise reference changes
    /// release holds rather than inventing an integer difference.
    pub wavelengths_m: BTreeMap<String, f64>,
    /// Per-ambiguity code-to-phase metre offsets relative to the selected arc
    /// reference, whose offset and integer are zero (no entry is needed).
    /// On changing to r', offsets are internally re-expressed as off_i-off_r',
    /// including -off_r' for the old reference. Caller maps are never mutated.
    pub offsets_m: BTreeMap<String, f64>,
    /// Per-epoch sequential-update controls (hold sigma, tolerances, dynamics,
    /// float-only systems, AR arming, ratio threshold). The
    /// receiver-antenna PCO/PCV corrections also live here
    /// ([`UpdateOpts::receiver_antenna_corrections`]) and are applied verbatim by
    /// the per-epoch [`update_epoch`] / core antenna-correction path.
    pub update_opts: UpdateOpts,
    /// Optional preprocessing chained ahead of the solve. [`Default`] (all stages
    /// off) preserves the bare core-solve behavior for existing callers.
    pub preprocessing: RtkArcPreprocessing,
}

impl RtkArcConfig {
    /// Build sequential RTK settings from every required arc configuration.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        base_m: [f64; 3],
        reference: BaselineReferenceSelection,
        model: MeasModel,
        baseline_prior_sigma_m: f64,
        ambiguity_prior_sigma_m: f64,
        initial_baseline_m: [f64; 3],
        wavelengths_m: BTreeMap<String, f64>,
        offsets_m: BTreeMap<String, f64>,
        update_opts: UpdateOpts,
        preprocessing: RtkArcPreprocessing,
    ) -> Self {
        Self {
            base_m,
            reference,
            model,
            baseline_prior_sigma_m,
            ambiguity_prior_sigma_m,
            initial_baseline_m,
            wavelengths_m,
            offsets_m,
            update_opts,
            preprocessing,
        }
    }
}

/// Configuration consumed by [`solve_static_rtk_arc`] for a static batch RTK
/// arc solve.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RtkStaticArcConfig {
    /// Sequential-arc settings reused for input preprocessing, geometry, and
    /// measurement and ambiguity modeling.
    pub arc: RtkArcConfig,
    /// Validated options for the float solve, fixed solve, and residual checks.
    pub opts: ValidatedFixedSolveOpts,
}

impl RtkStaticArcConfig {
    /// Build static RTK settings from the sequential arc and validated solve
    /// options.
    #[must_use]
    pub const fn new(arc: RtkArcConfig, opts: ValidatedFixedSolveOpts) -> Self {
        Self { arc, opts }
    }
}

/// Results assembled by [`solve_static_rtk_arc`], including both batch solver
/// outputs and metadata produced while preparing the input epochs.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkStaticArcSolution {
    /// Geometry observability and covariance-validation diagnostics for the
    /// static batch design. This mirrors the nested float solution's final
    /// design. Snapshot arc solves pass no propagated prior into geometry
    /// classification, so `ZeroRedundancy` bounds are unvalidated and `Weak`
    /// bounds are unclamped. Sequential filter epochs use the same instantaneous
    /// design diagnostics, but a full-rank propagated prior can validate
    /// zero-redundancy covariance.
    pub geometry_quality: GeometryQuality,
    /// Per-constellation reference single-difference ambiguity ids returned by
    /// `build_static_batch_epochs` for the batch epochs.
    pub references: BTreeMap<String, String>,
    /// Globally ordered ambiguity-column ids returned by
    /// `baseline_ambiguity_index_core` for the batch epochs.
    pub ambiguity_ids: Vec<String>,
    /// Physical satellite associated with each ambiguity-column id, as returned
    /// by `baseline_ambiguity_index_core` for scale lookup and reporting.
    pub ambiguity_satellites: BTreeMap<String, String>,
    /// Float baseline solution returned by `solve_float_baseline` over all batch
    /// epochs.
    pub float_solution: FloatBaselineSolution,
    /// Validated fixed baseline solution returned by
    /// `solve_fixed_baseline_validated` over the same batch and ambiguity set.
    pub fixed_solution: ValidatedFixedBaselineSolution,
    /// Satellites copied from `preprocess_arc` after cycle-slip preprocessing
    /// applies its drop policy.
    pub dropped_sats: Vec<String>,
    /// Arc records copied from `preprocess_arc` after cycle-slip preprocessing
    /// applies its split policy.
    pub split_cycle_slip_arcs: Vec<CycleSlipSplitArc>,
    /// Satellites copied from `preprocess_arc` after the configured elevation
    /// mask removes them.
    pub elevation_masked_sats: Vec<String>,
    /// Whether the static rover-side elevation screen stabilized before the
    /// configured float-iteration limit. This is `true` when no elevation mask
    /// is configured. On `false`, the returned solution is the last completed
    /// solve, so its state, residuals, ambiguity columns, and
    /// [`Self::elevation_masked_sats`] remain mutually consistent; the next
    /// proposed satellite set was not solved.
    pub elevation_mask_converged: bool,
}

/// Errors that [`solve_static_rtk_arc`] can return while preparing or solving a
/// static RTK arc.
#[derive(Debug, Clone, PartialEq)]
pub enum RtkStaticArcError {
    /// `solve_static_rtk_arc` failed during arc construction, reference or
    /// filter-state setup, or enabled preprocessing before the batch solvers
    /// completed.
    Arc(RtkArcError),
    /// `solve_float_baseline` failed after the batch was constructed.
    Float(FloatSolveError),
    /// `solve_fixed_baseline_validated` failed after the float solve.
    Fixed(ValidatedFixedSolveError),
}

impl core::fmt::Display for RtkStaticArcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Arc(error) => write!(f, "{error}"),
            Self::Float(error) => write!(f, "{error}"),
            Self::Fixed(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for RtkStaticArcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Arc(error) => Some(error),
            Self::Float(error) => Some(error),
            Self::Fixed(error) => Some(error),
        }
    }
}

impl From<RtkArcError> for RtkStaticArcError {
    fn from(error: RtkArcError) -> Self {
        Self::Arc(error)
    }
}

/// Dual-frequency values extracted by `dual_frequency_observations` for one
/// receiver and one RTK satellite.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkDualFrequencyObservation {
    /// Ambiguity-arc identifier, initialized from the satellite token by the
    /// RINEX builder and possibly renamed by cycle-slip splitting.
    pub ambiguity_id: String,
    /// First-frequency pseudorange in meters, copied by
    /// `dual_frequency_observations` and passed unchanged to dual-frequency
    /// preparation.
    pub p1_m: f64,
    /// Second-frequency pseudorange in meters, copied by
    /// `dual_frequency_observations` and passed unchanged to dual-frequency
    /// preparation.
    pub p2_m: f64,
    /// First-frequency carrier phase in cycles, copied by
    /// `dual_frequency_observations` and retained in cycles for dual-frequency
    /// preparation.
    pub phi1_cycles: f64,
    /// Second-frequency carrier phase in cycles, copied by
    /// `dual_frequency_observations` and retained in cycles for dual-frequency
    /// preparation.
    pub phi2_cycles: f64,
    /// First selected carrier frequency in hertz, looked up from the RINEX
    /// phase observable by `dual_frequency_observations`.
    pub f1_hz: f64,
    /// Second selected carrier frequency in hertz, looked up from the RINEX
    /// phase observable by `dual_frequency_observations`.
    pub f2_hz: f64,
    /// Optional loss-of-lock value from the first phase observable, consumed by
    /// dual-frequency cycle-slip preprocessing.
    pub lli1: Option<i64>,
    /// Optional loss-of-lock value from the second phase observable, consumed by
    /// dual-frequency cycle-slip preprocessing.
    pub lli2: Option<i64>,
}

/// A base/rover dual-frequency pair emitted by
/// `build_dual_frequency_rinex_rtk_arc` for one satellite and epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkDualFrequencySatelliteObservation {
    /// Satellite token used to pair observations and position-map entries.
    pub satellite_id: String,
    /// Complete selected dual-frequency observation extracted from the base
    /// RINEX epoch.
    pub base: RtkDualFrequencyObservation,
    /// Complete selected dual-frequency observation extracted from the rover
    /// RINEX epoch matched to the base epoch.
    pub rover: RtkDualFrequencyObservation,
}

/// A dual-frequency RTK epoch emitted by
/// `build_dual_frequency_rinex_rtk_arc` for wide-lane and ionosphere-free
/// preparation.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkDualFrequencyArcEpoch {
    /// Whole part returned by `civil_to_julian_split` for the civil epoch and
    /// passed to ionosphere-free setup.
    pub jd_whole: f64,
    /// Fractional part returned by `civil_to_julian_split` for the civil epoch
    /// and passed to ionosphere-free setup.
    pub jd_fraction: f64,
    /// Fixed-width civil timestamp used to order dual cycle-slip records; absent
    /// values fall back to the input index during preprocessing.
    pub epoch_sort_key: Option<String>,
    /// Seconds since J2000 used by dual cycle-slip preprocessing to detect gaps.
    pub gap_time_s: Option<f64>,
    /// The epoch held exactly, used by dual cycle-slip preprocessing to detect
    /// gaps exactly when consecutive epochs both carry one.
    pub gap_epoch: Option<ExactEpoch>,
    /// Base/rover pairs retained after complete signal, ephemeris, and position
    /// checks.
    pub observations: Vec<RtkDualFrequencySatelliteObservation>,
    /// Shared receive-time satellite ECEF positions, in meters, used for
    /// reference geometry.
    pub satellite_positions_m: BTreeMap<String, [f64; 3]>,
    /// Base transmit-time satellite ECEF positions, in meters; an empty map
    /// makes the driver use [`Self::satellite_positions_m`].
    pub base_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    /// Rover transmit-time satellite ECEF positions, in meters; an empty map
    /// makes the driver use [`Self::satellite_positions_m`].
    pub rover_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    /// Optional rover ECEF velocity, in meters per second, preserved for a later
    /// velocity-propagated sequential solve.
    pub velocity_mps: Option<[f64; 3]>,
    /// Optional seconds-since-J2000 coordinate preserved for prediction deltas
    /// in a later sequential solve.
    pub prediction_time_s: Option<f64>,
    /// Optional exact epoch preserved for prediction deltas in a later
    /// sequential solve (see [`RtkArcEpoch::prediction_epoch`]).
    pub prediction_epoch: Option<ExactEpoch>,
}

/// Settings forwarded by `prepare_dual_frequency_arc` to dual-frequency
/// cycle-slip preprocessing for a wide-lane arc.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct RtkDualCycleSlipConfig {
    /// Action selected by dual-frequency preprocessing when it detects a slip.
    pub policy: CycleSlipPolicy,
    /// Detector thresholds and gap-handling options forwarded unchanged to
    /// `prepare_dual_cycle_slip_baseline_epochs`.
    pub options: CycleSlipOptions,
}

impl RtkDualCycleSlipConfig {
    /// Build dual-frequency cycle-slip settings from both required policies.
    #[must_use]
    pub const fn new(policy: CycleSlipPolicy, options: CycleSlipOptions) -> Self {
        Self { policy, options }
    }
}

/// Inputs consumed by [`fix_wide_lane_rtk_arc`] for dual-frequency wide-lane
/// ambiguity estimation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RtkWideLaneArcConfig {
    /// Base-station ECEF position, in meters, used for reference geometry.
    pub base_m: [f64; 3],
    /// Policy used to select one reference satellite per constellation.
    pub reference: BaselineReferenceSelection,
    /// Wide-lane estimator options passed to each selected constellation.
    pub options: WideLaneOptions,
    /// Optional dual-frequency cycle-slip preparation before reference selection
    /// and wide-lane estimation.
    pub cycle_slip: Option<RtkDualCycleSlipConfig>,
}

impl RtkWideLaneArcConfig {
    /// Build wide-lane arc settings from the base position, reference policy,
    /// estimator options, and optional cycle-slip policy.
    #[must_use]
    pub const fn new(
        base_m: [f64; 3],
        reference: BaselineReferenceSelection,
        options: WideLaneOptions,
        cycle_slip: Option<RtkDualCycleSlipConfig>,
    ) -> Self {
        Self {
            base_m,
            reference,
            options,
            cycle_slip,
        }
    }
}

/// Output assembled by [`fix_wide_lane_rtk_arc`] from wide-lane estimates and
/// the dual-frequency epochs used to obtain them.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkWideLaneArcSolution {
    /// Geometry observability and covariance-validation diagnostics for the
    /// wide-lane ambiguity-estimation design. Each fixed ambiguity is one
    /// parameter and each usable ambiguity sample is one row. Snapshot wide-lane
    /// solves pass no propagated prior into geometry classification, so
    /// `ZeroRedundancy` bounds are unvalidated and `Weak` bounds are unclamped.
    /// Sequential filter epochs use the same instantaneous design diagnostics,
    /// but a full-rank propagated prior can validate zero-redundancy covariance.
    pub geometry_quality: GeometryQuality,
    /// Per-constellation physical reference satellite tokens.
    pub references: BTreeMap<String, String>,
    /// Integer wide-lane cycle estimates accumulated across the selected systems.
    pub wide_lane_cycles: BTreeMap<String, i64>,
    /// Input epochs after optional dual cycle-slip preparation.
    pub epochs: Vec<RtkDualFrequencyArcEpoch>,
    /// Satellites removed by dual cycle-slip preprocessing under its drop policy.
    pub dropped_sats: Vec<String>,
    /// Split-arc records produced by dual cycle-slip preprocessing.
    pub split_cycle_slip_arcs: Vec<CycleSlipSplitArc>,
}

/// Errors that [`fix_wide_lane_rtk_arc`] can return while preparing or estimating
/// a dual-frequency wide-lane arc.
#[derive(Debug, Clone, PartialEq)]
pub enum RtkWideLaneArcError {
    /// `fix_wide_lane_rtk_arc` found an empty epoch slice before preprocessing.
    EmptyEpochs,
    /// `dual_arc_references` failed to select reference satellites.
    Reference(DoubleDifferenceError),
    /// `prepare_dual_frequency_arc` failed during dual cycle-slip preprocessing.
    CycleSlipPrep(CycleSlipPrepError),
    /// `estimate_wide_lane_ambiguities` failed for a selected constellation.
    WideLane(WideLaneError),
}

impl core::fmt::Display for RtkWideLaneArcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyEpochs => write!(f, "RTK wide-lane arc requires at least one epoch"),
            Self::Reference(error) => write!(f, "RTK wide-lane reference failed: {error:?}"),
            Self::CycleSlipPrep(error) => {
                write!(f, "RTK wide-lane cycle-slip prep failed: {error:?}")
            }
            Self::WideLane(error) => write!(f, "RTK wide-lane fixing failed: {error:?}"),
        }
    }
}

impl std::error::Error for RtkWideLaneArcError {}

/// Settings consumed by [`prepare_ionosphere_free_rtk_arc`] when converting
/// dual-frequency epochs to ionosphere-free observations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RtkIonosphereFreeArcConfig {
    /// Base-station ECEF position in meters, passed to reference selection and
    /// ionosphere-free preparation.
    pub base_m: [f64; 3],
    /// Initial rover-minus-base ECEF baseline guess in meters, passed to each
    /// system's ionosphere-free setup.
    pub initial_baseline_m: [f64; 3],
    /// Policy passed to dual reference selection for each constellation.
    pub reference: BaselineReferenceSelection,
    /// Flag passed to the standalone preparation to select its troposphere
    /// correction branch.
    pub apply_troposphere: bool,
}

impl RtkIonosphereFreeArcConfig {
    /// Build ionosphere-free settings from both baseline states and the
    /// reference/troposphere policies.
    #[must_use]
    pub const fn new(
        base_m: [f64; 3],
        initial_baseline_m: [f64; 3],
        reference: BaselineReferenceSelection,
        apply_troposphere: bool,
    ) -> Self {
        Self {
            base_m,
            initial_baseline_m,
            reference,
            apply_troposphere,
        }
    }
}

/// Output assembled by [`prepare_ionosphere_free_rtk_arc`], including the
/// ionosphere-free epochs and ambiguity scales needed by the final RTK solve.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkIonosphereFreeArcSolution {
    /// Per-constellation physical reference satellite tokens returned by dual
    /// reference selection.
    pub references: BTreeMap<String, String>,
    /// Single-frequency ionosphere-free epochs merged by original input index
    /// from each system's setup result.
    pub epochs: Vec<RtkArcEpoch>,
    /// Wavelength scale per resulting single-difference ambiguity id, in meters,
    /// returned for the final fixed solve. Includes each reference carrier for
    /// use after a later reference change.
    pub wavelengths_m: BTreeMap<String, f64>,
    /// Code-to-phase offset scale per resulting single-difference ambiguity id,
    /// in meters, relative to the selected arc reference, returned for the final
    /// fixed solve. Includes zero for each reference carrier.
    pub offsets_m: BTreeMap<String, f64>,
}

/// Selects the final static or sequential solve matched by
/// [`solve_wide_lane_fixed_rtk_arc`].
#[derive(Debug, Clone, PartialEq)]
pub enum RtkWideLaneFixedArcSolveConfig {
    /// When selected, the combined driver calls `solve_static_rtk_arc` after
    /// injecting the ionosphere-free reference and scale maps.
    Static(RtkStaticArcConfig),
    /// When selected, the combined driver calls `solve_rtk_arc` after injecting
    /// the ionosphere-free reference and scale maps.
    Sequential(RtkArcConfig),
}

/// Configuration consumed by [`solve_wide_lane_fixed_rtk_arc`] for its
/// wide-lane, ionosphere-free, and final RTK stages.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RtkWideLaneFixedArcConfig {
    /// Settings consumed by the initial wide-lane preparation and estimation.
    pub wide_lane: RtkWideLaneArcConfig,
    /// Settings consumed by ionosphere-free conversion of the prepared epochs.
    pub ionosphere_free: RtkIonosphereFreeArcConfig,
    /// Settings and branch selection for the final static or sequential solve.
    pub solve: RtkWideLaneFixedArcSolveConfig,
}

impl RtkWideLaneFixedArcConfig {
    /// Build the combined wide-lane, ionosphere-free, and final-solve settings.
    #[must_use]
    pub const fn new(
        wide_lane: RtkWideLaneArcConfig,
        ionosphere_free: RtkIonosphereFreeArcConfig,
        solve: RtkWideLaneFixedArcSolveConfig,
    ) -> Self {
        Self {
            wide_lane,
            ionosphere_free,
            solve,
        }
    }
}

/// Identifies the branch recorded by `wide_lane_fixed_metadata` after a combined
/// RTK arc solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtkWideLaneFixedArcIntegerMethod {
    /// `solve_wide_lane_fixed_rtk_arc` selected the static branch, whose
    /// validated LAMBDA solve consumes the wide-lane scales.
    WideLaneNarrowLaneLambda,
    /// `solve_wide_lane_fixed_rtk_arc` selected the sequential branch, whose
    /// per-epoch ambiguity search consumes the wide-lane scales.
    WideLaneNarrowLaneSequential,
}

/// Metadata assembled by `wide_lane_fixed_metadata` to connect wide-lane fixing
/// with the final RTK arc result.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkWideLaneFixedArcMetadata {
    /// Final path marker set by `wide_lane_fixed_metadata` from the selected
    /// static or sequential branch.
    pub integer_method: RtkWideLaneFixedArcIntegerMethod,
    /// Set to `true` by `wide_lane_fixed_metadata` after wide-lane fixing has
    /// succeeded.
    pub wide_lane_fixed: bool,
    /// Wide-lane integer estimates filtered to satellites used by the final
    /// static ambiguity index or sequential epoch solutions.
    pub wide_lane_ambiguities_cycles: BTreeMap<String, i64>,
    /// Dropped-satellite list copied from the wide-lane stage's cycle-slip
    /// preparation.
    pub dropped_cycle_slip_sats: Vec<String>,
    /// Split-arc list copied from the wide-lane stage's cycle-slip preparation.
    pub split_cycle_slip_arcs: Vec<CycleSlipSplitArc>,
}

/// Static branch assembled by [`solve_wide_lane_fixed_rtk_arc`] from its wide-
/// lane, ionosphere-free, and final static results.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkWideLaneFixedStaticArcSolution {
    /// Wide-lane stage result used to derive the combined metadata.
    pub wide_lane: RtkWideLaneArcSolution,
    /// Ionosphere-free stage result whose epochs and scales feed the static solve.
    pub ionosphere_free: RtkIonosphereFreeArcSolution,
    /// Final static batch result after reference and scale injection.
    pub solution: RtkStaticArcSolution,
    /// Metadata identifying the static combined path and its wide-lane outcomes.
    pub metadata: RtkWideLaneFixedArcMetadata,
}

/// Sequential branch assembled by [`solve_wide_lane_fixed_rtk_arc`] from its
/// wide-lane, ionosphere-free, and final sequential results.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkWideLaneFixedSequentialArcSolution {
    /// Wide-lane stage result used to derive the combined metadata.
    pub wide_lane: RtkWideLaneArcSolution,
    /// Ionosphere-free stage result whose epochs and scales feed the sequential
    /// solve.
    pub ionosphere_free: RtkIonosphereFreeArcSolution,
    /// Final sequential arc result after reference and scale injection.
    pub solution: RtkArcSolution,
    /// Metadata identifying the sequential combined path and its wide-lane
    /// outcomes.
    pub metadata: RtkWideLaneFixedArcMetadata,
}

// Returned by value once per solve, never held in bulk, so the static-vs-sequential
// size difference is immaterial; boxing a variant would churn the public + binding API.
#[allow(clippy::large_enum_variant)]
/// Result returned by [`solve_wide_lane_fixed_rtk_arc`] after its wide-lane and
/// ionosphere-free stages and selected final solve.
#[derive(Debug, Clone, PartialEq)]
pub enum RtkWideLaneFixedArcSolution {
    /// The selected final solver was the static batch path.
    Static(RtkWideLaneFixedStaticArcSolution),
    /// The selected final solver was the sequential arc path.
    Sequential(RtkWideLaneFixedSequentialArcSolution),
}

/// Errors returned by [`solve_wide_lane_fixed_rtk_arc`] from its combined
/// wide-lane, ionosphere-free, or final RTK stages.
#[derive(Debug, Clone, PartialEq)]
pub enum RtkWideLaneFixedArcError {
    /// Reserved for compatibility with the single-system-only driver. The
    /// combined driver now supports one reference per constellation.
    UnsupportedMultiGnss,
    /// The initial wide-lane stage returned an error.
    WideLane(RtkWideLaneArcError),
    /// Ionosphere-free preparation over the wide-lane epochs returned an error.
    IonosphereFree(RtkIonosphereFreeArcError),
    /// The final static branch returned an `RtkStaticArcError`.
    Static(RtkStaticArcError),
    /// The final sequential branch returned an `RtkArcError`.
    Sequential(RtkArcError),
}

impl core::fmt::Display for RtkWideLaneFixedArcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedMultiGnss => {
                write!(f, "unsupported wide-lane RTK reference configuration")
            }
            Self::WideLane(error) => write!(f, "{error}"),
            Self::IonosphereFree(error) => write!(f, "{error}"),
            Self::Static(error) => write!(f, "{error}"),
            Self::Sequential(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for RtkWideLaneFixedArcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::WideLane(error) => Some(error),
            Self::IonosphereFree(error) => Some(error),
            Self::Static(error) => Some(error),
            Self::Sequential(error) => Some(error),
            Self::UnsupportedMultiGnss => None,
        }
    }
}

/// Errors that [`prepare_ionosphere_free_rtk_arc`] can return while preparing
/// dual-frequency ionosphere-free epochs.
#[derive(Debug, Clone, PartialEq)]
pub enum RtkIonosphereFreeArcError {
    /// `prepare_ionosphere_free_rtk_arc` found an empty epoch slice before
    /// reference selection.
    EmptyEpochs,
    /// `dual_arc_references` failed to select reference satellites.
    Reference(DoubleDifferenceError),
    /// The standalone ionosphere-free preparation failed, including its
    /// `NoEpochs` result when no system produced a usable epoch.
    IonosphereFree(IonosphereFreeBaselineError),
}

impl core::fmt::Display for RtkIonosphereFreeArcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyEpochs => write!(f, "RTK ionosphere-free arc requires at least one epoch"),
            Self::Reference(error) => {
                write!(f, "RTK ionosphere-free reference failed: {error:?}")
            }
            Self::IonosphereFree(error) => {
                write!(f, "RTK ionosphere-free setup failed: {error:?}")
            }
        }
    }
}

impl std::error::Error for RtkIonosphereFreeArcError {}

/// One epoch's reported baseline/ambiguity solution.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkArcEpochSolution {
    /// Ambiguity-conditioned reported baseline for this epoch (metres).
    pub reported_baseline_m: [f64; 3],
    /// Carried float (Kalman posterior) baseline after this epoch (metres).
    pub float_baseline_m: [f64; 3],
    /// Whether any integer ambiguity is held after this epoch.
    pub integer_fixed: bool,
    /// Integer ratio from this epoch's ambiguity search (`0.0` = no search ran).
    pub integer_ratio: f64,
    /// Single-difference ambiguity ids newly fixed this epoch.
    pub newly_fixed: Vec<String>,
    /// All held single-difference ambiguity ids after this epoch. A transported
    /// hold can include the original arc reference id.
    pub fixed_ids: Vec<String>,
    /// Holds released at a reference change. Their already accumulated
    /// information, including newly fixed pending constraints, is retained.
    pub released_holds: Vec<ReleasedRtkHold>,
    /// Reported single-difference ambiguities (id, metres) in column order.
    pub sd_ambiguities_m: Vec<(String, f64)>,
    /// Double-difference ambiguity ids fixed this epoch, against each one's own
    /// system reference (the reporting form used by RTKLIB-style consumers).
    pub fixed_double_difference_ids: Vec<String>,
    /// Satellites used this epoch (intersection availability), sorted.
    pub used_satellite_ids: Vec<String>,
    /// Physical reference satellites selected after this epoch's rover-side
    /// elevation screen, sorted by constellation.
    pub reference_satellite_ids: Vec<String>,
    /// Satellites rejected by the rover-side elevation mask at this epoch,
    /// sorted. Screening starts at the predicted filter position for every
    /// dynamics mode and repeats if an updated rover estimate changes the
    /// accepted set.
    pub elevation_masked_satellite_ids: Vec<String>,
    /// Whether the rover-mask satellite and reference sets stabilized at the
    /// updated position. True when masking is disabled. False means the
    /// iteration cap was reached and this is the last completed, internally
    /// consistent solve, with its own mask rejection records.
    pub elevation_mask_converged: bool,
    /// LAMBDA search diagnostics, if a search ran.
    pub search: Option<IntegerSearchMeta>,
    /// Public residual rows at the reported solution (when enabled in opts).
    pub residuals: Vec<FloatResidual>,
    /// Geometry observability and covariance-validation diagnostics for this
    /// epoch's instantaneous double-difference design. Snapshot solves flag
    /// zero-redundancy covariance as unvalidated; sequential filters validate it
    /// only after a full-rank propagated prior is carried into the epoch.
    pub geometry_quality: GeometryQuality,
}

/// Why a reference change cannot preserve a held integer coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtkHoldReleaseReason {
    /// The new reference has no held integer from which to form differences.
    UnheldReference,
    /// The constellation contains different carrier wavelengths (for example
    /// FDMA), so metre differences are not one common-wavelength integer lattice.
    IncompatibleWavelengths,
}

/// A held ambiguity released rather than silently assigned a different integer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReleasedRtkHold {
    /// Single-difference ambiguity identifier of the released hold.
    pub ambiguity_id: String,
    /// Reason its integer coordinate could not be transported.
    pub reason: RtkHoldReleaseReason,
}

/// Full sequential RTK arc solution.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkArcSolution {
    /// Per-constellation reference single-difference ambiguity ids.
    pub references: BTreeMap<String, String>,
    /// Per-epoch reported solutions, in input order.
    pub epochs: Vec<RtkArcEpochSolution>,
    /// Final carried filter state after the last epoch.
    pub final_state: FilterState,
    /// Satellites dropped during cycle-slip preprocessing under
    /// [`CycleSlipPolicy::DropSatellite`], sorted. Empty when cycle-slip
    /// preprocessing is disabled or no slip occurred.
    pub dropped_sats: Vec<String>,
    /// Split-arc metadata produced by cycle-slip preprocessing under
    /// [`CycleSlipPolicy::SplitArc`]. Empty otherwise.
    pub split_cycle_slip_arcs: Vec<CycleSlipSplitArc>,
    /// Satellites masked below the elevation mask in any epoch, sorted. Empty
    /// when elevation masking is disabled.
    pub elevation_masked_sats: Vec<String>,
    /// Posterior measurement covariance (row-major `n x n`, metres squared): the
    /// inverse of [`Self::final_state`]'s information matrix. Empty only if that
    /// inversion is singular.
    pub measurement_covariance: Vec<f64>,
}

/// Why the sequential RTK arc driver could not complete.
#[derive(Debug, Clone, PartialEq)]
pub enum RtkArcError {
    /// The arc has no epochs.
    EmptyEpochs,
    /// Fewer than four satellites appear across the whole arc.
    TooFewSatellites {
        /// Number of distinct satellites in the normalized arc availability union.
        count: usize,
        /// Required number of distinct arc satellites; currently four.
        minimum: usize,
    },
    /// Reference-satellite selection failed.
    Reference(DoubleDifferenceError),
    /// The constructed initial filter state was invalid.
    FilterState(FilterStateValidationError),
    /// A per-epoch sequential update failed.
    Update {
        /// Zero-based index of the epoch whose update failed.
        epoch_index: usize,
        /// Error returned by the sequential filter for that epoch.
        source: UpdateError,
    },
    /// A strict (velocity-propagated) run hit a missing/incomparable epoch time.
    InvalidEpochTime {
        /// Zero-based index of the non-first epoch without comparable time data.
        epoch_index: usize,
    },
    /// A satellite in the availability set is missing a required position.
    MissingPosition {
        /// Zero-based index of the epoch being assembled.
        epoch_index: usize,
        /// Satellite token whose shared or receiver-specific position is missing.
        satellite_id: String,
    },
    /// Cycle-slip preprocessing rejected the input or detected a slip under
    /// [`CycleSlipPolicy::Error`].
    CycleSlipPrep(CycleSlipPrepError),
    /// Hatch code-smoothing preprocessing rejected the input.
    CodeSmoothing(CodeSmoothingError),
    /// Elevation-mask preprocessing rejected the input geometry.
    ElevationMask(DoubleDifferenceError),
}

impl core::fmt::Display for RtkArcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyEpochs => write!(f, "RTK arc requires at least one epoch"),
            Self::TooFewSatellites { count, minimum } => write!(
                f,
                "RTK arc has {count} common satellites; need at least {minimum}"
            ),
            Self::Reference(error) => write!(f, "RTK reference selection failed: {error:?}"),
            Self::FilterState(error) => write!(f, "{error}"),
            Self::Update {
                epoch_index,
                source,
            } => write!(f, "RTK arc update failed at epoch {epoch_index}: {source}"),
            Self::InvalidEpochTime { epoch_index } => {
                write!(f, "RTK arc epoch {epoch_index} has no comparable time")
            }
            Self::MissingPosition {
                epoch_index,
                satellite_id,
            } => write!(
                f,
                "RTK arc epoch {epoch_index} is missing a position for {satellite_id}"
            ),
            Self::CycleSlipPrep(error) => {
                write!(f, "RTK arc cycle-slip preprocessing failed: {error:?}")
            }
            Self::CodeSmoothing(error) => {
                write!(f, "RTK arc code-smoothing preprocessing failed: {error:?}")
            }
            Self::ElevationMask(error) => {
                write!(f, "RTK arc elevation-mask preprocessing failed: {error:?}")
            }
        }
    }
}

impl std::error::Error for RtkArcError {}

/// A normalized epoch: per-satellite paired observations plus the three position
/// maps (per-receiver maps already defaulted to the shared map).
struct NormalizedEpoch<'a> {
    paired: BTreeMap<&'a str, (&'a RtkArcObservation, &'a RtkArcObservation)>,
    shared_positions: &'a BTreeMap<String, [f64; 3]>,
    base_positions: &'a BTreeMap<String, [f64; 3]>,
    rover_positions: &'a BTreeMap<String, [f64; 3]>,
    available: Vec<String>,
    velocity_mps: Option<[f64; 3]>,
}

/// The three core (numeric) outputs of the sequential filter, before the
/// preprocessing/metadata wrapper attaches its diagnostics.
struct PreparedSolution {
    references: BTreeMap<String, String>,
    epochs: Vec<RtkArcEpochSolution>,
    final_state: FilterState,
    elevation_masked_sats: Vec<String>,
}

struct BatchArc {
    references: BTreeMap<String, String>,
    epochs: Vec<Epoch>,
    ambiguity_ids: Vec<String>,
    ambiguity_satellites: BTreeMap<String, String>,
}

#[derive(Debug, Clone)]
struct MergedIonosphereFreeEpoch {
    satellite_ids: Vec<String>,
    base_observations: Vec<CoreRtkObservation>,
    rover_observations: Vec<CoreRtkObservation>,
}

type PreparedDualFrequencyArc = (
    Vec<RtkDualFrequencyArcEpoch>,
    Vec<String>,
    Vec<CycleSlipSplitArc>,
);

/// Solve a sequential RTK baseline arc from raw rover+base epochs.
///
/// Returns one reported baseline/ambiguity solution per input epoch plus the
/// final carried filter state, and the per-solve preprocessing/covariance
/// metadata. The reference satellites are selected once for the whole arc; an
/// epoch whose update fails aborts the arc at that epoch.
///
/// With the default [`RtkArcConfig::preprocessing`] (all stages off) this is the
/// bare core solve: the metadata `dropped_sats`/`split_cycle_slip_arcs`/
/// `elevation_masked_sats` are empty and the numeric outputs are unchanged from a
/// driver that never had preprocessing. When stages are enabled the driver runs
/// them (delegating to the standalone `crate::rtk` functions) before the solve,
/// so a thin binding can hand over the whole workflow.
pub fn solve_rtk_arc(
    epochs: &[RtkArcEpoch],
    config: &RtkArcConfig,
) -> Result<RtkArcSolution, RtkArcError> {
    if epochs.is_empty() {
        return Err(RtkArcError::EmptyEpochs);
    }

    let (prepared_epochs, dropped_sats, split_cycle_slip_arcs, elevation_masked_sats) =
        if config.preprocessing.is_active() {
            preprocess_arc(epochs, config)?
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
    let solve_input: &[RtkArcEpoch] = if config.preprocessing.is_active() {
        &prepared_epochs
    } else {
        epochs
    };

    let core = solve_prepared_arc(solve_input, config)?;
    let measurement_covariance = posterior_covariance(&core.final_state);
    let mut all_elevation_masked: BTreeSet<_> = elevation_masked_sats.into_iter().collect();
    all_elevation_masked.extend(core.elevation_masked_sats.iter().cloned());

    Ok(RtkArcSolution {
        references: core.references,
        epochs: core.epochs,
        final_state: core.final_state,
        dropped_sats,
        split_cycle_slip_arcs,
        elevation_masked_sats: all_elevation_masked.into_iter().collect(),
        measurement_covariance,
    })
}

/// Solve a static RTK baseline over a raw arc as one batch.
///
/// Enabled preprocessing runs before batch construction. The function then
/// derives the ambiguity columns and scale maps, calls the float and validated
/// fixed baseline solvers, and returns their results with preprocessing metadata.
/// The static-station wrapper uses this path after filling the arc scale maps
/// from RINEX. With a rover elevation mask, the batch solve and mask are
/// iterated at most [`super::FloatSolveOpts::max_iterations`] times. If the satellite
/// set has not stabilized then, the last completed solve is returned,
/// [`RtkStaticArcSolution::elevation_mask_converged`] is `false`, and
/// `elevation_masked_sats` describes only the observations excluded from that
/// solve. Returning the completed solve rather than the unsolved next mask keeps
/// its state, residuals, ambiguity columns, and rejection metadata consistent.
///
/// An empty epoch slice is returned as [`RtkStaticArcError::Arc`]. Other
/// preparation, float, and fixed failures retain their corresponding variants.
pub fn solve_static_rtk_arc(
    epochs: &[RtkArcEpoch],
    config: &RtkStaticArcConfig,
) -> Result<RtkStaticArcSolution, RtkStaticArcError> {
    if epochs.is_empty() {
        return Err(RtkArcError::EmptyEpochs.into());
    }

    let (prepared_epochs, dropped_sats, split_cycle_slip_arcs, elevation_masked_sats) =
        if config.arc.preprocessing.is_active() {
            preprocess_arc(epochs, &config.arc)?
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
    let solve_input: &[RtkArcEpoch] = if config.arc.preprocessing.is_active() {
        &prepared_epochs
    } else {
        epochs
    };
    let antenna = config.arc.update_opts.receiver_antenna_corrections.as_ref();
    let mut rover_masked_sats = BTreeSet::new();
    let mut static_input = solve_input.to_vec();
    let mut baseline_seed = config.arc.initial_baseline_m;
    let maximum_mask_iterations = config.opts.float.max_iterations.max(1);
    let mut mask_iteration = 0;
    let mut elevation_mask_converged = config.arc.preprocessing.elevation_mask_deg.is_none();
    let (batch, float_solution) = 'mask_iterations: loop {
        let batch = build_static_batch_arc(&static_input, &config.arc)?;
        let float_solution = solve_float_baseline(
            &batch.epochs,
            config.arc.base_m,
            &batch.ambiguity_ids,
            baseline_seed,
            &config.arc.model,
            config.opts.float,
            antenna,
        )
        .map_err(RtkStaticArcError::Float)?;
        let Some(mask_deg) = config.arc.preprocessing.elevation_mask_deg else {
            break 'mask_iterations (batch, float_solution);
        };
        let rover_m =
            std::array::from_fn(|axis| config.arc.base_m[axis] + float_solution.baseline_m[axis]);
        let (next_input, masked) = mask_static_rover_epochs(solve_input, rover_m, mask_deg)?;
        let next_masked: BTreeSet<_> = masked.into_iter().collect();
        if same_epoch_satellite_sets(&static_input, &next_input) {
            rover_masked_sats = next_masked;
            elevation_mask_converged = true;
            break 'mask_iterations (batch, float_solution);
        }
        if mask_iteration + 1 == maximum_mask_iterations {
            break 'mask_iterations (batch, float_solution);
        }
        static_input = next_input;
        rover_masked_sats = next_masked;
        baseline_seed = float_solution.baseline_m;
        mask_iteration += 1;
    };
    let (wavelengths_m, offsets_m) = expanded_batch_scale_maps(&batch, &config.arc);
    let fixed_solution = solve_fixed_baseline_validated(
        &batch.epochs,
        config.arc.base_m,
        AmbiguitySet {
            ids: &batch.ambiguity_ids,
            satellites: &batch.ambiguity_satellites,
            scale: AmbiguityScale {
                wavelengths_m: &wavelengths_m,
                offsets_m: &offsets_m,
            },
            float_only_systems: &config.arc.update_opts.float_only_systems,
        },
        config.arc.initial_baseline_m,
        &config.arc.model,
        config.opts,
        antenna,
    )
    .map_err(RtkStaticArcError::Fixed)?;

    Ok(RtkStaticArcSolution {
        geometry_quality: float_solution.geometry_quality,
        references: batch.references,
        ambiguity_ids: batch.ambiguity_ids,
        ambiguity_satellites: batch.ambiguity_satellites,
        float_solution,
        fixed_solution,
        dropped_sats,
        split_cycle_slip_arcs,
        elevation_masked_sats: elevation_masked_sats
            .into_iter()
            .chain(rover_masked_sats)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect(),
        elevation_mask_converged,
    })
}

fn mask_static_rover_epochs(
    epochs: &[RtkArcEpoch],
    rover_m: [f64; 3],
    mask_deg: f64,
) -> Result<(Vec<RtkArcEpoch>, Vec<String>), RtkStaticArcError> {
    let masks: Vec<_> = epochs
        .iter()
        .map(|epoch| ElevationMaskEpoch {
            satellite_positions_m: if epoch.rover_satellite_positions_m.is_empty() {
                epoch.satellite_positions_m.clone()
            } else {
                epoch.rover_satellite_positions_m.clone()
            },
        })
        .collect();
    let result = apply_elevation_mask(rover_m, &masks, mask_deg)
        .map_err(RtkArcError::ElevationMask)
        .map_err(RtkStaticArcError::Arc)?;
    let filtered = epochs
        .iter()
        .zip(&result.epochs)
        .map(|(epoch, mask)| thin_epoch_to_kept(epoch, &mask.kept_satellite_ids))
        .collect();
    Ok((filtered, result.masked_satellite_ids))
}

fn same_epoch_satellite_sets(left: &[RtkArcEpoch], right: &[RtkArcEpoch]) -> bool {
    left.iter().zip(right).all(|(left, right)| {
        left.rover
            .iter()
            .map(|obs| obs.satellite_id.as_str())
            .eq(right.rover.iter().map(|obs| obs.satellite_id.as_str()))
    })
}

/// Estimate wide-lane integer ambiguities across a dual-frequency RTK arc.
///
/// The function optionally applies dual-frequency cycle-slip preparation, selects
/// one reference satellite per constellation, runs the standalone wide-lane
/// estimator for each selected system, and returns the prepared epochs together
/// with geometry and cycle-slip metadata. The estimate map is accumulated across
/// the systems in deterministic map order.
///
/// An empty epoch slice returns [`RtkWideLaneArcError::EmptyEpochs`]. Failures in
/// reference selection, cycle-slip preparation, or wide-lane estimation are
/// wrapped in the matching error variant.
pub fn fix_wide_lane_rtk_arc(
    epochs: &[RtkDualFrequencyArcEpoch],
    config: &RtkWideLaneArcConfig,
) -> Result<RtkWideLaneArcSolution, RtkWideLaneArcError> {
    if epochs.is_empty() {
        return Err(RtkWideLaneArcError::EmptyEpochs);
    }

    let (prepared_epochs, dropped_sats, split_cycle_slip_arcs) =
        prepare_dual_frequency_arc(epochs, config.cycle_slip)?;
    let references = dual_arc_references(config.base_m, &prepared_epochs, config.reference.clone())
        .map_err(RtkWideLaneArcError::Reference)?;
    let mut wide_lane_cycles = BTreeMap::new();

    for (system, reference_satellite_id) in &references {
        let system_epochs = dual_epochs_for_system(&prepared_epochs, system);
        let fixed =
            estimate_wide_lane_ambiguities(&system_epochs, reference_satellite_id, config.options)
                .map_err(RtkWideLaneArcError::WideLane)?;
        wide_lane_cycles.extend(fixed);
    }
    let geometry_quality =
        wide_lane_geometry_quality(&prepared_epochs, &references, &wide_lane_cycles);

    Ok(RtkWideLaneArcSolution {
        geometry_quality,
        references,
        wide_lane_cycles,
        epochs: prepared_epochs,
        dropped_sats,
        split_cycle_slip_arcs,
    })
}

/// Convert a dual-frequency RTK arc to ionosphere-free single-frequency epochs.
///
/// References are selected per constellation, the standalone ionosphere-free
/// preparation runs for each system with the supplied wide-lane integers, and
/// the resulting records are merged by their original epoch index. The returned
/// wavelength and offset maps are intended for the subsequent fixed RTK solve.
///
/// Empty input returns [`RtkIonosphereFreeArcError::EmptyEpochs`]; if no system
/// produces an epoch, the function returns its `IonosphereFree` variant with
/// [`IonosphereFreeBaselineError::NoEpochs`].
pub fn prepare_ionosphere_free_rtk_arc(
    epochs: &[RtkDualFrequencyArcEpoch],
    wide_lane_cycles: &BTreeMap<String, i64>,
    config: &RtkIonosphereFreeArcConfig,
) -> Result<RtkIonosphereFreeArcSolution, RtkIonosphereFreeArcError> {
    if epochs.is_empty() {
        return Err(RtkIonosphereFreeArcError::EmptyEpochs);
    }

    let references = dual_arc_references(config.base_m, epochs, config.reference.clone())
        .map_err(RtkIonosphereFreeArcError::Reference)?;
    let mut merged_epochs = BTreeMap::<usize, MergedIonosphereFreeEpoch>::new();
    let mut wavelengths_m = BTreeMap::new();
    let mut offsets_m = BTreeMap::new();

    for (system, reference_satellite_id) in &references {
        let system_setup_epochs = dual_setup_epochs_for_system(epochs, system);
        let setup_epochs = system_setup_epochs
            .iter()
            .map(|(_, epoch)| epoch.clone())
            .collect::<Vec<_>>();
        let result = prepare_ionosphere_free_baseline_epochs(
            config.base_m,
            config.initial_baseline_m,
            &setup_epochs,
            reference_satellite_id,
            wide_lane_cycles,
            config.apply_troposphere,
        )
        .map_err(RtkIonosphereFreeArcError::IonosphereFree)?;

        for epoch in result.epochs {
            let original_index = system_setup_epochs[epoch.epoch_index].0;
            merge_ionosphere_free_epoch(&mut merged_epochs, original_index, epoch);
        }
        wavelengths_m.extend(result.wavelengths_m);
        offsets_m.extend(result.offsets_m);
    }

    if merged_epochs.is_empty() {
        return Err(RtkIonosphereFreeArcError::IonosphereFree(
            IonosphereFreeBaselineError::NoEpochs,
        ));
    }

    let epochs = merged_epochs
        .into_iter()
        .map(|(index, mut epoch)| {
            qualify_reference_dependent_ambiguities(&mut epoch, &references);
            ionosphere_free_arc_epoch(&epochs[index], epoch)
        })
        .collect();

    Ok(RtkIonosphereFreeArcSolution {
        references,
        epochs,
        wavelengths_m,
        offsets_m,
    })
}

/// Run the complete wide-lane, ionosphere-free, and final RTK arc pipeline.
///
/// After wide-lane fixing and ionosphere-free conversion, the function injects
/// the selected references (one per constellation)
/// and the derived wavelength/offset maps into either the configured static or
/// sequential solve, then returns the corresponding composite result.
///
/// Failures from each pipeline stage retain their matching error variant.
pub fn solve_wide_lane_fixed_rtk_arc(
    epochs: &[RtkDualFrequencyArcEpoch],
    config: &RtkWideLaneFixedArcConfig,
) -> Result<RtkWideLaneFixedArcSolution, RtkWideLaneFixedArcError> {
    let wide_lane = fix_wide_lane_rtk_arc(epochs, &config.wide_lane)
        .map_err(RtkWideLaneFixedArcError::WideLane)?;
    let ionosphere_free = prepare_ionosphere_free_rtk_arc(
        &wide_lane.epochs,
        &wide_lane.wide_lane_cycles,
        &config.ionosphere_free,
    )
    .map_err(RtkWideLaneFixedArcError::IonosphereFree)?;
    let reference = if ionosphere_free.references.len() == 1 {
        BaselineReferenceSelection::Satellite(
            ionosphere_free
                .references
                .values()
                .next()
                .cloned()
                .ok_or(RtkWideLaneFixedArcError::UnsupportedMultiGnss)?,
        )
    } else {
        BaselineReferenceSelection::PerSystem(ionosphere_free.references.clone())
    };

    match &config.solve {
        RtkWideLaneFixedArcSolveConfig::Static(static_config) => {
            let mut solve_config = static_config.clone();
            solve_config.arc.reference = reference;
            solve_config.arc.wavelengths_m = ionosphere_free.wavelengths_m.clone();
            solve_config.arc.offsets_m = ionosphere_free.offsets_m.clone();
            // Static batch ambiguities never include their reference carrier.
            // Keep its additive preparation metadata out of target-only search
            // diagnostics, preserving the existing static result exactly.
            for reference in ionosphere_free.references.values() {
                solve_config.arc.wavelengths_m.remove(reference);
                solve_config.arc.offsets_m.remove(reference);
            }
            let solution = solve_static_rtk_arc(&ionosphere_free.epochs, &solve_config)
                .map_err(RtkWideLaneFixedArcError::Static)?;
            let used = static_solution_used_satellites(&solution);
            let metadata = wide_lane_fixed_metadata(
                &wide_lane,
                &used,
                RtkWideLaneFixedArcIntegerMethod::WideLaneNarrowLaneLambda,
            );
            Ok(RtkWideLaneFixedArcSolution::Static(
                RtkWideLaneFixedStaticArcSolution {
                    wide_lane,
                    ionosphere_free,
                    solution,
                    metadata,
                },
            ))
        }
        RtkWideLaneFixedArcSolveConfig::Sequential(arc_config) => {
            let mut solve_config = arc_config.clone();
            solve_config.reference = reference;
            solve_config.wavelengths_m = ionosphere_free.wavelengths_m.clone();
            solve_config.offsets_m = ionosphere_free.offsets_m.clone();
            let solution = solve_rtk_arc(&ionosphere_free.epochs, &solve_config)
                .map_err(RtkWideLaneFixedArcError::Sequential)?;
            let used = sequential_solution_used_satellites(&solution);
            let metadata = wide_lane_fixed_metadata(
                &wide_lane,
                &used,
                RtkWideLaneFixedArcIntegerMethod::WideLaneNarrowLaneSequential,
            );
            Ok(RtkWideLaneFixedArcSolution::Sequential(
                RtkWideLaneFixedSequentialArcSolution {
                    wide_lane,
                    ionosphere_free,
                    solution,
                    metadata,
                },
            ))
        }
    }
}

#[cfg(test)]
fn single_reference_satellite(
    references: &BTreeMap<String, String>,
) -> Result<String, RtkWideLaneFixedArcError> {
    match references.values().next() {
        Some(reference) if references.len() == 1 => Ok(reference.clone()),
        _ => Err(RtkWideLaneFixedArcError::UnsupportedMultiGnss),
    }
}

fn wide_lane_fixed_metadata(
    wide_lane: &RtkWideLaneArcSolution,
    used_satellites: &BTreeSet<String>,
    integer_method: RtkWideLaneFixedArcIntegerMethod,
) -> RtkWideLaneFixedArcMetadata {
    RtkWideLaneFixedArcMetadata {
        integer_method,
        wide_lane_fixed: true,
        wide_lane_ambiguities_cycles: wide_lane
            .wide_lane_cycles
            .iter()
            .filter(|(satellite, _)| used_satellites.contains(*satellite))
            .map(|(satellite, cycles)| (satellite.clone(), *cycles))
            .collect(),
        dropped_cycle_slip_sats: wide_lane.dropped_sats.clone(),
        split_cycle_slip_arcs: wide_lane.split_cycle_slip_arcs.clone(),
    }
}

fn static_solution_used_satellites(solution: &RtkStaticArcSolution) -> BTreeSet<String> {
    solution
        .references
        .values()
        .chain(solution.ambiguity_satellites.values())
        .cloned()
        .collect()
}

fn sequential_solution_used_satellites(solution: &RtkArcSolution) -> BTreeSet<String> {
    solution
        .epochs
        .iter()
        .flat_map(|epoch| epoch.used_satellite_ids.iter().cloned())
        .collect()
}

/// Posterior measurement covariance (row-major, metres squared) from the final
/// filter-state information matrix, via the same crate inverse the LAMBDA search
/// uses. Returns an empty vector if the information matrix is singular.
fn posterior_covariance(state: &FilterState) -> Vec<f64> {
    let n = state.dim();
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|i| state.information[i * n..i * n + n].to_vec())
        .collect();
    invert_matrix_first_tie(&rows)
        .map(|cov| cov.into_iter().flatten().collect())
        .unwrap_or_default()
}

/// Run the enabled preprocessing stages in fixed order (cycle-slip handling, then
/// Hatch code smoothing, then elevation masking), each delegating verbatim to the
/// matching standalone `crate::rtk` function. Returns the prepared epochs plus the
/// dropped-satellite, split-arc, and elevation-masked-satellite metadata.
#[allow(clippy::type_complexity)]
fn preprocess_arc(
    epochs: &[RtkArcEpoch],
    config: &RtkArcConfig,
) -> Result<
    (
        Vec<RtkArcEpoch>,
        Vec<String>,
        Vec<CycleSlipSplitArc>,
        Vec<String>,
    ),
    RtkArcError,
> {
    let pre = &config.preprocessing;
    let mut work = epochs.to_vec();
    let mut dropped_sats = Vec::new();
    let mut split_cycle_slip_arcs = Vec::new();
    let mut elevation_masked_sats = Vec::new();

    if let Some(policy) = pre.cycle_slip {
        let cs_epochs: Vec<CodeSmoothingEpoch> = work.iter().map(to_code_smoothing_epoch).collect();
        let result = prepare_cycle_slip_baseline_epochs(&cs_epochs, policy)
            .map_err(RtkArcError::CycleSlipPrep)?;
        work = apply_prepared_observations(&work, &result.epochs);
        dropped_sats = result.dropped_sats;
        split_cycle_slip_arcs = result.split_arcs;
    }

    if let Some(window_cap) = pre.hatch_window_cap {
        let cs_epochs: Vec<CodeSmoothingEpoch> = work.iter().map(to_code_smoothing_epoch).collect();
        let smoothed = hatch_smooth_baseline_code_epochs(&cs_epochs, window_cap)
            .map_err(RtkArcError::CodeSmoothing)?;
        work = apply_prepared_observations(&work, &smoothed);
    }

    if let Some(mask_deg) = pre.elevation_mask_deg {
        // The fixed base is known here. The rover mask is applied after each
        // filter prediction, where the current rover state exists.
        let base_mask_epochs: Vec<ElevationMaskEpoch> = work
            .iter()
            .map(|epoch| ElevationMaskEpoch {
                satellite_positions_m: base_positions(epoch).clone(),
            })
            .collect();
        let base_result = apply_elevation_mask(config.base_m, &base_mask_epochs, mask_deg)
            .map_err(RtkArcError::ElevationMask)?;
        work = work
            .iter()
            .zip(base_result.epochs.iter())
            .map(|(epoch, base)| thin_epoch_to_kept(epoch, &base.kept_satellite_ids))
            .collect();
        elevation_masked_sats = base_result.masked_satellite_ids;
    }

    Ok((
        work,
        dropped_sats,
        split_cycle_slip_arcs,
        elevation_masked_sats,
    ))
}

fn build_static_batch_arc(
    epochs: &[RtkArcEpoch],
    config: &RtkArcConfig,
) -> Result<BatchArc, RtkStaticArcError> {
    let (references, batch_epochs) =
        build_static_batch_epochs(epochs, config).map_err(RtkStaticArcError::Arc)?;
    let (ambiguity_ids, ambiguity_satellites) =
        baseline_ambiguity_index_core(&batch_epochs).map_err(RtkStaticArcError::Fixed)?;
    Ok(BatchArc {
        references,
        epochs: batch_epochs,
        ambiguity_ids,
        ambiguity_satellites,
    })
}

fn expanded_sd_scale_maps(
    epochs: &[NormalizedEpoch],
    config: &RtkArcConfig,
) -> (BTreeMap<String, f64>, BTreeMap<String, f64>) {
    let mut wavelengths_m = config.wavelengths_m.clone();
    let mut offsets_m = config.offsets_m.clone();
    for epoch in epochs {
        for (sat, (base, rover)) in &epoch.paired {
            let sd_id = sd_ambiguity_token(sat, &base.ambiguity_id, &rover.ambiguity_id);
            insert_derived_scale(&mut wavelengths_m, &config.wavelengths_m, &sd_id, sat);
            insert_derived_scale(&mut offsets_m, &config.offsets_m, &sd_id, sat);
        }
    }
    (wavelengths_m, offsets_m)
}

fn expanded_batch_scale_maps(
    batch: &BatchArc,
    config: &RtkArcConfig,
) -> (BTreeMap<String, f64>, BTreeMap<String, f64>) {
    let mut wavelengths_m = config.wavelengths_m.clone();
    let mut offsets_m = config.offsets_m.clone();
    for ambiguity_id in &batch.ambiguity_ids {
        let Some(satellite_id) = batch.ambiguity_satellites.get(ambiguity_id) else {
            continue;
        };
        insert_derived_scale(
            &mut wavelengths_m,
            &config.wavelengths_m,
            ambiguity_id,
            satellite_id,
        );
        insert_derived_scale(
            &mut offsets_m,
            &config.offsets_m,
            ambiguity_id,
            satellite_id,
        );
    }
    (wavelengths_m, offsets_m)
}

fn insert_derived_scale(
    target: &mut BTreeMap<String, f64>,
    source: &BTreeMap<String, f64>,
    ambiguity_id: &str,
    satellite_id: &str,
) {
    if target.contains_key(ambiguity_id) {
        return;
    }
    if let Some(value) = scale_value_for(source, ambiguity_id, satellite_id) {
        target.insert(ambiguity_id.to_string(), value);
    }
}

/// The wavelength or offset a source holds for an ambiguity id. The id's own
/// entry comes first; then, for a double-difference id, its single-difference
/// part's; then the entry qualified by a reference, `<id>|ref=<reference>`,
/// which the ionosphere-free preparation keys the ambiguities it derives by;
/// then the same for the carrier arc the id was derived from by splitting or
/// reacquisition. The satellite's entry is taken only where the source holds
/// no entry for that carrier arc in any form, since an entry for another of the
/// satellite's carrier arcs holds another carrier's value.
fn scale_value_for(
    source: &BTreeMap<String, f64>,
    ambiguity_id: &str,
    satellite_id: &str,
) -> Option<f64> {
    let (sd_id, reference) = match ambiguity_id.split_once("|ref=") {
        Some((sd_id, reference)) => (sd_id, Some(reference)),
        None => (ambiguity_id, None),
    };
    let carrier_id = carrier_ambiguity_id(sd_id);
    // A double-difference id over a single difference already qualified by its
    // reference, `<sd>|ref=<reference>|ref=<reference>`, is keyed by the part
    // before its last qualifier.
    let qualified_sd = ambiguity_id
        .rsplit_once("|ref=")
        .map_or(ambiguity_id, |(sd, _)| sd);
    let found = source
        .get(ambiguity_id)
        .or_else(|| source.get(qualified_sd))
        .or_else(|| source.get(sd_id))
        .copied()
        .or_else(|| qualified_scale_value(source, sd_id, reference))
        .or_else(|| source.get(carrier_id).copied())
        .or_else(|| qualified_scale_value(source, carrier_id, reference));
    if found.is_some() {
        return found;
    }
    let carrier_held = source.keys().any(|key| {
        let sd_key = key.split_once("|ref=").map_or(key.as_str(), |(id, _)| id);
        carrier_ambiguity_id(sd_key) == carrier_id
    });
    if carrier_held {
        return None;
    }
    source.get(satellite_id).copied()
}

/// The value a source holds for `id` qualified by a reference. A
/// double-difference query takes the entry for its own reference only. A
/// single-difference query takes the entries for any reference when they hold
/// one value, and none when they differ, as then no one value is its own.
fn qualified_scale_value(
    source: &BTreeMap<String, f64>,
    id: &str,
    reference: Option<&str>,
) -> Option<f64> {
    if let Some(reference) = reference {
        return source.get(&format!("{id}|ref={reference}")).copied();
    }
    let prefix = format!("{id}|ref=");
    let mut values = source
        .iter()
        .filter(|(key, _)| key.starts_with(&prefix))
        .map(|(_, value)| *value);
    let first = values.next()?;
    values.all(|value| value == first).then_some(first)
}

/// The carrier arc a derived single-difference id was made from, whose scale it
/// keeps: the base side of a divergent id, without its reacquisition suffix
/// (`~ra<n>`) and its split suffix (`@<receiver>#<segment>`). A builder names a
/// satellite's carrier arcs `<satellite>` and `<satellite>~freq<n>`, and
/// splitting and reacquisition keep that name at the front of the id.
fn carrier_ambiguity_id(sd_id: &str) -> &str {
    let side = sd_id.split_once(":base=").map_or(sd_id, |(_, sides)| {
        sides.split_once(",rover=").map_or(sides, |(base, _)| base)
    });
    let side = match side.rfind("~ra") {
        Some(index)
            if side.len() > index + 3
                && side[index + 3..].bytes().all(|byte| byte.is_ascii_digit()) =>
        {
            &side[..index]
        }
        _ => side,
    };
    side.rsplit_once('@').map_or(side, |(carrier, _)| carrier)
}
fn build_static_batch_epochs(
    epochs: &[RtkArcEpoch],
    config: &RtkArcConfig,
) -> Result<(BTreeMap<String, String>, Vec<Epoch>), RtkArcError> {
    if epochs.is_empty() {
        return Err(RtkArcError::EmptyEpochs);
    }

    let normalized: Vec<NormalizedEpoch> = epochs.iter().map(normalize_epoch).collect();
    let arc_sats = arc_satellites(&normalized);
    if arc_sats.len() < MINIMUM_ARC_SATELLITES {
        return Err(RtkArcError::TooFewSatellites {
            count: arc_sats.len(),
            minimum: MINIMUM_ARC_SATELLITES,
        });
    }

    let reference_epochs = reference_epoch_terms(&normalized);
    let references_by_system =
        baseline_reference_satellites(config.base_m, &reference_epochs, config.reference.clone())
            .map_err(RtkArcError::Reference)?;
    let reference_sats: BTreeSet<&str> =
        references_by_system.values().map(String::as_str).collect();
    let references = reference_sd_ids(&normalized, &references_by_system);
    let batch_epochs = normalized
        .iter()
        .enumerate()
        .map(|(epoch_index, normalized_epoch)| {
            build_epoch(
                epoch_index,
                normalized_epoch,
                &references_by_system,
                &reference_sats,
                normalized_epoch.velocity_mps,
                0.0,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((references, batch_epochs))
}

fn prepare_dual_frequency_arc(
    epochs: &[RtkDualFrequencyArcEpoch],
    cycle_slip: Option<RtkDualCycleSlipConfig>,
) -> Result<PreparedDualFrequencyArc, RtkWideLaneArcError> {
    let Some(cycle_slip) = cycle_slip else {
        return Ok((epochs.to_vec(), Vec::new(), Vec::new()));
    };
    let dual_cycle_slip_epochs = epochs
        .iter()
        .enumerate()
        .map(|(epoch_index, epoch)| to_dual_cycle_slip_epoch(epoch_index, epoch))
        .collect::<Vec<_>>();
    let result = prepare_dual_cycle_slip_baseline_epochs(
        &dual_cycle_slip_epochs,
        cycle_slip.policy,
        cycle_slip.options,
    )
    .map_err(RtkWideLaneArcError::CycleSlipPrep)?;
    Ok((
        apply_prepared_dual_observations(epochs, &result.epochs),
        result.dropped_sats,
        result.split_arcs,
    ))
}

fn to_dual_cycle_slip_epoch(
    epoch_index: usize,
    epoch: &RtkDualFrequencyArcEpoch,
) -> DualCycleSlipEpoch {
    DualCycleSlipEpoch {
        epoch_sort_key: epoch
            .epoch_sort_key
            .clone()
            .unwrap_or_else(|| format!("{epoch_index:020}")),
        gap_time_s: epoch.gap_time_s,
        gap_epoch: epoch.gap_epoch,
        base_observations: epoch
            .observations
            .iter()
            .map(|observation| {
                to_dual_cycle_slip_observation(&observation.satellite_id, &observation.base)
            })
            .collect(),
        rover_observations: epoch
            .observations
            .iter()
            .map(|observation| {
                to_dual_cycle_slip_observation(&observation.satellite_id, &observation.rover)
            })
            .collect(),
    }
}

fn to_dual_cycle_slip_observation(
    satellite_id: &str,
    observation: &RtkDualFrequencyObservation,
) -> DualCycleSlipObservation {
    DualCycleSlipObservation {
        satellite_id: satellite_id.to_string(),
        ambiguity_id: observation.ambiguity_id.clone(),
        p1_m: observation.p1_m,
        p2_m: observation.p2_m,
        phi1_cycles: observation.phi1_cycles,
        phi2_cycles: observation.phi2_cycles,
        f1_hz: observation.f1_hz,
        f2_hz: observation.f2_hz,
        lli1: observation.lli1,
        lli2: observation.lli2,
    }
}

fn apply_prepared_dual_observations(
    original: &[RtkDualFrequencyArcEpoch],
    prepared: &[DualCycleSlipEpoch],
) -> Vec<RtkDualFrequencyArcEpoch> {
    original
        .iter()
        .zip(prepared.iter())
        .map(|(orig, prep)| {
            let base = prep
                .base_observations
                .iter()
                .map(|obs| (obs.satellite_id.as_str(), obs))
                .collect::<BTreeMap<_, _>>();
            let rover = prep
                .rover_observations
                .iter()
                .map(|obs| (obs.satellite_id.as_str(), obs))
                .collect::<BTreeMap<_, _>>();
            let keep = base
                .keys()
                .filter(|sat| rover.contains_key(*sat))
                .map(|sat| (*sat).to_string())
                .collect::<BTreeSet<_>>();
            let observations = keep
                .iter()
                .map(|sat| RtkDualFrequencySatelliteObservation {
                    satellite_id: sat.clone(),
                    base: from_dual_cycle_slip_observation(base[sat.as_str()]),
                    rover: from_dual_cycle_slip_observation(rover[sat.as_str()]),
                })
                .collect();
            RtkDualFrequencyArcEpoch {
                jd_whole: orig.jd_whole,
                jd_fraction: orig.jd_fraction,
                epoch_sort_key: orig.epoch_sort_key.clone(),
                gap_time_s: orig.gap_time_s,
                gap_epoch: orig.gap_epoch,
                observations,
                satellite_positions_m: retain_map_keys(&orig.satellite_positions_m, &keep),
                base_satellite_positions_m: retain_map_keys(
                    &orig.base_satellite_positions_m,
                    &keep,
                ),
                rover_satellite_positions_m: retain_map_keys(
                    &orig.rover_satellite_positions_m,
                    &keep,
                ),
                velocity_mps: orig.velocity_mps,
                prediction_time_s: orig.prediction_time_s,
                prediction_epoch: orig.prediction_epoch,
            }
        })
        .collect()
}

fn from_dual_cycle_slip_observation(
    observation: &DualCycleSlipObservation,
) -> RtkDualFrequencyObservation {
    RtkDualFrequencyObservation {
        ambiguity_id: observation.ambiguity_id.clone(),
        p1_m: observation.p1_m,
        p2_m: observation.p2_m,
        phi1_cycles: observation.phi1_cycles,
        phi2_cycles: observation.phi2_cycles,
        f1_hz: observation.f1_hz,
        f2_hz: observation.f2_hz,
        lli1: observation.lli1,
        lli2: observation.lli2,
    }
}

fn dual_arc_references(
    base_m: [f64; 3],
    epochs: &[RtkDualFrequencyArcEpoch],
    selection: BaselineReferenceSelection,
) -> Result<BTreeMap<String, String>, DoubleDifferenceError> {
    let reference_epochs = epochs
        .iter()
        .map(|epoch| {
            let available = dual_available_satellites(epoch);
            let keep = available.iter().cloned().collect::<BTreeSet<_>>();
            BaselineReferenceEpoch {
                available_satellite_ids: available,
                satellite_positions_m: retain_map_keys(&epoch.satellite_positions_m, &keep),
            }
        })
        .collect::<Vec<_>>();
    baseline_reference_satellites(base_m, &reference_epochs, selection)
}

fn dual_available_satellites(epoch: &RtkDualFrequencyArcEpoch) -> Vec<String> {
    let base_positions = dual_base_positions(epoch);
    let rover_positions = dual_rover_positions(epoch);
    epoch
        .observations
        .iter()
        .filter(|observation| {
            let sat = observation.satellite_id.as_str();
            epoch.satellite_positions_m.contains_key(sat)
                && base_positions.contains_key(sat)
                && rover_positions.contains_key(sat)
        })
        .map(|observation| observation.satellite_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn wide_lane_geometry_quality(
    epochs: &[RtkDualFrequencyArcEpoch],
    references: &BTreeMap<String, String>,
    fixed: &BTreeMap<String, i64>,
) -> GeometryQuality {
    let n_params = fixed.len();
    if n_params == 0 {
        return classify(
            0,
            0,
            0,
            1.0,
            0.0,
            false,
            GeometryQualityThresholds::default(),
        );
    }

    let ambiguity_ids = fixed.keys().collect::<Vec<_>>();
    let mut counts = vec![0_usize; n_params];
    for (system, reference_satellite_id) in references {
        for epoch in epochs {
            let available = dual_available_satellites(epoch)
                .into_iter()
                .collect::<BTreeSet<_>>();
            let Some(reference) = epoch.observations.iter().find(|observation| {
                observation.satellite_id == *reference_satellite_id
                    && available.contains(&observation.satellite_id)
            }) else {
                continue;
            };
            let reference_sd = sd_ambiguity_token(
                &reference.satellite_id,
                &reference.base.ambiguity_id,
                &reference.rover.ambiguity_id,
            );
            for observation in epoch.observations.iter().filter(|observation| {
                observation.satellite_id != *reference_satellite_id
                    && available.contains(&observation.satellite_id)
                    && constellation_letter(&observation.satellite_id) == system
            }) {
                let sat_sd = sd_ambiguity_token(
                    &observation.satellite_id,
                    &observation.base.ambiguity_id,
                    &observation.rover.ambiguity_id,
                );
                let dd = dd_ambiguity_token(
                    &observation.satellite_id,
                    &sat_sd,
                    reference_satellite_id,
                    &reference_sd,
                );
                if let Some(index) = ambiguity_ids.iter().position(|id| id.as_str() == dd) {
                    counts[index] += 1;
                }
            }
        }
    }

    let n_obs: usize = counts.iter().sum();
    let rank = counts.iter().filter(|&&count| count > 0).count();
    let condition_number = if rank < n_params {
        f64::INFINITY
    } else {
        let min = counts.iter().copied().min().unwrap_or(0) as f64;
        let max = counts.iter().copied().max().unwrap_or(0) as f64;
        (max / min).sqrt()
    };
    let gdop = if rank < n_params {
        f64::INFINITY
    } else {
        counts
            .iter()
            .map(|&count| 1.0 / count as f64)
            .sum::<f64>()
            .sqrt()
    };

    classify(
        rank,
        n_params,
        n_obs as i32 - n_params as i32,
        condition_number,
        gdop,
        false,
        GeometryQualityThresholds::default(),
    )
}

fn dual_epochs_for_system(epochs: &[RtkDualFrequencyArcEpoch], system: &str) -> Vec<DualEpoch> {
    epochs
        .iter()
        .filter_map(|epoch| {
            let available = dual_available_satellites(epoch)
                .into_iter()
                .collect::<BTreeSet<_>>();
            let observations = epoch
                .observations
                .iter()
                .filter(|observation| available.contains(&observation.satellite_id))
                .filter(|observation| constellation_letter(&observation.satellite_id) == system)
                .map(to_dual_satellite_observation)
                .collect::<Vec<_>>();
            (!observations.is_empty()).then_some(DualEpoch { observations })
        })
        .collect()
}

fn dual_setup_epochs_for_system(
    epochs: &[RtkDualFrequencyArcEpoch],
    system: &str,
) -> Vec<(usize, DualIonosphereFreeSetupEpoch)> {
    epochs
        .iter()
        .enumerate()
        .filter_map(|(index, epoch)| {
            let available = dual_available_satellites(epoch)
                .into_iter()
                .collect::<BTreeSet<_>>();
            let system_observations = epoch
                .observations
                .iter()
                .filter(|observation| available.contains(&observation.satellite_id))
                .filter(|observation| constellation_letter(&observation.satellite_id) == system)
                .map(to_dual_satellite_observation)
                .collect::<Vec<_>>();
            if system_observations.is_empty() {
                return None;
            }
            let keep = system_observations
                .iter()
                .map(|observation| observation.satellite_id.clone())
                .collect::<BTreeSet<_>>();
            Some((
                index,
                DualIonosphereFreeSetupEpoch {
                    jd_whole: epoch.jd_whole,
                    jd_fraction: epoch.jd_fraction,
                    observations: system_observations,
                    base_satellite_positions_m: retain_map_keys(dual_base_positions(epoch), &keep),
                    rover_satellite_positions_m: retain_map_keys(
                        dual_rover_positions(epoch),
                        &keep,
                    ),
                },
            ))
        })
        .collect()
}

fn to_dual_satellite_observation(
    observation: &RtkDualFrequencySatelliteObservation,
) -> DualSatelliteObservation {
    DualSatelliteObservation {
        satellite_id: observation.satellite_id.clone(),
        base: to_dual_observation(&observation.base),
        rover: to_dual_observation(&observation.rover),
    }
}

fn to_dual_observation(observation: &RtkDualFrequencyObservation) -> DualObservation {
    DualObservation {
        ambiguity_id: observation.ambiguity_id.clone(),
        p1_m: observation.p1_m,
        p2_m: observation.p2_m,
        phi1_cycles: observation.phi1_cycles,
        phi2_cycles: observation.phi2_cycles,
        f1_hz: observation.f1_hz,
        f2_hz: observation.f2_hz,
    }
}

/// The base's transmit-time satellite positions, or the shared ones where none are given.
fn base_positions(epoch: &RtkArcEpoch) -> &BTreeMap<String, [f64; 3]> {
    if epoch.base_satellite_positions_m.is_empty() {
        &epoch.satellite_positions_m
    } else {
        &epoch.base_satellite_positions_m
    }
}

fn dual_base_positions(epoch: &RtkDualFrequencyArcEpoch) -> &BTreeMap<String, [f64; 3]> {
    if epoch.base_satellite_positions_m.is_empty() {
        &epoch.satellite_positions_m
    } else {
        &epoch.base_satellite_positions_m
    }
}

fn dual_rover_positions(epoch: &RtkDualFrequencyArcEpoch) -> &BTreeMap<String, [f64; 3]> {
    if epoch.rover_satellite_positions_m.is_empty() {
        &epoch.satellite_positions_m
    } else {
        &epoch.rover_satellite_positions_m
    }
}

fn merge_ionosphere_free_epoch(
    merged: &mut BTreeMap<usize, MergedIonosphereFreeEpoch>,
    original_index: usize,
    epoch: IonosphereFreeBaselineEpoch,
) {
    let entry = merged
        .entry(original_index)
        .or_insert_with(|| MergedIonosphereFreeEpoch {
            satellite_ids: Vec::new(),
            base_observations: Vec::new(),
            rover_observations: Vec::new(),
        });
    entry.satellite_ids.extend(epoch.satellite_ids);
    entry.base_observations.extend(epoch.base_observations);
    entry.rover_observations.extend(epoch.rover_observations);
}

/// Give each non-reference satellite's ionosphere-free ambiguity, at an epoch
/// whose reference single-difference ambiguity is not the reference satellite's
/// first arc, the double-difference id its wavelength and offset are keyed by,
/// `<single difference>|ref=<reference single difference>`.
///
/// An ionosphere-free ambiguity's offset holds the double-difference wide-lane
/// integer, which changes with the reference's own wide-lane ambiguity, so one
/// single-difference ambiguity has another offset on every reference arc. Named
/// by its reference arc, each is an ambiguity of its own, which the solve
/// starts afresh at the reference's change as it starts a satellite's own new
/// arc, and whose offset is the one its reference arc gives. With the reference
/// on its first arc every offset is the one that arc gives, and the ids are left
/// as they are.
fn qualify_reference_dependent_ambiguities(
    epoch: &mut MergedIonosphereFreeEpoch,
    references: &BTreeMap<String, String>,
) {
    for (system, reference) in references {
        let reference_id = |observations: &[CoreRtkObservation]| {
            observations
                .iter()
                .find(|observation| observation.satellite_id == *reference)
                .map(|observation| observation.ambiguity_id.clone())
        };
        let (Some(base_id), Some(rover_id)) = (
            reference_id(&epoch.base_observations),
            reference_id(&epoch.rover_observations),
        ) else {
            continue;
        };
        let reference_sd = sd_ambiguity_token(reference, &base_id, &rover_id);
        if reference_sd == *reference {
            continue;
        }
        let rover_ids: BTreeMap<&str, &str> = epoch
            .rover_observations
            .iter()
            .map(|observation| {
                (
                    observation.satellite_id.as_str(),
                    observation.ambiguity_id.as_str(),
                )
            })
            .collect();
        let mut qualified = BTreeMap::new();
        for base in &epoch.base_observations {
            if base.satellite_id == *reference
                || constellation_letter(&base.satellite_id) != system.as_str()
            {
                continue;
            }
            let Some(rover_id) = rover_ids.get(base.satellite_id.as_str()) else {
                continue;
            };
            let sd = sd_ambiguity_token(&base.satellite_id, &base.ambiguity_id, rover_id);
            qualified.insert(
                base.satellite_id.clone(),
                dd_ambiguity_token(&base.satellite_id, &sd, &reference_sd, reference),
            );
        }
        for observation in epoch
            .base_observations
            .iter_mut()
            .chain(epoch.rover_observations.iter_mut())
        {
            if let Some(id) = qualified.get(&observation.satellite_id) {
                observation.ambiguity_id.clone_from(id);
            }
        }
    }
}

fn ionosphere_free_arc_epoch(
    original: &RtkDualFrequencyArcEpoch,
    epoch: MergedIonosphereFreeEpoch,
) -> RtkArcEpoch {
    let keep = epoch.satellite_ids.iter().cloned().collect::<BTreeSet<_>>();
    RtkArcEpoch {
        base: epoch
            .base_observations
            .iter()
            .map(ionosphere_free_observation_to_arc)
            .collect(),
        rover: epoch
            .rover_observations
            .iter()
            .map(ionosphere_free_observation_to_arc)
            .collect(),
        satellite_positions_m: retain_map_keys(&original.satellite_positions_m, &keep),
        base_satellite_positions_m: retain_map_keys(&original.base_satellite_positions_m, &keep),
        rover_satellite_positions_m: retain_map_keys(&original.rover_satellite_positions_m, &keep),
        velocity_mps: original.velocity_mps,
        prediction_time_s: original.prediction_time_s,
        prediction_epoch: original.prediction_epoch,
    }
}

fn ionosphere_free_observation_to_arc(observation: &CoreRtkObservation) -> RtkArcObservation {
    RtkArcObservation {
        satellite_id: observation.satellite_id.clone(),
        ambiguity_id: observation.ambiguity_id.clone(),
        code_m: observation.code_m,
        phase_m: observation.phase_m,
        lli: None,
    }
}

fn retain_map_keys<T: Clone>(
    input: &BTreeMap<String, T>,
    keep: &BTreeSet<String>,
) -> BTreeMap<String, T> {
    input
        .iter()
        .filter(|(key, _)| keep.contains(*key))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect()
}

/// Convert one arc epoch's paired observations to the single-frequency
/// preprocessing shape (a pure field copy, carrying `lli` for cycle-slip
/// detection and `code_m`/`phase_m` for Hatch smoothing).
fn to_code_smoothing_epoch(epoch: &RtkArcEpoch) -> CodeSmoothingEpoch {
    CodeSmoothingEpoch {
        base_observations: epoch.base.iter().map(to_code_smoothing_obs).collect(),
        rover_observations: epoch.rover.iter().map(to_code_smoothing_obs).collect(),
    }
}

fn to_code_smoothing_obs(obs: &RtkArcObservation) -> CodeSmoothingObservation {
    CodeSmoothingObservation {
        satellite_id: obs.satellite_id.clone(),
        ambiguity_id: obs.ambiguity_id.clone(),
        code_m: obs.code_m,
        phase_m: obs.phase_m,
        lli: obs.lli,
    }
}

fn from_code_smoothing_obs(obs: &CodeSmoothingObservation) -> RtkArcObservation {
    RtkArcObservation {
        satellite_id: obs.satellite_id.clone(),
        ambiguity_id: obs.ambiguity_id.clone(),
        code_m: obs.code_m,
        phase_m: obs.phase_m,
        lli: obs.lli,
    }
}

/// Rebuild arc epochs from preprocessed single-frequency observations, preserving
/// each original epoch's position maps, velocity, and time coordinate. Used after
/// cycle-slip prep (which can rename ambiguity ids and drop satellites) and after
/// Hatch smoothing (which rewrites `code_m`). No arithmetic: the new observation
/// values come straight from the standalone preprocessing function.
fn apply_prepared_observations(
    original: &[RtkArcEpoch],
    prepared: &[CodeSmoothingEpoch],
) -> Vec<RtkArcEpoch> {
    original
        .iter()
        .zip(prepared.iter())
        .map(|(orig, prep)| RtkArcEpoch {
            base: prep
                .base_observations
                .iter()
                .map(from_code_smoothing_obs)
                .collect(),
            rover: prep
                .rover_observations
                .iter()
                .map(from_code_smoothing_obs)
                .collect(),
            satellite_positions_m: orig.satellite_positions_m.clone(),
            base_satellite_positions_m: orig.base_satellite_positions_m.clone(),
            rover_satellite_positions_m: orig.rover_satellite_positions_m.clone(),
            velocity_mps: orig.velocity_mps,
            prediction_time_s: orig.prediction_time_s,
            prediction_epoch: orig.prediction_epoch,
        })
        .collect()
}

/// Thin one arc epoch to the satellites the elevation mask kept this epoch,
/// filtering both observation lists and all three position maps consistently.
fn thin_epoch_to_kept(epoch: &RtkArcEpoch, kept: &[String]) -> RtkArcEpoch {
    let keep: BTreeSet<&str> = kept.iter().map(String::as_str).collect();
    let filter_obs = |obs: &[RtkArcObservation]| {
        obs.iter()
            .filter(|o| keep.contains(o.satellite_id.as_str()))
            .cloned()
            .collect::<Vec<_>>()
    };
    let filter_positions = |map: &BTreeMap<String, [f64; 3]>| {
        map.iter()
            .filter(|(sat, _)| keep.contains(sat.as_str()))
            .map(|(sat, pos)| (sat.clone(), *pos))
            .collect::<BTreeMap<_, _>>()
    };
    RtkArcEpoch {
        base: filter_obs(&epoch.base),
        rover: filter_obs(&epoch.rover),
        satellite_positions_m: filter_positions(&epoch.satellite_positions_m),
        base_satellite_positions_m: filter_positions(&epoch.base_satellite_positions_m),
        rover_satellite_positions_m: filter_positions(&epoch.rover_satellite_positions_m),
        velocity_mps: epoch.velocity_mps,
        prediction_time_s: epoch.prediction_time_s,
        prediction_epoch: epoch.prediction_epoch,
    }
}

/// Core sequential filter solve over already-prepared epochs. This is the
/// behavior-preserving body of the historical `solve_rtk_arc`: the public entry
/// wraps it with optional preprocessing and metadata assembly.
fn solve_prepared_arc(
    epochs: &[RtkArcEpoch],
    config: &RtkArcConfig,
) -> Result<PreparedSolution, RtkArcError> {
    if epochs.is_empty() {
        return Err(RtkArcError::EmptyEpochs);
    }

    let normalized: Vec<NormalizedEpoch> = epochs.iter().map(normalize_epoch).collect();

    // Arc satellite set is the union of per-epoch availability, sorted.
    let arc_sats = arc_satellites(&normalized);
    if arc_sats.len() < MINIMUM_ARC_SATELLITES {
        return Err(RtkArcError::TooFewSatellites {
            count: arc_sats.len(),
            minimum: MINIMUM_ARC_SATELLITES,
        });
    }
    let (wavelengths_m, offsets_m) = expanded_sd_scale_maps(&normalized, config);

    // Per-system references for the whole arc (geometry-based by default).
    let reference_epochs = reference_epoch_terms(&normalized);
    let references_by_system =
        baseline_reference_satellites(config.base_m, &reference_epochs, config.reference.clone())
            .map_err(RtkArcError::Reference)?;
    let reference_sats: BTreeSet<&str> =
        references_by_system.values().map(String::as_str).collect();

    // Map each system letter to its reference SD ambiguity id, taken from the
    // first epoch in which that reference satellite is observed.
    let references = reference_sd_ids(&normalized, &references_by_system);

    // Build the initial filter state, pre-sizing the SD ambiguity columns in the
    // globally sorted (satellite, ambiguity_id) order with first-sighting seeds.
    let mut state = FilterState::new(
        references.clone(),
        config.initial_baseline_m,
        config.baseline_prior_sigma_m,
        config.ambiguity_prior_sigma_m,
    )
    .map_err(RtkArcError::FilterState)?;
    for (id, seed) in sorted_ambiguity_seeds(&normalized) {
        state.ensure_ambiguity(&id, seed);
    }

    let strict_time = config.update_opts.dynamics_model == DynamicsModel::VelocityPropagated;
    let mut previous_time = PredictionTime::default();
    let mut solutions = Vec::with_capacity(epochs.len());
    let mut elevation_masked_sats = BTreeSet::new();

    // Fixes made by the previous epoch's search are not in its carried float
    // information yet. Materialize them if changing reference will release
    // them; retained holds enter the ordinary next update exactly as before.
    let mut pending_holds = BTreeSet::new();

    for (epoch_index, normalized_epoch) in normalized.iter().enumerate() {
        let dt_s = prediction_dt_s(
            epoch_index,
            normalized_epoch.velocity_mps,
            PredictionTime {
                seconds: epochs[epoch_index].prediction_time_s,
                exact: epochs[epoch_index].prediction_epoch,
            },
            &mut previous_time,
            strict_time,
        )?;
        let epoch_template = build_epoch(
            epoch_index,
            normalized_epoch,
            &references_by_system,
            &reference_sats,
            normalized_epoch.velocity_mps,
            dt_s,
        )?;
        let reference_context = ArcReferenceContext {
            references: &references,
            wavelengths: &wavelengths_m,
            offsets: &offsets_m,
            pending: &pending_holds,
            hold_sigma_m: config.update_opts.hold_sigma_m,
        };
        let (epoch, update, epoch_masked, mask_converged, released_holds) = if let Some(mask_deg) =
            config.preprocessing.elevation_mask_deg
        {
            let slipped_satellites: BTreeSet<_> = epochs[epoch_index]
                .base
                .iter()
                .chain(&epochs[epoch_index].rover)
                .filter(|observation| observation.lli.is_some_and(|lli| lli & 1 != 0))
                .map(|observation| observation.satellite_id.as_str())
                .collect();
            let mut predicted = state.clone();
            super::update::propagate_baseline_mean(
                &mut predicted,
                &epoch_template,
                &config.update_opts,
            );
            let mut rover_m =
                std::array::from_fn(|axis| config.base_m[axis] + predicted.baseline_m[axis]);
            let mut mask_iteration = 0;
            'mask_iterations: loop {
                let mut epoch = epoch_template.clone();
                let masked =
                    apply_rover_elevation_mask(&mut epoch, rover_m, mask_deg, &slipped_satellites)?;
                let mut trial_state = state.clone();
                let released = reference_context
                    .prepare(&mut trial_state, &epoch)
                    .map_err(|source| RtkArcError::Update {
                        epoch_index,
                        source,
                    })?;
                let (wavelengths, offsets) =
                    reference_context.relative_maps(&trial_state.references);
                let update_opts =
                    reference_context.update_opts(&trial_state.references, &config.update_opts);
                let update = update_epoch(
                    trial_state,
                    &epoch,
                    config.base_m,
                    &config.model,
                    &wavelengths,
                    &offsets,
                    &update_opts,
                )
                .map_err(|source| RtkArcError::Update {
                    epoch_index,
                    source,
                })?;
                let next_rover_m =
                    std::array::from_fn(|axis| config.base_m[axis] + update.state.baseline_m[axis]);
                let mut next_epoch = epoch_template.clone();
                apply_rover_elevation_mask(
                    &mut next_epoch,
                    next_rover_m,
                    mask_deg,
                    &slipped_satellites,
                )?;
                let stable = epoch_satellite_ids(&epoch) == epoch_satellite_ids(&next_epoch)
                    && epoch
                        .references
                        .iter()
                        .map(|m| &m.sat)
                        .eq(next_epoch.references.iter().map(|m| &m.sat));
                if stable || mask_iteration + 1 == config.update_opts.max_iterations.max(1) {
                    break 'mask_iterations (epoch, update, masked, stable, released);
                }
                rover_m = next_rover_m;
                mask_iteration += 1;
            }
        } else {
            let epoch = epoch_template;
            let released = reference_context
                .prepare(&mut state, &epoch)
                .map_err(|source| RtkArcError::Update {
                    epoch_index,
                    source,
                })?;
            let (wavelengths, offsets) = reference_context.relative_maps(&state.references);
            let update_opts = reference_context.update_opts(&state.references, &config.update_opts);
            let update = update_epoch(
                state,
                &epoch,
                config.base_m,
                &config.model,
                &wavelengths,
                &offsets,
                &update_opts,
            )
            .map_err(|source| RtkArcError::Update {
                epoch_index,
                source,
            })?;
            (epoch, update, Vec::new(), true, released)
        };
        elevation_masked_sats.extend(epoch_masked.iter().cloned());

        let mut used_satellite_ids: Vec<_> = epoch
            .references
            .iter()
            .chain(epoch.nonref.iter())
            .map(|measurement| measurement.sat.clone())
            .collect();
        used_satellite_ids.sort();
        let reference_satellite_ids = epoch
            .references
            .iter()
            .map(|measurement| measurement.sat.clone())
            .collect();
        let mut solution = epoch_solution(
            &update,
            used_satellite_ids,
            reference_satellite_ids,
            epoch_masked,
            &update.state.references,
        );
        solution.elevation_mask_converged = mask_converged;
        solution.released_holds = released_holds;
        pending_holds = update.newly_fixed.iter().cloned().collect();
        solutions.push(solution);
        state = update.state;
    }

    Ok(PreparedSolution {
        references,
        epochs: solutions,
        final_state: state,
        elevation_masked_sats: elevation_masked_sats.into_iter().collect(),
    })
}

fn epoch_satellite_ids(epoch: &Epoch) -> BTreeSet<&str> {
    epoch
        .references
        .iter()
        .chain(&epoch.nonref)
        .map(|measurement| measurement.sat.as_str())
        .collect()
}

fn replaces_reference(current_elevation: Option<f64>, elevation: f64, slipped: bool) -> bool {
    match current_elevation {
        None => true,
        Some(current) => !slipped && elevation >= current,
    }
}

/// Apply RTKLIB's rover-side `zdres` elevation screen at the current predicted
/// rover position, then select the highest-elevation surviving satellite per
/// constellation as `ddres` does. A changed reference follows the ordinary
/// reference-change path before the epoch update.
fn apply_rover_elevation_mask(
    epoch: &mut Epoch,
    rover_m: [f64; 3],
    mask_deg: f64,
    slipped_satellites: &BTreeSet<&str>,
) -> Result<Vec<String>, RtkArcError> {
    let carried_references: BTreeMap<_, _> = epoch
        .references
        .iter()
        .map(|measurement| {
            (
                constellation_letter(&measurement.sat).to_string(),
                measurement.sat.clone(),
            )
        })
        .collect();
    let mut measurements: Vec<_> = epoch
        .references
        .drain(..)
        .chain(epoch.nonref.drain(..))
        .collect();
    let positions = measurements
        .iter()
        .map(|measurement| (measurement.sat.clone(), measurement.rover_tx_pos))
        .collect();
    let result = apply_elevation_mask(
        rover_m,
        &[ElevationMaskEpoch {
            satellite_positions_m: positions,
        }],
        mask_deg,
    )
    .map_err(RtkArcError::ElevationMask)?;
    let kept: BTreeSet<_> = result.epochs[0]
        .kept_satellite_ids
        .iter()
        .map(String::as_str)
        .collect();
    measurements.retain(|measurement| kept.contains(measurement.sat.as_str()));
    measurements.sort_by(|left, right| left.sat.cmp(&right.sat));
    let mut reference_by_system: BTreeMap<String, String> = carried_references
        .into_iter()
        .filter(|(_, reference)| measurements.iter().any(|item| item.sat == *reference))
        .collect();
    let carried_systems: BTreeSet<_> = reference_by_system.keys().cloned().collect();
    for measurement in &measurements {
        let (_, elevation) =
            crate::estimation::substrate::frames::satazel(rover_m, measurement.rover_tx_pos);
        let system = constellation_letter(&measurement.sat).to_string();
        if carried_systems.contains(&system) {
            continue;
        }
        let current_elevation = reference_by_system.get(&system).and_then(|current| {
            measurements
                .iter()
                .find(|candidate| candidate.sat == *current)
                .map(|current_measurement| {
                    crate::estimation::substrate::frames::satazel(
                        rover_m,
                        current_measurement.rover_tx_pos,
                    )
                    .1
                })
        });
        let replace = replaces_reference(
            current_elevation,
            elevation,
            slipped_satellites.contains(measurement.sat.as_str()),
        );
        if replace {
            reference_by_system.insert(system, measurement.sat.clone());
        }
    }
    for measurement in measurements {
        if reference_by_system.get(constellation_letter(&measurement.sat)) == Some(&measurement.sat)
        {
            epoch.references.push(measurement);
        } else {
            epoch.nonref.push(measurement);
        }
    }
    Ok(result.masked_satellite_ids)
}

/// Refer the filter to the reference ambiguities an epoch carries. A reference
/// satellite's ambiguity arc can change within an arc, at a slip or a carrier
/// change; its constellation's double differences then refer to another
/// ambiguity. An invertible common-mode coordinate change moves the carried
/// gauge to the new reference without changing any physical double difference.
/// Held integers are differenced exactly when the new reference is held;
/// otherwise that constellation's holds are released.
/// Arc-relative input scales are immutable. Derive current-reference offsets
/// at their consumption boundary, and transport or visibly release holds.
struct ArcReferenceContext<'a> {
    references: &'a BTreeMap<String, String>,
    wavelengths: &'a BTreeMap<String, f64>,
    offsets: &'a BTreeMap<String, f64>,
    pending: &'a BTreeSet<String>,
    hold_sigma_m: f64,
}

impl ArcReferenceContext<'_> {
    fn common_wavelength(&self, system: &str) -> Option<f64> {
        let mut values = self
            .wavelengths
            .iter()
            .filter(|(id, _)| constellation_letter(id) == system)
            .map(|(_, value)| *value);
        let first = values.next()?;
        values.all(|value| value == first).then_some(first)
    }

    fn relative_maps(
        &self,
        references: &BTreeMap<String, String>,
    ) -> (BTreeMap<String, f64>, BTreeMap<String, f64>) {
        let mut wavelengths = self.wavelengths.clone();
        let mut offsets = self.offsets.clone();
        for (system, reference) in references {
            let Some(origin) = self
                .references
                .get(system)
                .filter(|origin| *origin != reference)
            else {
                continue;
            };
            if let Some(wavelength) = self.common_wavelength(system) {
                wavelengths.entry(origin.clone()).or_insert(wavelength);
            }
            if let Some(&reference_offset) = self.offsets.get(reference) {
                for (id, offset) in &mut offsets {
                    if constellation_letter(id) == system {
                        *offset = if id == origin {
                            -reference_offset
                        } else {
                            *offset - reference_offset
                        };
                    }
                }
                offsets.insert(origin.clone(), -reference_offset);
                offsets.insert(reference.clone(), 0.0);
            }
        }
        (wavelengths, offsets)
    }

    fn update_opts(&self, references: &BTreeMap<String, String>, opts: &UpdateOpts) -> UpdateOpts {
        let mut opts = opts.clone();
        for (system, reference) in references {
            if self
                .references
                .get(system)
                .is_some_and(|origin| origin != reference)
                && self.common_wavelength(system).is_none()
                && !opts.float_only_systems.iter().any(|value| value == system)
            {
                opts.float_only_systems.push(system.clone());
            }
        }
        opts
    }

    fn prepare(
        &self,
        state: &mut FilterState,
        epoch: &Epoch,
    ) -> Result<Vec<ReleasedRtkHold>, UpdateError> {
        let mut released = Vec::new();
        for reference in &epoch.references {
            let system = constellation_letter(&reference.sat);
            let Some(old) = state.references.get(system) else {
                continue;
            };
            if *old == reference.sd_ambiguity_id {
                continue;
            }
            // Arc columns are pre-sized, but reject inconsistent state before
            // changing any reference, rather than silently skipping its gauge.
            for id in [old, &reference.sd_ambiguity_id] {
                if state.ambiguity_pos(id).is_none() {
                    return Err(UpdateError::MissingAmbiguityColumn(id.clone()));
                }
            }
            let reason = if self.common_wavelength(system).is_none() {
                Some(RtkHoldReleaseReason::IncompatibleWavelengths)
            } else if !state.fixed_cycles.contains_key(&reference.sd_ambiguity_id) {
                Some(RtkHoldReleaseReason::UnheldReference)
            } else {
                None
            };
            if let Some(reason) = reason {
                fold_pending_holds(state, system, self.pending, self.hold_sigma_m)?;
                released.extend(
                    state
                        .fixed_cycles
                        .keys()
                        .filter(|id| constellation_letter(id) == system)
                        .map(|id| ReleasedRtkHold {
                            ambiguity_id: id.clone(),
                            reason,
                        }),
                );
                if reason == RtkHoldReleaseReason::IncompatibleWavelengths {
                    state
                        .fixed_cycles
                        .retain(|id, _| constellation_letter(id) != system);
                    state
                        .fixed_m
                        .retain(|id, _| constellation_letter(id) != system);
                }
            }
        }
        refer_to_epoch_references(state, epoch)?;
        Ok(released)
    }
}

/// Absorb only newly fixed constraints which would otherwise disappear before
/// their first measurement update. Older holds already live in the carried
/// information; retained new holds enter the next update instead, not twice.
fn fold_pending_holds(
    state: &mut FilterState,
    system: &str,
    pending: &BTreeSet<String>,
    sigma: f64,
) -> Result<(), UpdateError> {
    let Some(reference) = state.references.get(system) else {
        return Err(UpdateError::MissingSystemReference(system.to_string()));
    };
    let holds: Vec<_> = pending
        .iter()
        .filter(|id| constellation_letter(id) == system)
        .filter_map(|id| {
            state.fixed_m.get(id).zip(state.fixed_cycles.get(id)).map(
                |(&fixed_m, &fixed_cycles)| super::update::Hold {
                    sat_sd_id: id.clone(),
                    ref_sd_id: reference.clone(),
                    fixed_m,
                    fixed_cycles,
                },
            )
        })
        .collect();
    if holds.is_empty() {
        return Ok(());
    }
    let mut information = state.information.clone();
    let mut rhs = vec![0.0; state.dim()];
    super::normal::fold_hold_block_with_ambiguities(
        &mut information,
        &mut rhs,
        state,
        &state.sd_ambiguities_m,
        &holds,
        1.0 / (sigma * sigma),
    )
    .ok_or(UpdateError::SingularGeometry)?;
    let delta = crate::astro::math::linear::solve_flat_normal_first_tie(&information, &rhs)
        .ok_or(UpdateError::SingularGeometry)?;
    for (value, step) in state
        .baseline_m
        .iter_mut()
        .chain(&mut state.sd_ambiguities_m)
        .zip(delta)
    {
        *value += step;
    }
    state.information = information;
    Ok(())
}

fn refer_to_epoch_references(state: &mut FilterState, epoch: &Epoch) -> Result<(), UpdateError> {
    for reference in &epoch.references {
        let system = constellation_letter(&reference.sat);
        if state
            .references
            .get(system)
            .is_none_or(|held| *held == reference.sd_ambiguity_id)
        {
            continue;
        }
        let old_reference = state.references[system].clone();
        let old = state
            .ambiguity_pos(&old_reference)
            .ok_or_else(|| UpdateError::MissingAmbiguityColumn(old_reference.clone()))?;
        let new = state
            .ambiguity_pos(&reference.sd_ambiguity_id)
            .ok_or_else(|| {
                UpdateError::MissingAmbiguityColumn(reference.sd_ambiguity_id.clone())
            })?;
        {
            let n = state.dim();
            let columns: Vec<_> = state
                .sd_ambiguity_ids
                .iter()
                .enumerate()
                .filter(|(_, id)| constellation_letter(id) == system)
                .map(|(index, _)| index + 3)
                .collect();
            // a' = a + u*(a_old-a_new), B = da/da' = I-u*v^T,
            // v=e_old-e_new, v^T*u=0.  Information transforms as B^T I B.
            // In particular the old gauge row e_old^T becomes e_new^T.
            let sums: Vec<f64> = (0..n)
                .map(|j| columns.iter().map(|&i| state.information[i * n + j]).sum())
                .collect();
            let total: f64 = columns.iter().map(|&j| sums[j]).sum();
            for i in 0..n {
                for j in 0..n {
                    let vi = if i == old + 3 {
                        1.0
                    } else if i == new + 3 {
                        -1.0
                    } else {
                        0.0
                    };
                    let vj = if j == old + 3 {
                        1.0
                    } else if j == new + 3 {
                        -1.0
                    } else {
                        0.0
                    };
                    state.information[i * n + j] += -vi * sums[j] - sums[i] * vj + vi * vj * total;
                }
            }
            let shift = state.sd_ambiguities_m[old] - state.sd_ambiguities_m[new];
            for &column in &columns {
                state.sd_ambiguities_m[column - 3] += shift;
            }
        }
        let held_origin = state
            .fixed_cycles
            .get(&reference.sd_ambiguity_id)
            .copied()
            .zip(state.fixed_m.get(&reference.sd_ambiguity_id).copied());
        let mut translated_cycles = BTreeMap::new();
        let mut translated_m = BTreeMap::new();
        if let Some((origin_cycles, origin_m)) = held_origin {
            for (id, cycles) in &state.fixed_cycles {
                if constellation_letter(id) == system && *id != reference.sd_ambiguity_id {
                    let value =
                        cycles
                            .checked_sub(origin_cycles)
                            .ok_or(UpdateError::InvalidInput {
                                field: "reference_change.fixed_cycles",
                                kind: super::RtkInputErrorKind::OutOfRange,
                            })?;
                    translated_cycles.insert(id.clone(), value);
                    translated_m.insert(id.clone(), state.fixed_m[id] - origin_m);
                }
            }
            let value = origin_cycles
                .checked_neg()
                .ok_or(UpdateError::InvalidInput {
                    field: "reference_change.fixed_cycles",
                    kind: super::RtkInputErrorKind::OutOfRange,
                })?;
            translated_cycles.insert(old_reference.clone(), value);
            translated_m.insert(old_reference, -origin_m);
        }
        state
            .references
            .insert(system.to_string(), reference.sd_ambiguity_id.clone());
        state
            .fixed_cycles
            .retain(|id, _| constellation_letter(id) != system);
        state
            .fixed_m
            .retain(|id, _| constellation_letter(id) != system);
        state.fixed_cycles.extend(translated_cycles);
        state.fixed_m.extend(translated_m);
    }
    Ok(())
}

/// Pair an epoch's base and rover observations by satellite and resolve the
/// per-epoch availability set (intersection of all five maps).
fn normalize_epoch(epoch: &RtkArcEpoch) -> NormalizedEpoch<'_> {
    // Per-receiver position maps default entirely to the shared map when empty,
    // matching the Elixir `base_satellite_positions_m default satellite_positions_m`.
    let base_positions = if epoch.base_satellite_positions_m.is_empty() {
        &epoch.satellite_positions_m
    } else {
        &epoch.base_satellite_positions_m
    };
    let rover_positions = if epoch.rover_satellite_positions_m.is_empty() {
        &epoch.satellite_positions_m
    } else {
        &epoch.rover_satellite_positions_m
    };

    let base_by_sat: BTreeMap<&str, &RtkArcObservation> = epoch
        .base
        .iter()
        .map(|obs| (obs.satellite_id.as_str(), obs))
        .collect();
    let rover_by_sat: BTreeMap<&str, &RtkArcObservation> = epoch
        .rover
        .iter()
        .map(|obs| (obs.satellite_id.as_str(), obs))
        .collect();

    let mut paired = BTreeMap::new();
    let mut available = Vec::new();
    for (sat, base_obs) in &base_by_sat {
        let Some(rover_obs) = rover_by_sat.get(sat) else {
            continue;
        };
        if epoch.satellite_positions_m.contains_key(*sat)
            && base_positions.contains_key(*sat)
            && rover_positions.contains_key(*sat)
        {
            paired.insert(*sat, (*base_obs, *rover_obs));
            available.push((*sat).to_string());
        }
    }
    available.sort();

    NormalizedEpoch {
        paired,
        shared_positions: &epoch.satellite_positions_m,
        base_positions,
        rover_positions,
        available,
        velocity_mps: epoch.velocity_mps,
    }
}

/// Union of per-epoch availability sets, sorted.
fn arc_satellites(epochs: &[NormalizedEpoch]) -> Vec<String> {
    let mut sats = BTreeSet::new();
    for epoch in epochs {
        for sat in &epoch.available {
            sats.insert(sat.clone());
        }
    }
    sats.into_iter().collect()
}

/// Build the reference-selection epoch terms (available ids + shared positions).
fn reference_epoch_terms(epochs: &[NormalizedEpoch]) -> Vec<BaselineReferenceEpoch> {
    epochs
        .iter()
        .map(|epoch| BaselineReferenceEpoch {
            available_satellite_ids: epoch.available.clone(),
            satellite_positions_m: epoch.shared_positions.clone(),
        })
        .collect()
}

/// Map each constellation letter to its reference single-difference ambiguity id,
/// taken from the first epoch in which that reference satellite is observed.
fn reference_sd_ids(
    epochs: &[NormalizedEpoch],
    references_by_system: &BTreeMap<String, String>,
) -> BTreeMap<String, String> {
    let mut references = BTreeMap::new();
    for (system, reference_sat) in references_by_system {
        let sd_id = epochs
            .iter()
            .find_map(|epoch| epoch.paired.get(reference_sat.as_str()))
            .map(|(base, rover)| {
                sd_ambiguity_token(reference_sat, &base.ambiguity_id, &rover.ambiguity_id)
            })
            .unwrap_or_else(|| reference_sat.clone());
        references.insert(system.clone(), sd_id);
    }
    references
}

/// First-sighting single-difference ambiguity seeds, returned in the globally
/// sorted `(satellite, ambiguity_id)` order that fixes the filter column layout.
fn sorted_ambiguity_seeds(epochs: &[NormalizedEpoch]) -> Vec<(String, f64)> {
    // Key the dedup by (satellite, sd_id) so the column order matches the Elixir
    // `Enum.sort_by({sat, ambiguity_id})`; record the first-sighting SD
    // phase-minus-code seed.
    let mut seeds: BTreeMap<(String, String), f64> = BTreeMap::new();
    for epoch in epochs {
        for (sat, (base, rover)) in &epoch.paired {
            let sd_id = sd_ambiguity_token(sat, &base.ambiguity_id, &rover.ambiguity_id);
            seeds
                .entry(((*sat).to_string(), sd_id))
                .or_insert_with(|| sd_phase_minus_code(base, rover));
        }
    }
    seeds
        .into_iter()
        .map(|((_, sd_id), seed)| (sd_id, seed))
        .collect()
}

/// Single-difference phase-minus-code seed (metres) for one satellite pair.
fn sd_phase_minus_code(base: &RtkArcObservation, rover: &RtkArcObservation) -> f64 {
    (rover.phase_m - base.phase_m) - (rover.code_m - base.code_m)
}

/// An epoch's time for prediction deltas: its seconds coordinate and its
/// exact epoch, either of which may be absent.
#[derive(Debug, Clone, Copy, Default)]
struct PredictionTime {
    seconds: Option<f64>,
    exact: Option<ExactEpoch>,
}

impl PredictionTime {
    fn is_some(self) -> bool {
        self.seconds.is_some() || self.exact.is_some()
    }

    /// Seconds from `earlier` to `self`: exact when both carry an exact
    /// epoch, otherwise the difference of the seconds coordinates.
    fn seconds_since(self, earlier: Self) -> Option<f64> {
        match (self.exact, earlier.exact) {
            (Some(current), Some(previous)) => Some(current.seconds_since(previous)),
            _ => Some(self.seconds? - earlier.seconds?),
        }
    }
}

/// Compute the prediction delta (seconds) before this epoch's update.
fn prediction_dt_s(
    epoch_index: usize,
    velocity_mps: Option<[f64; 3]>,
    current_time: PredictionTime,
    previous_time: &mut PredictionTime,
    strict_time: bool,
) -> Result<f64, RtkArcError> {
    // The dt is only consumed by the velocity-propagated branch; carry the time
    // forward regardless so a later epoch can still difference against it.
    let dt = match (epoch_index, current_time.seconds_since(*previous_time)) {
        (0, _) => 0.0,
        (_, Some(dt)) => dt,
        _ => {
            if strict_time {
                return Err(RtkArcError::InvalidEpochTime { epoch_index });
            }
            0.0
        }
    };
    if current_time.is_some() {
        *previous_time = current_time;
    }
    // A velocity-propagated epoch with no velocity still gets a zero delta (the
    // predict step is then an identity mean shift), matching the kernel guard.
    let _ = velocity_mps;
    Ok(dt)
}

/// Assemble the per-epoch [`Epoch`] (references + non-references) for the update.
fn build_epoch(
    epoch_index: usize,
    epoch: &NormalizedEpoch,
    references_by_system: &BTreeMap<String, String>,
    reference_sats: &BTreeSet<&str>,
    velocity_mps: Option<[f64; 3]>,
    dt_s: f64,
) -> Result<Epoch, RtkArcError> {
    // References present this epoch, in constellation-letter order.
    let mut references = Vec::new();
    for reference_sat in references_by_system.values() {
        if let Some((base, rover)) = epoch.paired.get(reference_sat.as_str()) {
            references.push(sat_meas(epoch_index, epoch, reference_sat, base, rover)?);
        }
    }

    // Non-reference satellites in sorted availability order.
    let mut nonref = Vec::new();
    for sat in &epoch.available {
        if reference_sats.contains(sat.as_str()) {
            continue;
        }
        let (base, rover) = epoch.paired[sat.as_str()];
        nonref.push(sat_meas(epoch_index, epoch, sat, base, rover)?);
    }

    Ok(Epoch {
        references,
        nonref,
        velocity_mps,
        dt_s,
    })
}

/// Build one satellite's [`SatMeas`] from its paired observations and positions.
fn sat_meas(
    epoch_index: usize,
    epoch: &NormalizedEpoch,
    sat: &str,
    base: &RtkArcObservation,
    rover: &RtkArcObservation,
) -> Result<SatMeas, RtkArcError> {
    let position = |map: &BTreeMap<String, [f64; 3]>| {
        map.get(sat)
            .copied()
            .ok_or_else(|| RtkArcError::MissingPosition {
                epoch_index,
                satellite_id: sat.to_string(),
            })
    };
    Ok(SatMeas {
        sat: sat.to_string(),
        sd_ambiguity_id: sd_ambiguity_token(sat, &base.ambiguity_id, &rover.ambiguity_id),
        base_code_m: base.code_m,
        base_phase_m: base.phase_m,
        rover_code_m: rover.code_m,
        rover_phase_m: rover.phase_m,
        base_tx_pos: position(epoch.base_positions)?,
        rover_tx_pos: position(epoch.rover_positions)?,
        pos: position(epoch.shared_positions)?,
    })
}

/// Translate one [`super::EpochUpdate`] into the public per-epoch solution,
/// including the double-difference ids of the newly fixed ambiguities.
fn epoch_solution(
    update: &super::EpochUpdate,
    used_satellite_ids: Vec<String>,
    reference_satellite_ids: Vec<String>,
    elevation_masked_satellite_ids: Vec<String>,
    references: &BTreeMap<String, String>,
) -> RtkArcEpochSolution {
    let reported = update
        .reported_sd_ambiguities_m
        .as_ref()
        .unwrap_or(&update.state.sd_ambiguities_m);
    let sd_ambiguities_m = update
        .state
        .sd_ambiguity_ids
        .iter()
        .cloned()
        .zip(reported.iter().copied())
        .collect();
    let fixed_double_difference_ids = update
        .newly_fixed
        .iter()
        .map(|sd_id| double_difference_id(sd_id, references))
        .collect();

    RtkArcEpochSolution {
        reported_baseline_m: update.reported_baseline_m,
        float_baseline_m: update.state.baseline_m,
        integer_fixed: update.integer_fixed,
        integer_ratio: update.integer_ratio,
        newly_fixed: update.newly_fixed.clone(),
        fixed_ids: update.fixed_ids.clone(),
        released_holds: Vec::new(),
        sd_ambiguities_m,
        fixed_double_difference_ids,
        used_satellite_ids,
        reference_satellite_ids,
        elevation_masked_satellite_ids,
        elevation_mask_converged: true,
        search: update.search.clone(),
        residuals: update.residuals.clone(),
        geometry_quality: update.geometry_quality,
    }
}

/// Double-difference ambiguity id for an SD id against its own-system reference.
fn double_difference_id(sd_id: &str, references: &BTreeMap<String, String>) -> String {
    let system = constellation_letter(sd_id);
    match references.get(system) {
        Some(ref_sd_id) => {
            // The reference satellite token is the reference SD id for a clean
            // arc; for a split reference the token form still differences against
            // the recorded reference SD id.
            dd_ambiguity_token(sd_id, sd_id, ref_sd_id, ref_sd_id)
        }
        None => sd_id.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::math::vec3::{norm3, sub3};
    use crate::constants::{C_M_S, F_L1_HZ, F_L2_HZ, RAD_TO_DEG};
    use crate::observables::{predict, ObservableState, ObservablesError, PredictOptions};
    use crate::rtk_filter::{
        defaults, FixedBaselineSolution, FixedSolveOpts, FloatSolveOpts, ResidualValidationOpts,
        SearchOpts, StochasticModel,
    };
    use crate::{GnssSatelliteId, GnssSystem};

    #[test]
    fn prediction_deltas_between_exact_epochs_are_exact() {
        let at = |second: f64| ExactEpoch::from_civil(2026, 9, 23, 6, 30, second).unwrap();
        let timed = |second: f64| PredictionTime {
            seconds: Some(at(second).j2000_seconds()),
            exact: Some(at(second)),
        };
        let mut previous = PredictionTime::default();
        assert_eq!(
            prediction_dt_s(0, None, timed(0.2), &mut previous, true).unwrap(),
            0.0
        );
        // Labels a tenth apart: the exact difference is 0.1, the difference
        // of their J2000 doubles is not.
        assert_eq!(
            prediction_dt_s(1, None, timed(0.3), &mut previous, true).unwrap(),
            0.1
        );
        assert_ne!(at(0.3).j2000_seconds() - at(0.2).j2000_seconds(), 0.1);
        // Without an exact epoch on both sides, the seconds are differenced.
        let seconds_only = PredictionTime {
            seconds: Some(at(0.4).j2000_seconds()),
            exact: None,
        };
        assert_eq!(
            prediction_dt_s(2, None, seconds_only, &mut previous, true).unwrap(),
            at(0.4).j2000_seconds() - at(0.3).j2000_seconds()
        );
        // An epoch with no time is an error when strict and zero otherwise,
        // and leaves the previous time in place.
        assert!(prediction_dt_s(3, None, PredictionTime::default(), &mut previous, true).is_err());
        assert_eq!(
            prediction_dt_s(3, None, PredictionTime::default(), &mut previous, false).unwrap(),
            0.0
        );
        assert_eq!(previous.seconds, Some(at(0.4).j2000_seconds()));
    }

    struct ArcSource {
        states: BTreeMap<GnssSatelliteId, [f64; 3]>,
    }

    impl ObservableEphemerisSource for ArcSource {
        fn observable_state_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(ObservableState {
                position_ecef_m: self
                    .states
                    .get(&sat)
                    .copied()
                    .ok_or(ObservablesError::NoEphemeris)?,
                clock_s: Some(0.0),
            })
        }
    }

    use crate::observables::ObservableEphemerisSource;

    fn sat_layout() -> [(u8, [f64; 3]); 6] {
        [
            (1, [14_350_000.0, 3_190_000.0, 21_440_000.0]),
            (2, [20_000_000.0, 3_000_000.0, 18_000_000.0]),
            (3, [9_000_000.0, 9_000_000.0, 22_000_000.0]),
            (4, [16_000_000.0, -4_000_000.0, 21_000_000.0]),
            (5, [10_000_000.0, -2_000_000.0, 24_000_000.0]),
            (6, [19_000_000.0, 8_000_000.0, 17_000_000.0]),
        ]
    }

    fn build() -> (ArcSource, Vec<GnssSatelliteId>, Vec<String>) {
        let layout = sat_layout();
        let ids: Vec<GnssSatelliteId> = layout
            .iter()
            .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("prn"))
            .collect();
        let tokens = ids.iter().map(|id| id.to_string()).collect();
        let states = ids
            .iter()
            .zip(layout.iter())
            .map(|(id, (_, position))| (*id, *position))
            .collect();
        (ArcSource { states }, ids, tokens)
    }

    fn range_m(source: &ArcSource, sat: GnssSatelliteId, receiver: [f64; 3]) -> f64 {
        predict(
            source,
            sat,
            receiver,
            0.0,
            PredictOptions {
                carrier_hz: F_L1_HZ,
                light_time: true,
                sagnac: true,
            },
        )
        .expect("predict")
        .geometric_range_m
    }

    fn observation(id: &str, code_m: f64, phase_m: f64) -> RtkArcObservation {
        RtkArcObservation {
            satellite_id: id.to_string(),
            ambiguity_id: id.to_string(),
            code_m,
            phase_m,
            lli: None,
        }
    }

    fn update_opts() -> UpdateOpts {
        UpdateOpts {
            hold_sigma_m: 1.0e-4,
            position_tol_m: defaults::POSITION_TOL_M,
            ambiguity_tol_m: defaults::AMBIGUITY_TOL_M,
            max_iterations: defaults::MAX_ITERATIONS,
            process_noise_baseline_sigma_m: 0.0,
            dynamics_model: DynamicsModel::ConstantPosition,
            float_only_systems: Vec::new(),
            report_residuals: false,
            force_report_iterate_failure: false,
            receiver_antenna_corrections: None,
            ar_arming_sigma_m: None,
            search: SearchOpts {
                ratio_threshold: defaults::RATIO_THRESHOLD,
            },
        }
    }

    fn config(
        wavelengths_m: BTreeMap<String, f64>,
        offsets_m: BTreeMap<String, f64>,
    ) -> RtkArcConfig {
        RtkArcConfig {
            base_m: [3_512_900.0, 780_500.0, 5_248_700.0],
            reference: BaselineReferenceSelection::Auto,
            model: MeasModel {
                code_sigma_m: defaults::CODE_SIGMA_M,
                phase_sigma_m: defaults::PHASE_SIGMA_M,
                sagnac: true,
                stochastic: StochasticModel::Simple {
                    elevation_weighting: false,
                },
            },
            baseline_prior_sigma_m: 100.0,
            ambiguity_prior_sigma_m: 1000.0,
            initial_baseline_m: [0.0, 0.0, 0.0],
            wavelengths_m,
            offsets_m,
            update_opts: update_opts(),
            preprocessing: RtkArcPreprocessing::default(),
        }
    }

    /// Build an arc where the rover sits at base+baseline with integer-cycle
    /// double-difference ambiguities, and confirm the driver recovers both.
    fn integer_arc() -> (Vec<RtkArcEpoch>, RtkArcConfig, [f64; 3], Vec<String>) {
        let (source, ids, tokens) = build();
        let base = [3_512_900.0, 780_500.0, 5_248_700.0];
        let baseline = [12.0, -7.0, 9.0];
        let rover = [
            base[0] + baseline[0],
            base[1] + baseline[1],
            base[2] + baseline[2],
        ];
        let wavelength_m = C_M_S / F_L1_HZ;
        // Per-satellite single-difference integer ambiguities (cycles).
        let sd_cycles = [3i64, -2, 5, 1, -4, 6];

        let positions: BTreeMap<String, [f64; 3]> = ids
            .iter()
            .zip(tokens.iter())
            .map(|(id, token)| (token.clone(), source.states[id]))
            .collect();

        let mut epochs = Vec::new();
        for _ in 0..6 {
            let mut base_obs = Vec::new();
            let mut rover_obs = Vec::new();
            for (idx, (id, token)) in ids.iter().zip(tokens.iter()).enumerate() {
                let base_range = range_m(&source, *id, base);
                let rover_range = range_m(&source, *id, rover);
                let ambiguity_m = sd_cycles[idx] as f64 * wavelength_m;
                base_obs.push(observation(token, base_range, base_range));
                rover_obs.push(observation(token, rover_range, rover_range + ambiguity_m));
            }
            epochs.push(RtkArcEpoch {
                base: base_obs,
                rover: rover_obs,
                satellite_positions_m: positions.clone(),
                base_satellite_positions_m: BTreeMap::new(),
                rover_satellite_positions_m: BTreeMap::new(),
                velocity_mps: None,
                prediction_time_s: None,
                prediction_epoch: None,
            });
        }

        let wavelengths_m: BTreeMap<String, f64> =
            tokens.iter().map(|t| (t.clone(), wavelength_m)).collect();
        let offsets_m: BTreeMap<String, f64> = tokens.iter().map(|t| (t.clone(), 0.0)).collect();
        (epochs, config(wavelengths_m, offsets_m), baseline, tokens)
    }

    /// G02 on a second carrier arc from `change_at`, `G02~freq2`, with loss of
    /// lock at epoch 3 on the base (`side` 0), the rover (1) or both (2).
    #[test]
    fn fixing_and_holding_keep_a_carrier_arcs_scale_after_a_split() {
        for side in 0..3 {
            for change_at in [2, 3, 4] {
                let (template, mut config, baseline, _) = integer_arc();
                let mut epochs = vec![template[0].clone(); 12];
                let lambda = config.wavelengths_m["G02"];
                config.wavelengths_m.insert("G02~freq2".into(), lambda);
                config.offsets_m.insert("G02~freq2".into(), 0.0);
                config.preprocessing.cycle_slip = Some(CycleSlipPolicy::SplitArc);
                for (index, epoch) in epochs.iter_mut().enumerate() {
                    for (receiver, observation) in
                        [(0, &mut epoch.base[1]), (1, &mut epoch.rover[1])]
                    {
                        if index >= change_at {
                            observation.ambiguity_id = "G02~freq2".into();
                        }
                        if index == 3 && (side == 2 || side == receiver) {
                            observation.lli = Some(1);
                        }
                        if receiver == 1 {
                            observation.phase_m = observation.code_m - 2.0 * lambda;
                        }
                    }
                }
                let what = format!("side {side}, change at {change_at}");
                let solved = solve_rtk_arc(&epochs, &config).expect(&what);
                let last = solved.epochs.last().expect("last epoch");
                assert!(
                    norm3(sub3(last.reported_baseline_m, baseline)) < 1e-3,
                    "{what}"
                );
                assert!(last.integer_fixed, "{what}");
                assert!(
                    last.fixed_ids.iter().any(|id| id.contains("G02~freq2")),
                    "{what}: {:?}",
                    last.fixed_ids
                );
            }
        }
    }

    /// The single-frequency arc of the test above solved with wavelength and
    /// offset maps keyed as the ionosphere-free preparation keys its own: a
    /// satellite on its own clean arc by its id, every other ambiguity by its
    /// double-difference id, `<single-difference id>|ref=G01`, and nothing for
    /// G02 alone. Fixing and holding look each single-difference id up in them.
    #[test]
    fn fixing_and_holding_resolve_ids_keyed_by_their_reference() {
        for side in 0..3 {
            for change_at in [2, 3, 4] {
                let (template, mut config, baseline, _) = integer_arc();
                let lambda = config.wavelengths_m["G02"];
                let mut epochs = vec![template[0].clone(); 12];
                config.preprocessing.cycle_slip = Some(CycleSlipPolicy::SplitArc);
                for (index, epoch) in epochs.iter_mut().enumerate() {
                    for (receiver, observation) in
                        [(0, &mut epoch.base[1]), (1, &mut epoch.rover[1])]
                    {
                        if index >= change_at {
                            observation.ambiguity_id = "G02~freq2".into();
                        }
                        if index == 3 && (side == 2 || side == receiver) {
                            observation.lli = Some(1);
                        }
                        if receiver == 1 {
                            observation.phase_m = observation.code_m - 2.0 * lambda;
                        }
                    }
                }
                let smoothing: Vec<CodeSmoothingEpoch> =
                    epochs.iter().map(to_code_smoothing_epoch).collect();
                let prepared =
                    prepare_cycle_slip_baseline_epochs(&smoothing, CycleSlipPolicy::SplitArc)
                        .expect("prepare");
                let mut wavelengths_m = BTreeMap::new();
                let mut offsets_m = BTreeMap::new();
                for epoch in &prepared.epochs {
                    for base in &epoch.base_observations {
                        let rover = epoch
                            .rover_observations
                            .iter()
                            .find(|rover| rover.satellite_id == base.satellite_id)
                            .expect("rover observation");
                        let sat = base.satellite_id.as_str();
                        let sd = sd_ambiguity_token(sat, &base.ambiguity_id, &rover.ambiguity_id);
                        let key = dd_ambiguity_token(sat, &sd, "G01", "G01");
                        wavelengths_m.insert(key.clone(), lambda);
                        offsets_m.insert(key, 0.0);
                    }
                }
                let what = format!("side {side}, change at {change_at}");
                assert!(!wavelengths_m.contains_key("G02"), "{what}");
                config.wavelengths_m = wavelengths_m;
                config.offsets_m = offsets_m;
                let solved = solve_rtk_arc(&epochs, &config)
                    .unwrap_or_else(|error| panic!("{what}: {error:?}"));
                assert_eq!(
                    solved.references.get("G").map(String::as_str),
                    Some("G01"),
                    "{what}"
                );
                // Fixed by the end and held over the last epochs.
                for last in &solved.epochs[solved.epochs.len() - 3..] {
                    assert!(
                        norm3(sub3(last.reported_baseline_m, baseline)) < 1e-3,
                        "{what}"
                    );
                    assert!(last.integer_fixed, "{what}");
                    assert!(
                        last.fixed_ids.iter().any(|id| id.contains("G02~freq2")),
                        "{what}: {:?}",
                        last.fixed_ids
                    );
                }
            }
        }
    }

    /// The dual-frequency pipeline with G02 on a second carrier arc from
    /// `change_at`, and loss of lock at epoch 3 on the base (`side` 0), the
    /// rover (1), both (2) or neither (3). The ionosphere-free maps are keyed by
    /// double-difference ids, `<id>|ref=G01`; the fixed solve looks each
    /// single-difference id up in them.
    #[test]
    fn dual_frequency_fixing_resolves_each_ambiguity_to_its_carrier_arcs_scale() {
        for side in 0..4 {
            for change_at in [2, 3, 4] {
                let (template, mut wide) = dual_arc();
                let mut epochs = vec![template[0].clone(); 12];
                wide.options.min_epochs = 1;
                let slip_options = CycleSlipOptions {
                    gf_threshold_m: 1e9,
                    mw_threshold_cycles: 1e9,
                    ..CycleSlipOptions::default()
                };
                wide.cycle_slip = Some(RtkDualCycleSlipConfig::new(
                    CycleSlipPolicy::SplitArc,
                    slip_options,
                ));
                for (index, epoch) in epochs.iter_mut().enumerate() {
                    epoch.epoch_sort_key = Some(format!("{index:03}"));
                    epoch.gap_time_s = Some(index as f64);
                    let observation = &mut epoch.observations[1];
                    for (receiver, carrier) in
                        [(0, &mut observation.base), (1, &mut observation.rover)]
                    {
                        if index >= change_at {
                            carrier.ambiguity_id = "G02~freq2".into();
                            if receiver == 1 {
                                carrier.phi1_cycles += 1.0;
                            }
                        }
                        if index == 3 && (side == 2 || side == receiver) {
                            carrier.lli1 = Some(1);
                        }
                    }
                }
                let combined = RtkWideLaneFixedArcConfig {
                    ionosphere_free: RtkIonosphereFreeArcConfig {
                        base_m: wide.base_m,
                        initial_baseline_m: [0.0; 3],
                        reference: wide.reference.clone(),
                        apply_troposphere: false,
                    },
                    wide_lane: wide,
                    solve: RtkWideLaneFixedArcSolveConfig::Sequential(config(
                        BTreeMap::new(),
                        BTreeMap::new(),
                    )),
                };
                let what = format!("side {side}, change at {change_at}");
                let wide_lane = fix_wide_lane_rtk_arc(&epochs, &combined.wide_lane).expect(&what);
                let ionosphere_free = prepare_ionosphere_free_rtk_arc(
                    &wide_lane.epochs,
                    &wide_lane.wide_lane_cycles,
                    &combined.ionosphere_free,
                )
                .expect(&what);
                // Each single-difference id resolves to the value its own
                // reference-qualified entry holds, never to another carrier
                // arc's or to the satellite's first.
                let mut resolved = 0;
                for epoch in &ionosphere_free.epochs {
                    for base in &epoch.base {
                        let Some(rover) = epoch
                            .rover
                            .iter()
                            .find(|rover| rover.satellite_id == base.satellite_id)
                        else {
                            continue;
                        };
                        let sd = sd_ambiguity_token(
                            &base.satellite_id,
                            &base.ambiguity_id,
                            &rover.ambiguity_id,
                        );
                        for map in [&ionosphere_free.wavelengths_m, &ionosphere_free.offsets_m] {
                            let own: Vec<f64> = map
                                .iter()
                                .filter(|(key, _)| {
                                    key.as_str() == sd
                                        || key.split_once("|ref=").is_some_and(|(id, _)| id == sd)
                                })
                                .map(|(_, value)| *value)
                                .collect();
                            let [expected] = own[..] else {
                                continue;
                            };
                            assert_eq!(
                                scale_value_for(map, &sd, &base.satellite_id),
                                Some(expected),
                                "{what}: {sd}"
                            );
                            resolved += 1;
                        }
                    }
                }
                assert!(resolved > 0, "{what}");
                let solution = solve_wide_lane_fixed_rtk_arc(&epochs, &combined)
                    .unwrap_or_else(|error| panic!("{what}: {error:?}"));
                let RtkWideLaneFixedArcSolution::Sequential(solution) = solution else {
                    panic!("{what}: the sequential solve was configured");
                };
                assert_eq!(solution.solution.epochs.len(), epochs.len(), "{what}");
            }
        }
    }

    /// The integer arc over twelve epochs with the reference, G01, slipping by
    /// one cycle on the rover at epoch 7, flagged on the rover. Its ambiguity
    /// arc changes there, and the arc fixes and holds again against the new one.
    #[test]
    fn single_frequency_fixing_holds_across_a_reference_slip() {
        let (template, mut config, baseline, _) = integer_arc();
        let lambda = config.wavelengths_m["G01"];
        config.preprocessing.cycle_slip = Some(CycleSlipPolicy::SplitArc);
        let mut epochs = vec![template[0].clone(); 12];
        for (index, epoch) in epochs.iter_mut().enumerate() {
            let reference = &mut epoch.rover[0];
            assert_eq!(reference.satellite_id, "G01");
            if index >= 7 {
                reference.phase_m += lambda;
            }
            if index == 7 {
                reference.lli = Some(1);
            }
        }
        let solved = solve_rtk_arc(&epochs, &config).unwrap_or_else(|error| panic!("{error:?}"));
        for epoch in &solved.epochs[9..] {
            assert!(epoch.integer_fixed);
            assert!(norm3(sub3(epoch.reported_baseline_m, baseline)) < 1e-3);
        }
    }

    /// The dual-frequency pipeline over sixteen epochs of the integer arc's
    /// geometry, G01 the reference. G02 moves to a second carrier arc at epoch 4
    /// and slips at 5; with `reference_change`, G01 moves to one at 7 and slips
    /// at 8. Each slip is flagged on the base (`side` 0), the rover (1) or both
    /// (2). The ionosphere-free offsets of every satellite depend on the
    /// reference's wide-lane ambiguity, which changes with the reference's arc.
    #[test]
    fn dual_frequency_fixing_holds_across_a_reference_ambiguity_change() {
        for reference_change in [false, true] {
            for side in 0..3 {
                let (single, solve, baseline, _) = integer_arc();
                let (template, mut wide) = dual_arc();
                wide.options.min_epochs = 1;
                wide.cycle_slip = Some(RtkDualCycleSlipConfig::new(
                    CycleSlipPolicy::SplitArc,
                    CycleSlipOptions {
                        gf_threshold_m: 1e9,
                        mw_threshold_cycles: 1e9,
                        ..CycleSlipOptions::default()
                    },
                ));
                let mut epochs = vec![template[0].clone(); 16];
                for (index, epoch) in epochs.iter_mut().enumerate() {
                    epoch.epoch_sort_key = Some(format!("{index:03}"));
                    epoch.gap_time_s = Some(index as f64);
                    epoch.satellite_positions_m = single[0].satellite_positions_m.clone();
                    epoch.observations = single[0]
                        .base
                        .iter()
                        .zip(&single[0].rover)
                        .enumerate()
                        .map(|(position, (base, rover))| {
                            let changed = (position == 1 && index >= 4)
                                || (position == 0 && reference_change && index >= 7);
                            let slipped = (position == 1 && index == 5)
                                || (position == 0 && reference_change && index == 8);
                            let carrier = if changed {
                                format!("{}~freq2", base.satellite_id)
                            } else {
                                base.satellite_id.clone()
                            };
                            let make = |receiver: usize, code_m: f64| {
                                let mut n1 = if receiver == 1 {
                                    position as f64 + 3.0
                                } else {
                                    0.0
                                };
                                let n2 = if receiver == 1 { position as f64 } else { 0.0 };
                                if receiver == 1 && position == 1 && index >= 4 {
                                    n1 += 2.0;
                                }
                                if receiver == 1 && position == 0 && reference_change && index >= 7
                                {
                                    n1 += 1.0;
                                }
                                RtkDualFrequencyObservation {
                                    ambiguity_id: carrier.clone(),
                                    p1_m: code_m,
                                    p2_m: code_m,
                                    phi1_cycles: code_m / (C_M_S / F_L1_HZ) + n1,
                                    phi2_cycles: code_m / (C_M_S / F_L2_HZ) + n2,
                                    f1_hz: F_L1_HZ,
                                    f2_hz: F_L2_HZ,
                                    lli1: Some(i64::from(
                                        slipped && (side == 2 || side == receiver),
                                    )),
                                    lli2: None,
                                }
                            };
                            RtkDualFrequencySatelliteObservation {
                                satellite_id: base.satellite_id.clone(),
                                base: make(0, base.code_m),
                                rover: make(1, rover.code_m),
                            }
                        })
                        .collect();
                }
                let combined = RtkWideLaneFixedArcConfig {
                    ionosphere_free: RtkIonosphereFreeArcConfig {
                        base_m: wide.base_m,
                        initial_baseline_m: [0.0; 3],
                        reference: wide.reference.clone(),
                        apply_troposphere: false,
                    },
                    wide_lane: wide,
                    solve: RtkWideLaneFixedArcSolveConfig::Sequential(solve),
                };
                let what = format!("reference change {reference_change}, side {side}");
                let solution = solve_wide_lane_fixed_rtk_arc(&epochs, &combined)
                    .unwrap_or_else(|error| panic!("{what}: {error:?}"));
                let RtkWideLaneFixedArcSolution::Sequential(solution) = solution else {
                    panic!("{what}: the sequential solve was configured");
                };
                // Fixed, and held, over the last three epochs.
                for epoch in &solution.solution.epochs[13..] {
                    assert!(epoch.integer_fixed, "{what}");
                    assert!(
                        norm3(sub3(epoch.reported_baseline_m, baseline)) < 1e-3,
                        "{what}"
                    );
                }
            }
        }
    }

    #[test]
    fn arc_recovers_baseline_and_fixes_integers() {
        let (epochs, config, baseline, _tokens) = integer_arc();
        let solution = solve_rtk_arc(&epochs, &config).expect("arc solves");
        assert_eq!(solution.epochs.len(), epochs.len());
        // The reference is the highest-elevation satellite (PRN 1, straight up).
        assert_eq!(
            solution.references.get("G").map(String::as_str),
            Some("G01")
        );

        let last = solution.epochs.last().expect("last epoch");
        let error_m = norm3(sub3(last.reported_baseline_m, baseline));
        assert!(error_m < 1.0e-3, "baseline error {error_m} m too large");
        assert!(
            last.integer_fixed,
            "arc should fix integers by the last epoch"
        );
    }

    #[test]
    fn arc_rejects_too_few_satellites() {
        let (mut epochs, config, _baseline, _tokens) = integer_arc();
        for epoch in &mut epochs {
            epoch.base.truncate(3);
            epoch.rover.truncate(3);
        }
        let error = solve_rtk_arc(&epochs, &config).expect_err("too few sats");
        assert_eq!(
            error,
            RtkArcError::TooFewSatellites {
                count: 3,
                minimum: MINIMUM_ARC_SATELLITES,
            }
        );
    }

    #[test]
    fn arc_strict_time_requires_epoch_times() {
        let (mut epochs, mut config, _baseline, _tokens) = integer_arc();
        config.update_opts.dynamics_model = DynamicsModel::VelocityPropagated;
        for epoch in &mut epochs {
            epoch.velocity_mps = Some([0.0, 0.0, 0.0]);
        }
        // No prediction_time_s set -> strict mode errors on the second epoch.
        let error = solve_rtk_arc(&epochs, &config).expect_err("strict time");
        assert_eq!(error, RtkArcError::InvalidEpochTime { epoch_index: 1 });
    }

    use crate::rtk_filter::{ReceiverAntennaCalibration, ReceiverAntennaCorrections};

    /// Cycle-slip splitting mints new single-difference ambiguity ids; a real
    /// IF/narrow-lane pipeline emits a wavelength/offset per resulting DD. Mirror
    /// that here by adding an L1 entry for every post-split SD id so the search has
    /// a scale for the freshly split arcs.
    fn augment_scale_for_cycle_slip(epochs: &[RtkArcEpoch], config: &mut RtkArcConfig) {
        let policy = config
            .preprocessing
            .cycle_slip
            .expect("cycle slip policy set");
        let cs: Vec<CodeSmoothingEpoch> = epochs.iter().map(to_code_smoothing_epoch).collect();
        let prepared = prepare_cycle_slip_baseline_epochs(&cs, policy).expect("cycle slip prep");
        let wavelength_m = C_M_S / F_L1_HZ;
        for epoch in &prepared.epochs {
            let base: BTreeMap<&str, &CodeSmoothingObservation> = epoch
                .base_observations
                .iter()
                .map(|o| (o.satellite_id.as_str(), o))
                .collect();
            for rover in &epoch.rover_observations {
                if let Some(b) = base.get(rover.satellite_id.as_str()) {
                    let sd = sd_ambiguity_token(
                        &rover.satellite_id,
                        &b.ambiguity_id,
                        &rover.ambiguity_id,
                    );
                    config
                        .wavelengths_m
                        .entry(sd.clone())
                        .or_insert(wavelength_m);
                    config.offsets_m.entry(sd).or_insert(0.0);
                }
            }
        }
    }

    /// Independent re-implementation of the driver's prepare-then-solve composition
    /// from the standalone `crate::rtk::*` functions, used to prove the wrapper adds
    /// no numerics of its own. The struct conversions are pure field copies, so the
    /// floating-point values come from the same standalone functions the driver
    /// calls, giving bit-for-bit equality.
    fn manual_solve(epochs: &[RtkArcEpoch], config: &RtkArcConfig) -> RtkArcSolution {
        let pre = &config.preprocessing;
        let mut work = epochs.to_vec();
        let mut dropped_sats = Vec::new();
        let mut split_cycle_slip_arcs = Vec::new();
        let mut elevation_masked_sats = Vec::new();

        let to_cs = |epoch: &RtkArcEpoch| CodeSmoothingEpoch {
            base_observations: epoch
                .base
                .iter()
                .map(|o| CodeSmoothingObservation {
                    satellite_id: o.satellite_id.clone(),
                    ambiguity_id: o.ambiguity_id.clone(),
                    code_m: o.code_m,
                    phase_m: o.phase_m,
                    lli: o.lli,
                })
                .collect(),
            rover_observations: epoch
                .rover
                .iter()
                .map(|o| CodeSmoothingObservation {
                    satellite_id: o.satellite_id.clone(),
                    ambiguity_id: o.ambiguity_id.clone(),
                    code_m: o.code_m,
                    phase_m: o.phase_m,
                    lli: o.lli,
                })
                .collect(),
        };
        let rebuild = |original: &[RtkArcEpoch], prepared: &[CodeSmoothingEpoch]| {
            original
                .iter()
                .zip(prepared.iter())
                .map(|(orig, prep)| {
                    let conv = |o: &CodeSmoothingObservation| RtkArcObservation {
                        satellite_id: o.satellite_id.clone(),
                        ambiguity_id: o.ambiguity_id.clone(),
                        code_m: o.code_m,
                        phase_m: o.phase_m,
                        lli: o.lli,
                    };
                    RtkArcEpoch {
                        base: prep.base_observations.iter().map(conv).collect(),
                        rover: prep.rover_observations.iter().map(conv).collect(),
                        satellite_positions_m: orig.satellite_positions_m.clone(),
                        base_satellite_positions_m: orig.base_satellite_positions_m.clone(),
                        rover_satellite_positions_m: orig.rover_satellite_positions_m.clone(),
                        velocity_mps: orig.velocity_mps,
                        prediction_time_s: orig.prediction_time_s,
                        prediction_epoch: orig.prediction_epoch,
                    }
                })
                .collect::<Vec<_>>()
        };

        if let Some(policy) = pre.cycle_slip {
            let cs: Vec<CodeSmoothingEpoch> = work.iter().map(to_cs).collect();
            let result = prepare_cycle_slip_baseline_epochs(&cs, policy).expect("cycle slip prep");
            work = rebuild(&work, &result.epochs);
            dropped_sats = result.dropped_sats;
            split_cycle_slip_arcs = result.split_arcs;
        }
        if let Some(cap) = pre.hatch_window_cap {
            let cs: Vec<CodeSmoothingEpoch> = work.iter().map(to_cs).collect();
            let smoothed = hatch_smooth_baseline_code_epochs(&cs, cap).expect("hatch smoothing");
            work = rebuild(&work, &smoothed);
        }
        if let Some(deg) = pre.elevation_mask_deg {
            let base_mask_epochs: Vec<ElevationMaskEpoch> = work
                .iter()
                .map(|e| ElevationMaskEpoch {
                    satellite_positions_m: if e.base_satellite_positions_m.is_empty() {
                        e.satellite_positions_m.clone()
                    } else {
                        e.base_satellite_positions_m.clone()
                    },
                })
                .collect();
            let base_result = apply_elevation_mask(config.base_m, &base_mask_epochs, deg)
                .expect("base elevation mask");
            work = work
                .iter()
                .zip(base_result.epochs.iter())
                .map(|(epoch, base_kept)| {
                    let keep: BTreeSet<&str> = base_kept
                        .kept_satellite_ids
                        .iter()
                        .map(String::as_str)
                        .collect();
                    let fobs = |obs: &[RtkArcObservation]| {
                        obs.iter()
                            .filter(|o| keep.contains(o.satellite_id.as_str()))
                            .cloned()
                            .collect::<Vec<_>>()
                    };
                    let fpos = |m: &BTreeMap<String, [f64; 3]>| {
                        m.iter()
                            .filter(|(s, _)| keep.contains(s.as_str()))
                            .map(|(s, p)| (s.clone(), *p))
                            .collect::<BTreeMap<_, _>>()
                    };
                    RtkArcEpoch {
                        base: fobs(&epoch.base),
                        rover: fobs(&epoch.rover),
                        satellite_positions_m: fpos(&epoch.satellite_positions_m),
                        base_satellite_positions_m: fpos(&epoch.base_satellite_positions_m),
                        rover_satellite_positions_m: fpos(&epoch.rover_satellite_positions_m),
                        velocity_mps: epoch.velocity_mps,
                        prediction_time_s: epoch.prediction_time_s,
                        prediction_epoch: epoch.prediction_epoch,
                    }
                })
                .collect();
            elevation_masked_sats = base_result.masked_satellite_ids;
        }

        let core = solve_prepared_arc(&work, config).expect("core solve");
        let mut masked: BTreeSet<_> = elevation_masked_sats.into_iter().collect();
        masked.extend(core.elevation_masked_sats.iter().cloned());
        let measurement_covariance = posterior_covariance(&core.final_state);
        RtkArcSolution {
            references: core.references,
            epochs: core.epochs,
            final_state: core.final_state,
            dropped_sats,
            split_cycle_slip_arcs,
            elevation_masked_sats: masked.into_iter().collect(),
            measurement_covariance,
        }
    }

    /// (a) Default (all-off) preprocessing makes the driver bit-identical to the
    /// bare core solve, with empty preprocessing metadata and the final-state
    /// posterior covariance.
    #[test]
    fn arc_default_preprocessing_is_bit_identical_to_core_solve() {
        let (epochs, config, _baseline, _tokens) = integer_arc();
        assert!(!config.preprocessing.is_active());

        let solution = solve_rtk_arc(&epochs, &config).expect("arc solves");
        let core = solve_prepared_arc(&epochs, &config).expect("core solves");

        assert_eq!(solution.references, core.references);
        assert_eq!(solution.epochs, core.epochs);
        assert_eq!(solution.final_state, core.final_state);

        assert!(solution.dropped_sats.is_empty());
        assert!(solution.split_cycle_slip_arcs.is_empty());
        assert!(solution.elevation_masked_sats.is_empty());
        assert_eq!(
            solution.measurement_covariance,
            posterior_covariance(&core.final_state)
        );
        assert!(!solution.measurement_covariance.is_empty());
    }

    #[test]
    fn non_excluding_mask_is_bit_identical_to_no_mask() {
        let (epochs, config, _baseline, _tokens) = integer_arc();
        let without_mask = solve_rtk_arc(&epochs, &config).expect("arc solves");
        let mut masked_config = config;
        masked_config.preprocessing.elevation_mask_deg = Some(-90.0);
        let with_mask = solve_rtk_arc(&epochs, &masked_config).expect("masked arc solves");

        assert_eq!(with_mask, without_mask);
        for (actual, expected) in with_mask
            .measurement_covariance
            .iter()
            .zip(&without_mask.measurement_covariance)
        {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn multi_system_non_excluding_mask_is_bit_identical_to_no_mask() {
        let (mut epochs, mut config, _, _) = integer_arc();
        let rename = |id: &str| {
            if ["G04", "G05", "G06"].contains(&id) {
                id.replacen('G', "E", 1)
            } else {
                id.to_string()
            }
        };
        for epoch in &mut epochs {
            for obs in epoch.base.iter_mut().chain(&mut epoch.rover) {
                obs.satellite_id = rename(&obs.satellite_id);
                obs.ambiguity_id = rename(&obs.ambiguity_id);
            }
            for positions in [
                &mut epoch.satellite_positions_m,
                &mut epoch.base_satellite_positions_m,
                &mut epoch.rover_satellite_positions_m,
            ] {
                *positions = positions
                    .iter()
                    .map(|(id, position)| (rename(id), *position))
                    .collect();
            }
        }
        config.wavelengths_m = config
            .wavelengths_m
            .iter()
            .map(|(id, value)| (rename(id), *value))
            .collect();
        config.offsets_m = config
            .offsets_m
            .iter()
            .map(|(id, value)| (rename(id), *value))
            .collect();
        let without = solve_rtk_arc(&epochs, &config).expect("multi-system arc");
        assert_eq!(without.references.len(), 2);
        let static_without = solve_static_rtk_arc(
            &epochs,
            &RtkStaticArcConfig {
                arc: config.clone(),
                opts: validated_opts(),
            },
        )
        .expect("multi-system static arc");
        config.preprocessing.elevation_mask_deg = Some(-90.0);
        let with = solve_rtk_arc(&epochs, &config).expect("multi-system masked arc");
        assert_eq!(with, without);
        let static_with = solve_static_rtk_arc(
            &epochs,
            &RtkStaticArcConfig {
                arc: config,
                opts: validated_opts(),
            },
        )
        .expect("multi-system masked static arc");
        assert_eq!(static_with, static_without);
        assert_eq!(
            with.measurement_covariance
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            without
                .measurement_covariance
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>()
        );
    }

    /// (b) Elevation masking: the driver equals manual composition, and the mask
    /// actually removes the lowest satellite (PRN 3 at ~66.6 deg).
    #[test]
    fn arc_elevation_mask_matches_manual_composition() {
        let (epochs, mut config, _baseline, _tokens) = integer_arc();
        config.preprocessing.elevation_mask_deg = Some(67.0);

        let solution = solve_rtk_arc(&epochs, &config).expect("arc solves");
        assert_eq!(solution, manual_solve(&epochs, &config));
        assert_eq!(solution.elevation_masked_sats, vec!["G03".to_string()]);
        for epoch in &solution.epochs {
            assert!(!epoch.used_satellite_ids.contains(&"G03".to_string()));
        }
    }

    /// (b) Cycle-slip split: an LLI flag mid-arc splits that satellite's arc; the
    /// driver equals manual composition and surfaces the split metadata.
    #[test]
    fn arc_cycle_slip_split_matches_manual_composition() {
        let (mut epochs, mut config, _baseline, _tokens) = integer_arc();
        // Flag a rover loss-of-lock on PRN 2 at epoch 3 (LLI bit 0).
        for obs in &mut epochs[3].rover {
            if obs.satellite_id == "G02" {
                obs.lli = Some(1);
            }
        }
        config.preprocessing.cycle_slip = Some(CycleSlipPolicy::SplitArc);
        augment_scale_for_cycle_slip(&epochs, &mut config);

        let solution = solve_rtk_arc(&epochs, &config).expect("arc solves");
        assert_eq!(solution, manual_solve(&epochs, &config));
        assert!(
            !solution.split_cycle_slip_arcs.is_empty(),
            "the LLI flag should split PRN 2's arc"
        );
        assert!(solution.dropped_sats.is_empty());
        assert!(solution
            .split_cycle_slip_arcs
            .iter()
            .any(|arc| arc.satellite_id == "G02"));
    }

    #[test]
    fn arc_cycle_slip_split_derives_scale_for_split_ids() {
        let (mut epochs, mut config, baseline, _tokens) = integer_arc();
        for obs in &mut epochs[3].rover {
            if obs.satellite_id == "G02" {
                obs.lli = Some(1);
            }
        }
        config.preprocessing.cycle_slip = Some(CycleSlipPolicy::SplitArc);

        let solution = solve_rtk_arc(&epochs, &config).expect("arc solves");
        assert!(!solution.split_cycle_slip_arcs.is_empty());
        let final_baseline = solution
            .epochs
            .last()
            .expect("final epoch")
            .reported_baseline_m;
        assert!(norm3(sub3(final_baseline, baseline)) < 1.0e-6);
    }

    /// (b) Cycle-slip drop: the same LLI flag under the drop policy removes the
    /// satellite entirely; the driver equals manual composition.
    #[test]
    fn arc_cycle_slip_drop_matches_manual_composition() {
        let (mut epochs, mut config, _baseline, _tokens) = integer_arc();
        for obs in &mut epochs[3].rover {
            if obs.satellite_id == "G02" {
                obs.lli = Some(1);
            }
        }
        config.preprocessing.cycle_slip = Some(CycleSlipPolicy::DropSatellite);

        let solution = solve_rtk_arc(&epochs, &config).expect("arc solves");
        assert_eq!(solution, manual_solve(&epochs, &config));
        assert_eq!(solution.dropped_sats, vec!["G02".to_string()]);
        assert!(solution.split_cycle_slip_arcs.is_empty());
        for epoch in &solution.epochs {
            assert!(!epoch.used_satellite_ids.contains(&"G02".to_string()));
        }
    }

    /// (b) Hatch code smoothing: with deterministic per-epoch code noise, the
    /// driver equals manual composition and the smoothed solve differs from the
    /// unsmoothed one (proving the stage took effect).
    #[test]
    fn arc_code_smoothing_matches_manual_composition() {
        let (mut epochs, mut config, _baseline, _tokens) = integer_arc();
        // Inject an alternating per-epoch code bias on the rover so Hatch
        // smoothing has a noise to reduce (clean geometry alone is a no-op).
        for (idx, epoch) in epochs.iter_mut().enumerate() {
            let bias = if idx % 2 == 0 { 0.5 } else { -0.5 };
            for obs in &mut epoch.rover {
                obs.code_m += bias;
            }
        }

        let unsmoothed = solve_rtk_arc(&epochs, &config).expect("unsmoothed arc");

        config.preprocessing.hatch_window_cap = Some(100);
        let solution = solve_rtk_arc(&epochs, &config).expect("smoothed arc");
        assert_eq!(solution, manual_solve(&epochs, &config));
        assert_ne!(
            solution.epochs.last().expect("epoch").reported_baseline_m,
            unsmoothed.epochs.last().expect("epoch").reported_baseline_m,
            "Hatch smoothing should change the reported baseline"
        );
    }

    fn flat_calibration(pco_neu_m: [f64; 3]) -> ReceiverAntennaCalibration {
        ReceiverAntennaCalibration {
            pco_neu_m,
            // Zero PCV across the full zenith span so only the PCO contributes.
            noazi_pcv_m: vec![(0.0, 0.0), (90.0, 0.0)],
            azi_pcv_m: Vec::new(),
        }
    }

    /// (b) Receiver-antenna corrections: configured on `update_opts`, applied by
    /// the per-epoch update. Enabling a nonzero rover PCO changes the reported
    /// baseline, and the driver equals manual composition either way.
    #[test]
    fn arc_receiver_antenna_corrections_are_applied() {
        let (epochs, mut config, _baseline, _tokens) = integer_arc();
        let baseline_none = solve_rtk_arc(&epochs, &config).expect("no-antenna arc");

        config.update_opts.receiver_antenna_corrections = Some(ReceiverAntennaCorrections {
            base: flat_calibration([0.0, 0.0, 0.0]),
            rover: flat_calibration([0.0, 0.0, 0.1]),
        });
        let solution = solve_rtk_arc(&epochs, &config).expect("antenna arc");

        // The driver still equals manual composition with the antenna model on.
        assert_eq!(solution, manual_solve(&epochs, &config));
        // And the antenna correction actually moved the reported baseline.
        assert_ne!(
            solution.epochs.last().expect("epoch").reported_baseline_m,
            baseline_none
                .epochs
                .last()
                .expect("epoch")
                .reported_baseline_m,
            "receiver-antenna PCO should change the reported baseline"
        );
    }

    /// (b) Combined preprocessing (cycle-slip split + smoothing + mask) still
    /// equals manual composition stage-for-stage.
    #[test]
    fn arc_combined_preprocessing_matches_manual_composition() {
        let (mut epochs, mut config, _baseline, _tokens) = integer_arc();
        for obs in &mut epochs[2].rover {
            if obs.satellite_id == "G04" {
                obs.lli = Some(1);
            }
        }
        for (idx, epoch) in epochs.iter_mut().enumerate() {
            let bias = if idx % 2 == 0 { 0.25 } else { -0.25 };
            for obs in &mut epoch.rover {
                obs.code_m += bias;
            }
        }
        config.preprocessing = RtkArcPreprocessing {
            cycle_slip: Some(CycleSlipPolicy::SplitArc),
            hatch_window_cap: Some(50),
            elevation_mask_deg: Some(67.0),
        };
        augment_scale_for_cycle_slip(&epochs, &mut config);

        let solution = solve_rtk_arc(&epochs, &config).expect("arc solves");
        assert_eq!(solution, manual_solve(&epochs, &config));
        assert_eq!(solution.elevation_masked_sats, vec!["G03".to_string()]);
        assert!(!solution.split_cycle_slip_arcs.is_empty());
    }

    fn validated_opts() -> ValidatedFixedSolveOpts {
        ValidatedFixedSolveOpts {
            float: FloatSolveOpts::default(),
            fixed: FixedSolveOpts::default(),
            residual: ResidualValidationOpts {
                threshold_sigma: None,
                max_exclusions: 0,
            },
        }
    }

    fn manual_static_arc(
        epochs: &[RtkArcEpoch],
        config: &RtkStaticArcConfig,
    ) -> RtkStaticArcSolution {
        let (prepared_epochs, dropped_sats, split_cycle_slip_arcs, elevation_masked_sats) =
            if config.arc.preprocessing.is_active() {
                preprocess_arc(epochs, &config.arc).expect("preprocess")
            } else {
                (Vec::new(), Vec::new(), Vec::new(), Vec::new())
            };
        let solve_input: &[RtkArcEpoch] = if config.arc.preprocessing.is_active() {
            &prepared_epochs
        } else {
            epochs
        };
        let batch = build_static_batch_arc(solve_input, &config.arc).expect("batch");
        let antenna = config.arc.update_opts.receiver_antenna_corrections.as_ref();
        let float_solution = solve_float_baseline(
            &batch.epochs,
            config.arc.base_m,
            &batch.ambiguity_ids,
            config.arc.initial_baseline_m,
            &config.arc.model,
            config.opts.float,
            antenna,
        )
        .expect("float");
        let fixed_solution = solve_fixed_baseline_validated(
            &batch.epochs,
            config.arc.base_m,
            AmbiguitySet {
                ids: &batch.ambiguity_ids,
                satellites: &batch.ambiguity_satellites,
                scale: AmbiguityScale {
                    wavelengths_m: &config.arc.wavelengths_m,
                    offsets_m: &config.arc.offsets_m,
                },
                float_only_systems: &config.arc.update_opts.float_only_systems,
            },
            config.arc.initial_baseline_m,
            &config.arc.model,
            config.opts,
            antenna,
        )
        .expect("fixed");
        RtkStaticArcSolution {
            geometry_quality: float_solution.geometry_quality,
            references: batch.references,
            ambiguity_ids: batch.ambiguity_ids,
            ambiguity_satellites: batch.ambiguity_satellites,
            float_solution,
            fixed_solution,
            dropped_sats,
            split_cycle_slip_arcs,
            elevation_masked_sats,
            elevation_mask_converged: true,
        }
    }

    fn assert_f64_bits(got: f64, expected: f64) {
        assert_eq!(got.to_bits(), expected.to_bits());
    }

    fn assert_vec3_bits(got: [f64; 3], expected: [f64; 3]) {
        for (got, expected) in got.into_iter().zip(expected) {
            assert_f64_bits(got, expected);
        }
    }

    fn assert_f64_vec_bits(got: &[f64], expected: &[f64]) {
        assert_eq!(got.len(), expected.len());
        for (got, expected) in got.iter().zip(expected) {
            assert_f64_bits(*got, *expected);
        }
    }

    fn assert_named_f64_bits(got: &[(String, f64)], expected: &[(String, f64)]) {
        assert_eq!(got.len(), expected.len());
        for ((got_id, got_value), (expected_id, expected_value)) in got.iter().zip(expected) {
            assert_eq!(got_id, expected_id);
            assert_f64_bits(*got_value, *expected_value);
        }
    }

    fn assert_float_solution_bits(got: &FloatBaselineSolution, expected: &FloatBaselineSolution) {
        assert_vec3_bits(got.baseline_m, expected.baseline_m);
        assert_named_f64_bits(&got.ambiguities_m, &expected.ambiguities_m);
        assert_f64_vec_bits(
            &got.ambiguity_covariance_m,
            &expected.ambiguity_covariance_m,
        );
        assert_f64_vec_bits(
            &got.ambiguity_covariance_inverse_m,
            &expected.ambiguity_covariance_inverse_m,
        );
        assert_f64_bits(got.code_rms_m, expected.code_rms_m);
        assert_f64_bits(got.phase_rms_m, expected.phase_rms_m);
        assert_f64_bits(got.weighted_rms_m, expected.weighted_rms_m);
        assert_eq!(got.iterations, expected.iterations);
        assert_eq!(got.converged, expected.converged);
        assert_eq!(got.status, expected.status);
        assert_eq!(got.n_observations, expected.n_observations);
        assert_eq!(got.geometry_quality, expected.geometry_quality);
        assert_eq!(got.residuals.len(), expected.residuals.len());
        for (got, expected) in got.residuals.iter().zip(&expected.residuals) {
            assert_eq!(got.epoch_index, expected.epoch_index);
            assert_eq!(got.satellite_id, expected.satellite_id);
            assert_eq!(got.reference_satellite_id, expected.reference_satellite_id);
            assert_eq!(got.ambiguity_id, expected.ambiguity_id);
            assert_f64_bits(got.code_m, expected.code_m);
            assert_f64_bits(got.phase_m, expected.phase_m);
            assert_f64_bits(got.code_sigma_m, expected.code_sigma_m);
            assert_f64_bits(got.phase_sigma_m, expected.phase_sigma_m);
            assert_f64_bits(got.code_normalized, expected.code_normalized);
            assert_f64_bits(got.phase_normalized, expected.phase_normalized);
        }
    }

    fn assert_fixed_solution_bits(got: &FixedBaselineSolution, expected: &FixedBaselineSolution) {
        assert_vec3_bits(got.baseline_m, expected.baseline_m);
        assert_named_f64_bits(&got.free_ambiguities_m, &expected.free_ambiguities_m);
        assert_eq!(
            got.fixed_ambiguities_cycles,
            expected.fixed_ambiguities_cycles
        );
        assert_named_f64_bits(&got.fixed_ambiguities_m, &expected.fixed_ambiguities_m);
        assert_eq!(got.search, expected.search);
        assert_eq!(got.iterations, expected.iterations);
        assert_eq!(got.converged, expected.converged);
        assert_eq!(got.status, expected.status);
        assert_f64_bits(got.code_rms_m, expected.code_rms_m);
        assert_f64_bits(got.phase_rms_m, expected.phase_rms_m);
        assert_f64_bits(got.weighted_rms_m, expected.weighted_rms_m);
        assert_eq!(got.n_observations, expected.n_observations);
        assert_eq!(got.residuals.len(), expected.residuals.len());
        for (got, expected) in got.residuals.iter().zip(&expected.residuals) {
            assert_eq!(got.epoch_index, expected.epoch_index);
            assert_eq!(got.satellite_id, expected.satellite_id);
            assert_eq!(got.reference_satellite_id, expected.reference_satellite_id);
            assert_eq!(got.ambiguity_id, expected.ambiguity_id);
            assert_f64_bits(got.code_m, expected.code_m);
            assert_f64_bits(got.phase_m, expected.phase_m);
            assert_f64_bits(got.code_sigma_m, expected.code_sigma_m);
            assert_f64_bits(got.phase_sigma_m, expected.phase_sigma_m);
            assert_f64_bits(got.code_normalized, expected.code_normalized);
            assert_f64_bits(got.phase_normalized, expected.phase_normalized);
        }
    }

    fn assert_static_solution_bits(got: &RtkStaticArcSolution, expected: &RtkStaticArcSolution) {
        assert_eq!(got.references, expected.references);
        assert_eq!(got.ambiguity_ids, expected.ambiguity_ids);
        assert_eq!(got.ambiguity_satellites, expected.ambiguity_satellites);
        assert_float_solution_bits(&got.float_solution, &expected.float_solution);
        assert_float_solution_bits(
            &got.fixed_solution.float_solution,
            &expected.fixed_solution.float_solution,
        );
        assert_fixed_solution_bits(
            &got.fixed_solution.fixed_solution,
            &expected.fixed_solution.fixed_solution,
        );
        assert_eq!(
            got.fixed_solution.residual_validation,
            expected.fixed_solution.residual_validation
        );
        assert_eq!(
            got.fixed_solution.ambiguity_ids,
            expected.fixed_solution.ambiguity_ids
        );
        assert_eq!(
            got.fixed_solution.ambiguity_satellites,
            expected.fixed_solution.ambiguity_satellites
        );
        assert_eq!(got.dropped_sats, expected.dropped_sats);
        assert_eq!(got.split_cycle_slip_arcs, expected.split_cycle_slip_arcs);
        assert_eq!(got.elevation_masked_sats, expected.elevation_masked_sats);
        assert_eq!(
            got.elevation_mask_converged,
            expected.elevation_mask_converged
        );
    }

    #[test]
    fn static_batch_driver_matches_manual_float_and_fixed_primitives() {
        let (epochs, mut arc_config, _baseline, _tokens) = integer_arc();
        arc_config.preprocessing.elevation_mask_deg = Some(67.0);
        let static_config = RtkStaticArcConfig {
            arc: arc_config,
            opts: validated_opts(),
        };

        let driver = solve_static_rtk_arc(&epochs, &static_config).expect("static driver");
        let manual = manual_static_arc(&epochs, &static_config);

        assert_eq!(driver, manual);
        assert_static_solution_bits(&driver, &manual);
        assert_eq!(driver.elevation_masked_sats, vec!["G03".to_string()]);
    }

    #[test]
    fn static_batch_rover_mask_uses_the_converged_rover_position() {
        let (epochs, mut arc_config, baseline, tokens) = integer_arc();
        let rover = std::array::from_fn(|axis| arc_config.base_m[axis] + baseline[axis]);
        let (satellite, base_el, rover_el) = tokens
            .iter()
            .filter_map(|satellite| {
                let position = epochs[0].satellite_positions_m[satellite];
                let (_, base_el) =
                    crate::estimation::substrate::frames::satazel(arc_config.base_m, position);
                let (_, rover_el) = crate::estimation::substrate::frames::satazel(rover, position);
                (rover_el < base_el).then_some((satellite.clone(), base_el, rover_el))
            })
            .max_by(|left, right| (left.1 - left.2).total_cmp(&(right.1 - right.2)))
            .expect("the generated arc has a satellite lower at the rover");
        arc_config.preprocessing.elevation_mask_deg = Some(0.5 * (base_el + rover_el) * RAD_TO_DEG);
        let solution = solve_static_rtk_arc(
            &epochs,
            &RtkStaticArcConfig {
                arc: arc_config,
                opts: validated_opts(),
            },
        )
        .expect("static arc solves after rover screening");

        assert!(solution.elevation_masked_sats.contains(&satellite));
        assert!(!solution
            .ambiguity_satellites
            .values()
            .any(|sat| sat == &satellite));
    }

    #[test]
    fn static_mask_iteration_cap_reports_last_completed_solve() {
        let (epochs, mut arc_config, baseline, tokens) = integer_arc();
        let rover = std::array::from_fn(|axis| arc_config.base_m[axis] + baseline[axis]);
        let (_, mask_rad) = tokens
            .iter()
            .filter_map(|satellite| {
                let position = epochs[0].satellite_positions_m[satellite];
                let (_, base_el) =
                    crate::estimation::substrate::frames::satazel(arc_config.base_m, position);
                let (_, rover_el) = crate::estimation::substrate::frames::satazel(rover, position);
                (rover_el < base_el).then_some((satellite, 0.5 * (base_el + rover_el)))
            })
            .next()
            .expect("a rover-lower satellite");
        arc_config.preprocessing.elevation_mask_deg = Some(mask_rad * RAD_TO_DEG);
        let mut options = validated_opts();
        options.float.max_iterations = 1;
        let solution = solve_static_rtk_arc(
            &epochs,
            &RtkStaticArcConfig {
                arc: arc_config,
                opts: options,
            },
        )
        .expect("last completed static solve");
        assert!(!solution.elevation_mask_converged);
        assert!(solution.elevation_masked_sats.is_empty());
    }

    #[test]
    fn reference_change_preserves_single_gauge_dd_and_held_integers() {
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/rtk_reference_gauge.json"
        ))
        .unwrap();
        let mut state = FilterState::new(
            BTreeMap::from([("G".to_string(), "G01".to_string())]),
            [0.0; 3],
            1.0,
            1.0,
        )
        .unwrap();
        for (index, id) in ["G01", "G02", "G03"].iter().enumerate() {
            state.ensure_ambiguity(id, oracle["estimate"][index].as_f64().unwrap());
        }
        let n = state.dim();
        for i in 0..3 {
            for j in 0..3 {
                state.information[(3 + i) * n + 3 + j] =
                    oracle["information"][i * 3 + j].as_f64().unwrap();
            }
        }
        for (id, cycles) in oracle["fixed_cycles"].as_object().unwrap() {
            state
                .fixed_cycles
                .insert(id.clone(), cycles.as_i64().unwrap());
            state
                .fixed_m
                .insert(id.clone(), cycles.as_i64().unwrap() as f64);
        }
        let reference = super::super::model::SatMeas {
            sat: "G02".to_string(),
            sd_ambiguity_id: "G02".to_string(),
            base_code_m: 0.0,
            base_phase_m: 0.0,
            rover_code_m: 0.0,
            rover_phase_m: 0.0,
            base_tx_pos: [0.0; 3],
            rover_tx_pos: [0.0; 3],
            pos: [0.0; 3],
        };
        refer_to_epoch_references(
            &mut state,
            &Epoch {
                references: vec![reference],
                nonref: Vec::new(),
                velocity_mps: None,
                dt_s: 0.0,
            },
        )
        .unwrap();
        assert_eq!(state.references["G"], "G02");
        // The next update pins the new reference again.  It must reinforce
        // the same gauge, leaving the physical difference covariance intact.
        state.information[4 * n + 4] += oracle["gauge_information"].as_f64().unwrap();
        let rows: Vec<Vec<_>> = state
            .information
            .chunks(n)
            .map(|row| row.to_vec())
            .collect();
        let covariance = invert_matrix_first_tie(&rows).unwrap();
        let variance = covariance[3][3] + covariance[4][4] - 2.0 * covariance[3][4];
        let expected_variance = oracle["dd_variance"].as_f64().unwrap();
        assert!((variance - expected_variance).abs() <= 64.0 * f64::EPSILON * expected_variance);
        assert!(
            (state.sd_ambiguities_m[0]
                - state.sd_ambiguities_m[1]
                - oracle["dd"].as_f64().unwrap())
            .abs()
                <= 4.0 * f64::EPSILON
        );
        for (id, cycles) in oracle["referred_fixed_cycles"].as_object().unwrap() {
            assert_eq!(state.fixed_cycles.get(id), cycles.as_i64().as_ref());
        }
        assert_eq!(state.fixed_cycles.len(), 2);
    }

    fn hold_transport_fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../../tests/fixtures/rtk_hold_transport.json")).unwrap()
    }

    #[test]
    fn pending_holds_survive_change_to_unheld_reference() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/rtk_reference_gauge.json"
        ))
        .unwrap();
        for case in fixture["pending_holds"].as_array().unwrap() {
            let mut state = FilterState::new(
                BTreeMap::from([("G".into(), "G01".into())]),
                std::array::from_fn(|i| case["estimate"][i].as_f64().unwrap()),
                1.0,
                1.0,
            )
            .unwrap();
            for i in 0..5 {
                state.ensure_ambiguity(
                    &format!("G{:02}", i + 1),
                    case["estimate"][i + 3].as_f64().unwrap(),
                );
            }
            state.information = case["information"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            state.fixed_cycles = case["fixed_cycles"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(id, n)| (id.clone(), n.as_i64().unwrap()))
                .collect();
            state.fixed_m = case["fixed_m"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(id, m)| (id.clone(), m.as_f64().unwrap()))
                .collect();
            let epoch = Epoch {
                references: vec![SatMeas {
                    sat: "G04".into(),
                    sd_ambiguity_id: "G04".into(),
                    base_code_m: 0.0,
                    rover_code_m: 0.0,
                    base_phase_m: 0.0,
                    rover_phase_m: 0.0,
                    base_tx_pos: [0.0; 3],
                    rover_tx_pos: [0.0; 3],
                    pos: [0.0; 3],
                }],
                nonref: Vec::new(),
                velocity_mps: None,
                dt_s: 0.0,
            };
            let references = state.references.clone();
            let wavelengths = (1..=5).map(|i| (format!("G{i:02}"), 0.19)).collect();
            let offsets = BTreeMap::new();
            let pending = state.fixed_cycles.keys().cloned().collect();
            let context = ArcReferenceContext {
                references: &references,
                wavelengths: &wavelengths,
                offsets: &offsets,
                pending: &pending,
                hold_sigma_m: case["hold_sigma_m"].as_f64().unwrap(),
            };
            let released = context.prepare(&mut state, &epoch).unwrap();
            assert_eq!(released.len(), pending.len());
            let matrix: Vec<Vec<_>> = state
                .information
                .chunks(state.dim())
                .map(|r| r.to_vec())
                .collect();
            let covariance = invert_matrix_first_tie(&matrix).unwrap();
            // Gaussian elimination, inversion and gauge transport cost less
            // than 16*n^3 operations. Use gamma_(16*n^3), unit roundoff
            // eps/2, and the maximum kappa_1 of the independent input,
            // constrained and transported normals; the forward bound is
            // gamma*kappa/(1-gamma*kappa), not its first-order approximation.
            let n = case["dimension"].as_u64().unwrap() as f64;
            let rounding = 16.0 * n * n * n * (0.5 * f64::EPSILON);
            let amplification =
                rounding / (1.0 - rounding) * case["condition_one"].as_f64().unwrap();
            assert!(amplification < 0.01);
            let bound = amplification / (1.0 - amplification);
            let sigma = libm::sqrt(covariance[0][0]);
            let expected = case["baseline_sigma_m"].as_f64().unwrap();
            assert!(
                (sigma - expected).abs() <= bound * expected,
                "sigma {sigma} expected {expected}"
            );
            let dd = state.sd_ambiguities_m[1] - state.sd_ambiguities_m[3];
            assert!(
                (dd - case["dd_m"].as_f64().unwrap()).abs()
                    <= bound * case["state_scale_m"].as_f64().unwrap()
            );
            let variance = covariance[4][4] + covariance[6][6] - 2.0 * covariance[4][6];
            assert!(
                (variance - case["dd_variance_m2"].as_f64().unwrap()).abs()
                    <= bound * variance.abs()
            );
            assert!(state.fixed_cycles.is_empty());
        }
    }

    fn hold_transport_arc(
        case: &serde_json::Value,
        multi: bool,
    ) -> (Vec<RtkArcEpoch>, RtkArcConfig) {
        let fixture = hold_transport_fixture();
        let vector = |v: &serde_json::Value| std::array::from_fn(|i| v[i].as_f64().unwrap());
        let systems = if multi { vec!["G", "E"] } else { vec!["G"] };
        let mut config = config(BTreeMap::new(), BTreeMap::new());
        config.base_m = vector(&fixture["base_m"]);
        config.initial_baseline_m = std::array::from_fn(|i| {
            case["epochs"][0]["rover"][i].as_f64().unwrap() - config.base_m[i]
        });
        config.reference = BaselineReferenceSelection::PerSystem(
            systems
                .iter()
                .map(|s| (s.to_string(), format!("{s}01")))
                .collect(),
        );
        config.model.sagnac = false;
        config.preprocessing.elevation_mask_deg = fixture["mask_deg"].as_f64();
        config.update_opts.dynamics_model = DynamicsModel::VelocityPropagated;
        for system in &systems {
            for prn in 2..=6 {
                config.wavelengths_m.insert(
                    format!("{system}{prn:02}"),
                    case["wavelength_m"].as_f64().unwrap(),
                );
                config.offsets_m.insert(
                    format!("{system}{prn:02}"),
                    case["offset_m"].as_f64().unwrap(),
                );
            }
        }
        let epochs = case["epochs"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(index, row)| {
                let mut positions = BTreeMap::new();
                let mut base = Vec::new();
                let mut rover = Vec::new();
                for system in &systems {
                    for (satellite, obs) in
                        row["observations"].as_array().unwrap().iter().enumerate()
                    {
                        let id = format!("{system}{:02}", satellite + 1);
                        positions.insert(id.clone(), vector(&row["positions"][satellite]));
                        base.push(observation(
                            &id,
                            obs["base_code"].as_f64().unwrap(),
                            obs["base_phase"].as_f64().unwrap(),
                        ));
                        rover.push(observation(
                            &id,
                            obs["rover_code"].as_f64().unwrap(),
                            obs["rover_phase"].as_f64().unwrap(),
                        ));
                    }
                }
                RtkArcEpoch {
                    base,
                    rover,
                    satellite_positions_m: positions,
                    base_satellite_positions_m: BTreeMap::new(),
                    rover_satellite_positions_m: BTreeMap::new(),
                    velocity_mps: Some(std::array::from_fn(|axis| {
                        if index == 0 {
                            0.0
                        } else {
                            (row["rover"][axis].as_f64().unwrap()
                                - case["epochs"][index - 1]["rover"][axis].as_f64().unwrap())
                                / 30.0
                        }
                    })),
                    prediction_time_s: Some(index as f64 * 30.0),
                    prediction_epoch: None,
                }
            })
            .collect();
        (epochs, config)
    }

    #[test]
    fn masked_reference_holds_keep_exact_integers_and_relative_offsets() {
        let fixture = hold_transport_fixture();
        for case in fixture["scenarios"].as_array().unwrap() {
            for multi in [false, true] {
                let (epochs, config) = hold_transport_arc(case, multi);
                for end in [2, 3] {
                    let result = solve_rtk_arc(&epochs[..end], &config).unwrap();
                    let expected = &case["epochs"][end - 1];
                    let mut references = vec![expected["reference"].as_str().unwrap().to_string()];
                    if multi {
                        references.insert(0, references[0].replacen('G', "E", 1));
                    }
                    assert_eq!(result.epochs[end - 1].reference_satellite_ids, references);
                    assert!(result.epochs[1].newly_fixed.iter().any(|id| id == "G02"));
                    for (id, value) in expected["expected"].as_object().unwrap() {
                        for system in if multi { vec!["G", "E"] } else { vec!["G"] } {
                            let id = id.replacen('G', system, 1);
                            if let Some(&cycles) = result.final_state.fixed_cycles.get(&id) {
                                assert_eq!(cycles, value["cycles"].as_i64().unwrap(), "{id}");
                                // Two products and a subtraction, allowing four ulps at metre scale.
                                assert!(
                                    (result.final_state.fixed_m[&id]
                                        - value["metres"].as_f64().unwrap())
                                    .abs()
                                        <= 4.0
                                            * f64::EPSILON
                                            * result.final_state.fixed_m[&id].abs().max(1.0)
                                );
                            } else {
                                assert!(id.ends_with("02") && end == 3, "missing hold {id}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn wide_lane_masked_reference_holds_survive_and_return() {
        let fixture = hold_transport_fixture();
        let case = &fixture["scenarios"][0];
        for multi in [false, true] {
            let (single, config) = hold_transport_arc(case, multi);
            let (template, mut wide_lane) = dual_arc();
            wide_lane.base_m = config.base_m;
            wide_lane.reference = config.reference.clone();
            let dual: Vec<_> = single
                .iter()
                .enumerate()
                .map(|(index, single)| {
                    let mut epoch = template[0].clone();
                    epoch.satellite_positions_m = single.satellite_positions_m.clone();
                    epoch.velocity_mps = single.velocity_mps;
                    epoch.prediction_time_s = single.prediction_time_s;
                    epoch.gap_time_s = single.prediction_time_s;
                    epoch.epoch_sort_key = Some(index.to_string());
                    epoch.observations = single
                        .base
                        .iter()
                        .map(|obs| {
                            let prn: usize = obs.satellite_id[1..].parse().unwrap();
                            let row = &case["epochs"][index]["observations"][prn - 1];
                            let make = |side: &str| RtkDualFrequencyObservation {
                                ambiguity_id: obs.satellite_id.clone(),
                                p1_m: row[format!("{side}_code")].as_f64().unwrap(),
                                p2_m: row[format!("{side}_code")].as_f64().unwrap(),
                                phi1_cycles: row[format!("{side}_phi1")].as_f64().unwrap(),
                                phi2_cycles: row[format!("{side}_phi2")].as_f64().unwrap(),
                                f1_hz: F_L1_HZ,
                                f2_hz: F_L2_HZ,
                                lli1: None,
                                lli2: None,
                            };
                            RtkDualFrequencySatelliteObservation {
                                satellite_id: obs.satellite_id.clone(),
                                base: make("base"),
                                rover: make("rover"),
                            }
                        })
                        .collect();
                    epoch
                })
                .collect();
            let combined = RtkWideLaneFixedArcConfig {
                ionosphere_free: RtkIonosphereFreeArcConfig {
                    base_m: config.base_m,
                    initial_baseline_m: config.initial_baseline_m,
                    reference: config.reference.clone(),
                    apply_troposphere: false,
                },
                wide_lane,
                solve: RtkWideLaneFixedArcSolveConfig::Sequential(config),
            };
            for end in [2, 3] {
                let result = solve_wide_lane_fixed_rtk_arc(&dual[..end], &combined).unwrap();
                let RtkWideLaneFixedArcSolution::Sequential(result) = result else {
                    panic!("sequential");
                };
                assert!(result.solution.epochs[0].integer_fixed);
                for reference in result.ionosphere_free.references.values() {
                    assert_eq!(
                        result.ionosphere_free.wavelengths_m[reference],
                        fixture["narrow_lane_wavelength_m"].as_f64().unwrap()
                    );
                    assert_eq!(result.ionosphere_free.offsets_m[reference], 0.0);
                }
                assert!(result.solution.epochs[1].search.is_some());
                let expected = &case["epochs"][end - 1];
                assert_eq!(
                    result.solution.epochs[end - 1]
                        .reference_satellite_ids
                        .iter()
                        .map(|id| &id[1..])
                        .collect::<Vec<_>>(),
                    vec![&expected["reference"].as_str().unwrap()[1..]; if multi { 2 } else { 1 }]
                );
                for (id, &cycles) in &result.solution.final_state.fixed_cycles {
                    let id = id.replacen('E', "G", 1);
                    assert_eq!(cycles, expected["expected"][id]["cycles"].as_i64().unwrap());
                }
            }
        }
    }

    #[test]
    fn fdma_reference_change_releases_holds_and_reports_float_only_epoch() {
        let fixture = hold_transport_fixture();
        let case = &fixture["fdma_scenario"];
        let (epochs, mut config) = hold_transport_arc(case, false);
        config.wavelengths_m = case["wavelengths_m"]
            .as_object()
            .unwrap()
            .iter()
            .map(|(id, value)| (id.clone(), value.as_f64().unwrap()))
            .collect();
        let first = solve_rtk_arc(&epochs[..1], &config).unwrap();
        assert!(first.final_state.fixed_cycles.len() >= 2);
        let changed = solve_rtk_arc(&epochs[..2], &config).unwrap();
        assert_eq!(
            changed.epochs[1].reference_satellite_ids,
            [case["epochs"][1]["reference"].as_str().unwrap()]
        );
        assert!(changed.final_state.fixed_cycles.is_empty());
        assert!(changed.epochs[1].search.is_none());
        assert_eq!(
            changed.epochs[1]
                .released_holds
                .iter()
                .map(|hold| &hold.ambiguity_id)
                .collect::<Vec<_>>(),
            first.final_state.fixed_cycles.keys().collect::<Vec<_>>()
        );
        assert!(changed.epochs[1]
            .released_holds
            .iter()
            .all(|hold| hold.reason == RtkHoldReleaseReason::IncompatibleWavelengths));
        let returned = solve_rtk_arc(&epochs, &config).unwrap();
        assert_eq!(returned.epochs[2].reference_satellite_ids, ["G01"]);
        assert!(returned.epochs[2].integer_fixed);
    }

    #[test]
    fn newly_fixed_holds_are_reported_when_an_unheld_reference_is_promoted() {
        let fixture = hold_transport_fixture();
        let (mut epochs, config) = hold_transport_arc(&fixture["scenarios"][0], false);
        let new_reference = fixture["scenarios"][0]["epochs"][1]["reference"]
            .as_str()
            .unwrap();
        epochs[0]
            .base
            .retain(|obs| obs.satellite_id != new_reference);
        epochs[0]
            .rover
            .retain(|obs| obs.satellite_id != new_reference);
        let first = solve_rtk_arc(&epochs[..1], &config).unwrap();
        assert!(!first.final_state.fixed_cycles.contains_key(new_reference));
        assert!(first.final_state.fixed_cycles.len() >= 2);
        let changed = solve_rtk_arc(&epochs[..2], &config).unwrap();
        assert_eq!(changed.epochs[1].reference_satellite_ids, [new_reference]);
        assert_eq!(
            changed.epochs[1]
                .released_holds
                .iter()
                .map(|hold| &hold.ambiguity_id)
                .collect::<Vec<_>>(),
            first.final_state.fixed_cycles.keys().collect::<Vec<_>>()
        );
        assert!(changed.epochs[1]
            .released_holds
            .iter()
            .all(|hold| hold.reason == RtkHoldReleaseReason::UnheldReference));
    }

    #[test]
    fn reference_change_requires_both_ambiguity_columns() {
        let fixture = hold_transport_fixture();
        let (epochs, _) = hold_transport_arc(&fixture["scenarios"][0], false);
        let normalized = normalize_epoch(&epochs[0]);
        let epoch = build_epoch(
            0,
            &normalized,
            &BTreeMap::from([("G".into(), "G04".into())]),
            &BTreeSet::from(["G04"]),
            None,
            0.0,
        )
        .unwrap();
        let mut state = FilterState::new(
            BTreeMap::from([("G".into(), "G01".into())]),
            [0.0; 3],
            1.0,
            1.0,
        )
        .unwrap();
        state.ensure_ambiguity("G01", 0.0);
        assert_eq!(
            refer_to_epoch_references(&mut state, &epoch),
            Err(UpdateError::MissingAmbiguityColumn("G04".into()))
        );
        assert_eq!(state.references["G"], "G01");
    }

    #[test]
    fn rtk_mask_requires_both_base_and_rover_to_pass() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/rtklib_rtk_mask_oracle.json"
        ))
        .expect("valid RTKLIB mask fixture");
        let vector = |value: &serde_json::Value| {
            let values = value.as_array().expect("three-vector");
            [
                values[0].as_f64().unwrap(),
                values[1].as_f64().unwrap(),
                values[2].as_f64().unwrap(),
            ]
        };
        let base = vector(&fixture["base_ecef_m"]);
        let rovers: Vec<_> = fixture["rover_ecef_m"]
            .as_array()
            .unwrap()
            .iter()
            .map(vector)
            .collect();
        let wavelength_m = C_M_S / F_L1_HZ;
        let epochs: Vec<_> = rovers
            .iter()
            .zip(fixture["epochs"].as_array().unwrap())
            .enumerate()
            .map(|(epoch_index, (&rover, oracle))| {
                let positions: BTreeMap<_, _> = oracle["satellite_ecef_m"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (format!("G{:02}", index + 1), vector(value)))
                    .collect();
                let mut base_observations = Vec::new();
                let mut rover_observations = Vec::new();
                for (index, (satellite, position)) in positions.iter().enumerate() {
                    let base_code = super::super::model::geometric_range_m(*position, base, false);
                    let rover_code =
                        super::super::model::geometric_range_m(*position, rover, false);
                    base_observations.push(observation(satellite, base_code, base_code));
                    let mut rover_observation = observation(
                        satellite,
                        rover_code,
                        rover_code + index as f64 * wavelength_m,
                    );
                    if oracle["slipped"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|value| value.as_str() == Some(satellite))
                    {
                        rover_observation.lli = Some(1);
                    }
                    rover_observations.push(rover_observation);
                }
                RtkArcEpoch {
                    base: base_observations,
                    rover: rover_observations,
                    satellite_positions_m: positions.clone(),
                    base_satellite_positions_m: positions.clone(),
                    rover_satellite_positions_m: positions.clone(),
                    velocity_mps: Some(if epoch_index > 3 {
                        [0.0; 3]
                    } else {
                        [2000.0 / 30.0, 1000.0 / 30.0, 200.0 / 30.0]
                    }),
                    prediction_time_s: Some(epoch_index as f64 * 30.0),
                    prediction_epoch: None,
                }
            })
            .collect();
        let mut config = config(
            (1..=6)
                .map(|index| (format!("G{index:02}"), wavelength_m))
                .collect(),
            (1..=6).map(|index| (format!("G{index:02}"), 0.0)).collect(),
        );
        config.base_m = base;
        config.initial_baseline_m = std::array::from_fn(|axis| rovers[0][axis] - base[axis]);
        config.reference = BaselineReferenceSelection::Satellite("G01".to_string());
        config.model.sagnac = false;
        config.update_opts.dynamics_model = DynamicsModel::VelocityPropagated;
        config.preprocessing.elevation_mask_deg = Some(fixture["mask_deg"].as_f64().unwrap());

        // Compare ddres's exact elevation operands too: at a one-ulp gap the
        // solved rover position may differ from the harness position, while
        // the candidate loop must still implement the exact >= predicate.
        for (epoch_index, oracle) in fixture["epochs"].as_array().unwrap().iter().enumerate() {
            let normalized = normalize_epoch(&epochs[epoch_index]);
            let mut actual = build_epoch(
                epoch_index,
                &normalized,
                &BTreeMap::from([("G".into(), "G01".into())]),
                &BTreeSet::from(["G01"]),
                None,
                0.0,
            )
            .unwrap();
            let slipped: BTreeSet<_> = oracle["slipped"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| id.as_str().unwrap())
                .collect();
            apply_rover_elevation_mask(
                &mut actual,
                rovers[epoch_index],
                fixture["mask_deg"].as_f64().unwrap(),
                &slipped,
            )
            .unwrap();
            let expected = if oracle["rover_masked"]
                .as_array()
                .unwrap()
                .iter()
                .any(|id| id == "G01")
            {
                oracle["reference"].as_str().unwrap()
            } else {
                "G01"
            };
            assert_eq!(
                actual.references[0].sat, expected,
                "actual rover screen, epoch {epoch_index}"
            );
            let mut chosen: Option<(String, f64)> = None;
            for id in oracle["used"].as_array().unwrap() {
                let id = id.as_str().unwrap();
                let index: usize = id[1..].parse::<usize>().unwrap() - 1;
                let elevation = oracle["elevation_rad"][index].as_f64().unwrap();
                let slipped = oracle["slipped"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|sat| sat.as_str() == Some(id));
                if replaces_reference(chosen.as_ref().map(|(_, el)| *el), elevation, slipped) {
                    chosen = Some((id.to_string(), elevation));
                }
            }
            assert_eq!(chosen.unwrap().0, oracle["reference"].as_str().unwrap());
        }
        let solution = solve_rtk_arc(&epochs[..7], &config).expect("moving-rover arc solves");
        for (epoch_index, (epoch, oracle)) in solution
            .epochs
            .iter()
            .zip(fixture["epochs"].as_array().unwrap())
            .enumerate()
        {
            let expected_used: Vec<_> = oracle["used"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().to_string())
                .collect();
            let expected_masked: Vec<_> = oracle["rover_masked"]
                .as_array()
                .unwrap()
                .iter()
                .map(|value| value.as_str().unwrap().to_string())
                .collect();
            assert_eq!(epoch.used_satellite_ids, expected_used);
            assert_eq!(epoch.elevation_masked_satellite_ids, expected_masked);
            let expected_reference = if expected_masked.iter().any(|sat| sat == "G01") {
                oracle["reference"].as_str().unwrap()
            } else {
                "G01"
            };
            assert_eq!(
                epoch.reference_satellite_ids,
                [expected_reference],
                "epoch {epoch_index}"
            );
        }
        assert_eq!(solution.epochs[1].elevation_masked_satellite_ids, ["G02"]);
        assert!(solution.epochs[1]
            .used_satellite_ids
            .iter()
            .all(|satellite| satellite != "G02"));

        // Removing the old reference through the rover mask must use the same
        // state transition as an ordinary availability-driven reference
        // change. Compare the two completed paths, including held integers and
        // a float double difference between two surviving ambiguity states.
        let mut masked_two_epochs = epochs[1..=2].to_vec();
        for (index, epoch) in masked_two_epochs.iter_mut().enumerate() {
            epoch.prediction_time_s = Some(index as f64 * 30.0);
        }
        let two_epoch_oracle = &fixture["epochs"].as_array().unwrap()[1..=2];
        let ordinary_epochs: Vec<_> = masked_two_epochs
            .iter()
            .zip(two_epoch_oracle)
            .map(|(epoch, oracle)| {
                let kept: Vec<_> = oracle["used"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|value| value.as_str().unwrap().to_string())
                    .collect();
                thin_epoch_to_kept(epoch, &kept)
            })
            .collect();
        let mut ordinary_config = config.clone();
        ordinary_config.initial_baseline_m =
            std::array::from_fn(|axis| rovers[1][axis] - base[axis]);
        ordinary_config.reference = BaselineReferenceSelection::Auto;
        ordinary_config.preprocessing.elevation_mask_deg = None;
        let ordinary = solve_rtk_arc(&ordinary_epochs, &ordinary_config)
            .expect("ordinary reference-change arc solves");
        let mut masked_config = ordinary_config.clone();
        masked_config.preprocessing.elevation_mask_deg =
            Some(fixture["mask_deg"].as_f64().unwrap());
        let masked = solve_rtk_arc(&masked_two_epochs, &masked_config)
            .expect("masked reference-change arc solves");
        assert!(masked.epochs[0].fixed_ids.len() >= 2);
        assert_eq!(
            masked.final_state.fixed_cycles,
            ordinary.final_state.fixed_cycles
        );
        assert_eq!(masked.final_state.fixed_m, ordinary.final_state.fixed_m);
        let float_dd = |state: &FilterState, left: &str, right: &str| {
            let left_index = state
                .sd_ambiguity_ids
                .iter()
                .position(|id| id == left)
                .expect("left ambiguity");
            let right_index = state
                .sd_ambiguity_ids
                .iter()
                .position(|id| id == right)
                .expect("right ambiguity");
            state.sd_ambiguities_m[left_index] - state.sd_ambiguities_m[right_index]
        };
        assert_eq!(
            float_dd(&masked.final_state, "G03", "G04").to_bits(),
            float_dd(&ordinary.final_state, "G03", "G04").to_bits()
        );
        assert_eq!(
            masked.final_state.information,
            ordinary.final_state.information
        );
        // A configured reference forces the physical reference to change;
        // Auto above deliberately keeps the common reference for its parity
        // comparison.  The first epoch establishes holds before G01 is masked.
        masked_config.reference = BaselineReferenceSelection::Satellite("G01".to_string());
        let changing = solve_rtk_arc(&masked_two_epochs, &masked_config).unwrap();
        assert_eq!(changing.epochs[0].reference_satellite_ids, ["G01"]);
        assert_eq!(
            changing.epochs[1].reference_satellite_ids,
            [two_epoch_oracle[1]["reference"].as_str().unwrap()]
        );
        assert_ne!(
            changing.epochs[0].reference_satellite_ids,
            changing.epochs[1].reference_satellite_ids
        );
        assert!(changing.epochs[0].fixed_ids.len() >= 2);
        let gauge_oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/rtk_reference_gauge.json"
        ))
        .unwrap();
        let expected = &gauge_oracle["moving_arc"];
        let first_fixed: Vec<_> = expected["first_epoch_held_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap())
            .collect();
        assert_eq!(changing.epochs[0].fixed_ids, first_fixed);
        for (id, cycles) in expected["translated_held_cycles"].as_object().unwrap() {
            assert_eq!(
                changing.final_state.fixed_cycles.get(id),
                cycles.as_i64().as_ref()
            );
        }
        let old_reference = expected["reference_ids"][0].as_str().unwrap();
        let new_reference = expected["reference_ids"][1].as_str().unwrap();
        let state = &changing.final_state;
        let old = state.ambiguity_pos(old_reference).unwrap() + 3;
        let new = state.ambiguity_pos(new_reference).unwrap() + 3;
        let rows: Vec<Vec<_>> = state
            .information
            .chunks(state.dim())
            .map(|row| row.to_vec())
            .collect();
        let covariance = invert_matrix_first_tie(&rows).unwrap();
        let dd_variance = covariance[old][old] + covariance[new][new] - 2.0 * covariance[old][new];
        let reference_variance = expected["dd_variance_m2"].as_f64().unwrap();
        // Independent least squares uses the exact ECEF geometry, all code
        // and phase DD blocks and the four established integer hold rows,
        // with one gauge direction. gamma_(128*n)*kappa(N) bounds normal
        // formation, coordinate transport, inversion and contrast evaluation.
        let n = expected["matrix_dimension"].as_f64().unwrap();
        let gamma = 128.0 * n * f64::EPSILON / (1.0 - 128.0 * n * f64::EPSILON);
        let amplification = gamma * expected["normal_condition_one"].as_f64().unwrap();
        let relative_bound = amplification / (1.0 - amplification);
        assert!(
            (dd_variance - reference_variance).abs() <= relative_bound * reference_variance,
            "variance {dd_variance} != {reference_variance}"
        );
        // Sixty-four rounded range/phase operations per input ECEF scale
        // bound the finite binary64 observation quantization in the mean.
        let position_bound =
            64.0 * f64::EPSILON * expected["roundoff_position_scale_m"].as_f64().unwrap();
        assert!(
            (float_dd(state, old_reference, new_reference) - expected["dd_m"].as_f64().unwrap())
                .abs()
                <= position_bound
        );

        let mut capped = config.clone();
        capped.update_opts.max_iterations = 1;
        let mut observed_unsettled = false;
        for displacement in [-10_000.0, 10_000.0] {
            capped.initial_baseline_m[2] = config.initial_baseline_m[2] + displacement;
            if let Ok(result) = solve_rtk_arc(&epochs[..1], &capped) {
                observed_unsettled |= !result.epochs[0].elevation_mask_converged;
            }
        }
        assert!(
            observed_unsettled,
            "a capped changing satellite set is reported"
        );
    }

    #[test]
    fn no_mask_entry_points_match_c2a2aa6_bit_for_bit() {
        use serde_json::{json, Value};

        fn bits(values: &[f64]) -> Vec<u64> {
            values.iter().map(|value| value.to_bits()).collect()
        }
        fn arc(value: &RtkArcSolution) -> Value {
            json!({
                "references": value.references,
                "epoch_reported_baseline_bits": value.epochs.iter().map(|epoch| bits(&epoch.reported_baseline_m)).collect::<Vec<_>>(),
                "epoch_float_baseline_bits": value.epochs.iter().map(|epoch| bits(&epoch.float_baseline_m)).collect::<Vec<_>>(),
                "epoch_sd_ambiguity_bits": value.epochs.iter().map(|epoch| epoch.sd_ambiguities_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>()).collect::<Vec<_>>(),
                "epoch_used": value.epochs.iter().map(|epoch| &epoch.used_satellite_ids).collect::<Vec<_>>(),
                "epoch_fixed_ids": value.epochs.iter().map(|epoch| &epoch.fixed_ids).collect::<Vec<_>>(),
                "epoch_newly_fixed": value.epochs.iter().map(|epoch| &epoch.newly_fixed).collect::<Vec<_>>(),
                "epoch_fixed_dd_ids": value.epochs.iter().map(|epoch| &epoch.fixed_double_difference_ids).collect::<Vec<_>>(),
                "epoch_integer_fixed": value.epochs.iter().map(|epoch| epoch.integer_fixed).collect::<Vec<_>>(),
                "epoch_integer_ratio_bits": value.epochs.iter().map(|epoch| epoch.integer_ratio.to_bits()).collect::<Vec<_>>(),
                // Debug uses shortest round-trip binary64 decimal formatting.
                // Equality therefore preserves every represented finite bit.
                "epoch_residuals": value.epochs.iter().map(|epoch| format!("{:?}", epoch.residuals)).collect::<Vec<_>>(),
                "epoch_search": value.epochs.iter().map(|epoch| format!("{:?}", epoch.search)).collect::<Vec<_>>(),
                "epoch_geometry": value.epochs.iter().map(|epoch| format!("{:?}", epoch.geometry_quality)).collect::<Vec<_>>(),
                "state_baseline_bits": bits(&value.final_state.baseline_m),
                "state_sd_ambiguity_bits": bits(&value.final_state.sd_ambiguities_m),
                "state_information_bits": bits(&value.final_state.information),
                "measurement_covariance_bits": bits(&value.measurement_covariance),
                "fixed_cycles": value.final_state.fixed_cycles,
                "fixed_m_bits": value.final_state.fixed_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
            })
        }
        fn static_arc(value: &RtkStaticArcSolution) -> Value {
            json!({
                "references": value.references,
                "ambiguity_ids": value.ambiguity_ids,
                "float_diagnostics": format!("{:?}", value.float_solution),
                "fixed_diagnostics": format!("{:?}", value.fixed_solution),
                "geometry": format!("{:?}", value.geometry_quality),
                "float_baseline_bits": bits(&value.float_solution.baseline_m),
                "float_ambiguity_bits": value.float_solution.ambiguities_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
                "float_covariance_bits": bits(&value.float_solution.ambiguity_covariance_m),
                "fixed_baseline_bits": bits(&value.fixed_solution.fixed_solution.baseline_m),
                "fixed_cycles": value.fixed_solution.fixed_solution.fixed_ambiguities_cycles,
                "fixed_m_bits": value.fixed_solution.fixed_solution.fixed_ambiguities_m.iter().map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
            })
        }
        fn ionosphere_free(value: &RtkIonosphereFreeArcSolution) -> Value {
            json!({
                "references": value.references,
                "epochs": value.epochs.iter().map(|epoch| json!({
                    "base": epoch.base.iter().map(|obs| (&obs.satellite_id, &obs.ambiguity_id, obs.code_m.to_bits(), obs.phase_m.to_bits())).collect::<Vec<_>>(),
                    "rover": epoch.rover.iter().map(|obs| (&obs.satellite_id, &obs.ambiguity_id, obs.code_m.to_bits(), obs.phase_m.to_bits())).collect::<Vec<_>>(),
                })).collect::<Vec<_>>(),
                // Reference entries are additive scale metadata for reference
                // changes. All original entries and every solved value retain
                // the historical bit comparison; reference metadata is checked
                // separately by wide_lane_masked_reference_holds_survive_and_return.
                "wavelength_bits": value.wavelengths_m.iter().filter(|(id, _)| !value.references.values().any(|reference| reference==*id)).map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
                "offset_bits": value.offsets_m.iter().filter(|(id, _)| !value.references.values().any(|reference| reference==*id)).map(|(id, value)| (id, value.to_bits())).collect::<Vec<_>>(),
            })
        }

        let expected: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/rtk_no_mask_c2a2aa6.json"
        ))
        .expect("historical fixture");
        assert_eq!(
            expected["provenance"]["source_commit"],
            "c2a2aa64f004e2238af5a609a7948f06a995037d"
        );
        let (epochs, mut config, _, _) = integer_arc();
        config.update_opts.report_residuals = true;
        let sequential = solve_rtk_arc(&epochs, &config).expect("sequential");
        assert!(sequential
            .epochs
            .iter()
            .all(|epoch| !epoch.residuals.is_empty()));
        assert!(sequential.epochs.iter().any(|epoch| epoch.search.is_some()));
        assert!(sequential
            .epochs
            .iter()
            .any(|epoch| !epoch.fixed_ids.is_empty()));
        assert_eq!(arc(&sequential), expected["solve_rtk_arc"]);
        let static_config = RtkStaticArcConfig {
            arc: config.clone(),
            opts: validated_opts(),
        };
        let static_solution = solve_static_rtk_arc(&epochs, &static_config).expect("static");
        assert_eq!(
            static_arc(&static_solution),
            expected["solve_static_rtk_arc"]
        );

        let (dual_epochs, wide_config) = dual_arc();
        let wide = fix_wide_lane_rtk_arc(&dual_epochs, &wide_config).expect("wide lane");
        assert_eq!(
            json!({"references": wide.references, "wide_lane_cycles": wide.wide_lane_cycles, "diagnostics": format!("{wide:?}")}),
            expected["fix_wide_lane_rtk_arc"]
        );
        let if_config = RtkIonosphereFreeArcConfig {
            base_m: wide_config.base_m,
            initial_baseline_m: [0.0; 3],
            reference: wide_config.reference.clone(),
            apply_troposphere: false,
        };
        let ionosphere =
            prepare_ionosphere_free_rtk_arc(&wide.epochs, &wide.wide_lane_cycles, &if_config)
                .expect("ionosphere free");
        assert_eq!(
            ionosphere_free(&ionosphere),
            expected["prepare_ionosphere_free_rtk_arc"]
        );
        for (solve, key) in [
            (
                RtkWideLaneFixedArcSolveConfig::Static(static_config),
                "solve_wide_lane_fixed_rtk_arc_static",
            ),
            (
                RtkWideLaneFixedArcSolveConfig::Sequential(config),
                "solve_wide_lane_fixed_rtk_arc_sequential",
            ),
        ] {
            let result = solve_wide_lane_fixed_rtk_arc(
                &dual_epochs,
                &RtkWideLaneFixedArcConfig {
                    wide_lane: wide_config.clone(),
                    ionosphere_free: if_config.clone(),
                    solve,
                },
            )
            .expect("combined solve");
            let actual = match result {
                RtkWideLaneFixedArcSolution::Static(value) => static_arc(&value.solution),
                RtkWideLaneFixedArcSolution::Sequential(value) => arc(&value.solution),
            };
            assert_eq!(actual, expected[key], "{key}");
        }
    }

    #[test]
    fn static_batch_cycle_slip_split_derives_scale_for_split_ids() {
        let (mut epochs, mut arc_config, baseline, _tokens) = integer_arc();
        for obs in &mut epochs[3].rover {
            if obs.satellite_id == "G02" {
                obs.lli = Some(1);
            }
        }
        arc_config.preprocessing.cycle_slip = Some(CycleSlipPolicy::SplitArc);
        let static_config = RtkStaticArcConfig {
            arc: arc_config,
            opts: validated_opts(),
        };

        let solution = solve_static_rtk_arc(&epochs, &static_config).expect("static driver");
        assert!(!solution.split_cycle_slip_arcs.is_empty());
        assert!(
            solution
                .fixed_solution
                .fixed_solution
                .search
                .integer_candidates
                > 0
        );
        assert!(
            norm3(sub3(
                solution.fixed_solution.fixed_solution.baseline_m,
                baseline,
            )) < 1.0e-6
        );
    }

    fn dual_observation(
        ambiguity_id: &str,
        p1_m: f64,
        p2_m: f64,
        wide_lane_phase_cycles: f64,
    ) -> RtkDualFrequencyObservation {
        RtkDualFrequencyObservation {
            ambiguity_id: ambiguity_id.to_string(),
            p1_m,
            p2_m,
            phi1_cycles: wide_lane_phase_cycles,
            phi2_cycles: 0.0,
            f1_hz: F_L1_HZ,
            f2_hz: F_L2_HZ,
            lli1: None,
            lli2: None,
        }
    }

    fn dual_satellite(
        sat: &str,
        base_wide_lane: f64,
        rover_wide_lane: f64,
    ) -> RtkDualFrequencySatelliteObservation {
        let base_code = 20_000_000.0 + base_wide_lane * 10.0;
        let rover_code = base_code + 25.0 + rover_wide_lane;
        RtkDualFrequencySatelliteObservation {
            satellite_id: sat.to_string(),
            base: dual_observation(sat, base_code, base_code + 2.0, base_wide_lane),
            rover: dual_observation(sat, rover_code, rover_code + 2.5, rover_wide_lane),
        }
    }

    fn dual_arc() -> (Vec<RtkDualFrequencyArcEpoch>, RtkWideLaneArcConfig) {
        let (_source, _ids, tokens) = build();
        let positions = sat_layout()
            .iter()
            .zip(tokens.iter())
            .map(|((_, position), token)| (token.clone(), *position))
            .collect::<BTreeMap<_, _>>();
        let epochs = (0..3)
            .map(|idx| RtkDualFrequencyArcEpoch {
                jd_whole: 2_460_100.5,
                jd_fraction: 0.25 + idx as f64 / crate::constants::SECONDS_PER_DAY,
                epoch_sort_key: Some(format!("{idx:03}")),
                gap_time_s: Some(idx as f64),
                gap_epoch: None,
                observations: vec![
                    dual_satellite("G01", 2.0, 5.0),
                    dual_satellite("G02", 1.0, 7.0),
                    dual_satellite("G03", -2.0, 0.0),
                    dual_satellite("G04", 4.0, 8.0),
                ],
                satellite_positions_m: positions.clone(),
                base_satellite_positions_m: BTreeMap::new(),
                rover_satellite_positions_m: BTreeMap::new(),
                velocity_mps: None,
                prediction_time_s: None,
                prediction_epoch: None,
            })
            .collect();
        (
            epochs,
            RtkWideLaneArcConfig {
                base_m: [3_512_900.0, 780_500.0, 5_248_700.0],
                reference: BaselineReferenceSelection::Auto,
                options: WideLaneOptions {
                    min_epochs: 2,
                    tolerance_cycles: 0.5,
                    skip_short_fragments: false,
                },
                cycle_slip: None,
            },
        )
    }

    #[test]
    fn wide_lane_driver_matches_manual_melbourne_wubbena_primitives() {
        let (epochs, config) = dual_arc();
        let driver = fix_wide_lane_rtk_arc(&epochs, &config).expect("wide lane driver");
        let references =
            dual_arc_references(config.base_m, &epochs, config.reference.clone()).expect("refs");
        let mut manual_cycles = BTreeMap::new();
        for (system, reference) in &references {
            let fixed = estimate_wide_lane_ambiguities(
                &dual_epochs_for_system(&epochs, system),
                reference,
                config.options,
            )
            .expect("wide lane");
            manual_cycles.extend(fixed);
        }

        assert_eq!(driver.references, references);
        assert_eq!(driver.wide_lane_cycles, manual_cycles);
        assert_eq!(driver.epochs, epochs);
        assert_eq!(driver.dropped_sats, Vec::<String>::new());
        assert_eq!(
            driver.split_cycle_slip_arcs,
            Vec::<CycleSlipSplitArc>::new()
        );
    }

    fn assert_arc_observation_bits(got: &RtkArcObservation, expected: &RtkArcObservation) {
        assert_eq!(got.satellite_id, expected.satellite_id);
        assert_eq!(got.ambiguity_id, expected.ambiguity_id);
        assert_f64_bits(got.code_m, expected.code_m);
        assert_f64_bits(got.phase_m, expected.phase_m);
        assert_eq!(got.lli, expected.lli);
    }

    fn assert_arc_epochs_bits(got: &[RtkArcEpoch], expected: &[RtkArcEpoch]) {
        assert_eq!(got.len(), expected.len());
        for (got, expected) in got.iter().zip(expected) {
            assert_eq!(got.base.len(), expected.base.len());
            assert_eq!(got.rover.len(), expected.rover.len());
            for (got, expected) in got.base.iter().zip(&expected.base) {
                assert_arc_observation_bits(got, expected);
            }
            for (got, expected) in got.rover.iter().zip(&expected.rover) {
                assert_arc_observation_bits(got, expected);
            }
            assert_eq!(got.satellite_positions_m, expected.satellite_positions_m);
            assert_eq!(
                got.base_satellite_positions_m,
                expected.base_satellite_positions_m
            );
            assert_eq!(
                got.rover_satellite_positions_m,
                expected.rover_satellite_positions_m
            );
            assert_eq!(got.velocity_mps, expected.velocity_mps);
            assert_eq!(got.prediction_time_s, expected.prediction_time_s);
        }
    }

    #[test]
    fn ionosphere_free_driver_matches_manual_combination_primitives() {
        let (epochs, wide_lane_config) = dual_arc();
        let wide_lane =
            fix_wide_lane_rtk_arc(&epochs, &wide_lane_config).expect("wide lane driver");
        let config = RtkIonosphereFreeArcConfig {
            base_m: wide_lane_config.base_m,
            initial_baseline_m: [0.0, 0.0, 0.0],
            reference: wide_lane_config.reference.clone(),
            apply_troposphere: false,
        };
        let driver = prepare_ionosphere_free_rtk_arc(&epochs, &wide_lane.wide_lane_cycles, &config)
            .expect("if driver");

        let reference = wide_lane
            .references
            .get("G")
            .expect("GPS reference")
            .clone();
        let setup = dual_setup_epochs_for_system(&epochs, "G");
        let setup_epochs = setup
            .iter()
            .map(|(_, epoch)| epoch.clone())
            .collect::<Vec<_>>();
        let manual_result = prepare_ionosphere_free_baseline_epochs(
            config.base_m,
            config.initial_baseline_m,
            &setup_epochs,
            &reference,
            &wide_lane.wide_lane_cycles,
            config.apply_troposphere,
        )
        .expect("manual if");
        let mut merged = BTreeMap::new();
        for epoch in manual_result.epochs {
            let original_index = setup[epoch.epoch_index].0;
            merge_ionosphere_free_epoch(&mut merged, original_index, epoch);
        }
        let manual_epochs = merged
            .into_iter()
            .map(|(index, epoch)| ionosphere_free_arc_epoch(&epochs[index], epoch))
            .collect::<Vec<_>>();

        assert_eq!(driver.references, wide_lane.references);
        assert_eq!(driver.wavelengths_m, manual_result.wavelengths_m);
        assert_eq!(driver.offsets_m, manual_result.offsets_m);
        assert_arc_epochs_bits(&driver.epochs, &manual_epochs);
    }

    #[test]
    fn wide_lane_fixed_static_driver_matches_manual_staging() {
        let (epochs, wide_lane_config) = dual_arc();
        let ionosphere_free_config = RtkIonosphereFreeArcConfig {
            base_m: wide_lane_config.base_m,
            initial_baseline_m: [0.0, 0.0, 0.0],
            reference: wide_lane_config.reference.clone(),
            apply_troposphere: false,
        };
        let static_config = RtkStaticArcConfig {
            arc: config(BTreeMap::new(), BTreeMap::new()),
            opts: validated_opts(),
        };
        let combined_config = RtkWideLaneFixedArcConfig {
            wide_lane: wide_lane_config.clone(),
            ionosphere_free: ionosphere_free_config.clone(),
            solve: RtkWideLaneFixedArcSolveConfig::Static(static_config.clone()),
        };

        let driver =
            solve_wide_lane_fixed_rtk_arc(&epochs, &combined_config).expect("combined static");

        let wide_lane =
            fix_wide_lane_rtk_arc(&epochs, &wide_lane_config).expect("manual wide lane");
        let ionosphere_free = prepare_ionosphere_free_rtk_arc(
            &wide_lane.epochs,
            &wide_lane.wide_lane_cycles,
            &ionosphere_free_config,
        )
        .expect("manual ionosphere-free");
        let reference_satellite =
            single_reference_satellite(&ionosphere_free.references).expect("single reference");
        let mut final_config = static_config;
        final_config.arc.reference = BaselineReferenceSelection::Satellite(reference_satellite);
        final_config.arc.wavelengths_m = ionosphere_free.wavelengths_m.clone();
        final_config.arc.offsets_m = ionosphere_free.offsets_m.clone();
        for reference in ionosphere_free.references.values() {
            final_config.arc.wavelengths_m.remove(reference);
            final_config.arc.offsets_m.remove(reference);
        }
        let solution =
            solve_static_rtk_arc(&ionosphere_free.epochs, &final_config).expect("manual static");
        let metadata = wide_lane_fixed_metadata(
            &wide_lane,
            &static_solution_used_satellites(&solution),
            RtkWideLaneFixedArcIntegerMethod::WideLaneNarrowLaneLambda,
        );
        let expected = RtkWideLaneFixedArcSolution::Static(RtkWideLaneFixedStaticArcSolution {
            wide_lane,
            ionosphere_free,
            solution,
            metadata,
        });

        assert_eq!(driver, expected);
        let RtkWideLaneFixedArcSolution::Static(driver) = &driver else {
            unreachable!("static combined result")
        };
        let RtkWideLaneFixedArcSolution::Static(expected) = &expected else {
            unreachable!("static expected result")
        };
        assert_static_solution_bits(&driver.solution, &expected.solution);
        assert_eq!(driver.metadata, expected.metadata);
    }

    #[test]
    fn wide_lane_fixed_sequential_driver_matches_manual_staging() {
        let (epochs, wide_lane_config) = dual_arc();
        let ionosphere_free_config = RtkIonosphereFreeArcConfig {
            base_m: wide_lane_config.base_m,
            initial_baseline_m: [0.0, 0.0, 0.0],
            reference: wide_lane_config.reference.clone(),
            apply_troposphere: false,
        };
        let arc_config = config(BTreeMap::new(), BTreeMap::new());
        let combined_config = RtkWideLaneFixedArcConfig {
            wide_lane: wide_lane_config.clone(),
            ionosphere_free: ionosphere_free_config.clone(),
            solve: RtkWideLaneFixedArcSolveConfig::Sequential(arc_config.clone()),
        };

        let driver =
            solve_wide_lane_fixed_rtk_arc(&epochs, &combined_config).expect("combined sequential");

        let wide_lane =
            fix_wide_lane_rtk_arc(&epochs, &wide_lane_config).expect("manual wide lane");
        let ionosphere_free = prepare_ionosphere_free_rtk_arc(
            &wide_lane.epochs,
            &wide_lane.wide_lane_cycles,
            &ionosphere_free_config,
        )
        .expect("manual ionosphere-free");
        let reference_satellite =
            single_reference_satellite(&ionosphere_free.references).expect("single reference");
        let mut final_config = arc_config;
        final_config.reference = BaselineReferenceSelection::Satellite(reference_satellite);
        final_config.wavelengths_m = ionosphere_free.wavelengths_m.clone();
        final_config.offsets_m = ionosphere_free.offsets_m.clone();
        let solution = solve_rtk_arc(&ionosphere_free.epochs, &final_config).expect("manual arc");
        let metadata = wide_lane_fixed_metadata(
            &wide_lane,
            &sequential_solution_used_satellites(&solution),
            RtkWideLaneFixedArcIntegerMethod::WideLaneNarrowLaneSequential,
        );
        let expected =
            RtkWideLaneFixedArcSolution::Sequential(RtkWideLaneFixedSequentialArcSolution {
                wide_lane,
                ionosphere_free,
                solution,
                metadata,
            });

        assert_eq!(driver, expected);
    }
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod carrier_arc_tests {
    use super::*;
    use crate::carrier_phase::CycleSlipOptions;
    use crate::rtk::{
        prepare_cycle_slip_baseline_epochs, prepare_dual_cycle_slip_baseline_epochs,
        sd_ambiguity_token, CycleSlipEpoch, CycleSlipObservation, CycleSlipPolicy,
        DualCycleSlipEpoch, DualCycleSlipObservation,
    };

    /// R01 over three epochs: the builder's carrier arcs, `R01` twice then
    /// `R01~freq2`, with loss of lock at the second epoch on both receivers.
    fn single_epochs() -> Vec<CycleSlipEpoch> {
        (0..3)
            .map(|index| {
                let observation = CycleSlipObservation {
                    satellite_id: "R01".to_string(),
                    ambiguity_id: if index < 2 { "R01" } else { "R01~freq2" }.to_string(),
                    code_m: 20_000_000.0,
                    phase_m: 100.0,
                    lli: Some(if index == 1 { 1 } else { 0 }),
                };
                CycleSlipEpoch {
                    base_observations: vec![observation.clone()],
                    rover_observations: vec![observation],
                }
            })
            .collect()
    }

    #[test]
    fn a_slip_split_keeps_a_carrier_change_apart_and_its_wavelength() {
        let prepared =
            prepare_cycle_slip_baseline_epochs(&single_epochs(), CycleSlipPolicy::SplitArc)
                .expect("prepare");
        let ids: Vec<(String, String)> = prepared
            .epochs
            .iter()
            .map(|epoch| {
                (
                    epoch.base_observations[0].ambiguity_id.clone(),
                    epoch.rover_observations[0].ambiguity_id.clone(),
                )
            })
            .collect();
        // A slip splits the arc, and so does the carrier change after it.
        assert_ne!(ids[0], ids[1], "{ids:?}");
        assert_ne!(ids[1], ids[2], "{ids:?}");
        // Every derived id keeps its carrier's wavelength.
        let wavelengths = BTreeMap::from([
            ("R01".to_string(), 0.187_597_455),
            ("R01~freq2".to_string(), 0.186_742_947),
        ]);
        for (index, (base, rover)) in ids.iter().enumerate() {
            let sd = sd_ambiguity_token("R01", base, rover);
            let expected = if index < 2 {
                0.187_597_455
            } else {
                0.186_742_947
            };
            assert_eq!(
                scale_value_for(&wavelengths, &sd, "R01"),
                Some(expected),
                "epoch {index}: {sd}"
            );
            let dd = crate::rtk::dd_ambiguity_token("R01", &sd, "R02", "R02");
            assert_eq!(
                scale_value_for(&wavelengths, &dd, "R01"),
                Some(expected),
                "{dd}"
            );
        }
    }

    #[test]
    fn a_dual_frequency_slip_split_keeps_a_carrier_change_apart() {
        // Only the loss of lock flags a slip.
        let options = CycleSlipOptions {
            gf_threshold_m: 1e9,
            mw_threshold_cycles: 1e9,
            min_arc_gap_s: 1000.0,
            ..CycleSlipOptions::default()
        };
        let epochs: Vec<DualCycleSlipEpoch> = (0..3)
            .map(|index| {
                let observation = DualCycleSlipObservation {
                    satellite_id: "R01".to_string(),
                    ambiguity_id: if index < 2 { "R01" } else { "R01~freq2" }.to_string(),
                    p1_m: 20_000_000.0,
                    p2_m: 20_000_000.0,
                    phi1_cycles: 100.0,
                    phi2_cycles: 90.0,
                    f1_hz: if index < 2 {
                        1_598_062_500.0
                    } else {
                        1_605_375_000.0
                    },
                    f2_hz: if index < 2 {
                        1_242_937_500.0
                    } else {
                        1_248_625_000.0
                    },
                    lli1: Some(if index == 1 { 1 } else { 0 }),
                    lli2: Some(0),
                };
                DualCycleSlipEpoch {
                    epoch_sort_key: format!("{index}"),
                    gap_time_s: Some(index as f64 * 10.0),
                    gap_epoch: None,
                    base_observations: vec![observation.clone()],
                    rover_observations: vec![observation],
                }
            })
            .collect();
        let prepared =
            prepare_dual_cycle_slip_baseline_epochs(&epochs, CycleSlipPolicy::SplitArc, options)
                .expect("prepare");
        let ids: Vec<String> = prepared
            .epochs
            .iter()
            .map(|epoch| epoch.base_observations[0].ambiguity_id.clone())
            .collect();
        assert_ne!(ids[0], ids[1], "{ids:?}");
        assert_ne!(ids[1], ids[2], "{ids:?}");
    }
}
