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
//!   per-epoch `prediction_time_s`: the first epoch is zero, a missing time is
//!   zero under [`DynamicsModel::ConstantPosition`] (lenient) and an error under
//!   [`DynamicsModel::VelocityPropagated`] (strict).

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::math::linear::invert_matrix_first_tie;
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
    /// Elevation mask (degrees) applied at the base receiver via
    /// [`crate::rtk::apply_elevation_mask`]. `None` skips masking.
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
    pub wavelengths_m: BTreeMap<String, f64>,
    /// Per-ambiguity code-to-phase metre offsets for the integer search.
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
    /// returned for the final fixed solve.
    pub wavelengths_m: BTreeMap<String, f64>,
    /// Code-to-phase offset scale per resulting single-difference ambiguity id,
    /// in meters, returned for the final fixed solve.
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
    /// `ensure_single_wide_lane_system` found multiple constellations, or
    /// `single_reference_satellite` did not find exactly one reference.
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
                write!(f, "wide-lane fixed RTK arc supports one constellation")
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
    /// All held single-difference ambiguity ids after this epoch.
    pub fixed_ids: Vec<String>,
    /// Reported single-difference ambiguities (id, metres) in column order.
    pub sd_ambiguities_m: Vec<(String, f64)>,
    /// Double-difference ambiguity ids fixed this epoch, against each one's own
    /// system reference (the reporting form used by RTKLIB-style consumers).
    pub fixed_double_difference_ids: Vec<String>,
    /// Satellites used this epoch (intersection availability), sorted.
    pub used_satellite_ids: Vec<String>,
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

    Ok(RtkArcSolution {
        references: core.references,
        epochs: core.epochs,
        final_state: core.final_state,
        dropped_sats,
        split_cycle_slip_arcs,
        elevation_masked_sats,
        measurement_covariance,
    })
}

/// Solve a static RTK baseline over a raw arc as one batch.
///
/// Enabled preprocessing runs before batch construction. The function then
/// derives the ambiguity columns and scale maps, calls the float and validated
/// fixed baseline solvers, and returns their results with preprocessing metadata.
/// The static-station wrapper uses this path after filling the arc scale maps
/// from RINEX.
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
    let batch = build_static_batch_arc(solve_input, &config.arc)?;
    let (wavelengths_m, offsets_m) = expanded_batch_scale_maps(&batch, &config.arc);
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
    .map_err(RtkStaticArcError::Float)?;
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
        elevation_masked_sats,
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
        .map(|(index, epoch)| ionosphere_free_arc_epoch(&epochs[index], epoch))
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
/// The input must contain one constellation. After wide-lane fixing and
/// ionosphere-free conversion, the function injects the sole selected reference
/// and the derived wavelength/offset maps into either the configured static or
/// sequential solve, then returns the corresponding composite result.
///
/// [`RtkWideLaneFixedArcError::UnsupportedMultiGnss`] is returned for multiple
/// constellations or for a reference map that is not singular; failures from
/// each pipeline stage retain their matching error variant.
pub fn solve_wide_lane_fixed_rtk_arc(
    epochs: &[RtkDualFrequencyArcEpoch],
    config: &RtkWideLaneFixedArcConfig,
) -> Result<RtkWideLaneFixedArcSolution, RtkWideLaneFixedArcError> {
    ensure_single_wide_lane_system(epochs)?;
    let wide_lane = fix_wide_lane_rtk_arc(epochs, &config.wide_lane)
        .map_err(RtkWideLaneFixedArcError::WideLane)?;
    let ionosphere_free = prepare_ionosphere_free_rtk_arc(
        &wide_lane.epochs,
        &wide_lane.wide_lane_cycles,
        &config.ionosphere_free,
    )
    .map_err(RtkWideLaneFixedArcError::IonosphereFree)?;
    let reference_satellite = single_reference_satellite(&ionosphere_free.references)?;

    match &config.solve {
        RtkWideLaneFixedArcSolveConfig::Static(static_config) => {
            let mut solve_config = static_config.clone();
            solve_config.arc.reference = BaselineReferenceSelection::Satellite(reference_satellite);
            solve_config.arc.wavelengths_m = ionosphere_free.wavelengths_m.clone();
            solve_config.arc.offsets_m = ionosphere_free.offsets_m.clone();
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
            solve_config.reference = BaselineReferenceSelection::Satellite(reference_satellite);
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

fn ensure_single_wide_lane_system(
    epochs: &[RtkDualFrequencyArcEpoch],
) -> Result<(), RtkWideLaneFixedArcError> {
    let mut systems = BTreeSet::new();
    for epoch in epochs {
        for observation in &epoch.observations {
            systems.insert(constellation_letter(&observation.satellite_id).to_string());
            if systems.len() > 1 {
                return Err(RtkWideLaneFixedArcError::UnsupportedMultiGnss);
            }
        }
    }
    Ok(())
}

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
        let mask_epochs: Vec<ElevationMaskEpoch> = work
            .iter()
            .map(|epoch| ElevationMaskEpoch {
                satellite_positions_m: epoch.satellite_positions_m.clone(),
            })
            .collect();
        let result = apply_elevation_mask(config.base_m, &mask_epochs, mask_deg)
            .map_err(RtkArcError::ElevationMask)?;
        work = work
            .iter()
            .zip(result.epochs.iter())
            .map(|(epoch, kept)| thin_epoch_to_kept(epoch, &kept.kept_satellite_ids))
            .collect();
        elevation_masked_sats = result.masked_satellite_ids;
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

fn scale_value_for(
    source: &BTreeMap<String, f64>,
    ambiguity_id: &str,
    satellite_id: &str,
) -> Option<f64> {
    source
        .get(ambiguity_id)
        .or_else(|| {
            ambiguity_id
                .split_once("|ref=")
                .and_then(|(sd_id, _)| source.get(sd_id))
        })
        .or_else(|| source.get(satellite_id))
        .copied()
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
    let mut previous_time: Option<f64> = None;
    let mut solutions = Vec::with_capacity(epochs.len());

    for (epoch_index, normalized_epoch) in normalized.iter().enumerate() {
        let dt_s = prediction_dt_s(
            epoch_index,
            normalized_epoch.velocity_mps,
            epochs[epoch_index].prediction_time_s,
            &mut previous_time,
            strict_time,
        )?;
        let epoch = build_epoch(
            epoch_index,
            normalized_epoch,
            &references_by_system,
            &reference_sats,
            normalized_epoch.velocity_mps,
            dt_s,
        )?;

        let update = update_epoch(
            state,
            &epoch,
            config.base_m,
            &config.model,
            &wavelengths_m,
            &offsets_m,
            &config.update_opts,
        )
        .map_err(|source| RtkArcError::Update {
            epoch_index,
            source,
        })?;

        solutions.push(epoch_solution(
            &update,
            normalized_epoch.available.clone(),
            &references,
        ));
        state = update.state;
    }

    Ok(PreparedSolution {
        references,
        epochs: solutions,
        final_state: state,
    })
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

/// Compute the prediction delta (seconds) before this epoch's update.
fn prediction_dt_s(
    epoch_index: usize,
    velocity_mps: Option<[f64; 3]>,
    prediction_time_s: Option<f64>,
    previous_time: &mut Option<f64>,
    strict_time: bool,
) -> Result<f64, RtkArcError> {
    // The dt is only consumed by the velocity-propagated branch; carry the time
    // forward regardless so a later epoch can still difference against it.
    let dt = match (epoch_index, *previous_time, prediction_time_s) {
        (0, _, _) => 0.0,
        (_, Some(previous), Some(current)) => current - previous,
        _ => {
            if strict_time {
                return Err(RtkArcError::InvalidEpochTime { epoch_index });
            }
            0.0
        }
    };
    if prediction_time_s.is_some() {
        *previous_time = prediction_time_s;
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
        sd_ambiguities_m,
        fixed_double_difference_ids,
        used_satellite_ids,
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
    use crate::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
    use crate::observables::{predict, ObservableState, ObservablesError, PredictOptions};
    use crate::rtk_filter::{
        defaults, FixedBaselineSolution, FixedSolveOpts, FloatSolveOpts, ResidualValidationOpts,
        SearchOpts, StochasticModel,
    };
    use crate::{GnssSatelliteId, GnssSystem};

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
            });
        }

        let wavelengths_m: BTreeMap<String, f64> =
            tokens.iter().map(|t| (t.clone(), wavelength_m)).collect();
        let offsets_m: BTreeMap<String, f64> = tokens.iter().map(|t| (t.clone(), 0.0)).collect();
        (epochs, config(wavelengths_m, offsets_m), baseline, tokens)
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
            let mask_epochs: Vec<ElevationMaskEpoch> = work
                .iter()
                .map(|e| ElevationMaskEpoch {
                    satellite_positions_m: e.satellite_positions_m.clone(),
                })
                .collect();
            let result =
                apply_elevation_mask(config.base_m, &mask_epochs, deg).expect("elevation mask");
            work = work
                .iter()
                .zip(result.epochs.iter())
                .map(|(epoch, kept)| {
                    let keep: BTreeSet<&str> =
                        kept.kept_satellite_ids.iter().map(String::as_str).collect();
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
                    }
                })
                .collect();
            elevation_masked_sats = result.masked_satellite_ids;
        }

        let core = solve_prepared_arc(&work, config).expect("core solve");
        let measurement_covariance = posterior_covariance(&core.final_state);
        RtkArcSolution {
            references: core.references,
            epochs: core.epochs,
            final_state: core.final_state,
            dropped_sats,
            split_cycle_slip_arcs,
            elevation_masked_sats,
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
