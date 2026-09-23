//! Multi-epoch static positioning over stacked pseudorange epochs.
//!
//! The solve is sans-I/O: callers provide already formed pseudorange
//! measurements grouped by receive epoch and receive one static receiver
//! position. The measurement model is the existing SPP model. The only new
//! layout is the parameter vector, ordered as a shared ECEF position followed
//! by epoch-local receiver clocks.
//!
//! ```
//! use sidereon_core::ephemeris::EphemerisSource;
//! use sidereon_core::static_positioning::{
//!     solve_static, StaticSolveError, StaticSolveOptions,
//! };
//! use sidereon_core::GnssSatelliteId;
//!
//! struct EmptyEphemeris;
//!
//! impl EphemerisSource for EmptyEphemeris {
//!     fn position_clock_at_j2000_s(
//!         &self,
//!         _sat: GnssSatelliteId,
//!         _t_j2000_s: f64,
//!     ) -> Option<([f64; 3], f64)> {
//!         None
//!     }
//! }
//!
//! let result = solve_static(&EmptyEphemeris, &[], StaticSolveOptions::default());
//! assert!(matches!(result, Err(StaticSolveError::EmptyEpochs)));
//! ```

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{DMatrix, DVector};

use crate::astro::math::least_squares::{
    self, normal_covariance, singular_value_diagnostics, LeastSquaresProblem, SolveOptions, Status,
    TrustRegionSolve,
};
use crate::astro::math::portable;
use crate::astro::math::robust::{huber_weight, mad_scale, RobustError};
use crate::dop::{rotate_covariance_ecef_to_enu_m2, LineOfSight};
use crate::estimation::substrate::frames::geodetic_from_ecef;
use crate::frame::{ItrfPositionM, Wgs84Geodetic};
use crate::geometry_quality::{classify, GeometryQuality, GeometryQualityThresholds};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::sbas::SbasIonoGrid;
use crate::spp::{
    clock_system, clock_systems, ionosphere_for, line_of_sight, lost_grid_coverage, model_env,
    residual_unweighted, rtklib_step, rtklib_step_norm, sat_model, select_at, solve_converged,
    validate_solve_inputs, weighted_design, Corrections, EphemerisSource, GalileoNequickCoeffs,
    IterateState, KlobucharCoeffs, Observation, PseudorangeCode, RejectedSat, RobustConfig,
    SolveInputs, SppError, SppInputErrorKind, SppModelRecipe, SurfaceMet, Ut1TrackedSource, C_M_S,
    MAX_SELECTION_PASSES, SELECTION_STEP_TOL_M,
};
use crate::validate;

const STATIC_SOLVER_GTOL: f64 = 1e-14;
const STATIC_SOLVER_FTOL: f64 = 1e-15;
const STATIC_SOLVER_XTOL: f64 = 1e-14;
const STATIC_SOLVER_MAX_NFEV: usize = 400;
// ECEF coordinates can be close to zero even though the receiver is on Earth.
// The generic sqrt-eps relative step would then perturb a coordinate by a few
// nanometres, and subtracting two ~20,000 km ranges turns that into a noisy
// finite-difference column. A decimetre floor keeps the static range Jacobian
// in its linear regime while avoiding cancellation; clock columns retain the
// generic relative step because their residual dependence is affine.
const STATIC_POSITION_FD_MIN_STEP_M: f64 = 0.1;

/// One receive epoch for [`solve_static`].
///
/// `measurements` are raw pseudorange measurements in meters. `weights`, when
/// present, must be aligned with `measurements` and are multiplied by the
/// existing SPP elevation weights. The clock seed is a receiver clock range
/// bias in meters for this epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticEpoch {
    /// Pseudorange measurements for this receive epoch.
    pub measurements: Vec<Observation>,
    /// Optional positive measurement-weight multipliers aligned with
    /// [`measurements`](Self::measurements).
    pub weights: Option<Vec<f64>>,
    /// Receive epoch, seconds since J2000 in the ephemeris source time scale.
    pub t_rx_j2000_s: f64,
    /// GPS second of day for the receive epoch.
    pub t_rx_second_of_day_s: f64,
    /// Fractional day of year for the receive epoch.
    pub day_of_year: f64,
    /// Initial receiver clock range bias for this epoch, in meters.
    pub clock_initial_m: f64,
    /// Correction terms applied to this epoch.
    pub corrections: Corrections,
    /// Broadcast Klobuchar coefficients used when ionosphere correction is on.
    pub klobuchar: KlobucharCoeffs,
    /// Optional BeiDou-specific Klobuchar coefficients.
    pub beidou_klobuchar: Option<KlobucharCoeffs>,
    /// Optional Galileo-specific NeQuick-G coefficients.
    pub galileo_nequick: Option<GalileoNequickCoeffs>,
    /// Optional SBAS ionosphere grid.
    pub sbas_iono: Option<SbasIonoGrid>,
    /// GLONASS FDMA channel numbers keyed by GLONASS slot.
    pub glonass_channels: BTreeMap<u8, i8>,
    /// Surface meteorology used when troposphere correction is on.
    pub met: SurfaceMet,
    /// Which code the pseudoranges are; see [`PseudorangeCode`].
    pub pseudorange_code: PseudorangeCode,
}

impl StaticEpoch {
    /// Build a static epoch from existing SPP inputs.
    ///
    /// The SPP observations become [`measurements`](Self::measurements),
    /// `initial_guess[3]` becomes [`clock_initial_m`](Self::clock_initial_m),
    /// and the measurement-model fields are copied. The SPP robust option is
    /// not copied because static robust reweighting is configured on
    /// [`StaticSolveOptions`].
    pub fn from_solve_inputs(inputs: SolveInputs) -> Self {
        Self {
            measurements: inputs.observations,
            weights: None,
            t_rx_j2000_s: inputs.t_rx_j2000_s,
            t_rx_second_of_day_s: inputs.t_rx_second_of_day_s,
            day_of_year: inputs.day_of_year,
            clock_initial_m: inputs.initial_guess[3],
            corrections: inputs.corrections,
            klobuchar: inputs.klobuchar,
            beidou_klobuchar: inputs.beidou_klobuchar,
            galileo_nequick: inputs.galileo_nequick,
            sbas_iono: inputs.sbas_iono,
            glonass_channels: inputs.glonass_channels,
            met: inputs.met,
            pseudorange_code: inputs.pseudorange_code,
        }
    }
}

/// Options for [`solve_static`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StaticSolveOptions {
    /// Initial shared receiver ECEF position in meters.
    pub initial_position_m: [f64; 3],
    /// Whether to return the solved position in geodetic coordinates too.
    pub with_geodetic: bool,
    /// Optional Huber iteratively reweighted least-squares configuration.
    pub robust: Option<RobustConfig>,
}

impl StaticSolveOptions {
    /// Build static options from an existing SPP input.
    ///
    /// The initial position and robust configuration are copied. Static epoch
    /// clock seeds are still taken from each [`StaticEpoch`].
    pub fn from_solve_inputs(inputs: &SolveInputs, with_geodetic: bool) -> Self {
        Self {
            initial_position_m: [
                inputs.initial_guess[0],
                inputs.initial_guess[1],
                inputs.initial_guess[2],
            ],
            with_geodetic,
            robust: inputs.robust,
        }
    }
}

impl Default for StaticSolveOptions {
    fn default() -> Self {
        Self {
            initial_position_m: [0.0; 3],
            with_geodetic: false,
            robust: None,
        }
    }
}

/// One solved epoch-local receiver clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticClockBias {
    /// Epoch index in the input slice.
    pub epoch_index: usize,
    /// GNSS system whose receiver clock column this value belongs to.
    pub system: GnssSystem,
    /// Receiver clock bias in seconds.
    pub clock_s: f64,
}

/// State covariance for a static solution.
///
/// The state order is `[x_m, y_m, z_m, epoch0_clock0_m, ...]`, where each clock
/// is a receiver clock range bias in meters. Clock columns are listed in
/// [`StaticSolution::per_epoch_clock`] order.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticCovariance {
    /// Full state covariance in square meters.
    pub state_m2: Vec<Vec<f64>>,
    /// ECEF position covariance block in square meters.
    pub position_ecef_m2: [[f64; 3]; 3],
    /// Local ENU position covariance block in square meters.
    pub position_enu_m2: [[f64; 3]; 3],
}

/// One post-fit residual from the static solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticResidual {
    /// Epoch index in the input slice.
    pub epoch_index: usize,
    /// Satellite associated with this residual.
    pub satellite_id: GnssSatelliteId,
    /// Unweighted observed-minus-computed pseudorange residual, in meters.
    pub residual_m: f64,
    /// Base row weight before robust reweighting.
    pub base_weight: f64,
    /// Final row weight after robust reweighting.
    pub effective_weight: f64,
    /// Ratio `effective_weight / base_weight`.
    pub robust_weight_ratio: f64,
}

/// Status for a leave-one-out diagnostic solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticInfluenceStatus {
    /// The leave-one-out solve completed.
    Solved,
    /// The omitted data left too few measurements.
    TooFewMeasurements,
    /// The omitted data left rank-deficient geometry.
    SingularGeometry,
    /// Input validation failed for the diagnostic subset.
    InvalidInput,
    /// Ephemeris was unavailable for the diagnostic subset.
    EphemerisUnavailable,
    /// The diagnostic subset failed for another solve reason.
    SolveFailed,
}

/// Leave-one-epoch-out diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticEpochInfluence {
    /// Epoch index omitted from the diagnostic solve.
    pub epoch_index: usize,
    /// Number of measurements omitted.
    pub omitted_measurements: usize,
    /// Diagnostic solve status.
    pub status: StaticInfluenceStatus,
    /// Difference `diagnostic_position - full_position`, in ECEF meters.
    pub position_delta_m: Option<[f64; 3]>,
    /// Norm of [`position_delta_m`](Self::position_delta_m), in meters.
    pub position_delta_norm_m: Option<f64>,
    /// Diagnostic solution residual RMS, in meters.
    pub residual_rms_m: Option<f64>,
    /// Minimum robust weight ratio among this epoch's used rows in the full solve.
    pub min_robust_weight_ratio: f64,
}

/// Leave-one-satellite-out diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticSatelliteInfluence {
    /// Epoch index containing the omitted satellite.
    pub epoch_index: usize,
    /// Satellite omitted from the diagnostic solve.
    pub satellite_id: GnssSatelliteId,
    /// Diagnostic solve status.
    pub status: StaticInfluenceStatus,
    /// Difference `diagnostic_position - full_position`, in ECEF meters.
    pub position_delta_m: Option<[f64; 3]>,
    /// Norm of [`position_delta_m`](Self::position_delta_m), in meters.
    pub position_delta_norm_m: Option<f64>,
    /// Diagnostic solution residual RMS, in meters.
    pub residual_rms_m: Option<f64>,
    /// Full-solve residual for this satellite, in meters.
    pub residual_m: f64,
    /// Base row weight before robust reweighting.
    pub base_weight: f64,
    /// Final row weight after robust reweighting.
    pub effective_weight: f64,
    /// Ratio `effective_weight / base_weight`.
    pub robust_weight_ratio: f64,
}

/// Leave-one-satellite-out diagnostic across all epochs where a satellite appears.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticSatelliteBatchInfluence {
    /// Satellite omitted from every epoch where it was used.
    pub satellite_id: GnssSatelliteId,
    /// Number of measurements omitted across the static batch.
    pub omitted_measurements: usize,
    /// Diagnostic solve status.
    pub status: StaticInfluenceStatus,
    /// Difference `diagnostic_position - full_position`, in ECEF meters.
    pub position_delta_m: Option<[f64; 3]>,
    /// Norm of [`position_delta_m`](Self::position_delta_m), in meters.
    pub position_delta_norm_m: Option<f64>,
    /// Diagnostic solution residual RMS, in meters.
    pub residual_rms_m: Option<f64>,
    /// Minimum robust weight ratio among this satellite's used rows in the full solve.
    pub min_robust_weight_ratio: f64,
}

/// Metadata describing the static solve.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticSolutionMetadata {
    /// Number of accepted iterations: the trust-region iterations of every solve
    /// the selection passes and the robust reweighting ran, plus one for each
    /// least-squares step taken between them.
    pub iterations: usize,
    /// Whether the solve converged, as [`crate::spp::SolutionMetadata::converged`].
    pub converged: bool,
    /// How the solve ended, as [`crate::spp::SolutionMetadata::status`].
    pub status: Status,
    /// Number of robust outer iterations performed.
    pub outer_iterations: usize,
    /// Final MAD robust scale in meters, when robust reweighting ran.
    pub final_robust_scale_m: Option<f64>,
    /// Number of measurements used by the final solve.
    pub used_measurements: usize,
    /// Number of fitted state parameters.
    pub n_parameters: usize,
    /// Degrees of freedom, `used_measurements - n_parameters`.
    pub redundancy: isize,
    /// The first UT1 departure the ephemeris source accepted while producing
    /// a satellite state for this solve, under a permissive UT1 policy. `None`
    /// when every state was produced inside UT1 coverage or did not read UT1.
    pub ut1_degraded: Option<crate::astro::time::DegradeReason>,
}

/// Multi-epoch static receiver solution.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticSolution {
    /// Shared receiver position, ITRF/IGS ECEF meters.
    pub position: ItrfPositionM,
    /// Geodetic form of the position, when requested.
    pub geodetic: Option<Wgs84Geodetic>,
    /// Epoch-local receiver clocks in seconds.
    pub per_epoch_clock: Vec<StaticClockBias>,
    /// State covariance from the stacked normal equations.
    pub covariance: StaticCovariance,
    /// Leave-one-epoch-out diagnostics.
    pub per_epoch_influence: Vec<StaticEpochInfluence>,
    /// Leave-one-satellite-out diagnostics.
    pub per_satellite_influence: Vec<StaticSatelliteInfluence>,
    /// Leave-one-satellite-out diagnostics across all epochs per satellite.
    pub per_satellite_batch_influence: Vec<StaticSatelliteBatchInfluence>,
    /// Geometry observability and covariance validation diagnostics.
    pub geometry_quality: GeometryQuality,
    /// Post-fit residuals for all used measurements.
    pub residuals_m: Vec<StaticResidual>,
    /// Used satellites by epoch, in solver row order.
    pub used_sats: Vec<Vec<GnssSatelliteId>>,
    /// Rejected satellites by epoch.
    pub rejected_sats: Vec<Vec<RejectedSat>>,
    /// Iteration and redundancy metadata.
    pub metadata: StaticSolutionMetadata,
}

impl StaticSolution {
    /// Root-mean-square of the unweighted post-fit residuals.
    pub fn residual_rms_m(&self) -> f64 {
        residual_rms(
            &self
                .residuals_m
                .iter()
                .map(|r| r.residual_m)
                .collect::<Vec<_>>(),
        )
    }
}

/// Error returned by [`solve_static`].
#[derive(Debug, Clone)]
pub enum StaticSolveError {
    /// No epochs were supplied.
    EmptyEpochs,
    /// A public static solve input was malformed.
    InvalidInput {
        /// The invalid input field.
        field: &'static str,
        /// The validation failure category.
        kind: SppInputErrorKind,
    },
    /// A per-epoch SPP input was malformed.
    EpochInput {
        /// Epoch index in the input slice.
        epoch_index: usize,
        /// Underlying SPP input error.
        source: SppError,
    },
    /// The same satellite appears twice in one epoch.
    DuplicateObservation {
        /// Epoch index in the input slice.
        epoch_index: usize,
        /// Satellite that was duplicated.
        satellite: GnssSatelliteId,
    },
    /// Too few accepted measurements remained for the stacked state.
    TooFewMeasurements {
        /// Accepted measurement count.
        used: usize,
        /// Required measurement count.
        required: usize,
    },
    /// A satellite lost ephemeris during the solve.
    EphemerisLost {
        /// Epoch index in the input slice.
        epoch_index: usize,
        /// Satellite whose ephemeris was unavailable.
        satellite: GnssSatelliteId,
    },
    /// The stacked design is rank deficient.
    Singular(least_squares::SolveError),
    /// The per-epoch satellite selection did not settle within
    /// [`MAX_SELECTION_PASSES`] passes, as the single-epoch SPP solve fails with
    /// [`SppError::SelectionUnsettled`].
    SelectionUnsettled {
        /// The number of passes run.
        passes: usize,
    },
    /// The ephemeris source refused a satellite state because producing it
    /// reads UT1 outside the UT1 table under a strict UT1 policy. The solve
    /// fails rather than dropping that satellite.
    Ut1OutsideCoverage(crate::astro::time::DegradeReason),
}

impl core::fmt::Display for StaticSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyEpochs => write!(f, "no static epochs supplied"),
            Self::InvalidInput { field, kind } => {
                write!(f, "invalid static solve input {field}: {kind}")
            }
            Self::EpochInput {
                epoch_index,
                source,
            } => write!(f, "invalid static epoch {epoch_index}: {source}"),
            Self::DuplicateObservation {
                epoch_index,
                satellite,
            } => write!(
                f,
                "static epoch {epoch_index} observes satellite {satellite} more than once"
            ),
            Self::TooFewMeasurements { used, required } => write!(
                f,
                "only {used} usable static measurements; need at least {required}"
            ),
            Self::EphemerisLost {
                epoch_index,
                satellite,
            } => write!(
                f,
                "static epoch {epoch_index} satellite {satellite} lost ephemeris during the solve"
            ),
            Self::Singular(error) => write!(f, "static geometry is singular: {error}"),
            Self::SelectionUnsettled { passes } => write!(
                f,
                "the static satellite selection did not settle in {passes} passes"
            ),
            Self::Ut1OutsideCoverage(reason) => {
                write!(
                    f,
                    "the ephemeris source refused a satellite state: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for StaticSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::EpochInput { source, .. } => Some(source),
            Self::Singular(error) => Some(error),
            _ => None,
        }
    }
}

/// Solve one static receiver position from multiple epochs of pseudoranges.
///
/// The stacked state has one shared ECEF position and epoch-local receiver
/// clocks. If an epoch contains several GNSS clock systems, that epoch gets one
/// clock column per system, matching the single-epoch SPP clock model.
///
/// A satellite state the ephemeris source refuses because producing it reads
/// UT1 outside the UT1 table under a strict UT1 policy fails the solve with
/// [`StaticSolveError::Ut1OutsideCoverage`]; a departure accepted under a
/// permissive policy is reported in [`StaticSolutionMetadata::ut1_degraded`].
pub fn solve_static(
    eph: &dyn EphemerisSource,
    epochs: &[StaticEpoch],
    options: StaticSolveOptions,
) -> Result<StaticSolution, StaticSolveError> {
    solve_static_with_model(eph, epochs, options, SppModelRecipe::reference())
}

/// [`solve_static`] with the geometric light-time measurement model the external
/// references (the Go fixture) were computed with: the transmission epoch iterated
/// from the receiver's time tag, which misses the receiver clock offset, where
/// [`solve_static`] places it from the pseudorange as RTKLIB `satposs` does. Only the
/// repository's replay of those references calls it.
#[cfg(feature = "test-replays")]
#[doc(hidden)]
pub fn solve_static_geometric_light_time_replay(
    eph: &dyn EphemerisSource,
    epochs: &[StaticEpoch],
    options: StaticSolveOptions,
) -> Result<StaticSolution, StaticSolveError> {
    solve_static_with_model(
        eph,
        epochs,
        options,
        SppModelRecipe::geometric_light_time_replay(),
    )
}

fn solve_static_with_model(
    eph: &dyn EphemerisSource,
    epochs: &[StaticEpoch],
    options: StaticSolveOptions,
    model: SppModelRecipe,
) -> Result<StaticSolution, StaticSolveError> {
    let tracked = Ut1TrackedSource::new(eph);
    let core = solve_static_core(&tracked, epochs, options, model);
    ut1_refusal(&tracked)?;
    let core = core?;
    let (per_epoch_influence, per_satellite_influence, per_satellite_batch_influence) =
        build_influence(&tracked, epochs, options, &core, model);
    ut1_refusal(&tracked)?;
    let mut solution = core.into_public(
        per_epoch_influence,
        per_satellite_influence,
        per_satellite_batch_influence,
    );
    solution.metadata.ut1_degraded = tracked.departure();
    Ok(solution)
}

pub(crate) fn solve_static_without_influence(
    eph: &dyn EphemerisSource,
    epochs: &[StaticEpoch],
    options: StaticSolveOptions,
) -> Result<StaticSolution, StaticSolveError> {
    let tracked = Ut1TrackedSource::new(eph);
    let core = solve_static_core(&tracked, epochs, options, SppModelRecipe::reference());
    ut1_refusal(&tracked)?;
    let mut solution = core?.into_public(Vec::new(), Vec::new(), Vec::new());
    solution.metadata.ut1_degraded = tracked.departure();
    Ok(solution)
}

/// A UT1 refusal seen by `tracked` fails the solve, taking precedence over the
/// error or the reduced solution the missing state led to.
fn ut1_refusal(tracked: &Ut1TrackedSource<'_>) -> Result<(), StaticSolveError> {
    match tracked.refusal() {
        Some(reason) => Err(StaticSolveError::Ut1OutsideCoverage(reason)),
        None => Ok(()),
    }
}

#[derive(Debug, Clone)]
struct PreparedEpoch {
    input_index: usize,
    inputs: SolveInputs,
    used: Vec<GnssSatelliteId>,
    rejected: Vec<RejectedSat>,
    systems: Vec<GnssSystem>,
    clock_offset: usize,
    obs_by_id: Vec<(GnssSatelliteId, f64)>,
}

#[derive(Debug, Clone, Copy)]
struct RowRef {
    epoch_index: usize,
    satellite_id: GnssSatelliteId,
    base_weight: f64,
}

#[derive(Debug, Clone)]
struct PreparedStatic {
    epochs: Vec<PreparedEpoch>,
    rows: Vec<RowRef>,
    base_weights: Vec<f64>,
    /// Per row: the line of sight at the state the rows were selected at.
    lines_of_sight: Vec<LineOfSight>,
    /// Per row: the state column of the row's clock.
    clock_columns: Vec<usize>,
    /// Per row: the residual at the state the rows were selected at.
    selection_residuals_m: Vec<f64>,
    /// The state the rows were selected at, in the stacked layout.
    x0: DVector<f64>,
    n_params: usize,
}

impl PreparedStatic {
    /// Each epoch's first clock column and clock systems.
    fn clock_groups(&self) -> Vec<(usize, &[GnssSystem])> {
        self.epochs
            .iter()
            .map(|epoch| (epoch.clock_offset, epoch.systems.as_slice()))
            .collect()
    }

    /// Whether every epoch uses the satellites it uses in `other`.
    fn same_sets(&self, other: &Self) -> bool {
        self.epochs.len() == other.epochs.len()
            && self
                .epochs
                .iter()
                .zip(&other.epochs)
                .all(|(a, b)| a.used == b.used)
    }
}

#[derive(Debug, Clone)]
struct CoreStaticSolution {
    position: ItrfPositionM,
    geodetic: Option<Wgs84Geodetic>,
    per_epoch_clock: Vec<StaticClockBias>,
    covariance: StaticCovariance,
    geometry_quality: GeometryQuality,
    residuals_m: Vec<StaticResidual>,
    used_sats: Vec<Vec<GnssSatelliteId>>,
    rejected_sats: Vec<Vec<RejectedSat>>,
    metadata: StaticSolutionMetadata,
}

impl CoreStaticSolution {
    fn into_public(
        self,
        per_epoch_influence: Vec<StaticEpochInfluence>,
        per_satellite_influence: Vec<StaticSatelliteInfluence>,
        per_satellite_batch_influence: Vec<StaticSatelliteBatchInfluence>,
    ) -> StaticSolution {
        StaticSolution {
            position: self.position,
            geodetic: self.geodetic,
            per_epoch_clock: self.per_epoch_clock,
            covariance: self.covariance,
            per_epoch_influence,
            per_satellite_influence,
            per_satellite_batch_influence,
            geometry_quality: self.geometry_quality,
            residuals_m: self.residuals_m,
            used_sats: self.used_sats,
            rejected_sats: self.rejected_sats,
            metadata: self.metadata,
        }
    }
}

fn solve_static_core(
    eph: &dyn EphemerisSource,
    epochs: &[StaticEpoch],
    options: StaticSolveOptions,
    model: SppModelRecipe,
) -> Result<CoreStaticSolution, StaticSolveError> {
    validate_static_options(options)?;
    if epochs.is_empty() {
        return Err(StaticSolveError::EmptyEpochs);
    }
    let epoch_inputs = validated_epoch_inputs(epochs, options)?;

    let opts = SolveOptions {
        gtol: STATIC_SOLVER_GTOL,
        ftol: STATIC_SOLVER_FTOL,
        xtol: STATIC_SOLVER_XTOL,
        max_nfev: STATIC_SOLVER_MAX_NFEV,
    };
    // Preserve the existing one-epoch SPP-equivalent path. The absolute
    // position floor is needed for the shared-position columns introduced by
    // stacked epochs; one epoch remains the established SPP compatibility path.
    let stacked = epochs.len() > 1;

    // Each epoch is selected, masked and weighted at the current iterate, as the
    // single-epoch SPP solve is ([`crate::spp`]): an iterate whose selection is new
    // in any epoch runs the trust-region solve over the stacked rows, and an
    // iterate whose selection holds in every epoch takes RTKLIB's least-squares
    // step over the stacked design `[-e, 1]`. The solve stops after the first step
    // below `SELECTION_STEP_TOL_M`, which it keeps. One epoch is the SPP solve pass
    // for pass.
    let mut state = StaticState::new(epochs, options);
    let mut iterations = 0usize;
    let mut passes = 0usize;
    // The rows of the last pass, when its selection can end the solve.
    let mut last: Option<PreparedStatic> = None;
    let mut prepared = loop {
        let prepared = prepare_static(eph, epochs, &epoch_inputs, model, &state)?;
        if last
            .as_ref()
            .is_some_and(|last_prepared| prepared.same_sets(last_prepared))
        {
            // The selection holds in every epoch: the step RTKLIB takes here over
            // the stacked design, from this iterate's residuals and weights.
            let dx = rtklib_step(
                &prepared.lines_of_sight,
                &prepared.clock_columns,
                prepared.n_params,
                &prepared.base_weights,
                &prepared.selection_residuals_m,
            )
            .ok_or(StaticSolveError::Singular(
                least_squares::SolveError::SingularJacobian,
            ))?;
            if passes == MAX_SELECTION_PASSES {
                return Err(StaticSolveError::SelectionUnsettled { passes });
            }
            let x = &prepared.x0 + DVector::from_row_slice(&dx);
            state.update(&prepared, &x);
            passes += 1;
            iterations += 1;
            if rtklib_step_norm(&dx, &prepared.clock_groups()) < SELECTION_STEP_TOL_M {
                // The rows at the position reached when every epoch keeps its
                // selection there, else the rows stepped with, evaluated there.
                let post = prepare_static(eph, epochs, &epoch_inputs, model, &state)?;
                if post.same_sets(&prepared) {
                    break post;
                }
                match residual_static_unweighted(eph, &prepared, x.as_slice(), model) {
                    Ok(_) => break prepared,
                    Err((epoch_index, satellite))
                        if static_coverage_lost(
                            eph,
                            &prepared,
                            model,
                            epoch_index,
                            satellite,
                            x.as_slice(),
                        ) =>
                    {
                        // A satellite left the augmentation grid with the last
                        // step: the selection changed.
                        last = None;
                        continue;
                    }
                    Err((epoch_index, satellite)) => {
                        return Err(StaticSolveError::EphemerisLost {
                            epoch_index,
                            satellite,
                        })
                    }
                }
            }
            last = Some(prepared);
            continue;
        }
        if passes == MAX_SELECTION_PASSES {
            return Err(StaticSolveError::SelectionUnsettled { passes });
        }
        // A new selection: the trust-region solve over it, from this iterate. Its
        // pass cannot end the solve; the next iterate is selected again.
        let weights = prepared.base_weights.clone();
        let end = solve_prepared(eph, &prepared, model, &opts, stacked, &weights)?;
        passes += 1;
        match end {
            StaticPassEnd::Solved(report) => {
                iterations += report.iterations;
                state.update(&prepared, &report.x);
                last = Some(prepared);
            }
            StaticPassEnd::CoverageLost(x) => {
                // The last accepted iterate: its selection, if the same, is stepped
                // from rather than solved again.
                state.update(&prepared, &x);
                last = Some(prepared);
            }
        }
    };

    let mut status = Status::SelectionSettled;
    let mut final_weights = prepared.base_weights.clone();
    let mut outer_iterations = 0usize;
    let mut final_robust_scale_m = None;

    // Huber reweighting from the settled solve: each iteration weights the
    // selection at the current state by `huber(r / s)` and re-solves; it settles
    // when the position moves less than `outer_tol_m` and the selection at the new
    // state is the one solved with, and a solve whose budget runs out first ends
    // with `Status::OuterBudgetExhausted`. The reported rows are the ones the last
    // solve used, at the state it reached.
    if let Some(robust) = options.robust {
        let mut current = prepare_static(eph, epochs, &epoch_inputs, model, &state)?;
        let mut settled = false;
        // How the last solve ended; a least-squares step ends at its own target.
        let mut last_inner = Status::SelectionSettled;
        // After a coverage loss the rows at the last accepted iterate, when every
        // epoch keeps the selection solved over, are stepped from rather than
        // solved again.
        let mut step_next = false;
        for _ in 0..robust.max_outer.saturating_sub(1) {
            let post = &current.selection_residuals_m;
            let scale = mad_scale(post, robust.scale_floor_m).map_err(map_robust_error)?;
            let effective: Vec<f64> = post
                .iter()
                .zip(current.base_weights.iter())
                .map(|(&r, &base)| base * huber_weight(r / scale, robust.huber_k))
                .collect();
            let x_prev = state.position_m();
            outer_iterations += 1;
            if step_next {
                step_next = false;
                let dx = rtklib_step(
                    &current.lines_of_sight,
                    &current.clock_columns,
                    current.n_params,
                    &effective,
                    &current.selection_residuals_m,
                )
                .ok_or(StaticSolveError::Singular(
                    least_squares::SolveError::SingularJacobian,
                ))?;
                let x = &current.x0 + DVector::from_row_slice(&dx);
                state.update(&current, &x);
                iterations += 1;
                last_inner = Status::SelectionSettled;
            } else {
                match solve_prepared(eph, &current, model, &opts, stacked, &effective)? {
                    StaticPassEnd::Solved(report) => {
                        iterations += report.iterations;
                        last_inner = report.status;
                        state.update(&current, &report.x);
                    }
                    StaticPassEnd::CoverageLost(x) => {
                        // The last accepted iterate, reported with the rows and
                        // elevation weights there until a reweighted solve or step
                        // replaces them.
                        state.update(&current, &x);
                        let next = prepare_static(eph, epochs, &epoch_inputs, model, &state)?;
                        step_next = next.same_sets(&current);
                        current = next;
                        prepared = current.clone();
                        final_weights = prepared.base_weights.clone();
                        final_robust_scale_m = None;
                        continue;
                    }
                }
            }
            final_weights = effective;
            final_robust_scale_m = Some(scale);
            let x_now = state.position_m();
            let dx = x_now[0] - x_prev[0];
            let dy = x_now[1] - x_prev[1];
            let dz = x_now[2] - x_prev[2];
            let dpos = (dx * dx + dy * dy + dz * dz).sqrt();
            let next = prepare_static(eph, epochs, &epoch_inputs, model, &state)?;
            let same_sets = next.same_sets(&current);
            prepared = std::mem::replace(&mut current, next);
            if !same_sets {
                let x_now = state.parameters(&prepared);
                if let Err((epoch_index, satellite)) =
                    residual_static_unweighted(eph, &prepared, x_now.as_slice(), model)
                {
                    if !static_coverage_lost(
                        eph,
                        &prepared,
                        model,
                        epoch_index,
                        satellite,
                        x_now.as_slice(),
                    ) {
                        return Err(StaticSolveError::EphemerisLost {
                            epoch_index,
                            satellite,
                        });
                    }
                    // A satellite left the augmentation grid: report the rows and
                    // elevation weights at the state reached.
                    prepared = current.clone();
                    final_weights = prepared.base_weights.clone();
                    final_robust_scale_m = None;
                }
            }
            if dpos < robust.outer_tol_m && same_sets {
                settled = true;
                break;
            }
        }
        status = if outer_iterations == 0 {
            Status::SelectionSettled
        } else if !solve_converged(last_inner) {
            last_inner
        } else if settled {
            Status::SelectionSettled
        } else {
            Status::OuterBudgetExhausted
        };
    }

    // The covariance and geometry diagnostics are the design at the reported
    // state, with the reported weights.
    let x = state.parameters(&prepared);
    let jacobian = static_jacobian(eph, &prepared, model, stacked, &x, &final_weights)?;
    finish_static(FinishStaticInput {
        eph,
        prepared: &prepared,
        options,
        model,
        x: x.as_slice(),
        jacobian: &jacobian,
        iterations,
        status,
        outer_iterations,
        final_robust_scale_m,
        final_weights: &final_weights,
    })
}

/// The line of sight of each row of `prepared` at the stacked state `x`.
fn static_lines_of_sight(
    eph: &dyn EphemerisSource,
    prepared: &PreparedStatic,
    x: &[f64],
    model: SppModelRecipe,
) -> Result<Vec<LineOfSight>, StaticSolveError> {
    let rx = [x[0], x[1], x[2]];
    let mut out = Vec::with_capacity(prepared.rows.len());
    for epoch in &prepared.epochs {
        let env = model_env(eph, &epoch.inputs, model, None);
        for &sat in &epoch.used {
            let lost = StaticSolveError::EphemerisLost {
                epoch_index: epoch.input_index,
                satellite: sat,
            };
            let p_meas = epoch
                .obs_by_id
                .iter()
                .find(|(id, _)| *id == sat)
                .map(|(_, p)| *p)
                .ok_or(lost.clone())?;
            let column = epoch.clock_offset
                + epoch
                    .systems
                    .iter()
                    .position(|s| *s == clock_system(sat.system))
                    .unwrap_or(0);
            let m = sat_model(
                &env,
                sat,
                rx,
                x[column],
                p_meas,
                ionosphere_for(sat.system, &epoch.inputs),
            )
            .ok_or(lost)?;
            out.push(line_of_sight(m.sat_rot_ecef_m, rx));
        }
    }
    Ok(out)
}

/// How one trust-region pass over the stacked rows ended.
enum StaticPassEnd {
    /// The solve over the rows ran to its end.
    Solved(least_squares::LeastSquaresReport),
    /// A satellite's line of sight left the augmentation ionosphere grid at a
    /// state the solve evaluated; the pass ends at the last accepted iterate, the
    /// given stacked state, as the SPP pass does.
    CoverageLost(DVector<f64>),
}

/// Whether `satellite` of epoch `epoch_index`, which has no model at the stacked
/// state `x`, has one there with the augmentation grid set aside.
fn static_coverage_lost(
    eph: &dyn EphemerisSource,
    prepared: &PreparedStatic,
    model: SppModelRecipe,
    epoch_index: usize,
    satellite: GnssSatelliteId,
    x: &[f64],
) -> bool {
    let Some(epoch) = prepared
        .epochs
        .iter()
        .find(|epoch| epoch.input_index == epoch_index)
    else {
        return false;
    };
    let Some(p_meas) = epoch
        .obs_by_id
        .iter()
        .find(|(id, _)| *id == satellite)
        .map(|(_, p)| *p)
    else {
        return false;
    };
    let column = epoch.clock_offset
        + epoch
            .systems
            .iter()
            .position(|s| *s == clock_system(satellite.system))
            .unwrap_or(0);
    let env = model_env(eph, &epoch.inputs, model, None);
    lost_grid_coverage(
        &env,
        &epoch.inputs,
        satellite,
        [x[0], x[1], x[2]],
        x[column],
        p_meas,
    )
}

/// The iterate of a static solve: the shared position and each epoch's clocks.
struct StaticState {
    epochs: Vec<IterateState>,
}

impl StaticState {
    fn new(epochs: &[StaticEpoch], options: StaticSolveOptions) -> Self {
        let p = options.initial_position_m;
        Self {
            epochs: epochs
                .iter()
                .map(|epoch| {
                    IterateState::from_initial_guess([p[0], p[1], p[2], epoch.clock_initial_m])
                })
                .collect(),
        }
    }

    fn position_m(&self) -> [f64; 3] {
        self.epochs
            .first()
            .map_or([0.0; 3], |epoch| epoch.rx_ecef_m)
    }

    /// The state in the stacked layout of `prepared`: the position, then each
    /// epoch's clocks for the systems it has columns for.
    fn parameters(&self, prepared: &PreparedStatic) -> DVector<f64> {
        let mut x = self.position_m().to_vec();
        for (state, epoch) in self.epochs.iter().zip(&prepared.epochs) {
            x.extend(epoch.systems.iter().map(|&system| state.clock_m(system)));
        }
        DVector::from_vec(x)
    }

    /// Take the position and each epoch's clocks from the solution `x` of the
    /// stacked layout `prepared`. An epoch with no clock column keeps its clocks.
    fn update(&mut self, prepared: &PreparedStatic, x: &DVector<f64>) {
        for (state, epoch) in self.epochs.iter_mut().zip(&prepared.epochs) {
            let mut local = vec![x[0], x[1], x[2]];
            local.extend((0..epoch.systems.len()).map(|i| x[epoch.clock_offset + i]));
            if epoch.systems.is_empty() {
                state.rx_ecef_m = [x[0], x[1], x[2]];
            } else {
                state.update(&epoch.systems, &DVector::from_vec(local));
            }
        }
    }
}

/// The weighted-residual Jacobian of the stacked rows of `prepared` at `x`, by the
/// forward difference the trust-region solve takes.
fn static_jacobian(
    eph: &dyn EphemerisSource,
    prepared: &PreparedStatic,
    model: SppModelRecipe,
    stacked: bool,
    x: &DVector<f64>,
    weights: &[f64],
) -> Result<DMatrix<f64>, StaticSolveError> {
    let lost = Cell::new(None::<(usize, GnssSatelliteId)>);
    let sqrt_weights = DVector::from_row_slice(weights).map(f64::sqrt);
    let residual = |state: &DVector<f64>| -> DVector<f64> {
        match residual_static_unweighted(eph, prepared, state.as_slice(), model) {
            Ok(values) => DVector::from_vec(values).component_mul(&sqrt_weights),
            Err((epoch_index, satellite)) => {
                lost.set(Some((epoch_index, satellite)));
                DVector::zeros(prepared.rows.len())
            }
        }
    };
    let min_steps = if stacked {
        static_fd_min_steps(prepared.n_params)
    } else {
        DVector::zeros(prepared.n_params)
    };
    let f0 = residual(x);
    let jacobian = least_squares::jacobian_2point_with_min_steps(residual, x, &f0, &min_steps);
    if let Some((epoch_index, satellite)) = lost.get() {
        return Err(StaticSolveError::EphemerisLost {
            epoch_index,
            satellite,
        });
    }
    jacobian.map_err(StaticSolveError::Singular)
}

/// One trust-region solve over the stacked rows of `prepared` with `weights`, from
/// the state `prepared` was made at.
fn solve_prepared(
    eph: &dyn EphemerisSource,
    prepared: &PreparedStatic,
    model: SppModelRecipe,
    opts: &SolveOptions,
    stacked: bool,
    weights: &[f64],
) -> Result<StaticPassEnd, StaticSolveError> {
    // The first measurement that has no model, and the state it had none at, are
    // recorded, and the closure returns a non-finite residual, which stops the
    // solve at once.
    let lost = std::cell::RefCell::new(None::<(usize, GnssSatelliteId, DVector<f64>)>);
    let residual = |x: &DVector<f64>| -> DVector<f64> {
        match residual_static_unweighted(eph, prepared, x.as_slice(), model) {
            Ok(values) => DVector::from_vec(values),
            Err((epoch_index, satellite)) => {
                let mut lost = lost.borrow_mut();
                if lost.is_none() {
                    *lost = Some((epoch_index, satellite, x.clone()));
                }
                DVector::from_element(prepared.rows.len(), f64::NAN)
            }
        }
    };
    let fd_min_steps = if stacked {
        static_fd_min_steps(prepared.n_params)
    } else {
        DVector::zeros(prepared.n_params)
    };
    let problem = LeastSquaresProblem::with_weights_and_fd_min_steps(
        &residual,
        prepared.x0.clone(),
        DVector::from_row_slice(weights),
        fd_min_steps,
    );
    let mut last_accepted: Option<DVector<f64>> = None;
    let result =
        least_squares::solve_trf_observed(&problem, opts, TrustRegionSolve::NalgebraLu, &mut |x| {
            last_accepted = Some(x.clone())
        });
    if let Some((epoch_index, satellite, x)) = lost.into_inner() {
        if static_coverage_lost(eph, prepared, model, epoch_index, satellite, x.as_slice()) {
            if let Some(accepted) = last_accepted {
                return Ok(StaticPassEnd::CoverageLost(accepted));
            }
        }
        return Err(StaticSolveError::EphemerisLost {
            epoch_index,
            satellite,
        });
    }
    result
        .map(StaticPassEnd::Solved)
        .map_err(StaticSolveError::Singular)
}

fn static_fd_min_steps(n_params: usize) -> DVector<f64> {
    DVector::from_iterator(
        n_params,
        (0..n_params).map(|index| {
            if index < 3 {
                STATIC_POSITION_FD_MIN_STEP_M
            } else {
                0.0
            }
        }),
    )
}

/// Each epoch's SPP inputs, validated once for every pass.
fn validated_epoch_inputs(
    epochs: &[StaticEpoch],
    options: StaticSolveOptions,
) -> Result<Vec<SolveInputs>, StaticSolveError> {
    epochs
        .iter()
        .enumerate()
        .map(|(epoch_index, epoch)| {
            validate_epoch_weights(epoch)?;
            if let Some(satellite) = duplicate_satellite(&epoch.measurements) {
                return Err(StaticSolveError::DuplicateObservation {
                    epoch_index,
                    satellite,
                });
            }
            let inputs = solve_inputs_for_epoch(epoch, options);
            validate_solve_inputs(&inputs).map_err(|source| StaticSolveError::EpochInput {
                epoch_index,
                source,
            })?;
            Ok(inputs)
        })
        .collect()
}

/// The stacked rows at `state`: each epoch's selection at the shared position
/// and that epoch's clocks.
fn prepare_static(
    eph: &dyn EphemerisSource,
    epochs: &[StaticEpoch],
    epoch_inputs: &[SolveInputs],
    model: SppModelRecipe,
    state: &StaticState,
) -> Result<PreparedStatic, StaticSolveError> {
    let mut prepared_epochs = Vec::with_capacity(epochs.len());
    let mut rows = Vec::new();
    let mut base_weights = Vec::new();
    let mut lines_of_sight = Vec::new();
    let mut clock_columns = Vec::new();
    let mut selection_residuals_m = Vec::new();
    let position = state.position_m();
    let mut x0 = position.to_vec();
    let mut clock_offset = 3usize;

    for (epoch_index, (epoch, inputs)) in epochs.iter().zip(epoch_inputs).enumerate() {
        // A satellite whose carrier an ionosphere-corrected epoch cannot scale
        // to is excluded by the selection, reported in `rejected_sats` with
        // `RejectionReason::IonosphereCarrierUnresolved`, and the epoch keeps
        // its other satellites.
        let epoch_state = &state.epochs[epoch_index];
        let selection = select_at(eph, inputs, model, None, position, &|system| {
            epoch_state.clock_m(system)
        });
        let systems = clock_systems(&selection.used);
        let weight_by_sat = measurement_weight_map(epoch);
        let obs_by_id: Vec<(GnssSatelliteId, f64)> = inputs
            .observations
            .iter()
            .map(|m| (m.satellite_id, m.pseudorange_m))
            .collect();

        for (row_idx, &satellite_id) in selection.used.iter().enumerate() {
            let multiplier = weight_by_sat.get(&satellite_id).copied().unwrap_or(1.0);
            let base_weight = selection.weights[row_idx] * multiplier;
            rows.push(RowRef {
                epoch_index,
                satellite_id,
                base_weight,
            });
            base_weights.push(base_weight);
            lines_of_sight.push(selection.lines_of_sight[row_idx]);
            selection_residuals_m.push(selection.residuals_m[row_idx]);
            let system_index = systems
                .iter()
                .position(|s| *s == clock_system(satellite_id.system))
                .unwrap_or(0);
            clock_columns.push(clock_offset + system_index);
        }

        x0.extend(systems.iter().map(|&system| epoch_state.clock_m(system)));

        prepared_epochs.push(PreparedEpoch {
            input_index: epoch_index,
            inputs: inputs.clone(),
            used: selection.used,
            rejected: selection.rejected,
            systems,
            clock_offset,
            obs_by_id,
        });
        clock_offset += prepared_epochs
            .last()
            .expect("prepared epoch just pushed")
            .systems
            .len();
    }

    let n_params = x0.len();
    if rows.len() < n_params {
        return Err(StaticSolveError::TooFewMeasurements {
            used: rows.len(),
            required: n_params,
        });
    }

    Ok(PreparedStatic {
        epochs: prepared_epochs,
        rows,
        base_weights,
        lines_of_sight,
        clock_columns,
        selection_residuals_m,
        x0: DVector::from_vec(x0),
        n_params,
    })
}

fn solve_inputs_for_epoch(epoch: &StaticEpoch, options: StaticSolveOptions) -> SolveInputs {
    SolveInputs {
        observations: epoch.measurements.clone(),
        t_rx_j2000_s: epoch.t_rx_j2000_s,
        t_rx_second_of_day_s: epoch.t_rx_second_of_day_s,
        day_of_year: epoch.day_of_year,
        initial_guess: [
            options.initial_position_m[0],
            options.initial_position_m[1],
            options.initial_position_m[2],
            epoch.clock_initial_m,
        ],
        corrections: epoch.corrections,
        klobuchar: epoch.klobuchar,
        beidou_klobuchar: epoch.beidou_klobuchar,
        galileo_nequick: epoch.galileo_nequick,
        sbas_iono: epoch.sbas_iono.clone(),
        glonass_channels: epoch.glonass_channels.clone(),
        met: epoch.met,
        robust: None,
        pseudorange_code: epoch.pseudorange_code,
    }
}

struct FinishStaticInput<'a> {
    eph: &'a dyn EphemerisSource,
    prepared: &'a PreparedStatic,
    options: StaticSolveOptions,
    model: SppModelRecipe,
    x: &'a [f64],
    jacobian: &'a DMatrix<f64>,
    iterations: usize,
    status: Status,
    outer_iterations: usize,
    final_robust_scale_m: Option<f64>,
    final_weights: &'a [f64],
}

fn finish_static(input: FinishStaticInput<'_>) -> Result<CoreStaticSolution, StaticSolveError> {
    let FinishStaticInput {
        eph,
        prepared,
        options,
        model,
        x,
        jacobian,
        iterations,
        status,
        outer_iterations,
        final_robust_scale_m,
        final_weights,
    } = input;
    let position = ItrfPositionM::new(x[0], x[1], x[2]).expect("valid static position");
    let receiver_geodetic = geodetic_from_ecef(model.frame, [x[0], x[1], x[2]]);
    let geodetic = if options.with_geodetic {
        Some(receiver_geodetic)
    } else {
        None
    };
    let per_epoch_clock = epoch_clocks(prepared, x);
    let residual_values = residual_static_unweighted(eph, prepared, x, model).map_err(
        |(epoch_index, satellite)| StaticSolveError::EphemerisLost {
            epoch_index,
            satellite,
        },
    )?;
    let residuals_m = residual_values
        .iter()
        .zip(prepared.rows.iter())
        .zip(final_weights.iter())
        .map(|((&residual_m, row), &effective_weight)| StaticResidual {
            epoch_index: row.epoch_index,
            satellite_id: row.satellite_id,
            residual_m,
            base_weight: row.base_weight,
            effective_weight,
            robust_weight_ratio: effective_weight / row.base_weight,
        })
        .collect::<Vec<_>>();

    let covariance_matrix = normal_covariance(jacobian, 1.0).map_err(StaticSolveError::Singular)?;
    let covariance = static_covariance(&covariance_matrix, receiver_geodetic)?;
    // The rank, singular values and condition number are those of the design
    // RTKLIB solves with, `sqrt(W) [-e, 1]` at the reported state with the reported
    // weights, as SPP reports them. The covariance keeps the finite-difference
    // design of the solve's own model, which carries the atmospheric and Sagnac
    // derivatives.
    let lines_of_sight = static_lines_of_sight(eph, prepared, x, model)?;
    let design = weighted_design(
        &lines_of_sight,
        &prepared.clock_columns,
        prepared.n_params,
        final_weights,
    )
    .ok_or(StaticSolveError::Singular(
        least_squares::SolveError::SingularJacobian,
    ))?;
    let svd = portable::svd(&design, false, false);
    let singular_values: Vec<f64> = svd.singular_values.iter().map(|value| value.0).collect();
    let diagnostics = singular_value_diagnostics(&singular_values, design.nrows(), design.ncols());
    if diagnostics.rank < prepared.n_params {
        return Err(StaticSolveError::Singular(
            least_squares::SolveError::SingularJacobian,
        ));
    }
    let gdop = covariance_trace(&covariance_matrix).sqrt();
    let redundancy = prepared.rows.len() as isize - prepared.n_params as isize;
    let geometry_quality = classify(
        diagnostics.rank,
        prepared.n_params,
        redundancy as i32,
        diagnostics.condition_number,
        gdop,
        false,
        GeometryQualityThresholds::default(),
    );
    let converged = solve_converged(status);

    Ok(CoreStaticSolution {
        position,
        geodetic,
        per_epoch_clock,
        covariance,
        geometry_quality,
        residuals_m,
        used_sats: prepared
            .epochs
            .iter()
            .map(|epoch| epoch.used.clone())
            .collect(),
        rejected_sats: prepared
            .epochs
            .iter()
            .map(|epoch| epoch.rejected.clone())
            .collect(),
        metadata: StaticSolutionMetadata {
            iterations,
            converged,
            status,
            outer_iterations,
            final_robust_scale_m,
            used_measurements: prepared.rows.len(),
            n_parameters: prepared.n_params,
            redundancy,
            ut1_degraded: None,
        },
    })
}

fn residual_static_unweighted(
    eph: &dyn EphemerisSource,
    prepared: &PreparedStatic,
    x: &[f64],
    model: SppModelRecipe,
) -> Result<Vec<f64>, (usize, GnssSatelliteId)> {
    let mut out = Vec::with_capacity(prepared.rows.len());
    for epoch in &prepared.epochs {
        if epoch.used.is_empty() {
            continue;
        }
        let mut local = Vec::with_capacity(3 + epoch.systems.len());
        local.extend_from_slice(&x[0..3]);
        for clock_idx in 0..epoch.systems.len() {
            local.push(x[epoch.clock_offset + clock_idx]);
        }
        let residuals = residual_unweighted(
            eph,
            &epoch.used,
            &epoch.obs_by_id,
            &local,
            &epoch.inputs,
            model,
        )
        .map_err(|satellite| (epoch.input_index, satellite))?;
        out.extend(residuals);
    }
    Ok(out)
}

fn static_covariance(
    covariance: &DMatrix<f64>,
    receiver: Wgs84Geodetic,
) -> Result<StaticCovariance, StaticSolveError> {
    let state_m2 = matrix_to_rows(covariance);
    let position_ecef_m2 = [
        [covariance[(0, 0)], covariance[(0, 1)], covariance[(0, 2)]],
        [covariance[(1, 0)], covariance[(1, 1)], covariance[(1, 2)]],
        [covariance[(2, 0)], covariance[(2, 1)], covariance[(2, 2)]],
    ];
    let position_enu_m2 = rotate_covariance_ecef_to_enu_m2(position_ecef_m2, receiver)
        .map_err(|_| StaticSolveError::Singular(least_squares::SolveError::SingularJacobian))?;
    Ok(StaticCovariance {
        state_m2,
        position_ecef_m2,
        position_enu_m2,
    })
}

fn epoch_clocks(prepared: &PreparedStatic, x: &[f64]) -> Vec<StaticClockBias> {
    let mut clocks = Vec::new();
    for epoch in &prepared.epochs {
        for (clock_idx, &system) in epoch.systems.iter().enumerate() {
            clocks.push(StaticClockBias {
                epoch_index: epoch.input_index,
                system,
                clock_s: x[epoch.clock_offset + clock_idx] / C_M_S,
            });
        }
    }
    clocks
}

fn validate_static_options(options: StaticSolveOptions) -> Result<(), StaticSolveError> {
    validate::finite_slice(&options.initial_position_m, "initial_position_m")
        .map_err(map_static_input_error)?;
    if let Some(robust) = options.robust {
        if robust.max_outer == 0 {
            return Err(StaticSolveError::InvalidInput {
                field: "robust.max_outer",
                kind: SppInputErrorKind::NotPositive,
            });
        }
        validate::finite_positive(robust.huber_k, "robust.huber_k")
            .map_err(map_static_input_error)?;
        validate::finite_positive(robust.scale_floor_m, "robust.scale_floor_m")
            .map_err(map_static_input_error)?;
        validate::finite_positive(robust.outer_tol_m, "robust.outer_tol_m")
            .map_err(map_static_input_error)?;
    }
    Ok(())
}

fn validate_epoch_weights(epoch: &StaticEpoch) -> Result<(), StaticSolveError> {
    if let Some(weights) = &epoch.weights {
        if weights.len() != epoch.measurements.len() {
            return Err(StaticSolveError::InvalidInput {
                field: "epoch.weights",
                kind: SppInputErrorKind::OutOfRange,
            });
        }
        for &weight in weights {
            validate::finite_positive(weight, "epoch.weights").map_err(map_static_input_error)?;
        }
    }
    Ok(())
}

fn measurement_weight_map(epoch: &StaticEpoch) -> BTreeMap<GnssSatelliteId, f64> {
    epoch
        .weights
        .as_ref()
        .map(|weights| {
            epoch
                .measurements
                .iter()
                .zip(weights.iter())
                .map(|(measurement, &weight)| (measurement.satellite_id, weight))
                .collect()
        })
        .unwrap_or_default()
}

fn duplicate_satellite(measurements: &[Observation]) -> Option<GnssSatelliteId> {
    let mut ids: Vec<GnssSatelliteId> = measurements.iter().map(|m| m.satellite_id).collect();
    ids.sort_unstable();
    ids.windows(2)
        .find(|pair| pair[0] == pair[1])
        .map(|pair| pair[0])
}

fn build_influence(
    eph: &dyn EphemerisSource,
    epochs: &[StaticEpoch],
    options: StaticSolveOptions,
    full: &CoreStaticSolution,
    model: SppModelRecipe,
) -> (
    Vec<StaticEpochInfluence>,
    Vec<StaticSatelliteInfluence>,
    Vec<StaticSatelliteBatchInfluence>,
) {
    let epoch_influence = (0..epochs.len())
        .map(|epoch_index| {
            let mut subset = epochs.to_vec();
            let omitted_measurements = subset[epoch_index].measurements.len();
            subset.remove(epoch_index);
            let result = solve_static_core(eph, &subset, options, model);
            let (status, position_delta_m, position_delta_norm_m, residual_rms_m) =
                influence_result(full.position.as_array(), result);
            StaticEpochInfluence {
                epoch_index,
                omitted_measurements,
                status,
                position_delta_m,
                position_delta_norm_m,
                residual_rms_m,
                min_robust_weight_ratio: min_epoch_weight_ratio(full, epoch_index),
            }
        })
        .collect();

    let satellite_ids = full
        .residuals_m
        .iter()
        .map(|row| row.satellite_id)
        .collect::<BTreeSet<_>>();
    let satellite_batch_influence = satellite_ids
        .into_iter()
        .map(|satellite_id| {
            let subset = omit_satellite_all_epochs(epochs, satellite_id);
            let result = solve_static_core(eph, &subset, options, model);
            let (status, position_delta_m, position_delta_norm_m, residual_rms_m) =
                influence_result(full.position.as_array(), result);
            StaticSatelliteBatchInfluence {
                satellite_id,
                omitted_measurements: full
                    .residuals_m
                    .iter()
                    .filter(|row| row.satellite_id == satellite_id)
                    .count(),
                status,
                position_delta_m,
                position_delta_norm_m,
                residual_rms_m,
                min_robust_weight_ratio: min_satellite_weight_ratio(full, satellite_id),
            }
        })
        .collect();

    let satellite_influence = full
        .residuals_m
        .iter()
        .map(|row| {
            let subset = omit_satellite(epochs, row.epoch_index, row.satellite_id);
            let result = solve_static_core(eph, &subset, options, model);
            let (status, position_delta_m, position_delta_norm_m, residual_rms_m) =
                influence_result(full.position.as_array(), result);
            StaticSatelliteInfluence {
                epoch_index: row.epoch_index,
                satellite_id: row.satellite_id,
                status,
                position_delta_m,
                position_delta_norm_m,
                residual_rms_m,
                residual_m: row.residual_m,
                base_weight: row.base_weight,
                effective_weight: row.effective_weight,
                robust_weight_ratio: row.robust_weight_ratio,
            }
        })
        .collect();

    (
        epoch_influence,
        satellite_influence,
        satellite_batch_influence,
    )
}

fn omit_satellite(
    epochs: &[StaticEpoch],
    epoch_index: usize,
    satellite_id: GnssSatelliteId,
) -> Vec<StaticEpoch> {
    let mut subset = epochs.to_vec();
    remove_satellite_from_epoch(&mut subset[epoch_index], satellite_id);
    subset
}

fn omit_satellite_all_epochs(
    epochs: &[StaticEpoch],
    satellite_id: GnssSatelliteId,
) -> Vec<StaticEpoch> {
    let mut subset = epochs.to_vec();
    for epoch in &mut subset {
        remove_satellite_from_epoch(epoch, satellite_id);
    }
    subset
}

fn remove_satellite_from_epoch(epoch: &mut StaticEpoch, satellite_id: GnssSatelliteId) {
    let old_measurements = std::mem::take(&mut epoch.measurements);
    let old_weights = epoch.weights.take();
    let mut measurements = Vec::with_capacity(old_measurements.len());
    let mut weights = old_weights
        .as_ref()
        .map(|_| Vec::with_capacity(old_measurements.len()));
    for (idx, measurement) in old_measurements.into_iter().enumerate() {
        if measurement.satellite_id == satellite_id {
            continue;
        }
        measurements.push(measurement);
        if let (Some(old), Some(new_weights)) = (&old_weights, &mut weights) {
            new_weights.push(old[idx]);
        }
    }
    epoch.measurements = measurements;
    epoch.weights = weights;
}

fn influence_result(
    full_position: [f64; 3],
    result: Result<CoreStaticSolution, StaticSolveError>,
) -> (
    StaticInfluenceStatus,
    Option<[f64; 3]>,
    Option<f64>,
    Option<f64>,
) {
    match result {
        Ok(solution) => {
            let position = solution.position.as_array();
            let delta = [
                position[0] - full_position[0],
                position[1] - full_position[1],
                position[2] - full_position[2],
            ];
            let norm = (delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2]).sqrt();
            let residual_values = solution
                .residuals_m
                .iter()
                .map(|row| row.residual_m)
                .collect::<Vec<_>>();
            (
                StaticInfluenceStatus::Solved,
                Some(delta),
                Some(norm),
                Some(residual_rms(&residual_values)),
            )
        }
        Err(error) => (influence_status(&error), None, None, None),
    }
}

fn influence_status(error: &StaticSolveError) -> StaticInfluenceStatus {
    match error {
        StaticSolveError::TooFewMeasurements { .. } | StaticSolveError::EmptyEpochs => {
            StaticInfluenceStatus::TooFewMeasurements
        }
        StaticSolveError::Singular(_) => StaticInfluenceStatus::SingularGeometry,
        StaticSolveError::InvalidInput { .. }
        | StaticSolveError::EpochInput { .. }
        | StaticSolveError::DuplicateObservation { .. } => StaticInfluenceStatus::InvalidInput,
        StaticSolveError::EphemerisLost { .. } | StaticSolveError::Ut1OutsideCoverage(_) => {
            StaticInfluenceStatus::EphemerisUnavailable
        }
        StaticSolveError::SelectionUnsettled { .. } => StaticInfluenceStatus::SolveFailed,
    }
}

fn min_epoch_weight_ratio(full: &CoreStaticSolution, epoch_index: usize) -> f64 {
    full.residuals_m
        .iter()
        .filter(|row| row.epoch_index == epoch_index)
        .map(|row| row.robust_weight_ratio)
        .fold(1.0, f64::min)
}

fn min_satellite_weight_ratio(full: &CoreStaticSolution, satellite_id: GnssSatelliteId) -> f64 {
    full.residuals_m
        .iter()
        .filter(|row| row.satellite_id == satellite_id)
        .map(|row| row.robust_weight_ratio)
        .fold(1.0, f64::min)
}

fn matrix_to_rows(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

fn covariance_trace(matrix: &DMatrix<f64>) -> f64 {
    (0..matrix.nrows().min(matrix.ncols()))
        .map(|idx| matrix[(idx, idx)])
        .sum()
}

fn residual_rms(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let sum_sq = residuals.iter().map(|r| r * r).sum::<f64>();
    (sum_sq / residuals.len() as f64).sqrt()
}

fn map_static_input_error(error: validate::FieldError) -> StaticSolveError {
    StaticSolveError::InvalidInput {
        field: error.field(),
        kind: SppInputErrorKind::from(&error),
    }
}

fn map_robust_error(error: RobustError) -> StaticSolveError {
    let field = match error.field() {
        "scale_floor" => "robust.scale_floor_m",
        "residuals" | "values" => "robust.residuals",
        other => other,
    };
    let kind = match error.reason() {
        "not finite" => SppInputErrorKind::NonFinite,
        "not positive" => SppInputErrorKind::NotPositive,
        "negative" => SppInputErrorKind::Negative,
        "out of range" => SppInputErrorKind::OutOfRange,
        _ => SppInputErrorKind::OutOfRange,
    };
    StaticSolveError::InvalidInput { field, kind }
}

#[cfg(test)]
mod tests;
