//! Geodetic position time-series velocity, trajectory, step, and field tools.
//!
//! This module is sans-IO. Callers supply positions already decoded from their
//! product format, with epochs in decimal years and coordinates in either local
//! ENU metres or ITRF/ECEF metres.

use core::fmt;

use nalgebra::DMatrix;
pub use trust_region_least_squares::loss::Loss;
use trust_region_least_squares::model::{solve_model_with, ResidualModel};
use trust_region_least_squares::trf::{TrfError, TrfOptions};

use crate::astro::math::least_squares::{
    covariance_from_jacobian, normal_covariance, singular_value_diagnostics,
};
use crate::astro::math::portable::{self, PortableNumerics};
use crate::astro::math::robust::{median, RobustError};
use crate::dop;
use crate::estimation::{mad_spread, PrimitiveError};
use crate::frame::{geodetic_to_itrf, Wgs84Geodetic};
use crate::geometry_quality::{
    classify, GeometryQuality, GeometryQualityThresholds, ObservabilityTier,
};

const DEFAULT_MIDAS_PERIOD_YEARS: f64 = 1.0;
const DEFAULT_MIDAS_PERIOD_TOLERANCE_YEARS: f64 = 0.001;
const DEFAULT_MIDAS_MIN_PAIRS: usize = 3;
const DEFAULT_STEP_WINDOW_YEARS: f64 = 0.75;
const DEFAULT_STEP_SCORE_THRESHOLD: f64 = 8.0;
const DEFAULT_STEP_MIN_OFFSET_M: f64 = 1.0e-4;
const DEFAULT_STEP_MIN_SAMPLES_EACH_SIDE: usize = 4;
const DEFAULT_STEP_MIN_SEPARATION_YEARS: f64 = 0.25;
const STEP_ZERO_OFFSET_TOLERANCE_M: f64 = 1.0e-12;
const TAU: f64 = core::f64::consts::PI * 2.0;

/// One position sample in a station time series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionSample {
    /// Epoch expressed as a decimal year in a continuous time scale.
    pub epoch_year: f64,
    /// Position vector in metres, interpreted by [`PositionFrame`].
    pub position_m: [f64; 3],
    /// Optional 3x3 coordinate covariance in square metres, in the same frame
    /// as [`position_m`](Self::position_m).
    pub covariance_m2: Option<[[f64; 3]; 3]>,
}

/// Coordinate frame of the supplied position samples.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PositionFrame {
    /// Position vectors are local east, north, up coordinates in metres.
    Enu,
    /// Position vectors are ITRF/ECEF metres and are differenced from the
    /// supplied reference before rotation into local ENU.
    Ecef {
        /// Geodetic reference position used to define the local ENU frame.
        reference: Wgs84Geodetic,
    },
}

/// Borrowed station time series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionSeries<'a> {
    /// Frame and reference metadata for every sample in the series.
    pub frame: PositionFrame,
    /// Position samples. They may be unsorted; duplicate epochs are rejected.
    pub samples: &'a [PositionSample],
}

/// Options for [`velocity_midas`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MidasOptions {
    /// Dominant period used for pair selection, in years.
    pub dominant_period_years: f64,
    /// Allowed absolute difference from the dominant period, in years, before
    /// the selector falls back to the nearest later sample.
    pub period_tolerance_years: f64,
    /// Minimum retained pair count required for each component.
    pub min_pairs: usize,
}

impl Default for MidasOptions {
    fn default() -> Self {
        Self {
            dominant_period_years: DEFAULT_MIDAS_PERIOD_YEARS,
            period_tolerance_years: DEFAULT_MIDAS_PERIOD_TOLERANCE_YEARS,
            min_pairs: DEFAULT_MIDAS_MIN_PAIRS,
        }
    }
}

/// Qualitative strength of a time-series estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeSeriesQuality {
    /// The series has at least one dominant period of span and enough selected
    /// pairs for the requested estimator.
    Nominal,
    /// The estimate is usable but has less than three dominant periods of span,
    /// so MIDAS does not have its single-step robustness guarantee.
    ShortSpan,
}

/// MIDAS diagnostics for one ENU component.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MidasComponentStats {
    /// Pair slopes selected before the MIDAS two-sigma trim.
    pub pair_count: usize,
    /// Pair slopes retained after the MIDAS two-sigma trim.
    pub retained_pair_count: usize,
    /// Robust standard deviation of retained pair slopes in metres per year.
    pub slope_sigma_m_per_yr: f64,
    /// Effective number of independent slope samples, `retained_pair_count / 4`.
    pub effective_pair_count: f64,
}

/// Robust ENU velocity estimate.
#[derive(Debug, Clone, PartialEq)]
pub struct Velocity {
    /// Velocity components `[east, north, up]` in metres per year.
    pub rate_enu_m_per_yr: [f64; 3],
    /// One-sigma MIDAS uncertainties `[east, north, up]` in metres per year.
    pub sigma_enu_m_per_yr: [f64; 3],
    /// Diagonal ENU velocity covariance in square metres per square year.
    pub covariance_enu_m2_per_yr2: [[f64; 3]; 3],
    /// Per-component MIDAS slope statistics.
    pub component_stats: [MidasComponentStats; 3],
    /// Number of position samples accepted after sorting and validation.
    pub sample_count: usize,
    /// Series span in years.
    pub span_years: f64,
    /// Strength flag for the estimate.
    pub quality: TimeSeriesQuality,
}

/// Trajectory model terms used by [`fit_trajectory`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TrajectoryTerm {
    /// Position at the reference epoch, metres.
    Position,
    /// Linear velocity, metres per year.
    Velocity,
    /// Annual sine coefficient, metres.
    AnnualSin,
    /// Annual cosine coefficient, metres.
    AnnualCos,
    /// Semiannual sine coefficient, metres.
    SemiannualSin,
    /// Semiannual cosine coefficient, metres.
    SemiannualCos,
    /// Heaviside offset coefficient, metres.
    Offset {
        /// Offset index in [`TrajectoryModel::offset_epochs_year`].
        index: usize,
        /// Offset epoch in decimal years.
        epoch_year: f64,
    },
}

/// Linear trajectory model shape for [`fit_trajectory`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryModel {
    /// Optional reference epoch. When `None`, the mean sample epoch is used.
    pub reference_epoch_year: Option<f64>,
    /// Include annual sine and cosine terms.
    pub include_annual: bool,
    /// Include semiannual sine and cosine terms.
    pub include_semiannual: bool,
    /// Known offset epochs modeled with a Heaviside step.
    pub offset_epochs_year: Vec<f64>,
}

impl Default for TrajectoryModel {
    fn default() -> Self {
        Self {
            reference_epoch_year: None,
            include_annual: true,
            include_semiannual: true,
            offset_epochs_year: Vec::new(),
        }
    }
}

/// Least-squares controls for [`fit_trajectory`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct TrajectoryFitOptions {
    /// Robust loss passed to the trust-region least-squares solver.
    pub loss: Loss,
    /// Robust-loss scale in metres. Ignored when [`Loss::Linear`] is selected.
    pub f_scale_m: f64,
    /// Optional maximum residual evaluations for the trust-region solve.
    pub max_nfev: Option<usize>,
}

impl Default for TrajectoryFitOptions {
    fn default() -> Self {
        Self {
            loss: Loss::Linear,
            f_scale_m: 1.0,
            max_nfev: None,
        }
    }
}

/// Fitted trajectory coefficients for one ENU component.
#[derive(Debug, Clone, PartialEq)]
pub struct TrajectoryComponent {
    /// Position at the reference epoch, metres.
    pub position_m: f64,
    /// Linear velocity in metres per year.
    pub velocity_m_per_yr: f64,
    /// Annual sine coefficient in metres, or `None` when the term is omitted.
    pub annual_sin_m: Option<f64>,
    /// Annual cosine coefficient in metres, or `None` when the term is omitted.
    pub annual_cos_m: Option<f64>,
    /// Semiannual sine coefficient in metres, or `None` when the term is
    /// omitted.
    pub semiannual_sin_m: Option<f64>,
    /// Semiannual cosine coefficient in metres, or `None` when the term is
    /// omitted.
    pub semiannual_cos_m: Option<f64>,
    /// Heaviside offset coefficients in metres, ordered like the model offsets.
    pub offsets_m: Vec<f64>,
}

/// Trajectory least-squares result.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory {
    /// Reference epoch used for the position parameter and harmonic phases.
    pub reference_epoch_year: f64,
    /// Parameter terms within each component block.
    pub terms: Vec<TrajectoryTerm>,
    /// ENU component coefficients, ordered `[east, north, up]`.
    pub components: [TrajectoryComponent; 3],
    /// Full parameter covariance in solver order. The order is all terms for
    /// east, then all terms for north, then all terms for up.
    pub parameter_covariance: Vec<Vec<f64>>,
    /// Root-mean-square residuals `[east, north, up]` in metres.
    pub residual_rms_enu_m: [f64; 3],
    /// Design observability and covariance-validation diagnostics.
    pub geometry_quality: GeometryQuality,
    /// Trust-region termination status.
    pub status: i32,
    /// Residual evaluations used by the solver.
    pub nfev: usize,
    /// Jacobian evaluations used by the solver.
    pub njev: usize,
    /// Final least-squares cost.
    pub cost: f64,
    /// Infinity norm of the final gradient.
    pub optimality: f64,
}

/// Controls for [`detect_steps`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StepDetectionOptions {
    /// Half-window around a candidate epoch, in years.
    pub window_years: f64,
    /// Minimum robust normalized offset score to report.
    pub score_threshold: f64,
    /// Minimum three-dimensional offset norm in metres to report.
    pub min_offset_m: f64,
    /// Minimum number of samples required on each side of a candidate.
    pub min_samples_each_side: usize,
    /// Minimum separation retained between reported candidates, in years.
    pub min_separation_years: f64,
    /// MIDAS controls used to detrend the series before scoring steps.
    pub midas: MidasOptions,
}

impl Default for StepDetectionOptions {
    fn default() -> Self {
        Self {
            window_years: DEFAULT_STEP_WINDOW_YEARS,
            score_threshold: DEFAULT_STEP_SCORE_THRESHOLD,
            min_offset_m: DEFAULT_STEP_MIN_OFFSET_M,
            min_samples_each_side: DEFAULT_STEP_MIN_SAMPLES_EACH_SIDE,
            min_separation_years: DEFAULT_STEP_MIN_SEPARATION_YEARS,
            midas: MidasOptions::default(),
        }
    }
}

/// Heuristic used to generate a step candidate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepDetectionHeuristic {
    /// Difference of pre-event and post-event residual medians after MIDAS
    /// detrending, scored by robust local spread.
    DetrendedSlidingMedian,
}

/// Candidate displacement step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCandidate {
    /// Candidate step epoch in decimal years.
    pub epoch_year: f64,
    /// Estimated offset `[east, north, up]` in metres, after minus before.
    pub offset_enu_m: [f64; 3],
    /// Robust normalized offset score. Larger means more step-like.
    pub score: f64,
    /// Number of samples before the candidate used by the score.
    pub before_count: usize,
    /// Number of samples after the candidate used by the score.
    pub after_count: usize,
    /// Explicit label that this diagnostic is a heuristic.
    pub heuristic: StepDetectionHeuristic,
}

/// Network field frame and filtering controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkFrame {
    /// Geodetic origin defining the output local ENU frame.
    pub origin: Wgs84Geodetic,
    /// Remove the unweighted mean velocity across stations in the output frame.
    pub remove_common_mode: bool,
}

/// Station input for [`network_field`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkStation<'a> {
    /// Caller-provided station identifier copied into the output.
    pub id: &'a str,
    /// Station reference position used to rotate a local ENU velocity into ECEF.
    pub reference: Wgs84Geodetic,
    /// Station position time series.
    pub series: PositionSeries<'a>,
}

/// One station motion in a network field.
#[derive(Debug, Clone, PartialEq)]
pub struct StationMotion {
    /// Station identifier copied from [`NetworkStation::id`].
    pub id: String,
    /// Velocity in the network frame after optional common-mode removal.
    pub rate_enu_m_per_yr: [f64; 3],
    /// Velocity in the network frame before common-mode removal.
    pub raw_rate_enu_m_per_yr: [f64; 3],
    /// One-sigma uncertainty in the network frame, square-rooted by component.
    pub sigma_enu_m_per_yr: [f64; 3],
    /// Station-local MIDAS velocity before rotation into the network frame.
    pub local_velocity: Velocity,
}

/// Network motion field.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionField {
    /// Output frame and filtering controls used for this field.
    pub frame: NetworkFrame,
    /// Station motions in the same order as the accepted inputs.
    pub stations: Vec<StationMotion>,
    /// Unweighted mean velocity removed from station rates, or zero when
    /// common-mode removal is disabled.
    pub common_mode_enu_m_per_yr: [f64; 3],
}

/// Error returned by geodetic time-series estimators.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GeodeticTimeSeriesError {
    /// A boundary input was malformed.
    #[error("invalid geodetic time-series input {field}: {reason}")]
    InvalidInput {
        /// Name of the malformed field.
        field: &'static str,
        /// Stable validation reason.
        reason: &'static str,
    },
    /// There are fewer position samples than the estimator requires.
    #[error("geodetic time series has {samples} samples; need at least {needed}")]
    TooFewSamples {
        /// Number of supplied samples.
        samples: usize,
        /// Minimum sample count required.
        needed: usize,
    },
    /// MIDAS could not select enough usable dominant-period pairs.
    #[error("geodetic time series has {pairs} usable pairs; need at least {needed}")]
    InsufficientPairs {
        /// Number of selected or retained pairs.
        pairs: usize,
        /// Minimum pair count required.
        needed: usize,
    },
    /// The trajectory design matrix is rank deficient.
    #[error("trajectory design is rank deficient")]
    SingularTrajectory,
    /// The trust-region least-squares solver exhausted or stopped without
    /// satisfying a convergence condition.
    #[error("trajectory solver did not converge, status {status}")]
    DidNotConverge {
        /// Trust-region status code.
        status: i32,
    },
    /// The trust-region least-squares solver failed.
    #[error("trajectory solver failed: {0}")]
    Solver(TrfError),
}

impl From<TrfError> for GeodeticTimeSeriesError {
    fn from(value: TrfError) -> Self {
        Self::Solver(value)
    }
}

impl From<PrimitiveError> for GeodeticTimeSeriesError {
    fn from(value: PrimitiveError) -> Self {
        match value {
            PrimitiveError::InvalidInput { field, reason } => Self::InvalidInput { field, reason },
        }
    }
}

impl From<RobustError> for GeodeticTimeSeriesError {
    fn from(value: RobustError) -> Self {
        match value {
            RobustError::InvalidInput { field, reason } => Self::InvalidInput { field, reason },
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedSample {
    epoch_year: f64,
    enu_m: [f64; 3],
    covariance_enu_m2: Option<[[f64; 3]; 3]>,
}

#[derive(Debug, Clone, Copy)]
struct Pair {
    first: usize,
    second: usize,
}

/// Estimate robust station velocity with the MIDAS median interannual
/// difference adjusted for skewness estimator.
///
/// Pair slopes are selected near [`MidasOptions::dominant_period_years`]. The
/// component estimate is the median slope, followed by one two-sigma robust MAD
/// trim and a recomputed median. The reported uncertainty is the MIDAS scaled
/// median standard error using `retained_pair_count / 4` independent slopes.
pub fn velocity_midas(
    series: &PositionSeries<'_>,
    options: MidasOptions,
) -> Result<Velocity, GeodeticTimeSeriesError> {
    validate_midas_options(options)?;
    let samples = prepare_samples(series)?;
    if samples.len() < 2 {
        return Err(GeodeticTimeSeriesError::TooFewSamples {
            samples: samples.len(),
            needed: 2,
        });
    }
    let span_years = samples.last().expect("checked nonempty").epoch_year
        - samples.first().expect("checked nonempty").epoch_year;
    if span_years < options.dominant_period_years {
        return Err(GeodeticTimeSeriesError::InsufficientPairs {
            pairs: 0,
            needed: options.min_pairs,
        });
    }

    let pairs = select_midas_pairs(&samples, options);
    if pairs.len() < options.min_pairs {
        return Err(GeodeticTimeSeriesError::InsufficientPairs {
            pairs: pairs.len(),
            needed: options.min_pairs,
        });
    }

    let mut rate = [0.0; 3];
    let mut sigma = [0.0; 3];
    let mut covariance = [[0.0; 3]; 3];
    let mut stats = [MidasComponentStats {
        pair_count: 0,
        retained_pair_count: 0,
        slope_sigma_m_per_yr: 0.0,
        effective_pair_count: 0.0,
    }; 3];

    for axis in 0..3 {
        let component = midas_component(&samples, &pairs, axis, options.min_pairs)?;
        rate[axis] = component.0;
        sigma[axis] = component.1;
        covariance[axis][axis] = component.1 * component.1;
        stats[axis] = component.2;
    }

    Ok(Velocity {
        rate_enu_m_per_yr: rate,
        sigma_enu_m_per_yr: sigma,
        covariance_enu_m2_per_yr2: covariance,
        component_stats: stats,
        sample_count: samples.len(),
        span_years,
        quality: if span_years < 3.0 * options.dominant_period_years {
            TimeSeriesQuality::ShortSpan
        } else {
            TimeSeriesQuality::Nominal
        },
    })
}

/// Fit a linear geodetic trajectory model over the workspace trust-region
/// least-squares solver.
///
/// The model is linear in its coefficients: position at reference epoch,
/// velocity, optional annual and semiannual sine/cosine terms, and known
/// Heaviside offsets. The state order is component-major: all terms for east,
/// all terms for north, then all terms for up.
pub fn fit_trajectory(
    series: &PositionSeries<'_>,
    model: &TrajectoryModel,
    options: TrajectoryFitOptions,
) -> Result<Trajectory, GeodeticTimeSeriesError> {
    validate_fit_options(options)?;
    let samples = prepare_samples(series)?;
    if samples.is_empty() {
        return Err(GeodeticTimeSeriesError::TooFewSamples {
            samples: 0,
            needed: 1,
        });
    }
    validate_model(model)?;
    let reference_epoch_year = model
        .reference_epoch_year
        .unwrap_or_else(|| mean_epoch(&samples));
    let terms = trajectory_terms(model);
    let terms_per_axis = terms.len();
    let n_params = terms_per_axis * 3;
    let m_residuals = samples.len() * 3;
    if m_residuals < n_params {
        return Err(GeodeticTimeSeriesError::TooFewSamples {
            samples: samples.len(),
            needed: n_params.div_ceil(3),
        });
    }

    let problem = TrajectoryProblem {
        samples: &samples,
        terms: &terms,
        reference_epoch_year,
    };
    let x0 = trajectory_initial_guess(&samples, &terms, reference_epoch_year);
    let mut solver_options = TrfOptions {
        loss: options.loss,
        f_scale: options.f_scale_m,
        max_nfev: options.max_nfev,
        ..TrfOptions::default()
    };
    if solver_options.max_nfev.is_none() {
        solver_options.max_nfev = Some(100 * n_params.max(1));
    }
    let solved = solve_model_with(&problem, &x0, &PortableNumerics, &solver_options)?;
    if !solved.success() {
        return Err(GeodeticTimeSeriesError::DidNotConverge {
            status: solved.status,
        });
    }

    let jacobian = DMatrix::from_row_slice(m_residuals, n_params, &solved.jac);
    let covariance = covariance_from_jacobian(&jacobian, solved.cost)
        .map_err(|_| GeodeticTimeSeriesError::SingularTrajectory)?;
    let geometry_quality = trajectory_geometry_quality(&jacobian);
    if geometry_quality.tier == ObservabilityTier::RankDeficient {
        return Err(GeodeticTimeSeriesError::SingularTrajectory);
    }

    let components = [
        trajectory_component(&solved.x, 0, &terms),
        trajectory_component(&solved.x, 1, &terms),
        trajectory_component(&solved.x, 2, &terms),
    ];
    let residual_rms_enu_m = residual_rms(&solved.fun);

    Ok(Trajectory {
        reference_epoch_year,
        terms,
        components,
        parameter_covariance: matrix_to_vecs(&covariance),
        residual_rms_enu_m,
        geometry_quality,
        status: solved.status,
        nfev: solved.nfev,
        njev: solved.njev,
        cost: solved.cost,
        optimality: solved.optimality,
    })
}

/// Detect candidate displacement steps with a labelled heuristic.
///
/// The series is first detrended with [`velocity_midas`]. Each candidate split
/// is scored from the difference of local pre-event and post-event residual
/// medians divided by robust local scatter. Reported candidates are proposals
/// only; they are not inserted into a trajectory model.
pub fn detect_steps(
    series: &PositionSeries<'_>,
    options: StepDetectionOptions,
) -> Result<Vec<StepCandidate>, GeodeticTimeSeriesError> {
    validate_step_options(options)?;
    let samples = prepare_samples(series)?;
    if samples.len() < options.min_samples_each_side * 2 {
        return Err(GeodeticTimeSeriesError::TooFewSamples {
            samples: samples.len(),
            needed: options.min_samples_each_side * 2,
        });
    }
    let velocity = velocity_midas(series, options.midas)?;
    let reference_epoch_year = samples[0].epoch_year;
    let residuals = samples
        .iter()
        .map(|sample| {
            let dt = sample.epoch_year - reference_epoch_year;
            [
                sample.enu_m[0] - velocity.rate_enu_m_per_yr[0] * dt,
                sample.enu_m[1] - velocity.rate_enu_m_per_yr[1] * dt,
                sample.enu_m[2] - velocity.rate_enu_m_per_yr[2] * dt,
            ]
        })
        .collect::<Vec<_>>();

    let mut candidates = Vec::new();
    for split in options.min_samples_each_side..=(samples.len() - options.min_samples_each_side) {
        let epoch = samples[split].epoch_year;
        let before = window_indices(&samples, 0, split, epoch, options.window_years);
        let after = window_indices(&samples, split, samples.len(), epoch, options.window_years);
        if before.len() < options.min_samples_each_side
            || after.len() < options.min_samples_each_side
        {
            continue;
        }
        let (offset, score) = step_score(&residuals, &before, &after)?;
        let offset_norm =
            (offset[0] * offset[0] + offset[1] * offset[1] + offset[2] * offset[2]).sqrt();
        if score >= options.score_threshold && offset_norm >= options.min_offset_m {
            candidates.push(StepCandidate {
                epoch_year: epoch,
                offset_enu_m: offset,
                score,
                before_count: before.len(),
                after_count: after.len(),
                heuristic: StepDetectionHeuristic::DetrendedSlidingMedian,
            });
        }
    }
    candidates.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut retained: Vec<StepCandidate> = Vec::new();
    for candidate in candidates {
        if retained.iter().all(|kept| {
            (kept.epoch_year - candidate.epoch_year).abs() >= options.min_separation_years
        }) {
            retained.push(candidate);
        }
    }
    retained.sort_by(|a, b| a.epoch_year.total_cmp(&b.epoch_year));
    Ok(retained)
}

/// Estimate a network motion field in one local ENU frame.
///
/// Each station is solved independently with [`velocity_midas`], rotated from
/// station-local ENU to ECEF, then into the network frame. If requested, the
/// unweighted mean network-frame velocity is subtracted from every station.
pub fn network_field(
    stations: &[NetworkStation<'_>],
    frame: NetworkFrame,
) -> Result<MotionField, GeodeticTimeSeriesError> {
    validate_geodetic(frame.origin, "frame.origin")?;
    if stations.is_empty() {
        return Err(GeodeticTimeSeriesError::TooFewSamples {
            samples: 0,
            needed: 1,
        });
    }
    let origin_rotation = dop::ecef_to_enu_rotation(frame.origin.lat_rad, frame.origin.lon_rad);
    let mut motions = Vec::with_capacity(stations.len());
    for station in stations {
        validate_geodetic(station.reference, "station.reference")?;
        if station.id.is_empty() {
            return Err(invalid_input("station.id", "empty"));
        }
        let local_velocity = velocity_midas(&station.series, MidasOptions::default())?;
        let station_rotation =
            dop::ecef_to_enu_rotation(station.reference.lat_rad, station.reference.lon_rad);
        let ecef_rate = enu_to_ecef(&station_rotation, local_velocity.rate_enu_m_per_yr);
        let raw_rate = mat3_vec(&origin_rotation, ecef_rate);
        let covariance_network = rotate_velocity_covariance(
            &origin_rotation,
            &station_rotation,
            local_velocity.covariance_enu_m2_per_yr2,
        );
        let sigma = [
            covariance_network[0][0].max(0.0).sqrt(),
            covariance_network[1][1].max(0.0).sqrt(),
            covariance_network[2][2].max(0.0).sqrt(),
        ];
        motions.push(StationMotion {
            id: station.id.to_string(),
            rate_enu_m_per_yr: raw_rate,
            raw_rate_enu_m_per_yr: raw_rate,
            sigma_enu_m_per_yr: sigma,
            local_velocity,
        });
    }

    let common_mode = if frame.remove_common_mode {
        let mut sum = [0.0; 3];
        for motion in &motions {
            for (axis, value) in sum.iter_mut().enumerate() {
                *value += motion.raw_rate_enu_m_per_yr[axis];
            }
        }
        let scale = 1.0 / motions.len() as f64;
        [sum[0] * scale, sum[1] * scale, sum[2] * scale]
    } else {
        [0.0; 3]
    };
    if frame.remove_common_mode {
        for motion in &mut motions {
            for (axis, value) in motion.rate_enu_m_per_yr.iter_mut().enumerate() {
                *value -= common_mode[axis];
            }
        }
    }

    Ok(MotionField {
        frame,
        stations: motions,
        common_mode_enu_m_per_yr: common_mode,
    })
}

fn validate_midas_options(options: MidasOptions) -> Result<(), GeodeticTimeSeriesError> {
    validate_positive(options.dominant_period_years, "dominant_period_years")?;
    validate_nonnegative(options.period_tolerance_years, "period_tolerance_years")?;
    if options.min_pairs == 0 {
        return Err(invalid_input("min_pairs", "must be positive"));
    }
    Ok(())
}

fn validate_fit_options(options: TrajectoryFitOptions) -> Result<(), GeodeticTimeSeriesError> {
    if options.loss != Loss::Linear {
        validate_positive(options.f_scale_m, "f_scale_m")?;
    } else {
        validate_finite(options.f_scale_m, "f_scale_m")?;
    }
    if options.max_nfev == Some(0) {
        return Err(invalid_input("max_nfev", "must be positive"));
    }
    Ok(())
}

fn validate_step_options(options: StepDetectionOptions) -> Result<(), GeodeticTimeSeriesError> {
    validate_midas_options(options.midas)?;
    validate_positive(options.window_years, "window_years")?;
    validate_positive(options.score_threshold, "score_threshold")?;
    validate_nonnegative(options.min_offset_m, "min_offset_m")?;
    validate_nonnegative(options.min_separation_years, "min_separation_years")?;
    if options.min_samples_each_side == 0 {
        return Err(invalid_input("min_samples_each_side", "must be positive"));
    }
    Ok(())
}

fn validate_model(model: &TrajectoryModel) -> Result<(), GeodeticTimeSeriesError> {
    if let Some(reference) = model.reference_epoch_year {
        validate_finite(reference, "reference_epoch_year")?;
    }
    for epoch in &model.offset_epochs_year {
        validate_finite(*epoch, "offset_epochs_year")?;
    }
    Ok(())
}

fn prepare_samples(
    series: &PositionSeries<'_>,
) -> Result<Vec<PreparedSample>, GeodeticTimeSeriesError> {
    if series.samples.is_empty() {
        return Err(GeodeticTimeSeriesError::TooFewSamples {
            samples: 0,
            needed: 1,
        });
    }
    let (reference_ecef_m, rotation) = match series.frame {
        PositionFrame::Enu => (None, None),
        PositionFrame::Ecef { reference } => {
            validate_geodetic(reference, "reference")?;
            let ecef = geodetic_to_itrf(reference)
                .map_err(|_| invalid_input("reference", "ECEF conversion failed"))?;
            (
                Some(ecef.as_array()),
                Some(dop::ecef_to_enu_rotation(
                    reference.lat_rad,
                    reference.lon_rad,
                )),
            )
        }
    };

    let mut samples = Vec::with_capacity(series.samples.len());
    for sample in series.samples {
        validate_finite(sample.epoch_year, "epoch_year")?;
        validate_vec3(sample.position_m, "position_m")?;
        let (enu_m, covariance_enu_m2) = match series.frame {
            PositionFrame::Enu => {
                let covariance = match sample.covariance_m2 {
                    Some(covariance) => {
                        validate_covariance(covariance, "covariance_m2")?;
                        Some(covariance)
                    }
                    None => None,
                };
                (sample.position_m, covariance)
            }
            PositionFrame::Ecef { .. } => {
                let reference = reference_ecef_m.expect("ECEF reference exists");
                let rotation = rotation.expect("ECEF rotation exists");
                let delta = [
                    sample.position_m[0] - reference[0],
                    sample.position_m[1] - reference[1],
                    sample.position_m[2] - reference[2],
                ];
                let covariance = match sample.covariance_m2 {
                    Some(covariance) => {
                        validate_covariance(covariance, "covariance_m2")?;
                        let rotated = rotate_covariance(&rotation, covariance);
                        validate_covariance_diagonal(rotated, "covariance_m2")?;
                        Some(rotated)
                    }
                    None => None,
                };
                (mat3_vec(&rotation, delta), covariance)
            }
        };
        samples.push(PreparedSample {
            epoch_year: sample.epoch_year,
            enu_m,
            covariance_enu_m2,
        });
    }
    samples.sort_by(|a, b| a.epoch_year.total_cmp(&b.epoch_year));
    for pair in samples.windows(2) {
        if pair[0].epoch_year == pair[1].epoch_year {
            return Err(invalid_input("epoch_year", "duplicate"));
        }
    }
    Ok(samples)
}

fn select_midas_pairs(samples: &[PreparedSample], options: MidasOptions) -> Vec<Pair> {
    let mut pairs = Vec::new();
    select_midas_pairs_forward(samples, options, &mut pairs);
    let reversed = samples.iter().rev().cloned().collect::<Vec<_>>();
    let mut reverse_pairs = Vec::new();
    select_midas_pairs_forward(&reversed, options, &mut reverse_pairs);
    let n = samples.len();
    for pair in reverse_pairs {
        let first = n - 1 - pair.second;
        let second = n - 1 - pair.first;
        pairs.push(Pair { first, second });
    }
    pairs.sort_by_key(|pair| (pair.first, pair.second));
    pairs.dedup_by_key(|pair| (pair.first, pair.second));
    pairs
}

fn select_midas_pairs_forward(
    samples: &[PreparedSample],
    options: MidasOptions,
    pairs: &mut Vec<Pair>,
) {
    for first in 0..samples.len() {
        let mut best: Option<(usize, f64, bool)> = None;
        for second in (first + 1)..samples.len() {
            let dt = samples[second].epoch_year - samples[first].epoch_year;
            if dt <= 0.0 {
                continue;
            }
            let distance = (dt - options.dominant_period_years).abs();
            let in_window = distance <= options.period_tolerance_years;
            if dt < options.dominant_period_years - options.period_tolerance_years {
                continue;
            }
            match best {
                None => best = Some((second, distance, in_window)),
                Some((_, best_distance, best_in_window)) => {
                    let better = if in_window != best_in_window {
                        in_window
                    } else {
                        distance < best_distance
                    };
                    if better {
                        best = Some((second, distance, in_window));
                    }
                }
            }
            if in_window && distance == 0.0 {
                break;
            }
            if dt > options.dominant_period_years + options.period_tolerance_years
                && best.map(|(_, _, in_window)| in_window).unwrap_or(false)
            {
                break;
            }
        }
        if let Some((second, _, _)) = best {
            pairs.push(Pair { first, second });
        }
    }
}

fn midas_component(
    samples: &[PreparedSample],
    pairs: &[Pair],
    axis: usize,
    min_pairs: usize,
) -> Result<(f64, f64, MidasComponentStats), GeodeticTimeSeriesError> {
    let slopes = pairs
        .iter()
        .map(|pair| {
            let first = &samples[pair.first];
            let second = &samples[pair.second];
            (second.enu_m[axis] - first.enu_m[axis]) / (second.epoch_year - first.epoch_year)
        })
        .collect::<Vec<_>>();
    if slopes.len() < min_pairs {
        return Err(GeodeticTimeSeriesError::InsufficientPairs {
            pairs: slopes.len(),
            needed: min_pairs,
        });
    }
    let initial_median = median(&slopes)?;
    let initial_sigma = mad_spread(&slopes, 0.0)?;
    let retained = slopes
        .iter()
        .copied()
        .filter(|slope| {
            let deviation = (*slope - initial_median).abs();
            if initial_sigma == 0.0 {
                deviation == 0.0
            } else {
                deviation < 2.0 * initial_sigma
            }
        })
        .collect::<Vec<_>>();
    if retained.len() < min_pairs {
        return Err(GeodeticTimeSeriesError::InsufficientPairs {
            pairs: retained.len(),
            needed: min_pairs,
        });
    }
    let final_median = median(&retained)?;
    let final_sigma = mad_spread(&retained, 0.0)?;
    let effective_pair_count = retained.len() as f64 / 4.0;
    let uncertainty =
        3.0 * (core::f64::consts::PI / 2.0).sqrt() * final_sigma / effective_pair_count.sqrt();
    Ok((
        final_median,
        uncertainty,
        MidasComponentStats {
            pair_count: slopes.len(),
            retained_pair_count: retained.len(),
            slope_sigma_m_per_yr: final_sigma,
            effective_pair_count,
        },
    ))
}

fn trajectory_terms(model: &TrajectoryModel) -> Vec<TrajectoryTerm> {
    let mut terms = vec![TrajectoryTerm::Position, TrajectoryTerm::Velocity];
    if model.include_annual {
        terms.push(TrajectoryTerm::AnnualSin);
        terms.push(TrajectoryTerm::AnnualCos);
    }
    if model.include_semiannual {
        terms.push(TrajectoryTerm::SemiannualSin);
        terms.push(TrajectoryTerm::SemiannualCos);
    }
    for (index, &epoch_year) in model.offset_epochs_year.iter().enumerate() {
        terms.push(TrajectoryTerm::Offset { index, epoch_year });
    }
    terms
}

fn basis_value(term: TrajectoryTerm, epoch_year: f64, reference_epoch_year: f64) -> f64 {
    let dt = epoch_year - reference_epoch_year;
    match term {
        TrajectoryTerm::Position => 1.0,
        TrajectoryTerm::Velocity => dt,
        TrajectoryTerm::AnnualSin => libm::sin(TAU * dt),
        TrajectoryTerm::AnnualCos => libm::cos(TAU * dt),
        TrajectoryTerm::SemiannualSin => libm::sin(2.0 * TAU * dt),
        TrajectoryTerm::SemiannualCos => libm::cos(2.0 * TAU * dt),
        TrajectoryTerm::Offset { epoch_year, .. } => {
            let step_dt = epoch_year - reference_epoch_year;
            if dt > step_dt {
                1.0
            } else if dt == step_dt {
                0.5
            } else {
                0.0
            }
        }
    }
}

fn trajectory_initial_guess(
    samples: &[PreparedSample],
    terms: &[TrajectoryTerm],
    reference_epoch_year: f64,
) -> Vec<f64> {
    let mut x0 = vec![0.0; terms.len() * 3];
    let first = samples.first().expect("nonempty samples");
    let last = samples.last().expect("nonempty samples");
    let span = (last.epoch_year - first.epoch_year).max(f64::MIN_POSITIVE);
    for axis in 0..3 {
        let base = axis * terms.len();
        let rate = (last.enu_m[axis] - first.enu_m[axis]) / span;
        for (term_index, term) in terms.iter().enumerate() {
            x0[base + term_index] = match term {
                TrajectoryTerm::Position => {
                    first.enu_m[axis] + rate * (reference_epoch_year - first.epoch_year)
                }
                TrajectoryTerm::Velocity => rate,
                _ => 0.0,
            };
        }
    }
    x0
}

struct TrajectoryProblem<'a> {
    samples: &'a [PreparedSample],
    terms: &'a [TrajectoryTerm],
    reference_epoch_year: f64,
}

impl ResidualModel for TrajectoryProblem<'_> {
    fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        let terms_per_axis = self.terms.len();
        for sample in self.samples {
            for axis in 0..3 {
                let base = axis * terms_per_axis;
                let mut predicted = 0.0;
                for (term_index, &term) in self.terms.iter().enumerate() {
                    predicted += x[base + term_index]
                        * basis_value(term, sample.epoch_year, self.reference_epoch_year);
                }
                let residual = predicted - sample.enu_m[axis];
                out.push(residual * sqrt_weight(sample, axis));
            }
        }
    }

    fn jacobian(&self, _x: &[f64], _f0: &[f64], out: &mut Vec<f64>) {
        out.clear();
        let terms_per_axis = self.terms.len();
        let n = terms_per_axis * 3;
        out.resize(self.samples.len() * 3 * n, 0.0);
        for (sample_index, sample) in self.samples.iter().enumerate() {
            for axis in 0..3 {
                let row = sample_index * 3 + axis;
                let base = axis * terms_per_axis;
                let weight = sqrt_weight(sample, axis);
                for (term_index, &term) in self.terms.iter().enumerate() {
                    out[row * n + base + term_index] =
                        basis_value(term, sample.epoch_year, self.reference_epoch_year) * weight;
                }
            }
        }
    }
}

fn sqrt_weight(sample: &PreparedSample, axis: usize) -> f64 {
    match sample.covariance_enu_m2 {
        Some(covariance) => {
            let variance = covariance[axis][axis];
            if variance > 0.0 {
                variance.sqrt().recip()
            } else {
                1.0
            }
        }
        None => 1.0,
    }
}

fn trajectory_component(x: &[f64], axis: usize, terms: &[TrajectoryTerm]) -> TrajectoryComponent {
    let base = axis * terms.len();
    let mut component = TrajectoryComponent {
        position_m: 0.0,
        velocity_m_per_yr: 0.0,
        annual_sin_m: None,
        annual_cos_m: None,
        semiannual_sin_m: None,
        semiannual_cos_m: None,
        offsets_m: Vec::new(),
    };
    for (term_index, term) in terms.iter().enumerate() {
        let value = x[base + term_index];
        match term {
            TrajectoryTerm::Position => component.position_m = value,
            TrajectoryTerm::Velocity => component.velocity_m_per_yr = value,
            TrajectoryTerm::AnnualSin => component.annual_sin_m = Some(value),
            TrajectoryTerm::AnnualCos => component.annual_cos_m = Some(value),
            TrajectoryTerm::SemiannualSin => component.semiannual_sin_m = Some(value),
            TrajectoryTerm::SemiannualCos => component.semiannual_cos_m = Some(value),
            TrajectoryTerm::Offset { .. } => component.offsets_m.push(value),
        }
    }
    component
}

fn trajectory_geometry_quality(jacobian: &DMatrix<f64>) -> GeometryQuality {
    let svd = portable::svd(jacobian, false, false);
    let singular_values: Vec<f64> = svd.singular_values.iter().map(|value| value.0).collect();
    let diagnostics =
        singular_value_diagnostics(&singular_values, jacobian.nrows(), jacobian.ncols());
    let gdop = normal_covariance(jacobian, 1.0)
        .map(|covariance| {
            let trace = (0..covariance.nrows())
                .map(|idx| covariance[(idx, idx)])
                .sum::<f64>();
            if trace >= 0.0 && trace.is_finite() {
                trace.sqrt()
            } else {
                f64::INFINITY
            }
        })
        .unwrap_or(f64::INFINITY);
    classify(
        diagnostics.rank,
        jacobian.ncols(),
        jacobian.nrows() as i32 - jacobian.ncols() as i32,
        diagnostics.condition_number,
        gdop,
        false,
        GeometryQualityThresholds::default(),
    )
}

fn residual_rms(residuals: &[f64]) -> [f64; 3] {
    let mut sums = [0.0; 3];
    let mut counts = [0usize; 3];
    for (idx, residual) in residuals.iter().enumerate() {
        let axis = idx % 3;
        sums[axis] += residual * residual;
        counts[axis] += 1;
    }
    [
        (sums[0] / counts[0] as f64).sqrt(),
        (sums[1] / counts[1] as f64).sqrt(),
        (sums[2] / counts[2] as f64).sqrt(),
    ]
}

fn mean_epoch(samples: &[PreparedSample]) -> f64 {
    samples.iter().map(|sample| sample.epoch_year).sum::<f64>() / samples.len() as f64
}

fn matrix_to_vecs(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    (0..matrix.nrows())
        .map(|row| (0..matrix.ncols()).map(|col| matrix[(row, col)]).collect())
        .collect()
}

fn window_indices(
    samples: &[PreparedSample],
    start: usize,
    end: usize,
    epoch: f64,
    window_years: f64,
) -> Vec<usize> {
    (start..end)
        .filter(|&idx| (samples[idx].epoch_year - epoch).abs() <= window_years)
        .collect()
}

fn step_score(
    residuals: &[[f64; 3]],
    before: &[usize],
    after: &[usize],
) -> Result<([f64; 3], f64), GeodeticTimeSeriesError> {
    let mut offset = [0.0; 3];
    let mut score_sq = 0.0;
    for axis in 0..3 {
        let before_values = before
            .iter()
            .map(|&idx| residuals[idx][axis])
            .collect::<Vec<_>>();
        let after_values = after
            .iter()
            .map(|&idx| residuals[idx][axis])
            .collect::<Vec<_>>();
        let before_median = median(&before_values)?;
        let after_median = median(&after_values)?;
        let delta = after_median - before_median;
        offset[axis] = delta;
        let mut centered = before_values
            .iter()
            .map(|value| value - before_median)
            .collect::<Vec<_>>();
        centered.extend(after_values.iter().map(|value| value - after_median));
        let spread = mad_spread(&centered, 0.0)?;
        let axis_score = if spread == 0.0 {
            if delta.abs() <= STEP_ZERO_OFFSET_TOLERANCE_M {
                0.0
            } else {
                f64::INFINITY
            }
        } else {
            delta.abs() / spread
        };
        score_sq += axis_score * axis_score;
    }
    Ok((offset, score_sq.sqrt()))
}

fn rotate_velocity_covariance(
    origin_rotation: &[[f64; 3]; 3],
    station_rotation: &[[f64; 3]; 3],
    covariance_station_enu: [[f64; 3]; 3],
) -> [[f64; 3]; 3] {
    let station_to_ecef = transpose3(station_rotation);
    let covariance_ecef = rotate_covariance(&station_to_ecef, covariance_station_enu);
    rotate_covariance(origin_rotation, covariance_ecef)
}

fn rotate_covariance(rotation: &[[f64; 3]; 3], covariance: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let rq = mat3_mul(rotation, &covariance);
    mat3_mul(&rq, &transpose3(rotation))
}

fn mat3_vec(matrix: &[[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]
}

fn enu_to_ecef(rotation: &[[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    mat3_vec(&transpose3(rotation), vector)
}

fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for row in 0..3 {
        for col in 0..3 {
            out[row][col] = a[row][0] * b[0][col] + a[row][1] * b[1][col] + a[row][2] * b[2][col];
        }
    }
    out
}

fn transpose3(matrix: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn validate_covariance(
    covariance: [[f64; 3]; 3],
    field: &'static str,
) -> Result<(), GeodeticTimeSeriesError> {
    crate::validate::validate_covariance_psd(&covariance, field)
        .map_err(|error| invalid_input(error.field(), error.reason()))?;
    validate_covariance_diagonal(covariance, field)
}

fn validate_covariance_diagonal(
    covariance: [[f64; 3]; 3],
    field: &'static str,
) -> Result<(), GeodeticTimeSeriesError> {
    for (axis, row) in covariance.iter().enumerate() {
        if row[axis] <= 0.0 {
            return Err(invalid_input(field, "diagonal must be positive"));
        }
    }
    Ok(())
}

fn validate_geodetic(
    geodetic: Wgs84Geodetic,
    field: &'static str,
) -> Result<(), GeodeticTimeSeriesError> {
    validate_finite(geodetic.lat_rad, field)?;
    validate_finite(geodetic.lon_rad, field)?;
    validate_finite(geodetic.height_m, field)?;
    if !(-core::f64::consts::FRAC_PI_2..=core::f64::consts::FRAC_PI_2).contains(&geodetic.lat_rad) {
        return Err(invalid_input(field, "latitude out of range"));
    }
    if !(-core::f64::consts::PI..=core::f64::consts::PI).contains(&geodetic.lon_rad) {
        return Err(invalid_input(field, "longitude out of range"));
    }
    Ok(())
}

fn validate_vec3(vector: [f64; 3], field: &'static str) -> Result<(), GeodeticTimeSeriesError> {
    for value in vector {
        validate_finite(value, field)?;
    }
    Ok(())
}

fn validate_finite(value: f64, field: &'static str) -> Result<(), GeodeticTimeSeriesError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn validate_positive(value: f64, field: &'static str) -> Result<(), GeodeticTimeSeriesError> {
    validate_finite(value, field)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be positive"))
    }
}

fn validate_nonnegative(value: f64, field: &'static str) -> Result<(), GeodeticTimeSeriesError> {
    validate_finite(value, field)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be non-negative"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> GeodeticTimeSeriesError {
    GeodeticTimeSeriesError::InvalidInput { field, reason }
}

impl fmt::Display for TimeSeriesQuality {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nominal => write!(f, "nominal"),
            Self::ShortSpan => write!(f, "short-span"),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Validation provenance: MIDAS equations and constants follow Blewitt,
    //! Kreemer, Hammond, and Gazeaux (2016), Journal of Geophysical Research:
    //! Solid Earth, doi:10.1002/2015JB012552. The trajectory model uses the
    //! constant velocity, annual/semiannual harmonic, and Heaviside jump terms
    //! described by Bevis and Brown (2014), Journal of Geodesy 88:283-311,
    //! doi:10.1007/s00190-013-0685-5. All oracle values below are analytic
    //! constants from those formulas or hand-constructed synthetic series.

    use super::*;
    use crate::estimation::MAD_GAUSSIAN_CONSISTENCY;

    fn enu_series(samples: &[(f64, [f64; 3])]) -> Vec<PositionSample> {
        samples
            .iter()
            .map(|&(epoch_year, position_m)| PositionSample {
                epoch_year,
                position_m,
                covariance_m2: None,
            })
            .collect()
    }

    fn series(samples: &[PositionSample]) -> PositionSeries<'_> {
        PositionSeries {
            frame: PositionFrame::Enu,
            samples,
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.1e}"
        );
    }

    #[test]
    fn midas_matches_published_five_year_two_step_breakdown_case() {
        let rate = [0.012, -0.006, 0.02];
        let mut raw = Vec::new();
        for day in 0..=(5 * 365) {
            let t = day as f64 / 365.0;
            let mut position = [rate[0] * t, rate[1] * t, rate[2] * t];
            if t >= 1.5 {
                position[0] += 100.0;
                position[2] -= 50.0;
            }
            if t >= 3.5 {
                position[0] += 80.0;
                position[2] -= 40.0;
            }
            raw.push((t, position));
        }
        let samples = enu_series(&raw);
        let velocity = velocity_midas(&series(&samples), MidasOptions::default()).unwrap();

        for (actual, expected) in velocity.rate_enu_m_per_yr.iter().zip(rate) {
            assert_close(*actual, expected, 2.0e-14);
        }
        assert_eq!(velocity.component_stats[0].pair_count, 1461);
        assert_eq!(velocity.component_stats[0].retained_pair_count, 731);
    }

    #[test]
    fn midas_and_lsq_recover_known_velocity_and_midas_uncertainty() {
        let rate = [0.01, -0.02, 0.03];
        let noise = [0.001, -0.002, 0.003, 0.0, 0.003, -0.002, 0.001];
        let raw = (0..=6)
            .map(|year| {
                let t = year as f64;
                (
                    t,
                    [
                        rate[0] * t + noise[year],
                        rate[1] * t + 2.0 * noise[year],
                        rate[2] * t - noise[year],
                    ],
                )
            })
            .collect::<Vec<_>>();
        let samples = enu_series(&raw);
        let velocity = velocity_midas(&series(&samples), MidasOptions::default()).unwrap();

        for (actual, expected) in velocity.rate_enu_m_per_yr.iter().zip(rate) {
            assert_close(*actual, expected, 1.0e-16);
        }
        let expected_sigma =
            3.0 * (core::f64::consts::PI / 2.0).sqrt() * MAD_GAUSSIAN_CONSISTENCY * 0.003
                / (6.0_f64 / 4.0).sqrt();
        assert_close(velocity.sigma_enu_m_per_yr[0], expected_sigma, 2.0e-17);

        let model = TrajectoryModel {
            reference_epoch_year: Some(3.0),
            include_annual: false,
            include_semiannual: false,
            offset_epochs_year: Vec::new(),
        };
        let trajectory =
            fit_trajectory(&series(&samples), &model, TrajectoryFitOptions::default()).unwrap();
        for (component, expected) in trajectory.components.iter().zip(rate) {
            assert_close(component.velocity_m_per_yr, expected, 2.0e-12);
        }
    }

    #[test]
    fn midas_resists_steps_seasons_and_outliers_that_bias_naive_lsq() {
        let true_rate = 0.011;
        let raw = (0..=24)
            .map(|quarter| {
                let t = quarter as f64 * 0.25;
                let seasonal = 0.012 * libm::sin(TAU * t) + 0.004 * libm::cos(TAU * t);
                let step = if t >= 2.25 { 0.09 } else { 0.0 };
                let outlier = if (t - 4.25).abs() < f64::EPSILON {
                    0.25
                } else {
                    0.0
                };
                (t, [true_rate * t + seasonal + step + outlier, 0.0, 0.0])
            })
            .collect::<Vec<_>>();
        let samples = enu_series(&raw);
        let midas = velocity_midas(&series(&samples), MidasOptions::default()).unwrap();
        assert_close(midas.rate_enu_m_per_yr[0], true_rate, 2.0e-15);

        let model = TrajectoryModel {
            reference_epoch_year: Some(3.0),
            include_annual: false,
            include_semiannual: false,
            offset_epochs_year: Vec::new(),
        };
        let naive = fit_trajectory(&series(&samples), &model, TrajectoryFitOptions::default())
            .unwrap()
            .components[0]
            .velocity_m_per_yr;
        assert!((naive - true_rate).abs() > 0.015);
    }

    #[test]
    fn trajectory_recovers_velocity_harmonics_and_offset() {
        let reference = 3.0;
        let offset_epoch = 2.3;
        let east = TrajectoryComponent {
            position_m: 0.25,
            velocity_m_per_yr: 0.017,
            annual_sin_m: Some(0.012),
            annual_cos_m: Some(-0.004),
            semiannual_sin_m: Some(0.006),
            semiannual_cos_m: Some(0.002),
            offsets_m: vec![0.08],
        };
        let raw = (0..=96)
            .map(|month| {
                let t = month as f64 / 12.0;
                let dt = t - reference;
                let value = east.position_m
                    + east.velocity_m_per_yr * dt
                    + east.annual_sin_m.unwrap() * libm::sin(TAU * dt)
                    + east.annual_cos_m.unwrap() * libm::cos(TAU * dt)
                    + east.semiannual_sin_m.unwrap() * libm::sin(2.0 * TAU * dt)
                    + east.semiannual_cos_m.unwrap() * libm::cos(2.0 * TAU * dt)
                    + if t > offset_epoch {
                        east.offsets_m[0]
                    } else {
                        0.0
                    };
                (t, [value, -0.5 * value, 0.25 * value])
            })
            .collect::<Vec<_>>();
        let samples = enu_series(&raw);
        let model = TrajectoryModel {
            reference_epoch_year: Some(reference),
            include_annual: true,
            include_semiannual: true,
            offset_epochs_year: vec![offset_epoch],
        };
        let trajectory =
            fit_trajectory(&series(&samples), &model, TrajectoryFitOptions::default()).unwrap();
        let actual = &trajectory.components[0];

        assert_close(actual.position_m, east.position_m, 2.0e-10);
        assert_close(actual.velocity_m_per_yr, east.velocity_m_per_yr, 2.0e-10);
        assert_close(
            actual.annual_sin_m.unwrap(),
            east.annual_sin_m.unwrap(),
            2.0e-10,
        );
        assert_close(
            actual.annual_cos_m.unwrap(),
            east.annual_cos_m.unwrap(),
            2.0e-10,
        );
        assert_close(
            actual.semiannual_sin_m.unwrap(),
            east.semiannual_sin_m.unwrap(),
            2.0e-10,
        );
        assert_close(
            actual.semiannual_cos_m.unwrap(),
            east.semiannual_cos_m.unwrap(),
            2.0e-10,
        );
        assert_close(actual.offsets_m[0], east.offsets_m[0], 2.0e-10);
    }

    #[test]
    fn detect_steps_flags_injected_offset_and_not_step_free_series() {
        let stepped = (0..=96)
            .map(|month| {
                let t = month as f64 / 12.0;
                let step = if t >= 3.0 { 0.12 } else { 0.0 };
                (t, [0.01 * t + step, -0.02 * t, 0.0])
            })
            .collect::<Vec<_>>();
        let stepped_samples = enu_series(&stepped);
        let candidates =
            detect_steps(&series(&stepped_samples), StepDetectionOptions::default()).unwrap();
        assert!(!candidates.is_empty());
        assert_close(candidates[0].epoch_year, 3.0, 0.25);
        assert!(candidates[0].offset_enu_m[0] > 0.10);

        let clean = (0..=96)
            .map(|month| {
                let t = month as f64 / 12.0;
                (t, [0.01 * t, -0.02 * t, 0.0])
            })
            .collect::<Vec<_>>();
        let clean_samples = enu_series(&clean);
        let clean_candidates =
            detect_steps(&series(&clean_samples), StepDetectionOptions::default()).unwrap();
        assert!(clean_candidates.is_empty());
    }

    #[test]
    fn short_sparse_series_returns_typed_error() {
        let samples = enu_series(&[(0.0, [0.0; 3]), (1.0, [1.0, 0.0, 0.0])]);
        let error = velocity_midas(&series(&samples), MidasOptions::default()).unwrap_err();
        assert!(matches!(
            error,
            GeodeticTimeSeriesError::InsufficientPairs {
                pairs: 1,
                needed: 3
            }
        ));
    }

    #[test]
    fn network_field_removes_common_mode_in_requested_frame() {
        let reference = Wgs84Geodetic::new(0.7, -1.2, 10.0).unwrap();
        let first_samples = enu_series(&[
            (0.0, [0.0; 3]),
            (1.0, [1.0, 2.0, 0.0]),
            (2.0, [2.0, 4.0, 0.0]),
            (3.0, [3.0, 6.0, 0.0]),
        ]);
        let second_samples = enu_series(&[
            (0.0, [0.0; 3]),
            (1.0, [3.0, 4.0, 0.0]),
            (2.0, [6.0, 8.0, 0.0]),
            (3.0, [9.0, 12.0, 0.0]),
        ]);
        let stations = [
            NetworkStation {
                id: "A",
                reference,
                series: series(&first_samples),
            },
            NetworkStation {
                id: "B",
                reference,
                series: series(&second_samples),
            },
        ];
        let field = network_field(
            &stations,
            NetworkFrame {
                origin: reference,
                remove_common_mode: true,
            },
        )
        .unwrap();

        assert_close(field.common_mode_enu_m_per_yr[0], 2.0, 1.0e-12);
        assert_close(field.common_mode_enu_m_per_yr[1], 3.0, 1.0e-12);
        assert_close(field.stations[0].rate_enu_m_per_yr[0], -1.0, 1.0e-12);
        assert_close(field.stations[0].rate_enu_m_per_yr[1], -1.0, 1.0e-12);
        assert_close(field.stations[1].rate_enu_m_per_yr[0], 1.0, 1.0e-12);
        assert_close(field.stations[1].rate_enu_m_per_yr[1], 1.0, 1.0e-12);
    }
}
