//! Source localization from arrival times.
//!
//! This sans-I/O module solves for an event position from sensors at known
//! Cartesian coordinates. Coordinates are metres in a caller-chosen 2D or 3D
//! frame, times are seconds, and propagation speeds are metres per second.
//!
//! # Measurement models
//!
//! Absolute time of arrival (ToA) uses
//!
//! ```text
//! t_i = t0 + ||x - s_i|| / c_i,
//! ```
//!
//! with state `[x, t0]`: two or three position coordinates followed by the
//! origin time. Time difference of arrival (TDOA) uses
//!
//! ```text
//! t_i - t_ref = ||x - s_i|| / c_i - ||x - s_ref|| / c_ref,
//! ```
//!
//! with position-only state `[x]`. After a TDOA position solve, the origin time
//! is recovered from the absolute arrivals. Linear loss uses their arithmetic
//! mean exactly; a non-linear loss applies one iteratively reweighted refinement
//! so an arrival downweighted by the position solve is also downweighted in the
//! reported time.
//!
//! Both modes require at least `dimension + 1` sensors: three in 2D or four in
//! 3D. The closed-form seed is the spherical-intersection linearization of
//! H. C. Schau and A. Z. Robinson, *IEEE Transactions on Acoustics, Speech, and
//! Signal Processing* 35(8), 1987, followed by a quadratic in the reference
//! range (TDOA) or emission-distance unknown `c * t0` (ToA). See
//! [`closed_form_initial_guess`].
//!
//! The solution covariance is `(J^T J)^-1 * timing_sigma_s^2`, formed from the
//! retained singular vectors and values of the final Jacobian. At the fitted
//! state this is the local estimate covariance. [`source_crlb`] evaluates the
//! same timing-information interpretation at a proposed source point through
//! the shared DOP machinery, where it is a Cramer-Rao lower bound (CRLB).
//!
//! By default [`locate_source`] also measures each sensor's influence by running
//! one complete nonlinear leave-one-out solve per sensor. Each record reports
//! the full-solution ToA residual, held-out ToA residual, state displacement,
//! robust-loss weight, and normalized residual magnitude. Call
//! [`locate_source_with`] with a [`SourceLocateConfig`] whose
//! `include_influence` is `false` to skip those re-solves.
//!
//! # Example
//!
//! ```
//! use sidereon_core::source_localization::{
//!     locate_source_with, Sensor, SourceLocateConfig, SourceLocateOptions, SourceSolveMode,
//! };
//!
//! let sensors = vec![
//!     Sensor::new(vec![0.0, 0.0]),
//!     Sensor::new(vec![100.0, 0.0]),
//!     Sensor::new(vec![0.0, 100.0]),
//!     Sensor::new(vec![100.0, 100.0]),
//! ];
//! let source_m = [30.0, 40.0];
//! let origin_time_s = 2.0;
//! let propagation_speed_m_s = 50.0;
//! let arrival_times_s = sensors
//!     .iter()
//!     .map(|sensor| {
//!         let dx = source_m[0] - sensor.position_m[0];
//!         let dy = source_m[1] - sensor.position_m[1];
//!         origin_time_s + (dx * dx + dy * dy).sqrt() / propagation_speed_m_s
//!     })
//!     .collect::<Vec<_>>();
//!
//! let options = SourceLocateOptions {
//!     mode: SourceSolveMode::Toa,
//!     ..SourceLocateOptions::default()
//! };
//! let mut config = SourceLocateConfig::from(options);
//! config.include_influence = false;
//! let solution = locate_source_with(
//!     &sensors,
//!     &arrival_times_s,
//!     propagation_speed_m_s,
//!     &config,
//! )?;
//!
//! assert!((solution.position_m[0] - source_m[0]).abs() < 1.0e-8);
//! assert!((solution.position_m[1] - source_m[1]).abs() < 1.0e-8);
//! assert!((solution.origin_time_s.unwrap() - origin_time_s).abs() < 1.0e-10);
//! # Ok::<(), sidereon_core::source_localization::SourceLocalizationError>(())
//! ```

use core::fmt;

pub use trust_region_least_squares::loss::Loss;
use trust_region_least_squares::model::{solve_model, ResidualModel};
use trust_region_least_squares::trf::{TrfError, TrfOptions, TrfResult, XScale};

use crate::astro::math::least_squares::singular_value_diagnostics;
use crate::dop::{self, Dop, DopError};
use crate::geometry_quality::{
    classify, GeometryQuality, GeometryQualityThresholds, ObservabilityTier,
};
use nalgebra::DMatrix;

/// Relative tolerance for classifying a quadratic coefficient or discriminant
/// as numerically zero. The closed-form coefficients contain several dot
/// products, so 32 machine epsilons covers their rounding accumulation while
/// remaining far below a meaningful root separation.
const QUADRATIC_REL_EPS: f64 = 32.0 * f64::EPSILON;

/// A sensor with a known Cartesian position.
///
/// `propagation_speed_m_s` overrides the call-level propagation speed for this
/// sensor when it is present. That is a simple per-path timing approximation;
/// no refraction or ray tracing is modeled.
#[derive(Debug, Clone, PartialEq)]
pub struct Sensor {
    /// Sensor position in metres. The vector length must be 2 or 3.
    pub position_m: Vec<f64>,
    /// Optional per-sensor propagation speed in metres per second.
    pub propagation_speed_m_s: Option<f64>,
}

impl Sensor {
    /// Construct a sensor that uses the call-level propagation speed.
    pub fn new(position_m: impl Into<Vec<f64>>) -> Self {
        Self {
            position_m: position_m.into(),
            propagation_speed_m_s: None,
        }
    }

    /// Construct a sensor with its own propagation speed.
    pub fn with_speed(position_m: impl Into<Vec<f64>>, propagation_speed_m_s: f64) -> Self {
        Self {
            position_m: position_m.into(),
            propagation_speed_m_s: Some(propagation_speed_m_s),
        }
    }
}

/// Measurement model used by [`locate_source`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SourceSolveMode {
    /// Absolute time of arrival. The state is `[position..., origin_time]`.
    #[default]
    Toa,
    /// Time difference of arrival against a reference sensor.
    ///
    /// The residual subtracts the reference sensor equation and does not solve
    /// an origin-time state. The returned origin time is estimated after the
    /// position solve from the absolute arrivals.
    Tdoa {
        /// Reference sensor index.
        reference_sensor: usize,
    },
}

/// Options for [`locate_source`].
///
/// This type keeps its 1.0 shape so struct literals stay valid. Settings added
/// after 1.0 live on [`SourceLocateConfig`] and are passed through
/// [`locate_source_with`].
#[derive(Debug, Clone, PartialEq)]
pub struct SourceLocateOptions {
    /// ToA or TDOA residual form.
    pub mode: SourceSolveMode,
    /// Timing standard deviation used for covariance, CRLB, and normalized
    /// influence scores.
    pub timing_sigma_s: f64,
    /// Loss function passed to the trust-region least-squares solver.
    pub loss: Loss,
    /// Residual scale in seconds for non-linear loss functions.
    pub f_scale_s: f64,
    /// Optional solver function tolerance.
    pub ftol: Option<f64>,
    /// Optional solver step tolerance.
    pub xtol: Option<f64>,
    /// Optional solver gradient tolerance.
    pub gtol: Option<f64>,
    /// Optional maximum residual evaluations.
    pub max_nfev: Option<usize>,
}

impl Default for SourceLocateOptions {
    fn default() -> Self {
        Self {
            mode: SourceSolveMode::Toa,
            timing_sigma_s: 1.0,
            loss: Loss::Linear,
            f_scale_s: 1.0,
            ftol: None,
            xtol: None,
            gtol: None,
            max_nfev: None,
        }
    }
}

/// Full configuration for [`locate_source_with`].
///
/// This type is `#[non_exhaustive]` so later settings can be added without
/// breaking callers. Construct it from a [`SourceLocateOptions`] with
/// [`SourceLocateConfig::from`] or start from [`SourceLocateConfig::default`],
/// then set fields:
///
/// ```
/// use sidereon_core::source_localization::{SourceLocateConfig, SourceLocateOptions};
///
/// let mut config = SourceLocateConfig::from(SourceLocateOptions::default());
/// config.include_influence = false;
/// ```
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SourceLocateConfig {
    /// Solver and measurement-model options, unchanged from [`locate_source`].
    pub options: SourceLocateOptions,
    /// Whether to compute per-sensor leave-one-out influence diagnostics.
    ///
    /// Influence runs one full nonlinear re-solve per sensor. The default is
    /// `true`; set this to `false` to skip all leave-one-out solves and return
    /// an empty [`SourceSolution::per_sensor_influence`] vector. Every other
    /// output is bit-identical either way.
    pub include_influence: bool,
}

impl Default for SourceLocateConfig {
    fn default() -> Self {
        Self {
            options: SourceLocateOptions::default(),
            include_influence: true,
        }
    }
}

impl From<SourceLocateOptions> for SourceLocateConfig {
    fn from(options: SourceLocateOptions) -> Self {
        Self {
            options,
            include_influence: true,
        }
    }
}

/// Closed-form seed used to start the iterative solve.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceInitialGuess {
    /// Initial position in metres.
    pub position_m: Vec<f64>,
    /// Initial origin time in seconds when it can be inferred.
    pub origin_time_s: Option<f64>,
    /// Root-mean-square residual of the seed in seconds.
    pub residual_rms_s: f64,
}

/// One residual associated with a sensor row.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceResidual {
    /// Sensor index in the caller's input slice.
    pub sensor_index: usize,
    /// Reference sensor for a TDOA residual, or `None` for ToA.
    pub reference_sensor_index: Option<usize>,
    /// Residual in seconds.
    pub residual_s: f64,
}

/// Per-sensor leave-one-out diagnostic.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSensorInfluence {
    /// Sensor index in the caller's input slice.
    pub sensor_index: usize,
    /// ToA residual at the full solution and estimated origin time, in seconds.
    ///
    /// This is a ToA residual in both solve modes, including TDOA mode.
    pub residual_s: f64,
    /// Held-out ToA residual after solving without this sensor, in seconds.
    ///
    /// This is evaluated at the leave-one-out estimated origin time in both
    /// solve modes, including TDOA mode.
    pub leave_one_out_residual_s: Option<f64>,
    /// Position change between the full and leave-one-out solutions, in metres.
    pub position_delta_m: Option<f64>,
    /// Origin-time change between the full and leave-one-out solutions, in seconds.
    pub origin_time_delta_s: Option<f64>,
    /// First-derivative loss weight for the full-solution residual.
    ///
    /// This field carries robust-loss downweighting separately from [`score`](Self::score).
    pub loss_weight: f64,
    /// Normalized residual magnitude in timing-sigma units.
    ///
    /// This is `max(|residual_s|, |leave_one_out_residual_s|) /
    /// timing_sigma_s`, or `|residual_s| / timing_sigma_s` when the
    /// leave-one-out solve is unavailable. Robust-loss downweighting is
    /// reported separately by [`loss_weight`](Self::loss_weight).
    pub score: f64,
}

/// Timing-information covariance for a source state.
///
/// This is `(J^T J)^-1 * timing_sigma_s^2` at the evaluation point. At a fitted
/// solution it is the local estimate covariance; at a proposed point it is the
/// Cramer-Rao lower bound (CRLB).
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCovariance {
    /// Full state covariance in solver state order.
    pub state: Vec<Vec<f64>>,
    /// Position covariance block in square metres.
    pub position_m2: Vec<Vec<f64>>,
    /// Origin-time variance in square seconds when origin time is in the state.
    pub origin_time_s2: Option<f64>,
    /// Timing sigma used to scale the cofactor.
    pub timing_sigma_s: f64,
}

/// Source solution from [`locate_source`].
#[derive(Debug, Clone, PartialEq)]
pub struct SourceSolution {
    /// Estimated source position in metres.
    pub position_m: Vec<f64>,
    /// Estimated origin time in seconds.
    pub origin_time_s: Option<f64>,
    /// State covariance scaled by [`SourceLocateOptions::timing_sigma_s`].
    pub covariance: Option<SourceCovariance>,
    /// Solver residuals in seconds.
    pub residuals: Vec<SourceResidual>,
    /// Per-sensor influence diagnostics.
    pub per_sensor_influence: Vec<SourceSensorInfluence>,
    /// Geometry observability and covariance-validation diagnostics for the
    /// final timing design. Snapshot source solves use no propagated prior, so
    /// `ZeroRedundancy` covariance bounds are unvalidated, `Weak` bounds are
    /// reported without clamping, and `RankDeficient` is routed through a typed
    /// geometry error instead of returning a solution.
    pub geometry_quality: GeometryQuality,
    /// Closed-form seed used to start the iterative solve.
    pub initial_guess: SourceInitialGuess,
    /// Trust-region termination code: `0` maximum evaluations, `1` gradient
    /// tolerance, `2` function tolerance, `3` step tolerance, or `4` both
    /// function and step tolerances.
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

impl SourceSolution {
    /// Return the covariance as the CRLB for the timing sigma used by the solve.
    pub fn crlb(&self) -> Option<&SourceCovariance> {
        self.covariance.as_ref()
    }
}

/// CRLB and DOP for a proposed sensor/source geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceCrlb {
    /// DOP scalars formed from the timing design matrix.
    pub dop: Dop,
    /// State covariance scaled by the requested timing sigma.
    pub covariance: SourceCovariance,
}

/// Source-localization failure.
#[derive(Debug, Clone, PartialEq)]
pub enum SourceLocalizationError {
    /// A boundary input is malformed.
    InvalidInput {
        /// Name of the malformed field.
        field: &'static str,
        /// Stable validation reason.
        reason: &'static str,
    },
    /// There are fewer sensors than the selected solve needs.
    TooFewSensors {
        /// Number of sensors supplied.
        sensors: usize,
        /// Minimum number of sensors required.
        needed: usize,
    },
    /// The closed-form initializer could not solve the geometry.
    InitializerSingular,
    /// Geometry DOP or CRLB failed.
    Geometry(DopError),
    /// The trust-region solver failed.
    Solver(TrfError),
    /// The trust-region solver exhausted its evaluation budget.
    DidNotConverge {
        /// Solver status code.
        status: i32,
    },
}

impl fmt::Display for SourceLocalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid source localization input {field}: {reason}")
            }
            Self::TooFewSensors { sensors, needed } => {
                write!(
                    f,
                    "source localization has {sensors} sensors; need at least {needed}"
                )
            }
            Self::InitializerSingular => write!(f, "closed-form source initializer is singular"),
            Self::Geometry(err) => write!(f, "source geometry failed: {err}"),
            Self::Solver(err) => write!(f, "source solver failed: {err}"),
            Self::DidNotConverge { status } => {
                write!(f, "source solver did not converge, status {status}")
            }
        }
    }
}

impl std::error::Error for SourceLocalizationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Geometry(err) => Some(err),
            Self::Solver(err) => Some(err),
            _ => None,
        }
    }
}

impl From<DopError> for SourceLocalizationError {
    fn from(value: DopError) -> Self {
        Self::Geometry(value)
    }
}

impl From<TrfError> for SourceLocalizationError {
    fn from(value: TrfError) -> Self {
        Self::Solver(value)
    }
}

/// Locate a source from sensor arrival times.
///
/// `sensors` and `arrival_times_s` must have matching length. Positions must
/// all be 2D or all be 3D. The call-level propagation speed is used for every
/// sensor without a per-sensor override.
///
/// # Errors
///
/// Returns [`SourceLocalizationError::InvalidInput`] for malformed, non-finite,
/// or inconsistent inputs; [`SourceLocalizationError::TooFewSensors`] when the
/// selected dimension has fewer than `dimension + 1` sensors;
/// [`SourceLocalizationError::InitializerSingular`] when the closed-form seed
/// geometry is degenerate; [`SourceLocalizationError::Solver`] when the
/// trust-region solver rejects the problem; [`SourceLocalizationError::DidNotConverge`]
/// when its evaluation budget is exhausted; or
/// [`SourceLocalizationError::Geometry`] when the final Jacobian is singular.
pub fn locate_source(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    options: &SourceLocateOptions,
) -> Result<SourceSolution, SourceLocalizationError> {
    locate_source_inner(
        sensors,
        arrival_times_s,
        propagation_speed_m_s,
        options,
        true,
    )
}

/// Locate a source from sensor arrival times with a full [`SourceLocateConfig`].
///
/// This is [`locate_source`] plus the settings that live on the config, such
/// as [`SourceLocateConfig::include_influence`]. With a default config the two
/// entry points are equivalent.
///
/// # Errors
///
/// The same [`SourceLocalizationError`] variants as [`locate_source`].
pub fn locate_source_with(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    config: &SourceLocateConfig,
) -> Result<SourceSolution, SourceLocalizationError> {
    locate_source_inner(
        sensors,
        arrival_times_s,
        propagation_speed_m_s,
        &config.options,
        config.include_influence,
    )
}

/// Compute the closed-form spherical-intersection seed used by [`locate_source`].
///
/// The seed uses the call-level propagation speed in the closed-form equations.
/// Per-sensor speed overrides are applied by the iterative residual model. The
/// TDOA branch is the spherical-intersection method of H. C. Schau and A. Z.
/// Robinson, *IEEE Transactions on Acoustics, Speech, and Signal Processing*
/// 35(8), 1987: position is affine in the unknown reference range, followed by
/// a quadratic in that range. The ToA branch uses the same linearization in the
/// emission-distance unknown `propagation_speed_m_s * origin_time_s` and always
/// uses sensor `0` as its algebraic reference; any sensor is mathematically
/// valid for that linearization.
///
/// # Errors
///
/// Returns [`SourceLocalizationError::InvalidInput`] for malformed, non-finite,
/// or inconsistent inputs; [`SourceLocalizationError::TooFewSensors`] when the
/// selected dimension has fewer than `dimension + 1` sensors; or
/// [`SourceLocalizationError::InitializerSingular`] when the linear system or
/// quadratic is degenerate or has no admissible root.
pub fn closed_form_initial_guess(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    mode: SourceSolveMode,
) -> Result<SourceInitialGuess, SourceLocalizationError> {
    let options = SourceLocateOptions {
        mode,
        ..SourceLocateOptions::default()
    };
    let resolved =
        resolve_locate_inputs(sensors, arrival_times_s, propagation_speed_m_s, &options)?;
    closed_form_initial_guess_resolved(sensors, arrival_times_s, propagation_speed_m_s, &resolved)
}

/// Deprecated name for [`closed_form_initial_guess`].
///
/// The implemented initializer is the Schau-Robinson spherical-intersection
/// linearization, not the two-stage Chan-Ho weighted least-squares method.
///
/// # Errors
///
/// Returns the same [`SourceLocalizationError`] variants as
/// [`closed_form_initial_guess`].
#[deprecated(
    since = "1.1.0",
    note = "use closed_form_initial_guess; this is Schau-Robinson spherical intersection, not Chan-Ho"
)]
pub fn chan_ho_initial_guess(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    mode: SourceSolveMode,
) -> Result<SourceInitialGuess, SourceLocalizationError> {
    closed_form_initial_guess(sensors, arrival_times_s, propagation_speed_m_s, mode)
}

/// Compute timing DOP for a proposed source location.
///
/// The returned position DOP values multiply timing sigma in seconds to produce
/// metres. The local Cartesian axes are used for the horizontal and vertical
/// split.
///
/// # Errors
///
/// Returns [`SourceLocalizationError::InvalidInput`] for malformed, non-finite,
/// inconsistent inputs, including a source coincident with a sensor. Returns
/// [`SourceLocalizationError::Geometry`] when the shared DOP machinery finds
/// too few sensors or a singular timing design.
pub fn source_dop(
    sensors: &[Sensor],
    source_position_m: &[f64],
    propagation_speed_m_s: f64,
) -> Result<Dop, SourceLocalizationError> {
    let resolved = resolve_geometry_inputs(sensors, source_position_m, propagation_speed_m_s)?;
    let rows = source_toa_design_rows(sensors, source_position_m, &resolved)?;
    let weights = vec![1.0; sensors.len()];
    dop::dop_from_design_rows(&rows, &weights, resolved.dimension, identity_rotation())
        .map_err(SourceLocalizationError::Geometry)
}

/// Compute a timing CRLB for a proposed source location.
///
/// The covariance is `(H^T H)^-1 * timing_sigma_s^2`, where each row is the ToA
/// timing derivative at `source_position_m`.
///
/// # Errors
///
/// Returns [`SourceLocalizationError::InvalidInput`] for malformed, non-finite,
/// non-positive, or inconsistent inputs. Returns
/// [`SourceLocalizationError::Geometry`] when the shared DOP machinery finds
/// too few sensors or a singular timing design. A source coincident with a
/// sensor is [`SourceLocalizationError::InvalidInput`].
pub fn source_crlb(
    sensors: &[Sensor],
    source_position_m: &[f64],
    propagation_speed_m_s: f64,
    timing_sigma_s: f64,
) -> Result<SourceCrlb, SourceLocalizationError> {
    validate_positive("timing_sigma_s", timing_sigma_s)?;
    let resolved = resolve_geometry_inputs(sensors, source_position_m, propagation_speed_m_s)?;
    let rows = source_toa_design_rows(sensors, source_position_m, &resolved)?;
    let weights = vec![1.0; sensors.len()];
    let rotation = identity_rotation();
    let cofactor =
        dop::geometry_cofactor_from_design_rows(&rows, &weights, resolved.dimension, rotation)?;
    let dop = dop::dop_from_design_rows(&rows, &weights, resolved.dimension, rotation)?;
    let covariance =
        covariance_from_state_cofactor(&cofactor.state, resolved.dimension, timing_sigma_s, true);
    Ok(SourceCrlb { dop, covariance })
}

#[derive(Debug, Clone)]
struct ResolvedInputs {
    dimension: usize,
    speeds_m_s: Vec<f64>,
    mode: SourceSolveMode,
}

#[derive(Debug, Clone)]
struct ResolvedGeometry {
    dimension: usize,
    speeds_m_s: Vec<f64>,
}

#[derive(Debug)]
struct SourceProblem<'a> {
    sensors: &'a [Sensor],
    arrival_times_s: &'a [f64],
    speeds_m_s: &'a [f64],
    dimension: usize,
    mode: SourceSolveMode,
}

impl SourceProblem<'_> {
    fn residual_records(&self, residuals: &[f64]) -> Vec<SourceResidual> {
        match self.mode {
            SourceSolveMode::Toa => residuals
                .iter()
                .enumerate()
                .map(|(sensor_index, &residual_s)| SourceResidual {
                    sensor_index,
                    reference_sensor_index: None,
                    residual_s,
                })
                .collect(),
            SourceSolveMode::Tdoa { reference_sensor } => {
                let mut out = Vec::with_capacity(residuals.len());
                let mut row = 0;
                for sensor_index in 0..self.sensors.len() {
                    if sensor_index == reference_sensor {
                        continue;
                    }
                    out.push(SourceResidual {
                        sensor_index,
                        reference_sensor_index: Some(reference_sensor),
                        residual_s: residuals[row],
                    });
                    row += 1;
                }
                out
            }
        }
    }
}

impl ResidualModel for SourceProblem<'_> {
    fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        match self.mode {
            SourceSolveMode::Toa => {
                let origin_time_s = x[self.dimension];
                for (i, sensor) in self.sensors.iter().enumerate() {
                    let range_m = distance(&x[..self.dimension], &sensor.position_m);
                    out.push(
                        origin_time_s + range_m / self.speeds_m_s[i] - self.arrival_times_s[i],
                    );
                }
            }
            SourceSolveMode::Tdoa { reference_sensor } => {
                let ref_range_m = distance(
                    &x[..self.dimension],
                    &self.sensors[reference_sensor].position_m,
                );
                let ref_time_s = ref_range_m / self.speeds_m_s[reference_sensor];
                for (i, sensor) in self.sensors.iter().enumerate() {
                    if i == reference_sensor {
                        continue;
                    }
                    let range_m = distance(&x[..self.dimension], &sensor.position_m);
                    let predicted_s = range_m / self.speeds_m_s[i] - ref_time_s;
                    let observed_s =
                        self.arrival_times_s[i] - self.arrival_times_s[reference_sensor];
                    out.push(predicted_s - observed_s);
                }
            }
        }
    }

    fn jacobian(&self, x: &[f64], _f0: &[f64], out: &mut Vec<f64>) {
        out.clear();
        match self.mode {
            SourceSolveMode::Toa => {
                let n = self.dimension + 1;
                out.resize(self.sensors.len() * n, 0.0);
                for (row, sensor) in self.sensors.iter().enumerate() {
                    fill_range_derivative(
                        &x[..self.dimension],
                        &sensor.position_m,
                        self.speeds_m_s[row],
                        &mut out[row * n..row * n + self.dimension],
                    );
                    out[row * n + self.dimension] = 1.0;
                }
            }
            SourceSolveMode::Tdoa { reference_sensor } => {
                let n = self.dimension;
                out.resize((self.sensors.len() - 1) * n, 0.0);
                let mut ref_derivative = vec![0.0; self.dimension];
                fill_range_derivative(
                    &x[..self.dimension],
                    &self.sensors[reference_sensor].position_m,
                    self.speeds_m_s[reference_sensor],
                    &mut ref_derivative,
                );
                let mut row = 0;
                for (i, sensor) in self.sensors.iter().enumerate() {
                    if i == reference_sensor {
                        continue;
                    }
                    let start = row * n;
                    fill_range_derivative(
                        &x[..self.dimension],
                        &sensor.position_m,
                        self.speeds_m_s[i],
                        &mut out[start..start + n],
                    );
                    for axis in 0..n {
                        out[start + axis] -= ref_derivative[axis];
                    }
                    row += 1;
                }
            }
        }
    }
}

fn locate_source_inner(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    options: &SourceLocateOptions,
    include_influence: bool,
) -> Result<SourceSolution, SourceLocalizationError> {
    let resolved = resolve_locate_inputs(sensors, arrival_times_s, propagation_speed_m_s, options)?;
    let initial_guess = closed_form_initial_guess_resolved(
        sensors,
        arrival_times_s,
        propagation_speed_m_s,
        &resolved,
    )?;
    let mut x0 = initial_guess.position_m.clone();
    if matches!(resolved.mode, SourceSolveMode::Toa) {
        let origin_time_s = initial_guess
            .origin_time_s
            .ok_or(SourceLocalizationError::InitializerSingular)?;
        x0.push(origin_time_s);
    }

    let problem = SourceProblem {
        sensors,
        arrival_times_s,
        speeds_m_s: &resolved.speeds_m_s,
        dimension: resolved.dimension,
        mode: resolved.mode,
    };
    let result = solve_model(&problem, &x0, &solver_options(options))?;
    if !result.success() {
        return Err(SourceLocalizationError::DidNotConverge {
            status: result.status,
        });
    }

    let mut solution = build_solution(
        &problem,
        &resolved,
        &initial_guess,
        result,
        options.timing_sigma_s,
        options.loss,
        options.f_scale_s,
    )?;
    if include_influence {
        solution.per_sensor_influence = compute_influence(
            &solution,
            sensors,
            arrival_times_s,
            propagation_speed_m_s,
            options,
        );
    }
    Ok(solution)
}

fn build_solution(
    problem: &SourceProblem<'_>,
    resolved: &ResolvedInputs,
    initial_guess: &SourceInitialGuess,
    result: TrfResult,
    timing_sigma_s: f64,
    loss: Loss,
    f_scale_s: f64,
) -> Result<SourceSolution, SourceLocalizationError> {
    let position_m = result.x[..resolved.dimension].to_vec();
    let origin_time_s = match resolved.mode {
        SourceSolveMode::Toa => Some(result.x[resolved.dimension]),
        SourceSolveMode::Tdoa { .. } => Some(estimate_origin_time_for_loss_s(
            problem.sensors,
            problem.arrival_times_s,
            problem.speeds_m_s,
            &position_m,
            loss,
            f_scale_s,
        )),
    };
    let residuals = problem.residual_records(&result.fun);
    let parameter_count = result.x.len();
    let residual_count = result.fun.len();
    let jacobian = jacobian_svd_diagnostics(&result.jac, residual_count, parameter_count)
        .ok_or(SourceLocalizationError::Geometry(DopError::Singular))?;
    let geometry_quality =
        source_geometry_quality_from_svd(&jacobian, residual_count, parameter_count);
    if geometry_quality.tier == ObservabilityTier::RankDeficient {
        return Err(SourceLocalizationError::Geometry(DopError::Singular));
    }
    let covariance = covariance_from_state_cofactor(
        &jacobian.cofactor,
        resolved.dimension,
        timing_sigma_s,
        parameter_count == resolved.dimension + 1,
    );
    Ok(SourceSolution {
        position_m,
        origin_time_s,
        covariance: Some(covariance),
        residuals,
        per_sensor_influence: Vec::new(),
        geometry_quality,
        initial_guess: initial_guess.clone(),
        status: result.status,
        nfev: result.nfev,
        njev: result.njev,
        cost: result.cost,
        optimality: result.optimality,
    })
}

#[cfg(test)]
fn source_geometry_quality_from_jacobian(
    jac: &[f64],
    m: usize,
    n: usize,
) -> Result<GeometryQuality, SourceLocalizationError> {
    let diagnostics = jacobian_svd_diagnostics(jac, m, n)
        .ok_or(SourceLocalizationError::Geometry(DopError::Singular))?;
    Ok(source_geometry_quality_from_svd(&diagnostics, m, n))
}

fn source_geometry_quality_from_svd(
    diagnostics: &JacobianSvdDiagnostics,
    m: usize,
    n: usize,
) -> GeometryQuality {
    let gdop = if diagnostics.rank < n {
        f64::INFINITY
    } else {
        let trace = cofactor_trace(&diagnostics.cofactor);
        if trace >= 0.0 && trace.is_finite() {
            trace.sqrt()
        } else {
            f64::INFINITY
        }
    };
    classify(
        diagnostics.rank,
        n,
        m as i32 - n as i32,
        diagnostics.condition_number,
        gdop,
        false,
        GeometryQualityThresholds::default(),
    )
}

fn closed_form_initial_guess_resolved(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    resolved: &ResolvedInputs,
) -> Result<SourceInitialGuess, SourceLocalizationError> {
    match resolved.mode {
        SourceSolveMode::Toa => {
            closed_form_toa_initial_guess(sensors, arrival_times_s, propagation_speed_m_s, resolved)
        }
        SourceSolveMode::Tdoa { reference_sensor } => closed_form_tdoa_initial_guess(
            sensors,
            arrival_times_s,
            propagation_speed_m_s,
            resolved,
            reference_sensor,
        ),
    }
}

fn closed_form_toa_initial_guess(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    resolved: &ResolvedInputs,
) -> Result<SourceInitialGuess, SourceLocalizationError> {
    let d = resolved.dimension;
    let ref_pos = &sensors[0].position_m;
    let z0 = propagation_speed_m_s * arrival_times_s[0];
    let ref_norm2 = dot(ref_pos, ref_pos);
    let mut a = Vec::with_capacity(sensors.len() - 1);
    let mut b = Vec::with_capacity(sensors.len() - 1);
    let mut h = Vec::with_capacity(sensors.len() - 1);
    for i in 1..sensors.len() {
        let row: Vec<f64> = sensors[i]
            .position_m
            .iter()
            .zip(ref_pos)
            .map(|(s, r)| s - r)
            .collect();
        let zi = propagation_speed_m_s * arrival_times_s[i];
        let delta_z = zi - z0;
        let delta_norm = dot(&sensors[i].position_m, &sensors[i].position_m) - ref_norm2;
        a.push(row);
        b.push(0.5 * (delta_norm - (zi * zi - z0 * z0)));
        h.push(delta_z);
    }
    let p0 = least_squares(&a, &b)?;
    let p1 = least_squares(&a, &h)?;
    let q: Vec<f64> = p0.iter().zip(ref_pos).map(|(p, r)| p - r).collect();
    let roots = quadratic_roots(
        dot(&p1, &p1) - 1.0,
        2.0 * dot(&q, &p1) + 2.0 * z0,
        dot(&q, &q) - z0 * z0,
    )?;

    let mut best: Option<SourceInitialGuess> = None;
    let mut best_sse = f64::INFINITY;
    for tau_m in roots {
        let position_m: Vec<f64> = (0..d).map(|axis| p0[axis] + p1[axis] * tau_m).collect();
        // `tau_m = c * t0` is the emission-distance unknown introduced by the
        // linearization. It carries the sign of `t0` in the caller's time base,
        // so a negative value is legitimate (an origin before the epoch the
        // arrivals are measured against), and its scale is problem-defined, so
        // no absolute cutoff is defensible either. Reject only a non-finite
        // candidate and let the ToA SSE select the physically consistent root.
        if !tau_m.is_finite() || position_m.iter().any(|value| !value.is_finite()) {
            continue;
        }
        let origin_time_s = tau_m / propagation_speed_m_s;
        let sse = toa_sse(
            sensors,
            arrival_times_s,
            &resolved.speeds_m_s,
            &position_m,
            origin_time_s,
        );
        if sse < best_sse {
            best_sse = sse;
            best = Some(SourceInitialGuess {
                position_m,
                origin_time_s: Some(origin_time_s),
                residual_rms_s: (sse / sensors.len() as f64).sqrt(),
            });
        }
    }
    best.ok_or(SourceLocalizationError::InitializerSingular)
}

fn closed_form_tdoa_initial_guess(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    resolved: &ResolvedInputs,
    reference_sensor: usize,
) -> Result<SourceInitialGuess, SourceLocalizationError> {
    let d = resolved.dimension;
    let ref_pos = &sensors[reference_sensor].position_m;
    let ref_norm2 = dot(ref_pos, ref_pos);
    let mut a = Vec::with_capacity(sensors.len() - 1);
    let mut b = Vec::with_capacity(sensors.len() - 1);
    let mut h = Vec::with_capacity(sensors.len() - 1);
    for (i, sensor) in sensors.iter().enumerate() {
        if i == reference_sensor {
            continue;
        }
        let row: Vec<f64> = sensor
            .position_m
            .iter()
            .zip(ref_pos)
            .map(|(s, r)| s - r)
            .collect();
        let delta_range_m =
            propagation_speed_m_s * (arrival_times_s[i] - arrival_times_s[reference_sensor]);
        let delta_norm = dot(&sensor.position_m, &sensor.position_m) - ref_norm2;
        a.push(row);
        b.push(0.5 * (delta_norm - delta_range_m * delta_range_m));
        h.push(-delta_range_m);
    }
    let p0 = least_squares(&a, &b)?;
    let p1 = least_squares(&a, &h)?;
    let q: Vec<f64> = p0.iter().zip(ref_pos).map(|(p, r)| p - r).collect();
    let roots = quadratic_roots(dot(&p1, &p1) - 1.0, 2.0 * dot(&q, &p1), dot(&q, &q))?;

    let mut best: Option<SourceInitialGuess> = None;
    let mut best_sse = f64::INFINITY;
    for rho_m in roots {
        if rho_m < 0.0 {
            continue;
        }
        let position_m: Vec<f64> = (0..d).map(|axis| p0[axis] + p1[axis] * rho_m).collect();
        let origin_time_s =
            estimate_origin_time_s(sensors, arrival_times_s, &resolved.speeds_m_s, &position_m);
        let sse = tdoa_sse(
            sensors,
            arrival_times_s,
            &resolved.speeds_m_s,
            &position_m,
            reference_sensor,
        );
        if sse < best_sse {
            best_sse = sse;
            best = Some(SourceInitialGuess {
                position_m,
                origin_time_s: Some(origin_time_s),
                residual_rms_s: (sse / (sensors.len() - 1) as f64).sqrt(),
            });
        }
    }
    best.ok_or(SourceLocalizationError::InitializerSingular)
}

fn compute_influence(
    solution: &SourceSolution,
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    options: &SourceLocateOptions,
) -> Vec<SourceSensorInfluence> {
    let speeds = match sensor_speeds(sensors, propagation_speed_m_s) {
        Ok(speeds) => speeds,
        Err(_) => return Vec::new(),
    };
    let origin_time_s = solution.origin_time_s.unwrap_or_else(|| {
        estimate_origin_time_for_loss_s(
            sensors,
            arrival_times_s,
            &speeds,
            &solution.position_m,
            options.loss,
            options.f_scale_s,
        )
    });
    let full_residuals = toa_residuals(
        sensors,
        arrival_times_s,
        &speeds,
        &solution.position_m,
        origin_time_s,
    );
    let sigma = options.timing_sigma_s.max(f64::MIN_POSITIVE);

    (0..sensors.len())
        .map(|sensor_index| {
            let loo = leave_one_out_solution(
                sensors,
                arrival_times_s,
                propagation_speed_m_s,
                options,
                sensor_index,
            );
            let (leave_one_out_residual_s, position_delta_m, origin_time_delta_s) =
                if let Some((loo_solution, loo_origin)) =
                    loo.and_then(|solution| solution.origin_time_s.map(|time| (solution, time)))
                {
                    let held_out_residual = single_toa_residual(
                        &sensors[sensor_index],
                        arrival_times_s[sensor_index],
                        speeds[sensor_index],
                        &loo_solution.position_m,
                        loo_origin,
                    );
                    (
                        Some(held_out_residual),
                        Some(distance(&solution.position_m, &loo_solution.position_m)),
                        Some((origin_time_s - loo_origin).abs()),
                    )
                } else {
                    (None, None, None)
                };
            let loss_weight = loss_weight(
                options.loss,
                options.f_scale_s,
                full_residuals[sensor_index],
            );
            SourceSensorInfluence {
                sensor_index,
                residual_s: full_residuals[sensor_index],
                leave_one_out_residual_s,
                position_delta_m,
                origin_time_delta_s,
                loss_weight,
                score: influence_score(
                    full_residuals[sensor_index],
                    leave_one_out_residual_s,
                    sigma,
                ),
            }
        })
        .collect()
}

fn leave_one_out_solution(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    options: &SourceLocateOptions,
    excluded: usize,
) -> Option<SourceSolution> {
    let mut sub_sensors = Vec::with_capacity(sensors.len() - 1);
    let mut sub_arrivals = Vec::with_capacity(arrival_times_s.len() - 1);
    for (i, sensor) in sensors.iter().enumerate() {
        if i == excluded {
            continue;
        }
        sub_sensors.push(sensor.clone());
        sub_arrivals.push(arrival_times_s[i]);
    }
    let mut sub_options = options.clone();
    sub_options.mode = match options.mode {
        SourceSolveMode::Toa => SourceSolveMode::Toa,
        SourceSolveMode::Tdoa { reference_sensor } => {
            if excluded == reference_sensor {
                SourceSolveMode::Tdoa {
                    reference_sensor: 0,
                }
            } else if excluded < reference_sensor {
                SourceSolveMode::Tdoa {
                    reference_sensor: reference_sensor - 1,
                }
            } else {
                SourceSolveMode::Tdoa { reference_sensor }
            }
        }
    };
    locate_source_inner(
        &sub_sensors,
        &sub_arrivals,
        propagation_speed_m_s,
        &sub_options,
        false,
    )
    .ok()
}

fn resolve_locate_inputs(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    propagation_speed_m_s: f64,
    options: &SourceLocateOptions,
) -> Result<ResolvedInputs, SourceLocalizationError> {
    if sensors.len() != arrival_times_s.len() {
        return Err(invalid_input(
            "arrival_times_s",
            "length must match sensors",
        ));
    }
    for &arrival in arrival_times_s {
        validate_finite("arrival_times_s", arrival)?;
    }
    validate_positive("timing_sigma_s", options.timing_sigma_s)?;
    if options.loss != Loss::Linear {
        validate_positive("f_scale_s", options.f_scale_s)?;
    }
    validate_optional_positive("ftol", options.ftol)?;
    validate_optional_positive("xtol", options.xtol)?;
    validate_optional_positive("gtol", options.gtol)?;
    if options.max_nfev == Some(0) {
        return Err(invalid_input("max_nfev", "must be positive"));
    }
    let geometry = resolve_geometry_inputs(
        sensors,
        sensors
            .first()
            .map(|sensor| sensor.position_m.as_slice())
            .unwrap_or(&[]),
        propagation_speed_m_s,
    )?;
    if let SourceSolveMode::Tdoa { reference_sensor } = options.mode {
        if reference_sensor >= sensors.len() {
            return Err(invalid_input("reference_sensor", "out of range"));
        }
    }
    let needed = geometry.dimension + 1;
    if sensors.len() < needed {
        return Err(SourceLocalizationError::TooFewSensors {
            sensors: sensors.len(),
            needed,
        });
    }
    Ok(ResolvedInputs {
        dimension: geometry.dimension,
        speeds_m_s: geometry.speeds_m_s,
        mode: options.mode,
    })
}

fn resolve_geometry_inputs(
    sensors: &[Sensor],
    source_position_m: &[f64],
    propagation_speed_m_s: f64,
) -> Result<ResolvedGeometry, SourceLocalizationError> {
    if sensors.is_empty() {
        return Err(invalid_input("sensors", "must not be empty"));
    }
    validate_positive("propagation_speed_m_s", propagation_speed_m_s)?;
    let dimension = sensors[0].position_m.len();
    if !(2..=3).contains(&dimension) {
        return Err(invalid_input("position_m", "length must be 2 or 3"));
    }
    if !source_position_m.is_empty() && source_position_m.len() != dimension {
        return Err(invalid_input(
            "source_position_m",
            "length must match sensors",
        ));
    }
    for sensor in sensors {
        if sensor.position_m.len() != dimension {
            return Err(invalid_input("position_m", "length must match sensors"));
        }
        for &value in &sensor.position_m {
            validate_finite("position_m", value)?;
        }
        if let Some(speed) = sensor.propagation_speed_m_s {
            validate_positive("sensor.propagation_speed_m_s", speed)?;
        }
    }
    for &value in source_position_m {
        validate_finite("source_position_m", value)?;
    }
    Ok(ResolvedGeometry {
        dimension,
        speeds_m_s: sensor_speeds(sensors, propagation_speed_m_s)?,
    })
}

fn sensor_speeds(
    sensors: &[Sensor],
    propagation_speed_m_s: f64,
) -> Result<Vec<f64>, SourceLocalizationError> {
    validate_positive("propagation_speed_m_s", propagation_speed_m_s)?;
    sensors
        .iter()
        .map(|sensor| {
            let speed = sensor
                .propagation_speed_m_s
                .unwrap_or(propagation_speed_m_s);
            validate_positive("sensor.propagation_speed_m_s", speed)?;
            Ok(speed)
        })
        .collect()
}

fn source_toa_design_rows(
    sensors: &[Sensor],
    source_position_m: &[f64],
    resolved: &ResolvedGeometry,
) -> Result<Vec<Vec<f64>>, SourceLocalizationError> {
    sensors
        .iter()
        .zip(&resolved.speeds_m_s)
        .map(|(sensor, &speed)| {
            let mut row = vec![0.0; resolved.dimension + 1];
            let range_m = distance(source_position_m, &sensor.position_m);
            if range_m <= 0.0 {
                return Err(invalid_input(
                    "source_position_m",
                    "coincident with a sensor",
                ));
            }
            for axis in 0..resolved.dimension {
                row[axis] = (source_position_m[axis] - sensor.position_m[axis]) / range_m / speed;
            }
            row[resolved.dimension] = 1.0;
            Ok(row)
        })
        .collect()
}

struct JacobianSvdDiagnostics {
    rank: usize,
    condition_number: f64,
    cofactor: Vec<Vec<f64>>,
}

#[cfg(test)]
fn cofactor_trace_from_jacobian(jac: &[f64], m: usize, n: usize) -> Option<f64> {
    let cofactor = cofactor_from_jacobian(jac, m, n)?;
    Some(cofactor_trace(&cofactor))
}

#[cfg(test)]
fn cofactor_from_jacobian(jac: &[f64], m: usize, n: usize) -> Option<Vec<Vec<f64>>> {
    Some(jacobian_svd_diagnostics(jac, m, n)?.cofactor)
}

fn jacobian_svd_diagnostics(jac: &[f64], m: usize, n: usize) -> Option<JacobianSvdDiagnostics> {
    if m == 0 || n == 0 || jac.len() != m.checked_mul(n)? {
        return None;
    }
    let matrix = DMatrix::from_row_slice(m, n, jac);
    let svd = matrix.svd(false, true);
    let diagnostics = singular_value_diagnostics(svd.singular_values.as_slice(), m, n);
    let v_t = svd.v_t?;
    let largest = svd.singular_values.iter().copied().fold(0.0_f64, f64::max);
    let threshold = largest * (m.max(n) as f64) * f64::EPSILON;
    let mut cofactor = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let mut value = 0.0;
            for (component, &singular_value) in svd.singular_values.iter().enumerate() {
                if singular_value > threshold {
                    let inverse_square = (singular_value * singular_value).recip();
                    value += v_t[(component, i)] * inverse_square * v_t[(component, j)];
                }
            }
            cofactor[i][j] = value;
            cofactor[j][i] = value;
        }
    }
    if cofactor.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    Some(JacobianSvdDiagnostics {
        rank: diagnostics.rank,
        condition_number: diagnostics.condition_number,
        cofactor,
    })
}

fn cofactor_trace(cofactor: &[Vec<f64>]) -> f64 {
    (0..cofactor.len()).map(|idx| cofactor[idx][idx]).sum()
}

fn covariance_from_state_cofactor(
    cofactor: &[Vec<f64>],
    dimension: usize,
    timing_sigma_s: f64,
    has_origin_time: bool,
) -> SourceCovariance {
    let scale = timing_sigma_s * timing_sigma_s;
    let state: Vec<Vec<f64>> = cofactor
        .iter()
        .map(|row| row.iter().map(|value| value * scale).collect())
        .collect();
    let position_m2: Vec<Vec<f64>> = (0..dimension)
        .map(|i| (0..dimension).map(|j| state[i][j]).collect())
        .collect();
    SourceCovariance {
        origin_time_s2: if has_origin_time {
            Some(state[dimension][dimension])
        } else {
            None
        },
        state,
        position_m2,
        timing_sigma_s,
    }
}

fn solver_options(config: &SourceLocateOptions) -> TrfOptions {
    let mut options = TrfOptions::default();
    if let Some(ftol) = config.ftol {
        options.ftol = ftol;
    }
    if let Some(xtol) = config.xtol {
        options.xtol = xtol;
    }
    if let Some(gtol) = config.gtol {
        options.gtol = gtol;
    }
    options.max_nfev = config.max_nfev;
    options.x_scale = XScale::Jac;
    options.loss = config.loss;
    options.f_scale = config.f_scale_s;
    options
}

fn least_squares(a: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, SourceLocalizationError> {
    let n = a.first().map(Vec::len).unwrap_or(0);
    if n == 0 || a.len() != y.len() || a.len() < n {
        return Err(SourceLocalizationError::InitializerSingular);
    }
    let mut normal = vec![vec![0.0_f64; n]; n];
    let mut rhs = vec![0.0_f64; n];
    for (row, &value) in a.iter().zip(y) {
        if row.len() != n {
            return Err(SourceLocalizationError::InitializerSingular);
        }
        for i in 0..n {
            rhs[i] += row[i] * value;
            for j in 0..n {
                normal[i][j] += row[i] * row[j];
            }
        }
    }
    let inv = crate::astro::math::linear::invert_symmetric_pd(&normal)
        .ok_or(SourceLocalizationError::InitializerSingular)?;
    Ok((0..n)
        .map(|i| (0..n).map(|j| inv[i][j] * rhs[j]).sum())
        .collect())
}

fn quadratic_roots(a: f64, b: f64, c: f64) -> Result<Vec<f64>, SourceLocalizationError> {
    if !a.is_finite() || !b.is_finite() || !c.is_finite() {
        return Err(SourceLocalizationError::InitializerSingular);
    }
    let coefficient_scale = a.abs().max(b.abs()).max(c.abs()).max(1.0);
    let coefficient_tolerance = QUADRATIC_REL_EPS * coefficient_scale;
    if a.abs() <= coefficient_tolerance {
        if b.abs() <= coefficient_tolerance {
            return Err(SourceLocalizationError::InitializerSingular);
        }
        return Ok(vec![-c / b]);
    }
    let b_squared = b * b;
    let four_ac = 4.0 * a * c;
    let disc = b_squared - four_ac;
    let discriminant_scale = b_squared.abs().max(four_ac.abs()).max(1.0);
    if disc < -QUADRATIC_REL_EPS * discriminant_scale || !disc.is_finite() {
        return Err(SourceLocalizationError::InitializerSingular);
    }
    let root = disc.max(0.0).sqrt();
    Ok(vec![(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)])
}

fn toa_sse(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    speeds_m_s: &[f64],
    position_m: &[f64],
    origin_time_s: f64,
) -> f64 {
    toa_residuals(
        sensors,
        arrival_times_s,
        speeds_m_s,
        position_m,
        origin_time_s,
    )
    .iter()
    .map(|value| value * value)
    .sum()
}

fn tdoa_sse(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    speeds_m_s: &[f64],
    position_m: &[f64],
    reference_sensor: usize,
) -> f64 {
    let ref_time =
        distance(position_m, &sensors[reference_sensor].position_m) / speeds_m_s[reference_sensor];
    let mut sse = 0.0;
    for (i, sensor) in sensors.iter().enumerate() {
        if i == reference_sensor {
            continue;
        }
        let predicted = distance(position_m, &sensor.position_m) / speeds_m_s[i] - ref_time;
        let observed = arrival_times_s[i] - arrival_times_s[reference_sensor];
        let residual = predicted - observed;
        sse += residual * residual;
    }
    sse
}

fn toa_residuals(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    speeds_m_s: &[f64],
    position_m: &[f64],
    origin_time_s: f64,
) -> Vec<f64> {
    sensors
        .iter()
        .enumerate()
        .map(|(i, sensor)| {
            single_toa_residual(
                sensor,
                arrival_times_s[i],
                speeds_m_s[i],
                position_m,
                origin_time_s,
            )
        })
        .collect()
}

fn single_toa_residual(
    sensor: &Sensor,
    arrival_time_s: f64,
    speed_m_s: f64,
    position_m: &[f64],
    origin_time_s: f64,
) -> f64 {
    origin_time_s + distance(position_m, &sensor.position_m) / speed_m_s - arrival_time_s
}

fn estimate_origin_time_s(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    speeds_m_s: &[f64],
    position_m: &[f64],
) -> f64 {
    let sum: f64 = sensors
        .iter()
        .enumerate()
        .map(|(i, sensor)| {
            arrival_times_s[i] - distance(position_m, &sensor.position_m) / speeds_m_s[i]
        })
        .sum();
    sum / sensors.len() as f64
}

fn estimate_origin_time_for_loss_s(
    sensors: &[Sensor],
    arrival_times_s: &[f64],
    speeds_m_s: &[f64],
    position_m: &[f64],
    loss: Loss,
    f_scale_s: f64,
) -> f64 {
    let unweighted = estimate_origin_time_s(sensors, arrival_times_s, speeds_m_s, position_m);
    if loss == Loss::Linear {
        // Preserve the historical expression path, including its rounding, for
        // the default loss.
        return unweighted;
    }

    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;
    for (i, sensor) in sensors.iter().enumerate() {
        let candidate_s =
            arrival_times_s[i] - distance(position_m, &sensor.position_m) / speeds_m_s[i];
        let residual_s = unweighted - candidate_s;
        let weight = loss_weight(loss, f_scale_s, residual_s);
        weighted_sum += weight * candidate_s;
        weight_sum += weight;
    }
    if weight_sum > 0.0 && weight_sum.is_finite() && weighted_sum.is_finite() {
        weighted_sum / weight_sum
    } else {
        unweighted
    }
}

fn fill_range_derivative(position_m: &[f64], sensor_m: &[f64], speed_m_s: f64, out: &mut [f64]) {
    let range_m = distance(position_m, sensor_m);
    if range_m <= 0.0 || !range_m.is_finite() {
        out.fill(0.0);
        return;
    }
    for axis in 0..out.len() {
        out[axis] = (position_m[axis] - sensor_m[axis]) / range_m / speed_m_s;
    }
}

fn loss_weight(loss: Loss, f_scale_s: f64, residual_s: f64) -> f64 {
    match loss {
        Loss::Linear => 1.0,
        Loss::Huber => {
            let z = (residual_s / f_scale_s) * (residual_s / f_scale_s);
            if z <= 1.0 {
                1.0
            } else {
                z.sqrt().recip()
            }
        }
        Loss::SoftL1 => {
            let z = (residual_s / f_scale_s) * (residual_s / f_scale_s);
            (1.0 + z).sqrt().recip()
        }
        Loss::Cauchy => {
            let z = (residual_s / f_scale_s) * (residual_s / f_scale_s);
            (1.0 + z).recip()
        }
        Loss::Arctan => {
            let z = (residual_s / f_scale_s) * (residual_s / f_scale_s);
            (1.0 + z * z).recip()
        }
    }
}

fn influence_score(residual_s: f64, leave_one_out_residual_s: Option<f64>, sigma_s: f64) -> f64 {
    leave_one_out_residual_s
        .unwrap_or(residual_s)
        .abs()
        .max(residual_s.abs())
        / sigma_s
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

const fn identity_rotation() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn invalid_input(field: &'static str, reason: &'static str) -> SourceLocalizationError {
    SourceLocalizationError::InvalidInput { field, reason }
}

fn validate_optional_positive(
    field: &'static str,
    value: Option<f64>,
) -> Result<(), SourceLocalizationError> {
    if let Some(value) = value {
        validate_positive(field, value)?;
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), SourceLocalizationError> {
    validate_finite(field, value)?;
    if value <= 0.0 {
        return Err(invalid_input(field, "must be > 0"));
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), SourceLocalizationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

#[cfg(test)]
mod tests {
    //! Analytic source-localization fixtures.
    //!
    //! The tests below use Euclidean ranges, closed-form normal equations, and
    //! synthetic corrupted arrivals. They do not compare against another
    //! implementation.

    // Exercise the construction pattern available to callers of the
    // non-exhaustive options type.
    #![allow(clippy::field_reassign_with_default)]

    use super::*;

    fn arrivals(sensors: &[Sensor], source: &[f64], origin: f64, speed: f64) -> Vec<f64> {
        sensors
            .iter()
            .map(|sensor| {
                let s = sensor.propagation_speed_m_s.unwrap_or(speed);
                origin + distance(source, &sensor.position_m) / s
            })
            .collect()
    }

    fn no_influence(options: SourceLocateOptions) -> SourceLocateConfig {
        SourceLocateConfig {
            options,
            include_influence: false,
        }
    }

    fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64) {
        for (axis, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "axis {axis}: actual {a}, expected {e}, tol {tol}"
            );
        }
    }

    fn assert_f64_bits(actual: f64, expected: f64, field: &str) {
        assert_eq!(
            actual.to_bits(),
            expected.to_bits(),
            "{field}: actual {actual:?}, expected {expected:?}"
        );
    }

    fn assert_covariance_bits(actual: &SourceCovariance, expected: &SourceCovariance, field: &str) {
        assert_eq!(actual.state.len(), expected.state.len());
        for (row_index, (actual_row, expected_row)) in
            actual.state.iter().zip(&expected.state).enumerate()
        {
            assert_eq!(actual_row.len(), expected_row.len());
            for (column_index, (&actual_value, &expected_value)) in
                actual_row.iter().zip(expected_row).enumerate()
            {
                assert_f64_bits(
                    actual_value,
                    expected_value,
                    &format!("{field}.state[{row_index}][{column_index}]"),
                );
            }
        }
        assert_eq!(actual.position_m2.len(), expected.position_m2.len());
        for (row_index, (actual_row, expected_row)) in actual
            .position_m2
            .iter()
            .zip(&expected.position_m2)
            .enumerate()
        {
            assert_eq!(actual_row.len(), expected_row.len());
            for (column_index, (&actual_value, &expected_value)) in
                actual_row.iter().zip(expected_row).enumerate()
            {
                assert_f64_bits(
                    actual_value,
                    expected_value,
                    &format!("{field}.position_m2[{row_index}][{column_index}]"),
                );
            }
        }
        match (actual.origin_time_s2, expected.origin_time_s2) {
            (Some(actual), Some(expected)) => {
                assert_f64_bits(actual, expected, &format!("{field}.origin_time_s2"));
            }
            (None, None) => {}
            pair => panic!("{field}.origin_time_s2 differs: {pair:?}"),
        }
        assert_f64_bits(
            actual.timing_sigma_s,
            expected.timing_sigma_s,
            &format!("{field}.timing_sigma_s"),
        );
    }

    fn assert_solution_bits_except_influence(actual: &SourceSolution, expected: &SourceSolution) {
        assert_eq!(actual.position_m.len(), expected.position_m.len());
        for (axis, (&actual, &expected)) in actual
            .position_m
            .iter()
            .zip(&expected.position_m)
            .enumerate()
        {
            assert_f64_bits(actual, expected, &format!("position_m[{axis}]"));
        }
        match (actual.origin_time_s, expected.origin_time_s) {
            (Some(actual), Some(expected)) => {
                assert_f64_bits(actual, expected, "origin_time_s");
            }
            (None, None) => {}
            pair => panic!("origin_time_s differs: {pair:?}"),
        }
        match (&actual.covariance, &expected.covariance) {
            (Some(actual), Some(expected)) => {
                assert_covariance_bits(actual, expected, "covariance");
            }
            (None, None) => {}
            pair => panic!("covariance presence differs: {pair:?}"),
        }
        assert_eq!(actual.residuals.len(), expected.residuals.len());
        for (index, (actual, expected)) in
            actual.residuals.iter().zip(&expected.residuals).enumerate()
        {
            assert_eq!(actual.sensor_index, expected.sensor_index);
            assert_eq!(
                actual.reference_sensor_index,
                expected.reference_sensor_index
            );
            assert_f64_bits(
                actual.residual_s,
                expected.residual_s,
                &format!("residuals[{index}].residual_s"),
            );
        }
        assert_eq!(actual.geometry_quality.tier, expected.geometry_quality.tier);
        assert_eq!(
            actual.geometry_quality.redundancy,
            expected.geometry_quality.redundancy
        );
        assert_eq!(actual.geometry_quality.rank, expected.geometry_quality.rank);
        assert_f64_bits(
            actual.geometry_quality.condition_number,
            expected.geometry_quality.condition_number,
            "geometry_quality.condition_number",
        );
        assert_f64_bits(
            actual.geometry_quality.gdop,
            expected.geometry_quality.gdop,
            "geometry_quality.gdop",
        );
        assert_eq!(
            actual.geometry_quality.raim_checkable,
            expected.geometry_quality.raim_checkable
        );
        assert_eq!(
            actual.geometry_quality.covariance_validated,
            expected.geometry_quality.covariance_validated
        );
        assert_eq!(
            actual.initial_guess.position_m.len(),
            expected.initial_guess.position_m.len()
        );
        for (axis, (&actual, &expected)) in actual
            .initial_guess
            .position_m
            .iter()
            .zip(&expected.initial_guess.position_m)
            .enumerate()
        {
            assert_f64_bits(
                actual,
                expected,
                &format!("initial_guess.position_m[{axis}]"),
            );
        }
        match (
            actual.initial_guess.origin_time_s,
            expected.initial_guess.origin_time_s,
        ) {
            (Some(actual), Some(expected)) => {
                assert_f64_bits(actual, expected, "initial_guess.origin_time_s");
            }
            (None, None) => {}
            pair => panic!("initial_guess.origin_time_s differs: {pair:?}"),
        }
        assert_f64_bits(
            actual.initial_guess.residual_rms_s,
            expected.initial_guess.residual_rms_s,
            "initial_guess.residual_rms_s",
        );
        assert_eq!(actual.status, expected.status);
        assert_eq!(actual.nfev, expected.nfev);
        assert_eq!(actual.njev, expected.njev);
        assert_f64_bits(actual.cost, expected.cost, "cost");
        assert_f64_bits(actual.optimality, expected.optimality, "optimality");
    }

    struct SplitMix64 {
        state: u64,
        spare_normal: Option<f64>,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self {
                state: seed,
                spare_normal: None,
            }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut value = self.state;
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            value ^ (value >> 31)
        }

        fn unit_f64(&mut self) -> f64 {
            let bits = 0x3ff0_0000_0000_0000 | (self.next_u64() >> 12);
            f64::from_bits(bits) - 1.0
        }

        fn standard_normal(&mut self) -> f64 {
            if let Some(value) = self.spare_normal.take() {
                return value;
            }
            loop {
                let u = 2.0 * self.unit_f64() - 1.0;
                let v = 2.0 * self.unit_f64() - 1.0;
                let radius_squared = u * u + v * v;
                if radius_squared > 0.0 && radius_squared < 1.0 {
                    let scale = (-2.0 * radius_squared.ln() / radius_squared).sqrt();
                    self.spare_normal = Some(v * scale);
                    return u * scale;
                }
            }
        }
    }

    #[test]
    fn closed_form_toa_initializer_recovers_clean_3d() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0, 0.0]),
            Sensor::new(vec![1200.0, 0.0, 0.0]),
            Sensor::new(vec![0.0, 900.0, 0.0]),
            Sensor::new(vec![0.0, 0.0, 700.0]),
            Sensor::new(vec![1100.0, 800.0, 600.0]),
        ];
        let source = vec![320.0, 260.0, 180.0];
        let origin = 12.5;
        let speed = 343.0;
        let times = arrivals(&sensors, &source, origin, speed);

        let seed =
            closed_form_initial_guess(&sensors, &times, speed, SourceSolveMode::Toa).expect("seed");
        assert_vec_close(&seed.position_m, &source, 1.0e-8);
        assert!((seed.origin_time_s.unwrap() - origin).abs() < 1.0e-10);
        assert!(seed.residual_rms_s < 1.0e-11);
    }

    #[test]
    fn locate_source_toa_recovers_clean_3d() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0, 0.0]),
            Sensor::new(vec![1200.0, 0.0, 0.0]),
            Sensor::new(vec![0.0, 900.0, 0.0]),
            Sensor::new(vec![0.0, 0.0, 700.0]),
            Sensor::new(vec![1100.0, 800.0, 600.0]),
        ];
        let source = vec![320.0, 260.0, 180.0];
        let origin = 12.5;
        let speed = 343.0;
        let times = arrivals(&sensors, &source, origin, speed);
        let mut options = SourceLocateOptions::default();
        options.timing_sigma_s = 0.001;

        let solution = locate_source(&sensors, &times, speed, &options).expect("solution");
        assert_vec_close(&solution.position_m, &source, 1.0e-7);
        assert!((solution.origin_time_s.unwrap() - origin).abs() < 1.0e-10);
        assert!(solution.covariance.is_some());
        assert!(solution
            .residuals
            .iter()
            .all(|row| row.residual_s.abs() < 1.0e-10));
    }

    #[test]
    fn locate_source_toa_recovers_clean_2d() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0]),
            Sensor::new(vec![700.0, 0.0]),
            Sensor::new(vec![0.0, 600.0]),
            Sensor::new(vec![650.0, 550.0]),
        ];
        let source = [210.0, 170.0];
        let origin = 2.75;
        let speed = 343.0;
        let times = arrivals(&sensors, &source, origin, speed);
        let options = SourceLocateOptions::default();
        let config = no_influence(options);

        let solution = locate_source_with(&sensors, &times, speed, &config).expect("solution");

        assert_vec_close(&solution.position_m, &source, 1.0e-8);
        assert!((solution.origin_time_s.unwrap() - origin).abs() < 1.0e-10);
        assert!(solution.per_sensor_influence.is_empty());
    }

    #[test]
    fn locate_source_toa_recovers_negative_origin_time() {
        // Arrival times measured against an epoch later than the emission:
        // the origin time is negative and the seed's emission-distance unknown
        // `c * t0` is negative with it. That is a valid problem, not a
        // rejected root.
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0, 0.0]),
            Sensor::new(vec![1200.0, 0.0, 0.0]),
            Sensor::new(vec![0.0, 900.0, 0.0]),
            Sensor::new(vec![0.0, 0.0, 700.0]),
            Sensor::new(vec![1100.0, 800.0, 600.0]),
        ];
        let source = [320.0, 260.0, 180.0];
        let origin = -4.5;
        let speed = 343.0;
        let times = arrivals(&sensors, &source, origin, speed);
        let options = SourceLocateOptions::default();
        let config = no_influence(options);

        let seed =
            closed_form_initial_guess(&sensors, &times, speed, SourceSolveMode::Toa).expect("seed");
        assert!((seed.origin_time_s.unwrap() - origin).abs() < 1.0e-9);

        let solution = locate_source_with(&sensors, &times, speed, &config).expect("solution");
        assert_vec_close(&solution.position_m, &source, 1.0e-7);
        assert!((solution.origin_time_s.unwrap() - origin).abs() < 1.0e-10);
    }

    #[test]
    fn influence_opt_out_preserves_every_other_output_bit() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0, 0.0]),
            Sensor::new(vec![1200.0, 0.0, 0.0]),
            Sensor::new(vec![0.0, 900.0, 0.0]),
            Sensor::new(vec![0.0, 0.0, 700.0]),
            Sensor::new(vec![1100.0, 800.0, 600.0]),
        ];
        let source = [320.0, 260.0, 180.0];
        let speed = 343.0;
        let mut times = arrivals(&sensors, &source, 12.5, speed);
        for (time, noise) in times
            .iter_mut()
            .zip([0.00031, -0.00022, 0.00017, -0.00008, 0.00041])
        {
            *time += noise;
        }
        let mut with_influence_options = SourceLocateOptions::default();
        with_influence_options.timing_sigma_s = 0.001;
        let with_influence = locate_source(&sensors, &times, speed, &with_influence_options)
            .expect("solution with influence");
        assert_eq!(with_influence.per_sensor_influence.len(), sensors.len());

        let without_influence_options = with_influence_options.clone();
        let without_influence_config = no_influence(without_influence_options);
        let without_influence =
            locate_source_with(&sensors, &times, speed, &without_influence_config)
                .expect("solution without influence");

        assert!(without_influence.per_sensor_influence.is_empty());
        assert_solution_bits_except_influence(&without_influence, &with_influence);
    }

    #[test]
    fn locate_source_toa_geometry_quality_is_nominal() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0, 0.0]),
            Sensor::new(vec![2.0, 0.0, 0.0]),
            Sensor::new(vec![0.0, 2.0, 0.0]),
            Sensor::new(vec![0.0, 0.0, 2.0]),
            Sensor::new(vec![2.0, 2.0, 2.0]),
        ];
        let source = vec![0.4, 0.6, 0.5];
        let origin = 1.25;
        let speed = 1.0;
        let times = arrivals(&sensors, &source, origin, speed);

        let solution = locate_source(&sensors, &times, speed, &SourceLocateOptions::default())
            .expect("well-posed source solve");

        assert_eq!(
            solution.geometry_quality.tier,
            crate::geometry_quality::ObservabilityTier::Nominal
        );
        assert_eq!(solution.geometry_quality.rank, 4);
        assert_eq!(solution.geometry_quality.redundancy, 1);
        assert!(solution.geometry_quality.raim_checkable);
        assert!(solution.geometry_quality.covariance_validated);
    }

    #[test]
    fn locate_source_tdoa_recovers_clean_2d() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0]),
            Sensor::new(vec![1000.0, 0.0]),
            Sensor::new(vec![0.0, 800.0]),
            Sensor::new(vec![900.0, 900.0]),
        ];
        let source = vec![300.0, 260.0];
        let origin = 4.0;
        let speed = 340.0;
        let times = arrivals(&sensors, &source, origin, speed);
        let mut options = SourceLocateOptions::default();
        options.mode = SourceSolveMode::Tdoa {
            reference_sensor: 0,
        };
        options.timing_sigma_s = 0.001;

        let solution = locate_source(&sensors, &times, speed, &options).expect("solution");
        assert_vec_close(&solution.position_m, &source, 1.0e-7);
        assert!((solution.origin_time_s.unwrap() - origin).abs() < 1.0e-9);
        assert_eq!(solution.residuals.len(), sensors.len() - 1);
    }

    #[test]
    fn locate_source_tdoa_recovers_clean_3d() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0, 0.0]),
            Sensor::new(vec![1000.0, 0.0, 0.0]),
            Sensor::new(vec![0.0, 900.0, 0.0]),
            Sensor::new(vec![0.0, 0.0, 800.0]),
            Sensor::new(vec![900.0, 850.0, 750.0]),
        ];
        let source = [280.0, 310.0, 190.0];
        let origin = 6.25;
        let speed = 343.0;
        let times = arrivals(&sensors, &source, origin, speed);
        let mut options = SourceLocateOptions::default();
        options.mode = SourceSolveMode::Tdoa {
            reference_sensor: 3,
        };
        let config = no_influence(options);

        let solution = locate_source_with(&sensors, &times, speed, &config).expect("solution");

        assert_vec_close(&solution.position_m, &source, 1.0e-7);
        assert!((solution.origin_time_s.unwrap() - origin).abs() < 1.0e-9);
        assert_eq!(solution.residuals.len(), sensors.len() - 1);
    }

    #[test]
    fn tdoa_influence_populates_excluded_reference_sensor() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0]),
            Sensor::new(vec![1000.0, 0.0]),
            Sensor::new(vec![0.0, 800.0]),
            Sensor::new(vec![900.0, 900.0]),
            Sensor::new(vec![-350.0, 500.0]),
        ];
        let source = [300.0, 260.0];
        let speed = 340.0;
        let times = arrivals(&sensors, &source, 4.0, speed);
        let reference_sensor = 2;
        let mut options = SourceLocateOptions::default();
        options.mode = SourceSolveMode::Tdoa { reference_sensor };
        options.timing_sigma_s = 0.001;

        let solution = locate_source(&sensors, &times, speed, &options).expect("solution");
        let reference = solution
            .per_sensor_influence
            .iter()
            .find(|record| record.sensor_index == reference_sensor)
            .expect("reference influence record");

        assert!(reference
            .leave_one_out_residual_s
            .is_some_and(f64::is_finite));
        assert!(reference
            .position_delta_m
            .is_some_and(|value| value.is_finite() && value >= 0.0));
        assert!(reference
            .origin_time_delta_s
            .is_some_and(|value| value.is_finite() && value >= 0.0));
        assert!(reference.score.is_finite());
    }

    #[test]
    fn per_sensor_speed_override_refines_from_uniform_seed() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0, 0.0]),
            Sensor::with_speed(vec![1200.0, 0.0, 0.0], 330.0),
            Sensor::new(vec![0.0, 900.0, 0.0]),
            Sensor::new(vec![0.0, 0.0, 700.0]),
            Sensor::new(vec![1100.0, 800.0, 600.0]),
        ];
        let source = vec![320.0, 260.0, 180.0];
        let origin = 12.5;
        let speed = 343.0;
        let times = arrivals(&sensors, &source, origin, speed);

        let solution =
            locate_source(&sensors, &times, speed, &SourceLocateOptions::default()).expect("solve");
        assert_vec_close(&solution.position_m, &source, 1.0e-6);
        assert!((solution.origin_time_s.unwrap() - origin).abs() < 1.0e-9);
    }

    #[test]
    fn source_dop_matches_hand_computed_square_layout() {
        let sensors = vec![
            Sensor::new(vec![100.0, 0.0]),
            Sensor::new(vec![-100.0, 0.0]),
            Sensor::new(vec![0.0, 100.0]),
            Sensor::new(vec![0.0, -100.0]),
        ];
        let source = vec![0.0, 0.0];
        let speed = 10.0;

        let d = source_dop(&sensors, &source, speed).expect("dop");
        assert!((d.pdop - 10.0).abs() < 1.0e-12);
        assert!((d.hdop - 10.0).abs() < 1.0e-12);
        assert_eq!(d.vdop.to_bits(), 0.0_f64.to_bits());
        assert!((d.tdop - 0.5).abs() < 1.0e-12);
        assert!((d.gdop - 100.25_f64.sqrt()).abs() < 1.0e-12);

        let crlb = source_crlb(&sensors, &source, speed, 0.01).expect("crlb");
        assert!((crlb.covariance.position_m2[0][0] - 0.005).abs() < 1.0e-15);
        assert!((crlb.covariance.position_m2[1][1] - 0.005).abs() < 1.0e-15);
        assert!((crlb.covariance.origin_time_s2.unwrap() - 0.000025).abs() < 1.0e-18);
    }

    #[test]
    fn corrupted_arrival_is_downweighted_and_flagged() {
        let sensors = vec![
            Sensor::new(vec![100.0, 0.0]),
            Sensor::new(vec![-100.0, 0.0]),
            Sensor::new(vec![0.0, 100.0]),
            Sensor::new(vec![0.0, -100.0]),
            Sensor::new(vec![120.0, 120.0]),
            Sensor::new(vec![-120.0, 80.0]),
            Sensor::new(vec![80.0, -140.0]),
            Sensor::new(vec![-160.0, -100.0]),
        ];
        let source = vec![15.0, -20.0];
        let origin = 1.25;
        let speed = 50.0;
        let mut times = arrivals(&sensors, &source, origin, speed);
        times[4] += 0.5;
        let mut options = SourceLocateOptions::default();
        options.loss = Loss::Huber;
        options.f_scale_s = 0.01;
        options.timing_sigma_s = 0.01;

        let solution = locate_source(&sensors, &times, speed, &options).expect("solution");
        let worst = solution
            .per_sensor_influence
            .iter()
            .max_by(|a, b| a.score.total_cmp(&b.score))
            .expect("influence");
        let runner_up = solution
            .per_sensor_influence
            .iter()
            .filter(|record| record.sensor_index != worst.sensor_index)
            .map(|record| record.score)
            .max_by(f64::total_cmp)
            .expect("runner-up influence");
        assert_eq!(worst.sensor_index, 4);
        assert!(worst.loss_weight < 0.05);
        assert!(
            worst.score > 20.0 * runner_up,
            "corrupted score {} was not twenty times runner-up {}",
            worst.score,
            runner_up
        );
        let expected = worst
            .leave_one_out_residual_s
            .unwrap_or(worst.residual_s)
            .abs()
            .max(worst.residual_s.abs())
            / options.timing_sigma_s;
        assert_f64_bits(worst.score, expected, "corrupted influence score");
    }

    #[test]
    fn influence_score_matches_hand_computation() {
        assert_f64_bits(
            influence_score(-0.03, Some(0.08), 0.01),
            8.0,
            "leave-one-out score",
        );
        assert_f64_bits(influence_score(-0.03, None, 0.01), 3.0, "fallback score");
    }

    #[test]
    fn tdoa_huber_origin_time_improves_on_unweighted_mean() {
        let sensors = vec![
            Sensor::new(vec![100.0, 0.0]),
            Sensor::new(vec![-100.0, 0.0]),
            Sensor::new(vec![0.0, 100.0]),
            Sensor::new(vec![0.0, -100.0]),
            Sensor::new(vec![120.0, 120.0]),
            Sensor::new(vec![-120.0, 80.0]),
        ];
        let source = [15.0, -20.0];
        let origin = 1.25;
        let speed = 50.0;
        let mut times = arrivals(&sensors, &source, origin, speed);
        times[4] += 0.5;
        let mut options = SourceLocateOptions::default();
        options.mode = SourceSolveMode::Tdoa {
            reference_sensor: 0,
        };
        options.loss = Loss::Huber;
        options.f_scale_s = 0.01;
        let config = no_influence(options);

        let solution = locate_source_with(&sensors, &times, speed, &config).expect("solution");
        let speeds = sensor_speeds(&sensors, speed).expect("speeds");
        let unweighted = estimate_origin_time_s(&sensors, &times, &speeds, &solution.position_m);
        let weighted = solution.origin_time_s.expect("TDOA origin time");

        assert!(
            (weighted - origin).abs() < (unweighted - origin).abs(),
            "weighted error {}, unweighted error {}",
            (weighted - origin).abs(),
            (unweighted - origin).abs()
        );
    }

    #[test]
    fn seeded_toa_noise_rms_tracks_crlb() {
        const TRIALS: usize = 256;
        const TIMING_SIGMA_S: f64 = 2.0e-4;

        let sensors = vec![
            Sensor::new(vec![-200.0, -100.0]),
            Sensor::new(vec![250.0, -80.0]),
            Sensor::new(vec![-150.0, 260.0]),
            Sensor::new(vec![220.0, 240.0]),
            Sensor::new(vec![20.0, -300.0]),
            Sensor::new(vec![350.0, 100.0]),
        ];
        let source = [40.0, 30.0];
        let origin = 3.0;
        let speed = 343.0;
        let clean_times = arrivals(&sensors, &source, origin, speed);
        let predicted = source_crlb(&sensors, &source, speed, TIMING_SIGMA_S)
            .expect("CRLB")
            .covariance
            .position_m2;
        let predicted_rms = (predicted[0][0] + predicted[1][1]).sqrt();
        let mut options = SourceLocateOptions::default();
        options.timing_sigma_s = TIMING_SIGMA_S;
        let config = no_influence(options);
        let mut rng = SplitMix64::new(0x534f_5552_4345_4d43);
        let mut squared_position_error_sum = 0.0;

        for _ in 0..TRIALS {
            let mut noisy_times = clean_times.clone();
            for time in &mut noisy_times {
                *time += TIMING_SIGMA_S * rng.standard_normal();
            }
            let solution =
                locate_source_with(&sensors, &noisy_times, speed, &config).expect("noisy solution");
            squared_position_error_sum += solution
                .position_m
                .iter()
                .zip(source)
                .map(|(estimated, truth)| (estimated - truth) * (estimated - truth))
                .sum::<f64>();
        }

        let sample_rms = (squared_position_error_sum / TRIALS as f64).sqrt();
        let ratio = sample_rms / predicted_rms;
        assert!(
            (0.8..=1.2).contains(&ratio),
            "sample RMS {sample_rms} m, predicted RMS {predicted_rms} m, ratio {ratio}"
        );
    }

    #[test]
    fn degenerate_seed_geometry_reports_initializer_singular() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0]),
            Sensor::new(vec![100.0, 0.0]),
            Sensor::new(vec![200.0, 0.0]),
            Sensor::new(vec![300.0, 0.0]),
        ];
        let times = arrivals(&sensors, &[50.0, 20.0], 1.0, 300.0);

        let error = closed_form_initial_guess(&sensors, &times, 300.0, SourceSolveMode::Toa)
            .expect_err("collinear seed must be singular");

        assert_eq!(error, SourceLocalizationError::InitializerSingular);
    }

    #[test]
    fn exhausted_solver_budget_reports_did_not_converge() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0, 0.0]),
            Sensor::new(vec![1200.0, 0.0, 0.0]),
            Sensor::new(vec![0.0, 900.0, 0.0]),
            Sensor::new(vec![0.0, 0.0, 700.0]),
            Sensor::new(vec![1100.0, 800.0, 600.0]),
        ];
        let source = [320.0, 260.0, 180.0];
        let speed = 343.0;
        let mut times = arrivals(&sensors, &source, 12.5, speed);
        for (time, noise) in times
            .iter_mut()
            .zip([0.00031, -0.00022, 0.00017, -0.00008, 0.00041])
        {
            *time += noise;
        }
        let mut options = SourceLocateOptions::default();
        options.max_nfev = Some(1);
        let config = no_influence(options);

        let error = locate_source_with(&sensors, &times, speed, &config)
            .expect_err("one evaluation cannot converge");

        assert_eq!(error, SourceLocalizationError::DidNotConverge { status: 0 });
    }

    #[test]
    fn empty_sensor_input_names_the_invalid_field() {
        let error = locate_source(&[], &[], 343.0, &SourceLocateOptions::default())
            .expect_err("empty sensors");

        assert_eq!(
            error,
            SourceLocalizationError::InvalidInput {
                field: "sensors",
                reason: "must not be empty",
            }
        );
    }

    #[test]
    fn degenerate_collinear_geometry_reports_singular_dop() {
        let sensors = vec![
            Sensor::new(vec![0.0, 0.0]),
            Sensor::new(vec![100.0, 0.0]),
            Sensor::new(vec![200.0, 0.0]),
            Sensor::new(vec![300.0, 0.0]),
        ];
        let err = source_dop(&sensors, &[50.0, 0.0], 300.0).expect_err("singular");
        assert!(matches!(
            err,
            SourceLocalizationError::Geometry(DopError::Singular)
        ));
    }

    #[test]
    fn source_collinear_timing_design_classifies_rank_deficient() {
        let jac = [
            1.0 / 300.0,
            0.0,
            1.0,
            -1.0 / 300.0,
            0.0,
            1.0,
            -1.0 / 300.0,
            0.0,
            1.0,
            -1.0 / 300.0,
            0.0,
            1.0,
        ];
        let quality = source_geometry_quality_from_jacobian(&jac, 4, 3).expect("quality");
        let pseudocofactor_trace =
            cofactor_trace_from_jacobian(&jac, 4, 3).expect("SVD pseudocofactor trace");

        assert_eq!(
            quality.tier,
            crate::geometry_quality::ObservabilityTier::RankDeficient
        );
        assert!(!quality.raim_checkable);
        assert!(!quality.covariance_validated);
        assert!(pseudocofactor_trace.is_finite() && pseudocofactor_trace >= 0.0);
    }
}
