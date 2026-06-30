//! Compact mean-element orbit approximation for fast position prediction.
//!
//! This is a **fitted, approximate** model of a satellite's motion - a small
//! set of mean elements that reproduce a position track over a window for
//! caching, transport, and visibility prediction. It is **not** orbit
//! determination: it discards short-period structure and is calibrated about the
//! residual it leaves behind (`rms_m`, `max_m`, and a source-backed drift
//! evaluation).
//!
//! Two models are available, selected at fit time via [`Model`]:
//!
//! - [`Model::CircularSecular`] (the default) - a circular orbit whose plane
//!   precesses at a constant nodal rate. Best for near-circular orbits
//!   (Galileo).
//! - [`Model::EccentricSecular`] - adds a nonsingular eccentricity so the
//!   radial `a·e` signal (~hundreds of km for GPS/BeiDou) is reproduced, while
//!   degrading smoothly to the circular case as `e -> 0`.
//!
//! # Model: `circular_secular`
//!
//! A circular orbit (eccentricity fixed at zero) whose plane precesses at a
//! constant nodal rate. The state is six mean elements plus an epoch:
//!
//! - semi-major axis `a`,
//! - inclination `i`,
//! - right ascension of the ascending node at epoch `raan0` and its rate
//!   `raan_rate`,
//! - argument of latitude at epoch `arg_lat0` and mean motion `n`.
//!
//! At an offset `dt = t - t0` the angles advance linearly,
//!
//! ```text
//! u(t)    = arg_lat0 + n * dt
//! raan(t) = raan0    + raan_rate * dt
//! e       = 0
//! ```
//!
//! and the inertial (GCRS) position is the in-plane circle rotated by the node
//! and inclination:
//!
//! ```text
//! r_orbit = a * [cos u, sin u, 0]
//! r_gcrs  = Rz(raan) * Rx(i) * r_orbit
//! ```
//!
//! # Model: `eccentric_secular`
//!
//! Adds eccentricity through a **nonsingular** `(h, k)` parameterization so
//! low-eccentricity fits stay well-conditioned and reduce exactly to the
//! circular model as `e -> 0`. The eight free elements are
//!
//! - `a`, `i`, `raan0`, `raan_rate` (as in the circular model),
//! - `h = e·sin ω`, `k = e·cos ω` (eccentricity vector components),
//! - `L0` - the mean argument of latitude at epoch (`ω + M0`),
//! - `n` - mean motion.
//!
//! Derived: `e = sqrt(h² + k²)`, `ω = atan2(h, k)`. At an offset `dt` the model
//! advances the mean argument of latitude linearly and solves Kepler's equation:
//!
//! ```text
//! λ(t) = L0 + n·dt                     # mean argument of latitude
//! M    = λ − ω                         # mean anomaly
//! E − e·sin E = M                      # Kepler (Newton)
//! ν    = atan2(sqrt(1−e²)·sin E, cos E − e)
//! r    = a·(1 − e·cos E)
//! u    = ω + ν                         # argument of latitude
//! r_gcrs = Rz(raan) · Rx(i) · [r·cos u, r·sin u, 0]
//! ```
//!
//! At `e = 0` this is identical to `circular_secular` with `arg_lat0 = L0`
//! (`u -> λ`), which is why the parameterization is nonsingular: no separate
//! `ω`/`M0` split is fitted, and conditioning stays good for near-circular
//! orbits.
//!
//! # Nodal regression seed
//!
//! `raan_rate` is **fitted**, but it is seeded from the J2 secular nodal
//! regression (Vallado, *Fundamentals of Astrodynamics and Applications*, the
//! mean-element secular rate):
//!
//! ```text
//! raan_rate_j2 = -1.5 * n * J2 * (Re / a)^2 * cos(i)
//! ```
//!
//! Both the fitted value and the J2 seed are retained on [`Elements`]
//! ([`Elements::raan_rate_rad_s`] and [`Elements::raan_rate_j2_rad_s`]); the
//! model does not claim to be a pure J2 propagation, only J2-seeded.
//!
//! # Frames and units
//!
//! Internally the fit and evaluation work in **GCRS**, kilometers, seconds.
//! Input samples are ITRF/IGS ECEF in **meters** (the SP3 convention); each is
//! converted to GCRS via the core IAU transform before fitting. Position output
//! is returned in **meters**, in ECEF by default or GCRS on request. ECEF
//! velocity includes the Earth-rotation (transport) term.

use crate::astro::constants::time::SECONDS_PER_DAY;
use crate::astro::constants::{J2_EARTH, MU_EARTH, RE_EARTH};
use crate::astro::frames::transforms::{
    gcrs_to_itrs_compute, gcrs_to_itrs_matrix, itrs_to_gcrs_compute, mat3_vec3_mul_unchecked,
    teme_to_gcrs_compute, TemeStateKm,
};
use crate::astro::math::least_squares::{
    self, solve_trf, LeastSquaresProblem, SolveOptions, Status,
};
use crate::astro::math::vec3::{cross3_ref as cross, norm3_ref as norm};
use crate::astro::sgp4::{JulianDate, Satellite};
use crate::astro::time::civil::{civil_from_julian_day_number, split_julian_date};
use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use crate::astro::time::scales::{julian_day_number, TimeScales};
use nalgebra::DVector;

use crate::constants::{M_PER_KM, OMEGA_E_DOT_RAD_S};
use crate::sp3::Sp3;
use crate::tolerances::{
    ECCENTRICITY_ZERO_EPS, REDUCED_ORBIT_KEPLER_STEP_EPS_RAD, REDUCED_ORBIT_SOLVER_TOL,
    VECTOR_NORM_ZERO_EPS,
};
use crate::validate;
use crate::GnssSatelliteId;

mod time;
use time::dt_seconds;
pub use time::CalendarEpoch;

/// Minimum number of samples the fitter accepts. The circular model solves five
/// free elements and the eccentric model eight; each ECEF sample contributes
/// three residuals, so four well-spread samples (twelve residuals) is the floor.
/// Below this the geometry seed and refinement are unreliable.
pub const MIN_SAMPLES: usize = 4;

/// Which mean-element model the fit and evaluation use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Model {
    /// Circular orbit, eccentricity fixed at zero. Best for near-circular
    /// orbits; the default.
    #[default]
    CircularSecular,
    /// Eccentric orbit via a nonsingular `(h, k)` parameterization. Reproduces
    /// the radial `a·e` signal and degrades to the circular case as `e -> 0`.
    EccentricSecular,
}

/// One fitting/drift truth sample: an epoch and an ECEF (ITRF) position in
/// meters.
#[derive(Debug, Clone, Copy)]
pub struct EcefSample {
    /// The sample epoch.
    pub epoch: CalendarEpoch,
    /// ECEF X in meters.
    pub x_m: f64,
    /// ECEF Y in meters.
    pub y_m: f64,
    /// ECEF Z in meters.
    pub z_m: f64,
}

impl EcefSample {
    /// Construct a sample from an epoch and ECEF meter components.
    pub const fn new(epoch: CalendarEpoch, x_m: f64, y_m: f64, z_m: f64) -> Self {
        Self {
            epoch,
            x_m,
            y_m,
            z_m,
        }
    }
}

/// The fitted mean elements plus the kept J2 seed.
///
/// Lengths are meters, angles radians, rates radians-or-meters per second. The
/// `raan_rate` is the fitted value; `raan_rate_j2` is the J2 nodal-regression
/// seed it started from (see the module documentation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Elements {
    /// Which model these elements belong to.
    pub model: Model,
    /// Reference epoch `t0`; all linear angle advances are measured from here.
    pub epoch: CalendarEpoch,
    /// Semi-major axis `a` in meters.
    pub a_m: f64,
    /// Eccentricity. `0.0` for the circular model; `sqrt(h² + k²)` for the
    /// eccentric model.
    pub e: f64,
    /// Inclination `i` in radians.
    pub i_rad: f64,
    /// Right ascension of the ascending node at `t0`, radians.
    pub raan_rad: f64,
    /// Fitted nodal regression rate, radians per second.
    pub raan_rate_rad_s: f64,
    /// J2 nodal-regression seed for `raan_rate`, radians per second.
    pub raan_rate_j2_rad_s: f64,
    /// Argument of latitude at `t0`, radians. For the circular model this is
    /// `arg_lat0`; for the eccentric model it is `L0 = ω + M0`, the mean
    /// argument of latitude at epoch (equal to `arg_lat0` at `e = 0`).
    pub arg_lat_rad: f64,
    /// Mean motion `n`, radians per second.
    pub mean_motion_rad_s: f64,
    /// Eccentricity vector component `h = e·sin ω`. Zero for the circular model.
    pub h: f64,
    /// Eccentricity vector component `k = e·cos ω`. Zero for the circular model.
    pub k: f64,
    /// Argument of perigee `ω = atan2(h, k)`, radians. Zero for the circular
    /// model (where it is undefined).
    pub arg_perigee_rad: f64,
}

/// Residual statistics from a fit, in meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitStats {
    /// Root-mean-square GCRS position residual over the samples, meters.
    pub rms_m: f64,
    /// Maximum GCRS position residual over the samples, meters.
    pub max_m: f64,
    /// Number of samples used in the fit.
    pub n_samples: usize,
}

/// A fitted model: elements plus the residual statistics of the fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedOrbit {
    /// The fitted mean elements.
    pub elements: Elements,
    /// Residual statistics of the fit.
    pub stats: FitStats,
}

/// Which reference frame a position/velocity result is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frame {
    /// Inertial GCRS (ECI).
    Gcrs,
    /// Earth-fixed ITRF/IGS ECEF.
    Ecef,
}

/// A per-epoch drift entry: the model-vs-truth position error at one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftEntry {
    /// The epoch evaluated.
    pub epoch: CalendarEpoch,
    /// Position error magnitude (model minus truth), meters.
    pub error_m: f64,
}

/// The result of a source-backed drift evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftReport {
    /// Per-epoch errors, in input order.
    pub per_epoch: Vec<DriftEntry>,
    /// Maximum error over the horizon, meters.
    pub max_m: f64,
    /// Root-mean-square error over the horizon, meters.
    pub rms_m: f64,
    /// The first epoch at which the error crosses the requested threshold, or
    /// `None` if it never does over the supplied horizon.
    pub threshold_horizon: Option<CalendarEpoch>,
    /// Index into [`DriftReport::per_epoch`] of the first entry whose error
    /// crosses the threshold, or `None` if it never does. This is the same
    /// crossing as [`DriftReport::threshold_horizon`] expressed as plain data, so
    /// a binding can marshal the threshold (and look the epoch up in `per_epoch`)
    /// without fabricating a placeholder epoch when there is no crossing.
    pub threshold_index: Option<usize>,
}

/// Source used by source-backed reduced-orbit fit and drift drivers.
#[derive(Debug, Clone, Copy)]
pub enum ReducedOrbitSource<'a> {
    /// Precise SP3 product plus the satellite to sample.
    Sp3 {
        product: &'a Sp3,
        satellite: GnssSatelliteId,
    },
    /// Initialized SGP4 satellite sampled in UTC.
    Sgp4 { satellite: &'a Satellite },
}

/// Sampling window and cadence for source-backed reduced-orbit drivers.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedOrbitSourceSampling {
    pub t0: CalendarEpoch,
    pub t1: CalendarEpoch,
    pub cadence_s: f64,
}

impl ReducedOrbitSourceSampling {
    pub const fn new(t0: CalendarEpoch, t1: CalendarEpoch, cadence_s: f64) -> Self {
        Self { t0, t1, cadence_s }
    }
}

/// Options for [`fit_reduced_orbit_source`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedOrbitSourceFitOptions {
    pub sampling: ReducedOrbitSourceSampling,
    pub model: Model,
}

/// Options for [`drift_reduced_orbit_source`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReducedOrbitSourceDriftOptions {
    pub sampling: ReducedOrbitSourceSampling,
    pub threshold_m: f64,
}

/// Options for [`fit_piecewise_reduced_orbit_source`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PiecewiseOrbitSourceFitOptions {
    pub sampling: ReducedOrbitSourceSampling,
    pub model: Model,
    pub segment_s: f64,
}

/// A source-backed single-model fit and its sampling metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ReducedOrbitSourceFit {
    pub orbit: ReducedOrbit,
    pub requested_samples: usize,
}

/// A source-backed single-model drift report and its sampling metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct ReducedOrbitSourceDrift {
    pub report: DriftReport,
    pub requested_samples: usize,
}

/// A source-backed piecewise fit and its sampling metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseOrbitSourceFit {
    pub orbit: PiecewiseOrbit,
    pub requested_samples: usize,
}

/// Error returned by source-backed reduced-orbit drivers.
#[derive(Debug, Clone)]
pub enum ReducedOrbitSourceError {
    InvalidWindow,
    InvalidCadence,
    InvalidSegment,
    TooFewSamples { got: usize, required: usize },
    Reduced(ReducedOrbitError),
    Piecewise(PiecewiseOrbitError),
}

impl core::fmt::Display for ReducedOrbitSourceError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidWindow => write!(f, "invalid reduced-orbit source window"),
            Self::InvalidCadence => write!(f, "invalid reduced-orbit source cadence"),
            Self::InvalidSegment => write!(f, "invalid reduced-orbit source segment"),
            Self::TooFewSamples { got, required } => {
                write!(f, "only {got} source samples; need at least {required}")
            }
            Self::Reduced(error) => write!(f, "{error}"),
            Self::Piecewise(error) => write!(f, "{error:?}"),
        }
    }
}

impl std::error::Error for ReducedOrbitSourceError {}

/// One fitted segment in a piecewise reduced-orbit model.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseSegment {
    /// Inclusive segment start.
    pub t0: CalendarEpoch,
    /// Exclusive segment end, except for the final segment where it is inclusive.
    pub t1: CalendarEpoch,
    /// Fitted reduced-orbit model for this segment.
    pub orbit: ReducedOrbit,
}

/// A long span represented by contiguous independently-fitted reduced-orbit segments.
#[derive(Debug, Clone, PartialEq)]
pub struct PiecewiseOrbit {
    /// Model fitted in every segment.
    pub model: Model,
    /// Advertised coverage start.
    pub t0: CalendarEpoch,
    /// Advertised coverage end, inclusive for the final segment.
    pub t1: CalendarEpoch,
    /// Rounded segment length used to tile the requested window, seconds.
    pub segment_s: i64,
    /// Contiguous fitted segments.
    pub segments: Vec<PiecewiseSegment>,
}

/// Errors from fitting or evaluating a reduced orbit.
#[derive(Debug, Clone)]
pub enum ReducedOrbitError {
    /// Fewer samples were supplied than the fit requires.
    TooFewSamples {
        /// The number of samples supplied.
        got: usize,
        /// The minimum number required ([`MIN_SAMPLES`]).
        required: usize,
    },
    /// The fit window is empty or inverted (`t1 <= t0`), or all samples share an
    /// epoch so no rate can be resolved.
    InvalidWindow,
    /// The samples are collinear or coincident, so the orbital plane normal is
    /// undefined and the seed cannot be built.
    SingularPlaneFit,
    /// The orbit is near-equatorial (`i ~ 0`): the ascending node, and hence
    /// `raan`, is undefined.
    RaanAmbiguous,
    /// The least-squares refinement hit a rank-deficient Jacobian.
    Singular(least_squares::SolveError),
    /// The least-squares refinement did not reach a stopping tolerance within
    /// the evaluation budget.
    FitDidNotConverge,
    /// A public evaluation input was non-finite or outside the model domain.
    InvalidInput {
        /// Input field name.
        field: &'static str,
        /// Human-readable validation failure.
        reason: &'static str,
    },
}

impl core::fmt::Display for ReducedOrbitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            ReducedOrbitError::TooFewSamples { got, required } => {
                write!(f, "only {got} samples; need at least {required}")
            }
            ReducedOrbitError::InvalidWindow => {
                write!(f, "the fit window is empty, inverted, or has no time span")
            }
            ReducedOrbitError::SingularPlaneFit => {
                write!(
                    f,
                    "samples are collinear or coincident; orbital plane undefined"
                )
            }
            ReducedOrbitError::RaanAmbiguous => {
                write!(f, "near-equatorial orbit; ascending node (raan) undefined")
            }
            ReducedOrbitError::Singular(e) => write!(f, "degenerate fit geometry: {e}"),
            ReducedOrbitError::FitDidNotConverge => {
                write!(f, "least-squares refinement did not converge")
            }
            ReducedOrbitError::InvalidInput { field, reason } => {
                write!(f, "invalid reduced-orbit input {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for ReducedOrbitError {}

impl From<least_squares::SolveError> for ReducedOrbitError {
    fn from(e: least_squares::SolveError) -> Self {
        ReducedOrbitError::Singular(e)
    }
}

/// Errors from fitting or evaluating a piecewise reduced orbit.
#[derive(Debug, Clone)]
pub enum PiecewiseOrbitError {
    /// Segment length is missing, non-positive, or rounds below one second.
    InvalidSegment,
    /// Query epoch is outside the piecewise model coverage.
    OutOfRange,
    /// A fit/drift operation did not have enough samples for the requested step.
    TooFewSamples {
        /// The number of usable samples supplied.
        got: usize,
        /// The minimum number required.
        required: usize,
    },
    /// The underlying single-segment model returned an error.
    Reduced(ReducedOrbitError),
}

impl From<ReducedOrbitError> for PiecewiseOrbitError {
    fn from(e: ReducedOrbitError) -> Self {
        PiecewiseOrbitError::Reduced(e)
    }
}

/// Near-equatorial threshold (radians) below which `raan` is treated as
/// undefined.
const MIN_INCLINATION_RAD: f64 = 1.0e-3;

/// Relative floor on the averaged plane-normal magnitude, scaled by `a_km^2`
/// (the cross product of two position vectors scales as `r^2 ~ a^2`). A normal
/// below this is treated as a vanishing cross product from collinear/coincident
/// samples, i.e. a singular plane fit.
const PLANE_NORMAL_SINGULAR_REL_EPS: f64 = 1.0e-9;

// ---------------------------------------------------------------------------
// Element evaluation (GCRS), kilometers internally.
// ---------------------------------------------------------------------------

/// Free parameters in fit order: `[a_km, i, raan0, raan_rate, arg_lat0, n]`.
/// Eccentricity is held at zero and is not a free parameter.
const N_PARAMS: usize = 6;

/// Map a normalized fit vector back to physical parameters
/// `[a_km, i, raan0, raan_rate, arg_lat0, n]`. `a` is carried in units of the
/// seed semi-major axis and the two rates as the angle swept across the window
/// (`rate * window`), so the solver sees comparably scaled columns.
fn unscale_params(x: &[f64], a_scale: f64, rate_scale: f64) -> [f64; N_PARAMS] {
    [
        x[0] * a_scale,
        x[1],
        x[2],
        x[3] / rate_scale,
        x[4],
        x[5] / rate_scale,
    ]
}

/// Free parameters of the eccentric model, in fit order:
/// `[a_km, i, raan0, raan_rate, h, k, L0, n]`.
const N_PARAMS_ECC: usize = 8;

/// Map a normalized eccentric fit vector back to physical parameters
/// `[a_km, i, raan0, raan_rate, h, k, L0, n]`. The same normalization as the
/// circular model: `a` in units of the seed axis, the two rates as swept angle.
/// `h` and `k` are unscaled - at unit magnitude they perturb the position at the
/// `~a` kilometre level, comparable to the angle columns.
fn unscale_params_ecc(x: &[f64], a_scale: f64, rate_scale: f64) -> [f64; N_PARAMS_ECC] {
    [
        x[0] * a_scale,
        x[1],
        x[2],
        x[3] / rate_scale,
        x[4],
        x[5],
        x[6],
        x[7] / rate_scale,
    ]
}

/// Solve Kepler's equation `E − e·sin E = M` for the eccentric anomaly `E`
/// (radians) by Newton's method. `e < 1`. `M` is wrapped to `[0, 2π)` for
/// conditioning; the GNSS eccentricities here (`e ~ 0.01-0.17`) converge in a
/// few iterations.
fn solve_kepler(m: f64, e: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    let m = m.rem_euclid(two_pi);
    let mut ee = if e < 0.8 { m } else { std::f64::consts::PI };
    for _ in 0..30 {
        let f = ee - e * ee.sin() - m;
        let fp = 1.0 - e * ee.cos();
        let d = f / fp;
        ee -= d;
        if d.abs() < REDUCED_ORBIT_KEPLER_STEP_EPS_RAD {
            break;
        }
    }
    ee
}

/// Evaluate the GCRS position (kilometers) of the `eccentric_secular` model from
/// a parameter vector `[a_km, i, raan0, raan_rate, h, k, L0, n]` at offset `dt`.
fn eval_gcrs_km_ecc(p: &[f64], dt: f64) -> [f64; 3] {
    let a = p[0];
    let i = p[1];
    let raan = p[2] + p[3] * dt;
    let h = p[4];
    let k = p[5];
    let lambda = p[6] + p[7] * dt;

    let e = (h * h + k * k).sqrt();
    if e < ECCENTRICITY_ZERO_EPS {
        // Circular fast path: u -> lambda, exactly the circular model.
        return eval_gcrs_km(&[a, i, p[2], p[3], p[6], p[7]], dt);
    }
    let omega = h.atan2(k);
    let mm = lambda - omega;
    let big_e = solve_kepler(mm, e);
    let (se, ce) = big_e.sin_cos();
    let r = a * (1.0 - e * ce);
    let nu = (((1.0 - e * e).sqrt()) * se).atan2(ce - e);
    let u = omega + nu;

    rotate_in_plane_km([r * u.cos(), r * u.sin()], i, raan)
}

/// Rotate an in-plane (node-aligned) 2-vector `[x, y]` km by `Rx(i)` then
/// `Rz(raan)` into GCRS, matching the circular model's outer rotation.
fn rotate_in_plane_km(xy: [f64; 2], i: f64, raan: f64) -> [f64; 3] {
    let (si, ci) = i.sin_cos();
    let (sr, cr) = raan.sin_cos();
    let x1 = xy[0];
    let y1 = xy[1] * ci;
    let z1 = xy[1] * si;
    [cr * x1 - sr * y1, sr * x1 + cr * y1, z1]
}

/// Analytic GCRS velocity (km/s) of the `eccentric_secular` model at offset `dt`.
/// Derivative of [`eval_gcrs_km_ecc`] with respect to time (`ω`, `e` constant,
/// `dM/dt = n`, `dRaan/dt = raan_rate`).
fn eval_gcrs_velocity_km_s_ecc(p: &[f64], dt: f64) -> [f64; 3] {
    let a = p[0];
    let i = p[1];
    let raan_rate = p[3];
    let h = p[4];
    let k = p[5];
    let n = p[7];
    let raan = p[2] + raan_rate * dt;
    let lambda = p[6] + n * dt;

    let e = (h * h + k * k).sqrt();
    if e < ECCENTRICITY_ZERO_EPS {
        return eval_gcrs_velocity_km_s(&[a, i, p[2], p[3], p[6], p[7]], dt);
    }
    let omega = h.atan2(k);
    let mm = lambda - omega;
    let big_e = solve_kepler(mm, e);
    let (se, ce) = big_e.sin_cos();
    // dE/dt from differentiating Kepler: (1 - e cos E) Edot = dM/dt = n.
    let edot = n / (1.0 - e * ce);
    let beta = (1.0 - e * e).sqrt();

    // Perifocal position/velocity (x toward perigee).
    let x_pf = a * (ce - e);
    let y_pf = a * beta * se;
    let xdot_pf = -a * se * edot;
    let ydot_pf = a * beta * ce * edot;

    // Rotate the perifocal frame into the node-aligned in-plane frame by ω.
    let (so, co) = omega.sin_cos();
    let x1 = co * x_pf - so * y_pf;
    let y1 = so * x_pf + co * y_pf;
    let dx1 = co * xdot_pf - so * ydot_pf;
    let dy1 = so * xdot_pf + co * ydot_pf;

    // Apply Rx(i) then Rz(raan) with the raan_rate coupling, exactly as the
    // circular velocity does (only the in-plane inputs differ).
    let (si, ci) = i.sin_cos();
    let (sr, cr) = raan.sin_cos();
    let y1i = y1 * ci;
    let dy1i = dy1 * ci;

    let vx = cr * dx1 - sr * dy1i + raan_rate * (-sr * x1 - cr * y1i);
    let vy = sr * dx1 + cr * dy1i + raan_rate * (cr * x1 - sr * y1i);
    let vz = dy1 * si;
    [vx, vy, vz]
}

/// Evaluate the GCRS position (kilometers) of the `circular_secular` model from
/// a parameter vector at offset `dt` seconds from epoch.
fn eval_gcrs_km(p: &[f64], dt: f64) -> [f64; 3] {
    let a = p[0];
    let i = p[1];
    let raan = p[2] + p[3] * dt;
    let u = p[4] + p[5] * dt;

    // In-plane circle.
    let (su, cu) = u.sin_cos();
    let xp = a * cu;
    let yp = a * su;

    // Rotate by inclination about x, then by raan about z.
    let (si, ci) = i.sin_cos();
    let (sr, cr) = raan.sin_cos();

    // Rx(i) * [xp, yp, 0] = [xp, yp*ci, yp*si]
    let x1 = xp;
    let y1 = yp * ci;
    let z1 = yp * si;

    // Rz(raan) * [x1, y1, z1]
    [cr * x1 - sr * y1, sr * x1 + cr * y1, z1]
}

/// Analytic GCRS velocity (km/s) of the model at offset `dt`. Derivative of
/// [`eval_gcrs_km`] with respect to time.
fn eval_gcrs_velocity_km_s(p: &[f64], dt: f64) -> [f64; 3] {
    let a = p[0];
    let i = p[1];
    let raan_rate = p[3];
    let n = p[5];
    let raan = p[2] + raan_rate * dt;
    let u = p[4] + n * dt;

    let (su, cu) = u.sin_cos();
    let (si, ci) = i.sin_cos();
    let (sr, cr) = raan.sin_cos();

    let xp = a * cu;
    let yp = a * su;
    // d/dt of in-plane position.
    let dxp = -a * su * n;
    let dyp = a * cu * n;

    // Position in the inclined-but-not-yet-noded frame.
    let x1 = xp;
    let y1 = yp * ci;
    // Its time derivative (i constant).
    let dx1 = dxp;
    let dy1 = dyp * ci;

    // r_gcrs = Rz(raan) * [x1, y1, z1]; differentiate including dRaan/dt.
    // d/dt(cr) = -sr*raan_rate, d/dt(sr) = cr*raan_rate.
    let vx = cr * dx1 - sr * dy1 + raan_rate * (-sr * x1 - cr * y1);
    let vy = sr * dx1 + cr * dy1 + raan_rate * (cr * x1 - sr * y1);
    let vz = dyp * si;
    [vx, vy, vz]
}

// ---------------------------------------------------------------------------
// Seed extraction.
// ---------------------------------------------------------------------------

struct GcrsSample {
    dt: f64,
    r_km: [f64; 3],
}

/// Build the seed parameter vector from GCRS samples.
fn seed_params(samples: &[GcrsSample]) -> Result<[f64; N_PARAMS], ReducedOrbitError> {
    // Mean radius -> a.
    let a_km = samples.iter().map(|s| norm(&s.r_km)).sum::<f64>() / samples.len() as f64;
    if !a_km.is_finite() || a_km <= 0.0 {
        return Err(ReducedOrbitError::SingularPlaneFit);
    }

    // Averaged plane normal from consecutive cross products.
    let mut h = [0.0_f64; 3];
    for w in samples.windows(2) {
        let c = cross(&w[0].r_km, &w[1].r_km);
        h[0] += c[0];
        h[1] += c[1];
        h[2] += c[2];
    }
    let hn = norm(&h);
    if !hn.is_finite() || hn <= a_km * a_km * PLANE_NORMAL_SINGULAR_REL_EPS {
        // Cross products vanish: collinear/coincident samples.
        return Err(ReducedOrbitError::SingularPlaneFit);
    }
    let hhat = [h[0] / hn, h[1] / hn, h[2] / hn];

    // Inclination from the normal's z component.
    let i = hhat[2].clamp(-1.0, 1.0).acos();
    if i < MIN_INCLINATION_RAD || (std::f64::consts::PI - i) < MIN_INCLINATION_RAD {
        return Err(ReducedOrbitError::RaanAmbiguous);
    }

    // RAAN from the node direction n = z_hat x h_hat (points to ascending node).
    let node = [-hhat[1], hhat[0], 0.0];
    let node_n = norm(&node);
    if node_n <= VECTOR_NORM_ZERO_EPS {
        return Err(ReducedOrbitError::RaanAmbiguous);
    }
    let raan0 = node[1].atan2(node[0]);
    let nhat = [node[0] / node_n, node[1] / node_n, 0.0];

    // Argument of latitude of the first sample: angle from the node, measured in
    // the orbit plane (positive toward h x n direction of motion).
    let r0 = &samples[0].r_km;
    let cos_u = (r0[0] * nhat[0] + r0[1] * nhat[1] + r0[2] * nhat[2]) / norm(r0);
    // In-plane "perpendicular to node" axis: h_hat x n_hat.
    let p_axis = cross(&hhat, &nhat);
    let sin_u = (r0[0] * p_axis[0] + r0[1] * p_axis[1] + r0[2] * p_axis[2]) / norm(r0);
    let arg_lat0 = sin_u.atan2(cos_u);

    // Mean motion from the Keplerian relation at the fitted a (km, MU in km^3/s^2).
    let n = (MU_EARTH / (a_km * a_km * a_km)).sqrt();

    // J2 nodal regression seed.
    let raan_rate = raan_rate_j2(n, i, a_km);

    Ok([a_km, i, raan0, raan_rate, arg_lat0, n])
}

/// J2 secular nodal-regression rate (Vallado), radians per second.
/// `a` in kilometers (matching `RE_EARTH`).
fn raan_rate_j2(n: f64, i: f64, a_km: f64) -> f64 {
    let re_over_a = RE_EARTH / a_km;
    -1.5 * n * J2_EARTH * re_over_a * re_over_a * i.cos()
}

// ---------------------------------------------------------------------------
// Fit.
// ---------------------------------------------------------------------------

/// Fit the default `circular_secular` model to ECEF samples, with all epochs
/// interpreted in `scale` (e.g. an SP3 product's GPST).
///
/// `samples` are `(epoch, ECEF meters)`; they are ordered by time, the earliest
/// becomes the model epoch `t0`, each is converted ITRS->GCRS at the correct
/// Earth orientation, and the five free elements (`a`, `i`, `raan0`, `raan_rate`,
/// `arg_lat0`, `n`) are refined to minimize stacked GCRS position residuals.
///
/// Equivalent to [`fit_with_model`] with [`Model::CircularSecular`].
pub fn fit(samples: &[EcefSample], scale: TimeScale) -> Result<ReducedOrbit, ReducedOrbitError> {
    fit_with_model(samples, scale, Model::CircularSecular)
}

/// Fit a chosen [`Model`] to ECEF samples interpreted in `scale`.
///
/// The seed (mean axis, plane normal, node, mean motion, J2 nodal rate) and the
/// normalized Levenberg-Marquardt refinement are shared by both models; the
/// eccentric model adds the `h`, `k` eccentricity-vector columns (seeded at
/// zero) and solves Kepler's equation per residual.
pub fn fit_with_model(
    samples: &[EcefSample],
    scale: TimeScale,
    model: Model,
) -> Result<ReducedOrbit, ReducedOrbitError> {
    match model {
        Model::CircularSecular => fit_circular(samples, scale),
        Model::EccentricSecular => fit_eccentric(samples, scale),
    }
}

fn fit_circular(
    samples: &[EcefSample],
    scale: TimeScale,
) -> Result<ReducedOrbit, ReducedOrbitError> {
    if samples.len() < MIN_SAMPLES {
        return Err(ReducedOrbitError::TooFewSamples {
            got: samples.len(),
            required: MIN_SAMPLES,
        });
    }
    validate_fit_epochs(samples, scale)?;

    // Order by absolute time so the seed, t0, and consecutive-pair plane fit do
    // not depend on the caller's sample order.
    let mut ordered: Vec<(TimeScales, &EcefSample)> = samples
        .iter()
        .map(|s| (s.epoch.time_scales(scale), s))
        .collect();
    ordered.sort_by(|a, b| {
        (a.0.jd_whole + a.0.tt_fraction).total_cmp(&(b.0.jd_whole + b.0.tt_fraction))
    });

    let t0_cal = ordered[0].1.epoch;
    let t0_ts = ordered[0].0;

    // Convert every ECEF sample to GCRS km at its own epoch.
    let mut gcrs: Vec<GcrsSample> = Vec::with_capacity(samples.len());
    for (ts, s) in &ordered {
        let (x, y, z) =
            itrs_to_gcrs_compute(s.x_m / M_PER_KM, s.y_m / M_PER_KM, s.z_m / M_PER_KM, ts)
                .expect("valid frame transform");
        let dt = dt_seconds(&t0_ts, ts);
        gcrs.push(GcrsSample {
            dt,
            r_km: [x, y, z],
        });
    }

    // Window must span time (also rejects a non-finite span).
    let dt_span = gcrs.iter().map(|s| s.dt).fold(f64::NEG_INFINITY, f64::max)
        - gcrs.iter().map(|s| s.dt).fold(f64::INFINITY, f64::min);
    if !dt_span.is_finite() || dt_span <= 0.0 {
        return Err(ReducedOrbitError::InvalidWindow);
    }

    let seed = seed_params(&gcrs)?;

    // The physical parameters span ~10 orders of magnitude (a ~ 2.7e4 km down to
    // the nodal rate ~ 1e-8 rad/s), and the core solver damps with a uniform
    // mu*I and no per-variable scaling. Left unscaled, the nodal rate is
    // effectively invisible to the solver and stays at its seed. Fit instead in a
    // normalized space where every parameter perturbs the position at the ~a
    // kilometre level: a is measured in units of the seed semi-major axis, and
    // the two rates are carried as the total angle they sweep across the window.
    let a_scale = seed[0];
    let rate_scale = dt_span; // raan_rate and n fit as (rate * window) = swept angle
    let seed_scaled = [
        seed[0] / a_scale,
        seed[1],
        seed[2],
        seed[3] * rate_scale,
        seed[4],
        seed[5] * rate_scale,
    ];

    // Stacked GCRS position residuals (km), one xyz block per sample.
    let m = 3 * gcrs.len();
    let residual = {
        let gcrs_ref: Vec<(f64, [f64; 3])> = gcrs.iter().map(|s| (s.dt, s.r_km)).collect();
        move |x: &DVector<f64>| -> DVector<f64> {
            let xs = x.as_slice();
            let p = unscale_params(xs, a_scale, rate_scale);
            let mut r = Vec::with_capacity(m);
            for (dt, obs) in &gcrs_ref {
                let model = eval_gcrs_km(&p, *dt);
                r.push(model[0] - obs[0]);
                r.push(model[1] - obs[1]);
                r.push(model[2] - obs[2]);
            }
            DVector::from_vec(r)
        }
    };

    let x0 = DVector::from_row_slice(&seed_scaled);
    let problem = LeastSquaresProblem::new(residual, x0);
    let opts = SolveOptions {
        gtol: REDUCED_ORBIT_SOLVER_TOL,
        ftol: REDUCED_ORBIT_SOLVER_TOL,
        xtol: REDUCED_ORBIT_SOLVER_TOL,
        max_nfev: 400,
    };
    let report = solve_trf(&problem, &opts)?;

    let converged = matches!(
        report.status,
        Status::GradientTolerance | Status::CostTolerance | Status::StepTolerance
    );
    if !converged {
        return Err(ReducedOrbitError::FitDidNotConverge);
    }

    let p = unscale_params(report.x.as_slice(), a_scale, rate_scale);
    // Residual stats in meters.
    let res = &report.residual;
    let n_samp = gcrs.len();
    let mut sumsq = 0.0;
    let mut maxsq = 0.0_f64;
    for k in 0..n_samp {
        let dx = res[3 * k] * M_PER_KM;
        let dy = res[3 * k + 1] * M_PER_KM;
        let dz = res[3 * k + 2] * M_PER_KM;
        let e2 = dx * dx + dy * dy + dz * dz;
        sumsq += e2;
        if e2 > maxsq {
            maxsq = e2;
        }
    }
    let rms_m = (sumsq / n_samp as f64).sqrt();
    let max_m = maxsq.sqrt();

    let elements = Elements {
        model: Model::CircularSecular,
        epoch: t0_cal,
        a_m: p[0] * M_PER_KM,
        e: 0.0,
        i_rad: p[1],
        raan_rad: p[2],
        raan_rate_rad_s: p[3],
        raan_rate_j2_rad_s: raan_rate_j2(p[5], p[1], p[0]),
        arg_lat_rad: p[4],
        mean_motion_rad_s: p[5],
        h: 0.0,
        k: 0.0,
        arg_perigee_rad: 0.0,
    };

    Ok(ReducedOrbit {
        elements,
        stats: FitStats {
            rms_m,
            max_m,
            n_samples: n_samp,
        },
    })
}

/// Convert ordered ECEF samples to GCRS-km samples and the model epoch.
fn to_gcrs_samples(
    samples: &[EcefSample],
    scale: TimeScale,
) -> Result<(CalendarEpoch, Vec<GcrsSample>, f64), ReducedOrbitError> {
    if samples.len() < MIN_SAMPLES {
        return Err(ReducedOrbitError::TooFewSamples {
            got: samples.len(),
            required: MIN_SAMPLES,
        });
    }
    validate_fit_epochs(samples, scale)?;

    let mut ordered: Vec<(TimeScales, &EcefSample)> = samples
        .iter()
        .map(|s| (s.epoch.time_scales(scale), s))
        .collect();
    ordered.sort_by(|a, b| {
        (a.0.jd_whole + a.0.tt_fraction).total_cmp(&(b.0.jd_whole + b.0.tt_fraction))
    });

    let t0_cal = ordered[0].1.epoch;
    let t0_ts = ordered[0].0;

    let mut gcrs: Vec<GcrsSample> = Vec::with_capacity(samples.len());
    for (ts, s) in &ordered {
        let (x, y, z) =
            itrs_to_gcrs_compute(s.x_m / M_PER_KM, s.y_m / M_PER_KM, s.z_m / M_PER_KM, ts)
                .expect("valid frame transform");
        let dt = dt_seconds(&t0_ts, ts);
        gcrs.push(GcrsSample {
            dt,
            r_km: [x, y, z],
        });
    }

    let dt_span = gcrs.iter().map(|s| s.dt).fold(f64::NEG_INFINITY, f64::max)
        - gcrs.iter().map(|s| s.dt).fold(f64::INFINITY, f64::min);
    if !dt_span.is_finite() || dt_span <= 0.0 {
        return Err(ReducedOrbitError::InvalidWindow);
    }

    Ok((t0_cal, gcrs, dt_span))
}

/// Fit the `eccentric_secular` model. Reuses the circular seed and the same
/// normalized LM, adding the `h`, `k` columns (seeded at zero) and solving
/// Kepler's equation per residual.
fn fit_eccentric(
    samples: &[EcefSample],
    scale: TimeScale,
) -> Result<ReducedOrbit, ReducedOrbitError> {
    let (t0_cal, gcrs, dt_span) = to_gcrs_samples(samples, scale)?;

    // The circular seed supplies a, i, raan0, raan_rate, arg_lat0 (=L0), n.
    let seed_c = seed_params(&gcrs)?;
    let a_scale = seed_c[0];
    let rate_scale = dt_span;

    // Seed e = 0 (h = k = 0); L0 from the first sample's argument of latitude.
    // Normalized seed: a in seed-axis units, rates as swept angle, h/k unscaled.
    let seed_scaled = [
        1.0,                    // a / a_scale
        seed_c[1],              // i
        seed_c[2],              // raan0
        seed_c[3] * rate_scale, // raan_rate (swept angle)
        0.0,                    // h
        0.0,                    // k
        seed_c[4],              // L0 = arg_lat0
        seed_c[5] * rate_scale, // n (swept angle)
    ];

    let m = 3 * gcrs.len();
    let residual = {
        let gcrs_ref: Vec<(f64, [f64; 3])> = gcrs.iter().map(|s| (s.dt, s.r_km)).collect();
        move |x: &DVector<f64>| -> DVector<f64> {
            let xs = x.as_slice();
            let p = unscale_params_ecc(xs, a_scale, rate_scale);
            let mut r = Vec::with_capacity(m);
            for (dt, obs) in &gcrs_ref {
                let model = eval_gcrs_km_ecc(&p, *dt);
                r.push(model[0] - obs[0]);
                r.push(model[1] - obs[1]);
                r.push(model[2] - obs[2]);
            }
            DVector::from_vec(r)
        }
    };

    let x0 = DVector::from_row_slice(&seed_scaled);
    let problem = LeastSquaresProblem::new(residual, x0);
    let opts = SolveOptions {
        gtol: REDUCED_ORBIT_SOLVER_TOL,
        ftol: REDUCED_ORBIT_SOLVER_TOL,
        xtol: REDUCED_ORBIT_SOLVER_TOL,
        max_nfev: 400,
    };
    let report = solve_trf(&problem, &opts)?;

    let converged = matches!(
        report.status,
        Status::GradientTolerance | Status::CostTolerance | Status::StepTolerance
    );
    if !converged {
        return Err(ReducedOrbitError::FitDidNotConverge);
    }

    let p = unscale_params_ecc(report.x.as_slice(), a_scale, rate_scale);
    let res = &report.residual;
    let n_samp = gcrs.len();
    let mut sumsq = 0.0;
    let mut maxsq = 0.0_f64;
    for k in 0..n_samp {
        let dx = res[3 * k] * M_PER_KM;
        let dy = res[3 * k + 1] * M_PER_KM;
        let dz = res[3 * k + 2] * M_PER_KM;
        let e2 = dx * dx + dy * dy + dz * dz;
        sumsq += e2;
        if e2 > maxsq {
            maxsq = e2;
        }
    }
    let rms_m = (sumsq / n_samp as f64).sqrt();
    let max_m = maxsq.sqrt();

    let h = p[4];
    let k = p[5];
    let e = (h * h + k * k).sqrt();
    let arg_perigee_rad = if e < ECCENTRICITY_ZERO_EPS {
        0.0
    } else {
        h.atan2(k)
    };

    let elements = Elements {
        model: Model::EccentricSecular,
        epoch: t0_cal,
        a_m: p[0] * M_PER_KM,
        e,
        i_rad: p[1],
        raan_rad: p[2],
        raan_rate_rad_s: p[3],
        raan_rate_j2_rad_s: raan_rate_j2(p[7], p[1], p[0]),
        arg_lat_rad: p[6],
        mean_motion_rad_s: p[7],
        h,
        k,
        arg_perigee_rad,
    };

    Ok(ReducedOrbit {
        elements,
        stats: FitStats {
            rms_m,
            max_m,
            n_samples: n_samp,
        },
    })
}

// ---------------------------------------------------------------------------
// Evaluation.
// ---------------------------------------------------------------------------

fn params_from_elements(e: &Elements) -> [f64; N_PARAMS] {
    [
        e.a_m / M_PER_KM,
        e.i_rad,
        e.raan_rad,
        e.raan_rate_rad_s,
        e.arg_lat_rad,
        e.mean_motion_rad_s,
    ]
}

fn params_from_elements_ecc(e: &Elements) -> [f64; N_PARAMS_ECC] {
    [
        e.a_m / M_PER_KM,
        e.i_rad,
        e.raan_rad,
        e.raan_rate_rad_s,
        e.h,
        e.k,
        e.arg_lat_rad,
        e.mean_motion_rad_s,
    ]
}

/// GCRS position (km) of `elements` at offset `dt`, dispatching on the model.
fn eval_position_km(elements: &Elements, dt: f64) -> [f64; 3] {
    match elements.model {
        Model::CircularSecular => eval_gcrs_km(&params_from_elements(elements), dt),
        Model::EccentricSecular => eval_gcrs_km_ecc(&params_from_elements_ecc(elements), dt),
    }
}

/// GCRS velocity (km/s) of `elements` at offset `dt`, dispatching on the model.
fn eval_velocity_km_s(elements: &Elements, dt: f64) -> [f64; 3] {
    match elements.model {
        Model::CircularSecular => eval_gcrs_velocity_km_s(&params_from_elements(elements), dt),
        Model::EccentricSecular => {
            eval_gcrs_velocity_km_s_ecc(&params_from_elements_ecc(elements), dt)
        }
    }
}

/// Evaluate the model position at `epoch` (interpreted in `scale`) in the
/// requested frame, meters.
pub fn position(
    elements: &Elements,
    epoch: CalendarEpoch,
    scale: TimeScale,
    frame: Frame,
) -> Result<[f64; 3], ReducedOrbitError> {
    validate_elements_for_evaluation(elements, scale)?;
    validate_calendar_epoch(epoch, scale, "epoch")?;
    let t0_ts = elements.epoch.time_scales(scale);
    let ts = epoch.time_scales(scale);
    let dt = dt_seconds(&t0_ts, &ts);
    validate_finite(dt, "dt_s")?;
    let r_gcrs_km = eval_position_km(elements, dt);
    let r = match frame {
        Frame::Gcrs => [
            r_gcrs_km[0] * M_PER_KM,
            r_gcrs_km[1] * M_PER_KM,
            r_gcrs_km[2] * M_PER_KM,
        ],
        Frame::Ecef => {
            let mat = gcrs_to_itrs_matrix(&ts)
                .map_err(|_| invalid_input("epoch", "invalid frame transform"))?;
            let r = mat3_vec3_mul_unchecked(&mat, &r_gcrs_km);
            [r[0] * M_PER_KM, r[1] * M_PER_KM, r[2] * M_PER_KM]
        }
    };
    validate_vec3(r, "position_m")?;
    Ok(r)
}

/// Evaluate the model position and velocity at `epoch` in the requested frame.
/// Returns `(position_m, velocity_m_s)`. ECEF velocity includes the
/// Earth-rotation transport term.
pub fn position_velocity(
    elements: &Elements,
    epoch: CalendarEpoch,
    scale: TimeScale,
    frame: Frame,
) -> Result<([f64; 3], [f64; 3]), ReducedOrbitError> {
    validate_elements_for_evaluation(elements, scale)?;
    validate_calendar_epoch(epoch, scale, "epoch")?;
    let t0_ts = elements.epoch.time_scales(scale);
    let ts = epoch.time_scales(scale);
    let dt = dt_seconds(&t0_ts, &ts);
    validate_finite(dt, "dt_s")?;
    let r_gcrs_km = eval_position_km(elements, dt);
    let v_gcrs_km_s = eval_velocity_km_s(elements, dt);

    let (r, v) = match frame {
        Frame::Gcrs => {
            let r = [
                r_gcrs_km[0] * M_PER_KM,
                r_gcrs_km[1] * M_PER_KM,
                r_gcrs_km[2] * M_PER_KM,
            ];
            let v = [
                v_gcrs_km_s[0] * M_PER_KM,
                v_gcrs_km_s[1] * M_PER_KM,
                v_gcrs_km_s[2] * M_PER_KM,
            ];
            (r, v)
        }
        Frame::Ecef => {
            let mat = gcrs_to_itrs_matrix(&ts)
                .map_err(|_| invalid_input("epoch", "invalid frame transform"))?;
            let r_itrs_km = mat3_vec3_mul_unchecked(&mat, &r_gcrs_km);
            let v_rot_km_s = mat3_vec3_mul_unchecked(&mat, &v_gcrs_km_s);
            // Transport term: v_itrs = R v_gcrs - omega x r_itrs.
            let vx = v_rot_km_s[0] + OMEGA_E_DOT_RAD_S * r_itrs_km[1];
            let vy = v_rot_km_s[1] - OMEGA_E_DOT_RAD_S * r_itrs_km[0];
            let vz = v_rot_km_s[2];
            let r = [
                r_itrs_km[0] * M_PER_KM,
                r_itrs_km[1] * M_PER_KM,
                r_itrs_km[2] * M_PER_KM,
            ];
            let v = [vx * M_PER_KM, vy * M_PER_KM, vz * M_PER_KM];
            (r, v)
        }
    };
    validate_vec3(r, "position_m")?;
    validate_vec3(v, "velocity_m_s")?;
    Ok((r, v))
}

fn validate_elements_for_evaluation(
    elements: &Elements,
    scale: TimeScale,
) -> Result<(), ReducedOrbitError> {
    validate_calendar_epoch(elements.epoch, scale, "elements.epoch")?;
    validate_positive(elements.a_m, "elements.a_m")?;
    validate_finite(elements.e, "elements.e")?;
    if !(0.0..1.0).contains(&elements.e) {
        return Err(invalid_input("elements.e", "must be in [0, 1)"));
    }
    validate_finite(elements.i_rad, "elements.i_rad")?;
    if !(0.0..=std::f64::consts::PI).contains(&elements.i_rad) {
        return Err(invalid_input("elements.i_rad", "must be in [0, pi]"));
    }
    validate_finite(elements.raan_rad, "elements.raan_rad")?;
    validate_finite(elements.raan_rate_rad_s, "elements.raan_rate_rad_s")?;
    validate_finite(elements.raan_rate_j2_rad_s, "elements.raan_rate_j2_rad_s")?;
    validate_finite(elements.arg_lat_rad, "elements.arg_lat_rad")?;
    validate_positive(elements.mean_motion_rad_s, "elements.mean_motion_rad_s")?;
    validate_finite(elements.h, "elements.h")?;
    validate_finite(elements.k, "elements.k")?;
    validate_finite(elements.arg_perigee_rad, "elements.arg_perigee_rad")?;
    if elements.model == Model::EccentricSecular {
        let derived_e = (elements.h * elements.h + elements.k * elements.k).sqrt();
        validate_finite(derived_e, "elements.h_k")?;
        if derived_e >= 1.0 {
            return Err(invalid_input("elements.h_k", "eccentricity must be < 1"));
        }
    }
    Ok(())
}

fn validate_calendar_epoch(
    epoch: CalendarEpoch,
    scale: TimeScale,
    field: &'static str,
) -> Result<(), ReducedOrbitError> {
    let second_policy = match scale {
        TimeScale::Utc => validate::CivilSecondPolicy::UtcLike,
        // GLONASST is UTC(SU)-based, but a civil GLONASST leap-second (:60) label
        // is not a supported civil input here: no time-system label parses to
        // GLONASST (RINEX/SP3 "GLO" is UTC), and GLONASST is reached numerically
        // via `timescale_offset_at_s`. Treat it as Continuous so a stray :60
        // GLONASST label is rejected rather than silently shifted by the -3h
        // conversion into the wrong second; ordinary GLONASST civil times still
        // convert via `scale_calendar_to_utc`.
        TimeScale::Glonasst
        | TimeScale::Tai
        | TimeScale::Tt
        | TimeScale::Tdb
        | TimeScale::Gpst
        | TimeScale::Gst
        | TimeScale::Bdt
        | TimeScale::Qzsst => validate::CivilSecondPolicy::Continuous,
    };
    validate::civil_datetime_with_second_policy(
        i64::from(epoch.year),
        i64::from(epoch.month),
        i64::from(epoch.day),
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        epoch.second,
        second_policy,
    )
    .map(|_| ())
    .map_err(|_| invalid_input(field, "invalid calendar epoch"))
}

fn validate_fit_epochs(samples: &[EcefSample], scale: TimeScale) -> Result<(), ReducedOrbitError> {
    for sample in samples {
        validate_calendar_epoch(sample.epoch, scale, "epoch")?;
        validate_finite(sample.x_m, "sample.x_m")?;
        validate_finite(sample.y_m, "sample.y_m")?;
        validate_finite(sample.z_m, "sample.z_m")?;
    }
    Ok(())
}

fn validate_truth_sample(sample: &EcefSample, scale: TimeScale) -> Result<(), ReducedOrbitError> {
    validate_calendar_epoch(sample.epoch, scale, "truth.epoch")?;
    validate_finite(sample.x_m, "truth.x_m")?;
    validate_finite(sample.y_m, "truth.y_m")?;
    validate_finite(sample.z_m, "truth.z_m")
}

fn validate_vec3(value: [f64; 3], field: &'static str) -> Result<(), ReducedOrbitError> {
    for component in value {
        validate_finite(component, field)?;
    }
    Ok(())
}

fn validate_finite(value: f64, field: &'static str) -> Result<(), ReducedOrbitError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

fn validate_positive(value: f64, field: &'static str) -> Result<(), ReducedOrbitError> {
    validate_finite(value, field)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be positive"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> ReducedOrbitError {
    ReducedOrbitError::InvalidInput { field, reason }
}

// ---------------------------------------------------------------------------
// Drift.
// ---------------------------------------------------------------------------

/// Evaluate the model against truth ECEF samples and report the per-epoch error,
/// its statistics, and the first epoch the error crosses `threshold_m`.
///
/// The error is computed in ECEF meters (model ECEF minus truth ECEF). This is
/// source-backed: the caller supplies the truth `(epoch, ECEF)` samples; the
/// function does not compare the model against itself.
pub fn drift(
    elements: &Elements,
    truth: &[EcefSample],
    scale: TimeScale,
    threshold_m: f64,
) -> Result<DriftReport, ReducedOrbitError> {
    validate_elements_for_evaluation(elements, scale)?;
    validate_finite(threshold_m, "threshold_m")?;
    let mut per_epoch = Vec::with_capacity(truth.len());
    let mut sumsq = 0.0;
    let mut max_m = 0.0_f64;
    let mut threshold_horizon = None;
    let mut threshold_index = None;

    for s in truth {
        validate_truth_sample(s, scale)?;
        let model = position(elements, s.epoch, scale, Frame::Ecef)?;
        let dx = model[0] - s.x_m;
        let dy = model[1] - s.y_m;
        let dz = model[2] - s.z_m;
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        validate_finite(err, "drift.error_m")?;
        sumsq += err * err;
        validate_finite(sumsq, "drift.sumsq")?;
        if err > max_m {
            max_m = err;
        }
        if threshold_horizon.is_none() && err > threshold_m {
            threshold_horizon = Some(s.epoch);
            threshold_index = Some(per_epoch.len());
        }
        per_epoch.push(DriftEntry {
            epoch: s.epoch,
            error_m: err,
        });
    }

    let rms_m = if per_epoch.is_empty() {
        0.0
    } else {
        (sumsq / per_epoch.len() as f64).sqrt()
    };
    validate_finite(max_m, "drift.max_m")?;
    validate_finite(rms_m, "drift.rms_m")?;

    Ok(DriftReport {
        per_epoch,
        max_m,
        rms_m,
        threshold_horizon,
        threshold_index,
    })
}

/// Sample an SP3 or SGP4 source and fit a reduced orbit.
pub fn fit_reduced_orbit_source(
    source: ReducedOrbitSource<'_>,
    options: ReducedOrbitSourceFitOptions,
) -> Result<ReducedOrbitSourceFit, ReducedOrbitSourceError> {
    let sampled = sample_reduced_orbit_source(source, options.sampling)?;
    let orbit = fit_with_model(&sampled.samples, sampled.scale, options.model)
        .map_err(ReducedOrbitSourceError::Reduced)?;
    Ok(ReducedOrbitSourceFit {
        orbit,
        requested_samples: sampled.requested,
    })
}

/// Sample an SP3 or SGP4 source and evaluate drift against those truth samples.
pub fn drift_reduced_orbit_source(
    elements: &Elements,
    source: ReducedOrbitSource<'_>,
    options: ReducedOrbitSourceDriftOptions,
) -> Result<ReducedOrbitSourceDrift, ReducedOrbitSourceError> {
    let sampled = sample_reduced_orbit_source(source, options.sampling)?;
    if sampled.samples.is_empty() {
        return Err(ReducedOrbitSourceError::TooFewSamples {
            got: 0,
            required: 1,
        });
    }
    let report = drift(
        elements,
        &sampled.samples,
        sampled.scale,
        options.threshold_m,
    )
    .map_err(ReducedOrbitSourceError::Reduced)?;
    Ok(ReducedOrbitSourceDrift {
        report,
        requested_samples: sampled.requested,
    })
}

/// Sample an SP3 or SGP4 source and fit a piecewise reduced orbit.
pub fn fit_piecewise_reduced_orbit_source(
    source: ReducedOrbitSource<'_>,
    options: PiecewiseOrbitSourceFitOptions,
) -> Result<PiecewiseOrbitSourceFit, ReducedOrbitSourceError> {
    let sampled = sample_reduced_orbit_source(source, options.sampling)?;
    let segment_s = rounded_segment_s(options.segment_s)?;
    let orbit = fit_piecewise(
        &sampled.samples,
        sampled.scale,
        options.model,
        options.sampling.t0,
        options.sampling.t1,
        segment_s,
    )
    .map_err(ReducedOrbitSourceError::Piecewise)?;
    Ok(PiecewiseOrbitSourceFit {
        orbit,
        requested_samples: sampled.requested,
    })
}

/// Sample an SP3 or SGP4 source and evaluate a piecewise model's drift.
pub fn drift_piecewise_reduced_orbit_source(
    piecewise: &PiecewiseOrbit,
    source: ReducedOrbitSource<'_>,
    options: ReducedOrbitSourceDriftOptions,
) -> Result<ReducedOrbitSourceDrift, ReducedOrbitSourceError> {
    let sampled = sample_reduced_orbit_source(source, options.sampling)?;
    if sampled.samples.is_empty() {
        return Err(ReducedOrbitSourceError::TooFewSamples {
            got: 0,
            required: 1,
        });
    }
    let report = piecewise_drift(
        piecewise,
        &sampled.samples,
        sampled.scale,
        options.threshold_m,
    )
    .map_err(ReducedOrbitSourceError::Piecewise)?;
    Ok(ReducedOrbitSourceDrift {
        report,
        requested_samples: sampled.requested,
    })
}

#[derive(Debug)]
struct SourceSamples {
    samples: Vec<EcefSample>,
    requested: usize,
    scale: TimeScale,
}

fn sample_reduced_orbit_source(
    source: ReducedOrbitSource<'_>,
    sampling: ReducedOrbitSourceSampling,
) -> Result<SourceSamples, ReducedOrbitSourceError> {
    let scale = match source {
        ReducedOrbitSource::Sp3 { product, .. } => product.header.time_scale,
        ReducedOrbitSource::Sgp4 { .. } => TimeScale::Utc,
    };
    let steps = source_steps(sampling, scale)?;
    let samples = match source {
        ReducedOrbitSource::Sp3 { product, satellite } => steps
            .iter()
            .filter_map(|epoch| sample_sp3_epoch(product, satellite, *epoch))
            .collect(),
        ReducedOrbitSource::Sgp4 { satellite } => steps
            .iter()
            .filter_map(|epoch| sample_sgp4_epoch(satellite, *epoch))
            .collect(),
    };
    Ok(SourceSamples {
        samples,
        requested: steps.len(),
        scale,
    })
}

fn source_steps(
    sampling: ReducedOrbitSourceSampling,
    scale: TimeScale,
) -> Result<Vec<CalendarEpoch>, ReducedOrbitSourceError> {
    validate_calendar_epoch(sampling.t0, scale, "window.start")
        .map_err(ReducedOrbitSourceError::Reduced)?;
    validate_calendar_epoch(sampling.t1, scale, "window.end")
        .map_err(ReducedOrbitSourceError::Reduced)?;
    if !sampling.cadence_s.is_finite() || sampling.cadence_s <= 0.0 {
        return Err(ReducedOrbitSourceError::InvalidCadence);
    }
    let span_s = calendar_seconds(sampling.t1) - calendar_seconds(sampling.t0);
    if !span_s.is_finite() || span_s <= 0.0 {
        return Err(ReducedOrbitSourceError::InvalidWindow);
    }
    let count = (span_s / sampling.cadence_s).trunc();
    if !count.is_finite() || count < 0.0 {
        return Err(ReducedOrbitSourceError::InvalidWindow);
    }
    let start_s = calendar_seconds(sampling.t0);
    Ok((0..=count as usize)
        .map(|k| calendar_from_seconds(start_s + (k as f64 * sampling.cadence_s).round()))
        .collect())
}

fn sample_sp3_epoch(
    product: &Sp3,
    satellite: GnssSatelliteId,
    epoch: CalendarEpoch,
) -> Option<EcefSample> {
    let instant = instant_from_calendar(epoch, product.header.time_scale).ok()?;
    let state = product.position(satellite, instant).ok()?;
    Some(EcefSample::new(
        epoch,
        state.position.x_m,
        state.position.y_m,
        state.position.z_m,
    ))
}

fn sample_sgp4_epoch(satellite: &Satellite, epoch: CalendarEpoch) -> Option<EcefSample> {
    let prediction = satellite
        .propagate_jd(julian_date_from_calendar(epoch))
        .ok()?;
    let ts = epoch.time_scales(TimeScale::Utc);
    let (gcrs, _) = teme_to_gcrs_compute(
        &TemeStateKm {
            position_km: prediction.position,
            velocity_km_s: prediction.velocity,
        },
        &ts,
        false,
    )
    .ok()?;
    let (x_km, y_km, z_km) = gcrs_to_itrs_compute(gcrs.0, gcrs.1, gcrs.2, &ts, false).ok()?;
    Some(EcefSample::new(
        epoch,
        x_km * M_PER_KM,
        y_km * M_PER_KM,
        z_km * M_PER_KM,
    ))
}

fn instant_from_calendar(
    epoch: CalendarEpoch,
    scale: TimeScale,
) -> Result<Instant, ReducedOrbitSourceError> {
    let (jd_whole, fraction) = split_julian_date(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second,
    );
    let split = JulianDateSplit::new(jd_whole, fraction)
        .map_err(|_| ReducedOrbitSourceError::InvalidWindow)?;
    Ok(Instant::from_julian_date(scale, split))
}

fn julian_date_from_calendar(epoch: CalendarEpoch) -> JulianDate {
    let (jd_whole, fraction) = split_julian_date(
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second,
    );
    JulianDate(jd_whole, fraction)
}

fn rounded_segment_s(segment_s: f64) -> Result<i64, ReducedOrbitSourceError> {
    if !segment_s.is_finite() || segment_s <= 0.0 {
        return Err(ReducedOrbitSourceError::InvalidSegment);
    }
    let rounded = segment_s.round();
    if rounded < 1.0 || rounded > i64::MAX as f64 {
        return Err(ReducedOrbitSourceError::InvalidSegment);
    }
    Ok(rounded as i64)
}

// ---------------------------------------------------------------------------
// Piecewise fit/evaluation.
// ---------------------------------------------------------------------------

// The fit/segmentation time axis is a monotonic continuous-seconds mapping on a
// JD-origin (`jdn * 86400 + sod`) axis. This must stay bit-identical: the large
// absolute magnitude sets the ULP at which sub-second epochs round near segment
// boundaries, so a different origin (e.g. J2000 seconds) would shift sample
// inclusion. The forward/inverse pair below is that exact legacy arithmetic.
fn calendar_seconds(t: CalendarEpoch) -> f64 {
    julian_day_number(t.year, t.month, t.day) as f64 * SECONDS_PER_DAY
        + (t.hour as f64) * 3600.0
        + (t.minute as f64) * 60.0
        + t.second
}

fn calendar_from_seconds(total_s: f64) -> CalendarEpoch {
    let mut jdn = (total_s / SECONDS_PER_DAY).floor() as i64;
    let mut sod = total_s - jdn as f64 * SECONDS_PER_DAY;
    if sod < 0.0 {
        jdn -= 1;
        sod += SECONDS_PER_DAY;
    }
    if sod >= SECONDS_PER_DAY {
        jdn += 1;
        sod -= SECONDS_PER_DAY;
    }

    let (year, month, day) = civil_from_julian_day_number(jdn);
    let hour = (sod / 3600.0).floor() as i32;
    let rem = sod - hour as f64 * 3600.0;
    let minute = (rem / 60.0).floor() as i32;
    let second = rem - minute as f64 * 60.0;

    CalendarEpoch::new(year as i32, month as i32, day as i32, hour, minute, second)
}

fn calendar_add_seconds(t: CalendarEpoch, seconds: i64) -> CalendarEpoch {
    calendar_from_seconds(calendar_seconds(t) + seconds as f64)
}

fn segment_bounds(
    t0: CalendarEpoch,
    t1: CalendarEpoch,
    segment_s: i64,
) -> Result<Vec<(CalendarEpoch, CalendarEpoch)>, PiecewiseOrbitError> {
    if segment_s < 1 {
        return Err(PiecewiseOrbitError::InvalidSegment);
    }
    if calendar_seconds(t1) <= calendar_seconds(t0) {
        return Err(PiecewiseOrbitError::Reduced(
            ReducedOrbitError::InvalidWindow,
        ));
    }

    let mut bounds = Vec::new();
    let mut seg_t0 = t0;
    let end_s = calendar_seconds(t1);
    while calendar_seconds(seg_t0) < end_s {
        let mut seg_t1 = calendar_add_seconds(seg_t0, segment_s);
        if calendar_seconds(seg_t1) > end_s {
            seg_t1 = t1;
        }
        bounds.push((seg_t0, seg_t1));
        seg_t0 = seg_t1;
    }
    Ok(bounds)
}

fn is_too_few_or_invalid_window(e: &ReducedOrbitError) -> bool {
    matches!(
        e,
        ReducedOrbitError::TooFewSamples { .. } | ReducedOrbitError::InvalidWindow
    )
}

fn sample_in_bounds(sample: CalendarEpoch, t0: CalendarEpoch, t1: CalendarEpoch) -> bool {
    let s = calendar_seconds(sample);
    s >= calendar_seconds(t0) && s <= calendar_seconds(t1)
}

/// Fit contiguous reduced-orbit segments over `[t0, t1]`.
///
/// `segment_s` is the positive, already-rounded segment length in whole
/// seconds. The final short segment may be dropped if it has too few samples;
/// interior under-covered segments surface an error because they would create a
/// coverage gap. Sample epochs are partitioned by civil calendar seconds, while
/// each segment fit still interprets those epochs in `scale` for frame
/// transforms.
pub fn fit_piecewise(
    samples: &[EcefSample],
    scale: TimeScale,
    model: Model,
    t0: CalendarEpoch,
    t1: CalendarEpoch,
    segment_s: i64,
) -> Result<PiecewiseOrbit, PiecewiseOrbitError> {
    let bounds = segment_bounds(t0, t1, segment_s)?;
    let last_index = bounds.len().saturating_sub(1);
    let mut segments = Vec::new();

    for (index, (seg_t0, seg_t1)) in bounds.into_iter().enumerate() {
        let subset: Vec<EcefSample> = samples
            .iter()
            .copied()
            .filter(|s| sample_in_bounds(s.epoch, seg_t0, seg_t1))
            .collect();

        match fit_with_model(&subset, scale, model) {
            Ok(orbit) => segments.push(PiecewiseSegment {
                t0: seg_t0,
                t1: seg_t1,
                orbit,
            }),
            Err(e) if index == last_index && is_too_few_or_invalid_window(&e) => {}
            Err(e) => return Err(PiecewiseOrbitError::Reduced(e)),
        }
    }

    let Some(last) = segments.last() else {
        return Err(PiecewiseOrbitError::TooFewSamples {
            got: 0,
            required: MIN_SAMPLES,
        });
    };

    Ok(PiecewiseOrbit {
        model,
        t0,
        t1: last.t1,
        segment_s,
        segments,
    })
}

/// Return the segment covering `epoch`.
///
/// Interior boundaries resolve to the later segment; the exact end of the final
/// segment resolves to that final segment.
pub fn select_piecewise_segment(
    piecewise: &PiecewiseOrbit,
    epoch: CalendarEpoch,
) -> Result<&PiecewiseSegment, PiecewiseOrbitError> {
    validate_piecewise_segments(piecewise)?;
    let e = calendar_seconds(epoch);
    if e < calendar_seconds(piecewise.t0) || e > calendar_seconds(piecewise.t1) {
        return Err(PiecewiseOrbitError::OutOfRange);
    }

    let Some((last, rest)) = piecewise.segments.split_last() else {
        return Err(PiecewiseOrbitError::OutOfRange);
    };

    for seg in rest {
        if e >= calendar_seconds(seg.t0) && e < calendar_seconds(seg.t1) {
            return Ok(seg);
        }
    }

    if e >= calendar_seconds(last.t0) && e <= calendar_seconds(last.t1) {
        Ok(last)
    } else {
        Err(PiecewiseOrbitError::OutOfRange)
    }
}

fn validate_piecewise_segments(piecewise: &PiecewiseOrbit) -> Result<(), PiecewiseOrbitError> {
    if piecewise.segment_s < 1 {
        return Err(PiecewiseOrbitError::InvalidSegment);
    }
    validate::require_strictly_increasing(
        piecewise
            .segments
            .iter()
            .map(|segment| calendar_seconds(segment.t0)),
        "piecewise.segments.t0",
    )
    .map_err(map_piecewise_order_error)?;

    for segment in &piecewise.segments {
        validate::require_strictly_increasing(
            [calendar_seconds(segment.t0), calendar_seconds(segment.t1)],
            "piecewise.segment bounds",
        )
        .map_err(map_piecewise_order_error)?;
    }
    Ok(())
}

fn map_piecewise_order_error(error: validate::FieldError) -> PiecewiseOrbitError {
    let reason = match error {
        validate::FieldError::NonFinite { .. } => "must be finite",
        validate::FieldError::OutOfRange { .. } => "must be strictly increasing",
        _ => error.reason(),
    };
    PiecewiseOrbitError::Reduced(invalid_input(error.field(), reason))
}

/// Evaluate a piecewise reduced orbit at `epoch`.
pub fn piecewise_position(
    piecewise: &PiecewiseOrbit,
    epoch: CalendarEpoch,
    scale: TimeScale,
    frame: Frame,
) -> Result<[f64; 3], PiecewiseOrbitError> {
    validate_calendar_epoch(epoch, scale, "epoch").map_err(PiecewiseOrbitError::Reduced)?;
    let seg = select_piecewise_segment(piecewise, epoch)?;
    position(&seg.orbit.elements, epoch, scale, frame).map_err(PiecewiseOrbitError::Reduced)
}

/// Evaluate piecewise reduced-orbit position and velocity at `epoch`.
pub fn piecewise_position_velocity(
    piecewise: &PiecewiseOrbit,
    epoch: CalendarEpoch,
    scale: TimeScale,
    frame: Frame,
) -> Result<([f64; 3], [f64; 3]), PiecewiseOrbitError> {
    validate_calendar_epoch(epoch, scale, "epoch").map_err(PiecewiseOrbitError::Reduced)?;
    let seg = select_piecewise_segment(piecewise, epoch)?;
    position_velocity(&seg.orbit.elements, epoch, scale, frame)
        .map_err(PiecewiseOrbitError::Reduced)
}

/// Evaluate a piecewise model against truth ECEF samples.
///
/// Truth samples outside the model span are skipped, matching the source-backed
/// single-model drift behavior when product coverage clips a requested horizon.
pub fn piecewise_drift(
    piecewise: &PiecewiseOrbit,
    truth: &[EcefSample],
    scale: TimeScale,
    threshold_m: f64,
) -> Result<DriftReport, PiecewiseOrbitError> {
    validate_finite(threshold_m, "threshold_m").map_err(PiecewiseOrbitError::Reduced)?;
    if truth.is_empty() {
        return Ok(DriftReport {
            per_epoch: Vec::new(),
            max_m: 0.0,
            rms_m: 0.0,
            threshold_horizon: None,
            threshold_index: None,
        });
    }

    let mut per_epoch = Vec::with_capacity(truth.len());
    let mut sumsq = 0.0;
    let mut max_m = 0.0_f64;
    let mut threshold_horizon = None;
    let mut threshold_index = None;

    for s in truth {
        validate_truth_sample(s, scale).map_err(PiecewiseOrbitError::Reduced)?;
        let Ok(model) = piecewise_position(piecewise, s.epoch, scale, Frame::Ecef) else {
            continue;
        };
        let dx = model[0] - s.x_m;
        let dy = model[1] - s.y_m;
        let dz = model[2] - s.z_m;
        let err = (dx * dx + dy * dy + dz * dz).sqrt();
        sumsq += err * err;
        if err > max_m {
            max_m = err;
        }
        if threshold_horizon.is_none() && err > threshold_m {
            threshold_horizon = Some(s.epoch);
            threshold_index = Some(per_epoch.len());
        }
        per_epoch.push(DriftEntry {
            epoch: s.epoch,
            error_m: err,
        });
    }

    if per_epoch.is_empty() {
        return Err(PiecewiseOrbitError::TooFewSamples {
            got: 0,
            required: 1,
        });
    }

    let rms_m = (sumsq / per_epoch.len() as f64).sqrt();
    Ok(DriftReport {
        per_epoch,
        max_m,
        rms_m,
        threshold_horizon,
        threshold_index,
    })
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
