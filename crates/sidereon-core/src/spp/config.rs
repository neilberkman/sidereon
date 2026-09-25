//! SPP solver policy constants.

pub use crate::constants::SPP_TRANSMIT_TIME_ITERATIONS as TRANSMIT_TIME_ITERATIONS;

const PI: f64 = std::f64::consts::PI;

/// Elevation mask in radians (10 degrees); a satellite is excluded iff its
/// elevation is strictly below this value.
pub const ELEVATION_MASK_RAD: f64 = 10.0 * PI / 180.0;

/// Ratio of the code to the carrier-phase error, RTKLIB `prcopt_default` `eratio[0]`:
/// the code error of RTKLIB `varerr` is this times the carrier-phase terms
/// [`PHASE_ERROR_BASE_M`] and [`PHASE_ERROR_ELEVATION_M`].
pub const CODE_PHASE_ERROR_RATIO: f64 = 300.0;

/// Carrier-phase error term independent of elevation (m), RTKLIB `prcopt_default`
/// `err[1]`.
pub const PHASE_ERROR_BASE_M: f64 = 0.003;

/// Carrier-phase error term divided by `sin(el)` (m), RTKLIB `prcopt_default`
/// `err[2]`.
pub const PHASE_ERROR_ELEVATION_M: f64 = 0.003;

/// Elevation (rad) below which RTKLIB `varerr` evaluates its elevation term at this
/// elevation instead, `MIN_EL` (5 degrees).
pub const ERROR_MODEL_MIN_ELEVATION_RAD: f64 = 5.0 * (PI / 180.0);

/// Code bias error standard deviation (m) of a single-frequency pseudorange, RTKLIB
/// `ERR_CBIAS`.
pub const CODE_BIAS_ERROR_M: f64 = 0.3;

/// Standard deviation (m) of the ionosphere delay a solve does not correct, RTKLIB
/// `ERR_ION`.
pub const UNCORRECTED_IONOSPHERE_ERROR_M: f64 = 5.0;

/// Error factor of a broadcast ionosphere model: the standard deviation is this times
/// the delay, RTKLIB `ERR_BRDCI`.
pub const BROADCAST_IONOSPHERE_ERROR_FACTOR: f64 = 0.5;

/// Standard deviation (m) of the troposphere delay a solve does not correct, RTKLIB
/// `ERR_TROP`.
pub const UNCORRECTED_TROPOSPHERE_ERROR_M: f64 = 3.0;

/// Relative humidity RTKLIB `tropcorr` gives `tropmodel` for its Saastamoinen option,
/// `REL_HUMI`.
pub const RTKLIB_TROPOSPHERE_RELATIVE_HUMIDITY: f64 = 0.7;

/// Zenith error (m) of the Saastamoinen troposphere model, RTKLIB `ERR_SAAS`, mapped
/// to the line of sight as `ERR_SAAS / (sin(el) + 0.1)`.
pub const TROPOSPHERE_MODEL_ERROR_M: f64 = 0.3;

/// Default Huber tuning constant for the opt-in robust reweighting path.
pub use crate::astro::math::robust::HUBER_K as DEFAULT_HUBER_K;

/// Default robust scale floor (m): the smallest MAD-derived scale allowed, so a
/// near-perfect fit cannot blow up the scaled residuals and down-weight every
/// satellite. Sized to the metre-class code noise of cheap single-frequency
/// receivers.
pub const DEFAULT_ROBUST_SCALE_FLOOR_M: f64 = 1.0;

/// Default maximum outer IRLS reweighting iterations (the warm-started static
/// solve at iteration 0 plus reweighted resolves up to this many total).
///
/// The reweighting ends when it settles (the position moves less than
/// [`DEFAULT_ROBUST_OUTER_TOL_M`] and the selection holds), so this is a safety
/// cap, not a working budget: a solve that reaches it has not converged and
/// reports [`crate::astro::math::least_squares::Status::OuterBudgetExhausted`].
/// A +300 m fault on one of eight satellites settles within 7 to 28 solves;
/// the cap of 100 leaves more than three times that.
pub const DEFAULT_ROBUST_MAX_OUTER: usize = 100;

/// Default outer-loop position step tolerance (m): the outer IRLS loop stops
/// when the L2 norm of the position change between successive reweighted solves
/// drops below this.
pub const DEFAULT_ROBUST_OUTER_TOL_M: f64 = 1e-4;

/// Maximum number of passes one SPP solve runs, RTKLIB `estpos` `MAXITR`. Each
/// pass selects the satellites and weights at the current iterate and then either
/// runs the trust-region solve over a new selection or takes RTKLIB's
/// least-squares step for the one it holds; a solve that has not ended after this
/// many passes fails with [`crate::spp::SppError::SelectionUnsettled`], as
/// `estpos` fails after `MAXITR` iterations.
pub const MAX_SELECTION_PASSES: usize = 10;

/// The step (m) that ends an SPP solve, RTKLIB `estpos`'s `norm(dx) < 1E-4`: the
/// norm of the whole least-squares step, receiver position and clocks, at the
/// iterate's own selection and weights.
pub const SELECTION_STEP_TOL_M: f64 = 1e-4;
