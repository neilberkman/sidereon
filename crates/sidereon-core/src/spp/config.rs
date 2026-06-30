//! SPP solver policy constants.

pub use crate::constants::SPP_TRANSMIT_TIME_ITERATIONS as TRANSMIT_TIME_ITERATIONS;

const PI: f64 = std::f64::consts::PI;

/// Elevation mask in radians (10 degrees); a satellite is excluded iff its
/// elevation is strictly below this value.
pub const ELEVATION_MASK_RAD: f64 = 10.0 * PI / 180.0;

/// Base measurement standard deviation (m) for the elevation weight model.
pub const SIGMA0_M: f64 = 1.0;

/// Default Huber tuning constant for the opt-in robust reweighting path.
pub use crate::astro::math::robust::HUBER_K as DEFAULT_HUBER_K;

/// Default robust scale floor (m): the smallest MAD-derived scale allowed, so a
/// near-perfect fit cannot blow up the scaled residuals and down-weight every
/// satellite. Sized to the metre-class code noise of cheap single-frequency
/// receivers.
pub const DEFAULT_ROBUST_SCALE_FLOOR_M: f64 = 1.0;

/// Default maximum outer IRLS reweighting iterations (the warm-started static
/// solve at iteration 0 plus reweighted resolves up to this many total).
pub const DEFAULT_ROBUST_MAX_OUTER: usize = 5;

/// Default outer-loop position step tolerance (m): the outer IRLS loop stops
/// when the L2 norm of the position change between successive reweighted solves
/// drops below this.
pub const DEFAULT_ROBUST_OUTER_TOL_M: f64 = 1e-4;
