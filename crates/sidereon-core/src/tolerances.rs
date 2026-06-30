//! Shared named numerical tolerances for GNSS modeling kernels.
//!
//! These names preserve existing numeric thresholds while making their
//! semantics explicit at call sites.

pub use crate::astro::tolerances::PIVOT_EPSILON;

/// Frequency difference accepted when checking that two observations use the
/// same configured carrier frequency.
pub const FREQUENCY_MATCH_EPS_HZ: f64 = 1.0e-6;

/// Frequency denominator threshold below which carrier combinations are
/// considered degenerate.
pub const FREQUENCY_DENOMINATOR_EPS_HZ: f64 = 1.0;

/// Whole-second epoch-lattice tolerance used by SP3 merge/decimation.
pub const WHOLE_SECOND_EPS_S: f64 = 1.0e-6;

/// Threshold below which a fitted eccentricity uses the circular fast path.
pub const ECCENTRICITY_ZERO_EPS: f64 = 1.0e-12;

/// Degenerate vector-norm threshold for geometry setup.
pub const VECTOR_NORM_ZERO_EPS: f64 = PIVOT_EPSILON;

/// Satellite-yaw singularity threshold in radians.
pub const YAW_SINGULARITY_EPS_RAD: f64 = 1.0e-12;

/// GLONASS fixed-step residual-time loop tolerance.
pub const GLONASS_TIME_EPS_S: f64 = 1.0e-9;

/// Signal Doppler-grid endpoint tolerance.
pub const DOPPLER_GRID_EDGE_EPS_HZ: f64 = 1.0e-9;

/// RTKLIB LAMBDA reduction permutation hysteresis.
pub const LAMBDA_REDUCTION_EPS: f64 = 1.0e-6;

/// Reduced-orbit trust-region solve convergence tolerance.
pub const REDUCED_ORBIT_SOLVER_TOL: f64 = 1.0e-12;

/// Reduced-orbit eccentric-anomaly Newton-step convergence threshold.
pub const REDUCED_ORBIT_KEPLER_STEP_EPS_RAD: f64 = 1.0e-14;
