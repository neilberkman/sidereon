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

/// Tolerance for comparing SP3 epoch instants carried as `f64` seconds: an
/// exact product's header start fields against the requested start, and two
/// statements of one epoch in the interpolator's node lookup. SP3 epoch intervals, grid steps and
/// the first epoch record are not compared with it: the merge, the merge-input
/// identity and exact-product validation count them exactly in the
/// 10-nanosecond ticks an SP3 epoch record states.
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

/// SBAS IGP coordinate equality tolerance in degrees for grid merge/lookups.
pub const SBAS_IGP_COORD_EPS_DEG: f64 = 1.0e-9;

/// Relative carrier-frequency tolerance for PPP code-bias observable checks;
/// scaled by the larger compared frequency to preserve the existing bound.
pub const PPP_FREQUENCY_REL_EPS: f64 = 1.0e-12;

/// Absolute carrier-frequency tolerance floor for PPP observable matching and
/// GLONASS channel inference, covering decimal lookup/parse roundoff in Hz.
pub const PPP_FREQUENCY_ABS_EPS_HZ: f64 = 1.0e-3;

/// RTKLIB LAMBDA reduction permutation hysteresis.
pub const LAMBDA_REDUCTION_EPS: f64 = 1.0e-6;

/// Reduced-orbit trust-region solve convergence tolerance.
pub const REDUCED_ORBIT_SOLVER_TOL: f64 = 1.0e-12;

/// Reduced-orbit eccentric-anomaly Newton-step convergence threshold.
pub const REDUCED_ORBIT_KEPLER_STEP_EPS_RAD: f64 = 1.0e-14;
