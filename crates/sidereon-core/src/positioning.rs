//! Single-point positioning and GNSS geometry diagnostics.

pub use crate::dop::{dop, Dop, DopError, LineOfSight};
pub use crate::spp::{
    residual_rms, solve, solve_broadcast, solve_spp_batch_parallel, solve_spp_batch_serial,
    solve_with_fallback, solve_with_policy, solve_with_solver, BroadcastReason, Corrections,
    EphemerisSource, FallbackError, FixSource, GalileoNequickCoeffs, KlobucharCoeffs, Observation,
    ReceiverSolution, RejectedSat, RejectionReason, RobustConfig, SolutionMetadata, SolveInputs,
    SolvePolicy, SolvePolicyError, SourcedSolution, SppError, SurfaceMet, DEFAULT_HUBER_K,
    DEFAULT_ROBUST_MAX_OUTER, DEFAULT_ROBUST_OUTER_TOL_M, DEFAULT_ROBUST_SCALE_FLOOR_M,
    ELEVATION_MASK_RAD, SIGMA0_M, TRANSMIT_TIME_ITERATIONS,
};

/// Role-oriented alias for a solved receiver state.
pub type Solution = ReceiverSolution;

/// Error type returned by [`solve`].
pub type Error = SppError;
