//! GNSS estimation substrate (Phase-2).
//!
//! The Phase-2 redesign collapses the three thick estimator stacks
//! (`spp`, `rtk`/`rtk_filter`, `precise_positioning`) onto one shared estimation
//! substrate plus thin, runtime-selectable strategies. The mechanism that keeps
//! every external reference bit-exact through the migration is the NAMED recipe:
//! a strategy selects the floating-point operation order it needs by enum value,
//! never by owning a private copy of a parity-sensitive helper.
//!
//! P0 landed [`recipe`] - the taxonomy of operation-order variants, each
//! documented against the current code path it names and defaulting to current
//! behavior. P1 adds `substrate` with the frame and range kernels; P2 adds the
//! parameter layout, the weighted measurement row, the measurement covariance
//! block, and the normal-equation assembler; P3 adds the shared ambiguity and qc
//! kernels, all routed through by the existing strategies via their current
//! recipe. P4 adds [`strategies`]: the runtime [`StrategyId`] selector
//! ([`strategies::estimate`]) that resolves a strategy into its recipe and
//! screen/ambiguity policy data and dispatches to the technique's reference
//! entry point, leaving every existing 0-ULP golden unchanged.

pub mod primitives;
pub mod recipe;
pub mod strategies;
pub(crate) mod substrate;
pub mod track;

pub use primitives::{
    alpha_beta_apply_measurement, alpha_beta_filter_step, alpha_beta_predict,
    alpha_beta_steady_state_gains, cfar_ca_false_alarm_probability, cfar_ca_multiplier_from_pfa,
    cfar_ca_pfa_from_multiplier, cfar_ca_threshold, ewma_update, ewma_update_power_of_two,
    kalman_cv_steady_state_gains, mad_spread, nis_expected_value, nis_gate_test,
    nis_gate_threshold, nis_statistic, normalized_innovation, AlphaBetaGains, AlphaBetaState,
    AlphaBetaStep, PrimitiveError, ScalarKalmanGains, MAD_GAUSSIAN_CONSISTENCY,
};
pub use recipe::{
    AmbiguityIdPolicy, DifferencingMode, EstimationRecipe, FrameRecipe, NormalRecipe,
    PartialResolution, RangeRecipe, ReferenceTarget, ResidualNormRecipe, SagnacRecipe, ScreenKind,
    SolverRecipe, StrategyId, Technique,
};
pub use strategies::{
    estimate, EstimateError, EstimateInput, EstimateOptions, EstimateOutput, ResolvedStrategy,
};
pub use track::{
    rts_smooth, smooth_track_rts, SmoothedTrack, SmoothedTrackEpoch, TrackCoordinateFrame,
    TrackError, TrackFilter, TrackFilterConfig, TrackGatedUpdate, TrackInnovation, TrackPrediction,
    TrackRtsEpoch, TrackRtsHistory, TrackRtsHistoryBuilder, TrackState, TrackUpdate,
};
