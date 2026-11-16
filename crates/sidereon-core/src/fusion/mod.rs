//! GNSS/INS fusion primitives.
//!
//! The module keeps the already-merged inertial mechanization surface available
//! while adding the indirect error-state prediction and EKF correction core.

pub mod ekf;
pub mod error_state;
pub mod loose;
pub mod serial;
pub mod smoother;
pub mod state;
pub mod tight;
pub mod timesync;
pub mod ukf;

pub use crate::inertial::*;
pub use ekf::{
    apply_closed_loop_error, ekf_correct_closed_loop, joseph_covariance_update, EkfCorrection,
    EkfCorrectionReport, EkfUpdateOptions, InnovationGate, InnovationGateReport,
    JointInnovationGateReport,
};
pub use error_state::{
    error_state_process_noise_discrete, error_state_system_matrix_ecef,
    error_state_system_matrix_ecef_with_imu_to_body, error_state_transition_matrix,
    linearize_error_state_ecef, linearize_error_state_ecef_with_imu_to_body,
    predict_error_state_covariance, ErrorStateImuKinematics, ErrorStateLinearization,
};
pub use loose::{
    loose_coupling_correction, velocity_match_outage, velocity_match_outage_to_state, FusionUpdate,
    GnssFixMeasurement, GnssFixStatus, GnssFixStatusWeighting, IggIiiMeasurementReweighting,
    InertialFilter, InertialFilterConfig, LooseCouplingConfig, NonHolonomicConstraintConfig,
    StationaryDetectorConfig, StationaryUpdateConfig, VelocityMatchState,
    VelocityMatchedTrajectory, VelocityMatchingConfig, YangPredictionAdaptiveFactor,
};
pub use serial::{
    F64Bits, FusionStateCodecError, SerializableErrorStateLayout, SerializableFusionSnapshot,
    SerializableFusionState, SerializableGnssFixStatus, SerializableImuSample,
    SerializableImuSampleKind, SerializableInsFilterState, SerializableLooseMeasurement,
    SerializableNavState, SerializableRateEndpoint, SerializableSatelliteId,
    SerializableStationarityDetectorSample, SerializableStoredCheckpoint,
    SerializableStoredGnssMeasurement, SerializableStoredImuSample,
    SerializableTightCarrierPhaseObservation, SerializableTightFilterState,
    SerializableTightGnssEpoch, SerializableTightGnssObservation,
    SerializableTightRangeRateObservation, SerializableTimeSyncHistory,
    SerializableTimeSyncHistoryConfig, FUSION_STATE_CODEC_VERSION,
};
pub use smoother::{
    smooth_fusion_rts, FusionRtsEpoch, FusionRtsHistory, FusionRtsHistoryBuilder,
    SmoothedFusionEpoch, SmoothedFusionTrajectory,
};
pub use state::{
    covariance_is_positive_semidefinite, reproject_covariance_psd, validate_covariance_matrix,
    ErrorStateLayout, ErrorStateVector, FusionError, FusionFilterKind, InsFilterState,
    ERROR_ACCEL_BIAS_INDEX, ERROR_ACCEL_SCALE_INDEX, ERROR_ATTITUDE_INDEX, ERROR_GYRO_BIAS_INDEX,
    ERROR_GYRO_SCALE_INDEX, ERROR_MOUNTING_MISALIGNMENT_INDEX,
    ERROR_MOUNTING_MISALIGNMENT_STATE_COUNT, ERROR_POSITION_INDEX, ERROR_STATE_DIMENSION_15,
    ERROR_STATE_DIMENSION_21, ERROR_VELOCITY_INDEX,
};
pub use tight::{
    TightCarrierPhaseObservation, TightClockState, TightCouplingConfig, TightFilterSnapshot,
    TightGnssEpoch, TightGnssObservation, TightRangeRateObservation, TIGHT_CLOCK_BIAS_OFFSET,
    TIGHT_CLOCK_DRIFT_OFFSET, TIGHT_CLOCK_STATE_COUNT,
};
pub use timesync::{
    validate_time_sync_gnss_order, validate_time_sync_imu_order, InertialFilterSnapshot,
    StationarityDetectorSnapshotSample, TimeSyncHistoryConfig, TimeSyncHistoryStatus,
    TimeSyncUpdate, DEFAULT_TIME_SYNC_CHECKPOINT_CAPACITY, DEFAULT_TIME_SYNC_IMU_CAPACITY,
};
pub use ukf::{ukf_correct_closed_loop, UkfUpdateOptions, UnscentedTransformOptions};
