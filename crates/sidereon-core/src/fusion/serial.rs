//! Versioned serialization for fusion filter checkpoints.

use super::loose::{GnssFixMeasurement, GnssFixStatus, InertialFilter};
use super::state::{
    validate_covariance_matrix, validate_finite_slice, ErrorStateLayout, ErrorStateVector,
    InsFilterState,
};
use super::tight::{
    augmented_dimension, TightCarrierPhaseObservation, TightFilterSnapshot, TightGnssEpoch,
    TightGnssObservation, TightRangeRateObservation,
};
use super::timesync::{
    InertialFilterSnapshot, RateEndpoint, StationarityDetectorSnapshotSample, StoredCheckpoint,
    StoredGnssMeasurement, StoredImuSample, TimeSyncHistoryConfig, TimeSyncHistorySnapshot,
};
use crate::inertial::{ImuSample, ImuSampleKind, NavState};
use crate::{GnssSatelliteId, GnssSystem};

const FUSION_STATE_MAGIC: [u8; 8] = *b"FUSSTAT\0";
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Current binary and serde schema version for fusion checkpoints.
pub const FUSION_STATE_CODEC_VERSION: u16 = 4;
const MIN_SUPPORTED_FUSION_STATE_CODEC_VERSION: u16 = 1;

/// Exact JSON representation of an `f64` bit pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct F64Bits {
    /// IEEE-754 binary64 bit pattern.
    pub bits: u64,
}

impl F64Bits {
    /// Convert an `f64` to its raw bit representation.
    pub const fn from_f64(value: f64) -> Self {
        Self {
            bits: value.to_bits(),
        }
    }

    /// Convert the raw bit representation back to `f64`.
    pub const fn to_f64(self) -> f64 {
        f64::from_bits(self.bits)
    }
}

/// Serializable error-state layout tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SerializableErrorStateLayout {
    /// Fifteen-state layout `dr, dv, psi, b_a, b_g`.
    Fifteen,
    /// Twenty-one-state layout with accelerometer and gyroscope scale factors.
    TwentyOne,
}

impl SerializableErrorStateLayout {
    /// Convert a native layout into its stable serialized tag.
    pub const fn from_native(layout: ErrorStateLayout) -> Self {
        match layout {
            ErrorStateLayout::Fifteen => Self::Fifteen,
            ErrorStateLayout::TwentyOne => Self::TwentyOne,
        }
    }

    /// Convert the stable serialized tag back to the native layout.
    pub const fn to_native(self) -> ErrorStateLayout {
        match self {
            Self::Fifteen => ErrorStateLayout::Fifteen,
            Self::TwentyOne => ErrorStateLayout::TwentyOne,
        }
    }
}

/// Serializable navigation state with exact floating-point bit storage.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableNavState {
    /// State time in seconds since J2000, stored as raw `f64` bits.
    pub t_j2000_s: F64Bits,
    /// IMU ECEF position in meters, stored as raw `f64` bits.
    pub position_ecef_m: [F64Bits; 3],
    /// IMU ECEF velocity in meters per second, stored as raw `f64` bits.
    pub velocity_ecef_mps: [F64Bits; 3],
    /// Body-to-ECEF direction cosine matrix, stored as raw `f64` bits.
    pub attitude_body_to_ecef: [[F64Bits; 3]; 3],
    /// Closed-loop accelerometer bias estimate, stored as raw `f64` bits.
    pub accel_bias_mps2: [F64Bits; 3],
    /// Closed-loop gyroscope bias estimate, stored as raw `f64` bits.
    pub gyro_bias_rps: [F64Bits; 3],
}

impl SerializableNavState {
    /// Convert a native navigation state into the stable serialized form.
    pub fn from_native(state: &NavState) -> Self {
        Self {
            t_j2000_s: F64Bits::from_f64(state.t_j2000_s),
            position_ecef_m: bits3(state.position_ecef_m),
            velocity_ecef_mps: bits3(state.velocity_ecef_mps),
            attitude_body_to_ecef: bits3x3(state.attitude_body_to_ecef),
            accel_bias_mps2: bits3(state.accel_bias_mps2),
            gyro_bias_rps: bits3(state.gyro_bias_rps),
        }
    }

    /// Convert the stable serialized form back to a validated native state.
    pub fn to_native(&self) -> Result<NavState, FusionStateCodecError> {
        let state = NavState {
            t_j2000_s: self.t_j2000_s.to_f64(),
            position_ecef_m: f643(self.position_ecef_m),
            velocity_ecef_mps: f643(self.velocity_ecef_mps),
            attitude_body_to_ecef: f643x3(self.attitude_body_to_ecef),
            accel_bias_mps2: f643(self.accel_bias_mps2),
            gyro_bias_rps: f643(self.gyro_bias_rps),
        };
        state.validate().map_err(invalid_state)?;
        Ok(state)
    }
}

/// Serializable INS filter state and covariance.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableInsFilterState {
    /// Error-state covariance layout.
    pub layout: SerializableErrorStateLayout,
    /// Nonlinear mechanized navigation state.
    pub nominal: SerializableNavState,
    /// Error-state vector values, stored as raw `f64` bits.
    pub error_state: Vec<F64Bits>,
    /// Error-state covariance matrix, stored as raw `f64` bits.
    pub covariance: Vec<Vec<F64Bits>>,
    /// Accelerometer scale-factor estimates, stored as raw `f64` bits.
    pub accel_scale_factor: [F64Bits; 3],
    /// Gyroscope scale-factor estimates, stored as raw `f64` bits.
    pub gyro_scale_factor: [F64Bits; 3],
}

impl SerializableInsFilterState {
    /// Convert a native INS filter state into the stable serialized form.
    pub fn from_native(state: &InsFilterState) -> Self {
        Self {
            layout: SerializableErrorStateLayout::from_native(state.layout()),
            nominal: SerializableNavState::from_native(&state.nominal),
            error_state: bits_slice(state.error_state.as_slice()),
            covariance: bits_matrix(&state.covariance),
            accel_scale_factor: bits3(state.accel_scale_factor),
            gyro_scale_factor: bits3(state.gyro_scale_factor),
        }
    }

    /// Convert the stable serialized form back to a validated native state.
    pub fn to_native(&self) -> Result<InsFilterState, FusionStateCodecError> {
        let layout = self.layout.to_native();
        let nominal = self.nominal.to_native()?;
        let error_state = ErrorStateVector::from_vec(layout, f64_vec(&self.error_state))
            .map_err(invalid_state)?;
        let state = InsFilterState {
            nominal,
            error_state,
            covariance: f64_matrix(&self.covariance),
            accel_scale_factor: f643(self.accel_scale_factor),
            gyro_scale_factor: f643(self.gyro_scale_factor),
        };
        state.validate().map_err(invalid_state)?;
        Ok(state)
    }
}

/// Serializable tight receiver-clock augmentation.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableTightFilterState {
    /// Receiver-clock range bias in meters, stored as raw `f64` bits.
    pub clock_bias_m: F64Bits,
    /// Receiver-clock drift in meters per second, stored as raw `f64` bits.
    pub clock_drift_m_s: F64Bits,
    /// Full augmented covariance, stored as raw `f64` bits.
    pub augmented_covariance: Vec<Vec<F64Bits>>,
}

impl SerializableTightFilterState {
    /// Convert a native tight snapshot into the stable serialized form.
    pub fn from_native(snapshot: &TightFilterSnapshot) -> Self {
        Self {
            clock_bias_m: F64Bits::from_f64(snapshot.clock_bias_m),
            clock_drift_m_s: F64Bits::from_f64(snapshot.clock_drift_m_s),
            augmented_covariance: bits_matrix(&snapshot.augmented_covariance),
        }
    }

    /// Convert the stable serialized form back to a native tight snapshot.
    pub fn to_native(
        &self,
        base_dimension: usize,
    ) -> Result<TightFilterSnapshot, FusionStateCodecError> {
        let snapshot = TightFilterSnapshot {
            clock_bias_m: self.clock_bias_m.to_f64(),
            clock_drift_m_s: self.clock_drift_m_s.to_f64(),
            augmented_covariance: f64_matrix(&self.augmented_covariance),
        };
        validate_finite_slice(
            &[snapshot.clock_bias_m, snapshot.clock_drift_m_s],
            "tight_clock",
        )
        .map_err(invalid_state)?;
        validate_covariance_matrix(
            &snapshot.augmented_covariance,
            augmented_dimension(base_dimension),
            "tight_augmented_covariance",
        )
        .map_err(invalid_state)?;
        Ok(snapshot)
    }
}

/// Serializable retained-history capacity settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct SerializableTimeSyncHistoryConfig {
    /// Number of retained IMU samples.
    pub imu_capacity: u32,
    /// Number of retained GNSS checkpoints.
    pub checkpoint_capacity: u32,
}

impl SerializableTimeSyncHistoryConfig {
    /// Build serialized history capacities from both required limits.
    #[must_use]
    pub const fn new(imu_capacity: u32, checkpoint_capacity: u32) -> Self {
        Self {
            imu_capacity,
            checkpoint_capacity,
        }
    }

    /// Convert native time-sync capacity settings into the serialized form.
    pub fn from_native(config: TimeSyncHistoryConfig) -> Result<Self, FusionStateCodecError> {
        Ok(Self {
            imu_capacity: checked_u32(config.imu_capacity)?,
            checkpoint_capacity: checked_u32(config.checkpoint_capacity)?,
        })
    }

    /// Convert the serialized capacity settings back to native settings.
    pub fn to_native(self) -> Result<TimeSyncHistoryConfig, FusionStateCodecError> {
        let config = TimeSyncHistoryConfig::new(
            self.imu_capacity as usize,
            self.checkpoint_capacity as usize,
        );
        config.validate().map_err(invalid_state)?;
        Ok(config)
    }
}

/// Serializable GNSS satellite identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableSatelliteId {
    /// GNSS constellation.
    pub system: GnssSystem,
    /// Within-system PRN or slot number.
    pub prn: u8,
}

impl SerializableSatelliteId {
    /// Convert a native satellite identifier into the serialized form.
    pub const fn from_native(id: GnssSatelliteId) -> Self {
        Self {
            system: id.system,
            prn: id.prn,
        }
    }

    /// Convert the serialized identifier back to the native type.
    pub fn to_native(self) -> Result<GnssSatelliteId, FusionStateCodecError> {
        GnssSatelliteId::new(self.system, self.prn).map_err(invalid_state)
    }
}

/// Serializable body-rate interpolation endpoint for retained IMU samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableRateEndpoint {
    /// Endpoint epoch in seconds since J2000, stored as raw `f64` bits.
    pub t_j2000_s: F64Bits,
    /// Specific force in body axes, stored as raw `f64` bits.
    pub specific_force_mps2: [F64Bits; 3],
    /// Angular rate in body axes, stored as raw `f64` bits.
    pub angular_rate_rps: [F64Bits; 3],
}

impl SerializableRateEndpoint {
    /// Convert a native rate endpoint into the serialized form.
    fn from_native(endpoint: RateEndpoint) -> Self {
        Self {
            t_j2000_s: F64Bits::from_f64(endpoint.t_j2000_s),
            specific_force_mps2: bits3(endpoint.specific_force_mps2),
            angular_rate_rps: bits3(endpoint.angular_rate_rps),
        }
    }

    /// Convert the serialized endpoint back to native values.
    fn to_native(self) -> RateEndpoint {
        RateEndpoint {
            t_j2000_s: self.t_j2000_s.to_f64(),
            specific_force_mps2: f643(self.specific_force_mps2),
            angular_rate_rps: f643(self.angular_rate_rps),
        }
    }
}

/// Serializable stationary-detector sample retained by a filter snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableStationarityDetectorSample {
    /// Specific-force norm error in m/s^2, stored as raw `f64` bits.
    pub specific_force_norm_error_mps2: F64Bits,
    /// Body angular-rate norm relative to ECEF in rad/s, stored as raw `f64` bits.
    pub body_rate_wrt_ecef_norm_rps: F64Bits,
}

impl SerializableStationarityDetectorSample {
    fn from_native(sample: StationarityDetectorSnapshotSample) -> Self {
        Self {
            specific_force_norm_error_mps2: F64Bits::from_f64(
                sample.specific_force_norm_error_mps2,
            ),
            body_rate_wrt_ecef_norm_rps: F64Bits::from_f64(sample.body_rate_wrt_ecef_norm_rps),
        }
    }

    fn to_native(self) -> StationarityDetectorSnapshotSample {
        StationarityDetectorSnapshotSample {
            specific_force_norm_error_mps2: self.specific_force_norm_error_mps2.to_f64(),
            body_rate_wrt_ecef_norm_rps: self.body_rate_wrt_ecef_norm_rps.to_f64(),
        }
    }
}

/// Serializable IMU sample payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SerializableImuSampleKind {
    /// Specific force and angular rate payload.
    Rate {
        /// Specific force in body axes, stored as raw `f64` bits.
        specific_force_mps2: [F64Bits; 3],
        /// Angular rate in body axes, stored as raw `f64` bits.
        angular_rate_rps: [F64Bits; 3],
    },
    /// Integrated delta-velocity and delta-angle payload.
    Increment {
        /// Body-frame delta velocity, stored as raw `f64` bits.
        delta_velocity_mps: [F64Bits; 3],
        /// Body-frame delta angle, stored as raw `f64` bits.
        delta_theta_rad: [F64Bits; 3],
        /// Sample integration interval, stored as raw `f64` bits.
        dt_s: F64Bits,
    },
}

impl SerializableImuSampleKind {
    /// Convert a native IMU sample payload into the serialized form.
    pub fn from_native(kind: ImuSampleKind) -> Self {
        match kind {
            ImuSampleKind::Rate {
                specific_force_mps2,
                angular_rate_rps,
            } => Self::Rate {
                specific_force_mps2: bits3(specific_force_mps2),
                angular_rate_rps: bits3(angular_rate_rps),
            },
            ImuSampleKind::Increment {
                delta_velocity_mps,
                delta_theta_rad,
                dt_s,
            } => Self::Increment {
                delta_velocity_mps: bits3(delta_velocity_mps),
                delta_theta_rad: bits3(delta_theta_rad),
                dt_s: F64Bits::from_f64(dt_s),
            },
        }
    }

    /// Convert the serialized payload back to native IMU sample values.
    pub fn to_native(self) -> ImuSampleKind {
        match self {
            Self::Rate {
                specific_force_mps2,
                angular_rate_rps,
            } => ImuSampleKind::Rate {
                specific_force_mps2: f643(specific_force_mps2),
                angular_rate_rps: f643(angular_rate_rps),
            },
            Self::Increment {
                delta_velocity_mps,
                delta_theta_rad,
                dt_s,
            } => ImuSampleKind::Increment {
                delta_velocity_mps: f643(delta_velocity_mps),
                delta_theta_rad: f643(delta_theta_rad),
                dt_s: dt_s.to_f64(),
            },
        }
    }
}

/// Serializable IMU sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableImuSample {
    /// Sample end epoch in seconds since J2000, stored as raw `f64` bits.
    pub t_j2000_s: F64Bits,
    /// Sample payload.
    pub kind: SerializableImuSampleKind,
}

impl SerializableImuSample {
    /// Convert a native IMU sample into the serialized form.
    pub fn from_native(sample: ImuSample) -> Self {
        Self {
            t_j2000_s: F64Bits::from_f64(sample.t_j2000_s),
            kind: SerializableImuSampleKind::from_native(sample.kind),
        }
    }

    /// Convert the serialized sample back to native IMU sample values.
    pub fn to_native(self) -> ImuSample {
        ImuSample {
            t_j2000_s: self.t_j2000_s.to_f64(),
            kind: self.kind.to_native(),
        }
    }
}

/// Serializable retained IMU sample with its replay interval metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableStoredImuSample {
    /// Previous sample boundary in seconds since J2000, stored as raw `f64` bits.
    pub previous_t_j2000_s: F64Bits,
    /// Stored sample at the interval end.
    pub sample: SerializableImuSample,
    /// Previous rate endpoint for fractional rate replay.
    pub previous_rate: Option<SerializableRateEndpoint>,
}

impl SerializableStoredImuSample {
    /// Convert a retained IMU sample into the serialized form.
    fn from_native(sample: StoredImuSample) -> Self {
        Self {
            previous_t_j2000_s: F64Bits::from_f64(sample.previous_t_j2000_s),
            sample: SerializableImuSample::from_native(sample.sample),
            previous_rate: sample
                .previous_rate
                .map(SerializableRateEndpoint::from_native),
        }
    }

    /// Convert the serialized retained IMU sample back to native values.
    fn to_native(self) -> StoredImuSample {
        StoredImuSample {
            previous_t_j2000_s: self.previous_t_j2000_s.to_f64(),
            sample: self.sample.to_native(),
            previous_rate: self.previous_rate.map(SerializableRateEndpoint::to_native),
        }
    }
}

/// Serializable loose GNSS fix measurement.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableLooseMeasurement {
    /// Measurement epoch in seconds since J2000, stored as raw `f64` bits.
    pub t_j2000_s: F64Bits,
    /// GNSS antenna position in ECEF meters, stored as raw `f64` bits.
    pub position_ecef_m: [F64Bits; 3],
    /// Optional GNSS antenna velocity in ECEF meters per second.
    pub velocity_ecef_mps: Option<[F64Bits; 3]>,
    /// Measurement covariance, stored as raw `f64` bits.
    pub covariance: Vec<Vec<F64Bits>>,
    /// Number of satellites used by the upstream GNSS fix.
    pub satellites_used: u32,
    /// Whether the upstream GNSS solver reported a successful fix.
    pub solution_valid: bool,
    /// Upstream fix class used for loose covariance scaling.
    #[serde(default = "default_serializable_fix_status")]
    pub fix_status: SerializableGnssFixStatus,
}

/// Serializable loose GNSS fix status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SerializableGnssFixStatus {
    /// Code-only or standalone GNSS fix.
    Single,
    /// Float carrier-phase ambiguity solution.
    Float,
    /// Fixed carrier-phase ambiguity solution.
    Fixed,
}

fn default_serializable_fix_status() -> SerializableGnssFixStatus {
    SerializableGnssFixStatus::Single
}

impl SerializableGnssFixStatus {
    fn from_native(status: GnssFixStatus) -> Self {
        match status {
            GnssFixStatus::Single => Self::Single,
            GnssFixStatus::Float => Self::Float,
            GnssFixStatus::Fixed => Self::Fixed,
        }
    }

    const fn to_native(self) -> GnssFixStatus {
        match self {
            Self::Single => GnssFixStatus::Single,
            Self::Float => GnssFixStatus::Float,
            Self::Fixed => GnssFixStatus::Fixed,
        }
    }
}

impl SerializableLooseMeasurement {
    /// Convert a native loose measurement into the serialized form.
    pub fn from_native(measurement: &GnssFixMeasurement) -> Result<Self, FusionStateCodecError> {
        Ok(Self {
            t_j2000_s: F64Bits::from_f64(measurement.t_j2000_s),
            position_ecef_m: bits3(measurement.position_ecef_m),
            velocity_ecef_mps: measurement.velocity_ecef_mps.map(bits3),
            covariance: bits_matrix(&measurement.covariance),
            satellites_used: checked_u32(measurement.satellites_used)?,
            solution_valid: measurement.solution_valid,
            fix_status: SerializableGnssFixStatus::from_native(measurement.fix_status),
        })
    }

    /// Convert the serialized loose measurement back to native values.
    pub fn to_native(&self) -> Result<GnssFixMeasurement, FusionStateCodecError> {
        let measurement = GnssFixMeasurement {
            t_j2000_s: self.t_j2000_s.to_f64(),
            position_ecef_m: f643(self.position_ecef_m),
            velocity_ecef_mps: self.velocity_ecef_mps.map(f643),
            covariance: f64_matrix(&self.covariance),
            satellites_used: self.satellites_used as usize,
            solution_valid: self.solution_valid,
            fix_status: self.fix_status.to_native(),
        };
        measurement.validate().map_err(invalid_state)?;
        Ok(measurement)
    }
}

/// Serializable tight range-rate observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableTightRangeRateObservation {
    /// Measured range rate, stored as raw `f64` bits.
    pub measured_range_rate_m_s: F64Bits,
    /// One-sigma range-rate uncertainty, stored as raw `f64` bits.
    pub sigma_m_s: F64Bits,
    /// Satellite clock drift as a range-rate bias, stored as raw `f64` bits.
    pub satellite_clock_drift_m_s: F64Bits,
}

impl SerializableTightRangeRateObservation {
    /// Convert a native range-rate observation into the serialized form.
    pub fn from_native(observation: TightRangeRateObservation) -> Self {
        Self {
            measured_range_rate_m_s: F64Bits::from_f64(observation.measured_range_rate_m_s),
            sigma_m_s: F64Bits::from_f64(observation.sigma_m_s),
            satellite_clock_drift_m_s: F64Bits::from_f64(observation.satellite_clock_drift_m_s),
        }
    }

    /// Convert the serialized range-rate observation back to native values.
    pub fn to_native(self) -> TightRangeRateObservation {
        TightRangeRateObservation {
            measured_range_rate_m_s: self.measured_range_rate_m_s.to_f64(),
            sigma_m_s: self.sigma_m_s.to_f64(),
            satellite_clock_drift_m_s: self.satellite_clock_drift_m_s.to_f64(),
        }
    }
}

/// Serializable tight carrier-phase observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableTightCarrierPhaseObservation {
    /// Carrier phase converted to range units, stored as raw `f64` bits.
    pub phase_range_m: F64Bits,
    /// One-sigma carrier-phase range uncertainty, stored as raw `f64` bits.
    pub sigma_m: F64Bits,
    /// Current float ambiguity estimate, stored as raw `f64` bits.
    pub float_ambiguity_m: F64Bits,
}

impl SerializableTightCarrierPhaseObservation {
    /// Convert a native carrier-phase observation into the serialized form.
    pub fn from_native(observation: TightCarrierPhaseObservation) -> Self {
        Self {
            phase_range_m: F64Bits::from_f64(observation.phase_range_m),
            sigma_m: F64Bits::from_f64(observation.sigma_m),
            float_ambiguity_m: F64Bits::from_f64(observation.float_ambiguity_m),
        }
    }

    /// Convert the serialized carrier-phase observation back to native values.
    pub fn to_native(self) -> TightCarrierPhaseObservation {
        TightCarrierPhaseObservation {
            phase_range_m: self.phase_range_m.to_f64(),
            sigma_m: self.sigma_m.to_f64(),
            float_ambiguity_m: self.float_ambiguity_m.to_f64(),
        }
    }
}

/// Serializable tight raw GNSS observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableTightGnssObservation {
    /// Satellite identifier.
    pub satellite_id: SerializableSatelliteId,
    /// Measured code pseudorange, stored as raw `f64` bits.
    pub pseudorange_m: F64Bits,
    /// One-sigma pseudorange uncertainty, stored as raw `f64` bits.
    pub pseudorange_sigma_m: F64Bits,
    /// Optional Doppler-derived range-rate row.
    pub range_rate: Option<SerializableTightRangeRateObservation>,
    /// Optional carrier-phase row.
    pub carrier_phase: Option<SerializableTightCarrierPhaseObservation>,
    /// Ionospheric group delay correction for code, stored as raw `f64` bits.
    pub ionosphere_delay_m: F64Bits,
    /// Tropospheric delay correction, stored as raw `f64` bits.
    pub troposphere_delay_m: F64Bits,
}

impl SerializableTightGnssObservation {
    /// Convert a native tight observation into the serialized form.
    pub fn from_native(observation: TightGnssObservation) -> Self {
        Self {
            satellite_id: SerializableSatelliteId::from_native(observation.satellite_id),
            pseudorange_m: F64Bits::from_f64(observation.pseudorange_m),
            pseudorange_sigma_m: F64Bits::from_f64(observation.pseudorange_sigma_m),
            range_rate: observation
                .range_rate
                .map(SerializableTightRangeRateObservation::from_native),
            carrier_phase: observation
                .carrier_phase
                .map(SerializableTightCarrierPhaseObservation::from_native),
            ionosphere_delay_m: F64Bits::from_f64(observation.ionosphere_delay_m),
            troposphere_delay_m: F64Bits::from_f64(observation.troposphere_delay_m),
        }
    }

    /// Convert the serialized tight observation back to native values.
    pub fn to_native(self) -> Result<TightGnssObservation, FusionStateCodecError> {
        let observation = TightGnssObservation {
            satellite_id: self.satellite_id.to_native()?,
            pseudorange_m: self.pseudorange_m.to_f64(),
            pseudorange_sigma_m: self.pseudorange_sigma_m.to_f64(),
            range_rate: self
                .range_rate
                .map(SerializableTightRangeRateObservation::to_native),
            carrier_phase: self
                .carrier_phase
                .map(SerializableTightCarrierPhaseObservation::to_native),
            ionosphere_delay_m: self.ionosphere_delay_m.to_f64(),
            troposphere_delay_m: self.troposphere_delay_m.to_f64(),
        };
        observation.validate().map_err(invalid_state)?;
        Ok(observation)
    }
}

/// Serializable tight raw GNSS epoch.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableTightGnssEpoch {
    /// Measurement epoch in seconds since J2000, stored as raw `f64` bits.
    pub t_j2000_s: F64Bits,
    /// Satellite observations at the epoch.
    pub observations: Vec<SerializableTightGnssObservation>,
}

impl SerializableTightGnssEpoch {
    /// Convert a native tight epoch into the serialized form.
    pub fn from_native(epoch: &TightGnssEpoch) -> Self {
        Self {
            t_j2000_s: F64Bits::from_f64(epoch.t_j2000_s),
            observations: epoch
                .observations
                .iter()
                .copied()
                .map(SerializableTightGnssObservation::from_native)
                .collect(),
        }
    }

    /// Convert the serialized tight epoch back to native values.
    pub fn to_native(&self) -> Result<TightGnssEpoch, FusionStateCodecError> {
        let observations = self
            .observations
            .iter()
            .copied()
            .map(SerializableTightGnssObservation::to_native)
            .collect::<Result<Vec<_>, _>>()?;
        let epoch = TightGnssEpoch {
            t_j2000_s: self.t_j2000_s.to_f64(),
            observations,
        };
        epoch.validate().map_err(invalid_state)?;
        Ok(epoch)
    }
}

/// Serializable retained GNSS measurement.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SerializableStoredGnssMeasurement {
    /// Retained loose GNSS fix.
    Loose(SerializableLooseMeasurement),
    /// Retained tight raw GNSS epoch.
    Tight(SerializableTightGnssEpoch),
}

impl SerializableStoredGnssMeasurement {
    /// Convert a retained GNSS measurement into the serialized form.
    fn from_native(measurement: &StoredGnssMeasurement) -> Result<Self, FusionStateCodecError> {
        match measurement {
            StoredGnssMeasurement::Loose(measurement) => Ok(Self::Loose(
                SerializableLooseMeasurement::from_native(measurement)?,
            )),
            StoredGnssMeasurement::Tight(epoch) => {
                Ok(Self::Tight(SerializableTightGnssEpoch::from_native(epoch)))
            }
        }
    }

    /// Convert the serialized GNSS measurement back to native values.
    fn to_native(&self) -> Result<StoredGnssMeasurement, FusionStateCodecError> {
        match self {
            Self::Loose(measurement) => Ok(StoredGnssMeasurement::Loose(measurement.to_native()?)),
            Self::Tight(epoch) => Ok(StoredGnssMeasurement::Tight(epoch.to_native()?)),
        }
    }
}

/// Serializable retained filter checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableStoredCheckpoint {
    /// Checkpoint epoch in seconds since J2000, stored as raw `f64` bits.
    pub t_j2000_s: F64Bits,
    /// Closed-loop filter snapshot at the checkpoint.
    pub snapshot: Box<SerializableFusionSnapshot>,
}

impl SerializableStoredCheckpoint {
    /// Convert a retained checkpoint into the serialized form.
    fn from_native(checkpoint: &StoredCheckpoint) -> Self {
        Self {
            t_j2000_s: F64Bits::from_f64(checkpoint.t_j2000_s),
            snapshot: Box::new(SerializableFusionSnapshot::from_snapshot(
                &checkpoint.snapshot,
            )),
        }
    }

    /// Convert the serialized checkpoint back to native values.
    fn to_native(&self) -> Result<StoredCheckpoint, FusionStateCodecError> {
        let snapshot = self.snapshot.to_snapshot()?;
        let checkpoint = StoredCheckpoint {
            t_j2000_s: self.t_j2000_s.to_f64(),
            snapshot,
        };
        if checkpoint.t_j2000_s != checkpoint.snapshot.state.nominal.t_j2000_s {
            return Err(FusionStateCodecError::InvalidState {
                reason: "checkpoint epoch must match snapshot".to_string(),
            });
        }
        Ok(checkpoint)
    }
}

/// Stable serializable fusion snapshot without retained replay history.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableFusionSnapshot {
    /// Closed-loop INS state and covariance.
    pub state: SerializableInsFilterState,
    /// Last propagated body angular rate relative to ECEF, stored as raw `f64` bits.
    pub last_body_rate_wrt_ecef_rps: [F64Bits; 3],
    /// Recent detector samples used by stationary-update gating.
    #[serde(default)]
    pub stationarity_window: Vec<SerializableStationarityDetectorSample>,
    /// Epoch of the last stationary pseudo-measurement, if any.
    #[serde(default)]
    pub last_stationary_update_t_j2000_s: Option<F64Bits>,
    /// Epoch of the last non-holonomic pseudo-measurement, if any.
    #[serde(default)]
    pub last_non_holonomic_update_t_j2000_s: Option<F64Bits>,
    /// Tight receiver-clock augmentation and augmented covariance.
    pub tight: SerializableTightFilterState,
}

impl SerializableFusionSnapshot {
    /// Convert a native fusion snapshot into the stable serialized form.
    pub fn from_snapshot(snapshot: &InertialFilterSnapshot) -> Self {
        Self {
            state: SerializableInsFilterState::from_native(&snapshot.state),
            last_body_rate_wrt_ecef_rps: bits3(snapshot.last_body_rate_wrt_ecef_rps),
            stationarity_window: snapshot
                .stationarity_window
                .iter()
                .copied()
                .map(SerializableStationarityDetectorSample::from_native)
                .collect(),
            last_stationary_update_t_j2000_s: snapshot
                .last_stationary_update_t_j2000_s
                .map(F64Bits::from_f64),
            last_non_holonomic_update_t_j2000_s: snapshot
                .last_non_holonomic_update_t_j2000_s
                .map(F64Bits::from_f64),
            tight: SerializableTightFilterState::from_native(&snapshot.tight),
        }
    }

    /// Convert the stable serialized form back to a validated native snapshot.
    pub fn to_snapshot(&self) -> Result<InertialFilterSnapshot, FusionStateCodecError> {
        let state = self.state.to_native()?;
        let last_body_rate_wrt_ecef_rps = f643(self.last_body_rate_wrt_ecef_rps);
        validate_finite_slice(&last_body_rate_wrt_ecef_rps, "last_body_rate_wrt_ecef_rps")
            .map_err(invalid_state)?;
        let stationarity_window = stationarity_window_to_native(&self.stationarity_window)?;
        let last_stationary_update_t_j2000_s = optional_epoch_to_native(
            self.last_stationary_update_t_j2000_s,
            "last_stationary_update_t_j2000_s",
        )?;
        let last_non_holonomic_update_t_j2000_s = optional_epoch_to_native(
            self.last_non_holonomic_update_t_j2000_s,
            "last_non_holonomic_update_t_j2000_s",
        )?;
        let tight = self.tight.to_native(state.dimension())?;
        Ok(InertialFilterSnapshot {
            state,
            last_body_rate_wrt_ecef_rps,
            stationarity_window,
            last_stationary_update_t_j2000_s,
            last_non_holonomic_update_t_j2000_s,
            tight,
        })
    }
}

fn stationarity_window_to_native(
    samples: &[SerializableStationarityDetectorSample],
) -> Result<Vec<StationarityDetectorSnapshotSample>, FusionStateCodecError> {
    let out = samples
        .iter()
        .copied()
        .map(SerializableStationarityDetectorSample::to_native)
        .collect::<Vec<_>>();
    for sample in &out {
        validate_finite_slice(
            &[
                sample.specific_force_norm_error_mps2,
                sample.body_rate_wrt_ecef_norm_rps,
            ],
            "stationarity_window",
        )
        .map_err(invalid_state)?;
    }
    Ok(out)
}

fn optional_epoch_to_native(
    value: Option<F64Bits>,
    field: &'static str,
) -> Result<Option<f64>, FusionStateCodecError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let value = value.to_f64();
    validate_finite_slice(&[value], field).map_err(invalid_state)?;
    Ok(Some(value))
}

/// Serializable retained time-sync replay history.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableTimeSyncHistory {
    /// Retained-history capacity settings.
    pub config: SerializableTimeSyncHistoryConfig,
    /// Retained IMU samples for replay.
    pub imu_samples: Vec<SerializableStoredImuSample>,
    /// Retained filter checkpoints at GNSS epochs.
    pub checkpoints: Vec<SerializableStoredCheckpoint>,
    /// Retained GNSS measurements for replay ordering.
    pub measurements: Vec<SerializableStoredGnssMeasurement>,
}

impl SerializableTimeSyncHistory {
    /// Convert native retained history into the serialized form.
    fn from_native(history: &TimeSyncHistorySnapshot) -> Result<Self, FusionStateCodecError> {
        Ok(Self {
            config: SerializableTimeSyncHistoryConfig::from_native(history.config)?,
            imu_samples: history
                .imu_samples
                .iter()
                .copied()
                .map(SerializableStoredImuSample::from_native)
                .collect(),
            checkpoints: history
                .checkpoints
                .iter()
                .map(SerializableStoredCheckpoint::from_native)
                .collect(),
            measurements: history
                .measurements
                .iter()
                .map(SerializableStoredGnssMeasurement::from_native)
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    /// Convert the serialized retained history back to native values.
    fn to_native(&self) -> Result<TimeSyncHistorySnapshot, FusionStateCodecError> {
        let snapshot = TimeSyncHistorySnapshot {
            config: self.config.to_native()?,
            imu_samples: self
                .imu_samples
                .iter()
                .copied()
                .map(SerializableStoredImuSample::to_native)
                .collect(),
            checkpoints: self
                .checkpoints
                .iter()
                .map(SerializableStoredCheckpoint::to_native)
                .collect::<Result<Vec<_>, _>>()?,
            measurements: self
                .measurements
                .iter()
                .map(SerializableStoredGnssMeasurement::to_native)
                .collect::<Result<Vec<_>, _>>()?,
        };
        validate_history_by_restore(snapshot)
    }
}

/// Stable serializable fusion checkpoint.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SerializableFusionState {
    /// Serialization schema version.
    pub version: u16,
    /// Closed-loop INS state and covariance.
    pub state: SerializableInsFilterState,
    /// Last propagated body angular rate relative to ECEF, stored as raw `f64` bits.
    pub last_body_rate_wrt_ecef_rps: [F64Bits; 3],
    /// Recent detector samples used by stationary-update gating.
    #[serde(default)]
    pub stationarity_window: Vec<SerializableStationarityDetectorSample>,
    /// Epoch of the last stationary pseudo-measurement, if any.
    #[serde(default)]
    pub last_stationary_update_t_j2000_s: Option<F64Bits>,
    /// Epoch of the last non-holonomic pseudo-measurement, if any.
    #[serde(default)]
    pub last_non_holonomic_update_t_j2000_s: Option<F64Bits>,
    /// Tight receiver-clock augmentation and augmented covariance.
    pub tight: SerializableTightFilterState,
    /// Retained time-sync replay history.
    pub time_sync: SerializableTimeSyncHistory,
}

impl SerializableFusionState {
    /// Convert a native fusion snapshot into the stable serialized form.
    pub fn from_snapshot(snapshot: &InertialFilterSnapshot) -> Self {
        let history = TimeSyncHistorySnapshot::from_filter_snapshot(snapshot.clone());
        let time_sync =
            SerializableTimeSyncHistory::from_native(&history).expect("default history encodes");
        Self {
            version: FUSION_STATE_CODEC_VERSION,
            state: SerializableInsFilterState::from_native(&snapshot.state),
            last_body_rate_wrt_ecef_rps: bits3(snapshot.last_body_rate_wrt_ecef_rps),
            stationarity_window: snapshot
                .stationarity_window
                .iter()
                .copied()
                .map(SerializableStationarityDetectorSample::from_native)
                .collect(),
            last_stationary_update_t_j2000_s: snapshot
                .last_stationary_update_t_j2000_s
                .map(F64Bits::from_f64),
            last_non_holonomic_update_t_j2000_s: snapshot
                .last_non_holonomic_update_t_j2000_s
                .map(F64Bits::from_f64),
            tight: SerializableTightFilterState::from_native(&snapshot.tight),
            time_sync,
        }
    }

    /// Convert a live fusion filter into the stable serialized form.
    pub fn from_filter(filter: &InertialFilter) -> Result<Self, FusionStateCodecError> {
        let snapshot = filter.snapshot();
        Ok(Self {
            version: FUSION_STATE_CODEC_VERSION,
            state: SerializableInsFilterState::from_native(&snapshot.state),
            last_body_rate_wrt_ecef_rps: bits3(snapshot.last_body_rate_wrt_ecef_rps),
            stationarity_window: snapshot
                .stationarity_window
                .iter()
                .copied()
                .map(SerializableStationarityDetectorSample::from_native)
                .collect(),
            last_stationary_update_t_j2000_s: snapshot
                .last_stationary_update_t_j2000_s
                .map(F64Bits::from_f64),
            last_non_holonomic_update_t_j2000_s: snapshot
                .last_non_holonomic_update_t_j2000_s
                .map(F64Bits::from_f64),
            tight: SerializableTightFilterState::from_native(&snapshot.tight),
            time_sync: SerializableTimeSyncHistory::from_native(
                &filter.time_sync.snapshot_history(),
            )?,
        })
    }

    /// Convert the stable serialized form back to a validated native snapshot.
    pub fn to_snapshot(&self) -> Result<InertialFilterSnapshot, FusionStateCodecError> {
        self.validate_version()?;
        let state = self.state.to_native()?;
        let last_body_rate_wrt_ecef_rps = f643(self.last_body_rate_wrt_ecef_rps);
        validate_finite_slice(&last_body_rate_wrt_ecef_rps, "last_body_rate_wrt_ecef_rps")
            .map_err(invalid_state)?;
        let stationarity_window = stationarity_window_to_native(&self.stationarity_window)?;
        let last_stationary_update_t_j2000_s = optional_epoch_to_native(
            self.last_stationary_update_t_j2000_s,
            "last_stationary_update_t_j2000_s",
        )?;
        let last_non_holonomic_update_t_j2000_s = optional_epoch_to_native(
            self.last_non_holonomic_update_t_j2000_s,
            "last_non_holonomic_update_t_j2000_s",
        )?;
        let tight = self.tight.to_native(state.dimension())?;
        Ok(InertialFilterSnapshot {
            state,
            last_body_rate_wrt_ecef_rps,
            stationarity_window,
            last_stationary_update_t_j2000_s,
            last_non_holonomic_update_t_j2000_s,
            tight,
        })
    }

    /// Convert the stable serialized form back to retained replay history.
    fn to_time_sync_history(&self) -> Result<TimeSyncHistorySnapshot, FusionStateCodecError> {
        self.validate_version()?;
        self.time_sync.to_native()
    }

    /// Encode the checkpoint with a binary magic, version, and checksum.
    pub fn encode_versioned(&self) -> Result<Vec<u8>, FusionStateCodecError> {
        self.validate_current_version()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FUSION_STATE_MAGIC);
        write_u16(&mut bytes, self.version);
        write_layout(&mut bytes, self.state.layout);
        write_nav(&mut bytes, &self.state.nominal);
        write_f64_vec(&mut bytes, &self.state.error_state)?;
        write_f64_matrix(&mut bytes, &self.state.covariance)?;
        write_f64_array(&mut bytes, &self.state.accel_scale_factor);
        write_f64_array(&mut bytes, &self.state.gyro_scale_factor);
        write_f64_array(&mut bytes, &self.last_body_rate_wrt_ecef_rps);
        write_stationarity_window(&mut bytes, &self.stationarity_window)?;
        write_optional_f64(&mut bytes, self.last_stationary_update_t_j2000_s);
        write_optional_f64(&mut bytes, self.last_non_holonomic_update_t_j2000_s);
        write_f64(&mut bytes, self.tight.clock_bias_m);
        write_f64(&mut bytes, self.tight.clock_drift_m_s);
        write_f64_matrix(&mut bytes, &self.tight.augmented_covariance)?;
        write_time_sync_history(&mut bytes, &self.time_sync)?;
        let checksum = fnv1a64(&bytes);
        write_u64(&mut bytes, checksum);
        Ok(bytes)
    }

    /// Decode a binary checkpoint produced by [`Self::encode_versioned`].
    pub fn decode_versioned(bytes: &[u8]) -> Result<Self, FusionStateCodecError> {
        let minimum = FUSION_STATE_MAGIC.len() + 2 + 8;
        if bytes.len() < minimum {
            return Err(FusionStateCodecError::Truncated {
                offset: 0,
                needed: minimum,
                actual: bytes.len(),
            });
        }
        if bytes[..FUSION_STATE_MAGIC.len()] != FUSION_STATE_MAGIC {
            return Err(FusionStateCodecError::InvalidMagic);
        }
        let checksum_offset = bytes.len() - 8;
        let expected = read_u64_at(bytes, checksum_offset)?;
        let found = fnv1a64(&bytes[..checksum_offset]);
        if expected != found {
            return Err(FusionStateCodecError::Checksum { expected, found });
        }

        let mut cursor = FUSION_STATE_MAGIC.len();
        let version = read_u16(bytes, &mut cursor, checksum_offset)?;
        if !is_supported_fusion_state_codec_version(version) {
            return Err(FusionStateCodecError::UnsupportedVersion { version });
        }
        let layout = read_layout(bytes, &mut cursor, checksum_offset)?;
        let nominal = read_nav(bytes, &mut cursor, checksum_offset)?;
        let error_state = read_f64_vec(bytes, &mut cursor, checksum_offset)?;
        let covariance = read_f64_matrix(bytes, &mut cursor, checksum_offset)?;
        let accel_scale_factor = read_f64_array(bytes, &mut cursor, checksum_offset)?;
        let gyro_scale_factor = read_f64_array(bytes, &mut cursor, checksum_offset)?;
        let last_body_rate_wrt_ecef_rps = read_f64_array(bytes, &mut cursor, checksum_offset)?;
        let stationarity_window = if version >= 3 {
            read_stationarity_window(bytes, &mut cursor, checksum_offset)?
        } else {
            Vec::new()
        };
        let (last_stationary_update_t_j2000_s, last_non_holonomic_update_t_j2000_s) =
            if version >= 4 {
                (
                    read_optional_f64(bytes, &mut cursor, checksum_offset)?,
                    read_optional_f64(bytes, &mut cursor, checksum_offset)?,
                )
            } else {
                (None, None)
            };
        let clock_bias_m = read_f64(bytes, &mut cursor, checksum_offset)?;
        let clock_drift_m_s = read_f64(bytes, &mut cursor, checksum_offset)?;
        let augmented_covariance = read_f64_matrix(bytes, &mut cursor, checksum_offset)?;
        let time_sync = read_time_sync_history(bytes, &mut cursor, checksum_offset, version)?;
        if cursor != checksum_offset {
            return Err(FusionStateCodecError::TrailingBytes {
                remaining: checksum_offset - cursor,
            });
        }
        let state = Self {
            version,
            state: SerializableInsFilterState {
                layout,
                nominal,
                error_state,
                covariance,
                accel_scale_factor,
                gyro_scale_factor,
            },
            last_body_rate_wrt_ecef_rps,
            stationarity_window,
            last_stationary_update_t_j2000_s,
            last_non_holonomic_update_t_j2000_s,
            tight: SerializableTightFilterState {
                clock_bias_m,
                clock_drift_m_s,
                augmented_covariance,
            },
            time_sync,
        };
        state.to_snapshot()?;
        state.to_time_sync_history()?;
        Ok(state)
    }

    /// Serialize this checkpoint to JSON using the serde representation.
    pub fn to_json_string(&self) -> Result<String, FusionStateCodecError> {
        self.validate_current_version()?;
        serde_json::to_string(self).map_err(|error| FusionStateCodecError::Json {
            message: error.to_string(),
        })
    }

    /// Parse a JSON checkpoint using the serde representation.
    pub fn from_json_str(text: &str) -> Result<Self, FusionStateCodecError> {
        let state: Self =
            serde_json::from_str(text).map_err(|error| FusionStateCodecError::Json {
                message: error.to_string(),
            })?;
        state.to_snapshot()?;
        state.to_time_sync_history()?;
        Ok(state)
    }

    fn validate_version(&self) -> Result<(), FusionStateCodecError> {
        if is_supported_fusion_state_codec_version(self.version) {
            Ok(())
        } else {
            Err(FusionStateCodecError::UnsupportedVersion {
                version: self.version,
            })
        }
    }

    fn validate_current_version(&self) -> Result<(), FusionStateCodecError> {
        if self.version == FUSION_STATE_CODEC_VERSION {
            Ok(())
        } else {
            Err(FusionStateCodecError::UnsupportedVersion {
                version: self.version,
            })
        }
    }
}

fn is_supported_fusion_state_codec_version(version: u16) -> bool {
    (MIN_SUPPORTED_FUSION_STATE_CODEC_VERSION..=FUSION_STATE_CODEC_VERSION).contains(&version)
}

impl InertialFilterSnapshot {
    /// Convert this snapshot into a stable serializable checkpoint.
    pub fn to_serializable_fusion_state(&self) -> SerializableFusionState {
        SerializableFusionState::from_snapshot(self)
    }

    /// Encode this snapshot with the versioned binary fusion codec.
    pub fn encode_fusion_state(&self) -> Result<Vec<u8>, FusionStateCodecError> {
        self.to_serializable_fusion_state().encode_versioned()
    }

    /// Decode a snapshot from the versioned binary fusion codec.
    pub fn decode_fusion_state(bytes: &[u8]) -> Result<Self, FusionStateCodecError> {
        SerializableFusionState::decode_versioned(bytes)?.to_snapshot()
    }
}

impl InertialFilter {
    /// Return the current filter checkpoint in the stable serializable form.
    pub fn serializable_state(&self) -> Result<SerializableFusionState, FusionStateCodecError> {
        SerializableFusionState::from_filter(self)
    }

    /// Encode the current filter checkpoint with the versioned binary codec.
    pub fn encode_state(&self) -> Result<Vec<u8>, FusionStateCodecError> {
        SerializableFusionState::from_filter(self)?.encode_versioned()
    }

    /// Restore this filter from a stable serializable checkpoint.
    pub fn restore_serializable_state(
        &mut self,
        state: &SerializableFusionState,
    ) -> Result<(), FusionStateCodecError> {
        let snapshot = state.to_snapshot()?;
        let history = state.to_time_sync_history()?;
        self.restore_snapshot(&snapshot).map_err(invalid_state)?;
        self.time_sync
            .restore_history(history)
            .map_err(invalid_state)
    }

    /// Restore this filter from the versioned binary fusion codec.
    pub fn restore_encoded_state(&mut self, bytes: &[u8]) -> Result<(), FusionStateCodecError> {
        let state = SerializableFusionState::decode_versioned(bytes)?;
        self.restore_serializable_state(&state)
    }
}

/// Errors returned by the versioned fusion-state codec.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FusionStateCodecError {
    /// The binary payload did not begin with the fusion-state magic bytes.
    #[error("fusion state payload has invalid magic")]
    InvalidMagic,
    /// The payload schema version is not supported by this decoder.
    #[error("fusion state version {version} is not supported")]
    UnsupportedVersion {
        /// Version tag found in the payload.
        version: u16,
    },
    /// The payload ended before a complete field could be read.
    #[error("fusion state payload truncated at {offset}, needed {needed} bytes, got {actual}")]
    Truncated {
        /// Byte offset of the attempted read.
        offset: usize,
        /// Number of bytes required for the read or minimum payload.
        needed: usize,
        /// Number of bytes available in the payload or read window.
        actual: usize,
    },
    /// The checksum did not match the decoded payload bytes.
    #[error("fusion state checksum expected {expected:#x} but found {found:#x}")]
    Checksum {
        /// Checksum stored in the payload.
        expected: u64,
        /// Checksum computed from the payload.
        found: u64,
    },
    /// The payload contained bytes after the versioned fields.
    #[error("fusion state payload has {remaining} trailing bytes")]
    TrailingBytes {
        /// Number of bytes not consumed before the checksum.
        remaining: usize,
    },
    /// The payload decoded but did not validate as a native fusion state.
    #[error("invalid fusion state payload: {reason}")]
    InvalidState {
        /// Stable validation reason.
        reason: String,
    },
    /// JSON serde encoding or decoding failed.
    #[error("fusion state JSON error: {message}")]
    Json {
        /// Serde error text.
        message: String,
    },
}

fn invalid_state(error: impl core::fmt::Display) -> FusionStateCodecError {
    FusionStateCodecError::InvalidState {
        reason: error.to_string(),
    }
}

fn checked_u32(value: usize) -> Result<u32, FusionStateCodecError> {
    u32::try_from(value).map_err(|_| FusionStateCodecError::InvalidState {
        reason: "length exceeds u32".to_string(),
    })
}

fn validate_history_by_restore(
    snapshot: TimeSyncHistorySnapshot,
) -> Result<TimeSyncHistorySnapshot, FusionStateCodecError> {
    snapshot.validate().map_err(invalid_state)?;
    Ok(snapshot)
}

fn bits3(values: [f64; 3]) -> [F64Bits; 3] {
    values.map(F64Bits::from_f64)
}

fn f643(values: [F64Bits; 3]) -> [f64; 3] {
    values.map(F64Bits::to_f64)
}

fn bits3x3(values: [[f64; 3]; 3]) -> [[F64Bits; 3]; 3] {
    values.map(bits3)
}

fn f643x3(values: [[F64Bits; 3]; 3]) -> [[f64; 3]; 3] {
    values.map(f643)
}

fn bits_slice(values: &[f64]) -> Vec<F64Bits> {
    values.iter().copied().map(F64Bits::from_f64).collect()
}

fn f64_vec(values: &[F64Bits]) -> Vec<f64> {
    values.iter().copied().map(F64Bits::to_f64).collect()
}

fn bits_matrix(values: &[Vec<f64>]) -> Vec<Vec<F64Bits>> {
    values.iter().map(|row| bits_slice(row)).collect()
}

fn f64_matrix(values: &[Vec<F64Bits>]) -> Vec<Vec<f64>> {
    values.iter().map(|row| f64_vec(row)).collect()
}

fn write_nav(bytes: &mut Vec<u8>, state: &SerializableNavState) {
    write_f64(bytes, state.t_j2000_s);
    write_f64_array(bytes, &state.position_ecef_m);
    write_f64_array(bytes, &state.velocity_ecef_mps);
    for row in &state.attitude_body_to_ecef {
        write_f64_array(bytes, row);
    }
    write_f64_array(bytes, &state.accel_bias_mps2);
    write_f64_array(bytes, &state.gyro_bias_rps);
}

fn read_nav(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableNavState, FusionStateCodecError> {
    Ok(SerializableNavState {
        t_j2000_s: read_f64(bytes, cursor, limit)?,
        position_ecef_m: read_f64_array(bytes, cursor, limit)?,
        velocity_ecef_mps: read_f64_array(bytes, cursor, limit)?,
        attitude_body_to_ecef: [
            read_f64_array(bytes, cursor, limit)?,
            read_f64_array(bytes, cursor, limit)?,
            read_f64_array(bytes, cursor, limit)?,
        ],
        accel_bias_mps2: read_f64_array(bytes, cursor, limit)?,
        gyro_bias_rps: read_f64_array(bytes, cursor, limit)?,
    })
}

fn write_layout(bytes: &mut Vec<u8>, layout: SerializableErrorStateLayout) {
    bytes.push(match layout {
        SerializableErrorStateLayout::Fifteen => 15,
        SerializableErrorStateLayout::TwentyOne => 21,
    });
}

fn read_layout(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableErrorStateLayout, FusionStateCodecError> {
    match read_u8(bytes, cursor, limit)? {
        15 => Ok(SerializableErrorStateLayout::Fifteen),
        21 => Ok(SerializableErrorStateLayout::TwentyOne),
        _ => Err(FusionStateCodecError::InvalidState {
            reason: "invalid error-state layout tag".to_string(),
        }),
    }
}

fn write_time_sync_history(
    bytes: &mut Vec<u8>,
    history: &SerializableTimeSyncHistory,
) -> Result<(), FusionStateCodecError> {
    write_u32_checked(bytes, history.config.imu_capacity as usize)?;
    write_u32_checked(bytes, history.config.checkpoint_capacity as usize)?;
    write_u32_checked(bytes, history.imu_samples.len())?;
    for sample in &history.imu_samples {
        write_stored_imu_sample(bytes, sample);
    }
    write_u32_checked(bytes, history.checkpoints.len())?;
    for checkpoint in &history.checkpoints {
        write_stored_checkpoint(bytes, checkpoint)?;
    }
    write_u32_checked(bytes, history.measurements.len())?;
    for measurement in &history.measurements {
        write_stored_measurement(bytes, measurement)?;
    }
    Ok(())
}

fn read_time_sync_history(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
    version: u16,
) -> Result<SerializableTimeSyncHistory, FusionStateCodecError> {
    let config = SerializableTimeSyncHistoryConfig {
        imu_capacity: read_u32(bytes, cursor, limit)?,
        checkpoint_capacity: read_u32(bytes, cursor, limit)?,
    };
    let imu_len = read_len(bytes, cursor, limit, 1, "imu_samples")?;
    let mut imu_samples = Vec::with_capacity(imu_len);
    for _ in 0..imu_len {
        imu_samples.push(read_stored_imu_sample(bytes, cursor, limit)?);
    }
    let checkpoint_len = read_len(bytes, cursor, limit, 1, "checkpoints")?;
    let mut checkpoints = Vec::with_capacity(checkpoint_len);
    for _ in 0..checkpoint_len {
        checkpoints.push(read_stored_checkpoint(bytes, cursor, limit, version)?);
    }
    let measurement_len = read_len(bytes, cursor, limit, 1, "gnss_measurements")?;
    let mut measurements = Vec::with_capacity(measurement_len);
    for _ in 0..measurement_len {
        measurements.push(read_stored_measurement(bytes, cursor, limit)?);
    }
    let history = SerializableTimeSyncHistory {
        config,
        imu_samples,
        checkpoints,
        measurements,
    };
    history.to_native()?;
    Ok(history)
}

fn write_stored_imu_sample(bytes: &mut Vec<u8>, sample: &SerializableStoredImuSample) {
    write_f64(bytes, sample.previous_t_j2000_s);
    write_imu_sample(bytes, &sample.sample);
    match sample.previous_rate {
        Some(endpoint) => {
            write_bool(bytes, true);
            write_rate_endpoint(bytes, endpoint);
        }
        None => write_bool(bytes, false),
    }
}

fn read_stored_imu_sample(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableStoredImuSample, FusionStateCodecError> {
    let previous_t_j2000_s = read_f64(bytes, cursor, limit)?;
    let sample = read_imu_sample(bytes, cursor, limit)?;
    let previous_rate = if read_bool(bytes, cursor, limit)? {
        Some(read_rate_endpoint(bytes, cursor, limit)?)
    } else {
        None
    };
    Ok(SerializableStoredImuSample {
        previous_t_j2000_s,
        sample,
        previous_rate,
    })
}

fn write_imu_sample(bytes: &mut Vec<u8>, sample: &SerializableImuSample) {
    write_f64(bytes, sample.t_j2000_s);
    match sample.kind {
        SerializableImuSampleKind::Rate {
            specific_force_mps2,
            angular_rate_rps,
        } => {
            bytes.push(0);
            write_f64_array(bytes, &specific_force_mps2);
            write_f64_array(bytes, &angular_rate_rps);
        }
        SerializableImuSampleKind::Increment {
            delta_velocity_mps,
            delta_theta_rad,
            dt_s,
        } => {
            bytes.push(1);
            write_f64_array(bytes, &delta_velocity_mps);
            write_f64_array(bytes, &delta_theta_rad);
            write_f64(bytes, dt_s);
        }
    }
}

fn read_imu_sample(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableImuSample, FusionStateCodecError> {
    let t_j2000_s = read_f64(bytes, cursor, limit)?;
    let kind = match read_u8(bytes, cursor, limit)? {
        0 => SerializableImuSampleKind::Rate {
            specific_force_mps2: read_f64_array(bytes, cursor, limit)?,
            angular_rate_rps: read_f64_array(bytes, cursor, limit)?,
        },
        1 => SerializableImuSampleKind::Increment {
            delta_velocity_mps: read_f64_array(bytes, cursor, limit)?,
            delta_theta_rad: read_f64_array(bytes, cursor, limit)?,
            dt_s: read_f64(bytes, cursor, limit)?,
        },
        _ => {
            return Err(FusionStateCodecError::InvalidState {
                reason: "invalid IMU sample kind tag".to_string(),
            });
        }
    };
    Ok(SerializableImuSample { t_j2000_s, kind })
}

fn write_rate_endpoint(bytes: &mut Vec<u8>, endpoint: SerializableRateEndpoint) {
    write_f64(bytes, endpoint.t_j2000_s);
    write_f64_array(bytes, &endpoint.specific_force_mps2);
    write_f64_array(bytes, &endpoint.angular_rate_rps);
}

fn read_rate_endpoint(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableRateEndpoint, FusionStateCodecError> {
    Ok(SerializableRateEndpoint {
        t_j2000_s: read_f64(bytes, cursor, limit)?,
        specific_force_mps2: read_f64_array(bytes, cursor, limit)?,
        angular_rate_rps: read_f64_array(bytes, cursor, limit)?,
    })
}

fn write_stored_checkpoint(
    bytes: &mut Vec<u8>,
    checkpoint: &SerializableStoredCheckpoint,
) -> Result<(), FusionStateCodecError> {
    write_f64(bytes, checkpoint.t_j2000_s);
    write_fusion_snapshot(bytes, checkpoint.snapshot.as_ref())
}

fn read_stored_checkpoint(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
    version: u16,
) -> Result<SerializableStoredCheckpoint, FusionStateCodecError> {
    Ok(SerializableStoredCheckpoint {
        t_j2000_s: read_f64(bytes, cursor, limit)?,
        snapshot: Box::new(read_fusion_snapshot(bytes, cursor, limit, version)?),
    })
}

fn write_fusion_snapshot(
    bytes: &mut Vec<u8>,
    snapshot: &SerializableFusionSnapshot,
) -> Result<(), FusionStateCodecError> {
    write_layout(bytes, snapshot.state.layout);
    write_nav(bytes, &snapshot.state.nominal);
    write_f64_vec(bytes, &snapshot.state.error_state)?;
    write_f64_matrix(bytes, &snapshot.state.covariance)?;
    write_f64_array(bytes, &snapshot.state.accel_scale_factor);
    write_f64_array(bytes, &snapshot.state.gyro_scale_factor);
    write_f64_array(bytes, &snapshot.last_body_rate_wrt_ecef_rps);
    write_stationarity_window(bytes, &snapshot.stationarity_window)?;
    write_optional_f64(bytes, snapshot.last_stationary_update_t_j2000_s);
    write_optional_f64(bytes, snapshot.last_non_holonomic_update_t_j2000_s);
    write_f64(bytes, snapshot.tight.clock_bias_m);
    write_f64(bytes, snapshot.tight.clock_drift_m_s);
    write_f64_matrix(bytes, &snapshot.tight.augmented_covariance)
}

fn read_fusion_snapshot(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
    version: u16,
) -> Result<SerializableFusionSnapshot, FusionStateCodecError> {
    let state = SerializableInsFilterState {
        layout: read_layout(bytes, cursor, limit)?,
        nominal: read_nav(bytes, cursor, limit)?,
        error_state: read_f64_vec(bytes, cursor, limit)?,
        covariance: read_f64_matrix(bytes, cursor, limit)?,
        accel_scale_factor: read_f64_array(bytes, cursor, limit)?,
        gyro_scale_factor: read_f64_array(bytes, cursor, limit)?,
    };
    let last_body_rate_wrt_ecef_rps = read_f64_array(bytes, cursor, limit)?;
    let stationarity_window = if version >= 3 {
        read_stationarity_window(bytes, cursor, limit)?
    } else {
        Vec::new()
    };
    let (last_stationary_update_t_j2000_s, last_non_holonomic_update_t_j2000_s) = if version >= 4 {
        (
            read_optional_f64(bytes, cursor, limit)?,
            read_optional_f64(bytes, cursor, limit)?,
        )
    } else {
        (None, None)
    };
    Ok(SerializableFusionSnapshot {
        state,
        last_body_rate_wrt_ecef_rps,
        stationarity_window,
        last_stationary_update_t_j2000_s,
        last_non_holonomic_update_t_j2000_s,
        tight: SerializableTightFilterState {
            clock_bias_m: read_f64(bytes, cursor, limit)?,
            clock_drift_m_s: read_f64(bytes, cursor, limit)?,
            augmented_covariance: read_f64_matrix(bytes, cursor, limit)?,
        },
    })
}

fn write_stationarity_window(
    bytes: &mut Vec<u8>,
    samples: &[SerializableStationarityDetectorSample],
) -> Result<(), FusionStateCodecError> {
    write_u32_checked(bytes, samples.len())?;
    for sample in samples {
        write_f64(bytes, sample.specific_force_norm_error_mps2);
        write_f64(bytes, sample.body_rate_wrt_ecef_norm_rps);
    }
    Ok(())
}

fn read_stationarity_window(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<Vec<SerializableStationarityDetectorSample>, FusionStateCodecError> {
    let len = read_len(bytes, cursor, limit, 16, "stationarity_window")?;
    let mut samples = Vec::with_capacity(len);
    for _ in 0..len {
        samples.push(SerializableStationarityDetectorSample {
            specific_force_norm_error_mps2: read_f64(bytes, cursor, limit)?,
            body_rate_wrt_ecef_norm_rps: read_f64(bytes, cursor, limit)?,
        });
    }
    Ok(samples)
}

fn write_optional_f64(bytes: &mut Vec<u8>, value: Option<F64Bits>) {
    match value {
        Some(value) => {
            write_bool(bytes, true);
            write_f64(bytes, value);
        }
        None => write_bool(bytes, false),
    }
}

fn read_optional_f64(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<Option<F64Bits>, FusionStateCodecError> {
    if read_bool(bytes, cursor, limit)? {
        Ok(Some(read_f64(bytes, cursor, limit)?))
    } else {
        Ok(None)
    }
}

fn write_stored_measurement(
    bytes: &mut Vec<u8>,
    measurement: &SerializableStoredGnssMeasurement,
) -> Result<(), FusionStateCodecError> {
    match measurement {
        SerializableStoredGnssMeasurement::Loose(measurement) => {
            bytes.push(2);
            write_loose_measurement(bytes, measurement)
        }
        SerializableStoredGnssMeasurement::Tight(epoch) => {
            bytes.push(1);
            write_tight_epoch(bytes, epoch)
        }
    }
}

fn read_stored_measurement(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableStoredGnssMeasurement, FusionStateCodecError> {
    match read_u8(bytes, cursor, limit)? {
        0 => Ok(SerializableStoredGnssMeasurement::Loose(
            read_loose_measurement_with_default_status(bytes, cursor, limit)?,
        )),
        1 => Ok(SerializableStoredGnssMeasurement::Tight(read_tight_epoch(
            bytes, cursor, limit,
        )?)),
        2 => Ok(SerializableStoredGnssMeasurement::Loose(
            read_loose_measurement(bytes, cursor, limit)?,
        )),
        _ => Err(FusionStateCodecError::InvalidState {
            reason: "invalid GNSS measurement tag".to_string(),
        }),
    }
}

fn write_loose_measurement(
    bytes: &mut Vec<u8>,
    measurement: &SerializableLooseMeasurement,
) -> Result<(), FusionStateCodecError> {
    write_f64(bytes, measurement.t_j2000_s);
    write_f64_array(bytes, &measurement.position_ecef_m);
    match measurement.velocity_ecef_mps {
        Some(velocity) => {
            write_bool(bytes, true);
            write_f64_array(bytes, &velocity);
        }
        None => write_bool(bytes, false),
    }
    write_f64_matrix(bytes, &measurement.covariance)?;
    write_u32_checked(bytes, measurement.satellites_used as usize)?;
    write_bool(bytes, measurement.solution_valid);
    write_fix_status(bytes, measurement.fix_status);
    Ok(())
}

fn read_loose_measurement(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableLooseMeasurement, FusionStateCodecError> {
    read_loose_measurement_core(bytes, cursor, limit, true)
}

fn read_loose_measurement_with_default_status(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableLooseMeasurement, FusionStateCodecError> {
    read_loose_measurement_core(bytes, cursor, limit, false)
}

fn read_loose_measurement_core(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
    read_status: bool,
) -> Result<SerializableLooseMeasurement, FusionStateCodecError> {
    let t_j2000_s = read_f64(bytes, cursor, limit)?;
    let position_ecef_m = read_f64_array(bytes, cursor, limit)?;
    let velocity_ecef_mps = if read_bool(bytes, cursor, limit)? {
        Some(read_f64_array(bytes, cursor, limit)?)
    } else {
        None
    };
    let covariance = read_f64_matrix(bytes, cursor, limit)?;
    let satellites_used = read_u32(bytes, cursor, limit)?;
    let solution_valid = read_bool(bytes, cursor, limit)?;
    let fix_status = if read_status {
        read_fix_status(bytes, cursor, limit)?
    } else {
        SerializableGnssFixStatus::Single
    };
    Ok(SerializableLooseMeasurement {
        t_j2000_s,
        position_ecef_m,
        velocity_ecef_mps,
        covariance,
        satellites_used,
        solution_valid,
        fix_status,
    })
}

fn write_fix_status(bytes: &mut Vec<u8>, status: SerializableGnssFixStatus) {
    bytes.push(match status {
        SerializableGnssFixStatus::Single => 0,
        SerializableGnssFixStatus::Float => 1,
        SerializableGnssFixStatus::Fixed => 2,
    });
}

fn read_fix_status(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableGnssFixStatus, FusionStateCodecError> {
    match read_u8(bytes, cursor, limit)? {
        0 => Ok(SerializableGnssFixStatus::Single),
        1 => Ok(SerializableGnssFixStatus::Float),
        2 => Ok(SerializableGnssFixStatus::Fixed),
        _ => Err(FusionStateCodecError::InvalidState {
            reason: "invalid loose fix status".to_string(),
        }),
    }
}

fn write_tight_epoch(
    bytes: &mut Vec<u8>,
    epoch: &SerializableTightGnssEpoch,
) -> Result<(), FusionStateCodecError> {
    write_f64(bytes, epoch.t_j2000_s);
    write_u32_checked(bytes, epoch.observations.len())?;
    for observation in &epoch.observations {
        write_tight_observation(bytes, observation);
    }
    Ok(())
}

fn read_tight_epoch(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableTightGnssEpoch, FusionStateCodecError> {
    let t_j2000_s = read_f64(bytes, cursor, limit)?;
    let len = read_len(bytes, cursor, limit, 1, "tight_observations")?;
    let mut observations = Vec::with_capacity(len);
    for _ in 0..len {
        observations.push(read_tight_observation(bytes, cursor, limit)?);
    }
    Ok(SerializableTightGnssEpoch {
        t_j2000_s,
        observations,
    })
}

fn write_tight_observation(bytes: &mut Vec<u8>, observation: &SerializableTightGnssObservation) {
    write_satellite_id(bytes, observation.satellite_id);
    write_f64(bytes, observation.pseudorange_m);
    write_f64(bytes, observation.pseudorange_sigma_m);
    match observation.range_rate {
        Some(range_rate) => {
            write_bool(bytes, true);
            write_range_rate_observation(bytes, range_rate);
        }
        None => write_bool(bytes, false),
    }
    match observation.carrier_phase {
        Some(carrier_phase) => {
            write_bool(bytes, true);
            write_carrier_phase_observation(bytes, carrier_phase);
        }
        None => write_bool(bytes, false),
    }
    write_f64(bytes, observation.ionosphere_delay_m);
    write_f64(bytes, observation.troposphere_delay_m);
}

fn read_tight_observation(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableTightGnssObservation, FusionStateCodecError> {
    let satellite_id = read_satellite_id(bytes, cursor, limit)?;
    let pseudorange_m = read_f64(bytes, cursor, limit)?;
    let pseudorange_sigma_m = read_f64(bytes, cursor, limit)?;
    let range_rate = if read_bool(bytes, cursor, limit)? {
        Some(read_range_rate_observation(bytes, cursor, limit)?)
    } else {
        None
    };
    let carrier_phase = if read_bool(bytes, cursor, limit)? {
        Some(read_carrier_phase_observation(bytes, cursor, limit)?)
    } else {
        None
    };
    Ok(SerializableTightGnssObservation {
        satellite_id,
        pseudorange_m,
        pseudorange_sigma_m,
        range_rate,
        carrier_phase,
        ionosphere_delay_m: read_f64(bytes, cursor, limit)?,
        troposphere_delay_m: read_f64(bytes, cursor, limit)?,
    })
}

fn write_range_rate_observation(
    bytes: &mut Vec<u8>,
    observation: SerializableTightRangeRateObservation,
) {
    write_f64(bytes, observation.measured_range_rate_m_s);
    write_f64(bytes, observation.sigma_m_s);
    write_f64(bytes, observation.satellite_clock_drift_m_s);
}

fn read_range_rate_observation(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableTightRangeRateObservation, FusionStateCodecError> {
    Ok(SerializableTightRangeRateObservation {
        measured_range_rate_m_s: read_f64(bytes, cursor, limit)?,
        sigma_m_s: read_f64(bytes, cursor, limit)?,
        satellite_clock_drift_m_s: read_f64(bytes, cursor, limit)?,
    })
}

fn write_carrier_phase_observation(
    bytes: &mut Vec<u8>,
    observation: SerializableTightCarrierPhaseObservation,
) {
    write_f64(bytes, observation.phase_range_m);
    write_f64(bytes, observation.sigma_m);
    write_f64(bytes, observation.float_ambiguity_m);
}

fn read_carrier_phase_observation(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableTightCarrierPhaseObservation, FusionStateCodecError> {
    Ok(SerializableTightCarrierPhaseObservation {
        phase_range_m: read_f64(bytes, cursor, limit)?,
        sigma_m: read_f64(bytes, cursor, limit)?,
        float_ambiguity_m: read_f64(bytes, cursor, limit)?,
    })
}

fn write_satellite_id(bytes: &mut Vec<u8>, id: SerializableSatelliteId) {
    bytes.push(match id.system {
        GnssSystem::Gps => 0,
        GnssSystem::Glonass => 1,
        GnssSystem::Galileo => 2,
        GnssSystem::BeiDou => 3,
        GnssSystem::Qzss => 4,
        GnssSystem::Navic => 5,
        GnssSystem::Sbas => 6,
    });
    bytes.push(id.prn);
}

fn read_satellite_id(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<SerializableSatelliteId, FusionStateCodecError> {
    let system = match read_u8(bytes, cursor, limit)? {
        0 => GnssSystem::Gps,
        1 => GnssSystem::Glonass,
        2 => GnssSystem::Galileo,
        3 => GnssSystem::BeiDou,
        4 => GnssSystem::Qzss,
        5 => GnssSystem::Navic,
        6 => GnssSystem::Sbas,
        _ => {
            return Err(FusionStateCodecError::InvalidState {
                reason: "invalid GNSS system tag".to_string(),
            });
        }
    };
    Ok(SerializableSatelliteId {
        system,
        prn: read_u8(bytes, cursor, limit)?,
    })
}

fn write_f64_matrix(
    bytes: &mut Vec<u8>,
    matrix: &[Vec<F64Bits>],
) -> Result<(), FusionStateCodecError> {
    write_u32_checked(bytes, matrix.len())?;
    let cols = matrix.first().map_or(0, Vec::len);
    write_u32_checked(bytes, cols)?;
    for row in matrix {
        if row.len() != cols {
            return Err(FusionStateCodecError::InvalidState {
                reason: "ragged matrix cannot be encoded".to_string(),
            });
        }
        write_f64_vec_body(bytes, row);
    }
    Ok(())
}

fn read_f64_matrix(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<Vec<Vec<F64Bits>>, FusionStateCodecError> {
    let rows = read_u32(bytes, cursor, limit)? as usize;
    let cols = read_u32(bytes, cursor, limit)? as usize;
    if rows == 0 || cols == 0 {
        return Err(FusionStateCodecError::InvalidState {
            reason: "matrix dimensions must be positive".to_string(),
        });
    }
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| FusionStateCodecError::InvalidState {
            reason: "matrix dimensions overflow usize".to_string(),
        })?;
    let needed = count
        .checked_mul(8)
        .ok_or_else(|| FusionStateCodecError::InvalidState {
            reason: "matrix byte length overflows usize".to_string(),
        })?;
    ensure_available(*cursor, needed, limit)?;
    let mut matrix = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            row.push(read_f64(bytes, cursor, limit)?);
        }
        matrix.push(row);
    }
    Ok(matrix)
}

fn write_f64_vec(bytes: &mut Vec<u8>, values: &[F64Bits]) -> Result<(), FusionStateCodecError> {
    write_u32_checked(bytes, values.len())?;
    write_f64_vec_body(bytes, values);
    Ok(())
}

fn write_f64_vec_body(bytes: &mut Vec<u8>, values: &[F64Bits]) {
    for value in values {
        write_f64(bytes, *value);
    }
}

fn read_f64_vec(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<Vec<F64Bits>, FusionStateCodecError> {
    let len = read_u32(bytes, cursor, limit)? as usize;
    let needed = len
        .checked_mul(8)
        .ok_or_else(|| FusionStateCodecError::InvalidState {
            reason: "vector byte length overflows usize".to_string(),
        })?;
    ensure_available(*cursor, needed, limit)?;
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(read_f64(bytes, cursor, limit)?);
    }
    Ok(values)
}

fn write_f64_array<const N: usize>(bytes: &mut Vec<u8>, values: &[F64Bits; N]) {
    for value in values {
        write_f64(bytes, *value);
    }
}

fn read_f64_array<const N: usize>(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<[F64Bits; N], FusionStateCodecError> {
    let mut out = [F64Bits { bits: 0 }; N];
    for value in &mut out {
        *value = read_f64(bytes, cursor, limit)?;
    }
    Ok(out)
}

fn write_f64(bytes: &mut Vec<u8>, value: F64Bits) {
    write_u64(bytes, value.bits);
}

fn read_f64(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<F64Bits, FusionStateCodecError> {
    Ok(F64Bits {
        bits: read_u64(bytes, cursor, limit)?,
    })
}

fn write_u16(bytes: &mut Vec<u8>, value: u16) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_u32_checked(bytes: &mut Vec<u8>, value: usize) -> Result<(), FusionStateCodecError> {
    let value = u32::try_from(value).map_err(|_| FusionStateCodecError::InvalidState {
        reason: "length exceeds u32".to_string(),
    })?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn write_u64(bytes: &mut Vec<u8>, value: u64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn write_bool(bytes: &mut Vec<u8>, value: bool) {
    bytes.push(u8::from(value));
}

fn read_bool(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
) -> Result<bool, FusionStateCodecError> {
    match read_u8(bytes, cursor, limit)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(FusionStateCodecError::InvalidState {
            reason: "invalid boolean tag".to_string(),
        }),
    }
}

fn read_u8(bytes: &[u8], cursor: &mut usize, limit: usize) -> Result<u8, FusionStateCodecError> {
    ensure_available(*cursor, 1, limit)?;
    let value = bytes[*cursor];
    *cursor += 1;
    Ok(value)
}

fn read_u16(bytes: &[u8], cursor: &mut usize, limit: usize) -> Result<u16, FusionStateCodecError> {
    let data = read_array::<2>(bytes, *cursor, limit)?;
    *cursor += 2;
    Ok(u16::from_le_bytes(data))
}

fn read_u32(bytes: &[u8], cursor: &mut usize, limit: usize) -> Result<u32, FusionStateCodecError> {
    let data = read_array::<4>(bytes, *cursor, limit)?;
    *cursor += 4;
    Ok(u32::from_le_bytes(data))
}

fn read_u64(bytes: &[u8], cursor: &mut usize, limit: usize) -> Result<u64, FusionStateCodecError> {
    let data = read_array::<8>(bytes, *cursor, limit)?;
    *cursor += 8;
    Ok(u64::from_le_bytes(data))
}

fn read_u64_at(bytes: &[u8], offset: usize) -> Result<u64, FusionStateCodecError> {
    Ok(u64::from_le_bytes(read_array::<8>(
        bytes,
        offset,
        bytes.len(),
    )?))
}

fn read_len(
    bytes: &[u8],
    cursor: &mut usize,
    limit: usize,
    min_element_bytes: usize,
    field: &'static str,
) -> Result<usize, FusionStateCodecError> {
    let len = read_u32(bytes, cursor, limit)? as usize;
    let needed =
        len.checked_mul(min_element_bytes)
            .ok_or_else(|| FusionStateCodecError::InvalidState {
                reason: format!("{field} byte length overflows usize"),
            })?;
    ensure_available(*cursor, needed, limit)?;
    Ok(len)
}

fn read_array<const N: usize>(
    bytes: &[u8],
    offset: usize,
    limit: usize,
) -> Result<[u8; N], FusionStateCodecError> {
    ensure_available(offset, N, limit)?;
    let end = offset + N;
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes[offset..end]);
    Ok(out)
}

fn ensure_available(
    offset: usize,
    needed: usize,
    limit: usize,
) -> Result<(), FusionStateCodecError> {
    let end = offset
        .checked_add(needed)
        .ok_or(FusionStateCodecError::Truncated {
            offset,
            needed,
            actual: limit.saturating_sub(offset),
        })?;
    if end <= limit {
        Ok(())
    } else {
        Err(FusionStateCodecError::Truncated {
            offset,
            needed,
            actual: limit.saturating_sub(offset),
        })
    }
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

#[cfg(test)]
mod tests {
    //! Provenance: serial codec tests assert the documented ABI contract for
    //! versioned little-endian fields and FNV-1a corruption detection. The
    //! numeric oracle is exact IEEE-754 bit identity before and after binary and
    //! serde JSON round trips.

    use super::*;
    use crate::astro::constants::earth::WGS84_A_M;
    use crate::fusion::state::{ErrorStateLayout, ERROR_STATE_DIMENSION_15};
    use crate::fusion::TimeSyncHistoryConfig;
    use crate::inertial::config::RANDOM_WALK_BIAS_TAU_S;
    use crate::inertial::state::mat3_identity;
    use crate::inertial::{ImuSample, ImuSpec, NavState};

    fn test_filter() -> InertialFilter {
        let nominal = NavState::new(
            12.5,
            [WGS84_A_M, -0.0, 3.25],
            [0.5, -0.25, 0.125],
            mat3_identity(),
        )
        .expect("nominal")
        .with_biases([0.01, -0.02, 0.03], [-0.001, 0.002, -0.003])
        .expect("biases");
        let mut diagonal = vec![1.0; ERROR_STATE_DIMENSION_15];
        diagonal[0] = 4.0;
        diagonal[1] = 9.0;
        let state = InsFilterState::from_diagonal(nominal, ErrorStateLayout::Fifteen, &diagonal)
            .expect("state");
        let spec = ImuSpec::datasheet(
            0.0,
            0.0,
            0.0,
            0.0,
            RANDOM_WALK_BIAS_TAU_S,
            RANDOM_WALK_BIAS_TAU_S,
            None,
            None,
        );
        InertialFilter::new(state, spec).expect("filter")
    }

    fn increment(t_j2000_s: f64, dt_s: f64) -> ImuSample {
        ImuSample::increment(
            t_j2000_s,
            [0.015625 * dt_s, -0.0078125 * dt_s, 0.00390625 * dt_s],
            [
                0.0009765625 * dt_s,
                -0.00048828125 * dt_s,
                0.000244140625 * dt_s,
            ],
            dt_s,
        )
    }

    fn measurement_at(t_j2000_s: f64, position_ecef_m: [f64; 3]) -> GnssFixMeasurement {
        GnssFixMeasurement::position(
            t_j2000_s,
            position_ecef_m,
            [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
            8,
        )
        .expect("measurement")
    }

    #[test]
    fn binary_stored_loose_measurement_preserves_fix_status() {
        let measurement = SerializableLooseMeasurement::from_native(
            &measurement_at(13.0, [WGS84_A_M + 0.25, -0.125, 3.5])
                .with_fix_status(GnssFixStatus::Float),
        )
        .expect("serial measurement");
        let stored = SerializableStoredGnssMeasurement::Loose(measurement.clone());
        let mut encoded = Vec::new();
        write_stored_measurement(&mut encoded, &stored).expect("write measurement");

        let mut cursor = 0usize;
        let decoded = read_stored_measurement(&encoded, &mut cursor, encoded.len()).expect("read");

        assert_eq!(decoded, stored);
        assert_eq!(cursor, encoded.len());
    }

    #[test]
    fn binary_stored_loose_measurement_old_tag_defaults_fix_status() {
        let measurement = SerializableLooseMeasurement::from_native(
            &measurement_at(13.0, [WGS84_A_M + 0.25, -0.125, 3.5])
                .with_fix_status(GnssFixStatus::Fixed),
        )
        .expect("serial measurement");
        let mut encoded = Vec::new();
        encoded.push(0);
        write_f64(&mut encoded, measurement.t_j2000_s);
        write_f64_array(&mut encoded, &measurement.position_ecef_m);
        if let Some(velocity) = measurement.velocity_ecef_mps {
            write_bool(&mut encoded, true);
            write_f64_array(&mut encoded, &velocity);
        } else {
            write_bool(&mut encoded, false);
        }
        write_f64_matrix(&mut encoded, &measurement.covariance).expect("covariance");
        write_u32_checked(&mut encoded, measurement.satellites_used as usize).expect("satellites");
        write_bool(&mut encoded, measurement.solution_valid);

        let mut cursor = 0usize;
        let decoded = read_stored_measurement(&encoded, &mut cursor, encoded.len()).expect("read");

        let SerializableStoredGnssMeasurement::Loose(decoded) = decoded else {
            panic!("expected loose measurement");
        };
        assert_eq!(decoded.fix_status, SerializableGnssFixStatus::Single);
        assert_eq!(decoded.t_j2000_s, measurement.t_j2000_s);
        assert_eq!(decoded.position_ecef_m, measurement.position_ecef_m);
        assert_eq!(cursor, encoded.len());
    }

    #[test]
    fn binary_and_json_round_trip_preserve_bits() {
        let filter = test_filter();
        let serial = filter.serializable_state().expect("serial state");
        let encoded = serial.encode_versioned().expect("encode");
        let decoded = SerializableFusionState::decode_versioned(&encoded).expect("decode");
        assert_eq!(decoded, serial);
        assert_snapshot_bits(
            &decoded.to_snapshot().expect("snapshot"),
            &filter.snapshot(),
        );

        let json = serial.to_json_string().expect("json");
        let decoded_json = SerializableFusionState::from_json_str(&json).expect("json decode");
        assert_eq!(decoded_json, serial);
        assert_snapshot_bits(
            &decoded_json.to_snapshot().expect("json snapshot"),
            &filter.snapshot(),
        );
    }

    #[test]
    fn binary_decoder_accepts_legacy_v2_without_stationarity_window() {
        let filter = test_filter();
        let serial = filter.serializable_state().expect("serial state");
        let encoded = encode_legacy_without_stationarity(&serial, 2).expect("legacy encode");

        let decoded = SerializableFusionState::decode_versioned(&encoded).expect("decode v2");
        assert_eq!(decoded.version, 2);
        let mut expected = filter.snapshot();
        expected.stationarity_window.clear();
        expected.last_stationary_update_t_j2000_s = None;
        expected.last_non_holonomic_update_t_j2000_s = None;
        assert_snapshot_bits(&decoded.to_snapshot().expect("snapshot"), &expected);
    }

    #[test]
    fn truncated_and_corrupted_payloads_are_typed_errors() {
        let serial = test_filter().serializable_state().expect("serial state");
        let encoded = serial.encode_versioned().expect("encode");
        let truncated = &encoded[..encoded.len() - 3];
        assert!(matches!(
            SerializableFusionState::decode_versioned(truncated),
            Err(FusionStateCodecError::Checksum { .. })
        ));

        let mut corrupted = encoded;
        let idx = corrupted.len() / 2;
        corrupted[idx] ^= 0x55;
        assert!(matches!(
            SerializableFusionState::decode_versioned(&corrupted),
            Err(FusionStateCodecError::Checksum { .. })
        ));

        let too_short = [0u8; 5];
        assert!(matches!(
            SerializableFusionState::decode_versioned(&too_short),
            Err(FusionStateCodecError::Truncated { .. })
        ));
    }

    #[test]
    fn malformed_matrix_dimensions_are_typed_errors() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        let mut cursor = 0usize;
        assert!(matches!(
            read_f64_matrix(&bytes, &mut cursor, bytes.len()),
            Err(FusionStateCodecError::InvalidState { .. })
        ));
    }

    #[test]
    fn restored_encoded_state_retains_time_sync_history_for_late_replay_bits() {
        let first = measurement_at(13.0, [WGS84_A_M + 0.25, -0.125, 3.5]);
        let late = measurement_at(13.25, [WGS84_A_M - 0.0625, 0.1875, 3.0]);
        let final_fix = measurement_at(13.5, [WGS84_A_M + 0.03125, -0.25, 3.125]);

        let mut direct = test_filter();
        direct
            .configure_time_sync_history(TimeSyncHistoryConfig::new(8, 8))
            .expect("history");
        direct.propagate(increment(13.0, 0.5)).expect("imu");
        direct.update_loose(&first).expect("first");
        direct.propagate(increment(13.25, 0.25)).expect("imu");
        direct.update_loose(&late).expect("late in order");
        direct.propagate(increment(13.5, 0.25)).expect("imu");
        direct.update_loose(&final_fix).expect("final");
        direct.propagate(increment(14.0, 0.5)).expect("imu");

        let mut delayed = test_filter();
        delayed
            .configure_time_sync_history(TimeSyncHistoryConfig::new(8, 8))
            .expect("history");
        delayed.propagate(increment(13.0, 0.5)).expect("imu");
        delayed.update_loose(&first).expect("first");
        delayed.propagate(increment(13.5, 0.5)).expect("imu");
        delayed.update_loose(&final_fix).expect("final");
        delayed.propagate(increment(14.0, 0.5)).expect("imu");
        let encoded = delayed.encode_state().expect("encode");

        let mut restored = test_filter();
        restored.restore_encoded_state(&encoded).expect("restore");
        let update = restored
            .update_loose_time_sync(&late)
            .expect("late replay after restore");

        assert!(update.late_measurement);
        assert_snapshot_bits(&restored.snapshot(), &direct.snapshot());
    }

    fn assert_snapshot_bits(actual: &InertialFilterSnapshot, expected: &InertialFilterSnapshot) {
        assert_eq!(
            actual.state.nominal.t_j2000_s.to_bits(),
            expected.state.nominal.t_j2000_s.to_bits()
        );
        for axis in 0..3 {
            assert_eq!(
                actual.state.nominal.position_ecef_m[axis].to_bits(),
                expected.state.nominal.position_ecef_m[axis].to_bits()
            );
            assert_eq!(
                actual.last_body_rate_wrt_ecef_rps[axis].to_bits(),
                expected.last_body_rate_wrt_ecef_rps[axis].to_bits()
            );
        }
        assert_eq!(
            actual.stationarity_window.len(),
            expected.stationarity_window.len()
        );
        for (actual_sample, expected_sample) in actual
            .stationarity_window
            .iter()
            .zip(expected.stationarity_window.iter())
        {
            assert_eq!(
                actual_sample.specific_force_norm_error_mps2.to_bits(),
                expected_sample.specific_force_norm_error_mps2.to_bits()
            );
            assert_eq!(
                actual_sample.body_rate_wrt_ecef_norm_rps.to_bits(),
                expected_sample.body_rate_wrt_ecef_norm_rps.to_bits()
            );
        }
        assert_eq!(
            actual.last_stationary_update_t_j2000_s.map(f64::to_bits),
            expected.last_stationary_update_t_j2000_s.map(f64::to_bits)
        );
        assert_eq!(
            actual.last_non_holonomic_update_t_j2000_s.map(f64::to_bits),
            expected
                .last_non_holonomic_update_t_j2000_s
                .map(f64::to_bits)
        );
        for row in 0..actual.state.covariance.len() {
            for col in 0..actual.state.covariance[row].len() {
                assert_eq!(
                    actual.state.covariance[row][col].to_bits(),
                    expected.state.covariance[row][col].to_bits()
                );
            }
        }
        assert_eq!(
            actual.tight.clock_bias_m.to_bits(),
            expected.tight.clock_bias_m.to_bits()
        );
        assert_eq!(
            actual.tight.clock_drift_m_s.to_bits(),
            expected.tight.clock_drift_m_s.to_bits()
        );
        for row in 0..actual.tight.augmented_covariance.len() {
            for col in 0..actual.tight.augmented_covariance[row].len() {
                assert_eq!(
                    actual.tight.augmented_covariance[row][col].to_bits(),
                    expected.tight.augmented_covariance[row][col].to_bits()
                );
            }
        }
    }

    fn encode_legacy_without_stationarity(
        serial: &SerializableFusionState,
        version: u16,
    ) -> Result<Vec<u8>, FusionStateCodecError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&FUSION_STATE_MAGIC);
        write_u16(&mut bytes, version);
        write_layout(&mut bytes, serial.state.layout);
        write_nav(&mut bytes, &serial.state.nominal);
        write_f64_vec(&mut bytes, &serial.state.error_state)?;
        write_f64_matrix(&mut bytes, &serial.state.covariance)?;
        write_f64_array(&mut bytes, &serial.state.accel_scale_factor);
        write_f64_array(&mut bytes, &serial.state.gyro_scale_factor);
        write_f64_array(&mut bytes, &serial.last_body_rate_wrt_ecef_rps);
        write_f64(&mut bytes, serial.tight.clock_bias_m);
        write_f64(&mut bytes, serial.tight.clock_drift_m_s);
        write_f64_matrix(&mut bytes, &serial.tight.augmented_covariance)?;
        write_time_sync_history_legacy_without_stationarity(&mut bytes, &serial.time_sync)?;
        let checksum = fnv1a64(&bytes);
        write_u64(&mut bytes, checksum);
        Ok(bytes)
    }

    fn write_time_sync_history_legacy_without_stationarity(
        bytes: &mut Vec<u8>,
        history: &SerializableTimeSyncHistory,
    ) -> Result<(), FusionStateCodecError> {
        write_u32_checked(bytes, history.config.imu_capacity as usize)?;
        write_u32_checked(bytes, history.config.checkpoint_capacity as usize)?;
        write_u32_checked(bytes, history.imu_samples.len())?;
        for sample in &history.imu_samples {
            write_stored_imu_sample(bytes, sample);
        }
        write_u32_checked(bytes, history.checkpoints.len())?;
        for checkpoint in &history.checkpoints {
            write_f64(bytes, checkpoint.t_j2000_s);
            write_fusion_snapshot_legacy_without_stationarity(bytes, checkpoint.snapshot.as_ref())?;
        }
        write_u32_checked(bytes, history.measurements.len())?;
        for measurement in &history.measurements {
            write_stored_measurement(bytes, measurement)?;
        }
        Ok(())
    }

    fn write_fusion_snapshot_legacy_without_stationarity(
        bytes: &mut Vec<u8>,
        snapshot: &SerializableFusionSnapshot,
    ) -> Result<(), FusionStateCodecError> {
        write_layout(bytes, snapshot.state.layout);
        write_nav(bytes, &snapshot.state.nominal);
        write_f64_vec(bytes, &snapshot.state.error_state)?;
        write_f64_matrix(bytes, &snapshot.state.covariance)?;
        write_f64_array(bytes, &snapshot.state.accel_scale_factor);
        write_f64_array(bytes, &snapshot.state.gyro_scale_factor);
        write_f64_array(bytes, &snapshot.last_body_rate_wrt_ecef_rps);
        write_f64(bytes, snapshot.tight.clock_bias_m);
        write_f64(bytes, snapshot.tight.clock_drift_m_s);
        write_f64_matrix(bytes, &snapshot.tight.augmented_covariance)
    }
}
