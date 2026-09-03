//! Loosely coupled GNSS PVT updates for the INS error-state filter.
//!
//! Measurement epochs are required to match the propagated INS epoch exactly.
//! A later time-synchronization layer can lift that requirement without
//! changing the measurement model here.

use crate::astro::constants::earth::OMEGA_E_DOT_RAD_S;
use crate::astro::math::mat3::{inline_rxr, inline_tr, mul_vec3, Mat3};
use crate::astro::math::vec3::{add3, cross3, norm3, scale3, sub3};
use crate::inertial::frames::gravity_ecef_mps2;
use crate::inertial::state::{mat3_identity, skew, validate_dcm_orthonormal};
use crate::inertial::{
    mechanize_ecef, validate_finite, validate_vec3, ImuCalibration, ImuErrorModel, ImuSample,
    ImuSpec, MechanizationConfig,
};
use std::collections::VecDeque;

use super::ekf::{
    ekf_correct_closed_loop, ekf_correct_closed_loop_with_predicted_covariance_scale,
    innovation_covariance, normalized_innovation_squared, EkfCorrection, EkfCorrectionReport,
    EkfUpdateOptions,
};
use super::error_state::{
    linearize_error_state_ecef_with_imu_to_body, predict_error_state_covariance,
    ErrorStateImuKinematics,
};
use super::smoother::FusionPredictionStep;
use super::state::FusionFilterKind;
use super::state::{
    invalid_input, validate_covariance_matrix, validate_nonnegative, validate_positive,
    FusionError, InsFilterState, ERROR_ATTITUDE_INDEX, ERROR_GYRO_BIAS_INDEX, ERROR_POSITION_INDEX,
    ERROR_STATE_DIMENSION_15, ERROR_STATE_DIMENSION_21, ERROR_VELOCITY_INDEX,
};
use super::tight::{TightCouplingConfig, TightFusionState};
use super::timesync::{StationarityDetectorSnapshotSample, TimeSyncHistory};
use super::ukf::{ukf_correct_closed_loop, UkfUpdateOptions};

const LOOSE_MIN_SATELLITES: usize = 4;
const POSITION_ROWS: usize = 3;
const POSITION_VELOCITY_ROWS: usize = 6;
const ZUPT_ZARU_ROWS: usize = 6;
const NHC_ROWS: usize = 2;
const IGG_III_REJECTION_VARIANCE_SCALE: f64 = 1.0e4;
const DEFAULT_PREDICTION_ADAPTATION_GATE_PROBABILITY: f64 = 0.99;

/// GNSS PVT measurement used by the loose-coupled INS update.
///
/// The covariance matrix is ordered as `[position_x, position_y, position_z]`
/// for a position-only fix and as `[position_x, position_y, position_z,
/// velocity_x, velocity_y, velocity_z]` when velocity is present.
#[derive(Debug, Clone, PartialEq)]
pub struct GnssFixMeasurement {
    /// Measurement epoch in seconds since J2000 on the caller's GNSS time scale.
    pub t_j2000_s: f64,
    /// GNSS antenna position in ECEF meters.
    pub position_ecef_m: [f64; 3],
    /// Optional GNSS antenna velocity in ECEF meters per second.
    pub velocity_ecef_mps: Option<[f64; 3]>,
    /// Measurement covariance in the order documented on this type.
    pub covariance: Vec<Vec<f64>>,
    /// Number of satellites used by the upstream GNSS fix.
    pub satellites_used: usize,
    /// Whether the upstream GNSS solver reported a successful fix.
    pub solution_valid: bool,
    /// Upstream ambiguity or code-only fix class for covariance scaling.
    pub fix_status: GnssFixStatus,
}

impl GnssFixMeasurement {
    /// Build a position-only GNSS fix measurement.
    pub fn position(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        position_covariance_m2: [[f64; 3]; 3],
        satellites_used: usize,
    ) -> Result<Self, FusionError> {
        let measurement = Self {
            t_j2000_s,
            position_ecef_m,
            velocity_ecef_mps: None,
            covariance: mat3_to_rows(position_covariance_m2),
            satellites_used,
            solution_valid: true,
            fix_status: GnssFixStatus::Single,
        };
        measurement.validate()?;
        Ok(measurement)
    }

    /// Build a position and velocity GNSS fix measurement.
    pub fn position_velocity(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        velocity_ecef_mps: [f64; 3],
        covariance: Vec<Vec<f64>>,
        satellites_used: usize,
    ) -> Result<Self, FusionError> {
        let measurement = Self {
            t_j2000_s,
            position_ecef_m,
            velocity_ecef_mps: Some(velocity_ecef_mps),
            covariance,
            satellites_used,
            solution_valid: true,
            fix_status: GnssFixStatus::Single,
        };
        measurement.validate()?;
        Ok(measurement)
    }

    /// Return this measurement tagged with an upstream fix status.
    pub fn with_fix_status(mut self, fix_status: GnssFixStatus) -> Self {
        self.fix_status = fix_status;
        self
    }

    /// Validate finite values, solver status, satellite count, and covariance.
    pub fn validate(&self) -> Result<(), FusionError> {
        validate_finite(self.t_j2000_s, "t_j2000_s").map_err(FusionError::from)?;
        validate_vec3(self.position_ecef_m, "position_ecef_m").map_err(FusionError::from)?;
        if let Some(velocity) = self.velocity_ecef_mps {
            validate_vec3(velocity, "velocity_ecef_mps").map_err(FusionError::from)?;
        }
        if !self.solution_valid {
            return Err(invalid_input(
                "solution_valid",
                "GNSS fix must be successful",
            ));
        }
        if self.satellites_used < LOOSE_MIN_SATELLITES {
            return Err(invalid_input(
                "satellites_used",
                "at least 4 satellites required",
            ));
        }
        validate_covariance_matrix(&self.covariance, self.row_count(), "gnss_covariance")
    }

    /// Return the number of measurement rows implied by this fix.
    pub fn row_count(&self) -> usize {
        if self.velocity_ecef_mps.is_some() {
            POSITION_VELOCITY_ROWS
        } else {
            POSITION_ROWS
        }
    }
}

/// Upstream GNSS solution class used by loose measurement weighting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GnssFixStatus {
    /// Code-only or standalone GNSS fix.
    Single,
    /// Float carrier-phase ambiguity solution.
    Float,
    /// Fixed carrier-phase ambiguity solution.
    Fixed,
}

/// Configuration for loose-coupled GNSS updates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct LooseCouplingConfig {
    /// Body-frame vector from IMU origin to GNSS antenna phase center, in meters.
    pub lever_arm_body_m: [f64; 3],
    /// Generic EKF correction options applied to each loose update.
    pub update_options: EkfUpdateOptions,
    /// Per-fix-status sigma multipliers applied to GNSS covariance.
    pub fix_status_weighting: GnssFixStatusWeighting,
    /// Optional IGG-III variance inflation on standardized innovation rows.
    pub measurement_reweighting: Option<IggIiiMeasurementReweighting>,
    /// Optional Yang two-segment predicted-covariance inflation.
    pub prediction_adaptation: Option<YangPredictionAdaptiveFactor>,
    /// Optional stationary zero-velocity and zero-angular-rate updates.
    pub stationary_updates: Option<StationaryUpdateConfig>,
    /// Optional wheeled-vehicle lateral and vertical velocity constraints.
    pub non_holonomic: Option<NonHolonomicConstraintConfig>,
}

impl Default for LooseCouplingConfig {
    fn default() -> Self {
        Self {
            lever_arm_body_m: [0.0; 3],
            update_options: EkfUpdateOptions::default(),
            fix_status_weighting: GnssFixStatusWeighting::default(),
            measurement_reweighting: None,
            prediction_adaptation: None,
            stationary_updates: None,
            non_holonomic: None,
        }
    }
}

impl LooseCouplingConfig {
    /// Validate finite lever-arm entries and nested update options.
    pub fn validate(&self) -> Result<(), FusionError> {
        validate_vec3(self.lever_arm_body_m, "lever_arm_body_m").map_err(FusionError::from)?;
        if let Some(gate) = self.update_options.innovation_gate {
            gate.validate()?;
        }
        self.fix_status_weighting.validate()?;
        if let Some(reweighting) = self.measurement_reweighting {
            reweighting.validate()?;
        }
        if let Some(adaptation) = self.prediction_adaptation {
            adaptation.validate()?;
        }
        if let Some(stationary) = self.stationary_updates {
            stationary.validate()?;
        }
        if let Some(non_holonomic) = self.non_holonomic {
            non_holonomic.validate()?;
        }
        Ok(())
    }
}

/// Sigma multipliers selected by [`GnssFixStatus`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnssFixStatusWeighting {
    /// Sigma multiplier for standalone GNSS fixes.
    pub single_sigma_multiplier: f64,
    /// Sigma multiplier for float ambiguity fixes.
    pub float_sigma_multiplier: f64,
    /// Sigma multiplier for fixed ambiguity fixes.
    pub fixed_sigma_multiplier: f64,
}

impl Default for GnssFixStatusWeighting {
    fn default() -> Self {
        Self {
            single_sigma_multiplier: 1.0,
            float_sigma_multiplier: 1.0,
            fixed_sigma_multiplier: 1.0,
        }
    }
}

impl GnssFixStatusWeighting {
    /// Return the sigma multiplier for a GNSS fix status.
    pub fn multiplier(self, status: GnssFixStatus) -> f64 {
        match status {
            GnssFixStatus::Single => self.single_sigma_multiplier,
            GnssFixStatus::Float => self.float_sigma_multiplier,
            GnssFixStatus::Fixed => self.fixed_sigma_multiplier,
        }
    }

    /// Validate that all multipliers are finite and positive.
    pub fn validate(&self) -> Result<(), FusionError> {
        validate_positive(self.single_sigma_multiplier, "single_sigma_multiplier")?;
        validate_positive(self.float_sigma_multiplier, "float_sigma_multiplier")?;
        validate_positive(self.fixed_sigma_multiplier, "fixed_sigma_multiplier")
    }
}

/// Stationarity detector and pseudo-measurement sigmas for ZUPT/ZARU.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StationaryUpdateConfig {
    /// Detector thresholds over a trailing IMU epoch window.
    pub detector: StationaryDetectorConfig,
    /// One-sigma zero-velocity pseudo-measurement noise in m/s.
    pub zero_velocity_sigma_mps: f64,
    /// One-sigma zero-angular-rate pseudo-measurement noise in rad/s.
    pub zero_angular_rate_sigma_rps: f64,
}

impl StationaryUpdateConfig {
    /// Build stationary-update settings from the detector and both required
    /// pseudo-measurement sigmas.
    #[must_use]
    pub const fn new(
        detector: StationaryDetectorConfig,
        zero_velocity_sigma_mps: f64,
        zero_angular_rate_sigma_rps: f64,
    ) -> Self {
        Self {
            detector,
            zero_velocity_sigma_mps,
            zero_angular_rate_sigma_rps,
        }
    }

    /// Validate detector settings and pseudo-measurement sigmas.
    pub fn validate(&self) -> Result<(), FusionError> {
        self.detector.validate()?;
        validate_positive(self.zero_velocity_sigma_mps, "zero_velocity_sigma_mps")?;
        validate_positive(
            self.zero_angular_rate_sigma_rps,
            "zero_angular_rate_sigma_rps",
        )
    }
}

/// Windowed accel and gyro magnitude detector for stationary updates.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct StationaryDetectorConfig {
    /// Number of propagated IMU epochs required before the detector can fire.
    pub window_len: usize,
    /// Maximum allowed specific-force norm error from local gravity.
    pub max_specific_force_norm_error_mps2: f64,
    /// Maximum body angular-rate norm relative to ECEF.
    pub max_body_rate_wrt_ecef_norm_rps: f64,
}

impl StationaryDetectorConfig {
    /// Build detector settings from all required window and threshold values.
    #[must_use]
    pub const fn new(
        window_len: usize,
        max_specific_force_norm_error_mps2: f64,
        max_body_rate_wrt_ecef_norm_rps: f64,
    ) -> Self {
        Self {
            window_len,
            max_specific_force_norm_error_mps2,
            max_body_rate_wrt_ecef_norm_rps,
        }
    }

    /// Validate finite, non-negative thresholds and non-empty window length.
    pub fn validate(&self) -> Result<(), FusionError> {
        if self.window_len == 0 {
            return Err(invalid_input("stationary_window_len", "must be nonzero"));
        }
        validate_nonnegative(
            self.max_specific_force_norm_error_mps2,
            "max_specific_force_norm_error_mps2",
        )?;
        validate_nonnegative(
            self.max_body_rate_wrt_ecef_norm_rps,
            "max_body_rate_wrt_ecef_norm_rps",
        )
    }
}

/// Non-holonomic wheeled-vehicle velocity constraint settings.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct NonHolonomicConstraintConfig {
    /// One-sigma lateral body velocity pseudo-measurement noise in m/s.
    pub lateral_velocity_sigma_mps: f64,
    /// One-sigma vertical body velocity pseudo-measurement noise in m/s.
    pub vertical_velocity_sigma_mps: f64,
    /// Minimum ECEF speed required before applying NHC.
    pub min_speed_mps: f64,
    /// Maximum body angular-rate norm relative to ECEF.
    pub max_body_rate_wrt_ecef_norm_rps: f64,
}

impl NonHolonomicConstraintConfig {
    /// Build non-holonomic settings from all required sigmas and motion gates.
    #[must_use]
    pub const fn new(
        lateral_velocity_sigma_mps: f64,
        vertical_velocity_sigma_mps: f64,
        min_speed_mps: f64,
        max_body_rate_wrt_ecef_norm_rps: f64,
    ) -> Self {
        Self {
            lateral_velocity_sigma_mps,
            vertical_velocity_sigma_mps,
            min_speed_mps,
            max_body_rate_wrt_ecef_norm_rps,
        }
    }

    /// Validate sigmas and motion gates.
    pub fn validate(&self) -> Result<(), FusionError> {
        validate_positive(
            self.lateral_velocity_sigma_mps,
            "lateral_velocity_sigma_mps",
        )?;
        validate_positive(
            self.vertical_velocity_sigma_mps,
            "vertical_velocity_sigma_mps",
        )?;
        validate_nonnegative(self.min_speed_mps, "nhc_min_speed_mps")?;
        validate_nonnegative(
            self.max_body_rate_wrt_ecef_norm_rps,
            "nhc_max_body_rate_wrt_ecef_norm_rps",
        )
    }
}

/// Endpoint matching settings for a GNSS outage span.
///
/// This is a near-real-time trajectory adjustment: it uses only the first good
/// position and velocity fix after an outage to blend a correction back to the
/// outage entry. It is not an RTS smoother because it does not recurse through
/// covariance, dynamics transitions, or later measurements.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct VelocityMatchingConfig {
    /// Maximum outage interval accepted by the matcher.
    pub max_outage_duration_s: f64,
}

impl VelocityMatchingConfig {
    /// Build endpoint-matching settings from the required outage limit.
    #[must_use]
    pub const fn new(max_outage_duration_s: f64) -> Self {
        Self {
            max_outage_duration_s,
        }
    }

    /// Validate finite, positive duration bound.
    pub fn validate(&self) -> Result<(), FusionError> {
        validate_positive(self.max_outage_duration_s, "max_outage_duration_s")
    }
}

/// IGG-III measurement variance inflation for loose GNSS updates.
///
/// The standardized innovation `v_i / sqrt(S_ii)` selects one of three
/// variance-domain segments from Remote Sensing 13(10):1943, Eq. 32: unchanged
/// below `k0`, IGG-III middle-segment inflation up to `k1`, and a fixed
/// `1e4` variance scale in the rejection segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IggIiiMeasurementReweighting {
    /// Lower standardized-innovation break point. Literature range: `[1.5, 3.0]`.
    pub k0_sigma: f64,
    /// Upper standardized-innovation break point. Literature range: `[3.0, 8.0]`.
    pub k1_sigma: f64,
}

impl IggIiiMeasurementReweighting {
    /// Common loose-GNSS setting inside the cited ranges.
    pub const fn standard() -> Self {
        Self {
            k0_sigma: 2.0,
            k1_sigma: 5.0,
        }
    }

    /// Validate IGG-III break points against the cited ranges.
    pub fn validate(&self) -> Result<(), FusionError> {
        validate_finite(self.k0_sigma, "igg_iii_k0_sigma").map_err(FusionError::from)?;
        validate_finite(self.k1_sigma, "igg_iii_k1_sigma").map_err(FusionError::from)?;
        if !(1.5..=3.0).contains(&self.k0_sigma) {
            return Err(invalid_input(
                "igg_iii_k0_sigma",
                "must be in the literature range [1.5, 3.0]",
            ));
        }
        if !(3.0..=8.0).contains(&self.k1_sigma) {
            return Err(invalid_input(
                "igg_iii_k1_sigma",
                "must be in the literature range [3.0, 8.0]",
            ));
        }
        if self.k0_sigma < self.k1_sigma {
            Ok(())
        } else {
            Err(invalid_input(
                "igg_iii_thresholds",
                "k0_sigma must be smaller than k1_sigma",
            ))
        }
    }
}

/// Yang two-segment predicted-residual adaptive factor for loose GNSS updates.
///
/// `threshold` is the `c` value used with the un-square-rooted statistic
/// `innovation^T innovation / tr(S)`. The Jiang-Zhang Sensors 2018 guard is
/// part of this option: if the raw innovation Mahalanobis distance exceeds
/// `chi2_inv(outlier_gate_probability, rows)`, prediction adaptation is
/// disabled for that epoch and measurement-side reweighting handles the fault.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct YangPredictionAdaptiveFactor {
    /// Two-segment threshold `c` for the predicted-residual statistic.
    pub threshold: f64,
    /// Probability used for the chi-square Mahalanobis measurement-outlier gate.
    pub outlier_gate_probability: f64,
}

impl YangPredictionAdaptiveFactor {
    /// Conservative default for the un-square-rooted statistic and 99% gate.
    pub const fn standard() -> Self {
        Self {
            threshold: 1.0,
            outlier_gate_probability: DEFAULT_PREDICTION_ADAPTATION_GATE_PROBABILITY,
        }
    }

    /// Validate threshold and chi-square probability.
    pub fn validate(&self) -> Result<(), FusionError> {
        validate_finite(self.threshold, "yang_prediction_threshold").map_err(FusionError::from)?;
        validate_finite(
            self.outlier_gate_probability,
            "yang_outlier_gate_probability",
        )
        .map_err(FusionError::from)?;
        if self.threshold <= 0.0 {
            return Err(invalid_input(
                "yang_prediction_threshold",
                "must be positive",
            ));
        }
        if self.outlier_gate_probability > 0.0 && self.outlier_gate_probability < 1.0 {
            Ok(())
        } else {
            Err(invalid_input(
                "yang_outlier_gate_probability",
                "must be in (0, 1)",
            ))
        }
    }
}

/// Configuration for an [`InertialFilter`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct InertialFilterConfig {
    /// IMU stochastic model used for covariance prediction.
    pub imu_spec: ImuSpec,
    /// Measurement-update algorithm used by loose and tight GNSS updates.
    pub filter_kind: FusionFilterKind,
    /// Deterministic IMU calibration applied before mechanization.
    pub imu_model: ImuErrorModel,
    /// Direction cosine matrix rotating IMU sensor axes into vehicle body axes.
    ///
    /// Bias and scale-factor error states remain resolved in IMU axes; corrected
    /// samples are rotated into body axes before mechanization and covariance
    /// prediction.
    pub imu_to_body_dcm: Mat3,
    /// Strapdown mechanization options.
    pub mechanization: MechanizationConfig,
    /// Loose GNSS update options.
    pub loose: LooseCouplingConfig,
    /// Tight raw GNSS update options.
    pub tight: TightCouplingConfig,
    /// UKF correction options used when [`Self::filter_kind`] is [`FusionFilterKind::Ukf`].
    pub ukf_update_options: UkfUpdateOptions,
}

impl InertialFilterConfig {
    /// Build a filter configuration with default calibration and loose settings.
    pub fn new(imu_spec: ImuSpec) -> Result<Self, FusionError> {
        let config = Self {
            imu_spec,
            filter_kind: FusionFilterKind::Ekf,
            imu_model: ImuErrorModel::default(),
            imu_to_body_dcm: mat3_identity(),
            mechanization: MechanizationConfig::default(),
            loose: LooseCouplingConfig::default(),
            tight: TightCouplingConfig::default(),
            ukf_update_options: UkfUpdateOptions::default(),
        };
        config.validate()?;
        Ok(config)
    }

    /// Validate IMU, mechanization, and loose-coupling settings.
    pub fn validate(&self) -> Result<(), FusionError> {
        self.imu_spec.validate().map_err(FusionError::from)?;
        self.imu_model.bias.validate().map_err(FusionError::from)?;
        self.imu_model
            .calibration
            .validate()
            .map_err(FusionError::from)?;
        validate_dcm_orthonormal(&self.imu_to_body_dcm, "imu_to_body_dcm")
            .map_err(FusionError::from)?;
        self.loose.validate()?;
        if self.configures_ukf_prediction_adaptation() {
            return Err(invalid_input(
                "loose_prediction_adaptation",
                "prediction adaptation is defined for EKF loose updates",
            ));
        }
        self.tight.validate()?;
        self.ukf_update_options
            .validate_for_dimension(ERROR_STATE_DIMENSION_15)?;
        self.ukf_update_options
            .validate_for_dimension(ERROR_STATE_DIMENSION_21)
    }

    fn configures_ukf_prediction_adaptation(&self) -> bool {
        self.filter_kind == FusionFilterKind::Ukf && self.loose.prediction_adaptation.is_some()
    }
}

/// Result of one fusion update.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionUpdate {
    /// Whether the update modified the nominal state and covariance.
    pub applied: bool,
    /// Normalized innovation squared for the update rows.
    pub nis: f64,
    /// Number of measurement rows entering the update.
    pub rows: usize,
    /// Number of rows accepted by any configured innovation screen.
    pub accepted_rows: usize,
    /// Number of rows rejected by any configured innovation screen.
    pub rejected_rows: usize,
    /// Full correction report from the selected update primitive.
    pub ekf: EkfCorrectionReport,
}

impl FusionUpdate {
    fn from_report(rows: usize, report: EkfCorrectionReport) -> Self {
        Self {
            applied: report.applied,
            nis: report.normalized_innovation_squared,
            rows,
            accepted_rows: report.accepted_rows,
            rejected_rows: report.rejected_rows,
            ekf: report,
        }
    }
}

/// Closed-loop INS filter with loose GNSS PVT updates.
#[derive(Debug, Clone, PartialEq)]
pub struct InertialFilter {
    pub(super) state: InsFilterState,
    pub(super) config: InertialFilterConfig,
    pub(super) last_body_rate_wrt_ecef_rps: [f64; 3],
    pub(super) stationarity_window: VecDeque<StationarityDetectorSnapshotSample>,
    pub(super) last_stationary_update_t_j2000_s: Option<f64>,
    pub(super) last_non_holonomic_update_t_j2000_s: Option<f64>,
    pub(super) time_sync: TimeSyncHistory,
    pub(super) tight: TightFusionState,
}

impl InertialFilter {
    /// Build a filter with default calibration and loose-coupling settings.
    pub fn new(state: InsFilterState, imu_spec: ImuSpec) -> Result<Self, FusionError> {
        let config = InertialFilterConfig::new(imu_spec)?;
        Self::with_config(state, config)
    }

    /// Build a filter with explicit configuration.
    pub fn with_config(
        state: InsFilterState,
        config: InertialFilterConfig,
    ) -> Result<Self, FusionError> {
        state.validate()?;
        config.validate()?;
        let tight = TightFusionState::from_filter_state(&state, config.tight)?;
        let time_sync = TimeSyncHistory::from_initial(&state, &tight);
        Ok(Self {
            state,
            config,
            last_body_rate_wrt_ecef_rps: [0.0; 3],
            stationarity_window: VecDeque::new(),
            last_stationary_update_t_j2000_s: None,
            last_non_holonomic_update_t_j2000_s: None,
            time_sync,
            tight,
        })
    }

    /// Borrow the current INS filter state.
    pub const fn state(&self) -> &InsFilterState {
        &self.state
    }

    /// Mutably borrow the current INS filter state.
    pub fn state_mut(&mut self) -> &mut InsFilterState {
        &mut self.state
    }

    /// Borrow the immutable filter configuration.
    pub const fn config(&self) -> &InertialFilterConfig {
        &self.config
    }

    /// Return the most recent body angular rate relative to ECEF, resolved in body axes.
    pub const fn last_body_rate_wrt_ecef_rps(&self) -> [f64; 3] {
        self.last_body_rate_wrt_ecef_rps
    }

    /// Propagate the nominal INS state and error covariance with one IMU sample.
    pub fn propagate(&mut self, sample: ImuSample) -> Result<&InsFilterState, FusionError> {
        let previous_t_j2000_s = self.state.nominal.t_j2000_s;
        self.time_sync
            .validate_next_imu(previous_t_j2000_s, sample)?;
        self.propagate_core(sample)?;
        self.time_sync.push_imu(previous_t_j2000_s, sample);
        Ok(&self.state)
    }

    pub(super) fn propagate_core(
        &mut self,
        sample: ImuSample,
    ) -> Result<FusionPredictionStep, FusionError> {
        self.state.validate()?;
        self.config.validate()?;
        self.tight.align_with_filter_state(&self.state)?;

        let previous = self.state.nominal;
        let imu_model = self.effective_imu_model()?;
        let increment = rotate_increment_imu_to_body(
            imu_model
                .correct_sample(&sample, previous.t_j2000_s)
                .map_err(FusionError::from)?,
            self.config.imu_to_body_dcm,
        );
        let kinematics = ErrorStateImuKinematics::new(
            scale3(increment.delta_velocity_mps, 1.0 / increment.dt_s),
            scale3(increment.delta_theta_rad, 1.0 / increment.dt_s),
        )?;
        let linearization = linearize_error_state_ecef_with_imu_to_body(
            &previous,
            kinematics,
            &self.config.imu_spec,
            increment.dt_s,
            self.state.layout(),
            self.config.imu_to_body_dcm,
        )?;
        let next_nominal = mechanize_ecef(&previous, &increment, self.config.mechanization)
            .map_err(FusionError::from)?;
        let body_rate_wrt_ecef_rps = body_rate_relative_to_ecef(
            &next_nominal.attitude_body_to_ecef,
            kinematics.angular_rate_body_rps,
        );

        predict_error_state_covariance(
            &mut self.state.covariance,
            &linearization.phi,
            &linearization.q_d,
        )?;
        self.tight.predict_covariance(
            &linearization.phi,
            &linearization.q_d,
            increment.dt_s,
            self.config.tight,
        )?;
        self.tight.copy_base_covariance_to_state(&mut self.state)?;
        self.state.nominal = next_nominal;
        self.state.reset_error_state();
        self.last_body_rate_wrt_ecef_rps = body_rate_wrt_ecef_rps;
        self.record_stationarity_sample(
            kinematics.specific_force_body_mps2,
            body_rate_wrt_ecef_rps,
        )?;
        self.state.validate()?;
        Ok(FusionPredictionStep {
            transition: linearization.phi,
        })
    }

    /// Apply a loose GNSS PVT update at the current propagated epoch.
    ///
    /// GNSS epochs must be strictly increasing across the filter's stateful
    /// update surface, matching the standalone time-sync order validator;
    /// duplicate or regressed epochs are rejected rather than fused twice.
    pub fn update_loose(
        &mut self,
        measurement: &GnssFixMeasurement,
    ) -> Result<FusionUpdate, FusionError> {
        if let Some(last) = self.time_sync.last_measurement_t_j2000_s() {
            if measurement.t_j2000_s <= last {
                return Err(invalid_input(
                    "t_j2000_s",
                    "GNSS measurement epochs must be strictly increasing",
                ));
            }
        }
        let update = self.update_loose_core(measurement)?;
        let snapshot = self.snapshot();
        self.time_sync
            .push_loose_measurement_and_checkpoint(measurement.clone(), snapshot);
        Ok(update)
    }

    pub(super) fn update_loose_core(
        &mut self,
        measurement: &GnssFixMeasurement,
    ) -> Result<FusionUpdate, FusionError> {
        let correction = loose_coupling_correction_with_imu_to_body(
            &self.state,
            measurement,
            self.config.loose.lever_arm_body_m,
            self.last_body_rate_wrt_ecef_rps,
            self.config.imu_to_body_dcm,
        )?;
        let correction = apply_fix_status_weighting(
            correction,
            measurement.fix_status,
            self.config.loose.fix_status_weighting,
        )?;
        self.apply_loose_style_correction(correction, self.config.loose)
    }

    /// Apply a gated zero-velocity and zero-angular-rate update.
    pub fn update_stationary(&mut self) -> Result<Option<FusionUpdate>, FusionError> {
        let Some(config) = self.config.loose.stationary_updates else {
            return Ok(None);
        };
        if !self.is_stationary(config.detector)? {
            return Ok(None);
        }
        let update_t_j2000_s = self.state.nominal.t_j2000_s;
        if self.last_stationary_update_t_j2000_s == Some(update_t_j2000_s) {
            return Err(invalid_input(
                "t_j2000_s",
                "stationary update already applied at this epoch",
            ));
        }
        let correction = stationary_correction(
            &self.state,
            self.last_body_rate_wrt_ecef_rps,
            config,
            self.config.imu_to_body_dcm,
        )?;
        let update =
            self.apply_loose_style_correction(correction, self.pseudo_measurement_config())?;
        self.last_stationary_update_t_j2000_s = Some(update_t_j2000_s);
        Ok(Some(update))
    }

    /// Apply a gated wheeled-vehicle non-holonomic constraint update.
    pub fn update_non_holonomic(&mut self) -> Result<Option<FusionUpdate>, FusionError> {
        let Some(config) = self.config.loose.non_holonomic else {
            return Ok(None);
        };
        if !self.nhc_motion_gate(config)? {
            return Ok(None);
        }
        let update_t_j2000_s = self.state.nominal.t_j2000_s;
        if self.last_non_holonomic_update_t_j2000_s == Some(update_t_j2000_s) {
            return Err(invalid_input(
                "t_j2000_s",
                "non-holonomic update already applied at this epoch",
            ));
        }
        let correction = non_holonomic_correction(&self.state, config)?;
        let update =
            self.apply_loose_style_correction(correction, self.pseudo_measurement_config())?;
        self.last_non_holonomic_update_t_j2000_s = Some(update_t_j2000_s);
        Ok(Some(update))
    }

    fn apply_loose_style_correction(
        &mut self,
        correction: EkfCorrection,
        loose_config: LooseCouplingConfig,
    ) -> Result<FusionUpdate, FusionError> {
        let prepared = prepare_loose_correction(&self.state, correction, loose_config)?;
        let rows = prepared.correction.row_count();
        let filter_kind = self.config.filter_kind;
        let ekf_options = self.config.loose.update_options;
        let ukf_options = self.config.ukf_update_options;
        let report = match filter_kind {
            FusionFilterKind::Ekf => {
                if prepared.predicted_covariance_scale == 1.0 {
                    ekf_correct_closed_loop(&mut self.state, &prepared.correction, ekf_options)?
                } else {
                    ekf_correct_closed_loop_with_predicted_covariance_scale(
                        &mut self.state,
                        &prepared.correction,
                        ekf_options,
                        prepared.predicted_covariance_scale,
                    )?
                }
            }
            FusionFilterKind::Ukf => {
                ukf_correct_closed_loop(&mut self.state, &prepared.correction, ukf_options)?
            }
        };
        self.tight
            .replace_base_covariance_and_clear_cross(&self.state.covariance)?;
        Ok(FusionUpdate::from_report(rows, report))
    }

    fn pseudo_measurement_config(&self) -> LooseCouplingConfig {
        LooseCouplingConfig {
            measurement_reweighting: None,
            prediction_adaptation: None,
            ..self.config.loose
        }
    }

    fn record_stationarity_sample(
        &mut self,
        specific_force_body_mps2: [f64; 3],
        body_rate_wrt_ecef_rps: [f64; 3],
    ) -> Result<(), FusionError> {
        let gravity_norm_mps2 = norm3(gravity_ecef_mps2(self.state.nominal.position_ecef_m)?);
        let sample = StationarityDetectorSnapshotSample {
            specific_force_norm_error_mps2: (norm3(specific_force_body_mps2) - gravity_norm_mps2)
                .abs(),
            body_rate_wrt_ecef_norm_rps: norm3(body_rate_wrt_ecef_rps),
        };
        validate_finite(
            sample.specific_force_norm_error_mps2,
            "specific_force_norm_error_mps2",
        )
        .map_err(FusionError::from)?;
        validate_finite(
            sample.body_rate_wrt_ecef_norm_rps,
            "body_rate_wrt_ecef_norm_rps",
        )
        .map_err(FusionError::from)?;
        self.stationarity_window.push_back(sample);
        let max_len = self
            .config
            .loose
            .stationary_updates
            .map_or(1, |config| config.detector.window_len);
        while self.stationarity_window.len() > max_len {
            self.stationarity_window.pop_front();
        }
        Ok(())
    }

    fn is_stationary(&self, detector: StationaryDetectorConfig) -> Result<bool, FusionError> {
        detector.validate()?;
        if self.stationarity_window.len() < detector.window_len {
            return Ok(false);
        }
        Ok(self
            .stationarity_window
            .iter()
            .rev()
            .take(detector.window_len)
            .all(|sample| {
                sample.specific_force_norm_error_mps2 <= detector.max_specific_force_norm_error_mps2
                    && sample.body_rate_wrt_ecef_norm_rps
                        <= detector.max_body_rate_wrt_ecef_norm_rps
            }))
    }

    fn nhc_motion_gate(&self, config: NonHolonomicConstraintConfig) -> Result<bool, FusionError> {
        config.validate()?;
        let speed_mps = norm3(self.state.nominal.velocity_ecef_mps);
        validate_finite(speed_mps, "nhc_speed_mps").map_err(FusionError::from)?;
        Ok(speed_mps >= config.min_speed_mps
            && norm3(self.last_body_rate_wrt_ecef_rps) <= config.max_body_rate_wrt_ecef_norm_rps)
    }

    fn effective_imu_model(&self) -> Result<ImuErrorModel, FusionError> {
        let mut bias = self.config.imu_model.bias;
        for axis in 0..3 {
            bias.accel_mps2[axis] += self.state.nominal.accel_bias_mps2[axis];
            bias.gyro_rps[axis] += self.state.nominal.gyro_bias_rps[axis];
        }
        let calibration = effective_calibration(
            self.config.imu_model.calibration,
            self.state.accel_scale_factor,
            self.state.gyro_scale_factor,
        )?;
        let model = ImuErrorModel { bias, calibration };
        model.bias.validate().map_err(FusionError::from)?;
        model.calibration.validate().map_err(FusionError::from)?;
        Ok(model)
    }
}

/// Build the loose-coupled GNSS EKF correction for an INS state.
///
/// The returned design matrix follows the nominal-error convention used by
/// [`InsFilterState`]: navigation errors are subtracted during closed-loop reset,
/// while bias errors are added to the closed-loop bias estimates.
///
/// `body_rate_wrt_ecef_rps` is the body angular rate relative to ECEF, resolved
/// in body axes. A body fixed in ECEF supplies zero for this rate even though
/// its gyroscopes measure Earth rate.
pub fn loose_coupling_correction(
    state: &InsFilterState,
    measurement: &GnssFixMeasurement,
    lever_arm_body_m: [f64; 3],
    body_rate_wrt_ecef_rps: [f64; 3],
) -> Result<EkfCorrection, FusionError> {
    loose_coupling_correction_with_imu_to_body(
        state,
        measurement,
        lever_arm_body_m,
        body_rate_wrt_ecef_rps,
        mat3_identity(),
    )
}

fn loose_coupling_correction_with_imu_to_body(
    state: &InsFilterState,
    measurement: &GnssFixMeasurement,
    lever_arm_body_m: [f64; 3],
    body_rate_wrt_ecef_rps: [f64; 3],
    imu_to_body_dcm: Mat3,
) -> Result<EkfCorrection, FusionError> {
    state.validate()?;
    measurement.validate()?;
    validate_vec3(lever_arm_body_m, "lever_arm_body_m").map_err(FusionError::from)?;
    validate_vec3(body_rate_wrt_ecef_rps, "body_rate_wrt_ecef_rps").map_err(FusionError::from)?;
    validate_dcm_orthonormal(&imu_to_body_dcm, "imu_to_body_dcm").map_err(FusionError::from)?;
    if measurement.t_j2000_s != state.nominal.t_j2000_s {
        return Err(invalid_input("t_j2000_s", "must equal nominal state epoch"));
    }

    let dimension = state.dimension();
    let c_b_e = state.nominal.attitude_body_to_ecef;
    let lever_ecef_m = mul_vec3(&c_b_e, lever_arm_body_m);
    let antenna_position_ecef_m = add3(state.nominal.position_ecef_m, lever_ecef_m);
    let lever_velocity_body_mps = cross3(body_rate_wrt_ecef_rps, lever_arm_body_m);
    let lever_velocity_ecef_mps = mul_vec3(&c_b_e, lever_velocity_body_mps);
    let antenna_velocity_ecef_mps = add3(state.nominal.velocity_ecef_mps, lever_velocity_ecef_mps);

    let mut innovation = Vec::with_capacity(measurement.row_count());
    let mut design = Vec::with_capacity(measurement.row_count());
    let position_residual = sub3(measurement.position_ecef_m, antenna_position_ecef_m);
    let lever_position_skew = skew(lever_ecef_m);
    for axis in 0..3 {
        let mut row = vec![0.0; dimension];
        row[ERROR_POSITION_INDEX + axis] = -1.0;
        row[ERROR_ATTITUDE_INDEX..ERROR_ATTITUDE_INDEX + 3]
            .copy_from_slice(&lever_position_skew[axis]);
        innovation.push(position_residual[axis]);
        design.push(row);
    }

    if let Some(velocity_ecef_mps) = measurement.velocity_ecef_mps {
        let velocity_residual = sub3(velocity_ecef_mps, antenna_velocity_ecef_mps);
        let lever_velocity_skew = skew(lever_velocity_ecef_mps);
        let gyro_bias_block = inline_rxr(
            &inline_rxr(&c_b_e, &skew(lever_arm_body_m)),
            &imu_to_body_dcm,
        );
        for axis in 0..3 {
            let mut row = vec![0.0; dimension];
            row[ERROR_VELOCITY_INDEX + axis] = -1.0;
            row[ERROR_ATTITUDE_INDEX..ERROR_ATTITUDE_INDEX + 3]
                .copy_from_slice(&lever_velocity_skew[axis]);
            row[ERROR_GYRO_BIAS_INDEX..ERROR_GYRO_BIAS_INDEX + 3]
                .copy_from_slice(&gyro_bias_block[axis]);
            innovation.push(velocity_residual[axis]);
            design.push(row);
        }
    }

    EkfCorrection::new(innovation, design, measurement.covariance.clone())
}

fn stationary_correction(
    state: &InsFilterState,
    body_rate_wrt_ecef_rps: [f64; 3],
    config: StationaryUpdateConfig,
    imu_to_body_dcm: Mat3,
) -> Result<EkfCorrection, FusionError> {
    state.validate()?;
    config.validate()?;
    validate_vec3(body_rate_wrt_ecef_rps, "body_rate_wrt_ecef_rps").map_err(FusionError::from)?;
    validate_dcm_orthonormal(&imu_to_body_dcm, "imu_to_body_dcm").map_err(FusionError::from)?;
    let dimension = state.dimension();
    let mut innovation = Vec::with_capacity(ZUPT_ZARU_ROWS);
    let mut design = Vec::with_capacity(ZUPT_ZARU_ROWS);
    let mut covariance = vec![vec![0.0; ZUPT_ZARU_ROWS]; ZUPT_ZARU_ROWS];
    let velocity_variance = config.zero_velocity_sigma_mps * config.zero_velocity_sigma_mps;
    let rate_variance = config.zero_angular_rate_sigma_rps * config.zero_angular_rate_sigma_rps;

    for axis in 0..3 {
        let mut row = vec![0.0; dimension];
        row[ERROR_VELOCITY_INDEX + axis] = -1.0;
        innovation.push(-state.nominal.velocity_ecef_mps[axis]);
        covariance[axis][axis] = velocity_variance;
        design.push(row);
    }
    for axis in 0..3 {
        let mut row = vec![0.0; dimension];
        for col in 0..3 {
            row[ERROR_GYRO_BIAS_INDEX + col] = -imu_to_body_dcm[axis][col];
        }
        innovation.push(-body_rate_wrt_ecef_rps[axis]);
        covariance[3 + axis][3 + axis] = rate_variance;
        design.push(row);
    }

    EkfCorrection::new(innovation, design, covariance)
}

fn non_holonomic_correction(
    state: &InsFilterState,
    config: NonHolonomicConstraintConfig,
) -> Result<EkfCorrection, FusionError> {
    state.validate()?;
    config.validate()?;
    let dimension = state.dimension();
    let c_e_b = inline_tr(&state.nominal.attitude_body_to_ecef);
    let velocity_body_mps = mul_vec3(&c_e_b, state.nominal.velocity_ecef_mps);
    let attitude_block = inline_rxr(&c_e_b, &skew(state.nominal.velocity_ecef_mps));
    let mut innovation = Vec::with_capacity(NHC_ROWS);
    let mut design = Vec::with_capacity(NHC_ROWS);
    let mut covariance = vec![vec![0.0; NHC_ROWS]; NHC_ROWS];
    let constrained_axes = [1usize, 2usize];

    for (row_idx, body_axis) in constrained_axes.into_iter().enumerate() {
        let mut row = vec![0.0; dimension];
        for axis in 0..3 {
            row[ERROR_VELOCITY_INDEX + axis] = -c_e_b[body_axis][axis];
            row[ERROR_ATTITUDE_INDEX + axis] = -attitude_block[body_axis][axis];
        }
        innovation.push(-velocity_body_mps[body_axis]);
        design.push(row);
        let sigma = if body_axis == 1 {
            config.lateral_velocity_sigma_mps
        } else {
            config.vertical_velocity_sigma_mps
        };
        covariance[row_idx][row_idx] = sigma * sigma;
    }

    EkfCorrection::new(innovation, design, covariance)
}

fn apply_fix_status_weighting(
    correction: EkfCorrection,
    status: GnssFixStatus,
    weighting: GnssFixStatusWeighting,
) -> Result<EkfCorrection, FusionError> {
    weighting.validate()?;
    let multiplier = weighting.multiplier(status);
    if multiplier.to_bits() == 1.0_f64.to_bits() {
        return Ok(correction);
    }
    let variance_scale = multiplier * multiplier;
    let covariance = correction
        .measurement_covariance
        .iter()
        .map(|row| row.iter().map(|value| value * variance_scale).collect())
        .collect();
    EkfCorrection::new(correction.innovation, correction.design, covariance)
}

fn rotate_increment_imu_to_body(
    increment: crate::inertial::CorrectedImuIncrement,
    imu_to_body_dcm: Mat3,
) -> crate::inertial::CorrectedImuIncrement {
    if imu_to_body_dcm == mat3_identity() {
        return increment;
    }
    crate::inertial::CorrectedImuIncrement {
        delta_velocity_mps: mul_vec3(&imu_to_body_dcm, increment.delta_velocity_mps),
        delta_theta_rad: mul_vec3(&imu_to_body_dcm, increment.delta_theta_rad),
        ..increment
    }
}

/// One position/velocity sample used by velocity matching.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityMatchState {
    /// Sample epoch in seconds since J2000.
    pub t_j2000_s: f64,
    /// INS position in ECEF meters.
    pub position_ecef_m: [f64; 3],
    /// INS velocity in ECEF meters per second.
    pub velocity_ecef_mps: [f64; 3],
}

impl VelocityMatchState {
    /// Build and validate one velocity-matching sample.
    pub fn new(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        velocity_ecef_mps: [f64; 3],
    ) -> Result<Self, FusionError> {
        validate_finite(t_j2000_s, "t_j2000_s").map_err(FusionError::from)?;
        validate_vec3(position_ecef_m, "position_ecef_m").map_err(FusionError::from)?;
        validate_vec3(velocity_ecef_mps, "velocity_ecef_mps").map_err(FusionError::from)?;
        Ok(Self {
            t_j2000_s,
            position_ecef_m,
            velocity_ecef_mps,
        })
    }
}

/// Output from endpoint velocity matching across one outage.
#[derive(Debug, Clone, PartialEq)]
pub struct VelocityMatchedTrajectory {
    /// Corrected samples in the same order as the input span.
    pub states: Vec<VelocityMatchState>,
    /// Position correction applied at the return-fix endpoint.
    pub endpoint_position_correction_ecef_m: [f64; 3],
    /// Velocity correction applied at the return-fix endpoint.
    pub endpoint_velocity_correction_ecef_mps: [f64; 3],
}

/// Blend a first good post-outage fix back over an outage span.
///
/// The input span starts at the last pre-outage state and ends at the
/// pre-update return-fix state. The first sample keeps zero correction, the
/// final sample matches the GNSS position and velocity, and interior samples
/// receive a cubic Hermite endpoint correction. When a caller has already
/// applied the return fix and wants continuity into the posterior trajectory,
/// use [`velocity_match_outage_to_state`] with that post-update endpoint.
pub fn velocity_match_outage(
    states: &[VelocityMatchState],
    first_good_fix: &GnssFixMeasurement,
    config: VelocityMatchingConfig,
) -> Result<VelocityMatchedTrajectory, FusionError> {
    first_good_fix.validate()?;
    let Some(fix_velocity) = first_good_fix.velocity_ecef_mps else {
        return Err(invalid_input(
            "velocity_ecef_mps",
            "return fix must include velocity",
        ));
    };
    let endpoint = VelocityMatchState::new(
        first_good_fix.t_j2000_s,
        first_good_fix.position_ecef_m,
        fix_velocity,
    )?;
    velocity_match_outage_to_state(states, endpoint, config)
}

/// Blend an outage span to a caller-supplied endpoint state.
///
/// Use this when the first good post-outage GNSS fix has already been fused and
/// continuity should land on the posterior filter state rather than the raw
/// GNSS position/velocity measurement.
pub fn velocity_match_outage_to_state(
    states: &[VelocityMatchState],
    endpoint: VelocityMatchState,
    config: VelocityMatchingConfig,
) -> Result<VelocityMatchedTrajectory, FusionError> {
    config.validate()?;
    if states.len() < 2 {
        return Err(invalid_input(
            "velocity_match_states",
            "must contain at least two states",
        ));
    }
    for state in states {
        VelocityMatchState::new(
            state.t_j2000_s,
            state.position_ecef_m,
            state.velocity_ecef_mps,
        )?;
    }
    for pair in states.windows(2) {
        if pair[1].t_j2000_s <= pair[0].t_j2000_s {
            return Err(invalid_input(
                "velocity_match_states",
                "epochs must be strictly increasing",
            ));
        }
    }
    let first = states[0];
    let last = states[states.len() - 1];
    let endpoint = VelocityMatchState::new(
        endpoint.t_j2000_s,
        endpoint.position_ecef_m,
        endpoint.velocity_ecef_mps,
    )?;
    if endpoint.t_j2000_s != last.t_j2000_s {
        return Err(invalid_input(
            "t_j2000_s",
            "endpoint state must match the last state epoch",
        ));
    }
    let duration_s = last.t_j2000_s - first.t_j2000_s;
    validate_positive(duration_s, "velocity_match_duration_s")?;
    if duration_s > config.max_outage_duration_s {
        return Err(invalid_input(
            "velocity_match_duration_s",
            "exceeds configured maximum",
        ));
    }

    let endpoint_position_correction_ecef_m = sub3(endpoint.position_ecef_m, last.position_ecef_m);
    let endpoint_velocity_correction_ecef_mps =
        sub3(endpoint.velocity_ecef_mps, last.velocity_ecef_mps);
    let mut matched = Vec::with_capacity(states.len());
    for state in states {
        let tau = (state.t_j2000_s - first.t_j2000_s) / duration_s;
        let tau2 = tau * tau;
        let tau3 = tau2 * tau;
        let h01 = -2.0 * tau3 + 3.0 * tau2;
        let h11 = tau3 - tau2;
        let dh01 = (-6.0 * tau2 + 6.0 * tau) / duration_s;
        let dh11 = 3.0 * tau2 - 2.0 * tau;
        let mut position = state.position_ecef_m;
        let mut velocity = state.velocity_ecef_mps;
        for axis in 0..3 {
            position[axis] += h01 * endpoint_position_correction_ecef_m[axis]
                + duration_s * h11 * endpoint_velocity_correction_ecef_mps[axis];
            velocity[axis] += dh01 * endpoint_position_correction_ecef_m[axis]
                + dh11 * endpoint_velocity_correction_ecef_mps[axis];
        }
        matched.push(VelocityMatchState::new(
            state.t_j2000_s,
            position,
            velocity,
        )?);
    }

    Ok(VelocityMatchedTrajectory {
        states: matched,
        endpoint_position_correction_ecef_m,
        endpoint_velocity_correction_ecef_mps,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct PreparedLooseCorrection {
    correction: EkfCorrection,
    predicted_covariance_scale: f64,
}

fn prepare_loose_correction(
    state: &InsFilterState,
    correction: EkfCorrection,
    config: LooseCouplingConfig,
) -> Result<PreparedLooseCorrection, FusionError> {
    if config.measurement_reweighting.is_none() && config.prediction_adaptation.is_none() {
        return Ok(PreparedLooseCorrection {
            correction,
            predicted_covariance_scale: 1.0,
        });
    }

    let raw_innovation_covariance = innovation_covariance(&state.covariance, &correction)?;
    let correction = if let Some(reweighting) = config.measurement_reweighting {
        apply_igg_iii_reweighting(&correction, &raw_innovation_covariance, reweighting)?
    } else {
        correction
    };
    let predicted_covariance_scale = if let Some(adaptation) = config.prediction_adaptation {
        yang_predicted_covariance_scale(state, &correction, &raw_innovation_covariance, adaptation)?
    } else {
        1.0
    };

    Ok(PreparedLooseCorrection {
        correction,
        predicted_covariance_scale,
    })
}

fn apply_igg_iii_reweighting(
    correction: &EkfCorrection,
    innovation_covariance: &[Vec<f64>],
    reweighting: IggIiiMeasurementReweighting,
) -> Result<EkfCorrection, FusionError> {
    reweighting.validate()?;
    let mut gammas = Vec::with_capacity(correction.row_count());
    let mut all_one = true;
    for (row, s_row) in innovation_covariance
        .iter()
        .enumerate()
        .take(correction.row_count())
    {
        let variance = s_row[row];
        validate_positive(variance, "innovation_covariance_diagonal")?;
        let standardized = (correction.innovation[row] / variance.sqrt()).abs();
        let gamma =
            igg_iii_variance_scale(standardized, reweighting.k0_sigma, reweighting.k1_sigma);
        all_one &= gamma.to_bits() == 1.0_f64.to_bits();
        gammas.push(gamma);
    }

    if all_one {
        return Ok(correction.clone());
    }

    let covariance = inflate_measurement_covariance(&correction.measurement_covariance, &gammas);
    EkfCorrection::new(
        correction.innovation.clone(),
        correction.design.clone(),
        covariance,
    )
}

fn igg_iii_variance_scale(abs_standardized: f64, k0_sigma: f64, k1_sigma: f64) -> f64 {
    if abs_standardized <= k0_sigma {
        1.0
    } else if abs_standardized < k1_sigma {
        let ratio = abs_standardized / k0_sigma;
        let transition = (k1_sigma - k0_sigma) / (k1_sigma - abs_standardized);
        ratio * transition * transition
    } else {
        IGG_III_REJECTION_VARIANCE_SCALE
    }
}

fn inflate_measurement_covariance(covariance: &[Vec<f64>], gammas: &[f64]) -> Vec<Vec<f64>> {
    let sqrt_gammas = gammas.iter().map(|gamma| gamma.sqrt()).collect::<Vec<_>>();
    let mut inflated = covariance.to_vec();
    for row in 0..inflated.len() {
        for col in 0..inflated[row].len() {
            inflated[row][col] *= sqrt_gammas[row] * sqrt_gammas[col];
        }
    }
    inflated
}

fn yang_predicted_covariance_scale(
    state: &InsFilterState,
    correction: &EkfCorrection,
    raw_innovation_covariance: &[Vec<f64>],
    adaptation: YangPredictionAdaptiveFactor,
) -> Result<f64, FusionError> {
    adaptation.validate()?;
    let raw_mahalanobis =
        normalized_innovation_squared(raw_innovation_covariance, &correction.innovation)?;
    let outlier_threshold =
        crate::quality::chi2_inv(adaptation.outlier_gate_probability, correction.row_count())
            .map_err(|_| {
                invalid_input(
                    "yang_outlier_gate_probability",
                    "must produce a chi-square threshold",
                )
            })?;
    // Jiang and Zhang, Sensors 2018: innovation-driven adaptation is disabled
    // when the Mahalanobis gate flags a measurement outlier.
    if raw_mahalanobis > outlier_threshold {
        return Ok(1.0);
    }

    let innovation_covariance = innovation_covariance(&state.covariance, correction)?;
    let trace = innovation_covariance
        .iter()
        .enumerate()
        .map(|(idx, row)| row[idx])
        .sum::<f64>();
    validate_positive(trace, "innovation_covariance_trace")?;
    let squared_norm = correction
        .innovation
        .iter()
        .map(|value| value * value)
        .sum::<f64>();
    let statistic = squared_norm / trace;
    if statistic <= adaptation.threshold {
        Ok(1.0)
    } else {
        Ok(statistic / adaptation.threshold)
    }
}

fn body_rate_relative_to_ecef(
    attitude_body_to_ecef: &Mat3,
    inertial_body_rate_rps: [f64; 3],
) -> [f64; 3] {
    let attitude_ecef_to_body = inline_tr(attitude_body_to_ecef);
    let earth_rate_body_rps = mul_vec3(&attitude_ecef_to_body, [0.0, 0.0, OMEGA_E_DOT_RAD_S]);
    sub3(inertial_body_rate_rps, earth_rate_body_rps)
}

fn effective_calibration(
    base: ImuCalibration,
    accel_scale_factor: [f64; 3],
    gyro_scale_factor: [f64; 3],
) -> Result<ImuCalibration, FusionError> {
    let mut calibration = base;
    for axis in 0..3 {
        calibration.accel_scale_misalignment[axis][axis] += accel_scale_factor[axis];
        calibration.gyro_scale_misalignment[axis][axis] += gyro_scale_factor[axis];
    }
    calibration.validate().map_err(FusionError::from)?;
    Ok(calibration)
}

fn mat3_to_rows(matrix: [[f64; 3]; 3]) -> Vec<Vec<f64>> {
    matrix.into_iter().map(Vec::from).collect()
}

#[cfg(test)]
mod tests {
    //! Provenance: loose-coupled GNSS/INS equations follow Groves, Principles
    //! of GNSS, Inertial, and Multisensor Integrated Navigation Systems, 2nd
    //! ed., Chapter 14, with the lever-arm position and velocity model stated
    //! in the build spec. Synthetic noise uses the SplitMix64 sequence pattern
    //! from `astro/propagator/covariance.rs`. NEES/NIS consistency bands use
    //! the Bar-Shalom two-sided chi-square test.

    use super::*;
    use crate::astro::constants::earth::{OMEGA_E_DOT_RAD_S, WGS84_A_M};
    use crate::astro::math::mat3::{inline_tr, Mat3};
    use crate::astro::math::vec3::{dot3, norm3};
    use crate::fusion::state::{
        ERROR_ACCEL_BIAS_INDEX, ERROR_GYRO_BIAS_INDEX, ERROR_STATE_DIMENSION_15,
    };
    use crate::inertial::frames::gravity_ecef_mps2;
    use crate::inertial::state::{mat3_identity, mat3_mul, mat3_mul_vec, reorthonormalize_dcm};
    use crate::inertial::{CorrectedImuIncrement, NavState};
    use nalgebra::{DMatrix, DVector};

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.17e}"
        );
    }

    fn covariance_from_diag(diagonal: &[f64]) -> Vec<Vec<f64>> {
        let mut covariance = vec![vec![0.0; diagonal.len()]; diagonal.len()];
        for (idx, value) in diagonal.iter().enumerate() {
            covariance[idx][idx] = *value;
        }
        covariance
    }

    fn reference_filter_state(
        nominal: NavState,
        diagonal: &[f64],
    ) -> Result<InsFilterState, FusionError> {
        InsFilterState::from_diagonal(
            nominal,
            super::super::state::ErrorStateLayout::Fifteen,
            diagonal,
        )
    }

    #[test]
    fn loose_correction_builds_lever_arm_rows_and_keeps_input_covariance() {
        let state = reference_filter_state(
            NavState::new(10.0, [10.0, 20.0, 30.0], [1.0, 2.0, 3.0], mat3_identity())
                .expect("state"),
            &[1.0; ERROR_STATE_DIMENSION_15],
        )
        .expect("filter state");
        let lever = [0.5, -1.0, 2.0];
        let omega = [0.1, 0.2, -0.3];
        let lever_position = lever;
        let lever_velocity = cross3(omega, lever);
        let position_residual = [1.0, -2.0, 3.0];
        let velocity_residual = [0.4, -0.5, 0.6];
        let covariance = covariance_from_diag(&[4.0, 5.0, 6.0, 0.7, 0.8, 0.9]);
        let measurement = GnssFixMeasurement::position_velocity(
            10.0,
            add3(
                add3(state.nominal.position_ecef_m, lever_position),
                position_residual,
            ),
            add3(
                add3(state.nominal.velocity_ecef_mps, lever_velocity),
                velocity_residual,
            ),
            covariance.clone(),
            6,
        )
        .expect("measurement");

        let correction =
            loose_coupling_correction(&state, &measurement, lever, omega).expect("correction");

        for axis in 0..3 {
            assert_close(
                correction.innovation[axis],
                position_residual[axis],
                2.0e-16,
            );
            assert_close(
                correction.innovation[3 + axis],
                velocity_residual[axis],
                2.0e-16,
            );
        }
        assert_eq!(correction.measurement_covariance, covariance);
        assert_eq!(
            correction.design[0][ERROR_POSITION_INDEX].to_bits(),
            (-1.0_f64).to_bits()
        );
        assert_eq!(
            correction.design[1][ERROR_POSITION_INDEX + 1].to_bits(),
            (-1.0_f64).to_bits()
        );
        let lever_skew = skew(lever);
        for (row, expected_row) in lever_skew.iter().enumerate() {
            for (col, expected) in expected_row.iter().enumerate() {
                assert_eq!(
                    correction.design[row][ERROR_ATTITUDE_INDEX + col].to_bits(),
                    expected.to_bits()
                );
            }
        }
        let gyro_block = skew(lever);
        for (row, expected_row) in gyro_block.iter().enumerate() {
            for (col, expected) in expected_row.iter().enumerate() {
                assert_eq!(
                    correction.design[3 + row][ERROR_GYRO_BIAS_INDEX + col].to_bits(),
                    expected.to_bits()
                );
            }
        }
    }

    #[test]
    fn loose_and_zaru_jacobians_rotate_imu_bias_axes() {
        let state = reference_filter_state(
            NavState::new(10.0, [10.0, 20.0, 30.0], [1.0, 2.0, 3.0], mat3_identity())
                .expect("state"),
            &[1.0; ERROR_STATE_DIMENSION_15],
        )
        .expect("filter state");
        let lever = [0.5, -1.0, 2.0];
        let imu_to_body = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let measurement = GnssFixMeasurement::position_velocity(
            10.0,
            state.nominal.position_ecef_m,
            state.nominal.velocity_ecef_mps,
            covariance_from_diag(&[1.0; 6]),
            6,
        )
        .expect("measurement");
        let correction = loose_coupling_correction_with_imu_to_body(
            &state,
            &measurement,
            lever,
            [0.0; 3],
            imu_to_body,
        )
        .expect("correction");
        let expected_gyro_block = inline_rxr(&skew(lever), &imu_to_body);
        for (row, expected_row) in expected_gyro_block.iter().enumerate() {
            for (col, expected) in expected_row.iter().enumerate() {
                assert_eq!(
                    correction.design[3 + row][ERROR_GYRO_BIAS_INDEX + col].to_bits(),
                    expected.to_bits()
                );
            }
        }

        let stationary = stationary_correction(
            &state,
            [0.01, -0.02, 0.03],
            StationaryUpdateConfig {
                detector: StationaryDetectorConfig {
                    window_len: 1,
                    max_specific_force_norm_error_mps2: 1.0,
                    max_body_rate_wrt_ecef_norm_rps: 1.0,
                },
                zero_velocity_sigma_mps: 0.1,
                zero_angular_rate_sigma_rps: 0.01,
            },
            imu_to_body,
        )
        .expect("stationary correction");
        for (row, expected_row) in imu_to_body.iter().enumerate() {
            for (col, expected) in expected_row.iter().enumerate() {
                assert_eq!(
                    stationary.design[3 + row][ERROR_GYRO_BIAS_INDEX + col].to_bits(),
                    (-*expected).to_bits()
                );
            }
        }
    }

    #[test]
    fn propagated_static_ecef_body_reports_zero_lever_velocity() {
        let lever = [1.0, 0.5, -0.25];
        let truth =
            NavState::new(0.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity()).expect("truth");
        let state =
            reference_filter_state(truth, &[1.0; ERROR_STATE_DIMENSION_15]).expect("filter state");
        let spec = ImuSpec::datasheet(
            0.0,
            0.0,
            0.0,
            0.0,
            crate::inertial::config::RANDOM_WALK_BIAS_TAU_S,
            crate::inertial::config::RANDOM_WALK_BIAS_TAU_S,
            None,
            None,
        );
        let mut config = InertialFilterConfig::new(spec).expect("config");
        config.loose.lever_arm_body_m = lever;
        let mut filter = InertialFilter::with_config(state, config).expect("filter");
        let (truth_next, sample, truth_body_rate_wrt_ecef) =
            inverted_static_sample(&truth, 1.0, 1.0, [0.0; 3], [0.0; 3]);

        for value in truth_body_rate_wrt_ecef {
            assert_close(value, 0.0, 0.0);
        }
        filter.propagate(sample).expect("propagate");
        for value in filter.last_body_rate_wrt_ecef_rps() {
            assert_close(value, 0.0, 0.0);
        }

        let antenna_position = add3(
            truth_next.position_ecef_m,
            mul_vec3(&truth_next.attitude_body_to_ecef, lever),
        );
        let measurement = GnssFixMeasurement::position_velocity(
            truth_next.t_j2000_s,
            antenna_position,
            truth_next.velocity_ecef_mps,
            covariance_from_diag(&[1.0, 1.0, 1.0, 1.0e-6, 1.0e-6, 1.0e-6]),
            8,
        )
        .expect("measurement");
        let correction = loose_coupling_correction(
            filter.state(),
            &measurement,
            lever,
            filter.last_body_rate_wrt_ecef_rps(),
        )
        .expect("correction");
        for axis in 0..3 {
            assert_close(correction.innovation[3 + axis], 0.0, 0.0);
        }
    }

    #[test]
    fn loose_update_rejects_failed_or_short_gnss_fix() {
        let measurement = GnssFixMeasurement {
            t_j2000_s: 0.0,
            position_ecef_m: [WGS84_A_M, 0.0, 0.0],
            velocity_ecef_mps: None,
            covariance: covariance_from_diag(&[1.0, 1.0, 1.0]),
            satellites_used: 3,
            solution_valid: true,
            fix_status: GnssFixStatus::Single,
        };
        assert!(matches!(
            measurement.validate(),
            Err(FusionError::InvalidInput {
                field: "satellites_used",
                reason: "at least 4 satellites required"
            })
        ));

        let failed = GnssFixMeasurement {
            satellites_used: 6,
            solution_valid: false,
            ..measurement
        };
        assert!(matches!(
            failed.validate(),
            Err(FusionError::InvalidInput {
                field: "solution_valid",
                reason: "GNSS fix must be successful"
            })
        ));
    }

    #[test]
    fn synthetic_static_truth_recovers_within_three_sigma_and_biases_converge() {
        let dt_s = 1.0;
        let steps = 20usize;
        let lever = [1.0, 0.5, -0.25];
        let accel_bias = [0.0015, -0.0010, 0.0020];
        let gyro_bias = [0.000009765625, -0.000009765625, 0.00001953125];
        let mut truth =
            NavState::new(0.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity()).expect("truth");
        let nominal = NavState::new(
            0.0,
            [WGS84_A_M + 2.0, -1.0, 0.5],
            [0.3, -0.2, 0.1],
            mat3_identity(),
        )
        .expect("nominal");
        let mut diagonal = vec![0.0; ERROR_STATE_DIMENSION_15];
        for axis in 0..3 {
            diagonal[ERROR_POSITION_INDEX + axis] = 25.0;
            diagonal[ERROR_VELOCITY_INDEX + axis] = 1.0;
            diagonal[ERROR_ATTITUDE_INDEX + axis] = 0.05 * 0.05;
            diagonal[ERROR_ACCEL_BIAS_INDEX + axis] = 0.05 * 0.05;
            diagonal[ERROR_GYRO_BIAS_INDEX + axis] = 0.003 * 0.003;
        }
        let state = reference_filter_state(nominal, &diagonal).expect("filter state");
        let spec = ImuSpec::datasheet(0.02, 0.001, 0.004, 2.0e-4, 300.0, 300.0, None, None);
        let mut config = InertialFilterConfig::new(spec).expect("config");
        config.loose.lever_arm_body_m = lever;
        let mut filter = InertialFilter::with_config(state, config).expect("filter");
        let mut rng = SplitMix64::new(0x4c4f_4f53_455f_0001);
        let position_sigma_m = 0.20;
        let velocity_sigma_mps = 0.030;
        let covariance = covariance_from_diag(&[
            position_sigma_m * position_sigma_m,
            position_sigma_m * position_sigma_m,
            position_sigma_m * position_sigma_m,
            velocity_sigma_mps * velocity_sigma_mps,
            velocity_sigma_mps * velocity_sigma_mps,
            velocity_sigma_mps * velocity_sigma_mps,
        ]);

        for step in 1..=steps {
            let (truth_next, sample, true_body_rate_wrt_ecef) =
                inverted_static_sample(&truth, step as f64 * dt_s, dt_s, accel_bias, gyro_bias);
            truth = truth_next;
            filter.propagate(sample).expect("propagate");
            let antenna_position = add3(
                truth.position_ecef_m,
                mul_vec3(&truth.attitude_body_to_ecef, lever),
            );
            let antenna_velocity = add3(
                truth.velocity_ecef_mps,
                mul_vec3(
                    &truth.attitude_body_to_ecef,
                    cross3(true_body_rate_wrt_ecef, lever),
                ),
            );
            let measurement = GnssFixMeasurement::position_velocity(
                truth.t_j2000_s,
                add_noise3(antenna_position, position_sigma_m, &mut rng),
                add_noise3(antenna_velocity, velocity_sigma_mps, &mut rng),
                covariance.clone(),
                8,
            )
            .expect("measurement");
            let update = filter.update_loose(&measurement).expect("loose update");
            assert!(update.applied);
            assert_eq!(
                update.nis.to_bits(),
                update.ekf.normalized_innovation_squared.to_bits()
            );
        }

        let state = filter.state();
        for (axis, expected_accel_bias) in accel_bias.iter().enumerate() {
            let position_error = state.nominal.position_ecef_m[axis] - truth.position_ecef_m[axis];
            let velocity_error =
                state.nominal.velocity_ecef_mps[axis] - truth.velocity_ecef_mps[axis];
            let position_bound = 3.0
                * state.covariance[ERROR_POSITION_INDEX + axis][ERROR_POSITION_INDEX + axis].sqrt();
            assert!(
                position_error.abs() <= position_bound,
                "position axis {axis} error {position_error:.17e}, bound {position_bound:.17e}"
            );
            assert!(
                velocity_error.abs()
                    <= 3.0
                        * state.covariance[ERROR_VELOCITY_INDEX + axis]
                            [ERROR_VELOCITY_INDEX + axis]
                            .sqrt(),
                "velocity axis {axis} error {velocity_error:.17e}"
            );
            let accel_bias_error = state.nominal.accel_bias_mps2[axis] - *expected_accel_bias;
            let accel_bias_bound = 3.0
                * state.covariance[ERROR_ACCEL_BIAS_INDEX + axis][ERROR_ACCEL_BIAS_INDEX + axis]
                    .sqrt();
            assert!(
                accel_bias_error.abs() <= accel_bias_bound,
                "accelerometer bias axis {axis} error {accel_bias_error:.17e}, bound {accel_bias_bound:.17e}"
            );
        }
    }

    #[test]
    fn lever_velocity_update_converges_observable_gyro_bias_components() {
        let dt_s = 0.1;
        let lever = [1.0, 0.5, -0.25];
        let gyro_bias = [0.0009765625, -0.0009765625, 0.001953125];
        let truth =
            NavState::new(0.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity()).expect("truth");
        let mut diagonal = vec![0.0; ERROR_STATE_DIMENSION_15];
        for axis in 0..3 {
            diagonal[ERROR_POSITION_INDEX + axis] = 1.0;
            diagonal[ERROR_VELOCITY_INDEX + axis] = 1.0e-10;
            diagonal[ERROR_ATTITUDE_INDEX + axis] = 1.0e-10;
            diagonal[ERROR_ACCEL_BIAS_INDEX + axis] = 1.0e-10;
            diagonal[ERROR_GYRO_BIAS_INDEX + axis] = 1.0e-4;
        }
        let state = reference_filter_state(truth, &diagonal).expect("filter state");
        let spec = ImuSpec::datasheet(
            0.0,
            0.0,
            0.0,
            0.0,
            crate::inertial::config::RANDOM_WALK_BIAS_TAU_S,
            crate::inertial::config::RANDOM_WALK_BIAS_TAU_S,
            None,
            None,
        );
        let mut config = InertialFilterConfig::new(spec).expect("config");
        config.loose.lever_arm_body_m = lever;
        let mut filter = InertialFilter::with_config(state, config).expect("filter");
        let (truth_next, sample, true_body_rate_wrt_ecef) =
            inverted_static_sample(&truth, dt_s, dt_s, [0.0; 3], gyro_bias);
        filter.propagate(sample).expect("propagate");

        let antenna_position = add3(
            truth_next.position_ecef_m,
            mul_vec3(&truth_next.attitude_body_to_ecef, lever),
        );
        let antenna_velocity = add3(
            truth_next.velocity_ecef_mps,
            mul_vec3(
                &truth_next.attitude_body_to_ecef,
                cross3(true_body_rate_wrt_ecef, lever),
            ),
        );
        let measurement = GnssFixMeasurement::position_velocity(
            truth_next.t_j2000_s,
            antenna_position,
            antenna_velocity,
            covariance_from_diag(&[1.0e6, 1.0e6, 1.0e6, 1.0e-8, 1.0e-8, 1.0e-8]),
            8,
        )
        .expect("measurement");
        let update = filter.update_loose(&measurement).expect("loose update");
        assert!(update.applied);

        let state = filter.state();
        for (axis, expected_gyro_bias) in gyro_bias.iter().enumerate() {
            let error = state.nominal.gyro_bias_rps[axis] - *expected_gyro_bias;
            let bound = 3.0
                * state.covariance[ERROR_GYRO_BIAS_INDEX + axis][ERROR_GYRO_BIAS_INDEX + axis]
                    .sqrt();
            assert!(
                error.abs() <= bound,
                "gyroscope bias axis {axis} error {error:.17e}, bound {bound:.17e}"
            );
        }
    }

    #[test]
    fn loose_nees_and_nis_land_inside_bar_shalom_chi_square_bands() {
        let trials = 40usize;
        let alpha = 0.05;
        let p_diag: [f64; 6] = [9.0, 4.0, 16.0, 0.25, 0.36, 0.49];
        let r_diag: [f64; 6] = [1.0, 1.44, 0.64, 0.04, 0.09, 0.16];
        let truth =
            NavState::new(20.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity()).expect("truth");
        let mut rng = SplitMix64::new(0x4241_5253_4841_4c4f);
        let mut nees_sum = 0.0;
        let mut nis_sum = 0.0;

        for _ in 0..trials {
            let mut initial_error = [0.0; 6];
            let mut measurement_noise = [0.0; 6];
            for idx in 0..6 {
                initial_error[idx] = p_diag[idx].sqrt() * rng.standard_normal();
                measurement_noise[idx] = r_diag[idx].sqrt() * rng.standard_normal();
            }
            let nominal = NavState::new(
                20.0,
                [
                    truth.position_ecef_m[0] + initial_error[0],
                    truth.position_ecef_m[1] + initial_error[1],
                    truth.position_ecef_m[2] + initial_error[2],
                ],
                [
                    truth.velocity_ecef_mps[0] + initial_error[3],
                    truth.velocity_ecef_mps[1] + initial_error[4],
                    truth.velocity_ecef_mps[2] + initial_error[5],
                ],
                mat3_identity(),
            )
            .expect("nominal");
            let mut diagonal = vec![0.0; ERROR_STATE_DIMENSION_15];
            diagonal[..6].copy_from_slice(&p_diag);
            for value in diagonal.iter_mut().take(ERROR_STATE_DIMENSION_15).skip(6) {
                *value = 1.0;
            }
            let state = reference_filter_state(nominal, &diagonal).expect("filter state");
            let spec = ImuSpec::datasheet(
                0.0,
                0.0,
                0.0,
                0.0,
                crate::inertial::config::RANDOM_WALK_BIAS_TAU_S,
                crate::inertial::config::RANDOM_WALK_BIAS_TAU_S,
                None,
                None,
            );
            let mut filter = InertialFilter::new(state, spec).expect("filter");
            let measurement = GnssFixMeasurement::position_velocity(
                20.0,
                [
                    truth.position_ecef_m[0] + measurement_noise[0],
                    truth.position_ecef_m[1] + measurement_noise[1],
                    truth.position_ecef_m[2] + measurement_noise[2],
                ],
                [
                    truth.velocity_ecef_mps[0] + measurement_noise[3],
                    truth.velocity_ecef_mps[1] + measurement_noise[4],
                    truth.velocity_ecef_mps[2] + measurement_noise[5],
                ],
                covariance_from_diag(&r_diag),
                8,
            )
            .expect("measurement");
            let expected_nis = (0..6)
                .map(|idx| {
                    let innovation = measurement_noise[idx] - initial_error[idx];
                    innovation * innovation / (p_diag[idx] + r_diag[idx])
                })
                .sum::<f64>();
            let update = filter.update_loose(&measurement).expect("loose update");
            assert_close(update.nis, expected_nis, 1.0e-9);
            nis_sum += update.nis;

            let updated = filter.state();
            for idx in 0..6 {
                let expected_variance = p_diag[idx] * r_diag[idx] / (p_diag[idx] + r_diag[idx]);
                assert_close(updated.covariance[idx][idx], expected_variance, 5.0e-15);
            }
            let dx = [
                updated.nominal.position_ecef_m[0] - truth.position_ecef_m[0],
                updated.nominal.position_ecef_m[1] - truth.position_ecef_m[1],
                updated.nominal.position_ecef_m[2] - truth.position_ecef_m[2],
                updated.nominal.velocity_ecef_mps[0] - truth.velocity_ecef_mps[0],
                updated.nominal.velocity_ecef_mps[1] - truth.velocity_ecef_mps[1],
                updated.nominal.velocity_ecef_mps[2] - truth.velocity_ecef_mps[2],
            ];
            nees_sum += quadratic_form(&updated.covariance, &dx, 6);
        }

        let nis_average = nis_sum / trials as f64;
        let nees_average = nees_sum / trials as f64;
        let dof = trials * 6;
        let lower = crate::quality::chi2_inv(alpha * 0.5, dof).expect("lower") / trials as f64;
        let upper =
            crate::quality::chi2_inv(1.0 - alpha * 0.5, dof).expect("upper") / trials as f64;
        assert!(
            (lower..=upper).contains(&nis_average),
            "NIS average {nis_average:.17e}, band [{lower:.17e}, {upper:.17e}]"
        );
        assert!(
            (lower..=upper).contains(&nees_average),
            "NEES average {nees_average:.17e}, band [{lower:.17e}, {upper:.17e}]"
        );
    }

    #[test]
    fn igg_iii_noop_region_matches_plain_l2_to_bits() {
        let measurement = direct_position_velocity_measurement(
            30.0,
            [WGS84_A_M + 0.25, -0.125, 0.0625],
            [0.03125, -0.015625, 0.0078125],
            1.0,
        );
        let mut plain = direct_update_filter(30.0, LooseCouplingConfig::default());
        let mut robust = direct_update_filter(
            30.0,
            LooseCouplingConfig {
                measurement_reweighting: Some(IggIiiMeasurementReweighting::standard()),
                ..LooseCouplingConfig::default()
            },
        );

        let plain_update = plain.update_loose(&measurement).expect("plain update");
        let robust_update = robust.update_loose(&measurement).expect("robust update");

        assert_eq!(plain_update, robust_update);
        assert_eq!(plain.state(), robust.state());
    }

    #[test]
    fn igg_iii_single_outlier_stays_within_tenth_sigma_of_clean_run() {
        // With P = R = 1 and gamma = 1e4, a 50 m rejection-row outlier moves
        // the robust scalar posterior by 50 / 10001 = 0.0071 clean posterior
        // sigma. The assertion uses 0.1 sigma to leave numerical margin.
        const X_SIGMA: f64 = 0.1;
        let clean_measurement =
            direct_position_velocity_measurement(40.0, [WGS84_A_M, 0.0, 0.0], [0.0; 3], 1.0);
        let outlier_measurement =
            direct_position_velocity_measurement(40.0, [WGS84_A_M + 50.0, 0.0, 0.0], [0.0; 3], 1.0);
        let robust_config = LooseCouplingConfig {
            measurement_reweighting: Some(IggIiiMeasurementReweighting::standard()),
            ..LooseCouplingConfig::default()
        };
        let mut clean = direct_update_filter(40.0, LooseCouplingConfig::default());
        let mut plain = direct_update_filter(40.0, LooseCouplingConfig::default());
        let mut robust = direct_update_filter(40.0, robust_config);

        clean
            .update_loose(&clean_measurement)
            .expect("clean update");
        plain
            .update_loose(&outlier_measurement)
            .expect("plain update");
        robust
            .update_loose(&outlier_measurement)
            .expect("robust update");

        let clean_x = clean.state().nominal.position_ecef_m[0];
        let clean_sigma =
            clean.state().covariance[ERROR_POSITION_INDEX][ERROR_POSITION_INDEX].sqrt();
        let robust_error = (robust.state().nominal.position_ecef_m[0] - clean_x).abs();
        let plain_error = (plain.state().nominal.position_ecef_m[0] - clean_x).abs();
        assert!(
            robust_error <= X_SIGMA * clean_sigma,
            "robust error {robust_error:.17e}, bound {:.17e}",
            X_SIGMA * clean_sigma
        );
        assert!(
            plain_error > X_SIGMA * clean_sigma,
            "plain error {plain_error:.17e}, bound {:.17e}",
            X_SIGMA * clean_sigma
        );
    }

    #[test]
    fn yang_prediction_adaptation_inflates_covariance_when_gate_passes() {
        let measurement =
            direct_position_velocity_measurement(50.0, [WGS84_A_M + 5.0, 0.0, 0.0], [0.0; 3], 1.0);
        let mut plain = direct_update_filter(50.0, LooseCouplingConfig::default());
        let mut adaptive = direct_update_filter(
            50.0,
            LooseCouplingConfig {
                prediction_adaptation: Some(YangPredictionAdaptiveFactor {
                    threshold: 0.1,
                    outlier_gate_probability: 0.99,
                }),
                ..LooseCouplingConfig::default()
            },
        );

        plain.update_loose(&measurement).expect("plain update");
        adaptive
            .update_loose(&measurement)
            .expect("adaptive update");

        let plain_error = (plain.state().nominal.position_ecef_m[0] - (WGS84_A_M + 5.0)).abs();
        let adaptive_error =
            (adaptive.state().nominal.position_ecef_m[0] - (WGS84_A_M + 5.0)).abs();
        assert!(
            adaptive_error < plain_error,
            "adaptive error {adaptive_error:.17e}, plain error {plain_error:.17e}"
        );
    }

    #[test]
    fn yang_prediction_adaptation_is_disabled_by_mahalanobis_outlier_gate() {
        let measurement =
            direct_position_velocity_measurement(60.0, [WGS84_A_M + 50.0, 0.0, 0.0], [0.0; 3], 1.0);
        let robust_only = LooseCouplingConfig {
            measurement_reweighting: Some(IggIiiMeasurementReweighting::standard()),
            ..LooseCouplingConfig::default()
        };
        let robust_and_adaptive = LooseCouplingConfig {
            prediction_adaptation: Some(YangPredictionAdaptiveFactor {
                threshold: 0.1,
                outlier_gate_probability: 0.99,
            }),
            ..robust_only
        };
        let mut robust = direct_update_filter(60.0, robust_only);
        let mut guarded = direct_update_filter(60.0, robust_and_adaptive);

        let robust_update = robust.update_loose(&measurement).expect("robust update");
        let guarded_update = guarded.update_loose(&measurement).expect("guarded update");

        assert_eq!(robust_update, guarded_update);
        assert_eq!(robust.state(), guarded.state());
    }

    #[test]
    fn stationary_pseudo_update_ignores_gnss_robust_options() {
        let stationary_updates = StationaryUpdateConfig {
            detector: StationaryDetectorConfig {
                window_len: 1,
                max_specific_force_norm_error_mps2: 1.0,
                max_body_rate_wrt_ecef_norm_rps: 1.0,
            },
            zero_velocity_sigma_mps: 1.0,
            zero_angular_rate_sigma_rps: 1.0,
        };
        let plain_config = LooseCouplingConfig {
            stationary_updates: Some(stationary_updates),
            ..LooseCouplingConfig::default()
        };
        let robust_config = LooseCouplingConfig {
            measurement_reweighting: Some(IggIiiMeasurementReweighting::standard()),
            prediction_adaptation: Some(YangPredictionAdaptiveFactor {
                threshold: 0.1,
                outlier_gate_probability: 0.99,
            }),
            stationary_updates: Some(stationary_updates),
            ..LooseCouplingConfig::default()
        };
        let mut plain = direct_update_filter(70.0, plain_config);
        let mut robust = direct_update_filter(70.0, robust_config);
        for filter in [&mut plain, &mut robust] {
            filter.state.nominal.velocity_ecef_mps = [50.0, 0.0, 0.0];
            filter
                .stationarity_window
                .push_back(StationarityDetectorSnapshotSample {
                    specific_force_norm_error_mps2: 0.0,
                    body_rate_wrt_ecef_norm_rps: 0.0,
                });
        }

        let plain_update = plain
            .update_stationary()
            .expect("plain stationary update")
            .expect("plain stationary update applied");
        let robust_update = robust
            .update_stationary()
            .expect("robust stationary update")
            .expect("robust stationary update applied");

        assert_eq!(plain_update, robust_update);
        assert_eq!(plain.state(), robust.state());
    }

    fn inverted_static_sample(
        state: &NavState,
        t_j2000_s: f64,
        dt_s: f64,
        accel_bias_mps2: [f64; 3],
        gyro_bias_rps: [f64; 3],
    ) -> (NavState, ImuSample, [f64; 3]) {
        let true_delta_theta_rad = [0.0, 0.0, OMEGA_E_DOT_RAD_S * dt_s];
        let true_delta_velocity_mps =
            inverse_delta_velocity(state, [0.0; 3], true_delta_theta_rad, dt_s);
        let increment = CorrectedImuIncrement {
            t_j2000_s,
            delta_velocity_mps: true_delta_velocity_mps,
            delta_theta_rad: true_delta_theta_rad,
            dt_s,
        };
        let truth_next =
            mechanize_ecef(state, &increment, MechanizationConfig::default()).expect("truth step");
        let sample = ImuSample::increment(
            t_j2000_s,
            add3(true_delta_velocity_mps, scale3(accel_bias_mps2, dt_s)),
            add3(true_delta_theta_rad, scale3(gyro_bias_rps, dt_s)),
            dt_s,
        );
        let true_body_rate_wrt_ecef = body_rate_relative_to_ecef(
            &truth_next.attitude_body_to_ecef,
            scale3(true_delta_theta_rad, 1.0 / dt_s),
        );
        (truth_next, sample, true_body_rate_wrt_ecef)
    }

    fn inverse_delta_velocity(
        state: &NavState,
        target_velocity_ecef_mps: [f64; 3],
        delta_theta_rad: [f64; 3],
        dt_s: f64,
    ) -> [f64; 3] {
        let c_avg = mid_interval_dcm(&state.attitude_body_to_ecef, delta_theta_rad, dt_s);
        let c_avg_t = inline_tr(&c_avg);
        let gravity = gravity_ecef_mps2(state.position_ecef_m).expect("gravity");
        let required_ecef = sub3(
            sub3(target_velocity_ecef_mps, state.velocity_ecef_mps),
            scale3(gravity, dt_s),
        );
        mat3_mul_vec(&c_avg_t, required_ecef)
    }

    fn mid_interval_dcm(
        attitude_body_to_ecef: &Mat3,
        delta_theta_rad: [f64; 3],
        dt_s: f64,
    ) -> Mat3 {
        let earth_half = earth_rotation_first_order(0.5 * dt_s);
        let body_half =
            crate::inertial::rodrigues_delta_dcm(scale3(delta_theta_rad, 0.5)).expect("body half");
        reorthonormalize_dcm(&mat3_mul(
            &mat3_mul(&earth_half, attitude_body_to_ecef),
            &body_half,
        ))
        .expect("mid dcm")
    }

    fn earth_rotation_first_order(dt_s: f64) -> Mat3 {
        [
            [1.0, OMEGA_E_DOT_RAD_S * dt_s, 0.0],
            [-OMEGA_E_DOT_RAD_S * dt_s, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    }

    fn add_noise3(value: [f64; 3], sigma: f64, rng: &mut SplitMix64) -> [f64; 3] {
        [
            value[0] + sigma * rng.symmetric_unit(),
            value[1] + sigma * rng.symmetric_unit(),
            value[2] + sigma * rng.symmetric_unit(),
        ]
    }

    fn quadratic_form(covariance: &[Vec<f64>], dx: &[f64], dimension: usize) -> f64 {
        let mut data = Vec::with_capacity(dimension * dimension);
        for row in covariance.iter().take(dimension) {
            data.extend(row.iter().take(dimension));
        }
        let matrix = DMatrix::from_row_slice(dimension, dimension, &data);
        let vector = DVector::from_column_slice(dx);
        let solved = matrix.cholesky().expect("covariance SPD").solve(&vector);
        dot_slice(dx, solved.as_slice())
    }

    fn dot_slice(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    fn direct_update_filter(t_j2000_s: f64, loose: LooseCouplingConfig) -> InertialFilter {
        let nominal = NavState::new(t_j2000_s, [WGS84_A_M, 0.0, 0.0], [0.0; 3], mat3_identity())
            .expect("nominal");
        let mut diagonal = vec![1.0; ERROR_STATE_DIMENSION_15];
        for value in diagonal.iter_mut().take(6) {
            *value = 1.0;
        }
        let state = reference_filter_state(nominal, &diagonal).expect("filter state");
        let spec = ImuSpec::datasheet(
            0.0,
            0.0,
            0.0,
            0.0,
            crate::inertial::config::RANDOM_WALK_BIAS_TAU_S,
            crate::inertial::config::RANDOM_WALK_BIAS_TAU_S,
            None,
            None,
        );
        let mut config = InertialFilterConfig::new(spec).expect("config");
        config.loose = loose;
        InertialFilter::with_config(state, config).expect("filter")
    }

    fn direct_position_velocity_measurement(
        t_j2000_s: f64,
        position_ecef_m: [f64; 3],
        velocity_ecef_mps: [f64; 3],
        sigma: f64,
    ) -> GnssFixMeasurement {
        GnssFixMeasurement::position_velocity(
            t_j2000_s,
            position_ecef_m,
            velocity_ecef_mps,
            covariance_from_diag(&[sigma * sigma; 6]),
            8,
        )
        .expect("measurement")
    }

    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn unit_f64(&mut self) -> f64 {
            let bits = 0x3ff0_0000_0000_0000 | (self.next_u64() >> 12);
            f64::from_bits(bits) - 1.0
        }

        fn symmetric_unit(&mut self) -> f64 {
            2.0 * self.unit_f64() - 1.0
        }

        fn standard_normal(&mut self) -> f64 {
            let u1 = self.unit_f64().max(f64::MIN_POSITIVE);
            let u2 = self.unit_f64();
            (-2.0 * libm::log(u1)).sqrt() * libm::cos(2.0 * core::f64::consts::PI * u2)
        }
    }

    #[test]
    fn splitmix_sequence_matches_covariance_fixture_pattern_bits() {
        let mut rng = SplitMix64::new(0x9876_5432_10fe_dcba);
        assert_eq!(rng.next_u64(), 0xaf45_24ce_f491_bb91);
        assert_eq!(rng.next_u64(), 0x25fc_5376_94a6_001c);
        let mut rng = SplitMix64::new(0x9876_5432_10fe_dcba);
        assert_eq!(rng.unit_f64().to_bits(), 0x3fe5_e8a4_99de_9236);
    }

    #[test]
    fn gyro_bias_test_vector_is_observable_for_non_axis_lever() {
        let lever = [1.0, 0.5, -0.25];
        let gyro_bias = [0.000009765625, -0.000009765625, 0.00001953125];
        assert_eq!(dot3(lever, gyro_bias).to_bits(), 0.0_f64.to_bits());
        assert!(norm3(cross3(gyro_bias, lever)) > 0.0);
    }
}
