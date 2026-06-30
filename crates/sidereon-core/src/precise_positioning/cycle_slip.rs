//! Dual-frequency cycle-slip detection for PPP preprocessing.
//!
//! ```
//! use sidereon_core::carrier_phase::SlipReason;
//! use sidereon_core::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
//! use sidereon_core::precise_positioning::{
//!     detect_cycle_slips, CycleSlipConfig, DualFrequencyEpoch, DualFrequencyObservation,
//! };
//!
//! fn observation(epoch: usize, wide_lane_cycles: f64) -> DualFrequencyObservation {
//!     let geometric_m = 23_000_000.0 + epoch as f64 * 100.0;
//!     let lambda1 = C_M_S / F_L1_HZ;
//!     let lambda2 = C_M_S / F_L2_HZ;
//!     let lambda_wl = C_M_S / (F_L1_HZ - F_L2_HZ);
//!     let l2_m = geometric_m + lambda_wl * wide_lane_cycles;
//!     let l1_m = l2_m;
//!     DualFrequencyObservation {
//!         satellite_id: "G01".to_string(),
//!         ambiguity_id: "G01".to_string(),
//!         p1_m: geometric_m,
//!         p2_m: geometric_m,
//!         phi1_cyc: l1_m / lambda1,
//!         phi2_cyc: l2_m / lambda2,
//!         f1_hz: F_L1_HZ,
//!         f2_hz: F_L2_HZ,
//!         lli1: None,
//!         lli2: None,
//!     }
//! }
//!
//! let epochs = (0..4)
//!     .map(|epoch| DualFrequencyEpoch {
//!         gap_time_s: Some(epoch as f64 * 30.0),
//!         observations: vec![observation(epoch, if epoch >= 2 { 9.0 } else { 5.0 })],
//!     })
//!     .collect::<Vec<_>>();
//! let config = CycleSlipConfig {
//!     melbourne_wubbena_threshold_cycles: 2.0,
//!     geometry_free_threshold_m: 0.05,
//!     maximum_gap_s: 120.0,
//!     ..CycleSlipConfig::default()
//! };
//!
//! let flags = detect_cycle_slips(&epochs, config)?;
//! assert!(!flags[1].observations[0].slip);
//! assert_eq!(
//!     flags[2].observations[0].reasons,
//!     vec![SlipReason::MelbourneWubbena]
//! );
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use std::collections::BTreeMap;

use crate::carrier_phase::{
    CarrierPhaseError, SlipReason, DEFAULT_GF_THRESHOLD_M, DEFAULT_MIN_ARC_GAP_S,
    DEFAULT_MW_THRESHOLD_CYCLES,
};
use crate::frequencies::wavelength_for_frequency;
use crate::tolerances::FREQUENCY_DENOMINATOR_EPS_HZ;

use super::prep::DualFrequencyObservation;

/// Default minimum number of usable samples required for a stable arc.
pub const DEFAULT_MINIMUM_ARC_LENGTH: usize = 2;

/// Default sigma multiplier applied to running Melbourne-Wubbena statistics.
pub const DEFAULT_RUNNING_STATISTIC_K_FACTOR: f64 = 4.0;

/// Configuration for dual-frequency PPP cycle-slip detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CycleSlipConfig {
    /// Melbourne-Wubbena absolute slip threshold, in wide-lane cycles.
    pub melbourne_wubbena_threshold_cycles: f64,
    /// Geometry-free phase step threshold, in meters.
    pub geometry_free_threshold_m: f64,
    /// Minimum usable sample count for an initialized ambiguity arc.
    pub minimum_arc_length: usize,
    /// Sigma multiplier for running Melbourne-Wubbena outlier detection.
    pub running_statistic_k_factor: f64,
    /// Maximum time gap allowed inside one continuous ambiguity arc, in seconds.
    pub maximum_gap_s: f64,
}

impl CycleSlipConfig {
    /// Validate that all configured thresholds and sample counts are usable.
    pub fn validate(&self) -> Result<(), CycleSlipConfigError> {
        validate_positive_finite(
            self.melbourne_wubbena_threshold_cycles,
            CycleSlipConfigError::InvalidMelbourneWubbenaThreshold,
        )?;
        validate_positive_finite(
            self.geometry_free_threshold_m,
            CycleSlipConfigError::InvalidGeometryFreeThreshold,
        )?;
        validate_positive_finite(
            self.running_statistic_k_factor,
            CycleSlipConfigError::InvalidRunningStatisticKFactor,
        )?;
        validate_positive_finite(self.maximum_gap_s, CycleSlipConfigError::InvalidMaximumGap)?;
        if self.minimum_arc_length == 0 {
            return Err(CycleSlipConfigError::InvalidMinimumArcLength);
        }
        Ok(())
    }
}

impl Default for CycleSlipConfig {
    fn default() -> Self {
        Self {
            melbourne_wubbena_threshold_cycles: DEFAULT_MW_THRESHOLD_CYCLES,
            geometry_free_threshold_m: DEFAULT_GF_THRESHOLD_M,
            minimum_arc_length: DEFAULT_MINIMUM_ARC_LENGTH,
            running_statistic_k_factor: DEFAULT_RUNNING_STATISTIC_K_FACTOR,
            maximum_gap_s: DEFAULT_MIN_ARC_GAP_S,
        }
    }
}

/// Validation error for [`CycleSlipConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleSlipConfigError {
    /// The Melbourne-Wubbena threshold was not positive and finite.
    InvalidMelbourneWubbenaThreshold,
    /// The geometry-free threshold was not positive and finite.
    InvalidGeometryFreeThreshold,
    /// The minimum arc length was zero.
    InvalidMinimumArcLength,
    /// The running-statistic sigma multiplier was not positive and finite.
    InvalidRunningStatisticKFactor,
    /// The maximum time gap was not positive and finite.
    InvalidMaximumGap,
}

impl core::fmt::Display for CycleSlipConfigError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidMelbourneWubbenaThreshold => {
                write!(f, "Melbourne-Wubbena threshold must be positive and finite")
            }
            Self::InvalidGeometryFreeThreshold => {
                write!(f, "geometry-free threshold must be positive and finite")
            }
            Self::InvalidMinimumArcLength => write!(f, "minimum arc length must be nonzero"),
            Self::InvalidRunningStatisticKFactor => {
                write!(f, "running-statistic k-factor must be positive and finite")
            }
            Self::InvalidMaximumGap => write!(f, "maximum gap must be positive and finite"),
        }
    }
}

impl std::error::Error for CycleSlipConfigError {}

/// Error produced while evaluating a cycle-slip detector sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CycleSlipError {
    /// Detector configuration failed validation.
    InvalidConfig(CycleSlipConfigError),
    /// One or more observation scalars were not finite.
    NonFiniteObservation,
    /// A supplied epoch time was not finite.
    NonFiniteEpochTime,
    /// Supplied epoch times were not ordered.
    EpochsNotOrdered,
    /// One or both carrier frequencies were not positive and finite.
    InvalidFrequency,
    /// The two carrier frequencies were too close for a wide-lane combination.
    EqualFrequencies,
}

impl core::fmt::Display for CycleSlipError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig(err) => write!(f, "invalid cycle-slip config: {err}"),
            Self::NonFiniteObservation => write!(f, "cycle-slip observation must be finite"),
            Self::NonFiniteEpochTime => write!(f, "cycle-slip epoch time must be finite"),
            Self::EpochsNotOrdered => write!(f, "cycle-slip epochs must be time ordered"),
            Self::InvalidFrequency => write!(f, "carrier frequency must be positive and finite"),
            Self::EqualFrequencies => write!(f, "carrier frequencies must be distinct"),
        }
    }
}

impl std::error::Error for CycleSlipError {}

impl From<CycleSlipConfigError> for CycleSlipError {
    fn from(err: CycleSlipConfigError) -> Self {
        Self::InvalidConfig(err)
    }
}

impl From<CarrierPhaseError> for CycleSlipError {
    fn from(err: CarrierPhaseError) -> Self {
        match err {
            CarrierPhaseError::EqualFrequencies => Self::EqualFrequencies,
            CarrierPhaseError::InvalidFrequency => Self::InvalidFrequency,
            CarrierPhaseError::InvalidObservation => Self::NonFiniteObservation,
            CarrierPhaseError::InvalidThreshold => {
                Self::InvalidConfig(CycleSlipConfigError::InvalidMelbourneWubbenaThreshold)
            }
        }
    }
}

/// Running mean and variance for a scalar observable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunningMeanVariance {
    /// Number of samples accumulated into the statistic.
    pub sample_count: usize,
    /// Running sample mean.
    pub mean: f64,
    /// Running sample variance.
    pub variance: f64,
}

impl RunningMeanVariance {
    /// Construct an empty running statistic.
    pub const fn new() -> Self {
        Self {
            sample_count: 0,
            mean: 0.0,
            variance: 0.0,
        }
    }

    /// Add one scalar sample using a numerically stable online update.
    pub fn push(&mut self, value: f64) {
        let previous_count = self.sample_count;
        self.sample_count += 1;

        if previous_count == 0 {
            self.mean = value;
            self.variance = 0.0;
            return;
        }

        let previous_mean = self.mean;
        let previous_m2 = self.variance * previous_count.saturating_sub(1) as f64;
        let delta = value - previous_mean;
        self.mean = previous_mean + delta / self.sample_count as f64;
        let delta_after_mean_update = value - self.mean;
        let m2 = previous_m2 + delta * delta_after_mean_update;
        self.variance = m2 / (self.sample_count - 1) as f64;
    }

    /// Return the sample standard deviation when at least two samples exist.
    pub fn standard_deviation(&self) -> Option<f64> {
        if self.sample_count >= 2 && self.variance.is_finite() && self.variance >= 0.0 {
            Some(self.variance.sqrt())
        } else {
            None
        }
    }
}

impl Default for RunningMeanVariance {
    fn default() -> Self {
        Self::new()
    }
}

/// Running cycle-slip detector state for one satellite/ambiguity arc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SatelliteCycleSlipState {
    /// Previous usable epoch time, in seconds, when supplied by the caller.
    pub previous_epoch_time_s: Option<f64>,
    /// Previous Melbourne-Wubbena combination, in wide-lane cycles.
    pub previous_melbourne_wubbena_cycles: Option<f64>,
    /// Running Melbourne-Wubbena mean and variance, in wide-lane cycles.
    pub melbourne_wubbena: RunningMeanVariance,
    /// Previous geometry-free phase combination, in meters.
    pub previous_geometry_free_m: Option<f64>,
}

impl SatelliteCycleSlipState {
    /// Construct an empty per-satellite/ambiguity detector state.
    pub const fn new() -> Self {
        Self {
            previous_epoch_time_s: None,
            previous_melbourne_wubbena_cycles: None,
            melbourne_wubbena: RunningMeanVariance::new(),
            previous_geometry_free_m: None,
        }
    }

    /// Clear arc-history fields after a gap or reacquisition.
    pub fn reset_arc(&mut self) {
        *self = Self::new();
    }
}

impl Default for SatelliteCycleSlipState {
    fn default() -> Self {
        Self::new()
    }
}

/// Detector-state map key `(satellite_id, ambiguity_id)`.
pub type CycleSlipStateKey = (String, String);

/// Stateful detector storage keyed by PPP satellite and ambiguity identifiers.
#[derive(Debug, Clone, PartialEq)]
pub struct CycleSlipDetectorState {
    /// Running state for each satellite/ambiguity arc currently tracked by the detector.
    pub satellites: BTreeMap<CycleSlipStateKey, SatelliteCycleSlipState>,
}

impl CycleSlipDetectorState {
    /// Construct an empty detector state.
    pub fn new() -> Self {
        Self {
            satellites: BTreeMap::new(),
        }
    }
}

impl Default for CycleSlipDetectorState {
    fn default() -> Self {
        Self::new()
    }
}

/// Result from updating one satellite's Melbourne-Wubbena detector state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MelbourneWubbenaUpdate {
    /// Melbourne-Wubbena combination, in wide-lane cycles.
    pub melbourne_wubbena_cycles: f64,
    /// True when the sample exceeds the configured Melbourne-Wubbena slip gate.
    pub slip: bool,
}

/// Result from updating one satellite's geometry-free detector state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometryFreeUpdate {
    /// Geometry-free phase combination, in meters.
    pub geometry_free_m: f64,
    /// True when the sample exceeds the configured geometry-free slip gate.
    pub slip: bool,
    /// True when a long gap reset the arc before this sample was accepted.
    pub reset: bool,
}

/// Cycle-slip flag for one satellite observation in an epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleSlipFlagObservation {
    /// Satellite identifier copied from the input observation.
    pub satellite_id: String,
    /// True when any cycle-slip detector flagged this satellite at the epoch.
    pub slip: bool,
    /// Slip reasons in deterministic detector order.
    pub reasons: Vec<SlipReason>,
}

/// Cycle-slip flags for one input dual-frequency epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct CycleSlipFlagEpoch {
    /// Comparable epoch coordinate copied from the input epoch.
    pub gap_time_s: Option<f64>,
    /// Per-satellite slip flags for observations in this epoch.
    pub observations: Vec<CycleSlipFlagObservation>,
}

/// Compute the Melbourne-Wubbena combination in wide-lane cycles.
pub fn melbourne_wubbena_cycles(
    observation: &DualFrequencyObservation,
) -> Result<f64, CycleSlipError> {
    validate_observation_finite(observation)?;
    let lambda_wl = wide_lane_wavelength_m(observation.f1_hz, observation.f2_hz)?;
    let narrow_lane_code_m = narrow_lane_code_m(observation)?;
    let wide_lane_phase_m = lambda_wl * (observation.phi1_cyc - observation.phi2_cyc);
    Ok((wide_lane_phase_m - narrow_lane_code_m) / lambda_wl)
}

/// Update one satellite state with a Melbourne-Wubbena sample and classify it.
pub fn update_melbourne_wubbena(
    state: &mut SatelliteCycleSlipState,
    observation: &DualFrequencyObservation,
    config: CycleSlipConfig,
) -> Result<MelbourneWubbenaUpdate, CycleSlipError> {
    config.validate()?;
    let mw_cycles = melbourne_wubbena_cycles(observation)?;
    let slip = melbourne_wubbena_slip(state, mw_cycles, config);

    state.previous_melbourne_wubbena_cycles = Some(mw_cycles);
    state.melbourne_wubbena.push(mw_cycles);

    Ok(MelbourneWubbenaUpdate {
        melbourne_wubbena_cycles: mw_cycles,
        slip,
    })
}

/// Compute the geometry-free phase combination `L1 - L2`, in meters.
pub fn geometry_free_m(observation: &DualFrequencyObservation) -> Result<f64, CycleSlipError> {
    validate_observation_finite(observation)?;
    let l1_m = phase_meters(observation.phi1_cyc, observation.f1_hz)?;
    let l2_m = phase_meters(observation.phi2_cyc, observation.f2_hz)?;
    Ok(l1_m - l2_m)
}

/// Update one satellite state with a geometry-free sample and classify it.
pub fn update_geometry_free(
    state: &mut SatelliteCycleSlipState,
    observation: &DualFrequencyObservation,
    epoch_time_s: Option<f64>,
    config: CycleSlipConfig,
) -> Result<GeometryFreeUpdate, CycleSlipError> {
    config.validate()?;
    validate_epoch_time(epoch_time_s)?;
    if epoch_time_goes_back(state.previous_epoch_time_s, epoch_time_s) {
        return Err(CycleSlipError::EpochsNotOrdered);
    }
    let gf_m = geometry_free_m(observation)?;
    let reset = gap_reset(
        state.previous_epoch_time_s,
        epoch_time_s,
        config.maximum_gap_s,
    );
    if reset {
        state.reset_arc();
    }
    let slip = !reset
        && state
            .previous_geometry_free_m
            .is_some_and(|prev| (gf_m - prev).abs() > config.geometry_free_threshold_m);

    remember_epoch_time(state, epoch_time_s);
    state.previous_geometry_free_m = Some(gf_m);

    Ok(GeometryFreeUpdate {
        geometry_free_m: gf_m,
        slip,
        reset,
    })
}

/// Detect cycle slips across an ordered sequence of dual-frequency PPP epochs.
pub fn detect_cycle_slips(
    epochs: &[super::prep::DualFrequencyEpoch],
    config: CycleSlipConfig,
) -> Result<Vec<CycleSlipFlagEpoch>, CycleSlipError> {
    config.validate()?;
    validate_epoch_order(epochs)?;
    let mut state = CycleSlipDetectorState::new();
    let mut out = Vec::with_capacity(epochs.len());

    for epoch in epochs {
        let observations = epoch
            .observations
            .iter()
            .map(|observation| {
                let ambiguity_state = state
                    .satellites
                    .entry(cycle_slip_state_key(observation))
                    .or_default();
                classify_dual_frequency_observation(
                    ambiguity_state,
                    observation,
                    epoch.gap_time_s,
                    config,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.push(CycleSlipFlagEpoch {
            gap_time_s: epoch.gap_time_s,
            observations,
        });
    }

    Ok(out)
}

fn cycle_slip_state_key(observation: &DualFrequencyObservation) -> CycleSlipStateKey {
    (
        observation.satellite_id.clone(),
        observation.ambiguity_id.clone(),
    )
}

fn validate_positive_finite(
    value: f64,
    error: CycleSlipConfigError,
) -> Result<(), CycleSlipConfigError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(error)
    }
}

fn validate_observation_finite(
    observation: &DualFrequencyObservation,
) -> Result<(), CycleSlipError> {
    if observation.p1_m.is_finite()
        && observation.p2_m.is_finite()
        && observation.phi1_cyc.is_finite()
        && observation.phi2_cyc.is_finite()
        && observation.f1_hz.is_finite()
        && observation.f2_hz.is_finite()
    {
        Ok(())
    } else {
        Err(CycleSlipError::NonFiniteObservation)
    }
}

fn validate_epoch_time(epoch_time_s: Option<f64>) -> Result<(), CycleSlipError> {
    if epoch_time_s.is_some_and(|time_s| !time_s.is_finite()) {
        Err(CycleSlipError::NonFiniteEpochTime)
    } else {
        Ok(())
    }
}

fn validate_epoch_order(epochs: &[super::prep::DualFrequencyEpoch]) -> Result<(), CycleSlipError> {
    let mut previous_time_s = None;
    for epoch in epochs {
        validate_epoch_time(epoch.gap_time_s)?;
        if let (Some(previous), Some(current)) = (previous_time_s, epoch.gap_time_s) {
            if current < previous {
                return Err(CycleSlipError::EpochsNotOrdered);
            }
        }
        if epoch.gap_time_s.is_some() {
            previous_time_s = epoch.gap_time_s;
        }
    }
    Ok(())
}

fn phase_meters(phi_cycles: f64, f_hz: f64) -> Result<f64, CycleSlipError> {
    validate_frequency(f_hz)?;
    let wavelength_m = wavelength_for_frequency(f_hz).ok_or(CycleSlipError::InvalidFrequency)?;
    let phase_m = wavelength_m * phi_cycles;
    if phase_m.is_finite() {
        Ok(phase_m)
    } else {
        Err(CycleSlipError::NonFiniteObservation)
    }
}

fn validate_frequency(f_hz: f64) -> Result<(), CycleSlipError> {
    if f_hz.is_finite() && f_hz > 0.0 {
        Ok(())
    } else {
        Err(CycleSlipError::InvalidFrequency)
    }
}

fn wide_lane_wavelength_m(f1_hz: f64, f2_hz: f64) -> Result<f64, CycleSlipError> {
    validate_frequency(f1_hz)?;
    validate_frequency(f2_hz)?;
    let lambda1 = wavelength_for_frequency(f1_hz).ok_or(CycleSlipError::InvalidFrequency)?;
    let lambda2 = wavelength_for_frequency(f2_hz).ok_or(CycleSlipError::InvalidFrequency)?;
    let inverse_wide_lane = 1.0 / lambda1 - 1.0 / lambda2;
    if inverse_wide_lane.abs() * crate::constants::C_M_S < FREQUENCY_DENOMINATOR_EPS_HZ {
        Err(CycleSlipError::EqualFrequencies)
    } else {
        Ok(1.0 / inverse_wide_lane)
    }
}

fn narrow_lane_code_m(observation: &DualFrequencyObservation) -> Result<f64, CycleSlipError> {
    let f1_hz = observation.f1_hz;
    let f2_hz = observation.f2_hz;
    validate_frequency(f1_hz)?;
    validate_frequency(f2_hz)?;
    if wavelength_for_frequency(f1_hz).is_none() || wavelength_for_frequency(f2_hz).is_none() {
        return Err(CycleSlipError::InvalidFrequency);
    }
    let denominator = f1_hz + f2_hz;
    if denominator.abs() < FREQUENCY_DENOMINATOR_EPS_HZ {
        Err(CycleSlipError::EqualFrequencies)
    } else {
        Ok((f1_hz * observation.p1_m + f2_hz * observation.p2_m) / denominator)
    }
}

fn melbourne_wubbena_slip(
    state: &SatelliteCycleSlipState,
    mw_cycles: f64,
    config: CycleSlipConfig,
) -> bool {
    let step_slip = state
        .previous_melbourne_wubbena_cycles
        .is_some_and(|prev| (mw_cycles - prev).abs() > config.melbourne_wubbena_threshold_cycles);
    let sigma_slip = state.melbourne_wubbena.sample_count >= config.minimum_arc_length
        && state
            .melbourne_wubbena
            .standard_deviation()
            .is_some_and(|sigma| {
                sigma > 0.0
                    && (mw_cycles - state.melbourne_wubbena.mean).abs()
                        > config.running_statistic_k_factor * sigma
            });
    step_slip || sigma_slip
}

fn epoch_time_goes_back(prev_time_s: Option<f64>, time_s: Option<f64>) -> bool {
    matches!(
        (prev_time_s, time_s),
        (Some(previous), Some(current)) if current < previous
    )
}

fn gap_reset(prev_time_s: Option<f64>, time_s: Option<f64>, maximum_gap_s: f64) -> bool {
    match (prev_time_s, time_s) {
        (Some(prev), Some(current)) => current - prev > maximum_gap_s,
        _ => false,
    }
}

fn remember_epoch_time(state: &mut SatelliteCycleSlipState, epoch_time_s: Option<f64>) {
    if let Some(time_s) = epoch_time_s {
        state.previous_epoch_time_s = Some(time_s);
    }
}

fn lli_slip(observation: &DualFrequencyObservation) -> bool {
    lli_bit0_set(observation.lli1) || lli_bit0_set(observation.lli2)
}

fn lli_bit0_set(lli: Option<i64>) -> bool {
    lli.is_some_and(|value| (value & 1) == 1)
}

fn classify_dual_frequency_observation(
    state: &mut SatelliteCycleSlipState,
    observation: &DualFrequencyObservation,
    epoch_time_s: Option<f64>,
    config: CycleSlipConfig,
) -> Result<CycleSlipFlagObservation, CycleSlipError> {
    let reset = gap_reset(
        state.previous_epoch_time_s,
        epoch_time_s,
        config.maximum_gap_s,
    );
    if reset {
        state.reset_arc();
    }

    let mw_cycles = melbourne_wubbena_cycles(observation)?;
    let gf_m = geometry_free_m(observation)?;
    let mut reasons = Vec::new();
    let lli = lli_slip(observation);
    if lli {
        reasons.push(SlipReason::Lli);
        state.reset_arc();
    }
    if !reset && !lli {
        if state
            .previous_geometry_free_m
            .is_some_and(|prev| (gf_m - prev).abs() > config.geometry_free_threshold_m)
        {
            reasons.push(SlipReason::GeometryFree);
        }
        if melbourne_wubbena_slip(state, mw_cycles, config) {
            reasons.push(SlipReason::MelbourneWubbena);
        }
    }

    remember_epoch_time(state, epoch_time_s);
    state.previous_geometry_free_m = Some(gf_m);
    state.previous_melbourne_wubbena_cycles = Some(mw_cycles);
    state.melbourne_wubbena.push(mw_cycles);

    Ok(CycleSlipFlagObservation {
        satellite_id: observation.satellite_id.clone(),
        slip: !reasons.is_empty(),
        reasons,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{C_M_S, F_L1_HZ, F_L2_HZ};

    #[test]
    fn cycle_slip_types_construct() {
        let config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 3.0,
            geometry_free_threshold_m: 0.04,
            minimum_arc_length: 3,
            running_statistic_k_factor: 5.0,
            maximum_gap_s: 120.0,
        };
        let satellite = SatelliteCycleSlipState {
            previous_epoch_time_s: Some(30.0),
            previous_melbourne_wubbena_cycles: Some(12.0),
            melbourne_wubbena: RunningMeanVariance {
                sample_count: 4,
                mean: 12.0,
                variance: 0.01,
            },
            previous_geometry_free_m: Some(0.2),
        };
        let mut detector = CycleSlipDetectorState::new();
        detector
            .satellites
            .insert(("G01".to_string(), "G01:L1L2".to_string()), satellite);

        assert_eq!(config.minimum_arc_length, 3);
        assert_eq!(
            detector
                .satellites
                .get(&("G01".to_string(), "G01:L1L2".to_string())),
            Some(&SatelliteCycleSlipState {
                previous_epoch_time_s: Some(30.0),
                previous_melbourne_wubbena_cycles: Some(12.0),
                melbourne_wubbena: RunningMeanVariance {
                    sample_count: 4,
                    mean: 12.0,
                    variance: 0.01,
                },
                previous_geometry_free_m: Some(0.2),
            })
        );
    }

    #[test]
    fn cycle_slip_default_config_is_well_formed() {
        let config = CycleSlipConfig::default();

        assert_eq!(
            config.melbourne_wubbena_threshold_cycles,
            DEFAULT_MW_THRESHOLD_CYCLES
        );
        assert_eq!(config.geometry_free_threshold_m, DEFAULT_GF_THRESHOLD_M);
        assert_eq!(config.maximum_gap_s, DEFAULT_MIN_ARC_GAP_S);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn cycle_slip_config_rejects_nonsensical_thresholds() {
        assert_eq!(
            CycleSlipConfig {
                melbourne_wubbena_threshold_cycles: 0.0,
                ..CycleSlipConfig::default()
            }
            .validate(),
            Err(CycleSlipConfigError::InvalidMelbourneWubbenaThreshold)
        );
        assert_eq!(
            CycleSlipConfig {
                geometry_free_threshold_m: f64::INFINITY,
                ..CycleSlipConfig::default()
            }
            .validate(),
            Err(CycleSlipConfigError::InvalidGeometryFreeThreshold)
        );
        assert_eq!(
            CycleSlipConfig {
                minimum_arc_length: 0,
                ..CycleSlipConfig::default()
            }
            .validate(),
            Err(CycleSlipConfigError::InvalidMinimumArcLength)
        );
        assert_eq!(
            CycleSlipConfig {
                running_statistic_k_factor: f64::NAN,
                ..CycleSlipConfig::default()
            }
            .validate(),
            Err(CycleSlipConfigError::InvalidRunningStatisticKFactor)
        );
        assert_eq!(
            CycleSlipConfig {
                maximum_gap_s: f64::NAN,
                ..CycleSlipConfig::default()
            }
            .validate(),
            Err(CycleSlipConfigError::InvalidMaximumGap)
        );
    }

    #[test]
    fn cycle_slip_clean_constant_mw_series_produces_no_slip() {
        let mut state = SatelliteCycleSlipState::new();
        let config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 2.0,
            minimum_arc_length: 2,
            running_statistic_k_factor: 3.0,
            ..CycleSlipConfig::default()
        };
        let updates = (0..5)
            .map(|epoch| {
                update_melbourne_wubbena(&mut state, &mw_observation("G01", epoch, 8.0), config)
                    .expect("MW update")
            })
            .collect::<Vec<_>>();

        assert!(updates.iter().all(|update| !update.slip));
        assert_eq!(state.melbourne_wubbena.sample_count, 5);
    }

    #[test]
    fn cycle_slip_injected_widelane_integer_jump_is_flagged() {
        let mut state = SatelliteCycleSlipState::new();
        let config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 2.0,
            minimum_arc_length: 2,
            running_statistic_k_factor: 3.0,
            ..CycleSlipConfig::default()
        };
        let slips = (0..5)
            .map(|epoch| {
                let wide_lane_cycles = if epoch >= 3 { 12.0 } else { 8.0 };
                update_melbourne_wubbena(
                    &mut state,
                    &mw_observation("G01", epoch, wide_lane_cycles),
                    config,
                )
                .expect("MW update")
                .slip
            })
            .collect::<Vec<_>>();

        assert_eq!(slips, vec![false, false, false, true, false]);
    }

    #[test]
    fn cycle_slip_smooth_geometry_free_variation_does_not_flag() {
        let mut state = SatelliteCycleSlipState::new();
        let config = CycleSlipConfig {
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };
        let slips = (0..5)
            .map(|epoch| {
                update_geometry_free(
                    &mut state,
                    &gf_observation("G01", epoch, epoch as f64 * 0.01),
                    Some(epoch as f64 * 30.0),
                    config,
                )
                .expect("GF update")
                .slip
            })
            .collect::<Vec<_>>();

        assert_eq!(slips, vec![false; 5]);
    }

    #[test]
    fn cycle_slip_geometry_free_phase_jump_on_one_frequency_flags() {
        let mut state = SatelliteCycleSlipState::new();
        let config = CycleSlipConfig {
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };
        let slips = (0..5)
            .map(|epoch| {
                let gf_m = if epoch >= 3 {
                    0.03 + 0.30
                } else {
                    epoch as f64 * 0.01
                };
                update_geometry_free(
                    &mut state,
                    &gf_observation("G01", epoch, gf_m),
                    Some(epoch as f64 * 30.0),
                    config,
                )
                .expect("GF update")
                .slip
            })
            .collect::<Vec<_>>();

        assert_eq!(slips, vec![false, false, false, true, false]);
    }

    #[test]
    fn cycle_slip_long_data_gap_resets_geometry_free_arc() {
        let mut state = SatelliteCycleSlipState::new();
        let config = CycleSlipConfig {
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };
        let samples = [
            (0.0, 0.00),
            (30.0, 0.01),
            (60.0, 0.02),
            (600.0, 0.50),
            (630.0, 0.51),
        ];
        let updates = samples
            .iter()
            .enumerate()
            .map(|(idx, (time_s, gf_m))| {
                update_geometry_free(
                    &mut state,
                    &gf_observation("G01", idx, *gf_m),
                    Some(*time_s),
                    config,
                )
                .expect("GF update")
            })
            .collect::<Vec<_>>();

        assert_eq!(
            updates.iter().map(|update| update.slip).collect::<Vec<_>>(),
            vec![false; 5]
        );
        assert_eq!(
            updates
                .iter()
                .map(|update| update.reset)
                .collect::<Vec<_>>(),
            vec![false, false, false, true, false]
        );
    }

    #[test]
    fn cycle_slip_geometry_free_rejects_backward_timestamp_without_resetting_arc() {
        let mut state = SatelliteCycleSlipState::new();
        let config = CycleSlipConfig {
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };

        update_geometry_free(
            &mut state,
            &gf_observation("G01", 0, 0.01),
            Some(30.0),
            config,
        )
        .expect("initial GF update");

        let err = update_geometry_free(
            &mut state,
            &gf_observation("G01", 1, 0.50),
            Some(0.0),
            config,
        )
        .expect_err("backward timestamp");

        assert_eq!(err, CycleSlipError::EpochsNotOrdered);
        assert_eq!(state.previous_epoch_time_s, Some(30.0));
        assert_close(state.previous_geometry_free_m.expect("GF state"), 0.01);
    }

    #[test]
    fn cycle_slip_geometry_free_forward_gap_still_resets_arc() {
        let mut state = SatelliteCycleSlipState::new();
        let config = CycleSlipConfig {
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };

        update_geometry_free(
            &mut state,
            &gf_observation("G01", 0, 0.01),
            Some(30.0),
            config,
        )
        .expect("initial GF update");
        let update = update_geometry_free(
            &mut state,
            &gf_observation("G01", 1, 0.50),
            Some(200.0),
            config,
        )
        .expect("forward gap");

        assert!(update.reset);
        assert!(!update.slip);
        assert_eq!(state.previous_epoch_time_s, Some(200.0));
        assert_close(state.previous_geometry_free_m.expect("GF state"), 0.50);
    }

    #[test]
    fn cycle_slip_driver_flags_exact_satellite_and_epoch() {
        let epochs = (0..5)
            .map(|epoch| super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(epoch as f64 * 30.0),
                observations: vec![
                    combined_observation("G01", epoch, 5.0, epoch as f64 * 0.01),
                    combined_observation(
                        "G02",
                        epoch,
                        if epoch >= 3 { 11.0 } else { 7.0 },
                        epoch as f64 * 0.01,
                    ),
                ],
            })
            .collect::<Vec<_>>();
        let config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 2.0,
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };

        let flags = detect_cycle_slips(&epochs, config).expect("cycle-slip driver");
        let slipped = slipped_epoch_satellites(&flags);

        assert_eq!(
            slipped,
            vec![(3, "G02".to_string(), vec![SlipReason::MelbourneWubbena])]
        );
    }

    #[test]
    fn cycle_slip_driver_keeps_same_satellite_ambiguities_independent() {
        let epochs = (0..5)
            .map(|epoch| super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(epoch as f64 * 30.0),
                observations: vec![
                    combined_observation_with_ambiguity(
                        "G01",
                        "G01:L1L2",
                        epoch,
                        if epoch >= 3 { 11.0 } else { 7.0 },
                        epoch as f64 * 0.01,
                    ),
                    combined_observation_with_ambiguity(
                        "G01",
                        "G01:L1L5",
                        epoch,
                        7.0,
                        epoch as f64 * 0.01,
                    ),
                ],
            })
            .collect::<Vec<_>>();
        let config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 2.0,
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };

        let flags = detect_cycle_slips(&epochs, config).expect("cycle-slip driver");
        let slipped = slipped_epoch_observations(&flags);

        assert_eq!(
            slipped,
            vec![(3, 0, "G01".to_string(), vec![SlipReason::MelbourneWubbena])]
        );
        assert_eq!(flags[3].observations[0].satellite_id, "G01");
        assert_eq!(flags[3].observations[1].satellite_id, "G01");
    }

    #[test]
    fn cycle_slip_driver_keeps_shared_ambiguity_different_satellites_independent() {
        let epochs = (0..=10)
            .map(|epoch| {
                let mut observations = Vec::new();
                if epoch == 0 {
                    observations.push(combined_observation_with_ambiguity(
                        "G01",
                        "shared-ambiguity",
                        epoch,
                        5.0,
                        0.0,
                    ));
                } else if epoch == 10 {
                    observations.push(combined_observation_with_ambiguity(
                        "G01",
                        "shared-ambiguity",
                        epoch,
                        12.0,
                        0.80,
                    ));
                }
                observations.push(combined_observation_with_ambiguity(
                    "G02",
                    "shared-ambiguity",
                    epoch,
                    5.0,
                    epoch as f64 * 0.005,
                ));
                super::super::prep::DualFrequencyEpoch {
                    gap_time_s: Some(epoch as f64 * 30.0),
                    observations,
                }
            })
            .collect::<Vec<_>>();
        let config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 2.0,
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };

        let flags = detect_cycle_slips(&epochs, config).expect("cycle-slip driver");

        assert!(slipped_epoch_observations(&flags).is_empty());
        assert_eq!(flags[10].observations[0].satellite_id, "G01");
        assert_eq!(flags[10].observations[1].satellite_id, "G02");
    }

    #[test]
    fn cycle_slip_driver_preserves_input_observation_order() {
        let epochs = vec![super::super::prep::DualFrequencyEpoch {
            gap_time_s: Some(0.0),
            observations: vec![
                combined_observation("G03", 0, 5.0, 0.0),
                combined_observation("G01", 0, 5.0, 0.0),
                combined_observation("G02", 0, 5.0, 0.0),
            ],
        }];

        let flags = detect_cycle_slips(&epochs, quiet_cycle_slip_config()).expect("cycle slips");
        let satellite_ids = flags[0]
            .observations
            .iter()
            .map(|observation| observation.satellite_id.as_str())
            .collect::<Vec<_>>();

        assert_eq!(satellite_ids, vec!["G03", "G01", "G02"]);
    }

    #[test]
    fn cycle_slip_driver_flags_lli_bit_without_combination_jump() {
        let mut epochs = constant_combination_epochs();
        epochs[1].observations[0].lli1 = Some(1);

        let flags = detect_cycle_slips(&epochs, quiet_cycle_slip_config()).expect("LLI flags");
        let slipped = slipped_epoch_satellites(&flags);

        assert_eq!(slipped, vec![(1, "G01".to_string(), vec![SlipReason::Lli])]);
    }

    #[test]
    fn cycle_slip_lli_epoch_restarts_arc_at_marked_epoch() {
        let mut state = SatelliteCycleSlipState::new();
        let config = quiet_cycle_slip_config();

        for epoch in 0..50 {
            let flag = classify_dual_frequency_observation(
                &mut state,
                &combined_observation("G01", epoch, 5.0, 0.0),
                Some(epoch as f64 * 30.0),
                config,
            )
            .expect("pre-slip epoch");
            assert!(!flag.slip);
        }

        let mut lli_epoch = combined_observation("G01", 50, 20.0, 0.50);
        lli_epoch.lli1 = Some(1);
        let lli_flag =
            classify_dual_frequency_observation(&mut state, &lli_epoch, Some(50.0 * 30.0), config)
                .expect("LLI epoch");

        assert_eq!(lli_flag.reasons, vec![SlipReason::Lli]);
        assert_eq!(state.previous_epoch_time_s, Some(50.0 * 30.0));
        assert_close(state.previous_geometry_free_m.expect("GF state"), 0.50);
        assert_close(
            state.previous_melbourne_wubbena_cycles.expect("MW state"),
            20.0,
        );
        assert_eq!(state.melbourne_wubbena.sample_count, 1);
        assert_close(state.melbourne_wubbena.mean, 20.0);
        assert_close(state.melbourne_wubbena.variance, 0.0);

        let clean_flag = classify_dual_frequency_observation(
            &mut state,
            &combined_observation("G01", 51, 20.05, 0.51),
            Some(51.0 * 30.0),
            config,
        )
        .expect("post-LLI epoch");

        assert!(!clean_flag.slip);
        assert_eq!(state.melbourne_wubbena.sample_count, 2);
    }

    #[test]
    fn cycle_slip_unmarked_epoch_keeps_existing_arc_history() {
        let mut state = SatelliteCycleSlipState::new();
        let config = quiet_cycle_slip_config();

        for epoch in 0..3 {
            let flag = classify_dual_frequency_observation(
                &mut state,
                &combined_observation("G01", epoch, 5.0 + epoch as f64 * 0.1, 0.0),
                Some(epoch as f64 * 30.0),
                config,
            )
            .expect("clean epoch");
            assert!(!flag.slip);
        }

        assert_eq!(state.previous_epoch_time_s, Some(60.0));
        assert_eq!(state.melbourne_wubbena.sample_count, 3);
        assert_close(state.melbourne_wubbena.mean, 5.1);
    }

    #[test]
    fn cycle_slip_driver_leaves_clean_lli_unflagged() {
        let mut epochs = constant_combination_epochs();
        epochs[1].observations[0].lli1 = Some(0);

        let flags = detect_cycle_slips(&epochs, quiet_cycle_slip_config()).expect("clean LLI");
        let slipped = slipped_epoch_satellites(&flags);

        assert!(slipped.is_empty());
    }

    #[test]
    fn cycle_slip_driver_initializes_new_and_reacquired_arcs_without_spurious_slips() {
        let epochs = vec![
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(0.0),
                observations: vec![combined_observation("G01", 0, 5.0, 0.00)],
            },
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(30.0),
                observations: vec![
                    combined_observation("G01", 1, 5.0, 0.01),
                    combined_observation("G02", 1, 20.0, 0.50),
                ],
            },
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(600.0),
                observations: vec![combined_observation("G01", 2, 20.0, 0.80)],
            },
        ];
        let config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 2.0,
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };

        let flags = detect_cycle_slips(&epochs, config).expect("cycle-slip driver");

        assert!(flags
            .iter()
            .flat_map(|epoch| &epoch.observations)
            .all(|observation| !observation.slip));
    }

    #[test]
    fn cycle_slip_driver_carries_time_across_untimed_samples() {
        let config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 2.0,
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        };
        let intermittent = vec![
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(0.0),
                observations: vec![combined_observation("G01", 0, 5.0, 0.00)],
            },
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: None,
                observations: vec![combined_observation("G01", 1, 5.2, 0.01)],
            },
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(200.0),
                observations: vec![combined_observation("G01", 2, 11.0, 0.50)],
            },
        ];
        let timed = vec![
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(0.0),
                observations: vec![combined_observation("G01", 0, 5.0, 0.00)],
            },
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(30.0),
                observations: vec![combined_observation("G01", 1, 5.2, 0.01)],
            },
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(60.0),
                observations: vec![combined_observation("G01", 2, 11.0, 0.50)],
            },
        ];

        let intermittent_flags =
            detect_cycle_slips(&intermittent, config).expect("intermittent timestamps");
        let timed_flags = detect_cycle_slips(&timed, config).expect("timed timestamps");

        assert!(slipped_epoch_satellites(&intermittent_flags).is_empty());
        assert_eq!(
            slipped_epoch_satellites(&timed_flags),
            vec![(
                2,
                "G01".to_string(),
                vec![SlipReason::GeometryFree, SlipReason::MelbourneWubbena]
            )]
        );
    }

    #[test]
    fn cycle_slip_driver_rejects_unordered_epoch_times() {
        let epochs = vec![
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(30.0),
                observations: Vec::new(),
            },
            super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(0.0),
                observations: Vec::new(),
            },
        ];

        assert_eq!(
            detect_cycle_slips(&epochs, CycleSlipConfig::default()),
            Err(CycleSlipError::EpochsNotOrdered)
        );
    }

    #[test]
    fn cycle_slip_geometry_free_rejects_nonfinite_phase_meter_output() {
        let mut observation = gf_observation("G01", 0, 0.25);
        observation.f1_hz = f64::MIN_POSITIVE;

        assert_eq!(
            geometry_free_m(&observation),
            Err(CycleSlipError::NonFiniteObservation)
        );
    }

    fn mw_observation(
        satellite_id: &str,
        epoch_index: usize,
        wide_lane_cycles: f64,
    ) -> DualFrequencyObservation {
        let geometric_m = 23_000_000.0 + epoch_index as f64 * 100.0;
        let n2 = 80_000.0;
        let n1 = n2 + wide_lane_cycles;
        let lambda1 = C_M_S / F_L1_HZ;
        let lambda2 = C_M_S / F_L2_HZ;

        DualFrequencyObservation {
            satellite_id: satellite_id.to_string(),
            ambiguity_id: satellite_id.to_string(),
            p1_m: geometric_m,
            p2_m: geometric_m,
            phi1_cyc: (geometric_m + n1 * lambda1) / lambda1,
            phi2_cyc: (geometric_m + n2 * lambda2) / lambda2,
            f1_hz: F_L1_HZ,
            f2_hz: F_L2_HZ,
            lli1: None,
            lli2: None,
        }
    }

    fn gf_observation(
        satellite_id: &str,
        epoch_index: usize,
        geometry_free_m: f64,
    ) -> DualFrequencyObservation {
        let geometric_m = 23_000_000.0 + epoch_index as f64 * 100.0;
        let lambda1 = C_M_S / F_L1_HZ;
        let lambda2 = C_M_S / F_L2_HZ;

        DualFrequencyObservation {
            satellite_id: satellite_id.to_string(),
            ambiguity_id: satellite_id.to_string(),
            p1_m: geometric_m,
            p2_m: geometric_m,
            phi1_cyc: geometric_m / lambda1,
            phi2_cyc: (geometric_m - geometry_free_m) / lambda2,
            f1_hz: F_L1_HZ,
            f2_hz: F_L2_HZ,
            lli1: None,
            lli2: None,
        }
    }

    fn combined_observation(
        satellite_id: &str,
        epoch_index: usize,
        melbourne_wubbena_cycles: f64,
        geometry_free_m: f64,
    ) -> DualFrequencyObservation {
        let geometric_m = 23_000_000.0 + epoch_index as f64 * 100.0;
        let lambda1 = C_M_S / F_L1_HZ;
        let lambda2 = C_M_S / F_L2_HZ;
        let lambda_wl = C_M_S / (F_L1_HZ - F_L2_HZ);
        let l2_m = geometric_m + lambda_wl * (melbourne_wubbena_cycles - geometry_free_m / lambda1);
        let l1_m = l2_m + geometry_free_m;

        DualFrequencyObservation {
            satellite_id: satellite_id.to_string(),
            ambiguity_id: satellite_id.to_string(),
            p1_m: geometric_m,
            p2_m: geometric_m,
            phi1_cyc: l1_m / lambda1,
            phi2_cyc: l2_m / lambda2,
            f1_hz: F_L1_HZ,
            f2_hz: F_L2_HZ,
            lli1: None,
            lli2: None,
        }
    }

    fn combined_observation_with_ambiguity(
        satellite_id: &str,
        ambiguity_id: &str,
        epoch_index: usize,
        melbourne_wubbena_cycles: f64,
        geometry_free_m: f64,
    ) -> DualFrequencyObservation {
        let mut observation = combined_observation(
            satellite_id,
            epoch_index,
            melbourne_wubbena_cycles,
            geometry_free_m,
        );
        observation.ambiguity_id = ambiguity_id.to_string();
        observation
    }

    fn constant_combination_epochs() -> Vec<super::super::prep::DualFrequencyEpoch> {
        (0..3)
            .map(|epoch| super::super::prep::DualFrequencyEpoch {
                gap_time_s: Some(epoch as f64 * 30.0),
                observations: vec![combined_observation("G01", epoch, 5.0, 0.0)],
            })
            .collect()
    }

    fn quiet_cycle_slip_config() -> CycleSlipConfig {
        CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 2.0,
            geometry_free_threshold_m: 0.05,
            maximum_gap_s: 120.0,
            ..CycleSlipConfig::default()
        }
    }

    fn slipped_epoch_satellites(
        flags: &[CycleSlipFlagEpoch],
    ) -> Vec<(usize, String, Vec<SlipReason>)> {
        flags
            .iter()
            .enumerate()
            .flat_map(|(epoch_index, epoch)| {
                epoch
                    .observations
                    .iter()
                    .filter(|observation| observation.slip)
                    .map(move |observation| {
                        (
                            epoch_index,
                            observation.satellite_id.clone(),
                            observation.reasons.clone(),
                        )
                    })
            })
            .collect()
    }

    fn slipped_epoch_observations(
        flags: &[CycleSlipFlagEpoch],
    ) -> Vec<(usize, usize, String, Vec<SlipReason>)> {
        flags
            .iter()
            .enumerate()
            .flat_map(|(epoch_index, epoch)| {
                epoch
                    .observations
                    .iter()
                    .enumerate()
                    .filter(|(_, observation)| observation.slip)
                    .map(move |(observation_index, observation)| {
                        (
                            epoch_index,
                            observation_index,
                            observation.satellite_id.clone(),
                            observation.reasons.clone(),
                        )
                    })
            })
            .collect()
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "expected {actual} to be within tolerance of {expected}"
        );
    }
}
