//! Dual-frequency Total Electron Content estimation helpers.
//!
//! This module starts with the absolute, noisy code-derived slant TEC estimate.
//! It uses the dual-frequency code geometry-free combination `P1 - P2` and the
//! dispersive ionospheric group-delay relationship
//! `delay_m = 40.308e16 * TECU * (1 / f1^2 - 1 / f2^2)`, where carrier
//! frequencies are in hertz and one TECU is `1e16` electrons per square meter.
//!
//! ```
//! use sidereon_core::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
//! use sidereon_core::precise_positioning::{
//!     estimate_tec, DualFrequencyObservation, TecConfig, TecEpoch, TecObservation,
//!     TEC_GROUP_DELAY_COEFFICIENT,
//! };
//!
//! fn observation(_epoch: usize, slant_tec_tecu: f64, phase_bias_tecu: f64) -> DualFrequencyObservation {
//!     let denominator = TEC_GROUP_DELAY_COEFFICIENT
//!         * (1.0 / (F_L1_HZ * F_L1_HZ) - 1.0 / (F_L2_HZ * F_L2_HZ));
//!     let code_geometry_free_m = denominator * slant_tec_tecu;
//!     let phase_geometry_free_m = -denominator * (slant_tec_tecu + phase_bias_tecu);
//!     DualFrequencyObservation {
//!         satellite_id: "G01".to_string(),
//!         ambiguity_id: "G01".to_string(),
//!         p1_m: 0.0,
//!         p2_m: -code_geometry_free_m,
//!         phi1_cyc: phase_geometry_free_m / (C_M_S / F_L1_HZ),
//!         phi2_cyc: 0.0,
//!         f1_hz: F_L1_HZ,
//!         f2_hz: F_L2_HZ,
//!         lli1: None,
//!         lli2: None,
//!     }
//! }
//!
//! let epochs = (0..2)
//!     .map(|epoch| TecEpoch {
//!         time_s: epoch as f64 * 30.0,
//!         receiver_latitude_rad: 0.0,
//!         receiver_longitude_rad: 0.0,
//!         observations: vec![TecObservation {
//!             observation: observation(epoch, 20.0, 5.0),
//!             elevation_rad: 60.0_f64.to_radians(),
//!             azimuth_rad: 90.0_f64.to_radians(),
//!         }],
//!     })
//!     .collect::<Vec<_>>();
//! let tec = estimate_tec(&epochs, TecConfig::default())?;
//! assert_eq!(tec.arcs.len(), 1);
//! assert_eq!(tec.arcs[0].samples.len(), 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use core::f64::consts::{FRAC_PI_2, PI, TAU};
use std::collections::BTreeMap;

use crate::constants::MEAN_EARTH_RADIUS_M;
use crate::tolerances::FREQUENCY_DENOMINATOR_EPS_HZ;
use crate::validate;

use super::cycle_slip::{geometry_free_m as phase_geometry_free_combination_m, CycleSlipError};
use super::prep::DualFrequencyObservation;

/// Default single-layer ionospheric shell height in meters.
pub const DEFAULT_IONOSPHERIC_SHELL_HEIGHT_M: f64 = 350_000.0;

/// Electrons per square meter represented by one TECU.
pub const ELECTRONS_PER_TECU_M2: f64 = 1.0e16;

/// Ionospheric group-delay coefficient for TECU inputs.
///
/// Frequencies are in hertz, so multiplying this coefficient by
/// `(1 / f1^2 - 1 / f2^2)` and a slant TEC value in TECU yields meters.
pub const TEC_GROUP_DELAY_COEFFICIENT: f64 = 40.308 * ELECTRONS_PER_TECU_M2;

/// Configuration for thin-shell TEC estimation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TecConfig {
    /// Ionospheric shell height above the spherical Earth, in meters.
    pub shell_height_m: f64,
    /// Spherical Earth radius used by the mapping function, in meters.
    pub earth_radius_m: f64,
}

impl TecConfig {
    /// Validate the shell geometry constants.
    pub fn validate(&self) -> Result<(), TecError> {
        validate::finite_positive(self.shell_height_m, "shell_height_m")
            .map_err(|_| TecError::InvalidShellHeight)?;
        validate::finite_positive(self.earth_radius_m, "earth_radius_m")
            .map_err(|_| TecError::InvalidEarthRadius)?;
        Ok(())
    }
}

impl Default for TecConfig {
    fn default() -> Self {
        Self {
            shell_height_m: DEFAULT_IONOSPHERIC_SHELL_HEIGHT_M,
            earth_radius_m: MEAN_EARTH_RADIUS_M,
        }
    }
}

/// One satellite's dual-frequency TEC observation with topocentric geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct TecObservation {
    /// Raw dual-frequency code and carrier-phase observation.
    pub observation: DualFrequencyObservation,
    /// Satellite elevation at the receiver, in radians.
    pub elevation_rad: f64,
    /// Satellite azimuth clockwise from north, in radians.
    pub azimuth_rad: f64,
}

/// One epoch of dual-frequency TEC observations.
#[derive(Debug, Clone, PartialEq)]
pub struct TecEpoch {
    /// Comparable epoch coordinate in seconds.
    pub time_s: f64,
    /// Receiver geodetic latitude, in radians.
    pub receiver_latitude_rad: f64,
    /// Receiver geodetic longitude, in radians.
    pub receiver_longitude_rad: f64,
    /// Satellite observations at this epoch.
    pub observations: Vec<TecObservation>,
}

/// TEC estimates for all continuous satellite arcs in a stream.
#[derive(Debug, Clone, PartialEq)]
pub struct TecEstimate {
    /// Per-satellite continuous arcs sorted by satellite and ambiguity id.
    pub arcs: Vec<TecSatelliteArc>,
}

/// TEC estimates for one satellite continuous arc.
#[derive(Debug, Clone, PartialEq)]
pub struct TecSatelliteArc {
    /// Satellite identifier copied from the source observations.
    pub satellite_id: String,
    /// Ambiguity or continuous-arc identifier copied from the source observations.
    pub ambiguity_id: String,
    /// Estimated phase ambiguity bias for this arc, in TECU.
    pub phase_bias_tecu: f64,
    /// Per-epoch estimates in input time order.
    pub samples: Vec<TecEstimateSample>,
}

/// One leveled TEC estimate at one epoch for one satellite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TecEstimateSample {
    /// Epoch coordinate copied from the input, in seconds.
    pub time_s: f64,
    /// Satellite elevation at the receiver, in radians.
    pub elevation_rad: f64,
    /// Satellite azimuth clockwise from north, in radians.
    pub azimuth_rad: f64,
    /// Code geometry-free combination `P1 - P2`, in meters.
    pub code_geometry_free_m: f64,
    /// Carrier phase geometry-free combination `L1 - L2`, in meters.
    pub phase_geometry_free_m: f64,
    /// Absolute, noisy code-derived slant TEC, in TECU.
    pub code_slant_tec_tecu: f64,
    /// Precise, biased phase-derived slant TEC, in TECU.
    pub phase_slant_tec_tecu: f64,
    /// Phase-derived slant TEC after removing the arc bias, in TECU.
    pub leveled_slant_tec_tecu: f64,
    /// Thin-shell mapping function used to map slant TEC to vertical TEC.
    pub mapping_function: f64,
    /// Leveled vertical TEC, in TECU.
    pub vertical_tec_tecu: f64,
    /// Ionospheric pierce point for this sample.
    pub pierce_point: IonosphericPiercePoint,
}

/// Code geometry-free slant TEC estimate for one dual-frequency observation.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeSlantTecEstimate {
    /// Satellite identifier copied from the source observation.
    pub satellite_id: String,
    /// Ambiguity or continuous-arc identifier copied from the source observation.
    pub ambiguity_id: String,
    /// Code geometry-free combination `P1 - P2`, in meters.
    pub code_geometry_free_m: f64,
    /// Absolute code-derived slant TEC, in TECU.
    pub slant_tec_tecu: f64,
}

/// Phase geometry-free slant TEC estimate for one dual-frequency observation.
#[derive(Debug, Clone, PartialEq)]
pub struct PhaseSlantTecEstimate {
    /// Satellite identifier copied from the source observation.
    pub satellite_id: String,
    /// Ambiguity or continuous-arc identifier copied from the source observation.
    pub ambiguity_id: String,
    /// Carrier phase geometry-free combination `L1 - L2`, in meters.
    pub phase_geometry_free_m: f64,
    /// Precise phase-derived slant TEC, in TECU, including the arc ambiguity bias.
    pub slant_tec_tecu: f64,
}

/// One dual-frequency slant TEC sample used by phase-code leveling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TecLevelingSample {
    /// Absolute, noisy code-derived slant TEC, in TECU.
    pub code_slant_tec_tecu: f64,
    /// Precise, biased phase-derived slant TEC, in TECU.
    pub phase_slant_tec_tecu: f64,
    /// Satellite elevation at the receiver, in radians.
    pub elevation_rad: f64,
}

/// One leveled TEC sample emitted for a continuous arc.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LeveledTecSample {
    /// Absolute, noisy code-derived slant TEC input, in TECU.
    pub code_slant_tec_tecu: f64,
    /// Precise, biased phase-derived slant TEC input, in TECU.
    pub phase_slant_tec_tecu: f64,
    /// Phase-derived slant TEC after removing the arc bias, in TECU.
    pub leveled_slant_tec_tecu: f64,
    /// Thin-shell mapping function used to map slant TEC to vertical TEC.
    pub mapping_function: f64,
    /// Leveled vertical TEC, in TECU.
    pub vertical_tec_tecu: f64,
}

/// Result of phase-code leveling across one continuous satellite arc.
#[derive(Debug, Clone, PartialEq)]
pub struct TecLevelingResult {
    /// Estimated phase ambiguity bias, in TECU.
    pub phase_bias_tecu: f64,
    /// Leveled samples in input order.
    pub samples: Vec<LeveledTecSample>,
}

/// Ionospheric pierce point on the configured thin shell.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IonosphericPiercePoint {
    /// Pierce-point geodetic latitude, in radians.
    pub latitude_rad: f64,
    /// Pierce-point geodetic longitude normalized to `[-pi, pi)`, in radians.
    pub longitude_rad: f64,
    /// Pierce-point geodetic latitude, in degrees.
    pub latitude_deg: f64,
    /// Pierce-point geodetic longitude normalized to `[-180, 180)`, in degrees.
    pub longitude_deg: f64,
    /// Earth-central angle from receiver to pierce point, in radians.
    pub earth_central_angle_rad: f64,
    /// Shell height used for the pierce point, in meters.
    pub shell_height_m: f64,
}

/// Error produced while estimating dual-frequency TEC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TecError {
    /// One or more observation scalars were not finite.
    NonFiniteObservation,
    /// The configured shell height was not positive and finite.
    InvalidShellHeight,
    /// The configured Earth radius was not positive and finite.
    InvalidEarthRadius,
    /// One or both carrier frequencies were not positive and finite.
    InvalidFrequency,
    /// The two carrier frequencies were too close to form a TEC denominator.
    EqualFrequencies,
    /// Receiver geodetic latitude was not finite or not in `[-pi/2, pi/2]`.
    InvalidReceiverLatitude,
    /// Receiver geodetic longitude was not finite.
    InvalidReceiverLongitude,
    /// Elevation was not finite or not in `[0, pi/2]`.
    InvalidElevation,
    /// Azimuth was not finite.
    InvalidAzimuth,
    /// A supplied TEC value was not finite.
    NonFiniteTec,
    /// The supplied continuous arc had no samples.
    EmptyArc,
    /// The input epoch stream had no epochs.
    NoEpochs,
    /// The input epoch stream contained no satellite observations.
    NoObservations,
    /// An epoch time was not finite.
    NonFiniteEpochTime,
    /// Epoch times were not ordered.
    EpochsNotOrdered,
    /// A satellite arc did not contain enough samples for leveling.
    InsufficientArcSamples,
}

impl core::fmt::Display for TecError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteObservation => write!(f, "TEC observation must be finite"),
            Self::InvalidShellHeight => write!(f, "TEC shell height must be positive and finite"),
            Self::InvalidEarthRadius => write!(f, "TEC Earth radius must be positive and finite"),
            Self::InvalidFrequency => write!(f, "carrier frequency must be positive and finite"),
            Self::EqualFrequencies => write!(f, "carrier frequencies must be distinct"),
            Self::InvalidReceiverLatitude => {
                write!(
                    f,
                    "receiver latitude must be finite and within [-pi/2, pi/2]"
                )
            }
            Self::InvalidReceiverLongitude => write!(f, "receiver longitude must be finite"),
            Self::InvalidElevation => {
                write!(f, "satellite elevation must be finite and within [0, pi/2]")
            }
            Self::InvalidAzimuth => write!(f, "satellite azimuth must be finite"),
            Self::NonFiniteTec => write!(f, "TEC value must be finite"),
            Self::EmptyArc => write!(f, "TEC leveling arc must contain at least one sample"),
            Self::NoEpochs => write!(f, "TEC epoch stream must contain at least one epoch"),
            Self::NoObservations => {
                write!(f, "TEC epoch stream must contain at least one observation")
            }
            Self::NonFiniteEpochTime => write!(f, "TEC epoch time must be finite"),
            Self::EpochsNotOrdered => write!(f, "TEC epochs must be time ordered"),
            Self::InsufficientArcSamples => {
                write!(f, "TEC satellite arc must contain at least two samples")
            }
        }
    }
}

impl std::error::Error for TecError {}

/// Compute the code geometry-free combination `P1 - P2`, in meters.
pub fn code_geometry_free_m(observation: &DualFrequencyObservation) -> Result<f64, TecError> {
    validate_code_observation(observation)?;
    let geometry_free_m = observation.p1_m - observation.p2_m;
    validate::finite(geometry_free_m, "code_geometry_free_m")
        .map_err(|_| TecError::NonFiniteObservation)?;
    Ok(geometry_free_m)
}

/// Convert a code geometry-free delay in meters into slant TEC in TECU.
pub fn slant_tec_from_code_geometry_free_m(
    code_geometry_free_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> Result<f64, TecError> {
    validate::finite(code_geometry_free_m, "code_geometry_free_m")
        .map_err(|_| TecError::NonFiniteObservation)?;
    if code_geometry_free_m == 0.0 {
        return Ok(0.0);
    }
    let denominator = tec_geometry_free_denominator_m_per_tecu(f1_hz, f2_hz)?;
    let slant_tec_tecu = code_geometry_free_m / denominator;
    validate::finite(slant_tec_tecu, "slant_tec_tecu").map_err(|_| TecError::NonFiniteTec)?;
    Ok(slant_tec_tecu)
}

/// Compute the carrier phase geometry-free combination `L1 - L2`, in meters.
pub fn phase_geometry_free_m(observation: &DualFrequencyObservation) -> Result<f64, TecError> {
    let geometry_free_m =
        phase_geometry_free_combination_m(observation).map_err(map_cycle_slip_error)?;
    validate::finite(geometry_free_m, "phase_geometry_free_m")
        .map_err(|_| TecError::NonFiniteObservation)?;
    Ok(geometry_free_m)
}

/// Convert a phase geometry-free delay in meters into biased slant TEC in TECU.
///
/// Carrier phase advances through the ionosphere with the opposite sign from
/// code group delay, so this conversion negates `L1 - L2` before applying the
/// same TEC denominator as the code geometry-free conversion.
pub fn slant_tec_from_phase_geometry_free_m(
    phase_geometry_free_m: f64,
    f1_hz: f64,
    f2_hz: f64,
) -> Result<f64, TecError> {
    validate::finite(phase_geometry_free_m, "phase_geometry_free_m")
        .map_err(|_| TecError::NonFiniteObservation)?;
    if phase_geometry_free_m == 0.0 {
        return Ok(0.0);
    }
    let denominator = tec_geometry_free_denominator_m_per_tecu(f1_hz, f2_hz)?;
    let slant_tec_tecu = -phase_geometry_free_m / denominator;
    validate::finite(slant_tec_tecu, "slant_tec_tecu").map_err(|_| TecError::NonFiniteTec)?;
    Ok(slant_tec_tecu)
}

/// Estimate absolute code-derived slant TEC for one dual-frequency observation.
pub fn estimate_code_slant_tec(
    observation: &DualFrequencyObservation,
) -> Result<CodeSlantTecEstimate, TecError> {
    let code_geometry_free_m = code_geometry_free_m(observation)?;
    let slant_tec_tecu = slant_tec_from_code_geometry_free_m(
        code_geometry_free_m,
        observation.f1_hz,
        observation.f2_hz,
    )?;
    Ok(CodeSlantTecEstimate {
        satellite_id: observation.satellite_id.clone(),
        ambiguity_id: observation.ambiguity_id.clone(),
        code_geometry_free_m,
        slant_tec_tecu,
    })
}

/// Estimate biased phase-derived slant TEC for one dual-frequency observation.
pub fn estimate_phase_slant_tec(
    observation: &DualFrequencyObservation,
) -> Result<PhaseSlantTecEstimate, TecError> {
    let phase_geometry_free_m = phase_geometry_free_m(observation)?;
    let slant_tec_tecu = slant_tec_from_phase_geometry_free_m(
        phase_geometry_free_m,
        observation.f1_hz,
        observation.f2_hz,
    )?;
    Ok(PhaseSlantTecEstimate {
        satellite_id: observation.satellite_id.clone(),
        ambiguity_id: observation.ambiguity_id.clone(),
        phase_geometry_free_m,
        slant_tec_tecu,
    })
}

/// Thin-shell obliquity factor mapping vertical TEC to slant TEC.
///
/// The mapping function is
/// `1 / sqrt(1 - (Re * cos(elevation) / (Re + H))^2)`, where `Re` and `H` come
/// from [`TecConfig`] and elevation is in radians.
pub fn thin_shell_mapping_function(elevation_rad: f64, config: TecConfig) -> Result<f64, TecError> {
    config.validate()?;
    validate_elevation(elevation_rad)?;
    let shell_radius_m = config.earth_radius_m + config.shell_height_m;
    validate::finite_positive(shell_radius_m, "shell_radius_m")
        .map_err(|_| TecError::InvalidShellHeight)?;
    let obliquity_arg = config.earth_radius_m * elevation_rad.cos() / shell_radius_m;
    validate::finite(obliquity_arg, "obliquity_arg").map_err(|_| TecError::InvalidShellHeight)?;
    let mapping_denominator = 1.0 - obliquity_arg * obliquity_arg;
    validate::finite_positive(mapping_denominator, "mapping_denominator")
        .map_err(|_| TecError::InvalidShellHeight)?;
    let mapping_function = 1.0 / mapping_denominator.sqrt();
    validate::finite(mapping_function, "mapping_function")
        .map_err(|_| TecError::InvalidShellHeight)?;
    Ok(mapping_function)
}

/// Convert slant TEC to vertical TEC with the configured thin-shell mapping.
pub fn vertical_tec_from_slant_tec(
    slant_tec_tecu: f64,
    elevation_rad: f64,
    config: TecConfig,
) -> Result<f64, TecError> {
    validate_tec(slant_tec_tecu)?;
    let mapping_function = thin_shell_mapping_function(elevation_rad, config)?;
    Ok(slant_tec_tecu / mapping_function)
}

/// Level a continuous arc of code and phase slant TEC samples.
///
/// The phase ambiguity bias is the arc mean of `phase_slant_tec_tecu -
/// code_slant_tec_tecu`. Each output slant TEC is the phase slant TEC minus that
/// bias, and each vertical TEC is the leveled slant TEC divided by the
/// thin-shell mapping function at the sample elevation.
pub fn level_slant_tec_arc(
    samples: &[TecLevelingSample],
    config: TecConfig,
) -> Result<TecLevelingResult, TecError> {
    config.validate()?;
    if samples.is_empty() {
        return Err(TecError::EmptyArc);
    }

    let mut bias_sum_tecu = 0.0;
    for sample in samples {
        validate_leveling_sample(sample)?;
        bias_sum_tecu += sample.phase_slant_tec_tecu - sample.code_slant_tec_tecu;
    }
    let phase_bias_tecu = bias_sum_tecu / samples.len() as f64;

    let leveled_samples = samples
        .iter()
        .map(|sample| {
            let mapping_function = thin_shell_mapping_function(sample.elevation_rad, config)?;
            let leveled_slant_tec_tecu = sample.phase_slant_tec_tecu - phase_bias_tecu;
            let vertical_tec_tecu = leveled_slant_tec_tecu / mapping_function;
            Ok(LeveledTecSample {
                code_slant_tec_tecu: sample.code_slant_tec_tecu,
                phase_slant_tec_tecu: sample.phase_slant_tec_tecu,
                leveled_slant_tec_tecu,
                mapping_function,
                vertical_tec_tecu,
            })
        })
        .collect::<Result<Vec<_>, TecError>>()?;

    Ok(TecLevelingResult {
        phase_bias_tecu,
        samples: leveled_samples,
    })
}

/// Estimate TEC over a time-ordered stream of dual-frequency epochs.
///
/// Observations are grouped into continuous arcs by `(satellite_id,
/// ambiguity_id)`. Each arc must contain at least two samples, then code and
/// phase slant TEC are leveled, mapped to vertical TEC, and paired with a
/// thin-shell ionospheric pierce point for every sample.
pub fn estimate_tec(epochs: &[TecEpoch], config: TecConfig) -> Result<TecEstimate, TecError> {
    validate_tec_epochs(epochs, config)?;

    let mut arcs = BTreeMap::<(String, String), Vec<TecArcBuildSample>>::new();
    for epoch in epochs {
        for observation in &epoch.observations {
            let code_estimate = estimate_code_slant_tec(&observation.observation)?;
            let phase_estimate = estimate_phase_slant_tec(&observation.observation)?;
            arcs.entry((
                observation.observation.satellite_id.clone(),
                observation.observation.ambiguity_id.clone(),
            ))
            .or_default()
            .push(TecArcBuildSample {
                time_s: epoch.time_s,
                receiver_latitude_rad: epoch.receiver_latitude_rad,
                receiver_longitude_rad: epoch.receiver_longitude_rad,
                elevation_rad: observation.elevation_rad,
                azimuth_rad: observation.azimuth_rad,
                code_geometry_free_m: code_estimate.code_geometry_free_m,
                phase_geometry_free_m: phase_estimate.phase_geometry_free_m,
                code_slant_tec_tecu: code_estimate.slant_tec_tecu,
                phase_slant_tec_tecu: phase_estimate.slant_tec_tecu,
            });
        }
    }

    if arcs.is_empty() {
        return Err(TecError::NoObservations);
    }

    let mut out_arcs = Vec::with_capacity(arcs.len());
    for ((satellite_id, ambiguity_id), samples) in arcs {
        if samples.len() < 2 {
            return Err(TecError::InsufficientArcSamples);
        }
        let leveling_samples = samples
            .iter()
            .map(|sample| TecLevelingSample {
                code_slant_tec_tecu: sample.code_slant_tec_tecu,
                phase_slant_tec_tecu: sample.phase_slant_tec_tecu,
                elevation_rad: sample.elevation_rad,
            })
            .collect::<Vec<_>>();
        let leveled = level_slant_tec_arc(&leveling_samples, config)?;
        let output_samples = samples
            .iter()
            .zip(leveled.samples.iter())
            .map(|(sample, leveled)| {
                let pierce_point = ionospheric_pierce_point(
                    sample.receiver_latitude_rad,
                    sample.receiver_longitude_rad,
                    sample.elevation_rad,
                    sample.azimuth_rad,
                    config,
                )?;
                Ok(TecEstimateSample {
                    time_s: sample.time_s,
                    elevation_rad: sample.elevation_rad,
                    azimuth_rad: sample.azimuth_rad,
                    code_geometry_free_m: sample.code_geometry_free_m,
                    phase_geometry_free_m: sample.phase_geometry_free_m,
                    code_slant_tec_tecu: sample.code_slant_tec_tecu,
                    phase_slant_tec_tecu: sample.phase_slant_tec_tecu,
                    leveled_slant_tec_tecu: leveled.leveled_slant_tec_tecu,
                    mapping_function: leveled.mapping_function,
                    vertical_tec_tecu: leveled.vertical_tec_tecu,
                    pierce_point,
                })
            })
            .collect::<Result<Vec<_>, TecError>>()?;
        out_arcs.push(TecSatelliteArc {
            satellite_id,
            ambiguity_id,
            phase_bias_tecu: leveled.phase_bias_tecu,
            samples: output_samples,
        });
    }

    Ok(TecEstimate { arcs: out_arcs })
}

/// Compute the ionospheric pierce point for a receiver and satellite look angle.
///
/// Receiver latitude, receiver longitude, satellite elevation, and satellite
/// azimuth are all radians. Azimuth is clockwise from north. The returned
/// longitude is normalized to `[-pi, pi)`.
pub fn ionospheric_pierce_point(
    receiver_latitude_rad: f64,
    receiver_longitude_rad: f64,
    elevation_rad: f64,
    azimuth_rad: f64,
    config: TecConfig,
) -> Result<IonosphericPiercePoint, TecError> {
    config.validate()?;
    validate_receiver_latitude(receiver_latitude_rad)?;
    validate_receiver_longitude(receiver_longitude_rad)?;
    validate_elevation(elevation_rad)?;
    validate_azimuth(azimuth_rad)?;

    let shell_radius_m = config.earth_radius_m + config.shell_height_m;
    let shell_scaled_cosine = config.earth_radius_m / shell_radius_m * elevation_rad.cos();
    let earth_central_angle_rad = FRAC_PI_2 - elevation_rad - shell_scaled_cosine.asin();

    let receiver_sin = receiver_latitude_rad.sin();
    let receiver_cos = receiver_latitude_rad.cos();
    let psi_sin = earth_central_angle_rad.sin();
    let psi_cos = earth_central_angle_rad.cos();
    let azimuth_sin = azimuth_rad.sin();
    let azimuth_cos = azimuth_rad.cos();

    let latitude_rad = (receiver_sin * psi_cos + receiver_cos * psi_sin * azimuth_cos).asin();
    let longitude_step_rad =
        (azimuth_sin * psi_sin * receiver_cos).atan2(psi_cos - receiver_sin * latitude_rad.sin());
    let longitude_rad = normalize_longitude_rad(receiver_longitude_rad + longitude_step_rad);

    Ok(IonosphericPiercePoint {
        latitude_rad,
        longitude_rad,
        latitude_deg: latitude_rad.to_degrees(),
        longitude_deg: longitude_rad.to_degrees(),
        earth_central_angle_rad,
        shell_height_m: config.shell_height_m,
    })
}

fn tec_geometry_free_denominator_m_per_tecu(f1_hz: f64, f2_hz: f64) -> Result<f64, TecError> {
    let f1_hz = validate_frequency(f1_hz)?;
    let f2_hz = validate_frequency(f2_hz)?;
    if (f1_hz - f2_hz).abs() < FREQUENCY_DENOMINATOR_EPS_HZ {
        return Err(TecError::EqualFrequencies);
    }
    let denominator = TEC_GROUP_DELAY_COEFFICIENT * (1.0 / (f1_hz * f1_hz) - 1.0 / (f2_hz * f2_hz));
    validate::finite(denominator, "tec_geometry_free_denominator_m_per_tecu")
        .map_err(|_| TecError::InvalidFrequency)?;
    if denominator == 0.0 {
        return Err(TecError::EqualFrequencies);
    }
    Ok(denominator)
}

fn validate_frequency(frequency_hz: f64) -> Result<f64, TecError> {
    validate::finite_positive(frequency_hz, "frequency_hz").map_err(|_| TecError::InvalidFrequency)
}

fn validate_code_observation(observation: &DualFrequencyObservation) -> Result<(), TecError> {
    if observation.p1_m.is_finite() && observation.p2_m.is_finite() {
        Ok(())
    } else {
        Err(TecError::NonFiniteObservation)
    }
}

#[derive(Debug, Clone)]
struct TecArcBuildSample {
    time_s: f64,
    receiver_latitude_rad: f64,
    receiver_longitude_rad: f64,
    elevation_rad: f64,
    azimuth_rad: f64,
    code_geometry_free_m: f64,
    phase_geometry_free_m: f64,
    code_slant_tec_tecu: f64,
    phase_slant_tec_tecu: f64,
}

fn validate_tec_epochs(epochs: &[TecEpoch], config: TecConfig) -> Result<(), TecError> {
    config.validate()?;
    if epochs.is_empty() {
        return Err(TecError::NoEpochs);
    }

    let mut previous_time_s = None;
    let mut observation_count = 0usize;
    for epoch in epochs {
        if !epoch.time_s.is_finite() {
            return Err(TecError::NonFiniteEpochTime);
        }
        if let Some(previous_time_s) = previous_time_s {
            if epoch.time_s < previous_time_s {
                return Err(TecError::EpochsNotOrdered);
            }
        }
        previous_time_s = Some(epoch.time_s);
        validate_receiver_latitude(epoch.receiver_latitude_rad)?;
        validate_receiver_longitude(epoch.receiver_longitude_rad)?;
        for observation in &epoch.observations {
            validate_elevation(observation.elevation_rad)?;
            validate_azimuth(observation.azimuth_rad)?;
            observation_count += 1;
        }
    }

    if observation_count == 0 {
        Err(TecError::NoObservations)
    } else {
        Ok(())
    }
}

fn validate_tec(value: f64) -> Result<(), TecError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(TecError::NonFiniteTec)
    }
}

fn validate_leveling_sample(sample: &TecLevelingSample) -> Result<(), TecError> {
    validate_tec(sample.code_slant_tec_tecu)?;
    validate_tec(sample.phase_slant_tec_tecu)?;
    validate_elevation(sample.elevation_rad)
}

fn map_cycle_slip_error(error: CycleSlipError) -> TecError {
    match error {
        CycleSlipError::NonFiniteObservation => TecError::NonFiniteObservation,
        CycleSlipError::InvalidFrequency => TecError::InvalidFrequency,
        CycleSlipError::EqualFrequencies => TecError::EqualFrequencies,
        CycleSlipError::InvalidConfig(_)
        | CycleSlipError::NonFiniteEpochTime
        | CycleSlipError::EpochsNotOrdered => TecError::NonFiniteObservation,
    }
}

fn validate_receiver_latitude(latitude_rad: f64) -> Result<(), TecError> {
    if latitude_rad.is_finite() && (-FRAC_PI_2..=FRAC_PI_2).contains(&latitude_rad) {
        Ok(())
    } else {
        Err(TecError::InvalidReceiverLatitude)
    }
}

fn validate_receiver_longitude(longitude_rad: f64) -> Result<(), TecError> {
    if longitude_rad.is_finite() {
        Ok(())
    } else {
        Err(TecError::InvalidReceiverLongitude)
    }
}

fn validate_elevation(elevation_rad: f64) -> Result<(), TecError> {
    if elevation_rad.is_finite() && (0.0..=FRAC_PI_2).contains(&elevation_rad) {
        Ok(())
    } else {
        Err(TecError::InvalidElevation)
    }
}

fn validate_azimuth(azimuth_rad: f64) -> Result<(), TecError> {
    if azimuth_rad.is_finite() {
        Ok(())
    } else {
        Err(TecError::InvalidAzimuth)
    }
}

fn normalize_longitude_rad(longitude_rad: f64) -> f64 {
    let mut normalized = (longitude_rad + PI) % TAU;
    if normalized < 0.0 {
        normalized += TAU;
    }
    normalized - PI
}

#[cfg(test)]
mod tests {
    use crate::constants::{F_L1_HZ, F_L2_HZ};

    use super::*;

    fn deg(value: f64) -> f64 {
        value.to_radians()
    }

    fn observation_with_code_geometry_free(code_geometry_free_m: f64) -> DualFrequencyObservation {
        let (p1_m, p2_m) = if code_geometry_free_m.is_sign_negative() {
            (0.0, -code_geometry_free_m)
        } else {
            (code_geometry_free_m, 0.0)
        };
        DualFrequencyObservation {
            satellite_id: "G01".to_string(),
            ambiguity_id: "G01".to_string(),
            p1_m,
            p2_m,
            phi1_cyc: 0.0,
            phi2_cyc: 0.0,
            f1_hz: F_L1_HZ,
            f2_hz: F_L2_HZ,
            lli1: None,
            lli2: None,
        }
    }

    fn observation_from_slant_tec(
        satellite_id: &str,
        ambiguity_id: &str,
        code_slant_tec_tecu: f64,
        phase_slant_tec_tecu: f64,
    ) -> DualFrequencyObservation {
        let denominator = tec_geometry_free_denominator_m_per_tecu(F_L1_HZ, F_L2_HZ)
            .expect("GPS L1/L2 TEC denominator");
        let code_geometry_free_m = denominator * code_slant_tec_tecu;
        let phase_geometry_free_m = -denominator * phase_slant_tec_tecu;
        DualFrequencyObservation {
            satellite_id: satellite_id.to_string(),
            ambiguity_id: ambiguity_id.to_string(),
            p1_m: 0.0,
            p2_m: -code_geometry_free_m,
            phi1_cyc: phase_geometry_free_m / (crate::constants::C_M_S / F_L1_HZ),
            phi2_cyc: 0.0,
            f1_hz: F_L1_HZ,
            f2_hz: F_L2_HZ,
            lli1: None,
            lli2: None,
        }
    }

    fn arc_by_satellite<'a>(estimate: &'a TecEstimate, satellite_id: &str) -> &'a TecSatelliteArc {
        estimate
            .arcs
            .iter()
            .find(|arc| arc.satellite_id == satellite_id)
            .expect("satellite arc")
    }

    fn assert_close(left: f64, right: f64, tolerance: f64) {
        assert!(
            (left - right).abs() <= tolerance,
            "{left} differs from {right} by more than {tolerance}"
        );
    }

    #[test]
    fn code_geometry_free_delay_maps_to_expected_slant_tec() {
        let expected_slant_tec_tecu = 17.25;
        let code_geometry_free_m = expected_slant_tec_tecu
            * tec_geometry_free_denominator_m_per_tecu(F_L1_HZ, F_L2_HZ)
                .expect("GPS L1/L2 TEC denominator");
        let observation = observation_with_code_geometry_free(code_geometry_free_m);

        let estimate = estimate_code_slant_tec(&observation).expect("code slant TEC");

        assert_close(estimate.code_geometry_free_m, code_geometry_free_m, 1.0e-9);
        assert_close(estimate.slant_tec_tecu, expected_slant_tec_tecu, 1.0e-12);
    }

    #[test]
    fn zero_code_geometry_free_delay_gives_zero_slant_tec() {
        let observation = observation_with_code_geometry_free(0.0);

        let estimate = estimate_code_slant_tec(&observation).expect("code slant TEC");

        assert_eq!(estimate.code_geometry_free_m.to_bits(), 0.0f64.to_bits());
        assert_eq!(estimate.slant_tec_tecu.to_bits(), 0.0f64.to_bits());
    }

    #[test]
    fn phase_geometry_free_delay_maps_to_biased_slant_tec() {
        let true_slant_tec_tecu = 21.0;
        let phase_bias_tecu = 9.5;
        let denominator = tec_geometry_free_denominator_m_per_tecu(F_L1_HZ, F_L2_HZ)
            .expect("GPS L1/L2 TEC denominator");
        let phase_geometry_free_m = -(true_slant_tec_tecu + phase_bias_tecu) * denominator;

        let slant_tec_tecu =
            slant_tec_from_phase_geometry_free_m(phase_geometry_free_m, F_L1_HZ, F_L2_HZ)
                .expect("phase slant TEC");

        assert_close(
            slant_tec_tecu,
            true_slant_tec_tecu + phase_bias_tecu,
            1.0e-12,
        );
    }

    #[test]
    fn phase_slant_tec_rejects_collapsed_frequency_denominator() {
        assert_eq!(
            slant_tec_from_phase_geometry_free_m(1.0, f64::MAX, f64::MAX / 2.0),
            Err(TecError::EqualFrequencies)
        );
    }

    #[test]
    fn mapping_function_is_one_at_zenith_and_increases_toward_horizon() {
        let config = TecConfig::default();

        let zenith = thin_shell_mapping_function(FRAC_PI_2, config).expect("zenith mapping");
        let high = thin_shell_mapping_function(deg(60.0), config).expect("high mapping");
        let low = thin_shell_mapping_function(deg(30.0), config).expect("low mapping");
        let horizon = thin_shell_mapping_function(0.0, config).expect("horizon mapping");

        assert_close(zenith, 1.0, 1.0e-15);
        assert!(high > zenith);
        assert!(low > high);
        assert!(horizon > low);
    }

    #[test]
    fn mapping_function_rejects_degenerate_shell_geometry() {
        let config = TecConfig {
            shell_height_m: f64::MIN_POSITIVE,
            earth_radius_m: 1.0,
        };

        assert_eq!(
            thin_shell_mapping_function(0.0, config),
            Err(TecError::InvalidShellHeight)
        );
    }

    #[test]
    fn synthetic_leveled_arc_recovers_constant_vertical_tec() {
        let config = TecConfig::default();
        let vertical_tec_tecu = 14.0;
        let phase_bias_tecu = 37.5;
        let noise_tecu = [0.6, -0.2, -0.4, 0.0];
        let elevations_rad = [deg(30.0), deg(45.0), deg(60.0), deg(75.0)];
        let samples = elevations_rad
            .iter()
            .zip(noise_tecu)
            .map(|(&elevation_rad, noise_tecu)| {
                let mapping_function =
                    thin_shell_mapping_function(elevation_rad, config).expect("mapping");
                let true_slant_tec_tecu = vertical_tec_tecu * mapping_function;
                TecLevelingSample {
                    code_slant_tec_tecu: true_slant_tec_tecu + noise_tecu,
                    phase_slant_tec_tecu: true_slant_tec_tecu + phase_bias_tecu,
                    elevation_rad,
                }
            })
            .collect::<Vec<_>>();

        let result = level_slant_tec_arc(&samples, config).expect("leveled TEC arc");

        assert_close(result.phase_bias_tecu, phase_bias_tecu, 1.0e-12);
        for sample in result.samples {
            assert_close(sample.vertical_tec_tecu, vertical_tec_tecu, 1.0e-12);
        }
    }

    #[test]
    fn known_elevation_profile_yields_expected_slant_to_vertical_reduction() {
        let config = TecConfig::default();
        let vertical_tec_tecu = 8.25;
        let elevations_rad = [deg(25.0), deg(55.0), deg(85.0)];
        let samples = elevations_rad
            .iter()
            .map(|&elevation_rad| {
                let mapping_function =
                    thin_shell_mapping_function(elevation_rad, config).expect("mapping");
                let slant_tec_tecu = vertical_tec_tecu * mapping_function;
                TecLevelingSample {
                    code_slant_tec_tecu: slant_tec_tecu,
                    phase_slant_tec_tecu: slant_tec_tecu,
                    elevation_rad,
                }
            })
            .collect::<Vec<_>>();

        let result = level_slant_tec_arc(&samples, config).expect("leveled TEC arc");

        assert_close(result.phase_bias_tecu, 0.0, 1.0e-12);
        for (sample, elevation_rad) in result.samples.iter().zip(elevations_rad) {
            let mapping_function =
                thin_shell_mapping_function(elevation_rad, config).expect("mapping");
            assert_close(sample.mapping_function, mapping_function, 1.0e-15);
            assert_close(sample.vertical_tec_tecu, vertical_tec_tecu, 1.0e-12);
        }
    }

    #[test]
    fn estimate_tec_multi_epoch_stream_returns_vertical_tec_and_pierce_points() {
        let config = TecConfig::default();
        let receiver_latitude_rad = 0.0;
        let receiver_longitude_rad = 0.0;
        let g01_vertical_tec_tecu = 11.0;
        let g02_vertical_tec_tecu = 16.0;
        let g01_phase_bias_tecu = 25.0;
        let g02_phase_bias_tecu = -13.0;
        let epochs = [0.0, 30.0, 60.0]
            .into_iter()
            .enumerate()
            .map(|(idx, time_s)| {
                let g01_elevation_rad = [deg(45.0), deg(55.0), deg(65.0)][idx];
                let g02_elevation_rad = [deg(40.0), deg(50.0), deg(70.0)][idx];
                let g01_mapping =
                    thin_shell_mapping_function(g01_elevation_rad, config).expect("G01 mapping");
                let g02_mapping =
                    thin_shell_mapping_function(g02_elevation_rad, config).expect("G02 mapping");
                let g01_slant_tec_tecu = g01_vertical_tec_tecu * g01_mapping;
                let g02_slant_tec_tecu = g02_vertical_tec_tecu * g02_mapping;
                TecEpoch {
                    time_s,
                    receiver_latitude_rad,
                    receiver_longitude_rad,
                    observations: vec![
                        TecObservation {
                            observation: observation_from_slant_tec(
                                "G01",
                                "G01",
                                g01_slant_tec_tecu,
                                g01_slant_tec_tecu + g01_phase_bias_tecu,
                            ),
                            elevation_rad: g01_elevation_rad,
                            azimuth_rad: deg(90.0),
                        },
                        TecObservation {
                            observation: observation_from_slant_tec(
                                "G02",
                                "G02",
                                g02_slant_tec_tecu,
                                g02_slant_tec_tecu + g02_phase_bias_tecu,
                            ),
                            elevation_rad: g02_elevation_rad,
                            azimuth_rad: 0.0,
                        },
                    ],
                }
            })
            .collect::<Vec<_>>();

        let estimate = estimate_tec(&epochs, config).expect("TEC estimate");

        assert_eq!(estimate.arcs.len(), 2);
        let g01 = arc_by_satellite(&estimate, "G01");
        let g02 = arc_by_satellite(&estimate, "G02");
        assert_close(g01.phase_bias_tecu, g01_phase_bias_tecu, 1.0e-12);
        assert_close(g02.phase_bias_tecu, g02_phase_bias_tecu, 1.0e-12);
        for sample in &g01.samples {
            assert_close(sample.vertical_tec_tecu, g01_vertical_tec_tecu, 1.0e-12);
            assert_close(sample.pierce_point.latitude_rad, 0.0, 1.0e-12);
            assert!(sample.pierce_point.longitude_rad > 0.0);
        }
        for sample in &g02.samples {
            assert_close(sample.vertical_tec_tecu, g02_vertical_tec_tecu, 1.0e-12);
            assert!(sample.pierce_point.latitude_rad > 0.0);
            assert_close(sample.pierce_point.longitude_rad, 0.0, 1.0e-12);
        }
    }

    #[test]
    fn estimate_tec_rejects_insufficient_and_invalid_inputs() {
        let config = TecConfig::default();
        assert_eq!(estimate_tec(&[], config), Err(TecError::NoEpochs));

        let single_epoch = vec![TecEpoch {
            time_s: 0.0,
            receiver_latitude_rad: 0.0,
            receiver_longitude_rad: 0.0,
            observations: vec![TecObservation {
                observation: observation_from_slant_tec("G01", "G01", 10.0, 12.0),
                elevation_rad: deg(45.0),
                azimuth_rad: 0.0,
            }],
        }];
        assert_eq!(
            estimate_tec(&single_epoch, config),
            Err(TecError::InsufficientArcSamples)
        );

        let unordered = vec![
            TecEpoch {
                time_s: 30.0,
                receiver_latitude_rad: 0.0,
                receiver_longitude_rad: 0.0,
                observations: Vec::new(),
            },
            TecEpoch {
                time_s: 0.0,
                receiver_latitude_rad: 0.0,
                receiver_longitude_rad: 0.0,
                observations: Vec::new(),
            },
        ];
        assert_eq!(
            estimate_tec(&unordered, config),
            Err(TecError::EpochsNotOrdered)
        );

        let invalid_elevation = vec![TecEpoch {
            time_s: 0.0,
            receiver_latitude_rad: 0.0,
            receiver_longitude_rad: 0.0,
            observations: vec![TecObservation {
                observation: observation_from_slant_tec("G01", "G01", 10.0, 12.0),
                elevation_rad: -0.1,
                azimuth_rad: 0.0,
            }],
        }];
        assert_eq!(
            estimate_tec(&invalid_elevation, config),
            Err(TecError::InvalidElevation)
        );
    }

    #[test]
    fn pierce_point_at_zenith_equals_receiver_horizontal_position() {
        let config = TecConfig::default();
        let receiver_latitude_rad = deg(34.25);
        let receiver_longitude_rad = deg(-118.125);

        let pierce_point = ionospheric_pierce_point(
            receiver_latitude_rad,
            receiver_longitude_rad,
            FRAC_PI_2,
            deg(127.0),
            config,
        )
        .expect("zenith pierce point");

        assert_close(pierce_point.latitude_rad, receiver_latitude_rad, 1.0e-12);
        assert_close(pierce_point.longitude_rad, receiver_longitude_rad, 1.0e-12);
        assert_close(pierce_point.earth_central_angle_rad, 0.0, 1.0e-12);
    }

    #[test]
    fn pierce_point_moves_toward_satellite_azimuth_as_elevation_decreases() {
        let config = TecConfig::default();
        let receiver_latitude_rad = 0.0;
        let receiver_longitude_rad = 0.0;
        let east_azimuth_rad = deg(90.0);

        let high = ionospheric_pierce_point(
            receiver_latitude_rad,
            receiver_longitude_rad,
            deg(80.0),
            east_azimuth_rad,
            config,
        )
        .expect("high-elevation pierce point");
        let low = ionospheric_pierce_point(
            receiver_latitude_rad,
            receiver_longitude_rad,
            deg(30.0),
            east_azimuth_rad,
            config,
        )
        .expect("low-elevation pierce point");

        assert_close(high.latitude_rad, 0.0, 1.0e-12);
        assert_close(low.latitude_rad, 0.0, 1.0e-12);
        assert!(high.longitude_rad > 0.0);
        assert!(low.longitude_rad > high.longitude_rad);
    }
}
