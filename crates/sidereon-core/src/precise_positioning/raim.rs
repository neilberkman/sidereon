//! Snapshot RAIM tests for PPP float residuals.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use std::collections::BTreeMap;
//! # use sidereon_core::constants::F_L1_HZ;
//! # use sidereon_core::observables::{
//! #     predict, ObservableEphemerisSource, ObservableState, ObservablesError, PredictOptions,
//! # };
//! # use sidereon_core::ppp_corrections::CivilDateTime;
//! # use sidereon_core::precise_positioning::{
//! #     solve_float_epoch_with_raim, FloatEpoch, FloatObservation, FloatSolveConfig,
//! #     FloatSolveOptions, FloatState, MeasurementWeights, RaimConfig, RangeCorrections,
//! #     TroposphereOptions,
//! # };
//! # use sidereon_core::{GnssSatelliteId, GnssSystem};
//! #
//! # #[derive(Debug, Clone)]
//! # struct Source {
//! #     states: BTreeMap<GnssSatelliteId, [f64; 3]>,
//! # }
//! #
//! # impl ObservableEphemerisSource for Source {
//! #     fn observable_state_at_j2000_s(
//! #         &self,
//! #         sat: GnssSatelliteId,
//! #         _t_j2000_s: f64,
//! #     ) -> Result<ObservableState, ObservablesError> {
//! #         Ok(ObservableState {
//! #             position_ecef_m: *self.states.get(&sat).ok_or(ObservablesError::NoEphemeris)?,
//! #             clock_s: Some(0.0),
//! #         })
//! #     }
//! # }
//! #
//! # let sat_positions = [
//! #     (1u8, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
//! #     (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
//! #     (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
//! #     (4, [-18_200_000.0, -16_000_000.0, 21_000_000.0]),
//! #     (5, [22_000_000.0, -12_000_000.0, 20_200_000.0]),
//! #     (6, [-12_000_000.0, 23_000_000.0, 18_000_000.0]),
//! # ];
//! # let ids = sat_positions
//! #     .iter()
//! #     .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn))
//! #     .collect::<Result<Vec<_>, _>>()?;
//! # let source = Source {
//! #     states: ids
//! #         .iter()
//! #         .zip(sat_positions.iter())
//! #         .map(|(id, (_, position))| (*id, *position))
//! #         .collect(),
//! # };
//! # let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
//! # let clock_m = 12.5;
//! # let ambiguities_m = ids
//! #     .iter()
//! #     .enumerate()
//! #     .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
//! #     .collect::<BTreeMap<_, _>>();
//! # let observations = ids
//! #     .iter()
//! #     .map(|id| {
//! #         let satellite_id = id.to_string();
//! #         let prediction = predict(
//! #             &source,
//! #             *id,
//! #             truth,
//! #             0.0,
//! #             PredictOptions {
//! #                 carrier_hz: F_L1_HZ,
//! #                 light_time: true,
//! #                 sagnac: true,
//! #             },
//! #         )?;
//! #         let code_bias_m = if satellite_id == "G05" { 50.0 } else { 0.0 };
//! #         let code_m = prediction.geometric_range_m + clock_m + code_bias_m;
//! #         let ambiguity_m = ambiguities_m.get(&satellite_id).copied().unwrap();
//! #         Ok(FloatObservation {
//! #             sat: *id,
//! #             satellite_id: satellite_id.clone(),
//! #             ambiguity_id: satellite_id,
//! #             code_m,
//! #             phase_m: prediction.geometric_range_m + clock_m + ambiguity_m,
//! #             freq1_hz: 0.0,
//! #             freq2_hz: 0.0,
//! #         })
//! #     })
//! #     .collect::<Result<Vec<_>, ObservablesError>>()?;
//! # let initial_ambiguities = observations
//! #     .iter()
//! #     .map(|obs| (obs.ambiguity_id.clone(), obs.phase_m - obs.code_m))
//! #     .collect();
//! # let epoch = FloatEpoch {
//! #     epoch: CivilDateTime {
//! #         year: 2020,
//! #         month: 6,
//! #         day: 24,
//! #         hour: 12,
//! #         minute: 0,
//! #         second: 0.0,
//! #     },
//! #     jd_whole: 2_459_024.5,
//! #     jd_fraction: 0.5,
//! #     t_rx_j2000_s: 0.0,
//! #     observations,
//! # };
//! # let initial_state = FloatState {
//! #     position_m: [truth[0] + 80.0, truth[1] - 60.0, truth[2] + 40.0],
//! #     clocks_m: vec![0.0],
//! #     ambiguities_m: initial_ambiguities,
//! #     ztd_m: 0.0,
//! # };
//! # let solve_config = FloatSolveConfig {
//! #     weights: MeasurementWeights {
//! #         code: 1.0,
//! #         phase: 1.0,
//! #         elevation_weighting: false,
//! #     },
//! #     tropo: TroposphereOptions::disabled(),
//! #     corrections: RangeCorrections::disabled(),
//! #     opts: FloatSolveOptions {
//! #         max_iterations: 50,
//! #         position_tolerance_m: 1.0e-7,
//! #         clock_tolerance_m: 1.0e-7,
//! #         ambiguity_tolerance_m: 1.0e-7,
//! #         ztd_tolerance_m: 1.0e-7,
//! #     },
//! #     residual_screen: false,
//! # };
//! let result = solve_float_epoch_with_raim(
//! #   &source,
//! #   epoch,
//! #   initial_state,
//! #   solve_config,
//!     RaimConfig {
//!         chi_square_threshold: Some(10.0),
//!         ..RaimConfig::default()
//!     },
//! )?;
//! assert_eq!(result.excluded_sats, vec!["G05".to_string()]);
//! # Ok(())
//! # }
//! ```

use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use crate::astro::math::linear::{invert_matrix_last_tie, invert_symmetric_pd};
use crate::constants::F_L1_HZ;
use crate::estimation::substrate::parameters::undifferenced_design_row;
use crate::geometry::{dop, DopError, LineOfSight, Wgs84Geodetic};
use crate::observables::{predict, ObservableEphemerisSource, PredictOptions};
use crate::quality::{chi2_inv, DEFAULT_P_FA};
use crate::validate;

use super::{
    no_ephemeris, solve_float_epoch, state_from_solution, ztd_unknown_count, FloatEpoch,
    FloatResidual, FloatSolution, FloatSolveConfig, FloatSolveError, FloatState,
    TroposphereOptions,
};

const DEFAULT_MISSED_DETECTION_PROBABILITY: f64 = 1.0e-3;
const DEFAULT_MEASUREMENT_SIGMA_M: f64 = 1.0;
const RESIDUAL_COMPONENTS_PER_ROW: usize = 2;
const SNAPSHOT_BASE_STATES: usize = 4;
const LEVERAGE_TOLERANCE: f64 = 1.0e-12;
const HIGH_LEVERAGE_RESIDUAL_ZERO_TOLERANCE: f64 = 1.0e-8;
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Configuration for PPP snapshot RAIM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaimConfig {
    /// False-alarm probability used to derive the chi-square threshold.
    pub false_alarm_probability: f64,
    /// Missed-detection probability reserved for protection-level scaling.
    pub missed_detection_probability: f64,
    /// Scalar residual sigma, meters, applied after each residual row's solver weight.
    pub measurement_sigma_m: f64,
    /// Optional fixed chi-square threshold. When present, it overrides
    /// [`false_alarm_probability`](Self::false_alarm_probability).
    pub chi_square_threshold: Option<f64>,
}

impl Default for RaimConfig {
    fn default() -> Self {
        Self {
            false_alarm_probability: DEFAULT_P_FA,
            missed_detection_probability: DEFAULT_MISSED_DETECTION_PROBABILITY,
            measurement_sigma_m: DEFAULT_MEASUREMENT_SIGMA_M,
            chi_square_threshold: None,
        }
    }
}

/// Status returned by a PPP RAIM global test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaimStatus {
    /// The geometry had enough redundancy and the global test passed.
    Passed,
    /// The global test statistic exceeded the chi-square threshold.
    FaultDetected,
    /// The residual set had zero or negative redundancy, so no test was run.
    NotEnoughRedundancy,
}

/// Result of a PPP RAIM global fault-detection test.
#[derive(Debug, Clone, PartialEq)]
pub struct RaimResult {
    /// Summary status for the test.
    pub status: RaimStatus,
    /// True when the test statistic exceeds the chi-square threshold.
    pub detected: bool,
    /// Sum of squared weighted residuals.
    pub test_statistic: f64,
    /// Chi-square threshold used by the test, absent when redundancy is not positive.
    pub threshold: Option<f64>,
    /// Redundancy of the residual set, `n_obs - n_states`.
    pub redundancy: isize,
    /// Per-satellite standardized residual statistics.
    pub satellite_statistics: Vec<SatelliteTestStatistic>,
    /// Satellite with the largest standardized residual statistic.
    pub most_likely_fault: Option<String>,
    /// Horizontal protection level, meters, when geometry is available.
    pub hpl_m: Option<f64>,
    /// Vertical protection level, meters, when geometry is available.
    pub vpl_m: Option<f64>,
}

/// Geometry row used by PPP snapshot RAIM.
#[derive(Debug, Clone, PartialEq)]
pub struct RaimGeometryRow {
    /// Satellite token, matching [`FloatResidual::satellite_id`].
    pub satellite_id: String,
    /// Receiver-to-satellite line of sight in ECEF.
    pub line_of_sight: LineOfSight,
}

/// Per-satellite standardized residual statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct SatelliteTestStatistic {
    /// Satellite token.
    pub satellite_id: String,
    /// Standardized absolute code residual.
    pub code: f64,
    /// Standardized absolute phase residual.
    pub phase: f64,
    /// Satellite statistic, currently `max(code, phase)`.
    pub statistic: f64,
}

/// Result of per-satellite RAIM residual identification.
#[derive(Debug, Clone, PartialEq)]
pub struct RaimIdentification {
    /// Standardized residual statistic for each satellite.
    pub statistics: Vec<SatelliteTestStatistic>,
    /// Satellite with the largest statistic.
    pub most_likely_fault: Option<String>,
}

/// Horizontal and vertical protection levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProtectionLevels {
    /// Horizontal protection level, meters.
    pub hpl_m: f64,
    /// Vertical protection level, meters.
    pub vpl_m: f64,
}

/// Terminal status of a PPP RAIM/FDE loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaimFdeStatus {
    /// The first solve passed RAIM without exclusions.
    Clean,
    /// At least one satellite was excluded and the final solve passed RAIM.
    Restored,
    /// A fault was detected, but excluding the candidate would exhaust redundancy.
    CannotExclude,
    /// A fault was detected, but no valid exclusion restored integrity.
    IntegrityNotRestored,
}

/// Result of a PPP RAIM/FDE exclusion loop.
#[derive(Debug, Clone, PartialEq)]
pub struct RaimFdeResult {
    /// Final attempted float solution.
    pub solution: FloatSolution,
    /// RAIM result for the final attempted solution.
    pub raim: RaimResult,
    /// Satellites excluded in exclusion order.
    pub excluded_sats: Vec<String>,
    /// Terminal FDE status.
    pub status: RaimFdeStatus,
}

/// Error returned by a PPP RAIM/FDE exclusion loop.
#[derive(Debug, Clone, PartialEq)]
pub enum RaimFdeError {
    /// The underlying float solve or geometry prediction failed.
    Solve(FloatSolveError),
    /// RAIM configuration, residuals, or geometry were invalid.
    Raim(RaimError),
}

impl core::fmt::Display for RaimFdeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Solve(error) => write!(f, "PPP FDE solve failed: {error}"),
            Self::Raim(error) => write!(f, "PPP FDE RAIM check failed: {error}"),
        }
    }
}

impl std::error::Error for RaimFdeError {}

/// Error returned when PPP RAIM inputs are invalid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RaimError {
    /// A configuration field was outside its valid range.
    InvalidConfig {
        /// Name of the invalid field.
        field: &'static str,
        /// Short reason for the failure.
        reason: &'static str,
    },
    /// A residual row had a non-finite residual or non-positive weight.
    InvalidResidual {
        /// Name of the invalid residual field.
        field: &'static str,
    },
    /// Geometry rows were malformed or inconsistent with residual rows.
    InvalidGeometry {
        /// Name of the invalid geometry field.
        field: &'static str,
        /// Short reason for the failure.
        reason: &'static str,
    },
    /// DOP geometry could not produce finite protection-level scaling.
    Dop(DopError),
    /// The geometry normal matrix could not be inverted.
    SingularGeometry,
}

impl core::fmt::Display for RaimError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidConfig { field, reason } => {
                write!(f, "invalid RAIM configuration {field}: {reason}")
            }
            Self::InvalidResidual { field } => write!(f, "invalid RAIM residual {field}"),
            Self::InvalidGeometry { field, reason } => {
                write!(f, "invalid RAIM geometry {field}: {reason}")
            }
            Self::Dop(error) => write!(f, "RAIM DOP failed: {error}"),
            Self::SingularGeometry => write!(f, "singular RAIM geometry"),
        }
    }
}

impl std::error::Error for RaimError {}

/// Run the PPP RAIM global chi-square test over post-fit float residual rows.
///
/// Each [`FloatResidual`] contributes one code residual and one phase residual,
/// so `n_obs` is twice `residuals.len()`. The residual row weights are treated as
/// inverse-sigma solver weights, then divided by
/// [`RaimConfig::measurement_sigma_m`] before accumulating the SSE.
pub fn global_test(
    residuals: &[FloatResidual],
    n_states: usize,
    config: RaimConfig,
) -> Result<RaimResult, RaimError> {
    validate_config(config)?;
    let test_statistic = weighted_sse(residuals, config.measurement_sigma_m)?;
    let n_obs = residuals
        .len()
        .checked_mul(RESIDUAL_COMPONENTS_PER_ROW)
        .ok_or(RaimError::InvalidConfig {
            field: "residuals",
            reason: "too many residual rows",
        })?;
    let redundancy = n_obs as isize - n_states as isize;

    if redundancy <= 0 {
        return Ok(RaimResult {
            status: RaimStatus::NotEnoughRedundancy,
            detected: false,
            test_statistic,
            threshold: None,
            redundancy,
            satellite_statistics: Vec::new(),
            most_likely_fault: None,
            hpl_m: None,
            vpl_m: None,
        });
    }

    let threshold =
        match config.chi_square_threshold {
            Some(threshold) => threshold,
            None => chi2_inv(1.0 - config.false_alarm_probability, redundancy as usize).map_err(
                |_| RaimError::InvalidConfig {
                    field: "false_alarm_probability",
                    reason: "cannot derive chi-square threshold",
                },
            )?,
        };
    validate::finite_positive(threshold, "chi_square_threshold").map_err(|_| {
        RaimError::InvalidConfig {
            field: "chi_square_threshold",
            reason: "must be positive and finite",
        }
    })?;
    let detected = test_statistic > threshold;
    Ok(RaimResult {
        status: if detected {
            RaimStatus::FaultDetected
        } else {
            RaimStatus::Passed
        },
        detected,
        test_statistic,
        threshold: Some(threshold),
        redundancy,
        satellite_statistics: Vec::new(),
        most_likely_fault: None,
        hpl_m: None,
        vpl_m: None,
    })
}

/// Run the PPP RAIM global test and per-satellite residual identification.
///
/// This snapshot geometry uses the state vector `[x, y, z, clock,
/// ambiguity_0..ambiguity_n]`: code rows carry position/clock columns and phase
/// rows carry position/clock plus their satellite's ambiguity column.
pub fn global_test_with_geometry(
    residuals: &[FloatResidual],
    geometry: &[RaimGeometryRow],
    config: RaimConfig,
) -> Result<RaimResult, RaimError> {
    let n_states = snapshot_state_count_for_geometry(geometry)?;
    let mut result = global_test(residuals, n_states, config)?;
    if result.status == RaimStatus::NotEnoughRedundancy {
        return Ok(result);
    }
    let identification = per_satellite_statistics(residuals, geometry, config)?;
    result.satellite_statistics = identification.statistics;
    result.most_likely_fault = identification.most_likely_fault;
    Ok(result)
}

/// Compute per-satellite standardized residual statistics for PPP RAIM.
///
/// The returned statistic is the larger of the code and phase standardized
/// residual magnitudes for each satellite using the snapshot PPP
/// `[x, y, z, clock, ambiguity_0..ambiguity_n]` design. The `most_likely_fault`
/// is the satellite with the largest statistic.
pub fn per_satellite_statistics(
    residuals: &[FloatResidual],
    geometry: &[RaimGeometryRow],
    config: RaimConfig,
) -> Result<RaimIdentification, RaimError> {
    per_satellite_statistics_with_ztd(residuals, geometry, None, config)
}

fn per_satellite_statistics_with_ztd(
    residuals: &[FloatResidual],
    geometry: &[RaimGeometryRow],
    ztd_mappings: Option<&[f64]>,
    config: RaimConfig,
) -> Result<RaimIdentification, RaimError> {
    validate_config(config)?;
    validate_geometry(residuals, geometry)?;
    if let Some(mappings) = ztd_mappings {
        validate_ztd_mappings(residuals.len(), mappings)?;
    }
    let rows = snapshot_design_rows(
        residuals,
        geometry,
        ztd_mappings,
        config.measurement_sigma_m,
    )?;
    let normal = normal_matrix(&rows)?;
    let q = invert_matrix_last_tie(&normal).ok_or(RaimError::SingularGeometry)?;

    let mut statistics = Vec::with_capacity(residuals.len());
    let mut most_likely_fault = None;
    let mut worst = f64::NEG_INFINITY;
    for (idx, residual) in residuals.iter().enumerate() {
        let code = standardized_abs(&rows[2 * idx], &q)?;
        let phase = standardized_abs(&rows[2 * idx + 1], &q)?;
        let statistic = code.max(phase);
        if !code.is_finite() || !phase.is_finite() || !statistic.is_finite() {
            return Err(RaimError::SingularGeometry);
        }
        if statistic > worst {
            worst = statistic;
            most_likely_fault = Some(residual.satellite_id.clone());
        }
        statistics.push(SatelliteTestStatistic {
            satellite_id: residual.satellite_id.clone(),
            code,
            phase,
            statistic,
        });
    }

    Ok(RaimIdentification {
        statistics,
        most_likely_fault,
    })
}

/// Compute geometry-based PPP RAIM protection levels.
///
/// This uses the DOP horizontal and vertical geometry factors as slope terms and
/// scales them by `measurement_sigma_m * sqrt(chi2_inv(1 - p_md, 1))`, where
/// `p_md` is [`RaimConfig::missed_detection_probability`].
pub fn protection_levels(
    geometry: &[RaimGeometryRow],
    receiver: Wgs84Geodetic,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    validate_config(config)?;
    validate_geometry_only(geometry)?;
    let los = geometry
        .iter()
        .map(|row| row.line_of_sight)
        .collect::<Vec<_>>();
    let weights = vec![1.0; los.len()];
    let d = dop(&los, &weights, receiver).map_err(RaimError::Dop)?;
    let k_md = missed_detection_multiplier(config)?;
    let hpl_m = k_md * config.measurement_sigma_m * d.hdop;
    let vpl_m = k_md * config.measurement_sigma_m * d.vdop;
    if hpl_m.is_finite() && hpl_m > 0.0 && vpl_m.is_finite() && vpl_m > 0.0 {
        Ok(ProtectionLevels { hpl_m, vpl_m })
    } else {
        Err(RaimError::SingularGeometry)
    }
}

fn protection_levels_with_ztd(
    geometry: &[RaimGeometryRow],
    ztd_mappings: Option<&[f64]>,
    receiver: Wgs84Geodetic,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    let Some(ztd_mappings) = ztd_mappings else {
        return protection_levels(geometry, receiver, config);
    };

    validate_config(config)?;
    validate_geometry_only(geometry)?;
    validate_ztd_mappings(geometry.len(), ztd_mappings)?;

    let q = protection_position_covariance_with_ztd(geometry, ztd_mappings)?;
    let enu = rotate_position_covariance_to_enu(&q, receiver);
    scaled_protection_levels(enu[0][0] + enu[1][1], enu[2][2], config)
}

/// Run PPP snapshot fault detection and exclusion for one float epoch.
///
/// The loop solves the current observation set, runs the global RAIM test with
/// per-satellite identification, excludes the most likely faulty satellite, and
/// repeats until RAIM passes or exclusion would leave no positive redundancy.
pub fn fde_float_epoch(
    source: &dyn ObservableEphemerisSource,
    epoch: FloatEpoch,
    initial_state: FloatState,
    solve_config: FloatSolveConfig,
    raim_config: RaimConfig,
) -> Result<RaimFdeResult, RaimFdeError> {
    let mut current_epoch = epoch;
    let mut seed_state = initial_state;
    let mut excluded_sats = Vec::new();

    loop {
        let solution = solve_float_epoch(
            source,
            current_epoch.clone(),
            seed_state.clone(),
            solve_config.clone(),
        )
        .map_err(RaimFdeError::Solve)?;
        let raim = raim_for_solution(
            source,
            &current_epoch,
            &solution,
            solve_config.tropo,
            raim_config,
        )
        .map_err(RaimFdeError::Raim)?;

        if raim.status == RaimStatus::NotEnoughRedundancy {
            return Ok(RaimFdeResult {
                solution,
                raim,
                excluded_sats,
                status: RaimFdeStatus::CannotExclude,
            });
        }

        if !raim.detected {
            return Ok(RaimFdeResult {
                solution,
                raim,
                status: if excluded_sats.is_empty() {
                    RaimFdeStatus::Clean
                } else {
                    RaimFdeStatus::Restored
                },
                excluded_sats,
            });
        }

        let Some(candidate) = raim.most_likely_fault.clone() else {
            return Ok(RaimFdeResult {
                solution,
                raim,
                excluded_sats,
                status: RaimFdeStatus::IntegrityNotRestored,
            });
        };
        let Some(next_epoch) = exclude_satellite(&current_epoch, &candidate) else {
            return Ok(RaimFdeResult {
                solution,
                raim,
                excluded_sats,
                status: RaimFdeStatus::IntegrityNotRestored,
            });
        };
        if !has_positive_redundancy_after_exclusion(
            next_epoch.observations.len(),
            solve_config.tropo,
        )
        .map_err(RaimFdeError::Raim)?
        {
            return Ok(RaimFdeResult {
                solution,
                raim,
                excluded_sats,
                status: RaimFdeStatus::CannotExclude,
            });
        }

        excluded_sats.push(candidate);
        seed_state = state_from_solution(&solution, &seed_state);
        current_epoch = next_epoch;
    }
}

/// Solve one PPP float epoch and run RAIM/FDE on the result.
///
/// This is the public driver for snapshot PPP integrity: it preserves the
/// existing float-solve behavior, then performs RAIM fault detection and, when
/// needed, fault detection and exclusion over reduced observation sets.
pub fn solve_float_epoch_with_raim(
    source: &dyn ObservableEphemerisSource,
    epoch: FloatEpoch,
    initial_state: FloatState,
    solve_config: FloatSolveConfig,
    raim_config: RaimConfig,
) -> Result<RaimFdeResult, RaimFdeError> {
    fde_float_epoch(source, epoch, initial_state, solve_config, raim_config)
}

fn validate_config(config: RaimConfig) -> Result<(), RaimError> {
    validate_probability(config.false_alarm_probability, "false_alarm_probability")?;
    validate_probability(
        config.missed_detection_probability,
        "missed_detection_probability",
    )?;
    validate::finite_positive(config.measurement_sigma_m, "measurement_sigma_m")
        .map(|_| ())
        .map_err(|_| RaimError::InvalidConfig {
            field: "measurement_sigma_m",
            reason: "must be positive and finite",
        })?;
    if let Some(threshold) = config.chi_square_threshold {
        validate::finite_positive(threshold, "chi_square_threshold")
            .map(|_| ())
            .map_err(|_| RaimError::InvalidConfig {
                field: "chi_square_threshold",
                reason: "must be positive and finite",
            })?;
    }
    Ok(())
}

fn validate_probability(value: f64, field: &'static str) -> Result<(), RaimError> {
    let value = validate::finite(value, field).map_err(|_| RaimError::InvalidConfig {
        field,
        reason: "must be finite",
    })?;
    if value > 0.0 && value < 1.0 {
        Ok(())
    } else {
        Err(RaimError::InvalidConfig {
            field,
            reason: "must be inside (0, 1)",
        })
    }
}

fn weighted_sse(residuals: &[FloatResidual], measurement_sigma_m: f64) -> Result<f64, RaimError> {
    let mut sse = 0.0;
    for row in residuals {
        let code_m = validate_residual(row.code_m, "code_m")?;
        let phase_m = validate_residual(row.phase_m, "phase_m")?;
        let code_weight = validate_weight(row.code_weight, "code_weight")?;
        let phase_weight = validate_weight(row.phase_weight, "phase_weight")?;
        let code = code_m * code_weight / measurement_sigma_m;
        let phase = phase_m * phase_weight / measurement_sigma_m;
        if !code.is_finite() || !phase.is_finite() {
            return Err(RaimError::InvalidResidual {
                field: "weighted_residual",
            });
        }
        sse += code * code + phase * phase;
        if !sse.is_finite() {
            return Err(RaimError::InvalidResidual {
                field: "weighted_sse",
            });
        }
    }
    Ok(sse)
}

fn validate_residual(value: f64, field: &'static str) -> Result<f64, RaimError> {
    validate::finite(value, field).map_err(|_| RaimError::InvalidResidual { field })
}

fn validate_weight(value: f64, field: &'static str) -> Result<f64, RaimError> {
    validate::finite_positive(value, field).map_err(|_| RaimError::InvalidResidual { field })
}

fn missed_detection_multiplier(config: RaimConfig) -> Result<f64, RaimError> {
    chi2_inv(1.0 - config.missed_detection_probability, 1)
        .map(f64::sqrt)
        .map_err(|_| RaimError::InvalidConfig {
            field: "missed_detection_probability",
            reason: "cannot derive missed-detection multiplier",
        })
}

fn raim_for_solution(
    source: &dyn ObservableEphemerisSource,
    epoch: &FloatEpoch,
    solution: &FloatSolution,
    tropo: TroposphereOptions,
    config: RaimConfig,
) -> Result<RaimResult, RaimError> {
    let geometry = geometry_for_solution(source, epoch, solution, tropo).map_err(|_| {
        RaimError::InvalidGeometry {
            field: "solution",
            reason: "could not build line-of-sight geometry",
        }
    })?;
    validate_geometry(&solution.residuals_m, &geometry.rows)?;
    let n_states = snapshot_state_count_for_solution(solution)?;
    let mut result = global_test(&solution.residuals_m, n_states, config)?;
    if result.status != RaimStatus::NotEnoughRedundancy {
        let identification = per_satellite_statistics_with_ztd(
            &solution.residuals_m,
            &geometry.rows,
            geometry.ztd_mappings.as_deref(),
            config,
        )?;
        result.satellite_statistics = identification.statistics;
        result.most_likely_fault = identification.most_likely_fault;
    }
    if result.status != RaimStatus::NotEnoughRedundancy {
        let receiver = receiver_geodetic(solution.position_m);
        let levels = protection_levels_with_ztd(
            &geometry.rows,
            geometry.ztd_mappings.as_deref(),
            receiver,
            config,
        )?;
        result.hpl_m = Some(levels.hpl_m);
        result.vpl_m = Some(levels.vpl_m);
    }
    Ok(result)
}

#[derive(Debug, Clone)]
struct RaimSolutionGeometry {
    rows: Vec<RaimGeometryRow>,
    ztd_mappings: Option<Vec<f64>>,
}

fn geometry_for_solution(
    source: &dyn ObservableEphemerisSource,
    epoch: &FloatEpoch,
    solution: &FloatSolution,
    tropo: TroposphereOptions,
) -> Result<RaimSolutionGeometry, FloatSolveError> {
    let mut rows = Vec::with_capacity(solution.residuals_m.len());
    let mut ztd_mappings = solution
        .ztd_residual_m
        .is_some()
        .then(|| Vec::with_capacity(solution.residuals_m.len()));
    for residual in &solution.residuals_m {
        let obs = epoch
            .observations
            .iter()
            .find(|obs| obs.satellite_id == residual.satellite_id)
            .ok_or(FloatSolveError::InvalidInput {
                field: "raim geometry satellite_id",
                reason: "residual satellite missing from epoch",
            })?;
        let pred = predict(
            source,
            obs.sat,
            solution.position_m,
            epoch.t_rx_j2000_s,
            PredictOptions {
                carrier_hz: F_L1_HZ,
                light_time: true,
                sagnac: true,
            },
        )
        .map_err(|error| no_ephemeris(obs, error))?;
        validate::finite_vec3(pred.los_unit, "raim geometry los_unit").map_err(|error| {
            FloatSolveError::InvalidInput {
                field: error.field(),
                reason: error.reason(),
            }
        })?;
        if let Some(mappings) = &mut ztd_mappings {
            let tropo_model =
                super::model::model_troposphere(&pred, solution.position_m, epoch, tropo)?;
            validate::finite(tropo_model.ztd_mapping, "raim geometry ztd_mapping").map_err(
                |error| FloatSolveError::InvalidInput {
                    field: error.field(),
                    reason: error.reason(),
                },
            )?;
            mappings.push(tropo_model.ztd_mapping);
        }
        rows.push(RaimGeometryRow {
            satellite_id: residual.satellite_id.clone(),
            line_of_sight: LineOfSight::new(pred.los_unit[0], pred.los_unit[1], pred.los_unit[2]),
        });
    }
    Ok(RaimSolutionGeometry { rows, ztd_mappings })
}

fn exclude_satellite(epoch: &FloatEpoch, satellite_id: &str) -> Option<FloatEpoch> {
    let mut next = epoch.clone();
    let before = next.observations.len();
    next.observations
        .retain(|obs| obs.satellite_id != satellite_id);
    (next.observations.len() < before).then_some(next)
}

fn has_positive_redundancy_after_exclusion(
    n_sats_after: usize,
    tropo: super::TroposphereOptions,
) -> Result<bool, RaimError> {
    let n_obs = n_sats_after * RESIDUAL_COMPONENTS_PER_ROW;
    let n_states = snapshot_state_count_from_parts(1, ztd_unknown_count(tropo), n_sats_after)?;
    Ok(n_obs > n_states)
}

fn receiver_geodetic(position_m: [f64; 3]) -> Wgs84Geodetic {
    let (lat_deg, lon_deg, height_km) = itrs_to_geodetic_compute(
        position_m[0] / 1000.0,
        position_m[1] / 1000.0,
        position_m[2] / 1000.0,
    )
    .expect("valid receiver ITRS coordinates");
    Wgs84Geodetic::new(
        lat_deg * DEG_TO_RAD,
        lon_deg * DEG_TO_RAD,
        height_km * 1000.0,
    )
    .expect("valid receiver geodetic coordinates")
}

fn snapshot_state_count_for_geometry(geometry: &[RaimGeometryRow]) -> Result<usize, RaimError> {
    snapshot_state_count_from_parts(1, 0, geometry.len())
}

fn snapshot_state_count_for_solution(solution: &FloatSolution) -> Result<usize, RaimError> {
    snapshot_state_count_from_parts(
        solution.epoch_clocks_m.len(),
        usize::from(solution.ztd_residual_m.is_some()),
        solution.ambiguities_m.len(),
    )
}

fn snapshot_state_count_from_parts(
    n_clocks: usize,
    n_ztd: usize,
    n_ambiguities: usize,
) -> Result<usize, RaimError> {
    SNAPSHOT_BASE_STATES
        .checked_add(n_clocks.saturating_sub(1))
        .and_then(|count| count.checked_add(n_ztd))
        .and_then(|count| count.checked_add(n_ambiguities))
        .ok_or(RaimError::InvalidGeometry {
            field: "geometry",
            reason: "too many rows",
        })
}

fn validate_geometry(
    residuals: &[FloatResidual],
    geometry: &[RaimGeometryRow],
) -> Result<(), RaimError> {
    if residuals.len() != geometry.len() {
        return Err(RaimError::InvalidGeometry {
            field: "geometry",
            reason: "length must match residuals",
        });
    }
    for (residual, row) in residuals.iter().zip(geometry) {
        if residual.satellite_id != row.satellite_id {
            return Err(RaimError::InvalidGeometry {
                field: "satellite_id",
                reason: "order must match residuals",
            });
        }
        validate_geometry_row(row)?;
    }
    Ok(())
}

fn validate_geometry_only(geometry: &[RaimGeometryRow]) -> Result<(), RaimError> {
    for row in geometry {
        validate_geometry_row(row)?;
    }
    Ok(())
}

fn validate_geometry_row(row: &RaimGeometryRow) -> Result<(), RaimError> {
    validate::finite(row.line_of_sight.e_x, "line_of_sight.e_x").map_err(|_| {
        RaimError::InvalidGeometry {
            field: "line_of_sight.e_x",
            reason: "must be finite",
        }
    })?;
    validate::finite(row.line_of_sight.e_y, "line_of_sight.e_y").map_err(|_| {
        RaimError::InvalidGeometry {
            field: "line_of_sight.e_y",
            reason: "must be finite",
        }
    })?;
    validate::finite(row.line_of_sight.e_z, "line_of_sight.e_z").map_err(|_| {
        RaimError::InvalidGeometry {
            field: "line_of_sight.e_z",
            reason: "must be finite",
        }
    })?;
    Ok(())
}

fn validate_ztd_mappings(expected_len: usize, mappings: &[f64]) -> Result<(), RaimError> {
    if mappings.len() != expected_len {
        return Err(RaimError::InvalidGeometry {
            field: "ztd_mapping",
            reason: "length must match geometry",
        });
    }
    for &mapping in mappings {
        validate::finite(mapping, "ztd_mapping").map_err(|_| RaimError::InvalidGeometry {
            field: "ztd_mapping",
            reason: "must be finite",
        })?;
    }
    Ok(())
}

fn protection_position_covariance_with_ztd(
    geometry: &[RaimGeometryRow],
    ztd_mappings: &[f64],
) -> Result<[[f64; 3]; 3], RaimError> {
    const PROTECTION_STATES_WITH_ZTD: usize = SNAPSHOT_BASE_STATES + 1;
    let mut normal = vec![vec![0.0_f64; PROTECTION_STATES_WITH_ZTD]; PROTECTION_STATES_WITH_ZTD];

    for (row, &ztd_mapping) in geometry.iter().zip(ztd_mappings) {
        let h = [
            -row.line_of_sight.e_x,
            -row.line_of_sight.e_y,
            -row.line_of_sight.e_z,
            1.0,
            ztd_mapping,
        ];
        for (i, normal_row) in normal.iter_mut().enumerate() {
            let h_i = h[i];
            for (j, normal_ij) in normal_row.iter_mut().enumerate() {
                *normal_ij += h_i * h[j];
            }
        }
    }

    let q = invert_symmetric_pd(&normal).ok_or(RaimError::SingularGeometry)?;
    Ok([
        [q[0][0], q[0][1], q[0][2]],
        [q[1][0], q[1][1], q[1][2]],
        [q[2][0], q[2][1], q[2][2]],
    ])
}

#[allow(clippy::needless_range_loop)]
fn rotate_position_covariance_to_enu(q: &[[f64; 3]; 3], receiver: Wgs84Geodetic) -> [[f64; 3]; 3] {
    let sphi = receiver.lat_rad.sin();
    let cphi = receiver.lat_rad.cos();
    let slam = receiver.lon_rad.sin();
    let clam = receiver.lon_rad.cos();
    let r = [
        [-slam, clam, 0.0],
        [-sphi * clam, -sphi * slam, cphi],
        [cphi * clam, cphi * slam, sphi],
    ];

    let mut rq = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += r[i][k] * q[k][j];
            }
            rq[i][j] = s;
        }
    }
    let mut enu = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += rq[i][k] * r[j][k];
            }
            enu[i][j] = s;
        }
    }
    enu
}

fn scaled_protection_levels(
    hdop_arg: f64,
    vdop_arg: f64,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    for arg in [hdop_arg, vdop_arg] {
        #[allow(clippy::neg_cmp_op_on_partial_ord)]
        let negative_or_nan = !(arg >= 0.0);
        if negative_or_nan || !arg.is_finite() {
            return Err(RaimError::SingularGeometry);
        }
    }

    let k_md = missed_detection_multiplier(config)?;
    let hpl_m = k_md * config.measurement_sigma_m * hdop_arg.sqrt();
    let vpl_m = k_md * config.measurement_sigma_m * vdop_arg.sqrt();
    if hpl_m.is_finite() && hpl_m > 0.0 && vpl_m.is_finite() && vpl_m > 0.0 {
        Ok(ProtectionLevels { hpl_m, vpl_m })
    } else {
        Err(RaimError::SingularGeometry)
    }
}

#[derive(Debug, Clone)]
struct StandardizationRow {
    h: Vec<f64>,
    residual: f64,
}

fn snapshot_design_rows(
    residuals: &[FloatResidual],
    geometry: &[RaimGeometryRow],
    ztd_mappings: Option<&[f64]>,
    measurement_sigma_m: f64,
) -> Result<Vec<StandardizationRow>, RaimError> {
    let mut rows = Vec::with_capacity(residuals.len() * RESIDUAL_COMPONENTS_PER_ROW);
    let n_ambiguities = residuals.len();
    for (ambiguity_idx, (residual, geometry)) in residuals.iter().zip(geometry).enumerate() {
        let code_residual = validate_residual(residual.code_m, "code_m")?;
        let phase_residual = validate_residual(residual.phase_m, "phase_m")?;
        let code_weight =
            validate_weight(residual.code_weight, "code_weight")? / measurement_sigma_m;
        let phase_weight =
            validate_weight(residual.phase_weight, "phase_weight")? / measurement_sigma_m;
        if !code_weight.is_finite() || !phase_weight.is_finite() {
            return Err(RaimError::InvalidResidual {
                field: "weighted_residual",
            });
        }
        let ztd_mapping = ztd_mappings.map(|mappings| mappings[ambiguity_idx]);
        let code = code_residual * code_weight;
        let phase = phase_residual * phase_weight;
        if !code.is_finite() || !phase.is_finite() {
            return Err(RaimError::InvalidResidual {
                field: "weighted_residual",
            });
        }
        rows.push(StandardizationRow {
            h: weighted_snapshot_row(
                geometry.line_of_sight,
                code_weight,
                ztd_mapping,
                n_ambiguities,
                None,
            ),
            residual: code,
        });
        rows.push(StandardizationRow {
            h: weighted_snapshot_row(
                geometry.line_of_sight,
                phase_weight,
                ztd_mapping,
                n_ambiguities,
                Some(ambiguity_idx),
            ),
            residual: phase,
        });
    }
    Ok(rows)
}

fn weighted_snapshot_row(
    line_of_sight: LineOfSight,
    weight: f64,
    ztd_mapping: Option<f64>,
    n_ambiguities: usize,
    active_ambiguity: Option<usize>,
) -> Vec<f64> {
    let mut row = undifferenced_design_row(
        [-line_of_sight.e_x, -line_of_sight.e_y, -line_of_sight.e_z],
        0,
        1,
        ztd_mapping,
        n_ambiguities,
        active_ambiguity,
    );
    for value in &mut row {
        *value *= weight;
    }
    row
}

fn normal_matrix(rows: &[StandardizationRow]) -> Result<Vec<Vec<f64>>, RaimError> {
    let n = rows.first().map(|row| row.h.len()).unwrap_or(0);
    if n == 0 {
        return Err(RaimError::SingularGeometry);
    }
    let mut normal = vec![vec![0.0; n]; n];
    for row in rows {
        for (i, normal_row) in normal.iter_mut().enumerate() {
            let h_i = row.h[i];
            for (j, normal_ij) in normal_row.iter_mut().enumerate() {
                *normal_ij += h_i * row.h[j];
            }
        }
    }
    Ok(normal)
}

fn standardized_abs(row: &StandardizationRow, q: &[Vec<f64>]) -> Result<f64, RaimError> {
    let leverage = row_leverage(&row.h, q)?;
    let variance_factor = 1.0 - leverage;
    if !variance_factor.is_finite() {
        return Err(RaimError::SingularGeometry);
    }
    if variance_factor.abs() <= LEVERAGE_TOLERANCE {
        return if row.residual.abs() <= HIGH_LEVERAGE_RESIDUAL_ZERO_TOLERANCE {
            Ok(0.0)
        } else {
            Err(RaimError::SingularGeometry)
        };
    }
    if variance_factor < 0.0 {
        return Err(RaimError::SingularGeometry);
    }
    let standardized = row.residual.abs() / variance_factor.sqrt();
    if standardized.is_finite() {
        Ok(standardized)
    } else {
        Err(RaimError::SingularGeometry)
    }
}

fn row_leverage(row: &[f64], q: &[Vec<f64>]) -> Result<f64, RaimError> {
    if q.len() != row.len() || q.iter().any(|q_row| q_row.len() != row.len()) {
        return Err(RaimError::SingularGeometry);
    }
    let mut value = 0.0;
    for i in 0..row.len() {
        for j in 0..row.len() {
            value += row[i] * q[i][j] * row[j];
        }
    }
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RaimError::SingularGeometry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    use crate::geometry::line_of_sight_from_az_el_deg;
    use crate::observables::{ObservableState, ObservablesError};
    use crate::ppp_corrections::CivilDateTime;
    use crate::{GnssSatelliteId, GnssSystem};

    fn residual(satellite_id: &str, code_m: f64, phase_m: f64) -> FloatResidual {
        FloatResidual {
            epoch_index: 0,
            satellite_id: satellite_id.to_string(),
            code_m,
            phase_m,
            code_weight: 1.0,
            phase_weight: 1.0,
        }
    }

    #[derive(Debug, Clone)]
    struct TestSource {
        states: BTreeMap<GnssSatelliteId, [f64; 3]>,
    }

    impl ObservableEphemerisSource for TestSource {
        fn observable_state_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(ObservableState {
                position_ecef_m: *self.states.get(&sat).ok_or(ObservablesError::NoEphemeris)?,
                clock_s: Some(0.0),
            })
        }
    }

    fn geometry(satellite_id: &str, line_of_sight: LineOfSight) -> RaimGeometryRow {
        RaimGeometryRow {
            satellite_id: satellite_id.to_string(),
            line_of_sight,
        }
    }

    fn clean_residuals() -> Vec<FloatResidual> {
        vec![
            residual("G01", 0.1, -0.1),
            residual("G02", -0.1, 0.1),
            residual("G03", 0.05, -0.05),
            residual("G04", -0.05, 0.05),
        ]
    }

    fn test_geometry() -> Vec<RaimGeometryRow> {
        vec![
            geometry("G01", LineOfSight::new(1.0, 0.0, 0.0)),
            geometry("G02", LineOfSight::new(-1.0, 0.0, 0.0)),
            geometry("G03", LineOfSight::new(0.0, 1.0, 0.0)),
            geometry("G04", LineOfSight::new(0.0, 0.0, 1.0)),
            geometry(
                "G05",
                LineOfSight::new(
                    1.0 / 3.0_f64.sqrt(),
                    1.0 / 3.0_f64.sqrt(),
                    1.0 / 3.0_f64.sqrt(),
                ),
            ),
        ]
    }

    fn protection_receiver() -> Wgs84Geodetic {
        Wgs84Geodetic::new(0.7, -1.2, 0.0).expect("valid geodetic receiver")
    }

    fn protection_geometry(points: &[(f64, f64)]) -> Vec<RaimGeometryRow> {
        points
            .iter()
            .enumerate()
            .map(|(idx, &(azimuth_deg, elevation_deg))| {
                geometry(
                    &format!("G{:02}", idx + 1),
                    line_of_sight_from_az_el_deg(azimuth_deg, elevation_deg, protection_receiver())
                        .expect("valid protection geometry"),
                )
            })
            .collect()
    }

    fn fde_config() -> FloatSolveConfig {
        FloatSolveConfig {
            weights: super::super::MeasurementWeights {
                code: 1.0,
                phase: 1.0,
                elevation_weighting: false,
            },
            tropo: super::super::TroposphereOptions::disabled(),
            corrections: super::super::RangeCorrections::disabled(),
            opts: super::super::FloatSolveOptions {
                max_iterations: 50,
                position_tolerance_m: 1.0e-7,
                clock_tolerance_m: 1.0e-7,
                ambiguity_tolerance_m: 1.0e-7,
                ztd_tolerance_m: 1.0e-7,
            },
            residual_screen: false,
        }
    }

    fn fde_raim_config() -> RaimConfig {
        RaimConfig {
            chi_square_threshold: Some(10.0),
            ..RaimConfig::default()
        }
    }

    fn synthetic_case(
        n_sats: usize,
        biased_satellite: Option<&str>,
    ) -> (TestSource, FloatEpoch, FloatState) {
        let sat_positions = [
            (1u8, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
            (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
            (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
            (4, [-18_200_000.0, -16_000_000.0, 21_000_000.0]),
            (5, [22_000_000.0, -12_000_000.0, 20_200_000.0]),
            (6, [-12_000_000.0, 23_000_000.0, 18_000_000.0]),
        ];
        let ids = sat_positions
            .iter()
            .take(n_sats)
            .map(|(prn, _)| {
                GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id")
            })
            .collect::<Vec<_>>();
        let source = TestSource {
            states: ids
                .iter()
                .zip(sat_positions.iter())
                .map(|(id, (_, position))| (*id, *position))
                .collect(),
        };
        let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
        let clock_m = 12.5;
        let ambiguities_m = ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
            .collect::<BTreeMap<_, _>>();
        let observations = ids
            .iter()
            .map(|id| {
                let satellite_id = id.to_string();
                let prediction = predict(
                    &source,
                    *id,
                    truth,
                    0.0,
                    PredictOptions {
                        carrier_hz: F_L1_HZ,
                        light_time: true,
                        sagnac: true,
                    },
                )
                .expect("synthetic prediction");
                let bias = if Some(satellite_id.as_str()) == biased_satellite {
                    50.0
                } else {
                    0.0
                };
                let code_m = prediction.geometric_range_m + clock_m + bias;
                let ambiguity_m = ambiguities_m.get(&satellite_id).copied().unwrap();
                super::super::FloatObservation {
                    sat: *id,
                    satellite_id: satellite_id.clone(),
                    ambiguity_id: satellite_id,
                    code_m,
                    phase_m: prediction.geometric_range_m + clock_m + ambiguity_m,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                }
            })
            .collect::<Vec<_>>();
        let initial_ambiguities = observations
            .iter()
            .map(|obs| (obs.ambiguity_id.clone(), obs.phase_m - obs.code_m))
            .collect();
        let epoch = FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24,
                hour: 12,
                minute: 0,
                second: 0.0,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5,
            t_rx_j2000_s: 0.0,
            observations,
        };
        let state = FloatState {
            position_m: [truth[0] + 80.0, truth[1] - 60.0, truth[2] + 40.0],
            clocks_m: vec![0.0],
            ambiguities_m: initial_ambiguities,
            ztd_m: 0.0,
        };
        (source, epoch, state)
    }

    fn assert_position_close(actual: [f64; 3], expected: [f64; 3], tolerance_m: f64) {
        for idx in 0..3 {
            assert!(
                (actual[idx] - expected[idx]).abs() <= tolerance_m,
                "axis {idx}: got {}, expected {}",
                actual[idx],
                expected[idx]
            );
        }
    }

    #[test]
    fn clean_residuals_pass_global_test() {
        let result = global_test(&clean_residuals(), 7, RaimConfig::default()).unwrap();
        assert_eq!(result.status, RaimStatus::Passed);
        assert!(!result.detected);
        assert_eq!(result.redundancy, 1);
        assert!(result.threshold.expect("threshold") > result.test_statistic);
    }

    #[test]
    fn injected_bias_trips_global_test() {
        let mut residuals = clean_residuals();
        residuals[2].code_m = 5.0;

        let result = global_test(&residuals, 7, RaimConfig::default()).unwrap();
        assert_eq!(result.status, RaimStatus::FaultDetected);
        assert!(result.detected);
        assert_eq!(result.redundancy, 1);
        assert!(result.test_statistic > result.threshold.expect("threshold"));
    }

    #[test]
    fn global_test_rejects_overflowed_weighted_sse() {
        let mut residuals = clean_residuals();
        residuals[0].code_m = f64::MAX;

        assert_eq!(
            global_test(&residuals, 7, RaimConfig::default()),
            Err(RaimError::InvalidResidual {
                field: "weighted_sse",
            })
        );
    }

    #[test]
    fn raim_redundancy_counts_estimated_ztd_state() {
        let (source, epoch, state) = synthetic_case(6, None);
        let mut solve_config = fde_config();
        let mut tropo = super::super::TroposphereOptions::disabled();
        tropo.enabled = true;
        tropo.estimate_ztd = true;
        solve_config.tropo = tropo;
        let solve_tropo = solve_config.tropo;
        let solution = solve_float_epoch(&source, epoch.clone(), state, solve_config)
            .expect("ZTD-estimated solve");

        assert!(solution.ztd_residual_m.is_some());
        let raim = raim_for_solution(
            &source,
            &epoch,
            &solution,
            solve_tropo,
            RaimConfig::default(),
        )
        .expect("RAIM for ZTD-estimated solution");
        let expected_states = 3 + solution.epoch_clocks_m.len() + 1 + solution.ambiguities_m.len();
        let expected_redundancy = (solution.residuals_m.len() * RESIDUAL_COMPONENTS_PER_ROW)
            as isize
            - expected_states as isize;
        let expected = global_test(
            &solution.residuals_m,
            expected_states,
            RaimConfig::default(),
        )
        .expect("expected-dof global test");
        let without_ztd = global_test(
            &solution.residuals_m,
            expected_states - 1,
            RaimConfig::default(),
        )
        .expect("old-dof global test");

        assert_eq!(expected_states, 11);
        assert_eq!(expected_redundancy, 1);
        assert_eq!(raim.redundancy, expected_redundancy);
        assert_eq!(
            raim.threshold.map(f64::to_bits),
            expected.threshold.map(f64::to_bits)
        );
        assert_ne!(
            raim.threshold.map(f64::to_bits),
            without_ztd.threshold.map(f64::to_bits)
        );
    }

    #[test]
    fn ztd_identification_uses_estimated_state_projection() {
        let (source, epoch, state) = synthetic_case(6, Some("G02"));
        let mut solve_config = fde_config();
        let mut tropo = super::super::TroposphereOptions::disabled();
        tropo.enabled = true;
        tropo.estimate_ztd = true;
        solve_config.tropo = tropo;
        let solve_tropo = solve_config.tropo;
        let mut solution =
            solve_float_epoch(&source, epoch.clone(), state, solve_config).expect("solve");
        assert!(solution.ztd_residual_m.is_some());

        for residual in &mut solution.residuals_m {
            residual.code_m = if residual.satellite_id == "G02" {
                1.0
            } else if residual.satellite_id == "G05" {
                2.0
            } else {
                0.0
            };
            residual.phase_m = 0.0;
        }

        let geometry =
            geometry_for_solution(&source, &epoch, &solution, solve_tropo).expect("geometry");
        let without_ztd =
            per_satellite_statistics(&solution.residuals_m, &geometry.rows, RaimConfig::default())
                .expect("without ztd");
        assert_eq!(without_ztd.most_likely_fault.as_deref(), Some("G05"));

        let raim = raim_for_solution(
            &source,
            &epoch,
            &solution,
            solve_tropo,
            RaimConfig::default(),
        )
        .expect("RAIM with ZTD projection");
        assert_eq!(raim.most_likely_fault.as_deref(), Some("G02"));
        let g02 = raim
            .satellite_statistics
            .iter()
            .find(|stat| stat.satellite_id == "G02")
            .expect("G02 statistic");
        let g05 = raim
            .satellite_statistics
            .iter()
            .find(|stat| stat.satellite_id == "G05")
            .expect("G05 statistic");
        assert!(g02.statistic > 10.0 * g05.statistic);
    }

    #[test]
    fn nonpositive_redundancy_returns_status() {
        let residuals = vec![residual("G01", 100.0, 0.0), residual("G02", 0.0, 0.0)];

        let zero = global_test(&residuals, 4, RaimConfig::default()).unwrap();
        assert_eq!(zero.status, RaimStatus::NotEnoughRedundancy);
        assert!(!zero.detected);
        assert_eq!(zero.threshold, None);
        assert_eq!(zero.redundancy, 0);

        let negative = global_test(&residuals, 5, RaimConfig::default()).unwrap();
        assert_eq!(negative.status, RaimStatus::NotEnoughRedundancy);
        assert!(!negative.detected);
        assert_eq!(negative.threshold, None);
        assert_eq!(negative.redundancy, -1);
    }

    #[test]
    fn injected_outlier_has_largest_normalized_residual() {
        let mut residuals = clean_residuals();
        residuals.push(residual("G05", 0.0, 0.0));
        for residual in &mut residuals {
            residual.phase_m = 0.0;
        }
        residuals[3].code_m = 5.0;

        let identification =
            per_satellite_statistics(&residuals, &test_geometry(), RaimConfig::default()).unwrap();

        assert_eq!(identification.most_likely_fault.as_deref(), Some("G04"));
        let g04 = identification
            .statistics
            .iter()
            .find(|stat| stat.satellite_id == "G04")
            .expect("G04 statistic");
        assert_eq!(g04.statistic, g04.code);
        for stat in &identification.statistics {
            if stat.satellite_id != "G04" {
                assert!(g04.statistic > stat.statistic);
            }
        }
    }

    #[test]
    fn phase_outlier_with_ambiguity_columns_is_unobservable() {
        let mut residuals = clean_residuals();
        residuals.push(residual("G05", 0.0, 0.0));
        for residual in &mut residuals {
            residual.code_m = 0.0;
            residual.phase_m = 0.0;
        }
        residuals[2].phase_m = 5.0;

        assert_eq!(
            per_satellite_statistics(&residuals, &test_geometry(), RaimConfig::default()),
            Err(RaimError::SingularGeometry)
        );
    }

    #[test]
    fn protection_levels_are_finite_and_positive() {
        let geometry = protection_geometry(&[
            (0.0, 70.0),
            (72.0, 55.0),
            (144.0, 50.0),
            (216.0, 45.0),
            (288.0, 40.0),
            (45.0, 65.0),
        ]);

        let levels =
            protection_levels(&geometry, protection_receiver(), RaimConfig::default()).unwrap();

        assert!(levels.hpl_m.is_finite() && levels.hpl_m > 0.0);
        assert!(levels.vpl_m.is_finite() && levels.vpl_m > 0.0);
    }

    #[test]
    fn protection_levels_without_ztd_match_public_dop_path() {
        let geometry = protection_geometry(&[
            (0.0, 70.0),
            (72.0, 55.0),
            (144.0, 50.0),
            (216.0, 45.0),
            (288.0, 40.0),
            (45.0, 65.0),
        ]);

        let public =
            protection_levels(&geometry, protection_receiver(), RaimConfig::default()).unwrap();
        let internal = protection_levels_with_ztd(
            &geometry,
            None,
            protection_receiver(),
            RaimConfig::default(),
        )
        .unwrap();

        assert_eq!(internal.hpl_m.to_bits(), public.hpl_m.to_bits());
        assert_eq!(internal.vpl_m.to_bits(), public.vpl_m.to_bits());
    }

    #[test]
    fn ztd_protection_levels_use_estimated_state_projection() {
        let (source, epoch, state) = synthetic_case(6, None);
        let mut solve_config = fde_config();
        let mut tropo = super::super::TroposphereOptions::disabled();
        tropo.enabled = true;
        tropo.estimate_ztd = true;
        solve_config.tropo = tropo;
        let solve_tropo = solve_config.tropo;
        let solution = solve_float_epoch(&source, epoch.clone(), state, solve_config)
            .expect("ZTD-estimated solve");
        let geometry =
            geometry_for_solution(&source, &epoch, &solution, solve_tropo).expect("geometry");
        let receiver = receiver_geodetic(solution.position_m);

        let without_ztd =
            protection_levels(&geometry.rows, receiver, RaimConfig::default()).unwrap();
        let with_ztd = protection_levels_with_ztd(
            &geometry.rows,
            geometry.ztd_mappings.as_deref(),
            receiver,
            RaimConfig::default(),
        )
        .unwrap();
        let raim = raim_for_solution(
            &source,
            &epoch,
            &solution,
            solve_tropo,
            RaimConfig::default(),
        )
        .expect("RAIM with ZTD protection levels");

        assert!(with_ztd.hpl_m.is_finite() && with_ztd.hpl_m > 0.0);
        assert!(with_ztd.vpl_m.is_finite() && with_ztd.vpl_m > 0.0);
        assert_ne!(with_ztd.hpl_m.to_bits(), without_ztd.hpl_m.to_bits());
        assert_ne!(with_ztd.vpl_m.to_bits(), without_ztd.vpl_m.to_bits());
        assert_eq!(raim.hpl_m.expect("HPL").to_bits(), with_ztd.hpl_m.to_bits());
        assert_eq!(raim.vpl_m.expect("VPL").to_bits(), with_ztd.vpl_m.to_bits());
    }

    #[test]
    fn protection_levels_grow_when_geometry_degrades() {
        let normal = protection_geometry(&[
            (0.0, 70.0),
            (72.0, 55.0),
            (144.0, 50.0),
            (216.0, 45.0),
            (288.0, 40.0),
            (45.0, 65.0),
        ]);
        let degraded = protection_geometry(&[
            (0.0, 25.0),
            (8.0, 28.0),
            (16.0, 30.0),
            (24.0, 32.0),
            (32.0, 35.0),
            (40.0, 38.0),
        ]);

        let normal =
            protection_levels(&normal, protection_receiver(), RaimConfig::default()).unwrap();
        let degraded =
            protection_levels(&degraded, protection_receiver(), RaimConfig::default()).unwrap();

        assert!(degraded.hpl_m > normal.hpl_m);
        assert!(degraded.vpl_m > normal.vpl_m);
    }

    #[test]
    fn ztd_protection_levels_grow_when_geometry_degrades() {
        let normal = protection_geometry(&[
            (0.0, 70.0),
            (72.0, 55.0),
            (144.0, 50.0),
            (216.0, 45.0),
            (288.0, 40.0),
            (45.0, 65.0),
        ]);
        let normal_mappings = [1.064, 1.221, 1.305, 1.414, 1.556, 1.103];
        let degraded = protection_geometry(&[
            (0.0, 25.0),
            (8.0, 28.0),
            (16.0, 30.0),
            (24.0, 32.0),
            (32.0, 35.0),
            (40.0, 38.0),
        ]);
        let degraded_mappings = [2.366, 2.134, 2.0, 1.887, 1.743, 1.624];

        let normal = protection_levels_with_ztd(
            &normal,
            Some(&normal_mappings),
            protection_receiver(),
            RaimConfig::default(),
        )
        .unwrap();
        let degraded = protection_levels_with_ztd(
            &degraded,
            Some(&degraded_mappings),
            protection_receiver(),
            RaimConfig::default(),
        )
        .unwrap();

        assert!(normal.hpl_m.is_finite() && normal.hpl_m > 0.0);
        assert!(normal.vpl_m.is_finite() && normal.vpl_m > 0.0);
        assert!(degraded.hpl_m.is_finite() && degraded.hpl_m > 0.0);
        assert!(degraded.vpl_m.is_finite() && degraded.vpl_m > 0.0);
        assert!(degraded.hpl_m > normal.hpl_m);
        assert!(degraded.vpl_m > normal.vpl_m);
    }

    #[test]
    fn fde_excludes_faulted_satellite_and_restores_solution() {
        let (clean_source, clean_epoch, clean_state) = synthetic_case(6, None);
        let clean = solve_float_epoch(&clean_source, clean_epoch, clean_state, fde_config())
            .expect("clean solve");
        let (biased_source, biased_epoch, biased_state) = synthetic_case(6, Some("G05"));

        let fde = fde_float_epoch(
            &biased_source,
            biased_epoch,
            biased_state,
            fde_config(),
            fde_raim_config(),
        )
        .expect("FDE");

        assert_eq!(fde.status, RaimFdeStatus::Restored);
        assert_eq!(fde.excluded_sats, vec!["G05".to_string()]);
        assert!(fde.raim.hpl_m.expect("HPL") > 0.0);
        assert!(fde.raim.vpl_m.expect("VPL") > 0.0);
        assert_position_close(fde.solution.position_m, clean.position_m, 1.0e-3);
    }

    #[test]
    fn fde_refuses_exclusion_when_redundancy_would_be_exhausted() {
        let (source, epoch, state) = synthetic_case(5, Some("G05"));

        let fde =
            fde_float_epoch(&source, epoch, state, fde_config(), fde_raim_config()).expect("FDE");

        assert_eq!(fde.status, RaimFdeStatus::CannotExclude);
        assert!(fde.excluded_sats.is_empty());
        assert!(fde.raim.detected);
    }
}
