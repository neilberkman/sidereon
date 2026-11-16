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
//! #         let mut predict_options = PredictOptions::default();
//! #         predict_options.carrier_hz = F_L1_HZ;
//! #         predict_options.light_time = true;
//! #         predict_options.sagnac = true;
//! #         let prediction = predict(
//! #             &source,
//! #             *id,
//! #             truth,
//! #             0.0,
//! #             predict_options,
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
//! #             glonass_channel: None,
//! #             signals: None,
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
//! #     tropo_gradient_north_m: 0.0,
//! #     tropo_gradient_east_m: 0.0,
//! #     residual_ionosphere_m: BTreeMap::new(),
//! # };
//! # let weights = MeasurementWeights {
//! #     code: 1.0,
//! #     phase: 1.0,
//! #     elevation_weighting: false,
//! # };
//! # let mut solve_options = FloatSolveOptions::default();
//! # solve_options.max_iterations = 50;
//! # solve_options.position_tolerance_m = 1.0e-7;
//! # solve_options.clock_tolerance_m = 1.0e-7;
//! # solve_options.ambiguity_tolerance_m = 1.0e-7;
//! # solve_options.ztd_tolerance_m = 1.0e-7;
//! # let solve_config = FloatSolveConfig::new(
//! #     weights,
//! #     TroposphereOptions::disabled(),
//! #     RangeCorrections::disabled(),
//! #     solve_options,
//! #     None,
//! #     false,
//! #     false,
//! # );
//! let result = solve_float_epoch_with_raim(
//! #   &source,
//! #   epoch,
//! #   initial_state,
//! #   solve_config,
//!     {
//!         let mut config = RaimConfig::default();
//!         config.chi_square_threshold = Some(10.0);
//!         config
//!     },
//! )?;
//! assert_eq!(result.excluded_sats, vec!["G05".to_string()]);
//! # Ok(())
//! # }
//! ```

use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use crate::astro::math::linear::invert_matrix_last_tie;
#[cfg(test)]
use crate::astro::math::linear::invert_symmetric_pd;
use crate::estimation::substrate::parameters::{
    undifferenced_design_row, UndifferencedDesignOptions,
};
use crate::geometry::{dop, DopError, LineOfSight, Wgs84Geodetic};
use crate::observables::ObservableEphemerisSource;
use crate::quality::{chi2_inv, chi2_isf, noncentral_chi2_log_cdf, DEFAULT_P_FA};
use crate::validate;

use super::{
    solve_float_epoch, state_from_solution, ztd_unknown_count, FloatEpoch, FloatResidual,
    FloatSolution, FloatSolveConfig, FloatSolveError, FloatState, SatelliteClockCorrections,
    TroposphereOptions,
};

const DEFAULT_MISSED_DETECTION_PROBABILITY: f64 = 1.0e-3;
const DEFAULT_MEASUREMENT_SIGMA_M: f64 = 1.0;
const RESIDUAL_COMPONENTS_PER_ROW: usize = 2;
const MAX_NONCENTRAL_DOF: usize = 100_000;
const SNAPSHOT_BASE_STATES: usize = 4;
const LEVERAGE_TOLERANCE: f64 = 1.0e-12;
const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;

/// Configuration for PPP snapshot RAIM.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
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
    /// Standardized absolute code residual; `0.0` where the code row is untestable
    /// ([`Self::code_testable`]).
    pub code: f64,
    /// Standardized absolute phase residual; `0.0` where the phase row is untestable
    /// ([`Self::phase_testable`]).
    pub phase: f64,
    /// Satellite statistic: the larger of the testable components, `0.0` where neither is.
    pub statistic: f64,
    /// Whether the code row has redundancy to be tested: `false` where the snapshot design
    /// gives it a leverage of 1, so the states it observes absorb any error in it.
    pub code_testable: bool,
    /// Whether the phase row has redundancy to be tested: `false` where it is the only
    /// observer of its ambiguity, as a single-epoch snapshot phase row always is, so the
    /// ambiguity absorbs any error in it and its residual carries no test information.
    pub phase_testable: bool,
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
    /// The solve succeeded, but no RAIM test ran because redundancy was zero.
    Untested,
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
    /// A protection-level probability evaluation or bias bracket failed
    /// numerically after the geometry and detection test were valid.
    NumericalProtectionLevel,
    /// The detectable-bias search requires a noncentrality above the supported
    /// maximum of 1e12. This is a domain limit, distinct from numerical failure.
    UnsupportedNoncentrality,
    /// The noncentral-CDF protection-level calculation was requested beyond
    /// the degree-of-freedom range with a validated accuracy bound.
    UnsupportedDof {
        /// Requested degrees of freedom.
        dof: usize,
        /// Largest supported value.
        max: usize,
    },
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
            Self::NumericalProtectionLevel => {
                write!(f, "RAIM protection-level probability evaluation failed")
            }
            Self::UnsupportedNoncentrality => {
                write!(f, "RAIM noncentrality exceeds supported maximum 1e12")
            }
            Self::UnsupportedDof { dof, max } => {
                write!(
                    f,
                    "RAIM degrees of freedom {dof} exceed supported maximum {max}"
                )
            }
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
    let redundancy = isize::try_from(n_obs)
        .ok()
        .and_then(|n_obs| {
            isize::try_from(n_states)
                .ok()
                .map(|n_states| n_obs - n_states)
        })
        .ok_or(RaimError::InvalidConfig {
            field: "n_states",
            reason: "does not fit the reported redundancy",
        })?;

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

    let threshold = match config.chi_square_threshold {
        Some(threshold) => threshold,
        None => chi2_isf(config.false_alarm_probability, redundancy as usize).map_err(|_| {
            RaimError::InvalidConfig {
                field: "false_alarm_probability",
                reason: "cannot derive chi-square threshold",
            }
        })?,
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
    validate_geometry(residuals, geometry)?;
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
    per_satellite_statistics_with_atmosphere(residuals, geometry, None, None, config)
}

fn per_satellite_statistics_with_atmosphere(
    residuals: &[FloatResidual],
    geometry: &[RaimGeometryRow],
    ztd_mappings: Option<&[f64]>,
    tropo_gradient_mappings: Option<&[[f64; 2]]>,
    config: RaimConfig,
) -> Result<RaimIdentification, RaimError> {
    validate_config(config)?;
    validate_geometry(residuals, geometry)?;
    if let Some(mappings) = ztd_mappings {
        validate_ztd_mappings(residuals.len(), mappings)?;
    }
    if let Some(mappings) = tropo_gradient_mappings {
        validate_tropo_gradient_mappings(residuals.len(), mappings)?;
    }
    let ambiguity_ids = residuals
        .iter()
        .map(|residual| residual.ambiguity_id.clone())
        .collect::<Vec<_>>();
    let rows = snapshot_design_rows(
        residuals,
        geometry,
        ztd_mappings,
        tropo_gradient_mappings,
        1,
        &[],
        &ambiguity_ids,
        config.measurement_sigma_m,
    )?;
    let normal = normal_matrix(&rows)?;
    let q = invert_matrix_last_tie(&normal).ok_or(RaimError::SingularGeometry)?;

    let mut statistics = Vec::with_capacity(residuals.len());
    let mut most_likely_fault = None;
    let mut worst = 0.0;
    for (idx, residual) in residuals.iter().enumerate() {
        let code = standardized_abs(&rows, 2 * idx, &q)?;
        let phase = standardized_abs(&rows, 2 * idx + 1, &q)?;
        let (code_testable, phase_testable) = (code.is_some(), phase.is_some());
        let (code, phase) = (code.unwrap_or(0.0), phase.unwrap_or(0.0));
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
            code_testable,
            phase_testable,
        });
    }

    Ok(RaimIdentification {
        statistics,
        most_likely_fault,
    })
}

/// Compute Brown (1992) slope-based snapshot-RAIM protection levels with unit
/// solver weights.
///
/// Use [`protection_levels_with_weights`] when the residual test used non-unit
/// weights. At least one degree of redundancy is required.
pub fn protection_levels(
    geometry: &[RaimGeometryRow],
    receiver: Wgs84Geodetic,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    protection_levels_with_weights(geometry, &vec![1.0; geometry.len()], receiver, config)
}

/// Compute Brown (1992) slope-based snapshot-RAIM protection levels using the
/// same inverse-sigma row weights as the residual test.
///
/// The detectable bias is the square root of the noncentrality whose
/// noncentral chi-square CDF at the `p_fa` detection threshold equals `p_md`.
/// This is the reference's one-sided missed-detection definition; unlike the
/// former two-sided normal multiplier (3.2905 at `p_md=0.001`), its one-degree
/// asymptote uses 3.0902 and includes the detection threshold.
///
/// One protection-level computation evaluates the noncentral CDF once at zero
/// noncentrality, once for each upper-bracket doubling, and up to 80 times for
/// bisection. One degree of freedom uses a folded normal; subnormal x uses
/// the leading gamma term. Otherwise an evaluation uses at most 10,000
/// Poisson/gamma terms through noncentrality 100, or 64/128 panels × 16 = up to
/// 2,048 quadrature nodes above 100 (128 panels above 16,384 perpendicular
/// degrees of freedom). The noncentrality domain ends at 1e12, limiting
/// bracket evaluation to 41 calls and total CDF evaluation to 122 calls:
/// at most 1,220,000 gamma terms or 249,856 quadrature nodes. These are
/// conservative separate bounds on the mixed sequence. An upper-mixture term
/// can require a second central-gamma tail evaluation in log form after
/// underflow, so the gamma-tail evaluation bound is 2,440,000; these kernel
/// counts do not count their internal series/continued-fraction iterations.
/// Matrix formation
/// and per-row slopes are additional. The path is supported through 100,000
/// residual degrees of freedom; larger requests return
/// [`RaimError::UnsupportedDof`], and a root outside the noncentrality domain
/// returns [`RaimError::UnsupportedNoncentrality`].
/// Invalid CDF threshold or degrees-of-freedom arguments retain their domain
/// classification (`InvalidConfig`, `InvalidGeometry`, or `UnsupportedDof`);
/// only a numerical evaluation failure becomes `NumericalProtectionLevel`.
pub fn protection_levels_with_weights(
    geometry: &[RaimGeometryRow],
    weights: &[f64],
    receiver: Wgs84Geodetic,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    validate_config(config)?;
    validate_geometry_only(geometry)?;
    if weights.len() != geometry.len() {
        return Err(RaimError::InvalidGeometry {
            field: "weights",
            reason: "length must match geometry",
        });
    }
    if geometry.len() <= SNAPSHOT_BASE_STATES {
        return Err(RaimError::InvalidGeometry {
            field: "geometry",
            reason: "protection levels require positive redundancy",
        });
    }
    let max_weight = weights.iter().copied().try_fold(0.0_f64, |max, weight| {
        validate_weight(weight, "protection_weight").map(|weight| max.max(weight))
    })?;
    let design = geometry
        .iter()
        .zip(weights)
        .map(|(row, &weight)| {
            let weight = validate_weight(weight, "protection_weight")? / max_weight;
            Ok(vec![
                -row.line_of_sight.e_x * weight,
                -row.line_of_sight.e_y * weight,
                -row.line_of_sight.e_z * weight,
                weight,
            ])
        })
        .collect::<Result<Vec<_>, RaimError>>()?;
    let mut normalized_config = config;
    normalized_config.measurement_sigma_m /= max_weight;
    slope_protection_levels(&design, receiver, normalized_config)
}

/// Return the former `K * sigma * DOP` quantity as a fault-free accuracy bound.
/// It is not a fault-mode RAIM protection level.
///
/// `weights` are inverse standard deviations, exactly as in
/// [`FloatResidual::code_weight`], not inverse variances.
pub fn fault_free_accuracy_bounds(
    geometry: &[RaimGeometryRow],
    weights: &[f64],
    receiver: Wgs84Geodetic,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    validate_config(config)?;
    validate_geometry_only(geometry)?;
    if weights.len() != geometry.len() {
        return Err(RaimError::InvalidGeometry {
            field: "weights",
            reason: "length must match geometry",
        });
    }
    let max_weight = weights.iter().copied().try_fold(0.0_f64, |max, weight| {
        validate_weight(weight, "accuracy_weight").map(|weight| max.max(weight))
    })?;
    let inverse_variance_weights = weights
        .iter()
        .map(|weight| {
            let normalized = weight / max_weight;
            normalized * normalized
        })
        .collect::<Vec<_>>();
    let los = geometry
        .iter()
        .map(|row| row.line_of_sight)
        .collect::<Vec<_>>();
    let d = dop(&los, &inverse_variance_weights, receiver).map_err(RaimError::Dop)?;
    let k_md = missed_detection_multiplier(config)?;
    let effective_sigma = config.measurement_sigma_m / max_weight;
    let hpl_m = k_md * effective_sigma * d.hdop;
    let vpl_m = k_md * effective_sigma * d.vdop;
    if hpl_m.is_finite() && hpl_m > 0.0 && vpl_m.is_finite() && vpl_m > 0.0 {
        Ok(ProtectionLevels { hpl_m, vpl_m })
    } else {
        Err(RaimError::SingularGeometry)
    }
}

fn slope_protection_levels(
    design: &[Vec<f64>],
    receiver: Wgs84Geodetic,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    slope_protection_levels_from_design(
        design,
        design.len().saturating_sub(4),
        receiver,
        config,
        config.measurement_sigma_m,
        None,
    )
}

fn slope_protection_levels_from_design(
    design: &[Vec<f64>],
    dof: usize,
    receiver: Wgs84Geodetic,
    config: RaimConfig,
    output_scale_m: f64,
    fault_row_indices: Option<&[usize]>,
) -> Result<ProtectionLevels, RaimError> {
    if dof == 0 {
        return Err(RaimError::InvalidGeometry {
            field: "geometry",
            reason: "protection levels require positive redundancy",
        });
    }
    let width = design.first().map_or(0, Vec::len);
    if width < 4 || design.iter().any(|row| row.len() != width) {
        return Err(RaimError::SingularGeometry);
    }
    let mut normal = vec![vec![0.0; width]; width];
    for h in design {
        for row in 0..width {
            for col in 0..width {
                normal[row][col] += h[row] * h[col];
            }
        }
    }
    let q = invert_matrix_last_tie(&normal).ok_or(RaimError::SingularGeometry)?;
    let r = enu_rotation(receiver);
    let mut max_h = 0.0_f64;
    let mut max_v = 0.0_f64;
    for (row_index, h) in design.iter().enumerate() {
        if fault_row_indices.is_some_and(|indices| !indices.contains(&row_index)) {
            continue;
        }
        let leverage = row_leverage(h, &q)?;
        let residual_sigma = (1.0 - leverage).max(0.0).sqrt();
        if residual_sigma <= LEVERAGE_TOLERANCE {
            return Err(RaimError::SingularGeometry);
        }
        let mut sensitivity = [0.0; 3];
        for state in 0..3 {
            sensitivity[state] = (0..width).map(|j| q[state][j] * h[j]).sum();
        }
        let east = (0..3).map(|j| r[0][j] * sensitivity[j]).sum::<f64>();
        let north = (0..3).map(|j| r[1][j] * sensitivity[j]).sum::<f64>();
        let up = (0..3).map(|j| r[2][j] * sensitivity[j]).sum::<f64>();
        max_h = max_h.max(libm::hypot(east, north) / residual_sigma);
        max_v = max_v.max(up.abs() / residual_sigma);
    }
    let bias = detectable_bias(dof, config)?;
    Ok(ProtectionLevels {
        hpl_m: max_h * bias * output_scale_m,
        vpl_m: max_v * bias * output_scale_m,
    })
}

fn detectable_bias(dof: usize, config: RaimConfig) -> Result<f64, RaimError> {
    if dof > MAX_NONCENTRAL_DOF {
        return Err(RaimError::UnsupportedDof {
            dof,
            max: MAX_NONCENTRAL_DOF,
        });
    }
    let threshold = config.chi_square_threshold.map_or_else(
        || {
            chi2_isf(config.false_alarm_probability, dof).map_err(|_| RaimError::InvalidConfig {
                field: "false_alarm_probability",
                reason: "cannot derive chi-square threshold",
            })
        },
        Ok,
    )?;
    let log_target = libm::log(config.missed_detection_probability);
    let central_log_cdf =
        noncentral_chi2_log_cdf(threshold, dof, 0.0).map_err(|error| cdf_error(error, dof))?;
    if central_log_cdf < log_target {
        return Err(RaimError::InvalidConfig {
            field: "missed_detection_probability",
            reason: "exceeds the fault-free detection-threshold CDF",
        });
    }
    let mut lo = 0.0;
    let mut hi = 1.0;
    while noncentral_chi2_log_cdf(threshold, dof, hi).map_err(|error| cdf_error(error, dof))?
        > log_target
    {
        if hi == 1.0e12 {
            return Err(RaimError::UnsupportedNoncentrality);
        }
        hi = (hi * 2.0).min(1.0e12);
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        let log_cdf =
            noncentral_chi2_log_cdf(threshold, dof, mid).map_err(|error| cdf_error(error, dof))?;
        if log_cdf > log_target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= 4.0 * f64::EPSILON * hi.max(1.0) {
            break;
        }
    }
    Ok((0.5 * (lo + hi)).sqrt())
}

fn cdf_error(error: crate::quality::NoncentralCdfError, dof: usize) -> RaimError {
    match error {
        crate::quality::NoncentralCdfError::NoncentralityOutOfRange => {
            RaimError::UnsupportedNoncentrality
        }
        crate::quality::NoncentralCdfError::InvalidX => RaimError::InvalidConfig {
            field: "chi_square_threshold",
            reason: "CDF argument must be finite and nonnegative",
        },
        crate::quality::NoncentralCdfError::InvalidDof if dof > MAX_NONCENTRAL_DOF => {
            RaimError::UnsupportedDof {
                dof,
                max: MAX_NONCENTRAL_DOF,
            }
        }
        crate::quality::NoncentralCdfError::InvalidDof => RaimError::InvalidGeometry {
            field: "dof",
            reason: "CDF requires positive degrees of freedom",
        },
        crate::quality::NoncentralCdfError::Numerical => RaimError::NumericalProtectionLevel,
    }
}

fn enu_rotation(receiver: Wgs84Geodetic) -> [[f64; 3]; 3] {
    let (sphi, cphi) = (libm::sin(receiver.lat_rad), libm::cos(receiver.lat_rad));
    let (slam, clam) = (libm::sin(receiver.lon_rad), libm::cos(receiver.lon_rad));
    [
        [-slam, clam, 0.0],
        [-sphi * clam, -sphi * slam, cphi],
        [cphi * clam, cphi * slam, sphi],
    ]
}

#[cfg(test)]
fn protection_levels_with_atmosphere(
    geometry: &[RaimGeometryRow],
    ztd_mappings: Option<&[f64]>,
    tropo_gradient_mappings: Option<&[[f64; 2]]>,
    receiver: Wgs84Geodetic,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    if ztd_mappings.is_none() && tropo_gradient_mappings.is_none() {
        return protection_levels(geometry, receiver, config);
    }

    validate_config(config)?;
    validate_geometry_only(geometry)?;
    if let Some(mappings) = ztd_mappings {
        validate_ztd_mappings(geometry.len(), mappings)?;
    }
    if let Some(mappings) = tropo_gradient_mappings {
        validate_tropo_gradient_mappings(geometry.len(), mappings)?;
    }

    let q = protection_position_covariance_with_atmosphere(
        geometry,
        ztd_mappings,
        tropo_gradient_mappings,
    )?;
    let enu = rotate_position_covariance_to_enu(&q, receiver);
    scaled_protection_levels(enu[0][0] + enu[1][1], enu[2][2], config)
}

#[cfg(test)]
fn protection_levels_with_ztd(
    geometry: &[RaimGeometryRow],
    ztd_mappings: Option<&[f64]>,
    receiver: Wgs84Geodetic,
    config: RaimConfig,
) -> Result<ProtectionLevels, RaimError> {
    protection_levels_with_atmosphere(geometry, ztd_mappings, None, receiver, config)
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
            solve_config.corrections.satellite_clock.as_ref(),
            raim_config,
        )
        .map_err(RaimFdeError::Raim)?;

        if raim.status == RaimStatus::NotEnoughRedundancy {
            return Ok(RaimFdeResult {
                solution,
                raim,
                excluded_sats,
                status: RaimFdeStatus::Untested,
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
        // Observations the solve leaves out for an unresolved SSR/HAS bias add no rows, so
        // only the others count towards the redundancy after this exclusion.
        let solved_after = next_epoch
            .observations
            .iter()
            .filter(|obs| {
                !solution
                    .ssr_bias_exclusions
                    .iter()
                    .any(|exclusion| exclusion.ambiguity_id == obs.ambiguity_id)
            })
            .count();
        if !has_positive_redundancy_after_exclusion(solved_after, solve_config.tropo)
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
    satellite_clock: Option<&SatelliteClockCorrections>,
    config: RaimConfig,
) -> Result<RaimResult, RaimError> {
    let geometry =
        geometry_for_solution(source, epoch, solution, tropo, satellite_clock).map_err(|_| {
            RaimError::InvalidGeometry {
                field: "solution",
                reason: "could not build line-of-sight geometry",
            }
        })?;
    validate_geometry(&solution.residuals_m, &geometry.rows)?;
    let n_states = snapshot_state_count_for_solution(solution)?;
    let mut result = global_test(&solution.residuals_m, n_states, config)?;
    if result.status != RaimStatus::NotEnoughRedundancy {
        let identification = per_satellite_statistics_with_atmosphere(
            &solution.residuals_m,
            &geometry.rows,
            geometry.ztd_mappings.as_deref(),
            geometry.tropo_gradient_mappings.as_deref(),
            config,
        )?;
        result.satellite_statistics = identification.statistics;
        result.most_likely_fault = identification.most_likely_fault;
    }
    if result.status != RaimStatus::NotEnoughRedundancy {
        let receiver = receiver_geodetic(solution.position_m);
        let ionosphere_ids = solution
            .residual_ionosphere_m
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        let ambiguity_ids = solution.ambiguities_m.keys().cloned().collect::<Vec<_>>();
        let design_rows = snapshot_design_rows(
            &solution.residuals_m,
            &geometry.rows,
            geometry.ztd_mappings.as_deref(),
            geometry.tropo_gradient_mappings.as_deref(),
            solution.epoch_clocks_m.len(),
            &ionosphere_ids,
            &ambiguity_ids,
            config.measurement_sigma_m,
        )?;
        let design = design_rows
            .iter()
            .map(|row| row.h.clone())
            .collect::<Vec<_>>();
        let code_rows = (0..solution.residuals_m.len())
            .map(|index| 2 * index)
            .collect::<Vec<_>>();
        let levels = slope_protection_levels_from_design(
            &design,
            result.redundancy as usize,
            receiver,
            config,
            1.0,
            Some(&code_rows),
        );
        match levels {
            Ok(levels) => {
                result.hpl_m = Some(levels.hpl_m);
                result.vpl_m = Some(levels.vpl_m);
            }
            Err(RaimError::SingularGeometry) => {
                // The detection result remains valid when a unit-leverage code
                // row makes fault-slope protection untestable.
                result.hpl_m = None;
                result.vpl_m = None;
            }
            Err(error) => return Err(error),
        }
    }
    Ok(result)
}

#[derive(Debug, Clone)]
struct RaimSolutionGeometry {
    rows: Vec<RaimGeometryRow>,
    ztd_mappings: Option<Vec<f64>>,
    tropo_gradient_mappings: Option<Vec<[f64; 2]>>,
}

fn geometry_for_solution(
    source: &dyn ObservableEphemerisSource,
    epoch: &FloatEpoch,
    solution: &FloatSolution,
    tropo: TroposphereOptions,
    satellite_clock: Option<&SatelliteClockCorrections>,
) -> Result<RaimSolutionGeometry, FloatSolveError> {
    let mut rows = Vec::with_capacity(solution.residuals_m.len());
    let mut ztd_mappings = solution
        .ztd_residual_m
        .is_some()
        .then(|| Vec::with_capacity(solution.residuals_m.len()));
    let mut tropo_gradient_mappings = solution
        .tropo_gradient_north_m
        .zip(solution.tropo_gradient_east_m)
        .map(|_| Vec::with_capacity(solution.residuals_m.len()));
    for residual in &solution.residuals_m {
        let obs = epoch
            .observations
            .iter()
            .find(|obs| obs.ambiguity_id == residual.ambiguity_id)
            .ok_or(FloatSolveError::InvalidInput {
                field: "raim geometry ambiguity_id",
                reason: "residual observation missing from epoch",
            })?;
        let pred = super::observation_geometry(
            source,
            obs,
            solution.position_m,
            epoch.t_rx_j2000_s,
            satellite_clock,
        )?;
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
        if let Some(mappings) = &mut tropo_gradient_mappings {
            let tropo_model =
                super::model::model_troposphere(&pred, solution.position_m, epoch, tropo)?;
            mappings.push(tropo_model.gradient_mapping);
        }
        rows.push(RaimGeometryRow {
            satellite_id: residual.satellite_id.clone(),
            line_of_sight: LineOfSight::new(pred.los_unit[0], pred.los_unit[1], pred.los_unit[2]),
        });
    }
    Ok(RaimSolutionGeometry {
        rows,
        ztd_mappings,
        tropo_gradient_mappings,
    })
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
    let atmospheric_states = ztd_unknown_count(tropo)
        .checked_add(if tropo.enabled && tropo.estimate_tropo_gradients {
            2
        } else {
            0
        })
        .ok_or(RaimError::InvalidGeometry {
            field: "geometry",
            reason: "too many atmospheric states",
        })?;
    let n_states = snapshot_state_count_from_parts(1, atmospheric_states, n_sats_after)?;
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
        usize::from(solution.ztd_residual_m.is_some())
            + usize::from(solution.tropo_gradient_north_m.is_some())
            + usize::from(solution.tropo_gradient_east_m.is_some()),
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

fn validate_tropo_gradient_mappings(
    expected_len: usize,
    mappings: &[[f64; 2]],
) -> Result<(), RaimError> {
    if mappings.len() != expected_len {
        return Err(RaimError::InvalidGeometry {
            field: "tropo_gradient_mapping",
            reason: "length must match geometry",
        });
    }
    for mapping in mappings {
        for &value in mapping {
            validate::finite(value, "tropo_gradient_mapping").map_err(|_| {
                RaimError::InvalidGeometry {
                    field: "tropo_gradient_mapping",
                    reason: "must be finite",
                }
            })?;
        }
    }
    Ok(())
}

#[cfg(test)]
fn protection_position_covariance_with_atmosphere(
    geometry: &[RaimGeometryRow],
    ztd_mappings: Option<&[f64]>,
    tropo_gradient_mappings: Option<&[[f64; 2]]>,
) -> Result<[[f64; 3]; 3], RaimError> {
    let atmospheric_states =
        usize::from(ztd_mappings.is_some()) + 2 * usize::from(tropo_gradient_mappings.is_some());
    let state_count = SNAPSHOT_BASE_STATES + atmospheric_states;
    let mut normal = vec![vec![0.0_f64; state_count]; state_count];

    for (idx, row) in geometry.iter().enumerate() {
        let mut h = vec![
            -row.line_of_sight.e_x,
            -row.line_of_sight.e_y,
            -row.line_of_sight.e_z,
            1.0,
        ];
        if let Some(mappings) = ztd_mappings {
            h.push(mappings[idx]);
        }
        if let Some(mappings) = tropo_gradient_mappings {
            h.extend_from_slice(&mappings[idx]);
        }
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

#[cfg(test)]
fn rotate_position_covariance_to_enu(q: &[[f64; 3]; 3], receiver: Wgs84Geodetic) -> [[f64; 3]; 3] {
    let sphi = libm::sin(receiver.lat_rad);
    let cphi = libm::cos(receiver.lat_rad);
    let slam = libm::sin(receiver.lon_rad);
    let clam = libm::cos(receiver.lon_rad);
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

#[cfg(test)]
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

#[allow(clippy::too_many_arguments)]
fn snapshot_design_rows(
    residuals: &[FloatResidual],
    geometry: &[RaimGeometryRow],
    ztd_mappings: Option<&[f64]>,
    tropo_gradient_mappings: Option<&[[f64; 2]]>,
    n_clocks: usize,
    residual_ionosphere_ids: &[String],
    ambiguity_ids: &[String],
    measurement_sigma_m: f64,
) -> Result<Vec<StandardizationRow>, RaimError> {
    let mut rows = Vec::with_capacity(residuals.len() * RESIDUAL_COMPONENTS_PER_ROW);
    let n_ambiguities = ambiguity_ids.len();
    for (ambiguity_idx, (residual, geometry)) in residuals.iter().zip(geometry).enumerate() {
        if residual.epoch_index >= n_clocks {
            return Err(RaimError::InvalidResidual {
                field: "epoch_index",
            });
        }
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
        let tropo_gradient_mapping =
            tropo_gradient_mappings.map(|mappings| mappings[ambiguity_idx]);
        let code = code_residual * code_weight;
        let phase = phase_residual * phase_weight;
        if !code.is_finite() || !phase.is_finite() {
            return Err(RaimError::InvalidResidual {
                field: "weighted_residual",
            });
        }
        let active_ionosphere = residual_ionosphere_ids
            .iter()
            .position(|id| id == &residual.ambiguity_id);
        let active_ambiguity = ambiguity_ids
            .iter()
            .position(|id| id == &residual.ambiguity_id)
            .ok_or(RaimError::InvalidResidual {
                field: "ambiguity_id",
            })?;
        rows.push(StandardizationRow {
            h: weighted_snapshot_row(
                geometry.line_of_sight,
                code_weight,
                residual.epoch_index,
                n_clocks,
                ztd_mapping,
                tropo_gradient_mapping,
                residual_ionosphere_ids.len(),
                active_ionosphere.map(|idx| (idx, 1.0)),
                n_ambiguities,
                None,
            ),
            residual: code,
        });
        rows.push(StandardizationRow {
            h: weighted_snapshot_row(
                geometry.line_of_sight,
                phase_weight,
                residual.epoch_index,
                n_clocks,
                ztd_mapping,
                tropo_gradient_mapping,
                residual_ionosphere_ids.len(),
                active_ionosphere.map(|idx| (idx, -1.0)),
                n_ambiguities,
                Some(active_ambiguity),
            ),
            residual: phase,
        });
    }
    Ok(rows)
}

#[allow(clippy::too_many_arguments)]
fn weighted_snapshot_row(
    line_of_sight: LineOfSight,
    weight: f64,
    epoch_index: usize,
    n_clocks: usize,
    ztd_mapping: Option<f64>,
    tropo_gradient_mapping: Option<[f64; 2]>,
    residual_ionosphere_columns: usize,
    active_residual_ionosphere: Option<(usize, f64)>,
    n_ambiguities: usize,
    active_ambiguity: Option<usize>,
) -> Vec<f64> {
    let mut row = undifferenced_design_row(
        [-line_of_sight.e_x, -line_of_sight.e_y, -line_of_sight.e_z],
        epoch_index,
        n_clocks,
        UndifferencedDesignOptions {
            ztd_mapping,
            tropo_gradient_mapping,
            residual_ionosphere_columns,
            active_residual_ionosphere,
            ambiguity_columns: n_ambiguities,
            active_ambiguity,
        },
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

/// Standardized residual magnitude of `rows[index]`, or `None` where the row has no
/// redundancy to be tested.
///
/// A row that is the only one observing some state, as a snapshot phase row is for its
/// ambiguity, has a leverage of exactly 1: that state absorbs any error in the row, so the
/// row's residual is rounding noise of the fit and carries no test information. That is
/// read from the structure rather than from `1 - h Q h`, whose rounding error grows with
/// the conditioning of the normal matrix and can fall on either side of zero.
fn standardized_abs(
    rows: &[StandardizationRow],
    index: usize,
    q: &[Vec<f64>],
) -> Result<Option<f64>, RaimError> {
    let row = &rows[index];
    if is_sole_observer(rows, index) {
        return Ok(None);
    }
    let variance_factor = 1.0 - row_leverage(&row.h, q)?;
    if !variance_factor.is_finite() {
        return Err(RaimError::SingularGeometry);
    }
    // A computed leverage of 1 leaves the row no redundancy either: untestable, whatever
    // its residual, which is then rounding noise of the fit.
    if variance_factor.abs() <= LEVERAGE_TOLERANCE {
        return Ok(None);
    }
    if variance_factor < 0.0 {
        return Err(RaimError::SingularGeometry);
    }
    let standardized = row.residual.abs() / variance_factor.sqrt();
    if standardized.is_finite() {
        Ok(Some(standardized))
    } else {
        Err(RaimError::SingularGeometry)
    }
}

/// Whether `rows[index]` is the only row with a nonzero entry in some column.
fn is_sole_observer(rows: &[StandardizationRow], index: usize) -> bool {
    rows[index].h.iter().enumerate().any(|(column, value)| {
        *value != 0.0
            && rows
                .iter()
                .enumerate()
                .all(|(other, row)| other == index || row.h.get(column).copied() == Some(0.0))
    })
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
    use serde_json::Value;
    use std::collections::BTreeMap;

    use crate::geometry::line_of_sight_from_az_el_deg;
    use crate::observables::{ObservableState, ObservablesError};
    use crate::ppp_corrections::CivilDateTime;
    use crate::{GnssSatelliteId, GnssSystem};

    #[test]
    fn public_protection_levels_report_unsupported_noncentrality() {
        let fixture: Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/noncentral_log_cdf_mpmath.json"
        ))
        .unwrap();
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| {
                row["dof"] == 1
                    && row["x"] == "1000000000000000.0"
                    && row["lambda"] == "1000000000000.0"
            })
            .unwrap();
        // The independent Chernoff certificate makes the CDF round to one
        // even at the supported noncentrality ceiling: no bias root exists
        // within that domain for the configured missed-detection probability.
        assert_eq!(case["tail"], "chernoff_underflow");
        let config = RaimConfig {
            chi_square_threshold: Some(case["x"].as_str().unwrap().parse().unwrap()),
            ..RaimConfig::default()
        };
        assert_eq!(
            protection_levels(&test_geometry(), protection_receiver(), config),
            Err(RaimError::UnsupportedNoncentrality)
        );
        use crate::quality::NoncentralCdfError;
        assert!(matches!(
            cdf_error(NoncentralCdfError::InvalidX, 1),
            RaimError::InvalidConfig {
                field: "chi_square_threshold",
                ..
            }
        ));
        assert!(matches!(
            cdf_error(NoncentralCdfError::InvalidDof, 0),
            RaimError::InvalidGeometry { field: "dof", .. }
        ));
        assert!(matches!(
            cdf_error(NoncentralCdfError::InvalidDof, MAX_NONCENTRAL_DOF + 1),
            RaimError::UnsupportedDof { .. }
        ));
        assert_eq!(
            cdf_error(NoncentralCdfError::NoncentralityOutOfRange, 1),
            RaimError::UnsupportedNoncentrality
        );
    }

    #[test]
    fn detectable_bias_matches_high_precision_noncentral_grid() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/ppp_raim_oracles.json"))
                .expect("PPP RAIM fixture");
        for case in fixture["noncentral_bias_grid"]
            .as_array()
            .expect("noncentral grid")
        {
            let dof = case["dof"].as_u64().unwrap() as usize;
            let threshold = case["threshold"].as_f64().unwrap();
            let p_md = case["p_md"].as_f64().unwrap();
            let config = RaimConfig {
                missed_detection_probability: p_md,
                chi_square_threshold: Some(threshold),
                ..RaimConfig::default()
            };
            let result = detectable_bias(dof, config);
            let Some(expected) = case["pbias"].as_str() else {
                assert!(
                    result.is_err(),
                    "dof={dof} threshold={threshold} p_md={p_md}"
                );
                continue;
            };
            let expected = expected.parse::<f64>().unwrap();
            // These degrees of freedom use 1,024 Gauss-Legendre nodes. For these analytic
            // integrands its 16-point panel remainder is below binary64
            // rounding; 256 rounded elementary operations along the active
            // log-sum path give this conservative forward bound.
            let tolerance = 256.0 * f64::EPSILON * expected.max(1.0);
            let actual = result.expect("detectable bias");
            assert!(
                (actual - expected).abs() <= tolerance,
                "dof={dof} threshold={threshold} p_md={p_md} actual={actual:.17e} expected={expected:.17e} tolerance={tolerance:.3e}"
            );
        }
    }

    #[test]
    fn detectable_bias_rejects_dof_above_supported_ceiling() {
        assert_eq!(
            detectable_bias(MAX_NONCENTRAL_DOF + 1, RaimConfig::default()),
            Err(RaimError::UnsupportedDof {
                dof: MAX_NONCENTRAL_DOF + 1,
                max: MAX_NONCENTRAL_DOF,
            })
        );
    }

    #[test]
    fn full_solver_design_protection_uses_its_clock_count_weights_and_dof() {
        let fixture: Value =
            serde_json::from_str(include_str!("../../tests/fixtures/ppp_raim_oracles.json"))
                .expect("PPP RAIM fixture");
        let geodetic = fixture["inputs"]["receiver_geodetic_rad"]
            .as_array()
            .unwrap();
        let receiver = Wgs84Geodetic::new(
            geodetic[0].as_f64().unwrap(),
            geodetic[1].as_f64().unwrap(),
            0.0,
        )
        .unwrap();
        let line_of_sight = fixture["line_of_sight_ecef"].as_array().unwrap();
        for case in fixture["solver_design_cases"].as_array().unwrap() {
            let clocks = case["clock_indices"].as_array().unwrap();
            let n_clocks = clocks
                .iter()
                .map(|value| value.as_u64().unwrap())
                .max()
                .unwrap()
                + 1;
            let code_weights = case["code_weights_inverse_sigma"].as_array().unwrap();
            let phase_weights = case["phase_weights_inverse_sigma"].as_array().unwrap();
            let satellites = clocks.len();
            let mut design = Vec::with_capacity(2 * satellites);
            for index in 0..satellites {
                let los = line_of_sight[index].as_array().unwrap();
                let clock = clocks[index].as_u64().unwrap() as usize;
                let common = [
                    -los[0].as_f64().unwrap(),
                    -los[1].as_f64().unwrap(),
                    -los[2].as_f64().unwrap(),
                ];
                let mut code = common.to_vec();
                code.extend(
                    (0..n_clocks as usize).map(|column| usize::from(column == clock) as f64),
                );
                code.extend(vec![0.0; satellites]);
                let code_weight = code_weights[index].as_f64().unwrap();
                code.iter_mut().for_each(|value| *value *= code_weight);
                design.push(code);
                let mut phase = common.to_vec();
                phase.extend(
                    (0..n_clocks as usize).map(|column| usize::from(column == clock) as f64),
                );
                phase.extend((0..satellites).map(|column| usize::from(column == index) as f64));
                let phase_weight = phase_weights[index].as_f64().unwrap();
                phase.iter_mut().for_each(|value| *value *= phase_weight);
                design.push(phase);
            }
            let dof = case["dof"].as_u64().unwrap() as usize;
            let config = RaimConfig {
                chi_square_threshold: Some(case["threshold"].as_f64().unwrap()),
                ..RaimConfig::default()
            };
            let code_rows = (0..satellites).map(|index| 2 * index).collect::<Vec<_>>();
            let actual = slope_protection_levels_from_design(
                &design,
                dof,
                receiver,
                config,
                1.0,
                Some(&code_rows),
            )
            .unwrap();
            let condition = case["normal_condition_number_2"].as_f64().unwrap();
            for (value, field) in [(actual.hpl_m, "hpl_m"), (actual.vpl_m, "vpl_m")] {
                let expected = case[field].as_f64().unwrap();
                let tolerance = 32.0 * f64::EPSILON * condition * expected.abs();
                assert!((value - expected).abs() <= tolerance);
            }
        }
    }

    fn residual(satellite_id: &str, code_m: f64, phase_m: f64) -> FloatResidual {
        FloatResidual {
            epoch_index: 0,
            satellite_id: satellite_id.to_string(),
            ambiguity_id: satellite_id.to_string(),
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
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
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
                let bias = if Some(satellite_id.as_str()) == biased_satellite {
                    50.0
                } else {
                    0.0
                };
                let (code_m, prediction) = crate::precise_positioning::synthetic_placed_code(
                    &source,
                    *id,
                    truth,
                    0.0,
                    |geometry| geometry.geometric_range_m + clock_m + bias,
                );
                let ambiguity_m = ambiguities_m.get(&satellite_id).copied().unwrap();
                super::super::FloatObservation {
                    sat: *id,
                    satellite_id: satellite_id.clone(),
                    ambiguity_id: satellite_id,
                    code_m,
                    phase_m: prediction.geometric_range_m + clock_m + ambiguity_m,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                    glonass_channel: None,
                    signals: None,
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
            tropo_gradient_north_m: 0.0,
            tropo_gradient_east_m: 0.0,
            residual_ionosphere_m: BTreeMap::new(),
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
    fn public_float_solve_reports_unestimated_variance_at_zero_redundancy() {
        let (source, epoch, state) = synthetic_case(4, None);
        let solution = solve_float_epoch(&source, epoch, state, fde_config())
            .expect("exactly determined public float solve");
        assert!(!solution.posterior_variance_factor_estimated);
        assert_eq!(
            solution.posterior_variance_factor.to_bits(),
            1.0f64.to_bits()
        );
        assert_eq!(
            solution.position_covariance,
            solution.formal_position_covariance
        );
        assert_eq!(
            solution.position_covariance_scale_factor.to_bits(),
            1.0f64.to_bits()
        );
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
            None,
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
            geometry_for_solution(&source, &epoch, &solution, solve_tropo, None).expect("geometry");
        let without_ztd =
            per_satellite_statistics(&solution.residuals_m, &geometry.rows, RaimConfig::default())
                .expect("without ztd");
        assert_eq!(without_ztd.most_likely_fault.as_deref(), Some("G05"));

        let raim = raim_for_solution(
            &source,
            &epoch,
            &solution,
            solve_tropo,
            None,
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
    fn unit_leverage_code_row_preserves_detection_and_omits_protection_levels() {
        let (mut source, epoch, state) = synthetic_case(6, None);
        let solve_config = fde_config();
        let tropo = solve_config.tropo;
        let solution =
            solve_float_epoch(&source, epoch.clone(), state, solve_config).expect("float solve");
        let receiver = solution.position_m;
        let planar = [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0]];
        for (index, direction) in planar.iter().enumerate() {
            let prn = index as u8 + 1;
            let id = GnssSatelliteId::new(GnssSystem::Gps, prn).unwrap();
            source.states.insert(
                id,
                [
                    receiver[0] + 20_000_000.0 * direction[0],
                    receiver[1] + 20_000_000.0 * direction[1],
                    receiver[2],
                ],
            );
        }
        source.states.insert(
            GnssSatelliteId::new(GnssSystem::Gps, 6).unwrap(),
            [receiver[0], receiver[1], receiver[2] + 20_000_000.0],
        );
        let raim = raim_for_solution(
            &source,
            &epoch,
            &solution,
            tropo,
            None,
            RaimConfig::default(),
        )
        .expect("detection result survives unit leverage");
        assert_ne!(raim.status, RaimStatus::NotEnoughRedundancy);
        assert_eq!(raim.hpl_m, None);
        assert_eq!(raim.vpl_m, None);
        let galileo_statistic = raim
            .satellite_statistics
            .iter()
            .find(|statistic| statistic.satellite_id == "G06")
            .unwrap();
        assert!(!galileo_statistic.code_testable);
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

        // Each phase row is the only observer of its ambiguity, which absorbs the outlier:
        // the rows are reported untestable and the outlier raises no statistic.
        let identification =
            per_satellite_statistics(&residuals, &test_geometry(), RaimConfig::default())
                .expect("untestable rows are not an error");
        for stat in &identification.statistics {
            assert!(!stat.phase_testable, "{}", stat.satellite_id);
            assert_eq!(stat.phase, 0.0);
            assert_eq!(stat.statistic, 0.0);
        }
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
    fn solver_protection_levels_use_the_complete_ztd_design() {
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
            geometry_for_solution(&source, &epoch, &solution, solve_tropo, None).expect("geometry");
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
            None,
            RaimConfig::default(),
        )
        .expect("RAIM with ZTD protection levels");

        assert!(with_ztd.hpl_m.is_finite() && with_ztd.hpl_m > 0.0);
        assert!(with_ztd.vpl_m.is_finite() && with_ztd.vpl_m > 0.0);
        assert_ne!(with_ztd.hpl_m.to_bits(), without_ztd.hpl_m.to_bits());
        assert_ne!(with_ztd.vpl_m.to_bits(), without_ztd.vpl_m.to_bits());
        // The legacy code-only helper contains the ZTD column but omits the
        // phase rows and ambiguity states.  Solver RAIM must instead use the
        // complete design that produced its detection statistic.
        assert_ne!(raim.hpl_m.expect("HPL").to_bits(), with_ztd.hpl_m.to_bits());
        assert_ne!(raim.vpl_m.expect("VPL").to_bits(), with_ztd.vpl_m.to_bits());
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

    /// An observation the solve leaves out for a missing SSR bias adds no rows, so it does
    /// not count towards the redundancy left after an exclusion. With G06 left out, six
    /// observations leave five solved, the same case as five satellites, and G05 cannot be
    /// excluded.
    #[test]
    fn fde_redundancy_counts_only_observations_the_solve_keeps() {
        let (source, epoch, state) = synthetic_case(6, Some("G05"));
        let mut lookup = super::super::PppCorrectionLookup {
            ssr_code_bias_enabled: true,
            phase_bias_enabled: true,
            ..Default::default()
        };
        for obs in &epoch.observations {
            if obs.satellite_id != "G06" {
                let key = (obs.sat, 0, obs.ambiguity_id.clone());
                lookup.ssr_code_bias_m.insert(key.clone(), 0.0);
                lookup.phase_bias_m.insert(key, 0.0);
            }
        }
        let mut config = fde_config();
        config.corrections.ppp = lookup;

        let fde = fde_float_epoch(&source, epoch, state, config, fde_raim_config()).expect("FDE");

        assert_eq!(fde.solution.ssr_bias_exclusions.len(), 1);
        assert_eq!(fde.solution.ssr_bias_exclusions[0].satellite_id, "G06");
        assert!(fde.raim.detected);
        assert_eq!(fde.status, RaimFdeStatus::CannotExclude);
        assert!(fde.excluded_sats.is_empty());
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
