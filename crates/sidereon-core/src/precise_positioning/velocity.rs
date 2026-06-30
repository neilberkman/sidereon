//! Single-epoch receiver velocity range-rate model.
//!
//! This leaf module starts with the deterministic Doppler/range-rate geometry
//! shared by the later least-squares driver: each row compares a measured
//! satellite range rate with the line-of-sight projection of satellite velocity,
//! receiver velocity, receiver clock drift, and satellite clock drift.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! use sidereon_core::precise_positioning::{
//!     predict_range_rate_m_s, solve_velocity, ReceiverVelocityState, VelocityConfig,
//!     VelocityObservation,
//! };
//! use sidereon_core::{GnssSatelliteId, GnssSystem};
//!
//! let receiver_position_m = [0.0, 0.0, 0.0];
//! let receiver_velocity_m_s = [3.0, -2.0, 0.5];
//! let clock_drift_m_s = 0.25;
//! let satellite_positions_m = [
//!     [20_200_000.0, 0.0, 0.0],
//!     [0.0, 21_100_000.0, 0.0],
//!     [0.0, 0.0, 22_300_000.0],
//!     [-20_900_000.0, 19_700_000.0, 21_500_000.0],
//!     [18_600_000.0, -20_400_000.0, 23_100_000.0],
//! ];
//! let observations = satellite_positions_m
//!     .iter()
//!     .enumerate()
//!     .map(|(idx, satellite_position_m)| {
//!         let mut observation = VelocityObservation {
//!             sat: GnssSatelliteId::new(GnssSystem::Gps, (idx + 1) as u8)
//!                 .expect("valid satellite id"),
//!             satellite_position_m: *satellite_position_m,
//!             satellite_velocity_m_s: [0.0, 0.0, 0.0],
//!             measured_range_rate_m_s: 0.0,
//!             sigma_m_s: 0.1,
//!             satellite_clock_drift_m_s: 0.0,
//!         };
//!         observation.measured_range_rate_m_s = predict_range_rate_m_s(
//!             &observation,
//!             ReceiverVelocityState {
//!                 position_m: receiver_position_m,
//!                 velocity_m_s: receiver_velocity_m_s,
//!                 clock_drift_m_s,
//!             },
//!         )
//!         .expect("nonzero synthetic line of sight")
//!         .range_rate_m_s;
//!         observation
//!     })
//!     .collect::<Vec<_>>();
//!
//! let solution = solve_velocity(
//!     &observations,
//!     receiver_position_m,
//!     VelocityConfig::default(),
//! )?;
//! assert!((solution.velocity_m_s[0] - receiver_velocity_m_s[0]).abs() < 1.0e-9);
//! assert!((solution.clock_drift_m_s - clock_drift_m_s).abs() < 1.0e-9);
//! # Ok(())
//! # }
//! ```

use crate::astro::math::robust::{huber_weight, mad_scale, HUBER_K};
use crate::astro::math::vec3::{dot3, sub3, unit3};
use crate::estimation::recipe::NormalRecipe;
use crate::estimation::substrate::normal::NormalAssembler;
use crate::estimation::substrate::rows::ResidualRow;
use crate::validate::{self, FieldError};
use crate::GnssSatelliteId;

const DEFAULT_MINIMUM_OBSERVATIONS: usize = 4;
const DEFAULT_ROBUST_SCALE_FLOOR_M_S: f64 = 0.05;
const DEFAULT_ROBUST_MAX_OUTER: usize = 2;
const DEFAULT_ROBUST_OUTER_TOL_M_S: f64 = 1.0e-6;
const VELOCITY_UNKNOWNS: usize = 4;
const VELOCITY_NORMAL_ASSEMBLER: NormalAssembler =
    NormalAssembler::new(NormalRecipe::PppDenseLastTie);

/// Configuration for a single-epoch Doppler/range-rate velocity solve.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityConfig {
    /// Minimum number of observations required for the four velocity unknowns.
    pub minimum_observations: usize,
    /// Optional Huber/MAD robust reweighting for range-rate outliers.
    pub robust: Option<VelocityRobustConfig>,
}

impl Default for VelocityConfig {
    fn default() -> Self {
        Self {
            minimum_observations: DEFAULT_MINIMUM_OBSERVATIONS,
            robust: None,
        }
    }
}

/// Opt-in robust reweighting configuration for the velocity solve.
///
/// When [`VelocityConfig::robust`] is `Some(_)`, the solver first runs the base
/// weighted least-squares solve, then recomputes post-fit range-rate residuals
/// and resolves with each inverse-sigma row weight multiplied by
/// `sqrt(huber(residual / scale))`, where `scale` is a floored MAD estimate.
/// The default velocity solve leaves this disabled.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityRobustConfig {
    /// Huber tuning constant `k`; scaled residuals below this keep full weight.
    pub huber_k: f64,
    /// Floor on the MAD range-rate residual scale, in meters per second.
    pub scale_floor_m_s: f64,
    /// Maximum total outer solves: the base solve plus reweighted resolves.
    pub max_outer: usize,
    /// Outer-loop velocity-and-clock step tolerance, in meters per second.
    pub outer_tol_m_s: f64,
}

impl Default for VelocityRobustConfig {
    fn default() -> Self {
        Self {
            huber_k: HUBER_K,
            scale_floor_m_s: DEFAULT_ROBUST_SCALE_FLOOR_M_S,
            max_outer: DEFAULT_ROBUST_MAX_OUTER,
            outer_tol_m_s: DEFAULT_ROBUST_OUTER_TOL_M_S,
        }
    }
}

/// Receiver state used by the range-rate prediction model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReceiverVelocityState {
    /// Receiver position in the same ECEF/ECI frame as the satellite state, in meters.
    pub position_m: [f64; 3],
    /// Receiver velocity in the same ECEF/ECI frame as the satellite state, in meters per second.
    pub velocity_m_s: [f64; 3],
    /// Receiver clock drift expressed as an equivalent range-rate bias, in meters per second.
    pub clock_drift_m_s: f64,
}

impl ReceiverVelocityState {
    /// Construct a receiver state with zero velocity and zero clock drift.
    pub const fn stationary_at(position_m: [f64; 3]) -> Self {
        Self {
            position_m,
            velocity_m_s: [0.0; 3],
            clock_drift_m_s: 0.0,
        }
    }
}

/// One Doppler-derived range-rate observation for a satellite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VelocityObservation {
    /// Satellite identifier.
    pub sat: GnssSatelliteId,
    /// Satellite position in the same ECEF/ECI frame as the receiver state, in meters.
    pub satellite_position_m: [f64; 3],
    /// Satellite velocity in the same ECEF/ECI frame as the receiver state, in meters per second.
    pub satellite_velocity_m_s: [f64; 3],
    /// Measured pseudorange rate from Doppler, in meters per second.
    pub measured_range_rate_m_s: f64,
    /// One-sigma uncertainty of the measured range rate, in meters per second.
    pub sigma_m_s: f64,
    /// Satellite clock drift expressed as an equivalent range-rate bias, in meters per second.
    pub satellite_clock_drift_m_s: f64,
}

/// Predicted range-rate geometry for one satellite observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeRatePrediction {
    /// Unit line-of-sight vector from receiver to satellite.
    pub los_unit: [f64; 3],
    /// Predicted range rate in meters per second.
    pub range_rate_m_s: f64,
}

/// Receiver velocity solve result for one Doppler/range-rate epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct VelocitySolution {
    /// Receiver velocity in the same frame as the observations, in meters per second.
    pub velocity_m_s: [f64; 3],
    /// Receiver clock drift expressed as an equivalent range-rate bias, in meters per second.
    pub clock_drift_m_s: f64,
    /// Root-mean-square post-fit range-rate residual in meters per second.
    pub residual_rms_m_s: f64,
    /// Satellite identifiers retained in the least-squares solution.
    pub used_sats: Vec<GnssSatelliteId>,
    /// Satellite identifiers that received a robust Huber multiplier below one.
    pub downweighted_sats: Vec<GnssSatelliteId>,
    /// Number of robust reweighted resolves after the base solve.
    pub robust_iterations: usize,
    /// Final robust residual scale in meters per second, if robust reweighting ran.
    pub robust_scale_m_s: Option<f64>,
}

/// Error returned by the Doppler/range-rate velocity least-squares solve.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VelocitySolveError {
    /// Fewer observations than velocity unknowns were supplied.
    TooFewObservations {
        /// Minimum number of observations required by the solve.
        required: usize,
        /// Number of observations supplied by the caller.
        actual: usize,
    },
    /// A boundary input was malformed before the normal equations could be solved.
    InvalidInput {
        /// Name of the malformed field.
        field: &'static str,
        /// Stable validation reason.
        reason: &'static str,
    },
    /// The design matrix does not provide full-rank velocity and clock-drift geometry.
    SingularGeometry,
}

impl core::fmt::Display for VelocitySolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TooFewObservations { required, actual } => write!(
                f,
                "too few velocity observations: required {required}, got {actual}"
            ),
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid velocity input {field}: {reason}")
            }
            Self::SingularGeometry => {
                write!(f, "velocity geometry is singular")
            }
        }
    }
}

impl std::error::Error for VelocitySolveError {}

/// Predict the range rate for one observation and receiver state.
///
/// Returns `None` when the receiver and satellite positions are coincident, so
/// a line-of-sight unit vector cannot be formed.
pub fn predict_range_rate_m_s(
    observation: &VelocityObservation,
    receiver: ReceiverVelocityState,
) -> Option<RangeRatePrediction> {
    if !finite3(observation.satellite_position_m)
        || !finite3(observation.satellite_velocity_m_s)
        || !finite3(receiver.position_m)
        || !finite3(receiver.velocity_m_s)
        || !receiver.clock_drift_m_s.is_finite()
        || !observation.satellite_clock_drift_m_s.is_finite()
    {
        return None;
    }
    let los_unit = unit3(sub3(observation.satellite_position_m, receiver.position_m))?;
    let relative_velocity_m_s = sub3(observation.satellite_velocity_m_s, receiver.velocity_m_s);
    let range_rate_m_s = dot3(los_unit, relative_velocity_m_s) + receiver.clock_drift_m_s
        - observation.satellite_clock_drift_m_s;
    if !finite3(los_unit) || !range_rate_m_s.is_finite() {
        return None;
    }
    Some(RangeRatePrediction {
        los_unit,
        range_rate_m_s,
    })
}

/// Solve receiver velocity and clock drift from precomputed range-rate observations.
///
/// This low-level least-squares entry point expects one epoch of satellite
/// position/velocity observations and a known receiver position. It builds the
/// range-rate design matrix directly from the observation geometry.
pub fn solve_velocity_least_squares(
    observations: &[VelocityObservation],
    receiver_position_m: [f64; 3],
    config: VelocityConfig,
) -> Result<VelocitySolution, VelocitySolveError> {
    validate_velocity_config(config)?;
    let rows = build_velocity_rows(observations, receiver_position_m, config)?;
    let mut solution = solve_velocity_rows(&rows)?;
    let mut residuals = postfit_residuals_m_s(&rows, solution);
    let mut robust_iterations = 0usize;
    let mut robust_scale_m_s = None;
    let mut robust_multipliers = vec![1.0; rows.len()];

    if let Some(robust) = config.robust {
        for _ in 0..robust.max_outer.saturating_sub(1) {
            let (reweighted_rows, multipliers, scale_m_s) =
                robust_reweighted_rows(&rows, &residuals, robust)?;
            let next_solution = solve_velocity_rows(&reweighted_rows)?;
            let step_m_s = solution_step_norm(solution, next_solution);
            solution = next_solution;
            residuals = postfit_residuals_m_s(&rows, solution);
            robust_iterations += 1;
            robust_scale_m_s = Some(scale_m_s);
            robust_multipliers = multipliers;
            if step_m_s < robust.outer_tol_m_s {
                break;
            }
        }
    }

    assemble_solution(
        observations,
        solution,
        &residuals,
        &robust_multipliers,
        robust_iterations,
        robust_scale_m_s,
    )
}

/// Solve receiver velocity and clock drift for one epoch of range-rate observations.
///
/// The observation values are pseudorange rates in meters per second, such as
/// Doppler converted with the standard GNSS sign convention. Satellite states
/// and the supplied receiver position must be expressed in one consistent frame.
pub fn solve_velocity(
    observations: &[VelocityObservation],
    receiver_position_m: [f64; 3],
    config: VelocityConfig,
) -> Result<VelocitySolution, VelocitySolveError> {
    solve_velocity_least_squares(observations, receiver_position_m, config)
}

fn build_velocity_rows(
    observations: &[VelocityObservation],
    receiver_position_m: [f64; 3],
    config: VelocityConfig,
) -> Result<Vec<ResidualRow>, VelocitySolveError> {
    let required = config.minimum_observations.max(VELOCITY_UNKNOWNS);
    if observations.len() < required {
        return Err(VelocitySolveError::TooFewObservations {
            required,
            actual: observations.len(),
        });
    }
    validate::finite_vec3(receiver_position_m, "velocity receiver position_m")
        .map_err(invalid_input)?;
    let receiver = ReceiverVelocityState::stationary_at(receiver_position_m);
    let mut rows = Vec::with_capacity(observations.len());
    for observation in observations {
        validate_observation(observation)?;
        let sigma_m_s =
            validate::finite_positive(observation.sigma_m_s, "velocity observation sigma_m_s")
                .map_err(invalid_input)?;
        let prediction = predict_range_rate_m_s(observation, receiver)
            .ok_or(VelocitySolveError::SingularGeometry)?;
        let y = observation.measured_range_rate_m_s - prediction.range_rate_m_s;
        validate::finite(y, "velocity observation residual_m_s").map_err(invalid_input)?;
        let weight = 1.0 / sigma_m_s;
        validate::finite_positive(weight, "velocity observation weight").map_err(invalid_input)?;
        rows.push(ResidualRow {
            h: vec![
                -prediction.los_unit[0],
                -prediction.los_unit[1],
                -prediction.los_unit[2],
                1.0,
            ],
            y,
            weight,
        });
    }
    Ok(rows)
}

fn validate_velocity_config(config: VelocityConfig) -> Result<(), VelocitySolveError> {
    if let Some(robust) = config.robust {
        validate::finite_positive(robust.huber_k, "velocity robust huber_k")
            .map_err(invalid_input)?;
        validate::finite_positive(robust.scale_floor_m_s, "velocity robust scale_floor_m_s")
            .map_err(invalid_input)?;
        validate::finite_positive(robust.outer_tol_m_s, "velocity robust outer_tol_m_s")
            .map_err(invalid_input)?;
        if robust.max_outer == 0 {
            return Err(VelocitySolveError::InvalidInput {
                field: "velocity robust max_outer",
                reason: "not positive",
            });
        }
    }
    Ok(())
}

fn validate_observation(observation: &VelocityObservation) -> Result<(), VelocitySolveError> {
    validate::finite_vec3(
        observation.satellite_position_m,
        "velocity observation satellite_position_m",
    )
    .map_err(invalid_input)?;
    validate::finite_vec3(
        observation.satellite_velocity_m_s,
        "velocity observation satellite_velocity_m_s",
    )
    .map_err(invalid_input)?;
    validate::finite(
        observation.measured_range_rate_m_s,
        "velocity observation measured_range_rate_m_s",
    )
    .map_err(invalid_input)?;
    validate::finite(
        observation.satellite_clock_drift_m_s,
        "velocity observation satellite_clock_drift_m_s",
    )
    .map_err(invalid_input)?;
    Ok(())
}

fn finite3(value: [f64; 3]) -> bool {
    value.iter().all(|x| x.is_finite())
}

fn solve_velocity_rows(rows: &[ResidualRow]) -> Result<[f64; 4], VelocitySolveError> {
    let solution = VELOCITY_NORMAL_ASSEMBLER
        .solve_dense_last_tie(rows.iter().map(ResidualRow::as_weighted), VELOCITY_UNKNOWNS)
        .ok_or(VelocitySolveError::SingularGeometry)?;
    let solution: [f64; 4] = solution
        .try_into()
        .map_err(|_| VelocitySolveError::SingularGeometry)?;
    validate_finite_slice(&solution, "velocity solution")?;
    Ok(solution)
}

fn postfit_residuals_m_s(rows: &[ResidualRow], solution: [f64; 4]) -> Vec<f64> {
    rows.iter()
        .map(|row| {
            row.y
                - (row.h[0] * solution[0]
                    + row.h[1] * solution[1]
                    + row.h[2] * solution[2]
                    + row.h[3] * solution[3])
        })
        .collect()
}

fn robust_reweighted_rows(
    rows: &[ResidualRow],
    residuals_m_s: &[f64],
    robust: VelocityRobustConfig,
) -> Result<(Vec<ResidualRow>, Vec<f64>, f64), VelocitySolveError> {
    let scale_m_s =
        mad_scale(residuals_m_s, robust.scale_floor_m_s).map_err(invalid_robust_input)?;
    let multipliers: Vec<f64> = residuals_m_s
        .iter()
        .map(|residual| huber_weight(residual / scale_m_s, robust.huber_k))
        .collect();
    validate_finite_slice(&multipliers, "velocity robust multiplier")?;
    let rows = rows
        .iter()
        .zip(multipliers.iter())
        .map(|(row, multiplier)| {
            let mut row = row.clone();
            row.weight *= multiplier.sqrt();
            validate::finite_positive(row.weight, "velocity robust row weight")
                .map_err(invalid_input)?;
            Ok(row)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((rows, multipliers, scale_m_s))
}

fn solution_step_norm(previous: [f64; 4], next: [f64; 4]) -> f64 {
    ((next[0] - previous[0]).powi(2)
        + (next[1] - previous[1]).powi(2)
        + (next[2] - previous[2]).powi(2)
        + (next[3] - previous[3]).powi(2))
    .sqrt()
}

fn assemble_solution(
    observations: &[VelocityObservation],
    solution: [f64; 4],
    residuals_m_s: &[f64],
    robust_multipliers: &[f64],
    robust_iterations: usize,
    robust_scale_m_s: Option<f64>,
) -> Result<VelocitySolution, VelocitySolveError> {
    validate_finite_slice(&solution, "velocity solution")?;
    validate_finite_slice(residuals_m_s, "velocity residual_m_s")?;
    validate_finite_slice(robust_multipliers, "velocity robust multiplier")?;
    if let Some(scale_m_s) = robust_scale_m_s {
        validate::finite_positive(scale_m_s, "velocity robust scale_m_s").map_err(invalid_input)?;
    }
    let residual_rms_m_s = rms(residuals_m_s);
    validate::finite(residual_rms_m_s, "velocity residual_rms_m_s").map_err(invalid_input)?;
    Ok(VelocitySolution {
        velocity_m_s: [solution[0], solution[1], solution[2]],
        clock_drift_m_s: solution[3],
        residual_rms_m_s,
        used_sats: observations.iter().map(|obs| obs.sat).collect(),
        downweighted_sats: observations
            .iter()
            .zip(robust_multipliers.iter())
            .filter_map(|(observation, multiplier)| (*multiplier < 1.0).then_some(observation.sat))
            .collect(),
        robust_iterations,
        robust_scale_m_s,
    })
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn validate_finite_slice(values: &[f64], field: &'static str) -> Result<(), VelocitySolveError> {
    for value in values {
        validate::finite(*value, field).map_err(invalid_input)?;
    }
    Ok(())
}

fn invalid_input(error: FieldError) -> VelocitySolveError {
    VelocitySolveError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn invalid_robust_input(error: crate::astro::math::robust::RobustError) -> VelocitySolveError {
    VelocitySolveError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GnssSystem;

    fn sat() -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid satellite id")
    }

    fn sat_prn(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn synthetic_observations(
        receiver_position_m: [f64; 3],
        receiver_velocity_m_s: [f64; 3],
        clock_drift_m_s: f64,
    ) -> Vec<VelocityObservation> {
        let satellite_positions_m = [
            [20_200_000.0, 0.0, 0.0],
            [0.0, 21_100_000.0, 0.0],
            [0.0, 0.0, 22_300_000.0],
            [-20_900_000.0, 19_700_000.0, 21_500_000.0],
            [18_600_000.0, -20_400_000.0, 23_100_000.0],
            [-18_800_000.0, -21_200_000.0, 19_400_000.0],
            [21_900_000.0, 18_300_000.0, -20_700_000.0],
            [-22_100_000.0, 17_900_000.0, -18_500_000.0],
        ];
        let satellite_velocities_m_s = [
            [120.0, -40.0, 30.0],
            [-80.0, 90.0, 10.0],
            [20.0, -70.0, 105.0],
            [35.0, 45.0, -95.0],
            [-55.0, 65.0, 75.0],
            [70.0, -85.0, 40.0],
            [-25.0, -45.0, 95.0],
            [45.0, 55.0, -65.0],
        ];
        satellite_positions_m
            .into_iter()
            .zip(satellite_velocities_m_s)
            .enumerate()
            .map(|(idx, (satellite_position_m, satellite_velocity_m_s))| {
                let satellite_clock_drift_m_s = 0.01 * idx as f64;
                let mut observation = VelocityObservation {
                    sat: sat_prn((idx + 1) as u8),
                    satellite_position_m,
                    satellite_velocity_m_s,
                    measured_range_rate_m_s: 0.0,
                    sigma_m_s: 0.1 + 0.05 * idx as f64,
                    satellite_clock_drift_m_s,
                };
                observation.measured_range_rate_m_s = predict_range_rate_m_s(
                    &observation,
                    ReceiverVelocityState {
                        position_m: receiver_position_m,
                        velocity_m_s: receiver_velocity_m_s,
                        clock_drift_m_s,
                    },
                )
                .expect("synthetic line of sight")
                .range_rate_m_s;
                observation
            })
            .collect()
    }

    #[test]
    fn velocity_types_construct() {
        let config = VelocityConfig::default();
        assert_eq!(config.minimum_observations, 4);
        assert_eq!(config.robust, None);

        let robust = VelocityRobustConfig::default();
        assert_eq!(robust.huber_k, HUBER_K);
        assert_eq!(robust.max_outer, 2);

        let receiver = ReceiverVelocityState {
            position_m: [1.0, 2.0, 3.0],
            velocity_m_s: [0.1, -0.2, 0.3],
            clock_drift_m_s: 0.4,
        };
        assert_eq!(receiver.position_m, [1.0, 2.0, 3.0]);
        assert_eq!(
            ReceiverVelocityState::stationary_at([4.0, 5.0, 6.0]),
            ReceiverVelocityState {
                position_m: [4.0, 5.0, 6.0],
                velocity_m_s: [0.0; 3],
                clock_drift_m_s: 0.0,
            }
        );

        let observation = VelocityObservation {
            sat: sat(),
            satellite_position_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
            satellite_velocity_m_s: [120.0, -30.0, 80.0],
            measured_range_rate_m_s: -425.0,
            sigma_m_s: 0.08,
            satellite_clock_drift_m_s: 0.02,
        };
        assert_eq!(observation.sat, sat());
        assert_eq!(observation.sigma_m_s, 0.08);
    }

    #[test]
    fn predicted_range_rate_zero_receiver_matches_los_projected_satellite_velocity() {
        let observation = VelocityObservation {
            sat: sat(),
            satellite_position_m: [3.0, 4.0, 0.0],
            satellite_velocity_m_s: [6.0, 8.0, 10.0],
            measured_range_rate_m_s: 0.0,
            sigma_m_s: 1.0,
            satellite_clock_drift_m_s: 0.0,
        };
        let receiver = ReceiverVelocityState::stationary_at([0.0, 0.0, 0.0]);

        let prediction =
            predict_range_rate_m_s(&observation, receiver).expect("nonzero line of sight");

        let expected = (3.0 / 5.0) * observation.satellite_velocity_m_s[0]
            + (4.0 / 5.0) * observation.satellite_velocity_m_s[1];
        assert!((prediction.range_rate_m_s - expected).abs() < 1.0e-12);
        assert!((prediction.los_unit[0] - 3.0 / 5.0).abs() < 1.0e-12);
        assert!((prediction.los_unit[1] - 4.0 / 5.0).abs() < 1.0e-12);
        assert_eq!(prediction.los_unit[2], 0.0);
    }

    #[test]
    fn predicted_range_rate_rejects_non_finite_state() {
        let observation = VelocityObservation {
            sat: sat(),
            satellite_position_m: [f64::NAN, 4.0, 0.0],
            satellite_velocity_m_s: [6.0, 8.0, 10.0],
            measured_range_rate_m_s: 0.0,
            sigma_m_s: 1.0,
            satellite_clock_drift_m_s: 0.0,
        };
        let receiver = ReceiverVelocityState::stationary_at([0.0, 0.0, 0.0]);

        assert_eq!(predict_range_rate_m_s(&observation, receiver), None);
    }

    #[test]
    fn design_row_matches_velocity_linearization() {
        let observation = VelocityObservation {
            sat: sat(),
            satellite_position_m: [3.0, 4.0, 0.0],
            satellite_velocity_m_s: [6.0, 8.0, 10.0],
            measured_range_rate_m_s: 42.0,
            sigma_m_s: 0.25,
            satellite_clock_drift_m_s: 0.5,
        };

        let rows = build_velocity_rows(
            &[observation; 4],
            [0.0, 0.0, 0.0],
            VelocityConfig::default(),
        )
        .expect("rows");

        assert!((rows[0].h[0] + 3.0 / 5.0).abs() < 1.0e-15);
        assert!((rows[0].h[1] + 4.0 / 5.0).abs() < 1.0e-15);
        assert_eq!(rows[0].h[2], -0.0);
        assert_eq!(rows[0].h[3], 1.0);
        let predicted_at_zero = dot(
            [3.0 / 5.0, 4.0 / 5.0, 0.0],
            observation.satellite_velocity_m_s,
        ) - observation.satellite_clock_drift_m_s;
        assert!(
            (rows[0].y - (observation.measured_range_rate_m_s - predicted_at_zero)).abs() < 1.0e-12
        );
        assert_eq!(rows[0].weight, 4.0);
    }

    #[test]
    fn weighted_least_squares_recovers_clean_synthetic_velocity() {
        let receiver_position_m = [3_512_900.0, 780_500.0, 5_248_700.0];
        let receiver_velocity_m_s = [12.25, -3.5, 0.75];
        let clock_drift_m_s = -0.35;
        let observations =
            synthetic_observations(receiver_position_m, receiver_velocity_m_s, clock_drift_m_s);

        let solution = solve_velocity_least_squares(
            &observations,
            receiver_position_m,
            VelocityConfig::default(),
        )
        .expect("velocity solution");

        for (got, expected) in solution
            .velocity_m_s
            .iter()
            .zip(receiver_velocity_m_s.iter())
        {
            assert!((got - expected).abs() < 1.0e-9, "{got} != {expected}");
        }
        assert!((solution.clock_drift_m_s - clock_drift_m_s).abs() < 1.0e-9);
        assert!(solution.residual_rms_m_s < 1.0e-9);
        assert_eq!(solution.used_sats.len(), observations.len());
        assert_eq!(solution.used_sats[0], sat_prn(1));
        assert_eq!(solution.downweighted_sats, Vec::<GnssSatelliteId>::new());
        assert_eq!(solution.robust_iterations, 0);
        assert_eq!(solution.robust_scale_m_s, None);
    }

    #[test]
    fn singular_or_underdetermined_geometry_returns_error() {
        let receiver_position_m = [0.0, 0.0, 0.0];
        let mut observations = synthetic_observations(receiver_position_m, [1.0, 2.0, 3.0], 0.4);
        let too_few = observations[..3].to_vec();
        assert_eq!(
            solve_velocity_least_squares(&too_few, receiver_position_m, VelocityConfig::default()),
            Err(VelocitySolveError::TooFewObservations {
                required: 4,
                actual: 3
            })
        );

        for (idx, observation) in observations.iter_mut().take(4).enumerate() {
            observation.satellite_position_m = [20_000_000.0 + idx as f64, 0.0, 0.0];
        }

        assert_eq!(
            solve_velocity_least_squares(
                &observations[..4],
                receiver_position_m,
                VelocityConfig::default()
            ),
            Err(VelocitySolveError::SingularGeometry)
        );
    }

    #[test]
    fn rejects_non_finite_velocity_inputs() {
        let receiver_position_m = [0.0, 0.0, 0.0];
        let mut observations = synthetic_observations(receiver_position_m, [1.0, 2.0, 3.0], 0.4);
        observations[0].measured_range_rate_m_s = f64::NAN;

        assert_eq!(
            solve_velocity_least_squares(
                &observations,
                receiver_position_m,
                VelocityConfig::default()
            ),
            Err(VelocitySolveError::InvalidInput {
                field: "velocity observation measured_range_rate_m_s",
                reason: "not finite"
            })
        );

        observations[0].measured_range_rate_m_s = 0.0;
        observations[0].sigma_m_s = 0.0;
        assert_eq!(
            solve_velocity_least_squares(
                &observations,
                receiver_position_m,
                VelocityConfig::default()
            ),
            Err(VelocitySolveError::InvalidInput {
                field: "velocity observation sigma_m_s",
                reason: "not positive"
            })
        );

        observations[0].sigma_m_s = f64::from_bits(1);
        assert_eq!(
            solve_velocity_least_squares(
                &observations,
                receiver_position_m,
                VelocityConfig::default()
            ),
            Err(VelocitySolveError::InvalidInput {
                field: "velocity observation weight",
                reason: "not finite"
            })
        );
        observations[0].sigma_m_s = 0.1;

        let mut config = VelocityConfig {
            robust: Some(VelocityRobustConfig {
                huber_k: f64::INFINITY,
                ..VelocityRobustConfig::default()
            }),
            ..VelocityConfig::default()
        };
        assert_eq!(
            solve_velocity_least_squares(&observations, receiver_position_m, config),
            Err(VelocitySolveError::InvalidInput {
                field: "velocity robust huber_k",
                reason: "not finite"
            })
        );

        config.robust = Some(VelocityRobustConfig {
            scale_floor_m_s: -1.0,
            ..VelocityRobustConfig::default()
        });
        assert_eq!(
            solve_velocity_least_squares(&observations, receiver_position_m, config),
            Err(VelocitySolveError::InvalidInput {
                field: "velocity robust scale_floor_m_s",
                reason: "not positive"
            })
        );
    }

    #[test]
    fn robust_velocity_rejects_zero_outer_limit_only_when_enabled() {
        let receiver_position_m = [3_512_900.0, 780_500.0, 5_248_700.0];
        let observations = synthetic_observations(receiver_position_m, [1.0, 2.0, 3.0], 0.4);

        let mut robust = VelocityRobustConfig {
            max_outer: 0,
            ..VelocityRobustConfig::default()
        };
        assert_eq!(
            solve_velocity_least_squares(
                &observations,
                receiver_position_m,
                VelocityConfig {
                    robust: Some(robust),
                    ..VelocityConfig::default()
                }
            ),
            Err(VelocitySolveError::InvalidInput {
                field: "velocity robust max_outer",
                reason: "not positive"
            })
        );

        robust.max_outer = 1;
        let robust_solution = solve_velocity_least_squares(
            &observations,
            receiver_position_m,
            VelocityConfig {
                robust: Some(robust),
                ..VelocityConfig::default()
            },
        )
        .expect("max_outer=1 robust velocity solution");
        assert_eq!(robust_solution.robust_iterations, 0);

        let non_robust = solve_velocity_least_squares(
            &observations,
            receiver_position_m,
            VelocityConfig::default(),
        )
        .expect("non-robust velocity solution");
        assert_eq!(non_robust.robust_iterations, 0);
    }

    #[test]
    fn robust_reweighting_downweights_doppler_outlier() {
        let receiver_position_m = [3_512_900.0, 780_500.0, 5_248_700.0];
        let receiver_velocity_m_s = [12.25, -3.5, 0.75];
        let clock_drift_m_s = -0.35;
        let observations =
            synthetic_observations(receiver_position_m, receiver_velocity_m_s, clock_drift_m_s);
        let clean = solve_velocity_least_squares(
            &observations,
            receiver_position_m,
            VelocityConfig::default(),
        )
        .expect("clean solution");

        let mut corrupted = observations.clone();
        let outlier_sat = corrupted[3].sat;
        corrupted[3].measured_range_rate_m_s += 75.0;

        let bare = solve_velocity_least_squares(
            &corrupted,
            receiver_position_m,
            VelocityConfig::default(),
        )
        .expect("bare outlier solution");
        let robust = solve_velocity_least_squares(
            &corrupted,
            receiver_position_m,
            VelocityConfig {
                robust: Some(VelocityRobustConfig {
                    scale_floor_m_s: 0.05,
                    max_outer: 5,
                    outer_tol_m_s: 1.0e-12,
                    ..VelocityRobustConfig::default()
                }),
                ..VelocityConfig::default()
            },
        )
        .expect("robust outlier solution");

        let bare_error = velocity_error_m_s(&bare.velocity_m_s, &clean.velocity_m_s);
        let robust_error = velocity_error_m_s(&robust.velocity_m_s, &clean.velocity_m_s);
        assert!(
            robust_error < bare_error,
            "robust_error={robust_error}, bare_error={bare_error}"
        );
        assert!(
            robust_error < 1.0,
            "robust velocity should stay close to truth, got {robust_error}"
        );
        assert!(robust.downweighted_sats.contains(&outlier_sat));
        assert!(robust.robust_iterations >= 1);
        assert!(robust.robust_scale_m_s.is_some());
    }

    #[test]
    fn robust_reweighting_applies_sqrt_huber_multiplier_to_velocity_rows() {
        let receiver_position_m = [3_512_900.0, 780_500.0, 5_248_700.0];
        let receiver_velocity_m_s = [12.25, -3.5, 0.75];
        let clock_drift_m_s = -0.35;
        let mut observations =
            synthetic_observations(receiver_position_m, receiver_velocity_m_s, clock_drift_m_s);
        observations[3].measured_range_rate_m_s += 75.0;

        let robust = VelocityRobustConfig {
            scale_floor_m_s: 0.05,
            max_outer: 2,
            outer_tol_m_s: 1.0e-12,
            ..VelocityRobustConfig::default()
        };
        let rows = build_velocity_rows(
            &observations,
            receiver_position_m,
            VelocityConfig {
                robust: Some(robust),
                ..VelocityConfig::default()
            },
        )
        .expect("velocity rows");
        let base_solution = solve_velocity_rows(&rows).expect("base velocity solution");
        let residuals = postfit_residuals_m_s(&rows, base_solution);
        let scale_m_s =
            mad_scale(&residuals, robust.scale_floor_m_s).expect("valid robust residual scale");
        let multipliers: Vec<f64> = residuals
            .iter()
            .map(|residual| huber_weight(residual / scale_m_s, robust.huber_k))
            .collect();
        assert!(multipliers.iter().any(|multiplier| *multiplier < 1.0));

        let intended_rows = scaled_rows(&rows, &multipliers, f64::sqrt);
        let squared_weight_rows = scaled_rows(&rows, &multipliers, core::convert::identity);
        let intended_solution =
            solve_velocity_rows(&intended_rows).expect("intended robust solution");
        let squared_weight_solution =
            solve_velocity_rows(&squared_weight_rows).expect("squared-weight solution");

        let robust_solution = solve_velocity_least_squares(
            &observations,
            receiver_position_m,
            VelocityConfig {
                robust: Some(robust),
                ..VelocityConfig::default()
            },
        )
        .expect("public robust solution");

        assert_eq!(robust_solution.robust_iterations, 1);
        assert_solution_close(
            solution_vector(&robust_solution),
            intended_solution,
            1.0e-10,
        );
        assert!(
            solution_delta(solution_vector(&robust_solution), squared_weight_solution) > 1.0e-3,
            "sqrt-Huber solution should differ from the old squared-weight behavior"
        );
    }

    #[test]
    fn public_solve_velocity_driver_matches_low_level_solve() {
        let receiver_position_m = [3_512_900.0, 780_500.0, 5_248_700.0];
        let observations = synthetic_observations(receiver_position_m, [1.0, 2.0, 3.0], 0.4);

        assert_eq!(
            solve_velocity(
                &observations,
                receiver_position_m,
                VelocityConfig::default()
            ),
            solve_velocity_least_squares(
                &observations,
                receiver_position_m,
                VelocityConfig::default()
            )
        );
    }

    fn velocity_error_m_s(got: &[f64; 3], expected: &[f64; 3]) -> f64 {
        ((got[0] - expected[0]).powi(2)
            + (got[1] - expected[1]).powi(2)
            + (got[2] - expected[2]).powi(2))
        .sqrt()
    }

    fn scaled_rows(
        rows: &[ResidualRow],
        multipliers: &[f64],
        scale_multiplier: impl Fn(f64) -> f64,
    ) -> Vec<ResidualRow> {
        rows.iter()
            .zip(multipliers)
            .map(|(row, multiplier)| {
                let mut row = row.clone();
                row.weight *= scale_multiplier(*multiplier);
                row
            })
            .collect()
    }

    fn solution_vector(solution: &VelocitySolution) -> [f64; 4] {
        [
            solution.velocity_m_s[0],
            solution.velocity_m_s[1],
            solution.velocity_m_s[2],
            solution.clock_drift_m_s,
        ]
    }

    fn assert_solution_close(actual: [f64; 4], expected: [f64; 4], tolerance: f64) {
        for idx in 0..4 {
            assert!(
                (actual[idx] - expected[idx]).abs() <= tolerance,
                "solution[{idx}] {} != {}",
                actual[idx],
                expected[idx]
            );
        }
    }

    fn solution_delta(a: [f64; 4], b: [f64; 4]) -> f64 {
        ((a[0] - b[0]).powi(2)
            + (a[1] - b[1]).powi(2)
            + (a[2] - b[2]).powi(2)
            + (a[3] - b[3]).powi(2))
        .sqrt()
    }
}
