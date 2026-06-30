//! Kinematic PPP EKF public state and configuration types.

use std::collections::{BTreeMap, BTreeSet};

use crate::ambiguity::AmbiguityId;
use crate::astro::math::linear::{invert_matrix_last_tie, matmul, matrix_sub, transpose};
use crate::estimation::recipe::NormalRecipe;
use crate::observables::ObservableEphemerisSource;

use super::rows::{build_rows, AmbiguityBinding, PppRowError};
use super::{
    estimates_ztd, FloatEpoch, FloatSolveError, FloatState, MeasurementWeights, MissingCorrection,
    NoEphemerisReason, RangeCorrections, TroposphereOptions,
};

const BASE_STATE_DIMENSION: usize = 5;
const CLOCK_INDEX: usize = 3;
const ZTD_INDEX: usize = 4;
// Covariance symmetry validation allows a tiny absolute floor in m^2 plus
// relative scaling for large entries.
const COVARIANCE_SYMMETRY_ABS_TOLERANCE_M2: f64 = 1.0e-9;
const COVARIANCE_SYMMETRY_REL_TOLERANCE: f64 = 1.0e-12;
// PSD validation allows the Cholesky residuals to dip slightly below zero
// from roundoff while still rejecting materially indefinite covariance.
const COVARIANCE_PSD_ABS_TOLERANCE_M2: f64 = 1.0e-9;
const COVARIANCE_PSD_REL_TOLERANCE: f64 = 1.0e-12;

/// Receiver state carried by the sequential kinematic PPP filter.
///
/// The state vector order is ECEF position, receiver clock range bias, zenith
/// wet-delay residual, then carrier-phase float ambiguities in the map's sorted
/// key order.
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicState {
    /// Receiver ECEF position `[x, y, z]`, in metres.
    pub position_m: [f64; 3],
    /// Receiver clock range bias, in metres.
    pub clock_m: f64,
    /// Zenith wet troposphere delay residual, in metres.
    pub ztd_residual_m: f64,
    /// Carrier-phase float ambiguity estimates, in metres, keyed by the static
    /// PPP observation ambiguity id.
    pub ambiguities_m: BTreeMap<String, f64>,
}

impl KinematicState {
    /// Return the covariance/state-vector dimension implied by this state.
    pub fn dimension(&self) -> usize {
        BASE_STATE_DIMENSION + self.ambiguities_m.len()
    }
}

impl Default for KinematicState {
    fn default() -> Self {
        Self {
            position_m: [0.0; 3],
            clock_m: 0.0,
            ztd_residual_m: 0.0,
            ambiguities_m: BTreeMap::new(),
        }
    }
}

/// Position process-noise model used by the kinematic PPP predict step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KinematicPositionProcessNoise {
    /// Position random walk with spectral density in square metres per second.
    RandomWalk {
        /// Random-walk spectral density, in `m^2/s`.
        spectral_density_m2_s: f64,
    },
    /// White-noise acceleration model with spectral density in `m^2/s^3`.
    WhiteNoiseAcceleration {
        /// Acceleration spectral density, in `m^2/s^3`.
        spectral_density_m2_s3: f64,
    },
}

impl Default for KinematicPositionProcessNoise {
    fn default() -> Self {
        Self::RandomWalk {
            spectral_density_m2_s: 1.0,
        }
    }
}

/// Deterministic receiver-position propagation model for the predict step.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum KinematicMotionModel {
    /// Hold the receiver position fixed between measurement updates.
    #[default]
    Hold,
    /// Propagate receiver position with a configured ECEF velocity.
    ConstantVelocity {
        /// Receiver ECEF velocity `[vx, vy, vz]`, in metres per second.
        velocity_m_s: [f64; 3],
    },
}

/// Process-noise spectral densities for the kinematic PPP EKF.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KinematicProcessNoise {
    /// Position process-noise model.
    pub position: KinematicPositionProcessNoise,
    /// Receiver clock white-noise spectral density, in `m^2/s`.
    pub clock_white_m2_s: f64,
    /// Zenith wet-delay random-walk spectral density, in `m^2/s`.
    pub ztd_random_walk_m2_s: f64,
    /// Ambiguity hold spectral density, in `m^2/s`.
    pub ambiguity_hold_m2_s: f64,
}

impl Default for KinematicProcessNoise {
    fn default() -> Self {
        Self {
            position: KinematicPositionProcessNoise::default(),
            clock_white_m2_s: 100.0,
            ztd_random_walk_m2_s: 1.0e-6,
            ambiguity_hold_m2_s: 0.0,
        }
    }
}

/// Configuration for a sequential kinematic PPP EKF solve.
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicConfig {
    /// Initial receiver state estimate.
    pub initial_state: KinematicState,
    /// Initial covariance matrix, in square metres, matching the initial state
    /// vector dimension and ambiguity key order.
    pub initial_covariance_m2: Vec<Vec<f64>>,
    /// Deterministic motion model used during EKF prediction.
    pub motion: KinematicMotionModel,
    /// Process-noise spectral densities used during EKF prediction.
    pub process_noise: KinematicProcessNoise,
    /// Initial variance, in square metres, assigned to ambiguities first seen
    /// after filter initialization.
    pub new_ambiguity_variance_m2: f64,
    /// Code/phase measurement weights reused from the static PPP options.
    pub weights: MeasurementWeights,
    /// Troposphere modelling and ZTD-estimation controls reused from static PPP.
    pub tropo: TroposphereOptions,
    /// Precomputed range corrections reused from static PPP.
    pub corrections: RangeCorrections,
}

/// Summary of one kinematic PPP EKF measurement update.
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicUpdateSummary {
    /// Root-mean-square prefit innovation residual, in metres.
    pub innovation_rms_m: f64,
    /// Public satellite ids used by the measurement update.
    pub used_sats: Vec<String>,
}

/// Per-epoch status returned by the kinematic PPP EKF driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KinematicEpochStatus {
    /// The epoch completed the EKF predict and measurement-update steps.
    Updated,
}

/// One epoch returned by [`solve_kinematic_ppp`].
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicEpochSolution {
    /// Receiver ECEF position `[x, y, z]`, in metres.
    pub position_m: [f64; 3],
    /// Receiver clock range bias, in metres.
    pub clock_m: f64,
    /// Zenith wet troposphere delay residual, in metres.
    pub ztd_residual_m: f64,
    /// Carrier-phase float ambiguity estimates, in metres.
    pub ambiguities_m: BTreeMap<String, f64>,
    /// ECEF position covariance block, in square metres.
    pub position_covariance_m2: [[f64; 3]; 3],
    /// Public satellite ids used by the measurement update.
    pub used_sats: Vec<String>,
    /// Root-mean-square prefit innovation residual, in metres.
    pub innovation_rms_m: f64,
    /// Per-epoch filter status.
    pub status: KinematicEpochStatus,
}

impl Default for KinematicConfig {
    fn default() -> Self {
        Self {
            initial_state: KinematicState::default(),
            initial_covariance_m2: diagonal_covariance(BASE_STATE_DIMENSION, 1.0e8),
            motion: KinematicMotionModel::default(),
            process_noise: KinematicProcessNoise::default(),
            new_ambiguity_variance_m2: 1.0e8,
            weights: MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
        }
    }
}

/// Kinematic PPP EKF solve errors.
#[derive(Debug, Clone, PartialEq)]
pub enum KinematicSolveError {
    /// Ephemeris or satellite clock data were unavailable for an observation.
    NoEphemeris {
        /// Public satellite token, e.g. `"G07"`.
        satellite_id: String,
        /// Specific ephemeris failure reason.
        reason: NoEphemerisReason,
    },
    /// The EKF geometry or innovation covariance was singular.
    SingularGeometry,
    /// A solve option was outside the supported finite range.
    InvalidSolveOption {
        /// Option field name.
        field: &'static str,
        /// Validation failure reason.
        reason: &'static str,
    },
    /// An input state, covariance, epoch, observation, or correction was invalid.
    InvalidInput {
        /// Input field name.
        field: &'static str,
        /// Validation failure reason.
        reason: &'static str,
    },
    /// A required PPP range correction was unavailable for an observation.
    MissingCorrection {
        /// Public satellite token, e.g. `"G07"`.
        satellite_id: String,
        /// Missing correction class.
        correction: MissingCorrection,
    },
}

impl core::fmt::Display for KinematicSolveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NoEphemeris {
                satellite_id,
                reason,
            } => write!(
                f,
                "missing kinematic PPP ephemeris for satellite {satellite_id}: {reason}"
            ),
            Self::SingularGeometry => write!(f, "kinematic PPP geometry is singular"),
            Self::InvalidSolveOption { field, reason } => {
                write!(f, "invalid kinematic PPP solve option {field}: {reason}")
            }
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid kinematic PPP input {field}: {reason}")
            }
            Self::MissingCorrection {
                satellite_id,
                correction,
            } => write!(
                f,
                "missing kinematic PPP correction for satellite {satellite_id}: {correction}"
            ),
        }
    }
}

impl std::error::Error for KinematicSolveError {}

/// Predict one kinematic PPP EKF state and covariance forward by `dt_s`.
///
/// The function updates the state in place, resizes ambiguity states to the
/// sorted `active_ambiguity_ids`, assigns configured variance to new
/// ambiguities, drops inactive ambiguities, applies the configured motion model,
/// and inflates the covariance by process noise scaled by elapsed time.
pub fn predict_kinematic_state(
    state: &mut KinematicState,
    covariance_m2: &mut Vec<Vec<f64>>,
    dt_s: f64,
    active_ambiguity_ids: &[String],
    config: &KinematicConfig,
) -> Result<(), KinematicSolveError> {
    validate_predict_inputs(state, covariance_m2, dt_s, config)?;
    align_ambiguities(
        state,
        covariance_m2,
        active_ambiguity_ids,
        config.new_ambiguity_variance_m2,
    );
    propagate_mean(state, dt_s, config.motion);
    inflate_covariance(covariance_m2, dt_s, config.process_noise);
    symmetrize(covariance_m2);
    Ok(())
}

/// Solve an ordered sequence of epochs with the kinematic PPP EKF.
///
/// The driver initializes from [`KinematicConfig`], then for each epoch performs
/// a time update using elapsed receiver time followed by a measurement update
/// from the shared static PPP row builder.
pub fn solve_kinematic_ppp(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    config: KinematicConfig,
) -> Result<Vec<KinematicEpochSolution>, KinematicSolveError> {
    validate_kinematic_solve_inputs(epochs, &config)?;
    let mut state = config.initial_state.clone();
    let mut covariance_m2 = config.initial_covariance_m2.clone();
    let mut solutions = Vec::with_capacity(epochs.len());
    let mut previous_t_rx_j2000_s = epochs[0].t_rx_j2000_s;

    for (epoch_idx, epoch) in epochs.iter().enumerate() {
        let dt_s = if epoch_idx == 0 {
            0.0
        } else {
            epoch.t_rx_j2000_s - previous_t_rx_j2000_s
        };
        let active_ambiguity_ids = epoch
            .observations
            .iter()
            .map(|obs| obs.ambiguity_id.clone())
            .collect::<Vec<_>>();
        predict_kinematic_state(
            &mut state,
            &mut covariance_m2,
            dt_s,
            &active_ambiguity_ids,
            &config,
        )?;
        let update =
            correct_kinematic_state(source, epoch, &mut state, &mut covariance_m2, &config)?;
        solutions.push(KinematicEpochSolution {
            position_m: state.position_m,
            clock_m: state.clock_m,
            ztd_residual_m: state.ztd_residual_m,
            ambiguities_m: state.ambiguities_m.clone(),
            position_covariance_m2: position_covariance_block(&covariance_m2),
            used_sats: update.used_sats,
            innovation_rms_m: update.innovation_rms_m,
            status: KinematicEpochStatus::Updated,
        });
        previous_t_rx_j2000_s = epoch.t_rx_j2000_s;
    }

    Ok(solutions)
}

/// Correct one kinematic PPP EKF state with a single epoch of code/phase rows.
///
/// This uses the same shared PPP model row builder as the static float solver,
/// then applies an EKF measurement update with diagonal measurement covariance
/// derived from those rows' inverse-sigma weights.
pub fn correct_kinematic_state(
    source: &dyn ObservableEphemerisSource,
    epoch: &FloatEpoch,
    state: &mut KinematicState,
    covariance_m2: &mut Vec<Vec<f64>>,
    config: &KinematicConfig,
) -> Result<KinematicUpdateSummary, KinematicSolveError> {
    validate_state(state)?;
    validate_covariance_shape_and_values(covariance_m2, state.dimension())?;
    validate_measurement_config(config)?;
    let float_state = float_state_from_kinematic(state);
    let corrections = &config.corrections;
    let ctx = super::ModelContext {
        source,
        weights: config.weights,
        tropo: config.tropo,
        corrections,
        normal: NormalRecipe::PppDenseLastTie,
    };
    let ambiguity_ids = state
        .ambiguities_m
        .keys()
        .cloned()
        .map(AmbiguityId::new)
        .collect::<Vec<_>>();
    let binding = AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &float_state.ambiguities_m,
    };
    let rows = build_rows(ctx, std::slice::from_ref(epoch), &binding, &float_state)
        .map_err(kinematic_error_from_row)?;
    let update = build_measurement_update(&rows, covariance_m2, config)?;
    apply_state_delta(state, &update.dx)?;
    *covariance_m2 = update.covariance_m2;
    symmetrize(covariance_m2);
    validate_state(state)?;
    validate_covariance_shape_and_values(covariance_m2, state.dimension())?;
    let innovation_rms_m = innovation_rms(&rows);
    validate_finite(innovation_rms_m, "kinematic PPP update innovation_rms_m")?;

    Ok(KinematicUpdateSummary {
        innovation_rms_m,
        used_sats: epoch
            .observations
            .iter()
            .map(|obs| obs.satellite_id.clone())
            .collect(),
    })
}

fn diagonal_covariance(dimension: usize, variance_m2: f64) -> Vec<Vec<f64>> {
    let mut covariance_m2 = vec![vec![0.0; dimension]; dimension];
    for (idx, row) in covariance_m2.iter_mut().enumerate() {
        row[idx] = variance_m2;
    }
    covariance_m2
}

struct MeasurementUpdate {
    dx: Vec<f64>,
    covariance_m2: Vec<Vec<f64>>,
}

fn validate_kinematic_solve_inputs(
    epochs: &[FloatEpoch],
    config: &KinematicConfig,
) -> Result<(), KinematicSolveError> {
    if epochs.is_empty() {
        return Err(KinematicSolveError::InvalidInput {
            field: "kinematic PPP epochs",
            reason: "must not be empty",
        });
    }
    validate_state(&config.initial_state)?;
    validate_covariance_shape_and_values(
        &config.initial_covariance_m2,
        config.initial_state.dimension(),
    )?;
    validate_config_for_predict(config)?;
    validate_measurement_config(config)?;
    validate_ordered_epochs(epochs)
}

fn validate_ordered_epochs(epochs: &[FloatEpoch]) -> Result<(), KinematicSolveError> {
    let mut previous_t_rx_j2000_s = None;
    for epoch in epochs {
        super::validate_epoch(epoch).map_err(kinematic_error_from_float)?;
        if epoch.observations.is_empty() {
            return Err(KinematicSolveError::InvalidInput {
                field: "kinematic PPP epoch observations",
                reason: "must not be empty",
            });
        }
        if let Some(previous) = previous_t_rx_j2000_s {
            if epoch.t_rx_j2000_s < previous {
                return Err(KinematicSolveError::InvalidInput {
                    field: "kinematic PPP epochs",
                    reason: "must be ordered by non-decreasing receiver time",
                });
            }
        }
        previous_t_rx_j2000_s = Some(epoch.t_rx_j2000_s);
    }
    Ok(())
}

fn validate_measurement_config(config: &KinematicConfig) -> Result<(), KinematicSolveError> {
    super::validate_measurement_weights(config.weights).map_err(kinematic_error_from_float)?;
    super::validate_troposphere_options(config.tropo).map_err(kinematic_error_from_float)?;
    super::validate_range_corrections(&config.corrections).map_err(kinematic_error_from_float)
}

fn validate_predict_inputs(
    state: &KinematicState,
    covariance_m2: &[Vec<f64>],
    dt_s: f64,
    config: &KinematicConfig,
) -> Result<(), KinematicSolveError> {
    validate_finite_nonnegative(dt_s, "kinematic PPP predict dt_s")?;
    validate_state(state)?;
    validate_covariance_shape_and_values(covariance_m2, state.dimension())?;
    validate_config_for_predict(config)?;
    Ok(())
}

fn validate_state(state: &KinematicState) -> Result<(), KinematicSolveError> {
    validate_vec3(state.position_m, "kinematic PPP state position_m")?;
    validate_finite(state.clock_m, "kinematic PPP state clock_m")?;
    validate_finite(state.ztd_residual_m, "kinematic PPP state ztd_residual_m")?;
    for value in state.ambiguities_m.values() {
        validate_finite(*value, "kinematic PPP state ambiguities_m")?;
    }
    Ok(())
}

fn validate_covariance_shape_and_values(
    covariance_m2: &[Vec<f64>],
    dimension: usize,
) -> Result<(), KinematicSolveError> {
    if covariance_m2.len() != dimension {
        return Err(KinematicSolveError::InvalidInput {
            field: "kinematic PPP covariance row count",
            reason: "must match state dimension",
        });
    }
    for (row_idx, row) in covariance_m2.iter().enumerate() {
        if row.len() != dimension {
            return Err(KinematicSolveError::InvalidInput {
                field: "kinematic PPP covariance column count",
                reason: "must match state dimension",
            });
        }
        for entry in row {
            validate_finite(*entry, "kinematic PPP covariance_m2")?;
        }
        if row[row_idx] < 0.0 {
            return Err(KinematicSolveError::InvalidInput {
                field: "kinematic PPP covariance variance",
                reason: "must be non-negative",
            });
        }
    }
    for (row_idx, row) in covariance_m2.iter().enumerate() {
        for (col_idx, value) in row.iter().enumerate().skip(row_idx + 1) {
            if !covariance_entries_symmetric(*value, covariance_m2[col_idx][row_idx]) {
                return Err(KinematicSolveError::InvalidInput {
                    field: "kinematic PPP covariance symmetry",
                    reason: "must be symmetric within tolerance",
                });
            }
        }
    }
    validate_covariance_positive_semidefinite(covariance_m2)?;
    Ok(())
}

fn covariance_entries_symmetric(a: f64, b: f64) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs()
        <= COVARIANCE_SYMMETRY_ABS_TOLERANCE_M2.max(COVARIANCE_SYMMETRY_REL_TOLERANCE * scale)
}

fn validate_covariance_positive_semidefinite(
    covariance_m2: &[Vec<f64>],
) -> Result<(), KinematicSolveError> {
    if covariance_is_positive_semidefinite(covariance_m2) {
        Ok(())
    } else {
        Err(KinematicSolveError::InvalidInput {
            field: "kinematic PPP covariance positive semidefinite",
            reason: "must be positive semidefinite within tolerance",
        })
    }
}

#[allow(clippy::needless_range_loop)]
fn covariance_is_positive_semidefinite(covariance_m2: &[Vec<f64>]) -> bool {
    let dimension = covariance_m2.len();
    let tolerance = covariance_psd_tolerance(covariance_m2);
    let mut lower = vec![vec![0.0; dimension]; dimension];

    for row_idx in 0..dimension {
        for col_idx in 0..=row_idx {
            let mut residual = covariance_m2[row_idx][col_idx];
            for prev_idx in 0..col_idx {
                residual -= lower[row_idx][prev_idx] * lower[col_idx][prev_idx];
            }

            if row_idx == col_idx {
                if !residual.is_finite() || residual < -tolerance {
                    return false;
                }
                if residual > 0.0 {
                    lower[row_idx][col_idx] = residual.sqrt();
                }
            } else if lower[col_idx][col_idx] > 0.0 {
                lower[row_idx][col_idx] = residual / lower[col_idx][col_idx];
            } else if residual.abs() > tolerance {
                return false;
            }
        }
    }

    true
}

fn covariance_psd_tolerance(covariance_m2: &[Vec<f64>]) -> f64 {
    let max_entry = covariance_m2
        .iter()
        .flat_map(|row| row.iter())
        .fold(1.0_f64, |max_entry, value| max_entry.max(value.abs()));
    COVARIANCE_PSD_ABS_TOLERANCE_M2.max(COVARIANCE_PSD_REL_TOLERANCE * max_entry)
}

fn validate_config_for_predict(config: &KinematicConfig) -> Result<(), KinematicSolveError> {
    validate_motion(config.motion)?;
    validate_process_noise(config.process_noise)?;
    validate_finite_nonnegative(
        config.new_ambiguity_variance_m2,
        "kinematic PPP new_ambiguity_variance_m2",
    )
}

fn validate_motion(motion: KinematicMotionModel) -> Result<(), KinematicSolveError> {
    match motion {
        KinematicMotionModel::Hold => Ok(()),
        KinematicMotionModel::ConstantVelocity { velocity_m_s } => {
            validate_vec3(velocity_m_s, "kinematic PPP motion velocity_m_s")
        }
    }
}

fn validate_process_noise(noise: KinematicProcessNoise) -> Result<(), KinematicSolveError> {
    match noise.position {
        KinematicPositionProcessNoise::RandomWalk {
            spectral_density_m2_s,
        } => validate_finite_nonnegative(
            spectral_density_m2_s,
            "kinematic PPP position random-walk spectral_density_m2_s",
        )?,
        KinematicPositionProcessNoise::WhiteNoiseAcceleration {
            spectral_density_m2_s3,
        } => validate_finite_nonnegative(
            spectral_density_m2_s3,
            "kinematic PPP position acceleration spectral_density_m2_s3",
        )?,
    }
    validate_finite_nonnegative(noise.clock_white_m2_s, "kinematic PPP clock_white_m2_s")?;
    validate_finite_nonnegative(
        noise.ztd_random_walk_m2_s,
        "kinematic PPP ztd_random_walk_m2_s",
    )?;
    validate_finite_nonnegative(
        noise.ambiguity_hold_m2_s,
        "kinematic PPP ambiguity_hold_m2_s",
    )
}

fn validate_vec3(value: [f64; 3], field: &'static str) -> Result<(), KinematicSolveError> {
    for entry in value {
        validate_finite(entry, field)?;
    }
    Ok(())
}

fn validate_finite_nonnegative(value: f64, field: &'static str) -> Result<(), KinematicSolveError> {
    validate_finite(value, field)?;
    if value < 0.0 {
        return Err(KinematicSolveError::InvalidInput {
            field,
            reason: "must be non-negative",
        });
    }
    Ok(())
}

fn validate_finite(value: f64, field: &'static str) -> Result<(), KinematicSolveError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(KinematicSolveError::InvalidInput {
            field,
            reason: "must be finite",
        })
    }
}

fn validate_finite_slice(values: &[f64], field: &'static str) -> Result<(), KinematicSolveError> {
    for value in values {
        validate_finite(*value, field)?;
    }
    Ok(())
}

fn validate_finite_matrix(
    matrix: &[Vec<f64>],
    field: &'static str,
) -> Result<(), KinematicSolveError> {
    for row in matrix {
        validate_finite_slice(row, field)?;
    }
    Ok(())
}

fn kinematic_error_from_row(error: PppRowError) -> KinematicSolveError {
    kinematic_error_from_float(error.into_float())
}

fn kinematic_error_from_float(error: FloatSolveError) -> KinematicSolveError {
    match error {
        FloatSolveError::NoEphemeris {
            satellite_id,
            reason,
        } => KinematicSolveError::NoEphemeris {
            satellite_id,
            reason,
        },
        FloatSolveError::SingularGeometry => KinematicSolveError::SingularGeometry,
        FloatSolveError::InvalidSolveOption { field, reason } => {
            KinematicSolveError::InvalidSolveOption { field, reason }
        }
        FloatSolveError::InvalidInput { field, reason } => {
            KinematicSolveError::InvalidInput { field, reason }
        }
        FloatSolveError::MissingCorrection {
            satellite_id,
            correction,
        } => KinematicSolveError::MissingCorrection {
            satellite_id,
            correction,
        },
        FloatSolveError::InvalidClockCount { .. } => KinematicSolveError::InvalidInput {
            field: "kinematic PPP clock state",
            reason: "must contain exactly one receiver clock",
        },
        FloatSolveError::MissingAmbiguity(_) => KinematicSolveError::InvalidInput {
            field: "kinematic PPP ambiguity state",
            reason: "must include every active ambiguity",
        },
    }
}

fn align_ambiguities(
    state: &mut KinematicState,
    covariance_m2: &mut Vec<Vec<f64>>,
    active_ambiguity_ids: &[String],
    new_ambiguity_variance_m2: f64,
) {
    let old_keys = ambiguity_keys(state);
    let new_keys = active_ambiguity_ids
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    if old_keys == new_keys {
        return;
    }

    let old_index_by_key = old_keys
        .iter()
        .enumerate()
        .map(|(idx, key)| (key.clone(), BASE_STATE_DIMENSION + idx))
        .collect::<BTreeMap<_, _>>();
    let new_dimension = BASE_STATE_DIMENSION + new_keys.len();
    let mut next_covariance_m2 = vec![vec![0.0; new_dimension]; new_dimension];

    for row in 0..BASE_STATE_DIMENSION {
        for col in 0..BASE_STATE_DIMENSION {
            next_covariance_m2[row][col] = covariance_m2[row][col];
        }
    }

    for (new_ambiguity_idx, new_key) in new_keys.iter().enumerate() {
        let new_idx = BASE_STATE_DIMENSION + new_ambiguity_idx;
        if let Some(&old_idx) = old_index_by_key.get(new_key) {
            copy_retained_ambiguity_covariance(
                covariance_m2,
                &mut next_covariance_m2,
                old_idx,
                new_idx,
                &new_keys,
                &old_index_by_key,
            );
        } else {
            next_covariance_m2[new_idx][new_idx] = new_ambiguity_variance_m2;
        }
    }

    state.ambiguities_m = new_keys
        .into_iter()
        .map(|key| {
            let value = state.ambiguities_m.get(&key).copied().unwrap_or(0.0);
            (key, value)
        })
        .collect();
    *covariance_m2 = next_covariance_m2;
}

fn copy_retained_ambiguity_covariance(
    old_covariance_m2: &[Vec<f64>],
    next_covariance_m2: &mut [Vec<f64>],
    old_idx: usize,
    new_idx: usize,
    new_keys: &[String],
    old_index_by_key: &BTreeMap<String, usize>,
) {
    for base_idx in 0..BASE_STATE_DIMENSION {
        next_covariance_m2[new_idx][base_idx] = old_covariance_m2[old_idx][base_idx];
        next_covariance_m2[base_idx][new_idx] = old_covariance_m2[base_idx][old_idx];
    }
    for (other_new_ambiguity_idx, other_key) in new_keys.iter().enumerate() {
        if let Some(&other_old_idx) = old_index_by_key.get(other_key) {
            let other_new_idx = BASE_STATE_DIMENSION + other_new_ambiguity_idx;
            next_covariance_m2[new_idx][other_new_idx] = old_covariance_m2[old_idx][other_old_idx];
        }
    }
}

fn ambiguity_keys(state: &KinematicState) -> Vec<String> {
    state.ambiguities_m.keys().cloned().collect()
}

fn propagate_mean(state: &mut KinematicState, dt_s: f64, motion: KinematicMotionModel) {
    match motion {
        KinematicMotionModel::Hold => {}
        KinematicMotionModel::ConstantVelocity { velocity_m_s } => {
            for (position, velocity) in state.position_m.iter_mut().zip(velocity_m_s) {
                *position += velocity * dt_s;
            }
        }
    }
}

fn build_measurement_update(
    rows: &[super::normal::Row],
    covariance_m2: &[Vec<f64>],
    config: &KinematicConfig,
) -> Result<MeasurementUpdate, KinematicSolveError> {
    if rows.is_empty() {
        return Err(KinematicSolveError::InvalidInput {
            field: "kinematic PPP epoch observations",
            reason: "must not be empty",
        });
    }
    let dimension = covariance_m2.len();
    let h = kinematic_design_matrix(rows, dimension, config)?;
    let innovation = rows
        .iter()
        .map(|row| {
            validate_finite(row.y, "kinematic PPP innovation_m")?;
            Ok(row.y)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let measurement_variance = rows
        .iter()
        .map(|row| {
            validate_finite_nonnegative(row.weight, "kinematic PPP measurement weight")?;
            if row.weight <= 0.0 {
                return Err(KinematicSolveError::InvalidInput {
                    field: "kinematic PPP measurement weight",
                    reason: "must be positive",
                });
            }
            let variance = 1.0 / (row.weight * row.weight);
            validate_finite_nonnegative(variance, "kinematic PPP measurement variance")?;
            Ok(variance)
        })
        .collect::<Result<Vec<_>, _>>()?;

    let h_t = transpose(&h).ok_or(KinematicSolveError::SingularGeometry)?;
    let hp = matmul(&h, covariance_m2).ok_or(KinematicSolveError::SingularGeometry)?;
    let mut innovation_covariance =
        matmul(&hp, &h_t).ok_or(KinematicSolveError::SingularGeometry)?;
    for (idx, variance) in measurement_variance.iter().enumerate() {
        innovation_covariance[idx][idx] += variance;
    }
    let innovation_covariance_inverse = invert_matrix_last_tie(&innovation_covariance)
        .ok_or(KinematicSolveError::SingularGeometry)?;
    let p_h_t = matmul(covariance_m2, &h_t).ok_or(KinematicSolveError::SingularGeometry)?;
    let mut kalman_gain = matmul(&p_h_t, &innovation_covariance_inverse)
        .ok_or(KinematicSolveError::SingularGeometry)?;
    let ztd_estimated = estimates_ztd(config.tropo);
    if !ztd_estimated {
        kalman_gain[ZTD_INDEX].fill(0.0);
    }
    let dx = matvec(&kalman_gain, &innovation)?;
    validate_finite_slice(&dx, "kinematic PPP state update")?;
    let mut covariance_update = joseph_covariance(
        covariance_m2,
        &h,
        &kalman_gain,
        &measurement_variance,
        dimension,
    )?;
    if !ztd_estimated {
        restore_frozen_ztd_covariance(&mut covariance_update, covariance_m2);
    }
    validate_finite_matrix(&covariance_update, "kinematic PPP covariance update")?;
    Ok(MeasurementUpdate {
        dx,
        covariance_m2: covariance_update,
    })
}

fn restore_frozen_ztd_covariance(covariance_m2: &mut [Vec<f64>], prior_covariance_m2: &[Vec<f64>]) {
    covariance_m2[ZTD_INDEX][..prior_covariance_m2.len()]
        .copy_from_slice(&prior_covariance_m2[ZTD_INDEX]);
    for (row_idx, row) in covariance_m2.iter_mut().enumerate() {
        row[ZTD_INDEX] = prior_covariance_m2[row_idx][ZTD_INDEX];
    }
}

fn kinematic_design_matrix(
    rows: &[super::normal::Row],
    dimension: usize,
    config: &KinematicConfig,
) -> Result<Vec<Vec<f64>>, KinematicSolveError> {
    rows.iter()
        .map(|row| kinematic_design_row(row, dimension, config))
        .collect()
}

fn kinematic_design_row(
    row: &super::normal::Row,
    dimension: usize,
    config: &KinematicConfig,
) -> Result<Vec<f64>, KinematicSolveError> {
    let ztd_estimated = estimates_ztd(config.tropo);
    let ztd_cols = usize::from(ztd_estimated);
    let static_ambiguity_start = 4 + ztd_cols;
    let expected_static_dim = static_ambiguity_start + dimension - BASE_STATE_DIMENSION;
    if row.h.len() != expected_static_dim {
        return Err(KinematicSolveError::InvalidInput {
            field: "kinematic PPP design row",
            reason: "static PPP row dimension does not match kinematic state",
        });
    }
    let mut h = vec![0.0; dimension];
    h[..3].copy_from_slice(&row.h[..3]);
    h[CLOCK_INDEX] = row.h[3];
    if ztd_estimated {
        h[ZTD_INDEX] = row.h[4];
    }
    let ambiguity_count = dimension - BASE_STATE_DIMENSION;
    h[BASE_STATE_DIMENSION..BASE_STATE_DIMENSION + ambiguity_count]
        .copy_from_slice(&row.h[static_ambiguity_start..static_ambiguity_start + ambiguity_count]);
    validate_finite_slice(&h, "kinematic PPP design row")?;
    Ok(h)
}

fn matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>, KinematicSolveError> {
    matrix
        .iter()
        .map(|row| {
            if row.len() != vector.len() {
                return Err(KinematicSolveError::SingularGeometry);
            }
            Ok(row.iter().zip(vector).map(|(a, b)| a * b).sum())
        })
        .collect()
}

fn joseph_covariance(
    covariance_m2: &[Vec<f64>],
    h: &[Vec<f64>],
    kalman_gain: &[Vec<f64>],
    measurement_variance: &[f64],
    dimension: usize,
) -> Result<Vec<Vec<f64>>, KinematicSolveError> {
    let kh = matmul(kalman_gain, h).ok_or(KinematicSolveError::SingularGeometry)?;
    let identity_minus_kh =
        matrix_sub(&identity(dimension), &kh).ok_or(KinematicSolveError::SingularGeometry)?;
    let left =
        matmul(&identity_minus_kh, covariance_m2).ok_or(KinematicSolveError::SingularGeometry)?;
    let right = transpose(&identity_minus_kh).ok_or(KinematicSolveError::SingularGeometry)?;
    let stabilized = matmul(&left, &right).ok_or(KinematicSolveError::SingularGeometry)?;
    let kr = scale_columns(kalman_gain, measurement_variance)?;
    let k_t = transpose(kalman_gain).ok_or(KinematicSolveError::SingularGeometry)?;
    let noise = matmul(&kr, &k_t).ok_or(KinematicSolveError::SingularGeometry)?;
    matrix_add(&stabilized, &noise).ok_or(KinematicSolveError::SingularGeometry)
}

fn identity(dimension: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; dimension]; dimension];
    for (idx, row) in matrix.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    matrix
}

fn scale_columns(
    matrix: &[Vec<f64>],
    scales: &[f64],
) -> Result<Vec<Vec<f64>>, KinematicSolveError> {
    matrix
        .iter()
        .map(|row| {
            if row.len() != scales.len() {
                return Err(KinematicSolveError::SingularGeometry);
            }
            Ok(row
                .iter()
                .zip(scales)
                .map(|(value, scale)| value * scale)
                .collect())
        })
        .collect()
}

fn matrix_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    if a.len() != b.len() {
        return None;
    }
    let mut out = Vec::with_capacity(a.len());
    for (row_a, row_b) in a.iter().zip(b) {
        if row_a.len() != row_b.len() {
            return None;
        }
        out.push(row_a.iter().zip(row_b).map(|(x, y)| x + y).collect());
    }
    Some(out)
}

fn apply_state_delta(state: &mut KinematicState, dx: &[f64]) -> Result<(), KinematicSolveError> {
    if dx.len() != state.dimension() {
        return Err(KinematicSolveError::SingularGeometry);
    }
    for (position, delta) in state.position_m.iter_mut().zip(&dx[..3]) {
        *position += delta;
    }
    state.clock_m += dx[CLOCK_INDEX];
    state.ztd_residual_m += dx[ZTD_INDEX];
    for (idx, value) in state.ambiguities_m.values_mut().enumerate() {
        *value += dx[BASE_STATE_DIMENSION + idx];
    }
    Ok(())
}

fn float_state_from_kinematic(state: &KinematicState) -> FloatState {
    FloatState {
        position_m: state.position_m,
        clocks_m: vec![state.clock_m],
        ambiguities_m: state.ambiguities_m.clone(),
        ztd_m: state.ztd_residual_m,
    }
}

fn innovation_rms(rows: &[super::normal::Row]) -> f64 {
    if rows.is_empty() {
        return 0.0;
    }
    let mean_square = rows.iter().map(|row| row.y * row.y).sum::<f64>() / rows.len() as f64;
    mean_square.sqrt()
}

fn position_covariance_block(covariance_m2: &[Vec<f64>]) -> [[f64; 3]; 3] {
    [
        [
            covariance_m2[0][0],
            covariance_m2[0][1],
            covariance_m2[0][2],
        ],
        [
            covariance_m2[1][0],
            covariance_m2[1][1],
            covariance_m2[1][2],
        ],
        [
            covariance_m2[2][0],
            covariance_m2[2][1],
            covariance_m2[2][2],
        ],
    ]
}

fn inflate_covariance(
    covariance_m2: &mut [Vec<f64>],
    dt_s: f64,
    process_noise: KinematicProcessNoise,
) {
    let position_variance_m2 = match process_noise.position {
        KinematicPositionProcessNoise::RandomWalk {
            spectral_density_m2_s,
        } => spectral_density_m2_s * dt_s,
        KinematicPositionProcessNoise::WhiteNoiseAcceleration {
            spectral_density_m2_s3,
        } => spectral_density_m2_s3 * dt_s.powi(3) / 3.0,
    };

    for (idx, row) in covariance_m2.iter_mut().enumerate().take(3) {
        row[idx] += position_variance_m2;
    }
    covariance_m2[CLOCK_INDEX][CLOCK_INDEX] += process_noise.clock_white_m2_s * dt_s;
    covariance_m2[ZTD_INDEX][ZTD_INDEX] += process_noise.ztd_random_walk_m2_s * dt_s;
    for (idx, row) in covariance_m2
        .iter_mut()
        .enumerate()
        .skip(BASE_STATE_DIMENSION)
    {
        row[idx] += process_noise.ambiguity_hold_m2_s * dt_s;
    }
}

#[allow(clippy::needless_range_loop)]
fn symmetrize(covariance_m2: &mut [Vec<f64>]) {
    for row in 0..covariance_m2.len() {
        for col in 0..row {
            let average = 0.5 * (covariance_m2[row][col] + covariance_m2[col][row]);
            covariance_m2[row][col] = average;
            covariance_m2[col][row] = average;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::FloatObservation;
    use super::*;
    use crate::constants::F_L1_HZ;
    use crate::estimation::substrate::rows::ResidualRow;
    use crate::observables::{predict, ObservableState, ObservablesError, PredictOptions};
    use crate::ppp_corrections::CivilDateTime;
    use crate::{GnssSatelliteId, GnssSystem};

    struct KinematicFakeSource {
        states: BTreeMap<GnssSatelliteId, [f64; 3]>,
    }

    impl ObservableEphemerisSource for KinematicFakeSource {
        fn observable_state_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            let position_ecef_m = self
                .states
                .get(&sat)
                .copied()
                .ok_or(ObservablesError::NoEphemeris)?;
            Ok(ObservableState {
                position_ecef_m,
                clock_s: Some(0.0),
            })
        }
    }

    #[test]
    fn kinematic_types_construct_and_default_config_is_well_formed() {
        let mut ambiguities_m = BTreeMap::new();
        ambiguities_m.insert("G07#1".to_string(), 12.5);
        let state = KinematicState {
            position_m: [1.0, 2.0, 3.0],
            clock_m: 4.0,
            ztd_residual_m: 0.12,
            ambiguities_m,
        };
        assert_eq!(state.dimension(), 6);

        let config = KinematicConfig {
            initial_covariance_m2: diagonal_covariance(state.dimension(), 25.0),
            initial_state: state,
            motion: KinematicMotionModel::ConstantVelocity {
                velocity_m_s: [1.0, 0.0, 0.0],
            },
            process_noise: KinematicProcessNoise {
                position: KinematicPositionProcessNoise::WhiteNoiseAcceleration {
                    spectral_density_m2_s3: 0.25,
                },
                clock_white_m2_s: 2.0,
                ztd_random_walk_m2_s: 1.0e-5,
                ambiguity_hold_m2_s: 1.0e-8,
            },
            new_ambiguity_variance_m2: 1.0e6,
            weights: MeasurementWeights {
                code: 0.5,
                phase: 50.0,
                elevation_weighting: true,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
        };
        assert!(config_is_well_formed(&config));

        let default = KinematicConfig::default();
        assert!(config_is_well_formed(&default));
        assert_eq!(default.initial_state.dimension(), BASE_STATE_DIMENSION);
    }

    fn config_is_well_formed(config: &KinematicConfig) -> bool {
        let dimension = config.initial_state.dimension();
        config.initial_covariance_m2.len() == dimension
            && config
                .initial_covariance_m2
                .iter()
                .all(|row| row.len() == dimension && row.iter().all(|entry| entry.is_finite()))
            && motion_is_well_formed(config.motion)
            && process_noise_is_well_formed(config.process_noise)
            && config.new_ambiguity_variance_m2.is_finite()
            && config.new_ambiguity_variance_m2 >= 0.0
            && config.weights.code.is_finite()
            && config.weights.code > 0.0
            && config.weights.phase.is_finite()
            && config.weights.phase > 0.0
    }

    fn process_noise_is_well_formed(process_noise: KinematicProcessNoise) -> bool {
        position_noise_is_well_formed(process_noise.position)
            && process_noise.clock_white_m2_s.is_finite()
            && process_noise.clock_white_m2_s >= 0.0
            && process_noise.ztd_random_walk_m2_s.is_finite()
            && process_noise.ztd_random_walk_m2_s >= 0.0
            && process_noise.ambiguity_hold_m2_s.is_finite()
            && process_noise.ambiguity_hold_m2_s >= 0.0
    }

    fn motion_is_well_formed(motion: KinematicMotionModel) -> bool {
        match motion {
            KinematicMotionModel::Hold => true,
            KinematicMotionModel::ConstantVelocity { velocity_m_s } => {
                velocity_m_s.iter().all(|entry| entry.is_finite())
            }
        }
    }

    fn position_noise_is_well_formed(position: KinematicPositionProcessNoise) -> bool {
        match position {
            KinematicPositionProcessNoise::RandomWalk {
                spectral_density_m2_s,
            } => spectral_density_m2_s.is_finite() && spectral_density_m2_s >= 0.0,
            KinematicPositionProcessNoise::WhiteNoiseAcceleration {
                spectral_density_m2_s3,
            } => spectral_density_m2_s3.is_finite() && spectral_density_m2_s3 >= 0.0,
        }
    }

    #[test]
    fn zero_dt_predict_keeps_mean_and_covariance_when_ambiguities_are_unchanged() {
        let mut state = state_with_ambiguities(["G07#1"]);
        let mut covariance_m2 = diagonal_covariance(state.dimension(), 4.0);
        let before_state = state.clone();
        let before_covariance_m2 = covariance_m2.clone();
        let config = KinematicConfig {
            motion: KinematicMotionModel::ConstantVelocity {
                velocity_m_s: [3.0, -2.0, 1.0],
            },
            process_noise: KinematicProcessNoise {
                position: KinematicPositionProcessNoise::RandomWalk {
                    spectral_density_m2_s: 20.0,
                },
                clock_white_m2_s: 30.0,
                ztd_random_walk_m2_s: 40.0,
                ambiguity_hold_m2_s: 50.0,
            },
            initial_state: state.clone(),
            initial_covariance_m2: covariance_m2.clone(),
            ..KinematicConfig::default()
        };

        predict_kinematic_state(
            &mut state,
            &mut covariance_m2,
            0.0,
            &["G07#1".to_string()],
            &config,
        )
        .expect("zero-dt predict should succeed");

        assert_eq!(state, before_state);
        assert_eq!(covariance_m2, before_covariance_m2);
    }

    #[test]
    fn predict_covariance_stays_symmetric_psd() {
        let mut state = state_with_ambiguities(["G07#1", "G08#1"]);
        let mut covariance_m2 = diagonal_covariance(state.dimension(), 10.0);
        covariance_m2[0][BASE_STATE_DIMENSION] = 0.5;
        covariance_m2[BASE_STATE_DIMENSION][0] = 0.5;
        let config = KinematicConfig {
            process_noise: KinematicProcessNoise {
                position: KinematicPositionProcessNoise::WhiteNoiseAcceleration {
                    spectral_density_m2_s3: 0.3,
                },
                clock_white_m2_s: 0.2,
                ztd_random_walk_m2_s: 0.1,
                ambiguity_hold_m2_s: 0.05,
            },
            initial_state: state.clone(),
            initial_covariance_m2: covariance_m2.clone(),
            ..KinematicConfig::default()
        };

        predict_kinematic_state(
            &mut state,
            &mut covariance_m2,
            5.0,
            &["G07#1".to_string(), "G08#1".to_string()],
            &config,
        )
        .expect("predict should succeed");

        assert!(is_symmetric(&covariance_m2));
        assert!(is_psd(&covariance_m2));
    }

    #[test]
    fn initial_covariance_rejects_asymmetry_and_negative_variance() {
        let (source, epoch, initial_state, mut config) = single_epoch_update_fixture();
        let epochs = vec![epoch];
        let dimension = initial_state.dimension();

        let mut asymmetric = diagonal_covariance(dimension, 25.0);
        asymmetric[0][1] = 0.25;
        config.initial_covariance_m2 = asymmetric;
        let err = solve_kinematic_ppp(&source, &epochs, config.clone())
            .expect_err("asymmetric covariance should be rejected");
        assert_invalid_kinematic_input(
            err,
            "kinematic PPP covariance symmetry",
            "must be symmetric within tolerance",
        );

        let mut negative_variance = diagonal_covariance(dimension, 25.0);
        negative_variance[2][2] = -1.0;
        config.initial_covariance_m2 = negative_variance;
        let err = solve_kinematic_ppp(&source, &epochs, config)
            .expect_err("negative covariance variance should be rejected");
        assert_invalid_kinematic_input(
            err,
            "kinematic PPP covariance variance",
            "must be non-negative",
        );
    }

    #[test]
    fn initial_covariance_rejects_symmetric_indefinite_matrix() {
        let (source, epoch, initial_state, mut config) = single_epoch_update_fixture();
        let epochs = vec![epoch];
        config.initial_covariance_m2 = indefinite_covariance(initial_state.dimension());

        let err = solve_kinematic_ppp(&source, &epochs, config)
            .expect_err("indefinite initial covariance should be rejected");

        assert_invalid_kinematic_input(
            err,
            "kinematic PPP covariance positive semidefinite",
            "must be positive semidefinite within tolerance",
        );
    }

    #[test]
    fn covariance_validation_accepts_symmetric_psd_unchanged() {
        let dimension = state_with_ambiguities(["G07#1"]).dimension();
        let mut covariance_m2 = diagonal_covariance(dimension, 4.0);
        covariance_m2[0][1] = 0.25;
        covariance_m2[1][0] = 0.25;
        let original = covariance_m2.clone();

        validate_covariance_shape_and_values(&covariance_m2, dimension)
            .expect("symmetric PSD covariance should be accepted");

        assert_eq!(covariance_m2, original);
    }

    #[test]
    fn predict_rejects_symmetric_indefinite_covariance() {
        let (_, epoch, mut state, config) = single_epoch_update_fixture();
        let mut covariance_m2 = indefinite_covariance(state.dimension());
        let active_ambiguity_ids = epoch
            .observations
            .iter()
            .map(|obs| obs.ambiguity_id.clone())
            .collect::<Vec<_>>();

        let err = predict_kinematic_state(
            &mut state,
            &mut covariance_m2,
            0.0,
            &active_ambiguity_ids,
            &config,
        )
        .expect_err("indefinite mutable covariance should be rejected");

        assert_invalid_kinematic_input(
            err,
            "kinematic PPP covariance positive semidefinite",
            "must be positive semidefinite within tolerance",
        );
    }

    #[test]
    fn predict_adds_and_removes_ambiguities_without_orphaned_covariance_entries() {
        let mut state = state_with_ambiguities(["G07#1"]);
        let mut covariance_m2 = diagonal_covariance(state.dimension(), 3.0);
        let config = KinematicConfig {
            new_ambiguity_variance_m2: 99.0,
            initial_state: state.clone(),
            initial_covariance_m2: covariance_m2.clone(),
            ..KinematicConfig::default()
        };

        predict_kinematic_state(
            &mut state,
            &mut covariance_m2,
            1.0,
            &["G07#1".to_string(), "G08#1".to_string()],
            &config,
        )
        .expect("adding ambiguity should succeed");

        assert_eq!(state.dimension(), BASE_STATE_DIMENSION + 2);
        assert_eq!(covariance_m2.len(), state.dimension());
        assert!(covariance_m2
            .iter()
            .all(|row| row.len() == state.dimension()));
        assert!(state.ambiguities_m.contains_key("G08#1"));
        assert_eq!(
            covariance_m2[BASE_STATE_DIMENSION + 1][BASE_STATE_DIMENSION + 1],
            99.0
        );
        assert!(is_symmetric(&covariance_m2));

        predict_kinematic_state(
            &mut state,
            &mut covariance_m2,
            1.0,
            &["G08#1".to_string()],
            &config,
        )
        .expect("removing ambiguity should succeed");

        assert_eq!(state.dimension(), BASE_STATE_DIMENSION + 1);
        assert_eq!(covariance_m2.len(), state.dimension());
        assert!(covariance_m2
            .iter()
            .all(|row| row.len() == state.dimension()));
        assert!(!state.ambiguities_m.contains_key("G07#1"));
        assert!(state.ambiguities_m.contains_key("G08#1"));
        assert!(is_symmetric(&covariance_m2));
    }

    fn state_with_ambiguities<const N: usize>(ids: [&str; N]) -> KinematicState {
        KinematicState {
            position_m: [1.0, 2.0, 3.0],
            clock_m: 4.0,
            ztd_residual_m: 0.5,
            ambiguities_m: ids
                .into_iter()
                .enumerate()
                .map(|(idx, id)| (id.to_string(), idx as f64 + 10.0))
                .collect(),
        }
    }

    fn is_symmetric(covariance_m2: &[Vec<f64>]) -> bool {
        covariance_m2.iter().enumerate().all(|(row_idx, row)| {
            row.iter()
                .enumerate()
                .all(|(col_idx, value)| (*value - covariance_m2[col_idx][row_idx]).abs() <= 1.0e-12)
        })
    }

    fn assert_invalid_kinematic_input(
        error: KinematicSolveError,
        field: &'static str,
        reason: &'static str,
    ) {
        assert_eq!(error, KinematicSolveError::InvalidInput { field, reason });
    }

    fn indefinite_covariance(dimension: usize) -> Vec<Vec<f64>> {
        let mut covariance_m2 = diagonal_covariance(dimension, 25.0);
        covariance_m2[0][0] = 1.0;
        covariance_m2[1][1] = 1.0;
        covariance_m2[0][1] = 2.0;
        covariance_m2[1][0] = 2.0;
        covariance_m2
    }

    #[allow(clippy::needless_range_loop)]
    fn is_psd(covariance_m2: &[Vec<f64>]) -> bool {
        let n = covariance_m2.len();
        let mut lower = vec![vec![0.0; n]; n];
        for row in 0..n {
            for col in 0..=row {
                let mut sum = covariance_m2[row][col];
                for k in 0..col {
                    sum -= lower[row][k] * lower[col][k];
                }
                if row == col {
                    if sum < -1.0e-10 {
                        return false;
                    }
                    lower[row][col] = sum.max(0.0).sqrt();
                } else if lower[col][col] > 1.0e-12 {
                    lower[row][col] = sum / lower[col][col];
                }
            }
        }
        true
    }

    #[test]
    fn update_pulls_position_toward_static_float_solution() {
        let (source, epoch, initial_state, config) = single_epoch_update_fixture();
        let static_solution = super::super::solve_float_epoch(
            &source,
            epoch.clone(),
            float_state_from_kinematic(&initial_state),
            super::super::FloatSolveConfig {
                weights: config.weights,
                tropo: config.tropo,
                corrections: config.corrections.clone(),
                opts: super::super::FloatSolveOptions {
                    max_iterations: 20,
                    position_tolerance_m: 1.0e-8,
                    clock_tolerance_m: 1.0e-8,
                    ambiguity_tolerance_m: 1.0e-8,
                    ztd_tolerance_m: 1.0e-8,
                },
                residual_screen: false,
            },
        )
        .expect("static float solve should converge");

        let mut state = initial_state.clone();
        let mut covariance_m2 = config.initial_covariance_m2.clone();
        predict_kinematic_state(
            &mut state,
            &mut covariance_m2,
            0.0,
            &epoch
                .observations
                .iter()
                .map(|obs| obs.ambiguity_id.clone())
                .collect::<Vec<_>>(),
            &config,
        )
        .expect("predict should succeed");
        let before = distance(state.position_m, static_solution.position_m);
        let update =
            correct_kinematic_state(&source, &epoch, &mut state, &mut covariance_m2, &config)
                .expect("measurement update should succeed");
        let after = distance(state.position_m, static_solution.position_m);

        assert!(after < before);
        assert!(after < 1.0);
        assert!(update.innovation_rms_m.is_finite());
        assert!(update.innovation_rms_m > 0.0);
        assert_eq!(update.used_sats.len(), epoch.observations.len());
    }

    #[test]
    fn update_covariance_remains_symmetric_psd() {
        let (source, epoch, mut state, config) = single_epoch_update_fixture();
        let mut covariance_m2 = config.initial_covariance_m2.clone();

        correct_kinematic_state(&source, &epoch, &mut state, &mut covariance_m2, &config)
            .expect("measurement update should succeed");

        assert!(is_symmetric(&covariance_m2));
        assert!(is_psd(&covariance_m2));
    }

    #[test]
    fn update_rejects_non_finite_internal_measurement_variance() {
        let (source, epoch, mut state, mut config) = single_epoch_update_fixture();
        config.weights.code = f64::MIN_POSITIVE;
        config.weights.phase = f64::MIN_POSITIVE;
        let mut covariance_m2 = config.initial_covariance_m2.clone();

        let err = correct_kinematic_state(&source, &epoch, &mut state, &mut covariance_m2, &config)
            .expect_err("overflowed measurement variance must be rejected");

        assert_eq!(
            err,
            KinematicSolveError::InvalidInput {
                field: "kinematic PPP measurement variance",
                reason: "must be finite",
            }
        );
    }

    #[test]
    fn disabled_ztd_estimation_freezes_ztd_state_and_cross_covariance() {
        let mut state = KinematicState {
            position_m: [0.0, 0.0, 0.0],
            clock_m: 0.0,
            ztd_residual_m: 0.42,
            ambiguities_m: BTreeMap::new(),
        };
        let mut covariance_m2 = diagonal_covariance(state.dimension(), 4.0);
        covariance_m2[ZTD_INDEX][ZTD_INDEX] = 9.0;
        covariance_m2[0][ZTD_INDEX] = 0.25;
        covariance_m2[ZTD_INDEX][0] = 0.25;
        covariance_m2[CLOCK_INDEX][ZTD_INDEX] = -0.125;
        covariance_m2[ZTD_INDEX][CLOCK_INDEX] = -0.125;
        let prior_state = state.clone();
        let prior_covariance_m2 = covariance_m2.clone();
        let row = ResidualRow {
            h: vec![1.0, 0.0, 0.0, 0.0],
            y: 10.0,
            weight: 1.0,
        };
        let config = KinematicConfig {
            tropo: TroposphereOptions::disabled(),
            ..KinematicConfig::default()
        };

        let update = build_measurement_update(&[row], &covariance_m2, &config)
            .expect("measurement update should be well conditioned");
        apply_state_delta(&mut state, &update.dx).expect("state delta should apply");

        assert!(state.position_m[0] > prior_state.position_m[0]);
        assert!(update.covariance_m2[0][0] < prior_covariance_m2[0][0]);
        assert_eq!(state.ztd_residual_m, prior_state.ztd_residual_m);
        assert_eq!(
            update.covariance_m2[ZTD_INDEX],
            prior_covariance_m2[ZTD_INDEX]
        );
        for (row_idx, row) in update.covariance_m2.iter().enumerate() {
            assert_eq!(row[ZTD_INDEX], prior_covariance_m2[row_idx][ZTD_INDEX]);
        }
    }

    #[test]
    fn enabled_ztd_estimation_updates_ztd_state() {
        let covariance_m2 = diagonal_covariance(BASE_STATE_DIMENSION, 4.0);
        let row = ResidualRow {
            h: vec![1.0, 0.0, 0.0, 0.0, 1.0],
            y: 10.0,
            weight: 1.0,
        };
        let mut tropo = TroposphereOptions::disabled();
        tropo.enabled = true;
        tropo.estimate_ztd = true;
        let config = KinematicConfig {
            tropo,
            ..KinematicConfig::default()
        };

        let update = build_measurement_update(&[row], &covariance_m2, &config)
            .expect("measurement update should be well conditioned");

        assert!(update.dx[ZTD_INDEX] > 0.0);
        assert_ne!(
            update.covariance_m2[ZTD_INDEX][ZTD_INDEX],
            covariance_m2[ZTD_INDEX][ZTD_INDEX]
        );
    }

    #[test]
    fn singular_innovation_covariance_is_reported() {
        let (source, epoch, mut state, mut config) = single_epoch_update_fixture();
        config.weights = MeasurementWeights {
            code: 1.0e300,
            phase: 1.0e300,
            elevation_weighting: false,
        };
        let mut covariance_m2 = vec![vec![0.0; state.dimension()]; state.dimension()];

        let err = correct_kinematic_state(&source, &epoch, &mut state, &mut covariance_m2, &config)
            .expect_err("singular innovation covariance should error");

        assert_eq!(err, KinematicSolveError::SingularGeometry);
    }

    #[test]
    fn driver_static_arc_converges_to_static_float_solution() {
        let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
        let truths = vec![truth; 6];
        let clocks = [12.5, -8.25, 4.0, 1.5, -2.0, 6.75];
        let (source, epochs, ambiguities_m) = synthetic_kinematic_arc(&truths, &clocks);
        let initial_state = KinematicState {
            position_m: [truth[0] + 5.0, truth[1] - 4.0, truth[2] + 3.0],
            clock_m: -20.0,
            ztd_residual_m: 0.0,
            ambiguities_m: ambiguities_m.clone(),
        };
        let config = driver_config(initial_state.clone());
        let static_solution = super::super::solve_float_epochs(
            &source,
            &epochs,
            FloatState {
                position_m: initial_state.position_m,
                clocks_m: vec![initial_state.clock_m; epochs.len()],
                ambiguities_m,
                ztd_m: 0.0,
            },
            float_config_from_kinematic(&config),
        )
        .expect("static float solve should converge");

        let solutions =
            solve_kinematic_ppp(&source, &epochs, config).expect("kinematic solve should succeed");
        let last = solutions.last().expect("kinematic solution");
        let penultimate = &solutions[solutions.len() - 2];

        assert_eq!(solutions.len(), epochs.len());
        assert_eq!(last.status, KinematicEpochStatus::Updated);
        assert!(distance(last.position_m, static_solution.position_m) < 0.05);
        assert!(distance(penultimate.position_m, static_solution.position_m) < 0.10);
        assert!(
            position_trace(last.position_covariance_m2)
                < position_trace(solutions[0].position_covariance_m2)
        );
    }

    #[test]
    fn driver_constant_velocity_track_is_recovered() {
        let start = [3_512_900.0, 780_500.0, 5_248_700.0];
        let velocity_m_s = [0.45, -0.20, 0.15];
        let truths = (0..6)
            .map(|idx| position_at(start, velocity_m_s, idx as f64 * 60.0))
            .collect::<Vec<_>>();
        let clocks = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5];
        let (source, epochs, ambiguities_m) = synthetic_kinematic_arc(&truths, &clocks);
        let initial_state = KinematicState {
            position_m: [start[0] + 3.0, start[1] - 2.0, start[2] + 1.0],
            clock_m: 0.0,
            ztd_residual_m: 0.0,
            ambiguities_m,
        };
        let config = KinematicConfig {
            motion: KinematicMotionModel::ConstantVelocity { velocity_m_s },
            ..driver_config(initial_state)
        };

        let solutions =
            solve_kinematic_ppp(&source, &epochs, config).expect("kinematic solve should succeed");

        for (solution, truth) in solutions.iter().zip(truths.iter()).skip(1) {
            assert!(distance(solution.position_m, *truth) < 0.25);
            assert!(solution.innovation_rms_m.is_finite());
            assert_eq!(solution.used_sats.len(), epochs[0].observations.len());
        }
    }

    #[test]
    fn driver_position_covariance_trace_decreases_over_static_arc() {
        let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
        let truths = vec![truth; 5];
        let clocks = [12.5, -8.25, 4.0, 1.5, -2.0];
        let (source, epochs, ambiguities_m) = synthetic_kinematic_arc(&truths, &clocks);
        let initial_state = KinematicState {
            position_m: [truth[0] + 5.0, truth[1] - 4.0, truth[2] + 3.0],
            clock_m: -20.0,
            ztd_residual_m: 0.0,
            ambiguities_m,
        };

        let solutions = solve_kinematic_ppp(&source, &epochs, driver_config(initial_state))
            .expect("kinematic solve should succeed");
        let traces = solutions
            .iter()
            .map(|solution| position_trace(solution.position_covariance_m2))
            .collect::<Vec<_>>();

        assert!(traces.windows(2).all(|trace| trace[1] <= trace[0] + 1.0e-8));
        assert!(traces.last().copied().unwrap() < traces[0] * 0.5);
    }

    fn single_epoch_update_fixture() -> (
        KinematicFakeSource,
        FloatEpoch,
        KinematicState,
        KinematicConfig,
    ) {
        let sats = [
            (1u8, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
            (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
            (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
            (4, [-18_200_000.0, -16_000_000.0, 21_000_000.0]),
            (5, [22_000_000.0, -12_000_000.0, 20_200_000.0]),
        ];
        let ids = sats
            .iter()
            .map(|(prn, _)| {
                GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id")
            })
            .collect::<Vec<_>>();
        let source = KinematicFakeSource {
            states: ids
                .iter()
                .zip(sats.iter())
                .map(|(id, (_, pos))| (*id, *pos))
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
                let pred = predict(
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
                .expect("synthetic satellite should predict");
                let code_m = pred.geometric_range_m + clock_m;
                let ambiguity_m = ambiguities_m.get(&id.to_string()).copied().unwrap();
                FloatObservation {
                    sat: *id,
                    satellite_id: id.to_string(),
                    ambiguity_id: id.to_string(),
                    code_m,
                    phase_m: code_m + ambiguity_m,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                }
            })
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
        let initial_state = KinematicState {
            position_m: [truth[0] + 5.0, truth[1] - 4.0, truth[2] + 3.0],
            clock_m: 0.0,
            ztd_residual_m: 0.0,
            ambiguities_m,
        };
        let config = KinematicConfig {
            initial_state: initial_state.clone(),
            initial_covariance_m2: diagonal_covariance(initial_state.dimension(), 1.0e8),
            weights: MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            ..KinematicConfig::default()
        };
        (source, epoch, initial_state, config)
    }

    fn synthetic_kinematic_arc(
        truths: &[[f64; 3]],
        clocks_m: &[f64],
    ) -> (KinematicFakeSource, Vec<FloatEpoch>, BTreeMap<String, f64>) {
        let sats = [
            (1u8, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
            (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
            (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
            (4, [-18_700_000.0, -18_200_000.0, 22_000_000.0]),
            (5, [23_500_000.0, 3_200_000.0, -18_900_000.0]),
            (6, [-7_500_000.0, 25_800_000.0, -16_000_000.0]),
        ];
        let ids = sats
            .iter()
            .map(|(prn, _)| {
                GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id")
            })
            .collect::<Vec<_>>();
        let source = KinematicFakeSource {
            states: ids
                .iter()
                .zip(sats.iter())
                .map(|(id, (_, pos))| (*id, *pos))
                .collect(),
        };
        let ambiguities_m = ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
            .collect::<BTreeMap<_, _>>();
        let epochs = truths
            .iter()
            .zip(clocks_m.iter())
            .enumerate()
            .map(|(epoch_idx, (truth, clock_m))| {
                let t_rx_j2000_s = epoch_idx as f64 * 60.0;
                let observations = ids
                    .iter()
                    .map(|id| {
                        let pred = predict(
                            &source,
                            *id,
                            *truth,
                            t_rx_j2000_s,
                            PredictOptions {
                                carrier_hz: F_L1_HZ,
                                light_time: true,
                                sagnac: true,
                            },
                        )
                        .expect("synthetic satellite should predict");
                        let code_m = pred.geometric_range_m + clock_m;
                        let ambiguity_m = ambiguities_m.get(&id.to_string()).copied().unwrap();
                        FloatObservation {
                            sat: *id,
                            satellite_id: id.to_string(),
                            ambiguity_id: id.to_string(),
                            code_m,
                            phase_m: code_m + ambiguity_m,
                            freq1_hz: 0.0,
                            freq2_hz: 0.0,
                        }
                    })
                    .collect();
                FloatEpoch {
                    epoch: CivilDateTime {
                        year: 2020,
                        month: 6,
                        day: 24,
                        hour: 12,
                        minute: epoch_idx as u8,
                        second: 0.0,
                    },
                    jd_whole: 2_459_024.5,
                    jd_fraction: 0.5 + t_rx_j2000_s / 86_400.0,
                    t_rx_j2000_s,
                    observations,
                }
            })
            .collect();
        (source, epochs, ambiguities_m)
    }

    fn driver_config(initial_state: KinematicState) -> KinematicConfig {
        KinematicConfig {
            initial_covariance_m2: diagonal_covariance(initial_state.dimension(), 1.0e6),
            initial_state,
            process_noise: KinematicProcessNoise {
                position: KinematicPositionProcessNoise::RandomWalk {
                    spectral_density_m2_s: 0.0,
                },
                clock_white_m2_s: 0.0,
                ztd_random_walk_m2_s: 0.0,
                ambiguity_hold_m2_s: 0.0,
            },
            weights: MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            ..KinematicConfig::default()
        }
    }

    fn float_config_from_kinematic(config: &KinematicConfig) -> super::super::FloatSolveConfig {
        super::super::FloatSolveConfig {
            weights: config.weights,
            tropo: config.tropo,
            corrections: config.corrections.clone(),
            opts: super::super::FloatSolveOptions {
                max_iterations: 20,
                position_tolerance_m: 1.0e-8,
                clock_tolerance_m: 1.0e-8,
                ambiguity_tolerance_m: 1.0e-8,
                ztd_tolerance_m: 1.0e-8,
            },
            residual_screen: false,
        }
    }

    fn position_at(start: [f64; 3], velocity_m_s: [f64; 3], dt_s: f64) -> [f64; 3] {
        [
            start[0] + velocity_m_s[0] * dt_s,
            start[1] + velocity_m_s[1] * dt_s,
            start[2] + velocity_m_s[2] * dt_s,
        ]
    }

    fn position_trace(covariance_m2: [[f64; 3]; 3]) -> f64 {
        covariance_m2[0][0] + covariance_m2[1][1] + covariance_m2[2][2]
    }

    fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}
