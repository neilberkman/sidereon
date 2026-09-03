//! Constant-velocity track filtering and fixed-interval RTS smoothing.
//!
//! This module is a no-IMU position-domain estimator. The state vector is
//! `[position, velocity]` in a caller-selected Cartesian frame. Position-only
//! updates use `H = [I, 0]`; position-and-velocity updates use `H = I`.
//!
//! Prediction uses the white-noise-acceleration constant-velocity model
//!
//! `F = [[I, dt I], [0, I]]`
//!
//! `Q = q [[dt^3 / 3 I, dt^2 / 2 I], [dt^2 / 2 I, dt I]]`
//!
//! where `q` is the acceleration variance spectral density in `m^2 / s^3`.
//!
//! Measurement correction uses the linear Kalman equations
//!
//! `nu = z - H x`
//!
//! `S = H P H' + R`
//!
//! `K = P H' S^-1`
//!
//! `x = x + K nu`
//!
//! `P = P - K H P`.
//!
//! The update methods also expose `nu`, `S`, `K`, and normalized innovation
//! squared, `nu' S^-1 nu`, so callers can gate fixes before applying them.
//!
//! The backward smoother follows the Rauch-Tung-Striebel recursion:
//!
//! `G_k = P_k F_{k+1}' (P_{k+1|k})^-1`
//!
//! `x_k^s = x_k + G_k (x_{k+1}^s - x_{k+1|k})`
//!
//! `P_k^s = P_k + G_k (P_{k+1}^s - P_{k+1|k}) G_k'`.

use crate::astro::math::linear::{solve_flat_normal_square_root_into, FlatCholeskySolveScratch};
use crate::dop::PositionCovariance;
use crate::estimation::primitives::{nis_gate_threshold, NisGate};
use crate::validate::{self, FieldError};

/// Cartesian frame used by a track filter.
///
/// The observation vector and observation covariance must be expressed in this
/// same frame. The filter does not rotate between frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackCoordinateFrame {
    /// Earth-Centered-Earth-Fixed position in metres.
    Ecef,
    /// Local East-North-Up position in metres, using an origin managed by the caller.
    Enu,
    /// Any fixed Cartesian frame in metres, with axes defined by the caller.
    CallerDefinedCartesian,
}

/// Error returned by no-IMU track filtering and smoothing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TrackError {
    /// A public input was non-finite or outside its documented range.
    #[error("invalid track input {field}: {reason}")]
    InvalidInput {
        /// Name of the invalid field.
        field: &'static str,
        /// Reason suitable for logs and tests.
        reason: &'static str,
    },
    /// A vector or matrix dimension did not match the filter state.
    #[error("invalid track dimension {field}: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Name of the invalid field.
        field: &'static str,
        /// Expected length, row count, or column count.
        expected: usize,
        /// Actual length, row count, or column count.
        actual: usize,
    },
    /// A covariance was not positive semidefinite within numerical bounds.
    #[error("track covariance {field} is not positive semidefinite")]
    NonPositiveSemidefinite {
        /// Name of the covariance field.
        field: &'static str,
    },
    /// A covariance that must be inverted was not positive definite.
    #[error("track covariance {field} is not positive definite")]
    NonPositiveDefinite {
        /// Name of the covariance field.
        field: &'static str,
    },
}

/// Configuration for [`TrackFilter`].
///
/// The state covariance is a `2n x 2n` row-major matrix over
/// `[position_m, velocity_m_s]`. Its blocks have units `m^2`, `m^2/s`, and
/// `m^2/s^2`.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct TrackFilterConfig {
    /// Cartesian frame of the state, observations, and covariances.
    pub frame: TrackCoordinateFrame,
    /// Initial epoch in seconds in a caller-selected monotonic time base.
    pub initial_t_s: f64,
    /// Initial position components in metres.
    pub initial_position_m: Vec<f64>,
    /// Initial velocity components in metres per second.
    pub initial_velocity_m_s: Vec<f64>,
    /// Initial covariance over `[position, velocity]`.
    pub initial_covariance: Vec<Vec<f64>>,
    /// White acceleration variance spectral density in `m^2 / s^3`.
    pub acceleration_variance_spectral_density_m2_s3: f64,
}

impl TrackFilterConfig {
    /// Build and validate a configuration from every state field.
    pub fn new(
        frame: TrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: Vec<f64>,
        initial_velocity_m_s: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> Result<Self, TrackError> {
        Self::from_position_velocity(
            frame,
            initial_t_s,
            initial_position_m,
            initial_velocity_m_s,
            initial_covariance,
            acceleration_variance_spectral_density_m2_s3,
        )
    }

    /// Build a configuration from a position fix and uncertain initial velocity.
    ///
    /// This is the position-only entry point. The velocity starts at zero and
    /// its diagonal variance is `initial_velocity_variance_m2_s2`.
    pub fn from_position(
        frame: TrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: Vec<f64>,
        position_covariance_m2: Vec<Vec<f64>>,
        initial_velocity_variance_m2_s2: f64,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> Result<Self, TrackError> {
        let dimension = initial_position_m.len();
        validate_dimension(dimension, "initial_position_m")?;
        validate_vector_len(&initial_position_m, dimension, "initial_position_m")?;
        validate_covariance_matrix(&position_covariance_m2, dimension, "position_covariance_m2")?;
        validate_nonnegative(
            initial_velocity_variance_m2_s2,
            "initial_velocity_variance_m2_s2",
        )?;

        let mut covariance = vec![vec![0.0; 2 * dimension]; 2 * dimension];
        for row in 0..dimension {
            for col in 0..dimension {
                covariance[row][col] = position_covariance_m2[row][col];
            }
            covariance[dimension + row][dimension + row] = initial_velocity_variance_m2_s2;
        }

        Ok(Self {
            frame,
            initial_t_s,
            initial_position_m,
            initial_velocity_m_s: vec![0.0; dimension],
            initial_covariance: covariance,
            acceleration_variance_spectral_density_m2_s3,
        })
    }

    /// Build a configuration from a position-and-velocity state.
    pub fn from_position_velocity(
        frame: TrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: Vec<f64>,
        initial_velocity_m_s: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> Result<Self, TrackError> {
        let config = Self {
            frame,
            initial_t_s,
            initial_position_m,
            initial_velocity_m_s,
            initial_covariance,
            acceleration_variance_spectral_density_m2_s3,
        };
        config.validate()?;
        Ok(config)
    }

    /// Return the position dimension.
    pub fn dimension(&self) -> usize {
        self.initial_position_m.len()
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), TrackError> {
        validate_time(self.initial_t_s, "initial_t_s")?;
        validate_dimension(self.dimension(), "initial_position_m")?;
        validate_vector_len(
            &self.initial_position_m,
            self.dimension(),
            "initial_position_m",
        )?;
        validate_vector_len(
            &self.initial_velocity_m_s,
            self.dimension(),
            "initial_velocity_m_s",
        )?;
        validate_covariance_matrix(
            &self.initial_covariance,
            2 * self.dimension(),
            "initial_covariance",
        )?;
        validate_nonnegative(
            self.acceleration_variance_spectral_density_m2_s3,
            "acceleration_variance_spectral_density_m2_s3",
        )
    }
}

/// State of a no-IMU constant-velocity track filter.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackState {
    /// Cartesian frame of the state and covariance.
    pub frame: TrackCoordinateFrame,
    /// Epoch in seconds in the caller's monotonic time base.
    pub t_s: f64,
    /// Position components in metres.
    pub position_m: Vec<f64>,
    /// Velocity components in metres per second.
    pub velocity_m_s: Vec<f64>,
    /// Covariance over `[position, velocity]`.
    pub covariance: Vec<Vec<f64>>,
}

impl TrackState {
    /// Build and validate one track state.
    pub fn new(
        frame: TrackCoordinateFrame,
        t_s: f64,
        position_m: Vec<f64>,
        velocity_m_s: Vec<f64>,
        covariance: Vec<Vec<f64>>,
    ) -> Result<Self, TrackError> {
        let state = Self {
            frame,
            t_s,
            position_m,
            velocity_m_s,
            covariance,
        };
        state.validate()?;
        Ok(state)
    }

    /// Return the position dimension.
    pub fn dimension(&self) -> usize {
        self.position_m.len()
    }

    /// Return the full state dimension.
    pub fn state_dimension(&self) -> usize {
        2 * self.dimension()
    }

    /// Return `[position, velocity]` as an owned vector.
    pub fn state_vector(&self) -> Vec<f64> {
        state_vector(&self.position_m, &self.velocity_m_s)
    }

    /// Return the leading position covariance block.
    pub fn position_covariance_m2(&self) -> Vec<Vec<f64>> {
        let dimension = self.dimension();
        self.covariance
            .iter()
            .take(dimension)
            .map(|row| row[..dimension].to_vec())
            .collect()
    }

    /// Return the 3D position covariance block.
    pub fn position_covariance3_m2(&self) -> Result<[[f64; 3]; 3], TrackError> {
        if self.dimension() != 3 {
            return Err(TrackError::DimensionMismatch {
                field: "position_covariance3_m2",
                expected: 3,
                actual: self.dimension(),
            });
        }
        Ok(matrix3_from_rows(&self.position_covariance_m2()))
    }

    /// Return the 3D position vector.
    pub fn position3_m(&self) -> Result<[f64; 3], TrackError> {
        if self.dimension() != 3 {
            return Err(TrackError::DimensionMismatch {
                field: "position3_m",
                expected: 3,
                actual: self.dimension(),
            });
        }
        Ok([self.position_m[0], self.position_m[1], self.position_m[2]])
    }

    /// Return the 3D velocity vector.
    pub fn velocity3_m_s(&self) -> Result<[f64; 3], TrackError> {
        if self.dimension() != 3 {
            return Err(TrackError::DimensionMismatch {
                field: "velocity3_m_s",
                expected: 3,
                actual: self.dimension(),
            });
        }
        Ok([
            self.velocity_m_s[0],
            self.velocity_m_s[1],
            self.velocity_m_s[2],
        ])
    }

    /// Validate vector dimensions, finiteness, and covariance.
    pub fn validate(&self) -> Result<(), TrackError> {
        validate_time(self.t_s, "t_s")?;
        validate_dimension(self.dimension(), "position_m")?;
        validate_vector_len(&self.position_m, self.dimension(), "position_m")?;
        validate_vector_len(&self.velocity_m_s, self.dimension(), "velocity_m_s")?;
        validate_covariance_matrix(&self.covariance, self.state_dimension(), "covariance")
    }
}

/// Prediction result returned by [`TrackFilter::predict`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrackPrediction {
    /// Time step in seconds.
    pub dt_s: f64,
    /// State-transition matrix used for the prediction.
    pub transition: Vec<Vec<f64>>,
    /// Discrete process-noise covariance added during prediction.
    pub process_noise: Vec<Vec<f64>>,
    /// Predicted state after the time step.
    pub predicted: TrackState,
}

/// Innovation report for a pending or applied track update.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackInnovation {
    /// Measurement residual `z - H x`.
    pub innovation: Vec<f64>,
    /// Innovation covariance `S = H P H' + R`.
    pub innovation_covariance: Vec<Vec<f64>>,
    /// Normalized innovation squared, `innovation' S^-1 innovation`.
    pub nis: f64,
}

impl TrackInnovation {
    /// Evaluate the innovation against a chi-square NIS gate.
    pub fn gate(&self, confidence: f64) -> Result<NisGate, TrackError> {
        let threshold =
            nis_gate_threshold(self.innovation.len(), confidence).map_err(|err| match err {
                crate::estimation::PrimitiveError::InvalidInput { field, reason } => {
                    invalid_input(field, reason)
                }
            })?;
        Ok(NisGate {
            nis: self.nis,
            threshold,
            in_gate: self.nis <= threshold,
            dof: self.innovation.len(),
        })
    }
}

/// Update result returned after a covariance-weighted correction.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackUpdate {
    /// State before the correction.
    pub predicted: TrackState,
    /// State after the correction.
    pub updated: TrackState,
    /// Innovation report for the correction.
    pub innovation: TrackInnovation,
    /// Kalman gain used to map the innovation into the state.
    pub kalman_gain: Vec<Vec<f64>>,
}

/// Gated update result.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackGatedUpdate {
    /// NIS gate result.
    pub gate: NisGate,
    /// Applied update when the measurement passed the gate.
    pub update: Option<TrackUpdate>,
    /// State after the gate decision.
    pub state: TrackState,
}

/// No-IMU constant-velocity Kalman filter over a Cartesian position track.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackFilter {
    state: TrackState,
    acceleration_variance_spectral_density_m2_s3: f64,
}

impl TrackFilter {
    /// Build a filter from a validated configuration.
    pub fn new(config: TrackFilterConfig) -> Result<Self, TrackError> {
        config.validate()?;
        let state = TrackState::new(
            config.frame,
            config.initial_t_s,
            config.initial_position_m,
            config.initial_velocity_m_s,
            config.initial_covariance,
        )?;
        Ok(Self {
            state,
            acceleration_variance_spectral_density_m2_s3: config
                .acceleration_variance_spectral_density_m2_s3,
        })
    }

    /// Build a filter from a position fix and uncertain initial velocity.
    pub fn from_position(
        frame: TrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: Vec<f64>,
        position_covariance_m2: Vec<Vec<f64>>,
        initial_velocity_variance_m2_s2: f64,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> Result<Self, TrackError> {
        Self::new(TrackFilterConfig::from_position(
            frame,
            initial_t_s,
            initial_position_m,
            position_covariance_m2,
            initial_velocity_variance_m2_s2,
            acceleration_variance_spectral_density_m2_s3,
        )?)
    }

    /// Build a filter from a 3D position fix and uncertain initial velocity.
    pub fn from_position3(
        frame: TrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: [f64; 3],
        position_covariance_m2: [[f64; 3]; 3],
        initial_velocity_variance_m2_s2: f64,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> Result<Self, TrackError> {
        Self::from_position(
            frame,
            initial_t_s,
            initial_position_m.to_vec(),
            matrix3_to_rows(position_covariance_m2),
            initial_velocity_variance_m2_s2,
            acceleration_variance_spectral_density_m2_s3,
        )
    }

    /// Build a filter from a position-and-velocity state.
    pub fn from_position_velocity(
        frame: TrackCoordinateFrame,
        initial_t_s: f64,
        initial_position_m: Vec<f64>,
        initial_velocity_m_s: Vec<f64>,
        initial_covariance: Vec<Vec<f64>>,
        acceleration_variance_spectral_density_m2_s3: f64,
    ) -> Result<Self, TrackError> {
        Self::new(TrackFilterConfig::from_position_velocity(
            frame,
            initial_t_s,
            initial_position_m,
            initial_velocity_m_s,
            initial_covariance,
            acceleration_variance_spectral_density_m2_s3,
        )?)
    }

    /// Borrow the current state.
    pub fn state(&self) -> &TrackState {
        &self.state
    }

    /// Return the position dimension.
    pub fn dimension(&self) -> usize {
        self.state.dimension()
    }

    /// Return the white acceleration variance spectral density in `m^2 / s^3`.
    pub fn acceleration_variance_spectral_density_m2_s3(&self) -> f64 {
        self.acceleration_variance_spectral_density_m2_s3
    }

    /// Advance the state with the constant-velocity prediction model.
    pub fn predict(&mut self, dt_s: f64) -> Result<TrackPrediction, TrackError> {
        validate_positive(dt_s, "dt_s")?;
        let dimension = self.dimension();
        let transition = transition_matrix(dimension, dt_s);
        let process_noise = process_noise_matrix(
            dimension,
            dt_s,
            self.acceleration_variance_spectral_density_m2_s3,
        );
        let predicted_vector = matvec(&transition, &self.state.state_vector())?;
        let transition_t = transpose(&transition)?;
        let fp = matmul(&transition, &self.state.covariance)?;
        let mut predicted_covariance = matrix_add(&matmul(&fp, &transition_t)?, &process_noise)?;
        copy_lower_to_upper(&mut predicted_covariance);
        validate_covariance_matrix(&predicted_covariance, 2 * dimension, "covariance")?;

        self.state = TrackState::new(
            self.state.frame,
            self.state.t_s + dt_s,
            predicted_vector[..dimension].to_vec(),
            predicted_vector[dimension..].to_vec(),
            predicted_covariance,
        )?;

        Ok(TrackPrediction {
            dt_s,
            transition,
            process_noise,
            predicted: self.state.clone(),
        })
    }

    /// Advance the state and record the prediction for RTS smoothing.
    pub fn predict_recorded(
        &mut self,
        dt_s: f64,
        history: &mut TrackRtsHistoryBuilder,
    ) -> Result<TrackPrediction, TrackError> {
        let mut working_filter = self.clone();
        let mut working_history = history.clone();
        let prediction = working_filter.predict(dt_s)?;
        working_history
            .record_prediction(prediction.predicted.clone(), prediction.transition.clone())?;
        *self = working_filter;
        *history = working_history;
        Ok(prediction)
    }

    /// Compute the innovation for a position-only observation without applying it.
    pub fn position_innovation(
        &self,
        observation_position_m: &[f64],
        observation_covariance_m2: &[Vec<f64>],
    ) -> Result<TrackInnovation, TrackError> {
        let dimension = self.dimension();
        validate_vector_len(observation_position_m, dimension, "observation_position_m")?;
        validate_covariance_matrix(
            observation_covariance_m2,
            dimension,
            "observation_covariance_m2",
        )?;
        let innovation = observation_position_m
            .iter()
            .zip(&self.state.position_m)
            .map(|(obs, pred)| obs - pred)
            .collect::<Vec<_>>();
        let predicted_position_covariance = self.state.position_covariance_m2();
        let innovation_covariance =
            matrix_add(&predicted_position_covariance, observation_covariance_m2)?;
        validate_spd_matrix(&innovation_covariance, dimension, "innovation_covariance")?;
        let nis = nis_from_innovation(&innovation, &innovation_covariance)?;
        Ok(TrackInnovation {
            innovation,
            innovation_covariance,
            nis,
        })
    }

    /// Compute the innovation for a position-and-velocity observation without applying it.
    pub fn state_innovation(
        &self,
        observation_state: &[f64],
        observation_covariance: &[Vec<f64>],
    ) -> Result<TrackInnovation, TrackError> {
        let state_dimension = self.state.state_dimension();
        validate_vector_len(observation_state, state_dimension, "observation_state")?;
        validate_covariance_matrix(
            observation_covariance,
            state_dimension,
            "observation_covariance",
        )?;
        let predicted = self.state.state_vector();
        let innovation = observation_state
            .iter()
            .zip(predicted)
            .map(|(obs, pred)| obs - pred)
            .collect::<Vec<_>>();
        let innovation_covariance = matrix_add(&self.state.covariance, observation_covariance)?;
        validate_spd_matrix(
            &innovation_covariance,
            state_dimension,
            "innovation_covariance",
        )?;
        let nis = nis_from_innovation(&innovation, &innovation_covariance)?;
        Ok(TrackInnovation {
            innovation,
            innovation_covariance,
            nis,
        })
    }

    /// Apply a position-only observation with covariance weighting.
    pub fn update_position(
        &mut self,
        observation_position_m: &[f64],
        observation_covariance_m2: &[Vec<f64>],
    ) -> Result<TrackUpdate, TrackError> {
        let predicted = self.state.clone();
        let update = self.position_update_from_predicted(
            predicted,
            observation_position_m,
            observation_covariance_m2,
        )?;
        self.state = update.updated.clone();
        Ok(update)
    }

    /// Apply a 3D position-only observation with covariance weighting.
    pub fn update_position3(
        &mut self,
        observation_position_m: [f64; 3],
        observation_covariance_m2: [[f64; 3]; 3],
    ) -> Result<TrackUpdate, TrackError> {
        self.update_position(
            &observation_position_m,
            &matrix3_to_rows(observation_covariance_m2),
        )
    }

    /// Apply a 3D position observation using an existing GNSS position covariance.
    ///
    /// `TrackCoordinateFrame::Ecef` consumes `PositionCovariance::ecef_m2`.
    /// `TrackCoordinateFrame::Enu` consumes `PositionCovariance::enu_m2`.
    pub fn update_position_covariance(
        &mut self,
        observation_position_m: [f64; 3],
        observation_covariance: &PositionCovariance,
    ) -> Result<TrackUpdate, TrackError> {
        let covariance = covariance_for_frame(self.state.frame, observation_covariance)?;
        self.update_position3(observation_position_m, covariance)
    }

    /// Apply a position-and-velocity observation with covariance weighting.
    pub fn update_state(
        &mut self,
        observation_state: &[f64],
        observation_covariance: &[Vec<f64>],
    ) -> Result<TrackUpdate, TrackError> {
        let predicted = self.state.clone();
        let update =
            self.state_update_from_predicted(predicted, observation_state, observation_covariance)?;
        self.state = update.updated.clone();
        Ok(update)
    }

    /// Apply a gated position-only observation.
    ///
    /// If the NIS gate rejects the observation, the current predicted state is
    /// left unchanged.
    pub fn update_position_gated(
        &mut self,
        observation_position_m: &[f64],
        observation_covariance_m2: &[Vec<f64>],
        confidence: f64,
    ) -> Result<TrackGatedUpdate, TrackError> {
        let innovation =
            self.position_innovation(observation_position_m, observation_covariance_m2)?;
        let gate = innovation.gate(confidence)?;
        if gate.in_gate {
            let update = self.update_position(observation_position_m, observation_covariance_m2)?;
            Ok(TrackGatedUpdate {
                gate,
                state: self.state.clone(),
                update: Some(update),
            })
        } else {
            Ok(TrackGatedUpdate {
                gate,
                state: self.state.clone(),
                update: None,
            })
        }
    }

    /// Apply a position-only observation and record the epoch for RTS smoothing.
    pub fn update_position_recorded(
        &mut self,
        observation_position_m: &[f64],
        observation_covariance_m2: &[Vec<f64>],
        history: &mut TrackRtsHistoryBuilder,
    ) -> Result<TrackUpdate, TrackError> {
        history.validate_update_ready()?;
        let predicted = self.state.clone();
        let mut working_filter = self.clone();
        let mut working_history = history.clone();
        let update =
            working_filter.update_position(observation_position_m, observation_covariance_m2)?;
        working_history.record_update(predicted, working_filter.state.clone())?;
        *self = working_filter;
        *history = working_history;
        Ok(update)
    }

    /// Apply a gated position-only observation and record accepted or rejected epochs.
    pub fn update_position_gated_recorded(
        &mut self,
        observation_position_m: &[f64],
        observation_covariance_m2: &[Vec<f64>],
        confidence: f64,
        history: &mut TrackRtsHistoryBuilder,
    ) -> Result<TrackGatedUpdate, TrackError> {
        history.validate_update_ready()?;
        let predicted = self.state.clone();
        let mut working_filter = self.clone();
        let mut working_history = history.clone();
        let gated = working_filter.update_position_gated(
            observation_position_m,
            observation_covariance_m2,
            confidence,
        )?;
        working_history.record_update(predicted, working_filter.state.clone())?;
        *self = working_filter;
        *history = working_history;
        Ok(gated)
    }

    /// Record the current predicted state as an epoch with no measurement update.
    pub fn record_prediction_only(
        &self,
        history: &mut TrackRtsHistoryBuilder,
    ) -> Result<(), TrackError> {
        history.record_update(self.state.clone(), self.state.clone())
    }

    fn position_update_from_predicted(
        &self,
        predicted: TrackState,
        observation_position_m: &[f64],
        observation_covariance_m2: &[Vec<f64>],
    ) -> Result<TrackUpdate, TrackError> {
        let dimension = predicted.dimension();
        let innovation =
            self.position_innovation(observation_position_m, observation_covariance_m2)?;
        let cross = predicted
            .covariance
            .iter()
            .map(|row| row[..dimension].to_vec())
            .collect::<Vec<_>>();
        let kalman_gain = gain_from_cross(&cross, &innovation.innovation_covariance)?;
        let update_delta = matvec(&kalman_gain, &innovation.innovation)?;
        let mut updated_vector = predicted.state_vector();
        for (value, delta) in updated_vector.iter_mut().zip(update_delta) {
            *value += delta;
        }

        let hp = predicted
            .covariance
            .iter()
            .take(dimension)
            .cloned()
            .collect::<Vec<_>>();
        let khp = matmul(&kalman_gain, &hp)?;
        let mut covariance = matrix_sub(&predicted.covariance, &khp)?;
        copy_lower_to_upper(&mut covariance);
        validate_covariance_matrix(&covariance, 2 * dimension, "covariance")?;

        let updated = TrackState::new(
            predicted.frame,
            predicted.t_s,
            updated_vector[..dimension].to_vec(),
            updated_vector[dimension..].to_vec(),
            covariance,
        )?;
        Ok(TrackUpdate {
            predicted,
            updated,
            innovation,
            kalman_gain,
        })
    }

    fn state_update_from_predicted(
        &self,
        predicted: TrackState,
        observation_state: &[f64],
        observation_covariance: &[Vec<f64>],
    ) -> Result<TrackUpdate, TrackError> {
        let state_dimension = predicted.state_dimension();
        let innovation = self.state_innovation(observation_state, observation_covariance)?;
        let kalman_gain =
            gain_from_cross(&predicted.covariance, &innovation.innovation_covariance)?;
        let update_delta = matvec(&kalman_gain, &innovation.innovation)?;
        let mut updated_vector = predicted.state_vector();
        for (value, delta) in updated_vector.iter_mut().zip(update_delta) {
            *value += delta;
        }

        let khp = matmul(&kalman_gain, &predicted.covariance)?;
        let mut covariance = matrix_sub(&predicted.covariance, &khp)?;
        copy_lower_to_upper(&mut covariance);
        validate_covariance_matrix(&covariance, state_dimension, "covariance")?;

        let dimension = predicted.dimension();
        let updated = TrackState::new(
            predicted.frame,
            predicted.t_s,
            updated_vector[..dimension].to_vec(),
            updated_vector[dimension..].to_vec(),
            covariance,
        )?;
        Ok(TrackUpdate {
            predicted,
            updated,
            innovation,
            kalman_gain,
        })
    }
}

/// One epoch in a recorded forward pass for track RTS smoothing.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackRtsEpoch {
    /// Epoch in seconds in the caller's monotonic time base.
    pub t_s: f64,
    /// Predicted state before a correction at this epoch.
    pub predicted: TrackState,
    /// Updated state after the correction, or the same as `predicted` for no update.
    pub updated: TrackState,
    /// Transition from the previous updated epoch to this predicted epoch.
    pub transition_from_previous: Option<Vec<Vec<f64>>>,
}

impl TrackRtsEpoch {
    /// Build and validate one recorded epoch.
    pub fn new(
        predicted: TrackState,
        updated: TrackState,
        transition_from_previous: Option<Vec<Vec<f64>>>,
    ) -> Result<Self, TrackError> {
        let epoch = Self {
            t_s: predicted.t_s,
            predicted,
            updated,
            transition_from_previous,
        };
        epoch.validate()?;
        Ok(epoch)
    }

    /// Validate state frames, dimensions, covariance, and transition dimensions.
    pub fn validate(&self) -> Result<(), TrackError> {
        self.predicted.validate()?;
        self.updated.validate()?;
        if self.predicted.frame != self.updated.frame {
            return Err(invalid_input(
                "frame",
                "predicted and updated frames differ",
            ));
        }
        if self.predicted.dimension() != self.updated.dimension() {
            return Err(TrackError::DimensionMismatch {
                field: "dimension",
                expected: self.predicted.dimension(),
                actual: self.updated.dimension(),
            });
        }
        if self.predicted.t_s.to_bits() != self.t_s.to_bits()
            || self.updated.t_s.to_bits() != self.t_s.to_bits()
        {
            return Err(invalid_input("t_s", "state epochs must match"));
        }
        if let Some(transition) = &self.transition_from_previous {
            validate_square_matrix(transition, self.updated.state_dimension(), "transition")?;
        }
        Ok(())
    }
}

/// Recorded history accepted by [`rts_smooth`].
#[derive(Debug, Clone, PartialEq)]
pub struct TrackRtsHistory {
    /// Ordered epochs to smooth.
    pub epochs: Vec<TrackRtsEpoch>,
}

impl TrackRtsHistory {
    /// Build and validate an ordered track history.
    pub fn new(epochs: Vec<TrackRtsEpoch>) -> Result<Self, TrackError> {
        let history = Self { epochs };
        history.validate()?;
        Ok(history)
    }

    /// Validate epoch order, frame consistency, and transition placement.
    pub fn validate(&self) -> Result<(), TrackError> {
        if self.epochs.is_empty() {
            return Err(invalid_input("history", "must not be empty"));
        }
        let frame = self.epochs[0].updated.frame;
        let dimension = self.epochs[0].updated.dimension();
        for (idx, epoch) in self.epochs.iter().enumerate() {
            epoch.validate()?;
            if epoch.updated.frame != frame {
                return Err(invalid_input("frame", "history frames differ"));
            }
            if epoch.updated.dimension() != dimension {
                return Err(TrackError::DimensionMismatch {
                    field: "dimension",
                    expected: dimension,
                    actual: epoch.updated.dimension(),
                });
            }
            match (idx, &epoch.transition_from_previous) {
                (0, None) => {}
                (0, Some(_)) => {
                    return Err(invalid_input(
                        "transition_from_previous",
                        "first epoch must not have a transition",
                    ));
                }
                (_, Some(transition)) => {
                    validate_square_matrix(transition, 2 * dimension, "transition")?;
                    if epoch.t_s <= self.epochs[idx - 1].t_s {
                        return Err(invalid_input("history", "epochs must be increasing"));
                    }
                }
                (_, None) => {
                    return Err(invalid_input(
                        "transition_from_previous",
                        "missing transition",
                    ));
                }
            }
        }
        Ok(())
    }
}

/// Builder for recording a forward track-filter pass for RTS smoothing.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackRtsHistoryBuilder {
    epochs: Vec<TrackRtsEpoch>,
    pending_transition: Option<Vec<Vec<f64>>>,
    pending_predicted: Option<TrackState>,
}

impl TrackRtsHistoryBuilder {
    /// Start a history from the filter's current state.
    pub fn from_filter(filter: &TrackFilter) -> Result<Self, TrackError> {
        let initial = TrackRtsEpoch::new(filter.state.clone(), filter.state.clone(), None)?;
        Ok(Self {
            epochs: vec![initial],
            pending_transition: None,
            pending_predicted: None,
        })
    }

    /// Start an empty history for callers that record epochs manually.
    pub const fn empty() -> Self {
        Self {
            epochs: Vec::new(),
            pending_transition: None,
            pending_predicted: None,
        }
    }

    /// Record a prediction interval.
    pub fn record_prediction(
        &mut self,
        predicted: TrackState,
        transition: Vec<Vec<f64>>,
    ) -> Result<(), TrackError> {
        predicted.validate()?;
        validate_square_matrix(&transition, predicted.state_dimension(), "transition")?;
        let combined = if let Some(previous) = &self.pending_transition {
            matmul(&transition, previous)?
        } else {
            transition
        };
        self.pending_transition = Some(combined);
        self.pending_predicted = Some(predicted);
        Ok(())
    }

    /// Append an update epoch, consuming the transition accumulated since the previous epoch.
    pub fn record_update(
        &mut self,
        predicted: TrackState,
        updated: TrackState,
    ) -> Result<(), TrackError> {
        let transition = if self.epochs.is_empty() {
            None
        } else {
            Some(
                self.pending_transition
                    .clone()
                    .ok_or_else(|| invalid_input("transition", "missing propagated interval"))?,
            )
        };
        if let Some(pending) = &self.pending_predicted {
            if pending.t_s.to_bits() != predicted.t_s.to_bits() {
                return Err(invalid_input(
                    "predicted",
                    "does not match pending prediction",
                ));
            }
            if pending.dimension() != predicted.dimension() || pending.frame != predicted.frame {
                return Err(invalid_input("predicted", "does not match pending state"));
            }
        }
        let epoch = TrackRtsEpoch::new(predicted, updated, transition)?;
        let mut epochs = self.epochs.clone();
        epochs.push(epoch);
        TrackRtsHistory {
            epochs: epochs.clone(),
        }
        .validate()?;
        self.epochs = epochs;
        self.pending_transition = None;
        self.pending_predicted = None;
        Ok(())
    }

    /// Return a validated history.
    pub fn finish(self) -> Result<TrackRtsHistory, TrackError> {
        if self.pending_transition.is_some() {
            return Err(invalid_input("transition", "unclosed propagated interval"));
        }
        TrackRtsHistory::new(self.epochs)
    }

    /// Validate that a recorded update can be appended.
    pub fn validate_update_ready(&self) -> Result<(), TrackError> {
        if self.epochs.is_empty() || self.pending_transition.is_some() {
            Ok(())
        } else {
            Err(invalid_input("transition", "missing propagated interval"))
        }
    }
}

/// One epoch in a smoothed track.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothedTrackEpoch {
    /// Epoch in seconds in the caller's monotonic time base.
    pub t_s: f64,
    /// Smoothed state and covariance.
    pub state: TrackState,
    /// RTS gain from this epoch to the next, absent at the final epoch.
    pub rts_gain_to_next: Option<Vec<Vec<f64>>>,
}

/// Smoothed track returned by [`rts_smooth`].
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothedTrack {
    /// Smoothed epochs in the same order as the recorded history.
    pub epochs: Vec<SmoothedTrackEpoch>,
}

/// Apply a fixed-interval RTS smoother to a recorded no-IMU track history.
pub fn rts_smooth(history: &TrackRtsHistory) -> Result<SmoothedTrack, TrackError> {
    history.validate()?;
    let len = history.epochs.len();
    let state_dimension = history.epochs[0].updated.state_dimension();
    let mut output: Vec<Option<SmoothedTrackEpoch>> = vec![None; len];

    let final_epoch = &history.epochs[len - 1];
    output[len - 1] = Some(SmoothedTrackEpoch {
        t_s: final_epoch.t_s,
        state: final_epoch.updated.clone(),
        rts_gain_to_next: None,
    });

    for idx in (0..len - 1).rev() {
        let current = &history.epochs[idx];
        let next = &history.epochs[idx + 1];
        let next_smoothed = output[idx + 1]
            .as_ref()
            .expect("next smoothed epoch is populated");
        let transition = next
            .transition_from_previous
            .as_ref()
            .ok_or_else(|| invalid_input("transition_from_previous", "missing transition"))?;
        let gain = rts_gain(
            &current.updated.covariance,
            transition,
            &next.predicted.covariance,
        )?;
        let next_delta = vector_sub(
            &next_smoothed.state.state_vector(),
            &next.predicted.state_vector(),
            "state_delta",
        )?;
        let correction = matvec(&gain, &next_delta)?;
        let mut smoothed_vector = current.updated.state_vector();
        for (value, delta) in smoothed_vector.iter_mut().zip(correction) {
            *value += delta;
        }

        let mut covariance = smoothed_covariance(
            &current.updated.covariance,
            &gain,
            &next.predicted.covariance,
            &next_smoothed.state.covariance,
        )?;
        copy_lower_to_upper(&mut covariance);
        validate_covariance_matrix(&covariance, state_dimension, "smoothed_covariance")?;

        let dimension = current.updated.dimension();
        let state = TrackState::new(
            current.updated.frame,
            current.t_s,
            smoothed_vector[..dimension].to_vec(),
            smoothed_vector[dimension..].to_vec(),
            covariance,
        )?;
        output[idx] = Some(SmoothedTrackEpoch {
            t_s: current.t_s,
            state,
            rts_gain_to_next: Some(gain),
        });
    }

    let epochs = output
        .into_iter()
        .map(|epoch| epoch.expect("all smoothed epochs populated"))
        .collect::<Vec<_>>();
    Ok(SmoothedTrack { epochs })
}

/// Alias for [`rts_smooth`] with the track domain in the function name.
pub fn smooth_track_rts(history: &TrackRtsHistory) -> Result<SmoothedTrack, TrackError> {
    rts_smooth(history)
}

fn covariance_for_frame(
    frame: TrackCoordinateFrame,
    covariance: &PositionCovariance,
) -> Result<[[f64; 3]; 3], TrackError> {
    match frame {
        TrackCoordinateFrame::Ecef => Ok(covariance.ecef_m2),
        TrackCoordinateFrame::Enu => Ok(covariance.enu_m2),
        TrackCoordinateFrame::CallerDefinedCartesian => Err(invalid_input(
            "frame",
            "PositionCovariance requires ECEF or ENU",
        )),
    }
}

fn rts_gain(
    filtered_covariance: &[Vec<f64>],
    transition: &[Vec<f64>],
    predicted_covariance_next: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, TrackError> {
    let dimension = filtered_covariance.len();
    validate_covariance_matrix(filtered_covariance, dimension, "filtered_covariance")?;
    validate_square_matrix(transition, dimension, "transition")?;
    validate_spd_matrix(
        predicted_covariance_next,
        dimension,
        "predicted_covariance_next",
    )?;
    let transition_t = transpose(transition)?;
    let cross = matmul(filtered_covariance, &transition_t)?;
    gain_from_cross(&cross, predicted_covariance_next)
}

fn smoothed_covariance(
    filtered_covariance: &[Vec<f64>],
    gain: &[Vec<f64>],
    predicted_covariance_next: &[Vec<f64>],
    smoothed_covariance_next: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, TrackError> {
    let dimension = filtered_covariance.len();
    validate_covariance_matrix(filtered_covariance, dimension, "filtered_covariance")?;
    validate_square_matrix(gain, dimension, "rts_gain")?;
    validate_covariance_matrix(
        predicted_covariance_next,
        dimension,
        "predicted_covariance_next",
    )?;
    validate_covariance_matrix(
        smoothed_covariance_next,
        dimension,
        "smoothed_covariance_next",
    )?;
    let delta = matrix_sub(smoothed_covariance_next, predicted_covariance_next)?;
    let left = matmul(gain, &delta)?;
    let gain_t = transpose(gain)?;
    let adjustment = matmul(&left, &gain_t)?;
    matrix_add(filtered_covariance, &adjustment)
}

fn gain_from_cross(
    cross: &[Vec<f64>],
    innovation_covariance: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, TrackError> {
    let measurement_dimension = innovation_covariance.len();
    validate_spd_matrix(
        innovation_covariance,
        measurement_dimension,
        "innovation_covariance",
    )?;
    if cross.is_empty() {
        return Err(invalid_input("cross_covariance", "must not be empty"));
    }
    for row in cross {
        validate_vector_len(row, measurement_dimension, "cross_covariance")?;
    }

    if measurement_dimension == 1 {
        let variance = innovation_covariance[0][0];
        return Ok(cross
            .iter()
            .map(|row| vec![row[0] / variance])
            .collect::<Vec<_>>());
    }

    let mut scratch = FlatCholeskySolveScratch::default();
    let flat = flatten(innovation_covariance)?;
    let mut gain = Vec::with_capacity(cross.len());
    for row in cross {
        let solved = solve_flat_normal_square_root_into(&flat, row, &mut scratch)
            .ok_or(TrackError::NonPositiveDefinite {
                field: "innovation_covariance",
            })?
            .to_vec();
        gain.push(solved);
    }
    Ok(gain)
}

fn nis_from_innovation(
    innovation: &[f64],
    innovation_covariance: &[Vec<f64>],
) -> Result<f64, TrackError> {
    if innovation.len() == 1
        && innovation_covariance.len() == 1
        && innovation_covariance[0].len() == 1
    {
        let variance = innovation_covariance[0][0];
        validate_positive(variance, "innovation_covariance")?;
        let nis = innovation[0] * innovation[0] / variance;
        validate_time(nis, "nis")?;
        return Ok(nis);
    }

    let mut scratch = FlatCholeskySolveScratch::default();
    let flat = flatten(innovation_covariance)?;
    let solved = solve_flat_normal_square_root_into(&flat, innovation, &mut scratch).ok_or(
        TrackError::NonPositiveDefinite {
            field: "innovation_covariance",
        },
    )?;
    let mut nis = 0.0;
    for (lhs, rhs) in innovation.iter().zip(solved) {
        nis += lhs * rhs;
    }
    validate_time(nis, "nis")?;
    Ok(nis)
}

fn transition_matrix(dimension: usize, dt_s: f64) -> Vec<Vec<f64>> {
    let state_dimension = 2 * dimension;
    let mut transition = vec![vec![0.0; state_dimension]; state_dimension];
    for axis in 0..dimension {
        transition[axis][axis] = 1.0;
        transition[axis][dimension + axis] = dt_s;
        transition[dimension + axis][dimension + axis] = 1.0;
    }
    transition
}

fn process_noise_matrix(
    dimension: usize,
    dt_s: f64,
    acceleration_variance_spectral_density_m2_s3: f64,
) -> Vec<Vec<f64>> {
    let state_dimension = 2 * dimension;
    let mut process_noise = vec![vec![0.0; state_dimension]; state_dimension];
    let q00 = acceleration_variance_spectral_density_m2_s3 * dt_s * dt_s * dt_s / 3.0;
    let q01 = acceleration_variance_spectral_density_m2_s3 * dt_s * dt_s / 2.0;
    let q11 = acceleration_variance_spectral_density_m2_s3 * dt_s;
    for axis in 0..dimension {
        process_noise[axis][axis] = q00;
        process_noise[axis][dimension + axis] = q01;
        process_noise[dimension + axis][axis] = q01;
        process_noise[dimension + axis][dimension + axis] = q11;
    }
    process_noise
}

fn state_vector(position_m: &[f64], velocity_m_s: &[f64]) -> Vec<f64> {
    let mut state = Vec::with_capacity(position_m.len() + velocity_m_s.len());
    state.extend(position_m);
    state.extend(velocity_m_s);
    state
}

fn matrix3_to_rows(matrix: [[f64; 3]; 3]) -> Vec<Vec<f64>> {
    matrix.iter().map(|row| row.to_vec()).collect()
}

fn matrix3_from_rows(rows: &[Vec<f64>]) -> [[f64; 3]; 3] {
    [
        [rows[0][0], rows[0][1], rows[0][2]],
        [rows[1][0], rows[1][1], rows[1][2]],
        [rows[2][0], rows[2][1], rows[2][2]],
    ]
}

fn validate_time(value: f64, field: &'static str) -> Result<(), TrackError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn validate_positive(value: f64, field: &'static str) -> Result<(), TrackError> {
    validate_time(value, field)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be positive"))
    }
}

fn validate_nonnegative(value: f64, field: &'static str) -> Result<(), TrackError> {
    validate_time(value, field)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be non-negative"))
    }
}

fn validate_dimension(dimension: usize, field: &'static str) -> Result<(), TrackError> {
    if dimension > 0 {
        Ok(())
    } else {
        Err(invalid_input(field, "dimension must be positive"))
    }
}

fn validate_vector_len(
    values: &[f64],
    expected: usize,
    field: &'static str,
) -> Result<(), TrackError> {
    if values.len() != expected {
        return Err(TrackError::DimensionMismatch {
            field,
            expected,
            actual: values.len(),
        });
    }
    validate::finite_slice(values, field).map_err(map_field_error)
}

fn validate_square_matrix(
    matrix: &[Vec<f64>],
    dimension: usize,
    field: &'static str,
) -> Result<(), TrackError> {
    if matrix.len() != dimension {
        return Err(TrackError::DimensionMismatch {
            field,
            expected: dimension,
            actual: matrix.len(),
        });
    }
    for row in matrix {
        if row.len() != dimension {
            return Err(TrackError::DimensionMismatch {
                field,
                expected: dimension,
                actual: row.len(),
            });
        }
        validate::finite_slice(row, field).map_err(map_field_error)?;
    }
    Ok(())
}

fn validate_covariance_matrix(
    matrix: &[Vec<f64>],
    dimension: usize,
    field: &'static str,
) -> Result<(), TrackError> {
    validate_square_matrix(matrix, dimension, field)?;
    let rows = matrix.iter().map(Vec::as_slice).collect::<Vec<_>>();
    validate::validate_covariance_psd_rows(&rows, field)
        .map_err(|_| TrackError::NonPositiveSemidefinite { field })
}

fn validate_spd_matrix(
    matrix: &[Vec<f64>],
    dimension: usize,
    field: &'static str,
) -> Result<(), TrackError> {
    validate_covariance_matrix(matrix, dimension, field)?;
    let flat = flatten(matrix)?;
    let mut scratch = FlatCholeskySolveScratch::default();
    let rhs = vec![0.0; dimension];
    solve_flat_normal_square_root_into(&flat, &rhs, &mut scratch)
        .map(|_| ())
        .ok_or(TrackError::NonPositiveDefinite { field })
}

fn map_field_error(error: FieldError) -> TrackError {
    invalid_input(error.field(), error.reason())
}

fn invalid_input(field: &'static str, reason: &'static str) -> TrackError {
    TrackError::InvalidInput { field, reason }
}

fn transpose(matrix: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TrackError> {
    if matrix.is_empty() {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    let rows = matrix.len();
    let cols = matrix[0].len();
    if cols == 0 {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    for row in matrix {
        validate_vector_len(row, cols, "matrix")?;
    }
    let mut out = vec![vec![0.0; rows]; cols];
    for row in 0..rows {
        for col in 0..cols {
            out[col][row] = matrix[row][col];
        }
    }
    Ok(out)
}

fn matvec(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>, TrackError> {
    if matrix.is_empty() {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    let cols = vector.len();
    if cols == 0 {
        return Err(invalid_input("vector", "must not be empty"));
    }
    for row in matrix {
        validate_vector_len(row, cols, "matrix")?;
    }
    validate_vector_len(vector, cols, "vector")?;
    let mut out = vec![0.0; matrix.len()];
    for row in 0..matrix.len() {
        for (col, value) in vector.iter().enumerate() {
            out[row] += matrix[row][col] * value;
        }
    }
    validate_vector_len(&out, matrix.len(), "matrix_vector_product")?;
    Ok(out)
}

fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TrackError> {
    if a.is_empty() || b.is_empty() {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    let inner = a[0].len();
    if inner == 0 {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    for row in a {
        validate_vector_len(row, inner, "matrix_a")?;
    }
    if b.len() != inner {
        return Err(TrackError::DimensionMismatch {
            field: "matrix_b",
            expected: inner,
            actual: b.len(),
        });
    }
    let cols = b[0].len();
    if cols == 0 {
        return Err(invalid_input("matrix_b", "must not be empty"));
    }
    for row in b {
        validate_vector_len(row, cols, "matrix_b")?;
    }
    let mut out = vec![vec![0.0; cols]; a.len()];
    for row in 0..a.len() {
        for col in 0..cols {
            for k in 0..inner {
                out[row][col] += a[row][k] * b[k][col];
            }
        }
    }
    for row in &out {
        validate_vector_len(row, cols, "matrix_product")?;
    }
    Ok(out)
}

fn matrix_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TrackError> {
    validate_same_matrix_size(a, b, "matrix_add")?;
    let mut out = vec![vec![0.0; a[0].len()]; a.len()];
    for row in 0..a.len() {
        for col in 0..a[0].len() {
            out[row][col] = a[row][col] + b[row][col];
        }
    }
    Ok(out)
}

fn matrix_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, TrackError> {
    validate_same_matrix_size(a, b, "matrix_sub")?;
    let mut out = vec![vec![0.0; a[0].len()]; a.len()];
    for row in 0..a.len() {
        for col in 0..a[0].len() {
            out[row][col] = a[row][col] - b[row][col];
        }
    }
    Ok(out)
}

fn vector_sub(a: &[f64], b: &[f64], field: &'static str) -> Result<Vec<f64>, TrackError> {
    if a.len() != b.len() {
        return Err(TrackError::DimensionMismatch {
            field,
            expected: a.len(),
            actual: b.len(),
        });
    }
    validate_vector_len(a, a.len(), field)?;
    validate_vector_len(b, b.len(), field)?;
    let out = a
        .iter()
        .zip(b)
        .map(|(lhs, rhs)| lhs - rhs)
        .collect::<Vec<_>>();
    validate_vector_len(&out, a.len(), field)?;
    Ok(out)
}

fn validate_same_matrix_size(
    a: &[Vec<f64>],
    b: &[Vec<f64>],
    field: &'static str,
) -> Result<(), TrackError> {
    if a.is_empty() || b.is_empty() {
        return Err(invalid_input(field, "must not be empty"));
    }
    if a.len() != b.len() {
        return Err(TrackError::DimensionMismatch {
            field,
            expected: a.len(),
            actual: b.len(),
        });
    }
    let cols = a[0].len();
    if cols == 0 {
        return Err(invalid_input(field, "must not be empty"));
    }
    for row in a {
        validate_vector_len(row, cols, field)?;
    }
    for row in b {
        validate_vector_len(row, cols, field)?;
    }
    Ok(())
}

fn flatten(matrix: &[Vec<f64>]) -> Result<Vec<f64>, TrackError> {
    if matrix.is_empty() {
        return Err(invalid_input("matrix", "must not be empty"));
    }
    let cols = matrix[0].len();
    for row in matrix {
        validate_vector_len(row, cols, "matrix")?;
    }
    let mut out = Vec::with_capacity(matrix.len() * cols);
    for row in matrix {
        out.extend(row);
    }
    Ok(out)
}

fn copy_lower_to_upper(matrix: &mut [Vec<f64>]) {
    let dimension = matrix.len();
    for row in 0..dimension {
        let (head, tail) = matrix.split_at_mut(row + 1);
        let row_values = &mut head[row];
        for (offset, lower_row) in tail.iter().enumerate() {
            let col = row + 1 + offset;
            row_values[col] = lower_row[row];
        }
    }
}

#[cfg(test)]
mod tests {
    //! Analytic oracles for no-IMU track filtering and smoothing.
    //!
    //! The scalar parity checks use the Bar-Shalom constant-velocity Kalman
    //! fixed point already exposed by the scalar estimation primitives. The
    //! covariance-weighting and RTS checks use closed-form Kalman and
    //! Rauch-Tung-Striebel equations evaluated in the test, not serialized
    //! fixtures or version-stamped outputs.

    use super::*;
    use crate::astro::math::linear::invert_symmetric_pd;
    use crate::estimation::kalman_cv_steady_state_gains;

    const TOL: f64 = 1.0e-12;

    #[test]
    fn scalar_one_step_matches_direct_cv_recurrence_bits() {
        let dt_s: f64 = 0.75;
        let q: f64 = 0.125;
        let measurement_variance: f64 = 0.8;
        let p00: f64 = 3.0;
        let p01: f64 = 0.25;
        let p11: f64 = 2.0;
        let x0: f64 = 4.0;
        let v0: f64 = -0.5;
        let observation_delta: f64 = 1.3;
        let mut filter = TrackFilter::from_position_velocity(
            TrackCoordinateFrame::CallerDefinedCartesian,
            0.0,
            vec![x0],
            vec![v0],
            vec![vec![p00, p01], vec![p01, p11]],
            q,
        )
        .unwrap();

        let q00 = q * dt_s * dt_s * dt_s / 3.0;
        let q01 = q * dt_s * dt_s / 2.0;
        let q11 = q * dt_s;
        let predicted_position = x0 + dt_s * v0;
        let p00_pred = p00 + 2.0 * dt_s * p01 + dt_s * dt_s * p11 + q00;
        let p01_pred = p01 + dt_s * p11 + q01;
        let p11_pred = p11 + q11;
        let s = p00_pred + measurement_variance;
        let k0 = p00_pred / s;
        let k1 = p01_pred / s;

        let observation = predicted_position + observation_delta;
        let innovation = observation - predicted_position;

        filter.predict(dt_s).unwrap();
        let update = filter
            .update_position(&[observation], &[vec![measurement_variance]])
            .unwrap();

        assert_eq!(update.kalman_gain[0][0].to_bits(), k0.to_bits());
        assert_eq!(update.kalman_gain[1][0].to_bits(), k1.to_bits());
        assert_eq!(
            update.updated.position_m[0].to_bits(),
            (predicted_position + k0 * innovation).to_bits()
        );
        assert_eq!(
            update.updated.velocity_m_s[0].to_bits(),
            (v0 + k1 * innovation).to_bits()
        );
        assert_eq!(
            update.updated.covariance[0][0].to_bits(),
            (p00_pred - k0 * p00_pred).to_bits()
        );
        assert_eq!(
            update.updated.covariance[0][1].to_bits(),
            (p01_pred - k1 * p00_pred).to_bits()
        );
        assert_eq!(
            update.updated.covariance[1][0].to_bits(),
            (p01_pred - k1 * p00_pred).to_bits()
        );
        assert_eq!(
            update.updated.covariance[1][1].to_bits(),
            (p11_pred - k1 * p01_pred).to_bits()
        );
    }

    #[test]
    fn scalar_cv_gains_match_estimation_primitives() {
        let tracking_index: f64 = 4.0;
        let dt_s: f64 = 1.0;
        let measurement_variance: f64 = 1.0;
        let q = tracking_index * measurement_variance / dt_s.powi(3);
        let expected =
            kalman_cv_steady_state_gains(tracking_index, dt_s, measurement_variance).unwrap();
        let mut filter = TrackFilter::from_position_velocity(
            TrackCoordinateFrame::CallerDefinedCartesian,
            0.0,
            vec![0.0],
            vec![0.0],
            vec![
                vec![measurement_variance, 0.0],
                vec![0.0, measurement_variance],
            ],
            q,
        )
        .unwrap();
        let observation_covariance = vec![vec![measurement_variance]];
        let mut last_gain = Vec::new();
        for _ in 0..5_000 {
            filter.predict(dt_s).unwrap();
            let update = filter
                .update_position(&[0.0], &observation_covariance)
                .unwrap();
            last_gain = update.kalman_gain;
        }
        filter.predict(dt_s).unwrap();
        let update = filter
            .update_position(&[0.0], &observation_covariance)
            .unwrap();
        assert!(
            (last_gain[0][0] - update.kalman_gain[0][0]).abs() <= TOL,
            "position gain did not settle"
        );
        assert!(
            (last_gain[1][0] - update.kalman_gain[1][0]).abs() <= TOL,
            "velocity gain did not settle"
        );
        assert!(
            (update.kalman_gain[0][0] - expected.position_gain).abs() <= TOL,
            "position gain mismatch: got {}, expected {}",
            update.kalman_gain[0][0],
            expected.position_gain
        );
        assert!(
            (update.kalman_gain[1][0] - expected.rate_gain).abs() <= TOL,
            "velocity gain mismatch: got {}, expected {}",
            update.kalman_gain[1][0],
            expected.rate_gain
        );
    }

    #[test]
    fn covariance_weighting_limits_wide_outlier_motion() {
        let mut filter = TrackFilter::from_position_velocity(
            TrackCoordinateFrame::CallerDefinedCartesian,
            0.0,
            vec![10.0],
            vec![1.0],
            vec![vec![4.0, 0.0], vec![0.0, 1.0]],
            0.0,
        )
        .unwrap();
        filter.predict(1.0).unwrap();
        let predicted_position = filter.state().position_m[0];
        let predicted_covariance = filter.state().covariance.clone();
        let wide_variance = 1.0e6;
        let outlier = predicted_position + 500.0;
        let update = filter
            .update_position(&[outlier], &[vec![wide_variance]])
            .unwrap();

        let s = predicted_covariance[0][0] + wide_variance;
        let expected_position_gain = predicted_covariance[0][0] / s;
        let expected_velocity_gain = predicted_covariance[1][0] / s;
        let innovation = outlier - predicted_position;
        assert_eq!(update.innovation.innovation, vec![innovation]);
        assert!(
            (update.kalman_gain[0][0] - expected_position_gain).abs() <= TOL,
            "position gain mismatch"
        );
        assert!(
            (update.kalman_gain[1][0] - expected_velocity_gain).abs() <= TOL,
            "velocity gain mismatch"
        );
        assert!(
            (update.updated.position_m[0]
                - (predicted_position + expected_position_gain * innovation))
                .abs()
                <= TOL,
            "position update mismatch"
        );
        assert!(
            (update.updated.position_m[0] - predicted_position).abs() < 0.01,
            "wide covariance outlier moved the track too far"
        );
    }

    #[test]
    fn tight_covariance_observation_pulls_track_more_than_wide_one() {
        let base = TrackFilter::from_position_velocity(
            TrackCoordinateFrame::CallerDefinedCartesian,
            0.0,
            vec![0.0],
            vec![0.0],
            vec![vec![9.0, 0.0], vec![0.0, 1.0]],
            0.0,
        )
        .unwrap();
        let mut wide = base.clone();
        let mut tight = base;
        let wide_update = wide.update_position(&[10.0], &[vec![10_000.0]]).unwrap();
        let tight_update = tight.update_position(&[10.0], &[vec![1.0]]).unwrap();

        assert!(wide_update.kalman_gain[0][0] < 0.001);
        assert!(tight_update.kalman_gain[0][0] > 0.8);
        assert!(wide_update.updated.position_m[0] < 0.01);
        assert!(tight_update.updated.position_m[0] > 8.0);
    }

    #[test]
    fn rts_smoother_matches_closed_form_two_epoch_recursion() {
        let mut filter = TrackFilter::from_position_velocity(
            TrackCoordinateFrame::CallerDefinedCartesian,
            0.0,
            vec![0.0],
            vec![1.0],
            vec![vec![2.0, 0.2], vec![0.2, 1.0]],
            0.5,
        )
        .unwrap();
        let mut history = TrackRtsHistoryBuilder::from_filter(&filter).unwrap();
        filter.predict_recorded(1.0, &mut history).unwrap();
        filter
            .update_position_recorded(&[1.2], &[vec![0.25]], &mut history)
            .unwrap();
        let history = history.finish().unwrap();
        let smoothed = rts_smooth(&history).unwrap();

        let first = &history.epochs[0];
        let second = &history.epochs[1];
        let transition = second.transition_from_previous.as_ref().unwrap();
        let expected_gain = closed_form_rts_gain(
            &first.updated.covariance,
            transition,
            &second.predicted.covariance,
        );
        let expected_delta = vector_sub(
            &second.updated.state_vector(),
            &second.predicted.state_vector(),
            "delta",
        )
        .unwrap();
        let expected_correction = matvec(&expected_gain, &expected_delta).unwrap();
        let expected_state = first
            .updated
            .state_vector()
            .iter()
            .zip(expected_correction)
            .map(|(value, correction)| value + correction)
            .collect::<Vec<_>>();
        let expected_covariance = smoothed_covariance(
            &first.updated.covariance,
            &expected_gain,
            &second.predicted.covariance,
            &second.updated.covariance,
        )
        .unwrap();

        let got_first = &smoothed.epochs[0];
        assert_close_vec(&got_first.state.state_vector(), &expected_state, TOL);
        assert_close_matrix(
            &got_first.rts_gain_to_next.clone().unwrap(),
            &expected_gain,
            TOL,
        );
        assert_close_matrix(&got_first.state.covariance, &expected_covariance, TOL);
        assert_eq!(smoothed.epochs[1].state, second.updated);
    }

    #[test]
    fn smoothing_covariance_does_not_exceed_filtering_covariance_diagonal() {
        let mut filter = TrackFilter::from_position_velocity(
            TrackCoordinateFrame::CallerDefinedCartesian,
            0.0,
            vec![0.0],
            vec![0.0],
            vec![vec![10.0, 0.0], vec![0.0, 10.0]],
            0.1,
        )
        .unwrap();
        let mut history = TrackRtsHistoryBuilder::from_filter(&filter).unwrap();
        for (idx, observation) in [0.1, 0.9, 2.2, 3.1].iter().enumerate() {
            filter.predict_recorded(1.0, &mut history).unwrap();
            if idx == 2 {
                filter.record_prediction_only(&mut history).unwrap();
            } else {
                filter
                    .update_position_recorded(&[*observation], &[vec![0.5]], &mut history)
                    .unwrap();
            }
        }
        let history = history.finish().unwrap();
        let smoothed = rts_smooth(&history).unwrap();
        assert_eq!(smoothed.epochs.len(), history.epochs.len());
        for (smoothed_epoch, filtered_epoch) in smoothed.epochs.iter().zip(&history.epochs) {
            for idx in 0..smoothed_epoch.state.state_dimension() {
                assert!(
                    smoothed_epoch.state.covariance[idx][idx]
                        <= filtered_epoch.updated.covariance[idx][idx] + 1.0e-10,
                    "smoothed covariance diagonal exceeded filtered covariance"
                );
            }
        }
    }

    #[test]
    fn gated_rejection_records_prediction_only_epoch() {
        let mut filter = TrackFilter::from_position_velocity(
            TrackCoordinateFrame::CallerDefinedCartesian,
            0.0,
            vec![0.0],
            vec![1.0],
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            0.1,
        )
        .unwrap();
        let mut history = TrackRtsHistoryBuilder::from_filter(&filter).unwrap();
        filter.predict_recorded(1.0, &mut history).unwrap();
        let predicted = filter.state().clone();
        let gated = filter
            .update_position_gated_recorded(&[100.0], &[vec![0.01]], 0.95, &mut history)
            .unwrap();
        assert!(!gated.gate.in_gate);
        assert!(gated.update.is_none());
        assert_eq!(*filter.state(), predicted);
        let history = history.finish().unwrap();
        assert_eq!(history.epochs.len(), 2);
        assert_eq!(history.epochs[1].predicted, predicted);
        assert_eq!(history.epochs[1].updated, predicted);
    }

    #[test]
    fn position_covariance_type_selects_frame_block() {
        let covariance = PositionCovariance {
            ecef_m2: [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
            enu_m2: [[4.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 6.0]],
        };
        let mut ecef = TrackFilter::from_position3(
            TrackCoordinateFrame::Ecef,
            0.0,
            [0.0, 0.0, 0.0],
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            1.0,
            0.0,
        )
        .unwrap();
        let update = ecef
            .update_position_covariance([1.0, 0.0, 0.0], &covariance)
            .unwrap();
        assert!((update.kalman_gain[0][0] - 10.0 / 11.0).abs() <= TOL);

        let mut enu = TrackFilter::from_position3(
            TrackCoordinateFrame::Enu,
            0.0,
            [0.0, 0.0, 0.0],
            [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            1.0,
            0.0,
        )
        .unwrap();
        let update = enu
            .update_position_covariance([1.0, 0.0, 0.0], &covariance)
            .unwrap();
        assert!((update.kalman_gain[0][0] - 10.0 / 14.0).abs() <= TOL);
    }

    fn closed_form_rts_gain(
        filtered_covariance: &[Vec<f64>],
        transition: &[Vec<f64>],
        predicted_covariance_next: &[Vec<f64>],
    ) -> Vec<Vec<f64>> {
        let transition_t = transpose(transition).unwrap();
        let cross = matmul(filtered_covariance, &transition_t).unwrap();
        let inverse = invert_symmetric_pd(predicted_covariance_next).unwrap();
        matmul(&cross, &inverse).unwrap()
    }

    fn assert_close_vec(got: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(got.len(), expected.len());
        for (lhs, rhs) in got.iter().zip(expected) {
            assert!(
                (lhs - rhs).abs() <= tolerance,
                "vector mismatch: got {lhs}, expected {rhs}"
            );
        }
    }

    fn assert_close_matrix(got: &[Vec<f64>], expected: &[Vec<f64>], tolerance: f64) {
        assert_eq!(got.len(), expected.len());
        for (row_lhs, row_rhs) in got.iter().zip(expected) {
            assert_close_vec(row_lhs, row_rhs, tolerance);
        }
    }
}
