//! Force-parameter estimation in the batch orbit fit: the method, weights,
//! time argument and validation are described in the parent module's
//! documentation, section "Force-parameter estimation".

use std::fmt;
use std::sync::Arc;

use nalgebra::{DMatrix, Vector3};

use super::{
    check_provider_validity, collect_provider_sp3_observations, dynamical_seconds_from_instant,
    fit_frame_error, OrbitFitError, OrbitObservation, OrbitResidualStats, Ut1ProviderRole,
    MIN_SEED_SAMPLES,
};
use crate::astro::covariance::rtn_to_eci_rotation;
use crate::astro::forces::{
    CompositeForceModel, DragParameters, ForceModel, ForceParameter, SpaceWeatherSource,
};
use crate::astro::frames::orientation::EarthOrientationProvider;
use crate::astro::frames::transforms::Ut1Gate;
use crate::astro::iod;
use crate::astro::math::portable;
use crate::astro::propagator::variational::propagate_trajectory;
use crate::astro::propagator::{
    propagate_variational, ForceModelKind, IntegratorKind, IntegratorOptions, PropagationContext,
    StatePropagator, VariationalEpoch, VariationalEquations, VariationalState,
};
use crate::astro::state::CartesianState;
use crate::astro::time::model::Instant;
use crate::astro::time::{Validated, ValidityMode};
use crate::constants::{M_PER_KM, SECONDS_PER_DAY};
use crate::sp3::Sp3;
use crate::GnssSatelliteId;

/// Names of the six initial-state parameters, the first rows and columns of
/// [`OrbitParameterCovariance::Estimated`]: inertial position in km and
/// velocity in km/s.
pub const STATE_PARAMETER_NAMES: [&str; 6] =
    ["x_km", "y_km", "z_km", "vx_km_s", "vy_km_s", "vz_km_s"];

const STATE_PARAMETER_COUNT: usize = STATE_PARAMETER_NAMES.len();

/// Most halvings of one Gauss-Newton step that raises the weighted sum of
/// squared residuals; the last halved step is taken.
const MAX_STEP_HALVINGS: usize = 8;

/// A step is checked against the weighted sum of squared residuals only when
/// the decrease the linearization predicts for it, `dx^T N dx`, exceeds both
/// this absolute amount (a step of one formal standard deviation) and
/// [`LINE_SEARCH_RELATIVE_DECREASE`] of the sum. Smaller steps lie well inside
/// the region where the linearization holds.
const LINE_SEARCH_MIN_DECREASE: f64 = 1.0;

/// The relative part of the line-search threshold. The sum carries the
/// integration error of the trajectory: on a one-day GNSS arc it changes by
/// about 1e-8 of itself between evaluations at estimates that differ by less
/// than the noise, so a predicted decrease below 1e-6 of the sum is not
/// resolved by comparing sums.
const LINE_SEARCH_RELATIVE_DECREASE: f64 = 1.0e-6;

/// Components of the weakest parameter combination reported with
/// singular or ill-conditioned normal equations: those at least this large in
/// the unit singular vector of the column-scaled design.
const WEAKEST_COMPONENT_REPORT: f64 = 0.1;

/// Options of a force-parameter orbit fit.
#[derive(Clone)]
#[non_exhaustive]
pub struct OrbitParameterFitOptions {
    /// Gravity and the other built-in force components.
    pub force_model: ForceModelKind,
    /// Optional atmospheric drag added to [`Self::force_model`].
    pub drag: Option<DragParameters>,
    /// Optional per-epoch space-weather source for the drag.
    pub space_weather: Option<SpaceWeatherSource>,
    /// Force models added after [`Self::force_model`] and the drag, in order,
    /// for example an
    /// [`EmpiricalRtnAcceleration`](crate::astro::forces::EmpiricalRtnAcceleration).
    pub additional_forces: Vec<Arc<dyn ForceModel>>,
    /// Names of the force-model parameters estimated with the initial state,
    /// in the order of the estimate and its covariance. Each must be listed by
    /// exactly one force model; the others keep the values their models hold.
    pub estimated_parameters: Vec<String>,
    /// Integrator of the trajectory and its variational equations.
    pub integrator: IntegratorKind,
    /// Step and tolerance controls of the integrator.
    pub integrator_options: IntegratorOptions,
    /// Propagation context shared with the force models.
    pub propagation_context: PropagationContext,
    /// A-priori standard deviation of each observed position component, m.
    /// It sets the weights; the reported covariance is scaled by the
    /// a-posteriori variance factor and does not depend on it.
    pub observation_sigma_m: f64,
    /// Most Gauss-Newton iterations.
    pub max_iterations: usize,
    /// The iteration has converged when the step is shorter than this many
    /// formal standard deviations, `sqrt(dx^T N dx)` with `N` the normal
    /// matrix of the a-priori weights (see the [`crate::orbit_determination`]
    /// documentation). Being formal, the length scales with
    /// [`Self::observation_sigma_m`].
    pub convergence_threshold: f64,
    /// Largest accepted condition number of the normal equations after their
    /// columns are scaled to unit diagonal; `f64::INFINITY` accepts every
    /// numerically regular system.
    pub max_condition_number: f64,
    /// Residual count below which the residual statistics are marked low-n.
    pub min_ledger_samples: usize,
}

impl Default for OrbitParameterFitOptions {
    fn default() -> Self {
        Self {
            force_model: ForceModelKind::earth_phase_a(None),
            drag: None,
            space_weather: None,
            additional_forces: Vec::new(),
            estimated_parameters: Vec::new(),
            integrator: IntegratorKind::Dp54,
            integrator_options: IntegratorOptions::default(),
            propagation_context: PropagationContext::default(),
            observation_sigma_m: 1.0,
            max_iterations: 25,
            convergence_threshold: 1.0e-3,
            max_condition_number: 1.0e12,
            min_ledger_samples: 3,
        }
    }
}

impl fmt::Debug for OrbitParameterFitOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let additional: Vec<Vec<String>> = self
            .additional_forces
            .iter()
            .map(|model| model.parameters().into_iter().map(|p| p.name).collect())
            .collect();
        f.debug_struct("OrbitParameterFitOptions")
            .field("force_model", &self.force_model)
            .field("drag", &self.drag)
            .field("space_weather", &self.space_weather.is_some())
            .field("additional_force_parameters", &additional)
            .field("estimated_parameters", &self.estimated_parameters)
            .field("integrator", &self.integrator)
            .field("integrator_options", &self.integrator_options)
            .field("observation_sigma_m", &self.observation_sigma_m)
            .field("max_iterations", &self.max_iterations)
            .field("convergence_threshold", &self.convergence_threshold)
            .field("max_condition_number", &self.max_condition_number)
            .field("min_ledger_samples", &self.min_ledger_samples)
            .finish_non_exhaustive()
    }
}

/// One observed position in the inertial frame of the propagation (GCRF).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InertialPositionSample {
    /// Observation epoch, in any time scale.
    pub epoch: Instant,
    /// Observed position, m.
    pub position_gcrf_m: [f64; 3],
}

/// One estimated force-model parameter.
#[derive(Debug, Clone, PartialEq)]
pub struct EstimatedForceParameter {
    /// Parameter name, as listed by its force model.
    pub name: String,
    /// The value the force model held, where the estimation started.
    pub initial_value: f64,
    /// The estimate, in the model's unit for the parameter.
    pub value: f64,
}

/// Covariance of a force-parameter orbit fit.
#[derive(Debug, Clone, PartialEq)]
pub enum OrbitParameterCovariance {
    /// The formal covariance scaled by the a-posteriori variance factor.
    Estimated {
        /// `s0^2 (H^T W H)^-1`, rows and columns in the order of
        /// [`OrbitParameterFit::parameter_names`]: km, km/s and the force
        /// parameters' own units.
        matrix: DMatrix<f64>,
        /// Correlation coefficients of `matrix`, unit diagonal.
        correlation: DMatrix<f64>,
        /// The a-posteriori variance factor `s0^2`: the weighted sum of
        /// squared post-fit residuals over the degrees of freedom.
        variance_factor: f64,
        /// Residual components minus estimated parameters.
        degrees_of_freedom: usize,
    },
    /// The arc has no residual degrees of freedom, so no variance factor can
    /// be formed.
    Unbounded,
}

/// A post-fit residual, observed minus computed, in the radial, along-track
/// and cross-track frame of the fitted trajectory at the observation epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitRtnResidual {
    /// Observation epoch, TDB seconds past J2000.
    pub epoch_tdb_j2000_s: f64,
    /// TT seconds elapsed from the epoch of [`OrbitParameterFit::initial_state`]
    /// to the observation: how far the fitted trajectory was integrated.
    pub elapsed_tt_s: f64,
    /// Radial component, m.
    pub radial_m: f64,
    /// Along-track component, m.
    pub along_m: f64,
    /// Cross-track component, m.
    pub cross_m: f64,
}

/// Result of a force-parameter orbit fit.
#[derive(Debug, Clone, PartialEq)]
pub struct OrbitParameterFit {
    /// Satellite fitted.
    pub satellite: GnssSatelliteId,
    /// Estimated initial state at the first observation, epoch in TDB.
    /// [`OrbitRtnResidual::variational_epoch`] gives the epochs to propagate
    /// it to.
    pub initial_state: CartesianState,
    /// Estimated force parameters, in the order of
    /// [`OrbitParameterFitOptions::estimated_parameters`].
    pub force_parameters: Vec<EstimatedForceParameter>,
    /// [`STATE_PARAMETER_NAMES`] followed by the estimated force parameters:
    /// the order of the covariance.
    pub parameter_names: Vec<String>,
    /// Covariance of the estimate.
    pub covariance: OrbitParameterCovariance,
    /// Post-fit residuals, one per observation.
    pub residuals: Vec<OrbitRtnResidual>,
    /// RMS of [`Self::residuals`] per axis and in 3D.
    pub residual_stats: OrbitResidualStats,
    /// Three-dimensional RMS residual of the initial guess, m.
    pub seed_rms_3d_m: f64,
    /// Gauss-Newton iterations taken.
    pub iterations: usize,
    /// Condition number of the normal equations at the estimate, after their
    /// columns are scaled to unit diagonal.
    pub condition_number: f64,
}

impl OrbitRtnResidual {
    /// The observation epoch as a [`propagate_variational`] epoch from
    /// [`OrbitParameterFit::initial_state`]: propagating the fitted state and
    /// parameters to it reproduces the fitted trajectory.
    pub fn variational_epoch(&self) -> VariationalEpoch {
        VariationalEpoch {
            tdb_j2000_s: self.epoch_tdb_j2000_s,
            elapsed_tt_s: self.elapsed_tt_s,
        }
    }
}

impl OrbitParameterFit {
    /// Standard deviation of a parameter from the scaled covariance, or `None`
    /// for an unknown name or an unbounded covariance.
    pub fn sigma(&self, name: &str) -> Option<f64> {
        let index = self.parameter_index(name)?;
        match &self.covariance {
            OrbitParameterCovariance::Estimated { matrix, .. } => {
                Some(matrix[(index, index)].sqrt())
            }
            OrbitParameterCovariance::Unbounded => None,
        }
    }

    /// Correlation coefficient of two parameters, or `None` for an unknown
    /// name or an unbounded covariance.
    pub fn correlation(&self, first: &str, second: &str) -> Option<f64> {
        let row = self.parameter_index(first)?;
        let column = self.parameter_index(second)?;
        match &self.covariance {
            OrbitParameterCovariance::Estimated { correlation, .. } => {
                Some(correlation[(row, column)])
            }
            OrbitParameterCovariance::Unbounded => None,
        }
    }

    /// The estimate of a force parameter by name.
    pub fn force_parameter(&self, name: &str) -> Option<&EstimatedForceParameter> {
        self.force_parameters
            .iter()
            .find(|parameter| parameter.name == name)
    }

    fn parameter_index(&self, name: &str) -> Option<usize> {
        self.parameter_names.iter().position(|entry| entry == name)
    }
}

/// Error returned by a force-parameter orbit fit.
#[derive(Debug, Clone, thiserror::Error)]
pub enum OrbitParameterFitError {
    /// Input, frame or propagation failure shared with the state-only fit.
    #[error(transparent)]
    Fit(#[from] OrbitFitError),
    /// A requested parameter is not estimable as named.
    #[error("estimated parameter {name:?} {reason}")]
    InvalidParameter {
        /// Requested name.
        name: String,
        /// Why it was refused.
        reason: &'static str,
    },
    /// A parameter's column of the design matrix is zero: the observations do
    /// not depend on it.
    #[error("satellite {satellite}: the observations do not depend on parameter {name:?}")]
    UnobservableParameter {
        /// Satellite fitted.
        satellite: GnssSatelliteId,
        /// Parameter name.
        name: String,
    },
    /// The normal equations are numerically singular: some combination of
    /// parameters has no effect on the observations within rounding.
    #[error(
        "satellite {satellite}: singular normal equations, condition number {condition_number:e}; weakest combination {weakest:?}"
    )]
    SingularNormalEquations {
        /// Satellite fitted.
        satellite: GnssSatelliteId,
        /// Condition number of the column-scaled normal equations,
        /// `f64::INFINITY` when a singular value is zero.
        condition_number: f64,
        /// The parameters of the weakest combination with their components
        /// in it, largest first.
        weakest: Vec<(String, f64)>,
    },
    /// The normal equations are regular but worse conditioned than
    /// [`OrbitParameterFitOptions::max_condition_number`].
    #[error(
        "satellite {satellite}: normal equations condition number {condition_number:e} exceeds {limit:e}; weakest combination {weakest:?}"
    )]
    IllConditionedNormalEquations {
        /// Satellite fitted.
        satellite: GnssSatelliteId,
        /// Condition number of the column-scaled normal equations.
        condition_number: f64,
        /// The accepted limit.
        limit: f64,
        /// The parameters of the weakest combination with their components
        /// in it, largest first.
        weakest: Vec<(String, f64)>,
    },
    /// A quantity of the fit is not finite, though every input is: the
    /// observations, weights or force models overflow binary64.
    #[error("satellite {satellite}: {quantity} not finite")]
    NumericalFailure {
        /// Satellite fitted.
        satellite: GnssSatelliteId,
        /// The quantity that overflowed.
        quantity: &'static str,
    },
    /// The iteration did not converge within
    /// [`OrbitParameterFitOptions::max_iterations`].
    #[error(
        "satellite {satellite}: fit did not converge in {iterations} iterations; the last step was {last_step_sigmas:e} formal standard deviations long"
    )]
    DidNotConverge {
        /// Satellite fitted.
        satellite: GnssSatelliteId,
        /// Iterations taken.
        iterations: usize,
        /// Length of the last step in formal standard deviations,
        /// `sqrt(dx^T N dx)`.
        last_step_sigmas: f64,
    },
}

/// Fit the initial state and selected force parameters of one satellite to
/// the positions of a parsed ECEF SP3 product.
///
/// Positions are rotated to GCRF with the provider's orientation at each
/// epoch; velocity records, where present, seed the initial velocity.
/// `initial_state`, when given, is the initial guess at the first epoch of
/// the satellite's arc (TDB); otherwise the guess comes from the first
/// observations. UT1 outside the table is refused; see
/// [`fit_sp3_ecef_orbit_parameters_with_validity`].
pub fn fit_sp3_ecef_orbit_parameters(
    product: &Sp3,
    satellite: GnssSatelliteId,
    orientation_provider: &dyn EarthOrientationProvider,
    initial_state: Option<CartesianState>,
    options: &OrbitParameterFitOptions,
) -> Result<OrbitParameterFit, OrbitParameterFitError> {
    fit_sp3_ecef_orbit_parameters_with_validity(
        product,
        satellite,
        orientation_provider,
        initial_state,
        options,
        ValidityMode::Strict,
    )
    .map(|validated| validated.value)
}

/// [`fit_sp3_ecef_orbit_parameters`] under an explicit UT1 [`ValidityMode`]:
/// [`ValidityMode::Strict`] refuses an epoch outside the UT1 table;
/// [`ValidityMode::Permissive`] fits with the long-term UT1 and reports the
/// first departure in [`Validated::degraded`].
pub fn fit_sp3_ecef_orbit_parameters_with_validity(
    product: &Sp3,
    satellite: GnssSatelliteId,
    orientation_provider: &dyn EarthOrientationProvider,
    initial_state: Option<CartesianState>,
    options: &OrbitParameterFitOptions,
    mode: ValidityMode,
) -> Result<Validated<OrbitParameterFit>, OrbitParameterFitError> {
    check_provider_validity(mode, Ut1ProviderRole::Orientation, orientation_provider)?;
    let gate = Ut1Gate::new(mode);
    let fit = with_ut1_record(options, &gate, |options| {
        validate_options(options)?;
        let position_samples = product.precise_ephemeris_samples();
        let state_samples = product.precise_ephemeris_state_samples();
        let observations = collect_provider_sp3_observations(
            &position_samples,
            &state_samples,
            satellite,
            orientation_provider,
            &gate,
        )?;
        let observations: Vec<ArcObservation> =
            observations.iter().map(ArcObservation::from).collect();
        fit_arc(satellite, &observations, initial_state, options)
    })?;
    gate.finish(fit)
        .map_err(|error| OrbitParameterFitError::Fit(fit_frame_error(error)))
}

/// Fit the initial state and selected force parameters of one satellite to
/// positions already in GCRF. `satellite` labels the result and its errors.
///
/// The samples need not be sorted; their epochs must differ. `initial_state`,
/// when given, is the initial guess at the earliest epoch (TDB).
pub fn fit_inertial_orbit_parameters(
    samples: &[InertialPositionSample],
    satellite: GnssSatelliteId,
    initial_state: Option<CartesianState>,
    options: &OrbitParameterFitOptions,
) -> Result<OrbitParameterFit, OrbitParameterFitError> {
    fit_inertial_orbit_parameters_with_validity(
        samples,
        satellite,
        initial_state,
        options,
        ValidityMode::Strict,
    )
    .map(|validated| validated.value)
}

/// [`fit_inertial_orbit_parameters`] under an explicit UT1 [`ValidityMode`].
/// The observations do not read UT1; a body-fixed frame provider of the
/// force models may, and its mode must match.
pub fn fit_inertial_orbit_parameters_with_validity(
    samples: &[InertialPositionSample],
    satellite: GnssSatelliteId,
    initial_state: Option<CartesianState>,
    options: &OrbitParameterFitOptions,
    mode: ValidityMode,
) -> Result<Validated<OrbitParameterFit>, OrbitParameterFitError> {
    let gate = Ut1Gate::new(mode);
    let fit = with_ut1_record(options, &gate, |options| {
        validate_options(options)?;
        let observations = collect_inertial_observations(samples, satellite)?;
        fit_arc(satellite, &observations, initial_state, options)
    })?;
    gate.finish(fit)
        .map_err(|error| OrbitParameterFitError::Fit(fit_frame_error(error)))
}

/// Run a fit with its own UT1 departure record on the propagation context,
/// then pass that departure through the fit's policy and back to the caller's
/// context, as the state-only fit does.
fn with_ut1_record<T>(
    options: &OrbitParameterFitOptions,
    gate: &Ut1Gate,
    fit: impl FnOnce(&OrbitParameterFitOptions) -> Result<T, OrbitParameterFitError>,
) -> Result<T, OrbitParameterFitError> {
    if let Some(provider) = options.propagation_context.body_fixed_frame_provider() {
        check_provider_validity(gate.mode(), Ut1ProviderRole::Propagation, provider)?;
    }
    let run_options = OrbitParameterFitOptions {
        propagation_context: options.propagation_context.with_fresh_departure_record(),
        ..options.clone()
    };
    let result = fit(&run_options);
    let departure = run_options.propagation_context.ut1_departure();
    options.propagation_context.merge_departure(departure);
    let value = result?;
    gate.note(departure)
        .map_err(|error| OrbitParameterFitError::Fit(fit_frame_error(error)))?;
    Ok(value)
}

fn validate_options(options: &OrbitParameterFitOptions) -> Result<(), OrbitFitError> {
    let invalid = |field: &'static str, reason: &'static str| {
        Err(OrbitFitError::InvalidOption { field, reason })
    };
    if !(options.observation_sigma_m.is_finite() && options.observation_sigma_m > 0.0) {
        return invalid("observation_sigma_m", "not finite and positive");
    }
    if options.max_iterations == 0 {
        return invalid("max_iterations", "not positive");
    }
    if !(options.convergence_threshold.is_finite() && options.convergence_threshold > 0.0) {
        return invalid("convergence_threshold", "not finite and positive");
    }
    if options.max_condition_number.is_nan() || options.max_condition_number <= 1.0 {
        return invalid("max_condition_number", "not greater than 1");
    }
    if options.min_ledger_samples == 0 {
        return invalid("min_ledger_samples", "not positive");
    }
    Ok(())
}

/// One observation of the arc, in GCRF.
#[derive(Debug, Clone, Copy)]
struct ArcObservation {
    epoch_tdb_s: f64,
    epoch_tt_s: f64,
    position_km: Vector3<f64>,
    velocity_km_s: Option<Vector3<f64>>,
}

impl ArcObservation {
    /// This epoch as a propagation epoch from `from`.
    fn propagation_epoch(&self, from: &ArcObservation) -> VariationalEpoch {
        VariationalEpoch {
            tdb_j2000_s: self.epoch_tdb_s,
            elapsed_tt_s: self.epoch_tt_s - from.epoch_tt_s,
        }
    }
}

impl From<&OrbitObservation> for ArcObservation {
    fn from(observation: &OrbitObservation) -> Self {
        Self {
            epoch_tdb_s: observation.epoch_tdb_j2000_s,
            epoch_tt_s: observation.epoch_tt_j2000_s,
            position_km: Vector3::from(observation.observed_gcrs_km),
            velocity_km_s: observation.observed_gcrs_velocity_km_s.map(Vector3::from),
        }
    }
}

fn collect_inertial_observations(
    samples: &[InertialPositionSample],
    satellite: GnssSatelliteId,
) -> Result<Vec<ArcObservation>, OrbitFitError> {
    let mut observations = Vec::with_capacity(samples.len());
    for sample in samples {
        if !sample.position_gcrf_m.iter().all(|value| value.is_finite()) {
            return Err(OrbitFitError::InvalidObservation {
                satellite,
                reason: "position components must be finite",
            });
        }
        let (epoch_tdb_s, epoch_tt_s) = dynamical_seconds_from_instant(sample.epoch, satellite)?;
        observations.push(ArcObservation {
            epoch_tdb_s,
            epoch_tt_s,
            position_km: Vector3::from(sample.position_gcrf_m) / M_PER_KM,
            velocity_km_s: None,
        });
    }
    observations.sort_by(|a, b| a.epoch_tt_s.total_cmp(&b.epoch_tt_s));
    if observations.len() < MIN_SEED_SAMPLES {
        return Err(OrbitFitError::TooFewSamples {
            satellite,
            got: observations.len(),
            required: MIN_SEED_SAMPLES,
        });
    }
    if observations
        .windows(2)
        .any(|window| window[1].epoch_tt_s <= window[0].epoch_tt_s)
    {
        return Err(OrbitFitError::NonMonotonicEpochs { satellite });
    }
    Ok(observations)
}

/// The fixed parts of one fit: observations, force model and selection.
struct ArcProblem<'a> {
    satellite: GnssSatelliteId,
    observations: &'a [ArcObservation],
    /// The observation epochs as propagation epochs from the first one.
    epochs: Vec<VariationalEpoch>,
    force: Box<dyn ForceModel>,
    /// Values of every listed force parameter; the selected ones are replaced
    /// by the estimate.
    held_values: Vec<f64>,
    /// Indices of the estimated parameters in `held_values`.
    selected: Vec<usize>,
    /// [`STATE_PARAMETER_NAMES`] and the selected names.
    names: Vec<String>,
    options: &'a OrbitParameterFitOptions,
}

/// The residuals and design matrix at one estimate, unweighted: every
/// component carries the same weight. The a-priori sigma sets only the
/// formal length of a step ([`ArcProblem::step_sigmas`]) and the variance
/// factor; the residuals, the design matrix, the step and the covariance do
/// not depend on it.
struct Linearization {
    /// `observed - computed`, km, three components per observation.
    residual_km: Vec<f64>,
    /// `d computed / d parameters`, one row per residual component.
    design: DMatrix<f64>,
    /// Sum of squared residuals, km^2.
    sum_squares_km2: f64,
    /// Propagated states at the observations.
    states: Vec<CartesianState>,
}

/// The least-squares solution of one linearization.
struct NormalSolution {
    /// Gauss-Newton step.
    step: Vec<f64>,
    /// `|H step|`, km: the step's length in formal standard deviations times
    /// the a-priori sigma.
    step_km: f64,
    /// `V diag(1/s^2) V^T`: `(H^T H)^-1` with its rows and columns scaled by
    /// the column norms of `H`.
    scaled_cofactor: DMatrix<f64>,
    /// The column norms of `H`.
    norms: Vec<ColumnNorm>,
    /// Condition number of the column-scaled normal equations.
    condition_number: f64,
}

fn fit_arc(
    satellite: GnssSatelliteId,
    observations: &[ArcObservation],
    initial_seed: Option<CartesianState>,
    options: &OrbitParameterFitOptions,
) -> Result<OrbitParameterFit, OrbitParameterFitError> {
    let problem = ArcProblem::new(satellite, observations, options)?;
    let mut estimate = problem.seed(initial_seed)?;
    estimate.extend(
        problem
            .selected
            .iter()
            .map(|&index| problem.held_values[index]),
    );

    let mut current = problem.linearize(&estimate)?;
    let seed_rms_3d_m = rms_3d_m(&current.residual_km);
    let mut iterations = 0;
    let mut last_step_sigmas = f64::INFINITY;
    while iterations < options.max_iterations {
        let solution = problem.solve(&current)?;
        iterations += 1;
        let full_step_km = solution.step_km;
        let step_sigmas = problem.step_sigmas(full_step_km);
        // The predicted decrease, `dx^T N dx`, is `step_sigmas^2` formal
        // variances and `full_step_km^2` km^2 of the sum.
        let resolvable = step_sigmas * step_sigmas > LINE_SEARCH_MIN_DECREASE
            && full_step_km * full_step_km
                > LINE_SEARCH_RELATIVE_DECREASE * current.sum_squares_km2;
        let mut scale = 1.0;
        let mut halvings = 0;
        let (candidate_estimate, candidate) = loop {
            let candidate_estimate: Vec<f64> = estimate
                .iter()
                .zip(&solution.step)
                .map(|(value, step)| value + scale * step)
                .collect();
            let candidate = problem.linearize(&candidate_estimate)?;
            if !resolvable
                || candidate.sum_squares_km2 < current.sum_squares_km2
                || halvings >= MAX_STEP_HALVINGS
            {
                break (candidate_estimate, candidate);
            }
            scale *= 0.5;
            halvings += 1;
        };
        estimate = candidate_estimate;
        current = candidate;
        last_step_sigmas = step_sigmas;
        if halvings == 0 && step_sigmas <= options.convergence_threshold {
            return problem.finish(estimate, current, iterations, seed_rms_3d_m);
        }
    }
    Err(OrbitParameterFitError::DidNotConverge {
        satellite,
        iterations,
        last_step_sigmas,
    })
}

fn rms_3d_m(residual_km: &[f64]) -> f64 {
    let sum: f64 = residual_km.iter().map(|value| value * value).sum();
    let observations = (residual_km.len() / 3) as f64;
    (sum / observations).sqrt() * M_PER_KM
}

impl<'a> ArcProblem<'a> {
    fn new(
        satellite: GnssSatelliteId,
        observations: &'a [ArcObservation],
        options: &'a OrbitParameterFitOptions,
    ) -> Result<Self, OrbitParameterFitError> {
        let force = build_force(options)
            .map_err(|source| OrbitFitError::Propagation { satellite, source })?;
        let listed = force.parameters();
        let selected = select_parameters(&listed, &options.estimated_parameters)?;
        let names = STATE_PARAMETER_NAMES
            .iter()
            .map(|name| (*name).to_string())
            .chain(selected.iter().map(|&index| listed[index].name.clone()))
            .collect();
        let first = &observations[0];
        Ok(Self {
            satellite,
            observations,
            epochs: observations
                .iter()
                .map(|observation| observation.propagation_epoch(first))
                .collect(),
            force,
            held_values: listed.iter().map(|parameter| parameter.value).collect(),
            selected,
            names,
            options,
        })
    }

    /// `|H dx| / sigma`: the length of a step in formal standard deviations,
    /// `sqrt(dx^T N dx)`, from its length `|H dx|` in km.
    ///
    /// The sigma stays in metres: in km a sigma below about 2.5e-321 m is
    /// zero. `|H dx|` is at most the residual norm, whose square is finite,
    /// so its metres do not overflow. The ratio and its square overflow or
    /// underflow only where the exact value lies beyond every threshold it
    /// is compared with, so each comparison has the outcome of the exact
    /// one, to the rounding of the ratio.
    fn step_sigmas(&self, step_km: f64) -> f64 {
        step_km * M_PER_KM / self.options.observation_sigma_m
    }

    fn fit_error(&self, reason: &'static str) -> OrbitParameterFitError {
        OrbitParameterFitError::Fit(OrbitFitError::InvalidObservation {
            satellite: self.satellite,
            reason,
        })
    }

    fn numerical_failure(&self, quantity: &'static str) -> OrbitParameterFitError {
        OrbitParameterFitError::NumericalFailure {
            satellite: self.satellite,
            quantity,
        }
    }

    fn propagation_error(&self, source: crate::astro::error::PropagationError) -> OrbitFitError {
        OrbitFitError::Propagation {
            satellite: self.satellite,
            source,
        }
    }

    /// The initial state guess, `[r, v]` at the first observation.
    fn seed(
        &self,
        initial_seed: Option<CartesianState>,
    ) -> Result<Vec<f64>, OrbitParameterFitError> {
        let first = &self.observations[0];
        let state = match initial_seed {
            Some(seed) => {
                if seed.epoch_tdb_seconds != first.epoch_tdb_s {
                    return Err(OrbitFitError::InvalidEpoch {
                        satellite: self.satellite,
                        reason: "initial-state seed epoch must match the first sample".to_string(),
                    }
                    .into());
                }
                [seed.position_km, seed.velocity_km_s]
            }
            None => self.seed_from_observations()?,
        };
        let values: Vec<f64> = state[0].iter().chain(state[1].iter()).copied().collect();
        if values.iter().all(|value| value.is_finite()) {
            Ok(values)
        } else {
            Err(self.fit_error("initial-state seed components must be finite"))
        }
    }

    /// A velocity record, else Herrick-Gibbs on the first three positions
    /// propagated back to the first epoch, else the first difference.
    fn seed_from_observations(&self) -> Result<[Vector3<f64>; 2], OrbitParameterFitError> {
        let first = &self.observations[0];
        if let Some(velocity) = first.velocity_km_s {
            return Ok([first.position_km, velocity]);
        }
        if let [o1, o2, o3, ..] = self.observations {
            let day = |observation: &ArcObservation| observation.epoch_tt_s / SECONDS_PER_DAY;
            let r = |observation: &ArcObservation| {
                [
                    observation.position_km.x,
                    observation.position_km.y,
                    observation.position_km.z,
                ]
            };
            if let Ok((v2, _, _, _)) =
                iod::hgibbs(&r(o1), &r(o2), &r(o3), day(o1), day(o2), day(o3))
            {
                let middle = CartesianState::new(o2.epoch_tdb_s, r(o2), v2);
                let equations = VariationalEquations {
                    force: self.force.as_ref(),
                    parameters: &self.held_values,
                    sensitivity_parameters: &[],
                };
                if let Ok(states) = propagate_trajectory(
                    middle,
                    &[o1.propagation_epoch(o2)],
                    &equations,
                    self.options.integrator,
                    &self.options.integrator_options,
                    &self.options.propagation_context,
                ) {
                    return Ok([first.position_km, states[0].velocity_km_s]);
                }
            }
        }
        let second = &self.observations[1];
        let dt = second.epoch_tt_s - first.epoch_tt_s;
        if !(dt.is_finite() && dt > 0.0) {
            return Err(OrbitFitError::NonMonotonicEpochs {
                satellite: self.satellite,
            }
            .into());
        }
        Ok([
            first.position_km,
            (second.position_km - first.position_km) / dt,
        ])
    }

    fn linearize(&self, estimate: &[f64]) -> Result<Linearization, OrbitParameterFitError> {
        if !estimate.iter().all(|value| value.is_finite()) {
            return Err(self.numerical_failure("estimate"));
        }
        let initial = CartesianState::new(
            self.observations[0].epoch_tdb_s,
            [estimate[0], estimate[1], estimate[2]],
            [estimate[3], estimate[4], estimate[5]],
        );
        let mut values = self.held_values.clone();
        for (offset, &index) in self.selected.iter().enumerate() {
            values[index] = estimate[STATE_PARAMETER_COUNT + offset];
        }
        let equations = VariationalEquations {
            force: self.force.as_ref(),
            parameters: &values,
            sensitivity_parameters: &self.selected,
        };
        let propagated = propagate_variational(
            initial,
            &self.epochs,
            &equations,
            self.options.integrator,
            &self.options.integrator_options,
            &self.options.propagation_context,
        )
        .map_err(|source| self.propagation_error(source))?;

        let rows = 3 * self.observations.len();
        let columns = self.names.len();
        let mut residual = Vec::with_capacity(rows);
        let mut design = DMatrix::zeros(rows, columns);
        for (index, (state, observation)) in propagated.iter().zip(self.observations).enumerate() {
            let VariationalState {
                state,
                state_transition,
                sensitivity,
            } = state;
            for axis in 0..3 {
                let row = 3 * index + axis;
                residual.push(observation.position_km[axis] - state.position_km[axis]);
                for column in 0..STATE_PARAMETER_COUNT {
                    design[(row, column)] = state_transition[(axis, column)];
                }
                for column in 0..sensitivity.ncols() {
                    design[(row, STATE_PARAMETER_COUNT + column)] = sensitivity[(axis, column)];
                }
            }
        }
        let sum_squares_km2: f64 = residual.iter().map(|value| value * value).sum();
        if !sum_squares_km2.is_finite() {
            return Err(self.numerical_failure("sum of squared residuals"));
        }
        Ok(Linearization {
            residual_km: residual,
            design,
            sum_squares_km2,
            states: propagated.into_iter().map(|state| state.state).collect(),
        })
    }

    /// Solve the column-scaled weighted design by singular value
    /// decomposition.
    fn solve(
        &self,
        linearization: &Linearization,
    ) -> Result<NormalSolution, OrbitParameterFitError> {
        let design = &linearization.design;
        let (rows, columns) = design.shape();
        let mut norms = Vec::with_capacity(columns);
        for column in 0..columns {
            let norm = column_norm(design.column(column).iter().copied())
                .ok_or_else(|| self.numerical_failure("design matrix"))?;
            if norm.fraction == 0.0 {
                return Err(OrbitParameterFitError::UnobservableParameter {
                    satellite: self.satellite,
                    name: self.names[column].clone(),
                });
            }
            norms.push(norm);
        }
        if rows < columns {
            return Err(OrbitParameterFitError::SingularNormalEquations {
                satellite: self.satellite,
                condition_number: f64::INFINITY,
                weakest: Vec::new(),
            });
        }
        let scaled = DMatrix::from_fn(rows, columns, |row, column| {
            norms[column].divide(design[(row, column)])
        });
        let svd = portable::svd(&scaled, true, true);
        let (Some(u), Some(v_t)) = (svd.u, svd.v_t) else {
            return Err(self.numerical_failure("singular value decomposition"));
        };
        let singular: Vec<f64> = svd.singular_values.iter().map(|value| value.0).collect();
        let Some((weakest_index, s_min, s_max)) = singular_value_extremes(&singular) else {
            return Err(self.numerical_failure("design matrix"));
        };
        let condition_number = if s_min > 0.0 {
            (s_max / s_min).powi(2)
        } else {
            f64::INFINITY
        };
        let weakest = || -> Vec<(String, f64)> {
            let mut components: Vec<(String, f64)> = (0..columns)
                .map(|column| (self.names[column].clone(), v_t[(weakest_index, column)].0))
                .filter(|(_, component)| component.abs() >= WEAKEST_COMPONENT_REPORT)
                .collect();
            components.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
            components
        };
        if s_min <= s_max * (rows as f64) * f64::EPSILON {
            return Err(OrbitParameterFitError::SingularNormalEquations {
                satellite: self.satellite,
                condition_number,
                weakest: weakest(),
            });
        }
        if condition_number > self.options.max_condition_number {
            return Err(OrbitParameterFitError::IllConditionedNormalEquations {
                satellite: self.satellite,
                condition_number,
                limit: self.options.max_condition_number,
                weakest: weakest(),
            });
        }

        // Step: V diag(1/s) U^T r, unscaled by the column norms. Its length
        // |H dx| is |U^T r|.
        let along_columns: Vec<f64> = (0..singular.len())
            .map(|k| {
                (0..rows)
                    .map(|row| u[(row, k)].0 * linearization.residual_km[row])
                    .sum::<f64>()
            })
            .collect();
        let step_km = along_columns
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let projected: Vec<f64> = along_columns
            .iter()
            .zip(&singular)
            .map(|(value, s)| value / s)
            .collect();
        let step = (0..columns)
            .map(|column| {
                norms[column].divide(
                    (0..singular.len())
                        .map(|k| v_t[(k, column)].0 * projected[k])
                        .sum::<f64>(),
                )
            })
            .collect();
        // V diag(1/s^2) V^T; each entry is a sum of the same products in the
        // same order as its transpose, so the matrix is exactly symmetric.
        let scaled_cofactor = DMatrix::from_fn(columns, columns, |row, column| {
            (0..singular.len())
                .map(|k| v_t[(k, row)].0 * v_t[(k, column)].0 / (singular[k] * singular[k]))
                .sum::<f64>()
        });
        Ok(NormalSolution {
            step,
            step_km,
            scaled_cofactor,
            norms,
            condition_number,
        })
    }

    /// The covariance scaled by the variance factor, from the unweighted sum
    /// of squared residuals.
    ///
    /// With uniform weights `1 / sigma^2`, `s0^2 (H^T W H)^-1` is
    /// `(sum r^2 / dof) (H^T H)^-1`, which does not depend on the sigma, and
    /// the correlations are those of the column-scaled cofactor, which
    /// depend on neither the sigma nor the residuals. Each entry is
    /// `cofactor * (rms / |h_r|) * (rms / |h_c|)` with the powers of two of
    /// the rms and the column norms applied last, in one scaling. Computing
    /// them so keeps every reported value finite whenever it is
    /// representable, also for zero residuals, a column norm outside the
    /// binary64 range and an a-priori sigma at the edge of it; a value below
    /// the smallest subnormal, the variance factor included, is rounded to
    /// zero.
    fn covariance(
        &self,
        solution: &NormalSolution,
        sum_squares_km2: f64,
        degrees_of_freedom: usize,
    ) -> Result<OrbitParameterCovariance, OrbitParameterFitError> {
        let columns = self.names.len();
        let rms_km = (sum_squares_km2 / degrees_of_freedom as f64).sqrt();
        // The rms is at most the root of a finite sum, so its metres do not
        // overflow; the sigma stays in metres (see `step_sigmas`).
        let variance_factor = (rms_km * M_PER_KM / self.options.observation_sigma_m).powi(2);
        if !variance_factor.is_finite() {
            return Err(self.numerical_failure("variance factor"));
        }
        // rms / |h_j|, the scale of row and column j, as a fraction below 2
        // and a power of two; zero residuals give a zero fraction.
        let (rms_fraction, rms_exponent) = libm::frexp(rms_km);
        let scales: Vec<(f64, i32)> = solution
            .norms
            .iter()
            .map(|norm| (rms_fraction / norm.fraction, rms_exponent - norm.exponent))
            .collect();
        let cofactor = &solution.scaled_cofactor;
        let mut matrix = DMatrix::zeros(columns, columns);
        let mut correlation = DMatrix::identity(columns, columns);
        for row in 0..columns {
            for column in row..columns {
                let (row_fraction, row_exponent) = scales[row];
                let (column_fraction, column_exponent) = scales[column];
                let value = libm::scalbn(
                    cofactor[(row, column)] * row_fraction * column_fraction,
                    row_exponent + column_exponent,
                );
                matrix[(row, column)] = value;
                matrix[(column, row)] = value;
                if column != row {
                    let rho = cofactor[(row, column)]
                        / (cofactor[(row, row)].sqrt() * cofactor[(column, column)].sqrt());
                    correlation[(row, column)] = rho;
                    correlation[(column, row)] = rho;
                }
            }
        }
        if !matrix
            .iter()
            .chain(correlation.iter())
            .all(|value| value.is_finite())
        {
            return Err(self.numerical_failure("covariance"));
        }
        Ok(OrbitParameterCovariance::Estimated {
            matrix,
            correlation,
            variance_factor,
            degrees_of_freedom,
        })
    }

    fn finish(
        &self,
        estimate: Vec<f64>,
        linearization: Linearization,
        iterations: usize,
        seed_rms_3d_m: f64,
    ) -> Result<OrbitParameterFit, OrbitParameterFitError> {
        let solution = self.solve(&linearization)?;
        let rows = linearization.residual_km.len();
        let columns = self.names.len();
        let covariance = if rows > columns {
            let degrees_of_freedom = rows - columns;
            self.covariance(&solution, linearization.sum_squares_km2, degrees_of_freedom)?
        } else {
            OrbitParameterCovariance::Unbounded
        };

        let mut residuals = Vec::with_capacity(self.observations.len());
        for (state, observation) in linearization.states.iter().zip(self.observations) {
            let rotation = rtn_to_eci_rotation(state.position_array(), state.velocity_array())
                .map_err(|reason| OrbitFitError::RtnFrame {
                    satellite: self.satellite,
                    reason,
                })?;
            let difference_m = (observation.position_km - state.position_km) * M_PER_KM;
            let project = |axis: usize| {
                (0..3)
                    .map(|component| difference_m[component] * rotation[component][axis])
                    .sum::<f64>()
            };
            residuals.push(OrbitRtnResidual {
                epoch_tdb_j2000_s: observation.epoch_tdb_s,
                elapsed_tt_s: observation.epoch_tt_s - self.observations[0].epoch_tt_s,
                radial_m: project(0),
                along_m: project(1),
                cross_m: project(2),
            });
        }
        let residual_stats = residual_statistics(&residuals, self.options.min_ledger_samples);
        if !(residual_stats.rms_3d_m.is_finite() && seed_rms_3d_m.is_finite()) {
            return Err(self.numerical_failure("post-fit residuals"));
        }

        let first = &self.observations[0];
        Ok(OrbitParameterFit {
            satellite: self.satellite,
            initial_state: CartesianState::new(
                first.epoch_tdb_s,
                [estimate[0], estimate[1], estimate[2]],
                [estimate[3], estimate[4], estimate[5]],
            ),
            force_parameters: self
                .selected
                .iter()
                .enumerate()
                .map(|(offset, &index)| EstimatedForceParameter {
                    name: self.names[STATE_PARAMETER_COUNT + offset].clone(),
                    initial_value: self.held_values[index],
                    value: estimate[STATE_PARAMETER_COUNT + offset],
                })
                .collect(),
            parameter_names: self.names.clone(),
            covariance,
            residuals,
            residual_stats,
            seed_rms_3d_m,
            iterations,
            condition_number: solution.condition_number,
        })
    }
}

/// The index and value of the smallest singular value and the largest
/// singular value, or `None` when there is none or any is not finite.
fn singular_value_extremes(singular: &[f64]) -> Option<(usize, f64, f64)> {
    if singular.is_empty() || !singular.iter().all(|value| value.is_finite()) {
        return None;
    }
    let (weakest_index, s_min) =
        singular
            .iter()
            .copied()
            .enumerate()
            .fold((0, f64::INFINITY), |acc, (index, value)| {
                if value < acc.1 {
                    (index, value)
                } else {
                    acc
                }
            });
    let s_max = singular.iter().copied().fold(0.0, f64::max);
    Some((weakest_index, s_min, s_max))
}

/// The Euclidean norm of a design column, `fraction * 2^exponent`. A
/// column of finite entries can have a norm beyond the largest binary64
/// value, `sqrt(2) * f64::MAX` for two entries of `f64::MAX`; kept apart,
/// the power of two never overflows, and dividing by the norm applies it
/// last.
#[derive(Debug, Clone, Copy, PartialEq)]
struct ColumnNorm {
    /// The norm of the column scaled by `2^-exponent`: in `[0.5, sqrt(n))`
    /// for `n` entries, zero for a zero column.
    fraction: f64,
    exponent: i32,
}

impl ColumnNorm {
    /// `value / norm`: the fractions divided, then the difference of the
    /// powers of two applied in one scaling. The scaling is exact unless the
    /// quotient is subnormal, where it rounds once more, or beyond the
    /// binary64 range, where it is infinite.
    fn divide(self, value: f64) -> f64 {
        let (fraction, exponent) = libm::frexp(value);
        libm::scalbn(fraction / self.fraction, exponent - self.exponent)
    }
}

/// The Euclidean norm of a column, from the column scaled by the power of
/// two of its largest magnitude so that no square overflows or loses its
/// value to underflow; `None` when an entry is not finite.
fn column_norm(column: impl Iterator<Item = f64> + Clone) -> Option<ColumnNorm> {
    let largest = column
        .clone()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    if !largest.is_finite() || column.clone().any(f64::is_nan) {
        return None;
    }
    if largest == 0.0 {
        return Some(ColumnNorm {
            fraction: 0.0,
            exponent: 0,
        });
    }
    // largest = m * 2^exponent with m in [0.5, 1): every scaled entry is
    // below 1 in magnitude and the largest is at least 0.5.
    let (_, exponent) = libm::frexp(largest);
    let sum: f64 = column
        .map(|value| libm::scalbn(value, -exponent).powi(2))
        .sum();
    Some(ColumnNorm {
        fraction: sum.sqrt(),
        exponent,
    })
}

fn residual_statistics(
    residuals: &[OrbitRtnResidual],
    min_ledger_samples: usize,
) -> OrbitResidualStats {
    let n = residuals.len();
    let mean_square = |component: fn(&OrbitRtnResidual) -> f64| {
        residuals
            .iter()
            .map(|residual| component(residual).powi(2))
            .sum::<f64>()
            / n as f64
    };
    let radial = mean_square(|residual| residual.radial_m);
    let along = mean_square(|residual| residual.along_m);
    let cross = mean_square(|residual| residual.cross_m);
    OrbitResidualStats {
        radial_rms_m: radial.sqrt(),
        along_rms_m: along.sqrt(),
        cross_rms_m: cross.sqrt(),
        rms_3d_m: (radial + along + cross).sqrt(),
        n,
        low_sample_count: n < min_ledger_samples,
    }
}

/// The force model of a fit: [`OrbitParameterFitOptions::force_model`] with
/// the drag, then the additional forces.
fn build_force(
    options: &OrbitParameterFitOptions,
) -> Result<Box<dyn ForceModel>, crate::astro::error::PropagationError> {
    let base = StatePropagator {
        initial: CartesianState::new(0.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        force_model: options.force_model,
        integrator: options.integrator,
        options: options.integrator_options,
        drag: options.drag,
        space_weather: options.space_weather.clone(),
    }
    .build_force()?;
    if options.additional_forces.is_empty() {
        return Ok(base);
    }
    let mut composite = CompositeForceModel::new();
    composite.add(base);
    for model in &options.additional_forces {
        composite.add(Box::new(Arc::clone(model)));
    }
    Ok(Box::new(composite))
}

/// Indices of the requested parameters in the force model's listing.
fn select_parameters(
    listed: &[ForceParameter],
    requested: &[String],
) -> Result<Vec<usize>, OrbitParameterFitError> {
    let mut selected = Vec::with_capacity(requested.len());
    for (position, name) in requested.iter().enumerate() {
        let invalid = |reason: &'static str| OrbitParameterFitError::InvalidParameter {
            name: name.clone(),
            reason,
        };
        if requested[..position].contains(name) {
            return Err(invalid("is selected more than once"));
        }
        if STATE_PARAMETER_NAMES.contains(&name.as_str()) {
            return Err(invalid("is the name of an initial-state parameter"));
        }
        let mut matches = listed
            .iter()
            .enumerate()
            .filter(|(_, parameter)| &parameter.name == name)
            .map(|(index, _)| index);
        let index = matches
            .next()
            .ok_or_else(|| invalid("is not a parameter of the force models"))?;
        if matches.next().is_some() {
            return Err(invalid("is listed by more than one force model"));
        }
        if !listed[index].value.is_finite() {
            return Err(invalid("has a value that is not finite"));
        }
        selected.push(index);
    }
    Ok(selected)
}

#[cfg(test)]
#[path = "parameters_tests.rs"]
mod tests;
