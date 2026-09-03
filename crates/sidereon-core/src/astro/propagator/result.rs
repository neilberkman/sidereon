use crate::astro::error::PropagationError;
use crate::astro::events::DetectedEvent;
use crate::astro::propagator::dense_output::DenseOutput;
use crate::astro::state::CartesianState;

#[derive(Debug, Clone)]
/// A Cartesian state sample emitted by a numerical integrator.
///
/// The built-in integrators emit the initial sample and, when dense output is
/// enabled, a sample at every accepted step endpoint. Without dense output,
/// they emit the initial and final states.
pub struct PropagationPoint {
    /// Absolute TDB epoch in seconds copied from the integrator's current time.
    /// The first sample uses the initial epoch, and later samples use accepted
    /// step endpoints.
    pub epoch_tdb_seconds: f64,
    /// Cartesian x, y, and z position components in kilometers, copied by
    /// [`CartesianState::position_array`]. A non-finite component is rejected
    /// before the propagation result is returned.
    pub position_km: [f64; 3],
    /// Cartesian x, y, and z velocity components in kilometers per second,
    /// copied by [`CartesianState::velocity_array`]. A non-finite component is
    /// rejected before the propagation result is returned.
    pub velocity_km_s: [f64; 3],
}

#[derive(Debug, Clone, Default)]
/// Work counters collected while building a [`PropagationResult`].
///
/// RK4 reports no rejected attempts, while DP54 records adaptive error-control
/// rejections; both integrators count derivative evaluations.
pub struct PropagationStats {
    /// Number of integration steps accepted and applied to advance the state.
    pub accepted_steps: u32,
    /// Number of DP54 attempts whose estimated error exceeded 1 and were
    /// retried with a smaller step; RK4 always reports zero.
    pub rejected_steps: u32,
    /// Number of right-hand-side derivative evaluations used by the integrator.
    /// RK4 uses four per step, while DP54 includes its initial FSAL evaluation
    /// and evaluations from accepted and rejected attempts.
    pub evaluations: u32,
}

#[derive(Debug, Clone)]
/// Aggregate returned by a numerical integrator after propagation.
///
/// The built-in RK4 and DP54 paths validate the final state and every emitted
/// [`PropagationPoint`] before returning this aggregate.
pub struct PropagationResult {
    /// Cartesian state held when the integrator reaches the requested end
    /// epoch. A zero-duration DP54 request returns the unchanged initial state.
    pub final_state: CartesianState,
    /// Emitted samples in integration order. With dense output enabled, entries
    /// after the first correspond to accepted step endpoints; otherwise only
    /// the initial and final states are emitted.
    pub points: Vec<PropagationPoint>,
    /// Event records returned by the propagation path. The built-in RK4 and
    /// DP54 paths set this to an empty vector because they do not invoke an
    /// event finder.
    pub events: Vec<DetectedEvent>,
    /// Step and derivative-evaluation counters from the selected integrator.
    /// A zero-duration DP54 request reports zero for every counter.
    pub stats: PropagationStats,
    /// DP54's optional continuous interpolant. It is `Some` only when dense
    /// output is enabled, with one segment per accepted step; RK4 always leaves
    /// it `None`. A zero-duration DP54 request has an empty interpolant when
    /// dense output is enabled.
    pub dense: Option<DenseOutput>,
}

pub(crate) fn validate_propagation_result(
    result: PropagationResult,
) -> Result<PropagationResult, PropagationError> {
    validate_epoch_finite(
        result.final_state.epoch_tdb_seconds,
        "final_state.epoch_tdb_seconds",
    )?;
    validate_state_vector(
        result.final_state.position_array(),
        "final_state.position_km",
    )?;
    validate_state_vector(
        result.final_state.velocity_array(),
        "final_state.velocity_km_s",
    )?;

    for point in &result.points {
        validate_epoch_finite(point.epoch_tdb_seconds, "points.epoch_tdb_seconds")?;
        validate_state_vector(point.position_km, "points.position_km")?;
        validate_state_vector(point.velocity_km_s, "points.velocity_km_s")?;
    }

    Ok(result)
}

fn validate_state_vector(values: [f64; 3], field: &'static str) -> Result<(), PropagationError> {
    crate::validate::finite_slice(&values, field).map_err(|error| {
        PropagationError::NumericalFailure(format!("{} {}", error.field(), error.reason()))
    })
}

fn validate_epoch_finite(value: f64, field: &'static str) -> Result<(), PropagationError> {
    crate::validate::finite(value, field)
        .map(|_| ())
        .map_err(|error| {
            PropagationError::NumericalFailure(format!("{} {}", error.field(), error.reason()))
        })
}
