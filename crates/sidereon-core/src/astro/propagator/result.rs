use crate::astro::error::PropagationError;
use crate::astro::events::DetectedEvent;
use crate::astro::propagator::dense_output::DenseOutput;
use crate::astro::state::CartesianState;

#[derive(Debug, Clone)]
pub struct PropagationPoint {
    pub epoch_tdb_seconds: f64,
    pub position_km: [f64; 3],
    pub velocity_km_s: [f64; 3],
}

#[derive(Debug, Clone, Default)]
pub struct PropagationStats {
    pub accepted_steps: u32,
    pub rejected_steps: u32,
    pub evaluations: u32,
}

#[derive(Debug, Clone)]
pub struct PropagationResult {
    pub final_state: CartesianState,
    pub points: Vec<PropagationPoint>,
    pub events: Vec<DetectedEvent>,
    pub stats: PropagationStats,
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
