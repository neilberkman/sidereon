use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::integrators::DynamicsModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::{CartesianState, StateDerivative};

pub struct OrbitalDynamics<'a> {
    pub force_model: &'a dyn ForceModel,
}

impl<'a> DynamicsModel for OrbitalDynamics<'a> {
    fn derivative(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<StateDerivative, PropagationError> {
        let accel = self.force_model.acceleration(state, ctx)?;
        Ok(StateDerivative {
            dpos_km_s: state.velocity_km_s,
            dvel_km_s2: accel,
        })
    }
}
