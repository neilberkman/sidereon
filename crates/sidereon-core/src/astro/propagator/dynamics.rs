use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::integrators::DynamicsModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::{CartesianState, StateDerivative};

/// Adapter that supplies Cartesian state rates from a [`ForceModel`].
///
/// [`crate::astro::propagator::StatePropagator`] constructs it around the force
/// model selected for propagation, and [`DynamicsModel`] consumers request its
/// derivative during [`crate::astro::integrators::RK4`] or
/// [`crate::astro::integrators::DP54`] integration. The derivative uses the
/// state's velocity as the position rate and the force model's returned
/// acceleration as the velocity rate, returning any [`PropagationError`]
/// unchanged.
pub struct OrbitalDynamics<'a> {
    /// Borrowed [`ForceModel`] used for acceleration evaluation.
    ///
    /// Each derivative call passes the current [`CartesianState`] and
    /// [`PropagationContext`] to this model. Its returned vector becomes
    /// [`StateDerivative::dvel_km_s2`].
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
