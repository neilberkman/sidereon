pub mod dp54;
pub mod rk4;
pub mod tableau;

pub use dp54::DP54;
pub use rk4::RK4;

use crate::astro::error::PropagationError;
use crate::astro::propagator::api::{IntegratorOptions, PropagationContext};
use crate::astro::propagator::result::PropagationResult;
use crate::astro::state::{CartesianState, StateDerivative};

pub trait DynamicsModel: Send + Sync {
    fn derivative(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<StateDerivative, PropagationError>;
}

pub trait Integrator: Send + Sync {
    fn propagate(
        &self,
        initial: CartesianState,
        t_end_seconds: f64,
        rhs: &dyn DynamicsModel,
        ctx: &PropagationContext,
        opts: &IntegratorOptions,
    ) -> Result<PropagationResult, PropagationError>;
}
