use crate::astro::error::PropagationError;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

/// Thread-safe source of spacecraft acceleration for numerical orbit propagation.
///
/// [`StatePropagator`](crate::astro::propagator::StatePropagator) builds concrete
/// force implementations behind this interface. A
/// [`CompositeForceModel`](crate::astro::forces::CompositeForceModel) evaluates
/// several implementations and adds their returned accelerations.
pub trait ForceModel: Send + Sync {
    /// Evaluate this model at a propagation state and epoch.
    ///
    /// The state contains an absolute TDB epoch, position in kilometers, and
    /// velocity in kilometers per second. Implementations return acceleration
    /// in kilometers per second squared in the propagator's inertial frame;
    /// [`OrbitalDynamics`](crate::astro::propagator::OrbitalDynamics) writes the
    /// result directly into [`StateDerivative`](crate::astro::state::StateDerivative)'s
    /// `dvel_km_s2` field. Models that need body-fixed coordinates obtain them
    /// from `ctx`; singularities, invalid parameters, and model or frame
    /// failures are returned as [`PropagationError`].
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError>;
}
