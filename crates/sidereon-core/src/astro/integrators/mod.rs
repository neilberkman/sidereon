/// Adaptive Dormand-Prince 5(4) integrator implementation ([`DP54`]) using a
/// 7-stage DOPRI5 tableau, PI step-size controller, and dense-output generation.
pub mod dp54;
/// Fixed-step fourth-order Runge-Kutta integrator implementation ([`RK4`])
/// advancing Cartesian states with four derivative stages per step.
pub mod rk4;
/// Butcher tableau structures for Runge-Kutta integrators, including the
/// DOPRI5 coefficients used by [`DP54`].
pub mod tableau;

pub use dp54::DP54;
pub use rk4::RK4;

use crate::astro::error::PropagationError;
use crate::astro::propagator::api::{IntegratorOptions, PropagationContext};
use crate::astro::propagator::result::PropagationResult;
use crate::astro::state::{CartesianState, StateDerivative};

/// Right-hand-side equations of motion `d/dt [r; v] = f(t, [r, v])` for
/// numerical orbit integration.
///
/// Implementations provide state derivatives for a given [`CartesianState`] and
/// [`PropagationContext`], such as
/// [`OrbitalDynamics`](crate::astro::propagator::dynamics::OrbitalDynamics)
/// querying configured force models.
pub trait DynamicsModel: Send + Sync {
    /// Evaluates the state derivative for a given [`CartesianState`] and
    /// [`PropagationContext`].
    ///
    /// Assigns position derivative `dpos_km_s` from velocity in km/s and
    /// velocity derivative `dvel_km_s2` from force model acceleration in km/s²,
    /// returning [`PropagationError`] on force evaluation failure.
    fn derivative(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<StateDerivative, PropagationError>;
}

/// Numerical integrator interface for advancing Cartesian orbital states across
/// time intervals in TDB seconds.
///
/// Implementations such as [`RK4`] and [`DP54`] advance an initial state to a
/// target epoch using equations of motion specified by a [`DynamicsModel`].
pub trait Integrator: Send + Sync {
    /// Integrates equations of motion `rhs` from `initial.epoch_tdb_seconds` to
    /// `t_end_seconds` in TDB.
    ///
    /// Supports forward and backward integration, validates finite epochs and
    /// step options, and returns [`PropagationError::MaxStepsExceeded`] if the
    /// step count reaches `opts.max_steps`.
    fn propagate(
        &self,
        initial: CartesianState,
        t_end_seconds: f64,
        rhs: &dyn DynamicsModel,
        ctx: &PropagationContext,
        opts: &IntegratorOptions,
    ) -> Result<PropagationResult, PropagationError>;
}
