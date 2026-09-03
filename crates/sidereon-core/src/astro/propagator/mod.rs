/// Propagation context and integrator configuration options.
///
/// Provides [`PropagationContext`] for passing optional body-fixed frame
/// providers and [`IntegratorOptions`] for configuring tolerances and step sizes.
pub mod api;
/// Adaptive step-size controller for Runge-Kutta integration.
///
/// Implements [`PIController`](controller::PIController) using a Hairer/Wanner
/// power-law factor to adjust signed integration steps based on normalized
/// error estimates.
pub mod controller;
pub mod covariance;
pub mod decay;
/// Continuous polynomial interpolation along accepted integration steps.
///
/// Implements Shampine's fourth-order continuous extension for DP5(4),
/// evaluated through [`DenseOutput`](dense_output::DenseOutput).
pub mod dense_output;
pub mod driver;
/// Right-hand-side equations of motion for orbital states.
///
/// Provides [`OrbitalDynamics`], evaluating Cartesian position derivatives
/// in km/s and force model accelerations in km/s².
pub mod dynamics;
pub mod numerical;
/// Output structures returned by numerical integrators.
///
/// Defines [`PropagationResult`], discrete [`PropagationPoint`] trajectory
/// samples at TDB epochs, and work counters in [`PropagationStats`].
pub mod result;

pub use crate::astro::forces::DragParameters;
pub use api::{IntegratorOptions, PropagationContext};
pub use covariance::{
    transport_covariance, CovarianceEphemeris, CovarianceFrame, CovarianceNode,
    CovariancePropagationOptions, CovarianceSegment, LabeledCovariance6, ProcessNoise,
};
pub use decay::{
    estimate_decay, estimate_decay_with_source, DecayConfig, DecayError, DecayEstimate,
};
pub use driver::{
    propagate_states, propagate_states_with_context, PropagationConfig, PropagationForceModel,
};
pub use dynamics::OrbitalDynamics;
pub use numerical::{
    ForceModelComponents, ForceModelKind, IntegratorKind, StatePropagator, StateTransitionMatrix,
};
pub use result::{PropagationPoint, PropagationResult, PropagationStats};
