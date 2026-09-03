use thiserror::Error;

/// Failure categories returned by the numerical propagators and force models.
/// [`crate::astro::integrators::Integrator::propagate`] and
/// [`crate::astro::forces::ForceModel::acceleration`] use this enum for input,
/// numerical, step-budget, and delegated force-model failures.
#[derive(Error, Debug, Clone)]
pub enum PropagationError {
    /// An integration or force-model input failed validation; the payload
    /// contains the field/reason text or a model-specific validation message.
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// A propagation calculation produced a non-finite result or encountered
    /// an unusable numerical intermediate; the payload identifies the context.
    #[error("Numerical failure: {0}")]
    NumericalFailure(String),

    /// The integrator reached [`crate::astro::propagator::api::IntegratorOptions::max_steps`]
    /// before reaching its target epoch.
    #[error("Maximum number of steps exceeded")]
    MaxStepsExceeded,

    /// When formatted, prefixes its payload with `Event failure: `.
    #[error("Event failure: {0}")]
    EventFailure(String),

    /// A force-model dependency or frame evaluation failed; the payload
    /// preserves contextual text added by the calling force implementation.
    #[error("Force model failure: {0}")]
    ForceModelFailure(String),
}
