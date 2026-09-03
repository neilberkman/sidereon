use crate::astro::error::PropagationError;
use crate::astro::frames::orientation::EarthOrientationProvider;
use crate::constants::SECONDS_PER_HOUR;
use std::sync::Arc;

/// Per-evaluation context shared with force models.
///
/// The default context is intentionally empty. A caller that wants a body-fixed
/// force to use the precise Earth-fixed frame can attach an
/// [`EarthOrientationProvider`], while existing force models and default
/// propagation remain bit-identical.
#[derive(Clone, Default)]
pub struct PropagationContext {
    body_fixed_frame_provider: Option<Arc<dyn EarthOrientationProvider>>,
}

impl core::fmt::Debug for PropagationContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PropagationContext")
            .field(
                "body_fixed_frame_provider",
                &self.body_fixed_frame_provider.is_some(),
            )
            .finish()
    }
}

impl PropagationContext {
    /// Build an empty propagation context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach a body-fixed frame provider.
    pub fn with_body_fixed_frame_provider(
        mut self,
        provider: Arc<dyn EarthOrientationProvider>,
    ) -> Self {
        self.body_fixed_frame_provider = Some(provider);
        self
    }

    /// Return the body-fixed frame provider, if one was attached.
    pub fn body_fixed_frame_provider(&self) -> Option<&dyn EarthOrientationProvider> {
        self.body_fixed_frame_provider
            .as_deref()
            .map(|provider| provider as &dyn EarthOrientationProvider)
    }
}

/// Options forwarded to the selected [`crate::astro::integrators::Integrator`]
/// by [`crate::astro::propagator::StatePropagator`].
///
/// [`crate::astro::integrators::RK4`] uses the initial step, step limit, and
/// point-output flag; [`crate::astro::integrators::DP54`] also uses the
/// tolerance and adaptive step fields.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct IntegratorOptions {
    /// Additive error scale used by DP54 for the position and velocity error
    /// estimates. The adaptive validator requires a finite positive value;
    /// RK4 does not read this field.
    pub abs_tol: f64,
    /// Relative error factor used by DP54 with the larger current/proposed
    /// position and velocity norms. The adaptive validator requires a finite
    /// positive value; RK4 does not read this field.
    pub rel_tol: f64,
    /// Minimum step magnitude in seconds accepted after a rejected DP54 step.
    /// If the controller proposes a smaller magnitude, DP54 returns a
    /// [`PropagationError::NumericalFailure`].
    pub min_step: f64,
    /// Maximum step magnitude in seconds used by DP54 when clamping its
    /// initial and controller-selected steps. RK4 validates this field but
    /// does not use it to select steps.
    pub max_step: f64,
    /// Initial step magnitude in seconds. Both integrators limit it to the
    /// absolute target span and apply the direction toward the target; DP54
    /// also clamps it to `max_step`.
    pub initial_step: f64,
    /// Maximum number of step outcomes allowed for one propagation. RK4 counts
    /// completed steps, while DP54 counts accepted and rejected steps; either
    /// integrator returns [`PropagationError::MaxStepsExceeded`] at the limit.
    pub max_steps: u32,
    /// Whether to retain every completed step in the `points` field of
    /// [`crate::astro::propagator::PropagationResult`].
    /// DP54 additionally captures its stages and returns dense output when this
    /// is enabled; with it disabled, DP54 returns no dense output.
    pub dense_output: bool,
}

impl Default for IntegratorOptions {
    fn default() -> Self {
        Self {
            abs_tol: 1e-9,
            rel_tol: 1e-12,
            min_step: 1e-6,
            max_step: SECONDS_PER_HOUR,
            initial_step: 60.0,
            max_steps: 1_000_000,
            dense_output: false,
        }
    }
}

pub(crate) fn validate_integrator_options(
    opts: &IntegratorOptions,
) -> Result<(), PropagationError> {
    validate_step_options(opts)
}

pub(crate) fn validate_adaptive_integrator_options(
    opts: &IntegratorOptions,
) -> Result<(), PropagationError> {
    validate_step_options(opts)?;
    crate::validate::finite_positive(opts.abs_tol, "abs_tol").map_err(map_field_error)?;
    crate::validate::finite_positive(opts.rel_tol, "rel_tol").map_err(map_field_error)?;
    Ok(())
}

pub(crate) fn validate_integrator_epoch(
    value: f64,
    field: &'static str,
) -> Result<(), PropagationError> {
    crate::validate::finite(value, field)
        .map(|_| ())
        .map_err(map_field_error)
}

fn validate_step_options(opts: &IntegratorOptions) -> Result<(), PropagationError> {
    crate::validate::positive_step(opts.initial_step, "initial_step").map_err(map_field_error)?;
    crate::validate::positive_step(opts.min_step, "min_step").map_err(map_field_error)?;
    crate::validate::positive_step(opts.max_step, "max_step").map_err(map_field_error)?;
    Ok(())
}

fn map_field_error(error: crate::validate::FieldError) -> PropagationError {
    PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
}
