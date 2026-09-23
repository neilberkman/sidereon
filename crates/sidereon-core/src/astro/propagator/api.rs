use crate::astro::error::PropagationError;
use crate::astro::frames::orientation::{EarthOrientation, EarthOrientationProvider};
use crate::astro::time::eop::Ut1DepartureRecord;
use crate::astro::time::DegradeReason;
use crate::constants::SECONDS_PER_HOUR;
use std::sync::Arc;

/// Per-evaluation context shared with force models.
///
/// The default context is intentionally empty. A caller that wants a body-fixed
/// force to use the precise Earth-fixed frame can attach an
/// [`EarthOrientationProvider`], while existing force models and default
/// propagation remain bit-identical.
///
/// A provider built under [`crate::astro::time::ValidityMode::Permissive`]
/// (for example [`crate::astro::frames::TdbEarthOrientationProvider::with_validity`])
/// evaluates epochs outside the UT1 table with the long-term UT1. Every such
/// orientation a force model uses is recorded here:
/// [`PropagationContext::ut1_departure`] returns the first departure recorded
/// through this context or any clone of it, and each propagation also returns
/// its own in [`crate::astro::propagator::PropagationResult::ut1_degraded`].
#[derive(Clone, Default)]
pub struct PropagationContext {
    body_fixed_frame_provider: Option<Arc<dyn EarthOrientationProvider>>,
    ut1_departures: Ut1DepartureRecord,
}

impl core::fmt::Debug for PropagationContext {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PropagationContext")
            .field(
                "body_fixed_frame_provider",
                &self.body_fixed_frame_provider.is_some(),
            )
            .field("ut1_departure", &self.ut1_departure())
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
    ///
    /// A force model that takes an orientation from it passes the result
    /// through [`PropagationContext::record_orientation`], so a departure from
    /// the UT1 table is reported.
    pub fn body_fixed_frame_provider(&self) -> Option<&dyn EarthOrientationProvider> {
        self.body_fixed_frame_provider
            .as_deref()
            .map(|provider| provider as &dyn EarthOrientationProvider)
    }

    /// Record the UT1 departure an orientation carries, if any, and return it.
    ///
    /// Custom force models that query [`PropagationContext::body_fixed_frame_provider`]
    /// call this on every orientation they use.
    pub fn record_orientation(&self, orientation: EarthOrientation) -> EarthOrientation {
        self.ut1_departures.record(orientation.ut1_degraded());
        orientation
    }

    /// The first UT1 departure recorded through this context or a clone of it,
    /// that is, the first epoch a force model evaluated outside the UT1 table
    /// with an orientation a permissive provider accepted. `None` when every
    /// orientation used came from inside the table.
    pub fn ut1_departure(&self) -> Option<DegradeReason> {
        self.ut1_departures.first()
    }

    /// This context with its own, empty departure record, for one run whose
    /// departure is reported separately; [`PropagationContext::merge_departure`]
    /// passes it back.
    pub(crate) fn with_fresh_departure_record(&self) -> Self {
        Self {
            body_fixed_frame_provider: self.body_fixed_frame_provider.clone(),
            ut1_departures: Ut1DepartureRecord::default(),
        }
    }

    /// Record a departure found by a run on a fresh-record copy.
    pub(crate) fn merge_departure(&self, departure: Option<DegradeReason>) {
        self.ut1_departures.record(departure);
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
