use crate::astro::error::PropagationError;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};
use std::sync::Arc;

/// One estimable parameter of a [`ForceModel`]: its name and the value the
/// model holds.
///
/// A parameter's name identifies it in an orbit fit and in the fit's
/// covariance, so it is unique among the force models of one propagation.
/// Names of this crate's models are dotted lowercase, for example
/// [`SolarRadiationPressure::CR_PARAMETER`](crate::astro::forces::SolarRadiationPressure::CR_PARAMETER).
#[derive(Debug, Clone, PartialEq)]
pub struct ForceParameter {
    /// Parameter name.
    pub name: String,
    /// The value the model holds, in the model's own unit for the parameter.
    pub value: f64,
}

impl ForceParameter {
    /// Pair a name with a value.
    pub fn new(name: impl Into<String>, value: f64) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }
}

/// Partial derivatives of an acceleration with respect to the state it was
/// evaluated at.
///
/// Entry `[(i, j)]` is the derivative of acceleration component `i` (km/s^2)
/// with respect to position component `j` (km) or velocity component `j`
/// (km/s), all in the propagator's inertial frame. These are the blocks the
/// variational equations of Montenbruck & Gill, "Satellite Orbits", chapter 7
/// take from each force.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AccelerationPartials {
    /// `d a / d r`, 1/s^2.
    pub wrt_position: Matrix3<f64>,
    /// `d a / d v`, 1/s.
    pub wrt_velocity: Matrix3<f64>,
}

impl AccelerationPartials {
    /// Both blocks zero.
    pub fn zeros() -> Self {
        Self {
            wrt_position: Matrix3::zeros(),
            wrt_velocity: Matrix3::zeros(),
        }
    }

    /// A position-only dependence: `d a / d v` is zero.
    pub fn position_only(wrt_position: Matrix3<f64>) -> Self {
        Self {
            wrt_position,
            wrt_velocity: Matrix3::zeros(),
        }
    }
}

/// Thread-safe source of spacecraft acceleration for numerical orbit propagation.
///
/// [`StatePropagator`](crate::astro::propagator::StatePropagator) builds concrete
/// force implementations behind this interface. A
/// [`CompositeForceModel`](crate::astro::forces::CompositeForceModel) evaluates
/// several implementations and adds their returned accelerations.
///
/// # Estimable parameters
///
/// A model that has parameters an orbit fit can estimate lists them with
/// [`ForceModel::parameters`] and evaluates itself at caller-supplied values
/// with [`ForceModel::acceleration_with_parameters`],
/// [`ForceModel::acceleration_partials`] and [`ForceModel::parameter_partials`].
/// The values are passed in, in the order of [`ForceModel::parameters`]; the
/// model itself is never modified, so one model can be shared between fits.
/// A model without estimable parameters implements only
/// [`ForceModel::acceleration`] and, where it has a closed form,
/// [`ForceModel::acceleration_partials`].
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

    /// The model's estimable parameters, in a fixed order, with the values the
    /// model holds. The default is no parameters.
    fn parameters(&self) -> Vec<ForceParameter> {
        Vec::new()
    }

    /// The number of entries of [`ForceModel::parameters`]. The default counts
    /// them; a model with parameters can return the number directly.
    fn parameter_count(&self) -> usize {
        self.parameters().len()
    }

    /// [`ForceModel::acceleration`] with the estimable parameters replaced by
    /// `parameters`, one value per entry of [`ForceModel::parameters`] in the
    /// same order.
    ///
    /// The default serves a model without estimable parameters: it accepts an
    /// empty slice and returns [`ForceModel::acceleration`]. A model that
    /// lists parameters overrides this method; a slice of another length is
    /// [`PropagationError::InvalidInput`].
    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        expect_parameter_count(parameters, 0)?;
        self.acceleration(state, ctx)
    }

    /// Partial derivatives of [`ForceModel::acceleration_with_parameters`]
    /// with respect to position and velocity.
    ///
    /// The default approximates them with central difference quotients of the
    /// acceleration, one pair of evaluations per state component
    /// ([`difference_quotient_partials`]). A model with a closed form
    /// overrides it.
    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        difference_quotient_partials(self, parameters, state, ctx)
    }

    /// Partial derivative of [`ForceModel::acceleration_with_parameters`] with
    /// respect to each estimable parameter, one vector (km/s^2 per unit of the
    /// parameter) per entry of [`ForceModel::parameters`] in the same order.
    ///
    /// The default serves a model without estimable parameters: it accepts an
    /// empty slice and returns no vectors.
    fn parameter_partials(
        &self,
        parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        expect_parameter_count(parameters, 0)?;
        Ok(Vec::new())
    }
}

impl<T: ForceModel + ?Sized> ForceModel for Arc<T> {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        (**self).acceleration(state, ctx)
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        (**self).parameters()
    }

    fn parameter_count(&self) -> usize {
        (**self).parameter_count()
    }

    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        (**self).acceleration_with_parameters(parameters, state, ctx)
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        (**self).acceleration_partials(parameters, state, ctx)
    }

    fn parameter_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        (**self).parameter_partials(parameters, state, ctx)
    }
}

/// Outer product `a bᵀ`.
pub(crate) fn outer(a: &Vector3<f64>, b: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::from_fn(|row, col| a[row] * b[col])
}

/// Refuse a state with a non-finite position or velocity component.
pub(crate) fn validate_finite_state(state: &CartesianState) -> Result<(), PropagationError> {
    if state
        .position_km
        .iter()
        .chain(state.velocity_km_s.iter())
        .all(|value| value.is_finite())
    {
        Ok(())
    } else {
        Err(PropagationError::InvalidInput(
            "state must be finite".to_string(),
        ))
    }
}

/// Pass an acceleration through when every component is finite; overflow of
/// a finite input is [`PropagationError::NumericalFailure`].
pub(crate) fn finite_acceleration(
    acceleration: Vector3<f64>,
) -> Result<Vector3<f64>, PropagationError> {
    if acceleration.iter().all(|value| value.is_finite()) {
        Ok(acceleration)
    } else {
        Err(PropagationError::NumericalFailure(
            "acceleration not finite".to_string(),
        ))
    }
}

/// Pass parameter partials through when every component is finite; overflow
/// of a finite input is [`PropagationError::NumericalFailure`].
pub(crate) fn finite_parameter_partials(
    partials: Vec<Vector3<f64>>,
) -> Result<Vec<Vector3<f64>>, PropagationError> {
    if partials
        .iter()
        .all(|partial| partial.iter().all(|value| value.is_finite()))
    {
        Ok(partials)
    } else {
        Err(PropagationError::NumericalFailure(
            "parameter partials not finite".to_string(),
        ))
    }
}

/// Pass partials through when every entry is finite; overflow of a finite
/// input is [`PropagationError::NumericalFailure`].
pub(crate) fn finite_partials(
    partials: AccelerationPartials,
) -> Result<AccelerationPartials, PropagationError> {
    if partials
        .wrt_position
        .iter()
        .chain(partials.wrt_velocity.iter())
        .all(|value| value.is_finite())
    {
        Ok(partials)
    } else {
        Err(PropagationError::NumericalFailure(
            "acceleration partials not finite".to_string(),
        ))
    }
}

/// Check that a parameter slice has the length a model expects.
pub fn expect_parameter_count(parameters: &[f64], expected: usize) -> Result<(), PropagationError> {
    if parameters.len() == expected {
        Ok(())
    } else {
        Err(PropagationError::InvalidInput(format!(
            "force model takes {expected} parameter values, got {}",
            parameters.len()
        )))
    }
}

/// Relative position step of [`difference_quotient_partials`].
const DIFFERENCE_QUOTIENT_POSITION_STEP: f64 = 1.0e-6;
/// Relative velocity step of [`difference_quotient_partials`].
const DIFFERENCE_QUOTIENT_VELOCITY_STEP: f64 = 1.0e-6;
/// Smallest position step of [`difference_quotient_partials`], km.
const DIFFERENCE_QUOTIENT_MIN_POSITION_STEP_KM: f64 = 1.0e-6;
/// Smallest velocity step of [`difference_quotient_partials`], km/s.
const DIFFERENCE_QUOTIENT_MIN_VELOCITY_STEP_KM_S: f64 = 1.0e-9;

/// Central difference quotients of a model's acceleration with respect to
/// position and velocity.
///
/// Each of the six state components is displaced by plus and minus a step of
/// one millionth of the position or velocity magnitude (at least 1 mm and
/// 1 um/s), and the acceleration difference is divided by twice the step. This
/// is the default of [`ForceModel::acceleration_partials`], for models whose
/// derivatives have no closed form here; it costs twelve acceleration
/// evaluations.
pub fn difference_quotient_partials<F: ForceModel + ?Sized>(
    model: &F,
    parameters: &[f64],
    state: &CartesianState,
    ctx: &PropagationContext,
) -> Result<AccelerationPartials, PropagationError> {
    validate_finite_state(state)?;
    let position_step = (state.position_km.norm() * DIFFERENCE_QUOTIENT_POSITION_STEP)
        .max(DIFFERENCE_QUOTIENT_MIN_POSITION_STEP_KM);
    let velocity_step = (state.velocity_km_s.norm() * DIFFERENCE_QUOTIENT_VELOCITY_STEP)
        .max(DIFFERENCE_QUOTIENT_MIN_VELOCITY_STEP_KM_S);
    if !position_step.is_finite() || !velocity_step.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "state too large for difference quotients".to_string(),
        ));
    }

    let mut partials = AccelerationPartials::zeros();
    for axis in 0..3 {
        let mut plus = *state;
        let mut minus = *state;
        plus.position_km[axis] += position_step;
        minus.position_km[axis] -= position_step;
        let column = (model.acceleration_with_parameters(parameters, &plus, ctx)?
            - model.acceleration_with_parameters(parameters, &minus, ctx)?)
            / (plus.position_km[axis] - minus.position_km[axis]);
        partials.wrt_position.set_column(axis, &column);

        let mut plus = *state;
        let mut minus = *state;
        plus.velocity_km_s[axis] += velocity_step;
        minus.velocity_km_s[axis] -= velocity_step;
        let column = (model.acceleration_with_parameters(parameters, &plus, ctx)?
            - model.acceleration_with_parameters(parameters, &minus, ctx)?)
            / (plus.velocity_km_s[axis] - minus.velocity_km_s[axis]);
        partials.wrt_velocity.set_column(axis, &column);
    }
    finite_partials(partials)
}
