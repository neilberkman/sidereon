use super::scaled::ForceSum;
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_parameter_partials, finite_partials,
    AccelerationPartials, ForceModel, ForceParameter,
};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};

#[derive(Default)]
/// An additive force model that combines the accelerations returned by its
/// component [`ForceModel`] implementations.
///
/// `ForceModelKind::build` uses it for two-body-plus-J2 and configured composite
/// propagation, while `StatePropagator::force_model_with_drag` uses it to layer drag over
/// gravity. The summed vector is consumed by
/// [`crate::astro::propagator::dynamics::OrbitalDynamics`] as the km/s²
/// velocity derivative.
///
/// Its estimable parameters are those of its components, concatenated in
/// component order; the state partials are the sums of the components'. A
/// sum that overflows is [`PropagationError::NumericalFailure`].
pub struct CompositeForceModel {
    /// Force components in the order in which they are evaluated.
    ///
    /// [`Self::add`] appends each model, and
    /// [`ForceModel::acceleration`] visits every entry, adds its returned
    /// vector to a zero vector, and returns zero when the list is empty. An
    /// error from a component is returned before later components are run.
    pub models: Vec<Box<dyn ForceModel>>,
}

impl CompositeForceModel {
    /// Creates an empty composite with no force components.
    ///
    /// This calls the derived default, so [`ForceModel::acceleration`] returns
    /// a zero vector until a model is added.
    pub fn new() -> Self {
        Self::default()
    }

    /// Appends a boxed force model to the end of [`Self::models`].
    ///
    /// The model is evaluated after models already present, so the acceleration
    /// sum and any component error follow insertion order. Numerical builders
    /// use this ordering to place gravity before J2 or other perturbations and
    /// before an added drag model.
    pub fn add(&mut self, model: Box<dyn ForceModel>) {
        self.models.push(model);
    }
}

impl ForceModel for CompositeForceModel {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let mut accel = ForceSum::new();
        for model in &self.models {
            accel.add(model.acceleration(state, ctx)?.into());
        }
        finite_acceleration(Vector3::from(accel.values()))
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        self.models
            .iter()
            .flat_map(|model| model.parameters())
            .collect()
    }

    fn parameter_count(&self) -> usize {
        self.models
            .iter()
            .map(|model| model.parameter_count())
            .sum()
    }

    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let mut accel = ForceSum::new();
        self.for_each_component(parameters, |model, own| {
            accel.add(
                finite_acceleration(model.acceleration_with_parameters(own, state, ctx)?)?.into(),
            );
            Ok(())
        })?;
        finite_acceleration(Vector3::from(accel.values()))
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        let mut position: ForceSum<9> = ForceSum::new();
        let mut velocity: ForceSum<9> = ForceSum::new();
        self.for_each_component(parameters, |model, own| {
            let component = model.acceleration_partials(own, state, ctx)?;
            let component = finite_partials(component)?;
            position.add(core::array::from_fn(|i| component.wrt_position[i]));
            velocity.add(core::array::from_fn(|i| component.wrt_velocity[i]));
            Ok(())
        })?;
        finite_partials(AccelerationPartials {
            wrt_position: Matrix3::from_column_slice(&position.values()),
            wrt_velocity: Matrix3::from_column_slice(&velocity.values()),
        })
    }

    fn parameter_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        let mut partials = Vec::with_capacity(parameters.len());
        self.for_each_component(parameters, |model, own| {
            let component = model.parameter_partials(own, state, ctx)?;
            if component.len() != own.len() {
                return Err(PropagationError::InvalidInput(format!(
                    "force model returned {} parameter partials for {} parameters",
                    component.len(),
                    own.len()
                )));
            }
            partials.extend(component);
            Ok(())
        })?;
        finite_parameter_partials(partials)
    }
}

impl CompositeForceModel {
    /// Visit each component with its own slice of the concatenated parameter
    /// values.
    fn for_each_component(
        &self,
        parameters: &[f64],
        mut visit: impl FnMut(&dyn ForceModel, &[f64]) -> Result<(), PropagationError>,
    ) -> Result<(), PropagationError> {
        let mut offset = 0_usize;
        for model in &self.models {
            let count = model.parameter_count();
            let end = offset.checked_add(count).ok_or_else(|| {
                PropagationError::InvalidInput("composite parameter count overflows".to_string())
            })?;
            let own = parameters.get(offset..end).ok_or_else(|| {
                PropagationError::InvalidInput(format!(
                    "composite force model takes at least {} parameter values, got {}",
                    end,
                    parameters.len()
                ))
            })?;
            visit(model.as_ref(), own)?;
            offset = end;
        }
        expect_parameter_count(parameters, offset)
    }
}
