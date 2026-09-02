use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

#[derive(Default)]
/// An additive force model that combines the accelerations returned by its
/// component [`ForceModel`] implementations.
///
/// `ForceModelKind::build` uses it for two-body-plus-J2 and configured composite
/// propagation, while `StatePropagator::build_force` uses it to layer drag over
/// gravity. The summed vector is consumed by
/// [`crate::astro::propagator::dynamics::OrbitalDynamics`] as the km/s²
/// velocity derivative.
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
        let mut accel = Vector3::zeros();
        for model in &self.models {
            accel += model.acceleration(state, ctx)?;
        }
        Ok(accel)
    }
}
