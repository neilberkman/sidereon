use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

#[derive(Default)]
pub struct CompositeForceModel {
    pub models: Vec<Box<dyn ForceModel>>,
}

impl CompositeForceModel {
    pub fn new() -> Self {
        Self::default()
    }

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
