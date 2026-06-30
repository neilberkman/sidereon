use crate::astro::constants::MU_EARTH;
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

pub struct TwoBodyGravity {
    pub mu: f64,
}

impl Default for TwoBodyGravity {
    fn default() -> Self {
        Self { mu: MU_EARTH }
    }
}

impl ForceModel for TwoBodyGravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let r_mag2 = state.position_km.norm_squared();
        if r_mag2 == 0.0 {
            return Err(PropagationError::NumericalFailure(
                "Zero position magnitude".to_string(),
            ));
        }
        let r_mag = r_mag2.sqrt();
        Ok(state.position_km * (-self.mu / (r_mag2 * r_mag)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::propagator::api::PropagationContext;
    use crate::astro::state::CartesianState;

    #[test]
    fn acceleration_matches_orbis_force_wrapper_bits() {
        let state = CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [0.0, 0.0, 0.0]);
        let acceleration = TwoBodyGravity::default()
            .acceleration(&state, &PropagationContext::default())
            .unwrap();

        assert_eq!(acceleration.x.to_bits(), 13_798_562_943_973_640_097);
        assert_eq!(acceleration.y.to_bits(), 4_563_548_234_789_153_053);
        assert_eq!(acceleration.z.to_bits(), 13_787_359_517_156_423_902);
    }
}
