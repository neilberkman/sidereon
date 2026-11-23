use crate::astro::constants::MU_EARTH;
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_partials, outer, validate_finite_state, AccelerationPartials,
    ForceModel,
};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};

/// Central point-mass gravity model for Cartesian propagation.
///
/// [`ForceModel::acceleration`] returns `-mu * position_km / |position_km|³`
/// in the propagator's inertial frame. [`Default::default`] uses
/// [`MU_EARTH`], and an exact zero position norm is reported as
/// [`PropagationError::NumericalFailure`].
/// [`ForceModel::acceleration_partials`] is the closed form
/// `-mu / r³ * (I - 3 r̂ r̂ᵀ)` (Montenbruck & Gill, "Satellite Orbits",
/// section 7.3).
pub struct TwoBodyGravity {
    /// Gravitational parameter in km³/s² used to scale the central point-mass
    /// acceleration.
    ///
    /// [`Default::default`] sets this to [`MU_EARTH`], while
    /// `ForceModelKind::build` copies the configured central-gravity parameter
    /// into it for legacy and composite force-model variants.
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

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        expect_parameter_count(parameters, 0)?;
        validate_finite_state(state)?;
        point_mass_gradient(state.position_km, self.mu)
            .map(|gradient| AccelerationPartials::position_only(-gradient))
            .and_then(finite_partials)
    }
}

/// `mu / d³ * (I - 3 d̂ d̂ᵀ)` for a separation vector `d`: the negated position
/// gradient of the point-mass acceleration `-mu d / d³`.
pub(crate) fn point_mass_gradient(
    separation_km: Vector3<f64>,
    mu_km3_s2: f64,
) -> Result<Matrix3<f64>, PropagationError> {
    let d2 = separation_km.norm_squared();
    if d2 == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Zero position magnitude".to_string(),
        ));
    }
    let d = d2.sqrt();
    let unit = separation_km / d;
    Ok((Matrix3::identity() - outer(&unit, &unit) * 3.0) * (mu_km3_s2 / (d2 * d)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::propagator::api::PropagationContext;
    use crate::astro::state::CartesianState;

    #[test]
    fn acceleration_matches_pinned_bits() {
        let state = CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [0.0, 0.0, 0.0]);
        let acceleration = TwoBodyGravity::default()
            .acceleration(&state, &PropagationContext::default())
            .unwrap();

        assert_eq!(acceleration.x.to_bits(), 13_798_562_943_973_640_097);
        assert_eq!(acceleration.y.to_bits(), 4_563_548_234_789_153_053);
        assert_eq!(acceleration.z.to_bits(), 13_787_359_517_156_423_902);
    }
}
