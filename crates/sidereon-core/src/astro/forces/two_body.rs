use super::scaled::{wide_norm_squared, ForceScalar, Wide};
use crate::astro::constants::MU_EARTH;
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_partials, outer, validate_finite,
    validate_position, AccelerationPartials, ForceModel,
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
        validate_position(state)?;
        validate_finite([self.mu], "mu")?;
        let position = state.position_km;
        let r2 = position.norm_squared();
        let denominator = r2 * r2.sqrt();
        let acceleration = position * (-self.mu / denominator);
        if denominator.is_finite() && acceleration.iter().all(|value| value.is_finite()) {
            return Ok(acceleration);
        }
        let (position, denominator) = scaled_point_mass(state.position_array().map(Wide::from))?;
        let factor = Wide::from(-self.mu) / denominator;
        finite_acceleration(Vector3::from(
            position.map(|value| (value * factor).value()),
        ))
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        expect_parameter_count(parameters, 0)?;
        validate_position(state)?;
        validate_finite([self.mu], "mu")?;
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
    if let Some(gradient) = binary64_point_mass_gradient(separation_km, mu_km3_s2) {
        return Ok(gradient);
    }
    let gradient =
        scaled_point_mass_gradient(<[f64; 3]>::from(separation_km).map(Wide::from), mu_km3_s2)?;
    Ok(Matrix3::from_column_slice(&gradient.map(Wide::value)))
}

pub(super) fn binary64_point_mass_gradient(
    separation_km: Vector3<f64>,
    mu_km3_s2: f64,
) -> Option<Matrix3<f64>> {
    let d2 = separation_km.norm_squared();
    let d = d2.sqrt();
    let denominator = d2 * d;
    let unit = separation_km / d;
    let gradient = (Matrix3::identity() - outer(&unit, &unit) * 3.0) * (mu_km3_s2 / denominator);
    (denominator.is_finite() && gradient.iter().all(|value| value.is_finite())).then_some(gradient)
}

pub(super) fn scaled_point_mass_gradient(
    separation_km: [Wide; 3],
    mu_km3_s2: f64,
) -> Result<[Wide; 9], PropagationError> {
    let (separation, denominator) = scaled_point_mass(separation_km)?;
    let radius = wide_norm_squared(separation).sqrt();
    let unit = separation.map(|value| value / radius);
    let factor = Wide::from(mu_km3_s2) / denominator;
    Ok(core::array::from_fn(|index| {
        let row = index % 3;
        let col = index / 3;
        (Wide::from(f64::from(row == col)) - unit[row] * unit[col] * Wide::from(3.0)) * factor
    }))
}

/// Extending the exponent range keeps the radius cubed available even when
/// binary64 would evaluate it as zero or infinity.
fn scaled_point_mass(position: [Wide; 3]) -> Result<([Wide; 3], Wide), PropagationError> {
    let r2 = wide_norm_squared(position);
    if position.iter().all(|value| value.is_zero()) {
        return Err(PropagationError::NumericalFailure(
            "Zero position magnitude".to_string(),
        ));
    }
    Ok((position, r2 * r2.sqrt()))
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
