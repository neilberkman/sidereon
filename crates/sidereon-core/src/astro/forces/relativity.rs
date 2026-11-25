//! Geocentric Schwarzschild relativistic correction.
//!
//! The acceleration is the single central-body term
//! `GM/(c^2 r^3) * [ (4 GM/r - v^2) r + 4 (r . v) v ]`, as given by
//! Montenbruck & Gill, "Satellite Orbits", equation 3.146. Position is in km,
//! velocity is in km/s, and the returned acceleration is km/s^2. The term
//! depends on both position and velocity; both partial-derivative blocks are
//! the closed-form derivatives of that expression.

use super::scaled::{wide_dot, wide_norm_squared, Wide};
use crate::astro::constants::physics::SPEED_OF_LIGHT_KM_S;
use crate::astro::constants::MU_EARTH;
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_partials, outer, validate_finite,
    validate_finite_state, AccelerationPartials, ForceModel,
};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};

/// Schwarzschild correction for a single central body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SchwarzschildRelativity {
    /// Central-body gravitational parameter, km^3/s^2.
    pub mu_km3_s2: f64,
    /// Speed of light, km/s.
    pub c_km_s: f64,
}

impl Default for SchwarzschildRelativity {
    fn default() -> Self {
        Self {
            mu_km3_s2: MU_EARTH,
            c_km_s: SPEED_OF_LIGHT_KM_S,
        }
    }
}

impl SchwarzschildRelativity {
    /// Build a Schwarzschild correction with explicit constants.
    pub fn new(mu_km3_s2: f64, c_km_s: f64) -> Self {
        Self { mu_km3_s2, c_km_s }
    }
}

impl ForceModel for SchwarzschildRelativity {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.validate(state)?;
        if self.c_km_s == f64::INFINITY {
            return Ok(Vector3::zeros());
        }
        let r2 = state.position_km.norm_squared();
        let r = r2.sqrt();
        let r3 = r2 * r;
        let v2 = state.velocity_km_s.norm_squared();
        let r_dot_v = state.position_km.dot(&state.velocity_km_s);
        let denominator = self.c_km_s * self.c_km_s * r3;
        let factor = self.mu_km3_s2 / denominator;
        let bracket = state.position_km * (4.0 * self.mu_km3_s2 / r - v2)
            + state.velocity_km_s * (4.0 * r_dot_v);

        let acceleration = bracket * factor;
        if denominator.is_finite() && acceleration.iter().all(|value| value.is_finite()) {
            return Ok(acceleration);
        }
        let (position, velocity, mu, c) = self.scaled_inputs(state);
        let r2 = wide_norm_squared(position);
        let r = r2.sqrt();
        let v2 = wide_norm_squared(velocity);
        let rv = wide_dot(position, velocity);
        let four = Wide::from(4.0);
        let factor = mu / (c * c * (r2 * r));
        finite_acceleration(Vector3::from(core::array::from_fn(|axis| {
            ((position[axis] * (four * mu / r - v2) + velocity[axis] * (four * rv)) * factor)
                .value()
        })))
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        expect_parameter_count(parameters, 0)?;
        self.validate(state)?;
        if self.c_km_s == f64::INFINITY {
            return Ok(AccelerationPartials::zeros());
        }
        let position = state.position_km;
        let velocity = state.velocity_km_s;
        let r2 = position.norm_squared();
        // a = k [4 mu r / r⁴ - v² r / r³ + 4 (r . v) v / r³], k = mu / c².
        let r = r2.sqrt();
        let r3 = r2 * r;
        let r4 = r2 * r2;
        let r5 = r3 * r2;
        let r6 = r3 * r3;
        let v2 = velocity.norm_squared();
        let r_dot_v = position.dot(&velocity);
        let c2 = self.c_km_s * self.c_km_s;
        let k = self.mu_km3_s2 / c2;
        let identity = Matrix3::identity();
        let rr = outer(&position, &position);
        let wrt_position = (identity / r4 - rr * (4.0 / r6)) * (4.0 * self.mu_km3_s2)
            - (identity / r3 - rr * (3.0 / r5)) * v2
            + outer(&velocity, &velocity) * (4.0 / r3)
            - outer(&velocity, &position) * (12.0 * r_dot_v / r5);
        let wrt_velocity = (outer(&velocity, &position) * 4.0 - outer(&position, &velocity) * 2.0
            + identity * (4.0 * r_dot_v))
            / r3;
        let partials = AccelerationPartials {
            wrt_position: wrt_position * k,
            wrt_velocity: wrt_velocity * k,
        };
        if r6.is_finite() && c2.is_finite() {
            if let Ok(partials) = finite_partials(partials) {
                return Ok(partials);
            }
        }
        finite_partials(self.scaled_partials(state))
    }
}

impl SchwarzschildRelativity {
    fn validate(&self, state: &CartesianState) -> Result<(), PropagationError> {
        validate_finite_state(state)?;
        validate_finite([self.mu_km3_s2], "relativity constants")?;
        if self.c_km_s != f64::INFINITY {
            validate_finite([self.c_km_s], "relativity constants")?;
            if self.c_km_s <= 0.0 {
                return Err(PropagationError::InvalidInput(
                    "c_km_s must be positive".to_string(),
                ));
            }
        }
        if state.position_km.iter().all(|&value| value == 0.0) {
            return Err(PropagationError::NumericalFailure(
                "Zero position magnitude".to_string(),
            ));
        }
        Ok(())
    }

    fn scaled_inputs(&self, state: &CartesianState) -> ([Wide; 3], [Wide; 3], Wide, Wide) {
        (
            state.position_array().map(Wide::from),
            state.velocity_array().map(Wide::from),
            Wide::from(self.mu_km3_s2),
            Wide::from(self.c_km_s),
        )
    }

    /// Keep the powers of radius and speed available before combining the
    /// derivative terms, even when their binary64 intermediates overflow.
    fn scaled_partials(&self, state: &CartesianState) -> AccelerationPartials {
        let (position, velocity, mu, c) = self.scaled_inputs(state);
        let r2 = wide_norm_squared(position);
        let r3 = r2 * r2.sqrt();
        let r4 = r2 * r2;
        let r5 = r3 * r2;
        let r6 = r3 * r3;
        let v2 = wide_norm_squared(velocity);
        let rv = wide_dot(position, velocity);
        let k = mu / (c * c);
        let two = Wide::from(2.0);
        let three = Wide::from(3.0);
        let four = Wide::from(4.0);
        let twelve = Wide::from(12.0);
        AccelerationPartials {
            wrt_position: Matrix3::from_fn(|i, j| {
                let id = Wide::from(f64::from(i == j));
                let rr = position[i] * position[j];
                let vv = velocity[i] * velocity[j];
                let vr = velocity[i] * position[j];
                (((id / r4 - rr * (four / r6)) * (four * mu) - (id / r3 - rr * (three / r5)) * v2
                    + vv * (four / r3)
                    - vr * (twelve * rv / r5))
                    * k)
                    .value()
            }),
            wrt_velocity: Matrix3::from_fn(|i, j| {
                let id = Wide::from(f64::from(i == j));
                let vr = velocity[i] * position[j];
                let rv_matrix = position[i] * velocity[j];
                (((vr * four - rv_matrix * two + id * (four * rv)) / r3) * k).value()
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    //! Clean-room validation anchors:
    //!
    //! For a circular GNSS-altitude orbit, `r . v = 0` and `v^2 = GM/r`, so
    //! Montenbruck & Gill equation 3.146 reduces to magnitude
    //! `3 * GM^2 / (c^2 * r^3)`. At `r = 26560 km`, this is
    //! `2.830552329290399e-10 m/s^2`.

    use super::*;

    #[test]
    fn gnss_circular_magnitude_matches_closed_form() {
        let r = 26_560.0;
        let v = (MU_EARTH / r).sqrt();
        let state = CartesianState::new(0.0, [r, 0.0, 0.0], [0.0, v, 0.0]);
        let accel = SchwarzschildRelativity::default()
            .acceleration(&state, &PropagationContext::default())
            .expect("relativistic acceleration");
        let expected = 2.830_552_329_290_399e-13;

        assert!((accel.x - expected).abs() < 1.0e-27);
        assert_eq!(accel.y, 0.0);
        assert_eq!(accel.z, 0.0);
        assert!((1.0e-10..1.0e-9).contains(&(accel.norm() * 1000.0)));
    }

    #[test]
    fn acceleration_vanishes_in_newtonian_limit() {
        let state = CartesianState::new(0.0, [26_560.0, 0.0, 0.0], [0.0, 3.874, 0.0]);
        let accel = SchwarzschildRelativity::new(MU_EARTH, f64::INFINITY)
            .acceleration(&state, &PropagationContext::default())
            .expect("relativistic acceleration");
        assert_eq!(accel, Vector3::zeros());
    }
}
