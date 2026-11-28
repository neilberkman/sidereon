use super::scaled::{wide_dot, Wide};
use crate::astro::constants::{J2_EARTH, MU_EARTH, RE_EARTH};
use crate::astro::error::PropagationError;
use crate::astro::events::eclipse::{
    ad::{FieldVec3, Grad, ScaledF64, ScaledGrad, Vec3},
    fast_math,
};
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_partials, validate_finite,
    validate_position, AccelerationPartials, ForceModel,
};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};

/// Degree-2 zonal-harmonic perturbation model for Cartesian propagation.
///
/// The [`crate::astro::forces::ForceModel`] implementation uses `state.position_km`
/// to return the perturbation acceleration in km/s². It evaluates the
/// acceleration and its position partials in the order of Orekit 13.1.10's
/// full `J2OnlyPerturbation.acceleration` evaluations, using `Gradient` for
/// the partials and including the same-frame static transforms.
/// The J2 axis is the z axis of the propagation frame. For
/// [`crate::astro::propagator::ForceModelKind::TwoBodyJ2`], the equivalent
/// Orekit configuration uses GCRF for both the propagation and J2 frames.
/// A zero position returns [`PropagationError::NumericalFailure`].
pub struct J2Gravity {
    /// Gravitational parameter in km³/s². `Default::default` sets this to
    /// [`crate::astro::constants::MU_EARTH`], `ForceModelKind::build` copies
    /// `TwoBodyJ2::mu_km3_s2` here.
    pub mu: f64,
    /// Gravity-model reference equatorial radius in km. `Default::default` sets
    /// this to [`crate::astro::constants::RE_EARTH`], `ForceModelKind::build`
    /// copies `TwoBodyJ2::re_km` here, and `acceleration` uses its squared
    /// ratio to the position magnitude.
    pub re: f64,
    /// Dimensionless degree-2 zonal harmonic coefficient. `Default::default`
    /// sets this to [`crate::astro::constants::J2_EARTH`], `ForceModelKind::build`
    /// copies `TwoBodyJ2::j2` here, and `acceleration` uses it to scale the
    /// perturbation. [`crate::astro::constants::J2_EARTH`] is EGM96's
    /// tide-free value; a solid Earth tide force added beside this model has
    /// to be told the tide system of `j2` through
    /// [`crate::astro::forces::SolidEarthTideGravity::tide_system`].
    pub j2: f64,
}

impl Default for J2Gravity {
    fn default() -> Self {
        Self {
            mu: MU_EARTH,
            re: RE_EARTH,
            j2: J2_EARTH,
        }
    }
}

impl ForceModel for J2Gravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.validate(state)?;
        let [x, y, z] = state.position_array();
        let r2 = Vec3::new(x, y, z).norm_sq();
        let ratio_times_five = 5.0 * (z * z) / r2;
        let equatorial = ratio_times_five - 1.0;
        let axial = ratio_times_five - 3.0;
        let components = Vec3::new(x * equatorial, y * equatorial, z * axial);
        let re2 = self.re * self.re;
        let radii_ratio = re2 / r2;
        let denominator = 2.0 * (r2 * fast_math::sqrt(r2));
        let j2_mu = (3.0 * self.j2) * self.mu;
        let numerator = j2_mu * radii_ratio;
        let factor = numerator / denominator;
        let acceleration = components.scalar_mul(factor);
        if denominator.is_finite()
            && [acceleration.x, acceleration.y, acceleration.z]
                .into_iter()
                .all(f64::is_finite)
        {
            return Ok(Vector3::new(acceleration.x, acceleration.y, acceleration.z));
        }

        let position = [x, y, z].map(Wide::from);
        let [x, y, z] = position;
        let r2 = wide_dot(position, position);
        let ratio_times_five = Wide::from(5.0) * (z * z) / r2;
        let equatorial = ratio_times_five - Wide::from(1.0);
        let components = [
            x * equatorial,
            y * equatorial,
            z * (ratio_times_five - Wide::from(3.0)),
        ];
        let re = Wide::from(self.re);
        let radii_ratio = (re * re) / r2;
        let cubed_radius = r2 * r2.sqrt();
        let factor = Wide::from(3.0) * Wide::from(self.j2) * Wide::from(self.mu) * radii_ratio
            / (Wide::from(2.0) * cubed_radius);
        finite_acceleration(Vector3::from(
            components.map(|component| (factor * component).value()),
        ))
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        expect_parameter_count(parameters, 0)?;
        self.validate(state)?;
        let [x, y, z] = state.position_array();
        let position = FieldVec3::new(
            Grad::variable(0, x),
            Grad::variable(1, y),
            Grad::variable(2, z),
        );
        let translated = FieldVec3::new(Grad::ZERO, Grad::ZERO, Grad::ZERO).add(position);
        let [x, y, z] = rotate_identity([translated.x, translated.y, translated.z], Grad::ONE);
        let r2 = FieldVec3::new(x, y, z).norm_sq();
        let ratio_times_five = z.square().mul_f64(5.0).div(r2);
        let equatorial = ratio_times_five.sub_f64(1.0);
        let components = FieldVec3::new(
            x.mul(equatorial),
            y.mul(equatorial),
            z.mul(ratio_times_five.sub_f64(3.0)),
        );
        let re2 = self.re * self.re;
        let radii_ratio = r2.reciprocal().mul_f64(re2);
        let radius = r2.sqrt();
        let denominator = r2.mul(radius).mul_f64(2.0);
        let j2_mu = Grad::constant(self.j2).mul_f64(self.mu).mul_f64(3.0);
        let numerator = j2_mu.mul(radii_ratio);
        let factor = numerator.div(denominator);
        let acceleration = components.scalar_mul(factor);
        let acceleration = rotate_identity(
            [acceleration.x, acceleration.y, acceleration.z],
            Grad::ONE.negate(),
        );
        if gradient_divisor_is_finite(r2)
            && gradient_divisor_is_finite(denominator)
            && acceleration.into_iter().all(Grad::is_all_finite)
        {
            return Ok(AccelerationPartials::position_only(Matrix3::from_rows(
                &acceleration.map(|component| component.grad.into()),
            )));
        }
        finite_partials(AccelerationPartials::position_only(
            self.scaled_gradient(position),
        ))
    }
}

impl J2Gravity {
    fn validate(&self, state: &CartesianState) -> Result<(), PropagationError> {
        validate_position(state)?;
        validate_finite([self.mu, self.re, self.j2], "J2 constants")?;
        if state.position_km.iter().all(|&value| value == 0.0) {
            return Err(PropagationError::NumericalFailure(
                "Zero position magnitude".to_string(),
            ));
        }
        Ok(())
    }

    fn scaled_gradient(&self, position: FieldVec3) -> Matrix3<f64> {
        let [x, y, z] = [position.x, position.y, position.z]
            .map(|component| ScaledGrad::ZERO.add(ScaledGrad::from_grad(component)));
        let r2 = x.square().add(y.square()).add(z.square());
        let ratio_times_five = z.square().mul_scalar(ScaledF64::from_f64(5.0)).div(r2);
        let equatorial = ratio_times_five.sub_scalar(1.0);
        let components = [
            x.mul_ordered(equatorial),
            y.mul_ordered(equatorial),
            z.mul_ordered(ratio_times_five.sub_scalar(3.0)),
        ];
        let re = ScaledF64::from_f64(self.re);
        let radii_ratio = r2.reciprocal().mul_scalar(re.mul(re));
        let cubed_radius = r2.mul_ordered(r2.sqrt());
        let three = ScaledF64::from_f64(3.0);
        let two = ScaledF64::from_f64(2.0);
        let factor = ScaledGrad::from_grad(Grad::constant(self.j2))
            .mul_scalar(ScaledF64::from_f64(self.mu))
            .mul_scalar(three)
            .mul_ordered(radii_ratio)
            .div(cubed_radius.mul_scalar(two));
        Matrix3::from_rows(&components.map(|component| {
            ScaledGrad::ZERO
                .add(component.mul_ordered(factor))
                .grad
                .map(ScaledF64::to_f64)
                .into()
        }))
    }
}

/// Gradient division and reciprocal both square the inverse.
fn gradient_divisor_is_finite(divisor: Grad) -> bool {
    let inverse = 1.0 / divisor.value;
    divisor.is_all_finite() && (inverse * inverse).is_finite()
}

/// Hipparchus's field quaternion rotation with q1 = q2 = q3 = +0.
/// For finite lanes, the zero quaternion products only affect signs of zeros.
/// The final double and subtract reproduce the original nonzero lanes exactly
/// when the double is finite, and return +0 for zero lanes. In scaled
/// arithmetic, adding +0 has the same effect without a binary64 range limit.
fn rotate_identity([x, y, z]: [Grad; 3], q0: Grad) -> [Grad; 3] {
    let q1_x = Grad::ZERO.mul(x);
    let q2_y = Grad::ZERO.mul(y);
    let q3_z = Grad::ZERO.mul(z);
    let s = q1_x.add(q2_y).add(q3_z);
    let rotate = |component: Grad, cross: Grad| {
        q0.mul(component.mul(q0).sub(cross))
            .add(s.mul(Grad::ZERO))
            .mul_f64(2.0)
            .sub(component)
    };
    [
        rotate(x, q3_z.sub(q2_y)),
        rotate(y, q1_x.sub(q3_z)),
        rotate(z, q2_y.sub(q1_x)),
    ]
}
