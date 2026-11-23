//! Cannonball solar radiation pressure.
//!
//! The acceleration follows Orekit 13.1.10's `SolarRadiationPressure` and
//! `IsotropicRadiationSingleCoefficient` evaluation order:
//! `k_ref = P_sr * d_ref^2`, `sun_sat = r_sat - r_sun` (in metres),
//! `r2 = |sun_sat|^2`, `raw_p = nu * k_ref / r2`,
//! `flux = (raw_p / sqrt(r2)) * sun_sat`,
//! `a_m = (1 * (A/m) * C_R / 1) * flux`, converted from m/s^2 to km/s^2 for
//! the propagation state.
//!
//! [`ForceModel::acceleration`], [`ForceModel::acceleration_with_parameters`],
//! [`ForceModel::acceleration_partials`] and
//! [`ForceModel::parameter_partials`] take the Sun and Moon positions from
//! the analytic series in GCRS
//! ([`sun_moon_gcrs_at_tdb_seconds`](crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds)),
//! the frame of the propagation state, with the series evaluated at the TT of
//! the state's TDB epoch and rotated to GCRS at TDB;
//! [`SolarRadiationPressure::acceleration_with_sun_moon`] takes caller
//! positions in the state's frame.
//!
//! `nu` is the unshadowed lighting ratio (`1 - shadow`). By default the shadow
//! is the Earth-only conical model
//! ([`shadow_fraction`](crate::astro::events::eclipse::shadow_fraction)).
//! [`SolarRadiationPressure::with_earth_shadow_model`] selects the WGS84
//! oblate Earth ([`shadow_fraction_with_model`]), and
//! [`SolarRadiationPressure::with_moon_eclipse`] replaces it with the combined
//! Earth and Moon apparent-disc occultation
//! ([`shadow_fraction_with_model_and_moon`]) under the selected Earth figure.
//!
//! `C_R` is the model's one estimable parameter
//! ([`SolarRadiationPressure::CR_PARAMETER`]). The acceleration is linear in
//! it, so its partial is the acceleration at `C_R = 1` under the same shadow
//! model. The position partials are computed by forward-mode automatic
//! differentiation in Hipparchus 4.0.3 `Gradient` arithmetic following Orekit
//! 13.1.10's `FieldSpacecraftState` `acceleration` and
//! `IsotropicRadiationSingleCoefficient` order. Where Orekit's evaluation
//! produces a `NaN` or an infinity in any acceleration value or partial (for
//! example, on the exact shadow axis in an annular eclipse, when a near-axis
//! cross-product norm underflows to zero, or when intermediates such as
//! `k_ref` or `factor` overflow even in umbra), the acceleration and position
//! partials return [`PropagationError::NumericalFailure`].
use crate::astro::constants::astro::{AU_KM, SOLAR_RADIATION_PRESSURE_N_M2};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::error::PropagationError;
use crate::astro::events::eclipse::{
    ad::{FieldVec3, Grad, Vec3},
    conical_shadow_field, earth_moon_lighting_ratio_double, earth_moon_lighting_ratio_field,
    fast_math, shadow_fraction_with_model, EarthShadowModel,
};
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_partials, validate_finite_state,
    AccelerationPartials, ForceModel, ForceParameter,
};
use crate::astro::forces::third_body::sun_moon_positions_km;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, RowVector3, Vector3};

/// Cannonball solar radiation pressure parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarRadiationPressure {
    cr: f64,
    area_to_mass_m2_kg: f64,
    pressure_n_m2: f64,
    au_km: f64,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
}

impl SolarRadiationPressure {
    /// Name of the estimable reflectivity coefficient `C_R` in
    /// [`ForceModel::parameters`].
    pub const CR_PARAMETER: &'static str = "srp.cr";

    /// Build from reflectivity coefficient `C_R` and area-to-mass `A/m` in m^2/kg.
    pub fn new(cr: f64, area_to_mass_m2_kg: f64) -> Result<Self, PropagationError> {
        Self::with_pressure(cr, area_to_mass_m2_kg, SOLAR_RADIATION_PRESSURE_N_M2, AU_KM)
    }

    /// Build with explicit pressure and AU constants.
    pub fn with_pressure(
        cr: f64,
        area_to_mass_m2_kg: f64,
        pressure_n_m2: f64,
        au_km: f64,
    ) -> Result<Self, PropagationError> {
        validate_positive("cr", cr)?;
        validate_positive("area_to_mass_m2_kg", area_to_mass_m2_kg)?;
        validate_positive("pressure_n_m2", pressure_n_m2)?;
        validate_positive("au_km", au_km)?;
        Ok(Self {
            cr,
            area_to_mass_m2_kg,
            pressure_n_m2,
            au_km,
            earth_shadow_model: EarthShadowModel::Spherical,
            include_moon_eclipse: false,
        })
    }

    /// Enable or disable lunar occultation in the shadow factor.
    ///
    /// Off by default, which keeps the Earth-only conical shadow. When
    /// enabled, every evaluation (the acceleration, the acceleration at
    /// estimated parameter values and both kinds of partials) uses the
    /// combined Earth and Moon apparent-disc occultation, with the Moon
    /// position from
    /// [`sun_moon_gcrs_at_tdb_seconds`](crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds).
    pub fn with_moon_eclipse(mut self, enabled: bool) -> Self {
        self.include_moon_eclipse = enabled;
        self
    }

    /// Whether lunar occultation is included in the shadow factor.
    pub fn moon_eclipse_enabled(&self) -> bool {
        self.include_moon_eclipse
    }

    /// Select the Earth figure model used for shadow evaluation, in the
    /// conical shadow and in the apparent-disc occultation alike.
    ///
    /// Defaults to [`EarthShadowModel::Spherical`].
    pub fn with_earth_shadow_model(mut self, model: EarthShadowModel) -> Self {
        self.earth_shadow_model = model;
        self
    }

    /// Earth figure model used for shadow evaluation.
    pub fn earth_shadow_model(&self) -> EarthShadowModel {
        self.earth_shadow_model
    }

    /// Reflectivity coefficient `C_R`.
    pub fn cr(&self) -> f64 {
        self.cr
    }

    /// Area-to-mass ratio `A/m`, m^2/kg.
    pub fn area_to_mass_m2_kg(&self) -> f64 {
        self.area_to_mass_m2_kg
    }

    /// Solar radiation pressure at 1 AU, N/m^2.
    pub fn pressure_n_m2(&self) -> f64 {
        self.pressure_n_m2
    }

    /// Astronomical unit used in the inverse-square scale, km.
    pub fn au_km(&self) -> f64 {
        self.au_km
    }

    /// Evaluate cannonball SRP acceleration (km/s^2) using explicit inertial
    /// Sun and Moon positions (km).
    ///
    /// When [`Self::moon_eclipse_enabled`] is `true`, `moon_pos_km` enters the
    /// combined Earth-and-Moon apparent-disc shadow function. When `false`,
    /// `moon_pos_km` is still validated for finiteness while the Earth-only
    /// shadow function is evaluated.
    pub fn acceleration_with_sun_moon(
        &self,
        sat_pos_km: Vector3<f64>,
        sun_pos_km: Vector3<f64>,
        moon_pos_km: Vector3<f64>,
    ) -> Result<Vector3<f64>, PropagationError> {
        validate_vector("sat_pos_km", sat_pos_km)?;
        validate_vector("sun_pos_km", sun_pos_km)?;
        validate_vector("moon_pos_km", moon_pos_km)?;
        srp_acceleration_with_bodies(sat_pos_km, sun_pos_km, Some(moon_pos_km), *self)
    }

    fn with_estimated_cr(&self, parameters: &[f64]) -> Result<Self, PropagationError> {
        expect_parameter_count(parameters, 1)?;
        let cr = parameters[0];
        if !cr.is_finite() {
            return Err(PropagationError::InvalidInput(
                "cr must be finite".to_string(),
            ));
        }
        Ok(Self { cr, ..*self })
    }

    /// The Sun position at a TDB epoch, and the Moon's when the shadow
    /// function uses it.
    fn bodies_km(
        &self,
        epoch_tdb_seconds: f64,
    ) -> Result<(Vector3<f64>, Option<Vector3<f64>>), PropagationError> {
        let bodies = sun_moon_positions_km(epoch_tdb_seconds)?;
        let moon = self.include_moon_eclipse.then_some(bodies.moon_km);
        Ok((bodies.sun_km, moon))
    }
}

impl ForceModel for SolarRadiationPressure {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let (sun_pos_km, moon_pos_km) = self.bodies_km(state.epoch_tdb_seconds)?;
        srp_acceleration_with_bodies(state.position_km, sun_pos_km, moon_pos_km, *self)
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        vec![ForceParameter::new(Self::CR_PARAMETER, self.cr)]
    }

    fn parameter_count(&self) -> usize {
        1
    }

    /// The acceleration with `C_R` replaced by `parameters[0]`. An estimated
    /// `C_R` is a free scale: any finite value is accepted here, also zero or
    /// a negative one, which [`SolarRadiationPressure::new`] refuses.
    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let model = self.with_estimated_cr(parameters)?;
        let (sun_pos_km, moon_pos_km) = model.bodies_km(state.epoch_tdb_seconds)?;
        srp_acceleration_with_bodies(state.position_km, sun_pos_km, moon_pos_km, model)
    }

    /// The position partials via forward-mode automatic differentiation in
    /// Hipparchus `Gradient` arithmetic following Orekit's field `acceleration`.
    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        let model = self.with_estimated_cr(parameters)?;
        validate_finite_state(state)?;
        let (sun_pos_km, moon_pos_km) = model.bodies_km(state.epoch_tdb_seconds)?;
        srp_position_partials_with_bodies(state.position_km, sun_pos_km, moon_pos_km, model)
            .map(AccelerationPartials::position_only)
            .and_then(finite_partials)
    }

    fn parameter_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        self.with_estimated_cr(parameters)?;
        let (sun_pos_km, moon_pos_km) = self.bodies_km(state.epoch_tdb_seconds)?;
        let per_unit_cr = Self { cr: 1.0, ..*self };
        Ok(vec![srp_acceleration_with_bodies(
            state.position_km,
            sun_pos_km,
            moon_pos_km,
            per_unit_cr,
        )?])
    }
}

/// Refuse a satellite at the Sun's position, a Sun at the geocenter, and
/// finite positions whose distance from each other overflows, as numerical
/// failures. Non-finite positions are left to the shadow function, which
/// refuses them.
fn validate_sun_position(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
) -> Result<(), PropagationError> {
    let sat_to_sun = sun_pos_km - sat_pos_km;
    if sat_to_sun.norm_squared() == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Satellite coincident with Sun".to_string(),
        ));
    }
    if sun_pos_km.norm_squared() == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Zero Sun position magnitude".to_string(),
        ));
    }
    let finite_positions = sat_pos_km
        .iter()
        .chain(sun_pos_km.iter())
        .all(|value| value.is_finite());
    if finite_positions && !sat_to_sun.norm().is_finite() {
        return Err(distance_overflow());
    }
    Ok(())
}

fn distance_overflow() -> PropagationError {
    PropagationError::NumericalFailure("satellite-Sun distance overflows".to_string())
}

fn as_array(vector: Vector3<f64>) -> [f64; 3] {
    [vector.x, vector.y, vector.z]
}

fn missing_moon() -> PropagationError {
    PropagationError::ForceModelFailure("eclipse: missing Moon position".to_string())
}

fn eclipse_failure(error: impl std::fmt::Display) -> PropagationError {
    PropagationError::ForceModelFailure(format!("eclipse: {error}"))
}

/// Lighting ratio `nu = 1 - shadow` in `double`, without round-tripping through
/// `1 - shadow` when Orekit's apparent-disc model is active.
fn lighting_ratio_of(
    model: SolarRadiationPressure,
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
) -> Result<f64, PropagationError> {
    let (sat, sun) = (as_array(sat_pos_km), as_array(sun_pos_km));
    if model.include_moon_eclipse {
        let moon = moon_pos_km.ok_or_else(missing_moon)?;
        earth_moon_lighting_ratio_double(sat, sun, as_array(moon), model.earth_shadow_model)
            .map_err(eclipse_failure)
    } else {
        shadow_fraction_with_model(sat, sun, model.earth_shadow_model)
            .map(|shadow| 1.0 - shadow)
            .map_err(eclipse_failure)
    }
}

/// Forward-mode lighting ratio `nu` with partials in `1/m` (`d(nu)/d(pos_m)`).
fn lighting_ratio_field_of(
    model: SolarRadiationPressure,
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
) -> Result<Grad, PropagationError> {
    let (sat, sun) = (as_array(sat_pos_km), as_array(sun_pos_km));
    let ratio = if model.include_moon_eclipse {
        let moon = moon_pos_km.ok_or_else(missing_moon)?;
        earth_moon_lighting_ratio_field(sat, sun, as_array(moon), model.earth_shadow_model)
            .map_err(eclipse_failure)?
    } else {
        let shadow_km =
            conical_shadow_field(sat, sun, model.earth_shadow_model).map_err(eclipse_failure)?;
        Grad {
            value: 1.0 - shadow_km.value,
            grad: [
                -shadow_km.grad[0] / M_PER_KM,
                -shadow_km.grad[1] / M_PER_KM,
                -shadow_km.grad[2] / M_PER_KM,
            ],
        }
    };
    if !ratio.is_all_finite() {
        return Err(PropagationError::NumericalFailure(
            "eclipse: shadow fraction gradient is not finite".to_string(),
        ));
    }
    Ok(ratio)
}

/// Evaluate Orekit `SolarRadiationPressure.acceleration` +
/// `IsotropicRadiationSingleCoefficient.radiationPressureAcceleration` in m/s^2.
fn srp_acceleration_m_s2(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio: f64,
    model: SolarRadiationPressure,
) -> Result<Vec3, PropagationError> {
    let d_ref = model.au_km * M_PER_KM;
    let k_ref = model.pressure_n_m2 * d_ref * d_ref;
    let position = Vec3::new(
        sat_pos_km.x * M_PER_KM,
        sat_pos_km.y * M_PER_KM,
        sat_pos_km.z * M_PER_KM,
    );
    let sun_position = Vec3::new(
        sun_pos_km.x * M_PER_KM,
        sun_pos_km.y * M_PER_KM,
        sun_pos_km.z * M_PER_KM,
    );
    let sun_sat_vector = position.sub(sun_position);
    let r2 = sun_sat_vector.norm_sq();
    if !r2.is_finite() {
        return Err(distance_overflow());
    }
    if r2 == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Satellite coincident with Sun".to_string(),
        ));
    }
    let factor = 1.0 * model.area_to_mass_m2_kg * model.cr / 1.0;
    let raw_p = ratio * k_ref / r2;
    let flux = Vec3::from_scaled(raw_p / fast_math::sqrt(r2), sun_sat_vector);
    Ok(Vec3::from_scaled(factor, flux))
}

fn srp_acceleration_with_bodies(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    validate_sun_position(sat_pos_km, sun_pos_km)?;
    let ratio = lighting_ratio_of(model, sat_pos_km, sun_pos_km, moon_pos_km)?;
    let accel_m = srp_acceleration_m_s2(sat_pos_km, sun_pos_km, ratio, model)?;
    finite_acceleration(Vector3::new(
        accel_m.x / M_PER_KM,
        accel_m.y / M_PER_KM,
        accel_m.z / M_PER_KM,
    ))
}

/// Evaluate Orekit `Field<Gradient>` `SolarRadiationPressure.acceleration` +
/// `IsotropicRadiationSingleCoefficient.radiationPressureAcceleration` in m/s^2,
/// with partials in `1/s^2`.
fn srp_acceleration_field_m_s2(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio: Grad,
    model: SolarRadiationPressure,
) -> Result<FieldVec3, PropagationError> {
    let d_ref = model.au_km * M_PER_KM;
    let k_ref = model.pressure_n_m2 * d_ref * d_ref;
    let position = FieldVec3::new(
        Grad::variable(0, sat_pos_km.x * M_PER_KM),
        Grad::variable(1, sat_pos_km.y * M_PER_KM),
        Grad::variable(2, sat_pos_km.z * M_PER_KM),
    );
    let sun_position = FieldVec3::constant(Vec3::new(
        sun_pos_km.x * M_PER_KM,
        sun_pos_km.y * M_PER_KM,
        sun_pos_km.z * M_PER_KM,
    ));
    let sun_sat_vector = position.sub(sun_position);
    let r2 = sun_sat_vector.norm_sq();
    if !r2.value.is_finite() {
        return Err(distance_overflow());
    }
    if r2.value == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Satellite coincident with Sun".to_string(),
        ));
    }
    let factor = Grad::ONE
        .reciprocal()
        .mul(Grad::ONE)
        .mul_f64(model.area_to_mass_m2_kg)
        .mul(Grad::constant(model.cr));
    let raw_p = ratio.mul_f64(k_ref).div(r2);
    let flux = FieldVec3::from_scaled(raw_p.div(r2.sqrt()), sun_sat_vector);
    Ok(FieldVec3::from_scaled(factor, flux))
}

/// Position partials of [`srp_acceleration_with_bodies`], in 1/s^2.
fn srp_position_partials_with_bodies(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    model: SolarRadiationPressure,
) -> Result<Matrix3<f64>, PropagationError> {
    validate_sun_position(sat_pos_km, sun_pos_km)?;
    let ratio = lighting_ratio_field_of(model, sat_pos_km, sun_pos_km, moon_pos_km)?;
    let accel = srp_acceleration_field_m_s2(sat_pos_km, sun_pos_km, ratio, model)?;
    if !(accel.x.is_all_finite() && accel.y.is_all_finite() && accel.z.is_all_finite()) {
        return Err(PropagationError::NumericalFailure(
            "SRP acceleration or position partial is not finite".to_string(),
        ));
    }
    Ok(Matrix3::from_rows(&[
        RowVector3::from(accel.x.grad),
        RowVector3::from(accel.y.grad),
        RowVector3::from(accel.z.grad),
    ]))
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), PropagationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be finite and positive"
        )));
    }
    Ok(())
}

fn validate_vector(field: &'static str, vector: Vector3<f64>) -> Result<(), PropagationError> {
    if !vector.iter().all(|v| v.is_finite()) {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be finite"
        )));
    }
    Ok(())
}

#[cfg(test)]
#[path = "srp_tests.rs"]
mod tests;
