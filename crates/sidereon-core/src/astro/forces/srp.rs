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
//! the context's [DE ephemeris](PropagationContext::de_ephemeris) or the
//! default analytic series in GCRS
//! ([`sun_moon_gcrs_at_tdb_seconds`](crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds)),
//! the frame of the propagation state, with the series evaluated at the TT of
//! the state's TDB epoch and rotated to GCRS at TDB;
//! [`SolarRadiationPressure::acceleration_with_sun_moon`] takes caller
//! positions in the state's frame.
//!
//! `nu` is the unshadowed lighting ratio (`1 - shadow`). By default `nu` is
//! Orekit 13.1.10's apparent-disc lighting ratio
//! ([`ShadowFunction::ApparentDisc`]) with the WGS84 oblate Earth
//! ([`EarthShadowModel::Wgs84Oblate`]) and the Moon as a second occulting body
//! (see [`crate::astro::events::eclipse`] for the documented departures from
//! Orekit). [`SolarRadiationPressure::with_shadow_function`] selects between
//! [`ShadowFunction::ApparentDisc`] and the Earth-only linear conical lighting
//! ratio ([`ShadowFunction::Conical`], evaluated directly as
//! `(rho - r_umbra) / (r_penumbra - r_umbra)` in penumbra and
//! `rho / r_penumbra` in antumbra from the shared conical geometry rather than
//! `1 - shadow_fraction_with_model`),
//! [`SolarRadiationPressure::with_earth_shadow_model`] selects
//! [`EarthShadowModel::Wgs84Oblate`] or [`EarthShadowModel::Spherical`], and
//! [`SolarRadiationPressure::with_moon_eclipse`] enables or disables lunar
//! occultation.
//!
//! `C_R` is the model's one estimable parameter
//! ([`SolarRadiationPressure::CR_PARAMETER`]). Under
//! [`ShadowFunction::ApparentDisc`], both the position partials and the `C_R`
//! parameter partial are computed by forward-mode automatic differentiation in
//! Hipparchus 4.0.3 `Gradient` arithmetic following Orekit 13.1.10's
//! `FieldSpacecraftState` `acceleration` and
//! `IsotropicRadiationSingleCoefficient` order; for the `C_R` partial, when the
//! field lighting ratio fails on an exact-axis annular eclipse where the double
//! lighting ratio is defined, the double lighting ratio value is used. Under
//! [`ShadowFunction::Conical`] (which has no Orekit counterpart), full sunlight
//! (`nu == 1` with zero gradient) keeps the Orekit path and its bits; outside
//! full sunlight (umbra, penumbra, and antumbra), acceleration, position
//! partials, and `C_R` parameter partials are evaluated in exponent-separated
//! arithmetic throughout and rounded once at the output, so the double
//! acceleration equals the field value part bit for bit, tiny lighting ratios
//! and gradients never underflow into binary64 subnormals or flush to zero, and
//! umbra returns exact zero (with the exact antumbra axis returning zero
//! acceleration and `C_R` partial and [`PropagationError::NumericalFailure`]
//! from `acceleration_partials`, where only the lighting gradient is
//! undefined).
//!
//! The extended-exponent SRP evaluation uses the original km inputs and
//! lighting gradients in `1/km`. It also handles a squared distance that
//! rounds to zero for distinct positions. Umbra with a zero lighting
//! gradient returns exact zero.
//!
//! Under both shadow functions, `acceleration` returns
//! [`PropagationError::NumericalFailure`] when its own km/s^2 output overflows
//! (or when the satellite coincides with the Sun or the Sun position is zero),
//! whereas `acceleration_partials` and `parameter_partials` return `Ok`
//! whenever every requested partial entry is finite and defined, even if the
//! acceleration value itself overflows binary64, returning
//! [`PropagationError::NumericalFailure`] only when a requested partial
//! overflows or is undefined (such as position partials on the exact shadow
//! axis in an annular eclipse, when a near-axis apparent-disc cross-product
//! norm underflows to zero, or on the exact conical antumbra axis).
use crate::astro::constants::astro::{AU_KM, SOLAR_RADIATION_PRESSURE_N_M2};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::error::PropagationError;
use crate::astro::events::eclipse::{
    ad::{FieldVec3, Grad, ScaledF64, ScaledFieldVec3, ScaledGrad, ScaledVec3, Vec3},
    conical_lighting_ratio_double_scaled, conical_lighting_ratio_field_scaled,
    earth_lighting_ratio_double, earth_lighting_ratio_field, earth_moon_lighting_ratio_double,
    earth_moon_lighting_ratio_field, fast_math, EarthShadowModel, ShadowFunction,
};
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_partials, validate_position,
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
    shadow_function: ShadowFunction,
    earth_shadow_model: EarthShadowModel,
    include_moon_eclipse: bool,
}

impl SolarRadiationPressure {
    /// Name of the estimable reflectivity coefficient `C_R` in
    /// [`ForceModel::parameters`].
    pub const CR_PARAMETER: &'static str = "srp.cr";

    /// Build from reflectivity coefficient `C_R` and area-to-mass `A/m` in m^2/kg.
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`],
    /// [`EarthShadowModel::Wgs84Oblate`], and lunar occultation enabled.
    pub fn new(cr: f64, area_to_mass_m2_kg: f64) -> Result<Self, PropagationError> {
        Self::with_pressure(cr, area_to_mass_m2_kg, SOLAR_RADIATION_PRESSURE_N_M2, AU_KM)
    }

    /// Build with explicit pressure and AU constants.
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`],
    /// [`EarthShadowModel::Wgs84Oblate`], and lunar occultation enabled.
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
            shadow_function: ShadowFunction::ApparentDisc,
            earth_shadow_model: EarthShadowModel::Wgs84Oblate,
            include_moon_eclipse: true,
        })
    }

    /// Select the shadow function used to compute the solar lighting ratio `nu`.
    ///
    /// Defaults to [`ShadowFunction::ApparentDisc`]. Selecting
    /// [`ShadowFunction::Conical`] switches to the Earth-only linear conical
    /// lighting ratio (evaluated directly as
    /// `(rho - r_umbra) / (r_penumbra - r_umbra)` in penumbra and
    /// `rho / r_penumbra` in antumbra from the shared conical geometry) and
    /// disables lunar occultation ([`Self::moon_eclipse_enabled`] becomes
    /// `false`). Selecting [`ShadowFunction::ApparentDisc`] keeps the current
    /// [`Self::moon_eclipse_enabled`] setting.
    pub fn with_shadow_function(mut self, function: ShadowFunction) -> Self {
        self.shadow_function = function;
        if matches!(function, ShadowFunction::Conical) {
            self.include_moon_eclipse = false;
        }
        self
    }

    /// Shadow function used to compute the solar lighting ratio `nu`.
    pub fn shadow_function(&self) -> ShadowFunction {
        self.shadow_function
    }

    /// Enable or disable lunar occultation in the shadow factor.
    ///
    /// Enabled by default under [`ShadowFunction::ApparentDisc`], where every
    /// evaluation (the acceleration, the acceleration at estimated parameter
    /// values and both kinds of partials) uses the combined Earth and Moon
    /// apparent-disc occultation, using the Moon position at the state epoch.
    /// Setting `false` disables lunar occultation while keeping the selected
    /// [`Self::shadow_function`] (so on the default model it evaluates
    /// Earth-only apparent-disc occultation). Setting `true` enables lunar
    /// occultation and, if [`ShadowFunction::Conical`] was selected, switches
    /// to [`ShadowFunction::ApparentDisc`].
    pub fn with_moon_eclipse(mut self, enabled: bool) -> Self {
        self.include_moon_eclipse = enabled;
        if enabled {
            self.shadow_function = ShadowFunction::ApparentDisc;
        }
        self
    }

    /// Whether lunar occultation is included in the shadow factor.
    pub fn moon_eclipse_enabled(&self) -> bool {
        self.include_moon_eclipse
    }

    /// Select the Earth figure model used for shadow evaluation, in the
    /// apparent-disc occultation and in the conical shadow alike.
    ///
    /// Defaults to [`EarthShadowModel::Wgs84Oblate`].
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
    /// selected shadow function. Earth-only shadow functions ignore `moon_pos_km`.
    pub fn acceleration_with_sun_moon(
        &self,
        sat_pos_km: Vector3<f64>,
        sun_pos_km: Vector3<f64>,
        moon_pos_km: Vector3<f64>,
    ) -> Result<Vector3<f64>, PropagationError> {
        validate_vector("sat_pos_km", sat_pos_km)?;
        validate_vector("sun_pos_km", sun_pos_km)?;
        let moon = if self.include_moon_eclipse {
            validate_vector("moon_pos_km", moon_pos_km)?;
            Some(moon_pos_km)
        } else {
            None
        };
        srp_acceleration_with_bodies(sat_pos_km, sun_pos_km, moon, *self)
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
        ctx: &PropagationContext,
    ) -> Result<(Vector3<f64>, Option<Vector3<f64>>), PropagationError> {
        let bodies = sun_moon_positions_km(epoch_tdb_seconds, ctx)?;
        let moon = self.include_moon_eclipse.then_some(bodies.moon_km);
        Ok((bodies.sun_km, moon))
    }
}

impl ForceModel for SolarRadiationPressure {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        validate_position(state)?;
        let (sun_pos_km, moon_pos_km) = self.bodies_km(state.epoch_tdb_seconds, ctx)?;
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
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let model = self.with_estimated_cr(parameters)?;
        validate_position(state)?;
        let (sun_pos_km, moon_pos_km) = model.bodies_km(state.epoch_tdb_seconds, ctx)?;
        srp_acceleration_with_bodies(state.position_km, sun_pos_km, moon_pos_km, model)
    }

    /// The position partials via forward-mode automatic differentiation in
    /// Hipparchus `Gradient` arithmetic following Orekit's field `acceleration`.
    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        let model = self.with_estimated_cr(parameters)?;
        validate_position(state)?;
        let (sun_pos_km, moon_pos_km) = model.bodies_km(state.epoch_tdb_seconds, ctx)?;
        srp_position_partials_with_bodies(state.position_km, sun_pos_km, moon_pos_km, model)
            .map(AccelerationPartials::position_only)
            .and_then(finite_partials)
    }

    /// The `C_R` parameter partial via forward-mode automatic differentiation
    /// in Hipparchus `Gradient` arithmetic following Orekit's field
    /// `acceleration` at `C_R = parameters[0]`, falling back to
    /// exponent-separated arithmetic whenever Orekit's evaluation has a
    /// non-finite intermediate or output (returning `Ok` whenever the partial
    /// itself is finite, even if the acceleration value overflows binary64).
    fn parameter_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        let model = self.with_estimated_cr(parameters)?;
        validate_position(state)?;
        let (sun_pos_km, moon_pos_km) = model.bodies_km(state.epoch_tdb_seconds, ctx)?;
        Ok(vec![srp_parameter_partial_with_bodies(
            state.position_km,
            sun_pos_km,
            moon_pos_km,
            model,
        )?])
    }
}

/// Refuse a satellite at the Sun's position or a Sun at the geocenter as
/// numerical failures. Non-finite positions are left to the shadow function,
/// which refuses them. Overflowing satellite-Sun distance is not refused here
/// because exponent-separated evaluation handles it cleanly.
fn validate_sun_position(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
) -> Result<(), PropagationError> {
    let sat_to_sun = sun_pos_km - sat_pos_km;
    if sat_to_sun.x == 0.0 && sat_to_sun.y == 0.0 && sat_to_sun.z == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Satellite coincident with Sun".to_string(),
        ));
    }
    if sun_pos_km.x == 0.0 && sun_pos_km.y == 0.0 && sun_pos_km.z == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Zero Sun position magnitude".to_string(),
        ));
    }
    Ok(())
}

#[inline]
fn vec3_is_finite(v: Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

#[inline]
fn field_vec3_is_all_finite<const N: usize>(v: FieldVec3<N>) -> bool {
    v.x.is_all_finite() && v.y.is_all_finite() && v.z.is_all_finite()
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
    match (model.shadow_function, model.include_moon_eclipse) {
        (ShadowFunction::ApparentDisc, true) => {
            let moon = moon_pos_km.ok_or_else(missing_moon)?;
            earth_moon_lighting_ratio_double(sat, sun, as_array(moon), model.earth_shadow_model)
                .map_err(eclipse_failure)
        }
        (ShadowFunction::ApparentDisc, false) => {
            earth_lighting_ratio_double(sat, sun, model.earth_shadow_model).map_err(eclipse_failure)
        }
        (ShadowFunction::Conical, _) => {
            conical_lighting_ratio_double_scaled(sat, sun, model.earth_shadow_model)
                .map(|(nu, _)| nu.to_f64())
                .map_err(eclipse_failure)
        }
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
    let ratio = match (model.shadow_function, model.include_moon_eclipse) {
        (ShadowFunction::ApparentDisc, true) => {
            let moon = moon_pos_km.ok_or_else(missing_moon)?;
            earth_moon_lighting_ratio_field(sat, sun, as_array(moon), model.earth_shadow_model)
                .map_err(eclipse_failure)?
        }
        (ShadowFunction::ApparentDisc, false) => {
            earth_lighting_ratio_field(sat, sun, model.earth_shadow_model)
                .map_err(eclipse_failure)?
        }
        (ShadowFunction::Conical, _) => {
            let (ratio_km, _) =
                conical_lighting_ratio_field_scaled(sat, sun, model.earth_shadow_model)
                    .map_err(eclipse_failure)?;
            if !ratio_km.is_all_finite() {
                return Err(PropagationError::NumericalFailure(
                    "eclipse: shadow fraction gradient is not finite".to_string(),
                ));
            }
            let m_per_km = ScaledF64::from_f64(M_PER_KM);
            Grad {
                value: ratio_km.val.to_f64(),
                grad: [
                    ratio_km.grad[0].div(m_per_km).to_f64(),
                    ratio_km.grad[1].div(m_per_km).to_f64(),
                    ratio_km.grad[2].div(m_per_km).to_f64(),
                ],
            }
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
/// `IsotropicRadiationSingleCoefficient.radiationPressureAcceleration` in m/s^2,
/// returning `None` if any intermediate or output is non-finite or `r2 == 0`.
fn srp_acceleration_m_s2_fast(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio: f64,
    model: SolarRadiationPressure,
) -> Option<Vec3> {
    if !ratio.is_finite() {
        return None;
    }
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
    if !(d_ref.is_finite()
        && k_ref.is_finite()
        && vec3_is_finite(position)
        && vec3_is_finite(sun_position)
        && vec3_is_finite(sun_sat_vector)
        && r2 > 0.0
        && r2.is_finite())
    {
        return None;
    }
    let factor = 1.0 * model.area_to_mass_m2_kg * model.cr / 1.0;
    let ratio_k = ratio * k_ref;
    let raw_p = ratio_k / r2;
    let scale = raw_p / fast_math::sqrt(r2);
    let flux = Vec3::from_scaled(scale, sun_sat_vector);
    let accel_m = Vec3::from_scaled(factor, flux);
    if !(factor.is_finite()
        && ratio_k.is_finite()
        && raw_p.is_finite()
        && scale.is_finite()
        && vec3_is_finite(flux)
        && vec3_is_finite(accel_m))
    {
        return None;
    }
    Some(accel_m)
}

#[cfg(test)]
fn srp_acceleration_m_s2(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio: f64,
    model: SolarRadiationPressure,
) -> Result<Vec3, PropagationError> {
    srp_acceleration_m_s2_fast(sat_pos_km, sun_pos_km, ratio, model).ok_or_else(|| {
        PropagationError::NumericalFailure("SRP acceleration is not finite".to_string())
    })
}

/// Acceleration (km/s^2) evaluated entirely in exponent-separated arithmetic
/// from km inputs, rounding once at the output. Matches the value part of
/// [`srp_conical_position_field_scaled`] and [`srp_conical_cr_field_scaled`]
/// bit for bit.
fn srp_double_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    nu_s: ScaledF64,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    if !nu_s.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "SRP lighting ratio is not finite".to_string(),
        ));
    }
    if nu_s.is_zero() {
        return Ok(Vector3::zeros());
    }
    let d_ref_s = ScaledF64::from_f64(model.au_km);
    let k_ref_s = ScaledF64::from_f64(model.pressure_n_m2)
        .mul(d_ref_s)
        .mul(d_ref_s);
    let factor_s = ScaledF64::from_f64(model.area_to_mass_m2_kg).mul(ScaledF64::from_f64(model.cr));
    let sat_s = ScaledVec3::from_vec3(Vec3::new(sat_pos_km.x, sat_pos_km.y, sat_pos_km.z));
    let sun_s = ScaledVec3::from_vec3(Vec3::new(sun_pos_km.x, sun_pos_km.y, sun_pos_km.z));
    let (u_d, r_s) = sat_s.sub(sun_s).normalize_scaled().ok_or_else(|| {
        PropagationError::NumericalFailure("Satellite coincident with Sun".to_string())
    })?;
    let scale_s = factor_s.mul(k_ref_s).div(r_s.mul(r_s));
    let a1_s = u_d.scalar_mul(scale_s);
    let accel_km = a1_s
        .scalar_mul(nu_s)
        .scalar_div(ScaledF64::from_f64(M_PER_KM))
        .to_vec3()
        .ok_or_else(|| {
            PropagationError::NumericalFailure("SRP acceleration is not finite".to_string())
        })?;
    finite_acceleration(Vector3::new(accel_km.x, accel_km.y, accel_km.z))
}

fn srp_acceleration_orekit_or_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio: f64,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    if let Some(accel_m) = srp_acceleration_m_s2_fast(sat_pos_km, sun_pos_km, ratio, model) {
        let accel_km = Vector3::new(
            accel_m.x / M_PER_KM,
            accel_m.y / M_PER_KM,
            accel_m.z / M_PER_KM,
        );
        if accel_km.iter().all(|c| c.is_finite()) {
            return Ok(accel_km);
        }
    }
    srp_double_scaled(sat_pos_km, sun_pos_km, ScaledF64::from_f64(ratio), model)
}

fn srp_acceleration_with_bodies(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    validate_sun_position(sat_pos_km, sun_pos_km)?;
    if matches!(model.shadow_function, ShadowFunction::Conical) {
        let (nu_s, is_sunlit) = conical_lighting_ratio_double_scaled(
            as_array(sat_pos_km),
            as_array(sun_pos_km),
            model.earth_shadow_model,
        )
        .map_err(eclipse_failure)?;
        if is_sunlit {
            return srp_acceleration_orekit_or_scaled(sat_pos_km, sun_pos_km, 1.0, model);
        }
        return srp_double_scaled(sat_pos_km, sun_pos_km, nu_s, model);
    }
    let ratio = lighting_ratio_of(model, sat_pos_km, sun_pos_km, moon_pos_km)?;
    srp_acceleration_orekit_or_scaled(sat_pos_km, sun_pos_km, ratio, model)
}

/// Evaluate Orekit `Field<Gradient>` `SolarRadiationPressure.acceleration` +
/// `IsotropicRadiationSingleCoefficient.radiationPressureAcceleration` in m/s^2,
/// with partials in `1/s^2`, returning `None` if any intermediate or output
/// value or gradient entry is non-finite or `r2.value == 0`.
fn srp_acceleration_field_m_s2_fast(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio: Grad,
    model: SolarRadiationPressure,
) -> Option<FieldVec3> {
    if !ratio.is_all_finite() {
        return None;
    }
    let d_ref = model.au_km * M_PER_KM;
    let k_ref = model.pressure_n_m2 * d_ref * d_ref;
    let sat_m = Vec3::new(
        sat_pos_km.x * M_PER_KM,
        sat_pos_km.y * M_PER_KM,
        sat_pos_km.z * M_PER_KM,
    );
    let sun_m = Vec3::new(
        sun_pos_km.x * M_PER_KM,
        sun_pos_km.y * M_PER_KM,
        sun_pos_km.z * M_PER_KM,
    );
    if !(d_ref.is_finite() && k_ref.is_finite() && vec3_is_finite(sat_m) && vec3_is_finite(sun_m)) {
        return None;
    }
    let position = FieldVec3::new(
        Grad::variable(0, sat_m.x),
        Grad::variable(1, sat_m.y),
        Grad::variable(2, sat_m.z),
    );
    let sun_position = FieldVec3::constant(sun_m);
    let sun_sat_vector = position.sub(sun_position);
    let r2 = sun_sat_vector.norm_sq();
    if !(field_vec3_is_all_finite(sun_sat_vector) && r2.is_all_finite() && r2.value > 0.0) {
        return None;
    }
    let factor = Grad::ONE
        .reciprocal()
        .mul(Grad::ONE)
        .mul_f64(model.area_to_mass_m2_kg)
        .mul(Grad::constant(model.cr));
    if !factor.is_all_finite() {
        return None;
    }
    let ratio_k = ratio.mul_f64(k_ref);
    let raw_p = ratio_k.div(r2);
    let r = r2.sqrt();
    let scale = raw_p.div(r);
    if !(ratio_k.is_all_finite()
        && raw_p.is_all_finite()
        && r.is_all_finite()
        && scale.is_all_finite())
    {
        return None;
    }
    let flux = FieldVec3::from_scaled(scale, sun_sat_vector);
    if !field_vec3_is_all_finite(flux) {
        return None;
    }
    let accel = FieldVec3::from_scaled(factor, flux);
    field_vec3_is_all_finite(accel).then_some(accel)
}

#[cfg(test)]
fn srp_acceleration_field_m_s2(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio: Grad,
    model: SolarRadiationPressure,
) -> Result<FieldVec3, PropagationError> {
    srp_acceleration_field_m_s2_fast(sat_pos_km, sun_pos_km, ratio, model).ok_or_else(|| {
        PropagationError::NumericalFailure(
            "SRP acceleration or position partial is not finite".to_string(),
        )
    })
}

/// Acceleration (km/s^2) and position partials (1/s^2) evaluated entirely in
/// exponent-separated arithmetic from km inputs. The value part equals
/// [`srp_double_scaled`] bit for bit.
fn srp_conical_position_field_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio_km: ScaledGrad<3>,
    model: SolarRadiationPressure,
) -> Result<ScaledFieldVec3<3>, PropagationError> {
    let d_ref_s = ScaledF64::from_f64(model.au_km);
    let k_ref_s = ScaledF64::from_f64(model.pressure_n_m2)
        .mul(d_ref_s)
        .mul(d_ref_s);
    let factor_s = ScaledF64::from_f64(model.area_to_mass_m2_kg).mul(ScaledF64::from_f64(model.cr));
    let sat_s = ScaledVec3::from_vec3(Vec3::new(sat_pos_km.x, sat_pos_km.y, sat_pos_km.z));
    let sun_s = ScaledVec3::from_vec3(Vec3::new(sun_pos_km.x, sun_pos_km.y, sun_pos_km.z));
    let (u_d, r_s) = sat_s.sub(sun_s).normalize_scaled().ok_or_else(|| {
        PropagationError::NumericalFailure("Satellite coincident with Sun".to_string())
    })?;
    let scale_s = factor_s.mul(k_ref_s).div(r_s.mul(r_s));
    let coeff_s = scale_s.div(r_s);
    let u = [u_d.x, u_d.y, u_d.z];
    let one = ScaledF64::from_f64(1.0);
    let neg_three = ScaledF64::from_f64(-3.0);
    let a1_comp = |row: usize| -> ScaledGrad<3> {
        let val = u[row].mul(scale_s);
        let mut grad = [ScaledF64::ZERO; 3];
        for col in 0..3 {
            let term = if row == col {
                ScaledF64::lincomb2(one, one, neg_three, u[row].mul(u[col]))
            } else {
                neg_three.mul(u[row]).mul(u[col])
            };
            grad[col] = term.mul(coeff_s);
        }
        ScaledGrad { val, grad }
    };
    let a1_s = ScaledFieldVec3 {
        x: a1_comp(0),
        y: a1_comp(1),
        z: a1_comp(2),
    };
    Ok(a1_s
        .scalar_mul(ratio_km)
        .scalar_div(ScaledF64::from_f64(M_PER_KM)))
}

fn srp_position_partials_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio_km: ScaledGrad<3>,
    model: SolarRadiationPressure,
) -> Result<Matrix3<f64>, PropagationError> {
    if !ratio_km.is_all_finite() {
        return Err(PropagationError::NumericalFailure(
            "SRP lighting ratio gradient is not finite".to_string(),
        ));
    }
    if ratio_km.val.is_zero() && ratio_km.grad.iter().all(|g| g.is_zero()) {
        return Ok(Matrix3::zeros());
    }
    let accel_s = srp_conical_position_field_scaled(sat_pos_km, sun_pos_km, ratio_km, model)?;
    let comps = [accel_s.x, accel_s.y, accel_s.z];
    let mut mat = Matrix3::zeros();
    for row in 0..3 {
        for col in 0..3 {
            let entry = comps[row].grad[col].to_f64();
            if !entry.is_finite() {
                return Err(PropagationError::NumericalFailure(
                    "SRP position partial is not finite".to_string(),
                ));
            }
            mat[(row, col)] = entry;
        }
    }
    Ok(mat)
}

fn srp_position_partials_orekit_or_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio: Grad,
    ratio_km: ScaledGrad<3>,
    model: SolarRadiationPressure,
) -> Result<Matrix3<f64>, PropagationError> {
    if let Some(accel) = srp_acceleration_field_m_s2_fast(sat_pos_km, sun_pos_km, ratio, model) {
        return Ok(Matrix3::from_rows(&[
            RowVector3::from(accel.x.grad),
            RowVector3::from(accel.y.grad),
            RowVector3::from(accel.z.grad),
        ]));
    }
    srp_position_partials_scaled(sat_pos_km, sun_pos_km, ratio_km, model)
}

/// Position partials of [`srp_acceleration_with_bodies`], in 1/s^2.
fn srp_position_partials_with_bodies(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    model: SolarRadiationPressure,
) -> Result<Matrix3<f64>, PropagationError> {
    validate_sun_position(sat_pos_km, sun_pos_km)?;
    if matches!(model.shadow_function, ShadowFunction::Conical) {
        let (ratio_km, is_sunlit) = conical_lighting_ratio_field_scaled(
            as_array(sat_pos_km),
            as_array(sun_pos_km),
            model.earth_shadow_model,
        )
        .map_err(eclipse_failure)?;
        if !ratio_km.is_all_finite() {
            return Err(PropagationError::NumericalFailure(
                "eclipse: shadow fraction gradient is not finite".to_string(),
            ));
        }
        if is_sunlit {
            return srp_position_partials_orekit_or_scaled(
                sat_pos_km,
                sun_pos_km,
                Grad::constant(1.0),
                ratio_km,
                model,
            );
        }
        return srp_position_partials_scaled(sat_pos_km, sun_pos_km, ratio_km, model);
    }
    let ratio = lighting_ratio_field_of(model, sat_pos_km, sun_pos_km, moon_pos_km)?;
    let m_per_km = ScaledF64::from_f64(M_PER_KM);
    let ratio_km = ScaledGrad {
        val: ScaledF64::from_f64(ratio.value),
        grad: [
            ScaledF64::from_f64(ratio.grad[0]).mul(m_per_km),
            ScaledF64::from_f64(ratio.grad[1]).mul(m_per_km),
            ScaledF64::from_f64(ratio.grad[2]).mul(m_per_km),
        ],
    };
    srp_position_partials_orekit_or_scaled(sat_pos_km, sun_pos_km, ratio, ratio_km, model)
}

/// Lighting ratio value for the `C_R` parameter partial.
///
/// Takes the value from the field lighting ratio when finite (matching Orekit's
/// `FieldSpacecraftState<Gradient>` parameter derivative at 0 ULP), and falls
/// back to the double lighting ratio when field evaluation hits
/// `NumericalFailure` (such as an exact-axis annular eclipse where double
/// evaluation is defined).
fn lighting_ratio_for_parameter_partial(
    model: SolarRadiationPressure,
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
) -> Result<f64, PropagationError> {
    match lighting_ratio_field_of(model, sat_pos_km, sun_pos_km, moon_pos_km) {
        Ok(ratio) => Ok(ratio.value),
        Err(PropagationError::NumericalFailure(_)) => {
            let ratio = lighting_ratio_of(model, sat_pos_km, sun_pos_km, moon_pos_km)?;
            if !ratio.is_finite() {
                return Err(PropagationError::NumericalFailure(
                    "eclipse: lighting ratio is not finite".to_string(),
                ));
            }
            Ok(ratio)
        }
        Err(err) => Err(err),
    }
}

/// Evaluate Orekit `Field<Gradient>` `SolarRadiationPressure.acceleration` +
/// `IsotropicRadiationSingleCoefficient.radiationPressureAcceleration` in m/s^2
/// with `C_R` as the free parameter, returning `None` if any intermediate or
/// output value or gradient entry is non-finite or `r2.value == 0`.
fn srp_acceleration_field_cr_m_s2_fast(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio_val: f64,
    model: SolarRadiationPressure,
) -> Option<FieldVec3<1>> {
    if !ratio_val.is_finite() {
        return None;
    }
    let d_ref = model.au_km * M_PER_KM;
    let k_ref = model.pressure_n_m2 * d_ref * d_ref;
    let sat_m = Vec3::new(
        sat_pos_km.x * M_PER_KM,
        sat_pos_km.y * M_PER_KM,
        sat_pos_km.z * M_PER_KM,
    );
    let sun_m = Vec3::new(
        sun_pos_km.x * M_PER_KM,
        sun_pos_km.y * M_PER_KM,
        sun_pos_km.z * M_PER_KM,
    );
    if !(d_ref.is_finite() && k_ref.is_finite() && vec3_is_finite(sat_m) && vec3_is_finite(sun_m)) {
        return None;
    }
    let position = FieldVec3::<1>::constant(sat_m);
    let sun_position = FieldVec3::<1>::constant(sun_m);
    let sun_sat_vector = position.sub(sun_position);
    let r2 = sun_sat_vector.norm_sq();
    if !(field_vec3_is_all_finite(sun_sat_vector) && r2.is_all_finite() && r2.value > 0.0) {
        return None;
    }
    let factor = Grad::<1>::ONE
        .reciprocal()
        .mul(Grad::<1>::ONE)
        .mul_f64(model.area_to_mass_m2_kg)
        .mul(Grad::<1>::variable(0, model.cr));
    if !factor.is_all_finite() {
        return None;
    }
    let ratio = Grad::<1>::constant(ratio_val);
    let ratio_k = ratio.mul_f64(k_ref);
    let raw_p = ratio_k.div(r2);
    let r = r2.sqrt();
    let scale = raw_p.div(r);
    if !(ratio_k.is_all_finite()
        && raw_p.is_all_finite()
        && r.is_all_finite()
        && scale.is_all_finite())
    {
        return None;
    }
    let flux = FieldVec3::from_scaled(scale, sun_sat_vector);
    if !field_vec3_is_all_finite(flux) {
        return None;
    }
    let accel = FieldVec3::from_scaled(factor, flux);
    field_vec3_is_all_finite(accel).then_some(accel)
}

#[cfg(test)]
fn srp_acceleration_field_cr_m_s2(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio_val: f64,
    model: SolarRadiationPressure,
) -> Result<FieldVec3<1>, PropagationError> {
    srp_acceleration_field_cr_m_s2_fast(sat_pos_km, sun_pos_km, ratio_val, model).ok_or_else(|| {
        PropagationError::NumericalFailure(
            "SRP acceleration or parameter partial is not finite".to_string(),
        )
    })
}

/// Acceleration (km/s^2) and `C_R` parameter partial (km/s^2) evaluated
/// entirely in exponent-separated arithmetic from km inputs. The value part
/// equals [`srp_double_scaled`] bit for bit.
fn srp_conical_cr_field_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    nu_s: ScaledF64,
    model: SolarRadiationPressure,
) -> Result<ScaledFieldVec3<1>, PropagationError> {
    let d_ref_s = ScaledF64::from_f64(model.au_km);
    let k_ref_s = ScaledF64::from_f64(model.pressure_n_m2)
        .mul(d_ref_s)
        .mul(d_ref_s);
    let area_s = ScaledF64::from_f64(model.area_to_mass_m2_kg);
    let factor_val = area_s.mul(ScaledF64::from_f64(model.cr));
    let sat_s = ScaledVec3::from_vec3(Vec3::new(sat_pos_km.x, sat_pos_km.y, sat_pos_km.z));
    let sun_s = ScaledVec3::from_vec3(Vec3::new(sun_pos_km.x, sun_pos_km.y, sun_pos_km.z));
    let (u_d, r_s) = sat_s.sub(sun_s).normalize_scaled().ok_or_else(|| {
        PropagationError::NumericalFailure("Satellite coincident with Sun".to_string())
    })?;
    let r2_s = r_s.mul(r_s);
    let scale_val = factor_val.mul(k_ref_s).div(r2_s);
    let scale_grad = area_s.mul(k_ref_s).div(r2_s);
    let a1_s = ScaledFieldVec3 {
        x: ScaledGrad {
            val: u_d.x.mul(scale_val),
            grad: [u_d.x.mul(scale_grad)],
        },
        y: ScaledGrad {
            val: u_d.y.mul(scale_val),
            grad: [u_d.y.mul(scale_grad)],
        },
        z: ScaledGrad {
            val: u_d.z.mul(scale_val),
            grad: [u_d.z.mul(scale_grad)],
        },
    };
    let nu_g = ScaledGrad {
        val: nu_s,
        grad: [ScaledF64::ZERO],
    };
    Ok(a1_s
        .scalar_mul(nu_g)
        .scalar_div(ScaledF64::from_f64(M_PER_KM)))
}

fn srp_parameter_partial_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    nu_s: ScaledF64,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    if !nu_s.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "SRP lighting ratio is not finite".to_string(),
        ));
    }
    if nu_s.is_zero() {
        return Ok(Vector3::zeros());
    }
    let accel_s = srp_conical_cr_field_scaled(sat_pos_km, sun_pos_km, nu_s, model)?;
    finite_acceleration(Vector3::new(
        accel_s.x.grad[0].to_f64(),
        accel_s.y.grad[0].to_f64(),
        accel_s.z.grad[0].to_f64(),
    ))
}

fn srp_parameter_partial_orekit_or_scaled(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    ratio_val: f64,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    if let Some(accel) =
        srp_acceleration_field_cr_m_s2_fast(sat_pos_km, sun_pos_km, ratio_val, model)
    {
        let partial = Vector3::new(
            accel.x.grad[0] / M_PER_KM,
            accel.y.grad[0] / M_PER_KM,
            accel.z.grad[0] / M_PER_KM,
        );
        if partial.iter().all(|c| c.is_finite()) {
            return Ok(partial);
        }
    }
    srp_parameter_partial_scaled(
        sat_pos_km,
        sun_pos_km,
        ScaledF64::from_f64(ratio_val),
        model,
    )
}

/// Parameter partial `d(accel)/d(C_R)` in km/s^2, following Orekit's
/// `FieldSpacecraftState<Gradient>` evaluation with `C_R` as a free parameter.
fn srp_parameter_partial_with_bodies(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    validate_sun_position(sat_pos_km, sun_pos_km)?;
    if matches!(model.shadow_function, ShadowFunction::Conical) {
        let (nu_s, is_sunlit) = conical_lighting_ratio_double_scaled(
            as_array(sat_pos_km),
            as_array(sun_pos_km),
            model.earth_shadow_model,
        )
        .map_err(eclipse_failure)?;
        if is_sunlit {
            return srp_parameter_partial_orekit_or_scaled(sat_pos_km, sun_pos_km, 1.0, model);
        }
        return srp_parameter_partial_scaled(sat_pos_km, sun_pos_km, nu_s, model);
    }
    let ratio_val =
        lighting_ratio_for_parameter_partial(model, sat_pos_km, sun_pos_km, moon_pos_km)?;
    srp_parameter_partial_orekit_or_scaled(sat_pos_km, sun_pos_km, ratio_val, model)
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
