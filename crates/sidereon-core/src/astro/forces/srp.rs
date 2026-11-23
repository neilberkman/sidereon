//! Cannonball solar radiation pressure.
//!
//! The acceleration follows the Montenbruck & Gill cannonball SRP model,
//! `-P_sr * (AU / |r_sun|)^2 * C_R * (A/m) * nu * s_hat`, where `s_hat`
//! points from the satellite to the Sun. SI pressure and area-to-mass units are
//! used internally, then converted from m/s^2 to km/s^2 for the propagation
//! state.
//!
//! By default, the shadow fraction `1 - nu` uses the Earth-only conical shadow
//! model ([`shadow_fraction`]). Callers can opt into lunar occultation via
//! [`SolarRadiationPressure::with_moon_eclipse`], which evaluates combined
//! Earth and Moon apparent-disc occultation
//! ([`shadow_fraction_with_model_and_moon`]).
//!
//! [`ForceModel::acceleration`] takes the Sun and Moon positions from the
//! analytic series in GCRS ([`sun_moon_gcrs_at_tdb_seconds`]), the frame of the
//! propagation state; [`SolarRadiationPressure::acceleration_with_sun_moon`]
//! takes caller positions in the state's frame.

use crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds;
use crate::astro::constants::astro::{AU_KM, SOLAR_RADIATION_PRESSURE_N_M2};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::error::PropagationError;
use crate::astro::events::eclipse::{
    shadow_fraction, shadow_fraction_with_model, shadow_fraction_with_model_and_moon,
    EarthShadowModel,
};
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

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
    /// Off by default so existing Earth-only propagation results stay
    /// bit-identical. When enabled, [`ForceModel::acceleration`] combines
    /// Earth and Moon apparent-disc occultation using the Moon position from
    /// [`sun_moon_gcrs_at_tdb_seconds`].
    pub fn with_moon_eclipse(mut self, enabled: bool) -> Self {
        self.include_moon_eclipse = enabled;
        self
    }

    /// Whether lunar occultation is included in the shadow factor.
    pub fn moon_eclipse_enabled(&self) -> bool {
        self.include_moon_eclipse
    }

    /// Select the Earth figure model used for shadow evaluation.
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
}

impl ForceModel for SolarRadiationPressure {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        if !self.include_moon_eclipse && self.earth_shadow_model == EarthShadowModel::Spherical {
            let sun_pos_km = sun_position_km(state.epoch_tdb_seconds)?;
            return srp_acceleration_with_sun(state.position_km, sun_pos_km, *self);
        }
        let (sun_pos_km, moon_pos_km) = sun_moon_positions_km(state.epoch_tdb_seconds)?;
        srp_acceleration_with_bodies(state.position_km, sun_pos_km, Some(moon_pos_km), *self)
    }
}

fn sun_position_km(epoch_tdb_seconds: f64) -> Result<Vector3<f64>, PropagationError> {
    let (sun, _moon) = sun_moon_positions_km(epoch_tdb_seconds)?;
    Ok(sun)
}

fn sun_moon_positions_km(
    epoch_tdb_seconds: f64,
) -> Result<(Vector3<f64>, Vector3<f64>), PropagationError> {
    let bodies = sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds)
        .map_err(|error| PropagationError::ForceModelFailure(format!("Sun/Moon: {error}")))?;
    Ok((
        Vector3::new(
            bodies.sun[0] / M_PER_KM,
            bodies.sun[1] / M_PER_KM,
            bodies.sun[2] / M_PER_KM,
        ),
        Vector3::new(
            bodies.moon[0] / M_PER_KM,
            bodies.moon[1] / M_PER_KM,
            bodies.moon[2] / M_PER_KM,
        ),
    ))
}

fn srp_acceleration_with_sun(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    srp_acceleration_with_bodies(sat_pos_km, sun_pos_km, None, model)
}

fn srp_acceleration_with_bodies(
    sat_pos_km: Vector3<f64>,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    model: SolarRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    let sat_to_sun = sun_pos_km - sat_pos_km;
    let sat_to_sun_r2 = sat_to_sun.norm_squared();
    if sat_to_sun_r2 == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Satellite coincident with Sun".to_string(),
        ));
    }
    let sun_r2 = sun_pos_km.norm_squared();
    if sun_r2 == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Zero Sun position magnitude".to_string(),
        ));
    }

    let sat_arr = [sat_pos_km.x, sat_pos_km.y, sat_pos_km.z];
    let sun_arr = [sun_pos_km.x, sun_pos_km.y, sun_pos_km.z];
    let shadow = if model.include_moon_eclipse {
        let moon = moon_pos_km.ok_or_else(|| {
            PropagationError::ForceModelFailure("eclipse: missing Moon position".to_string())
        })?;
        shadow_fraction_with_model_and_moon(
            sat_arr,
            sun_arr,
            [moon.x, moon.y, moon.z],
            model.earth_shadow_model,
        )
    } else if model.earth_shadow_model == EarthShadowModel::Spherical {
        shadow_fraction(sat_arr, sun_arr)
    } else {
        shadow_fraction_with_model(sat_arr, sun_arr, model.earth_shadow_model)
    }
    .map_err(|error| PropagationError::ForceModelFailure(format!("eclipse: {error}")))?;
    if !sat_to_sun_r2.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "Non-finite satellite-to-Sun distance".to_string(),
        ));
    }
    if !sun_r2.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "Non-finite Sun position magnitude".to_string(),
        ));
    }
    let illumination = 1.0 - shadow;
    if illumination == 0.0 {
        return Ok(Vector3::zeros());
    }

    let s_hat = sat_to_sun / sat_to_sun_r2.sqrt();
    let pressure_scale = model.pressure_n_m2 * (model.au_km * model.au_km / sun_r2);
    let accel_m_s2 = -pressure_scale * model.cr * model.area_to_mass_m2_kg * illumination * s_hat;
    let accel_km_s2 = accel_m_s2 / M_PER_KM;
    if !accel_km_s2.iter().all(|v| v.is_finite()) {
        return Err(PropagationError::NumericalFailure(
            "Non-finite SRP acceleration".to_string(),
        ));
    }
    Ok(accel_km_s2)
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
mod tests {
    //! Clean-room validation anchors:
    //!
    //! The expected 1 AU acceleration magnitude is the published cannonball SRP
    //! pressure `4.56e-6 N/m^2` multiplied by `C_R * A/m` and converted from
    //! m/s^2 to km/s^2. The shadow tests use fixed conical geometry and the
    //! existing eclipse helper, whose return value is geometric shadow fraction.

    use super::*;
    use crate::astro::constants::astro::AU_KM;

    #[test]
    fn one_au_magnitude_matches_pressure_area_mass_scale() {
        let model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
        let sat = Vector3::new(7000.0, 0.0, 0.0);
        let sun = Vector3::new(AU_KM, 0.0, 0.0);
        let accel = srp_acceleration_with_sun(sat, sun, model).expect("SRP acceleration");

        let expected_km_s2 = 4.56e-6 * 1.2 * 0.02 / M_PER_KM;
        assert!((accel.norm() - expected_km_s2).abs() < 1.0e-24);
    }

    #[test]
    fn acceleration_is_zero_in_deep_umbra() {
        let model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
        let sat = Vector3::new(-7000.0, 0.0, 0.0);
        let sun = Vector3::new(AU_KM, 0.0, 0.0);
        let accel = srp_acceleration_with_sun(sat, sun, model).expect("SRP acceleration");

        assert_eq!(accel, Vector3::zeros());
    }

    #[test]
    fn direction_is_away_from_sun() {
        let model = SolarRadiationPressure::new(1.0, 0.01).expect("valid SRP");
        let sat = Vector3::new(7000.0, 0.0, 0.0);
        let sun = Vector3::new(AU_KM, 0.0, 0.0);
        let accel = srp_acceleration_with_sun(sat, sun, model).expect("SRP acceleration");

        assert!(accel.x < 0.0);
        assert_eq!(accel.y, 0.0);
        assert_eq!(accel.z, 0.0);
    }

    #[test]
    fn moon_eclipse_is_opt_in_and_attenuates_srp() {
        let default_model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
        assert!(!default_model.moon_eclipse_enabled());
        assert_eq!(
            default_model.earth_shadow_model(),
            EarthShadowModel::Spherical
        );

        let moon_model = default_model.with_moon_eclipse(true);
        assert!(moon_model.moon_eclipse_enabled());

        let sat = Vector3::new(7_000.0, 0.0, 0.0);
        let sun = Vector3::new(AU_KM, 0.0, 0.0);
        let moon_clear = Vector3::new(0.0, 384_400.0, 0.0);
        let moon_total = Vector3::new(7_000.0 + 356_000.0, 0.0, 0.0);
        let moon_annular = Vector3::new(7_000.0 + 405_000.0, 0.0, 0.0);

        // Clear Moon matches default bit-for-bit in full sunlight.
        let accel_default = default_model
            .acceleration_with_sun_moon(sat, sun, moon_clear)
            .expect("default SRP");
        let accel_clear = moon_model
            .acceleration_with_sun_moon(sat, sun, moon_clear)
            .expect("moon-enabled clear SRP");
        for i in 0..3 {
            assert_eq!(accel_clear[i].to_bits(), accel_default[i].to_bits());
        }

        // Default ignores lunar umbra; opt-in zeroes acceleration.
        let accel_default_total = default_model
            .acceleration_with_sun_moon(sat, sun, moon_total)
            .expect("default ignores moon");
        for i in 0..3 {
            assert_eq!(accel_default_total[i].to_bits(), accel_default[i].to_bits());
        }
        let accel_total = moon_model
            .acceleration_with_sun_moon(sat, sun, moon_total)
            .expect("lunar total SRP");
        assert_eq!(accel_total, Vector3::zeros());

        // Annular lunar eclipse reduces SRP without zeroing it.
        let accel_annular = moon_model
            .acceleration_with_sun_moon(sat, sun, moon_annular)
            .expect("lunar annular SRP");
        assert!(accel_annular.x < 0.0);
        assert!(accel_annular.norm() < 0.25 * accel_clear.norm());
        assert!(accel_annular.norm() > 0.1 * accel_clear.norm());

        // Non-finite Moon position returns a typed error.
        let err = moon_model
            .acceleration_with_sun_moon(sat, sun, Vector3::new(f64::NAN, 0.0, 0.0))
            .expect_err("non-finite moon rejected");
        assert!(matches!(err, PropagationError::InvalidInput(_)));
    }

    #[test]
    fn non_finite_state_position_preserves_force_model_error_variant() {
        let model = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP");
        let state = CartesianState::new(0.0, [f64::NAN, 0.0, 0.0], [0.0, 7.5, 0.0]);
        let ctx = PropagationContext::default();
        let err = model
            .acceleration(&state, &ctx)
            .expect_err("non-finite state position must fail");
        assert!(
            matches!(err, PropagationError::ForceModelFailure(_)),
            "expected ForceModelFailure, got {err:?}"
        );
    }

    /// TDB epochs from 1990 to 2050, one at the 2024-04-08 solar eclipse.
    fn epochs_1990_to_2050() -> [f64; 4] {
        use crate::astro::time::civil::j2000_seconds;
        [
            j2000_seconds(1990, 2, 1, 0, 0, 0.0),
            j2000_seconds(2024, 4, 8, 18, 17, 0.0),
            j2000_seconds(2026, 11, 23, 0, 0, 0.0),
            j2000_seconds(2050, 6, 30, 12, 0, 0.0),
        ]
    }

    fn gcrs_sun_moon_km(epoch_tdb_seconds: f64) -> (Vector3<f64>, Vector3<f64>) {
        let bodies =
            crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds)
                .expect("GCRS Sun/Moon");
        let km = |v: [f64; 3]| Vector3::new(v[0] / M_PER_KM, v[1] / M_PER_KM, v[2] / M_PER_KM);
        (km(bodies.sun), km(bodies.moon))
    }

    fn mean_of_date_sun_moon_km(epoch_tdb_seconds: f64) -> (Vector3<f64>, Vector3<f64>) {
        let t = crate::astro::time::scales::tt_seconds_from_tdb_seconds(epoch_tdb_seconds)
            / (36_525.0 * 86_400.0);
        let bodies = crate::astro::bodies::sun_moon::sun_moon_eci(t).expect("mean of date");
        let km = |v: [f64; 3]| Vector3::new(v[0] / M_PER_KM, v[1] / M_PER_KM, v[2] / M_PER_KM);
        (km(bodies.sun), km(bodies.moon))
    }

    #[test]
    fn acceleration_equals_the_caller_vector_path_with_the_gcrs_sun_and_moon() {
        let ctx = PropagationContext::default();
        let base = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
        let models = [
            base,
            base.with_moon_eclipse(true),
            base.with_earth_shadow_model(EarthShadowModel::Wgs84Oblate),
            base.with_moon_eclipse(true)
                .with_earth_shadow_model(EarthShadowModel::Wgs84Oblate),
        ];
        for epoch in epochs_1990_to_2050() {
            let (sun, moon) = gcrs_sun_moon_km(epoch);
            let s_hat = sun.normalize();
            let across = s_hat.cross(&Vector3::z()).normalize();
            // Sunlit, Earth penumbra, GEO behind the Earth, and a GNSS orbit.
            let positions = [
                s_hat * 7_000.0,
                -s_hat * 7_000.0 + across * 6_365.0,
                -s_hat * 42_164.0 + across * 1_000.0,
                Vector3::new(26_560.0, 1_200.0, -3_400.0),
            ];
            for position in positions {
                let state = CartesianState::new(epoch, position.into(), [0.0, 7.5, 0.0]);
                for model in models {
                    let from_state = model.acceleration(&state, &ctx).expect("acceleration");
                    let from_vectors = model
                        .acceleration_with_sun_moon(position, sun, moon)
                        .expect("caller vectors");
                    assert_eq!(
                        from_state, from_vectors,
                        "epoch {epoch} position {position:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn acceleration_turns_away_from_the_mean_of_date_sun_by_the_precession() {
        // Read as GCRS, the mean of date Sun is off by the precession since
        // J2000.0 plus the frame bias, 0.376 deg at 2026-11-23; SRP on a sunlit
        // satellite turns by that angle between the two.
        let epoch = crate::astro::time::civil::j2000_seconds(2026, 11, 23, 0, 0, 0.0);
        let (sun, moon) = gcrs_sun_moon_km(epoch);
        let (mean_sun, mean_moon) = mean_of_date_sun_moon_km(epoch);
        let model = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
        let position = sun.normalize() * 7_000.0;
        let state = CartesianState::new(epoch, position.into(), [0.0, 7.5, 0.0]);
        let used = model
            .acceleration(&state, &PropagationContext::default())
            .expect("acceleration");
        let with_mean = model
            .acceleration_with_sun_moon(position, mean_sun, mean_moon)
            .expect("mean of date Sun");
        let angle_deg = used.angle(&with_mean).to_degrees();
        assert!((0.37..0.38).contains(&angle_deg), "{angle_deg} deg");
        assert_eq!(
            used,
            model
                .acceleration_with_sun_moon(position, sun, moon)
                .expect("GCRS")
        );
    }
}
