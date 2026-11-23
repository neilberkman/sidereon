//! Earth albedo and infrared radiation pressure.
//!
//! This module implements the Knocke-Ries-Tapley diffuse Earth radiation
//! pressure model for a cannonball spacecraft. The Earth is treated as a
//! Lambertian sphere, divided into the original 19 equal apparent-area elements
//! consisting of a nadir cap, six first-ring segments, and twelve outer-ring
//! segments. Each element contributes reflected shortwave radiation when sunlit
//! plus emitted longwave radiation.

use crate::astro::bodies::sun_moon::sun_moon_eci;
use crate::astro::constants::astro::AU_KM;
use crate::astro::constants::earth::WGS84_A_KM;
use crate::astro::constants::time::{
    DAYS_PER_JULIAN_CENTURY, DAYS_PER_JULIAN_YEAR, J2000_JD, SECONDS_PER_DAY,
};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::error::PropagationError;
use crate::astro::events::eclipse::moon_shadow_fraction;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

const KNOCKE_ELEMENT_COUNT: usize = 19;
const KNOCKE_RING_SEGMENTS: [usize; 3] = [1, 6, 12];
const KNOCKE_SOLAR_PRESSURE_N_M2: f64 = f64::from_bits(0x3ed3_20e8_2eb2_57f2);
const KNOCKE_REFERENCE_JD: f64 = f64::from_bits(0x4142_a750_4000_0000);

const ALBEDO_A0: f64 = f64::from_bits(0x3fd5_c28f_5c28_f5c3);
const ALBEDO_C0: f64 = 0.0;
const ALBEDO_C1: f64 = f64::from_bits(0x3fb9_9999_9999_999a);
const ALBEDO_C2: f64 = 0.0;
const ALBEDO_A2: f64 = f64::from_bits(0x3fd2_8f5c_28f5_c28f);

const EMISSIVITY_E0: f64 = f64::from_bits(0x3fe5_c28f_5c28_f5c3);
const EMISSIVITY_K0: f64 = 0.0;
const EMISSIVITY_K1: f64 = f64::from_bits(0xbfb1_eb85_1eb8_51ec);
const EMISSIVITY_K2: f64 = 0.0;
const EMISSIVITY_E2: f64 = f64::from_bits(0xbfc7_0a3d_70a3_d70a);
const MAX_LATITUDE_RAD: f64 = std::f64::consts::FRAC_PI_2;

/// Cannonball Earth albedo and infrared radiation pressure parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthRadiationPressure {
    cr: f64,
    area_to_mass_m2_kg: f64,
    include_moon_eclipse: bool,
}

impl EarthRadiationPressure {
    /// Build the Knocke-Ries-Tapley Earth radiation pressure model.
    ///
    /// `cr` is the cannonball radiation coefficient and
    /// `area_to_mass_m2_kg` is the spacecraft area-to-mass ratio.
    pub fn new(cr: f64, area_to_mass_m2_kg: f64) -> Result<Self, PropagationError> {
        validate_positive("cr", cr)?;
        validate_positive("area_to_mass_m2_kg", area_to_mass_m2_kg)?;
        Ok(Self {
            cr,
            area_to_mass_m2_kg,
            include_moon_eclipse: false,
        })
    }

    /// Enable or disable lunar occultation of sunlight on Earth's surface
    /// elements.
    ///
    /// Off by default so existing Earth-only albedo results stay bit-identical.
    /// When enabled, each sunlit Earth surface element attenuates its reflected
    /// shortwave albedo flux by `1.0 - moon_shadow_fraction` using the Moon
    /// position from [`sun_moon_eci`], while emitted thermal infrared remains
    /// unchanged.
    pub fn with_moon_eclipse(mut self, enabled: bool) -> Self {
        self.include_moon_eclipse = enabled;
        self
    }

    /// Whether lunar occultation of Earth surface elements is enabled.
    pub fn moon_eclipse_enabled(&self) -> bool {
        self.include_moon_eclipse
    }

    /// Cannonball radiation coefficient.
    pub fn cr(&self) -> f64 {
        self.cr
    }

    /// Spacecraft area-to-mass ratio, m^2/kg.
    pub fn area_to_mass_m2_kg(&self) -> f64 {
        self.area_to_mass_m2_kg
    }

    /// Solar radiation pressure at 1 AU used by the model, N/m^2.
    pub fn pressure_n_m2(&self) -> f64 {
        KNOCKE_SOLAR_PRESSURE_N_M2
    }

    /// Spherical Earth radius used for surface elements, km.
    pub fn earth_radius_km(&self) -> f64 {
        WGS84_A_KM
    }

    /// Astronomical unit used in the inverse-square solar flux scale, km.
    pub fn au_km(&self) -> f64 {
        AU_KM
    }

    /// Evaluate the Knocke albedo and emissivity coefficients at latitude.
    pub fn albedo_emissivity(
        &self,
        epoch_tdb_seconds: f64,
        latitude_rad: f64,
    ) -> Result<(f64, f64), PropagationError> {
        validate_finite("epoch_tdb_seconds", epoch_tdb_seconds)?;
        validate_latitude(latitude_rad)?;
        Ok(knocke_albedo_emissivity(epoch_tdb_seconds, latitude_rad))
    }

    /// Evaluate Earth albedo and infrared radiation pressure acceleration
    /// (km/s^2) using explicit inertial Sun and Moon positions (km).
    ///
    /// When [`Self::moon_eclipse_enabled`] is `true`, each sunlit surface
    /// element scales its reflected shortwave contribution by the unocculted
    /// solar fraction past the Moon. When `false`, `moon_pos_km` is still
    /// validated for finiteness and the result is bit-identical to the
    /// Earth-only model.
    pub fn acceleration_with_sun_moon(
        &self,
        state: &CartesianState,
        sun_pos_km: Vector3<f64>,
        moon_pos_km: Vector3<f64>,
    ) -> Result<Vector3<f64>, PropagationError> {
        validate_vector(moon_pos_km, "moon_pos_km")?;
        let moon = if self.include_moon_eclipse {
            Some(moon_pos_km)
        } else {
            None
        };
        earth_radiation_acceleration_with_bodies(state, sun_pos_km, moon, *self)
    }
}

impl ForceModel for EarthRadiationPressure {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        if !self.include_moon_eclipse {
            let sun_pos_km = sun_position_km(state.epoch_tdb_seconds)?;
            return earth_radiation_acceleration_with_sun(state, sun_pos_km, *self);
        }
        let (sun_pos_km, moon_pos_km) = sun_moon_positions_km(state.epoch_tdb_seconds)?;
        earth_radiation_acceleration_with_bodies(state, sun_pos_km, Some(moon_pos_km), *self)
    }
}

fn sun_position_km(epoch_tdb_seconds: f64) -> Result<Vector3<f64>, PropagationError> {
    let (sun, _moon) = sun_moon_positions_km(epoch_tdb_seconds)?;
    Ok(sun)
}

fn sun_moon_positions_km(
    epoch_tdb_seconds: f64,
) -> Result<(Vector3<f64>, Vector3<f64>), PropagationError> {
    let t_tt_centuries = epoch_tdb_seconds / (DAYS_PER_JULIAN_CENTURY * SECONDS_PER_DAY);
    let bodies = sun_moon_eci(t_tt_centuries)
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

fn earth_radiation_acceleration_with_sun(
    state: &CartesianState,
    sun_pos_km: Vector3<f64>,
    model: EarthRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    earth_radiation_acceleration_with_bodies(state, sun_pos_km, None, model)
}

fn earth_radiation_acceleration_with_bodies(
    state: &CartesianState,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    model: EarthRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    validate_state(state)?;
    validate_vector(sun_pos_km, "sun_pos_km")?;
    if let Some(moon) = moon_pos_km {
        validate_vector(moon, "moon_pos_km")?;
    }

    let sat_radius_km = state.position_km.norm();
    if sat_radius_km <= model.earth_radius_km() {
        return Err(PropagationError::InvalidInput(
            "satellite radius must exceed Earth radius".to_string(),
        ));
    }
    let sun_r2 = sun_pos_km.norm_squared();
    if sun_r2 == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Zero Sun position magnitude".to_string(),
        ));
    }
    if !sun_r2.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "Non-finite Sun position magnitude".to_string(),
        ));
    }
    if let Some(moon) = moon_pos_km {
        let moon_r2 = moon.norm_squared();
        if moon_r2 == 0.0 {
            return Err(PropagationError::NumericalFailure(
                "Zero Moon position magnitude".to_string(),
            ));
        }
        if !moon_r2.is_finite() {
            return Err(PropagationError::NumericalFailure(
                "Non-finite Moon position magnitude".to_string(),
            ));
        }
    }

    let radial_hat = state.position_km / sat_radius_km;
    let (east_hat, north_hat) = tangent_basis(radial_hat)?;
    let sun_hat = sun_pos_km / sun_r2.sqrt();
    let solar_pressure = model.pressure_n_m2() * model.au_km() * model.au_km() / sun_r2;
    let disk_half_angle = libm::asin(model.earth_radius_km() / sat_radius_km);
    let one_minus_cos_disk = 1.0 - libm::cos(disk_half_angle);
    let apparent_weight = 2.0 * one_minus_cos_disk / KNOCKE_ELEMENT_COUNT as f64;

    let mut accel_m_s2 = Vector3::zeros();
    let mut previous_elements = 0usize;
    for (ring_index, &segments) in KNOCKE_RING_SEGMENTS.iter().enumerate() {
        if ring_index == 0 {
            let sample = SurfaceElementSample {
                element_pos_km: radial_hat * model.earth_radius_km(),
                normal: radial_hat,
                element_to_sat_hat: radial_hat,
            };
            accel_m_s2 += element_acceleration(
                state.epoch_tdb_seconds,
                sample,
                sun_pos_km,
                moon_pos_km,
                sun_hat,
                solar_pressure,
                apparent_weight,
                model,
            )?;
        } else {
            let current_elements = previous_elements + segments;
            let inner_mu =
                1.0 - previous_elements as f64 / KNOCKE_ELEMENT_COUNT as f64 * one_minus_cos_disk;
            let outer_mu =
                1.0 - current_elements as f64 / KNOCKE_ELEMENT_COUNT as f64 * one_minus_cos_disk;
            let psi = libm::acos(0.5 * (inner_mu + outer_mu));
            let (sin_psi, cos_psi) = libm::sincos(psi);
            for segment in 0..segments {
                let azimuth = (segment as f64 + 0.5) * std::f64::consts::TAU / segments as f64;
                let (sin_azimuth, cos_azimuth) = libm::sincos(azimuth);
                let sat_to_element_hat = -cos_psi * radial_hat
                    + sin_psi * (cos_azimuth * east_hat + sin_azimuth * north_hat);
                let sample = surface_sample_from_ray(
                    state.position_km,
                    sat_radius_km,
                    sat_to_element_hat,
                    sin_psi,
                    cos_psi,
                    model.earth_radius_km(),
                )?;
                accel_m_s2 += element_acceleration(
                    state.epoch_tdb_seconds,
                    sample,
                    sun_pos_km,
                    moon_pos_km,
                    sun_hat,
                    solar_pressure,
                    apparent_weight,
                    model,
                )?;
            }
        }
        previous_elements += segments;
    }

    let accel_km_s2 = accel_m_s2 / M_PER_KM;
    if !accel_km_s2.iter().all(|v| v.is_finite()) {
        return Err(PropagationError::NumericalFailure(
            "Non-finite Earth radiation acceleration".to_string(),
        ));
    }
    Ok(accel_km_s2)
}

#[derive(Debug, Clone, Copy)]
struct SurfaceElementSample {
    element_pos_km: Vector3<f64>,
    normal: Vector3<f64>,
    element_to_sat_hat: Vector3<f64>,
}

fn surface_sample_from_ray(
    sat_pos_km: Vector3<f64>,
    sat_radius_km: f64,
    sat_to_element_hat: Vector3<f64>,
    sin_psi: f64,
    cos_psi: f64,
    earth_radius_km: f64,
) -> Result<SurfaceElementSample, PropagationError> {
    let discriminant =
        earth_radius_km * earth_radius_km - sat_radius_km * sat_radius_km * sin_psi * sin_psi;
    if discriminant < 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Earth radiation ray misses Earth".to_string(),
        ));
    }
    let path_km = sat_radius_km * cos_psi - discriminant.sqrt();
    let element_pos_km = sat_pos_km + path_km * sat_to_element_hat;
    let normal = element_pos_km / earth_radius_km;
    Ok(SurfaceElementSample {
        element_pos_km,
        normal,
        element_to_sat_hat: -sat_to_element_hat,
    })
}

#[allow(clippy::too_many_arguments)]
fn element_acceleration(
    epoch_tdb_seconds: f64,
    sample: SurfaceElementSample,
    sun_pos_km: Vector3<f64>,
    moon_pos_km: Option<Vector3<f64>>,
    sun_hat: Vector3<f64>,
    solar_pressure_n_m2: f64,
    apparent_weight: f64,
    model: EarthRadiationPressure,
) -> Result<Vector3<f64>, PropagationError> {
    let latitude_rad = libm::asin(sample.normal.z.clamp(-1.0, 1.0));
    let (albedo, emissivity) = knocke_albedo_emissivity(epoch_tdb_seconds, latitude_rad);
    let cos_solar_zenith = sample.normal.dot(&sun_hat);
    let reflected = if cos_solar_zenith > 0.0 {
        if let Some(moon) = moon_pos_km {
            let shadow = moon_shadow_fraction(
                [
                    sample.element_pos_km.x,
                    sample.element_pos_km.y,
                    sample.element_pos_km.z,
                ],
                [sun_pos_km.x, sun_pos_km.y, sun_pos_km.z],
                [moon.x, moon.y, moon.z],
            )
            .map_err(|error| PropagationError::ForceModelFailure(format!("eclipse: {error}")))?;
            albedo * cos_solar_zenith * (1.0 - shadow)
        } else {
            albedo * cos_solar_zenith
        }
    } else {
        0.0
    };
    let pressure_scale = solar_pressure_n_m2 * (reflected + 0.25 * emissivity) * apparent_weight;

    Ok(sample.element_to_sat_hat * (model.cr * model.area_to_mass_m2_kg * pressure_scale))
}

fn knocke_albedo_emissivity(epoch_tdb_seconds: f64, latitude_rad: f64) -> (f64, f64) {
    let jd = J2000_JD + epoch_tdb_seconds / SECONDS_PER_DAY;
    let seasonal_phase = std::f64::consts::TAU * (jd - KNOCKE_REFERENCE_JD) / DAYS_PER_JULIAN_YEAR;
    let (phase_sin, phase_cos) = libm::sincos(seasonal_phase);
    let p1 = libm::sin(latitude_rad);
    let p2 = 0.5 * (3.0 * p1 * p1 - 1.0);

    let a1 = ALBEDO_C0 + ALBEDO_C1 * phase_cos + ALBEDO_C2 * phase_sin;
    let e1 = EMISSIVITY_K0 + EMISSIVITY_K1 * phase_cos + EMISSIVITY_K2 * phase_sin;
    let albedo = ALBEDO_A0 + a1 * p1 + ALBEDO_A2 * p2;
    let emissivity = EMISSIVITY_E0 + e1 * p1 + EMISSIVITY_E2 * p2;
    (albedo, emissivity)
}

fn tangent_basis(
    radial_hat: Vector3<f64>,
) -> Result<(Vector3<f64>, Vector3<f64>), PropagationError> {
    let reference = if radial_hat.z.abs() < 0.9 {
        Vector3::z_axis().into_inner()
    } else {
        Vector3::x_axis().into_inner()
    };
    let east = reference.cross(&radial_hat);
    let east_norm = east.norm();
    if east_norm == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Cannot build Earth radiation tangent basis".to_string(),
        ));
    }
    let east_hat = east / east_norm;
    let north_hat = radial_hat.cross(&east_hat);
    Ok((east_hat, north_hat))
}

fn validate_state(state: &CartesianState) -> Result<(), PropagationError> {
    validate_finite("epoch_tdb_seconds", state.epoch_tdb_seconds)?;
    validate_vector(state.position_km, "position_km")?;
    validate_vector(state.velocity_km_s, "velocity_km_s")
}

fn validate_vector(vector: Vector3<f64>, field: &'static str) -> Result<(), PropagationError> {
    for component in [vector.x, vector.y, vector.z] {
        validate_finite(field, component)?;
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), PropagationError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be finite and positive"
        )));
    }
    Ok(())
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), PropagationError> {
    if !value.is_finite() {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be finite"
        )));
    }
    Ok(())
}

fn validate_latitude(latitude_rad: f64) -> Result<(), PropagationError> {
    validate_finite("latitude_rad", latitude_rad)?;
    if !(-MAX_LATITUDE_RAD..=MAX_LATITUDE_RAD).contains(&latitude_rad) {
        return Err(PropagationError::InvalidInput(
            "latitude_rad must be in [-pi/2, pi/2]".to_string(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Clean-room validation anchors:
    //!
    //! Knocke, Ries, and Tapley (1988) model the zonal albedo and emissivity as
    //! degree-2 Legendre expansions with seasonal first-degree terms referenced
    //! to 1981-12-22. The coefficients and solar momentum flux are pinned to the
    //! f64 encodings of the published decimals. The force equation is the
    //! Lambertian element sum used in GEODYN's KRT description, evaluated over
    //! the 19 equal apparent-area elements in the KRT cap plus two-ring layout.

    use super::*;
    use crate::astro::constants::earth::WGS84_A_KM;

    fn reference_epoch_seconds() -> f64 {
        (KNOCKE_REFERENCE_JD - J2000_JD) * SECONDS_PER_DAY
    }

    fn dayside_leo_state() -> CartesianState {
        let r = WGS84_A_KM + 700.0;
        CartesianState::new(0.0, [r, 0.0, 0.0], [0.0, 7.5, 0.0])
    }

    #[test]
    fn published_knocke_coefficients_match_f64_bits() {
        assert_eq!(KNOCKE_SOLAR_PRESSURE_N_M2.to_bits(), 0x3ed3_20e8_2eb2_57f2);
        assert_eq!(KNOCKE_REFERENCE_JD.to_bits(), 0x4142_a750_4000_0000);
        assert_eq!(ALBEDO_A0.to_bits(), 0x3fd5_c28f_5c28_f5c3);
        assert_eq!(ALBEDO_C0.to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(ALBEDO_C1.to_bits(), 0x3fb9_9999_9999_999a);
        assert_eq!(ALBEDO_C2.to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(ALBEDO_A2.to_bits(), 0x3fd2_8f5c_28f5_c28f);
        assert_eq!(EMISSIVITY_E0.to_bits(), 0x3fe5_c28f_5c28_f5c3);
        assert_eq!(EMISSIVITY_K0.to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(EMISSIVITY_K1.to_bits(), 0xbfb1_eb85_1eb8_51ec);
        assert_eq!(EMISSIVITY_K2.to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(EMISSIVITY_E2.to_bits(), 0xbfc7_0a3d_70a3_d70a);
    }

    #[test]
    fn reference_epoch_equator_matches_legendre_closed_form() {
        let model = EarthRadiationPressure::new(1.2, 0.011).expect("valid model");
        let (albedo, emissivity) = model
            .albedo_emissivity(reference_epoch_seconds(), 0.0)
            .expect("coefficients");

        assert_eq!(albedo.to_bits(), 0x3fc8_f5c2_8f5c_28f7);
        assert_eq!(emissivity.to_bits(), 0x3fe8_a3d7_0a3d_70a4);
    }

    #[test]
    fn dayside_leo_magnitude_matches_krt_order_and_golden_bits() {
        let model = EarthRadiationPressure::new(1.2, 0.011).expect("valid model");
        let state = dayside_leo_state();
        let sun = Vector3::new(AU_KM, 0.0, 0.0);
        let accel = earth_radiation_acceleration_with_sun(&state, sun, model)
            .expect("Earth radiation acceleration");

        assert_eq!(accel.x.to_bits(), 0x3db4_ea69_7dee_a111);
        assert!(accel.y.abs() <= accel.norm() * f64::EPSILON);
        let independent_z = f64::from_bits(0xbd50_e131_737f_badd);
        assert!(accel.z.to_bits().abs_diff(independent_z.to_bits()) <= 1);

        let magnitude_m_s2 = accel.norm() * M_PER_KM;
        assert!((1.0e-8..1.0e-7).contains(&magnitude_m_s2));
        assert_eq!(magnitude_m_s2.to_bits(), 0x3e54_6d55_70d2_55ce);
    }

    #[test]
    fn dayside_leo_sum_is_radially_dominated_outward() {
        let model = EarthRadiationPressure::new(1.2, 0.011).expect("valid model");
        let state = dayside_leo_state();
        let sun = Vector3::new(AU_KM, 0.0, 0.0);
        let accel = earth_radiation_acceleration_with_sun(&state, sun, model)
            .expect("Earth radiation acceleration");
        let radial_hat = state.position_km.normalize();

        assert!(accel.dot(&radial_hat) > 0.0);
        assert!(accel.dot(&radial_hat) / accel.norm() > 0.999);
    }

    #[test]
    fn rejects_spacecraft_inside_earth() {
        let model = EarthRadiationPressure::new(1.2, 0.011).expect("valid model");
        let state = CartesianState::new(0.0, [WGS84_A_KM, 0.0, 0.0], [0.0, 7.5, 0.0]);
        let err =
            earth_radiation_acceleration_with_sun(&state, Vector3::new(AU_KM, 0.0, 0.0), model)
                .expect_err("surface state must be rejected");

        assert!(matches!(err, PropagationError::InvalidInput(_)));
    }

    #[test]
    fn public_albedo_emissivity_rejects_invalid_latitude() {
        let model = EarthRadiationPressure::new(1.2, 0.011).expect("valid model");
        let err = model
            .albedo_emissivity(0.0, MAX_LATITUDE_RAD.next_up())
            .expect_err("latitude outside physical domain must be rejected");

        assert!(matches!(err, PropagationError::InvalidInput(_)));
    }

    #[test]
    fn moon_eclipse_is_opt_in_and_attenuates_reflected_albedo_only() {
        let default_model = EarthRadiationPressure::new(1.2, 0.011).expect("valid model");
        assert!(!default_model.moon_eclipse_enabled());
        let moon_model = default_model.with_moon_eclipse(true);
        assert!(moon_model.moon_eclipse_enabled());

        let state = dayside_leo_state();
        let sun = Vector3::new(AU_KM, 0.0, 0.0);
        let moon_clear = Vector3::new(0.0, 384_400.0, 0.0);
        let moon_eclipse = Vector3::new(356_000.0, 0.0, 0.0);

        // When the Moon is off-axis, moon-enabled albedo is bit-identical to default.
        let accel_default = default_model
            .acceleration_with_sun_moon(&state, sun, moon_clear)
            .expect("default albedo");
        let accel_clear = moon_model
            .acceleration_with_sun_moon(&state, sun, moon_clear)
            .expect("clear moon albedo");
        for i in 0..3 {
            assert_eq!(accel_clear[i].to_bits(), accel_default[i].to_bits());
        }

        // During a central solar eclipse over the sub-satellite cap, reflected
        // shortwave drops while emitted thermal IR keeps the acceleration positive.
        let accel_eclipse = moon_model
            .acceleration_with_sun_moon(&state, sun, moon_eclipse)
            .expect("lunar-eclipsed albedo");
        assert!(accel_eclipse.x > 0.0);
        assert!(accel_eclipse.x < accel_default.x);

        // Compare against nightside (pure thermal IR): eclipsed dayside must
        // still exceed pure IR because outer ring elements lie outside lunar umbra.
        let accel_night = default_model
            .acceleration_with_sun_moon(&state, -sun, moon_clear)
            .expect("nightside IR");
        assert!(accel_eclipse.x > accel_night.x);

        let err = moon_model
            .acceleration_with_sun_moon(&state, sun, Vector3::new(f64::INFINITY, 0.0, 0.0))
            .expect_err("non-finite moon rejected");
        assert!(matches!(err, PropagationError::InvalidInput(_)));
    }

    #[test]
    fn moon_eclipse_rejects_overflowing_moon_position_on_both_dayside_and_nightside() {
        let moon_model = EarthRadiationPressure::new(1.2, 0.011)
            .expect("valid model")
            .with_moon_eclipse(true);
        let sun = Vector3::new(AU_KM, 0.0, 0.0);
        let overflowing_moon = Vector3::new(1.0e200, 0.0, 0.0);

        let dayside = dayside_leo_state();
        let nightside =
            CartesianState::new(0.0, [-(WGS84_A_KM + 700.0), 0.0, 0.0], [0.0, 7.5, 0.0]);

        assert!(
            moon_model
                .acceleration_with_sun_moon(&dayside, sun, overflowing_moon)
                .is_err(),
            "dayside must reject overflowing moon position"
        );
        assert!(
            moon_model
                .acceleration_with_sun_moon(&nightside, sun, overflowing_moon)
                .is_err(),
            "nightside must also reject overflowing moon position"
        );
    }
}
