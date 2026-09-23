//! Solid Earth tide and solid Earth pole tide geopotential perturbations.
//!
//! These forces implement the low-degree time-variable geopotential corrections
//! from IERS Conventions (2010), Chapter 6, for numerical propagation:
//!
//! * [`SolidEarthTideGravity`] applies Chapter 6, Section 6.2.1, Step 1:
//!   frequency-independent anelastic Love-number corrections to fully
//!   normalized `Cnm`/`Snm` from the Sun and Moon. It includes degree 2, degree
//!   3, and the degree 4 corrections produced by degree-2 tides. Step 2
//!   frequency-dependent constituent corrections are not included here; they
//!   require the constituent argument tables and are a documented follow-up.
//! * [`SolidEarthPoleTideGravity`] applies the Chapter 6 solid Earth pole tide
//!   correction to normalized `C21` and `S21`, using the polar motion stored in
//!   the propagation context's Earth-orientation provider.
//!
//! Both models are perturbation-only additive forces. They require a
//! body-fixed frame provider in [`PropagationContext`] because the coefficients
//! are evaluated in the terrestrial frame and then rotated back to GCRF.

use crate::astro::bodies::sun_moon::sun_moon_ecef_with_polar_motion;
use crate::astro::constants::astro::{GM_MOON_KM3_S2, GM_SUN_KM3_S2};
use crate::astro::constants::time::J2000_JD;
use crate::astro::constants::units::{ARCSEC_TO_RAD, M_PER_KM};
use crate::astro::error::PropagationError;
use crate::astro::forces::geopotential::{
    SphericalHarmonicCoefficient, SphericalHarmonicGravity, EGM96_MU_KM3_S2,
    EGM96_REFERENCE_RADIUS_KM,
};
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::frames::orientation::EarthOrientation;
use crate::astro::frames::transforms::{with_ut1_validity, PolarMotion};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use crate::astro::time::scales::TimeScales;
use crate::astro::time::ValidityMode;
use nalgebra::Vector3;

const SOLID_TIDE_MAX_DEGREE: u16 = 4;
const SOLID_TIDE_MAX_ORDER: u16 = 3;
const POLE_TIDE_MAX_DEGREE: u16 = 2;
const POLE_TIDE_MAX_ORDER: u16 = 1;

/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Re k20`.
pub const SOLID_EARTH_TIDE_K20_REAL: f64 = 0.30190;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Im k20`.
pub const SOLID_EARTH_TIDE_K20_IMAG: f64 = 0.0;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `k20+`.
pub const SOLID_EARTH_TIDE_K20_PLUS: f64 = -0.00089;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Re k21`.
pub const SOLID_EARTH_TIDE_K21_REAL: f64 = 0.29830;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Im k21`.
pub const SOLID_EARTH_TIDE_K21_IMAG: f64 = -0.00144;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `k21+`.
pub const SOLID_EARTH_TIDE_K21_PLUS: f64 = -0.00080;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Re k22` = 0.30102.
pub const SOLID_EARTH_TIDE_K22_REAL: f64 = f64::from_bits(0x3fd3_43e9_63dc_486b);
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `Im k22`.
pub const SOLID_EARTH_TIDE_K22_IMAG: f64 = -0.00130;
/// IERS Conventions (2010), Chapter 6, Table 6.3 anelastic `k22+`.
pub const SOLID_EARTH_TIDE_K22_PLUS: f64 = -0.00057;
/// IERS Conventions (2010), Chapter 6, Table 6.3 `k30`.
pub const SOLID_EARTH_TIDE_K30_REAL: f64 = 0.093;
/// IERS Conventions (2010), Chapter 6, Table 6.3 `k31`.
pub const SOLID_EARTH_TIDE_K31_REAL: f64 = 0.093;
/// IERS Conventions (2010), Chapter 6, Table 6.3 `k32`.
pub const SOLID_EARTH_TIDE_K32_REAL: f64 = 0.093;
/// IERS Conventions (2010), Chapter 6, Table 6.3 `k33`.
pub const SOLID_EARTH_TIDE_K33_REAL: f64 = 0.094;

/// Chapter 6 solid Earth pole tide normalized-coefficient scale.
pub const SOLID_EARTH_POLE_TIDE_SCALE: f64 = -1.333e-9;
/// Chapter 6 solid Earth pole tide imaginary Love-number coupling.
pub const SOLID_EARTH_POLE_TIDE_IMAG_COUPLING: f64 = 0.0115;

#[derive(Debug, Clone, Copy, PartialEq)]
struct ComplexLoveNumber {
    real: f64,
    imag: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Degree2LoveNumber {
    primary: ComplexLoveNumber,
    plus: f64,
}

const DEGREE2_LOVE: [Degree2LoveNumber; 3] = [
    Degree2LoveNumber {
        primary: ComplexLoveNumber {
            real: SOLID_EARTH_TIDE_K20_REAL,
            imag: SOLID_EARTH_TIDE_K20_IMAG,
        },
        plus: SOLID_EARTH_TIDE_K20_PLUS,
    },
    Degree2LoveNumber {
        primary: ComplexLoveNumber {
            real: SOLID_EARTH_TIDE_K21_REAL,
            imag: SOLID_EARTH_TIDE_K21_IMAG,
        },
        plus: SOLID_EARTH_TIDE_K21_PLUS,
    },
    Degree2LoveNumber {
        primary: ComplexLoveNumber {
            real: SOLID_EARTH_TIDE_K22_REAL,
            imag: SOLID_EARTH_TIDE_K22_IMAG,
        },
        plus: SOLID_EARTH_TIDE_K22_PLUS,
    },
];

const DEGREE3_LOVE: [ComplexLoveNumber; 4] = [
    ComplexLoveNumber {
        real: SOLID_EARTH_TIDE_K30_REAL,
        imag: 0.0,
    },
    ComplexLoveNumber {
        real: SOLID_EARTH_TIDE_K31_REAL,
        imag: 0.0,
    },
    ComplexLoveNumber {
        real: SOLID_EARTH_TIDE_K32_REAL,
        imag: 0.0,
    },
    ComplexLoveNumber {
        real: SOLID_EARTH_TIDE_K33_REAL,
        imag: 0.0,
    },
];

/// Step-1 solid Earth tide geopotential perturbation force.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidEarthTideGravity {
    /// Earth gravitational parameter used by the correction potential, km^3/s^2.
    pub mu_earth_km3_s2: f64,
    /// Reference equatorial radius used by the correction potential, km.
    pub reference_radius_km: f64,
    /// Solar gravitational parameter, km^3/s^2.
    pub gm_sun_km3_s2: f64,
    /// Lunar gravitational parameter, km^3/s^2.
    pub gm_moon_km3_s2: f64,
}

impl Default for SolidEarthTideGravity {
    fn default() -> Self {
        Self {
            mu_earth_km3_s2: EGM96_MU_KM3_S2,
            reference_radius_km: EGM96_REFERENCE_RADIUS_KM,
            gm_sun_km3_s2: GM_SUN_KM3_S2,
            gm_moon_km3_s2: GM_MOON_KM3_S2,
        }
    }
}

impl SolidEarthTideGravity {
    /// Build with explicit Earth, Sun, and Moon parameters.
    pub fn new(
        mu_earth_km3_s2: f64,
        reference_radius_km: f64,
        gm_sun_km3_s2: f64,
        gm_moon_km3_s2: f64,
    ) -> Self {
        Self {
            mu_earth_km3_s2,
            reference_radius_km,
            gm_sun_km3_s2,
            gm_moon_km3_s2,
        }
    }

    /// Coefficient corrections from already body-fixed Sun and Moon positions.
    pub fn coefficient_corrections_for_body_fixed_bodies(
        &self,
        sun_itrf_km: [f64; 3],
        moon_itrf_km: [f64; 3],
    ) -> Result<[SphericalHarmonicCoefficient; 10], PropagationError> {
        validate_positive(self.mu_earth_km3_s2, "mu_earth_km3_s2")?;
        validate_positive(self.reference_radius_km, "reference_radius_km")?;
        validate_positive(self.gm_sun_km3_s2, "gm_sun_km3_s2")?;
        validate_positive(self.gm_moon_km3_s2, "gm_moon_km3_s2")?;

        let mut corrections = empty_solid_tide_coefficients();
        add_body_tide_coefficients(self, self.gm_sun_km3_s2, sun_itrf_km, &mut corrections)?;
        add_body_tide_coefficients(self, self.gm_moon_km3_s2, moon_itrf_km, &mut corrections)?;
        Ok(corrections)
    }

    /// Coefficient corrections at an epoch using the context's body-fixed frame provider.
    pub fn coefficient_corrections_at_epoch(
        &self,
        epoch_tdb_seconds: f64,
        ctx: &PropagationContext,
    ) -> Result<[SphericalHarmonicCoefficient; 10], PropagationError> {
        let orientation = orientation_at_state(ctx, epoch_tdb_seconds)?;
        let bodies = sun_moon_itrf_km(&orientation)?;
        self.coefficient_corrections_for_body_fixed_bodies(bodies.sun_itrf_km, bodies.moon_itrf_km)
    }
}

impl ForceModel for SolidEarthTideGravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let orientation = orientation_at_state(ctx, state.epoch_tdb_seconds)?;
        let position_itrf_km = orientation
            .gcrf_to_itrf_position_km(state.position_array())
            .map_err(|error| {
                PropagationError::ForceModelFailure(format!(
                    "solid Earth tide body-fixed position rotation failed: {error}"
                ))
            })?;
        let bodies = sun_moon_itrf_km(&orientation)?;
        let corrections = self.coefficient_corrections_for_body_fixed_bodies(
            bodies.sun_itrf_km,
            bodies.moon_itrf_km,
        )?;
        let tide = SphericalHarmonicGravity::from_normalized_coefficients(
            self.mu_earth_km3_s2,
            self.reference_radius_km,
            SOLID_TIDE_MAX_DEGREE,
            SOLID_TIDE_MAX_ORDER,
            &corrections,
        )?;
        let accel_itrf = tide.body_fixed_acceleration_km_s2(position_itrf_km)?;
        let accel_gcrf = orientation
            .itrf_to_gcrf_position_km(accel_itrf)
            .map_err(|error| {
                PropagationError::ForceModelFailure(format!(
                    "solid Earth tide inertial acceleration rotation failed: {error}"
                ))
            })?;
        Ok(Vector3::new(accel_gcrf[0], accel_gcrf[1], accel_gcrf[2]))
    }
}

/// Solid Earth pole tide geopotential perturbation force.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolidEarthPoleTideGravity {
    /// Earth gravitational parameter used by the correction potential, km^3/s^2.
    pub mu_earth_km3_s2: f64,
    /// Reference equatorial radius used by the correction potential, km.
    pub reference_radius_km: f64,
}

impl Default for SolidEarthPoleTideGravity {
    fn default() -> Self {
        Self {
            mu_earth_km3_s2: EGM96_MU_KM3_S2,
            reference_radius_km: EGM96_REFERENCE_RADIUS_KM,
        }
    }
}

impl SolidEarthPoleTideGravity {
    /// Build with explicit Earth gravity parameters.
    pub fn new(mu_earth_km3_s2: f64, reference_radius_km: f64) -> Self {
        Self {
            mu_earth_km3_s2,
            reference_radius_km,
        }
    }

    /// Coefficient correction from time scales and polar motion.
    pub fn coefficient_correction(
        &self,
        time_scales: TimeScales,
        polar_motion: PolarMotion,
    ) -> Result<SphericalHarmonicCoefficient, PropagationError> {
        validate_positive(self.mu_earth_km3_s2, "mu_earth_km3_s2")?;
        validate_positive(self.reference_radius_km, "reference_radius_km")?;
        if !polar_motion.xp_rad.is_finite() || !polar_motion.yp_rad.is_finite() {
            return Err(PropagationError::InvalidInput(
                "polar motion components must be finite".to_string(),
            ));
        }
        let (m1, m2) = wobble_arcsec(time_scales, polar_motion)?;
        Ok(SphericalHarmonicCoefficient {
            degree: 2,
            order: 1,
            c: SOLID_EARTH_POLE_TIDE_SCALE * (m1 + SOLID_EARTH_POLE_TIDE_IMAG_COUPLING * m2),
            s: SOLID_EARTH_POLE_TIDE_SCALE * (m2 - SOLID_EARTH_POLE_TIDE_IMAG_COUPLING * m1),
        })
    }
}

impl ForceModel for SolidEarthPoleTideGravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let orientation = orientation_at_state(ctx, state.epoch_tdb_seconds)?;
        let position_itrf_km = orientation
            .gcrf_to_itrf_position_km(state.position_array())
            .map_err(|error| {
                PropagationError::ForceModelFailure(format!(
                    "solid Earth pole tide body-fixed position rotation failed: {error}"
                ))
            })?;
        let correction =
            self.coefficient_correction(orientation.time_scales(), orientation.polar_motion())?;
        let tide = SphericalHarmonicGravity::from_normalized_coefficients(
            self.mu_earth_km3_s2,
            self.reference_radius_km,
            POLE_TIDE_MAX_DEGREE,
            POLE_TIDE_MAX_ORDER,
            &[correction],
        )?;
        let accel_itrf = tide.body_fixed_acceleration_km_s2(position_itrf_km)?;
        let accel_gcrf = orientation
            .itrf_to_gcrf_position_km(accel_itrf)
            .map_err(|error| {
                PropagationError::ForceModelFailure(format!(
                    "solid Earth pole tide inertial acceleration rotation failed: {error}"
                ))
            })?;
        Ok(Vector3::new(accel_gcrf[0], accel_gcrf[1], accel_gcrf[2]))
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SunMoonItrfKm {
    sun_itrf_km: [f64; 3],
    moon_itrf_km: [f64; 3],
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BodyGeometry {
    radius_km: f64,
    cos_m: [f64; 4],
    sin_m: [f64; 4],
    p2: [f64; 3],
    p3: [f64; 4],
}

fn orientation_at_state(
    ctx: &PropagationContext,
    epoch_tdb_seconds: f64,
) -> Result<EarthOrientation, PropagationError> {
    let provider = ctx.body_fixed_frame_provider().ok_or_else(|| {
        PropagationError::InvalidInput(
            "solid Earth tide geopotential forces require a body-fixed frame provider".to_string(),
        )
    })?;
    provider
        .orientation_at_tdb_seconds(epoch_tdb_seconds)
        .map(|orientation| ctx.record_orientation(orientation))
        .map_err(|error| {
            PropagationError::from_frame(
                "solid Earth tide body-fixed frame evaluation failed",
                error,
            )
        })
}

fn sun_moon_itrf_km(orientation: &EarthOrientation) -> Result<SunMoonItrfKm, PropagationError> {
    // The orientation's provider already applied the UT1 policy and the
    // context recorded any departure, so its time scales are accepted as is.
    let bodies = with_ut1_validity(&orientation.time_scales(), ValidityMode::Permissive, |ts| {
        sun_moon_ecef_with_polar_motion(ts, orientation.polar_motion())
    })
    .map(|validated| validated.value)
    .map_err(|error| PropagationError::ForceModelFailure(format!("Sun/Moon: {error}")))?;
    Ok(SunMoonItrfKm {
        sun_itrf_km: meters_to_km(bodies.sun),
        moon_itrf_km: meters_to_km(bodies.moon),
    })
}

fn meters_to_km(position_m: [f64; 3]) -> [f64; 3] {
    [
        position_m[0] / M_PER_KM,
        position_m[1] / M_PER_KM,
        position_m[2] / M_PER_KM,
    ]
}

fn empty_solid_tide_coefficients() -> [SphericalHarmonicCoefficient; 10] {
    [
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 0,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 1,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 2,
            order: 2,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 3,
            order: 0,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 3,
            order: 1,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 3,
            order: 2,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 3,
            order: 3,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 4,
            order: 0,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 4,
            order: 1,
            c: 0.0,
            s: 0.0,
        },
        SphericalHarmonicCoefficient {
            degree: 4,
            order: 2,
            c: 0.0,
            s: 0.0,
        },
    ]
}

fn add_body_tide_coefficients(
    model: &SolidEarthTideGravity,
    gm_body_km3_s2: f64,
    body_itrf_km: [f64; 3],
    corrections: &mut [SphericalHarmonicCoefficient; 10],
) -> Result<(), PropagationError> {
    let geometry = body_geometry(body_itrf_km)?;
    let gm_ratio = gm_body_km3_s2 / model.mu_earth_km3_s2;

    for order in 0..=2 {
        let love = DEGREE2_LOVE[order as usize];
        let base = gm_ratio
            * (model.reference_radius_km / geometry.radius_km).powi(3)
            * geometry.p2[order as usize]
            / 5.0;
        let (dc, ds) = coefficient_delta(
            base,
            love.primary,
            geometry.cos_m[order as usize],
            geometry.sin_m[order as usize],
        );
        let idx = order as usize;
        corrections[idx].c += dc;
        corrections[idx].s += ds;

        let dc4 = base * love.plus * geometry.cos_m[order as usize];
        let ds4 = base * love.plus * geometry.sin_m[order as usize];
        let idx4 = 7 + order as usize;
        corrections[idx4].c += dc4;
        corrections[idx4].s += ds4;
    }

    for order in 0..=3 {
        let love = DEGREE3_LOVE[order as usize];
        let base = gm_ratio
            * (model.reference_radius_km / geometry.radius_km).powi(4)
            * geometry.p3[order as usize]
            / 7.0;
        let (dc, ds) = coefficient_delta(
            base,
            love,
            geometry.cos_m[order as usize],
            geometry.sin_m[order as usize],
        );
        let idx = 3 + order as usize;
        corrections[idx].c += dc;
        corrections[idx].s += ds;
    }

    Ok(())
}

fn coefficient_delta(
    base: f64,
    love: ComplexLoveNumber,
    cos_m_lambda: f64,
    sin_m_lambda: f64,
) -> (f64, f64) {
    (
        base * (love.real * cos_m_lambda + love.imag * sin_m_lambda),
        base * (love.real * sin_m_lambda - love.imag * cos_m_lambda),
    )
}

fn body_geometry(position_km: [f64; 3]) -> Result<BodyGeometry, PropagationError> {
    validate_vec3(position_km, "body position")?;
    let x = position_km[0];
    let y = position_km[1];
    let z = position_km[2];
    let rho2 = x * x + y * y;
    let radius2 = rho2 + z * z;
    if radius2 == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "zero tide-raising body position magnitude".to_string(),
        ));
    }
    let radius_km = radius2.sqrt();
    let rho = rho2.sqrt();
    let sin_lat = z / radius_km;
    let cos_lat = rho / radius_km;
    let (cos_m, sin_m) = longitude_trig(x, y, rho);
    let p2 = degree2_legendre(sin_lat, cos_lat);
    let p3 = degree3_legendre(sin_lat, cos_lat);
    Ok(BodyGeometry {
        radius_km,
        cos_m,
        sin_m,
        p2,
        p3,
    })
}

fn longitude_trig(x: f64, y: f64, rho: f64) -> ([f64; 4], [f64; 4]) {
    let mut cos_m = [0.0_f64; 4];
    let mut sin_m = [0.0_f64; 4];
    cos_m[0] = 1.0;
    if rho == 0.0 {
        cos_m[1] = 1.0;
    } else {
        cos_m[1] = x / rho;
        sin_m[1] = y / rho;
    }
    for order in 2..=3 {
        cos_m[order] = cos_m[order - 1] * cos_m[1] - sin_m[order - 1] * sin_m[1];
        sin_m[order] = sin_m[order - 1] * cos_m[1] + cos_m[order - 1] * sin_m[1];
    }
    (cos_m, sin_m)
}

fn degree2_legendre(sin_lat: f64, cos_lat: f64) -> [f64; 3] {
    let sqrt5 = 5.0_f64.sqrt();
    let sqrt15 = 15.0_f64.sqrt();
    [
        0.5 * sqrt5 * (3.0 * sin_lat * sin_lat - 1.0),
        sqrt15 * sin_lat * cos_lat,
        0.5 * sqrt15 * cos_lat * cos_lat,
    ]
}

fn degree3_legendre(sin_lat: f64, cos_lat: f64) -> [f64; 4] {
    let sqrt7 = 7.0_f64.sqrt();
    let sqrt42 = 42.0_f64.sqrt();
    let sqrt70 = 70.0_f64.sqrt();
    let sqrt105 = 105.0_f64.sqrt();
    let sin2 = sin_lat * sin_lat;
    let cos2 = cos_lat * cos_lat;
    [
        0.5 * sqrt7 * (5.0 * sin_lat * sin2 - 3.0 * sin_lat),
        0.25 * sqrt42 * cos_lat * (5.0 * sin2 - 1.0),
        0.5 * sqrt105 * sin_lat * cos2,
        0.25 * sqrt70 * cos_lat * cos2,
    ]
}

fn wobble_arcsec(
    time_scales: TimeScales,
    polar_motion: PolarMotion,
) -> Result<(f64, f64), PropagationError> {
    let xp_arcsec = polar_motion.xp_rad / ARCSEC_TO_RAD;
    let yp_arcsec = polar_motion.yp_rad / ARCSEC_TO_RAD;
    let (mean_x_arcsec, mean_y_arcsec) = conventional_mean_pole_arcsec(time_scales)?;
    Ok((xp_arcsec - mean_x_arcsec, -(yp_arcsec - mean_y_arcsec)))
}

fn conventional_mean_pole_arcsec(time_scales: TimeScales) -> Result<(f64, f64), PropagationError> {
    if !time_scales.jd_tt.is_finite() {
        return Err(PropagationError::InvalidInput(
            "time_scales.jd_tt must be finite".to_string(),
        ));
    }
    let year = 2000.0 + (time_scales.jd_tt - J2000_JD) / 365.25;
    let dt = year - 2000.0;
    let (x_mas, y_mas) = if year <= 2010.0 {
        (
            55.974 + dt * (1.8243 + dt * (0.18413 + dt * 0.007024)),
            346.346 + dt * (1.7896 + dt * (-0.10729 - dt * 0.000908)),
        )
    } else {
        (23.513 + dt * 7.6141, 358.891 - dt * 0.6287)
    };
    Ok((x_mas / 1000.0, y_mas / 1000.0))
}

fn validate_positive(value: f64, field: &'static str) -> Result<(), PropagationError> {
    if !value.is_finite() {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be finite"
        )));
    }
    if value <= 0.0 {
        return Err(PropagationError::InvalidInput(format!(
            "{field} must be positive"
        )));
    }
    Ok(())
}

fn validate_vec3(values: [f64; 3], field: &'static str) -> Result<(), PropagationError> {
    for value in values {
        if !value.is_finite() {
            return Err(PropagationError::InvalidInput(format!(
                "{field} components must be finite"
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::frames::orientation::TdbEarthOrientationProvider;
    use crate::astro::time::scales::TimeScales;
    use serde::Deserialize;
    use std::sync::Arc;

    const COEFFICIENT_FIXTURE: &str =
        include_str!("../../../tests/fixtures/tides/geopotential_tide_coefficients.json");

    #[derive(Debug, Deserialize)]
    struct CoefficientFixture {
        rows: Vec<FixtureRow>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureRow {
        id: String,
        source: String,
        sun_itrf_km: Option<[f64; 3]>,
        moon_itrf_km: Option<[f64; 3]>,
        earth_mu_km3_s2: Option<f64>,
        earth_radius_km: Option<f64>,
        gm_sun_km3_s2: Option<f64>,
        gm_moon_km3_s2: Option<f64>,
        jd_tt: Option<f64>,
        xp_arcsec: Option<f64>,
        yp_arcsec: Option<f64>,
        utc: Option<String>,
        max_abs_error: Option<f64>,
        coefficients: Vec<FixtureCoefficient>,
    }

    #[derive(Debug, Deserialize)]
    struct FixtureCoefficient {
        degree: u16,
        order: u16,
        c: f64,
        s: f64,
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.17e}, expected {expected:.17e}, tolerance {tolerance:.17e}"
        );
    }

    fn coefficient(
        coefficients: &[SphericalHarmonicCoefficient],
        degree: u16,
        order: u16,
    ) -> &SphericalHarmonicCoefficient {
        coefficients
            .iter()
            .find(|coefficient| coefficient.degree == degree && coefficient.order == order)
            .expect("coefficient present")
    }

    fn fixture_row(id: &str) -> FixtureRow {
        let fixture: CoefficientFixture =
            serde_json::from_str(COEFFICIENT_FIXTURE).expect("parse coefficient fixture");
        fixture
            .rows
            .into_iter()
            .find(|row| row.id == id)
            .expect("fixture row present")
    }

    #[test]
    fn love_numbers_match_iers_table_6_3() {
        assert_eq!(SOLID_EARTH_TIDE_K20_REAL.to_bits(), 0.30190_f64.to_bits());
        assert_eq!(SOLID_EARTH_TIDE_K20_IMAG.to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            SOLID_EARTH_TIDE_K20_PLUS.to_bits(),
            (-0.00089_f64).to_bits()
        );
        assert_eq!(SOLID_EARTH_TIDE_K21_REAL.to_bits(), 0.29830_f64.to_bits());
        assert_eq!(
            SOLID_EARTH_TIDE_K21_IMAG.to_bits(),
            (-0.00144_f64).to_bits()
        );
        assert_eq!(
            SOLID_EARTH_TIDE_K21_PLUS.to_bits(),
            (-0.00080_f64).to_bits()
        );
        let expected_k22_real: f64 = "0.30102".parse().expect("published k22 decimal");
        assert_eq!(
            SOLID_EARTH_TIDE_K22_REAL.to_bits(),
            expected_k22_real.to_bits()
        );
        assert_eq!(
            SOLID_EARTH_TIDE_K22_IMAG.to_bits(),
            (-0.00130_f64).to_bits()
        );
        assert_eq!(
            SOLID_EARTH_TIDE_K22_PLUS.to_bits(),
            (-0.00057_f64).to_bits()
        );
        assert_eq!(SOLID_EARTH_TIDE_K30_REAL.to_bits(), 0.093_f64.to_bits());
        assert_eq!(SOLID_EARTH_TIDE_K31_REAL.to_bits(), 0.093_f64.to_bits());
        assert_eq!(SOLID_EARTH_TIDE_K32_REAL.to_bits(), 0.093_f64.to_bits());
        assert_eq!(SOLID_EARTH_TIDE_K33_REAL.to_bits(), 0.094_f64.to_bits());
    }

    #[test]
    fn step1_axis_aligned_body_matches_iers_equation_6_6_fixture() {
        let row = fixture_row("solid_earth_step1_greenwich_equator_unit_body");
        assert!(row.source.contains("Eq. (6.6)"));
        let model = SolidEarthTideGravity::new(
            row.earth_mu_km3_s2.expect("earth mu"),
            row.earth_radius_km.expect("earth radius"),
            row.gm_sun_km3_s2.expect("sun gm"),
            row.gm_moon_km3_s2.expect("moon gm"),
        );
        let coefficients = model
            .coefficient_corrections_for_body_fixed_bodies(
                row.sun_itrf_km.expect("sun position"),
                row.moon_itrf_km.expect("moon position"),
            )
            .expect("coefficient corrections");

        for expected in row.coefficients {
            let actual = coefficient(&coefficients, expected.degree, expected.order);
            assert_close(actual.c, expected.c, 2.0e-17);
            assert_close(actual.s, expected.s, 2.0e-17);
        }
    }

    #[test]
    fn pole_tide_coefficients_match_iers_chapter_6_equation() {
        let row = fixture_row("solid_earth_pole_tide_j2000_unit_wobble");
        assert!(row.source.contains("Table 7.7"));
        let time_scales = TimeScales {
            jd_whole: row.jd_tt.expect("jd_tt"),
            ut1_fraction: 0.0,
            tt_fraction: 0.0,
            tdb_fraction: 0.0,
            jd_ut1: row.jd_tt.expect("jd_tt"),
            jd_tt: row.jd_tt.expect("jd_tt"),
            jd_tdb: row.jd_tt.expect("jd_tt"),
            ut1_degraded: None,
        };
        let pole =
            PolarMotion::from_arcseconds(row.xp_arcsec.expect("xp"), row.yp_arcsec.expect("yp"))
                .expect("polar motion");
        let correction = SolidEarthPoleTideGravity::default()
            .coefficient_correction(time_scales, pole)
            .expect("pole tide coefficient");
        let expected = row.coefficients.first().expect("pole tide coefficient");

        assert_eq!(correction.degree, expected.degree);
        assert_eq!(correction.order, expected.order);
        assert_close(correction.c, expected.c, 1.0e-18);
        assert_close(correction.s, expected.s, 1.0e-18);
    }

    #[test]
    fn mean_pole_model_is_boundary_pinned_to_iers_table_7_7() {
        let at_2010 = TimeScales {
            jd_whole: J2000_JD + 10.0 * 365.25,
            ut1_fraction: 0.0,
            tt_fraction: 0.0,
            tdb_fraction: 0.0,
            jd_ut1: J2000_JD + 10.0 * 365.25,
            jd_tt: J2000_JD + 10.0 * 365.25,
            jd_tdb: J2000_JD + 10.0 * 365.25,
            ut1_degraded: None,
        };
        let after_2010 = TimeScales {
            jd_whole: J2000_JD + 11.0 * 365.25,
            ut1_fraction: 0.0,
            tt_fraction: 0.0,
            tdb_fraction: 0.0,
            jd_ut1: J2000_JD + 11.0 * 365.25,
            jd_tt: J2000_JD + 11.0 * 365.25,
            jd_tdb: J2000_JD + 11.0 * 365.25,
            ut1_degraded: None,
        };

        let (x_2010, y_2010) = conventional_mean_pole_arcsec(at_2010).expect("mean pole");
        let (x_2011, y_2011) = conventional_mean_pole_arcsec(after_2010).expect("mean pole");

        assert_close(x_2010, 0.099_654, 1.0e-15);
        assert_close(y_2010, 0.352_605, 1.0e-15);
        assert_close(x_2011, 0.107_268_1, 1.0e-15);
        assert_close(y_2011, 0.351_975_3, 1.0e-15);
    }

    #[test]
    fn step1_real_epoch_degree3_degree4_match_orekit_oracle() {
        let row = fixture_row("solid_earth_step1_orekit_2003_05_06_degree3_degree4");
        assert!(row.source.contains("Orekit"));
        assert_eq!(row.utc.as_deref(), Some("2003-05-06T13:43:32.125Z"));
        let scales =
            TimeScales::from_utc(2003, 5, 6, 13, 43, 32.125).expect("valid UTC time scales");
        let epoch_tdb_seconds = crate::astro::time::civil::j2000_seconds_from_split(
            scales.jd_whole,
            scales.tdb_fraction,
        );
        let ctx = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::default()));
        let coefficients = SolidEarthTideGravity::default()
            .coefficient_corrections_at_epoch(epoch_tdb_seconds, &ctx)
            .expect("coefficient corrections");
        let tolerance = row.max_abs_error.expect("max_abs_error");

        for expected in row.coefficients {
            let actual = coefficient(&coefficients, expected.degree, expected.order);
            assert_close(actual.c, expected.c, tolerance);
            assert_close(actual.s, expected.s, tolerance);
        }
    }

    #[test]
    fn tide_forces_require_body_fixed_frame_provider() {
        let state = CartesianState::new(0.0, [7000.0, 100.0, 30.0], [0.0, 7.5, 0.0]);
        let error = SolidEarthTideGravity::default()
            .acceleration(&state, &PropagationContext::default())
            .expect_err("missing provider fails");
        assert!(format!("{error}").contains("body-fixed frame provider"));

        let ctx = PropagationContext::new()
            .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::default()));
        SolidEarthTideGravity::default()
            .acceleration(&state, &ctx)
            .expect("solid tide acceleration");
        SolidEarthPoleTideGravity::default()
            .acceleration(&state, &ctx)
            .expect("pole tide acceleration");
    }
}
