//! Ground-site observation convenience helpers for the Sun and Moon.
//!
//! These round out the "observe the sky from a site" lens by wiring the
//! analytic Sun/Moon ephemeris ([`crate::astro::bodies::sun_moon::sun_moon_ecef`])
//! into the existing observation geometry rather than re-deriving any math:
//!
//! - Topocentric azimuth/elevation/range of the Sun or Moon from a ground site
//!   reuses the station-to-target ENU reduction
//!   [`crate::astro::frames::transforms::itrs_to_topocentric`] (the same one the
//!   satellite look-angle path uses).
//! - The Moon's illuminated fraction reuses the Sun-target-observer phase angle
//!   [`crate::astro::angles::phase_angle`] that the satellite visual-magnitude and
//!   eclipse geometry already consume.
//!
//! Precision follows the underlying analytic series (sub-degree positions, see
//! [`crate::astro::bodies::sun_moon`]); this is a planning/visualization lens, not
//! an almanac-grade reduction.

use crate::astro::angles::{phase_angle, AngleError};
use crate::astro::bodies::sun_moon::{sun_moon_ecef, sun_moon_eci_at, SunMoonError};
use crate::astro::constants::astro::GM_SUN_KM3_S2;
use crate::astro::constants::earth::OMEGA_E_DOT_RAD_S;
use crate::astro::constants::physics::SPEED_OF_LIGHT_M_S;
use crate::astro::constants::time::{J2000_JD, SECONDS_PER_DAY};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::frames::nutation::{
    build_skyfield_nutation_matrix, skyfield_iau2000a_radians, skyfield_mean_obliquity_radians,
};
use crate::astro::frames::transforms::{
    gcrs_to_itrs_matrix, gcrs_to_itrs_matrix_with_polar_motion, gcrs_to_true_of_date_matrix,
    geodetic_to_itrs, greenwich_apparent_sidereal_time_radians, itrs_to_gcrs_compute,
    itrs_to_gcrs_compute_with_polar_motion, itrs_to_gcrs_matrix,
    itrs_to_gcrs_matrix_with_polar_motion, itrs_to_topocentric, mat3_vec3_mul_unchecked,
    FrameTransformError, GeodeticStationKm, PolarMotion,
};
use crate::astro::math::vec3::{add3, dot3, norm3, scale3, sub3, unit3};
use crate::astro::passes::UtcInstant;
use crate::astro::spk::{Spk, SpkError, SpkState};
use crate::astro::time::scales::TimeScales;

const C_KM_S: f64 = SPEED_OF_LIGHT_M_S / M_PER_KM;
const LIGHT_TIME_TOLERANCE_S: f64 = 1.0e-9;
const LIGHT_TIME_REFINEMENTS: usize = 3;
const SPK_FRAME_J2000: i32 = 1;
const NAIF_SSB: i32 = 0;
const NAIF_SUN: i32 = 10;
const NAIF_MOON: i32 = 301;
const NAIF_EARTH: i32 = 399;
const SOLAR_DEFLECTION_DENOM_FLOOR: f64 = 1.0e-6;

/// Error returned by ground-site Sun/Moon observation helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum BodyObservationError {
    /// The analytic Sun/Moon ephemeris rejected the instant or produced a
    /// non-finite vector.
    #[error("Sun/Moon ephemeris failed: {0}")]
    Ephemeris(#[from] SunMoonError),
    /// The station-to-body topocentric reduction rejected an input.
    #[error("topocentric reduction failed: {0}")]
    FrameTransform(#[from] FrameTransformError),
    /// The phase-angle geometry rejected an input (e.g. a degenerate vector).
    #[error("phase-angle geometry failed: {0}")]
    Angle(#[from] AngleError),
}

/// Topocentric look angle of a body from a ground site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyAzEl {
    /// Azimuth, degrees clockwise from north on `[0, 360)`.
    pub azimuth_deg: f64,
    /// Elevation above the local horizon, degrees on `[-90, 90]`.
    pub elevation_deg: f64,
    /// Slant range from the site to the body, kilometres.
    pub range_km: f64,
}

/// The Moon's illuminated state as seen from a ground site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonIllumination {
    /// Sunlit fraction of the lunar disk on `[0, 1]` (0 = new, 1 = full).
    pub illuminated_fraction: f64,
    /// Sun-Moon-observer phase angle, degrees on `[0, 180]` (0 = full).
    pub phase_angle_deg: f64,
}

/// Equatorial spherical coordinates with distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Equatorial {
    /// Right ascension in degrees on `[0, 360)`.
    pub right_ascension_deg: f64,
    /// Right ascension in hours on `[0, 24)`.
    pub right_ascension_hours: f64,
    /// Declination in degrees on `[-90, 90]`.
    pub declination_deg: f64,
    /// Distance in kilometers.
    pub distance_km: f64,
}

/// Horizontal topocentric look angle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Horizontal {
    /// Azimuth in degrees clockwise from north on `[0, 360)`.
    pub azimuth_deg: f64,
    /// Elevation in degrees. Refraction-corrected only when requested.
    pub elevation_deg: f64,
    /// Topocentric range in kilometers.
    pub range_km: f64,
}

/// Ecliptic spherical coordinates, true ecliptic and equinox of date.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ecliptic {
    /// Ecliptic longitude in degrees on `[0, 360)`.
    pub longitude_deg: f64,
    /// Ecliptic latitude in degrees on `[-90, 90]`.
    pub latitude_deg: f64,
    /// Distance in kilometers.
    pub distance_km: f64,
}

/// Full topocentric observation of a target from a ground site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observation {
    /// Astrometric place, light-time only, on ICRS axes for full-chain targets.
    ///
    /// Reduced Sun/Moon analytic variants carry mean-of-date RA/Dec here, not an
    /// ICRS astrometric place. See [`Observation::reduced`].
    pub astrometric: Equatorial,
    /// Apparent place on ICRF axes before the true-of-date rotation.
    pub apparent_icrs: Equatorial,
    /// Apparent place, true equator and equinox of date.
    pub apparent: Equatorial,
    /// Apparent horizontal coordinates.
    pub horizontal: Horizontal,
    /// Local apparent hour angle in degrees on `(-180, 180]`.
    pub hour_angle_deg: f64,
    /// Local apparent hour angle in hours on `(-12, 12]`.
    pub hour_angle_hours: f64,
    /// Apparent ecliptic coordinates of date.
    pub ecliptic: Ecliptic,
    /// True for the reduced analytic Sun/Moon variants.
    pub reduced: bool,
}

/// Optional Bennett atmospheric-refraction inputs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Refraction {
    /// Pressure in millibars.
    pub pressure_mbar: f64,
    /// Temperature in degrees Celsius.
    pub temperature_c: f64,
}

/// Options for general ground-site observation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObserveOptions {
    /// Optional polar-motion coordinates. `None` uses zero pole coordinates.
    pub polar_motion: Option<PolarMotion>,
    /// Optional output-side Bennett refraction for elevation only.
    pub refraction: Option<Refraction>,
    /// Apply solar gravitational light deflection on the full-chain path.
    pub deflection: bool,
    /// Apply annual plus diurnal aberration on the full-chain path.
    pub aberration: bool,
}

impl Default for ObserveOptions {
    fn default() -> Self {
        Self {
            polar_motion: None,
            refraction: None,
            deflection: true,
            aberration: true,
        }
    }
}

/// What to observe.
#[derive(Debug, Clone, Copy)]
pub enum Target<'a> {
    /// SPK-covered body by NAIF id.
    Spk {
        /// Kernel used for the target, Earth, and Sun state queries.
        kernel: &'a Spk,
        /// NAIF identifier of the target body queried relative to the SSB.
        naif_id: i32,
    },
    /// Analytic low-precision Sun, no kernel required.
    Sun,
    /// Analytic low-precision Moon, no kernel required.
    Moon,
    /// Caller-supplied SSB-centered ICRS/J2000 state plus a kernel for Earth/Sun.
    BarycentricState {
        /// Kernel used for the Earth barycentric state and solar-deflection Sun state.
        kernel: &'a Spk,
        /// Target position at the observation epoch, in SSB-centered ICRS/J2000 kilometers.
        ///
        /// The light-time reducer advances this position with `velocity_km_s`.
        position_km: [f64; 3],
        /// Target velocity in SSB-centered ICRS/J2000 kilometers per second.
        ///
        /// The light-time reducer treats this velocity as constant during its iteration.
        velocity_km_s: [f64; 3],
    },
}

/// Error returned by general ground-site observation.
#[derive(Debug, Clone, thiserror::Error)]
pub enum ObserveError {
    /// SPK state evaluation failed.
    #[error("SPK state failed: {0}")]
    Spk(#[from] SpkError),
    /// Frame transformation failed.
    #[error("frame transform failed: {0}")]
    FrameTransform(#[from] FrameTransformError),
    /// Analytic Sun/Moon ephemeris failed.
    #[error("Sun/Moon ephemeris failed: {0}")]
    SunMoon(#[from] SunMoonError),
    /// Angle geometry failed.
    #[error("angle geometry failed: {0}")]
    Angle(#[from] AngleError),
    /// SPK state used a reference frame this reduction does not support.
    #[error("unsupported SPK reference frame {frame}")]
    UnsupportedSpkFrame {
        /// NAIF reference-frame identifier returned by the SPK state.
        frame: i32,
    },
    /// A public input or computed intermediate was not finite.
    #[error("non-finite observation input or intermediate")]
    NonFinite,
    /// The requested geometry collapsed to a zero-length direction.
    #[error("degenerate observation geometry")]
    DegenerateGeometry,
}

/// General topocentric observation.
pub fn observe(
    station: &GeodeticStationKm,
    time: UtcInstant,
    target: Target<'_>,
    options: ObserveOptions,
) -> Result<Observation, ObserveError> {
    observe_with_time_scales(station, &time.time_scales(), target, options)
}

/// Convenience wrapper for observing an SPK body with default options.
pub fn observe_spk_body(
    station: &GeodeticStationKm,
    time: UtcInstant,
    kernel: &Spk,
    naif_id: i32,
) -> Result<Observation, ObserveError> {
    observe(
        station,
        time,
        Target::Spk { kernel, naif_id },
        ObserveOptions::default(),
    )
}

/// Geocentric apparent position of an SPK target in the true equator and equinox
/// of date, in metres.
pub(crate) fn apparent_geocentric_spk_true_of_date_m(
    target_naif: i32,
    ts: &TimeScales,
    kernel: &Spk,
) -> Result<[f64; 3], ObserveError> {
    let et = (ts.jd_tdb - J2000_JD) * SECONDS_PER_DAY;
    ensure_finite(et)?;

    let earth = spk_state_j2000(kernel, NAIF_EARTH, NAIF_SSB, et)?;
    let v_earth_km_s = spk_velocity_j2000(kernel, NAIF_EARTH, NAIF_SSB, et, earth)?;
    let light_time = solve_spk_light_time(kernel, target_naif, et, earth.position_km)?;
    let u_astro = unit_checked(light_time.rho_vec_km)?;
    let u_deflected = if target_naif == NAIF_SUN {
        u_astro
    } else {
        apply_solar_deflection(
            kernel,
            light_time.et_emit,
            earth.position_km,
            light_time.target_position_km,
            u_astro,
        )?
    };
    let u_app = apply_aberration(u_deflected, v_earth_km_s)?;

    let tod = gcrs_to_true_of_date_matrix(ts)?;
    let true_km = mat3_vec3_mul_unchecked(&tod, &scale3(u_app, light_time.distance_km));
    validate_vec3(true_km)?;
    Ok(scale_km_to_m(true_km))
}

/// Geocentric apparent analytic Sun/Moon position in the true equator and
/// equinox of date, in metres.
pub(crate) fn apparent_geocentric_analytic_true_of_date_m(
    target_naif: i32,
    ts: &TimeScales,
) -> Result<[f64; 3], ObserveError> {
    let eci = sun_moon_eci_at(ts)?;
    let mean_m = match target_naif {
        NAIF_SUN => eci.sun,
        NAIF_MOON => eci.moon,
        _ => return Err(ObserveError::DegenerateGeometry),
    };
    let nutation = nutation_matrix(ts)?;
    let true_m = mat3_vec3_mul_unchecked(&nutation, &mean_m);
    validate_vec3(true_m)?;
    Ok(true_m)
}

/// General topocentric observation with caller-supplied time scales.
pub fn observe_with_time_scales(
    station: &GeodeticStationKm,
    ts: &TimeScales,
    target: Target<'_>,
    options: ObserveOptions,
) -> Result<Observation, ObserveError> {
    match target {
        Target::Sun => observe_reduced_sun_moon(station, ts, ReducedBody::Sun, options),
        Target::Moon => observe_reduced_sun_moon(station, ts, ReducedBody::Moon, options),
        Target::Spk { kernel, naif_id } => {
            observe_full_chain(station, ts, FullTarget::Spk { kernel, naif_id }, options)
        }
        Target::BarycentricState {
            kernel,
            position_km,
            velocity_km_s,
        } => observe_full_chain(
            station,
            ts,
            FullTarget::BarycentricState {
                kernel,
                position_km,
                velocity_km_s,
            },
            options,
        ),
    }
}

/// Topocentric azimuth/elevation/range of the Sun from a ground site at an
/// instant.
///
/// Reuses [`sun_moon_ecef`] for the Earth-fixed Sun vector and
/// [`itrs_to_topocentric`] for the station-to-target ENU reduction, so the
/// azimuth/elevation convention matches the satellite look-angle path.
pub fn sun_az_el(
    station: &GeodeticStationKm,
    time: UtcInstant,
) -> Result<BodyAzEl, BodyObservationError> {
    let sun_ecef_m = sun_moon_ecef(&time.time_scales())?.sun;
    body_az_el(station, sun_ecef_m)
}

/// Topocentric azimuth/elevation/range of the Moon from a ground site at an
/// instant.
///
/// The Earth-fixed Moon vector from [`sun_moon_ecef`] is geocentric, so the
/// station-to-target subtraction in [`itrs_to_topocentric`] applies the
/// topocentric (diurnal) parallax that matters for the nearby Moon.
pub fn moon_az_el(
    station: &GeodeticStationKm,
    time: UtcInstant,
) -> Result<BodyAzEl, BodyObservationError> {
    let moon_ecef_m = sun_moon_ecef(&time.time_scales())?.moon;
    body_az_el(station, moon_ecef_m)
}

fn body_az_el(
    station: &GeodeticStationKm,
    body_ecef_m: [f64; 3],
) -> Result<BodyAzEl, BodyObservationError> {
    let body_ecef_km = [
        body_ecef_m[0] / M_PER_KM,
        body_ecef_m[1] / M_PER_KM,
        body_ecef_m[2] / M_PER_KM,
    ];
    let (azimuth_deg, elevation_deg, range_km) = itrs_to_topocentric(body_ecef_km, station)?;
    Ok(BodyAzEl {
        azimuth_deg,
        elevation_deg,
        range_km,
    })
}

/// Illuminated fraction of the Moon as seen from a ground site at an instant.
///
/// The Sun-Moon-observer phase angle is the existing
/// [`phase_angle`] (the satellite Sun-target-observer geometry) evaluated with
/// the Moon as the target and the site as the observer, both from the Earth-fixed
/// Sun/Moon vectors of [`sun_moon_ecef`]. The illuminated fraction follows from
/// the half-angle relation `k = (1 + cos(phase)) / 2`. Topocentric and geocentric
/// fractions differ only negligibly; the site-relative phase angle is used for
/// consistency with the other site helpers in this module.
pub fn moon_illumination(
    station: &GeodeticStationKm,
    time: UtcInstant,
) -> Result<MoonIllumination, BodyObservationError> {
    let sun_moon = sun_moon_ecef(&time.time_scales())?;
    let sun_km = scale_m_to_km(sun_moon.sun);
    let moon_km = scale_m_to_km(sun_moon.moon);
    let (stn_x, stn_y, stn_z) = geodetic_to_itrs(
        station.latitude_deg,
        station.longitude_deg,
        station.altitude_km,
    )?;
    let observer_km = [stn_x, stn_y, stn_z];

    let phase_angle_deg = phase_angle(moon_km, sun_km, observer_km)?;
    let illuminated_fraction = (1.0 + libm::cos(phase_angle_deg.to_radians())) / 2.0;
    Ok(MoonIllumination {
        illuminated_fraction,
        phase_angle_deg,
    })
}

fn scale_m_to_km(v: [f64; 3]) -> [f64; 3] {
    [v[0] / M_PER_KM, v[1] / M_PER_KM, v[2] / M_PER_KM]
}

fn scale_km_to_m(v: [f64; 3]) -> [f64; 3] {
    [v[0] * M_PER_KM, v[1] * M_PER_KM, v[2] * M_PER_KM]
}

#[derive(Debug, Clone, Copy)]
enum FullTarget<'a> {
    Spk {
        kernel: &'a Spk,
        naif_id: i32,
    },
    BarycentricState {
        kernel: &'a Spk,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
    },
}

impl<'a> FullTarget<'a> {
    fn kernel(self) -> &'a Spk {
        match self {
            FullTarget::Spk { kernel, .. } | FullTarget::BarycentricState { kernel, .. } => kernel,
        }
    }

    fn is_spk_sun(self) -> bool {
        matches!(
            self,
            FullTarget::Spk {
                naif_id: NAIF_SUN,
                ..
            }
        )
    }
}

#[derive(Debug, Clone, Copy)]
enum ReducedBody {
    Sun,
    Moon,
}

#[derive(Debug, Clone, Copy)]
struct ObserverBarycentric {
    r_geo_km: [f64; 3],
    r_bary_km: [f64; 3],
    v_bary_km_s: [f64; 3],
}

#[derive(Debug, Clone, Copy)]
struct LightTimeSolution {
    target_position_km: [f64; 3],
    rho_vec_km: [f64; 3],
    distance_km: f64,
    et_emit: f64,
    #[cfg(test)]
    residual_s: f64,
    #[cfg(test)]
    iterations: usize,
}

fn observe_full_chain(
    station: &GeodeticStationKm,
    ts: &TimeScales,
    target: FullTarget<'_>,
    options: ObserveOptions,
) -> Result<Observation, ObserveError> {
    validate_refraction(options.refraction)?;
    let et = (ts.jd_tdb - J2000_JD) * SECONDS_PER_DAY;
    ensure_finite(et)?;

    let observer = observer_barycentric(station, ts, target.kernel(), et, options.polar_motion)?;
    let light_time = solve_light_time(target, et, observer.r_bary_km)?;
    let u_astro = unit_checked(light_time.rho_vec_km)?;
    let astrometric = equatorial_from_unit(u_astro, light_time.distance_km);

    let u_deflected = if options.deflection && !target.is_spk_sun() {
        apply_solar_deflection(
            target.kernel(),
            light_time.et_emit,
            observer.r_bary_km,
            light_time.target_position_km,
            u_astro,
        )?
    } else {
        u_astro
    };
    let u_app = if options.aberration {
        apply_aberration(u_deflected, observer.v_bary_km_s)?
    } else {
        u_deflected
    };
    let apparent_icrs = equatorial_from_unit(u_app, light_time.distance_km);

    let tod = gcrs_to_true_of_date_matrix(ts)?;
    let u_tod = unit_checked(mat3_vec3_mul_unchecked(&tod, &u_app))?;
    let apparent = equatorial_from_unit(u_tod, light_time.distance_km);
    let horizontal = apparent_horizontal(
        station,
        ts,
        observer.r_geo_km,
        u_app,
        light_time.distance_km,
        options,
    )?;
    let (hour_angle_deg, hour_angle_hours) = hour_angle(station, ts, apparent)?;
    let ecliptic = ecliptic_from_true_equatorial(ts, u_tod, light_time.distance_km)?;

    Ok(Observation {
        astrometric,
        apparent_icrs,
        apparent,
        horizontal,
        hour_angle_deg,
        hour_angle_hours,
        ecliptic,
        reduced: false,
    })
}

fn observe_reduced_sun_moon(
    station: &GeodeticStationKm,
    ts: &TimeScales,
    body: ReducedBody,
    options: ObserveOptions,
) -> Result<Observation, ObserveError> {
    validate_refraction(options.refraction)?;

    let eci = sun_moon_eci_at(ts)?;
    let mean_m = match body {
        ReducedBody::Sun => eci.sun,
        ReducedBody::Moon => eci.moon,
    };
    let mean_km = scale_m_to_km(mean_m);
    let astrometric = equatorial_from_vector(mean_km)?;
    let apparent_icrs = astrometric;

    let nutation = nutation_matrix(ts)?;
    let true_m = mat3_vec3_mul_unchecked(&nutation, &mean_m);
    let true_km = scale_m_to_km(true_m);
    let u_tod = unit_checked(true_km)?;
    let apparent = equatorial_from_unit(u_tod, norm_checked(true_km)?);

    let ecef = sun_moon_ecef(ts)?;
    let body_ecef_km = match body {
        ReducedBody::Sun => scale_m_to_km(ecef.sun),
        ReducedBody::Moon => scale_m_to_km(ecef.moon),
    };
    let (azimuth_deg, elevation_deg, range_km) = itrs_to_topocentric(body_ecef_km, station)?;
    let horizontal = Horizontal {
        azimuth_deg,
        elevation_deg: apply_refraction(elevation_deg, options.refraction)?,
        range_km,
    };
    let (hour_angle_deg, hour_angle_hours) = hour_angle(station, ts, apparent)?;
    let ecliptic = ecliptic_from_true_equatorial(ts, u_tod, apparent.distance_km)?;

    Ok(Observation {
        astrometric,
        apparent_icrs,
        apparent,
        horizontal,
        hour_angle_deg,
        hour_angle_hours,
        ecliptic,
        reduced: true,
    })
}

fn observer_barycentric(
    station: &GeodeticStationKm,
    ts: &TimeScales,
    kernel: &Spk,
    et: f64,
    polar_motion: Option<PolarMotion>,
) -> Result<ObserverBarycentric, ObserveError> {
    let (x, y, z) = geodetic_to_itrs(
        station.latitude_deg,
        station.longitude_deg,
        station.altitude_km,
    )?;
    let r_itrs = [x, y, z];
    let r_geo_tuple = match polar_motion {
        Some(pole) => itrs_to_gcrs_compute_with_polar_motion(x, y, z, ts, pole)?,
        None => itrs_to_gcrs_compute(x, y, z, ts)?,
    };
    let r_geo_km = [r_geo_tuple.0, r_geo_tuple.1, r_geo_tuple.2];

    let v_itrs = [
        -OMEGA_E_DOT_RAD_S * r_itrs[1],
        OMEGA_E_DOT_RAD_S * r_itrs[0],
        0.0,
    ];
    let itrs_to_gcrs = match polar_motion {
        Some(pole) => itrs_to_gcrs_matrix_with_polar_motion(ts, pole)?,
        None => itrs_to_gcrs_matrix(ts)?,
    };
    let v_geo_km_s = mat3_vec3_mul_unchecked(&itrs_to_gcrs, &v_itrs);
    validate_vec3(v_geo_km_s)?;

    let earth = spk_state_j2000(kernel, NAIF_EARTH, NAIF_SSB, et)?;
    let v_earth_km_s = spk_velocity_j2000(kernel, NAIF_EARTH, NAIF_SSB, et, earth)?;
    Ok(ObserverBarycentric {
        r_geo_km,
        r_bary_km: add3(earth.position_km, r_geo_km),
        v_bary_km_s: add3(v_earth_km_s, v_geo_km_s),
    })
}

fn solve_light_time(
    target: FullTarget<'_>,
    et: f64,
    r_obs_bary_km: [f64; 3],
) -> Result<LightTimeSolution, ObserveError> {
    match target {
        FullTarget::Spk { kernel, naif_id } => {
            solve_spk_light_time(kernel, naif_id, et, r_obs_bary_km)
        }
        FullTarget::BarycentricState {
            position_km,
            velocity_km_s,
            ..
        } => solve_supplied_state_light_time(position_km, velocity_km_s, et, r_obs_bary_km),
    }
}

fn solve_spk_light_time(
    kernel: &Spk,
    naif_id: i32,
    et: f64,
    r_obs_bary_km: [f64; 3],
) -> Result<LightTimeSolution, ObserveError> {
    let initial = spk_state_j2000(kernel, naif_id, NAIF_SSB, et)?.position_km;
    let mut tau_s = norm_checked(sub3(initial, r_obs_bary_km))? / C_KM_S;
    let mut target_position_km = initial;
    let mut rho_vec_km = sub3(target_position_km, r_obs_bary_km);
    let mut et_emit = et - tau_s;
    #[cfg(test)]
    let mut residual_s = f64::INFINITY;
    #[cfg(test)]
    let mut iterations = 0_usize;

    for _ in 1..=LIGHT_TIME_REFINEMENTS {
        et_emit = et - tau_s;
        target_position_km = spk_state_j2000(kernel, naif_id, NAIF_SSB, et_emit)?.position_km;
        rho_vec_km = sub3(target_position_km, r_obs_bary_km);
        let next_tau_s = norm_checked(rho_vec_km)? / C_KM_S;
        #[cfg(test)]
        {
            residual_s = (next_tau_s - tau_s).abs();
            iterations += 1;
        }
        let converged = (next_tau_s - tau_s).abs() < LIGHT_TIME_TOLERANCE_S;
        tau_s = next_tau_s;
        if converged {
            break;
        }
    }

    Ok(LightTimeSolution {
        target_position_km,
        rho_vec_km,
        distance_km: norm_checked(rho_vec_km)?,
        et_emit,
        #[cfg(test)]
        residual_s,
        #[cfg(test)]
        iterations,
    })
}

fn solve_supplied_state_light_time(
    position_km: [f64; 3],
    velocity_km_s: [f64; 3],
    et: f64,
    r_obs_bary_km: [f64; 3],
) -> Result<LightTimeSolution, ObserveError> {
    validate_vec3(position_km)?;
    validate_vec3(velocity_km_s)?;

    let mut tau_s = norm_checked(sub3(position_km, r_obs_bary_km))? / C_KM_S;
    let mut target_position_km = position_km;
    let mut rho_vec_km = sub3(target_position_km, r_obs_bary_km);
    let mut et_emit = et - tau_s;
    #[cfg(test)]
    let mut residual_s = f64::INFINITY;
    #[cfg(test)]
    let mut iterations = 0_usize;

    for _ in 1..=LIGHT_TIME_REFINEMENTS {
        et_emit = et - tau_s;
        target_position_km = sub3(position_km, scale3(velocity_km_s, tau_s));
        rho_vec_km = sub3(target_position_km, r_obs_bary_km);
        let next_tau_s = norm_checked(rho_vec_km)? / C_KM_S;
        #[cfg(test)]
        {
            residual_s = (next_tau_s - tau_s).abs();
            iterations += 1;
        }
        let converged = (next_tau_s - tau_s).abs() < LIGHT_TIME_TOLERANCE_S;
        tau_s = next_tau_s;
        if converged {
            break;
        }
    }

    Ok(LightTimeSolution {
        target_position_km,
        rho_vec_km,
        distance_km: norm_checked(rho_vec_km)?,
        et_emit,
        #[cfg(test)]
        residual_s,
        #[cfg(test)]
        iterations,
    })
}

fn spk_state_j2000(
    kernel: &Spk,
    target: i32,
    center: i32,
    et: f64,
) -> Result<SpkState, ObserveError> {
    let state = kernel.spk_state(target, center, et)?;
    if state.frame != SPK_FRAME_J2000 {
        return Err(ObserveError::UnsupportedSpkFrame { frame: state.frame });
    }
    validate_vec3(state.position_km)?;
    if let Some(velocity) = state.velocity_km_s {
        validate_vec3(velocity)?;
    }
    Ok(state)
}

fn spk_velocity_j2000(
    kernel: &Spk,
    target: i32,
    center: i32,
    et: f64,
    state: SpkState,
) -> Result<[f64; 3], ObserveError> {
    if let Some(velocity) = state.velocity_km_s {
        return Ok(velocity);
    }

    let dt = 1.0;
    let before = spk_state_j2000(kernel, target, center, et - dt);
    let after = spk_state_j2000(kernel, target, center, et + dt);
    let velocity = match (before, after) {
        (Ok(before), Ok(after)) => scale3(sub3(after.position_km, before.position_km), 0.5 / dt),
        (Ok(before), Err(_)) => scale3(sub3(state.position_km, before.position_km), 1.0 / dt),
        (Err(_), Ok(after)) => scale3(sub3(after.position_km, state.position_km), 1.0 / dt),
        (Err(error), Err(_)) => return Err(error),
    };
    validate_vec3(velocity)?;
    Ok(velocity)
}

fn apply_solar_deflection(
    kernel: &Spk,
    et_emit: f64,
    r_obs_bary_km: [f64; 3],
    r_tgt_bary_km: [f64; 3],
    p: [f64; 3],
) -> Result<[f64; 3], ObserveError> {
    let r_sun_bary_km = spk_state_j2000(kernel, NAIF_SUN, NAIF_SSB, et_emit)?.position_km;
    let obs_minus_sun = sub3(r_obs_bary_km, r_sun_bary_km);
    let target_minus_sun = sub3(r_tgt_bary_km, r_sun_bary_km);
    let e = unit_checked(obs_minus_sun)?;
    let q = unit_checked(target_minus_sun)?;
    let d_sun_km = norm_checked(obs_minus_sun)?;
    let g1 = 2.0 * GM_SUN_KM3_S2 / (C_KM_S * C_KM_S * d_sun_km);
    let denom = (1.0 + dot3(q, e)).max(SOLAR_DEFLECTION_DENOM_FLOOR);
    let correction = scale3(
        sub3(scale3(e, dot3(p, q)), scale3(q, dot3(e, p))),
        g1 / denom,
    );
    unit_checked(add3(p, correction))
}

fn apply_aberration(p: [f64; 3], v_obs_bary_km_s: [f64; 3]) -> Result<[f64; 3], ObserveError> {
    validate_vec3(v_obs_bary_km_s)?;
    let beta = scale3(v_obs_bary_km_s, 1.0 / C_KM_S);
    let beta2 = dot3(beta, beta);
    if !beta2.is_finite() || !(0.0..1.0).contains(&beta2) {
        return Err(ObserveError::NonFinite);
    }
    let inv_beta = (1.0 - beta2).sqrt();
    let p_dot_beta = dot3(p, beta);
    let denom = 1.0 + p_dot_beta;
    if !denom.is_finite() || denom == 0.0 {
        return Err(ObserveError::DegenerateGeometry);
    }
    let f = 1.0 + p_dot_beta / (1.0 + inv_beta);
    let numerator = add3(scale3(p, inv_beta), scale3(beta, f));
    unit_checked(scale3(numerator, 1.0 / denom))
}

fn apparent_horizontal(
    station: &GeodeticStationKm,
    ts: &TimeScales,
    r_obs_geo_km: [f64; 3],
    u_app: [f64; 3],
    distance_km: f64,
    options: ObserveOptions,
) -> Result<Horizontal, ObserveError> {
    let target_gcrs_km = add3(r_obs_geo_km, scale3(u_app, distance_km));
    validate_vec3(target_gcrs_km)?;
    let gcrs_to_itrs = match options.polar_motion {
        Some(pole) => gcrs_to_itrs_matrix_with_polar_motion(ts, pole)?,
        None => gcrs_to_itrs_matrix(ts)?,
    };
    let target_itrs_km = mat3_vec3_mul_unchecked(&gcrs_to_itrs, &target_gcrs_km);
    let (azimuth_deg, elevation_deg, _range_km) = itrs_to_topocentric(target_itrs_km, station)?;
    Ok(Horizontal {
        azimuth_deg,
        elevation_deg: apply_refraction(elevation_deg, options.refraction)?,
        range_km: distance_km,
    })
}

fn hour_angle(
    station: &GeodeticStationKm,
    ts: &TimeScales,
    apparent: Equatorial,
) -> Result<(f64, f64), ObserveError> {
    let gast_deg = greenwich_apparent_sidereal_time_radians(ts)?.to_degrees();
    let hour_angle_deg =
        wrap_pm180(gast_deg + station.longitude_deg - apparent.right_ascension_deg);
    Ok((hour_angle_deg, hour_angle_deg / 15.0))
}

fn ecliptic_from_true_equatorial(
    ts: &TimeScales,
    u_tod: [f64; 3],
    distance_km: f64,
) -> Result<Ecliptic, ObserveError> {
    let eps_true = true_obliquity_radians(ts)?;
    let cos_eps = libm::cos(eps_true);
    let sin_eps = libm::sin(eps_true);
    let x = u_tod[0];
    let y = cos_eps * u_tod[1] + sin_eps * u_tod[2];
    let z = -sin_eps * u_tod[1] + cos_eps * u_tod[2];
    Ok(Ecliptic {
        longitude_deg: wrap_360(libm::atan2(y, x).to_degrees()),
        latitude_deg: libm::asin(clamp_unit(z)).to_degrees(),
        distance_km,
    })
}

fn true_obliquity_radians(ts: &TimeScales) -> Result<f64, ObserveError> {
    let mean = skyfield_mean_obliquity_radians(ts.jd_tdb).map_err(|_| ObserveError::NonFinite)?;
    let (_dpsi, deps) = skyfield_iau2000a_radians(ts.jd_tt).map_err(|_| ObserveError::NonFinite)?;
    let eps = mean + deps;
    ensure_finite(eps)?;
    Ok(eps)
}

fn nutation_matrix(ts: &TimeScales) -> Result<[[f64; 3]; 3], ObserveError> {
    let mean = skyfield_mean_obliquity_radians(ts.jd_tdb).map_err(|_| ObserveError::NonFinite)?;
    let (dpsi, deps) = skyfield_iau2000a_radians(ts.jd_tt).map_err(|_| ObserveError::NonFinite)?;
    build_skyfield_nutation_matrix(mean, mean + deps, dpsi).map_err(|_| ObserveError::NonFinite)
}

fn equatorial_from_vector(vector_km: [f64; 3]) -> Result<Equatorial, ObserveError> {
    let distance_km = norm_checked(vector_km)?;
    Ok(equatorial_from_unit(
        scale3(vector_km, 1.0 / distance_km),
        distance_km,
    ))
}

fn equatorial_from_unit(unit: [f64; 3], distance_km: f64) -> Equatorial {
    let right_ascension_deg = wrap_360(libm::atan2(unit[1], unit[0]).to_degrees());
    Equatorial {
        right_ascension_deg,
        right_ascension_hours: right_ascension_deg / 15.0,
        declination_deg: libm::asin(clamp_unit(unit[2])).to_degrees(),
        distance_km,
    }
}

fn apply_refraction(
    elevation_deg: f64,
    refraction: Option<Refraction>,
) -> Result<f64, ObserveError> {
    ensure_finite(elevation_deg)?;
    let Some(refraction) = refraction else {
        return Ok(elevation_deg);
    };
    ensure_finite(refraction.pressure_mbar)?;
    ensure_finite(refraction.temperature_c)?;
    if !(-1.0..=89.9).contains(&elevation_deg) {
        return Ok(elevation_deg);
    }
    let argument_deg = elevation_deg + 10.3 / (elevation_deg + 5.11);
    let r_arcmin = refraction.pressure_mbar / 1010.0 * 283.0 / (273.0 + refraction.temperature_c)
        * 1.02
        / libm::tan(argument_deg.to_radians());
    let corrected = elevation_deg + r_arcmin / 60.0;
    ensure_finite(corrected)?;
    Ok(corrected)
}

fn validate_refraction(refraction: Option<Refraction>) -> Result<(), ObserveError> {
    if let Some(refraction) = refraction {
        ensure_finite(refraction.pressure_mbar)?;
        ensure_finite(refraction.temperature_c)?;
    }
    Ok(())
}

fn norm_checked(vector: [f64; 3]) -> Result<f64, ObserveError> {
    validate_vec3(vector)?;
    let norm = norm3(vector);
    if !norm.is_finite() {
        return Err(ObserveError::NonFinite);
    }
    if norm == 0.0 {
        return Err(ObserveError::DegenerateGeometry);
    }
    Ok(norm)
}

fn unit_checked(vector: [f64; 3]) -> Result<[f64; 3], ObserveError> {
    unit3(vector).ok_or(ObserveError::DegenerateGeometry)
}

fn validate_vec3(vector: [f64; 3]) -> Result<(), ObserveError> {
    if vector.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(ObserveError::NonFinite)
    }
}

fn ensure_finite(value: f64) -> Result<(), ObserveError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ObserveError::NonFinite)
    }
}

fn wrap_360(deg: f64) -> f64 {
    deg.rem_euclid(360.0)
}

fn wrap_pm180(deg: f64) -> f64 {
    let wrapped = (deg + 180.0).rem_euclid(360.0) - 180.0;
    if wrapped <= -180.0 {
        wrapped + 360.0
    } else {
        wrapped
    }
}

fn clamp_unit(value: f64) -> f64 {
    value.clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference site: Royal Observatory, Greenwich (WGS84), altitude ~46 m.
    fn greenwich() -> GeodeticStationKm {
        GeodeticStationKm {
            latitude_deg: 51.4769,
            longitude_deg: 0.0,
            altitude_km: 0.046,
        }
    }

    fn observe_fixture_spk() -> Spk {
        Spk::from_bytes(include_bytes!(
            "../../../tests/fixtures/bodies/observe_de.bsp"
        ))
        .expect("fixture SPK")
    }

    #[test]
    fn sun_az_el_at_solar_transit_is_due_south_and_high() {
        // Apparent solar upper transit at Greenwich on 2024-06-20 is 12:01:42 UTC
        // (Skyfield de421 meridian_transits). At that instant the Sun is on the
        // meridian: Skyfield gives az 180.000 deg, alt 61.960 deg, range
        // 1.52011e8 km (near aphelion). The low-precision analytic series is held
        // to 0.5 deg in az/el and 5.0e5 km in range.
        let time = UtcInstant::from_utc(2024, 6, 20, 12, 1, 42, 0).expect("valid UTC");
        let look = sun_az_el(&greenwich(), time).expect("valid sun geometry");
        assert!(
            (look.azimuth_deg - 180.0).abs() < 0.5,
            "sun azimuth {}",
            look.azimuth_deg
        );
        assert!(
            (look.elevation_deg - 61.960).abs() < 0.5,
            "sun elevation {}",
            look.elevation_deg
        );
        assert!(
            (look.range_km - 1.52011e8).abs() < 5.0e5,
            "sun range {}",
            look.range_km
        );
    }

    #[test]
    fn moon_illumination_tracks_full_moon() {
        // Full moon of 2024-04-23 23:49 UTC. Skyfield's (de421) geocentric
        // illuminated fraction at this instant is 0.9998 (phase angle 1.68 deg).
        // The low-precision analytic series is held to a coarse tolerance:
        // fraction within 0.02, phase angle below 11 deg.
        let time = UtcInstant::from_utc(2024, 4, 23, 23, 49, 0, 0).expect("valid UTC");
        let illum = moon_illumination(&greenwich(), time).expect("valid moon illumination");
        assert!(
            (illum.illuminated_fraction - 0.998).abs() < 0.02,
            "full-moon fraction {}",
            illum.illuminated_fraction
        );
        assert!(
            illum.phase_angle_deg < 11.0,
            "full-moon phase angle {}",
            illum.phase_angle_deg
        );
    }

    #[test]
    fn moon_illumination_tracks_new_moon() {
        // New moon of 2024-04-08 18:21 UTC (the total-eclipse new moon). Skyfield's
        // (de421) geocentric fraction is 0.0000 (phase angle 179.65 deg). Held to
        // fraction within 0.02 and phase angle above 168 deg.
        let time = UtcInstant::from_utc(2024, 4, 8, 18, 21, 0, 0).expect("valid UTC");
        let illum = moon_illumination(&greenwich(), time).expect("valid moon illumination");
        assert!(
            illum.illuminated_fraction < 0.02,
            "new-moon fraction {}",
            illum.illuminated_fraction
        );
        assert!(
            illum.phase_angle_deg > 168.0,
            "new-moon phase angle {}",
            illum.phase_angle_deg
        );
    }

    #[test]
    fn moon_illumination_near_first_quarter() {
        // First quarter of 2024-04-15 19:13 UTC: about half the disk lit. Skyfield's
        // (de421) geocentric fraction is 0.5013 (phase angle 89.85 deg). Held to
        // fraction within 0.05.
        let time = UtcInstant::from_utc(2024, 4, 15, 19, 13, 0, 0).expect("valid UTC");
        let illum = moon_illumination(&greenwich(), time).expect("valid moon illumination");
        assert!(
            (illum.illuminated_fraction - 0.50).abs() < 0.05,
            "first-quarter fraction {}",
            illum.illuminated_fraction
        );
    }

    #[test]
    fn moon_az_el_matches_reference_at_transit() {
        // Moon upper transit at Greenwich on 2024-04-23 is 23:55:59 UTC. At that
        // instant Skyfield (de421, apparent topocentric altaz) gives az 180.000
        // deg, alt 23.120 deg, range 397206 km. The low-precision analytic series
        // is held to 0.3 deg in az/el and 1000 km in range; the band check also
        // confirms the parallax-corrected station subtraction ran.
        let time = UtcInstant::from_utc(2024, 4, 23, 23, 55, 59, 0).expect("valid UTC");
        let look = moon_az_el(&greenwich(), time).expect("valid moon geometry");
        assert!(
            (look.azimuth_deg - 180.0).abs() < 0.3,
            "moon azimuth {}",
            look.azimuth_deg
        );
        assert!(
            (look.elevation_deg - 23.120).abs() < 0.3,
            "moon elevation {}",
            look.elevation_deg
        );
        assert!(
            (look.range_km - 397_206.0).abs() < 1000.0,
            "moon range {}",
            look.range_km
        );
    }

    #[test]
    fn spk_light_time_converges_for_outer_planet() {
        let kernel = observe_fixture_spk();
        let station = greenwich();
        let ts = UtcInstant::from_utc(2024, 1, 1, 0, 0, 0, 0)
            .expect("valid UTC")
            .time_scales();
        let et = (ts.jd_tdb - J2000_JD) * SECONDS_PER_DAY;
        let observer =
            observer_barycentric(&station, &ts, &kernel, et, None).expect("observer state");
        let solution =
            solve_spk_light_time(&kernel, 5, et, observer.r_bary_km).expect("light time");

        assert!(solution.iterations <= LIGHT_TIME_REFINEMENTS);
        assert!(
            solution.residual_s < LIGHT_TIME_TOLERANCE_S,
            "residual {}",
            solution.residual_s
        );
    }

    #[test]
    fn degenerate_zero_range_returns_typed_error() {
        let kernel = observe_fixture_spk();
        let station = greenwich();
        let time = UtcInstant::from_utc(2024, 1, 1, 0, 0, 0, 0).expect("valid UTC");
        let ts = time.time_scales();
        let et = (ts.jd_tdb - J2000_JD) * SECONDS_PER_DAY;
        let observer =
            observer_barycentric(&station, &ts, &kernel, et, None).expect("observer state");
        let err = observe_with_time_scales(
            &station,
            &ts,
            Target::BarycentricState {
                kernel: &kernel,
                position_km: observer.r_bary_km,
                velocity_km_s: [0.0, 0.0, 0.0],
            },
            ObserveOptions::default(),
        )
        .unwrap_err();

        assert!(matches!(err, ObserveError::DegenerateGeometry));
    }
}
