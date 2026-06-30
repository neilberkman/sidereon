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
use crate::astro::bodies::sun_moon::{sun_moon_ecef, SunMoonError};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::frames::transforms::{
    geodetic_to_itrs, itrs_to_topocentric, FrameTransformError, GeodeticStationKm,
};
use crate::astro::passes::UtcInstant;

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
    let illuminated_fraction = (1.0 + phase_angle_deg.to_radians().cos()) / 2.0;
    Ok(MoonIllumination {
        illuminated_fraction,
        phase_angle_deg,
    })
}

fn scale_m_to_km(v: [f64; 3]) -> [f64; 3] {
    [v[0] / M_PER_KM, v[1] / M_PER_KM, v[2] / M_PER_KM]
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
}
