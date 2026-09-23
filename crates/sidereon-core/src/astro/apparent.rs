//! Shared apparent-place reducers for body observations and almanac events.

use crate::astro::almanac::{AlmanacError, EphemerisSource};
use crate::astro::bodies::observe::{
    apparent_geocentric_analytic_true_of_date_m, apparent_geocentric_spk_true_of_date_m,
    observe_with_time_scales, ObserveError, ObserveOptions, Target,
};
use crate::astro::bodies::sun_moon::SunMoonError;
use crate::astro::frames::transforms::{FrameTransformError, GeodeticStationKm};
use crate::astro::time::scales::TimeScales;

const NAIF_SUN: i32 = 10;
const NAIF_MOON: i32 = 301;

/// Apparent topocentric right ascension and declination of a body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaDec {
    /// Right ascension in degrees on `[0, 360)`.
    pub right_ascension_deg: f64,
    /// Declination in degrees on `[-90, 90]`.
    pub declination_deg: f64,
    /// Distance from the observer, kilometres.
    pub distance_km: f64,
}

/// Apparent topocentric reduction used by meridian transit finders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopocentricApparent {
    /// Topocentric apparent right ascension and declination.
    pub radec: RaDec,
    /// Local apparent hour angle in degrees on `(-180, 180]`.
    pub hour_angle_deg: f64,
    /// Topocentric apparent geometric altitude, degrees, without refraction.
    pub altitude_deg: f64,
}

/// Geocentric apparent position of a body, true equator and equinox of date, metres.
pub fn apparent_geocentric(
    target_naif: i32,
    ts: &TimeScales,
    source: EphemerisSource<'_>,
) -> Result<[f64; 3], AlmanacError> {
    match source {
        EphemerisSource::Spk(kernel) => {
            apparent_geocentric_spk_true_of_date_m(target_naif, ts, kernel).map_err(map_observe)
        }
        EphemerisSource::Analytic => match target_naif {
            NAIF_SUN | NAIF_MOON => {
                apparent_geocentric_analytic_true_of_date_m(target_naif, ts).map_err(map_observe)
            }
            _ => Err(AlmanacError::EphemerisRequired),
        },
    }
}

/// Topocentric apparent right ascension, declination, hour angle, and altitude.
pub fn topocentric_apparent(
    target_naif: i32,
    station: &GeodeticStationKm,
    ts: &TimeScales,
    source: EphemerisSource<'_>,
) -> Result<TopocentricApparent, AlmanacError> {
    let target = match source {
        EphemerisSource::Spk(kernel) => Target::Spk {
            kernel,
            naif_id: target_naif,
        },
        EphemerisSource::Analytic => match target_naif {
            NAIF_SUN => Target::Sun,
            NAIF_MOON => Target::Moon,
            _ => return Err(AlmanacError::EphemerisRequired),
        },
    };
    let observation = observe_with_time_scales(station, ts, target, ObserveOptions::default())
        .map_err(map_observe)?;
    Ok(TopocentricApparent {
        radec: RaDec {
            right_ascension_deg: observation.apparent.right_ascension_deg,
            declination_deg: observation.apparent.declination_deg,
            distance_km: observation.apparent.distance_km,
        },
        hour_angle_deg: observation.hour_angle_deg,
        altitude_deg: observation.horizontal.elevation_deg,
    })
}

fn map_observe(error: ObserveError) -> AlmanacError {
    match error {
        ObserveError::Spk(error) => AlmanacError::Spk(error),
        ObserveError::FrameTransform(error)
        | ObserveError::SunMoon(SunMoonError::FrameTransform(
            error @ FrameTransformError::Ut1OutsideCoverage { .. },
        )) => AlmanacError::from(error),
        ObserveError::SunMoon(_) => AlmanacError::Frame("sun_moon"),
        ObserveError::Angle(_) => AlmanacError::Frame("angle"),
        ObserveError::NonFinite => AlmanacError::InvalidInput {
            field: "geometry",
            reason: "must be finite",
        },
        ObserveError::DegenerateGeometry => AlmanacError::InvalidInput {
            field: "geometry",
            reason: "degenerate",
        },
    }
}
