//! Astronomical almanac event finders built on the shared event finder.

use std::cell::RefCell;
use std::fmt;

use crate::astro::apparent::apparent_geocentric;
use crate::astro::events::{EventFinder, EventFinderError};
use crate::astro::frames::transforms::{geodetic_to_itrs, FrameTransformError, GeodeticStationKm};
use crate::astro::math::vec3::{dot3, norm3};
use crate::astro::passes::UtcInstant;
use crate::astro::spk::{Spk, SpkError};
use crate::astro::{
    constants::{
        time::{SECONDS_PER_DAY, SECONDS_PER_HOUR},
        units::MICROSECONDS_PER_SECOND,
    },
    events::CrossingEvent,
};
use crate::validate;

mod eclipse;
mod ecliptic;
mod phases;
mod planets;
mod seasons;
#[cfg(test)]
mod tests;
mod transits;

pub use eclipse::lunar_solar_eclipses;
pub use ecliptic::{geocentric_ecliptic, EclipticLonLat};
pub use phases::{moon_phase_deg, moon_phases};
pub use planets::planetary_events;
pub use seasons::seasons;
pub use transits::meridian_transits;

pub(crate) const NAIF_SUN: i32 = 10;
pub(crate) const NAIF_MOON: i32 = 301;

pub(crate) const SEASON_PLANET_STEP_MAX_SECONDS: f64 = SECONDS_PER_DAY;
pub(crate) const PHASE_STEP_MAX_SECONDS: f64 = 3.0 * SECONDS_PER_DAY;
pub(crate) const TRANSIT_STEP_MAX_SECONDS: f64 = SECONDS_PER_HOUR;

/// Which ephemeris backs an almanac computation.
#[derive(Clone, Copy)]
pub enum EphemerisSource<'a> {
    /// DE-series kernel already loaded by the caller.
    Spk(&'a Spk),
    /// Analytic Sun/Moon series.
    Analytic,
}

/// Seasonal marker.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonKind {
    MarchEquinox,
    JuneSolstice,
    SeptemberEquinox,
    DecemberSolstice,
}

/// Principal lunar phase.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonPhaseKind {
    New,
    FirstQuarter,
    Full,
    LastQuarter,
}

/// Planetary ecliptic-longitude event.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanetaryEventKind {
    Conjunction,
    Opposition,
}

/// Meridian transit kind.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CulminationKind {
    Upper,
    Lower,
}

/// Lunar or solar eclipse kind.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EclipseKind {
    LunarPenumbral,
    LunarPartial,
    LunarTotal,
    SolarPartial,
    SolarAnnular,
    SolarTotal,
    SolarHybrid,
}

/// Planet selector for almanac event finders.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Planet {
    Mercury,
    Venus,
    Mars,
    Jupiter,
    Saturn,
    Uranus,
    Neptune,
}

/// Body selector for meridian transits.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitBody {
    Sun,
    Moon,
    Planet(Planet),
}

/// One seasonal marker event.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeasonEvent {
    pub time: UtcInstant,
    pub kind: SeasonKind,
}

/// One lunar phase event.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoonPhaseEvent {
    pub time: UtcInstant,
    pub kind: MoonPhaseKind,
}

/// One planetary opposition or conjunction.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetaryEvent {
    pub time: UtcInstant,
    pub planet: Planet,
    pub kind: PlanetaryEventKind,
    pub elongation_deg: f64,
}

/// One upper or lower culmination.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CulminationEvent {
    pub time: UtcInstant,
    pub kind: CulminationKind,
    pub altitude_deg: f64,
}

/// One lunar or solar eclipse event.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EclipseEvent {
    pub time_maximum: UtcInstant,
    pub kind: EclipseKind,
    pub magnitude: f64,
    pub moon_latitude_deg: f64,
    pub gamma: f64,
    pub uncertain: bool,
}

/// Error returned by almanac computations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum AlmanacError {
    Finder(EventFinderError),
    Spk(SpkError),
    Frame(&'static str),
    EphemerisRequired,
    InferiorPlanetOpposition,
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for AlmanacError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Finder(error) => write!(f, "event finder failed: {error}"),
            Self::Spk(error) => write!(f, "SPK failed: {error}"),
            Self::Frame(label) => write!(f, "frame reduction failed: {label}"),
            Self::EphemerisRequired => write!(f, "SPK ephemeris is required"),
            Self::InferiorPlanetOpposition => {
                write!(f, "opposition is not defined for an inferior planet")
            }
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid almanac input {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for AlmanacError {}

pub(crate) fn validate_scan_controls(
    step_seconds: f64,
    time_tolerance_seconds: f64,
    step_max_seconds: f64,
) -> Result<(), AlmanacError> {
    validate::positive_step(step_seconds, "step_seconds").map_err(map_field_error)?;
    validate::positive_step(time_tolerance_seconds, "time_tolerance_seconds")
        .map_err(map_field_error)?;
    if step_seconds > step_max_seconds {
        return Err(AlmanacError::InvalidInput {
            field: "step_seconds",
            reason: "exceeds maximum",
        });
    }
    Ok(())
}

pub(crate) fn validate_station(station: &GeodeticStationKm) -> Result<(), AlmanacError> {
    geodetic_to_itrs(
        station.latitude_deg,
        station.longitude_deg,
        station.altitude_km,
    )
    .map(|_| ())
    .map_err(map_frame_input)
}

pub(crate) fn event_finder(
    start: UtcInstant,
    end: UtcInstant,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<EventFinder, AlmanacError> {
    EventFinder::new(
        0.0,
        seconds_between(start, end)?,
        step_seconds,
        time_tolerance_seconds,
    )
    .map_err(AlmanacError::Finder)
}

pub(crate) fn seconds_between(start: UtcInstant, end: UtcInstant) -> Result<f64, AlmanacError> {
    let span = end
        .unix_microseconds()
        .checked_sub(start.unix_microseconds())
        .ok_or(AlmanacError::Finder(EventFinderError::InvalidInput {
            field: "time_window",
            reason: "start/end span overflows i64 microseconds",
        }))?;
    Ok(span as f64 / MICROSECONDS_PER_SECOND)
}

pub(crate) fn instant_at_offset_seconds(start: UtcInstant, offset_seconds: f64) -> UtcInstant {
    UtcInstant::from_unix_microseconds(
        start.unix_microseconds() + (offset_seconds * MICROSECONDS_PER_SECOND).floor() as i64,
    )
}

pub(crate) fn offset_instant(start: UtcInstant, offset_seconds: f64) -> UtcInstant {
    instant_at_offset_seconds(start, offset_seconds)
}

pub(crate) fn body_ecliptic(
    source: EphemerisSource<'_>,
    target_naif: i32,
    time: UtcInstant,
) -> Result<EclipticLonLat, AlmanacError> {
    let ts = time.time_scales();
    let pos = apparent_geocentric(target_naif, &ts, source)?;
    geocentric_ecliptic(pos, &ts)
}

pub(crate) fn apparent_km(
    source: EphemerisSource<'_>,
    target_naif: i32,
    time: UtcInstant,
) -> Result<[f64; 3], AlmanacError> {
    let pos_m = apparent_geocentric(target_naif, &time.time_scales(), source)?;
    Ok([pos_m[0] * 1.0e-3, pos_m[1] * 1.0e-3, pos_m[2] * 1.0e-3])
}

pub(crate) fn find_angle_crossing_times<F>(
    start: UtcInstant,
    end: UtcInstant,
    step_seconds: f64,
    time_tolerance_seconds: f64,
    target_deg: f64,
    angle_fn: F,
) -> Result<Vec<UtcInstant>, AlmanacError>
where
    F: Fn(UtcInstant) -> Result<f64, AlmanacError>,
{
    let finder = event_finder(start, end, step_seconds, time_tolerance_seconds)?;
    let latch = RefCell::new(None);
    let crossings = finder
        .find_crossings(
            |offset_seconds| {
                latch_scalar(&latch, || {
                    let time = instant_at_offset_seconds(start, offset_seconds);
                    let angle = angle_fn(time)?;
                    Ok(libm::sin((angle - target_deg).to_radians()))
                })
            },
            0.0,
        )
        .map_err(|error| latched_or_finder(error, &latch))?;

    let mut times = Vec::new();
    for crossing in crossings {
        let time = instant_at_offset_seconds(start, crossing.time_seconds);
        let angle = angle_fn(time)?;
        if libm::cos((angle - target_deg).to_radians()) > 0.0 {
            times.push(time);
        }
    }
    Ok(times)
}

pub(crate) fn latch_scalar<F>(latch: &RefCell<Option<AlmanacError>>, f: F) -> f64
where
    F: FnOnce() -> Result<f64, AlmanacError>,
{
    match f() {
        Ok(value) if value.is_finite() => value,
        Ok(_) => {
            latch_error(
                latch,
                AlmanacError::InvalidInput {
                    field: "predicate",
                    reason: "not finite",
                },
            );
            f64::NAN
        }
        Err(error) => {
            latch_error(latch, error);
            f64::NAN
        }
    }
}

pub(crate) fn latched_or_finder(
    error: EventFinderError,
    latch: &RefCell<Option<AlmanacError>>,
) -> AlmanacError {
    latch
        .borrow()
        .clone()
        .unwrap_or(AlmanacError::Finder(error))
}

pub(crate) fn latch_error(latch: &RefCell<Option<AlmanacError>>, error: AlmanacError) {
    if latch.borrow().is_none() {
        *latch.borrow_mut() = Some(error);
    }
}

pub(crate) fn crossing_time(start: UtcInstant, crossing: CrossingEvent) -> UtcInstant {
    instant_at_offset_seconds(start, crossing.time_seconds)
}

pub(crate) fn planet_naif(planet: Planet) -> i32 {
    match planet {
        Planet::Mercury => 1,
        Planet::Venus => 2,
        Planet::Mars => 4,
        Planet::Jupiter => 5,
        Planet::Saturn => 6,
        Planet::Uranus => 7,
        Planet::Neptune => 8,
    }
}

pub(crate) fn transit_body_naif(body: TransitBody) -> i32 {
    match body {
        TransitBody::Sun => NAIF_SUN,
        TransitBody::Moon => NAIF_MOON,
        TransitBody::Planet(planet) => planet_naif(planet),
    }
}

pub(crate) fn is_inferior(planet: Planet) -> bool {
    matches!(planet, Planet::Mercury | Planet::Venus)
}

pub(crate) fn wrap360(degrees: f64) -> f64 {
    degrees.rem_euclid(360.0)
}

pub(crate) fn angular_separation_rad(a: [f64; 3], b: [f64; 3]) -> Result<f64, AlmanacError> {
    let na = norm_checked(a, "a")?;
    let nb = norm_checked(b, "b")?;
    let cos_sep = (dot3(a, b) / (na * nb)).clamp(-1.0, 1.0);
    Ok(libm::acos(cos_sep))
}

pub(crate) fn norm_checked(vector: [f64; 3], field: &'static str) -> Result<f64, AlmanacError> {
    if vector.iter().any(|value| !value.is_finite()) {
        return Err(AlmanacError::InvalidInput {
            field,
            reason: "components must be finite",
        });
    }
    let norm = norm3(vector);
    if !norm.is_finite() {
        return Err(AlmanacError::InvalidInput {
            field,
            reason: "norm must be finite",
        });
    }
    if norm == 0.0 {
        return Err(AlmanacError::InvalidInput {
            field,
            reason: "degenerate",
        });
    }
    Ok(norm)
}

pub(crate) fn map_field_error(error: validate::FieldError) -> AlmanacError {
    AlmanacError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn map_frame_input(error: FrameTransformError) -> AlmanacError {
    let FrameTransformError::InvalidInput { field, reason } = error;
    AlmanacError::InvalidInput { field, reason }
}
