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
    /// The Sun's ecliptic longitude crossing at `0` degrees.
    MarchEquinox,
    /// The Sun's ecliptic longitude crossing at `90` degrees.
    JuneSolstice,
    /// The Sun's ecliptic longitude crossing at `180` degrees.
    SeptemberEquinox,
    /// The Sun's ecliptic longitude crossing at `270` degrees.
    DecemberSolstice,
}

/// Principal lunar phase.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonPhaseKind {
    /// The wrapped Moon-minus-Sun ecliptic longitude is `0` degrees.
    New,
    /// The wrapped Moon-minus-Sun ecliptic longitude is `90` degrees.
    FirstQuarter,
    /// The wrapped Moon-minus-Sun ecliptic longitude is `180` degrees.
    Full,
    /// The wrapped Moon-minus-Sun ecliptic longitude is `270` degrees.
    LastQuarter,
}

/// Planetary ecliptic-longitude event.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanetaryEventKind {
    /// The planet-minus-Sun ecliptic longitude is `0` degrees.
    Conjunction,
    /// The planet-minus-Sun ecliptic longitude is `180` degrees; `planetary_events` rejects this request for Mercury and Venus.
    Opposition,
}

/// Meridian transit kind.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CulminationKind {
    /// The cosine of the topocentric apparent hour angle is positive.
    Upper,
    /// The cosine of the topocentric apparent hour angle is negative; a zero-cosine crossing is skipped.
    Lower,
}

/// Lunar or solar eclipse kind.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EclipseKind {
    /// The Moon's apparent disk overlaps the penumbral shadow after the umbral tests fail.
    LunarPenumbral,
    /// The Moon's apparent disk overlaps the umbra without fitting fully inside it.
    LunarPartial,
    /// The Moon's apparent disk fits fully within the umbral shadow.
    LunarTotal,
    /// The apparent disks do not meet the total or annular containment tests, including the non-overlap case.
    SolarPartial,
    /// The Moon's apparent disk is smaller than the Sun's and is fully inside it.
    SolarAnnular,
    /// The Moon's apparent disk is at least as large as the Sun's and fully covers it.
    SolarTotal,
    /// The solar eclipse switches between total and annular across Earth's near and far intersections.
    SolarHybrid,
}

/// Planet selector for almanac event finders.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Planet {
    /// NAIF body ID `1`.
    Mercury,
    /// NAIF body ID `2`.
    Venus,
    /// NAIF body ID `4`.
    Mars,
    /// NAIF body ID `5`.
    Jupiter,
    /// NAIF body ID `6`.
    Saturn,
    /// NAIF body ID `7`.
    Uranus,
    /// NAIF body ID `8`.
    Neptune,
}

/// Body selector for meridian transits.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitBody {
    /// The Sun, with NAIF body ID `10`.
    Sun,
    /// The Moon, with NAIF body ID `301`.
    Moon,
    /// A planet resolved through `planet_naif`.
    Planet(Planet),
}

/// One seasonal marker event.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeasonEvent {
    /// UTC instant of the crossing; returned events are sorted by this field.
    pub time: UtcInstant,
    /// The seasonal marker whose Sun-longitude target produced the crossing.
    pub kind: SeasonKind,
}

/// One lunar phase event.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoonPhaseEvent {
    /// UTC instant of the phase crossing; returned events are sorted by this field.
    pub time: UtcInstant,
    /// The phase whose Moon-minus-Sun angle target produced the crossing.
    pub kind: MoonPhaseKind,
}

/// One planetary opposition or conjunction.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlanetaryEvent {
    /// UTC instant of the planet-minus-Sun longitude crossing.
    pub time: UtcInstant,
    /// The planet supplied to [`planetary_events`].
    pub planet: Planet,
    /// The conjunction or opposition supplied to [`planetary_events`].
    pub kind: PlanetaryEventKind,
    /// Wrapped planet-minus-Sun ecliptic longitude at `time`, in degrees on `[0, 360)`.
    pub elongation_deg: f64,
}

/// One upper or lower culmination.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CulminationEvent {
    /// UTC instant of the hour-angle crossing.
    pub time: UtcInstant,
    /// Whether the crossing is upper or lower, based on the sign of the hour-angle cosine.
    pub kind: CulminationKind,
    /// Topocentric apparent geometric altitude at `time`, in degrees without refraction.
    pub altitude_deg: f64,
}

/// One lunar or solar eclipse event.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EclipseEvent {
    /// UTC instant of the selected eclipse maximum within the requested window.
    pub time_maximum: UtcInstant,
    /// The lunar shadow or solar disk-overlap classification.
    pub kind: EclipseKind,
    /// For lunar events, `(rho_u + s_moon - sigma) / (2.0 * s_moon)` is used for total or partial events and `(rho_p + s_moon - sigma) / (2.0 * s_moon)` for penumbral events; solar events use `max(0.0, (s_sun + s_moon - sep) / (2.0 * s_sun))`.
    pub magnitude: f64,
    /// Apparent geocentric ecliptic latitude of the Moon at `time_maximum`, in degrees.
    pub moon_latitude_deg: f64,
    /// `0.0` for lunar events; for solar events, the norm of the shadow axis's perpendicular offset from Earth's center divided by the mean Earth radius.
    pub gamma: f64,
    /// True when a lunar shadow boundary, solar shadow-reach boundary, or solar limb boundary is within `0.01`, or when near/far intersections switch the solar class.
    pub uncertain: bool,
}

/// Error returned by almanac computations.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum AlmanacError {
    /// An event-finder window or search returned an error.
    Finder(EventFinderError),
    /// An SPK-backed apparent-place reduction returned an error.
    Spk(SpkError),
    /// A frame reduction failed; the payload identifies the reduction stage.
    Frame(&'static str),
    /// An analytic source was asked for a body other than the Sun or Moon, or for a planetary event or transit that requires SPK data.
    EphemerisRequired,
    /// Opposition was requested for Mercury or Venus.
    InferiorPlanetOpposition,
    /// A scan control, station coordinate, predicate, vector, geometry, or other intermediate quantity failed a finiteness, positivity, range, or degeneracy check.
    InvalidInput {
        /// Static label identifying the rejected input or intermediate quantity, such as `step_seconds`, `predicate`, `geometry`, or a vector label.
        field: &'static str,
        /// Static explanation supplied by validation, such as `not finite`, `exceeds maximum`, `degenerate`, or `components must be finite`.
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
