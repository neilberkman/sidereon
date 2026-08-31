//! Event-finder-backed Sun elevation threshold crossings.
//!
//! This is intentionally a low-precision analytic-body helper: it uses the
//! existing Montenbruck-Gill Sun model from [`crate::astro::bodies::sun_moon`]
//! and the shared event finder to demonstrate the same crossing machinery on a
//! non-satellite predicate. Use it for geometric sunrise/sunset or twilight
//! thresholds where sub-degree solar-position accuracy is adequate; SPK-grade
//! almanac work belongs behind a higher-precision ephemeris source.

use crate::astro::almanac::{
    meridian_transits, AlmanacError, CulminationKind, EphemerisSource, TransitBody,
};
use crate::astro::bodies::observe::moon_az_el;
use crate::astro::bodies::sun_moon::sun_moon_ecef;
use crate::astro::constants::units::{MICROSECONDS_PER_SECOND, M_PER_KM};
use crate::astro::events::{
    CrossingDirection, CrossingEvent, EventFinder, EventFinderError, ScalarEventPredicate,
};
use crate::astro::frames::transforms::{geodetic_to_itrs, FrameTransformError, GeodeticStationKm};
use crate::astro::passes::UtcInstant;
use crate::validate;

/// Options for Sun elevation threshold crossings.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunElevationOptions {
    /// Topocentric Sun elevation threshold, degrees. Use `0.0` for geometric
    /// sunrise/sunset, or e.g. `-6.0` for civil twilight.
    pub elevation_threshold_deg: f64,
    /// Uniform event-finder scan step, seconds.
    pub step_seconds: f64,
    /// Crossing-time refinement tolerance, seconds.
    pub time_tolerance_seconds: f64,
}

impl Default for SunElevationOptions {
    fn default() -> Self {
        Self {
            elevation_threshold_deg: -0.833,
            step_seconds: 300.0,
            time_tolerance_seconds: 1.0,
        }
    }
}

/// Direction of a Sun elevation threshold crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SunElevationCrossingKind {
    /// The Sun crossed upward through the threshold.
    Rising,
    /// The Sun crossed downward through the threshold.
    Setting,
}

/// One refined Sun elevation threshold crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunElevationCrossing {
    /// Refined UTC instant of the crossing.
    pub time: UtcInstant,
    /// Crossing direction.
    pub kind: SunElevationCrossingKind,
    /// Topocentric Sun elevation at the refined instant, degrees.
    pub elevation_deg: f64,
}

/// Find Sun elevation threshold crossings for a station and UTC window.
///
/// The scalar predicate is topocentric Sun elevation in degrees. It is sampled
/// and refined by [`EventFinder::find_crossings`], so this follows the same
/// event-finder path used by the satellite pass finder rather than carrying a
/// separate rise/set bracketing implementation.
pub fn find_sun_elevation_crossings(
    station: &GeodeticStationKm,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: SunElevationOptions,
) -> Result<Vec<SunElevationCrossing>, EventFinderError> {
    validate_station(station)?;
    let crossings = elevation_crossings(
        start_time,
        end_time,
        options.elevation_threshold_deg,
        options.step_seconds,
        options.time_tolerance_seconds,
        |time| sun_elevation_deg(station, time),
    )?;

    Ok(crossings
        .into_iter()
        .map(|crossing| {
            let time = instant_at_offset_seconds(start_time, crossing.time_seconds);
            SunElevationCrossing {
                time,
                kind: match crossing.direction {
                    CrossingDirection::Rising => SunElevationCrossingKind::Rising,
                    CrossingDirection::Falling => SunElevationCrossingKind::Setting,
                },
                elevation_deg: sun_elevation_deg(station, time),
            }
        })
        .collect())
}

/// Topocentric geometric Sun elevation at a station and UTC instant, degrees.
pub fn sun_elevation_deg(station: &GeodeticStationKm, time: UtcInstant) -> f64 {
    let sun = sun_moon_ecef(&time.time_scales())
        .expect("UtcInstant time scales produce finite Sun/Moon geometry")
        .sun;
    let (station_x_km, station_y_km, station_z_km) = geodetic_to_itrs(
        station.latitude_deg,
        station.longitude_deg,
        station.altitude_km,
    )
    .expect("valid geodetic station for Sun elevation");
    let dx = sun[0] / M_PER_KM - station_x_km;
    let dy = sun[1] / M_PER_KM - station_y_km;
    let dz = sun[2] / M_PER_KM - station_z_km;
    let range = (dx * dx + dy * dy + dz * dz).sqrt();

    let lat = station.latitude_deg.to_radians();
    let lon = station.longitude_deg.to_radians();
    let up = [
        libm::cos(lat) * libm::cos(lon),
        libm::cos(lat) * libm::sin(lon),
        libm::sin(lat),
    ];
    let sin_elevation = ((up[0] * dx + up[1] * dy + up[2] * dz) / range).clamp(-1.0, 1.0);
    libm::asin(sin_elevation).to_degrees()
}

/// Options for Moon elevation threshold crossings (moonrise / moonset).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonElevationOptions {
    /// Topocentric Moon (disk-center) elevation threshold, degrees. The default
    /// `-0.833` is the standard upper-limb-on-the-horizon convention (about
    /// `34'` refraction plus the lunar semidiameter); topocentric parallax is
    /// already handled by the station-to-Moon geometry.
    pub elevation_threshold_deg: f64,
    /// Uniform event-finder scan step, seconds.
    pub step_seconds: f64,
    /// Crossing-time refinement tolerance, seconds.
    pub time_tolerance_seconds: f64,
}

impl Default for MoonElevationOptions {
    fn default() -> Self {
        Self {
            elevation_threshold_deg: -0.833,
            step_seconds: 300.0,
            time_tolerance_seconds: 1.0,
        }
    }
}

/// Direction of a Moon elevation threshold crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonElevationCrossingKind {
    /// The Moon crossed upward through the threshold (moonrise).
    Rising,
    /// The Moon crossed downward through the threshold (moonset).
    Setting,
}

/// One refined Moon elevation threshold crossing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonElevationCrossing {
    /// Refined UTC instant of the crossing.
    pub time: UtcInstant,
    /// Crossing direction.
    pub kind: MoonElevationCrossingKind,
    /// Topocentric Moon elevation at the refined instant, degrees.
    pub elevation_deg: f64,
}

/// Upper or lower culmination of the Moon (meridian transit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MoonTransitKind {
    /// Upper culmination: the Moon crosses the observer's local meridian due
    /// south (topocentric azimuth through 180 deg), highest in the sky.
    Upper,
    /// Lower culmination: the Moon crosses the local meridian due north
    /// (topocentric azimuth through 0/360 deg), lowest in the sky.
    Lower,
}

/// One refined Moon meridian transit (culmination).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MoonTransit {
    /// Refined UTC instant of the culmination.
    pub time: UtcInstant,
    /// Whether this is the upper or lower culmination.
    pub kind: MoonTransitKind,
    /// Topocentric Moon elevation at the refined instant, degrees.
    pub elevation_deg: f64,
}

/// Topocentric geometric Moon (disk-center) elevation at a station and UTC
/// instant, degrees.
///
/// Sibling of [`sun_elevation_deg`]. Unlike that low-precision geocentric-up
/// helper, this routes through the full station-to-target ENU reduction
/// ([`crate::astro::bodies::observe::moon_az_el`]), so it includes the
/// topocentric (diurnal) parallax that matters for the nearby Moon.
pub fn moon_elevation_deg(station: &GeodeticStationKm, time: UtcInstant) -> f64 {
    moon_az_el(station, time)
        .expect("UtcInstant time scales produce finite Moon geometry")
        .elevation_deg
}

/// Find Moon elevation threshold crossings (moonrise / moonset) for a station
/// and UTC window.
///
/// The direct sibling of [`find_sun_elevation_crossings`]: the topocentric Moon
/// elevation is sampled and refined by [`EventFinder::find_crossings`], the same
/// event-finder path the satellite pass finder uses.
pub fn find_moon_elevation_crossings(
    station: &GeodeticStationKm,
    start_time: UtcInstant,
    end_time: UtcInstant,
    options: MoonElevationOptions,
) -> Result<Vec<MoonElevationCrossing>, EventFinderError> {
    validate_station(station)?;
    let crossings = elevation_crossings(
        start_time,
        end_time,
        options.elevation_threshold_deg,
        options.step_seconds,
        options.time_tolerance_seconds,
        |time| moon_elevation_deg(station, time),
    )?;

    Ok(crossings
        .into_iter()
        .map(|crossing| {
            let time = instant_at_offset_seconds(start_time, crossing.time_seconds);
            MoonElevationCrossing {
                time,
                kind: match crossing.direction {
                    CrossingDirection::Rising => MoonElevationCrossingKind::Rising,
                    CrossingDirection::Falling => MoonElevationCrossingKind::Setting,
                },
                elevation_deg: moon_elevation_deg(station, time),
            }
        })
        .collect())
}

/// Find Moon meridian transits (upper and lower culminations) for a station and
/// UTC window.
pub fn find_moon_transits(
    station: &GeodeticStationKm,
    start_time: UtcInstant,
    end_time: UtcInstant,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<MoonTransit>, EventFinderError> {
    if end_time <= start_time {
        return Ok(Vec::new());
    }
    let transits = meridian_transits(
        EphemerisSource::Analytic,
        TransitBody::Moon,
        station,
        start_time,
        end_time,
        step_seconds,
        time_tolerance_seconds,
    )
    .map_err(map_almanac_error)?;

    Ok(transits
        .into_iter()
        .map(|transit| MoonTransit {
            time: transit.time,
            kind: match transit.kind {
                CulminationKind::Upper => MoonTransitKind::Upper,
                CulminationKind::Lower => MoonTransitKind::Lower,
            },
            elevation_deg: transit.altitude_deg,
        })
        .collect())
}

/// Run the event finder's threshold-crossing search over a topocentric elevation
/// closure. Shared by the Sun and Moon rise/set helpers.
fn elevation_crossings<F>(
    start_time: UtcInstant,
    end_time: UtcInstant,
    threshold_deg: f64,
    step_seconds: f64,
    time_tolerance_seconds: f64,
    elevation_fn: F,
) -> Result<Vec<CrossingEvent>, EventFinderError>
where
    F: Fn(UtcInstant) -> f64,
{
    if end_time <= start_time {
        return Ok(Vec::new());
    }
    let threshold =
        validate::finite(threshold_deg, "elevation_threshold_deg").map_err(map_event_input)?;
    let finder = elevation_finder(start_time, end_time, step_seconds, time_tolerance_seconds)?;
    finder.find_crossings(
        ClosurePredicate {
            start_time,
            value_fn: elevation_fn,
        },
        threshold,
    )
}

/// Validate the scan cadence and build the finder over `[0, span_seconds]`.
fn elevation_finder(
    start_time: UtcInstant,
    end_time: UtcInstant,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<EventFinder, EventFinderError> {
    let step_seconds =
        validate::positive_step(step_seconds, "step_seconds").map_err(map_event_input)?;
    let time_tolerance_seconds =
        validate::positive_step(time_tolerance_seconds, "time_tolerance_seconds")
            .map_err(map_event_input)?;
    let span_micros = end_time
        .unix_microseconds()
        .checked_sub(start_time.unix_microseconds())
        .ok_or(EventFinderError::InvalidInput {
            field: "time_window",
            reason: "start/end span overflows i64 microseconds",
        })?;
    let span_seconds = span_micros as f64 / MICROSECONDS_PER_SECOND;
    EventFinder::new(0.0, span_seconds, step_seconds, time_tolerance_seconds)
}

/// Scalar predicate backing the event finder, over a closure mapping a UTC
/// instant to a topocentric elevation.
struct ClosurePredicate<F> {
    start_time: UtcInstant,
    value_fn: F,
}

impl<F> ScalarEventPredicate for ClosurePredicate<F>
where
    F: Fn(UtcInstant) -> f64,
{
    fn value_at(&self, offset_seconds: f64) -> f64 {
        (self.value_fn)(instant_at_offset_seconds(self.start_time, offset_seconds))
    }
}

fn instant_at_offset_seconds(start_time: UtcInstant, offset_seconds: f64) -> UtcInstant {
    UtcInstant::from_unix_microseconds(
        start_time.unix_microseconds() + (offset_seconds * MICROSECONDS_PER_SECOND).floor() as i64,
    )
}

/// Validate a ground station's geodetic coordinates up front, returning a typed
/// [`EventFinderError`] so no public rise/set/transit path can panic on invalid
/// station input inside the per-sample elevation/azimuth reduction.
fn validate_station(station: &GeodeticStationKm) -> Result<(), EventFinderError> {
    geodetic_to_itrs(
        station.latitude_deg,
        station.longitude_deg,
        station.altitude_km,
    )
    .map(|_| ())
    .map_err(map_frame_input)
}

fn map_frame_input(error: FrameTransformError) -> EventFinderError {
    let FrameTransformError::InvalidInput { field, reason } = error;
    EventFinderError::InvalidInput { field, reason }
}

fn map_event_input(error: validate::FieldError) -> EventFinderError {
    EventFinderError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn map_almanac_error(error: AlmanacError) -> EventFinderError {
    match error {
        AlmanacError::Finder(error) => error,
        AlmanacError::InvalidInput { field, reason } => {
            EventFinderError::InvalidInput { field, reason }
        }
        AlmanacError::Frame(reason) => EventFinderError::InvalidInput {
            field: "frame",
            reason,
        },
        AlmanacError::Spk(_) => EventFinderError::InvalidInput {
            field: "spk",
            reason: "unexpected SPK error",
        },
        AlmanacError::EphemerisRequired => EventFinderError::InvalidInput {
            field: "source",
            reason: "ephemeris required",
        },
        AlmanacError::InferiorPlanetOpposition => EventFinderError::InvalidInput {
            field: "planet",
            reason: "inferior planet opposition",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn greenwich() -> GeodeticStationKm {
        GeodeticStationKm {
            latitude_deg: 51.4769,
            longitude_deg: 0.0,
            altitude_km: 0.046,
        }
    }

    fn day_start() -> UtcInstant {
        UtcInstant::from_utc(2024, 3, 20, 0, 0, 0, 0).expect("valid UTC")
    }

    #[test]
    fn sun_twilight_crossings_use_event_finder() {
        let station = greenwich();
        let start = day_start();
        let end = UtcInstant::from_utc(2024, 3, 21, 0, 0, 0, 0).expect("valid UTC");
        let options = SunElevationOptions {
            elevation_threshold_deg: -6.0,
            step_seconds: 900.0,
            time_tolerance_seconds: 1.0,
        };

        let events =
            find_sun_elevation_crossings(&station, start, end, options).expect("valid search");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, SunElevationCrossingKind::Rising);
        assert_eq!(events[1].kind, SunElevationCrossingKind::Setting);
        assert!(events[0].time < events[1].time);

        let rise_hour = hours_after(start, events[0].time);
        let set_hour = hours_after(start, events[1].time);
        assert!(
            (5.0..6.5).contains(&rise_hour),
            "unexpected civil-dawn hour {rise_hour}"
        );
        assert!(
            (18.0..19.5).contains(&set_hour),
            "unexpected civil-dusk hour {set_hour}"
        );
        for event in events {
            assert!(
                (event.elevation_deg - options.elevation_threshold_deg).abs() < 0.02,
                "refined elevation {} not near threshold",
                event.elevation_deg
            );
        }
    }

    #[test]
    fn sun_crossing_times_are_stable_under_finer_scan_step() {
        let station = greenwich();
        let start = day_start();
        let end = UtcInstant::from_utc(2024, 3, 21, 0, 0, 0, 0).expect("valid UTC");
        let coarse =
            find_sun_elevation_crossings(&station, start, end, SunElevationOptions::default())
                .expect("valid coarse search");
        let fine = find_sun_elevation_crossings(
            &station,
            start,
            end,
            SunElevationOptions {
                step_seconds: 60.0,
                ..SunElevationOptions::default()
            },
        )
        .expect("valid fine search");

        assert_eq!(coarse.len(), fine.len());
        assert_eq!(coarse.len(), 2);
        for (coarse_event, fine_event) in coarse.iter().zip(fine.iter()) {
            assert_eq!(coarse_event.kind, fine_event.kind);
            assert!(
                (coarse_event.time.unix_microseconds() - fine_event.time.unix_microseconds()).abs()
                    <= 1_000_000,
                "coarse and fine event times diverged"
            );
        }
    }

    #[test]
    fn sun_crossing_options_reject_invalid_steps() {
        let station = greenwich();
        let start = day_start();
        let end = UtcInstant::from_utc(2024, 3, 21, 0, 0, 0, 0).expect("valid UTC");

        let err = find_sun_elevation_crossings(
            &station,
            start,
            end,
            SunElevationOptions {
                step_seconds: 0.0,
                ..SunElevationOptions::default()
            },
        )
        .expect_err("zero step must be rejected");
        assert_invalid_field(err, "step_seconds", "not positive");

        let err = find_sun_elevation_crossings(
            &station,
            start,
            end,
            SunElevationOptions {
                time_tolerance_seconds: 0.0,
                ..SunElevationOptions::default()
            },
        )
        .expect_err("zero tolerance must be rejected");
        assert_invalid_field(err, "time_tolerance_seconds", "not positive");
    }

    fn moon_day_start() -> UtcInstant {
        UtcInstant::from_utc(2024, 4, 23, 0, 0, 0, 0).expect("valid UTC")
    }

    fn moon_day_end() -> UtcInstant {
        UtcInstant::from_utc(2024, 4, 24, 0, 0, 0, 0).expect("valid UTC")
    }

    #[test]
    fn moon_rise_and_set_match_reference() {
        // Greenwich, UTC day 2024-04-23. Skyfield (de421,
        // almanac.risings_and_settings) gives moonset 04:27:56 and moonrise
        // 19:00:01. The low-precision analytic series is held to 10 minutes.
        let station = greenwich();
        let events = find_moon_elevation_crossings(
            &station,
            moon_day_start(),
            moon_day_end(),
            MoonElevationOptions::default(),
        )
        .expect("valid search");

        assert_eq!(events.len(), 2);
        assert_eq!(events[0].kind, MoonElevationCrossingKind::Setting);
        assert_eq!(events[1].kind, MoonElevationCrossingKind::Rising);

        let set_hour = hours_after(moon_day_start(), events[0].time);
        let rise_hour = hours_after(moon_day_start(), events[1].time);
        assert!(
            (set_hour - 4.4656).abs() < 1.0 / 6.0,
            "moonset hour {set_hour}"
        );
        assert!(
            (rise_hour - 19.0003).abs() < 1.0 / 6.0,
            "moonrise hour {rise_hour}"
        );
        for event in events {
            assert!(
                (event.elevation_deg - MoonElevationOptions::default().elevation_threshold_deg)
                    .abs()
                    < 0.05,
                "refined elevation {} not near threshold",
                event.elevation_deg
            );
        }
    }

    #[test]
    fn moon_transits_match_reference() {
        // Greenwich, UTC day 2024-04-23. Skyfield (de421,
        // almanac.meridian_transits) gives lower culmination 11:34:44 and upper
        // culmination 23:55:59 (apparent topocentric altitude 23.120 deg there).
        // Held to 10 minutes in time and 0.5 deg in elevation.
        let station = greenwich();
        let transits = find_moon_transits(&station, moon_day_start(), moon_day_end(), 300.0, 1.0)
            .expect("valid search");

        assert_eq!(transits.len(), 2);
        assert_eq!(transits[0].kind, MoonTransitKind::Lower);
        assert_eq!(transits[1].kind, MoonTransitKind::Upper);

        let lower_hour = hours_after(moon_day_start(), transits[0].time);
        let upper_hour = hours_after(moon_day_start(), transits[1].time);
        assert!(
            (lower_hour - 11.5789).abs() < 1.0 / 6.0,
            "lower culmination hour {lower_hour}"
        );
        assert!(
            (upper_hour - 23.9331).abs() < 1.0 / 6.0,
            "upper culmination hour {upper_hour}"
        );
        assert!(
            (transits[1].elevation_deg - 23.120).abs() < 0.5,
            "upper culmination elevation {}",
            transits[1].elevation_deg
        );
        // The upper culmination is higher than the lower culmination.
        assert!(transits[1].elevation_deg > transits[0].elevation_deg);
    }

    #[test]
    fn moon_crossing_options_reject_invalid_steps() {
        let station = greenwich();
        let err = find_moon_elevation_crossings(
            &station,
            moon_day_start(),
            moon_day_end(),
            MoonElevationOptions {
                step_seconds: 0.0,
                ..MoonElevationOptions::default()
            },
        )
        .expect_err("zero step must be rejected");
        assert_invalid_field(err, "step_seconds", "not positive");
    }

    #[test]
    fn invalid_station_returns_typed_error_without_panic() {
        // An out-of-range latitude must surface as a typed InvalidInput from the
        // Result-returning public paths, never an unwind from the per-sample
        // azimuth/elevation reduction.
        let bad = GeodeticStationKm {
            latitude_deg: 120.0,
            longitude_deg: 0.0,
            altitude_km: 0.0,
        };

        let err = find_moon_transits(&bad, moon_day_start(), moon_day_end(), 300.0, 1.0)
            .expect_err("invalid station latitude must be rejected");
        assert_invalid_field(err, "latitude_deg", "must be in [-90, 90]");

        let err = find_moon_elevation_crossings(
            &bad,
            moon_day_start(),
            moon_day_end(),
            MoonElevationOptions::default(),
        )
        .expect_err("invalid station latitude must be rejected");
        assert_invalid_field(err, "latitude_deg", "must be in [-90, 90]");

        let err = find_sun_elevation_crossings(
            &bad,
            day_start(),
            UtcInstant::from_utc(2024, 3, 21, 0, 0, 0, 0).expect("valid UTC"),
            SunElevationOptions::default(),
        )
        .expect_err("invalid station latitude must be rejected");
        assert_invalid_field(err, "latitude_deg", "must be in [-90, 90]");
    }

    fn hours_after(start: UtcInstant, time: UtcInstant) -> f64 {
        (time.unix_microseconds() - start.unix_microseconds()) as f64
            / MICROSECONDS_PER_SECOND
            / crate::astro::constants::time::SECONDS_PER_HOUR
    }

    fn assert_invalid_field(
        error: EventFinderError,
        expected_field: &'static str,
        expected_reason: &'static str,
    ) {
        let EventFinderError::InvalidInput { field, reason } = error;
        assert_eq!(field, expected_field);
        assert_eq!(reason, expected_reason);
    }
}
