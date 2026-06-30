//! Event-finder-backed Sun elevation threshold crossings.
//!
//! This is intentionally a low-precision analytic-body helper: it uses the
//! existing Montenbruck-Gill Sun model from [`crate::astro::bodies::sun_moon`]
//! and the shared event finder to demonstrate the same crossing machinery on a
//! non-satellite predicate. Use it for geometric sunrise/sunset or twilight
//! thresholds where sub-degree solar-position accuracy is adequate; SPK-grade
//! almanac work belongs behind a higher-precision ephemeris source.

use crate::astro::bodies::sun_moon::sun_moon_ecef;
use crate::astro::constants::units::{MICROSECONDS_PER_SECOND, M_PER_KM};
use crate::astro::events::{
    CrossingDirection, EventFinder, EventFinderError, ScalarEventPredicate,
};
use crate::astro::frames::transforms::{geodetic_to_itrs, GeodeticStationKm};
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
    if end_time <= start_time {
        return Ok(Vec::new());
    }

    let threshold = validate::finite(options.elevation_threshold_deg, "elevation_threshold_deg")
        .map_err(map_event_input)?;
    let step_seconds =
        validate::positive_step(options.step_seconds, "step_seconds").map_err(map_event_input)?;
    let time_tolerance_seconds =
        validate::positive_step(options.time_tolerance_seconds, "time_tolerance_seconds")
            .map_err(map_event_input)?;
    let span_seconds = (end_time.unix_microseconds() - start_time.unix_microseconds()) as f64
        / MICROSECONDS_PER_SECOND;
    let predicate = SunElevationPredicate {
        station,
        start_time,
    };

    EventFinder::new(0.0, span_seconds, step_seconds, time_tolerance_seconds)?
        .find_crossings(predicate, threshold)?
        .into_iter()
        .map(|crossing| {
            let time = instant_at_offset_seconds(start_time, crossing.time_seconds);
            Ok(SunElevationCrossing {
                time,
                kind: match crossing.direction {
                    CrossingDirection::Rising => SunElevationCrossingKind::Rising,
                    CrossingDirection::Falling => SunElevationCrossingKind::Setting,
                },
                elevation_deg: sun_elevation_deg(station, time),
            })
        })
        .collect()
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
    let up = [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()];
    let sin_elevation = ((up[0] * dx + up[1] * dy + up[2] * dz) / range).clamp(-1.0, 1.0);
    sin_elevation.asin().to_degrees()
}

#[derive(Clone, Copy)]
struct SunElevationPredicate<'a> {
    station: &'a GeodeticStationKm,
    start_time: UtcInstant,
}

impl ScalarEventPredicate for SunElevationPredicate<'_> {
    fn value_at(&self, offset_seconds: f64) -> f64 {
        sun_elevation_deg(
            self.station,
            instant_at_offset_seconds(self.start_time, offset_seconds),
        )
    }
}

fn instant_at_offset_seconds(start_time: UtcInstant, offset_seconds: f64) -> UtcInstant {
    UtcInstant::from_unix_microseconds(
        start_time.unix_microseconds() + (offset_seconds * MICROSECONDS_PER_SECOND).floor() as i64,
    )
}

fn map_event_input(error: validate::FieldError) -> EventFinderError {
    EventFinderError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
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

    fn hours_after(start: UtcInstant, time: UtcInstant) -> f64 {
        (time.unix_microseconds() - start.unix_microseconds()) as f64
            / MICROSECONDS_PER_SECOND
            / 3600.0
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
