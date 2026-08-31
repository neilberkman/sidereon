use std::cell::RefCell;

use crate::astro::almanac::{
    crossing_time, event_finder, latch_scalar, latched_or_finder, transit_body_naif,
    validate_scan_controls, validate_station, AlmanacError, CulminationEvent, CulminationKind,
    EphemerisSource, TransitBody, TRANSIT_STEP_MAX_SECONDS,
};
use crate::astro::apparent::topocentric_apparent;
use crate::astro::frames::transforms::GeodeticStationKm;
use crate::astro::passes::UtcInstant;

/// Meridian transits of a body from a station.
pub fn meridian_transits(
    source: EphemerisSource<'_>,
    body: TransitBody,
    station: &GeodeticStationKm,
    start: UtcInstant,
    end: UtcInstant,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<CulminationEvent>, AlmanacError> {
    validate_scan_controls(
        step_seconds,
        time_tolerance_seconds,
        TRANSIT_STEP_MAX_SECONDS,
    )?;
    validate_station(station)?;
    if matches!(
        (source, body),
        (EphemerisSource::Analytic, TransitBody::Planet(_))
    ) {
        return Err(AlmanacError::EphemerisRequired);
    }

    let target_naif = transit_body_naif(body);
    let finder = event_finder(start, end, step_seconds, time_tolerance_seconds)?;
    let latch = RefCell::new(None);
    let crossings = finder
        .find_crossings(
            |offset_seconds| {
                latch_scalar(&latch, || {
                    let time =
                        crate::astro::almanac::instant_at_offset_seconds(start, offset_seconds);
                    let apparent =
                        topocentric_apparent(target_naif, station, &time.time_scales(), source)?;
                    Ok(libm::sin(apparent.hour_angle_deg.to_radians()))
                })
            },
            0.0,
        )
        .map_err(|error| latched_or_finder(error, &latch))?;

    let mut events = Vec::new();
    for crossing in crossings {
        let time = crossing_time(start, crossing);
        let apparent = topocentric_apparent(target_naif, station, &time.time_scales(), source)?;
        let cos_h = libm::cos(apparent.hour_angle_deg.to_radians());
        let kind = if cos_h > 0.0 {
            CulminationKind::Upper
        } else if cos_h < 0.0 {
            CulminationKind::Lower
        } else {
            continue;
        };
        events.push(CulminationEvent {
            time,
            kind,
            altitude_deg: apparent.altitude_deg,
        });
    }
    Ok(events)
}
