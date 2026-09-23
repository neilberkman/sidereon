use std::cell::RefCell;

use crate::astro::almanac::{
    crossing_time, event_finder, latch_scalar, latched_or_finder, transit_body_naif,
    validate_scan_controls, validate_station, AlmanacError, CulminationEvent, CulminationKind,
    EphemerisSource, TransitBody, TRANSIT_STEP_MAX_SECONDS,
};
use crate::astro::apparent::topocentric_apparent;
use crate::astro::frames::transforms::{GeodeticStationKm, Ut1Gate};
use crate::astro::passes::UtcInstant;
use crate::astro::time::{Validated, ValidityMode};

/// Meridian transits of a body from a station.
///
/// Refuses a window with an instant outside the UT1 table; see
/// [`meridian_transits_with_validity`].
pub fn meridian_transits(
    source: EphemerisSource<'_>,
    body: TransitBody,
    station: &GeodeticStationKm,
    start: UtcInstant,
    end: UtcInstant,
    step_seconds: f64,
    time_tolerance_seconds: f64,
) -> Result<Vec<CulminationEvent>, AlmanacError> {
    meridian_transits_with_validity(
        source,
        body,
        station,
        start,
        end,
        step_seconds,
        time_tolerance_seconds,
        ValidityMode::Strict,
    )
    .map(|validated| validated.value)
}

/// [`meridian_transits`] under an explicit UT1 [`ValidityMode`].
///
/// Every instant the search evaluates passes the UT1 policy.
/// [`ValidityMode::Strict`] refuses the search if any of them lies outside the
/// UT1 table, so a window reaching past the table is refused rather than
/// returning the transits before it. [`ValidityMode::Permissive`] evaluates
/// them all with the long-term UT1 and reports the first departure in
/// [`Validated::degraded`].
#[allow(clippy::too_many_arguments)]
pub fn meridian_transits_with_validity(
    source: EphemerisSource<'_>,
    body: TransitBody,
    station: &GeodeticStationKm,
    start: UtcInstant,
    end: UtcInstant,
    step_seconds: f64,
    time_tolerance_seconds: f64,
    mode: ValidityMode,
) -> Result<Validated<Vec<CulminationEvent>>, AlmanacError> {
    let gate = Ut1Gate::new(mode);
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
                    let ts = gate.admit(time.time_scales())?;
                    let apparent = topocentric_apparent(target_naif, station, &ts, source)?;
                    Ok(libm::sin(apparent.hour_angle_deg.to_radians()))
                })
            },
            0.0,
        )
        .map_err(|error| latched_or_finder(error, &latch))?;

    let mut events = Vec::new();
    for crossing in crossings {
        let time = crossing_time(start, crossing);
        let ts = gate.admit(time.time_scales())?;
        let apparent = topocentric_apparent(target_naif, station, &ts, source)?;
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
    Ok(gate.finish(events)?)
}
