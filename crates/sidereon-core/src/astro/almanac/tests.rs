use super::*;
use crate::astro::frames::transforms::GeodeticStationKm;

fn spk() -> Spk {
    Spk::from_bytes(include_bytes!(
        "../../../tests/fixtures/almanac/almanac_de421.spk"
    ))
    .expect("almanac SPK fixture")
}

fn utc(
    year: i32,
    month: i32,
    day: i32,
    hour: i32,
    minute: i32,
    second: i32,
    microsecond: i32,
) -> UtcInstant {
    UtcInstant::from_utc(year, month, day, hour, minute, second, microsecond).expect("valid UTC")
}

fn seconds_between_abs(a: UtcInstant, b: UtcInstant) -> f64 {
    (a.unix_microseconds() - b.unix_microseconds()).abs() as f64 / 1_000_000.0
}

fn assert_within(actual: UtcInstant, expected: UtcInstant, tolerance_seconds: f64, label: &str) {
    let error = seconds_between_abs(actual, expected);
    assert!(
        error <= tolerance_seconds,
        "{label} time error {error} s exceeds {tolerance_seconds} s"
    );
}

fn greenwich() -> GeodeticStationKm {
    GeodeticStationKm {
        latitude_deg: 51.4769,
        longitude_deg: 0.0,
        altitude_km: 0.046,
    }
}

#[test]
fn seasons_match_reference_2025() {
    let spk = spk();
    let events = seasons(
        EphemerisSource::Spk(&spk),
        utc(2025, 1, 1, 0, 0, 0, 0),
        utc(2026, 1, 1, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("seasons");

    let expected = [
        (
            SeasonKind::MarchEquinox,
            utc(2025, 3, 20, 9, 1, 28, 934_474),
            utc(2025, 3, 20, 9, 1, 0, 0),
        ),
        (
            SeasonKind::JuneSolstice,
            utc(2025, 6, 21, 2, 42, 15, 683_236),
            utc(2025, 6, 21, 2, 42, 0, 0),
        ),
        (
            SeasonKind::SeptemberEquinox,
            utc(2025, 9, 22, 18, 19, 20, 499_855),
            utc(2025, 9, 22, 18, 19, 0, 0),
        ),
        (
            SeasonKind::DecemberSolstice,
            utc(2025, 12, 21, 15, 3, 5, 141_637),
            utc(2025, 12, 21, 15, 3, 0, 0),
        ),
    ];

    assert_eq!(events.len(), expected.len());
    for ((kind, skyfield_time, usno_time), event) in expected.into_iter().zip(events.iter()) {
        assert_eq!(event.kind, kind);
        assert_within(event.time, skyfield_time, 5.0, "Skyfield season");
        assert_within(event.time, usno_time, 60.0, "USNO season");
    }
    assert!(events.windows(2).all(|pair| pair[0].time < pair[1].time));
}

#[test]
fn moon_phases_match_reference_2025() {
    let spk = spk();
    let events = moon_phases(
        EphemerisSource::Spk(&spk),
        utc(2025, 1, 1, 0, 0, 0, 0),
        utc(2025, 2, 1, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("moon phases");

    let full = events
        .iter()
        .find(|event| event.kind == MoonPhaseKind::Full)
        .expect("full moon");
    let new = events
        .iter()
        .find(|event| event.kind == MoonPhaseKind::New)
        .expect("new moon");
    assert_within(
        full.time,
        utc(2025, 1, 13, 22, 26, 54, 547_309),
        5.0,
        "full moon",
    );
    assert_within(
        new.time,
        utc(2025, 1, 29, 12, 35, 58, 908_240),
        5.0,
        "new moon",
    );

    let mut previous =
        moon_phase_deg(EphemerisSource::Spk(&spk), utc(2025, 1, 2, 0, 0, 0, 0)).expect("phase");
    let mut unwrapped = previous;
    for day in 3..29 {
        let phase = moon_phase_deg(EphemerisSource::Spk(&spk), utc(2025, 1, day, 0, 0, 0, 0))
            .expect("phase");
        let mut delta = phase - previous;
        if delta < -180.0 {
            delta += 360.0;
        }
        unwrapped += delta;
        assert!(delta > 0.0, "phase did not advance on January {day}");
        previous = phase;
    }
    assert!(unwrapped > 300.0);
}

#[test]
fn planetary_opposition_matches_reference() {
    let spk = spk();
    let mars = planetary_events(
        EphemerisSource::Spk(&spk),
        Planet::Mars,
        PlanetaryEventKind::Opposition,
        utc(2025, 1, 1, 0, 0, 0, 0),
        utc(2025, 2, 1, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("mars opposition");
    assert_eq!(mars.len(), 1);
    assert_within(
        mars[0].time,
        utc(2025, 1, 16, 2, 38, 35, 259_890),
        5.0,
        "Mars opposition",
    );
    assert!((mars[0].elongation_deg - 180.0).abs() < 0.01);

    let saturn = planetary_events(
        EphemerisSource::Spk(&spk),
        Planet::Saturn,
        PlanetaryEventKind::Opposition,
        utc(2025, 9, 1, 0, 0, 0, 0),
        utc(2025, 10, 15, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("saturn opposition");
    assert_eq!(saturn.len(), 1);
    assert_within(
        saturn[0].time,
        utc(2025, 9, 21, 5, 45, 38, 456_186),
        5.0,
        "Saturn opposition",
    );
    assert!((saturn[0].elongation_deg - 180.0).abs() < 0.01);

    let venus_error = planetary_events(
        EphemerisSource::Spk(&spk),
        Planet::Venus,
        PlanetaryEventKind::Opposition,
        utc(2025, 1, 1, 0, 0, 0, 0),
        utc(2025, 12, 31, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect_err("Venus opposition must be rejected");
    assert_eq!(venus_error, AlmanacError::InferiorPlanetOpposition);

    let venus_conjunction = planetary_events(
        EphemerisSource::Spk(&spk),
        Planet::Venus,
        PlanetaryEventKind::Conjunction,
        utc(2025, 3, 1, 0, 0, 0, 0),
        utc(2025, 4, 15, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("Venus conjunction");
    assert!(!venus_conjunction.is_empty());
}

#[test]
fn meridian_transits_reproduce_moon_reference() {
    let station = greenwich();
    let start = utc(2024, 4, 23, 0, 0, 0, 0);
    let end = utc(2024, 4, 24, 0, 0, 0, 0);
    let legacy =
        crate::astro::bodies::rise_set::find_moon_transits(&station, start, end, 300.0, 1.0)
            .expect("legacy moon transits");
    let general = meridian_transits(
        EphemerisSource::Analytic,
        TransitBody::Moon,
        &station,
        start,
        end,
        300.0,
        1.0,
    )
    .expect("general moon transits");
    assert_eq!(legacy.len(), general.len());
    for (legacy, general) in legacy.iter().zip(general.iter()) {
        let expected_kind = match legacy.kind {
            crate::astro::bodies::rise_set::MoonTransitKind::Upper => CulminationKind::Upper,
            crate::astro::bodies::rise_set::MoonTransitKind::Lower => CulminationKind::Lower,
        };
        assert_eq!(general.kind, expected_kind);
        assert_within(general.time, legacy.time, 1.0, "Moon transit delegation");
    }

    let sun = meridian_transits(
        EphemerisSource::Analytic,
        TransitBody::Sun,
        &station,
        utc(2025, 3, 20, 0, 0, 0, 0),
        utc(2025, 3, 21, 0, 0, 0, 0),
        300.0,
        1.0,
    )
    .expect("Sun transits");
    assert!(sun.iter().any(|event| {
        event.kind == CulminationKind::Upper
            && (11.5..12.5).contains(
                &((event.time.unix_microseconds()
                    - utc(2025, 3, 20, 0, 0, 0, 0).unix_microseconds()) as f64
                    / 3_600_000_000.0),
            )
    }));
}

#[test]
fn eclipses_detect_and_classify() {
    let spk = spk();
    let lunar = lunar_solar_eclipses(
        EphemerisSource::Spk(&spk),
        utc(2025, 3, 14, 0, 0, 0, 0),
        utc(2025, 3, 15, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("lunar eclipse");
    let lunar = lunar
        .iter()
        .find(|event| event.kind == EclipseKind::LunarTotal)
        .expect("total lunar eclipse");
    assert_within(
        lunar.time_maximum,
        utc(2025, 3, 14, 6, 59, 0, 0),
        5.0 * 60.0,
        "total lunar eclipse",
    );
    assert!(lunar.magnitude > 1.1);
    assert!(!lunar.uncertain);

    let solar = lunar_solar_eclipses(
        EphemerisSource::Spk(&spk),
        utc(2026, 8, 12, 0, 0, 0, 0),
        utc(2026, 8, 13, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("solar eclipse");
    let solar = solar
        .iter()
        .find(|event| event.kind == EclipseKind::SolarTotal)
        .expect("total solar eclipse");
    assert_within(
        solar.time_maximum,
        utc(2026, 8, 12, 17, 46, 0, 0),
        5.0 * 60.0,
        "total solar eclipse",
    );
    assert!(
        (1.0..=1.08).contains(&solar.magnitude),
        "solar magnitude {}",
        solar.magnitude
    );
    assert!(
        (solar.gamma - 0.8978).abs() < 0.02,
        "solar gamma {}",
        solar.gamma
    );
    assert!(!solar.uncertain, "solar event should not be uncertain");

    let none = lunar_solar_eclipses(
        EphemerisSource::Spk(&spk),
        utc(2025, 2, 1, 0, 0, 0, 0),
        utc(2025, 2, 3, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("control window");
    assert!(none.is_empty());
}

#[test]
fn analytic_tier_looser_tolerance() {
    let season = seasons(
        EphemerisSource::Analytic,
        utc(2025, 3, 1, 0, 0, 0, 0),
        utc(2025, 4, 1, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("analytic season");
    assert_eq!(season.len(), 1);
    assert_within(
        season[0].time,
        utc(2025, 3, 20, 9, 1, 0, 0),
        15.0 * 60.0,
        "analytic season",
    );

    let phases = moon_phases(
        EphemerisSource::Analytic,
        utc(2025, 1, 10, 0, 0, 0, 0),
        utc(2025, 1, 31, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("analytic phases");
    assert!(phases.iter().any(|event| {
        event.kind == MoonPhaseKind::Full
            && seconds_between_abs(event.time, utc(2025, 1, 13, 22, 27, 0, 0)) <= 15.0 * 60.0
    }));
    assert!(phases.iter().any(|event| {
        event.kind == MoonPhaseKind::New
            && seconds_between_abs(event.time, utc(2025, 1, 29, 12, 36, 0, 0)) <= 15.0 * 60.0
    }));

    let lunar = lunar_solar_eclipses(
        EphemerisSource::Analytic,
        utc(2025, 3, 14, 0, 0, 0, 0),
        utc(2025, 3, 15, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("analytic eclipse");
    assert!(lunar
        .iter()
        .any(|event| event.kind == EclipseKind::LunarTotal));

    let error = planetary_events(
        EphemerisSource::Analytic,
        Planet::Mars,
        PlanetaryEventKind::Opposition,
        utc(2025, 1, 1, 0, 0, 0, 0),
        utc(2025, 2, 1, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect_err("analytic planet event must fail");
    assert_eq!(error, AlmanacError::EphemerisRequired);
}

#[test]
fn finder_edge_cases() {
    let spk = spk();
    let empty = seasons(
        EphemerisSource::Spk(&spk),
        utc(2025, 1, 1, 0, 0, 0, 0),
        utc(2025, 1, 10, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect("event-free window");
    assert!(empty.is_empty());

    let inverted = seasons(
        EphemerisSource::Spk(&spk),
        utc(2025, 1, 2, 0, 0, 0, 0),
        utc(2025, 1, 1, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect_err("inverted window");
    assert!(matches!(inverted, AlmanacError::Finder(_)));

    let invalid = seasons(
        EphemerisSource::Spk(&spk),
        utc(2025, 1, 1, 0, 0, 0, 0),
        utc(2025, 1, 2, 0, 0, 0, 0),
        f64::NAN,
        1.0,
    )
    .expect_err("non-finite step");
    assert!(matches!(invalid, AlmanacError::InvalidInput { .. }));

    let spk_error = seasons(
        EphemerisSource::Spk(&spk),
        utc(2027, 1, 1, 0, 0, 0, 0),
        utc(2027, 1, 10, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect_err("SPK coverage error");
    assert!(matches!(spk_error, AlmanacError::Spk(_)));

    let eclipse_error = lunar_solar_eclipses(
        EphemerisSource::Spk(&spk),
        utc(2027, 1, 1, 0, 0, 0, 0),
        utc(2027, 1, 10, 0, 0, 0, 0),
        21_600.0,
        1.0,
    )
    .expect_err("SPK coverage error through eclipse path");
    assert!(matches!(eclipse_error, AlmanacError::Spk(_)));
}

#[test]
fn meridian_transits_after_the_ut1_table_are_refused_or_reported() {
    use crate::astro::time::{DegradeReason, ValidityMode};

    let start = utc(2027, 9, 1, 0, 0, 0, 0);
    let end = utc(2027, 9, 2, 0, 0, 0, 0);
    let strict = meridian_transits(
        EphemerisSource::Analytic,
        TransitBody::Moon,
        &greenwich(),
        start,
        end,
        600.0,
        1.0,
    );
    assert!(
        matches!(
            strict,
            Err(AlmanacError::Ut1OutsideCoverage(
                DegradeReason::AfterCoverage
            ))
        ),
        "{strict:?}"
    );

    let permissive = meridian_transits_with_validity(
        EphemerisSource::Analytic,
        TransitBody::Moon,
        &greenwich(),
        start,
        end,
        600.0,
        1.0,
        ValidityMode::Permissive,
    )
    .expect("permissive transits");
    assert_eq!(permissive.degraded, Some(DegradeReason::AfterCoverage));
    assert!(
        !permissive.value.is_empty(),
        "the Moon crosses the meridian in a day"
    );

    // A window from inside the table to past its end is refused rather than
    // returning the transits before the edge.
    let straddle = meridian_transits(
        EphemerisSource::Analytic,
        TransitBody::Moon,
        &greenwich(),
        utc(2027, 7, 1, 0, 0, 0, 0),
        utc(2027, 7, 6, 0, 0, 0, 0),
        600.0,
        1.0,
    );
    assert!(
        matches!(
            straddle,
            Err(AlmanacError::Ut1OutsideCoverage(
                DegradeReason::AfterCoverage
            ))
        ),
        "{straddle:?}"
    );
}
