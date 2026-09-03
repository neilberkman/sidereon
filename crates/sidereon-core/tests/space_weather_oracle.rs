//! Provenance for committed CelesTrak CSSI space-weather fixtures.
//!
//! CSV source URL: https://celestrak.org/SpaceData/SW-All.csv
//! TXT source URL: https://celestrak.org/SpaceData/SW-All.txt
//! Retrieved: 2026-07-02T17:50:26Z
//! Full CSV SHA-256: 1bb9ee6e102d5dd88b94a9850fe53c5ab2d30bd129206c6e358d684d6757397d
//! Full TXT SHA-256: d0547ac0117eb8aef2ebe2c22d66917fa6ea8c69f2fc4c8eed337bb9a55d689b
//! Trim CSV SHA-256: c9c339aaccc4be3bc20f94fcecfd26a68cc54152159a08e7809050408d7a1310
//! Trim TXT SHA-256: 815cf91852aad32681ba88a41d9a521395b14fa9f855dae684c3ebb214c245a5

use sidereon_core::astro::constants::{MU_EARTH, RE_EARTH};
use sidereon_core::astro::forces::{DragForce, DragParameters, SpaceWeather, SpaceWeatherSource};
use sidereon_core::astro::propagator::{
    estimate_decay, estimate_decay_with_source, DecayConfig, DecayEstimate, IntegratorOptions,
};
use sidereon_core::astro::space_weather::{
    encode_csv, encode_txt, parse_csv, parse_txt, ObservationClass, SpaceWeatherDay,
    SpaceWeatherTable,
};
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::time::civil::j2000_seconds;
use std::sync::Arc;

const CSV: &str = include_str!("fixtures/space_weather/SW-All-20260702-trim.csv");
const TXT: &str = include_str!("fixtures/space_weather/SW-All-20260702-trim.txt");

fn csv_table() -> SpaceWeatherTable {
    parse_csv(CSV).expect("CSV fixture parses").value
}

fn assert_space_weather_bits(actual: SpaceWeather, expected: SpaceWeather) {
    assert_eq!(actual.f107.to_bits(), expected.f107.to_bits());
    assert_eq!(actual.f107a.to_bits(), expected.f107a.to_bits());
    assert_eq!(actual.ap.to_bits(), expected.ap.to_bits());
}

fn assert_estimate_bits(left: DecayEstimate, right: DecayEstimate) {
    assert_eq!(
        left.time_to_decay_s.to_bits(),
        right.time_to_decay_s.to_bits()
    );
    assert_eq!(
        left.reentry_altitude_km.to_bits(),
        right.reentry_altitude_km.to_bits()
    );
    assert_eq!(
        left.reentry_state.epoch_tdb_seconds.to_bits(),
        right.reentry_state.epoch_tdb_seconds.to_bits()
    );
    for idx in 0..3 {
        assert_eq!(
            left.reentry_state.position_array()[idx].to_bits(),
            right.reentry_state.position_array()[idx].to_bits()
        );
        assert_eq!(
            left.reentry_state.velocity_array()[idx].to_bits(),
            right.reentry_state.velocity_array()[idx].to_bits()
        );
    }
}

fn circular_state(epoch: f64, altitude_km: f64) -> CartesianState {
    let radius_km = RE_EARTH + altitude_km;
    let speed_km_s = (MU_EARTH / radius_km).sqrt();
    CartesianState::new(epoch, [radius_km, 0.0, 0.0], [0.0, speed_km_s, 0.0])
}

fn decay_config(space_weather: SpaceWeather) -> DecayConfig {
    let drag = DragParameters::from_bc_factor_m2_kg(
        0.8,
        space_weather,
        DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
    )
    .expect("valid drag");
    let mut integrator_options = IntegratorOptions::default();
    integrator_options.abs_tol = 1.0e-8;
    integrator_options.rel_tol = 1.0e-10;
    integrator_options.initial_step = 5.0;
    integrator_options.min_step = 1.0e-6;
    integrator_options.max_step = 30.0;
    integrator_options.max_steps = 200_000;
    integrator_options.dense_output = false;
    DecayConfig::new(drag)
        .with_options(integrator_options)
        .with_scan_step_s(60.0)
        .with_crossing_tolerance_s(2.0)
        .with_max_duration_s(50_000.0)
        .with_max_scan_samples(2_000)
}

#[test]
fn hand_verified_space_weather_lookups_match_real_fixture_rows() {
    let table = csv_table();

    // Source rows:
    // 2026-06-30,2630,18,3,3,7,40,33,47,47,33,213,2,2,3,27,18,39,39,18,18,1.0,5,126,202.6,209.3,OBS,145.7,129.2,150.2,132.2
    // 2026-07-01,2630,19,40,30,13,13,33,23,27,17,197,27,15,5,5,18,9,12,6,12,0.7,3,148,249.7,258.1,OBS,145.9,131.1,150.4,134.2
    let modern = table
        .sample_at(j2000_seconds(2026, 7, 1, 12, 0, 0.0))
        .expect("modern observed lookup");
    assert_space_weather_bits(
        modern.space_weather,
        SpaceWeather {
            f107: 202.6,
            f107a: 145.9,
            ap: 12.0,
        },
    );
    assert_eq!(modern.class, ObservationClass::Observed);
    assert!(!modern.ap_defaulted);

    // Source rows:
    // 2003-10-29,2323,27,47,40,90,80,77,77,87,87,583,39,27,400,207,179,179,300,300,204,2.1,9,250,291.7,287.7,OBS,146.8,127.6,144.8,128.4
    // 2003-10-30,2324,1,87,73,53,47,50,70,90,90,560,300,154,56,39,48,132,400,400,191,2.1,9,250,271.4,267.6,OBS,146.5,129.3,144.6,130.0
    let storm = table
        .sample_at(j2000_seconds(2003, 10, 30, 12, 0, 0.0))
        .expect("storm lookup");
    assert_space_weather_bits(
        storm.space_weather,
        SpaceWeather {
            f107: 291.7,
            f107a: 146.5,
            ap: 191.0,
        },
    );
    assert_eq!(storm.class, ObservationClass::Observed);

    // Source rows:
    // 2026-07-02,2630,20,13,7,7,7,13,3,27,27,103,5,3,3,3,5,2,12,12,5,0.3,1,123,198.3,205.0,PRD,146.0,132.3,150.6,135.5
    // 2026-07-03,2630,21,39,39,39,39,39,39,39,39,312,25,25,25,25,25,25,25,25,25,1.2,6,91,193.5,200.0,PRD,146.0,133.5,150.6,136.8
    let predicted = table
        .sample_at(j2000_seconds(2026, 7, 3, 12, 0, 0.0))
        .expect("daily predicted lookup");
    assert_space_weather_bits(
        predicted.space_weather,
        SpaceWeather {
            f107: 198.3,
            f107a: 146.0,
            ap: 25.0,
        },
    );
    assert_eq!(predicted.class, ObservationClass::DailyPredicted);

    // Source row:
    // 2026-09-01,2632,27,,,,,,,,,,,,,,,,,,,,,87,118.9,121.1,PRM,128.7,142.6,131.0,146.9
    // Monthly-predicted rows have blank AP_AVG, so Ap defaults to 4.0.
    let monthly = table
        .sample_at(j2000_seconds(2026, 9, 15, 12, 0, 0.0))
        .expect("monthly predicted lookup");
    assert_space_weather_bits(
        monthly.space_weather,
        SpaceWeather {
            f107: 118.9,
            f107a: 128.7,
            ap: 4.0,
        },
    );
    assert_eq!(monthly.class, ObservationClass::MonthlyPredicted);
    assert!(monthly.ap_defaulted);
}

#[test]
fn hand_verified_ap_array_uses_real_three_hour_rows() {
    let table = csv_table();
    let ap = table
        .ap_array_at(j2000_seconds(2003, 10, 31, 13, 0, 0.0))
        .expect("AP history");

    // Source row:
    // 2003-10-31,2324,2,83,77,73,67,73,47,40,43,503,236,179,154,111,154,39,27,32,116,1.9,8,239,248.9,245.2,OBS,146.2,130.8,144.2,131.4
    // Current 12-15 UT bin is AP5 = 154; prior bins are AP4 = 111,
    // AP3 = 154, AP2 = 179.
    // Mean 12..33 h before = (154+56+39+48+132+400+400+236)/8 = 183.125.
    // Mean 36..57 h before = (27+400+207+179+179+300+300+300)/8 = 236.5.
    assert_eq!(ap[0], 116.0);
    assert_eq!(ap[1], 154.0);
    assert_eq!(ap[2], 111.0);
    assert_eq!(ap[3], 154.0);
    assert_eq!(ap[4], 179.0);
    assert_eq!(ap[5], 183.125);
    assert_eq!(ap[6], 236.5);
}

#[test]
fn real_fixtures_round_trip_and_match_across_formats() {
    let csv = parse_csv(CSV).expect("CSV fixture parses");
    assert!(csv.diagnostics.is_empty());
    assert_eq!(encode_csv(&csv.value), CSV);

    let txt = parse_txt(TXT).expect("TXT fixture parses");
    assert!(txt.diagnostics.is_empty());
    assert_eq!(encode_txt(&txt.value), TXT);

    assert_eq!(
        normalized_days(csv.value.days()),
        normalized_days(txt.value.days())
    );
    assert_eq!(
        normalized_days(csv.value.monthly()),
        normalized_days(txt.value.monthly())
    );
}

#[test]
fn decay_with_fixed_source_is_bit_identical_and_table_direction_is_sane() {
    let initial = circular_state(j2000_seconds(2003, 10, 30, 12, 0, 0.0), 125.0);
    let config = decay_config(SpaceWeather::default());
    let fixed = SpaceWeatherSource::Fixed(SpaceWeather::default());
    let baseline = estimate_decay(initial, &config).expect("fixed decay");
    let sourced =
        estimate_decay_with_source(initial, &config, &fixed).expect("sourced fixed decay");
    assert_estimate_bits(baseline, sourced);

    let table = Arc::new(csv_table());
    let active = estimate_decay_with_source(initial, &config, &SpaceWeatherSource::Table(table))
        .expect("table decay");
    assert!(active.time_to_decay_s < baseline.time_to_decay_s);

    let quiet_initial = circular_state(j2000_seconds(2026, 1, 7, 12, 0, 0.0), 125.0);
    let quiet_table = Arc::new(csv_table());
    let quiet_fixed = estimate_decay(quiet_initial, &config).expect("quiet fixed decay");
    let quiet_sourced = estimate_decay_with_source(
        quiet_initial,
        &config,
        &SpaceWeatherSource::Table(quiet_table),
    )
    .expect("quiet table decay");
    let rel = ((quiet_sourced.time_to_decay_s - quiet_fixed.time_to_decay_s)
        / quiet_fixed.time_to_decay_s)
        .abs();
    assert!(rel <= 0.25, "relative lifetime difference {rel}");
}

fn normalized_days(days: &[SpaceWeatherDay]) -> Vec<SpaceWeatherDay> {
    days.iter()
        .map(|row| {
            let mut row = *row;
            row.flux_qualifier = None;
            row
        })
        .collect()
}
