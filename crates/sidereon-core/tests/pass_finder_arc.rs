#![cfg(sidereon_repo_tests)]
//! Validated pass-finding arc over a committed real ISS TLE and a real ground
//! station, plus the env-gated emitter that dumps it as a JSON fixture for the
//! Python binding's pytest.
//!
//! Truth model (no invented numbers): the dense-sample finder
//! ([`find_passes`]) is cross-checked against the engine's existing
//! pass-prediction path ([`predict_passes`]) run at a fine step -- the two must
//! agree on every pass to a tight tolerance -- and a deliberately coarse
//! [`predict_passes`] run is shown to MISS passes the dense finder keeps. The
//! TLE and topocentric path are already pinned elsewhere (Vallado SGP4 oracle,
//! frozen ISS-London look-angle golden), so freezing this arc is a regression
//! lock, not an oracle.

use std::path::PathBuf;

use sidereon_core::astro::passes::{
    find_passes, predict_passes, GroundStation, PassFinderOptions, PassPredictionOptions,
    SatellitePass, UtcInstant,
};
use sidereon_core::astro::tle::parse as parse_tle;

// Canonical ISS TLE (epoch 2018-184.80969102), the same lines used by the SGP4
// and topocentric goldens.
const ISS_L1: &str = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
const ISS_L2: &str = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";

const STATION: GroundStation = GroundStation {
    latitude_deg: 51.5074,
    longitude_deg: -0.1278,
    altitude_m: 80.0,
};

fn window() -> (UtcInstant, UtcInstant) {
    (
        UtcInstant::from_utc(2018, 7, 3, 12, 0, 0, 0).unwrap(),
        UtcInstant::from_utc(2018, 7, 4, 12, 0, 0, 0).unwrap(),
    )
}

/// Agreement tolerances. The dense finder and the fine reference bisect to the
/// same crossings, so AOS/LOS/culmination agree to well under a millisecond and
/// the elevation to far below a microdegree (observed: <300 us, <1e-9 deg).
const TIME_AGREEMENT_US: i64 = 1_000; // 1 ms
const ELEVATION_AGREEMENT_DEG: f64 = 1.0e-7;

#[test]
fn find_passes_agrees_with_reference_and_keeps_what_coarse_drops() {
    let elements = parse_tle(ISS_L1, ISS_L2)
        .unwrap()
        .elements
        .to_element_set()
        .expect("valid TLE bridge");
    let (start, end) = window();

    // Authoritative reference: the existing pass-prediction path at a fine step.
    let reference = predict_passes(
        &elements,
        STATION,
        start,
        end,
        PassPredictionOptions {
            min_elevation_deg: 0.0,
            step_seconds: 10,
        },
    )
    .expect("valid pass-prediction step");
    // The dense-sample finder at mask 0 must reproduce the reference passes.
    let mine = find_passes(
        &elements,
        STATION,
        start,
        end,
        PassFinderOptions {
            elevation_mask_deg: 0.0,
            coarse_step_seconds: 10.0,
            time_tolerance_seconds: 1.0e-3,
        },
    )
    .expect("valid pass-finder step");
    // A deliberately coarse step of the same reference method drops passes.
    let coarse = predict_passes(
        &elements,
        STATION,
        start,
        end,
        PassPredictionOptions {
            min_elevation_deg: 0.0,
            step_seconds: 900,
        },
    )
    .expect("valid pass-prediction step");

    assert!(
        reference.len() >= 2,
        "expected several passes in the window"
    );
    assert_eq!(
        mine.len(),
        reference.len(),
        "dense finder must find every reference pass"
    );
    assert!(
        coarse.len() < reference.len(),
        "coarse step ({}) should drop passes the dense finder keeps ({})",
        coarse.len(),
        reference.len()
    );

    for (m, r) in mine.iter().zip(reference.iter()) {
        assert!(
            (m.aos.unix_microseconds() - r.rise.unix_microseconds()).abs() < TIME_AGREEMENT_US,
            "AOS disagreement: {} vs {}",
            m.aos.unix_microseconds(),
            r.rise.unix_microseconds()
        );
        assert!(
            (m.los.unix_microseconds() - r.set.unix_microseconds()).abs() < TIME_AGREEMENT_US,
            "LOS disagreement: {} vs {}",
            m.los.unix_microseconds(),
            r.set.unix_microseconds()
        );
        assert!(
            (m.culmination.unix_microseconds() - r.max_elevation_time.unix_microseconds()).abs()
                < TIME_AGREEMENT_US,
            "culmination disagreement: {} vs {}",
            m.culmination.unix_microseconds(),
            r.max_elevation_time.unix_microseconds()
        );
        assert!(
            (m.max_elevation_deg - r.max_elevation_deg).abs() < ELEVATION_AGREEMENT_DEG,
            "max-elevation disagreement: {} vs {}",
            m.max_elevation_deg,
            r.max_elevation_deg
        );
    }

    // Frozen-bits regression lock on the first pass (full set cross-checked
    // Python-side against the dumped fixture).
    let first = mine[0];
    assert_eq!(first.aos.unix_microseconds(), 1_530_672_731_076_964);
    assert_eq!(first.los.unix_microseconds(), 1_530_673_217_359_923);
    assert_eq!(first.culmination.unix_microseconds(), 1_530_672_973_724_542);
    assert_eq!(first.max_elevation_deg.to_bits(), 0x4021_3c92_daf7_062a);

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture(&mine);
    }
}

/// Env-gated emitter (`SIDEREON_DUMP_FIXTURES=1`) that serializes the TLE,
/// station, window, options, and the found passes to the JSON fixture consumed
/// by the Python binding's pytest. Never runs in a normal `cargo test`.
fn dump_fixture(passes: &[SatellitePass]) {
    use serde_json::{json, Value};

    let (start, end) = window();
    let passes_json: Vec<Value> = passes
        .iter()
        .map(|p| {
            json!({
                "aos_unix_us": p.aos.unix_microseconds(),
                "los_unix_us": p.los.unix_microseconds(),
                "culmination_unix_us": p.culmination.unix_microseconds(),
                "max_elevation_deg": p.max_elevation_deg,
                "max_elevation_deg_hex": format!("0x{:016x}", p.max_elevation_deg.to_bits()),
            })
        })
        .collect();

    let doc = json!({
        "source": "pass_finder_arc",
        "tle": { "line1": ISS_L1, "line2": ISS_L2 },
        "station": {
            "latitude_deg": STATION.latitude_deg,
            "longitude_deg": STATION.longitude_deg,
            "altitude_m": STATION.altitude_m,
        },
        "window": {
            "start_unix_us": start.unix_microseconds(),
            "end_unix_us": end.unix_microseconds(),
        },
        "options": {
            "elevation_mask_deg": 0.0,
            "coarse_step_seconds": 10.0,
            "time_tolerance_seconds": 1.0e-3,
        },
        "passes": passes_json,
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/pass_finder.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped pass finder fixture to {out:?}");
}
