#![cfg(sidereon_repo_tests)]
//! SGP4 + topocentric arc over the canonical ISS element set from London,
//! and the env-gated emitter that dumps it as a JSON fixture for the Python
//! binding's pytest.
//!
//! The arc runs along the path the binding uses
//! (`Satellite::from_tle_with_opsmode(.., Improved)` →
//! `passes::propagate_teme_arc` / `passes::look_angle_arc`). The same ten
//! instants from the same station are checked against Skyfield 1.54 in
//! `skyfield_passes.rs` (the fixture's ISS rows from London at 80 m); this
//! test checks the arc walkers against the single-instant calls.

use std::path::PathBuf;

use sidereon_core::astro::passes::{
    look_angle, look_angle_arc, propagate_teme_arc, GroundStation, UtcInstant,
};
use sidereon_core::astro::sgp4::{JulianDate, OpsMode, Satellite};
use sidereon_core::astro::tle;

// Canonical ISS TLE (epoch 2018-184.80969102). Real, committed, and validated:
// the same two lines appear in the `tle` and `sgp4` module doctests/tests.
const ISS_L1: &str = "1 25544U 98067A   18184.80969102  .00001614  00000-0  31745-4 0  9993";
const ISS_L2: &str = "2 25544  51.6414 295.8524 0003435 262.6267 204.2868 15.54005638121106";

// London ground station, matching the crate's topocentric goldens' site family.
const STATION: GroundStation = GroundStation {
    latitude_deg: 51.5074,
    longitude_deg: -0.1278,
    altitude_m: 80.0,
};

/// Epoch grid: ten one-minute steps from just after the TLE epoch, where SGP4
/// is most accurate. Returned as `UtcInstant`s plus their unix-microsecond keys.
fn epoch_grid() -> Vec<UtcInstant> {
    (0..10)
        .map(|i| UtcInstant::from_utc(2018, 7, 3, 19, 30 + i, 0, 0).unwrap())
        .collect()
}

fn build_satellite() -> Satellite {
    Satellite::from_tle_with_opsmode(ISS_L1, ISS_L2, OpsMode::Improved)
        .expect("committed ISS TLE initializes")
}

#[test]
fn iss_arc_walkers_match_single_instant_calls() {
    let satellite = build_satellite();
    let epochs = epoch_grid();

    let positions = propagate_teme_arc(&satellite, &epochs).expect("propagate arc");
    let looks = look_angle_arc(&satellite, STATION, &epochs).expect("look-angle arc");

    assert_eq!(positions.len(), 10);
    assert_eq!(looks.len(), 10);

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture(&epochs, &positions, &looks);
    }

    let elements = tle::parse(ISS_L1, ISS_L2)
        .expect("committed ISS TLE parses")
        .elements
        .to_element_set()
        .expect("element set");
    for (index, ((dt, pos), look)) in epochs.iter().zip(&positions).zip(&looks).enumerate() {
        let single = satellite
            .propagate_jd(JulianDate::from_unix_microseconds(dt.unix_microseconds()))
            .expect("single-instant propagation");
        assert_eq!(&single, pos, "{index}");
        assert_eq!(
            &look_angle(&elements, STATION, *dt).expect("single-instant look angle"),
            look,
            "{index}"
        );
    }
}

/// Env-gated emitter (`SIDEREON_DUMP_FIXTURES=1`) that serializes the committed
/// TLE, station, epoch grid, and the engine's TEME states and topocentric look
/// angles (raw f64 plus IEEE-754 hex bits) to the JSON fixture the Python
/// binding's pytest compares the binding against. It never runs in a normal
/// `cargo test`.
fn dump_fixture(
    epochs: &[UtcInstant],
    positions: &[sidereon_core::astro::sgp4::Prediction],
    looks: &[sidereon_core::astro::passes::LookAngle],
) {
    use serde_json::{json, Value};

    let hex = |v: f64| -> String { format!("0x{:016x}", v.to_bits()) };
    let hex3 = |v: &[f64; 3]| -> Vec<String> { v.iter().map(|&x| hex(x)).collect() };

    let epochs_json: Vec<Value> = epochs
        .iter()
        .zip(positions.iter())
        .zip(looks.iter())
        .map(|((dt, pos), look)| {
            json!({
                "unix_microseconds": dt.unix_microseconds(),
                "position_km": pos.position,
                "velocity_km_s": pos.velocity,
                "position_km_hex": hex3(&pos.position),
                "velocity_km_s_hex": hex3(&pos.velocity),
                "azimuth_deg": look.azimuth_deg,
                "elevation_deg": look.elevation_deg,
                "range_km": look.range_km,
                "azimuth_deg_hex": hex(look.azimuth_deg),
                "elevation_deg_hex": hex(look.elevation_deg),
                "range_km_hex": hex(look.range_km),
            })
        })
        .collect();

    let doc = json!({
        "source": "iss_arc_walkers_match_single_instant_calls",
        "opsmode": "improved",
        "tle": { "line1": ISS_L1, "line2": ISS_L2 },
        "station": {
            "latitude_deg": STATION.latitude_deg,
            "longitude_deg": STATION.longitude_deg,
            "altitude_m": STATION.altitude_m,
        },
        "epochs": epochs_json,
    });

    let text = serde_json::to_string_pretty(&doc).unwrap() + "\n";
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let out = manifest.join("../../bindings/python/tests/fixtures/sgp4_topocentric.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, &text).expect("dump: write fixture");
    eprintln!("dumped SGP4 topocentric fixture to {out:?}");
}
