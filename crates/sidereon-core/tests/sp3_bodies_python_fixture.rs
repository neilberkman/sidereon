#![cfg(sidereon_repo_tests)]
//! Env-gated emitter that dumps the Area 4 (bodies + SP3) reference numbers as a
//! JSON fixture for the Python binding's pytest (`test_sp3_bodies.py`).
//!
//! Every value is computed with the exact functions the binding calls: Sun/Moon
//! come from `sun_moon_eci_at` / `sun_moon_ecef` driven by the precise
//! `UtcInstant::time_scales` path, and the SP3 numbers come from the same
//! `epochs_j2000_seconds` / `position_at_j2000_seconds` / `to_sp3_string` the
//! binding marshals. The binding loads the SAME committed SP3 fixture verbatim, so
//! the fixture carries the engine's numbers (emitted as IEEE-754 hex bits for an
//! exact cross-check), not invented truth. The dump runs only under
//! `SIDEREON_DUMP_FIXTURES=1`; a normal `cargo test` only self-validates.

use std::path::PathBuf;

use sidereon_core::astro::bodies::{sun_moon_ecef, sun_moon_eci_at};
use sidereon_core::astro::passes::UtcInstant;
use sidereon_core::ephemeris::Sp3;
use sidereon_core::GnssSatelliteId;

/// (year, month, day, hour, minute, second, microsecond) UTC epochs to dump for
/// the Sun/Moon series. Real instants spanning a couple of decades.
const BODY_EPOCHS: &[(i32, i32, i32, i32, i32, i32, i32)] = &[
    (2000, 1, 1, 12, 0, 0, 0),
    (2020, 6, 24, 12, 34, 56, 0),
    (2026, 4, 30, 9, 45, 0, 0),
];

/// SP3 fixture, relative to this crate's manifest dir, loaded verbatim by both the
/// emitter and the Python test (no copy).
const SP3_FIXTURE: &str = "tests/fixtures/sp3/IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3";

/// Satellites queried for the interpolation cross-check (present in the fixture).
const QUERY_SATS: &[&str] = &["G01", "G05", "G15"];

fn hex(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn hex3(values: [f64; 3]) -> Vec<String> {
    values.iter().map(|&v| hex(v)).collect()
}

fn sp3_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(SP3_FIXTURE)
}

fn load_sp3_fixture() -> Sp3 {
    let bytes = std::fs::read(sp3_path()).expect("read SP3 fixture");
    Sp3::parse(&bytes).expect("parse SP3 fixture")
}

/// Interior query times for the interpolation check: midpoints offset into the
/// node axis, guaranteed inside coverage for an 11-node product.
fn query_times(axis: &[f64]) -> Vec<f64> {
    [
        axis[2] + 450.0,
        axis[4] + 225.0,
        axis[6] + 450.0,
        axis[8] + 450.0,
    ]
    .to_vec()
}

#[test]
fn sp3_bodies_reference_self_validates() {
    // Bodies: the time-tagged ECI series and the ECEF rotation produce finite,
    // physically-scaled vectors on the committed sample.
    let ts = UtcInstant::from_unix_microseconds(
        UtcInstant::from_utc(2026, 4, 30, 9, 45, 0, 0)
            .unwrap()
            .unix_microseconds(),
    )
    .time_scales();
    let eci = sun_moon_eci_at(&ts).expect("valid time scales");
    let ecef = sun_moon_ecef(&ts).expect("valid time scales");
    let sun_norm = (eci.sun[0].powi(2) + eci.sun[1].powi(2) + eci.sun[2].powi(2)).sqrt();
    assert!(
        (1.0e11..2.5e11).contains(&sun_norm),
        "sun distance out of range: {sun_norm} m"
    );
    // The rotation preserves the vector magnitude (it is a pure rotation).
    let ecef_sun_norm = (ecef.sun[0].powi(2) + ecef.sun[1].powi(2) + ecef.sun[2].powi(2)).sqrt();
    assert!(
        (sun_norm - ecef_sun_norm).abs() < 1.0,
        "rotation changed |sun|"
    );

    // SP3: the node axis is ascending and interior interpolation succeeds.
    let sp3 = load_sp3_fixture();
    let axis = sp3.epochs_j2000_seconds();
    assert_eq!(axis.len(), sp3.epoch_count());
    assert!(axis.windows(2).all(|w| w[1] > w[0]));
    let g01 = "G01".parse::<GnssSatelliteId>().unwrap();
    let q = query_times(&axis);
    let state = sp3
        .position_at_j2000_seconds(g01, q[0])
        .expect("interior interpolation succeeds");
    assert!(state.position.x_m.is_finite());

    // Write round-trips structurally: re-parsing the serialized text yields the
    // same epoch count and satellite list.
    let text = sp3.to_sp3_string();
    let reparsed = Sp3::parse(text.as_bytes()).expect("re-parse written SP3");
    assert_eq!(reparsed.epoch_count(), sp3.epoch_count());
    assert_eq!(reparsed.satellites(), sp3.satellites());

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture();
    }
}

fn body_epoch_json(epoch: (i32, i32, i32, i32, i32, i32, i32)) -> serde_json::Value {
    use serde_json::json;

    let (y, mo, d, h, mi, s, us) = epoch;
    let unix_micros = UtcInstant::from_utc(y, mo, d, h, mi, s, us)
        .expect("dump: valid calendar epoch")
        .unix_microseconds();
    let ts = UtcInstant::from_unix_microseconds(unix_micros).time_scales();
    let eci = sun_moon_eci_at(&ts).expect("valid time scales");
    let ecef = sun_moon_ecef(&ts).expect("valid time scales");

    json!({
        "unix_micros": unix_micros,
        "sun_eci_m_hex": hex3(eci.sun),
        "moon_eci_m_hex": hex3(eci.moon),
        "sun_ecef_m_hex": hex3(ecef.sun),
        "moon_ecef_m_hex": hex3(ecef.moon),
    })
}

fn dump_fixture() {
    use serde_json::json;

    let bodies: Vec<_> = BODY_EPOCHS.iter().map(|&e| body_epoch_json(e)).collect();

    let sp3 = load_sp3_fixture();
    let axis = sp3.epochs_j2000_seconds();
    let queries = query_times(&axis);

    // Per-satellite interpolation: query times shared, expected position + clock
    // straight from the engine recipe.
    let interpolation: Vec<_> = QUERY_SATS
        .iter()
        .map(|&token| {
            let sat = token
                .parse::<GnssSatelliteId>()
                .expect("dump: valid sat token");
            let states: Vec<_> = queries
                .iter()
                .map(|&q| {
                    let st = sp3
                        .position_at_j2000_seconds(sat, q)
                        .unwrap_or_else(|e| panic!("dump: {token} @ {q}: {e}"));
                    json!({
                        "position_m_hex": hex3(st.position.as_array()),
                        "clock_s_hex": st.clock_s.map(hex),
                    })
                })
                .collect();
            json!({ "satellite": token, "states": states })
        })
        .collect();

    // First parsed record of G01 (exact, non-interpolated state accessor).
    let g01 = "G01".parse::<GnssSatelliteId>().unwrap();
    let rec = sp3.state(g01, 0).expect("dump: G01 at epoch 0");

    let doc = json!({
        "source": "sp3_bodies_reference_self_validates",
        "bodies": bodies,
        "sp3_fixture": SP3_FIXTURE,
        "epoch_count": sp3.epoch_count(),
        "satellites": sp3.satellites().iter().map(|s| s.to_string()).collect::<Vec<_>>(),
        "epochs_j2000_seconds_hex": axis.iter().map(|&v| hex(v)).collect::<Vec<_>>(),
        "query_j2000_seconds_hex": queries.iter().map(|&v| hex(v)).collect::<Vec<_>>(),
        "interpolation": interpolation,
        "state_g01_epoch0": {
            "position_m_hex": hex3(rec.position.as_array()),
            "clock_s_hex": rec.clock_s.map(hex),
            "velocity_m_s_hex": rec.velocity.map(|v| hex3(v.as_array())),
            "clock_event": rec.flags.clock_event,
            "clock_predicted": rec.flags.clock_predicted,
            "maneuver": rec.flags.maneuver,
            "orbit_predicted": rec.flags.orbit_predicted,
        },
        "to_sp3_string": sp3.to_sp3_string(),
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/sp3_bodies.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped sp3+bodies fixture to {out:?}");
}
