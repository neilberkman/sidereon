#![cfg(sidereon_repo_tests)]
//! Env-gated emitter for eclipse, angle, and DOP Python binding fixtures.
//!
//! The JSON fixture consumed by `test_events_bodies_dop.py` is produced entirely
//! from the public `sidereon-core` kernels that the binding calls. Floating-point
//! outputs are serialized as IEEE-754 hex bits for exact wrapper cross-checks.

use std::path::PathBuf;

use sidereon_core::astro::angles::{
    earth_angular_radius, moon_angle, phase_angle, sun_angle, sun_elevation,
};
use sidereon_core::astro::events::eclipse::{shadow_fraction, status, EclipseStatus};
use sidereon_core::geometry::{dop, LineOfSight, Wgs84Geodetic};

const SUN_AU_KM: [f64; 3] = [149_597_870.7, 0.0, 0.0];

fn hex(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn hex3(values: [f64; 3]) -> Vec<String> {
    values.iter().map(|&v| hex(v)).collect()
}

fn status_name(value: EclipseStatus) -> &'static str {
    match value {
        EclipseStatus::Sunlit => "SUNLIT",
        EclipseStatus::Penumbra => "PENUMBRA",
        EclipseStatus::Umbra => "UMBRA",
    }
}

#[test]
fn events_bodies_dop_reference_self_validates() {
    let shadow =
        shadow_fraction([-7000.0, 6370.0, 0.0], SUN_AU_KM).expect("valid eclipse geometry");
    assert!(shadow > 0.0 && shadow < 1.0);
    assert_eq!(
        status([-7000.0, 0.0, 0.0], SUN_AU_KM).expect("valid eclipse geometry"),
        EclipseStatus::Umbra
    );

    let sat = [6778.0, 123.0, -456.0];
    let sun = [149_597_870.0, 1_000_000.0, -500_000.0];
    assert!(sun_angle(sat, sun)
        .expect("valid angle geometry")
        .is_finite());
    assert!(sun_elevation(sat, sun)
        .expect("valid angle geometry")
        .is_finite());

    let los = dop_los();
    let weights = [1.0, 1.0, 1.0, 1.0];
    let receiver = Wgs84Geodetic::new(std::f64::consts::FRAC_PI_4, 0.17453292519943295, 0.0)
        .expect("valid WGS84 geodetic position");
    let result = dop(&los, &weights, receiver).expect("DOP fixture geometry");
    assert!(result.gdop.is_finite());
    assert!(result.pdop.is_finite());

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture();
    }
}

fn dop_los() -> [LineOfSight; 4] {
    [
        LineOfSight::new(0.0, 0.34202014332566877, 0.9396926207859084),
        LineOfSight::new(0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(-0.5, -0.25, 0.8290375725550417),
        LineOfSight::new(0.8137976813493737, 0.46984631039295416, 0.3420201433256687),
    ]
}

fn dump_fixture() {
    use serde_json::json;

    let eclipse_cases = [
        ([7000.0, 0.0, 0.0], SUN_AU_KM),
        ([-7000.0, 0.0, 0.0], SUN_AU_KM),
        ([-7000.0, 6370.0, 0.0], SUN_AU_KM),
        ([-7000.0, 6410.0, 0.0], SUN_AU_KM),
    ];
    let eclipse: Vec<_> = eclipse_cases
        .iter()
        .map(|&(sat, sun)| {
            json!({
                "satellite_position_km_hex": hex3(sat),
                "sun_position_km_hex": hex3(sun),
                "shadow_fraction_hex": hex(shadow_fraction(sat, sun).expect("valid eclipse geometry")),
                "status": status_name(status(sat, sun).expect("valid eclipse geometry")),
            })
        })
        .collect();

    let angle_cases = [
        (
            [6778.0, 0.0, 0.0],
            [149_597_870.0, 0.0, 0.0],
            [200_000.0, 300_000.0, 50_000.0],
            [0.0, 6378.0, 0.0],
        ),
        (
            [6778.0, 123.0, -456.0],
            [149_597_870.0, 1_000_000.0, -500_000.0],
            [-384_400.0, 12_345.0, 6_789.0],
            [-6378.0, 100.0, 50.0],
        ),
    ];
    let angles: Vec<_> = angle_cases
        .iter()
        .map(|&(sat, sun, moon, observer)| {
            json!({
                "satellite_position_km_hex": hex3(sat),
                "sun_position_km_hex": hex3(sun),
                "moon_position_km_hex": hex3(moon),
                "observer_position_km_hex": hex3(observer),
                "sun_angle_deg_hex": hex(sun_angle(sat, sun).expect("valid angle geometry")),
                "moon_angle_deg_hex": hex(moon_angle(sat, moon).expect("valid angle geometry")),
                "sun_elevation_deg_hex": hex(sun_elevation(sat, sun).expect("valid angle geometry")),
                "phase_angle_deg_hex": hex(phase_angle(sat, sun, observer).expect("valid angle geometry")),
                "earth_angular_radius_deg_hex": hex(earth_angular_radius(sat).expect("valid angle geometry")),
            })
        })
        .collect();

    let receiver = Wgs84Geodetic::new(std::f64::consts::FRAC_PI_4, 0.17453292519943295, 12.0)
        .expect("valid WGS84 geodetic position");
    let los = dop_los();
    let weights = [1.0, 1.0, 1.0, 1.0];
    let result = dop(&los, &weights, receiver).expect("DOP fixture geometry");

    let doc = json!({
        "source": "events_bodies_dop_reference_self_validates",
        "eclipse": eclipse,
        "angles": angles,
        "dop": {
            "line_of_sight_hex": los.iter().map(|r| hex3([r.e_x, r.e_y, r.e_z])).collect::<Vec<_>>(),
            "weights_hex": weights.iter().map(|&w| hex(w)).collect::<Vec<_>>(),
            "receiver": {
                "lat_rad_hex": hex(receiver.lat_rad),
                "lon_rad_hex": hex(receiver.lon_rad),
                "height_m_hex": hex(receiver.height_m),
            },
            "gdop_hex": hex(result.gdop),
            "pdop_hex": hex(result.pdop),
            "hdop_hex": hex(result.hdop),
            "vdop_hex": hex(result.vdop),
            "tdop_hex": hex(result.tdop),
        },
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/events_bodies_dop.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped events+bodies+DOP fixture to {out:?}");
}
