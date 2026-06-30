#![cfg(sidereon_repo_tests)]
//! Physical-truth gate for broadcast-vs-precise comparison (the SISRE orbit/clock
//! accuracy check), over the real 2020 DOY177 IGS data committed crate-side:
//! ESBC00DNK mixed broadcast navigation differenced against the COD MGEX final
//! precise SP3, GPS, the full UTC day at a 15 min step.
//!
//! The per-epoch evaluation keys (broadcast J2000 seconds + SP3 split Julian dates
//! for the epoch and its `+/-` velocity neighbours) are the exact terms the Sidereon
//! interface marshals, captured to `broadcast_comparison_golden.json`. Feeding them
//! to the core comparison must reproduce the expected GPS broadcast accuracy: an
//! overall 3D orbit RMS of roughly 1-2 m, dominated by along-track and radial. The
//! lower bound is non-tautological (a zeroed/broken eval collapses to ~0); the
//! upper bound flags a parse/eval/coverage regression. RAC orthonormality and the
//! clock-datum shrink are checked as structural invariants of the difference
//! algebra. The bit-exact operation-order pins for the RAC projection, the
//! finite-difference velocity, and the RMS/median/datum aggregation live as unit
//! tests in the module itself.

use serde_json::Value;
use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::broadcast_comparison::{
    compare, compare_window, compare_window_epochs, CompareReport, CompareWindow, EpochInputs,
};
use sidereon_core::ephemeris::{BroadcastEphemeris, Sp3};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::path::PathBuf;

const GOLDEN: &str = include_str!("fixtures/broadcast_comparison_golden.json");

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn broadcast() -> BroadcastEphemeris {
    let text = std::fs::read_to_string(fixture_path("nav/ESBC00DNK_R_20201770000_01D_MN.rnx"))
        .expect("read ESBC broadcast NAV fixture");
    BroadcastEphemeris::from_nav(&text).expect("parse ESBC broadcast NAV")
}

fn precise() -> Sp3 {
    let bytes = std::fs::read(fixture_path("sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3"))
        .expect("read COD precise SP3 fixture");
    Sp3::parse(&bytes).expect("parse COD precise SP3")
}

fn hex_bits(value: &Value) -> f64 {
    let raw = value.as_str().expect("hex bits string");
    let hex = raw.strip_prefix("0x").unwrap_or(raw);
    f64::from_bits(u64::from_str_radix(hex, 16).expect("hex bits"))
}

fn parse_token(token: &str) -> GnssSatelliteId {
    let mut chars = token.chars();
    let system =
        GnssSystem::from_letter(chars.next().expect("system letter")).expect("known system letter");
    let prn: u8 = chars.as_str().parse().expect("satellite PRN");
    GnssSatelliteId::new(system, prn).expect("valid satellite id")
}

fn run() -> CompareReport {
    let doc: Value = serde_json::from_str(GOLDEN).expect("parse golden");
    let inputs = &doc["inputs"];

    let satellites: Vec<GnssSatelliteId> = inputs["satellites"]
        .as_array()
        .expect("satellites")
        .iter()
        .map(|token| parse_token(token.as_str().expect("satellite token")))
        .collect();

    let velocity_half_s = inputs["velocity_half_s"].as_f64().expect("velocity_half_s");

    let epochs: Vec<EpochInputs> = inputs["epochs"]
        .as_array()
        .expect("epochs")
        .iter()
        .map(|row| {
            let row = row.as_array().expect("epoch row");
            EpochInputs {
                broadcast_t_j2000_s: hex_bits(&row[0]),
                precise: JulianDateSplit::new(hex_bits(&row[1]), hex_bits(&row[2]))
                    .expect("valid split Julian date"),
                precise_plus: JulianDateSplit::new(hex_bits(&row[3]), hex_bits(&row[4]))
                    .expect("valid split Julian date"),
                precise_minus: JulianDateSplit::new(hex_bits(&row[5]), hex_bits(&row[6]))
                    .expect("valid split Julian date"),
            }
        })
        .collect();

    compare(
        &broadcast(),
        &precise(),
        &satellites,
        &epochs,
        velocity_half_s,
    )
    .expect("valid broadcast comparison inputs")
}

#[test]
fn window_driver_matches_precomputed_grid() {
    let doc: Value = serde_json::from_str(GOLDEN).expect("parse golden");
    let inputs = &doc["inputs"];

    let satellites: Vec<GnssSatelliteId> = inputs["satellites"]
        .as_array()
        .expect("satellites")
        .iter()
        .map(|token| parse_token(token.as_str().expect("satellite token")))
        .collect();
    let velocity_half_s = inputs["velocity_half_s"].as_f64().expect("velocity_half_s");

    // Anchor a regular window at the golden's first epoch (so the queries land
    // inside the product coverage) and step across the next several epochs.
    let first = inputs["epochs"].as_array().expect("epochs")[0]
        .as_array()
        .expect("epoch row");
    let t0 = hex_bits(&first[0]);
    let precise_start =
        JulianDateSplit::new(hex_bits(&first[1]), hex_bits(&first[2])).expect("valid split");
    let step_s = 900.0;
    let window = CompareWindow {
        broadcast_window_j2000_s: (t0, t0 + 5.0 * step_s),
        precise_start,
        step_s,
        velocity_half_s,
    };

    let broadcast = broadcast();
    let precise = precise();

    let grid = compare_window_epochs(&window).expect("window grid");
    assert!(grid.len() > 1, "window produced too few epochs");

    let report_window =
        compare_window(&broadcast, &precise, &satellites, &window).expect("window comparison");
    let report_grid = compare(&broadcast, &precise, &satellites, &grid, velocity_half_s)
        .expect("precomputed comparison");

    assert_eq!(
        report_window, report_grid,
        "window driver must match compare fed the equivalent grid"
    );
    assert!(
        report_window.overall.count > 0,
        "window comparison compared no epochs"
    );
}

#[test]
fn broadcast_comparison_invalid_epoch_inputs_are_rejected() {
    let sat = parse_token("G01");
    let bad_epoch = EpochInputs {
        broadcast_t_j2000_s: f64::NAN,
        precise: JulianDateSplit {
            jd_whole: f64::NAN,
            fraction: 0.0,
        },
        precise_plus: JulianDateSplit {
            jd_whole: 2_451_545.0,
            fraction: 2.0,
        },
        precise_minus: JulianDateSplit::new(2_451_545.0, 0.0).expect("valid split Julian date"),
    };

    let err = compare(&broadcast(), &precise(), &[sat], &[bad_epoch], 450.0)
        .expect_err("invalid epochs must not be skipped into empty stats");
    assert!(
        matches!(err, sidereon_core::Error::InvalidInput(_)),
        "expected InvalidInput, got {err:?}"
    );
}

#[test]
fn gps_orbit_agreement_is_broadcast_accuracy_class() {
    let report = run();
    let overall = report.overall;

    let rms = overall.orbit_3d_rms_m.expect("orbit RMS");
    let max = overall.orbit_3d_max_m.expect("orbit max");
    assert!(
        overall.count > 1000,
        "too few compared epochs: {}",
        overall.count
    );
    assert!(rms > 0.3 && rms < 3.0, "GPS orbit RMS out of band: {rms} m");
    assert!(max < 6.0, "GPS orbit max out of band: {max} m");
}

#[test]
fn rac_decomposition_is_orthonormal() {
    let overall = run().overall;
    let rms = overall.orbit_3d_rms_m.expect("orbit RMS");
    let radial = overall.radial_rms_m.expect("radial RMS");
    let along = overall.along_rms_m.expect("along RMS");
    let cross = overall.cross_rms_m.expect("cross RMS");

    assert!(radial > 0.0 && along > 0.0 && cross > 0.0);
    // RAC is an orthonormal rotation of the difference, so the 3D RMS equals the
    // quadrature sum of the component RMS values.
    let quadrature = (radial * radial + along * along + cross * cross).sqrt();
    assert!((rms - quadrature).abs() < 1.0e-6, "RAC quadrature mismatch");
}

#[test]
fn removing_the_clock_datum_shrinks_the_clock_error() {
    let overall = run().overall;
    let raw = overall.clock_rms_m.expect("raw clock RMS");
    let datum_removed = overall
        .clock_datum_removed_rms_m
        .expect("datum-removed clock RMS");

    assert!(
        raw > 0.0 && raw < 50.0,
        "raw clock RMS out of band: {raw} m"
    );
    assert!(datum_removed > 0.0);
    assert!(
        datum_removed < raw,
        "datum removal did not shrink the clock error: {datum_removed} >= {raw}"
    );
}

#[test]
fn per_satellite_stats_and_missing_are_populated() {
    let report = run();
    assert!(report.per_satellite.len() > 20);
    assert!(!report.missing.is_empty());
    // Every reported satellite contributed at least one compared epoch.
    assert!(report
        .per_satellite
        .iter()
        .any(|(_sat, stats)| stats.count > 0));
}
