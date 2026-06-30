#![cfg(sidereon_repo_tests)]
//! Env-gated emitter that dumps the frames + time reference numbers as a JSON
//! fixture for the Python binding's pytest (`test_frames_time.py`).
//!
//! Every value is computed with the exact functions the binding calls: the
//! precise time scales come from `UtcInstant::time_scales`, the transforms are
//! the engine's own `*_compute`, and sidereal time / nutation / precession come
//! from the public frame helpers. The binding marshals these same calls, so the
//! fixture carries the engine's numbers (emitted as IEEE-754 hex bits for an exact
//! cross-check), not invented truth. The dump runs only under
//! `SIDEREON_DUMP_FIXTURES=1`; in a normal `cargo test` this file only
//! self-validates a couple of invariants.

use std::path::PathBuf;

use sidereon_core::astro::frames::nutation::{
    build_skyfield_nutation_matrix, skyfield_iau2000a_radians, skyfield_mean_obliquity_radians,
};
use sidereon_core::astro::frames::precession::compute_skyfield_precession_matrix;
use sidereon_core::astro::frames::transforms::{
    gcrs_to_itrs_compute, geodetic_to_itrs, greenwich_apparent_sidereal_time_radians,
    greenwich_mean_sidereal_time_radians, itrs_to_gcrs_compute, itrs_to_geodetic_compute,
    teme_to_gcrs_compute, TemeStateKm,
};
use sidereon_core::astro::passes::UtcInstant;
use sidereon_core::astro::time::scales::{
    find_leap_seconds, julian_day_number, leap_second_table, ut1_coverage,
};
use sidereon_core::astro::time::{GnssWeekTow, TimeScale, TimeScales};

const SECONDS_PER_DAY: f64 = 86_400.0;

/// (year, month, day, hour, minute, second, microsecond) UTC epochs to dump.
/// Real instants spanning the embedded EOP coverage.
const EPOCHS: &[(i32, i32, i32, i32, i32, i32, i32)] = &[
    (2000, 1, 1, 12, 0, 0, 0),
    (2018, 7, 3, 19, 25, 57, 304128),
    (2020, 6, 24, 12, 34, 56, 0),
    (2023, 11, 15, 6, 0, 0, 500000),
];

/// A real-magnitude sample state used to exercise the transforms (km, km/s).
const SAMPLE_POS_KM: [f64; 3] = [4321.0, -5678.0, 3210.0];
const SAMPLE_VEL_KM_S: [f64; 3] = [-1.234, 5.678, 7.012];
/// A geodetic sample (lat_deg, lon_deg, alt_km) for the geodetic<->ECEF round.
const SAMPLE_GEODETIC: [f64; 3] = [51.4779, -0.0015, 0.046];

fn hex(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn hex3(values: [f64; 3]) -> Vec<String> {
    values.iter().map(|&v| hex(v)).collect()
}

fn hex_mat3(matrix: &[[f64; 3]; 3]) -> Vec<Vec<String>> {
    matrix.iter().map(|row| hex3(*row)).collect()
}

#[test]
fn frames_time_reference_self_validates() {
    // The sidereal-time wrappers and the geodetic round-trip behave sanely on the
    // committed sample, independent of the dump.
    let ts = UtcInstant::from_unix_microseconds(
        UtcInstant::from_utc(2020, 6, 24, 12, 34, 56, 0)
            .unwrap()
            .unix_microseconds(),
    )
    .time_scales();
    let gmst = greenwich_mean_sidereal_time_radians(&ts).expect("valid sidereal time");
    let gast = greenwich_apparent_sidereal_time_radians(&ts).expect("valid sidereal time");
    assert!(gmst.is_finite() && gast.is_finite());

    let (x, y, z) = geodetic_to_itrs(SAMPLE_GEODETIC[0], SAMPLE_GEODETIC[1], SAMPLE_GEODETIC[2])
        .expect("valid geodetic coordinates");
    let (lat, lon, _alt) = itrs_to_geodetic_compute(x, y, z).expect("valid ITRS coordinates");
    assert!((lat - SAMPLE_GEODETIC[0]).abs() < 1e-6, "lat {lat}");
    assert!((lon - SAMPLE_GEODETIC[1]).abs() < 1e-6, "lon {lon}");

    // GnssWeekTow rollover behaviour the binding re-exposes.
    let normalized = GnssWeekTow::new(TimeScale::Gpst, 100, SECONDS_PER_DAY * 8.0)
        .expect("valid week/TOW")
        .normalized()
        .expect("valid normalized week/TOW");
    assert_eq!(normalized.week, 101);

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture();
    }
}

fn epoch_json(epoch: (i32, i32, i32, i32, i32, i32, i32)) -> serde_json::Value {
    use serde_json::json;

    let (y, mo, d, h, mi, s, us) = epoch;
    let unix_micros = UtcInstant::from_utc(y, mo, d, h, mi, s, us)
        .expect("dump: valid calendar epoch")
        .unix_microseconds();
    let ts: TimeScales = UtcInstant::from_unix_microseconds(unix_micros).time_scales();

    let mean_ob = skyfield_mean_obliquity_radians(ts.jd_tdb).expect("valid TDB Julian date");
    let (dpsi, deps) = skyfield_iau2000a_radians(ts.jd_tt).expect("valid TT Julian date");
    let precession = compute_skyfield_precession_matrix(ts.jd_tdb).expect("valid TDB Julian date");
    let nutation = build_skyfield_nutation_matrix(mean_ob, mean_ob + deps, dpsi)
        .expect("valid nutation angles");
    let delta_t = (ts.tt_fraction - ts.ut1_fraction) * SECONDS_PER_DAY;

    // Transforms on the shared sample state, both compat modes where applicable.
    let (gcrs_pos_sky, gcrs_vel_sky) = teme_to_gcrs_compute(
        &TemeStateKm {
            position_km: SAMPLE_POS_KM,
            velocity_km_s: SAMPLE_VEL_KM_S,
        },
        &ts,
        true,
    )
    .expect("valid frame transform");
    let (gcrs_pos_dir, gcrs_vel_dir) = teme_to_gcrs_compute(
        &TemeStateKm {
            position_km: SAMPLE_POS_KM,
            velocity_km_s: SAMPLE_VEL_KM_S,
        },
        &ts,
        false,
    )
    .expect("valid frame transform");
    let itrs_sky = gcrs_to_itrs_compute(
        SAMPLE_POS_KM[0],
        SAMPLE_POS_KM[1],
        SAMPLE_POS_KM[2],
        &ts,
        true,
    )
    .expect("valid frame transform");
    let itrs_dir = gcrs_to_itrs_compute(
        SAMPLE_POS_KM[0],
        SAMPLE_POS_KM[1],
        SAMPLE_POS_KM[2],
        &ts,
        false,
    )
    .expect("valid frame transform");
    let back_gcrs = itrs_to_gcrs_compute(SAMPLE_POS_KM[0], SAMPLE_POS_KM[1], SAMPLE_POS_KM[2], &ts)
        .expect("valid frame transform");

    json!({
        "calendar": { "year": y, "month": mo, "day": d, "hour": h, "minute": mi, "second": s, "microsecond": us },
        "unix_micros": unix_micros,
        "jd_whole_hex": hex(ts.jd_whole),
        "tt_jd_hex": hex(ts.jd_tt),
        "ut1_jd_hex": hex(ts.jd_ut1),
        "tdb_jd_hex": hex(ts.jd_tdb),
        "tt_fraction_hex": hex(ts.tt_fraction),
        "ut1_fraction_hex": hex(ts.ut1_fraction),
        "tdb_fraction_hex": hex(ts.tdb_fraction),
        "delta_t_seconds_hex": hex(delta_t),
        "gmst_radians_hex": hex(greenwich_mean_sidereal_time_radians(&ts).expect("valid sidereal time")),
        "gast_radians_hex": hex(greenwich_apparent_sidereal_time_radians(&ts).expect("valid sidereal time")),
        "mean_obliquity_radians_hex": hex(mean_ob),
        "nutation_dpsi_hex": hex(dpsi),
        "nutation_deps_hex": hex(deps),
        "precession_matrix_hex": hex_mat3(&precession),
        "nutation_matrix_hex": hex_mat3(&nutation),
        "teme_to_gcrs_skyfield": { "position_hex": hex3([gcrs_pos_sky.0, gcrs_pos_sky.1, gcrs_pos_sky.2]), "velocity_hex": hex3([gcrs_vel_sky.0, gcrs_vel_sky.1, gcrs_vel_sky.2]) },
        "teme_to_gcrs_direct": { "position_hex": hex3([gcrs_pos_dir.0, gcrs_pos_dir.1, gcrs_pos_dir.2]), "velocity_hex": hex3([gcrs_vel_dir.0, gcrs_vel_dir.1, gcrs_vel_dir.2]) },
        "gcrs_to_itrs_skyfield_hex": hex3([itrs_sky.0, itrs_sky.1, itrs_sky.2]),
        "gcrs_to_itrs_direct_hex": hex3([itrs_dir.0, itrs_dir.1, itrs_dir.2]),
        "itrs_to_gcrs_hex": hex3([back_gcrs.0, back_gcrs.1, back_gcrs.2]),
    })
}

fn dump_fixture() {
    use serde_json::json;

    let epochs: Vec<_> = EPOCHS.iter().map(|&e| epoch_json(e)).collect();

    let (gx, gy, gz) = geodetic_to_itrs(SAMPLE_GEODETIC[0], SAMPLE_GEODETIC[1], SAMPLE_GEODETIC[2])
        .expect("valid geodetic coordinates");
    let (glat, glon, galt) =
        itrs_to_geodetic_compute(SAMPLE_POS_KM[0], SAMPLE_POS_KM[1], SAMPLE_POS_KM[2])
            .expect("valid ITRS coordinates");

    let leap_table = leap_second_table();
    let coverage = ut1_coverage();

    // Leap-second cases computed through the same engine path the binding uses
    // (julian_day_number at UTC midnight -> find_leap_seconds), never hand-set.
    let leap_dates = [(1999, 1, 1), (2017, 6, 1), (2024, 1, 1)];
    let leap_cases: Vec<_> = leap_dates
        .iter()
        .map(|&(y, m, d)| {
            let value = find_leap_seconds(julian_day_number(y, m, d) as f64 - 0.5);
            json!({ "year": y, "month": m, "day": d, "value_hex": hex(value) })
        })
        .collect();

    // A GnssWeekTow rollover example (exact integer/float arithmetic).
    let week_tow =
        GnssWeekTow::new(TimeScale::Gpst, 100, SECONDS_PER_DAY * 8.0).expect("valid week/TOW");
    let normalized = week_tow.normalized().expect("valid normalized week/TOW");

    let doc = json!({
        "source": "frames_time_reference_self_validates",
        "sample": {
            "position_km_hex": hex3(SAMPLE_POS_KM),
            "velocity_km_s_hex": hex3(SAMPLE_VEL_KM_S),
            "geodetic_lat_lon_alt_hex": hex3(SAMPLE_GEODETIC),
        },
        "epochs": epochs,
        "geodetic_to_ecef": { "input_hex": hex3(SAMPLE_GEODETIC), "ecef_km_hex": hex3([gx, gy, gz]) },
        "ecef_to_geodetic": { "input_km_hex": hex3(SAMPLE_POS_KM), "geodetic_hex": hex3([glat, glon, galt]) },
        "leap_seconds_cases": leap_cases,
        "leap_second_table": {
            "source": leap_table.source,
            "first_mjd": leap_table.first_mjd,
            "last_mjd": leap_table.last_mjd,
            "entries": leap_table.entries,
        },
        "ut1_coverage": {
            "source": coverage.source,
            "first_mjd": coverage.first_mjd,
            "last_mjd": coverage.last_mjd,
            "first_jd_tt_hex": hex(coverage.first_jd_tt),
            "last_jd_tt_hex": hex(coverage.last_jd_tt),
            "entries": coverage.entries,
        },
        "gnss_week_tow": {
            "system": week_tow.system.abbrev(),
            "input_week": week_tow.week,
            "input_tow_s_hex": hex(week_tow.tow_s),
            "normalized_week": normalized.week,
            "normalized_tow_s_hex": hex(normalized.tow_s),
            "unrolled_week_2_rollovers": week_tow
                .unrolled_week(2)
                .expect("valid unrolled week"),
        },
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/frames_time.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped frames+time fixture to {out:?}");
}
