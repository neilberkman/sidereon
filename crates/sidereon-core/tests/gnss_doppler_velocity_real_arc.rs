#![cfg(sidereon_repo_tests)]
//! Fixed regression gate on one real Pixel-5 GPS-L1 Doppler arc (2021-12-15 US-MTV-1, GSDC); the velocity reference is a central finite-difference of the GSDC carrier-phase truth ECEF track (baked into the fixture truth_velocity_m_s); bars: median 3D velocity error at most 0.50 m/s, p95 at most 2.50 m/s, n at least 1000 eligible epochs; bars are never loosened to pass.

use serde_json::Value;
use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::astro::time::split_julian_date;
use sidereon_core::ephemeris::BroadcastEphemeris;
use sidereon_core::observables::j2000_seconds_from_split;
use sidereon_core::velocity::{
    solve, VelocityObservable, VelocityObservation, VelocitySolveOptions,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const NAV: &str =
    include_str!("fixtures/rtk/doppler_velocity_gsdc_2021_12_15_mtv1_pixel5_l1_gps.nav");
const INPUTS: &str =
    include_str!("fixtures/rtk/doppler_velocity_gsdc_2021_12_15_mtv1_pixel5_l1_inputs.json");

#[derive(Debug, Clone, Copy)]
struct CivilEpoch {
    year: i64,
    month: i64,
    day: i64,
    hour: u8,
    minute: u8,
    second: f64,
}

fn sat_id(token: &str) -> GnssSatelliteId {
    let mut chars = token.chars();
    let system = chars
        .next()
        .and_then(GnssSystem::from_letter)
        .unwrap_or_else(|| panic!("bad satellite token {token:?}"));
    let prn = chars
        .as_str()
        .parse::<u8>()
        .unwrap_or_else(|_| panic!("bad satellite token {token:?}"));
    GnssSatelliteId::new(system, prn).expect("valid satellite id")
}

fn parse_iso_epoch(token: &str) -> CivilEpoch {
    let (date, time) = token
        .split_once('T')
        .unwrap_or_else(|| panic!("bad ISO epoch {token:?}"));
    let mut date_parts = date.split('-');
    let year = date_parts
        .next()
        .unwrap_or_else(|| panic!("bad ISO epoch {token:?}"))
        .parse::<i64>()
        .unwrap_or_else(|_| panic!("bad year in ISO epoch {token:?}"));
    let month = date_parts
        .next()
        .unwrap_or_else(|| panic!("bad ISO epoch {token:?}"))
        .parse::<i64>()
        .unwrap_or_else(|_| panic!("bad month in ISO epoch {token:?}"));
    let day = date_parts
        .next()
        .unwrap_or_else(|| panic!("bad ISO epoch {token:?}"))
        .parse::<i64>()
        .unwrap_or_else(|_| panic!("bad day in ISO epoch {token:?}"));
    assert!(date_parts.next().is_none(), "bad ISO epoch {token:?}");

    let mut time_parts = time.split(':');
    let hour = time_parts
        .next()
        .unwrap_or_else(|| panic!("bad ISO epoch {token:?}"))
        .parse::<u8>()
        .unwrap_or_else(|_| panic!("bad hour in ISO epoch {token:?}"));
    let minute = time_parts
        .next()
        .unwrap_or_else(|| panic!("bad ISO epoch {token:?}"))
        .parse::<u8>()
        .unwrap_or_else(|_| panic!("bad minute in ISO epoch {token:?}"));
    let second = time_parts
        .next()
        .unwrap_or_else(|| panic!("bad ISO epoch {token:?}"))
        .parse::<f64>()
        .unwrap_or_else(|_| panic!("bad second in ISO epoch {token:?}"));
    assert!(time_parts.next().is_none(), "bad ISO epoch {token:?}");

    CivilEpoch {
        year,
        month,
        day,
        hour,
        minute,
        second,
    }
}

fn civil_to_julian_split(epoch: CivilEpoch) -> JulianDateSplit {
    let (jd_whole, fraction) = split_julian_date(
        epoch.year as i32,
        epoch.month as i32,
        epoch.day as i32,
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    );
    JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date")
}

fn j2000_seconds_from_iso(token: &str) -> f64 {
    let split = civil_to_julian_split(parse_iso_epoch(token));
    j2000_seconds_from_split(split.jd_whole, split.fraction).expect("valid split")
}

fn finite_number(value: &Value, label: &str) -> f64 {
    let number = value
        .as_f64()
        .unwrap_or_else(|| panic!("{label}: expected JSON number"));
    assert!(number.is_finite(), "{label}: expected finite JSON number");
    number
}

fn object_vec3(value: &Value, label: &str) -> [f64; 3] {
    [
        finite_number(&value["x"], &format!("{label}.x")),
        finite_number(&value["y"], &format!("{label}.y")),
        finite_number(&value["z"], &format!("{label}.z")),
    ]
}

fn norm3(vector: [f64; 3]) -> f64 {
    (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt()
}

fn percentile(sorted: &[f64], pct: f64) -> f64 {
    assert!(!sorted.is_empty(), "percentile needs at least one value");
    let rank = pct * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[test]
fn pixel5_gsdc_l1_doppler_velocity_arc_stays_within_bars() {
    let store = BroadcastEphemeris::from_nav(NAV).expect("parse GSDC broadcast NAV");
    let doc: Value = serde_json::from_str(INPUTS).expect("parse Doppler velocity inputs");
    let carrier_hz = finite_number(&doc["carrier_hz"], "carrier_hz");
    let epochs = doc["epochs"].as_array().expect("epochs array");
    assert!(
        epochs.len() >= 1000,
        "fixture must include at least 1000 epochs, got {}",
        epochs.len()
    );

    let options = VelocitySolveOptions {
        observable: VelocityObservable::Doppler,
        light_time: true,
        sagnac: true,
    };
    let mut errors = Vec::with_capacity(epochs.len());
    let mut truth_speeds = Vec::with_capacity(epochs.len());
    let mut solved = 0_usize;
    let mut failed = 0_usize;
    let mut first_failure: Option<String> = None;

    for (index, epoch) in epochs.iter().enumerate() {
        let epoch_token = epoch["epoch"].as_str().expect("epoch string");
        let t_rx_j2000_s = j2000_seconds_from_iso(epoch_token);
        let receiver_ecef_m = object_vec3(&epoch["receiver_ecef_m"], "receiver_ecef_m");
        let truth_velocity_m_s = object_vec3(&epoch["truth_velocity_m_s"], "truth_velocity_m_s");
        let doppler_rows = epoch["doppler_hz"].as_array().expect("doppler_hz array");
        let observations: Vec<_> = doppler_rows
            .iter()
            .map(|row| {
                let row = row.as_array().expect("doppler row");
                assert_eq!(row.len(), 2, "doppler row must have satellite and value");
                VelocityObservation {
                    satellite_id: sat_id(row[0].as_str().expect("satellite token")),
                    value: finite_number(&row[1], "doppler_hz"),
                    carrier_hz,
                    sat_clock_drift_s_s: 0.0,
                }
            })
            .collect();

        match solve(
            &store,
            &observations,
            receiver_ecef_m,
            t_rx_j2000_s,
            options,
        ) {
            Ok(solution) => {
                let delta = [
                    solution.velocity_m_s[0] - truth_velocity_m_s[0],
                    solution.velocity_m_s[1] - truth_velocity_m_s[1],
                    solution.velocity_m_s[2] - truth_velocity_m_s[2],
                ];
                errors.push(norm3(delta));
                truth_speeds.push(norm3(truth_velocity_m_s));
                solved += 1;
            }
            Err(error) => {
                failed += 1;
                if first_failure.is_none() {
                    first_failure = Some(format!("epoch {index} {epoch_token}: {error}"));
                }
            }
        }
    }

    assert_eq!(
        failed,
        0,
        "every eligible epoch must solve; first failure: {}",
        first_failure.unwrap_or_else(|| "none".to_string())
    );
    assert!(
        solved >= 1000,
        "solved epoch count {solved} is below required 1000"
    );
    assert_eq!(errors.len(), solved, "one error per solved epoch");

    errors.sort_by(|a, b| a.total_cmp(b));
    truth_speeds.sort_by(|a, b| a.total_cmp(b));
    let median = percentile(&errors, 0.5);
    let p95 = percentile(&errors, 0.95);
    let median_truth_speed = percentile(&truth_speeds, 0.5);

    assert!(
        median_truth_speed > 5.0,
        "median truth speed {median_truth_speed:.6} m/s must exceed 5.0 m/s"
    );
    assert!(
        median <= 0.50,
        "median 3D velocity error {median:.6} m/s exceeds bar 0.50 m/s"
    );
    assert!(
        p95 <= 2.50,
        "p95 3D velocity error {p95:.6} m/s exceeds bar 2.50 m/s"
    );
}
