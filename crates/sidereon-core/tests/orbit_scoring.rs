#![cfg(sidereon_repo_tests)]
//! Orbit scoring against final products (`sidereon_core::orbit_scoring`).
//!
//! - The radial/transverse/normal projection against Orekit 13.1.10
//!   `LOFType.QSW` and `LOFType.TNW` (`fixtures/orbit_scoring/orekit_lof.json`,
//!   written by `fixtures-generators/orekit_lof_oracle`): on identical inertial
//!   states, and end to end on the IGS ultra-rapid predicted records against
//!   the IGS final records of 2026-05-13, where Orekit takes the Earth-fixed
//!   state to GCRF with the full IERS 2010 transformation.
//! - The orbit-only and clock SISRE against Montenbruck, Steigenberger and
//!   Hauschild (2018), evaluated here from the Orekit components, and the
//!   clock datum against the per-epoch median formed here from the SP3
//!   records.
//! - A broadcast source scored on the Earth-fixed frame reproduces every
//!   statistic of `broadcast_comparison::compare` bit for bit.
//! - Prediction origins from SP3 flags and cut-offs, coverage accounting, and
//!   the typed errors.
//! - Trajectory samples at record epochs found where the reference reads its
//!   records, on synthetic products whose epochs reduce off their whole
//!   seconds (2000-01-01) or carry a leap second (2016-12-31 23:59:60 UTC).

use serde_json::Value;
use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::broadcast_comparison::{compare, CompareStats, EpochInputs};
use sidereon_core::constants::{C_M_S, OMEGA_E_DOT_RAD_S};
use sidereon_core::ephemeris::{
    BroadcastEphemeris, Sp3, Sp3FrameLabelSet, Sp3FrameReconciliationOptions, Sp3TimeSystem,
};
use sidereon_core::frame_catalog::TerrestrialFrame;
use sidereon_core::orbit_scoring::{
    rtn_components, score_orbit, BroadcastOrbit, ClockDatum, ErrorStats, FrameAgreement, GapPolicy,
    OrbitFrame, OrbitReferencePoint, OrbitRegime, OrbitScoreError, OrbitScoreOptions,
    OrbitScoreReport, OrbitSource, OrbitSourceState, OrbitTrajectory, PredictedSp3, SatelliteScore,
    ScoreCoverage, ScoreEpochs, ScoredProduct, SisreWeights, TrajectorySample, TriadVelocity,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::path::PathBuf;

const OREKIT_LOF: &str = include_str!("fixtures/orbit_scoring/orekit_lof.json");
const BROADCAST_GOLDEN: &str = include_str!("fixtures/broadcast_comparison_golden.json");

const FINAL_SP3: &str = "sp3/IGS0OPSFIN_20261330000_03H_15M_ORB.SP3";
const ULTRA_RAPID_1800: &str = "sp3/IGS0OPSULT_20261311800_02D_15M_ORB_2245_0415.SP3";
const ULTRA_RAPID_0000: &str = "sp3/IGS0OPSULT_20261320000_02D_15M_ORB_2145_0415.SP3";

/// Julian date of 2026-05-12 00:00 (MJD 61172).
const JD_2026_05_12: f64 = 2_461_172.5;
/// Seconds since J2000 of 2026-05-13 00:00 GPST, the first final record.
const FINAL_START_J2000_S: f64 = 831_902_400.0;
/// Record spacing of all three products, seconds.
const STEP_S: f64 = 900.0;

/// Identical-state agreement with Orekit, as a fraction of `|d|`. Both sides
/// rotate the same double-precision difference by a rotation built from the
/// same position and velocity: Orekit composes a quaternion from the two
/// vectors and applies it, the crate dots the difference with three unit
/// vectors. Each route rounds the rotation to a few units in the last place
/// (2.2e-16), so the components agree to about 1e-15 of `|d|`; the bound
/// leaves a factor of ten for the quaternion's square roots.
const IDENTICAL_STATE_REL: f64 = 1.0e-14;

/// Radial agreement with Orekit on the SP3 records, meters. The radial
/// component is the difference projected on the position direction, which no
/// velocity enters and the ITRF-to-GCRF rotation leaves unchanged. Orekit
/// differences the records themselves; the crate differences its interpolated
/// positions at the record epochs, which reproduce each record to a few units
/// in the last place of a 26 000 km coordinate (3.7e-9 m each).
const SP3_RADIAL_TOL_M: f64 = 5.0e-8;

/// Bound on the rotation between the crate's inertial orbit frame and Orekit's
/// QSW of the GCRF state, radians. The crate adds `omega x r` about the
/// Earth-fixed z-axis at the nominal rate; the Earth's rotation axis is off
/// that axis by the polar motion (under 0.6 arcsecond, 3e-6 rad), which turns
/// the inertial velocity about the radial direction by at most
/// `3e-6 * omega * r / v`, below 1.6e-6 rad for GPS. The two velocities also
/// come from different interpolants of the 15-minute records (Orekit's
/// 11-record Hermite derivative, the crate's centered difference of its own
/// 11-record interpolation), whose slopes differ by millimeters per second, a
/// rotation below 1e-6 rad.
const INERTIAL_FRAME_TILT_RAD: f64 = 3.0e-6;

/// The same bound for the Earth-fixed frame, radians: no Earth rotation model
/// enters, so only the two interpolants' slopes differ.
const EARTH_FIXED_FRAME_TILT_RAD: f64 = 1.0e-6;

/// Clock agreement with the per-epoch median formed from the SP3 records,
/// meters. The crate interpolates the clocks at the record epochs, which
/// reproduces each record to a unit in the last place of a 3e-4 s clock
/// (5.4e-20 s), 1.6e-11 m after scaling by the speed of light.
const CLOCK_TOL_M: f64 = 1.0e-9;

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

fn load_sp3(name: &str) -> Sp3 {
    let bytes = std::fs::read(fixture_path(name)).expect("read SP3 fixture");
    Sp3::parse(&bytes).expect("parse SP3 fixture")
}

fn fixture_text(name: &str) -> String {
    std::fs::read_to_string(fixture_path(name)).expect("read SP3 fixture")
}

fn parse_text(text: &str) -> Sp3 {
    Sp3::parse(text.as_bytes()).expect("parse SP3 text")
}

fn sat(token: &str) -> GnssSatelliteId {
    let mut chars = token.chars();
    let system =
        GnssSystem::from_letter(chars.next().expect("system letter")).expect("known system letter");
    let prn: u8 = chars.as_str().trim().parse().expect("satellite PRN");
    GnssSatelliteId::new(system, prn).expect("valid satellite id")
}

fn vec3(value: &Value) -> [f64; 3] {
    let array = value.as_array().expect("3-vector");
    assert_eq!(array.len(), 3);
    [
        array[0].as_f64().expect("number"),
        array[1].as_f64().expect("number"),
        array[2].as_f64().expect("number"),
    ]
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// A GPST instant `days` whole days and `fraction` of a day after 2026-05-12
/// 00:00.
fn gpst(days: f64, fraction: f64) -> Instant {
    Instant::from_julian_date(
        TimeScale::Gpst,
        JulianDateSplit::new(JD_2026_05_12 + days, fraction).expect("valid split"),
    )
}

/// 2026-05-12 18:00 GPST: the end of the fitted half of the ultra-rapid issued
/// 2026-05-11 18:00 (the trimmed copy holds only predicted records).
fn ultra_rapid_1800_cutoff() -> Instant {
    gpst(0.0, 0.75)
}

fn orekit() -> Value {
    serde_json::from_str(OREKIT_LOF).expect("parse Orekit fixture")
}

/// Every pair asked for is in exactly one coverage count.
fn assert_coverage_closes(coverage: &ScoreCoverage) {
    assert_eq!(
        coverage.epochs,
        coverage.scored
            + coverage.outside_horizon_bins
            + coverage.no_prediction_origin
            + coverage.source_satellite_missing
            + coverage.source_epoch_missing
            + coverage.reference_satellite_missing
            + coverage.reference_epoch_missing,
        "coverage does not account for every pair: {coverage:?}"
    );
    assert!(coverage.clocked <= coverage.scored);
}

fn assert_report_coverage_closes(report: &OrbitScoreReport) {
    assert_coverage_closes(&report.coverage);
    for score in &report.per_satellite {
        assert_coverage_closes(&score.coverage);
    }
    for score in &report.per_system {
        assert_coverage_closes(&score.coverage);
    }
}

fn satellite_score(report: &OrbitScoreReport, satellite: GnssSatelliteId) -> &SatelliteScore {
    report
        .per_satellite
        .iter()
        .find(|score| score.satellite == satellite)
        .expect("satellite in report")
}

/// Bin edges that put each of the final product's 13 records in a bin of its
/// own, for a source whose prediction starts at `origin_j2000_s`.
fn one_record_per_bin_edges(origin_j2000_s: f64) -> Vec<f64> {
    let first = FINAL_START_J2000_S - origin_j2000_s;
    let mut edges = vec![first - 1.0];
    edges.extend((0..13).map(|k| first + STEP_S * k as f64));
    edges
}

fn per_record_report(
    source: &PredictedSp3<'_>,
    reference: &Sp3,
    origin_j2000_s: f64,
    triad: TriadVelocity,
) -> OrbitScoreReport {
    let mut options = OrbitScoreOptions::new(one_record_per_bin_edges(origin_j2000_s));
    options.triad = triad;
    let report = score_orbit(source, reference, reference.satellites(), &options)
        .expect("score the predicted records");
    assert_report_coverage_closes(&report);
    report
}

/// The two ultra-rapid sources with their prediction origins, seconds since
/// J2000: the 18:00 issue from its cut-off, the 00:00 issue from its flags.
fn with_ultra_rapid_sources(mut check: impl FnMut(&str, &PredictedSp3<'_>, f64)) {
    let early = load_sp3(ULTRA_RAPID_1800);
    let early_source =
        PredictedSp3::with_cutoff(&early, ultra_rapid_1800_cutoff()).expect("cut-off");
    check(
        "IGS0OPSULT_20261311800_02D_15M_ORB_2245_0415.SP3",
        &early_source,
        FINAL_START_J2000_S - 6.0 * 3600.0,
    );

    let late = load_sp3(ULTRA_RAPID_0000);
    let late_source = PredictedSp3::from_prediction_flags(&late).expect("flags");
    check(
        "IGS0OPSULT_20261320000_02D_15M_ORB_2145_0415.SP3",
        &late_source,
        FINAL_START_J2000_S - STEP_S,
    );
}

fn orekit_states<'a>(doc: &'a Value, file: &str) -> &'a Vec<Value> {
    doc["sp3"]["sources"]
        .as_array()
        .expect("sources")
        .iter()
        .find(|source| source["file"] == file)
        .expect("source in fixture")["states"]
        .as_array()
        .expect("states")
}

#[test]
fn fixture_records_the_pinned_oracle() {
    let doc = orekit();
    let source = &doc["_source"];
    assert_eq!(source["orekit"], "13.1.10");
    assert_eq!(source["hipparchus"], "4.0.3");
    // The orekit-data commit supplies the Earth orientation the GCRF case uses.
    let commit = source["orekit_data_commit"]
        .as_str()
        .expect("orekit-data commit");
    assert!(commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(
        doc["sp3"]["reference"],
        "IGS0OPSFIN_20261330000_03H_15M_ORB.SP3"
    );
}

/// `(name, r, v, d, Orekit QSW components)` of one identical-state case.
type IdenticalStateCase = (String, [f64; 3], [f64; 3], [f64; 3], [f64; 3]);

// The crate's frame is Orekit's QSW, component for component, on the same
// inertial state: synthetic Keplerian states and the GCRF states Orekit formed
// from the SP3 records.
#[test]
fn rtn_is_orekit_qsw_on_identical_states() {
    let doc = orekit();
    let mut cases: Vec<IdenticalStateCase> = Vec::new();
    for case in doc["synthetic"].as_array().expect("synthetic") {
        cases.push((
            case["name"].as_str().expect("name").to_string(),
            vec3(&case["r_m"]),
            vec3(&case["v_m_s"]),
            vec3(&case["d_m"]),
            vec3(&case["qsw_m"]),
        ));
    }
    for source in doc["sp3"]["sources"].as_array().expect("sources") {
        for state in source["states"].as_array().expect("states") {
            cases.push((
                format!("{} {}", state["satellite"], state["epoch"]),
                vec3(&state["r_gcrf_m"]),
                vec3(&state["v_gcrf_m_s"]),
                vec3(&state["d_gcrf_m"]),
                vec3(&state["qsw_gcrf_m"]),
            ));
        }
    }
    assert_eq!(cases.len(), 27 + 2 * 96);

    for (name, r, v, d, qsw) in cases {
        let rtn = rtn_components(d, r, v).expect("defined frame");
        let tol = IDENTICAL_STATE_REL * norm(d);
        for (component, crate_value, orekit_value) in [
            ("R", rtn.radial_m, qsw[0]),
            ("T", rtn.transverse_m, qsw[1]),
            ("N", rtn.normal_m, qsw[2]),
        ] {
            assert!(
                (crate_value - orekit_value).abs() <= tol,
                "{name} {component}: crate {crate_value:e}, Orekit QSW {orekit_value:e}"
            );
        }
    }
}

// Orekit's TNW is not the crate's frame: its first axis is the velocity. It
// equals the crate's components turned about the normal by the flight-path
// angle, `T_TNW = (v_R R + v_T T) / |v|`, `N_TNW = (v_R T - v_T R) / |v|`,
// `W = N`, with `(v_R, v_T)` the velocity resolved on the crate's frame.
#[test]
fn orekit_tnw_is_rtn_turned_by_the_flight_path_angle() {
    let doc = orekit();
    let mut largest_turn = 0.0_f64;
    for case in doc["synthetic"].as_array().expect("synthetic") {
        let name = case["name"].as_str().expect("name");
        let (r, v, d) = (vec3(&case["r_m"]), vec3(&case["v_m_s"]), vec3(&case["d_m"]));
        let tnw = vec3(&case["tnw_m"]);
        let rtn = rtn_components(d, r, v).expect("defined frame");
        let velocity = rtn_components(v, r, v).expect("defined frame");
        let speed = libm::hypot(velocity.radial_m, velocity.transverse_m);
        let t =
            (velocity.radial_m * rtn.radial_m + velocity.transverse_m * rtn.transverse_m) / speed;
        let n =
            (velocity.radial_m * rtn.transverse_m - velocity.transverse_m * rtn.radial_m) / speed;
        let tol = IDENTICAL_STATE_REL * norm(d);
        for (component, crate_value, orekit_value) in [
            ("T", t, tnw[0]),
            ("N", n, tnw[1]),
            ("W", rtn.normal_m, tnw[2]),
        ] {
            assert!(
                (crate_value - orekit_value).abs() <= tol,
                "{name} {component}: from RTN {crate_value:e}, Orekit TNW {orekit_value:e}"
            );
        }
        largest_turn = largest_turn.max((tnw[0] - rtn.transverse_m).abs() / norm(d));
    }
    // The eccentric cases have flight-path angles of tens of degrees, so the
    // check above is not satisfied by TNW and QSW coinciding.
    assert!(largest_turn > 0.3, "largest TNW/QSW turn {largest_turn}");
}

/// Check the per-record scores of one source against the Orekit fixture states:
/// each state's satellite has one sample in the bin of its record.
fn compare_records(
    file: &str,
    report: &OrbitScoreReport,
    states: &[Value],
    orekit_key: &str,
    tilt_rad: f64,
) {
    let weights =
        SisreWeights::montenbruck_2018(GnssSystem::Gps, OrbitRegime::Meo).expect("GPS weights");
    assert_eq!(states.len(), 96, "{file}");
    for state in states {
        let satellite = sat(state["satellite"].as_str().expect("satellite"));
        let record = state["reference_record"].as_u64().expect("record") as usize;
        let qsw = vec3(&state[orekit_key]);
        let d = norm(vec3(&state["d_itrf_m"]));
        let score = satellite_score(report, satellite);
        let bin = &score.bins[record];
        assert_eq!(bin.count, 1, "{file} {satellite} record {record}");
        let radial = bin.radial.expect("radial").mean_m;
        let transverse = bin.transverse.expect("transverse").mean_m;
        let normal = bin.normal.expect("normal").mean_m;
        let label = format!("{file} {satellite} record {record} {orekit_key}");

        assert!(
            (radial - qsw[0]).abs() <= SP3_RADIAL_TOL_M,
            "{label} R: crate {radial:e}, Orekit {:e}",
            qsw[0]
        );
        let tol = tilt_rad * d + SP3_RADIAL_TOL_M;
        assert!(
            (transverse - qsw[1]).abs() <= tol,
            "{label} T: crate {transverse:e}, Orekit {:e}",
            qsw[1]
        );
        assert!(
            (normal - qsw[2]).abs() <= tol,
            "{label} N: crate {normal:e}, Orekit {:e}",
            qsw[2]
        );
        let orbit_3d = bin.orbit_3d.expect("3D").mean_m;
        assert!((orbit_3d - d).abs() <= SP3_RADIAL_TOL_M, "{label} 3D");

        // The orbit-only SISRE from Orekit's components. `T^2 + N^2 = |d|^2 - R^2` is unchanged
        // by a turn of the frame about the radial axis, so the frame tilt does not
        // enter and the radial tolerance applies.
        let expected_sisre = ((weights.radial() * qsw[0]).powi(2)
            + weights.transverse_normal_squared() * (qsw[1] * qsw[1] + qsw[2] * qsw[2]))
            .sqrt();
        let sisre = bin.sisre_orbit.expect("orbit SISRE").mean_m;
        assert!(
            (sisre - expected_sisre).abs() <= SP3_RADIAL_TOL_M,
            "{label} SISRE(orb): crate {sisre:e}, from Orekit {expected_sisre:e}"
        );
    }
}

// End to end: the scores of the predicted records equal Orekit's QSW of the
// GCRF state, which Orekit forms with the full IERS 2010 Earth rotation.
#[test]
fn inertial_frame_matches_orekit_qsw_through_the_itrf_to_gcrf_transformation() {
    let reference = load_sp3(FINAL_SP3);
    let doc = orekit();
    with_ultra_rapid_sources(|file, source, origin| {
        let report = per_record_report(source, &reference, origin, TriadVelocity::Inertial);
        compare_records(
            file,
            &report,
            orekit_states(&doc, file),
            "qsw_gcrf_m",
            INERTIAL_FRAME_TILT_RAD,
        );
    });
}

// The Earth-fixed frame is Orekit's QSW of the Earth-fixed state taken as it
// stands, the frame `broadcast_comparison` reports in.
#[test]
fn earth_fixed_frame_matches_orekit_qsw_of_the_earth_fixed_state() {
    let reference = load_sp3(FINAL_SP3);
    let doc = orekit();
    with_ultra_rapid_sources(|file, source, origin| {
        let report = per_record_report(source, &reference, origin, TriadVelocity::EarthFixed);
        compare_records(
            file,
            &report,
            orekit_states(&doc, file),
            "qsw_itrf_m",
            EARTH_FIXED_FRAME_TILT_RAD,
        );
    });
}

/// The median of a non-empty list, the two middle values averaged for an even
/// count.
fn median_of(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        values[mid]
    } else {
        (values[mid - 1] + values[mid]) / 2.0
    }
}

// The clock error is the clock difference less the median over the
// constellation's satellites at the epoch, formed here from the SP3 records,
// and the clock SISRE is that of Montenbruck et al. (2018).
#[test]
fn clock_error_removes_the_per_epoch_constellation_median() {
    let reference = load_sp3(FINAL_SP3);
    let weights =
        SisreWeights::montenbruck_2018(GnssSystem::Gps, OrbitRegime::Meo).expect("GPS weights");
    with_ultra_rapid_sources(|file, source, origin| {
        let report = per_record_report(source, &reference, origin, TriadVelocity::Inertial);
        let product = source.product();
        let source_seconds = product.epochs_j2000_seconds();
        let mut checked = 0usize;
        for (record, &t) in reference.epochs_j2000_seconds().iter().enumerate() {
            let source_index = source_seconds
                .iter()
                .position(|&s| s == t)
                .expect("record epoch in source");
            let reference_states = reference.states_at(record).expect("record");
            let source_states = product.states_at(source_index).expect("record");
            let differences: Vec<(GnssSatelliteId, f64)> = reference_states
                .iter()
                .filter_map(|(satellite, state)| {
                    let source_clock = source_states.get(satellite)?.clock_s?;
                    Some((*satellite, (source_clock - state.clock_s?) * C_M_S))
                })
                .collect();
            assert!(differences.len() >= 2, "{file} record {record}");
            let datum = median_of(differences.iter().map(|(_, d)| *d).collect());
            for (satellite, difference) in differences {
                let bin = &satellite_score(&report, satellite).bins[record];
                assert_eq!(bin.clock_count, 1, "{file} {satellite} record {record}");
                let clock = bin.clock.expect("clock").mean_m;
                assert!(
                    (clock - (difference - datum)).abs() <= CLOCK_TOL_M,
                    "{file} {satellite} record {record}: crate {clock:e}, records {:e}",
                    difference - datum
                );
                let r = bin.radial.expect("radial").mean_m;
                let t = bin.transverse.expect("transverse").mean_m;
                let n = bin.normal.expect("normal").mean_m;
                let expected = ((weights.radial() * r - clock).powi(2)
                    + weights.transverse_normal_squared() * (t * t + n * n))
                    .sqrt();
                let sisre = bin.sisre.expect("SISRE").mean_m;
                assert!(
                    (sisre - expected).abs() <= 1.0e-12,
                    "{file} {satellite} record {record} SISRE"
                );
                checked += 1;
            }
        }
        assert!(checked > 300, "{file}: {checked} clocks checked");
        assert_eq!(report.coverage.clocked, checked, "{file}");
    });
}

// The ultra-rapid issued 2026-05-12 00:00 flags its records from 00:00 on
// 2026-05-13 as predicted, so each satellite's prediction starts at its last
// fitted record, 23:45.
#[test]
fn ultra_rapid_flags_give_the_prediction_origin() {
    let reference = load_sp3(FINAL_SP3);
    let product = load_sp3(ULTRA_RAPID_0000);
    let source = PredictedSp3::from_prediction_flags(&product).expect("flags");
    let origin = product.epochs_j2000_seconds()[8];
    assert_eq!(origin, FINAL_START_J2000_S - STEP_S);

    let options = OrbitScoreOptions::new(vec![-3600.0, 0.0, 3600.0, 7200.0, 10_800.0]);
    let report = score_orbit(&source, &reference, reference.satellites(), &options).expect("score");
    assert_report_coverage_closes(&report);
    assert_eq!(report.frame, FrameAgreement::SameLabel);
    assert_eq!(
        report.source_reference_point,
        OrbitReferencePoint::CenterOfMass
    );

    let satellites = reference.satellites().len();
    assert_eq!(satellites, 32);
    for score in &report.per_satellite {
        assert_eq!(score.prediction_origin_j2000_s, Some(origin));
        // The final records start at 00:00, a horizon of 900 s; the last, at
        // 03:00, is 11 700 s out and outside the bins.
        assert_eq!(score.coverage.epochs, 13);
        assert_eq!(score.coverage.scored, 12, "{}", score.satellite);
        assert_eq!(score.coverage.outside_horizon_bins, 1);
        let counts: Vec<usize> = score.bins.iter().map(|bin| bin.count).collect();
        assert_eq!(counts, vec![0, 4, 4, 4]);
        assert!(score.bins[0].radial.is_none() && score.bins[0].sisre.is_none());
    }

    let gps = &report.per_system[0];
    assert_eq!(gps.system, GnssSystem::Gps);
    assert_eq!(gps.coverage.scored, 12 * satellites);
    for bin in &gps.bins[1..] {
        assert_eq!(bin.count, 4 * satellites);
        // An ultra-rapid prediction a few hours out is good to centimeters.
        let rms = bin.orbit_3d.expect("3D").rms_m;
        assert!(rms > 0.005 && rms < 0.15, "3D RMS {rms} m");
        let sisre = bin.sisre_orbit.expect("SISRE").rms_m;
        assert!(sisre < rms, "SISRE(orb) {sisre} m >= 3D {rms} m");
    }
}

// The trimmed copy of the 18:00 ultra-rapid holds predicted records only, so
// its flags give no origin; every pair is counted as such, and refused with
// the typed error under `GapPolicy::Refuse`. A cut-off scores it.
#[test]
fn predicted_records_without_a_fitted_record_need_a_cutoff() {
    let reference = load_sp3(FINAL_SP3);
    let product = load_sp3(ULTRA_RAPID_1800);
    let satellites = reference.satellites();

    let flagged = PredictedSp3::from_prediction_flags(&product).expect("flags");
    let mut options = OrbitScoreOptions::new(vec![18_000.0, 25_200.0, 32_400.0]);
    let report = score_orbit(&flagged, &reference, satellites, &options).expect("score");
    assert_report_coverage_closes(&report);
    assert_eq!(report.coverage.no_prediction_origin, 13 * satellites.len());
    assert_eq!(report.coverage.scored, 0);
    assert!(report
        .per_satellite
        .iter()
        .all(|score| score.prediction_origin_j2000_s.is_none() && score.sisre_weights.is_none()));

    options.gaps = GapPolicy::Refuse;
    assert_eq!(
        score_orbit(&flagged, &reference, satellites, &options),
        Err(OrbitScoreError::NoPredictionOrigin {
            satellite: satellites[0]
        })
    );

    let cut = PredictedSp3::with_cutoff(&product, ultra_rapid_1800_cutoff()).expect("cut-off");
    let report = score_orbit(&cut, &reference, satellites, &options).expect("score");
    assert_report_coverage_closes(&report);
    assert_eq!(report.coverage.scored, 13 * satellites.len());
    let gps = &report.per_system[0];
    // 00:00 to 01:00 is 6 to 7 hours past the cut-off, 01:15 to 03:00 is 7.25 to 9.
    assert_eq!(gps.bins[0].count, 5 * satellites.len());
    assert_eq!(gps.bins[1].count, 8 * satellites.len());
    for bin in &gps.bins {
        let rms = bin.orbit_3d.expect("3D").rms_m;
        assert!(rms > 0.005 && rms < 0.2, "3D RMS {rms} m");
    }
    let weights = report.per_satellite[0].sisre_weights.expect("weights");
    assert_eq!(weights.radial(), 0.98);
    assert_eq!(weights.transverse_normal_squared(), 1.0 / 49.0);
}

// A product with no predicted record has its last record as origin, so no
// epoch of it is at a positive horizon.
#[test]
fn a_product_without_predicted_records_has_no_positive_horizon() {
    let reference = load_sp3(FINAL_SP3);
    let source = PredictedSp3::from_prediction_flags(&reference).expect("flags");
    let options = OrbitScoreOptions::new(vec![-11_000.0, 0.0, 86_400.0]);
    let report = score_orbit(&source, &reference, reference.satellites(), &options).expect("score");
    assert_report_coverage_closes(&report);
    let last = *reference.epochs_j2000_seconds().last().expect("epochs");
    for score in &report.per_satellite {
        assert_eq!(score.prediction_origin_j2000_s, Some(last));
        assert_eq!(score.bins[0].count, 13);
        assert_eq!(score.bins[1].count, 0);
        // The product against itself.
        assert_eq!(score.bins[0].orbit_3d.expect("3D").max_abs_m, 0.0);
    }
}

fn golden_epochs() -> (Vec<GnssSatelliteId>, Vec<EpochInputs>, f64) {
    let doc: Value = serde_json::from_str(BROADCAST_GOLDEN).expect("parse golden");
    let inputs = &doc["inputs"];
    let hex = |value: &Value| {
        let raw = value.as_str().expect("hex bits");
        f64::from_bits(u64::from_str_radix(raw.trim_start_matches("0x"), 16).expect("hex bits"))
    };
    let satellites = inputs["satellites"]
        .as_array()
        .expect("satellites")
        .iter()
        .map(|token| sat(token.as_str().expect("satellite token")))
        .collect();
    let epochs = inputs["epochs"]
        .as_array()
        .expect("epochs")
        .iter()
        .map(|row| {
            let row = row.as_array().expect("epoch row");
            let split = |i: usize| {
                JulianDateSplit::new(hex(&row[i]), hex(&row[i + 1])).expect("valid split")
            };
            EpochInputs {
                broadcast_t_j2000_s: hex(&row[0]),
                precise: split(1),
                precise_plus: split(3),
                precise_minus: split(5),
            }
        })
        .collect();
    let half = inputs["velocity_half_s"].as_f64().expect("velocity_half_s");
    (satellites, epochs, half)
}

fn split_seconds(split: JulianDateSplit) -> f64 {
    (split.jd_whole - 2_451_545.0) * 86_400.0 + split.fraction * 86_400.0
}

// A broadcast ephemeris scored on the Earth-fixed frame at the broadcast
// comparison's epochs reproduces its statistics bit for bit: the same
// differences, the same velocity, the same folds.
#[test]
fn broadcast_source_reproduces_the_broadcast_comparison_bit_for_bit() {
    let nav = std::fs::read_to_string(fixture_path("nav/ESBC00DNK_R_20201770000_01D_MN.rnx"))
        .expect("read broadcast fixture");
    let broadcast = BroadcastEphemeris::from_nav(&nav).expect("parse broadcast");
    let precise = load_sp3("sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3");
    let (satellites, epochs, half) = golden_epochs();

    // The comparison queries the broadcast store at `broadcast_t_j2000_s` and
    // the precise product at the split dates; on this whole-second grid they
    // are the same seconds, so scoring at the split dates queries both
    // products at exactly the instants the comparison does.
    for epoch in &epochs {
        let t = split_seconds(epoch.precise);
        assert_eq!(t.to_bits(), epoch.broadcast_t_j2000_s.to_bits());
        assert_eq!(
            split_seconds(epoch.precise_plus).to_bits(),
            (t + half).to_bits()
        );
        assert_eq!(
            split_seconds(epoch.precise_minus).to_bits(),
            (t - half).to_bits()
        );
    }

    let expected = compare(&broadcast, &precise, &satellites, &epochs, half).expect("compare");

    let instants: Vec<Instant> = epochs
        .iter()
        .map(|epoch| Instant::from_julian_date(TimeScale::Gpst, epoch.precise))
        .collect();
    let source = BroadcastOrbit::new(&broadcast, instants[0]).expect("broadcast source");
    let mut options = OrbitScoreOptions::new(vec![-1.0e6, 1.0e6]);
    options.epochs = ScoreEpochs::Instants(instants);
    options.triad = TriadVelocity::EarthFixed;
    options.velocity_half_step_s = half;

    let bits = |value: Option<f64>| value.map(f64::to_bits);
    for datum in [ClockDatum::Unadjusted, ClockDatum::EpochSystemMedian] {
        options.clock_datum = datum;
        let report = score_orbit(&source, &precise, &satellites, &options).expect("score");
        assert_report_coverage_closes(&report);
        assert_eq!(report.frame, FrameAgreement::BroadcastDatums);
        assert_eq!(
            report.source_reference_point,
            OrbitReferencePoint::AntennaPhaseCenter
        );

        let missing = |satellite: GnssSatelliteId| {
            expected
                .missing
                .iter()
                .find(|(s, _)| *s == satellite)
                .map_or(0, |(_, count)| *count)
        };
        let mut pairs: Vec<(String, &CompareStats, _)> = Vec::new();
        for (satellite, stats) in &expected.per_satellite {
            let score = satellite_score(&report, *satellite);
            assert_eq!(
                score.coverage.epochs - score.coverage.scored,
                missing(*satellite),
                "{satellite}"
            );
            pairs.push((satellite.to_string(), stats, score.bins[0]));
        }
        pairs.push((
            "GPS".to_string(),
            &expected.overall,
            report.per_system[0].bins[0],
        ));

        for (label, stats, bin) in pairs {
            assert_eq!(bin.count, stats.count, "{label}");
            let rms = |s: Option<ErrorStats>| s.map(|s| s.rms_m);
            let max = |s: Option<ErrorStats>| s.map(|s| s.max_abs_m);
            assert_eq!(
                bits(rms(bin.orbit_3d)),
                bits(stats.orbit_3d_rms_m),
                "{label} 3D"
            );
            assert_eq!(
                bits(max(bin.orbit_3d)),
                bits(stats.orbit_3d_max_m),
                "{label} 3D"
            );
            assert_eq!(bits(rms(bin.radial)), bits(stats.radial_rms_m), "{label} R");
            assert_eq!(bits(max(bin.radial)), bits(stats.radial_max_m), "{label} R");
            assert_eq!(
                bits(rms(bin.transverse)),
                bits(stats.along_rms_m),
                "{label} T"
            );
            assert_eq!(
                bits(max(bin.transverse)),
                bits(stats.along_max_m),
                "{label} T"
            );
            assert_eq!(bits(rms(bin.normal)), bits(stats.cross_rms_m), "{label} N");
            assert_eq!(bits(max(bin.normal)), bits(stats.cross_max_m), "{label} N");
            let (clock_rms, clock_max) = match datum {
                ClockDatum::Unadjusted => (stats.clock_rms_m, stats.clock_max_m),
                _ => (
                    stats.clock_datum_removed_rms_m,
                    stats.clock_datum_removed_max_m,
                ),
            };
            assert_eq!(
                bits(rms(bin.clock)),
                bits(clock_rms),
                "{label} clock {datum:?}"
            );
            assert_eq!(
                bits(max(bin.clock)),
                bits(clock_max),
                "{label} clock {datum:?}"
            );
        }
    }
}

/// The inertial orbit frame of `reference` at `t`, built here from the
/// documented definition: centered difference over one second, plus
/// `omega x r`.
fn reference_frame(reference: &Sp3, satellite: GnssSatelliteId, t: f64) -> [[f64; 3]; 3] {
    let position = |t: f64| {
        reference
            .position_at_j2000_seconds(satellite, t)
            .expect("reference state")
            .position
            .as_array()
    };
    let r = position(t);
    let (plus, minus) = (position(t + 0.5), position(t - 0.5));
    let v = [
        (plus[0] - minus[0]) - OMEGA_E_DOT_RAD_S * r[1],
        (plus[1] - minus[1]) + OMEGA_E_DOT_RAD_S * r[0],
        plus[2] - minus[2],
    ];
    let unit = |a: [f64; 3]| {
        let n = norm(a);
        [a[0] / n, a[1] / n, a[2] / n]
    };
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let radial = unit(r);
    let normal = unit(cross(r, v));
    [radial, cross(normal, radial), normal]
}

/// Offsets `(R, T, N)` in meters and a clock offset in seconds per satellite.
const OFFSETS: [(&str, [f64; 3], f64); 4] = [
    ("G01", [0.12, -0.34, 0.05], 1.0e-9),
    ("G02", [-0.08, 0.61, -0.22], 3.0e-9),
    ("G03", [0.31, 0.02, 0.17], -2.0e-9),
    ("G04", [0.0, -0.45, -0.09], 6.0e-9),
];

/// A trajectory that is the reference product displaced by [`OFFSETS`] on its
/// own orbit frame, at every reference record.
fn displaced_trajectory(reference: &Sp3, frame_label: &str) -> OrbitTrajectory {
    let seconds = reference.epochs_j2000_seconds();
    let mut samples = Vec::new();
    for (index, &t) in seconds.iter().enumerate() {
        for (token, offset, clock_offset_s) in OFFSETS {
            let satellite = sat(token);
            let state = reference
                .position_at_j2000_seconds(satellite, t)
                .expect("reference state");
            let r = state.position.as_array();
            let axes = reference_frame(reference, satellite, t);
            let mut position = r;
            for (axis, amount) in axes.iter().zip(offset) {
                for k in 0..3 {
                    position[k] += amount * axis[k];
                }
            }
            samples.push(TrajectorySample {
                satellite,
                epoch: reference.epochs[index],
                position_ecef_m: position,
                clock_s: state.clock_s.map(|clock| clock + clock_offset_s),
            });
        }
    }
    OrbitTrajectory::new(frame_label, ultra_rapid_1800_cutoff(), samples).expect("trajectory")
}

fn trajectory_satellites() -> Vec<GnssSatelliteId> {
    OFFSETS.iter().map(|(token, _, _)| sat(token)).collect()
}

// A caller's trajectory displaced from the reference by known amounts on the
// orbit frame scores to those amounts: per-satellite RMS, mean and maximum of
// a constant are the constant, the clock difference is the offset less the
// per-epoch median of the four offsets.
#[test]
fn trajectory_offsets_are_recovered_on_the_orbit_frame() {
    let reference = load_sp3(FINAL_SP3);
    let trajectory = displaced_trajectory(&reference, "IGc20");
    let satellites = trajectory_satellites();
    assert_eq!(trajectory.satellites().collect::<Vec<_>>(), satellites);
    assert_eq!(trajectory.epochs(), reference.epochs);

    let mut options = OrbitScoreOptions::new(vec![0.0, 40_000.0]);
    let report = score_orbit(&trajectory, &reference, &satellites, &options).expect("score");
    assert_report_coverage_closes(&report);
    assert_eq!(report.coverage.scored, 13 * satellites.len());

    // The displaced positions are rounded to the f64 spacing of a 26 000 km
    // coordinate, 3.7e-9 m per axis.
    let tol = 2.0e-8;
    // Median of the clock offsets 1, 3, -2 and 6 ns: 2 ns.
    let median_clock_offset_s = 2.0e-9;
    for (token, offset, clock_offset_s) in OFFSETS {
        let score = satellite_score(&report, sat(token));
        let bin = &score.bins[0];
        assert_eq!(bin.count, 13);
        for (stats, expected) in [
            (bin.radial, offset[0]),
            (bin.transverse, offset[1]),
            (bin.normal, offset[2]),
        ] {
            let stats = stats.expect("stats");
            assert!((stats.mean_m - expected).abs() <= tol, "{token} mean");
            assert!((stats.rms_m - expected.abs()).abs() <= tol, "{token} RMS");
            assert!(
                (stats.max_abs_m - expected.abs()).abs() <= tol,
                "{token} max"
            );
        }
        let expected_3d = norm(offset);
        assert!((bin.orbit_3d.expect("3D").mean_m - expected_3d).abs() <= tol);
        let clock = bin.clock.expect("clock");
        let expected_clock = (clock_offset_s - median_clock_offset_s) * C_M_S;
        // The offset clock rounds at the spacing of a 3e-4 s clock, 5.4e-20 s,
        // 1.6e-11 m after scaling by the speed of light.
        assert!(
            (clock.mean_m - expected_clock).abs() <= CLOCK_TOL_M,
            "{token} clock"
        );
        assert!((clock.max_abs_m - expected_clock.abs()).abs() <= CLOCK_TOL_M);
    }

    options.clock_datum = ClockDatum::Unadjusted;
    let report = score_orbit(&trajectory, &reference, &satellites, &options).expect("score");
    for (token, _, clock_offset_s) in OFFSETS {
        let clock = satellite_score(&report, sat(token)).bins[0]
            .clock
            .expect("clock");
        assert!(
            (clock.mean_m - clock_offset_s * C_M_S).abs() <= CLOCK_TOL_M,
            "{token}"
        );
    }
}

#[test]
fn time_systems_and_frames_must_match() {
    let reference = load_sp3(FINAL_SP3);
    let satellites = trajectory_satellites();
    let options = OrbitScoreOptions::new(vec![0.0, 40_000.0]);

    let utc_origin = Instant::from_julian_date(
        TimeScale::Utc,
        JulianDateSplit::new(JD_2026_05_12, 0.75).expect("valid split"),
    );
    let utc = OrbitTrajectory::new(
        "IGc20",
        utc_origin,
        [TrajectorySample {
            satellite: satellites[0],
            epoch: utc_origin,
            position_ecef_m: [2.0e7, 1.0e7, 1.0e7],
            clock_s: None,
        }],
    )
    .expect("trajectory");
    assert_eq!(
        score_orbit(&utc, &reference, &satellites, &options),
        Err(OrbitScoreError::TimeSystemMismatch {
            orbit_source: TimeScale::Utc,
            reference: TimeScale::Gpst,
        })
    );
    assert!(matches!(
        PredictedSp3::with_cutoff(&reference, utc_origin),
        Err(OrbitScoreError::InstantTimeSystemMismatch {
            field: "cutoff",
            found: TimeScale::Utc,
            expected: TimeScale::Gpst,
            ..
        })
    ));
    let nav = std::fs::read_to_string(fixture_path("nav/ESBC00DNK_R_20201770000_01D_MN.rnx"))
        .expect("read broadcast fixture");
    let broadcast = BroadcastEphemeris::from_nav(&nav).expect("parse broadcast");
    assert!(matches!(
        BroadcastOrbit::new(&broadcast, utc_origin),
        Err(OrbitScoreError::InstantTimeSystemMismatch {
            field: "prediction_origin",
            ..
        })
    ));
    let mut with_utc_epochs = options.clone();
    with_utc_epochs.epochs = ScoreEpochs::Instants(vec![utc_origin]);
    let gpst_trajectory = displaced_trajectory(&reference, "IGc20");
    assert!(matches!(
        score_orbit(&gpst_trajectory, &reference, &satellites, &with_utc_epochs),
        Err(OrbitScoreError::InstantTimeSystemMismatch {
            field: "epochs",
            index: 0,
            ..
        })
    ));

    // A trajectory labelled ITRF2014 against the IGc20 reference.
    let itrf2014 = displaced_trajectory(&reference, "ITRF2014");
    assert_eq!(
        score_orbit(&itrf2014, &reference, &satellites, &options),
        Err(OrbitScoreError::FrameMismatch {
            orbit_source: "ITRF2014".to_string(),
            reference: "IGc20".to_string(),
        })
    );

    let mut asserted = options.clone();
    asserted.frame_reconciliation.asserted_equivalent_label_sets =
        vec![Sp3FrameLabelSet::new(["ITRF2014", "IGc20"])];
    let report = score_orbit(&itrf2014, &reference, &satellites, &asserted).expect("asserted");
    assert_eq!(report.frame, FrameAgreement::AssertedEquivalent);
    let as_stated = satellite_score(&report, satellites[0]).bins[0]
        .radial
        .expect("radial")
        .mean_m;

    let mut helmert = options.clone();
    helmert.frame_reconciliation = Sp3FrameReconciliationOptions::helmert();
    let report = score_orbit(&itrf2014, &reference, &satellites, &helmert).expect("Helmert");
    assert_eq!(
        report.frame,
        FrameAgreement::Helmert {
            from: TerrestrialFrame::Itrf2014,
            to: TerrestrialFrame::Itrf2020,
        }
    );
    // The ITRF2014 to ITRF2020 transformation in 2026 moves a GNSS position by
    // millimeters, not by nothing and not by meters.
    let transformed = satellite_score(&report, satellites[0]).bins[0]
        .radial
        .expect("radial")
        .mean_m;
    let shift = (transformed - as_stated).abs();
    assert!(shift > 1.0e-5 && shift < 0.05, "Helmert shift {shift} m");

    let unknown = displaced_trajectory(&reference, "LOCAL");
    assert!(matches!(
        score_orbit(&unknown, &reference, &satellites, &helmert),
        Err(OrbitScoreError::FrameMismatch { .. })
    ));
}

#[test]
fn missing_satellites_and_epochs_are_counted_or_refused() {
    let reference = load_sp3(FINAL_SP3);
    let trajectory = displaced_trajectory(&reference, "IGc20");
    let g05 = sat("G05");
    let e01 = sat("E01");
    let g01 = sat("G01");
    let mut options = OrbitScoreOptions::new(vec![0.0, 40_000.0]);

    // G05 is in the reference only, E01 in neither: counted, not dropped.
    let report = score_orbit(&trajectory, &reference, &[g01, g05, e01], &options).expect("score");
    assert_report_coverage_closes(&report);
    assert_eq!(
        satellite_score(&report, g05)
            .coverage
            .source_satellite_missing,
        13
    );
    assert_eq!(
        satellite_score(&report, e01)
            .coverage
            .reference_satellite_missing,
        13
    );
    assert_eq!(report.coverage.scored, 13);
    assert_eq!(report.per_system.len(), 2);
    assert_eq!(report.per_system[1].system, GnssSystem::Galileo);
    assert_eq!(report.per_system[1].bins[0].count, 0);

    options.gaps = GapPolicy::Refuse;
    assert_eq!(
        score_orbit(&trajectory, &reference, &[g01, g05], &options),
        Err(OrbitScoreError::MissingSatellite {
            product: ScoredProduct::Source,
            satellite: g05,
        })
    );
    assert_eq!(
        score_orbit(&trajectory, &reference, &[e01], &options),
        Err(OrbitScoreError::MissingSatellite {
            product: ScoredProduct::Reference,
            satellite: e01,
        })
    );

    // An instant between two trajectory samples: the reference has it, the
    // trajectory does not.
    let between = gpst(1.0, 450.0 / 86_400.0);
    let mut between_options = OrbitScoreOptions::new(vec![0.0, 40_000.0]);
    between_options.epochs = ScoreEpochs::Instants(vec![reference.epochs[0], between]);
    let report = score_orbit(&trajectory, &reference, &[g01], &between_options).expect("score");
    assert_report_coverage_closes(&report);
    assert_eq!(report.coverage.source_epoch_missing, 1);
    assert_eq!(report.coverage.scored, 1);
    between_options.gaps = GapPolicy::Refuse;
    assert!(matches!(
        score_orbit(&trajectory, &reference, &[g01], &between_options),
        Err(OrbitScoreError::MissingEpoch {
            product: ScoredProduct::Source,
            epoch_index: 1,
            ..
        })
    ));

    // A day after the reference ends: the reference has no state.
    let late = gpst(2.0, 0.0);
    let mut late_options = OrbitScoreOptions::new(vec![0.0, 200_000.0]);
    late_options.epochs = ScoreEpochs::Instants(vec![late]);
    let report = score_orbit(&trajectory, &reference, &[g01], &late_options).expect("score");
    assert_eq!(report.coverage.reference_epoch_missing, 1);
    late_options.gaps = GapPolicy::Refuse;
    assert!(matches!(
        score_orbit(&trajectory, &reference, &[g01], &late_options),
        Err(OrbitScoreError::MissingEpoch {
            product: ScoredProduct::Reference,
            epoch_index: 0,
            ..
        })
    ));

    // An epoch outside every bin is not a gap, even under `Refuse`. The first
    // record is 21 600 s past the trajectory's origin.
    let mut narrow = OrbitScoreOptions::new(vec![0.0, 21_000.0]);
    narrow.gaps = GapPolicy::Refuse;
    let report = score_orbit(&trajectory, &reference, &[g01], &narrow).expect("score");
    assert_eq!(report.coverage.outside_horizon_bins, 13);
}

#[test]
fn invalid_inputs_are_refused() {
    let reference = load_sp3(FINAL_SP3);
    let trajectory = displaced_trajectory(&reference, "IGc20");
    let g01 = sat("G01");
    let refused = |options: &OrbitScoreOptions, satellites: &[GnssSatelliteId]| match score_orbit(
        &trajectory,
        &reference,
        satellites,
        options,
    ) {
        Err(OrbitScoreError::InvalidInput { field, .. }) => field,
        other => panic!("expected InvalidInput, got {other:?}"),
    };

    assert_eq!(
        refused(&OrbitScoreOptions::new(vec![0.0]), &[g01]),
        "horizon_bin_edges_s"
    );
    assert_eq!(
        refused(&OrbitScoreOptions::new(vec![0.0, f64::NAN]), &[g01]),
        "horizon_bin_edges_s"
    );
    assert_eq!(
        refused(&OrbitScoreOptions::new(vec![0.0, f64::INFINITY]), &[g01]),
        "horizon_bin_edges_s"
    );
    assert_eq!(
        refused(&OrbitScoreOptions::new(vec![0.0, 10.0, 10.0]), &[g01]),
        "horizon_bin_edges_s"
    );
    let mut options = OrbitScoreOptions::new(vec![0.0, 40_000.0]);
    for half in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        options.velocity_half_step_s = half;
        assert_eq!(refused(&options, &[g01]), "velocity_half_step_s");
    }
    options.velocity_half_step_s = 0.5;
    assert_eq!(refused(&options, &[g01, g01]), "satellites");
    options.epochs = ScoreEpochs::Instants(vec![reference.epochs[1], reference.epochs[0]]);
    assert_eq!(refused(&options, &[g01]), "epochs");

    let origin = ultra_rapid_1800_cutoff();
    let sample = TrajectorySample {
        satellite: g01,
        epoch: reference.epochs[0],
        position_ecef_m: [2.0e7, 1.0e7, 1.0e7],
        clock_s: None,
    };
    let field = |result: Result<OrbitTrajectory, OrbitScoreError>| match result {
        Err(OrbitScoreError::InvalidInput { field, .. }) => field,
        other => panic!("expected InvalidInput, got {other:?}"),
    };
    assert_eq!(field(OrbitTrajectory::new("IGc20", origin, [])), "samples");
    assert_eq!(
        field(OrbitTrajectory::new("IGc20", origin, [sample, sample])),
        "samples"
    );
    let mut bad = sample;
    bad.position_ecef_m[1] = f64::NAN;
    assert_eq!(
        field(OrbitTrajectory::new("IGc20", origin, [bad])),
        "samples.position_ecef_m"
    );
    let mut bad = sample;
    bad.clock_s = Some(f64::INFINITY);
    assert_eq!(
        field(OrbitTrajectory::new("IGc20", origin, [bad])),
        "samples.clock_s"
    );
}

/// A caller's source that returns one fixed state for every satellite and
/// instant.
struct FixedSource {
    state: OrbitSourceState,
    origin_j2000_s: Option<f64>,
    frame_label: &'static str,
}

impl OrbitSource for FixedSource {
    fn time_scale(&self) -> TimeScale {
        TimeScale::Gpst
    }

    fn frame(&self) -> OrbitFrame {
        OrbitFrame::Label(self.frame_label.to_string())
    }

    fn state_at_j2000_s(
        &self,
        _satellite: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Result<OrbitSourceState, sidereon_core::Error> {
        Ok(self.state)
    }

    fn prediction_origin_j2000_s(&self, _satellite: GnssSatelliteId) -> Option<f64> {
        self.origin_j2000_s
    }
}

// A source that hands back a non-finite state, an overflowing one, or a
// non-finite prediction origin is refused with a typed error, never scored
// into NaN statistics.
#[test]
fn non_finite_source_values_are_refused() {
    let reference = load_sp3(FINAL_SP3);
    let g01 = sat("G01");
    let options = OrbitScoreOptions::new(vec![-1.0e6, 1.0e6]);
    let origin = Some(FINAL_START_J2000_S);
    let source = |position_ecef_m: [f64; 3], clock_s: Option<f64>, origin_j2000_s| FixedSource {
        state: OrbitSourceState {
            position_ecef_m,
            clock_s,
        },
        origin_j2000_s,
        frame_label: "IGc20",
    };

    for state in [
        source([f64::NAN, 1.0e7, 1.0e7], None, origin),
        source([2.0e7, 1.0e7, 1.0e7], Some(f64::INFINITY), origin),
    ] {
        assert_eq!(
            score_orbit(&state, &reference, &[g01], &options),
            Err(OrbitScoreError::NonFiniteSourceState {
                satellite: g01,
                epoch_index: 0,
            })
        );
    }
    // A difference of 1.0e308 m per axis is 1.73e308 m long, a finite number,
    // and is scored; 1.7e308 m per axis is 2.9e308 m, beyond any `f64`.
    let report = score_orbit(
        &source([1.0e308, 1.0e308, 1.0e308], None, origin),
        &reference,
        &[g01],
        &options,
    )
    .expect("a finite difference is scored");
    let orbit_3d = report.per_satellite[0].bins[0].orbit_3d.expect("3D");
    assert!((orbit_3d.max_abs_m / (3.0_f64.sqrt() * 1.0e308) - 1.0).abs() < 1.0e-12);
    assert_eq!(
        score_orbit(
            &source([1.7e308, 1.7e308, 1.7e308], None, origin),
            &reference,
            &[g01],
            &options
        ),
        Err(OrbitScoreError::NonFiniteResult { satellite: g01 })
    );
    assert!(matches!(
        score_orbit(
            &source([2.0e7, 1.0e7, 1.0e7], None, Some(f64::NAN)),
            &reference,
            &[g01],
            &options
        ),
        Err(OrbitScoreError::InvalidInput {
            field: "prediction_origin",
            ..
        })
    ));
}

// A satellite one of the products has no records of is missing from that
// product, whatever the source states about its prediction origin. The
// ultra-rapid issued 2026-05-12 00:00 with G05's records taken out, read from
// its flags or with a cut-off, counts G05 as missing from the source and E01
// (in neither product) as missing from the reference, at every epoch, and
// refuses each under `GapPolicy::Refuse` with the product that lacks it.
#[test]
fn a_satellite_without_records_is_missing_not_unpredicted() {
    let reference = load_sp3(FINAL_SP3);
    let without_g05: String = fixture_text(ULTRA_RAPID_0000)
        .lines()
        .filter(|line| !line.starts_with("PG05"))
        .map(|line| format!("{line}\n"))
        .collect();
    let product = parse_text(&without_g05);
    let (g01, g05, e01) = (sat("G01"), sat("G05"), sat("E01"));
    let flagged = PredictedSp3::from_prediction_flags(&product).expect("flags");
    assert_eq!(flagged.prediction_origin_j2000_s(g05), None);
    // 2026-05-12 23:45, the flagged origin of the satellites that have one.
    let cut = PredictedSp3::with_cutoff(&product, gpst(0.0, 23.75 / 24.0)).expect("cut-off");

    for (label, source) in [("flags", &flagged), ("cut-off", &cut)] {
        let mut options = OrbitScoreOptions::new(vec![-3600.0, 0.0, 3600.0, 7200.0, 10_800.0]);
        let report = score_orbit(source, &reference, &[g01, g05, e01], &options).expect("score");
        assert_report_coverage_closes(&report);
        let g05_coverage = satellite_score(&report, g05).coverage;
        assert_eq!(g05_coverage.source_satellite_missing, 13, "{label}");
        assert_eq!(g05_coverage.no_prediction_origin, 0, "{label}");
        assert_eq!(g05_coverage.outside_horizon_bins, 0, "{label}");
        let e01_coverage = satellite_score(&report, e01).coverage;
        assert_eq!(e01_coverage.reference_satellite_missing, 13, "{label}");
        assert_eq!(satellite_score(&report, g01).coverage.scored, 12, "{label}");

        options.gaps = GapPolicy::Refuse;
        assert_eq!(
            score_orbit(source, &reference, &[g05], &options),
            Err(OrbitScoreError::MissingSatellite {
                product: ScoredProduct::Source,
                satellite: g05,
            }),
            "{label}"
        );
        assert_eq!(
            score_orbit(source, &reference, &[e01], &options),
            Err(OrbitScoreError::MissingSatellite {
                product: ScoredProduct::Reference,
                satellite: e01,
            }),
            "{label}"
        );
    }

    // The trimmed 18:00 issue states no origin for anything; a satellite in
    // neither product is still missing from the reference.
    let early = load_sp3(ULTRA_RAPID_1800);
    let unflagged = PredictedSp3::from_prediction_flags(&early).expect("flags");
    let options = OrbitScoreOptions::new(vec![18_000.0, 32_400.0]);
    let report = score_orbit(&unflagged, &reference, &[g01, e01], &options).expect("score");
    assert_eq!(
        satellite_score(&report, g01).coverage.no_prediction_origin,
        13
    );
    assert_eq!(
        satellite_score(&report, e01)
            .coverage
            .reference_satellite_missing,
        13
    );
}

/// One civil epoch label `(year, month, day, hour, minute, second)` and the
/// seconds of motion it is at.
type SyntheticEpoch = ((i32, u32, u32, u32, u32, f64), f64);

/// A synthetic SP3-c product in `time_system`: G01 and G02 on circular orbits
/// of 26 560 km inclined 55 degrees, one record each per label, the position
/// advanced by the label's seconds of motion and turned into the Earth-fixed
/// frame by the nominal Earth rotation over those seconds.
fn synthetic_sp3(time_system: &str, line2: &str, epochs: &[SyntheticEpoch]) -> Sp3 {
    let ((year, month, day, hour, minute, second), _) = epochs[0];
    let mut text = format!(
        "#cP{year:4} {month:2} {day:2} {hour:2} {minute:2} {second:11.8}     {count:3} ORBIT IGS14 FIT  TST\n\
         {line2}\n\
         +    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
         %c G  cc {time_system} ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
         %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
         %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         %i    0    0    0    0      0      0      0      0         0\n\
         /* SYNTHETIC ORBIT SCORING FIXTURE\n",
        count = epochs.len()
    );
    let inclination = 55.0_f64.to_radians();
    let mean_motion = core::f64::consts::TAU / 43_082.0;
    for &((year, month, day, hour, minute, second), elapsed_s) in epochs {
        text.push_str(&format!(
            "*  {year:4} {month:2} {day:2} {hour:2} {minute:2} {second:11.8}\n"
        ));
        for (prn, phase) in [(1, 0.3), (2, 2.1)] {
            let u = phase + mean_motion * elapsed_s;
            let (xi, yi, z) = (
                26_560.0 * libm::cos(u),
                26_560.0 * libm::sin(u) * libm::cos(inclination),
                26_560.0 * libm::sin(u) * libm::sin(inclination),
            );
            let earth_angle = OMEGA_E_DOT_RAD_S * elapsed_s;
            let (x, y) = (
                libm::cos(earth_angle) * xi + libm::sin(earth_angle) * yi,
                libm::cos(earth_angle) * yi - libm::sin(earth_angle) * xi,
            );
            let clock_us = 10.0 * f64::from(prn) + 1.0e-6 * elapsed_s;
            text.push_str(&format!(
                "PG{prn:02}{x:14.6}{y:14.6}{z:14.6}{clock_us:14.6}\n"
            ));
        }
    }
    text.push_str("EOF\n");
    parse_text(&text)
}

/// A trajectory that is `reference` offset by a fixed Earth-fixed vector, with
/// one sample per satellite at each of the reference's record epochs.
fn offset_trajectory(reference: &Sp3, offset_m: [f64; 3]) -> OrbitTrajectory {
    let mut samples = Vec::new();
    for (epoch, t) in reference
        .epochs
        .iter()
        .zip(reference.epochs_j2000_seconds())
    {
        for satellite in [sat("G01"), sat("G02")] {
            let state = reference
                .position_at_j2000_seconds(satellite, t)
                .expect("record");
            let r = state.position.as_array();
            samples.push(TrajectorySample {
                satellite,
                epoch: *epoch,
                position_ecef_m: [r[0] + offset_m[0], r[1] + offset_m[1], r[2] + offset_m[2]],
                clock_s: state.clock_s,
            });
        }
    }
    OrbitTrajectory::new("IGS14", reference.epochs[0], samples).expect("trajectory")
}

/// Every record epoch of both satellites is scored, under either gap policy,
/// and the offset comes back as the 3D difference.
fn assert_every_record_scored(trajectory: &OrbitTrajectory, reference: &Sp3, offset_m: [f64; 3]) {
    let satellites = [sat("G01"), sat("G02")];
    let records = reference.epochs.len();
    for gaps in [GapPolicy::Count, GapPolicy::Refuse] {
        let mut options = OrbitScoreOptions::new(vec![-1.0e6, 1.0e6]);
        options.gaps = gaps;
        let report = score_orbit(trajectory, reference, &satellites, &options).expect("score");
        assert_report_coverage_closes(&report);
        assert_eq!(report.coverage.scored, 2 * records, "{gaps:?}");
        for score in &report.per_satellite {
            let orbit_3d = score.bins[0].orbit_3d.expect("3D");
            // The offset is added to positions of 2.7e7 m, which round at
            // 3.7e-9 m per axis.
            assert!(
                (orbit_3d.max_abs_m - norm(offset_m)).abs() < 1.0e-7,
                "{} {}",
                score.satellite,
                orbit_3d.max_abs_m
            );
        }
    }
    // Scored at its own instants, the trajectory is found at every one too.
    let mut options = OrbitScoreOptions::new(vec![-1.0e6, 1.0e6]);
    options.epochs = ScoreEpochs::Instants(trajectory.epochs());
    let report = score_orbit(trajectory, reference, &satellites, &options).expect("score");
    assert_eq!(report.coverage.scored, 2 * records);
}

// On 2000-01-01 the split Julian dates of 06:10, 06:35 and 06:55 reduce to
// seconds a unit in the last place away from the whole seconds the reader
// puts the records at. A trajectory sampled at the record epochs is found at
// all thirteen.
#[test]
fn trajectory_samples_at_record_epochs_are_found_near_j2000() {
    let epochs: Vec<SyntheticEpoch> = (0..13)
        .map(|k| {
            (
                (2000, 1, 1, 6 + (k * 5) / 60, (k * 5) % 60, 0.0),
                300.0 * f64::from(k),
            )
        })
        .collect();
    let reference = synthetic_sp3(
        "GPS",
        "## 1042 540000.00000000   300.00000000 51544 0.2500000000000",
        &epochs,
    );
    let off_their_second = reference
        .epochs
        .iter()
        .zip(reference.epochs_j2000_seconds())
        .filter(|(epoch, t)| split_seconds(epoch.julian_date().expect("split")) != *t)
        .count();
    assert_eq!(off_their_second, 3);

    let offset_m = [1.0, -2.0, 0.5];
    let trajectory = offset_trajectory(&reference, offset_m);
    assert_every_record_scored(&trajectory, &reference, offset_m);
}

// A UTC product with a record at the leap second 2016-12-31 23:59:60. The
// reader puts that record one second after 23:59:59; a trajectory sampled at
// the record epochs has its 23:59:59 and 23:59:60 samples one second apart
// too, and is found at all sixteen records.
#[test]
fn trajectory_samples_at_a_leap_second_record_are_found() {
    let epochs: Vec<SyntheticEpoch> = (0..16)
        .map(|k| ((2016, 12, 31, 23, 59, 45.0 + f64::from(k)), f64::from(k)))
        .collect();
    let reference = synthetic_sp3(
        "UTC",
        "## 1929 604785.00000000     1.00000000 57753 0.9998263888889",
        &epochs,
    );
    let seconds = reference.epochs_j2000_seconds();
    assert_eq!(seconds[15] - seconds[14], 1.0);

    let offset_m = [0.25, 0.75, -1.5];
    let trajectory = offset_trajectory(&reference, offset_m);
    assert_eq!(trajectory.epochs().len(), 16);
    assert_every_record_scored(&trajectory, &reference, offset_m);
}

// A velocity half step below the spacing of `f64` seconds at the epochs
// (1.2e-7 s in 2026) leaves the difference no step: the reference velocity
// would be zero and the frame built from the Earth rotation alone. It is
// refused.
#[test]
fn a_velocity_half_step_that_rounds_away_is_refused() {
    let reference = load_sp3(FINAL_SP3);
    let trajectory = displaced_trajectory(&reference, "IGc20");
    let mut options = OrbitScoreOptions::new(vec![0.0, 40_000.0]);
    for half_step_s in [1.0e-8, 5.0e-8] {
        options.velocity_half_step_s = half_step_s;
        assert!(matches!(
            score_orbit(&trajectory, &reference, &trajectory_satellites(), &options),
            Err(OrbitScoreError::InvalidInput {
                field: "velocity_half_step_s",
                ..
            })
        ));
    }
    // Two units of that spacing is a step, and is scored.
    options.velocity_half_step_s = 2.5e-7;
    let report =
        score_orbit(&trajectory, &reference, &trajectory_satellites(), &options).expect("score");
    assert_eq!(report.coverage.scored, 13 * 4);
}

// An SP3 product in IRNSS time is read on the GPS time axis but is not in GPS
// time. Scoring it against a GPS-time product, or a GPS-time source against
// it, is refused, as merging the two is; two IRNSS-time products score as two
// GPS-time ones do.
#[test]
fn irnss_time_is_not_gps_time() {
    let irnss =
        |name: &str| parse_text(&fixture_text(name).replacen("%c G  cc GPS", "%c G  cc IRN", 1));
    let reference = load_sp3(FINAL_SP3);
    let irnss_reference = irnss(FINAL_SP3);
    assert_eq!(irnss_reference.header.time_system, Sp3TimeSystem::Irnss);
    assert_eq!(irnss_reference.header.time_scale, TimeScale::Gpst);
    let options = OrbitScoreOptions::new(vec![-3600.0, 0.0, 3600.0, 7200.0, 10_800.0]);
    let satellites = reference.satellites();

    let gps_product = load_sp3(ULTRA_RAPID_0000);
    let irnss_product = irnss(ULTRA_RAPID_0000);
    let gps_source = PredictedSp3::from_prediction_flags(&gps_product).expect("flags");
    let irnss_source = PredictedSp3::from_prediction_flags(&irnss_product).expect("flags");
    assert_eq!(
        score_orbit(&irnss_source, &reference, satellites, &options),
        Err(OrbitScoreError::TimeSystemLabelMismatch {
            orbit_source: Sp3TimeSystem::Irnss,
            reference: Sp3TimeSystem::Gps,
        })
    );
    let gps_mismatch = Err(OrbitScoreError::TimeSystemLabelMismatch {
        orbit_source: Sp3TimeSystem::Gps,
        reference: Sp3TimeSystem::Irnss,
    });
    assert_eq!(
        score_orbit(&gps_source, &irnss_reference, satellites, &options),
        gps_mismatch
    );
    let trajectory = displaced_trajectory(&reference, "IGc20");
    assert_eq!(
        score_orbit(&trajectory, &irnss_reference, satellites, &options),
        gps_mismatch
    );
    let nav = std::fs::read_to_string(fixture_path("nav/ESBC00DNK_R_20201770000_01D_MN.rnx"))
        .expect("read broadcast fixture");
    let broadcast = BroadcastEphemeris::from_nav(&nav).expect("parse broadcast");
    let broadcast_source =
        BroadcastOrbit::new(&broadcast, ultra_rapid_1800_cutoff()).expect("broadcast source");
    assert_eq!(
        score_orbit(&broadcast_source, &irnss_reference, satellites, &options),
        gps_mismatch
    );

    let irnss_report =
        score_orbit(&irnss_source, &irnss_reference, satellites, &options).expect("score");
    let gps_report = score_orbit(&gps_source, &reference, satellites, &options).expect("score");
    assert_eq!(irnss_report, gps_report);
}

// A blank frame label states no frame. A trajectory is refused one; an
// asserted label set holding a blank label is refused, as the merge refuses
// it; and a source with a blank label never matches the reference.
#[test]
fn blank_frame_labels_are_refused() {
    let reference = load_sp3(FINAL_SP3);
    let g01 = sat("G01");
    let sample = TrajectorySample {
        satellite: g01,
        epoch: reference.epochs[0],
        position_ecef_m: [2.0e7, 1.0e7, 1.0e7],
        clock_s: None,
    };
    for label in ["", "   "] {
        assert!(matches!(
            OrbitTrajectory::new(label, ultra_rapid_1800_cutoff(), [sample]),
            Err(OrbitScoreError::InvalidInput {
                field: "frame_label",
                ..
            })
        ));
    }

    let blank = FixedSource {
        state: OrbitSourceState {
            position_ecef_m: [2.0e7, 1.0e7, 1.0e7],
            clock_s: None,
        },
        origin_j2000_s: Some(FINAL_START_J2000_S),
        frame_label: "",
    };
    let mut options = OrbitScoreOptions::new(vec![-1.0e6, 1.0e6]);
    assert_eq!(
        score_orbit(&blank, &reference, &[g01], &options),
        Err(OrbitScoreError::FrameMismatch {
            orbit_source: String::new(),
            reference: "IGc20".to_string(),
        })
    );
    for labels in [vec!["", "IGc20"], vec!["IGc20"]] {
        options.frame_reconciliation.asserted_equivalent_label_sets =
            vec![Sp3FrameLabelSet::new(labels)];
        assert!(matches!(
            score_orbit(&blank, &reference, &[g01], &options),
            Err(OrbitScoreError::InvalidInput {
                field: "frame_reconciliation.asserted_equivalent_label_sets",
                ..
            })
        ));
    }
}

/// A synthetic GPS-time product with records every 300 s from 11:00 to 13:00
/// on 2000-01-01, so that one record is at J2000 itself (`t = 0`).
fn j2000_reference() -> Sp3 {
    let epochs: Vec<SyntheticEpoch> = (0..25)
        .map(|k| {
            (
                (2000, 1, 1, 11 + (k * 5) / 60, (k * 5) % 60, 0.0),
                300.0 * f64::from(k),
            )
        })
        .collect();
    synthetic_sp3(
        "GPS",
        "## 1042 558000.00000000   300.00000000 51544 0.4583333333333",
        &epochs,
    )
}

// At J2000 the seconds axis is fine enough that `t +- h` stays apart from `t`
// for any half step. Far below the 1e-8 s resolution of SP3 epochs the
// interpolated positions, rounding at 3.7e-9 m, no longer resolve the motion:
// at 1e-11 s the frame turns by two degrees, from 1e-12 s down the velocity
// is rounding alone and no orbit regime is found, and at 1e-310 s the
// reciprocal step overflows. Every step below 1e-8 s is refused as too short,
// under either frame.
#[test]
fn a_velocity_half_step_below_the_sp3_epoch_resolution_is_refused() {
    let reference = j2000_reference();
    assert!(reference.epochs_j2000_seconds().contains(&0.0));
    let trajectory = offset_trajectory(&reference, [1.0, -2.0, 0.5]);
    let satellites = [sat("G01"), sat("G02")];
    for t_s in [0, 300] {
        let mut options = OrbitScoreOptions::new(vec![-1.0e6, 1.0e6]);
        options.epochs = ScoreEpochs::Instants(vec![Instant::from_nanos(
            TimeScale::Gpst,
            i128::from(t_s) * 1_000_000_000,
        )]);
        for triad in [TriadVelocity::Inertial, TriadVelocity::EarthFixed] {
            options.triad = triad;
            for half_step_s in [5.0e-9, 1.0e-11, 1.0e-12, 1.0e-300, 1.0e-310] {
                options.velocity_half_step_s = half_step_s;
                assert!(
                    matches!(
                        score_orbit(&trajectory, &reference, &satellites, &options),
                        Err(OrbitScoreError::InvalidInput {
                            field: "velocity_half_step_s",
                            ..
                        })
                    ),
                    "t {t_s} s, {triad:?}, h {half_step_s:e}"
                );
            }
            // At 1e-8 s the motion over the step is 2e4 times the rounding.
            options.velocity_half_step_s = 1.0e-8;
            let report =
                score_orbit(&trajectory, &reference, &satellites, &options).expect("score");
            assert_eq!(report.coverage.scored, 2, "t {t_s} s, {triad:?}");
        }
    }
}
