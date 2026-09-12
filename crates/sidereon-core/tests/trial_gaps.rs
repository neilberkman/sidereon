use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::astro::time::split_julian_date;
use sidereon_core::constants::{
    C_M_S, F_L1_HZ, SECONDS_PER_DAY, SECONDS_PER_HOUR, SECONDS_PER_MINUTE,
};
use sidereon_core::ephemeris::BroadcastEphemeris;
use sidereon_core::observables::{
    j2000_seconds_from_split, predict, ObservableEphemerisSource, ObservableState,
    ObservablesError, PredictOptions,
};
use sidereon_core::positioning::{
    solve, solve_doppler_velocity, solve_with_doppler_velocity, Corrections, DopplerObservation,
    DopplerVelocityInputs, EphemerisSource, KlobucharCoeffs, Observation, SolveInputs, SurfaceMet,
};
use sidereon_core::rinex::nav::parse_nav;
use sidereon_core::rinex::observations::{
    observation_values, ObsEpoch, ObsEpochTime, ObservationFilter, RinexObs,
};
use sidereon_core::rinex::qc::{repair_obs_text, RepairOptions};
use sidereon_core::velocity::range_rate_to_doppler;
use sidereon_core::{Error, GnssSatelliteId, GnssSystem};

#[derive(Debug, Clone, Copy)]
struct SatState {
    id: GnssSatelliteId,
    position0_m: [f64; 3],
    velocity_m_s: [f64; 3],
}

#[derive(Debug, Clone)]
struct LinearSource {
    t0_s: f64,
    sats: Vec<SatState>,
}

impl LinearSource {
    fn state(&self, sat: GnssSatelliteId, t_s: f64) -> Option<[f64; 3]> {
        let state = self.sats.iter().find(|state| state.id == sat)?;
        let dt = t_s - self.t0_s;
        Some([
            state.position0_m[0] + state.velocity_m_s[0] * dt,
            state.position0_m[1] + state.velocity_m_s[1] * dt,
            state.position0_m[2] + state.velocity_m_s[2] * dt,
        ])
    }
}

impl EphemerisSource for LinearSource {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        Some((self.state(sat, t_j2000_s)?, 0.0))
    }
}

impl ObservableEphemerisSource for LinearSource {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let Some(position_ecef_m) = self.state(sat, t_j2000_s) else {
            return Err(ObservablesError::NoEphemeris);
        };
        Ok(ObservableState {
            position_ecef_m,
            clock_s: Some(0.0),
        })
    }
}

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS satellite")
}

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn unit(v: [f64; 3]) -> [f64; 3] {
    let n = norm(v);
    [v[0] / n, v[1] / n, v[2] / n]
}

fn add_scaled(a: [f64; 3], b: [f64; 3], scale: f64) -> [f64; 3] {
    [
        a[0] + b[0] * scale,
        a[1] + b[1] * scale,
        a[2] + b[2] * scale,
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn synthetic_case() -> (LinearSource, [f64; 3], [f64; 3], f64) {
    let t0_s = 800_000.0;
    let receiver0_m = [6_378_137.0, 0.0, 0.0];
    let receiver_velocity_m_s = [8.0, -3.0, 2.0];
    let directions = [
        [0.75, 0.50, 0.43],
        [0.65, -0.60, 0.46],
        [0.85, 0.25, -0.46],
        [0.55, -0.10, 0.83],
        [0.70, 0.70, -0.14],
        [0.90, -0.35, 0.26],
    ];
    let sat_velocities = [
        [80.0, -320.0, 120.0],
        [-40.0, 260.0, -160.0],
        [140.0, 40.0, 210.0],
        [-180.0, -130.0, 70.0],
        [30.0, 180.0, -220.0],
        [110.0, -90.0, -140.0],
    ];
    let sats = directions
        .into_iter()
        .zip(sat_velocities)
        .enumerate()
        .map(|(index, (direction, velocity_m_s))| SatState {
            id: gps((index + 1) as u8),
            position0_m: add_scaled(receiver0_m, unit(direction), 23_000_000.0),
            velocity_m_s,
        })
        .collect();

    (
        LinearSource { t0_s, sats },
        receiver0_m,
        receiver_velocity_m_s,
        t0_s,
    )
}

fn receiver_position_at(
    receiver0_m: [f64; 3],
    velocity_m_s: [f64; 3],
    t0_s: f64,
    t_s: f64,
) -> [f64; 3] {
    add_scaled(receiver0_m, velocity_m_s, t_s - t0_s)
}

fn pseudorange_inputs(
    source: &LinearSource,
    receiver0_m: [f64; 3],
    receiver_velocity_m_s: [f64; 3],
    t0_s: f64,
    t_s: f64,
) -> SolveInputs {
    let receiver_m = receiver_position_at(receiver0_m, receiver_velocity_m_s, t0_s, t_s);
    let observations = source
        .sats
        .iter()
        .map(|sat| {
            let predicted = predict(source, sat.id, receiver_m, t_s, PredictOptions::default())
                .expect("synthetic range prediction");
            Observation {
                satellite_id: sat.id,
                pseudorange_m: predicted.geometric_range_m,
            }
        })
        .collect();
    SolveInputs {
        observations,
        t_rx_j2000_s: t_s,
        t_rx_second_of_day_s: t_s.rem_euclid(86_400.0),
        day_of_year: 100.0,
        initial_guess: [
            receiver_m[0] + 40.0,
            receiver_m[1] - 30.0,
            receiver_m[2] + 20.0,
            0.0,
        ],
        corrections: Corrections::NONE,
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: Default::default(),
        met: SurfaceMet::default(),
        robust: None,
    }
}

fn doppler_observations(
    source: &LinearSource,
    receiver0_m: [f64; 3],
    receiver_velocity_m_s: [f64; 3],
    t0_s: f64,
    clock_drift_s_s: f64,
) -> Vec<DopplerObservation> {
    source
        .sats
        .iter()
        .map(|sat| {
            let predicted = predict(source, sat.id, receiver0_m, t0_s, PredictOptions::default())
                .expect("synthetic Doppler prediction");
            let range_rate_m_s = predicted.range_rate_m_s
                - dot(predicted.los_unit, receiver_velocity_m_s)
                + C_M_S * clock_drift_s_s;
            DopplerObservation {
                satellite_id: sat.id,
                doppler_hz: range_rate_to_doppler(range_rate_m_s, F_L1_HZ)
                    .expect("range-rate to Doppler"),
                carrier_hz: F_L1_HZ,
                sat_clock_drift_s_s: 0.0,
            }
        })
        .collect()
}

fn synthetic_rinex_obs_epoch(
    source: &LinearSource,
    receiver0_m: [f64; 3],
    receiver_velocity_m_s: [f64; 3],
    t0_s: f64,
    clock_drift_s_s: f64,
) -> String {
    let mut text = String::new();
    text.push_str("3.04 OBSERVATION DATA M RINEX VERSION / TYPE\n");
    text.push_str(&obs_header_line("TEST", "MARKER NAME"));
    text.push_str(&obs_header_line(
        "PGM                 RUN                 DATE",
        "PGM / RUN BY / DATE",
    ));
    text.push_str(&obs_header_line(
        "OBS                 AGENCY",
        "OBSERVER / AGENCY",
    ));
    text.push_str(&obs_header_line(
        "REC                 TYPE                VERS",
        "REC # / TYPE / VERS",
    ));
    text.push_str(&obs_header_line("ANT                 TYPE", "ANT # / TYPE"));
    text.push_str(&obs_header_line(
        "  6378137.0000        0.0000        0.0000",
        "APPROX POSITION XYZ",
    ));
    text.push_str(&obs_header_line(
        "        0.0000        0.0000        0.0000",
        "ANTENNA: DELTA H/E/N",
    ));
    text.push_str("G 2 C1C D1C SYS / # / OBS TYPES\n");
    text.push_str(&obs_header_line(
        "  2020     1     1     0     0    0.0000000     GPS",
        "TIME OF FIRST OBS",
    ));
    text.push_str(&obs_header_line("", "END OF HEADER"));
    text.push_str(&format!(
        "> 2020 01 01 00 00  0.0000000  0{:3}\n",
        source.sats.len()
    ));

    for sat in &source.sats {
        let predicted = predict(source, sat.id, receiver0_m, t0_s, PredictOptions::default())
            .expect("synthetic RINEX prediction");
        let range_rate_m_s = predicted.range_rate_m_s
            - dot(predicted.los_unit, receiver_velocity_m_s)
            + C_M_S * clock_drift_s_s;
        let doppler_hz =
            range_rate_to_doppler(range_rate_m_s, F_L1_HZ).expect("range-rate to Doppler");
        text.push_str(&format!(
            "{:<3}{:14.3}  {:14.3}  \n",
            sat.id, predicted.geometric_range_m, doppler_hz
        ));
    }

    text
}

fn parsed_l1_rows(obs: &RinexObs) -> (Vec<Observation>, Vec<DopplerObservation>) {
    let codes = obs.obs_codes(GnssSystem::Gps).expect("GPS code table");
    let c1c = codes.iter().position(|code| code == "C1C").expect("C1C");
    let d1c = codes.iter().position(|code| code == "D1C").expect("D1C");
    let epoch = &obs.epochs()[0];
    let mut ranges = Vec::new();
    let mut doppler = Vec::new();
    for (sat, values) in &epoch.sats {
        ranges.push(Observation {
            satellite_id: *sat,
            pseudorange_m: values[c1c].value.expect("C1C value"),
        });
        doppler.push(DopplerObservation {
            satellite_id: *sat,
            doppler_hz: values[d1c].value.expect("D1C value"),
            carrier_hz: F_L1_HZ,
            sat_clock_drift_s_s: 0.0,
        });
    }
    (ranges, doppler)
}

fn fixture_text(parts: &[&str]) -> String {
    let mut path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    for part in parts {
        path.push(part);
    }
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read fixture {path:?}: {err}"))
}

fn civil_to_julian_split(epoch: ObsEpochTime) -> JulianDateSplit {
    let (jd_whole, fraction) = split_julian_date(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    );
    JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date")
}

fn j2000_seconds(epoch: ObsEpochTime) -> f64 {
    let split = civil_to_julian_split(epoch);
    j2000_seconds_from_split(split.jd_whole, split.fraction).expect("valid split Julian date")
}

fn second_of_day(epoch: ObsEpochTime) -> f64 {
    f64::from(epoch.hour) * SECONDS_PER_HOUR
        + f64::from(epoch.minute) * SECONDS_PER_MINUTE
        + epoch.second
}

fn esbc_epoch_rows(
    obs: &RinexObs,
    epoch: &ObsEpoch,
) -> (Vec<Observation>, Vec<DopplerObservation>) {
    let filter = ObservationFilter::from_entries([(
        GnssSystem::Gps,
        vec!["C1C".to_string(), "D1C".to_string()],
    )]);
    let values = observation_values(obs, epoch, &filter).expect("observation values");
    let mut ranges = Vec::new();
    let mut doppler = Vec::new();
    for (sat, rows) in values {
        if sat.system != GnssSystem::Gps {
            continue;
        }
        if let Some(pseudorange_m) = rows
            .iter()
            .find(|row| row.code == "C1C")
            .and_then(|row| row.value)
        {
            ranges.push(Observation {
                satellite_id: sat,
                pseudorange_m,
            });
        }
        if let Some(doppler_hz) = rows
            .iter()
            .find(|row| row.code == "D1C")
            .and_then(|row| row.value)
        {
            doppler.push(DopplerObservation {
                satellite_id: sat,
                doppler_hz,
                carrier_hz: F_L1_HZ,
                sat_clock_drift_s_s: 0.0,
            });
        }
    }
    (ranges, doppler)
}

fn esbc_solve_inputs(obs: &RinexObs, epoch: &ObsEpoch) -> SolveInputs {
    let (observations, _) = esbc_epoch_rows(obs, epoch);
    assert!(
        observations.len() >= 5,
        "need a redundant GPS set, got {}",
        observations.len()
    );
    let approx = obs.header().approx_position_m.expect("APPROX POSITION XYZ");
    let sod = second_of_day(epoch.epoch);
    SolveInputs {
        observations,
        t_rx_j2000_s: j2000_seconds(epoch.epoch),
        t_rx_second_of_day_s: sod,
        day_of_year: 177.0 + sod / SECONDS_PER_DAY,
        initial_guess: [approx[0], approx[1], approx[2], 0.0],
        corrections: Corrections {
            ionosphere: false,
            troposphere: true,
        },
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: Default::default(),
        met: SurfaceMet::default(),
        robust: None,
    }
}

#[test]
fn spp_doppler_velocity_matches_finite_difference_position_oracle() {
    let (source, receiver0_m, receiver_velocity_m_s, t0_s) = synthetic_case();
    let half_dt_s = 20.0;
    let before = pseudorange_inputs(
        &source,
        receiver0_m,
        receiver_velocity_m_s,
        t0_s,
        t0_s - half_dt_s,
    );
    let after = pseudorange_inputs(
        &source,
        receiver0_m,
        receiver_velocity_m_s,
        t0_s,
        t0_s + half_dt_s,
    );
    let before_solution = solve(&source, &before, false).expect("solve before epoch");
    let after_solution = solve(&source, &after, false).expect("solve after epoch");
    let before_pos = before_solution.position.as_array();
    let after_pos = after_solution.position.as_array();
    let finite_difference_m_s = [
        (after_pos[0] - before_pos[0]) / (2.0 * half_dt_s),
        (after_pos[1] - before_pos[1]) / (2.0 * half_dt_s),
        (after_pos[2] - before_pos[2]) / (2.0 * half_dt_s),
    ];

    let clock_drift_s_s = 2.0e-9;
    let doppler = doppler_observations(
        &source,
        receiver0_m,
        receiver_velocity_m_s,
        t0_s,
        clock_drift_s_s,
    );
    let velocity = solve_doppler_velocity(
        &source,
        &DopplerVelocityInputs {
            observations: doppler.clone(),
            receiver_ecef_m: receiver0_m,
            t_rx_j2000_s: t0_s,
            light_time: true,
            sagnac: true,
        },
    )
    .expect("solve Doppler velocity");

    let error_m_s = norm([
        velocity.velocity_m_s[0] - finite_difference_m_s[0],
        velocity.velocity_m_s[1] - finite_difference_m_s[1],
        velocity.velocity_m_s[2] - finite_difference_m_s[2],
    ]);
    assert!(error_m_s < 0.10, "velocity error {error_m_s}");
    assert!((velocity.clock_drift_s_s - clock_drift_s_s).abs() < 1.0e-12);
    assert!(velocity
        .state_covariance
        .iter()
        .flatten()
        .all(|value| value.is_finite()));
    for idx in 0..4 {
        assert!(velocity.state_covariance[idx][idx] > 0.0);
    }
}

#[test]
fn spp_wrapper_uses_doppler_parsed_from_rinex_obs_epoch() {
    let (source, receiver0_m, receiver_velocity_m_s, t0_s) = synthetic_case();
    let clock_drift_s_s = -1.5e-9;
    let rinex_text = synthetic_rinex_obs_epoch(
        &source,
        receiver0_m,
        receiver_velocity_m_s,
        t0_s,
        clock_drift_s_s,
    );
    let obs = RinexObs::parse(&rinex_text).expect("strict parse synthetic RINEX OBS");
    assert_eq!(
        obs.obs_codes(GnssSystem::Gps).expect("GPS code table"),
        ["C1C".to_string(), "D1C".to_string()]
    );
    let (ranges, doppler) = parsed_l1_rows(&obs);
    let mut inputs = pseudorange_inputs(&source, receiver0_m, receiver_velocity_m_s, t0_s, t0_s);
    inputs.observations = ranges;

    let fused = solve_with_doppler_velocity(&source, &inputs, &doppler, false)
        .expect("solve RINEX parsed Doppler epoch");
    let velocity = fused.velocity.expect("Doppler velocity solution");
    assert_eq!(fused.velocity_error, None);
    assert_eq!(
        fused.receiver.rx_clock_drift_s_s,
        Some(velocity.clock_drift_s_s)
    );

    let error_m_s = norm([
        velocity.velocity_m_s[0] - receiver_velocity_m_s[0],
        velocity.velocity_m_s[1] - receiver_velocity_m_s[1],
        velocity.velocity_m_s[2] - receiver_velocity_m_s[2],
    ]);
    assert!(error_m_s < 0.02, "RINEX parsed velocity error {error_m_s}");
    assert!((velocity.clock_drift_s_s - clock_drift_s_s).abs() < 2.0e-12);
}

#[test]
fn spp_wrapper_solves_real_rinex_obs_nav_doppler_against_position_difference() {
    let nav_text = fixture_text(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"]);
    let store = BroadcastEphemeris::from_nav(&nav_text).expect("parse ESBC broadcast NAV");
    let obs_text = fixture_text(&["obs", "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"]);
    let obs = RinexObs::parse(&obs_text).expect("parse ESBC OBS");
    assert!(
        obs.epochs().len() >= 2,
        "ESBC trim must carry two epochs for the finite difference"
    );
    let first = &obs.epochs()[0];
    let second = &obs.epochs()[1];
    let first_inputs = esbc_solve_inputs(&obs, first);
    let second_inputs = esbc_solve_inputs(&obs, second);
    let (_, doppler) = esbc_epoch_rows(&obs, first);
    assert!(
        doppler.len() >= 4,
        "need at least four real D1C rows, got {}",
        doppler.len()
    );

    let first_position = solve(&store, &first_inputs, false).expect("solve first SPP epoch");
    let second_position = solve(&store, &second_inputs, false).expect("solve second SPP epoch");
    let dt_s = second_inputs.t_rx_j2000_s - first_inputs.t_rx_j2000_s;
    assert!(dt_s > 0.0, "epochs must be increasing");
    let p0 = first_position.position.as_array();
    let p1 = second_position.position.as_array();
    let finite_difference_m_s = [
        (p1[0] - p0[0]) / dt_s,
        (p1[1] - p0[1]) / dt_s,
        (p1[2] - p0[2]) / dt_s,
    ];

    let fused = solve_with_doppler_velocity(&store, &first_inputs, &doppler, false)
        .expect("solve public SPP wrapper from real OBS plus NAV");
    let velocity = fused.velocity.expect("real Doppler velocity solution");
    assert_eq!(fused.velocity_error, None);
    assert_eq!(
        fused.receiver.rx_clock_drift_s_s,
        Some(velocity.clock_drift_s_s)
    );
    let error_m_s = norm([
        velocity.velocity_m_s[0] - finite_difference_m_s[0],
        velocity.velocity_m_s[1] - finite_difference_m_s[1],
        velocity.velocity_m_s[2] - finite_difference_m_s[2],
    ]);
    assert!(
        error_m_s < 0.75,
        "real OBS/NAV wrapper velocity differs from SPP position difference by {error_m_s} m/s"
    );
}

#[test]
fn receiver_solution_carries_optional_clock_drift_from_doppler_rows() {
    let (source, receiver0_m, receiver_velocity_m_s, t0_s) = synthetic_case();
    let inputs = pseudorange_inputs(&source, receiver0_m, receiver_velocity_m_s, t0_s, t0_s);
    let pseudorange_only = solve(&source, &inputs, false).expect("solve SPP");
    assert_eq!(pseudorange_only.rx_clock_drift_s_s, None);

    let drift_s_s = 2.0e-9;
    let doppler =
        doppler_observations(&source, receiver0_m, receiver_velocity_m_s, t0_s, drift_s_s);
    let fused =
        solve_with_doppler_velocity(&source, &inputs, &doppler, false).expect("solve fused epoch");
    let velocity = fused.velocity.expect("Doppler velocity solution");
    assert_eq!(fused.velocity_error, None);
    assert_eq!(
        fused.receiver.rx_clock_drift_s_s,
        Some(velocity.clock_drift_s_s)
    );
}

#[test]
fn receiver_solution_position_covariance_exposes_ecef_and_consistent_enu() {
    let (source, receiver0_m, receiver_velocity_m_s, t0_s) = synthetic_case();
    let inputs = pseudorange_inputs(&source, receiver0_m, receiver_velocity_m_s, t0_s, t0_s);
    let solution = solve(&source, &inputs, true).expect("solve SPP");
    let geodetic = solution.geodetic.expect("geodetic solution");
    let rotated = sidereon_core::dop::rotate_covariance_ecef_to_enu_m2(
        solution.position_covariance.ecef_m2,
        geodetic,
    )
    .expect("rotate covariance");
    for (row, (got_row, expected_row)) in solution
        .position_covariance
        .enu_m2
        .iter()
        .zip(rotated.iter())
        .enumerate()
    {
        for (col, (&got, &expected)) in got_row.iter().zip(expected_row.iter()).enumerate() {
            assert!(
                (got - expected).abs() <= 1.0e-9,
                "covariance[{row}][{col}] got {got} expected {expected}"
            );
        }
    }
}

fn obs_header_line(body: &str, label: &str) -> String {
    format!("{body:<60}{label}\n")
}

fn minimal_obs_text(version_line: &str, obs_types_line: &str, sat: &str) -> String {
    let mut text = String::new();
    text.push_str(version_line);
    text.push('\n');
    text.push_str(&obs_header_line("TEST", "MARKER NAME"));
    text.push_str(&obs_header_line(
        "PGM                 RUN                 DATE",
        "PGM / RUN BY / DATE",
    ));
    text.push_str(&obs_header_line(
        "OBS                 AGENCY",
        "OBSERVER / AGENCY",
    ));
    text.push_str(&obs_header_line(
        "REC                 TYPE                VERS",
        "REC # / TYPE / VERS",
    ));
    text.push_str(&obs_header_line("ANT                 TYPE", "ANT # / TYPE"));
    text.push_str(&obs_header_line(
        "  6378137.0000        0.0000        0.0000",
        "APPROX POSITION XYZ",
    ));
    text.push_str(&obs_header_line(
        "        0.0000        0.0000        0.0000",
        "ANTENNA: DELTA H/E/N",
    ));
    text.push_str(obs_types_line);
    text.push('\n');
    text.push_str(&obs_header_line(
        "  2020     1     1     0     0    0.0000000     GPS",
        "TIME OF FIRST OBS",
    ));
    text.push_str(&obs_header_line("", "END OF HEADER"));
    text.push_str("> 2020 01 01 00 00  0.0000000  0  1\n");
    text.push_str(&format!("{sat:<3}{:14.3}\n", 20_200_000.0));
    text
}

fn extended_count_obs_text() -> String {
    let codes = [
        "C1C", "L1C", "D1C", "S1C", "C2W", "L2W", "D2W", "S2W", "C5Q", "L5Q", "D5Q", "S5Q", "C1W",
    ];
    let mut text = String::new();
    text.push_str(
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n",
    );
    text.push_str(&obs_header_line("TEST", "MARKER NAME"));
    text.push_str(&obs_header_line(
        "PGM                 RUN                 DATE",
        "PGM / RUN BY / DATE",
    ));
    text.push_str(&obs_header_line(
        "OBS                 AGENCY",
        "OBSERVER / AGENCY",
    ));
    text.push_str(&obs_header_line(
        "REC                 TYPE                VERS",
        "REC # / TYPE / VERS",
    ));
    text.push_str(&obs_header_line("ANT                 TYPE", "ANT # / TYPE"));
    text.push_str(&obs_header_line(
        "  6378137.0000        0.0000        0.0000",
        "APPROX POSITION XYZ",
    ));
    text.push_str(&obs_header_line(
        "        0.0000        0.0000        0.0000",
        "ANTENNA: DELTA H/E/N",
    ));
    let mut obs_types = format!("G  {:>3}", codes.len());
    for code in &codes {
        obs_types.push_str(&format!(" {code:>3}"));
    }
    text.push_str(&obs_header_line(&obs_types, "SYS / # / OBS TYPES"));

    let mut scale_first = format!("G {:>4}  {:>2}", 10, codes.len());
    for code in &codes[..12] {
        scale_first.push_str(&format!(" {code:>3}"));
    }
    let mut scale_second = " ".repeat(10);
    for code in &codes[12..] {
        scale_second.push_str(&format!(" {code:>3}"));
    }
    text.push_str(&obs_header_line(&scale_first, "SYS / SCALE FACTOR"));
    text.push_str(&obs_header_line(&scale_second, "SYS / SCALE FACTOR"));

    // `3X,A1,I2,9I6`, continued as `6X,9I6`.
    let mut prn_first = "   G01".to_string();
    for value in 1..=9 {
        prn_first.push_str(&format!("{value:6}"));
    }
    let mut prn_second = " ".repeat(6);
    for value in 10..=13 {
        prn_second.push_str(&format!("{value:6}"));
    }
    text.push_str(&obs_header_line(&prn_first, "PRN / # OF OBS"));
    text.push_str(&obs_header_line(&prn_second, "PRN / # OF OBS"));
    text.push_str(&obs_header_line(
        "  2020     1     1     0     0    0.0000000     GPS",
        "TIME OF FIRST OBS",
    ));
    text.push_str(&obs_header_line("", "END OF HEADER"));
    text.push_str("> 2020 01 01 00 00  0.0000000  0  1\n");
    text.push_str(&format!("G01{:14.3}\n", 20_200_000.0));
    text
}

fn assert_strict_header_columns(text: &str) {
    let labels = [
        "RINEX VERSION / TYPE",
        "PGM / RUN BY / DATE",
        "MARKER NAME",
        "SYS / # / OBS TYPES",
        "END OF HEADER",
    ];
    for line in text.lines() {
        if line.contains("END OF HEADER") {
            assert_eq!(line.get(60..).unwrap_or("").trim(), "END OF HEADER");
            break;
        }
        if labels.iter().any(|label| line.contains(label)) {
            assert!(line.len() >= 60, "{line:?}");
            let label = line.get(60..).unwrap_or("").trim();
            assert!(labels.contains(&label), "{line:?}");
        }
    }
}

#[test]
fn repair_obs_text_outputs_strict_parseable_headers_for_trial_whitespace_cases() {
    let cases = [
        minimal_obs_text(
            "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE",
            "G    1 C1C                                                  SYS / # / OBS TYPES",
            "G01",
        ),
        minimal_obs_text(
            "3.04 OBSERVATION DATA M RINEX VERSION / TYPE",
            "G 1 C1C SYS / # / OBS TYPES",
            "G 1",
        ),
        minimal_obs_text(
            "     3.04           OBSERVATION DATA    M                      RINEX VERSION / TYPE",
            "G    1 C1C                                                     SYS / # / OBS TYPES",
            "G01",
        ),
    ];

    for case in cases {
        let repair = repair_obs_text(&case, &RepairOptions::default()).expect("repair OBS text");
        let output = repair.repaired.to_rinex_string();
        assert_strict_header_columns(&output);
        RinexObs::parse(&output).expect("repaired output strict parses");
    }
}

#[test]
fn obs_writer_wraps_scale_factors_and_prn_counts_without_truncation() {
    let obs = RinexObs::parse(&extended_count_obs_text()).expect("parse extended count headers");
    assert_eq!(obs.header().scale_factors[0].codes.len(), 13);
    assert_eq!(
        obs.header()
            .prn_obs_counts
            .get(&gps(1))
            .expect("G01 counts")
            .len(),
        13
    );

    let output = obs.to_rinex_string();
    assert_eq!(
        output
            .lines()
            .filter(|line| line.get(60..).unwrap_or("").trim() == "SYS / SCALE FACTOR")
            .count(),
        2
    );
    assert_eq!(
        output
            .lines()
            .filter(|line| line.get(60..).unwrap_or("").trim() == "PRN / # OF OBS")
            .count(),
        2
    );
    let reparsed = RinexObs::parse(&output).expect("reparse wrapped headers");
    assert_eq!(reparsed.header().scale_factors[0].codes.len(), 13);
    assert_eq!(
        reparsed
            .header()
            .prn_obs_counts
            .get(&gps(1))
            .expect("G01 counts")
            .len(),
        13
    );
}

#[test]
fn obs_type_whitespace_fallback_handles_continuation_and_rejects_surplus() {
    let compact_continuation = minimal_obs_text(
        "3.04 OBSERVATION DATA M RINEX VERSION / TYPE",
        "G 3 C1C SYS / # / OBS TYPES\nL1C D1C SYS / # / OBS TYPES",
        "G01",
    );
    let obs = RinexObs::parse(&compact_continuation).expect("parse compact continuation");
    assert_eq!(
        obs.obs_codes(GnssSystem::Gps).expect("GPS codes"),
        ["C1C".to_string(), "L1C".to_string(), "D1C".to_string()]
    );

    let strict_surplus_line = obs_header_line("G    1 C1C D1C", "SYS / # / OBS TYPES");
    let strict_surplus = minimal_obs_text(
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE",
        strict_surplus_line.trim_end(),
        "G01",
    );
    let strict_error = RinexObs::parse(&strict_surplus).expect_err("strict surplus must fail");
    let compact_surplus = minimal_obs_text(
        "3.04 OBSERVATION DATA M RINEX VERSION / TYPE",
        "G 1 C1C D1C SYS / # / OBS TYPES",
        "G01",
    );
    let compact_error = RinexObs::parse(&compact_surplus).expect_err("compact surplus must fail");
    match (strict_error, compact_error) {
        (Error::Parse(strict), Error::Parse(compact)) => {
            assert!(strict.contains("lists more codes than declared"));
            assert!(compact.contains("lists more codes than declared"));
        }
        (strict, compact) => panic!("expected parse errors, got {strict:?} and {compact:?}"),
    }
}

#[test]
fn non_padded_satellite_ids_parse_equivalently_in_obs_and_nav() {
    let obs_padded = minimal_obs_text(
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE",
        "G    1 C1C                                                  SYS / # / OBS TYPES",
        "G01",
    );
    let obs_non_padded = minimal_obs_text(
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE",
        "G    1 C1C                                                  SYS / # / OBS TYPES",
        "G 1",
    );
    let padded = RinexObs::parse(&obs_padded).expect("parse padded OBS");
    let non_padded = RinexObs::parse(&obs_non_padded).expect("parse non-padded OBS");
    let padded_sat = padded.epochs()[0].sats.keys().next().copied();
    let non_padded_sat = non_padded.epochs()[0].sats.keys().next().copied();
    assert_eq!(padded_sat, Some(gps(1)));
    assert_eq!(non_padded_sat, padded_sat);

    let nav_padded = include_str!("fixtures/nav/BRD400DLR_S_20261800000_01H_MN_trim.rnx");
    let nav_non_padded = nav_padded.replacen("G01 ", "G 1 ", 1);
    let padded_nav = parse_nav(nav_padded).expect("parse padded NAV");
    let non_padded_nav = parse_nav(&nav_non_padded).expect("parse non-padded NAV");
    assert_eq!(padded_nav[0].satellite_id, gps(1));
    assert_eq!(non_padded_nav[0].satellite_id, padded_nav[0].satellite_id);
}
