#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::constants::C_M_S;
use sidereon_core::dgnss::{
    apply_corrections, pseudorange_corrections, solve_position, CodeObservation, DgnssError,
};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::observables::{
    predict, ObservableEphemerisSource, ObservableState, ObservablesError, PredictOptions,
};
use sidereon_core::positioning::{
    solve, Corrections, KlobucharCoeffs, Observation, SolveInputs, SurfaceMet,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const GOLDEN: &str = include_str!("fixtures/orbis_gnss_application_golden.json");
const T_RX_J2000_S: f64 = 646_272_000.0;

struct ClocklessSatSource<'a> {
    inner: &'a Sp3,
    clockless: GnssSatelliteId,
}

impl ObservableEphemerisSource for ClocklessSatSource<'_> {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let mut state = self.inner.observable_state_at_j2000_s(sat, t_j2000_s)?;
        if sat == self.clockless {
            state.clock_s = None;
        }
        Ok(state)
    }
}

fn parse_hex_float(s: &str) -> f64 {
    let (sign, body) = if let Some(rest) = s.strip_prefix('-') {
        (-1.0, rest)
    } else {
        (1.0, s)
    };
    let body = body
        .strip_prefix("0x")
        .unwrap_or_else(|| panic!("not a hex float (missing 0x): {s:?}"));
    let (mantissa, exponent) = body
        .split_once('p')
        .unwrap_or_else(|| panic!("not a hex float (missing p exponent): {s:?}"));
    let exponent: i32 = exponent
        .parse()
        .unwrap_or_else(|_| panic!("bad hex exponent in {s:?}"));
    let (whole, frac) = mantissa.split_once('.').unwrap_or((mantissa, ""));
    let mut value = u64::from_str_radix(whole, 16)
        .unwrap_or_else(|_| panic!("bad integer hex digits in {s:?}")) as f64;
    let mut scale = 1.0 / 16.0;
    for c in frac.chars() {
        let digit = c
            .to_digit(16)
            .unwrap_or_else(|| panic!("bad hex frac digit {c:?} in {s:?}"));
        value += digit as f64 * scale;
        scale /= 16.0;
    }
    sign * value * 2.0_f64.powi(exponent)
}

fn hexf(v: &Value) -> f64 {
    parse_hex_float(v.as_str().expect("hex float string"))
}

fn vec3(value: &Value) -> [f64; 3] {
    [hexf(&value[0]), hexf(&value[1]), hexf(&value[2])]
}

fn observations(value: &Value) -> Vec<CodeObservation> {
    value
        .as_array()
        .expect("observation array")
        .iter()
        .map(|row| CodeObservation::new(row["sat"].as_str().unwrap(), hexf(&row["pseudorange_m"])))
        .collect()
}

fn sp3_fixture() -> Sp3 {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    );
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
    Sp3::parse(&bytes).expect("parse SP3 fixture")
}

fn application_dgnss_golden() -> Value {
    let doc: Value = serde_json::from_str(GOLDEN).expect("parse application golden");
    doc["sp3_application"]["dgnss"].clone()
}

fn solve_inputs(observations: Vec<Observation>, initial_guess: [f64; 4]) -> SolveInputs {
    SolveInputs {
        observations,
        t_rx_j2000_s: T_RX_J2000_S,
        t_rx_second_of_day_s: 43_200.0,
        day_of_year: 176.5,
        initial_guess,
        corrections: Corrections::NONE,
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met: SurfaceMet {
            pressure_hpa: 1013.25,
            temperature_k: 288.15,
            relative_humidity: 0.5,
        },
        robust: None,
    }
}

fn sat_from_token(token: &str) -> GnssSatelliteId {
    let letter = token.chars().next().unwrap();
    let system = GnssSystem::from_letter(letter).unwrap();
    let prn = token[letter.len_utf8()..].parse::<u8>().unwrap();
    GnssSatelliteId::new(system, prn).expect("valid satellite id")
}

fn spp_observations(obs: &[CodeObservation]) -> Vec<Observation> {
    obs.iter()
        .map(|o| Observation {
            satellite_id: sat_from_token(&o.satellite_id),
            pseudorange_m: o.pseudorange_m,
        })
        .collect()
}

fn assert_dgnss_invalid_input(err: DgnssError, field: &'static str, reason: &'static str) {
    match err {
        DgnssError::InvalidInput {
            field: got_field,
            reason: got_reason,
        } => {
            assert_eq!(got_field, field);
            assert_eq!(got_reason, reason);
        }
        other => panic!("expected DGNSS invalid input, got {other:?}"),
    }
}

fn visible_gps(sp3: &Sp3, station: [f64; 3]) -> Vec<GnssSatelliteId> {
    sp3.satellites()
        .iter()
        .copied()
        .filter(|sat| sat.system == GnssSystem::Gps)
        .filter(|sat| {
            predict(sp3, *sat, station, T_RX_J2000_S, PredictOptions::default())
                .map(|obs| obs.elevation_deg >= 10.0)
                .unwrap_or(false)
        })
        .collect()
}

fn synth(
    sp3: &Sp3,
    sats: &[GnssSatelliteId],
    station: [f64; 3],
    rx_clock_s: f64,
) -> Vec<CodeObservation> {
    sats.iter()
        .map(|sat| {
            let obs = predict(sp3, *sat, station, T_RX_J2000_S, PredictOptions::default())
                .expect("predict visible satellite");
            CodeObservation::new(
                sat.to_string(),
                obs.geometric_range_m
                    + C_M_S * (rx_clock_s - obs.sat_clock_s.expect("visible satellite clock")),
            )
        })
        .collect()
}

fn inject(obs: &[CodeObservation], errors: &[f64]) -> Vec<CodeObservation> {
    obs.iter()
        .zip(errors.iter())
        .map(|(obs, error)| {
            CodeObservation::new(obs.satellite_id.clone(), obs.pseudorange_m + error)
        })
        .collect()
}

fn dist(position: [f64; 3], truth: [f64; 3]) -> f64 {
    ((position[0] - truth[0]).powi(2)
        + (position[1] - truth[1]).powi(2)
        + (position[2] - truth[2]).powi(2))
    .sqrt()
}

#[test]
fn dgnss_corrections_and_apply_match_application_oracle_bits() {
    let sp3 = sp3_fixture();
    let golden = application_dgnss_golden();
    let base = vec3(&golden["base_ecef_m"]);
    let base_obs = observations(&golden["base_observations"]);
    let rover_obs = observations(&golden["rover_observations"]);

    let corrections = pseudorange_corrections(&sp3, base, &base_obs, T_RX_J2000_S)
        .expect("compute DGNSS corrections");
    for (sat, expected) in golden["corrections_m"].as_object().unwrap() {
        let got = corrections
            .get(sat)
            .unwrap_or_else(|| panic!("missing correction {sat}"));
        assert_eq!(got.to_bits(), hexf(expected).to_bits(), "{sat} correction");
    }

    let applied = apply_corrections(&rover_obs, &corrections).expect("apply DGNSS corrections");
    assert!(applied.dropped.is_empty());

    let expected = golden["corrected_rover"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| (row["sat"].as_str().unwrap(), hexf(&row["pseudorange_m"])))
        .collect::<std::collections::BTreeMap<_, _>>();
    for obs in applied.corrected {
        assert_eq!(
            obs.pseudorange_m.to_bits(),
            expected[obs.satellite_id.as_str()].to_bits(),
            "{} corrected rover pseudorange",
            obs.satellite_id
        );
    }
}

#[test]
fn dgnss_corrections_exclude_satellite_with_missing_clock() {
    let sp3 = sp3_fixture();
    let golden = application_dgnss_golden();
    let base = vec3(&golden["base_ecef_m"]);
    let base_obs = observations(&golden["base_observations"]);
    let rover_obs = observations(&golden["rover_observations"]);
    let clockless_token = base_obs[0].satellite_id.clone();
    let clockless = sat_from_token(&clockless_token);
    let source = ClocklessSatSource {
        inner: &sp3,
        clockless,
    };

    let corrections = pseudorange_corrections(&source, base, &base_obs, T_RX_J2000_S)
        .expect("compute DGNSS corrections");

    assert!(!corrections.contains_key(&clockless_token));
    assert_eq!(corrections.len(), base_obs.len() - 1);
    let applied = apply_corrections(&rover_obs, &corrections).expect("apply DGNSS corrections");
    assert_eq!(applied.dropped, vec![clockless_token]);
}

#[test]
fn dgnss_helpers_reject_invalid_corrections_and_observations() {
    let sp3 = sp3_fixture();
    let golden = application_dgnss_golden();
    let base = vec3(&golden["base_ecef_m"]);
    let mut base_obs = observations(&golden["base_observations"]);
    base_obs[0].pseudorange_m = f64::NAN;

    let err = pseudorange_corrections(&sp3, base, &base_obs, T_RX_J2000_S)
        .expect_err("non-finite base observation should be rejected");
    assert_dgnss_invalid_input(err, "base_observation.pseudorange_m", "not finite");

    let mut corrections = std::collections::BTreeMap::new();
    corrections.insert("G01".to_string(), f64::INFINITY);
    let err = apply_corrections(&[CodeObservation::new("G01", 20_000_000.0)], &corrections)
        .expect_err("non-finite correction should be rejected");
    assert_dgnss_invalid_input(err, "pseudorange_correction_m", "not finite");

    let mut corrections = std::collections::BTreeMap::new();
    corrections.insert("G01".to_string(), 1.0);
    let err = apply_corrections(&[CodeObservation::new("G01", f64::NAN)], &corrections)
        .expect_err("non-finite rover observation should be rejected");
    assert_dgnss_invalid_input(err, "rover_observation.pseudorange_m", "not finite");

    let mut corrections = std::collections::BTreeMap::new();
    corrections.insert("G01".to_string(), -f64::MAX);
    let err = apply_corrections(&[CodeObservation::new("G01", f64::MAX)], &corrections)
        .expect_err("overflowed corrected pseudorange should be rejected");
    assert_dgnss_invalid_input(err, "corrected_pseudorange_m", "not finite");
}

#[test]
fn dgnss_common_mode_error_cancels_in_position_solve() {
    let sp3 = sp3_fixture();
    let base = [3_512_900.0, 780_500.0, 5_248_700.0];
    let rover = [base[0] + 2_000.0, base[1] + 1_000.0, base[2] + 1_500.0];

    let base_visible = visible_gps(&sp3, base);
    let rover_visible = visible_gps(&sp3, rover);
    let mut sats: Vec<GnssSatelliteId> = base_visible
        .into_iter()
        .filter(|sat| rover_visible.contains(sat))
        .collect();
    sats.sort_unstable();
    assert!(sats.len() >= 5);

    let base_clean = synth(&sp3, &sats, base, 1.0e-6);
    let rover_clean = synth(&sp3, &sats, rover, -2.0e-6);
    let errors: Vec<f64> = sats
        .iter()
        .enumerate()
        .map(|(idx, _)| (idx as f64 - 3.0) * 9.75)
        .collect();
    let base_obs = inject(&base_clean, &errors);
    let rover_obs = inject(&rover_clean, &errors);

    let absolute = solve(
        &sp3,
        &solve_inputs(
            spp_observations(&rover_obs),
            [rover[0], rover[1], rover[2], 0.0],
        ),
        false,
    )
    .expect("absolute SPP solve");
    let absolute_error = dist(absolute.position.as_array(), rover);

    let dgnss = solve_position(
        &sp3,
        base,
        &base_obs,
        &rover_obs,
        solve_inputs(Vec::new(), [rover[0], rover[1], rover[2], 0.0]),
        false,
    )
    .expect("DGNSS solve");
    let dgnss_error = dist(dgnss.solution.position.as_array(), rover);

    let clean = solve_position(
        &sp3,
        base,
        &base_clean,
        &rover_clean,
        solve_inputs(Vec::new(), [rover[0], rover[1], rover[2], 0.0]),
        false,
    )
    .expect("clean DGNSS solve");
    let clean_error = dist(clean.solution.position.as_array(), rover);

    assert_eq!(
        clean.solution.position.as_array().map(f64::to_bits),
        [0x414ad10a000812ad, 0x4127d9780005c95e, 0x41540725fffa9093],
        "clean DGNSS position frozen bits"
    );
    assert_eq!(
        clean.solution.rx_clock_s.to_bits(),
        0xbec92a737b8703b3,
        "clean DGNSS receiver clock frozen bits"
    );
    assert_eq!(
        clean.baseline_vector_m.map(f64::to_bits),
        [0x409f400040956800, 0x408f400017257800, 0x40976fffa9093000],
        "clean DGNSS baseline vector frozen bits"
    );
    assert_eq!(
        clean.baseline_m.to_bits(),
        0x40a5092a32b6435d,
        "clean DGNSS baseline length frozen bits"
    );
    assert_eq!(
        clean
            .solution
            .residuals_m
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![
            0xbf2f8d6000000000,
            0xbef1420000000000,
            0xbedb2c0000000000,
            0xbf208bc000000000,
            0xbea0000000000000,
            0x3ef3ee0000000000,
            0x3f10330000000000,
            0xbf0c2e8000000000,
            0xbf01a68000000000,
            0x3f12edc000000000,
        ],
        "clean DGNSS residual frozen bits"
    );

    assert!(absolute_error > 5.0);
    assert!((dgnss_error - clean_error).abs() <= 1.0e-3);
    assert!(dgnss_error < absolute_error / 100.0);
    assert_eq!(dgnss.dropped_sats, Vec::<String>::new());
    assert!((clean.baseline_m - dist(base, rover)).abs() <= 1.0e-2);
}
