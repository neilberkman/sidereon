#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::constants::C_M_S;
use sidereon_core::dgnss::{
    apply_corrections, pseudorange_corrections, solve_position, CodeObservation, DgnssError,
};
use sidereon_core::ephemeris::{BroadcastEphemeris, Sp3};
use sidereon_core::observables::{
    predict, pseudorange_transmit_epoch_j2000_s, pseudorange_transmit_geometry,
    rounded_microsecond_replay, ObservableEphemerisSource, ObservableState, ObservablesError,
    PredictOptions,
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
        sbas_iono: None,
        glonass_channels: std::collections::BTreeMap::new(),
        met: SurfaceMet {
            pressure_hpa: 1013.25,
            temperature_k: 288.15,
            relative_humidity: 0.5,
        },
        robust: None,
        pseudorange_code: sidereon_core::positioning::PseudorangeCode::SingleFrequency,
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

/// The base model of the DGNSS corrections for pseudorange `pseudorange_m`, formed as
/// `pseudorange_corrections` forms it: the transmission epoch placed from the pseudorange
/// as RTKLIB `satposs` places it, the `geodist` range there, and `c` times the satellite
/// clock with the `peph2pos` term and less the single-frequency group delay, all of the
/// record selected at the reception epoch.
fn placed_base_model_m(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    station: [f64; 3],
    t_rx_j2000_s: f64,
    pseudorange_m: f64,
) -> f64 {
    let t_tx = pseudorange_transmit_epoch_j2000_s(source, sat, t_rx_j2000_s, pseudorange_m)
        .expect("placed transmission epoch");
    let geometry = pseudorange_transmit_geometry(source, sat, station, t_rx_j2000_s, t_tx, true)
        .expect("placed geometry");
    let sat_clock_s = geometry.sat_clock_s.expect("satellite clock");
    let sat_clock_s = match source.clock_relativity_s(sat, t_tx) {
        sidereon_core::positioning::ClockRelativity::NotApplicable => sat_clock_s,
        sidereon_core::positioning::ClockRelativity::Term(relativity_s) => {
            sat_clock_s + relativity_s
        }
        sidereon_core::positioning::ClockRelativity::Unavailable => {
            panic!("{sat}: no peph2pos term")
        }
    };
    let group_delay = source
        .try_observable_state_group_delay_selected_at_j2000_s(sat, t_tx, t_rx_j2000_s)
        .expect("placed state")
        .value
        .1;
    let sat_clock_s = match group_delay {
        Some(group_delay_s) => sat_clock_s - group_delay_s,
        None => sat_clock_s,
    };
    geometry.geometric_range_m - C_M_S * sat_clock_s
}

/// A pseudorange the positioning models reproduce: the fixed point of
/// `P = model(P) + c · rx_clock_s + extra_m`, with the model placing the transmission
/// epoch from `P` itself ([`placed_base_model_m`]). Each step moves the epoch by the
/// change in `P / c`, which moves the range by `rdot / c` of that change, so four steps
/// from the geometric prediction settle it.
fn synth_placed(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    station: [f64; 3],
    t_rx_j2000_s: f64,
    rx_clock_s: f64,
    extra_m: f64,
) -> f64 {
    let seed = predict(
        source,
        sat,
        station,
        t_rx_j2000_s,
        PredictOptions::default(),
    )
    .expect("predict visible satellite");
    let mut pseudorange_m = seed.geometric_range_m + C_M_S * rx_clock_s + extra_m;
    for _ in 0..4 {
        pseudorange_m = placed_base_model_m(source, sat, station, t_rx_j2000_s, pseudorange_m)
            + C_M_S * rx_clock_s
            + extra_m;
    }
    pseudorange_m
}

fn synth(
    sp3: &Sp3,
    sats: &[GnssSatelliteId],
    station: [f64; 3],
    rx_clock_s: f64,
) -> Vec<CodeObservation> {
    // The synthetic pseudorange carries the satellite clock the positioning models use,
    // the SP3 clock with the relativistic term RTKLIB `peph2pos` applies, and places its
    // transmission epoch as they place it.
    sats.iter()
        .map(|sat| {
            CodeObservation::new(
                sat.to_string(),
                synth_placed(sp3, *sat, station, T_RX_J2000_S, rx_clock_s, 0.0),
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
    // The oracle's base model uses the SP3 clock as written, with no relativistic term,
    // and the geometric light time from the reception epoch with the transmission epoch
    // rounded to whole microseconds. The positioning models apply the term RTKLIB
    // `peph2pos` applies to a precise clock (see
    // `zim2_sp3_spp_with_the_peph2pos_relativity_term_is_closer_to_truth`), and place the
    // transmission epoch from the base pseudorange as RTKLIB `satposs` places it, with
    // the `geodist` range there. The oracle still certifies the rest of the model bit for
    // bit: its correction is `P - (rho - c·clk)` from this crate's prediction replayed
    // with the rounded epoch. Ours is `P - (rho' - c·(clk' + term'))` at the placed epoch,
    // and differs from the oracle's with the term by the range rate times the difference
    // of the two epochs, within 0.5 mm: for a GPS orbit RTKLIB's first-order Sagnac term
    // and the closed-form rotation differ by about 0.1 mm.
    let base_pseudorange = base_obs
        .iter()
        .map(|o| (o.satellite_id.clone(), o.pseudorange_m))
        .collect::<std::collections::BTreeMap<_, _>>();
    let rover_pseudorange = rover_obs
        .iter()
        .map(|o| (o.satellite_id.clone(), o.pseudorange_m))
        .collect::<std::collections::BTreeMap<_, _>>();
    let expected_rover = golden["corrected_rover"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            (
                row["sat"].as_str().unwrap().to_string(),
                hexf(&row["pseudorange_m"]),
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for (token, expected) in golden["corrections_m"].as_object().unwrap() {
        let sat = sat_from_token(token);
        let prediction = rounded_microsecond_replay::predict(
            &sp3,
            sat,
            base,
            T_RX_J2000_S,
            PredictOptions::default(),
        )
        .expect("predict base satellite");
        let clock_s = prediction.sat_clock_s.expect("SP3 clock");
        let relativity_s = ObservableEphemerisSource::clock_relativity_s(
            &sp3,
            sat,
            prediction.transmit_time_j2000_s,
        )
        .term()
        .expect("peph2pos relativistic term");
        let pseudorange_m = base_pseudorange[token];
        let oracle_model_m = pseudorange_m - (prediction.geometric_range_m - C_M_S * clock_s);
        assert_eq!(
            oracle_model_m.to_bits(),
            hexf(expected).to_bits(),
            "{token}: oracle correction"
        );
        let got = corrections
            .get(token)
            .unwrap_or_else(|| panic!("missing correction {token}"));
        let placed_m =
            pseudorange_m - placed_base_model_m(&sp3, sat, base, T_RX_J2000_S, pseudorange_m);
        assert_eq!(got.to_bits(), placed_m.to_bits(), "{token}: correction");
        let with_term_m =
            pseudorange_m - (prediction.geometric_range_m - C_M_S * (clock_s + relativity_s));
        let placed_epoch_s =
            pseudorange_transmit_epoch_j2000_s(&sp3, sat, T_RX_J2000_S, pseudorange_m)
                .expect("placed transmission epoch");
        let epoch_shift_s = placed_epoch_s - prediction.transmit_time_j2000_s;
        let moved_m = (got - with_term_m) + prediction.range_rate_m_s * epoch_shift_s;
        assert!(
            moved_m.abs() <= 5.0e-4,
            "{token}: the correction moved {} m where the epoch shift of {epoch_shift_s} s \
             moves the range by {} m",
            got - with_term_m,
            prediction.range_rate_m_s * epoch_shift_s
        );
        assert_eq!(
            (rover_pseudorange[token] - hexf(expected)).to_bits(),
            expected_rover[token.as_str()].to_bits(),
            "{token}: oracle corrected rover pseudorange"
        );
    }

    let applied = apply_corrections(&rover_obs, &corrections).expect("apply DGNSS corrections");
    assert!(applied.dropped.is_empty());
    for obs in applied.corrected {
        assert_eq!(
            obs.pseudorange_m.to_bits(),
            (rover_pseudorange[&obs.satellite_id] - corrections[&obs.satellite_id]).to_bits(),
            "{} corrected rover pseudorange",
            obs.satellite_id
        );
    }
}

/// On a broadcast source the DGNSS corrections are what they were before the broadcast
/// clock left the TGD out: the base model subtracts the source's single-frequency group
/// delay from the `satposs` clock, which is the former TGD-inclusive clock bit for bit.
/// Each correction therefore recovers the error injected into the base pseudorange, and
/// the TGD-free clock alone would miss it by `c·TGD`.
#[test]
fn dgnss_corrections_on_a_broadcast_source_keep_the_group_delay() {
    let nav = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
    ))
    .expect("read NAV fixture");
    let store = BroadcastEphemeris::from_nav(&nav).expect("parse NAV fixture");
    let t_rx = 646_358_400.0;
    let base = [3_512_900.0, 780_500.0, 5_248_700.0];
    let mut base_obs = Vec::new();
    let mut expected = std::collections::BTreeMap::new();
    let mut largest_group_delay_m = 0.0_f64;
    for prn in 1..=32_u8 {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id");
        let Ok(prediction) = predict(&store, sat, base, t_rx, PredictOptions::default()) else {
            continue;
        };
        if prediction.elevation_deg < 10.0 {
            continue;
        }
        let group_delay_s = ObservableEphemerisSource::single_frequency_group_delay_s(
            &store,
            sat,
            prediction.transmit_time_j2000_s,
        )
        .expect("GPS TGD");
        largest_group_delay_m = largest_group_delay_m.max((C_M_S * group_delay_s).abs());
        let injected_m = 3.0 + f64::from(prn) * 0.25;
        let pseudorange_m = synth_placed(&store, sat, base, t_rx, 0.0, injected_m);
        let modeled_m = placed_base_model_m(&store, sat, base, t_rx, pseudorange_m);
        base_obs.push(CodeObservation::new(sat.to_string(), pseudorange_m));
        expected.insert(sat.to_string(), (pseudorange_m - modeled_m, injected_m));
    }
    assert!(base_obs.len() >= 5, "need visible GPS satellites");
    assert!(largest_group_delay_m > 0.1, "{largest_group_delay_m} m");

    let corrections =
        pseudorange_corrections(&store, base, &base_obs, t_rx).expect("compute DGNSS corrections");
    assert_eq!(corrections.len(), expected.len());
    for (sat, (bits_expected, injected_m)) in &expected {
        let got = corrections[sat];
        assert_eq!(got.to_bits(), bits_expected.to_bits(), "{sat}");
        assert!(
            (got - injected_m).abs() < 1.0e-6,
            "{sat}: {got} vs {injected_m}"
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

    // Frozen bits of the clean DGNSS solve, all compared at once and printed together
    // on a mismatch. The synthetic pseudoranges carry the `peph2pos` relativistic term
    // the base and rover models apply; it cancels in the correction, so the solve
    // recovers the rover to well under a millimetre.
    let clean_bits = [
        (
            "position",
            clean
                .solution
                .position
                .as_array()
                .map(f64::to_bits)
                .to_vec(),
        ),
        ("rx_clock", vec![clean.solution.rx_clock_s.to_bits()]),
        (
            "baseline_vector",
            clean.baseline_vector_m.map(f64::to_bits).to_vec(),
        ),
        ("baseline", vec![clean.baseline_m.to_bits()]),
        (
            "residuals",
            clean
                .solution
                .residuals_m
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
        ),
    ];
    // Re-frozen when the base and rover models moved to RTKLIB `satposs` placement: each
    // satellite sits at t_rx - P / c - dts, the rover's placed from its raw pseudoranges.
    // The clean solve is then exact to rounding: every residual is zero and the baseline
    // is its designed (2000, 1000, 1500) m to within a few ulps.
    let frozen_bits: [(&str, Vec<u64>); 5] = [
        (
            "position",
            vec![0x414ad10a00000000, 0x4127d977fffffffa, 0x4154072600000001],
        ),
        ("rx_clock", vec![0xbec92a737110dee3]),
        (
            "baseline_vector",
            vec![0x409f400000000000, 0x408f3fffffffe800, 0x4097700000001000],
        ),
        ("baseline", vec![0x40a5092a30cce712]),
        ("residuals", vec![0x0; 10]),
    ];
    assert_eq!(
        clean_bits
            .iter()
            .map(|(label, bits)| format!("{label}: {bits:#x?}"))
            .collect::<Vec<_>>(),
        frozen_bits
            .iter()
            .map(|(label, bits)| format!("{label}: {bits:#x?}"))
            .collect::<Vec<_>>(),
        "clean DGNSS frozen bits"
    );

    assert!(absolute_error > 5.0);
    assert!((dgnss_error - clean_error).abs() <= 1.0e-3);
    assert!(dgnss_error < absolute_error / 100.0);
    assert_eq!(dgnss.dropped_sats, Vec::<String>::new());
    assert!((clean.baseline_m - dist(base, rover)).abs() <= 1.0e-2);
}

#[test]
fn dgnss_covariance_is_exactly_twice_the_spp_covariance_for_the_same_geometry() {
    // The single-difference correction sums rover and reference code noise, so
    // under the equal-independent-noise model the corrected observable carries
    // twice the variance of a raw pseudorange. The solver expresses that as a
    // 2x scale on the SPP-derived position covariance. SPP covariance depends
    // only on the converged geometry and the noise model, never on the
    // observation values, so on clean synthetic data (both solves converge to
    // the same rover position over the same satellites) the DGNSS covariance
    // must be the SPP covariance scaled by exactly two. If either side of the
    // model is dropped or double-counted, this ratio moves off 2.
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

    let spp = solve(
        &sp3,
        &solve_inputs(
            spp_observations(&rover_clean),
            [rover[0], rover[1], rover[2], 0.0],
        ),
        false,
    )
    .expect("clean SPP solve");

    let dgnss = solve_position(
        &sp3,
        base,
        &base_clean,
        &rover_clean,
        solve_inputs(Vec::new(), [rover[0], rover[1], rover[2], 0.0]),
        false,
    )
    .expect("clean DGNSS solve");

    let spp_cov = &spp.position_covariance;
    let dgnss_cov = &dgnss.solution.position_covariance;
    for row in 0..3 {
        for col in 0..3 {
            let expected_ecef = 2.0 * spp_cov.ecef_m2[row][col];
            let expected_enu = 2.0 * spp_cov.enu_m2[row][col];
            let scale_ecef = spp_cov.ecef_m2[row][row].max(spp_cov.ecef_m2[col][col]);
            let scale_enu = spp_cov.enu_m2[row][row].max(spp_cov.enu_m2[col][col]);
            assert!(
                (dgnss_cov.ecef_m2[row][col] - expected_ecef).abs() <= 1.0e-9 * scale_ecef,
                "ECEF[{row}][{col}]: dgnss {} vs 2x spp {}",
                dgnss_cov.ecef_m2[row][col],
                expected_ecef
            );
            assert!(
                (dgnss_cov.enu_m2[row][col] - expected_enu).abs() <= 1.0e-9 * scale_enu,
                "ENU[{row}][{col}]: dgnss {} vs 2x spp {}",
                dgnss_cov.enu_m2[row][col],
                expected_enu
            );
        }
    }
    // The unscaled covariances must genuinely differ, so the pin cannot pass
    // vacuously if the DGNSS scale is removed.
    assert!(
        (dgnss_cov.enu_m2[0][0] - spp_cov.enu_m2[0][0]).abs() > 0.4 * spp_cov.enu_m2[0][0],
        "scaled and unscaled covariance are indistinguishable; the pin is vacuous"
    );
}

/// The rover's satellites are placed from its raw pseudoranges, as RTKLIB `rtkpos` calls
/// `satposs` with the rover's own observations; the corrected pseudoranges form only the
/// residuals. A 1 µs base receiver clock shifts every correction by the same 300 m, which
/// the rover clock absorbs, and leaves each rover transmission epoch where it was: the
/// rover position does not move. Placed from the corrected code instead, each rover
/// satellite would move by 1 µs of its motion, about 4 mm, and the position by a
/// fraction of a millimetre, well above the 1 µm this test allows.
#[test]
fn dgnss_places_the_rover_from_its_raw_pseudoranges() {
    let sp3 = sp3_fixture();
    let base = [3_512_900.0, 780_500.0, 5_248_700.0];
    let rover = [base[0] + 2_000.0, base[1] + 1_000.0, base[2] + 1_500.0];
    let base_visible = visible_gps(&sp3, base);
    let mut sats: Vec<GnssSatelliteId> = visible_gps(&sp3, rover)
        .into_iter()
        .filter(|sat| base_visible.contains(sat))
        .collect();
    sats.sort_unstable();
    assert!(sats.len() >= 5);

    let rover_obs = synth(&sp3, &sats, rover, -2.0e-6);
    let solve_with_base_clock = |base_clock_s: f64| {
        solve_position(
            &sp3,
            base,
            &synth(&sp3, &sats, base, base_clock_s),
            &rover_obs,
            solve_inputs(Vec::new(), [rover[0], rover[1], rover[2], 0.0]),
            false,
        )
        .expect("DGNSS solve")
    };
    let steered = solve_with_base_clock(0.0);
    let offset = solve_with_base_clock(1.0e-6);

    let moved = dist(
        offset.solution.position.as_array(),
        steered.solution.position.as_array(),
    );
    assert!(
        moved < 1.0e-6,
        "the base clock moved the rover by {moved} m"
    );
    let clock_moved = offset.solution.rx_clock_s - steered.solution.rx_clock_s;
    assert!(
        (clock_moved + 1.0e-6).abs() < 1.0e-12,
        "the rover clock absorbs the base clock: moved {clock_moved} s"
    );
}
