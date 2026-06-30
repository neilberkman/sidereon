#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::carrier_phase::{
    code_minus_carrier, detect_cycle_slips, geometry_free, melbourne_wubbena, narrow_lane_code,
    phase_meters, smooth_code, smooth_iono_free_code, wide_lane_wavelength, ArcEpoch,
    CycleSlipOptions, SlipReason,
};
use sidereon_core::rinex::{
    decode_crinex,
    observations::{carrier_phase_rows, observation_values, ObservationFilter, RinexObs},
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const GOLDEN: &str = include_str!("fixtures/carrier_phase_golden.json");
const ESBC_CRX: &str = include_str!("fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx");

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

fn maybe_hexf(v: &Value) -> Option<f64> {
    if v.is_null() {
        None
    } else {
        Some(hexf(v))
    }
}

fn assert_bits(actual: f64, expected: &Value, label: &str) {
    let expected = hexf(expected);
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{label}: actual={actual:?} expected={expected:?}"
    );
}

fn assert_maybe_bits(actual: Option<f64>, expected: &Value, label: &str) {
    match maybe_hexf(expected) {
        Some(expected) => {
            let actual = actual.unwrap_or_else(|| panic!("{label}: expected numeric value"));
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label}: actual={actual:?} expected={expected:?}"
            );
        }
        None => assert!(actual.is_none(), "{label}: expected nil, got {actual:?}"),
    }
}

fn maybe_u8(v: &Value) -> Option<u8> {
    v.as_u64().map(|value| value as u8)
}

fn golden() -> Value {
    serde_json::from_str(GOLDEN).expect("parse carrier phase golden")
}

fn parsed_obs() -> RinexObs {
    let decoded = decode_crinex(ESBC_CRX).expect("decode CRINEX fixture");
    RinexObs::parse(&decoded).expect("parse decoded RINEX OBS")
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

fn arc_from_golden(rows: &[Value]) -> Vec<ArcEpoch> {
    rows.iter()
        .map(|row| ArcEpoch {
            phi1_cycles: maybe_hexf(&row["phi1"]),
            phi2_cycles: maybe_hexf(&row["phi2"]),
            p1_m: maybe_hexf(&row["p1"]),
            p2_m: maybe_hexf(&row["p2"]),
            lli1: row["lli1"].as_i64(),
            lli2: row["lli2"].as_i64(),
            f1_hz: maybe_hexf(&row["f1"]),
            f2_hz: maybe_hexf(&row["f2"]),
            gap_time_s: row["epoch"].as_i64().map(|epoch| epoch as f64),
        })
        .collect()
}

fn reason_name(reason: SlipReason) -> &'static str {
    match reason {
        SlipReason::Lli => "lli",
        SlipReason::DataGap => "data_gap",
        SlipReason::GeometryFree => "geometry_free",
        SlipReason::MelbourneWubbena => "melbourne_wubbena",
    }
}

#[test]
fn rinex_observation_values_match_georinex_fixture_bits() {
    let doc = golden();
    let obs = parsed_obs();
    let by_epoch: Vec<_> = obs
        .epochs()
        .iter()
        .map(|epoch| {
            observation_values(&obs, epoch, &ObservationFilter::all())
                .expect("valid observation values")
        })
        .collect();

    for row in doc["rinex_observations"]["rows"].as_array().unwrap() {
        let epoch_index = row["epoch_index"].as_u64().unwrap() as usize;
        let sat = sat_id(row["sat"].as_str().unwrap());
        let code = row["code"].as_str().unwrap();
        let rows = by_epoch[epoch_index]
            .iter()
            .find(|(candidate, _)| *candidate == sat)
            .unwrap_or_else(|| panic!("missing satellite {sat} at epoch {epoch_index}"))
            .1
            .as_slice();
        let actual = rows
            .iter()
            .find(|candidate| candidate.code == code)
            .unwrap_or_else(|| panic!("missing {sat} {code} at epoch {epoch_index}"));

        assert_eq!(actual.kind.as_str(), row["kind"].as_str().unwrap());
        assert_eq!(actual.kind.units_str(), row["units"].as_str().unwrap());
        assert_eq!(actual.lli, maybe_u8(&row["lli"]));
        assert_eq!(actual.ssi, maybe_u8(&row["ssi"]));
        assert_maybe_bits(actual.value, &row["value"], &format!("{sat} {code} value"));
    }
}

#[test]
fn rinex_carrier_phase_rows_match_georinex_fixture_bits() {
    let doc = golden();
    let obs = parsed_obs();
    let by_epoch: Vec<_> = obs
        .epochs()
        .iter()
        .map(|epoch| {
            carrier_phase_rows(&obs, epoch, &ObservationFilter::all())
                .expect("valid carrier-phase rows")
        })
        .collect();

    for row in doc["rinex_observations"]["phases"].as_array().unwrap() {
        let epoch_index = row["epoch_index"].as_u64().unwrap() as usize;
        let sat = sat_id(row["sat"].as_str().unwrap());
        let code = row["code"].as_str().unwrap();
        let rows = by_epoch[epoch_index]
            .iter()
            .find(|(candidate, _)| *candidate == sat)
            .unwrap_or_else(|| panic!("missing satellite {sat} at epoch {epoch_index}"))
            .1
            .as_slice();
        let actual = rows
            .iter()
            .find(|candidate| candidate.code == code)
            .unwrap_or_else(|| panic!("missing {sat} {code} at epoch {epoch_index}"));

        assert_eq!(actual.lli, maybe_u8(&row["lli"]));
        assert_eq!(actual.ssi, maybe_u8(&row["ssi"]));
        assert_eq!(actual.phase_shift_cycles.to_bits(), 0.0_f64.to_bits());
        assert_maybe_bits(
            actual.value_cycles,
            &row["value_cycles"],
            &format!("{sat} {code} cycles"),
        );
        assert_maybe_bits(
            actual.frequency_hz,
            &row["frequency_hz"],
            &format!("{sat} {code} frequency"),
        );
        assert_maybe_bits(
            actual.wavelength_m,
            &row["wavelength_m"],
            &format!("{sat} {code} wavelength"),
        );
        assert_maybe_bits(
            actual.value_m,
            &row["value_m"],
            &format!("{sat} {code} metres"),
        );
    }
}

#[test]
fn scalar_combinations_match_python_fixture_bits() {
    let doc = golden();
    let carrier = &doc["carrier_phase"];
    let scalars = &carrier["scalar_cases"];
    let f1 = hexf(&carrier["constants"]["f_l1_hz"]);
    let f2 = hexf(&carrier["constants"]["f_l2_hz"]);

    assert_bits(
        phase_meters(123_456_789.25, f1).unwrap(),
        &scalars["phase_meters_m"],
        "phase metres",
    );
    assert_bits(
        code_minus_carrier(23_000_010.25, 123_456_789.25, f1).unwrap(),
        &scalars["code_minus_carrier_m"],
        "code-minus-carrier",
    );
    assert_bits(
        geometry_free(100.0, 60.0).unwrap(),
        &scalars["geometry_free_m"],
        "geometry-free",
    );
    assert_bits(
        wide_lane_wavelength(f1, f2).unwrap(),
        &scalars["wide_lane_wavelength_m"],
        "wide-lane wavelength",
    );
    assert_bits(
        narrow_lane_code(10.0, 12.0, f1, f2).unwrap(),
        &scalars["narrow_lane_code_m"],
        "narrow-lane code",
    );
    assert_bits(
        melbourne_wubbena(5.0, 3.0, 10.0, 12.0, f1, f2).unwrap(),
        &scalars["melbourne_wubbena_m"],
        "Melbourne-Wubbena",
    );
}

#[test]
fn cycle_slips_and_hatch_smoothing_match_python_fixture_bits() {
    let doc = golden();
    let carrier = &doc["carrier_phase"];
    let arc = arc_from_golden(carrier["arc"].as_array().unwrap());

    let slips = detect_cycle_slips(&arc, CycleSlipOptions::default()).expect("valid slip arc");
    for (actual, expected) in slips
        .iter()
        .zip(carrier["detect_cycle_slips"].as_array().unwrap())
    {
        assert_eq!(actual.slip, expected["slip"].as_bool().unwrap());
        assert_eq!(
            actual
                .reasons
                .iter()
                .map(|reason| reason_name(*reason))
                .collect::<Vec<_>>(),
            expected["reasons"]
                .as_array()
                .unwrap()
                .iter()
                .map(|reason| reason.as_str().unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(actual.skipped, expected["skipped"].as_bool().unwrap());
        assert_maybe_bits(actual.gf_m, &expected["gf"], "geometry-free slip value");
        assert_maybe_bits(actual.mw_m, &expected["mw"], "Melbourne-Wubbena slip value");
    }

    let smoothed = smooth_code(&arc, CycleSlipOptions::default(), 100).expect("valid smoothing");
    for (actual, expected) in smoothed
        .iter()
        .zip(carrier["smooth_code"].as_array().unwrap())
    {
        assert_eq!(actual.window, expected["window"].as_u64().unwrap() as usize);
        assert_eq!(actual.reset, expected["reset"].as_bool().unwrap());
        assert_maybe_bits(actual.p_smooth_m, &expected["p_smooth"], "Hatch code");
    }

    let smoothed_if =
        smooth_iono_free_code(&arc, CycleSlipOptions::default(), 100).expect("valid smoothing");
    for (actual, expected) in smoothed_if
        .iter()
        .zip(carrier["smooth_iono_free_code"].as_array().unwrap())
    {
        assert_eq!(actual.window, expected["window"].as_u64().unwrap() as usize);
        assert_eq!(actual.reset, expected["reset"].as_bool().unwrap());
        assert_maybe_bits(
            actual.p_smooth_m,
            &expected["p_smooth"],
            "ionosphere-free Hatch code",
        );
        assert_maybe_bits(actual.p_if_m, &expected["p_if"], "ionosphere-free code");
        assert_maybe_bits(actual.l_if_m, &expected["l_if"], "ionosphere-free phase");
    }
}
