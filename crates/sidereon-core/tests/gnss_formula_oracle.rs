#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::astro::time::split_julian_date;
use sidereon_core::constants::F_L1_HZ;
use sidereon_core::ephemeris::Sp3;
use sidereon_core::observables::{j2000_seconds_from_split, predict, PredictOptions};
use sidereon_core::quality::{
    chi2_inv, pseudorange_variance, sigmas, weight_vector, PseudorangeVarianceModel,
    PseudorangeVarianceOptions, WeightEntry,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const GOLDEN: &str = include_str!("fixtures/sidereon_gnss_formula_golden.json");
const SP3: &str = include_str!("fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3");

#[derive(Debug, Clone, Copy)]
struct CivilEpoch {
    year: i64,
    month: i64,
    day: i64,
    hour: u8,
    minute: u8,
    second: f64,
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

fn hex3(v: &Value) -> [f64; 3] {
    let values = v.as_array().expect("hex float array");
    assert_eq!(values.len(), 3, "expected three entries");
    [hexf(&values[0]), hexf(&values[1]), hexf(&values[2])]
}

fn assert_bits(actual: f64, expected: f64, label: &str) {
    assert_eq!(
        actual.to_bits(),
        expected.to_bits(),
        "{label}: actual={actual:?} expected={expected:?}"
    );
}

fn assert_in_delta(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "{label}: actual={actual:?} expected={expected:?} delta={delta:?} tolerance={tolerance:?}"
    );
}

fn golden() -> Value {
    serde_json::from_str(GOLDEN).expect("parse GNSS formula golden")
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

#[test]
fn ionosphere_free_formula_cases_match_golden_bits() {
    let doc = golden();
    let cases = doc["ionosphere_free"]["cases"]
        .as_array()
        .expect("ionosphere-free cases");
    assert_eq!(cases.len(), 3, "ionosphere-free cases must be complete");

    for (index, case) in cases.iter().enumerate() {
        let f1_hz = hexf(&case["f1_hz"]);
        let f2_hz = hexf(&case["f2_hz"]);
        let pr1_m = hexf(&case["pr1_m"]);
        let pr2_m = hexf(&case["pr2_m"]);
        let phase1_m = hexf(&case["phase1_m"]);
        let phase2_m = hexf(&case["phase2_m"]);
        let phi1_cycles = hexf(&case["phi1_cycles"]);
        let phi2_cycles = hexf(&case["phi2_cycles"]);

        assert_bits(
            sidereon_core::combinations::gamma(f1_hz, f2_hz).unwrap(),
            hexf(&case["gamma"]),
            &format!("case {index} gamma"),
        );
        assert_bits(
            sidereon_core::combinations::noise_amplification(f1_hz, f2_hz).unwrap(),
            hexf(&case["noise_amplification"]),
            &format!("case {index} noise_amplification"),
        );
        assert_bits(
            sidereon_core::combinations::ionosphere_free(pr1_m, pr2_m, f1_hz, f2_hz).unwrap(),
            hexf(&case["iono_free_m"]),
            &format!("case {index} iono_free_m"),
        );
        assert_bits(
            sidereon_core::combinations::ionosphere_free_phase_m(phase1_m, phase2_m, f1_hz, f2_hz)
                .unwrap(),
            hexf(&case["iono_free_phase_m"]),
            &format!("case {index} iono_free_phase_m"),
        );
        assert_bits(
            sidereon_core::combinations::ionosphere_free_phase_cycles(
                phi1_cycles,
                phi2_cycles,
                f1_hz,
                f2_hz,
            )
            .unwrap(),
            hexf(&case["iono_free_phase_from_cycles_m"]),
            &format!("case {index} iono_free_phase_from_cycles_m"),
        );
    }
}

#[test]
fn quality_formula_cases_match_golden() {
    let doc = golden();

    let variance_cases = doc["qc"]["variance_cases"]
        .as_array()
        .expect("variance cases");
    assert_eq!(variance_cases.len(), 5, "variance cases must be complete");
    for (index, case) in variance_cases.iter().enumerate() {
        let elevation_deg = hexf(&case["elevation_deg"]);
        let a_m = hexf(&case["a_m"]);
        let b_m = hexf(&case["b_m"]);
        let options = PseudorangeVarianceOptions::default();
        assert_eq!(
            a_m.to_bits(),
            options.a_m.to_bits(),
            "variance case {index} fixture a_m must match default"
        );
        assert_eq!(
            b_m.to_bits(),
            options.b_m.to_bits(),
            "variance case {index} fixture b_m must match default"
        );

        assert_bits(
            pseudorange_variance(elevation_deg, options).unwrap(),
            hexf(&case["variance_m2"]),
            &format!("variance case {index} variance_m2"),
        );

        let entries = vec![WeightEntry {
            satellite_id: "G01".to_string(),
            elevation_deg,
            cn0_dbhz: None,
        }];
        let sigma_by_sat = sigmas(&entries, options);
        let weights_by_sat = weight_vector(&entries, options);
        assert_bits(
            *sigma_by_sat.get("G01").expect("sigma for G01"),
            hexf(&case["sigma_m"]),
            &format!("variance case {index} sigma_m"),
        );
        assert_bits(
            *weights_by_sat.get("G01").expect("weight for G01"),
            hexf(&case["weight"]),
            &format!("variance case {index} weight"),
        );
    }

    let cn0_cases = doc["qc"]["cn0_cases"].as_array().expect("C/N0 cases");
    assert_eq!(cn0_cases.len(), 3, "C/N0 cases must be complete");
    for (index, case) in cn0_cases.iter().enumerate() {
        let elevation_deg = hexf(&case["elevation_deg"]);
        let cn0_dbhz = hexf(&case["cn0_dbhz"]);
        let cn0_scale = hexf(&case["cn0_scale"]);
        #[allow(clippy::field_reassign_with_default)]
        let options = {
            let mut options = PseudorangeVarianceOptions::default();
            options.model = PseudorangeVarianceModel::ElevationCn0;
            options.cn0_dbhz = Some(cn0_dbhz);
            options
        };
        assert_eq!(
            cn0_scale.to_bits(),
            options.cn0_scale_m2.to_bits(),
            "C/N0 case {index} fixture cn0_scale must match default"
        );

        assert_bits(
            pseudorange_variance(elevation_deg, options).unwrap(),
            hexf(&case["variance_m2"]),
            &format!("C/N0 case {index} variance_m2"),
        );
    }

    let chi2_cases = doc["qc"]["chi2_cases"]
        .as_array()
        .expect("chi-square cases");
    assert_eq!(chi2_cases.len(), 21, "chi-square cases must execute");
    for (index, case) in chi2_cases.iter().enumerate() {
        let p = hexf(&case["p"]);
        let dof = case["dof"].as_u64().expect("dof integer") as usize;
        assert_in_delta(
            chi2_inv(p, dof).unwrap(),
            hexf(&case["critical"]),
            1.0e-10,
            &format!("chi-square case {index}"),
        );
    }
}

#[test]
fn observable_formula_cases_match_golden_bits() {
    let doc = golden();
    let cases = doc["observables"]["cases"]
        .as_array()
        .expect("observable cases");
    assert_eq!(cases.len(), 3, "observable cases must be complete");
    let sp3 = Sp3::parse_str(SP3).expect("parse SP3 fixture");

    for (index, case) in cases.iter().enumerate() {
        let sat = sat_id(case["sat"].as_str().expect("satellite token"));
        let receiver_ecef_m = hex3(&case["receiver_ecef_m"]);
        let t_rx_j2000_s = j2000_seconds_from_iso(case["epoch"].as_str().expect("epoch"));
        let options = PredictOptions {
            light_time: case["light_time"].as_bool().expect("light_time"),
            sagnac: case["sagnac"].as_bool().expect("sagnac"),
            carrier_hz: F_L1_HZ,
        };

        let predicted = predict(&sp3, sat, receiver_ecef_m, t_rx_j2000_s, options)
            .unwrap_or_else(|err| panic!("observable case {index} {sat}: {err}"));

        assert_bits(
            predicted.geometric_range_m,
            hexf(&case["geometric_range_m"]),
            &format!("observable case {index} geometric_range_m"),
        );
        assert_bits(
            predicted.range_rate_m_s,
            hexf(&case["range_rate_m_s"]),
            &format!("observable case {index} range_rate_m_s"),
        );
        assert_bits(
            predicted.doppler_hz,
            hexf(&case["doppler_hz"]),
            &format!("observable case {index} doppler_hz"),
        );
        assert_bits(
            predicted.sat_clock_s.expect("satellite clock"),
            hexf(&case["sat_clock_s"]),
            &format!("observable case {index} sat_clock_s"),
        );
        assert_bits(
            predicted.elevation_deg,
            hexf(&case["elevation_deg"]),
            &format!("observable case {index} elevation_deg"),
        );
        assert_bits(
            predicted.azimuth_deg,
            hexf(&case["azimuth_deg"]),
            &format!("observable case {index} azimuth_deg"),
        );
        let expected_los = hex3(&case["los_unit"]);
        for (axis, expected) in expected_los.iter().enumerate() {
            assert_bits(
                predicted.los_unit[axis],
                *expected,
                &format!("observable case {index} los_unit[{axis}]"),
            );
        }

        let transmit_time_j2000_s =
            j2000_seconds_from_iso(case["transmit_time"].as_str().expect("transmit_time"));
        assert_in_delta(
            predicted.transmit_time_j2000_s,
            transmit_time_j2000_s,
            1.0e-6,
            &format!("observable case {index} transmit_time"),
        );
        assert_in_delta(
            t_rx_j2000_s - predicted.transmit_time_j2000_s,
            hexf(&case["tau_s"]),
            1.0e-6,
            &format!("observable case {index} tau_s"),
        );
    }
}
