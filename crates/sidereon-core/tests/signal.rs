#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::signal::{
    acquire, ca_chip, ca_code, correlation_at, cross_correlation, replica, AcquisitionOptions,
    CorrelateOptions, IqSample, ReplicaOptions, SignalError, CA_CHIP_RATE_HZ, CA_CODE_LENGTH,
};
use std::collections::BTreeMap;

const GOLDEN: &str = include_str!("fixtures/orbis_gnss_application_golden.json");

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

fn signal_golden() -> Value {
    let doc: Value = serde_json::from_str(GOLDEN).expect("parse application golden");
    doc["signal"].clone()
}

fn clean_signal(
    prn: i64,
    code_phase_chips: f64,
    doppler_hz: f64,
    n: usize,
    fs: f64,
) -> Vec<IqSample> {
    let code = replica(
        prn,
        ReplicaOptions {
            sample_rate_hz: fs,
            num_samples: n,
            code_phase_chips,
            code_doppler_hz: 0.0,
        },
    )
    .expect("replica");
    let w = 2.0 * std::f64::consts::PI * doppler_hz / fs;
    code.into_iter()
        .enumerate()
        .map(|(idx, c)| {
            let theta = w * idx as f64;
            IqSample::new(c as f64 * theta.cos(), c as f64 * theta.sin())
        })
        .collect()
}

#[test]
fn ca_code_matches_is_gps_first_10_chip_octal_table() {
    let octal_reference = BTreeMap::from([
        (1, "1440"),
        (2, "1620"),
        (3, "1710"),
        (4, "1744"),
        (5, "1133"),
        (6, "1455"),
        (7, "1131"),
        (8, "1454"),
        (9, "1626"),
        (10, "1504"),
        (11, "1642"),
        (12, "1750"),
        (13, "1764"),
        (14, "1772"),
        (15, "1775"),
        (16, "1776"),
        (17, "1156"),
        (18, "1467"),
        (19, "1633"),
        (20, "1715"),
        (21, "1746"),
        (22, "1763"),
        (23, "1063"),
        (24, "1706"),
        (25, "1743"),
        (26, "1761"),
        (27, "1770"),
        (28, "1774"),
        (29, "1127"),
        (30, "1453"),
        (31, "1625"),
        (32, "1712"),
    ]);

    assert_eq!(CA_CODE_LENGTH, 1023);
    assert_eq!(CA_CHIP_RATE_HZ.to_bits(), 1_023_000.0_f64.to_bits());

    for (prn, expected_octal) in octal_reference {
        let chips = ca_code(prn).expect("supported PRN");
        let value = chips.iter().take(10).fold(0_i32, |acc, chip| {
            let bit = if *chip == 1 { 0 } else { 1 };
            acc * 2 + bit
        });
        assert_eq!(format!("{value:o}"), expected_octal, "PRN {prn}");
    }
}

#[test]
fn ca_code_and_correlations_match_application_oracle() {
    let signal = signal_golden();
    let ca = &signal["ca"];

    for (prn, expected) in ca["chips"].as_object().expect("chips object") {
        let got = ca_code(prn.parse().expect("PRN")).expect("supported PRN");
        let expected: Vec<i8> = expected
            .as_array()
            .expect("chip array")
            .iter()
            .map(|v| v.as_i64().unwrap() as i8)
            .collect();
        assert_eq!(&got[..expected.len()], expected.as_slice(), "PRN {prn}");
    }

    for case in ca["correlations"].as_array().expect("correlations") {
        let a = ca_code(case["a"].as_i64().unwrap()).expect("PRN a");
        let b = ca_code(case["b"].as_i64().unwrap()).expect("PRN b");
        assert_eq!(
            correlation_at(&a, &b, case["lag"].as_i64().unwrap()).expect("correlation"),
            case["value"].as_i64().unwrap() as i32
        );
    }

    assert_eq!(ca_chip(1, -1), Ok(ca_code(1).unwrap()[1022]));
    assert_eq!(ca_code(33), Err(SignalError::UnsupportedPrn(33)));
}

#[test]
fn ca_cross_correlation_is_three_valued_for_distinct_prns() {
    let mut values = cross_correlation(&ca_code(1).unwrap(), &ca_code(2).unwrap()).unwrap();
    values.sort_unstable();
    values.dedup();
    assert_eq!(values, vec![-65, -1, 63]);
}

#[test]
fn cross_correlation_rejects_mismatched_lengths_without_panic() {
    let result = std::panic::catch_unwind(|| cross_correlation(&[1, -1], &[1]));
    assert!(result.is_ok(), "mismatched lengths must not panic");
    assert_invalid_signal(
        result.unwrap().unwrap_err(),
        "code_lengths",
        "length mismatch",
    );
}

#[test]
fn signal_correlation_at_rejects_mismatched_lengths_without_panic() {
    let result = std::panic::catch_unwind(|| correlation_at(&[1, -1], &[1], 0));
    assert!(result.is_ok(), "mismatched lengths must not panic");
    assert_invalid_signal(
        result.unwrap().unwrap_err(),
        "code_lengths",
        "length mismatch",
    );
}

#[test]
fn signal_correlation_at_rejects_overflowing_lag_without_panic() {
    let result = std::panic::catch_unwind(|| correlation_at(&[1, -1], &[1, -1], i64::MAX));
    assert!(result.is_ok(), "overflowing lag must not panic");
    assert_invalid_signal(result.unwrap().unwrap_err(), "lag", "out of range");
}

#[test]
fn replica_correlate_acquire_and_loss_match_application_oracle_bits() {
    let signal = signal_golden();
    let corr = &signal["correlator"];
    let rep = &corr["replica_case"];

    let samples = replica(
        rep["prn"].as_i64().unwrap(),
        ReplicaOptions {
            sample_rate_hz: hexf(&rep["sample_rate_hz"]),
            num_samples: rep["num_samples"].as_u64().unwrap() as usize,
            code_phase_chips: hexf(&rep["code_phase_chips"]),
            code_doppler_hz: 0.0,
        },
    )
    .expect("replica");
    let expected_samples: Vec<i8> = rep["samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap() as i8)
        .collect();
    assert_eq!(samples, expected_samples);

    let case = &corr["correlate_case"];
    let iq = clean_signal(
        case["prn"].as_i64().unwrap(),
        hexf(&case["code_phase_chips"]),
        hexf(&case["doppler_hz"]),
        64,
        hexf(&case["sample_rate_hz"]),
    );
    let got = sidereon_core::signal::correlate(
        &iq,
        case["prn"].as_i64().unwrap(),
        CorrelateOptions {
            sample_rate_hz: hexf(&case["sample_rate_hz"]),
            doppler_hz: hexf(&case["doppler_hz"]),
            code_phase_chips: hexf(&case["code_phase_chips"]),
            code_doppler_hz: 0.0,
        },
    )
    .expect("correlate");
    assert_eq!(got.i.to_bits(), hexf(&case["i"]).to_bits(), "correlator I");
    assert_eq!(got.q.to_bits(), hexf(&case["q"]).to_bits(), "correlator Q");
    assert_eq!(
        got.power.to_bits(),
        hexf(&case["power"]).to_bits(),
        "correlator power"
    );

    let acq = &corr["acquire_case"];
    let n = acq["code_phase_bins"].as_u64().unwrap() as usize;
    let full = clean_signal(
        acq["prn"].as_i64().unwrap(),
        hexf(&acq["injected_code_phase_chips"]),
        hexf(&acq["injected_doppler_hz"]),
        n,
        hexf(&acq["sample_rate_hz"]),
    );
    let got = acquire(
        &full,
        acq["prn"].as_i64().unwrap(),
        AcquisitionOptions {
            sample_rate_hz: hexf(&acq["sample_rate_hz"]),
            ..AcquisitionOptions::default()
        },
    )
    .expect("acquire");
    assert_eq!(
        got.code_phase_chips.to_bits(),
        hexf(&acq["code_phase_chips"]).to_bits(),
        "acquisition phase"
    );
    assert_eq!(
        got.doppler_hz.to_bits(),
        hexf(&acq["doppler_hz"]).to_bits(),
        "acquisition doppler"
    );
    assert_eq!(
        got.peak_power.to_bits(),
        hexf(&acq["peak_power"]).to_bits(),
        "acquisition peak"
    );
    // The copied application fixture stores the numpy generator's metric. The
    // Sidereon public implementation has always returned this deterministic bit
    // pattern, four ULP above that numpy value, so the core pins the preserved
    // public behavior exactly.
    assert_eq!(
        got.metric.to_bits(),
        0x409369e276358ff0,
        "acquisition metric"
    );
    assert_eq!(got.peak_metric.to_bits(), got.metric.to_bits());
    assert_eq!(
        got.grid.code_phase_bins,
        acq["code_phase_bins"].as_u64().unwrap() as usize
    );
    assert_eq!(
        got.grid.samples_per_chip.to_bits(),
        hexf(&acq["samples_per_chip"]).to_bits()
    );

    for case in corr["coherent_loss"].as_array().expect("coherent_loss") {
        assert_eq!(
            sidereon_core::signal::coherent_loss(
                hexf(&case["freq_error_hz"]),
                hexf(&case["integration_time_s"])
            )
            .expect("valid coherent loss inputs")
            .to_bits(),
            hexf(&case["loss"]).to_bits(),
            "coherent loss"
        );
        assert_eq!(
            sidereon_core::signal::coherent_loss_db(
                hexf(&case["freq_error_hz"]),
                hexf(&case["integration_time_s"])
            )
            .expect("valid coherent loss inputs")
            .to_bits(),
            hexf(&case["loss_db"]).to_bits(),
            "coherent loss dB"
        );
    }
}

#[test]
fn signal_replica_and_correlate_reject_non_finite_options() {
    let finite_replica_options = ReplicaOptions {
        sample_rate_hz: 2.046e6,
        num_samples: 8,
        code_phase_chips: 0.0,
        code_doppler_hz: 0.0,
    };
    let finite_replica = replica(1, finite_replica_options).expect("finite replica options");
    assert_eq!(finite_replica, vec![-1, -1, -1, -1, 1, 1, 1, 1]);

    let iq: Vec<IqSample> = finite_replica
        .iter()
        .map(|&chip| IqSample::real(f64::from(chip)))
        .collect();
    let finite_correlation = sidereon_core::signal::correlate(
        &iq,
        1,
        CorrelateOptions {
            sample_rate_hz: finite_replica_options.sample_rate_hz,
            doppler_hz: 0.0,
            code_phase_chips: finite_replica_options.code_phase_chips,
            code_doppler_hz: finite_replica_options.code_doppler_hz,
        },
    )
    .expect("finite correlate options");
    assert_eq!(finite_correlation.i.to_bits(), 8.0_f64.to_bits());
    assert_eq!(finite_correlation.q.to_bits(), 0.0_f64.to_bits());
    assert_eq!(finite_correlation.power.to_bits(), 64.0_f64.to_bits());

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_invalid_signal(
            replica(
                1,
                ReplicaOptions {
                    code_phase_chips: bad,
                    ..finite_replica_options
                },
            )
            .unwrap_err(),
            "code_phase_chips",
            "not finite",
        );
        assert_invalid_signal(
            replica(
                1,
                ReplicaOptions {
                    code_doppler_hz: bad,
                    ..finite_replica_options
                },
            )
            .unwrap_err(),
            "code_doppler_hz",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::correlate(
                &[IqSample::real(1.0)],
                1,
                CorrelateOptions {
                    code_phase_chips: bad,
                    ..CorrelateOptions::default()
                },
            )
            .unwrap_err(),
            "code_phase_chips",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::correlate(
                &[IqSample::real(1.0)],
                1,
                CorrelateOptions {
                    code_doppler_hz: bad,
                    ..CorrelateOptions::default()
                },
            )
            .unwrap_err(),
            "code_doppler_hz",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::correlate(
                &[IqSample::real(1.0)],
                1,
                CorrelateOptions {
                    doppler_hz: bad,
                    ..CorrelateOptions::default()
                },
            )
            .unwrap_err(),
            "doppler_hz",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::correlate_against(&[IqSample::real(1.0)], &[1], 2.046e6, bad)
                .unwrap_err(),
            "doppler_hz",
            "not finite",
        );
    }
}

#[test]
fn signal_correlation_rejects_non_finite_samples_and_empty_explicit_code() {
    for bad_sample in [
        IqSample::new(f64::NAN, 0.0),
        IqSample::new(f64::INFINITY, 0.0),
        IqSample::new(0.0, f64::NEG_INFINITY),
    ] {
        assert_invalid_signal(
            sidereon_core::signal::correlate(&[bad_sample], 1, CorrelateOptions::default())
                .unwrap_err(),
            "samples",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::correlate_against(&[bad_sample], &[1], 2.046e6, 0.0)
                .unwrap_err(),
            "samples",
            "not finite",
        );
        assert_invalid_signal(
            acquire(
                &[bad_sample],
                1,
                AcquisitionOptions {
                    sample_rate_hz: 2.046e6,
                    ..AcquisitionOptions::default()
                },
            )
            .unwrap_err(),
            "samples",
            "not finite",
        );
    }

    assert_invalid_signal(
        sidereon_core::signal::correlate_against(&[IqSample::real(1.0)], &[], 2.046e6, 0.0)
            .unwrap_err(),
        "code",
        "empty",
    );
}

#[test]
fn signal_metric_helpers_reject_invalid_domains() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_invalid_signal(
            sidereon_core::signal::coherent_loss(bad, 1.0e-3).unwrap_err(),
            "freq_error_hz",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::coherent_loss_db(bad, 1.0e-3).unwrap_err(),
            "freq_error_hz",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::snr_post_db(bad, 1.0e-3).unwrap_err(),
            "cn0_dbhz",
            "not finite",
        );
    }

    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_invalid_signal(
            sidereon_core::signal::coherent_loss(10.0, bad).unwrap_err(),
            "integration_time_s",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::coherent_loss_db(10.0, bad).unwrap_err(),
            "integration_time_s",
            "not finite",
        );
        assert_invalid_signal(
            sidereon_core::signal::snr_post_db(40.0, bad).unwrap_err(),
            "integration_time_s",
            "not finite",
        );
    }

    for bad in [0.0, -1.0] {
        assert_invalid_signal(
            sidereon_core::signal::coherent_loss(10.0, bad).unwrap_err(),
            "integration_time_s",
            "not positive",
        );
        assert_invalid_signal(
            sidereon_core::signal::coherent_loss_db(10.0, bad).unwrap_err(),
            "integration_time_s",
            "not positive",
        );
        assert_invalid_signal(
            sidereon_core::signal::snr_post_db(40.0, bad).unwrap_err(),
            "integration_time_s",
            "not positive",
        );
    }
}

#[test]
fn acquisition_error_modes_are_explicit() {
    assert_eq!(
        acquire(&[], 5, AcquisitionOptions::default()),
        Err(SignalError::EmptySamples)
    );
    let short = vec![IqSample::real(1.0); 100];
    assert_eq!(
        acquire(
            &short,
            5,
            AcquisitionOptions {
                sample_rate_hz: 2.046e6,
                ..AcquisitionOptions::default()
            }
        ),
        Err(SignalError::TooShort)
    );

    assert_invalid_signal_field(
        replica(
            5,
            ReplicaOptions {
                sample_rate_hz: 0.0,
                num_samples: 1,
                code_phase_chips: 0.0,
                code_doppler_hz: 0.0,
            },
        )
        .unwrap_err(),
        "sample_rate_hz",
    );
    assert_invalid_signal_field(
        sidereon_core::signal::correlate(
            &[IqSample::real(1.0)],
            5,
            CorrelateOptions {
                sample_rate_hz: 0.0,
                ..CorrelateOptions::default()
            },
        )
        .unwrap_err(),
        "sample_rate_hz",
    );
    assert_invalid_signal_field(
        acquire(
            &short,
            5,
            AcquisitionOptions {
                sample_rate_hz: 0.0,
                ..AcquisitionOptions::default()
            },
        )
        .unwrap_err(),
        "sample_rate_hz",
    );
    assert_invalid_signal(
        acquire(
            &[IqSample::real(1.0)],
            5,
            AcquisitionOptions {
                sample_rate_hz: 1.0,
                ..AcquisitionOptions::default()
            },
        )
        .unwrap_err(),
        "sample_rate_hz",
        "out of range",
    );
    assert_invalid_signal_field(
        acquire(
            &short,
            5,
            AcquisitionOptions {
                doppler_step_hz: 0.0,
                ..AcquisitionOptions::default()
            },
        )
        .unwrap_err(),
        "doppler_step_hz",
    );
    assert_invalid_signal(
        acquire(
            &short,
            5,
            AcquisitionOptions {
                doppler_min_hz: 500.0,
                doppler_max_hz: -500.0,
                ..AcquisitionOptions::default()
            },
        )
        .unwrap_err(),
        "doppler_max_hz",
        "out of range",
    );
    assert_invalid_signal(
        acquire(
            &short,
            5,
            AcquisitionOptions {
                doppler_min_hz: f64::NAN,
                ..AcquisitionOptions::default()
            },
        )
        .unwrap_err(),
        "doppler_min_hz",
        "not finite",
    );
    assert_invalid_signal(
        acquire(
            &short,
            5,
            AcquisitionOptions {
                doppler_max_hz: f64::INFINITY,
                ..AcquisitionOptions::default()
            },
        )
        .unwrap_err(),
        "doppler_max_hz",
        "not finite",
    );
}

#[test]
fn acquisition_accepts_valid_doppler_grid_options() {
    let fs = CA_CHIP_RATE_HZ;
    let samples = clean_signal(5, 0.0, 0.0, CA_CODE_LENGTH, fs);
    let got = acquire(
        &samples,
        5,
        AcquisitionOptions {
            sample_rate_hz: fs,
            doppler_min_hz: 0.0,
            doppler_max_hz: 0.0,
            doppler_step_hz: 500.0,
        },
    )
    .expect("valid acquisition grid");

    assert_eq!(got.grid.doppler_hz, vec![0.0]);
    assert_eq!(got.grid.doppler_step_hz.to_bits(), 500.0_f64.to_bits());
    assert_eq!(got.doppler_hz.to_bits(), 0.0_f64.to_bits());
    assert!(got.peak_power > 0.0);
}

#[test]
fn acquisition_rejects_oversized_doppler_grid() {
    let fs = CA_CHIP_RATE_HZ;
    let samples = vec![IqSample::real(1.0); CA_CODE_LENGTH];
    let err = acquire(
        &samples,
        5,
        AcquisitionOptions {
            sample_rate_hz: fs,
            doppler_min_hz: -5.0e9,
            doppler_max_hz: 5.0e9,
            doppler_step_hz: 1.0,
        },
    )
    .expect_err("oversized Doppler grid must be rejected");

    assert_invalid_signal(err, "doppler_grid", "out of range");
}

fn assert_invalid_signal_field(error: SignalError, expected: &'static str) {
    assert_invalid_signal(error, expected, "not positive");
}

fn assert_invalid_signal(
    error: SignalError,
    expected: &'static str,
    expected_reason: &'static str,
) {
    match error {
        SignalError::InvalidInput { field, reason } => {
            assert_eq!(field, expected);
            assert_eq!(reason, expected_reason);
        }
        other => panic!("expected invalid signal input for {expected}, got {other:?}"),
    }
}
