use super::*;
use crate::ambiguity::AmbiguityId;
use crate::astro::math::vec3::{add3, cross3, norm3, scale3, sub3, unit3};
use crate::carrier_phase::{CycleSlipOptions, SlipReason};
use crate::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
use crate::has::{
    HasCodeBias, HasCodeBiasBlock, HasGnssMask, HasMaskBlock, HasMt1Header, HasMt1Message,
    HasPhaseBias, HasPhaseBiasBlock,
};
use crate::observables::{predict, ObservableState, ObservablesError};
use crate::ppp_corrections::{CivilDateTime, CodeBiasOptions, PppCorrectionsOptions};
use crate::ssr::SsrCorrectionStore;
use crate::{GnssSatelliteId, GnssSystem};
use std::collections::BTreeSet;

const REAL_CODE_BIA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/bias/CODE.BIA"
));

struct FakeSource {
    states: BTreeMap<GnssSatelliteId, [f64; 3]>,
}

impl ObservableEphemerisSource for FakeSource {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let position_ecef_m = self
            .states
            .get(&sat)
            .copied()
            .ok_or(ObservablesError::NoEphemeris)?;
        Ok(ObservableState {
            position_ecef_m,
            clock_s: Some(0.0),
        })
    }
}

struct NoClockSource {
    states: BTreeMap<GnssSatelliteId, [f64; 3]>,
}

impl ObservableEphemerisSource for NoClockSource {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let position_ecef_m = self
            .states
            .get(&sat)
            .copied()
            .ok_or(ObservablesError::NoEphemeris)?;
        Ok(ObservableState {
            position_ecef_m,
            clock_s: None,
        })
    }
}

fn single_obs_clock_epoch(sat: GnssSatelliteId) -> FloatEpoch {
    FloatEpoch {
        epoch: CivilDateTime {
            year: 2020,
            month: 6,
            day: 24,
            hour: 12,
            minute: 0,
            second: 0.0,
        },
        jd_whole: 2_459_024.5,
        jd_fraction: 0.5,
        t_rx_j2000_s: 0.0,
        observations: vec![FloatObservation {
            sat,
            satellite_id: sat.to_string(),
            ambiguity_id: sat.to_string(),
            code_m: 23_000_000.0,
            phase_m: 23_000_010.0,
            freq1_hz: 0.0,
            freq2_hz: 0.0,
            glonass_channel: None,
        }],
    }
}

fn single_obs_clock_state(epoch: &FloatEpoch) -> FloatState {
    FloatState {
        position_m: [3_512_900.0, 780_500.0, 5_248_700.0],
        clocks_m: vec![0.0],
        ambiguities_m: initial_ambiguities(std::slice::from_ref(epoch)),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    }
}

fn single_obs_clock_config(corrections: RangeCorrections) -> FloatSolveConfig {
    FloatSolveConfig {
        weights: MeasurementWeights {
            code: 1.0,
            phase: 100.0,
            elevation_weighting: false,
        },
        tropo: TroposphereOptions::disabled(),
        corrections,
        opts: FloatSolveOptions {
            max_iterations: 1,
            position_tolerance_m: 1.0e-4,
            clock_tolerance_m: 1.0e-4,
            ambiguity_tolerance_m: 1.0e-4,
            ztd_tolerance_m: 1.0e-4,
        },
        elevation_cutoff_deg: None,
        residual_screen: false,
        estimate_residual_ionosphere: false,
    }
}

fn assert_missing_satellite_clock(error: FloatSolveError, sat: GnssSatelliteId) {
    assert_eq!(
        error,
        FloatSolveError::NoEphemeris {
            satellite_id: sat.to_string(),
            reason: NoEphemerisReason::MissingSatelliteClock,
        }
    );
}

fn assert_missing_correction(
    error: FloatSolveError,
    sat: GnssSatelliteId,
    correction: MissingCorrection,
) {
    assert_eq!(
        error,
        FloatSolveError::MissingCorrection {
            satellite_id: sat.to_string(),
            correction,
        }
    );
}

fn assert_invalid_clock_count(error: FloatSolveError, expected: usize, actual: usize) {
    assert_eq!(
        error,
        FloatSolveError::InvalidClockCount { expected, actual }
    );
}

fn assert_invalid_solve_option(error: FloatSolveError, field: &'static str, reason: &'static str) {
    assert_eq!(error, FloatSolveError::InvalidSolveOption { field, reason });
}

fn assert_invalid_input(error: FloatSolveError, field: &'static str, reason: &'static str) {
    assert_eq!(error, FloatSolveError::InvalidInput { field, reason });
}

fn unit_position_covariance() -> crate::dop::PositionCovariance {
    crate::dop::PositionCovariance {
        ecef_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        enu_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }
}

fn unit_temporal_correlation() -> TemporalCorrelationSummary {
    TemporalCorrelationSummary {
        lag1_autocorrelation: 0.0,
        decorrelation_time_epochs: 0.0,
        decorrelation_time_s: None,
        nominal_sample_count: 0,
        effective_sample_count: 0.0,
        variance_inflation_factor: 1.0,
        arcs_used: 0,
    }
}

fn assert_position_covariance_positive_definite(covariance: &crate::dop::PositionCovariance) {
    fn assert_matrix(name: &str, matrix: [[f64; 3]; 3]) {
        for (idx, row) in matrix.iter().enumerate() {
            assert!(
                row[idx].is_finite() && row[idx] > 0.0,
                "{name} covariance diagonal {idx} was {}",
                row[idx]
            );
            for (jdx, other_row) in matrix.iter().enumerate().skip(idx + 1) {
                assert!(
                    (row[jdx] - other_row[idx]).abs() < 1.0e-10,
                    "{name} covariance is asymmetric at {idx},{jdx}"
                );
            }
        }
        let dense = matrix
            .iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f64>>>();
        assert!(
            crate::astro::math::linear::invert_symmetric_pd(&dense).is_some(),
            "{name} covariance was not positive definite"
        );
    }

    assert_matrix("ECEF", covariance.ecef_m2);
    assert_matrix("ENU", covariance.enu_m2);
}

fn assert_position_covariance_scaled_by_factor(
    scaled: &crate::dop::PositionCovariance,
    formal: &crate::dop::PositionCovariance,
    factor: f64,
) {
    fn assert_matrix(scaled: [[f64; 3]; 3], formal: [[f64; 3]; 3], factor: f64) {
        for row in 0..3 {
            for col in 0..3 {
                let expected = formal[row][col] * factor;
                let got = scaled[row][col];
                let tolerance = expected.abs().max(got.abs()).max(1.0) * 1.0e-12;
                assert!(
                    (got - expected).abs() <= tolerance,
                    "scaled covariance [{row}][{col}] {got} != formal * factor {expected}"
                );
            }
        }
    }

    assert_matrix(scaled.ecef_m2, formal.ecef_m2, factor);
    assert_matrix(scaled.enu_m2, formal.enu_m2, factor);
}

fn assert_temporal_covariance_not_smaller(solution: &FloatSolution) {
    assert!(
        solution.temporal_position_covariance_scale_factor >= 1.0,
        "temporal covariance scale factor {} was below one",
        solution.temporal_position_covariance_scale_factor
    );
    for idx in 0..3 {
        assert!(
            solution.temporal_position_covariance.ecef_m2[idx][idx]
                >= solution.formal_position_covariance.ecef_m2[idx][idx],
            "ECEF temporal covariance diagonal {idx} was below formal"
        );
        assert!(
            solution.temporal_position_covariance.enu_m2[idx][idx]
                >= solution.formal_position_covariance.enu_m2[idx][idx],
            "ENU temporal covariance diagonal {idx} was below formal"
        );
    }
}

fn ppp_cutoff_sat_position(receiver_m: [f64; 3], az_deg: f64, el_deg: f64) -> [f64; 3] {
    let az = az_deg.to_radians();
    let el = el_deg.to_radians();
    let range_m = 26_000_000.0;
    let los = [
        libm::sin(el),
        libm::cos(el) * libm::sin(az),
        libm::cos(el) * libm::cos(az),
    ];
    [
        receiver_m[0] + range_m * los[0],
        receiver_m[1] + range_m * los[1],
        receiver_m[2] + range_m * los[2],
    ]
}

fn ppp_elevation_cutoff_arc() -> (FakeSource, Vec<FloatEpoch>, FloatState, Vec<String>) {
    let truth = [6_378_137.0, 0.0, 0.0];
    let sat_specs = [
        (1u8, 0.0, 60.0),
        (2, 90.0, 55.0),
        (3, 180.0, 50.0),
        (4, 270.0, 45.0),
        (5, 45.0, 10.0),
        (6, 225.0, 5.0),
    ];
    let ids = sat_specs
        .iter()
        .map(|(prn, _, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).unwrap())
        .collect::<Vec<_>>();
    let source = FakeSource {
        states: ids
            .iter()
            .zip(sat_specs.iter())
            .map(|(id, (_, az_deg, el_deg))| {
                (*id, ppp_cutoff_sat_position(truth, *az_deg, *el_deg))
            })
            .collect(),
    };
    let clocks = [12.5, -8.25, 4.0];
    let ambiguities = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
        .collect::<BTreeMap<_, _>>();
    let mut epochs = Vec::new();
    for (epoch_idx, clock) in clocks.iter().enumerate() {
        let t_rx_j2000_s = epoch_idx as f64 * 900.0;
        let observations = ids
            .iter()
            .map(|id| {
                let pred = predict(
                    &source,
                    *id,
                    truth,
                    t_rx_j2000_s,
                    PredictOptions {
                        carrier_hz: F_L1_HZ,
                        light_time: true,
                        sagnac: true,
                    },
                )
                .unwrap();
                let code_m = pred.geometric_range_m + clock;
                let ambiguity_m = ambiguities[id.to_string().as_str()];
                FloatObservation {
                    sat: *id,
                    satellite_id: id.to_string(),
                    ambiguity_id: id.to_string(),
                    code_m,
                    phase_m: code_m + ambiguity_m,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                    glonass_channel: None,
                }
            })
            .collect();
        epochs.push(FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24,
                hour: 12,
                minute: epoch_idx as u8 * 15,
                second: 0.0,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5 + t_rx_j2000_s / crate::constants::SECONDS_PER_DAY,
            t_rx_j2000_s,
            observations,
        });
    }
    let state = FloatState {
        position_m: truth,
        clocks_m: vec![0.0; epochs.len()],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let low_sats = ["G05", "G06"].iter().map(|sat| sat.to_string()).collect();
    (source, epochs, state, low_sats)
}

fn ppp_cutoff_config(cutoff_deg: Option<f64>) -> FloatSolveConfig {
    FloatSolveConfig {
        weights: MeasurementWeights {
            code: 1.0,
            phase: 100.0,
            elevation_weighting: true,
        },
        tropo: TroposphereOptions::disabled(),
        corrections: RangeCorrections::disabled(),
        opts: FloatSolveOptions {
            max_iterations: 8,
            position_tolerance_m: 1.0e-4,
            clock_tolerance_m: 1.0e-4,
            ambiguity_tolerance_m: 1.0e-4,
            ztd_tolerance_m: 1.0e-4,
        },
        elevation_cutoff_deg: cutoff_deg,
        residual_screen: false,
        estimate_residual_ionosphere: false,
    }
}

fn ppp_float_solution_bits(solution: &FloatSolution) -> Vec<u64> {
    let mut bits = Vec::new();
    bits.extend(solution.position_m.iter().map(|v| v.to_bits()));
    bits.extend(solution.epoch_clocks_m.iter().map(|v| v.to_bits()));
    bits.extend(solution.ambiguities_m.values().map(|v| v.to_bits()));
    for residual in &solution.residuals_m {
        bits.push(residual.code_m.to_bits());
        bits.push(residual.phase_m.to_bits());
        bits.push(residual.code_weight.to_bits());
        bits.push(residual.phase_weight.to_bits());
    }
    bits.push(solution.code_rms_m.to_bits());
    bits.push(solution.phase_rms_m.to_bits());
    bits.push(solution.weighted_rms_m.to_bits());
    bits
}

#[test]
fn float_solution_output_validation_rejects_nonfinite_values() {
    let solution = FloatSolution {
        position_m: [0.0, f64::NAN, 0.0],
        position_covariance: unit_position_covariance(),
        formal_position_covariance: unit_position_covariance(),
        posterior_variance_factor: 1.0,
        position_covariance_scale_factor: 1.0,
        temporal_position_covariance: unit_position_covariance(),
        temporal_position_covariance_scale_factor: 1.0,
        temporal_correlation: unit_temporal_correlation(),
        epoch_clocks_m: vec![0.0],
        ambiguities_m: BTreeMap::new(),
        residual_ionosphere_m: BTreeMap::new(),
        ztd_residual_m: None,
        tropo_gradient_north_m: None,
        tropo_gradient_east_m: None,
        tropo_gradient_covariance_m2: None,
        formal_tropo_gradient_covariance_m2: None,
        residuals_m: Vec::new(),
        used_sats: Vec::new(),
        iterations: 1,
        converged: false,
        status: FloatStatus::MaxIterations,
        code_rms_m: 0.0,
        phase_rms_m: 0.0,
        weighted_rms_m: 0.0,
    };

    assert_invalid_input(
        validate_float_solution_output(&solution, 1).expect_err("nonfinite output must error"),
        "ppp float_solution position_m",
        "not finite",
    );
}

fn gps_l2_hz() -> f64 {
    crate::frequencies::frequency_hz(GnssSystem::Gps, crate::frequencies::CarrierBand::L2)
        .expect("canonical GPS L2 carrier exists")
}

#[test]
fn ppp_lookup_applies_real_glonass_osb_with_observation_fdma_channel() {
    let sp3_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    );
    let sp3_bytes =
        std::fs::read(sp3_path).unwrap_or_else(|e| panic!("read SP3 fixture {sp3_path}: {e}"));
    let sp3 = Sp3::parse(&sp3_bytes).expect("parse SP3 fixture");
    let bias_set = crate::bias::BiasSet::parse_bias_sinex(REAL_CODE_BIA)
        .expect("parse real CODE Bias-SINEX")
        .value;
    let sat = GnssSatelliteId::new(GnssSystem::Glonass, 2).expect("valid GLONASS satellite");
    let channel = -4;
    let freq1_hz = crate::frequencies::rinex_observation_frequency_hz(
        GnssSystem::Glonass,
        "C1C",
        3.04,
        Some(channel),
    )
    .expect("GLONASS C1C frequency");
    let freq2_hz = crate::frequencies::rinex_observation_frequency_hz(
        GnssSystem::Glonass,
        "C2C",
        3.04,
        Some(channel),
    )
    .expect("GLONASS C2C frequency");
    let epoch = CivilDateTime {
        year: 2026,
        month: 6,
        day: 24,
        hour: 12,
        minute: 0,
        second: 0.0,
    };
    let (jd_whole, jd_fraction) = crate::astro::time::split_julian_date(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    );
    let mut used_observables_default = BTreeMap::new();
    used_observables_default.insert(GnssSystem::Glonass, ("C1C".to_string(), "C2C".to_string()));
    let epochs = vec![FloatEpoch {
        epoch,
        jd_whole,
        jd_fraction,
        t_rx_j2000_s: crate::observables::j2000_seconds_from_split(jd_whole, jd_fraction)
            .expect("valid split Julian date"),
        observations: vec![FloatObservation {
            sat,
            satellite_id: sat.to_string(),
            ambiguity_id: sat.to_string(),
            code_m: 0.0,
            phase_m: 0.0,
            freq1_hz,
            freq2_hz,
            glonass_channel: Some(channel),
        }],
    }];
    let lookup = build_ppp_lookup(
        &sp3,
        &epochs,
        [3_512_900.0, 780_500.0, 5_248_700.0],
        &PppCorrectionsOptions {
            solid_earth_tide: false,
            pole_tide: None,
            ocean_loading: None,
            phase_windup: false,
            satellite_antenna: None,
            code_bias: Some(CodeBiasOptions {
                bias_set,
                used_observables_per_sat: BTreeMap::new(),
                used_observables_default,
                clock_reference: None,
            }),
        },
    )
    .expect("build PPP lookup with real GLONASS OSBs");

    let (alpha, beta) = crate::bias::ionosphere_free_coefficients(freq1_hz, freq2_hz).unwrap();
    let used_if = alpha * (0.2114_f64 * 1.0e-9) + beta * (2.6597_f64 * 1.0e-9);
    let ref_if = alpha * (1.7840_f64 * 1.0e-9) + beta * (2.9490_f64 * 1.0e-9);
    let expected = (used_if - ref_if) * C_M_S;

    assert_eq!(
        lookup.code_bias_m.get(&(sat, 0)).copied().map(f64::to_bits),
        Some(expected.to_bits())
    );
}

#[test]
fn float_ppp_errors_when_predicted_satellite_clock_is_missing() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = NoClockSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(RangeCorrections::disabled()),
    )
    .expect_err("missing satellite clock must error");

    assert_missing_satellite_clock(err, sat);
}

#[test]
fn float_ppp_errors_when_enabled_satellite_clock_table_has_gap() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = FakeSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        satellite_clock: Some(SatelliteClockCorrections::default()),
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("enabled satellite clock product gap must error");

    assert_missing_satellite_clock(err, sat);
}

#[test]
fn float_ppp_external_clock_can_replace_missing_predicted_clock() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = NoClockSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        satellite_clock: Some(SatelliteClockCorrections {
            series: BTreeMap::from([(sat, vec![(0.0, 1.0e-6), (1.0e12, 1.0e-6)])]),
        }),
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("one satellite still has singular geometry");

    assert_eq!(err, FloatSolveError::SingularGeometry);
}

#[test]
fn float_ppp_rejects_unsorted_external_satellite_clock_series() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = FakeSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        satellite_clock: Some(SatelliteClockCorrections {
            series: BTreeMap::from([(sat, vec![(1.0e12, 1.0e-6), (0.0, 1.0e-6)])]),
        }),
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("unsorted satellite clock product must error before interpolation");

    assert_invalid_input(err, "ppp satellite clock epoch_s", "out of range");
}

#[test]
fn float_ppp_errors_when_enabled_tide_lookup_has_gap() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = FakeSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        ppp: PppCorrectionLookup {
            tide_enabled: true,
            ..Default::default()
        },
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("enabled tide lookup gap must error");

    assert_missing_correction(err, sat, MissingCorrection::SolidEarthTide);
}

#[test]
fn float_ppp_errors_when_enabled_windup_lookup_has_gap() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = FakeSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        ppp: PppCorrectionLookup {
            windup_enabled: true,
            ..Default::default()
        },
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("enabled phase wind-up lookup gap must error");

    assert_missing_correction(err, sat, MissingCorrection::PhaseWindup);
}

#[test]
fn float_ppp_errors_when_enabled_satellite_antenna_pco_lookup_has_gap() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = FakeSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        ppp: PppCorrectionLookup {
            satellite_antenna_enabled: true,
            ..Default::default()
        },
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("enabled satellite antenna PCO lookup gap must error");

    assert_missing_correction(err, sat, MissingCorrection::SatelliteAntennaPco);
}

#[test]
fn float_ppp_errors_when_enabled_satellite_antenna_pcv_lookup_has_gap() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = FakeSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        ppp: PppCorrectionLookup {
            satellite_antenna_enabled: true,
            sat_pco_ecef: BTreeMap::from([((sat, 0), [0.0, 0.0, 0.0])]),
            ..Default::default()
        },
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("enabled satellite antenna PCV lookup gap must error");

    assert_missing_correction(err, sat, MissingCorrection::SatelliteAntennaPcv);
}

#[test]
fn float_ppp_errors_when_enabled_receiver_antenna_frequency_is_missing() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = FakeSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        receiver_antenna: Some(ReceiverAntennaOptions {
            freq1_label: "G01".to_string(),
            freq1_hz: F_L1_HZ,
            freq2_label: "G02".to_string(),
            freq2_hz: gps_l2_hz(),
            frequencies: Vec::new(),
        }),
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("enabled receiver antenna frequency gap must error");

    assert_missing_correction(
        err,
        sat,
        MissingCorrection::ReceiverAntennaFrequency("G01".to_string()),
    );
}

#[test]
fn float_ppp_errors_when_enabled_receiver_antenna_pcv_grid_is_empty() {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite id");
    let source = FakeSource {
        states: BTreeMap::from([(sat, [20_200_000.0, 13_000_000.0, 21_500_000.0])]),
    };
    let epoch = single_obs_clock_epoch(sat);
    let corrections = RangeCorrections {
        receiver_antenna: Some(ReceiverAntennaOptions {
            freq1_label: "G01".to_string(),
            freq1_hz: F_L1_HZ,
            freq2_label: "G02".to_string(),
            freq2_hz: gps_l2_hz(),
            frequencies: vec![
                ReceiverAntennaFrequency {
                    label: "G01".to_string(),
                    pco_m: [0.0, 0.0, 0.0],
                    pcv_samples: Vec::new(),
                },
                ReceiverAntennaFrequency {
                    label: "G02".to_string(),
                    pco_m: [0.0, 0.0, 0.0],
                    pcv_samples: Vec::new(),
                },
            ],
        }),
        ..RangeCorrections::disabled()
    };
    let err = solve_float_epoch(
        &source,
        epoch.clone(),
        single_obs_clock_state(&epoch),
        single_obs_clock_config(corrections),
    )
    .expect_err("enabled receiver antenna empty PCV grid must error");

    assert_missing_correction(
        err,
        sat,
        MissingCorrection::ReceiverAntennaPcv("G01".to_string()),
    );
}

fn ppp_dual_epochs(slip: bool) -> Vec<DualFrequencyEpoch> {
    (0..3)
        .map(|epoch_idx| DualFrequencyEpoch {
            gap_time_s: Some(epoch_idx as f64 * 30.0),
            observations: (0..4)
                .map(|sat_idx| {
                    let slip_cycles = if slip && sat_idx == 0 && epoch_idx >= 1 {
                        8.0
                    } else {
                        0.0
                    };
                    let lli1 = if slip && sat_idx == 0 && epoch_idx == 1 {
                        Some(1)
                    } else {
                        None
                    };
                    ppp_dual_observation(sat_idx, epoch_idx, slip_cycles, lli1)
                })
                .collect(),
        })
        .collect()
}

fn ppp_dual_observation(
    sat_idx: usize,
    epoch_idx: usize,
    slip_cycles: f64,
    lli1: Option<i64>,
) -> DualFrequencyObservation {
    let satellite_id = format!("G{:02}", sat_idx + 1);
    let base = 23_000_000.0 + epoch_idx as f64 * 200.0 + sat_idx as f64 * 500.0;
    let n1 = 80_000.0 + sat_idx as f64 * 37.0 + slip_cycles;
    let nw = 5.0 + sat_idx as f64;
    let n2 = 80_000.0 + sat_idx as f64 * 37.0 - nw;
    let lambda1 = C_M_S / F_L1_HZ;
    let f2_hz = gps_l2_hz();
    let lambda2 = C_M_S / f2_hz;
    DualFrequencyObservation {
        satellite_id: satellite_id.clone(),
        ambiguity_id: satellite_id,
        p1_m: base,
        p2_m: base,
        phi1_cyc: (base + n1 * lambda1) / lambda1,
        phi2_cyc: (base + n2 * lambda2) / lambda2,
        f1_hz: F_L1_HZ,
        f2_hz,
        lli1,
        lli2: None,
    }
}

#[test]
fn widelane_fixed_prep_pins_split_and_if_bits() {
    let result = prepare_widelane_fixed_epochs(
        &ppp_dual_epochs(true),
        WideLanePrepOptions {
            min_epochs: 2,
            tolerance_cycles: 0.01,
        },
        CycleSlipPolicy::SplitArc,
        CycleSlipOptions {
            gf_threshold_m: 0.05,
            mw_threshold_cycles: 4.0,
            min_arc_gap_s: 1_000.0,
        },
    )
    .unwrap();

    assert_eq!(
        result.wide_lane_cycles,
        BTreeMap::from([
            ("G01#2".to_string(), 13),
            ("G02".to_string(), 6),
            ("G03".to_string(), 7),
            ("G04".to_string(), 8),
        ])
    );
    assert_eq!(result.dropped_sats, Vec::<String>::new());
    assert_eq!(
        result.split_arcs,
        vec![PppSplitArc {
            satellite_id: "G01".to_string(),
            ambiguity_id: "G01#2".to_string(),
            start_epoch_index: 1,
            end_epoch_index: 2,
            n_epochs: 2,
        }]
    );
    assert_eq!(
        result
            .wavelengths_m
            .iter()
            .map(|(sat, value)| (sat.as_str(), value.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            ("G01#2", 0x3fbb614bed5136b9),
            ("G02", 0x3fbb614bed5136b9),
            ("G03", 0x3fbb614bed5136b9),
            ("G04", 0x3fbb614bed5136b9),
        ]
    );
    assert_eq!(
        result
            .offsets_m
            .iter()
            .map(|(sat, value)| (sat.as_str(), value.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            ("G01#2", 0x4013a10c147d0bf0),
            ("G02", 0x40021e814dfd4618),
            ("G03", 0x40052396dafcd1c7),
            ("G04", 0x400828ac67fc5d76),
        ]
    );
    assert_eq!(
        result
            .epochs
            .iter()
            .flat_map(|epoch| {
                epoch.observations.iter().map(move |obs| {
                    (
                        epoch.epoch_index,
                        obs.satellite_id.as_str(),
                        obs.ambiguity_id.as_str(),
                        obs.code_m.to_bits(),
                        obs.phase_m.to_bits(),
                    )
                })
            })
            .collect::<Vec<_>>(),
        vec![
            (0, "G02", "G02", 0x4175ef5b40000000, 0x4175f17267e0f54a),
            (0, "G03", "G03", 0x4175ef7a80000000, 0x4175f191ed3c1ffa),
            (0, "G04", "G04", 0x4175ef99c0000000, 0x4175f1b172974aa8),
            (1, "G01", "G01#2", 0x4175ef4880000000, 0x4175f15fa087c962),
            (1, "G02", "G02", 0x4175ef67c0000000, 0x4175f17ee7e0f54a),
            (1, "G03", "G03", 0x4175ef8700000000, 0x4175f19e6d3c1ffa),
            (1, "G04", "G04", 0x4175efa640000000, 0x4175f1bdf2974aa8),
            (2, "G01", "G01#2", 0x4175ef5500000000, 0x4175f16c2087c962),
            (2, "G02", "G02", 0x4175ef7440000000, 0x4175f18b67e0f54a),
            (2, "G03", "G03", 0x4175ef9380000000, 0x4175f1aaed3c1ffa),
            (2, "G04", "G04", 0x4175efb2c0000000, 0x4175f1ca72974aa8),
        ]
    );
}

#[test]
fn widelane_fixed_prep_pins_error_and_drop_policies() {
    let epochs = ppp_dual_epochs(true);
    let options = WideLanePrepOptions {
        min_epochs: 2,
        tolerance_cycles: 0.01,
    };
    let slip_options = CycleSlipOptions {
        gf_threshold_m: 0.05,
        mw_threshold_cycles: 4.0,
        min_arc_gap_s: 1_000.0,
    };

    assert_eq!(
        prepare_widelane_fixed_epochs(&epochs, options, CycleSlipPolicy::Error, slip_options),
        Err(WideLanePrepError::CycleSlipDetected {
            satellite_id: "G01".to_string(),
            epoch_index: 1,
            reasons: vec![
                SlipReason::Lli,
                SlipReason::GeometryFree,
                SlipReason::MelbourneWubbena,
            ],
        })
    );

    let dropped = prepare_widelane_fixed_epochs(
        &epochs,
        options,
        CycleSlipPolicy::DropSatellite,
        slip_options,
    )
    .unwrap();
    assert_eq!(dropped.dropped_sats, vec!["G01".to_string()]);
    assert_eq!(
        dropped.wide_lane_cycles,
        BTreeMap::from([
            ("G02".to_string(), 6),
            ("G03".to_string(), 7),
            ("G04".to_string(), 8),
        ])
    );
}

#[test]
fn float_cycle_slip_split_tags_are_core_owned() {
    let epochs = ppp_dual_epochs(true)
        .into_iter()
        .map(|epoch| FloatCycleSlipEpoch {
            gap_time_s: epoch.gap_time_s,
            observations: epoch
                .observations
                .into_iter()
                .map(|raw| FloatCycleSlipObservation {
                    satellite_id: raw.satellite_id.clone(),
                    ambiguity_id: raw.satellite_id.clone(),
                    raw: Some(raw),
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let tagged = split_float_cycle_slip_epochs(
        &epochs,
        CycleSlipOptions {
            gf_threshold_m: 0.05,
            mw_threshold_cycles: 4.0,
            min_arc_gap_s: 1_000.0,
        },
    );

    assert_eq!(
        tagged
            .iter()
            .map(|epoch| {
                epoch
                    .observations
                    .iter()
                    .map(|obs| (obs.satellite_id.as_str(), obs.ambiguity_id.as_str()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>(),
        vec![
            vec![
                ("G01", "G01#1"),
                ("G02", "G02"),
                ("G03", "G03"),
                ("G04", "G04")
            ],
            vec![
                ("G01", "G01#2"),
                ("G02", "G02"),
                ("G03", "G03"),
                ("G04", "G04")
            ],
            vec![
                ("G01", "G01#2"),
                ("G02", "G02"),
                ("G03", "G03"),
                ("G04", "G04")
            ],
        ]
    );

    let no_slip = split_float_cycle_slip_epochs(
        &ppp_dual_epochs(false)
            .into_iter()
            .map(|epoch| FloatCycleSlipEpoch {
                gap_time_s: epoch.gap_time_s,
                observations: epoch
                    .observations
                    .into_iter()
                    .map(|raw| FloatCycleSlipObservation {
                        satellite_id: raw.satellite_id.clone(),
                        ambiguity_id: raw.satellite_id.clone(),
                        raw: Some(raw),
                    })
                    .collect(),
            })
            .collect::<Vec<_>>(),
        CycleSlipOptions {
            gf_threshold_m: 0.05,
            mw_threshold_cycles: 4.0,
            min_arc_gap_s: 1_000.0,
        },
    );
    assert_eq!(
        no_slip[0]
            .observations
            .iter()
            .map(|obs| (obs.satellite_id.as_str(), obs.ambiguity_id.as_str()))
            .collect::<Vec<_>>(),
        vec![
            ("G01", "G01"),
            ("G02", "G02"),
            ("G03", "G03"),
            ("G04", "G04")
        ]
    );
}

#[test]
fn static_float_solver_recovers_synthetic_arc() {
    let sats = [
        (1, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
        (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
        (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
        (4, [-18_700_000.0, -18_200_000.0, 22_000_000.0]),
        (5, [23_500_000.0, 3_200_000.0, -18_900_000.0]),
        (6, [-7_500_000.0, 25_800_000.0, -16_000_000.0]),
    ];
    let ids: Vec<GnssSatelliteId> = sats
        .iter()
        .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id"))
        .collect();
    let source = FakeSource {
        states: ids
            .iter()
            .zip(sats.iter())
            .map(|(id, (_, pos))| (*id, *pos))
            .collect(),
    };
    let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
    let clocks = [12.5, -8.25, 4.0];
    let ambiguities: BTreeMap<String, f64> = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
        .collect();
    let mut epochs = Vec::new();
    for (epoch_idx, clock) in clocks.iter().enumerate() {
        let observations = ids
            .iter()
            .map(|id| {
                let pred = predict(
                    &source,
                    *id,
                    truth,
                    epoch_idx as f64 * 900.0,
                    PredictOptions {
                        carrier_hz: F_L1_HZ,
                        light_time: true,
                        sagnac: true,
                    },
                )
                .unwrap();
                let code = pred.geometric_range_m + clock;
                let ambiguity = ambiguities.get(&id.to_string()).copied().unwrap();
                FloatObservation {
                    sat: *id,
                    satellite_id: id.to_string(),
                    ambiguity_id: id.to_string(),
                    code_m: code,
                    phase_m: code + ambiguity,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                    glonass_channel: None,
                }
            })
            .collect();
        epochs.push(FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24,
                hour: 12,
                minute: epoch_idx as u8 * 15,
                second: 0.0,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5 + epoch_idx as f64 * 900.0 / crate::constants::SECONDS_PER_DAY,
            t_rx_j2000_s: epoch_idx as f64 * 900.0,
            observations,
        });
    }
    let initial = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0; epochs.len()],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let solution = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 8,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .unwrap();
    assert_eq!(
        solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(solution.ztd_residual_m, None);
    assert!(solution.code_rms_m < 1.0e-8);
    assert!(solution.phase_rms_m < 1.0e-8);
    assert!(solution.weighted_rms_m < 1.0e-6);
    let err = norm3(sub3(solution.position_m, truth));
    assert!(err < 1.0e-3, "position error {err}");
    for (actual, expected) in solution.epoch_clocks_m.iter().zip(clocks) {
        assert!((actual - expected).abs() < 1.0e-4);
    }
    for (sat, expected) in ambiguities {
        assert!((solution.ambiguities_m[&sat] - expected).abs() < 1.0e-4);
    }
    assert_position_covariance_positive_definite(&solution.formal_position_covariance);
    assert_position_covariance_scaled_by_factor(
        &solution.position_covariance,
        &solution.formal_position_covariance,
        solution.position_covariance_scale_factor,
    );
    assert_temporal_covariance_not_smaller(&solution);
    assert_eq!(solution.status, FloatStatus::StateTolerance);
    assert!(solution.converged);
}

#[test]
fn elevation_cutoff_none_preserves_static_float_fixture_bits() {
    let (source, epochs, initial, _) = ppp_elevation_cutoff_arc();
    let config = ppp_cutoff_config(None);
    assert!(!config.tropo.estimate_tropo_gradients);
    let solution = solve_float_epochs(&source, &epochs, initial, config).unwrap();
    assert_eq!(
        solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(solution.tropo_gradient_north_m, None);
    assert_eq!(solution.tropo_gradient_east_m, None);
    assert_eq!(solution.tropo_gradient_covariance_m2, None);
    assert_eq!(solution.formal_tropo_gradient_covariance_m2, None);
    assert_eq!(
        ppp_float_solution_bits(&solution),
        vec![
            4708606483430899711,
            4452733082576154772,
            4453493932956835639,
            4623226492472189013,
            13844205992025595820,
            4616189618053415252,
            4598175219544437634,
            4599976659423301089,
            4601778099233554315,
            4603129179142392862,
            4604029899052293746,
            4604930618990457261,
            0,
            0,
            4605975682916587671,
            4635794528945706806,
            0,
            0,
            4605553524466321826,
            4635464717656436615,
            0,
            0,
            4605075134482219749,
            4635090975481356867,
            0,
            0,
            4604544223951464880,
            4634676201629204626,
            0,
            0,
            4595424520664219441,
            4625581108599069534,
            0,
            0,
            4590944325920908238,
            4621095794037370361,
            0,
            0,
            4605975682916587671,
            4635794528945706806,
            0,
            0,
            4605553524466321826,
            4635464717656436615,
            0,
            0,
            4605075134482219749,
            4635090975481356867,
            0,
            0,
            4604544223951464880,
            4634676201629204626,
            0,
            0,
            4595424520664219441,
            4625581108599069534,
            0,
            0,
            4590944325920908238,
            4621095794037370361,
            0,
            0,
            4605975682916587671,
            4635794528945706806,
            0,
            0,
            4605553524466321826,
            4635464717656436615,
            0,
            0,
            4605075134482219749,
            4635090975481356867,
            0,
            0,
            4604544223951464880,
            4634676201629204626,
            0,
            0,
            4595424520664219441,
            4625581108599069534,
            0,
            0,
            4590944325920908238,
            4621095794037370361,
            0,
            0,
            0,
        ]
    );
}

#[test]
fn elevation_cutoff_removes_low_satellites_before_solve() {
    let (source, epochs, initial, low_sats) = ppp_elevation_cutoff_arc();
    let low_count = epochs[0]
        .observations
        .iter()
        .filter(|obs| {
            let pred = predict(
                &source,
                obs.sat,
                initial.position_m,
                epochs[0].t_rx_j2000_s,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: true,
                    sagnac: true,
                },
            )
            .unwrap();
            pred.elevation_deg < 15.0
        })
        .count();
    assert_eq!(low_count, 2);
    assert_eq!(low_sats, ["G05", "G06"]);

    let no_cutoff =
        solve_float_epochs(&source, &epochs, initial.clone(), ppp_cutoff_config(None)).unwrap();
    let cutoff =
        solve_float_epochs(&source, &epochs, initial, ppp_cutoff_config(Some(15.0))).unwrap();

    assert_eq!(no_cutoff.residuals_m.len(), 18);
    assert_eq!(cutoff.residuals_m.len(), 12);
    assert_eq!(
        no_cutoff.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(cutoff.used_sats, ["G01", "G02", "G03", "G04"]);
    assert!(cutoff.converged);
    assert_eq!(cutoff.status, FloatStatus::StateTolerance);
}

#[test]
fn aggressive_elevation_cutoff_returns_typed_error() {
    let (source, epochs, initial, _) = ppp_elevation_cutoff_arc();
    let err = solve_float_epochs(
        &source,
        &epochs,
        initial.clone(),
        ppp_cutoff_config(Some(89.0)),
    )
    .expect_err("over-masked PPP solve should fail before normal assembly");
    assert_eq!(
        err,
        FloatSolveError::InsufficientObservationsAfterElevationCutoff {
            cutoff_deg: 89.0,
            retained_observations: 0,
            required_observations: 4,
        }
    );

    let float_solution =
        solve_float_epochs(&source, &epochs, initial, ppp_cutoff_config(None)).unwrap();
    let wavelengths_m = float_solution
        .used_sats
        .iter()
        .map(|sat| (sat.clone(), 0.190_293_672_798_365))
        .collect();
    let offsets_m = float_solution
        .used_sats
        .iter()
        .map(|sat| (sat.clone(), 0.0))
        .collect();
    let fixed_err = solve_fixed_from_float(
        &source,
        &epochs,
        float_solution,
        FixedSolveConfig {
            weights: ppp_cutoff_config(None).weights,
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 8,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: Some(89.0),
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m,
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("over-masked fixed PPP solve should fail before integer search");
    assert_eq!(
        fixed_err,
        FixedSolveError::Float(
            FloatSolveError::InsufficientObservationsAfterElevationCutoff {
                cutoff_deg: 89.0,
                retained_observations: 0,
                required_observations: 4,
            }
        )
    );
}

#[test]
fn static_float_solver_reports_unit_variance_factor_on_weighted_synthetic_noise() {
    let sats = [
        (1, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
        (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
        (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
        (4, [-18_700_000.0, -18_200_000.0, 22_000_000.0]),
        (5, [23_500_000.0, 3_200_000.0, -18_900_000.0]),
        (6, [-7_500_000.0, 25_800_000.0, -16_000_000.0]),
    ];
    let ids: Vec<GnssSatelliteId> = sats
        .iter()
        .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id"))
        .collect();
    let source = FakeSource {
        states: ids
            .iter()
            .zip(sats.iter())
            .map(|(id, (_, pos))| (*id, *pos))
            .collect(),
    };
    let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
    let ambiguities: BTreeMap<String, f64> = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
        .collect();
    let epoch_count = 20;
    let mut epochs = Vec::new();
    let mut sample_idx = 0;
    for epoch_idx in 0..epoch_count {
        let t_rx_j2000_s = epoch_idx as f64 * 30.0;
        let clock = 12.5 + (epoch_idx % 11) as f64 * 0.15;
        let observations = ids
            .iter()
            .map(|id| {
                let pred = predict(
                    &source,
                    *id,
                    truth,
                    t_rx_j2000_s,
                    PredictOptions {
                        carrier_hz: F_L1_HZ,
                        light_time: true,
                        sagnac: true,
                    },
                )
                .unwrap();
                let code_noise_m = deterministic_unit_noise(sample_idx);
                let phase_noise_m = deterministic_unit_noise(sample_idx + 17) / 100.0;
                sample_idx += 1;
                let code = pred.geometric_range_m + clock;
                let ambiguity = ambiguities.get(&id.to_string()).copied().unwrap();
                FloatObservation {
                    sat: *id,
                    satellite_id: id.to_string(),
                    ambiguity_id: id.to_string(),
                    code_m: code + code_noise_m,
                    phase_m: code + ambiguity + phase_noise_m,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                    glonass_channel: None,
                }
            })
            .collect();
        epochs.push(FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24,
                hour: ((epoch_idx * 30) / 3600) as u8,
                minute: (((epoch_idx * 30) % 3600) / 60) as u8,
                second: ((epoch_idx * 30) % 60) as f64,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5 + t_rx_j2000_s / crate::constants::SECONDS_PER_DAY,
            t_rx_j2000_s,
            observations,
        });
    }
    let initial = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0; epochs.len()],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let solution = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 8,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect("weighted noisy synthetic PPP solve");
    eprintln!(
        "weighted synthetic PPP variance_factor={:.3}",
        solution.posterior_variance_factor
    );

    assert!(
        (0.5..=1.5).contains(&solution.posterior_variance_factor),
        "variance factor {} outside clean synthetic band",
        solution.posterior_variance_factor
    );
    assert_position_covariance_positive_definite(&solution.formal_position_covariance);
    assert_position_covariance_positive_definite(&solution.position_covariance);
    assert_position_covariance_scaled_by_factor(
        &solution.position_covariance,
        &solution.formal_position_covariance,
        solution.position_covariance_scale_factor,
    );
    assert_temporal_covariance_not_smaller(&solution);
}

fn deterministic_unit_noise(index: usize) -> f64 {
    let centered = ((index * 37 + 13) % 101) as f64 - 50.0;
    centered / 29.15
}

#[test]
fn static_float_solver_handles_multi_hundred_epoch_arc() {
    let sats = [
        (1, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
        (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
        (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
        (4, [-18_700_000.0, -18_200_000.0, 22_000_000.0]),
        (5, [23_500_000.0, 3_200_000.0, -18_900_000.0]),
        (6, [-7_500_000.0, 25_800_000.0, -16_000_000.0]),
    ];
    let ids: Vec<GnssSatelliteId> = sats
        .iter()
        .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id"))
        .collect();
    let source = FakeSource {
        states: ids
            .iter()
            .zip(sats.iter())
            .map(|(id, (_, pos))| (*id, *pos))
            .collect(),
    };
    let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
    let ambiguities: BTreeMap<String, f64> = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
        .collect();
    let epoch_count = std::env::var("SIDEREON_PPP_TRACT_EPOCHS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(360);
    let mut epochs = Vec::with_capacity(epoch_count);
    for epoch_idx in 0..epoch_count {
        let t_rx_j2000_s = epoch_idx as f64 * 30.0;
        let clock = 12.5 + (epoch_idx % 17) as f64 * 0.1;
        let observations = ids
            .iter()
            .map(|id| {
                let pred = predict(
                    &source,
                    *id,
                    truth,
                    t_rx_j2000_s,
                    PredictOptions {
                        carrier_hz: F_L1_HZ,
                        light_time: true,
                        sagnac: true,
                    },
                )
                .unwrap();
                let code = pred.geometric_range_m + clock;
                let ambiguity = ambiguities.get(&id.to_string()).copied().unwrap();
                FloatObservation {
                    sat: *id,
                    satellite_id: id.to_string(),
                    ambiguity_id: id.to_string(),
                    code_m: code,
                    phase_m: code + ambiguity,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                    glonass_channel: None,
                }
            })
            .collect();
        let total_s = epoch_idx * 30;
        epochs.push(FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24 + (total_s / 86_400) as u8,
                hour: ((total_s / 3600) % 24) as u8,
                minute: ((total_s % 3600) / 60) as u8,
                second: (total_s % 60) as f64,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5 + t_rx_j2000_s / crate::constants::SECONDS_PER_DAY,
            t_rx_j2000_s,
            observations,
        });
    }
    let initial = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0; epochs.len()],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let start = std::time::Instant::now();
    let solution = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 8,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect("multi-hundred epoch static PPP solve");
    let elapsed = start.elapsed();
    eprintln!("synthetic static PPP {epoch_count} epochs solved in {elapsed:?}");

    assert!(
        elapsed < std::time::Duration::from_secs(30),
        "multi-hundred epoch static PPP solve took {elapsed:?}"
    );
    assert!(norm3(sub3(solution.position_m, truth)) < 1.0e-3);
    assert!(solution.weighted_rms_m < 1.0e-6);
    assert_position_covariance_positive_definite(&solution.formal_position_covariance);
    assert_position_covariance_scaled_by_factor(
        &solution.position_covariance,
        &solution.formal_position_covariance,
        solution.position_covariance_scale_factor,
    );
    assert_eq!(solution.status, FloatStatus::StateTolerance);
    assert!(solution.converged);
}

#[test]
fn static_float_solver_rejects_short_clock_vector() {
    let (source, epochs, mut initial, _ambiguity_ids) = ppp_row_trace_arc();
    initial.clocks_m.pop();

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("short PPP clock vector must be rejected");

    assert_invalid_clock_count(err, epochs.len(), epochs.len() - 1);
}

#[test]
fn static_float_solver_rejects_nan_tolerance() {
    let (source, epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: f64::NAN,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("NaN PPP tolerance must be rejected");

    assert_invalid_solve_option(err, "position_tolerance_m", "must be finite");
}

#[test]
fn static_float_solver_rejects_iteration_cap_and_nonpositive_tolerances() {
    let (source, epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial.clone(),
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 0,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("zero PPP max_iterations must be rejected");
    assert_invalid_solve_option(err, "max_iterations", "must be positive");

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial.clone(),
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: usize::MAX,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("oversized PPP max_iterations must be rejected");
    assert_invalid_solve_option(err, "max_iterations", "exceeds the PPP iteration cap");

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial.clone(),
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 0.0,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("zero PPP tolerance must be rejected");
    assert_invalid_solve_option(err, "position_tolerance_m", "must be positive");

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: -1.0,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("negative PPP tolerance must be rejected");
    assert_invalid_solve_option(err, "position_tolerance_m", "must be positive");
}

#[test]
fn static_float_solver_rejects_nan_observation() {
    let (source, mut epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();
    epochs[0].observations[0].code_m = f64::NAN;

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("NaN PPP observation must be rejected");

    assert_invalid_input(err, "ppp observation code_m", "not finite");
}

#[test]
fn static_float_solver_rejects_nan_initial_state() {
    let (source, epochs, mut initial, _ambiguity_ids) = ppp_row_trace_arc();
    initial.position_m[0] = f64::NAN;

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("NaN PPP initial state must be rejected");

    assert_invalid_input(err, "ppp state position_m", "not finite");
}

#[test]
fn static_float_solver_rejects_zero_measurement_weight() {
    let (source, epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: MeasurementWeights {
                code: 0.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("non-positive PPP measurement weight must be rejected");

    assert_invalid_input(err, "ppp measurement weight code", "not positive");
}

#[test]
fn static_float_solver_rejects_nonfinite_measurement_weights() {
    let (source, epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();

    for (weights, field) in [
        (
            MeasurementWeights {
                code: f64::NAN,
                phase: 100.0,
                elevation_weighting: false,
            },
            "ppp measurement weight code",
        ),
        (
            MeasurementWeights {
                code: 1.0,
                phase: f64::INFINITY,
                elevation_weighting: false,
            },
            "ppp measurement weight phase",
        ),
    ] {
        let err = solve_float_epochs(
            &source,
            &epochs,
            initial.clone(),
            FloatSolveConfig {
                weights,
                tropo: TroposphereOptions::disabled(),
                corrections: RangeCorrections::disabled(),
                opts: FloatSolveOptions {
                    max_iterations: 1,
                    position_tolerance_m: 1.0e-4,
                    clock_tolerance_m: 1.0e-4,
                    ambiguity_tolerance_m: 1.0e-4,
                    ztd_tolerance_m: 1.0e-4,
                },
                elevation_cutoff_deg: None,
                residual_screen: false,
                estimate_residual_ionosphere: false,
            },
        )
        .expect_err("non-finite PPP measurement weight must be rejected");

        assert_invalid_input(err, field, "not finite");
    }
}

#[test]
fn static_float_solver_ignores_unused_met_when_troposphere_disabled() {
    let (source, epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();

    let standard = solve_float_epochs(
        &source,
        &epochs,
        initial.clone(),
        ppp_row_trace_float_config(TroposphereOptions::disabled()),
    )
    .expect("solve with disabled troposphere and standard met");

    let mut zero_met = TroposphereOptions::disabled();
    zero_met.met = crate::tropo::Met::new_unchecked(0.0, 0.0, 0.0);
    let placeholder = solve_float_epochs(
        &source,
        &epochs,
        initial,
        ppp_row_trace_float_config(zero_met),
    )
    .expect("solve with disabled troposphere and unused zero met");

    assert_eq!(placeholder, standard);
}

#[test]
fn static_float_solver_ignores_ztd_estimate_when_troposphere_disabled() {
    let (source, epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();

    let standard = solve_float_epochs(
        &source,
        &epochs,
        initial.clone(),
        ppp_row_trace_float_config(TroposphereOptions::disabled()),
    )
    .expect("solve with disabled troposphere");

    let tropo = TroposphereOptions {
        estimate_ztd: true,
        ..TroposphereOptions::disabled()
    };
    assert_eq!(ztd_unknown_count(tropo), 0);

    let solution = solve_float_epochs(&source, &epochs, initial, ppp_row_trace_float_config(tropo))
        .expect("disabled troposphere must not estimate a degenerate ZTD column");

    assert_eq!(solution, standard);
}

#[test]
fn static_float_design_rows_keep_enabled_ztd_estimation_column() {
    let (source, epochs, state, ambiguity_ids) = ppp_row_trace_arc();
    let tropo = TroposphereOptions {
        enabled: true,
        estimate_ztd: true,
        ..TroposphereOptions::disabled()
    };
    assert_eq!(ztd_unknown_count(tropo), 1);
    let corrections = RangeCorrections::disabled();
    let ctx = ModelContext {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo,
        corrections: &corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
    };
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };

    let rows = super::rows::build_rows(ctx, &epochs, &binding, &state).unwrap();

    let ztd_column = 3 + epochs.len();
    assert_eq!(rows[0].h.len(), 3 + epochs.len() + 1 + ambiguity_ids.len());
    assert!(rows.iter().any(|row| row.h[ztd_column] > 0.0));
}

#[test]
fn static_float_solver_recovers_injected_tropo_gradients_and_partials() {
    let injected = [0.012, -0.007];
    let (source, epochs, initial, _truth, _ambiguities) = tropo_gradient_synthetic_arc(injected);
    let solution = solve_float_epochs(
        &source,
        &epochs,
        initial.clone(),
        tropo_gradient_float_config(true),
    )
    .expect("gradient synthetic PPP solve");

    let north = solution
        .tropo_gradient_north_m
        .expect("north gradient estimate");
    let east = solution
        .tropo_gradient_east_m
        .expect("east gradient estimate");
    let north_error = north - injected[0];
    let east_error = east - injected[1];
    eprintln!(
        "synthetic gradient recovery north={north:.6} east={east:.6} north_error={north_error:.3e} east_error={east_error:.3e}"
    );
    assert_abs_close(north, injected[0], 2.0e-5, "north gradient recovery");
    assert_abs_close(east, injected[1], 2.0e-5, "east gradient recovery");
    assert!(solution.tropo_gradient_covariance_m2.is_some());
    assert!(solution.formal_tropo_gradient_covariance_m2.is_some());

    let ambiguity_ids = test_ambiguity_ids(&epochs);
    let corrections = RangeCorrections::disabled();
    let ctx = ModelContext {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo: tropo_gradient_options(true),
        corrections: &corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
    };
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &initial.ambiguities_m,
    };
    let rows =
        super::rows::build_rows(ctx, &epochs, &binding, &initial).expect("gradient design rows");
    let gradient_column = 3 + epochs.len();
    finite_difference_gradient_partial(ctx, &epochs, &binding, &initial, 0, gradient_column, true);
    finite_difference_gradient_partial(
        ctx,
        &epochs,
        &binding,
        &initial,
        0,
        gradient_column + 1,
        false,
    );
    finite_difference_gradient_partial(ctx, &epochs, &binding, &initial, 1, gradient_column, true);
    finite_difference_gradient_partial(
        ctx,
        &epochs,
        &binding,
        &initial,
        1,
        gradient_column + 1,
        false,
    );
    assert!(rows[0].h[gradient_column].abs() > 0.0);
    assert!(rows[0].h[gradient_column + 1].abs() > 0.0);
}

#[test]
fn static_float_zero_tropo_gradient_matches_no_gradient_solve() {
    let (source, epochs, initial, _truth, _ambiguities) = tropo_gradient_synthetic_arc([0.0, 0.0]);
    let enabled = solve_float_epochs(
        &source,
        &epochs,
        initial.clone(),
        tropo_gradient_float_config(true),
    )
    .expect("zero-gradient enabled solve");
    let disabled = solve_float_epochs(
        &source,
        &epochs,
        initial,
        tropo_gradient_float_config(false),
    )
    .expect("zero-gradient disabled solve");

    let north = enabled
        .tropo_gradient_north_m
        .expect("north gradient estimate");
    let east = enabled
        .tropo_gradient_east_m
        .expect("east gradient estimate");
    eprintln!("zero synthetic gradient recovery north={north:.3e} east={east:.3e}");
    assert_abs_close(north, 0.0, 2.0e-5, "zero north gradient");
    assert_abs_close(east, 0.0, 2.0e-5, "zero east gradient");
    assert_eq!(disabled.tropo_gradient_north_m, None);
    assert_eq!(disabled.tropo_gradient_east_m, None);
    assert_vec3_close(
        enabled.position_m,
        disabled.position_m,
        2.0e-5,
        "zero-gradient position",
    );
    for (enabled_clock, disabled_clock) in
        enabled.epoch_clocks_m.iter().zip(&disabled.epoch_clocks_m)
    {
        assert_abs_close(
            *enabled_clock,
            *disabled_clock,
            2.0e-5,
            "zero-gradient clock",
        );
    }
}

fn tropo_gradient_options(estimate_tropo_gradients: bool) -> TroposphereOptions {
    TroposphereOptions {
        enabled: true,
        estimate_ztd: false,
        estimate_tropo_gradients,
        met: crate::tropo::Met::new(1013.25, 288.15, 0.5).expect("valid met"),
        mapping: TropoMapping::Niell,
    }
}

fn tropo_gradient_float_config(estimate_tropo_gradients: bool) -> FloatSolveConfig {
    FloatSolveConfig {
        weights: ppp_row_trace_weights(),
        tropo: tropo_gradient_options(estimate_tropo_gradients),
        corrections: RangeCorrections::disabled(),
        opts: FloatSolveOptions {
            max_iterations: 12,
            position_tolerance_m: 1.0e-7,
            clock_tolerance_m: 1.0e-7,
            ambiguity_tolerance_m: 1.0e-7,
            ztd_tolerance_m: 1.0e-7,
        },
        elevation_cutoff_deg: None,
        residual_screen: false,
        estimate_residual_ionosphere: false,
    }
}

fn tropo_gradient_synthetic_arc(
    injected_gradient_m: [f64; 2],
) -> (
    FakeSource,
    Vec<FloatEpoch>,
    FloatState,
    [f64; 3],
    BTreeMap<String, f64>,
) {
    let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
    let az_el = [
        (15.0, 38.0),
        (55.0, 52.0),
        (105.0, 33.0),
        (145.0, 47.0),
        (205.0, 31.0),
        (250.0, 56.0),
        (300.0, 42.0),
        (335.0, 64.0),
    ];
    let ids = (1..=az_el.len())
        .map(|prn| GnssSatelliteId::new(GnssSystem::Gps, prn as u8).expect("valid GPS id"))
        .collect::<Vec<_>>();
    let states = ids
        .iter()
        .zip(az_el)
        .map(|(id, (az_deg, el_deg))| (*id, synthetic_satellite_position(truth, az_deg, el_deg)))
        .collect::<BTreeMap<_, _>>();
    let source = FakeSource { states };
    let ambiguities = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.to_string(), 0.35 + idx as f64 * 0.08))
        .collect::<BTreeMap<_, _>>();
    let corrections = RangeCorrections::disabled();
    let tropo = tropo_gradient_options(true);
    let epoch_count = 6;
    let truth_state = FloatState {
        position_m: truth,
        clocks_m: vec![0.0; epoch_count],
        ambiguities_m: ambiguities.clone(),
        ztd_m: 0.0,
        tropo_gradient_north_m: injected_gradient_m[0],
        tropo_gradient_east_m: injected_gradient_m[1],
        residual_ionosphere_m: BTreeMap::new(),
    };
    let mut epochs = Vec::new();
    for epoch_idx in 0..epoch_count {
        let t_rx_j2000_s = epoch_idx as f64 * 300.0;
        let clock_m = 4.0 + epoch_idx as f64 * 0.17;
        let mut epoch = FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24,
                hour: 12,
                minute: (epoch_idx * 5) as u8,
                second: 0.0,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5 + t_rx_j2000_s / crate::constants::SECONDS_PER_DAY,
            t_rx_j2000_s,
            observations: Vec::new(),
        };
        for id in &ids {
            let mut obs = FloatObservation {
                sat: *id,
                satellite_id: id.to_string(),
                ambiguity_id: id.to_string(),
                code_m: 0.0,
                phase_m: 0.0,
                freq1_hz: 0.0,
                freq2_hz: 0.0,
                glonass_channel: None,
            };
            let pred = predict(
                &source,
                *id,
                truth,
                t_rx_j2000_s,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: true,
                    sagnac: true,
                },
            )
            .expect("synthetic prediction");
            let tropo_model = super::model::model_troposphere(&pred, truth, &epoch, tropo)
                .expect("synthetic tropo model");
            let corrections_m = super::model::range_corrections_m(
                &pred,
                truth,
                epoch_idx,
                &obs,
                &tropo_model,
                &truth_state,
                &corrections,
            )
            .expect("synthetic range corrections");
            let model_range_m = pred.geometric_range_m + clock_m + corrections_m;
            let ambiguity_m = ambiguities[&id.to_string()];
            obs.code_m = model_range_m;
            obs.phase_m = model_range_m + ambiguity_m;
            epoch.observations.push(obs);
        }
        epochs.push(epoch);
    }
    let initial = FloatState {
        position_m: [truth[0] + 3.0, truth[1] - 2.5, truth[2] + 1.5],
        clocks_m: vec![0.0; epochs.len()],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    (source, epochs, initial, truth, ambiguities)
}

fn synthetic_satellite_position(receiver_m: [f64; 3], az_deg: f64, el_deg: f64) -> [f64; 3] {
    let up = unit3(receiver_m).expect("nonzero receiver vector");
    let east = unit3([-receiver_m[1], receiver_m[0], 0.0]).expect("non-polar receiver");
    let north = cross3(up, east);
    let az = az_deg.to_radians();
    let el = el_deg.to_radians();
    let horizontal = el.cos();
    let los = add3(
        add3(
            scale3(north, horizontal * az.cos()),
            scale3(east, horizontal * az.sin()),
        ),
        scale3(up, el.sin()),
    );
    add3(receiver_m, scale3(los, 26_000_000.0))
}

fn test_ambiguity_ids(epochs: &[FloatEpoch]) -> Vec<AmbiguityId> {
    epochs
        .iter()
        .flat_map(|epoch| {
            epoch
                .observations
                .iter()
                .map(|obs| AmbiguityId::new(obs.ambiguity_id.clone()))
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn finite_difference_gradient_partial(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    binding: &super::rows::AmbiguityBinding<'_>,
    state: &FloatState,
    row_idx: usize,
    column_idx: usize,
    north: bool,
) {
    let eps_m = 1.0e-2;
    let base_rows =
        super::rows::build_rows(ctx, epochs, binding, state).expect("base gradient rows");
    let mut minus = state.clone();
    let mut plus = state.clone();
    if north {
        minus.tropo_gradient_north_m -= eps_m;
        plus.tropo_gradient_north_m += eps_m;
    } else {
        minus.tropo_gradient_east_m -= eps_m;
        plus.tropo_gradient_east_m += eps_m;
    }
    let minus_rows =
        super::rows::build_rows(ctx, epochs, binding, &minus).expect("minus gradient rows");
    let plus_rows =
        super::rows::build_rows(ctx, epochs, binding, &plus).expect("plus gradient rows");
    let finite_difference_model_partial =
        -(plus_rows[row_idx].y - minus_rows[row_idx].y) / (2.0 * eps_m);
    assert_abs_close(
        finite_difference_model_partial,
        base_rows[row_idx].h[column_idx],
        1.0e-6,
        "gradient finite-difference partial",
    );
}

fn assert_abs_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    let delta = (actual - expected).abs();
    assert!(
        delta <= tolerance,
        "{label}: actual {actual:.12e}, expected {expected:.12e}, delta {delta:.3e}, tolerance {tolerance:.3e}"
    );
}

fn assert_vec3_close(actual: [f64; 3], expected: [f64; 3], tolerance: f64, label: &str) {
    for idx in 0..3 {
        assert_abs_close(actual[idx], expected[idx], tolerance, label);
        assert_abs_close(expected[idx], actual[idx], tolerance, label);
    }
}

#[test]
fn static_float_rows_apply_ssr_code_and_phase_biases_with_expected_signs() {
    let (source, mut epochs, mut state, _ambiguity_ids) = ppp_row_trace_arc();
    epochs[0].observations.truncate(1);
    epochs.truncate(1);
    state.clocks_m.truncate(1);
    state.ambiguities_m = initial_ambiguities(&epochs);
    let obs = &epochs[0].observations[0];
    let sat = obs.sat;
    let ambiguity_ids = vec![AmbiguityId::new(obs.ambiguity_id.clone())];

    let base_corrections = RangeCorrections::disabled();
    let base_ctx = ModelContext {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo: TroposphereOptions::disabled(),
        corrections: &base_corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
    };
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let base_rows = super::rows::build_rows(base_ctx, &epochs, &binding, &state).unwrap();

    let code_l1_m = 0.24;
    let code_l2_m = -0.46;
    let phase_l1_cycles = 1.25;
    let phase_l2_cycles = -2.5;
    let phase_l1_m = phase_l1_cycles * C_M_S / F_L1_HZ;
    let phase_l2_m = phase_l2_cycles * C_M_S / F_L2_HZ;
    let has = HasMt1Message {
        header: HasMt1Header {
            toh_s: 0,
            mask: true,
            orbit: false,
            clock_full_set: false,
            clock_subset: false,
            code_bias: true,
            phase_bias: true,
            reserved: 0,
            mask_id: 1,
            iod_set_id: 1,
        },
        mask: Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: sat.system,
                satellites: vec![sat.prn],
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
        }),
        orbit: None,
        clock_full_set: None,
        clock_subset: None,
        code_bias: Some(HasCodeBiasBlock {
            validity_interval: 5,
            records: vec![
                HasCodeBias {
                    sat,
                    signal_id: 0,
                    bias_m: code_l1_m,
                },
                HasCodeBias {
                    sat,
                    signal_id: 9,
                    bias_m: code_l2_m,
                },
            ],
        }),
        phase_bias: Some(HasPhaseBiasBlock {
            validity_interval: 5,
            records: vec![
                HasPhaseBias {
                    sat,
                    signal_id: 0,
                    bias_cycles: phase_l1_cycles,
                    bias_m: phase_l1_m,
                    discontinuity_indicator: 0,
                },
                HasPhaseBias {
                    sat,
                    signal_id: 9,
                    bias_cycles: phase_l2_cycles,
                    bias_m: phase_l2_m,
                    discontinuity_indicator: 0,
                },
            ],
        }),
        padding_bits: Vec::new(),
    };
    let decoded = HasMt1Message::decode(&has.encode()).expect("decode HAS MT1");
    let mut store = SsrCorrectionStore::new();
    let reception = crate::astro::time::model::GnssWeekTow::new(
        crate::astro::time::model::TimeScale::Gst,
        1,
        0.0,
    )
    .unwrap();
    store.ingest_has_mt1(&decoded, reception).unwrap();
    let mut options = SsrPppBiasOptions::default();
    options.per_system.insert(
        sat.system,
        SsrPppBiasSignalPair {
            code1_signal: 0,
            code2_signal: 9,
            phase1_signal: 0,
            phase2_signal: 9,
            freq1_hz: F_L1_HZ,
            freq2_hz: F_L2_HZ,
        },
    );
    let biased_lookup = PppCorrectionLookup::default().with_ssr_biases(&store, &epochs, &options);
    let biased_corrections = RangeCorrections {
        ppp: biased_lookup,
        ..RangeCorrections::disabled()
    };
    let biased_ctx = ModelContext {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo: TroposphereOptions::disabled(),
        corrections: &biased_corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
    };
    let biased_rows = super::rows::build_rows(biased_ctx, &epochs, &binding, &state).unwrap();

    let gamma = F_L1_HZ * F_L1_HZ / (F_L1_HZ * F_L1_HZ - F_L2_HZ * F_L2_HZ);
    let expected_code_if = gamma * code_l1_m - (gamma - 1.0) * code_l2_m;
    let expected_phase_if = gamma * phase_l1_m - (gamma - 1.0) * phase_l2_m;
    let code_delta = biased_rows[0].y - base_rows[0].y;
    let phase_delta = biased_rows[1].y - base_rows[1].y;
    assert!(
        (code_delta - expected_code_if).abs() < 1.0e-8,
        "code delta {code_delta}, expected {expected_code_if}"
    );
    assert!(
        (phase_delta - expected_phase_if).abs() < 1.0e-8,
        "phase delta {phase_delta}, expected {expected_phase_if}"
    );
}

#[test]
fn static_float_design_rows_handle_antimeridian_tropo_receiver() {
    let (source, epochs, mut state, ambiguity_ids) = ppp_row_trace_arc();
    state.position_m = [-6_378_137.0, 0.0, 0.0];
    let tropo = TroposphereOptions {
        enabled: true,
        estimate_ztd: true,
        ..TroposphereOptions::disabled()
    };
    let corrections = RangeCorrections::disabled();
    let ctx = ModelContext {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo,
        corrections: &corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
    };
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::rows::build_rows(ctx, &epochs, &binding, &state)
    }));

    assert!(result.is_ok(), "antimeridian tropo receiver must not panic");
    let rows = result
        .expect("antimeridian tropo receiver should not unwind")
        .expect("antimeridian tropo receiver should build rows");
    assert!(!rows.is_empty());
}

#[test]
fn static_float_design_rows_reject_invalid_tropo_julian_split_without_panic() {
    let (source, mut epochs, state, ambiguity_ids) = ppp_row_trace_arc();
    epochs[0].jd_fraction = 1.0 + f64::EPSILON;
    let tropo = TroposphereOptions {
        enabled: true,
        estimate_ztd: true,
        ..TroposphereOptions::disabled()
    };
    let corrections = RangeCorrections::disabled();
    let ctx = ModelContext {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo,
        corrections: &corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
    };
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        super::rows::build_rows(ctx, &epochs, &binding, &state)
    }));

    assert!(result.is_ok(), "invalid tropo Julian split must not panic");
    let err = result
        .expect("invalid tropo Julian split should not unwind")
        .expect_err("invalid tropo Julian split must error")
        .into_float();
    assert_invalid_input(
        err,
        "ppp epoch jd_fraction",
        "must be within one residual day",
    );
}

#[test]
fn static_float_solver_rejects_invalid_met_when_troposphere_enabled() {
    let (source, epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();
    let tropo = TroposphereOptions {
        enabled: true,
        estimate_ztd: false,
        estimate_tropo_gradients: false,
        met: crate::tropo::Met::new_unchecked(0.0, 288.15, 0.5),
        mapping: TropoMapping::Niell,
    };

    let err = solve_float_epochs(&source, &epochs, initial, ppp_row_trace_float_config(tropo))
        .expect_err("invalid enabled-troposphere met must be rejected");

    assert_invalid_input(err, "ppp tropo pressure_hpa", "not positive");
}

#[test]
fn static_float_solver_rejects_nan_correction_table_value() {
    let (source, epochs, initial, _ambiguity_ids) = ppp_row_trace_arc();
    let sat = epochs[0].observations[0].sat;
    let mut corrections = RangeCorrections::disabled();
    corrections.ppp.windup_m.insert((sat, 0), f64::NAN);

    let err = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections,
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("NaN PPP correction table value must be rejected");

    assert_invalid_input(err, "ppp correction windup_m", "not finite");
}

#[test]
fn single_epoch_float_solver_recovers_synthetic_snapshot() {
    let sats = [
        (1, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
        (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
        (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
        (4, [-18_700_000.0, -18_200_000.0, 22_000_000.0]),
        (5, [23_500_000.0, 3_200_000.0, -18_900_000.0]),
        (6, [-7_500_000.0, 25_800_000.0, -16_000_000.0]),
    ];
    let ids: Vec<GnssSatelliteId> = sats
        .iter()
        .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id"))
        .collect();
    let source = FakeSource {
        states: ids
            .iter()
            .zip(sats.iter())
            .map(|(id, (_, pos))| (*id, *pos))
            .collect(),
    };
    let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
    let clock = 12.5;
    let ambiguities: BTreeMap<String, f64> = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
        .collect();
    let observations = ids
        .iter()
        .map(|id| {
            let pred = predict(
                &source,
                *id,
                truth,
                0.0,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: true,
                    sagnac: true,
                },
            )
            .unwrap();
            let code = pred.geometric_range_m + clock;
            let ambiguity = ambiguities.get(&id.to_string()).copied().unwrap();
            FloatObservation {
                sat: *id,
                satellite_id: id.to_string(),
                ambiguity_id: id.to_string(),
                code_m: code,
                phase_m: code + ambiguity,
                freq1_hz: 0.0,
                freq2_hz: 0.0,
                glonass_channel: None,
            }
        })
        .collect::<Vec<_>>();
    let epoch = FloatEpoch {
        epoch: CivilDateTime {
            year: 2020,
            month: 6,
            day: 24,
            hour: 12,
            minute: 0,
            second: 0.0,
        },
        jd_whole: 2_459_024.5,
        jd_fraction: 0.5,
        t_rx_j2000_s: 0.0,
        observations,
    };
    let initial = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0],
        ambiguities_m: initial_ambiguities(std::slice::from_ref(&epoch)),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let solution = solve_float_epoch(
        &source,
        epoch,
        initial,
        FloatSolveConfig {
            weights: MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 8,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .unwrap();
    assert_eq!(
        solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(solution.ztd_residual_m, None);
    assert!(norm3(sub3(solution.position_m, truth)) < 1.0e-3);
    assert!((solution.epoch_clocks_m[0] - clock).abs() < 1.0e-4);
    for (sat, expected) in ambiguities {
        assert!((solution.ambiguities_m[&sat] - expected).abs() < 1.0e-4);
    }
    assert!(solution.code_rms_m < 1.0e-8);
    assert!(solution.phase_rms_m < 1.0e-8);
    assert!(solution.weighted_rms_m < 1.0e-6);
    assert_position_covariance_positive_definite(&solution.formal_position_covariance);
    assert_position_covariance_scaled_by_factor(
        &solution.position_covariance,
        &solution.formal_position_covariance,
        solution.position_covariance_scale_factor,
    );
    assert_eq!(solution.status, FloatStatus::StateTolerance);
    assert!(solution.converged);
    assert_eq!(solution.iterations, 3);
}

#[test]
fn single_epoch_fixed_solver_uses_custom_ambiguity_ids() {
    let sats = [
        (1, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
        (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
        (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
        (4, [-18_700_000.0, -18_200_000.0, 22_000_000.0]),
        (5, [23_500_000.0, 3_200_000.0, -18_900_000.0]),
        (6, [-7_500_000.0, 25_800_000.0, -16_000_000.0]),
    ];
    let ids: Vec<GnssSatelliteId> = sats
        .iter()
        .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id"))
        .collect();
    let ambiguity_ids = ids
        .iter()
        .map(|id| {
            let token = id.to_string();
            if token == "G01" {
                "G01#2".to_string()
            } else {
                token
            }
        })
        .collect::<Vec<_>>();
    let source = FakeSource {
        states: ids
            .iter()
            .zip(sats.iter())
            .map(|(id, (_, pos))| (*id, *pos))
            .collect(),
    };
    let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
    let clock = 12.5;
    let wavelength = C_M_S / F_L1_HZ;
    let fixed_cycles: BTreeMap<String, i64> = ambiguity_ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.clone(), 80_000 + idx as i64 * 37))
        .collect();
    let observations = ids
        .iter()
        .zip(ambiguity_ids.iter())
        .map(|(id, ambiguity_id)| {
            let pred = predict(
                &source,
                *id,
                truth,
                0.0,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: true,
                    sagnac: true,
                },
            )
            .unwrap();
            let code = pred.geometric_range_m + clock;
            let ambiguity = fixed_cycles[ambiguity_id] as f64 * wavelength;
            FloatObservation {
                sat: *id,
                satellite_id: id.to_string(),
                ambiguity_id: ambiguity_id.clone(),
                code_m: code,
                phase_m: code + ambiguity,
                freq1_hz: 0.0,
                freq2_hz: 0.0,
                glonass_channel: None,
            }
        })
        .collect::<Vec<_>>();
    let epochs = vec![FloatEpoch {
        epoch: CivilDateTime {
            year: 2020,
            month: 6,
            day: 24,
            hour: 12,
            minute: 0,
            second: 0.0,
        },
        jd_whole: 2_459_024.5,
        jd_fraction: 0.5,
        t_rx_j2000_s: 0.0,
        observations,
    }];
    let initial = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let weights = MeasurementWeights {
        code: 1.0,
        phase: 100.0,
        elevation_weighting: false,
    };
    let tropo = TroposphereOptions::disabled();
    let opts = FloatSolveOptions {
        max_iterations: 8,
        position_tolerance_m: 1.0e-4,
        clock_tolerance_m: 1.0e-4,
        ambiguity_tolerance_m: 1.0e-4,
        ztd_tolerance_m: 1.0e-4,
    };
    let corrections = RangeCorrections::disabled();
    let float_solution = solve_float_epoch(
        &source,
        epochs[0].clone(),
        initial,
        FloatSolveConfig {
            weights,
            tropo,
            corrections: corrections.clone(),
            opts,
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .unwrap();

    assert_eq!(float_solution.used_sats, ambiguity_ids);
    assert!(float_solution.ambiguities_m.contains_key("G01#2"));
    assert!(!float_solution.ambiguities_m.contains_key("G01"));

    let wavelengths_m = fixed_cycles
        .keys()
        .map(|id| (id.clone(), wavelength))
        .collect();
    let offsets_m = fixed_cycles.keys().map(|id| (id.clone(), 0.0)).collect();
    let solution = solve_fixed_from_float(
        &source,
        &epochs,
        float_solution,
        FixedSolveConfig {
            weights,
            tropo,
            corrections,
            opts,
            elevation_cutoff_deg: None,
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m,
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .unwrap();

    assert_eq!(solution.used_sats, ambiguity_ids);
    assert_eq!(solution.fixed_ambiguities_cycles, fixed_cycles);
    assert_eq!(solution.integer.ambiguity_search.order, solution.used_sats);
}

#[test]
fn static_fixed_solver_recovers_synthetic_arc() {
    let sats = [
        (1, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
        (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
        (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
        (4, [-18_700_000.0, -18_200_000.0, 22_000_000.0]),
        (5, [23_500_000.0, 3_200_000.0, -18_900_000.0]),
        (6, [-7_500_000.0, 25_800_000.0, -16_000_000.0]),
    ];
    let ids: Vec<GnssSatelliteId> = sats
        .iter()
        .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id"))
        .collect();
    let source = FakeSource {
        states: ids
            .iter()
            .zip(sats.iter())
            .map(|(id, (_, pos))| (*id, *pos))
            .collect(),
    };
    let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
    let clocks = [12.5, -8.25, 4.0];
    let wavelength = C_M_S / F_L1_HZ;
    let fixed_cycles: BTreeMap<String, i64> = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.to_string(), 80_000 + idx as i64 * 37))
        .collect();
    let mut epochs = Vec::new();
    for (epoch_idx, clock) in clocks.iter().enumerate() {
        let observations = ids
            .iter()
            .map(|id| {
                let pred = predict(
                    &source,
                    *id,
                    truth,
                    epoch_idx as f64 * 900.0,
                    PredictOptions {
                        carrier_hz: F_L1_HZ,
                        light_time: true,
                        sagnac: true,
                    },
                )
                .unwrap();
                let code = pred.geometric_range_m + clock;
                let ambiguity = fixed_cycles[&id.to_string()] as f64 * wavelength;
                FloatObservation {
                    sat: *id,
                    satellite_id: id.to_string(),
                    ambiguity_id: id.to_string(),
                    code_m: code,
                    phase_m: code + ambiguity,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                    glonass_channel: None,
                }
            })
            .collect();
        epochs.push(FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24,
                hour: 12,
                minute: epoch_idx as u8 * 15,
                second: 0.0,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5 + epoch_idx as f64 * 900.0 / crate::constants::SECONDS_PER_DAY,
            t_rx_j2000_s: epoch_idx as f64 * 900.0,
            observations,
        });
    }
    let initial = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0; epochs.len()],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let weights = MeasurementWeights {
        code: 1.0,
        phase: 100.0,
        elevation_weighting: false,
    };
    let tropo = TroposphereOptions::disabled();
    let opts = FloatSolveOptions {
        max_iterations: 8,
        position_tolerance_m: 1.0e-4,
        clock_tolerance_m: 1.0e-4,
        ambiguity_tolerance_m: 1.0e-4,
        ztd_tolerance_m: 1.0e-4,
    };
    let corrections = RangeCorrections::disabled();
    let float_solution = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights,
            tropo,
            corrections: corrections.clone(),
            opts,
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .unwrap();
    let wavelengths_m = fixed_cycles
        .keys()
        .map(|sat| (sat.clone(), wavelength))
        .collect();
    let offsets_m = fixed_cycles.keys().map(|sat| (sat.clone(), 0.0)).collect();
    let solution = solve_fixed_from_float(
        &source,
        &epochs,
        float_solution,
        FixedSolveConfig {
            weights,
            tropo,
            corrections,
            opts,
            elevation_cutoff_deg: None,
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m,
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .unwrap();
    assert_eq!(solution.fixed_ambiguities_cycles, fixed_cycles);
    for (sat, cycles) in &fixed_cycles {
        let expected_m = *cycles as f64 * wavelength;
        assert!((solution.fixed_ambiguities_m[sat] - expected_m).abs() < 1.0e-12);
    }
    assert_eq!(
        solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(solution.ztd_residual_m, None);
    assert_eq!(solution.status, FloatStatus::StateTolerance);
    assert!(solution.converged);
    assert_eq!(solution.iterations, 1);
    assert_eq!(solution.integer.integer_status, IntegerStatus::Fixed);
    assert!(solution.integer.integer_ratio > 1.0e10);
    assert!(solution.integer.integer_best_score < 1.0e-10);
    assert!(solution.integer.integer_second_best_score.unwrap() > 0.5);
    assert_eq!(solution.integer.integer_candidates, 2);
    assert!(solution.code_rms_m < 1.0e-8);
    assert!(solution.phase_rms_m < 1.0e-8);
    assert!(solution.weighted_rms_m < 1.0e-6);
    assert_eq!(
        solution.integer.ambiguity_search.order,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    for (sat, cycles) in &fixed_cycles {
        let float_cycles = solution.integer.ambiguity_search.float_cycles[sat];
        assert!((float_cycles - *cycles as f64).abs() < 1.0e-4);
    }
    assert_position_covariance_positive_definite(&solution.formal_position_covariance);
    assert_position_covariance_scaled_by_factor(
        &solution.position_covariance,
        &solution.formal_position_covariance,
        solution.position_covariance_scale_factor,
    );
    assert!(norm3(sub3(solution.position_m, truth)) < 1.0e-3);
    for (actual, expected) in solution.epoch_clocks_m.iter().zip(clocks) {
        assert!((actual - expected).abs() < 1.0e-4);
    }
}

#[test]
fn static_fixed_solver_rejects_short_float_solution_clock_vector() {
    let (source, epochs, state, _ambiguity_ids) = ppp_row_trace_arc();
    let used_sats = state.ambiguities_m.keys().cloned().collect::<Vec<_>>();
    let wavelength = C_M_S / F_L1_HZ;
    let float_solution = FloatSolution {
        position_m: state.position_m,
        position_covariance: unit_position_covariance(),
        formal_position_covariance: unit_position_covariance(),
        posterior_variance_factor: 1.0,
        position_covariance_scale_factor: 1.0,
        temporal_position_covariance: unit_position_covariance(),
        temporal_position_covariance_scale_factor: 1.0,
        temporal_correlation: unit_temporal_correlation(),
        epoch_clocks_m: vec![0.0; epochs.len() - 1],
        ambiguities_m: state.ambiguities_m,
        residual_ionosphere_m: BTreeMap::new(),
        ztd_residual_m: None,
        tropo_gradient_north_m: None,
        tropo_gradient_east_m: None,
        tropo_gradient_covariance_m2: None,
        formal_tropo_gradient_covariance_m2: None,
        residuals_m: Vec::new(),
        used_sats: used_sats.clone(),
        iterations: 1,
        converged: true,
        status: FloatStatus::StateTolerance,
        code_rms_m: 0.0,
        phase_rms_m: 0.0,
        weighted_rms_m: 0.0,
    };

    let err = solve_fixed_from_float(
        &source,
        &epochs,
        float_solution,
        FixedSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m: used_sats
                    .iter()
                    .map(|sat| (sat.clone(), wavelength))
                    .collect(),
                offsets_m: used_sats.iter().map(|sat| (sat.clone(), 0.0)).collect(),
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("short fixed PPP float-solution clock vector must be rejected");

    assert_eq!(
        err,
        FixedSolveError::Float(FloatSolveError::InvalidClockCount {
            expected: epochs.len(),
            actual: epochs.len() - 1,
        })
    );
}

#[test]
fn static_fixed_solver_rejects_nan_tolerance() {
    let (source, epochs, state, _ambiguity_ids) = ppp_row_trace_arc();
    let used_sats = state.ambiguities_m.keys().cloned().collect::<Vec<_>>();
    let wavelength = C_M_S / F_L1_HZ;
    let float_solution = FloatSolution {
        position_m: state.position_m,
        position_covariance: unit_position_covariance(),
        formal_position_covariance: unit_position_covariance(),
        posterior_variance_factor: 1.0,
        position_covariance_scale_factor: 1.0,
        temporal_position_covariance: unit_position_covariance(),
        temporal_position_covariance_scale_factor: 1.0,
        temporal_correlation: unit_temporal_correlation(),
        epoch_clocks_m: vec![0.0; epochs.len()],
        ambiguities_m: state.ambiguities_m,
        residual_ionosphere_m: BTreeMap::new(),
        ztd_residual_m: None,
        tropo_gradient_north_m: None,
        tropo_gradient_east_m: None,
        tropo_gradient_covariance_m2: None,
        formal_tropo_gradient_covariance_m2: None,
        residuals_m: Vec::new(),
        used_sats: used_sats.clone(),
        iterations: 1,
        converged: true,
        status: FloatStatus::StateTolerance,
        code_rms_m: 0.0,
        phase_rms_m: 0.0,
        weighted_rms_m: 0.0,
    };

    let err = solve_fixed_from_float(
        &source,
        &epochs,
        float_solution,
        FixedSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: f64::NAN,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m: used_sats
                    .iter()
                    .map(|sat| (sat.clone(), wavelength))
                    .collect(),
                offsets_m: used_sats.iter().map(|sat| (sat.clone(), 0.0)).collect(),
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("NaN fixed PPP tolerance must be rejected");

    assert_eq!(
        err,
        FixedSolveError::Float(FloatSolveError::InvalidSolveOption {
            field: "position_tolerance_m",
            reason: "must be finite",
        })
    );
}

#[test]
fn static_fixed_solver_rejects_nan_wavelength() {
    let (source, epochs, state, _ambiguity_ids) = ppp_row_trace_arc();
    let used_sats = state.ambiguities_m.keys().cloned().collect::<Vec<_>>();
    let mut wavelengths_m: BTreeMap<String, f64> =
        used_sats.iter().map(|sat| (sat.clone(), 0.190)).collect();
    wavelengths_m.insert(used_sats[0].clone(), f64::NAN);
    let float_solution = FloatSolution {
        position_m: state.position_m,
        position_covariance: unit_position_covariance(),
        formal_position_covariance: unit_position_covariance(),
        posterior_variance_factor: 1.0,
        position_covariance_scale_factor: 1.0,
        temporal_position_covariance: unit_position_covariance(),
        temporal_position_covariance_scale_factor: 1.0,
        temporal_correlation: unit_temporal_correlation(),
        epoch_clocks_m: state.clocks_m,
        ambiguities_m: state.ambiguities_m,
        residual_ionosphere_m: BTreeMap::new(),
        ztd_residual_m: None,
        tropo_gradient_north_m: None,
        tropo_gradient_east_m: None,
        tropo_gradient_covariance_m2: None,
        formal_tropo_gradient_covariance_m2: None,
        residuals_m: Vec::new(),
        used_sats: used_sats.clone(),
        iterations: 0,
        converged: false,
        status: FloatStatus::MaxIterations,
        code_rms_m: 0.0,
        phase_rms_m: 0.0,
        weighted_rms_m: 0.0,
    };

    let err = solve_fixed_from_float(
        &source,
        &epochs,
        float_solution,
        FixedSolveConfig {
            weights: ppp_row_trace_weights(),
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions {
                max_iterations: 1,
                position_tolerance_m: 1.0e-4,
                clock_tolerance_m: 1.0e-4,
                ambiguity_tolerance_m: 1.0e-4,
                ztd_tolerance_m: 1.0e-4,
            },
            elevation_cutoff_deg: None,
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m: used_sats.iter().map(|sat| (sat.clone(), 0.0)).collect(),
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .expect_err("NaN fixed PPP wavelength must be rejected");

    assert_eq!(
        err,
        FixedSolveError::Float(FloatSolveError::InvalidInput {
            field: "ppp fixed ambiguity wavelength_m",
            reason: "not finite",
        })
    );
}

// ---------------------------------------------------------------------------
// Phase-2 P0: row-level PPP design-row golden traces.
//
// The existing solver goldens freeze the final solution and the POST-fit
// residual rows. These freeze the PRE-fit undifferenced design rows (the design
// vector `h`, prefit residual `y`, and measurement weight) emitted by the float
// (`build_multi_rows`) and fixed (`build_fixed_multi_rows`) row builders, so the
// later substrate extraction (P1/P2) is provably behavior-preserving at the row
// level. Any change to the undifferenced code/phase model, the design-row column
// layout, or the weighting shifts these bits.
// ---------------------------------------------------------------------------

// Three satellites over two epochs; perfect synthetic observations:
// code = geometric range + receiver clock, phase = code + ambiguity.
fn ppp_row_trace_arc() -> (FakeSource, Vec<FloatEpoch>, FloatState, Vec<AmbiguityId>) {
    let sats = [
        (1u8, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
        (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
        (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
    ];
    let ids: Vec<GnssSatelliteId> = sats
        .iter()
        .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid satellite id"))
        .collect();
    let source = FakeSource {
        states: ids
            .iter()
            .zip(sats.iter())
            .map(|(id, (_, pos))| (*id, *pos))
            .collect(),
    };
    let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
    let clocks = [12.5, -8.25];
    let ambiguities: BTreeMap<String, f64> = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
        .collect();
    let mut epochs = Vec::new();
    for (epoch_idx, clock) in clocks.iter().enumerate() {
        let observations = ids
            .iter()
            .map(|id| {
                let pred = predict(
                    &source,
                    *id,
                    truth,
                    epoch_idx as f64 * 900.0,
                    PredictOptions {
                        carrier_hz: F_L1_HZ,
                        light_time: true,
                        sagnac: true,
                    },
                )
                .unwrap();
                let code = pred.geometric_range_m + clock;
                let ambiguity = ambiguities.get(&id.to_string()).copied().unwrap();
                FloatObservation {
                    sat: *id,
                    satellite_id: id.to_string(),
                    ambiguity_id: id.to_string(),
                    code_m: code,
                    phase_m: code + ambiguity,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                    glonass_channel: None,
                }
            })
            .collect();
        epochs.push(FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24,
                hour: 12,
                minute: epoch_idx as u8 * 15,
                second: 0.0,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5 + epoch_idx as f64 * 900.0 / crate::constants::SECONDS_PER_DAY,
            t_rx_j2000_s: epoch_idx as f64 * 900.0,
            observations,
        });
    }
    // Linearize away from truth so every prefit residual and design partial is
    // exercised with a non-trivial value.
    let state = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0; epochs.len()],
        ambiguities_m: ambiguities,
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let ambiguity_ids = ids
        .iter()
        .map(|id| AmbiguityId::new(id.to_string()))
        .collect();
    (source, epochs, state, ambiguity_ids)
}

fn ppp_row_trace_weights() -> MeasurementWeights {
    MeasurementWeights {
        code: 1.0,
        phase: 100.0,
        elevation_weighting: false,
    }
}

fn ppp_row_trace_float_config(tropo: TroposphereOptions) -> FloatSolveConfig {
    FloatSolveConfig {
        weights: ppp_row_trace_weights(),
        tropo,
        corrections: RangeCorrections::disabled(),
        opts: FloatSolveOptions {
            max_iterations: 8,
            position_tolerance_m: 1.0e-4,
            clock_tolerance_m: 1.0e-4,
            ambiguity_tolerance_m: 1.0e-4,
            ztd_tolerance_m: 1.0e-4,
        },
        elevation_cutoff_deg: None,
        residual_screen: false,
        estimate_residual_ionosphere: false,
    }
}

fn ppp_row_bits(rows: &[super::normal::Row]) -> Vec<u64> {
    let mut bits = Vec::new();
    for r in rows {
        for &h in &r.h {
            bits.push(h.to_bits());
        }
        bits.push(r.y.to_bits());
        bits.push(r.weight.to_bits());
    }
    bits
}

#[test]
fn float_design_rows_have_frozen_bits_golden() {
    let (source, epochs, state, ambiguity_ids) = ppp_row_trace_arc();
    let corrections = RangeCorrections::disabled();
    let ctx = ModelContext {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo: TroposphereOptions::disabled(),
        corrections: &corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
    };

    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let rows = super::rows::build_rows(ctx, &epochs, &binding, &state).unwrap();

    // 2 epochs x 3 sats x (code + phase) = 12 rows; design width =
    // 3 position + 2 per-epoch clocks + 3 ambiguities (tropo disabled).
    assert_eq!(rows.len(), 12);
    assert_eq!(rows[0].h.len(), 8);
    assert_eq!(ppp_row_bits(&rows).as_slice(), PPP_FLOAT_DESIGN_ROW_GOLDEN);
}

#[test]
fn fixed_design_rows_have_frozen_bits_golden() {
    let (source, epochs, state, _ambiguity_ids) = ppp_row_trace_arc();
    let corrections = RangeCorrections::disabled();
    let ctx = ModelContext {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo: TroposphereOptions::disabled(),
        corrections: &corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
    };
    // The fixed solver holds every ambiguity; here at its truth value.
    let fixed_m: BTreeMap<String, f64> = state.ambiguities_m.clone();

    let binding = super::rows::AmbiguityBinding::Held { values: &fixed_m };
    let rows = super::rows::build_rows(ctx, &epochs, &binding, &state).unwrap();

    // Same 12 rows; design width = 3 position + 2 clocks (no ambiguity columns
    // once fixed, tropo disabled).
    assert_eq!(rows.len(), 12);
    assert_eq!(rows[0].h.len(), 5);
    assert_eq!(ppp_row_bits(&rows).as_slice(), PPP_FIXED_DESIGN_ROW_GOLDEN);
}

// Generated by running each test once and freezing the observed bits; see the
// module comment. Regenerate only with a deliberate, reviewed behavior change.
const PPP_FLOAT_DESIGN_ROW_GOLDEN: &[u64] = &[
    13827261380611783850,
    13825412640259596458,
    13827112186925804208,
    4607182418800017408,
    0,
    0,
    0,
    0,
    4644851261086957568,
    4607182418800017408,
    13827261380611783850,
    13825412640259596458,
    13827112186925804208,
    4607182418800017408,
    0,
    4607182418800017408,
    0,
    0,
    4644851261086957568,
    4636737291354636288,
    4605096716435247059,
    13824697895126236484,
    13825663558865739684,
    4607182418800017408,
    0,
    0,
    0,
    0,
    13868731662273609728,
    4607182418800017408,
    4605096716435247059,
    13824697895126236484,
    13825663558865739684,
    4607182418800017408,
    0,
    0,
    4607182418800017408,
    0,
    13868731662273609728,
    4636737291354636288,
    13824228316578245539,
    4605177694311148212,
    13825804969787867149,
    4607182418800017408,
    0,
    0,
    0,
    0,
    4649269908014563328,
    4607182418800017408,
    13824228316578245539,
    4605177694311148212,
    13825804969787867149,
    4607182418800017408,
    0,
    0,
    0,
    4607182418800017408,
    4649269908014563328,
    4636737291354636288,
    13827261380611783850,
    13825412640259596458,
    13827112186925804208,
    0,
    4607182418800017408,
    0,
    0,
    0,
    4644486223226535936,
    4607182418800017408,
    13827261380611783850,
    13825412640259596458,
    13827112186925804208,
    0,
    4607182418800017408,
    4607182418800017408,
    0,
    0,
    4644486223226535936,
    4636737291354636288,
    4605096716435247059,
    13824697895126236484,
    13825663558865739684,
    0,
    4607182418800017408,
    0,
    0,
    0,
    13869096700134031360,
    4607182418800017408,
    4605096716435247059,
    13824697895126236484,
    13825663558865739684,
    0,
    4607182418800017408,
    0,
    4607182418800017408,
    0,
    13869096700134031360,
    4636737291354636288,
    13824228316578245539,
    4605177694311148212,
    13825804969787867149,
    0,
    4607182418800017408,
    0,
    0,
    0,
    4649087389084352512,
    4607182418800017408,
    13824228316578245539,
    4605177694311148212,
    13825804969787867149,
    0,
    4607182418800017408,
    0,
    0,
    4607182418800017408,
    4649087389084352512,
    4636737291354636288,
];
const PPP_FIXED_DESIGN_ROW_GOLDEN: &[u64] = &[
    13827261380611783850,
    13825412640259596458,
    13827112186925804208,
    4607182418800017408,
    0,
    4644851261086957568,
    4607182418800017408,
    13827261380611783850,
    13825412640259596458,
    13827112186925804208,
    4607182418800017408,
    0,
    4644851261086957568,
    4636737291354636288,
    4605096716435247059,
    13824697895126236484,
    13825663558865739684,
    4607182418800017408,
    0,
    13868731662273609728,
    4607182418800017408,
    4605096716435247059,
    13824697895126236484,
    13825663558865739684,
    4607182418800017408,
    0,
    13868731662273609728,
    4636737291354636288,
    13824228316578245539,
    4605177694311148212,
    13825804969787867149,
    4607182418800017408,
    0,
    4649269908014563328,
    4607182418800017408,
    13824228316578245539,
    4605177694311148212,
    13825804969787867149,
    4607182418800017408,
    0,
    4649269908014563328,
    4636737291354636288,
    13827261380611783850,
    13825412640259596458,
    13827112186925804208,
    0,
    4607182418800017408,
    4644486223226535936,
    4607182418800017408,
    13827261380611783850,
    13825412640259596458,
    13827112186925804208,
    0,
    4607182418800017408,
    4644486223226535936,
    4636737291354636288,
    4605096716435247059,
    13824697895126236484,
    13825663558865739684,
    0,
    4607182418800017408,
    13869096700134031360,
    4607182418800017408,
    4605096716435247059,
    13824697895126236484,
    13825663558865739684,
    0,
    4607182418800017408,
    13869096700134031360,
    4636737291354636288,
    13824228316578245539,
    4605177694311148212,
    13825804969787867149,
    0,
    4607182418800017408,
    4649087389084352512,
    4607182418800017408,
    13824228316578245539,
    4605177694311148212,
    13825804969787867149,
    0,
    4607182418800017408,
    4649087389084352512,
    4636737291354636288,
];

#[test]
fn vmf_site_series_interpolation_is_bounded_past_the_span() {
    // 6-hourly series over one day.
    let series = VmfSiteSeries::new(&[
        VmfSiteSample {
            mjd: 61173.00,
            ah: 0.00121738,
            aw: 0.00058796,
        },
        VmfSiteSample {
            mjd: 61173.25,
            ah: 0.00121388,
            aw: 0.00053850,
        },
        VmfSiteSample {
            mjd: 61173.50,
            ah: 0.00121315,
            aw: 0.00048897,
        },
        VmfSiteSample {
            mjd: 61173.75,
            ah: 0.00121222,
            aw: 0.00052133,
        },
    ])
    .expect("valid VMF series");

    // Inside the span: interpolates (matches the clamping path).
    let mid = series
        .interpolate_checked(61173.10)
        .expect("in-span epoch resolves");
    assert_eq!(mid, series.interpolate(61173.10));

    // Within one sampling step (6 h = 0.25 day) past the last node: still covered,
    // clamped to the endpoint (the legitimate final-block case).
    let near = series
        .interpolate_checked(61173.95)
        .expect("epoch within one step past the last node is covered");
    assert_eq!(near, (0.00121222, 0.00052133));

    // More than one step past the last node: out of coverage, flagged - not the
    // stale endpoint coefficient reused for every later epoch.
    assert_eq!(series.interpolate_checked(61174.10), None);
    // Symmetrically before the first node.
    assert_eq!(series.interpolate_checked(61172.50), None);
    // The unbounded clamp would still return the endpoint here; the checked path
    // is what refuses it.
    assert_eq!(series.interpolate(61174.10), (0.00121222, 0.00052133));
}
