use super::*;
use crate::ambiguity::AmbiguityId;
use crate::astro::math::vec3::{add3, cross3, norm3, scale3, sub3, unit3};
use crate::carrier_phase::{CycleSlipOptions, SlipReason};
use crate::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
use crate::has::{
    HasClockBlock, HasClockCorrection, HasClockSystem, HasCodeBias, HasCodeBiasBlock, HasGnssMask,
    HasMaskBlock, HasMt1Header, HasMt1Message, HasOrbitBlock, HasOrbitCorrection, HasPhaseBias,
    HasPhaseBiasBlock,
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
        ssr_bias_exclusions: Vec::new(),
        solved_epoch_indices: vec![0],
        ssr_bias_readmissions: Vec::new(),
        ssr_bias_last_pass: 0,
        residual_screen: false,
        solve_options: FloatSolveOptions::default(),
        residual_screen_removals: Vec::new(),
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
        correction_epoch_indices: None,
        ssr_bias_pass: 0,
        ssr_bias_stage: SsrBiasExclusionStage::BeforeSolve,
        ssr_bias_deferred: &[],
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
        correction_epoch_indices: None,
        ssr_bias_pass: 0,
        ssr_bias_stage: SsrBiasExclusionStage::BeforeSolve,
        ssr_bias_deferred: &[],
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
            let pred = crate::observables::predict_transmit_geometry(
                &source,
                *id,
                truth,
                t_rx_j2000_s,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: true,
                    sagnac: true,
                },
                crate::observables::NOMINAL_SIGNAL_FLIGHT_TIME_S,
            )
            .expect("synthetic prediction");
            let sat_velocity_m_s = corrections.sat_clock_relativity.then(|| {
                crate::observables::transmit_velocity_m_s(&source, *id, &pred, true)
                    .expect("synthetic velocity")
            });
            let tropo_model = super::model::model_troposphere(&pred, truth, &epoch, tropo)
                .expect("synthetic tropo model");
            let corrections_m = super::model::range_corrections_m(
                super::model::CorrectedObservation {
                    pred: &pred,
                    sat_velocity_m_s,
                    rx_pos: truth,
                    epoch_idx,
                    obs: &obs,
                },
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
    let horizontal = libm::cos(el);
    let los = add3(
        add3(
            scale3(north, horizontal * libm::cos(az)),
            scale3(east, horizontal * libm::sin(az)),
        ),
        scale3(up, libm::sin(el)),
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

// SSR/HAS PPP bias application.
//
// Every scenario runs at GPS week 2425 TOW 345600, the reference time of the GPS LNAV
// record in the SSR NAV fixture, relabelled for G01..G03, so the SSR-corrected ephemeris
// applies HAS orbit and clock corrections to the synthetic satellites. A HAS TOH of `n`
// seconds received at TOW 345600 + n has reference epoch `ssr_test_t0() + n`.

const SSR_TEST_WEEK: u32 = 2425;
const SSR_TEST_TOW_S: f64 = 345_600.0;
/// IODE of the fixture's G31 record, which every relabelled satellite carries.
const SSR_TEST_IODE: u32 = 67;
/// HAS validity interval index for 60 s.
const HAS_VI_60_S: u8 = 5;
/// HAS validity interval index for 5 s.
const HAS_VI_5_S: u8 = 0;

/// Receiver position of the row-trace arc, used to place transmission times.
fn ssr_test_receiver() -> [f64; 3] {
    [3_512_900.0, 780_500.0, 5_248_700.0]
}

fn ssr_test_t0() -> f64 {
    f64::from(SSR_TEST_WEEK) * crate::constants::SECONDS_PER_WEEK + SSR_TEST_TOW_S
        - crate::constants::GPS_EPOCH_TO_J2000_S
}

/// Broadcast ephemeris holding the fixture's G31 LNAV record under G01, G02 and G03.
fn ssr_test_broadcast() -> crate::ephemeris::BroadcastEphemeris {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
    ));
    let header_label = text.find("END OF HEADER").expect("NAV header end");
    let body_start = header_label + text[header_label..].find('\n').expect("header line end") + 1;
    let record_start = text.find("\nG31 ").expect("G31 record") + 1;
    let record_end = record_start + text[record_start..].find("\nG30 ").expect("G30 record") + 1;
    let record = &text[record_start..record_end];
    let mut nav = text[..body_start].to_string();
    for prn in ["G01", "G02", "G03"] {
        nav.push_str(&record.replacen("G31", prn, 1));
    }
    crate::ephemeris::BroadcastEphemeris::from_nav(&nav).expect("parse relabelled NAV")
}

fn has_test_reception(offset_s: u16) -> crate::astro::time::model::GnssWeekTow {
    crate::astro::time::model::GnssWeekTow::new(
        crate::astro::time::model::TimeScale::Gst,
        SSR_TEST_WEEK,
        SSR_TEST_TOW_S + f64::from(offset_s),
    )
    .expect("GST reception")
}

/// HAS code and phase biases on signals 0 and 9, the same for every satellite of a message.
#[derive(Clone, Copy)]
struct HasTestBiases {
    code_m: [Option<f64>; 2],
    phase_cycles: [Option<f64>; 2],
    pdi: u8,
}

impl HasTestBiases {
    fn usable(code_m: [f64; 2], phase_cycles: [f64; 2]) -> Self {
        Self {
            code_m: code_m.map(Some),
            phase_cycles: phase_cycles.map(Some),
            pdi: 0,
        }
    }

    fn unavailable() -> Self {
        Self {
            code_m: [None, None],
            phase_cycles: [None, None],
            pdi: 0,
        }
    }
}

/// HAS MT1 message with a GPS mask for `sats` (ascending), orbit and clock blocks when
/// `orbit_clock_vi` is set, and code and phase biases when `biases` is set.
fn has_test_message(
    sats: &[GnssSatelliteId],
    toh_s: u16,
    mask_id: u8,
    iod_set_id: u8,
    orbit_clock_vi: Option<u8>,
    biases: Option<HasTestBiases>,
) -> HasMt1Message {
    HasMt1Message {
        header: HasMt1Header {
            toh_s,
            mask: true,
            orbit: orbit_clock_vi.is_some(),
            clock_full_set: orbit_clock_vi.is_some(),
            clock_subset: false,
            code_bias: biases.is_some(),
            phase_bias: biases.is_some(),
            reserved: 0,
            mask_id,
            iod_set_id,
        },
        mask: Some(HasMaskBlock {
            systems: vec![HasGnssMask {
                system: GnssSystem::Gps,
                satellites: sats.iter().map(|sat| sat.prn).collect(),
                signals: vec![0, 9],
                cell_mask: None,
                nav_message: 0,
            }],
            reserved: 0,
        }),
        orbit: orbit_clock_vi.map(|validity_interval| HasOrbitBlock {
            validity_interval,
            records: sats
                .iter()
                .map(|sat| HasOrbitCorrection {
                    sat: *sat,
                    nav_message: 0,
                    iode: SSR_TEST_IODE,
                    radial_m: Some(1.25),
                    along_m: Some(-2.0),
                    cross_m: Some(3.0),
                })
                .collect(),
        }),
        clock_full_set: orbit_clock_vi.map(|validity_interval| HasClockBlock {
            validity_interval,
            systems: vec![HasClockSystem {
                system: GnssSystem::Gps,
                multiplier_index: 0,
            }],
            records: sats
                .iter()
                .map(|sat| HasClockCorrection {
                    sat: *sat,
                    nav_message: 0,
                    correction_m: Some(-0.75),
                    do_not_use: false,
                })
                .collect(),
        }),
        clock_subset: None,
        code_bias: biases.map(|b| HasCodeBiasBlock {
            validity_interval: HAS_VI_60_S,
            records: sats
                .iter()
                .flat_map(|sat| {
                    [0, 9]
                        .into_iter()
                        .zip(b.code_m)
                        .map(|(signal_id, bias_m)| HasCodeBias {
                            sat: *sat,
                            signal_id,
                            bias_m,
                        })
                })
                .collect(),
        }),
        phase_bias: biases.map(|b| HasPhaseBiasBlock {
            validity_interval: HAS_VI_60_S,
            records: sats
                .iter()
                .flat_map(|sat| {
                    [0, 9]
                        .into_iter()
                        .zip(b.phase_cycles)
                        .map(|(signal_id, bias_cycles)| HasPhaseBias {
                            sat: *sat,
                            signal_id,
                            bias_cycles,
                            discontinuity_indicator: b.pdi,
                        })
                })
                .collect(),
        }),
        padding_bits: Vec::new(),
    }
}

/// Encode and decode a HAS MT1 message, then ingest it at TOW 345600 + its TOH.
fn has_test_ingest(store: &mut SsrCorrectionStore, message: &HasMt1Message) {
    let decoded =
        HasMt1Message::decode(&message.encode().expect("encode HAS MT1")).expect("decode HAS MT1");
    store
        .ingest_has_mt1(&decoded, has_test_reception(message.header.toh_s))
        .expect("ingest HAS MT1");
}

fn gps_l1_l2_signal_pair() -> SsrPppBiasSignalPair {
    SsrPppBiasSignalPair {
        code1_signal: 0,
        code2_signal: 9,
        phase1_signal: 0,
        phase2_signal: 9,
        freq1_hz: F_L1_HZ,
        freq2_hz: F_L2_HZ,
    }
}

fn gps_l1_l2_options() -> SsrPppBiasOptions {
    SsrPppBiasOptions::new().with_system_signal_pair(GnssSystem::Gps, gps_l1_l2_signal_pair())
}

/// One epoch of the row-trace arc with its first `n_obs` observations, received one
/// second after `ssr_test_t0()`, so its signal left after the corrections' reference epoch.
fn ssr_test_arc(n_obs: usize) -> (FakeSource, Vec<FloatEpoch>, FloatState, Vec<AmbiguityId>) {
    let (source, mut epochs, mut state, _) = ppp_row_trace_arc();
    epochs[0].observations.truncate(n_obs);
    epochs.truncate(1);
    epochs[0].t_rx_j2000_s = ssr_test_t0() + 1.0;
    state.clocks_m.truncate(1);
    state.ambiguities_m = initial_ambiguities(&epochs);
    let ambiguity_ids = epochs[0]
        .observations
        .iter()
        .map(|o| AmbiguityId::new(o.ambiguity_id.clone()))
        .collect();
    (source, epochs, state, ambiguity_ids)
}

fn ionosphere_free(first: f64, second: f64) -> f64 {
    let gamma = F_L1_HZ * F_L1_HZ / (F_L1_HZ * F_L1_HZ - F_L2_HZ * F_L2_HZ);
    gamma * first - (gamma - 1.0) * second
}

fn has_solution(provider_id: u16, solution_id: u8) -> crate::ssr::SsrSolution {
    crate::ssr::SsrSolution {
        source: crate::ssr::SsrSource::GalileoHas,
        provider_id,
        solution_id,
    }
}

fn row_trace_ctx<'a>(
    source: &'a dyn ObservableEphemerisSource,
    corrections: &'a RangeCorrections,
) -> ModelContext<'a> {
    ModelContext {
        source,
        weights: ppp_row_trace_weights(),
        tropo: TroposphereOptions::disabled(),
        corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
        correction_epoch_indices: None,
        ssr_bias_pass: 0,
        ssr_bias_stage: SsrBiasExclusionStage::BeforeSolve,
        ssr_bias_deferred: &[],
    }
}

#[test]
fn static_float_rows_apply_ssr_code_and_phase_biases_with_expected_signs() {
    let (_source, epochs, state, ambiguity_ids) = ssr_test_arc(1);
    let sat = epochs[0].observations[0].sat;
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let code_m = [0.24, -0.46];
    let phase_cycles = [1.25, -2.5];
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable(code_m, phase_cycles)),
        ),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    // The solve's source is the SSR-corrected ephemeris the biases were resolved against.
    let base_corrections = RangeCorrections::disabled();
    let base_rows = super::rows::build_rows(
        row_trace_ctx(&ephemeris, &base_corrections),
        &epochs,
        &binding,
        &state,
    )
    .unwrap();
    let (biased_lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    assert_eq!(report.status, SsrPppAggregateStatus::AllApplied);
    assert_eq!(report.code_applied_count, 1);
    assert_eq!(report.phase_applied_count, 1);
    assert_eq!(
        report.observation_reports[0].applied_orbit_clock_solution,
        Some(has_solution(1, 1))
    );
    let biased_corrections = RangeCorrections {
        ppp: biased_lookup,
        ..RangeCorrections::disabled()
    };
    let biased_rows = super::rows::build_rows(
        row_trace_ctx(&ephemeris, &biased_corrections),
        &epochs,
        &binding,
        &state,
    )
    .unwrap();

    let expected_code_if = ionosphere_free(code_m[0], code_m[1]);
    let expected_phase_if = ionosphere_free(
        phase_cycles[0] * (C_M_S / F_L1_HZ),
        phase_cycles[1] * (C_M_S / F_L2_HZ),
    );
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

/// The broadcast states of a store, served as if their clock were a precise clock that
/// leaves the relativistic term to the user.
struct PreciseClockView<'a>(&'a crate::ephemeris::BroadcastEphemeris);

impl ObservableEphemerisSource for PreciseClockView<'_> {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        self.0.observable_state_at_j2000_s(sat, t_j2000_s)
    }

    fn velocity_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<Result<[f64; 3], ObservablesError>> {
        self.0.velocity_at_j2000_s(sat, t_j2000_s)
    }
}

/// An SSR-corrected clock carries `-2 r·v / c²` (RTKLIB `satpos_ssr`, HAS SIS ICD Eq. 24)
/// and a broadcast clock carries the broadcast relativistic term (RTKLIB `eph2pos`), so
/// with the satellite clock relativity correction enabled the rows add no second term for
/// either: the rows are bit for bit those with it disabled. The term is added for a source
/// whose clock leaves it out, and whenever a CLK series replaces the source's clock.
#[test]
fn rows_add_no_second_satellite_clock_relativity_term() {
    let (fake, epochs, state, ambiguity_ids) = ssr_test_arc(1);
    let sat = epochs[0].observations[0].sat;
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(&[sat], 0, 1, 1, Some(HAS_VI_60_S), None),
    );
    let broadcast = ssr_test_broadcast();
    let ssr = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let precise_view = PreciseClockView(&broadcast);
    let off = RangeCorrections::disabled();
    let on = RangeCorrections {
        sat_clock_relativity: true,
        ..RangeCorrections::disabled()
    };
    let with_clk = RangeCorrections {
        sat_clock_relativity: true,
        satellite_clock: Some(SatelliteClockCorrections::default()),
        ..RangeCorrections::disabled()
    };

    assert!(ssr.clock_includes_relativity());
    assert!(broadcast.clock_includes_relativity());
    assert!(!precise_view.clock_includes_relativity());
    assert!(!fake.clock_includes_relativity());
    let adds = super::rows::adds_sat_clock_relativity;
    assert!(!adds(&ssr, &off));
    assert!(!adds(&ssr, &on));
    assert!(!adds(&broadcast, &on));
    assert!(adds(&precise_view, &on));
    assert!(adds(&fake, &on));
    assert!(adds(&ssr, &with_clk));
    assert!(adds(&broadcast, &with_clk));

    let rows = |source: &dyn ObservableEphemerisSource, corrections: &RangeCorrections| {
        super::rows::build_rows(
            row_trace_ctx(source, corrections),
            &epochs,
            &binding,
            &state,
        )
        .expect("rows")
    };
    let sources: [(&str, &dyn ObservableEphemerisSource); 2] =
        [("SSR-corrected", &ssr), ("broadcast", &broadcast)];
    for (label, source) in sources {
        let disabled = rows(source, &off);
        let enabled = rows(source, &on);
        assert_eq!(disabled.len(), enabled.len(), "{label}");
        for (a, b) in disabled.iter().zip(&enabled) {
            assert_eq!(a.y.to_bits(), b.y.to_bits(), "{label}");
        }
    }
    // Control: the same broadcast states from a source whose clock leaves the term out
    // take it, so the rows move.
    let disabled = rows(&precise_view, &off);
    let enabled = rows(&precise_view, &on);
    let delta = enabled[0].y - disabled[0].y;
    assert!(delta.is_finite() && delta != 0.0, "{delta}");
}

/// Biases transmitted as unavailable are reported per observation, the requirement flags
/// stay set, and a solve leaves those observations out with the report row as the reason.
#[test]
fn test_ppp_ssr_biases_unavailable_excludes_observations_and_retains_all_records() {
    let (source, epochs, state, ambiguity_ids) = ssr_test_arc(2);
    let sat1 = epochs[0].observations[0].sat;
    let sat2 = epochs[0].observations[1].sat;
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let orig_t_rx = epochs[0].t_rx_j2000_s;

    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat1, sat2],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::unavailable()),
        ),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );

    assert_eq!(report.status, SsrPppAggregateStatus::NoneApplied);
    assert_eq!(report.code_applied_count, 0);
    assert_eq!(report.code_failed_count, 2);
    assert_eq!(report.phase_applied_count, 0);
    assert_eq!(report.phase_failed_count, 2);
    assert_eq!(report.observation_reports.len(), 2);
    for obs_rep in &report.observation_reports {
        assert_eq!(
            obs_rep.code_status,
            SsrIfCombinationStatus::SignalUnavailable
        );
        assert_eq!(
            obs_rep.phase_status,
            SsrIfCombinationStatus::SignalUnavailable
        );
        assert!(obs_rep.applied_code_if_m.is_none());
        assert!(obs_rep.applied_phase_if_m.is_none());
        for status in [
            obs_rep.code1_report.as_ref().unwrap().query_result.status,
            obs_rep.code2_report.as_ref().unwrap().query_result.status,
            obs_rep.phase1_report.as_ref().unwrap().query_result.status,
            obs_rep.phase2_report.as_ref().unwrap().query_result.status,
        ] {
            assert_eq!(status, crate::ssr::SsrBiasStatus::Unavailable);
        }
    }
    assert!(lookup.ssr_code_bias_enabled);
    assert!(lookup.phase_bias_enabled);
    assert!(lookup.ssr_code_bias_m.is_empty());
    assert!(lookup.phase_bias_m.is_empty());
    assert_eq!(lookup.ssr_bias_report.as_ref(), Some(&report));
    assert_eq!(epochs[0].t_rx_j2000_s, orig_t_rx);
    assert_eq!(epochs[0].observations.len(), 2);

    // The solves leave both observations out, each with its report row as the reason.
    let (retained, exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
        &ephemeris,
        &epochs,
        0,
        ssr_test_receiver(),
        &lookup,
        0,
        SsrBiasExclusionStage::BeforeSolve,
    );
    assert_eq!(retained.len(), 1, "the epoch keeps its position");
    assert!(retained[0].observations.is_empty());
    assert_eq!(exclusions.len(), 2);
    for (exclusion, row) in exclusions.iter().zip(&report.observation_reports) {
        assert_eq!(exclusion.epoch_index, 0);
        assert_eq!(exclusion.satellite_id, row.satellite_id);
        assert_eq!(exclusion.ambiguity_id, row.ambiguity_id);
        assert!(exclusion.code_bias_missing);
        assert!(exclusion.phase_bias_missing);
        assert_eq!(exclusion.application.as_ref(), Some(row));
    }

    // Row assembly on observations that were not filtered still refuses a missing
    // required bias rather than treating it as zero.
    let corrections = RangeCorrections {
        ppp: lookup,
        ..RangeCorrections::disabled()
    };
    let err = super::rows::build_rows(
        row_trace_ctx(&source, &corrections),
        &epochs,
        &binding,
        &state,
    )
    .unwrap_err()
    .into_float();
    assert_missing_correction(err, sat1, MissingCorrection::SsrCodeBias);
}

/// Bias application is requested by default. `SsrPppBiasOptions::default()` carries no
/// signal pairs, so every observation is reported `NoSignalPairConfigured` and a solve
/// leaves every observation out. Opting out explicitly runs without SSR biases.
#[test]
fn test_ppp_ssr_biases_default_options_without_signal_pairs_exclude_every_observation() {
    let (source, epochs, state, ambiguity_ids) = ssr_test_arc(2);
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };
    // The store is never queried: resolution stops at the missing signal pair.
    let store = SsrCorrectionStore::new();
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);

    let options = SsrPppBiasOptions::default();
    assert!(
        options.apply_code_biases,
        "code biases are requested by default"
    );
    assert!(
        options.apply_phase_biases,
        "phase biases are requested by default"
    );
    assert!(options.per_satellite.is_empty());
    assert!(options.per_system.is_empty());

    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &options,
    );
    assert_eq!(report.status, SsrPppAggregateStatus::NoneApplied);
    assert_eq!(report.code_failed_count, 2);
    assert_eq!(report.phase_failed_count, 2);
    assert_eq!(report.observation_reports.len(), 2);
    for obs_rep in &report.observation_reports {
        assert_eq!(
            obs_rep.code_status,
            SsrIfCombinationStatus::NoSignalPairConfigured
        );
        assert_eq!(
            obs_rep.phase_status,
            SsrIfCombinationStatus::NoSignalPairConfigured
        );
        assert!(obs_rep.code1_report.is_none());
        assert!(obs_rep.code2_report.is_none());
        assert!(obs_rep.phase1_report.is_none());
        assert!(obs_rep.phase2_report.is_none());
    }
    assert!(lookup.ssr_code_bias_enabled);
    assert!(lookup.phase_bias_enabled);

    let (retained, exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
        &ephemeris,
        &epochs,
        0,
        ssr_test_receiver(),
        &lookup,
        0,
        SsrBiasExclusionStage::BeforeSolve,
    );
    assert!(retained[0].observations.is_empty());
    assert_eq!(exclusions.len(), 2);
    for exclusion in &exclusions {
        assert_eq!(
            exclusion.application.as_ref().unwrap().code_status,
            SsrIfCombinationStatus::NoSignalPairConfigured
        );
    }

    let opted_out = SsrPppBiasOptions::default()
        .with_apply_code_biases(false)
        .with_apply_phase_biases(false);
    let (opted_lookup, opted_report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &opted_out,
    );
    assert_eq!(opted_report.status, SsrPppAggregateStatus::EmptyOrOptedOut);
    assert_eq!(opted_report.code_failed_count, 0);
    assert_eq!(opted_report.phase_failed_count, 0);
    assert_eq!(opted_report.observation_reports.len(), 2);
    for obs_rep in &opted_report.observation_reports {
        assert_eq!(obs_rep.code_status, SsrIfCombinationStatus::OptedOut);
        assert_eq!(obs_rep.phase_status, SsrIfCombinationStatus::OptedOut);
    }
    assert!(!opted_lookup.ssr_code_bias_enabled);
    assert!(!opted_lookup.phase_bias_enabled);
    let (opted_retained, opted_exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
        &ephemeris,
        &epochs,
        0,
        ssr_test_receiver(),
        &opted_lookup,
        0,
        SsrBiasExclusionStage::BeforeSolve,
    );
    assert!(opted_exclusions.is_empty());
    assert_eq!(opted_retained, epochs);
    let opted_corrections = RangeCorrections {
        ppp: opted_lookup,
        ..RangeCorrections::disabled()
    };
    super::rows::build_rows(
        row_trace_ctx(&source, &opted_corrections),
        &epochs,
        &binding,
        &state,
    )
    .expect("explicit opt-out lets the rows build without SSR biases");
}

#[test]
fn test_ppp_ssr_biases_partial_survival_and_unrelated_records() {
    let (_source, epochs, _state, _ambiguity_ids) = ssr_test_arc(2);
    let sat1 = epochs[0].observations[0].sat;
    let sat2 = epochs[0].observations[1].sat;
    let code_m = [1.24, -0.76];
    let phase_cycles = [0.2, -0.3];

    // sat1 has usable biases; sat2 has unavailable code biases and usable phase biases.
    let mut message = has_test_message(
        &[sat1, sat2],
        0,
        1,
        1,
        Some(HAS_VI_60_S),
        Some(HasTestBiases::usable(code_m, phase_cycles)),
    );
    for record in &mut message.code_bias.as_mut().unwrap().records {
        if record.sat == sat2 {
            record.bias_m = None;
        }
    }
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(&mut store, &message);
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );

    assert_eq!(report.status, SsrPppAggregateStatus::PartiallyApplied);
    assert_eq!(report.code_applied_count, 1);
    assert_eq!(report.code_failed_count, 1);
    assert_eq!(report.phase_applied_count, 2);
    assert_eq!(report.phase_failed_count, 0);

    let key1 = (sat1, 0, epochs[0].observations[0].ambiguity_id.clone());
    let key2 = (sat2, 0, epochs[0].observations[1].ambiguity_id.clone());
    assert_eq!(report.observation_reports[0].sat, sat1);
    assert_eq!(
        report.observation_reports[0].code_status,
        SsrIfCombinationStatus::Applied
    );
    assert_eq!(
        report.observation_reports[0].phase_status,
        SsrIfCombinationStatus::Applied
    );
    assert!(lookup.ssr_code_bias_m.contains_key(&key1));
    assert!(lookup.phase_bias_m.contains_key(&key1));

    assert_eq!(report.observation_reports[1].sat, sat2);
    assert_eq!(
        report.observation_reports[1].code_status,
        SsrIfCombinationStatus::SignalUnavailable
    );
    assert_eq!(
        report.observation_reports[1].phase_status,
        SsrIfCombinationStatus::Applied
    );
    assert!(!lookup.ssr_code_bias_m.contains_key(&key2));
    assert!(lookup.phase_bias_m.contains_key(&key2));

    let actual_code = lookup.ssr_code_bias_m[&key1];
    assert!((actual_code + ionosphere_free(code_m[0], code_m[1])).abs() < 1.0e-8);

    // Only sat2 is left out, for its code bias alone.
    let (retained, exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
        &ephemeris,
        &epochs,
        0,
        ssr_test_receiver(),
        &lookup,
        0,
        SsrBiasExclusionStage::BeforeSolve,
    );
    assert_eq!(retained[0].observations, epochs[0].observations[..1]);
    assert_eq!(exclusions.len(), 1);
    assert_eq!(exclusions[0].satellite_id, sat2.to_string());
    assert!(exclusions[0].code_bias_missing);
    assert!(!exclusions[0].phase_bias_missing);
}

#[test]
fn test_ppp_ssr_biases_opt_out_incompatible_pairs_and_independent_ambiguities() {
    let (_source, mut epochs, mut state, _ambiguity_ids) = ssr_test_arc(1);
    let sat = epochs[0].observations[0].sat;
    let t0 = ssr_test_t0();

    // Split arcs G01#1 and G01#2 of one satellite in one epoch.
    let mut obs1 = epochs[0].observations[0].clone();
    obs1.ambiguity_id = "G01#1".to_string();
    let mut obs2 = epochs[0].observations[0].clone();
    obs2.ambiguity_id = "G01#2".to_string();
    epochs[0].observations = vec![obs1.clone(), obs2.clone()];
    state.ambiguities_m = initial_ambiguities(&epochs);

    let phase_cycles = [0.25, -0.15];
    let first = has_test_message(
        &[sat],
        0,
        1,
        1,
        Some(HAS_VI_60_S),
        Some(HasTestBiases::usable([1.0, -0.5], phase_cycles)),
    );
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(&mut store, &first);

    // Tokens of the arc before the break.
    let pre_break_token0 = store
        .query_phase_bias(sat, 0, t0, None)
        .continuity_token
        .expect("token for the first arc, signal 0");
    let pre_break_token9 = store
        .query_phase_bias(sat, 9, t0, None)
        .continuity_token
        .expect("token for the first arc, signal 9");

    // A second message 60 s later with PDI 1 breaks the arc.
    let mut second = first.clone();
    second.header.toh_s = 60;
    for record in &mut second.phase_bias.as_mut().unwrap().records {
        record.discontinuity_indicator = 1;
    }
    has_test_ingest(&mut store, &second);

    // A query without a token starts a new arc: not a reset, with the PDI change in the
    // details. Its token acknowledges the break.
    let q_p1 = store.query_phase_bias(sat, 0, t0 + 60.0, None);
    assert_eq!(q_p1.status, crate::ssr::SsrBiasStatus::Available);
    assert_eq!(
        q_p1.discontinuity_details,
        Some(crate::ssr::SsrDiscontinuityDetails::HasPdiChanged {
            previous: 0,
            current: 1
        })
    );
    let token_reset = q_p1
        .continuity_token
        .expect("token for the new arc, signal 0");
    let q_p2 = store.query_phase_bias(sat, 9, t0 + 60.0, None);
    assert_eq!(q_p2.status, crate::ssr::SsrBiasStatus::Available);
    let token_reset2 = q_p2
        .continuity_token
        .expect("token for the new arc, signal 9");

    // G01#1 acknowledges the new arc; G01#2 still holds the tokens from before the break.
    let mut epochs_60 = epochs.clone();
    epochs_60[0].t_rx_j2000_s = t0 + 61.0;
    let options_ack = gps_l1_l2_options()
        .with_phase_continuity_token("G01#1", 0, token_reset)
        .with_phase_continuity_token("G01#1", 9, token_reset2)
        .with_phase_continuity_token("G01#2", 0, pre_break_token0)
        .with_phase_continuity_token("G01#2", 9, pre_break_token9);
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);

    let key1 = (sat, 0, "G01#1".to_string());
    let key2 = (sat, 0, "G01#2".to_string());
    let expected_phase_if = ionosphere_free(
        phase_cycles[0] * (C_M_S / F_L1_HZ),
        phase_cycles[1] * (C_M_S / F_L2_HZ),
    );
    for order in [[&obs1, &obs2], [&obs2, &obs1]] {
        let mut ordered = epochs_60.clone();
        ordered[0].observations = order.iter().map(|obs| (*obs).clone()).collect();
        let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
            &ephemeris,
            &ordered,
            ssr_test_receiver(),
            &options_ack,
        );
        assert_eq!(report.phase_discontinuity_resets_needed, 1);
        for row in &report.observation_reports {
            let expected = if row.ambiguity_id == "G01#1" {
                SsrIfCombinationStatus::Applied
            } else {
                SsrIfCombinationStatus::PhaseDiscontinuityNeedsReset
            };
            assert_eq!(row.phase_status, expected, "{}", row.ambiguity_id);
            assert_eq!(row.code_status, SsrIfCombinationStatus::Applied);
        }
        // Code biases are held per ambiguity; phase only for the acknowledged arc.
        assert!(lookup.ssr_code_bias_m.contains_key(&key1));
        assert!(lookup.ssr_code_bias_m.contains_key(&key2));
        assert!(lookup.phase_bias_m.contains_key(&key1));
        assert!(!lookup.phase_bias_m.contains_key(&key2));
        let stored_phase_if = lookup.phase_bias_m[&key1];
        assert!(
            (stored_phase_if - expected_phase_if).abs() < 1.0e-9,
            "stored {stored_phase_if}, expected {expected_phase_if}"
        );

        let corrections = RangeCorrections {
            ppp: lookup.clone(),
            ..RangeCorrections::disabled()
        };
        assert_eq!(
            super::model::phase_bias_m(&obs1, 0, &corrections).unwrap(),
            stored_phase_if
        );
        assert_missing_correction(
            super::model::phase_bias_m(&obs2, 0, &corrections).unwrap_err(),
            sat,
            MissingCorrection::PhaseBias,
        );

        // A solve leaves G01#2 out and keeps G01#1, whichever comes first.
        let (retained, exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
            &ephemeris,
            &ordered,
            0,
            ssr_test_receiver(),
            &lookup,
            0,
            SsrBiasExclusionStage::BeforeSolve,
        );
        assert_eq!(retained[0].observations, vec![obs1.clone()]);
        assert_eq!(exclusions.len(), 1);
        assert_eq!(exclusions[0].ambiguity_id, "G01#2");
        assert!(!exclusions[0].code_bias_missing);
        assert!(exclusions[0].phase_bias_missing);
        assert_eq!(
            exclusions[0].application.as_ref().unwrap().phase_status,
            SsrIfCombinationStatus::PhaseDiscontinuityNeedsReset
        );
        let ids = vec![AmbiguityId::new("G01#1".to_string())];
        let binding = super::rows::AmbiguityBinding::Estimated {
            ids: &ids,
            values: &state.ambiguities_m,
        };
        let rows = super::rows::build_rows(
            row_trace_ctx(&ephemeris, &corrections),
            &retained,
            &binding,
            &state,
        )
        .expect("rows build for the acknowledged arc");
        assert_eq!(rows.len(), 2, "one code row and one phase row");
    }

    let options_code_only = options_ack.clone().with_apply_phase_biases(false);
    let (lookup_code_only, report_code_only) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs_60,
        ssr_test_receiver(),
        &options_code_only,
    );
    assert!(lookup_code_only.ssr_code_bias_enabled);
    assert!(!lookup_code_only.phase_bias_enabled);
    for row in &report_code_only.observation_reports {
        assert_eq!(row.phase_status, SsrIfCombinationStatus::OptedOut);
    }

    let options_phase_only = options_ack.clone().with_apply_code_biases(false);
    let (lookup_phase_only, report_phase_only) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs_60,
        ssr_test_receiver(),
        &options_phase_only,
    );
    assert!(!lookup_phase_only.ssr_code_bias_enabled);
    assert!(lookup_phase_only.phase_bias_enabled);
    assert_eq!(
        report_phase_only.observation_reports[0].code_status,
        SsrIfCombinationStatus::OptedOut
    );

    // Equal carrier frequencies cannot form an ionosphere-free combination.
    let options_bad_freq = SsrPppBiasOptions::default().with_system_signal_pair(
        GnssSystem::Gps,
        SsrPppBiasSignalPair {
            freq2_hz: F_L1_HZ,
            ..gps_l1_l2_signal_pair()
        },
    );
    let (_lookup_bad_freq, report_bad_freq) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs_60,
        ssr_test_receiver(),
        &options_bad_freq,
    );
    assert_eq!(
        report_bad_freq.observation_reports[0].code_status,
        SsrIfCombinationStatus::InvalidFrequencies
    );
    assert_eq!(
        report_bad_freq.observation_reports[0].phase_status,
        SsrIfCombinationStatus::InvalidFrequencies
    );
}

/// The SSR code bias is held per ambiguity: with the signal pair's frequencies unset, each
/// observation's own frequencies form the combination, so two ambiguities of one satellite
/// in one epoch can differ. Here one has valid frequencies and the other equal ones; the
/// valid one is applied and only the other is left out.
#[test]
fn test_ppp_ssr_code_bias_is_held_per_ambiguity() {
    let (_source, mut epochs, _state, _ambiguity_ids) = ssr_test_arc(1);
    let sat = epochs[0].observations[0].sat;
    let mut valid = epochs[0].observations[0].clone();
    valid.ambiguity_id = "G01#1".to_string();
    valid.freq1_hz = F_L1_HZ;
    valid.freq2_hz = F_L2_HZ;
    let mut equal = epochs[0].observations[0].clone();
    equal.ambiguity_id = "G01#2".to_string();
    equal.freq1_hz = F_L1_HZ;
    equal.freq2_hz = F_L1_HZ;
    epochs[0].observations = vec![valid.clone(), equal.clone()];

    let code_m = [1.24, -0.76];
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable(code_m, [0.2, -0.3])),
        ),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let options = SsrPppBiasOptions::new().with_system_signal_pair(
        GnssSystem::Gps,
        SsrPppBiasSignalPair {
            freq1_hz: 0.0,
            freq2_hz: 0.0,
            ..gps_l1_l2_signal_pair()
        },
    );
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &options,
    );

    assert_eq!(
        report.observation_reports[0].code_status,
        SsrIfCombinationStatus::Applied
    );
    assert_eq!(
        report.observation_reports[1].code_status,
        SsrIfCombinationStatus::InvalidFrequencies
    );
    let valid_key = (sat, 0, "G01#1".to_string());
    let equal_key = (sat, 0, "G01#2".to_string());
    assert!(
        (lookup.ssr_code_bias_m[&valid_key] + ionosphere_free(code_m[0], code_m[1])).abs() < 1.0e-8
    );
    assert!(!lookup.ssr_code_bias_m.contains_key(&equal_key));

    let corrections = RangeCorrections {
        ppp: lookup.clone(),
        ..RangeCorrections::disabled()
    };
    assert_eq!(
        super::model::ssr_code_bias_m(&valid, 0, &corrections.ppp).unwrap(),
        lookup.ssr_code_bias_m[&valid_key]
    );
    assert_missing_correction(
        super::model::ssr_code_bias_m(&equal, 0, &corrections.ppp).unwrap_err(),
        sat,
        MissingCorrection::SsrCodeBias,
    );

    let (retained, exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
        &ephemeris,
        &epochs,
        0,
        ssr_test_receiver(),
        &lookup,
        0,
        SsrBiasExclusionStage::BeforeSolve,
    );
    assert_eq!(retained[0].observations, vec![valid]);
    assert_eq!(exclusions.len(), 1);
    assert_eq!(exclusions[0].ambiguity_id, "G01#2");
    assert!(exclusions[0].code_bias_missing);
}

/// A bias is applied only against the orbit and clock corrections of its own SSR solution,
/// as the ephemeris source applies them at the epoch.
///
/// Both satellites receive biases, orbits and clocks under mask 1 / IOD set 1 at t0; sat2
/// then receives an orbit and clock alone under mask 2 / IOD set 2 at t0 + 30 s. At t0 + 31
/// its biases (valid for 60 s) are still available, but they belong to another solution
/// than the orbit and clock the ephemeris applies, so code and phase are refused and the
/// observation is left out. Biases with no orbit and clock at all are refused the same way.
#[test]
fn test_ppp_ssr_biases_refuse_bias_from_other_solution_than_orbit_clock() {
    let (_source, mut epochs, _state, _ambiguity_ids) = ssr_test_arc(2);
    let t = ssr_test_t0() + 31.0;
    epochs[0].t_rx_j2000_s = t;
    let sat1 = epochs[0].observations[0].sat;
    let sat2 = epochs[0].observations[1].sat;
    let biases = Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3]));

    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(&[sat1, sat2], 0, 1, 1, Some(HAS_VI_60_S), biases),
    );
    has_test_ingest(
        &mut store,
        &has_test_message(&[sat2], 30, 2, 2, Some(HAS_VI_60_S), None),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    assert_eq!(
        ephemeris.applied_orbit_clock_solution(sat1, t),
        Some(has_solution(1, 1))
    );
    assert_eq!(
        ephemeris.applied_orbit_clock_solution(sat2, t),
        Some(has_solution(2, 2))
    );

    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    assert_eq!(report.status, SsrPppAggregateStatus::PartiallyApplied);
    assert_eq!(report.code_applied_count, 1);
    assert_eq!(report.code_failed_count, 1);
    assert_eq!(report.phase_applied_count, 1);
    assert_eq!(report.phase_failed_count, 1);
    assert_eq!(report.observation_reports.len(), 2);

    let applied = &report.observation_reports[0];
    assert_eq!(applied.sat, sat1);
    assert_eq!(applied.code_status, SsrIfCombinationStatus::Applied);
    assert_eq!(applied.phase_status, SsrIfCombinationStatus::Applied);

    let refused = &report.observation_reports[1];
    assert_eq!(refused.sat, sat2);
    assert_eq!(
        refused.applied_orbit_clock_solution,
        Some(has_solution(2, 2))
    );
    assert_eq!(
        refused.code_status,
        SsrIfCombinationStatus::OrbitClockSolutionMismatch
    );
    assert_eq!(
        refused.phase_status,
        SsrIfCombinationStatus::OrbitClockSolutionMismatch
    );
    assert!(refused.applied_code_if_m.is_none());
    assert!(refused.applied_phase_if_m.is_none());
    for query in [
        &refused.code1_report.as_ref().unwrap().query_result,
        &refused.code2_report.as_ref().unwrap().query_result,
    ] {
        assert_eq!(query.status, crate::ssr::SsrBiasStatus::Available);
        assert_eq!(query.solution, Some(has_solution(1, 1)));
    }
    for query in [
        &refused.phase1_report.as_ref().unwrap().query_result,
        &refused.phase2_report.as_ref().unwrap().query_result,
    ] {
        assert_eq!(query.status, crate::ssr::SsrBiasStatus::Available);
        assert_eq!(query.solution, Some(has_solution(1, 1)));
    }

    let (retained, exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
        &ephemeris,
        &epochs,
        0,
        ssr_test_receiver(),
        &lookup,
        0,
        SsrBiasExclusionStage::BeforeSolve,
    );
    assert_eq!(retained[0].observations, epochs[0].observations[..1]);
    assert_eq!(exclusions.len(), 1);
    assert_eq!(exclusions[0].satellite_id, sat2.to_string());
    assert_eq!(exclusions[0].application.as_ref(), Some(refused));

    // Biases with no orbit or clock correction have no solution to be applied against. A
    // source with a broadcast fallback still places the satellites, on broadcast clocks.
    let mut bias_only = SsrCorrectionStore::new();
    has_test_ingest(
        &mut bias_only,
        &has_test_message(&[sat1, sat2], 0, 1, 1, None, biases),
    );
    let bias_only_ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &bias_only)
        .with_fallback(broadcast_fallback());
    let (bias_only_lookup, bias_only_report) = PppCorrectionLookup::default().with_ssr_biases(
        &bias_only_ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    assert_eq!(bias_only_report.status, SsrPppAggregateStatus::NoneApplied);
    for row in &bias_only_report.observation_reports {
        assert_eq!(row.applied_orbit_clock_solution, None);
        assert_eq!(
            row.code_status,
            SsrIfCombinationStatus::OrbitClockSolutionUnavailable
        );
        assert_eq!(
            row.phase_status,
            SsrIfCombinationStatus::OrbitClockSolutionUnavailable
        );
    }
    assert!(bias_only_lookup.ssr_code_bias_m.is_empty());
    assert!(bias_only_lookup.phase_bias_m.is_empty());

    // A source that declines satellites without SSR orbit and clock cannot place them at
    // all, so their transmission time is unavailable.
    let declining = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &bias_only);
    let (_, declined_report) = PppCorrectionLookup::default().with_ssr_biases(
        &declining,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    for row in &declined_report.observation_reports {
        assert_eq!(row.transmit_time_j2000_s, None);
        assert_eq!(
            row.code_status,
            SsrIfCombinationStatus::TransmitTimeUnavailable
        );
    }
}

/// A fresh bias is not applied when its orbit and clock have expired, even though a
/// fallback policy lets the ephemeris source return the plain broadcast state: the solve
/// would then combine the SSR bias with a broadcast clock.
#[test]
fn test_ppp_ssr_biases_refuse_fresh_bias_over_expired_orbit_clock_with_broadcast_fallback() {
    let (_source, mut epochs, _state, _ambiguity_ids) = ssr_test_arc(1);
    let t = ssr_test_t0() + 30.0;
    epochs[0].t_rx_j2000_s = t;
    let sat = epochs[0].observations[0].sat;

    // Orbit and clock valid for 5 s, biases for 60 s.
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_5_S),
            Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3])),
        ),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store).with_fallback(
        crate::ssr::SsrFallbackPolicy {
            on_missing_correction: crate::ssr::MissingCorrectionAction::FallBackToBroadcast,
            regional: crate::ssr::RegionalPolicy::DeclineRegional,
        },
    );
    assert_eq!(
        ephemeris.corrected_state(sat, t),
        crate::spp::EphemerisSource::position_clock_at_j2000_s(&broadcast, sat, t),
        "the source falls back to the broadcast state"
    );
    assert!(ephemeris.corrected_state(sat, t).is_some());
    assert_eq!(ephemeris.applied_orbit_clock_solution(sat, t), None);
    assert_eq!(
        ephemeris.applied_orbit_clock_solution(sat, ssr_test_t0()),
        Some(has_solution(1, 1)),
        "within the orbit and clock validity the SSR solution applies"
    );

    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    let row = &report.observation_reports[0];
    assert_eq!(row.applied_orbit_clock_solution, None);
    assert_eq!(
        row.code_status,
        SsrIfCombinationStatus::OrbitClockSolutionUnavailable
    );
    assert_eq!(
        row.phase_status,
        SsrIfCombinationStatus::OrbitClockSolutionUnavailable
    );
    assert_eq!(
        row.code1_report.as_ref().unwrap().query_result.status,
        crate::ssr::SsrBiasStatus::Available
    );
    assert_eq!(
        row.phase1_report.as_ref().unwrap().query_result.status,
        crate::ssr::SsrBiasStatus::Available
    );
    assert_eq!(report.status, SsrPppAggregateStatus::NoneApplied);
    assert!(lookup.ssr_code_bias_m.is_empty());
    assert!(lookup.phase_bias_m.is_empty());
}

/// An epoch received exactly on the TOH of new corrections cannot use them: the signal
/// left before their reference epoch, when the biases are not yet valid and a source with
/// a broadcast fallback returns the broadcast state. The bias is refused rather than
/// applied to a broadcast clock; one second later it applies.
#[test]
fn test_ppp_ssr_biases_refuse_epoch_on_the_toh_under_broadcast_fallback() {
    let (_source, mut epochs, _state, _ambiguity_ids) = ssr_test_arc(1);
    let t0 = ssr_test_t0();
    epochs[0].t_rx_j2000_s = t0;
    let sat = epochs[0].observations[0].sat;
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3])),
        ),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());
    let t_tx = t0 - 0.075;
    assert_eq!(
        ephemeris.applied_orbit_clock_solution(sat, t0),
        Some(has_solution(1, 1))
    );
    assert_eq!(ephemeris.applied_orbit_clock_solution(sat, t_tx), None);
    assert_eq!(
        ephemeris.corrected_state(sat, t_tx),
        crate::spp::EphemerisSource::position_clock_at_j2000_s(&broadcast, sat, t_tx),
        "the signal left before the TOH, when the source returns the broadcast state"
    );

    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    let row = &report.observation_reports[0];
    assert!(row.transmit_time_j2000_s.is_some_and(|t| t < t0));
    assert_eq!(row.applied_orbit_clock_solution, None);
    assert_eq!(row.code_status, SsrIfCombinationStatus::SignalUnavailable);
    assert_eq!(row.phase_status, SsrIfCombinationStatus::SignalUnavailable);
    assert!(lookup.ssr_code_bias_m.is_empty());
    assert!(lookup.phase_bias_m.is_empty());

    epochs[0].t_rx_j2000_s = t0 + 1.0;
    let (_, later) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    assert_eq!(later.status, SsrPppAggregateStatus::AllApplied);
    assert_eq!(
        later.observation_reports[0].applied_orbit_clock_solution,
        Some(has_solution(1, 1))
    );
}

fn broadcast_fallback() -> crate::ssr::SsrFallbackPolicy {
    crate::ssr::SsrFallbackPolicy {
        on_missing_correction: crate::ssr::MissingCorrectionAction::FallBackToBroadcast,
        regional: crate::ssr::RegionalPolicy::DeclineRegional,
    }
}

/// Transmission time of `obs` for reception at `t_rx`, predicted as the solve predicts it:
/// seeded from its pseudorange.
fn ssr_test_transmit_time(
    ephemeris: &crate::ssr::SsrCorrectedEphemeris<'_>,
    obs: &FloatObservation,
    t_rx: f64,
) -> f64 {
    crate::observables::transmit_epoch_j2000_s(
        ephemeris,
        obs.sat,
        ssr_test_receiver(),
        t_rx,
        crate::observables::TransmitTimeOptions::default(),
        crate::observables::flight_time_seed_s(obs.code_m),
    )
    .expect("transmission time")
}

/// A bias whose TOH falls between the transmission and the reception of a signal is not
/// yet valid when the signal left, so it is refused even though it is valid at reception.
#[test]
fn test_ppp_ssr_biases_refuse_toh_between_transmission_and_reception() {
    let (_source, mut epochs, _state, _ambiguity_ids) = ssr_test_arc(1);
    let t0 = ssr_test_t0();
    let t_rx = t0 + 0.03;
    epochs[0].t_rx_j2000_s = t_rx;
    let sat = epochs[0].observations[0].sat;
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3])),
        ),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());
    let t_tx = ssr_test_transmit_time(&ephemeris, &epochs[0].observations[0], t_rx);
    assert!(t_tx < t0 && t0 < t_rx, "t_tx {t_tx}, t0 {t0}, t_rx {t_rx}");

    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    let row = &report.observation_reports[0];
    assert_eq!(row.transmit_time_j2000_s, Some(t_tx));
    assert_eq!(row.applied_orbit_clock_solution, None);
    assert_eq!(row.code_status, SsrIfCombinationStatus::SignalUnavailable);
    assert_eq!(row.phase_status, SsrIfCombinationStatus::SignalUnavailable);
    assert_eq!(
        row.code1_report.as_ref().unwrap().query_result.status,
        crate::ssr::SsrBiasStatus::NotYetValid
    );
    assert_eq!(
        row.phase1_report.as_ref().unwrap().query_result.status,
        crate::ssr::SsrBiasStatus::NotYetValid
    );
    assert!(lookup.ssr_code_bias_m.is_empty());
    assert!(lookup.phase_bias_m.is_empty());
}

/// A MEO signal received 0.1 s after the TOH left after it too, so its biases apply,
/// and the solve on the same source keeps the observation: its recorded biases hold at
/// the transmission time.
#[test]
fn test_ppp_ssr_biases_accept_meo_epoch_transmitted_after_toh() {
    let (_source, mut epochs, state, ambiguity_ids) = ssr_test_arc(1);
    let t0 = ssr_test_t0();
    let t_rx = t0 + 0.1;
    epochs[0].t_rx_j2000_s = t_rx;
    let sat = epochs[0].observations[0].sat;
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3])),
        ),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let t_tx = ssr_test_transmit_time(&ephemeris, &epochs[0].observations[0], t_rx);
    assert!(
        t0 < t_tx && t_tx < t_rx,
        "t_tx {t_tx}, t0 {t0}, t_rx {t_rx}"
    );

    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    assert_eq!(report.status, SsrPppAggregateStatus::AllApplied);
    assert_eq!(
        report.observation_reports[0].transmit_time_j2000_s,
        Some(t_tx)
    );

    let (retained, exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
        &ephemeris,
        &epochs,
        0,
        ssr_test_receiver(),
        &lookup,
        0,
        SsrBiasExclusionStage::BeforeSolve,
    );
    assert!(exclusions.is_empty());
    assert_eq!(retained, epochs);
    assert_eq!(
        super::rows::ssr_bias_records_hold(
            &ephemeris,
            &epochs[0].observations[0],
            0,
            &lookup,
            Some(t_tx),
        ),
        Ok(())
    );
    // The rows need the source only at the transmission time, not half a second before
    // it, before the TOH, where this source declines the satellite.
    let corrections = RangeCorrections {
        ppp: lookup,
        ..RangeCorrections::disabled()
    };
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let rows = super::rows::build_rows(
        row_trace_ctx(&ephemeris, &corrections),
        &epochs,
        &binding,
        &state,
    )
    .expect("the biases hold at the transmission time");
    assert_eq!(rows.len(), 2);
}

/// A satellite a HAS do-not-use indication excludes is reported as `SatelliteExcluded`,
/// with the indication in each signal report, not as an unavailable signal.
#[test]
fn test_ppp_ssr_biases_report_do_not_use_as_satellite_excluded() {
    let (_source, epochs, _state, _ambiguity_ids) = ssr_test_arc(1);
    let sat = epochs[0].observations[0].sat;
    let mut message = has_test_message(
        &[sat],
        0,
        1,
        1,
        Some(HAS_VI_60_S),
        Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3])),
    );
    for record in &mut message.clock_full_set.as_mut().unwrap().records {
        record.correction_m = None;
        record.do_not_use = true;
    }
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(&mut store, &message);
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());

    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    let row = &report.observation_reports[0];
    assert_eq!(row.transmit_time_j2000_s, None);
    assert_eq!(row.code_status, SsrIfCombinationStatus::SatelliteExcluded);
    assert_eq!(row.phase_status, SsrIfCombinationStatus::SatelliteExcluded);
    let query = &row.code1_report.as_ref().unwrap().query_result;
    assert_eq!(query.status, crate::ssr::SsrBiasStatus::Excluded);
    assert!(matches!(
        query.details,
        crate::ssr::SsrBiasResolutionDetails::ExcludedByDoNotUse { .. }
    ));
    assert!(lookup.ssr_code_bias_m.is_empty());
    assert!(lookup.phase_bias_m.is_empty());
}

/// Biases resolved against an SSR-corrected source are left out of a solve on a source
/// that applies no SSR corrections, which has no orbit and clock of their solution.
#[test]
fn test_ppp_ssr_biases_left_out_of_a_solve_on_a_source_without_ssr() {
    let (source, epochs, state, ambiguity_ids) = ssr_test_arc(1);
    let sat = epochs[0].observations[0].sat;
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3])),
        ),
    );
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    assert_eq!(report.status, SsrPppAggregateStatus::AllApplied);

    let (retained, exclusions) = super::rows::exclude_unresolved_ssr_bias_observations(
        &source,
        &epochs,
        0,
        ssr_test_receiver(),
        &lookup,
        0,
        SsrBiasExclusionStage::BeforeSolve,
    );
    assert!(retained[0].observations.is_empty());
    assert_eq!(exclusions.len(), 1);
    assert!(!exclusions[0].code_bias_missing);
    assert!(!exclusions[0].phase_bias_missing);
    assert_eq!(
        exclusions[0].transmit_time_failure,
        Some(SsrTransmitTimeFailure::SourceWithoutSsrCorrections)
    );

    let corrections = RangeCorrections {
        ppp: lookup,
        ..RangeCorrections::disabled()
    };
    let binding = super::rows::AmbiguityBinding::Estimated {
        ids: &ambiguity_ids,
        values: &state.ambiguities_m,
    };
    let err = super::rows::build_rows(
        row_trace_ctx(&source, &corrections),
        &epochs,
        &binding,
        &state,
    )
    .unwrap_err()
    .into_float();
    assert_missing_correction(err, sat, MissingCorrection::SsrCodeBias);
}

/// Broadcast ephemeris with `prns` spread over distinct orbits: the fixture's G31 record,
/// with its mean anomaly and right ascension offset per satellite so the satellites stand
/// apart in one part of the sky.
fn ssr_spread_broadcast(prns: &[u8]) -> crate::ephemeris::BroadcastEphemeris {
    let text = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/ssr/BRDC00WRD_S_20261820000_G30_G31.rnx"
    ));
    let header_label = text.find("END OF HEADER").expect("NAV header end");
    let body_start = header_label + text[header_label..].find('\n').expect("header line end") + 1;
    let record_start = text.find("\nG31 ").expect("G31 record") + 1;
    let record_end = record_start + text[record_start..].find("\nG30 ").expect("G30 record") + 1;
    let record = &text[record_start..record_end];
    // RINEX writes a 19-character field with a two-digit signed exponent.
    let field = |value: f64| {
        let text = format!("{value:.12e}");
        let (mantissa, exponent) = text.split_once('e').expect("exponent");
        let exponent: i32 = exponent.parse().expect("exponent value");
        format!("{:>19}", format!("{mantissa}e{exponent:+03}"))
    };
    let mut nav = text[..body_start].to_string();
    for (index, prn) in prns.iter().enumerate() {
        let spread = index as f64 - (prns.len() as f64 - 1.0) / 2.0;
        let mean_anomaly = -3.407_619_013_467e-1 + 0.25 * spread;
        let right_ascension = -2.186_740_086_731 + 0.2 * libm::sin(spread * 1.7);
        nav.push_str(
            &record
                .replacen("G31", &format!("G{prn:02}"), 1)
                .replacen("-3.407619013467e-01", &field(mean_anomaly), 1)
                .replacen("-2.186740086731e+00", &field(right_ascension), 1),
        );
    }
    crate::ephemeris::BroadcastEphemeris::from_nav(&nav).expect("parse spread NAV")
}

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
}

/// A receiver on the Earth's surface under the satellites' mean position at `t`.
fn receiver_under(
    broadcast: &crate::ephemeris::BroadcastEphemeris,
    prns: &[u8],
    t: f64,
) -> [f64; 3] {
    let mut sum = [0.0; 3];
    for prn in prns {
        let (position, _) =
            crate::spp::EphemerisSource::position_clock_at_j2000_s(broadcast, gps(*prn), t)
                .expect("broadcast state");
        for axis in 0..3 {
            sum[axis] += position[axis];
        }
    }
    scale3(unit3(sum).expect("mean direction"), 6_378_137.0)
}

/// One epoch of noise-free observations of `prns` from `receiver` at `t_rx`, modelled on
/// `source` as the PPP rows model them: code = range + clock - satellite clock, phase =
/// code + ambiguity.
fn ssr_spread_epoch(
    source: &dyn ObservableEphemerisSource,
    prns: &[u8],
    receiver: [f64; 3],
    t_rx: f64,
    clock_m: f64,
) -> FloatEpoch {
    let observations = prns
        .iter()
        .enumerate()
        .map(|(index, prn)| {
            let sat = gps(*prn);
            let geometry = crate::observables::predict_transmit_geometry(
                source,
                sat,
                receiver,
                t_rx,
                PredictOptions {
                    carrier_hz: F_L1_HZ,
                    light_time: true,
                    sagnac: true,
                },
                crate::observables::NOMINAL_SIGNAL_FLIGHT_TIME_S,
            )
            .expect("transmit geometry");
            let code = geometry.geometric_range_m + clock_m
                - C_M_S * geometry.sat_clock_s.expect("satellite clock");
            FloatObservation {
                sat,
                satellite_id: sat.to_string(),
                ambiguity_id: sat.to_string(),
                code_m: code,
                phase_m: code + 0.3 + 0.1 * index as f64,
                freq1_hz: 0.0,
                freq2_hz: 0.0,
                glonass_channel: None,
            }
        })
        .collect();
    FloatEpoch {
        epoch: CivilDateTime {
            year: 2026,
            month: 7,
            day: 2,
            hour: 0,
            minute: 0,
            second: 0.0,
        },
        jd_whole: 2_461_223.5,
        jd_fraction: 0.0,
        t_rx_j2000_s: t_rx,
        observations,
    }
}

fn ssr_spread_config(corrections: RangeCorrections, cutoff_deg: Option<f64>) -> FloatSolveConfig {
    FloatSolveConfig {
        weights: MeasurementWeights {
            code: 1.0,
            phase: 100.0,
            elevation_weighting: false,
        },
        tropo: TroposphereOptions::disabled(),
        corrections,
        opts: FloatSolveOptions {
            max_iterations: 10,
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

fn ssr_spread_state(epoch: &FloatEpoch, position_m: [f64; 3]) -> FloatState {
    FloatState {
        position_m,
        clocks_m: vec![0.0],
        ambiguities_m: initial_ambiguities(std::slice::from_ref(epoch)),
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    }
}

/// An observation received 0.3 s after the TOH was transmitted within half a second of it,
/// before which a declining source has no state. The rows need the source only at the
/// transmission time, so the solve succeeds, and prediction takes the velocity from the
/// broadcast record instead of differencing across the TOH.
#[test]
fn ssr_decline_source_solves_within_half_a_second_after_the_toh() {
    let prns = [1, 2, 3, 4, 5];
    let sats = prns.map(gps);
    let t0 = ssr_test_t0();
    let broadcast = ssr_spread_broadcast(&prns);
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(&sats, 0, 1, 1, Some(HAS_VI_60_S), None),
    );
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let truth = receiver_under(&broadcast, &prns, t0);
    let t_rx = t0 + 0.3;
    let epoch = ssr_spread_epoch(&ephemeris, &prns, truth, t_rx, 12.5);
    let state = ssr_spread_state(
        &epoch,
        [truth[0] + 300.0, truth[1] - 200.0, truth[2] + 100.0],
    );

    let solution = solve_float_epoch(
        &ephemeris,
        epoch,
        state,
        ssr_spread_config(RangeCorrections::disabled(), None),
    )
    .expect("the solve needs the source only at the transmission times");
    let error = norm3(sub3(solution.position_m, truth));
    assert!(error < 1.0e-3, "position error {error}");

    let prediction = predict(
        &ephemeris,
        sats[0],
        truth,
        t_rx,
        PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: true,
            sagnac: true,
        },
    )
    .expect("prediction takes the SSR state's own velocity");
    assert!(prediction.range_rate_m_s.is_finite());
}

/// Where a source with a broadcast fallback applies SSR corrections at an epoch but not
/// half a second earlier, its velocity is the SSR state's (the IODE record's, as RTKLIB
/// `satpos_ssr` takes it), not a difference between a broadcast and an SSR position.
#[test]
fn ssr_fallback_source_velocity_is_the_ssr_state_velocity() {
    let prns = [1];
    let sat = gps(1);
    let t0 = ssr_test_t0();
    let broadcast = ssr_spread_broadcast(&prns);
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(&[sat], 0, 1, 1, Some(HAS_VI_60_S), None),
    );
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());
    let t = t0 + 0.2;
    assert_eq!(
        ephemeris.applied_orbit_clock_solution(sat, t),
        Some(has_solution(1, 1))
    );
    assert_eq!(ephemeris.applied_orbit_clock_solution(sat, t - 0.5), None);

    let velocity = ephemeris
        .velocity_at_j2000_s(sat, t)
        .expect("the source defines its velocity")
        .expect("velocity");
    assert_eq!(ephemeris.corrected_velocity(sat, t), Some(velocity));
    // RTKLIB `ephpos` on the IODE record: its positions at `tk` and `tk` + 1 ms.
    let record = broadcast
        .select_record_at(sat, t)
        .expect("broadcast record");
    let sow =
        (t + crate::constants::GPS_EPOCH_TO_J2000_S).rem_euclid(crate::constants::SECONDS_PER_WEEK);
    let tk = sow - record.elements.toe_sow;
    let position = |tk_s: f64| {
        crate::broadcast::satellite_position_ecef_at_tk_unchecked(
            &record.elements,
            None,
            &record.constants(),
            tk_s,
            false,
        )
        .position()
        .expect("finite position")
        .as_array()
    };
    let (start, end) = (position(tk), position(tk + 1.0e-3));
    for axis in 0..3 {
        let expected = (end[axis] - start[axis]) / 1.0e-3;
        assert_eq!(velocity[axis].to_bits(), expected.to_bits(), "axis {axis}");
    }

    let (after, _) = ephemeris.corrected_state(sat, t + 0.5).expect("SSR state");
    let (before, _) = ephemeris
        .corrected_state(sat, t - 0.5)
        .expect("broadcast state");
    let blend = sub3(after, before);
    assert!(
        norm3(sub3(blend, velocity)) > 1.0,
        "the blend differs from the SSR velocity by the correction step"
    );
}

/// SSR bias exclusion runs before the elevation cutoff: a satellite the source cannot
/// place has no resolved biases and is left out first, instead of failing the cutoff's
/// geometry, and the rest of the arc solves.
#[test]
fn ssr_declined_satellite_is_excluded_before_the_elevation_cutoff() {
    let prns = [1, 2, 3, 4, 5, 6];
    let corrected = [1, 2, 3, 4, 5].map(gps);
    let t0 = ssr_test_t0();
    let broadcast = ssr_spread_broadcast(&prns);
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &corrected,
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable([0.0, 0.0], [0.0, 0.0])),
        ),
    );
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let truth = receiver_under(&broadcast, &prns, t0);
    let t_rx = t0 + 1.0;
    let mut epoch = ssr_spread_epoch(&ephemeris, &prns[..5], truth, t_rx, 12.5);
    // G06 has no SSR correction, so the declining source cannot place it; its observation
    // comes from the broadcast orbit.
    epoch
        .observations
        .extend(ssr_spread_epoch(&broadcast, &prns[5..], truth, t_rx, 12.5).observations);
    let start = [truth[0] + 300.0, truth[1] - 200.0, truth[2] + 100.0];
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        std::slice::from_ref(&epoch),
        start,
        &gps_l1_l2_options(),
    );
    assert_eq!(
        report.observation_reports[5].code_status,
        SsrIfCombinationStatus::TransmitTimeUnavailable
    );
    let state = ssr_spread_state(&epoch, start);
    let solution = solve_float_epoch(
        &ephemeris,
        epoch,
        state,
        ssr_spread_config(
            RangeCorrections {
                ppp: lookup,
                ..RangeCorrections::disabled()
            },
            Some(5.0),
        ),
    )
    .expect("the declined satellite is left out before the cutoff");
    assert_eq!(solution.used_sats, ["G01", "G02", "G03", "G04", "G05"]);
    assert_eq!(solution.ssr_bias_exclusions.len(), 1);
    assert_eq!(solution.ssr_bias_exclusions[0].satellite_id, "G06");
    assert_eq!(solution.ssr_bias_exclusions[0].pass, 0);
    let error = norm3(sub3(solution.position_m, truth));
    assert!(error < 1.0e-3, "position error {error}");
}

/// A record boundary that the transmission time crosses while the solve iterates: at the
/// starting position G06's signal left after its new corrections' TOH, at the true
/// position just before it. The solve excludes G06 on the pass that crossed, starts
/// again from the state it reached, and converges on the other satellites.
#[test]
fn ssr_bias_flip_during_the_iteration_excludes_and_restarts() {
    let prns = [1, 2, 3, 4, 5, 6];
    let sats = prns.map(gps);
    let flip = gps(6);
    let t0 = ssr_test_t0();
    let toh = t0 + 30.0;
    let broadcast = ssr_spread_broadcast(&prns);
    let mut store = SsrCorrectionStore::new();
    let zero = Some(HasTestBiases::usable([0.0, 0.0], [0.0, 0.0]));
    has_test_ingest(
        &mut store,
        &has_test_message(&sats, 0, 1, 1, Some(HAS_VI_60_S), zero),
    );
    has_test_ingest(
        &mut store,
        &has_test_message(&[flip], 30, 2, 2, Some(HAS_VI_60_S), zero),
    );
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());
    let truth = receiver_under(&broadcast, &prns, toh);
    let (flip_position, _) =
        crate::spp::EphemerisSource::position_clock_at_j2000_s(&broadcast, flip, toh)
            .expect("broadcast state");
    // 3 km towards G06 shortens its signal flight by about 10 µs.
    let start = add3(
        truth,
        scale3(
            unit3(sub3(flip_position, truth)).expect("direction"),
            3_000.0,
        ),
    );
    let flight = |position: [f64; 3], t_rx: f64| {
        t_rx - crate::observables::transmit_epoch_j2000_s(
            &ephemeris,
            flip,
            position,
            t_rx,
            crate::observables::TransmitTimeOptions::default(),
            crate::observables::NOMINAL_SIGNAL_FLIGHT_TIME_S,
        )
        .expect("transmission time")
    };
    let guess = toh + 0.07;
    let t_rx = toh + 0.5 * (flight(start, guess) + flight(truth, guess));
    assert!(
        t_rx - flight(start, t_rx) >= toh,
        "at the start G06 left after the TOH"
    );
    assert!(
        t_rx - flight(truth, t_rx) < toh,
        "at the truth G06 left before the TOH"
    );

    let epoch = ssr_spread_epoch(&ephemeris, &prns, truth, t_rx, 12.5);
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        std::slice::from_ref(&epoch),
        start,
        &gps_l1_l2_options(),
    );
    assert_eq!(report.status, SsrPppAggregateStatus::AllApplied);
    let state = ssr_spread_state(&epoch, start);
    let solution = solve_float_epoch(
        &ephemeris,
        epoch,
        state,
        ssr_spread_config(
            RangeCorrections {
                ppp: lookup,
                ..RangeCorrections::disabled()
            },
            None,
        ),
    )
    .expect("the flip is excluded and the solve restarts");
    assert_eq!(solution.used_sats, ["G01", "G02", "G03", "G04", "G05"]);
    assert_eq!(solution.ssr_bias_exclusions.len(), 1);
    let exclusion = &solution.ssr_bias_exclusions[0];
    assert_eq!(exclusion.satellite_id, "G06");
    assert_eq!(exclusion.pass, 1);
    // The first Gauss-Newton step moves to the truth, where G06's biases no longer hold:
    // the exclusion comes from the iteration, not from the check at convergence.
    assert_eq!(exclusion.stage, SsrBiasExclusionStage::DuringIteration);
    assert!(matches!(
        exclusion.transmit_time_failure,
        Some(SsrTransmitTimeFailure::OrbitClockSolution { applied: None, .. })
    ));
    let error = norm3(sub3(solution.position_m, truth));
    assert!(error < 1.0e-3, "position error {error}");
}

/// The light-time iteration starts from the pseudorange, so the first ephemeris query is
/// near the transmission time. With a do-not-use indication starting at the reception
/// time, the signal left before it: the satellite is not excluded there, its HAS clock is
/// withdrawn, and a source with a broadcast fallback keeps it on the broadcast orbit, so
/// the bias is refused for want of an SSR orbit and clock.
#[test]
fn ssr_do_not_use_onset_at_reception_leaves_the_signal_on_broadcast() {
    let (_source, mut epochs, _state, _ambiguity_ids) = ssr_test_arc(1);
    let t0 = ssr_test_t0();
    let t_rx = t0 + 10.0;
    epochs[0].t_rx_j2000_s = t_rx;
    let sat = epochs[0].observations[0].sat;
    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3])),
        ),
    );
    let mut do_not_use = has_test_message(&[sat], 10, 1, 1, Some(HAS_VI_60_S), None);
    for record in &mut do_not_use.clock_full_set.as_mut().unwrap().records {
        record.correction_m = None;
        record.do_not_use = true;
    }
    has_test_ingest(&mut store, &do_not_use);
    let broadcast = ssr_test_broadcast();
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());
    assert!(
        ephemeris.corrected_state(sat, t_rx).is_none(),
        "excluded at reception"
    );

    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        ssr_test_receiver(),
        &gps_l1_l2_options(),
    );
    let row = &report.observation_reports[0];
    let t_tx = row
        .transmit_time_j2000_s
        .expect("the signal left before the onset");
    assert!(t_tx < t_rx);
    assert_eq!(
        ephemeris.corrected_state(sat, t_tx),
        crate::spp::EphemerisSource::position_clock_at_j2000_s(&broadcast, sat, t_tx),
        "the fallback keeps the satellite on broadcast"
    );
    assert_eq!(row.applied_orbit_clock_solution, None);
    assert_eq!(
        row.code_status,
        SsrIfCombinationStatus::OrbitClockSolutionUnavailable
    );
    assert_eq!(
        row.phase_status,
        SsrIfCombinationStatus::OrbitClockSolutionUnavailable
    );
    assert!(lookup.ssr_code_bias_m.is_empty());
}

/// Signal flight time of `sat` to `position` for reception at `t_rx`, from `source`.
fn flight_time(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    position: [f64; 3],
    t_rx: f64,
) -> f64 {
    t_rx - crate::observables::transmit_epoch_j2000_s(
        source,
        sat,
        position,
        t_rx,
        crate::observables::TransmitTimeOptions::default(),
        crate::observables::NOMINAL_SIGNAL_FLIGHT_TIME_S,
    )
    .expect("transmission time")
}

/// An exclusion made from a starting position several kilometres off, where G06's signal
/// left before its corrections' TOH, is checked again at the converged position. There the
/// signal left after the TOH, so G06 is admitted again and kept.
#[test]
fn ssr_bias_exclusion_from_an_unconverged_position_is_admitted_again() {
    let prns = [1, 2, 3, 4, 5, 6];
    let sats = prns.map(gps);
    let flip = gps(6);
    let t0 = ssr_test_t0();
    let toh = t0 + 30.0;
    let broadcast = ssr_spread_broadcast(&prns);
    let mut store = SsrCorrectionStore::new();
    let zero = Some(HasTestBiases::usable([0.0, 0.0], [0.0, 0.0]));
    has_test_ingest(
        &mut store,
        &has_test_message(&sats, 0, 1, 1, Some(HAS_VI_60_S), zero),
    );
    has_test_ingest(
        &mut store,
        &has_test_message(&[flip], 30, 2, 2, Some(HAS_VI_60_S), zero),
    );
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());
    let truth = receiver_under(&broadcast, &prns, toh);
    let (flip_position, _) =
        crate::spp::EphemerisSource::position_clock_at_j2000_s(&broadcast, flip, toh)
            .expect("broadcast state");
    // 3 km away from G06 lengthens its signal flight by about 10 µs.
    let seed = sub3(
        truth,
        scale3(
            unit3(sub3(flip_position, truth)).expect("direction"),
            3_000.0,
        ),
    );
    let guess = toh + 0.07;
    let t_rx = toh
        + 0.5
            * (flight_time(&ephemeris, flip, seed, guess)
                + flight_time(&ephemeris, flip, truth, guess));
    assert!(t_rx - flight_time(&ephemeris, flip, seed, t_rx) < toh);
    assert!(t_rx - flight_time(&ephemeris, flip, truth, t_rx) >= toh);

    // Two epochs; G06 is observed only in the first, so while it is excluded no kept
    // observation carries its ambiguity.
    let mut epochs = vec![
        ssr_spread_epoch(&ephemeris, &prns, truth, t_rx, 12.5),
        ssr_spread_epoch(&ephemeris, &prns[..5], truth, t_rx + 1.0, -8.25),
    ];
    epochs[1].epoch.second = 1.0;
    // The biases are resolved at the true position, where they hold.
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        truth,
        &gps_l1_l2_options(),
    );
    assert_eq!(report.status, SsrPppAggregateStatus::AllApplied);
    let solution = solve_float_epochs(
        &ephemeris,
        &epochs,
        FloatState {
            clocks_m: vec![0.0; 2],
            ..ssr_spread_state(&epochs[0], seed)
        },
        ssr_spread_config(
            RangeCorrections {
                ppp: lookup,
                ..RangeCorrections::disabled()
            },
            None,
        ),
    )
    .expect("the solve converges with G06 admitted again");
    assert_eq!(
        solution.ssr_bias_readmissions,
        [(0, "G06".to_string())],
        "G06 was excluded from the seed and admitted again"
    );
    assert!(solution.ssr_bias_exclusions.is_empty());
    assert_eq!(
        solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    let error = norm3(sub3(solution.position_m, truth));
    assert!(error < 1.0e-3, "position error {error}");
}

/// A flip inside the residual screen's re-solve: with a 50 km code outlier the unscreened
/// solution sits kilometres from the truth, where one satellite's first-epoch signal left
/// after its new corrections' TOH; the screen removes the outlier and re-solves towards
/// the truth, where that signal left before the TOH. The fixed-point solve excludes that
/// observation and starts again instead of refusing the solve.
#[test]
fn ssr_bias_flip_inside_the_residual_screen_excludes_and_restarts() {
    let prns = [1, 2, 3, 4, 5, 6];
    let sats = prns.map(gps);
    let t0 = ssr_test_t0();
    let toh = t0 + 30.0;
    let broadcast = ssr_spread_broadcast(&prns);
    let zero = Some(HasTestBiases::usable([0.0, 0.0], [0.0, 0.0]));
    let mut first_store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut first_store,
        &has_test_message(&sats, 0, 1, 1, Some(HAS_VI_60_S), zero),
    );
    let first = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &first_store)
        .with_fallback(broadcast_fallback());
    let truth = receiver_under(&broadcast, &prns, toh);
    let arc = |source: &dyn ObservableEphemerisSource, t_rx: f64| {
        let mut epochs = vec![
            ssr_spread_epoch(source, &prns, truth, t_rx, 12.5),
            ssr_spread_epoch(source, &prns, truth, t_rx + 1.0, -8.25),
        ];
        epochs[1].epoch.second = 1.0;
        epochs[0].observations[0].code_m += 50_000.0;
        epochs
    };

    // Where the unscreened solve lands with the outlier.
    let guess = toh + 0.07;
    let biased_epochs = arc(&first, guess);
    let biased = solve_float_epochs(
        &first,
        &biased_epochs,
        FloatState {
            clocks_m: vec![0.0; 2],
            ..ssr_spread_state(&biased_epochs[0], truth)
        },
        ssr_spread_config(RangeCorrections::disabled(), None),
    )
    .expect("unscreened solve with the outlier")
    .position_m;
    let flip = sats[1..]
        .iter()
        .copied()
        .max_by(|a, b| {
            let gain = |sat| {
                flight_time(&first, sat, truth, guess) - flight_time(&first, sat, biased, guess)
            };
            gain(*a).total_cmp(&gain(*b))
        })
        .expect("a satellite");
    assert!(
        flight_time(&first, flip, truth, guess) - flight_time(&first, flip, biased, guess) > 3.0e-6,
        "the outlier moves the solution kilometres towards {flip}"
    );

    let mut store = first_store.clone();
    has_test_ingest(
        &mut store,
        &has_test_message(&[flip], 30, 2, 2, Some(HAS_VI_60_S), zero),
    );
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());
    let t_rx = toh
        + 0.5
            * (flight_time(&ephemeris, flip, biased, guess)
                + flight_time(&ephemeris, flip, truth, guess));
    let epochs = arc(&ephemeris, t_rx);
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        biased,
        &gps_l1_l2_options(),
    );
    assert_eq!(report.status, SsrPppAggregateStatus::AllApplied);
    let mut config = ssr_spread_config(
        RangeCorrections {
            ppp: lookup,
            ..RangeCorrections::disabled()
        },
        None,
    );
    config.residual_screen = true;
    let solution = solve_float_epochs(
        &ephemeris,
        &epochs,
        FloatState {
            clocks_m: vec![0.0; 2],
            ..ssr_spread_state(&epochs[0], biased)
        },
        config,
    )
    .expect("the flip in the screen is excluded and the solve restarts");
    assert_eq!(solution.ssr_bias_exclusions.len(), 1);
    let exclusion = &solution.ssr_bias_exclusions[0];
    assert_eq!(exclusion.satellite_id, flip.to_string());
    assert_eq!(exclusion.epoch_index, 0);
    assert_eq!(exclusion.stage, SsrBiasExclusionStage::DuringIteration);
    let error = norm3(sub3(solution.position_m, truth));
    assert!(error < 1.0e-3, "position error {error}");
}

/// A flip inside the fixed re-solve: the float solution handed to it sits 3 km towards
/// G06, where G06's signal left after its new corrections' TOH, and the fixed re-solve
/// moves to the truth, where it left before. The fixed solve excludes G06, solves the
/// float arc again from the float state, and fixes again.
#[test]
fn ssr_bias_flip_inside_the_fixed_solve_resolves_the_float_arc_again() {
    let prns = [1, 2, 3, 4, 5, 6];
    let sats = prns.map(gps);
    let flip = gps(6);
    let t0 = ssr_test_t0();
    let toh = t0 + 30.0;
    let broadcast = ssr_spread_broadcast(&prns);
    let mut store = SsrCorrectionStore::new();
    let zero = Some(HasTestBiases::usable([0.0, 0.0], [0.0, 0.0]));
    has_test_ingest(
        &mut store,
        &has_test_message(&sats, 0, 1, 1, Some(HAS_VI_60_S), zero),
    );
    has_test_ingest(
        &mut store,
        &has_test_message(&[flip], 30, 2, 2, Some(HAS_VI_60_S), zero),
    );
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store)
        .with_fallback(broadcast_fallback());
    let truth = receiver_under(&broadcast, &prns, toh);
    let (flip_position, _) =
        crate::spp::EphemerisSource::position_clock_at_j2000_s(&broadcast, flip, toh)
            .expect("broadcast state");
    let float_position = add3(
        truth,
        scale3(
            unit3(sub3(flip_position, truth)).expect("direction"),
            3_000.0,
        ),
    );
    let guess = toh + 0.07;
    let t_rx = toh
        + 0.5
            * (flight_time(&ephemeris, flip, float_position, guess)
                + flight_time(&ephemeris, flip, truth, guess));
    let wavelength = C_M_S / F_L1_HZ;
    let mut epoch = ssr_spread_epoch(&ephemeris, &prns, truth, t_rx, 12.5);
    let mut ambiguities_m = BTreeMap::new();
    let mut wavelengths_m = BTreeMap::new();
    let mut offsets_m = BTreeMap::new();
    for (index, obs) in epoch.observations.iter_mut().enumerate() {
        let ambiguity_m = (40_000 + 17 * index as i64) as f64 * wavelength;
        obs.phase_m = obs.code_m + ambiguity_m;
        ambiguities_m.insert(obs.ambiguity_id.clone(), ambiguity_m);
        wavelengths_m.insert(obs.ambiguity_id.clone(), wavelength);
        offsets_m.insert(obs.ambiguity_id.clone(), 0.0);
    }
    let epochs = vec![epoch];
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs,
        float_position,
        &gps_l1_l2_options(),
    );
    assert_eq!(report.status, SsrPppAggregateStatus::AllApplied);
    let corrections = RangeCorrections {
        ppp: lookup,
        ..RangeCorrections::disabled()
    };
    let float_solution = FloatSolution {
        position_m: float_position,
        position_covariance: unit_position_covariance(),
        formal_position_covariance: unit_position_covariance(),
        posterior_variance_factor: 1.0,
        position_covariance_scale_factor: 1.0,
        temporal_position_covariance: unit_position_covariance(),
        temporal_position_covariance_scale_factor: 1.0,
        temporal_correlation: unit_temporal_correlation(),
        epoch_clocks_m: vec![12.5],
        ambiguities_m,
        residual_ionosphere_m: BTreeMap::new(),
        ztd_residual_m: None,
        tropo_gradient_north_m: None,
        tropo_gradient_east_m: None,
        tropo_gradient_covariance_m2: None,
        formal_tropo_gradient_covariance_m2: None,
        residuals_m: Vec::new(),
        used_sats: sats.iter().map(|sat| sat.to_string()).collect(),
        iterations: 1,
        converged: true,
        status: FloatStatus::StateTolerance,
        code_rms_m: 0.0,
        phase_rms_m: 0.0,
        weighted_rms_m: 0.0,
        ssr_bias_exclusions: Vec::new(),
        solved_epoch_indices: vec![0],
        ssr_bias_readmissions: Vec::new(),
        ssr_bias_last_pass: 0,
        residual_screen: false,
        solve_options: FloatSolveOptions::default(),
        residual_screen_removals: Vec::new(),
    };
    let float_config = ssr_spread_config(corrections.clone(), None);
    let solution = solve_fixed_from_float(
        &ephemeris,
        &epochs,
        float_solution,
        FixedSolveConfig {
            weights: float_config.weights,
            tropo: float_config.tropo,
            corrections,
            opts: float_config.opts,
            elevation_cutoff_deg: None,
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m,
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .expect("the flip is excluded and the arc fixed again");
    assert_eq!(solution.ssr_bias_exclusions.len(), 1);
    let exclusion = &solution.ssr_bias_exclusions[0];
    assert_eq!(exclusion.satellite_id, "G06");
    // Found in the fixed re-solve, on the pass after the float solve's.
    assert_eq!(exclusion.stage, SsrBiasExclusionStage::FixedResolve);
    assert_eq!(exclusion.pass, 1);
    assert!(!solution.fixed_ambiguities_cycles.contains_key("G06"));
    assert_eq!(
        solution.float_solution.ssr_bias_exclusions, solution.ssr_bias_exclusions,
        "the float solution is the arc solved again without G06"
    );
    let error = norm3(sub3(solution.position_m, truth));
    assert!(error < 1.0e-3, "position error {error}");
}

/// The fixed solve prepares its arc with no ambiguity seeds, and the arc it prepares can
/// observe an ambiguity the float solution never solved, such as a satellite the float
/// solve's elevation cutoff removed and the fixed configuration's lower cutoff, or none,
/// keeps. An ambiguity the state lacks starts from phase minus code of its first
/// observation, as `initial_ambiguities` seeds it, instead of failing the rows.
#[test]
fn prepared_arc_seeds_a_missing_ambiguity_from_phase_minus_code() {
    let (source, epochs, mut state, _) = ppp_row_trace_arc();
    let missing = epochs[0].observations[1].ambiguity_id.clone();
    state.ambiguities_m.remove(&missing);
    let corrections = RangeCorrections::disabled();
    let arc = super::float::ArcSettings {
        source: &source,
        weights: ppp_row_trace_weights(),
        tropo: TroposphereOptions::disabled(),
        corrections: &corrections,
        normal: crate::estimation::recipe::NormalRecipe::PppDenseLastTie,
        estimate_residual_ionosphere: false,
        elevation_cutoff_deg: None,
    };
    let (prepared, exclusions) = super::float::prepare_arc(
        &arc,
        &epochs,
        &super::float::LeftOut {
            excluded: &[],
            screened: &[],
            deferred: &[],
            seed_ambiguities: &BTreeMap::new(),
        },
        &state,
        0,
    )
    .expect("prepared arc");
    assert!(exclusions.is_empty());
    let first = &epochs[0].observations[1];
    assert_eq!(
        prepared.state.ambiguities_m[&missing].to_bits(),
        (first.phase_m - first.code_m).to_bits()
    );
    // Ambiguities the state carries are kept as they are.
    for (id, value) in &state.ambiguities_m {
        assert_eq!(prepared.state.ambiguities_m[id].to_bits(), value.to_bits());
    }
}

/// A Galileo HAS IOD set change with an unchanged PDI keeps the phase arc (HAS SIS ICD
/// 5.2.6.1), so the ambiguity carried across it is not reset and the new set's biases apply.
///
/// The first message (mask 1, IOD set 1, PDI 0) is applied at t0 without a token, and its
/// tokens are passed back at t0 + 31 s, after a message with IOD set 2 and PDI 0 replaced
/// the orbit, clock and biases. The biases and the orbit and clock move to the new solution
/// together, so they stay consistent.
#[test]
fn test_ppp_ssr_biases_has_iod_set_change_keeps_ambiguity() {
    let (_source, epochs, _state, _ambiguity_ids) = ssr_test_arc(1);
    let sat = epochs[0].observations[0].sat;
    let ambiguity_id = epochs[0].observations[0].ambiguity_id.clone();
    let broadcast = ssr_test_broadcast();

    let mut store = SsrCorrectionStore::new();
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            0,
            1,
            1,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable([1.24, -0.76], [0.2, -0.3])),
        ),
    );
    let options = gps_l1_l2_options();
    let (token0, token9) = {
        let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
        let (_, first) = PppCorrectionLookup::default().with_ssr_biases(
            &ephemeris,
            &epochs,
            ssr_test_receiver(),
            &options,
        );
        assert_eq!(first.status, SsrPppAggregateStatus::AllApplied);
        let first_row = &first.observation_reports[0];
        let phase1 = &first_row.phase1_report.as_ref().unwrap().query_result;
        let phase2 = &first_row.phase2_report.as_ref().unwrap().query_result;
        assert_eq!(
            phase1.discontinuity_details,
            Some(crate::ssr::SsrDiscontinuityDetails::InitialTokenEstablished)
        );
        (
            phase1.continuity_token.unwrap(),
            phase2.continuity_token.unwrap(),
        )
    };

    let code_m = [0.5, -0.3];
    let phase_cycles = [0.1, -0.4];
    has_test_ingest(
        &mut store,
        &has_test_message(
            &[sat],
            30,
            1,
            2,
            Some(HAS_VI_60_S),
            Some(HasTestBiases::usable(code_m, phase_cycles)),
        ),
    );
    let ephemeris = crate::ssr::SsrCorrectedEphemeris::new(&broadcast, &store);
    let mut epochs_30 = epochs.clone();
    epochs_30[0].t_rx_j2000_s = ssr_test_t0() + 31.0;
    let options_ack = options
        .clone()
        .with_phase_continuity_token(ambiguity_id.clone(), 0, token0)
        .with_phase_continuity_token(ambiguity_id.clone(), 9, token9);
    let (lookup, report) = PppCorrectionLookup::default().with_ssr_biases(
        &ephemeris,
        &epochs_30,
        ssr_test_receiver(),
        &options_ack,
    );

    assert_eq!(report.status, SsrPppAggregateStatus::AllApplied);
    assert_eq!(report.phase_discontinuity_resets_needed, 0);
    let row = &report.observation_reports[0];
    assert_eq!(row.applied_orbit_clock_solution, Some(has_solution(1, 2)));
    assert_eq!(row.code_status, SsrIfCombinationStatus::Applied);
    assert_eq!(row.phase_status, SsrIfCombinationStatus::Applied);
    for (signal_report, token) in [
        (row.phase1_report.as_ref().unwrap(), token0),
        (row.phase2_report.as_ref().unwrap(), token9),
    ] {
        let query = &signal_report.query_result;
        assert_eq!(query.status, crate::ssr::SsrBiasStatus::Available);
        assert_eq!(query.solution, Some(has_solution(1, 2)));
        assert_eq!(
            query.discontinuity_details,
            Some(crate::ssr::SsrDiscontinuityDetails::Continuous)
        );
        assert_eq!(
            query.continuity_token.unwrap().generation(),
            token.generation()
        );
    }

    let key = (sat, 0, ambiguity_id);
    let expected_phase_if = ionosphere_free(
        phase_cycles[0] * (C_M_S / F_L1_HZ),
        phase_cycles[1] * (C_M_S / F_L2_HZ),
    );
    let expected_code_if = ionosphere_free(code_m[0], code_m[1]);
    let stored_phase_if = lookup.phase_bias_m[&key];
    let stored_code_if = lookup.ssr_code_bias_m[&key];
    assert!(
        (stored_phase_if - expected_phase_if).abs() < 1.0e-9,
        "stored {stored_phase_if}, expected {expected_phase_if}"
    );
    assert!(
        (stored_code_if + expected_code_if).abs() < 1.0e-9,
        "stored {stored_code_if}, expected {}",
        -expected_code_if
    );
}

/// A float solve leaves out every observation whose required SSR bias is absent and
/// solves the rest as if those observations had not been supplied, listing each one in
/// `ssr_bias_exclusions`. The lookup here is filled directly with zero biases for every
/// satellite but G02, so it carries no application report.
#[test]
fn float_solve_excludes_observations_without_required_ssr_bias() {
    let (source, epochs, initial, _) = ppp_elevation_cutoff_arc();
    let mut lookup = PppCorrectionLookup {
        ssr_code_bias_enabled: true,
        phase_bias_enabled: true,
        ..Default::default()
    };
    for (epoch_index, epoch) in epochs.iter().enumerate() {
        for obs in &epoch.observations {
            if obs.satellite_id != "G02" {
                let key = (obs.sat, epoch_index, obs.ambiguity_id.clone());
                lookup.ssr_code_bias_m.insert(key.clone(), 0.0);
                lookup.phase_bias_m.insert(key, 0.0);
            }
        }
    }
    let mut config = ppp_cutoff_config(None);
    config.corrections.ppp = lookup;
    let solution = solve_float_epochs(&source, &epochs, initial, config).unwrap();

    assert_eq!(solution.used_sats, ["G01", "G03", "G04", "G05", "G06"]);
    assert_eq!(solution.residuals_m.len(), 5 * epochs.len());
    assert!(solution
        .residuals_m
        .iter()
        .all(|residual| residual.satellite_id != "G02"));
    let err = norm3(sub3(solution.position_m, [6_378_137.0, 0.0, 0.0]));
    assert!(err < 1.0e-3, "position error {err}");
    assert_eq!(solution.ssr_bias_exclusions.len(), epochs.len());
    for (epoch_index, exclusion) in solution.ssr_bias_exclusions.iter().enumerate() {
        assert_eq!(exclusion.epoch_index, epoch_index);
        assert_eq!(exclusion.satellite_id, "G02");
        assert_eq!(exclusion.ambiguity_id, "G02");
        assert!(exclusion.code_bias_missing);
        assert!(exclusion.phase_bias_missing);
        assert_eq!(exclusion.application, None);
    }
}

/// A residual screen that removes the only observation of the first epoch keeps every
/// later epoch on its own per-epoch corrections.
///
/// Each epoch carries its own SSR code and phase biases, and the observations carry the
/// same biases, so only the right epoch's biases fit. The first epoch holds a single G01
/// observation with a 100 m code error, which the screen removes, emptying that epoch. The
/// screened solution matches, bit for bit, a solve that was never given the first epoch,
/// with its lookup keyed from the second epoch.
#[test]
fn residual_screen_emptying_an_epoch_keeps_later_epochs_on_their_corrections() {
    let (source, mut epochs, initial, _) = ppp_elevation_cutoff_arc();
    let code_bias =
        |epoch_idx: usize, prn: u8| 0.5 + 0.7 * epoch_idx as f64 + 0.01 * f64::from(prn);
    let phase_bias =
        |epoch_idx: usize, prn: u8| -0.03 * (epoch_idx as f64 + 1.0) + 0.002 * f64::from(prn);
    epochs[0].observations.truncate(1);
    epochs[0].observations[0].code_m += 100.0;
    let mut lookup = PppCorrectionLookup {
        ssr_code_bias_enabled: true,
        phase_bias_enabled: true,
        ..Default::default()
    };
    for (epoch_idx, epoch) in epochs.iter_mut().enumerate() {
        for obs in &mut epoch.observations {
            let code_m = code_bias(epoch_idx, obs.sat.prn);
            let phase_m = phase_bias(epoch_idx, obs.sat.prn);
            obs.code_m += code_m;
            obs.phase_m -= phase_m;
            let key = (obs.sat, epoch_idx, obs.ambiguity_id.clone());
            lookup.ssr_code_bias_m.insert(key.clone(), code_m);
            lookup.phase_bias_m.insert(key, phase_m);
        }
    }
    let state = FloatState {
        ambiguities_m: initial_ambiguities(&epochs),
        ..initial.clone()
    };
    let mut config = ppp_cutoff_config(None);
    config.residual_screen = true;
    config.corrections.ppp = lookup.clone();
    let screened = solve_float_epochs(&source, &epochs, state, config).unwrap();

    // The same arc without the first epoch, its lookup keyed from the second epoch.
    let later_epochs = epochs[1..].to_vec();
    let mut later_lookup = PppCorrectionLookup {
        ssr_code_bias_enabled: true,
        phase_bias_enabled: true,
        ..Default::default()
    };
    for ((sat, epoch_idx, ambiguity_id), value) in &lookup.ssr_code_bias_m {
        if *epoch_idx > 0 {
            later_lookup
                .ssr_code_bias_m
                .insert((*sat, epoch_idx - 1, ambiguity_id.clone()), *value);
        }
    }
    for ((sat, epoch_idx, ambiguity_id), value) in &lookup.phase_bias_m {
        if *epoch_idx > 0 {
            later_lookup
                .phase_bias_m
                .insert((*sat, epoch_idx - 1, ambiguity_id.clone()), *value);
        }
    }
    let later_state = FloatState {
        position_m: initial.position_m,
        clocks_m: vec![initial.clocks_m[0]; later_epochs.len()],
        ambiguities_m: initial_ambiguities(&later_epochs),
        ztd_m: initial.ztd_m,
        tropo_gradient_north_m: initial.tropo_gradient_north_m,
        tropo_gradient_east_m: initial.tropo_gradient_east_m,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let mut later_config = ppp_cutoff_config(None);
    later_config.corrections.ppp = later_lookup;
    let reference = solve_float_epochs(&source, &later_epochs, later_state, later_config).unwrap();

    assert_eq!(screened.epoch_clocks_m.len(), later_epochs.len());
    assert_eq!(screened.solved_epoch_indices, [1, 2]);
    assert_eq!(reference.solved_epoch_indices, [0, 1]);
    assert_eq!(screened.used_sats, reference.used_sats);
    assert_eq!(
        ppp_float_solution_bits(&screened),
        ppp_float_solution_bits(&reference)
    );
    // Residuals carry input epoch indices, one above the reference's.
    assert_eq!(screened.residuals_m.len(), reference.residuals_m.len());
    for (screened_row, reference_row) in screened.residuals_m.iter().zip(&reference.residuals_m) {
        assert_eq!(screened_row.epoch_index, reference_row.epoch_index + 1);
        assert_eq!(screened_row.satellite_id, reference_row.satellite_id);
    }
    assert!(screened.weighted_rms_m < 1.0e-6);
    assert!(screened.ssr_bias_exclusions.is_empty());
}

/// An epoch whose every observation lacks its required SSR bias is left out of a static
/// solve instead of leaving its receiver clock without rows. The arc solves exactly as it
/// does without that epoch, and the solution names the input epochs it solved.
#[test]
fn float_solve_drops_an_epoch_emptied_by_ssr_bias_exclusion() {
    let (source, epochs, initial, _) = ppp_elevation_cutoff_arc();
    let empty_epoch = 1;
    let mut lookup = PppCorrectionLookup {
        ssr_code_bias_enabled: true,
        phase_bias_enabled: true,
        ..Default::default()
    };
    for (epoch_idx, epoch) in epochs.iter().enumerate() {
        for obs in &epoch.observations {
            if epoch_idx != empty_epoch {
                let key = (obs.sat, epoch_idx, obs.ambiguity_id.clone());
                lookup.ssr_code_bias_m.insert(key.clone(), 0.0);
                lookup.phase_bias_m.insert(key, 0.0);
            }
        }
    }
    let mut config = ppp_cutoff_config(None);
    config.corrections.ppp = lookup;
    let solution = solve_float_epochs(&source, &epochs, initial.clone(), config)
        .expect("an emptied epoch is left out, not solved");

    let kept = [0, 2];
    let kept_epochs: Vec<FloatEpoch> = kept.iter().map(|&i| epochs[i].clone()).collect();
    let mut kept_lookup = PppCorrectionLookup {
        ssr_code_bias_enabled: true,
        phase_bias_enabled: true,
        ..Default::default()
    };
    for (epoch_idx, epoch) in kept_epochs.iter().enumerate() {
        for obs in &epoch.observations {
            let key = (obs.sat, epoch_idx, obs.ambiguity_id.clone());
            kept_lookup.ssr_code_bias_m.insert(key.clone(), 0.0);
            kept_lookup.phase_bias_m.insert(key, 0.0);
        }
    }
    let kept_state = FloatState {
        clocks_m: kept.iter().map(|&i| initial.clocks_m[i]).collect(),
        ..initial
    };
    let mut kept_config = ppp_cutoff_config(None);
    kept_config.corrections.ppp = kept_lookup;
    let reference = solve_float_epochs(&source, &kept_epochs, kept_state, kept_config).unwrap();

    assert_eq!(solution.solved_epoch_indices, kept);
    assert_eq!(reference.solved_epoch_indices, [0, 1]);
    assert_eq!(
        ppp_float_solution_bits(&solution),
        ppp_float_solution_bits(&reference)
    );
    for (row, reference_row) in solution.residuals_m.iter().zip(&reference.residuals_m) {
        assert_eq!(row.epoch_index, kept[reference_row.epoch_index]);
    }
    assert_eq!(
        solution.ssr_bias_exclusions.len(),
        epochs[empty_epoch].observations.len()
    );
    assert!(solution
        .ssr_bias_exclusions
        .iter()
        .all(|exclusion| exclusion.epoch_index == empty_epoch));
}

/// An observation the float solve leaves out for a missing SSR bias stays out of the fixed
/// re-solve, which would otherwise refuse the missing bias.
#[test]
fn fixed_solve_keeps_float_ssr_bias_exclusions() {
    let (source, epochs, initial, fixed_cycles, wavelength, truth, _clocks) = fixed_synthetic_arc();
    let mut lookup = PppCorrectionLookup {
        ssr_code_bias_enabled: true,
        phase_bias_enabled: true,
        ..Default::default()
    };
    for (epoch_idx, epoch) in epochs.iter().enumerate() {
        for obs in &epoch.observations {
            if epoch_idx == 1 && obs.satellite_id == "G06" {
                continue;
            }
            let key = (obs.sat, epoch_idx, obs.ambiguity_id.clone());
            lookup.ssr_code_bias_m.insert(key.clone(), 0.0);
            lookup.phase_bias_m.insert(key, 0.0);
        }
    }
    let corrections = RangeCorrections {
        ppp: lookup,
        ..RangeCorrections::disabled()
    };
    let weights = MeasurementWeights {
        code: 1.0,
        phase: 100.0,
        elevation_weighting: false,
    };
    let opts = FloatSolveOptions {
        max_iterations: 8,
        position_tolerance_m: 1.0e-4,
        clock_tolerance_m: 1.0e-4,
        ambiguity_tolerance_m: 1.0e-4,
        ztd_tolerance_m: 1.0e-4,
    };
    let float_solution = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights,
            tropo: TroposphereOptions::disabled(),
            corrections: corrections.clone(),
            opts,
            elevation_cutoff_deg: None,
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .unwrap();
    assert_eq!(float_solution.ssr_bias_exclusions.len(), 1);
    assert_eq!(float_solution.residuals_m.len(), 17);
    let float_exclusions = float_solution.ssr_bias_exclusions.clone();

    let solution = solve_fixed_from_float(
        &source,
        &epochs,
        float_solution,
        FixedSolveConfig {
            weights,
            tropo: TroposphereOptions::disabled(),
            corrections,
            opts,
            elevation_cutoff_deg: None,
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m: fixed_cycles
                    .keys()
                    .map(|sat| (sat.clone(), wavelength))
                    .collect(),
                offsets_m: fixed_cycles.keys().map(|sat| (sat.clone(), 0.0)).collect(),
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .expect("the fixed re-solve leaves the excluded observation out");

    assert_eq!(solution.ssr_bias_exclusions, float_exclusions);
    let exclusion = &solution.ssr_bias_exclusions[0];
    assert_eq!(exclusion.epoch_index, 1);
    assert_eq!(exclusion.satellite_id, "G06");
    assert!(exclusion.code_bias_missing);
    assert!(exclusion.phase_bias_missing);
    assert_eq!(solution.residuals_m.len(), 17);
    assert!(!solution
        .residuals_m
        .iter()
        .any(|residual| residual.epoch_index == 1 && residual.satellite_id == "G06"));
    assert_eq!(solution.fixed_ambiguities_cycles, fixed_cycles);
    assert!(norm3(sub3(solution.position_m, truth)) < 1.0e-3);
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
        correction_epoch_indices: None,
        ssr_bias_pass: 0,
        ssr_bias_stage: SsrBiasExclusionStage::BeforeSolve,
        ssr_bias_deferred: &[],
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
        correction_epoch_indices: None,
        ssr_bias_pass: 0,
        ssr_bias_stage: SsrBiasExclusionStage::BeforeSolve,
        ssr_bias_deferred: &[],
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

/// Six satellites over three epochs with integer ambiguities; returns the source, the
/// epochs, a seed state, the integer cycles, the wavelength, the truth and the clocks.
#[allow(clippy::type_complexity)]
fn fixed_synthetic_arc() -> (
    FakeSource,
    Vec<FloatEpoch>,
    FloatState,
    BTreeMap<String, i64>,
    f64,
    [f64; 3],
    [f64; 3],
) {
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
    (
        source,
        epochs,
        initial,
        fixed_cycles,
        wavelength,
        truth,
        clocks,
    )
}

#[test]
fn static_fixed_solver_recovers_synthetic_arc() {
    let (source, epochs, initial, fixed_cycles, wavelength, truth, clocks) = fixed_synthetic_arc();
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

/// A fixed configuration without the float solve's elevation cutoff keeps a satellite the
/// float solve cut, so the fixed arc observes an ambiguity the float solution never solved.
/// The fixed solve solves the float arc again over the fixed arc and fixes every ambiguity.
#[test]
fn fixed_solve_resolves_the_float_arc_for_a_satellite_the_float_cutoff_removed() {
    let (source, epochs, initial, fixed_cycles, wavelength, truth, _clocks) = fixed_synthetic_arc();
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
    // G06 is about 36 degrees below the receiver's horizon and the next lowest, G05, about
    // 16, so a -20 degree cutoff removes G06 alone.
    let float_solution = solve_float_epochs(
        &source,
        &epochs,
        initial,
        FloatSolveConfig {
            weights,
            tropo,
            corrections: corrections.clone(),
            opts,
            elevation_cutoff_deg: Some(-20.0),
            residual_screen: false,
            estimate_residual_ionosphere: false,
        },
    )
    .expect("float solve with a cutoff");
    assert_eq!(
        float_solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05"]
    );
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
                wavelengths_m: fixed_cycles
                    .keys()
                    .map(|sat| (sat.clone(), wavelength))
                    .collect(),
                offsets_m: fixed_cycles.keys().map(|sat| (sat.clone(), 0.0)).collect(),
                ratio_threshold: 3.0,
            },
            estimate_residual_ionosphere: false,
        },
    )
    .expect("fixed solve without a cutoff");
    assert_eq!(
        solution.float_solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"],
        "the float arc is solved again over the fixed arc"
    );
    assert_eq!(solution.fixed_ambiguities_cycles, fixed_cycles);
    assert_eq!(solution.integer.integer_status, IntegerStatus::Fixed);
    assert!(norm3(sub3(solution.position_m, truth)) < 1.0e-3);
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
        ssr_bias_exclusions: Vec::new(),
        solved_epoch_indices: (0..epochs.len()).collect(),
        ssr_bias_readmissions: Vec::new(),
        ssr_bias_last_pass: 0,
        residual_screen: false,
        solve_options: FloatSolveOptions::default(),
        residual_screen_removals: Vec::new(),
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
        ssr_bias_exclusions: Vec::new(),
        solved_epoch_indices: (0..epochs.len()).collect(),
        ssr_bias_readmissions: Vec::new(),
        ssr_bias_last_pass: 0,
        residual_screen: false,
        solve_options: FloatSolveOptions::default(),
        residual_screen_removals: Vec::new(),
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
        ssr_bias_exclusions: Vec::new(),
        solved_epoch_indices: (0..epochs.len()).collect(),
        ssr_bias_readmissions: Vec::new(),
        ssr_bias_last_pass: 0,
        residual_screen: false,
        solve_options: FloatSolveOptions::default(),
        residual_screen_removals: Vec::new(),
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

/// The fixed solve re-solves the float arc with the float solution's own options and
/// leaves out the observations it names, so it refuses options the float solve would
/// refuse and a removal or readmission that names no observation of the input epochs.
#[test]
fn static_fixed_solver_rejects_float_solution_options_and_keys_it_cannot_apply() {
    let (source, epochs, state, _ambiguity_ids) = ppp_row_trace_arc();
    let used_sats = state.ambiguities_m.keys().cloned().collect::<Vec<_>>();
    let float_solution = FloatSolution {
        position_m: state.position_m,
        position_covariance: unit_position_covariance(),
        formal_position_covariance: unit_position_covariance(),
        posterior_variance_factor: 1.0,
        position_covariance_scale_factor: 1.0,
        temporal_position_covariance: unit_position_covariance(),
        temporal_position_covariance_scale_factor: 1.0,
        temporal_correlation: unit_temporal_correlation(),
        epoch_clocks_m: state.clocks_m.clone(),
        ambiguities_m: state.ambiguities_m.clone(),
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
        ssr_bias_exclusions: Vec::new(),
        solved_epoch_indices: (0..epochs.len()).collect(),
        ssr_bias_readmissions: Vec::new(),
        ssr_bias_last_pass: 0,
        residual_screen: false,
        solve_options: FloatSolveOptions::default(),
        residual_screen_removals: Vec::new(),
    };
    let config = || FixedSolveConfig {
        weights: ppp_row_trace_weights(),
        tropo: TroposphereOptions::disabled(),
        corrections: RangeCorrections::disabled(),
        opts: FloatSolveOptions::default(),
        elevation_cutoff_deg: None,
        ambiguity: FixedAmbiguityOptions {
            wavelengths_m: used_sats.iter().map(|sat| (sat.clone(), 0.190)).collect(),
            offsets_m: used_sats.iter().map(|sat| (sat.clone(), 0.0)).collect(),
            ratio_threshold: 3.0,
        },
        estimate_residual_ionosphere: false,
    };
    let refuse = |solution: FloatSolution| {
        solve_fixed_from_float(&source, &epochs, solution, config())
            .expect_err("the fixed solve must refuse the float solution")
    };

    let mut bad_options = float_solution.clone();
    bad_options.solve_options.position_tolerance_m = f64::NAN;
    assert_eq!(
        refuse(bad_options),
        FixedSolveError::Float(FloatSolveError::InvalidSolveOption {
            field: "position_tolerance_m",
            reason: "must be finite",
        })
    );

    let removal_error = FixedSolveError::Float(FloatSolveError::InvalidInput {
        field: "ppp float_solution residual_screen_removals",
        reason: "must name observations of the input epochs",
    });
    let mut removal_past_the_arc = float_solution.clone();
    removal_past_the_arc
        .residual_screen_removals
        .push((epochs.len(), used_sats[0].clone()));
    assert_eq!(refuse(removal_past_the_arc), removal_error);
    let mut removal_of_no_observation = float_solution.clone();
    removal_of_no_observation
        .residual_screen_removals
        .push((0, "G99".to_string()));
    assert_eq!(refuse(removal_of_no_observation), removal_error);

    let readmission_error = FixedSolveError::Float(FloatSolveError::InvalidInput {
        field: "ppp float_solution ssr_bias_readmissions",
        reason: "must name observations of the input epochs",
    });
    let mut readmission_past_the_arc = float_solution.clone();
    readmission_past_the_arc
        .ssr_bias_readmissions
        .push((epochs.len(), used_sats[0].clone()));
    assert_eq!(refuse(readmission_past_the_arc), readmission_error);
    let mut readmission_of_no_observation = float_solution.clone();
    readmission_of_no_observation
        .ssr_bias_readmissions
        .push((1, "G99".to_string()));
    assert_eq!(refuse(readmission_of_no_observation), readmission_error);

    // A pass number past u32::MAX exists only where usize is wider than 32 bits.
    if let Ok(pass) = usize::try_from(u64::from(u32::MAX) + 1) {
        let mut pass_past_u32 = float_solution;
        pass_past_u32.ssr_bias_last_pass = pass;
        assert_eq!(
            refuse(pass_past_u32),
            FixedSolveError::Float(FloatSolveError::InvalidInput {
                field: "ppp float_solution ssr_bias_last_pass",
                reason: "exceeds u32::MAX",
            })
        );
    }
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
        correction_epoch_indices: None,
        ssr_bias_pass: 0,
        ssr_bias_stage: SsrBiasExclusionStage::BeforeSolve,
        ssr_bias_deferred: &[],
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
        correction_epoch_indices: None,
        ssr_bias_pass: 0,
        ssr_bias_stage: SsrBiasExclusionStage::BeforeSolve,
        ssr_bias_deferred: &[],
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
