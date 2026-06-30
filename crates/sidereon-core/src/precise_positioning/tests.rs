use super::*;
use crate::ambiguity::AmbiguityId;
use crate::astro::math::vec3::{norm3, sub3};
use crate::carrier_phase::{CycleSlipOptions, SlipReason};
use crate::constants::C_M_S;
use crate::observables::{predict, ObservableState, ObservablesError};
use crate::ppp_corrections::CivilDateTime;
use crate::{GnssSatelliteId, GnssSystem};

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
        }],
    }
}

fn single_obs_clock_state(epoch: &FloatEpoch) -> FloatState {
    FloatState {
        position_m: [3_512_900.0, 780_500.0, 5_248_700.0],
        clocks_m: vec![0.0],
        ambiguities_m: initial_ambiguities(std::slice::from_ref(epoch)),
        ztd_m: 0.0,
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
        residual_screen: false,
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

#[test]
fn float_solution_output_validation_rejects_nonfinite_values() {
    let solution = FloatSolution {
        position_m: [0.0, f64::NAN, 0.0],
        epoch_clocks_m: vec![0.0],
        ambiguities_m: BTreeMap::new(),
        ztd_residual_m: None,
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
            jd_fraction: 0.5 + epoch_idx as f64 * 900.0 / 86_400.0,
            t_rx_j2000_s: epoch_idx as f64 * 900.0,
            observations,
        });
    }
    let initial = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0; epochs.len()],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
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
            residual_screen: false,
        },
    )
    .unwrap();
    assert_eq!(
        solution.position_m.map(f64::to_bits),
        [0x414acd21ffffffff, 0x4127d1a800000013, 0x415405aeffffffff,]
    );
    assert_eq!(
        solution
            .epoch_clocks_m
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        [0x4028fffffffd0d43, 0xc02080000002f2b7, 0x400ffffffff4351c,]
    );
    assert_eq!(
        solution
            .ambiguities_m
            .iter()
            .map(|(sat, value)| (sat.as_str(), value.to_bits()))
            .collect::<Vec<_>>()
            .as_slice(),
        &[
            ("G01", 0x3fd00000013c69a8),
            ("G02", 0x3fd666666672728d),
            ("G03", 0x3fdcccccce092ebf),
            ("G04", 0x3fe1999997edbf4e),
            ("G05", 0x3fe4cccccaacf7d5),
            ("G06", 0x3fe7ffffffa548b4),
        ]
    );
    let residual_bits = solution
        .residuals_m
        .iter()
        .map(|row| {
            (
                row.epoch_index,
                row.satellite_id.as_str(),
                row.code_m.to_bits(),
                row.phase_m.to_bits(),
                row.code_weight.to_bits(),
                row.phase_weight.to_bits(),
            )
        })
        .collect::<Vec<_>>();
    let expected_residuals = [
        ("G01", 0x0000000000000000, 0x0000000000000000),
        ("G02", 0x0000000000000000, 0x0000000000000000),
        ("G03", 0xbe30000000000000, 0xbe40000000000000),
        ("G04", 0x0000000000000000, 0x0000000000000000),
        ("G05", 0x0000000000000000, 0x3e30000000000000),
        ("G06", 0x0000000000000000, 0x0000000000000000),
    ];
    let expected_residual_bits = (0..3)
        .flat_map(|epoch_index| {
            expected_residuals.iter().map(move |(sat, code, phase)| {
                (
                    epoch_index,
                    *sat,
                    *code,
                    *phase,
                    0x3ff0000000000000,
                    0x4059000000000000,
                )
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(residual_bits, expected_residual_bits);
    assert_eq!(
        solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(solution.ztd_residual_m, None);
    assert_eq!(solution.code_rms_m.to_bits(), 0x3e1a20bd700c2c3e);
    assert_eq!(solution.phase_rms_m.to_bits(), 0x3e2d363d1848dcbf);
    assert_eq!(solution.weighted_rms_m.to_bits(), 0x3e9023393a69dfe2);
    let err = norm3(sub3(solution.position_m, truth));
    assert!(err < 1.0e-3, "position error {err}");
    for (actual, expected) in solution.epoch_clocks_m.iter().zip(clocks) {
        assert!((actual - expected).abs() < 1.0e-4);
    }
    for (sat, expected) in ambiguities {
        assert!((solution.ambiguities_m[&sat] - expected).abs() < 1.0e-4);
    }
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
            residual_screen: false,
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
            residual_screen: false,
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
            residual_screen: false,
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
            residual_screen: false,
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
            residual_screen: false,
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
            residual_screen: false,
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
            residual_screen: false,
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
            residual_screen: false,
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
            residual_screen: false,
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
                residual_screen: false,
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
            residual_screen: false,
        },
    )
    .expect_err("NaN PPP correction table value must be rejected");

    assert_invalid_input(err, "ppp correction windup_m", "not finite");
}

#[test]
fn single_epoch_float_solver_has_frozen_bits_golden() {
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
            residual_screen: false,
        },
    )
    .unwrap();
    assert_eq!(
        solution.position_m.map(f64::to_bits),
        [0x414acd21ffffffff, 0x4127d1a7fffffffc, 0x415405af00000004]
    );
    assert_eq!(
        solution
            .epoch_clocks_m
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        [0x40290000000f7fd1]
    );
    assert_eq!(
        solution
            .ambiguities_m
            .iter()
            .map(|(sat, value)| (sat.as_str(), value.to_bits()))
            .collect::<Vec<_>>()
            .as_slice(),
        &[
            ("G01", 0x3fd0000000253e02),
            ("G02", 0x3fd6666669789a97),
            ("G03", 0x3fdccccccbfff0b5),
            ("G04", 0x3fe1999998df2955),
            ("G05", 0x3fe4cccccb81cd10),
            ("G06", 0x3fe8000000202d74),
        ]
    );
    assert_eq!(
        solution
            .residuals_m
            .iter()
            .map(|row| {
                (
                    row.epoch_index,
                    row.satellite_id.as_str(),
                    row.code_m.to_bits(),
                    row.phase_m.to_bits(),
                    row.code_weight.to_bits(),
                    row.phase_weight.to_bits(),
                )
            })
            .collect::<Vec<_>>(),
        ["G01", "G02", "G03", "G04", "G05", "G06"]
            .map(|sat| {
                (
                    0,
                    sat,
                    0x0000000000000000,
                    0x0000000000000000,
                    0x3ff0000000000000,
                    0x4059000000000000,
                )
            })
            .to_vec()
    );
    assert_eq!(
        solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(solution.ztd_residual_m, None);
    assert_eq!(solution.code_rms_m.to_bits(), 0x0000000000000000);
    assert_eq!(solution.phase_rms_m.to_bits(), 0x0000000000000000);
    assert_eq!(solution.weighted_rms_m.to_bits(), 0x0000000000000000);
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
            residual_screen: false,
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
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m,
                ratio_threshold: 3.0,
            },
        },
    )
    .unwrap();

    assert_eq!(solution.used_sats, ambiguity_ids);
    assert_eq!(solution.fixed_ambiguities_cycles, fixed_cycles);
    assert_eq!(solution.integer.ambiguity_search.order, solution.used_sats);
}

#[test]
fn static_fixed_solver_has_frozen_bits_golden() {
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
            jd_fraction: 0.5 + epoch_idx as f64 * 900.0 / 86_400.0,
            t_rx_j2000_s: epoch_idx as f64 * 900.0,
            observations,
        });
    }
    let initial = FloatState {
        position_m: [truth[0] + 500.0, truth[1] - 400.0, truth[2] + 300.0],
        clocks_m: vec![-20.0; epochs.len()],
        ambiguities_m: initial_ambiguities(&epochs),
        ztd_m: 0.0,
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
            residual_screen: false,
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
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m,
                ratio_threshold: 3.0,
            },
        },
    )
    .unwrap();
    assert_eq!(
        solution.position_m.map(f64::to_bits),
        [0x414acd2200000000, 0x4127d1a800000006, 0x415405aeffffffff]
    );
    assert_eq!(
        solution
            .epoch_clocks_m
            .iter()
            .copied()
            .map(f64::to_bits)
            .collect::<Vec<_>>(),
        [0x4028fffffff72706, 0xc02080000008d8f4, 0x400fffffffdc9c29]
    );
    assert_eq!(solution.fixed_ambiguities_cycles, fixed_cycles);
    assert_eq!(
        solution
            .fixed_ambiguities_m
            .iter()
            .map(|(sat, value)| (sat.as_str(), value.to_bits()))
            .collect::<Vec<_>>()
            .as_slice(),
        &[
            ("G01", 0x40cdbbbf359edc14),
            ("G02", 0x40cdbf4470b6d237),
            ("G03", 0x40cdc2c9abcec85a),
            ("G04", 0x40cdc64ee6e6be7d),
            ("G05", 0x40cdc9d421feb4a0),
            ("G06", 0x40cdcd595d16aac3),
        ]
    );
    assert_eq!(
        solution
            .residuals_m
            .iter()
            .map(|row| {
                (
                    row.epoch_index,
                    row.satellite_id.as_str(),
                    row.code_m.to_bits(),
                    row.phase_m.to_bits(),
                    row.code_weight.to_bits(),
                    row.phase_weight.to_bits(),
                )
            })
            .collect::<Vec<_>>(),
        (0..3)
            .flat_map(|epoch_idx| {
                ["G01", "G02", "G03", "G04", "G05", "G06"].map(move |sat| {
                    (
                        epoch_idx,
                        sat,
                        0x0000000000000000,
                        0x0000000000000000,
                        0x3ff0000000000000,
                        0x4059000000000000,
                    )
                })
            })
            .collect::<Vec<_>>()
    );
    assert_eq!(
        solution.used_sats,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(solution.ztd_residual_m, None);
    assert_eq!(solution.status, FloatStatus::StateTolerance);
    assert!(solution.converged);
    assert_eq!(solution.iterations, 1);
    assert_eq!(solution.integer.integer_status, IntegerStatus::Fixed);
    assert_eq!(solution.integer.integer_ratio.to_bits(), 0x4276126515ba5d18);
    assert_eq!(
        solution.integer.integer_best_score.to_bits(),
        0x3d5e3cacec751f88
    );
    assert_eq!(
        solution.integer.integer_second_best_score.map(f64::to_bits),
        Some(0x3fe4db1887df3c00)
    );
    assert_eq!(solution.integer.integer_candidates, 2);
    assert_eq!(solution.code_rms_m.to_bits(), 0x0000000000000000);
    assert_eq!(solution.phase_rms_m.to_bits(), 0x0000000000000000);
    assert_eq!(solution.weighted_rms_m.to_bits(), 0x0000000000000000);
    assert_eq!(
        solution.integer.ambiguity_search.order,
        ["G01", "G02", "G03", "G04", "G05", "G06"]
    );
    assert_eq!(
        solution
            .integer
            .ambiguity_search
            .float_cycles
            .iter()
            .map(|(sat, value)| (sat.as_str(), value.to_bits()))
            .collect::<Vec<_>>()
            .as_slice(),
        &[
            ("G01", 0x40f3880000000433),
            ("G02", 0x40f38a4ffffffc81),
            ("G03", 0x40f38ca000000271),
            ("G04", 0x40f38ef0000000cf),
            ("G05", 0x40f3913ffffffebd),
            ("G06", 0x40f3939000000282),
        ]
    );
    assert_eq!(
        solution
            .integer
            .ambiguity_search
            .covariance_cycles
            .iter()
            .map(|row| row.iter().copied().map(f64::to_bits).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![
            vec![
                0x40204d556724966a,
                0x3ffb59ca0689fc9b,
                0x3ff69795b8a75fe4,
                0xbffe45da405ab3a1,
                0x3fbf3b1876c9d95b,
                0xbfd2f78fa095dc18,
            ],
            vec![
                0x3ffb59ca0689fc9b,
                0x4014f5d392dddc47,
                0xbfec2cfa3b71f6f2,
                0x4004cbecce26de9e,
                0xbfffcbac718a652c,
                0x4004329afc1ba336,
            ],
            vec![
                0x3ff69795b8a75fe4,
                0xbfec2cfa3b71f6f2,
                0x4016a4544b41aacc,
                0x4008a52f3d81a840,
                0x3fff12fdc105fc9a,
                0xc00011e3d65d5856,
            ],
            vec![
                0xbffe45da405ab3a1,
                0x4004cbecce26de9e,
                0x4008a52f3d81a840,
                0x401686636ab5560f,
                0xbfde9fa4dbbe8dc6,
                0x3fd0f7f0319f0254,
            ],
            vec![
                0x3fbf3b1876c9d95b,
                0xbfffcbac718a652c,
                0x3fff12fdc105fc9a,
                0xbfde9fa4dbbe8dc6,
                0x401a0f2308a4ef10,
                0x4008be2c4a5bc3b4,
            ],
            vec![
                0xbfd2f78fa095dc18,
                0x4004329afc1ba336,
                0xc00011e3d65d5856,
                0x3fd0f7f0319f0254,
                0x4008be2c4a5bc3b4,
                0x40168387d4b345a7,
            ],
        ]
    );
}

#[test]
fn static_fixed_solver_rejects_short_float_solution_clock_vector() {
    let (source, epochs, state, _ambiguity_ids) = ppp_row_trace_arc();
    let used_sats = state.ambiguities_m.keys().cloned().collect::<Vec<_>>();
    let wavelength = C_M_S / F_L1_HZ;
    let float_solution = FloatSolution {
        position_m: state.position_m,
        epoch_clocks_m: vec![0.0; epochs.len() - 1],
        ambiguities_m: state.ambiguities_m,
        ztd_residual_m: None,
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
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m: used_sats
                    .iter()
                    .map(|sat| (sat.clone(), wavelength))
                    .collect(),
                offsets_m: used_sats.iter().map(|sat| (sat.clone(), 0.0)).collect(),
                ratio_threshold: 3.0,
            },
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
        epoch_clocks_m: vec![0.0; epochs.len()],
        ambiguities_m: state.ambiguities_m,
        ztd_residual_m: None,
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
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m: used_sats
                    .iter()
                    .map(|sat| (sat.clone(), wavelength))
                    .collect(),
                offsets_m: used_sats.iter().map(|sat| (sat.clone(), 0.0)).collect(),
                ratio_threshold: 3.0,
            },
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
        epoch_clocks_m: state.clocks_m,
        ambiguities_m: state.ambiguities_m,
        ztd_residual_m: None,
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
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m: used_sats.iter().map(|sat| (sat.clone(), 0.0)).collect(),
                ratio_threshold: 3.0,
            },
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
            jd_fraction: 0.5 + epoch_idx as f64 * 900.0 / 86_400.0,
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
        residual_screen: false,
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
