#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::geometry::{line_of_sight_from_az_el_deg, LineOfSight};
use sidereon_core::observables::{
    predict, ObservableEphemerisSource, ObservableState, ObservablesError, PredictOptions,
};
use sidereon_core::ppp_corrections::CivilDateTime;
use sidereon_core::precise_positioning::raim::{
    fault_free_accuracy_bounds, fde_float_epoch, global_test, global_test_with_geometry,
    per_satellite_statistics, protection_levels, protection_levels_with_weights,
    solve_float_epoch_with_raim, RaimConfig, RaimError, RaimFdeError, RaimFdeStatus,
    RaimGeometryRow, RaimStatus,
};
use sidereon_core::precise_positioning::{
    solve_float_epoch, FloatEpoch, FloatObservation, FloatResidual, FloatSolveConfig,
    FloatSolveOptions, FloatState, MeasurementWeights, RangeCorrections, TroposphereOptions,
};
use sidereon_core::{GnssSatelliteId, GnssSystem, Wgs84Geodetic};
use std::collections::BTreeMap;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/ppp_raim_oracles.json")).unwrap()
}

fn residual(id: usize, values: [f64; 4]) -> FloatResidual {
    FloatResidual {
        epoch_index: 0,
        satellite_id: format!("G{id:02}"),
        ambiguity_id: format!("G{id:02}"),
        code_m: values[0],
        phase_m: values[1],
        code_weight: values[2],
        phase_weight: values[3],
    }
}

fn geometry(n: usize) -> Vec<RaimGeometryRow> {
    fixture()["line_of_sight_ecef"].as_array().unwrap()[..n]
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let x = v.as_array().unwrap();
            RaimGeometryRow {
                satellite_id: format!("G{:02}", i + 1),
                line_of_sight: LineOfSight::new(
                    x[0].as_f64().unwrap(),
                    x[1].as_f64().unwrap(),
                    x[2].as_f64().unwrap(),
                ),
            }
        })
        .collect()
}

fn zeros(n: usize) -> Vec<FloatResidual> {
    (1..=n).map(|i| residual(i, [0.0, 0.0, 1.0, 1.0])).collect()
}

fn assert_close(actual: f64, expected: f64, tol: f64) {
    assert!(
        (actual - expected).abs() <= tol,
        "actual={actual:.17e} expected={expected:.17e} tol={tol:.1e}"
    );
}

fn decimal(value: &Value) -> f64 {
    value.as_str().unwrap().parse().unwrap()
}

#[derive(Clone)]
struct PublicSource {
    states: BTreeMap<GnssSatelliteId, [f64; 3]>,
}

impl ObservableEphemerisSource for PublicSource {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        _: f64,
    ) -> Result<ObservableState, ObservablesError> {
        Ok(ObservableState {
            position_ecef_m: *self.states.get(&sat).ok_or(ObservablesError::NoEphemeris)?,
            clock_s: Some(0.0),
        })
    }
}

fn gradient_case(code_bias_m: f64) -> (PublicSource, FloatEpoch, FloatState, FloatSolveConfig) {
    let truth = [3_512_900., 780_500., 5_248_700.];
    let receiver = sidereon_core::itrf_to_geodetic(
        sidereon_core::ItrfPositionM::new(truth[0], truth[1], truth[2]).unwrap(),
    )
    .unwrap();
    let az_el = [
        (15., 38.),
        (55., 52.),
        (105., 33.),
        (145., 47.),
        (205., 31.),
        (250., 56.),
        (300., 42.),
        (335., 64.),
    ];
    let positions = az_el.map(|(az, el)| {
        let los = line_of_sight_from_az_el_deg(az, el, receiver).unwrap();
        [
            truth[0] + 25_000_000. * los.e_x,
            truth[1] + 25_000_000. * los.e_y,
            truth[2] + 25_000_000. * los.e_z,
        ]
    });
    let ids = (1..=8)
        .map(|prn| GnssSatelliteId::new(GnssSystem::Gps, prn).unwrap())
        .collect::<Vec<_>>();
    let source = PublicSource {
        states: ids.iter().copied().zip(positions).collect(),
    };
    let observations = ids
        .iter()
        .enumerate()
        .map(|(idx, id)| {
            let token = id.to_string();
            let prediction = predict(&source, *id, truth, 0.0, PredictOptions::default()).unwrap();
            FloatObservation {
                sat: *id,
                satellite_id: token.clone(),
                ambiguity_id: token.clone(),
                code_m: prediction.geometric_range_m
                    + 12.5
                    + if token == "G05" { code_bias_m } else { 0.0 },
                phase_m: prediction.geometric_range_m + 12.5 + 0.25 + idx as f64 * 0.1,
                freq1_hz: 0.0,
                freq2_hz: 0.0,
                glonass_channel: None,
                signals: None,
            }
        })
        .collect::<Vec<_>>();
    let ambiguities_m = observations
        .iter()
        .map(|o| (o.ambiguity_id.clone(), o.phase_m - o.code_m))
        .collect();
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
    let state = FloatState {
        position_m: [truth[0] + 80., truth[1] - 60., truth[2] + 40.],
        clocks_m: vec![0.],
        ambiguities_m,
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let mut tropo = TroposphereOptions::disabled();
    tropo.enabled = true;
    tropo.estimate_ztd = true;
    tropo.estimate_tropo_gradients = true;
    let config = FloatSolveConfig::new(
        MeasurementWeights {
            code: 1.,
            phase: 1.,
            elevation_weighting: false,
        },
        tropo,
        RangeCorrections::disabled(),
        FloatSolveOptions::default(),
        None,
        false,
        false,
    );
    (source, epoch, state, config)
}

fn fixture_weighted_solver_case() -> (PublicSource, FloatEpoch, FloatState, FloatSolveConfig) {
    let f = fixture();
    let truth_values = f["inputs"]["receiver_ecef_m"].as_array().unwrap();
    let truth = [
        truth_values[0].as_f64().unwrap(),
        truth_values[1].as_f64().unwrap(),
        truth_values[2].as_f64().unwrap(),
    ];
    let satellites = f["inputs"]["satellite_ecef_m"].as_array().unwrap();
    let ids = (1..=satellites.len())
        .map(|prn| GnssSatelliteId::new(GnssSystem::Gps, prn as u8).unwrap())
        .collect::<Vec<_>>();
    let positions = satellites.iter().map(|point| {
        let point = point.as_array().unwrap();
        [
            point[0].as_f64().unwrap(),
            point[1].as_f64().unwrap(),
            point[2].as_f64().unwrap(),
        ]
    });
    let source = PublicSource {
        states: ids.iter().copied().zip(positions).collect(),
    };
    let observations = ids
        .iter()
        .enumerate()
        .map(|(index, id)| {
            let token = id.to_string();
            let prediction = predict(&source, *id, truth, 0.0, PredictOptions::default()).unwrap();
            FloatObservation {
                sat: *id,
                satellite_id: token.clone(),
                ambiguity_id: token,
                code_m: prediction.geometric_range_m + 12.5,
                phase_m: prediction.geometric_range_m + 12.75 + index as f64 * 0.1,
                freq1_hz: 0.0,
                freq2_hz: 0.0,
                glonass_channel: None,
                signals: None,
            }
        })
        .collect::<Vec<_>>();
    let ambiguities_m = observations
        .iter()
        .map(|observation| {
            (
                observation.ambiguity_id.clone(),
                observation.phase_m - observation.code_m,
            )
        })
        .collect();
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
    let state = FloatState {
        position_m: [truth[0] + 20.0, truth[1] - 15.0, truth[2] + 10.0],
        clocks_m: vec![0.0],
        ambiguities_m,
        ztd_m: 0.0,
        tropo_gradient_north_m: 0.0,
        tropo_gradient_east_m: 0.0,
        residual_ionosphere_m: BTreeMap::new(),
    };
    let config = FloatSolveConfig::new(
        MeasurementWeights {
            code: 0.1,
            phase: 1.0,
            elevation_weighting: false,
        },
        TroposphereOptions::disabled(),
        RangeCorrections::disabled(),
        FloatSolveOptions::default(),
        None,
        false,
        false,
    );
    (source, epoch, state, config)
}

#[test]
fn global_weighted_chi_square_agrees_with_numpy_and_scipy() {
    let f = fixture();
    let rows = f["global"]["residuals"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let a = v.as_array().unwrap();
            residual(
                i + 1,
                [
                    a[0].as_f64().unwrap(),
                    a[1].as_f64().unwrap(),
                    a[2].as_f64().unwrap(),
                    a[3].as_f64().unwrap(),
                ],
            )
        })
        .collect::<Vec<_>>();
    let mut c = RaimConfig::default();
    c.measurement_sigma_m = 2.0;
    let got = global_test(&rows, 5, c).unwrap();
    assert_close(
        got.test_statistic,
        f["global"]["weighted_sse"].as_f64().unwrap(),
        2.0e-15,
    ); // 12 elementary operations, bounded by 8 eps times the result.
    let expected_threshold = f["chi_square_isf_0p001"][0].as_f64().unwrap();
    assert_close(
        got.threshold.unwrap(),
        expected_threshold,
        8.0 * f64::EPSILON * expected_threshold.abs(),
    ); // Safeguarded Newton refinement contributes at most a few ulps here; eight allows the gamma-tail rounding.
    assert_eq!(got.redundancy, 1);
    assert_eq!(got.status, RaimStatus::Passed);
}

#[test]
fn global_edges_dof_zero_one_and_nonfinite() {
    assert_eq!(
        global_test(&zeros(2), 4, RaimConfig::default())
            .unwrap()
            .status,
        RaimStatus::NotEnoughRedundancy
    );
    assert_eq!(
        global_test(&zeros(3), 5, RaimConfig::default())
            .unwrap()
            .redundancy,
        1
    );
    for bad in [f64::NAN, f64::INFINITY] {
        let mut r = zeros(3);
        r[0].code_m = bad;
        assert!(matches!(
            global_test(&r, 5, RaimConfig::default()),
            Err(RaimError::InvalidResidual { .. })
        ));
    }
    assert_eq!(
        global_test(&[], 0, RaimConfig::default()).unwrap().status,
        RaimStatus::NotEnoughRedundancy
    );
}

#[test]
fn global_test_rejects_overflowing_state_count() {
    let result = global_test(&[], usize::MAX, RaimConfig::default());
    assert!(
        matches!(result, Err(RaimError::InvalidConfig { .. }))
            || result.unwrap().status == RaimStatus::NotEnoughRedundancy
    );
}

#[test]
fn protection_levels_are_unavailable_at_zero_redundancy() {
    let geo = &fixture()["inputs"]["receiver_geodetic_rad"];
    let receiver =
        Wgs84Geodetic::new(geo[0].as_f64().unwrap(), geo[1].as_f64().unwrap(), 0.0).unwrap();
    assert!(protection_levels(&geometry(4), receiver, RaimConfig::default()).is_err());
}

#[test]
fn clean_zero_redundancy_fde_is_untested() {
    let (source, mut epoch, state, mut config) = gradient_case(0.0);
    epoch.observations.truncate(4);
    config.tropo.enabled = false;
    config.tropo.estimate_ztd = false;
    config.tropo.estimate_tropo_gradients = false;
    let result = fde_float_epoch(&source, epoch, state, config, RaimConfig::default()).unwrap();
    assert_eq!(result.raim.status, RaimStatus::NotEnoughRedundancy);
    assert_eq!(result.raim.redundancy, 0);
    assert!(!result.raim.detected);
    assert_eq!(result.raim.threshold, None);
    assert!(result.raim.test_statistic.is_finite() && result.raim.test_statistic >= 0.0);
    assert!(result.raim.satellite_statistics.is_empty());
    assert_eq!(result.raim.most_likely_fault, None);
    assert_eq!(result.raim.hpl_m, None);
    assert_eq!(result.raim.vpl_m, None);
    assert!(result.excluded_sats.is_empty());
    assert_eq!(result.status, RaimFdeStatus::Untested);
}

#[test]
fn singular_geometry_is_rejected() {
    let mut g = geometry(6);
    let repeated = g[0].line_of_sight;
    for row in &mut g {
        row.line_of_sight = repeated;
    }
    assert_eq!(
        per_satellite_statistics(&zeros(6), &g, RaimConfig::default()),
        Err(RaimError::SingularGeometry)
    );
}

#[test]
fn zero_statistics_have_no_most_likely_fault() {
    assert_eq!(
        per_satellite_statistics(&zeros(6), &geometry(6), RaimConfig::default())
            .unwrap()
            .most_likely_fault,
        None
    );
}

#[test]
fn geometry_length_mismatch_is_invalid_even_without_redundancy() {
    assert!(matches!(
        global_test_with_geometry(&zeros(2), &geometry(6), RaimConfig::default()),
        Err(RaimError::InvalidGeometry {
            field: "geometry",
            ..
        })
    ));
}

#[test]
fn protection_level_uses_fault_slope_and_detectable_bias() {
    let f = fixture();
    let geo = &f["inputs"]["receiver_geodetic_rad"];
    let receiver =
        Wgs84Geodetic::new(geo[0].as_f64().unwrap(), geo[1].as_f64().unwrap(), 0.0).unwrap();
    for (i, n) in [6, 5].into_iter().enumerate() {
        let got = protection_levels(&geometry(n), receiver, RaimConfig::default()).unwrap();
        let oracle = &f["slope_cases"][i];
        let high_precision = &f["mpmath_slope_cases"][i];
        // The matrix/rotation forward bound is 8*eps*kappa_2*PL. The log-CDF
        // quadrature accumulates at most 128 rounded active-path operations;
        // propagating 128*eps relative CDF error through the fixture's
        // d(lambda)/d(CDF), plus the four-ulp root bracket, gives the second
        // term. This contains only conditioning and rounding bounds.
        let condition = oracle["normal_condition_number_2"].as_f64().unwrap();
        let magnitude = decimal(&high_precision["hpl_slope"])
            .abs()
            .max(decimal(&high_precision["vpl_slope"]).abs());
        let lambda = decimal(&high_precision["pbias"]).powi(2);
        let inverse_condition = decimal(&high_precision["d_lambda_d_cdf"]).abs();
        let root_relative =
            4.0 * f64::EPSILON + 128.0 * f64::EPSILON * 1.0e-3 * inverse_condition / lambda;
        let tolerance_m = 8.0 * f64::EPSILON * condition * magnitude + root_relative * magnitude;
        assert_close(
            got.hpl_m,
            decimal(&high_precision["hpl_slope"]),
            tolerance_m,
        );
        assert_close(
            got.vpl_m,
            decimal(&high_precision["vpl_slope"]),
            tolerance_m,
        );
    }
}

#[test]
fn module_multiplier_is_two_sided_normal_quantile() {
    let f = fixture();
    let p_md = f["inputs"]["missed_detection_probability"]
        .as_f64()
        .unwrap();
    let engine = sidereon_core::quality::chi2_isf(p_md, 1).unwrap().sqrt();
    assert_close(
        engine,
        f["normal"]["two_sided"].as_f64().unwrap(),
        4.0 * f64::EPSILON * engine.abs(),
    );
}

#[test]
fn gradient_states_are_counted_in_public_solver_raim_redundancy() {
    let (source, epoch, state, config) = gradient_case(0.0);
    let solution =
        solve_float_epoch(&source, epoch.clone(), state.clone(), config.clone()).unwrap();
    let expected_states = 3 + solution.epoch_clocks_m.len() + 1 + 2 + solution.ambiguities_m.len();
    let expected_dof = (2 * solution.residuals_m.len()) as isize - expected_states as isize;
    assert_eq!(expected_dof, 1);
    let mut raim = RaimConfig::default();
    raim.chi_square_threshold = Some(1.0e10);
    let got = solve_float_epoch_with_raim(&source, epoch, state, config, raim).unwrap();
    assert_eq!(got.raim.redundancy, expected_dof);
    assert_eq!(got.status, RaimFdeStatus::Clean);
    assert_eq!(got.raim.status, RaimStatus::Passed);
    assert!(!got.raim.detected);
    assert!(got.raim.threshold.is_some());
    assert!(!got.raim.satellite_statistics.is_empty());
    assert!(got.raim.hpl_m.is_some());
    assert!(got.raim.vpl_m.is_some());
}

#[test]
fn residual_ionosphere_states_are_counted_by_solver_admission() {
    let (source, epoch, state, mut config) = gradient_case(0.0);
    config.tropo.estimate_tropo_gradients = false;
    config.estimate_residual_ionosphere = true;
    // There are 16 code/phase rows but 3 position + 1 clock + 8 ionosphere +
    // 8 ambiguity states. The solver adds no constraint/pseudo-observation rows.
    let result = solve_float_epoch_with_raim(&source, epoch, state, config, RaimConfig::default());
    assert!(
        matches!(result, Err(RaimFdeError::Solve(_))),
        "solver unexpectedly admitted the underdetermined ionosphere solve: {result:?}"
    );
}

#[test]
fn gradient_fde_refuses_exclusion_before_the_public_solver_fails() {
    let (source, epoch, state, config) = gradient_case(50.0);
    let mut raim = RaimConfig::default();
    raim.chi_square_threshold = Some(1.0e-4);
    let direct = fde_float_epoch(&source, epoch.clone(), state.clone(), config.clone(), raim);
    let wrapper = solve_float_epoch_with_raim(&source, epoch, state, config, raim);
    let direct = direct.unwrap();
    let wrapper = wrapper.unwrap();
    for got in [&direct, &wrapper] {
        assert_eq!(got.status, RaimFdeStatus::CannotExclude);
        assert_eq!(got.raim.status, RaimStatus::FaultDetected);
        assert!(got.raim.detected);
        assert_eq!(got.raim.redundancy, 1);
        assert!(got.raim.threshold.is_some());
        assert!(!got.raim.satellite_statistics.is_empty());
    }
}

#[test]
fn protection_levels_use_the_residual_tests_non_unit_weights() {
    let f = fixture();
    let rows = (1..=6)
        .map(|i| residual(i, [1.0, 0.0, 0.1, 1.0]))
        .collect::<Vec<_>>();
    let weighted = global_test(&rows, 10, RaimConfig::default()).unwrap();
    assert_close(weighted.test_statistic, 0.06, 2.0e-17);
    let geo = &f["inputs"]["receiver_geodetic_rad"];
    let receiver =
        Wgs84Geodetic::new(geo[0].as_f64().unwrap(), geo[1].as_f64().unwrap(), 0.0).unwrap();
    let got = protection_levels_with_weights(
        &geometry(6),
        &rows.iter().map(|row| row.code_weight).collect::<Vec<_>>(),
        receiver,
        RaimConfig::default(),
    )
    .unwrap();
    let slope = &f["slope_cases"][0];
    let high_precision = &f["mpmath_slope_cases"][0];
    let lambda = decimal(&high_precision["pbias"]).powi(2);
    let root_relative = 4.0 * f64::EPSILON
        + 128.0 * f64::EPSILON * 1.0e-3 * decimal(&high_precision["d_lambda_d_cdf"]).abs() / lambda;
    let weighted_tolerance_m = 10.0
        * (8.0
            * f64::EPSILON
            * slope["normal_condition_number_2"].as_f64().unwrap()
            * slope["vpl_slope"].as_f64().unwrap()
            + root_relative * slope["vpl_slope"].as_f64().unwrap());
    assert_close(
        got.hpl_m,
        f["weighted_protection"]["hpl_slope_m"].as_f64().unwrap(),
        weighted_tolerance_m, // Ten-metre effective sigma scales the derived bound by ten.
    );
    assert_close(
        got.vpl_m,
        f["weighted_protection"]["vpl_slope_m"].as_f64().unwrap(),
        weighted_tolerance_m,
    );
}

#[test]
fn solver_protection_levels_use_float_residual_weights() {
    let (source, epoch, state, config) = fixture_weighted_solver_case();
    let result =
        solve_float_epoch_with_raim(&source, epoch, state, config, RaimConfig::default()).unwrap();
    let f = fixture();
    let oracle = &f["weighted_protection"];
    let condition = f["slope_cases"][0]["normal_condition_number_2"]
        .as_f64()
        .unwrap();
    let h_expected = oracle["hpl_slope_m"].as_f64().unwrap();
    let v_expected = oracle["vpl_slope_m"].as_f64().unwrap();
    assert_close(
        result.raim.hpl_m.unwrap(),
        h_expected,
        64.0 * f64::EPSILON * condition * h_expected,
    );
    assert_close(
        result.raim.vpl_m.unwrap(),
        v_expected,
        64.0 * f64::EPSILON * condition * v_expected,
    );
}

#[test]
fn nonuniform_inverse_sigma_weights_match_numpy_scipy() {
    let f = fixture();
    let oracle = &f["nonuniform_weighted_protection"];
    let weights = oracle["weights_inverse_sigma"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_f64().unwrap())
        .collect::<Vec<_>>();
    let geo = &f["inputs"]["receiver_geodetic_rad"];
    let receiver =
        Wgs84Geodetic::new(geo[0].as_f64().unwrap(), geo[1].as_f64().unwrap(), 0.0).unwrap();
    let protection =
        protection_levels_with_weights(&geometry(6), &weights, receiver, RaimConfig::default())
            .unwrap();
    let accuracy =
        fault_free_accuracy_bounds(&geometry(6), &weights, receiver, RaimConfig::default())
            .unwrap();
    let condition = oracle["normal_condition_number_2"].as_f64().unwrap();
    for (actual, key) in [
        (protection.hpl_m, "hpl_slope_m"),
        (protection.vpl_m, "vpl_slope_m"),
        (accuracy.hpl_m, "fault_free_h_m"),
        (accuracy.vpl_m, "fault_free_v_m"),
    ] {
        let expected = oracle[key].as_f64().unwrap();
        let tolerance = 32.0 * f64::EPSILON * condition * expected.abs();
        assert_close(actual, expected, tolerance);
    }
}
