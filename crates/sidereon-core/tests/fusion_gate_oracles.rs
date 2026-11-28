#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::astro::constants::earth::WGS84_A_M;
use sidereon_core::fusion::{
    ekf_correct_closed_loop, EkfCorrection, EkfUpdateOptions, ErrorStateLayout, FusionFilterKind,
    GnssFixMeasurement, IggIiiMeasurementReweighting, InertialFilter, InertialFilterConfig,
    InnovationGate, InsFilterState, LooseCouplingConfig, UkfUpdateOptions,
    YangPredictionAdaptiveFactor, ERROR_STATE_DIMENSION_15,
};
use sidereon_core::inertial::{ImuGrade, ImuSpec, NavState};

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/fusion_gate_oracles.json")).unwrap()
}

#[test]
fn ukf_zero_accepted_rows_coasts_when_min_rows_is_zero() {
    let mut st = state(1., 1.);
    let before = st.clone();
    let mut h = vec![vec![0.; ERROR_STATE_DIMENSION_15]; 3];
    h[0][0] = 1.;
    h[1][1] = 1.;
    h[2][2] = 1.;
    let c = EkfCorrection::new(
        vec![50.; 3],
        h,
        vec![vec![1., 0., 0.], vec![0., 1., 0.], vec![0., 0., 1.]],
    )
    .unwrap();
    let mut options = UkfUpdateOptions::default();
    options.innovation_gate = gated(3., 0).innovation_gate;
    let report = sidereon_core::fusion::ukf_correct_closed_loop(&mut st, &c, options).unwrap();
    assert!(!report.applied);
    assert_eq!((report.accepted_rows, report.rejected_rows), (0, 3));
    assert_eq!(st, before);
}
fn identity3() -> [[f64; 3]; 3] {
    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
}
fn state(p0: f64, p1: f64) -> InsFilterState {
    let n = NavState::new(0.0, [WGS84_A_M, 0., 0.], [0.; 3], identity3()).unwrap();
    let mut d = vec![1.; ERROR_STATE_DIMENSION_15];
    d[0] = p0;
    d[1] = p1;
    InsFilterState::from_diagonal(n, ErrorStateLayout::Fifteen, &d).unwrap()
}
fn fixture_bits(value: &Value) -> u64 {
    u64::from_str_radix(value.as_str().unwrap().strip_prefix("0x").unwrap(), 16).unwrap()
}

fn fixture_number(value: &Value) -> f64 {
    f64::from_bits(fixture_bits(value))
}

fn fixture_vector(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(fixture_number)
        .collect()
}

fn gated(threshold_sigma: f64, min_rows: usize) -> EkfUpdateOptions {
    let mut options = EkfUpdateOptions::default();
    options.innovation_gate = Some(InnovationGate {
        threshold_sigma,
        min_rows,
    });
    options
}

#[test]
fn gated_update_matches_joseph_reference_bits() {
    let f = fixture();
    let u = &f["update"];
    let mut st = state(2., 3.);
    let mut h = vec![vec![0.; ERROR_STATE_DIMENSION_15]; 2];
    h[0][0] = 1.;
    h[1][0] = 0.5;
    h[1][1] = 1.;
    let corr = EkfCorrection::new(vec![1.25, -4.], h, vec![vec![1., 0.2], vec![0.2, 2.]]).unwrap();
    let report = ekf_correct_closed_loop(&mut st, &corr, gated(1.5, 1)).unwrap();
    assert_eq!((report.accepted_rows, report.rejected_rows), (1, 1));
    for row in 0..2 {
        assert_eq!(
            st.nominal.position_ecef_m[row].to_bits(),
            fixture_bits(&u["nominal_position_ecef_m_bits"][row]),
            "position axis {row}"
        );
        for column in 0..2 {
            assert_eq!(
                st.covariance[row][column].to_bits(),
                fixture_bits(&u["covariance_bits"][row][column]),
                "covariance ({row}, {column})"
            );
        }
    }
}

#[test]
fn per_row_gate_accepts_exact_threshold_and_is_not_joint_mahalanobis() {
    let mut st = state(0.5, 0.5);
    let mut h = vec![vec![0.; ERROR_STATE_DIMENSION_15]; 2];
    h[0][0] = 1.;
    h[1][1] = 1.;
    let corr =
        EkfCorrection::new(vec![3., 3.], h, vec![vec![0.5, -0.45], vec![-0.45, 0.5]]).unwrap();
    let report = ekf_correct_closed_loop(&mut st, &corr, gated(3., 1)).unwrap();
    assert_eq!(report.accepted_rows, 2);
    assert!(report.applied);
    let c = &fixture()["correlated_verdict"];
    assert!(c["row_gate_pass"].as_bool().unwrap());
    assert!(!c["joint_pass"].as_bool().unwrap());
}

#[test]
fn joint_nis_gate_rejects_correlated_rows_for_ekf_and_ukf() {
    let oracle = &fixture()["correlated_verdict"];
    let mut h = vec![vec![0.; ERROR_STATE_DIMENSION_15]; 2];
    h[0][0] = 1.;
    h[1][1] = 1.;
    let innovation = fixture_vector(&oracle["innovation_bits"]);
    let measurement_covariance = oracle["r_bits"]
        .as_array()
        .unwrap()
        .iter()
        .map(fixture_vector)
        .collect();
    let correction = EkfCorrection::new(innovation, h, measurement_covariance).unwrap();
    let probability = fixture_number(&oracle["joint_false_alarm_probability_bits"]);
    let p = &oracle["p_block_bits"];
    let mut filter_state = state(fixture_number(&p[0][0]), fixture_number(&p[1][1]));
    filter_state.covariance[0][1] = fixture_number(&p[0][1]);
    filter_state.covariance[1][0] = fixture_number(&p[1][0]);

    let mut ekf_options = EkfUpdateOptions::default();
    ekf_options.joint_innovation_gate_probability = Some(probability);
    let ekf = ekf_correct_closed_loop(&mut filter_state.clone(), &correction, ekf_options).unwrap();

    let mut ukf_options = UkfUpdateOptions::default();
    ukf_options.joint_innovation_gate_probability = Some(probability);
    let ukf =
        sidereon_core::fusion::ukf_correct_closed_loop(&mut filter_state, &correction, ukf_options)
            .unwrap();

    for (report, expected) in [(ekf, "joint_ekf_nis_bits"), (ukf, "joint_ukf_nis_bits")] {
        let nis = fixture_bits(&oracle[expected]);
        assert_eq!(
            report.normalized_innovation_squared.to_bits(),
            nis,
            "{expected}"
        );
        assert!(!report.applied);
        assert!(report.innovation_gate.is_none());
        let gate = report.joint_innovation_gate.expect("joint gate report");
        assert_eq!(gate.statistic.to_bits(), nis, "{expected}");
        assert_eq!(gate.degrees_of_freedom, 2);
        assert_eq!(
            gate.threshold.to_bits(),
            fixture_bits(&oracle["joint_threshold_bits"])
        );
        assert_eq!(
            gate.false_alarm_probability.to_bits(),
            probability.to_bits()
        );
        assert!(!gate.accepted);
    }
}

#[test]
fn zero_accepted_rows_coast_even_when_min_rows_is_zero() {
    let mut st = state(1., 1.);
    let mut h = vec![vec![0.; ERROR_STATE_DIMENSION_15]; 1];
    h[0][0] = 1.;
    let c = EkfCorrection::new(vec![50.], h, vec![vec![1.]]).unwrap();
    let r = ekf_correct_closed_loop(&mut st, &c, gated(3., 0)).unwrap();
    assert!(!r.applied);
    assert_eq!(r.accepted_rows, 0);
}

fn loose_filter(robust: bool, gate: bool) -> InertialFilter {
    loose_filter_with_diagonal(robust, gate, [1.; 6], None)
}

fn loose_filter_with_diagonal(
    robust: bool,
    gate: bool,
    first_six: [f64; 6],
    adaptation: Option<YangPredictionAdaptiveFactor>,
) -> InertialFilter {
    let nominal = NavState::new(0., [WGS84_A_M, 0., 0.], [0.; 3], identity3()).unwrap();
    let mut diagonal = [1.; ERROR_STATE_DIMENSION_15];
    diagonal[..6].copy_from_slice(&first_six);
    let state =
        InsFilterState::from_diagonal(nominal, ErrorStateLayout::Fifteen, &diagonal).unwrap();
    let spec = ImuSpec::preset(ImuGrade::Mems);
    let mut cfg = InertialFilterConfig::new(spec).unwrap();
    let mut loose = LooseCouplingConfig::default();
    loose.measurement_reweighting = robust.then(IggIiiMeasurementReweighting::standard);
    loose.prediction_adaptation = adaptation;
    if gate {
        loose.update_options = gated(3., 1);
    }
    cfg.loose = loose;
    InertialFilter::with_config(state, cfg).unwrap()
}
fn measurement_with_covariance(
    position: [f64; 3],
    velocity: [f64; 3],
    diagonal: [f64; 6],
) -> GnssFixMeasurement {
    GnssFixMeasurement::position_velocity(
        0.,
        position,
        velocity,
        (0..6)
            .map(|i| {
                (0..6)
                    .map(|j| if i == j { diagonal[i] } else { 0. })
                    .collect()
            })
            .collect(),
        8,
    )
    .unwrap()
}
fn measurement(x: f64) -> GnssFixMeasurement {
    GnssFixMeasurement::position_velocity(
        0.,
        [WGS84_A_M + x, 0., 0.],
        [0.; 3],
        (0..6)
            .map(|i| (0..6).map(|j| if i == j { 1. } else { 0. }).collect())
            .collect(),
        8,
    )
    .unwrap()
}

#[test]
fn gate_precedes_igg_reweighting() {
    let mut f = loose_filter(true, true);
    let u = f.update_loose(&measurement(50.)).unwrap();
    assert_eq!(u.accepted_rows, 5);
    assert_eq!(u.rejected_rows, 1);
}

#[test]
fn gate_uses_raw_covariance_after_igg_removes_rows() {
    let oracle = &fixture()["raw_gate_after_igg_removal"];
    let innovation = fixture_vector(&oracle["innovation_bits"]);
    let nominal = NavState::new(0., [WGS84_A_M, 0., 0.], [0.; 3], identity3()).unwrap();
    let measurement = measurement_with_covariance(
        [WGS84_A_M + innovation[0], innovation[1], innovation[2]],
        [innovation[3], innovation[4], innovation[5]],
        [1.0; 6],
    );
    for filter_kind in [FusionFilterKind::Ekf, FusionFilterKind::Ukf] {
        let state = InsFilterState::from_diagonal(
            nominal,
            ErrorStateLayout::Fifteen,
            &[1.; ERROR_STATE_DIMENSION_15],
        )
        .unwrap();
        let mut config = InertialFilterConfig::new(ImuSpec::preset(ImuGrade::Mems)).unwrap();
        config.filter_kind = filter_kind;
        config.loose.measurement_reweighting = Some(IggIiiMeasurementReweighting {
            k0_sigma: 2.0,
            k1_sigma: 5.0,
        });
        if filter_kind == FusionFilterKind::Ekf {
            config.loose.update_options = gated(3.0, 1);
        } else {
            config.ukf_update_options.innovation_gate = gated(3.0, 1).innovation_gate;
        }
        let mut filter = InertialFilter::with_config(state, config).unwrap();
        let before = filter.state().clone();
        let report = filter.update_loose(&measurement).unwrap();
        assert!(!report.applied);
        assert_eq!(report.accepted_rows, 0);
        assert_eq!(report.rejected_rows, 6);
        assert_eq!(filter.state(), &before);
    }
}

#[test]
fn joint_gate_reports_raw_nis_after_igg_removes_rows() {
    let oracle = &fixture()["raw_gate_after_igg_removal"];
    let values = fixture_vector(&oracle["innovation_bits"]);
    let expected = fixture_bits(&oracle["retained_nis_bits"]);
    let measurement = measurement_with_covariance(
        [WGS84_A_M + values[0], values[1], values[2]],
        [values[3], values[4], values[5]],
        [1.0; 6],
    );
    for filter_kind in [FusionFilterKind::Ekf, FusionFilterKind::Ukf] {
        let nominal = NavState::new(0., [WGS84_A_M, 0., 0.], [0.; 3], identity3()).unwrap();
        let state = InsFilterState::from_diagonal(
            nominal,
            ErrorStateLayout::Fifteen,
            &[1.; ERROR_STATE_DIMENSION_15],
        )
        .unwrap();
        let mut config = InertialFilterConfig::new(ImuSpec::preset(ImuGrade::Mems)).unwrap();
        config.filter_kind = filter_kind;
        config.loose.measurement_reweighting = Some(IggIiiMeasurementReweighting {
            k0_sigma: 2.0,
            k1_sigma: 5.0,
        });
        if filter_kind == FusionFilterKind::Ekf {
            config
                .loose
                .update_options
                .joint_innovation_gate_probability = Some(0.01);
        } else {
            config.ukf_update_options.joint_innovation_gate_probability = Some(0.01);
        }
        let mut filter = InertialFilter::with_config(state, config).unwrap();
        let report = filter.update_loose(&measurement).unwrap();
        assert!(!report.applied);
        assert_eq!((report.accepted_rows, report.rejected_rows), (0, 6));
        assert_eq!(report.nis.to_bits(), expected, "{filter_kind:?}");
        let gate = report.ekf.joint_innovation_gate.expect("joint gate report");
        assert_eq!(gate.statistic.to_bits(), expected, "{filter_kind:?}");
        assert_eq!(gate.degrees_of_freedom, 1);
        assert!(!gate.accepted);
    }
}

#[test]
fn igg_zero_weight_region_has_no_state_effect() {
    let mut f = loose_filter(true, false);
    f.update_loose(&measurement(50.)).unwrap();
    assert_eq!(
        f.state().nominal.position_ecef_m[0].to_bits(),
        WGS84_A_M.to_bits()
    );
}

#[test]
fn igg_all_zero_weight_rows_coast_without_changing_state_or_covariance() {
    let mut filter = loose_filter(true, false);
    let before = filter.state().clone();
    let measurement =
        measurement_with_covariance([WGS84_A_M + 50.0, 50.0, 50.0], [50.0; 3], [1.0; 6]);
    let report = filter.update_loose(&measurement).unwrap();
    assert!(!report.applied);
    assert_eq!(report.accepted_rows, 0);
    assert_eq!(report.rejected_rows, 6);
    assert_eq!(report.rows, 6);
    assert_eq!(filter.state(), &before);
}

#[test]
fn scipy_yang_guard_thresholds_are_monotone_for_rows_one_to_twelve() {
    let a = fixture()["yang_chi2_0p99"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(a.len(), 12);
    assert!(a.windows(2).all(|x| x[0] < x[1]));
}

#[test]
fn published_igg_weights_cover_breakpoints_and_fine_grid() {
    let f = fixture();
    let w = f["weights"].as_array().unwrap();
    assert_eq!(w[1][1].as_f64().unwrap(), 1.0);
    assert_eq!(w[7][1].as_f64().unwrap(), 0.0);
    assert_eq!(w[8][1].as_f64().unwrap(), 0.0);
    let g = f["transition_grid"]["weights"].as_array().unwrap();
    assert!(g
        .windows(2)
        .all(|x| x[1].as_f64().unwrap() <= x[0].as_f64().unwrap()));
}

#[test]
fn applied_igg_weight_is_non_increasing_across_rejection_boundary() {
    let standardized = fixture()["applied_scale_scan"]["standardized"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect::<Vec<_>>();
    let mut applied_weights = Vec::new();
    for value in standardized {
        let mut filter = loose_filter(true, false);
        let innovation = value * libm::sqrt(2.0);
        filter.update_loose(&measurement(innovation)).unwrap();
        let delta = filter.state().nominal.position_ecef_m[0] - WGS84_A_M;
        let variance_scale = innovation / delta - 1.0;
        applied_weights.push(1.0 / variance_scale);
    }
    assert!(applied_weights.windows(2).all(|pair| pair[1] <= pair[0]));
}

#[test]
fn yang_mixed_position_velocity_statistic_is_unit_consistent() {
    let oracle = &fixture()["yang_mixed_units"];
    let innovation = fixture_vector(&oracle["innovation_bits"]);
    let predicted_variances: [f64; 6] = fixture_vector(&oracle["predicted_variance_bits"])
        .try_into()
        .unwrap();
    let measurement_variances: [f64; 6] = fixture_vector(&oracle["measurement_variance_bits"])
        .try_into()
        .unwrap();
    let trace_statistic = |rows: std::ops::Range<usize>| {
        let squared_norm = rows
            .clone()
            .map(|i| innovation[i] * innovation[i])
            .sum::<f64>();
        let trace = rows
            .map(|i| predicted_variances[i] + measurement_variances[i])
            .sum::<f64>();
        squared_norm / trace
    };
    assert_eq!(
        trace_statistic(0..6).to_bits(),
        fixture_bits(&oracle["full_trace_statistic_bits"])
    );
    assert_eq!(
        trace_statistic(3..6).to_bits(),
        fixture_bits(&oracle["velocity_block_statistic_bits"])
    );

    let adaptation = YangPredictionAdaptiveFactor {
        threshold: 1.0,
        outlier_gate_probability: 0.99,
    };
    let mut filter =
        loose_filter_with_diagonal(false, false, predicted_variances, Some(adaptation));
    let measurement = measurement_with_covariance(
        [WGS84_A_M + innovation[0], innovation[1], innovation[2]],
        [innovation[3], innovation[4], innovation[5]],
        measurement_variances,
    );
    filter.update_loose(&measurement).unwrap();
    assert_eq!(
        filter.state().covariance[3][3].to_bits(),
        fixture_bits(&oracle["posterior_velocity_variance_bits"])
    );
}

#[test]
fn yang_statistic_is_independent_of_igg_measurement_inflation() {
    let adaptation = YangPredictionAdaptiveFactor {
        threshold: 0.000674,
        outlier_gate_probability: 0.99999,
    };
    let diagonal = [1., 1., 1., 0.001, 0.001, 0.001];
    let measurement = measurement_with_covariance(
        [WGS84_A_M, 0., 0.],
        [0.45, 0., 0.],
        [99., 99., 99., 0.009, 0.009, 0.009],
    );
    let mut robust = loose_filter_with_diagonal(true, false, diagonal, None);
    let mut combined = loose_filter_with_diagonal(true, false, diagonal, Some(adaptation));
    robust.update_loose(&measurement).unwrap();
    combined.update_loose(&measurement).unwrap();
    assert_ne!(
        combined.state(),
        robust.state(),
        "oracle computes Yang adaptation from raw, unit-consistent covariance before IGG inflation"
    );
}
