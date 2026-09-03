use approx::assert_abs_diff_eq;
use nalgebra::{DMatrix, DVector};
use sidereon_core::astro::constants::earth::WGS84_A_M;
use sidereon_core::fusion::{
    covariance_is_positive_semidefinite, smooth_fusion_rts, ErrorStateLayout, FusionRtsEpoch,
    FusionRtsHistory, FusionRtsHistoryBuilder, GnssFixMeasurement, InertialFilter,
    InertialFilterConfig, InertialFilterSnapshot, InsFilterState, StationaryDetectorConfig,
    StationaryUpdateConfig, TightFilterSnapshot, ERROR_STATE_DIMENSION_15,
};
use sidereon_core::inertial::{
    simulate_imu_samples_from_increments, true_imu_increment_between, ImuSimulationOptions,
    ImuSpec, NavState,
};

const T0_J2000_S: f64 = 646_229_123.25;
const ANALYTIC_POSITION_BASE_M: f64 = 1_000.0;
const CLOCK_BIAS_VARIANCE_M2: f64 = 1.0e8;
const CLOCK_DRIFT_VARIANCE_M2_S2: f64 = 1.0e4;
const IDENTITY_3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[test]
fn scalar_history_matches_closed_form_fixed_interval_solution() {
    let p0 = DMatrix::from_row_slice(1, 1, &[2.0]);
    let phi = DMatrix::from_row_slice(1, 1, &[1.2]);
    let q = DMatrix::from_row_slice(1, 1, &[0.3]);
    let p1_pred = &phi * &p0 * phi.transpose() + q;
    let p1_filt = DMatrix::from_row_slice(1, 1, &[0.7]);
    let x0 = DVector::from_row_slice(&[10.0]);
    let x1_pred = DVector::from_row_slice(&[12.0]);
    let x1_filt = DVector::from_row_slice(&[11.5]);

    let history = two_epoch_history(&x0, &p0, &x1_pred, &p1_pred, &x1_filt, &p1_filt, &phi);
    let smoothed = smooth_fusion_rts(&history).expect("smooth");

    let gain = &p0 * phi.transpose() * p1_pred.clone().try_inverse().expect("inverse");
    let expected_x0 = &x0 + &gain * (&x1_filt - &x1_pred);
    let expected_p0 = &p0 + &gain * (&p1_filt - &p1_pred) * gain.transpose();
    let actual_x0 =
        smoothed.epochs[0].snapshot.state.nominal.position_ecef_m[0] - ANALYTIC_POSITION_BASE_M;

    assert_abs_diff_eq!(actual_x0, expected_x0[0], epsilon = 2.0e-14);
    assert_abs_diff_eq!(
        smoothed.epochs[0].covariance[0][0],
        expected_p0[(0, 0)],
        epsilon = 2.0e-14
    );
}

#[test]
fn multivariate_history_matches_closed_form_fixed_interval_solution() {
    let p0 = DMatrix::from_row_slice(2, 2, &[2.0, 0.3, 0.3, 1.1]);
    let phi = DMatrix::from_row_slice(2, 2, &[1.0, 0.5, 0.0, 1.0]);
    let q = DMatrix::from_row_slice(2, 2, &[0.2, 0.05, 0.05, 0.15]);
    let p1_pred = &phi * &p0 * phi.transpose() + q;
    let p1_filt = DMatrix::from_row_slice(2, 2, &[0.8, 0.1, 0.1, 0.6]);
    let x0 = DVector::from_row_slice(&[4.0, -0.5]);
    let x1_pred = DVector::from_row_slice(&[5.0, 0.3]);
    let x1_filt = DVector::from_row_slice(&[5.2, 0.25]);

    let history = two_epoch_history(&x0, &p0, &x1_pred, &p1_pred, &x1_filt, &p1_filt, &phi);
    let smoothed = smooth_fusion_rts(&history).expect("smooth");

    let gain = &p0 * phi.transpose() * p1_pred.clone().try_inverse().expect("inverse");
    let expected_x0 = &x0 + &gain * (&x1_filt - &x1_pred);
    let expected_p0 = &p0 + &gain * (&p1_filt - &p1_pred) * gain.transpose();
    let actual = [
        smoothed.epochs[0].snapshot.state.nominal.position_ecef_m[0] - ANALYTIC_POSITION_BASE_M,
        smoothed.epochs[0].snapshot.state.nominal.position_ecef_m[1],
    ];

    for axis in 0..2 {
        assert_abs_diff_eq!(actual[axis], expected_x0[axis], epsilon = 5.0e-14);
        for col in 0..2 {
            assert_abs_diff_eq!(
                smoothed.epochs[0].covariance[axis][col],
                expected_p0[(axis, col)],
                epsilon = 1.0e-13
            );
        }
    }
}

#[test]
fn smoothed_covariance_is_bounded_by_filtered_covariance() {
    let history = three_epoch_property_history();
    let smoothed = smooth_fusion_rts(&history).expect("smooth");

    for (idx, epoch) in history.epochs.iter().enumerate() {
        let difference = matrix_sub(
            &epoch.updated.tight.augmented_covariance,
            &smoothed.epochs[idx].covariance,
        );
        assert!(
            covariance_is_positive_semidefinite(&difference).expect("PSD check"),
            "epoch {idx} covariance ordering failed"
        );
    }

    let filtered_final = &history
        .epochs
        .last()
        .expect("history final")
        .updated
        .tight
        .augmented_covariance;
    let smoothed_final = &smoothed.epochs.last().expect("smoothed final").covariance;
    for row in 0..filtered_final.len() {
        for col in 0..filtered_final.len() {
            assert_eq!(
                smoothed_final[row][col].to_bits(),
                filtered_final[row][col].to_bits()
            );
        }
    }
}

#[test]
fn augmented_tight_clock_state_is_smoothed() {
    let base = DVector::from_row_slice(&[0.0]);
    let base_covariance = DMatrix::from_row_slice(1, 1, &[2.0]);
    let mut transition = identity_matrix(ERROR_STATE_DIMENSION_15 + 2);
    transition[ERROR_STATE_DIMENSION_15][ERROR_STATE_DIMENSION_15 + 1] = 0.0;
    let p0 = augmented_diagonal_covariance(2.0, 4.0, 1.0);
    let p1_pred = augmented_diagonal_covariance(2.0, 4.0, 1.0);
    let p1_filt = augmented_diagonal_covariance(2.0, 1.0, 1.0);

    let history = FusionRtsHistory::new(vec![
        FusionRtsEpoch::new(
            snapshot_with_augmented_covariance(0.0, &base, &base_covariance, 0.0, p0.clone()),
            snapshot_with_augmented_covariance(0.0, &base, &base_covariance, 0.0, p0),
            None,
        )
        .expect("epoch 0"),
        FusionRtsEpoch::new(
            snapshot_with_augmented_covariance(1.0, &base, &base_covariance, 0.0, p1_pred),
            snapshot_with_augmented_covariance(1.0, &base, &base_covariance, 2.0, p1_filt),
            Some(transition),
        )
        .expect("epoch 1"),
    ])
    .expect("history");

    let smoothed = smooth_fusion_rts(&history).expect("smooth");
    assert_abs_diff_eq!(
        smoothed.epochs[0].snapshot.tight.clock_bias_m,
        2.0,
        epsilon = 1.0e-14
    );
    assert_abs_diff_eq!(
        smoothed.epochs[0].covariance[ERROR_STATE_DIMENSION_15][ERROR_STATE_DIMENSION_15],
        1.0,
        epsilon = 1.0e-14
    );
}

#[test]
fn invalid_tight_snapshot_is_rejected_without_panic() {
    let x = DVector::from_row_slice(&[0.0]);
    let p = DMatrix::from_row_slice(1, 1, &[1.0]);
    let mut bad = snapshot_from_vector(0.0, &x, &p);
    bad.tight.augmented_covariance.clear();

    assert!(FusionRtsEpoch::new(bad.clone(), bad, None).is_err());
}

#[test]
fn tight_base_covariance_mismatch_is_rejected() {
    let x = DVector::from_row_slice(&[0.0]);
    let p = DMatrix::from_row_slice(1, 1, &[1.0]);
    let mut bad = snapshot_from_vector(0.0, &x, &p);
    bad.tight.augmented_covariance[0][0] = 2.0;

    assert!(FusionRtsEpoch::new(bad.clone(), bad, None).is_err());
}

#[test]
fn recorded_update_failure_does_not_mutate_filter_or_history() {
    let truth = scenario_truth(1, 1.0);
    let spec = ImuSpec::datasheet(0.015, 0.001, 0.001, 1.0e-4, 300.0, 300.0, None, None);
    let mut filter = scenario_filter(truth[0], spec);
    let before = filter.snapshot();
    let mut history = FusionRtsHistoryBuilder::from_filter(&filter).expect("history");
    let measurement = scenario_measurement(truth[1], 4.0, [0.0; 3]);

    assert!(filter
        .update_loose_recorded(&measurement, &mut history)
        .is_err());
    assert_eq!(filter.snapshot(), before);
    assert_eq!(history.clone().finish().expect("history").epochs.len(), 1);
}

#[test]
fn recorded_stationary_update_and_loose_update_can_share_epoch() {
    let start =
        NavState::new(T0_J2000_S, [WGS84_A_M, 0.0, 0.0], [0.0; 3], IDENTITY_3).expect("start");
    let end = NavState::new(
        T0_J2000_S + 1.0,
        [WGS84_A_M, 0.0, 0.0],
        [0.0; 3],
        IDENTITY_3,
    )
    .expect("end");
    let spec = ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 300.0, 300.0, None, None);
    let sequence = simulate_imu_samples_from_increments(
        &[true_imu_increment_between(&start, &end).expect("truth increment")],
        spec,
        ImuSimulationOptions::default(),
    )
    .expect("simulated IMU");
    let mut diagonal = vec![0.01; ERROR_STATE_DIMENSION_15];
    diagonal[3] = 1.0;
    diagonal[4] = 1.0;
    diagonal[5] = 1.0;
    let state =
        InsFilterState::from_diagonal(start, ErrorStateLayout::Fifteen, &diagonal).expect("state");
    let mut config = InertialFilterConfig::new(spec).expect("filter config");
    let detector = StationaryDetectorConfig::new(1, 1.0e-6, 1.0e-6);
    config.loose.stationary_updates = Some(StationaryUpdateConfig::new(detector, 0.01, 1.0e-4));
    let mut filter = InertialFilter::with_config(state, config).expect("filter");
    let mut history = FusionRtsHistoryBuilder::from_filter(&filter).expect("history");

    filter
        .propagate_recorded(sequence.samples[0], &mut history)
        .expect("propagate");
    let stationary = filter
        .update_stationary_recorded(&mut history)
        .expect("stationary update");
    assert!(stationary.is_some());
    let measurement = GnssFixMeasurement::position(
        end.t_j2000_s,
        end.position_ecef_m,
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        8,
    )
    .expect("measurement");
    filter
        .update_loose_recorded(&measurement, &mut history)
        .expect("loose update");

    let history = history.finish().expect("complete history");
    assert_eq!(history.epochs.len(), 3);
    assert_eq!(
        history.epochs[1].t_j2000_s.to_bits(),
        history.epochs[2].t_j2000_s.to_bits()
    );
    let smoothed = smooth_fusion_rts(&history).expect("smooth");
    assert_eq!(smoothed.epochs.len(), history.epochs.len());
}

#[test]
fn recorded_scenario_arc_smoothing_reduces_3d_rms() {
    const STEPS: usize = 8;
    const DT_S: f64 = 1.0;
    const POSITION_SIGMA_M: f64 = 4.0;

    let truth = scenario_truth(STEPS, DT_S);
    let increments = truth
        .windows(2)
        .map(|pair| true_imu_increment_between(&pair[0], &pair[1]).expect("truth increment"))
        .collect::<Vec<_>>();
    let spec = ImuSpec::datasheet(0.015, 0.001, 0.001, 1.0e-4, 300.0, 300.0, None, None);
    let sequence = simulate_imu_samples_from_increments(&increments, spec, {
        let mut options = ImuSimulationOptions::default();
        options.seed = 0x51d3_7e0f_29a4_d61b;
        options
    })
    .expect("simulated IMU");

    let mut filter = scenario_filter(truth[0], spec);
    let mut history = FusionRtsHistoryBuilder::from_filter(&filter).expect("history");
    for (idx, sample) in sequence.samples.into_iter().enumerate() {
        filter
            .propagate_recorded(sample, &mut history)
            .expect("propagate");
        let measurement = scenario_measurement(
            truth[idx + 1],
            POSITION_SIGMA_M,
            SCENARIO_POSITION_NOISE[idx],
        );
        filter
            .update_loose_recorded(&measurement, &mut history)
            .expect("update");
    }

    let history = history.finish().expect("complete history");
    let smoothed = smooth_fusion_rts(&history).expect("smooth");
    let filtered_rms = rms_position_error(
        history
            .epochs
            .iter()
            .zip(truth.iter())
            .map(|(epoch, truth)| (epoch.updated.state.nominal.position_ecef_m, *truth)),
    );
    let smoothed_rms = rms_position_error(
        smoothed
            .epochs
            .iter()
            .zip(truth.iter())
            .map(|(epoch, truth)| (epoch.snapshot.state.nominal.position_ecef_m, *truth)),
    );

    assert!(
        smoothed_rms < filtered_rms,
        "smoothed {smoothed_rms:.17e} filtered {filtered_rms:.17e}"
    );
    assert_abs_diff_eq!(filtered_rms, 3.191_344_235_098_076_5, epsilon = 2.0e-12);
    assert_abs_diff_eq!(smoothed_rms, 0.849_374_281_703_253_5, epsilon = 2.0e-12);
}

const SCENARIO_POSITION_NOISE: [[f64; 3]; 8] = [
    [2.4, -1.1, 0.8],
    [1.7, 1.2, -0.6],
    [-0.9, 1.9, 0.4],
    [-2.1, 0.6, 1.1],
    [0.3, -2.2, -0.9],
    [1.2, -0.7, 0.6],
    [-1.6, 0.4, -1.2],
    [0.9, 1.5, 0.3],
];

fn two_epoch_history(
    x0: &DVector<f64>,
    p0: &DMatrix<f64>,
    x1_pred: &DVector<f64>,
    p1_pred: &DMatrix<f64>,
    x1_filt: &DVector<f64>,
    p1_filt: &DMatrix<f64>,
    phi: &DMatrix<f64>,
) -> FusionRtsHistory {
    FusionRtsHistory::new(vec![
        FusionRtsEpoch::new(
            snapshot_from_vector(0.0, x0, p0),
            snapshot_from_vector(0.0, x0, p0),
            None,
        )
        .expect("epoch 0"),
        FusionRtsEpoch::new(
            snapshot_from_vector(1.0, x1_pred, p1_pred),
            snapshot_from_vector(1.0, x1_filt, p1_filt),
            Some(transition_from_matrix(phi)),
        )
        .expect("epoch 1"),
    ])
    .expect("history")
}

fn three_epoch_property_history() -> FusionRtsHistory {
    let x0 = DVector::from_row_slice(&[1.0, 0.2]);
    let p0 = DMatrix::from_row_slice(2, 2, &[1.4, 0.2, 0.2, 0.9]);
    let phi1 = DMatrix::from_row_slice(2, 2, &[1.0, 0.4, 0.0, 1.0]);
    let q1 = DMatrix::from_row_slice(2, 2, &[0.12, 0.03, 0.03, 0.10]);
    let p1_pred = &phi1 * &p0 * phi1.transpose() + q1;
    let p1_filt = DMatrix::from_row_slice(2, 2, &[0.75, 0.12, 0.12, 0.62]);
    let x1_pred = DVector::from_row_slice(&[1.3, 0.25]);
    let x1_filt = DVector::from_row_slice(&[1.05, 0.4]);

    let phi2 = DMatrix::from_row_slice(2, 2, &[1.0, 0.6, 0.0, 1.0]);
    let q2 = DMatrix::from_row_slice(2, 2, &[0.18, 0.04, 0.04, 0.12]);
    let p2_pred = &phi2 * &p1_filt * phi2.transpose() + q2;
    let p2_filt = DMatrix::from_row_slice(2, 2, &[0.5, 0.08, 0.08, 0.42]);
    let x2_pred = DVector::from_row_slice(&[1.7, 0.35]);
    let x2_filt = DVector::from_row_slice(&[1.55, 0.32]);

    FusionRtsHistory::new(vec![
        FusionRtsEpoch::new(
            snapshot_from_vector(0.0, &x0, &p0),
            snapshot_from_vector(0.0, &x0, &p0),
            None,
        )
        .expect("epoch 0"),
        FusionRtsEpoch::new(
            snapshot_from_vector(1.0, &x1_pred, &p1_pred),
            snapshot_from_vector(1.0, &x1_filt, &p1_filt),
            Some(transition_from_matrix(&phi1)),
        )
        .expect("epoch 1"),
        FusionRtsEpoch::new(
            snapshot_from_vector(2.0, &x2_pred, &p2_pred),
            snapshot_from_vector(2.0, &x2_filt, &p2_filt),
            Some(transition_from_matrix(&phi2)),
        )
        .expect("epoch 2"),
    ])
    .expect("history")
}

fn snapshot_from_vector(
    t_j2000_s: f64,
    x: &DVector<f64>,
    p: &DMatrix<f64>,
) -> InertialFilterSnapshot {
    let mut position = [ANALYTIC_POSITION_BASE_M, 0.0, 0.0];
    for idx in 0..x.len().min(3) {
        position[idx] += x[idx];
    }
    let nominal = NavState::new(t_j2000_s, position, [0.0; 3], IDENTITY_3).expect("nominal");
    let covariance = covariance_from_matrix(p);
    let state =
        InsFilterState::new(nominal, ErrorStateLayout::Fifteen, covariance.clone()).expect("state");
    InertialFilterSnapshot {
        state,
        last_body_rate_wrt_ecef_rps: [0.0; 3],
        stationarity_window: Vec::new(),
        last_stationary_update_t_j2000_s: None,
        last_non_holonomic_update_t_j2000_s: None,
        tight: TightFilterSnapshot {
            clock_bias_m: 0.0,
            clock_drift_m_s: 0.0,
            augmented_covariance: augmented_covariance(&covariance),
        },
    }
}

fn covariance_from_matrix(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut covariance = vec![vec![0.0; ERROR_STATE_DIMENSION_15]; ERROR_STATE_DIMENSION_15];
    for (idx, row) in covariance.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            covariance[row][col] = matrix[(row, col)];
        }
    }
    covariance
}

fn transition_from_matrix(matrix: &DMatrix<f64>) -> Vec<Vec<f64>> {
    let mut transition = vec![vec![0.0; ERROR_STATE_DIMENSION_15]; ERROR_STATE_DIMENSION_15];
    for (idx, row) in transition.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            transition[row][col] = matrix[(row, col)];
        }
    }
    transition
}

fn identity_matrix(dimension: usize) -> Vec<Vec<f64>> {
    let mut matrix = vec![vec![0.0; dimension]; dimension];
    for (idx, row) in matrix.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    matrix
}

fn augmented_diagonal_covariance(
    position_variance: f64,
    clock_bias_variance: f64,
    clock_drift_variance: f64,
) -> Vec<Vec<f64>> {
    let mut covariance = identity_matrix(ERROR_STATE_DIMENSION_15 + 2);
    covariance[0][0] = position_variance;
    covariance[ERROR_STATE_DIMENSION_15][ERROR_STATE_DIMENSION_15] = clock_bias_variance;
    covariance[ERROR_STATE_DIMENSION_15 + 1][ERROR_STATE_DIMENSION_15 + 1] = clock_drift_variance;
    covariance
}

fn snapshot_with_augmented_covariance(
    t_j2000_s: f64,
    x: &DVector<f64>,
    p: &DMatrix<f64>,
    clock_bias_m: f64,
    augmented_covariance: Vec<Vec<f64>>,
) -> InertialFilterSnapshot {
    let mut snapshot = snapshot_from_vector(t_j2000_s, x, p);
    snapshot.tight.clock_bias_m = clock_bias_m;
    snapshot.tight.augmented_covariance = augmented_covariance;
    snapshot
}

fn augmented_covariance(base: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let base_dim = base.len();
    let mut covariance = vec![vec![0.0; base_dim + 2]; base_dim + 2];
    for row in 0..base_dim {
        covariance[row][..base_dim].copy_from_slice(&base[row][..base_dim]);
    }
    covariance[base_dim][base_dim] = CLOCK_BIAS_VARIANCE_M2;
    covariance[base_dim + 1][base_dim + 1] = CLOCK_DRIFT_VARIANCE_M2_S2;
    covariance
}

fn matrix_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b.iter())
        .map(|(row_a, row_b)| {
            row_a
                .iter()
                .zip(row_b.iter())
                .map(|(left, right)| left - right)
                .collect()
        })
        .collect()
}

fn scenario_truth(steps: usize, dt_s: f64) -> Vec<NavState> {
    let mut states = Vec::with_capacity(steps + 1);
    let mut position = [WGS84_A_M + 15.0, -20.0, 8.0];
    let mut velocity = [1.4, -0.55, 0.35];
    let acceleration = [0.08, 0.035, -0.025];
    states.push(NavState::new(T0_J2000_S, position, velocity, IDENTITY_3).expect("truth initial"));
    for step in 1..=steps {
        let next_velocity = [
            velocity[0] + acceleration[0] * dt_s,
            velocity[1] + acceleration[1] * dt_s,
            velocity[2] + acceleration[2] * dt_s,
        ];
        for axis in 0..3 {
            position[axis] += 0.5 * (velocity[axis] + next_velocity[axis]) * dt_s;
        }
        velocity = next_velocity;
        states.push(
            NavState::new(
                T0_J2000_S + step as f64 * dt_s,
                position,
                velocity,
                IDENTITY_3,
            )
            .expect("truth"),
        );
    }
    states
}

fn scenario_filter(initial_truth: NavState, spec: ImuSpec) -> InertialFilter {
    let initial_nominal = NavState::new(
        initial_truth.t_j2000_s,
        [
            initial_truth.position_ecef_m[0] + 6.0,
            initial_truth.position_ecef_m[1] - 4.0,
            initial_truth.position_ecef_m[2] + 2.5,
        ],
        [
            initial_truth.velocity_ecef_mps[0] + 0.25,
            initial_truth.velocity_ecef_mps[1] - 0.15,
            initial_truth.velocity_ecef_mps[2] + 0.08,
        ],
        initial_truth.attitude_body_to_ecef,
    )
    .expect("initial nominal");
    let mut diagonal = vec![0.01; ERROR_STATE_DIMENSION_15];
    diagonal[0] = 64.0;
    diagonal[1] = 64.0;
    diagonal[2] = 64.0;
    diagonal[3] = 1.0;
    diagonal[4] = 1.0;
    diagonal[5] = 1.0;
    let state =
        InsFilterState::from_diagonal(initial_nominal, ErrorStateLayout::Fifteen, &diagonal)
            .expect("initial state");
    InertialFilter::new(state, spec).expect("filter")
}

fn scenario_measurement(truth: NavState, sigma_m: f64, noise: [f64; 3]) -> GnssFixMeasurement {
    let measured = [
        truth.position_ecef_m[0] + noise[0],
        truth.position_ecef_m[1] + noise[1],
        truth.position_ecef_m[2] + noise[2],
    ];
    GnssFixMeasurement::position(
        truth.t_j2000_s,
        measured,
        [
            [sigma_m * sigma_m, 0.0, 0.0],
            [0.0, sigma_m * sigma_m, 0.0],
            [0.0, 0.0, sigma_m * sigma_m],
        ],
        8,
    )
    .expect("measurement")
}

fn rms_position_error<I>(samples: I) -> f64
where
    I: Iterator<Item = ([f64; 3], NavState)>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for (position, truth) in samples {
        let mut squared = 0.0;
        for (estimated, truth) in position.iter().zip(truth.position_ecef_m.iter()) {
            let error = *estimated - *truth;
            squared += error * error;
        }
        sum += squared;
        count += 1;
    }
    (sum / count as f64).sqrt()
}
