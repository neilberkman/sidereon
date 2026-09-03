//! Monte-Carlo GNSS/INS fusion consistency campaign.
//!
//! Provenance: ensemble NEES and NIS acceptance bands follow Bar-Shalom,
//! Li, and Kirubarajan, Estimation with Applications to Tracking and
//! Navigation, 2001, using the two-sided chi-square consistency test at 95%.
//! Truth-driven IMU samples use the public simulator in this crate, whose
//! stochastic model follows Groves, Principles of GNSS, Inertial, and
//! Multisensor Integrated Navigation Systems, 2nd ed.

use nalgebra::{DMatrix, DVector};
use sidereon_core::astro::constants::earth::WGS84_A_M;
use sidereon_core::astro::math::mat3::mul_vec3;
use sidereon_core::astro::math::vec3::{add3, cross3, norm3, scale3};
use sidereon_core::constants::C_M_S;
use sidereon_core::fusion::{
    ErrorStateLayout, FusionUpdate, GnssFixMeasurement, IggIiiMeasurementReweighting,
    InertialFilter, InertialFilterConfig, InsFilterState, LooseCouplingConfig, TightCouplingConfig,
    TightGnssEpoch, TightGnssObservation, TightRangeRateObservation, ERROR_ACCEL_BIAS_INDEX,
    ERROR_ATTITUDE_INDEX, ERROR_GYRO_BIAS_INDEX, ERROR_POSITION_INDEX, ERROR_STATE_DIMENSION_15,
    ERROR_VELOCITY_INDEX,
};
use sidereon_core::inertial::{
    mechanize_ecef, simulate_imu_samples_from_increments, true_imu_increment_between,
    CorrectedImuIncrement, ImuBias, ImuGrade, ImuSimulationOptions, ImuSpec, MechanizationConfig,
    NavState,
};
use sidereon_core::observables::{
    transmit_time_satellite_state, ObservableEphemerisSource, ObservableState, ObservablesError,
    TransmitTimeOptions,
};
use sidereon_core::precise_positioning::{
    predict_range_rate_m_s, ReceiverVelocityState, VelocityObservation,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const RUNS_PER_SCENARIO: usize = 200;
const ALPHA: f64 = 0.05;
const STEP_COUNT: usize = 2;
const DT_S: f64 = 0.5;
const T0_J2000_S: f64 = 646_229_123.25;
const POSITION_SIGMA_M: f64 = 4.75;
const VELOCITY_SIGMA_MPS: f64 = 0.21;
const CODE_SIGMA_M: f64 = 5.2;
const RANGE_RATE_SIGMA_MPS: f64 = 0.24;
const MISMATCH_NOISE_SCALE: f64 = 4.0;
const TURNTABLE_STEP_COUNT: usize = 10;
const TURNTABLE_DT_S: f64 = 0.6;
const TURNTABLE_YAW_RATE_RPS: f64 = 0.35;
const TURNTABLE_LEVER_ARM_BODY_M: [f64; 3] = [5.0, -2.0, 1.0];
const TURNTABLE_CODE_SIGMA_M: f64 = 12.0;
const TURNTABLE_RANGE_RATE_SIGMA_MPS: f64 = 0.10;
const TURNTABLE_GYRO_BIAS_NOMINAL_RPS: [f64; 3] = [0.012, -0.008, 0.018];
const TURNTABLE_GYRO_BIAS_ERROR_RPS: [f64; 3] = [0.006, -0.004, 0.003];
const TURNTABLE_GYRO_BIAS_ERROR_SIGMA_RPS: f64 = 0.0015;
const TURNTABLE_SEED_TAG: u64 = 0x7e12_a11e_1e4e_a2b5;
const TURNTABLE_NEES_DIMENSION: usize = 6;
const LOOSE_OUTLIER_BUDGET_STRIDE: usize = 10;
const LOOSE_OUTLIER_MAGNITUDE_M: f64 = 120.0;

#[derive(Debug, Clone, Copy)]
enum FilterKind {
    Loose,
    Tight,
}

impl FilterKind {
    const ALL: [Self; 2] = [Self::Loose, Self::Tight];

    const fn label(self) -> &'static str {
        match self {
            Self::Loose => "loose",
            Self::Tight => "tight",
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum MotionScenario {
    Static,
    ConstantVelocity,
    ConstantAcceleration,
    AggressiveDynamics,
}

impl MotionScenario {
    const ALL: [Self; 4] = [
        Self::Static,
        Self::ConstantVelocity,
        Self::ConstantAcceleration,
        Self::AggressiveDynamics,
    ];

    const fn label(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::ConstantVelocity => "constant-velocity",
            Self::ConstantAcceleration => "constant-acceleration",
            Self::AggressiveDynamics => "aggressive-dynamics",
        }
    }

    const fn seed_tag(self) -> u64 {
        match self {
            Self::Static => 0x51a7_1c00_0000_0001,
            Self::ConstantVelocity => 0xc057_71e0_0000_0002,
            Self::ConstantAcceleration => 0xacce_1e00_0000_0003,
            Self::AggressiveDynamics => 0xa66e_5510_0000_0004,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ImuProfile {
    label: &'static str,
    seed_tag: u64,
    spec: ImuSpec,
}

impl ImuProfile {
    fn all() -> [Self; 2] {
        [
            Self {
                label: "consumer",
                seed_tag: 0xc0a5_51f1_ed00_0055,
                spec: ImuSpec::preset(ImuGrade::Mems),
            },
            Self {
                label: "tactical",
                seed_tag: 0x7ac7_1ca1_0000_0099,
                spec: ImuSpec::preset(ImuGrade::Tactical),
            },
        ]
    }
}

#[derive(Debug, Clone, Copy)]
struct EnsembleResult {
    nees_average: f64,
    nis_average: f64,
    nees_lower: f64,
    nees_upper: f64,
    nis_lower: f64,
    nis_upper: f64,
    nees_dof: usize,
    nis_dof: usize,
}

#[test]
fn matched_noise_campaign_keeps_loose_and_tight_filters_inside_chi_square_bands() {
    let mut failures = Vec::new();
    for filter_kind in FilterKind::ALL {
        for profile in ImuProfile::all() {
            for scenario in MotionScenario::ALL {
                let result = run_ensemble(filter_kind, profile, scenario, 1.0);
                if !(result.nees_lower..=result.nees_upper).contains(&result.nees_average) {
                    failures.push(format!(
                        "{} {} {} NEES {:.17e} outside [{:.17e}, {:.17e}] at dof {}",
                        filter_kind.label(),
                        profile.label,
                        scenario.label(),
                        result.nees_average,
                        result.nees_lower,
                        result.nees_upper,
                        result.nees_dof
                    ));
                }
                if !(result.nis_lower..=result.nis_upper).contains(&result.nis_average) {
                    failures.push(format!(
                        "{} {} {} NIS {:.17e} outside [{:.17e}, {:.17e}] at dof {}",
                        filter_kind.label(),
                        profile.label,
                        scenario.label(),
                        result.nis_average,
                        result.nis_lower,
                        result.nis_upper,
                        result.nis_dof
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn mismatched_noise_campaign_drives_nees_outside_chi_square_band() {
    for filter_kind in FilterKind::ALL {
        let result = run_ensemble(
            filter_kind,
            ImuProfile::all()[0],
            MotionScenario::AggressiveDynamics,
            MISMATCH_NOISE_SCALE,
        );
        assert!(
            result.nees_average > result.nees_upper,
            "{} mismatched NEES {:.17e} did not exceed upper band {:.17e}",
            filter_kind.label(),
            result.nees_average,
            result.nees_upper
        );
    }
}

#[test]
fn loose_igg_iii_outlier_budget_keeps_nees_in_band_where_plain_exits() {
    let robust = run_loose_outlier_budget_ensemble(true);
    assert!(
        (robust.nees_lower..=robust.nees_upper).contains(&robust.nees_average),
        "robust loose outlier-budget NEES {:.17e} outside [{:.17e}, {:.17e}] at dof {}",
        robust.nees_average,
        robust.nees_lower,
        robust.nees_upper,
        robust.nees_dof
    );

    let plain = run_loose_outlier_budget_ensemble(false);
    assert!(
        plain.nees_average > plain.nees_upper,
        "plain loose outlier-budget NEES {:.17e} did not exceed upper band {:.17e}",
        plain.nees_average,
        plain.nees_upper
    );
}

#[test]
fn turntable_lever_arm_tight_campaign_keeps_nees_inside_chi_square_band() {
    // Development check: flipping the range-rate gyro-bias lever-arm row in
    // fusion/tight.rs from `-los * gyro_bias_velocity_block` to `+los * ...`
    // drove this scenario's NEES above its chi-square band.
    let result = run_turntable_lever_arm_ensemble();
    assert!(
        (result.nees_lower..=result.nees_upper).contains(&result.nees_average),
        "tight turntable lever-arm NEES {:.17e} outside [{:.17e}, {:.17e}] at dof {}",
        result.nees_average,
        result.nees_lower,
        result.nees_upper,
        result.nees_dof
    );
    assert!(
        (result.nis_lower..=result.nis_upper).contains(&result.nis_average),
        "tight turntable lever-arm NIS {:.17e} outside [{:.17e}, {:.17e}] at dof {}",
        result.nis_average,
        result.nis_lower,
        result.nis_upper,
        result.nis_dof
    );
}

fn run_ensemble(
    filter_kind: FilterKind,
    profile: ImuProfile,
    scenario: MotionScenario,
    actual_measurement_noise_scale: f64,
) -> EnsembleResult {
    let trajectory = truth_trajectory(scenario);
    let increments = truth_increments(&trajectory);
    let source = LinearSource::from_receiver(trajectory.last().expect("truth").position_ecef_m);
    let mut nees_sum = 0.0;
    let mut nis_sum = 0.0;
    let mut nis_dof = 0usize;

    for run in 0..RUNS_PER_SCENARIO {
        let seed = ensemble_seed(filter_kind, profile, scenario, run);
        let mut rng = SplitMix64::new(seed);
        let initial = trajectory[0];
        let truth = *trajectory.last().expect("final truth");
        let nominal = initial_nominal(initial, &mut rng);
        let mut filter = make_filter(nominal, profile.spec);
        let sequence = simulate_imu_samples_from_increments(&increments, profile.spec, {
            let mut options = ImuSimulationOptions::default();
            options.seed = seed ^ 0x6d5f_c2a4_1c37_9e21;
            options
        })
        .expect("IMU sequence");

        for sample in sequence.samples {
            filter.propagate(sample).expect("propagate");
        }

        let update = match filter_kind {
            FilterKind::Loose => {
                let measurement =
                    loose_measurement(&truth, &mut rng, actual_measurement_noise_scale);
                filter.update_loose(&measurement).expect("loose update")
            }
            FilterKind::Tight => {
                let epoch = tight_epoch(&source, &truth, &mut rng, actual_measurement_noise_scale);
                filter.update_tight(&source, &epoch).expect("tight update")
            }
        };

        nis_sum += update.nis;
        nis_dof += update.rows;
        nees_sum += nees_position_velocity(&filter, &truth);
        assert_update_shape(update, filter_kind);
    }

    let nees_dof = RUNS_PER_SCENARIO * 6;
    let (nees_lower, nees_upper) = chi_square_average_band(nees_dof);
    let (nis_lower, nis_upper) = chi_square_average_band(nis_dof);
    EnsembleResult {
        nees_average: nees_sum / RUNS_PER_SCENARIO as f64,
        nis_average: nis_sum / RUNS_PER_SCENARIO as f64,
        nees_lower,
        nees_upper,
        nis_lower,
        nis_upper,
        nees_dof,
        nis_dof,
    }
}

fn run_turntable_lever_arm_ensemble() -> EnsembleResult {
    let trajectory = turntable_truth_trajectory();
    let increments = truth_increments(&trajectory);
    let truth = *trajectory.last().expect("final truth");
    let body_rate_wrt_ecef_rps = turntable_body_rate_wrt_ecef_rps();
    let final_antenna =
        antenna_truth_kinematics(&truth, TURNTABLE_LEVER_ARM_BODY_M, body_rate_wrt_ecef_rps);
    let source = LinearSource::from_receiver(final_antenna.position_ecef_m);
    let spec = turntable_imu_spec();
    let covariance_diagonal = turntable_initial_covariance_diagonal();
    let mut nees_sum = 0.0;
    let mut nis_sum = 0.0;
    let mut nis_dof = 0usize;

    for run in 0..RUNS_PER_SCENARIO {
        let seed = turntable_seed(run);
        let mut rng = SplitMix64::new(seed);
        let initial = trajectory[0];
        let nominal = initial_nominal(initial, &mut rng)
            .with_biases([0.0; 3], TURNTABLE_GYRO_BIAS_NOMINAL_RPS)
            .expect("nominal bias");
        let true_initial_gyro_bias = add3(
            TURNTABLE_GYRO_BIAS_NOMINAL_RPS,
            TURNTABLE_GYRO_BIAS_ERROR_RPS,
        );
        let mut filter = make_filter_with_tight_config(
            nominal,
            spec,
            TURNTABLE_LEVER_ARM_BODY_M,
            &covariance_diagonal,
        );
        let sequence = simulate_imu_samples_from_increments(&increments, spec, {
            let mut options = ImuSimulationOptions::default();
            options.seed = seed ^ 0xa95c_7d63_0f5a_19b3;
            options.initial_bias = ImuBias {
                accel_mps2: [0.0; 3],
                gyro_rps: true_initial_gyro_bias,
            };
            options
        })
        .expect("IMU sequence");

        for (idx, (sample, truth_step)) in sequence
            .samples
            .into_iter()
            .zip(trajectory.iter().skip(1))
            .enumerate()
        {
            filter.propagate(sample).expect("propagate");
            if idx + 1 != TURNTABLE_STEP_COUNT {
                continue;
            }
            let epoch = tight_epoch_with_kinematics(
                &source,
                truth_step,
                TURNTABLE_LEVER_ARM_BODY_M,
                body_rate_wrt_ecef_rps,
                TURNTABLE_CODE_SIGMA_M,
                TURNTABLE_RANGE_RATE_SIGMA_MPS,
                &mut rng,
                1.0,
            );
            let update = filter.update_tight(&source, &epoch).expect("tight update");
            nis_sum += update.nis;
            nis_dof += update.rows;
            assert_update_shape(update, FilterKind::Tight);
        }

        nees_sum += nees_position_velocity(&filter, &truth);
    }

    let nees_dof = RUNS_PER_SCENARIO * TURNTABLE_NEES_DIMENSION;
    let (nees_lower, nees_upper) = chi_square_average_band(nees_dof);
    let (nis_lower, nis_upper) = chi_square_average_band(nis_dof);
    EnsembleResult {
        nees_average: nees_sum / RUNS_PER_SCENARIO as f64,
        nis_average: nis_sum / RUNS_PER_SCENARIO as f64,
        nees_lower,
        nees_upper,
        nis_lower,
        nis_upper,
        nees_dof,
        nis_dof,
    }
}

fn run_loose_outlier_budget_ensemble(robust: bool) -> EnsembleResult {
    let profile = ImuProfile::all()[0];
    let scenario = MotionScenario::AggressiveDynamics;
    let trajectory = truth_trajectory(scenario);
    let increments = truth_increments(&trajectory);
    let truth = *trajectory.last().expect("final truth");
    let loose = if robust {
        let mut config = LooseCouplingConfig::default();
        config.measurement_reweighting = Some(IggIiiMeasurementReweighting::standard());
        config
    } else {
        LooseCouplingConfig::default()
    };
    let mut nees_sum = 0.0;
    let mut nis_sum = 0.0;
    let mut nis_dof = 0usize;

    for run in 0..RUNS_PER_SCENARIO {
        let seed = ensemble_seed(FilterKind::Loose, profile, scenario, run);
        let mut rng = SplitMix64::new(seed);
        let initial = trajectory[0];
        let nominal = initial_nominal(initial, &mut rng);
        let mut filter = make_filter_with_loose_config(nominal, profile.spec, loose);
        let sequence = simulate_imu_samples_from_increments(&increments, profile.spec, {
            let mut options = ImuSimulationOptions::default();
            options.seed = seed ^ 0x6d5f_c2a4_1c37_9e21;
            options
        })
        .expect("IMU sequence");

        for sample in sequence.samples {
            filter.propagate(sample).expect("propagate");
        }

        let mut measurement = loose_measurement(&truth, &mut rng, 1.0);
        if run % LOOSE_OUTLIER_BUDGET_STRIDE == 0 {
            measurement.position_ecef_m[0] += LOOSE_OUTLIER_MAGNITUDE_M;
        }
        let update = filter.update_loose(&measurement).expect("loose update");
        assert!(update.applied);
        assert_eq!(update.rows, 6);
        assert_eq!(update.accepted_rows, update.rows);
        assert_eq!(update.rejected_rows, 0);
        nis_sum += update.nis;
        nis_dof += update.rows;
        nees_sum += nees_position_velocity(&filter, &truth);
    }

    let nees_dof = RUNS_PER_SCENARIO * 6;
    let (nees_lower, nees_upper) = chi_square_average_band(nees_dof);
    let (nis_lower, nis_upper) = chi_square_average_band(nis_dof);
    EnsembleResult {
        nees_average: nees_sum / RUNS_PER_SCENARIO as f64,
        nis_average: nis_sum / RUNS_PER_SCENARIO as f64,
        nees_lower,
        nees_upper,
        nis_lower,
        nis_upper,
        nees_dof,
        nis_dof,
    }
}

fn make_filter(nominal: NavState, spec: ImuSpec) -> InertialFilter {
    let covariance_diagonal = initial_covariance_diagonal();
    make_filter_with_tight_config(nominal, spec, [0.0; 3], &covariance_diagonal)
}

fn make_filter_with_loose_config(
    nominal: NavState,
    spec: ImuSpec,
    loose: LooseCouplingConfig,
) -> InertialFilter {
    let covariance_diagonal = initial_covariance_diagonal();
    let state =
        InsFilterState::from_diagonal(nominal, ErrorStateLayout::Fifteen, &covariance_diagonal)
            .expect("filter state");
    let mut config = InertialFilterConfig::new(spec).expect("filter config");
    config.loose = loose;
    InertialFilter::with_config(state, config).expect("filter")
}

fn make_filter_with_tight_config(
    nominal: NavState,
    spec: ImuSpec,
    lever_arm_body_m: [f64; 3],
    covariance_diagonal: &[f64; ERROR_STATE_DIMENSION_15],
) -> InertialFilter {
    let state =
        InsFilterState::from_diagonal(nominal, ErrorStateLayout::Fifteen, covariance_diagonal)
            .expect("filter state");
    let mut config = InertialFilterConfig::new(spec).expect("filter config");
    let mut tight = TightCouplingConfig::default();
    tight.lever_arm_body_m = lever_arm_body_m;
    tight.light_time = false;
    tight.sagnac = false;
    tight.initial_clock_bias_variance_m2 = 1.0e-6;
    tight.initial_clock_drift_variance_m2_s2 = 1.0e-6;
    tight.clock_bias_random_walk_m2_s = 0.0;
    tight.clock_drift_random_walk_m2_s3 = 0.0;
    tight.update_options = Default::default();
    config.tight = tight;
    InertialFilter::with_config(state, config).expect("filter")
}

fn initial_covariance_diagonal() -> [f64; ERROR_STATE_DIMENSION_15] {
    let mut diagonal = [0.0; ERROR_STATE_DIMENSION_15];
    for axis in 0..3 {
        diagonal[ERROR_POSITION_INDEX + axis] = 36.0;
        diagonal[ERROR_VELOCITY_INDEX + axis] = 0.16;
        diagonal[ERROR_ATTITUDE_INDEX + axis] = 0.0;
        diagonal[ERROR_ACCEL_BIAS_INDEX + axis] = 0.0;
        diagonal[ERROR_GYRO_BIAS_INDEX + axis] = 0.0;
    }
    diagonal
}

fn turntable_initial_covariance_diagonal() -> [f64; ERROR_STATE_DIMENSION_15] {
    let mut diagonal = initial_covariance_diagonal();
    for axis in 0..3 {
        diagonal[ERROR_ATTITUDE_INDEX + axis] = 1.0e-10;
        diagonal[ERROR_ACCEL_BIAS_INDEX + axis] = 1.0e-12;
        diagonal[ERROR_GYRO_BIAS_INDEX + axis] =
            TURNTABLE_GYRO_BIAS_ERROR_SIGMA_RPS * TURNTABLE_GYRO_BIAS_ERROR_SIGMA_RPS;
    }
    diagonal
}

fn turntable_imu_spec() -> ImuSpec {
    ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, f64::INFINITY, f64::INFINITY, None, None)
}

fn initial_nominal(truth: NavState, rng: &mut SplitMix64) -> NavState {
    let mut position = truth.position_ecef_m;
    let mut velocity = truth.velocity_ecef_mps;
    for axis in 0..3 {
        position[axis] += 6.0 * rng.standard_normal();
        velocity[axis] += 0.4 * rng.standard_normal();
    }
    NavState::new(
        truth.t_j2000_s,
        position,
        velocity,
        truth.attitude_body_to_ecef,
    )
    .expect("nominal")
}

fn truth_trajectory(scenario: MotionScenario) -> Vec<NavState> {
    let mut states = Vec::with_capacity(STEP_COUNT + 1);
    let start_position = [WGS84_A_M + 17.25, -23.5, 41.75];
    let attitude = identity_dcm();
    match scenario {
        MotionScenario::Static => {
            for step in 0..=STEP_COUNT {
                states.push(
                    NavState::new(
                        T0_J2000_S + step as f64 * DT_S,
                        start_position,
                        [0.0; 3],
                        attitude,
                    )
                    .expect("truth"),
                );
            }
        }
        MotionScenario::ConstantVelocity => {
            let velocity = [8.75, -3.5, 1.125];
            for step in 0..=STEP_COUNT {
                let t = step as f64 * DT_S;
                states.push(
                    NavState::new(
                        T0_J2000_S + t,
                        add3(start_position, scale3(velocity, t)),
                        velocity,
                        attitude,
                    )
                    .expect("truth"),
                );
            }
        }
        MotionScenario::ConstantAcceleration => {
            let v0 = [2.25, -1.75, 0.625];
            let acceleration = [3.5, -1.25, 0.875];
            for step in 0..=STEP_COUNT {
                let t = step as f64 * DT_S;
                let velocity = add3(v0, scale3(acceleration, t));
                let position = add3(
                    add3(start_position, scale3(v0, t)),
                    scale3(acceleration, 0.5 * t * t),
                );
                states.push(
                    NavState::new(T0_J2000_S + t, position, velocity, attitude).expect("truth"),
                );
            }
        }
        MotionScenario::AggressiveDynamics => {
            let mut previous_position = start_position;
            let mut previous_velocity = aggressive_velocity(0.0);
            states.push(
                NavState::new(T0_J2000_S, previous_position, previous_velocity, attitude)
                    .expect("truth"),
            );
            for step in 1..=STEP_COUNT {
                let t = step as f64 * DT_S;
                let velocity = aggressive_velocity(t);
                let average_velocity = scale3(add3(previous_velocity, velocity), 0.5);
                let position = add3(previous_position, scale3(average_velocity, DT_S));
                states.push(
                    NavState::new(T0_J2000_S + t, position, velocity, attitude).expect("truth"),
                );
                previous_position = position;
                previous_velocity = velocity;
            }
        }
    }
    states
}

fn turntable_truth_trajectory() -> Vec<NavState> {
    let mut states = Vec::with_capacity(TURNTABLE_STEP_COUNT + 1);
    let position = [WGS84_A_M + 17.25, -23.5, 41.75];
    for step in 0..=TURNTABLE_STEP_COUNT {
        let t = step as f64 * TURNTABLE_DT_S;
        states.push(
            NavState::new(
                T0_J2000_S + t,
                position,
                [0.0; 3],
                yaw_body_to_ecef(TURNTABLE_YAW_RATE_RPS * t),
            )
            .expect("turntable truth"),
        );
    }
    states
}

fn turntable_body_rate_wrt_ecef_rps() -> [f64; 3] {
    [0.0, 0.0, TURNTABLE_YAW_RATE_RPS]
}

fn aggressive_velocity(t_s: f64) -> [f64; 3] {
    [
        14.0 * libm::sin(1.7 * t_s) + 3.25,
        -11.0 * libm::cos(1.3 * t_s + 0.4),
        5.5 * libm::sin(2.1 * t_s + 0.2) - 0.75,
    ]
}

fn truth_increments(trajectory: &[NavState]) -> Vec<CorrectedImuIncrement> {
    trajectory
        .windows(2)
        .map(|window| {
            let increment =
                true_imu_increment_between(&window[0], &window[1]).expect("truth increment");
            let reached = mechanize_ecef(&window[0], &increment, MechanizationConfig::default())
                .expect("truth mechanization");
            assert_state_close(&reached, &window[1], 2.5e-8);
            increment
        })
        .collect()
}

fn loose_measurement(
    truth: &NavState,
    rng: &mut SplitMix64,
    actual_noise_scale: f64,
) -> GnssFixMeasurement {
    let position_sigma = POSITION_SIGMA_M * actual_noise_scale;
    let velocity_sigma = VELOCITY_SIGMA_MPS * actual_noise_scale;
    GnssFixMeasurement::position_velocity(
        truth.t_j2000_s,
        add_noise3(truth.position_ecef_m, position_sigma, rng),
        add_noise3(truth.velocity_ecef_mps, velocity_sigma, rng),
        diagonal_covariance(&[
            POSITION_SIGMA_M * POSITION_SIGMA_M,
            POSITION_SIGMA_M * POSITION_SIGMA_M,
            POSITION_SIGMA_M * POSITION_SIGMA_M,
            VELOCITY_SIGMA_MPS * VELOCITY_SIGMA_MPS,
            VELOCITY_SIGMA_MPS * VELOCITY_SIGMA_MPS,
            VELOCITY_SIGMA_MPS * VELOCITY_SIGMA_MPS,
        ]),
        8,
    )
    .expect("loose measurement")
}

fn tight_epoch(
    source: &LinearSource,
    truth: &NavState,
    rng: &mut SplitMix64,
    actual_noise_scale: f64,
) -> TightGnssEpoch {
    tight_epoch_with_kinematics(
        source,
        truth,
        [0.0; 3],
        [0.0; 3],
        CODE_SIGMA_M,
        RANGE_RATE_SIGMA_MPS,
        rng,
        actual_noise_scale,
    )
}

#[allow(clippy::too_many_arguments)]
fn tight_epoch_with_kinematics(
    source: &LinearSource,
    truth: &NavState,
    lever_arm_body_m: [f64; 3],
    body_rate_wrt_ecef_rps: [f64; 3],
    code_sigma_m: f64,
    range_rate_sigma_mps: f64,
    rng: &mut SplitMix64,
    actual_noise_scale: f64,
) -> TightGnssEpoch {
    let mut options = TransmitTimeOptions::default();
    options.light_time = false;
    options.sagnac = false;
    let antenna = antenna_truth_kinematics(truth, lever_arm_body_m, body_rate_wrt_ecef_rps);
    let code_noise_sigma = code_sigma_m * actual_noise_scale;
    let range_rate_noise_sigma = range_rate_sigma_mps * actual_noise_scale;
    let observations = source
        .states
        .iter()
        .map(|state| {
            let satellite = transmit_time_satellite_state(
                source,
                state.satellite_id,
                antenna.position_ecef_m,
                truth.t_j2000_s,
                options,
            )
            .expect("satellite state");
            let range_rate = predict_range_rate_m_s(
                &VelocityObservation {
                    sat: state.satellite_id,
                    satellite_position_m: satellite.position_ecef_m,
                    satellite_velocity_m_s: satellite.velocity_m_s,
                    measured_range_rate_m_s: 0.0,
                    sigma_m_s: range_rate_sigma_mps,
                    satellite_clock_drift_m_s: 0.0,
                },
                ReceiverVelocityState {
                    position_m: antenna.position_ecef_m,
                    velocity_m_s: antenna.velocity_ecef_mps,
                    clock_drift_m_s: 0.0,
                },
            )
            .expect("range rate");
            TightGnssObservation {
                satellite_id: state.satellite_id,
                pseudorange_m: satellite.geometric_range_m - C_M_S * state.clock_s
                    + code_noise_sigma * rng.standard_normal(),
                pseudorange_sigma_m: code_sigma_m,
                range_rate: Some(TightRangeRateObservation {
                    measured_range_rate_m_s: range_rate.range_rate_m_s
                        + range_rate_noise_sigma * rng.standard_normal(),
                    sigma_m_s: range_rate_sigma_mps,
                    satellite_clock_drift_m_s: 0.0,
                }),
                carrier_phase: None,
                ionosphere_delay_m: 0.0,
                troposphere_delay_m: 0.0,
            }
        })
        .collect();
    TightGnssEpoch::new(truth.t_j2000_s, observations).expect("tight epoch")
}

fn nees_position_velocity(filter: &InertialFilter, truth: &NavState) -> f64 {
    let state = filter.state();
    let error = DVector::from_row_slice(&[
        state.nominal.position_ecef_m[0] - truth.position_ecef_m[0],
        state.nominal.position_ecef_m[1] - truth.position_ecef_m[1],
        state.nominal.position_ecef_m[2] - truth.position_ecef_m[2],
        state.nominal.velocity_ecef_mps[0] - truth.velocity_ecef_mps[0],
        state.nominal.velocity_ecef_mps[1] - truth.velocity_ecef_mps[1],
        state.nominal.velocity_ecef_mps[2] - truth.velocity_ecef_mps[2],
    ]);
    let indices = [
        ERROR_POSITION_INDEX,
        ERROR_POSITION_INDEX + 1,
        ERROR_POSITION_INDEX + 2,
        ERROR_VELOCITY_INDEX,
        ERROR_VELOCITY_INDEX + 1,
        ERROR_VELOCITY_INDEX + 2,
    ];
    let covariance = DMatrix::from_fn(6, 6, |row, col| {
        state.covariance[indices[row]][indices[col]]
    });
    let solved = covariance
        .cholesky()
        .expect("NEES covariance")
        .solve(&error);
    error.dot(&solved)
}

fn chi_square_average_band(dof: usize) -> (f64, f64) {
    let lower = sidereon_core::quality::chi2_inv(ALPHA * 0.5, dof).expect("chi-square lower")
        / RUNS_PER_SCENARIO as f64;
    let upper = sidereon_core::quality::chi2_inv(1.0 - ALPHA * 0.5, dof).expect("chi-square upper")
        / RUNS_PER_SCENARIO as f64;
    (lower, upper)
}

fn assert_update_shape(update: FusionUpdate, filter_kind: FilterKind) {
    assert!(update.applied);
    match filter_kind {
        FilterKind::Loose => assert_eq!(update.rows, 6),
        FilterKind::Tight => assert_eq!(update.rows, 16),
    }
    assert_eq!(update.accepted_rows, update.rows);
    assert_eq!(update.rejected_rows, 0);
}

fn diagonal_covariance(diagonal: &[f64]) -> Vec<Vec<f64>> {
    let mut covariance = vec![vec![0.0; diagonal.len()]; diagonal.len()];
    for (idx, value) in diagonal.iter().enumerate() {
        covariance[idx][idx] = *value;
    }
    covariance
}

fn add_noise3(value: [f64; 3], sigma: f64, rng: &mut SplitMix64) -> [f64; 3] {
    [
        value[0] + sigma * rng.standard_normal(),
        value[1] + sigma * rng.standard_normal(),
        value[2] + sigma * rng.standard_normal(),
    ]
}

fn identity_dcm() -> [[f64; 3]; 3] {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn yaw_body_to_ecef(yaw_rad: f64) -> [[f64; 3]; 3] {
    let (sin_yaw, cos_yaw) = libm::sincos(yaw_rad);
    [
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ]
}

#[derive(Debug, Clone, Copy)]
struct AntennaTruthKinematics {
    position_ecef_m: [f64; 3],
    velocity_ecef_mps: [f64; 3],
}

fn antenna_truth_kinematics(
    truth: &NavState,
    lever_arm_body_m: [f64; 3],
    body_rate_wrt_ecef_rps: [f64; 3],
) -> AntennaTruthKinematics {
    let lever_arm_ecef_m = mul_vec3(&truth.attitude_body_to_ecef, lever_arm_body_m);
    let lever_velocity_body_mps = cross3(body_rate_wrt_ecef_rps, lever_arm_body_m);
    let lever_velocity_ecef_mps = mul_vec3(&truth.attitude_body_to_ecef, lever_velocity_body_mps);
    AntennaTruthKinematics {
        position_ecef_m: add3(truth.position_ecef_m, lever_arm_ecef_m),
        velocity_ecef_mps: add3(truth.velocity_ecef_mps, lever_velocity_ecef_mps),
    }
}

fn assert_state_close(actual: &NavState, expected: &NavState, tolerance: f64) {
    for axis in 0..3 {
        assert!(
            (actual.position_ecef_m[axis] - expected.position_ecef_m[axis]).abs() <= tolerance,
            "position axis {axis}: {:.17e} vs {:.17e}",
            actual.position_ecef_m[axis],
            expected.position_ecef_m[axis]
        );
        assert!(
            (actual.velocity_ecef_mps[axis] - expected.velocity_ecef_mps[axis]).abs() <= tolerance,
            "velocity axis {axis}: {:.17e} vs {:.17e}",
            actual.velocity_ecef_mps[axis],
            expected.velocity_ecef_mps[axis]
        );
    }
}

fn ensemble_seed(
    filter_kind: FilterKind,
    profile: ImuProfile,
    scenario: MotionScenario,
    run: usize,
) -> u64 {
    let kind = match filter_kind {
        FilterKind::Loose => 0x1000_0000_0000_0001,
        FilterKind::Tight => 0x2000_0000_0000_0002,
    };
    kind ^ profile.seed_tag
        ^ scenario.seed_tag()
        ^ (run as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

fn turntable_seed(run: usize) -> u64 {
    TURNTABLE_SEED_TAG ^ (run as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15)
}

#[derive(Debug, Clone)]
struct LinearSource {
    t0_j2000_s: f64,
    states: Vec<SatelliteState>,
}

impl LinearSource {
    fn from_receiver(receiver: [f64; 3]) -> Self {
        let range_m = 23_456_789.125;
        let directions = [
            [0.81, 0.42, 0.39],
            [0.76, -0.51, 0.42],
            [0.63, 0.21, -0.75],
            [0.54, -0.69, -0.48],
            [-0.36, 0.84, 0.41],
            [-0.47, -0.68, 0.56],
            [-0.72, 0.33, -0.61],
            [0.24, -0.13, 0.96],
        ];
        let velocities = [
            [318.25, -91.5, 44.125],
            [-204.75, -133.625, 78.5],
            [126.5, 211.25, -63.75],
            [-88.25, 249.5, 102.375],
            [172.75, -182.125, -94.25],
            [-245.5, 67.875, 119.0],
            [54.625, 154.75, -137.5],
            [-31.25, -221.5, 83.625],
        ];
        let states = directions
            .iter()
            .zip(velocities)
            .enumerate()
            .map(|(idx, (direction, velocity_ecef_mps))| {
                let unit = unit3(*direction);
                SatelliteState {
                    satellite_id: GnssSatelliteId::new(GnssSystem::Gps, (idx + 1) as u8)
                        .expect("satellite id"),
                    position_ecef_m: add3(receiver, scale3(unit, range_m)),
                    velocity_ecef_mps,
                    clock_s: 0.0,
                }
            })
            .collect();
        Self {
            t0_j2000_s: T0_J2000_S,
            states,
        }
    }
}

impl ObservableEphemerisSource for LinearSource {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let state = self
            .states
            .iter()
            .find(|state| state.satellite_id == sat)
            .ok_or(ObservablesError::NoEphemeris)?;
        let dt_s = t_j2000_s - self.t0_j2000_s;
        Ok(ObservableState {
            position_ecef_m: add3(state.position_ecef_m, scale3(state.velocity_ecef_mps, dt_s)),
            clock_s: Some(state.clock_s),
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SatelliteState {
    satellite_id: GnssSatelliteId,
    position_ecef_m: [f64; 3],
    velocity_ecef_mps: [f64; 3],
    clock_s: f64,
}

fn unit3(value: [f64; 3]) -> [f64; 3] {
    let n = norm3(value);
    scale3(value, 1.0 / n)
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn unit_f64(&mut self) -> f64 {
        let bits = 0x3ff0_0000_0000_0000 | (self.next_u64() >> 12);
        f64::from_bits(bits) - 1.0
    }

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        loop {
            let u = 2.0 * self.unit_f64() - 1.0;
            let v = 2.0 * self.unit_f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let scale = libm::sqrt(-2.0 * libm::log(s) / s);
                self.spare_normal = Some(v * scale);
                return u * scale;
            }
        }
    }
}
