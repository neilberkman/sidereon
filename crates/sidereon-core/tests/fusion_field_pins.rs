use nalgebra::{DMatrix, DVector};
use sidereon_core::astro::constants::earth::WGS84_A_M;
use sidereon_core::astro::math::vec3::{norm3, sub3};
use sidereon_core::astro::time::civil::{
    civil_from_j2000_seconds, day_of_year, j2000_seconds, second_of_day,
};
use sidereon_core::constants::C_M_S;
use sidereon_core::fusion::{
    smooth_fusion_rts, ErrorStateLayout, FusionRtsHistoryBuilder, GnssFixMeasurement,
    GnssFixStatus, GnssFixStatusWeighting, IggIiiMeasurementReweighting, InertialFilter,
    InertialFilterConfig, InsFilterState, LooseCouplingConfig, NonHolonomicConstraintConfig,
    StationaryDetectorConfig, StationaryUpdateConfig, TightCouplingConfig, TightGnssEpoch,
    TightGnssObservation, TightRangeRateObservation, VelocityMatchState, VelocityMatchingConfig,
    YangPredictionAdaptiveFactor, ERROR_ACCEL_BIAS_INDEX, ERROR_GYRO_BIAS_INDEX,
    ERROR_POSITION_INDEX, ERROR_STATE_DIMENSION_15, ERROR_VELOCITY_INDEX,
};
use sidereon_core::inertial::{
    simulate_imu_samples_from_increments, true_imu_increment_between, ImuBias, ImuGrade, ImuSample,
    ImuSimulationOptions, ImuSpec, NavState,
};
use sidereon_core::positioning::{
    solve_with_doppler_velocity, Corrections, DopplerObservation, KlobucharCoeffs, Observation,
    SolveInputs, SurfaceMet,
};
use sidereon_core::scenario::{
    simulate_scenario, Scenario, ScenarioClockModel, ScenarioConstellation, ScenarioEpochRange,
    ScenarioErrorBudget, ScenarioGeodeticPosition, ScenarioIonosphereModel, ScenarioReceiver,
    ScenarioReceiverWaypoint, ScenarioSignal, ScenarioSpecularMultipath, ScenarioThermalNoise,
    ScenarioTroposphereModel, SyntheticKeplerOrbit, SyntheticKeplerSource, SyntheticObservationSet,
    SCENARIO_SCHEMA_VERSION,
};
use sidereon_core::velocity::doppler_to_range_rate;
use sidereon_core::{GnssSatelliteId, GnssSystem};

const SEED: u64 = 0x51d3_7e0f_f1e1_d511;
const EPOCH_COUNT: usize = 50;
const CADENCE_S: f64 = 1.0;
const OUTLIER_STRIDE: usize = 10;
const OUTLIER_OFFSET: usize = 5;
const OUTLIER_RANGE_M: f64 = 55.0;
const POSITION_SIGMA_FLOOR_M: f64 = 3.25;
const VELOCITY_SIGMA_FLOOR_M_S: f64 = 0.075;
const TIGHT_CODE_SIGMA_M: f64 = 1.0;
const TIGHT_RANGE_RATE_SIGMA_M_S: f64 = 0.01;
const LOW_SAT_START: usize = 18;
const LOW_SAT_LEN: usize = 12;
const LOW_SAT_RUNS: usize = 16;
const OUTAGE_START: usize = 20;
const OUTAGE_LEN: usize = 10;
const RECONVERGENCE_EPOCHS: usize = 4;
const IDENTITY_3: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
const IONO: KlobucharCoeffs = KlobucharCoeffs {
    alpha: [7.5e-9, 1.1e-8, -6.0e-9, 0.0],
    beta: [90_000.0, 0.0, -60_000.0, 0.0],
};
const MET: SurfaceMet = SurfaceMet {
    pressure_hpa: 1009.0,
    temperature_k: 292.0,
    relative_humidity: 0.58,
};

#[test]
fn fused_beats_own_gnss_with_smoother_on_faulted_field_scenario() {
    let scenario = field_scenario();
    let simulated = simulate_scenario(&scenario).expect("simulate scenario");
    let source = source_from_scenario(&scenario);
    let truth = truth_nav_states(&simulated);
    let gnss = solve_gnss_epochs(&simulated, &source, true);

    let run = run_loose_fusion(&truth, &gnss, None, true);
    let filtered_rms = rms_position_error(
        run.positions
            .iter()
            .zip(truth.iter())
            .map(|(position, truth)| (*position, truth.position_ecef_m)),
    );
    let history = run.history.expect("recorded history");
    let smoothed = smooth_fusion_rts(&history).expect("smooth fusion");
    let gnss_rms = rms_position_error(
        gnss.iter()
            .zip(truth.iter())
            .map(|(solution, truth)| (solution.position_ecef_m, truth.position_ecef_m)),
    );
    let smoothed_rms = rms_position_error(smoothed.epochs.iter().zip(truth.iter()).map(
        |(epoch, truth)| {
            (
                epoch.snapshot.state.nominal.position_ecef_m,
                truth.position_ecef_m,
            )
        },
    ));

    assert!(
        smoothed_rms < gnss_rms,
        "smoothed RMS {smoothed_rms:.6} m, GNSS-only RMS {gnss_rms:.6} m"
    );
    assert_eq!(outlier_epoch_count(EPOCH_COUNT), 5);
    assert!(
        smoothed_rms < filtered_rms,
        "smoothed RMS {smoothed_rms:.6} m, filtered RMS {filtered_rms:.6} m"
    );
    assert!(
        smoothed_rms * 20.0 < gnss_rms,
        "smoothed RMS {smoothed_rms:.6} m, GNSS-only RMS {gnss_rms:.6} m"
    );
    // Scenario: 50 epochs at 1 Hz, 8 GPS L1 satellites, 1.4 m code noise,
    // 0.035 Hz Doppler noise, Klobuchar plus Saastamoinen, 0.32 m multipath,
    // and 5 deterministic single-satellite 55 m code faults. The deterministic
    // run achieves about 0.31 m smoothed RMS versus about 24.22 m GNSS-only
    // RMS; these bounds leave roughly 25 to 30 percent headroom.
    assert!(smoothed_rms < 0.40, "smoothed RMS {smoothed_rms:.6} m");
    assert!(gnss_rms < 30.0, "GNSS-only RMS {gnss_rms:.6} m");
}

#[test]
fn outage_coast_stays_inside_imu_grade_bound_and_recovers() {
    let scenario = field_scenario();
    let simulated = simulate_scenario(&scenario).expect("simulate scenario");
    let source = source_from_scenario(&scenario);
    let truth = truth_nav_states(&simulated);
    let gnss = solve_gnss_epochs(&simulated, &source, true);
    let run = run_loose_fusion(
        &truth,
        &gnss,
        Some(OUTAGE_START..OUTAGE_START + OUTAGE_LEN),
        false,
    );

    let outage_entry_epoch = OUTAGE_START - 1;
    let outage_end_epoch = OUTAGE_START + OUTAGE_LEN - 1;
    let outage_entry_error = distance3(
        run.positions[outage_entry_epoch],
        truth[outage_entry_epoch].position_ecef_m,
    );
    let outage_end_error = distance3(
        run.positions[outage_end_epoch],
        truth[outage_end_epoch].position_ecef_m,
    );
    let max_outage_error = (OUTAGE_START..OUTAGE_START + OUTAGE_LEN)
        .map(|idx| distance3(run.positions[idx], truth[idx].position_ecef_m))
        .fold(0.0_f64, f64::max);
    let outage_span_s = truth[outage_end_epoch].t_j2000_s - truth[outage_entry_epoch].t_j2000_s;
    let tactical = ImuSpec::preset(ImuGrade::Tactical);
    let accel_vrw_bound_m = 3.0 * tactical.accel_vrw_mps_sqrt_s * libm::pow(outage_span_s, 1.5);
    let accel_bias_bound_m = 1.5 * tactical.accel_bias_instab_mps2 * outage_span_s.powi(2);
    let imu_coast_growth_bound_m = accel_vrw_bound_m + accel_bias_bound_m + 0.30;

    // Outage spans epochs 20 through 29 after epoch 19 was corrected. Tactical
    // IMU accel VRW is 0.005 m/s/sqrt(s), so 10 s contributes
    // 3 * VRW * t^(3/2) = 0.47 m. The 0.005 m/s^2 accel bias term contributes
    // 0.5 * 3 * bias * t^2 = 0.75 m. The 1.52 m bound includes 0.30 m for
    // attitude and linearization residuals, and is applied as coast growth over
    // the last corrected epoch rather than as a loose prior-velocity envelope.
    assert!(
        outage_end_error < outage_entry_error + imu_coast_growth_bound_m,
        "outage end error {outage_end_error:.6} m, entry error {outage_entry_error:.6} m, growth bound {imu_coast_growth_bound_m:.6} m"
    );
    assert!(
        max_outage_error < outage_entry_error + imu_coast_growth_bound_m,
        "max outage error {max_outage_error:.6} m, entry error {outage_entry_error:.6} m, growth bound {imu_coast_growth_bound_m:.6} m"
    );

    let return_epoch = OUTAGE_START + OUTAGE_LEN;
    let reconverged_epoch = (return_epoch..=return_epoch + RECONVERGENCE_EPOCHS)
        .find(|epoch| {
            let error = distance3(run.positions[*epoch], truth[*epoch].position_ecef_m);
            let nees = nees_position_velocity(
                &run.positions[*epoch],
                &run.velocities[*epoch],
                &run.covariances[*epoch],
                &truth[*epoch],
            );
            error < 4.0 && chi_square_band(6, 1).contains(&nees)
        })
        .expect("reconverged within configured epoch limit");
    assert!(
        reconverged_epoch <= return_epoch + RECONVERGENCE_EPOCHS,
        "reconverged epoch {reconverged_epoch}, return epoch {return_epoch}"
    );
}

#[test]
fn low_sat_tight_window_remains_bounded_and_consistent() {
    let scenario = low_sat_scenario();
    let simulated = simulate_scenario(&scenario).expect("simulate scenario");
    let source = source_from_scenario(&scenario);
    let truth = truth_nav_states(&simulated);
    let reference_seed = SEED ^ 0x7168_7000_5a75_0f11;
    let reference = run_tight_fusion(
        &truth,
        &simulated,
        &source,
        LOW_SAT_START..LOW_SAT_START + LOW_SAT_LEN,
        true,
        reference_seed,
    );
    let coast = run_tight_fusion(
        &truth,
        &simulated,
        &source,
        LOW_SAT_START..LOW_SAT_START + LOW_SAT_LEN,
        false,
        reference_seed,
    );

    let mut max_error = 0.0_f64;
    let mut nees_sum_by_epoch = [0.0_f64; LOW_SAT_LEN];
    for run_idx in 0..LOW_SAT_RUNS {
        let seed =
            SEED ^ 0x7168_7000_5a75_0f11 ^ (run_idx as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut run_scenario = low_sat_scenario();
        run_scenario.seed = seed;
        let run_simulated = simulate_scenario(&run_scenario).expect("simulate low-sat run");
        let run_source = source_from_scenario(&run_scenario);
        let run_truth = truth_nav_states(&run_simulated);
        let run = run_tight_fusion(
            &run_truth,
            &run_simulated,
            &run_source,
            LOW_SAT_START..LOW_SAT_START + LOW_SAT_LEN,
            true,
            seed,
        );
        for (idx, truth_state) in run_truth
            .iter()
            .enumerate()
            .skip(LOW_SAT_START)
            .take(LOW_SAT_LEN)
        {
            max_error = max_error.max(distance3(run.positions[idx], truth_state.position_ecef_m));
            nees_sum_by_epoch[idx - LOW_SAT_START] +=
                nees_position(&run.positions[idx], &run.covariances[idx], truth_state);
        }
    }
    let tail_probability = 0.025 / LOW_SAT_LEN as f64;
    let band = chi_square_band_with_probability(
        tail_probability,
        1.0 - tail_probability,
        3 * LOW_SAT_RUNS,
        LOW_SAT_RUNS,
    );

    // Scenario: epochs 18 through 29 expose only 3 of 8 GPS satellites to the
    // tight filter while inertial propagation uses the same tactical IMU grade.
    // Atmosphere is disabled for this pin, so the 1.0 m code row sigma covers
    // the 0.7 m simulator code noise, 0.32 m multipath, and clock-model
    // residuals without oracle media corrections. NEES is evaluated on the 3D
    // position covariance because this pin is about bounded position integrity
    // in the masked window. Each masked epoch is checked across 16 deterministic
    // simulator, prior, and IMU seeds, so sequential filter epochs are not
    // counted as independent samples. The chi-square band is Bonferroni-adjusted
    // to keep a 95 percent family band across the 12 masked epochs.
    assert!(max_error < 2.0, "low-sat max error {max_error:.6} m");
    assert!(
        position_trace(&reference.covariances[LOW_SAT_START + LOW_SAT_LEN - 1])
            < position_trace(&coast.covariances[LOW_SAT_START + LOW_SAT_LEN - 1]),
        "low-sat update did not reduce position covariance trace"
    );
    for (offset, nees_sum) in nees_sum_by_epoch.into_iter().enumerate() {
        let nees_average = nees_sum / LOW_SAT_RUNS as f64;
        assert!(
            band.contains(&nees_average),
            "low-sat NEES {nees_average:.6} outside [{:.6}, {:.6}] at epoch {}",
            band.start(),
            band.end(),
            LOW_SAT_START + offset
        );
    }
}

#[test]
fn stationary_zupt_zaru_bounds_static_drift_and_estimates_biases() {
    let steps = 80usize;
    let dt_s = 1.0;
    let truth = static_truth(steps, dt_s);
    let accel_bias = [0.020, 0.0, 0.0];
    let gyro_bias = [0.0010, -0.0008, 0.0006];
    let spec = ImuSpec::datasheet(0.0015, 2.0e-5, 0.0002, 2.0e-6, 600.0, 600.0, None, None);
    let sequence = simulate_imu_samples_from_increments(
        &truth_increments(&truth),
        spec,
        ImuSimulationOptions {
            seed: SEED ^ 0x5a75_7100_0000_0001,
            initial_bias: ImuBias {
                accel_mps2: accel_bias,
                gyro_rps: gyro_bias,
            },
            ..ImuSimulationOptions::default()
        },
    )
    .expect("stationary IMU");

    let no_updates = run_stationary_constraint_case(&truth, sequence.samples.clone(), spec, false);
    let with_updates = run_stationary_constraint_case(&truth, sequence.samples, spec, true);
    let no_update_drift = distance3(
        no_updates.state().nominal.position_ecef_m,
        truth[steps].position_ecef_m,
    );
    let zupt_drift = distance3(
        with_updates.state().nominal.position_ecef_m,
        truth[steps].position_ecef_m,
    );
    let accel_bias_error = (with_updates.state().nominal.accel_bias_mps2[0] - accel_bias[0]).abs();
    let gyro_bias_error = norm3(sub3(with_updates.state().nominal.gyro_bias_rps, gyro_bias));
    assert!(
        no_update_drift > 40.0,
        "no-ZUPT drift {no_update_drift:.6} m"
    );
    assert!(zupt_drift < 0.80, "ZUPT drift {zupt_drift:.6} m");
    assert!(
        zupt_drift * 100.0 < no_update_drift,
        "ZUPT drift {zupt_drift:.6} m, no-ZUPT drift {no_update_drift:.6} m"
    );
    assert!(
        accel_bias_error < 0.006,
        "accel x-bias error {accel_bias_error:.6} m/s^2, estimate {:?}, injected {:?}",
        with_updates.state().nominal.accel_bias_mps2,
        accel_bias
    );
    assert!(
        gyro_bias_error < 0.00025,
        "gyro bias error {gyro_bias_error:.9} rad/s"
    );
}

#[test]
fn non_holonomic_constraint_removes_lateral_velocity_error() {
    let steps = 30usize;
    let dt_s = 1.0;
    let truth = straight_vehicle_truth(steps, dt_s);
    let spec = ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 600.0, 600.0, None, None);
    let sequence = simulate_imu_samples_from_increments(
        &truth_increments(&truth),
        spec,
        ImuSimulationOptions {
            seed: SEED ^ 0x5a75_7100_0000_0002,
            ..ImuSimulationOptions::default()
        },
    )
    .expect("vehicle IMU");

    let coast = run_nhc_case(&truth, sequence.samples.clone(), spec, false);
    let constrained = run_nhc_case(&truth, sequence.samples, spec, true);
    let coast_lateral = coast.state().nominal.velocity_ecef_mps[1].abs();
    let constrained_lateral = constrained.state().nominal.velocity_ecef_mps[1].abs();
    assert!(
        coast_lateral > 1.45,
        "coast lateral velocity {coast_lateral:.6} m/s"
    );
    assert!(
        constrained_lateral < 0.015,
        "NHC lateral velocity {constrained_lateral:.6} m/s"
    );
    assert!(
        constrained_lateral * 100.0 < coast_lateral,
        "NHC lateral velocity {constrained_lateral:.6} m/s, coast {coast_lateral:.6} m/s"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
#[allow(clippy::needless_range_loop)]
fn stationary_update_rejects_short_windows_and_nonstationary_samples() {
    let truth = static_truth(3, 1.0);
    let spec = ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 600.0, 600.0, None, None);
    let mut config = LooseCouplingConfig::default();
    config.stationary_updates = Some(StationaryUpdateConfig {
        detector: StationaryDetectorConfig {
            window_len: 3,
            max_specific_force_norm_error_mps2: 0.08,
            max_body_rate_wrt_ecef_norm_rps: 0.003,
        },
        zero_velocity_sigma_mps: 0.015,
        zero_angular_rate_sigma_rps: 0.00008,
    });
    let mut filter = direct_filter(truth[0], spec, initial_covariance_diagonal(), config);
    let stationary_force_mps2 = [9.80665, 0.0, 0.0];
    for step in 1..=2 {
        filter
            .propagate(ImuSample::rate(
                truth[step].t_j2000_s,
                stationary_force_mps2,
                [0.0; 3],
            ))
            .expect("short-window propagate");
        assert!(
            filter
                .update_stationary()
                .expect("short-window stationary update")
                .is_none(),
            "stationary update must wait for a full detector window"
        );
    }
    filter
        .propagate(ImuSample::rate(
            truth[3].t_j2000_s,
            [20.0, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ))
        .expect("moving propagate");
    assert!(
        filter
            .update_stationary()
            .expect("moving stationary update")
            .is_none(),
        "stationary update must reject high force/rate samples"
    );
}

#[test]
#[allow(clippy::field_reassign_with_default)]
fn stationary_update_rejects_duplicate_epoch_application() {
    let truth = static_truth(3, 1.0);
    let spec = ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 600.0, 600.0, None, None);
    let mut config = LooseCouplingConfig::default();
    config.stationary_updates = Some(StationaryUpdateConfig {
        detector: StationaryDetectorConfig {
            window_len: 3,
            max_specific_force_norm_error_mps2: 0.08,
            max_body_rate_wrt_ecef_norm_rps: 0.003,
        },
        zero_velocity_sigma_mps: 0.015,
        zero_angular_rate_sigma_rps: 0.00008,
    });
    let mut filter = direct_filter(truth[0], spec, initial_covariance_diagonal(), config);
    for state in truth.iter().take(4).skip(1) {
        filter
            .propagate(ImuSample::rate(
                state.t_j2000_s,
                [9.80665, 0.0, 0.0],
                [0.0; 3],
            ))
            .expect("stationary propagate");
    }

    assert!(filter
        .update_stationary()
        .expect("first stationary update")
        .is_some());
    assert!(
        filter.update_stationary().is_err(),
        "stationary update must reject a second application at the same epoch"
    );
}

#[test]
fn non_holonomic_constraint_rejects_sub_min_speed_and_high_rate() {
    let truth = straight_vehicle_truth(1, 1.0);
    let spec = ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 600.0, 600.0, None, None);
    let config = LooseCouplingConfig {
        non_holonomic: Some(NonHolonomicConstraintConfig {
            lateral_velocity_sigma_mps: 0.03,
            vertical_velocity_sigma_mps: 0.03,
            min_speed_mps: 2.0,
            max_body_rate_wrt_ecef_norm_rps: 0.01,
        }),
        ..LooseCouplingConfig::default()
    };
    let mut slow_nominal = truth[0];
    slow_nominal.velocity_ecef_mps = [0.5, 0.0, 0.0];
    let mut slow_filter = direct_filter(slow_nominal, spec, initial_covariance_diagonal(), config);
    slow_filter
        .propagate(ImuSample::rate(
            truth[1].t_j2000_s,
            [9.80665, 0.0, 0.0],
            [0.0; 3],
        ))
        .expect("slow propagate");
    assert!(
        slow_filter
            .update_non_holonomic()
            .expect("slow NHC update")
            .is_none(),
        "NHC must reject speeds below the configured gate"
    );

    let mut high_rate_filter = direct_filter(truth[0], spec, initial_covariance_diagonal(), config);
    high_rate_filter
        .propagate(ImuSample::rate(
            truth[1].t_j2000_s,
            [9.80665, 0.0, 0.0],
            [0.02, 0.0, 0.0],
        ))
        .expect("high-rate propagate");
    assert!(
        high_rate_filter
            .update_non_holonomic()
            .expect("high-rate NHC update")
            .is_none(),
        "NHC must reject body rates above the configured gate"
    );
}

#[test]
fn non_holonomic_constraint_rejects_duplicate_epoch_application() {
    let truth = straight_vehicle_truth(1, 1.0);
    let spec = ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 600.0, 600.0, None, None);
    let config = LooseCouplingConfig {
        non_holonomic: Some(NonHolonomicConstraintConfig {
            lateral_velocity_sigma_mps: 0.03,
            vertical_velocity_sigma_mps: 0.03,
            min_speed_mps: 2.0,
            max_body_rate_wrt_ecef_norm_rps: 0.01,
        }),
        ..LooseCouplingConfig::default()
    };
    let mut filter = direct_filter(truth[0], spec, initial_covariance_diagonal(), config);
    filter
        .propagate(ImuSample::rate(
            truth[1].t_j2000_s,
            [9.80665, 0.0, 0.0],
            [0.0; 3],
        ))
        .expect("NHC propagate");

    assert!(filter
        .update_non_holonomic()
        .expect("first NHC update")
        .is_some());
    assert!(
        filter.update_non_holonomic().is_err(),
        "NHC must reject a second application at the same epoch"
    );
}

#[test]
fn velocity_matching_reduces_outage_peak_error_and_keeps_span_continuous() {
    let scenario = field_scenario();
    let simulated = simulate_scenario(&scenario).expect("simulate scenario");
    let source = source_from_scenario(&scenario);
    let truth = truth_nav_states(&simulated);
    let gnss = solve_gnss_epochs(&simulated, &source, true);
    let return_epoch = OUTAGE_START + OUTAGE_LEN;
    let coast = run_loose_fusion_with_filter_and_spec(
        &truth,
        &gnss,
        Some(OUTAGE_START..return_epoch + 1),
        false,
        ImuSpec::preset(ImuGrade::Mems),
        loose_filter,
    );
    let span = (OUTAGE_START - 1..=return_epoch)
        .map(|idx| {
            VelocityMatchState::new(
                truth[idx].t_j2000_s,
                coast.positions[idx],
                coast.velocities[idx],
            )
            .expect("velocity-match state")
        })
        .collect::<Vec<_>>();
    let return_fix = GnssFixMeasurement::position_velocity(
        truth[return_epoch].t_j2000_s,
        truth[return_epoch].position_ecef_m,
        truth[return_epoch].velocity_ecef_mps,
        loose_covariance(&IDENTITY_3),
        8,
    )
    .expect("return fix");
    let matched = sidereon_core::fusion::velocity_match_outage(
        &span,
        &return_fix,
        VelocityMatchingConfig {
            max_outage_duration_s: 20.0,
        },
    )
    .expect("velocity match");

    let coast_peak = (OUTAGE_START..=return_epoch)
        .map(|idx| distance3(coast.positions[idx], truth[idx].position_ecef_m))
        .fold(0.0_f64, f64::max);
    let matched_peak = matched
        .states
        .iter()
        .skip(1)
        .zip(truth.iter().skip(OUTAGE_START))
        .map(|(state, truth)| distance3(state.position_ecef_m, truth.position_ecef_m))
        .fold(0.0_f64, f64::max);
    let return_step = distance3(
        matched.states[matched.states.len() - 1].position_ecef_m,
        matched.states[matched.states.len() - 2].position_ecef_m,
    );
    let truth_return_step = distance3(
        truth[return_epoch].position_ecef_m,
        truth[return_epoch - 1].position_ecef_m,
    );
    assert!(coast_peak > 4.0, "coast peak {coast_peak:.6} m");
    assert!(
        matched_peak * 2.0 < coast_peak,
        "matched peak {matched_peak:.6} m, coast peak {coast_peak:.6} m"
    );
    assert!(
        return_step < truth_return_step + 1.0,
        "return step {return_step:.6} m, truth step {truth_return_step:.6} m"
    );
}

#[test]
fn velocity_matching_can_land_on_posterior_endpoint_state() {
    let states = [
        VelocityMatchState::new(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]).expect("state 0"),
        VelocityMatchState::new(1.0, [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]).expect("state 1"),
        VelocityMatchState::new(2.0, [2.0, 0.0, 0.0], [1.0, 0.0, 0.0]).expect("state 2"),
    ];
    let endpoint =
        VelocityMatchState::new(2.0, [2.4, -0.2, 0.1], [0.7, 0.3, -0.1]).expect("endpoint");

    let matched = sidereon_core::fusion::velocity_match_outage_to_state(
        &states,
        endpoint,
        VelocityMatchingConfig {
            max_outage_duration_s: 5.0,
        },
    )
    .expect("velocity match to state");

    assert_eq!(matched.states[0], states[0]);
    assert_eq!(matched.states[2], endpoint);
    assert_vec3_close(
        matched.endpoint_position_correction_ecef_m,
        [0.4, -0.2, 0.1],
        1.0e-15,
    );
    assert_vec3_close(
        matched.endpoint_velocity_correction_ecef_mps,
        [-0.3, 0.3, -0.1],
        1.0e-15,
    );
}

#[test]
fn fix_status_weighting_inflates_float_covariance_by_configured_sigma() {
    let spec = ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 600.0, 600.0, None, None);
    let nominal =
        NavState::new(0.0, [WGS84_A_M + 1.0, 0.0, 0.0], [0.0; 3], IDENTITY_3).expect("nominal");
    let mut diagonal = [1.0e-9; ERROR_STATE_DIMENSION_15];
    for axis in 0..3 {
        diagonal[ERROR_POSITION_INDEX + axis] = 4.0;
    }
    let measurement = GnssFixMeasurement::position(0.0, [WGS84_A_M, 0.0, 0.0], IDENTITY_3, 8)
        .expect("measurement")
        .with_fix_status(GnssFixStatus::Float);
    let mut unweighted = direct_filter(nominal, spec, diagonal, LooseCouplingConfig::default());
    let mut weighted = direct_filter(
        nominal,
        spec,
        diagonal,
        LooseCouplingConfig {
            fix_status_weighting: GnssFixStatusWeighting {
                float_sigma_multiplier: 3.0,
                ..GnssFixStatusWeighting::default()
            },
            ..LooseCouplingConfig::default()
        },
    );

    unweighted
        .update_loose(&measurement)
        .expect("unweighted update");
    weighted
        .update_loose(&measurement)
        .expect("weighted update");
    let unweighted_var = unweighted.state().covariance[ERROR_POSITION_INDEX][ERROR_POSITION_INDEX];
    let weighted_var = weighted.state().covariance[ERROR_POSITION_INDEX][ERROR_POSITION_INDEX];
    let expected_unweighted = 4.0 / 5.0;
    let expected_weighted = 36.0 / 13.0;
    assert!((unweighted_var - expected_unweighted).abs() < 1.0e-14);
    assert!((weighted_var - expected_weighted).abs() < 1.0e-14);
    assert!(
        weighted_var > unweighted_var,
        "weighted variance {weighted_var:.12}, unweighted {unweighted_var:.12}"
    );
}

#[test]
fn field_mode_defaults_keep_existing_loose_fixture_bits() {
    const EXPECTED_DEFAULT_HASH: u64 = 0x06cb_d2e4_b7b1_729c;
    let scenario = field_scenario();
    let simulated = simulate_scenario(&scenario).expect("simulate scenario");
    let source = source_from_scenario(&scenario);
    let truth = truth_nav_states(&simulated);
    let gnss = solve_gnss_epochs(&simulated, &source, true);
    let baseline = run_loose_fusion_with_filter(&truth, &gnss, None, false, loose_filter);
    let explicit_defaulted =
        run_loose_fusion_with_filter(&truth, &gnss, None, false, |truth, spec| {
            let state = InsFilterState::from_diagonal(
                initial_nominal(truth),
                ErrorStateLayout::Fifteen,
                &initial_covariance_diagonal(),
            )
            .expect("filter state");
            let mut config = InertialFilterConfig::new(spec).expect("filter config");
            config.loose = LooseCouplingConfig {
                fix_status_weighting: GnssFixStatusWeighting::default(),
                stationary_updates: None,
                non_holonomic: None,
                measurement_reweighting: Some(IggIiiMeasurementReweighting::standard()),
                prediction_adaptation: Some(YangPredictionAdaptiveFactor::standard()),
                ..LooseCouplingConfig::default()
            };
            InertialFilter::with_config(state, config).expect("filter")
        });

    let default_hash = fusion_run_bits_hash(&baseline);
    assert_eq!(
        default_hash, EXPECTED_DEFAULT_HASH,
        "default fixture hash {default_hash:#018x}"
    );
    assert_eq!(default_hash, fusion_run_bits_hash(&explicit_defaulted));
}

#[derive(Debug, Clone)]
struct GnssEpochSolution {
    position_ecef_m: [f64; 3],
    velocity_ecef_m_s: [f64; 3],
    covariance: Vec<Vec<f64>>,
    satellites_used: usize,
}

#[derive(Debug, Clone)]
struct FusionRun {
    positions: Vec<[f64; 3]>,
    velocities: Vec<[f64; 3]>,
    covariances: Vec<Vec<Vec<f64>>>,
    history: Option<sidereon_core::fusion::FusionRtsHistory>,
}

fn fusion_run_bits_hash(run: &FusionRun) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    hash_usize(&mut hash, run.positions.len());
    for position in &run.positions {
        for value in position {
            hash_u64(&mut hash, value.to_bits());
        }
    }
    hash_usize(&mut hash, run.velocities.len());
    for velocity in &run.velocities {
        for value in velocity {
            hash_u64(&mut hash, value.to_bits());
        }
    }
    hash_usize(&mut hash, run.covariances.len());
    for covariance in &run.covariances {
        hash_usize(&mut hash, covariance.len());
        for row in covariance {
            hash_usize(&mut hash, row.len());
            for value in row {
                hash_u64(&mut hash, value.to_bits());
            }
        }
    }
    hash
}

fn hash_usize(hash: &mut u64, value: usize) {
    hash_u64(hash, value as u64);
}

fn hash_u64(hash: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

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

fn field_scenario() -> Scenario {
    Scenario {
        schema_version: SCENARIO_SCHEMA_VERSION,
        seed: SEED,
        epochs: ScenarioEpochRange {
            start_j2000_s: start_j2000_s(),
            count: EPOCH_COUNT,
            cadence_s: CADENCE_S,
        },
        receiver: ScenarioReceiver::KinematicWaypoints {
            waypoints: vec![
                waypoint(0.0, 0.0, 0.0, 18.0),
                waypoint(49.0, 0.000_080, 0.000_215, 24.0),
            ],
        },
        constellation: ScenarioConstellation::SyntheticKeplerian {
            satellites: gps_field_orbits(),
        },
        signals: vec![ScenarioSignal::l1_ca(GnssSystem::Gps)],
        error_budget: ScenarioErrorBudget {
            receiver_clock: ScenarioClockModel {
                enabled: true,
                bias_s: 3.0e-8,
                drift_s_s: 1.0e-10,
                power_law_coefficients: [0.0, 0.0, 1.0e-23, 0.0, 0.0],
            },
            satellite_clock: ScenarioClockModel::disabled(),
            ionosphere: ScenarioIonosphereModel::Klobuchar {
                alpha: IONO.alpha,
                beta: IONO.beta,
            },
            troposphere: ScenarioTroposphereModel::SaastamoinenNiell {
                pressure_hpa: MET.pressure_hpa,
                temperature_k: MET.temperature_k,
                relative_humidity: MET.relative_humidity,
            },
            thermal_noise: ScenarioThermalNoise {
                enabled: true,
                pseudorange_sigma_m: 1.4,
                carrier_phase_sigma_m: 0.012,
                doppler_sigma_hz: 0.035,
            },
            multipath: ScenarioSpecularMultipath {
                enabled: true,
                amplitude_m: 0.32,
                reflector_height_m: 1.45,
                phase_rad: 0.4,
            },
            elevation_mask_deg: 5.0,
        },
    }
}

fn low_sat_scenario() -> Scenario {
    let mut scenario = field_scenario();
    scenario.error_budget.ionosphere = ScenarioIonosphereModel::Off;
    scenario.error_budget.troposphere = ScenarioTroposphereModel::Off;
    scenario.error_budget.thermal_noise.pseudorange_sigma_m = 0.7;
    scenario
}

fn waypoint(offset_s: f64, lat_rad: f64, lon_rad: f64, height_m: f64) -> ScenarioReceiverWaypoint {
    ScenarioReceiverWaypoint {
        offset_s,
        position: ScenarioGeodeticPosition {
            lat_rad,
            lon_rad,
            height_m,
        },
        velocity_ecef_m_s: None,
    }
}

fn gps_field_orbits() -> Vec<SyntheticKeplerOrbit> {
    let a = 26_560_000.0;
    let u45 = core::f64::consts::FRAC_PI_4;
    let u60 = core::f64::consts::PI / 3.0;
    [
        (1, 0.0, 0.0, 0.0),
        (2, 0.0, 0.0, u60),
        (3, 0.0, 0.0, -u60),
        (4, 0.0, core::f64::consts::FRAC_PI_2, u60),
        (5, 0.0, core::f64::consts::FRAC_PI_2, -u60),
        (6, u60, 0.0, 0.0),
        (7, -u60, 0.0, 0.0),
        (8, u45, core::f64::consts::FRAC_PI_2, u45),
    ]
    .into_iter()
    .map(
        |(prn, raan_rad, inclination_rad, mean_anomaly_rad)| SyntheticKeplerOrbit {
            satellite_id: GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS PRN"),
            semi_major_axis_m: a,
            eccentricity: 0.0,
            inclination_rad,
            raan_rad,
            arg_perigee_rad: 0.0,
            mean_anomaly_rad,
            epoch_j2000_s: start_j2000_s(),
            clock_bias_s: 0.0,
            clock_drift_s_s: 0.0,
        },
    )
    .collect()
}

fn start_j2000_s() -> f64 {
    j2000_seconds(2026, 1, 1, 0, 0, 0.0)
}

fn source_from_scenario(scenario: &Scenario) -> SyntheticKeplerSource {
    let ScenarioConstellation::SyntheticKeplerian { satellites } = &scenario.constellation else {
        panic!("synthetic scenario expected");
    };
    SyntheticKeplerSource::new(satellites.clone()).expect("source")
}

fn truth_nav_states(set: &SyntheticObservationSet) -> Vec<NavState> {
    set.receiver_truth
        .iter()
        .map(|truth| {
            NavState::new(
                truth.t_rx_j2000_s,
                truth.position_ecef_m,
                truth.velocity_ecef_m_s,
                IDENTITY_3,
            )
            .expect("truth nav state")
        })
        .collect()
}

fn solve_gnss_epochs(
    set: &SyntheticObservationSet,
    source: &SyntheticKeplerSource,
    inject_outliers: bool,
) -> Vec<GnssEpochSolution> {
    let mut out = Vec::with_capacity(set.receiver_truth.len());
    let mut initial_guess = [
        set.receiver_truth[0].position_ecef_m[0],
        set.receiver_truth[0].position_ecef_m[1],
        set.receiver_truth[0].position_ecef_m[2],
        0.0,
    ];

    for epoch_index in 0..set.receiver_truth.len() {
        let truth = set.receiver_truth[epoch_index];
        let observations = spp_observations_for_epoch(set, epoch_index, inject_outliers);
        let doppler = doppler_observations_for_epoch(set, epoch_index);
        let time = spp_time_context(truth.t_rx_j2000_s);
        let inputs = SolveInputs {
            observations,
            t_rx_j2000_s: truth.t_rx_j2000_s,
            t_rx_second_of_day_s: time.second_of_day_s,
            day_of_year: time.day_of_year,
            initial_guess,
            corrections: Corrections::IONO_TROPO,
            klobuchar: IONO,
            met: MET,
            robust: None,
            ..SolveInputs::default()
        };
        let solved = solve_with_doppler_velocity(source, &inputs, &doppler, false)
            .expect("SPP plus Doppler solution");
        let velocity = solved
            .velocity
            .as_ref()
            .expect("Doppler velocity")
            .velocity_m_s;
        let covariance = loose_covariance(&solved.receiver.position_covariance.ecef_m2);
        initial_guess = [
            solved.receiver.position.as_array()[0],
            solved.receiver.position.as_array()[1],
            solved.receiver.position.as_array()[2],
            solved.receiver.rx_clock_s * C_M_S,
        ];
        out.push(GnssEpochSolution {
            position_ecef_m: solved.receiver.position.as_array(),
            velocity_ecef_m_s: velocity,
            covariance,
            satellites_used: solved.receiver.metadata.used_count,
        });
    }
    out
}

fn spp_observations_for_epoch(
    set: &SyntheticObservationSet,
    epoch_index: usize,
    inject_outliers: bool,
) -> Vec<Observation> {
    let start = set.observations.epoch_offsets[epoch_index];
    let end = set.observations.epoch_offsets[epoch_index + 1];
    let outlier_epoch = inject_outliers
        && epoch_index >= OUTLIER_OFFSET
        && (epoch_index - OUTLIER_OFFSET).is_multiple_of(OUTLIER_STRIDE);
    (start..end)
        .map(|index| {
            let mut pseudorange_m = set.observations.pseudorange_m[index];
            if outlier_epoch && index == start {
                pseudorange_m += OUTLIER_RANGE_M;
            }
            Observation {
                satellite_id: set.observations.satellite_id[index],
                pseudorange_m,
            }
        })
        .collect()
}

fn doppler_observations_for_epoch(
    set: &SyntheticObservationSet,
    epoch_index: usize,
) -> Vec<DopplerObservation> {
    let start = set.observations.epoch_offsets[epoch_index];
    let end = set.observations.epoch_offsets[epoch_index + 1];
    (start..end)
        .map(|index| DopplerObservation {
            satellite_id: set.observations.satellite_id[index],
            doppler_hz: set.observations.doppler_hz[index],
            carrier_hz: set.observations.carrier_hz[index],
            sat_clock_drift_s_s: 0.0,
        })
        .collect()
}

fn loose_covariance(position_ecef_m2: &[[f64; 3]; 3]) -> Vec<Vec<f64>> {
    let mut covariance = vec![vec![0.0; 6]; 6];
    for row in 0..3 {
        for col in 0..3 {
            covariance[row][col] = position_ecef_m2[row][col];
        }
        covariance[row][row] = covariance[row][row].max(POSITION_SIGMA_FLOOR_M.powi(2));
        covariance[row + 3][row + 3] = VELOCITY_SIGMA_FLOOR_M_S.powi(2);
    }
    covariance
}

fn run_loose_fusion(
    truth: &[NavState],
    gnss: &[GnssEpochSolution],
    outage: Option<std::ops::Range<usize>>,
    recorded: bool,
) -> FusionRun {
    run_loose_fusion_with_filter(truth, gnss, outage, recorded, loose_filter)
}

fn run_loose_fusion_with_filter<F>(
    truth: &[NavState],
    gnss: &[GnssEpochSolution],
    outage: Option<std::ops::Range<usize>>,
    recorded: bool,
    build_filter: F,
) -> FusionRun
where
    F: Fn(NavState, ImuSpec) -> InertialFilter,
{
    run_loose_fusion_with_filter_and_spec(
        truth,
        gnss,
        outage,
        recorded,
        ImuSpec::preset(ImuGrade::Tactical),
        build_filter,
    )
}

fn run_loose_fusion_with_filter_and_spec<F>(
    truth: &[NavState],
    gnss: &[GnssEpochSolution],
    outage: Option<std::ops::Range<usize>>,
    recorded: bool,
    spec: ImuSpec,
    build_filter: F,
) -> FusionRun
where
    F: Fn(NavState, ImuSpec) -> InertialFilter,
{
    let mut filter = build_filter(truth[0], spec);
    let first = loose_measurement(truth[0].t_j2000_s, &gnss[0]);
    filter.update_loose(&first).expect("initial loose update");
    let mut history = if recorded {
        Some(FusionRtsHistoryBuilder::from_filter(&filter).expect("history"))
    } else {
        None
    };
    let mut positions = vec![filter.state().nominal.position_ecef_m];
    let mut velocities = vec![filter.state().nominal.velocity_ecef_mps];
    let mut covariances = vec![filter.state().covariance.clone()];
    let increments = truth_increments(truth);
    let sequence = simulate_imu_samples_from_increments(
        &increments,
        spec,
        ImuSimulationOptions {
            seed: SEED ^ 0xa11c_e5e1_5e1d_0f11,
            ..ImuSimulationOptions::default()
        },
    )
    .expect("simulated IMU");

    for (step, sample) in sequence.samples.into_iter().enumerate() {
        let epoch_index = step + 1;
        if let Some(history) = &mut history {
            filter
                .propagate_recorded(sample, history)
                .expect("recorded propagate");
        } else {
            filter.propagate(sample).expect("propagate");
        }
        let in_outage = outage
            .as_ref()
            .is_some_and(|range| range.contains(&epoch_index));
        if !in_outage {
            let measurement = loose_measurement(truth[epoch_index].t_j2000_s, &gnss[epoch_index]);
            if let Some(history) = &mut history {
                filter
                    .update_loose_recorded(&measurement, history)
                    .expect("recorded loose update");
            } else {
                filter.update_loose(&measurement).expect("loose update");
            }
        }
        positions.push(filter.state().nominal.position_ecef_m);
        velocities.push(filter.state().nominal.velocity_ecef_mps);
        covariances.push(filter.state().covariance.clone());
    }

    FusionRun {
        positions,
        velocities,
        covariances,
        history: history.map(|history| history.finish().expect("finished history")),
    }
}

fn static_truth(steps: usize, dt_s: f64) -> Vec<NavState> {
    (0..=steps)
        .map(|idx| {
            NavState::new(
                start_j2000_s() + idx as f64 * dt_s,
                [WGS84_A_M, 0.0, 0.0],
                [0.0; 3],
                IDENTITY_3,
            )
            .expect("static truth")
        })
        .collect()
}

fn straight_vehicle_truth(steps: usize, dt_s: f64) -> Vec<NavState> {
    let velocity = [12.0, 0.0, 0.0];
    (0..=steps)
        .map(|idx| {
            let t = idx as f64 * dt_s;
            NavState::new(
                start_j2000_s() + t,
                [WGS84_A_M + velocity[0] * t, 0.0, 0.0],
                velocity,
                IDENTITY_3,
            )
            .expect("straight truth")
        })
        .collect()
}

fn run_stationary_constraint_case(
    truth: &[NavState],
    samples: Vec<sidereon_core::inertial::ImuSample>,
    spec: ImuSpec,
    enable_updates: bool,
) -> InertialFilter {
    let mut diagonal = [1.0e-8; ERROR_STATE_DIMENSION_15];
    for axis in 0..3 {
        diagonal[ERROR_POSITION_INDEX + axis] = 1.0;
        diagonal[ERROR_VELOCITY_INDEX + axis] = 1.0;
        diagonal[ERROR_ACCEL_BIAS_INDEX + axis] = 0.05 * 0.05;
        diagonal[ERROR_GYRO_BIAS_INDEX + axis] = 0.003 * 0.003;
    }
    let mut config = LooseCouplingConfig::default();
    if enable_updates {
        config.stationary_updates = Some(StationaryUpdateConfig {
            detector: StationaryDetectorConfig {
                window_len: 3,
                max_specific_force_norm_error_mps2: 0.08,
                max_body_rate_wrt_ecef_norm_rps: 0.003,
            },
            zero_velocity_sigma_mps: 0.015,
            zero_angular_rate_sigma_rps: 0.00008,
        });
    }
    let mut filter = direct_filter(truth[0], spec, diagonal, config);
    let mut applied_updates = 0usize;
    for sample in samples {
        filter.propagate(sample).expect("stationary propagate");
        if enable_updates {
            let update = filter.update_stationary().expect("stationary update");
            applied_updates += usize::from(update.is_some());
        }
    }
    if enable_updates {
        assert!(
            applied_updates + 2 >= truth.len() - 1,
            "stationary updates applied {applied_updates}"
        );
    }
    filter
}

fn run_nhc_case(
    truth: &[NavState],
    samples: Vec<sidereon_core::inertial::ImuSample>,
    spec: ImuSpec,
    enable_updates: bool,
) -> InertialFilter {
    let mut nominal = truth[0];
    nominal.velocity_ecef_mps[1] = 1.5;
    nominal.velocity_ecef_mps[2] = -0.4;
    let mut diagonal = [1.0e-8; ERROR_STATE_DIMENSION_15];
    for axis in 0..3 {
        diagonal[ERROR_POSITION_INDEX + axis] = 1.0;
        diagonal[ERROR_VELOCITY_INDEX + axis] = 4.0;
    }
    let mut config = LooseCouplingConfig::default();
    if enable_updates {
        config.non_holonomic = Some(NonHolonomicConstraintConfig {
            lateral_velocity_sigma_mps: 0.03,
            vertical_velocity_sigma_mps: 0.03,
            min_speed_mps: 2.0,
            max_body_rate_wrt_ecef_norm_rps: 0.01,
        });
    }
    let mut filter = direct_filter(nominal, spec, diagonal, config);
    let mut applied_updates = 0usize;
    for sample in samples {
        filter.propagate(sample).expect("vehicle propagate");
        if enable_updates {
            let update = filter.update_non_holonomic().expect("NHC update");
            applied_updates += usize::from(update.is_some());
        }
    }
    if enable_updates {
        assert_eq!(applied_updates, truth.len() - 1);
    }
    filter
}

fn direct_filter(
    nominal: NavState,
    spec: ImuSpec,
    diagonal: [f64; ERROR_STATE_DIMENSION_15],
    loose: LooseCouplingConfig,
) -> InertialFilter {
    let state = InsFilterState::from_diagonal(nominal, ErrorStateLayout::Fifteen, &diagonal)
        .expect("filter state");
    let mut config = InertialFilterConfig::new(spec).expect("filter config");
    config.loose = loose;
    InertialFilter::with_config(state, config).expect("filter")
}

fn run_tight_fusion(
    truth: &[NavState],
    set: &SyntheticObservationSet,
    source: &SyntheticKeplerSource,
    low_sat_window: std::ops::Range<usize>,
    apply_low_sat_updates: bool,
    imu_seed: u64,
) -> FusionRun {
    let spec = ImuSpec::preset(ImuGrade::Tactical);
    let initial = prior_sampled_initial_nominal(truth[0], imu_seed ^ 0xc17c_5eed_19c0_0f11);
    let mut filter = tight_filter(initial, spec);
    let first = tight_epoch_for_index(set, 0, false);
    filter
        .update_tight(source, &first)
        .expect("initial tight update");
    let mut positions = vec![filter.state().nominal.position_ecef_m];
    let mut velocities = vec![filter.state().nominal.velocity_ecef_mps];
    let mut covariances = vec![filter.state().covariance.clone()];
    let increments = truth_increments(truth);
    let sequence = simulate_imu_samples_from_increments(
        &increments,
        spec,
        ImuSimulationOptions {
            seed: imu_seed ^ 0x1a5e_d1f1_f7c0_5eed,
            ..ImuSimulationOptions::default()
        },
    )
    .expect("simulated IMU");

    for (step, sample) in sequence.samples.into_iter().enumerate() {
        let epoch_index = step + 1;
        filter.propagate(sample).expect("propagate");
        let low_sat = low_sat_window.contains(&epoch_index);
        if !low_sat || apply_low_sat_updates {
            let epoch = tight_epoch_for_index(set, epoch_index, low_sat);
            let update = filter.update_tight(source, &epoch).expect("tight update");
            if low_sat {
                assert_eq!(update.rows, 6);
                assert!(update.applied);
            }
        }
        positions.push(filter.state().nominal.position_ecef_m);
        velocities.push(filter.state().nominal.velocity_ecef_mps);
        covariances.push(filter.state().covariance.clone());
    }

    FusionRun {
        positions,
        velocities,
        covariances,
        history: None,
    }
}

fn loose_filter(initial_truth: NavState, spec: ImuSpec) -> InertialFilter {
    let state = InsFilterState::from_diagonal(
        initial_nominal(initial_truth),
        ErrorStateLayout::Fifteen,
        &initial_covariance_diagonal(),
    )
    .expect("filter state");
    let mut config = InertialFilterConfig::new(spec).expect("filter config");
    config.loose = LooseCouplingConfig {
        measurement_reweighting: Some(IggIiiMeasurementReweighting::standard()),
        prediction_adaptation: Some(YangPredictionAdaptiveFactor::standard()),
        ..LooseCouplingConfig::default()
    };
    InertialFilter::with_config(state, config).expect("filter")
}

fn tight_filter(initial_nominal: NavState, spec: ImuSpec) -> InertialFilter {
    let state = InsFilterState::from_diagonal(
        initial_nominal,
        ErrorStateLayout::Fifteen,
        &tight_initial_covariance_diagonal(),
    )
    .expect("filter state");
    let mut config = InertialFilterConfig::new(spec).expect("filter config");
    config.tight = TightCouplingConfig {
        initial_clock_bias_variance_m2: 2.5e3,
        initial_clock_drift_variance_m2_s2: 0.25,
        clock_bias_random_walk_m2_s: 0.02,
        clock_drift_random_walk_m2_s3: 1.0e-4,
        ..TightCouplingConfig::default()
    };
    InertialFilter::with_config(state, config).expect("filter")
}

fn prior_sampled_initial_nominal(truth: NavState, seed: u64) -> NavState {
    let mut rng = SplitMix64::new(seed);
    let mut position = truth.position_ecef_m;
    let mut velocity = truth.velocity_ecef_mps;
    for axis in 0..3 {
        position[axis] += 4.0 * rng.standard_normal();
        velocity[axis] += 0.5 * rng.standard_normal();
    }
    NavState::new(
        truth.t_j2000_s,
        position,
        velocity,
        truth.attitude_body_to_ecef,
    )
    .expect("sampled initial nominal")
}

fn initial_nominal(truth: NavState) -> NavState {
    NavState::new(
        truth.t_j2000_s,
        [
            truth.position_ecef_m[0] + 3.5,
            truth.position_ecef_m[1] - 2.5,
            truth.position_ecef_m[2] + 1.5,
        ],
        [
            truth.velocity_ecef_mps[0] + 0.18,
            truth.velocity_ecef_mps[1] - 0.12,
            truth.velocity_ecef_mps[2] + 0.07,
        ],
        truth.attitude_body_to_ecef,
    )
    .expect("initial nominal")
}

fn initial_covariance_diagonal() -> [f64; ERROR_STATE_DIMENSION_15] {
    let mut diagonal = [1.0e-8; ERROR_STATE_DIMENSION_15];
    for axis in 0..3 {
        diagonal[ERROR_POSITION_INDEX + axis] = 36.0;
        diagonal[ERROR_VELOCITY_INDEX + axis] = 0.75 * 0.75;
    }
    diagonal
}

fn tight_initial_covariance_diagonal() -> [f64; ERROR_STATE_DIMENSION_15] {
    let mut diagonal = [1.0e-8; ERROR_STATE_DIMENSION_15];
    for axis in 0..3 {
        diagonal[ERROR_POSITION_INDEX + axis] = 16.0;
        diagonal[ERROR_VELOCITY_INDEX + axis] = 0.5 * 0.5;
    }
    diagonal
}

fn loose_measurement(t_j2000_s: f64, solution: &GnssEpochSolution) -> GnssFixMeasurement {
    GnssFixMeasurement::position_velocity(
        t_j2000_s,
        solution.position_ecef_m,
        solution.velocity_ecef_m_s,
        solution.covariance.clone(),
        solution.satellites_used,
    )
    .expect("loose measurement")
}

fn tight_epoch_for_index(
    set: &SyntheticObservationSet,
    epoch_index: usize,
    low_sat: bool,
) -> TightGnssEpoch {
    let start = set.observations.epoch_offsets[epoch_index];
    let end = set.observations.epoch_offsets[epoch_index + 1];
    let limit = if low_sat { (start + 3).min(end) } else { end };
    let observations = (start..limit)
        .map(|index| {
            let range_rate_m_s = doppler_to_range_rate(
                set.observations.doppler_hz[index],
                set.observations.carrier_hz[index],
            )
            .expect("Doppler to range rate");
            TightGnssObservation {
                satellite_id: set.observations.satellite_id[index],
                pseudorange_m: set.observations.pseudorange_m[index],
                pseudorange_sigma_m: TIGHT_CODE_SIGMA_M,
                range_rate: Some(TightRangeRateObservation {
                    measured_range_rate_m_s: range_rate_m_s,
                    sigma_m_s: TIGHT_RANGE_RATE_SIGMA_M_S,
                    satellite_clock_drift_m_s: 0.0,
                }),
                carrier_phase: None,
                ionosphere_delay_m: 0.0,
                troposphere_delay_m: 0.0,
            }
        })
        .collect();
    TightGnssEpoch::new(set.receiver_truth[epoch_index].t_rx_j2000_s, observations)
        .expect("tight epoch")
}

fn truth_increments(truth: &[NavState]) -> Vec<sidereon_core::CorrectedImuIncrement> {
    truth
        .windows(2)
        .map(|pair| true_imu_increment_between(&pair[0], &pair[1]).expect("truth increment"))
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct SppTimeContext {
    second_of_day_s: f64,
    day_of_year: f64,
}

fn spp_time_context(t_j2000_s: f64) -> SppTimeContext {
    let whole = t_j2000_s.floor() as i64;
    let frac = t_j2000_s - whole as f64;
    let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(whole);
    let second = second as f64 + frac;
    SppTimeContext {
        second_of_day_s: second_of_day(hour as i32, minute as i32, second),
        day_of_year: day_of_year(
            year as i32,
            month as i32,
            day as i32,
            hour as i32,
            minute as i32,
            second,
        ),
    }
}

fn rms_position_error<I>(samples: I) -> f64
where
    I: Iterator<Item = ([f64; 3], [f64; 3])>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for (estimated, truth) in samples {
        let error = distance3(estimated, truth);
        sum += error * error;
        count += 1;
    }
    (sum / count as f64).sqrt()
}

fn distance3(a: [f64; 3], b: [f64; 3]) -> f64 {
    norm3(sub3(a, b))
}

fn assert_vec3_close(actual: [f64; 3], expected: [f64; 3], tolerance: f64) {
    for axis in 0..3 {
        assert!(
            (actual[axis] - expected[axis]).abs() <= tolerance,
            "axis {axis}: actual {:.17e}, expected {:.17e}, tolerance {:.17e}",
            actual[axis],
            expected[axis],
            tolerance
        );
    }
}

fn outlier_epoch_count(epoch_count: usize) -> usize {
    (0..epoch_count)
        .filter(|epoch_index| {
            *epoch_index >= OUTLIER_OFFSET
                && (*epoch_index - OUTLIER_OFFSET).is_multiple_of(OUTLIER_STRIDE)
        })
        .count()
}

fn nees_position_velocity(
    position_ecef_m: &[f64; 3],
    velocity_ecef_m_s: &[f64; 3],
    covariance: &[Vec<f64>],
    truth: &NavState,
) -> f64 {
    let error = DVector::from_row_slice(&[
        position_ecef_m[0] - truth.position_ecef_m[0],
        position_ecef_m[1] - truth.position_ecef_m[1],
        position_ecef_m[2] - truth.position_ecef_m[2],
        velocity_ecef_m_s[0] - truth.velocity_ecef_mps[0],
        velocity_ecef_m_s[1] - truth.velocity_ecef_mps[1],
        velocity_ecef_m_s[2] - truth.velocity_ecef_mps[2],
    ]);
    let indices = [
        ERROR_POSITION_INDEX,
        ERROR_POSITION_INDEX + 1,
        ERROR_POSITION_INDEX + 2,
        ERROR_VELOCITY_INDEX,
        ERROR_VELOCITY_INDEX + 1,
        ERROR_VELOCITY_INDEX + 2,
    ];
    let covariance = DMatrix::from_fn(6, 6, |row, col| covariance[indices[row]][indices[col]]);
    let solved = covariance
        .cholesky()
        .expect("NEES covariance")
        .solve(&error);
    error.dot(&solved)
}

fn nees_position(position_ecef_m: &[f64; 3], covariance: &[Vec<f64>], truth: &NavState) -> f64 {
    let error = DVector::from_row_slice(&[
        position_ecef_m[0] - truth.position_ecef_m[0],
        position_ecef_m[1] - truth.position_ecef_m[1],
        position_ecef_m[2] - truth.position_ecef_m[2],
    ]);
    let covariance = DMatrix::from_fn(3, 3, |row, col| {
        covariance[ERROR_POSITION_INDEX + row][ERROR_POSITION_INDEX + col]
    });
    let solved = covariance
        .cholesky()
        .expect("position NEES covariance")
        .solve(&error);
    error.dot(&solved)
}

fn position_trace(covariance: &[Vec<f64>]) -> f64 {
    (0..3)
        .map(|axis| covariance[ERROR_POSITION_INDEX + axis][ERROR_POSITION_INDEX + axis])
        .sum()
}

fn chi_square_band(dimension: usize, samples: usize) -> std::ops::RangeInclusive<f64> {
    chi_square_band_with_probability(0.025, 0.975, dimension, samples)
}

fn chi_square_band_with_probability(
    lower_probability: f64,
    upper_probability: f64,
    dimension: usize,
    samples: usize,
) -> std::ops::RangeInclusive<f64> {
    let lower = sidereon_core::quality::chi2_inv(lower_probability, dimension)
        .expect("chi-square lower")
        / samples as f64;
    let upper = sidereon_core::quality::chi2_inv(upper_probability, dimension)
        .expect("chi-square upper")
        / samples as f64;
    lower..=upper
}
