use std::collections::BTreeMap;
use std::path::PathBuf;

use nalgebra::{Matrix3, Vector3};
use sidereon_core::dgnss::{solve_position, CodeObservation};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::fusion::GnssFixStatus;
use sidereon_core::positioning::{
    solve_static_reference_station_rinex, spp_inputs_from_rinex_obs, RinexSppOptions,
    StaticReferenceCarrierRinexOptions, StaticReferenceFixStatus, StaticReferenceModeError,
    StaticReferenceModeReport, StaticReferenceModeStatus, StaticReferenceStationError,
    StaticReferenceStationMode, StaticReferenceStationRinexOptions,
};
use sidereon_core::rinex::observations::RinexObs;
use sidereon_core::rtk::BaselineReferenceSelection;
use sidereon_core::rtk_filter::{CycleSlipPolicy, IntegerStatus};
use sidereon_core::rtk_filter::{
    DynamicsModel, FixedSolveOpts, FloatSolveOpts, MeasModel, ResidualValidationOpts, RtkArcConfig,
    RtkArcPreprocessing, RtkRinexArcOptions, RtkStaticArcConfig, SearchOpts, StochasticModel,
    UpdateOpts, ValidatedFixedSolveOpts,
};

const WTZR_MARKER_M: [f64; 3] = [4075580.3111, 931854.0543, 4801568.2808];
const WTZZ_MARKER_M: [f64; 3] = [4075579.1913, 931853.3696, 4801569.1897];

fn fixture_path(parts: &[&str]) -> PathBuf {
    parts.iter().fold(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        |path, part| path.join(part),
    )
}

fn load_text(parts: &[&str]) -> String {
    std::fs::read_to_string(fixture_path(parts)).expect("read fixture")
}

fn load_sp3() -> Sp3 {
    let text = load_text(&["sp3", "GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3"]);
    Sp3::parse(text.as_bytes()).expect("parse SP3")
}

fn load_wettzell_obs() -> (RinexObs, RinexObs) {
    let reference_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZR OBS");
    let rover_obs = RinexObs::parse(&load_text(&[
        "obs",
        "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse WTZZ OBS");
    (reference_obs, rover_obs)
}

fn arp_position(marker_m: [f64; 3], obs: &RinexObs) -> [f64; 3] {
    let [height_m, east_m, north_m] = obs
        .header()
        .antenna_delta_hen_m
        .expect("RINEX antenna delta H/E/N");
    assert_eq!(east_m, 0.0);
    assert_eq!(north_m, 0.0);
    let inv_norm = 1.0
        / (marker_m[0] * marker_m[0] + marker_m[1] * marker_m[1] + marker_m[2] * marker_m[2])
            .sqrt();
    [
        marker_m[0] + marker_m[0] * inv_norm * height_m,
        marker_m[1] + marker_m[1] * inv_norm * height_m,
        marker_m[2] + marker_m[2] * inv_norm * height_m,
    ]
}

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

fn simple_model() -> MeasModel {
    MeasModel {
        code_sigma_m: 2.0,
        phase_sigma_m: 0.01,
        sagnac: true,
        stochastic: StochasticModel::Simple {
            elevation_weighting: true,
        },
    }
}

fn carrier_options(
    reference_position_m: [f64; 3],
    max_epochs: usize,
) -> StaticReferenceCarrierRinexOptions {
    let mut arc_options = RtkRinexArcOptions::gps_l1_c();
    arc_options.max_epochs = Some(max_epochs);
    arc_options.include_prediction_time = false;
    let update_opts = UpdateOpts {
        hold_sigma_m: 1.0e-4,
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 10,
        process_noise_baseline_sigma_m: 0.0,
        dynamics_model: DynamicsModel::ConstantPosition,
        float_only_systems: Vec::new(),
        report_residuals: true,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: 3.0,
        },
    };
    let preprocessing = RtkArcPreprocessing {
        cycle_slip: Some(CycleSlipPolicy::SplitArc),
        hatch_window_cap: None,
        elevation_mask_deg: None,
    };
    let arc = RtkArcConfig::new(
        reference_position_m,
        BaselineReferenceSelection::Auto,
        simple_model(),
        30.0,
        30.0,
        [0.0, 0.0, 0.0],
        BTreeMap::new(),
        BTreeMap::new(),
        update_opts,
        preprocessing,
    );
    let opts = ValidatedFixedSolveOpts {
        float: FloatSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
        },
        fixed: FixedSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 10,
            ratio_threshold: 3.0,
            partial_ambiguity_resolution: true,
            partial_min_ambiguities: 4,
        },
        residual: ResidualValidationOpts {
            threshold_sigma: None,
            max_exclusions: 0,
        },
    };
    StaticReferenceCarrierRinexOptions::new(arc_options, RtkStaticArcConfig::new(arc, opts))
}

fn assert_covariance_psd(covariance: [[f64; 3]; 3]) {
    let matrix = Matrix3::from_row_slice(&[
        covariance[0][0],
        covariance[0][1],
        covariance[0][2],
        covariance[1][0],
        covariance[1][1],
        covariance[1][2],
        covariance[2][0],
        covariance[2][1],
        covariance[2][2],
    ]);
    let eigen = matrix.symmetric_eigen();
    let scale = covariance
        .iter()
        .flatten()
        .fold(1.0_f64, |acc, value| acc.max(value.abs()));
    for value in eigen.eigenvalues.iter() {
        assert!(
            *value >= -1.0e-12 * scale,
            "negative covariance eigenvalue {value}"
        );
    }
}

fn assert_error_inside_three_sigma(error_m: [f64; 3], covariance_m2: [[f64; 3]; 3]) {
    let error_norm_m = norm3(error_m);
    assert!(error_norm_m > 0.0);
    let matrix = Matrix3::from_row_slice(&[
        covariance_m2[0][0],
        covariance_m2[0][1],
        covariance_m2[0][2],
        covariance_m2[1][0],
        covariance_m2[1][1],
        covariance_m2[1][2],
        covariance_m2[2][0],
        covariance_m2[2][1],
        covariance_m2[2][2],
    ]);
    let unit = Vector3::new(error_m[0], error_m[1], error_m[2]) / error_norm_m;
    let variance_along_error_m2 = unit.dot(&(matrix * unit));
    assert!(variance_along_error_m2 > 0.0);
    let three_sigma_m = 3.0 * variance_along_error_m2.sqrt();
    assert!(
        error_norm_m <= three_sigma_m,
        "error {error_norm_m:.6} m exceeds 3-sigma bound {three_sigma_m:.6} m"
    );
    assert!(
        three_sigma_m < 0.30,
        "covariance is too loose for the WTZR/WTZZ fast-static oracle: {three_sigma_m:.6} m"
    );
}

#[test]
fn wettzell_reference_station_static_returns_coordinate_and_covariance() {
    let sp3 = load_sp3();
    let (mut reference_obs, mut rover_obs) = load_wettzell_obs();
    reference_obs.epochs.truncate(24);
    rover_obs.epochs.truncate(24);
    let reference_arp_m = arp_position(WTZR_MARKER_M, &reference_obs);
    let rover_arp_truth_m = arp_position(WTZZ_MARKER_M, &rover_obs);
    let code_options = RinexSppOptions::default_for(&rover_obs).expect("default signal policy");
    let options = StaticReferenceStationRinexOptions::code_and_carrier(
        code_options,
        carrier_options(reference_arp_m, 24),
        true,
    );

    let solution = solve_static_reference_station_rinex(
        &sp3,
        &reference_obs,
        &rover_obs,
        reference_arp_m,
        &options,
    )
    .expect("static reference-station solve");

    let position_m = solution.position.as_array();
    let error_m = sub3(position_m, rover_arp_truth_m);
    let error_norm_m = norm3(error_m);
    assert_eq!(solution.mode, StaticReferenceStationMode::CarrierFixed);
    assert_eq!(solution.fix_status, StaticReferenceFixStatus::CarrierFixed);
    assert!(error_norm_m < 0.08, "WTZZ ARP error {error_norm_m:.6} m");
    assert_covariance_psd(solution.covariance.position_ecef_m2);
    assert_error_inside_three_sigma(error_m, solution.covariance.position_ecef_m2);

    let carrier = solution.carrier_solution.as_ref().expect("carrier detail");
    assert_eq!(carrier.integer_status, IntegerStatus::Fixed);
    assert!(carrier.integer_ratio.expect("integer ratio") > 3.0);
    assert_eq!(
        carrier.rtk_solution.references,
        BTreeMap::from([("G".to_string(), "G30".to_string())])
    );
    assert_eq!(carrier.diagnostics.len(), 24);
    let carrier_report = solution
        .mode_reports
        .iter()
        .find(|report| report.mode == StaticReferenceStationMode::CarrierFixed)
        .expect("carrier report");
    assert_eq!(
        carrier_report.used_measurements,
        carrier
            .rtk_solution
            .fixed_solution
            .fixed_solution
            .n_observations
    );

    let code = solution.code_solution.as_ref().expect("code detail");
    assert!(code.static_solution.is_some());
    assert_eq!(code.diagnostics.len(), 24);
    assert!(solution
        .mode_reports
        .iter()
        .all(|report| report.status == StaticReferenceModeStatus::Solved));

    eprintln!(
        "WTZR/WTZZ static reference station: mode={:?} pos={:?} error={:.6} m 3sigma_along={:.6} m",
        solution.mode,
        position_m,
        error_norm_m,
        three_sigma_along(error_m, solution.covariance.position_ecef_m2)
    );
}

#[test]
fn code_solution_outprioritizes_unfixed_carrier_float() {
    let sp3 = load_sp3();
    let (mut reference_obs, mut rover_obs) = load_wettzell_obs();
    reference_obs.epochs.truncate(24);
    rover_obs.epochs.truncate(24);
    let reference_arp_m = arp_position(WTZR_MARKER_M, &reference_obs);
    let code_options = RinexSppOptions::default_for(&rover_obs).expect("default signal policy");
    let mut carrier = carrier_options(reference_arp_m, 24);
    carrier.static_config.opts.fixed.ratio_threshold = 1.0e12;
    carrier.static_config.arc.update_opts.search.ratio_threshold = 1.0e12;
    let options = StaticReferenceStationRinexOptions::code_and_carrier(code_options, carrier, true);

    let solution = solve_static_reference_station_rinex(
        &sp3,
        &reference_obs,
        &rover_obs,
        reference_arp_m,
        &options,
    )
    .expect("static reference-station solve");

    assert_eq!(solution.mode, StaticReferenceStationMode::CodeDgnss);
    assert_eq!(solution.fix_status, StaticReferenceFixStatus::CodeDgnss);
    assert!(solution.code_solution.is_some());
    let carrier = solution.carrier_solution.as_ref().expect("carrier detail");
    assert_eq!(carrier.integer_status, IntegerStatus::NotFixed);
    assert!(solution.mode_reports.iter().any(|report| report.mode
        == StaticReferenceStationMode::CarrierFloat
        && report.status == StaticReferenceModeStatus::Solved));
}

#[test]
fn all_modes_failed_display_lists_mode_errors_without_debug_dump() {
    let error = StaticReferenceStationError::AllModesFailed {
        mode_reports: vec![
            StaticReferenceModeReport {
                mode: StaticReferenceStationMode::CodeDgnss,
                status: StaticReferenceModeStatus::Failed,
                used_epochs: 0,
                skipped_epochs: 0,
                used_measurements: 0,
                error: Some(StaticReferenceModeError::NoMatchedCodeEpochs),
            },
            StaticReferenceModeReport {
                mode: StaticReferenceStationMode::CarrierFixed,
                status: StaticReferenceModeStatus::Failed,
                used_epochs: 0,
                skipped_epochs: 0,
                used_measurements: 0,
                error: Some(StaticReferenceModeError::CarrierSolve {
                    reason: "singular geometry".to_string(),
                }),
            },
        ],
    };
    let display = error.to_string();
    assert!(display.contains("code-DGNSS: no matched epochs"));
    assert!(display.contains("carrier-fixed: carrier RTK solve failed: singular geometry"));
    assert!(!display.contains("StaticReferenceModeReport"));
}

#[test]
fn static_reference_fix_status_converts_to_loose_gnss_fix_status() {
    assert_eq!(
        GnssFixStatus::from(StaticReferenceFixStatus::CodeDgnss),
        GnssFixStatus::Single
    );
    assert_eq!(
        GnssFixStatus::from(StaticReferenceFixStatus::CarrierFloat),
        GnssFixStatus::Float
    );
    assert_eq!(
        GnssFixStatus::from(StaticReferenceFixStatus::CarrierFixed),
        GnssFixStatus::Fixed
    );
}

#[test]
fn single_epoch_code_dgnss_matches_per_epoch_solver_bits() {
    let sp3 = load_sp3();
    let (mut reference_obs, mut rover_obs) = load_wettzell_obs();
    reference_obs.epochs.truncate(1);
    rover_obs.epochs.truncate(1);
    let reference_arp_m = arp_position(WTZR_MARKER_M, &reference_obs);
    let code_options = RinexSppOptions::default_for(&rover_obs).expect("default signal policy");

    let options = StaticReferenceStationRinexOptions::code_only(code_options.clone(), true);
    let wrapped = solve_static_reference_station_rinex(
        &sp3,
        &reference_obs,
        &rover_obs,
        reference_arp_m,
        &options,
    )
    .expect("single-epoch code wrapper");

    let reference_epoch = &spp_inputs_from_rinex_obs(&reference_obs, &sp3, &code_options)
        .expect("reference SPP inputs")[0];
    let rover_epoch =
        &spp_inputs_from_rinex_obs(&rover_obs, &sp3, &code_options).expect("rover SPP inputs")[0];
    let base_observations = reference_epoch
        .inputs
        .observations
        .iter()
        .map(|obs| CodeObservation::new(obs.satellite_id.to_string(), obs.pseudorange_m))
        .collect::<Vec<_>>();
    let rover_observations = rover_epoch
        .inputs
        .observations
        .iter()
        .map(|obs| CodeObservation::new(obs.satellite_id.to_string(), obs.pseudorange_m))
        .collect::<Vec<_>>();
    let direct = solve_position(
        &sp3,
        reference_arp_m,
        &base_observations,
        &rover_observations,
        rover_epoch.inputs.clone(),
        true,
    )
    .expect("direct DGNSS");

    assert_eq!(
        wrapped.position.as_array().map(f64::to_bits),
        direct.solution.position.as_array().map(f64::to_bits)
    );
    assert_eq!(
        wrapped
            .covariance
            .position_ecef_m2
            .map(|row| row.map(f64::to_bits)),
        direct
            .solution
            .position_covariance
            .ecef_m2
            .map(|row| row.map(f64::to_bits))
    );
    assert_eq!(wrapped.mode, StaticReferenceStationMode::CodeDgnss);
    assert!(wrapped
        .code_solution
        .as_ref()
        .expect("code detail")
        .static_solution
        .is_none());
}

fn three_sigma_along(error_m: [f64; 3], covariance_m2: [[f64; 3]; 3]) -> f64 {
    let error_norm_m = norm3(error_m);
    let matrix = Matrix3::from_row_slice(&[
        covariance_m2[0][0],
        covariance_m2[0][1],
        covariance_m2[0][2],
        covariance_m2[1][0],
        covariance_m2[1][1],
        covariance_m2[1][2],
        covariance_m2[2][0],
        covariance_m2[2][1],
        covariance_m2[2][2],
    ]);
    let unit = Vector3::new(error_m[0], error_m[1], error_m[2]) / error_norm_m;
    3.0 * unit.dot(&(matrix * unit)).sqrt()
}
