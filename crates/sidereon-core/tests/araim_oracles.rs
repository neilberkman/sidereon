#![cfg(sidereon_repo_tests)]

use sidereon_core::araim::ism::{ConstellationIsm, Ism, SatelliteIsm, SatelliteIsmModel};
use sidereon_core::araim::{
    araim, enumerate_fault_modes, reliability_araim, reliability_design, wtest_noncentrality,
    wtest_noncentrality_components, AraimError, AraimGeometry, AraimRow, IntegrityAllocation,
    ProtectionModel, RangeReliabilityRow, ReliabilityOptions,
};
use sidereon_core::dop::LineOfSight;
use sidereon_core::frame::Wgs84Geodetic;
use sidereon_core::quality::QualityError;
use sidereon_core::{GnssSatelliteId, GnssSystem};

const FIXTURE: &str = include_str!("fixtures/araim_oracles.json");
const MAAST_MATRIX: &str = include_str!("fixtures/araim_maast_matrix.json");
const APP_D: [(GnssSystem, u8, [f64; 3], f64, f64); 10] = [
    (
        GnssSystem::Gps,
        1,
        [0.0225, 0.9951, -0.0966],
        3.8865,
        3.5740,
    ),
    (
        GnssSystem::Gps,
        2,
        [0.6750, -0.6900, -0.2612],
        1.4377,
        1.1252,
    ),
    (
        GnssSystem::Gps,
        3,
        [0.0723, -0.6601, -0.7477],
        0.8604,
        0.5479,
    ),
    (
        GnssSystem::Gps,
        4,
        [-0.9398, 0.2553, -0.2269],
        1.6383,
        1.3258,
    ),
    (
        GnssSystem::Gps,
        5,
        [-0.5907, -0.7539, -0.2877],
        1.3229,
        1.0104,
    ),
    (
        GnssSystem::Galileo,
        1,
        [-0.3236, -0.0354, -0.9455],
        0.8434,
        0.5309,
    ),
    (
        GnssSystem::Galileo,
        2,
        [-0.6748, 0.4356, -0.5957],
        0.8963,
        0.5838,
    ),
    (
        GnssSystem::Galileo,
        3,
        [0.0938, -0.7004, -0.7075],
        0.8669,
        0.5544,
    ),
    (
        GnssSystem::Galileo,
        4,
        [0.5571, 0.3088, -0.7709],
        0.8573,
        0.5448,
    ),
    (
        GnssSystem::Galileo,
        5,
        [0.6622, 0.6958, -0.2780],
        1.3616,
        1.0491,
    ),
];

#[test]
fn appendix_d_v3_0_agrees_with_add_roots_and_maast_intermediates() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    let result = araim(
        &app_d_geometry(false),
        &app_d_ism(1.0e-4),
        &IntegrityAllocation::lpv_200(),
    )
    .expect("Appendix D solve");
    let oracle = &fixture["appendix_d_v3_0"];
    assert_eq!(
        result.fault_modes.len() - 1,
        oracle["mode_count"].as_u64().unwrap() as usize
    );
    assert_close(result.emt_m, oracle["emt_m"].as_f64().unwrap(), 1.0e-6);
    assert_close(
        result.sigma_acc_v_m,
        oracle["sigma_acc_v_m"].as_f64().unwrap(),
        1.0e-6,
    );
    // Vertical bisection returns within 1e-4 m of the ADD Eq. (31) root. Each
    // horizontal component may be 0.05 m high, giving sqrt(2)*0.05 m in HPL.
    assert_close(
        result.vpl_m,
        oracle["add_roots"]["vpl_m"].as_f64().unwrap(),
        1.0e-4 + oracle["threshold_rounding_bound_m"].as_f64().unwrap(),
    );
    assert_close(
        result.hpl_m,
        oracle["add_roots"]["hpl_m"].as_f64().unwrap(),
        2.0_f64.sqrt() * (5.0e-2 + oracle["threshold_rounding_bound_m"].as_f64().unwrap()),
    );
    // MAAST returns the upper endpoint after at most ten log-space bisections;
    // its published Appendix D PL therefore includes 0..TOL_PL stopping slack.
    assert!(
        oracle["maast_upper_bisection"]["vpl_m"].as_f64().unwrap()
            >= oracle["add_roots"]["vpl_m"].as_f64().unwrap()
    );
}

#[test]
fn maast_two_constellation_synthetic_control_agrees_with_add_roots() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    let oracle = &fixture["edge_cases"]["synthetic_two_constellations"];
    let geometry = synthetic_geometry(22, 2);
    let result = araim(
        &geometry,
        &synthetic_add_ism(&geometry, 2, 1.0e-5, 1.0e-4),
        &IntegrityAllocation::lpv_200(),
    )
    .expect("synthetic two-constellation solve");
    assert_eq!(result.fault_modes.len() - 1, 24);
    let rounding = oracle["threshold_rounding_bound_m"].as_f64().unwrap();
    assert_close(
        result.vpl_m,
        oracle["vpl_m"].as_f64().unwrap(),
        1.0e-4 + rounding,
    );
    assert_close(
        result.hpl_m,
        oracle["hpl_m"].as_f64().unwrap(),
        2.0_f64.sqrt() * (5.0e-2 + rounding),
    );
    assert_close(result.emt_m, oracle["emt_m"].as_f64().unwrap(), 1.0e-6);
}

#[test]
fn maast_matrix_n06_c1_nominal() {
    compare_maast_matrix_case("n06_c1_nominal");
}

#[test]
fn maast_matrix_n12_c2_consolidation() {
    compare_maast_matrix_case("n12_c2_consolidation");
}

#[test]
fn maast_matrix_n10_c1_zero_probability() {
    compare_maast_matrix_case("n10_c1_zero_probability");
}

#[test]
fn maast_matrix_n12_c2_low_prior() {
    compare_maast_matrix_case("n12_c2_low_prior");
}

#[test]
fn maast_matrix_n18_c3_milestone3() {
    compare_maast_matrix_case("n18_c3_milestone3");
}

#[test]
fn maast_matrix_n24_c2_varied() {
    compare_maast_matrix_case("n24_c2_varied");
}

#[test]
fn maast_matrix_n30_c3_low_prior() {
    compare_maast_matrix_case("n30_c3_low_prior");
}

#[test]
fn constellation_fault_multi_event_modes_are_enumerated() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    let oracle = &fixture["edge_cases"]["synthetic_fifteen_satellites_three_constellations"];
    let geometry = synthetic_geometry(15, 3);
    let result = araim(
        &geometry,
        &synthetic_add_ism(&geometry, 3, 1.0e-5, 1.0e-4),
        &IntegrityAllocation::lpv_200(),
    )
    .unwrap();
    assert_eq!(result.fault_modes.len() - 1, 19);
    let rounding = oracle["threshold_rounding_bound_m"].as_f64().unwrap();
    assert_close(
        result.vpl_m,
        oracle["vpl_m"].as_f64().unwrap(),
        1.0e-4 + rounding,
    );
    assert_close(
        result.hpl_m,
        oracle["hpl_m"].as_f64().unwrap(),
        2.0_f64.sqrt() * (5.0e-2 + rounding),
    );
}

#[test]
fn unmonitored_mass_reduces_budget_instead_of_forcing_unavailability() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    let oracle = &fixture["edge_cases"]["single_constellation_unmonitored_mass"];
    let mut allocation = IntegrityAllocation::lpv_200();
    allocation.p_threshold_unmonitored = 8.0e-8;
    let result = araim(
        &app_d_geometry(true),
        &app_d_one_constellation_ism(9.0e-8),
        &allocation,
    )
    .unwrap();
    assert!(result.available);
    let rounding = oracle["threshold_rounding_bound_m"].as_f64().unwrap();
    assert_close(
        result.vpl_m,
        oracle["vpl_m"].as_f64().unwrap(),
        1.0e-4 + rounding,
    );
    assert_close(
        result.hpl_m,
        oracle["hpl_m"].as_f64().unwrap(),
        2.0_f64.sqrt() * (5.0e-2 + rounding),
    );
}

#[test]
fn first_order_selection_uses_exact_mode_probabilities() {
    let result = araim(
        &app_d_geometry(true),
        &app_d_one_constellation_ism(1.0e-8),
        &IntegrityAllocation::lpv_200(),
    )
    .unwrap();
    assert_eq!(result.fault_modes.len() - 1, 10);
    let enumerated = enumerate_fault_modes(
        &app_d_geometry(true),
        &app_d_one_constellation_ism(1.0e-8),
        &IntegrityAllocation::lpv_200(),
    )
    .unwrap();
    assert_eq!(result.fault_modes[0].prior, enumerated[0].prior);
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    assert_close(
        result.p_unmonitored,
        fixture["edge_cases"]["low_constellation_prior"]["p_unmonitored"]
            .as_f64()
            .unwrap(),
        5.0e-14,
    );
    assert_eq!(result.emt_m, 0.0);
    // ADD Eq. (2): P_sat (1-P_sat)^9 (1-P_const).
    assert_close(
        result.fault_modes[1].prior,
        fixture["edge_cases"]["low_constellation_prior"]["first_order_satellite_prior"]
            .as_f64()
            .unwrap(),
        5.0e-15,
    );
}

#[test]
fn fault_mode_count_includes_only_observable_modes() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    let oracle = &fixture["edge_cases"]["unobservable_constellation"];
    let result = araim(
        &app_d_geometry(true),
        &app_d_one_constellation_ism(1.0e-8),
        &IntegrityAllocation::lpv_200(),
    )
    .unwrap();
    let rounding = oracle["threshold_rounding_bound_m"].as_f64().unwrap();
    assert_close(
        result.vpl_m,
        oracle["vpl_m"].as_f64().unwrap(),
        1.0e-4 + rounding,
    );
    assert_close(
        result.hpl_m,
        oracle["hpl_m"].as_f64().unwrap(),
        2.0_f64.sqrt() * (5.0e-2 + rounding),
    );
}

#[test]
fn rnp_horizontal_only_allocation_is_valid() {
    let allocation = IntegrityAllocation {
        phmi_total: 1e-7,
        phmi_vert: 0.0,
        phmi_hor: 1e-7,
        pfa_vert: 0.0,
        pfa_hor: 1e-6,
        p_threshold_unmonitored: 4e-8,
        p_emt: 1e-5,
        max_fault_order: 2,
    };
    assert!(araim(&app_d_geometry(false), &app_d_ism(1.0e-4), &allocation).is_ok());
}

#[test]
fn default_sigma_is_a_documented_local_model_not_add_appendix_a() {
    // At zenith the local variance is 0.3^2 + 0.3^2 = 0.18 m^2; ADD v3.0
    // Eq. (3), Eqs. (57)-(58) give 0.27847436379754925 m^2.
    let local = 0.3_f64.powi(2) + 0.3_f64.powi(2);
    assert_close(local, 0.18, 1.0e-15);
    assert!((local - 0.27847436379754925).abs() > 0.09);
}

#[test]
fn add_v3_0_sigma_constructor_matches_appendix_a_worked_values() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    for case in fixture["sigma_model"].as_array().unwrap() {
        let elevation_deg = case["elevation_deg"].as_f64().unwrap();
        let model = SatelliteIsmModel::add_v3_0(
            GnssSystem::Gps,
            elevation_deg.to_radians(),
            0.75,
            0.5,
            0.5,
            1.0e-5,
        )
        .unwrap();
        let local = model.effective_sigma_int_m.unwrap().powi(2) - 0.75_f64.powi(2);
        assert_close(local, case["v3_0_variance_m2"].as_f64().unwrap(), 2.0e-15);
    }
}

#[test]
fn add_v3_0_uses_the_galileo_appendix_a_table() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    let table = fixture["galileo_user_sigma_model"].as_array().unwrap();
    for case in table {
        assert_galileo_user_sigma(
            case["elevation_deg"].as_f64().unwrap(),
            case["user_sigma_m"].as_f64().unwrap(),
        );
    }
    for pair in table.windows(2) {
        let low_elevation = pair[0]["elevation_deg"].as_f64().unwrap();
        let high_elevation = pair[1]["elevation_deg"].as_f64().unwrap();
        let expected = 0.5
            * (pair[0]["user_sigma_m"].as_f64().unwrap()
                + pair[1]["user_sigma_m"].as_f64().unwrap());
        assert_galileo_user_sigma(0.5 * (low_elevation + high_elevation), expected);
    }
    for elevation in [f64::NAN, 0.0, 2.0_f64.to_radians(), 91.0_f64.to_radians()] {
        assert_eq!(
            SatelliteIsmModel::add_v3_0(GnssSystem::Galileo, elevation, 0.0, 0.0, 0.0, 1.0e-5),
            Err(AraimError::InvalidIsm)
        );
    }
    assert_eq!(
        SatelliteIsmModel::add_v3_0(
            GnssSystem::BeiDou,
            30.0_f64.to_radians(),
            0.75,
            0.5,
            0.5,
            1.0e-5,
        ),
        Err(AraimError::UnsupportedConstellation(GnssSystem::BeiDou))
    );
}

fn assert_galileo_user_sigma(elevation_deg: f64, expected: f64) {
    let elevation_rad = elevation_deg.to_radians();
    let model =
        SatelliteIsmModel::add_v3_0(GnssSystem::Galileo, elevation_rad, 0.0, 0.0, 0.0, 1.0e-5)
            .unwrap();
    let tropo = 0.12 * 1.001 / libm::sqrt(0.002001 + libm::sin(elevation_rad).powi(2));
    let recovered_user = (model.effective_sigma_int_m.unwrap().powi(2) - tropo.powi(2)).sqrt();
    let rounding_bound = 8.0 * f64::EPSILON * expected.abs().max(1.0);
    assert_close(recovered_user, expected, rounding_bound);
}

#[test]
fn large_marginal_priors_form_a_valid_probability_partition() {
    let geometry = synthetic_geometry(5, 1);
    let ism = synthetic_ism(1, 0.3, 0.0);
    let mut allocation = IntegrityAllocation::lpv_200();
    allocation.max_fault_order = 1;
    allocation.p_threshold_unmonitored = 0.0;
    let modes = enumerate_fault_modes(&geometry, &ism, &allocation).unwrap();
    assert_close(modes[0].prior, 0.16807, 1.0e-15);
    assert_close(modes[1].prior, 0.07203, 1.0e-15);
}

#[test]
fn negative_wtest_noncentrality_is_rejected() {
    assert_eq!(
        wtest_noncentrality_components(0.9, 0.9),
        Err(QualityError::InvalidReliabilityParameter)
    );
}

#[test]
fn unmonitored_threshold_must_be_below_phmi() {
    let mut allocation = IntegrityAllocation::lpv_200();
    allocation.p_threshold_unmonitored = 1.5e-7;
    assert_eq!(
        araim(&app_d_geometry(false), &app_d_ism(1e-4), &allocation),
        Err(AraimError::InvalidAllocation)
    );
}

#[test]
fn default_sigma_rejects_nonpositive_elevation_per_row() {
    for elevation in [0.0, (-10.0_f64).to_radians()] {
        let mut geometry = app_d_geometry(false);
        geometry.rows[0].elevation_rad = elevation;
        assert!(araim(&geometry, &app_d_ism(1e-4), &IntegrityAllocation::lpv_200()).is_err());
    }
}

#[test]
fn varied_prior_two_constellation_stop_test_matches_reference() {
    let result = araim(
        &synthetic_geometry(24, 2),
        &synthetic_ism(2, 1.2e-5, 1e-4),
        &IntegrityAllocation::lpv_200(),
    )
    .unwrap();
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    let oracle = &fixture["edge_cases"]["varied_prior_two_constellations"];
    let rounding = oracle["threshold_rounding_bound_m"].as_f64().unwrap();
    assert!(result.available);
    assert_eq!(
        result.fault_modes.len(),
        oracle["mode_count"].as_u64().unwrap() as usize
    );
    assert_close(
        result.p_unmonitored,
        oracle["p_unmonitored"].as_f64().unwrap(),
        64.0 * f64::EPSILON,
    );
    assert_close(
        result.vpl_m,
        oracle["vpl_m"].as_f64().unwrap(),
        1.0e-4 + rounding,
    );
    assert_close(
        result.hpl_m,
        oracle["hpl_m"].as_f64().unwrap(),
        2.0_f64.sqrt() * (5.0e-2 + rounding),
    );
}

#[test]
fn reliability_matches_scipy_and_hand_derived_equal_weight_design() {
    let lambda = wtest_noncentrality(1.0e-3, 0.2).unwrap();
    assert_close(lambda, 17.074646805189236, 1.0e-14);
    let rows = vec![
        RangeReliabilityRow {
            id: "a".into(),
            design_row: vec![1.0],
            sigma_m: 1.0,
        },
        RangeReliabilityRow {
            id: "b".into(),
            design_row: vec![1.0],
            sigma_m: 1.0,
        },
    ];
    let report = reliability_design(&rows, &ReliabilityOptions::default()).unwrap();
    for row in report.per_observation {
        assert_close(row.redundancy, 0.5, 1.0e-15);
        assert_close(row.mdb_m.unwrap(), (2.0 * lambda).sqrt(), 1.0e-14);
        assert_close(row.bias_to_noise.unwrap(), lambda.sqrt(), 1.0e-14);
    }
}

#[test]
fn reliability_noncentral_chi_square_fixture_covers_multiple_tail_allocations() {
    let fixture: serde_json::Value = serde_json::from_str(FIXTURE).expect("valid fixture");
    for case in fixture["reliability"]["noncentral_chi_square_cases"]
        .as_array()
        .unwrap()
    {
        let alpha = case["alpha"].as_f64().unwrap();
        let beta = case["beta"].as_f64().unwrap();
        let components = wtest_noncentrality_components(alpha, beta).unwrap();
        assert_close(components.delta0, case["delta0"].as_f64().unwrap(), 4.0e-15);
        assert_close(
            components.lambda0,
            case["lambda0"].as_f64().unwrap(),
            2.0e-14,
        );
        let cdf = case["ncx2_cdf_df1"].as_f64().unwrap();
        let survival = case["ncx2_sf_df1"].as_f64().unwrap();
        assert_close(cdf + survival, 1.0, 2.0e-15);
        assert!(case["chi_square_critical"].as_f64().unwrap() > 0.0);
    }
}

#[test]
fn public_ism_constructors_and_protection_model_match_closed_forms() {
    let row = AraimRow {
        id: sat(GnssSystem::Gps, 1),
        line_of_sight: LineOfSight::new(1.0, 0.0, 0.0),
        system: GnssSystem::Gps,
        elevation_rad: core::f64::consts::FRAC_PI_2,
    };
    let default = Ism::new(
        vec![ConstellationIsm::new(
            GnssSystem::Gps,
            0.0,
            SatelliteIsmModel::new(0.75, 0.50, 0.25, 0.0),
        )],
        vec![],
    );
    assert_close(
        default.sigma_int_m(&row).unwrap(),
        libm::sqrt(0.75_f64.powi(2) + 0.18),
        1.0e-15,
    );
    assert_close(
        default.sigma_acc_m(&row).unwrap(),
        libm::sqrt(0.50_f64.powi(2) + 0.18),
        1.0e-15,
    );

    let direct = Ism::new(
        vec![ConstellationIsm::new(
            GnssSystem::Gps,
            0.0,
            SatelliteIsmModel::new_with_effective_sigmas(
                0.75,
                0.50,
                0.25,
                0.0,
                1.2345678901234567,
                0.9876543210123456,
            ),
        )],
        vec![SatelliteIsm::new_with_effective_sigmas(
            row.id,
            0.75,
            0.50,
            0.25,
            0.0,
            1.345678901234567,
            0.8765432101234567,
        )],
    );
    assert_eq!(direct.sigma_int_m(&row).unwrap(), 1.345678901234567);
    assert_eq!(direct.sigma_acc_m(&row).unwrap(), 0.8765432101234567);

    let allocation = IntegrityAllocation::lpv_200();
    assert_eq!(allocation.phmi_total, 1.0e-7);
    assert_eq!(allocation.phmi_vert, 9.8e-8);
    assert_eq!(allocation.phmi_hor, 2.0e-9);
    assert_eq!(allocation.pfa_vert, 3.9e-6);
    assert_eq!(allocation.pfa_hor, 9.0e-8);
    assert_eq!(allocation.p_threshold_unmonitored, 8.0e-8);
    assert_eq!(allocation.p_emt, 1.0e-5);
}

#[test]
fn reliability_araim_matches_the_same_hand_formed_design() {
    let geometry = synthetic_geometry(8, 1);
    let ism = synthetic_ism(1, 0.0, 0.0);
    let mut options = ReliabilityOptions::default();
    options.lambda0_override = Some(17.075);
    let araim_report = reliability_araim(&geometry, &ism, &options).unwrap();
    let design_rows: Vec<_> = geometry
        .rows
        .iter()
        .map(|row| RangeReliabilityRow {
            id: row.id.to_string(),
            design_row: vec![
                -row.line_of_sight.e_x,
                -row.line_of_sight.e_y,
                -row.line_of_sight.e_z,
                1.0,
            ],
            sigma_m: 1.4,
        })
        .collect();
    let design_report = reliability_design(&design_rows, &options).unwrap();
    assert_close(
        araim_report.summary.sum_redundancy,
        design_report.summary.sum_redundancy,
        2.0e-14,
    );
    for (actual, expected) in araim_report
        .per_observation
        .iter()
        .zip(&design_report.per_observation)
    {
        assert_close(actual.redundancy, expected.redundancy, 2.0e-14);
        assert_close(actual.mdb_m.unwrap(), expected.mdb_m.unwrap(), 2.0e-13);
        assert_close(
            actual.bias_to_noise.unwrap(),
            expected.bias_to_noise.unwrap(),
            2.0e-13,
        );
    }
}

#[test]
fn invalid_and_degenerate_inputs_are_typed_errors() {
    let mut too_few = synthetic_geometry(3, 1);
    assert_eq!(
        araim(
            &too_few,
            &synthetic_ism(1, 0.0, 0.0),
            &IntegrityAllocation::lpv_200()
        ),
        Err(AraimError::InsufficientGeometry)
    );
    too_few.rows[0].elevation_rad = f64::NAN;
    assert_eq!(
        araim(
            &too_few,
            &synthetic_ism(1, 0.0, 0.0),
            &IntegrityAllocation::lpv_200()
        ),
        Err(AraimError::InsufficientGeometry)
    );
    let mut singular = synthetic_geometry(6, 1);
    for row in &mut singular.rows {
        row.line_of_sight = LineOfSight::new(1.0, 0.0, 0.0);
    }
    assert_eq!(
        araim(
            &singular,
            &synthetic_ism(1, 0.0, 0.0),
            &IntegrityAllocation::lpv_200()
        ),
        Err(AraimError::InsufficientGeometry)
    );
    let bad_ism = Ism::new(
        vec![ConstellationIsm::new(
            GnssSystem::Gps,
            f64::INFINITY,
            SatelliteIsmModel::new(1.0, 1.0, 0.0, 0.0),
        )],
        vec![],
    );
    assert_eq!(
        araim(
            &synthetic_geometry(6, 1),
            &bad_ism,
            &IntegrityAllocation::lpv_200()
        ),
        Err(AraimError::InvalidIsm)
    );

    let mut infinite_geometry = synthetic_geometry(6, 1);
    infinite_geometry.rows[0].line_of_sight = LineOfSight::new(f64::INFINITY, 0.0, 0.0);
    assert_eq!(
        araim(
            &infinite_geometry,
            &synthetic_ism(1, 0.0, 0.0),
            &IntegrityAllocation::lpv_200()
        ),
        Err(AraimError::InsufficientGeometry)
    );
    let mut bad_allocation = IntegrityAllocation::lpv_200();
    bad_allocation.pfa_hor = f64::NAN;
    assert_eq!(
        araim(
            &synthetic_geometry(6, 1),
            &synthetic_ism(1, 0.0, 0.0),
            &bad_allocation
        ),
        Err(AraimError::InvalidAllocation)
    );
}

#[test]
fn singular_fault_subset_is_unmonitorable_without_invalidating_full_geometry() {
    let horizontal = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ];
    let mut rows: Vec<_> = horizontal
        .into_iter()
        .enumerate()
        .map(|(idx, los)| AraimRow {
            id: sat(GnssSystem::Gps, idx as u8 + 1),
            line_of_sight: LineOfSight::new(los[0], los[1], los[2]),
            system: GnssSystem::Gps,
            elevation_rad: 0.1,
        })
        .collect();
    rows.push(AraimRow {
        id: sat(GnssSystem::Gps, 5),
        line_of_sight: LineOfSight::new(0.0, 0.0, 1.0),
        system: GnssSystem::Gps,
        elevation_rad: core::f64::consts::FRAC_PI_2,
    });
    let geometry = AraimGeometry {
        rows,
        receiver: receiver(),
        clock_systems: vec![GnssSystem::Gps],
        ut1_degraded: None,
    };
    let mut allocation = IntegrityAllocation::lpv_200();
    allocation.p_threshold_unmonitored = 1.0e-12;
    let result = araim(&geometry, &synthetic_ism(1, 1.0e-9, 0.0), &allocation).unwrap();
    let unmonitorable = result
        .fault_modes
        .iter()
        .find(|mode| mode.excluded == vec![sat(GnssSystem::Gps, 5)])
        .expect("unmonitorable fault remains represented");
    assert!(!unmonitorable.monitorable);
    assert!(result.p_unmonitored >= 1.0e-9);
}

#[test]
fn public_enumeration_honors_the_unmonitored_threshold() {
    let geometry = synthetic_geometry(10, 1);
    let mut allocation = IntegrityAllocation::lpv_200();
    allocation.max_fault_order = 2;
    let modes = enumerate_fault_modes(&geometry, &app_d_one_constellation_ism(1.0e-8), &allocation)
        .unwrap();
    assert_eq!(modes.len(), 11);
}

#[test]
fn selection_has_no_first_order_exemption() {
    let geometry = synthetic_geometry(10, 2);
    let modes = enumerate_fault_modes(
        &geometry,
        &synthetic_ism(2, 1.0e-9, 1.0e-4),
        &IntegrityAllocation::lpv_200(),
    )
    .unwrap();
    assert_eq!(modes.len(), 3);
}

#[test]
fn equal_probability_modes_follow_maast_stable_event_order() {
    let geometry = synthetic_geometry(6, 2);
    let mut allocation = IntegrityAllocation::lpv_200();
    allocation.max_fault_order = 1;
    allocation.p_threshold_unmonitored = 0.0;
    let modes =
        enumerate_fault_modes(&geometry, &synthetic_ism(2, 1.0e-5, 0.0), &allocation).unwrap();
    let actual: Vec<_> = modes
        .iter()
        .skip(1)
        .map(|mode| mode.excluded.as_slice())
        .collect();
    let expected: Vec<_> = geometry
        .rows
        .iter()
        .map(|row| core::slice::from_ref(&row.id))
        .collect();
    assert_eq!(actual, expected);
}

fn compare_maast_matrix_case(name: &str) {
    let fixture: serde_json::Value = serde_json::from_str(MAAST_MATRIX).expect("valid matrix");
    let case = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["name"] == name)
        .expect("named matrix case");
    let n = case["n_sat"].as_u64().unwrap() as usize;
    let nc = case["n_const"].as_u64().unwrap() as usize;
    let p_sat = case["p_sat"].as_f64().unwrap();
    let p_const = case["p_const"].as_f64().unwrap();
    let ura = case["ura_m"].as_f64().unwrap();
    let ure = case["ure_m"].as_f64().unwrap();
    let bias = case["bias_m"].as_f64().unwrap();
    let geometry = synthetic_geometry(n, nc);
    let cint = json_f64_vec(&case["cint_m2"]);
    let cacc = json_f64_vec(&case["cacc_m2"]);
    let systems = [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou];
    let defaults = systems[..nc]
        .iter()
        .map(|&system| {
            ConstellationIsm::new(
                system,
                p_const,
                SatelliteIsmModel::new(ura, ure, bias, p_sat),
            )
        })
        .collect();
    let satellites = geometry
        .rows
        .iter()
        .enumerate()
        .map(|(idx, row)| {
            SatelliteIsm::new_with_effective_sigmas(
                row.id,
                ura,
                ure,
                bias,
                p_sat,
                libm::sqrt(cint[idx]),
                libm::sqrt(cacc[idx]),
            )
        })
        .collect();
    let mut allocation = IntegrityAllocation::lpv_200();
    allocation.max_fault_order = case["max_fault_order"].as_u64().unwrap() as usize;
    allocation.p_threshold_unmonitored = case["p_threshold_unmonitored"].as_f64().unwrap();
    let result =
        araim(&geometry, &Ism::new(defaults, satellites), &allocation).expect("matrix solve");

    let add = &case["add_reference"];
    if !add.is_null() {
        // These values are the direct ADD calculation. Gain formation, the
        // differently ordered variance dot product, and the final square root
        // contribute at most 64 rounded operations on this path.
        let sigma_reference = add["sigma_acc_v_m"].as_f64().unwrap();
        assert_close(
            result.sigma_acc_v_m,
            sigma_reference,
            64.0 * f64::EPSILON * sigma_reference.abs().max(1.0),
        );
        assert_close(result.emt_m, add["emt_m"].as_f64().unwrap(), 1.0e-12);
        let vpl_root = add["vpl_root_m"].as_f64().unwrap();
        let hpl_root = add["hpl_root_m"].as_f64().unwrap();
        // The vertical search returns `lo + 1e-4`, so it is conservative by
        // at most 1e-4 m. Each horizontal coordinate returns `lo + 0.05`;
        // monotonicity of hypot bounds the combined excess by sqrt(2)*0.05.
        assert!(result.vpl_m >= vpl_root && result.vpl_m - vpl_root <= 1.0e-4);
        assert!(result.hpl_m >= hpl_root && result.hpl_m - hpl_root <= 2.0_f64.sqrt() * 5.0e-2);
        assert_close(
            result.p_unmonitored,
            add["p_unmonitored"].as_f64().unwrap(),
            5.0e-15,
        );
        assert_eq!(
            result.fault_modes.len(),
            add["mode_count"].as_u64().unwrap() as usize
        );

        // ADD v3.0 only requires each PL root to be within TOL_PL=0.05 m.
        // MAAST returns the other bisection endpoint; both endpoints therefore
        // conform, and horizontal hypot combines two such roots.
        if !matches!(name, "n24_c2_varied" | "n12_c2_consolidation") {
            assert!((result.vpl_m - case["vpl_m"].as_f64().unwrap()).abs() <= 5.0e-2);
            assert!(
                (result.hpl_m - case["hpl_m"].as_f64().unwrap()).abs() <= 2.0_f64.sqrt() * 5.0e-2
            );
        }
    }

    if add.is_null() {
        assert_close(
            result.sigma_acc_v_m,
            case["sigma_acc_v_m"].as_f64().unwrap(),
            1.0e-9,
        );
        assert_close(result.emt_m, case["emt_m"].as_f64().unwrap(), 1.0e-6);
    }
    // MAAST returns an upper bisection endpoint: one vertical TOL_PL and
    // sqrt(2)*TOL_PL for the two horizontal roots bound the comparison.
    if add.is_null() {
        assert_close(result.vpl_m, case["vpl_m"].as_f64().unwrap(), 5.0e-2);
        assert_close(
            result.hpl_m,
            case["hpl_m"].as_f64().unwrap(),
            2.0_f64.sqrt() * 5.0e-2,
        );
        assert_close(
            result.p_unmonitored,
            case["p_unmonitored"].as_f64().unwrap(),
            5.0e-12,
        );
    }

    let probabilities = json_f64_vec(&case["mode_probability"]);
    let excluded = json_matrix(&case["excluded"]);
    let thresholds = json_matrix(&case["threshold_enu_m"]);
    let add_thresholds = (!add.is_null()).then(|| json_matrix(&add["threshold_enu_m"]));
    let sigmas = json_matrix(&case["sigma_int_enu_m"]);
    let biases = json_matrix(&case["bias_enu_m"]);
    let monitored: Vec<_> = result
        .fault_modes
        .iter()
        .filter(|mode| mode.monitorable)
        .collect();
    if matches!(name, "n24_c2_varied" | "n12_c2_consolidation") {
        // n24 differs because MAAST applies FC per small mode rather than to
        // the Eq. (16) sum. The consolidation case has matching selected modes
        // and Eq. (17) priors: its PL/EMT difference is MAAST's optional
        // Eqs. (35)-(37) second pass, removing saturated allocations and
        // recomputing thresholds. The ADD says this loss "can be mitigated";
        // the first-pass roots used here also conform.
        assert_eq!(
            probabilities.len() as i64 - monitored.len() as i64,
            add["maast_mode_count_departure"].as_i64().unwrap()
        );
        assert_close(
            add["maast_emt_departure_m"].as_f64().unwrap(),
            case["emt_m"].as_f64().unwrap() - result.emt_m,
            1.0e-12,
        );
        assert_close(
            add["maast_p_unmonitored_departure"].as_f64().unwrap(),
            case["p_unmonitored"].as_f64().unwrap() - result.p_unmonitored,
            5.0e-15,
        );
        // The public-default root bounds above check its documented search
        // resolution. MAAST's smaller departures require finer discrimination:
        // mhss::departure_tests resolves the same engine equations at 1e-5 m
        // and asserts both ADD roots and recorded departures against that
        // engine value, not fixture arithmetic against fixture arithmetic.
        if name == "n12_c2_consolidation" {
            for (index, system) in [GnssSystem::Gps, GnssSystem::Galileo].iter().enumerate() {
                let mode = monitored
                    .iter()
                    .find(|mode| {
                        mode.excluded_constellation == Some(*system) && mode.excluded.is_empty()
                    })
                    .unwrap();
                assert_close(
                    mode.prior,
                    add["constellation_priors"][index].as_f64().unwrap(),
                    64.0 * f64::EPSILON * mode.prior,
                );
            }
        }
        for (system, oracle_index, field) in [
            (
                GnssSystem::Gps,
                1,
                "maast_gps_constellation_prior_departure",
            ),
            (
                GnssSystem::Galileo,
                2,
                "maast_galileo_constellation_prior_departure",
            ),
        ] {
            if name != "n24_c2_varied" {
                break;
            }
            let mode = monitored
                .iter()
                .find(|mode| {
                    mode.excluded_constellation == Some(system) && mode.excluded.is_empty()
                })
                .expect("constellation mode");
            assert_close(
                probabilities[oracle_index] - mode.prior,
                add[field].as_f64().unwrap(),
                5.0e-15,
            );
        }
    }
    if name != "n24_c2_varied" {
        assert_eq!(monitored.len(), probabilities.len());
    }
    let mut previous_sid_idx = None;
    for oracle_idx in 0..probabilities.len() {
        let sid_idx = monitored
            .iter()
            .position(|mode| {
                geometry.rows.iter().enumerate().all(|(row_idx, row)| {
                    let removed = mode.excluded.contains(&row.id)
                        || mode.excluded_constellation == Some(row.system);
                    removed == (excluded[oracle_idx][row_idx] > 0.5)
                })
            })
            .expect("MAAST mode in sidereon result");
        if name == "n24_c2_varied" {
            assert!(previous_sid_idx.is_none_or(|previous| previous < sid_idx));
            previous_sid_idx = Some(sid_idx);
        }
        let mode = monitored[sid_idx];
        if !matches!(name, "n24_c2_varied" | "n12_c2_consolidation") {
            if oracle_idx == 0 {
                let p0 = (1.0 - p_sat).powi(n as i32) * (1.0 - p_const).powi(nc as i32);
                assert_close(mode.prior, p0, 5.0e-14);
            } else {
                assert_close(mode.prior, probabilities[oracle_idx], 5.0e-14);
            }
        }
        for axis in 0..3 {
            assert_close(mode.sigma_int_enu_m[axis], sigmas[oracle_idx][axis], 1.0e-9);
            assert_close(mode.bias_enu_m[axis], biases[oracle_idx][axis], 1.0e-9);
            // The generated bound is 256*n*eps*kappa(N)*max(1,T): backward
            // error for normal formation/inversion, followed by gain,
            // separation and threshold products. The Octave offset is asserted
            // separately.
            let expected_threshold = add_thresholds
                .as_ref()
                .map_or(thresholds[oracle_idx][axis], |values| values[sid_idx][axis]);
            let threshold_tolerance = if add_thresholds.is_some() {
                add["threshold_rounding_bound_m"].as_f64().unwrap()
            } else {
                1.0e-9
            };
            assert_close(
                mode.threshold_enu_m[axis],
                expected_threshold,
                threshold_tolerance,
            );
        }
    }
    if let Some(reference_thresholds) = add_thresholds {
        let observed = thresholds
            .iter()
            .flatten()
            .zip(reference_thresholds.iter().flatten())
            .map(|(maast, reference)| (maast - reference).abs())
            .fold(0.0_f64, f64::max);
        if let Some(expected_departure) = add["max_maast_threshold_departure_m"].as_f64() {
            assert_close(
                observed,
                expected_departure,
                2.0 * f64::EPSILON * observed.max(1.0),
            );
        }
    }
}

fn json_f64_vec(value: &serde_json::Value) -> Vec<f64> {
    match value.as_array() {
        Some(values) => values.iter().map(|value| value.as_f64().unwrap()).collect(),
        None => vec![value.as_f64().unwrap()],
    }
}

fn json_matrix(value: &serde_json::Value) -> Vec<Vec<f64>> {
    let rows = value.as_array().unwrap();
    if rows.first().is_some_and(serde_json::Value::is_number) {
        return vec![json_f64_vec(value)];
    }
    rows.iter().map(json_f64_vec).collect()
}

fn app_d_geometry(one_constellation: bool) -> AraimGeometry {
    let rows = APP_D
        .iter()
        .enumerate()
        .map(|(idx, &(system, prn, los, _, _))| {
            let actual = if one_constellation {
                GnssSystem::Gps
            } else {
                system
            };
            AraimRow {
                id: sat(
                    actual,
                    if one_constellation {
                        idx as u8 + 1
                    } else {
                        prn
                    },
                ),
                // At latitude/longitude zero, ECEF [x,y,z] is ENU [up,east,north].
                // ADD G is minus receiver-to-satellite LOS.
                line_of_sight: LineOfSight::new(-los[2], -los[0], -los[1]),
                system: actual,
                elevation_rad: libm::asin(-los[2]),
            }
        })
        .collect();
    AraimGeometry {
        rows,
        receiver: receiver(),
        clock_systems: if one_constellation {
            vec![GnssSystem::Gps]
        } else {
            vec![GnssSystem::Gps, GnssSystem::Galileo]
        },
        ut1_degraded: None,
    }
}

fn app_d_ism(p_const: f64) -> Ism {
    let constellations = [GnssSystem::Gps, GnssSystem::Galileo]
        .into_iter()
        .map(|system| {
            ConstellationIsm::new(
                system,
                p_const,
                SatelliteIsmModel::new(1.0, 1.0, 0.5, 1.0e-5),
            )
        })
        .collect();
    let satellites = APP_D
        .iter()
        .map(|&(system, prn, _, cint, cacc)| {
            SatelliteIsm::new_with_effective_sigmas(
                sat(system, prn),
                0.75,
                0.5,
                0.5,
                1.0e-5,
                cint.sqrt(),
                cacc.sqrt(),
            )
        })
        .collect();
    Ism::new(constellations, satellites)
}

fn app_d_one_constellation_ism(p_const: f64) -> Ism {
    let satellites = APP_D
        .iter()
        .enumerate()
        .map(|(idx, &(_, _, _, cint, cacc))| {
            SatelliteIsm::new_with_effective_sigmas(
                sat(GnssSystem::Gps, idx as u8 + 1),
                0.75,
                0.5,
                0.5,
                1.0e-5,
                libm::sqrt(cint),
                libm::sqrt(cacc),
            )
        })
        .collect();
    Ism::new(
        vec![ConstellationIsm::new(
            GnssSystem::Gps,
            p_const,
            SatelliteIsmModel::new(0.75, 0.5, 0.5, 1.0e-5),
        )],
        satellites,
    )
}

fn synthetic_geometry(n: usize, constellations: usize) -> AraimGeometry {
    let systems = [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou];
    let rows = (0..n)
        .map(|i| {
            let az = (137.50776 * i as f64).to_radians();
            let el = (15.0 + 60.0 * ((0.61803 * i as f64).fract())).to_radians();
            let system = systems[i % constellations];
            AraimRow {
                id: sat(system, (i / constellations + 1) as u8),
                // Synthetic ADD design is [-cos(el)sin(az), -cos(el)cos(az),
                // -sin(el)]. Convert minus that ENU row to ECEF at lat/lon 0.
                line_of_sight: LineOfSight::new(
                    libm::sin(el),
                    libm::cos(el) * libm::sin(az),
                    libm::cos(el) * libm::cos(az),
                ),
                system,
                elevation_rad: el,
            }
        })
        .collect();
    AraimGeometry {
        rows,
        receiver: receiver(),
        clock_systems: systems[..constellations].to_vec(),
        ut1_degraded: None,
    }
}

fn synthetic_ism(constellations: usize, p_sat: f64, p_const: f64) -> Ism {
    let systems = [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou];
    Ism::new(
        systems[..constellations]
            .iter()
            .map(|&system| {
                ConstellationIsm::new(
                    system,
                    p_const,
                    SatelliteIsmModel::new_with_effective_sigmas(0.75, 0.5, 0.75, p_sat, 1.4, 1.1),
                )
            })
            .collect(),
        vec![],
    )
}

fn synthetic_add_ism(
    geometry: &AraimGeometry,
    constellations: usize,
    p_sat: f64,
    p_const: f64,
) -> Ism {
    let systems = [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou];
    let defaults = systems[..constellations]
        .iter()
        .map(|&system| {
            ConstellationIsm::new(
                system,
                p_const,
                SatelliteIsmModel::new(0.75, 0.5, 0.75, p_sat),
            )
        })
        .collect();
    let satellites = geometry
        .rows
        .iter()
        .map(|row| {
            let local = 0.25 + 2.0 * libm::exp(-row.elevation_rad.to_degrees() / 15.0);
            SatelliteIsm::new_with_effective_sigmas(
                row.id,
                0.75,
                0.5,
                0.75,
                p_sat,
                libm::sqrt(0.75_f64.powi(2) + local),
                libm::sqrt(0.5_f64.powi(2) + local),
            )
        })
        .collect();
    Ism::new(defaults, satellites)
}

fn sat(system: GnssSystem, prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(system, prn).unwrap()
}
fn receiver() -> Wgs84Geodetic {
    Wgs84Geodetic::new(0.0, 0.0, 0.0).unwrap()
}
fn assert_close(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "actual={actual:.17e} expected={expected:.17e} tolerance={tolerance:.3e}"
    );
}
