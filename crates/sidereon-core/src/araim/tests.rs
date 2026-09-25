//! ARAIM tests from public MHSS equations and published ARAIM references.
//! External anchors are the LPV-200 allocation from Blanch et al. 2015 and
//! WG-C Milestone 3, the WG-C Reference ADD v3.0 Appendix D numerical example
//! <https://web.stanford.edu/group/scpnt/gpslab/website_files/maast/ARAIM_TSG_Reference_ADD_v3.0.pdf>,
//! hand-worked S-matrix values, closed-form fault-free PL, and analytic fault
//! mass. Appendix D assertions pin ADD Eqs. (31), (32), (38), (42), and the
//! published intermediate values in Eq. (77). Synthetic geometries below are
//! regression cases for mode handling only.

use super::fault_modes::enumerate_fault_modes;
use super::ism::{ConstellationIsm, Ism, SatelliteIsm, SatelliteIsmModel};
use super::protection::{gain_matrix_enu, metric_bias, metric_sigma};
use super::{araim, AraimError, AraimGeometry, AraimRow, IntegrityAllocation};
use crate::astro::math::least_squares::Status;
use crate::astro::math::special::normal_q_inv;
use crate::dop::{ecef_to_enu_rotation, line_of_sight_from_az_el_deg, LineOfSight};
use crate::frame::{ItrfPositionM, Wgs84Geodetic};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::quality::{raim_fde_design, RangeFdeOptions, RangeFdeRow};
use crate::spp::{EphemerisSource, ReceiverSolution, SolutionMetadata};

const INV_SQRT_3: f64 = 0.577_350_269_189_625_8;
const WG_C_SIGMA_URA_M: f64 = 0.75;
const WG_C_SIGMA_URE_M: f64 = 0.5;
const WG_C_B_NOM_M: f64 = 0.75;
const WG_C_EXAMPLE_B_NOM_M: f64 = 0.5;
const WG_C_P_SAT: f64 = 1.0e-5;
const WG_C_P_CONST_GPS: f64 = 0.0;
const WG_C_P_CONST_GAL: f64 = 1.0e-4;
const WG_C_EXAMPLE_P_CONST: f64 = 1.0e-4;
const WG_C_LPV_200_PHMI_TOTAL: f64 = 1.0e-7;
const WG_C_LPV_200_PHMI_VERT: f64 = 9.8e-8;
const WG_C_LPV_200_PHMI_HOR: f64 = 2.0e-9;
const WG_C_LPV_200_PFA_VERT: f64 = 3.9e-6;
const WG_C_LPV_200_PFA_HOR: f64 = 9.0e-8;
const WG_C_LPV_200_PTHRES: f64 = 8.0e-8;
const WG_C_LPV_200_PEMT: f64 = 1.0e-5;
const PL_REFERENCE_TOL_M: f64 = 0.05;
const SIGMA_REFERENCE_TOL_M: f64 = 0.02;
const WG_C_INTERMEDIATE_TOL_M: f64 = 5.0e-4;
const WG_C_ADD_V3_ROWS: [(GnssSystem, u8, [f64; 3], f64, f64); 10] = [
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
fn fault_free_only_reduces_to_closed_form_pl() {
    let geometry = gps_geometry();
    let ism = Ism::new(
        vec![ConstellationIsm::new(
            GnssSystem::Gps,
            0.0,
            SatelliteIsmModel::new(2.0, 1.0, 0.25, 0.0),
        )],
        Vec::new(),
    );
    let allocation = IntegrityAllocation {
        phmi_total: 1.0e-7,
        phmi_vert: WG_C_LPV_200_PHMI_VERT,
        phmi_hor: WG_C_LPV_200_PHMI_HOR,
        pfa_vert: WG_C_LPV_200_PFA_VERT,
        pfa_hor: WG_C_LPV_200_PFA_HOR,
        p_threshold_unmonitored: 0.0,
        p_emt: WG_C_LPV_200_PEMT,
        max_fault_order: 0,
    };

    let result = araim(&geometry, &ism, &allocation).expect("fault-free ARAIM");
    let fault_free = &result.fault_modes[0];
    let k = normal_q_inv(allocation.phmi_vert * 0.5).expect("valid PHMI");
    let expected_vpl = k * fault_free.sigma_int_enu_m[2] + fault_free.bias_enu_m[2];
    assert_abs_diff(result.vpl_m, expected_vpl, 1.0e-9);

    let k_h = normal_q_inv(allocation.phmi_hor * 0.25).expect("valid PHMI");
    let expected_e = k_h * fault_free.sigma_int_enu_m[0] + fault_free.bias_enu_m[0];
    let expected_n = k_h * fault_free.sigma_int_enu_m[1] + fault_free.bias_enu_m[1];
    let expected_hpl = (expected_e * expected_e + expected_n * expected_n).sqrt();
    assert_abs_diff(result.hpl_m, expected_hpl, 1.0e-9);
    assert_eq!(result.emt_m, 0.0);
    assert!(result.available);
    assert!(result.availability);
}

#[test]
fn weight_zeroed_gain_matches_row_deleted_wls() {
    let geometry = gps_geometry_with_extra_row();
    let weights = vec![0.0, 1.0, 1.0, 1.0, 1.0];
    let gain_zeroed = gain_matrix_enu(&geometry, &weights).expect("zeroed gain");

    let deleted_geometry = AraimGeometry {
        rows: geometry.rows[1..].to_vec(),
        receiver: geometry.receiver,
        clock_systems: geometry.clock_systems.clone(),
        ut1_degraded: None,
    };
    let deleted_weights = vec![1.0; deleted_geometry.rows.len()];
    let gain_deleted = gain_matrix_enu(&deleted_geometry, &deleted_weights).expect("deleted gain");

    for coord in 0..3 {
        assert_eq!(gain_zeroed.enu_rows[coord][0], 0.0);
        for idx in 1..geometry.rows.len() {
            assert_abs_diff(
                gain_zeroed.enu_rows[coord][idx],
                gain_deleted.enu_rows[coord][idx - 1],
                2.0e-12,
            );
        }
    }

    let residuals = [0.0, 0.4, -0.2, 0.7, -0.1];
    let enu_from_gain = [
        dot(&gain_zeroed.enu_rows[0], &residuals),
        dot(&gain_zeroed.enu_rows[1], &residuals),
        dot(&gain_zeroed.enu_rows[2], &residuals),
    ];
    let fde_rows: Vec<RangeFdeRow> = geometry.rows[1..]
        .iter()
        .zip(residuals[1..].iter())
        .map(|(row, &residual_m)| RangeFdeRow {
            id: row.id.to_string(),
            residual_m,
            design_row: vec![
                -row.line_of_sight.e_x,
                -row.line_of_sight.e_y,
                -row.line_of_sight.e_z,
                1.0,
            ],
            weight: 1.0,
        })
        .collect();
    let fit = raim_fde_design(
        &fde_rows,
        &RangeFdeOptions {
            max_exclusions: 0,
            ..RangeFdeOptions::default()
        },
    )
    .expect("row-deleted WLS");
    let r = ecef_to_enu_rotation(geometry.receiver.lat_rad, geometry.receiver.lon_rad);
    let dx = &fit.state_correction;
    let enu_from_wls = [
        r[0][0] * dx[0] + r[0][1] * dx[1] + r[0][2] * dx[2],
        r[1][0] * dx[0] + r[1][1] * dx[1] + r[1][2] * dx[2],
        r[2][0] * dx[0] + r[2][1] * dx[1] + r[2][2] * dx[2],
    ];
    for coord in 0..3 {
        assert_abs_diff(enu_from_gain[coord], enu_from_wls[coord], 2.0e-12);
    }
}

#[test]
fn fault_modes_are_prior_ordered_and_pinned() {
    let geometry = mixed_geometry_for_enumeration();
    let ism = Ism::new(
        vec![
            ConstellationIsm::new(
                GnssSystem::Gps,
                1.0e-3,
                SatelliteIsmModel::new(2.0, 1.0, 0.0, 0.01),
            ),
            ConstellationIsm::new(
                GnssSystem::Galileo,
                2.0e-3,
                SatelliteIsmModel::new(2.0, 1.0, 0.0, 0.03),
            ),
        ],
        vec![SatelliteIsm::new(
            sat(GnssSystem::Gps, 2),
            2.0,
            1.0,
            0.0,
            0.02,
        )],
    );
    let allocation = IntegrityAllocation {
        phmi_total: 1.0e-7,
        phmi_vert: WG_C_LPV_200_PHMI_VERT,
        phmi_hor: WG_C_LPV_200_PHMI_HOR,
        pfa_vert: WG_C_LPV_200_PFA_VERT,
        pfa_hor: WG_C_LPV_200_PFA_HOR,
        p_threshold_unmonitored: 0.0,
        p_emt: WG_C_LPV_200_PEMT,
        max_fault_order: 2,
    };

    let modes = enumerate_fault_modes(&geometry, &ism, &allocation).expect("fault modes");
    assert_eq!(modes.len(), 9);
    assert_eq!(modes[0].excluded, Vec::<GnssSatelliteId>::new());
    assert_eq!(modes[1].excluded, vec![sat(GnssSystem::Gps, 1)]);
    assert_eq!(modes[2].excluded, vec![sat(GnssSystem::Gps, 2)]);
    assert_eq!(modes[3].excluded, vec![sat(GnssSystem::Galileo, 1)]);
    assert_eq!(modes[4].excluded_constellation, Some(GnssSystem::Gps));
    assert_eq!(modes[5].excluded_constellation, Some(GnssSystem::Galileo));
    assert_eq!(
        modes[6].excluded,
        vec![sat(GnssSystem::Gps, 2), sat(GnssSystem::Galileo, 1)]
    );
    assert_eq!(
        modes[7].excluded,
        vec![sat(GnssSystem::Gps, 1), sat(GnssSystem::Galileo, 1)]
    );
    assert_eq!(
        modes[8].excluded,
        vec![sat(GnssSystem::Gps, 1), sat(GnssSystem::Gps, 2)]
    );
    assert_abs_diff(modes[6].prior, 6.0e-4, 0.0);
    assert_abs_diff(modes[7].prior, 3.0e-4, 0.0);
    assert_abs_diff(modes[8].prior, 2.0e-4, 0.0);
}

#[test]
fn constellation_fault_modes_drop_excluded_clock_and_stay_monitorable() {
    let geometry = reference_gps_galileo_geometry();
    let ism = reference_gps_galileo_ism();
    let allocation = IntegrityAllocation::lpv_200();

    let result = araim(&geometry, &ism, &allocation).expect("GPS plus Galileo ARAIM");

    assert!(result.availability);
    assert!(result.hpl_m.is_finite());
    assert!(result.vpl_m.is_finite());
    assert!(result.p_unmonitored <= allocation.p_threshold_unmonitored);
    let galileo_constellation = result
        .fault_modes
        .iter()
        .find(|mode| mode.excluded_constellation == Some(GnssSystem::Galileo))
        .expect("Galileo constellation mode");
    assert!(galileo_constellation.monitorable);
}

#[test]
fn araim_availability_distinguishes_budget_support_from_bad_input() {
    let allocation = IntegrityAllocation::lpv_200();

    let sparse_result =
        araim(&gps_geometry(), &reference_gps_ism(), &allocation).expect("sparse ARAIM result");
    assert!(!sparse_result.available);
    assert!(!sparse_result.availability);
    assert!(sparse_result.hpl_m.is_infinite());
    assert!(sparse_result.vpl_m.is_infinite());
    assert!(sparse_result.p_unmonitored > allocation.p_threshold_unmonitored);

    let full_result = araim(&reference_gps_geometry(), &reference_gps_ism(), &allocation)
        .expect("full ARAIM result");
    assert!(full_result.available);
    assert!(full_result.availability);

    let mut bad_geometry = reference_gps_geometry();
    bad_geometry.rows[0].line_of_sight.e_x = f64::NAN;
    let error = araim(&bad_geometry, &reference_gps_ism(), &allocation)
        .expect_err("NaN LOS remains invalid input");
    assert_eq!(error, AraimError::InsufficientGeometry);
}

#[test]
fn lpv_200_allocation_matches_wg_c_baseline() {
    let allocation = IntegrityAllocation::lpv_200();
    assert_abs_diff(allocation.phmi_total, WG_C_LPV_200_PHMI_TOTAL, 0.0);
    assert_abs_diff(allocation.phmi_vert, WG_C_LPV_200_PHMI_VERT, 0.0);
    assert_abs_diff(allocation.phmi_hor, WG_C_LPV_200_PHMI_HOR, 0.0);
    assert_abs_diff(allocation.pfa_vert, WG_C_LPV_200_PFA_VERT, 0.0);
    assert_abs_diff(allocation.pfa_hor, WG_C_LPV_200_PFA_HOR, 0.0);
    assert_abs_diff(allocation.p_threshold_unmonitored, WG_C_LPV_200_PTHRES, 0.0);
    assert_abs_diff(allocation.p_emt, WG_C_LPV_200_PEMT, 0.0);
    assert!(phmi_split_fits_total(
        allocation.phmi_vert,
        allocation.phmi_hor,
        allocation.phmi_total
    ));
}

#[test]
fn allocation_rejects_phmi_split_above_total() {
    let mut allocation = IntegrityAllocation::lpv_200();
    allocation.phmi_vert = 1.0e-7;
    allocation.phmi_hor = 1.0e-7;
    let error = araim(&reference_gps_geometry(), &reference_gps_ism(), &allocation)
        .expect_err("invalid PHMI split");
    assert_eq!(error, AraimError::InvalidAllocation);
}

#[test]
fn wg_c_add_v3_numerical_example_matches_published_pls() {
    let allocation = IntegrityAllocation::lpv_200();
    let result = araim(
        &wg_c_add_v3_numerical_example_geometry(),
        &wg_c_add_v3_numerical_example_ism(),
        &allocation,
    )
    .expect("WG-C ADD v3.0 Appendix D ARAIM");

    assert!(result.availability);
    let n_fault_modes = result.fault_modes.len() - 1;
    assert_eq!(n_fault_modes, 12);
    let k_fa_3 =
        normal_q_inv(allocation.pfa_vert / (2.0 * n_fault_modes as f64)).expect("valid PFA");
    assert_abs_diff(k_fa_3, 5.1083, WG_C_INTERMEDIATE_TOL_M);
    assert_constellation_intermediates(&result, GnssSystem::Gps, k_fa_3, 2.5760, 1.5307, 2.8935);
    assert_constellation_intermediates(
        &result,
        GnssSystem::Galileo,
        k_fa_3,
        2.5577,
        1.5292,
        2.0875,
    );
    assert_abs_diff(result.vpl_m, 19.2, PL_REFERENCE_TOL_M);
    assert_abs_diff(result.hpl_m, 14.5, PL_REFERENCE_TOL_M);
    assert_abs_diff(result.emt_m, 7.8, PL_REFERENCE_TOL_M);
    assert_abs_diff(result.sigma_acc_v_m, 1.47, SIGMA_REFERENCE_TOL_M);
}

#[test]
fn synthetic_lpv_200_modes_are_regression_pinned() {
    let allocation = IntegrityAllocation::lpv_200();

    let gps_result = araim(&reference_gps_geometry(), &reference_gps_ism(), &allocation)
        .expect("GPS-only LPV-200 ARAIM");
    assert!(gps_result.availability);
    assert_eq!(
        monitored_mode_keys(&gps_result),
        (1..=6)
            .map(|prn| sat(GnssSystem::Gps, prn).to_string())
            .collect::<Vec<_>>()
    );

    let mixed_result = araim(
        &reference_gps_galileo_geometry(),
        &reference_gps_galileo_ism(),
        &allocation,
    )
    .expect("GPS plus Galileo LPV-200 ARAIM");
    assert!(mixed_result.availability);
    let mut expected_mixed = (1..=6)
        .map(|prn| sat(GnssSystem::Gps, prn).to_string())
        .collect::<Vec<_>>();
    expected_mixed.extend((1..=5).map(|prn| sat(GnssSystem::Galileo, prn).to_string()));
    expected_mixed.push("Galileo*".to_string());
    assert_eq!(monitored_mode_keys(&mixed_result), expected_mixed);
}

#[test]
fn hand_worked_s_matrix_and_fault_free_pl_are_pinned() {
    let geometry = gps_geometry();
    let weights = vec![1.0; geometry.rows.len()];
    let gain = gain_matrix_enu(&geometry, &weights).expect("gain matrix");
    let s = 0.75 * INV_SQRT_3;
    let expected = [vec![-s, s, -s, s], vec![-s, s, s, -s], vec![-s, -s, s, s]];
    for (coord, expected_row) in expected.iter().enumerate() {
        for (idx, &expected_value) in expected_row.iter().enumerate() {
            assert_abs_diff(gain.enu_rows[coord][idx], expected_value, 2.0e-15);
        }
    }

    let sigmas = vec![1.0; geometry.rows.len()];
    let biases = vec![0.25; geometry.rows.len()];
    let expected_sigma = 2.0 * s;
    let expected_bias = s;
    let target = IntegrityAllocation::lpv_200().phmi_vert * 0.5;
    let expected_pl = normal_q_inv(target).expect("valid PHMI") * expected_sigma + expected_bias;
    let sigma = metric_sigma(&gain.enu_rows[2], &sigmas);
    let bias = metric_bias(&gain.enu_rows[2], &biases);
    assert_abs_diff(sigma, expected_sigma, 2.0e-15);
    assert_abs_diff(bias, expected_bias, 2.0e-15);
    assert_abs_diff(
        normal_q_inv(target).expect("valid PHMI") * sigma + bias,
        expected_pl,
        2.0e-12,
    );
}

#[test]
fn max_fault_order_one_counts_pair_and_higher_mass_unmonitored() {
    let geometry = gps_geometry_with_extra_row();
    let p_sat = 0.01;
    let ism = Ism::new(
        vec![ConstellationIsm::new(
            GnssSystem::Gps,
            0.0,
            SatelliteIsmModel::new(2.0, 1.0, 0.25, p_sat),
        )],
        Vec::new(),
    );
    let allocation = IntegrityAllocation {
        phmi_total: 1.0e-2,
        phmi_vert: 9.8e-3,
        phmi_hor: 2.0e-4,
        pfa_vert: WG_C_LPV_200_PFA_VERT,
        pfa_hor: WG_C_LPV_200_PFA_HOR,
        p_threshold_unmonitored: 0.01,
        p_emt: WG_C_LPV_200_PEMT,
        max_fault_order: 1,
    };

    let result = araim(&geometry, &ism, &allocation).expect("ARAIM with unmonitored tail");
    let expected_pair_plus =
        10.0 * p_sat.powi(2) + 10.0 * p_sat.powi(3) + 5.0 * p_sat.powi(4) + p_sat.powi(5);
    assert_abs_diff(result.p_unmonitored, expected_pair_plus, 2.0e-16);
}

#[test]
fn receiver_solution_adapter_rebuilds_geometry() {
    let source_geometry = gps_geometry();
    let receiver_ecef = [6_378_137.0, 0.0, 0.0];
    let positions = source_geometry
        .rows
        .iter()
        .map(|row| {
            (
                row.id,
                [
                    receiver_ecef[0] + 20_200_000.0 * row.line_of_sight.e_x,
                    receiver_ecef[1] + 20_200_000.0 * row.line_of_sight.e_y,
                    receiver_ecef[2] + 20_200_000.0 * row.line_of_sight.e_z,
                ],
            )
        })
        .collect();
    let eph = StaticEphemeris { positions };
    let solution = receiver_solution(
        source_geometry.rows.iter().map(|row| row.id).collect(),
        vec![GnssSystem::Gps],
    );

    let rebuilt = AraimGeometry::from_receiver_solution(&solution, &eph, 12_345.0)
        .expect("rebuilt ARAIM geometry");

    assert_eq!(rebuilt.clock_systems, vec![GnssSystem::Gps]);
    assert_eq!(rebuilt.rows.len(), source_geometry.rows.len());
    for (rebuilt, source) in rebuilt.rows.iter().zip(source_geometry.rows.iter()) {
        assert_eq!(rebuilt.id, source.id);
        assert_eq!(rebuilt.system, source.system);
        assert_abs_diff(rebuilt.line_of_sight.e_x, source.line_of_sight.e_x, 2.0e-16);
        assert_abs_diff(rebuilt.line_of_sight.e_y, source.line_of_sight.e_y, 2.0e-16);
        assert_abs_diff(rebuilt.line_of_sight.e_z, source.line_of_sight.e_z, 2.0e-16);
        assert!(rebuilt.elevation_rad.is_finite());
    }
}

/// Like an SSR source outside the UT1 table: one satellite's state is refused
/// under `Strict` and accepted with a reported departure under `Permissive`.
struct Ut1PolicyEphemeris {
    inner: StaticEphemeris,
    satellite: GnssSatelliteId,
    mode: crate::astro::time::ValidityMode,
}

impl EphemerisSource for Ut1PolicyEphemeris {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.try_position_clock_at_j2000_s(sat, t_j2000_s)
            .ok()
            .flatten()
            .map(|state| state.value)
    }

    fn try_position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<Option<crate::astro::time::Validated<crate::spp::PositionClock>>, crate::Error>
    {
        use crate::astro::time::{DegradeReason, Validated, ValidityMode};
        let state = self.inner.position_clock_at_j2000_s(sat, t_j2000_s);
        if sat != self.satellite {
            return Ok(state.map(Validated::ok));
        }
        match self.mode {
            ValidityMode::Strict => Err(crate::Error::Ut1OutsideCoverage(
                DegradeReason::AfterCoverage,
            )),
            ValidityMode::Permissive => {
                Ok(state.map(|state| Validated::degraded(state, DegradeReason::AfterCoverage)))
            }
        }
    }
}

#[test]
fn araim_geometry_keeps_a_ut1_refusal_typed_and_reports_a_departure() {
    use crate::astro::time::{DegradeReason, ValidityMode};
    let source_geometry = gps_geometry_with_extra_row();
    let receiver_ecef = [6_378_137.0, 0.0, 0.0];
    let positions: Vec<_> = source_geometry
        .rows
        .iter()
        .map(|row| {
            (
                row.id,
                [
                    receiver_ecef[0] + 20_200_000.0 * row.line_of_sight.e_x,
                    receiver_ecef[1] + 20_200_000.0 * row.line_of_sight.e_y,
                    receiver_ecef[2] + 20_200_000.0 * row.line_of_sight.e_z,
                ],
            )
        })
        .collect();
    let solution = receiver_solution(
        source_geometry.rows.iter().map(|row| row.id).collect(),
        vec![GnssSystem::Gps],
    );
    let refused = source_geometry.rows[0].id;
    let source = |mode| Ut1PolicyEphemeris {
        inner: StaticEphemeris {
            positions: positions.clone(),
        },
        satellite: refused,
        mode,
    };

    // Strict: the refusal is its own error, not insufficient geometry.
    assert_eq!(
        AraimGeometry::from_receiver_solution(&solution, &source(ValidityMode::Strict), 12_345.0),
        Err(AraimError::Ut1OutsideCoverage(DegradeReason::AfterCoverage))
    );

    // Permissive: every row is built, and the departure is reported.
    let permissive = AraimGeometry::from_receiver_solution(
        &solution,
        &source(ValidityMode::Permissive),
        12_345.0,
    )
    .expect("permissive geometry");
    assert_eq!(permissive.rows.len(), source_geometry.rows.len());
    assert_eq!(permissive.ut1_degraded, Some(DegradeReason::AfterCoverage));

    // In coverage nothing changes.
    let plain = AraimGeometry::from_receiver_solution(
        &solution,
        &StaticEphemeris {
            positions: positions.clone(),
        },
        12_345.0,
    )
    .expect("in-table geometry");
    assert_eq!(plain.ut1_degraded, None);
    assert_eq!(plain.rows, permissive.rows);
}

fn gps_geometry() -> AraimGeometry {
    AraimGeometry {
        rows: vec![
            row(GnssSystem::Gps, 1, [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3]),
            row(GnssSystem::Gps, 2, [INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3]),
            row(GnssSystem::Gps, 3, [-INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3]),
            row(GnssSystem::Gps, 4, [-INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3]),
        ],
        receiver: receiver(),
        clock_systems: vec![GnssSystem::Gps],
        ut1_degraded: None,
    }
}

fn gps_geometry_with_extra_row() -> AraimGeometry {
    let mut geometry = gps_geometry();
    geometry
        .rows
        .push(row(GnssSystem::Gps, 5, [0.2, 0.6, 0.774_596_669_241_483_4]));
    geometry
}

fn mixed_geometry_for_enumeration() -> AraimGeometry {
    AraimGeometry {
        rows: vec![
            row(GnssSystem::Gps, 1, [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3]),
            row(GnssSystem::Gps, 2, [INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3]),
            row(
                GnssSystem::Galileo,
                1,
                [-INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
            ),
        ],
        receiver: receiver(),
        clock_systems: vec![GnssSystem::Gps, GnssSystem::Galileo],
        ut1_degraded: None,
    }
}

fn reference_gps_geometry() -> AraimGeometry {
    AraimGeometry {
        rows: vec![
            row_az_el(GnssSystem::Gps, 1, 0.0, 65.0),
            row_az_el(GnssSystem::Gps, 2, 60.0, 35.0),
            row_az_el(GnssSystem::Gps, 3, 120.0, 55.0),
            row_az_el(GnssSystem::Gps, 4, 190.0, 30.0),
            row_az_el(GnssSystem::Gps, 5, 250.0, 50.0),
            row_az_el(GnssSystem::Gps, 6, 310.0, 40.0),
        ],
        receiver: receiver(),
        clock_systems: vec![GnssSystem::Gps],
        ut1_degraded: None,
    }
}

fn reference_gps_galileo_geometry() -> AraimGeometry {
    let mut geometry = reference_gps_geometry();
    geometry.rows.extend([
        row_az_el(GnssSystem::Galileo, 1, 20.0, 55.0),
        row_az_el(GnssSystem::Galileo, 2, 95.0, 35.0),
        row_az_el(GnssSystem::Galileo, 3, 170.0, 60.0),
        row_az_el(GnssSystem::Galileo, 4, 245.0, 40.0),
        row_az_el(GnssSystem::Galileo, 5, 320.0, 50.0),
    ]);
    geometry.clock_systems = vec![GnssSystem::Gps, GnssSystem::Galileo];
    geometry
}

fn wg_c_add_v3_numerical_example_geometry() -> AraimGeometry {
    AraimGeometry {
        rows: WG_C_ADD_V3_ROWS
            .into_iter()
            .map(|(system, prn, design_enu, _, _)| row_from_wg_c_design(system, prn, design_enu))
            .collect(),
        receiver: receiver(),
        clock_systems: vec![GnssSystem::Gps, GnssSystem::Galileo],
        ut1_degraded: None,
    }
}

fn wg_c_add_v3_numerical_example_ism() -> Ism {
    let model = SatelliteIsmModel::new(
        WG_C_SIGMA_URA_M,
        WG_C_SIGMA_URE_M,
        WG_C_EXAMPLE_B_NOM_M,
        WG_C_P_SAT,
    );
    let satellites = WG_C_ADD_V3_ROWS
        .into_iter()
        .map(|(system, prn, _, c_int_m2, c_acc_m2)| {
            SatelliteIsm::new_with_effective_sigmas(
                sat(system, prn),
                WG_C_SIGMA_URA_M,
                WG_C_SIGMA_URE_M,
                WG_C_EXAMPLE_B_NOM_M,
                WG_C_P_SAT,
                c_int_m2.sqrt(),
                c_acc_m2.sqrt(),
            )
        })
        .collect();
    Ism::new(
        vec![
            ConstellationIsm::new(GnssSystem::Gps, WG_C_EXAMPLE_P_CONST, model),
            ConstellationIsm::new(GnssSystem::Galileo, WG_C_EXAMPLE_P_CONST, model),
        ],
        satellites,
    )
}

fn reference_gps_ism() -> Ism {
    Ism::new(
        vec![ConstellationIsm::new(
            GnssSystem::Gps,
            WG_C_P_CONST_GPS,
            reference_sat_model(),
        )],
        Vec::new(),
    )
}

fn reference_gps_galileo_ism() -> Ism {
    Ism::new(
        vec![
            ConstellationIsm::new(GnssSystem::Gps, WG_C_P_CONST_GPS, reference_sat_model()),
            ConstellationIsm::new(GnssSystem::Galileo, WG_C_P_CONST_GAL, reference_sat_model()),
        ],
        Vec::new(),
    )
}

const fn reference_sat_model() -> SatelliteIsmModel {
    SatelliteIsmModel::new(WG_C_SIGMA_URA_M, WG_C_SIGMA_URE_M, WG_C_B_NOM_M, WG_C_P_SAT)
}

fn row(system: GnssSystem, prn: u8, los: [f64; 3]) -> AraimRow {
    AraimRow {
        id: sat(system, prn),
        line_of_sight: LineOfSight::new(los[0], los[1], los[2]),
        system,
        elevation_rad: core::f64::consts::FRAC_PI_2,
    }
}

fn row_az_el(system: GnssSystem, prn: u8, az_deg: f64, el_deg: f64) -> AraimRow {
    AraimRow {
        id: sat(system, prn),
        line_of_sight: line_of_sight_from_az_el_deg(az_deg, el_deg, receiver())
            .expect("valid az/el"),
        system,
        elevation_rad: el_deg.to_radians(),
    }
}

fn sat(system: GnssSystem, prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(system, prn).expect("valid satellite")
}

fn receiver() -> Wgs84Geodetic {
    Wgs84Geodetic::new(0.0, 0.0, 0.0).expect("valid receiver")
}

fn dot(row: &[f64], residuals: &[f64]) -> f64 {
    row.iter().zip(residuals).map(|(a, b)| a * b).sum()
}

fn monitored_mode_keys(result: &super::AraimResult) -> Vec<String> {
    result
        .fault_modes
        .iter()
        .skip(1)
        .filter(|mode| mode.monitorable)
        .map(|mode| {
            if let Some(system) = mode.excluded_constellation {
                format!("{system:?}*")
            } else {
                mode.excluded
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("+")
            }
        })
        .collect()
}

fn assert_constellation_intermediates(
    result: &super::AraimResult,
    system: GnssSystem,
    k_fa_3: f64,
    sigma_3_m: f64,
    sigma_ss_3_m: f64,
    b_3_m: f64,
) {
    let mode = result
        .fault_modes
        .iter()
        .find(|mode| mode.excluded_constellation == Some(system))
        .expect("constellation mode");
    assert_abs_diff(mode.sigma_int_enu_m[2], sigma_3_m, WG_C_INTERMEDIATE_TOL_M);
    assert_abs_diff(
        mode.threshold_enu_m[2] / k_fa_3,
        sigma_ss_3_m,
        WG_C_INTERMEDIATE_TOL_M,
    );
    assert_abs_diff(mode.bias_enu_m[2], b_3_m, WG_C_INTERMEDIATE_TOL_M);
}

fn row_from_wg_c_design(system: GnssSystem, prn: u8, design_enu: [f64; 3]) -> AraimRow {
    let receiver = receiver();
    let r = ecef_to_enu_rotation(receiver.lat_rad, receiver.lon_rad);
    let east = -design_enu[0];
    let north = -design_enu[1];
    let up = -design_enu[2];
    AraimRow {
        id: sat(system, prn),
        line_of_sight: LineOfSight::new(
            r[0][0] * east + r[1][0] * north + r[2][0] * up,
            r[0][1] * east + r[1][1] * north + r[2][1] * up,
            r[0][2] * east + r[1][2] * north + r[2][2] * up,
        ),
        system,
        elevation_rad: core::f64::consts::FRAC_PI_2,
    }
}

fn phmi_split_fits_total(phmi_vert: f64, phmi_hor: f64, phmi_total: f64) -> bool {
    phmi_vert + phmi_hor <= phmi_total + phmi_total * 16.0 * f64::EPSILON
}

fn receiver_solution(
    used_sats: Vec<GnssSatelliteId>,
    systems: Vec<GnssSystem>,
) -> ReceiverSolution {
    let used_count = used_sats.len();
    let redundancy = used_count as isize - (3 + systems.len()) as isize;
    let n_params = 3 + systems.len();
    let rank = used_count.min(n_params);
    ReceiverSolution {
        position: ItrfPositionM::new(6_378_137.0, 0.0, 0.0).expect("valid receiver"),
        geodetic: Some(receiver()),
        rx_clock_s: 0.0,
        rx_clock_drift_s_s: None,
        system_clocks_s: systems.iter().map(|&system| (system, 0.0)).collect(),
        dop: None,
        system_tdops: Vec::new(),
        position_covariance: crate::dop::PositionCovariance {
            ecef_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            enu_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        },
        residuals_m: vec![0.0; used_count],
        pseudorange_variances_m2: vec![1.0; used_count],
        weights: vec![1.0; used_count],
        used_sats,
        rejected_sats: Vec::new(),
        geometry_quality: crate::geometry_quality::classify(
            rank,
            n_params,
            redundancy as i32,
            1.0,
            1.0,
            false,
            crate::geometry_quality::GeometryQualityThresholds::default(),
        ),
        metadata: SolutionMetadata {
            iterations: 1,
            converged: true,
            status: Status::StepTolerance,
            ionosphere_applied: false,
            troposphere_applied: false,
            outer_iterations: 0,
            final_robust_scale_m: None,
            used_count,
            systems,
            redundancy,
            raim_checkable: redundancy >= 1,
            ut1_degraded: None,
        },
    }
}

struct StaticEphemeris {
    positions: Vec<(GnssSatelliteId, [f64; 3])>,
}

impl EphemerisSource for StaticEphemeris {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.positions
            .iter()
            .find(|(id, _)| *id == sat)
            .map(|&(_, position)| (position, 0.0))
    }
}

fn assert_abs_diff(left: f64, right: f64, tol: f64) {
    let diff = (left - right).abs();
    assert!(
        diff <= tol,
        "left={left} right={right} diff={diff} tol={tol}"
    );
}
