//! Clean-room validation for stacked static least squares.
//!
//! Provenance: public GNSS geodesy practice solves static sessions by stacking
//! many raw range epochs into one weighted least-squares problem with a shared
//! receiver position and epoch-local receiver clocks. The fixtures below are
//! synthetic range geometries formed directly in this crate.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};

use super::*;
use crate::astro::math::least_squares::jacobian_2point_with_min_steps;
use crate::dop::line_of_sight_from_az_el_deg;
use crate::spp::{
    ionosphere_for, sat_model, solve, KlobucharCoeffs, SatModelEnv, SppModelRecipe,
    DEFAULT_HUBER_K, DEFAULT_ROBUST_MAX_OUTER, DEFAULT_ROBUST_OUTER_TOL_M,
    DEFAULT_ROBUST_SCALE_FLOOR_M,
};

#[derive(Debug, Clone)]
struct FixedEphemeris {
    positions: BTreeMap<GnssSatelliteId, [f64; 3]>,
}

impl EphemerisSource for FixedEphemeris {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        _t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.positions
            .get(&sat)
            .copied()
            .map(|position| (position, 0.0))
    }
}

const TRUTH: [f64; 3] = [6_378_137.0, 0.0, 0.0];
const RANGE_M: f64 = 20_200_000.0;

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS id")
}

fn galileo(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Galileo, prn).expect("valid Galileo id")
}

fn zero_klobuchar() -> KlobucharCoeffs {
    KlobucharCoeffs {
        alpha: [0.0; 4],
        beta: [0.0; 4],
    }
}

fn options() -> StaticSolveOptions {
    StaticSolveOptions {
        initial_position_m: [TRUTH[0] + 120.0, TRUTH[1] - 80.0, TRUTH[2] + 50.0],
        with_geodetic: true,
        robust: None,
    }
}

fn robust_options() -> StaticSolveOptions {
    StaticSolveOptions {
        robust: Some(RobustConfig {
            huber_k: DEFAULT_HUBER_K,
            scale_floor_m: DEFAULT_ROBUST_SCALE_FLOOR_M,
            max_outer: DEFAULT_ROBUST_MAX_OUTER,
            outer_tol_m: DEFAULT_ROBUST_OUTER_TOL_M,
        }),
        ..options()
    }
}

fn sat_position(azimuth_deg: f64, elevation_deg: f64) -> [f64; 3] {
    let los = line_of_sight_from_az_el_deg(
        azimuth_deg,
        elevation_deg,
        Wgs84Geodetic {
            lat_rad: 0.0,
            lon_rad: 0.0,
            height_m: 0.0,
        },
    )
    .expect("valid LOS");
    [
        TRUTH[0] + RANGE_M * los.e_x,
        TRUTH[1] + RANGE_M * los.e_y,
        TRUTH[2] + RANGE_M * los.e_z,
    ]
}

fn epoch_angles(epoch_index: usize) -> Vec<(f64, f64)> {
    match epoch_index {
        0 => vec![
            (0.0, 58.0),
            (60.0, 47.0),
            (130.0, 42.0),
            (210.0, 53.0),
            (300.0, 38.0),
            (20.0, 32.0),
        ],
        1 => vec![
            (25.0, 55.0),
            (90.0, 44.0),
            (155.0, 49.0),
            (240.0, 36.0),
            (315.0, 46.0),
            (350.0, 31.0),
        ],
        _ => vec![
            (45.0, 51.0),
            (115.0, 39.0),
            (185.0, 56.0),
            (260.0, 43.0),
            (335.0, 35.0),
            (10.0, 48.0),
        ],
    }
}

fn make_store(epoch_count: usize) -> FixedEphemeris {
    let mut positions = BTreeMap::new();
    for epoch_index in 0..epoch_count {
        for (sat_index, (azimuth_deg, elevation_deg)) in
            epoch_angles(epoch_index).into_iter().enumerate()
        {
            let sat = gps((epoch_index * 10 + sat_index + 1) as u8);
            positions.insert(sat, sat_position(azimuth_deg, elevation_deg));
        }
    }
    FixedEphemeris { positions }
}

fn pseudorange(
    eph: &FixedEphemeris,
    sat: GnssSatelliteId,
    t_rx_j2000_s: f64,
    day_of_year: f64,
    clock_m: f64,
) -> f64 {
    let inputs = SolveInputs {
        observations: Vec::new(),
        t_rx_j2000_s,
        t_rx_second_of_day_s: 12_000.0,
        day_of_year,
        initial_guess: [TRUTH[0], TRUTH[1], TRUTH[2], clock_m],
        corrections: Corrections::NONE,
        klobuchar: zero_klobuchar(),
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: BTreeMap::new(),
        met: SurfaceMet::default(),
        robust: None,
    };
    let env = SatModelEnv {
        eph,
        t_rx_j2000_s,
        t_rx_second_of_day_s: inputs.t_rx_second_of_day_s,
        day_of_year,
        corrections: Corrections::NONE,
        met: &inputs.met,
        glonass_channels: &inputs.glonass_channels,
        model: SppModelRecipe::reference(),
    };
    sat_model(
        &env,
        sat,
        TRUTH,
        clock_m,
        RANGE_M,
        ionosphere_for(sat.system, &inputs),
    )
    .expect("synthetic satellite in store")
    .p_hat_m
}

fn make_epoch(
    eph: &FixedEphemeris,
    epoch_index: usize,
    clock_m: f64,
    corruptions_m: &[f64],
) -> StaticEpoch {
    let t_rx_j2000_s = 1000.0 + epoch_index as f64 * 30.0;
    let day_of_year = 120.0 + epoch_index as f64;
    let measurements = epoch_angles(epoch_index)
        .into_iter()
        .enumerate()
        .map(|(sat_index, _)| {
            let sat = gps((epoch_index * 10 + sat_index + 1) as u8);
            let corruption = corruptions_m.get(sat_index).copied().unwrap_or(0.0);
            Observation {
                satellite_id: sat,
                pseudorange_m: pseudorange(eph, sat, t_rx_j2000_s, day_of_year, clock_m)
                    + corruption,
            }
        })
        .collect();
    StaticEpoch {
        measurements,
        weights: None,
        t_rx_j2000_s,
        t_rx_second_of_day_s: 12_000.0,
        day_of_year,
        clock_initial_m: 0.0,
        corrections: Corrections::NONE,
        klobuchar: zero_klobuchar(),
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: BTreeMap::new(),
        met: SurfaceMet::default(),
    }
}

fn clean_epochs(eph: &FixedEphemeris, epoch_count: usize) -> Vec<StaticEpoch> {
    (0..epoch_count)
        .map(|epoch_index| make_epoch(eph, epoch_index, 12.0 + epoch_index as f64 * 4.0, &[]))
        .collect()
}

fn mixed_system_store_and_epoch() -> (FixedEphemeris, StaticEpoch) {
    let sats = [
        (gps(1), 0.0, 58.0),
        (gps(2), 70.0, 45.0),
        (gps(3), 150.0, 42.0),
        (gps(4), 235.0, 50.0),
        (galileo(1), 25.0, 54.0),
        (galileo(2), 105.0, 40.0),
        (galileo(3), 200.0, 47.0),
        (galileo(4), 315.0, 36.0),
    ];
    let positions = sats
        .iter()
        .map(|(sat, azimuth_deg, elevation_deg)| (*sat, sat_position(*azimuth_deg, *elevation_deg)))
        .collect::<BTreeMap<_, _>>();
    let eph = FixedEphemeris { positions };
    let t_rx_j2000_s = 1000.0;
    let day_of_year = 120.0;
    let clock_m = 12.0;
    let measurements = sats
        .iter()
        .map(|(sat, _, _)| Observation {
            satellite_id: *sat,
            pseudorange_m: pseudorange(&eph, *sat, t_rx_j2000_s, day_of_year, clock_m),
        })
        .collect();
    (
        eph,
        StaticEpoch {
            measurements,
            weights: None,
            t_rx_j2000_s,
            t_rx_second_of_day_s: 12_000.0,
            day_of_year,
            clock_initial_m: 0.0,
            corrections: Corrections::NONE,
            klobuchar: zero_klobuchar(),
            beidou_klobuchar: None,
            galileo_nequick: None,
            sbas_iono: None,
            glonass_channels: BTreeMap::new(),
            met: SurfaceMet::default(),
        },
    )
}

fn trace3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] + matrix[1][1] + matrix[2][2]
}

fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{label}: got {got:e}, want {want:e}, tol {tol:e}"
    );
}

fn assert_position_close(got: [f64; 3], want: [f64; 3], tol_m: f64) {
    for idx in 0..3 {
        assert_close(got[idx], want[idx], tol_m, "position");
    }
}

#[test]
fn single_epoch_static_matches_spp_bits() {
    let eph = make_store(1);
    let epochs = clean_epochs(&eph, 1);
    let static_solution = solve_static(&eph, &epochs, options()).expect("static solve");
    let spp_inputs = solve_inputs_for_epoch(&epochs[0], options());
    let spp_solution = solve(&eph, &spp_inputs, true).expect("SPP solve");

    assert_eq!(
        static_solution.position.x_m.to_bits(),
        spp_solution.position.x_m.to_bits()
    );
    assert_eq!(
        static_solution.position.y_m.to_bits(),
        spp_solution.position.y_m.to_bits()
    );
    assert_eq!(
        static_solution.position.z_m.to_bits(),
        spp_solution.position.z_m.to_bits()
    );
    assert_eq!(
        static_solution.per_epoch_clock[0].clock_s.to_bits(),
        spp_solution.rx_clock_s.to_bits()
    );
    assert_eq!(static_solution.used_sats[0], spp_solution.used_sats);
}

#[test]
fn single_epoch_multi_system_static_matches_spp_bits() {
    let (eph, epoch) = mixed_system_store_and_epoch();
    let epochs = vec![epoch];
    let static_solution = solve_static(&eph, &epochs, options()).expect("static solve");
    let spp_inputs = solve_inputs_for_epoch(&epochs[0], options());
    let spp_solution = solve(&eph, &spp_inputs, true).expect("SPP solve");

    assert_eq!(
        static_solution.position.x_m.to_bits(),
        spp_solution.position.x_m.to_bits()
    );
    assert_eq!(
        static_solution.position.y_m.to_bits(),
        spp_solution.position.y_m.to_bits()
    );
    assert_eq!(
        static_solution.position.z_m.to_bits(),
        spp_solution.position.z_m.to_bits()
    );
    assert_eq!(
        static_solution
            .per_epoch_clock
            .iter()
            .map(|clock| (clock.system, clock.clock_s.to_bits()))
            .collect::<Vec<_>>(),
        spp_solution
            .system_clocks_s
            .iter()
            .map(|(system, clock_s)| (*system, clock_s.to_bits()))
            .collect::<Vec<_>>()
    );
}

#[test]
fn stacked_solution_recovers_truth_and_covariance_matches_hand_normal_matrix() {
    let eph = make_store(3);
    let epochs = clean_epochs(&eph, 3);
    let solution = solve_static(&eph, &epochs, options()).expect("static solve");

    assert_position_close(solution.position.as_array(), TRUTH, 2.0e-4);
    for (idx, clock) in solution.per_epoch_clock.iter().enumerate() {
        let want_m = 12.0 + idx as f64 * 4.0;
        assert_close(clock.clock_s * C_M_S, want_m, 2.0e-4, "clock");
    }

    let independent = independent_jacobian_covariance(&eph, &epochs, &solution);
    for i in 0..independent.nrows() {
        for j in 0..independent.ncols() {
            assert_close(
                solution.covariance.state_m2[i][j],
                independent[(i, j)],
                1.0e-8,
                "covariance",
            );
        }
    }

    let hand = hand_covariance(&eph, &epochs, &solution);
    for i in 0..hand.nrows() {
        for j in 0..hand.ncols() {
            assert_close(
                solution.covariance.state_m2[i][j],
                hand[(i, j)],
                1.0e-1,
                "hand covariance",
            );
        }
    }
}

#[test]
fn covariance_shrinks_by_geometry_not_naive_epoch_count() {
    let eph = make_store(3);
    let one_epoch = clean_epochs(&eph, 1);
    let three_epochs = clean_epochs(&eph, 3);
    let single = solve_static(&eph, &one_epoch, options()).expect("single epoch");
    let stacked = solve_static(&eph, &three_epochs, options()).expect("stacked epochs");

    let single_std = trace3(single.covariance.position_ecef_m2).sqrt();
    let stacked_std = trace3(stacked.covariance.position_ecef_m2).sqrt();
    let naive_std = single_std / (3.0_f64).sqrt();

    assert!(stacked_std < single_std);
    assert!(
        (stacked_std - naive_std).abs() > single_std * 1.0e-3,
        "stacked covariance should follow geometry, not naive epoch count"
    );
}

#[test]
fn rank_deficient_geometry_returns_typed_error() {
    let mut eph = FixedEphemeris {
        positions: BTreeMap::new(),
    };
    let duplicate_los_position = sat_position(20.0, 35.0);
    let mut measurements = Vec::new();
    for prn in 1..=5 {
        let sat = gps(prn);
        eph.positions.insert(sat, duplicate_los_position);
        measurements.push(Observation {
            satellite_id: sat,
            pseudorange_m: RANGE_M + 5.0,
        });
    }
    let epoch = StaticEpoch {
        measurements,
        weights: None,
        t_rx_j2000_s: 1000.0,
        t_rx_second_of_day_s: 12_000.0,
        day_of_year: 120.0,
        clock_initial_m: 0.0,
        corrections: Corrections::NONE,
        klobuchar: zero_klobuchar(),
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: BTreeMap::new(),
        met: SurfaceMet::default(),
    };

    let err = solve_static(&eph, &[epoch], options()).expect_err("rank deficient");
    assert!(matches!(err, StaticSolveError::Singular(_)));
}

#[test]
fn robust_static_diagnostics_surface_bad_epoch_and_satellite() {
    let eph = make_store(3);
    let mut epochs = clean_epochs(&eph, 3);
    let bad_sat = gps(13);
    epochs[1] = make_epoch(&eph, 1, 16.0, &[80.0, -70.0, 220.0, -90.0, 65.0, -55.0]);

    let solution = solve_static(&eph, &epochs, robust_options()).expect("robust static");

    assert!(solution.metadata.outer_iterations > 0);
    let bad_row = solution
        .residuals_m
        .iter()
        .find(|row| row.satellite_id == bad_sat)
        .expect("bad satellite row");
    assert!(
        bad_row.robust_weight_ratio < 0.2,
        "bad satellite should be down-weighted"
    );

    let worst_epoch = solution
        .per_epoch_influence
        .iter()
        .min_by(|a, b| {
            a.min_robust_weight_ratio
                .total_cmp(&b.min_robust_weight_ratio)
        })
        .expect("epoch influence");
    assert_eq!(worst_epoch.epoch_index, 1);
    assert!(worst_epoch.min_robust_weight_ratio < 0.2);

    let worst_sat = solution
        .per_satellite_influence
        .iter()
        .min_by(|a, b| a.robust_weight_ratio.total_cmp(&b.robust_weight_ratio))
        .expect("satellite influence");
    assert_eq!(worst_sat.satellite_id, bad_sat);
    assert!(matches!(worst_sat.status, StaticInfluenceStatus::Solved));

    let batch_sat = solution
        .per_satellite_batch_influence
        .iter()
        .find(|influence| influence.satellite_id == bad_sat)
        .expect("batch satellite influence");
    assert_eq!(batch_sat.omitted_measurements, 1);
    assert!(batch_sat.min_robust_weight_ratio < 0.2);
}

fn independent_jacobian_covariance(
    eph: &FixedEphemeris,
    epochs: &[StaticEpoch],
    solution: &StaticSolution,
) -> DMatrix<f64> {
    let prepared =
        prepare_static(eph, epochs, options(), SppModelRecipe::reference()).expect("prepared");
    let x = solution_state(solution);
    let weights = solution
        .residuals_m
        .iter()
        .map(|row| row.effective_weight.sqrt())
        .collect::<Vec<_>>();
    let residual = |state: &DVector<f64>| -> DVector<f64> {
        let unweighted = residual_static_unweighted(
            eph,
            &prepared,
            state.as_slice(),
            SppModelRecipe::reference(),
        )
        .expect("residuals");
        DVector::from_iterator(
            unweighted.len(),
            unweighted
                .into_iter()
                .zip(weights.iter())
                .map(|(residual_m, sqrt_weight)| residual_m * sqrt_weight),
        )
    };
    let f0 = residual(&x);
    let min_steps = super::static_fd_min_steps(x.len());
    let jacobian = jacobian_2point_with_min_steps(residual, &x, &f0, &min_steps)
        .expect("independent jacobian");
    normal_covariance(&jacobian, 1.0).expect("independent covariance")
}

fn solution_state(solution: &StaticSolution) -> DVector<f64> {
    let mut state = Vec::with_capacity(3 + solution.per_epoch_clock.len());
    let position = solution.position.as_array();
    state.extend_from_slice(&position);
    state.extend(
        solution
            .per_epoch_clock
            .iter()
            .map(|clock| clock.clock_s * C_M_S),
    );
    DVector::from_vec(state)
}

fn hand_covariance(
    eph: &FixedEphemeris,
    epochs: &[StaticEpoch],
    solution: &StaticSolution,
) -> DMatrix<f64> {
    let n_params = solution.covariance.state_m2.len();
    let mut design = DMatrix::zeros(solution.residuals_m.len(), n_params);
    let rx = solution.position.as_array();
    let clock_m_by_epoch = solution
        .per_epoch_clock
        .iter()
        .map(|clock| (clock.epoch_index, clock.clock_s * C_M_S))
        .collect::<BTreeMap<_, _>>();
    let clock_columns = solution
        .per_epoch_clock
        .iter()
        .enumerate()
        .map(|(clock_index, clock)| ((clock.epoch_index, clock.system), 3 + clock_index))
        .collect::<BTreeMap<_, _>>();
    for (row_index, row) in solution.residuals_m.iter().enumerate() {
        let epoch = &epochs[row.epoch_index];
        let p_meas = epoch
            .measurements
            .iter()
            .find(|measurement| measurement.satellite_id == row.satellite_id)
            .expect("measurement in epoch")
            .pseudorange_m;
        let env = SatModelEnv {
            eph,
            t_rx_j2000_s: epoch.t_rx_j2000_s,
            t_rx_second_of_day_s: epoch.t_rx_second_of_day_s,
            day_of_year: epoch.day_of_year,
            corrections: epoch.corrections,
            met: &epoch.met,
            glonass_channels: &epoch.glonass_channels,
            model: SppModelRecipe::reference(),
        };
        let inputs = solve_inputs_for_epoch(epoch, options());
        let sat = sat_model(
            &env,
            row.satellite_id,
            rx,
            clock_m_by_epoch[&row.epoch_index],
            p_meas,
            ionosphere_for(row.satellite_id.system, &inputs),
        )
        .expect("synthetic satellite model");
        let sat_pos = sat.sat_rot_ecef_m;
        let dx = sat_pos[0] - rx[0];
        let dy = sat_pos[1] - rx[1];
        let dz = sat_pos[2] - rx[2];
        let norm = (dx * dx + dy * dy + dz * dz).sqrt();
        let scale = row.effective_weight.sqrt();
        design[(row_index, 0)] = -dx / norm * scale;
        design[(row_index, 1)] = -dy / norm * scale;
        design[(row_index, 2)] = -dz / norm * scale;
        let clock_col = clock_columns[&(row.epoch_index, clock_system_for(row.satellite_id))];
        design[(row_index, clock_col)] = scale;
    }
    normal_covariance(&design, 1.0).expect("hand covariance")
}

fn clock_system_for(satellite_id: GnssSatelliteId) -> GnssSystem {
    match satellite_id.system {
        GnssSystem::Sbas => GnssSystem::Gps,
        system => system,
    }
}

// A single measurement-starved epoch (3 usable measurements) is underdetermined
// (3 meas < 4 unknowns = position3 + clock1) -> singular normal matrix -> no
// valid covariance -> rejected. Stacking N static epochs shares ONE position
// across per-epoch clocks: 3N measurements vs (3 + N) unknowns -> redundancy
// 2N-3. This proves stacking restores redundancy, yields a finite covariance,
// and recovers truth where the single snapshot cannot.
#[test]
fn starved_epochs_stack_to_recover_redundancy_and_truth() {
    let eph = make_store(3);
    let starved = |n: usize| -> Vec<StaticEpoch> {
        (0..n)
            .map(|i| {
                let mut e = make_epoch(&eph, i, 12.0 + i as f64 * 4.0, &[]);
                e.measurements.truncate(3); // starve to 3 satellites
                e
            })
            .collect()
    };

    // 1 starved epoch: 3 meas, 4 unknowns -> underdetermined -> no valid fix.
    let one = solve_static(&eph, &starved(1), options());
    match &one {
        Err(_) => {}
        Ok(s) => assert!(
            s.metadata.redundancy < 0,
            "single 3-sat epoch must be underdetermined, got redundancy {}",
            s.metadata.redundancy
        ),
    }
    eprintln!(
        "N=1 (single starved snapshot): {}",
        match &one {
            Err(e) => format!("REJECTED ({e:?})"),
            Ok(s) => format!("redundancy {}", s.metadata.redundancy),
        }
    );

    // Stacking N starved (3-sat) epochs: redundancy 2N-3, finite covariance, recovers truth.
    // (Covariance magnitude follows geometry, not naive epoch count, so it is not
    // asserted monotonic here -- see covariance_shrinks_by_geometry_not_naive_epoch_count.)
    for n in 2..=3 {
        let sol = solve_static(&eph, &starved(n), options())
            .unwrap_or_else(|e| panic!("{n} stacked 3-sat epochs should solve, got {e:?}"));
        let p = sol.position.as_array();
        let err_m =
            ((p[0] - TRUTH[0]).powi(2) + (p[1] - TRUTH[1]).powi(2) + (p[2] - TRUTH[2]).powi(2))
                .sqrt();
        let std_m = trace3(sol.covariance.position_ecef_m2).sqrt();
        eprintln!(
            "N={n}: redundancy {:>2}  pos_err {err_m:8.2} m  cov_std {std_m:8.2} m",
            sol.metadata.redundancy
        );
        assert_eq!(
            sol.metadata.redundancy,
            2 * n as isize - 3,
            "redundancy should be 2N-3"
        );
        assert!(
            std_m.is_finite() && std_m > 0.0,
            "stacked covariance must be finite and positive, got {std_m}"
        );
        assert!(
            err_m < 100.0,
            "stacked fix should recover truth within 100 m, got {err_m} m"
        );
    }
}
