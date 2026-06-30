use std::collections::BTreeMap;

use crate::astro::math::vec3::sub3;

use super::antenna::ReceiverAntennaScratch;
use super::*;
use crate::ambiguity::AmbiguityId;
use crate::constants::{C_M_S, F_L1_HZ};

// Per-system reference map from reference satellite ids (system = first byte).
fn refs_of(ids: &[&str]) -> BTreeMap<String, String> {
    ids.iter()
        .map(|id| (id[..1].to_string(), id.to_string()))
        .collect()
}

fn filter_state(
    reference_ids: &[&str],
    baseline_m: [f64; 3],
    baseline_prior_sigma_m: f64,
    ambiguity_prior_sigma_m: f64,
) -> FilterState {
    FilterState::new(
        refs_of(reference_ids),
        baseline_m,
        baseline_prior_sigma_m,
        ambiguity_prior_sigma_m,
    )
    .unwrap()
}

// Reference prior: diagonal information, 1/σ² on baseline (i<3) and each
// ambiguity diagonal, zero off-diagonal (Elixir sequential_initial_information).
fn presized(n: usize, baseline_sigma: f64, ambiguity_sigma: f64) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        m[i * n + i] = if i < 3 {
            1.0 / (baseline_sigma * baseline_sigma)
        } else {
            1.0 / (ambiguity_sigma * ambiguity_sigma)
        };
    }
    m
}

#[test]
fn new_seeds_diagonal_baseline_information() {
    let s = filter_state(&["G01"], [1.0, 2.0, 3.0], 10.0, 5.7);
    assert_eq!(s.dim(), 3);
    assert_eq!(s.information, presized(3, 10.0, 5.7));
    assert_eq!(s.baseline_m, [1.0, 2.0, 3.0]);
    assert!(s.sd_ambiguity_ids.is_empty());
}

#[test]
fn new_rejects_invalid_priors_and_baseline() {
    assert_eq!(
        FilterState::new(refs_of(&["G01"]), [f64::NAN, 0.0, 0.0], 10.0, 5.7),
        Err(FilterStateValidationError {
            field: "state.baseline_m",
            kind: FilterStateValidationKind::NonFinite,
        })
    );
    assert_eq!(
        FilterState::new(refs_of(&["G01"]), [0.0; 3], 0.0, 5.7),
        Err(FilterStateValidationError {
            field: "state.baseline_prior_sigma_m",
            kind: FilterStateValidationKind::NotPositive,
        })
    );
    assert_eq!(
        FilterState::new(refs_of(&["G01"]), [0.0; 3], f64::MIN_POSITIVE, 5.7),
        Err(FilterStateValidationError {
            field: "state.baseline_prior_sigma_m",
            kind: FilterStateValidationKind::NonFinite,
        })
    );
    assert_eq!(
        FilterState::new(refs_of(&["G01"]), [0.0; 3], 10.0, f64::MAX),
        Err(FilterStateValidationError {
            field: "state.ambiguity_prior_sigma_m",
            kind: FilterStateValidationKind::NotPositive,
        })
    );
}

#[test]
fn validate_for_update_rejects_invalid_information_geometry() {
    let mut negative_variance = filter_state(&["G01"], [0.0; 3], 10.0, 5.7);
    negative_variance.information[0] = -1.0;
    assert_eq!(
        negative_variance.validate_for_update(),
        Err(FilterStateValidationError {
            field: "state.information",
            kind: FilterStateValidationKind::NotPositiveSemidefinite,
        })
    );

    let mut asymmetric = filter_state(&["G01"], [0.0; 3], 10.0, 5.7);
    asymmetric.information[1] = 0.5;
    assert_eq!(
        asymmetric.validate_for_update(),
        Err(FilterStateValidationError {
            field: "state.information",
            kind: FilterStateValidationKind::NotSymmetric,
        })
    );

    let mut indefinite = filter_state(&["G01"], [0.0; 3], 10.0, 5.7);
    indefinite.information = vec![1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(
        indefinite.validate_for_update(),
        Err(FilterStateValidationError {
            field: "state.information",
            kind: FilterStateValidationKind::NotPositiveSemidefinite,
        })
    );
}

#[test]
fn growing_columns_matches_presized_prior() {
    // Streaming add-on-first-sighting must equal batch pre-sizing.
    let mut s = filter_state(&["G01"], [0.0; 3], 10.0, 5.7);
    s.ensure_ambiguity("G02", 0.42);
    s.ensure_ambiguity("G05~ra1", -1.3);
    assert_eq!(s.dim(), 5);
    assert_eq!(s.information, presized(5, 10.0, 5.7));
    assert_eq!(s.sd_ambiguity_ids, vec!["G02", "G05~ra1"]);
    assert_eq!(s.sd_ambiguities_m, vec![0.42, -1.3]);
    assert_eq!(s.index_of("G02"), Some(3));
    assert_eq!(s.index_of("G05~ra1"), Some(4));
    assert_eq!(s.index_of("absent"), None);
}

#[test]
fn ensure_ambiguity_is_idempotent() {
    let mut s = filter_state(&["G01"], [0.0; 3], 10.0, 5.7);
    s.ensure_ambiguity("G02", 0.42);
    s.ensure_ambiguity("G02", 99.0); // same id: no-op, value unchanged
    assert_eq!(s.dim(), 4);
    assert_eq!(s.sd_ambiguities_m, vec![0.42]);
}

#[test]
fn normal_equations_fold_and_solve_recover_known_state() {
    // Two correlated measurements of a 2-state system:
    //   x0 + x1 = 10, x0 - x1 = 2  ->  x = [6, 4].
    let n = 2;
    let mut lambda = vec![0.0; n * n];
    let mut eta = vec![0.0; n];
    fold_measurement(&mut lambda, &mut eta, &[1.0, 1.0], 1.0, 10.0);
    fold_measurement(&mut lambda, &mut eta, &[1.0, -1.0], 1.0, 2.0);
    // Λ = [[2,0],[0,2]] (the off-diagonal +1 and -1 cancel), η = [12, 8].
    assert_eq!(lambda, vec![2.0, 0.0, 0.0, 2.0]);
    assert_eq!(eta, vec![12.0, 8.0]);
    let x = solve_normal(&lambda, &eta).unwrap();
    assert!((x[0] - 6.0).abs() < 1e-12);
    assert!((x[1] - 4.0).abs() < 1e-12);
}

#[test]
fn solve_normal_reports_singular() {
    // Λ = [[1,2],[2,4]] is singular.
    assert!(solve_normal(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0]).is_none());
}

#[test]
fn weighted_measurement_recovers_weighted_mean() {
    // Scalar state, two measurements with weights 1 and 3 -> weighted mean.
    let mut lambda = vec![0.0];
    let mut eta = vec![0.0];
    fold_measurement(&mut lambda, &mut eta, &[1.0], 1.0, 10.0);
    fold_measurement(&mut lambda, &mut eta, &[1.0], 3.0, 12.0);
    let x = solve_normal(&lambda, &eta).unwrap();
    assert!((x[0] - 11.5).abs() < 1e-12); // (10 + 3*12) / 4
}

#[test]
fn measurement_block_uses_shared_reference_covariance() {
    // Two DD measurements with equal single-difference variances v=2:
    // R = v*(I + J) = [[4, 2], [2, 4]], so
    // R^-1 = [[1/3, -1/6], [-1/6, 1/3]]. Diagonal row folding would produce
    // [[1/4, 0], [0, 1/4]] instead and miss the reference-satellite coupling.
    let row1 = DdRow {
        kind: RowKind::Code,
        sat: "G02".into(),
        ref_sat: "G01".into(),
        sd_ambiguity_id: "G02".into(),
        h: vec![1.0, 0.0],
        y: 10.0,
        sd_variance_m2: 2.0,
        ref_sd_variance_m2: 2.0,
        weight: 0.25,
    };
    let row2 = DdRow {
        kind: RowKind::Code,
        sat: "G03".into(),
        ref_sat: "G01".into(),
        sd_ambiguity_id: "G03".into(),
        h: vec![0.0, 1.0],
        y: 20.0,
        sd_variance_m2: 2.0,
        ref_sd_variance_m2: 2.0,
        weight: 0.25,
    };
    let rows = [&row1, &row2];

    let r_inv = double_difference_inverse_covariance(&rows).unwrap();
    let expected = vec![1.0 / 3.0, -1.0 / 6.0, -1.0 / 6.0, 1.0 / 3.0];
    for (got, want) in r_inv.iter().zip(&expected) {
        assert!((got - want).abs() < 1e-15, "{got} != {want}");
    }

    let mut lambda = vec![0.0; 4];
    let mut eta = vec![0.0; 2];
    fold_measurement_block(&mut lambda, &mut eta, &rows).unwrap();

    for (got, want) in lambda.iter().zip(&expected) {
        assert!((got - want).abs() < 1e-15, "{got} != {want}");
    }
    assert!(eta[0].abs() < 1e-14, "eta[0] = {}", eta[0]);
    assert!((eta[1] - 5.0).abs() < 1e-14, "eta[1] = {}", eta[1]);
}

// Reference values below are computed independently from the exact formulas
// (geocentric up = base/|base|, sin el = LOS·up, clamp at 0.05) at full f64
// precision - they pin the parity semantics, not Rust self-consistency.
const VAR_BASE: [f64; 3] = [4_075_580.0, 931_854.0, 4_801_568.0];
const SAT_HIGH: [f64; 3] = [15_000_000.0, 7_000_000.0, 21_000_000.0]; // sin el ≈ 0.982
const SAT_LOW: [f64; 3] = [-12_000_000.0, 18_000_000.0, 19_000_000.0]; // sin el ≈ 0.106

fn simple_single_dd_fixture() -> ([f64; 3], Epoch, FilterState, MeasModel, Vec<String>) {
    let base = VAR_BASE;
    let baseline = [1.2, -0.85, 0.91];
    let rover = [
        base[0] + baseline[0],
        base[1] + baseline[1],
        base[2] + baseline[2],
    ];
    let g01 = SAT_HIGH;
    let g02 = SAT_LOW;
    let mk = |sat: [f64; 3], id: &str| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(sat, base),
        base_phase_m: range_m(sat, base),
        rover_code_m: range_m(sat, rover),
        rover_phase_m: range_m(sat, rover),
        base_tx_pos: sat,
        rover_tx_pos: sat,
        pos: sat,
    };
    let epoch = Epoch {
        references: vec![mk(g01, "G01")],
        nonref: vec![mk(g02, "G02")],
        velocity_mps: None,
        dt_s: 0.0,
    };
    let mut state = filter_state(&["G01"], baseline, 10.0, 5.7);
    state.ensure_ambiguity("G01", 0.0);
    state.ensure_ambiguity("G02", 0.0);
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    (base, epoch, state, model, vec!["G02".to_string()])
}

fn single_dd_update_opts() -> UpdateOpts {
    UpdateOpts {
        hold_sigma_m: 1.0,
        position_tol_m: 1e-3,
        ambiguity_tol_m: 1e-6,
        max_iterations: 10,
        process_noise_baseline_sigma_m: 0.0,
        dynamics_model: DynamicsModel::ConstantPosition,
        float_only_systems: vec![],
        innovation_screen: None,
        report_residuals: false,
        force_report_iterate_failure: false,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: 3.0,
        },
    }
}

#[test]
fn update_epoch_rejects_malformed_filter_state_before_indexing() {
    let (base, epoch, state, model, ids) = simple_single_dd_fixture();
    let lambda = C_M_S / F_L1_HZ;
    let wavelengths: BTreeMap<String, f64> = ids.iter().map(|id| (id.clone(), lambda)).collect();
    let offsets: BTreeMap<String, f64> = ids.iter().map(|id| (id.clone(), 0.0)).collect();
    let opts = single_dd_update_opts();

    let mut short_info = state.clone();
    short_info.information.pop();
    assert_eq!(
        update_epoch(
            short_info,
            &epoch,
            base,
            &model,
            &wavelengths,
            &offsets,
            &opts,
        ),
        Err(UpdateError::InvalidState {
            field: "state.information",
            kind: InvalidStateKind::Length {
                expected: state.dim() * state.dim(),
                actual: state.information.len() - 1,
            },
        })
    );

    let mut short_sd = state.clone();
    short_sd.sd_ambiguities_m.pop();
    assert_eq!(
        update_epoch(
            short_sd,
            &epoch,
            base,
            &model,
            &wavelengths,
            &offsets,
            &opts,
        ),
        Err(UpdateError::InvalidState {
            field: "state.sd_ambiguities_m",
            kind: InvalidStateKind::Length {
                expected: state.sd_ambiguity_ids.len(),
                actual: state.sd_ambiguities_m.len() - 1,
            },
        })
    );

    let mut nonfinite = state.clone();
    nonfinite.baseline_m[0] = f64::NAN;
    assert_eq!(
        update_epoch(
            nonfinite,
            &epoch,
            base,
            &model,
            &wavelengths,
            &offsets,
            &opts,
        ),
        Err(UpdateError::InvalidState {
            field: "state.baseline_m",
            kind: InvalidStateKind::NonFinite,
        })
    );

    let mut nonpositive_sigma = state.clone();
    nonpositive_sigma.ambiguity_prior_sigma_m = 0.0;
    assert_eq!(
        update_epoch(
            nonpositive_sigma,
            &epoch,
            base,
            &model,
            &wavelengths,
            &offsets,
            &opts,
        ),
        Err(UpdateError::InvalidState {
            field: "state.ambiguity_prior_sigma_m",
            kind: InvalidStateKind::NotPositive,
        })
    );

    let mut tiny_sigma = state.clone();
    tiny_sigma.ambiguity_prior_sigma_m = f64::MIN_POSITIVE;
    assert_eq!(
        update_epoch(
            tiny_sigma,
            &epoch,
            base,
            &model,
            &wavelengths,
            &offsets,
            &opts
        ),
        Err(UpdateError::InvalidState {
            field: "state.ambiguity_prior_sigma_m",
            kind: InvalidStateKind::NonFinite,
        })
    );
}

#[test]
fn update_epoch_rejects_invalid_update_options_before_math() {
    let (base, epoch, state, model, ids) = simple_single_dd_fixture();
    let lambda = C_M_S / F_L1_HZ;
    let wavelengths: BTreeMap<String, f64> = ids.iter().map(|id| (id.clone(), lambda)).collect();
    let offsets: BTreeMap<String, f64> = ids.iter().map(|id| (id.clone(), 0.0)).collect();
    let valid_opts = single_dd_update_opts();

    let assert_invalid = |opts: UpdateOpts, field, kind| {
        assert_eq!(
            update_epoch(
                state.clone(),
                &epoch,
                base,
                &model,
                &wavelengths,
                &offsets,
                &opts,
            ),
            Err(UpdateError::InvalidInput { field, kind })
        );
    };

    let mut opts = valid_opts.clone();
    opts.hold_sigma_m = 0.0;
    assert_invalid(
        opts,
        "rtk.update.hold_sigma_m",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts.clone();
    opts.hold_sigma_m = f64::MIN_POSITIVE;
    assert_invalid(
        opts,
        "rtk.update.hold_sigma_m",
        RtkInputErrorKind::NonFinite,
    );

    let mut opts = valid_opts.clone();
    opts.position_tol_m = f64::NAN;
    assert_invalid(
        opts,
        "rtk.update.position_tol_m",
        RtkInputErrorKind::NonFinite,
    );

    let mut opts = valid_opts.clone();
    opts.ambiguity_tol_m = 0.0;
    assert_invalid(
        opts,
        "rtk.update.ambiguity_tol_m",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts.clone();
    opts.max_iterations = 0;
    assert_invalid(
        opts,
        "rtk.update.max_iterations",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts.clone();
    opts.process_noise_baseline_sigma_m = -1.0;
    assert_invalid(
        opts,
        "rtk.update.process_noise_baseline_sigma_m",
        RtkInputErrorKind::Negative,
    );

    let mut opts = valid_opts.clone();
    opts.innovation_screen = Some(InnovationScreenOpts {
        threshold_sigma: 0.0,
        min_rows: 1,
    });
    assert_invalid(
        opts,
        "rtk.update.innovation_screen.threshold_sigma",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts.clone();
    opts.innovation_screen = Some(InnovationScreenOpts {
        threshold_sigma: 3.0,
        min_rows: 0,
    });
    assert_invalid(
        opts,
        "rtk.update.innovation_screen.min_rows",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts.clone();
    opts.ar_arming_sigma_m = Some(0.0);
    assert_invalid(
        opts,
        "rtk.update.ar_arming_sigma_m",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts;
    opts.search.ratio_threshold = f64::NAN;
    assert_invalid(
        opts,
        "rtk.update.search.ratio_threshold",
        RtkInputErrorKind::NonFinite,
    );
}

#[test]
fn elevation_sin_is_geocentric_and_matches_reference() {
    let high = elevation_sin(VAR_BASE, SAT_HIGH);
    let low = elevation_sin(VAR_BASE, SAT_LOW);
    assert!((high - 0.9823712838936762).abs() < 1e-15, "{high}");
    assert!((low - 0.10636728642290873).abs() < 1e-15, "{low}");

    // Straight down (sat at base/2): LOS is -up, sin el = -1 (to FP rounding;
    // the dot of two opposite unit vectors can land an ulp outside [-1, 1]).
    let below = elevation_sin(VAR_BASE, [2_037_790.0, 465_927.0, 2_400_784.0]);
    assert!((below + 1.0).abs() < 1e-12, "{below}");
}

#[test]
fn variance_models_match_reference_formulas() {
    let sigma = 0.3;
    let simple = StochasticModel::Simple {
        elevation_weighting: false,
    };
    let elev = StochasticModel::Simple {
        elevation_weighting: true,
    };

    // Simple: 2σ², elevation-independent.
    assert_eq!(
        single_difference_variance(sigma, simple, VAR_BASE, SAT_LOW),
        0.18
    );

    // Elevation-weighted: 2(σ/sin el)².
    let v = single_difference_variance(sigma, elev, VAR_BASE, SAT_HIGH);
    assert!((v - 0.18651818780129176).abs() < 1e-15, "{v}");
    let v = single_difference_variance(sigma, elev, VAR_BASE, SAT_LOW);
    assert!((v - 15.909493196935287).abs() < 1e-12, "{v}");

    // RTKLIB: 2(σ² + σ²/sin²el).
    let v = single_difference_variance(sigma, StochasticModel::Rtklib, VAR_BASE, SAT_HIGH);
    assert!((v - 0.36651818780129175).abs() < 1e-15, "{v}");
    let v = single_difference_variance(sigma, StochasticModel::Rtklib, VAR_BASE, SAT_LOW);
    assert!((v - 16.089493196935287).abs() < 1e-12, "{v}");

    // Below the horizon, sin el clamps to 0.05: elevation-weighted variance
    // is 2(0.3/0.05)² = 72 and RTKLIB 2(0.09 + 0.09/0.0025) = 72.18 (to FP
    // rounding; 0.3/0.05 is not exact in binary).
    let down = [2_037_790.0, 465_927.0, 2_400_784.0];
    let v = single_difference_variance(sigma, elev, VAR_BASE, down);
    assert!((v - 72.0).abs() < 1e-12, "{v}");
    let v = single_difference_variance(sigma, StochasticModel::Rtklib, VAR_BASE, down);
    assert!((v - 72.18).abs() < 1e-12, "{v}");
}

#[test]
fn unequal_variance_block_matches_dense_inverse() {
    // RTKLIB variances at distinct elevations exercise the general
    // Sherman-Morrison path (unequal D) that the Elixir fast path lacks.
    // Reference R⁻¹ computed independently by dense inversion of
    // R = D + v_ref·11ᵀ with D = diag(16.089493…, 15.965508…), v_ref = 0.366518….
    let mk = |sat: &str, var: f64, ref_var: f64, h: Vec<f64>, y: f64| DdRow {
        kind: RowKind::Code,
        sat: sat.into(),
        ref_sat: "G01".into(),
        sd_ambiguity_id: sat.into(),
        h,
        y,
        sd_variance_m2: var,
        ref_sd_variance_m2: ref_var,
        weight: 1.0 / (var + ref_var),
    };
    let v_ref = 0.36651818780129175;
    let row1 = mk("G02", 16.089493196935287, v_ref, vec![1.0, 0.0], 1.0);
    let row2 = mk("G03", 15.965508421574619, v_ref, vec![0.0, 1.0], 1.0);
    let r_inv = double_difference_inverse_covariance(&[&row1, &row2]).unwrap();

    let expected = [
        0.06079845603340234,
        -0.0013644197661107447,
        -0.0013644197661107447,
        0.06126000823962085,
    ];
    for (got, want) in r_inv.iter().zip(&expected) {
        assert!((got - want).abs() < 1e-15, "{got} != {want}");
    }
}

#[test]
fn baseline_block_is_preserved_when_growing() {
    // A non-trivial baseline information block must survive column growth.
    let mut s = filter_state(&["G01"], [0.0; 3], 10.0, 5.7);
    // Simulate accumulated baseline correlation.
    s.information[1] = 0.001; // [0][1]
    s.information[3] = 0.001; // [1][0]
    s.ensure_ambiguity("G02", 0.0);
    assert_eq!(s.info(0, 1), 0.001);
    assert_eq!(s.info(1, 0), 0.001);
    assert_eq!(s.info(3, 3), 1.0 / (5.7 * 5.7));
    assert_eq!(s.info(0, 3), 0.0);
}

#[test]
fn dd_rows_match_perfect_synthetic_geometry() {
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let baseline = [1.2, -0.85, 0.91];
    let rover = [
        base[0] + baseline[0],
        base[1] + baseline[1],
        base[2] + baseline[2],
    ];
    let g01 = [15_000_000.0, 7_000_000.0, 21_000_000.0];
    let g02 = [-12_000_000.0, 18_000_000.0, 19_000_000.0];
    let amb_g01 = 3.0;
    let amb_g02 = -7.0;

    // Perfect synthetic observations: code = geometric range, phase = range +
    // ambiguity, zero receiver clocks (they cancel in the double difference).
    let mk = |sat: [f64; 3], id: &str, amb: f64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(sat, base),
        base_phase_m: range_m(sat, base),
        rover_code_m: range_m(sat, rover),
        rover_phase_m: range_m(sat, rover) + amb,
        base_tx_pos: sat,
        rover_tx_pos: sat,
        pos: sat,
    };
    let epoch = Epoch {
        references: vec![mk(g01, "G01", amb_g01)],
        nonref: vec![mk(g02, "G02", amb_g02)],
        velocity_mps: None,
        dt_s: 0.0,
    };

    let mut state = filter_state(&["G01"], baseline, 10.0, 5.7);
    state.ensure_ambiguity("G01", amb_g01);
    state.ensure_ambiguity("G02", amb_g02);

    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let rows = epoch_dd_rows(&epoch, base, &state, &model).unwrap();

    assert_eq!(rows.len(), 2);
    let (code, phase) = (&rows[0], &rows[1]);
    assert_eq!(code.kind, RowKind::Code);
    assert_eq!(phase.kind, RowKind::Phase);

    // Perfect synthetic measurement -> ~zero prefit residual.
    assert!(code.y.abs() < 1e-6, "code y = {}", code.y);
    assert!(phase.y.abs() < 1e-6, "phase y = {}", phase.y);

    // Geometry block = LOS(rover->G02) - LOS(rover->G01).
    let expected = sub3(range_derivative(rover, g02), range_derivative(rover, g01));
    for (k, &e) in expected.iter().enumerate() {
        assert!((code.h[k] - e).abs() < 1e-12);
        assert!((phase.h[k] - e).abs() < 1e-12);
    }
    // Code carries no ambiguity columns; phase has +1 at the sat SD column
    // (G02 -> index 4) and -1 at the reference SD column (G01 -> index 3).
    assert_eq!(code.h[3], 0.0);
    assert_eq!(code.h[4], 0.0);
    assert_eq!(phase.h[3], -1.0);
    assert_eq!(phase.h[4], 1.0);

    // Information weights = 1/(4σ²) (simple model: SD var 2σ², DD = sum).
    assert!((code.weight - 1.0 / (4.0 * 0.3 * 0.3)).abs() < 1e-12);
    assert!((phase.weight - 1.0 / (4.0 * 0.003 * 0.003)).abs() < 1e-9);
}

#[test]
fn dd_rows_reject_nonfinite_satellite_measurement() {
    let (base, mut epoch, state, model, _) = simple_single_dd_fixture();
    epoch.nonref[0].rover_code_m = f64::NAN;
    let mut scratch = EpochRowsScratch::default();

    let err = dd_epoch_rows_into(
        MeasContext {
            base,
            model: &model,
            antenna: None,
        },
        &epoch,
        0,
        state.baseline_m,
        DdRowRecipe::SequentialFilter {
            sd_ambiguity_ids: &state.sd_ambiguity_ids,
            sd_ambiguities_m: &state.sd_ambiguities_m,
        },
        &mut scratch,
    )
    .unwrap_err();

    assert_eq!(
        err,
        DdRowError::InvalidInput {
            field: "rtk.rover_code_m",
            kind: RtkInputErrorKind::NonFinite,
        }
    );
}

#[test]
fn float_solve_rejects_nonpositive_measurement_sigma() {
    let (base, epoch, _, mut model, ambiguity_ids) = simple_single_dd_fixture();
    model.code_sigma_m = 0.0;
    let epochs = vec![epoch];
    let err = solve_float_baseline(
        &epochs,
        base,
        &ambiguity_ids,
        [0.0; 3],
        &model,
        FloatSolveOpts {
            position_tol_m: 1.0e-4,
            ambiguity_tol_m: 1.0e-4,
            max_iterations: 2,
        },
        None,
    )
    .unwrap_err();

    assert_eq!(
        err,
        FloatSolveError::InvalidInput {
            field: "rtk.code_sigma_m",
            kind: RtkInputErrorKind::NotPositive,
        }
    );
}

#[test]
fn float_solve_rejects_nonfinite_measurement_sigma() {
    for (code_sigma_m, phase_sigma_m, field) in [
        (f64::NAN, 0.003, "rtk.code_sigma_m"),
        (0.3, f64::INFINITY, "rtk.phase_sigma_m"),
    ] {
        let (base, epoch, _, mut model, ambiguity_ids) = simple_single_dd_fixture();
        model.code_sigma_m = code_sigma_m;
        model.phase_sigma_m = phase_sigma_m;
        let epochs = vec![epoch];
        let err = solve_float_baseline(
            &epochs,
            base,
            &ambiguity_ids,
            [0.0; 3],
            &model,
            FloatSolveOpts {
                position_tol_m: 1.0e-4,
                ambiguity_tol_m: 1.0e-4,
                max_iterations: 2,
            },
            None,
        )
        .unwrap_err();

        assert_eq!(
            err,
            FloatSolveError::InvalidInput {
                field,
                kind: RtkInputErrorKind::NonFinite,
            }
        );
    }
}

#[test]
fn float_solve_rejects_invalid_options_before_math() {
    let (base, epoch, _, model, ambiguity_ids) = simple_single_dd_fixture();
    let epochs = vec![epoch];
    let valid_opts = FloatSolveOpts {
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 2,
    };

    let assert_invalid = |opts: FloatSolveOpts, field, kind| {
        assert_eq!(
            solve_float_baseline(&epochs, base, &ambiguity_ids, [0.0; 3], &model, opts, None),
            Err(FloatSolveError::InvalidInput { field, kind })
        );
    };

    let mut opts = valid_opts;
    opts.position_tol_m = f64::NAN;
    assert_invalid(
        opts,
        "rtk.float.position_tol_m",
        RtkInputErrorKind::NonFinite,
    );

    let mut opts = valid_opts;
    opts.ambiguity_tol_m = 0.0;
    assert_invalid(
        opts,
        "rtk.float.ambiguity_tol_m",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts;
    opts.max_iterations = 0;
    assert_invalid(
        opts,
        "rtk.float.max_iterations",
        RtkInputErrorKind::NotPositive,
    );
}

#[test]
fn fixed_solve_rejects_invalid_options_before_math() {
    let ids = Vec::<String>::new();
    let satellites = BTreeMap::<String, String>::new();
    let wavelengths = BTreeMap::<String, f64>::new();
    let offsets = BTreeMap::<String, f64>::new();
    let float_only = Vec::<String>::new();
    let float_ambiguities = Vec::<(String, f64)>::new();
    let covariance = Vec::<f64>::new();
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let ambiguities = AmbiguitySet {
        ids: &ids,
        satellites: &satellites,
        scale: AmbiguityScale {
            wavelengths_m: &wavelengths,
            offsets_m: &offsets,
        },
        float_only_systems: &float_only,
    };
    let float_prior = FloatPrior {
        baseline_m: [0.0; 3],
        ambiguities_m: &float_ambiguities,
        covariance_m: &covariance,
    };
    let valid_opts = FixedSolveOpts {
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 2,
        ratio_threshold: 3.0,
        partial_ambiguity_resolution: false,
        partial_min_ambiguities: 1,
    };

    let assert_invalid = |opts: FixedSolveOpts, field, kind| {
        assert_eq!(
            solve_fixed_baseline(&[], [0.0; 3], ambiguities, float_prior, &model, opts, None,),
            Err(FixedSolveError::InvalidInput { field, kind })
        );
    };

    let mut opts = valid_opts;
    opts.position_tol_m = f64::INFINITY;
    assert_invalid(
        opts,
        "rtk.fixed.position_tol_m",
        RtkInputErrorKind::NonFinite,
    );

    let mut opts = valid_opts;
    opts.ambiguity_tol_m = 0.0;
    assert_invalid(
        opts,
        "rtk.fixed.ambiguity_tol_m",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts;
    opts.max_iterations = 0;
    assert_invalid(
        opts,
        "rtk.fixed.max_iterations",
        RtkInputErrorKind::NotPositive,
    );

    let mut opts = valid_opts;
    opts.ratio_threshold = f64::NAN;
    assert_invalid(
        opts,
        "rtk.fixed.ratio_threshold",
        RtkInputErrorKind::NonFinite,
    );

    let mut opts = valid_opts;
    opts.partial_ambiguity_resolution = true;
    opts.partial_min_ambiguities = 0;
    assert_invalid(
        opts,
        "rtk.fixed.partial_min_ambiguities",
        RtkInputErrorKind::NotPositive,
    );
}

#[test]
fn fixed_solve_rejects_invalid_float_prior_covariance_geometry() {
    let ids = vec!["G01-G02-L1".to_string(), "G01-G03-L1".to_string()];
    let satellites = BTreeMap::from([
        ("G01-G02-L1".to_string(), "G02".to_string()),
        ("G01-G03-L1".to_string(), "G03".to_string()),
    ]);
    let wavelengths = BTreeMap::<String, f64>::new();
    let offsets = BTreeMap::<String, f64>::new();
    let float_only = vec!["G".to_string()];
    let float_ambiguities = vec![
        ("G01-G02-L1".to_string(), 0.0),
        ("G01-G03-L1".to_string(), 0.0),
    ];
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let ambiguities = AmbiguitySet {
        ids: &ids,
        satellites: &satellites,
        scale: AmbiguityScale {
            wavelengths_m: &wavelengths,
            offsets_m: &offsets,
        },
        float_only_systems: &float_only,
    };
    let opts = FixedSolveOpts {
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 2,
        ratio_threshold: 3.0,
        partial_ambiguity_resolution: false,
        partial_min_ambiguities: 1,
    };
    let solve = |covariance_m: &[f64]| {
        solve_fixed_baseline(
            &[],
            [0.0; 3],
            ambiguities,
            FloatPrior {
                baseline_m: [0.0; 3],
                ambiguities_m: &float_ambiguities,
                covariance_m,
            },
            &model,
            opts,
            None,
        )
    };
    let invalid_covariance = |covariance_m: &[f64]| {
        assert_eq!(
            solve(covariance_m),
            Err(FixedSolveError::InvalidInput {
                field: "rtk.fixed.float_covariance_m",
                kind: RtkInputErrorKind::NotPositive,
            })
        );
    };

    invalid_covariance(&[-1.0, 0.0, 0.0, 1.0]);
    invalid_covariance(&[1.0, 0.25, 0.0, 1.0]);
    invalid_covariance(&[1.0, 2.0, 2.0, 1.0]);
    assert_eq!(
        solve(&[1.0, 0.250_000_000_05, 0.249_999_999_95, 1.0]),
        Err(FixedSolveError::SingularGeometry)
    );
    assert_eq!(
        solve(&[1.0, 0.25, 0.25, 1.0]),
        Err(FixedSolveError::SingularGeometry)
    );
}

#[test]
fn receiver_antenna_pcv_interpolates_zenith_and_azimuth() {
    let cal = ReceiverAntennaCalibration {
        pco_neu_m: [0.0, 0.0, 0.0],
        noazi_pcv_m: vec![(0.0, 0.0), (10.0, 10.0)],
        azi_pcv_m: vec![
            (0.0, 0.0, 0.0),
            (0.0, 10.0, 10.0),
            (90.0, 0.0, 90.0),
            (90.0, 10.0, 100.0),
        ],
    };
    let mut scratch = ReceiverAntennaScratch::default();

    assert_eq!(pcv_m(&cal, 5.0, 45.0, &mut scratch).unwrap(), 50.0);
    assert_eq!(pcv_m(&cal, -1.0, 45.0, &mut scratch).unwrap(), 45.0);
    assert_eq!(pcv_m(&cal, 99.0, 45.0, &mut scratch).unwrap(), 55.0);
}

#[test]
fn provided_receiver_antenna_errors_when_pcv_grid_is_missing() {
    let cal = ReceiverAntennaCalibration {
        pco_neu_m: [0.0, 0.0, 0.0],
        noazi_pcv_m: Vec::new(),
        azi_pcv_m: Vec::new(),
    };
    let mut scratch = ReceiverAntennaScratch::default();

    assert_eq!(
        pcv_m(&cal, 5.0, 45.0, &mut scratch),
        Err(ReceiverAntennaError::MissingPcv)
    );
}

#[test]
fn provided_receiver_antenna_does_not_fallback_to_noazi_after_bad_azimuth_grid() {
    let cal = ReceiverAntennaCalibration {
        pco_neu_m: [0.0, 0.0, 0.0],
        noazi_pcv_m: vec![(0.0, 0.0)],
        azi_pcv_m: vec![(f64::NAN, 0.0, 1.0)],
    };
    let mut scratch = ReceiverAntennaScratch::default();

    assert_eq!(
        pcv_m(&cal, 5.0, 45.0, &mut scratch),
        Err(ReceiverAntennaError::MissingPcv)
    );
}

#[test]
fn provided_receiver_antenna_errors_when_geometry_is_invalid() {
    let cal = ReceiverAntennaCalibration {
        pco_neu_m: [0.0, 0.0, 0.0],
        noazi_pcv_m: vec![(0.0, 0.0)],
        azi_pcv_m: Vec::new(),
    };
    let pos = [4_075_580.0, 931_854.0, 4_801_568.0];
    let mut scratch = ReceiverAntennaScratch::default();

    assert_eq!(
        receiver_antenna_correction(pos, pos, &cal, &mut scratch),
        Err(ReceiverAntennaError::InvalidGeometry)
    );
}

#[test]
fn provided_receiver_antenna_errors_when_receiver_position_cannot_define_frame() {
    let cal = ReceiverAntennaCalibration {
        pco_neu_m: [0.0, 0.0, 0.0],
        noazi_pcv_m: vec![(0.0, 0.0)],
        azi_pcv_m: Vec::new(),
    };
    let mut scratch = ReceiverAntennaScratch::default();

    for receiver_pos in [[0.0; 3], [f64::INFINITY, 0.0, 0.0]] {
        assert_eq!(
            receiver_antenna_correction(
                [15_000_000.0, 7_000_000.0, 21_000_000.0],
                receiver_pos,
                &cal,
                &mut scratch,
            ),
            Err(ReceiverAntennaError::InvalidGeometry)
        );
    }
}

#[test]
fn corrected_dd_rows_zero_synthetic_receiver_antenna_residuals() {
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let baseline = [1.2, -0.85, 0.91];
    let rover = [
        base[0] + baseline[0],
        base[1] + baseline[1],
        base[2] + baseline[2],
    ];
    let g01 = [15_000_000.0, 7_000_000.0, 21_000_000.0];
    let g02 = [-12_000_000.0, 18_000_000.0, 19_000_000.0];
    let base_cal = ReceiverAntennaCalibration {
        pco_neu_m: [0.0, 0.0, 0.10],
        noazi_pcv_m: vec![(0.0, 0.0)],
        azi_pcv_m: Vec::new(),
    };
    let rover_cal = ReceiverAntennaCalibration {
        pco_neu_m: [0.0, 0.0, 0.05],
        noazi_pcv_m: vec![(0.0, 0.0)],
        azi_pcv_m: Vec::new(),
    };
    let corrections = ReceiverAntennaCorrections {
        base: base_cal,
        rover: rover_cal,
    };
    let mut antenna_scratch = ReceiverAntennaScratch::default();
    let mk = |sat: [f64; 3],
              id: &str,
              corrections: &ReceiverAntennaCorrections,
              scratch: &mut ReceiverAntennaScratch| {
        let base_corr = receiver_antenna_correction(sat, base, &corrections.base, scratch).unwrap();
        let rover_corr =
            receiver_antenna_correction(sat, rover, &corrections.rover, scratch).unwrap();
        SatMeas {
            sat: id.into(),
            sd_ambiguity_id: id.into(),
            base_code_m: range_m(sat, base) - base_corr,
            base_phase_m: range_m(sat, base) - base_corr,
            rover_code_m: range_m(sat, rover) - rover_corr,
            rover_phase_m: range_m(sat, rover) - rover_corr,
            base_tx_pos: sat,
            rover_tx_pos: sat,
            pos: sat,
        }
    };
    let epoch = Epoch {
        references: vec![mk(g01, "G01", &corrections, &mut antenna_scratch)],
        nonref: vec![mk(g02, "G02", &corrections, &mut antenna_scratch)],
        velocity_mps: None,
        dt_s: 0.0,
    };
    let mut state = filter_state(&["G01"], baseline, 10.0, 5.7);
    state.ensure_ambiguity("G01", 0.0);
    state.ensure_ambiguity("G02", 0.0);
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let mut scratch = EpochRowsScratch::default();
    let rows = dd_epoch_rows_into(
        MeasContext {
            base,
            model: &model,
            antenna: Some(&corrections),
        },
        &epoch,
        0,
        state.baseline_m,
        DdRowRecipe::SequentialFilter {
            sd_ambiguity_ids: &state.sd_ambiguity_ids,
            sd_ambiguities_m: &state.sd_ambiguities_m,
        },
        &mut scratch,
    )
    .unwrap();

    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].kind, RowKind::Code);
    assert_eq!(rows[1].kind, RowKind::Phase);
    assert!(rows[0].y.abs() < 1.0e-8, "code y = {}", rows[0].y);
    assert!(rows[1].y.abs() < 1.0e-8, "phase y = {}", rows[1].y);
}

#[test]
fn dd_rows_error_when_provided_receiver_antenna_has_no_pcv_samples() {
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let baseline = [1.2, -0.85, 0.91];
    let rover = [
        base[0] + baseline[0],
        base[1] + baseline[1],
        base[2] + baseline[2],
    ];
    let g01 = [15_000_000.0, 7_000_000.0, 21_000_000.0];
    let g02 = [-12_000_000.0, 18_000_000.0, 19_000_000.0];
    let missing_pcv = ReceiverAntennaCalibration {
        pco_neu_m: [0.0, 0.0, 0.0],
        noazi_pcv_m: Vec::new(),
        azi_pcv_m: Vec::new(),
    };
    let corrections = ReceiverAntennaCorrections {
        base: missing_pcv.clone(),
        rover: missing_pcv,
    };
    let mk = |sat: [f64; 3], id: &str| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(sat, base),
        base_phase_m: range_m(sat, base),
        rover_code_m: range_m(sat, rover),
        rover_phase_m: range_m(sat, rover),
        base_tx_pos: sat,
        rover_tx_pos: sat,
        pos: sat,
    };
    let epoch = Epoch {
        references: vec![mk(g01, "G01")],
        nonref: vec![mk(g02, "G02")],
        velocity_mps: None,
        dt_s: 0.0,
    };
    let mut state = filter_state(&["G01"], baseline, 10.0, 5.7);
    state.ensure_ambiguity("G01", 0.0);
    state.ensure_ambiguity("G02", 0.0);
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let mut scratch = EpochRowsScratch::default();

    let err = dd_epoch_rows_into(
        MeasContext {
            base,
            model: &model,
            antenna: Some(&corrections),
        },
        &epoch,
        0,
        state.baseline_m,
        DdRowRecipe::SequentialFilter {
            sd_ambiguity_ids: &state.sd_ambiguity_ids,
            sd_ambiguities_m: &state.sd_ambiguities_m,
        },
        &mut scratch,
    )
    .unwrap_err();

    assert_eq!(
        err,
        DdRowError::ReceiverAntenna(ReceiverAntennaError::MissingPcv)
    );
}

#[test]
fn iterate_epoch_recovers_baseline_and_dd_ambiguities() {
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let truth = [1.2, -0.85, 0.91];
    let rover = [base[0] + truth[0], base[1] + truth[1], base[2] + truth[2]];
    // Reference G01 + 4 non-reference satellites; true SD ambiguities (m).
    let sats: [(&str, [f64; 3], f64); 5] = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0.0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 0.6),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -1.4),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 1.0),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -0.3),
    ];
    let mk = |pos: [f64; 3], id: &str, amb: f64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(pos, base),
        base_phase_m: range_m(pos, base),
        rover_code_m: range_m(pos, rover),
        rover_phase_m: range_m(pos, rover) + amb,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    let epoch = Epoch {
        references: vec![mk(sats[0].1, sats[0].0, sats[0].2)],
        nonref: sats[1..].iter().map(|&(id, p, a)| mk(p, id, a)).collect(),
        velocity_mps: None,
        dt_s: 0.0,
    };

    // Wrong initial baseline + very loose priors so the measurement dominates
    // (a finite prior pulling the wrong center leaves a bias ~ prior_info /
    // meas_info × offset; σ=1e4 makes it ~1e-7 m, negligible).
    let mut state = filter_state(&["G01"], [-30.0, 25.0, -10.0], 1.0e4, 1.0e4);
    for &(id, _, _) in &sats {
        state.ensure_ambiguity(id, 0.0);
    }
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };

    let post = iterate_epoch(
        MeasContext {
            base,
            model: &model,
            antenna: None,
        },
        &state,
        &epoch,
        &[],
        IterateControls {
            hold_sigma_m: 1.0,
            position_tol_m: 1e-3,
            ambiguity_tol_m: 1e-6,
            max_iterations: 10,
        },
    )
    .unwrap();

    // Baseline converges to truth.
    for (k, &t) in truth.iter().enumerate() {
        assert!(
            (post.baseline_m[k] - t).abs() < 1e-3,
            "baseline[{k}] = {} (truth {t})",
            post.baseline_m[k]
        );
    }
    // DD float ambiguities (sat - ref) converge to the true DDs.
    let pos = |id: &str| state.ambiguity_pos(id).unwrap();
    let ref_amb = post.sd_ambiguities_m[pos("G01")];
    for &(id, _, true_amb) in &sats[1..] {
        let dd = post.sd_ambiguities_m[pos(id)] - ref_amb;
        assert!(
            (dd - true_amb).abs() < 1e-3,
            "{id} dd = {dd} (truth {true_amb})"
        );
    }
}

#[test]
fn dd_covariance_transform_matches_hand_computation() {
    // SD ambiguity covariance (A=0, B=1, reference C=2).
    #[rustfmt::skip]
    let sd = vec![
        4.0, 1.0, 0.5,
        1.0, 3.0, 0.5,
        0.5, 0.5, 2.0,
    ];
    // DD targets A-C and B-C; λ = 1 (cycles == metres).
    let dd = [(0usize, 2usize), (1usize, 2usize)];
    // DD_cov[i][j] = C[si][sj] - C[si][rj] - C[ri][sj] + C[ri][rj]:
    //   [0][0] = 4 - .5 - .5 + 2 = 5 ; [0][1]=[1][0] = 1 - .5 - .5 + 2 = 2
    //   [1][1] = 3 - .5 - .5 + 2 = 4
    assert_eq!(
        dd_covariance_cycles(&sd, 3, &dd, &[1.0, 1.0]),
        vec![5.0, 2.0, 2.0, 4.0]
    );
    // Wavelength scaling divides entry (i,j) by λ_i·λ_j.
    assert_eq!(
        dd_covariance_cycles(&sd, 3, &dd, &[2.0, 1.0]),
        vec![5.0 / 4.0, 2.0 / 2.0, 2.0 / 2.0, 4.0 / 1.0]
    );
}

#[test]
fn search_and_hold_fixes_integer_double_differences() {
    let lambda = C_M_S / F_L1_HZ;
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let truth = [1.2, -0.85, 0.91];
    let rover = [base[0] + truth[0], base[1] + truth[1], base[2] + truth[2]];
    // Reference G01 + 4 non-reference; integer cycle ambiguities so the DD
    // floats land on integers and the ratio test passes.
    let sats: [(&str, [f64; 3], i64); 5] = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 3),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -7),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 12),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -4),
    ];
    let mk = |pos: [f64; 3], id: &str, ncyc: i64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(pos, base),
        base_phase_m: range_m(pos, base),
        rover_code_m: range_m(pos, rover),
        rover_phase_m: range_m(pos, rover) + (ncyc as f64) * lambda,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    let epoch = Epoch {
        references: vec![mk(sats[0].1, sats[0].0, sats[0].2)],
        nonref: sats[1..].iter().map(|&(id, p, n)| mk(p, id, n)).collect(),
        velocity_mps: None,
        dt_s: 0.0,
    };

    let mut state = filter_state(&["G01"], [-30.0, 25.0, -10.0], 1.0e4, 1.0e4);
    for &(id, _, _) in &sats {
        state.ensure_ambiguity(id, 0.0);
    }
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let post = iterate_epoch(
        MeasContext {
            base,
            model: &model,
            antenna: None,
        },
        &state,
        &epoch,
        &[],
        IterateControls {
            hold_sigma_m: 1.0,
            position_tol_m: 1e-3,
            ambiguity_tol_m: 1e-6,
            max_iterations: 10,
        },
    )
    .unwrap();

    let mut state_post = state.clone();
    state_post.baseline_m = post.baseline_m;
    state_post.sd_ambiguities_m = post.sd_ambiguities_m.clone();

    let wl: BTreeMap<String, f64> = sats[1..]
        .iter()
        .map(|&(id, _, _)| (id.to_string(), lambda))
        .collect();
    let off: BTreeMap<String, f64> = sats[1..]
        .iter()
        .map(|&(id, _, _)| (id.to_string(), 0.0))
        .collect();

    let (holds, search) = search_and_hold(
        &state_post,
        &post.information,
        &epoch,
        AmbiguityScale {
            wavelengths_m: &wl,
            offsets_m: &off,
        },
        &[],
        SearchPolicy {
            float_only_systems: &[],
            ar_arming_sigma_m: None,
            ratio: SearchOpts {
                ratio_threshold: 3.0,
            },
        },
    )
    .unwrap();
    let search = search.expect("integer search ran");

    assert_eq!(holds.len(), 4);
    assert_eq!(search.integer_status, IntegerStatus::Fixed);
    assert!(
        search.integer_ratio.unwrap() >= 3.0,
        "ratio = {}",
        search.integer_ratio.unwrap()
    );
    assert_eq!(
        search.ambiguity_search.order,
        vec![
            "G02".to_string(),
            "G03".to_string(),
            "G04".to_string(),
            "G05".to_string(),
        ]
    );
    let by_sat: BTreeMap<&str, &Hold> = holds.iter().map(|h| (h.sat_sd_id.as_str(), h)).collect();
    for &(id, _, ncyc) in &sats[1..] {
        let h = by_sat[id];
        assert_eq!(h.ref_sd_id, "G01");
        assert_eq!((h.fixed_m / lambda).round() as i64, ncyc, "{id}");
    }

    // AR arming gate: a threshold below any achievable baseline posterior
    // sigma withholds the fix; the same call without the gate fixes (above).
    let (gated_holds, gated_search) = search_and_hold(
        &state_post,
        &post.information,
        &epoch,
        AmbiguityScale {
            wavelengths_m: &wl,
            offsets_m: &off,
        },
        &[],
        SearchPolicy {
            float_only_systems: &[],
            ar_arming_sigma_m: Some(1.0e-9),
            ratio: SearchOpts {
                ratio_threshold: 3.0,
            },
        },
    )
    .unwrap();

    assert!(gated_holds.is_empty());
    assert_eq!(gated_search, None);
}

#[test]
fn update_epoch_grows_searches_and_carries_held_ambiguities() {
    let lambda = C_M_S / F_L1_HZ;
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let truth = [1.2, -0.85, 0.91];
    let rover = [base[0] + truth[0], base[1] + truth[1], base[2] + truth[2]];
    let sats: [(&str, [f64; 3], i64); 5] = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 3),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -7),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 12),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -4),
    ];
    let mk = |pos: [f64; 3], id: &str, ncyc: i64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(pos, base),
        base_phase_m: range_m(pos, base),
        rover_code_m: range_m(pos, rover),
        rover_phase_m: range_m(pos, rover) + (ncyc as f64) * lambda,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    let epoch = Epoch {
        references: vec![mk(sats[0].1, sats[0].0, sats[0].2)],
        nonref: sats[1..].iter().map(|&(id, p, n)| mk(p, id, n)).collect(),
        velocity_mps: None,
        dt_s: 0.0,
    };
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let wavelengths: BTreeMap<String, f64> = sats[1..]
        .iter()
        .map(|&(id, _, _)| (id.to_string(), lambda))
        .collect();
    let offsets: BTreeMap<String, f64> = sats[1..]
        .iter()
        .map(|&(id, _, _)| (id.to_string(), 0.0))
        .collect();
    let opts = UpdateOpts {
        hold_sigma_m: 1.0,
        position_tol_m: 1e-3,
        ambiguity_tol_m: 1e-6,
        max_iterations: 10,
        process_noise_baseline_sigma_m: 0.0,
        dynamics_model: DynamicsModel::ConstantPosition,
        float_only_systems: vec![],
        innovation_screen: None,
        report_residuals: true,
        force_report_iterate_failure: false,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: 3.0,
        },
    };

    let state = filter_state(&["G01"], [-30.0, 25.0, -10.0], 1.0e4, 1.0e4);

    assert_eq!(
        update_epoch(
            filter_state(&["G99"], [-30.0, 25.0, -10.0], 1.0e4, 1.0e4),
            &epoch,
            base,
            &model,
            &wavelengths,
            &offsets,
            &opts,
        ),
        Err(UpdateError::ReferenceChanged {
            system: "G".into(),
            expected: "G99".into(),
            actual: "G01".into()
        })
    );

    assert_eq!(
        update_epoch(
            state.clone(),
            &epoch,
            base,
            &model,
            &BTreeMap::new(),
            &offsets,
            &opts,
        ),
        Err(UpdateError::MissingWavelength("G02".into()))
    );

    let mut forced_report_failure_opts = opts.clone();
    forced_report_failure_opts.force_report_iterate_failure = true;
    assert_eq!(
        update_epoch(
            state.clone(),
            &epoch,
            base,
            &model,
            &wavelengths,
            &offsets,
            &forced_report_failure_opts,
        ),
        Err(UpdateError::SingularGeometry)
    );

    let mut presized = state.clone();
    for &(id, _, _) in &sats {
        presized.ensure_ambiguity(id, 0.0);
    }
    let mut noisy_opts = opts.clone();
    noisy_opts.process_noise_baseline_sigma_m = 30.0;
    let first_presized_static = update_epoch(
        presized.clone(),
        &epoch,
        base,
        &model,
        &wavelengths,
        &offsets,
        &opts,
    )
    .unwrap();
    let first_presized_kinematic = update_epoch(
        presized,
        &epoch,
        base,
        &model,
        &wavelengths,
        &offsets,
        &noisy_opts,
    )
    .unwrap();
    assert_eq!(first_presized_kinematic.state.epoch_count, 1);
    assert_eq!(
        first_presized_kinematic.state.baseline_m,
        first_presized_static.state.baseline_m
    );
    assert_eq!(
        first_presized_kinematic.reported_baseline_m,
        first_presized_static.reported_baseline_m
    );

    let first = update_epoch(state, &epoch, base, &model, &wavelengths, &offsets, &opts).unwrap();
    let search = first.search.as_ref().unwrap();

    assert_eq!(first.state.epoch_count, 1);
    assert_eq!(
        first.state.baseline_m.map(f64::to_bits),
        [
            4608083138681313863,
            13829203375495744273,
            4606371768656218455,
        ]
    );
    assert_eq!(
        first.reported_baseline_m.map(f64::to_bits),
        [
            4608083138690834122,
            13829203375529663515,
            4606371769568303258,
        ]
    );
    assert_eq!(
        first.state.sd_ambiguity_ids,
        vec!["G01", "G02", "G03", "G04", "G05"]
    );
    assert_eq!(
        first
            .state
            .sd_ambiguities_m
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![
            13613108460524664361,
            4603317259442403299,
            13832049901430948297,
            4612324458027589563,
            13828403307552428188,
        ]
    );
    assert_eq!(
        first
            .state
            .information
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![
            4681264931201542695,
            13881346678350041970,
            4666156153696531199,
            13896947148113222101,
            4674865055982464510,
            13898773047604719128,
            4676080871945708629,
            13890154270958116157,
            13881346678350041952,
            4680497685361993378,
            4622624441926614725,
            13887762493859457285,
            13895997895502045124,
            4675192752126573711,
            4674976635521603952,
            13898901733286082812,
            4666156153696531201,
            4622624441926614981,
            4660888360597764781,
            13891709647419622879,
            4657217059163069282,
            4653808952727724788,
            4660496225342060666,
            4662784245674674448,
            13896947148113222104,
            13887762493859457288,
            13891709647419622878,
            4676341348954248088,
            13890706049115328059,
            13890706049115328059,
            13890706049115328057,
            13890706049115328056,
            4674865055982464510,
            13895997895502045124,
            4657217059163069282,
            13890706049115328059,
            4676341211515294615,
            13890706049115328057,
            13890706049115328057,
            13890706049115328057,
            13898773047604719128,
            4675192752126573711,
            4653808952727724788,
            13890706049115328059,
            13890706049115328057,
            4676341211515294615,
            13890706049115328057,
            13890706049115328057,
            4676080871945708629,
            4674976635521603952,
            4660496225342060666,
            13890706049115328057,
            13890706049115328057,
            13890706049115328057,
            4676341211515294615,
            13890706049115328057,
            13890154270958116157,
            13898901733286082812,
            4662784245674674448,
            13890706049115328056,
            13890706049115328057,
            13890706049115328057,
            13890706049115328057,
            4676341211515294615,
        ]
    );
    assert!(first.integer_fixed);
    assert!(
        first.integer_ratio >= 3.0,
        "ratio = {}",
        first.integer_ratio
    );
    assert_eq!(
        first
            .residuals
            .iter()
            .map(|r| {
                (
                    r.satellite_id.as_str(),
                    r.ambiguity_id.as_str(),
                    r.code_m.to_bits(),
                    r.phase_m.to_bits(),
                    r.code_sigma_m.to_bits(),
                    r.phase_sigma_m.to_bits(),
                    r.code_normalized.to_bits(),
                    r.phase_normalized.to_bits(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            (
                "G02",
                "G02",
                4498533077789704192,
                13687977338317307904,
                4603579539098121011,
                4573567551181324026,
                4501629302533521408,
                13721327623631885653,
            ),
            (
                "G03",
                "G03",
                4491214728395227136,
                4482129680173891584,
                4603579539098121011,
                4573567551181324026,
                4494780078100228779,
                4515331938674453163,
            ),
            (
                "G04",
                "G04",
                4495155378069176320,
                4482316749085081600,
                4603579539098121011,
                4573567551181324026,
                4498533077789704192,
                4515575517985898496,
            ),
            (
                "G05",
                "G05",
                4499658977696546816,
                4480564203786076160,
                4603579539098121011,
                4573567551181324026,
                4503036677417074688,
                4513630423486911829,
            ),
        ]
    );
    assert_eq!(first.newly_fixed, vec!["G02", "G03", "G04", "G05"]);
    assert_eq!(first.fixed_ids, vec!["G02", "G03", "G04", "G05"]);
    assert_eq!(
        first
            .state
            .fixed_m
            .iter()
            .map(|(id, v)| (id.as_str(), v.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            ("G02", 4603317258628710491),
            ("G03", 13832049901624762474),
            ("G04", 4612324457883451483),
            ("G05", 13828403308511297657),
        ]
    );
    assert_eq!(search.integer_status, IntegerStatus::Fixed);
    assert_eq!(search.integer_method, "lambda");
    assert_eq!(
        search.integer_ratio.map(f64::to_bits),
        Some(4801742217831138971)
    );
    assert_eq!(
        (
            search.integer_best_score.map(f64::to_bits),
            search.integer_second_best_score.map(f64::to_bits),
            search.integer_candidates,
        ),
        (Some(4413362087218700288), Some(4607988026349360128), 2)
    );
    assert_eq!(
        search.ambiguity_search.order,
        vec![
            "G02".to_string(),
            "G03".to_string(),
            "G04".to_string(),
            "G05".to_string(),
        ]
    );
    assert_eq!(
        search
            .ambiguity_search
            .float_cycles
            .iter()
            .map(|(id, v)| (id.as_str(), v.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            ("G02", 4613937819310069329),
            ("G03", 13842939354375436968),
            ("G04", 4622945017685176834),
            ("G05", 13839561653649810905),
        ]
    );
    assert_eq!(
        search
            .ambiguity_search
            .covariance_cycles
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![
            4618986091170763471,
            4613489696586301028,
            4618532907155592128,
            4618649524167860868,
            4613489696586301028,
            4621020153654190212,
            4618067342216480117,
            4618140515825375320,
            4618532907155592128,
            4618067342216480117,
            4621586235636813543,
            4616844677784971110,
            4618649524167860868,
            4618140515825375320,
            4616844677784971110,
            4621547064612735951,
        ]
    );
    assert_eq!(
        search
            .ambiguity_search
            .covariance_inverse_cycles
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![
            4651267730143274806,
            4647397396204746397,
            13871682947449427258,
            13871981167164000694,
            4647397396204746397,
            4643398445417966949,
            13867581939454941704,
            13867903682842663122,
            13871682947449427258,
            13867581939454941704,
            4645166799438831687,
            4645540841134138012,
            13871981167164000694,
            13867903682842663122,
            4645540841134138012,
            4645944042038695186,
        ]
    );
    for (k, &t) in truth.iter().enumerate() {
        assert!((first.state.baseline_m[k] - t).abs() < 1e-3);
    }
    for &(id, _, ncyc) in &sats[1..] {
        assert_eq!(first.state.fixed_cycles[id], ncyc);
        assert!((first.state.fixed_m[id] - (ncyc as f64) * lambda).abs() < 1e-9);
    }

    let second = update_epoch(
        first.state.clone(),
        &epoch,
        base,
        &model,
        &wavelengths,
        &offsets,
        &opts,
    )
    .unwrap();

    assert!(second.integer_fixed);
    assert_eq!(second.state.epoch_count, 2);
    assert_eq!(second.integer_ratio, 0.0);
    assert_eq!(second.search, None);
    assert_eq!(
        second.state.baseline_m.map(f64::to_bits),
        [
            4608083138702578392,
            13829203375631608601,
            4606371770124687551,
        ]
    );
    assert_eq!(
        second
            .state
            .sd_ambiguities_m
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![
            13610600356374962006,
            4603317258935914865,
            13832049901570251117,
            4612324457928310211,
            13828403308152410048,
        ]
    );
    assert!(second.newly_fixed.is_empty());
    assert_eq!(second.fixed_ids, vec!["G02", "G03", "G04", "G05"]);
    assert_eq!(second.state.fixed_cycles, first.state.fixed_cycles);
}

#[test]
fn float_batch_solver_has_frozen_bits_golden() {
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let truth = [1.2, -0.85, 0.91];
    let rover = [base[0] + truth[0], base[1] + truth[1], base[2] + truth[2]];
    let sats: [(&str, [f64; 3], f64); 5] = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0.0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 0.6),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -1.4),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 1.0),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -0.3),
    ];
    let mk = |pos: [f64; 3], id: &str, amb: f64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(pos, base),
        base_phase_m: range_m(pos, base),
        rover_code_m: range_m(pos, rover),
        rover_phase_m: range_m(pos, rover) + amb,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    let epoch = Epoch {
        references: vec![mk(sats[0].1, sats[0].0, sats[0].2)],
        nonref: sats[1..].iter().map(|&(id, p, a)| mk(p, id, a)).collect(),
        velocity_mps: None,
        dt_s: 0.0,
    };
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let ambiguity_ids = vec![
        "G02".to_string(),
        "G03".to_string(),
        "G04".to_string(),
        "G05".to_string(),
    ];

    let solution = solve_float_baseline(
        &[epoch],
        base,
        &ambiguity_ids,
        [-30.0, 25.0, -10.0],
        &model,
        FloatSolveOpts {
            position_tol_m: 1.0e-3,
            ambiguity_tol_m: 1.0e-6,
            max_iterations: 10,
        },
        None,
    )
    .unwrap();

    assert_eq!(
        solution.baseline_m.map(f64::to_bits),
        [
            4608083138711216895,
            13829203375771827050,
            4606371770980413522
        ]
    );
    assert_eq!(
        solution
            .ambiguities_m
            .iter()
            .map(|(id, v)| (id.as_str(), v.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            ("G02", 4603579539121730254),
            ("G03", 13832355895495834363),
            ("G04", 4607182418796748041),
            ("G05", 13822447976346096982),
        ]
    );
    assert_eq!(
        solution
            .ambiguity_covariance_m
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![
            4597627153645619126,
            4591973129486990364,
            4597102015884777572,
            4597237148704408334,
            4591973129487002460,
            4599269118287750919,
            4596562531490693629,
            4596647323280684590,
            4597102015884781964,
            4596562531490677596,
            4599925079199018693,
            4595145739977959156,
            4597237148704409051,
            4596647323280690710,
            4595145739977965506,
            4599879688953613067,
        ]
    );
    assert_eq!(
        solution
            .ambiguity_covariance_inverse_m
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        vec![
            4672681834198124872,
            4668724730942489085,
            13893399532418635347,
            13893759549429636983,
            4668724730942489085,
            4664818209387620136,
            13889346020803542089,
            13889623679292115552,
            13893399532418635348,
            13889346020803542085,
            4666799767912208517,
            4667122558855991330,
            13893759549429636981,
            13889623679292115555,
            4667122558855991330,
            4667470513648549313,
        ]
    );
    assert_eq!(
        (
            solution.code_rms_m.to_bits(),
            solution.phase_rms_m.to_bits(),
            solution.weighted_rms_m.to_bits(),
            solution.iterations,
            solution.converged,
            solution.status,
            solution.n_observations,
            solution.residuals.len(),
        ),
        (
            4476578029606273024,
            4476286243613641754,
            4507120841446053534,
            3,
            true,
            FloatSolveStatus::StateTolerance,
            8,
            4,
        )
    );
}

#[test]
fn fixed_batch_solver_has_frozen_bits_golden() {
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let truth = [1.2, -0.85, 0.91];
    let rover = [base[0] + truth[0], base[1] + truth[1], base[2] + truth[2]];
    let lambda = C_M_S / F_L1_HZ;
    let sats: [(&str, [f64; 3], i64); 5] = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 4),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -7),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 9),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -3),
    ];
    let mk = |pos: [f64; 3], id: &str, cycles: i64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(pos, base),
        base_phase_m: range_m(pos, base),
        rover_code_m: range_m(pos, rover),
        rover_phase_m: range_m(pos, rover) + cycles as f64 * lambda,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    let epoch = Epoch {
        references: vec![mk(sats[0].1, sats[0].0, sats[0].2)],
        nonref: sats[1..].iter().map(|&(id, p, c)| mk(p, id, c)).collect(),
        velocity_mps: None,
        dt_s: 0.0,
    };
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let ambiguity_ids = vec![
        "G02".to_string(),
        "G03".to_string(),
        "G04".to_string(),
        "G05".to_string(),
    ];
    let float = solve_float_baseline(
        std::slice::from_ref(&epoch),
        base,
        &ambiguity_ids,
        [-30.0, 25.0, -10.0],
        &model,
        FloatSolveOpts {
            position_tol_m: 1.0e-3,
            ambiguity_tol_m: 1.0e-6,
            max_iterations: 10,
        },
        None,
    )
    .unwrap();
    let wavelengths = ambiguity_ids
        .iter()
        .map(|id| (id.clone(), lambda))
        .collect::<BTreeMap<_, _>>();
    let offsets = ambiguity_ids
        .iter()
        .map(|id| (id.clone(), 0.0))
        .collect::<BTreeMap<_, _>>();
    let ambiguity_satellites = ambiguity_ids
        .iter()
        .map(|id| (id.clone(), id.clone()))
        .collect::<BTreeMap<_, _>>();

    let fixed = solve_fixed_baseline(
        &[epoch],
        base,
        AmbiguitySet {
            ids: &ambiguity_ids,
            satellites: &ambiguity_satellites,
            scale: AmbiguityScale {
                wavelengths_m: &wavelengths,
                offsets_m: &offsets,
            },
            float_only_systems: &[],
        },
        FloatPrior {
            baseline_m: float.baseline_m,
            ambiguities_m: &float.ambiguities_m,
            covariance_m: &float.ambiguity_covariance_m,
        },
        &model,
        FixedSolveOpts {
            position_tol_m: 1.0e-3,
            ambiguity_tol_m: 1.0e-6,
            max_iterations: 10,
            ratio_threshold: 3.0,
            partial_ambiguity_resolution: false,
            partial_min_ambiguities: 4,
        },
        None,
    )
    .unwrap();

    assert_eq!(
        fixed.baseline_m.map(f64::to_bits),
        [
            4608083138718600110,
            13829203375783517462,
            4606371770890763384
        ]
    );
    assert_eq!(
        fixed.fixed_ambiguities_cycles,
        vec![
            ("G02".to_string(), 4),
            ("G03".to_string(), -7),
            ("G04".to_string(), 9),
            ("G05".to_string(), -3),
        ]
    );
    assert_eq!(
        fixed
            .fixed_ambiguities_m
            .iter()
            .map(|(id, value)| (id.as_str(), value.to_bits()))
            .collect::<Vec<_>>(),
        vec![
            ("G02", 4605031271656521849),
            ("G03", 13832049901624762474),
            ("G04", 4610391877797798024),
            ("G05", 13826689295483486299),
        ]
    );
    assert_eq!(
        (
            fixed.code_rms_m.to_bits(),
            fixed.phase_rms_m.to_bits(),
            fixed.weighted_rms_m.to_bits(),
            fixed.iterations,
            fixed.converged,
            fixed.status,
            fixed.search.integer_ratio.map(f64::to_bits),
            fixed.search.integer_best_score.map(f64::to_bits),
            fixed.search.integer_second_best_score.map(f64::to_bits),
            fixed.n_observations,
            fixed.residuals.len(),
        ),
        (
            4476578029606273024,
            4474709011174730901,
            4505668759362556277,
            1,
            true,
            FloatSolveStatus::StateTolerance,
            Some(4804512108740441140),
            Some(4410367603048024711),
            Some(4607988026904240128),
            8,
            4,
        )
    );
    assert_eq!(
        fixed
            .residuals
            .iter()
            .map(|r| {
                (
                    r.satellite_id.as_str(),
                    r.code_m.to_bits(),
                    r.phase_m.to_bits(),
                    r.code_normalized.to_bits(),
                    r.phase_normalized.to_bits(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            ("G02", 0, 13696818721769652224, 0, 13730118916893922645),
            ("G03", 0, 13692895372223971328, 0, 13726370851227085482),
            ("G04", 0, 13699659941420204032, 0, 13732646842749943808),
            (
                "G05",
                13704453666088419328,
                13699998420545044480,
                13707456065839999659,
                13732898687286946474,
            ),
        ]
    );
    assert_eq!(fixed.search.integer_status, IntegerStatus::Fixed);
    assert_eq!(fixed.search.integer_method, "lambda");
    assert_eq!(
        (
            fixed.search.ambiguity_search.order.clone(),
            fixed
                .search
                .ambiguity_search
                .float_cycles
                .iter()
                .map(|(id, value)| (id.as_str(), value.to_bits()))
                .collect::<Vec<_>>(),
            fixed
                .search
                .ambiguity_search
                .covariance_cycles
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            fixed
                .search
                .ambiguity_search
                .covariance_inverse_cycles
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        ),
        (
            vec![
                "G02".to_string(),
                "G03".to_string(),
                "G04".to_string(),
                "G05".to_string(),
            ],
            vec![
                ("G02", 4616189618058813080),
                ("G03", 13842939354626956446),
                ("G04", 4621256167628413179),
                ("G05", 13837309855081934673),
            ],
            vec![
                4618986091316149770,
                4613489696968715012,
                4618532907311373116,
                4618649524391582917,
                4613489696968715012,
                4621020153805794876,
                4618067342432090414,
                4618140516151120674,
                4618532907311373116,
                4618067342432090414,
                4621586235727367051,
                4616844678019924302,
                4618649524391582917,
                4618140516151120674,
                4616844678019924302,
                4621547064799804680,
            ],
            vec![
                4651267730143278344,
                4647397396204759295,
                13871682947449432362,
                13871981167164004052,
                4647397396204759295,
                4643398445417946591,
                13867581939454945248,
                13867903682842689616,
                13871682947449432362,
                13867581939454945248,
                4645166799438828770,
                4645540841134148320,
                13871981167164004052,
                13867903682842689616,
                4645540841134148320,
                4645944042038664782,
            ],
        )
    );
    assert_eq!(
        fixed
            .search
            .ambiguity_offsets_m
            .iter()
            .map(|(id, value)| (id.as_str(), value.to_bits()))
            .collect::<Vec<_>>(),
        vec![("G02", 0), ("G03", 0), ("G04", 0), ("G05", 0)]
    );
    assert_eq!(
        (
            fixed.search.partial.enabled,
            fixed.search.partial.fixed,
            fixed.search.partial.fixed_ambiguities,
            fixed.search.partial.free_ambiguities,
            fixed.search.partial.full_set,
            fixed.search.partial.exhaustive_subsets_evaluated,
        ),
        (
            false,
            false,
            vec![
                "G02".to_string(),
                "G03".to_string(),
                "G04".to_string(),
                "G05".to_string(),
            ],
            Vec::<String>::new(),
            None,
            None,
        )
    );
}

#[test]
fn residual_validated_fixed_solver_excludes_biased_satellite_with_frozen_bits() {
    let base = [4_075_580.0, 931_854.0, 4_801_568.0];
    let truth = [1.2, -0.85, 0.91];
    let rover = [base[0] + truth[0], base[1] + truth[1], base[2] + truth[2]];
    let lambda = C_M_S / F_L1_HZ;
    let sats: [(&str, [f64; 3], i64); 5] = [
        ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0),
        ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 4),
        ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -7),
        ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 9),
        ("G05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -3),
    ];
    let mk = |pos: [f64; 3], id: &str, cycles: i64, rover_code_noise_m: f64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(pos, base),
        base_phase_m: range_m(pos, base),
        rover_code_m: range_m(pos, rover) + rover_code_noise_m,
        rover_phase_m: range_m(pos, rover) + cycles as f64 * lambda,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    let epochs = [40.0, -40.0, 40.0]
        .into_iter()
        .map(|g05_noise| Epoch {
            references: vec![mk(sats[0].1, sats[0].0, sats[0].2, 0.0)],
            nonref: sats[1..]
                .iter()
                .map(|&(id, p, c)| mk(p, id, c, if id == "G05" { g05_noise } else { 0.0 }))
                .collect(),
            velocity_mps: None,
            dt_s: 0.0,
        })
        .collect::<Vec<_>>();
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    };
    let ambiguity_ids = vec![
        "G02".to_string(),
        "G03".to_string(),
        "G04".to_string(),
        "G05".to_string(),
    ];
    let ambiguity_satellites = ambiguity_ids
        .iter()
        .map(|id| (id.clone(), id.clone()))
        .collect::<BTreeMap<_, _>>();
    let wavelengths = ambiguity_ids
        .iter()
        .map(|id| (id.clone(), lambda))
        .collect::<BTreeMap<_, _>>();
    let offsets = ambiguity_ids
        .iter()
        .map(|id| (id.clone(), 0.0))
        .collect::<BTreeMap<_, _>>();
    let float_opts = FloatSolveOpts {
        position_tol_m: 1.0e-3,
        ambiguity_tol_m: 1.0e-6,
        max_iterations: 10,
    };
    let fixed_opts = FixedSolveOpts {
        position_tol_m: 1.0e-3,
        ambiguity_tol_m: 1.0e-6,
        max_iterations: 10,
        ratio_threshold: 3.0,
        partial_ambiguity_resolution: false,
        partial_min_ambiguities: 4,
    };

    let solve_with_residual = |residual: ResidualValidationOpts| {
        solve_fixed_baseline_validated(
            &epochs,
            base,
            AmbiguitySet {
                ids: &ambiguity_ids,
                satellites: &ambiguity_satellites,
                scale: AmbiguityScale {
                    wavelengths_m: &wavelengths,
                    offsets_m: &offsets,
                },
                float_only_systems: &[],
            },
            [-30.0, 25.0, -10.0],
            &model,
            ValidatedFixedSolveOpts {
                float: float_opts,
                fixed: fixed_opts,
                residual,
            },
            None,
        )
    };

    for (threshold_sigma, kind) in [
        (f64::NAN, RtkInputErrorKind::NonFinite),
        (f64::INFINITY, RtkInputErrorKind::NonFinite),
        (0.0, RtkInputErrorKind::NotPositive),
        (-1.0, RtkInputErrorKind::NotPositive),
    ] {
        let err = solve_with_residual(ResidualValidationOpts {
            threshold_sigma: Some(threshold_sigma),
            max_exclusions: 1,
        })
        .unwrap_err();
        assert_eq!(
            err,
            ValidatedFixedSolveError::Fixed(FixedSolveError::InvalidInput {
                field: "rtk.residual_threshold_sigma",
                kind,
            })
        );
    }

    let failed = solve_with_residual(ResidualValidationOpts {
        threshold_sigma: Some(6.0),
        max_exclusions: 0,
    })
    .unwrap_err();
    let ValidatedFixedSolveError::ResidualValidationFailed {
        outlier,
        exclusions,
    } = failed
    else {
        panic!("unexpected residual-validation error: {failed:?}");
    };
    assert_eq!(exclusions, Vec::new());
    assert_eq!(outlier.satellite_id, "G05");
    assert_eq!(outlier.kind, ResidualComponentKind::Code);
    assert!(outlier.normalized_residual.abs() > outlier.threshold_sigma);

    let solved = solve_with_residual(ResidualValidationOpts {
        threshold_sigma: Some(6.0),
        max_exclusions: 1,
    })
    .unwrap();
    let meta = solved.residual_validation.as_ref().unwrap();

    assert_eq!(meta.excluded_sats, vec!["G05"]);
    assert_eq!(meta.exclusions.len(), 1);
    assert_eq!(meta.exclusions[0].satellite_id, "G05");
    assert_eq!(meta.exclusions[0].kind, ResidualComponentKind::Code);
    assert_eq!(solved.ambiguity_ids, vec!["G02", "G03", "G04"]);
    assert_eq!(
        solved.fixed_solution.fixed_ambiguities_cycles,
        vec![
            ("G02".to_string(), 4),
            ("G03".to_string(), -7),
            ("G04".to_string(), 9),
        ]
    );
    assert_eq!(
        solved.fixed_solution.baseline_m.map(f64::to_bits),
        [
            4608083138710742523,
            13829203375790525355,
            4606371770872105237
        ]
    );
    assert_eq!(
        solved
            .fixed_solution
            .residuals
            .iter()
            .map(|r| {
                (
                    r.satellite_id.as_str(),
                    r.code_m.to_bits(),
                    r.phase_m.to_bits(),
                    r.code_normalized.to_bits(),
                    r.phase_normalized.to_bits(),
                )
            })
            .collect::<Vec<_>>(),
        vec![
            ("G02", 0, 13696818721769652224, 0, 13730118916893922645),
            ("G03", 0, 13692895372223971328, 0, 13726370851227085482),
            (
                "G04",
                4481081629233643520,
                4476723092126695424,
                4484084028985223851,
                4509652572875434666,
            ),
            ("G02", 0, 13696818721769652224, 0, 13730118916893922645),
            ("G03", 0, 13692895372223971328, 0, 13726370851227085482),
            (
                "G04",
                4481081629233643520,
                4476723092126695424,
                4484084028985223851,
                4509652572875434666,
            ),
            ("G02", 0, 13696818721769652224, 0, 13730118916893922645),
            ("G03", 0, 13692895372223971328, 0, 13726370851227085482),
            (
                "G04",
                4481081629233643520,
                4476723092126695424,
                4484084028985223851,
                4509652572875434666,
            ),
        ]
    );
}

#[test]
fn velocity_prediction_moves_prior_after_first_epoch_only_when_enabled() {
    let epoch = Epoch {
        references: Vec::new(),
        nonref: Vec::new(),
        velocity_mps: Some([0.5, -1.0, 2.0]),
        dt_s: 4.0,
    };
    let mut opts = UpdateOpts {
        hold_sigma_m: 1.0,
        position_tol_m: 1e-3,
        ambiguity_tol_m: 1e-6,
        max_iterations: 10,
        process_noise_baseline_sigma_m: 0.0,
        dynamics_model: DynamicsModel::VelocityPropagated,
        float_only_systems: vec![],
        innovation_screen: None,
        report_residuals: false,
        force_report_iterate_failure: false,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: 3.0,
        },
    };

    let mut first = filter_state(&["G01"], [1.0, 2.0, 3.0], 10.0, 10.0);
    propagate_baseline_mean(&mut first, &epoch, &opts);
    assert_eq!(first.baseline_m, [1.0, 2.0, 3.0]);

    let mut later = filter_state(&["G01"], [1.0, 2.0, 3.0], 10.0, 10.0);
    later.epoch_count = 1;
    propagate_baseline_mean(&mut later, &epoch, &opts);
    assert_eq!(later.baseline_m, [3.0, -2.0, 11.0]);

    opts.dynamics_model = DynamicsModel::ConstantPosition;
    let mut default_path = filter_state(&["G01"], [1.0, 2.0, 3.0], 10.0, 10.0);
    default_path.epoch_count = 1;
    propagate_baseline_mean(&mut default_path, &epoch, &opts);
    assert_eq!(default_path.baseline_m, [1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------
// Multi-GNSS per-system references (Track B)
// ---------------------------------------------------------------------

const MG_BASE: [f64; 3] = [4_075_580.0, 931_854.0, 4_801_568.0];
const MG_TRUTH: [f64; 3] = [1.2, -0.85, 0.91];
// G ref + 3 G nonref, E ref + 2 E nonref; integer cycle ambiguities.
const MG_SATS: [(&str, [f64; 3], i64); 7] = [
    ("G01", [15_000_000.0, 7_000_000.0, 21_000_000.0], 0),
    ("G02", [-12_000_000.0, 18_000_000.0, 19_000_000.0], 3),
    ("G03", [20_000_000.0, -10_000_000.0, 17_000_000.0], -7),
    ("G04", [-19_000_000.0, -13_000_000.0, 20_000_000.0], 12),
    ("E05", [9_000_000.0, 22_000_000.0, 16_000_000.0], -4),
    ("E07", [-6_000_000.0, -20_000_000.0, 18_000_000.0], 8),
    ("E11", [23_000_000.0, 2_000_000.0, 17_500_000.0], 5),
];

fn mg_epoch(lambda: f64) -> Epoch {
    let rover = [
        MG_BASE[0] + MG_TRUTH[0],
        MG_BASE[1] + MG_TRUTH[1],
        MG_BASE[2] + MG_TRUTH[2],
    ];
    let mk = |pos: [f64; 3], id: &str, ncyc: i64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(pos, MG_BASE),
        base_phase_m: range_m(pos, MG_BASE),
        rover_code_m: range_m(pos, rover),
        rover_phase_m: range_m(pos, rover) + (ncyc as f64) * lambda,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    Epoch {
        references: vec![
            mk(MG_SATS[0].1, MG_SATS[0].0, MG_SATS[0].2),
            mk(MG_SATS[4].1, MG_SATS[4].0, MG_SATS[4].2),
        ],
        nonref: [1usize, 2, 3, 5, 6]
            .iter()
            .map(|&i| mk(MG_SATS[i].1, MG_SATS[i].0, MG_SATS[i].2))
            .collect(),
        velocity_mps: None,
        dt_s: 0.0,
    }
}

fn mg_state() -> FilterState {
    let mut state = filter_state(&["G01", "E05"], [-30.0, 25.0, -10.0], 1.0e4, 1.0e4);
    for &(id, _, _) in &MG_SATS {
        state.ensure_ambiguity(id, 0.0);
    }
    state
}

fn mg_model() -> MeasModel {
    MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    }
}

fn mg_maps(lambda: f64) -> (BTreeMap<String, f64>, BTreeMap<String, f64>) {
    let nonref = ["G02", "G03", "G04", "E07", "E11"];
    (
        nonref.iter().map(|&id| (id.to_string(), lambda)).collect(),
        nonref.iter().map(|&id| (id.to_string(), 0.0)).collect(),
    )
}

fn mg_opts() -> UpdateOpts {
    UpdateOpts {
        hold_sigma_m: 1.0,
        position_tol_m: 1e-3,
        ambiguity_tol_m: 1e-6,
        max_iterations: 10,
        process_noise_baseline_sigma_m: 0.0,
        dynamics_model: DynamicsModel::ConstantPosition,
        float_only_systems: vec![],
        innovation_screen: None,
        report_residuals: false,
        force_report_iterate_failure: false,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: 3.0,
        },
    }
}

#[test]
fn multi_system_rows_pair_with_their_own_reference() {
    let lambda = C_M_S / F_L1_HZ;
    let epoch = mg_epoch(lambda);
    let state = mg_state();
    let rows = epoch_dd_rows(&epoch, MG_BASE, &state, &mg_model()).unwrap();

    assert_eq!(rows.len(), 10);
    for row in &rows {
        let expected_ref = if row.sat.starts_with('G') {
            "G01"
        } else {
            "E05"
        };
        assert_eq!(row.ref_sat, expected_ref, "{}", row.sat);
        if row.kind == RowKind::Phase {
            // +1 at the sat SD column, -1 at the OWN system's reference column.
            let sat_col = 3 + state.ambiguity_pos(&row.sd_ambiguity_id).unwrap();
            let ref_col = 3 + state.ambiguity_pos(expected_ref).unwrap();
            assert_eq!(row.h[sat_col], 1.0);
            assert_eq!(row.h[ref_col], -1.0);
        }
    }

    // A non-reference satellite whose system has no reference makes the row
    // builder return None (update_epoch guards this with a typed error).
    let mut orphan = epoch.clone();
    orphan.references.retain(|r| r.sat != "E05");
    assert!(epoch_dd_rows(&orphan, MG_BASE, &state, &mg_model()).is_none());
}

#[test]
fn multi_system_update_fixes_per_system_integers() {
    let lambda = C_M_S / F_L1_HZ;
    let epoch = mg_epoch(lambda);
    let (wl, off) = mg_maps(lambda);
    let update = update_epoch(
        mg_state(),
        &epoch,
        MG_BASE,
        &mg_model(),
        &wl,
        &off,
        &mg_opts(),
    )
    .unwrap();

    assert!(update.integer_fixed);
    assert_eq!(update.fixed_ids, vec!["E07", "E11", "G02", "G03", "G04"]);
    for (k, &t) in MG_TRUTH.iter().enumerate() {
        assert!((update.state.baseline_m[k] - t).abs() < 1e-3);
    }
    // Fixed integers are the per-system DDs (sat - own reference).
    let cycles: BTreeMap<&str, i64> = MG_SATS.iter().map(|&(id, _, n)| (id, n)).collect();
    for (id, &fixed) in &update.state.fixed_cycles {
        let ref_id = if id.starts_with('G') { "G01" } else { "E05" };
        assert_eq!(fixed, cycles[id.as_str()] - cycles[ref_id], "{id}");
    }
}

#[test]
fn float_only_systems_stay_out_of_the_search_set() {
    let lambda = C_M_S / F_L1_HZ;
    let epoch = mg_epoch(lambda);
    let (wl, off) = mg_maps(lambda);
    let mut opts = mg_opts();
    opts.float_only_systems = vec!["E".to_string()];
    let update = update_epoch(mg_state(), &epoch, MG_BASE, &mg_model(), &wl, &off, &opts).unwrap();

    // Galileo never enters the integer search; GPS still fixes.
    assert!(update.integer_fixed);
    assert_eq!(update.fixed_ids, vec!["G02", "G03", "G04"]);
    assert!(!update
        .state
        .fixed_cycles
        .keys()
        .any(|id| id.starts_with('E')));
}

#[test]
fn multi_system_reference_guards_are_typed() {
    let lambda = C_M_S / F_L1_HZ;
    let epoch = mg_epoch(lambda);
    let (wl, off) = mg_maps(lambda);

    // Untracked constellation.
    let g_only = filter_state(&["G01"], [-30.0, 25.0, -10.0], 1.0e4, 1.0e4);
    assert_eq!(
        update_epoch(g_only, &epoch, MG_BASE, &mg_model(), &wl, &off, &mg_opts()),
        Err(UpdateError::UnknownReferenceSystem("E".into()))
    );

    // Reference ambiguity arc changed within a tracked constellation.
    let changed = filter_state(&["G01", "E99"], [-30.0, 25.0, -10.0], 1.0e4, 1.0e4);
    assert_eq!(
        update_epoch(changed, &epoch, MG_BASE, &mg_model(), &wl, &off, &mg_opts()),
        Err(UpdateError::ReferenceChanged {
            system: "E".into(),
            expected: "E99".into(),
            actual: "E05".into()
        })
    );

    // Non-reference satellite with no same-system reference this epoch.
    let mut orphan = epoch.clone();
    orphan.references.retain(|r| r.sat != "E05");
    let state = filter_state(&["G01", "E05"], [-30.0, 25.0, -10.0], 1.0e4, 1.0e4);
    assert_eq!(
        update_epoch(state, &orphan, MG_BASE, &mg_model(), &wl, &off, &mg_opts()),
        Err(UpdateError::MissingSystemReference("E".into()))
    );
}

// ---------------------------------------------------------------------------
// Phase-2 P0: row-level double-difference golden traces.
//
// These freeze the per-row double-difference design (`h`), prefit residual
// (`y`), single-difference variances, and diagnostic weight for the three RTK
// solvers (sequential `update`, static `float`, static `fixed`) on one shared
// synthetic epoch. They make the later substrate extraction (P1/P2) provably
// behavior-preserving at the ROW level, not just at the final-solution level:
// any change to the DD geometry, observation differencing, or stochastic model
// shifts these bits. The fixtures are independent of (and additive to) the
// existing final-solution frozen-bits goldens.
// ---------------------------------------------------------------------------

const ROW_TRACE_BASE: [f64; 3] = [4_075_580.0, 931_854.0, 4_801_568.0];
const ROW_TRACE_TRUTH_BASELINE: [f64; 3] = [1.2, -0.85, 0.91];
// Linearization point deliberately far from truth so every prefit residual and
// design partial is exercised with a non-trivial value.
const ROW_TRACE_LINEARIZE_BASELINE: [f64; 3] = [-30.0, 25.0, -10.0];
const ROW_TRACE_G02_AMB: f64 = 0.6;
const ROW_TRACE_G03_AMB: f64 = -1.4;

fn row_trace_model() -> MeasModel {
    MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: false,
        stochastic: StochasticModel::Simple {
            elevation_weighting: false,
        },
    }
}

// One reference (G01) and two non-reference satellites (G02, G03), perfect
// synthetic observations: code = geometric range, phase = range + ambiguity.
fn row_trace_epoch() -> Epoch {
    let rover = [
        ROW_TRACE_BASE[0] + ROW_TRACE_TRUTH_BASELINE[0],
        ROW_TRACE_BASE[1] + ROW_TRACE_TRUTH_BASELINE[1],
        ROW_TRACE_BASE[2] + ROW_TRACE_TRUTH_BASELINE[2],
    ];
    let g01 = [15_000_000.0, 7_000_000.0, 21_000_000.0];
    let g02 = [-12_000_000.0, 18_000_000.0, 19_000_000.0];
    let g03 = [20_000_000.0, -10_000_000.0, 17_000_000.0];
    let mk = |pos: [f64; 3], id: &str, amb: f64| SatMeas {
        sat: id.into(),
        sd_ambiguity_id: id.into(),
        base_code_m: range_m(pos, ROW_TRACE_BASE),
        base_phase_m: range_m(pos, ROW_TRACE_BASE),
        rover_code_m: range_m(pos, rover),
        rover_phase_m: range_m(pos, rover) + amb,
        base_tx_pos: pos,
        rover_tx_pos: pos,
        pos,
    };
    Epoch {
        references: vec![mk(g01, "G01", 0.0)],
        nonref: vec![
            mk(g02, "G02", ROW_TRACE_G02_AMB),
            mk(g03, "G03", ROW_TRACE_G03_AMB),
        ],
        velocity_mps: None,
        dt_s: 0.0,
    }
}

// Flatten the per-row numerics (design vector, prefit residual, both
// single-difference variances, diagnostic weight) to frozen bits in row order.
fn dd_row_bits(rows: &[DdRow]) -> Vec<u64> {
    let mut bits = Vec::new();
    for r in rows {
        for &h in &r.h {
            bits.push(h.to_bits());
        }
        bits.push(r.y.to_bits());
        bits.push(r.sd_variance_m2.to_bits());
        bits.push(r.ref_sd_variance_m2.to_bits());
        bits.push(r.weight.to_bits());
    }
    bits
}

fn dd_row_shape(rows: &[DdRow]) -> Vec<(RowKind, &str, &str, usize)> {
    rows.iter()
        .map(|r| (r.kind, r.sat.as_str(), r.ref_sat.as_str(), r.h.len()))
        .collect()
}

#[test]
fn update_dd_rows_have_frozen_bits_golden() {
    let epoch = row_trace_epoch();
    let model = row_trace_model();
    let mut state = filter_state(&["G01"], ROW_TRACE_LINEARIZE_BASELINE, 10.0, 5.7);
    state.ensure_ambiguity("G01", 0.0);
    state.ensure_ambiguity("G02", ROW_TRACE_G02_AMB);
    state.ensure_ambiguity("G03", ROW_TRACE_G03_AMB);

    let rows = epoch_dd_rows(&epoch, ROW_TRACE_BASE, &state, &model).unwrap();

    assert_eq!(
        dd_row_shape(&rows),
        vec![
            (RowKind::Code, "G02", "G01", 6),
            (RowKind::Phase, "G02", "G01", 6),
            (RowKind::Code, "G03", "G01", 6),
            (RowKind::Phase, "G03", "G01", 6),
        ]
    );
    assert_eq!(dd_row_bits(&rows).as_slice(), UPDATE_DD_ROW_GOLDEN);
}

#[test]
fn float_dd_rows_have_frozen_bits_golden() {
    let epoch = row_trace_epoch();
    let model = row_trace_model();
    let ambiguity_ids = vec!["G02".to_string(), "G03".to_string()];

    let rows = float_epoch_rows(
        ROW_TRACE_BASE,
        &epoch,
        &ambiguity_ids,
        ROW_TRACE_LINEARIZE_BASELINE,
        vec![ROW_TRACE_G02_AMB, ROW_TRACE_G03_AMB],
        &model,
    )
    .unwrap();

    assert_eq!(
        dd_row_shape(&rows),
        vec![
            (RowKind::Code, "G02", "G01", 5),
            (RowKind::Phase, "G02", "G01", 5),
            (RowKind::Code, "G03", "G01", 5),
            (RowKind::Phase, "G03", "G01", 5),
        ]
    );
    assert_eq!(dd_row_bits(&rows).as_slice(), FLOAT_DD_ROW_GOLDEN);
}

#[test]
fn fixed_dd_rows_have_frozen_bits_golden() {
    let epoch = row_trace_epoch();
    let model = row_trace_model();
    // G02 still free, G03 held to its (truth) integer-metre ambiguity.
    let free_ids = vec![AmbiguityId::new("G02")];
    let mut fixed_m = BTreeMap::new();
    fixed_m.insert(AmbiguityId::new("G03"), ROW_TRACE_G03_AMB);

    let rows = fixed_epoch_rows(
        ROW_TRACE_BASE,
        &epoch,
        &free_ids,
        &fixed_m,
        ROW_TRACE_LINEARIZE_BASELINE,
        vec![ROW_TRACE_G02_AMB],
        &model,
    )
    .unwrap();

    assert_eq!(
        dd_row_shape(&rows),
        vec![
            (RowKind::Code, "G02", "G01", 4),
            (RowKind::Phase, "G02", "G01", 4),
            (RowKind::Code, "G03", "G01", 4),
            (RowKind::Phase, "G03", "G01", 4),
        ]
    );
    assert_eq!(dd_row_bits(&rows).as_slice(), FIXED_DD_ROW_GOLDEN);
}

// Generated by running each test once and freezing the observed bits; see the
// module comment. Regenerate only with a deliberate, reviewed behavior change.
const UPDATE_DD_ROW_GOLDEN: &[u64] = &[
    4607724878617960170,
    13822917776373150503,
    4598603506408569600,
    0,
    0,
    0,
    4631723385753698304,
    4595653203753948938,
    4595653203753948938,
    4613437418282476430,
    4607724878617960170,
    13822917776373150503,
    4598603506408569600,
    13830554455654793216,
    4607182418800017408,
    0,
    4631723385753908019,
    4535933887427947327,
    4535933887427947327,
    4673364711370944967,
    13818415843756610176,
    4605156947929057044,
    4598315760069752456,
    0,
    0,
    0,
    13850345733173018624,
    4595653203753948938,
    4595653203753948938,
    4613437418282476430,
    13818415843756610176,
    4605156947929057044,
    4598315760069752456,
    13830554455654793216,
    0,
    4607182418800017408,
    13850345733172599194,
    4535933887427947327,
    4535933887427947327,
    4673364711370944967,
];
const FLOAT_DD_ROW_GOLDEN: &[u64] = &[
    4607724878617960170,
    13822917776373150503,
    4598603506408569600,
    0,
    0,
    4631723385753698304,
    4595653203753948938,
    4595653203753948938,
    4610184818551597739,
    4607724878617960170,
    13822917776373150503,
    4598603506408569600,
    4607182418800017408,
    0,
    4631723385753908019,
    4535933887427947327,
    4535933887427947327,
    4640068078579045717,
    13818415843756610176,
    4605156947929057044,
    4598315760069752456,
    0,
    0,
    13850345733173018624,
    4595653203753948938,
    4595653203753948938,
    4610184818551597739,
    13818415843756610176,
    4605156947929057044,
    4598315760069752456,
    0,
    4607182418800017408,
    13850345733172599194,
    4535933887427947327,
    4535933887427947327,
    4640068078579045717,
];
const FIXED_DD_ROW_GOLDEN: &[u64] = &[
    4607724878617960170,
    13822917776373150503,
    4598603506408569600,
    0,
    4631723385753698304,
    4595653203753948938,
    4595653203753948938,
    4610184818551597739,
    4607724878617960170,
    13822917776373150503,
    4598603506408569600,
    4607182418800017408,
    4631723385753908019,
    4535933887427947327,
    4535933887427947327,
    4640068078579045717,
    13818415843756610176,
    4605156947929057044,
    4598315760069752456,
    0,
    13850345733173018624,
    4595653203753948938,
    4595653203753948938,
    4610184818551597739,
    13818415843756610176,
    4605156947929057044,
    4598315760069752456,
    0,
    13850345733172599194,
    4535933887427947327,
    4535933887427947327,
    4640068078579045717,
];
