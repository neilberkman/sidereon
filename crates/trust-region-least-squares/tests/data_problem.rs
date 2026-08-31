//! Native (default in-crate SVD) coverage for the data-driven / model / batch
//! layers. These run in the default `cargo test` (no host LAPACK): they exercise
//! the new public API end-to-end, prove the trait path is byte-identical to the
//! hand-wired closure path, prove the analytic Jacobian agrees with the 2-point
//! default, and prove the parallel leave-one-out matches an independent serial
//! drop-`i` solve bit-for-bit. Bit-exactness vs SciPy lives in
//! `data_problem_fixtures.rs` (host-LAPACK gated).

use trust_region_least_squares::batch::{
    solve_data_problem_drop_one, solve_data_problem_drop_one_serial, solve_drop_one,
    solve_perturbed,
};
use trust_region_least_squares::data::{solve_data_problem, BuiltinResidual, DataProblem};
use trust_region_least_squares::loss::Loss;
use trust_region_least_squares::model::{solve_model, ResidualModel};
use trust_region_least_squares::parity::assert_f64_slice_bits_eq;
use trust_region_least_squares::trf::{
    jacobian_2point, trf_no_bounds, JacobianFn, NalgebraThinSvd, ResidualFn, TrfError, TrfOptions,
    TrfResult,
};

fn linear_problem() -> BuiltinResidual {
    // residual = A x - b, A row-major 4x2; least-squares solution ~ [1, 2].
    BuiltinResidual::Linear {
        a: vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, -1.0],
        b: vec![1.0, 2.0, 3.0, 0.0],
        m: 4,
        n: 2,
    }
}

fn polynomial_problem() -> BuiltinResidual {
    let t: Vec<f64> = (0..8).map(|i| -1.0 + 0.3 * i as f64).collect();
    // truth coeffs [0.5, -1.0, 2.0] (degree 2); sample with a mild perturbation.
    let y: Vec<f64> = t
        .iter()
        .map(|&ti| 0.5 - 1.0 * ti + 2.0 * ti * ti + 0.01 * ti)
        .collect();
    BuiltinResidual::Polynomial { degree: 2, t, y }
}

fn exponential_problem() -> BuiltinResidual {
    let t: Vec<f64> = (0..10).map(|i| 0.2 * i as f64).collect();
    let y: Vec<f64> = t.iter().map(|&ti| 2.0 * (-0.5 * ti).exp() + 0.3).collect();
    BuiltinResidual::Exponential { t, y }
}

fn all_kinds() -> Vec<(&'static str, BuiltinResidual, Vec<f64>)> {
    vec![
        ("linear", linear_problem(), vec![0.0, 0.0]),
        ("polynomial", polynomial_problem(), vec![0.0, 0.0, 0.0]),
        ("exponential", exponential_problem(), vec![1.0, -0.1, 0.0]),
    ]
}

const LOSSES: [Loss; 5] = [
    Loss::Linear,
    Loss::SoftL1,
    Loss::Huber,
    Loss::Cauchy,
    Loss::Arctan,
];

#[test]
fn native_solve_converges_for_each_kind_and_loss() {
    for (name, kind, x0) in all_kinds() {
        for loss in LOSSES {
            let problem = DataProblem {
                loss,
                f_scale: 1.0,
                ..DataProblem::new(kind.clone(), x0.clone())
            };
            let result = solve_data_problem(&problem)
                .unwrap_or_else(|e| panic!("{name}/{loss:?} solve failed: {e:?}"));
            assert!(
                result.success(),
                "{name}/{loss:?} did not converge (status {})",
                result.status
            );
            assert!(
                result.x.iter().all(|v| v.is_finite()),
                "{name}/{loss:?} produced non-finite x"
            );
        }
    }
}

/// The trait/model path through `solve_model` must be byte-identical to wiring
/// the same residual up as raw closures through `trf_no_bounds` with the same
/// SVD backend and options.
#[test]
fn trait_path_matches_closure_path_byte_for_byte() {
    for (name, kind, x0) in all_kinds() {
        for loss in LOSSES {
            let options = TrfOptions {
                loss,
                f_scale: 1.0,
                ..TrfOptions::default()
            };

            let trait_result = solve_model(&kind, &x0, &options).expect("trait solve");

            let mut fun = |x: &[f64], out: &mut Vec<f64>| kind.eval_residual(x, out);
            let mut jac = |x: &[f64], f0: &[f64], out: &mut Vec<f64>| {
                let mut scratch = Vec::new();
                let mut inner = |xx: &[f64], o: &mut Vec<f64>| kind.eval_residual(xx, o);
                jacobian_2point(&mut inner as &mut ResidualFn<'_>, x, f0, out, &mut scratch)
                    .expect("2-point jacobian");
            };
            let closure_result = trf_no_bounds(
                &mut fun as &mut ResidualFn<'_>,
                &mut jac as &mut JacobianFn<'_>,
                &x0,
                &NalgebraThinSvd,
                &options,
            )
            .expect("closure solve");

            assert_results_byte_equal(&format!("{name}/{loss:?}"), &trait_result, &closure_result);
        }
    }
}

/// The analytic Jacobian must agree with the 2-point default (within finite
/// difference tolerance) at the starting point and at a perturbed point.
#[test]
fn analytic_jacobian_agrees_with_2point_default() {
    for (name, kind, x0) in all_kinds() {
        let probe_points = [x0.clone(), x0.iter().map(|v| v + 0.37).collect::<Vec<_>>()];
        for x in probe_points {
            let mut f0 = Vec::new();
            kind.residual(&x, &mut f0);

            let mut two_point = Vec::new();
            // The default ResidualModel::jacobian is the 2-point finite difference.
            ResidualModel::jacobian(&kind, &x, &f0, &mut two_point);

            let mut analytic = Vec::new();
            kind.eval_jacobian_analytic(&x, &mut analytic);

            assert_eq!(analytic.len(), two_point.len(), "{name} jacobian length");
            for (idx, (a, fd)) in analytic.iter().zip(&two_point).enumerate() {
                let tol = 1e-6 * (1.0 + a.abs());
                assert!(
                    (a - fd).abs() <= tol,
                    "{name} jac[{idx}] analytic {a} vs 2-point {fd} (tol {tol})"
                );
            }
        }
    }
}

/// Each parallel leave-one-out result must equal an independent serial drop-`i`
/// solve bit-for-bit, and the base must equal the full solve.
#[test]
fn drop_one_matches_serial_drop_i_byte_for_byte() {
    for (name, kind, x0) in all_kinds() {
        let options = TrfOptions::default();
        let report = solve_drop_one(&kind, &x0, &options).expect("drop-one");

        let base = solve_model(&kind, &x0, &options).expect("base solve");
        assert_results_byte_equal(&format!("{name} base"), &report.base, &base);

        let m = base.fun.len();
        assert_eq!(report.drops.len(), m, "{name} drop count");
        for i in 0..m {
            let masked = MaskRow {
                inner: &kind,
                drop: i,
            };
            let serial = solve_model(&masked, &x0, &options).expect("serial drop");
            assert_results_byte_equal(&format!("{name} drop {i}"), &report.drops[i], &serial);
            let delta = serial.cost - base.cost;
            assert_eq!(
                report.cost_delta[i].to_bits(),
                delta.to_bits(),
                "{name} cost_delta {i}"
            );
        }
    }
}

/// Each parallel multi-start run must equal an independent serial solve from the
/// same start bit-for-bit.
#[test]
fn perturbed_matches_serial_byte_for_byte() {
    for (name, kind, x0) in all_kinds() {
        let options = TrfOptions::default();
        let starts: Vec<Vec<f64>> = (0..4)
            .map(|k| x0.iter().map(|v| v + 0.1 * k as f64).collect())
            .collect();

        let report = solve_perturbed(&kind, &x0, &starts, &options).expect("perturbed");
        assert_eq!(report.runs.len(), starts.len(), "{name} run count");
        for (i, start) in starts.iter().enumerate() {
            let serial = solve_model(&kind, start, &options).expect("serial start");
            assert_results_byte_equal(&format!("{name} start {i}"), &report.runs[i], &serial);
        }
    }
}

/// The serial leave-one-out twin must produce a `DropOneReport` that is
/// byte-identical to the rayon-parallel one on the same `DataProblem`: base,
/// every drop's `x`/`cost`, and every `cost_delta`. This is what lets the wasm /
/// single-threaded binding delegate to the serial entry point instead of
/// re-implementing the loop.
#[test]
fn data_problem_drop_one_serial_matches_parallel_byte_for_byte() {
    for (name, kind, x0) in all_kinds() {
        let problem = DataProblem::new(kind, x0);

        let parallel = solve_data_problem_drop_one(&problem).expect("parallel drop-one");
        let serial = solve_data_problem_drop_one_serial(&problem).expect("serial drop-one");

        assert_results_byte_equal(&format!("{name} base"), &serial.base, &parallel.base);

        assert_eq!(
            serial.drops.len(),
            parallel.drops.len(),
            "{name} drop count"
        );
        for i in 0..parallel.drops.len() {
            assert_results_byte_equal(
                &format!("{name} drop {i}"),
                &serial.drops[i],
                &parallel.drops[i],
            );
            assert_eq!(
                serial.cost_delta[i].to_bits(),
                parallel.cost_delta[i].to_bits(),
                "{name} cost_delta {i}"
            );
        }
    }
}

/// A model that masks one residual row of an inner model (serial reference for
/// the leave-one-out parallel path).
struct MaskRow<'a> {
    inner: &'a BuiltinResidual,
    drop: usize,
}

impl ResidualModel for MaskRow<'_> {
    fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
        let mut full = Vec::new();
        self.inner.residual(x, &mut full);
        out.clear();
        for (i, value) in full.into_iter().enumerate() {
            if i != self.drop {
                out.push(value);
            }
        }
    }
}

fn assert_results_byte_equal(context: &str, a: &TrfResult, b: &TrfResult) {
    assert_eq!(a.status, b.status, "{context} status");
    assert_eq!(a.nfev, b.nfev, "{context} nfev");
    assert_eq!(a.njev, b.njev, "{context} njev");
    assert_f64_slice_bits_eq(&format!("{context} x"), &a.x, &b.x);
    assert_f64_slice_bits_eq(&format!("{context} fun"), &a.fun, &b.fun);
    assert_f64_slice_bits_eq(&format!("{context} jac"), &a.jac, &b.jac);
    assert_f64_slice_bits_eq(&format!("{context} grad"), &a.grad, &b.grad);
    assert_eq!(a.cost.to_bits(), b.cost.to_bits(), "{context} cost");
    assert_eq!(
        a.optimality.to_bits(),
        b.optimality.to_bits(),
        "{context} optimality"
    );
}

#[test]
fn polynomial_degree_overflow_is_a_typed_error_not_a_panic() {
    // A degree of usize::MAX makes the coefficient count degree+1 overflow usize.
    // Validation must surface that as a typed TrfError::DegreeOverflow before any
    // infallible dims()/allocation can wrap or panic.
    let problem = BuiltinResidual::Polynomial {
        degree: usize::MAX,
        t: vec![0.0, 1.0, 2.0],
        y: vec![0.0, 1.0, 4.0],
    };
    assert_eq!(
        problem.validate(&[0.0, 0.0]),
        Err(TrfError::DegreeOverflow { degree: usize::MAX })
    );

    let data_problem = DataProblem::new(problem, vec![0.0, 0.0]);
    assert_eq!(
        solve_data_problem(&data_problem),
        Err(TrfError::DegreeOverflow { degree: usize::MAX })
    );

    // dims() itself stays infallible and panic-free (saturating), so a direct
    // call on the degenerate degree does not unwind.
    let saturated = BuiltinResidual::Polynomial {
        degree: usize::MAX,
        t: vec![0.0, 1.0, 2.0],
        y: vec![0.0, 1.0, 4.0],
    };
    assert_eq!(saturated.dims(), (3, usize::MAX));
}
