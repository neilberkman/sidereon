//! Coverage for the injectable host-numerics power seam.
//!
//! `ThinSvd::power` (elementwise `values ** exponent`) and
//! `ThinSvd::power_scalar` (scalar `base ** exponent`) let a host backend supply
//! the exact results an external numerical runtime produces for the two places
//! SciPy raises to a non-fast-path exponent: the robust-loss derivative powers
//! (`z ** -0.5`, `z ** -1.5`) and the trust-region alpha seed
//! (`(alpha_lower * alpha_upper) ** 0.5`).
//!
//! These are behavioral tests of the seam, not bit-exact replays: every backend
//! here either declines (`Ok(None)`, so the crate keeps its own Rust arithmetic)
//! or violates the contract on purpose.

use std::cell::RefCell;

use trust_region_least_squares::loss::{Loss, LossError, LossFunction};
use trust_region_least_squares::model::{solve_model_with, ResidualModel};
use trust_region_least_squares::trf::{
    NalgebraThinSvd, SvdError, ThinSvd, TrfError, TrfOptions, TrfResult,
};

/// A backend written before the power hooks existed: it implements only the one
/// required `ThinSvd` method. It must keep compiling and keep producing the
/// crate's existing results through the trait defaults.
struct SvdOnlyBackend;

impl ThinSvd for SvdOnlyBackend {
    fn svd(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError> {
        NalgebraThinSvd.svd(a, m, n)
    }
}

/// Records every power dispatch and then declines it, so the solve it observes
/// is bit-identical to the one the default path produces.
#[derive(Default)]
struct RecordingBackend {
    vector_calls: RefCell<Vec<(Vec<f64>, f64)>>,
    scalar_calls: RefCell<Vec<(f64, f64)>>,
}

impl RecordingBackend {
    fn vector_calls(&self) -> Vec<(Vec<f64>, f64)> {
        self.vector_calls.borrow().clone()
    }

    fn scalar_calls(&self) -> Vec<(f64, f64)> {
        self.scalar_calls.borrow().clone()
    }
}

impl ThinSvd for RecordingBackend {
    fn svd(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError> {
        NalgebraThinSvd.svd(a, m, n)
    }

    fn power(&self, values: &[f64], exponent: f64) -> Result<Option<Vec<f64>>, SvdError> {
        self.vector_calls
            .borrow_mut()
            .push((values.to_vec(), exponent));
        Ok(None)
    }

    fn power_scalar(&self, base: f64, exponent: f64) -> Result<Option<f64>, SvdError> {
        self.scalar_calls.borrow_mut().push((base, exponent));
        Ok(None)
    }
}

/// Violates the vector-power contract by returning one value too many.
struct TooLongPowerBackend;

impl ThinSvd for TooLongPowerBackend {
    fn svd(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError> {
        NalgebraThinSvd.svd(a, m, n)
    }

    fn power(&self, values: &[f64], _exponent: f64) -> Result<Option<Vec<f64>>, SvdError> {
        Ok(Some(vec![1.0; values.len() + 1]))
    }
}

/// A well-conditioned overdetermined residual, `m = 5`, `n = 2`, with outliers
/// large enough that huber's `z > 1` branch (and therefore the derivative
/// powers) is exercised.
struct OutlierModel;

impl ResidualModel for OutlierModel {
    fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.push(x[0] - 1.0);
        out.push(x[1] - 2.0);
        out.push(x[0] + x[1] - 3.0);
        out.push(x[0] - 9.0);
        out.push(x[1] + 8.0);
    }
}

/// A rank-deficient residual: `x1` never appears, so the Jacobian has a zero
/// column, the thin SVD is rank deficient, and `solve_lsq_trust_region` takes
/// its alpha-seeding branch instead of the Gauss-Newton shortcut.
struct RankDeficientModel;

impl ResidualModel for RankDeficientModel {
    fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        out.push(x[0] - 1.0);
        out.push(x[0] - 2.0);
        out.push(x[0] + 1.0);
    }
}

fn robust_options(loss: Loss) -> TrfOptions {
    TrfOptions {
        loss,
        f_scale: 1.0,
        ..TrfOptions::default()
    }
}

fn solve<M: ResidualModel>(
    model: &M,
    x0: &[f64],
    svd: &dyn ThinSvd,
    options: &TrfOptions,
) -> Result<TrfResult, TrfError> {
    solve_model_with(model, x0, svd, options)
}

#[test]
fn backend_without_power_hooks_compiles_and_matches_the_default_backend() {
    let options = robust_options(Loss::Huber);
    let legacy =
        solve(&OutlierModel, &[0.0, 0.0], &SvdOnlyBackend, &options).expect("legacy solve");
    let default =
        solve(&OutlierModel, &[0.0, 0.0], &NalgebraThinSvd, &options).expect("default solve");
    assert_eq!(legacy, default, "trait defaults changed the solve result");
}

#[test]
fn declining_backend_matches_the_default_backend_bit_for_bit() {
    for loss in [
        Loss::Linear,
        Loss::SoftL1,
        Loss::Huber,
        Loss::Cauchy,
        Loss::Arctan,
    ] {
        let options = robust_options(loss);
        let recorder = RecordingBackend::default();
        let recorded = solve(&OutlierModel, &[0.0, 0.0], &recorder, &options).expect("solve");
        let default = solve(&OutlierModel, &[0.0, 0.0], &NalgebraThinSvd, &options).expect("solve");
        assert_eq!(recorded, default, "{loss:?} diverged from the default path");
    }
}

#[test]
fn huber_routes_outlier_derivative_powers_through_the_vector_hook() {
    let recorder = RecordingBackend::default();
    let loss = LossFunction::new(Loss::Huber, 1.0);
    // z = f**2 = [0.25, 4.0, 9.0, 0.0625]; the mask `z <= 1` keeps entries 0
    // and 3 on the quadratic branch, so only [4.0, 9.0] is raised to a power.
    let f = [0.5, 2.0, -3.0, 0.25];
    let rho = loss.evaluate_with(&f, &recorder).expect("rho");

    let calls = recorder.vector_calls();
    assert_eq!(calls.len(), 2, "expected one call per derivative power");
    assert_eq!(calls[0], (vec![4.0, 9.0], -0.5));
    assert_eq!(calls[1], (vec![4.0, 9.0], -1.5));
    assert_eq!(
        rho,
        loss.evaluate(&f),
        "declining must not change the result"
    );
}

#[test]
fn soft_l1_routes_derivative_powers_over_the_full_shifted_vector() {
    let recorder = RecordingBackend::default();
    let loss = LossFunction::new(Loss::SoftL1, 1.0);
    // t = 1 + z = 1 + f**2, applied to every entry (soft_l1 has no mask).
    let f = [0.5, 2.0, -3.0];
    let rho = loss.evaluate_with(&f, &recorder).expect("rho");

    let calls = recorder.vector_calls();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0], (vec![1.25, 5.0, 10.0], -0.5));
    assert_eq!(calls[1], (vec![1.25, 5.0, 10.0], -1.5));
    assert_eq!(rho, loss.evaluate(&f));
}

#[test]
fn losses_without_a_power_expression_do_not_dispatch_to_the_hook() {
    // cauchy and arctan express their derivatives as reciprocals of products,
    // and huber's `z <= 1` branch is a plain copy: no `**` reaches the hook.
    for loss in [Loss::Cauchy, Loss::Arctan, Loss::Linear] {
        let recorder = RecordingBackend::default();
        let function = LossFunction::new(loss, 1.0);
        function
            .evaluate_with(&[0.5, 2.0, -3.0], &recorder)
            .expect("rho");
        assert!(
            recorder.vector_calls().is_empty(),
            "{loss:?} dispatched an unexpected power"
        );
    }

    let recorder = RecordingBackend::default();
    LossFunction::new(Loss::Huber, 1.0)
        .evaluate_with(&[0.5, 0.25], &recorder)
        .expect("rho");
    assert!(
        recorder.vector_calls().is_empty(),
        "huber dispatched a power with every residual inside the mask"
    );
}

#[test]
fn robust_solve_dispatches_the_vector_power_hook() {
    for (loss, expected_exponents) in [
        (Loss::Huber, vec![-1.5, -0.5]),
        (Loss::SoftL1, vec![-1.5, -0.5]),
    ] {
        let recorder = RecordingBackend::default();
        solve(&OutlierModel, &[0.0, 0.0], &recorder, &robust_options(loss)).expect("solve");
        let calls = recorder.vector_calls();
        assert!(!calls.is_empty(), "{loss:?} never dispatched a power");
        let mut exponents: Vec<f64> = calls.iter().map(|(_, e)| *e).collect();
        exponents.sort_by(|a, b| a.partial_cmp(b).expect("finite exponents"));
        exponents.dedup();
        assert_eq!(exponents, expected_exponents, "{loss:?} exponents");
        assert!(
            calls.iter().all(|(values, _)| !values.is_empty()),
            "{loss:?} dispatched an empty power"
        );
    }
}

#[test]
fn alpha_seed_dispatches_the_scalar_power_hook() {
    let recorder = RecordingBackend::default();
    solve(
        &RankDeficientModel,
        &[0.0, 0.0],
        &recorder,
        &TrfOptions::default(),
    )
    .expect("solve");

    let calls = recorder.scalar_calls();
    assert!(
        !calls.is_empty(),
        "the rank-deficient alpha seed never dispatched a scalar power"
    );
    assert!(
        calls.iter().all(|(_, exponent)| *exponent == 0.5),
        "alpha seed dispatched an unexpected exponent: {calls:?}"
    );
}

#[test]
fn vector_power_hook_length_mismatch_is_reported_by_the_loss_api() {
    let err = LossFunction::new(Loss::SoftL1, 1.0)
        .evaluate_with(&[0.5, 2.0], &TooLongPowerBackend)
        .unwrap_err();
    assert_eq!(
        err,
        LossError::HostPowerLength {
            expected: 2,
            got: 3
        },
        "{err}"
    );
}

#[test]
fn vector_power_hook_length_mismatch_fails_the_solve() {
    let err = solve(
        &OutlierModel,
        &[0.0, 0.0],
        &TooLongPowerBackend,
        &robust_options(Loss::Huber),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            TrfError::InvalidSliceLength {
                what: "host power result",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn host_power_failure_propagates_as_a_typed_error() {
    struct FailingBackend;
    impl ThinSvd for FailingBackend {
        fn svd(
            &self,
            a: &[f64],
            m: usize,
            n: usize,
        ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError> {
            NalgebraThinSvd.svd(a, m, n)
        }

        fn power(&self, _values: &[f64], _exponent: f64) -> Result<Option<Vec<f64>>, SvdError> {
            Err(SvdError::Failed("no host power".to_string()))
        }
    }

    let err = solve(
        &OutlierModel,
        &[0.0, 0.0],
        &FailingBackend,
        &robust_options(Loss::SoftL1),
    )
    .unwrap_err();
    assert!(matches!(err, TrfError::Svd(SvdError::Failed(_))), "{err:?}");
}

#[test]
fn supplied_host_power_results_reach_the_rho_output() {
    struct ConstantPowerBackend;
    impl ThinSvd for ConstantPowerBackend {
        fn svd(
            &self,
            a: &[f64],
            m: usize,
            n: usize,
        ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError> {
            NalgebraThinSvd.svd(a, m, n)
        }

        fn power(&self, values: &[f64], exponent: f64) -> Result<Option<Vec<f64>>, SvdError> {
            // A deliberately wrong but well-formed result, so the test can see
            // that the supplied values (not the crate's own powf) are used.
            Ok(Some(values.iter().map(|_| exponent).collect()))
        }
    }

    let rho = LossFunction::new(Loss::SoftL1, 1.0)
        .evaluate_with(&[0.5, 2.0], &ConstantPowerBackend)
        .expect("rho");
    assert_eq!(rho.rho1, vec![-0.5, -0.5]);
    assert_eq!(rho.rho2, vec![0.75, 0.75]);
}

#[test]
fn scalar_power_failure_propagates_out_of_the_solve() {
    struct FailingScalarBackend;
    impl ThinSvd for FailingScalarBackend {
        fn svd(
            &self,
            a: &[f64],
            m: usize,
            n: usize,
        ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError> {
            NalgebraThinSvd.svd(a, m, n)
        }

        fn power_scalar(&self, _base: f64, _exponent: f64) -> Result<Option<f64>, SvdError> {
            Err(SvdError::Failed("no host scalar power".to_string()))
        }
    }

    let err = solve(
        &RankDeficientModel,
        &[0.0, 0.0],
        &FailingScalarBackend,
        &TrfOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, TrfError::Svd(SvdError::Failed(_))), "{err:?}");
}
