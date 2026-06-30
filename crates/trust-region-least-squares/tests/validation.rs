//! Input-validation coverage for the public API. These assert that malformed
//! input produces a typed [`TrfError`] instead of a panic, an out-of-bounds
//! index, or a silent NaN/Inf result. The native solves here use the pure-Rust
//! `nalgebra` SVD seam (see `tests/support/nalgebra_svd.rs`); they are not
//! bit-exact checks.

#[path = "support/nalgebra_svd.rs"]
mod nalgebra_svd;

use nalgebra_svd::NalgebraSvd;
use trust_region_least_squares::loss::{scale_for_robust_loss, Loss, LossError, Rho};
use trust_region_least_squares::trf::{
    compute_grad, compute_jac_scale, evaluate_quadratic, jacobian_2point, solve_lsq_trust_region,
    trf_no_bounds, JacobianFn, ResidualFn, SvdError, ThinSvd, TrfError, TrfOptions, XScale,
};

/// A small well-posed `m=3`, `n=2` linear residual, used as the happy-path
/// substrate that several validation cases perturb.
fn linear_residual(x: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.push(x[0] - 1.0);
    out.push(x[1] - 2.0);
    out.push(x[0] + x[1] - 3.0);
}

fn linear_jac(x: &[f64], f0: &[f64], out: &mut Vec<f64>) {
    let mut scratch = Vec::new();
    jacobian_2point(&mut linear_residual, x, f0, out, &mut scratch).expect("jacobian");
}

fn solve_with(
    fun: &mut ResidualFn<'_>,
    jac: &mut JacobianFn<'_>,
    x0: &[f64],
    options: &TrfOptions,
) -> Result<trust_region_least_squares::trf::TrfResult, TrfError> {
    trf_no_bounds(fun, jac, x0, &NalgebraSvd, options)
}

#[test]
fn happy_path_native_solve_converges() {
    let mut fun = linear_residual;
    let mut jac = linear_jac;
    let result =
        solve_with(&mut fun, &mut jac, &[0.0, 0.0], &TrfOptions::default()).expect("native solve");
    assert!(result.success(), "status was {}", result.status);
    // x -> [1, 2] for this consistent system.
    assert!((result.x[0] - 1.0).abs() < 1e-6, "x0 = {}", result.x[0]);
    assert!((result.x[1] - 2.0).abs() < 1e-6, "x1 = {}", result.x[1]);
}

#[test]
fn empty_parameters_error() {
    let mut fun = linear_residual;
    let mut jac = linear_jac;
    let err = solve_with(&mut fun, &mut jac, &[], &TrfOptions::default()).unwrap_err();
    assert!(matches!(err, TrfError::EmptyParameters), "{err:?}");
}

#[test]
fn non_finite_parameters_error() {
    let mut fun = linear_residual;
    let mut jac = linear_jac;
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let err = solve_with(&mut fun, &mut jac, &[bad, 0.0], &TrfOptions::default()).unwrap_err();
        assert!(matches!(err, TrfError::NonFiniteParameters), "{err:?}");
    }
}

#[test]
fn empty_residual_error() {
    let mut fun = |_x: &[f64], out: &mut Vec<f64>| out.clear();
    let mut jac = linear_jac;
    let err = solve_with(&mut fun, &mut jac, &[0.0, 0.0], &TrfOptions::default()).unwrap_err();
    assert!(matches!(err, TrfError::EmptyResidual), "{err:?}");
}

#[test]
fn non_finite_initial_residual_error() {
    let mut fun = |_x: &[f64], out: &mut Vec<f64>| {
        out.clear();
        out.extend_from_slice(&[0.0, f64::NAN, 0.0]);
    };
    let mut jac = linear_jac;
    let err = solve_with(&mut fun, &mut jac, &[0.0, 0.0], &TrfOptions::default()).unwrap_err();
    assert!(matches!(err, TrfError::NonFiniteInitialResidual), "{err:?}");
}

#[test]
fn insufficient_rows_error() {
    // m = 2 residuals, n = 3 parameters.
    let mut fun = |x: &[f64], out: &mut Vec<f64>| {
        out.clear();
        out.push(x[0] - 1.0);
        out.push(x[1] + x[2]);
    };
    let mut jac = |_x: &[f64], _f0: &[f64], out: &mut Vec<f64>| out.clear();
    let err = solve_with(&mut fun, &mut jac, &[0.0, 0.0, 0.0], &TrfOptions::default()).unwrap_err();
    assert!(
        matches!(err, TrfError::InsufficientRows { m: 2, n: 3 }),
        "{err:?}"
    );
}

#[test]
fn invalid_f_scale_for_robust_loss_error() {
    let mut jac = linear_jac;
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        let mut fun = linear_residual;
        let options = TrfOptions {
            loss: Loss::SoftL1,
            f_scale: bad,
            ..TrfOptions::default()
        };
        let err = solve_with(&mut fun, &mut jac, &[0.0, 0.0], &options).unwrap_err();
        assert!(
            matches!(err, TrfError::InvalidFScale { f_scale } if f_scale.to_bits() == bad.to_bits()),
            "expected InvalidFScale for {bad:?}, got {err:?}"
        );
    }
}

#[test]
fn zero_f_scale_with_linear_loss_is_accepted() {
    // SciPy ignores `f_scale` for the linear loss; so do we (no error).
    let mut fun = linear_residual;
    let mut jac = linear_jac;
    let options = TrfOptions {
        loss: Loss::Linear,
        f_scale: 0.0,
        ..TrfOptions::default()
    };
    let result = solve_with(&mut fun, &mut jac, &[0.0, 0.0], &options).expect("solve");
    assert!(result.success());
}

#[test]
fn invalid_jacobian_length_error() {
    let mut fun = linear_residual;
    // n=2, m=3 -> expected 6; return 5.
    let mut jac = |_x: &[f64], _f0: &[f64], out: &mut Vec<f64>| {
        out.clear();
        out.extend_from_slice(&[0.0; 5]);
    };
    let err = solve_with(&mut fun, &mut jac, &[0.0, 0.0], &TrfOptions::default()).unwrap_err();
    assert!(
        matches!(
            err,
            TrfError::InvalidJacobianLength {
                expected: 6,
                got: 5
            }
        ),
        "{err:?}"
    );
}

#[test]
fn trial_residual_length_change_error() {
    // Correct length on the initial evaluation, wrong length on the trial step.
    let mut fun_calls = 0usize;
    let mut fun = move |x: &[f64], out: &mut Vec<f64>| {
        fun_calls += 1;
        out.clear();
        if fun_calls >= 2 {
            out.extend_from_slice(&[0.0, 0.0]); // length 2, not 3
        } else {
            out.push(x[0] - 1.0);
            out.push(x[1] - 2.0);
            out.push(x[0] + x[1] - 3.0);
        }
    };
    // Independent, always-correct Jacobian so the failure is isolated to the
    // trust-region trial evaluation, not the finite-difference step.
    let mut jac = linear_jac;
    let err = solve_with(&mut fun, &mut jac, &[5.0, -4.0], &TrfOptions::default()).unwrap_err();
    assert!(
        matches!(
            err,
            TrfError::InvalidResidualLength {
                expected: 3,
                got: 2
            }
        ),
        "{err:?}"
    );
}

#[test]
fn x_scale_wrong_length_error() {
    let mut fun = linear_residual;
    let mut jac = linear_jac;
    let options = TrfOptions {
        x_scale: XScale::Values(vec![1.0]),
        ..TrfOptions::default()
    };
    let err = solve_with(&mut fun, &mut jac, &[0.0, 0.0], &options).unwrap_err();
    assert!(
        matches!(
            err,
            TrfError::InvalidXScaleLength {
                expected: 2,
                got: 1
            }
        ),
        "{err:?}"
    );
}

#[test]
fn x_scale_non_positive_value_error() {
    let mut fun = linear_residual;
    let mut jac = linear_jac;
    for (bad, idx) in [(0.0, 1usize), (-2.0, 1), (f64::NAN, 1)] {
        let options = TrfOptions {
            x_scale: XScale::Values(vec![1.0, bad]),
            ..TrfOptions::default()
        };
        let err = solve_with(&mut fun, &mut jac, &[0.0, 0.0], &options).unwrap_err();
        assert!(
            matches!(err, TrfError::InvalidXScaleValue { index, value }
                if index == idx && value.to_bits() == bad.to_bits()),
            "expected InvalidXScaleValue for {bad:?}, got {err:?}"
        );
    }
}

// --- Public helper validation -------------------------------------------------

#[test]
fn compute_grad_validates_lengths() {
    // jac too short: expected m*n = 6.
    assert!(matches!(
        compute_grad(&[0.0; 5], &[0.0; 3], 3, 2),
        Err(TrfError::InvalidSliceLength {
            what: "jac",
            expected: 6,
            got: 5
        })
    ));
    // f wrong length: expected m = 3.
    assert!(matches!(
        compute_grad(&[0.0; 6], &[0.0; 2], 3, 2),
        Err(TrfError::InvalidSliceLength {
            what: "f",
            expected: 3,
            got: 2
        })
    ));
    // valid.
    assert!(compute_grad(&[0.0; 6], &[0.0; 3], 3, 2).is_ok());
}

#[test]
fn compute_grad_overflow_error() {
    assert!(matches!(
        compute_grad(&[], &[], usize::MAX, 2),
        Err(TrfError::SizeOverflow { .. })
    ));
}

#[test]
fn compute_jac_scale_validates_lengths() {
    assert!(matches!(
        compute_jac_scale(&[0.0; 5], 3, 2, None),
        Err(TrfError::InvalidSliceLength { what: "jac", .. })
    ));
    assert!(matches!(
        compute_jac_scale(&[0.0; 6], 3, 2, Some(&[1.0])),
        Err(TrfError::InvalidSliceLength {
            what: "scale_inv_old",
            expected: 2,
            got: 1
        })
    ));
    assert!(compute_jac_scale(&[1.0; 6], 3, 2, None).is_ok());
}

#[test]
fn evaluate_quadratic_validates_lengths() {
    assert!(matches!(
        evaluate_quadratic(&[0.0; 5], &[0.0; 2], &[0.0; 2], 3, 2),
        Err(TrfError::InvalidSliceLength { what: "jac", .. })
    ));
    assert!(matches!(
        evaluate_quadratic(&[0.0; 6], &[0.0; 1], &[0.0; 2], 3, 2),
        Err(TrfError::InvalidSliceLength { what: "g", .. })
    ));
    assert!(matches!(
        evaluate_quadratic(&[0.0; 6], &[0.0; 2], &[0.0; 1], 3, 2),
        Err(TrfError::InvalidSliceLength { what: "step", .. })
    ));
    assert!(evaluate_quadratic(&[0.0; 6], &[0.0; 2], &[0.0; 2], 3, 2).is_ok());
}

#[test]
fn solve_lsq_trust_region_validates_inputs() {
    assert!(matches!(
        solve_lsq_trust_region(0, 0, &[], &[], &[], 1.0, 0.0),
        Err(TrfError::EmptyParameters)
    ));
    // n = 2 requires uf.len()=2, s.len()=2, vt.len()=4.
    assert!(matches!(
        solve_lsq_trust_region(2, 3, &[1.0], &[1.0, 1.0], &[1.0; 4], 1.0, 0.0),
        Err(TrfError::InvalidSliceLength { what: "uf", .. })
    ));
    assert!(matches!(
        solve_lsq_trust_region(2, 3, &[1.0, 1.0], &[1.0, 1.0], &[1.0; 3], 1.0, 0.0),
        Err(TrfError::InvalidSliceLength { what: "vt", .. })
    ));
    // Well-formed (identity-ish) inputs solve without panicking.
    assert!(solve_lsq_trust_region(
        2,
        2,
        &[1.0, 1.0],
        &[2.0, 1.0],
        &[1.0, 0.0, 0.0, 1.0],
        10.0,
        0.0
    )
    .is_ok());
}

#[test]
fn jacobian_2point_validates_residual_length() {
    let x0 = [0.0, 0.0];
    let f0 = [0.0, 0.0, 0.0];
    let mut jac = Vec::new();
    let mut scratch = Vec::new();
    // Closure writes 2 values; f0 says m = 3.
    let mut bad = |_x: &[f64], out: &mut Vec<f64>| {
        out.clear();
        out.extend_from_slice(&[1.0, 2.0]);
    };
    let err = jacobian_2point(&mut bad, &x0, &f0, &mut jac, &mut scratch).unwrap_err();
    assert!(
        matches!(
            err,
            TrfError::InvalidResidualLength {
                expected: 3,
                got: 2
            }
        ),
        "{err:?}"
    );
}

#[test]
fn zero_max_nfev_is_rejected() {
    let mut fun = linear_residual;
    let mut jac = linear_jac;
    let options = TrfOptions {
        max_nfev: Some(0),
        ..TrfOptions::default()
    };
    let err = solve_with(&mut fun, &mut jac, &[0.0, 0.0], &options).unwrap_err();
    assert!(matches!(err, TrfError::InvalidMaxNfev), "{err:?}");
}

/// A backend that delegates the SVD to `nalgebra` but returns a wrong-length
/// `power3`, to prove the solver validates that injected output rather than
/// indexing out of bounds.
struct BadPower3;
impl ThinSvd for BadPower3 {
    fn svd(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError> {
        NalgebraSvd.svd(a, m, n)
    }
    fn power3(&self, x: &[f64]) -> Result<Option<Vec<f64>>, SvdError> {
        Ok(Some(vec![0.0; x.len() + 1]))
    }
}

#[test]
fn bad_power3_output_length_is_rejected() {
    // x0 far from the solution [1, 2] forces a step past the trust radius, so the
    // alpha search runs and `power3` is consulted.
    let mut fun = linear_residual;
    let mut jac = linear_jac;
    let err = trf_no_bounds(
        &mut fun as &mut ResidualFn<'_>,
        &mut jac as &mut JacobianFn<'_>,
        &[0.0, 0.0],
        &BadPower3,
        &TrfOptions::default(),
    )
    .unwrap_err();
    assert!(matches!(err, TrfError::InvalidSvdOutput(_)), "{err:?}");
}

#[test]
fn scale_for_robust_loss_validates_lengths() {
    // m = f.len() = 2, n = 2 -> jac must be length 4 and rho1/rho2 length 2.
    let rho_ok = Rho {
        rho0: vec![0.0; 2],
        rho1: vec![1.0; 2],
        rho2: vec![0.0; 2],
    };
    // rho1 too short.
    let mut jac = vec![1.0; 4];
    let mut f = vec![1.0; 2];
    let rho_bad = Rho {
        rho0: vec![0.0; 2],
        rho1: vec![1.0; 1],
        rho2: vec![0.0; 2],
    };
    assert!(matches!(
        scale_for_robust_loss(&mut jac, &mut f, &rho_bad, 2),
        Err(LossError::LengthMismatch { what: "rho1", .. })
    ));
    // jac wrong length.
    let mut jac = vec![1.0; 3];
    let mut f = vec![1.0; 2];
    assert!(matches!(
        scale_for_robust_loss(&mut jac, &mut f, &rho_ok, 2),
        Err(LossError::LengthMismatch {
            what: "jac",
            expected: 4,
            got: 3
        })
    ));
    // valid.
    let mut jac = vec![1.0; 4];
    let mut f = vec![1.0; 2];
    assert!(scale_for_robust_loss(&mut jac, &mut f, &rho_ok, 2).is_ok());
}

#[test]
fn error_display_is_nonempty() {
    // Smoke-test the Display impl so the error type is usable with `?`/anyhow.
    let err = TrfError::InsufficientRows { m: 1, n: 2 };
    let text = format!("{err}");
    assert!(text.contains("m < n"), "{text}");
    let _: &dyn std::error::Error = &err;
}
