//! A Rust-side residual model, so the residual, its Jacobian, and any
//! per-evaluation corrections live in Rust instead of behind a closure boundary
//! supplied by the caller.
//!
//! [`ResidualModel`] is the trait a problem implements: it owns the residual and
//! optionally an analytic Jacobian (the default delegates to the same bit-exact
//! 2-point finite difference [`crate::trf`] uses). [`solve_model`] drives the
//! trust-region iteration through the default in-crate SVD; [`solve_model_with`]
//! takes an explicit [`ThinSvd`] backend (inject [`crate::hostlapack::LapackSvd`]
//! for bit-for-bit SciPy parity).
//!
//! The model is consulted on every iteration without re-entering any host
//! language, which is what lets a thin PyO3/NIF/wasm/C wrapper drive the solver
//! with zero per-iteration callback: wrap the data once as a model (see
//! [`crate::data`]) and the loop stays in Rust.

use crate::trf::{
    jacobian_2point, trf_no_bounds, JacobianFn, NalgebraThinSvd, ResidualFn, ThinSvd, TrfError,
    TrfOptions, TrfResult,
};

/// A residual `r: R^n -> R^m` whose evaluation (and any per-evaluation
/// corrections) lives entirely in Rust.
///
/// Implement [`residual`](ResidualModel::residual); the
/// [`jacobian`](ResidualModel::jacobian) default reproduces the crate's bit-exact
/// 2-point finite difference (the same path SciPy's `jac='2-point'` takes), so a
/// model only overrides it to supply an analytic Jacobian.
pub trait ResidualModel {
    /// Write the residual at `x` into `out` (which the implementation must clear
    /// and then fill with exactly `m` values; `m` is fixed by the first
    /// evaluation).
    fn residual(&self, x: &[f64], out: &mut Vec<f64>);

    /// Write the row-major `m`-by-`n` Jacobian at `x` into `out`, given the
    /// residual `f0 = residual(x)`.
    ///
    /// The default reproduces [`jacobian_2point`] exactly. On the (degenerate)
    /// event that the finite difference cannot be formed (a size overflow, or a
    /// residual whose length changes between evaluations), `out` is cleared so
    /// the solver surfaces a typed [`TrfError::InvalidJacobianLength`] rather
    /// than iterating on a malformed Jacobian.
    fn jacobian(&self, x: &[f64], f0: &[f64], out: &mut Vec<f64>) {
        let mut scratch = Vec::new();
        let mut fun = |xx: &[f64], o: &mut Vec<f64>| self.residual(xx, o);
        if jacobian_2point(&mut fun as &mut ResidualFn<'_>, x, f0, out, &mut scratch).is_err() {
            out.clear();
        }
    }
}

/// Solve `model` from `x0` through an injected [`ThinSvd`] backend. Inject
/// [`crate::hostlapack::LapackSvd`] for bit-for-bit SciPy parity, or any other
/// [`ThinSvd`] implementation.
///
/// This is the seam-explicit form of [`solve_model`]; it adapts the model to the
/// closure shims [`trf_no_bounds`] expects, leaving that engine untouched.
pub fn solve_model_with<M: ResidualModel + ?Sized>(
    model: &M,
    x0: &[f64],
    svd: &dyn ThinSvd,
    options: &TrfOptions,
) -> Result<TrfResult, TrfError> {
    let mut fun = |x: &[f64], out: &mut Vec<f64>| model.residual(x, out);
    let mut jac = |x: &[f64], f0: &[f64], out: &mut Vec<f64>| model.jacobian(x, f0, out);
    trf_no_bounds(
        &mut fun as &mut ResidualFn<'_>,
        &mut jac as &mut JacobianFn<'_>,
        x0,
        svd,
        options,
    )
}

/// Solve `model` from `x0` using the default in-crate [`NalgebraThinSvd`]
/// backend, so callers never have to hand-wire an SVD seam.
///
/// This is the native pure-Rust path (not bit-identical to SciPy); use
/// [`solve_model_with`] with the [`crate::hostlapack`] backend for bit-for-bit
/// parity.
pub fn solve_model<M: ResidualModel + ?Sized>(
    model: &M,
    x0: &[f64],
    options: &TrfOptions,
) -> Result<TrfResult, TrfError> {
    solve_model_with(model, x0, &NalgebraThinSvd, options)
}
