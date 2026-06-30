//! Data-driven built-in residuals: the path a host-language binding drives
//! without ever re-entering the host language inside the solver loop.
//!
//! A binding picks a [`BuiltinResidual`] kind, hands over the data arrays, and
//! [`solve_data_problem`] runs the whole trust-region iteration in Rust. The
//! residual and Jacobian for every iteration are evaluated here, so a thin
//! PyO3/NIF/wasm/C wrapper pays one boundary crossing (in) and one (out), not
//! one per function evaluation.
//!
//! Each kind carries its own data (the design matrix, or the sample points), so
//! a problem is fully self-describing. The residual evaluation is defined to be
//! reproducible scalar-for-scalar against the SciPy fixture generator (sequential
//! accumulation, not a BLAS dot), so the data path is held to the same bit-exact
//! bar as the rest of the crate when driven through the [`crate::hostlapack`]
//! SVD backend.

use crate::loss::Loss;
use crate::model::{solve_model, solve_model_with, ResidualModel};
use crate::trf::{ThinSvd, TrfError, TrfOptions, TrfResult, XScale};

/// A built-in residual `r: R^n -> R^m` evaluated entirely in Rust.
///
/// Every variant owns its data, so the problem is self-describing; the parameter
/// vector `x` (length `n`) is what the solver fits.
#[derive(Debug, Clone, PartialEq)]
pub enum BuiltinResidual {
    /// General dense linear least squares: `residual_i = (sum_j a[i*n+j]*x[j]) -
    /// b[i]`, with `a` the row-major `m`-by-`n` design matrix and `b` the length
    /// `m` right-hand side. The linear combination is a sequential left-to-right
    /// accumulation (not a BLAS dot) so the float-op order is reproducible.
    Linear {
        a: Vec<f64>,
        b: Vec<f64>,
        m: usize,
        n: usize,
    },
    /// Polynomial fit of `degree` (so `n = degree + 1` coefficients,
    /// lowest-order first): `residual_i = horner(x, t_i) - y_i`, evaluated by
    /// Horner's method over the `t`/`y` sample pairs.
    Polynomial {
        degree: usize,
        t: Vec<f64>,
        y: Vec<f64>,
    },
    /// Exponential model with `n = 3` parameters `[amp, rate, offset]`:
    /// `residual_i = (x[0] * exp(x[1] * t_i) + x[2]) - y_i` over the `t`/`y`
    /// sample pairs.
    Exponential { t: Vec<f64>, y: Vec<f64> },
}

impl BuiltinResidual {
    /// The number of residual rows `m` and parameters `n` this problem has.
    ///
    /// Infallible and panic-free: a degenerate polynomial `degree` of
    /// `usize::MAX` saturates the coefficient count rather than overflowing.
    /// [`validate`](BuiltinResidual::validate) is the authoritative gate and
    /// rejects such a degree with [`TrfError::DegreeOverflow`] before any solve;
    /// callers should validate first and then trust `dims`.
    pub fn dims(&self) -> (usize, usize) {
        match self {
            BuiltinResidual::Linear { m, n, .. } => (*m, *n),
            BuiltinResidual::Polynomial { degree, t, .. } => (t.len(), degree.saturating_add(1)),
            BuiltinResidual::Exponential { t, .. } => (t.len(), 3),
        }
    }

    /// The `(m, n)` dimensions with the polynomial coefficient count computed by
    /// checked addition, so an untrusted `degree` near `usize::MAX` surfaces as a
    /// typed [`TrfError::DegreeOverflow`] instead of overflowing.
    fn checked_dims(&self) -> Result<(usize, usize), TrfError> {
        match self {
            BuiltinResidual::Linear { m, n, .. } => Ok((*m, *n)),
            BuiltinResidual::Polynomial { degree, t, .. } => {
                let n = degree
                    .checked_add(1)
                    .ok_or(TrfError::DegreeOverflow { degree: *degree })?;
                Ok((t.len(), n))
            }
            BuiltinResidual::Exponential { t, .. } => Ok((t.len(), 3)),
        }
    }

    /// Validate the stored data against `x0`, returning a typed [`TrfError`] for
    /// any inconsistent shape so a malformed problem cannot index out of bounds
    /// inside the solver loop.
    pub fn validate(&self, x0: &[f64]) -> Result<(), TrfError> {
        let (m, n) = self.checked_dims()?;
        if m == 0 {
            return Err(TrfError::EmptyResidual);
        }
        if n == 0 {
            return Err(TrfError::EmptyParameters);
        }
        match self {
            BuiltinResidual::Linear { a, b, m, n } => {
                let mn = m
                    .checked_mul(*n)
                    .ok_or(TrfError::SizeOverflow { m: *m, n: *n })?;
                check_len("matrix", a.len(), mn)?;
                check_len("rhs", b.len(), *m)?;
            }
            BuiltinResidual::Polynomial { t, y, .. } | BuiltinResidual::Exponential { t, y } => {
                check_len("data y", y.len(), t.len())?;
            }
        }
        check_len("x0", x0.len(), n)?;
        Ok(())
    }

    /// Write the residual at `x` into `out`. Assumes the problem has been
    /// validated (see [`validate`](BuiltinResidual::validate)).
    pub fn eval_residual(&self, x: &[f64], out: &mut Vec<f64>) {
        out.clear();
        match self {
            BuiltinResidual::Linear { a, b, n, .. } => {
                for (i, &bi) in b.iter().enumerate() {
                    let row = i * n;
                    // Sequential left-to-right accumulation (not a BLAS dot) so
                    // the float-op order is reproducible against the generator.
                    let mut acc = 0.0f64;
                    for j in 0..*n {
                        acc += a[row + j] * x[j];
                    }
                    out.push(acc - bi);
                }
            }
            BuiltinResidual::Polynomial { degree, t, y } => {
                for (i, &ti) in t.iter().enumerate() {
                    // Horner from the highest-order coefficient down.
                    let mut acc = x[*degree];
                    for k in (0..*degree).rev() {
                        acc = acc * ti + x[k];
                    }
                    out.push(acc - y[i]);
                }
            }
            BuiltinResidual::Exponential { t, y } => {
                for (i, &ti) in t.iter().enumerate() {
                    out.push((x[0] * (x[1] * ti).exp() + x[2]) - y[i]);
                }
            }
        }
    }

    /// Write the analytic row-major `m`-by-`n` Jacobian at `x` into `out`.
    ///
    /// This is offered alongside the 2-point default so callers (and the
    /// agreement test) can use a closed-form Jacobian; the bit-exact data path
    /// uses the 2-point finite difference to match SciPy's `jac='2-point'`.
    pub fn eval_jacobian_analytic(&self, x: &[f64], out: &mut Vec<f64>) {
        let (m, n) = self.dims();
        out.clear();
        out.resize(m * n, 0.0);
        match self {
            BuiltinResidual::Linear { a, .. } => {
                out.copy_from_slice(a);
            }
            BuiltinResidual::Polynomial { degree, t, .. } => {
                for (i, &ti) in t.iter().enumerate() {
                    let row = i * n;
                    for k in 0..=*degree {
                        out[row + k] = ti.powi(k as i32);
                    }
                }
            }
            BuiltinResidual::Exponential { t, .. } => {
                for (i, &ti) in t.iter().enumerate() {
                    let row = i * 3;
                    let e = (x[1] * ti).exp();
                    out[row] = e;
                    out[row + 1] = x[0] * ti * e;
                    out[row + 2] = 1.0;
                }
            }
        }
    }
}

impl ResidualModel for BuiltinResidual {
    fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
        self.eval_residual(x, out);
    }
    // Jacobian intentionally left at the 2-point default so the data path stays
    // bit-exact with SciPy's `jac='2-point'`; the analytic form is available via
    // `eval_jacobian_analytic`.
}

/// A fully specified data-driven problem: a [`BuiltinResidual`] kind plus the
/// starting point and the solve configuration.
///
/// The fields are flat (no nested options struct) so a host-language binding can
/// fill them field-by-field over an FFI boundary. [`DataProblem::new`] seeds the
/// SciPy `least_squares` defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct DataProblem {
    /// The residual kind, carrying its data.
    pub kind: BuiltinResidual,
    /// Starting parameter vector, length `n`.
    pub x0: Vec<f64>,
    /// SciPy `loss`.
    pub loss: Loss,
    /// SciPy `f_scale` (only consulted for a robust loss).
    pub f_scale: f64,
    /// SciPy `x_scale`.
    pub x_scale: XScale,
    /// SciPy `max_nfev` (`None` selects the default `100 * n`).
    pub max_nfev: Option<usize>,
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
}

impl DataProblem {
    /// Construct a problem with the SciPy `least_squares` defaults (linear loss,
    /// `f_scale = 1`, unit `x_scale`, `ftol = xtol = 1e-8`, `gtol = 1e-10`).
    pub fn new(kind: BuiltinResidual, x0: Vec<f64>) -> Self {
        let defaults = TrfOptions::default();
        Self {
            kind,
            x0,
            loss: defaults.loss,
            f_scale: defaults.f_scale,
            x_scale: defaults.x_scale,
            max_nfev: defaults.max_nfev,
            ftol: defaults.ftol,
            xtol: defaults.xtol,
            gtol: defaults.gtol,
        }
    }

    /// Assemble the [`TrfOptions`] this problem's fields describe.
    pub fn options(&self) -> TrfOptions {
        TrfOptions {
            ftol: self.ftol,
            xtol: self.xtol,
            gtol: self.gtol,
            max_nfev: self.max_nfev,
            x_scale: self.x_scale.clone(),
            loss: self.loss,
            f_scale: self.f_scale,
        }
    }
}

/// Solve a [`DataProblem`] using the default in-crate SVD backend.
///
/// This is the entry a host-language binding calls after selecting a kind and
/// passing data arrays; the trust-region loop never re-enters the host language.
/// The default backend is the native pure-Rust path (not bit-identical to SciPy);
/// use [`solve_data_problem_with`] with the [`crate::hostlapack`] backend for
/// bit-for-bit parity.
pub fn solve_data_problem(problem: &DataProblem) -> Result<TrfResult, TrfError> {
    problem.kind.validate(&problem.x0)?;
    solve_model(&problem.kind, &problem.x0, &problem.options())
}

/// Solve a [`DataProblem`] through an injected [`ThinSvd`] backend (inject
/// [`crate::hostlapack::LapackSvd`] for bit-for-bit SciPy parity).
pub fn solve_data_problem_with(
    problem: &DataProblem,
    svd: &dyn ThinSvd,
) -> Result<TrfResult, TrfError> {
    problem.kind.validate(&problem.x0)?;
    solve_model_with(&problem.kind, &problem.x0, svd, &problem.options())
}

fn check_len(what: &'static str, got: usize, expected: usize) -> Result<(), TrfError> {
    if got == expected {
        Ok(())
    } else {
        Err(TrfError::InvalidSliceLength {
            what,
            expected,
            got,
        })
    }
}
