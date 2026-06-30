//! Generic weighted least-squares substrate.
//!
//! Domain-free numerical building blocks for nonlinear least-squares fitting:
//! a forward-difference Jacobian and a trust-region (trf-style) Gauss-Newton
//! solver. Nothing here knows about GNSS, orbits, or any physical units; the
//! caller supplies a residual closure `r: R^n -> R^m` and optional diagonal
//! weights.
//!
//! Two distinct numerical regimes live in this module:
//!
//! - The residual evaluation and the finite-difference Jacobian are pure
//!   `f64` arithmetic plus libm `exp`/etc. inside the caller's closure. Their
//!   operation order is fixed and reproducible, so they reproduce a reference
//!   implementation (e.g. scipy `approx_derivative`) bit-for-bit when the same
//!   recipe and the same libm are used.
//! - The solver's trust-region step solves a linear subproblem via a matrix
//!   factorization. That factorization goes through dense linear algebra whose
//!   last bits depend on the BLAS/LAPACK backend, so the converged solution is
//!   reproducible only to a tight tolerance, not bit-for-bit. The owned
//!   [`TrustRegionSolve::OwnedGaussianFirstTie`] variant replaces only the
//!   factorization with a fixed-order scalar kernel; the normal-matrix /
//!   gradient / norm reductions that build the subproblem stay on nalgebra, so
//!   its cross-platform bit guarantee is scoped to the factorization (see that
//!   variant's docs).
//!
//! Keeping the finite-difference primitive separate from the linear-algebra
//! step lets callers assert the former to the bit while treating the latter as
//! a tolerance-bound agreement.
//!
//! # Relationship to the `trust-region-least-squares` crate
//!
//! The workspace also ships [`trust-region-least-squares`], a standalone,
//! publishable solver that reproduces SciPy's trust-region-reflective
//! `least_squares` (its dense unbounded `n = 3`, linear-loss, 2-point-Jacobian
//! path) bit-for-bit, via an SVD-based iteration with an injectable SVD/BLAS
//! seam. This module is a *different* algorithm: a Levenberg-damped Gauss-Newton
//! trust region with no SVD and no reflection, tuned for the GNSS estimation
//! stack and unconstrained in dimension. The two are deliberately not unified.
//!
//! The reason is bit-exactness, not convenience. The only callers of
//! [`solve_trf`]/[`solve_trf_with`] are the SPP solve and the reduced-orbit fit;
//! their converged outputs are pinned to bit-exact goldens (SPP's
//! geometry/clock reference is Skyfield; the reduced-orbit fit is pinned to an
//! independent Astropy/SciPy oracle arc). RTK and PPP do not use this solver.
//! Those goldens were produced by *this* iteration's exact floating-point
//! trajectory; repointing the callers at the SVD/TRF crate's iteration would
//! change the converged values and break the goldens, because the two solvers
//! take different steps even when both converge. So there is no
//! behavior-preserving merge: the public scipy-compatible solver is the crate,
//! the GNSS-tuned solver is this module, and unifying them is a
//! golden-rebaselining decision reserved for the repo owner rather than
//! something to force here.
//!
//! [`trust-region-least-squares`]: https://docs.rs/trust-region-least-squares

use nalgebra::{DMatrix, DVector};

/// Relative finite-difference step for a 2-point (forward) scheme: `sqrt(eps)`
/// for `f64`, i.e. `2^-26`. This matches scipy's `_eps_for_method` choice for
/// the `"2-point"` method.
pub const FD_REL_STEP_2POINT: f64 = 1.4901161193847656e-8; // 0x1.0p-26 == sqrt(2^-52)

/// Default first-order optimality tolerance (scipy `least_squares` `gtol`).
const TRF_DEFAULT_GTOL: f64 = 1e-10;
/// Default relative-cost-reduction tolerance (scipy `least_squares` `ftol`).
const TRF_DEFAULT_FTOL: f64 = 1e-8;
/// Default relative-step tolerance (scipy `least_squares` `xtol`).
const TRF_DEFAULT_XTOL: f64 = 1e-8;
/// Default maximum residual evaluations.
const TRF_DEFAULT_MAX_NFEV: usize = 300;
/// Initial Levenberg damping as a fraction of the largest Gauss-Newton normal
/// diagonal: `mu0 = TRF_INITIAL_DAMPING_SCALE * max_i (J^T J)_ii`.
const TRF_INITIAL_DAMPING_SCALE: f64 = 1e-3;

/// Per-parameter step pieces for a single forward-difference column, recorded
/// in evaluation order so they can be inspected or compared against a
/// reference trace.
#[derive(Debug, Clone, PartialEq)]
pub struct FdStep {
    /// Index of the perturbed parameter.
    pub param_index: usize,
    /// `+1.0` if `x0[i] >= 0`, else `-1.0` (the `(x0>=0)*2 - 1` convention;
    /// note `x0[i] == 0` yields `+1.0`).
    pub sign_x0: f64,
    /// Nominal step `rel_step * sign_x0 * max(1, |x0[i]|)`.
    pub h: f64,
    /// Effective step after rounding: `(x0[i] + h) - x0[i]`. This is the
    /// denominator actually used for the column, recomputed rather than reused
    /// from `h`.
    pub dx: f64,
    /// The perturbed parameter vector (only component `i` bumped by `h`).
    pub x_perturbed: DVector<f64>,
}

/// Compute the per-parameter forward-difference step pieces for `x0`.
///
/// `sign_x0[i] = +1 if x0[i] >= 0 else -1`, `h[i] = rel_step * sign_x0[i] *
/// max(1, |x0[i]|)`, and the effective step `dx[i] = (x0[i] + h[i]) - x0[i]`.
/// The post-rounding `dx` is the value used as the column denominator.
pub fn fd_steps(x0: &DVector<f64>, rel_step: f64) -> Result<Vec<FdStep>, SolveError> {
    let rel_step = crate::validate::positive_step(rel_step, "rel_step").map_err(map_field_error)?;
    fd_steps_checked(x0, rel_step)
}

fn fd_steps_checked(x0: &DVector<f64>, rel_step: f64) -> Result<Vec<FdStep>, SolveError> {
    validate_nonempty_vector(x0, "parameters")?;
    validate_vector(x0, "parameters")?;
    let steps = fd_steps_unchecked(x0, rel_step);
    for step in &steps {
        validate_value(step.h, "fd_step")?;
        validate_value(step.dx, "fd_step")?;
        if step.dx == 0.0 {
            return Err(invalid_input("fd_step", "zero"));
        }
        validate_vector(&step.x_perturbed, "perturbed parameters")?;
    }
    Ok(steps)
}

fn fd_steps_unchecked(x0: &DVector<f64>, rel_step: f64) -> Vec<FdStep> {
    (0..x0.len())
        .map(|i| {
            let xi = x0[i];
            let sign_x0 = if xi >= 0.0 { 1.0 } else { -1.0 };
            let h = rel_step * sign_x0 * xi.abs().max(1.0);
            let mut x_perturbed = x0.clone();
            x_perturbed[i] = xi + h;
            let dx = x_perturbed[i] - xi;
            FdStep {
                param_index: i,
                sign_x0,
                h,
                dx,
                x_perturbed,
            }
        })
        .collect()
}

/// Forward (2-point) finite-difference Jacobian of `residual` at `x0`, given a
/// precomputed `f0 = residual(x0)`.
///
/// Column `i` is `(residual(x0 + h_i e_i) - f0) / dx_i`, where `h_i` and `dx_i`
/// come from [`fd_steps`]. The arithmetic is plain `f64` (no fused multiply-add):
/// each entry is one subtraction followed by one division, matching scipy's
/// `approx_derivative` operation order.
///
/// `f0` is passed in (rather than recomputed) so the caller controls the base
/// evaluation and so the same `f0` used elsewhere is reused exactly.
pub fn jacobian_2point<F>(
    residual: F,
    x0: &DVector<f64>,
    f0: &DVector<f64>,
) -> Result<DMatrix<f64>, SolveError>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    jacobian_2point_checked(|x| Ok(residual(x)), x0, f0)
}

fn jacobian_2point_checked<F>(
    residual: F,
    x0: &DVector<f64>,
    f0: &DVector<f64>,
) -> Result<DMatrix<f64>, SolveError>
where
    F: Fn(&DVector<f64>) -> Result<DVector<f64>, SolveError>,
{
    validate_nonempty_vector(x0, "parameters")?;
    validate_vector(x0, "parameters")?;
    validate_nonempty_vector(f0, "residual")?;
    validate_vector(f0, "residual")?;
    let m = f0.len();
    let n = x0.len();
    let steps = fd_steps_checked(x0, FD_REL_STEP_2POINT)?;
    let mut jac = DMatrix::zeros(m, n);
    for step in &steps {
        let f1 = residual(&step.x_perturbed)?;
        validate_nonempty_vector(&f1, "residual")?;
        validate_vector(&f1, "residual")?;
        if f1.len() != m {
            return Err(invalid_input("residual", "length mismatch"));
        }
        let i = step.param_index;
        for row in 0..m {
            jac[(row, i)] = (f1[row] - f0[row]) / step.dx;
        }
    }
    validate_matrix(&jac, "jacobian")?;
    Ok(jac)
}

/// Termination state of a [`solve_trf`] run, mirroring the scipy
/// `least_squares` status codes for the conditions this solver detects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// `||J^T r||_inf` fell below `gtol` (first-order optimality).
    GradientTolerance,
    /// The relative cost reduction fell below `ftol`.
    CostTolerance,
    /// The relative step size fell below `xtol`.
    StepTolerance,
    /// The maximum number of residual evaluations was reached.
    MaxEvaluations,
}

/// Stopping tolerances and evaluation budget for [`solve_trf`].
#[derive(Debug, Clone, Copy)]
pub struct SolveOptions {
    /// First-order optimality tolerance on `||J^T r||_inf`.
    pub gtol: f64,
    /// Relative-cost-reduction tolerance.
    pub ftol: f64,
    /// Relative-step tolerance.
    pub xtol: f64,
    /// Maximum number of residual evaluations.
    pub max_nfev: usize,
}

impl Default for SolveOptions {
    fn default() -> Self {
        // scipy's defaults for these tolerances.
        Self {
            gtol: TRF_DEFAULT_GTOL,
            ftol: TRF_DEFAULT_FTOL,
            xtol: TRF_DEFAULT_XTOL,
            max_nfev: TRF_DEFAULT_MAX_NFEV,
        }
    }
}

/// Result of a [`solve_trf`] run.
#[derive(Debug, Clone)]
pub struct LeastSquaresReport {
    /// Converged parameter vector.
    pub x: DVector<f64>,
    /// Residual at `x`.
    pub residual: DVector<f64>,
    /// Cost `0.5 * dot(r, r)` at `x`.
    pub cost: f64,
    /// Finite-difference Jacobian at `x`.
    pub jacobian: DMatrix<f64>,
    /// First-order optimality `||J^T r||_inf` at `x`.
    pub optimality_inf: f64,
    /// Number of accepted iterations.
    pub iterations: usize,
    /// Why the solve stopped.
    pub status: Status,
}

/// How the trust-region subproblem `(J^T J + mu I) dx = -J^T r` is solved at
/// each iterate. Both are dense small-system solves; they differ only in the
/// factorization, which is the one place the converged bits move *between the
/// two variants*. The surrounding Gauss-Newton reductions that build the
/// subproblem (the normal matrix `J^T J`, the gradient `J^T r`, the cost dot
/// product, the step/state norms, the optimality `amax`) are shared, identical
/// for both variants, and computed with nalgebra's dense algebra.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TrustRegionSolve {
    /// nalgebra LU factorization. This is the legacy SPP path: its last bit is
    /// BLAS/LAPACK-backend dependent, so its converged solution is reproducible
    /// only to a tight tolerance, not bit-for-bit across machines.
    #[default]
    NalgebraLu,
    /// Owned deterministic Gaussian elimination with partial pivoting and a
    /// fixed reduction order ([`crate::astro::math::linear::solve_linear_first_tie`])
    /// for the dense trust-region subproblem `(J^T J + mu I) dx = -J^T r`: no
    /// nalgebra LU and no black-box BLAS in that factorization, so the
    /// factorization step is reproducible to the bit on any platform.
    ///
    /// Determinism scope: this variant owns ONLY the subproblem factorization.
    /// The reductions that FORM the subproblem each iterate -- the normal
    /// matrix `J^T J` and gradient `J^T r` (nalgebra GEMM/GEMV), the cost dot
    /// product, and the step/state norms -- still go through nalgebra's dense
    /// `DMatrix`/`DVector` algebra, which (nalgebra 0.33 without the BLAS
    /// feature) dispatches to `matrixmultiply`'s CPU-tuned FMA kernels. Those
    /// reductions are therefore NOT bit-portable across CPU targets. The
    /// converged solution is bit-reproducible run-to-run for a fixed build, so
    /// its frozen-bits golden pins THAT build's output; the cross-platform bit
    /// guarantee is scoped to the factorization, not the whole solve. Owning the
    /// assembly reductions too (fixed-order scalar `J^T J`/`J^T r`/dot/norm)
    /// would make the entire solve portable, but is a separate,
    /// behavior-changing step that would re-pin the owned bits.
    OwnedGaussianFirstTie,
}

/// Error from [`solve_trf`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum SolveError {
    /// The Jacobian is rank-deficient / the trust-region subproblem has no
    /// usable descent direction (degenerate geometry).
    #[error("singular or rank-deficient Jacobian: no usable descent direction")]
    SingularJacobian,
    /// A boundary input or derived least-squares quantity was malformed.
    #[error("invalid least-squares {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

/// Cost `0.5 * dot(r, r)`, a plain fold of `f64` operations.
pub fn cost(residual: &DVector<f64>) -> Result<f64, SolveError> {
    validate_nonempty_vector(residual, "residual")?;
    validate_vector(residual, "residual")?;
    validate_value(0.5 * residual.dot(residual), "cost")
}

// --- Jacobian-derived geometry: covariance and Hessian-trace primitives -----

/// Parameter covariance from a design (Jacobian) matrix via the Gauss-Newton
/// normal equations: `sigma^2 (J^T J)^-1`.
///
/// `jacobian` is an `m x n` design matrix with `m >= n` (at least as many
/// residuals as parameters). `variance_scale` (`sigma^2`) multiplies the raw
/// inverse normal matrix: pass the post-fit reduced chi-square to get the fitted
/// parameter covariance, or `1.0` for the bare `(J^T J)^-1` cofactor.
///
/// The covariance is formed from the thin SVD of `J` directly, not from a
/// factorization of `J^T J`. With `J = U S V^T`, the normal-equation inverse is
/// `(J^T J)^-1 = V S^-2 V^T`, so the covariance is
/// `variance_scale * V diag(1/sigma_i^2) V^T`. Going through the SVD of `J`
/// rather than inverting `J^T J` avoids squaring the condition number: a
/// full-rank but near-collinear Jacobian (large `cond(J)`) would become
/// numerically singular under `cond(J^T J) = cond(J)^2`, whereas the SVD path
/// keeps the conditioning at `cond(J)`. A genuinely rank-deficient Jacobian
/// (a singular value at or below the relative rank threshold) yields
/// [`SolveError::SingularJacobian`].
///
/// This is the same quantity, and the same construction, that
/// `scipy.optimize.curve_fit` reports as `pcov` (`pcov = (J^T J)^-1 * s_sq`,
/// formed from the SVD of `J`); the two agree to a tight tolerance for a
/// well-conditioned `J` and stay stable as `J` approaches collinearity.
pub fn normal_covariance(
    jacobian: &DMatrix<f64>,
    variance_scale: f64,
) -> Result<DMatrix<f64>, SolveError> {
    validate_matrix(jacobian, "jacobian")?;
    let m = jacobian.nrows();
    let n = jacobian.ncols();
    if n == 0 || m == 0 {
        return Err(invalid_input("jacobian", "empty"));
    }
    if m < n {
        return Err(invalid_input("jacobian", "fewer rows than columns"));
    }
    crate::validate::finite_nonneg(variance_scale, "variance_scale").map_err(map_field_error)?;

    // Thin SVD of J (right singular vectors only). cov = variance_scale * V S^-2 V^T.
    let svd = jacobian.clone().svd(false, true);
    let v_t = svd.v_t.ok_or(SolveError::SingularJacobian)?;
    let singular = svd.singular_values;

    // Rank guard: a singular value at or below the relative threshold means the
    // covariance is unbounded, i.e. the Jacobian is rank-deficient. This is the
    // SVD analogue of the previous Cholesky-failure check, but it also catches
    // the near-collinear case that squaring into J^T J would have masked.
    let smax = singular.iter().cloned().fold(0.0_f64, f64::max);
    if smax == 0.0 {
        return Err(SolveError::SingularJacobian);
    }
    let threshold = smax * (m.max(n) as f64) * f64::EPSILON;
    if singular.iter().any(|&s| s <= threshold) {
        return Err(SolveError::SingularJacobian);
    }

    // cov[i][j] = variance_scale * sum_k V[i,k] (1/sigma_k^2) V[j,k].
    // v_t is n x n with row k = the k-th right singular vector, so V[i,k] = v_t[(k, i)].
    let mut cov = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                let inv_s2 = 1.0 / (singular[k] * singular[k]);
                acc += v_t[(k, i)] * v_t[(k, j)] * inv_s2;
            }
            cov[(i, j)] = acc * variance_scale;
        }
    }
    validate_matrix(&cov, "covariance")?;
    Ok(cov)
}

/// Trace of the Gauss-Newton Hessian approximation `J^T J`, i.e. the sum of the
/// squared column norms of `jacobian`.
///
/// No inverse is formed: this is `sum_i ||J[:, i]||^2 == trace(J^T J)`, summed
/// column-by-column. It equals `numpy.trace(jac.T @ jac)` to a tight tolerance
/// (the reductions differ only in summation order).
pub fn hessian_trace(jacobian: &DMatrix<f64>) -> f64 {
    let n = jacobian.ncols();
    let m = jacobian.nrows();
    let mut trace = 0.0;
    for i in 0..n {
        let mut col = 0.0;
        for r in 0..m {
            let v = jacobian[(r, i)];
            col += v * v;
        }
        trace += col;
    }
    trace
}

/// Fitted parameter covariance directly from a design (Jacobian) matrix and the
/// post-fit cost, with the redundancy taken from the Jacobian's own shape.
///
/// This is the binding-facing primitive: it forms the covariance straight from
/// the design matrix and the scalar cost, with no [`LeastSquaresReport`] and no
/// fabricated residual / parameter vectors. The degrees of freedom come from the
/// Jacobian's dimensions alone (`m = jacobian.nrows()`, `n = jacobian.ncols()`),
/// so there are no redundant lengths to keep consistent. It scales `(J^T J)^-1`
/// by the post-fit reduced chi-square `s_sq = 2 * cost / (m - n)` (the residual
/// sum of squares over the redundancy), the same scale `scipy.optimize.curve_fit`
/// applies to its `pcov`. Requires positive redundancy `m > n`; otherwise
/// returns [`SolveError::InvalidInput`].
pub fn covariance_from_jacobian(
    jacobian: &DMatrix<f64>,
    cost: f64,
) -> Result<DMatrix<f64>, SolveError> {
    let m = jacobian.nrows();
    let n = jacobian.ncols();
    if m <= n {
        return Err(invalid_input("degrees_of_freedom", "not positive"));
    }
    let dof = (m - n) as f64;
    let s_sq = validate_value(2.0 * cost / dof, "reduced_chi_square")?;
    normal_covariance(jacobian, s_sq)
}

/// Fitted parameter covariance from a converged [`LeastSquaresReport`].
///
/// Convenience over [`covariance_from_jacobian`] for real-report callers: it
/// validates that the report's Jacobian shape agrees with its residual / `x`
/// lengths, then delegates to [`covariance_from_jacobian`] so the report path
/// and the Jacobian path share a single reduced-chi-square scaling. Requires
/// positive redundancy `m > n`; otherwise returns [`SolveError::InvalidInput`].
pub fn covariance_from_report(report: &LeastSquaresReport) -> Result<DMatrix<f64>, SolveError> {
    let m = report.residual.len();
    let n = report.x.len();
    // `LeastSquaresReport`'s fields are public, so the residual/x lengths that
    // set the degrees of freedom and the Jacobian that sets the scale can be
    // inconsistent. Reject a Jacobian whose shape does not match (m x n) rather
    // than silently scaling a covariance of the Jacobian's dimensions by a
    // reduced chi-square derived from unrelated vectors.
    if report.jacobian.nrows() != m {
        return Err(invalid_input("jacobian", "rows must match residual length"));
    }
    if report.jacobian.ncols() != n {
        return Err(invalid_input(
            "jacobian",
            "columns must match parameter length",
        ));
    }
    covariance_from_jacobian(&report.jacobian, report.cost)
}

/// A nonlinear least-squares problem: a residual closure, optional diagonal
/// weights, and a starting point. The weighted form scales both the residual
/// and the Jacobian rows by `sqrt(weight)`; with all weights `1` it reduces to
/// the ordinary (unweighted) least-squares problem.
pub struct LeastSquaresProblem<F> {
    residual: F,
    /// `sqrt` of the diagonal weights, or `None` for the identity weighting.
    sqrt_weights: Option<DVector<f64>>,
    x0: DVector<f64>,
}

impl<F> LeastSquaresProblem<F>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    /// An unweighted problem (identity weighting).
    pub fn new(residual: F, x0: DVector<f64>) -> Self {
        Self {
            residual,
            sqrt_weights: None,
            x0,
        }
    }

    /// A problem with diagonal weights `W`; residual and Jacobian rows are
    /// scaled by `sqrt(W)`.
    pub fn with_weights(residual: F, x0: DVector<f64>, weights: DVector<f64>) -> Self {
        let sqrt_weights = weights.map(f64::sqrt);
        Self {
            residual,
            sqrt_weights: Some(sqrt_weights),
            x0,
        }
    }

    /// Weighted residual at `x`.
    fn weighted_residual(&self, x: &DVector<f64>) -> Result<DVector<f64>, SolveError> {
        validate_nonempty_vector(x, "parameters")?;
        validate_vector(x, "parameters")?;
        let r = (self.residual)(x);
        validate_nonempty_vector(&r, "residual")?;
        validate_vector(&r, "residual")?;
        match &self.sqrt_weights {
            Some(sw) => {
                validate_nonempty_vector(sw, "weights")?;
                validate_vector(sw, "weights")?;
                if sw.len() != r.len() {
                    return Err(invalid_input("weights", "length mismatch"));
                }
                let weighted = r.component_mul(sw);
                validate_vector(&weighted, "weighted residual")?;
                Ok(weighted)
            }
            None => Ok(r),
        }
    }
}

/// Trust-region (trf-style) Gauss-Newton solve.
///
/// At each iterate the weighted residual and its forward-difference Jacobian
/// are formed, then a Levenberg-damped Gauss-Newton step `(J^T J + mu I) dx =
/// -J^T r` is taken inside a trust region; the damping `mu` is grown on a
/// rejected step and shrunk on an accepted one. The linear solve uses a dense
/// factorization, so the converged solution is reproducible to a tight
/// tolerance rather than to the bit.
///
/// Returns [`SolveError::SingularJacobian`] if the normal-equation system
/// cannot be solved (degenerate geometry).
///
/// Uses the legacy [`TrustRegionSolve::NalgebraLu`] subproblem solver; for the
/// owned deterministic factorization call [`solve_trf_with`].
pub fn solve_trf<F>(
    problem: &LeastSquaresProblem<F>,
    opts: &SolveOptions,
) -> Result<LeastSquaresReport, SolveError>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    solve_trf_with(problem, opts, TrustRegionSolve::NalgebraLu)
}

/// Solve the trust-region subproblem `(J^T J + mu I) dx = rhs` with the
/// selected factorization. The two arms produce the same algebra; they differ
/// only in the dense solve's operation order (see [`TrustRegionSolve`]).
fn solve_subproblem(
    lhs: &DMatrix<f64>,
    rhs: &DVector<f64>,
    linear_solve: TrustRegionSolve,
) -> Option<DVector<f64>> {
    match linear_solve {
        TrustRegionSolve::NalgebraLu => lhs.clone().lu().solve(rhs),
        TrustRegionSolve::OwnedGaussianFirstTie => {
            let n = rhs.len();
            let a: Vec<Vec<f64>> = (0..n)
                .map(|i| (0..n).map(|j| lhs[(i, j)]).collect())
                .collect();
            let b: Vec<f64> = rhs.iter().copied().collect();
            crate::astro::math::linear::solve_linear_first_tie(&a, &b).map(DVector::from_vec)
        }
    }
}

/// [`solve_trf`] with an explicit choice of the trust-region subproblem solver.
/// `NalgebraLu` reproduces the legacy SPP path; `OwnedGaussianFirstTie` is the
/// owned deterministic kernel pinned to its own frozen-bits goldens.
pub fn solve_trf_with<F>(
    problem: &LeastSquaresProblem<F>,
    opts: &SolveOptions,
    linear_solve: TrustRegionSolve,
) -> Result<LeastSquaresReport, SolveError>
where
    F: Fn(&DVector<f64>) -> DVector<f64>,
{
    validate_options(opts)?;
    let n = problem.x0.len();

    let mut x = problem.x0.clone();
    validate_nonempty_vector(&x, "initial parameters")?;
    validate_vector(&x, "initial parameters")?;
    let mut r = problem.weighted_residual(&x)?;
    let mut f0 = r.clone();
    let mut jac = jacobian_2point_checked(|p| problem.weighted_residual(p), &x, &f0)?;
    let mut nfev = 1usize; // the f0 above
    let mut cur_cost = cost(&r)?;

    // Initial Levenberg damping scaled to the Gauss-Newton normal matrix.
    let jtj0 = jac.transpose() * &jac;
    validate_matrix(&jtj0, "normal matrix")?;
    let mut mu = TRF_INITIAL_DAMPING_SCALE
        * (0..n)
            .map(|i| jtj0[(i, i)])
            .fold(0.0_f64, f64::max)
            .max(1.0);

    let mut iterations = 0usize;

    loop {
        let jt = jac.transpose();
        let grad = &jt * &r;
        validate_vector(&grad, "gradient")?;
        let optimality_inf = validate_value(grad.amax(), "optimality")?;

        if optimality_inf < opts.gtol {
            return finish(x, r, cur_cost, jac, iterations, Status::GradientTolerance);
        }
        if nfev >= opts.max_nfev {
            return finish(x, r, cur_cost, jac, iterations, Status::MaxEvaluations);
        }

        let jtj = &jt * &jac;
        validate_matrix(&jtj, "normal matrix")?;

        // Levenberg-damped Gauss-Newton subproblem.
        let mut accepted = false;
        for _ in 0..30 {
            let mut lhs = jtj.clone();
            for i in 0..n {
                lhs[(i, i)] += mu;
            }
            let rhs = -&grad;
            validate_matrix(&lhs, "subproblem matrix")?;
            validate_vector(&rhs, "subproblem rhs")?;
            let step = match solve_subproblem(&lhs, &rhs, linear_solve) {
                Some(s) => s,
                None => return Err(SolveError::SingularJacobian),
            };
            validate_vector(&step, "step")?;

            let x_trial = &x + &step;
            let r_trial = problem.weighted_residual(&x_trial)?;
            nfev += 1;
            let cost_trial = cost(&r_trial)?;

            if cost_trial < cur_cost {
                // Accept; relative-cost and relative-step stopping checks.
                let cost_reduction = (cur_cost - cost_trial) / cur_cost.max(f64::MIN_POSITIVE);
                let step_norm = step.norm();
                let x_norm = x.norm();
                let rel_step = step_norm / x_norm.max(f64::MIN_POSITIVE);

                x = x_trial;
                r = r_trial;
                cur_cost = cost_trial;
                f0 = r.clone();
                jac = jacobian_2point_checked(|p| problem.weighted_residual(p), &x, &f0)?;
                nfev += n; // FD probes for the new Jacobian
                iterations += 1;
                mu *= 0.5;
                accepted = true;

                if cost_reduction < opts.ftol {
                    return finish(x, r, cur_cost, jac, iterations, Status::CostTolerance);
                }
                if rel_step < opts.xtol {
                    return finish(x, r, cur_cost, jac, iterations, Status::StepTolerance);
                }
                break;
            } else {
                // Reject: grow damping and retry the subproblem.
                mu *= 2.0;
            }
        }

        if !accepted {
            // Could not find an improving step within the damping sweep.
            return finish(x, r, cur_cost, jac, iterations, Status::StepTolerance);
        }
    }
}

fn finish(
    x: DVector<f64>,
    residual: DVector<f64>,
    cost_value: f64,
    jacobian: DMatrix<f64>,
    iterations: usize,
    status: Status,
) -> Result<LeastSquaresReport, SolveError> {
    validate_nonempty_vector(&x, "solution")?;
    validate_vector(&x, "solution")?;
    validate_nonempty_vector(&residual, "residual")?;
    validate_vector(&residual, "residual")?;
    validate_value(cost_value, "cost")?;
    validate_matrix(&jacobian, "jacobian")?;
    let optimality_inf = validate_value((jacobian.transpose() * &residual).amax(), "optimality")?;
    Ok(LeastSquaresReport {
        x,
        residual,
        cost: cost_value,
        jacobian,
        optimality_inf,
        iterations,
        status,
    })
}

fn validate_value(value: f64, field: &'static str) -> Result<f64, SolveError> {
    crate::validate::finite(value, field).map_err(map_field_error)
}

fn validate_options(opts: &SolveOptions) -> Result<(), SolveError> {
    crate::validate::positive_step(opts.gtol, "gtol").map_err(map_field_error)?;
    crate::validate::positive_step(opts.ftol, "ftol").map_err(map_field_error)?;
    crate::validate::positive_step(opts.xtol, "xtol").map_err(map_field_error)?;
    if opts.max_nfev == 0 {
        return Err(invalid_input("max_nfev", "not positive"));
    }
    Ok(())
}

fn validate_nonempty_vector(vector: &DVector<f64>, field: &'static str) -> Result<(), SolveError> {
    if vector.is_empty() {
        Err(invalid_input(field, "empty"))
    } else {
        Ok(())
    }
}

fn validate_vector(vector: &DVector<f64>, field: &'static str) -> Result<(), SolveError> {
    crate::validate::finite_slice(vector.as_slice(), field).map_err(map_field_error)
}

fn validate_matrix(matrix: &DMatrix<f64>, field: &'static str) -> Result<(), SolveError> {
    crate::validate::finite_slice(matrix.as_slice(), field).map_err(map_field_error)
}

fn map_field_error(error: crate::validate::FieldError) -> SolveError {
    invalid_input(error.field(), error.reason())
}

fn invalid_input(field: &'static str, reason: &'static str) -> SolveError {
    SolveError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_rel_step_is_sqrt_eps() {
        assert_eq!(FD_REL_STEP_2POINT, (2.0_f64.powi(-52)).sqrt());
        assert_eq!(FD_REL_STEP_2POINT, 2.0_f64.powi(-26));
    }

    #[test]
    fn fd_step_sign_convention() {
        let x0 = DVector::from_vec(vec![5.0, -2.0, 0.0]);
        let steps = fd_steps(&x0, FD_REL_STEP_2POINT).unwrap();
        assert_eq!(steps[0].sign_x0, 1.0);
        assert_eq!(steps[1].sign_x0, -1.0);
        assert_eq!(steps[2].sign_x0, 1.0); // x == 0 -> +1
    }

    #[test]
    fn fd_steps_rejects_zero_relative_step() {
        let x0 = DVector::from_vec(vec![1.0]);
        assert_invalid_field(fd_steps(&x0, 0.0).unwrap_err(), "rel_step");
    }

    #[test]
    fn fd_steps_rejects_nonfinite_parameters() {
        let x0 = DVector::from_vec(vec![1.0, f64::NAN]);
        assert_invalid_field(fd_steps(&x0, FD_REL_STEP_2POINT).unwrap_err(), "parameters");
    }

    #[test]
    fn jacobian_rejects_residual_length_mismatch() {
        let x0 = DVector::from_vec(vec![1.0, 2.0]);
        let f0 = DVector::from_vec(vec![1.0, 2.0]);
        let residual = |_: &DVector<f64>| DVector::from_vec(vec![1.0]);
        assert_invalid_field(jacobian_2point(residual, &x0, &f0).unwrap_err(), "residual");
    }

    #[test]
    fn cost_rejects_nonfinite_residual() {
        assert_invalid_field(
            cost(&DVector::from_vec(vec![1.0, f64::INFINITY])).unwrap_err(),
            "residual",
        );
    }

    #[test]
    fn exp_fit_converges() {
        // a*exp(b*t) + c with a known minimum near the generated data.
        let t = vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0];
        let y = vec![
            3.0123, 2.2083, 1.6889, 1.3713, 1.0903, 0.9302, 0.8104, 0.6303,
        ];
        let tt = t.clone();
        let yy = y.clone();
        let residual = move |p: &DVector<f64>| {
            let (a, b, c) = (p[0], p[1], p[2]);
            DVector::from_iterator(
                tt.len(),
                tt.iter()
                    .zip(&yy)
                    .map(|(&tk, &yk)| a * (b * tk).exp() + c - yk),
            )
        };
        let problem = LeastSquaresProblem::new(residual, DVector::from_vec(vec![5.0, -2.0, 2.0]));
        let report = solve_trf(&problem, &SolveOptions::default()).unwrap();
        assert!(report.cost < 1.0, "cost did not reduce: {}", report.cost);
    }

    #[test]
    fn solve_trf_rejects_nonfinite_initial_residual() {
        fn residual(_: &DVector<f64>) -> DVector<f64> {
            DVector::from_element(1, f64::NAN)
        }
        let problem = LeastSquaresProblem::new(residual, DVector::from_element(1, 0.0));
        assert_invalid_field(
            solve_trf(&problem, &SolveOptions::default()).unwrap_err(),
            "residual",
        );
    }

    #[test]
    fn solve_trf_rejects_nonfinite_initial_cost() {
        fn residual(_: &DVector<f64>) -> DVector<f64> {
            DVector::from_element(1, f64::MAX)
        }
        let problem = LeastSquaresProblem::new(residual, DVector::from_element(1, 0.0));
        assert_invalid_field(
            solve_trf(&problem, &SolveOptions::default()).unwrap_err(),
            "cost",
        );
    }

    #[test]
    fn solve_trf_rejects_nonfinite_trial_residual_instead_of_converging() {
        use std::cell::Cell;

        let calls = Cell::new(0usize);
        let residual = move |p: &DVector<f64>| {
            let call = calls.get();
            calls.set(call + 1);
            if call >= 2 {
                DVector::from_element(1, f64::NAN)
            } else {
                DVector::from_element(1, p[0])
            }
        };
        let problem = LeastSquaresProblem::new(residual, DVector::from_element(1, 1.0));
        assert_invalid_field(
            solve_trf(&problem, &SolveOptions::default()).unwrap_err(),
            "residual",
        );
    }

    #[test]
    fn solve_trf_rejects_invalid_options() {
        fn residual(p: &DVector<f64>) -> DVector<f64> {
            DVector::from_element(1, p[0])
        }
        let problem = LeastSquaresProblem::new(residual, DVector::from_element(1, 1.0));
        let opts = SolveOptions {
            gtol: f64::NAN,
            ..SolveOptions::default()
        };
        assert_invalid_field(solve_trf(&problem, &opts).unwrap_err(), "gtol");

        let opts = SolveOptions {
            max_nfev: 0,
            ..SolveOptions::default()
        };
        assert_invalid_field(solve_trf(&problem, &opts).unwrap_err(), "max_nfev");
    }

    #[test]
    fn solve_trf_rejects_weight_residual_dimension_mismatch() {
        fn residual(_: &DVector<f64>) -> DVector<f64> {
            DVector::from_vec(vec![1.0, 2.0])
        }
        let problem = LeastSquaresProblem::with_weights(
            residual,
            DVector::from_element(1, 0.0),
            DVector::from_vec(vec![1.0]),
        );
        assert_invalid_field(
            solve_trf(&problem, &SolveOptions::default()).unwrap_err(),
            "weights",
        );
    }

    fn assert_invalid_field(error: SolveError, expected: &'static str) {
        match error {
            SolveError::InvalidInput { field, .. } => assert_eq!(field, expected),
            other => panic!("expected invalid input for {expected}, got {other:?}"),
        }
    }

    /// The exp-fit residual used by the owned-solver tests.
    fn exp_fit_problem() -> LeastSquaresProblem<impl Fn(&DVector<f64>) -> DVector<f64>> {
        let t = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0];
        let y = [
            3.0123, 2.2083, 1.6889, 1.3713, 1.0903, 0.9302, 0.8104, 0.6303,
        ];
        let residual = move |p: &DVector<f64>| {
            let (a, b, c) = (p[0], p[1], p[2]);
            DVector::from_iterator(
                t.len(),
                t.iter()
                    .zip(&y)
                    .map(|(&tk, &yk)| a * (b * tk).exp() + c - yk),
            )
        };
        LeastSquaresProblem::new(residual, DVector::from_vec(vec![5.0, -2.0, 2.0]))
    }

    /// The owned deterministic subproblem solver converges on the exp-fit
    /// problem and reproduces its solution bit-for-bit run to run. The pinned
    /// bits are the owned kernel's own frozen-bits golden (a different
    /// factorization than the legacy nalgebra path, so its own value). The
    /// owned kernel owns only the subproblem factorization; the surrounding
    /// `J^T J`/`J^T r`/norm reductions are shared nalgebra dense algebra, so
    /// these bits are this build's reproducible output (the run-to-run check
    /// below), not a cross-platform constant.
    #[test]
    fn owned_trf_converges_to_frozen_bits() {
        let problem = exp_fit_problem();
        let report = solve_trf_with(
            &problem,
            &SolveOptions::default(),
            TrustRegionSolve::OwnedGaussianFirstTie,
        )
        .unwrap();
        assert!(
            report.cost < 1.0,
            "owned cost did not reduce: {}",
            report.cost
        );
        assert_eq!(report.x[0].to_bits(), 0x4003c3674cdfadef);
        assert_eq!(report.x[1].to_bits(), 0xbfe799e0d1929220);
        assert_eq!(report.x[2].to_bits(), 0x3fe0d5c96d9d3b35);

        // Determinism: a second run is bit-identical.
        let again = solve_trf_with(
            &problem,
            &SolveOptions::default(),
            TrustRegionSolve::OwnedGaussianFirstTie,
        )
        .unwrap();
        for i in 0..3 {
            assert_eq!(report.x[i].to_bits(), again.x[i].to_bits());
        }
    }

    /// Reference Jacobian for the covariance/trace primitives.
    fn covariance_fixture_jacobian() -> DMatrix<f64> {
        // J (m=5, n=2): a linear-fit design matrix [1, t].
        DMatrix::from_row_slice(5, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0])
    }

    #[test]
    fn hessian_trace_matches_numpy() {
        // numpy: trace((J.T @ J)) == 35.0 for the fixture Jacobian.
        let trace = hessian_trace(&covariance_fixture_jacobian());
        assert!((trace - 35.0).abs() < 1e-12, "trace {trace}");
    }

    #[test]
    fn normal_covariance_matches_numpy_pcov() {
        // numpy: inv(J.T @ J) for the fixture Jacobian.
        let inv = normal_covariance(&covariance_fixture_jacobian(), 1.0).unwrap();
        let expected = [[0.6000000000000001, -0.2], [-0.2, 0.1]];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (inv[(i, j)] - expected[i][j]).abs() < 1e-12,
                    "inv[{i}][{j}] = {}",
                    inv[(i, j)]
                );
            }
        }

        // With the post-fit reduced chi-square scale s_sq = SSR/(m-n).
        let s_sq = 0.085 / 3.0;
        let cov = normal_covariance(&covariance_fixture_jacobian(), s_sq).unwrap();
        let expected_cov = [
            [0.017000000000000005, -0.005666666666666667],
            [-0.005666666666666667, 0.0028333333333333335],
        ];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (cov[(i, j)] - expected_cov[i][j]).abs() < 1e-12,
                    "cov[{i}][{j}] = {}",
                    cov[(i, j)]
                );
            }
        }
    }

    #[test]
    fn normal_covariance_rejects_underdetermined_and_negative_scale() {
        let wide = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        assert!(matches!(
            normal_covariance(&wide, 1.0),
            Err(SolveError::InvalidInput {
                field: "jacobian",
                ..
            })
        ));
        assert!(matches!(
            normal_covariance(&covariance_fixture_jacobian(), -1.0),
            Err(SolveError::InvalidInput {
                field: "variance_scale",
                ..
            })
        ));
    }

    #[test]
    fn normal_covariance_matches_closed_form_inverse_for_collinear_jacobian() {
        // A full-rank but collinear design: the second column is the first plus a
        // small ramp, so the two columns are nearly parallel (raised condition
        // number). The SVD path forms the covariance from the SVD of J directly,
        // not from (J^T J)^-1, so it keeps the conditioning at cond(J) rather than
        // squaring it. Compare against the closed-form 2x2 inverse of J^T J (the
        // analytic answer the SVD covariance must reproduce).
        let eps = 1e-2;
        let col1: Vec<f64> = (0..5).map(|k| 1.0 + (k as f64) * eps).collect();
        let mut data = Vec::with_capacity(10);
        for &c1 in &col1 {
            data.push(1.0);
            data.push(c1);
        }
        let jac = DMatrix::from_row_slice(5, 2, &data);
        let scale = 2.5;
        let cov = normal_covariance(&jac, scale).unwrap();

        // Closed-form (J^T J)^-1 * scale for this moderately conditioned design.
        let s00 = 5.0_f64;
        let s01: f64 = col1.iter().sum();
        let s11: f64 = col1.iter().map(|c| c * c).sum();
        let det = s00 * s11 - s01 * s01;
        let inv = [[s11 / det, -s01 / det], [-s01 / det, s00 / det]];
        for i in 0..2 {
            for j in 0..2 {
                let expected = inv[i][j] * scale;
                let tol = 1e-9 * expected.abs().max(1.0);
                assert!(
                    (cov[(i, j)] - expected).abs() < tol,
                    "cov[{i}][{j}] = {} (expected {expected})",
                    cov[(i, j)]
                );
            }
        }
        // Symmetric to roundoff.
        assert!((cov[(0, 1)] - cov[(1, 0)]).abs() <= 1e-12 * cov[(0, 0)].abs().max(1.0));
    }

    #[test]
    fn covariance_from_report_rejects_jacobian_dimension_mismatch() {
        // A report whose Jacobian shape disagrees with the residual/x lengths
        // (public fields let a caller build one) must be rejected, not used to
        // scale a covariance of the Jacobian's own dimensions.
        let jac = covariance_fixture_jacobian(); // 5 x 2
        let mismatched_rows = LeastSquaresReport {
            x: DVector::from_vec(vec![0.0, 0.0]),
            residual: DVector::from_vec(vec![0.1, -0.2, 0.15, 0.05]), // len 4 != 5 rows
            cost: 0.1,
            jacobian: jac.clone(),
            optimality_inf: 0.0,
            iterations: 0,
            status: Status::GradientTolerance,
        };
        assert_invalid_field(
            covariance_from_report(&mismatched_rows).unwrap_err(),
            "jacobian",
        );

        let mismatched_cols = LeastSquaresReport {
            x: DVector::from_vec(vec![0.0, 0.0, 0.0]), // len 3 != 2 cols
            residual: DVector::from_vec(vec![0.1, -0.2, 0.15, 0.05, -0.1]),
            cost: 0.1,
            jacobian: jac,
            optimality_inf: 0.0,
            iterations: 0,
            status: Status::GradientTolerance,
        };
        assert_invalid_field(
            covariance_from_report(&mismatched_cols).unwrap_err(),
            "jacobian",
        );
    }

    #[test]
    fn covariance_from_jacobian_matches_report_path_bit_for_bit() {
        // The Jacobian-only primitive must produce bit-identical covariance to
        // the report path on a matching report (same jacobian/cost, with
        // residual/x lengths chosen to match the Jacobian's m x n shape), and
        // must equal normal_covariance at the explicit reduced-chi-square scale.
        let jac = covariance_fixture_jacobian(); // 5 x 2
        let residual = DVector::from_vec(vec![0.1, -0.2, 0.15, 0.05, -0.1]);
        let cost = 0.5 * residual.dot(&residual);
        let report = LeastSquaresReport {
            x: DVector::from_vec(vec![0.0, 0.0]),
            cost,
            residual,
            jacobian: jac.clone(),
            optimality_inf: 0.0,
            iterations: 0,
            status: Status::GradientTolerance,
        };

        let from_jac = covariance_from_jacobian(&jac, cost).unwrap();
        let from_report = covariance_from_report(&report).unwrap();

        let m = jac.nrows();
        let n = jac.ncols();
        let explicit = normal_covariance(&jac, 2.0 * cost / ((m - n) as f64)).unwrap();

        assert_eq!(from_jac.shape(), from_report.shape());
        for (a, (b, c)) in from_jac.iter().zip(from_report.iter().zip(explicit.iter())) {
            assert_eq!(a.to_bits(), b.to_bits());
            assert_eq!(a.to_bits(), c.to_bits());
        }
    }

    #[test]
    fn covariance_from_jacobian_rejects_insufficient_dof() {
        // m <= n: a square (m == n) and an underdetermined (m < n) design both
        // have non-positive redundancy and must return the typed error, not a
        // panic or a NaN-laden covariance.
        let square = DMatrix::from_row_slice(2, 2, &[1.0, 0.0, 1.0, 1.0]);
        assert_invalid_field(
            covariance_from_jacobian(&square, 0.1).unwrap_err(),
            "degrees_of_freedom",
        );

        let wide = DMatrix::from_row_slice(2, 3, &[1.0, 0.0, 1.0, 0.0, 1.0, 1.0]);
        assert_invalid_field(
            covariance_from_jacobian(&wide, 0.1).unwrap_err(),
            "degrees_of_freedom",
        );
    }

    #[test]
    fn covariance_from_report_uses_reduced_chi_square() {
        // Build a report by hand: residual r and Jacobian J fix the scale.
        let jac = covariance_fixture_jacobian();
        let residual = DVector::from_vec(vec![0.1, -0.2, 0.15, 0.05, -0.1]);
        let report = LeastSquaresReport {
            x: DVector::from_vec(vec![0.0, 0.0]),
            cost: 0.5 * residual.dot(&residual),
            residual,
            jacobian: jac,
            optimality_inf: 0.0,
            iterations: 0,
            status: Status::GradientTolerance,
        };
        let cov = covariance_from_report(&report).unwrap();
        let expected_cov = [
            [0.017000000000000005, -0.005666666666666667],
            [-0.005666666666666667, 0.0028333333333333335],
        ];
        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (cov[(i, j)] - expected_cov[i][j]).abs() < 1e-12,
                    "cov[{i}][{j}] = {}",
                    cov[(i, j)]
                );
            }
        }
    }
}
