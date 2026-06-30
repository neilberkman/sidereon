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
}
