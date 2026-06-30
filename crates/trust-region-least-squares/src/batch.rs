//! Parallel leave-one-out and multi-start re-solves.
//!
//! These fan a family of independent re-solves across a [`rayon`] work-stealing
//! pool. Each re-solve is a self-contained call into the same trust-region
//! engine with the same inputs, so the parallel result at index `i` is
//! bit-identical to running that one re-solve serially: there is no shared
//! mutable state and the work is collected back in index order. (This mirrors
//! the per-index determinism guarantee of the GNSS batch solvers.)
//!
//! [`solve_drop_one`] re-solves with each residual row masked in turn (an
//! influence / jackknife diagnostic); [`solve_perturbed`] re-solves from a set
//! of alternative starting points (a multi-start / sensitivity sweep).

use rayon::prelude::*;

use crate::data::DataProblem;
use crate::model::{solve_model_with, ResidualModel};
use crate::trf::{NalgebraThinSvd, ThinSvd, TrfError, TrfOptions, TrfResult};

/// The result of a leave-one-out sweep: the base solve over all rows, plus one
/// re-solve per masked row.
#[derive(Debug, Clone)]
pub struct DropOneReport {
    /// The solve over the full residual.
    pub base: TrfResult,
    /// `drops[i]` is the solve with residual row `i` masked out. Its order is
    /// the residual-row order, and each entry is bit-identical to an independent
    /// serial drop-`i` solve.
    pub drops: Vec<TrfResult>,
    /// `cost_delta[i] = drops[i].cost - base.cost`: how much the optimum cost
    /// moves when row `i` is removed.
    pub cost_delta: Vec<f64>,
}

/// The result of a multi-start sweep: the base solve from the base start, plus
/// one re-solve per alternative starting point.
#[derive(Debug, Clone)]
pub struct PerturbedReport {
    /// The solve from the base starting point.
    pub base: TrfResult,
    /// `runs[i]` is the solve from `starts[i]`, in the given order; each entry
    /// is bit-identical to an independent serial solve from that start.
    pub runs: Vec<TrfResult>,
}

/// A model that masks one residual row of an inner model. The masked residual
/// has `m - 1` rows; the default 2-point Jacobian follows automatically because
/// it is built from this residual.
struct DropRow<'a, M: ?Sized> {
    inner: &'a M,
    drop: usize,
}

impl<M: ResidualModel + ?Sized> ResidualModel for DropRow<'_, M> {
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

/// Leave-one-out over an injected [`ThinSvd`] backend. Solves the base problem
/// once, then re-solves with each residual row masked, fanning the independent
/// re-solves across a [`rayon`] pool with index-preserving collection.
///
/// The backend must be [`Sync`] so it can be shared across the pool; the
/// in-crate [`NalgebraThinSvd`] and [`crate::hostlapack::LapackSvd`] both are.
pub fn solve_drop_one_with<M>(
    model: &M,
    x0: &[f64],
    svd: &(dyn ThinSvd + Sync),
    options: &TrfOptions,
) -> Result<DropOneReport, TrfError>
where
    M: ResidualModel + Sync + ?Sized,
{
    let base = solve_model_with(model, x0, svd, options)?;
    let m = base.fun.len();

    // Collect every re-solve in index order, then surface the lowest-index error
    // (rayon's `collect` into a `Vec` preserves index order, so this is
    // deterministic across thread schedules; a bare `collect::<Result<_,_>>()`
    // would return whichever error finished the race first).
    let results: Vec<Result<TrfResult, TrfError>> = (0..m)
        .into_par_iter()
        .map(|drop| solve_model_with(&DropRow { inner: model, drop }, x0, svd, options))
        .collect();
    let drops = first_error_in_index_order(results)?;

    let cost_delta = drops.iter().map(|r| r.cost - base.cost).collect();
    Ok(DropOneReport {
        base,
        drops,
        cost_delta,
    })
}

/// Leave-one-out using the default in-crate [`NalgebraThinSvd`] backend.
pub fn solve_drop_one<M>(
    model: &M,
    x0: &[f64],
    options: &TrfOptions,
) -> Result<DropOneReport, TrfError>
where
    M: ResidualModel + Sync + ?Sized,
{
    solve_drop_one_with(model, x0, &NalgebraThinSvd, options)
}

/// Multi-start over an injected [`ThinSvd`] backend. Solves from `x0_base` once,
/// then re-solves from each entry of `starts`, fanned across a [`rayon`] pool
/// with index-preserving collection.
pub fn solve_perturbed_with<M>(
    model: &M,
    x0_base: &[f64],
    starts: &[Vec<f64>],
    svd: &(dyn ThinSvd + Sync),
    options: &TrfOptions,
) -> Result<PerturbedReport, TrfError>
where
    M: ResidualModel + Sync + ?Sized,
{
    let base = solve_model_with(model, x0_base, svd, options)?;
    // Index-ordered collection then lowest-index error: deterministic regardless
    // of which parallel re-solve fails first (see `solve_drop_one_with`).
    let results: Vec<Result<TrfResult, TrfError>> = starts
        .par_iter()
        .map(|start| solve_model_with(model, start, svd, options))
        .collect();
    let runs = first_error_in_index_order(results)?;
    Ok(PerturbedReport { base, runs })
}

/// Reduce an index-ordered vector of per-re-solve results to either all the
/// successes or the lowest-index error. The input must already be in index order
/// (rayon's `collect` into a `Vec` guarantees this); iterating in order and
/// returning on the first `Err` therefore selects the lowest-index failure
/// deterministically, independent of the order the parallel tasks completed.
fn first_error_in_index_order(
    results: Vec<Result<TrfResult, TrfError>>,
) -> Result<Vec<TrfResult>, TrfError> {
    let mut ordered = Vec::with_capacity(results.len());
    for result in results {
        ordered.push(result?);
    }
    Ok(ordered)
}

/// Multi-start using the default in-crate [`NalgebraThinSvd`] backend.
pub fn solve_perturbed<M>(
    model: &M,
    x0_base: &[f64],
    starts: &[Vec<f64>],
    options: &TrfOptions,
) -> Result<PerturbedReport, TrfError>
where
    M: ResidualModel + Sync + ?Sized,
{
    solve_perturbed_with(model, x0_base, starts, &NalgebraThinSvd, options)
}

/// Leave-one-out for a data-driven [`DataProblem`], using the default in-crate
/// SVD backend.
pub fn solve_data_problem_drop_one(problem: &DataProblem) -> Result<DropOneReport, TrfError> {
    problem.kind.validate(&problem.x0)?;
    solve_drop_one_with(
        &problem.kind,
        &problem.x0,
        &NalgebraThinSvd,
        &problem.options(),
    )
}

/// Leave-one-out for a data-driven [`DataProblem`] through an injected
/// [`ThinSvd`] backend (inject [`crate::hostlapack::LapackSvd`] for bit-for-bit
/// parity on every drop).
pub fn solve_data_problem_drop_one_with(
    problem: &DataProblem,
    svd: &(dyn ThinSvd + Sync),
) -> Result<DropOneReport, TrfError> {
    problem.kind.validate(&problem.x0)?;
    solve_drop_one_with(&problem.kind, &problem.x0, svd, &problem.options())
}
