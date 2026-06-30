//! A dense trust-region-reflective (TRF) nonlinear least-squares solver that
//! reproduces [`scipy.optimize.least_squares`] (`method='trf'`, 2-point
//! Jacobian) bit-for-bit. It covers SciPy's dense unbounded path for arbitrary
//! problem dimension `n` and every SciPy loss (`linear`, `soft_l1`, `huber`,
//! `cauchy`, `arctan`) with the `f_scale` robust-reweighting parameter; see
//! [Scope](#scope) for the supported dimensionality.
//!
//! It is built to be a general-purpose solver, not a one-off port: give it a
//! residual `r: R^n -> R^m` and a starting point and it runs the same
//! trust-region Newton iteration SciPy does, down to the last bit. The
//! linear-algebra
//! operations that determine the last bits of the trajectory -- the thin SVD of
//! the scaled Jacobian and the small BLAS reductions around it -- are *injected*
//! through the [`trf::ThinSvd`] trait. Supplying an implementation backed by a
//! host LAPACK/BLAS (see the optional `hostlapack` module) lets the solver
//! reproduce that backend's numerical trajectory exactly, which is what makes
//! bit-for-bit agreement with a pinned SciPy/NumPy runtime achievable rather
//! than merely tolerance-close.
//!
//! Use it when you need scipy-identical least-squares results in Rust: porting a
//! Python pipeline, cross-checking a Rust solver against SciPy, or pinning a
//! numerical result so it cannot silently drift between language runtimes.
//!
//! # Scope
//!
//! - [`trf`]: the dense unbounded trust-region-reflective iteration matching
//!   `scipy.optimize._lsq.trf.trf_no_bounds`, with the injectable
//!   [`trf::ThinSvd`] SVD/BLAS seam and the `loss`/`f_scale` robust path. Ships
//!   the default in-crate [`trf::NalgebraThinSvd`] backend and the convenience
//!   [`trf::trf_solve`] entry point so the solver runs out of the box.
//! - [`model`]: the [`model::ResidualModel`] trait and
//!   [`model::solve_model`], so a residual (and any per-evaluation corrections)
//!   can live entirely Rust-side instead of behind a closure boundary.
//! - [`data`]: data-driven [`data::BuiltinResidual`] kinds and
//!   [`data::DataProblem`] / [`data::solve_data_problem`]: a host-language
//!   binding selects a kind, passes data arrays, and the trust-region loop
//!   never re-enters the host language.
//! - [`batch`]: parallel leave-one-out ([`batch::solve_drop_one`]) and
//!   multi-start ([`batch::solve_perturbed`]) re-solves over a `rayon` pool,
//!   with per-index results bit-identical to the equivalent serial solve.
//!   Serial leave-one-out twins ([`batch::solve_drop_one_serial`] and friends)
//!   give single-threaded / `wasm32` consumers the same report without rayon.
//! - [`loss`]: SciPy's robust loss functions (`construct_loss_function` +
//!   `IMPLEMENTED_LOSSES`) and `scale_for_robust_loss_function`, reproduced
//!   bit-for-bit.
//! - [`numdiff`]: the dense two-point finite-difference Jacobian matching
//!   SciPy's `_numdiff.approx_derivative(..., method="2-point")` path.
//! - [`parity`]: hex-bit fixture helpers, feature-gated trace output, and
//!   first-divergence reporting for diagnosing where two trajectories split.
//! - [`hostlapack`]: a [`trf::ThinSvd`] implementation backed by a dynamically
//!   loaded host LAPACK/BLAS, used to reproduce a pinned SciPy runtime's exact
//!   SVD/BLAS results bit-for-bit. Compiled into the default build; point it at
//!   the host LAPACK/BLAS at runtime to get bit-exact parity.
//!
//! The iteration is general in `n`: give it a residual `r: R^n -> R^m` for any
//! `n >= 1` (with `m >= n` for the dense exact trust-region solve) and it runs
//! SciPy's `trf_no_bounds` trajectory bit-for-bit. All five SciPy losses are
//! supported at every dimension. Bit-exact parity is enforced by committed
//! fixtures across `n` in {2, 3, 4, 5, 6, 8} times each loss; see the
//! `fixtures-generators` directory and the `tests` suite.
//!
//! # Example
//!
//! The solver runs out of the box: hand [`model::solve_model`] a residual model
//! and a starting point and it drives the trust-region iteration through the
//! default in-crate SVD. This fits the overdetermined system whose
//! least-squares solution is `[1.0, 2.0]`:
//!
//! ```
//! use trust_region_least_squares::model::{solve_model, ResidualModel};
//! use trust_region_least_squares::trf::TrfOptions;
//!
//! // r(x) = [x0 - 1, x1 - 2, x0 + x1 - 3]; minimized at x = [1, 2].
//! struct Consistent;
//! impl ResidualModel for Consistent {
//!     fn residual(&self, x: &[f64], out: &mut Vec<f64>) {
//!         out.clear();
//!         out.push(x[0] - 1.0);
//!         out.push(x[1] - 2.0);
//!         out.push(x[0] + x[1] - 3.0);
//!     }
//!     // The Jacobian defaults to the bit-exact 2-point finite difference.
//! }
//!
//! let result = solve_model(&Consistent, &[0.0, 0.0], &TrfOptions::default())
//!     .expect("solve");
//!
//! assert!(result.success());
//! assert!((result.x[0] - 1.0).abs() < 1e-6);
//! assert!((result.x[1] - 2.0).abs() < 1e-6);
//! ```
//!
//! The thin SVD is still an injectable seam. The default
//! [`trf::NalgebraThinSvd`] is a legitimate independent SVD but is *not*
//! bit-identical to a pinned SciPy/NumPy runtime; for bit-for-bit parity inject
//! the [`hostlapack::LapackSvd`] backend (the `*_with` entry points take a
//! [`trf::ThinSvd`]). The iteration is identical, only the SVD/BLAS seam
//! changes.
//!
//! [`scipy.optimize.least_squares`]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

pub mod batch;
pub mod data;
// `hostlapack` loads a host LAPACK/BLAS at runtime through `libloading`, which
// has no wasm32 backend (no `dlopen`), so it cannot compile for wasm targets.
// Gate it off there; wasm consumers use the default in-crate `NalgebraThinSvd`
// backend. Every non-wasm target keeps the bit-exact LAPACK seam.
#[cfg(not(target_arch = "wasm32"))]
pub mod hostlapack;
pub mod loss;
pub mod model;
pub mod numdiff;
pub mod parity;
pub mod trf;
