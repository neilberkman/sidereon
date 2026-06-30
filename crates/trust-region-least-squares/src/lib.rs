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
//!   [`trf::ThinSvd`] SVD/BLAS seam and the `loss`/`f_scale` robust path.
//! - [`loss`]: SciPy's robust loss functions (`construct_loss_function` +
//!   `IMPLEMENTED_LOSSES`) and `scale_for_robust_loss_function`, reproduced
//!   bit-for-bit.
//! - [`numdiff`]: the dense two-point finite-difference Jacobian matching
//!   SciPy's `_numdiff.approx_derivative(..., method="2-point")` path.
//! - [`parity`]: hex-bit fixture helpers, feature-gated trace output, and
//!   first-divergence reporting for diagnosing where two trajectories split.
//! - `hostlapack` (feature `host-lapack`): a [`trf::ThinSvd`] implementation
//!   backed by a dynamically loaded host LAPACK/BLAS, used to reproduce a
//!   pinned SciPy runtime's exact SVD/BLAS results.
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
//! Give the solver a residual `r: Rⁿ → Rᵐ`, a Jacobian, a starting point, and a
//! [`trf::ThinSvd`] backend. The thin SVD is *injected*, so you can plug in any
//! native Rust linear-algebra library — no Python required. This example wires
//! [`nalgebra`](https://docs.rs/nalgebra) as the SVD seam and fits the
//! overdetermined system whose least-squares solution is `[1.0, 2.0]`:
//!
//! ```
//! use nalgebra::DMatrix;
//! use trust_region_least_squares::trf::{
//!     jacobian_2point, trf_no_bounds, JacobianFn, ResidualFn, SvdError, ThinSvd, TrfOptions,
//! };
//!
//! // A pure-Rust thin SVD (equivalent to full_matrices=False) backing the seam.
//! struct NalgebraSvd;
//! impl ThinSvd for NalgebraSvd {
//!     fn svd(&self, a: &[f64], m: usize, n: usize)
//!         -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError>
//!     {
//!         let svd = DMatrix::from_row_slice(m, n, a).svd(true, true);
//!         let u = svd.u.ok_or_else(|| SvdError::Failed("no U".into()))?;
//!         let vt = svd.v_t.ok_or_else(|| SvdError::Failed("no V_t".into()))?;
//!         // Re-pack column-major nalgebra matrices as row-major U (m×n) and VT (n×n).
//!         let mut u_rm = vec![0.0; m * n];
//!         for i in 0..m { for j in 0..n { u_rm[i * n + j] = u[(i, j)]; } }
//!         let mut vt_rm = vec![0.0; n * n];
//!         for i in 0..n { for j in 0..n { vt_rm[i * n + j] = vt[(i, j)]; } }
//!         Ok((u_rm, svd.singular_values.iter().copied().collect(), vt_rm))
//!     }
//! }
//!
//! // r(x) = [x0 - 1, x1 - 2, x0 + x1 - 3]; minimized at x = [1, 2].
//! fn residual(x: &[f64], out: &mut Vec<f64>) {
//!     out.clear();
//!     out.push(x[0] - 1.0);
//!     out.push(x[1] - 2.0);
//!     out.push(x[0] + x[1] - 3.0);
//! }
//!
//! let mut fun = residual;
//! let mut jac = |x: &[f64], f0: &[f64], out: &mut Vec<f64>| {
//!     let mut scratch = Vec::new();
//!     let mut inner = residual;
//!     jacobian_2point(&mut inner, x, f0, out, &mut scratch).unwrap();
//! };
//!
//! let result = trf_no_bounds(
//!     &mut fun as &mut ResidualFn<'_>,
//!     &mut jac as &mut JacobianFn<'_>,
//!     &[0.0, 0.0],
//!     &NalgebraSvd,
//!     &TrfOptions::default(),
//! )
//! .expect("solve");
//!
//! assert!(result.success());
//! assert!((result.x[0] - 1.0).abs() < 1e-6);
//! assert!((result.x[1] - 2.0).abs() < 1e-6);
//! ```
//!
//! For bit-for-bit agreement with a pinned SciPy/NumPy runtime, inject a
//! host-LAPACK backend instead (the optional `host-lapack` feature's
//! `hostlapack::LapackSvd`); the iteration is identical, only the SVD/BLAS seam
//! changes.
//!
//! [`scipy.optimize.least_squares`]: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

pub mod loss;
pub mod numdiff;
pub mod parity;
pub mod trf;

#[cfg(feature = "host-lapack")]
pub mod hostlapack;
