//! A pure-Rust `ThinSvd` backed by `nalgebra`, shared by the benchmarks and the
//! validation tests.
//!
//! This is the "native" path a real Rust user gets without any Python/LAPACK
//! injection: only the thin SVD is provided here; every other `ThinSvd` hook
//! (`dot`, `fortran_matvec`, `row_major_matvec`, `power3`) falls back to the
//! crate's own pure-Rust reductions. It is intentionally NOT bit-exact with
//! SciPy — it is a legitimate independent SVD — so it must never be used for the
//! bit-exact fixtures; it exists to exercise and time the native solve.
//!
//! This file is pulled in via `#[path = ...] mod nalgebra_svd;` rather than being
//! its own test binary (it lives under `tests/support/`, which Cargo does not
//! compile as a test target).

#![allow(dead_code)]

use nalgebra::DMatrix;
use trust_region_least_squares::trf::{SvdError, ThinSvd};

/// Thin SVD (`full_matrices=False`) via `nalgebra`, returning row-major `U`
/// (`m`-by-`n`), the `n` singular values in descending order, and row-major
/// `VT` (`n`-by-`n`).
#[derive(Debug, Default, Clone, Copy)]
pub struct NalgebraSvd;

impl ThinSvd for NalgebraSvd {
    fn svd(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), SvdError> {
        let expected = m
            .checked_mul(n)
            .ok_or_else(|| SvdError::Failed(format!("m*n overflows usize: m={m}, n={n}")))?;
        if a.len() != expected {
            return Err(SvdError::BadDimensions {
                expected_m: m,
                expected_n: n,
                got: a.len(),
            });
        }
        if n == 0 || m < n {
            return Err(SvdError::BadDimensions {
                expected_m: m,
                expected_n: n,
                got: a.len(),
            });
        }

        let mat = DMatrix::<f64>::from_row_slice(m, n, a);
        let svd = mat.svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| SvdError::Failed("nalgebra returned no U".to_string()))?;
        let vt = svd
            .v_t
            .ok_or_else(|| SvdError::Failed("nalgebra returned no V_t".to_string()))?;
        let s = svd.singular_values;

        // `nalgebra`'s thin SVD gives `U` as m-by-min(m,n) = m-by-n (m >= n).
        let mut u_row_major = vec![0.0; expected];
        for i in 0..m {
            for j in 0..n {
                u_row_major[i * n + j] = u[(i, j)];
            }
        }
        let mut vt_row_major = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                vt_row_major[i * n + j] = vt[(i, j)];
            }
        }
        let s_vec: Vec<f64> = s.iter().copied().collect();
        Ok((u_row_major, s_vec, vt_row_major))
    }
}
