//! 3x3 matrix operations with exact floating-point rounding control.
//!
//! These routines are ported from the C++ inline_rxr, inline_tr, and
//! inline_mxmxm functions which were compiled under `fp-contract=off`
//! to match Python/Skyfield arithmetic.  Rust's default arithmetic
//! operators do not fuse multiply-adds, so no special annotation is
//! needed -- just avoid `f64::mul_add`.

/// A row-major 3x3 matrix.
pub type Mat3 = [[f64; 3]; 3];

/// Standard matrix multiply: `result = a * b`.
pub fn inline_rxr(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut w = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0_f64;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            w[i][j] = s;
        }
    }
    w
}

/// Matrix transpose: `result = r^T`.
pub fn inline_tr(r: &Mat3) -> Mat3 {
    let mut rt = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            rt[i][j] = r[j][i];
        }
    }
    rt
}

/// Triple matrix product with Kahan-compensated summation.
///
/// Computes `result = A * B * C` without materialising the intermediate
/// A*B matrix.  Each output element accumulates all 9 terms
/// (matching numpy `einsum('ij,jk,kl->il')`) using Kahan summation to
/// reduce floating-point drift.
pub fn inline_mxmxm(a: &Mat3, b: &Mat3, c: &Mat3) -> Mat3 {
    let mut w = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for l in 0..3 {
            let mut s = 0.0_f64;
            let mut comp = 0.0_f64; // Kahan compensation
            for j in 0..3 {
                for k in 0..3 {
                    let term = a[i][j] * b[j][k] * c[k][l];
                    let y = term - comp;
                    let t = s + y;
                    comp = (t - s) - y;
                    s = t;
                }
            }
            w[i][l] = s;
        }
    }
    w
}
