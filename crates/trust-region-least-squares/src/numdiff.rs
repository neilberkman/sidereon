//! Dense 2-point finite differences matching SciPy 1.11 `_numdiff`.
//!
//! This is the narrow path used by `scipy.optimize.least_squares` when it
//! calls `approx_derivative(..., method="2-point", rel_step=None, f0=f,
//! bounds=(-inf, inf), sparsity=None)`.

use std::error::Error;
use std::fmt;

const EPS_FOR_2POINT_F64: f64 = 1.490_116_119_384_765_6e-8;

#[derive(Clone, Debug)]
pub struct ApproxDerivative2Point {
    pub m: usize,
    pub n: usize,
    /// Row-major `m x n` dense Jacobian, matching `np.atleast_2d(J)`.
    pub jacobian: Vec<f64>,
    /// The perturbed `x` vectors evaluated by SciPy, in column order.
    pub evaluation_points: Vec<Vec<f64>>,
    /// The requested absolute steps after unbounded scheme adjustment.
    pub h: Vec<f64>,
    /// Returned by SciPy's `_adjust_scheme_to_bounds`; true for this path.
    pub use_one_sided: Vec<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NumDiffError {
    OutputLength { expected: usize, actual: usize },
}

impl fmt::Display for NumDiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumDiffError::OutputLength { expected, actual } => write!(
                f,
                "finite-difference callback returned {actual} values, expected {expected}"
            ),
        }
    }
}

impl Error for NumDiffError {}

fn sign_x0(x: f64) -> f64 {
    if x >= 0.0 {
        1.0
    } else {
        -1.0
    }
}

fn numpy_maximum(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        a
    } else if b.is_nan() || b > a {
        b
    } else {
        a
    }
}

/// Port of SciPy `_compute_absolute_step(None, x0, f0, "2-point")` for f64.
///
/// The `f0` argument is present to mirror SciPy's signature. In this port both
/// `x0` and `f0` are f64, so `_eps_for_method` always selects f64 epsilon.
pub fn compute_absolute_step_2point(x0: &[f64], _f0: &[f64]) -> Vec<f64> {
    let mut h = Vec::with_capacity(x0.len());
    for &x in x0 {
        let signed = EPS_FOR_2POINT_F64 * sign_x0(x);
        let scale = numpy_maximum(1.0, x.abs());
        h.push(signed * scale);
    }
    h
}

/// Port of SciPy `_adjust_scheme_to_bounds` for the unbounded 1-sided path.
pub fn adjust_scheme_to_bounds_unbounded_1sided(h: &[f64]) -> (Vec<f64>, Vec<bool>) {
    (h.to_vec(), vec![true; h.len()])
}

/// Dense SciPy-compatible 2-point Jacobian.
///
/// The callback receives the candidate `x` slice and writes the residual vector
/// into the supplied output buffer.
pub fn approx_derivative_2point<F>(
    x0: &[f64],
    f0: &[f64],
    mut fun: F,
) -> Result<ApproxDerivative2Point, NumDiffError>
where
    F: FnMut(&[f64], &mut Vec<f64>),
{
    let n = x0.len();
    let m = f0.len();
    let h0 = compute_absolute_step_2point(x0, f0);
    let (h, use_one_sided) = adjust_scheme_to_bounds_unbounded_1sided(&h0);

    let mut jacobian = vec![0.0; m * n];
    let mut evaluation_points = Vec::with_capacity(n);
    let mut f_eval = Vec::new();

    for j in 0..n {
        let mut x = Vec::with_capacity(n);
        for (k, &x0k) in x0.iter().enumerate() {
            let step = if k == j { h[j] } else { 0.0 };
            x.push(x0k + step);
        }
        let dx = x[j] - x0[j];

        f_eval.clear();
        fun(&x, &mut f_eval);
        if f_eval.len() != m {
            return Err(NumDiffError::OutputLength {
                expected: m,
                actual: f_eval.len(),
            });
        }

        evaluation_points.push(x);
        for i in 0..m {
            jacobian[i * n + j] = (f_eval[i] - f0[i]) / dx;
        }
    }

    Ok(ApproxDerivative2Point {
        m,
        n,
        jacobian,
        evaluation_points,
        h,
        use_one_sided,
    })
}
