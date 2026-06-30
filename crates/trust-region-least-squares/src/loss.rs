//! SciPy robust loss functions, reproduced bit-for-bit.
//!
//! This module mirrors the exact float sequence SciPy 1.18.0 uses for robust
//! (non-`linear`) losses, so the trust-region iteration in [`crate::trf`] can
//! reproduce `scipy.optimize.least_squares(..., loss=..., f_scale=...)` down to
//! the last bit. Three SciPy source locations are mirrored verbatim:
//!
//! - `scipy/optimize/_lsq/least_squares.py`: the `IMPLEMENTED_LOSSES` rho
//!   functions (`huber`, `soft_l1`, `cauchy`, `arctan`) and
//!   `construct_loss_function` (the `z = (f / f_scale) ** 2` substitution, the
//!   `rho[0] *= f_scale**2` / `rho[2] /= f_scale**2` rescaling, and the
//!   `cost_only` shortcut `0.5 * f_scale**2 * sum(rho[0])`).
//! - `scipy/optimize/_lsq/common.py`: `scale_for_robust_loss_function`, the
//!   IRLS reweighting `J_scale = rho[1] + 2*rho[2]*f**2`, the `< EPS` floor, the
//!   `J_scale **= 0.5`, the `f *= rho[1] / J_scale`, and the per-row
//!   `left_multiply(J, J_scale)`.
//! - `numpy.sum`'s pairwise summation, reproduced in [`pairwise_sum`] so the
//!   robust cost `0.5 * sum(rho[0])` is bit-identical.
//!
//! The float primitives were verified bit-exact against the pinned NumPy 2.5.0
//! / SciPy 1.18.0 runtime on this target: `**2` is `x*x`, `**0.5` is `sqrt`,
//! `**-0.5` / `**-1.5` are libm `pow` (== Rust [`f64::powf`]), `log1p` ==
//! [`f64::ln_1p`], `arctan` == [`f64::atan`].

/// `numpy.finfo(float).eps`, used as the `scale_for_robust_loss_function` floor.
const EPS: f64 = f64::EPSILON;

/// Errors from the robust-loss reweighting helper when its slice arguments are
/// inconsistent, so the public API returns a typed error rather than panicking
/// on an out-of-bounds index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LossError {
    /// A slice argument had a length inconsistent with `m = f.len()` / `n`.
    LengthMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

impl std::fmt::Display for LossError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LossError::LengthMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what} has length {got}, expected {expected}"),
        }
    }
}

impl std::error::Error for LossError {}

/// SciPy's `loss` selector for `least_squares`.
///
/// `Linear` is the identity loss (`rho(z) = z`); the others are SciPy's robust
/// `IMPLEMENTED_LOSSES`. With `Linear`, no reweighting is applied and the
/// solver follows its ordinary least-squares trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Loss {
    #[default]
    Linear,
    SoftL1,
    Huber,
    Cauchy,
    Arctan,
}

/// The `(3, m)` rho array SciPy materializes for a robust loss: `rho0 = rho(z)`,
/// `rho1 = rho'(z)`, `rho2 = rho''(z)` (after `construct_loss_function`'s
/// `f_scale` rescaling of `rho0`/`rho2`).
#[derive(Debug, Clone, PartialEq)]
pub struct Rho {
    pub rho0: Vec<f64>,
    pub rho1: Vec<f64>,
    pub rho2: Vec<f64>,
}

/// SciPy's constructed `loss_function`, carrying `loss` and `f_scale`.
#[derive(Debug, Clone, Copy)]
pub struct LossFunction {
    pub loss: Loss,
    pub f_scale: f64,
}

impl LossFunction {
    pub fn new(loss: Loss, f_scale: f64) -> Self {
        Self { loss, f_scale }
    }

    /// Mirrors `construct_loss_function`'s full path (`cost_only=False`):
    /// `z = (f / f_scale)**2`, fill `rho`, then `rho[0] *= f_scale**2` and
    /// `rho[2] /= f_scale**2`.
    pub fn evaluate(&self, f: &[f64]) -> Rho {
        let mut rho = self.eval_z_rho(f, false);
        let fs2 = self.f_scale * self.f_scale;
        for value in &mut rho.rho0 {
            *value *= fs2;
        }
        for value in &mut rho.rho2 {
            *value /= fs2;
        }
        rho
    }

    /// Mirrors `construct_loss_function`'s `cost_only=True` path:
    /// `0.5 * f_scale**2 * sum(rho[0])` over the unscaled `rho[0]`.
    pub fn cost_only(&self, f: &[f64]) -> f64 {
        let rho = self.eval_z_rho(f, true);
        let fs2 = self.f_scale * self.f_scale;
        0.5 * fs2 * pairwise_sum(&rho.rho0)
    }

    /// Computes `z = (f / f_scale)**2` then dispatches to the rho function. With
    /// `cost_only`, only `rho0` is meaningful (matching SciPy's early return);
    /// `rho1`/`rho2` are left zero-filled.
    fn eval_z_rho(&self, f: &[f64], cost_only: bool) -> Rho {
        let fs = self.f_scale;
        let z: Vec<f64> = f
            .iter()
            .map(|&fi| {
                let d = fi / fs;
                d * d
            })
            .collect();
        rho_for_loss(self.loss, &z, cost_only)
    }
}

/// Fills `(rho0, rho1, rho2)` for `loss` over `z`, mirroring the corresponding
/// `IMPLEMENTED_LOSSES` entry. `Linear` has no rho entry in SciPy; it is
/// included here as the identity (`rho0=z, rho1=1, rho2=0`) for completeness and
/// is never used by the robust path.
pub fn rho_for_loss(loss: Loss, z: &[f64], cost_only: bool) -> Rho {
    let m = z.len();
    let mut rho0 = vec![0.0; m];
    let mut rho1 = vec![0.0; m];
    let mut rho2 = vec![0.0; m];

    match loss {
        Loss::Linear => {
            for i in 0..m {
                rho0[i] = z[i];
                if cost_only {
                    continue;
                }
                rho1[i] = 1.0;
                rho2[i] = 0.0;
            }
        }
        // huber(z, rho, cost_only): mask = z <= 1.
        Loss::Huber => {
            for i in 0..m {
                let zi = z[i];
                if zi <= 1.0 {
                    rho0[i] = zi;
                    if cost_only {
                        continue;
                    }
                    rho1[i] = 1.0;
                    rho2[i] = 0.0;
                } else {
                    // rho[0] = 2 * z**0.5 - 1
                    rho0[i] = 2.0 * zi.sqrt() - 1.0;
                    if cost_only {
                        continue;
                    }
                    // rho[1] = z**-0.5 ; rho[2] = -0.5 * z**-1.5
                    rho1[i] = zi.powf(-0.5);
                    rho2[i] = -0.5 * zi.powf(-1.5);
                }
            }
        }
        // soft_l1(z, rho, cost_only): t = 1 + z.
        Loss::SoftL1 => {
            for i in 0..m {
                let t = 1.0 + z[i];
                // rho[0] = 2 * (t**0.5 - 1)
                rho0[i] = 2.0 * (t.sqrt() - 1.0);
                if cost_only {
                    continue;
                }
                // rho[1] = t**-0.5 ; rho[2] = -0.5 * t**-1.5
                rho1[i] = t.powf(-0.5);
                rho2[i] = -0.5 * t.powf(-1.5);
            }
        }
        // cauchy(z, rho, cost_only): rho[0] = log1p(z).
        Loss::Cauchy => {
            for i in 0..m {
                rho0[i] = z[i].ln_1p();
                if cost_only {
                    continue;
                }
                let t = 1.0 + z[i];
                // rho[1] = 1 / t ; rho[2] = -1 / t**2
                rho1[i] = 1.0 / t;
                rho2[i] = -1.0 / (t * t);
            }
        }
        // arctan(z, rho, cost_only): rho[0] = arctan(z).
        Loss::Arctan => {
            for i in 0..m {
                let zi = z[i];
                rho0[i] = zi.atan();
                if cost_only {
                    continue;
                }
                // t = 1 + z**2 ; rho[1] = 1 / t ; rho[2] = -2 * z / t**2
                let t = 1.0 + zi * zi;
                rho1[i] = 1.0 / t;
                rho2[i] = -2.0 * zi / (t * t);
            }
        }
    }

    Rho { rho0, rho1, rho2 }
}

/// Mirrors `common.scale_for_robust_loss_function`, modifying the residual `f`
/// and the row-major `m`-by-`n` Jacobian `jac` in place.
///
/// ```text
/// J_scale = rho[1] + 2 * rho[2] * f**2
/// J_scale[J_scale < EPS] = EPS
/// J_scale **= 0.5
/// f *= rho[1] / J_scale
/// J = left_multiply(J, J_scale)   # row i scaled by J_scale[i]
/// ```
///
/// # Errors
///
/// Returns [`LossError::LengthMismatch`] if `rho.rho1`/`rho.rho2` are not length
/// `f.len()`, or if `jac.len()` is not `f.len() * n` (`m * n` overflow included).
pub fn scale_for_robust_loss(
    jac: &mut [f64],
    f: &mut [f64],
    rho: &Rho,
    n: usize,
) -> Result<(), LossError> {
    let m = f.len();
    let check = |what: &'static str, got: usize, expected: usize| {
        if got == expected {
            Ok(())
        } else {
            Err(LossError::LengthMismatch {
                what,
                expected,
                got,
            })
        }
    };
    check("rho1", rho.rho1.len(), m)?;
    check("rho2", rho.rho2.len(), m)?;
    let mn = m.checked_mul(n).ok_or(LossError::LengthMismatch {
        what: "jac",
        expected: usize::MAX,
        got: jac.len(),
    })?;
    check("jac", jac.len(), mn)?;

    for (i, fi) in f.iter_mut().enumerate() {
        // J_scale = rho[1] + 2 * rho[2] * f**2, then floor at EPS, then sqrt.
        let mut j_scale = rho.rho1[i] + 2.0 * rho.rho2[i] * (*fi * *fi);
        if j_scale < EPS {
            j_scale = EPS;
        }
        j_scale = j_scale.sqrt();

        // f *= rho[1] / J_scale (the residual update uses the pre-scaled f).
        *fi *= rho.rho1[i] / j_scale;

        // left_multiply: scale row i of J by J_scale[i].
        let row = i * n;
        for value in &mut jac[row..row + n] {
            *value *= j_scale;
        }
    }
    Ok(())
}

/// Reproduces `numpy.sum` over a contiguous f64 slice (C order, default
/// `PW_BLOCKSIZE = 128` pairwise summation) so reductions are bit-identical.
pub fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n == 0 {
        return 0.0;
    }
    if n < 8 {
        let mut res = 0.0;
        for &value in a {
            res += value;
        }
        return res;
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        let limit = n - (n % 8);
        while i < limit {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}
