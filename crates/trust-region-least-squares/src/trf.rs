//! Dense unbounded trust-region reflective least-squares matching
//! `scipy.optimize._lsq.trf.trf_no_bounds` for arbitrary problem dimension `n`.
//!
//! The thin SVD, BLAS reductions, and selected power operations are injected
//! through [`HostNumerics`] so a pinned SciPy/NumPy runtime can provide the
//! exact numerical trajectory when required.

#[cfg(feature = "trace")]
use crate::parity::{TraceValue, TraceWriter};

use crate::loss::{pairwise_sum, scale_for_robust_loss, Loss, LossError, LossFunction};

const EPS: f64 = f64::EPSILON;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendError {
    Failed(String),
    BadDimensions {
        expected_m: usize,
        expected_n: usize,
        got: usize,
    },
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Failed(message) => {
                write!(f, "host numerics backend failed: {message}")
            }
            BackendError::BadDimensions {
                expected_m,
                expected_n,
                got,
            } => write!(
                f,
                "SVD input has length {got}, expected {expected_m}x{expected_n} = {}",
                expected_m.saturating_mul(*expected_n)
            ),
        }
    }
}

impl std::error::Error for BackendError {}

/// Host-runtime numerical operations that can affect bit-exact SciPy parity.
///
/// The required [`HostNumerics::svd`] operation supplies the thin SVD. The
/// defaulted hooks supply BLAS reductions, the NumPy power dispatch, and the
/// loss transcendental operations used at the specific solver sites where the
/// pinned runtime's last bits can differ from Rust arithmetic. `Ok(None)`
/// declines a defaulted hook and keeps the crate's pure-Rust computation.
pub trait HostNumerics {
    /// Return row-major U, singular values, and row-major VT for a row-major
    /// m-by-n input matrix, equivalent to scipy.linalg.svd(..., full_matrices=False).
    #[allow(clippy::type_complexity)]
    fn svd(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), BackendError>;

    fn dot(&self, _a: &[f64], _b: &[f64]) -> Result<Option<f64>, BackendError> {
        Ok(None)
    }

    fn fortran_matvec(
        &self,
        _a: &[f64],
        _m: usize,
        _n: usize,
        _x: &[f64],
        _transpose: bool,
    ) -> Result<Option<Vec<f64>>, BackendError> {
        Ok(None)
    }

    fn row_major_matvec(
        &self,
        _a: &[f64],
        _m: usize,
        _n: usize,
        _x: &[f64],
        _transpose: bool,
    ) -> Result<Option<Vec<f64>>, BackendError> {
        Ok(None)
    }

    /// Elementwise power mirroring `np.power(values, scalar_exponent)` over a
    /// contiguous f64 array: NumPy's stride-0 ufunc dispatch.
    ///
    /// Before selecting an SVML or `npy_pow` kernel, that dispatch applies this
    /// table to the scalar exponent:
    ///
    /// - `-1.0`: `1.0 / value`
    /// - `0.0`: literal `1.0`
    /// - `0.5`: `sqrt(value)`
    /// - `1.0`: `value`
    /// - `2.0`: `value * value`
    ///
    /// Every other exponent falls through to the per-CPU kernel. The three
    /// exponents routed by the solver (`-0.5`, `-1.5`, and `3.0`) all take that
    /// fall-through path. At the robust-loss sites, SciPy's ndarray `**`
    /// operation reaches the ufunc without an outer shortcut, then the scalar
    /// exponent misses this inner-loop table.
    ///
    /// `Ok(None)` declines, and the solver keeps its own [`f64::powf`]. A
    /// supplied `Some(values)` **must** have the same length as `values`; the
    /// solver rejects any other length with a typed error rather than
    /// iterating on a truncated array.
    ///
    /// The solver also reaches this hook with exponent `3.0` for
    /// `phi_and_derivative`'s `denom ** 3` expression.
    fn power(&self, _values: &[f64], _exponent: f64) -> Result<Option<Vec<f64>>, BackendError> {
        Ok(None)
    }

    /// Scalar `base ** exponent`, mirroring the scalar power of the reference
    /// runtime. NumPy scalars and Python floats both take the platform C `pow`;
    /// they do not enter the array ufunc's stride-0 table.
    ///
    /// The solver reaches this for the trust-region alpha seed, which SciPy
    /// writes as `(alpha_lower * alpha_upper) ** 0.5`. `Ok(None)` declines, and
    /// the solver keeps its own square root.
    fn power_scalar(&self, _base: f64, _exponent: f64) -> Result<Option<f64>, BackendError> {
        Ok(None)
    }

    /// Scalar natural logarithm of one plus `x` for the Cauchy loss.
    ///
    /// `Ok(None)` declines and leaves the crate's published `f64::ln_1p`
    /// calculation in place.
    fn log1p(&self, _x: f64) -> Result<Option<f64>, BackendError> {
        Ok(None)
    }

    /// Scalar arctangent for the Arctan loss.
    ///
    /// `Ok(None)` declines and leaves the crate's published `f64::atan`
    /// calculation in place.
    fn atan(&self, _x: f64) -> Result<Option<f64>, BackendError> {
        Ok(None)
    }
}

/// Pure-Rust thin-SVD backend built on `nalgebra`, the crate's default
/// [`HostNumerics`] seam used when no host LAPACK is configured.
///
/// It is a legitimate independent singular value decomposition but is **not**
/// bit-exact with `scipy.linalg.svd` / the host-LAPACK path, so it must never
/// back the bit-exact parity fixtures; it powers the native solve. The optional
/// BLAS and power hooks ([`HostNumerics::dot`], the matvecs,
/// [`HostNumerics::power`], [`HostNumerics::power_scalar`],
/// [`HostNumerics::log1p`], [`HostNumerics::atan`]) are left at their defaults
/// so the trust-region loop falls back to its own deterministic arithmetic.
#[derive(Debug, Clone, Copy, Default)]
pub struct NalgebraThinSvd;

impl HostNumerics for NalgebraThinSvd {
    fn svd(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), BackendError> {
        if a.len() != m.saturating_mul(n) {
            return Err(BackendError::BadDimensions {
                expected_m: m,
                expected_n: n,
                got: a.len(),
            });
        }
        let k = m.min(n);
        let matrix = nalgebra::DMatrix::<f64>::from_row_slice(m, n, a);
        let svd = matrix.svd(true, true);
        let u = svd
            .u
            .ok_or_else(|| BackendError::Failed("nalgebra SVD did not produce U".to_string()))?;
        let vt = svd
            .v_t
            .ok_or_else(|| BackendError::Failed("nalgebra SVD did not produce V^T".to_string()))?;

        // Row-major thin U (m x k), matching scipy.linalg.svd(full_matrices=False).
        let mut u_out = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                u_out[i * k + j] = u[(i, j)];
            }
        }
        let s_out = svd.singular_values.as_slice().to_vec();
        // Row-major thin V^T (k x n).
        let mut vt_out = vec![0.0; k * n];
        for i in 0..k {
            for j in 0..n {
                vt_out[i * n + j] = vt[(i, j)];
            }
        }
        Ok((u_out, s_out, vt_out))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TrfError {
    /// The residual vector was empty (`m == 0`).
    EmptyResidual,
    /// The parameter vector `x0` was empty (`n == 0`).
    EmptyParameters,
    /// `x0` contained a non-finite (NaN or infinite) entry. SciPy rejects such
    /// starting points before iterating.
    NonFiniteParameters,
    /// The residual at the initial point contained a non-finite entry. Mirrors
    /// SciPy's "Residuals are not finite in the initial point." error.
    NonFiniteInitialResidual,
    /// `m < n`: this dense exact trust-region solve requires at least as many
    /// residuals as parameters (the thin SVD produces `n` singular values).
    InsufficientRows { m: usize, n: usize },
    /// `m * n` overflowed `usize`; the problem is too large to allocate.
    SizeOverflow { m: usize, n: usize },
    /// A polynomial `degree` was so large that the coefficient count
    /// `degree + 1` overflows `usize`; the problem cannot be represented.
    DegreeOverflow { degree: usize },
    /// `max_nfev` was `Some(0)`; SciPy requires a positive evaluation budget
    /// (`None` selects the default `100 * n`).
    InvalidMaxNfev,
    /// A robust loss was requested with an `f_scale` that is not finite and
    /// strictly positive. SciPy divides residuals by `f_scale`, so a zero or
    /// non-finite value yields NaN/Inf; this surfaces it as a typed error.
    InvalidFScale { f_scale: f64 },
    /// `XScale::Values` had the wrong length for the problem.
    InvalidXScaleLength { expected: usize, got: usize },
    /// `XScale::Values` contained a non-finite or non-positive entry; the
    /// per-parameter scale must be a positive finite number.
    InvalidXScaleValue { index: usize, value: f64 },
    /// The user Jacobian callback returned a buffer whose length is not `m * n`.
    InvalidJacobianLength { expected: usize, got: usize },
    /// The residual callback returned a vector whose length changed from the
    /// initial `m` on a later evaluation.
    InvalidResidualLength { expected: usize, got: usize },
    /// A slice argument to a public helper had the wrong length for the stated
    /// `m`/`n`.
    InvalidSliceLength {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// An injected [`HostNumerics`] returned a matrix/vector of unexpected shape.
    InvalidSvdOutput(String),
    /// The injected [`HostNumerics`] backend reported a failure.
    Backend(BackendError),
}

impl std::fmt::Display for TrfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrfError::EmptyResidual => write!(f, "residual vector is empty (m == 0)"),
            TrfError::EmptyParameters => write!(f, "parameter vector x0 is empty (n == 0)"),
            TrfError::NonFiniteParameters => write!(f, "x0 contains a non-finite (NaN/Inf) entry"),
            TrfError::NonFiniteInitialResidual => {
                write!(f, "residuals are not finite at the initial point")
            }
            TrfError::InsufficientRows { m, n } => write!(
                f,
                "m < n: dense exact trust-region requires m >= n, got m={m}, n={n}"
            ),
            TrfError::SizeOverflow { m, n } => {
                write!(f, "problem size m*n overflows usize: m={m}, n={n}")
            }
            TrfError::DegreeOverflow { degree } => {
                write!(
                    f,
                    "polynomial coefficient count degree+1 overflows usize: degree={degree}"
                )
            }
            TrfError::InvalidMaxNfev => {
                write!(
                    f,
                    "max_nfev must be None or a positive integer, got Some(0)"
                )
            }
            TrfError::InvalidFScale { f_scale } => write!(
                f,
                "f_scale must be finite and > 0 for a robust loss, got {f_scale:?}"
            ),
            TrfError::InvalidXScaleLength { expected, got } => {
                write!(f, "x_scale has length {got}, expected {expected}")
            }
            TrfError::InvalidXScaleValue { index, value } => {
                write!(f, "x_scale[{index}] must be finite and > 0, got {value:?}")
            }
            TrfError::InvalidJacobianLength { expected, got } => {
                write!(f, "Jacobian has length {got}, expected m*n = {expected}")
            }
            TrfError::InvalidResidualLength { expected, got } => write!(
                f,
                "residual callback returned length {got}, expected m = {expected}"
            ),
            TrfError::InvalidSliceLength {
                what,
                expected,
                got,
            } => write!(f, "{what} has length {got}, expected {expected}"),
            TrfError::InvalidSvdOutput(message) => write!(f, "invalid SVD output: {message}"),
            TrfError::Backend(err) => write!(f, "host numerics backend error: {err}"),
        }
    }
}

impl std::error::Error for TrfError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TrfError::Backend(err) => Some(err),
            _ => None,
        }
    }
}

impl From<BackendError> for TrfError {
    fn from(value: BackendError) -> Self {
        TrfError::Backend(value)
    }
}

impl From<LossError> for TrfError {
    fn from(value: LossError) -> Self {
        match value {
            LossError::LengthMismatch {
                what,
                expected,
                got,
            } => TrfError::InvalidSliceLength {
                what,
                expected,
                got,
            },
            LossError::HostPowerLength { expected, got } => TrfError::InvalidSliceLength {
                what: "host power result",
                expected,
                got,
            },
            LossError::Backend(err) => TrfError::Backend(err),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum XScale {
    Unit,
    Values(Vec<f64>),
    Jac,
}

#[derive(Debug, Clone)]
pub struct TrfOptions {
    pub ftol: f64,
    pub xtol: f64,
    pub gtol: f64,
    /// None matches scipy's default max_nfev = 100 * n.
    pub max_nfev: Option<usize>,
    /// scipy least_squares default is x_scale=1.0, represented by Unit.
    pub x_scale: XScale,
    /// scipy `loss`; default `Loss::Linear` reproduces the ordinary
    /// least-squares trajectory (no robust reweighting).
    pub loss: Loss,
    /// scipy `f_scale`; only consulted when `loss != Loss::Linear`.
    pub f_scale: f64,
}

impl Default for TrfOptions {
    fn default() -> Self {
        Self {
            ftol: 1e-8,
            xtol: 1e-8,
            gtol: 1e-10,
            max_nfev: None,
            x_scale: XScale::Unit,
            loss: Loss::Linear,
            f_scale: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TrfResult {
    /// Solution vector, length `n`.
    pub x: Vec<f64>,
    pub cost: f64,
    pub fun: Vec<f64>,
    /// Row-major m-by-n Jacobian at the returned point.
    pub jac: Vec<f64>,
    /// Gradient `J^T f`, length `n`.
    pub grad: Vec<f64>,
    pub optimality: f64,
    pub nfev: usize,
    pub njev: usize,
    /// SciPy-compatible status: 0 max evaluations, 1 gtol, 2 ftol, 3 xtol,
    /// 4 both ftol and xtol.
    pub status: i32,
}

impl TrfResult {
    pub fn success(&self) -> bool {
        self.status > 0
    }
}

pub type ResidualFn<'a> = dyn FnMut(&[f64], &mut Vec<f64>) + 'a;
pub type JacobianFn<'a> = dyn FnMut(&[f64], &[f64], &mut Vec<f64>) + 'a;

/// scipy.optimize._numdiff dense 2-point Jacobian for unbounded f64 input.
///
/// Function evaluations made here are intentionally not counted in TRF's nfev,
/// matching scipy.optimize.least_squares for method='trf', jac='2-point'.
///
/// `f0` defines the residual length `m = f0.len()` and `x0` defines `n =
/// x0.len()`; on return `jac` is the row-major `m`-by-`n` Jacobian. `scratch` is
/// a reusable buffer the residual closure is expected to fill with exactly `m`
/// values per call.
///
/// # Errors
///
/// Returns [`TrfError::SizeOverflow`] if `m * n` overflows `usize`, or
/// [`TrfError::InvalidResidualLength`] if the closure writes a number of values
/// other than `m` on any evaluation.
pub fn jacobian_2point(
    fun: &mut ResidualFn<'_>,
    x0: &[f64],
    f0: &[f64],
    jac: &mut Vec<f64>,
    scratch: &mut Vec<f64>,
) -> Result<(), TrfError> {
    let m = f0.len();
    let n = x0.len();
    let mn = m.checked_mul(n).ok_or(TrfError::SizeOverflow { m, n })?;
    jac.clear();
    jac.resize(mn, 0.0);

    for j in 0..n {
        let mut x = x0.to_vec();
        let h = finite_difference_step(x0[j]);
        x[j] += h;
        let dx = x[j] - x0[j];
        fun(&x, scratch);
        if scratch.len() != m {
            return Err(TrfError::InvalidResidualLength {
                expected: m,
                got: scratch.len(),
            });
        }

        for i in 0..m {
            jac[i * n + j] = (scratch[i] - f0[i]) / dx;
        }
    }
    Ok(())
}

pub fn trf_no_bounds(
    fun: &mut ResidualFn<'_>,
    jac: &mut JacobianFn<'_>,
    x0: &[f64],
    host: &dyn HostNumerics,
    options: &TrfOptions,
) -> Result<TrfResult, TrfError> {
    #[cfg(feature = "trace")]
    let mut trace = TraceWriter::from_env("trf");
    let n = x0.len();
    if n == 0 {
        return Err(TrfError::EmptyParameters);
    }
    if !all_finite(x0) {
        return Err(TrfError::NonFiniteParameters);
    }
    if options.max_nfev == Some(0) {
        return Err(TrfError::InvalidMaxNfev);
    }
    let robust = options.loss != Loss::Linear;
    // SciPy divides residuals by `f_scale` for robust losses; a zero or
    // non-finite scale produces NaN/Inf, so reject it up front (the `Linear`
    // path never consults `f_scale`, matching SciPy).
    if robust && !(options.f_scale.is_finite() && options.f_scale > 0.0) {
        return Err(TrfError::InvalidFScale {
            f_scale: options.f_scale,
        });
    }
    let mut x = x0.to_vec();
    let mut f = Vec::new();
    fun(&x, &mut f);
    #[cfg(feature = "trace")]
    trace.event(&[
        ("event", TraceValue::Str("initial_fun")),
        ("x", TraceValue::F64Slice(&x)),
        ("f", TraceValue::F64Slice(&f)),
    ]);
    if f.is_empty() {
        return Err(TrfError::EmptyResidual);
    }
    let m = f.len();
    if m < n {
        return Err(TrfError::InsufficientRows { m, n });
    }
    // Guard the `m * n` allocations below against `usize` overflow on
    // pathological sizes before any buffer is sized from it.
    let mn = m.checked_mul(n).ok_or(TrfError::SizeOverflow { m, n })?;
    if !all_finite(&f) {
        return Err(TrfError::NonFiniteInitialResidual);
    }
    let mut f_true = f.clone();
    let mut nfev = 1usize;

    let mut j_mat = Vec::new();
    jac(&x, &f, &mut j_mat);
    let mut njev = 1usize;
    validate_jacobian_len(&j_mat, mn)?;

    let loss_function = LossFunction::new(options.loss, options.f_scale);
    #[cfg(feature = "trace")]
    trace.event(&[
        ("event", TraceValue::Str("initial_jac")),
        ("x", TraceValue::F64Slice(&x)),
        ("f0", TraceValue::F64Slice(&f)),
        ("jac", TraceValue::F64Slice(&j_mat)),
    ]);

    // SciPy `trf_no_bounds`: when a robust loss is present, the cost is
    // `0.5 * sum(rho[0])` and the Jacobian/residual are reweighted in place
    // before the gradient and SVD; otherwise the ordinary `0.5 * f.f` path runs.
    let mut cost = if robust {
        let rho = loss_function.evaluate_with(&f, host)?;
        let cost = 0.5 * pairwise_sum(&rho.rho0);
        scale_for_robust_loss(&mut j_mat, &mut f, &rho, n)?;
        cost
    } else {
        half_dot_with_host(host, &f, &f)?
    };
    let mut g = compute_grad_with_host(host, &j_mat, &f, m, n)?;
    #[cfg(feature = "trace")]
    trace.event(&[
        ("event", TraceValue::Str("initial_state")),
        ("cost", TraceValue::F64(cost)),
        ("grad", TraceValue::F64Slice(&g)),
    ]);

    let jac_scale = matches!(options.x_scale, XScale::Jac);
    let (mut scale, mut scale_inv) = match &options.x_scale {
        XScale::Unit => (vec![1.0; n], vec![1.0; n]),
        XScale::Values(scale) => {
            if scale.len() != n {
                return Err(TrfError::InvalidXScaleLength {
                    expected: n,
                    got: scale.len(),
                });
            }
            for (index, &value) in scale.iter().enumerate() {
                if !(value.is_finite() && value > 0.0) {
                    return Err(TrfError::InvalidXScaleValue { index, value });
                }
            }
            let scale_inv = scale.iter().map(|&v| 1.0 / v).collect();
            (scale.clone(), scale_inv)
        }
        XScale::Jac => compute_jac_scale(&j_mat, m, n, None)?,
    };

    let mut delta_vec = vec![0.0; n];
    for i in 0..n {
        delta_vec[i] = x0[i] * scale_inv[i];
    }
    let mut delta = norm_with_host(host, &delta_vec)?;
    if delta == 0.0 {
        delta = 1.0;
    }
    #[cfg(feature = "trace")]
    trace.event(&[
        ("event", TraceValue::Str("initial_delta")),
        ("delta", TraceValue::F64(delta)),
        ("scale", TraceValue::F64Slice(&scale)),
        ("scale_inv", TraceValue::F64Slice(&scale_inv)),
    ]);

    let max_nfev = options.max_nfev.unwrap_or(100 * n);
    let mut alpha = 0.0;
    let mut termination_status: Option<i32> = None;
    #[cfg(feature = "trace")]
    let mut iteration = 0usize;
    let g_norm = loop {
        let current_g_norm = inf_norm(&g);
        #[cfg(feature = "trace")]
        trace.event(&[
            ("event", TraceValue::Str("iteration_start")),
            ("iteration", TraceValue::Usize(iteration)),
            ("nfev", TraceValue::Usize(nfev)),
            ("njev", TraceValue::Usize(njev)),
            ("cost", TraceValue::F64(cost)),
            ("g_norm", TraceValue::F64(current_g_norm)),
            ("delta", TraceValue::F64(delta)),
            ("alpha", TraceValue::F64(alpha)),
            ("x", TraceValue::F64Slice(&x)),
            ("f", TraceValue::F64Slice(&f)),
            ("grad", TraceValue::F64Slice(&g)),
        ]);
        if current_g_norm < options.gtol {
            termination_status = Some(1);
        }

        if termination_status.is_some() || nfev == max_nfev {
            break current_g_norm;
        }

        let d = &scale;
        let mut g_h = vec![0.0; n];
        for i in 0..n {
            g_h[i] = d[i] * g[i];
        }

        let mut j_h = vec![0.0; mn];
        for i in 0..m {
            for j in 0..n {
                j_h[i * n + j] = j_mat[i * n + j] * d[j];
            }
        }
        #[cfg(feature = "trace")]
        trace.event(&[
            ("event", TraceValue::Str("svd_input")),
            ("iteration", TraceValue::Usize(iteration)),
            ("g_h", TraceValue::F64Slice(&g_h)),
            ("j_h", TraceValue::F64Slice(&j_h)),
        ]);

        let (u, s, vt) = host.svd(&j_h, m, n)?;
        validate_svd_output(&u, &s, &vt, m, n)?;
        #[cfg(feature = "trace")]
        trace.event(&[
            ("event", TraceValue::Str("svd_output")),
            ("iteration", TraceValue::Usize(iteration)),
            ("u", TraceValue::F64Slice(&u)),
            ("s", TraceValue::F64Slice(&s)),
            ("vt", TraceValue::F64Slice(&vt)),
        ]);
        let uf = u_transpose_dot_f_with_host(host, &u, &f, m, n)?;
        #[cfg(feature = "trace")]
        trace.event(&[
            ("event", TraceValue::Str("uf")),
            ("iteration", TraceValue::Usize(iteration)),
            ("uf", TraceValue::F64Slice(&uf)),
        ]);

        let mut actual_reduction = -1.0;
        let mut x_new = x.clone();
        let mut f_new = Vec::new();
        let mut cost_new = cost;
        let mut delta_new;

        while actual_reduction <= 0.0 && nfev < max_nfev {
            let (step_h, solved_alpha) =
                solve_lsq_trust_region_with_host(host, n, m, &uf, &s, &vt, delta, alpha)?;
            alpha = solved_alpha;
            let predicted_reduction =
                -evaluate_quadratic_with_host(host, &j_h, &g_h, &step_h, m, n)?;
            #[cfg(feature = "trace")]
            trace.event(&[
                ("event", TraceValue::Str("trust_step")),
                ("iteration", TraceValue::Usize(iteration)),
                ("delta", TraceValue::F64(delta)),
                ("alpha", TraceValue::F64(alpha)),
                ("step_h", TraceValue::F64Slice(&step_h)),
                ("predicted_reduction", TraceValue::F64(predicted_reduction)),
            ]);

            let mut step = vec![0.0; n];
            for i in 0..n {
                step[i] = d[i] * step_h[i];
                x_new[i] = x[i] + step[i];
            }

            fun(&x_new, &mut f_new);
            nfev += 1;
            if f_new.len() != m {
                return Err(TrfError::InvalidResidualLength {
                    expected: m,
                    got: f_new.len(),
                });
            }
            #[cfg(feature = "trace")]
            trace.event(&[
                ("event", TraceValue::Str("trial_fun")),
                ("iteration", TraceValue::Usize(iteration)),
                ("nfev", TraceValue::Usize(nfev)),
                ("step", TraceValue::F64Slice(&step)),
                ("x_new", TraceValue::F64Slice(&x_new)),
                ("f_new", TraceValue::F64Slice(&f_new)),
            ]);

            let step_h_norm = norm_with_host(host, &step_h)?;
            if !all_finite(&f_new) {
                delta = 0.25 * step_h_norm;
                #[cfg(feature = "trace")]
                trace.event(&[
                    ("event", TraceValue::Str("nonfinite_trial")),
                    ("iteration", TraceValue::Usize(iteration)),
                    ("step_h_norm", TraceValue::F64(step_h_norm)),
                    ("delta", TraceValue::F64(delta)),
                ]);
                continue;
            }

            cost_new = if robust {
                loss_function.cost_only(&f_new)
            } else {
                half_dot_with_host(host, &f_new, &f_new)?
            };
            actual_reduction = cost - cost_new;
            let updated = update_tr_radius(
                delta,
                actual_reduction,
                predicted_reduction,
                step_h_norm,
                step_h_norm > 0.95 * delta,
            );
            delta_new = updated.0;
            let ratio = updated.1;
            #[cfg(feature = "trace")]
            trace.event(&[
                ("event", TraceValue::Str("trial_quality")),
                ("iteration", TraceValue::Usize(iteration)),
                ("cost", TraceValue::F64(cost)),
                ("cost_new", TraceValue::F64(cost_new)),
                ("actual_reduction", TraceValue::F64(actual_reduction)),
                ("predicted_reduction", TraceValue::F64(predicted_reduction)),
                ("step_h_norm", TraceValue::F64(step_h_norm)),
                ("delta_new", TraceValue::F64(delta_new)),
                ("ratio", TraceValue::F64(ratio)),
            ]);

            let step_norm = norm_with_host(host, &step)?;
            let x_norm = norm_with_host(host, &x)?;
            termination_status = check_termination(
                actual_reduction,
                cost,
                step_norm,
                x_norm,
                ratio,
                options.ftol,
                options.xtol,
            );
            #[cfg(feature = "trace")]
            trace.event(&[
                ("event", TraceValue::Str("termination_check")),
                ("iteration", TraceValue::Usize(iteration)),
                ("step_norm", TraceValue::F64(step_norm)),
                ("x_norm", TraceValue::F64(x_norm)),
                ("status", TraceValue::I32(termination_status.unwrap_or(-1))),
            ]);
            if termination_status.is_some() {
                break;
            }

            alpha *= delta / delta_new;
            delta = delta_new;
        }

        if actual_reduction > 0.0 {
            x = x_new;
            std::mem::swap(&mut f, &mut f_new);
            f_true = f.clone();
            cost = cost_new;

            jac(&x, &f, &mut j_mat);
            validate_jacobian_len(&j_mat, mn)?;
            njev += 1;
            // SciPy reweights the freshly evaluated (unscaled) Jacobian and
            // residual here, before the gradient and the jac-scale update.
            if robust {
                let rho = loss_function.evaluate_with(&f, host)?;
                scale_for_robust_loss(&mut j_mat, &mut f, &rho, n)?;
            }
            g = compute_grad_with_host(host, &j_mat, &f, m, n)?;
            #[cfg(feature = "trace")]
            trace.event(&[
                ("event", TraceValue::Str("accepted_state")),
                ("iteration", TraceValue::Usize(iteration)),
                ("x", TraceValue::F64Slice(&x)),
                ("f", TraceValue::F64Slice(&f)),
                ("cost", TraceValue::F64(cost)),
                ("jac", TraceValue::F64Slice(&j_mat)),
                ("grad", TraceValue::F64Slice(&g)),
            ]);

            if jac_scale {
                let scaled = compute_jac_scale(&j_mat, m, n, Some(&scale_inv))?;
                scale = scaled.0;
                scale_inv = scaled.1;
            }
        }
        #[cfg(feature = "trace")]
        {
            iteration += 1;
        }
    };

    let status = termination_status.unwrap_or(0);
    #[cfg(feature = "trace")]
    trace.event(&[
        ("event", TraceValue::Str("final")),
        ("status", TraceValue::I32(status)),
        ("x", TraceValue::F64Slice(&x)),
        ("cost", TraceValue::F64(cost)),
        ("fun", TraceValue::F64Slice(&f_true)),
        ("jac", TraceValue::F64Slice(&j_mat)),
        ("grad", TraceValue::F64Slice(&g)),
        ("optimality", TraceValue::F64(g_norm)),
        ("nfev", TraceValue::Usize(nfev)),
        ("njev", TraceValue::Usize(njev)),
    ]);
    Ok(TrfResult {
        x,
        cost,
        fun: f_true,
        jac: j_mat,
        grad: g,
        optimality: g_norm,
        nfev,
        njev,
        status,
    })
}

/// Mirrors `common.compute_grad` for a dense Jacobian: `J^T f`. The row-major
/// `m`-by-`n` Jacobian is reduced column by column, matching BLAS `ddot`.
///
/// # Errors
///
/// Returns [`TrfError::SizeOverflow`] if `m * n` overflows `usize`, or
/// [`TrfError::InvalidSliceLength`] if `jac.len() != m * n` or `f.len() != m`.
pub fn compute_grad(jac: &[f64], f: &[f64], m: usize, n: usize) -> Result<Vec<f64>, TrfError> {
    let mn = m.checked_mul(n).ok_or(TrfError::SizeOverflow { m, n })?;
    check_len("jac", jac.len(), mn)?;
    check_len("f", f.len(), m)?;
    Ok(compute_grad_unchecked(jac, f, m, n))
}

fn compute_grad_unchecked(jac: &[f64], f: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut g = vec![0.0; n];
    for (col, value) in g.iter_mut().enumerate().take(n) {
        *value = ddot_column_like_blas(jac, f, m, n, col);
    }
    g
}

fn compute_grad_with_host(
    host: &dyn HostNumerics,
    jac: &[f64],
    f: &[f64],
    m: usize,
    n: usize,
) -> Result<Vec<f64>, TrfError> {
    if let Some(out) = host.fortran_matvec(jac, m, n, f, true)? {
        if out.len() != n {
            return Err(TrfError::InvalidSvdOutput(format!(
                "BLAS transpose matvec returned length {}, expected {}",
                out.len(),
                n
            )));
        }
        return Ok(out);
    }

    Ok(compute_grad_unchecked(jac, f, m, n))
}

fn ddot_column_like_blas(jac: &[f64], f: &[f64], m: usize, n: usize, col: usize) -> f64 {
    let mut lane0 = 0.0;
    let mut lane1 = 0.0;
    let mut i = 0usize;
    while i + 4 <= m {
        lane0 = jac[i * n + col].mul_add(f[i], lane0);
        lane1 = jac[(i + 1) * n + col].mul_add(f[i + 1], lane1);
        lane0 = jac[(i + 2) * n + col].mul_add(f[i + 2], lane0);
        lane1 = jac[(i + 3) * n + col].mul_add(f[i + 3], lane1);
        i += 4;
    }

    let mut acc = lane0 + lane1;
    while i < m {
        acc = jac[i * n + col].mul_add(f[i], acc);
        i += 1;
    }
    acc
}

/// Mirrors `common.compute_jac_scale`: `scale_inv = sum(J**2, axis=0)**0.5`
/// (sequential column reduction over the `m` rows), with the `None`/`Some`
/// floor handling.
///
/// # Errors
///
/// Returns [`TrfError::SizeOverflow`] if `m * n` overflows `usize`, or
/// [`TrfError::InvalidSliceLength`] if `jac.len() != m * n` or, when supplied,
/// `scale_inv_old.len() != n`.
pub fn compute_jac_scale(
    jac: &[f64],
    m: usize,
    n: usize,
    scale_inv_old: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<f64>), TrfError> {
    let mn = m.checked_mul(n).ok_or(TrfError::SizeOverflow { m, n })?;
    check_len("jac", jac.len(), mn)?;
    if let Some(old) = scale_inv_old {
        check_len("scale_inv_old", old.len(), n)?;
    }
    let mut scale_inv = vec![0.0; n];
    for (j, slot) in scale_inv.iter_mut().enumerate().take(n) {
        let mut acc = 0.0;
        for i in 0..m {
            let v = jac[i * n + j];
            acc += v * v;
        }
        *slot = acc.sqrt();
    }

    match scale_inv_old {
        None => {
            for value in &mut scale_inv {
                if *value == 0.0 {
                    *value = 1.0;
                }
            }
        }
        Some(old) => {
            for (value, &o) in scale_inv.iter_mut().zip(old.iter()) {
                *value = value.max(o);
            }
        }
    }

    let scale = scale_inv.iter().map(|&v| 1.0 / v).collect();
    Ok((scale, scale_inv))
}

/// Solve the dense trust-region subproblem from a thin SVD of the scaled
/// Jacobian, using the pure-Rust reductions (no injected BLAS).
///
/// # Errors
///
/// Returns [`TrfError::EmptyParameters`] if `n == 0`, [`TrfError::SizeOverflow`]
/// if `n * n` overflows `usize`, or [`TrfError::InvalidSliceLength`] if
/// `uf.len() != n`, `s.len() != n`, or `vt.len() != n * n`.
pub fn solve_lsq_trust_region(
    n: usize,
    m: usize,
    uf: &[f64],
    s: &[f64],
    vt: &[f64],
    delta: f64,
    initial_alpha: f64,
) -> Result<(Vec<f64>, f64), TrfError> {
    if n == 0 {
        return Err(TrfError::EmptyParameters);
    }
    let nn = n.checked_mul(n).ok_or(TrfError::SizeOverflow { m: n, n })?;
    check_len("uf", uf.len(), n)?;
    check_len("s", s.len(), n)?;
    check_len("vt", vt.len(), nn)?;
    solve_lsq_trust_region_impl(None, n, m, uf, s, vt, delta, initial_alpha)
}

#[allow(clippy::too_many_arguments)]
fn solve_lsq_trust_region_with_host(
    host: &dyn HostNumerics,
    n: usize,
    m: usize,
    uf: &[f64],
    s: &[f64],
    vt: &[f64],
    delta: f64,
    initial_alpha: f64,
) -> Result<(Vec<f64>, f64), TrfError> {
    solve_lsq_trust_region_impl(Some(host), n, m, uf, s, vt, delta, initial_alpha)
}

#[allow(clippy::too_many_arguments)]
fn solve_lsq_trust_region_impl(
    host: Option<&dyn HostNumerics>,
    n: usize,
    m: usize,
    uf: &[f64],
    s: &[f64],
    vt: &[f64],
    delta: f64,
    initial_alpha: f64,
) -> Result<(Vec<f64>, f64), TrfError> {
    let mut suf = vec![0.0; n];
    for i in 0..n {
        suf[i] = s[i] * uf[i];
    }

    let threshold = EPS * m as f64 * s[0];
    let full_rank = m >= n && s[n - 1] > threshold;

    if full_rank {
        let mut rhs = vec![0.0; n];
        for i in 0..n {
            rhs[i] = uf[i] / s[i];
        }
        let p = neg_v_dot_maybe(host, &rhs, vt, n)?;
        if norm_maybe(host, &p)? <= delta {
            return Ok((p, 0.0));
        }
    }

    let mut alpha_upper = norm_maybe(host, &suf)? / delta;
    let mut alpha_lower = if full_rank {
        let (phi, phi_prime) = phi_and_derivative(host, 0.0, &suf, s, delta, n)?;
        -phi / phi_prime
    } else {
        0.0
    };

    // SciPy seeds (and re-seeds) alpha with `(alpha_lower * alpha_upper) ** 0.5`
    // over NumPy scalars, which is the platform C `pow` and does not enter the
    // array ufunc's stride-0 table. The expression is mathematically a square
    // root, so the crate's own default stays `sqrt`; a host backend can supply
    // the exact scalar-power result its runtime produces instead.
    let mut alpha = if !full_rank && initial_alpha == 0.0 {
        (0.001 * alpha_upper).max(alpha_seed(host, alpha_lower, alpha_upper)?)
    } else {
        initial_alpha
    };
    let mut it = 0usize;
    while it < 10 {
        if alpha < alpha_lower || alpha > alpha_upper {
            alpha = (0.001 * alpha_upper).max(alpha_seed(host, alpha_lower, alpha_upper)?);
        }

        let (phi, phi_prime) = phi_and_derivative(host, alpha, &suf, s, delta, n)?;
        if phi < 0.0 {
            alpha_upper = alpha;
        }

        let ratio = phi / phi_prime;
        alpha_lower = alpha_lower.max(alpha - ratio);
        alpha -= (phi + delta) * ratio / delta;

        if phi.abs() < 0.01 * delta {
            break;
        }
        it += 1;
    }

    let mut rhs = vec![0.0; n];
    for i in 0..n {
        rhs[i] = suf[i] / (s[i] * s[i] + alpha);
    }
    let mut p = neg_v_dot_maybe(host, &rhs, vt, n)?;
    let scale = delta / norm_maybe(host, &p)?;
    for value in &mut p {
        *value *= scale;
    }
    Ok((p, alpha))
}

/// Mirrors `common.evaluate_quadratic` (1-D step): `0.5 * (Js).(Js) + s.g`,
/// where `Js = J.dot(s)`.
///
/// # Errors
///
/// Returns [`TrfError::SizeOverflow`] if `m * n` overflows `usize`, or
/// [`TrfError::InvalidSliceLength`] if `jac.len() != m * n`, `g.len() != n`, or
/// `step.len() != n`.
pub fn evaluate_quadratic(
    jac: &[f64],
    g: &[f64],
    step: &[f64],
    m: usize,
    n: usize,
) -> Result<f64, TrfError> {
    let mn = m.checked_mul(n).ok_or(TrfError::SizeOverflow { m, n })?;
    check_len("jac", jac.len(), mn)?;
    check_len("g", g.len(), n)?;
    check_len("step", step.len(), n)?;
    let mut q = 0.0;
    for i in 0..m {
        let row = i * n;
        let mut js = 0.0;
        for k in 0..n {
            js += jac[row + k] * step[k];
        }
        q += js * js;
    }
    let mut l = 0.0;
    for k in 0..n {
        l += step[k] * g[k];
    }
    Ok(0.5 * q + l)
}

fn evaluate_quadratic_with_host(
    host: &dyn HostNumerics,
    jac: &[f64],
    g: &[f64],
    step: &[f64],
    m: usize,
    n: usize,
) -> Result<f64, TrfError> {
    let js = match host.fortran_matvec(jac, m, n, step, false)? {
        Some(out) => {
            if out.len() != m {
                return Err(TrfError::InvalidSvdOutput(format!(
                    "BLAS matvec returned length {}, expected {}",
                    out.len(),
                    m
                )));
            }
            out
        }
        None => {
            let mut out = vec![0.0; m];
            for (i, value) in out.iter_mut().enumerate().take(m) {
                let row = i * n;
                let mut acc = 0.0;
                for k in 0..n {
                    acc += jac[row + k] * step[k];
                }
                *value = acc;
            }
            out
        }
    };

    let q = dot_with_host(host, &js, &js)?;
    let l = dot_with_host(host, step, g)?;
    Ok(0.5 * q + l)
}

pub fn update_tr_radius(
    mut delta: f64,
    actual_reduction: f64,
    predicted_reduction: f64,
    step_norm: f64,
    bound_hit: bool,
) -> (f64, f64) {
    let ratio = if predicted_reduction > 0.0 {
        actual_reduction / predicted_reduction
    } else if predicted_reduction == 0.0 && actual_reduction == 0.0 {
        1.0
    } else {
        0.0
    };

    if ratio < 0.25 {
        delta = 0.25 * step_norm;
    } else if ratio > 0.75 && bound_hit {
        delta *= 2.0;
    }

    (delta, ratio)
}

pub fn check_termination(
    d_f: f64,
    f_cost: f64,
    dx_norm: f64,
    x_norm: f64,
    ratio: f64,
    ftol: f64,
    xtol: f64,
) -> Option<i32> {
    let ftol_satisfied = d_f < ftol * f_cost && ratio > 0.25;
    let xtol_satisfied = dx_norm < xtol * (xtol + x_norm);

    if ftol_satisfied && xtol_satisfied {
        Some(4)
    } else if ftol_satisfied {
        Some(2)
    } else if xtol_satisfied {
        Some(3)
    } else {
        None
    }
}

fn finite_difference_step(xi: f64) -> f64 {
    // SciPy's `_numdiff` writes the relative step as the scalar `EPS ** 0.5`,
    // which is a `pow` rather than a `sqrt`. It is deliberately not routed
    // through the host-numerics power seam: `EPS` is `2**-52`, so the result is
    // the exactly representable `2**-26` under either operation, and no
    // conforming runtime can disagree about it.
    let sign = if xi >= 0.0 { 1.0 } else { -1.0 };
    EPS.sqrt() * sign * xi.abs().max(1.0)
}

fn validate_jacobian_len(jac: &[f64], expected: usize) -> Result<(), TrfError> {
    if jac.len() == expected {
        Ok(())
    } else {
        Err(TrfError::InvalidJacobianLength {
            expected,
            got: jac.len(),
        })
    }
}

/// Validates that a slice argument has the expected length, returning a typed
/// error (rather than panicking on a later out-of-bounds index) when it does
/// not. Used to keep the public helpers total on malformed input.
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

fn validate_svd_output(
    u: &[f64],
    s: &[f64],
    vt: &[f64],
    m: usize,
    n: usize,
) -> Result<(), TrfError> {
    if u.len() != m * n {
        return Err(TrfError::InvalidSvdOutput(format!(
            "U has length {}, expected {}",
            u.len(),
            m * n
        )));
    }
    if s.len() != n {
        return Err(TrfError::InvalidSvdOutput(format!(
            "s has length {}, expected {}",
            s.len(),
            n
        )));
    }
    if vt.len() != n * n {
        return Err(TrfError::InvalidSvdOutput(format!(
            "VT has length {}, expected {}",
            vt.len(),
            n * n
        )));
    }
    Ok(())
}

fn half_dot_with_host(host: &dyn HostNumerics, a: &[f64], b: &[f64]) -> Result<f64, TrfError> {
    Ok(0.5 * dot_with_host(host, a, b)?)
}

fn dot_with_host(host: &dyn HostNumerics, a: &[f64], b: &[f64]) -> Result<f64, TrfError> {
    if a.len() != b.len() {
        return Err(TrfError::InvalidSvdOutput(format!(
            "BLAS dot length mismatch {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if let Some(out) = host.dot(a, b)? {
        return Ok(out);
    }

    let mut acc = 0.0;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    Ok(acc)
}

fn norm_slice(x: &[f64]) -> f64 {
    let mut acc = 0.0;
    for value in x {
        acc += *value * *value;
    }
    acc.sqrt()
}

fn norm_with_host(host: &dyn HostNumerics, x: &[f64]) -> Result<f64, TrfError> {
    Ok(dot_with_host(host, x, x)?.sqrt())
}

fn norm_maybe(host: Option<&dyn HostNumerics>, x: &[f64]) -> Result<f64, TrfError> {
    match host {
        Some(host) => norm_with_host(host, x),
        None => Ok(norm_slice(x)),
    }
}

/// `base ** exponent` through the host-numerics seam, falling back to
/// `rust_default` (the value the crate's own arithmetic produces for the same
/// expression) when no backend result is supplied.
fn scalar_power_maybe(
    host: Option<&dyn HostNumerics>,
    base: f64,
    exponent: f64,
    rust_default: f64,
) -> Result<f64, TrfError> {
    if let Some(host) = host {
        if let Some(out) = host.power_scalar(base, exponent)? {
            return Ok(out);
        }
    }
    Ok(rust_default)
}

/// SciPy's `(alpha_lower * alpha_upper) ** 0.5` alpha seed, shared by the
/// initialization and the in-loop re-entry site.
fn alpha_seed(
    host: Option<&dyn HostNumerics>,
    alpha_lower: f64,
    alpha_upper: f64,
) -> Result<f64, TrfError> {
    let product = alpha_lower * alpha_upper;
    scalar_power_maybe(host, product, 0.5, product.sqrt())
}

fn power_maybe(
    host: Option<&dyn HostNumerics>,
    values: &[f64],
    exponent: f64,
) -> Result<Vec<f64>, TrfError> {
    if let Some(host) = host {
        if let Some(out) = host.power(values, exponent)? {
            if out.len() != values.len() {
                return Err(TrfError::InvalidSliceLength {
                    what: "host power result",
                    expected: values.len(),
                    got: out.len(),
                });
            }
            return Ok(out);
        }
    }
    Ok(values.iter().map(|value| value.powf(exponent)).collect())
}

fn inf_norm(x: &[f64]) -> f64 {
    let mut out = 0.0;
    for value in x {
        let abs = value.abs();
        if abs > out {
            out = abs;
        }
    }
    out
}

fn all_finite(x: &[f64]) -> bool {
    x.iter().all(|value| value.is_finite())
}

fn u_transpose_dot_f(u: &[f64], f: &[f64], m: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (j, slot) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0;
        for i in 0..m {
            acc += u[i * n + j] * f[i];
        }
        *slot = acc;
    }
    out
}

fn u_transpose_dot_f_with_host(
    host: &dyn HostNumerics,
    u: &[f64],
    f: &[f64],
    m: usize,
    n: usize,
) -> Result<Vec<f64>, TrfError> {
    // uf = U^T f. scipy's U (from SVD) is C-contiguous, so numpy's `U.T.dot(f)`
    // takes the row-major cblas path (RowMajor, Trans) - distinct from the
    // column-major Fortran path used for the F-contiguous Jacobian gradient.
    if let Some(out) = host.row_major_matvec(u, m, n, f, true)? {
        if out.len() != n {
            return Err(TrfError::InvalidSvdOutput(format!(
                "BLAS U transpose matvec returned length {}, expected {}",
                out.len(),
                n
            )));
        }
        return Ok(out);
    }

    Ok(u_transpose_dot_f(u, f, m, n))
}

fn neg_v_dot(rhs: &[f64], vt: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n];
    for (i, slot) in out.iter_mut().enumerate().take(n) {
        let mut acc = 0.0;
        for k in 0..n {
            acc += vt[k * n + i] * rhs[k];
        }
        *slot = -acc;
    }
    out
}

fn neg_v_dot_maybe(
    host: Option<&dyn HostNumerics>,
    rhs: &[f64],
    vt: &[f64],
    n: usize,
) -> Result<Vec<f64>, TrfError> {
    // SciPy computes the step as `-V.dot(rhs)` where `V` (the SVD `Vh`) is an
    // F-contiguous n-by-n array. Our `vt` is row-major V^T, which is byte-for-byte
    // the same buffer as scipy's column-major `V`, so numpy's `V.dot(rhs)` is
    // reproduced bit-exactly by `cblas_dgemv(RowMajor, Trans)` on `vt` (verified
    // identical to numpy's F-contiguous V.dot for n>=4, where a sequential sum
    // diverges - it only coincided at n=3).
    if let Some(host) = host {
        if let Some(out) = host.row_major_matvec(vt, n, n, rhs, true)? {
            if out.len() != n {
                return Err(TrfError::InvalidSvdOutput(format!(
                    "BLAS V matvec returned length {}, expected {}",
                    out.len(),
                    n
                )));
            }
            return Ok(out.into_iter().map(|value| -value).collect());
        }
    }

    Ok(neg_v_dot(rhs, vt, n))
}

fn phi_and_derivative(
    host: Option<&dyn HostNumerics>,
    alpha: f64,
    suf: &[f64],
    s: &[f64],
    delta: f64,
    n: usize,
) -> Result<(f64, f64), TrfError> {
    let mut quot = vec![0.0; n];
    let mut denom = vec![0.0; n];
    for i in 0..n {
        denom[i] = s[i] * s[i] + alpha;
        quot[i] = suf[i] / denom[i];
    }
    let denom3 = power_maybe(host, &denom, 3.0)?;
    // SciPy: `np.sum(suf ** 2 / denom ** 3)` over the `n` terms, which is
    // NumPy's pairwise summation (identical to a sequential add for `n < 8`).
    let mut terms = vec![0.0; n];
    for i in 0..n {
        terms[i] = suf[i] * suf[i] / denom3[i];
    }
    let derivative_sum = pairwise_sum(&terms);
    let p_norm = norm_maybe(host, &quot)?;
    let phi = p_norm - delta;
    let phi_prime = -derivative_sum / p_norm;
    Ok((phi, phi_prime))
}

#[cfg(test)]
mod tests {
    //! Site-level coverage for the two alpha seeds inside
    //! `solve_lsq_trust_region_impl`. They live here rather than in an
    //! integration test because each site needs its own hand-built rank
    //! deficient SVD and `initial_alpha`, which the public entry points do not
    //! expose.

    use super::*;
    use std::cell::RefCell;

    /// Records every scalar-power dispatch and answers with `outcome`.
    struct ScalarPowerBackend {
        calls: RefCell<Vec<(f64, f64)>>,
        outcome: Result<Option<f64>, BackendError>,
    }

    impl ScalarPowerBackend {
        fn answering(value: Option<f64>) -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                outcome: Ok(value),
            }
        }

        fn failing() -> Self {
            Self {
                calls: RefCell::new(Vec::new()),
                outcome: Err(BackendError::Failed("no host scalar power".to_string())),
            }
        }

        fn calls(&self) -> Vec<(f64, f64)> {
            self.calls.borrow().clone()
        }
    }

    impl HostNumerics for ScalarPowerBackend {
        fn svd(
            &self,
            _a: &[f64],
            _m: usize,
            _n: usize,
        ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), BackendError> {
            Err(BackendError::Failed(
                "solve_lsq_trust_region never calls host".to_string(),
            ))
        }

        fn power_scalar(&self, base: f64, exponent: f64) -> Result<Option<f64>, BackendError> {
            self.calls.borrow_mut().push((base, exponent));
            self.outcome.clone()
        }
    }

    /// A rank-deficient thin SVD: `s[n - 1] == 0` puts
    /// `solve_lsq_trust_region` on its alpha-seeding branch instead of the
    /// Gauss-Newton shortcut. `alpha_lower` is 0 and `alpha_upper` is
    /// `norm(s * uf) / delta = 10`, and `phi` has its root at `alpha = 9`,
    /// strictly inside that bracket.
    fn rank_deficient_inputs() -> (usize, usize, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
        let n = 2;
        let m = 3;
        let uf = vec![10.0, 1.0];
        let s = vec![1.0, 0.0];
        let vt = vec![1.0, 0.0, 0.0, 1.0];
        let delta = 1.0;
        (n, m, uf, s, vt, delta)
    }

    #[test]
    fn alpha_initialization_site_consults_the_scalar_power_hook() {
        let (n, m, uf, s, vt, delta) = rank_deficient_inputs();
        let backend = ScalarPowerBackend::failing();
        // `initial_alpha == 0.0` selects the initialization site.
        let err = solve_lsq_trust_region_impl(Some(&backend), n, m, &uf, &s, &vt, delta, 0.0)
            .expect_err("the failing hook must abort the solve at the seed");

        assert!(
            matches!(err, TrfError::Backend(BackendError::Failed(_))),
            "{err:?}"
        );
        // alpha_lower is 0 on the rank-deficient branch, so the seed base is
        // `alpha_lower * alpha_upper == 0.0` and the exponent is 0.5.
        assert_eq!(backend.calls(), vec![(0.0, 0.5)]);
    }

    #[test]
    fn alpha_reentry_site_consults_the_scalar_power_hook() {
        let (n, m, uf, s, vt, delta) = rank_deficient_inputs();
        let backend = ScalarPowerBackend::failing();
        // A non-zero `initial_alpha` skips the initialization site entirely, and
        // one far above `alpha_upper = 10.0` trips the in-loop re-entry site on
        // the first pass, so the single recorded call can only be that site.
        let err = solve_lsq_trust_region_impl(Some(&backend), n, m, &uf, &s, &vt, delta, 1.0e6)
            .expect_err("the failing hook must abort the solve at the re-entry");

        assert!(
            matches!(err, TrfError::Backend(BackendError::Failed(_))),
            "{err:?}"
        );
        assert_eq!(backend.calls(), vec![(0.0, 0.5)]);
    }

    #[test]
    fn declining_scalar_power_hook_keeps_the_rust_alpha() {
        let (n, m, uf, s, vt, delta) = rank_deficient_inputs();
        let backend = ScalarPowerBackend::answering(None);
        let (declined_step, declined_alpha) =
            solve_lsq_trust_region_impl(Some(&backend), n, m, &uf, &s, &vt, delta, 0.0)
                .expect("declining backend");
        let (rust_step, rust_alpha) =
            solve_lsq_trust_region_impl(None, n, m, &uf, &s, &vt, delta, 0.0).expect("rust path");

        assert!(!backend.calls().is_empty(), "the hook was never consulted");
        assert_eq!(declined_alpha.to_bits(), rust_alpha.to_bits());
        assert_eq!(declined_step, rust_step);
    }

    #[test]
    fn alpha_seed_returns_the_supplied_scalar_power() {
        // Both call sites route through `alpha_seed`, so this pins that the
        // backend's value is what they use -- a wrong-but-recognizable answer
        // comes straight back out instead of the crate's own square root.
        let backend = ScalarPowerBackend::answering(Some(42.0));
        assert_eq!(alpha_seed(Some(&backend), 3.0, 12.0).expect("seed"), 42.0);
        assert_eq!(backend.calls(), vec![(36.0, 0.5)]);
    }

    #[test]
    fn alpha_seed_falls_back_to_the_rust_square_root() {
        let expected = 36.0_f64.sqrt().to_bits();
        let declining = ScalarPowerBackend::answering(None);
        assert_eq!(
            alpha_seed(Some(&declining), 3.0, 12.0)
                .expect("seed")
                .to_bits(),
            expected
        );
        assert_eq!(
            alpha_seed(None, 3.0, 12.0).expect("seed").to_bits(),
            expected
        );
    }
}
