//! Host LAPACK and BLAS bridge for pinned SciPy 1.18.0 parity runs.
//!
//! This module resolves symbols from a configured dynamic library and exposes
//! thin SVD plus the small BLAS operations used by the trust-region solver.

use crate::trf;
use libloading::{Library, Symbol};
use std::env;
use std::error::Error;
use std::ffi::{c_char, c_int, OsString};
use std::fmt;
use std::path::{Path, PathBuf};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{__m512d, _mm512_loadu_pd, _mm512_storeu_pd};

const LAPACK_PATH_ENV: &str = "TRUST_REGION_LEAST_SQUARES_LAPACK_PATH";
const NUMPY_BLAS_PATH_ENV: &str = "TRUST_REGION_LEAST_SQUARES_NUMPY_BLAS_PATH";

type LapackInt = c_int;
type BlasWide = i64;

type Dgesdd = unsafe extern "C" fn(
    jobz: *const c_char,
    m: *const LapackInt,
    n: *const LapackInt,
    a: *mut f64,
    lda: *const LapackInt,
    s: *mut f64,
    u: *mut f64,
    ldu: *const LapackInt,
    vt: *mut f64,
    ldvt: *const LapackInt,
    work: *mut f64,
    lwork: *const LapackInt,
    iwork: *mut LapackInt,
    info: *mut LapackInt,
);

type Dgemv = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const LapackInt,
    n: *const LapackInt,
    alpha: *const f64,
    a: *const f64,
    lda: *const LapackInt,
    x: *const f64,
    incx: *const LapackInt,
    beta: *const f64,
    y: *mut f64,
    incy: *const LapackInt,
);

type Ddot = unsafe extern "C" fn(
    n: *const LapackInt,
    dx: *const f64,
    incx: *const LapackInt,
    dy: *const f64,
    incy: *const LapackInt,
) -> f64;

type CblasDgemv = unsafe extern "C" fn(
    order: LapackInt,
    trans_a: LapackInt,
    m: LapackInt,
    n: LapackInt,
    alpha: f64,
    a: *const f64,
    lda: LapackInt,
    x: *const f64,
    incx: LapackInt,
    beta: f64,
    y: *mut f64,
    incy: LapackInt,
);

type Dgemv64 = unsafe extern "C" fn(
    trans: *const c_char,
    m: *const BlasWide,
    n: *const BlasWide,
    alpha: *const f64,
    a: *const f64,
    lda: *const BlasWide,
    x: *const f64,
    incx: *const BlasWide,
    beta: *const f64,
    y: *mut f64,
    incy: *const BlasWide,
);

type DdotWide = unsafe extern "C" fn(
    n: *const BlasWide,
    dx: *const f64,
    incx: *const BlasWide,
    dy: *const f64,
    incy: *const BlasWide,
) -> f64;

type CblasDgemv64 = unsafe extern "C" fn(
    order: LapackInt,
    trans_a: LapackInt,
    m: BlasWide,
    n: BlasWide,
    alpha: f64,
    a: *const f64,
    lda: BlasWide,
    x: *const f64,
    incx: BlasWide,
    beta: f64,
    y: *mut f64,
    incy: BlasWide,
);

#[cfg(target_arch = "x86_64")]
#[allow(improper_ctypes_definitions)]
type SvmlPow8 = unsafe extern "C" fn(__m512d, __m512d) -> __m512d;

#[derive(Clone, Debug, PartialEq)]
pub struct ThinSvdResult {
    pub m: usize,
    pub n: usize,
    /// Row-major m x n matrix matching `scipy.linalg.svd(..., full_matrices=False)[0]`.
    pub u: Vec<f64>,
    pub s: Vec<f64>,
    /// Row-major n x n matrix matching `scipy.linalg.svd(..., full_matrices=False)[2]`.
    pub vt: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
pub struct LapackSvd {
    path: Option<PathBuf>,
    blas_path: Option<PathBuf>,
}

impl LapackSvd {
    pub fn from_env() -> Self {
        Self {
            path: None,
            blas_path: None,
        }
    }

    pub fn with_path(path: impl Into<PathBuf>) -> Self {
        Self {
            path: Some(path.into()),
            blas_path: None,
        }
    }
}

impl trf::ThinSvd for LapackSvd {
    fn svd(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), trf::SvdError> {
        let result = match &self.path {
            Some(path) => thin_svd_with_lapack_path(path, m, n, a),
            None => thin_svd_from_env(m, n, a),
        }
        .map_err(trf::SvdError::from)?;
        Ok((result.u, result.s, result.vt))
    }

    fn dot(&self, a: &[f64], b: &[f64]) -> Result<Option<f64>, trf::SvdError> {
        let path = self.resolve_blas_path().map_err(trf::SvdError::from)?;
        blas_dot_with_path(path, a, b)
            .map(Some)
            .map_err(trf::SvdError::from)
    }

    fn fortran_matvec(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
        x: &[f64],
        transpose: bool,
    ) -> Result<Option<Vec<f64>>, trf::SvdError> {
        let path = self.resolve_blas_path().map_err(trf::SvdError::from)?;
        // Column-major Fortran `dgemv`. scipy applies this product to operands it
        // holds F-contiguous (the Jacobian: gradient J^T f and J @ step), where
        // numpy's `J.T.dot`/`J.dot` resolve to the column-major BLAS path. The
        // C-contiguous transpose product (uf = U^T f, U is C-contiguous) goes
        // through `row_major_matvec` with transpose instead, matching numpy there.
        blas_fortran_matvec_with_path(path, m, n, a, x, transpose)
            .map(Some)
            .map_err(trf::SvdError::from)
    }

    fn row_major_matvec(
        &self,
        a: &[f64],
        m: usize,
        n: usize,
        x: &[f64],
        transpose: bool,
    ) -> Result<Option<Vec<f64>>, trf::SvdError> {
        let path = self.resolve_blas_path().map_err(trf::SvdError::from)?;
        blas_row_major_matvec_with_path(path, m, n, a, x, transpose)
            .map(Some)
            .map_err(trf::SvdError::from)
    }

    fn power3(&self, x: &[f64]) -> Result<Option<Vec<f64>>, trf::SvdError> {
        let path = self.resolve_blas_path().map_err(trf::SvdError::from)?;
        numpy_svml_power3(path, x).map_err(trf::SvdError::from)
    }
}

impl LapackSvd {
    fn resolve_path(&self) -> Result<PathBuf, LapackError> {
        match &self.path {
            Some(path) => Ok(path.clone()),
            None => {
                let path = env::var_os(LAPACK_PATH_ENV).ok_or(LapackError::MissingEnv {
                    name: LAPACK_PATH_ENV,
                })?;
                if path.is_empty() {
                    return Err(LapackError::InvalidEnv {
                        name: LAPACK_PATH_ENV,
                        value: path,
                    });
                }
                Ok(PathBuf::from(path))
            }
        }
    }

    fn resolve_blas_path(&self) -> Result<PathBuf, LapackError> {
        if let Some(path) = &self.blas_path {
            return Ok(path.clone());
        }
        if let Some(path) = env::var_os(NUMPY_BLAS_PATH_ENV) {
            if path.is_empty() {
                return Err(LapackError::InvalidEnv {
                    name: NUMPY_BLAS_PATH_ENV,
                    value: path,
                });
            }
            return Ok(PathBuf::from(path));
        }

        let lapack_path = self.resolve_path()?;
        Ok(infer_numpy_blas_path(&lapack_path).unwrap_or(lapack_path))
    }
}

#[derive(Debug)]
pub enum LapackError {
    MissingEnv {
        name: &'static str,
    },
    InvalidEnv {
        name: &'static str,
        value: OsString,
    },
    InvalidDims {
        m: usize,
        n: usize,
    },
    InputLen {
        m: usize,
        n: usize,
        len: usize,
    },
    DotLen {
        a: usize,
        b: usize,
    },
    VectorLen {
        expected: usize,
        actual: usize,
    },
    IntOverflow {
        what: &'static str,
        value: usize,
    },
    Dlopen {
        path: PathBuf,
        message: String,
    },
    Symbol {
        path: PathBuf,
        symbol: &'static str,
        message: String,
    },
    LapackIllegalArgument {
        argument: i32,
    },
    DidNotConverge {
        info: i32,
    },
    InvalidWorkspace {
        value: f64,
    },
}

impl fmt::Display for LapackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LapackError::MissingEnv { name } => write!(f, "{name} is not set"),
            LapackError::InvalidEnv { name, value } => {
                write!(f, "{name} is not valid UTF-8/path text: {value:?}")
            }
            LapackError::InvalidDims { m, n } => {
                write!(f, "thin SVD requires m >= n and n > 0; got {m}x{n}")
            }
            LapackError::InputLen { m, n, len } => {
                write!(f, "input length {len} does not match row-major {m}x{n}")
            }
            LapackError::DotLen { a, b } => {
                write!(f, "dot input lengths differ: {a} vs {b}")
            }
            LapackError::VectorLen { expected, actual } => {
                write!(
                    f,
                    "vector length {actual} does not match expected {expected}"
                )
            }
            LapackError::IntOverflow { what, value } => {
                write!(f, "{what} value {value} does not fit LAPACK int")
            }
            LapackError::Dlopen { path, message } => {
                write!(
                    f,
                    "failed to load LAPACK library {}: {message}",
                    path.display()
                )
            }
            LapackError::Symbol {
                path,
                symbol,
                message,
            } => write!(
                f,
                "failed to resolve {symbol} in LAPACK library {}: {message}",
                path.display()
            ),
            LapackError::LapackIllegalArgument { argument } => {
                write!(f, "dgesdd_ rejected argument {argument}")
            }
            LapackError::DidNotConverge { info } => write!(f, "dgesdd_ did not converge: {info}"),
            LapackError::InvalidWorkspace { value } => {
                write!(f, "dgesdd_ returned invalid lwork query value {value:?}")
            }
        }
    }
}

impl Error for LapackError {}

impl From<LapackError> for trf::SvdError {
    fn from(value: LapackError) -> Self {
        trf::SvdError::Failed(value.to_string())
    }
}

pub fn thin_svd_from_env(
    m: usize,
    n: usize,
    a_row_major: &[f64],
) -> Result<ThinSvdResult, LapackError> {
    let path = env::var_os(LAPACK_PATH_ENV).ok_or(LapackError::MissingEnv {
        name: LAPACK_PATH_ENV,
    })?;
    if path.is_empty() {
        return Err(LapackError::InvalidEnv {
            name: LAPACK_PATH_ENV,
            value: path,
        });
    }
    thin_svd_with_lapack_path(Path::new(&path), m, n, a_row_major)
}

pub fn thin_svd_with_lapack_path(
    lapack_path: impl AsRef<Path>,
    m: usize,
    n: usize,
    a_row_major: &[f64],
) -> Result<ThinSvdResult, LapackError> {
    validate_inputs(m, n, a_row_major)?;
    let path = lapack_path.as_ref();

    // SAFETY: Loading a dynamic library is unsafe because constructors may run
    // and symbol types cannot be verified by Rust. The caller supplies the
    // scipy/OpenBLAS LAPACK library path, and every resolved symbol is used only
    // with the ABI LAPACK exposes for dgesdd_.
    let library = unsafe { Library::new(path) }.map_err(|err| LapackError::Dlopen {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;

    // SAFETY: The symbol name and signature match the scipy bundled
    // OpenBLAS/LAPACK LP64 dgesdd entry point used by scipy.linalg.svd here.
    // scipy/numpy >=2.x ship OpenBLAS built with the `scipy_` symbol prefix
    // (SYMBOLPREFIX=scipy_), so the public entry point is `scipy_dgesdd_`; older
    // wheels exported the bare `dgesdd_`. Try the bare name first, then prefixed.
    let dgesdd: Symbol<'_, Dgesdd> = unsafe {
        library
            .get(b"dgesdd_\0")
            .or_else(|_| library.get(b"scipy_dgesdd_\0"))
    }
    .map_err(|err| LapackError::Symbol {
        path: path.to_path_buf(),
        symbol: "dgesdd_ or scipy_dgesdd_",
        message: err.to_string(),
    })?;

    call_dgesdd(*dgesdd, m, n, a_row_major)
}

fn blas_dot_with_path(
    blas_path: impl AsRef<Path>,
    a: &[f64],
    b: &[f64],
) -> Result<f64, LapackError> {
    if a.len() != b.len() {
        return Err(LapackError::DotLen {
            a: a.len(),
            b: b.len(),
        });
    }
    let path = blas_path.as_ref();

    // SAFETY: The configured scipy/OpenBLAS library is loaded only to resolve
    // a BLAS ddot symbol used with valid input slices and unit strides.
    let library = unsafe { Library::new(path) }.map_err(|err| LapackError::Dlopen {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;

    // numpy >=2.x bundles an ILP64 OpenBLAS with the `scipy_` symbol prefix, so
    // the wide entry point is `scipy_ddot_64_`; older wheels exported the bare
    // `ddot_64_`. Prefer the ILP64 path (numpy's BLAS) when present.
    if let Ok(ddot) = unsafe {
        library
            .get::<DdotWide>(b"ddot_64_\0")
            .or_else(|_| library.get::<DdotWide>(b"scipy_ddot_64_\0"))
    } {
        let n = to_blas_wide("n", a.len())?;
        let inc = 1 as BlasWide;
        // SAFETY: ddot reads n elements from both live slices with unit strides.
        return Ok(unsafe { ddot(&n, a.as_ptr(), &inc, b.as_ptr(), &inc) });
    }

    // LP64 fallback: bare `ddot_` (old wheels) or prefixed `scipy_ddot_` (>=2.x).
    let ddot: Symbol<'_, Ddot> = unsafe {
        library
            .get(b"ddot_\0")
            .or_else(|_| library.get(b"scipy_ddot_\0"))
    }
    .map_err(|err| LapackError::Symbol {
        path: path.to_path_buf(),
        symbol: "ddot_64_ / scipy_ddot_64_ / ddot_ / scipy_ddot_",
        message: err.to_string(),
    })?;

    let n = to_lapack_int("n", a.len())?;
    let inc = 1 as LapackInt;
    // SAFETY: ddot_ reads n elements from both live slices with unit strides.
    Ok(unsafe { ddot(&n, a.as_ptr(), &inc, b.as_ptr(), &inc) })
}

fn blas_fortran_matvec_with_path(
    blas_path: impl AsRef<Path>,
    m: usize,
    n: usize,
    a_row_major: &[f64],
    x: &[f64],
    transpose: bool,
) -> Result<Vec<f64>, LapackError> {
    validate_inputs(m, n, a_row_major)?;
    let expected_x = if transpose { m } else { n };
    let out_len = if transpose { n } else { m };
    if x.len() != expected_x {
        return Err(LapackError::VectorLen {
            expected: expected_x,
            actual: x.len(),
        });
    }
    let path = blas_path.as_ref();

    // SAFETY: The configured scipy/OpenBLAS library is loaded only to resolve
    // a BLAS dgemv symbol used with buffers sized per BLAS contract.
    let library = unsafe { Library::new(path) }.map_err(|err| LapackError::Dlopen {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;
    let alpha = 1.0;
    let beta = 0.0;
    let trans = if transpose { b"T\0" } else { b"N\0" };
    let a_col_major = row_major_to_col_major(m, n, a_row_major);
    let mut y = vec![0.0; out_len];

    if let Ok(dgemv) = unsafe {
        library
            .get::<Dgemv64>(b"dgemv_64_\0")
            .or_else(|_| library.get::<Dgemv64>(b"scipy_dgemv_64_\0"))
    } {
        let m_i = to_blas_wide("m", m)?;
        let n_i = to_blas_wide("n", n)?;
        let lda = m_i;
        let inc = 1 as BlasWide;
        // SAFETY: dgemv_64_ reads the column-major m-by-n matrix, reads x with
        // unit stride, and writes exactly out_len elements into y.
        unsafe {
            dgemv(
                trans.as_ptr().cast::<c_char>(),
                &m_i,
                &n_i,
                &alpha,
                a_col_major.as_ptr(),
                &lda,
                x.as_ptr(),
                &inc,
                &beta,
                y.as_mut_ptr(),
                &inc,
            );
        }
        return Ok(y);
    }

    let dgemv: Symbol<'_, Dgemv> = unsafe {
        library
            .get(b"dgemv_\0")
            .or_else(|_| library.get(b"scipy_dgemv_\0"))
    }
    .map_err(|err| LapackError::Symbol {
        path: path.to_path_buf(),
        symbol: "dgemv_64_ / scipy_dgemv_64_ / dgemv_ / scipy_dgemv_",
        message: err.to_string(),
    })?;

    let m_i = to_lapack_int("m", m)?;
    let n_i = to_lapack_int("n", n)?;
    let lda = m_i;
    let inc = 1 as LapackInt;
    // SAFETY: dgemv_ reads the column-major m-by-n matrix, reads x with unit
    // stride, and writes exactly out_len elements into y.
    unsafe {
        dgemv(
            trans.as_ptr().cast::<c_char>(),
            &m_i,
            &n_i,
            &alpha,
            a_col_major.as_ptr(),
            &lda,
            x.as_ptr(),
            &inc,
            &beta,
            y.as_mut_ptr(),
            &inc,
        );
    }
    Ok(y)
}

fn blas_row_major_matvec_with_path(
    blas_path: impl AsRef<Path>,
    m: usize,
    n: usize,
    a_row_major: &[f64],
    x: &[f64],
    transpose: bool,
) -> Result<Vec<f64>, LapackError> {
    validate_inputs(m, n, a_row_major)?;
    // `cblas_dgemv(RowMajor, Trans/NoTrans, m, n, ...)` on the row-major m-by-n
    // matrix reproduces numpy's `A.dot(x)` (NoTrans: x len n, y len m) and
    // `A.T.dot(x)` (Trans: x len m, y len n) bit-for-bit, because it issues the
    // exact same OpenBLAS call numpy does (same kernel, same summation order).
    // The previous column-major Fortran-`dgemv` path used a different kernel and
    // diverged by 1 ULP from numpy on some value patterns.
    let (x_len, y_len) = if transpose { (m, n) } else { (n, m) };
    if x.len() != x_len {
        return Err(LapackError::VectorLen {
            expected: x_len,
            actual: x.len(),
        });
    }
    let path = blas_path.as_ref();

    // SAFETY: The configured scipy/OpenBLAS library is loaded only to resolve
    // cblas_dgemv with LP64 integer arguments and valid row-major buffers.
    let library = unsafe { Library::new(path) }.map_err(|err| LapackError::Dlopen {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;
    let row_major = 101 as LapackInt;
    // CblasNoTrans = 111, CblasTrans = 112.
    let trans = if transpose { 112 } else { 111 } as LapackInt;
    let mut y = vec![0.0; y_len];

    if let Ok(cblas_dgemv) = unsafe {
        library
            .get::<CblasDgemv64>(b"cblas_dgemv64_\0")
            .or_else(|_| library.get::<CblasDgemv64>(b"scipy_cblas_dgemv64_\0"))
    } {
        let m_i = to_blas_wide("m", m)?;
        let n_i = to_blas_wide("n", n)?;
        let lda = n_i;
        let inc = 1 as BlasWide;
        // SAFETY: cblas_dgemv64_ reads the row-major m-by-n matrix and x with
        // unit stride, then writes y_len values into y with unit stride.
        unsafe {
            cblas_dgemv(
                row_major,
                trans,
                m_i,
                n_i,
                1.0,
                a_row_major.as_ptr(),
                lda,
                x.as_ptr(),
                inc,
                0.0,
                y.as_mut_ptr(),
                inc,
            );
        }
        return Ok(y);
    }

    let cblas_dgemv: Symbol<'_, CblasDgemv> = unsafe {
        library
            .get(b"cblas_dgemv\0")
            .or_else(|_| library.get(b"scipy_cblas_dgemv\0"))
    }
    .map_err(|err| LapackError::Symbol {
        path: path.to_path_buf(),
        symbol: "cblas_dgemv64_ / scipy_cblas_dgemv64_ / cblas_dgemv / scipy_cblas_dgemv",
        message: err.to_string(),
    })?;

    let m_i = to_lapack_int("m", m)?;
    let n_i = to_lapack_int("n", n)?;
    let lda = n_i;
    let inc = 1 as LapackInt;
    // SAFETY: cblas_dgemv reads the row-major m-by-n matrix and x with unit
    // stride, then writes y_len values into y with unit stride.
    unsafe {
        cblas_dgemv(
            row_major,
            trans,
            m_i,
            n_i,
            1.0,
            a_row_major.as_ptr(),
            lda,
            x.as_ptr(),
            inc,
            0.0,
            y.as_mut_ptr(),
            inc,
        );
    }
    Ok(y)
}

#[cfg(target_arch = "x86_64")]
fn numpy_svml_power3(
    blas_path: impl AsRef<Path>,
    x: &[f64],
) -> Result<Option<Vec<f64>>, LapackError> {
    if !std::is_x86_feature_detected!("avx512f") {
        return Ok(None);
    }
    let Some(umath_path) = infer_numpy_umath_path(blas_path.as_ref()) else {
        return Ok(None);
    };

    // SAFETY: The NumPy extension is already loaded by the Python process in
    // live parity runs. Loading it here is only to resolve the same SVML pow
    // symbol used by NumPy's AVX-512 power ufunc.
    let library = unsafe { Library::new(&umath_path) }.map_err(|err| LapackError::Dlopen {
        path: umath_path.clone(),
        message: err.to_string(),
    })?;
    let pow8: Symbol<'_, SvmlPow8> =
        unsafe { library.get(b"__svml_pow8\0") }.map_err(|err| LapackError::Symbol {
            path: umath_path.clone(),
            symbol: "__svml_pow8",
            message: err.to_string(),
        })?;

    let mut out = vec![0.0; x.len()];
    for (chunk_index, chunk) in x.chunks(8).enumerate() {
        let mut bases = [1.0; 8];
        let exponents = [3.0; 8];
        bases[..chunk.len()].copy_from_slice(chunk);

        // SAFETY: AVX-512 support was checked above. The arrays have exactly
        // eight f64 lanes, and only the initialized output lanes are copied.
        let powered = unsafe { call_svml_pow8(*pow8, &bases, &exponents) };

        let start = chunk_index * 8;
        out[start..start + chunk.len()].copy_from_slice(&powered[..chunk.len()]);
    }
    Ok(Some(out))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn call_svml_pow8(pow8: SvmlPow8, bases: &[f64; 8], exponents: &[f64; 8]) -> [f64; 8] {
    let base_vec = _mm512_loadu_pd(bases.as_ptr());
    let exponent_vec = _mm512_loadu_pd(exponents.as_ptr());
    let out_vec = pow8(base_vec, exponent_vec);
    let mut powered = [0.0; 8];
    _mm512_storeu_pd(powered.as_mut_ptr(), out_vec);
    powered
}

#[cfg(not(target_arch = "x86_64"))]
fn numpy_svml_power3(
    _blas_path: impl AsRef<Path>,
    _x: &[f64],
) -> Result<Option<Vec<f64>>, LapackError> {
    Ok(None)
}

fn validate_inputs(m: usize, n: usize, a_row_major: &[f64]) -> Result<(), LapackError> {
    if n == 0 || m < n {
        return Err(LapackError::InvalidDims { m, n });
    }
    let expected = m.checked_mul(n).ok_or(LapackError::InputLen {
        m,
        n,
        len: a_row_major.len(),
    })?;
    if a_row_major.len() != expected {
        return Err(LapackError::InputLen {
            m,
            n,
            len: a_row_major.len(),
        });
    }
    Ok(())
}

fn to_lapack_int(what: &'static str, value: usize) -> Result<LapackInt, LapackError> {
    LapackInt::try_from(value).map_err(|_| LapackError::IntOverflow { what, value })
}

fn to_blas_wide(what: &'static str, value: usize) -> Result<BlasWide, LapackError> {
    BlasWide::try_from(value).map_err(|_| LapackError::IntOverflow { what, value })
}

fn infer_numpy_blas_path(lapack_path: &Path) -> Option<PathBuf> {
    if path_looks_like_numpy_blas(lapack_path) {
        return Some(lapack_path.to_path_buf());
    }

    let parent = lapack_path.parent()?;
    let parent_name = parent.file_name()?.to_string_lossy();
    if parent_name == "scipy.libs" {
        let site_packages = parent.parent()?;
        return find_openblas(site_packages.join("numpy.libs"));
    }

    if parent_name == ".dylibs" {
        let package_dir = parent.parent()?;
        if package_dir.file_name()?.to_string_lossy() == "numpy" {
            return Some(lapack_path.to_path_buf());
        }
        let site_packages = package_dir.parent()?;
        return find_openblas(site_packages.join("numpy").join(".dylibs"));
    }

    None
}

fn path_looks_like_numpy_blas(path: &Path) -> bool {
    path.components().any(|component| {
        let text = component.as_os_str().to_string_lossy();
        text == "numpy.libs" || text == "numpy"
    })
}

fn find_openblas(dir: PathBuf) -> Option<PathBuf> {
    let mut matches = Vec::new();
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy();
        if name.contains("openblas") {
            matches.push(path);
        }
    }
    matches.sort();
    matches.into_iter().next()
}

#[cfg(target_arch = "x86_64")]
fn infer_numpy_umath_path(blas_path: &Path) -> Option<PathBuf> {
    let mut candidates = Vec::new();

    let parent = blas_path.parent()?;
    let parent_name = parent.file_name()?.to_string_lossy();
    if parent_name == "numpy.libs" {
        let site_packages = parent.parent()?;
        candidates.push(site_packages.join("numpy").join("core"));
        candidates.push(site_packages.join("numpy").join("_core"));
    } else if parent_name == ".dylibs" {
        let package_dir = parent.parent()?;
        candidates.push(package_dir.join("core"));
        candidates.push(package_dir.join("_core"));
    }

    for dir in candidates {
        if let Some(path) = find_multiarray_umath(dir) {
            return Some(path);
        }
    }
    None
}

#[cfg(target_arch = "x86_64")]
fn find_multiarray_umath(dir: PathBuf) -> Option<PathBuf> {
    let mut matches = Vec::new();
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name()?.to_string_lossy();
        // Only the compiled extension carries `__svml_pow8`. NumPy 2's legacy
        // `numpy/core/` shim dir ships a pure-Python `_multiarray_umath.py`
        // re-export alongside the real `_core/*.so`; matching it would dlopen a
        // text file ("invalid ELF header"), so require a shared object.
        if name.starts_with("_multiarray_umath") && name.ends_with(".so") {
            matches.push(path);
        }
    }
    matches.sort();
    matches.into_iter().next()
}

fn call_dgesdd(
    dgesdd: Dgesdd,
    m: usize,
    n: usize,
    a_row_major: &[f64],
) -> Result<ThinSvdResult, LapackError> {
    let m_i = to_lapack_int("m", m)?;
    let n_i = to_lapack_int("n", n)?;
    let lda = m_i;
    let ldu = m_i;
    let ldvt = n_i;

    let mut a_col_major = row_major_to_col_major(m, n, a_row_major);
    let mut s = vec![0.0; n];
    let mut u_col_major = vec![0.0; m * n];
    let mut vt_col_major = vec![0.0; n * n];
    let mut iwork = vec![0 as LapackInt; 8 * n];

    let mut a_query = a_col_major.clone();
    let lwork = query_lwork(
        dgesdd,
        m_i,
        n_i,
        lda,
        ldu,
        ldvt,
        &mut a_query,
        &mut s,
        &mut u_col_major,
        &mut vt_col_major,
        &mut iwork,
    )?;

    let mut work = vec![0.0; usize::try_from(lwork).expect("positive lwork fits usize")];
    let mut info = 0 as LapackInt;
    let jobz = b"S\0";

    // SAFETY: All pointers refer to live, mutable buffers sized per LAPACK's
    // dgesdd_ contract for JOBZ='S', m>=n, LP64 integers, and the queried
    // workspace. LAPACK is allowed to overwrite A, WORK, IWORK, U, VT, and S.
    unsafe {
        dgesdd(
            jobz.as_ptr().cast::<c_char>(),
            &m_i,
            &n_i,
            a_col_major.as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            u_col_major.as_mut_ptr(),
            &ldu,
            vt_col_major.as_mut_ptr(),
            &ldvt,
            work.as_mut_ptr(),
            &lwork,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    check_info(info)?;

    Ok(ThinSvdResult {
        m,
        n,
        u: col_major_to_row_major(m, n, &u_col_major),
        s,
        vt: col_major_to_row_major(n, n, &vt_col_major),
    })
}

#[allow(clippy::too_many_arguments)]
fn query_lwork(
    dgesdd: Dgesdd,
    m: LapackInt,
    n: LapackInt,
    lda: LapackInt,
    ldu: LapackInt,
    ldvt: LapackInt,
    a: &mut [f64],
    s: &mut [f64],
    u: &mut [f64],
    vt: &mut [f64],
    iwork: &mut [LapackInt],
) -> Result<LapackInt, LapackError> {
    let mut work_query = [0.0];
    let lwork_query = -1 as LapackInt;
    let mut info = 0 as LapackInt;
    let jobz = b"S\0";

    // SAFETY: LAPACK workspace query uses the same valid buffers and dimensions
    // as the real call, with LWORK=-1 and a one-element WORK array.
    unsafe {
        dgesdd(
            jobz.as_ptr().cast::<c_char>(),
            &m,
            &n,
            a.as_mut_ptr(),
            &lda,
            s.as_mut_ptr(),
            u.as_mut_ptr(),
            &ldu,
            vt.as_mut_ptr(),
            &ldvt,
            work_query.as_mut_ptr(),
            &lwork_query,
            iwork.as_mut_ptr(),
            &mut info,
        );
    }
    check_info(info)?;

    let value = work_query[0];
    if !value.is_finite() || value < 1.0 || value > LapackInt::MAX as f64 {
        return Err(LapackError::InvalidWorkspace { value });
    }

    Ok(value as LapackInt)
}

fn check_info(info: LapackInt) -> Result<(), LapackError> {
    match info.cmp(&0) {
        std::cmp::Ordering::Equal => Ok(()),
        std::cmp::Ordering::Less => Err(LapackError::LapackIllegalArgument { argument: -info }),
        std::cmp::Ordering::Greater => Err(LapackError::DidNotConverge { info }),
    }
}

fn row_major_to_col_major(rows: usize, cols: usize, row_major: &[f64]) -> Vec<f64> {
    let mut col_major = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            col_major[col * rows + row] = row_major[row * cols + col];
        }
    }
    col_major
}

fn col_major_to_row_major(rows: usize, cols: usize, col_major: &[f64]) -> Vec<f64> {
    let mut row_major = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            row_major[row * cols + col] = col_major[col * rows + row];
        }
    }
    row_major
}
