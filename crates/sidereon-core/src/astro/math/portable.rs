//! Portable scalar and linear-algebra boundary helpers.
//!
//! `nalgebra` selects architecture-specific matrixmultiply kernels for plain
//! `f64` dynamic matrices.  `Portable` keeps the same binary64 arithmetic while
//! making that dispatch ineligible, and routes every transcendental operation
//! through the Rust `libm` implementation.

use std::fmt;
use std::num::ParseFloatError;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use std::str::FromStr;

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::{DMatrix, DVector, Dyn, SMatrix, SVector, SVD};
use num_traits::{FromPrimitive, Num, One, Signed, ToPrimitive, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};
use trust_region_least_squares::trf::{BackendError, HostNumerics};

/// A transparent binary64 value used for portable nalgebra operations.
#[repr(transparent)]
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Portable(pub f64);

/// Core-owned numerical backend for the public trust-region solver.
///
/// The backend deliberately uses the same portable scalar SVD as the core
/// covariance paths and fixed-order scalar loops for the BLAS-like hooks.  It
/// is zero-sized, so sharing one value across independent solves is free.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PortableNumerics;

impl HostNumerics for PortableNumerics {
    fn svd(
        &self,
        values: &[f64],
        rows: usize,
        cols: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), BackendError> {
        thin_svd(values, rows, cols).map_err(BackendError::Failed)
    }

    fn dot(&self, lhs: &[f64], rhs: &[f64]) -> Result<Option<f64>, BackendError> {
        if lhs.len() != rhs.len() {
            return Err(BackendError::Failed(format!(
                "dot length mismatch: {} and {}",
                lhs.len(),
                rhs.len()
            )));
        }
        let mut result = 0.0;
        for index in 0..lhs.len() {
            result += lhs[index] * rhs[index];
        }
        Ok(Some(result))
    }

    fn fortran_matvec(
        &self,
        matrix: &[f64],
        rows: usize,
        cols: usize,
        vector: &[f64],
        transpose: bool,
    ) -> Result<Option<Vec<f64>>, BackendError> {
        matvec(matrix, rows, cols, vector, transpose, true)
    }

    fn row_major_matvec(
        &self,
        matrix: &[f64],
        rows: usize,
        cols: usize,
        vector: &[f64],
        transpose: bool,
    ) -> Result<Option<Vec<f64>>, BackendError> {
        matvec(matrix, rows, cols, vector, transpose, false)
    }

    fn power(&self, values: &[f64], exponent: f64) -> Result<Option<Vec<f64>>, BackendError> {
        Ok(Some(
            values
                .iter()
                .copied()
                .map(|value| libm::pow(value, exponent))
                .collect(),
        ))
    }

    fn power_scalar(&self, base: f64, exponent: f64) -> Result<Option<f64>, BackendError> {
        Ok(Some(libm::pow(base, exponent)))
    }
}

fn matvec(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    vector: &[f64],
    transpose: bool,
    column_major: bool,
) -> Result<Option<Vec<f64>>, BackendError> {
    let (input_len, output_len) = if transpose {
        (rows, cols)
    } else {
        (cols, rows)
    };
    if matrix.len() != rows.saturating_mul(cols) || vector.len() != input_len {
        return Err(BackendError::Failed(format!(
            "matvec dimensions {}x{} with vector length {}",
            rows,
            cols,
            vector.len()
        )));
    }
    let mut result = vec![0.0; output_len];
    for (output, slot) in result.iter_mut().enumerate() {
        let mut sum = 0.0;
        for (input, value) in vector.iter().enumerate() {
            let index = if column_major {
                if transpose {
                    output * rows + input
                } else {
                    input * rows + output
                }
            } else if transpose {
                input * cols + output
            } else {
                output * cols + input
            };
            sum += matrix[index] * value;
        }
        *slot = sum;
    }
    Ok(Some(result))
}

/// Convert a row-major binary64 slice to a dynamic portable matrix.
#[inline]
pub fn matrix_from_row_slice(rows: usize, cols: usize, values: &[f64]) -> DMatrix<Portable> {
    DMatrix::from_row_slice(
        rows,
        cols,
        &values.iter().copied().map(Portable).collect::<Vec<_>>(),
    )
}

/// Convert a binary64 dynamic matrix to the portable scalar representation.
#[inline]
pub fn matrix_from_f64(matrix: &DMatrix<f64>) -> DMatrix<Portable> {
    DMatrix::from_iterator(
        matrix.nrows(),
        matrix.ncols(),
        matrix.iter().copied().map(Portable),
    )
}

/// Convert a portable dynamic matrix back to binary64 without changing bits.
#[inline]
pub fn matrix_to_f64(matrix: &DMatrix<Portable>) -> DMatrix<f64> {
    DMatrix::from_iterator(
        matrix.nrows(),
        matrix.ncols(),
        matrix.iter().map(|value| value.0),
    )
}

/// Convert a binary64 dynamic vector to the portable scalar representation.
#[inline]
pub fn vector_from_f64(vector: &DVector<f64>) -> DVector<Portable> {
    DVector::from_iterator(vector.len(), vector.iter().copied().map(Portable))
}

/// Convert a portable dynamic vector back to binary64 without changing bits.
#[inline]
pub fn vector_to_f64(vector: &DVector<Portable>) -> DVector<f64> {
    DVector::from_iterator(vector.len(), vector.iter().map(|value| value.0))
}

/// Thin SVD of a row-major binary64 matrix using the portable scalar.
#[allow(clippy::type_complexity)]
pub fn thin_svd(
    values: &[f64],
    rows: usize,
    cols: usize,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
    if values.len() != rows.saturating_mul(cols) {
        return Err(format!(
            "SVD input has length {}, expected {}x{}",
            values.len(),
            rows,
            cols
        ));
    }
    let matrix = matrix_from_row_slice(rows, cols, values);
    let svd = matrix.svd(true, true);
    let u = svd
        .u
        .ok_or_else(|| "portable SVD did not produce U".to_string())?;
    let vt = svd
        .v_t
        .ok_or_else(|| "portable SVD did not produce V^T".to_string())?;
    let k = rows.min(cols);
    let mut u_out = vec![0.0; rows * k];
    for row in 0..rows {
        for col in 0..k {
            u_out[row * k + col] = u[(row, col)].0;
        }
    }
    let s_out = svd.singular_values.iter().map(|value| value.0).collect();
    let mut vt_out = vec![0.0; k * cols];
    for row in 0..k {
        for col in 0..cols {
            vt_out[row * cols + col] = vt[(row, col)].0;
        }
    }
    Ok((u_out, s_out, vt_out))
}

/// A dynamic matrix product evaluated by nalgebra with the portable scalar.
#[inline]
pub fn product(lhs: &DMatrix<f64>, rhs: &DMatrix<f64>) -> DMatrix<f64> {
    matrix_to_f64(&(matrix_from_f64(lhs) * matrix_from_f64(rhs)))
}

/// A dynamic matrix/vector product evaluated by nalgebra with the portable scalar.
#[inline]
pub fn product_vector(lhs: &DMatrix<f64>, rhs: &DVector<f64>) -> DVector<f64> {
    vector_to_f64(&(matrix_from_f64(lhs) * vector_from_f64(rhs)))
}

/// Fixed-size matrix product evaluated through the portable scalar.
#[inline]
pub fn product_fixed<const N: usize>(
    lhs: &SMatrix<f64, N, N>,
    rhs: &SMatrix<f64, N, N>,
) -> SMatrix<f64, N, N> {
    let lhs_portable = SMatrix::<Portable, N, N>::from_fn(|row, col| Portable(lhs[(row, col)]));
    let rhs_portable = SMatrix::<Portable, N, N>::from_fn(|row, col| Portable(rhs[(row, col)]));
    let product = lhs_portable * rhs_portable;
    SMatrix::from_fn(|row, col| product[(row, col)].0)
}

/// Solve a dynamic square system through nalgebra's LU decomposition on the
/// portable scalar.
#[inline]
pub fn solve_lu(lhs: &DMatrix<f64>, rhs: &DVector<f64>) -> Option<DVector<f64>> {
    matrix_from_f64(lhs)
        .lu()
        .solve(&vector_from_f64(rhs))
        .map(|solution| vector_to_f64(&solution))
}

/// Cholesky solve for a dynamic positive-definite system through `Portable`.
#[inline]
pub fn solve_cholesky(lhs: &DMatrix<f64>, rhs: &DVector<f64>) -> Option<DVector<f64>> {
    matrix_from_f64(lhs)
        .cholesky()
        .map(|factor| vector_to_f64(&factor.solve(&vector_from_f64(rhs))))
}

/// Cholesky lower factor for a dynamic positive-definite matrix through
/// `Portable`.
#[inline]
pub fn cholesky_lower_dynamic(lhs: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    matrix_from_f64(lhs)
        .cholesky()
        .map(|factor| matrix_to_f64(&factor.l()))
}

/// Symmetric eigendecomposition for a dynamic real matrix through `Portable`.
#[inline]
pub fn symmetric_eigen_dynamic(matrix: &DMatrix<f64>) -> (DMatrix<f64>, DVector<f64>) {
    let eigen = matrix_from_f64(matrix).symmetric_eigen();
    (
        matrix_to_f64(&eigen.eigenvectors),
        vector_to_f64(&eigen.eigenvalues),
    )
}

/// Symmetric eigendecomposition for a fixed-size 6x6 real matrix through
/// `Portable`.
#[inline]
pub fn symmetric_eigen6(matrix: &SMatrix<f64, 6, 6>) -> (SMatrix<f64, 6, 6>, SVector<f64, 6>) {
    let portable = SMatrix::<Portable, 6, 6>::from_fn(|row, col| Portable(matrix[(row, col)]));
    let eigen = portable.symmetric_eigen();
    (
        SMatrix::from_fn(|row, col| eigen.eigenvectors[(row, col)].0),
        SVector::from_fn(|row, _| eigen.eigenvalues[row].0),
    )
}

/// Cholesky lower factor for a fixed-size real matrix through `Portable`.
#[inline]
pub fn cholesky_lower<const N: usize>(matrix: &SMatrix<f64, N, N>) -> Option<SMatrix<f64, N, N>> {
    let portable = SMatrix::<Portable, N, N>::from_fn(|row, col| Portable(matrix[(row, col)]));
    portable
        .cholesky()
        .map(|factor| SMatrix::from_fn(|row, col| factor.l()[(row, col)].0))
}

/// SVD of a dynamic binary64 matrix through the portable scalar.
#[inline]
pub fn svd(matrix: &DMatrix<f64>, compute_u: bool, compute_v: bool) -> SVD<Portable, Dyn, Dyn> {
    matrix_from_f64(matrix).svd(compute_u, compute_v)
}

impl Portable {
    #[inline]
    pub const fn new(value: f64) -> Self {
        Self(value)
    }

    #[inline]
    pub const fn get(self) -> f64 {
        self.0
    }
}

impl PartialOrd for Portable {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl fmt::Display for Portable {
    #[inline]
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl From<f64> for Portable {
    #[inline]
    fn from(value: f64) -> Self {
        Self(value)
    }
}

impl From<Portable> for f64 {
    #[inline]
    fn from(value: Portable) -> Self {
        value.0
    }
}

impl Add for Portable {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        Self(self.0 + rhs.0)
    }
}

impl AddAssign for Portable {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.0 += rhs.0;
    }
}

impl Sub for Portable {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        Self(self.0 - rhs.0)
    }
}

impl SubAssign for Portable {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.0 -= rhs.0;
    }
}

impl Mul for Portable {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self {
        Self(self.0 * rhs.0)
    }
}

impl MulAssign for Portable {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        self.0 *= rhs.0;
    }
}

impl Div for Portable {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self {
        Self(self.0 / rhs.0)
    }
}

impl DivAssign for Portable {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        self.0 /= rhs.0;
    }
}

impl Rem for Portable {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self {
        Self(self.0 % rhs.0)
    }
}

impl RemAssign for Portable {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        self.0 %= rhs.0;
    }
}

impl Neg for Portable {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl Zero for Portable {
    #[inline]
    fn zero() -> Self {
        Self(0.0)
    }

    #[inline]
    fn is_zero(&self) -> bool {
        self.0 == 0.0
    }
}

impl One for Portable {
    #[inline]
    fn one() -> Self {
        Self(1.0)
    }
}

impl Num for Portable {
    type FromStrRadixErr = ParseFloatError;

    #[inline]
    fn from_str_radix(src: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        if radix == 10 {
            src.parse().map(Self)
        } else {
            // The scalar is used for binary64 matrices; retain the standard
            // parser's error type while supporting the radix-independent path.
            src.parse().map(Self)
        }
    }
}

impl Signed for Portable {
    #[inline]
    fn abs(&self) -> Self {
        Self(libm::fabs(self.0))
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        if self.0 <= other.0 {
            Self::zero()
        } else {
            Self(self.0 - other.0)
        }
    }

    #[inline]
    fn signum(&self) -> Self {
        Self(self.0.signum())
    }

    #[inline]
    fn is_positive(&self) -> bool {
        self.0 > 0.0
    }

    #[inline]
    fn is_negative(&self) -> bool {
        self.0 < 0.0
    }
}

impl ToPrimitive for Portable {
    #[inline]
    fn to_i64(&self) -> Option<i64> {
        self.0.to_i64()
    }

    #[inline]
    fn to_u64(&self) -> Option<u64> {
        self.0.to_u64()
    }

    #[inline]
    fn to_f64(&self) -> Option<f64> {
        Some(self.0)
    }

    #[inline]
    fn to_f32(&self) -> Option<f32> {
        Some(self.0 as f32)
    }
}

impl FromPrimitive for Portable {
    #[inline]
    fn from_i64(value: i64) -> Option<Self> {
        Some(Self(value as f64))
    }

    #[inline]
    fn from_u64(value: u64) -> Option<Self> {
        Some(Self(value as f64))
    }

    #[inline]
    fn from_f64(value: f64) -> Option<Self> {
        Some(Self(value))
    }

    #[inline]
    fn from_f32(value: f32) -> Option<Self> {
        Some(Self(value as f64))
    }
}

impl SubsetOf<Portable> for f64 {
    #[inline]
    fn to_superset(&self) -> Portable {
        Portable(*self)
    }

    #[inline]
    fn from_superset_unchecked(element: &Portable) -> Self {
        element.0
    }

    #[inline]
    fn is_in_subset(_: &Portable) -> bool {
        true
    }
}

impl SubsetOf<Portable> for f32 {
    #[inline]
    fn to_superset(&self) -> Portable {
        Portable(f64::from(*self))
    }

    #[inline]
    fn from_superset_unchecked(element: &Portable) -> Self {
        element.0 as f32
    }

    #[inline]
    fn is_in_subset(_: &Portable) -> bool {
        true
    }
}

impl SubsetOf<f64> for Portable {
    #[inline]
    fn to_superset(&self) -> f64 {
        self.0
    }

    #[inline]
    fn from_superset_unchecked(element: &f64) -> Self {
        Self(*element)
    }

    #[inline]
    fn is_in_subset(_: &f64) -> bool {
        true
    }
}

impl SubsetOf<f32> for Portable {
    #[inline]
    fn to_superset(&self) -> f32 {
        self.0 as f32
    }

    #[inline]
    fn from_superset_unchecked(element: &f32) -> Self {
        Self(f64::from(*element))
    }

    #[inline]
    fn is_in_subset(_: &f32) -> bool {
        true
    }
}

impl SubsetOf<Portable> for Portable {
    #[inline]
    fn to_superset(&self) -> Self {
        *self
    }

    #[inline]
    fn from_superset_unchecked(element: &Self) -> Self {
        *element
    }

    #[inline]
    fn is_in_subset(_: &Self) -> bool {
        true
    }
}

impl Field for Portable {}

impl SimdValue for Portable {
    const LANES: usize = 1;
    type Element = Self;
    type SimdBool = bool;

    #[inline]
    fn splat(value: Self::Element) -> Self {
        value
    }

    #[inline]
    fn extract(&self, _: usize) -> Self::Element {
        *self
    }

    #[inline]
    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }

    #[inline]
    fn replace(&mut self, _: usize, value: Self::Element) {
        *self = value;
    }

    #[inline]
    unsafe fn replace_unchecked(&mut self, _: usize, value: Self::Element) {
        *self = value;
    }

    #[inline]
    fn select(self, condition: bool, other: Self) -> Self {
        if condition {
            self
        } else {
            other
        }
    }
}

impl PrimitiveSimdValue for Portable {}

impl AbsDiffEq for Portable {
    type Epsilon = Self;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        Self(f64::EPSILON)
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        libm::fabs(self.0 - other.0) <= epsilon.0
    }

    #[inline]
    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        !self.abs_diff_eq(other, epsilon)
    }
}

impl RelativeEq for Portable {
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        Self(f64::EPSILON)
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        if self.abs_diff_eq(other, epsilon) {
            true
        } else {
            libm::fabs(self.0 - other.0) <= max_relative.0 * self.0.abs().max(other.0.abs())
        }
    }

    #[inline]
    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        !self.relative_eq(other, epsilon, max_relative)
    }
}

impl UlpsEq for Portable {
    #[inline]
    fn default_max_ulps() -> u32 {
        4
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.0.to_bits().abs_diff(other.0.to_bits()) <= u64::from(max_ulps)
            || self.abs_diff_eq(other, epsilon)
    }

    #[inline]
    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        !self.ulps_eq(other, epsilon, max_ulps)
    }
}

impl ComplexField for Portable {
    type RealField = Self;

    #[inline]
    fn from_real(re: Self) -> Self {
        re
    }

    #[inline]
    fn real(self) -> Self {
        self
    }

    #[inline]
    fn imaginary(self) -> Self {
        Self::zero()
    }

    #[inline]
    fn norm1(self) -> Self {
        self.abs()
    }

    #[inline]
    fn modulus(self) -> Self {
        self.abs()
    }

    #[inline]
    fn modulus_squared(self) -> Self {
        self * self
    }

    #[inline]
    fn argument(self) -> Self {
        if self >= Self::zero() {
            Self::zero()
        } else {
            Self::pi()
        }
    }

    #[inline]
    fn to_exp(self) -> (Self, Self) {
        if self >= Self::zero() {
            (self, Self::one())
        } else {
            (-self, -Self::one())
        }
    }

    #[inline]
    fn recip(self) -> Self {
        Self(self.0.recip())
    }

    #[inline]
    fn conjugate(self) -> Self {
        self
    }

    #[inline]
    fn scale(self, factor: Self) -> Self {
        self * factor
    }

    #[inline]
    fn unscale(self, factor: Self) -> Self {
        self / factor
    }

    #[inline]
    fn floor(self) -> Self {
        Self(libm::floor(self.0))
    }

    #[inline]
    fn ceil(self) -> Self {
        Self(libm::ceil(self.0))
    }

    #[inline]
    fn round(self) -> Self {
        Self(libm::round(self.0))
    }

    #[inline]
    fn trunc(self) -> Self {
        Self(libm::trunc(self.0))
    }

    #[inline]
    fn fract(self) -> Self {
        Self(self.0 - libm::trunc(self.0))
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        Self(self.0.mul_add(a.0, b.0))
    }

    #[inline]
    fn abs(self) -> Self {
        Self(libm::fabs(self.0))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self(libm::hypot(self.0, other.0))
    }

    #[inline]
    fn powi(self, exponent: i32) -> Self {
        if exponent == 0 {
            return Self::one();
        }
        let negative = exponent < 0;
        let mut power = exponent.unsigned_abs();
        let mut base = self;
        let mut result = Self::one();
        while power != 0 {
            if power & 1 != 0 {
                result *= base;
            }
            base *= base;
            power >>= 1;
        }
        if negative {
            Self::one() / result
        } else {
            result
        }
    }

    #[inline]
    fn powf(self, exponent: Self) -> Self {
        Self(libm::pow(self.0, exponent.0))
    }

    #[inline]
    fn powc(self, exponent: Self) -> Self {
        self.powf(exponent)
    }

    #[inline]
    fn sqrt(self) -> Self {
        Self(libm::sqrt(self.0))
    }

    #[inline]
    fn try_sqrt(self) -> Option<Self> {
        if self >= Self::zero() {
            Some(self.sqrt())
        } else {
            None
        }
    }

    #[inline]
    fn exp(self) -> Self {
        Self(libm::exp(self.0))
    }

    #[inline]
    fn exp2(self) -> Self {
        Self(libm::exp2(self.0))
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self(libm::expm1(self.0))
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Self(libm::log1p(self.0))
    }

    #[inline]
    fn ln(self) -> Self {
        Self(libm::log(self.0))
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        Self(libm::log(self.0) / libm::log(base.0))
    }

    #[inline]
    fn log2(self) -> Self {
        Self(libm::log2(self.0))
    }

    #[inline]
    fn log10(self) -> Self {
        Self(libm::log10(self.0))
    }

    #[inline]
    fn cbrt(self) -> Self {
        Self(libm::cbrt(self.0))
    }

    #[inline]
    fn sin(self) -> Self {
        Self(libm::sin(self.0))
    }

    #[inline]
    fn cos(self) -> Self {
        Self(libm::cos(self.0))
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        let (sin, cos) = libm::sincos(self.0);
        (Self(sin), Self(cos))
    }

    #[inline]
    fn tan(self) -> Self {
        Self(libm::tan(self.0))
    }

    #[inline]
    fn asin(self) -> Self {
        Self(libm::asin(self.0))
    }

    #[inline]
    fn acos(self) -> Self {
        Self(libm::acos(self.0))
    }

    #[inline]
    fn atan(self) -> Self {
        Self(libm::atan(self.0))
    }

    #[inline]
    fn sinh(self) -> Self {
        Self(libm::sinh(self.0))
    }

    #[inline]
    fn cosh(self) -> Self {
        Self(libm::cosh(self.0))
    }

    #[inline]
    fn tanh(self) -> Self {
        Self(libm::tanh(self.0))
    }

    #[inline]
    fn asinh(self) -> Self {
        Self(libm::asinh(self.0))
    }

    #[inline]
    fn acosh(self) -> Self {
        Self(libm::acosh(self.0))
    }

    #[inline]
    fn atanh(self) -> Self {
        Self(libm::atanh(self.0))
    }

    #[inline]
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
}

impl RealField for Portable {
    #[inline]
    fn is_sign_positive(&self) -> bool {
        self.0.is_sign_positive()
    }

    #[inline]
    fn is_sign_negative(&self) -> bool {
        self.0.is_sign_negative()
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        Self(libm::copysign(self.0, sign.0))
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        Self(self.0.max(other.0))
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        Self(self.0.min(other.0))
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        Self(self.0.clamp(min.0, max.0))
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self(libm::atan2(self.0, other.0))
    }

    #[inline]
    fn min_value() -> Option<Self> {
        Some(Self(f64::MIN))
    }

    #[inline]
    fn max_value() -> Option<Self> {
        Some(Self(f64::MAX))
    }

    #[inline]
    fn pi() -> Self {
        Self(std::f64::consts::PI)
    }

    #[inline]
    fn two_pi() -> Self {
        Self(std::f64::consts::PI + std::f64::consts::PI)
    }

    #[inline]
    fn frac_pi_2() -> Self {
        Self(std::f64::consts::FRAC_PI_2)
    }

    #[inline]
    fn frac_pi_3() -> Self {
        Self(std::f64::consts::FRAC_PI_3)
    }

    #[inline]
    fn frac_pi_4() -> Self {
        Self(std::f64::consts::FRAC_PI_4)
    }

    #[inline]
    fn frac_pi_6() -> Self {
        Self(std::f64::consts::FRAC_PI_6)
    }

    #[inline]
    fn frac_pi_8() -> Self {
        Self(std::f64::consts::FRAC_PI_8)
    }

    #[inline]
    fn frac_1_pi() -> Self {
        Self(std::f64::consts::FRAC_1_PI)
    }

    #[inline]
    fn frac_2_pi() -> Self {
        Self(std::f64::consts::FRAC_2_PI)
    }

    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self(std::f64::consts::FRAC_2_SQRT_PI)
    }

    #[inline]
    fn e() -> Self {
        Self(std::f64::consts::E)
    }

    #[inline]
    fn log2_e() -> Self {
        Self(std::f64::consts::LOG2_E)
    }

    #[inline]
    fn log10_e() -> Self {
        Self(std::f64::consts::LOG10_E)
    }

    #[inline]
    fn ln_2() -> Self {
        Self(std::f64::consts::LN_2)
    }

    #[inline]
    fn ln_10() -> Self {
        Self(std::f64::consts::LN_10)
    }
}

impl FromStr for Portable {
    type Err = ParseFloatError;

    #[inline]
    fn from_str(value: &str) -> Result<Self, Self::Err> {
        value.parse().map(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn samples(count: usize) -> Vec<f64> {
        let mut state = 0x9e3779b97f4a7c15_u64;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(0xd1342543de82ef95)
                    .wrapping_add(0xa4093822299f31d0);
                let fraction = f64::from_bits(0x3ff0000000000000 | (state >> 12)) - 1.0;
                let signed = if state & 1 == 0 { fraction } else { -fraction };
                signed * 64.0
            })
            .collect()
    }

    fn repeated_square(value: f64, exponent: i32) -> f64 {
        let negative = exponent < 0;
        let mut power = exponent.unsigned_abs();
        let mut base = value;
        let mut result = 1.0;
        while power != 0 {
            if power & 1 != 0 {
                result *= base;
            }
            base *= base;
            power >>= 1;
        }
        if negative {
            1.0 / result
        } else {
            result
        }
    }

    #[test]
    fn arithmetic_is_a_bit_copy_of_binary64() {
        let mut values = samples(128);
        values.extend([0.0, -0.0, 1.25, -3.5, f64::MIN_POSITIVE, f64::MAX]);
        for &left in &values {
            for &right in &values {
                assert_eq!(
                    (Portable(left) + Portable(right)).0.to_bits(),
                    (left + right).to_bits()
                );
                assert_eq!(
                    (Portable(left) - Portable(right)).0.to_bits(),
                    (left - right).to_bits()
                );
                assert_eq!(
                    (Portable(left) * Portable(right)).0.to_bits(),
                    (left * right).to_bits()
                );
                assert_eq!(
                    (Portable(left) / Portable(right)).0.to_bits(),
                    (left / right).to_bits()
                );
                assert_eq!(
                    Portable(left).partial_cmp(&Portable(right)),
                    left.partial_cmp(&right)
                );
                assert_eq!(Portable(left) < Portable(right), left < right);
                assert_eq!(Portable(left) <= Portable(right), left <= right);
                assert_eq!(Portable(left) > Portable(right), left > right);
                assert_eq!(Portable(left) >= Portable(right), left >= right);
            }
        }
    }

    #[test]
    fn transcendental_delegates_match_libm() {
        for value in samples(128) {
            let portable = Portable(value);
            assert_eq!(portable.sin().0.to_bits(), libm::sin(value).to_bits());
            assert_eq!(portable.cos().0.to_bits(), libm::cos(value).to_bits());
            let (sin, cos) = portable.sin_cos();
            let (expected_sin, expected_cos) = libm::sincos(value);
            assert_eq!(sin.0.to_bits(), expected_sin.to_bits());
            assert_eq!(cos.0.to_bits(), expected_cos.to_bits());
            assert_eq!(portable.tan().0.to_bits(), libm::tan(value).to_bits());
            assert_eq!(portable.sinh().0.to_bits(), libm::sinh(value).to_bits());
            assert_eq!(portable.cosh().0.to_bits(), libm::cosh(value).to_bits());
            assert_eq!(portable.tanh().0.to_bits(), libm::tanh(value).to_bits());
            assert_eq!(portable.asinh().0.to_bits(), libm::asinh(value).to_bits());
            assert_eq!(portable.cbrt().0.to_bits(), libm::cbrt(value).to_bits());
            assert_eq!(
                portable.hypot(portable).0.to_bits(),
                libm::hypot(value, value).to_bits()
            );
            assert_eq!(portable.sqrt().0.to_bits(), libm::sqrt(value).to_bits());
            assert_eq!(portable.exp().0.to_bits(), libm::exp(value).to_bits());
            assert_eq!(portable.exp2().0.to_bits(), libm::exp2(value).to_bits());
            assert_eq!(portable.exp_m1().0.to_bits(), libm::expm1(value).to_bits());
            assert_eq!(portable.atan().0.to_bits(), libm::atan(value).to_bits());
            assert_eq!(
                portable.atan2(portable).0.to_bits(),
                libm::atan2(value, value).to_bits()
            );
            let positive = value.abs() + 0.25;
            assert_eq!(
                Portable(positive).ln().0.to_bits(),
                libm::log(positive).to_bits()
            );
            assert_eq!(
                Portable(positive).log2().0.to_bits(),
                libm::log2(positive).to_bits()
            );
            assert_eq!(
                Portable(positive).log10().0.to_bits(),
                libm::log10(positive).to_bits()
            );
            assert_eq!(
                Portable(value / 64.0).asin().0.to_bits(),
                libm::asin(value / 64.0).to_bits()
            );
            assert_eq!(
                Portable(value / 64.0).acos().0.to_bits(),
                libm::acos(value / 64.0).to_bits()
            );
            assert_eq!(
                Portable(value / 64.0).atanh().0.to_bits(),
                libm::atanh(value / 64.0).to_bits()
            );
            assert_eq!(
                Portable(value / 64.0 + 0.5).ln_1p().0.to_bits(),
                libm::log1p(value / 64.0 + 0.5).to_bits()
            );
            assert_eq!(
                Portable(value / 64.0 + 1.0).acosh().0.to_bits(),
                libm::acosh(value / 64.0 + 1.0).to_bits()
            );
            let exponent = (value as i32) % 8;
            assert_eq!(
                Portable(value).powi(exponent).0.to_bits(),
                repeated_square(value, exponent).to_bits()
            );
            assert_eq!(
                Portable(positive).powf(Portable(value / 8.0)).0.to_bits(),
                libm::pow(positive, value / 8.0).to_bits()
            );
            assert_eq!(
                Portable(positive).log(Portable(positive + 0.5)).0.to_bits(),
                (libm::log(positive) / libm::log(positive + 0.5)).to_bits()
            );
        }
    }
}
