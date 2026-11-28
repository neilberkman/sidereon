//! Forward-mode automatic differentiation (`Grad`) and 3D vector arithmetic
//! (`Vec3`, `FieldVec3`) reproducing Hipparchus 4.0.3's `Gradient`,
//! `Vector3D`, and `FieldVector3D<Gradient>`.

use super::fast_math;

/// First-order forward-mode derivative with `N` free parameters, reproducing
/// Hipparchus 4.0.3 `org.hipparchus.analysis.differentiation.Gradient`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Grad<const N: usize = 3> {
    pub(crate) value: f64,
    pub(crate) grad: [f64; N],
}

impl<const N: usize> Grad<N> {
    pub(crate) const ZERO: Self = Self {
        value: 0.0,
        grad: [0.0; N],
    };

    pub(crate) const ONE: Self = Self {
        value: 1.0,
        grad: [0.0; N],
    };

    pub(crate) const PI: Self = Self {
        value: fast_math::PI,
        grad: [0.0; N],
    };

    #[inline]
    pub(crate) const fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; N],
        }
    }

    #[inline]
    pub(crate) const fn variable(index: usize, value: f64) -> Self {
        let mut grad = [0.0; N];
        grad[index] = 1.0;
        Self { value, grad }
    }

    /// Embed a 3-parameter (position-only) gradient into the first three
    /// derivatives of an `N`-parameter gradient (`N >= 3`), with all remaining
    /// derivatives set to `0.0`.
    #[inline]
    pub(crate) const fn from_pos3(g: Grad<3>) -> Self {
        let mut grad = [0.0; N];
        if N >= 3 {
            grad[0] = g.grad[0];
            grad[1] = g.grad[1];
            grad[2] = g.grad[2];
        }
        Self {
            value: g.value,
            grad,
        }
    }

    #[inline]
    pub(crate) fn is_nan(self) -> bool {
        self.value.is_nan()
    }

    /// Hipparchus `Gradient.equals(ZERO)` (`FieldElement.isZero()`).
    #[inline]
    pub(crate) fn is_zero(self) -> bool {
        if self.value != 0.0 {
            return false;
        }
        let mut i = 0;
        while i < N {
            if !fast_math::precision_equals_1ulp(self.grad[i], 0.0) {
                return false;
            }
            i += 1;
        }
        true
    }

    #[inline]
    pub(crate) fn is_all_finite(self) -> bool {
        if !self.value.is_finite() {
            return false;
        }
        let mut i = 0;
        while i < N {
            if !self.grad[i].is_finite() {
                return false;
            }
            i += 1;
        }
        true
    }

    #[inline]
    pub(crate) fn compose(self, f0: f64, f1: f64) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = f1 * self.grad[i];
        }
        Self { value: f0, grad }
    }

    #[inline]
    pub(crate) fn add(self, a: Self) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = self.grad[i] + a.grad[i];
        }
        Self {
            value: self.value + a.value,
            grad,
        }
    }

    #[inline]
    pub(crate) fn add_f64(self, a: f64) -> Self {
        Self {
            value: self.value + a,
            grad: self.grad,
        }
    }

    #[inline]
    pub(crate) fn sub(self, a: Self) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = self.grad[i] - a.grad[i];
        }
        Self {
            value: self.value - a.value,
            grad,
        }
    }

    #[inline]
    pub(crate) fn sub_f64(self, a: f64) -> Self {
        Self {
            value: self.value - a,
            grad: self.grad,
        }
    }

    #[inline]
    pub(crate) fn mul_f64(self, a: f64) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = self.grad[i] * a;
        }
        Self {
            value: self.value * a,
            grad,
        }
    }

    #[inline]
    pub(crate) fn mul(self, a: Self) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = self.grad[i] * a.value + self.value * a.grad[i];
        }
        Self {
            value: self.value * a.value,
            grad,
        }
    }

    #[inline]
    pub(crate) fn div_f64(self, a: f64) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = self.grad[i] / a;
        }
        Self {
            value: self.value / a,
            grad,
        }
    }

    /// Hipparchus `Gradient.divide(Gradient)`: computes `inv1 = 1 / a.value`,
    /// `value = self.value * inv1`, and `(grad * a.value - value * a.grad) * inv1^2`.
    #[inline]
    pub(crate) fn div(self, a: Self) -> Self {
        let inv1 = 1.0 / a.value;
        let inv2 = inv1 * inv1;
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = (self.grad[i] * a.value - self.value * a.grad[i]) * inv2;
        }
        Self {
            value: self.value * inv1,
            grad,
        }
    }

    #[inline]
    pub(crate) fn negate(self) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = -self.grad[i];
        }
        Self {
            value: -self.value,
            grad,
        }
    }

    #[inline]
    pub(crate) fn abs(self) -> Self {
        if fast_math::double_to_long_bits(self.value) < 0 {
            self.negate()
        } else {
            self
        }
    }

    #[inline]
    pub(crate) fn copy_sign(self, sign: Self) -> Self {
        let m = fast_math::double_to_long_bits(self.value);
        let s = fast_math::double_to_long_bits(sign.value);
        if (m >= 0 && s >= 0) || (m < 0 && s < 0) {
            self
        } else {
            self.negate()
        }
    }

    #[inline]
    pub(crate) fn square(self) -> Self {
        let f0 = self.value;
        self.compose(f0 * f0, 2.0 * f0)
    }

    #[inline]
    pub(crate) fn reciprocal(self) -> Self {
        let inv1 = 1.0 / self.value;
        let inv2 = -inv1 * inv1;
        self.compose(inv1, inv2)
    }

    #[inline]
    pub(crate) fn sqrt(self) -> Self {
        let s = fast_math::sqrt(self.value);
        self.compose(s, 1.0 / (2.0 * s))
    }

    #[inline]
    pub(crate) fn asin(self) -> Self {
        let f0 = self.value;
        self.compose(fast_math::asin(f0), 1.0 / fast_math::sqrt(1.0 - f0 * f0))
    }

    #[inline]
    pub(crate) fn acos(self) -> Self {
        let f0 = self.value;
        self.compose(fast_math::acos(f0), -1.0 / fast_math::sqrt(1.0 - f0 * f0))
    }

    #[inline]
    pub(crate) fn sin(self) -> Self {
        let (s, c) = fast_math::sin_cos(self.value);
        self.compose(s, c)
    }

    #[inline]
    pub(crate) fn sin_cos(self) -> (Self, Self) {
        let (s, c) = fast_math::sin_cos(self.value);
        let mut grad_sin = [0.0; N];
        let mut grad_cos = [0.0; N];
        for i in 0..N {
            grad_sin[i] = self.grad[i] * c;
            grad_cos[i] = -self.grad[i] * s;
        }
        let sin = Self {
            value: s,
            grad: grad_sin,
        };
        let cos = Self {
            value: c,
            grad: grad_cos,
        };
        (sin, cos)
    }

    #[inline]
    pub(crate) fn atan2(self, x: Self) -> Self {
        self.atan2_with_fn(x, fast_math::atan2)
    }

    /// Hipparchus `Gradient.atan2`, returning `None` for a non-finite evaluation.
    #[inline]
    pub(crate) fn atan2_unscaled_with_fn(
        self,
        x: Self,
        atan2_fn: fn(f64, f64) -> f64,
    ) -> Option<Self> {
        let denom = self.value * self.value + x.value * x.value;
        if !denom.is_finite() || denom == 0.0 {
            return None;
        }
        let inv = 1.0 / denom;
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = (x.value * self.grad[i] - x.grad[i] * self.value) * inv;
        }
        let out = Self {
            value: atan2_fn(self.value, x.value),
            grad,
        };
        out.is_all_finite().then_some(out)
    }

    #[inline]
    pub(crate) fn atan2_with_fn(self, x: Self, atan2_fn: fn(f64, f64) -> f64) -> Self {
        if self.is_all_finite() && x.is_all_finite() {
            if let Some(exact) = self.atan2_unscaled_with_fn(x, atan2_fn) {
                return exact;
            }
            let max_abs = fast_math::abs(self.value).max(fast_math::abs(x.value));
            if max_abs != 0.0 {
                let value = atan2_fn(self.value, x.value);
                let exp_v = exact_exponent(max_abs);
                let sy = libm::ldexp(self.value, -exp_v);
                let sx = libm::ldexp(x.value, -exp_v);
                let s_denom = sy * sy + sx * sx;
                let (e_x, mx) = if x.value == 0.0 {
                    (0, 0.0)
                } else {
                    let e = exact_exponent(x.value);
                    (e, libm::ldexp(x.value, -e))
                };
                let (e_y, my) = if self.value == 0.0 {
                    (0, 0.0)
                } else {
                    let e = exact_exponent(self.value);
                    (e, libm::ldexp(self.value, -e))
                };
                let mut scaled_grad = [0.0; N];
                for i in 0..N {
                    let dy = self.grad[i];
                    let dx = x.grad[i];
                    let t1 = if mx != 0.0 && dy != 0.0 {
                        let e_dy = exact_exponent(dy);
                        Some((mx, libm::ldexp(dy, -e_dy), e_x + e_dy))
                    } else {
                        None
                    };
                    let t2 = if my != 0.0 && dx != 0.0 {
                        let e_dx = exact_exponent(dx);
                        Some((my, libm::ldexp(dx, -e_dx), e_y + e_dx))
                    } else {
                        None
                    };
                    scaled_grad[i] = match (t1, t2) {
                        (None, None) => 0.0,
                        (Some((a1, b1, e1)), None) => {
                            libm::ldexp((a1 * b1) / s_denom, (e1 - 2 * exp_v).clamp(-2200, 2200))
                        }
                        (None, Some((a2, b2, e2))) => {
                            libm::ldexp(-(a2 * b2) / s_denom, (e2 - 2 * exp_v).clamp(-2200, 2200))
                        }
                        (Some((a1, b1, e1)), Some((a2, b2, e2))) => {
                            let e_max = e1.max(e2);
                            let num = if e1 - e2 > 60 {
                                a1 * b1
                            } else if e2 - e1 > 60 {
                                -(a2 * b2)
                            } else if e1 >= e2 {
                                fast_math::linear_combination_2(
                                    a1,
                                    b1,
                                    -a2,
                                    libm::ldexp(b2, e2 - e1),
                                )
                            } else {
                                fast_math::linear_combination_2(
                                    a1,
                                    libm::ldexp(b1, e1 - e2),
                                    -a2,
                                    b2,
                                )
                            };
                            libm::ldexp(num / s_denom, (e_max - 2 * exp_v).clamp(-2200, 2200))
                        }
                    };
                }
                return Self {
                    value,
                    grad: scaled_grad,
                };
            }
        }
        let denom = self.value * self.value + x.value * x.value;
        let inv = 1.0 / denom;
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = (x.value * self.grad[i] - x.grad[i] * self.value) * inv;
        }
        Self {
            value: atan2_fn(self.value, x.value),
            grad,
        }
    }

    #[inline]
    pub(crate) fn scalb(self, n: i32) -> Self {
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = fast_math::scalb(self.grad[i], n);
        }
        Self {
            value: fast_math::scalb(self.value, n),
            grad,
        }
    }

    pub(crate) fn hypot(self, y: Self) -> Self {
        if self.value.is_infinite() || y.value.is_infinite() {
            Self::constant(f64::INFINITY)
        } else if self.value.is_nan() || y.value.is_nan() {
            Self::constant(f64::NAN)
        } else {
            let exp_x = fast_math::get_exponent(self.value);
            let exp_y = fast_math::get_exponent(y.value);
            if exp_x > exp_y + 27 {
                self.abs()
            } else if exp_y > exp_x + 27 {
                y.abs()
            } else {
                let middle_exp = (exp_x + exp_y) / 2;
                let scaled_x = self.scalb(-middle_exp);
                let scaled_y = y.scalb(-middle_exp);
                let scaled_h = scaled_x.mul(scaled_x).add(scaled_y.mul(scaled_y)).sqrt();
                scaled_h.scalb(middle_exp)
            }
        }
    }

    #[inline]
    pub(crate) fn linear_combination_2(a1: Self, b1: Self, a2: Self, b2: Self) -> Self {
        let value = fast_math::linear_combination_2(a1.value, b1.value, a2.value, b2.value);
        let mut grad = [0.0; N];
        for i in 0..N {
            grad[i] = fast_math::linear_combination_4(
                a1.value, b1.grad[i], a1.grad[i], b1.value, a2.value, b2.grad[i], a2.grad[i],
                b2.value,
            );
        }
        Self { value, grad }
    }

    #[inline]
    pub(crate) fn linear_combination_3(
        a1: Self,
        b1: Self,
        a2: Self,
        b2: Self,
        a3: Self,
        b3: Self,
    ) -> Self {
        let value = fast_math::linear_combination_3(
            a1.value, b1.value, a2.value, b2.value, a3.value, b3.value,
        );
        let mut a = [a1.value, 0.0, a2.value, 0.0, a3.value, 0.0];
        let mut b = [0.0, b1.value, 0.0, b2.value, 0.0, b3.value];
        let mut grad = [0.0; N];
        for i in 0..N {
            a[1] = a1.grad[i];
            a[3] = a2.grad[i];
            a[5] = a3.grad[i];
            b[0] = b1.grad[i];
            b[2] = b2.grad[i];
            b[4] = b3.grad[i];
            grad[i] = fast_math::linear_combination_arr(&a, &b);
        }
        Self { value, grad }
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub(crate) fn linear_combination_4(
        a1: Self,
        b1: Self,
        a2: Self,
        b2: Self,
        a3: Self,
        b3: Self,
        a4: Self,
        b4: Self,
    ) -> Self {
        let value = fast_math::linear_combination_4(
            a1.value, b1.value, a2.value, b2.value, a3.value, b3.value, a4.value, b4.value,
        );
        let mut a = [a1.value, 0.0, a2.value, 0.0, a3.value, 0.0, a4.value, 0.0];
        let mut b = [0.0, b1.value, 0.0, b2.value, 0.0, b3.value, 0.0, b4.value];
        let mut grad = [0.0; N];
        for i in 0..N {
            a[1] = a1.grad[i];
            a[3] = a2.grad[i];
            a[5] = a3.grad[i];
            a[7] = a4.grad[i];
            b[0] = b1.grad[i];
            b[2] = b2.grad[i];
            b[4] = b3.grad[i];
            b[6] = b4.grad[i];
            grad[i] = fast_math::linear_combination_arr(&a, &b);
        }
        Self { value, grad }
    }
}

/// 3D vector reproducing Hipparchus 4.0.3 `Vector3D`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Vec3 {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) z: f64,
}

impl Vec3 {
    pub(crate) const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[inline]
    pub(crate) const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// `new Vector3D(a, u)`.
    #[inline]
    pub(crate) fn from_scaled(a: f64, u: Self) -> Self {
        Self {
            x: a * u.x,
            y: a * u.y,
            z: a * u.z,
        }
    }

    /// `new Vector3D(a1, u1, a2, u2)`.
    #[inline]
    pub(crate) fn lincomb2(a1: f64, u1: Self, a2: f64, u2: Self) -> Self {
        Self {
            x: fast_math::linear_combination_2(a1, u1.x, a2, u2.x),
            y: fast_math::linear_combination_2(a1, u1.y, a2, u2.y),
            z: fast_math::linear_combination_2(a1, u1.z, a2, u2.z),
        }
    }

    /// `new Vector3D(a1, u1, a2, u2, a3, u3)`.
    #[inline]
    pub(crate) fn lincomb3(a1: f64, u1: Self, a2: f64, u2: Self, a3: f64, u3: Self) -> Self {
        Self {
            x: fast_math::linear_combination_3(a1, u1.x, a2, u2.x, a3, u3.x),
            y: fast_math::linear_combination_3(a1, u1.y, a2, u2.y, a3, u3.y),
            z: fast_math::linear_combination_3(a1, u1.z, a2, u2.z, a3, u3.z),
        }
    }

    #[inline]
    pub(crate) fn add(self, v: Self) -> Self {
        Self {
            x: self.x + v.x,
            y: self.y + v.y,
            z: self.z + v.z,
        }
    }

    #[inline]
    pub(crate) fn sub(self, v: Self) -> Self {
        Self {
            x: self.x - v.x,
            y: self.y - v.y,
            z: self.z - v.z,
        }
    }

    #[inline]
    pub(crate) fn negate(self) -> Self {
        Self {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    #[inline]
    pub(crate) fn scalar_mul(self, a: f64) -> Self {
        Self {
            x: a * self.x,
            y: a * self.y,
            z: a * self.z,
        }
    }

    #[inline]
    pub(crate) fn norm_sq(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    #[inline]
    pub(crate) fn norm(self) -> f64 {
        fast_math::sqrt(self.norm_sq())
    }

    #[inline]
    pub(crate) fn normalize(self) -> Option<Self> {
        let s = self.norm();
        if s == 0.0 {
            return None;
        }
        if s.is_finite() {
            return Some(self.scalar_mul(1.0 / s));
        }
        if !(self.x.is_finite() && self.y.is_finite() && self.z.is_finite()) {
            return None;
        }
        let max_abs = fast_math::abs(self.x)
            .max(fast_math::abs(self.y))
            .max(fast_math::abs(self.z));
        if max_abs == 0.0 {
            return None;
        }
        // Since s == +Inf with finite components, exp = get_exponent(max_abs) is in [511, 1023].
        // shift = 508 - exp is in [-515, -3], so fast_math::scalb always uses the exact IEEE
        // multiplication path (n > -1023), scaled.norm_sq() < 3 * 2^1018 < f64::MAX, 1 / scaled_s
        // is normal, and any component whose unit value is >= 2^-1075 stays >= 2^-567 (strictly
        // normal, zero rounding error in prescaling).
        let shift = 508 - fast_math::get_exponent(max_abs);
        let scaled = Self::new(
            fast_math::scalb(self.x, shift),
            fast_math::scalb(self.y, shift),
            fast_math::scalb(self.z, shift),
        );
        let scaled_s = scaled.norm();
        if scaled_s == 0.0 || !scaled_s.is_finite() {
            None
        } else {
            Some(scaled.scalar_mul(1.0 / scaled_s))
        }
    }

    #[inline]
    pub(crate) fn dot(self, v: Self) -> f64 {
        fast_math::linear_combination_3(self.x, v.x, self.y, v.y, self.z, v.z)
    }

    #[inline]
    pub(crate) fn cross(self, v: Self) -> Self {
        Self {
            x: fast_math::linear_combination_2(self.y, v.z, -self.z, v.y),
            y: fast_math::linear_combination_2(self.z, v.x, -self.x, v.z),
            z: fast_math::linear_combination_2(self.x, v.y, -self.y, v.x),
        }
    }

    /// `Vector3D.distance(v1, v2)` -> `v1.distance(v2)`.
    #[inline]
    pub(crate) fn distance(self, v: Self) -> f64 {
        let dx = v.x - self.x;
        let dy = v.y - self.y;
        let dz = v.z - self.z;
        fast_math::sqrt(dx * dx + dy * dy + dz * dz)
    }

    /// `Vector3D.orthogonal()`.
    pub(crate) fn orthogonal(self) -> Option<Self> {
        let s = self.norm();
        if s == 0.0 {
            return None;
        }
        if !s.is_finite() {
            if !(self.x.is_finite() && self.y.is_finite() && self.z.is_finite()) {
                return None;
            }
            let max_abs = fast_math::abs(self.x)
                .max(fast_math::abs(self.y))
                .max(fast_math::abs(self.z));
            if max_abs == 0.0 {
                return None;
            }
            let shift = 508 - fast_math::get_exponent(max_abs);
            let scaled = Self::new(
                fast_math::scalb(self.x, shift),
                fast_math::scalb(self.y, shift),
                fast_math::scalb(self.z, shift),
            );
            return scaled.orthogonal();
        }
        let threshold = 0.6 * s;
        if threshold == 0.0 {
            return None;
        }
        if fast_math::abs(self.x) <= threshold {
            let inverse = 1.0 / fast_math::sqrt(self.y * self.y + self.z * self.z);
            Some(Self::new(0.0, inverse * self.z, -inverse * self.y))
        } else if fast_math::abs(self.y) <= threshold {
            let inverse = 1.0 / fast_math::sqrt(self.x * self.x + self.z * self.z);
            Some(Self::new(-inverse * self.z, 0.0, inverse * self.x))
        } else {
            let inverse = 1.0 / fast_math::sqrt(self.x * self.x + self.y * self.y);
            Some(Self::new(inverse * self.y, -inverse * self.x, 0.0))
        }
    }

    /// `Vector3D.angle(v1, v2)`.
    pub(crate) fn angle(v1: Self, v2: Self) -> Option<f64> {
        let norm_product = v1.norm() * v2.norm();
        if norm_product == 0.0 {
            return None;
        }
        if norm_product.is_finite() {
            let dot = v1.dot(v2);
            let threshold = norm_product * 0.9999;
            let result = if dot < -threshold || dot > threshold {
                let v3 = v1.cross(v2);
                if dot >= 0.0 {
                    fast_math::asin(v3.norm() / norm_product)
                } else {
                    fast_math::PI - fast_math::asin(v3.norm() / norm_product)
                }
            } else {
                fast_math::acos(dot / norm_product)
            };
            if result.is_finite() {
                return Some(result);
            }
        }
        if !(v1.x.is_finite()
            && v1.y.is_finite()
            && v1.z.is_finite()
            && v2.x.is_finite()
            && v2.y.is_finite()
            && v2.z.is_finite())
        {
            return None;
        }
        let m1 = fast_math::abs(v1.x)
            .max(fast_math::abs(v1.y))
            .max(fast_math::abs(v1.z));
        let m2 = fast_math::abs(v2.x)
            .max(fast_math::abs(v2.y))
            .max(fast_math::abs(v2.z));
        if m1 == 0.0 || m2 == 0.0 {
            return None;
        }
        let s1 = -exact_exponent(m1);
        let s2 = -exact_exponent(m2);
        let u1 = Self::new(
            libm::ldexp(v1.x, s1),
            libm::ldexp(v1.y, s1),
            libm::ldexp(v1.z, s1),
        );
        let u2 = Self::new(
            libm::ldexp(v2.x, s2),
            libm::ldexp(v2.y, s2),
            libm::ldexp(v2.z, s2),
        );
        let np = u1.norm() * u2.norm();
        if np == 0.0 || !np.is_finite() {
            return None;
        }
        let dot = u1.dot(u2);
        let threshold = np * 0.9999;
        if dot < -threshold || dot > threshold {
            let v3 = u1.cross(u2);
            if dot >= 0.0 {
                Some(fast_math::asin(v3.norm() / np))
            } else {
                Some(fast_math::PI - fast_math::asin(v3.norm() / np))
            }
        } else {
            Some(fast_math::acos(dot / np))
        }
    }

    /// `Rotation.IDENTITY.applyTo(self)` (`q0 = 1, q1 = q2 = q3 = 0`).
    #[inline]
    pub(crate) fn apply_identity_rotation(self) -> Self {
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let s = 0.0 * x + 0.0 * y + 0.0 * z;
        Self {
            x: 2.0 * (1.0 * (x * 1.0 - (0.0 * z - 0.0 * y)) + s * 0.0) - x,
            y: 2.0 * (1.0 * (y * 1.0 - (0.0 * x - 0.0 * z)) + s * 0.0) - y,
            z: 2.0 * (1.0 * (z * 1.0 - (0.0 * y - 0.0 * x)) + s * 0.0) - z,
        }
    }
}

/// 3D vector reproducing Hipparchus 4.0.3 `FieldVector3D<Gradient>`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldVec3<const N: usize = 3> {
    pub(crate) x: Grad<N>,
    pub(crate) y: Grad<N>,
    pub(crate) z: Grad<N>,
}

impl<const N: usize> FieldVec3<N> {
    pub(crate) const ZERO: Self = Self {
        x: Grad::ZERO,
        y: Grad::ZERO,
        z: Grad::ZERO,
    };

    #[inline]
    pub(crate) const fn new(x: Grad<N>, y: Grad<N>, z: Grad<N>) -> Self {
        Self { x, y, z }
    }

    #[inline]
    pub(crate) const fn constant(v: Vec3) -> Self {
        Self {
            x: Grad::constant(v.x),
            y: Grad::constant(v.y),
            z: Grad::constant(v.z),
        }
    }

    /// `new FieldVector3D<>(a, u)` (`a.multiply(u.x)`).
    #[inline]
    pub(crate) fn from_scaled(a: Grad<N>, u: Self) -> Self {
        Self {
            x: a.mul(u.x),
            y: a.mul(u.y),
            z: a.mul(u.z),
        }
    }

    /// `new FieldVector3D<>(a1, u1, a2, u2)`.
    #[inline]
    pub(crate) fn lincomb2(a1: Grad<N>, u1: Self, a2: Grad<N>, u2: Self) -> Self {
        Self {
            x: Grad::linear_combination_2(a1, u1.x, a2, u2.x),
            y: Grad::linear_combination_2(a1, u1.y, a2, u2.y),
            z: Grad::linear_combination_2(a1, u1.z, a2, u2.z),
        }
    }

    /// `new FieldVector3D<>(a1, u1, a2, u2, a3, u3)`.
    #[inline]
    pub(crate) fn lincomb3(
        a1: Grad<N>,
        u1: Self,
        a2: Grad<N>,
        u2: Self,
        a3: Grad<N>,
        u3: Self,
    ) -> Self {
        Self {
            x: Grad::linear_combination_3(a1, u1.x, a2, u2.x, a3, u3.x),
            y: Grad::linear_combination_3(a1, u1.y, a2, u2.y, a3, u3.y),
            z: Grad::linear_combination_3(a1, u1.z, a2, u2.z, a3, u3.z),
        }
    }

    #[inline]
    pub(crate) fn add(self, v: Self) -> Self {
        Self {
            x: self.x.add(v.x),
            y: self.y.add(v.y),
            z: self.z.add(v.z),
        }
    }

    #[inline]
    pub(crate) fn sub(self, v: Self) -> Self {
        Self {
            x: self.x.sub(v.x),
            y: self.y.sub(v.y),
            z: self.z.sub(v.z),
        }
    }

    #[inline]
    pub(crate) fn negate(self) -> Self {
        Self {
            x: self.x.negate(),
            y: self.y.negate(),
            z: self.z.negate(),
        }
    }

    /// `FieldVector3D.scalarMultiply(T a)` (`x.multiply(a)`).
    #[inline]
    pub(crate) fn scalar_mul(self, a: Grad<N>) -> Self {
        Self {
            x: self.x.mul(a),
            y: self.y.mul(a),
            z: self.z.mul(a),
        }
    }

    #[inline]
    pub(crate) fn norm_sq(self) -> Grad<N> {
        self.x.square().add(self.y.square()).add(self.z.square())
    }

    #[inline]
    pub(crate) fn norm(self) -> Grad<N> {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub(crate) fn normalize(self) -> Option<Self> {
        let s = self.norm();
        if s.value == 0.0 {
            return None;
        }
        if s.value.is_finite() {
            return Some(self.scalar_mul(s.reciprocal()));
        }
        if !(self.x.value.is_finite() && self.y.value.is_finite() && self.z.value.is_finite()) {
            return None;
        }
        let max_abs = fast_math::abs(self.x.value)
            .max(fast_math::abs(self.y.value))
            .max(fast_math::abs(self.z.value));
        if max_abs == 0.0 {
            return None;
        }
        let exp = fast_math::get_exponent(max_abs);
        let shift = 508 - exp;
        let v_scaled = Vec3::new(
            fast_math::scalb(self.x.value, shift),
            fast_math::scalb(self.y.value, shift),
            fast_math::scalb(self.z.value, shift),
        );
        let scaled_s = v_scaled.norm();
        if scaled_s == 0.0 || !scaled_s.is_finite() {
            return None;
        }
        let u = v_scaled.scalar_mul(1.0 / scaled_s);
        // |self| = s_unit * 2^exp with s_unit in [1, 2*sqrt(3)) (exact shift by -508).
        let s_unit = fast_math::scalb(scaled_s, -508);
        let mut gx = [0.0; N];
        let mut gy = [0.0; N];
        let mut gz = [0.0; N];
        for i in 0..N {
            let dx = self.x.grad[i];
            let dy = self.y.grad[i];
            let dz = self.z.grad[i];
            if dx == 0.0 && dy == 0.0 && dz == 0.0 {
                continue;
            }
            if dx.is_finite() && dy.is_finite() && dz.is_finite() {
                let g_max = fast_math::abs(dx)
                    .max(fast_math::abs(dy))
                    .max(fast_math::abs(dz));
                let shift_g = (960 - exact_exponent(g_max)).min(0);
                let gs = Vec3::new(
                    libm::ldexp(dx, shift_g),
                    libm::ldexp(dy, shift_g),
                    libm::ldexp(dz, shift_g),
                );
                let proj = u.dot(gs);
                let exp_out = -(exp + shift_g);
                gx[i] = libm::ldexp((gs.x - u.x * proj) / s_unit, exp_out);
                gy[i] = libm::ldexp((gs.y - u.y * proj) / s_unit, exp_out);
                gz[i] = libm::ldexp((gs.z - u.z * proj) / s_unit, exp_out);
            } else {
                let proj = u.x * dx + u.y * dy + u.z * dz;
                gx[i] = (dx - u.x * proj) / s_unit;
                gy[i] = (dy - u.y * proj) / s_unit;
                gz[i] = (dz - u.z * proj) / s_unit;
            }
        }
        Some(Self::new(
            Grad {
                value: u.x,
                grad: gx,
            },
            Grad {
                value: u.y,
                grad: gy,
            },
            Grad {
                value: u.z,
                grad: gz,
            },
        ))
    }

    #[inline]
    pub(crate) fn dot(self, v: Self) -> Grad<N> {
        Grad::linear_combination_3(self.x, v.x, self.y, v.y, self.z, v.z)
    }

    #[inline]
    pub(crate) fn cross(self, v: Self) -> Self {
        Self {
            x: Grad::linear_combination_2(self.y, v.z, self.z.negate(), v.y),
            y: Grad::linear_combination_2(self.z, v.x, self.x.negate(), v.z),
            z: Grad::linear_combination_2(self.x, v.y, self.y.negate(), v.x),
        }
    }

    /// `FieldVector3D.distance(v1, v2)` -> `v1.distance(v2)`.
    #[inline]
    pub(crate) fn distance(self, v: Self) -> Grad<N> {
        let dx = v.x.sub(self.x);
        let dy = v.y.sub(self.y);
        let dz = v.z.sub(self.z);
        dx.square().add(dy.square()).add(dz.square()).sqrt()
    }

    /// `FieldVector3D.orthogonal()`.
    pub(crate) fn orthogonal(self) -> Option<Self> {
        let s = self.norm().value;
        if s == 0.0 {
            return None;
        }
        if !s.is_finite() {
            if !(self.x.value.is_finite() && self.y.value.is_finite() && self.z.value.is_finite()) {
                return None;
            }
            let max_abs = fast_math::abs(self.x.value)
                .max(fast_math::abs(self.y.value))
                .max(fast_math::abs(self.z.value));
            if max_abs == 0.0 {
                return None;
            }
            let shift = 508 - fast_math::get_exponent(max_abs);
            let v_scaled = Vec3::new(
                fast_math::scalb(self.x.value, shift),
                fast_math::scalb(self.y.value, shift),
                fast_math::scalb(self.z.value, shift),
            );
            let threshold = 0.6 * v_scaled.norm();
            if threshold == 0.0 || !threshold.is_finite() {
                return None;
            }
            return if fast_math::abs(v_scaled.x) <= threshold {
                Self::new(Grad::ZERO, self.z, self.y.negate()).normalize()
            } else if fast_math::abs(v_scaled.y) <= threshold {
                Self::new(self.z.negate(), Grad::ZERO, self.x).normalize()
            } else {
                Self::new(self.y, self.x.negate(), Grad::ZERO).normalize()
            };
        }
        let threshold = 0.6 * s;
        if threshold == 0.0 {
            return None;
        }
        if fast_math::abs(self.x.value) <= threshold {
            let inverse = self.y.square().add(self.z.square()).sqrt().reciprocal();
            Some(Self::new(
                Grad::ZERO,
                inverse.mul(self.z),
                inverse.mul(self.y).negate(),
            ))
        } else if fast_math::abs(self.y.value) <= threshold {
            let inverse = self.x.square().add(self.z.square()).sqrt().reciprocal();
            Some(Self::new(
                inverse.mul(self.z).negate(),
                Grad::ZERO,
                inverse.mul(self.x),
            ))
        } else {
            let inverse = self.x.square().add(self.y.square()).sqrt().reciprocal();
            Some(Self::new(
                inverse.mul(self.y),
                inverse.mul(self.x).negate(),
                Grad::ZERO,
            ))
        }
    }

    /// `FieldVector3D.angle(v1, v2)`.
    pub(crate) fn angle(v1: Self, v2: Self) -> Option<Grad<N>> {
        let norm_product = v1.norm().mul(v2.norm());
        if norm_product.value == 0.0 {
            return None;
        }
        if norm_product.value.is_finite() {
            let dot = v1.dot(v2);
            let threshold = norm_product.value * 0.9999;
            let result = if dot.value < -threshold || dot.value > threshold {
                let v3 = v1.cross(v2);
                if dot.value >= 0.0 {
                    v3.norm().div(norm_product).asin()
                } else {
                    v3.norm().div(norm_product).asin().sub(Grad::PI).negate()
                }
            } else {
                dot.div(norm_product).acos()
            };
            if result.value.is_finite() {
                return Some(result);
            }
        }
        if !(v1.x.value.is_finite()
            && v1.y.value.is_finite()
            && v1.z.value.is_finite()
            && v2.x.value.is_finite()
            && v2.y.value.is_finite()
            && v2.z.value.is_finite())
        {
            return None;
        }
        let m1 = fast_math::abs(v1.x.value)
            .max(fast_math::abs(v1.y.value))
            .max(fast_math::abs(v1.z.value));
        let m2 = fast_math::abs(v2.x.value)
            .max(fast_math::abs(v2.y.value))
            .max(fast_math::abs(v2.z.value));
        if m1 == 0.0 || m2 == 0.0 {
            return None;
        }
        let scale_grad = |g: Grad<N>, s: i32| -> Grad<N> {
            let mut grad = [0.0; N];
            for i in 0..N {
                grad[i] = libm::ldexp(g.grad[i], s);
            }
            Grad {
                value: libm::ldexp(g.value, s),
                grad,
            }
        };
        let s1 = -exact_exponent(m1);
        let s2 = -exact_exponent(m2);
        let u1 = Self::new(
            scale_grad(v1.x, s1),
            scale_grad(v1.y, s1),
            scale_grad(v1.z, s1),
        );
        let u2 = Self::new(
            scale_grad(v2.x, s2),
            scale_grad(v2.y, s2),
            scale_grad(v2.z, s2),
        );
        let np = u1.norm().mul(u2.norm());
        if np.value == 0.0 || !np.value.is_finite() {
            return None;
        }
        let dot = u1.dot(u2);
        let threshold = np.value * 0.9999;
        if dot.value < -threshold || dot.value > threshold {
            let v3 = u1.cross(u2);
            if dot.value >= 0.0 {
                Some(v3.norm().div(np).asin())
            } else {
                Some(v3.norm().div(np).asin().sub(Grad::PI).negate())
            }
        } else {
            Some(dot.div(np).acos())
        }
    }

    /// `FieldRotation.getIdentity(field).applyTo(self)` (`q0 = 1, q1 = q2 = q3 = 0`).
    #[inline]
    pub(crate) fn apply_identity_rotation(self) -> Self {
        let q0 = Grad::ONE;
        let q1 = Grad::ZERO;
        let q2 = Grad::ZERO;
        let q3 = Grad::ZERO;
        let x = self.x;
        let y = self.y;
        let z = self.z;
        let s = q1.mul(x).add(q2.mul(y)).add(q3.mul(z));
        Self {
            x: q0
                .mul(x.mul(q0).sub(q2.mul(z).sub(q3.mul(y))))
                .add(s.mul(q1))
                .mul_f64(2.0)
                .sub(x),
            y: q0
                .mul(y.mul(q0).sub(q3.mul(x).sub(q1.mul(z))))
                .add(s.mul(q2))
                .mul_f64(2.0)
                .sub(y),
            z: q0
                .mul(z.mul(q0).sub(q1.mul(y).sub(q2.mul(x))))
                .add(s.mul(q3))
                .mul_f64(2.0)
                .sub(z),
        }
    }
}

/// Error-free sum: `a + b == s + err` exactly in IEEE-754 binary64 (Knuth TwoSum).
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let b_virt = s - a;
    let a_virt = s - b_virt;
    let b_err = b - b_virt;
    let a_err = a - a_virt;
    (s, a_err + b_err)
}

/// Error-free product: `a * b == p + err` exactly in IEEE-754 binary64 when no
/// overflow or underflow occurs.
#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let err = libm::fma(a, b, -p);
    (p, err)
}

/// Append `b` to a non-overlapping Shewchuk expansion `e[0..*len]` without losing
/// any bits (Shewchuk `Grow-Expansion`).
#[inline]
fn grow_expansion<const M: usize>(e: &mut [f64; M], len: &mut usize, b: f64) {
    if b == 0.0 {
        return;
    }
    let mut q = b;
    let mut out = 0usize;
    for i in 0..*len {
        let (q_new, h) = two_sum(q, e[i]);
        q = q_new;
        if h != 0.0 {
            e[out] = h;
            out += 1;
        }
    }
    if q != 0.0 {
        e[out] = q;
        out += 1;
    }
    *len = out;
}

/// Compress a non-overlapping Shewchuk expansion `e[0..len]` in place, returning
/// the new length (`0` iff the exact sum is zero).
#[inline]
fn compress_expansion<const M: usize>(e: &mut [f64; M], len: usize) -> usize {
    if len <= 1 {
        return if len == 1 && e[0] != 0.0 { 1 } else { 0 };
    }
    let mut bottom = len - 1;
    let mut q = e[bottom];
    for i in (0..len - 1).rev() {
        let (q_new, h) = two_sum(q, e[i]);
        if h != 0.0 {
            e[bottom] = q_new;
            bottom -= 1;
            q = h;
        } else {
            q = q_new;
        }
    }
    e[bottom] = q;
    let mut top = 0usize;
    for i in (bottom + 1)..len {
        let (q_new, h) = two_sum(e[i], q);
        if h != 0.0 {
            e[top] = h;
            top += 1;
        }
        q = q_new;
    }
    if q != 0.0 {
        e[top] = q;
        top += 1;
    }
    top
}

/// Exact product of two finite nonzero `f64` values as `(p, r, e)` such that
/// `a * b == (p + r) * 2^e` identically with `|p| in [1.0, 2.0)` and `|r| <= 2^-53`.
/// Returns `None` if either factor is zero.
#[inline]
fn exact_prod_normalized(a: f64, b: f64) -> Option<(f64, f64, i32)> {
    if a == 0.0 || b == 0.0 {
        return None;
    }
    let sa = ScaledF64::from_f64(a);
    let sb = ScaledF64::from_f64(b);
    let (mut p, mut r) = two_prod(sa.m, sb.m);
    let mut e = sa.e + sb.e;
    if fast_math::abs(p) >= 2.0 {
        p *= 0.5;
        r *= 0.5;
        e += 1;
    }
    Some((p, r, e))
}

/// Exact predicate: true if and only if `a1 * b1 == a2 * b2` mathematically for
/// finite `f64` inputs (no intermediate overflow or underflow).
#[inline]
pub(crate) fn exact_prod_eq(a1: f64, b1: f64, a2: f64, b2: f64) -> bool {
    match (exact_prod_normalized(a1, b1), exact_prod_normalized(a2, b2)) {
        (None, None) => true,
        (Some((p1, r1, e1)), Some((p2, r2, e2))) => e1 == e2 && p1 == p2 && r1 == r2,
        _ => false,
    }
}

/// Exact predicate: true if and only if `a x b == 0` mathematically for finite
/// 3D `f64` vectors (no intermediate overflow or underflow).
#[inline]
pub(crate) fn exact_collinear3(a: [f64; 3], b: [f64; 3]) -> bool {
    exact_prod_eq(a[1], b[2], a[2], b[1])
        && exact_prod_eq(a[2], b[0], a[0], b[2])
        && exact_prod_eq(a[0], b[1], a[1], b[0])
}

/// Exact predicate: true if and only if `a . b == 0` mathematically for finite
/// 3D `f64` vectors (error-free products and exact Shewchuk expansion sums in
/// exponent-separated arithmetic, so no product underflows or overflows).
#[inline]
pub(crate) fn exact_dot3_is_zero(a: [f64; 3], b: [f64; 3]) -> bool {
    let mut terms = [(0.0, 0.0, 0_i32); 3];
    let mut count = 0usize;
    for i in 0..3 {
        if let Some(t) = exact_prod_normalized(a[i], b[i]) {
            terms[count] = t;
            count += 1;
        }
    }
    match count {
        0 => true,
        1 => false,
        2 => {
            let (p0, r0, e0) = terms[0];
            let (p1, r1, e1) = terms[1];
            e0 == e1 && p0 == -p1 && r0 == -r1
        }
        _ => {
            // Sort the 3 terms by exponent descending: e0 >= e1 >= e2.
            if terms[0].2 < terms[1].2 {
                terms.swap(0, 1);
            }
            if terms[1].2 < terms[2].2 {
                terms.swap(1, 2);
            }
            if terms[0].2 < terms[1].2 {
                terms.swap(0, 1);
            }
            let (p0, r0, e0) = terms[0];
            let (p1, r1, e1) = terms[1];
            let (p2, r2, e2) = terms[2];
            // |term0| >= 2^e0 while |term1 + term2| < 4 * 2^e1; if e0 - e1 > 110
            // they cannot cancel.
            if e0 - e1 > 110 {
                return false;
            }
            let mut exp = [0.0; 6];
            let mut len = 0usize;
            grow_expansion(&mut exp, &mut len, p0);
            grow_expansion(&mut exp, &mut len, r0);
            let d1 = e1 - e0;
            grow_expansion(&mut exp, &mut len, libm::ldexp(p1, d1));
            grow_expansion(&mut exp, &mut len, libm::ldexp(r1, d1));
            len = compress_expansion(&mut exp, len);
            if len == 0 {
                // term0 + term1 == 0, so sum == term2 != 0.
                return false;
            }
            let e_top = e0 + exact_exponent(exp[len - 1]);
            // Compressed leading component bounds |term0 + term1| within (0.5, 2) * 2^e_top
            // while |term2| is in [2^e2, 2^(e2+1)), so |e_top - e2| > 4 cannot cancel.
            if (e_top - e2).abs() > 4 {
                return false;
            }
            let shift_exp = e0 - e_top;
            for v in &mut exp[..len] {
                *v = libm::ldexp(*v, shift_exp);
            }
            let d2 = e2 - e_top;
            grow_expansion(&mut exp, &mut len, libm::ldexp(p2, d2));
            grow_expansion(&mut exp, &mut len, libm::ldexp(r2, d2));
            compress_expansion(&mut exp, len) == 0
        }
    }
}

/// Exact unbiased binary exponent `e` such that `|x| * 2^-e` is in `[1.0, 2.0)`
/// for any finite nonzero `x` (normal or subnormal).
#[inline]
pub(crate) fn exact_exponent(x: f64) -> i32 {
    let a = fast_math::abs(x);
    if a < fast_math::SAFE_MIN {
        fast_math::get_exponent(a * ((1_u64 << 54) as f64)) - 54
    } else {
        fast_math::get_exponent(a)
    }
}

/// Binary64 mantissa paired with an unbounded `i32` base-2 exponent
/// (`value == m * 2^e`, with `m == 0.0` or `|m| in [1.0, 2.0)` when finite).
///
/// Used for overflow/underflow-free arithmetic in ECOM1 and as a fallback when
/// unscaled Hipparchus arithmetic loses a finite intermediate.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScaledF64 {
    pub(crate) m: f64,
    pub(crate) e: i32,
}

impl ScaledF64 {
    pub(crate) const ZERO: Self = Self { m: 0.0, e: 0 };

    #[inline]
    pub(crate) fn from_f64(x: f64) -> Self {
        if x == 0.0 || !x.is_finite() {
            Self { m: x, e: 0 }
        } else {
            let e = exact_exponent(x);
            Self {
                m: libm::ldexp(x, -e),
                e,
            }
        }
    }

    #[inline]
    pub(crate) fn from_scaled(m: f64, e: i32) -> Self {
        if m == 0.0 || !m.is_finite() {
            Self { m, e: 0 }
        } else {
            let de = exact_exponent(m);
            Self {
                m: libm::ldexp(m, -de),
                e: e.saturating_add(de),
            }
        }
    }

    #[inline]
    pub(crate) fn to_f64(self) -> f64 {
        if self.m == 0.0 || !self.m.is_finite() {
            self.m
        } else {
            libm::ldexp(self.m, self.e.clamp(-2200, 2200))
        }
    }

    #[inline]
    pub(crate) fn sqrt(self) -> Self {
        let exponent = self.e.div_euclid(2);
        let mantissa = libm::ldexp(self.m, self.e.rem_euclid(2));
        Self::from_scaled(mantissa.sqrt(), exponent)
    }

    #[inline]
    pub(crate) fn is_zero(self) -> bool {
        self.m == 0.0
    }

    #[inline]
    pub(crate) fn is_finite(self) -> bool {
        self.m.is_finite()
    }

    #[inline]
    pub(crate) fn negate(self) -> Self {
        Self {
            m: -self.m,
            e: self.e,
        }
    }

    #[inline]
    pub(crate) fn mul(self, rhs: Self) -> Self {
        if !self.is_finite() || !rhs.is_finite() {
            return Self {
                m: self.m * rhs.m,
                e: 0,
            };
        }
        if self.is_zero() || rhs.is_zero() {
            return Self::ZERO;
        }
        Self::from_scaled(self.m * rhs.m, self.e.saturating_add(rhs.e))
    }

    #[inline]
    pub(crate) fn div(self, rhs: Self) -> Self {
        if !self.is_finite() || !rhs.is_finite() || rhs.is_zero() {
            return Self {
                m: self.m / rhs.m,
                e: 0,
            };
        }
        if self.is_zero() {
            return Self::ZERO;
        }
        Self::from_scaled(self.m / rhs.m, self.e.saturating_sub(rhs.e))
    }

    #[inline]
    pub(crate) fn add(self, rhs: Self) -> Self {
        if !self.is_finite() || !rhs.is_finite() {
            return Self {
                m: self.m + rhs.m,
                e: 0,
            };
        }
        if self.is_zero() {
            return rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        let e_max = self.e.max(rhs.e);
        let da = self.e - e_max;
        let db = rhs.e - e_max;
        if da < -60 {
            return rhs;
        }
        if db < -60 {
            return self;
        }
        Self::from_scaled(libm::ldexp(self.m, da) + libm::ldexp(rhs.m, db), e_max)
    }

    #[inline]
    pub(crate) fn sub(self, rhs: Self) -> Self {
        self.add(rhs.negate())
    }

    /// Euclidean norm `sqrt(x^2 + y^2)` in `ScaledF64` arithmetic without overflow or underflow.
    #[inline]
    pub(crate) fn hypot(y: Self, x: Self) -> Option<Self> {
        if !(y.is_finite() && x.is_finite()) {
            return None;
        }
        if y.is_zero() {
            return (!x.is_zero()).then_some(Self {
                m: x.m.abs(),
                e: x.e,
            });
        }
        if x.is_zero() {
            return Some(Self {
                m: y.m.abs(),
                e: y.e,
            });
        }
        let e_max = y.e.max(x.e);
        let dy = y.e - e_max;
        let dx = x.e - e_max;
        if dy < -60 {
            return Some(Self {
                m: x.m.abs(),
                e: x.e,
            });
        }
        if dx < -60 {
            return Some(Self {
                m: y.m.abs(),
                e: y.e,
            });
        }
        let sy = libm::ldexp(y.m, dy);
        let sx = libm::ldexp(x.m, dx);
        let h = libm::hypot(sy, sx);
        (h > 0.0 && h.is_finite()).then_some(Self::from_scaled(h, e_max))
    }

    /// Compute `(sin u, cos u) = (y / rho, x / rho)` with `rho = hypot(x, y)` directly in
    /// `ScaledF64` arithmetic without binary64 `u`, `sin`, or `cos` intermediates.
    #[inline]
    pub(crate) fn sin_cos_from_yx(y: Self, x: Self) -> Option<(Self, Self)> {
        let rho = Self::hypot(y, x)?;
        Some((y.div(rho), x.div(rho)))
    }

    /// Compensated two-term linear combination `a1 * b1 + a2 * b2` without overflow or underflow.
    ///
    /// Uses error-free products (`two_prod`) and an exact Shewchuk expansion sum on
    /// exponent-aligned mantissas, so the result is `ScaledF64::ZERO` if and only if
    /// `a1 * b1 + a2 * b2 == 0` mathematically.
    #[inline]
    pub(crate) fn lincomb2(a1: Self, b1: Self, a2: Self, b2: Self) -> Self {
        if !(a1.is_finite() && b1.is_finite() && a2.is_finite() && b2.is_finite()) {
            return Self { m: f64::NAN, e: 0 };
        }
        let z1 = a1.is_zero() || b1.is_zero();
        let z2 = a2.is_zero() || b2.is_zero();
        if z1 && z2 {
            return Self::ZERO;
        }
        if z1 {
            return a2.mul(b2);
        }
        if z2 {
            return a1.mul(b1);
        }
        let e1 = a1.e.saturating_add(b1.e);
        let e2 = a2.e.saturating_add(b2.e);
        if e1 - e2 > 60 {
            return a1.mul(b1);
        }
        if e2 - e1 > 60 {
            return a2.mul(b2);
        }
        let (e_ref, (p1, r1), (p2, r2)) = if e1 >= e2 {
            (
                e1,
                two_prod(a1.m, b1.m),
                two_prod(a2.m, libm::ldexp(b2.m, e2 - e1)),
            )
        } else {
            (
                e2,
                two_prod(a1.m, libm::ldexp(b1.m, e1 - e2)),
                two_prod(a2.m, b2.m),
            )
        };
        let mut exp = [0.0; 4];
        let mut len = 0usize;
        grow_expansion(&mut exp, &mut len, p1);
        grow_expansion(&mut exp, &mut len, r1);
        grow_expansion(&mut exp, &mut len, p2);
        grow_expansion(&mut exp, &mut len, r2);
        let top = compress_expansion(&mut exp, len);
        if top == 0 {
            return Self::ZERO;
        }
        let mut sum = 0.0;
        for &v in &exp[..top] {
            sum += v;
        }
        Self::from_scaled(sum, e_ref)
    }

    /// Compensated sum of products over `K` terms without overflow, underflow, or
    /// large-term cancellation hiding a smaller remaining term.
    #[inline]
    pub(crate) fn lincomb_slice<const K: usize>(pairs: [(Self, Self); K]) -> Self {
        let mut active_a = [0.0; K];
        let mut active_b = [0.0; K];
        let mut active_e = [0_i32; K];
        let mut count = 0usize;
        for (a, b) in pairs {
            if !a.is_finite() || !b.is_finite() {
                return Self { m: f64::NAN, e: 0 };
            }
            if !a.is_zero() && !b.is_zero() {
                active_a[count] = a.m;
                active_b[count] = b.m;
                active_e[count] = a.e.saturating_add(b.e);
                count += 1;
            }
        }
        if count == 0 {
            return Self::ZERO;
        }
        let mut acc = Self::ZERO;
        let mut used = [false; K];
        loop {
            let mut e_max = i32::MIN;
            let mut rem = 0usize;
            for i in 0..count {
                if !used[i] {
                    e_max = e_max.max(active_e[i]);
                    rem += 1;
                }
            }
            if rem == 0 {
                return acc;
            }
            // Group every remaining term within 110 bits of e_max so shifts are exact.
            let mut ga = [0.0; K];
            let mut gb = [0.0; K];
            let mut g_count = 0usize;
            for i in 0..count {
                if !used[i] && active_e[i] >= e_max - 110 {
                    ga[g_count] = active_a[i];
                    gb[g_count] = libm::ldexp(active_b[i], active_e[i] - e_max);
                    g_count += 1;
                    used[i] = true;
                }
            }
            let group_val = match g_count {
                1 => ga[0] * gb[0],
                2 => fast_math::linear_combination_2(ga[0], gb[0], ga[1], gb[1]),
                3 => fast_math::linear_combination_3(ga[0], gb[0], ga[1], gb[1], ga[2], gb[2]),
                4 => fast_math::linear_combination_4(
                    ga[0], gb[0], ga[1], gb[1], ga[2], gb[2], ga[3], gb[3],
                ),
                _ => {
                    let mut a_arr = [0.0; K];
                    let mut b_arr = [0.0; K];
                    a_arr[..g_count].copy_from_slice(&ga[..g_count]);
                    b_arr[..g_count].copy_from_slice(&gb[..g_count]);
                    fast_math::linear_combination_arr(&a_arr, &b_arr)
                }
            };
            acc = acc.add(Self::from_scaled(group_val, e_max));
        }
    }
}

/// 3D vector of [`ScaledF64`] components for overflow/underflow-free fallback geometry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScaledVec3 {
    pub(crate) x: ScaledF64,
    pub(crate) y: ScaledF64,
    pub(crate) z: ScaledF64,
}

impl ScaledVec3 {
    pub(crate) const ZERO: Self = Self {
        x: ScaledF64::ZERO,
        y: ScaledF64::ZERO,
        z: ScaledF64::ZERO,
    };

    #[inline]
    pub(crate) fn from_vec3(v: Vec3) -> Self {
        Self {
            x: ScaledF64::from_f64(v.x),
            y: ScaledF64::from_f64(v.y),
            z: ScaledF64::from_f64(v.z),
        }
    }

    #[inline]
    pub(crate) fn to_vec3(self) -> Option<Vec3> {
        let v = Vec3::new(self.x.to_f64(), self.y.to_f64(), self.z.to_f64());
        (v.x.is_finite() && v.y.is_finite() && v.z.is_finite()).then_some(v)
    }

    #[inline]
    pub(crate) fn scalar_mul(self, a: ScaledF64) -> Self {
        Self {
            x: self.x.mul(a),
            y: self.y.mul(a),
            z: self.z.mul(a),
        }
    }

    #[inline]
    pub(crate) fn scalar_div(self, a: ScaledF64) -> Self {
        Self {
            x: self.x.div(a),
            y: self.y.div(a),
            z: self.z.div(a),
        }
    }

    #[inline]
    pub(crate) fn lincomb3(
        a1: ScaledF64,
        u1: Self,
        a2: ScaledF64,
        u2: Self,
        a3: ScaledF64,
        u3: Self,
    ) -> Self {
        Self {
            x: ScaledF64::lincomb_slice([(a1, u1.x), (a2, u2.x), (a3, u3.x)]),
            y: ScaledF64::lincomb_slice([(a1, u1.y), (a2, u2.y), (a3, u3.y)]),
            z: ScaledF64::lincomb_slice([(a1, u1.z), (a2, u2.z), (a3, u3.z)]),
        }
    }

    #[inline]
    pub(crate) fn sub(self, v: Self) -> Self {
        Self {
            x: self.x.sub(v.x),
            y: self.y.sub(v.y),
            z: self.z.sub(v.z),
        }
    }

    #[inline]
    pub(crate) fn dot(self, v: Self) -> ScaledF64 {
        ScaledF64::lincomb_slice([(self.x, v.x), (self.y, v.y), (self.z, v.z)])
    }

    #[inline]
    pub(crate) fn cross(self, v: Self) -> Self {
        Self {
            x: ScaledF64::lincomb2(self.y, v.z, self.z.negate(), v.y),
            y: ScaledF64::lincomb2(self.z, v.x, self.x.negate(), v.z),
            z: ScaledF64::lincomb2(self.x, v.y, self.y.negate(), v.x),
        }
    }

    /// Euclidean norm in [`ScaledF64`] arithmetic, returning `None` if zero or non-finite.
    #[inline]
    pub(crate) fn norm(self) -> Option<ScaledF64> {
        if !(self.x.is_finite() && self.y.is_finite() && self.z.is_finite()) {
            return None;
        }
        let mut e_max = i32::MIN;
        for c in [self.x, self.y, self.z] {
            if !c.is_zero() {
                e_max = e_max.max(c.e);
            }
        }
        if e_max == i32::MIN {
            return None;
        }
        let shift_comp = |c: ScaledF64| -> f64 {
            if c.is_zero() {
                0.0
            } else {
                let de = c.e - e_max + 508;
                if de < -1080 {
                    0.0
                } else {
                    libm::ldexp(c.m, de)
                }
            }
        };
        let v_scaled = Vec3::new(shift_comp(self.x), shift_comp(self.y), shift_comp(self.z));
        let s_scaled = v_scaled.norm();
        if s_scaled == 0.0 || !s_scaled.is_finite() {
            return None;
        }
        Some(ScaledF64::from_scaled(s_scaled, e_max - 508))
    }

    /// Normalize to a [`ScaledVec3`] unit vector plus its [`ScaledF64`] Euclidean norm,
    /// keeping exact exponents without flushing small components through binary64.
    #[inline]
    pub(crate) fn normalize_scaled(self) -> Option<(Self, ScaledF64)> {
        let norm = self.norm()?;
        let u = Self {
            x: self.x.div(norm),
            y: self.y.div(norm),
            z: self.z.div(norm),
        };
        Some((u, norm))
    }

    /// Normalize to a finite `Vec3` unit vector plus its [`ScaledF64`] Euclidean norm.
    #[inline]
    pub(crate) fn normalize(self) -> Option<(Vec3, ScaledF64)> {
        let (u_s, norm) = self.normalize_scaled()?;
        let u = Vec3::new(u_s.x.to_f64(), u_s.y.to_f64(), u_s.z.to_f64());
        if u.x.is_finite() && u.y.is_finite() && u.z.is_finite() {
            Some((u, norm))
        } else {
            None
        }
    }
}

/// First-order forward-mode derivative in [`ScaledF64`] arithmetic.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScaledGrad<const N: usize> {
    pub(crate) val: ScaledF64,
    pub(crate) grad: [ScaledF64; N],
}

impl<const N: usize> ScaledGrad<N> {
    pub(crate) const ZERO: Self = Self {
        val: ScaledF64::ZERO,
        grad: [ScaledF64::ZERO; N],
    };

    #[inline]
    pub(crate) fn from_grad(g: Grad<N>) -> Self {
        let mut grad = [ScaledF64::ZERO; N];
        for i in 0..N {
            grad[i] = ScaledF64::from_f64(g.grad[i]);
        }
        Self {
            val: ScaledF64::from_f64(g.value),
            grad,
        }
    }

    #[inline]
    pub(crate) fn to_grad(self) -> Option<Grad<N>> {
        let value = self.val.to_f64();
        if !value.is_finite() {
            return None;
        }
        let mut grad = [0.0; N];
        for i in 0..N {
            let g = self.grad[i].to_f64();
            if !g.is_finite() {
                return None;
            }
            grad[i] = g;
        }
        Some(Grad { value, grad })
    }

    #[inline]
    pub(crate) fn negate(self) -> Self {
        let mut grad = [ScaledF64::ZERO; N];
        for i in 0..N {
            grad[i] = self.grad[i].negate();
        }
        Self {
            val: self.val.negate(),
            grad,
        }
    }

    #[inline]
    pub(crate) fn sub(self, rhs: Self) -> Self {
        let mut grad = [ScaledF64::ZERO; N];
        for i in 0..N {
            grad[i] = self.grad[i].sub(rhs.grad[i]);
        }
        Self {
            val: self.val.sub(rhs.val),
            grad,
        }
    }

    #[inline]
    pub(crate) fn add(self, rhs: Self) -> Self {
        let mut grad = [ScaledF64::ZERO; N];
        for i in 0..N {
            grad[i] = self.grad[i].add(rhs.grad[i]);
        }
        Self {
            val: self.val.add(rhs.val),
            grad,
        }
    }

    #[inline]
    pub(crate) fn mul(self, rhs: Self) -> Self {
        let mut grad = [ScaledF64::ZERO; N];
        for i in 0..N {
            grad[i] = ScaledF64::lincomb2(self.grad[i], rhs.val, self.val, rhs.grad[i]);
        }
        Self {
            val: self.val.mul(rhs.val),
            grad,
        }
    }

    /// The ordered two products in Hipparchus `Gradient.multiply`.
    #[inline]
    pub(crate) fn mul_ordered(self, rhs: Self) -> Self {
        Self {
            val: self.val.mul(rhs.val),
            grad: core::array::from_fn(|i| {
                self.grad[i].mul(rhs.val).add(self.val.mul(rhs.grad[i]))
            }),
        }
    }

    #[inline]
    pub(crate) fn mul_scalar(self, scalar: ScaledF64) -> Self {
        Self {
            val: self.val.mul(scalar),
            grad: self.grad.map(|entry| entry.mul(scalar)),
        }
    }

    #[inline]
    pub(crate) fn sub_scalar(self, scalar: f64) -> Self {
        Self {
            val: self.val.sub(ScaledF64::from_f64(scalar)),
            grad: self.grad,
        }
    }

    #[inline]
    pub(crate) fn div(self, rhs: Self) -> Self {
        let inverse = ScaledF64::from_f64(1.0).div(rhs.val);
        let inverse_squared = inverse.mul(inverse);
        Self {
            val: self.val.mul(inverse),
            grad: core::array::from_fn(|i| {
                self.grad[i]
                    .mul(rhs.val)
                    .sub(self.val.mul(rhs.grad[i]))
                    .mul(inverse_squared)
            }),
        }
    }

    #[inline]
    pub(crate) fn square(self) -> Self {
        self.compose(
            self.val.mul(self.val),
            ScaledF64::from_f64(2.0).mul(self.val),
        )
    }

    #[inline]
    pub(crate) fn reciprocal(self) -> Self {
        let inverse = ScaledF64::from_f64(1.0).div(self.val);
        self.compose(inverse, inverse.negate().mul(inverse))
    }

    #[inline]
    pub(crate) fn sqrt(self) -> Self {
        let root = self.val.sqrt();
        self.compose(
            root,
            ScaledF64::from_f64(1.0).div(ScaledF64::from_f64(2.0).mul(root)),
        )
    }

    #[inline]
    fn compose(self, val: ScaledF64, slope: ScaledF64) -> Self {
        Self {
            val,
            grad: self.grad.map(|entry| slope.mul(entry)),
        }
    }

    #[inline]
    pub(crate) fn is_all_finite(self) -> bool {
        self.val.is_finite() && self.grad.iter().all(|g| g.is_finite())
    }

    #[inline]
    pub(crate) fn div_scaled(self, rhs: ScaledF64) -> Self {
        let mut grad = [ScaledF64::ZERO; N];
        for i in 0..N {
            grad[i] = self.grad[i].div(rhs);
        }
        Self {
            val: self.val.div(rhs),
            grad,
        }
    }

    /// Compute `(sin u, cos u) = (y / rho, x / rho)` with `rho = hypot(x, y)` and their
    /// forward-mode derivatives directly in `ScaledF64` arithmetic, without binary64
    /// `u`, `sin`, or `cos` intermediates.
    #[inline]
    pub(crate) fn sin_cos_from_yx(self, x: Self) -> Option<(Self, Self)> {
        let (s, c) = ScaledF64::sin_cos_from_yx(self.val, x.val)?;
        let rho_sq = ScaledF64::lincomb2(x.val, x.val, self.val, self.val);
        if !rho_sq.is_finite() || rho_sq.is_zero() {
            return None;
        }
        let neg_s = s.negate();
        let mut grad_sin = [ScaledF64::ZERO; N];
        let mut grad_cos = [ScaledF64::ZERO; N];
        for i in 0..N {
            if !(self.grad[i].is_finite() && x.grad[i].is_finite()) {
                return None;
            }
            let num = ScaledF64::lincomb2(x.val, self.grad[i], self.val.negate(), x.grad[i]);
            let du = num.div(rho_sq);
            if !du.is_finite() {
                return None;
            }
            grad_sin[i] = c.mul(du);
            grad_cos[i] = neg_s.mul(du);
        }
        Some((
            Self {
                val: s,
                grad: grad_sin,
            },
            Self {
                val: c,
                grad: grad_cos,
            },
        ))
    }

    #[inline]
    pub(crate) fn sin_cos_from_scaled(
        angle_val: f64,
        angle_grad: [ScaledF64; N],
        sin_cos_fn: fn(f64) -> (f64, f64),
    ) -> (Self, Self) {
        let (s, c) = sin_cos_fn(angle_val);
        let s_s = ScaledF64::from_f64(s);
        let c_s = ScaledF64::from_f64(c);
        let neg_s_s = s_s.negate();
        let mut grad_sin = [ScaledF64::ZERO; N];
        let mut grad_cos = [ScaledF64::ZERO; N];
        for i in 0..N {
            grad_sin[i] = angle_grad[i].mul(c_s);
            grad_cos[i] = angle_grad[i].mul(neg_s_s);
        }
        (
            Self {
                val: s_s,
                grad: grad_sin,
            },
            Self {
                val: c_s,
                grad: grad_cos,
            },
        )
    }

    #[inline]
    pub(crate) fn atan2_scaled(
        self,
        x: Self,
        atan2_fn: fn(f64, f64) -> f64,
    ) -> Option<(f64, [ScaledF64; N])> {
        if !self.val.is_finite() || !x.val.is_finite() {
            return None;
        }
        let e_max = match (self.val.is_zero(), x.val.is_zero()) {
            (true, true) => return None,
            (true, false) => x.val.e,
            (false, true) => self.val.e,
            (false, false) => self.val.e.max(x.val.e),
        };
        let sy = if self.val.is_zero() {
            0.0
        } else {
            libm::ldexp(self.val.m, (self.val.e - e_max).max(-1100))
        };
        let sx = if x.val.is_zero() {
            0.0
        } else {
            libm::ldexp(x.val.m, (x.val.e - e_max).max(-1100))
        };
        let value = atan2_fn(sy, sx);
        if !value.is_finite() {
            return None;
        }
        let denom = ScaledF64::from_scaled(sy * sy + sx * sx, e_max.saturating_mul(2));
        let mut grad = [ScaledF64::ZERO; N];
        for i in 0..N {
            let num = ScaledF64::lincomb2(x.val, self.grad[i], self.val.negate(), x.grad[i]);
            let g = num.div(denom);
            if !g.is_finite() {
                return None;
            }
            grad[i] = g;
        }
        Some((value, grad))
    }
}

/// 3D vector of [`ScaledGrad`] components for overflow/underflow-free fallback AD.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScaledFieldVec3<const N: usize> {
    pub(crate) x: ScaledGrad<N>,
    pub(crate) y: ScaledGrad<N>,
    pub(crate) z: ScaledGrad<N>,
}

impl<const N: usize> ScaledFieldVec3<N> {
    pub(crate) const ZERO: Self = Self {
        x: ScaledGrad::ZERO,
        y: ScaledGrad::ZERO,
        z: ScaledGrad::ZERO,
    };

    #[inline]
    pub(crate) fn from_field_vec3(v: FieldVec3<N>) -> Self {
        Self {
            x: ScaledGrad::from_grad(v.x),
            y: ScaledGrad::from_grad(v.y),
            z: ScaledGrad::from_grad(v.z),
        }
    }

    #[inline]
    pub(crate) fn to_field_vec3(self) -> Option<FieldVec3<N>> {
        Some(FieldVec3::new(
            self.x.to_grad()?,
            self.y.to_grad()?,
            self.z.to_grad()?,
        ))
    }

    #[inline]
    pub(crate) fn val_vec(self) -> ScaledVec3 {
        ScaledVec3 {
            x: self.x.val,
            y: self.y.val,
            z: self.z.val,
        }
    }

    #[inline]
    pub(crate) fn scalar_mul(self, a: ScaledGrad<N>) -> Self {
        Self {
            x: self.x.mul(a),
            y: self.y.mul(a),
            z: self.z.mul(a),
        }
    }

    #[inline]
    pub(crate) fn scalar_div(self, a: ScaledF64) -> Self {
        Self {
            x: self.x.div_scaled(a),
            y: self.y.div_scaled(a),
            z: self.z.div_scaled(a),
        }
    }

    #[inline]
    pub(crate) fn lincomb3(
        a1: ScaledGrad<N>,
        u1: Self,
        a2: ScaledGrad<N>,
        u2: Self,
        a3: ScaledGrad<N>,
        u3: Self,
    ) -> Self {
        let comp = |c1: ScaledGrad<N>, c2: ScaledGrad<N>, c3: ScaledGrad<N>| -> ScaledGrad<N> {
            let val =
                ScaledF64::lincomb_slice([(a1.val, c1.val), (a2.val, c2.val), (a3.val, c3.val)]);
            let mut grad = [ScaledF64::ZERO; N];
            for i in 0..N {
                grad[i] = ScaledF64::lincomb_slice([
                    (a1.val, c1.grad[i]),
                    (a1.grad[i], c1.val),
                    (a2.val, c2.grad[i]),
                    (a2.grad[i], c2.val),
                    (a3.val, c3.grad[i]),
                    (a3.grad[i], c3.val),
                ]);
            }
            ScaledGrad { val, grad }
        };
        Self {
            x: comp(u1.x, u2.x, u3.x),
            y: comp(u1.y, u2.y, u3.y),
            z: comp(u1.z, u2.z, u3.z),
        }
    }

    #[inline]
    pub(crate) fn sub(self, v: Self) -> Self {
        Self {
            x: self.x.sub(v.x),
            y: self.y.sub(v.y),
            z: self.z.sub(v.z),
        }
    }

    #[inline]
    pub(crate) fn dot(self, v: Self) -> ScaledGrad<N> {
        let val = self.val_vec().dot(v.val_vec());
        let mut grad = [ScaledF64::ZERO; N];
        for i in 0..N {
            grad[i] = ScaledF64::lincomb_slice([
                (self.x.val, v.x.grad[i]),
                (self.x.grad[i], v.x.val),
                (self.y.val, v.y.grad[i]),
                (self.y.grad[i], v.y.val),
                (self.z.val, v.z.grad[i]),
                (self.z.grad[i], v.z.val),
            ]);
        }
        ScaledGrad { val, grad }
    }

    #[inline]
    pub(crate) fn cross(self, v: Self) -> Self {
        let val = self.val_vec().cross(v.val_vec());
        let mut gx = [ScaledF64::ZERO; N];
        let mut gy = [ScaledF64::ZERO; N];
        let mut gz = [ScaledF64::ZERO; N];
        for i in 0..N {
            gx[i] = ScaledF64::lincomb_slice([
                (self.y.val, v.z.grad[i]),
                (self.y.grad[i], v.z.val),
                (self.z.val.negate(), v.y.grad[i]),
                (self.z.grad[i].negate(), v.y.val),
            ]);
            gy[i] = ScaledF64::lincomb_slice([
                (self.z.val, v.x.grad[i]),
                (self.z.grad[i], v.x.val),
                (self.x.val.negate(), v.z.grad[i]),
                (self.x.grad[i].negate(), v.z.val),
            ]);
            gz[i] = ScaledF64::lincomb_slice([
                (self.x.val, v.y.grad[i]),
                (self.x.grad[i], v.y.val),
                (self.y.val.negate(), v.x.grad[i]),
                (self.y.grad[i].negate(), v.x.val),
            ]);
        }
        Self {
            x: ScaledGrad {
                val: val.x,
                grad: gx,
            },
            y: ScaledGrad {
                val: val.y,
                grad: gy,
            },
            z: ScaledGrad {
                val: val.z,
                grad: gz,
            },
        }
    }

    #[inline]
    pub(crate) fn normalize(self) -> Option<Self> {
        let (u_s, norm) = self.val_vec().normalize_scaled()?;
        let mut gx = [ScaledF64::ZERO; N];
        let mut gy = [ScaledF64::ZERO; N];
        let mut gz = [ScaledF64::ZERO; N];
        for i in 0..N {
            let dv = ScaledVec3 {
                x: self.x.grad[i],
                y: self.y.grad[i],
                z: self.z.grad[i],
            };
            if !(dv.x.is_finite() && dv.y.is_finite() && dv.z.is_finite()) {
                return None;
            }
            if dv.x.is_zero() && dv.y.is_zero() && dv.z.is_zero() {
                continue;
            }
            let proj = u_s.dot(dv);
            gx[i] = dv.x.sub(u_s.x.mul(proj)).div(norm);
            gy[i] = dv.y.sub(u_s.y.mul(proj)).div(norm);
            gz[i] = dv.z.sub(u_s.z.mul(proj)).div(norm);
        }
        Some(Self {
            x: ScaledGrad {
                val: u_s.x,
                grad: gx,
            },
            y: ScaledGrad {
                val: u_s.y,
                grad: gy,
            },
            z: ScaledGrad {
                val: u_s.z,
                grad: gz,
            },
        })
    }
}
