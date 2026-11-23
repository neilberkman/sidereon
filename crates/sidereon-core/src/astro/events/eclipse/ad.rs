//! Forward-mode automatic differentiation (`Grad`) and 3D vector arithmetic
//! (`Vec3`, `FieldVec3`) reproducing Hipparchus 4.0.3's `Gradient`,
//! `Vector3D`, and `FieldVector3D<Gradient>`.

use super::fast_math;

/// First-order forward-mode derivative with three free parameters, reproducing
/// Hipparchus 4.0.3 `org.hipparchus.analysis.differentiation.Gradient`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Grad {
    pub(crate) value: f64,
    pub(crate) grad: [f64; 3],
}

impl Grad {
    pub(crate) const ZERO: Self = Self {
        value: 0.0,
        grad: [0.0; 3],
    };

    pub(crate) const ONE: Self = Self {
        value: 1.0,
        grad: [0.0; 3],
    };

    pub(crate) const PI: Self = Self {
        value: fast_math::PI,
        grad: [0.0; 3],
    };

    #[inline]
    pub(crate) const fn constant(value: f64) -> Self {
        Self {
            value,
            grad: [0.0; 3],
        }
    }

    #[inline]
    pub(crate) const fn variable(index: usize, value: f64) -> Self {
        let mut grad = [0.0; 3];
        grad[index] = 1.0;
        Self { value, grad }
    }

    #[inline]
    pub(crate) fn is_nan(self) -> bool {
        self.value.is_nan()
    }

    /// Hipparchus `Gradient.equals(ZERO)` (`FieldElement.isZero()`).
    #[inline]
    pub(crate) fn is_zero(self) -> bool {
        self.value == 0.0
            && fast_math::precision_equals_1ulp(self.grad[0], 0.0)
            && fast_math::precision_equals_1ulp(self.grad[1], 0.0)
            && fast_math::precision_equals_1ulp(self.grad[2], 0.0)
    }

    #[inline]
    pub(crate) fn is_all_finite(self) -> bool {
        self.value.is_finite()
            && self.grad[0].is_finite()
            && self.grad[1].is_finite()
            && self.grad[2].is_finite()
    }

    #[inline]
    pub(crate) fn compose(self, f0: f64, f1: f64) -> Self {
        Self {
            value: f0,
            grad: [f1 * self.grad[0], f1 * self.grad[1], f1 * self.grad[2]],
        }
    }

    #[inline]
    pub(crate) fn add(self, a: Self) -> Self {
        Self {
            value: self.value + a.value,
            grad: [
                self.grad[0] + a.grad[0],
                self.grad[1] + a.grad[1],
                self.grad[2] + a.grad[2],
            ],
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
        Self {
            value: self.value - a.value,
            grad: [
                self.grad[0] - a.grad[0],
                self.grad[1] - a.grad[1],
                self.grad[2] - a.grad[2],
            ],
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
        Self {
            value: self.value * a,
            grad: [self.grad[0] * a, self.grad[1] * a, self.grad[2] * a],
        }
    }

    #[inline]
    pub(crate) fn mul(self, a: Self) -> Self {
        Self {
            value: self.value * a.value,
            grad: [
                self.grad[0] * a.value + self.value * a.grad[0],
                self.grad[1] * a.value + self.value * a.grad[1],
                self.grad[2] * a.value + self.value * a.grad[2],
            ],
        }
    }

    #[inline]
    pub(crate) fn div_f64(self, a: f64) -> Self {
        Self {
            value: self.value / a,
            grad: [self.grad[0] / a, self.grad[1] / a, self.grad[2] / a],
        }
    }

    /// Hipparchus `Gradient.divide(Gradient)`: computes `inv1 = 1 / a.value`,
    /// `value = self.value * inv1`, and `(grad * a.value - value * a.grad) * inv1^2`.
    #[inline]
    pub(crate) fn div(self, a: Self) -> Self {
        let inv1 = 1.0 / a.value;
        let inv2 = inv1 * inv1;
        Self {
            value: self.value * inv1,
            grad: [
                (self.grad[0] * a.value - self.value * a.grad[0]) * inv2,
                (self.grad[1] * a.value - self.value * a.grad[1]) * inv2,
                (self.grad[2] * a.value - self.value * a.grad[2]) * inv2,
            ],
        }
    }

    #[inline]
    pub(crate) fn negate(self) -> Self {
        Self {
            value: -self.value,
            grad: [-self.grad[0], -self.grad[1], -self.grad[2]],
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
        let sin = Self {
            value: s,
            grad: [self.grad[0] * c, self.grad[1] * c, self.grad[2] * c],
        };
        let cos = Self {
            value: c,
            grad: [-self.grad[0] * s, -self.grad[1] * s, -self.grad[2] * s],
        };
        (sin, cos)
    }

    #[inline]
    pub(crate) fn atan2(self, x: Self) -> Self {
        let inv = 1.0 / (self.value * self.value + x.value * x.value);
        Self {
            value: fast_math::atan2(self.value, x.value),
            grad: [
                (x.value * self.grad[0] - x.grad[0] * self.value) * inv,
                (x.value * self.grad[1] - x.grad[1] * self.value) * inv,
                (x.value * self.grad[2] - x.grad[2] * self.value) * inv,
            ],
        }
    }

    #[inline]
    pub(crate) fn scalb(self, n: i32) -> Self {
        Self {
            value: fast_math::scalb(self.value, n),
            grad: [
                fast_math::scalb(self.grad[0], n),
                fast_math::scalb(self.grad[1], n),
                fast_math::scalb(self.grad[2], n),
            ],
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
        let mut grad = [0.0; 3];
        for i in 0..3 {
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
        let mut grad = [0.0; 3];
        for i in 0..3 {
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
        let mut grad = [0.0; 3];
        for i in 0..3 {
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
            None
        } else {
            Some(self.scalar_mul(1.0 / s))
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
        let threshold = 0.6 * self.norm();
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
        let dot = v1.dot(v2);
        let threshold = norm_product * 0.9999;
        if dot < -threshold || dot > threshold {
            let v3 = v1.cross(v2);
            if dot >= 0.0 {
                Some(fast_math::asin(v3.norm() / norm_product))
            } else {
                Some(fast_math::PI - fast_math::asin(v3.norm() / norm_product))
            }
        } else {
            Some(fast_math::acos(dot / norm_product))
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
pub(crate) struct FieldVec3 {
    pub(crate) x: Grad,
    pub(crate) y: Grad,
    pub(crate) z: Grad,
}

impl FieldVec3 {
    pub(crate) const ZERO: Self = Self {
        x: Grad::ZERO,
        y: Grad::ZERO,
        z: Grad::ZERO,
    };

    #[inline]
    pub(crate) const fn new(x: Grad, y: Grad, z: Grad) -> Self {
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
    pub(crate) fn from_scaled(a: Grad, u: Self) -> Self {
        Self {
            x: a.mul(u.x),
            y: a.mul(u.y),
            z: a.mul(u.z),
        }
    }

    /// `new FieldVector3D<>(a1, u1, a2, u2)`.
    #[inline]
    pub(crate) fn lincomb2(a1: Grad, u1: Self, a2: Grad, u2: Self) -> Self {
        Self {
            x: Grad::linear_combination_2(a1, u1.x, a2, u2.x),
            y: Grad::linear_combination_2(a1, u1.y, a2, u2.y),
            z: Grad::linear_combination_2(a1, u1.z, a2, u2.z),
        }
    }

    /// `new FieldVector3D<>(a1, u1, a2, u2, a3, u3)`.
    #[inline]
    pub(crate) fn lincomb3(a1: Grad, u1: Self, a2: Grad, u2: Self, a3: Grad, u3: Self) -> Self {
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
    pub(crate) fn scalar_mul(self, a: Grad) -> Self {
        Self {
            x: self.x.mul(a),
            y: self.y.mul(a),
            z: self.z.mul(a),
        }
    }

    #[inline]
    pub(crate) fn norm_sq(self) -> Grad {
        self.x.square().add(self.y.square()).add(self.z.square())
    }

    #[inline]
    pub(crate) fn norm(self) -> Grad {
        self.norm_sq().sqrt()
    }

    #[inline]
    pub(crate) fn normalize(self) -> Option<Self> {
        let s = self.norm();
        if s.value == 0.0 {
            None
        } else {
            Some(self.scalar_mul(s.reciprocal()))
        }
    }

    #[inline]
    pub(crate) fn dot(self, v: Self) -> Grad {
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
    pub(crate) fn distance(self, v: Self) -> Grad {
        let dx = v.x.sub(self.x);
        let dy = v.y.sub(self.y);
        let dz = v.z.sub(self.z);
        dx.square().add(dy.square()).add(dz.square()).sqrt()
    }

    /// `FieldVector3D.orthogonal()`.
    pub(crate) fn orthogonal(self) -> Option<Self> {
        let threshold = 0.6 * self.norm().value;
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
    pub(crate) fn angle(v1: Self, v2: Self) -> Option<Grad> {
        let norm_product = v1.norm().mul(v2.norm());
        if norm_product.value == 0.0 {
            return None;
        }
        let dot = v1.dot(v2);
        let threshold = norm_product.value * 0.9999;
        if dot.value < -threshold || dot.value > threshold {
            let v3 = v1.cross(v2);
            if dot.value >= 0.0 {
                Some(v3.norm().div(norm_product).asin())
            } else {
                Some(v3.norm().div(norm_product).asin().sub(Grad::PI).negate())
            }
        } else {
            Some(dot.div(norm_product).acos())
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
