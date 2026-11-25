use crate::astro::events::eclipse::ad::ScaledF64;

use core::ops::{Add, Div, Mul, Neg, Sub};

/// Binary64 arithmetic with extended exponents for force formulas. The
/// operators retain IEEE signed zeros when the mantissas are zero.
#[derive(Clone, Copy)]
pub(super) struct Wide(ScaledF64);

impl Wide {
    pub(super) fn from_scaled(mantissa: f64, exponent: i32) -> Self {
        Self(ScaledF64::from_scaled(mantissa, exponent))
    }

    pub(super) fn sqrt(self) -> Self {
        Self(self.0.sqrt())
    }

    pub(super) fn powi(self, mut exponent: u32) -> Self {
        let mut result = Self::from(1.0);
        let mut base = self;
        while exponent != 0 {
            if exponent & 1 != 0 {
                result = result * base;
            }
            exponent >>= 1;
            base = base * base;
        }
        result
    }

    pub(super) fn value(self) -> f64 {
        self.0.to_f64()
    }
}

impl From<f64> for Wide {
    fn from(value: f64) -> Self {
        Self(ScaledF64::from_f64(value))
    }
}

impl Add for Wide {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        if self.0.is_zero() && rhs.0.is_zero() {
            return Self::from(self.0.m + rhs.0.m);
        }
        Self(self.0.add(rhs.0))
    }
}

impl Sub for Wide {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self + -rhs
    }
}

impl Mul for Wide {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        if self.0.is_zero() || rhs.0.is_zero() {
            return Self::from(self.0.m * rhs.0.m);
        }
        Self(self.0.mul(rhs.0))
    }
}

impl Div for Wide {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        if self.0.is_zero() {
            return Self::from(self.0.m / rhs.0.m);
        }
        Self(self.0.div(rhs.0))
    }
}

impl Neg for Wide {
    type Output = Self;

    fn neg(self) -> Self {
        Self(self.0.negate())
    }
}

pub(super) fn wide_dot(left: [Wide; 3], right: [Wide; 3]) -> Wide {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

/// The squared norm of a nalgebra column vector starts at positive zero.
pub(super) fn wide_norm_squared(vector: [Wide; 3]) -> Wide {
    Wide::from(0.0) + wide_dot(vector, vector)
}

pub(super) trait ForceScalar:
    Copy
    + From<f64>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn sqrt(self) -> Self;
    fn value(self) -> f64;
    fn is_zero(self) -> bool;
    fn is_negative(self) -> bool;
}

impl ForceScalar for f64 {
    fn value(self) -> f64 {
        self
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn is_zero(self) -> bool {
        self == 0.0
    }

    fn is_negative(self) -> bool {
        self < 0.0
    }
}

impl ForceScalar for Wide {
    fn value(self) -> f64 {
        self.value()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn is_zero(self) -> bool {
        self.0.is_zero()
    }

    fn is_negative(self) -> bool {
        self.0.m < 0.0
    }
}

/// Ordered force sum. Promote only a running sum that exceeds binary64.
pub(super) struct ForceSum<const N: usize> {
    values: [f64; N],
    extended: Option<[Wide; N]>,
}

impl<const N: usize> ForceSum<N> {
    pub(super) fn new() -> Self {
        Self {
            values: [0.0; N],
            extended: None,
        }
    }

    pub(super) fn add(&mut self, component: [f64; N]) {
        if let Some(extended) = &mut self.extended {
            for (sum, value) in extended.iter_mut().zip(component) {
                *sum = *sum + Wide::from(value);
            }
        } else {
            let next = core::array::from_fn(|i| self.values[i] + component[i]);
            if next.iter().all(|value| value.is_finite()) {
                self.values = next;
            } else {
                self.extended = Some(core::array::from_fn(|i| {
                    Wide::from(self.values[i]) + Wide::from(component[i])
                }));
            }
        }
    }

    pub(super) fn values(self) -> [f64; N] {
        self.extended
            .map_or(self.values, |sum| sum.map(Wide::value))
    }
}
