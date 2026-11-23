//! First-order forward-mode derivatives, `value + derivative * dt`, used to
//! take the time derivative of the Eckstein-Hechler position.

use core::ops::{Add, Div, Mul, Neg, Sub};

/// A value with its first derivative with respect to time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Dual {
    pub v: f64,
    pub d: f64,
}

impl Dual {
    pub const fn new(v: f64, d: f64) -> Self {
        Self { v, d }
    }

    pub const fn constant(v: f64) -> Self {
        Self { v, d: 0.0 }
    }

    pub fn sin(self) -> Self {
        Self::new(libm::sin(self.v), libm::cos(self.v) * self.d)
    }

    pub fn cos(self) -> Self {
        Self::new(libm::cos(self.v), -libm::sin(self.v) * self.d)
    }

    pub fn sqrt(self) -> Self {
        let s = libm::sqrt(self.v);
        Self::new(s, self.d / (2.0 * s))
    }

    pub fn recip(self) -> Self {
        let r = 1.0 / self.v;
        Self::new(r, -self.d * r * r)
    }

    pub fn scale(self, k: f64) -> Self {
        Self::new(self.v * k, self.d * k)
    }

    pub fn offset(self, k: f64) -> Self {
        Self::new(self.v + k, self.d)
    }
}

impl Add for Dual {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.v + o.v, self.d + o.d)
    }
}

impl Sub for Dual {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.v - o.v, self.d - o.d)
    }
}

impl Mul for Dual {
    type Output = Self;
    fn mul(self, o: Self) -> Self {
        Self::new(self.v * o.v, self.d * o.v + self.v * o.d)
    }
}

impl Div for Dual {
    type Output = Self;
    fn div(self, o: Self) -> Self {
        let q = self.v / o.v;
        Self::new(q, (self.d - q * o.d) / o.v)
    }
}

impl Mul<f64> for Dual {
    type Output = Self;
    fn mul(self, k: f64) -> Self {
        self.scale(k)
    }
}

impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self::new(-self.v, -self.d)
    }
}

#[cfg(test)]
mod tests {
    use super::Dual;

    #[test]
    fn derivatives_match_central_differences() {
        // f(t) = sqrt(1 + t^2) * sin(3 t) / cos(t) + 1 / (2 + t)
        let f = |t: Dual| {
            let one = Dual::constant(1.0);
            (one + t * t).sqrt() * t.scale(3.0).sin() / t.cos() + t.offset(2.0).recip()
        };
        for t in [-0.7, 0.0, 0.3, 1.1] {
            let x = f(Dual::new(t, 1.0));
            let h = 1.0e-6;
            let fd = (f(Dual::constant(t + h)).v - f(Dual::constant(t - h)).v) / (2.0 * h);
            // Central differences with h = 1e-6 carry an O(h^2) truncation
            // error and an O(eps / h) rounding error, both below 1e-8 here.
            assert!((x.d - fd).abs() < 1.0e-8, "t = {t}: {} vs {fd}", x.d);
        }
    }
}
