//! Special functions with a deterministic, cross-platform implementation.
//!
//! The Gauss error function `erf` is provided by the pure-Rust `libm` crate
//! rather than the platform C math library. The system `libm` is not
//! bit-identical across platforms, so binding it via `extern "C"` would break
//! cross-platform 0-ULP determinism. `libm::erf` is a deterministic port that
//! produces the same bits on every platform.

/// Binary64 natural logarithm with a portable, correctly-rounded result for
/// the finite positive inputs used by the core math paths.
///
/// `libm::log` is deterministic, but its fdlibm implementation documents an
/// error bound below one ULP rather than correct rounding; in particular it
/// returns the lower neighbor for `log(3.0)`. The small expansion kernel below
/// keeps the logarithm series and the `ln(2)` constant in four error-free
/// limbs, so the final conversion to binary64 selects the nearest result
/// without consulting the host math library. Non-positive and non-finite
/// values retain `libm`'s documented special-value behavior.
#[inline]
pub fn portable_log(x: f64) -> f64 {
    if !x.is_finite() || x <= 0.0 {
        return libm::log(x);
    }
    if x == 1.0 {
        return 0.0;
    }

    let mut scaled = x;
    let mut exponent = ((scaled.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    if exponent == -1023 {
        scaled *= 18_014_398_509_481_984.0; // 2^54
        exponent = ((scaled.to_bits() >> 52) & 0x7ff) as i32 - 1023 - 54;
    }
    let mantissa =
        f64::from_bits((scaled.to_bits() & 0x000f_ffff_ffff_ffff) | 0x3ff0_0000_0000_0000);
    let z = (mantissa - 1.0) / (mantissa + 1.0);
    let z2 = Quad::from(z * z);
    let mut term = Quad::from(z);
    let mut sum = Quad::from(z);
    for odd in (3..=601).step_by(2) {
        term = term.mul(z2);
        let addend = term.scale(1.0 / odd as f64);
        sum = sum.add(addend);
        if addend.values[0].abs() < f64::from_bits(0x38f0_0000_0000_0000) {
            break;
        }
    }

    let ln_mantissa = sum.scale(2.0);
    let ln_two = Quad {
        values: [
            f64::from_bits(0x3fe6_2e42_fee0_0000),
            f64::from_bits(0x3dea_39ef_3579_3c76),
            f64::from_bits(0x3a8c_c01f_97b5_7a08),
            f64::from_bits(0xb729_79b3_1ace_93a5),
        ],
    };
    ln_mantissa.add(ln_two.scale(exponent as f64)).to_f64()
}

#[derive(Clone, Copy)]
struct Quad {
    values: [f64; 4],
}

impl Quad {
    const fn from(value: f64) -> Self {
        Self {
            values: [value, 0.0, 0.0, 0.0],
        }
    }

    fn add(self, other: Self) -> Self {
        let mut result = self;
        for &value in &other.values {
            result.add_scalar(value);
        }
        result
    }

    fn add_scalar(&mut self, value: f64) {
        let mut carry = value;
        for index in 0..self.values.len() {
            let (sum, error) = two_sum(self.values[index], carry);
            self.values[index] = sum;
            carry = error;
        }
    }

    fn scale(self, factor: f64) -> Self {
        let mut result = Self::from(0.0);
        for &value in &self.values {
            let (product, error) = two_prod(value, factor);
            result.add_scalar(product);
            result.add_scalar(error);
        }
        result
    }

    fn mul(self, other: Self) -> Self {
        let mut result = Self::from(0.0);
        for &left in &self.values {
            for &right in &other.values {
                let (product, error) = two_prod(left, right);
                result.add_scalar(product);
                result.add_scalar(error);
            }
        }
        result
    }

    fn to_f64(self) -> f64 {
        self.values.iter().copied().sum()
    }
}

#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    (sum, a_roundoff + b_roundoff)
}

#[inline]
fn two_prod(a: f64, b: f64) -> (f64, f64) {
    const SPLITTER: f64 = 134_217_729.0; // 2^27 + 1
    let product = a * b;
    let a_split = SPLITTER * a;
    let a_high = a_split - (a_split - a);
    let a_low = a - a_high;
    let b_split = SPLITTER * b;
    let b_high = b_split - (b_split - b);
    let b_low = b - b_high;
    let error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low;
    (product, error)
}

/// Gauss error function, deterministic across platforms via the `libm` crate.
#[inline]
pub fn erf(x: f64) -> f64 {
    libm::erf(x)
}

/// Complementary error function, deterministic across platforms via `libm`.
#[inline]
pub fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

/// Upper standard-normal tail probability `Q(x) = P(Z > x)`.
#[inline]
pub fn normal_q(x: f64) -> f64 {
    0.5 * erfc(x * core::f64::consts::FRAC_1_SQRT_2)
}

/// Inverse complementary error function for `y` in `(0, 2)`.
#[inline]
pub fn erfc_inv(y: f64) -> Option<f64> {
    if !(0.0..2.0).contains(&y) || !y.is_finite() {
        return None;
    }
    erfc_inv_raw(y)
}

/// Inverse upper standard-normal tail probability for `p` in `(0, 1)`.
#[inline]
pub fn normal_q_inv(p: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&p) || !p.is_finite() {
        return None;
    }
    Some(core::f64::consts::SQRT_2 * erfc_inv(2.0 * p)?)
}

fn erfc_inv_raw(y: f64) -> Option<f64> {
    let p_tail = 0.5 * y;
    let p_cdf = 1.0 - p_tail;
    let mut x = inverse_normal_cdf_approx(p_cdf)?;
    for _ in 0..2 {
        let f = normal_q(x) - p_tail;
        let phi = normal_pdf(x);
        if phi == 0.0 || !phi.is_finite() {
            break;
        }
        let ratio = f / phi;
        let den = 1.0 - 0.5 * x * ratio;
        if den == 0.0 || !den.is_finite() {
            x += ratio;
        } else {
            x += ratio / den;
        }
    }
    if x.is_finite() {
        Some(x * core::f64::consts::FRAC_1_SQRT_2)
    } else {
        None
    }
}

fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * libm::exp(-0.5 * x * x)
}

fn inverse_normal_cdf_approx(p: f64) -> Option<f64> {
    if !(0.0..1.0).contains(&p) || !p.is_finite() {
        return None;
    }

    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    const P_LOW: f64 = 0.024_25;
    const P_HIGH: f64 = 1.0 - P_LOW;

    if p < P_LOW {
        let q = (-2.0 * portable_log(p)).sqrt();
        Some(poly6(&C, q) / poly4p1(&D, q))
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        Some(q * poly6(&A, r) / poly5p1(&B, r))
    } else {
        let q = (-2.0 * portable_log(1.0 - p)).sqrt();
        Some(-poly6(&C, q) / poly4p1(&D, q))
    }
}

fn poly6(c: &[f64; 6], x: f64) -> f64 {
    (((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x) + c[5]
}

fn poly5p1(c: &[f64; 5], x: f64) -> f64 {
    (((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x + c[4]) * x) + 1.0
}

fn poly4p1(c: &[f64; 4], x: f64) -> f64 {
    ((((c[0] * x + c[1]) * x + c[2]) * x + c[3]) * x) + 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn erf_has_frozen_bits() {
        // Frozen-bits golden of the deterministic `libm::erf`. These bits are
        // identical on every platform; a tolerance check would have hidden the
        // 1-ULP drift the platform C `erf` introduced at erf(1.0).
        assert_eq!(erf(0.0).to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(erf(0.75).to_bits(), 0x3fe6_c1c9_759d_0e60);
        assert_eq!(erf(1.0).to_bits(), 0x3fea_f767_a741_088b);
        assert_eq!(erf(1.5).to_bits(), 0x3fee_ea55_5713_7ae0);
        assert_eq!(erf(2.0).to_bits(), 0x3fef_d9ae_1427_95e3);
        assert_eq!(erf(6.0).to_bits(), 0x3ff0_0000_0000_0000);
        // erf is odd: erf(-x) == -erf(x), exactly.
        assert_eq!(erf(-0.75).to_bits(), (-erf(0.75)).to_bits());
        assert_eq!(erf(-6.0).to_bits(), (-1.0_f64).to_bits());
    }

    #[test]
    fn normal_q_inv_round_trips_tail_probabilities() {
        for p in [1.0e-12, 1.0e-9, 5.0e-8, 1.0e-4, 0.01, 0.5, 0.9] {
            let x = normal_q_inv(p).expect("valid tail probability");
            let got = normal_q(x);
            let tol = (p * 2.0e-12).max(1.0e-15);
            assert!(
                (got - p).abs() <= tol,
                "p={p} x={x} got={got} diff={}",
                (got - p).abs()
            );
        }
    }

    #[test]
    fn normal_q_inv_has_frozen_bits() {
        assert_eq!(normal_q_inv(0.5).unwrap().to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(normal_q_inv(0.01).unwrap().to_bits(), 0x4002_9c5c_4630_ff0f);
        assert_eq!(
            normal_q_inv(5.0e-8).unwrap().to_bits(),
            0x4015_4e90_b4db_5fad
        );
    }

    #[test]
    fn portable_log_resolves_libm_rounding_counterexample() {
        assert_eq!(portable_log(3.0).to_bits(), 0x3ff1_93ea_7aad_030b);
        assert_eq!(portable_log(1.0).to_bits(), 0x0000_0000_0000_0000);
        assert_eq!(portable_log(0.5).to_bits(), 0xbfe6_2e42_fefa_39ef);
        assert_eq!(portable_log(f64::INFINITY), f64::INFINITY);
        assert!(portable_log(-1.0).is_nan());
    }
}
