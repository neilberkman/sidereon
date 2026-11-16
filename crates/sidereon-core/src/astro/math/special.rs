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

    fn div(self, denominator: Self) -> Self {
        let mut quotient = Self::from(0.0);
        let mut remainder = self;
        for _ in 0..3 {
            let digit = remainder.to_f64() / denominator.to_f64();
            quotient = quotient.add(Self::from(digit));
            remainder = remainder.add(denominator.scale(-digit));
        }
        quotient
    }

    fn logarithm(self) -> Self {
        let leading = self.to_f64();
        let remainder = self.add(Self::from(-leading)).scale(1.0 / leading);
        quad_logarithm(leading).add(remainder)
    }

    // A two-limb product suffices for the bounded sine polynomial: its
    // relative error is O(u²), even after the radial kernel's conditioning.
    fn pair_product(self, other: Self) -> Self {
        let (leading, error) = two_prod(self.values[0], other.values[0]);
        let (leading, error) = two_sum(
            leading,
            error + self.values[0] * other.values[1] + self.values[1] * other.values[0],
        );
        Self {
            values: [leading, error, 0.0, 0.0],
        }
    }

    fn pair_divide_scalar(self, divisor: f64) -> Self {
        let leading = self.values[0] / divisor;
        let (product, error) = two_prod(leading, divisor);
        let remainder = ((self.values[0] - product) - error) + self.values[1];
        Self::from(leading).pair_sum(Self::from(remainder / divisor))
    }

    fn pair_sum(self, other: Self) -> Self {
        let (leading, error) = two_sum(self.values[0], other.values[0]);
        let (leading, error) = two_sum(leading, error + self.values[1] + other.values[1]);
        Self {
            values: [leading, error, 0.0, 0.0],
        }
    }

    fn pair_divide(self, divisor: Self) -> Self {
        let leading = self.values[0] / divisor.values[0];
        let (product, error) = two_prod(leading, divisor.values[0]);
        let remainder =
            (((self.values[0] - product) - error) + self.values[1]) - leading * divisor.values[1];
        Self::from(leading).pair_sum(Self::from(remainder / divisor.values[0]))
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

// Gamma scaling needs more than a rounded logarithm: a*log(x) and log Gamma(a)
// can each be hundreds of thousands while their difference is near unity.
fn quad_logarithm(x: f64) -> Quad {
    let (fraction, power) = libm::frexp(x);
    let m = fraction * 2.0;
    let (denominator, denominator_error) = two_sum(m, 1.0);
    let numerator = m - 1.0;
    let z = numerator / denominator;
    let (product, product_error) = two_prod(z, denominator);
    let correction = ((numerator - product) - product_error - z * denominator_error) / denominator;
    let z = Quad::from(z).add(Quad::from(correction));
    let square = z.pair_product(z);
    let mut term = z;
    let mut sum = z;
    for odd in (3..=201).step_by(2) {
        term = term.pair_product(square);
        let reciprocal = 1.0 / odd as f64;
        let (product, error) = two_prod(reciprocal, odd as f64);
        let remainder = ((1.0 - product) - error) / odd as f64;
        sum = sum
            .pair_sum(term.pair_product(Quad::from(reciprocal)))
            .pair_sum(term.pair_product(Quad::from(remainder)));
        if term.values[0].abs() < 1.0e-36 {
            break;
        }
    }
    sum.scale(2.0).add(
        Quad {
            values: [
                f64::from_bits(0x3fe6_2e42_fee0_0000),
                f64::from_bits(0x3dea_39ef_3579_3c76),
                f64::from_bits(0x3a8c_c01f_97b5_7a08),
                f64::from_bits(0xb729_79b3_1ace_93a5),
            ],
        }
        .scale((power - 1) as f64),
    )
}

fn gamma_exponent(a: f64, x: f64) -> Quad {
    if a < 32.0 {
        if 2.0 * a == libm::floor(2.0 * a) {
            // Integer and half-integer shapes have an exact finite gamma
            // product. Retaining that product avoids repeated logarithms and
            // the cancellation of a shifted Stirling expansion.
            let half_integer = a != libm::floor(a);
            let mut product = Quad::from(1.0);
            let mut factor = if half_integer { 0.5 } else { 1.0 };
            while factor < a {
                product = product.scale(factor);
                factor += 1.0;
            }
            let mut log_gamma = product.logarithm();
            if half_integer {
                let pi =
                    Quad::from(core::f64::consts::PI).add(Quad::from(1.224_646_799_147_353_2e-16));
                log_gamma = log_gamma.add(pi.logarithm().scale(0.5));
            }
            return quad_logarithm(x)
                .scale(a)
                .add(Quad::from(-x))
                .add(log_gamma.scale(-1.0));
        }
        let shift = libm::ceil(32.0 - a) as usize;
        let mut result =
            gamma_exponent(a + shift as f64, x).add(quad_logarithm(x).scale(-(shift as f64)));
        for index in 0..shift {
            result = result.add(quad_logarithm(a + index as f64));
        }
        return result;
    }
    let r = 1.0 / a;
    let r2 = r * r;
    let stirling = r
        * (1.0 / 12.0
            + r2 * (-1.0 / 360.0
                + r2 * (1.0 / 1260.0
                    + r2 * (-1.0 / 1680.0 + r2 * (1.0 / 1188.0 - r2 * 691.0 / 360360.0)))));
    quad_logarithm(x)
        .scale(a)
        .add(quad_logarithm(a).scale(0.5 - a))
        .add(Quad::from(-x))
        .add(Quad::from(a))
        .add(Quad::from(-0.918_938_533_204_672_8))
        .add(Quad::from(-stirling))
}

fn gamma_normalization_uncached(a: f64) -> Quad {
    if a < 32.0 {
        return gamma_exponent(a, 1.0).add(Quad::from(1.0));
    }
    // The lower and upper tails must share an absolute normalization, not
    // merely the same rounded scale: a scale error e moves their inverse
    // roots in opposite directions when one tail is complemented. Retain the
    // low limb of log(sqrt(2*pi)) and the rational Stirling coefficients.
    // For a>=32 the first omitted term is <4e-29 (the alternating-remainder
    // bound), below the two-limb tail recurrence error near the median.
    let r = Quad::from(1.0).pair_divide_scalar(a);
    let r2 = r.pair_product(r);
    let mut stirling = Quad::from(43_867.0).pair_divide_scalar(244_188.0);
    for (numerator, denominator) in [
        (-3_617.0, 122_400.0),
        (1.0, 156.0),
        (-691.0, 360_360.0),
        (1.0, 1_188.0),
        (-1.0, 1_680.0),
        (1.0, 1_260.0),
        (-1.0, 360.0),
        (1.0, 12.0),
    ] {
        stirling = stirling
            .pair_product(r2)
            .pair_sum(Quad::from(numerator).pair_divide_scalar(denominator));
    }
    let stirling = stirling.pair_product(r);
    quad_logarithm(a)
        .scale(0.5 - a)
        .add(Quad::from(a))
        .add(Quad::from(-0.918_938_533_204_672_8))
        .add(Quad::from(3.878_294_158_067_241_4e-17))
        .add(stirling.scale(-1.0))
}

/// Compensated gamma-tail comparison for inverse distributions. The same
/// positive series or continued fraction defines every probability regime;
/// subtracting the target before rounding preserves subnormal inputs.
pub(crate) struct GammaTailComparison {
    a: f64,
    constant: Quad,
}

fn gamma_normalization_constant(a: f64) -> Quad {
    if (0.5..=32.0).contains(&a) && 2.0 * a == libm::floor(2.0 * a) {
        static CONSTANTS: std::sync::OnceLock<[Quad; 64]> = std::sync::OnceLock::new();
        let values = CONSTANTS.get_or_init(|| {
            std::array::from_fn(|index| gamma_normalization_uncached(0.5 * (index + 1) as f64))
        });
        values[(2.0 * a - 1.0) as usize]
    } else {
        gamma_normalization_uncached(a)
    }
}

impl GammaTailComparison {
    pub(crate) fn new(a: f64, probability: f64, complement: bool) -> Self {
        let target = if complement {
            Quad::from(1.0).add(Quad::from(-probability)).logarithm()
        } else {
            quad_logarithm(probability)
        };
        Self {
            a,
            constant: gamma_normalization_constant(a).add(target.scale(-1.0)),
        }
    }

    pub(crate) fn evaluate(&self, x: f64, upper: bool) -> (f64, f64) {
        let a = self.a;
        let sum = if upper {
            let mut b = Quad::from(x).add(Quad::from(1.0 - a));
            let mut c = Quad::from(1.0e300);
            let mut d = Quad::from(1.0).pair_divide(b);
            let mut h = d;
            for index in 1..100_000 {
                let n = index as f64;
                let an = -n * (n - a);
                b = b.pair_sum(Quad::from(2.0));
                d = Quad::from(1.0).pair_divide(d.scale(an).pair_sum(b));
                c = b.pair_sum(Quad::from(an).pair_divide(c));
                let delta = d.pair_product(c);
                h = h.pair_product(delta);
                if delta.add(Quad::from(-1.0)).to_f64().abs() < 1.0e-30 {
                    break;
                }
            }
            h
        } else {
            let mut term = Quad::from(1.0).pair_divide_scalar(a);
            let mut sum = term;
            for index in 1..100_000 {
                term = term
                    .pair_product(Quad::from(x))
                    .pair_divide_scalar(a + index as f64);
                sum = sum.pair_sum(term);
                if term.to_f64() < sum.to_f64() * 1.0e-32 {
                    break;
                }
            }
            sum
        };
        let ratio = self
            .constant
            .add(quad_logarithm(x).scale(a))
            .add(Quad::from(-x))
            .add(sum.logarithm())
            .to_f64();
        (ratio, sum.to_f64())
    }
}

pub(crate) fn gamma_scale(a: f64, x: f64) -> f64 {
    let exponent = gamma_exponent(a, x);
    gamma_exponential(exponent)
}

fn quad_sqrt(value: Quad) -> Quad {
    let root = libm::sqrt(value.to_f64());
    Quad::from(root).add(
        value
            .add(Quad::from(root).mul(Quad::from(-root)))
            .scale(0.5 / root),
    )
}

/// Complementary folded-normal probability on the circle rho²+s²=x.
/// Product residuals and square-root corrections keep the normal argument
/// accurate before applying its sharply conditioned complementary error.
pub(crate) struct RadialNormalKernel {
    x: f64,
    a: f64,
    root_x: Quad,
    displacement: Quad,
    constant: Quad,
}

impl RadialNormalKernel {
    pub(crate) fn new(x: f64, a: f64, lambda: f64) -> Self {
        Self {
            x,
            a,
            root_x: quad_sqrt(Quad::from(x)),
            displacement: quad_sqrt(Quad::from(lambda)),
            constant: gamma_normalization_constant(a).add(quad_logarithm(2.0).scale(1.0 - a)),
        }
    }

    pub(crate) fn evaluate(&self, panel: usize, width: f64, node: f64) -> f64 {
        let angle = Quad::from(panel as f64 + 0.5)
            .add(Quad::from(0.5 * node))
            .scale(width);
        let square = angle.pair_product(angle).scale(-1.0);
        let mut term = angle;
        let mut sine = angle;
        // On [0,pi/2], the omitted degree-43 term is below 5e-45.
        for index in 1..=20 {
            term = term
                .pair_product(square)
                .pair_divide_scalar((2 * index * (2 * index + 1)) as f64);
            sine = sine.add(term);
        }
        let radius = self.root_x.mul(sine);
        let radius_square = radius.mul(radius);
        let density = gamma_exponential(
            self.constant
                .add(radius.logarithm().scale(2.0 * self.a - 1.0))
                .add(radius_square.scale(-0.5)),
        );
        let remaining = quad_sqrt(Quad::from(self.x).add(radius_square.scale(-1.0)));
        let sqrt_two =
            Quad::from(core::f64::consts::SQRT_2).add(Quad::from(-9.667_293_313_452_913e-17));
        let tail = |argument: Quad| {
            let z = argument.div(sqrt_two);
            let leading = z.to_f64();
            let residual = z.add(Quad::from(-leading)).to_f64();
            0.5 * libm::erfc(leading)
                - residual * libm::exp(-leading * leading) / libm::sqrt(core::f64::consts::PI)
        };
        density
            * remaining.to_f64()
            * (tail(remaining.add(self.displacement.scale(-1.0)))
                + tail(remaining.add(self.displacement)))
    }
}

fn gamma_exponential(exponent: Quad) -> f64 {
    if exponent.values[0] < libm::log(f64::MIN_POSITIVE) {
        // The low limbs are not individually bounded by the exponential's
        // range. Condense before exponentiation when the leading exponential
        // would underflow, avoiding an otherwise possible 0 * infinity.
        libm::exp(exponent.values.iter().sum())
    } else {
        libm::exp(exponent.values[0]) * libm::exp(exponent.values[1..].iter().sum())
    }
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
    if p > 0.5 {
        return normal_q_inv(1.0 - p).map(|value| -value);
    }
    let mut value = core::f64::consts::SQRT_2 * erfc_inv(2.0 * p)?;
    // The final multiplication by sqrt(2) rounds independently of the
    // inverse-erfc solve.  One Newton correction in the requested normal-tail
    // variable restores the correctly rounded root at ordinary probabilities.
    let density = normal_pdf(value);
    if p >= 1.0e-4 && density > 0.0 {
        value += (normal_q(value) - p) / density;
    }
    Some(if value == 0.0 { 0.0 } else { value })
}

fn erfc_inv_raw(y: f64) -> Option<f64> {
    let central = 1.0 - y;
    if central.abs() <= 0.01 {
        // Sterbenz makes 1-y exact here. Invert erf directly around zero;
        // The first omitted term is (4369/22680)*z^9, less than
        // 7e-20 on this interval (less than 0.04 ulp at its endpoint).
        let z = 0.886_226_925_452_758 * central;
        let z2 = z * z;
        return Some(z * (1.0 + z2 * (1.0 / 3.0 + z2 * (7.0 / 30.0 + z2 * 127.0 / 630.0))));
    }
    if central.abs() <= 0.5 {
        // Sterbenz-exact target and a direct erf residual avoid the absolute
        // error incurred by subtracting Q(z) from a probability near 0.5.
        let mut z = 0.886_226_925_452_758 * central;
        for _ in 0..8 {
            let next = z
                - (libm::erf(z) - central)
                    / (core::f64::consts::FRAC_2_SQRT_PI * libm::exp(-z * z));
            if next == z {
                break;
            }
            z = next;
        }
        return Some(z);
    }
    if y > 1.0 {
        return erfc_inv_raw(2.0 - y).map(|value| -value);
    }
    if y < 0.1 {
        return erfc_inv_from_log(portable_log(y));
    }
    let p_tail = 0.5 * y;
    // Work in the supplied lower tail directly. Forming `1 - p_tail` loses
    // all useful bits for integrity probabilities below one ulp at unity.
    let mut x = -inverse_normal_cdf_approx(p_tail)?;
    for _ in 0..4 {
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

fn erfc_inv_from_log(log_y: f64) -> Option<f64> {
    // erfc(x) ~ exp(-x^2)/(sqrt(pi) x).  Starting from that logarithmic
    // asymptote avoids halving a subnormal input, and Newton corrections use
    // the same expansion without ever materializing the underflowed tail.
    let target = -log_y - 0.5 * portable_log(core::f64::consts::PI);
    let mut x = libm::sqrt(target - 0.5 * portable_log(target));
    for _ in 0..8 {
        let log_erfc = log_erfc_continued_fraction(x)?;
        let log_derivative_magnitude =
            portable_log(2.0) - 0.5 * portable_log(core::f64::consts::PI) - x * x - log_erfc;
        let derivative = -libm::exp(log_derivative_magnitude);
        x -= (log_erfc - log_y) / derivative;
    }
    x.is_finite().then_some(x)
}

fn log_erfc_continued_fraction(x: f64) -> Option<f64> {
    let argument = x * x;
    let mut b = argument + 0.5;
    let mut c = 1.0e300;
    let mut d = 1.0 / b;
    let mut h = d;
    for index in 1..=256 {
        let index = index as f64;
        let coefficient = -index * (index - 0.5);
        b += 2.0;
        d = coefficient * d + b;
        if d.abs() < f64::MIN_POSITIVE {
            d = f64::MIN_POSITIVE;
        }
        c = b + coefficient / c;
        if c.abs() < f64::MIN_POSITIVE {
            c = f64::MIN_POSITIVE;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= 0.5 * f64::EPSILON {
            break;
        }
    }
    (h.is_finite() && h > 0.0).then(|| {
        -argument + 0.5 * portable_log(argument) - 0.5 * portable_log(core::f64::consts::PI)
            + portable_log(h)
    })
}

fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * libm::exp(-0.5 * x * x)
}

fn inverse_normal_cdf_approx(p: f64) -> Option<f64> {
    inverse_normal_with_log(p, portable_log)
}

/// Approximation only: final gamma inversion does not depend on its rounding.
pub(crate) fn gamma_normal_seed(p: f64) -> Option<f64> {
    inverse_normal_with_log(p, libm::log)
}

fn inverse_normal_with_log(p: f64, logarithm: fn(f64) -> f64) -> Option<f64> {
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
        let q = (-2.0 * logarithm(p)).sqrt();
        Some(poly6(&C, q) / poly4p1(&D, q))
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        Some(q * poly6(&A, r) / poly5p1(&B, r))
    } else {
        let q = (-2.0 * logarithm(1.0 - p)).sqrt();
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
    fn gamma_scale_large_arguments_match_mpmath() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/chi2_extended_mpmath.json"
        ))
        .unwrap();
        for case in fixture["gamma_scale_large_argument"].as_array().unwrap() {
            let a = case["a"].as_f64().unwrap();
            let x: f64 = case["x"].as_str().unwrap().parse().unwrap();
            let expected: f64 = case["value"].as_str().unwrap().parse().unwrap();
            assert_eq!(gamma_scale(a, x), expected, "a={a}, x={x}");
        }
    }

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
