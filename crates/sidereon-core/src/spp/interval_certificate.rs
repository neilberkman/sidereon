//! Finite outward-rounded intervals for independently certifying SPP inputs.
//!
//! Arithmetic requires IEEE 754 binary64 round-to-nearest, ties-to-even,
//! correctly rounded basic operations and square root, no reassociation,
//! contraction, or fast-math transformations, and gradual underflow. Every
//! intermediate must remain finite; unsupported execution modes and values
//! outside the transcendental domains are refused. Transcendentals here use
//! interval Taylor recurrences with explicit geometric tail bounds, not
//! platform library functions.
//!
//! Sine and cosine use 32 terms on `[-8, 8]`; each remainder is bounded by
//! the first omitted term divided by `1 - q`, with subsequent absolute-term
//! ratios bounded by `q = 64/(66*67)` and `q = 64/(65*66)`, respectively.
//! Exponential uses 32 terms on `[-8, 8]` with `q = 8/33`. Natural logarithm
//! uses 32 atanh-series terms on `[0.5, 2]`, where `|z| <= 1/3`, and bounds
//! its tail by `2|z|^65/(65(1-z²))`.

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct Interval {
    lower: f64,
    upper: f64,
}

impl Interval {
    pub(super) fn new(lower: f64, upper: f64) -> Self {
        assert!(lower.is_finite() && upper.is_finite() && lower <= upper);
        Self { lower, upper }
    }

    pub(super) fn point(value: f64) -> Self {
        assert!(value.is_finite());
        Self::new(value, value)
    }

    pub(super) const fn lower(self) -> f64 {
        self.lower
    }

    pub(super) const fn upper(self) -> f64 {
        self.upper
    }

    pub(super) fn width(self) -> f64 {
        next_up(self.upper - self.lower)
    }

    pub(super) fn contains(self, value: f64) -> bool {
        value.is_finite() && self.lower <= value && value <= self.upper
    }

    pub(super) fn contains_interval(self, other: Self) -> bool {
        self.lower <= other.lower && other.upper <= self.upper
    }

    pub(super) fn hull(self, other: Self) -> Self {
        Self::new(self.lower.min(other.lower), self.upper.max(other.upper))
    }

    pub(super) fn add(self, other: Self) -> Self {
        Self::new(
            next_down(self.lower + other.lower),
            next_up(self.upper + other.upper),
        )
    }

    pub(super) fn sub(self, other: Self) -> Self {
        Self::new(
            next_down(self.lower - other.upper),
            next_up(self.upper - other.lower),
        )
    }

    pub(super) fn mul(self, other: Self) -> Self {
        let products = [
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper,
        ];
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for product in products {
            lower = lower.min(next_down(product));
            upper = upper.max(next_up(product));
        }
        Self::new(lower, upper)
    }

    pub(super) fn div(self, other: Self) -> Self {
        assert!(!other.contains(0.0));
        let quotients = [
            self.lower / other.lower,
            self.lower / other.upper,
            self.upper / other.lower,
            self.upper / other.upper,
        ];
        let mut lower = f64::INFINITY;
        let mut upper = f64::NEG_INFINITY;
        for quotient in quotients {
            lower = lower.min(next_down(quotient));
            upper = upper.max(next_up(quotient));
        }
        Self::new(lower, upper)
    }

    pub(super) fn square(self) -> Self {
        if self.contains(0.0) {
            let maximum = self.lower.abs().max(self.upper.abs());
            Self::new(0.0, next_up(maximum * maximum))
        } else if self.upper < 0.0 {
            Self::new(
                next_down(self.upper * self.upper).max(0.0),
                next_up(self.lower * self.lower),
            )
        } else {
            Self::new(
                next_down(self.lower * self.lower).max(0.0),
                next_up(self.upper * self.upper),
            )
        }
    }

    pub(super) fn sqrt(self) -> Self {
        assert!(self.lower >= 0.0);
        Self::new(
            next_down(self.lower.sqrt()).max(0.0),
            next_up(self.upper.sqrt()),
        )
    }

    /// Enclose sine by 32 Taylor terms and a geometric remainder for `[-8, 8]`.
    pub(super) fn sin(self) -> Self {
        assert!(self.lower >= -8.0 && self.upper <= 8.0);
        let square = self.square();
        let mut term = self;
        let mut sum = Self::point(0.0);
        for index in 0..32 {
            sum = sum.add(term);
            let first_factor = (2 * index + 2) as f64;
            let second_factor = (2 * index + 3) as f64;
            term = term
                .mul(square)
                .div(Self::point(first_factor * second_factor))
                .mul(Self::point(-1.0));
        }
        let omitted_term = term.absolute_upper();
        let ratio = Self::point(64.0).div(Self::point(66.0 * 67.0)).upper();
        sum.add(symmetric_error(geometric_tail(omitted_term, ratio)))
    }

    /// Enclose cosine by 32 Taylor terms and a geometric remainder for `[-8, 8]`.
    pub(super) fn cos(self) -> Self {
        assert!(self.lower >= -8.0 && self.upper <= 8.0);
        let square = self.square();
        let mut term = Self::point(1.0);
        let mut sum = Self::point(0.0);
        for index in 0..32 {
            sum = sum.add(term);
            let first_factor = (2 * index + 1) as f64;
            let second_factor = (2 * index + 2) as f64;
            term = term
                .mul(square)
                .div(Self::point(first_factor * second_factor))
                .mul(Self::point(-1.0));
        }
        let omitted_term = term.absolute_upper();
        let ratio = Self::point(64.0).div(Self::point(65.0 * 66.0)).upper();
        sum.add(symmetric_error(geometric_tail(omitted_term, ratio)))
    }

    /// Enclose exponential by 32 Taylor terms and a geometric remainder for `[-8, 8]`.
    pub(super) fn exp(self) -> Self {
        assert!(self.lower >= -8.0 && self.upper <= 8.0);
        let mut term = Self::point(1.0);
        let mut sum = Self::point(0.0);
        for index in 0..32 {
            sum = sum.add(term);
            term = term.mul(self).div(Self::point((index + 1) as f64));
        }
        let ratio = Self::point(8.0).div(Self::point(33.0)).upper();
        sum.add(symmetric_error(geometric_tail(
            term.absolute_upper(),
            ratio,
        )))
    }

    /// Enclose natural logarithm by 32 atanh-series terms for `[0.5, 2]`.
    pub(super) fn ln(self) -> Self {
        assert!(self.lower >= 0.5 && self.upper <= 2.0);
        let one = Self::point(1.0);
        let transformed = self.sub(one).div(self.add(one));
        let transformed_squared = transformed.square();
        let mut power = transformed;
        let mut sum = Self::point(0.0);
        for index in 0..32 {
            sum = sum.add(power.div(Self::point((2 * index + 1) as f64)));
            power = power.mul(transformed_squared);
        }
        let maximum_transformed = transformed.absolute_upper();
        assert!(maximum_transformed < 1.0);
        let power_bound = positive_power_upper(maximum_transformed, 65);
        let transformed_bound = Self::point(maximum_transformed);
        let denominator = Self::point(65.0).mul(one.sub(transformed_bound.square()));
        let tail = Self::point(2.0)
            .mul(Self::point(power_bound))
            .div(denominator)
            .upper();
        sum.mul(Self::point(2.0)).add(symmetric_error(tail))
    }

    fn absolute_upper(self) -> f64 {
        self.lower.abs().max(self.upper.abs())
    }
}

fn symmetric_error(radius: f64) -> Interval {
    assert!(radius.is_finite() && radius >= 0.0);
    Interval::new(-radius, radius)
}

fn geometric_tail(first_term_upper: f64, ratio_upper: f64) -> f64 {
    assert!(first_term_upper.is_finite() && first_term_upper >= 0.0);
    assert!(ratio_upper.is_finite() && (0.0..1.0).contains(&ratio_upper));
    Interval::point(first_term_upper)
        .div(Interval::point(1.0).sub(Interval::point(ratio_upper)))
        .upper()
}

fn positive_power_upper(base_upper: f64, exponent: usize) -> f64 {
    assert!(base_upper.is_finite() && base_upper >= 0.0);
    let base = Interval::point(base_upper);
    let mut result = Interval::point(1.0);
    for _ in 0..exponent {
        result = result.mul(base);
    }
    result.upper()
}

fn next_up(value: f64) -> f64 {
    assert!(value.is_finite());
    if value == 0.0 {
        return f64::from_bits(1);
    }
    let bits = value.to_bits();
    let next = if value > 0.0 { bits + 1 } else { bits - 1 };
    let result = f64::from_bits(next);
    assert!(result.is_finite());
    result
}

fn next_down(value: f64) -> f64 {
    assert!(value.is_finite());
    if value == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = value.to_bits();
    let next = if value > 0.0 { bits - 1 } else { bits + 1 };
    let result = f64::from_bits(next);
    assert!(result.is_finite());
    result
}

#[cfg(test)]
mod tests {
    use super::Interval;

    #[test]
    fn elementary_interval_operations_enclose_exact_values() {
        let sum = Interval::point(3.0).add(Interval::point(4.0));
        let difference = Interval::point(7.0).sub(Interval::point(4.0));
        let product = Interval::point(-3.0).mul(Interval::point(4.0));
        let quotient = Interval::point(6.0).div(Interval::point(3.0));
        let square = Interval::new(-3.0, 2.0).square();
        let root = Interval::point(4.0).sqrt();

        assert!(sum.contains(7.0));
        assert!(difference.contains(3.0));
        assert!(product.contains(-12.0));
        assert!(quotient.contains(2.0));
        assert!(square.contains(0.0) && square.contains(9.0));
        assert!(root.contains(2.0));
    }

    #[test]
    fn hull_membership_and_width_cover_the_union() {
        let first = Interval::new(-2.0, 1.0);
        let second = Interval::new(0.5, 4.0);
        let hull = first.hull(second);

        assert_eq!(hull.lower(), -2.0);
        assert_eq!(hull.upper(), 4.0);
        assert!(hull.contains_interval(first));
        assert!(hull.contains_interval(second));
        assert!(hull.contains(3.0));
        assert!(hull.width() >= 6.0);
    }

    #[test]
    fn transcendental_enclosures_contain_exact_identities() {
        let zero = Interval::point(0.0);
        let one = Interval::point(1.0);
        let angle = Interval::new(-1.0, 1.0);
        let sine = angle.sin();
        let cosine = angle.cos();
        let identity = sine.square().add(cosine.square());

        assert!(zero.sin().contains(0.0));
        assert!(zero.cos().contains(1.0));
        assert!(zero.exp().contains(1.0));
        assert!(one.ln().contains(0.0));
        assert!(Interval::point(0.5).exp().ln().contains(0.5));
        assert!(identity.contains(1.0));
    }

    #[test]
    fn transcendental_domain_boundaries_enclose_exact_identities() {
        let sine_negative_boundary = Interval::point(-8.0).sin();
        let sine_positive_boundary = Interval::point(8.0).sin();
        let cosine_negative_boundary = Interval::point(-8.0).cos();
        let cosine_positive_boundary = Interval::point(8.0).cos();
        let exponential_negative_boundary = Interval::point(-8.0).exp();
        let exponential_positive_boundary = Interval::point(8.0).exp();
        let logarithm_lower_boundary = Interval::point(0.5).ln();
        let logarithm_upper_boundary = Interval::point(2.0).ln();

        assert!(sine_negative_boundary
            .add(sine_positive_boundary)
            .contains(0.0));
        assert!(cosine_negative_boundary.contains_interval(cosine_positive_boundary));
        assert!(cosine_positive_boundary.contains_interval(cosine_negative_boundary));
        assert!(exponential_negative_boundary
            .mul(exponential_positive_boundary)
            .contains(1.0));
        assert!(logarithm_lower_boundary
            .add(logarithm_upper_boundary)
            .contains(0.0));
    }

    #[test]
    fn transcendental_domains_refuse_adjacent_outside_values() {
        let below_negative_eight = f64::from_bits((-8.0f64).to_bits() + 1);
        let above_eight = f64::from_bits(8.0f64.to_bits() + 1);
        let below_half = f64::from_bits(0.5f64.to_bits() - 1);
        let above_two = f64::from_bits(2.0f64.to_bits() + 1);

        assert!(std::panic::catch_unwind(|| Interval::point(below_negative_eight).sin()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(above_eight).sin()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(below_negative_eight).cos()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(above_eight).cos()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(below_negative_eight).exp()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(above_eight).exp()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(below_half).ln()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(above_two).ln()).is_err());
    }

    #[test]
    fn decimal100_reference_fixture_covers_certified_domains() {
        use sha2::{Digest, Sha256};

        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/spp_interval_elementary.json"
        ))
        .expect("parse independent Decimal100 interval fixture");
        let generator =
            include_bytes!("../../fixtures-generators/generate_spp_interval_elementary.py");
        let generator_digest = format!("{:x}", Sha256::digest(generator));
        let cases = fixture["cases"]
            .as_array()
            .expect("fixture cases are an array");
        let expected_arguments = [
            (
                "sin",
                &[-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0][..],
            ),
            (
                "cos",
                &[-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0][..],
            ),
            (
                "exp",
                &[-8.0, -4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0][..],
            ),
            ("ln", &[0.5, 1.0, 1.5, 2.0][..]),
        ];

        assert_eq!(fixture["decimal_precision"].as_u64(), Some(100));
        assert_eq!(fixture["trigonometric_terms"].as_u64(), Some(80));
        assert_eq!(
            fixture["generator_sha256"].as_str(),
            Some(generator_digest.as_str())
        );
        assert_eq!(cases.len(), 37);

        for (function, arguments) in expected_arguments {
            let function_cases: Vec<_> = cases
                .iter()
                .filter(|case| case["function"].as_str() == Some(function))
                .collect();
            assert_eq!(
                function_cases.len(),
                arguments.len(),
                "{function} case count"
            );

            for (case, argument) in function_cases.into_iter().zip(arguments) {
                let argument: f64 = *argument;
                let expected_bits = format!("0x{:016x}", argument.to_bits());
                assert_eq!(case["argument_bits"].as_str(), Some(expected_bits.as_str()));

                let lower_bits = u64::from_str_radix(
                    case["lower_bits"]
                        .as_str()
                        .expect("lower endpoint bits are text")
                        .trim_start_matches("0x"),
                    16,
                )
                .expect("lower endpoint bits are hexadecimal");
                let upper_bits = u64::from_str_radix(
                    case["upper_bits"]
                        .as_str()
                        .expect("upper endpoint bits are text")
                        .trim_start_matches("0x"),
                    16,
                )
                .expect("upper endpoint bits are hexadecimal");
                let reference =
                    Interval::new(f64::from_bits(lower_bits), f64::from_bits(upper_bits));
                let certified = match function {
                    "sin" => Interval::point(argument).sin(),
                    "cos" => Interval::point(argument).cos(),
                    "exp" => Interval::point(argument).exp(),
                    "ln" => Interval::point(argument).ln(),
                    _ => unreachable!("only expected functions are visited"),
                };
                assert!(
                    certified.contains_interval(reference),
                    "{function} reference interval not enclosed at {argument}"
                );
            }
        }
    }

    #[test]
    fn invalid_domains_and_nonfinite_intervals_are_refused() {
        assert!(std::panic::catch_unwind(|| Interval::point(f64::INFINITY)).is_err());
        assert!(std::panic::catch_unwind(|| {
            Interval::point(1.0).div(Interval::new(-1.0, 1.0))
        })
        .is_err());
        assert!(std::panic::catch_unwind(|| Interval::new(-1.0, 1.0).sqrt()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(8.5).sin()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(-9.0).exp()).is_err());
        assert!(std::panic::catch_unwind(|| Interval::point(0.0).ln()).is_err());
    }
}
