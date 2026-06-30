//! Scalar linear-interpolation primitives with pinned operation order.
//!
//! GNSS interpolators are bit-exact against deployed references, so the two
//! distinct evaluation orders below are kept as separate named helpers rather
//! than folded into one: callers that precompute a fraction need
//! divide-before-multiply, while clock interpolators pin multiply-before-divide.

/// Linear interpolation by a precomputed fraction: `a + (b - a) * t`.
///
/// `t` is the interpolation parameter (`0.0` returns `a`, `1.0` returns `b`).
/// The caller is responsible for forming `t`; this helper fixes only the
/// `a + (b - a) * t` evaluation order.
#[inline]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Linear interpolation by an explicit ratio, multiplying before dividing:
/// `a + (b - a) * num / den`.
///
/// This is `lerp` with `t = num / den` but with the division applied last, which
/// is a different floating-point rounding than `lerp(a, b, num / den)`. The
/// clock interpolators pin this order for bit-exact parity, so it has its own
/// helper. The caller guarantees `den` is nonzero.
#[inline]
pub fn lerp_ratio(a: f64, b: f64, num: f64, den: f64) -> f64 {
    a + (b - a) * num / den
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lerp_matches_explicit_recipe_bits() {
        let (a, b, t) = (1.25_f64, -3.5_f64, 0.3_f64);
        assert_eq!(lerp(a, b, t).to_bits(), (a + (b - a) * t).to_bits());
    }

    #[test]
    fn lerp_endpoints_are_exact() {
        assert_eq!(lerp(2.0, 5.0, 0.0).to_bits(), 2.0_f64.to_bits());
        assert_eq!(lerp(2.0, 5.0, 1.0).to_bits(), 5.0_f64.to_bits());
    }

    #[test]
    fn lerp_ratio_matches_explicit_recipe_bits() {
        let (a, b, num, den) = (1.0e-6_f64, 1.3e-6_f64, 7.0_f64, 30.0_f64);
        assert_eq!(
            lerp_ratio(a, b, num, den).to_bits(),
            (a + (b - a) * num / den).to_bits()
        );
    }

    #[test]
    fn lerp_ratio_preserves_multiply_before_divide_order() {
        // A case where multiply-before-divide and divide-before-multiply round
        // differently, so the helper is pinned to the former.
        let (a, b, num, den) = (0.0_f64, 1.0_f64, 1.0_f64, 3.0_f64);
        assert_eq!(
            lerp_ratio(a, b, num, den).to_bits(),
            (a + (b - a) * num / den).to_bits()
        );
    }
}
