//! Portability proof for the standard-library `f64::powi` calls used by the
//! crate. The reference is the exponentiation-by-squaring order used by the
//! portable scalar's `ComplexField::powi` implementation.

/// The literal exponents found by `rg '\\.powi\\(' crates/sidereon-core/src
/// crates/trust-region-least-squares/src`, plus the runtime exponent interval
/// used by the format and force-model paths.
const USED_EXPONENTS: &[i32] = &[
    -52, -35, -28, -26, -14, -3, -2, -1, 0, 2, 3, 4, 5, 6, 7, 8, 24, 29, 53,
];

fn repeated_square(base: f64, exponent: i32) -> f64 {
    let negative = exponent < 0;
    let mut power = exponent.unsigned_abs();
    let mut base = base;
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

fn next_base(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(0xd1342543de82ef95)
        .wrapping_add(0xa4093822299f31d0);
    // Keep all tested powers finite and away from underflow while retaining
    // 52 random mantissa bits and both sides of one.
    let fraction = f64::from_bits(0x3fe0000000000000 | (*state >> 12)) - 0.5;
    if *state & 1 == 0 {
        1.0 + fraction
    } else {
        1.0 - fraction
    }
}

#[test]
fn f64_powi_matches_fixed_repeated_multiplication_order() {
    let mut state = 0x9e3779b97f4a7c15_u64;
    for &exponent in USED_EXPONENTS {
        for _ in 0..10_000 {
            let base = next_base(&mut state);
            assert_eq!(
                base.powi(exponent).to_bits(),
                repeated_square(base, exponent).to_bits(),
                "f64::powi changed its multiplication order for exponent {exponent}"
            );
        }
    }

    // Variable exponent call sites cover this signed interval in production
    // (URA, decimal formatting, harmonic degree, and parsed radix fields).
    for exponent in -64..=64 {
        for _ in 0..10_000 {
            let base = next_base(&mut state);
            assert_eq!(
                base.powi(exponent).to_bits(),
                repeated_square(base, exponent).to_bits(),
                "f64::powi changed its multiplication order for runtime exponent {exponent}"
            );
        }
    }
}
