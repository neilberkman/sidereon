//! Format-faithful numeric formatting helpers.

use libm::{floor, log10, pow};

/// Decimal places carried by the assumed-decimal mantissa.
const ASSUMED_DECIMAL_MANTISSA_DECIMALS: usize = 5;
/// Number of mantissa digits emitted in an assumed-decimal field.
const ASSUMED_DECIMAL_MANTISSA_DIGITS: usize = 5;

/// Fixed-decimal formatting copied from the TLE encoder.
///
/// This is a bit-exact-load-bearing copy for the sans-I/O layer. The math must
/// not change.
pub(crate) fn fixed_decimals(value: f64, decimals: usize) -> String {
    format!("{value:.decimals$}")
}

/// Shortest round-tripping decimal copied from the OMM/CDM encoders.
///
/// This is a bit-exact-load-bearing copy for the sans-I/O layer. The math must
/// not change.
pub(crate) fn fmt_num(value: f64) -> String {
    format!("{value}")
}

/// Format an assumed-decimal field copied from the TLE encoder.
///
/// This is a bit-exact-load-bearing copy for the sans-I/O layer. The math must
/// not change.
pub(crate) fn fmt_assumed_decimal(val: f64) -> String {
    if val == 0.0 {
        return " 00000-0".to_string();
    }
    let sign = if val < 0.0 { '-' } else { ' ' };
    let av = val.abs();
    let raw_exp = floor(log10(av)) as i32;
    let mut exp = raw_exp + 1;
    let mantissa = av / pow(10.0, exp as f64);
    let mut mant_full = fixed_decimals(mantissa, ASSUMED_DECIMAL_MANTISSA_DECIMALS);
    if mant_full.starts_with("1.") {
        exp += 1;
        mant_full = fixed_decimals(mantissa / 10.0, ASSUMED_DECIMAL_MANTISSA_DECIMALS);
    }
    let mant_str: String = mant_full
        .chars()
        .skip(2)
        .take(ASSUMED_DECIMAL_MANTISSA_DIGITS)
        .collect();
    let exp_sign = if exp >= 0 { '+' } else { '-' };
    format!("{sign}{mant_str}{exp_sign}{}", exp.abs())
}

/// Decode an assumed-decimal field copied from the TLE parser.
///
/// This is a bit-exact-load-bearing copy for the sans-I/O layer. The math must
/// not change.
pub(crate) fn decode_assumed_decimal_field(field: &str) -> f64 {
    let sign = if field.starts_with('-') { -1.0 } else { 1.0 };
    let body = &field[1..];
    let mantissa_digits = &body[..ASSUMED_DECIMAL_MANTISSA_DIGITS];
    let exp_field = &body[ASSUMED_DECIMAL_MANTISSA_DIGITS..];
    let exp_field = exp_field.strip_prefix('+').unwrap_or(exp_field);
    let mantissa: f64 = format!("0.{mantissa_digits}").parse().unwrap_or(0.0);
    let exp: i32 = exp_field.parse().unwrap_or(0);
    sign * mantissa * 10.0_f64.powi(exp)
}

/// Quantize through the assumed-decimal grid copied from the TLE bridge.
///
/// This is a bit-exact-load-bearing copy for the sans-I/O layer. The math must
/// not change.
pub(crate) fn assumed_decimal_quantize(value: f64) -> f64 {
    if value == 0.0 {
        return 0.0;
    }
    decode_assumed_decimal_field(&fmt_assumed_decimal(value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_decimals_matches_requested_precision() {
        assert_eq!(fixed_decimals(1.23456, 2), "1.23");
    }

    #[test]
    fn fmt_num_round_trips_value() {
        let value = 1.0 / 3.0;
        let encoded = fmt_num(value);
        let decoded = encoded.parse::<f64>().unwrap();
        assert_eq!(decoded.to_bits(), value.to_bits());
    }

    #[test]
    fn assumed_decimal_quantize_matches_encode_decode() {
        for value in [0.0, 3.1745e-5, -1.23456e-3, 9.999996e-5] {
            assert_eq!(
                assumed_decimal_quantize(value),
                decode_assumed_decimal_field(&fmt_assumed_decimal(value))
            );
        }
    }

    #[test]
    fn assumed_decimal_formats_zero_and_iss_bstar() {
        assert_eq!(fmt_assumed_decimal(0.0), " 00000-0");
        assert_eq!(fmt_assumed_decimal(3.1745e-5), " 31745-4");
        assert_eq!(
            decode_assumed_decimal_field(&fmt_assumed_decimal(3.1745e-5)),
            3.1745e-5
        );
    }

    #[test]
    fn assumed_decimal_rounding_carry_bumps_exponent() {
        let encoded = fmt_assumed_decimal(9.999996e-5);
        assert_eq!(encoded, " 10000-3");
        assert_eq!(decode_assumed_decimal_field(&encoded), 1.0e-4);
        assert_eq!(assumed_decimal_quantize(9.999996e-5), 1.0e-4);
    }
}
