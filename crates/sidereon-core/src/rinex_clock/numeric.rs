//! Fortran `E19.12` clock-value formatting with exact readback.

use crate::validate;

use super::{invalid_input, RinexClockError};

pub(super) fn field_name_for_value_index(idx: usize) -> &'static str {
    match idx {
        0 => "bias",
        1 => "sigma",
        2 => "rate",
        3 => "rate_sigma",
        4 => "acceleration",
        5 => "acceleration_sigma",
        _ => "additional_values",
    }
}

pub(super) fn format_e19_12(value: f64, field: &'static str) -> Result<String, RinexClockError> {
    if !value.is_finite() {
        return Err(invalid_input(field, "must be finite"));
    }
    if value == 0.0 {
        let sign = if value.is_sign_negative() { '-' } else { ' ' };
        return Ok(format!("{sign}0.000000000000E+00"));
    }

    let sign = if value.is_sign_negative() { '-' } else { ' ' };
    let abs_val = value.abs();

    if let Some(formatted) = try_format_leading_zero(sign, abs_val) {
        if formatted.len() == 19 {
            if let Ok(reparsed) = formatted.trim().parse::<f64>() {
                if reparsed.to_bits() == value.to_bits() {
                    return Ok(formatted);
                }
            }
        }
    }

    if let Some(formatted) = try_format_nonzero_leading(sign, abs_val) {
        if formatted.len() == 19 {
            if let Ok(reparsed) = formatted.trim().parse::<f64>() {
                if reparsed.to_bits() == value.to_bits() {
                    return Ok(formatted);
                }
            }
        }
    }

    if let Some(formatted) = try_format_3digit_exp(sign, abs_val) {
        if formatted.len() == 19 {
            if let Ok(reparsed) = formatted.trim().parse::<f64>() {
                if reparsed.to_bits() == value.to_bits() {
                    return Ok(formatted);
                }
            }
        }
    }

    if let Some(formatted) = try_format_canonical_scientific(sign, abs_val) {
        if formatted.len() == 19 {
            if let Ok(reparsed) = formatted.trim().parse::<f64>() {
                if reparsed.to_bits() == value.to_bits() {
                    return Ok(formatted);
                }
            }
        }
    }

    if let Some(formatted) = try_format_scientific_fallback(value) {
        return Ok(formatted);
    }

    Err(invalid_input(
        field,
        "value cannot be represented in Fortran E19.12 format without loss of precision",
    ))
}

fn try_format_leading_zero(sign: char, abs_val: f64) -> Option<String> {
    let s = format!("{abs_val:.11e}");
    let (mantissa_str, exp_str) = s.split_once('e')?;
    let rust_exp: i32 = exp_str.parse().ok()?;
    let (d0, rest) = mantissa_str.split_once('.')?;
    let new_exp = rust_exp + 1;
    if !(-99..=99).contains(&new_exp) {
        return None;
    }
    let formatted_exp = if new_exp >= 0 {
        format!("E+{new_exp:02}")
    } else {
        format!("E-{:02}", new_exp.abs())
    };
    Some(format!("{sign}0.{d0}{rest}{formatted_exp}"))
}

fn try_format_nonzero_leading(sign: char, abs_val: f64) -> Option<String> {
    let s = format!("{abs_val:.12e}");
    let (mantissa_str, exp_str) = s.split_once('e')?;
    let rust_exp: i32 = exp_str.parse().ok()?;
    if !(-99..=99).contains(&rust_exp) {
        return None;
    }
    let formatted_exp = if rust_exp >= 0 {
        format!("E+{rust_exp:02}")
    } else {
        format!("E-{:02}", rust_exp.abs())
    };
    Some(format!("{sign}{mantissa_str}{formatted_exp}"))
}

fn try_format_3digit_exp(sign: char, abs_val: f64) -> Option<String> {
    let s = format!("{abs_val:.11e}");
    let (mantissa_str, exp_str) = s.split_once('e')?;
    let rust_exp: i32 = exp_str.parse().ok()?;
    let (d0, rest) = mantissa_str.split_once('.')?;
    let new_exp = rust_exp + 1;
    if !(-999..=-100).contains(&new_exp) && !(100..=999).contains(&new_exp) {
        return None;
    }
    let formatted_exp = if new_exp >= 0 {
        format!("E+{new_exp:03}")
    } else {
        format!("E-{:03}", new_exp.abs())
    };
    Some(format!("{sign}.{d0}{rest}{formatted_exp}"))
}

fn try_format_canonical_scientific(sign: char, abs_val: f64) -> Option<String> {
    let s = format!("{abs_val:.12e}");
    let (mantissa_str, exp_str) = s.split_once('e')?;
    let rust_exp: i32 = exp_str.parse().ok()?;
    let formatted_exp = if (-99..=99).contains(&rust_exp) {
        if rust_exp >= 0 {
            format!("E+{rust_exp:02}")
        } else {
            format!("E-{:02}", rust_exp.abs())
        }
    } else if (-999..=999).contains(&rust_exp) {
        if rust_exp >= 0 {
            format!("E+{rust_exp:03}")
        } else {
            format!("E-{:03}", rust_exp.abs())
        }
    } else {
        return None;
    };

    let raw = if sign == '-' {
        format!("-{mantissa_str}{formatted_exp}")
    } else {
        format!("{mantissa_str}{formatted_exp}")
    };

    if raw.len() > 19 {
        return None;
    }

    Some(format!("{raw:>19}"))
}

/// Formats a finite non-zero floating-point value into an exact 19-column
/// scientific representation when standard preferred formatters cannot fit within 19 bytes.
/// Evaluates finite 1..=17 significant digit candidates strictly containing an
/// explicit decimal point and 'E', testing finite point placement and exponent
/// adjustments alongside optional positive plus signs for input compatibility rather
/// than canonical Fortran output. Candidates of length <= 19 bytes are left-padded with
/// spaces to exactly 19 bytes and accepted only on strict bit readback (`to_bits()`).
///
/// This does not guarantee that all legally representable mathematical values fit
/// within the 19-column budget; values exceeding candidate width limits are refused.
fn try_format_scientific_fallback(value: f64) -> Option<String> {
    let abs_val = value.abs();
    let sign_prefix = if value.is_sign_negative() { "-" } else { "" };

    for sig_digits in (1..=17).rev() {
        let prec = sig_digits - 1;
        let s = format!("{abs_val:.prec$e}");
        let Some((mantissa_part, exp_part)) = s.split_once('e') else {
            continue;
        };
        let Ok(rust_exp) = exp_part.parse::<i32>() else {
            continue;
        };
        let digits: String = mantissa_part
            .chars()
            .filter(|c| c.is_ascii_digit())
            .collect();
        if digits.len() != sig_digits {
            continue;
        }

        let mut mantissa_candidates = Vec::with_capacity(sig_digits + 2);

        // 1. Standard normalized form (decimal point after first digit).
        if sig_digits == 1 {
            mantissa_candidates.push((format!("{sign_prefix}{digits}."), 0));
        } else {
            mantissa_candidates
                .push((format!("{sign_prefix}{}.{}", &digits[..1], &digits[1..]), 0));
        }

        // 2. Leading zero form (0.dddd...).
        mantissa_candidates.push((format!("{sign_prefix}0.{digits}"), 1));

        // 3. Leading dot form (.dddd...).
        mantissa_candidates.push((format!("{sign_prefix}.{digits}"), 1));

        // 4. Shift decimal point to the right across the remaining positions.
        if sig_digits > 1 {
            for k in 2..=sig_digits {
                let exp_delta = -(k as i32 - 1);
                if k < sig_digits {
                    mantissa_candidates.push((
                        format!("{sign_prefix}{}.{}", &digits[..k], &digits[k..]),
                        exp_delta,
                    ));
                } else {
                    mantissa_candidates.push((format!("{sign_prefix}{digits}."), exp_delta));
                }
            }
        }

        for (mantissa, exp_delta) in mantissa_candidates {
            let adj_exp = rust_exp + exp_delta;
            if !(-999..=999).contains(&adj_exp) {
                continue;
            }

            let mut exp_spellings = Vec::with_capacity(4);
            if adj_exp >= 0 {
                if adj_exp <= 99 {
                    exp_spellings.push(format!("E+{adj_exp:02}"));
                    exp_spellings.push(format!("E{adj_exp:02}"));
                    exp_spellings.push(format!("E+{adj_exp:03}"));
                    exp_spellings.push(format!("E{adj_exp:03}"));
                } else {
                    exp_spellings.push(format!("E+{adj_exp:03}"));
                    exp_spellings.push(format!("E{adj_exp:03}"));
                }
            } else {
                let abs_exp = adj_exp.unsigned_abs();
                if abs_exp <= 99 {
                    exp_spellings.push(format!("E-{abs_exp:02}"));
                    exp_spellings.push(format!("E-{abs_exp:03}"));
                } else {
                    exp_spellings.push(format!("E-{abs_exp:03}"));
                }
            }

            for exp_spelling in exp_spellings {
                let candidate_raw = format!("{mantissa}{exp_spelling}");
                if candidate_raw.len() > 19 {
                    continue;
                }
                let candidate = format!("{candidate_raw:>19}");
                if let Ok(reparsed) = validate::strict_f64(&candidate, "bias") {
                    if reparsed.to_bits() == value.to_bits() {
                        return Some(candidate);
                    }
                }
            }
        }
    }

    None
}
