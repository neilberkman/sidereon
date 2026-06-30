//! Fixed-width column helpers for forgiving format readers.

use crate::validate::FieldError;

/// Largest char boundary `<= index` and `<= line.len()`.
pub(crate) fn floor_char_boundary(line: &str, index: usize) -> usize {
    let mut i = index.min(line.len());
    while i > 0 && !line.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Return a trimmed fixed-column field, or `None` for a blank or missing range.
pub(crate) fn field(line: &str, start: usize, end: usize) -> Option<&str> {
    let s = floor_char_boundary(line, start);
    let e = floor_char_boundary(line, end);
    if e <= s {
        return None;
    }
    let value = line[s..e].trim();
    if value.is_empty() {
        None
    } else {
        Some(value)
    }
}

/// Return an untrimmed fixed-column field, or `""` for an empty range.
pub(crate) fn raw_field(line: &str, start: usize, end: usize) -> &str {
    let s = floor_char_boundary(line, start);
    let e = floor_char_boundary(line, end);
    if e <= s {
        return "";
    }
    &line[s..e]
}

/// Return the untrimmed remainder of a fixed-column line.
pub(crate) fn raw_field_from(line: &str, start: usize) -> &str {
    let s = floor_char_boundary(line, start);
    &line[s..]
}

/// Return an inclusive byte-window slice without panicking on UTF-8 input.
pub(crate) fn slice_inclusive(line: &str, start: usize, end_inclusive: usize) -> &str {
    raw_field(line, start, end_inclusive.saturating_add(1))
}

/// Return the char whose byte offset starts at `index`.
pub(crate) fn char_at(line: &str, index: usize) -> Option<char> {
    if !line.is_char_boundary(index) {
        return None;
    }
    line.get(index..)?.chars().next()
}

/// Parse a numeric value with the lenient validator and caller-supplied label.
///
/// This accepts Fortran `D`/`d` exponents, integer-only fields, and leading-dot
/// fields while rejecting non-finite values. Keep this distinct from
/// [`reference_float`], which intentionally preserves SGP4 reference behavior.
pub(crate) fn strict_f64(value: &str, field: &'static str) -> Result<f64, FieldError> {
    crate::validate::strict_f64(value, field)
}

/// Parse a float with SGP4 reference-compatible restrictions.
///
/// This strips a leading `+`, handles `-` normally, rejects integer-only input,
/// and rejects leading-dot or trailing-dot forms. Keep this distinct from
/// [`strict_f64`], which is the lenient numeric reader.
pub(crate) fn reference_float(text: &str, field: &'static str) -> Result<f64, FieldError> {
    let trimmed = text.trim();
    let normalized = trimmed.strip_prefix('+').unwrap_or(trimmed);
    if !normalized.contains('.') {
        return float_parse_error(text, field);
    }
    let body = normalized.strip_prefix('-').unwrap_or(normalized);
    if body.starts_with('.') || body.ends_with('.') {
        return float_parse_error(text, field);
    }
    normalized
        .parse::<f64>()
        .map_err(|_| FieldError::FloatParse {
            field,
            value: text.to_string(),
        })
}

/// Parse a fixed-column Fortran-style float, returning `None` on failure.
pub(crate) fn fortran_f64(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
) -> Option<f64> {
    let s = self::field(line, start, end)?;
    strict_f64(s, field).ok()
}

fn float_parse_error<T>(text: &str, field: &'static str) -> Result<T, FieldError> {
    Err(FieldError::FloatParse {
        field,
        value: text.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn out_of_bounds_ranges_are_empty() {
        assert_eq!(field("abc", 8, 10), None);
        assert_eq!(field("abc", 1, 1), None);
        assert_eq!(field("a   ", 1, 4), None);
        assert_eq!(raw_field("abc", 8, 10), "");
        assert_eq!(raw_field_from("abc", 8), "");
        assert_eq!(slice_inclusive("abc", 8, 10), "");
    }

    #[test]
    fn multibyte_ranges_respect_char_boundaries() {
        let line = "é🙂abc";
        assert_eq!(raw_field(line, 1, 7), "é🙂a");
        assert_eq!(field(line, 1, 7), Some("é🙂a"));
        assert_eq!(slice_inclusive(line, 1, 6), "é🙂a");
        assert_eq!(char_at(line, 1), None);
        assert_eq!(char_at(line, 2), Some('🙂'));
        assert_eq!(char_at(line, 6), Some('a'));
    }

    #[test]
    fn reference_float_rejects_reference_invalid_forms() {
        assert!(matches!(
            reference_float("5", "value"),
            Err(FieldError::FloatParse { field: "value", .. })
        ));
        assert!(matches!(
            reference_float(".5", "value"),
            Err(FieldError::FloatParse { field: "value", .. })
        ));
        assert!(matches!(
            reference_float("5.", "value"),
            Err(FieldError::FloatParse { field: "value", .. })
        ));
    }

    #[test]
    fn reference_float_accepts_reference_valid_forms() {
        assert_eq!(reference_float("5.0", "value"), Ok(5.0));
        assert_eq!(reference_float("-5.0", "value"), Ok(-5.0));
        assert_eq!(reference_float("+5.0", "value"), Ok(5.0));
    }

    #[test]
    fn strict_f64_accepts_lenient_numeric_forms() {
        assert_eq!(strict_f64("5", "numeric field"), Ok(5.0));
        assert_eq!(strict_f64("1.25D+03", "numeric field"), Ok(1250.0));
        assert_eq!(
            strict_f64("NaN", "numeric field"),
            Err(FieldError::NonFinite {
                field: "numeric field"
            })
        );
    }

    #[test]
    fn fortran_f64_reads_fixed_columns() {
        assert_eq!(
            fortran_f64("xx 1.25D+03 yy", 3, 12, "numeric field"),
            Some(1250.0)
        );
        assert_eq!(fortran_f64("xx NaN yy", 3, 6, "numeric field"), None);
    }
}
