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

/// Read a record laid out in fixed columns, when the line really is laid out
/// that way.
///
/// `fields` gives each field's half-open byte range, in order. The read succeeds
/// only when every non-space byte of `line` falls inside one of those ranges, so
/// a line whose content strays into the gaps a format reserves between its
/// fields, or past the last one, is reported as not being in this layout.
///
/// That test is what lets a column reader stand in front of a looser one without
/// changing how any loosely formatted line is read. A fixed-column record whose
/// fields abut - a satellite count filling its `I3` columns against the epoch
/// flag before it, say - is unreadable by splitting on whitespace, but a line
/// that is not in the layout at all is still left to whatever looser reading the
/// caller supports.
///
/// Ranges must be ordered and must not overlap. A field starting past the end of
/// the line reads as blank, which is how an omitted trailing field appears.
/// Fields are returned trimmed.
pub(crate) fn fixed_record<const N: usize>(
    line: &str,
    fields: [(usize, usize); N],
) -> Option<[&str; N]> {
    debug_assert!(
        fields.iter().all(|(start, end)| start <= end)
            && fields.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "fixed-column fields must be ordered and must not overlap"
    );
    // Columns are byte offsets, which only line up with characters while the
    // line is ASCII. A line carrying anything else is not read as a layout at
    // all, so a field boundary can never fall inside a character.
    if !line.is_ascii() {
        return None;
    }
    let outside_every_field = |index: usize| {
        !fields
            .iter()
            .any(|(start, end)| index >= *start && index < *end)
    };
    if line
        .bytes()
        .enumerate()
        .any(|(index, byte)| byte != b' ' && outside_every_field(index))
    {
        return None;
    }
    let mut read = [""; N];
    for (slot, (start, end)) in read.iter_mut().zip(fields) {
        *slot = field(line, start, end).unwrap_or_default();
    }
    Some(read)
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

    /// A RINEX 3 epoch line, laid out as the format specifies: the `>` marker,
    /// the six date and time fields, then the flag and satellite count with the
    /// two reserved columns before them.
    const EPOCH: [(usize, usize); 9] = [
        (0, 1),
        (2, 6),
        (7, 9),
        (10, 12),
        (13, 15),
        (16, 18),
        (18, 29),
        (31, 32),
        (32, 35),
    ];

    #[test]
    fn fixed_record_reads_fields_that_abut() {
        // A count of 100 fills its columns, so it touches the flag before it and
        // no whitespace reading can separate the two.
        assert_eq!(
            fixed_record("> 2020 06 24 00 00  0.0000000  0100", EPOCH),
            Some([">", "2020", "06", "24", "00", "00", "0.0000000", "0", "100"])
        );
        // The same line with the count clear of its first column reads the same
        // way, so the layout is not doing anything special for the merged case.
        assert_eq!(
            fixed_record("> 2020 06 24 00 00  0.0000000  0 99", EPOCH),
            Some([">", "2020", "06", "24", "00", "00", "0.0000000", "0", "99"])
        );
    }

    #[test]
    fn fixed_record_refuses_a_line_that_strays_outside_the_layout() {
        // Content in the columns the format reserves after the count means the
        // line is not in this layout, whatever else it may be. Both of these
        // have a reading of their own that a looser reader still gives them.
        for line in [
            "> 2020 06 24 00 00  0.0000000  0000 1",
            "> 2020 06 24 00 00  0.0000000  0100 0",
        ] {
            assert_eq!(fixed_record(line, EPOCH), None, "{line:?}");
        }
    }

    #[test]
    fn fixed_record_reads_a_blank_or_absent_trailing_field() {
        // Three F14.4 columns, the last left off the end of the line.
        let fields = [(0, 14), (14, 28), (28, 42)];
        assert_eq!(
            fixed_record("        0.0000-10000000.0000", fields),
            Some(["0.0000", "-10000000.0000", ""])
        );
    }

    #[test]
    fn fixed_record_refuses_a_line_whose_columns_are_not_characters() {
        // Byte offsets only line up with characters while the line is ASCII, so
        // a field boundary must never be allowed to fall inside one.
        assert_eq!(fixed_record("é", [(0, 1), (1, 2)]), None);
        assert_eq!(fixed_record("ab", [(0, 1), (1, 2)]), Some(["a", "b"]));
    }

    #[test]
    fn fixed_record_refuses_a_wider_layout_than_it_was_given() {
        // Fifteen-column fields: every fourteen-column slice still parses as a
        // number, so only the stray content reveals that the layout is wrong.
        let body = format!("{:15}{:15}{:15}", 12, 34, 56);
        assert_eq!(fixed_record(&body, [(0, 14), (14, 28), (28, 42)]), None);
    }

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
