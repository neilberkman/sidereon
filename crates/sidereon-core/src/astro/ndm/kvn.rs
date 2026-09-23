//! Line classification and unit handling shared by the CCSDS KVN readers.
//!
//! The ODM (CCSDS 502.0-B-3 section 7) and CDM (CCSDS 508.0-B-1 section 6)
//! share one line grammar: blank lines, `COMMENT` lines, `keyword = value`
//! assignments, and the structural or data lines a message defines for itself.
//! Units are optional documentation after a value, enclosed in square
//! brackets, and apply only to keywords whose table lists a unit.

/// One physical line of a CCSDS KVN message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KvnLine<'a> {
    /// A blank line, which carries no meaning (502.0-B-3 7.3.5).
    Blank,
    /// A `COMMENT` line. The value is the text after the keyword and its one
    /// separating blank; further leading whitespace belongs to the comment
    /// (502.0-B-3 7.8.5), and trailing whitespace is not significant (7.4.7).
    Comment(&'a str),
    /// A `keyword = value` assignment with the whitespace around the keyword,
    /// the equals sign, and the end of line removed (502.0-B-3 7.4.5-7.4.7).
    /// The value is split at the first `=`, so a value may itself contain `=`.
    Assignment {
        /// The keyword.
        key: &'a str,
        /// The value text, verbatim apart from the surrounding whitespace.
        value: &'a str,
    },
    /// Any other non-blank line, trimmed: a structural keyword such as
    /// `META_START`, an OEM data line, or text that is none of these.
    Other(&'a str),
}

/// Split a KVN message into physical lines. A line ends at a single carriage
/// return, a single line feed, a CR LF pair, or an LF CR pair (502.0-B-3 7.3.7,
/// 508.0-B-1 6.2.2.4). A final line without a terminator is kept.
pub(crate) fn kvn_lines(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut lines = Vec::new();
    let mut start = 0usize;
    let mut index = 0usize;
    while index < bytes.len() {
        let byte = bytes[index];
        if byte == b'\n' || byte == b'\r' {
            // Both terminators are ASCII, so every slice point is a character
            // boundary.
            lines.push(&text[start..index]);
            let pair = matches!(
                (byte, bytes.get(index + 1)),
                (b'\r', Some(b'\n')) | (b'\n', Some(b'\r'))
            );
            index += if pair { 2 } else { 1 };
            start = index;
        } else {
            index += 1;
        }
    }
    if start < bytes.len() {
        lines.push(&text[start..]);
    }
    lines
}

/// Classify one physical line.
pub(crate) fn classify(line: &str) -> KvnLine<'_> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return KvnLine::Blank;
    }
    if let Some(rest) = trimmed.strip_prefix("COMMENT") {
        if rest.is_empty() {
            return KvnLine::Comment("");
        }
        if rest.starts_with(|ch: char| ch.is_ascii_whitespace()) {
            // The separating blank is one ASCII byte, so slicing past it stays
            // on a character boundary.
            return KvnLine::Comment(&rest[1..]);
        }
    }
    match trimmed.split_once('=') {
        Some((key, value)) => KvnLine::Assignment {
            key: key.trim(),
            value: value.trim(),
        },
        None => KvnLine::Other(trimmed),
    }
}

/// Split a numeric KVN value into its number text and an optional trailing
/// `[unit]` (502.0-B-3 7.7.1.1, 508.0-B-1 6.3.3).
///
/// The unit is the text between the final `]` and the `[` that balances it, so
/// a unit that itself contains brackets, such as the OMM `BSTAR` unit
/// `1/[Earth radii]` of 502.0-B-3 table 4-3, stays whole. A value that does not
/// end in `]`, or whose brackets do not balance, carries no unit and is
/// returned unchanged for the number parser to judge.
pub(crate) fn split_unit(value: &str) -> (&str, Option<&str>) {
    let trimmed = value.trim_end();
    if !trimmed.ends_with(']') {
        return (trimmed, None);
    }
    let mut depth = 0usize;
    for (index, byte) in trimmed.bytes().enumerate().rev() {
        match byte {
            b']' => depth += 1,
            b'[' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    // `[` and `]` are ASCII, so both slice points are character
                    // boundaries.
                    let unit = trimmed[index + 1..trimmed.len() - 1].trim();
                    return (trimmed[..index].trim_end(), Some(unit));
                }
            }
            _ => {}
        }
    }
    (trimmed, None)
}

/// A unit that contradicts the unit a message table defines for a keyword.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct UnitMismatch {
    /// The unit the message stated.
    pub(crate) unit: String,
    /// The unit the table defines, or `None` for a dimensionless keyword.
    pub(crate) expected: Option<&'static str>,
}

/// Check a stated unit against the spellings a table allows for a keyword.
///
/// `allowed` is empty for a dimensionless keyword, for which the tables show
/// `n/a` and no unit may be displayed (502.0-B-3 7.7.1.2-7.7.1.3, 508.0-B-1
/// 6.2.4.2). A value without a unit is accepted: the table fixes the unit it
/// is read in. A stated unit must match one allowed spelling exactly, including
/// case (502.0-B-3 7.7.1.1, 508.0-B-1 6.2.4.1).
pub(crate) fn check_unit(
    unit: Option<&str>,
    allowed: &'static [&'static str],
) -> Result<(), UnitMismatch> {
    match unit {
        None => Ok(()),
        Some(unit) if allowed.contains(&unit) => Ok(()),
        Some(unit) => Err(UnitMismatch {
            unit: unit.to_string(),
            expected: allowed.first().copied(),
        }),
    }
}

/// Render the expected unit of a [`UnitMismatch`] for an error message.
pub(crate) fn expected_unit_label(expected: Option<&str>) -> String {
    match expected {
        Some(unit) => format!("[{unit}]"),
        None => "no unit".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_distinguishes_blank_comment_assignment_and_other_lines() {
        assert_eq!(classify("   "), KvnLine::Blank);
        assert_eq!(classify("COMMENT"), KvnLine::Comment(""));
        assert_eq!(
            classify("COMMENT  indented text  "),
            KvnLine::Comment(" indented text")
        );
        assert_eq!(
            classify(" OBJECT_NAME = A = B "),
            KvnLine::Assignment {
                key: "OBJECT_NAME",
                value: "A = B",
            }
        );
        assert_eq!(classify("META_START"), KvnLine::Other("META_START"));
        assert_eq!(
            classify("COMMENTARY = 1"),
            KvnLine::Assignment {
                key: "COMMENTARY",
                value: "1",
            }
        );
    }

    #[test]
    fn kvn_lines_accepts_every_terminator_of_7_3_7() {
        assert_eq!(
            kvn_lines("A\nB\rC\r\nD\n\rE"),
            vec!["A", "B", "C", "D", "E"]
        );
        assert_eq!(kvn_lines("A\r\n\r\nB\n"), vec!["A", "", "B"]);
        assert_eq!(kvn_lines(""), Vec::<&str>::new());
    }

    #[test]
    fn split_unit_keeps_nested_brackets_and_leaves_unbalanced_text() {
        assert_eq!(split_unit("7000.0 [km]"), ("7000.0", Some("km")));
        assert_eq!(split_unit("7000.0"), ("7000.0", None));
        assert_eq!(
            split_unit(".17172E-3 [1/[Earth radii]]"),
            (".17172E-3", Some("1/[Earth radii]"))
        );
        assert_eq!(split_unit("1 km]"), ("1 km]", None));
    }

    #[test]
    fn check_unit_accepts_listed_spellings_and_names_the_expected_unit() {
        assert_eq!(check_unit(None, &["km"]), Ok(()));
        assert_eq!(check_unit(Some("km"), &["km"]), Ok(()));
        assert_eq!(
            check_unit(Some("m"), &["km"]),
            Err(UnitMismatch {
                unit: "m".to_string(),
                expected: Some("km"),
            })
        );
        assert_eq!(
            check_unit(Some("n/a"), &[]),
            Err(UnitMismatch {
                unit: "n/a".to_string(),
                expected: None,
            })
        );
    }
}
