//! Text checks shared by the CCSDS NDM writers.
//!
//! A writer refuses a value that its reader would not return unchanged rather
//! than write text that reads back as something else.

use std::fmt;

/// Why a CCSDS NDM writer refuses a text value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextIssue {
    /// The text contains a carriage return or line feed, which ends a KVN line
    /// (CCSDS 502.0-B-3 7.3.7, 508.0-B-1 6.2.2.4).
    LineBreak,
    /// The text begins or ends with whitespace that the reader removes
    /// (502.0-B-3 7.4.5-7.4.7); a comment keeps leading whitespace but not
    /// trailing.
    SurroundingWhitespace,
    /// The text contains whitespace where the format separates items with
    /// whitespace, such as the epoch of an OEM ephemeris data line
    /// (502.0-B-3 5.2.4.3).
    InteriorWhitespace,
    /// A `USER_DEFINED_*` parameter name contains `=`, which the reader takes
    /// as the end of the keyword.
    KeywordSeparator,
    /// The text contains a character XML 1.0 cannot carry.
    XmlIllegalCharacter,
    /// A required value is empty and would read back as absent or as its
    /// default.
    Empty,
    /// Comments of a block with no keyword of that block after them would read
    /// back as comments of another block.
    DetachedComment,
    /// A `USER_DEFINED_*` parameter name occurs more than once. The readers
    /// keep one of two equal repeats and refuse two that differ, so the
    /// parameters would not read back as given.
    RepeatedParameter,
    /// The text is a comment and the encoding has no form for comments, as
    /// GP CSV has none, so writing it would drop it.
    CommentNotCarried,
}

impl fmt::Display for TextIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::LineBreak => "contains a line break",
            Self::SurroundingWhitespace => "has whitespace the reader would remove",
            Self::InteriorWhitespace => "contains whitespace that separates items",
            Self::KeywordSeparator => "contains the keyword separator '='",
            Self::XmlIllegalCharacter => "contains a character XML 1.0 cannot carry",
            Self::Empty => "is empty and would not read back",
            Self::DetachedComment => "has no keyword of its block to precede",
            Self::RepeatedParameter => "names a user-defined parameter given more than once",
            Self::CommentNotCarried => "is a comment, which the encoding cannot carry",
        };
        f.write_str(label)
    }
}

fn has_line_break(value: &str) -> bool {
    value.contains(['\n', '\r'])
}

/// The issue with a KVN value, which the reader trims.
pub(crate) fn kvn_value_issue(value: &str) -> Option<TextIssue> {
    if has_line_break(value) {
        Some(TextIssue::LineBreak)
    } else if value.trim() != value {
        Some(TextIssue::SurroundingWhitespace)
    } else {
        None
    }
}

/// The issue with a required KVN value, which must also be non-empty.
pub(crate) fn kvn_required_issue(value: &str) -> Option<TextIssue> {
    if value.is_empty() {
        Some(TextIssue::Empty)
    } else {
        kvn_value_issue(value)
    }
}

/// The issue with a KVN comment, whose trailing whitespace the reader removes.
pub(crate) fn kvn_comment_issue(comment: &str) -> Option<TextIssue> {
    if has_line_break(comment) {
        Some(TextIssue::LineBreak)
    } else if comment.trim_end() != comment {
        Some(TextIssue::SurroundingWhitespace)
    } else {
        None
    }
}

/// The issue with a `USER_DEFINED_*` parameter name in KVN.
pub(crate) fn kvn_parameter_issue(parameter: &str) -> Option<TextIssue> {
    if has_line_break(parameter) {
        Some(TextIssue::LineBreak)
    } else if parameter.contains('=') {
        Some(TextIssue::KeywordSeparator)
    } else if parameter.trim_end() != parameter {
        Some(TextIssue::SurroundingWhitespace)
    } else {
        None
    }
}

/// The issue with XML element text, which the reader trims.
pub(crate) fn xml_value_issue(value: &str) -> Option<TextIssue> {
    if value.trim() != value {
        Some(TextIssue::SurroundingWhitespace)
    } else if crate::astro::xml::first_illegal_xml_1_0_char(value).is_some() {
        Some(TextIssue::XmlIllegalCharacter)
    } else {
        None
    }
}

/// The issue with required XML element text, which must also be non-empty.
pub(crate) fn xml_required_issue(value: &str) -> Option<TextIssue> {
    if value.is_empty() {
        Some(TextIssue::Empty)
    } else {
        xml_value_issue(value)
    }
}

/// The issue with an XML `COMMENT`, whose trailing whitespace the reader
/// removes.
pub(crate) fn xml_comment_issue(comment: &str) -> Option<TextIssue> {
    if comment.trim_end() != comment {
        Some(TextIssue::SurroundingWhitespace)
    } else if crate::astro::xml::first_illegal_xml_1_0_char(comment).is_some() {
        Some(TextIssue::XmlIllegalCharacter)
    } else {
        None
    }
}

/// The issue with an XML attribute value, which the reader takes verbatim.
pub(crate) fn xml_attribute_issue(value: &str) -> Option<TextIssue> {
    crate::astro::xml::first_illegal_xml_1_0_char(value).map(|_| TextIssue::XmlIllegalCharacter)
}

/// Escape text for an XML attribute value. Beyond element-text escaping, tab
/// and line feed become character references, since XML attribute-value
/// normalization would otherwise read them back as spaces (XML 1.0 3.3.3).
pub(crate) fn escape_attribute(value: &str) -> String {
    crate::astro::xml::escape(value)
        .replace('\t', "&#x9;")
        .replace('\n', "&#xA;")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kvn_and_xml_checks_name_each_issue() {
        assert_eq!(kvn_value_issue("A B"), None);
        assert_eq!(kvn_value_issue("A\nB"), Some(TextIssue::LineBreak));
        assert_eq!(
            kvn_value_issue(" A"),
            Some(TextIssue::SurroundingWhitespace)
        );
        assert_eq!(kvn_required_issue(""), Some(TextIssue::Empty));
        assert_eq!(kvn_comment_issue("  indented"), None);
        assert_eq!(
            kvn_comment_issue("x "),
            Some(TextIssue::SurroundingWhitespace)
        );
        assert_eq!(
            kvn_parameter_issue("A=B"),
            Some(TextIssue::KeywordSeparator)
        );
        assert_eq!(xml_value_issue("A\nB"), None);
        assert_eq!(
            xml_value_issue("A\u{1}"),
            Some(TextIssue::XmlIllegalCharacter)
        );
        assert_eq!(xml_required_issue(""), Some(TextIssue::Empty));
        assert_eq!(xml_comment_issue(" lead"), None);
        assert_eq!(xml_attribute_issue(" A "), None);
        assert_eq!(escape_attribute("a\tb\nc"), "a&#x9;b&#xA;c");
    }
}
