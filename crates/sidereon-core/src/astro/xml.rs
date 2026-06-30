//! Minimal XML text helpers shared by the CCSDS message encoders (CDM, OMM).
//!
//! Decoding uses the `roxmltree` DOM reader; only encoding needs to escape the
//! handful of characters that are significant in element text. This lives in one
//! place so the message encoders do not each carry their own copy.

/// Return the first character that cannot appear in XML 1.0 text.
pub(crate) fn first_illegal_xml_1_0_char(value: &str) -> Option<char> {
    value.chars().find(|&ch| !is_xml_1_0_char(ch))
}

fn is_xml_1_0_char(ch: char) -> bool {
    matches!(
        ch,
        '\u{9}' | '\u{A}' | '\u{D}'
            | '\u{20}'..='\u{D7FF}'
            | '\u{E000}'..='\u{FFFD}'
            | '\u{10000}'..='\u{10FFFF}'
    )
}

/// Escape XML metacharacters in element text.
pub(crate) fn escape(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
        .replace('\r', "&#xD;")
}

/// Escape an optional string, treating `None` as empty.
pub(crate) fn escape_opt(value: &Option<String>) -> String {
    value.as_deref().map(escape).unwrap_or_default()
}
