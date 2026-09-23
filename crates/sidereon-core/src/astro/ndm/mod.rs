//! Shared CCSDS Navigation Data Message primitives.
//!
//! This crate-internal module holds header, epoch, covariance-block, KVN line
//! and unit, and scoped XML helpers shared across the CCSDS NDM family. OEM and
//! OPM readers use `read_lower_triangle6` for 6x6 state covariance blocks,
//! held exactly as read, and `mirror_lower_triangle6` to validate one; the
//! OMM, OPM, OEM and CDM readers share the KVN line classifier, the unit check,
//! and the scoped XML element access.

#![allow(dead_code, unused_imports)]

/// Shared CCSDS NDM covariance-block helpers.
pub(crate) mod covariance_block;
/// Shared CCSDS NDM epoch helpers.
pub(crate) mod epoch;
/// Shared CCSDS NDM header helpers.
pub(crate) mod header;
/// Shared CCSDS KVN line classification and unit helpers.
pub(crate) mod kvn;
/// Shared writer text checks.
pub(crate) mod text;
/// Shared scoped NDM/XML element access.
pub(crate) mod xml_tree;

/// Why a CCSDS NDM writer refuses a text value.
pub use text::TextIssue;

/// Re-export XML text helpers shared by NDM encoders.
pub(crate) use crate::astro::xml::{escape, escape_opt, first_illegal_xml_1_0_char};
/// Re-export KVN tokenization and field lookup helpers for NDM callers.
pub(crate) use crate::format::kvn::{tokenize, ConflictingField, FieldMap};
/// Re-export the shared 6x6 covariance block reader and writer.
pub(crate) use covariance_block::{
    covariance6_from_lower_triangle, covariance6_lower_triangle, covariance6_unit,
    mirror_lower_triangle6, read_covariance6, read_lower_triangle6, write_covariance6,
    COVARIANCE6_KEYS,
};
/// Re-export the shared CCSDS NDM epoch value.
pub(crate) use epoch::NdmEpoch;
/// Re-export the shared CCSDS NDM header value.
pub(crate) use header::NdmHeader;
/// Re-export the shared KVN line classifier and unit helpers.
pub(crate) use kvn::{
    check_unit, classify, expected_unit_label, kvn_lines, split_unit, KvnLine, UnitMismatch,
};
/// Re-export the scoped NDM/XML element helpers.
pub(crate) use xml_tree::{
    carries_data, child_element, child_elements, comment_text, element_children, leaf_descendants,
    leaf_text, message_elements, nested_element, units_attribute,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::covariance::Covariance6;
    use crate::validate::CivilSecondPolicy;

    #[test]
    fn root_reexports_header_epoch_covariance_and_text_helpers() {
        let header = NdmHeader::read(
            &FieldMap::parse("CCSDS_OMM_VERS = 2.0\nORIGINATOR = SIDEREON\n"),
            "CCSDS_OMM_VERS",
        );
        assert_eq!(header.vers, "2.0");
        assert_eq!(header.originator.as_deref(), Some("SIDEREON"));

        let epoch =
            NdmEpoch::parse("2026-06-17T04:32:52.099296Z", CivilSecondPolicy::UtcLike).unwrap();
        assert_eq!(epoch.to_iso8601(), "2026-06-17T04:32:52.099296");

        let covariance = Covariance6::from_diagonal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let lines = write_covariance6(&covariance);
        let recovered = read_covariance6(&FieldMap::from_pairs(tokenize(&lines.join("\n"))))
            .expect("covariance round-trip");
        assert_eq!(recovered.as_matrix(), covariance.as_matrix());

        assert_eq!(escape("<&>"), "&lt;&amp;&gt;");
        assert_eq!(escape_opt(&None), "");
        assert_eq!(first_illegal_xml_1_0_char("\u{0}"), Some('\u{0}'));
    }
}
