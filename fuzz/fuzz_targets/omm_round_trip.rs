#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::omm;

// Round-trip class: a parsed OMM must re-encode (in each format) to text that
// reparses to an equal value. A mismatch means the parser accepted state the
// encoder cannot faithfully reproduce (or the encoder is lossy).
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    if let Ok(original) = omm::parse_kvn(&text) {
        let encoded = omm::encode_kvn(&original).expect("a parsed OMM must encode as KVN");
        let reparsed = omm::parse_kvn(&encoded).expect("encoded OMM KVN must reparse");
        assert_eq!(reparsed, original);

        // GP JSON and GP CSV carry one comment, a single header comment: a
        // record holding any other is refused rather than written without it.
        let has_comments = original.comments.header.len() > 1
            || !original.comments.metadata.is_empty()
            || !original.comments.mean_elements.is_empty()
            || !original.comments.tle_parameters.is_empty()
            || !original.comments.user_defined.is_empty()
            || original
                .spacecraft
                .as_ref()
                .is_some_and(|spacecraft| !spacecraft.comments.is_empty())
            || original
                .covariance
                .as_ref()
                .is_some_and(|covariance| !covariance.comments.is_empty());
        if has_comments {
            for refused in [
                omm::encode_json(&original),
                omm::encode_csv(std::slice::from_ref(&original)),
            ] {
                let refused_comment = match refused {
                    Err(omm::OmmError::UnwritableText { issue, .. }) => {
                        issue == omm::TextIssue::CommentNotCarried
                    }
                    Err(omm::OmmError::InRecord { source, .. }) => matches!(
                        *source,
                        omm::OmmError::UnwritableText {
                            issue: omm::TextIssue::CommentNotCarried,
                            ..
                        }
                    ),
                    _ => false,
                };
                assert!(refused_comment, "a comment must be refused, not dropped");
            }
        }
    }

    if let Ok(original) = omm::parse_xml(&text) {
        let encoded = omm::encode_xml(&original).expect("a parsed OMM must encode as XML");
        let reparsed = omm::parse_xml(&encoded).expect("encoded OMM XML must reparse");
        assert_eq!(reparsed, original);
    }

    if let Ok(original) = omm::parse_json(&text) {
        let encoded = omm::encode_json(&original).expect("a parsed OMM must encode as JSON");
        let reparsed = omm::parse_json(&encoded).expect("encoded OMM JSON must reparse");
        assert_eq!(reparsed, original);
        // A GP JSON record holds no comments, so discarding them changes nothing.
        assert_eq!(
            omm::encode_json_discarding_comments(&original).ok(),
            Some(encoded)
        );
    }

    if let Ok(original) = omm::parse_json_array(&text) {
        let encoded =
            omm::encode_json_array(&original.omms).expect("parsed OMMs must encode as JSON");
        let reparsed =
            omm::parse_json_array(&encoded).expect("encoded OMM JSON array must reparse");
        assert_eq!(reparsed.omms, original.omms);
        assert!(reparsed.skipped.is_empty());
        assert_eq!(
            omm::encode_json_array_discarding_comments(&original.omms).ok(),
            Some(encoded)
        );
    }

    if let Ok(original) = omm::parse_csv_array(&text) {
        let encoded = omm::encode_csv(&original.omms).expect("parsed OMMs must encode as CSV");
        let reparsed = omm::parse_csv_array(&encoded).expect("encoded OMM CSV must reparse");
        assert_eq!(reparsed.omms, original.omms);
        assert!(reparsed.skipped.is_empty());
    }
});
