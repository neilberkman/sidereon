#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::omm;

// Round-trip class: a parsed OMM must re-encode (in each format) to text that
// reparses to an equal value. A mismatch means the parser accepted state the
// encoder cannot faithfully reproduce (or the encoder is lossy).
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    if let Ok(original) = omm::parse_kvn(&text) {
        let reparsed =
            omm::parse_kvn(&omm::encode_kvn(&original)).expect("encoded OMM KVN must reparse");
        assert_eq!(reparsed, original);
    }

    if let Ok(original) = omm::parse_xml(&text) {
        let reparsed =
            omm::parse_xml(&omm::encode_xml(&original)).expect("encoded OMM XML must reparse");
        assert_eq!(reparsed, original);
    }

    if let Ok(original) = omm::parse_json(&text) {
        let reparsed =
            omm::parse_json(&omm::encode_json(&original)).expect("encoded OMM JSON must reparse");
        assert_eq!(reparsed, original);
    }
});
