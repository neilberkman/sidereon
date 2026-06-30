#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::cdm;

// Round-trip class: a parsed CDM must re-encode to text that reparses to an
// equal value. A mismatch means the parser accepted state the encoder cannot
// faithfully reproduce (or the encoder is lossy).
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    if let Ok(original) = cdm::parse_kvn(&text) {
        let Ok(encoded) = cdm::encode_kvn(&original) else {
            return;
        };
        let reparsed = cdm::parse_kvn(&encoded).expect("encoded CDM KVN must reparse");
        assert_eq!(reparsed, original);
    }

    if let Ok(original) = cdm::parse_xml(&text) {
        let Ok(encoded) = cdm::encode_xml(&original) else {
            return;
        };
        let reparsed = cdm::parse_xml(&encoded).expect("encoded CDM XML must reparse");
        assert_eq!(reparsed, original);
    }
});
