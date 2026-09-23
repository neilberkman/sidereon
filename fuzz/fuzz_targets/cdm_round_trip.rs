#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::cdm;

// Round-trip class: a parsed CDM must re-encode to text that reparses to an
// equal value. A refusal or a mismatch means the parser accepted state the
// encoder cannot faithfully reproduce (or the encoder is lossy).
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    if let Ok(original) = cdm::parse_kvn(&text) {
        let encoded = cdm::encode_kvn(&original).expect("a parsed CDM must encode as KVN");
        let reparsed = cdm::parse_kvn(&encoded).expect("encoded CDM KVN must reparse");
        assert_eq!(reparsed, original);
    }

    if let Ok(original) = cdm::parse_xml(&text) {
        let encoded = cdm::encode_xml(&original).expect("a parsed CDM must encode as XML");
        let reparsed = cdm::parse_xml(&encoded).expect("encoded CDM XML must reparse");
        assert_eq!(reparsed, original);
    }
});
