#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::oem;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    if let Ok(original) = oem::parse_kvn(&text) {
        let encoded = oem::encode_kvn(&original).expect("a parsed OEM must encode as KVN");
        let reparsed = oem::parse_kvn(&encoded).expect("encoded OEM KVN must reparse");
        // Skipped ephemeris lines are reported, not retained, so the encoder
        // has nothing to write for them and the reparse reports none.
        assert!(reparsed.skipped_states.is_empty());
        let mut expected = original;
        expected.skipped_states.clear();
        assert_eq!(reparsed, expected);
    }

    if let Ok(original) = oem::parse_xml(&text) {
        let encoded = oem::encode_xml(&original).expect("a parsed OEM must encode as XML");
        let reparsed = oem::parse_xml(&encoded).expect("encoded OEM XML must reparse");
        assert_eq!(reparsed, original);
    }
});
