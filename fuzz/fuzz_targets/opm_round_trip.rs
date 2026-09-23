#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::opm;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    if let Ok(original) = opm::parse_kvn(&text) {
        let encoded = opm::encode_kvn(&original).expect("a parsed OPM must encode as KVN");
        let reparsed = opm::parse_kvn(&encoded).expect("encoded OPM KVN must reparse");
        assert_eq!(reparsed, original);
    }

    if let Ok(original) = opm::parse_xml(&text) {
        let encoded = opm::encode_xml(&original).expect("a parsed OPM must encode as XML");
        let reparsed = opm::parse_xml(&encoded).expect("encoded OPM XML must reparse");
        assert_eq!(reparsed, original);
    }
});
