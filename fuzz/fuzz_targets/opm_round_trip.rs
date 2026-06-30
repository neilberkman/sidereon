#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::opm;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    if let Ok(original) = opm::parse_kvn(&text) {
        let reparsed =
            opm::parse_kvn(&opm::encode_kvn(&original)).expect("encoded OPM KVN must reparse");
        assert_eq!(reparsed, original);
    }

    if let Ok(original) = opm::parse_xml(&text) {
        let reparsed =
            opm::parse_xml(&opm::encode_xml(&original)).expect("encoded OPM XML must reparse");
        assert_eq!(reparsed, original);
    }
});
