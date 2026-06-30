#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::oem;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    if let Ok(original) = oem::parse_kvn(&text) {
        let reparsed =
            oem::parse_kvn(&oem::encode_kvn(&original)).expect("encoded OEM KVN must reparse");
        assert_eq!(reparsed, original);
    }

    if let Ok(original) = oem::parse_xml(&text) {
        let reparsed =
            oem::parse_xml(&oem::encode_xml(&original)).expect("encoded OEM XML must reparse");
        assert_eq!(reparsed, original);
    }
});
