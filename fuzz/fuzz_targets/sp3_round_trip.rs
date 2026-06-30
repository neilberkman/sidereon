#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::ephemeris::Sp3;

fuzz_target!(|data: &[u8]| {
    let Ok(original) = Sp3::parse(data) else {
        return;
    };

    let encoded = original.to_sp3_string();
    let reparsed = Sp3::parse(encoded.as_bytes()).expect("encoded SP3 must reparse");
    assert_eq!(reparsed, original);
});
