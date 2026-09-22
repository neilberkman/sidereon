#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::antex::Antex;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let Ok(original) = Antex::parse(&text) else {
        return;
    };
    // The round-trip contract covers a product parsed without skipped or
    // inconsistent records, and the writer refuses a product it cannot state
    // exactly; there is nothing to round-trip then.
    if original.skipped_records() != 0 {
        return;
    }
    let Ok(encoded) = original.encode() else {
        return;
    };
    let reparsed = Antex::parse(&encoded).expect("encoded ANTEX must reparse");
    assert_eq!(reparsed.skipped_records(), 0);
    assert_eq!(reparsed, original);
    assert_eq!(
        reparsed.encode().expect("a written product rewrites"),
        encoded
    );
});
