#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::atmosphere::Ionex;

fuzz_target!(|data: &[u8]| {
    let Ok(original) = Ionex::parse(data) else {
        return;
    };
    // Records skipped as unrepresentable (e.g. an AUX DATA block) are not
    // re-emitted, so they would not survive a round trip; the skip count is part
    // of `Ionex`'s derived equality. Restrict the invariant to clean products.
    if original.skipped_records() != 0 {
        return;
    }

    let encoded = original.to_ionex_string();
    let reparsed = Ionex::parse(encoded.as_bytes()).expect("encoded IONEX must reparse");
    assert_eq!(reparsed, original);
});
