#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::rinex::observations::RinexObs;

// Round-trip class: a parsed observation product must re-encode to text that
// reparses to an equal product. A mismatch means the serializer is lossy or the
// parser accepts state the serializer cannot reproduce.
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let Ok(original) = RinexObs::parse(&text) else {
        return;
    };
    // Records skipped as unrepresentable are not re-emitted, so they would not
    // survive a round trip; restrict the invariant to clean products.
    if original.skipped_records != 0 {
        return;
    }
    let encoded = original.to_rinex_string();
    let reparsed = RinexObs::parse(&encoded).expect("encoded RINEX OBS must reparse");
    assert_eq!(reparsed, original);
});
