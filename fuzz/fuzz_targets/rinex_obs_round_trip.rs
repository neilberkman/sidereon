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
    let mut reparsed = RinexObs::parse(&encoded).expect("encoded RINEX OBS must reparse");
    // The labels of header records that were read and not retained describe the
    // source text rather than the product, and the writer does not re-emit them,
    // so a clean re-parse reports none. Normalise that one diagnostic rather
    // than skipping the whole product, which would stop checking its records.
    let mut original = original;
    original.header.unretained_header_labels.clear();
    reparsed.header.unretained_header_labels.clear();
    assert_eq!(reparsed, original);
});
