#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::rinex::clock::RinexClock;

// Round-trip class: a parsed clock product must re-encode to text that reparses
// to an equal product (time scale + per-satellite series).
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let Ok(original) = RinexClock::parse(&text) else {
        return;
    };
    let encoded = original.to_rinex_string();
    let reparsed = RinexClock::parse(&encoded).expect("encoded RINEX clock must reparse");
    assert_eq!(reparsed, original);
});
