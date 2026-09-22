#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::rinex::clock::{RinexClock, RinexClockError};

// Round-trip class: a parsed clock product must re-encode to text that reparses
// to an equal product (time scale + per-satellite series).
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let Ok(original) = RinexClock::parse(&text) else {
        return;
    };
    if !original.skipped_records.is_empty() {
        return;
    }
    let encoded = match original.to_rinex_string() {
        Ok(s) => s,
        Err(RinexClockError::InvalidInput { reason, .. })
            if reason.contains("without loss of precision") =>
        {
            return;
        }
        Err(err) => panic!("unexpected serialization error: {err:?}"),
    };
    let reparsed = RinexClock::parse(&encoded).expect("encoded RINEX clock must reparse");
    assert_eq!(reparsed, original);
});
