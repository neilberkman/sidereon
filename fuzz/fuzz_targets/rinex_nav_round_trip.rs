#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::rinex::nav::{encode_nav, parse_nav};

// Round-trip class: a parsed broadcast-record set must re-encode to text that
// reparses to the same records.
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let Ok(original) = parse_nav(&text) else {
        return;
    };
    let encoded = encode_nav(&original);
    let reparsed = parse_nav(&encoded).expect("encoded RINEX NAV must reparse");
    assert_eq!(reparsed, original);
});
