#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::rinex::nav::{encode_nav, encode_nav_file, parse_nav, parse_nav_file};

// Round-trip class: a parsed broadcast-record set must reach a stable
// canonical encoding. Input fields can carry more precision than the fixed
// RINEX columns preserve, so comparing the first parse directly is too strong.
// A whole file read with `parse_nav_file` is written back byte for byte.
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    if let Ok(file) = parse_nav_file(&text) {
        let written = encode_nav_file(&file).expect("a file as read is representable");
        assert_eq!(written, text);
    }
    let Ok(original) = parse_nav(&text) else {
        return;
    };
    let encoded = encode_nav(&original).expect("a parsed record set is representable");
    let reparsed = parse_nav(&encoded).expect("encoded RINEX NAV must reparse");
    assert_eq!(
        encode_nav(&reparsed).expect("a reparsed record set is representable"),
        encoded
    );
});
