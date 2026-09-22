#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::rinex::clock::RinexClock;

// Round-trip class: a product read from text restates that text byte for byte,
// whether the strict or the lossy reader produced it, and reading the restated
// text gives an equal product.
fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);

    let lossy = RinexClock::parse_lossy(&text);
    let restated = lossy
        .to_rinex_string()
        .expect("an unedited product restates its input");
    assert_eq!(restated, text);
    assert_eq!(RinexClock::parse_lossy(&restated), lossy);

    let Ok(strict) = RinexClock::parse(&text) else {
        return;
    };
    assert_eq!(strict, lossy);
    let encoded = strict
        .to_rinex_string()
        .expect("an unedited product restates its input");
    assert_eq!(encoded, text);
    let reparsed = RinexClock::parse(&encoded).expect("restated RINEX clock must reparse");
    assert_eq!(reparsed, strict);
});
