#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = sidereon_core::data::parse_archive_listing(&text);
    let _ = sidereon_core::data::parse_skadi_tile_id(&text);
});
