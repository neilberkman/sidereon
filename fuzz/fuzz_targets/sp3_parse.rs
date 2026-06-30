#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let _ = sidereon_core::ephemeris::Sp3::parse(data);
});
