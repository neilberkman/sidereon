#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let text = String::from_utf8_lossy(data);
    let _ = sidereon_core::astro::opm::parse_kvn(&text);
    let _ = sidereon_core::astro::opm::parse_xml(&text);
});
