#![no_main]

use libfuzzer_sys::fuzz_target;

// Exercise the NAVCEN GPS constellation-status HTML parser on arbitrary bytes.
// The parser scans without an HTML crate, so the case-insensitive substring
// search and the `<tr>`/`<td>` slicing must stay panic-free on hostile input.
fuzz_target!(|data: &[u8]| {
    let _ = sidereon_core::constellation::parse_navcen(data);
});
