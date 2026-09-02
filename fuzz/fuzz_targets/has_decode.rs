#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::has::HasMt1Message;

fuzz_target!(|data: &[u8]| {
    let _ = HasMt1Message::decode(data);
});
