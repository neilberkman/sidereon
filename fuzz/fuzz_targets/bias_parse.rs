#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::bias::BiasSet;

fuzz_target!(|data: &[u8]| {
    let _ = BiasSet::parse_bias_sinex(data);
    let _ = BiasSet::parse_code_dcb(data, None);
});
