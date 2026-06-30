#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::spk::{parse_daf_spk, Spk};

// Exercise the DAF/SPK binary container parser plus the type-2/3/21 segment
// evaluators on arbitrary bytes. Parsing must never panic; queries derived from
// the parsed segments drive the in-coverage Chebyshev/MDA record readers so the
// binary-offset arithmetic is hit on hostile records, not just the header.
fuzz_target!(|data: &[u8]| {
    let _ = parse_daf_spk(data);

    let Ok(spk) = Spk::from_bytes(data) else {
        return;
    };

    for segment in spk.segments().iter().take(16) {
        let mid = 0.5 * (segment.start_et + segment.stop_et);
        for et in [
            segment.start_et,
            segment.stop_et,
            mid,
            0.0,
            f64::NAN,
            f64::INFINITY,
        ] {
            let _ = spk.spk_state(segment.target, segment.center, et);
        }
    }
});
