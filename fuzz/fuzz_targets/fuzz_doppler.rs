#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::doppler;
use sidereon_core::astro::time::TimeScales;

#[derive(Debug, Arbitrary)]
struct Input {
    ts: [f64; 7],
    pos: [f64; 3],
    vel: [f64; 3],
    station: [f64; 3],
    frequency_hz: f64,
}

fn time_scales(raw: [f64; 7]) -> TimeScales {
    TimeScales {
        jd_whole: raw[0],
        ut1_fraction: raw[1],
        tt_fraction: raw[2],
        tdb_fraction: raw[3],
        jd_ut1: raw[4],
        jd_tt: raw[5],
        jd_tdb: raw[6],
    }
}

// Doppler is a compute path, not a parser: degenerate but finite inputs (e.g. a
// zero range vector) can legitimately produce non-finite outputs, so the bar is
// only that the geometry/frame transport never panics on arbitrary finite bytes.
fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };
    let ts = time_scales(input.ts);

    let _ = doppler::range_rate_and_ratio(
        input.pos,
        input.vel,
        input.station[0],
        input.station[1],
        input.station[2],
        &ts,
    );
    let _ = doppler::doppler_shift(
        input.pos,
        input.vel,
        input.station[0],
        input.station[1],
        input.station[2],
        &ts,
        input.frequency_hz,
    );
});
