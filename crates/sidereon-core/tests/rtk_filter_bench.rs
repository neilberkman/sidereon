//! Manual single-core throughput bench for the RTK filter hot path.
//!
//! `#[ignore]`d on purpose: wall-clock is environment-sensitive, so this is NOT
//! a CI gate - run it by hand to refresh the committed baseline:
//!
//! ```text
//! cargo test -p sidereon-core --release --test rtk_filter_bench -- --ignored --nocapture
//! ```
//!
//! The deterministic allocation count is the CI-gated metric (`rtk_filter_alloc.rs`);
//! this number is for release-note provenance only.

mod common;

use sidereon_core::rtk_filter::{
    update_epoch_with_scratch, AmbiguityScale, FilterState, RtkFilterScratch,
};
use std::collections::BTreeMap;
use std::time::Instant;

#[test]
#[ignore = "manual throughput bench; run with --ignored --release"]
fn update_epoch_throughput() {
    let (epoch, base, model, wl, off, opts) = common::inputs();
    let mut state = FilterState::new(
        BTreeMap::from([("G".to_string(), "G01".to_string())]),
        [-30.0, 25.0, -10.0],
        1.0e4,
        1.0e4,
    )
    .expect("valid RTK filter state");
    let mut scratch = RtkFilterScratch::new();
    let scale = AmbiguityScale {
        wavelengths_m: &wl,
        offsets_m: &off,
    };

    // Warm up (steady state + let the CPU clock up) before timing.
    for _ in 0..200 {
        state = update_epoch_with_scratch(state, &epoch, base, &model, scale, &opts, &mut scratch)
            .unwrap()
            .state;
    }

    let n = 200_000;
    let start = Instant::now();
    for _ in 0..n {
        state = update_epoch_with_scratch(state, &epoch, base, &model, scale, &opts, &mut scratch)
            .unwrap()
            .state;
    }
    let elapsed = start.elapsed();

    // Touch the final state so the optimizer cannot elide the loop.
    std::hint::black_box(&state);

    let per_solve_us = elapsed.as_secs_f64() * 1.0e6 / n as f64;
    let solves_per_sec = n as f64 / elapsed.as_secs_f64();
    eprintln!(
        "rtk_filter update_epoch_with_scratch: {n} solves in {:.3?} = {per_solve_us:.3} us/solve, \
         {solves_per_sec:.0} solves/sec/core (6-sat epoch)",
        elapsed
    );
}
