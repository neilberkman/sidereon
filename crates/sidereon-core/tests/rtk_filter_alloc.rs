//! Deterministic allocation gate for the RTK filter hot path.
//!
//! Counts heap allocations per `update_epoch` solve via a process-wide counting
//! allocator (scoped to this integration-test binary, so it does not affect the
//! crate or other tests). Allocations are deterministic, so this is a stable
//! CI-gateable number - it guards the "no per-solve allocation churn" rule by
//! catching regressions (allocations going UP). It is NOT a wall-clock benchmark;
//! the counting allocator inflates timing, so the solves/sec bench lives in its
//! own binary (`rtk_filter_bench.rs`) under the default allocator.

mod common;

use sidereon_core::rtk_filter::{
    update_epoch_with_scratch, AmbiguityScale, FilterState, RtkFilterScratch,
};
use std::alloc::{GlobalAlloc, Layout, System};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};

static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(layout)
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

#[test]
fn update_epoch_allocations_per_solve_bounded() {
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

    // Warm up to steady state (ambiguity columns added + fixes held).
    for _ in 0..5 {
        state = update_epoch_with_scratch(state, &epoch, base, &model, scale, &opts, &mut scratch)
            .unwrap()
            .state;
    }

    let n = 200;
    let before = ALLOCS.load(Ordering::Relaxed);
    for _ in 0..n {
        state = update_epoch_with_scratch(state, &epoch, base, &model, scale, &opts, &mut scratch)
            .unwrap()
            .state;
    }
    let per_solve = (ALLOCS.load(Ordering::Relaxed) - before) / n;
    eprintln!("rtk_filter update_epoch_with_scratch allocations/solve = {per_solve}");

    // Exact steady-state pin for the scratch-backed hot path.
    const EXPECTED: usize = 6;
    assert_eq!(per_solve, EXPECTED, "allocations/solve changed");
}
