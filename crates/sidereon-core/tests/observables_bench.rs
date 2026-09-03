//! Manual throughput bench for the batch range hot path (Item 3).
//!
//! `#[ignore]`d on purpose: wall-clock is environment-sensitive, so this is NOT
//! a CI gate. It contrasts the vectorized [`predict_ranges`] kernel (which drops
//! the finite-difference velocity evaluation a range consumer never uses)
//! against the pre-optimization path that projected the range fields out of the
//! full [`transmit_time_satellite_state`] (velocity included). Both are
//! byte-identical on the range geometry (proven by
//! `observables::public_api_tests::predict_ranges_batch_matches_scalar_calls_bitwise`
//! and `predict_ranges_matches_transmit_time_loop_bitwise`); this measures the
//! throughput the removed ephemeris evaluations buy. Run:
//!
//! ```text
//! cargo test -p sidereon-core --release --test observables_bench -- --ignored --nocapture
//! ```

#![cfg(sidereon_repo_tests)]

use std::time::Instant;

use sidereon_core::ephemeris::Sp3;
use sidereon_core::observables::{
    predict_ranges, transmit_time_satellite_state, PredictOptions, RangePrediction,
    RangePredictionRequest, TransmitTimeOptions,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

fn sp3_fixture() -> Sp3 {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
    );
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
    Sp3::parse(&bytes).expect("parse SP3 fixture")
}

fn requests(sp3: &Sp3) -> Vec<RangePredictionRequest> {
    let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
    let rx = [3_512_900.0, 780_500.0, 5_248_700.0];
    // Build receive epochs strictly inside interpolation coverage: node epochs
    // and interior midpoints from the fixture's own grid, cycled to a large batch.
    let epochs = sp3.epochs_j2000_seconds();
    let mut times = Vec::new();
    for w in epochs.windows(2) {
        times.push(w[0]);
        times.push(0.5 * (w[0] + w[1]));
    }
    assert!(!times.is_empty(), "fixture has no covered epochs");
    (0..2_000)
        .map(|i| RangePredictionRequest::new(sat, rx, times[i % times.len()]))
        .collect()
}

#[test]
#[ignore = "manual throughput bench; run with --ignored --release"]
fn predict_ranges_throughput() {
    let sp3 = sp3_fixture();
    let requests = requests(&sp3);
    let options = PredictOptions::default();
    let mut tt_options = TransmitTimeOptions::default();
    tt_options.light_time = options.light_time;
    tt_options.sagnac = options.sagnac;
    let zero = RangePrediction {
        geometric_range_m: 0.0,
        sat_clock_s: None,
        transmit_time_j2000_s: 0.0,
        sat_pos_ecef_m: [0.0; 3],
    };
    let passes = 40usize;
    let total = passes * requests.len();

    // Warm up.
    let mut out = vec![zero; requests.len()];
    predict_ranges(&sp3, &requests, options, &mut out).expect("warmup");

    // Pre-optimization path: full transmit-time state (velocity finite-difference
    // included) projected to the range fields, in a loop.
    let mut acc = 0.0f64;
    let start = Instant::now();
    for _ in 0..passes {
        for request in &requests {
            let state = transmit_time_satellite_state(
                &sp3,
                request.sat,
                request.receiver_ecef_m,
                request.t_rx_j2000_s,
                tt_options,
            )
            .expect("transmit-time state");
            acc += state.geometric_range_m;
        }
    }
    let full_elapsed = start.elapsed();

    // Vectorized batch kernel.
    let start = Instant::now();
    for _ in 0..passes {
        predict_ranges(&sp3, &requests, options, &mut out).expect("batch ranges");
        acc += out[0].geometric_range_m;
    }
    let batch_elapsed = start.elapsed();

    std::hint::black_box(acc);

    let full_us = full_elapsed.as_secs_f64() * 1.0e6 / total as f64;
    let batch_us = batch_elapsed.as_secs_f64() * 1.0e6 / total as f64;
    let speedup = full_us / batch_us;
    eprintln!(
        "predict_ranges: {total} requests\n  full transmit-time (velocity): {full_us:.4} \
         us/request\n  vectorized range batch:        {batch_us:.4} us/request\n  speedup: \
         {speedup:.2}x"
    );
}
