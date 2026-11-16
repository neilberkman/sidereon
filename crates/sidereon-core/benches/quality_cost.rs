//! Release cost of the chi-square inverse functions, in microseconds per call.
//! Run with `cargo bench -p sidereon-core --bench quality_cost`.

use sidereon_core::quality::{chi2_inv, chi2_isf};
use std::{hint::black_box, time::Instant};

fn main() {
    let probabilities = [
        1e-300,
        1e-100,
        1e-20,
        1e-9,
        1e-6,
        0.001,
        0.01,
        0.1,
        0.5,
        0.9,
        0.99,
        0.999,
        1.0 - 1e-9,
    ];
    // Warm all paths before timing short calls, including the normalization
    // cache and the large-shape recurrences.
    for dof in [1, 4, 8, 12, 30, 50, 100, 1000, 100000] {
        for _ in 0..64 {
            for p in probabilities {
                black_box(chi2_inv(black_box(p), black_box(dof)).unwrap());
                black_box(chi2_isf(black_box(p), black_box(dof)).unwrap());
            }
        }
    }
    println!("dof,chi2_inv_us,chi2_isf_us");
    for dof in [1, 4, 8, 12, 30, 50, 100, 1000, 100000] {
        let mut timings = Vec::new();
        for inverse in [chi2_inv, chi2_isf] {
            let start = Instant::now();
            for _ in 0..200 {
                for p in probabilities {
                    black_box(inverse(black_box(p), black_box(dof)).unwrap());
                }
            }
            timings.push(start.elapsed().as_secs_f64() * 1e6 / (200 * probabilities.len()) as f64);
        }
        println!("{dof},{:.3},{:.3}", timings[0], timings[1]);
    }
}
