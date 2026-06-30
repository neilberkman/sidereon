//! Native-Rust solve throughput for `trust_region_least_squares`.
//!
//! This times the crate's NATIVE path: the trust-region iteration driving a
//! pure-Rust `nalgebra` SVD (`tests/support/nalgebra_svd.rs`) with the crate's
//! own pure-Rust dot/matvec reductions — i.e. what a Rust user gets with no
//! Python and no injected LAPACK. It deliberately does NOT use the bit-exact
//! parity backend (that injects SciPy's own LAPACK/BLAS, so timing it would be
//! scipy-vs-scipy).
//!
//! The problems are loaded from `bench_problems.json`, the same set the SciPy
//! timing harness (`fixtures-generators/bench_scipy.py`) solves, so the two
//! numbers are directly comparable. Throughput is reported as solves/sec.
//!
//! Run with: `cargo bench -p trust-region-least-squares`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use serde_json::Value;

use trust_region_least_squares::loss::Loss;
use trust_region_least_squares::trf::{
    jacobian_2point, trf_no_bounds, JacobianFn, ResidualFn, TrfOptions,
};

#[path = "../tests/support/nalgebra_svd.rs"]
mod nalgebra_svd;
use nalgebra_svd::NalgebraSvd;

const PROBLEMS: &str = include_str!("bench_problems.json");

fn bits(value: &Value) -> f64 {
    let s = value.as_str().expect("hex-bit string");
    let raw = s.strip_prefix("0x").unwrap_or(s);
    f64::from_bits(u64::from_str_radix(raw, 16).expect("valid hex"))
}

fn bits_vec(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(bits).collect()
}

fn loss_from_str(name: &str) -> Loss {
    match name {
        "linear" => Loss::Linear,
        "soft_l1" => Loss::SoftL1,
        "huber" => Loss::Huber,
        "cauchy" => Loss::Cauchy,
        "arctan" => Loss::Arctan,
        other => panic!("unknown loss {other}"),
    }
}

/// Idiomatic residual model shared with the Python bench harness: a linear part
/// plus a mild elementwise nonlinearity. (Timing comparison, not bit-exact.)
fn residual_values(matrix: &[f64], target: &[f64], x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let m = target.len();
    let mut out = Vec::with_capacity(m);
    for i in 0..m {
        let mut acc = 0.0f64;
        let row = i * n;
        for j in 0..n {
            acc += matrix[row + j] * x[j];
        }
        out.push((acc - target[i]) + (0.25 * x[i % n]).sin());
    }
    out
}

struct Problem {
    name: String,
    loss: Loss,
    f_scale: f64,
    matrix: Vec<f64>,
    target: Vec<f64>,
    x0: Vec<f64>,
}

fn load_problems() -> Vec<Problem> {
    let doc: Value = serde_json::from_str(PROBLEMS).expect("parse bench problems");
    doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|case| Problem {
            name: case["name"].as_str().unwrap().to_string(),
            loss: loss_from_str(case["loss"].as_str().unwrap()),
            f_scale: bits(&case["f_scale"]),
            matrix: bits_vec(&case["matrix"]),
            target: bits_vec(&case["target"]),
            x0: bits_vec(&case["x0"]),
        })
        .collect()
}

fn solve_once(problem: &Problem) {
    let matrix = &problem.matrix;
    let target = &problem.target;
    let mut fun = |x: &[f64], out: &mut Vec<f64>| {
        out.clear();
        out.extend(residual_values(matrix, target, x));
    };
    let mut jac = |x: &[f64], f0: &[f64], out: &mut Vec<f64>| {
        let mut scratch = Vec::new();
        let mut jac_fun = |x: &[f64], out: &mut Vec<f64>| {
            out.clear();
            out.extend(residual_values(matrix, target, x));
        };
        jacobian_2point(&mut jac_fun, x, f0, out, &mut scratch).expect("jacobian");
    };
    let options = TrfOptions {
        loss: problem.loss,
        f_scale: problem.f_scale,
        ..TrfOptions::default()
    };
    let result = trf_no_bounds(
        &mut fun as &mut ResidualFn<'_>,
        &mut jac as &mut JacobianFn<'_>,
        &problem.x0,
        &NalgebraSvd,
        &options,
    )
    .expect("native solve");
    black_box(result);
}

fn bench_solves(c: &mut Criterion) {
    let problems = load_problems();
    let mut group = c.benchmark_group("native_solve");
    // Each iteration is one full least-squares solve; report solves/sec.
    group.throughput(Throughput::Elements(1));
    for problem in &problems {
        group.bench_with_input(
            BenchmarkId::from_parameter(&problem.name),
            problem,
            |b, p| b.iter(|| solve_once(p)),
        );
    }
    group.finish();
}

criterion_group!(benches, bench_solves);
criterion_main!(benches);
