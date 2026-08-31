use nalgebra::DMatrix;
use sidereon_core::astro::math::portable;

fn matrix(rows: usize, cols: usize, seed: u64) -> DMatrix<f64> {
    let mut state = seed;
    DMatrix::from_fn(rows, cols, |_, _| {
        state = state
            .wrapping_mul(0xd1342543de82ef95)
            .wrapping_add(0xa4093822299f31d0);
        let unit = f64::from_bits(0x3ff0000000000000 | (state >> 12)) - 1.0;
        (unit * 2.0 - 1.0) * 0.5
    })
}

fn digest(matrix: &DMatrix<f64>) -> u64 {
    matrix.iter().fold(0xcbf29ce484222325, |hash, value| {
        (hash ^ value.to_bits()).wrapping_mul(0x100000001b3)
    })
}

#[test]
fn compare_dynamic_product_digests() {
    for &(rows, inner, cols, seed) in &[(64, 8, 64, 1), (200, 40, 200, 2)] {
        let lhs = matrix(rows, inner, seed);
        let rhs = matrix(inner, cols, seed + 10);
        let host = &lhs * &rhs;
        let portable = portable::product(&lhs, &rhs);
        println!(
            "product {rows}x{inner} by {inner}x{cols}: host={:016x} portable={:016x}",
            digest(&host),
            digest(&portable)
        );
    }
}
