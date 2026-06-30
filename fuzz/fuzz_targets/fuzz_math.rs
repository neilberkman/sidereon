#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use nalgebra::DVector;
use sidereon_core::astro::math::{
    least_squares::{
        cost, fd_steps, jacobian_2point, solve_trf, solve_trf_with, LeastSquaresProblem,
        SolveOptions, TrustRegionSolve,
    },
    linear, mat3, robust, vec3,
};

#[derive(Debug, Arbitrary)]
struct Input {
    scalars: [f64; 12],
    a3: [f64; 3],
    b3: [f64; 3],
    m3a: [[f64; 3]; 3],
    m3b: [[f64; 3]; 3],
    m3c: [[f64; 3]; 3],
    m4: [[f64; 4]; 4],
    values: Vec<f64>,
    rhs: Vec<f64>,
    weights: Vec<f64>,
    rows4: Vec<[f64; 4]>,
    dims: [u8; 4],
}

fn finite_clamped(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(-1.0e5, 1.0e5)
    } else {
        0.0
    }
}

fn finite_clamped3(values: [f64; 3]) -> [f64; 3] {
    values.map(finite_clamped)
}

fn finite_clamped_mat3(values: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    values.map(finite_clamped3)
}

fn finite_clamped4(values: [f64; 4]) -> [f64; 4] {
    values.map(finite_clamped)
}

fn finite_clamped_mat4(values: [[f64; 4]; 4]) -> [[f64; 4]; 4] {
    values.map(finite_clamped4)
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };

    let n = bounded_usize(input.dims[0], 1, MAX_MATRIX_DIM);
    let rows = bounded_usize(input.dims[1], 1, MAX_MATRIX_DIM);
    let cols = bounded_usize(input.dims[2], 1, MAX_MATRIX_DIM);
    let a = square_from_flat(&input.values, n);
    let bvec: Vec<f64> = (0..n)
        .map(|idx| {
            input
                .rhs
                .get(idx)
                .copied()
                .unwrap_or(input.scalars[idx % 12])
        })
        .collect();
    let bmat = matrix_from_flat(&input.rhs, n, cols);
    let rect_a = matrix_from_flat(&input.values, rows, cols);
    let rect_b = matrix_from_flat(&input.rhs, rows, cols);

    assert_ok_finite_or_err("vec3::checked_add3", vec3::checked_add3(input.a3, input.b3));
    let a3 = finite_clamped3(input.a3);
    let b3 = finite_clamped3(input.b3);
    let scale = finite_clamped(input.scalars[0]);
    assert_success("vec3::add3", vec3::add3(a3, b3));
    assert_success("vec3::sub3", vec3::sub3(a3, b3));
    assert_success("vec3::neg3", vec3::neg3(a3));
    assert_success("vec3::scale3", vec3::scale3(a3, scale));
    assert_success("vec3::dot3", vec3::dot3(a3, b3));
    assert_success("vec3::norm3", vec3::norm3(a3));
    assert_option_finite("vec3::unit3", vec3::unit3(a3));
    assert_success("vec3::cross3", vec3::cross3(a3, b3));

    let m3a = finite_clamped_mat3(input.m3a);
    let m3b = finite_clamped_mat3(input.m3b);
    let m3c = finite_clamped_mat3(input.m3c);
    assert_success("mat3::inline_rxr", mat3::inline_rxr(&m3a, &m3b));
    assert_success("mat3::inline_tr", mat3::inline_tr(&m3a));
    assert_success("mat3::inline_mxmxm", mat3::inline_mxmxm(&m3a, &m3b, &m3c));

    assert_option_finite(
        "linear::solve_linear_first_tie",
        linear::solve_linear_first_tie(&a, &bvec),
    );
    assert_option_finite(
        "linear::solve_linear_last_tie",
        linear::solve_linear_last_tie(a.clone(), bvec.clone()),
    );
    assert_option_finite(
        "linear::invert_matrix_first_tie",
        linear::invert_matrix_first_tie(&a),
    );
    assert_option_finite(
        "linear::invert_matrix_last_tie",
        linear::invert_matrix_last_tie(&a),
    );
    assert_option_finite(
        "linear::solve_matrix_last_tie",
        linear::solve_matrix_last_tie(&a, &bmat),
    );
    assert_option_finite("linear::matrix_sub", linear::matrix_sub(&rect_a, &rect_b));
    assert_option_finite("linear::matmul", linear::matmul(&a, &bmat));
    assert_option_finite("linear::transpose", linear::transpose(&rect_a));

    let normal_rows_owned: Vec<(Vec<f64>, f64, f64)> = (0..rows)
        .map(|idx| {
            let row = (0..n)
                .map(|j| input.values.get(idx * n + j).copied().unwrap_or(0.0))
                .collect::<Vec<_>>();
            let y = input.rhs.get(idx).copied().unwrap_or(0.0);
            let w = input.weights.get(idx).copied().unwrap_or(1.0);
            (row, y, w)
        })
        .collect();
    assert_option_finite(
        "linear::normal_equations_weighted",
        linear::normal_equations_weighted(
            normal_rows_owned
                .iter()
                .map(|(row, y, w)| (row.as_slice(), *y, *w)),
            n,
        ),
    );

    let flat = flat_square(&input.values, n);
    let rhs_flat: Vec<f64> = (0..(n * cols))
        .map(|idx| input.rhs.get(idx).copied().unwrap_or(0.0))
        .collect();
    let mut out = Vec::new();
    let mut scratch = linear::FlatLinearScratch::default();
    if linear::invert_flat_first_tie_into(&flat, n, &mut out, &mut scratch).is_some() {
        assert_success("linear::invert_flat_first_tie_into", out.clone());
    }
    if linear::solve_matrix_flat_first_tie_into(&flat, n, &rhs_flat, cols, &mut out, &mut scratch)
        .is_some()
    {
        assert_success("linear::solve_matrix_flat_first_tie_into", out.clone());
    }

    assert_option_finite(
        "linear::solve_flat_normal_first_tie",
        linear::solve_flat_normal_first_tie(&flat, &bvec),
    );
    let rows4: Vec<[f64; 4]> = cap_vec(input.rows4.clone(), MAX_VEC)
        .into_iter()
        .map(finite_clamped4)
        .collect();
    assert_success(
        "linear::normal_matrix_4_unweighted_row_outer",
        linear::normal_matrix_4_unweighted_row_outer(&rows4),
    );
    let weights4: Vec<f64> = (0..rows4.len())
        .map(|idx| finite_clamped(input.weights.get(idx).copied().unwrap_or(1.0)))
        .collect();
    assert_ok_finite_or_err(
        "linear::normal_matrix_4_weighted_column_outer",
        linear::normal_matrix_4_weighted_column_outer(&rows4, &weights4),
    );
    let m4 = finite_clamped_mat4(input.m4);
    assert_success(
        "linear::mat4_vec4",
        linear::mat4_vec4(&m4, &[finite_clamped(input.scalars[0]); 4]),
    );
    assert_success(
        "linear::dot4",
        linear::dot4(&m4[0], &[finite_clamped(input.scalars[1]); 4]),
    );
    assert_success("linear::det4_cofactor", linear::det4_cofactor(&m4));
    assert_success(
        "linear::minor3_of_4",
        linear::minor3_of_4(
            &m4,
            usize::from(input.dims[2] % 4),
            usize::from(input.dims[3] % 4),
        ),
    );
    assert_option_finite(
        "linear::invert_4x4_cofactor",
        linear::invert_4x4_cofactor(&m4),
    );
    assert_option_finite(
        "linear::invert_3x3_adjugate",
        linear::invert_3x3_adjugate(&m3a),
    );
    assert_option_finite(
        "linear::invert_symmetric_pd",
        linear::invert_symmetric_pd(&a),
    );

    let x0 = DVector::from_iterator(n, (0..n).map(|idx| bvec[idx]));
    assert_ok_finite_or_err("least_squares::fd_steps", fd_steps(&x0, input.scalars[2]));
    let f0 = DVector::from_iterator(rows, (0..rows).map(|idx| x0[idx % n] - input.scalars[3]));
    let residual = |x: &DVector<f64>| {
        DVector::from_iterator(
            rows,
            (0..rows).map(|idx| x[idx % n] * input.scalars[4] + input.scalars[5]),
        )
    };
    assert_ok_finite_or_err(
        "least_squares::jacobian_2point",
        jacobian_2point(residual, &x0, &f0),
    );
    assert_ok_finite_or_err("least_squares::cost", cost(&f0));
    let problem = LeastSquaresProblem::new(residual, x0.clone());
    let opts = SolveOptions {
        gtol: input.scalars[6],
        ftol: input.scalars[7],
        xtol: input.scalars[8],
        max_nfev: bounded_usize(input.dims[3], 1, 8),
    };
    assert_ok_finite_or_err("least_squares::solve_trf", solve_trf(&problem, &opts));
    assert_ok_finite_or_err(
        "least_squares::solve_trf_with",
        solve_trf_with(&problem, &opts, TrustRegionSolve::OwnedGaussianFirstTie),
    );

    let residuals = cap_vec(input.values, MAX_VEC);
    assert_ok_finite_or_err("robust::median", robust::median(&residuals));
    assert_ok_finite_or_err(
        "robust::mad_scale",
        robust::mad_scale(&residuals, input.scalars[9]),
    );
    assert_success(
        "robust::huber_weight",
        robust::huber_weight(
            finite_clamped(input.scalars[10]),
            finite_clamped(input.scalars[11])
                .abs()
                .max(f64::MIN_POSITIVE),
        ),
    );
});
