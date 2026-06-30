//! Deterministic small linear-algebra kernels.
//!
//! These routines keep scalar operation order explicit for parity-sensitive
//! GNSS callers. When pivot tie-breaking or accumulation order matters, the
//! variant name states the policy instead of hiding it in a local copy.

use crate::astro::tolerances::PIVOT_EPSILON;
use crate::validate;

#[derive(Debug, Default, Clone)]
pub struct FlatLinearScratch {
    rows: Vec<f64>,
    x: Vec<f64>,
}

#[derive(Debug, Default, Clone)]
pub struct FlatNormalSolveScratch {
    a: Vec<f64>,
    b: Vec<f64>,
    x: Vec<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LinearError {
    #[error("invalid linear algebra {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

#[allow(clippy::needless_range_loop)]
pub fn solve_linear_first_tie(a: &[Vec<f64>], b: &[f64]) -> Option<Vec<f64>> {
    let n = validate_dense_system(a, b)?;
    let mut rows: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = rows[col][col].abs();
        for idx in (col + 1)..n {
            let v = rows[idx][col].abs();
            if v > pivot_abs {
                pivot_abs = v;
                pivot_row = idx;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= PIVOT_EPSILON {
            return None;
        }
        rows.swap(col, pivot_row);

        let pivot = rows[col].clone();
        let pivot_value = pivot[col];
        for idx in (col + 1)..n {
            let factor = rows[idx][col] / pivot_value;
            for j in 0..=n {
                rows[idx][j] -= factor * pivot[j];
            }
        }
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut known = 0.0;
        for j in (i + 1)..n {
            known += rows[i][j] * x[j];
        }
        x[i] = (rows[i][n] - known) / rows[i][i];
    }
    validate::finite_slice(&x, "solution").ok()?;
    Some(x)
}

#[allow(clippy::needless_range_loop)]
pub fn solve_linear_last_tie(mut a: Vec<Vec<f64>>, b: Vec<f64>) -> Option<Vec<f64>> {
    let n = validate_dense_system(&a, &b)?;
    for (row, bi) in a.iter_mut().zip(b) {
        row.push(bi);
    }
    for col in 0..n {
        let (pivot_row, pivot_abs) = (col..n)
            .map(|idx| (idx, a[idx][col].abs()))
            .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
            .unwrap();
        if !pivot_abs.is_finite() || pivot_abs <= PIVOT_EPSILON {
            return None;
        }
        a.swap(col, pivot_row);
        let pivot = a[col].clone();
        let pivot_value = pivot[col];
        for row in a.iter_mut().take(n).skip(col + 1) {
            let factor = row[col] / pivot_value;
            for j in col..=n {
                row[j] -= factor * pivot[j];
            }
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let tail_sum: f64 = ((i + 1)..n).map(|j| a[i][j] * x[j]).sum();
        x[i] = (a[i][n] - tail_sum) / a[i][i];
    }
    validate::finite_slice(&x, "solution").ok()?;
    Some(x)
}

pub fn invert_matrix_first_tie(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 {
        return None;
    }
    let mut columns: Vec<Vec<f64>> = Vec::with_capacity(n);
    for col in 0..n {
        let mut e = vec![0.0; n];
        e[col] = 1.0;
        columns.push(solve_linear_first_tie(a, &e)?);
    }
    Some(
        (0..n)
            .map(|i| (0..n).map(|j| columns[j][i]).collect())
            .collect(),
    )
}

pub fn invert_matrix_last_tie(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut columns = Vec::with_capacity(n);
    for col in 0..n {
        let unit = (0..n)
            .map(|idx| if idx == col { 1.0 } else { 0.0 })
            .collect();
        columns.push(solve_linear_last_tie(a.to_vec(), unit)?);
    }
    transpose(&columns)
}

pub fn solve_matrix_last_tie(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let columns = transpose(b)?;
    let mut solved_columns = Vec::with_capacity(columns.len());
    for col in columns {
        solved_columns.push(solve_linear_last_tie(a.to_vec(), col)?);
    }
    transpose(&solved_columns)
}

pub fn normal_equations_weighted<'a, I>(rows: I, n: usize) -> Option<(Vec<Vec<f64>>, Vec<f64>)>
where
    I: IntoIterator<Item = (&'a [f64], f64, f64)>,
{
    if n == 0 {
        return None;
    }
    let mut ata = vec![vec![0.0; n]; n];
    let mut aty = vec![0.0; n];
    for (row_h, row_y, row_weight) in rows {
        if row_h.len() != n {
            return None;
        }
        validate::finite_slice(row_h, "normal row").ok()?;
        validate::finite(row_y, "normal residual").ok()?;
        validate::finite(row_weight, "normal weight").ok()?;
        let h: Vec<f64> = row_h.iter().map(|v| v * row_weight).collect();
        let y = row_y * row_weight;
        for i in 0..n {
            aty[i] += h[i] * y;
            for j in 0..n {
                ata[i][j] += h[i] * h[j];
            }
        }
    }
    for row in &ata {
        validate::finite_slice(row, "normal matrix").ok()?;
    }
    validate::finite_slice(&aty, "normal rhs").ok()?;
    Some((ata, aty))
}

pub fn matrix_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let (rows, cols) = validate_same_shape(a, b)?;
    let out: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row_a, row_b)| row_a.iter().zip(row_b).map(|(x, y)| x - y).collect())
        .collect();
    debug_assert_eq!(out.len(), rows);
    debug_assert!(out.iter().all(|row| row.len() == cols));
    for row in &out {
        validate::finite_slice(row, "matrix difference").ok()?;
    }
    Some(out)
}

pub fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let b_t = transpose(b)?;
    let rows = a.len();
    let shared = b_t.first()?.len();
    if rows == 0 || shared == 0 {
        return None;
    }
    for row in a {
        if row.len() != shared {
            return None;
        }
        validate::finite_slice(row, "matrix").ok()?;
    }
    let out: Vec<Vec<f64>> = a
        .iter()
        .map(|row| {
            b_t.iter()
                .map(|col| row.iter().zip(col).fold(0.0, |acc, (x, y)| acc + x * y))
                .collect()
        })
        .collect();
    for row in &out {
        validate::finite_slice(row, "matrix product").ok()?;
    }
    Some(out)
}

pub fn transpose(matrix: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let cols = matrix.first()?.len();
    if cols == 0 {
        return None;
    }
    for row in matrix {
        if row.len() != cols {
            return None;
        }
        validate::finite_slice(row, "matrix").ok()?;
    }
    Some(
        (0..cols)
            .map(|col| matrix.iter().map(|row| row[col]).collect())
            .collect(),
    )
}

pub fn invert_flat_first_tie_into(
    a: &[f64],
    n: usize,
    out: &mut Vec<f64>,
    scratch: &mut FlatLinearScratch,
) -> Option<()> {
    validate_flat_square(a, n, "matrix")?;
    out.resize(n * n, 0.0);
    scratch.rows.resize(n * (n + 1), 0.0);
    scratch.x.resize(n, 0.0);

    for col in 0..n {
        for i in 0..n {
            let src = i * n;
            let dst = i * (n + 1);
            scratch.rows[dst..(dst + n)].copy_from_slice(&a[src..(src + n)]);
            scratch.rows[dst + n] = if i == col { 1.0 } else { 0.0 };
        }
        solve_augmented_flat_first_tie_in_place(&mut scratch.rows, n, &mut scratch.x)?;
        for i in 0..n {
            out[i * n + col] = scratch.x[i];
        }
    }

    Some(())
}

pub fn solve_matrix_flat_first_tie_into(
    a: &[f64],
    n: usize,
    b: &[f64],
    cols: usize,
    out: &mut Vec<f64>,
    scratch: &mut FlatLinearScratch,
) -> Option<()> {
    validate_flat_square(a, n, "matrix")?;
    if cols == 0 || b.len() != n.checked_mul(cols)? {
        return None;
    }
    validate::finite_slice(b, "rhs").ok()?;
    out.resize(n.checked_mul(cols)?, 0.0);
    scratch.rows.resize(n * (n + 1), 0.0);
    scratch.x.resize(n, 0.0);

    for col in 0..cols {
        for i in 0..n {
            let src = i * n;
            let dst = i * (n + 1);
            scratch.rows[dst..(dst + n)].copy_from_slice(&a[src..(src + n)]);
            scratch.rows[dst + n] = b[i * cols + col];
        }
        solve_augmented_flat_first_tie_in_place(&mut scratch.rows, n, &mut scratch.x)?;
        for i in 0..n {
            out[i * cols + col] = scratch.x[i];
        }
    }
    Some(())
}

#[allow(clippy::needless_range_loop)]
pub fn solve_augmented_flat_first_tie_in_place(
    rows: &mut [f64],
    n: usize,
    x: &mut [f64],
) -> Option<()> {
    let stride = n + 1;
    if n == 0 || rows.len() != n.checked_mul(stride)? || x.len() != n {
        return None;
    }
    validate::finite_slice(rows, "augmented matrix").ok()?;

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = rows[col * stride + col].abs();
        for idx in (col + 1)..n {
            let v = rows[idx * stride + col].abs();
            if v > pivot_abs {
                pivot_abs = v;
                pivot_row = idx;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= PIVOT_EPSILON {
            return None;
        }
        if pivot_row != col {
            for j in 0..=n {
                rows.swap(col * stride + j, pivot_row * stride + j);
            }
        }

        let pivot_value = rows[col * stride + col];
        for idx in (col + 1)..n {
            let factor = rows[idx * stride + col] / pivot_value;
            for j in 0..=n {
                rows[idx * stride + j] -= factor * rows[col * stride + j];
            }
        }
    }

    for i in (0..n).rev() {
        let mut known = 0.0;
        for j in (i + 1)..n {
            known += rows[i * stride + j] * x[j];
        }
        x[i] = (rows[i * stride + n] - known) / rows[i * stride + i];
    }

    validate::finite_slice(x, "solution").ok()?;
    Some(())
}

pub fn solve_flat_normal_first_tie(lambda: &[f64], eta: &[f64]) -> Option<Vec<f64>> {
    let mut scratch = FlatNormalSolveScratch::default();
    solve_flat_normal_first_tie_into(lambda, eta, &mut scratch).map(<[f64]>::to_vec)
}

#[allow(clippy::needless_range_loop)]
pub fn solve_flat_normal_first_tie_into<'a>(
    lambda: &[f64],
    eta: &[f64],
    scratch: &'a mut FlatNormalSolveScratch,
) -> Option<&'a [f64]> {
    let n = eta.len();
    if n == 0 || lambda.len() != n.checked_mul(n)? {
        return None;
    }
    validate::finite_slice(lambda, "normal matrix").ok()?;
    validate::finite_slice(eta, "normal rhs").ok()?;

    scratch.a.resize(n * n, 0.0);
    scratch.a.copy_from_slice(lambda);
    scratch.b.resize(n, 0.0);
    scratch.b.copy_from_slice(eta);

    for k in 0..n {
        let mut pivot = k;
        let mut pivot_abs = scratch.a[k * n + k].abs();
        for i in (k + 1)..n {
            let candidate = scratch.a[i * n + k].abs();
            if candidate > pivot_abs {
                pivot = i;
                pivot_abs = candidate;
            }
        }
        if !pivot_abs.is_finite() || pivot_abs <= PIVOT_EPSILON {
            return None;
        }
        if pivot != k {
            for j in 0..n {
                scratch.a.swap(k * n + j, pivot * n + j);
            }
            scratch.b.swap(k, pivot);
        }

        let diag = scratch.a[k * n + k];
        for i in (k + 1)..n {
            let factor = scratch.a[i * n + k] / diag;
            scratch.a[i * n + k] = 0.0;
            for j in (k + 1)..n {
                scratch.a[i * n + j] -= factor * scratch.a[k * n + j];
            }
            scratch.b[i] -= factor * scratch.b[k];
        }
    }

    scratch.x.resize(n, 0.0);
    for i in (0..n).rev() {
        let mut known = 0.0;
        for j in (i + 1)..n {
            known += scratch.a[i * n + j] * scratch.x[j];
        }
        scratch.x[i] = (scratch.b[i] - known) / scratch.a[i * n + i];
    }
    validate::finite_slice(&scratch.x, "solution").ok()?;
    Some(&scratch.x)
}

/// Reusable buffers for the owned Cholesky (square-root) solve
/// ([`solve_flat_normal_square_root_into`]): the lower-triangular factor `L`
/// (row-major `n x n`), the forward-substitution vector `z`, and the solution
/// `x`. Held across solves so a steady-state iteration does not allocate.
#[derive(Debug, Default, Clone)]
pub struct FlatCholeskySolveScratch {
    l: Vec<f64>,
    z: Vec<f64>,
    x: Vec<f64>,
}

/// Solve the symmetric positive-definite information system `Λ x = η` by an owned
/// deterministic Cholesky (square-root) factorization `Λ = L Lᵀ`, then forward
/// substitution `L z = η` and back substitution `Lᵀ x = z`. `lambda` is the
/// row-major `n x n` information matrix, `eta` the length-`n` information vector.
///
/// The Cholesky factor `L` is the information-matrix square root, so this is the
/// square-root-information solve. Unlike the general first-tie Gaussian
/// elimination ([`solve_flat_normal_first_tie_into`]) it needs no pivoting: the
/// system is SPD, so the fixed `i`/`j`/`k` reduction order (identical to
/// [`invert_symmetric_pd`]) is the entire op-order and the result is
/// bit-reproducible with no pivot-dependent branching. Returns `None` if `Λ` is
/// not positive definite (a non-positive or non-finite pivot), which for a
/// weighted least-squares normal matrix means rank-deficient geometry.
#[allow(clippy::needless_range_loop)]
pub fn solve_flat_normal_square_root_into<'a>(
    lambda: &[f64],
    eta: &[f64],
    scratch: &'a mut FlatCholeskySolveScratch,
) -> Option<&'a [f64]> {
    let n = eta.len();
    if n == 0 || lambda.len() != n.checked_mul(n)? {
        return None;
    }
    validate::finite_slice(lambda, "normal matrix").ok()?;
    validate::finite_slice(eta, "normal rhs").ok()?;
    validate_flat_symmetric(lambda, n)?;
    scratch.l.resize(n * n, 0.0);
    scratch.l.fill(0.0);

    // Cholesky Λ = L Lᵀ, the same factorization order as `invert_symmetric_pd`.
    for i in 0..n {
        for j in 0..=i {
            let mut s = lambda[i * n + j];
            for k in 0..j {
                s -= scratch.l[i * n + k] * scratch.l[j * n + k];
            }
            if i == j {
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                let nonpositive_or_nan = !(s > 0.0);
                if nonpositive_or_nan || !s.is_finite() {
                    return None;
                }
                scratch.l[i * n + j] = s.sqrt();
            } else {
                scratch.l[i * n + j] = s / scratch.l[j * n + j];
            }
        }
    }

    // Forward substitution L z = η.
    scratch.z.resize(n, 0.0);
    for i in 0..n {
        let mut s = eta[i];
        for k in 0..i {
            s -= scratch.l[i * n + k] * scratch.z[k];
        }
        scratch.z[i] = s / scratch.l[i * n + i];
    }
    validate::finite_slice(&scratch.z, "solution work vector").ok()?;

    // Back substitution Lᵀ x = z.
    scratch.x.resize(n, 0.0);
    for i in (0..n).rev() {
        let mut s = scratch.z[i];
        for k in (i + 1)..n {
            s -= scratch.l[k * n + i] * scratch.x[k];
        }
        scratch.x[i] = s / scratch.l[i * n + i];
    }
    validate::finite_slice(&scratch.x, "solution").ok()?;
    Some(scratch.x.as_slice())
}

fn validate_flat_symmetric(matrix: &[f64], n: usize) -> Option<()> {
    let mut scale = 1.0_f64;
    for value in matrix {
        scale = scale.max(value.abs());
    }
    let tol = symmetry_tolerance(n, scale);
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[i * n + j] - matrix[j * n + i]).abs() > tol {
                return None;
            }
        }
    }
    Some(())
}

#[allow(clippy::needless_range_loop)]
fn validate_rows_symmetric(matrix: &[Vec<f64>]) -> Option<()> {
    let n = matrix.len();
    let mut scale = 1.0_f64;
    for row in matrix {
        for value in row {
            scale = scale.max(value.abs());
        }
    }
    let tol = symmetry_tolerance(n, scale);
    for i in 0..n {
        for j in (i + 1)..n {
            if (matrix[i][j] - matrix[j][i]).abs() > tol {
                return None;
            }
        }
    }
    Some(())
}

fn symmetry_tolerance(n: usize, scale: f64) -> f64 {
    128.0 * f64::EPSILON * (n.max(1) as f64) * scale.max(1.0)
}

fn validate_dense_system(a: &[Vec<f64>], b: &[f64]) -> Option<usize> {
    let n = b.len();
    if n == 0 || a.len() != n {
        return None;
    }
    validate::finite_slice(b, "rhs").ok()?;
    for row in a {
        if row.len() != n {
            return None;
        }
        validate::finite_slice(row, "matrix").ok()?;
    }
    Some(n)
}

fn validate_same_shape(a: &[Vec<f64>], b: &[Vec<f64>]) -> Option<(usize, usize)> {
    let rows = a.len();
    if rows == 0 || b.len() != rows {
        return None;
    }
    let cols = a.first()?.len();
    if cols == 0 {
        return None;
    }
    for row in a {
        if row.len() != cols {
            return None;
        }
        validate::finite_slice(row, "matrix").ok()?;
    }
    for row in b {
        if row.len() != cols {
            return None;
        }
        validate::finite_slice(row, "matrix").ok()?;
    }
    Some((rows, cols))
}

fn validate_flat_square(a: &[f64], n: usize, field: &'static str) -> Option<()> {
    if n == 0 || a.len() != n.checked_mul(n)? {
        return None;
    }
    validate::finite_slice(a, field).ok()
}

fn map_linear_field_error(error: validate::FieldError) -> LinearError {
    linear_invalid_input(error.field(), error.reason())
}

fn linear_invalid_input(field: &'static str, reason: &'static str) -> LinearError {
    LinearError::InvalidInput { field, reason }
}

#[allow(clippy::needless_range_loop)]
pub fn normal_matrix_4_weighted_column_outer(
    rows: &[[f64; 4]],
    weights: &[f64],
) -> Result<[[f64; 4]; 4], LinearError> {
    if weights.len() != rows.len() {
        return Err(linear_invalid_input("weights", "length must match rows"));
    }
    validate::finite_slice(weights, "weights").map_err(map_linear_field_error)?;
    for row in rows {
        validate::finite_slice(row, "rows").map_err(map_linear_field_error)?;
    }

    let mut a = [[0.0_f64; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0_f64;
            for k in 0..rows.len() {
                s += rows[k][i] * weights[k] * rows[k][j];
            }
            a[i][j] = s;
        }
    }
    for row in &a {
        validate::finite_slice(row, "normal matrix").map_err(map_linear_field_error)?;
    }
    Ok(a)
}

#[allow(clippy::needless_range_loop)]
pub fn normal_matrix_4_unweighted_row_outer(rows: &[[f64; 4]]) -> [[f64; 4]; 4] {
    let mut a = [[0.0_f64; 4]; 4];
    for row in rows {
        for i in 0..4 {
            for j in 0..4 {
                a[i][j] += row[i] * row[j];
            }
        }
    }
    a
}

pub fn mat4_vec4(m: &[[f64; 4]; 4], v: &[f64; 4]) -> [f64; 4] {
    [
        dot4(&m[0], v),
        dot4(&m[1], v),
        dot4(&m[2], v),
        dot4(&m[3], v),
    ]
}

pub fn dot4(row: &[f64; 4], v: &[f64; 4]) -> f64 {
    row[0] * v[0] + row[1] * v[1] + row[2] * v[2] + row[3] * v[3]
}

pub fn det4_cofactor(a: &[[f64; 4]; 4]) -> f64 {
    let m01 = a[2][0] * a[3][1] - a[2][1] * a[3][0];
    let m02 = a[2][0] * a[3][2] - a[2][2] * a[3][0];
    let m03 = a[2][0] * a[3][3] - a[2][3] * a[3][0];
    let m12 = a[2][1] * a[3][2] - a[2][2] * a[3][1];
    let m13 = a[2][1] * a[3][3] - a[2][3] * a[3][1];
    let m23 = a[2][2] * a[3][3] - a[2][3] * a[3][2];

    let c0 = a[1][1] * m23 - a[1][2] * m13 + a[1][3] * m12;
    let c1 = a[1][0] * m23 - a[1][2] * m03 + a[1][3] * m02;
    let c2 = a[1][0] * m13 - a[1][1] * m03 + a[1][3] * m01;
    let c3 = a[1][0] * m12 - a[1][1] * m02 + a[1][2] * m01;

    a[0][0] * c0 - a[0][1] * c1 + a[0][2] * c2 - a[0][3] * c3
}

pub fn minor3_of_4(a: &[[f64; 4]; 4], skip_r: usize, skip_c: usize) -> f64 {
    let mut rows = [0_usize; 3];
    let mut cols = [0_usize; 3];
    let mut row_idx = 0;
    let mut col_idx = 0;
    for row in 0..4 {
        if row != skip_r {
            rows[row_idx] = row;
            row_idx += 1;
        }
    }
    for col in 0..4 {
        if col != skip_c {
            cols[col_idx] = col;
            col_idx += 1;
        }
    }

    let b00 = a[rows[0]][cols[0]];
    let b01 = a[rows[0]][cols[1]];
    let b02 = a[rows[0]][cols[2]];
    let b10 = a[rows[1]][cols[0]];
    let b11 = a[rows[1]][cols[1]];
    let b12 = a[rows[1]][cols[2]];
    let b20 = a[rows[2]][cols[0]];
    let b21 = a[rows[2]][cols[1]];
    let b22 = a[rows[2]][cols[2]];

    b00 * (b11 * b22 - b12 * b21) - b01 * (b10 * b22 - b12 * b20) + b02 * (b10 * b21 - b11 * b20)
}

#[allow(clippy::needless_range_loop)]
pub fn invert_4x4_cofactor(a: &[[f64; 4]; 4]) -> Option<[[f64; 4]; 4]> {
    let det = det4_cofactor(a);
    if det == 0.0 || !det.is_finite() {
        return None;
    }

    let mut inv = [[0.0_f64; 4]; 4];
    for j in 0..4 {
        for i in 0..4 {
            let sign = if (i + j) % 2 == 0 { 1.0 } else { -1.0 };
            inv[j][i] = sign * minor3_of_4(a, i, j) / det;
        }
    }
    if inv.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    Some(inv)
}

pub fn invert_3x3_adjugate(m: &[[f64; 3]; 3]) -> Option<[[f64; 3]; 3]> {
    let [[a, b, c], [d, e, f], [g, h, i]] = *m;
    let det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if det.abs() <= PIVOT_EPSILON || !det.is_finite() {
        return None;
    }
    let inv_det = 1.0 / det;
    let inverse = [
        [
            (e * i - f * h) * inv_det,
            (c * h - b * i) * inv_det,
            (b * f - c * e) * inv_det,
        ],
        [
            (f * g - d * i) * inv_det,
            (a * i - c * g) * inv_det,
            (c * d - a * f) * inv_det,
        ],
        [
            (d * h - e * g) * inv_det,
            (b * g - a * h) * inv_det,
            (a * e - b * d) * inv_det,
        ],
    ];
    if inverse.iter().flatten().any(|value| !value.is_finite()) {
        return None;
    }
    Some(inverse)
}

#[allow(clippy::needless_range_loop)]
pub fn invert_symmetric_pd(n: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let p = n.len();
    if p == 0 {
        return None;
    }
    for row in n {
        if row.len() != p {
            return None;
        }
        validate::finite_slice(row, "matrix").ok()?;
    }
    validate_rows_symmetric(n)?;
    let mut l = vec![vec![0.0_f64; p]; p];
    for i in 0..p {
        for j in 0..=i {
            let mut s = n[i][j];
            for k in 0..j {
                s -= l[i][k] * l[j][k];
            }
            if i == j {
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                let nonpositive_or_nan = !(s > 0.0);
                if nonpositive_or_nan || !s.is_finite() {
                    return None;
                }
                l[i][j] = s.sqrt();
            } else {
                l[i][j] = s / l[j][j];
            }
        }
    }

    let mut li = vec![vec![0.0_f64; p]; p];
    for i in 0..p {
        li[i][i] = 1.0 / l[i][i];
        for j in 0..i {
            let mut s = 0.0_f64;
            for k in j..i {
                s -= l[i][k] * li[k][j];
            }
            li[i][j] = s / l[i][i];
        }
    }

    let mut inv = vec![vec![0.0_f64; p]; p];
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0_f64;
            for k in 0..p {
                s += li[k][i] * li[k][j];
            }
            inv[i][j] = s;
        }
    }
    for row in &inv {
        validate::finite_slice(row, "inverse").ok()?;
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_tie_solver_inverts_known_matrix() {
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert_matrix_first_tie(&a).unwrap();
        assert_eq!(inv[0][0].to_bits(), 0.6000000000000001f64.to_bits());
        assert_eq!(inv[0][1].to_bits(), (-0.7000000000000001f64).to_bits());
        assert_eq!(inv[1][0].to_bits(), (-0.2f64).to_bits());
        assert_eq!(inv[1][1].to_bits(), 0.4f64.to_bits());
    }

    #[test]
    fn dense_solvers_reject_nonfinite_and_bad_shapes() {
        let good_rhs = [1.0, 2.0];
        let ragged = vec![vec![1.0], vec![0.0, 1.0]];
        assert!(solve_linear_first_tie(&ragged, &good_rhs).is_none());
        assert!(solve_linear_last_tie(ragged, good_rhs.to_vec()).is_none());

        let nonfinite_matrix = vec![vec![1.0, f64::NAN], vec![0.0, 1.0]];
        assert!(solve_linear_first_tie(&nonfinite_matrix, &good_rhs).is_none());
        assert!(solve_linear_last_tie(nonfinite_matrix, good_rhs.to_vec()).is_none());

        let good_matrix = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(solve_linear_first_tie(&good_matrix, &[1.0, f64::INFINITY]).is_none());
        assert!(solve_linear_first_tie(&[], &[]).is_none());
        assert!(invert_matrix_first_tie(&[]).is_none());
    }

    #[test]
    fn weighted_column_outer_rejects_short_weights() {
        let rows = [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -1.0, 1.0]];
        assert_eq!(
            normal_matrix_4_weighted_column_outer(&rows, &[0.5]),
            Err(LinearError::InvalidInput {
                field: "weights",
                reason: "length must match rows"
            })
        );
    }

    #[test]
    fn weighted_column_outer_accumulates_valid_inputs() {
        let rows = [[1.0, 2.0, 3.0, 4.0], [2.0, 0.0, -1.0, 1.0]];
        let weights = [0.5, 2.0];
        assert_eq!(
            normal_matrix_4_weighted_column_outer(&rows, &weights).unwrap(),
            [
                [8.5, 1.0, -2.5, 6.0],
                [1.0, 2.0, 3.0, 4.0],
                [-2.5, 3.0, 6.5, 4.0],
                [6.0, 4.0, 4.0, 10.0],
            ]
        );
    }

    #[test]
    fn transpose_rejects_empty_ragged_and_nonfinite_matrices() {
        assert!(transpose(&[]).is_none());
        assert!(transpose(&[vec![1.0], vec![]]).is_none());
        assert!(transpose(&[vec![f64::INFINITY]]).is_none());
    }

    #[test]
    fn normal_equations_reject_malformed_or_nonfinite_rows() {
        let short = [1.0];
        assert!(normal_equations_weighted([(short.as_slice(), 1.0, 1.0)], 2).is_none());

        let nonfinite_row = [1.0, f64::NAN];
        assert!(normal_equations_weighted([(nonfinite_row.as_slice(), 1.0, 1.0)], 2).is_none());

        let good_row = [1.0, 2.0];
        assert!(normal_equations_weighted([(good_row.as_slice(), f64::NAN, 1.0)], 2).is_none());
        assert!(
            normal_equations_weighted([(good_row.as_slice(), 1.0, f64::INFINITY)], 2).is_none()
        );
    }

    #[test]
    fn flat_solvers_reject_nonfinite_inputs() {
        let mut out = Vec::new();
        let mut scratch = FlatLinearScratch::default();
        assert!(invert_flat_first_tie_into(&[f64::NAN], 1, &mut out, &mut scratch).is_none());

        assert!(solve_flat_normal_first_tie(&[f64::NAN], &[1.0]).is_none());
        assert!(solve_flat_normal_first_tie(&[1.0], &[f64::INFINITY]).is_none());

        let mut cholesky = FlatCholeskySolveScratch::default();
        assert!(solve_flat_normal_square_root_into(&[1.0], &[f64::NAN], &mut cholesky).is_none());
    }

    #[test]
    fn flat_normal_solver_reports_singular() {
        assert!(solve_flat_normal_first_tie(&[1.0, 2.0, 2.0, 4.0], &[1.0, 2.0]).is_none());
    }

    #[test]
    fn cofactor_inverse_rejects_singular_4x4() {
        let a = [[0.0; 4]; 4];
        assert!(invert_4x4_cofactor(&a).is_none());
    }

    #[test]
    fn cholesky_square_root_solves_spd_system() {
        // Λ = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]] (the classic SPD
        // Cholesky example), η chosen so the exact solution is [1, 2, 3].
        let lambda = [
            4.0, 12.0, -16.0, //
            12.0, 37.0, -43.0, //
            -16.0, -43.0, 98.0,
        ];
        let eta = [
            4.0 * 1.0 + 12.0 * 2.0 - 16.0 * 3.0,
            12.0 * 1.0 + 37.0 * 2.0 - 43.0 * 3.0,
            -16.0 * 1.0 - 43.0 * 2.0 + 98.0 * 3.0,
        ];
        let mut scratch = FlatCholeskySolveScratch::default();
        let x = solve_flat_normal_square_root_into(&lambda, &eta, &mut scratch).unwrap();
        for (got, want) in x.iter().zip([1.0_f64, 2.0, 3.0]) {
            assert!((got - want).abs() < 1.0e-12, "got {got}, want {want}");
        }
    }

    #[test]
    fn cholesky_square_root_agrees_with_first_tie_to_roundoff() {
        // The square-root solve and the first-tie Gaussian solve of the same SPD
        // system must agree to roundoff: they differ only in factorization order.
        let lambda = [
            6.0, 2.0, 1.0, //
            2.0, 5.0, 2.0, //
            1.0, 2.0, 4.0,
        ];
        let eta = [9.0, 9.0, 7.0];
        let mut sqrt_scratch = FlatCholeskySolveScratch::default();
        let sqrt_x = solve_flat_normal_square_root_into(&lambda, &eta, &mut sqrt_scratch)
            .unwrap()
            .to_vec();
        let first_tie_x = solve_flat_normal_first_tie(&lambda, &eta).unwrap();
        for (s, f) in sqrt_x.iter().zip(&first_tie_x) {
            assert!((s - f).abs() < 1.0e-12, "square-root {s} vs first-tie {f}");
        }
    }

    #[test]
    fn cholesky_square_root_frozen_bits() {
        // Frozen-bits golden on an exactly-representable SPD system
        // (Λ = L Lᵀ with L = [[2,0,0],[1,2,0],[0,0,1]]), so every factor and
        // substitution step is exact in f64 and the solution bits are a portable
        // constant: f64 sqrt is IEEE-754 correctly rounded, so these bits hold
        // across platforms, not merely run-to-run on one build.
        let lambda = [
            4.0, 2.0, 0.0, //
            2.0, 5.0, 0.0, //
            0.0, 0.0, 1.0,
        ];
        // η = Λ·[2, 0.5, 3].
        let eta = [9.0, 6.5, 3.0];
        let mut scratch = FlatCholeskySolveScratch::default();
        let x = solve_flat_normal_square_root_into(&lambda, &eta, &mut scratch).unwrap();
        assert_eq!(x[0].to_bits(), 2.0f64.to_bits());
        assert_eq!(x[1].to_bits(), 0.5f64.to_bits());
        assert_eq!(x[2].to_bits(), 3.0f64.to_bits());
    }

    #[test]
    fn cholesky_square_root_rejects_non_pd() {
        // A singular (rank-deficient) matrix has a non-positive Cholesky pivot.
        assert!(solve_flat_normal_square_root_into(
            &[1.0, 2.0, 2.0, 4.0],
            &[1.0, 2.0],
            &mut Default::default()
        )
        .is_none());
    }

    #[test]
    fn cholesky_square_root_rejects_invalid_information_geometry() {
        let eta = [1.0, 2.0];
        let mut scratch = FlatCholeskySolveScratch::default();

        let negative_variance = [-1.0, 0.0, 0.0, 1.0];
        assert!(
            solve_flat_normal_square_root_into(&negative_variance, &eta, &mut scratch).is_none()
        );

        let asymmetric = [1.0, 0.5, 0.0, 1.0];
        assert!(solve_flat_normal_square_root_into(&asymmetric, &eta, &mut scratch).is_none());

        let indefinite = [1.0, 2.0, 2.0, 1.0];
        assert!(solve_flat_normal_square_root_into(&indefinite, &eta, &mut scratch).is_none());
    }

    #[test]
    fn symmetric_pd_inverse_rejects_invalid_matrix_geometry() {
        let negative_variance = vec![vec![-1.0, 0.0], vec![0.0, 1.0]];
        assert!(invert_symmetric_pd(&negative_variance).is_none());

        let asymmetric = vec![vec![1.0, 0.5], vec![0.0, 1.0]];
        assert!(invert_symmetric_pd(&asymmetric).is_none());

        let indefinite = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert!(invert_symmetric_pd(&indefinite).is_none());

        let overflow_inverse = vec![vec![f64::from_bits(1)]];
        assert!(invert_symmetric_pd(&overflow_inverse).is_none());
    }
}
