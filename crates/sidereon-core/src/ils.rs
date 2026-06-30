//! Integer least squares - ambiguity-resolution kernels for precise / RTK
//! positioning.
//!
//! The bounded kernel preserves the historical public score/order contract:
//! Gaussian elimination with partial pivoting (max-abs pivot, first-index tie
//! break, `<= PIVOT_EPSILON` singular guard), `Δᵀ Q⁻¹ Δ` summation order (i-outer,
//! j-inner, left-associated products), lattice enumeration, and `{score, cycles}`
//! candidate ordering. The LAMBDA kernel is a faithful RTKLIB `lambda()` port.
//! Crate-side tests pin the RTKLIB oracle fixture plus frozen output bits. All
//! arithmetic is plain `*` / `-` / `+` (no FMA), per the crate's reproducibility
//! rule.
//!
//! These kernels live in Rust because the bounded search, LAMBDA, and the
//! partial-ambiguity subset search built on top of them are compute hot paths for
//! multi-epoch RTK arcs.

use crate::astro::math::linear::{invert_matrix_first_tie, solve_linear_first_tie};

use crate::tolerances::LAMBDA_REDUCTION_EPS;
use crate::validate::{self, FieldError};
use crate::{Error, Result};

const ILS_RATIO_THRESHOLD_FIELD: &str = "ils ratio_threshold";

/// Why a bounded ILS search could not produce a result. Mapped by the NIF onto
/// the reference Elixir error tuples (`:singular_geometry`,
/// `{:no_integer_candidates, n}`, `{:too_many_integer_candidates, n, limit}`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IlsError {
    /// The covariance matrix is singular (degenerate geometry).
    Singular,
    /// The lattice yielded no candidate (an empty search box).
    NoCandidates(usize),
    /// The lattice exceeded `candidate_limit`.
    TooManyCandidates { evaluated: usize, limit: usize },
    /// `float_cycles` was empty, or `covariance` was not exactly `n x n` for
    /// `n = float_cycles.len()` (`rows` is the offending row count, or the length
    /// of the first row that was not `n` wide).
    InvalidDimensions { n: usize, rows: usize },
    /// A `float_cycles` or `covariance` entry was NaN or infinite.
    NonFinite,
    /// A public ILS option was malformed.
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    /// The MLAMBDA search did not converge within `LAMBDA_LOOP_MAX` iterations
    /// (distinct from a singular/degenerate covariance).
    SearchLimitExceeded,
}

impl core::fmt::Display for IlsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Singular => write!(f, "integer least-squares covariance is singular"),
            Self::NoCandidates(evaluated) => write!(
                f,
                "integer least-squares search found no candidates after {evaluated} evaluations"
            ),
            Self::TooManyCandidates { evaluated, limit } => write!(
                f,
                "integer least-squares search evaluated {evaluated} candidates, exceeding limit {limit}"
            ),
            Self::InvalidDimensions { n, rows } => write!(
                f,
                "integer least-squares input dimensions are invalid: {n} ambiguities, {rows} covariance rows"
            ),
            Self::NonFinite => write!(f, "integer least-squares inputs contain NaN or infinity"),
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid integer least-squares input {field}: {reason}")
            }
            Self::SearchLimitExceeded => {
                write!(f, "integer least-squares search did not converge")
            }
        }
    }
}

impl std::error::Error for IlsError {}

/// Validate inputs before any indexing or arithmetic: `covariance` must be a
/// square `n x n` matrix matching the number of float ambiguities (`n >= 1`),
/// and every value must be finite. Without the shape check an undersized
/// covariance indexes out of bounds (panic) and an oversized one is silently
/// truncated to a wrong-dimension submatrix; without the finite check NaN/Inf
/// propagate into a garbage "fix".
fn validate_inputs(
    float_cycles: &[f64],
    covariance: &[Vec<f64>],
) -> core::result::Result<(), IlsError> {
    let n = float_cycles.len();
    if n == 0 {
        return Err(IlsError::InvalidDimensions { n, rows: 0 });
    }
    if covariance.len() != n {
        return Err(IlsError::InvalidDimensions {
            n,
            rows: covariance.len(),
        });
    }
    for row in covariance {
        if row.len() != n {
            return Err(IlsError::InvalidDimensions { n, rows: row.len() });
        }
    }
    if float_cycles.iter().any(|v| !v.is_finite())
        || covariance.iter().flatten().any(|v| !v.is_finite())
    {
        return Err(IlsError::NonFinite);
    }
    Ok(())
}

fn validate_ratio_threshold(ratio_threshold: f64) -> core::result::Result<(), IlsError> {
    validate::finite_nonneg(ratio_threshold, ILS_RATIO_THRESHOLD_FIELD)
        .map(|_| ())
        .map_err(invalid_input)
}

fn validate_covariance_geometry(covariance: &[Vec<f64>]) -> core::result::Result<(), IlsError> {
    let rows: Vec<&[f64]> = covariance.iter().map(Vec::as_slice).collect();
    validate::validate_covariance_psd_rows(&rows, "ils covariance").map_err(invalid_input)
}

fn invalid_input(error: FieldError) -> IlsError {
    IlsError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

/// Outcome of a bounded ILS search.
#[derive(Debug, Clone, PartialEq)]
pub struct IlsResult {
    /// Best integer vector, parallel to the input `float_cycles`.
    pub fixed: Vec<i64>,
    /// Whether the ratio test passes at the requested threshold.
    pub fixed_status: bool,
    /// Runner-up / best score ratio. Saturates to `f64::MAX` when the best score
    /// is exactly zero with a positive runner-up; `0.0` when there is no runner-up.
    pub ratio: f64,
    /// Best (lowest) quadratic score `Δᵀ Q⁻¹ Δ`.
    pub best_score: f64,
    /// Runner-up score, if a second lattice point exists.
    pub second_best_score: Option<f64>,
    /// Count of candidates considered. Its meaning depends on the search:
    /// [`bounded_ils_search`] reports the number of lattice points actually
    /// evaluated inside the box; the LAMBDA search ([`lambda_ils_search`]) does
    /// not enumerate a box, so it reports the number of candidate vectors produced
    /// (the requested `ncands`, typically 2), not a lattice-point count.
    pub candidates_evaluated: usize,
    /// Symmetrized covariance actually used.
    pub covariance: Vec<Vec<f64>>,
    /// Symmetrized inverse covariance.
    pub covariance_inverse: Vec<Vec<f64>>,
}

/// Bounded integer least squares over the lattice within `radius` integers of
/// each rounded float ambiguity.
///
/// Returns the best integer vector and its ratio-test verdict, or an error when
/// the covariance is singular or the lattice exceeds `candidate_limit`.
pub fn bounded_ils_search(
    float_cycles: &[f64],
    covariance: &[Vec<f64>],
    radius: i64,
    candidate_limit: usize,
    ratio_threshold: f64,
) -> core::result::Result<IlsResult, IlsError> {
    validate_inputs(float_cycles, covariance)?;
    validate_covariance_geometry(covariance)?;
    validate_ratio_threshold(ratio_threshold)?;
    let q = symmetrize(covariance);
    let q_inv = symmetrize(&invert(&q).map_err(|_| IlsError::Singular)?);
    ensure_candidate_limit(float_cycles.len(), radius, candidate_limit)?;

    // Per-ambiguity candidate integers, ordered by |value - float| then value
    // (matches `integers_near/3`; the final top-two is order-independent, but we
    // mirror the reference exactly).
    let ranges: Vec<Vec<i64>> = float_cycles
        .iter()
        .map(|&f| bounded_integer_candidates(f, radius))
        .collect::<core::result::Result<_, _>>()?;

    let mut top: Vec<(f64, Vec<i64>)> = Vec::with_capacity(2);
    let mut evaluated: usize = 0;
    let mut current: Vec<i64> = Vec::with_capacity(float_cycles.len());

    let ctx = LatticeEnum {
        ranges: &ranges,
        float_cycles,
        q_inv: &q_inv,
        limit: candidate_limit,
    };
    enumerate(&ctx, 0, &mut current, &mut evaluated, &mut top)?;

    let (best_score, fixed) = match top.first() {
        Some((s, c)) => (*s, c.clone()),
        None => return Err(IlsError::NoCandidates(evaluated)),
    };
    let second_best_score = top.get(1).map(|(s, _)| *s);
    let ratio = integer_ratio(best_score, second_best_score);

    Ok(IlsResult {
        fixed,
        fixed_status: ratio_pass(ratio, ratio_threshold),
        ratio,
        best_score,
        second_best_score,
        candidates_evaluated: evaluated,
        covariance: q,
        covariance_inverse: q_inv,
    })
}

// --- lattice enumeration -------------------------------------------------

/// Immutable inputs shared across the recursive lattice walk: the per-ambiguity
/// candidate ranges, the float cycles and inverse covariance scoring the leaves,
/// and the candidate-count cap. Bundled so the recursion threads one context
/// instead of repeating four positional arguments at every call.
struct LatticeEnum<'a> {
    ranges: &'a [Vec<i64>],
    float_cycles: &'a [f64],
    q_inv: &'a [Vec<f64>],
    limit: usize,
}

fn enumerate(
    ctx: &LatticeEnum,
    depth: usize,
    current: &mut Vec<i64>,
    evaluated: &mut usize,
    top: &mut Vec<(f64, Vec<i64>)>,
) -> core::result::Result<(), IlsError> {
    if depth == ctx.ranges.len() {
        *evaluated += 1;
        if *evaluated > ctx.limit {
            return Err(IlsError::TooManyCandidates {
                evaluated: *evaluated,
                limit: ctx.limit,
            });
        }
        let score = quadratic_score(ctx.float_cycles, current, ctx.q_inv);
        insert_top_two(top, score, current);
        return Ok(());
    }

    for &value in &ctx.ranges[depth] {
        current.push(value);
        enumerate(ctx, depth + 1, current, evaluated, top)?;
        current.pop();
    }
    Ok(())
}

/// Keep the two lowest `(score, cycles)` candidates - same ordering as the
/// reference `integer_top_two/1` (score ascending, then cycles lexicographic).
fn insert_top_two(top: &mut Vec<(f64, Vec<i64>)>, score: f64, cycles: &[i64]) {
    top.push((score, cycles.to_vec()));
    top.sort_by(|(sa, ca), (sb, cb)| {
        sa.partial_cmp(sb)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| ca.cmp(cb))
    });
    top.truncate(2);
}

fn quadratic_score(float_cycles: &[f64], fixed: &[i64], q_inv: &[Vec<f64>]) -> f64 {
    let n = float_cycles.len();
    // delta = float - fixed, matching `a - z`.
    let deltas: Vec<f64> = (0..n).map(|i| float_cycles[i] - fixed[i] as f64).collect();

    // i-outer, j-inner, acc + delta[i] * q_inv[i][j] * delta[j] (left-assoc).
    let mut acc = 0.0;
    for i in 0..n {
        for j in 0..n {
            acc += deltas[i] * q_inv[i][j] * deltas[j];
        }
    }
    acc
}

fn integers_near(center: f64, low: i64, high: i64) -> Vec<i64> {
    let mut values: Vec<i64> = (low..=high).collect();
    values.sort_by(|&a, &b| {
        let da = (a as f64 - center).abs();
        let db = (b as f64 - center).abs();
        da.partial_cmp(&db)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| a.cmp(&b))
    });
    values
}

fn bounded_integer_candidates(
    float_cycle: f64,
    radius: i64,
) -> core::result::Result<Vec<i64>, IlsError> {
    if radius < 0 {
        return Ok(Vec::new());
    }

    let rounded = float_cycle.round(); // Elixir round/1: half away from zero
    const I64_MAX_EXCLUSIVE: f64 = 9_223_372_036_854_775_808.0;
    if rounded < i64::MIN as f64 || rounded >= I64_MAX_EXCLUSIVE {
        return Err(IlsError::InvalidInput {
            field: "ils float_cycles",
            reason: "outside integer search range",
        });
    }

    let center_i64 = rounded as i64;
    let low = center_i64
        .checked_sub(radius)
        .ok_or(IlsError::InvalidInput {
            field: "ils float_cycles",
            reason: "outside integer search range",
        })?;
    let high = center_i64
        .checked_add(radius)
        .ok_or(IlsError::InvalidInput {
            field: "ils float_cycles",
            reason: "outside integer search range",
        })?;
    Ok(integers_near(float_cycle, low, high))
}

fn ensure_candidate_limit(
    dimensions: usize,
    radius: i64,
    limit: usize,
) -> core::result::Result<(), IlsError> {
    let per_dimension = if radius < 0 {
        0usize
    } else {
        let width = radius
            .checked_mul(2)
            .and_then(|width| width.checked_add(1))
            .ok_or(IlsError::TooManyCandidates {
                evaluated: usize::MAX,
                limit,
            })?;
        usize::try_from(width).map_err(|_| IlsError::TooManyCandidates {
            evaluated: usize::MAX,
            limit,
        })?
    };

    let mut candidates = 1usize;
    for _ in 0..dimensions {
        candidates = candidates
            .checked_mul(per_dimension)
            .ok_or(IlsError::TooManyCandidates {
                evaluated: usize::MAX,
                limit,
            })?;
        if candidates > limit {
            return Err(IlsError::TooManyCandidates {
                evaluated: candidates,
                limit,
            });
        }
    }

    Ok(())
}

fn integer_ratio(best_score: f64, second_best_score: Option<f64>) -> f64 {
    match second_best_score {
        None => 0.0,
        Some(second) => {
            if best_score == 0.0 && second > 0.0 {
                f64::MAX
            } else if best_score == 0.0 {
                0.0
            } else {
                second / best_score
            }
        }
    }
}

fn ratio_pass(ratio: f64, threshold: f64) -> bool {
    ratio >= threshold
}

// --- linear algebra (bit-identical to LinearAlgebra) ---------------------

fn symmetrize(m: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = m.len();
    (0..n)
        .map(|i| (0..n).map(|j| (m[i][j] + m[j][i]) / 2.0).collect())
        .collect()
}

/// Invert by solving `A x = eᵢ` for each unit column, exactly as the reference
/// `invert_matrix/1`. `pub(crate)` so the RTK-filter kernel can invert the
/// posterior information into covariance for the ambiguity search.
pub(crate) fn invert(a: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
    invert_matrix_first_tie(a).ok_or_else(|| Error::InvalidInput("singular matrix".into()))
}

// Index-based loops mirror the reference Gaussian elimination (pivot scan and
// row updates index `rows` by position); an iterator form would obscure it.
// `pub(crate)` so the RTK-filter kernel reuses the same Gaussian elimination
// (partial pivoting, PIVOT_EPSILON singular guard) for its normal-equations solve.
pub(crate) fn solve_linear(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    solve_linear_first_tie(a, b).ok_or_else(|| Error::InvalidInput("singular matrix".into()))
}

// =========================================================================
// LAMBDA / MLAMBDA integer least squares (Teunissen 1995; Chang-Yang-Zhou 2005)
// -------------------------------------------------------------------------
// A faithful port of RTKLIB's `lambda()` (BSD-2, _tools/RTKLIB/src/lambda.c):
// LtDL factorization + integer-Gauss/permutation decorrelation reduction +
// modified-LAMBDA depth-first search. Unlike `bounded_ils_search` (a naive
// ±radius box that only finds the true ILS optimum when it lies within the box),
// this is a *correct* ILS solver for any positive-definite covariance - it is
// gated against RTKLIB's own committed reference vectors (incl. the strongly-
// correlated utest2 the box search cannot reach). Validation target is RTKLIB,
// not bit-identity; the algorithm differs, so agreement is to round-off.
//
// Matrices follow RTKLIB's COLUMN-MAJOR convention verbatim - element (row i,
// col j) of an n×n matrix is `flat[i + j*n]` - so the port reads line-for-line
// against lambda.c.

const LAMBDA_LOOP_MAX: usize = 10000;

#[inline]
fn lam_round(x: f64) -> f64 {
    (x + 0.5).floor() // RTKLIB ROUND(x) = floor(x+0.5)
}

#[inline]
fn lam_sgn(x: f64) -> f64 {
    if x <= 0.0 {
        -1.0
    } else {
        1.0
    }
}

/// LtDL factorization `Q = Lᵀ·diag(D)·L` (column-major). Returns `None` if Q is
/// not positive-definite (a pivot `D[i] <= 0`).
fn lam_ld(n: usize, q: &[f64]) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut a = q.to_vec();
    let mut l = vec![0.0f64; n * n];
    let mut d = vec![0.0f64; n];
    for i in (0..n).rev() {
        d[i] = a[i + i * n];
        if d[i] <= 0.0 {
            return None;
        }
        let ai = d[i].sqrt();
        for j in 0..=i {
            l[i + j * n] = a[i + j * n] / ai;
        }
        for j in 0..i {
            for k in 0..=j {
                a[j + k * n] -= l[i + k * n] * l[i + j * n];
            }
        }
        for j in 0..=i {
            l[i + j * n] /= l[i + i * n];
        }
    }
    Some((l, d))
}

/// Integer Gauss transformation on column `j` using column `i`.
fn lam_gauss(n: usize, l: &mut [f64], z: &mut [f64], i: usize, j: usize) {
    let mu = lam_round(l[i + j * n]) as i64;
    if mu != 0 {
        let muf = mu as f64;
        for k in i..n {
            l[k + j * n] -= muf * l[k + i * n];
        }
        for k in 0..n {
            z[k + j * n] -= muf * z[k + i * n];
        }
    }
}

/// Permutation of adjacent ambiguities `j` and `j+1`.
fn lam_perm(n: usize, l: &mut [f64], d: &mut [f64], j: usize, del: f64, z: &mut [f64]) {
    let eta = d[j] / del;
    let lam = d[j + 1] * l[j + 1 + j * n] / del;
    d[j] = eta * d[j + 1];
    d[j + 1] = del;
    for k in 0..j {
        let a0 = l[j + k * n];
        let a1 = l[j + 1 + k * n];
        l[j + k * n] = -l[j + 1 + j * n] * a0 + a1;
        l[j + 1 + k * n] = eta * a0 + lam * a1;
    }
    l[j + 1 + j * n] = lam;
    for k in (j + 2)..n {
        l.swap(k + j * n, k + (j + 1) * n);
    }
    for k in 0..n {
        z.swap(k + j * n, k + (j + 1) * n);
    }
}

/// LAMBDA reduction: decorrelate via integer Gauss transformations + adjacent
/// permutations, accumulating the unimodular transform `Z`.
fn lam_reduction(n: usize, l: &mut [f64], d: &mut [f64], z: &mut [f64]) {
    let mut j: isize = n as isize - 2;
    let mut k: isize = n as isize - 2;
    while j >= 0 {
        let ju = j as usize;
        if j <= k {
            for i in (ju + 1)..n {
                lam_gauss(n, l, z, i, ju);
            }
        }
        let del = d[ju] + l[ju + 1 + ju * n] * l[ju + 1 + ju * n] * d[ju + 1];
        if del + LAMBDA_REDUCTION_EPS < d[ju + 1] {
            lam_perm(n, l, d, ju, del, z);
            k = j;
            j = n as isize - 2;
        } else {
            j -= 1;
        }
    }
}

/// Modified-LAMBDA (mlambda) search for the `m` best integer vectors in the
/// decorrelated space. Returns `(zn, s)` where `zn` is `n*m` column-major
/// candidates and `s[k]` is the squared residual of candidate `k` (sorted
/// ascending). `None` on search-loop overflow.
fn lam_search(
    n: usize,
    m: usize,
    l: &[f64],
    d: &[f64],
    zs: &[f64],
) -> Option<(Vec<f64>, Vec<f64>)> {
    let mut s = vec![0.0f64; m];
    let mut zn = vec![0.0f64; n * m];
    let mut smat = vec![0.0f64; n * n];
    let mut dist = vec![0.0f64; n];
    let mut zb = vec![0.0f64; n];
    let mut z = vec![0.0f64; n];
    let mut step = vec![0.0f64; n];

    let mut nn: usize = 0;
    let mut imax: usize = 0;
    let mut maxdist = 1.0e99;

    let mut k: isize = n as isize - 1;
    let ku = k as usize;
    dist[ku] = 0.0;
    zb[ku] = zs[ku];
    z[ku] = lam_round(zb[ku]);
    let mut y = zb[ku] - z[ku];
    step[ku] = lam_sgn(y);

    let mut c = 0usize;
    while c < LAMBDA_LOOP_MAX {
        let kk = k as usize;
        let newdist = dist[kk] + y * y / d[kk];
        if newdist < maxdist {
            if k != 0 {
                k -= 1;
                let kk = k as usize;
                dist[kk] = newdist;
                for i in 0..=kk {
                    smat[kk + i * n] =
                        smat[kk + 1 + i * n] + (z[kk + 1] - zb[kk + 1]) * l[kk + 1 + i * n];
                }
                zb[kk] = zs[kk] + smat[kk + kk * n];
                z[kk] = lam_round(zb[kk]);
                y = zb[kk] - z[kk];
                step[kk] = lam_sgn(y);
            } else {
                if nn < m {
                    if nn == 0 || newdist > s[imax] {
                        imax = nn;
                    }
                    for i in 0..n {
                        zn[i + nn * n] = z[i];
                    }
                    s[nn] = newdist;
                    nn += 1;
                } else {
                    if newdist < s[imax] {
                        for i in 0..n {
                            zn[i + imax * n] = z[i];
                        }
                        s[imax] = newdist;
                        imax = 0;
                        for i in 0..m {
                            if s[imax] < s[i] {
                                imax = i;
                            }
                        }
                    }
                    maxdist = s[imax];
                }
                z[0] += step[0];
                y = zb[0] - z[0];
                step[0] = -step[0] - lam_sgn(step[0]);
            }
        } else if k == n as isize - 1 {
            break;
        } else {
            k += 1;
            let kk = k as usize;
            z[kk] += step[kk];
            y = zb[kk] - z[kk];
            step[kk] = -step[kk] - lam_sgn(step[kk]);
        }
        c += 1;
    }

    if c >= LAMBDA_LOOP_MAX {
        return None;
    }

    // Sort the m candidates by ascending residual (RTKLIB's selection sort).
    for i in 0..m.saturating_sub(1) {
        for j in (i + 1)..m {
            if s[i] < s[j] {
                continue;
            }
            s.swap(i, j);
            for k in 0..n {
                zn.swap(k + i * n, k + j * n);
            }
        }
    }
    Some((zn, s))
}

/// Correct integer-least-squares via the LAMBDA method (RTKLIB `lambda()` port).
///
/// Finds the true ILS optimum and runner-up for any positive-definite
/// covariance - no search box, no combinatorial blow-up. Returns the same
/// [`IlsResult`] shape as [`bounded_ils_search`] so it is a drop-in: in the
/// weakly-correlated regime both select the identical integer vector and ratio;
/// on strongly-correlated geometry only this one is correct.
pub fn lambda_ils_search(
    float_cycles: &[f64],
    covariance: &[Vec<f64>],
    ratio_threshold: f64,
) -> core::result::Result<IlsResult, IlsError> {
    validate_inputs(float_cycles, covariance)?;
    validate_covariance_geometry(covariance)?;
    validate_ratio_threshold(ratio_threshold)?;
    let n = float_cycles.len();
    let q = symmetrize(covariance);
    // Inverse is kept only for the diagnostic metadata (LAMBDA itself uses LtDL).
    let q_inv = symmetrize(&invert(&q).map_err(|_| IlsError::Singular)?);

    // Column-major copy of the symmetrized covariance for the RTKLIB port.
    let mut q_cm = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            q_cm[i + j * n] = q[i][j];
        }
    }

    let (mut l, mut d) = lam_ld(n, &q_cm).ok_or(IlsError::Singular)?;
    let mut z = {
        // Z = identity (column-major).
        let mut e = vec![0.0f64; n * n];
        for i in 0..n {
            e[i + i * n] = 1.0;
        }
        e
    };
    lam_reduction(n, &mut l, &mut d, &mut z);

    // zs = Zᵀ·a.
    let mut zs = vec![0.0f64; n];
    for i in 0..n {
        let mut acc = 0.0;
        for k in 0..n {
            acc += z[k + i * n] * float_cycles[k];
        }
        zs[i] = acc;
    }

    // Best + runner-up, for the ratio test. `lam_ld` already failed
    // `Singular` above; a `None` here is search-loop overflow.
    let m = 2usize;
    let (zn, _s) = lam_search(n, m, &l, &d, &zs).ok_or(IlsError::SearchLimitExceeded)?;

    // Back-transform each decorrelated candidate: F = (Zᵀ)⁻¹·E (RTKLIB solve("T",Z,E)).
    // Z is unimodular, so the result is integer up to round-off.
    let mut zt = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        for j in 0..n {
            zt[i][j] = z[j + i * n]; // (Zᵀ)[i][j] = Z[j][i]
        }
    }
    let mut fixed_candidates: Vec<Vec<i64>> = Vec::with_capacity(m);
    for col in 0..m {
        let b: Vec<f64> = (0..n).map(|i| zn[i + col * n]).collect();
        let x = solve_linear(&zt, &b).map_err(|_| IlsError::Singular)?;
        fixed_candidates.push(x.iter().map(|&v| lam_round(v) as i64).collect());
    }

    // LAMBDA's mlambda distance `s` is computed in the decorrelated LtDL space; to
    // keep the reported scores consistent with `bounded_ils_search` (and bit-exact
    // against the explicit `Δᵀ Q⁻¹ Δ` reference / numpy goldens), recompute each
    // candidate's score with the same quadratic form and order them the same way
    // (score ascending, then cycles lexicographic). LAMBDA's only job here is to
    // FIND the candidate set; scoring/ratio use the canonical formula.
    let mut scored: Vec<(f64, Vec<i64>)> = fixed_candidates
        .into_iter()
        .map(|c| (quadratic_score(float_cycles, &c, &q_inv), c))
        .collect();
    scored.sort_by(|(sa, ca), (sb, cb)| {
        sa.partial_cmp(sb)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then_with(|| ca.cmp(cb))
    });

    let best_score = scored[0].0;
    let fixed = scored[0].1.clone();
    let second_best_score = scored.get(1).map(|(s, _)| *s);
    let ratio = integer_ratio(best_score, second_best_score);

    Ok(IlsResult {
        fixed,
        fixed_status: ratio_pass(ratio, ratio_threshold),
        ratio,
        best_score,
        second_best_score,
        // LAMBDA does not enumerate a box; report the number of candidate vectors.
        candidates_evaluated: m,
        covariance: q,
        covariance_inverse: q_inv,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_a_known_matrix() {
        let a = vec![vec![4.0, 7.0], vec![2.0, 6.0]];
        let inv = invert(&a).unwrap();
        // [[0.6, -0.7], [-0.2, 0.4]]
        assert!((inv[0][0] - 0.6).abs() < 1e-12);
        assert!((inv[0][1] + 0.7).abs() < 1e-12);
        assert!((inv[1][0] + 0.2).abs() < 1e-12);
        assert!((inv[1][1] - 0.4).abs() < 1e-12);
    }

    #[test]
    fn rejects_a_singular_matrix() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        assert!(invert(&a).is_err());
    }

    #[test]
    fn fixes_a_well_separated_lattice_point() {
        // Float ambiguities very close to integers, tight diagonal covariance:
        // the nearest lattice point dominates and the ratio test passes.
        let float = vec![3.02, -1.98, 5.01];
        let cov = vec![
            vec![0.01, 0.0, 0.0],
            vec![0.0, 0.01, 0.0],
            vec![0.0, 0.0, 0.01],
        ];
        let r = bounded_ils_search(&float, &cov, 1, 200_000, 3.0).unwrap();
        assert_eq!(r.fixed, vec![3, -2, 5]);
        assert!(r.fixed_status);
        assert!(r.ratio > 3.0);
        assert_eq!(r.candidates_evaluated, 27); // 3^3
    }

    #[test]
    fn refuses_an_ambiguous_lattice() {
        // Half-integer floats: nearest points are equidistant -> low ratio.
        let float = vec![0.5, 0.5];
        let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = bounded_ils_search(&float, &cov, 1, 200_000, 3.0).unwrap();
        assert!(!r.fixed_status);
        assert!(r.ratio < 3.0);
    }

    #[test]
    fn errors_when_the_lattice_exceeds_the_candidate_limit() {
        let float = vec![0.0, 0.0, 0.0];
        let cov = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        // 3^3 = 27 lattice points, limit 10 -> error.
        assert_eq!(
            bounded_ils_search(&float, &cov, 1, 10, 3.0),
            Err(IlsError::TooManyCandidates {
                evaluated: 27,
                limit: 10
            })
        );
    }

    #[test]
    fn rejects_pathological_lattice_before_allocating_ranges() {
        let float = vec![0.0, 0.0];
        let cov = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        let err = bounded_ils_search(&float, &cov, 1_000_000_000, 100, 3.0)
            .expect_err("over-limit lattice must be rejected before range allocation");
        assert!(matches!(
            err,
            IlsError::TooManyCandidates {
                evaluated,
                limit: 100
            } if evaluated > 100
        ));

        let normal = bounded_ils_search(&float, &cov, 1, 9, 3.0)
            .expect("within-limit lattice should still enumerate normally");
        assert_eq!(normal.fixed, vec![0, 0]);
        assert_eq!(normal.candidates_evaluated, 9);
    }

    #[test]
    fn rejects_bounded_search_ranges_outside_i64_domain() {
        let cov = vec![vec![1.0]];
        let expected = Err(IlsError::InvalidInput {
            field: "ils float_cycles",
            reason: "outside integer search range",
        });

        assert_eq!(bounded_ils_search(&[f64::MAX], &cov, 1, 3, 3.0), expected);
        assert_eq!(
            bounded_ils_search(&[i64::MAX as f64], &cov, 1, 3, 3.0),
            expected
        );
        assert_eq!(
            bounded_ils_search(&[i64::MIN as f64], &cov, 1, 3, 3.0),
            expected
        );
    }

    // --- LAMBDA port vs RTKLIB's own committed reference vectors ----------
    // (t_lambda.c utest1/utest2; see parity/generator/lambda_ref). RTKLIB's
    // unit test tolerates 1e-4 on the residuals; we hold the same.

    fn full_matrix(flat: &[f64], n: usize) -> Vec<Vec<f64>> {
        (0..n)
            .map(|i| (0..n).map(|j| flat[i * n + j]).collect())
            .collect()
    }

    #[test]
    fn lambda_matches_rtklib_utest1() {
        let a = [
            1585184.171,
            -6716599.430,
            3915742.905,
            7627233.455,
            9565990.879,
            989457273.200,
        ];
        #[rustfmt::skip]
        let q = full_matrix(&[
            0.227134, 0.112202, 0.112202, 0.112202, 0.112202, 0.103473,
            0.112202, 0.227134, 0.112202, 0.112202, 0.112202, 0.103473,
            0.112202, 0.112202, 0.227134, 0.112202, 0.112202, 0.103473,
            0.112202, 0.112202, 0.112202, 0.227134, 0.112202, 0.103473,
            0.112202, 0.112202, 0.112202, 0.112202, 0.227134, 0.103473,
            0.103473, 0.103473, 0.103473, 0.103473, 0.103473, 0.434339,
        ], 6);

        let r = lambda_ils_search(&a, &q, 3.0).unwrap();
        assert_eq!(
            r.fixed,
            vec![1585184, -6716599, 3915743, 7627234, 9565991, 989457273]
        );
        assert!((r.best_score - 3.5079844392).abs() < 1e-4);
        assert!((r.second_best_score.unwrap() - 3.70845619249).abs() < 1e-4);
    }

    #[test]
    fn lambda_matches_rtklib_utest2_strongly_correlated() {
        // The case the bounded box search cannot solve: the ILS optimum is up
        // to 14 cycles from componentwise rounding. LAMBDA gets it exactly.
        let a = [
            -13324172.755747,
            -10668894.713608,
            -7157225.010770,
            -6149367.974367,
            -7454133.571066,
            -5969200.494550,
            8336734.058423,
            6186974.084502,
            -17549093.883655,
            -13970158.922370,
        ];
        #[rustfmt::skip]
        let q = full_matrix(&[
            0.446320,0.223160,0.223160,0.223160,0.223160,0.572775,0.286388,0.286388,0.286388,0.286388,
            0.223160,0.446320,0.223160,0.223160,0.223160,0.286388,0.572775,0.286388,0.286388,0.286388,
            0.223160,0.223160,0.446320,0.223160,0.223160,0.286388,0.286388,0.572775,0.286388,0.286388,
            0.223160,0.223160,0.223160,0.446320,0.223160,0.286388,0.286388,0.286388,0.572775,0.286388,
            0.223160,0.223160,0.223160,0.223160,0.446320,0.286388,0.286388,0.286388,0.286388,0.572775,
            0.572775,0.286388,0.286388,0.286388,0.286388,0.735063,0.367531,0.367531,0.367531,0.367531,
            0.286388,0.572775,0.286388,0.286388,0.286388,0.367531,0.735063,0.367531,0.367531,0.367531,
            0.286388,0.286388,0.572775,0.286388,0.286388,0.367531,0.367531,0.735063,0.367531,0.367531,
            0.286388,0.286388,0.286388,0.572775,0.286388,0.367531,0.367531,0.367531,0.735063,0.367531,
            0.286388,0.286388,0.286388,0.286388,0.572775,0.367531,0.367531,0.367531,0.367531,0.735063,
        ], 10);

        let r = lambda_ils_search(&a, &q, 3.0).unwrap();
        assert_eq!(
            r.fixed,
            vec![
                -13324188, -10668901, -7157236, -6149379, -7454143, -5969220, 8336726, 6186960,
                -17549108, -13970171
            ]
        );
        assert!((r.best_score - 1506.43578925).abs() < 1e-4);
        assert!((r.second_best_score.unwrap() - 1612.81176533).abs() < 1e-4);
    }

    #[test]
    fn lambda_matches_rtklib_near_tie_low_ratio() {
        // Near-tie regime: the two best candidates are close, so RTKLIB's ratio
        // s[1]/s[0] sits at ~2.0 - squarely in the typical 1.5-3 acceptance band
        // and below our 3.0 threshold, so the fix is NOT accepted. Exercises the
        // ratio test rather than the integer selection.
        let a = [
            2.381283532896866,
            -4.153279079035503,
            6.181180039414691,
            -1.1716816183885634,
            3.144312353800454,
        ];
        #[rustfmt::skip]
        let q = full_matrix(&[
            0.30250000000000005, 0.11549999999999999, 0.09625, 0.12512500000000001, 0.11165,
            0.11549999999999999, 0.36, 0.105, 0.13649999999999998, 0.12179999999999998,
            0.09625, 0.105, 0.25, 0.11374999999999999, 0.10149999999999999,
            0.12512500000000001, 0.13649999999999998, 0.11374999999999999, 0.42250000000000004, 0.13194999999999998,
            0.11165, 0.12179999999999998, 0.10149999999999999, 0.13194999999999998, 0.3364,
        ], 5);

        let r = lambda_ils_search(&a, &q, 3.0).unwrap();
        assert_eq!(r.fixed, vec![2, -4, 6, -1, 3]);
        assert!((r.best_score - 1.1061496957026506).abs() < 1e-4);
        assert!((r.second_best_score.unwrap() - 2.2123104750064506).abs() < 1e-4);
        assert!((r.ratio - 2.0000100199830024).abs() < 1e-6);
        assert!(!r.fixed_status); // ratio < 3.0
    }

    #[test]
    fn lambda_matches_rtklib_easy_near_diagonal() {
        // Well-conditioned anchor: near-diagonal Q with float ambiguities very
        // close to integers. The ratio is huge (~249), the fix is accepted, and
        // LAMBDA and RTKLIB agree exactly.
        let a = [4.03, -2.97, 1.02, 5.98];
        #[rustfmt::skip]
        let q = full_matrix(&[
            0.018, 0.002, 0.0,    0.0,
            0.002, 0.025, 0.0,    0.0,
            0.0,   0.0,   0.012,  0.0015,
            0.0,   0.0,   0.0015, 0.03,
        ], 4);

        let r = lambda_ils_search(&a, &q, 3.0).unwrap();
        assert_eq!(r.fixed, vec![4, -3, 1, 6]);
        assert!((r.best_score - 0.12901401697831202).abs() < 1e-4);
        assert!((r.second_best_score.unwrap() - 32.16255699391752).abs() < 1e-4);
        assert!((r.ratio - 249.29505915100856).abs() < 1e-6);
        assert!(r.fixed_status); // ratio >> 3.0
    }

    #[test]
    fn lambda_agrees_with_box_search_in_regime() {
        // Weakly-correlated, ILS optimum near rounding: both kernels must agree.
        let a = vec![0.30, -0.40, 1.20];
        let q = vec![
            vec![0.50, 0.10, 0.05],
            vec![0.10, 0.50, 0.10],
            vec![0.05, 0.10, 0.50],
        ];
        let lam = lambda_ils_search(&a, &q, 3.0).unwrap();
        let box_ = bounded_ils_search(&a, &q, 1, 200_000, 3.0).unwrap();
        assert_eq!(lam.fixed, box_.fixed);
        assert!((lam.best_score - box_.best_score).abs() < 1e-9);
        assert!((lam.second_best_score.unwrap() - box_.second_best_score.unwrap()).abs() < 1e-9);
    }

    // --- input validation (both kernels reject malformed inputs cleanly) -----

    #[test]
    fn rejects_undersized_covariance() {
        // 2 ambiguities, 1x1 covariance - would index out of bounds without the guard.
        let a = vec![0.1, 0.2];
        let q = vec![vec![1.0]];
        assert_eq!(
            bounded_ils_search(&a, &q, 1, 200_000, 3.0),
            Err(IlsError::InvalidDimensions { n: 2, rows: 1 })
        );
        assert_eq!(
            lambda_ils_search(&a, &q, 3.0),
            Err(IlsError::InvalidDimensions { n: 2, rows: 1 })
        );
    }

    #[test]
    fn rejects_oversized_covariance() {
        // 1 ambiguity, 2x2 covariance - would silently use a submatrix.
        let a = vec![0.1];
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            bounded_ils_search(&a, &q, 1, 200_000, 3.0),
            Err(IlsError::InvalidDimensions { n: 1, rows: 2 })
        );
        assert_eq!(
            lambda_ils_search(&a, &q, 3.0),
            Err(IlsError::InvalidDimensions { n: 1, rows: 2 })
        );
    }

    #[test]
    fn rejects_ragged_covariance() {
        // Square row count but a row of the wrong width.
        let a = vec![0.1, 0.2];
        let q = vec![vec![1.0, 0.0], vec![0.0]];
        assert_eq!(
            bounded_ils_search(&a, &q, 1, 200_000, 3.0),
            Err(IlsError::InvalidDimensions { n: 2, rows: 1 })
        );
        assert_eq!(
            lambda_ils_search(&a, &q, 3.0),
            Err(IlsError::InvalidDimensions { n: 2, rows: 1 })
        );
    }

    #[test]
    fn bounded_search_rejects_invalid_covariance_geometry() {
        let a = vec![0.1, 0.2];
        let expected = Err(IlsError::InvalidInput {
            field: "ils covariance",
            reason: "not positive",
        });

        let negative_variance = vec![vec![-1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            bounded_ils_search(&a, &negative_variance, 1, 200_000, 3.0),
            expected
        );

        let asymmetric = vec![vec![1.0, 0.5], vec![0.4, 1.0]];
        assert_eq!(
            bounded_ils_search(&a, &asymmetric, 1, 200_000, 3.0),
            expected
        );

        let indefinite = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert_eq!(
            bounded_ils_search(&a, &indefinite, 1, 200_000, 3.0),
            expected
        );
    }

    #[test]
    fn lambda_search_rejects_invalid_covariance_geometry() {
        let a = vec![0.1, 0.2];
        let expected = Err(IlsError::InvalidInput {
            field: "ils covariance",
            reason: "not positive",
        });

        let negative_variance = vec![vec![-1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(lambda_ils_search(&a, &negative_variance, 3.0), expected);

        let asymmetric = vec![vec![1.0, 0.5], vec![0.4, 1.0]];
        assert_eq!(lambda_ils_search(&a, &asymmetric, 3.0), expected);

        let indefinite = vec![vec![1.0, 2.0], vec![2.0, 1.0]];
        assert_eq!(lambda_ils_search(&a, &indefinite, 3.0), expected);
    }

    #[test]
    fn rejects_empty_input() {
        let a: Vec<f64> = vec![];
        let q: Vec<Vec<f64>> = vec![];
        assert_eq!(
            bounded_ils_search(&a, &q, 1, 200_000, 3.0),
            Err(IlsError::InvalidDimensions { n: 0, rows: 0 })
        );
        assert_eq!(
            lambda_ils_search(&a, &q, 3.0),
            Err(IlsError::InvalidDimensions { n: 0, rows: 0 })
        );
    }

    #[test]
    fn rejects_non_finite_input() {
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            bounded_ils_search(&[f64::NAN, 0.2], &q, 1, 200_000, 3.0),
            Err(IlsError::NonFinite)
        );
        let q_inf = vec![vec![f64::INFINITY, 0.0], vec![0.0, 1.0]];
        assert_eq!(
            lambda_ils_search(&[0.1, 0.2], &q_inf, 3.0),
            Err(IlsError::NonFinite)
        );
    }

    #[test]
    fn rejects_invalid_ratio_thresholds() {
        let a = vec![0.1, 0.2];
        let q = vec![vec![1.0, 0.0], vec![0.0, 1.0]];

        for (threshold, reason) in [
            (-1.0, "negative"),
            (f64::NAN, "not finite"),
            (f64::INFINITY, "not finite"),
        ] {
            let expected = Err(IlsError::InvalidInput {
                field: ILS_RATIO_THRESHOLD_FIELD,
                reason,
            });
            assert_eq!(bounded_ils_search(&a, &q, 1, 200_000, threshold), expected);
            assert_eq!(lambda_ils_search(&a, &q, threshold), expected);
        }
    }

    #[test]
    fn exact_integer_fix_reports_finite_saturated_ratio() {
        let a = vec![1.0];
        let q = vec![vec![1.0]];

        let bounded = bounded_ils_search(&a, &q, 1, 3, 3.0).unwrap();
        assert_eq!(bounded.best_score, 0.0);
        assert_eq!(bounded.second_best_score, Some(1.0));
        assert_eq!(bounded.ratio, f64::MAX);
        assert!(bounded.ratio.is_finite());
        assert!(bounded.fixed_status);

        let lambda = lambda_ils_search(&a, &q, 3.0).unwrap();
        assert_eq!(lambda.best_score, 0.0);
        assert_eq!(lambda.second_best_score, Some(1.0));
        assert_eq!(lambda.ratio, f64::MAX);
        assert!(lambda.ratio.is_finite());
        assert!(lambda.fixed_status);
    }

    #[test]
    fn valid_ratio_threshold_still_controls_fix_status() {
        let a = vec![3.02, -1.98, 5.01];
        let q = vec![
            vec![0.01, 0.0, 0.0],
            vec![0.0, 0.01, 0.0],
            vec![0.0, 0.0, 0.01],
        ];

        let bounded_fixed = bounded_ils_search(&a, &q, 1, 200_000, 3.0).unwrap();
        let bounded_held = bounded_ils_search(&a, &q, 1, 200_000, 1.0e12).unwrap();
        assert!(bounded_fixed.fixed_status);
        assert!(!bounded_held.fixed_status);
        assert_eq!(bounded_fixed.fixed, bounded_held.fixed);

        let lambda_fixed = lambda_ils_search(&a, &q, 3.0).unwrap();
        let lambda_held = lambda_ils_search(&a, &q, 1.0e12).unwrap();
        assert!(lambda_fixed.fixed_status);
        assert!(!lambda_held.fixed_status);
        assert_eq!(lambda_fixed.fixed, lambda_held.fixed);
    }
}
