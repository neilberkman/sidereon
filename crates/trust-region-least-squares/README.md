# trust-region-least-squares

A dense trust-region-reflective (TRF) nonlinear least-squares solver that
reproduces [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
(`method='trf'`, 2-point Jacobian) **bit-for-bit**. It covers SciPy's dense
unbounded path for arbitrary problem dimension `n` and every SciPy loss
(`linear`, `soft_l1`, `huber`, `cauchy`, `arctan`) with the `f_scale`
robust-reweighting parameter; see [Status](#status) for details.

It is built to be a general-purpose solver, not a one-off port. Give it a
residual `r: Rⁿ → Rᵐ` and a starting point and it runs the same trust-region
Newton iteration SciPy does, down to the last bit. The operations that
determine the last bits of the trajectory, the thin SVD of the scaled Jacobian,
the small BLAS reductions around it, and the elementwise and scalar powers SciPy
writes as `**`, are *injected* through the `HostNumerics` trait. It is the
host-runtime numerics backend seam: SVD, BLAS reductions, and NumPy power
dispatch in one contract. Backing it with a pinned LAPACK/BLAS/NumPy runtime
lets the solver reproduce that runtime's numerical trajectory exactly rather
than merely remaining tolerance-close.

Only `svd` is required. Every other hook (`dot`, the matvecs, `power`, and
`power_scalar`) defaults to `Ok(None)`, meaning "no host result, use your own
arithmetic", so an implementation overrides exactly what it needs to pin.

## When to use it

When you need scipy-identical least-squares results in Rust:

- porting a Python/SciPy pipeline to Rust without changing converged values,
- cross-checking a Rust solver against SciPy as a reference oracle,
- pinning a numerical result so it cannot silently drift between language
  runtimes.

If you just want a fast least-squares solver and do not care about reproducing
SciPy's exact bits, a tolerance-based solver will be simpler and faster; this
crate trades that for an exact, reproducible numerical trajectory.

## Usage

Give the solver a residual, a Jacobian, a starting point, and a `HostNumerics`
backend. This example implements only the required thin SVD with
[`nalgebra`](https://docs.rs/nalgebra); the defaulted BLAS and power hooks
decline to the pure-Rust path. It fits the system whose least-squares solution
is `[1.0, 2.0]`:

```rust
use nalgebra::DMatrix;
use trust_region_least_squares::trf::{
    jacobian_2point, trf_no_bounds, JacobianFn, ResidualFn, BackendError, HostNumerics, TrfOptions,
};

struct NalgebraSvd;
impl HostNumerics for NalgebraSvd {
    fn svd(&self, a: &[f64], m: usize, n: usize)
        -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), BackendError>
    {
        let svd = DMatrix::from_row_slice(m, n, a).svd(true, true);
        let u = svd.u.ok_or_else(|| BackendError::Failed("no U".into()))?;
        let vt = svd.v_t.ok_or_else(|| BackendError::Failed("no V_t".into()))?;
        let mut u_rm = vec![0.0; m * n];
        for i in 0..m { for j in 0..n { u_rm[i * n + j] = u[(i, j)]; } }
        let mut vt_rm = vec![0.0; n * n];
        for i in 0..n { for j in 0..n { vt_rm[i * n + j] = vt[(i, j)]; } }
        Ok((u_rm, svd.singular_values.iter().copied().collect(), vt_rm))
    }
}

fn residual(x: &[f64], out: &mut Vec<f64>) {
    out.clear();
    out.push(x[0] - 1.0);
    out.push(x[1] - 2.0);
    out.push(x[0] + x[1] - 3.0);
}

let mut fun = residual;
let mut jac = |x: &[f64], f0: &[f64], out: &mut Vec<f64>| {
    let mut scratch = Vec::new();
    let mut inner = residual;
    jacobian_2point(&mut inner, x, f0, out, &mut scratch).unwrap();
};

let result = trf_no_bounds(
    &mut fun as &mut ResidualFn<'_>,
    &mut jac as &mut JacobianFn<'_>,
    &[0.0, 0.0],
    &NalgebraSvd,
    &TrfOptions::default(),
)
.unwrap();
assert!(result.success());
```

For bit-for-bit agreement with a pinned SciPy/NumPy runtime, inject the
host-LAPACK backend (`hostlapack::LapackSvd`) instead; it is compiled into the
single build and selected at runtime by pointing
`TRUST_REGION_LEAST_SQUARES_LAPACK_PATH` at the host LAPACK/BLAS. The iteration
is identical, only the host-numerics seam changes. Call
`LapackSvd::install` once at startup to record which runtime the process is
pinned to: installing the same configuration again is a no-op, and a conflicting
reconfiguration is rejected rather than silently applied. It is a guard, not a
router -- a solve still uses the backend you hand it.

Malformed input is rejected with a typed `TrfError` rather than a panic: empty
or non-finite `x0`, non-finite initial residuals, `m < n`, a wrong-length
Jacobian or residual, a non-positive/non-finite `f_scale` under a robust loss,
and bad `x_scale` are all surfaced as errors.

## Modules

- `trf`: the dense unbounded trust-region-reflective iteration matching
  `scipy.optimize._lsq.trf.trf_no_bounds`, with the injectable `HostNumerics`
  SVD/BLAS/power seam.
- `loss`: SciPy's robust loss functions (`construct_loss_function` +
  `IMPLEMENTED_LOSSES`) and `scale_for_robust_loss_function`, reproduced
  bit-for-bit, driven by `TrfOptions { loss, f_scale }`. The `_with` variants
  route the `z ** -0.5` / `z ** -1.5` derivative powers (`huber`, `soft_l1`)
  through the host-numerics seam.
- `numdiff`: the dense two-point finite-difference Jacobian matching SciPy's
  `_numdiff.approx_derivative(..., method="2-point")` path.
- `parity`: hex-bit fixture helpers, feature-gated trace output, and
  first-divergence reporting for diagnosing where two trajectories split.
- `hostlapack`: a `HostNumerics` implementation backed by a dynamically loaded
  host LAPACK/BLAS/NumPy runtime, used to reproduce a pinned SciPy runtime's exact
  SVD/BLAS/power results. Compiled into the single build and activated at
  runtime via `TRUST_REGION_LEAST_SQUARES_LAPACK_PATH`.

## Status

The iteration is general in `n`. Give it a residual `r: Rⁿ → Rᵐ` for any
`n ≥ 1` (with `m ≥ n` for the dense exact trust-region solve) and it follows
SciPy's `trf_no_bounds` trajectory bit-for-bit, for all five losses (`linear`,
`soft_l1`, `huber`, `cauchy`, `arctan`) plus `f_scale`. Bit-exact parity is
enforced by committed fixtures spanning `n ∈ {2, 3, 4, 5, 6, 8}` crossed with
every loss, replayed end-to-end through the host-LAPACK backend, alongside the
original `n = 3` regression fixtures.

### Reproducibility scope

Bit-for-bit floating-point agreement with a numerical library is intrinsically
**platform- and version-specific**: there is no cross-platform bit-exactness for
BLAS- and libm-heavy code. The committed parity is certified on:

- **Architecture:** Linux **x86_64** (glibc `libm`).
- **SciPy 1.18.0 / NumPy 2.5.0 / Python 3.12**, and the bundled **OpenBLAS**
  (`scipy-openblas`) shipped in those wheels.
- **OpenBLAS pinned deterministic:** `OPENBLAS_NUM_THREADS=1` (multi-threaded
  reductions sum in nondeterministic order) and a fixed `OPENBLAS_CORETYPE`
  (e.g. `HASWELL`) so the same SIMD kernel is selected regardless of host CPU.

Change any of these and the low bits move: Apple **Accelerate** (the macOS arm64
default), a different OpenBLAS build or CPU kernel (`AVX-512` vs `Haswell`), or a
different `libm` each produce a *different* trajectory, still internally correct.
The contiguity-sensitive products are matched to the exact call NumPy
makes on **this** stack: `Jᵀf` / `J·step` on the F-contiguous Jacobian via the
column-major BLAS path, `Uᵀf` / `V·rhs` via the C-contiguous row-major path.

The agreement is also "given the same host-numerics runtime": the injected
`HostNumerics` implementation pins the SVD, BLAS reductions, and relevant power
dispatch. The default pure-Rust `nalgebra` SVD is self-consistent but uses a
different numerical path, so it does not match SciPy bit-for-bit.

## Benchmarks

The crate's payoff is throughput on **small problems solved many times** (the
GNSS/TDOA hot-path: tiny systems re-solved millions of times), where SciPy's
per-call Python orchestration and array allocation dominate and a native Rust
loop skips all of it. The numbers below time the crate's **native path**:
`trf_no_bounds` driving a pure-Rust `nalgebra` thin SVD plus the crate's own
pure-Rust dot/matvec reductions, with no Python and no injected LAPACK, against
`scipy.optimize.least_squares` on the *same input data* (identical `matrix`,
`target`, `x0`, loss, and `f_scale`, loaded from one shared file) and the *same
mathematical residual* model. Each side evaluates that residual idiomatically (a
native row loop in Rust, vectorized `matrix @ x` in NumPy), so this is a timing
comparison, not a bit-for-bit one.

Measured on an Apple M5 Max (macOS 26.5.1, arm64), single-threaded BLAS, SciPy
1.18.0 / NumPy 2.5.0. Native times are criterion medians; SciPy times are the
best of seven batches. Per-solve wall-clock, lower is better:

| problem (`n`×`m`, loss)        | native Rust | SciPy      | speedup |
| ------------------------------ | ----------- | ---------- | ------- |
| small, 3×9, linear             | 6.1 µs      | 266 µs     | ~44×    |
| small, 4×11, linear            | 9.8 µs      | 274 µs     | ~28×    |
| small, 5×13, linear            | 13.1 µs     | 287 µs     | ~22×    |
| small, 3×9, soft_l1            | 10.9 µs     | 431 µs     | ~39×    |
| small, 4×11, huber             | 14.3 µs     | 531 µs     | ~37×    |
| large, 20×400, linear          | 864 µs      | 1.30 ms    | ~1.5×   |
| large, 40×120, linear          | 1.09 ms     | 1.89 ms    | ~1.7×   |

In the small/repeated regime the native path is **~20–45× faster**, because
SciPy's overhead is per *call* (input validation, building the `OptimizeResult`,
the Python-level trust-region loop). On a single large solve both sides are
SVD-bound and the gap narrows toward parity (still ~1.5–1.7× here, with `nalgebra`
competitive with OpenBLAS `gesdd` at these sizes).

Caveat on fairness: the parity (host-LAPACK) backend injects SciPy's *own*
LAPACK/BLAS, so benchmarking it would be SciPy-vs-SciPy; these numbers
deliberately use the native Rust SVD instead. Each side evaluates the residual
idiomatically (not a deliberately slow callback), so the comparison reflects
solver overhead. Reproduce with:

```sh
cargo bench -p trust-region-least-squares
python fixtures-generators/bench_scipy.py    # in the pinned venv
```

## Tests and fixtures

Parity is enforced against committed reference fixtures generated from a pinned
SciPy 1.18.0 / NumPy 2.5.0 runtime. All floating-point payloads are serialized
as f64 hex-bit strings and compared with `f64::to_bits`, never tolerances.
Regenerate them with the scripts in `fixtures-generators/` inside the pinned
Python environment (`fixtures-generators/requirements.txt`).

When an oracle package version changes, apply the repository's [Python oracle
version-pinning rules](../../docs/oracle-version-pinning.md). In particular,
hash only the meaningful FITPACK coefficient prefix, pin NumPy together with
its BLAS/LAPACK substrate for pseudo-inverse fixtures, and treat regeneration
under another package version as a new fixture.

The host-LAPACK parity test skips unless `TRUST_REGION_LEAST_SQUARES_LAPACK_PATH`
points at a LAPACK library; the backend itself is always compiled in.
The NumPy `power` fixture is CPU-dispatch-sensitive and is regenerated from the
pinned interpreter by `scripts/bitexact_gate.sh` before its replay.

## License

MIT
