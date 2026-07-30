# Changelog

All notable changes to `trust-region-least-squares` are documented here.

## [Unreleased]

### Added

- Injectable power operations on the host-numerics seam: `ThinSvd::power`
  (elementwise `values ** exponent`) and `ThinSvd::power_scalar` (scalar
  `base ** exponent`). Both default to `Ok(None)`, so existing implementors
  compile unchanged and keep their current results. A supplied vector result is
  length-checked and a mismatch is surfaced as a typed error.
- `LossFunction::evaluate_with` and `rho_for_loss_with`, which route the
  robust-loss derivative powers (`z ** -0.5` and `z ** -1.5`, the only `**`
  expressions in SciPy's losses that miss numpy's `fast_scalar_power`
  shortcuts) through that seam. `huber` dispatches over the compressed
  `z[z > 1]` subset numpy's boolean mask produces; `soft_l1` over the whole
  `1 + z` vector. `linear`, `cauchy`, `arctan`, and every `cost_only` path have
  no power expression and dispatch nothing.
- Both trust-region alpha seeds (`(alpha_lower * alpha_upper) ** 0.5`, at
  initialization and at in-loop re-entry) now consult the scalar-power hook.
  Direct square roots are preserved everywhere numpy itself uses `sqrt` rather
  than the `power` ufunc.
- `hostlapack`: the NumPy host dispatch now implements both power hooks. The
  vector path keeps the existing CPU-feature gating (the `__svml_pow8` kernel
  from the umath extension of the exact NumPy runtime the backend is bound to
  when AVX-512 is present, numpy's scalar `npy_pow` inner loop otherwise); the
  scalar path uses the platform C `pow`. Unlike the older `power3` hook, these
  fail closed: a configured backend that cannot bind its runtime or resolve the
  kernel its CPU gate selected returns a typed error instead of falling back to
  Rust arithmetic.
- `LapackSvd::install` / `LapackSvd::installed` / `LapackSvd::with_numpy_blas_path`,
  giving the host configuration a process-wide identity. Installing an identical
  configuration is idempotent; a conflicting reconfiguration fails with
  `LapackError::ConflictingHostInstall`.

### Changed

- Public solve entry points, their behavior with no host result supplied, and
  the `power3` hook are all unchanged.

## [0.9.2] - 2026-07-20

- Included the repository MIT license and SciPy's full BSD 3-Clause terms in
  the crates.io source package. Numerical behavior and public APIs are
  unchanged.

## [0.9.1]

### Added

- A generic, data-driven solver surface so callers can use the engine without
  hand-wiring closures or an SVD backend: a default in-crate `NalgebraThinSvd`
  plus `trf_solve`; a `ResidualModel` trait with `solve_model`; built-in residual
  kinds (`BuiltinResidual::{Linear, Polynomial, Exponential}`) driven through
  `DataProblem` and `solve_data_problem`, with residual and Jacobian evaluated
  entirely in Rust (no per-iteration host-language callback).
- Batch leave-one-out / perturbed re-solve entries for the RAIM/FDE pattern:
  `solve_drop_one`, `solve_perturbed`, `solve_data_problem_drop_one` (rayon), and
  bit-identical serial twins (`solve_drop_one_serial`,
  `solve_data_problem_drop_one_serial`, ...) for single-threaded and wasm
  consumers.

### Changed

- Retargeted the bit-exact parity fixtures to the latest SciPy (1.18.0 / NumPy
  2.5.0). The replays reproduce SciPy bit-for-bit only on a non-AVX-512 x86_64
  host and are skipped by default; opt in with `SIDEREON_BITEXACT=1` (see
  `scripts/bitexact_gate.sh`).

- Removed the `nalgebra`, `rayon`, and `host-lapack` cargo features. There is now
  one build with every capability compiled in: the `nalgebra` thin-SVD backend,
  the `rayon`-fanned leave-one-out / multi-start batch paths, and the bit-exact
  host-LAPACK backend are all always available. The host-LAPACK backend is
  selected at runtime by pointing `TRUST_REGION_LEAST_SQUARES_LAPACK_PATH` at the
  host LAPACK/BLAS library (no feature flag, no recompile). The only remaining
  feature is `trace`, which gates zero-cost-when-off diagnostic output through the
  hot solver loop.
