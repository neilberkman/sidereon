# Changelog

All notable changes to `trust-region-least-squares` are documented here.

## [0.10.0] - 2026-08-22

### Added

- A host-runtime numerics backend seam for bit-exact reproduction of a pinned
  SciPy/NumPy runtime. `HostNumerics::power` routes elementwise
  `values ** exponent`, and `HostNumerics::power_scalar` routes scalar power;
  both default to `Ok(None)` to retain the pure-Rust calculation. Supplied
  vector results are length-checked and failures remain typed.
- `LossFunction::evaluate_with` and `rho_for_loss_with`, routing the robust-loss
  derivative powers (`z ** -0.5` and `z ** -1.5`) through the host backend.
  Huber dispatches the compressed `z[z > 1]` subset and soft-L1 dispatches the
  full `1 + z` vector. Losses and cost-only paths without a NumPy `power` call
  do not dispatch.
- Scalar-power routing for both trust-region alpha seeds written by SciPy as
  `(alpha_lower * alpha_upper) ** 0.5`. Direct square roots, including the
  exactly representable finite-difference step, remain direct operations.
- `LapackSvd::install`, `LapackSvd::installed`, and
  `LapackSvd::with_numpy_blas_path` for a write-once process record of the
  pinned runtime. Conflicting configuration returns
  `LapackError::ConflictingHostInstall`.
- Host NumPy vector-power dispatch reproducing the stride-0 scalar-exponent
  table (`-1.0` division, `0.0` literal one, `0.5` square root, `1.0` identity,
  and `2.0` multiplication) before using the runtime's `__svml_pow8` or scalar
  `npy_pow`/platform-`pow` fall-through kernel. Vector and scalar power
  deliberately disagree for `(-0.0) ** 0.5`, because NumPy's vector square-root
  row preserves negative zero while its scalar `pow` returns positive zero. A
  configured backend that cannot bind the selected runtime still fails closed.

### Changed

- **Breaking:** renamed the `ThinSvd` trait to `HostNumerics`, `SvdError` to
  `BackendError`, `TrfError::Svd` to `TrfError::Backend`, and
  `LossError::Host` to `LossError::Backend`. No compatibility aliases are
  provided.
- **Breaking:** removed the dedicated `power3` hook. The
  `phi_and_derivative` `denom ** 3` expression now calls the unified
  `HostNumerics::power(values, 3.0)` hook. A configured but unresolvable host
  therefore returns a typed error instead of quietly using Rust arithmetic;
  `Ok(None)` still explicitly declines to the Rust fallback.

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
