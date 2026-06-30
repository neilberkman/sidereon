# Changelog

All notable changes to `trust-region-least-squares` are documented here.

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
