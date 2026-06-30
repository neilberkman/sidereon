# Changelog

All notable changes to `trust-region-least-squares` are documented here.

## [Unreleased]

### Changed

- Removed the `nalgebra`, `rayon`, and `host-lapack` cargo features. There is now
  one build with every capability compiled in: the `nalgebra` thin-SVD backend,
  the `rayon`-fanned leave-one-out / multi-start batch paths, and the bit-exact
  host-LAPACK backend are all always available. The host-LAPACK backend is
  selected at runtime by pointing `TRUST_REGION_LEAST_SQUARES_LAPACK_PATH` at the
  host LAPACK/BLAS library (no feature flag, no recompile). The only remaining
  feature is `trace`, which gates zero-cost-when-off diagnostic output through the
  hot solver loop.
