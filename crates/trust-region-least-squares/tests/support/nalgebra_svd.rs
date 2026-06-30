//! The pure-Rust `nalgebra` thin-SVD backend, re-exported under the historical
//! `NalgebraSvd` name for the benchmarks and validation tests.
//!
//! The implementation now lives in the library as
//! [`trust_region_least_squares::trf::NalgebraThinSvd`] (the crate's default SVD
//! seam). It is a legitimate independent SVD, intentionally NOT bit-exact with
//! SciPy, so it must never back the bit-exact fixtures; it exercises and times
//! the native solve.
//!
//! This file is pulled in via `#[path = ...] mod nalgebra_svd;` rather than being
//! its own test binary (it lives under `tests/support/`, which Cargo does not
//! compile as a test target).

#![allow(unused_imports)]

pub use trust_region_least_squares::trf::NalgebraThinSvd as NalgebraSvd;
