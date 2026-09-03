//! Embedded reference-data tables, relocated from `orbis_nif` so the core
//! crate is usable from Rust without Rustler or the BEAM.
//!
//! These tables are parity-critical: their numeric contents are reproduced
//! byte-for-byte from the upstream sources and must not be regenerated or
//! reformatted in ways that alter any literal.

/// IAU 2000A (MHB2000) lunisolar and planetary nutation series tables.
///
/// Contains coefficients from IERS Conventions (2010) Chapter 5 in 0.1
/// microarcsecond units used by Earth orientation frame transformations.
pub mod iau2000a;
pub mod iers;
