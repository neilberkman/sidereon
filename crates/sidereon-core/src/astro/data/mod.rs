//! Embedded reference-data tables, relocated from `orbis_nif` so the core
//! crate is usable from Rust without Rustler or the BEAM.
//!
//! These tables are parity-critical: their numeric contents are reproduced
//! byte-for-byte from the upstream sources and must not be regenerated or
//! reformatted in ways that alter any literal.

pub mod iau2000a;
pub mod iers;
