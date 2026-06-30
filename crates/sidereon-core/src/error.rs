//! Crate error type.
//!
//! The variants cover broad parsing, lookup, interpolation, and invalid-input
//! failures across the crate.

use core::fmt;

/// Result alias for fallible `sidereon-core` operations.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors produced by the `sidereon-core` crate.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A product (SP3/RINEX/IONEX) could not be parsed.
    Parse(String),
    /// A requested satellite is not present in the product.
    UnknownSatellite(crate::GnssSatelliteId),
    /// A requested epoch lies outside the sampled / valid span.
    EpochOutOfRange,
    /// An operation received inputs it cannot combine (e.g. an empty set of
    /// products to merge, or products on mismatched time scales, epoch grids, or
    /// coordinate-system labels).
    InvalidInput(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(msg) => write!(f, "parse error: {msg}"),
            Error::UnknownSatellite(id) => write!(f, "unknown satellite: {id}"),
            Error::EpochOutOfRange => write!(f, "epoch out of range"),
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
        }
    }
}

impl std::error::Error for Error {}
