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
    /// A GLONASS G1/G2 frequency lookup did not receive an FDMA channel.
    MissingGlonassChannel,
    /// A requested terrain tile is not present in the terrain store.
    MissingTerrainTile {
        /// Integer latitude tile id.
        lat_index: i32,
        /// Integer longitude tile id.
        lon_index: i32,
    },
    /// A terrain lookup weights a posting that the product marks as an unknown
    /// elevation (the DTED null value, all bits set), so the query has no
    /// height.
    UnknownTerrainElevation {
        /// Integer latitude tile id.
        lat_index: i32,
        /// Integer longitude tile id.
        lon_index: i32,
        /// Zero-based latitude posting index of the null posting in the tile.
        latitude_posting: usize,
        /// Zero-based longitude posting (profile) index of the null posting.
        longitude_posting: usize,
    },
    /// A terrain tile states a horizontal datum other than WGS84, so it cannot
    /// answer a WGS84 geodetic query without a datum transformation the
    /// terrain readers do not perform.
    NonWgs84TerrainTile {
        /// Integer latitude tile id.
        lat_index: i32,
        /// Integer longitude tile id.
        lon_index: i32,
        /// Datum the tile states.
        datum: crate::terrain::DtedHorizontalDatum,
    },
    /// An IONEX slant-delay query lies outside the product coverage.
    IonexOutOfCoverage(crate::ionex::IonexCoverageError),
    /// An IONEX slant-delay interpolation weights grid nodes the product gives
    /// as non-available.
    IonexNodesNotAvailable(Box<crate::ionex::IonexNodeGap>),
    /// An IONEX product gives no slant delay under the requested policy.
    IonexSlantUnavailable(crate::ionex::IonexSlantRefusal),
    /// An IONEX map epoch has no exact whole UTC second an epoch record can
    /// state.
    IonexEpoch(crate::ionex::IonexEpochError),
    /// A requested epoch lies outside the sampled / valid span.
    EpochOutOfRange,
    /// An operation received inputs it cannot combine (e.g. an empty set of
    /// products to merge, or products on mismatched time scales, epoch grids, or
    /// coordinate-system labels).
    InvalidInput(String),
    /// An SBAS block holds a value the SBAS wire form cannot carry as held.
    SbasEncode(Box<crate::sbas::SbasEncodeError>),
    /// The operation reads UT1 (Earth rotation) at an instant outside the UT1
    /// table and was not asked to accept the long-term UT1 there.
    Ut1OutsideCoverage(crate::astro::time::DegradeReason),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(msg) => write!(f, "parse error: {msg}"),
            Error::UnknownSatellite(id) => write!(f, "unknown satellite: {id}"),
            Error::MissingGlonassChannel => write!(f, "missing GLONASS FDMA channel"),
            Error::MissingTerrainTile {
                lat_index,
                lon_index,
            } => write!(f, "missing terrain tile ({lat_index},{lon_index})"),
            Error::UnknownTerrainElevation {
                lat_index,
                lon_index,
                latitude_posting,
                longitude_posting,
            } => write!(
                f,
                "unknown terrain elevation at posting lon={longitude_posting} lat={latitude_posting} of tile ({lat_index},{lon_index})"
            ),
            Error::NonWgs84TerrainTile {
                lat_index,
                lon_index,
                datum,
            } => write!(
                f,
                "terrain tile ({lat_index},{lon_index}) states horizontal datum {datum}, not WGS84"
            ),
            Error::IonexOutOfCoverage(error) => write!(f, "IONEX out of coverage: {error}"),
            Error::IonexNodesNotAvailable(gap) => write!(f, "IONEX nodes not available: {gap}"),
            Error::IonexSlantUnavailable(refusal) => {
                write!(f, "IONEX slant delay unavailable: {refusal}")
            }
            Error::IonexEpoch(error) => write!(f, "invalid input: {error}"),
            Error::EpochOutOfRange => write!(f, "epoch out of range"),
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::SbasEncode(error) => write!(f, "SBAS encode error: {error}"),
            Error::Ut1OutsideCoverage(reason) => write!(f, "UT1 outside the table: {reason}"),
        }
    }
}

impl std::error::Error for Error {}
