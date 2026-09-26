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
    /// A terrain tile file named for a one-degree cell could not be read as a
    /// DTED tile, or a lookup in a tile failed for a reason other than an
    /// unknown elevation.
    TerrainTile {
        /// Integer latitude tile id.
        lat_index: i32,
        /// Integer longitude tile id.
        lon_index: i32,
        /// Why the tile could not be read or queried.
        error: Box<crate::terrain::DtedTileError>,
    },
    /// A terrain tile file states an origin other than the one-degree cell
    /// its name gives, so its postings are not where the name places them.
    TerrainTileOrigin {
        /// The tile file.
        path: std::path::PathBuf,
        /// Latitude tile id the file name gives.
        lat_index: i32,
        /// Longitude tile id the file name gives.
        lon_index: i32,
        /// Origin latitude the file states, whole degrees.
        origin_latitude: i32,
        /// Origin longitude the file states, whole degrees.
        origin_longitude: i32,
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
    /// A precise-orbit position query falls in a contiguous run of fewer
    /// nodes than the interpolator takes, so no position is served there. RTKLIB
    /// `preceph.c` pephpos refuses on the same count.
    InsufficientPreciseNodes {
        /// The satellite queried.
        sat: crate::GnssSatelliteId,
        /// Nodes in the run serving the query.
        nodes: usize,
        /// Nodes the interpolator takes.
        required: usize,
    },
    /// An operation received inputs it cannot combine (e.g. an empty set of
    /// products to merge, or products on mismatched time scales, epoch grids, or
    /// coordinate-system labels).
    InvalidInput(String),
    /// A value given as an SP3 epoch interval is not a positive whole number of
    /// the 10-nanosecond ticks an SP3 epoch states. It names the field, the
    /// value and the reason.
    Sp3EpochInterval(crate::sp3::Sp3EpochIntervalError),
    /// A continuity check's speed bound or residual tolerance is not a finite
    /// number at least zero. It names the field, the value and the reason.
    ContinuityOptions(crate::sp3::ContinuityOptionsError),
    /// An SBAS block holds a value the SBAS wire form cannot carry as held.
    SbasEncode(Box<crate::sbas::SbasEncodeError>),
    /// An RTCM encoder refuses a value it cannot write as held.
    RtcmEncode(Box<crate::rtcm::RtcmEncodeError>),
    /// A decoded RTCM ephemeris names no satellite, or no broadcast record can
    /// be built from it.
    RtcmConversion(Box<crate::rtcm::RtcmConversionError>),
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
            Error::TerrainTile {
                lat_index,
                lon_index,
                error,
            } => write!(f, "terrain tile ({lat_index},{lon_index}): {error}"),
            Error::TerrainTileOrigin {
                path,
                lat_index,
                lon_index,
                origin_latitude,
                origin_longitude,
            } => write!(
                f,
                "{}: DTED origin ({origin_latitude},{origin_longitude}) does not match tile \
                 ({lat_index},{lon_index}) named by the file",
                path.display()
            ),
            Error::IonexOutOfCoverage(error) => write!(f, "IONEX out of coverage: {error}"),
            Error::IonexNodesNotAvailable(gap) => write!(f, "IONEX nodes not available: {gap}"),
            Error::IonexSlantUnavailable(refusal) => {
                write!(f, "IONEX slant delay unavailable: {refusal}")
            }
            Error::IonexEpoch(error) => write!(f, "invalid input: {error}"),
            Error::EpochOutOfRange => write!(f, "epoch out of range"),
            Error::InsufficientPreciseNodes {
                sat,
                nodes,
                required,
            } => write!(
                f,
                "{sat}: {nodes} precise orbit nodes serve the query, {required} are needed"
            ),
            Error::InvalidInput(msg) => write!(f, "invalid input: {msg}"),
            Error::Sp3EpochInterval(error) => write!(f, "invalid input: {error}"),
            Error::ContinuityOptions(error) => write!(f, "invalid input: {error}"),
            Error::SbasEncode(error) => write!(f, "SBAS encode error: {error}"),
            Error::RtcmEncode(error) => write!(f, "invalid input: {error}"),
            Error::RtcmConversion(error) => write!(f, "invalid input: {error}"),
            Error::Ut1OutsideCoverage(reason) => write!(f, "UT1 outside the table: {reason}"),
        }
    }
}

impl std::error::Error for Error {}
