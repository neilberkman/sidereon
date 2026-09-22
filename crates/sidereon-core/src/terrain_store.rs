//! Memory-mappable terrain tile store with an explicit vertical datum contract.
//!
//! The store is a single canonical container: a fixed header, a sorted tile
//! index, and one aligned `i16` posting payload per tile. Payloads are decoded
//! DTED posting values in orthometric metres, stored longitude-major with
//! latitude as the inner index. The reader keeps the input bytes in place and
//! indexes posting bytes directly, so an application can pass an mmap-backed
//! slice through [`MmapTerrain::from_bytes`].
//!
//! DTED and SRTM postings are orthometric heights, `H`, above the EGM96 mean sea
//! level geoid. Ellipsoidal height conversion is an explicit `h = H + N` step
//! using [`TerrainGeoidModel`].
//!
//! A DTED null posting (all bits set, the unknown elevation of MIL-PRF-89020B
//! 3.11.3.1) is stored as [`TERRAIN_STORE_NULL_POSTING`], the value that bit
//! pattern has under DTED signed magnitude and the value the specification
//! itself names as the null placeholder. No DTED height decodes to it, and a
//! lookup that weights such a posting returns
//! [`Error::UnknownTerrainElevation`] instead of a height. When the posting a
//! nearest lookup selects is a null on the tile edge, a neighbouring tile's
//! posting at exactly the same coordinates answers; a bilinear query exactly on
//! a tile edge is answered by the next tile sharing that edge, since every
//! weighted posting then lies on it.
//!
//! Each tile index record covers exactly the one-degree cell its integer tile
//! id names: latitude ids lie in `-90..=89`, longitude ids in `-180..=179`, and
//! the four stored bounds equal the id and the id plus one. The parser refuses
//! any other bounds, so lookups only ever see in-tile offsets.

use crate::artifact_bytes::ArtifactBytes;
pub use crate::artifact_bytes::DigestProvenance;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use crate::geoid::{egm96_undulation, GeoidError, GeoidGrid};
use crate::terrain::{
    self, terrain_grid_candidates, validate_lookup_coordinates, DtedHorizontalDatum,
    DtedInterpolation, DtedLookupOptions, DtedTile, TileGrid,
};
use crate::{Error, Result};

const STORE_MAGIC: &[u8; 8] = b"TMMAP001";
const STORE_VERSION: u16 = 1;
const STORE_ALIGNMENT: usize = 4096;
const STORE_HEADER_LEN: usize = 64;
const STORE_INDEX_RECORD_LEN: usize = 80;
const HEADER_VERSION_OFFSET: usize = 8;
const HEADER_DATUM_OFFSET: usize = 10;
const HEADER_TILE_COUNT_OFFSET: usize = 12;
const HEADER_INDEX_OFFSET_OFFSET: usize = 16;
const HEADER_DATA_OFFSET_OFFSET: usize = 24;
const HEADER_TOTAL_LEN_OFFSET: usize = 32;
const INDEX_LAT_OFFSET: usize = 0;
const INDEX_LON_OFFSET: usize = 4;
const INDEX_LON_COUNT_OFFSET: usize = 8;
const INDEX_LAT_COUNT_OFFSET: usize = 12;
const INDEX_DATA_OFFSET_OFFSET: usize = 16;
const INDEX_DATA_LEN_OFFSET: usize = 24;
const INDEX_CHECKSUM_OFFSET: usize = 32;
const INDEX_MIN_LAT_OFFSET: usize = 40;
const INDEX_MIN_LON_OFFSET: usize = 48;
const INDEX_MAX_LAT_OFFSET: usize = 56;
const INDEX_MAX_LON_OFFSET: usize = 64;
const INDEX_DATUM_OFFSET: usize = 72;
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
const MIN_TILE_LAT_INDEX: i32 = -90;
const MAX_TILE_LAT_INDEX: i32 = 89;
const MIN_TILE_LON_INDEX: i32 = -180;
const MAX_TILE_LON_INDEX: i32 = 179;

/// Stored posting value marking a DTED null (unknown) elevation.
///
/// It is the value DTED's all-bits-set null decodes to under signed magnitude
/// (MIL-PRF-89020B 3.11.3.1 names it as -32,767), so stores written before the
/// null was typed read back the same way.
pub const TERRAIN_STORE_NULL_POSTING: i16 = -32_767;
const EGM96_DAC_REMEDIATION: &str =
    "obtain the public NGA EGM96 15-arcminute WW15MGH.DAC file and load it with Egm96FifteenMinuteGeoid::from_ww15mgh_dac_path or Egm96FifteenMinuteGeoid::from_ww15mgh_dac_bytes";

/// Vertical datum carried by terrain store tile index records.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerticalDatum {
    /// Orthometric height `H` in metres above the EGM96 mean sea level geoid.
    Egm96MslOrthometric,
}

impl VerticalDatum {
    fn tag(self) -> u8 {
        match self {
            Self::Egm96MslOrthometric => 1,
        }
    }

    fn from_tag(tag: u8) -> core::result::Result<Self, TerrainStoreError> {
        match tag {
            1 => Ok(Self::Egm96MslOrthometric),
            other => Err(TerrainStoreError::UnsupportedDatum { tag: other }),
        }
    }
}

/// Orthometric height `H` in metres above the EGM96 mean sea level geoid.
///
/// DTED/SRTM terrain postings use this datum. Convert to ellipsoidal height
/// only through [`Self::to_ellipsoidal_height_deg`] or
/// [`Self::to_ellipsoidal_height_rad`], which require a pinned geoid tier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrthometricHeightM {
    /// Orthometric height `H` in metres.
    pub value_m: f64,
}

impl OrthometricHeightM {
    /// Build an orthometric height `H` in metres.
    #[must_use]
    pub const fn new(value_m: f64) -> Self {
        Self { value_m }
    }

    /// Return the orthometric height `H` in metres.
    #[must_use]
    pub const fn metres(self) -> f64 {
        self.value_m
    }

    /// Convert this orthometric height to ellipsoidal height `h = H + N`.
    ///
    /// Inputs are geodetic `(latitude_deg, longitude_deg)`, matching the geoid
    /// module's axis order. Terrain lookup APIs use `(longitude_deg,
    /// latitude_deg)`, so call sites should pass the axes deliberately.
    ///
    /// [`TerrainGeoidModel::Egm96OneDegree`] uses the embedded EGM96 1-degree
    /// grid. It agrees with the full EGM96 15-arcminute grid to about 0.4 m RMS,
    /// so byte-identical terrain heights do not imply byte-identical
    /// ellipsoidal heights across geoid tiers.
    pub fn to_ellipsoidal_height_deg(
        self,
        latitude_deg: f64,
        longitude_deg: f64,
        geoid: TerrainGeoidModel<'_>,
    ) -> core::result::Result<EllipsoidalHeightM, TerrainDatumError> {
        Ok(EllipsoidalHeightM::new(
            self.value_m + geoid.undulation_deg(latitude_deg, longitude_deg),
        ))
    }

    /// Convert this orthometric height to ellipsoidal height `h = H + N`.
    ///
    /// Inputs are geodetic `(latitude_rad, longitude_rad)`, matching the geoid
    /// module's axis order. [`TerrainGeoidModel::Egm96OneDegree`] uses the
    /// embedded EGM96 1-degree grid. It agrees with the full EGM96
    /// 15-arcminute grid to about 0.4 m RMS, so byte-identical terrain heights
    /// do not imply byte-identical ellipsoidal heights across geoid tiers.
    pub fn to_ellipsoidal_height_rad(
        self,
        latitude_rad: f64,
        longitude_rad: f64,
        geoid: TerrainGeoidModel<'_>,
    ) -> core::result::Result<EllipsoidalHeightM, TerrainDatumError> {
        Ok(EllipsoidalHeightM::new(
            self.value_m + geoid.undulation_rad(latitude_rad, longitude_rad),
        ))
    }
}

/// Ellipsoidal height `h` in metres above the WGS84 reference ellipsoid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EllipsoidalHeightM {
    /// Ellipsoidal height `h` in metres.
    pub value_m: f64,
}

impl EllipsoidalHeightM {
    /// Build an ellipsoidal height `h` in metres.
    #[must_use]
    pub const fn new(value_m: f64) -> Self {
        Self { value_m }
    }

    /// Return the ellipsoidal height `h` in metres.
    #[must_use]
    pub const fn metres(self) -> f64 {
        self.value_m
    }
}

/// Loaded EGM96 15-arcminute geoid grid for explicit terrain datum conversion.
///
/// This type never falls back to the embedded 1-degree grid. A missing
/// `WW15MGH.DAC` file returns [`TerrainDatumError::MissingEgm96Dac`].
#[derive(Clone, Debug, PartialEq)]
pub struct Egm96FifteenMinuteGeoid {
    grid: GeoidGrid,
}

impl Egm96FifteenMinuteGeoid {
    /// Load `WW15MGH.DAC` bytes as an EGM96 15-arcminute geoid grid.
    pub fn from_ww15mgh_dac_bytes(bytes: &[u8]) -> core::result::Result<Self, TerrainDatumError> {
        let grid = GeoidGrid::from_egm96_dac(bytes).map_err(TerrainDatumError::Geoid)?;
        Ok(Self { grid })
    }

    /// Read and load `WW15MGH.DAC` from disk as an EGM96 15-arcminute geoid
    /// grid.
    ///
    /// If the file is absent, this returns
    /// [`TerrainDatumError::MissingEgm96Dac`] with a remediation string naming
    /// the required grid and loader. It does not fall back to the embedded
    /// EGM96 1-degree grid.
    pub fn from_ww15mgh_dac_path(
        path: impl AsRef<Path>,
    ) -> core::result::Result<Self, TerrainDatumError> {
        let path = path.as_ref();
        let bytes = match fs::read(path) {
            Ok(bytes) => bytes,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                return Err(TerrainDatumError::MissingEgm96Dac {
                    path: path.to_path_buf(),
                    remediation: EGM96_DAC_REMEDIATION,
                });
            }
            Err(err) => {
                return Err(TerrainDatumError::Io {
                    path: path.to_path_buf(),
                    message: err.to_string(),
                });
            }
        };
        Self::from_ww15mgh_dac_bytes(&bytes)
    }

    /// Borrow the loaded EGM96 15-arcminute geoid grid.
    #[must_use]
    pub const fn grid(&self) -> &GeoidGrid {
        &self.grid
    }
}

/// Geoid tier used to convert terrain orthometric height `H` to ellipsoidal
/// height `h`.
#[derive(Clone, Copy, Debug)]
pub enum TerrainGeoidModel<'a> {
    /// Embedded EGM96 1-degree grid, always available in-process.
    ///
    /// This tier agrees with the full EGM96 15-arcminute grid to about 0.4 m RMS.
    /// It is the zero-setup path for `h = H + N` terrain conversion.
    Egm96OneDegree,
    /// Caller-supplied EGM96 15-arcminute `WW15MGH.DAC` grid.
    ///
    /// Build this with [`Egm96FifteenMinuteGeoid::from_ww15mgh_dac_path`] or
    /// [`Egm96FifteenMinuteGeoid::from_ww15mgh_dac_bytes`]. Missing files fail
    /// closed with [`TerrainDatumError::MissingEgm96Dac`].
    Egm96FifteenMinute(&'a Egm96FifteenMinuteGeoid),
}

impl TerrainGeoidModel<'_> {
    fn undulation_deg(self, latitude_deg: f64, longitude_deg: f64) -> f64 {
        match self {
            Self::Egm96OneDegree => {
                egm96_undulation(latitude_deg.to_radians(), longitude_deg.to_radians())
            }
            Self::Egm96FifteenMinute(grid) => grid.grid.undulation_deg(latitude_deg, longitude_deg),
        }
    }

    fn undulation_rad(self, latitude_rad: f64, longitude_rad: f64) -> f64 {
        match self {
            Self::Egm96OneDegree => egm96_undulation(latitude_rad, longitude_rad),
            Self::Egm96FifteenMinute(grid) => grid.grid.undulation_rad(latitude_rad, longitude_rad),
        }
    }
}

/// Errors from vertical-datum conversion and optional geoid-grid loading.
#[derive(Debug, Clone, PartialEq)]
pub enum TerrainDatumError {
    /// Terrain lookup failed before datum conversion.
    Terrain(Error),
    /// A geoid grid could not be parsed.
    Geoid(GeoidError),
    /// Reading a geoid grid failed for a reason other than absence.
    Io {
        /// Path that could not be read.
        path: PathBuf,
        /// I/O error text.
        message: String,
    },
    /// The EGM96 15-arcminute `WW15MGH.DAC` grid was requested but is absent.
    MissingEgm96Dac {
        /// Path that was requested.
        path: PathBuf,
        /// Remediation text naming the required grid and loader.
        remediation: &'static str,
    },
}

impl core::fmt::Display for TerrainDatumError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Terrain(err) => write!(f, "terrain lookup failed: {err}"),
            Self::Geoid(err) => write!(f, "geoid grid failed: {err}"),
            Self::Io { path, message } => {
                write!(f, "{} could not be read: {message}", path.display())
            }
            Self::MissingEgm96Dac { path, remediation } => {
                write!(f, "{} is missing; {remediation}", path.display())
            }
        }
    }
}

impl std::error::Error for TerrainDatumError {}

impl From<Error> for TerrainDatumError {
    fn from(value: Error) -> Self {
        Self::Terrain(value)
    }
}

/// Metadata for one tile index record in a memory-mappable terrain store.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TerrainStoreTileIndex {
    /// Integer latitude tile id, e.g. `36` for a tile covering `36..37` degrees.
    pub lat_index: i32,
    /// Integer longitude tile id, e.g. `-107` for a tile covering
    /// `-107..-106` degrees.
    pub lon_index: i32,
    /// Western edge longitude in degrees.
    pub min_longitude_deg: f64,
    /// Southern edge latitude in degrees.
    pub min_latitude_deg: f64,
    /// Eastern edge longitude in degrees.
    pub max_longitude_deg: f64,
    /// Northern edge latitude in degrees.
    pub max_latitude_deg: f64,
    /// Number of longitude postings.
    pub lon_count: u32,
    /// Number of latitude postings.
    pub lat_count: u32,
    /// Byte offset of this tile's posting payload in the store.
    pub data_offset: u64,
    /// Byte length of this tile's posting payload in the store.
    pub data_len: u64,
    /// FNV-1a checksum of this tile's posting payload bytes.
    pub checksum64: u64,
    /// Vertical datum for the tile's posting payload.
    pub vertical_datum: VerticalDatum,
}

/// Integer terrain tile id used by DTED and terrain-store accessors.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TerrainTileId {
    /// Integer latitude tile id, e.g. `36` for a tile covering `36..37` degrees.
    pub lat_index: i32,
    /// Integer longitude tile id, e.g. `-107` for a tile covering
    /// `-107..-106` degrees.
    pub lon_index: i32,
}

impl TerrainTileId {
    /// Build an integer terrain tile id.
    #[must_use]
    pub const fn new(lat_index: i32, lon_index: i32) -> Self {
        Self {
            lat_index,
            lon_index,
        }
    }
}

/// One explicit DTED tile source for list-based terrain-store conversion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DtedTileListEntry {
    /// Expected integer tile id for `path`.
    pub tile_id: TerrainTileId,
    /// Path to the DTED `.dt2` tile bytes.
    pub path: PathBuf,
}

impl DtedTileListEntry {
    /// Build a tile-list entry from a tile id and DTED path.
    #[must_use]
    pub fn new(tile_id: TerrainTileId, path: impl Into<PathBuf>) -> Self {
        Self {
            tile_id,
            path: path.into(),
        }
    }

    /// Build a tile-list entry from integer tile indices and a DTED path.
    #[must_use]
    pub fn from_indices(lat_index: i32, lon_index: i32, path: impl Into<PathBuf>) -> Self {
        Self::new(TerrainTileId::new(lat_index, lon_index), path)
    }
}

/// Errors from terrain store conversion, serialization, and parsing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TerrainStoreError {
    /// File or directory I/O failed.
    Io {
        /// Path being accessed.
        path: PathBuf,
        /// I/O error text.
        message: String,
    },
    /// DTED or terrain store bytes could not be parsed.
    Parse {
        /// Human-readable parse reason.
        reason: String,
    },
    /// The terrain store version is not supported.
    UnsupportedVersion {
        /// Version tag found in the store header.
        version: u16,
    },
    /// The terrain store datum tag is not supported.
    UnsupportedDatum {
        /// Datum tag found in the store header or tile index.
        tag: u8,
    },
    /// Two input DTED files resolved to the same integer tile id.
    DuplicateTile {
        /// Latitude tile id.
        lat_index: i32,
        /// Longitude tile id.
        lon_index: i32,
    },
    /// A list-builder entry's supplied id did not match the DTED file origin.
    TileIdMismatch {
        /// Path whose parsed DTED origin did not match the supplied id.
        path: PathBuf,
        /// Expected tile id supplied by the caller.
        expected: TerrainTileId,
        /// Tile id parsed from the DTED file.
        found: TerrainTileId,
    },
    /// A tile payload checksum did not match its index record.
    Checksum {
        /// Latitude tile id.
        lat_index: i32,
        /// Longitude tile id.
        lon_index: i32,
        /// Checksum stored in the index record.
        expected: u64,
        /// Checksum computed from the posting payload.
        found: u64,
    },
    /// An attested full-store checksum did not match the bytes opened.
    AttestedChecksumMismatch {
        /// Checksum asserted by the caller.
        expected: u64,
        /// Checksum computed from the full store byte span.
        found: u64,
    },
    /// A tile index record names a tile id outside the coordinate domain:
    /// latitude ids lie in `-90..=89` and longitude ids in `-180..=179`.
    TileIdOutOfRange {
        /// Latitude tile id.
        lat_index: i32,
        /// Longitude tile id.
        lon_index: i32,
    },
    /// A tile index bound is not the edge of the one-degree cell its tile id
    /// names.
    TileBoundsMismatch {
        /// Latitude tile id.
        lat_index: i32,
        /// Longitude tile id.
        lon_index: i32,
        /// Index field whose value disagrees with the tile id.
        field: &'static str,
    },
    /// A DTED input states a horizontal datum other than WGS84. The store
    /// records no datum and answers WGS84 queries, so such a tile is refused
    /// rather than stored as if it were WGS84.
    NonWgs84Tile {
        /// Path of the DTED tile.
        path: PathBuf,
        /// Datum the tile's DSI record states.
        datum: DtedHorizontalDatum,
    },
}

impl core::fmt::Display for TerrainStoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io { path, message } => write!(f, "{} failed: {message}", path.display()),
            Self::Parse { reason } => write!(f, "terrain store parse error: {reason}"),
            Self::UnsupportedVersion { version } => {
                write!(f, "terrain store version {version} is not supported")
            }
            Self::UnsupportedDatum { tag } => {
                write!(f, "terrain store vertical datum tag {tag} is not supported")
            }
            Self::DuplicateTile {
                lat_index,
                lon_index,
            } => write!(f, "duplicate terrain tile ({lat_index},{lon_index})"),
            Self::TileIdMismatch {
                path,
                expected,
                found,
            } => write!(
                f,
                "{} tile id expected ({},{}) but DTED origin is ({},{})",
                path.display(),
                expected.lat_index,
                expected.lon_index,
                found.lat_index,
                found.lon_index
            ),
            Self::Checksum {
                lat_index,
                lon_index,
                expected,
                found,
            } => write!(
                f,
                "terrain tile ({lat_index},{lon_index}) checksum expected {expected:#x} but found {found:#x}"
            ),
            Self::AttestedChecksumMismatch { expected, found } => write!(
                f,
                "attested terrain store checksum expected {expected:#x} but found {found:#x}"
            ),
            Self::TileIdOutOfRange {
                lat_index,
                lon_index,
            } => write!(
                f,
                "terrain tile id ({lat_index},{lon_index}) is outside the coordinate domain"
            ),
            Self::TileBoundsMismatch {
                lat_index,
                lon_index,
                field,
            } => write!(
                f,
                "terrain tile ({lat_index},{lon_index}) {field} is not an edge of that one-degree tile"
            ),
            Self::NonWgs84Tile { path, datum } => write!(
                f,
                "{} states horizontal datum {datum}, not WGS84",
                path.display()
            ),
        }
    }
}

impl std::error::Error for TerrainStoreError {}

#[derive(Clone, Debug)]
struct MmapTile {
    index: TerrainStoreTileIndex,
}

impl MmapTile {
    fn contains(&self, longitude_deg: f64, latitude_deg: f64) -> bool {
        latitude_deg >= self.index.min_latitude_deg
            && latitude_deg <= self.index.max_latitude_deg
            && longitude_deg >= self.index.min_longitude_deg
            && longitude_deg <= self.index.max_longitude_deg
    }

    fn get_elevation(&self, bytes: &[u8], longitude_deg: f64, latitude_deg: f64) -> Result<i16> {
        if !self.contains(longitude_deg, latitude_deg) {
            return Err(Error::Parse(format!(
                "point ({longitude_deg},{latitude_deg}) is outside terrain store tile ({},{})",
                self.index.min_longitude_deg, self.index.min_latitude_deg
            )));
        }

        let lat_count = self.index.lat_count as usize;
        let lon_count = self.index.lon_count as usize;
        let latitude_index = terrain::nearest_posting_index(
            latitude_deg - self.index.min_latitude_deg,
            lat_count - 1,
        )
        .map_err(Error::Parse)?;
        let longitude_index = terrain::nearest_posting_index(
            longitude_deg - self.index.min_longitude_deg,
            lon_count - 1,
        )
        .map_err(Error::Parse)?;
        self.posting(bytes, longitude_index, latitude_index)
    }

    /// Stored posting `(longitude_index, latitude_index)`, or
    /// [`Error::UnknownTerrainElevation`] for the null marker.
    fn posting(&self, bytes: &[u8], longitude_index: usize, latitude_index: usize) -> Result<i16> {
        let lat_count = self.index.lat_count as usize;
        let lon_count = self.index.lon_count as usize;
        if latitude_index >= lat_count || longitude_index >= lon_count {
            return Err(Error::Parse(format!(
                "posting index out of bounds lon={longitude_index} lat={latitude_index}"
            )));
        }

        let sample_start =
            self.index.data_offset as usize + 2 * (longitude_index * lat_count + latitude_index);
        let value = i16::from_le_bytes([bytes[sample_start], bytes[sample_start + 1]]);
        if value == TERRAIN_STORE_NULL_POSTING {
            return Err(Error::UnknownTerrainElevation {
                lat_index: self.index.lat_index,
                lon_index: self.index.lon_index,
                latitude_posting: latitude_index,
                longitude_posting: longitude_index,
            });
        }
        Ok(value)
    }

    fn grid(&self) -> TileGrid {
        TileGrid {
            lat_index: self.index.lat_index,
            lon_index: self.index.lon_index,
            lon_count: self.index.lon_count as usize,
            lat_count: self.index.lat_count as usize,
        }
    }
}

/// Memory-mappable terrain reader backed by a terrain store byte span.
///
/// Scalar and batch terrain lookups return orthometric metres, `H`, above the
/// EGM96 mean sea level geoid. Use [`Self::ellipsoidal_height_m`] or
/// [`OrthometricHeightM::to_ellipsoidal_height_deg`] for the explicit
/// `h = H + N` conversion to WGS84 ellipsoidal height.
#[derive(Clone, Debug)]
pub struct MmapTerrain<'a> {
    bytes: ArtifactBytes<'a>,
    tiles: Vec<MmapTile>,
    by_grid: HashMap<(i32, i32), usize>,
    tile_index: Vec<TerrainStoreTileIndex>,
    tile_ids: Vec<TerrainTileId>,
    vertical_datum: VerticalDatum,
    digest_provenance: DigestProvenance,
    attested_checksum64: Option<u64>,
}

#[derive(Clone, Copy)]
enum ChecksumValidation {
    Verified,
    Attested(u64),
}

impl ChecksumValidation {
    const fn digest_provenance(self) -> DigestProvenance {
        match self {
            Self::Verified => DigestProvenance::Verified,
            Self::Attested(_) => DigestProvenance::Attested,
        }
    }

    const fn attested_checksum64(self) -> Option<u64> {
        match self {
            Self::Verified => None,
            Self::Attested(checksum64) => Some(checksum64),
        }
    }

    const fn verifies_payloads(self) -> bool {
        matches!(self, Self::Verified)
    }
}

impl MmapTerrain<'static> {
    /// Parse an owned terrain store byte vector.
    pub fn from_vec(bytes: Vec<u8>) -> core::result::Result<Self, TerrainStoreError> {
        Self::from_backing(ArtifactBytes::Owned(bytes), ChecksumValidation::Verified)
    }

    /// Parse owned terrain store bytes using a caller-attested checksum.
    ///
    /// The byte-based counterpart of [`Self::from_path_attested`], for callers
    /// that already hold the store bytes (an interface layer, an object-store
    /// read) alongside a trustworthy content measurement. Same contract:
    /// structural checks run unconditionally, the per-tile payload hashing is
    /// replaced by the claim, and the handle reports
    /// [`DigestProvenance::Attested`] until [`Self::verify`] succeeds.
    pub fn from_vec_attested(
        bytes: Vec<u8>,
        claimed_checksum64: u64,
    ) -> core::result::Result<Self, TerrainStoreError> {
        Self::from_backing(
            ArtifactBytes::Owned(bytes),
            ChecksumValidation::Attested(claimed_checksum64),
        )
    }

    /// Open and parse a terrain store file.
    ///
    /// With the `mmap` feature the file is memory-mapped read-only and this
    /// reader owns the mapping; without it the file is read into memory. The
    /// entry point is the same either way, so enabling the feature speeds up
    /// every existing caller rather than asking anyone to migrate.
    ///
    /// Mapping is what makes a very large store usable at all: construction
    /// parses only the header, datum tag, and tile index, and lookups are
    /// demand-paged, so a reader querying a geographically local region never
    /// faults in the rest of the file.
    pub fn from_path(path: impl AsRef<Path>) -> core::result::Result<Self, TerrainStoreError> {
        let path = path.as_ref();

        #[cfg(feature = "mmap")]
        {
            let bytes = crate::artifact_bytes::map_file_read_only(path).map_err(|err| {
                TerrainStoreError::Io {
                    path: path.to_path_buf(),
                    message: err.to_string(),
                }
            })?;
            Self::from_backing(bytes, ChecksumValidation::Verified)
        }

        #[cfg(not(feature = "mmap"))]
        {
            let bytes = fs::read(path).map_err(|err| TerrainStoreError::Io {
                path: path.to_path_buf(),
                message: err.to_string(),
            })?;
            Self::from_vec(bytes)
        }
    }

    /// Open a terrain store using a caller-attested full-store checksum.
    ///
    /// This performs the same header, index, dimension, length, and tile-bound
    /// validation as [`Self::from_path`] without hashing tile payloads. Terrain
    /// headers do not carry a full-store checksum, so the claim is recorded as
    /// supplied and can be checked later with [`Self::verify`]. With the `mmap`
    /// feature the file is memory-mapped read-only; without it the file is read
    /// into memory.
    pub fn from_path_attested(
        path: impl AsRef<Path>,
        claimed_checksum64: u64,
    ) -> core::result::Result<Self, TerrainStoreError> {
        let path = path.as_ref();

        #[cfg(feature = "mmap")]
        {
            let bytes = crate::artifact_bytes::map_file_read_only(path).map_err(|err| {
                TerrainStoreError::Io {
                    path: path.to_path_buf(),
                    message: err.to_string(),
                }
            })?;
            Self::from_backing(bytes, ChecksumValidation::Attested(claimed_checksum64))
        }

        #[cfg(not(feature = "mmap"))]
        {
            let bytes = fs::read(path).map_err(|err| TerrainStoreError::Io {
                path: path.to_path_buf(),
                message: err.to_string(),
            })?;
            Self::from_backing(
                ArtifactBytes::Owned(bytes),
                ChecksumValidation::Attested(claimed_checksum64),
            )
        }
    }
}

impl<'a> MmapTerrain<'a> {
    /// Parse a borrowed terrain store byte span.
    ///
    /// The reader keeps the byte span in place and indexes posting payloads by
    /// offset. Passing an mmap-backed slice gives a zero-copy reader.
    pub fn from_bytes(bytes: &'a [u8]) -> core::result::Result<Self, TerrainStoreError> {
        Self::from_backing(ArtifactBytes::Borrowed(bytes), ChecksumValidation::Verified)
    }

    fn from_backing(
        bytes: ArtifactBytes<'a>,
        checksum_validation: ChecksumValidation,
    ) -> core::result::Result<Self, TerrainStoreError> {
        let parsed = parse_store(bytes.as_slice(), checksum_validation)?;
        Ok(Self {
            bytes,
            tiles: parsed.tiles,
            by_grid: parsed.by_grid,
            tile_index: parsed.tile_index,
            tile_ids: parsed.tile_ids,
            vertical_datum: parsed.vertical_datum,
            digest_provenance: checksum_validation.digest_provenance(),
            attested_checksum64: checksum_validation.attested_checksum64(),
        })
    }

    /// Borrow the original terrain store bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        self.bytes.as_slice()
    }

    /// Whether this reader is backed by a memory map rather than a copy in
    /// process memory.
    #[must_use]
    pub fn is_memory_mapped(&self) -> bool {
        self.bytes.is_memory_mapped()
    }

    /// Return the store's file-level vertical datum.
    #[must_use]
    pub const fn vertical_datum(&self) -> VerticalDatum {
        self.vertical_datum
    }

    /// Borrow the parsed tile index records.
    #[must_use]
    pub fn tile_index(&self) -> &[TerrainStoreTileIndex] {
        &self.tile_index
    }

    /// Return the number of tiles present in this terrain store.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.tile_ids.len()
    }

    /// Borrow the sorted integer tile ids present in this terrain store.
    #[must_use]
    pub fn tile_ids(&self) -> &[TerrainTileId] {
        &self.tile_ids
    }

    /// Return who computed the checksum carried by this reader.
    #[must_use]
    pub const fn digest_provenance(&self) -> DigestProvenance {
        self.digest_provenance
    }

    /// Return the full-store checksum carried by this reader.
    ///
    /// Verified readers compute the FNV-1a checksum on demand. Attested readers
    /// return the caller's claim without hashing the byte span.
    #[must_use]
    pub fn checksum64(&self) -> u64 {
        match self.digest_provenance {
            DigestProvenance::Verified => terrain_store_checksum64(self.bytes.as_ref()),
            DigestProvenance::Attested => self
                .attested_checksum64
                .expect("attested terrain reader carries a checksum"),
        }
    }

    /// Re-verify tile payloads and any caller-attested full-store checksum.
    ///
    /// Successful verification changes the digest provenance to
    /// [`DigestProvenance::Verified`].
    pub fn verify(&mut self) -> core::result::Result<(), TerrainStoreError> {
        parse_store(self.bytes.as_ref(), ChecksumValidation::Verified)?;
        if let Some(expected) = self.attested_checksum64 {
            let found = terrain_store_checksum64(self.bytes.as_ref());
            if expected != found {
                return Err(TerrainStoreError::AttestedChecksumMismatch { expected, found });
            }
        }
        self.digest_provenance = DigestProvenance::Verified;
        self.attested_checksum64 = None;
        Ok(())
    }

    /// Re-serialize this parsed terrain store into canonical bytes.
    ///
    /// A store accepted by [`Self::from_bytes`] is already canonical, so this
    /// returns bytes identical to [`Self::as_bytes`].
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let pending = self
            .tiles
            .iter()
            .map(|tile| PendingTile {
                lat_index: tile.index.lat_index,
                lon_index: tile.index.lon_index,
                min_latitude_deg: tile.index.min_latitude_deg,
                min_longitude_deg: tile.index.min_longitude_deg,
                max_latitude_deg: tile.index.max_latitude_deg,
                max_longitude_deg: tile.index.max_longitude_deg,
                lon_count: tile.index.lon_count,
                lat_count: tile.index.lat_count,
                data: self.tile_payload(tile).to_vec(),
                vertical_datum: tile.index.vertical_datum,
            })
            .collect();
        build_store(pending).expect("parsed terrain store can be reserialized")
    }

    /// Return the bilinearly interpolated orthometric height `H` in metres at a
    /// longitude-first geodetic position in degrees.
    pub fn height_m(&mut self, longitude_deg: f64, latitude_deg: f64) -> Result<f64> {
        self.height_m_with_options(longitude_deg, latitude_deg, DtedLookupOptions::default())
    }

    /// Return the orthometric height `H` in metres at a longitude-first geodetic
    /// position in degrees using explicit lookup options.
    pub fn height_m_with_options(
        &mut self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: DtedLookupOptions,
    ) -> Result<f64> {
        self.orthometric_height_m_with_options(longitude_deg, latitude_deg, options)
            .map(OrthometricHeightM::metres)
    }

    /// Return the bilinearly interpolated orthometric height `H` in metres as a
    /// typed value at a longitude-first geodetic position in degrees.
    pub fn orthometric_height_m(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> Result<OrthometricHeightM> {
        self.orthometric_height_m_with_options(
            longitude_deg,
            latitude_deg,
            DtedLookupOptions::default(),
        )
    }

    /// Return the orthometric height `H` in metres as a typed value at a
    /// longitude-first geodetic position in degrees using explicit lookup
    /// options.
    pub fn orthometric_height_m_with_options(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: DtedLookupOptions,
    ) -> Result<OrthometricHeightM> {
        validate_lookup_coordinates(longitude_deg, latitude_deg)?;
        self.height_from_candidates(longitude_deg, latitude_deg, options)
            .1
            .map(OrthometricHeightM::new)
    }

    /// Evaluate `(longitude_deg, latitude_deg)` points in order as orthometric
    /// heights `H` in metres.
    ///
    /// The tuple order is longitude-first, matching [`Self::height_m`]. Each
    /// output element is independent, so an invalid point or parse failure is
    /// returned only for that element.
    pub fn height_batch(
        &mut self,
        points: &[(f64, f64)],
        options: DtedLookupOptions,
    ) -> Vec<Result<f64>> {
        self.orthometric_height_batch(points, options)
            .into_iter()
            .map(|result| result.map(OrthometricHeightM::metres))
            .collect()
    }

    /// Evaluate `(longitude_deg, latitude_deg)` points in order as typed
    /// orthometric heights `H` in metres.
    ///
    /// The tuple order is longitude-first. Each output element is independent,
    /// so an invalid point or parse failure is returned only for that element.
    pub fn orthometric_height_batch(
        &self,
        points: &[(f64, f64)],
        options: DtedLookupOptions,
    ) -> Vec<Result<OrthometricHeightM>> {
        let mut out = Vec::with_capacity(points.len());
        let mut current = None;

        for &(longitude_deg, latitude_deg) in points {
            if let Err(err) = validate_lookup_coordinates(longitude_deg, latitude_deg) {
                out.push(Err(err));
                continue;
            }

            // The primary grid is always the first candidate, so its tile
            // answers unless it gives the point an unknown elevation.
            let primary_grid = terrain::terrain_grid(longitude_deg, latitude_deg);
            if current == Some(primary_grid) {
                if let Some(&tile_idx) = self.by_grid.get(&primary_grid) {
                    let tile = &self.tiles[tile_idx];
                    if tile.contains(longitude_deg, latitude_deg) {
                        let result = height_from_tile(
                            self.bytes.as_ref(),
                            tile,
                            longitude_deg,
                            latitude_deg,
                            options,
                        );
                        if !matches!(result, Err(Error::UnknownTerrainElevation { .. })) {
                            out.push(result.map(OrthometricHeightM::new));
                            continue;
                        }
                    }
                }
            }

            let (grid, result) = self.height_from_candidates(longitude_deg, latitude_deg, options);
            current = grid;
            out.push(result.map(OrthometricHeightM::new));
        }

        out
    }

    /// Return ellipsoidal height `h` in metres using the embedded EGM96
    /// 1-degree grid for `h = H + N`.
    ///
    /// The input position is terrain order `(longitude_deg, latitude_deg)`.
    /// Internally, the geoid call is made with `(latitude_deg, longitude_deg)`.
    /// The embedded EGM96 1-degree grid agrees with the full EGM96
    /// 15-arcminute grid to about 0.4 m RMS.
    pub fn ellipsoidal_height_m(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> core::result::Result<EllipsoidalHeightM, TerrainDatumError> {
        self.ellipsoidal_height_m_with_options(
            longitude_deg,
            latitude_deg,
            DtedLookupOptions::default(),
        )
    }

    /// Return ellipsoidal height `h` in metres using the embedded EGM96
    /// 1-degree grid for `h = H + N` and explicit terrain lookup options.
    ///
    /// The input position is terrain order `(longitude_deg, latitude_deg)`.
    /// Internally, the geoid call is made with `(latitude_deg, longitude_deg)`.
    pub fn ellipsoidal_height_m_with_options(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: DtedLookupOptions,
    ) -> core::result::Result<EllipsoidalHeightM, TerrainDatumError> {
        self.ellipsoidal_height_m_with_model(
            longitude_deg,
            latitude_deg,
            options,
            TerrainGeoidModel::Egm96OneDegree,
        )
    }

    /// Return ellipsoidal height `h` in metres using an explicit geoid tier for
    /// `h = H + N`.
    ///
    /// The input position is terrain order `(longitude_deg, latitude_deg)`.
    /// Internally, the geoid call is made with `(latitude_deg, longitude_deg)`.
    /// Choosing [`TerrainGeoidModel::Egm96FifteenMinute`] requires a loaded
    /// `WW15MGH.DAC` grid and never falls back to the embedded EGM96 1-degree
    /// grid.
    pub fn ellipsoidal_height_m_with_model(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: DtedLookupOptions,
        geoid: TerrainGeoidModel<'_>,
    ) -> core::result::Result<EllipsoidalHeightM, TerrainDatumError> {
        let orthometric = self
            .orthometric_height_m_with_options(longitude_deg, latitude_deg, options)
            .map_err(TerrainDatumError::Terrain)?;
        orthometric.to_ellipsoidal_height_deg(latitude_deg, longitude_deg, geoid)
    }

    /// Answer a query from the tiles that contain it, in candidate order.
    ///
    /// Returns the first containing tile's grid, if any, with the result.
    /// When the first tile gives the point an unknown elevation, a nearest
    /// lookup takes a neighbouring tile's posting at exactly the coordinates
    /// of the null edge posting, and a bilinear lookup defers to the next
    /// containing tile, which exists only for a point on a tile edge, where
    /// every weighted posting lies on that shared edge. The first unknown is
    /// returned when no neighbour knows the height.
    fn height_from_candidates(
        &self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: DtedLookupOptions,
    ) -> (Option<(i32, i32)>, Result<f64>) {
        let mut first_grid = None;
        let mut unknown = None;
        for grid_idx in terrain_grid_candidates(longitude_deg, latitude_deg) {
            let Some(&tile_idx) = self.by_grid.get(&grid_idx) else {
                continue;
            };
            let tile = &self.tiles[tile_idx];
            if !tile.contains(longitude_deg, latitude_deg) {
                continue;
            }
            if first_grid.is_none() {
                first_grid = Some(grid_idx);
            }
            let result = height_from_tile(
                self.bytes.as_ref(),
                tile,
                longitude_deg,
                latitude_deg,
                options,
            );
            let unknown_at = result.as_ref().err().and_then(terrain::unknown_posting);
            let Some((lon_posting, lat_posting)) = unknown_at else {
                // A later candidate is consulted only because an earlier one
                // gave an unknown elevation; its failure leaves that unknown
                // standing and the remaining candidates still get a turn.
                if unknown.is_some() && result.is_err() {
                    continue;
                }
                return (first_grid, result);
            };
            if options.interpolation == DtedInterpolation::NearestPosting {
                let neighbour = self.nearest_from_edge_neighbours(tile, lon_posting, lat_posting);
                return (first_grid, neighbour.map_or(result, Ok));
            }
            if unknown.is_none() {
                unknown = result.err();
            }
        }
        let missing = || missing_terrain_tile(longitude_deg, latitude_deg);
        (first_grid, Err(unknown.unwrap_or_else(missing)))
    }

    /// Value of a neighbouring tile's posting at exactly the coordinates of
    /// the null posting `(lon_posting, lat_posting)` of `tile`, when that
    /// posting lies on the tile edge and a neighbour holds a known height
    /// there. Matches the raw DTED reader.
    fn nearest_from_edge_neighbours(
        &self,
        tile: &MmapTile,
        lon_posting: usize,
        lat_posting: usize,
    ) -> Option<f64> {
        let primary = tile.grid();
        for neighbour_grid in terrain::edge_neighbour_grids(primary, lon_posting, lat_posting) {
            let Some(&tile_idx) = self.by_grid.get(&neighbour_grid) else {
                continue;
            };
            let neighbour = &self.tiles[tile_idx];
            let Some((lon_index, lat_index)) =
                terrain::coincident_posting(primary, lon_posting, lat_posting, neighbour.grid())
            else {
                continue;
            };
            if let Ok(value) = neighbour.posting(self.bytes.as_ref(), lon_index, lat_index) {
                return Some(f64::from(value));
            }
        }
        None
    }

    fn tile_payload(&self, tile: &MmapTile) -> &[u8] {
        let start = tile.index.data_offset as usize;
        let end = start + tile.index.data_len as usize;
        &self.bytes.as_slice()[start..end]
    }
}

/// Convert a DTED tile tree into canonical memory-mappable terrain store bytes.
///
/// Input `.dt2` files are discovered recursively below `root`, following
/// symlinked directories and symlinked `.dt2` files. A symlinked file is treated
/// as DTED when either the link name or the resolved target name ends in
/// `.dt2`. Tiles are then sorted by integer tile id. DTED signed-magnitude
/// postings are decoded once into `i16` orthometric metres, matching the lazy
/// DTED reader. DTED null postings, including the SRTM voids that
/// [`crate::data::hgt_to_dted`] writes as nulls, are stored as
/// [`TERRAIN_STORE_NULL_POSTING`] and read back as unknown elevations. A tile
/// whose DSI states a horizontal datum other than WGS84 is refused with
/// [`TerrainStoreError::NonWgs84Tile`].
pub fn dted_tree_to_mmap_store(
    root: impl AsRef<Path>,
) -> core::result::Result<Vec<u8>, TerrainStoreError> {
    let root = root.as_ref();
    let mut paths = Vec::new();
    collect_dted_tile_paths(root, &mut paths)?;
    dted_paths_to_mmap_store(paths)
}

/// Convert an explicit DTED tile list into canonical memory-mappable terrain
/// store bytes.
///
/// Each entry supplies the expected integer tile id and the DTED `.dt2` path.
/// The converter parses each DTED header and fails if the file origin does not
/// match the supplied id. The decoded payload, sorting, duplicate detection,
/// alignment, and checksums are the same as [`dted_tree_to_mmap_store`].
pub fn dted_tile_list_to_mmap_store(
    entries: &[DtedTileListEntry],
) -> core::result::Result<Vec<u8>, TerrainStoreError> {
    let mut pending = Vec::with_capacity(entries.len());
    for entry in entries {
        pending.push(pending_tile_from_dted_path(
            &entry.path,
            Some(entry.tile_id),
        )?);
    }
    build_store(pending)
}

/// Convert a DTED tile tree and write canonical memory-mappable terrain store
/// bytes to `output_path`.
///
/// Symlinked directories and symlinked `.dt2` files below `root` are followed.
/// A symlinked file is treated as DTED when either the link name or the resolved
/// target name ends in `.dt2`.
pub fn write_dted_tree_to_mmap_store(
    root: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> core::result::Result<(), TerrainStoreError> {
    let bytes = dted_tree_to_mmap_store(root)?;
    let output_path = output_path.as_ref();
    fs::write(output_path, &bytes).map_err(|err| TerrainStoreError::Io {
        path: output_path.to_path_buf(),
        message: err.to_string(),
    })
}

/// Convert an explicit DTED tile list and write canonical memory-mappable
/// terrain store bytes to `output_path`.
pub fn write_dted_tile_list_to_mmap_store(
    entries: &[DtedTileListEntry],
    output_path: impl AsRef<Path>,
) -> core::result::Result<(), TerrainStoreError> {
    let bytes = dted_tile_list_to_mmap_store(entries)?;
    let output_path = output_path.as_ref();
    fs::write(output_path, &bytes).map_err(|err| TerrainStoreError::Io {
        path: output_path.to_path_buf(),
        message: err.to_string(),
    })
}

/// Return an FNV-1a checksum for terrain store bytes.
///
/// This checksum is for deterministic local verification and is not a
/// cryptographic digest.
#[must_use]
pub fn terrain_store_checksum64(bytes: &[u8]) -> u64 {
    fnv1a64(bytes)
}

#[derive(Debug)]
struct PendingTile {
    lat_index: i32,
    lon_index: i32,
    min_latitude_deg: f64,
    min_longitude_deg: f64,
    max_latitude_deg: f64,
    max_longitude_deg: f64,
    lon_count: u32,
    lat_count: u32,
    data: Vec<u8>,
    vertical_datum: VerticalDatum,
}

#[derive(Debug)]
struct ParsedStore {
    vertical_datum: VerticalDatum,
    tiles: Vec<MmapTile>,
    by_grid: HashMap<(i32, i32), usize>,
    tile_index: Vec<TerrainStoreTileIndex>,
    tile_ids: Vec<TerrainTileId>,
}

fn height_from_tile(
    bytes: &[u8],
    tile: &MmapTile,
    longitude_deg: f64,
    latitude_deg: f64,
    options: DtedLookupOptions,
) -> Result<f64> {
    if options.interpolation == DtedInterpolation::NearestPosting {
        return tile
            .get_elevation(bytes, longitude_deg, latitude_deg)
            .map(|v| v as f64);
    }

    let postings_per_deg_lon = tile.index.lon_count as usize - 1;
    let postings_per_deg_lat = tile.index.lat_count as usize - 1;

    let lon = terrain::in_tile_cell_fraction(
        longitude_deg,
        tile.index.min_longitude_deg,
        postings_per_deg_lon,
    );
    let lat = terrain::in_tile_cell_fraction(
        latitude_deg,
        tile.index.min_latitude_deg,
        postings_per_deg_lat,
    );
    let lon_lo = lon.cell;
    let lat_lo = lat.cell;
    let fx = lon.fraction;
    let fy = lat.fraction;

    let mut z = 0.0;
    for (di, wx) in [(0i64, 1.0 - fx), (1i64, fx)] {
        for (dj, wy) in [(0i64, 1.0 - fy), (1i64, fy)] {
            let w = wx * wy;
            if w == 0.0 {
                continue;
            }
            let posting_lon =
                tile.index.min_longitude_deg + (lon_lo + di) as f64 / postings_per_deg_lon as f64;
            let posting_lat =
                tile.index.min_latitude_deg + (lat_lo + dj) as f64 / postings_per_deg_lat as f64;
            z += w * f64::from(tile.get_elevation(bytes, posting_lon, posting_lat)?);
        }
    }
    Ok(z)
}

fn dted_paths_to_mmap_store(
    mut paths: Vec<PathBuf>,
) -> core::result::Result<Vec<u8>, TerrainStoreError> {
    paths.sort();

    let mut pending = Vec::with_capacity(paths.len());
    for path in paths {
        pending.push(pending_tile_from_dted_path(&path, None)?);
    }
    build_store(pending)
}

fn pending_tile_from_dted_path(
    path: &Path,
    expected_id: Option<TerrainTileId>,
) -> core::result::Result<PendingTile, TerrainStoreError> {
    let tile = DtedTile::from_path(path).map_err(|reason| TerrainStoreError::Parse {
        reason: format!("{}: {reason}", path.display()),
    })?;
    if !tile.horizontal_datum().is_wgs84_compatible() {
        return Err(TerrainStoreError::NonWgs84Tile {
            path: path.to_path_buf(),
            datum: tile.horizontal_datum().clone(),
        });
    }
    let decoded = tile
        .decoded_postings_lon_major()
        .map_err(|reason| TerrainStoreError::Parse {
            reason: format!("{}: {reason}", path.display()),
        })?;
    let mut data = Vec::with_capacity(decoded.len() * 2);
    for posting in decoded {
        let stored = posting.unwrap_or(TERRAIN_STORE_NULL_POSTING);
        data.extend_from_slice(&stored.to_le_bytes());
    }
    let lat_index = tile.origin_latitude().floor() as i32;
    let lon_index = tile.origin_longitude().floor() as i32;
    let found = TerrainTileId::new(lat_index, lon_index);
    if let Some(expected) = expected_id {
        if expected != found {
            return Err(TerrainStoreError::TileIdMismatch {
                path: path.to_path_buf(),
                expected,
                found,
            });
        }
    }
    Ok(PendingTile {
        lat_index,
        lon_index,
        min_latitude_deg: tile.origin_latitude(),
        min_longitude_deg: tile.origin_longitude(),
        max_latitude_deg: tile.origin_latitude() + 1.0,
        max_longitude_deg: tile.origin_longitude() + 1.0,
        lon_count: u32::try_from(tile.lon_count()).map_err(|_| TerrainStoreError::Parse {
            reason: format!("{} longitude count exceeds u32", path.display()),
        })?,
        lat_count: u32::try_from(tile.lat_count()).map_err(|_| TerrainStoreError::Parse {
            reason: format!("{} latitude count exceeds u32", path.display()),
        })?,
        data,
        vertical_datum: VerticalDatum::Egm96MslOrthometric,
    })
}

fn collect_dted_tile_paths(
    root: &Path,
    out: &mut Vec<PathBuf>,
) -> core::result::Result<(), TerrainStoreError> {
    let mut visited_dirs = HashSet::new();
    collect_dted_tile_paths_inner(root, out, &mut visited_dirs)
}

fn collect_dted_tile_paths_inner(
    path: &Path,
    out: &mut Vec<PathBuf>,
    visited_dirs: &mut HashSet<PathBuf>,
) -> core::result::Result<(), TerrainStoreError> {
    let metadata = fs::metadata(path).map_err(|err| TerrainStoreError::Io {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;

    if metadata.is_dir() {
        let canonical = fs::canonicalize(path).map_err(|err| TerrainStoreError::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;
        if !visited_dirs.insert(canonical) {
            return Ok(());
        }
        let entries = fs::read_dir(path).map_err(|err| TerrainStoreError::Io {
            path: path.to_path_buf(),
            message: err.to_string(),
        })?;
        for entry in entries {
            let entry = entry.map_err(|err| TerrainStoreError::Io {
                path: path.to_path_buf(),
                message: err.to_string(),
            })?;
            collect_dted_tile_paths_inner(&entry.path(), out, visited_dirs)?;
        }
    } else if metadata.is_file() && is_dted_tile_source(path)? {
        out.push(path.to_path_buf());
    }
    Ok(())
}

fn is_dted_tile_source(path: &Path) -> core::result::Result<bool, TerrainStoreError> {
    if is_dted_tile_path(path) {
        return Ok(true);
    }

    let canonical = fs::canonicalize(path).map_err(|err| TerrainStoreError::Io {
        path: path.to_path_buf(),
        message: err.to_string(),
    })?;
    Ok(is_dted_tile_path(&canonical))
}

fn is_dted_tile_path(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| name.ends_with(".dt2"))
}

fn parse_store(
    bytes: &[u8],
    checksum_validation: ChecksumValidation,
) -> core::result::Result<ParsedStore, TerrainStoreError> {
    if bytes.len() < STORE_HEADER_LEN {
        return Err(TerrainStoreError::Parse {
            reason: format!(
                "store has {} bytes but needs at least {STORE_HEADER_LEN}",
                bytes.len()
            ),
        });
    }
    if &bytes[..STORE_MAGIC.len()] != STORE_MAGIC {
        return Err(TerrainStoreError::Parse {
            reason: "missing terrain store magic".to_string(),
        });
    }
    let version = read_u16(bytes, HEADER_VERSION_OFFSET)?;
    if version != STORE_VERSION {
        return Err(TerrainStoreError::UnsupportedVersion { version });
    }
    ensure_zero(bytes, 11, 12, "header reserved byte")?;
    ensure_zero(bytes, 40, STORE_HEADER_LEN, "header reserved bytes")?;

    let vertical_datum = VerticalDatum::from_tag(bytes[HEADER_DATUM_OFFSET])?;
    let tile_count = wire_usize(
        u64::from(read_u32(bytes, HEADER_TILE_COUNT_OFFSET)?),
        "tile count",
    )?;
    let index_offset = wire_usize(read_u64(bytes, HEADER_INDEX_OFFSET_OFFSET)?, "index offset")?;
    let data_offset = wire_usize(read_u64(bytes, HEADER_DATA_OFFSET_OFFSET)?, "data offset")?;
    let total_len = wire_usize(read_u64(bytes, HEADER_TOTAL_LEN_OFFSET)?, "total length")?;
    if total_len != bytes.len() {
        return Err(TerrainStoreError::Parse {
            reason: format!(
                "header total length {total_len} does not match {}",
                bytes.len()
            ),
        });
    }
    if index_offset != STORE_HEADER_LEN {
        return Err(TerrainStoreError::Parse {
            reason: format!("index offset must be {STORE_HEADER_LEN}, got {index_offset}"),
        });
    }

    let index_len = tile_count
        .checked_mul(STORE_INDEX_RECORD_LEN)
        .ok_or_else(|| TerrainStoreError::Parse {
            reason: "tile index length overflows usize".to_string(),
        })?;
    let index_end =
        index_offset
            .checked_add(index_len)
            .ok_or_else(|| TerrainStoreError::Parse {
                reason: "tile index end overflows usize".to_string(),
            })?;
    if index_end > bytes.len() {
        return Err(TerrainStoreError::Parse {
            reason: "tile index extends past store length".to_string(),
        });
    }
    let expected_data_offset = align_up(index_end, STORE_ALIGNMENT)?;
    if data_offset != expected_data_offset {
        return Err(TerrainStoreError::Parse {
            reason: format!("data offset must be {expected_data_offset}, got {data_offset}"),
        });
    }
    ensure_zero(bytes, index_end, data_offset, "index padding")?;

    let mut tiles = Vec::with_capacity(tile_count);
    let mut tile_index = Vec::with_capacity(tile_count);
    let mut tile_ids = Vec::with_capacity(tile_count);
    let mut by_grid = HashMap::with_capacity(tile_count);
    let mut previous_id = None;
    let mut expected_next = data_offset;

    for idx in 0..tile_count {
        let record_offset = index_offset + idx * STORE_INDEX_RECORD_LEN;
        let record = &bytes[record_offset..record_offset + STORE_INDEX_RECORD_LEN];
        let lat_index = read_i32(record, INDEX_LAT_OFFSET)?;
        let lon_index = read_i32(record, INDEX_LON_OFFSET)?;
        if !(MIN_TILE_LAT_INDEX..=MAX_TILE_LAT_INDEX).contains(&lat_index)
            || !(MIN_TILE_LON_INDEX..=MAX_TILE_LON_INDEX).contains(&lon_index)
        {
            return Err(TerrainStoreError::TileIdOutOfRange {
                lat_index,
                lon_index,
            });
        }
        let tile_id = (lat_index, lon_index);
        if previous_id.is_some_and(|previous| tile_id <= previous) {
            return Err(TerrainStoreError::Parse {
                reason: "tile index records are not strictly sorted".to_string(),
            });
        }
        previous_id = Some(tile_id);

        let lon_count = read_u32(record, INDEX_LON_COUNT_OFFSET)?;
        let lat_count = read_u32(record, INDEX_LAT_COUNT_OFFSET)?;
        if lon_count < 2 || lat_count < 2 {
            return Err(TerrainStoreError::Parse {
                reason: format!(
                    "tile ({lat_index},{lon_index}) has invalid dimensions lon_count={lon_count} lat_count={lat_count}"
                ),
            });
        }
        let offset = wire_usize(
            read_u64(record, INDEX_DATA_OFFSET_OFFSET)?,
            "tile data offset",
        )?;
        let data_len = wire_usize(read_u64(record, INDEX_DATA_LEN_OFFSET)?, "tile data length")?;
        let expected_len = (lon_count as usize)
            .checked_mul(lat_count as usize)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| TerrainStoreError::Parse {
                reason: format!("tile ({lat_index},{lon_index}) data length overflows usize"),
            })?;
        if data_len != expected_len {
            return Err(TerrainStoreError::Parse {
                reason: format!(
                    "tile ({lat_index},{lon_index}) data length must be {expected_len}, got {data_len}"
                ),
            });
        }

        let expected_offset = align_up(expected_next, STORE_ALIGNMENT)?;
        ensure_zero(bytes, expected_next, expected_offset, "tile padding")?;
        if offset != expected_offset {
            return Err(TerrainStoreError::Parse {
                reason: format!(
                    "tile ({lat_index},{lon_index}) data offset must be {expected_offset}, got {offset}"
                ),
            });
        }
        let end = offset
            .checked_add(data_len)
            .ok_or_else(|| TerrainStoreError::Parse {
                reason: format!("tile ({lat_index},{lon_index}) data end overflows usize"),
            })?;
        if end > bytes.len() {
            return Err(TerrainStoreError::Parse {
                reason: format!("tile ({lat_index},{lon_index}) data extends past store length"),
            });
        }

        let checksum64 = read_u64(record, INDEX_CHECKSUM_OFFSET)?;
        if checksum_validation.verifies_payloads() {
            let found = fnv1a64(&bytes[offset..end]);
            if found != checksum64 {
                return Err(TerrainStoreError::Checksum {
                    lat_index,
                    lon_index,
                    expected: checksum64,
                    found,
                });
            }
        }

        let min_latitude_deg = read_f64(record, INDEX_MIN_LAT_OFFSET)?;
        let min_longitude_deg = read_f64(record, INDEX_MIN_LON_OFFSET)?;
        let max_latitude_deg = read_f64(record, INDEX_MAX_LAT_OFFSET)?;
        let max_longitude_deg = read_f64(record, INDEX_MAX_LON_OFFSET)?;
        let south = f64::from(lat_index);
        let west = f64::from(lon_index);
        for (field, value, edge) in [
            ("min_latitude_deg", min_latitude_deg, south),
            ("min_longitude_deg", min_longitude_deg, west),
            ("max_latitude_deg", max_latitude_deg, south + 1.0),
            ("max_longitude_deg", max_longitude_deg, west + 1.0),
        ] {
            if !value.is_finite() {
                return Err(TerrainStoreError::Parse {
                    reason: format!("tile ({lat_index},{lon_index}) {field} is not finite"),
                });
            }
            // Exact equality: the cell edges are small integers, and a bound
            // off by any amount would place postings where the payload does
            // not hold them. A signed zero equals zero.
            if value != edge {
                return Err(TerrainStoreError::TileBoundsMismatch {
                    lat_index,
                    lon_index,
                    field,
                });
            }
        }
        let tile_datum = VerticalDatum::from_tag(record[INDEX_DATUM_OFFSET])?;
        if tile_datum != vertical_datum {
            return Err(TerrainStoreError::Parse {
                reason: format!("tile ({lat_index},{lon_index}) datum differs from header"),
            });
        }
        ensure_zero(
            record,
            INDEX_DATUM_OFFSET + 1,
            STORE_INDEX_RECORD_LEN,
            "tile index reserved bytes",
        )?;

        let index = TerrainStoreTileIndex {
            lat_index,
            lon_index,
            min_longitude_deg,
            min_latitude_deg,
            max_longitude_deg,
            max_latitude_deg,
            lon_count,
            lat_count,
            data_offset: offset as u64,
            data_len: data_len as u64,
            checksum64,
            vertical_datum: tile_datum,
        };
        by_grid.insert(tile_id, tiles.len());
        tiles.push(MmapTile { index });
        tile_index.push(index);
        tile_ids.push(TerrainTileId::new(lat_index, lon_index));
        expected_next = end;
    }

    if expected_next != bytes.len() {
        return Err(TerrainStoreError::Parse {
            reason: format!(
                "store has trailing bytes: expected length {expected_next}, got {}",
                bytes.len()
            ),
        });
    }

    Ok(ParsedStore {
        vertical_datum,
        tiles,
        by_grid,
        tile_index,
        tile_ids,
    })
}

/// Convert a wire integer to `usize`, refusing a value this target cannot
/// address instead of truncating it.
fn wire_usize(value: u64, field: &str) -> core::result::Result<usize, TerrainStoreError> {
    usize::try_from(value).map_err(|_| TerrainStoreError::Parse {
        reason: format!("{field} {value} does not fit this target's address width"),
    })
}

fn missing_terrain_tile(longitude_deg: f64, latitude_deg: f64) -> Error {
    let (lat_index, lon_index) = terrain::terrain_grid(longitude_deg, latitude_deg);
    Error::MissingTerrainTile {
        lat_index,
        lon_index,
    }
}

fn build_store(mut tiles: Vec<PendingTile>) -> core::result::Result<Vec<u8>, TerrainStoreError> {
    tiles.sort_by_key(|tile| (tile.lat_index, tile.lon_index));
    for pair in tiles.windows(2) {
        if (pair[0].lat_index, pair[0].lon_index) == (pair[1].lat_index, pair[1].lon_index) {
            return Err(TerrainStoreError::DuplicateTile {
                lat_index: pair[0].lat_index,
                lon_index: pair[0].lon_index,
            });
        }
    }

    let index_end = STORE_HEADER_LEN
        .checked_add(
            tiles
                .len()
                .checked_mul(STORE_INDEX_RECORD_LEN)
                .ok_or_else(|| TerrainStoreError::Parse {
                    reason: "tile index length overflows usize".to_string(),
                })?,
        )
        .ok_or_else(|| TerrainStoreError::Parse {
            reason: "tile index end overflows usize".to_string(),
        })?;
    let data_offset = align_up(index_end, STORE_ALIGNMENT)?;
    let mut offsets = Vec::with_capacity(tiles.len());
    let mut cursor = data_offset;
    for tile in &tiles {
        cursor = align_up(cursor, STORE_ALIGNMENT)?;
        offsets.push(cursor);
        cursor = cursor
            .checked_add(tile.data.len())
            .ok_or_else(|| TerrainStoreError::Parse {
                reason: "store length overflows usize".to_string(),
            })?;
    }

    let mut out = vec![0u8; cursor];
    out[..STORE_MAGIC.len()].copy_from_slice(STORE_MAGIC);
    write_u16(&mut out, HEADER_VERSION_OFFSET, STORE_VERSION);
    out[HEADER_DATUM_OFFSET] = VerticalDatum::Egm96MslOrthometric.tag();
    write_u32(
        &mut out,
        HEADER_TILE_COUNT_OFFSET,
        u32::try_from(tiles.len()).map_err(|_| TerrainStoreError::Parse {
            reason: "tile count exceeds u32".to_string(),
        })?,
    );
    write_u64(
        &mut out,
        HEADER_INDEX_OFFSET_OFFSET,
        STORE_HEADER_LEN as u64,
    );
    write_u64(&mut out, HEADER_DATA_OFFSET_OFFSET, data_offset as u64);
    write_u64(&mut out, HEADER_TOTAL_LEN_OFFSET, cursor as u64);

    for (idx, tile) in tiles.iter().enumerate() {
        let record_offset = STORE_HEADER_LEN + idx * STORE_INDEX_RECORD_LEN;
        let offset = offsets[idx];
        let data_len = tile.data.len();
        let expected_len = (tile.lon_count as usize)
            .checked_mul(tile.lat_count as usize)
            .and_then(|count| count.checked_mul(2))
            .ok_or_else(|| TerrainStoreError::Parse {
                reason: format!(
                    "tile ({},{}) data length overflows usize",
                    tile.lat_index, tile.lon_index
                ),
            })?;
        if data_len != expected_len {
            return Err(TerrainStoreError::Parse {
                reason: format!(
                    "tile ({},{}) data length must be {expected_len}, got {data_len}",
                    tile.lat_index, tile.lon_index
                ),
            });
        }

        let record = &mut out[record_offset..record_offset + STORE_INDEX_RECORD_LEN];
        write_i32(record, INDEX_LAT_OFFSET, tile.lat_index);
        write_i32(record, INDEX_LON_OFFSET, tile.lon_index);
        write_u32(record, INDEX_LON_COUNT_OFFSET, tile.lon_count);
        write_u32(record, INDEX_LAT_COUNT_OFFSET, tile.lat_count);
        write_u64(record, INDEX_DATA_OFFSET_OFFSET, offset as u64);
        write_u64(record, INDEX_DATA_LEN_OFFSET, data_len as u64);
        write_u64(record, INDEX_CHECKSUM_OFFSET, fnv1a64(&tile.data));
        write_f64(record, INDEX_MIN_LAT_OFFSET, tile.min_latitude_deg);
        write_f64(record, INDEX_MIN_LON_OFFSET, tile.min_longitude_deg);
        write_f64(record, INDEX_MAX_LAT_OFFSET, tile.max_latitude_deg);
        write_f64(record, INDEX_MAX_LON_OFFSET, tile.max_longitude_deg);
        record[INDEX_DATUM_OFFSET] = tile.vertical_datum.tag();
        out[offset..offset + data_len].copy_from_slice(&tile.data);
    }

    Ok(out)
}

fn align_up(value: usize, alignment: usize) -> core::result::Result<usize, TerrainStoreError> {
    let rem = value % alignment;
    if rem == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - rem)
            .ok_or_else(|| TerrainStoreError::Parse {
                reason: "aligned offset overflows usize".to_string(),
            })
    }
}

fn ensure_zero(
    bytes: &[u8],
    start: usize,
    end: usize,
    context: &str,
) -> core::result::Result<(), TerrainStoreError> {
    if start > end || end > bytes.len() {
        return Err(TerrainStoreError::Parse {
            reason: format!("{context} range is out of bounds"),
        });
    }
    if bytes[start..end].iter().any(|&byte| byte != 0) {
        return Err(TerrainStoreError::Parse {
            reason: format!("{context} must be zero-filled"),
        });
    }
    Ok(())
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET_BASIS, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(FNV_PRIME)
    })
}

fn read_u16(bytes: &[u8], offset: usize) -> core::result::Result<u16, TerrainStoreError> {
    Ok(u16::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u32(bytes: &[u8], offset: usize) -> core::result::Result<u32, TerrainStoreError> {
    Ok(u32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_i32(bytes: &[u8], offset: usize) -> core::result::Result<i32, TerrainStoreError> {
    Ok(i32::from_le_bytes(read_array(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: usize) -> core::result::Result<u64, TerrainStoreError> {
    Ok(u64::from_le_bytes(read_array(bytes, offset)?))
}

fn read_f64(bytes: &[u8], offset: usize) -> core::result::Result<f64, TerrainStoreError> {
    Ok(f64::from_le_bytes(read_array(bytes, offset)?))
}

fn read_array<const N: usize>(
    bytes: &[u8],
    offset: usize,
) -> core::result::Result<[u8; N], TerrainStoreError> {
    let end = offset
        .checked_add(N)
        .ok_or_else(|| TerrainStoreError::Parse {
            reason: "numeric field offset overflows usize".to_string(),
        })?;
    let slice = bytes
        .get(offset..end)
        .ok_or_else(|| TerrainStoreError::Parse {
            reason: "numeric field extends past record".to_string(),
        })?;
    slice.try_into().map_err(|_| TerrainStoreError::Parse {
        reason: "numeric field has wrong length".to_string(),
    })
}

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_i32(bytes: &mut [u8], offset: usize, value: i32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_f64(bytes: &mut [u8], offset: usize, value: f64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

#[cfg(test)]
mod wire_width_tests {
    use super::{wire_usize, TerrainStoreError};

    /// A wire value above the target's address width is refused by name. On a
    /// 64-bit target every `u64` fits, so the refusal is exercised where it
    /// can occur and the exact conversion everywhere else.
    #[test]
    fn wire_integers_convert_exactly_or_are_refused() {
        let above_u32 = (1u64 << 32) + 4146;
        match usize::try_from(above_u32) {
            Ok(expected) => assert_eq!(wire_usize(above_u32, "total length"), Ok(expected)),
            Err(_) => assert_eq!(
                wire_usize(above_u32, "total length"),
                Err(TerrainStoreError::Parse {
                    reason: format!(
                        "total length {above_u32} does not fit this target's address width"
                    ),
                })
            ),
        }
        assert_eq!(wire_usize(4146, "total length"), Ok(4146));
    }
}
