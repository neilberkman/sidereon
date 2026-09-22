//! DTED tile reader and bilinear terrain lookup.
//!
//! Tiles are read as MIL-PRF-89020B one-degree cells with full profiles: the
//! UHL origin is a whole degree with the hemisphere letters of its axis, any
//! stated UHL data interval spans one degree over the posting count, and every
//! data record declares the longitude count of its position and latitude count
//! zero. The partial profiles of magnetic-tape cells are refused by name. A
//! posting holding the null value is an unknown elevation: a lookup that
//! weights it returns [`Error::UnknownTerrainElevation`], never a height. When
//! the posting a nearest lookup selects is a null on the tile edge, a
//! neighbouring tile's posting at exactly the same coordinates answers; a
//! bilinear query exactly on a shared edge is answered by the neighbouring tile
//! when the first gives it an unknown elevation.
//!
//! A tile keeps the horizontal datum its DSI record states
//! ([`DtedTile::horizontal_datum`]). Cells compiled on an earlier WGS, such as
//! WGS72 (MIL-PRF-89020B DSI note n), read as tiles, but [`DtedTerrain`]
//! answers WGS84 geodetic queries and refuses such a tile rather than
//! transforming it; a blank datum field is read as WGS84, as before.
//!
//! Some producers wrote negative postings in two's complement instead of
//! signed magnitude. As GDAL's DTED driver does (`dted_api.c`, citing
//! `w_069_s50.dt0`), a negative signed-magnitude value below -16000 m other
//! than the null is reinterpreted as two's complement; no conforming posting
//! lies there, since MIL-PRF-89020B 3.11.2 bounds terrain at -12,000 m.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::Error;

pub(crate) const UHL_SIZE: usize = 80;
pub(crate) const DSI_SIZE: usize = 648;
pub(crate) const ACC_SIZE: usize = 2700;
pub(crate) const DATA_OFFSET: usize = UHL_SIZE + DSI_SIZE + ACC_SIZE;
pub(crate) const DATA_SENTINEL: u8 = 0xAA;
/// Raw DTED posting with every bit set: the null (unknown) elevation of
/// MIL-PRF-89020B 3.11.3.1. Under signed magnitude it reads as -32767.
pub(crate) const DTED_NULL_POSTING_RAW: u16 = 0xFFFF;
/// Arc length of one tile side in tenths of an arc second, the unit of the UHL
/// data interval fields.
const ONE_DEGREE_TENTHS_ARCSEC: u64 = 36_000;
/// Offset of the DSI horizontal datum code (DSI character 145).
const DSI_HORIZONTAL_DATUM: std::ops::Range<usize> = UHL_SIZE + 144..UHL_SIZE + 149;
pub(crate) const DTED_SUFFIX: &str = concat!("_1arc_v3.d", "t", "2");
const MIN_LOOKUP_LATITUDE_DEG: f64 = -90.0;
const MAX_LOOKUP_LATITUDE_DEG: f64 = 90.0;
const MIN_LOOKUP_LONGITUDE_DEG: f64 = -180.0;
const MAX_LOOKUP_LONGITUDE_DEG: f64 = 180.0;

/// Error returned when a DTED tile cannot be read, validated, or queried.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum DtedTileError {
    /// The tile could not be read from disk.
    #[error("{path}: {message}")]
    Io {
        /// Display string of the path passed to [`DtedTile::from_path`].
        path: String,
        /// Text returned by the underlying filesystem read error.
        message: String,
    },
    /// The tile does not contain the fixed DTED header area.
    #[error("{path} is too short for DTED headers")]
    TooShort {
        /// Display string of the path passed to [`DtedTile::from_path`].
        path: String,
    },
    /// The tile does not start with the DTED UHL1 marker.
    #[error("{path} missing UHL1 header")]
    MissingUhl1 {
        /// Display string of the path passed to [`DtedTile::from_path`].
        path: String,
    },
    /// A fixed-width field was not valid UTF-8.
    #[error("{0}")]
    InvalidEncoding(String),
    /// A numeric field could not be parsed.
    #[error("{0}")]
    InvalidField(String),
    /// The tile dimensions are too small to define a grid cell.
    #[error(
        "{path} has invalid DTED dimensions lon_count={lon_count} lat_count={lat_count}; both must be at least 2"
    )]
    InvalidDimensions {
        /// Display string of the path passed to [`DtedTile::from_path`].
        path: String,
        /// Longitude block count parsed from header bytes `47..51`.
        lon_count: usize,
        /// Latitude posting count parsed from header bytes `51..55`.
        lat_count: usize,
    },
    /// The tile ends before its declared data blocks end.
    #[error("{path} has {actual} bytes but expected at least {expected}")]
    Truncated {
        /// Display string of the path passed to [`DtedTile::from_path`].
        path: String,
        /// Total number of bytes returned by `fs::read`.
        actual: usize,
        /// Minimum length computed from the fixed data offset and declared blocks.
        expected: usize,
    },
    /// A query is outside this tile's one-degree extent.
    #[error("point ({longitude},{latitude}) is outside DTED tile ({origin_longitude},{origin_latitude})")]
    Outside {
        /// Longitude supplied to [`DtedTile::get_elevation`].
        longitude: f64,
        /// Latitude supplied to [`DtedTile::get_elevation`].
        latitude: f64,
        /// Parsed tile origin longitude used by the containment check.
        origin_longitude: f64,
        /// Parsed tile origin latitude used by the containment check.
        origin_latitude: f64,
    },
    /// A rounded query did not map to a declared posting.
    #[error("posting index out of bounds lon={longitude_index} lat={latitude_index}")]
    PostingIndexOutOfBounds {
        /// Nearest longitude posting index computed from the query offset.
        longitude_index: usize,
        /// Nearest latitude posting index computed from the query offset.
        latitude_index: usize,
    },
    /// A DTED data block is missing its sentinel byte.
    #[error("DTED block {longitude_index} missing data sentinel")]
    MissingDataSentinel {
        /// Zero-based longitude data-block index whose first byte was not `0xAA`.
        longitude_index: usize,
    },
    /// A DTED data block checksum does not match its contents.
    #[error("DTED checksum failed for block {longitude_index}: expected {checksum}, found {sum}")]
    Checksum {
        /// Zero-based longitude data-block index whose checksum comparison failed.
        longitude_index: usize,
        /// Signed big-endian `i32` decoded from the block's final four bytes.
        checksum: i32,
        /// Signed `i32` sum of the bytes before the final four checksum bytes.
        sum: i32,
    },
    /// A coordinate field is empty.
    #[error("empty DTED coordinate")]
    EmptyCoordinate,
    /// A coordinate field has an unsupported hemisphere suffix.
    #[error("invalid DTED hemisphere {hemisphere}")]
    InvalidHemisphere {
        /// Final coordinate-field character that was not `N`, `E`, `S`, or `W`.
        hemisphere: char,
    },
    /// The rounded coordinate is negative and cannot be a posting index.
    #[error("cannot round negative posting index {index}")]
    NegativePostingIndex {
        /// Signed nearest posting index that could not convert to `usize`.
        index: i64,
    },
    /// A UHL origin field states degrees outside its axis, or minutes or
    /// seconds outside `0..60`.
    #[error("DTED {field} {text:?} is out of range")]
    CoordinateOutOfRange {
        /// Name of the UHL field.
        field: &'static str,
        /// Field text as read.
        text: String,
    },
    /// A UHL origin field carries a hemisphere letter of the other axis.
    #[error("DTED {field} has hemisphere {hemisphere}, expected {expected}")]
    WrongHemisphere {
        /// Name of the UHL field.
        field: &'static str,
        /// Hemisphere letter found in the field.
        hemisphere: char,
        /// Hemisphere letters the field allows.
        expected: &'static str,
    },
    /// A UHL origin is not a whole degree. MIL-PRF-89020B states the UHL
    /// origin as a full degree value, and every lookup assumes a tile spans
    /// exactly one degree from it.
    #[error("DTED {field} {text:?} is not a whole degree")]
    OriginNotWholeDegree {
        /// Name of the UHL field.
        field: &'static str,
        /// Field text as read.
        text: String,
    },
    /// A UHL data interval and posting count do not span one degree, so the
    /// postings are not where the reader places them.
    #[error(
        "DTED {field} of {interval_tenths_arcsec} tenths of an arc second over {count} postings does not span one degree"
    )]
    IntervalCountMismatch {
        /// Name of the UHL interval field.
        field: &'static str,
        /// Interval in tenths of an arc second.
        interval_tenths_arcsec: u32,
        /// Posting count on the same axis.
        count: usize,
    },
    /// A data record's longitude count does not match its position in the
    /// file, so its postings belong to a different meridian.
    #[error("DTED block {longitude_index} declares longitude count {declared}")]
    ProfileLongitudeCountMismatch {
        /// Zero-based position of the data record in the file.
        longitude_index: usize,
        /// Longitude count declared by the record.
        declared: i32,
    },
    /// A data record's latitude count is not zero. Only full profiles, whose
    /// first posting lies on the origin parallel, are read; the partial
    /// profiles MIL-PRF-89020B allows on magnetic tape (3.11.3.2.2) are
    /// refused rather than read at the wrong latitudes.
    #[error(
        "DTED block {longitude_index} is a partial profile starting at latitude count {first_latitude_index}"
    )]
    UnsupportedPartialProfile {
        /// Zero-based position of the data record in the file.
        longitude_index: usize,
        /// Latitude count declared by the record.
        first_latitude_index: i32,
    },
    /// The posting holds the DTED null value (all bits set, MIL-PRF-89020B
    /// 3.11.3.1), an unknown elevation rather than a height.
    #[error(
        "DTED posting lon={longitude_index} lat={latitude_index} is a null (unknown) elevation"
    )]
    NullPosting {
        /// Zero-based longitude posting (profile) index.
        longitude_index: usize,
        /// Zero-based latitude posting index.
        latitude_index: usize,
    },
}

#[cfg(test)]
mod error_display_tests {
    use super::{parse_dted_coord, DtedTileError};

    #[test]
    fn dted_error_display_preserves_parser_messages() {
        let cases = [
            (
                DtedTileError::Io {
                    path: "tile.dt2".to_string(),
                    message: "permission denied".to_string(),
                },
                "tile.dt2: permission denied",
            ),
            (
                DtedTileError::TooShort {
                    path: "tile.dt2".to_string(),
                },
                "tile.dt2 is too short for DTED headers",
            ),
            (
                DtedTileError::MissingUhl1 {
                    path: "tile.dt2".to_string(),
                },
                "tile.dt2 missing UHL1 header",
            ),
            (
                DtedTileError::InvalidEncoding("invalid utf-8".to_string()),
                "invalid utf-8",
            ),
            (
                DtedTileError::InvalidField("invalid digit".to_string()),
                "invalid digit",
            ),
            (
                DtedTileError::InvalidDimensions {
                    path: "tile.dt2".to_string(),
                    lon_count: 1,
                    lat_count: 0,
                },
                "tile.dt2 has invalid DTED dimensions lon_count=1 lat_count=0; both must be at least 2",
            ),
            (
                DtedTileError::Truncated {
                    path: "tile.dt2".to_string(),
                    actual: 10,
                    expected: 20,
                },
                "tile.dt2 has 10 bytes but expected at least 20",
            ),
            (
                DtedTileError::Outside {
                    longitude: 2.0,
                    latitude: 3.0,
                    origin_longitude: 0.0,
                    origin_latitude: 1.0,
                },
                "point (2,3) is outside DTED tile (0,1)",
            ),
            (
                DtedTileError::PostingIndexOutOfBounds {
                    longitude_index: 4,
                    latitude_index: 5,
                },
                "posting index out of bounds lon=4 lat=5",
            ),
            (
                DtedTileError::MissingDataSentinel { longitude_index: 6 },
                "DTED block 6 missing data sentinel",
            ),
            (
                DtedTileError::Checksum {
                    longitude_index: 7,
                    checksum: 8,
                    sum: 9,
                },
                "DTED checksum failed for block 7: expected 8, found 9",
            ),
            (DtedTileError::EmptyCoordinate, "empty DTED coordinate"),
            (
                DtedTileError::InvalidHemisphere { hemisphere: 'X' },
                "invalid DTED hemisphere X",
            ),
            (
                DtedTileError::NegativePostingIndex { index: -1 },
                "cannot round negative posting index -1",
            ),
            (
                DtedTileError::NullPosting {
                    longitude_index: 2,
                    latitude_index: 3,
                },
                "DTED posting lon=2 lat=3 is a null (unknown) elevation",
            ),
            (
                DtedTileError::ProfileLongitudeCountMismatch {
                    longitude_index: 1,
                    declared: 2,
                },
                "DTED block 1 declares longitude count 2",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn malformed_coordinate_returns_typed_error() {
        assert!(matches!(
            parse_dted_coord("N"),
            Err(DtedTileError::InvalidField(_))
        ));
    }
}

/// Interpolation mode for DTED terrain lookups.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtedInterpolation {
    /// Return the nearest DTED posting as an orthometric height in metres.
    NearestPosting,
    /// Bilinearly interpolate the four surrounding DTED postings as an
    /// orthometric height in metres.
    Bilinear,
}

/// Lookup options for DTED terrain queries.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct DtedLookupOptions {
    /// Interpolation mode used for each orthometric height query.
    pub interpolation: DtedInterpolation,
}

impl Default for DtedLookupOptions {
    fn default() -> Self {
        Self {
            interpolation: DtedInterpolation::Bilinear,
        }
    }
}

/// Lazy DTED terrain reader backed by raw `.dt2` tile bytes.
///
/// Heights returned by this reader are orthometric metres, `H`, above the
/// EGM96 mean sea level geoid used by DTED/SRTM terrain products. They are not
/// ellipsoidal heights above the WGS84 reference ellipsoid.
#[derive(Debug)]
pub struct DtedTerrain {
    root: PathBuf,
    tiles: HashMap<(i32, i32), DtedTile>,
}

impl DtedTerrain {
    /// Build a terrain reader rooted at a directory containing DTED `.dt2`
    /// tiles, either directly or under the repository's block directories.
    #[must_use]
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            tiles: HashMap::new(),
        }
    }

    /// Return the bilinearly interpolated orthometric height `H` in metres at a
    /// longitude-first geodetic position in degrees.
    pub fn height_m(&mut self, longitude_deg: f64, latitude_deg: f64) -> crate::Result<f64> {
        self.height_m_with_options(longitude_deg, latitude_deg, DtedLookupOptions::default())
    }

    /// Return the orthometric height `H` in metres at a longitude-first
    /// geodetic position in degrees using explicit lookup options.
    ///
    /// A missing tile reads as sea level, `0.0`. A lookup that gives nonzero
    /// weight to a null posting returns [`Error::UnknownTerrainElevation`],
    /// unless a neighbouring present tile knows the height at the same place.
    /// For a nearest lookup that is the neighbour's posting at exactly the
    /// coordinates of a null edge posting. For a bilinear lookup exactly on a
    /// shared edge it is the neighbour's interpolation there: every weighted
    /// posting lies on that edge, so both tiles describe the same point. A
    /// tile whose DSI names a horizontal datum other than WGS84 is refused with
    /// [`Error::NonWgs84TerrainTile`], since queries are WGS84 positions.
    pub fn height_m_with_options(
        &mut self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: DtedLookupOptions,
    ) -> crate::Result<f64> {
        validate_lookup_coordinates(longitude_deg, latitude_deg)?;
        self.height_from_candidates(longitude_deg, latitude_deg, options)
            .1
    }

    /// Evaluate `(longitude_deg, latitude_deg)` points in order using one
    /// mutable borrow of the resident tile cache.
    ///
    /// The tuple order is intentionally longitude-first, matching
    /// [`Self::height_m_with_options`], even though geoid batch helpers use
    /// latitude-first points. Each result equals the scalar lookup's.
    pub fn height_batch(
        &mut self,
        points: &[(f64, f64)],
        options: DtedLookupOptions,
    ) -> Vec<crate::Result<f64>> {
        let mut out = Vec::with_capacity(points.len());
        let mut current = None;

        for &(longitude_deg, latitude_deg) in points {
            if let Err(err) = validate_lookup_coordinates(longitude_deg, latitude_deg) {
                out.push(Err(err));
                continue;
            }

            // The primary grid is always the first candidate, so its tile
            // answers unless it gives the point an unknown elevation.
            let primary_grid = terrain_grid(longitude_deg, latitude_deg);
            if current == Some(primary_grid) {
                if let Some(tile) = self.tiles.get(&primary_grid) {
                    if tile.contains(longitude_deg, latitude_deg) {
                        let result = height_from_tile(tile, longitude_deg, latitude_deg, options);
                        if !matches!(result, Err(Error::UnknownTerrainElevation { .. })) {
                            out.push(result);
                            continue;
                        }
                    }
                }
            }

            let (grid, result) = self.height_from_candidates(longitude_deg, latitude_deg, options);
            current = grid;
            out.push(result);
        }

        out
    }

    /// Answer a query from the tiles that contain it, in candidate order.
    ///
    /// Returns the first containing tile's grid, if any, with the result. No
    /// containing tile at all reads as sea level.
    ///
    /// When the first tile gives the point an unknown elevation, a nearest
    /// lookup takes the value of a neighbouring tile's posting at exactly the
    /// coordinates of the null posting, when that posting lies on the tile
    /// edge ([`edge_neighbour_grids`], [`coincident_posting`]). A bilinear
    /// lookup defers to the next containing candidate, which exists only for a
    /// point on a tile edge, where every weighted posting lies on that edge.
    /// A neighbouring tile is consulted only because the first tile gave an
    /// unknown elevation, so a neighbour that cannot be read, or that states
    /// another datum, leaves that unknown standing rather than replacing it
    /// with an error about a different tile, and the remaining candidates are
    /// still tried.
    fn height_from_candidates(
        &mut self,
        longitude: f64,
        latitude: f64,
        options: DtedLookupOptions,
    ) -> (Option<(i32, i32)>, crate::Result<f64>) {
        let mut first_grid = None;
        let mut unknown = None;
        for grid_idx in terrain_grid_candidates(longitude, latitude) {
            match self.load_tile(grid_idx) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(_) if unknown.is_some() => continue,
                Err(err) => return (first_grid, Err(err)),
            }
            let Some(tile) = self.tiles.get(&grid_idx) else {
                continue;
            };
            if !tile.contains(longitude, latitude) {
                continue;
            }
            if first_grid.is_none() {
                first_grid = Some(grid_idx);
            }
            let result = height_from_tile(tile, longitude, latitude, options);
            let unknown_at = result.as_ref().err().and_then(unknown_posting);
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
                let neighbour =
                    self.nearest_from_edge_neighbours(grid_idx, lon_posting, lat_posting);
                return (first_grid, neighbour.map_or(result, Ok));
            }
            if unknown.is_none() {
                unknown = result.err();
            }
        }
        (first_grid, unknown.map_or(Ok(0.0), Err))
    }

    /// Value of a neighbouring tile's posting at exactly the coordinates of
    /// the null posting `(lon_posting, lat_posting)` of the tile at `grid`,
    /// when that posting lies on the tile edge and a neighbour holds a known
    /// height there.
    fn nearest_from_edge_neighbours(
        &mut self,
        grid: (i32, i32),
        lon_posting: usize,
        lat_posting: usize,
    ) -> Option<f64> {
        let primary = self.tiles.get(&grid)?.grid();
        for neighbour_grid in edge_neighbour_grids(primary, lon_posting, lat_posting) {
            if !matches!(self.load_tile(neighbour_grid), Ok(true)) {
                continue;
            }
            let Some(neighbour) = self.tiles.get(&neighbour_grid) else {
                continue;
            };
            let Some((lon_index, lat_index)) =
                coincident_posting(primary, lon_posting, lat_posting, neighbour.grid())
            else {
                continue;
            };
            if let Ok(value) = neighbour.posting(lon_index, lat_index) {
                return Some(f64::from(value));
            }
        }
        None
    }

    /// Load the tile for `grid_idx` into the cache if its file is present,
    /// reporting whether it is cached.
    fn load_tile(&mut self, grid_idx: (i32, i32)) -> crate::Result<bool> {
        if !self.tiles.contains_key(&grid_idx) {
            let Some(path) = self.terrain_path_for_grid(grid_idx.0, grid_idx.1) else {
                return Ok(false);
            };
            if !path.is_file() {
                return Ok(false);
            }
            let tile =
                DtedTile::from_path(&path).map_err(|error| Error::Parse(error.to_string()))?;
            if tile.origin_latitude != f64::from(grid_idx.0)
                || tile.origin_longitude != f64::from(grid_idx.1)
            {
                return Err(Error::Parse(format!(
                    "{}: DTED origin ({},{}) does not match tile ({},{}) named by the file",
                    path.display(),
                    tile.origin_latitude,
                    tile.origin_longitude,
                    grid_idx.0,
                    grid_idx.1
                )));
            }
            if !tile.horizontal_datum.is_wgs84_compatible() {
                return Err(Error::NonWgs84TerrainTile {
                    lat_index: grid_idx.0,
                    lon_index: grid_idx.1,
                    datum: tile.horizontal_datum.clone(),
                });
            }
            self.tiles.insert(grid_idx, tile);
        }
        Ok(true)
    }

    fn terrain_path_for_grid(&self, latitude_index: i32, longitude_index: i32) -> Option<PathBuf> {
        let tile_name = format!(
            "{}_{}{}",
            format_lat(latitude_index),
            format_lon(longitude_index),
            DTED_SUFFIX
        );

        let direct = self.root.join(&tile_name);
        if direct.is_file() {
            return Some(direct);
        }

        let block_dir = terrain_block_dir(latitude_index, longitude_index);
        let nested = self.root.join(&block_dir).join(&tile_name);
        if nested.is_file() {
            return Some(nested);
        }

        let sibling = self.root.parent()?.join(&block_dir).join(&tile_name);
        sibling.is_file().then_some(sibling)
    }
}

fn height_from_tile(
    tile: &DtedTile,
    longitude_deg: f64,
    latitude_deg: f64,
    options: DtedLookupOptions,
) -> crate::Result<f64> {
    if options.interpolation == DtedInterpolation::NearestPosting {
        return tile
            .get_elevation(longitude_deg, latitude_deg)
            .map(f64::from)
            .map_err(|error| tile_lookup_error(tile, error));
    }

    let postings_per_deg_lon = tile.lon_count - 1;
    let postings_per_deg_lat = tile.lat_count - 1;

    let lon = in_tile_cell_fraction(longitude_deg, tile.origin_longitude, postings_per_deg_lon);
    let lat = in_tile_cell_fraction(latitude_deg, tile.origin_latitude, postings_per_deg_lat);
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
                tile.origin_longitude + (lon_lo + di) as f64 / postings_per_deg_lon as f64;
            let posting_lat =
                tile.origin_latitude + (lat_lo + dj) as f64 / postings_per_deg_lat as f64;
            z += w * f64::from(
                tile.get_elevation(posting_lon, posting_lat)
                    .map_err(|error| tile_lookup_error(tile, error))?,
            );
        }
    }
    Ok(z)
}

/// Map a tile lookup failure to the crate error, keeping a null posting typed.
fn tile_lookup_error(tile: &DtedTile, error: DtedTileError) -> Error {
    match error {
        DtedTileError::NullPosting {
            longitude_index,
            latitude_index,
        } => Error::UnknownTerrainElevation {
            // The origin is a validated whole degree inside [-180, 180).
            lat_index: tile.origin_latitude as i32,
            lon_index: tile.origin_longitude as i32,
            latitude_posting: latitude_index,
            longitude_posting: longitude_index,
        },
        other => Error::Parse(other.to_string()),
    }
}

/// Posting indices `(longitude, latitude)` of an unknown-elevation error.
pub(crate) fn unknown_posting(error: &Error) -> Option<(usize, usize)> {
    match error {
        Error::UnknownTerrainElevation {
            latitude_posting,
            longitude_posting,
            ..
        } => Some((*longitude_posting, *latitude_posting)),
        _ => None,
    }
}

/// Whole-degree origin and posting counts of a one-degree tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct TileGrid {
    pub(crate) lat_index: i32,
    pub(crate) lon_index: i32,
    pub(crate) lon_count: usize,
    pub(crate) lat_count: usize,
}

/// Tiles other than `tile` that share the location of its posting
/// `(lon_posting, lat_posting)`: none unless the posting lies on the tile
/// edge, up to three at a corner. Latitude-major order.
pub(crate) fn edge_neighbour_grids(
    tile: TileGrid,
    lon_posting: usize,
    lat_posting: usize,
) -> Vec<(i32, i32)> {
    let axis = |index: i32, posting: usize, count: usize| {
        if posting == 0 {
            vec![index, index - 1]
        } else if posting + 1 == count {
            vec![index, index + 1]
        } else {
            vec![index]
        }
    };
    let lats = axis(tile.lat_index, lat_posting, tile.lat_count);
    let lons = axis(tile.lon_index, lon_posting, tile.lon_count);
    let mut out = Vec::new();
    for &lat in &lats {
        for &lon in &lons {
            if (lat, lon) != (tile.lat_index, tile.lon_index) {
                out.push((lat, lon));
            }
        }
    }
    out
}

/// Posting indices `(longitude, latitude)` in `neighbour` at exactly the
/// coordinates of posting `(lon_posting, lat_posting)` of `tile`, if it has a
/// posting there.
///
/// Posting `i` of an axis with origin `o` and count `c` lies at
/// `o + i / (c - 1)` degrees; the comparison is on those rationals in integer
/// arithmetic, so neighbours with a different posting interval match only
/// where their postings coincide.
pub(crate) fn coincident_posting(
    tile: TileGrid,
    lon_posting: usize,
    lat_posting: usize,
    neighbour: TileGrid,
) -> Option<(usize, usize)> {
    Some((
        coincident_index(
            tile.lon_index,
            lon_posting,
            tile.lon_count,
            neighbour.lon_index,
            neighbour.lon_count,
        )?,
        coincident_index(
            tile.lat_index,
            lat_posting,
            tile.lat_count,
            neighbour.lat_index,
            neighbour.lat_count,
        )?,
    ))
}

/// Index `k` with `other_origin + k / (other_count - 1)` equal to
/// `origin + index / (count - 1)`, if one exists.
fn coincident_index(
    origin: i32,
    index: usize,
    count: usize,
    other_origin: i32,
    other_count: usize,
) -> Option<usize> {
    let intervals = count as i128 - 1;
    let other_intervals = other_count as i128 - 1;
    if intervals <= 0 || other_intervals <= 0 {
        return None;
    }
    // k = ((origin - other_origin) * intervals + index) * other_intervals / intervals
    let numerator = ((i128::from(origin) - i128::from(other_origin)) * intervals + index as i128)
        * other_intervals;
    if numerator % intervals != 0 {
        return None;
    }
    let k = numerator / intervals;
    if (0..=other_intervals).contains(&k) {
        usize::try_from(k).ok()
    } else {
        None
    }
}

pub(crate) fn validate_lookup_coordinates(
    longitude_deg: f64,
    latitude_deg: f64,
) -> crate::Result<()> {
    if !longitude_deg.is_finite() {
        return Err(Error::InvalidInput(
            "longitude_deg must be finite".to_string(),
        ));
    }
    if !latitude_deg.is_finite() {
        return Err(Error::InvalidInput(
            "latitude_deg must be finite".to_string(),
        ));
    }
    if !(MIN_LOOKUP_LONGITUDE_DEG..=MAX_LOOKUP_LONGITUDE_DEG).contains(&longitude_deg) {
        return Err(Error::InvalidInput(
            "longitude_deg must be within [-180, 180]".to_string(),
        ));
    }
    if !(MIN_LOOKUP_LATITUDE_DEG..=MAX_LOOKUP_LATITUDE_DEG).contains(&latitude_deg) {
        return Err(Error::InvalidInput(
            "latitude_deg must be within [-90, 90]".to_string(),
        ));
    }
    Ok(())
}

/// Horizontal datum stated by a DTED tile's DSI record (DSI character 145).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DtedHorizontalDatum {
    /// `WGS84`, the datum MIL-PRF-89020B 3.2.1 requires.
    Wgs84,
    /// `WGS72`, stated by cells compiled on the earlier World Geodetic System.
    Wgs72,
    /// The field is blank or zero-filled.
    Unstated,
    /// Any other field content, as read (lossily decoded if not UTF-8).
    Other(String),
}

impl DtedHorizontalDatum {
    /// Classify the five-byte DSI horizontal datum field. The codes compare
    /// without regard to ASCII case, as GDAL's DTED driver compares them.
    fn from_dsi_field(bytes: &[u8]) -> Self {
        if bytes.iter().all(|&b| b == b' ' || b == 0) {
            Self::Unstated
        } else if bytes.eq_ignore_ascii_case(b"WGS84") {
            Self::Wgs84
        } else if bytes.eq_ignore_ascii_case(b"WGS72") {
            Self::Wgs72
        } else {
            Self::Other(String::from_utf8_lossy(bytes).into_owned())
        }
    }

    /// Whether positions in this tile are WGS84 positions: the field states
    /// WGS84 or is blank.
    #[must_use]
    pub fn is_wgs84_compatible(&self) -> bool {
        matches!(self, Self::Wgs84 | Self::Unstated)
    }
}

impl core::fmt::Display for DtedHorizontalDatum {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Wgs84 => f.write_str("WGS84"),
            Self::Wgs72 => f.write_str("WGS72"),
            Self::Unstated => f.write_str("unstated"),
            Self::Other(text) => write!(f, "{text:?}"),
        }
    }
}

/// Parsed DTED tile backed by raw `.dt2` bytes.
///
/// Posting values are decoded lazily from DTED signed-magnitude samples.
/// Returned heights are orthometric metres, `H`, above the EGM96 mean sea level
/// geoid, on the horizontal datum [`Self::horizontal_datum`] reports.
#[derive(Debug)]
pub struct DtedTile {
    origin_latitude: f64,
    origin_longitude: f64,
    lon_count: usize,
    lat_count: usize,
    data_block_length: usize,
    horizontal_datum: DtedHorizontalDatum,
    bytes: Vec<u8>,
}

impl DtedTile {
    /// Read and parse a DTED `.dt2` tile from disk.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, DtedTileError> {
        let path = path.as_ref();
        let path_display = path.display().to_string();
        let bytes = fs::read(path).map_err(|error| DtedTileError::Io {
            path: path_display.clone(),
            message: error.to_string(),
        })?;
        if bytes.len() < DATA_OFFSET {
            return Err(DtedTileError::TooShort { path: path_display });
        }
        if &bytes[0..4] != b"UHL1" {
            return Err(DtedTileError::MissingUhl1 { path: path_display });
        }

        let origin_longitude = parse_origin(&bytes[4..12], OriginAxis::Longitude)?;
        let origin_latitude = parse_origin(&bytes[12..20], OriginAxis::Latitude)?;
        let lon_count = parse_ascii_usize(&bytes[47..51])?;
        let lat_count = parse_ascii_usize(&bytes[51..55])?;
        if lon_count < 2 || lat_count < 2 {
            return Err(DtedTileError::InvalidDimensions {
                path: path_display,
                lon_count,
                lat_count,
            });
        }
        validate_interval(&bytes[20..24], "longitude data interval", lon_count)?;
        validate_interval(&bytes[24..28], "latitude data interval", lat_count)?;
        let horizontal_datum = DtedHorizontalDatum::from_dsi_field(&bytes[DSI_HORIZONTAL_DATUM]);
        let data_block_length = 12 + 2 * lat_count;
        let expected_len = DATA_OFFSET + lon_count * data_block_length;
        if bytes.len() < expected_len {
            return Err(DtedTileError::Truncated {
                path: path_display,
                actual: bytes.len(),
                expected: expected_len,
            });
        }

        Ok(Self {
            origin_latitude,
            origin_longitude,
            lon_count,
            lat_count,
            data_block_length,
            horizontal_datum,
            bytes,
        })
    }

    /// Horizontal datum stated by the tile's DSI record.
    #[must_use]
    pub fn horizontal_datum(&self) -> &DtedHorizontalDatum {
        &self.horizontal_datum
    }

    /// Return the nearest orthometric posting value in metres for a
    /// longitude-first geodetic position in degrees.
    ///
    /// A posting holding the DTED null value returns
    /// [`DtedTileError::NullPosting`].
    pub fn get_elevation(&self, longitude: f64, latitude: f64) -> Result<i16, DtedTileError> {
        if !self.contains(longitude, latitude) {
            return Err(DtedTileError::Outside {
                longitude,
                latitude,
                origin_longitude: self.origin_longitude,
                origin_latitude: self.origin_latitude,
            });
        }

        let latitude_index =
            nearest_posting_index(latitude - self.origin_latitude, self.lat_count - 1)?;
        let longitude_index =
            nearest_posting_index(longitude - self.origin_longitude, self.lon_count - 1)?;
        if latitude_index >= self.lat_count || longitude_index >= self.lon_count {
            return Err(DtedTileError::PostingIndexOutOfBounds {
                longitude_index,
                latitude_index,
            });
        }

        self.posting(longitude_index, latitude_index)
    }

    /// Decoded posting `(longitude_index, latitude_index)`, or
    /// [`DtedTileError::NullPosting`] for the null value.
    pub(crate) fn posting(
        &self,
        longitude_index: usize,
        latitude_index: usize,
    ) -> Result<i16, DtedTileError> {
        if latitude_index >= self.lat_count || longitude_index >= self.lon_count {
            return Err(DtedTileError::PostingIndexOutOfBounds {
                longitude_index,
                latitude_index,
            });
        }
        let block = self.validated_block(longitude_index)?;
        posting_value(block, latitude_index).ok_or(DtedTileError::NullPosting {
            longitude_index,
            latitude_index,
        })
    }

    /// Whole-degree origin and posting counts. The origin is validated as a
    /// whole degree inside the coordinate domain, so the casts are exact.
    pub(crate) fn grid(&self) -> TileGrid {
        TileGrid {
            lat_index: self.origin_latitude as i32,
            lon_index: self.origin_longitude as i32,
            lon_count: self.lon_count,
            lat_count: self.lat_count,
        }
    }

    pub(crate) fn origin_latitude(&self) -> f64 {
        self.origin_latitude
    }

    pub(crate) fn origin_longitude(&self) -> f64 {
        self.origin_longitude
    }

    pub(crate) fn lon_count(&self) -> usize {
        self.lon_count
    }

    pub(crate) fn lat_count(&self) -> usize {
        self.lat_count
    }

    /// Decoded postings, longitude-major, with `None` for each null posting.
    pub(crate) fn decoded_postings_lon_major(&self) -> Result<Vec<Option<i16>>, DtedTileError> {
        let mut out = Vec::with_capacity(self.lon_count * self.lat_count);
        for longitude_index in 0..self.lon_count {
            let block = self.validated_block(longitude_index)?;
            for latitude_index in 0..self.lat_count {
                out.push(posting_value(block, latitude_index));
            }
        }
        Ok(out)
    }

    fn contains(&self, longitude: f64, latitude: f64) -> bool {
        latitude >= self.origin_latitude
            && latitude <= self.origin_latitude + 1.0
            && longitude >= self.origin_longitude
            && longitude <= self.origin_longitude + 1.0
    }

    fn validated_block(&self, longitude_index: usize) -> Result<&[u8], DtedTileError> {
        let block_start = DATA_OFFSET + longitude_index * self.data_block_length;
        let block_end = block_start + self.data_block_length;
        let block = &self.bytes[block_start..block_end];
        if block[0] != DATA_SENTINEL {
            return Err(DtedTileError::MissingDataSentinel { longitude_index });
        }
        let checksum = i32::from_be_bytes([
            block[block.len() - 4],
            block[block.len() - 3],
            block[block.len() - 2],
            block[block.len() - 1],
        ]);
        let sum = block[..block.len() - 4]
            .iter()
            .fold(0i32, |acc, b| acc + i32::from(*b));
        if sum != checksum {
            return Err(DtedTileError::Checksum {
                longitude_index,
                checksum,
                sum,
            });
        }
        // The data block count (bytes 1..4) is a tape sequencing counter that
        // no lookup depends on, and is not checked. The longitude and latitude
        // counts place the record's postings.
        let declared_longitude = signed_magnitude_field(block[4], block[5]);
        if i64::from(declared_longitude) != longitude_index as i64 {
            return Err(DtedTileError::ProfileLongitudeCountMismatch {
                longitude_index,
                declared: declared_longitude,
            });
        }
        let first_latitude_index = signed_magnitude_field(block[6], block[7]);
        if first_latitude_index != 0 {
            return Err(DtedTileError::UnsupportedPartialProfile {
                longitude_index,
                first_latitude_index,
            });
        }
        Ok(block)
    }
}

/// Posting `latitude_index` of a validated data record, or `None` for the
/// null value.
fn posting_value(block: &[u8], latitude_index: usize) -> Option<i16> {
    let sample_start = 8 + latitude_index * 2;
    let raw = u16::from_be_bytes([block[sample_start], block[sample_start + 1]]);
    (raw != DTED_NULL_POSTING_RAW).then(|| convert_signed_magnitude(raw as i16))
}

/// A two-byte signed-magnitude record count ("Fixed Binary" in
/// MIL-PRF-89020B).
fn signed_magnitude_field(high: u8, low: u8) -> i32 {
    let raw = u16::from_be_bytes([high, low]);
    let magnitude = i32::from(raw & 0x7fff);
    if raw & 0x8000 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

pub(crate) fn terrain_grid(longitude: f64, latitude: f64) -> (i32, i32) {
    (latitude.floor() as i32, longitude.floor() as i32)
}

pub(crate) fn terrain_grid_candidates(longitude: f64, latitude: f64) -> Vec<(i32, i32)> {
    let (lat, lon) = terrain_grid(longitude, latitude);
    let mut out = vec![(lat, lon)];
    let on_lat_edge = latitude == latitude.floor();
    let on_lon_edge = longitude == longitude.floor();
    if on_lat_edge {
        out.push((lat - 1, lon));
    }
    if on_lon_edge {
        out.push((lat, lon - 1));
    }
    if on_lat_edge && on_lon_edge {
        out.push((lat - 1, lon - 1));
    }
    out
}

pub(crate) fn format_lat(latitude_index: i32) -> String {
    if latitude_index >= 0 {
        format!("n{latitude_index:02}")
    } else {
        format!("s{:02}", -latitude_index)
    }
}

pub(crate) fn format_lon(longitude_index: i32) -> String {
    if longitude_index >= 0 {
        format!("e{longitude_index:03}")
    } else {
        format!("w{:03}", -longitude_index)
    }
}

pub(crate) fn terrain_block_dir(latitude_index: i32, longitude_index: i32) -> String {
    format!(
        "{}_{}",
        format_block_lat(latitude_index),
        format_block_lon(longitude_index)
    )
}

fn format_block_lat(latitude_index: i32) -> String {
    let origin = block_origin(latitude_index);
    if latitude_index >= 0 {
        format!("n{origin:02}")
    } else {
        format!("s{origin:02}")
    }
}

fn format_block_lon(longitude_index: i32) -> String {
    let origin = block_origin(longitude_index);
    if longitude_index >= 0 {
        format!("e{origin:03}")
    } else {
        format!("w{origin:03}")
    }
}

pub(crate) fn block_origin(index: i32) -> u32 {
    (index.unsigned_abs() / 10) * 10
}

fn parse_ascii_usize(bytes: &[u8]) -> Result<usize, DtedTileError> {
    std::str::from_utf8(bytes)
        .map_err(|error| DtedTileError::InvalidEncoding(error.to_string()))?
        .trim()
        .parse::<usize>()
        .map_err(|error| DtedTileError::InvalidField(error.to_string()))
}

/// Degrees, minutes, seconds and hemisphere of a DTED coordinate field, as
/// written; ranges are checked by the caller, which knows the axis.
#[derive(Clone, Copy, Debug, PartialEq)]
struct DtedCoordinate {
    degree: u32,
    minute: u32,
    second: f64,
    hemisphere: char,
}

/// Split a `D..DMMSS[.S]H` coordinate field into its subfields. Every subfield
/// must be unsigned decimal digits.
fn parse_dted_coord(input: &str) -> Result<DtedCoordinate, DtedTileError> {
    let invalid = || DtedTileError::InvalidField("invalid DTED coordinate".to_string());
    let (hemi_start, hemisphere) = input
        .char_indices()
        .last()
        .ok_or(DtedTileError::EmptyCoordinate)?;
    if !matches!(hemisphere, 'N' | 'S' | 'E' | 'W') {
        return Err(DtedTileError::InvalidHemisphere { hemisphere });
    }
    let coord = &input[..hemi_start];
    if !coord.is_ascii() {
        return Err(invalid());
    }
    let seconds_index = if coord.as_bytes().get(coord.len().saturating_sub(2)) == Some(&b'.') {
        coord.len().checked_sub(4)
    } else {
        coord.len().checked_sub(2)
    }
    .ok_or_else(invalid)?;
    let minutes_index = seconds_index
        .checked_sub(2)
        .filter(|&index| index > 0)
        .ok_or_else(invalid)?;
    let degree_text = &coord[..minutes_index];
    let minute_text = &coord[minutes_index..seconds_index];
    let second_text = &coord[seconds_index..];
    let seconds_are_digits = second_text
        .split('.')
        .all(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()));
    if !all_ascii_digits(degree_text) || !all_ascii_digits(minute_text) || !seconds_are_digits {
        return Err(invalid());
    }
    let degree = degree_text
        .parse::<u32>()
        .map_err(|error| DtedTileError::InvalidField(error.to_string()))?;
    let minute = minute_text
        .parse::<u32>()
        .map_err(|error| DtedTileError::InvalidField(error.to_string()))?;
    let second = second_text
        .parse::<f64>()
        .map_err(|error| DtedTileError::InvalidField(error.to_string()))?;
    Ok(DtedCoordinate {
        degree,
        minute,
        second,
        hemisphere,
    })
}

fn all_ascii_digits(text: &str) -> bool {
    !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit())
}

#[derive(Clone, Copy, Debug)]
enum OriginAxis {
    Longitude,
    Latitude,
}

impl OriginAxis {
    const fn field(self) -> &'static str {
        match self {
            Self::Longitude => "longitude of origin",
            Self::Latitude => "latitude of origin",
        }
    }

    /// Positive and negative hemisphere letters.
    const fn hemispheres(self) -> (char, char) {
        match self {
            Self::Longitude => ('E', 'W'),
            Self::Latitude => ('N', 'S'),
        }
    }

    const fn expected(self) -> &'static str {
        match self {
            Self::Longitude => "E or W",
            Self::Latitude => "N or S",
        }
    }

    /// Largest whole-degree origin on each side of zero for a one-degree tile:
    /// the tile must end at or before 90 N / 180 E and start at or after
    /// 90 S / 180 W.
    const fn max_degrees(self) -> (u32, u32) {
        match self {
            Self::Longitude => (179, 180),
            Self::Latitude => (89, 90),
        }
    }
}

/// Parse and validate a UHL origin field as signed whole degrees.
fn parse_origin(bytes: &[u8], axis: OriginAxis) -> Result<f64, DtedTileError> {
    let text = std::str::from_utf8(bytes)
        .map_err(|error| DtedTileError::InvalidEncoding(error.to_string()))?;
    let coordinate = parse_dted_coord(text)?;
    let field = axis.field();
    let (positive, negative) = axis.hemispheres();
    let is_negative = if coordinate.hemisphere == positive {
        false
    } else if coordinate.hemisphere == negative {
        true
    } else {
        return Err(DtedTileError::WrongHemisphere {
            field,
            hemisphere: coordinate.hemisphere,
            expected: axis.expected(),
        });
    };
    let (max_positive, max_negative) = axis.max_degrees();
    let max_degree = if is_negative {
        max_negative
    } else {
        max_positive
    };
    if coordinate.minute >= 60 || coordinate.second >= 60.0 || coordinate.degree > max_degree {
        return Err(DtedTileError::CoordinateOutOfRange {
            field,
            text: text.to_string(),
        });
    }
    if coordinate.minute != 0 || coordinate.second != 0.0 {
        return Err(DtedTileError::OriginNotWholeDegree {
            field,
            text: text.to_string(),
        });
    }
    let degrees = f64::from(coordinate.degree);
    Ok(if is_negative { -degrees } else { degrees })
}

/// Check a UHL data interval against the posting count on the same axis.
///
/// The reader places posting `i` at `origin + i / (count - 1)` degrees, so a
/// stated interval must make `count - 1` intervals span exactly one degree. A
/// blank interval field states nothing and leaves that placement in force.
fn validate_interval(bytes: &[u8], field: &'static str, count: usize) -> Result<(), DtedTileError> {
    if bytes.iter().all(|&b| b == b' ') {
        return Ok(());
    }
    let text = std::str::from_utf8(bytes)
        .map_err(|error| DtedTileError::InvalidEncoding(error.to_string()))?
        .trim();
    if !all_ascii_digits(text) {
        return Err(DtedTileError::InvalidField(format!(
            "invalid DTED {field} {text:?}"
        )));
    }
    let interval_tenths_arcsec = text
        .parse::<u32>()
        .map_err(|error| DtedTileError::InvalidField(error.to_string()))?;
    let span = u64::from(interval_tenths_arcsec) * (count as u64 - 1);
    if span != ONE_DEGREE_TENTHS_ARCSEC {
        return Err(DtedTileError::IntervalCountMismatch {
            field,
            interval_tenths_arcsec,
            count,
        });
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct ScaledCellFraction {
    pub(crate) cell: i64,
    pub(crate) fraction: f64,
    nearest: i64,
}

/// Scale an exact binary64 value by an integer without first rounding their
/// product to binary64.
// invariant: callers pass an in-tile offset and a positive posting count.
#[allow(clippy::expect_used)]
pub(crate) fn scaled_cell_fraction(offset: f64, postings_per_degree: usize) -> ScaledCellFraction {
    debug_assert!(offset.is_finite());
    debug_assert!(postings_per_degree > 0);

    let bits = offset.to_bits();
    let negative = bits >> 63 != 0;
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let stored_significand = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (stored_significand, -1074)
    } else {
        (
            stored_significand | (1_u64 << 52),
            exponent_bits - 1023 - 52,
        )
    };
    if significand == 0 {
        return ScaledCellFraction {
            cell: 0,
            fraction: 0.0,
            nearest: 0,
        };
    }

    let numerator = u128::from(significand) * postings_per_degree as u128;
    if exponent >= 0 {
        let magnitude = numerator
            .checked_shl(exponent as u32)
            .expect("in-tile scaled coordinate must fit u128");
        let magnitude =
            i64::try_from(magnitude).expect("in-tile scaled coordinate must fit a posting index");
        let cell = if negative { -magnitude } else { magnitude };
        return ScaledCellFraction {
            cell,
            fraction: 0.0,
            nearest: cell,
        };
    }

    let denominator_exponent = (-exponent) as u32;
    if denominator_exponent >= 128 {
        if negative {
            return ScaledCellFraction {
                cell: -1,
                fraction: one_minus_dyadic(numerator, denominator_exponent),
                nearest: 0,
            };
        }
        return ScaledCellFraction {
            cell: 0,
            fraction: dyadic_to_f64(numerator, exponent),
            nearest: 0,
        };
    }

    let denominator = 1_u128 << denominator_exponent;
    let integer = numerator >> denominator_exponent;
    let remainder = numerator & (denominator - 1);
    let (cell, euclidean_remainder) = if negative {
        if remainder == 0 {
            (-(integer as i64), 0)
        } else {
            (-(integer as i64) - 1, denominator - remainder)
        }
    } else {
        (integer as i64, remainder)
    };
    let half = denominator >> 1;
    let nearest = if euclidean_remainder < half || (euclidean_remainder == half && cell % 2 == 0) {
        cell
    } else {
        cell + 1
    };
    ScaledCellFraction {
        cell,
        fraction: dyadic_to_f64(euclidean_remainder, exponent),
        nearest,
    }
}

/// Locate a coordinate within a one-degree tile, in postings.
///
/// The cell offset is measured from whichever tile edge is nearer, so the
/// subtraction that produces it is exact, and the complement is then taken on
/// the exact integer ratio rather than on a rounded binary64 fraction.
/// Measuring from the tile origin alone is exact for most tiles but not for a
/// tile whose origin index is -1: there the upper edge is zero, so the offset
/// is `coordinate + 1`, which discards every bit of the coordinate below one
/// ulp of 1. That loss moves the interpolation weights and the interpolated
/// height in its last bits across the one-degree band south of the equator and
/// the one west of the prime meridian.
pub(crate) fn in_tile_cell_fraction(
    coordinate: f64,
    origin: f64,
    postings_per_degree: usize,
) -> ScaledCellFraction {
    let from_lower = coordinate - origin;
    let from_upper = (origin + 1.0) - coordinate;
    if from_lower <= from_upper {
        scaled_cell_fraction(from_lower, postings_per_degree)
    } else {
        complement_cell_fraction(from_upper, postings_per_degree)
    }
}

/// `postings_per_degree * (1 - offset)` as a cell index and fraction, for a
/// non-negative `offset` no greater than half a degree.
///
/// The fractional part is complemented on the exact dyadic remainder of
/// `postings_per_degree * offset`, because complementing a binary64 fraction
/// instead would reintroduce the rounding this path exists to avoid.
fn complement_cell_fraction(offset: f64, postings_per_degree: usize) -> ScaledCellFraction {
    let postings = postings_per_degree as i64;
    let (integer, remainder, denominator_exponent) =
        exact_scaled_parts(offset, postings_per_degree);
    if remainder == 0 {
        let cell = postings - integer;
        return ScaledCellFraction {
            cell,
            fraction: 0.0,
            nearest: cell,
        };
    }

    let cell = postings - integer - 1;
    let (fraction, rounds_up) = if denominator_exponent >= 128 {
        (one_minus_dyadic(remainder, denominator_exponent), true)
    } else {
        let denominator = 1_u128 << denominator_exponent;
        let complement = denominator - remainder;
        let half = denominator >> 1;
        let rounds_up = complement > half || (complement == half && cell.rem_euclid(2) != 0);
        (
            dyadic_to_f64(complement, -(denominator_exponent as i32)),
            rounds_up,
        )
    };
    ScaledCellFraction {
        cell,
        fraction,
        nearest: if rounds_up { cell + 1 } else { cell },
    }
}

/// Exact `postings_per_degree * offset` for a non-negative `offset`, as an
/// integer part and a remainder over a power of two.
// invariant: callers pass a non-negative in-tile offset and a positive posting
// count, so the scaled significand fits u128.
#[allow(clippy::expect_used)]
fn exact_scaled_parts(offset: f64, postings_per_degree: usize) -> (i64, u128, u32) {
    debug_assert!(offset.is_finite());
    debug_assert!(offset >= 0.0);
    debug_assert!(postings_per_degree > 0);

    let bits = offset.to_bits();
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let stored_significand = bits & ((1_u64 << 52) - 1);
    let (significand, exponent) = if exponent_bits == 0 {
        (stored_significand, -1074)
    } else {
        (
            stored_significand | (1_u64 << 52),
            exponent_bits - 1023 - 52,
        )
    };
    if significand == 0 {
        return (0, 0, 0);
    }

    let numerator = u128::from(significand) * postings_per_degree as u128;
    if exponent >= 0 {
        let magnitude = numerator
            .checked_shl(exponent as u32)
            .expect("in-tile scaled coordinate must fit u128");
        let magnitude =
            i64::try_from(magnitude).expect("in-tile scaled coordinate must fit a posting index");
        return (magnitude, 0, 0);
    }

    let denominator_exponent = (-exponent) as u32;
    if denominator_exponent >= 128 {
        return (0, numerator, denominator_exponent);
    }
    let integer = i64::try_from(numerator >> denominator_exponent)
        .expect("in-tile scaled coordinate must fit a posting index");
    let remainder = numerator & ((1_u128 << denominator_exponent) - 1);
    (integer, remainder, denominator_exponent)
}

pub(crate) fn nearest_posting_index<E>(offset: f64, postings_per_degree: usize) -> Result<usize, E>
where
    E: From<DtedTileError>,
{
    let scaled = scaled_cell_fraction(offset, postings_per_degree);
    usize::try_from(scaled.nearest)
        .map_err(|_| DtedTileError::NegativePostingIndex {
            index: scaled.nearest,
        })
        .map_err(E::from)
}

impl From<DtedTileError> for String {
    fn from(error: DtedTileError) -> Self {
        error.to_string()
    }
}

fn one_minus_dyadic(numerator: u128, denominator_exponent: u32) -> f64 {
    let deficit_units = round_shift_right(numerator, denominator_exponent - 53);
    1.0 - deficit_units as f64 * (f64::EPSILON / 2.0)
}

// invariant: binary64 decomposition bounds the normal exponent before encoding.
#[allow(clippy::expect_used)]
fn dyadic_to_f64(numerator: u128, exponent: i32) -> f64 {
    if numerator == 0 {
        return 0.0;
    }

    let bit_length = 128 - numerator.leading_zeros();
    let mut binary_exponent = bit_length as i32 - 1 + exponent;
    if binary_exponent >= -1022 {
        let mut significand = if bit_length <= 53 {
            numerator << (53 - bit_length)
        } else {
            round_shift_right(numerator, bit_length - 53)
        };
        if significand == 1_u128 << 53 {
            significand >>= 1;
            binary_exponent += 1;
        }
        let exponent_bits = u64::try_from(binary_exponent + 1023)
            .expect("normal binary64 exponent must be nonnegative");
        let fraction_bits = significand as u64 & ((1_u64 << 52) - 1);
        return f64::from_bits((exponent_bits << 52) | fraction_bits);
    }

    let subnormal_shift = exponent + 1074;
    let significand = if subnormal_shift >= 0 {
        numerator << subnormal_shift as u32
    } else {
        round_shift_right(numerator, (-subnormal_shift) as u32)
    };
    f64::from_bits(significand as u64)
}

fn round_shift_right(value: u128, shift: u32) -> u128 {
    if shift == 0 {
        return value;
    }
    if shift > 128 {
        return 0;
    }

    let quotient = if shift == 128 { 0 } else { value >> shift };
    let remainder = if shift == 128 {
        value
    } else {
        value & ((1_u128 << shift) - 1)
    };
    let half = 1_u128 << (shift - 1);
    if remainder > half || (remainder == half && !quotient.is_multiple_of(2)) {
        quotient + 1
    } else {
        quotient
    }
}

/// Lowest decoded signed-magnitude value GDAL keeps; below it a negative
/// posting is read as two's complement.
const TWOS_COMPLEMENT_THRESHOLD_M: i32 = -16_000;

/// Decode a non-null posting.
///
/// Signed magnitude per MIL-PRF-89020B, except that a negative value below
/// -16000 is reinterpreted as two's complement, matching GDAL's
/// `DTEDReadProfileEx` and `DTEDReadPoint` for producers that wrote negatives
/// that way (GDAL cites `w_069_s50.dt0`). The null pattern never reaches this
/// function.
fn convert_signed_magnitude(raw: i16) -> i16 {
    if raw >= 0 {
        return raw;
    }
    let signed_magnitude = -32768i32 - i32::from(raw);
    if signed_magnitude < TWOS_COMPLEMENT_THRESHOLD_M {
        raw
    } else {
        signed_magnitude as i16
    }
}

#[cfg(all(test, sidereon_repo_tests))]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    //! DTED batch fixture provenance: adjacent synthetic tiles under
    //! `tests/fixtures/dted/tiles` are written by
    //! `crates/sidereon-core/fixtures-generators/generate_dted_points.py` using
    //! the public DTED UHL/DSI/ACC/data-record layout. Tests compare
    //! `f64::to_bits` exactly, never tolerances.

    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::Value;

    use crate::test_parity::f64_from_hex;
    use crate::Error;

    use super::{coincident_posting, edge_neighbour_grids, TileGrid};
    use super::{
        in_tile_cell_fraction, nearest_posting_index, scaled_cell_fraction, terrain_block_dir,
        DtedHorizontalDatum, DtedInterpolation, DtedLookupOptions, DtedTerrain, DtedTile,
        DtedTileError, DATA_OFFSET, DATA_SENTINEL, DTED_SUFFIX,
    };

    #[test]
    fn exact_scaling_preserves_the_split_point_fraction() {
        let tile_origin = -107.0;
        let coordinate = -106.265_141_029_846_36;
        let offset = coordinate - tile_origin;
        assert_eq!(offset, 0.734_858_970_153_638_3);

        let scaled = scaled_cell_fraction(offset, 3600);
        let naive = offset * 3600.0;
        let naive_fraction = naive - naive.floor();
        assert_eq!(scaled.cell, 2645);
        assert_eq!(scaled.fraction.to_bits(), 0x3fdf_81b8_9fe7_b000);
        assert_eq!(naive.floor() as i64, 2645);
        assert_eq!(naive_fraction.to_bits(), 0x3fdf_81b8_9fe7_c000);
        assert_eq!(naive_fraction.to_bits() - scaled.fraction.to_bits(), 4096);

        let nondiscriminating_offset = -0.265_141_029_846_361_7;
        let nondiscriminating = scaled_cell_fraction(nondiscriminating_offset, 3600);
        let naive = nondiscriminating_offset * 3600.0;
        assert_eq!(
            nondiscriminating.fraction.to_bits(),
            (naive - naive.floor()).to_bits()
        );
    }

    #[test]
    fn exact_scaling_keeps_a_coordinate_below_the_posting_in_the_lower_cell() {
        let posting = 3.0_f64 / 3600.0;
        let coordinate = f64::from_bits(posting.to_bits() - 1);
        assert!(coordinate < posting);
        assert_eq!(coordinate * 3600.0, 3.0);

        let scaled = scaled_cell_fraction(coordinate, 3600);
        assert_eq!(scaled.cell, 2);
        assert_eq!(scaled.fraction.to_bits(), 0x3fef_ffff_ffff_fffe);
        assert_eq!(scaled.nearest, 3);
    }

    #[test]
    fn nearest_posting_rounds_the_exact_product_instead_of_the_binary64_product() {
        let half_posting = 1.5_f64 / 3600.0;
        let coordinate = f64::from_bits(half_posting.to_bits() - 1);
        assert!(coordinate < half_posting);
        assert_eq!(coordinate * 3600.0, 1.5);

        assert_eq!(
            nearest_posting_index::<DtedTileError>(coordinate, 3600),
            Ok(1)
        );
    }

    #[test]
    fn exact_scaled_value_tracks_the_binary64_product_over_deterministic_offsets() {
        let mut state = 0x764e_279d_9f41_2c03_u64;
        for _ in 0..10_000 {
            state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut random = state;
            random = (random ^ (random >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            random = (random ^ (random >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            random ^= random >> 31;
            let offset = f64::from_bits(0x3ff0_0000_0000_0000 | (random >> 12)) - 1.0;

            let exact = scaled_cell_fraction(offset, 3600);
            let reconstructed = exact.cell as f64 + exact.fraction;
            let naive = offset * 3600.0;
            assert!(
                reconstructed.to_bits().abs_diff(naive.to_bits()) <= 1,
                "offset={offset} exact={reconstructed} naive={naive}"
            );
            assert!(exact.cell <= naive.floor() as i64, "offset={offset}");
        }
    }

    #[test]
    fn terrain_block_dir_matches_reference_bucket_names() {
        assert_eq!(terrain_block_dir(36, -107), "n30_w100");
        assert_eq!(terrain_block_dir(32, -117), "n30_w110");
        assert_eq!(terrain_block_dir(43, -112), "n40_w110");
        assert_eq!(terrain_block_dir(20, -103), "n20_w100");
        assert_eq!(terrain_block_dir(36, 107), "n30_e100");
        assert_eq!(terrain_block_dir(-1, -1), "s00_w000");
        assert_eq!(terrain_block_dir(1, 1), "n00_e000");
        assert_eq!(terrain_block_dir(-1, 1), "s00_e000");
        assert_eq!(terrain_block_dir(32, -110), "n30_w110");
        assert_eq!(terrain_block_dir(32, -111), "n30_w110");
        assert_eq!(terrain_block_dir(32, -1), "n30_w000");
        assert_eq!(terrain_block_dir(32, -10), "n30_w010");
    }

    #[test]
    fn negative_tile_indices_resolve_to_negative_block_dir() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "sidereon-dted-negative-block-{}-{nonce}",
            std::process::id()
        ));
        let tile_dir = root.join("s00_w000");
        let tile_path = tile_dir.join("s01_w001_1arc_v3.dt2");
        fs::create_dir_all(&tile_dir).expect("create nested DTED block dir");
        fs::write(&tile_path, []).expect("create nested DTED tile path");

        let terrain = DtedTerrain::new(&root);
        let got = terrain
            .terrain_path_for_grid(-1, -1)
            .expect("negative nested tile path");
        assert_eq!(got, tile_path);

        fs::remove_dir_all(root).expect("remove temp DTED block dir");
    }

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("dted")
            .join(name)
    }

    fn bits(v: &Value) -> f64 {
        f64_from_hex(v.as_str().expect("hex-bit string")).expect("valid f64 bits")
    }

    fn temp_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("sidereon-{name}-{}-{nonce}", std::process::id()))
    }

    fn scalar_loop(
        root: &Path,
        points: &[(f64, f64)],
        options: DtedLookupOptions,
    ) -> Vec<crate::Result<f64>> {
        let mut terrain = DtedTerrain::new(root);
        points
            .iter()
            .map(|&(lon, lat)| terrain.height_m_with_options(lon, lat, options))
            .collect()
    }

    fn assert_height_results_match(
        got: &[crate::Result<f64>],
        want: &[crate::Result<f64>],
        context: &str,
    ) {
        assert_eq!(got.len(), want.len(), "{context} result length");
        for (idx, (got, want)) in got.iter().zip(want).enumerate() {
            match (got, want) {
                (Ok(got), Ok(want)) => assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{context} index {idx} height bits"
                ),
                (Err(got), Err(want)) => {
                    assert_eq!(got, want, "{context} index {idx} error")
                }
                (got, want) => panic!("{context} index {idx} mismatch: {got:?} != {want:?}"),
            }
        }
    }

    fn copy_fixture_tile(root: &Path, tile_name: &str) {
        fs::copy(
            fixture_path(&format!("tiles/{tile_name}")),
            root.join(tile_name),
        )
        .expect("copy DTED fixture tile");
    }

    fn copy_primary_fixture_root(name: &str) -> PathBuf {
        let root = temp_path(name);
        fs::create_dir_all(&root).expect("create temp DTED dir");
        copy_fixture_tile(&root, "n36_w107_1arc_v3.dt2");
        root
    }

    fn write_synthetic_dted_tile(
        path: &Path,
        lon_count: usize,
        lat_count: usize,
        sample: impl Fn(usize, usize) -> i16,
    ) {
        write_synthetic_dted_tile_at(path, b"1070000W", b"0360000N", lon_count, lat_count, sample);
    }

    fn write_synthetic_dted_tile_at(
        path: &Path,
        longitude_origin: &[u8; 8],
        latitude_origin: &[u8; 8],
        lon_count: usize,
        lat_count: usize,
        sample: impl Fn(usize, usize) -> i16,
    ) {
        let data_block_length = 12 + 2 * lat_count;
        let mut bytes = vec![b' '; DATA_OFFSET];
        bytes[0..4].copy_from_slice(b"UHL1");
        bytes[4..12].copy_from_slice(longitude_origin);
        bytes[12..20].copy_from_slice(latitude_origin);
        bytes[47..51].copy_from_slice(format!("{lon_count:04}").as_bytes());
        bytes[51..55].copy_from_slice(format!("{lat_count:04}").as_bytes());

        for lon_index in 0..lon_count {
            let mut block = vec![0u8; data_block_length];
            block[0] = DATA_SENTINEL;
            // MIL-PRF-89020B data record: block count, then the longitude
            // count naming this profile's meridian, then latitude count 0 for
            // a full profile.
            block[1..4].copy_from_slice(&(lon_index as u32).to_be_bytes()[1..4]);
            block[4..6].copy_from_slice(&(lon_index as u16).to_be_bytes());
            for lat_index in 0..lat_count {
                let sample_start = 8 + lat_index * 2;
                block[sample_start..sample_start + 2]
                    .copy_from_slice(&sample(lon_index, lat_index).to_be_bytes());
            }
            let checksum = block[..block.len() - 4]
                .iter()
                .fold(0i32, |acc, b| acc + i32::from(*b));
            let checksum_start = block.len() - 4;
            block[checksum_start..].copy_from_slice(&checksum.to_be_bytes());
            bytes.extend(block);
        }

        fs::write(path, bytes).expect("write synthetic DTED tile");
    }

    /// Cell index and fraction for coordinates carrying bits below one ulp of
    /// 1, pinned to the exact rational value of `postings * (coordinate -
    /// origin)` computed outside this crate.
    ///
    /// The probes are arbitrary binary64 values inside each tile, not
    /// `origin + random()`. A coordinate written that way carries no bits
    /// below ulp(1), which is the only place a float offset differs from the
    /// exact one, so a sweep built that way passes whether or not the offset
    /// is computed correctly.
    #[test]
    fn bilinear_cell_offset_is_exact_in_every_tile() {
        // (coordinate bits, tile origin, expected cell, expected fraction bits)
        const CASES: &[(u64, f64, i64, u64)] = &[
            (0xbf1a36e2eb1c432d, -1.0, 3599, 0x3fe47ae147ae147b),
            (0xbf1a36e2eb1c432c, -1.0, 3599, 0x3fe47ae147ae147b),
            (0xbfd73a99165fe501, -1.0, 2293, 0x3fd7f7355b7ba1f0),
            (0xbfdfedcba9876543, -1.0, 1804, 0x3d1d000000000000),
            (0xbfeffffffff24190, -1.0, 0, 0x3e9828c0e0000000),
            (0xbfe0000000000000, -1.0, 1800, 0x0000000000000000),
            (0xbd719799812dea11, -1.0, 3599, 0x3feffffffe113843),
            (0xbfe8000000000001, -1.0, 899, 0x3feffffffffff1f0),
            (0xbfd0000000000002, -1.0, 2699, 0x3feffffffffff1f0),
            (0xbfeccccccccccccd, -1.0, 359, 0x3feffffffffffd30),
            (0x3f1a36e2eb1c432d, 0.0, 0, 0x3fd70a3d70a3d70b),
            (0x3fd73a99165fe501, 0.0, 1306, 0x3fe4046552422f08),
            (0x3fdfedcba9876543, 0.0, 1795, 0x3fefffffffffff18),
            (0x3feffffffff24190, 0.0, 3599, 0x3fefffff3eb9f900),
            (0x3fe0000000000000, 0.0, 1800, 0x0000000000000000),
            (0x3d719799812dea11, 0.0, 0, 0x3e2eec7bd512b572),
            (0x3fe8000000000000, 0.0, 2700, 0x0000000000000000),
            (0x3fd0000000000002, 0.0, 900, 0x3d5c200000000000),
            (0x4049800346dc5d64, 51.0, 0, 0x3fd70a3d70a72000),
            (0x4049ae75322cbfca, 51.0, 1306, 0x3fe4046552422800),
            (0x4049bfdb97530ecb, 51.0, 1796, 0x3daac00000000000),
            (0x4049ffffffffc906, 51.0, 3599, 0x3fefffff3eb91800),
            (0x4049c00000000000, 51.0, 1800, 0x0000000000000000),
            (0xc05a8001a36e2eb2, -107.0, 3599, 0x3fe47ae147ac7000),
            (0xc05a973a99165fe5, -107.0, 2293, 0x3fd7f7355b7bb000),
            (0xc05abfedcba98765, -107.0, 4, 0x3dad800000000000),
            (0xc05aa00000000000, -107.0, 1800, 0x0000000000000000),
        ];

        for (coordinate_bits, origin, expected_cell, expected_fraction_bits) in CASES {
            let coordinate = f64::from_bits(*coordinate_bits);
            let scaled = in_tile_cell_fraction(coordinate, *origin, 3600);
            assert_eq!(
                scaled.cell, *expected_cell,
                "cell for {coordinate:e} in tile at {origin}"
            );
            assert_eq!(
                scaled.fraction.to_bits(),
                *expected_fraction_bits,
                "fraction for {coordinate:e} in tile at {origin}: got {:e}, want {:e}",
                scaled.fraction,
                f64::from_bits(*expected_fraction_bits)
            );
        }
    }

    /// Interpolated heights in the one-degree tile south of the equator and
    /// west of the prime meridian, pinned to the values that exact cell
    /// offsets produce. Measuring the offset from the tile origin there means
    /// adding 1 to the coordinate, which discards its low bits and moves both
    /// the weights and, near a cell boundary, the cell itself.
    #[test]
    fn bilinear_height_is_exact_south_of_equator_and_west_of_meridian() {
        let root = temp_path("dted-zero-edge-precision");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let postings = 1200;
        write_synthetic_dted_tile_at(
            &root.join(format!("s01_w001{DTED_SUFFIX}")),
            b"0010000W",
            b"0010000S",
            postings + 1,
            postings + 1,
            |lon_index, lat_index| {
                if (lon_index + lat_index) % 2 == 0 {
                    0
                } else {
                    8849
                }
            },
        );

        // (longitude bits, latitude bits, expected height bits)
        const CASES: &[(u64, u64, u64)] = &[
            (0xbf1a36e2eb1c432d, 0xbf1a36e2eb1c432c, 0x409d33a29c779a6b),
            (0xbfd73a99165fe501, 0xbfd73a99165fe500, 0x40b129828ba475c4),
            (0xbfdfedcba9876543, 0xbfdfedcba9876542, 0x40aeb9c71c71c93c),
            (0xbfeffffffff24190, 0xbfeffffffff920c8, 0x3f5a18c574f03816),
            (0xbd3c25c268497682, 0xbd4c25c268497682, 0x3ecab91c416da2d4),
            (0xbfe0000000000000, 0xbfe0000000000000, 0x0000000000000000),
        ];

        let mut terrain = DtedTerrain::new(&root);
        let options = DtedLookupOptions {
            interpolation: DtedInterpolation::Bilinear,
        };
        for (longitude_bits, latitude_bits, expected_bits) in CASES {
            let longitude = f64::from_bits(*longitude_bits);
            let latitude = f64::from_bits(*latitude_bits);
            let height = terrain
                .height_m_with_options(longitude, latitude, options)
                .expect("bilinear height");
            assert_eq!(
                height.to_bits(),
                *expected_bits,
                "height at ({longitude:e}, {latitude:e}): got {height:?}, want {:?}",
                f64::from_bits(*expected_bits)
            );
        }

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    #[test]
    fn dted_rejects_degenerate_header_counts() {
        let root = temp_path("dted-degenerate-counts");
        fs::create_dir_all(&root).expect("create temp DTED dir");

        for (lon_count, lat_count) in [(0, 2), (1, 2), (2, 0), (2, 1)] {
            let tile_path = root.join(format!("tile-{lon_count}-{lat_count}.dt2"));
            write_synthetic_dted_tile(&tile_path, lon_count, lat_count, |_, _| 0);

            let err = DtedTile::from_path(&tile_path).expect_err("degenerate counts must error");
            assert!(
                err.to_string().contains("invalid DTED dimensions"),
                "unexpected error for lon_count={lon_count} lat_count={lat_count}: {err}"
            );
        }

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    #[test]
    fn dted_lookup_rejects_nonfinite_coordinates() {
        let root = temp_path("dted-nonfinite-coordinates");
        let mut terrain = DtedTerrain::new(&root);

        for (lon, lat, field) in [
            (f64::NAN, 36.5, "longitude_deg"),
            (f64::INFINITY, 36.5, "longitude_deg"),
            (f64::NEG_INFINITY, 36.5, "longitude_deg"),
            (-106.5, f64::NAN, "latitude_deg"),
            (-106.5, f64::INFINITY, "latitude_deg"),
            (-106.5, f64::NEG_INFINITY, "latitude_deg"),
        ] {
            assert_eq!(
                terrain
                    .height_m_with_options(lon, lat, DtedLookupOptions::default())
                    .expect_err("non-finite DTED coordinate must error"),
                Error::InvalidInput(format!("{field} must be finite"))
            );
        }

        assert_eq!(
            terrain
                .height_m(f64::NAN, 36.5)
                .expect_err("height_m must also reject non-finite coordinates"),
            Error::InvalidInput("longitude_deg must be finite".to_string())
        );
    }

    #[test]
    fn dted_lookup_rejects_out_of_range_coordinates() {
        let root = temp_path("dted-out-of-range-coordinates");
        let mut terrain = DtedTerrain::new(&root);

        for (lon, lat, error) in [
            (
                -106.5,
                91.0,
                Error::InvalidInput("latitude_deg must be within [-90, 90]".to_string()),
            ),
            (
                -106.5,
                -90.5,
                Error::InvalidInput("latitude_deg must be within [-90, 90]".to_string()),
            ),
            (
                200.0,
                36.5,
                Error::InvalidInput("longitude_deg must be within [-180, 180]".to_string()),
            ),
            (
                -180.5,
                36.5,
                Error::InvalidInput("longitude_deg must be within [-180, 180]".to_string()),
            ),
        ] {
            assert_eq!(
                terrain
                    .height_m_with_options(lon, lat, DtedLookupOptions::default())
                    .expect_err("out-of-range DTED coordinate must error"),
                error
            );
        }

        assert_eq!(
            terrain
                .height_m(-106.5, 36.5)
                .expect("missing in-range tile keeps sea-level fallback"),
            0.0
        );
    }

    #[test]
    fn dted_valid_minimum_tile_parses_and_interpolates() {
        let root = temp_path("dted-valid-minimum");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let tile_path = root.join("n36_w107_1arc_v3.dt2");
        write_synthetic_dted_tile(&tile_path, 2, 2, |lon_index, lat_index| {
            match (lon_index, lat_index) {
                (0, 0) => 10,
                (0, 1) => 30,
                (1, 0) => 50,
                (1, 1) => 70,
                _ => unreachable!("2x2 synthetic tile"),
            }
        });

        DtedTile::from_path(&tile_path).expect("valid 2x2 DTED tile");
        let mut terrain = DtedTerrain::new(&root);
        assert_eq!(
            terrain
                .height_m_with_options(
                    -106.5,
                    36.5,
                    DtedLookupOptions {
                        interpolation: DtedInterpolation::Bilinear,
                    },
                )
                .expect("bilinear height"),
            40.0
        );

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    // Fixture provenance: `tests/fixtures/dted/tiles/n36_w107_1arc_v3.dt2` is a
    // synthetic public-format DTED tile written by the committed generator
    // `crates/sidereon-core/fixtures-generators/generate_dted_points.py` using the
    // DTED UHL/DSI/ACC/data-record layout (tile id `n36_w107`, elevation formula
    // `z_m = -20 + 7*lon_i - 5*lat_i + lon_i*lat_i`); no external terrain payload is
    // copied. `tests/fixtures/dted/dted_points.json` holds nearest-posting and
    // bilinear lookup cases generated from that tile. Floating-point fixture
    // values are serialized as f64 hex-bit strings and must be compared with
    // `f64::to_bits`, never tolerances.
    #[test]
    fn dted_lookup_matches_generated_fixture_bits() {
        let raw =
            std::fs::read_to_string(fixture_path("dted_points.json")).expect("read dted fixture");
        let doc: Value = serde_json::from_str(&raw).expect("parse dted fixture");
        assert_eq!(doc["schema"], "gnss-dted-points-v1");

        let root = copy_primary_fixture_root("dted-fixture-single-scalar");
        let mut terrain = DtedTerrain::new(&root);
        let nearest = DtedLookupOptions {
            interpolation: DtedInterpolation::NearestPosting,
        };
        let bilinear = DtedLookupOptions {
            interpolation: DtedInterpolation::Bilinear,
        };

        let mut checked = 0usize;
        for case in doc["nearest_cases"].as_array().expect("nearest_cases") {
            let lon = bits(&case["longitude_bits"]);
            let lat = bits(&case["latitude_bits"]);
            let got = terrain
                .height_m_with_options(lon, lat, nearest)
                .expect("nearest DTED height");
            let want = bits(&case["elevation_bits"]);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "nearest DTED {},{}",
                lon,
                lat
            );
            checked += 1;
        }

        for case in doc["bilinear_cases"].as_array().expect("bilinear_cases") {
            let lon = bits(&case["longitude_bits"]);
            let lat = bits(&case["latitude_bits"]);
            let got = terrain
                .height_m_with_options(lon, lat, bilinear)
                .expect("bilinear DTED height");
            let want = bits(&case["elevation_bits"]);
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "bilinear DTED {},{}",
                lon,
                lat
            );
            checked += 1;
        }
        assert!(checked > 0, "empty DTED fixture");
        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    #[test]
    fn height_batch_matches_scalar_loop_on_fixture_bits() {
        let raw =
            std::fs::read_to_string(fixture_path("dted_points.json")).expect("read dted fixture");
        let doc: Value = serde_json::from_str(&raw).expect("parse dted fixture");
        assert_eq!(doc["schema"], "gnss-dted-points-v1");

        let points: Vec<(f64, f64)> = ["nearest_cases", "bilinear_cases"]
            .into_iter()
            .flat_map(|cases_key| {
                doc[cases_key]
                    .as_array()
                    .expect(cases_key)
                    .iter()
                    .map(|case| (bits(&case["longitude_bits"]), bits(&case["latitude_bits"])))
            })
            .collect();

        for options in [
            DtedLookupOptions {
                interpolation: DtedInterpolation::NearestPosting,
            },
            DtedLookupOptions {
                interpolation: DtedInterpolation::Bilinear,
            },
        ] {
            let root = copy_primary_fixture_root("dted-fixture-single-batch");
            let want = scalar_loop(&root, &points, options);
            let mut terrain = DtedTerrain::new(&root);
            let got = terrain.height_batch(&points, options);
            assert_height_results_match(&got, &want, "single-tile fixture batch");
            fs::remove_dir_all(root).expect("remove temp DTED dir");
        }
    }

    #[test]
    fn height_batch_matches_scalar_loop_across_adjacent_tiles_bits() {
        let root = fixture_path("tiles");
        let options = DtedLookupOptions {
            interpolation: DtedInterpolation::Bilinear,
        };
        let raw =
            std::fs::read_to_string(fixture_path("dted_points.json")).expect("read dted fixture");
        let doc: Value = serde_json::from_str(&raw).expect("parse dted fixture");
        for case in doc["multi_tile_cases"]
            .as_array()
            .expect("multi_tile_cases")
        {
            let lon = bits(&case["longitude_bits"]);
            let lat = bits(&case["latitude_bits"]);
            let expected = bits(&case["bilinear_bits"]);
            let mut terrain = DtedTerrain::new(&root);
            let got = terrain
                .height_m_with_options(lon, lat, options)
                .expect("multi-tile generated bilinear height");
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "multi-tile generated case {}",
                case["case_id"].as_str().expect("case_id")
            );
        }

        let sequences = [
            (
                "all_in_a_then_all_in_b",
                vec![
                    (-106.875, 36.125),
                    (-106.625, 36.375),
                    (-105.875, 36.125),
                    (-105.625, 36.375),
                ],
            ),
            (
                "interleaved_a_b_a_b",
                vec![
                    (-106.875, 36.625),
                    (-105.875, 36.625),
                    (-106.625, 36.125),
                    (-105.625, 36.125),
                ],
            ),
            (
                "boundary_after_a_then_missing",
                vec![
                    (-106.875, 36.5),
                    (-106.0, 36.5),
                    (-104.5, 36.5),
                    (-105.875, 36.5),
                ],
            ),
        ];

        for (name, points) in sequences {
            let want = scalar_loop(&root, &points, options);
            let mut terrain = DtedTerrain::new(&root);
            let got = terrain.height_batch(&points, options);
            assert_height_results_match(&got, &want, name);
        }

        let mut terrain = DtedTerrain::new(&root);
        let missing = terrain.height_batch(&[(-104.5, 36.5)], options);
        assert_eq!(
            missing[0].as_ref().map(|v| v.to_bits()),
            Ok(0.0f64.to_bits())
        );
    }

    #[test]
    fn height_batch_places_errors_at_input_indices() {
        let root = temp_path("dted-batch-errors");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        copy_fixture_tile(&root, "n36_w107_1arc_v3.dt2");
        copy_fixture_tile(&root, "n36_w106_1arc_v3.dt2");
        fs::write(root.join("n37_w107_1arc_v3.dt2"), b"not a DTED tile")
            .expect("write corrupt DTED tile");

        let points = [
            (-106.875, 36.125),
            (-106.5, f64::NAN),
            (-105.875, 36.125),
            (-106.5, 37.5),
            (-106.625, 36.375),
        ];
        let options = DtedLookupOptions {
            interpolation: DtedInterpolation::Bilinear,
        };
        let want = scalar_loop(&root, &points, options);
        let mut terrain = DtedTerrain::new(&root);
        let got = terrain.height_batch(&points, options);
        assert_height_results_match(&got, &want, "batch error placement");

        assert!(got[0].is_ok(), "index 0 remains valid");
        assert_eq!(
            got[1],
            Err(Error::InvalidInput(
                "latitude_deg must be finite".to_string()
            ))
        );
        assert!(got[2].is_ok(), "index 2 remains valid");
        assert!(
            matches!(&got[3], Err(Error::Parse(msg)) if msg.contains("too short")),
            "index 3 is the corrupt-tile error: {:?}",
            got[3]
        );
        assert!(got[4].is_ok(), "index 4 remains valid");

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    const FIXTURE_COUNT: usize = 5;

    fn fixture_block_start(lon_index: usize) -> usize {
        DATA_OFFSET + lon_index * (12 + 2 * FIXTURE_COUNT)
    }

    /// Recompute the byte-sum checksum of one data record of the committed
    /// 5x5 fixture tile after its bytes were edited.
    fn rewrite_fixture_checksum(bytes: &mut [u8], lon_index: usize) {
        let start = fixture_block_start(lon_index);
        let checksum_start = start + 12 + 2 * FIXTURE_COUNT - 4;
        let sum = bytes[start..checksum_start]
            .iter()
            .fold(0i32, |acc, b| acc + i32::from(*b));
        bytes[checksum_start..checksum_start + 4].copy_from_slice(&sum.to_be_bytes());
    }

    fn primary_fixture_bytes() -> Vec<u8> {
        fs::read(fixture_path("tiles/n36_w107_1arc_v3.dt2")).expect("read DTED fixture tile")
    }

    /// The committed n36_w107 fixture with posting (lon 2, lat 3) replaced by
    /// the null bit pattern and that profile's checksum recomputed.
    fn fixture_with_null_posting() -> Vec<u8> {
        let mut bytes = primary_fixture_bytes();
        let sample = fixture_block_start(2) + 8 + 2 * 3;
        bytes[sample..sample + 2].copy_from_slice(&[0xFF, 0xFF]);
        rewrite_fixture_checksum(&mut bytes, 2);
        bytes
    }

    #[test]
    fn null_posting_is_an_unknown_elevation_not_a_height() {
        let root = temp_path("dted-null-posting");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let tile_path = root.join("n36_w107_1arc_v3.dt2");
        fs::write(&tile_path, fixture_with_null_posting()).expect("write null-posting tile");

        let tile = DtedTile::from_path(&tile_path).expect("null postings are valid DTED");
        assert_eq!(
            tile.get_elevation(-106.5, 36.75),
            Err(DtedTileError::NullPosting {
                longitude_index: 2,
                latitude_index: 3,
            })
        );
        // Posting (lon 3, lat 3) is -20 + 7*3 - 5*3 + 3*3 = -5.
        assert_eq!(tile.get_elevation(-106.25, 36.75), Ok(-5));

        let unknown = Err(Error::UnknownTerrainElevation {
            lat_index: 36,
            lon_index: -107,
            latitude_posting: 3,
            longitude_posting: 2,
        });
        let nearest = DtedLookupOptions {
            interpolation: DtedInterpolation::NearestPosting,
        };
        let bilinear = DtedLookupOptions {
            interpolation: DtedInterpolation::Bilinear,
        };
        let cases = [
            // The null posting itself, and a point that rounds to it.
            ((-106.5, 36.75), nearest, unknown.clone()),
            ((-106.52, 36.74), nearest, unknown.clone()),
            ((-106.5, 36.75), bilinear, unknown.clone()),
            // Interiors of cells with the null posting as a corner.
            ((-106.375, 36.625), bilinear, unknown.clone()),
            ((-106.625, 36.875), bilinear, unknown.clone()),
            ((-106.5, 36.7), bilinear, unknown.clone()),
            // A known posting next to it gives the null posting zero weight.
            ((-106.25, 36.75), bilinear, Ok(-5.0)),
            ((-106.25, 36.75), nearest, Ok(-5.0)),
            // A cell that does not touch it.
            ((-106.875, 36.125), nearest, Ok(-20.0)),
        ];
        let mut terrain = DtedTerrain::new(&root);
        for ((lon, lat), options, want) in &cases {
            assert_eq!(
                &terrain.height_m_with_options(*lon, *lat, *options),
                want,
                "height at ({lon}, {lat}) with {options:?}"
            );
        }
        for options in [nearest, bilinear] {
            let points = cases
                .iter()
                .filter(|(_, case_options, _)| *case_options == options)
                .map(|(point, _, _)| *point)
                .collect::<Vec<_>>();
            let want = cases
                .iter()
                .filter(|(_, case_options, _)| *case_options == options)
                .map(|(_, _, want)| want.clone())
                .collect::<Vec<_>>();
            assert_eq!(DtedTerrain::new(&root).height_batch(&points, options), want);
        }

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    /// Negative postings written in two's complement are read as GDAL reads
    /// them: a signed-magnitude value below -16000 m, other than the null, is
    /// reinterpreted as two's complement.
    #[test]
    fn twos_complement_negative_postings_are_read_as_gdal_reads_them() {
        use crate::terrain_store::{dted_tree_to_mmap_store, MmapTerrain};

        let root = temp_path("dted-twos-complement");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let mut bytes = primary_fixture_bytes();
        let block = fixture_block_start(1);
        for (lat_index, raw) in [
            // -5 in two's complement; as signed magnitude it would be -32763.
            (1, [0xFF, 0xFB]),
            // -16000 in signed magnitude, at the threshold: kept.
            (2, [0xBE, 0x80]),
            // Signed magnitude -16001, below the threshold: 0xBE81 in two's
            // complement is -16767.
            (3, [0xBE, 0x81]),
            // -5 in signed magnitude.
            (4, [0x80, 0x05]),
        ] {
            let sample = block + 8 + 2 * lat_index;
            bytes[sample..sample + 2].copy_from_slice(&raw);
        }
        rewrite_fixture_checksum(&mut bytes, 1);
        fs::write(root.join("n36_w107_1arc_v3.dt2"), bytes).expect("write tile");

        let tile = DtedTile::from_path(root.join("n36_w107_1arc_v3.dt2")).expect("tile reads");
        let store = dted_tree_to_mmap_store(&root).expect("convert tile");
        let mapped = MmapTerrain::from_bytes(&store).expect("parse store");
        let nearest = DtedLookupOptions {
            interpolation: DtedInterpolation::NearestPosting,
        };
        for (latitude, want) in [(36.25, -5), (36.5, -16000), (36.75, -16767), (37.0, -5)] {
            assert_eq!(
                tile.get_elevation(-106.75, latitude),
                Ok(want),
                "{latitude}"
            );
            assert_eq!(
                mapped
                    .orthometric_height_m_with_options(-106.75, latitude, nearest)
                    .map(|height| height.metres()),
                Ok(f64::from(want)),
                "{latitude}"
            );
        }

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    #[test]
    fn edge_postings_map_to_coincident_neighbour_postings_in_integers() {
        let tile = |lat_index, lon_index, lon_count, lat_count| TileGrid {
            lat_index,
            lon_index,
            lon_count,
            lat_count,
        };
        let east = tile(36, -106, 5, 5);
        // Interior postings have no neighbours; edges have one; corners three.
        assert!(edge_neighbour_grids(east, 2, 2).is_empty());
        assert_eq!(edge_neighbour_grids(east, 0, 2), vec![(36, -107)]);
        assert_eq!(edge_neighbour_grids(east, 4, 2), vec![(36, -105)]);
        assert_eq!(edge_neighbour_grids(east, 2, 4), vec![(37, -106)]);
        assert_eq!(
            edge_neighbour_grids(east, 0, 0),
            vec![(36, -107), (35, -106), (35, -107)]
        );

        let west = tile(36, -107, 5, 5);
        assert_eq!(coincident_posting(east, 0, 2, west), Some((4, 2)));
        assert_eq!(
            coincident_posting(east, 0, 0, tile(35, -107, 5, 5)),
            Some((4, 4))
        );
        // A neighbour with twice the longitude interval along a parallel edge
        // shares every other posting.
        let north_coarse = tile(37, -106, 3, 5);
        assert_eq!(coincident_posting(east, 2, 4, north_coarse), Some((1, 0)));
        assert_eq!(coincident_posting(east, 1, 4, north_coarse), None);
        // Postings off the neighbour's extent have no counterpart.
        assert_eq!(coincident_posting(east, 2, 2, west), None);
    }

    fn origin_result(longitude: &[u8; 8], latitude: &[u8; 8]) -> Result<(), DtedTileError> {
        let root = temp_path("dted-origin-field");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let tile_path = root.join("tile.dt2");
        write_synthetic_dted_tile_at(&tile_path, longitude, latitude, 2, 2, |_, _| 0);
        let result = DtedTile::from_path(&tile_path).map(|_| ());
        fs::remove_dir_all(root).expect("remove temp DTED dir");
        result
    }

    #[test]
    fn uhl_origin_fields_are_validated_per_axis() {
        let out_of_range = |field: &'static str, text: &str| -> Result<(), DtedTileError> {
            Err(DtedTileError::CoordinateOutOfRange {
                field,
                text: text.to_string(),
            })
        };
        let lon = "longitude of origin";
        let lat = "latitude of origin";

        assert_eq!(
            origin_result(b"0006000E", b"0360000N"),
            out_of_range(lon, "0006000E")
        );
        assert_eq!(
            origin_result(b"1070060W", b"0360000N"),
            out_of_range(lon, "1070060W")
        );
        assert_eq!(
            origin_result(b"1810000W", b"0360000N"),
            out_of_range(lon, "1810000W")
        );
        assert_eq!(
            origin_result(b"1800000E", b"0360000N"),
            out_of_range(lon, "1800000E")
        );
        assert_eq!(
            origin_result(b"1070000W", b"0900000N"),
            out_of_range(lat, "0900000N")
        );
        assert_eq!(
            origin_result(b"1070000W", b"0910000S"),
            out_of_range(lat, "0910000S")
        );
        assert_eq!(
            origin_result(b"1070000W", b"0360000E"),
            Err(DtedTileError::WrongHemisphere {
                field: lat,
                hemisphere: 'E',
                expected: "N or S",
            })
        );
        assert_eq!(
            origin_result(b"1070000N", b"0360000N"),
            Err(DtedTileError::WrongHemisphere {
                field: lon,
                hemisphere: 'N',
                expected: "E or W",
            })
        );
        assert_eq!(
            origin_result(b"1070030W", b"0360000N"),
            Err(DtedTileError::OriginNotWholeDegree {
                field: lon,
                text: "1070030W".to_string(),
            })
        );
        assert!(matches!(
            origin_result(b"-070000W", b"0360000N"),
            Err(DtedTileError::InvalidField(_))
        ));

        // Every whole-degree tile origin on both axes is accepted, including
        // the edges of the coordinate domain.
        for (longitude, latitude) in [
            (b"1800000W", b"0900000S"),
            (b"1790000E", b"0890000N"),
            (b"0000000E", b"0000000N"),
            (b"0000000W", b"0000000S"),
        ] {
            assert_eq!(origin_result(longitude, latitude), Ok(()));
        }
    }

    fn with_uhl_patch(range: std::ops::Range<usize>, text: &[u8]) -> Result<(), DtedTileError> {
        let root = temp_path("dted-uhl-patch");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let tile_path = root.join("n36_w107_1arc_v3.dt2");
        let mut bytes = primary_fixture_bytes();
        bytes[range].copy_from_slice(text);
        fs::write(&tile_path, bytes).expect("write patched tile");
        let result = DtedTile::from_path(&tile_path).map(|_| ());
        fs::remove_dir_all(root).expect("remove temp DTED dir");
        result
    }

    #[test]
    fn uhl_intervals_must_span_one_degree_over_the_counts() {
        // Five postings at 900 arc seconds (9000 tenths) span one degree.
        assert_eq!(with_uhl_patch(20..28, b"90009000"), Ok(()));
        // The blank intervals of the committed fixture state nothing.
        assert_eq!(with_uhl_patch(20..28, b"        "), Ok(()));
        assert_eq!(
            with_uhl_patch(20..28, b"90000030"),
            Err(DtedTileError::IntervalCountMismatch {
                field: "latitude data interval",
                interval_tenths_arcsec: 30,
                count: 5,
            })
        );
        assert_eq!(
            with_uhl_patch(20..24, b"0010"),
            Err(DtedTileError::IntervalCountMismatch {
                field: "longitude data interval",
                interval_tenths_arcsec: 10,
                count: 5,
            })
        );
        assert!(matches!(
            with_uhl_patch(20..24, b"9X00"),
            Err(DtedTileError::InvalidField(_))
        ));
    }

    fn tile_with_datum(root: &Path, datum: &[u8; 5]) -> PathBuf {
        let tile_path = root.join("n36_w107_1arc_v3.dt2");
        let mut bytes = primary_fixture_bytes();
        bytes[224..229].copy_from_slice(datum);
        fs::write(&tile_path, bytes).expect("write datum tile");
        tile_path
    }

    #[test]
    fn dsi_horizontal_datum_is_kept_on_the_tile_and_refused_for_wgs84_queries() {
        let cases = [
            (b"WGS84", DtedHorizontalDatum::Wgs84),
            (b"wgs84", DtedHorizontalDatum::Wgs84),
            (b"     ", DtedHorizontalDatum::Unstated),
            (b"\0\0\0\0\0", DtedHorizontalDatum::Unstated),
            (b"WGS72", DtedHorizontalDatum::Wgs72),
            (b"NAD27", DtedHorizontalDatum::Other("NAD27".to_string())),
        ];
        for (field, datum) in cases {
            let root = temp_path("dted-datum");
            fs::create_dir_all(&root).expect("create temp DTED dir");
            let tile_path = tile_with_datum(&root, field);

            // The tile itself reads, whatever datum it states.
            let tile = DtedTile::from_path(&tile_path).expect("tile reads");
            assert_eq!(tile.horizontal_datum(), &datum);
            assert_eq!(tile.get_elevation(-107.0, 36.0), Ok(-20));

            // A WGS84 query is answered only from a WGS84 or blank tile.
            let got = DtedTerrain::new(&root).height_m(-106.875, 36.125);
            if datum.is_wgs84_compatible() {
                assert!(got.is_ok(), "{datum:?}: {got:?}");
            } else {
                assert_eq!(
                    got,
                    Err(Error::NonWgs84TerrainTile {
                        lat_index: 36,
                        lon_index: -107,
                        datum: datum.clone(),
                    })
                );
            }
            fs::remove_dir_all(root).expect("remove temp DTED dir");
        }
    }

    #[test]
    fn swapped_profiles_are_refused_by_their_longitude_counts() {
        let root = temp_path("dted-swapped-profiles");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let tile_path = root.join("n36_w107_1arc_v3.dt2");
        let mut bytes = primary_fixture_bytes();
        let block_len = 12 + 2 * FIXTURE_COUNT;
        let one = fixture_block_start(1);
        let two = fixture_block_start(2);
        let first = bytes[one..one + block_len].to_vec();
        let second = bytes[two..two + block_len].to_vec();
        bytes[one..one + block_len].copy_from_slice(&second);
        bytes[two..two + block_len].copy_from_slice(&first);
        fs::write(&tile_path, bytes).expect("write swapped tile");

        let tile = DtedTile::from_path(&tile_path).expect("headers are intact");
        assert_eq!(
            tile.get_elevation(-106.75, 36.0),
            Err(DtedTileError::ProfileLongitudeCountMismatch {
                longitude_index: 1,
                declared: 2,
            })
        );
        assert_eq!(
            tile.get_elevation(-106.5, 36.0),
            Err(DtedTileError::ProfileLongitudeCountMismatch {
                longitude_index: 2,
                declared: 1,
            })
        );
        assert_eq!(tile.get_elevation(-107.0, 36.0), Ok(-20));
        let err = DtedTerrain::new(&root)
            .height_m(-106.625, 36.5)
            .expect_err("a swapped profile must not be read");
        assert!(
            matches!(&err, Error::Parse(msg) if msg.contains("declares longitude count")),
            "{err:?}"
        );

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    #[test]
    fn partial_profiles_are_refused_by_name() {
        let root = temp_path("dted-partial-profile");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let tile_path = root.join("n36_w107_1arc_v3.dt2");
        let mut bytes = primary_fixture_bytes();
        let start = fixture_block_start(0);
        bytes[start + 6..start + 8].copy_from_slice(&1u16.to_be_bytes());
        rewrite_fixture_checksum(&mut bytes, 0);
        fs::write(&tile_path, bytes).expect("write partial-profile tile");

        let tile = DtedTile::from_path(&tile_path).expect("headers are intact");
        assert_eq!(
            tile.get_elevation(-107.0, 36.5),
            Err(DtedTileError::UnsupportedPartialProfile {
                longitude_index: 0,
                first_latitude_index: 1,
            })
        );
        assert_eq!(tile.get_elevation(-106.75, 36.0), Ok(-13));

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    #[test]
    fn a_tile_whose_origin_disagrees_with_its_name_is_refused() {
        let root = temp_path("dted-origin-name");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        write_synthetic_dted_tile_at(
            &root.join(format!("n36_w107{DTED_SUFFIX}")),
            b"1060000W",
            b"0360000N",
            2,
            2,
            |_, _| 7,
        );
        let err = DtedTerrain::new(&root)
            .height_m(-106.5, 36.5)
            .expect_err("a misnamed tile must not read as sea level");
        assert!(
            matches!(&err, Error::Parse(msg) if msg.contains("does not match tile (36,-107)")),
            "{err:?}"
        );
        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }

    /// The mapped store's bilinear lookup locates the cell exactly as the raw
    /// reader does. The probes are the exact-fraction coordinates of
    /// `bilinear_cell_offset_is_exact_in_every_tile`, which carry bits below
    /// one ulp of 1; subtracting the tile origin directly rounds them away in
    /// the tile at -1, so a store that did so would disagree with the raw
    /// reader and with the pinned heights.
    #[test]
    fn mapped_bilinear_lookup_matches_raw_dted_at_exact_fractions() {
        use crate::terrain_store::{dted_tile_list_to_mmap_store, DtedTileListEntry, MmapTerrain};

        let root = temp_path("dted-mapped-parity");
        fs::create_dir_all(&root).expect("create temp DTED dir");
        let postings = 1200;
        let checkerboard = |lon_index: usize, lat_index: usize| {
            if (lon_index + lat_index).is_multiple_of(2) {
                0
            } else {
                8849
            }
        };
        let tiles: [(&[u8; 8], &[u8; 8], i32, i32); 4] = [
            (b"0010000W", b"0010000S", -1, -1),
            (b"0000000E", b"0000000N", 0, 0),
            (b"0010000W", b"0510000N", 51, -1),
            (b"1070000W", b"0360000N", 36, -107),
        ];
        let mut entries = Vec::new();
        for (longitude, latitude, lat_index, lon_index) in tiles {
            let name = format!(
                "{}_{}{DTED_SUFFIX}",
                super::format_lat(lat_index),
                super::format_lon(lon_index)
            );
            let path = root.join(name);
            write_synthetic_dted_tile_at(
                &path,
                longitude,
                latitude,
                postings + 1,
                postings + 1,
                checkerboard,
            );
            entries.push(DtedTileListEntry::from_indices(lat_index, lon_index, path));
        }
        let store = dted_tile_list_to_mmap_store(&entries).expect("build store");
        let mapped = MmapTerrain::from_bytes(&store).expect("parse store");
        let mut raw = DtedTerrain::new(&root);
        let bilinear = DtedLookupOptions {
            interpolation: DtedInterpolation::Bilinear,
        };

        // (coordinate bits, tile origin) from the exact cell-offset cases.
        const COORDINATES: &[(u64, f64)] = &[
            (0xbf1a36e2eb1c432d, -1.0),
            (0xbf1a36e2eb1c432c, -1.0),
            (0xbfd73a99165fe501, -1.0),
            (0xbfdfedcba9876543, -1.0),
            (0xbfeffffffff24190, -1.0),
            (0xbd719799812dea11, -1.0),
            (0xbfe8000000000001, -1.0),
            (0xbfd0000000000002, -1.0),
            (0xbfeccccccccccccd, -1.0),
            (0x3f1a36e2eb1c432d, 0.0),
            (0x3fd73a99165fe501, 0.0),
            (0x3fdfedcba9876543, 0.0),
            (0x3feffffffff24190, 0.0),
            (0x3d719799812dea11, 0.0),
            (0x3fd0000000000002, 0.0),
            (0x4049800346dc5d64, 51.0),
            (0x4049ae75322cbfca, 51.0),
            (0x4049bfdb97530ecb, 51.0),
            (0x4049ffffffffc906, 51.0),
            (0xc05a8001a36e2eb2, -107.0),
            (0xc05a973a99165fe5, -107.0),
            (0xc05abfedcba98765, -107.0),
        ];
        let mut points = Vec::new();
        for &(bits, origin) in COORDINATES {
            let coordinate = f64::from_bits(bits);
            // Both axes in the tiles at (-1, -1) and (0, 0); latitude in the
            // tile at (51, -1); longitude in the tile at (36, -107).
            let point = if origin == -1.0 || origin == 0.0 {
                (coordinate, coordinate)
            } else if origin == 51.0 {
                (-0.5, coordinate)
            } else {
                (coordinate, 36.5)
            };
            points.push(point);
        }
        // Pinned heights of `bilinear_height_is_exact_south_of_equator_and_west_of_meridian`.
        const PINNED: &[(u64, u64, u64)] = &[
            (0xbf1a36e2eb1c432d, 0xbf1a36e2eb1c432c, 0x409d33a29c779a6b),
            (0xbfd73a99165fe501, 0xbfd73a99165fe500, 0x40b129828ba475c4),
            (0xbfdfedcba9876543, 0xbfdfedcba9876542, 0x40aeb9c71c71c93c),
            (0xbfeffffffff24190, 0xbfeffffffff920c8, 0x3f5a18c574f03816),
            (0xbd3c25c268497682, 0xbd4c25c268497682, 0x3ecab91c416da2d4),
        ];
        for &(lon_bits, lat_bits, height_bits) in PINNED {
            let point = (f64::from_bits(lon_bits), f64::from_bits(lat_bits));
            let height = mapped
                .orthometric_height_m_with_options(point.0, point.1, bilinear)
                .expect("mapped pinned height")
                .metres();
            assert_eq!(height.to_bits(), height_bits, "mapped height at {point:?}");
            points.push(point);
        }

        for (lon, lat) in points {
            let raw_height = raw
                .height_m_with_options(lon, lat, bilinear)
                .expect("raw bilinear height");
            let mapped_height = mapped
                .orthometric_height_m_with_options(lon, lat, bilinear)
                .expect("mapped bilinear height")
                .metres();
            assert_eq!(
                mapped_height.to_bits(),
                raw_height.to_bits(),
                "raw/mapped bilinear at ({lon:e}, {lat:e})"
            );
        }

        fs::remove_dir_all(root).expect("remove temp DTED dir");
    }
}
