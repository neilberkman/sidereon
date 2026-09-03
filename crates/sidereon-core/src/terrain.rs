//! DTED tile reader and bilinear terrain lookup.

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
    Io { path: String, message: String },
    /// The tile does not contain the fixed DTED header area.
    #[error("{path} is too short for DTED headers")]
    TooShort { path: String },
    /// The tile does not start with the DTED UHL1 marker.
    #[error("{path} missing UHL1 header")]
    MissingUhl1 { path: String },
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
        path: String,
        lon_count: usize,
        lat_count: usize,
    },
    /// The tile ends before its declared data blocks end.
    #[error("{path} has {actual} bytes but expected at least {expected}")]
    Truncated {
        path: String,
        actual: usize,
        expected: usize,
    },
    /// A query is outside this tile's one-degree extent.
    #[error("point ({longitude},{latitude}) is outside DTED tile ({origin_longitude},{origin_latitude})")]
    Outside {
        longitude: f64,
        latitude: f64,
        origin_longitude: f64,
        origin_latitude: f64,
    },
    /// A rounded query did not map to a declared posting.
    #[error("posting index out of bounds lon={longitude_index} lat={latitude_index}")]
    PostingIndexOutOfBounds {
        longitude_index: usize,
        latitude_index: usize,
    },
    /// A DTED data block is missing its sentinel byte.
    #[error("DTED block {longitude_index} missing data sentinel")]
    MissingDataSentinel { longitude_index: usize },
    /// A DTED data block checksum does not match its contents.
    #[error("DTED checksum failed for block {longitude_index}: expected {checksum}, found {sum}")]
    Checksum {
        longitude_index: usize,
        checksum: i32,
        sum: i32,
    },
    /// A coordinate field is empty.
    #[error("empty DTED coordinate")]
    EmptyCoordinate,
    /// A coordinate field has an unsupported hemisphere suffix.
    #[error("invalid DTED hemisphere {hemisphere}")]
    InvalidHemisphere { hemisphere: char },
    /// The rounded coordinate is negative and cannot be a posting index.
    #[error("cannot round negative posting index {index}")]
    NegativePostingIndex { index: i64 },
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
    pub fn height_m_with_options(
        &mut self,
        longitude_deg: f64,
        latitude_deg: f64,
        options: DtedLookupOptions,
    ) -> crate::Result<f64> {
        validate_lookup_coordinates(longitude_deg, latitude_deg)?;
        let Some(tile) = self.load_tile(longitude_deg, latitude_deg)? else {
            return Ok(0.0);
        };
        height_from_tile(tile, longitude_deg, latitude_deg, options)
    }

    /// Evaluate `(longitude_deg, latitude_deg)` points in order using one
    /// mutable borrow of the resident tile cache.
    ///
    /// The tuple order is intentionally longitude-first, matching
    /// [`Self::height_m_with_options`], even though geoid batch helpers use
    /// latitude-first points.
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

            let primary_grid = terrain_grid(longitude_deg, latitude_deg);
            if current == Some(primary_grid) {
                if let Some(tile) = self.tiles.get(&primary_grid) {
                    if tile.contains(longitude_deg, latitude_deg) {
                        out.push(height_from_tile(tile, longitude_deg, latitude_deg, options));
                        continue;
                    }
                }
            }

            match self.resolve_grid(longitude_deg, latitude_deg) {
                Ok(Some(grid_idx)) => {
                    current = Some(grid_idx);
                    let Some(tile) = self.tiles.get(&grid_idx) else {
                        out.push(Err(Error::Parse(
                            "resolved DTED grid is missing from the tile cache".to_string(),
                        )));
                        continue;
                    };
                    out.push(height_from_tile(tile, longitude_deg, latitude_deg, options));
                }
                Ok(None) => {
                    current = None;
                    out.push(Ok(0.0));
                }
                Err(err) => out.push(Err(err)),
            }
        }

        out
    }

    fn load_tile(&mut self, longitude: f64, latitude: f64) -> crate::Result<Option<&DtedTile>> {
        let Some(grid_idx) = self.resolve_grid(longitude, latitude)? else {
            return Ok(None);
        };
        Ok(self.tiles.get(&grid_idx))
    }

    fn resolve_grid(&mut self, longitude: f64, latitude: f64) -> crate::Result<Option<(i32, i32)>> {
        for grid_idx in terrain_grid_candidates(longitude, latitude) {
            if !self.tiles.contains_key(&grid_idx) {
                let Some(path) = self.terrain_path_for_grid(grid_idx.0, grid_idx.1) else {
                    continue;
                };
                if !path.is_file() {
                    continue;
                }
                let tile =
                    DtedTile::from_path(path).map_err(|error| Error::Parse(error.to_string()))?;
                self.tiles.insert(grid_idx, tile);
            }
            if let Some(tile) = self.tiles.get(&grid_idx) {
                if tile.contains(longitude, latitude) {
                    return Ok(Some(grid_idx));
                }
            }
        }
        Ok(None)
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
            .map(|v| v as f64)
            .map_err(|error| Error::Parse(error.to_string()));
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
                    .map_err(|error| Error::Parse(error.to_string()))?,
            );
        }
    }
    Ok(z)
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

/// Parsed DTED tile backed by raw `.dt2` bytes.
///
/// Posting values are decoded lazily from DTED signed-magnitude samples.
/// Returned heights are orthometric metres, `H`, above the EGM96 mean sea level
/// geoid.
#[derive(Debug)]
pub struct DtedTile {
    origin_latitude: f64,
    origin_longitude: f64,
    lon_count: usize,
    lat_count: usize,
    data_block_length: usize,
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

        let origin_longitude = parse_dted_coord(
            std::str::from_utf8(&bytes[4..12])
                .map_err(|error| DtedTileError::InvalidEncoding(error.to_string()))?,
        )?;
        let origin_latitude = parse_dted_coord(
            std::str::from_utf8(&bytes[12..20])
                .map_err(|error| DtedTileError::InvalidEncoding(error.to_string()))?,
        )?;
        let lon_count = parse_ascii_usize(&bytes[47..51])?;
        let lat_count = parse_ascii_usize(&bytes[51..55])?;
        if lon_count < 2 || lat_count < 2 {
            return Err(DtedTileError::InvalidDimensions {
                path: path_display,
                lon_count,
                lat_count,
            });
        }
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
            bytes,
        })
    }

    /// Return the nearest orthometric posting value in metres for a
    /// longitude-first geodetic position in degrees.
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

        let block = self.validated_block(longitude_index)?;

        let sample_start = 8 + latitude_index * 2;
        let raw = i16::from_be_bytes([block[sample_start], block[sample_start + 1]]);
        Ok(convert_signed_magnitude(raw))
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

    pub(crate) fn decoded_postings_lon_major(&self) -> Result<Vec<i16>, DtedTileError> {
        let mut out = Vec::with_capacity(self.lon_count * self.lat_count);
        for longitude_index in 0..self.lon_count {
            let block = self.validated_block(longitude_index)?;
            for latitude_index in 0..self.lat_count {
                let sample_start = 8 + latitude_index * 2;
                let raw = i16::from_be_bytes([block[sample_start], block[sample_start + 1]]);
                out.push(convert_signed_magnitude(raw));
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
        Ok(block)
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

fn parse_dted_coord(input: &str) -> Result<f64, DtedTileError> {
    let (hemi_start, hemi) = input
        .char_indices()
        .last()
        .ok_or(DtedTileError::EmptyCoordinate)?;
    let sign = match hemi {
        'S' | 'W' => -1.0,
        'N' | 'E' => 1.0,
        _ => return Err(DtedTileError::InvalidHemisphere { hemisphere: hemi }),
    };
    let coord = &input[..hemi_start];
    if !coord.is_ascii() {
        return Err(DtedTileError::InvalidField(
            "invalid DTED coordinate".to_string(),
        ));
    }
    let seconds_index = if coord.as_bytes().get(coord.len().saturating_sub(2)) == Some(&b'.') {
        coord.len().checked_sub(4)
    } else {
        coord.len().checked_sub(2)
    }
    .ok_or_else(|| DtedTileError::InvalidField("invalid DTED coordinate".to_string()))?;
    let minutes_index = seconds_index
        .checked_sub(2)
        .filter(|&index| index > 0)
        .ok_or_else(|| DtedTileError::InvalidField("invalid DTED coordinate".to_string()))?;
    let degree = coord[..minutes_index]
        .parse::<i32>()
        .map_err(|error| DtedTileError::InvalidField(error.to_string()))?;
    let minute = coord[minutes_index..seconds_index]
        .parse::<i32>()
        .map_err(|error| DtedTileError::InvalidField(error.to_string()))?;
    let second = coord[seconds_index..]
        .parse::<f64>()
        .map_err(|error| DtedTileError::InvalidField(error.to_string()))?;
    Ok(sign * (degree as f64 + ((minute as f64 + second / 60.0) / 60.0)))
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

fn convert_signed_magnitude(raw: i16) -> i16 {
    if raw < 0 {
        (-32768i32 - i32::from(raw)) as i16
    } else {
        raw
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

    use super::{
        in_tile_cell_fraction, nearest_posting_index, scaled_cell_fraction, terrain_block_dir,
        DtedInterpolation, DtedLookupOptions, DtedTerrain, DtedTile, DtedTileError, DATA_OFFSET,
        DATA_SENTINEL, DTED_SUFFIX,
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
}
