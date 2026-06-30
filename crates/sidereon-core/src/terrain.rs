//! DTED tile reader and bilinear terrain lookup.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use crate::Error;

const UHL_SIZE: usize = 80;
const DSI_SIZE: usize = 648;
const ACC_SIZE: usize = 2700;
const DATA_OFFSET: usize = UHL_SIZE + DSI_SIZE + ACC_SIZE;
const DATA_SENTINEL: u8 = 0xAA;
const DTED_SUFFIX: &str = concat!("_1arc_v3.d", "t", "2");
const MIN_LOOKUP_LATITUDE_DEG: f64 = -90.0;
const MAX_LOOKUP_LATITUDE_DEG: f64 = 90.0;
const MIN_LOOKUP_LONGITUDE_DEG: f64 = -180.0;
const MAX_LOOKUP_LONGITUDE_DEG: f64 = 180.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtedInterpolation {
    NearestPosting,
    Bilinear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DtedLookupOptions {
    pub interpolation: DtedInterpolation,
}

impl Default for DtedLookupOptions {
    fn default() -> Self {
        Self {
            interpolation: DtedInterpolation::Bilinear,
        }
    }
}

#[derive(Debug)]
pub struct DtedTerrain {
    root: PathBuf,
    tiles: HashMap<(i32, i32), DtedTile>,
}

impl DtedTerrain {
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self {
            root: root.into(),
            tiles: HashMap::new(),
        }
    }

    pub fn height_m(&mut self, longitude_deg: f64, latitude_deg: f64) -> crate::Result<f64> {
        self.height_m_with_options(longitude_deg, latitude_deg, DtedLookupOptions::default())
    }

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
        if options.interpolation == DtedInterpolation::NearestPosting {
            return tile
                .get_elevation(longitude_deg, latitude_deg)
                .map(|v| v as f64)
                .map_err(Error::Parse);
        }

        let postings_per_deg_lon = tile.lon_count - 1;
        let postings_per_deg_lat = tile.lat_count - 1;

        let lon_idx = (longitude_deg - tile.origin_longitude) * postings_per_deg_lon as f64;
        let lat_idx = (latitude_deg - tile.origin_latitude) * postings_per_deg_lat as f64;
        let lon_lo = lon_idx.floor() as i64;
        let lat_lo = lat_idx.floor() as i64;
        let fx = lon_idx - lon_lo as f64;
        let fy = lat_idx - lat_lo as f64;

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
                        .map_err(Error::Parse)?,
                );
            }
        }
        Ok(z)
    }

    fn load_tile(&mut self, longitude: f64, latitude: f64) -> crate::Result<Option<&DtedTile>> {
        let mut selected = None;
        for grid_idx in terrain_grid_candidates(longitude, latitude) {
            if !self.tiles.contains_key(&grid_idx) {
                let Some(path) = self.terrain_path_for_grid(grid_idx.0, grid_idx.1) else {
                    continue;
                };
                if !path.is_file() {
                    continue;
                }
                let tile = DtedTile::from_path(path).map_err(Error::Parse)?;
                self.tiles.insert(grid_idx, tile);
            }
            if let Some(tile) = self.tiles.get(&grid_idx) {
                if tile.contains(longitude, latitude) {
                    selected = Some(grid_idx);
                    break;
                }
            }
        }
        Ok(selected.and_then(|grid_idx| self.tiles.get(&grid_idx)))
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

fn validate_lookup_coordinates(longitude_deg: f64, latitude_deg: f64) -> crate::Result<()> {
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
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, String> {
        let bytes =
            fs::read(path.as_ref()).map_err(|e| format!("{}: {e}", path.as_ref().display()))?;
        if bytes.len() < DATA_OFFSET {
            return Err(format!(
                "{} is too short for DTED headers",
                path.as_ref().display()
            ));
        }
        if &bytes[0..4] != b"UHL1" {
            return Err(format!("{} missing UHL1 header", path.as_ref().display()));
        }

        let origin_longitude =
            parse_dted_coord(std::str::from_utf8(&bytes[4..12]).map_err(|e| e.to_string())?)?;
        let origin_latitude =
            parse_dted_coord(std::str::from_utf8(&bytes[12..20]).map_err(|e| e.to_string())?)?;
        let lon_count = parse_ascii_usize(&bytes[47..51])?;
        let lat_count = parse_ascii_usize(&bytes[51..55])?;
        if lon_count < 2 || lat_count < 2 {
            return Err(format!(
                "{} has invalid DTED dimensions lon_count={} lat_count={}; both must be at least 2",
                path.as_ref().display(),
                lon_count,
                lat_count
            ));
        }
        let data_block_length = 12 + 2 * lat_count;
        let expected_len = DATA_OFFSET + lon_count * data_block_length;
        if bytes.len() < expected_len {
            return Err(format!(
                "{} has {} bytes but expected at least {}",
                path.as_ref().display(),
                bytes.len(),
                expected_len
            ));
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

    pub fn get_elevation(&self, longitude: f64, latitude: f64) -> Result<i16, String> {
        if !self.contains(longitude, latitude) {
            return Err(format!(
                "point ({longitude},{latitude}) is outside DTED tile ({},{})",
                self.origin_longitude, self.origin_latitude
            ));
        }

        let latitude_index =
            py_round_to_usize((latitude - self.origin_latitude) * (self.lat_count - 1) as f64)?;
        let longitude_index =
            py_round_to_usize((longitude - self.origin_longitude) * (self.lon_count - 1) as f64)?;
        if latitude_index >= self.lat_count || longitude_index >= self.lon_count {
            return Err(format!(
                "posting index out of bounds lon={longitude_index} lat={latitude_index}"
            ));
        }

        let block_start = DATA_OFFSET + longitude_index * self.data_block_length;
        let block_end = block_start + self.data_block_length;
        let block = &self.bytes[block_start..block_end];
        if block[0] != DATA_SENTINEL {
            return Err(format!(
                "DTED block {longitude_index} missing data sentinel"
            ));
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
            return Err(format!(
                "DTED checksum failed for block {longitude_index}: expected {checksum}, found {sum}"
            ));
        }

        let sample_start = 8 + latitude_index * 2;
        let raw = i16::from_be_bytes([block[sample_start], block[sample_start + 1]]);
        Ok(convert_signed_magnitude(raw))
    }

    fn contains(&self, longitude: f64, latitude: f64) -> bool {
        latitude >= self.origin_latitude
            && latitude <= self.origin_latitude + 1.0
            && longitude >= self.origin_longitude
            && longitude <= self.origin_longitude + 1.0
    }
}

fn terrain_grid(longitude: f64, latitude: f64) -> (i32, i32) {
    (latitude.floor() as i32, longitude.floor() as i32)
}

fn terrain_grid_candidates(longitude: f64, latitude: f64) -> Vec<(i32, i32)> {
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

fn format_lat(latitude_index: i32) -> String {
    if latitude_index >= 0 {
        format!("n{latitude_index:02}")
    } else {
        format!("s{:02}", -latitude_index)
    }
}

fn format_lon(longitude_index: i32) -> String {
    if longitude_index >= 0 {
        format!("e{longitude_index:03}")
    } else {
        format!("w{:03}", -longitude_index)
    }
}

fn terrain_block_dir(latitude_index: i32, longitude_index: i32) -> String {
    format!(
        "{}_{}",
        format_lat(block_origin(latitude_index)),
        format_lon(block_origin(longitude_index))
    )
}

fn block_origin(index: i32) -> i32 {
    index.div_euclid(10) * 10
}

fn parse_ascii_usize(bytes: &[u8]) -> Result<usize, String> {
    std::str::from_utf8(bytes)
        .map_err(|e| e.to_string())?
        .trim()
        .parse::<usize>()
        .map_err(|e| e.to_string())
}

fn parse_dted_coord(input: &str) -> Result<f64, String> {
    let hemi = input
        .chars()
        .last()
        .ok_or_else(|| "empty DTED coordinate".to_string())?;
    let sign = match hemi {
        'S' | 'W' => -1.0,
        'N' | 'E' => 1.0,
        _ => return Err(format!("invalid DTED hemisphere {hemi}")),
    };
    let coord = &input[..input.len() - 1];
    let seconds_index = if coord.as_bytes().get(coord.len().saturating_sub(2)) == Some(&b'.') {
        coord.len() - 4
    } else {
        coord.len() - 2
    };
    let minutes_index = seconds_index - 2;
    let degree = coord[..minutes_index]
        .parse::<i32>()
        .map_err(|e| e.to_string())?;
    let minute = coord[minutes_index..seconds_index]
        .parse::<i32>()
        .map_err(|e| e.to_string())?;
    let second = coord[seconds_index..]
        .parse::<f64>()
        .map_err(|e| e.to_string())?;
    Ok(sign * (degree as f64 + ((minute as f64 + second / 60.0) / 60.0)))
}

fn py_round_to_usize(value: f64) -> Result<usize, String> {
    if value < 0.0 {
        return Err(format!("cannot round negative posting index {value}"));
    }
    let lo = value.floor();
    let frac = value - lo;
    let rounded = if frac < 0.5 {
        lo
    } else if frac > 0.5 {
        lo + 1.0
    } else {
        let lo_i = lo as u64;
        if lo_i.is_multiple_of(2) {
            lo
        } else {
            lo + 1.0
        }
    };
    Ok(rounded as usize)
}

fn convert_signed_magnitude(raw: i16) -> i16 {
    if raw < 0 {
        (-32768i32 - i32::from(raw)) as i16
    } else {
        raw
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    use serde_json::Value;

    use crate::test_parity::f64_from_hex;
    use crate::Error;

    use super::{
        terrain_block_dir, DtedInterpolation, DtedLookupOptions, DtedTerrain, DtedTile,
        DATA_OFFSET, DATA_SENTINEL,
    };

    #[test]
    fn terrain_block_dir_matches_reference_bucket_names() {
        assert_eq!(terrain_block_dir(36, -107), "n30_w110");
        assert_eq!(terrain_block_dir(32, -117), "n30_w120");
        assert_eq!(terrain_block_dir(43, -112), "n40_w120");
        assert_eq!(terrain_block_dir(20, -103), "n20_w110");
        assert_eq!(terrain_block_dir(36, 107), "n30_e100");
        assert_eq!(terrain_block_dir(-1, -1), "s10_w010");
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
        let tile_dir = root.join("s10_w010");
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

    fn write_synthetic_dted_tile(
        path: &Path,
        lon_count: usize,
        lat_count: usize,
        sample: impl Fn(usize, usize) -> i16,
    ) {
        let data_block_length = 12 + 2 * lat_count;
        let mut bytes = vec![b' '; DATA_OFFSET];
        bytes[0..4].copy_from_slice(b"UHL1");
        bytes[4..12].copy_from_slice(b"1070000W");
        bytes[12..20].copy_from_slice(b"0360000N");
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

    #[test]
    fn dted_rejects_degenerate_header_counts() {
        let root = temp_path("dted-degenerate-counts");
        fs::create_dir_all(&root).expect("create temp DTED dir");

        for (lon_count, lat_count) in [(0, 2), (1, 2), (2, 0), (2, 1)] {
            let tile_path = root.join(format!("tile-{lon_count}-{lat_count}.dt2"));
            write_synthetic_dted_tile(&tile_path, lon_count, lat_count, |_, _| 0);

            let err = DtedTile::from_path(&tile_path).expect_err("degenerate counts must error");
            assert!(
                err.contains("invalid DTED dimensions"),
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
    // bilinear lookup cases generated from that tile. Generated with Python 3.11.15
    // on macOS-26.5.1-arm64. Floating-point fixture values are serialized as f64
    // hex-bit strings and must be compared with `f64::to_bits`, never tolerances.
    #[test]
    fn dted_lookup_matches_generated_fixture_bits() {
        let raw =
            std::fs::read_to_string(fixture_path("dted_points.json")).expect("read dted fixture");
        let doc: Value = serde_json::from_str(&raw).expect("parse dted fixture");
        assert_eq!(doc["schema"], "gnss-dted-points-v1");

        let mut terrain = DtedTerrain::new(fixture_path("tiles"));
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
    }
}
