//! Geoid undulation (geoid height) lookup with bilinear interpolation.
//!
//! The geoid undulation `N` is the height of the geoid (mean sea level
//! equipotential surface) above the WGS84 reference ellipsoid, in metres.
//! GNSS positioning yields the ellipsoidal height `h`; the orthometric height
//! `H` (height above mean sea level) is
//!
//! ```text
//! H = h - N
//! ```
//!
//! A geoid model is published as a regular latitude/longitude grid of `N`
//! samples (EGM96, EGM2008, and the national models all ship this way). This
//! module provides:
//!
//! - [`GeoidGrid`], a regular grid of undulation samples with bilinear
//!   interpolation ([`GeoidGrid::undulation_rad`] / [`GeoidGrid::undulation_deg`]);
//! - [`GeoidGrid::from_text`], a data-loading hook that parses a simple,
//!   documented grid text format so a caller can supply a full EGM grid;
//! - [`GeoidGrid::from_egm96_dac`], a loader for the authoritative NGA EGM96
//!   15-arcminute binary grid (`WW15MGH.DAC`) for decimetre-class datum work;
//! - [`egm96_undulation`] / [`egm96_grid`], a zero-setup lookup against an
//!   embedded genuine EGM96 1-degree global grid (a higher-accuracy alternative
//!   to the coarse built-in);
//! - [`geoid_undulation`], a zero-setup lookup against a small COARSE built-in
//!   global grid, plus [`orthometric_height_m`] / [`ellipsoidal_height_m`] height
//!   conversion helpers.
//!
//! ## Choosing a grid
//!
//! Three accuracy tiers are available, in increasing fidelity:
//!
//! 1. [`geoid_undulation`] - the COARSE 30-degree built-in. It reproduces the
//!    large-scale character of the geoid (the Indian Ocean low, the North
//!    Atlantic / New Guinea highs, the polar offsets) and is fine for tests,
//!    sanity checks, and metre-scale fallback, but it is NOT survey-grade
//!    (decametre-level error).
//! 2. [`egm96_undulation`] - an embedded GENUINE EGM96 1-degree global grid,
//!    decimated from the official NGA 15-arcminute model. Its bilinear lookup
//!    agrees with the full 15-arcminute EGM96 grid to ~0.4 m RMS (95th
//!    percentile ~0.7 m; up to a few metres over the steepest geoid gradients).
//!    This is the recommended zero-setup default for metre-class datum work.
//! 3. [`GeoidGrid::from_egm96_dac`] with the official `WW15MGH.DAC` file (a
//!    ~2 MB download, not vendored here) - the full 15-arcminute resolution. Its
//!    bilinear lookup tracks the geoid to roughly decimetre RMS, but the
//!    worst-case bilinear interpolation error can still exceed 1 m over the
//!    steepest geoid gradients (see
//!    <https://geographiclib.sourceforge.io/html/geoid.html> for the egm96-15
//!    error envelope), so this path supports decimetre-class typical datum work
//!    rather than guaranteed sub-metre accuracy everywhere. Embedding the full
//!    grid is impractical (the 15-arcminute grid is ~1 M samples and EGM2008
//!    1-minute is ~2.3 GB), so the high-resolution path loads the file at
//!    runtime.
//!
//! A caller with any other vendor grid can lower it to [`GeoidGrid::from_text`]
//! or build a [`GeoidGrid`] via [`GeoidGrid::new`] and call
//! [`GeoidGrid::undulation_rad`] directly.

use std::sync::OnceLock;

/// Why a geoid grid could not be constructed or parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GeoidError {
    /// A grid dimension was zero, or the value count did not equal `n_lat * n_lon`.
    InvalidDimensions {
        /// What was expected.
        expected: usize,
        /// What was supplied.
        found: usize,
    },
    /// A grid spacing or origin was non-finite or non-positive.
    InvalidSpacing {
        /// The offending field.
        field: &'static str,
    },
    /// A grid sample value was non-finite.
    NonFiniteValue {
        /// Row-major index of the offending sample.
        index: usize,
    },
    /// The grid text could not be parsed.
    Parse {
        /// A human-readable reason.
        reason: String,
    },
}

impl core::fmt::Display for GeoidError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidDimensions { expected, found } => {
                write!(
                    f,
                    "geoid grid expected {expected} samples but found {found}"
                )
            }
            Self::InvalidSpacing { field } => {
                write!(f, "geoid grid {field} must be finite and positive")
            }
            Self::NonFiniteValue { index } => {
                write!(f, "geoid grid sample {index} is not finite")
            }
            Self::Parse { reason } => write!(f, "geoid grid parse error: {reason}"),
        }
    }
}

impl std::error::Error for GeoidError {}

/// A regular latitude/longitude grid of geoid undulation samples (metres) with
/// bilinear interpolation.
///
/// Samples are stored row-major with latitude ascending (outer) and longitude
/// ascending (inner): `values_m[i * n_lon + j]` is the undulation at latitude
/// `lat_min_deg + i * dlat_deg` and longitude `lon_min_deg + j * dlon_deg`.
///
/// Latitude inputs are clamped to the grid's latitude span. Longitude inputs are
/// normalized to `[-180, 180)` and then, when the grid spans a full 360 degrees
/// of longitude, wrapped across the antimeridian; otherwise they are clamped to
/// the grid's longitude span (so a regional grid does not wrap).
#[derive(Debug, Clone, PartialEq)]
pub struct GeoidGrid {
    lat_min_deg: f64,
    lon_min_deg: f64,
    dlat_deg: f64,
    dlon_deg: f64,
    n_lat: usize,
    n_lon: usize,
    values_m: Vec<f64>,
}

impl GeoidGrid {
    /// Build a geoid grid from its origin, spacing, dimensions, and row-major
    /// samples (metres).
    ///
    /// Returns [`GeoidError`] when a dimension is zero, the sample count does not
    /// equal `n_lat * n_lon`, a spacing/origin is non-finite or a spacing is
    /// non-positive, or a sample is non-finite.
    pub fn new(
        lat_min_deg: f64,
        lon_min_deg: f64,
        dlat_deg: f64,
        dlon_deg: f64,
        n_lat: usize,
        n_lon: usize,
        values_m: Vec<f64>,
    ) -> Result<Self, GeoidError> {
        if n_lat == 0 || n_lon == 0 {
            return Err(GeoidError::InvalidDimensions {
                expected: 1,
                found: 0,
            });
        }
        let expected = n_lat * n_lon;
        if values_m.len() != expected {
            return Err(GeoidError::InvalidDimensions {
                expected,
                found: values_m.len(),
            });
        }
        if !lat_min_deg.is_finite() {
            return Err(GeoidError::InvalidSpacing { field: "lat_min" });
        }
        if !lon_min_deg.is_finite() {
            return Err(GeoidError::InvalidSpacing { field: "lon_min" });
        }
        if !dlat_deg.is_finite() || dlat_deg <= 0.0 {
            return Err(GeoidError::InvalidSpacing { field: "dlat" });
        }
        if !dlon_deg.is_finite() || dlon_deg <= 0.0 {
            return Err(GeoidError::InvalidSpacing { field: "dlon" });
        }
        for (index, value) in values_m.iter().enumerate() {
            if !value.is_finite() {
                return Err(GeoidError::NonFiniteValue { index });
            }
        }
        Ok(Self {
            lat_min_deg,
            lon_min_deg,
            dlat_deg,
            dlon_deg,
            n_lat,
            n_lon,
            values_m,
        })
    }

    /// Parse a geoid grid from a simple, documented text format (the data-loading
    /// hook for full EGM grids).
    ///
    /// The format is whitespace-delimited with `#` line comments. The first
    /// non-comment token sequence is a six-field header:
    ///
    /// ```text
    /// lat_min lon_min dlat dlon n_lat n_lon
    /// ```
    ///
    /// followed by exactly `n_lat * n_lon` undulation samples in metres, in
    /// row-major order (latitude ascending outer, longitude ascending inner).
    /// All angles are in degrees. This is deliberately a minimal, line-oriented
    /// format; a caller converting a vendor grid (EGM `.gri`/`.ndp`, a GeoTIFF,
    /// etc.) lowers it to this shape or builds a [`GeoidGrid`] via [`new`].
    ///
    /// [`new`]: GeoidGrid::new
    pub fn from_text(text: &str) -> Result<Self, GeoidError> {
        let mut tokens = text
            .lines()
            .map(|line| line.split('#').next().unwrap_or(""))
            .flat_map(str::split_whitespace);

        let mut next_field = |field: &'static str| -> Result<f64, GeoidError> {
            let token = tokens.next().ok_or_else(|| GeoidError::Parse {
                reason: format!("missing header field {field}"),
            })?;
            token.parse::<f64>().map_err(|_| GeoidError::Parse {
                reason: format!("header field {field} is not a number: {token:?}"),
            })
        };

        let lat_min_deg = next_field("lat_min")?;
        let lon_min_deg = next_field("lon_min")?;
        let dlat_deg = next_field("dlat")?;
        let dlon_deg = next_field("dlon")?;
        let n_lat = parse_count(next_field("n_lat")?, "n_lat")?;
        let n_lon = parse_count(next_field("n_lon")?, "n_lon")?;

        let expected = n_lat.checked_mul(n_lon).ok_or_else(|| GeoidError::Parse {
            reason: "n_lat * n_lon overflows".to_string(),
        })?;
        let mut values_m = Vec::with_capacity(expected);
        for token in tokens {
            let value = token.parse::<f64>().map_err(|_| GeoidError::Parse {
                reason: format!("sample is not a number: {token:?}"),
            })?;
            values_m.push(value);
        }

        Self::new(
            lat_min_deg,
            lon_min_deg,
            dlat_deg,
            dlon_deg,
            n_lat,
            n_lon,
            values_m,
        )
    }

    /// Parse the authoritative NGA EGM96 15-arcminute binary geoid grid
    /// (`WW15MGH.DAC`) for decimetre-class datum work.
    ///
    /// This is the highest-resolution path in the module. Its bilinear lookup
    /// tracks the geoid to roughly decimetre RMS, but the worst-case bilinear
    /// interpolation error can still exceed 1 m over the steepest geoid
    /// gradients, so it does not guarantee sub-metre accuracy everywhere.
    ///
    /// The file is a headerless block of `721 * 1440` big-endian `INTEGER*2`
    /// samples in centimetres, arranged north-to-south by record (record 1 at
    /// latitude `+90`, last record at `-90`, in `0.25`-degree steps) and, within
    /// each record, west-to-east by longitude from `0` to `359.75` degrees in
    /// `0.25`-degree steps. Each sample is divided by 100 to get metres. The rows
    /// are flipped to the latitude-ascending storage order of [`GeoidGrid`], so
    /// the resulting grid is global in longitude and wraps across the
    /// antimeridian like any other full-span grid.
    ///
    /// The file is not vendored in this crate (it is a ~2 MB public-domain NGA
    /// download); fetch `WW15MGH.DAC` from the NGA EGM96 distribution and pass its
    /// bytes here. For a zero-setup metre-class default without the download, use
    /// [`egm96_undulation`] instead.
    ///
    /// Returns [`GeoidError::Parse`] if the byte length is not exactly
    /// `721 * 1440 * 2` bytes.
    pub fn from_egm96_dac(bytes: &[u8]) -> Result<Self, GeoidError> {
        let expected = EGM96_DAC_N_LAT * EGM96_DAC_N_LON * 2;
        if bytes.len() != expected {
            return Err(GeoidError::Parse {
                reason: format!(
                    "EGM96 WW15MGH.DAC must be {expected} bytes ({EGM96_DAC_N_LAT} x {EGM96_DAC_N_LON} big-endian int16), got {}",
                    bytes.len()
                ),
            });
        }
        let mut values_m = vec![0.0f64; EGM96_DAC_N_LAT * EGM96_DAC_N_LON];
        for i in 0..EGM96_DAC_N_LAT {
            // DAC record 0 is +90 (north); GeoidGrid stores latitude ascending,
            // so internal row i (latitude -90 + i*0.25) reads DAC record N-1-i.
            let src_row = EGM96_DAC_N_LAT - 1 - i;
            for c in 0..EGM96_DAC_N_LON {
                let off = (src_row * EGM96_DAC_N_LON + c) * 2;
                let cm = i16::from_be_bytes([bytes[off], bytes[off + 1]]);
                values_m[i * EGM96_DAC_N_LON + c] = f64::from(cm) / 100.0;
            }
        }
        Self::new(
            -90.0,
            0.0,
            0.25,
            0.25,
            EGM96_DAC_N_LAT,
            EGM96_DAC_N_LON,
            values_m,
        )
    }

    /// Whether the grid spans a full 360 degrees of longitude (and therefore
    /// wraps across the antimeridian during interpolation).
    fn is_global_longitude(&self) -> bool {
        ((self.n_lon as f64 - 1.0) * self.dlon_deg - 360.0).abs() <= 1.0e-6
            || (self.n_lon as f64 * self.dlon_deg - 360.0).abs() <= 1.0e-6
    }

    /// Bilinearly interpolated undulation `N` (metres) at a geodetic position in
    /// radians (latitude positive north, longitude positive east).
    pub fn undulation_rad(&self, lat_rad: f64, lon_rad: f64) -> f64 {
        self.undulation_deg(lat_rad.to_degrees(), lon_rad.to_degrees())
    }

    /// Bilinearly interpolated undulation `N` (metres) at a geodetic position in
    /// degrees (latitude positive north, longitude positive east).
    pub fn undulation_deg(&self, lat_deg: f64, lon_deg: f64) -> f64 {
        let lat = lat_deg.clamp(self.lat_min_deg, self.lat_max_deg());
        let (i0, i1, ty) = self.lat_bracket(lat);

        let (j0, j1, tx) = self.lon_bracket(lon_deg);

        let v00 = self.sample(i0, j0);
        let v01 = self.sample(i0, j1);
        let v10 = self.sample(i1, j0);
        let v11 = self.sample(i1, j1);

        let bottom = v00 + (v01 - v00) * tx;
        let top = v10 + (v11 - v10) * tx;
        bottom + (top - bottom) * ty
    }

    fn lat_max_deg(&self) -> f64 {
        self.lat_min_deg + (self.n_lat as f64 - 1.0) * self.dlat_deg
    }

    /// Latitude bracketing cell indices and fractional position within the cell.
    fn lat_bracket(&self, lat_deg: f64) -> (usize, usize, f64) {
        if self.n_lat == 1 {
            return (0, 0, 0.0);
        }
        let pos = (lat_deg - self.lat_min_deg) / self.dlat_deg;
        let pos = pos.clamp(0.0, self.n_lat as f64 - 1.0);
        let i0 = (pos.floor() as usize).min(self.n_lat - 2);
        (i0, i0 + 1, pos - i0 as f64)
    }

    /// Longitude bracketing cell indices and fractional position within the cell.
    /// Wraps across the antimeridian for a global grid; clamps for a regional one.
    fn lon_bracket(&self, lon_deg: f64) -> (usize, usize, f64) {
        if self.n_lon == 1 {
            return (0, 0, 0.0);
        }
        let lon = normalize_longitude_deg(lon_deg);
        if self.is_global_longitude() {
            let span = self.n_lon as f64 * self.dlon_deg;
            let mut offset = (lon - self.lon_min_deg).rem_euclid(span);
            // Guard the rare case where rounding lands offset exactly on span.
            if offset >= span {
                offset -= span;
            }
            let pos = offset / self.dlon_deg;
            let j0 = (pos.floor() as usize) % self.n_lon;
            let j1 = (j0 + 1) % self.n_lon;
            (j0, j1, pos - pos.floor())
        } else {
            let pos =
                ((lon - self.lon_min_deg) / self.dlon_deg).clamp(0.0, self.n_lon as f64 - 1.0);
            let j0 = (pos.floor() as usize).min(self.n_lon - 2);
            (j0, j0 + 1, pos - j0 as f64)
        }
    }

    fn sample(&self, i: usize, j: usize) -> f64 {
        self.values_m[i * self.n_lon + j]
    }
}

/// Parse a non-negative grid count from a float token.
fn parse_count(value: f64, field: &'static str) -> Result<usize, GeoidError> {
    if !value.is_finite() || value < 1.0 || value.fract() != 0.0 {
        return Err(GeoidError::Parse {
            reason: format!("{field} must be a positive integer, got {value}"),
        });
    }
    Ok(value as usize)
}

/// Normalize a longitude in degrees to the half-open interval `[-180, 180)`.
fn normalize_longitude_deg(lon_deg: f64) -> f64 {
    let wrapped = (lon_deg + 180.0).rem_euclid(360.0) - 180.0;
    // rem_euclid can yield +180.0 for inputs at the boundary; fold it to -180.0.
    if wrapped >= 180.0 {
        wrapped - 360.0
    } else {
        wrapped
    }
}

/// Geoid undulation `N` (metres above the WGS84 ellipsoid) at a geodetic
/// position in radians, from the COARSE built-in global grid.
///
/// Latitude is positive north, longitude positive east, both in radians. See
/// the module docs for the built-in-grid-vs-real-model trade-off: for accuracy
/// load a real model with [`GeoidGrid::from_text`] and call
/// [`GeoidGrid::undulation_rad`].
pub fn geoid_undulation(lat_rad: f64, lon_rad: f64) -> f64 {
    builtin_grid().undulation_rad(lat_rad, lon_rad)
}

/// Orthometric height `H = h - N` (metres above mean sea level) from an
/// ellipsoidal height and a geodetic position in radians, using the COARSE
/// 30-degree built-in model's undulation (decametre-level error, NOT
/// survey-grade). For metre-class conversion use [`egm96_orthometric_height_m`];
/// for a real model, subtract [`GeoidGrid::undulation_rad`] directly.
pub fn orthometric_height_m(ellipsoidal_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    ellipsoidal_height_m - geoid_undulation(lat_rad, lon_rad)
}

/// Ellipsoidal height `h = H + N` (metres above the WGS84 ellipsoid) from an
/// orthometric height and a geodetic position in radians, using the COARSE
/// 30-degree built-in model's undulation (decametre-level error, NOT
/// survey-grade). For metre-class conversion use [`egm96_ellipsoidal_height_m`];
/// for a real model, add [`GeoidGrid::undulation_rad`] directly.
pub fn ellipsoidal_height_m(orthometric_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    orthometric_height_m + geoid_undulation(lat_rad, lon_rad)
}

/// Orthometric height `H = h - N` (metres above mean sea level) from an
/// ellipsoidal height and a geodetic position in radians, using the embedded
/// GENUINE EGM96 1-degree model via [`egm96_undulation`].
///
/// This is the recommended zero-setup height converter for metre-class datum
/// work; the [`orthometric_height_m`] sibling uses the COARSE 30-degree built-in
/// instead and is only suitable for sanity checks.
pub fn egm96_orthometric_height_m(ellipsoidal_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    ellipsoidal_height_m - egm96_undulation(lat_rad, lon_rad)
}

/// Ellipsoidal height `h = H + N` (metres above the WGS84 ellipsoid) from an
/// orthometric height and a geodetic position in radians, using the embedded
/// GENUINE EGM96 1-degree model via [`egm96_undulation`].
///
/// This is the recommended zero-setup height converter for metre-class datum
/// work; the [`ellipsoidal_height_m`] sibling uses the COARSE 30-degree built-in
/// instead and is only suitable for sanity checks.
pub fn egm96_ellipsoidal_height_m(orthometric_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    orthometric_height_m + egm96_undulation(lat_rad, lon_rad)
}

/// Latitude record count of the NGA EGM96 `WW15MGH.DAC` 15-arcminute grid.
const EGM96_DAC_N_LAT: usize = 721;
/// Longitude sample count per record of the NGA EGM96 `WW15MGH.DAC` grid.
const EGM96_DAC_N_LON: usize = 1440;

/// Latitude row count of the embedded genuine EGM96 1-degree grid.
const EGM96_1DEG_N_LAT: usize = 181;
/// Longitude column count of the embedded genuine EGM96 1-degree grid.
const EGM96_1DEG_N_LON: usize = 360;

// Provenance of the embedded EGM96 1-degree undulation grid
// (`egm96_geoid_1deg.bin`):
//
// Source model: EGM96 (Earth Gravitational Model 1996), the joint NIMA (now
// NGA) / NASA GSFC / Ohio State University global geopotential model. The geoid
// undulation grid is a work of the U.S. Government and is in the public domain;
// NGA distributes it without restriction. See THIRD-PARTY-NOTICES.md.
//
// Origin file: the official NGA 15-arcminute binary grid `WW15MGH.DAC`
// (721 x 1440 big-endian INTEGER*2 centimetres, north-to-south records,
// longitude 0..359.75 E), obtained from the public OpenSGeo PROJ vdatum mirror
// (download.osgeo.org/proj/vdatum/egm96_15/). `egm96_geoid_1deg.bin` is that
// grid decimated to a 1-degree lattice: each sample is the exact `WW15MGH.DAC`
// value at the corresponding integer-degree node (no resampling or smoothing -
// 1 degree is an integer multiple of the 0.25-degree source spacing), so every
// value is a genuine EGM96 undulation, not a fabricated or fitted figure. The
// packed format is 181 x 360 big-endian INTEGER*2 centimetres in
// latitude-ascending (-90..+90), longitude-ascending (0..359 E) row-major order.
// Decimating to 1 degree keeps the embedded data tractable (~127 KB) while its
// bilinear lookup tracks the full 15-arcminute grid to ~0.4 m RMS.

/// Bytes of the embedded genuine EGM96 1-degree undulation grid (big-endian
/// int16 centimetres, latitude-ascending, longitude-ascending row-major).
const EGM96_1DEG_BYTES: &[u8] = include_bytes!("egm96_geoid_1deg.bin");

/// The embedded genuine EGM96 1-degree global geoid, decoded once on first use.
///
/// See [`egm96_undulation`] for the recommended scalar entry point and the
/// module docs for the accuracy tiers.
pub fn egm96_grid() -> &'static GeoidGrid {
    static GRID: OnceLock<GeoidGrid> = OnceLock::new();
    GRID.get_or_init(|| {
        assert_eq!(
            EGM96_1DEG_BYTES.len(),
            EGM96_1DEG_N_LAT * EGM96_1DEG_N_LON * 2,
            "embedded EGM96 1-degree grid has the wrong byte length"
        );
        let mut values_m = vec![0.0f64; EGM96_1DEG_N_LAT * EGM96_1DEG_N_LON];
        for (k, value) in values_m.iter_mut().enumerate() {
            let cm = i16::from_be_bytes([EGM96_1DEG_BYTES[k * 2], EGM96_1DEG_BYTES[k * 2 + 1]]);
            *value = f64::from(cm) / 100.0;
        }
        GeoidGrid::new(
            -90.0,
            0.0,
            1.0,
            1.0,
            EGM96_1DEG_N_LAT,
            EGM96_1DEG_N_LON,
            values_m,
        )
        .expect("embedded EGM96 1-degree grid is well-formed")
    })
}

/// Geoid undulation `N` (metres above the WGS84 ellipsoid) at a geodetic
/// position in radians, from the embedded GENUINE EGM96 1-degree global grid.
///
/// Latitude is positive north, longitude positive east, both in radians. This is
/// the recommended zero-setup default for metre-class datum work: its bilinear
/// lookup agrees with the full 15-arcminute EGM96 grid to ~0.4 m RMS. For the
/// full-resolution model load the official `WW15MGH.DAC` via
/// [`GeoidGrid::from_egm96_dac`]; for the lowest-fidelity legacy fallback use
/// [`geoid_undulation`].
pub fn egm96_undulation(lat_rad: f64, lon_rad: f64) -> f64 {
    egm96_grid().undulation_rad(lat_rad, lon_rad)
}

/// The coarse 30-degree built-in global geoid, built once on first use.
fn builtin_grid() -> &'static GeoidGrid {
    static GRID: OnceLock<GeoidGrid> = OnceLock::new();
    GRID.get_or_init(|| {
        GeoidGrid::new(
            -90.0,
            -180.0,
            30.0,
            30.0,
            BUILTIN_N_LAT,
            BUILTIN_N_LON,
            BUILTIN_VALUES_M.to_vec(),
        )
        .expect("built-in geoid grid is well-formed")
    })
}

const BUILTIN_N_LAT: usize = 7; // latitudes -90, -60, -30, 0, 30, 60, 90
const BUILTIN_N_LON: usize = 13; // longitudes -180 .. 180 step 30 (col 0 == col 12)

/// A COARSE 30-degree global geoid undulation field (metres). Row-major, latitude
/// ascending then longitude ascending. The values approximate the large-scale
/// EGM character (Gulf of Guinea / North Atlantic / New Guinea highs, the Indian
/// Ocean low, polar offsets); they are NOT survey-grade. The first and last
/// longitude columns coincide on the antimeridian so the global wrap is
/// continuous.
#[rustfmt::skip]
const BUILTIN_VALUES_M: [f64; BUILTIN_N_LAT * BUILTIN_N_LON] = [
    // lat = -90 (south pole)
    -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0, -30.0,
    // lat = -60
    -15.0, -20.0, -25.0, -10.0,   5.0,  15.0,  20.0,  10.0,   0.0,  -5.0, -10.0, -12.0, -15.0,
    // lat = -30
     20.0,  10.0,  -5.0, -25.0, -15.0,   5.0,  25.0,  30.0,  20.0,  35.0,  40.0,  25.0,  20.0,
    // lat = 0 (equator)
    -10.0, -20.0, -15.0,  -8.0,  -5.0,   5.0,  17.0,  10.0, -30.0, -60.0,  30.0,  55.0, -10.0,
    // lat = 30
      5.0,   0.0, -15.0, -10.0, -40.0,  50.0,  45.0,  20.0, -25.0, -45.0,   0.0,  20.0,   5.0,
    // lat = 60
      0.0, -10.0, -20.0, -35.0, -20.0,  60.0,  45.0,  25.0,  10.0,  -5.0, -15.0,  -5.0,   0.0,
    // lat = 90 (north pole)
     13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,  13.0,
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_returns_exact_node_values() {
        // (lat 0, lon 0) is the Gulf of Guinea node, a documented +17 m sample.
        assert_eq!(geoid_undulation(0.0, 0.0), 17.0);
        // (lat 0, lon 90 deg) is the Indian Ocean low node.
        assert_eq!(geoid_undulation(0.0, 90.0_f64.to_radians()), -60.0);
        // (lat 60 N, lon -30 deg) is the North Atlantic / Iceland high node.
        assert_eq!(
            geoid_undulation(60.0_f64.to_radians(), (-30.0_f64).to_radians()),
            60.0
        );
    }

    #[test]
    fn builtin_captures_major_geoid_features_by_sign() {
        // The Indian Ocean is the global geoid low: undulation is strongly negative.
        let indian_ocean = geoid_undulation(0.0, 80.0_f64.to_radians());
        assert!(indian_ocean < -20.0, "indian ocean N = {indian_ocean}");
        // The North Atlantic is a geoid high: undulation is positive.
        let north_atlantic = geoid_undulation(55.0_f64.to_radians(), (-25.0_f64).to_radians());
        assert!(north_atlantic > 20.0, "north atlantic N = {north_atlantic}");
    }

    #[test]
    fn bilinear_midpoint_is_the_corner_average() {
        let grid = GeoidGrid::new(0.0, 0.0, 10.0, 10.0, 2, 2, vec![1.0, 3.0, 5.0, 11.0]).unwrap();
        // Cell-center: equal weight to all four corners -> their mean.
        let center = grid.undulation_deg(5.0, 5.0);
        assert!((center - (1.0 + 3.0 + 5.0 + 11.0) / 4.0).abs() <= 1.0e-12);
        // Edge midpoints interpolate along one axis only.
        assert!((grid.undulation_deg(0.0, 5.0) - 2.0).abs() <= 1.0e-12);
        assert!((grid.undulation_deg(5.0, 0.0) - 3.0).abs() <= 1.0e-12);
        // Corners return the node values exactly.
        assert_eq!(grid.undulation_deg(0.0, 0.0), 1.0);
        assert_eq!(grid.undulation_deg(10.0, 10.0), 11.0);
    }

    #[test]
    fn global_grid_wraps_across_the_antimeridian() {
        // A global grid whose +180 column equals its -180 column interpolates
        // continuously across the seam: two points a hair either side of the
        // antimeridian return nearly the same undulation (no discontinuity).
        let east = geoid_undulation(0.0, 179.999_f64.to_radians());
        let west = geoid_undulation(0.0, (-179.999_f64).to_radians());
        assert!((east - west).abs() < 0.01, "seam jump: {east} vs {west}");
        // The antimeridian node itself is -10 m on the equator row.
        assert!((east - (-10.0)).abs() < 0.05, "east seam N = {east}");
        assert!((west - (-10.0)).abs() < 0.05, "west seam N = {west}");
        // Exactly +180 and -180 are the same physical meridian -> same value.
        let plus = geoid_undulation(0.0, 180.0_f64.to_radians());
        let minus = geoid_undulation(0.0, (-180.0_f64).to_radians());
        assert_eq!(plus, minus);
        assert_eq!(plus, -10.0);
    }

    #[test]
    fn orthometric_height_subtracts_undulation() {
        let lat = 0.0;
        let lon = 0.0;
        let n = geoid_undulation(lat, lon);
        assert_eq!(n, 17.0);
        // h = 117 m ellipsoidal -> H = 117 - 17 = 100 m above mean sea level.
        assert_eq!(orthometric_height_m(117.0, lat, lon), 100.0);
        // H = 100 m orthometric -> h = 100 + 17 = 117 m ellipsoidal.
        assert_eq!(ellipsoidal_height_m(100.0, lat, lon), 117.0);
    }

    #[test]
    fn egm96_height_converters_use_the_egm96_undulation() {
        // A known point well away from the coarse-grid agreement; the egm96
        // converters must subtract/add the genuine EGM96 1-degree undulation, not
        // the coarse 30-degree built-in.
        let lat = 37.0_f64.to_radians();
        let lon = (-122.0_f64).to_radians();
        let n = egm96_undulation(lat, lon);
        let h = 250.0;
        let big_h = egm96_orthometric_height_m(h, lat, lon);
        assert_eq!(big_h, h - n);
        assert_eq!(egm96_ellipsoidal_height_m(big_h, lat, lon), big_h + n);
        // The egm96 path differs from the coarse path here (different model).
        assert_ne!(
            egm96_orthometric_height_m(h, lat, lon),
            orthometric_height_m(h, lat, lon)
        );
    }

    #[test]
    fn from_text_round_trips_a_grid() {
        let text = "\
# coarse 2x3 regional grid
# lat_min lon_min dlat dlon n_lat n_lon
10.0 20.0 5.0 5.0 2 3
  1.0  2.0  3.0   # lat 10 row
  4.0  5.0  6.0   # lat 15 row
";
        let grid = GeoidGrid::from_text(text).expect("parse grid");
        assert_eq!(grid.undulation_deg(10.0, 20.0), 1.0);
        assert_eq!(grid.undulation_deg(15.0, 30.0), 6.0);
        // Cell center of the lower-left cell -> mean of the four corners.
        let center = grid.undulation_deg(12.5, 22.5);
        assert!((center - (1.0 + 2.0 + 4.0 + 5.0) / 4.0).abs() <= 1.0e-12);
        // A regional grid clamps rather than wraps outside its longitude span.
        assert_eq!(
            grid.undulation_deg(10.0, 0.0),
            grid.undulation_deg(10.0, 20.0)
        );
    }

    #[test]
    fn from_text_rejects_short_data() {
        let text = "0.0 0.0 1.0 1.0 2 2\n1.0 2.0 3.0\n";
        assert_eq!(
            GeoidGrid::from_text(text),
            Err(GeoidError::InvalidDimensions {
                expected: 4,
                found: 3
            })
        );
    }

    #[test]
    fn new_rejects_bad_inputs() {
        assert!(matches!(
            GeoidGrid::new(0.0, 0.0, 1.0, 1.0, 2, 2, vec![1.0, 2.0, 3.0]),
            Err(GeoidError::InvalidDimensions { .. })
        ));
        assert!(matches!(
            GeoidGrid::new(0.0, 0.0, 0.0, 1.0, 2, 2, vec![0.0; 4]),
            Err(GeoidError::InvalidSpacing { field: "dlat" })
        ));
        assert!(matches!(
            GeoidGrid::new(0.0, 0.0, 1.0, 1.0, 2, 2, vec![0.0, f64::NAN, 0.0, 0.0]),
            Err(GeoidError::NonFiniteValue { index: 1 })
        ));
    }

    #[test]
    fn longitude_normalization_folds_into_half_open_interval() {
        assert!((normalize_longitude_deg(190.0) - (-170.0)).abs() <= 1.0e-12);
        assert!((normalize_longitude_deg(-190.0) - 170.0).abs() <= 1.0e-12);
        assert!((normalize_longitude_deg(180.0) - (-180.0)).abs() <= 1.0e-12);
        assert!((normalize_longitude_deg(360.0)).abs() <= 1.0e-12);
    }

    /// The embedded EGM96 1-degree grid returns its genuine node values exactly
    /// at integer-degree positions (a node query is an exact bilinear hit). The
    /// expected figures are the corresponding `WW15MGH.DAC` samples (cm/100),
    /// transcribed from the source grid; see the provenance note in this module.
    #[test]
    fn egm96_grid_reproduces_genuine_nodes() {
        // (lat_deg, lon_deg, expected EGM96 undulation in metres).
        let nodes: [(f64, f64, f64); 5] = [
            (0.0, 0.0, 17.16),    // Gulf of Guinea
            (0.0, 80.0, -102.69), // Indian Ocean low
            (60.0, -30.0, 63.80), // North Atlantic high (lon -30 == 330 E)
            (-90.0, 0.0, -29.53), // south pole
            (90.0, 0.0, 13.61),   // north pole
        ];
        for (lat, lon, want) in nodes {
            let got = egm96_undulation(lat.to_radians(), lon.to_radians());
            assert!(
                (got - want).abs() <= 1.0e-9,
                "egm96 node ({lat},{lon}): got {got}, want {want}"
            );
        }
    }

    /// The embedded EGM96 grid matches the independently published EGM96 geoid
    /// height at a known checkpoint within the tolerance set by its 1-degree
    /// resolution, and is far closer to truth than the coarse built-in.
    ///
    /// Reference: GeographicLib `GeoidEval` (egm96-5) reports `28.7068` m at
    /// `16:46:33N 3:00:34W` (Timbuktu); see
    /// `https://geographiclib.sourceforge.io/C++/doc/GeoidEval.1.html`. The full
    /// 15-arcminute EGM96 grid bilinearly interpolates to `28.6976` m there; the
    /// embedded 1-degree grid lands at `28.6746` m, i.e. within ~0.03 m of the
    /// published value, well inside a 1-degree-resolution tolerance.
    #[test]
    fn egm96_grid_matches_published_checkpoint() {
        let lat = (16.0 + 46.0 / 60.0 + 33.0 / 3600.0_f64).to_radians();
        let lon = (-(3.0 + 0.0 / 60.0 + 34.0 / 3600.0_f64)).to_radians();
        let published = 28.7068;

        let egm96 = egm96_undulation(lat, lon);
        assert!(
            (egm96 - published).abs() < 0.5,
            "egm96 Timbuktu {egm96} not within 0.5 m of published {published}"
        );

        // The genuine 1-degree grid must be strictly closer to the published
        // value than the decametre-scale 30-degree built-in.
        let coarse = geoid_undulation(lat, lon);
        assert!(
            (egm96 - published).abs() < (coarse - published).abs(),
            "egm96 ({egm96}) should beat the coarse built-in ({coarse}) vs {published}"
        );
    }

    /// `from_egm96_dac` decodes the NGA `WW15MGH.DAC` layout: big-endian int16
    /// centimetres, north-to-south records flipped to latitude-ascending storage,
    /// longitude `0..359.75` E. Validated against an independently built grid of
    /// the same samples, plus the byte-length guard.
    #[test]
    fn from_egm96_dac_decodes_the_nga_layout() {
        let n_lat = super::EGM96_DAC_N_LAT;
        let n_lon = super::EGM96_DAC_N_LON;
        // A deterministic per-(record, column) pattern, well within int16 cm.
        let cm = |record: usize, col: usize| -> i16 {
            ((record as i32) - 360 + (col as i32 % 11) - 5) as i16
        };

        let mut bytes = Vec::with_capacity(n_lat * n_lon * 2);
        for record in 0..n_lat {
            for col in 0..n_lon {
                bytes.extend_from_slice(&cm(record, col).to_be_bytes());
            }
        }
        let parsed = GeoidGrid::from_egm96_dac(&bytes).expect("parse synthetic DAC");

        // Independent reconstruction: internal row i (latitude -90 + i*0.25) is
        // DAC record n_lat-1-i, columns unchanged, centimetres -> metres.
        let mut values_m = vec![0.0f64; n_lat * n_lon];
        for i in 0..n_lat {
            let record = n_lat - 1 - i;
            for col in 0..n_lon {
                values_m[i * n_lon + col] = f64::from(cm(record, col)) / 100.0;
            }
        }
        let expected =
            GeoidGrid::new(-90.0, 0.0, 0.25, 0.25, n_lat, n_lon, values_m).expect("reference grid");
        assert_eq!(parsed, expected);

        // A wrong byte length is rejected, not silently misread.
        assert!(matches!(
            GeoidGrid::from_egm96_dac(&bytes[..bytes.len() - 2]),
            Err(GeoidError::Parse { .. })
        ));
    }
}
