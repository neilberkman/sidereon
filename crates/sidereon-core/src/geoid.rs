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
//! - [`GeoidGrid::from_proj_egm96_gtx`], a loader for PROJ's public EGM96
//!   15-arcminute GTX grid, paired with [`GeoidGrid::undulation_proj_rad`] and
//!   an explicit [`ProjVgridshiftArithmetic`] recipe for PROJ 9.3.0
//!   vertical-grid interpolation;
//! - [`GeoidGrid::from_egm2008_raster`], a loader for the NGA EGM2008
//!   row-framed `REAL*4` raster grids at 2.5-arcminute and 1-arcminute spacing;
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
//! For bit-level interoperability with a PROJ vertical-grid pipeline, load the
//! public `egm96_15.gtx` with [`GeoidGrid::from_proj_egm96_gtx`] and call
//! [`GeoidGrid::undulation_proj_rad`]. This path preserves the GTX float samples
//! and reproduces PROJ 9.3.0's radian indexing and blend order. Because PROJ's
//! C++ source does not prescribe floating-point contraction, the caller also
//! selects whether the multiply-add steps are fused or separately rounded.
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

/// Why PROJ vertical-grid interpolation could not evaluate a coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjVgridshiftError {
    /// A lookup coordinate was not finite.
    NonFiniteCoordinate {
        /// The offending coordinate.
        field: &'static str,
    },
    /// A lookup coordinate was outside the grid extent.
    CoordinateOutsideGrid {
        /// The offending coordinate.
        field: &'static str,
    },
}

impl core::fmt::Display for ProjVgridshiftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonFiniteCoordinate { field } => {
                write!(f, "PROJ vertical-grid {field} coordinate is not finite")
            }
            Self::CoordinateOutsideGrid { field } => {
                write!(
                    f,
                    "PROJ vertical-grid {field} coordinate is outside the grid"
                )
            }
        }
    }
}

impl std::error::Error for ProjVgridshiftError {}

/// Floating-point evaluation recipe for PROJ vertical-grid interpolation.
///
/// PROJ 9.3.0 expresses its final three accumulation steps as ordinary C++
/// multiply/add statements and does not set a contraction policy. Consequently,
/// conforming builds can differ by one ULP depending on compiler, target, and
/// build flags. Selecting the recipe explicitly makes the requested behavior
/// reproducible instead of guessing from the Rust compilation target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjVgridshiftArithmetic {
    /// Round the multiplication and addition separately.
    ///
    /// This matches PROJ builds where floating-point contraction is disabled,
    /// including the reviewed default x86-64 Clang build of PROJ 9.3.0.
    SeparateMultiplyAdd,
    /// Evaluate each accumulation as a fused multiply-add with one rounding.
    ///
    /// This matches PROJ builds where the compiler contracts the statements,
    /// including the AArch64 PROJ 9.3.0 build used for the dense fixture.
    FusedMultiplyAdd,
}

/// Supported NGA EGM2008 interpolation-raster spacings.
///
/// The official rasters store `REAL*4` geoid undulation samples in Fortran
/// sequential records. Rows run north-to-south, columns run west-to-east from
/// longitude `0` degrees east, and there is no duplicate `360` degree column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Egm2008GridSpacing {
    /// The 1-arcminute EGM2008 grid, `10801 x 21600` nodes.
    OneMinute,
    /// The 2.5-arcminute EGM2008 grid, `4321 x 8640` nodes.
    TwoPointFiveMinute,
}

impl Egm2008GridSpacing {
    /// Grid spacing in arcminutes.
    pub fn arc_minutes(self) -> f64 {
        match self {
            Self::OneMinute => 1.0,
            Self::TwoPointFiveMinute => 2.5,
        }
    }

    /// Grid spacing in degrees.
    pub fn degrees(self) -> f64 {
        self.arc_minutes() / 60.0
    }

    /// Official global row and column counts for this spacing.
    pub fn global_dimensions(self) -> (usize, usize) {
        match self {
            Self::OneMinute => (EGM2008_1_MIN_N_LAT, EGM2008_1_MIN_N_LON),
            Self::TwoPointFiveMinute => (EGM2008_2P5_MIN_N_LAT, EGM2008_2P5_MIN_N_LON),
        }
    }
}

/// A full or cropped EGM2008 row-framed raster window.
///
/// The window describes the bytes passed to
/// [`GeoidGrid::from_egm2008_raster_window`]. The byte stream contains one
/// Fortran sequential record per latitude row, ordered north-to-south, with
/// `n_lon` `REAL*4` samples per row. The resulting [`GeoidGrid`] stores rows
/// latitude-ascending and uses the same bilinear interpolation path as every
/// other geoid grid in this module.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Egm2008RasterWindow {
    spacing: Egm2008GridSpacing,
    lat_min_deg: f64,
    lon_min_deg: f64,
    n_lat: usize,
    n_lon: usize,
}

impl Egm2008RasterWindow {
    /// Build a window descriptor for EGM2008 row-framed raster bytes.
    ///
    /// `lat_min_deg` and `lon_min_deg` are the southwest node of the resulting
    /// grid in degrees. `n_lat` and `n_lon` are the node counts in the supplied
    /// byte stream. Returns [`GeoidError`] if a dimension is zero, an origin is
    /// not finite, the latitude span falls outside `[-90, 90]`, or the longitude
    /// span exceeds a full global revolution.
    pub fn new(
        spacing: Egm2008GridSpacing,
        lat_min_deg: f64,
        lon_min_deg: f64,
        n_lat: usize,
        n_lon: usize,
    ) -> Result<Self, GeoidError> {
        if n_lat == 0 || n_lon == 0 {
            return Err(GeoidError::InvalidDimensions {
                expected: 1,
                found: 0,
            });
        }
        if !lat_min_deg.is_finite() {
            return Err(GeoidError::InvalidSpacing { field: "lat_min" });
        }
        if !lon_min_deg.is_finite() {
            return Err(GeoidError::InvalidSpacing { field: "lon_min" });
        }
        let d = spacing.degrees();
        let lat_max_deg = lat_min_deg + (n_lat as f64 - 1.0) * d;
        if lat_min_deg < -90.0 - 1.0e-12 || lat_max_deg > 90.0 + 1.0e-12 {
            return Err(GeoidError::Parse {
                reason: format!(
                    "EGM2008 latitude window [{lat_min_deg}, {lat_max_deg}] exceeds [-90, 90]"
                ),
            });
        }
        let lon_span_deg = n_lon as f64 * d;
        if lon_span_deg > 360.0 + 1.0e-12 {
            return Err(GeoidError::Parse {
                reason: format!("EGM2008 longitude span {lon_span_deg} exceeds 360 degrees"),
            });
        }
        Ok(Self {
            spacing,
            lat_min_deg,
            lon_min_deg,
            n_lat,
            n_lon,
        })
    }

    /// Build the official full-global EGM2008 window for a spacing.
    pub fn global(spacing: Egm2008GridSpacing) -> Self {
        let (n_lat, n_lon) = spacing.global_dimensions();
        Self::new(spacing, -90.0, 0.0, n_lat, n_lon)
            .expect("EGM2008 global raster dimensions are valid")
    }

    /// Raster spacing for this window.
    pub fn spacing(self) -> Egm2008GridSpacing {
        self.spacing
    }

    /// Southwest latitude of this window in degrees.
    pub fn lat_min_deg(self) -> f64 {
        self.lat_min_deg
    }

    /// Western longitude of this window in degrees.
    pub fn lon_min_deg(self) -> f64 {
        self.lon_min_deg
    }

    /// Latitude node count in this window.
    pub fn n_lat(self) -> usize {
        self.n_lat
    }

    /// Longitude node count in this window.
    pub fn n_lon(self) -> usize {
        self.n_lon
    }
}

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

    /// Parse PROJ's public EGM96 15-arcminute vertical-shift grid
    /// (`egm96_15.gtx`).
    ///
    /// The grid is distributed by the OSGeo PROJ vdatum mirror at
    /// <https://download.osgeo.org/proj/vdatum/egm96_15/egm96_15.gtx>. Its GTX
    /// header and samples are big-endian: four `f64` fields (`south`, `west`,
    /// latitude spacing, longitude spacing), two `i32` dimensions, then
    /// `721 * 1440` row-major `f32` metre offsets. Rows are already ordered
    /// south-to-north, as required by PROJ's vertical-grid interpolation.
    ///
    /// Use [`GeoidGrid::undulation_proj_rad`] with the returned grid and the
    /// [`ProjVgridshiftArithmetic`] recipe matching the reference PROJ build.
    /// The ordinary [`GeoidGrid::undulation_rad`] method intentionally retains
    /// this crate's general degree-space interpolation behavior.
    pub fn from_proj_egm96_gtx(bytes: &[u8]) -> Result<Self, GeoidError> {
        let expected =
            PROJ_EGM96_GTX_HEADER_BYTES + PROJ_EGM96_GTX_N_LAT * PROJ_EGM96_GTX_N_LON * 4;
        if bytes.len() != expected {
            return Err(GeoidError::Parse {
                reason: format!(
                    "PROJ egm96_15.gtx must be {expected} bytes, got {}",
                    bytes.len()
                ),
            });
        }

        let south_deg = read_be_f64(bytes, 0);
        let west_deg = read_be_f64(bytes, 8);
        let dlat_deg = read_be_f64(bytes, 16);
        let dlon_deg = read_be_f64(bytes, 24);
        let n_lat = read_be_i32(bytes, 32);
        let n_lon = read_be_i32(bytes, 36);
        if south_deg.to_bits() != (-90.0f64).to_bits()
            || west_deg.to_bits() != (-180.0f64).to_bits()
            || dlat_deg.to_bits() != 0.25f64.to_bits()
            || dlon_deg.to_bits() != 0.25f64.to_bits()
            || n_lat != PROJ_EGM96_GTX_N_LAT as i32
            || n_lon != PROJ_EGM96_GTX_N_LON as i32
        {
            return Err(GeoidError::Parse {
                reason: format!(
                    "PROJ egm96_15.gtx header mismatch: south={south_deg}, west={west_deg}, dlat={dlat_deg}, dlon={dlon_deg}, rows={n_lat}, columns={n_lon}"
                ),
            });
        }

        let mut values_m = Vec::with_capacity(PROJ_EGM96_GTX_N_LAT * PROJ_EGM96_GTX_N_LON);
        for chunk in bytes[PROJ_EGM96_GTX_HEADER_BYTES..].as_chunks::<4>().0 {
            let value = f32::from_be_bytes(*chunk);
            if !value.is_finite() {
                return Err(GeoidError::NonFiniteValue {
                    index: values_m.len(),
                });
            }
            values_m.push(f64::from(value));
        }

        Self::new(
            south_deg,
            west_deg,
            dlat_deg,
            dlon_deg,
            PROJ_EGM96_GTX_N_LAT,
            PROJ_EGM96_GTX_N_LON,
            values_m,
        )
    }

    /// Parse an official full-global NGA EGM2008 interpolation raster.
    ///
    /// The byte stream must be the `Und_min1x1_...` or `Und_min2.5x2.5_...`
    /// raster for the supplied spacing. Both the original big-endian files and
    /// the NGA small-endian variants are accepted. Each row is a Fortran
    /// sequential record whose leading and trailing record lengths must match
    /// `n_lon * 4` bytes, and each sample is decoded as a finite `REAL*4`
    /// undulation in metres.
    ///
    /// Use [`GeoidGrid::from_egm2008_raster_window`] when validating or loading
    /// a cropped raster window with the same record layout.
    pub fn from_egm2008_raster(
        bytes: &[u8],
        spacing: Egm2008GridSpacing,
    ) -> Result<Self, GeoidError> {
        Self::from_egm2008_raster_window(bytes, Egm2008RasterWindow::global(spacing))
    }

    /// Parse a full or cropped NGA EGM2008 interpolation raster window.
    ///
    /// `window` supplies the grid spacing, southwest node, and node dimensions.
    /// The byte stream must contain exactly one north-to-south Fortran
    /// sequential record per latitude row, with `n_lon` `REAL*4` samples in each
    /// row. Both big-endian and small-endian record/sample encodings are
    /// accepted and are detected from the first record marker.
    pub fn from_egm2008_raster_window(
        bytes: &[u8],
        window: Egm2008RasterWindow,
    ) -> Result<Self, GeoidError> {
        let values_m = parse_egm2008_raster_values(bytes, window)?;
        Self::new(
            window.lat_min_deg,
            window.lon_min_deg,
            window.spacing.degrees(),
            window.spacing.degrees(),
            window.n_lat,
            window.n_lon,
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

    /// Bilinearly interpolated undulation using PROJ 9.3.0's
    /// `read_vgrid_value` indexing and operation order.
    ///
    /// Unlike [`GeoidGrid::undulation_rad`], this method constructs the grid
    /// extent and reciprocal resolution in radians, indexes latitude from the
    /// south, and evaluates the four bilinear terms in PROJ's A/B/C/D order.
    /// Pair it with [`GeoidGrid::from_proj_egm96_gtx`] and select the
    /// [`ProjVgridshiftArithmetic`] used by the reference PROJ build for
    /// bit-exact EGM96 vertical-grid results. Inputs are finite geodetic radians.
    /// Latitude must be within the grid extent; full-world grids wrap every
    /// finite longitude. Invalid coordinates return [`ProjVgridshiftError`]
    /// rather than panicking or extrapolating.
    pub fn undulation_proj_rad(
        &self,
        lat_rad: f64,
        lon_rad: f64,
        arithmetic: ProjVgridshiftArithmetic,
    ) -> Result<f64, ProjVgridshiftError> {
        if !lat_rad.is_finite() {
            return Err(ProjVgridshiftError::NonFiniteCoordinate { field: "latitude" });
        }
        if !lon_rad.is_finite() {
            return Err(ProjVgridshiftError::NonFiniteCoordinate { field: "longitude" });
        }

        let west = self.lon_min_deg * PROJ_DEG_TO_RAD;
        let south = self.lat_min_deg * PROJ_DEG_TO_RAD;
        let res_x = self.dlon_deg * PROJ_DEG_TO_RAD;
        let res_y = self.dlat_deg * PROJ_DEG_TO_RAD;
        let east = (self.lon_min_deg + self.dlon_deg * (self.n_lon as f64 - 1.0)) * PROJ_DEG_TO_RAD;
        let north =
            (self.lat_min_deg + self.dlat_deg * (self.n_lat as f64 - 1.0)) * PROJ_DEG_TO_RAD;
        let inv_res_x = 1.0 / res_x;
        let inv_res_y = 1.0 / res_y;
        let full_world_longitude = east - west + res_x >= 2.0 * core::f64::consts::PI - 1.0e-10;

        if lat_rad < south || lat_rad > north {
            return Err(ProjVgridshiftError::CoordinateOutsideGrid { field: "latitude" });
        }

        let longitude_delta = lon_rad - west;
        let mut grid_x = longitude_delta * inv_res_x;
        if full_world_longitude && !grid_x.is_finite() {
            grid_x = longitude_delta.rem_euclid(2.0 * core::f64::consts::PI) * inv_res_x;
        }
        if lon_rad < west {
            if full_world_longitude {
                let width = self.n_lon as f64;
                grid_x = ((grid_x + width) % width + width) % width;
            } else {
                grid_x = (lon_rad + 2.0 * core::f64::consts::PI - west) * inv_res_x;
            }
        } else if lon_rad > east {
            if full_world_longitude {
                let width = self.n_lon as f64;
                grid_x = ((grid_x + width) % width + width) % width;
            } else {
                grid_x = (lon_rad - 2.0 * core::f64::consts::PI - west) * inv_res_x;
            }
        }
        let mut grid_y = (lat_rad - south) * inv_res_y;

        let max_grid_x = if full_world_longitude {
            self.n_lon as f64
        } else {
            self.n_lon as f64 - 1.0
        };
        if !grid_x.is_finite() || grid_x < 0.0 || grid_x > max_grid_x {
            return Err(ProjVgridshiftError::CoordinateOutsideGrid { field: "longitude" });
        }
        if !grid_y.is_finite() || grid_y < 0.0 || grid_y > self.n_lat as f64 - 1.0 {
            return Err(ProjVgridshiftError::CoordinateOutsideGrid { field: "latitude" });
        }

        let grid_ix = grid_x.floor() as usize;
        let grid_iy = grid_y.floor() as usize;
        if grid_ix >= self.n_lon {
            return Err(ProjVgridshiftError::CoordinateOutsideGrid { field: "longitude" });
        }
        if grid_iy >= self.n_lat {
            return Err(ProjVgridshiftError::CoordinateOutsideGrid { field: "latitude" });
        }
        grid_x -= grid_ix as f64;
        grid_y -= grid_iy as f64;

        let grid_ix2 = if grid_ix + 1 >= self.n_lon {
            if full_world_longitude {
                0
            } else {
                self.n_lon - 1
            }
        } else {
            grid_ix + 1
        };
        let grid_iy2 = (grid_iy + 1).min(self.n_lat - 1);

        let value_a = self.sample(grid_iy, grid_ix);
        let value_b = self.sample(grid_iy, grid_ix2);
        let value_c = self.sample(grid_iy2, grid_ix);
        let value_d = self.sample(grid_iy2, grid_ix2);

        let grid_x_y = grid_x * grid_y;
        let weight_a = 1.0 - grid_x - grid_y + grid_x_y;
        let mut value = value_a * weight_a;
        let weight_b = grid_x - grid_x_y;
        let weight_c = grid_y - grid_x_y;
        let weight_d = grid_x_y;
        match arithmetic {
            ProjVgridshiftArithmetic::SeparateMultiplyAdd => {
                value += value_b * weight_b;
                value += value_c * weight_c;
                value += value_d * weight_d;
            }
            ProjVgridshiftArithmetic::FusedMultiplyAdd => {
                value = libm::fma(value_b, weight_b, value);
                value = libm::fma(value_c, weight_c, value);
                value = libm::fma(value_d, weight_d, value);
            }
        }
        Ok(value)
    }

    /// Batch bilinear undulation lookup for geodetic positions in radians.
    ///
    /// Each input tuple is `(lat_rad, lon_rad)`, with latitude positive north and
    /// longitude positive east. Output element `i` is exactly the scalar
    /// [`undulation_rad`](Self::undulation_rad) result for input element `i`.
    pub fn undulations_rad(&self, points_rad: &[(f64, f64)]) -> Vec<f64> {
        points_rad
            .iter()
            .map(|&(lat_rad, lon_rad)| self.undulation_rad(lat_rad, lon_rad))
            .collect()
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

    /// Batch bilinear undulation lookup for geodetic positions in degrees.
    ///
    /// Each input tuple is `(lat_deg, lon_deg)`, with latitude positive north and
    /// longitude positive east. Output element `i` is exactly the scalar
    /// [`undulation_deg`](Self::undulation_deg) result for input element `i`.
    pub fn undulations_deg(&self, points_deg: &[(f64, f64)]) -> Vec<f64> {
        points_deg
            .iter()
            .map(|&(lat_deg, lon_deg)| self.undulation_deg(lat_deg, lon_deg))
            .collect()
    }

    /// Orthometric height `H = h - N` (metres above mean sea level) from an
    /// ellipsoidal height and a geodetic position in radians, using this grid's
    /// undulation.
    pub fn orthometric_height_rad(
        &self,
        ellipsoidal_height_m: f64,
        lat_rad: f64,
        lon_rad: f64,
    ) -> f64 {
        ellipsoidal_height_m - self.undulation_rad(lat_rad, lon_rad)
    }

    /// Ellipsoidal height `h = H + N` (metres above the WGS84 ellipsoid) from an
    /// orthometric height and a geodetic position in radians, using this grid's
    /// undulation.
    pub fn ellipsoidal_height_rad(
        &self,
        orthometric_height_m: f64,
        lat_rad: f64,
        lon_rad: f64,
    ) -> f64 {
        orthometric_height_m + self.undulation_rad(lat_rad, lon_rad)
    }

    /// Orthometric height `H = h - N` (metres above mean sea level) from an
    /// ellipsoidal height and a geodetic position in degrees, using this grid's
    /// undulation.
    pub fn orthometric_height_deg(
        &self,
        ellipsoidal_height_m: f64,
        lat_deg: f64,
        lon_deg: f64,
    ) -> f64 {
        ellipsoidal_height_m - self.undulation_deg(lat_deg, lon_deg)
    }

    /// Ellipsoidal height `h = H + N` (metres above the WGS84 ellipsoid) from an
    /// orthometric height and a geodetic position in degrees, using this grid's
    /// undulation.
    pub fn ellipsoidal_height_deg(
        &self,
        orthometric_height_m: f64,
        lat_deg: f64,
        lon_deg: f64,
    ) -> f64 {
        orthometric_height_m + self.undulation_deg(lat_deg, lon_deg)
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

/// Batch geoid undulation lookup against the COARSE built-in global grid.
///
/// Each input tuple is `(lat_rad, lon_rad)`, with latitude positive north and
/// longitude positive east. Output element `i` is exactly the scalar
/// [`geoid_undulation`] result for input element `i`.
pub fn geoid_undulations_rad(points_rad: &[(f64, f64)]) -> Vec<f64> {
    builtin_grid().undulations_rad(points_rad)
}

/// Batch geoid undulation lookup against the COARSE built-in global grid.
///
/// Each input tuple is `(lat_deg, lon_deg)`, with latitude positive north and
/// longitude positive east.
pub fn geoid_undulations_deg(points_deg: &[(f64, f64)]) -> Vec<f64> {
    builtin_grid().undulations_deg(points_deg)
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

/// Byte length of a GTX header (four big-endian f64 fields and two i32 fields).
const PROJ_EGM96_GTX_HEADER_BYTES: usize = 40;
/// Latitude row count of PROJ's EGM96 15-arcminute GTX grid.
const PROJ_EGM96_GTX_N_LAT: usize = 721;
/// Longitude column count of PROJ's EGM96 15-arcminute GTX grid.
const PROJ_EGM96_GTX_N_LON: usize = 1440;
/// PROJ 9.3.0's `DEG_TO_RAD` binary64 constant from `proj_internal.h`.
const PROJ_DEG_TO_RAD: f64 = 0.017453292519943296;

fn read_be_f64(bytes: &[u8], offset: usize) -> f64 {
    f64::from_be_bytes(
        bytes[offset..offset + 8]
            .try_into()
            .expect("validated GTX header length"),
    )
}

fn read_be_i32(bytes: &[u8], offset: usize) -> i32 {
    i32::from_be_bytes(
        bytes[offset..offset + 4]
            .try_into()
            .expect("validated GTX header length"),
    )
}

/// Latitude row count of the NGA EGM2008 1-arcminute raster.
const EGM2008_1_MIN_N_LAT: usize = 10801;
/// Longitude column count of the NGA EGM2008 1-arcminute raster.
const EGM2008_1_MIN_N_LON: usize = 21600;
/// Latitude row count of the NGA EGM2008 2.5-arcminute raster.
const EGM2008_2P5_MIN_N_LAT: usize = 4321;
/// Longitude column count of the NGA EGM2008 2.5-arcminute raster.
const EGM2008_2P5_MIN_N_LON: usize = 8640;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RasterEndian {
    Little,
    Big,
}

impl RasterEndian {
    fn read_u32(self, bytes: [u8; 4]) -> u32 {
        match self {
            Self::Little => u32::from_le_bytes(bytes),
            Self::Big => u32::from_be_bytes(bytes),
        }
    }

    fn read_f32(self, bytes: [u8; 4]) -> f32 {
        match self {
            Self::Little => f32::from_le_bytes(bytes),
            Self::Big => f32::from_be_bytes(bytes),
        }
    }
}

fn parse_egm2008_raster_values(
    bytes: &[u8],
    window: Egm2008RasterWindow,
) -> Result<Vec<f64>, GeoidError> {
    let row_value_bytes = window
        .n_lon
        .checked_mul(4)
        .ok_or_else(|| GeoidError::Parse {
            reason: "EGM2008 row byte count overflows".to_string(),
        })?;
    let row_record_bytes = row_value_bytes
        .checked_add(8)
        .ok_or_else(|| GeoidError::Parse {
            reason: "EGM2008 record byte count overflows".to_string(),
        })?;
    let expected = window
        .n_lat
        .checked_mul(row_record_bytes)
        .ok_or_else(|| GeoidError::Parse {
            reason: "EGM2008 raster byte count overflows".to_string(),
        })?;
    if bytes.len() != expected {
        return Err(GeoidError::Parse {
            reason: format!(
                "EGM2008 raster window must be {expected} bytes ({} x {} REAL*4 row records), got {}",
                window.n_lat,
                window.n_lon,
                bytes.len()
            ),
        });
    }

    let row_marker = u32::try_from(row_value_bytes).map_err(|_| GeoidError::Parse {
        reason: "EGM2008 row marker exceeds u32".to_string(),
    })?;
    let endian = detect_egm2008_endian(bytes, row_marker)?;
    let mut values_m = vec![0.0f64; window.n_lat * window.n_lon];
    for src_row in 0..window.n_lat {
        let row_off = src_row * row_record_bytes;
        let start_marker = endian.read_u32([
            bytes[row_off],
            bytes[row_off + 1],
            bytes[row_off + 2],
            bytes[row_off + 3],
        ]);
        let end_off = row_off + 4 + row_value_bytes;
        let end_marker = endian.read_u32([
            bytes[end_off],
            bytes[end_off + 1],
            bytes[end_off + 2],
            bytes[end_off + 3],
        ]);
        if start_marker != row_marker || end_marker != row_marker {
            return Err(GeoidError::Parse {
                reason: format!(
                    "EGM2008 record {src_row} marker mismatch: start {start_marker}, end {end_marker}, expected {row_marker}"
                ),
            });
        }
        let dst_row = window.n_lat - 1 - src_row;
        for col in 0..window.n_lon {
            let sample_off = row_off + 4 + col * 4;
            let value = endian.read_f32([
                bytes[sample_off],
                bytes[sample_off + 1],
                bytes[sample_off + 2],
                bytes[sample_off + 3],
            ]);
            let index = dst_row * window.n_lon + col;
            if !value.is_finite() {
                return Err(GeoidError::NonFiniteValue { index });
            }
            values_m[index] = f64::from(value);
        }
    }
    Ok(values_m)
}

fn detect_egm2008_endian(bytes: &[u8], row_marker: u32) -> Result<RasterEndian, GeoidError> {
    if bytes.len() < 4 {
        return Err(GeoidError::Parse {
            reason: "EGM2008 raster is too short for a record marker".to_string(),
        });
    }
    let first = [bytes[0], bytes[1], bytes[2], bytes[3]];
    let little = u32::from_le_bytes(first) == row_marker;
    let big = u32::from_be_bytes(first) == row_marker;
    match (little, big) {
        (true, false) => Ok(RasterEndian::Little),
        (false, true) => Ok(RasterEndian::Big),
        (true, true) => Err(GeoidError::Parse {
            reason: "EGM2008 record marker has ambiguous byte order".to_string(),
        }),
        (false, false) => Err(GeoidError::Parse {
            reason: format!("EGM2008 first record marker does not match {row_marker}"),
        }),
    }
}

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

/// Batch geoid undulation lookup against the embedded GENUINE EGM96 1-degree
/// global grid.
///
/// Each input tuple is `(lat_rad, lon_rad)`, with latitude positive north and
/// longitude positive east. Output element `i` is exactly the scalar
/// [`egm96_undulation`] result for input element `i`.
pub fn egm96_undulations_rad(points_rad: &[(f64, f64)]) -> Vec<f64> {
    egm96_grid().undulations_rad(points_rad)
}

/// Batch geoid undulation lookup against the embedded GENUINE EGM96 1-degree
/// global grid.
///
/// Each input tuple is `(lat_deg, lon_deg)`, with latitude positive north and
/// longitude positive east.
pub fn egm96_undulations_deg(points_deg: &[(f64, f64)]) -> Vec<f64> {
    egm96_grid().undulations_deg(points_deg)
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
    //! Geoid validation provenance:
    //!
    //! EGM96 PROJ fixtures use `us_nga_egm96_15.tif` through PROJ `cct` and
    //! assert 5 mm agreement with sparse real `WW15MGH.DAC` centimetre nodes.
    //!
    //! The contracted-arithmetic EGM96 fixture is generated from an AArch64
    //! build of public PROJ tag 9.3.0 `read_vgrid_value` and the public OSGeo
    //! `egm96_15.gtx` (SHA-256
    //! `c02a6eb70a7a78efebe5adf3ade626eb75390e170bb8b3f36136a2c28f5326a0`).
    //! It contains 13,051 radian input/result bit triples and the 52,012 source
    //! float nodes needed by those points. The generator also requires a second
    //! PROJ build exposed through pyproj to match every result at 0 ULP. Fixture
    //! SHA-256:
    //! `6de70d99b857ea1cf8efa6e820537f0b30742fe65cc0966ff1f4439a13c7e966`.
    //!
    //! EGM2008 fixtures use the public NGA `EGM2008_Interpolation_Grid.zip`
    //! archive from `https://earth-info.nga.mil/php/download.php?file=egm-08interpolation`.
    //! Archive SHA-256:
    //! `0f65f16e6fd3f89a6b8022d7a89375d0c29fb275a551927175669bb610904cd0`.
    //! Source member:
    //! `Und_min2.5x2.5_egm2008_isw=82_WGS84_TideFree_SE`.
    //! Source raster SHA-256:
    //! `ab6f8b94076f78707d1cdae7b066b93a786a0c64b52449e20cf1a1a2f4e74daf`.
    //! Crop fixture:
    //! `tests/fixtures/geoid/egm2008_25_norcal_crop.bin`, 25 x 25 nodes,
    //! 2.5 arcminute spacing, latitude 37.0 to 38.0 degrees, longitude
    //! -123.0 to -122.0 degrees. Crop SHA-256:
    //! `e66da6cbde7bb4015dc8b9c436fd93f16af3734e97017700fa3ab632f71f569d`.
    //! The crop preserves the NGA small-endian `REAL*4` row records for those
    //! nodes, with record lengths reduced to the cropped row width.
    //!
    //! EGM2008 oracle values use PROJ 9.8.1 `cct` with
    //! `us_nga_egm08_25.tif`, SHA-256
    //! `4191d471eefebf24091b56dbc604353cb3b8cf8cc70e448bb9ae56a272bef17a`.
    //! Command:
    //! `PROJ_DATA=/Volumes/ExternalSSD/sidereon-fleet/.tmp-egm2008/proj cct -d 12 +proj=pipeline +step +inv +proj=vgridshift +grids=us_nga_egm08_25.tif +multiplier=1`.
    //! With input height zero, the undulation is `-output_z`. The crop test
    //! asserts agreement to 5 mm.

    use super::*;

    #[derive(Clone, Copy)]
    struct ProjGeoidFixture {
        lat_deg: f64,
        lon_deg: f64,
        undulation_m: f64,
    }

    const EGM2008_NORCAL_CROP_BYTES: &[u8] =
        include_bytes!("../tests/fixtures/geoid/egm2008_25_norcal_crop.bin");
    const PROJ_EGM96_930_DENSE_BYTES: &[u8] =
        include_bytes!("../tests/fixtures/geoid/proj_egm96_930_dense.bin");

    // PROJ oracle provenance for the 15-arcminute EGM96 fixture below:
    //
    // Tool: PROJ 9.8.1 (`cct`, Rel. 9.8.1, April 10th, 2026).
    // Grid: `us_nga_egm96_15.tif`, fetched with
    // `projsync --target-dir /tmp/sidereon-proj-egm96 --file us_nga_egm96_15.tif`.
    // Grid SHA-256:
    // db493027562c9b004d7220fa881f5603adada4e1c5029b933fa7de4547b0e78d.
    // Command:
    // `PROJ_DATA=/tmp/sidereon-proj-egm96 cct -d 12 +proj=pipeline +step +inv
    //  +proj=vgridshift +grids=us_nga_egm96_15.tif +multiplier=1`.
    //
    // `cct` returns orthometric height for an ellipsoidal-height input. With
    // input height 0, the geoid undulation is `-output_z`.
    const PROJ_EGM96_FIXTURES: &[ProjGeoidFixture] = &[
        ProjGeoidFixture {
            lat_deg: 0.000000,
            lon_deg: 0.000000,
            undulation_m: 17.161579132080,
        },
        ProjGeoidFixture {
            lat_deg: 0.000000,
            lon_deg: 80.000000,
            undulation_m: -102.687904357910,
        },
        ProjGeoidFixture {
            lat_deg: 60.000000,
            lon_deg: -30.000000,
            undulation_m: 63.799266815186,
        },
        ProjGeoidFixture {
            lat_deg: 45.625000,
            lon_deg: 12.375000,
            undulation_m: 44.181870460510,
        },
        ProjGeoidFixture {
            lat_deg: 0.125000,
            lon_deg: 179.875000,
            undulation_m: 21.099070549011,
        },
        ProjGeoidFixture {
            lat_deg: 0.125000,
            lon_deg: -179.875000,
            undulation_m: 20.864660263062,
        },
        ProjGeoidFixture {
            lat_deg: -10.500000,
            lon_deg: 179.990000,
            undulation_m: 38.607539978027,
        },
        ProjGeoidFixture {
            lat_deg: -10.500000,
            lon_deg: -179.990000,
            undulation_m: 38.540365447998,
        },
        ProjGeoidFixture {
            lat_deg: 89.875000,
            lon_deg: 45.000000,
            undulation_m: 13.639517307281,
        },
        ProjGeoidFixture {
            lat_deg: -89.875000,
            lon_deg: 123.625000,
            undulation_m: -29.676423549652,
        },
        ProjGeoidFixture {
            lat_deg: 37.774900,
            lon_deg: -122.419400,
            undulation_m: -32.242452185586,
        },
    ];

    const PROJ_EGM2008_FIXTURES: &[ProjGeoidFixture] = &[
        ProjGeoidFixture {
            lat_deg: 37.774900,
            lon_deg: -122.419400,
            undulation_m: -32.163558372373,
        },
        ProjGeoidFixture {
            lat_deg: 37.500000,
            lon_deg: -122.750000,
            undulation_m: -33.605857849121,
        },
        ProjGeoidFixture {
            lat_deg: 37.875000,
            lon_deg: -122.125000,
            undulation_m: -31.847370147705,
        },
        ProjGeoidFixture {
            lat_deg: 38.000000,
            lon_deg: -122.000000,
            undulation_m: -31.767843246460,
        },
        ProjGeoidFixture {
            lat_deg: 37.000000,
            lon_deg: -123.000000,
            undulation_m: -36.499370574951,
        },
    ];

    // Real EGM96 15-arcminute node values, rounded to the centimetre grid the
    // NGA `WW15MGH.DAC` format stores. The sparse test grid writes only these
    // nodes into an otherwise-zero DAC-sized byte buffer; each oracle point
    // above falls in a cell whose four corners are present here. This avoids
    // committing the full 2 MB grid while still checking node registration,
    // antimeridian wrap, pole-row handling, and bilinear cell selection against
    // PROJ-derived values. The largest measured PROJ-vs-DAC-centimetre
    // difference in these fixtures is 0.0032 m.
    const SPARSE_EGM96_DAC_NODES_CM: &[(f64, f64, i16)] = &[
        (-90.00, 123.50, -2953),
        (-90.00, 123.75, -2953),
        (-89.75, 123.50, -2982),
        (-89.75, 123.75, -2982),
        (-10.50, 179.75, 3919),
        (-10.50, 180.00, 3858),
        (-10.50, 180.25, 3751),
        (-10.25, 179.75, 3733),
        (-10.25, 180.00, 3697),
        (-10.25, 180.25, 3611),
        (0.00, 0.00, 1716),
        (0.00, 0.25, 1708),
        (0.00, 80.00, -10269),
        (0.00, 80.25, -10255),
        (0.00, 179.75, 2138),
        (0.00, 180.00, 2115),
        (0.00, 180.25, 2095),
        (0.25, 0.00, 1719),
        (0.25, 0.25, 1711),
        (0.25, 80.00, -10286),
        (0.25, 80.25, -10276),
        (0.25, 179.75, 2109),
        (0.25, 180.00, 2078),
        (0.25, 180.25, 2058),
        (37.75, 237.50, -3237),
        (37.75, 237.75, -3204),
        (38.00, 237.50, -3211),
        (38.00, 237.75, -3200),
        (45.50, 12.25, 4398),
        (45.50, 12.50, 4355),
        (45.75, 12.25, 4498),
        (45.75, 12.50, 4421),
        (60.00, 330.00, 6380),
        (60.00, 330.25, 6400),
        (60.25, 330.00, 6365),
        (60.25, 330.25, 6388),
        (89.75, 45.00, 1367),
        (89.75, 45.25, 1367),
        (90.00, 45.00, 1361),
        (90.00, 45.25, 1361),
    ];

    fn sparse_egm96_dac_bytes() -> Vec<u8> {
        let mut bytes = vec![0u8; super::EGM96_DAC_N_LAT * super::EGM96_DAC_N_LON * 2];
        for &(lat_deg, lon_east_deg, cm) in SPARSE_EGM96_DAC_NODES_CM {
            let record = ((90.0 - lat_deg) / 0.25).round() as usize;
            let col = (lon_east_deg.rem_euclid(360.0) / 0.25).round() as usize;
            assert!(record < super::EGM96_DAC_N_LAT);
            assert!(col < super::EGM96_DAC_N_LON);
            let off = (record * super::EGM96_DAC_N_LON + col) * 2;
            bytes[off..off + 2].copy_from_slice(&cm.to_be_bytes());
        }
        bytes
    }

    fn egm2008_norcal_window() -> Egm2008RasterWindow {
        Egm2008RasterWindow::new(Egm2008GridSpacing::TwoPointFiveMinute, 37.0, -123.0, 25, 25)
            .expect("EGM2008 crop window")
    }

    fn egm2008_test_raster_bytes(
        window: Egm2008RasterWindow,
        little_endian: bool,
        value: impl Fn(usize, usize) -> f32,
    ) -> Vec<u8> {
        let row_value_bytes = window.n_lon() * 4;
        let mut bytes = Vec::with_capacity(window.n_lat() * (row_value_bytes + 8));
        for src_row in 0..window.n_lat() {
            if little_endian {
                bytes.extend_from_slice(&(row_value_bytes as u32).to_le_bytes());
            } else {
                bytes.extend_from_slice(&(row_value_bytes as u32).to_be_bytes());
            }
            for col in 0..window.n_lon() {
                let sample = value(src_row, col);
                if little_endian {
                    bytes.extend_from_slice(&sample.to_le_bytes());
                } else {
                    bytes.extend_from_slice(&sample.to_be_bytes());
                }
            }
            if little_endian {
                bytes.extend_from_slice(&(row_value_bytes as u32).to_le_bytes());
            } else {
                bytes.extend_from_slice(&(row_value_bytes as u32).to_be_bytes());
            }
        }
        bytes
    }

    fn sparse_proj_egm96_gtx_from_dense_fixture() -> (Vec<u8>, usize, usize) {
        assert_eq!(&PROJ_EGM96_930_DENSE_BYTES[..8], b"SIDGEO93");
        let node_count = u32::from_be_bytes(
            PROJ_EGM96_930_DENSE_BYTES[8..12]
                .try_into()
                .expect("fixture node count"),
        ) as usize;
        let point_count = u32::from_be_bytes(
            PROJ_EGM96_930_DENSE_BYTES[12..16]
                .try_into()
                .expect("fixture point count"),
        ) as usize;
        let point_offset = 16 + node_count * 8;
        assert_eq!(
            PROJ_EGM96_930_DENSE_BYTES.len(),
            point_offset + point_count * 24
        );

        let mut gtx = vec![
            0u8;
            super::PROJ_EGM96_GTX_HEADER_BYTES
                + super::PROJ_EGM96_GTX_N_LAT * super::PROJ_EGM96_GTX_N_LON * 4
        ];
        gtx[0..8].copy_from_slice(&(-90.0f64).to_be_bytes());
        gtx[8..16].copy_from_slice(&(-180.0f64).to_be_bytes());
        gtx[16..24].copy_from_slice(&0.25f64.to_be_bytes());
        gtx[24..32].copy_from_slice(&0.25f64.to_be_bytes());
        gtx[32..36].copy_from_slice(&(super::PROJ_EGM96_GTX_N_LAT as i32).to_be_bytes());
        gtx[36..40].copy_from_slice(&(super::PROJ_EGM96_GTX_N_LON as i32).to_be_bytes());

        for record in PROJ_EGM96_930_DENSE_BYTES[16..point_offset]
            .as_chunks::<8>()
            .0
        {
            let index = u32::from_be_bytes(record[..4].try_into().expect("fixture node index"));
            let offset = super::PROJ_EGM96_GTX_HEADER_BYTES + index as usize * 4;
            gtx[offset..offset + 4].copy_from_slice(&record[4..8]);
        }
        (gtx, point_offset, point_count)
    }

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
    fn batch_undulation_entries_match_scalar_lookup() {
        let points_deg = [(0.0, 0.0), (45.625, 12.375), (0.125, -179.875)];
        let got_deg = egm96_undulations_deg(&points_deg);
        let expected_deg: Vec<f64> = points_deg
            .iter()
            .map(|&(lat, lon)| egm96_grid().undulation_deg(lat, lon))
            .collect();
        assert_eq!(got_deg, expected_deg);

        let points_rad: Vec<(f64, f64)> = points_deg
            .iter()
            .map(|&(lat, lon)| (lat.to_radians(), lon.to_radians()))
            .collect();
        let got_rad = egm96_undulations_rad(&points_rad);
        let expected_rad: Vec<f64> = points_rad
            .iter()
            .map(|&(lat, lon)| egm96_undulation(lat, lon))
            .collect();
        assert_eq!(got_rad, expected_rad);

        assert_eq!(
            geoid_undulations_deg(&points_deg),
            points_deg
                .iter()
                .map(|&(lat, lon)| geoid_undulation(lat.to_radians(), lon.to_radians()))
                .collect::<Vec<_>>()
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
    fn from_egm2008_raster_window_decodes_little_and_big_endian_records() {
        let d = Egm2008GridSpacing::TwoPointFiveMinute.degrees();
        let window =
            Egm2008RasterWindow::new(Egm2008GridSpacing::TwoPointFiveMinute, 10.0, 20.0, 2, 3)
                .expect("EGM2008 test window");
        for little_endian in [true, false] {
            let bytes = egm2008_test_raster_bytes(window, little_endian, |src_row, col| {
                (src_row * 10 + col) as f32
            });
            let grid =
                GeoidGrid::from_egm2008_raster_window(&bytes, window).expect("parse EGM2008");
            assert_eq!(grid.undulation_deg(10.0, 20.0), 10.0);
            assert!((grid.undulation_deg(10.0 + d, 20.0 + 2.0 * d) - 2.0).abs() <= 1.0e-12);
            assert!((grid.undulation_deg(10.0 + 0.5 * d, 20.0 + d) - 6.0).abs() <= 1.0e-12);
        }
    }

    #[test]
    fn from_egm2008_raster_rejects_bad_record_layout() {
        assert!(matches!(
            GeoidGrid::from_egm2008_raster(
                EGM2008_NORCAL_CROP_BYTES,
                Egm2008GridSpacing::TwoPointFiveMinute,
            ),
            Err(GeoidError::Parse { .. })
        ));

        let window = egm2008_norcal_window();
        let mut bytes = EGM2008_NORCAL_CROP_BYTES.to_vec();
        bytes[0] = 0;
        assert!(matches!(
            GeoidGrid::from_egm2008_raster_window(&bytes, window),
            Err(GeoidError::Parse { .. })
        ));
    }

    #[test]
    fn egm2008_crop_matches_proj_oracle() {
        let grid = GeoidGrid::from_egm2008_raster_window(
            EGM2008_NORCAL_CROP_BYTES,
            egm2008_norcal_window(),
        )
        .expect("parse EGM2008 crop");
        for fixture in PROJ_EGM2008_FIXTURES {
            let got = grid.undulation_deg(fixture.lat_deg, fixture.lon_deg);
            assert!(
                (got - fixture.undulation_m).abs() <= 0.005,
                "PROJ EGM2008 fixture ({}, {}): got {got}, want {}",
                fixture.lat_deg,
                fixture.lon_deg,
                fixture.undulation_m
            );
        }
    }

    #[test]
    fn egm2008_regional_crop_clamps_grid_edges() {
        let grid = GeoidGrid::from_egm2008_raster_window(
            EGM2008_NORCAL_CROP_BYTES,
            egm2008_norcal_window(),
        )
        .expect("parse EGM2008 crop");
        assert_eq!(
            grid.undulation_deg(36.0, -124.0),
            grid.undulation_deg(37.0, -123.0)
        );
        assert_eq!(
            grid.undulation_deg(39.0, -121.0),
            grid.undulation_deg(38.0, -122.0)
        );
    }

    #[test]
    fn egm2008_global_longitude_window_wraps_through_shared_kernel() {
        let spacing = Egm2008GridSpacing::TwoPointFiveMinute;
        let d = spacing.degrees();
        let (_, n_lon) = spacing.global_dimensions();
        let window = Egm2008RasterWindow::new(spacing, 0.0, 0.0, 2, n_lon)
            .expect("global-longitude EGM2008 window");
        let bytes = egm2008_test_raster_bytes(window, true, |_, col| {
            if col == 0 {
                100.0
            } else if col == n_lon - 1 {
                200.0
            } else {
                10.0
            }
        });
        let grid =
            GeoidGrid::from_egm2008_raster_window(&bytes, window).expect("parse EGM2008 wrap");

        assert_eq!(grid.undulation_deg(0.0, 360.0), 100.0);
        assert_eq!(grid.undulation_deg(0.0, -0.5 * d), 150.0);
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

    #[test]
    fn egm96_embedded_outputs_are_bit_pinned() {
        let fixtures = [
            (37.0_f64, -122.0_f64, 0xc040_accc_cccc_cccdu64),
            (37.5_f64, -122.5_f64, 0xc040_de66_6666_6666u64),
            (
                16.0 + 46.0 / 60.0 + 33.0 / 3600.0,
                -(3.0 + 34.0 / 3600.0),
                0x403c_acb4_79a8_1af4u64,
            ),
            (0.125_f64, -179.875_f64, 0x4034_cbf5_c28f_5c29u64),
        ];
        for (lat_deg, lon_deg, bits) in fixtures {
            let got = egm96_undulation(lat_deg.to_radians(), lon_deg.to_radians());
            assert_eq!(
                got.to_bits(),
                bits,
                "EGM96 bit pin ({lat_deg}, {lon_deg}) got {got}"
            );
        }
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

    #[test]
    fn from_proj_egm96_gtx_rejects_wrong_layout_and_nonfinite_samples() {
        let (mut bytes, _, _) = sparse_proj_egm96_gtx_from_dense_fixture();
        assert!(matches!(
            GeoidGrid::from_proj_egm96_gtx(&bytes[..bytes.len() - 4]),
            Err(GeoidError::Parse { .. })
        ));

        bytes[16..24].copy_from_slice(&0.5f64.to_be_bytes());
        assert!(matches!(
            GeoidGrid::from_proj_egm96_gtx(&bytes),
            Err(GeoidError::Parse { .. })
        ));

        bytes[16..24].copy_from_slice(&0.25f64.to_be_bytes());
        bytes[super::PROJ_EGM96_GTX_HEADER_BYTES..super::PROJ_EGM96_GTX_HEADER_BYTES + 4]
            .copy_from_slice(&f32::NAN.to_be_bytes());
        assert_eq!(
            GeoidGrid::from_proj_egm96_gtx(&bytes),
            Err(GeoidError::NonFiniteValue { index: 0 })
        );
    }

    #[test]
    fn proj_930_egm96_fused_dense_sample_is_zero_ulp() {
        let (bytes, point_offset, point_count) = sparse_proj_egm96_gtx_from_dense_fixture();
        let grid = GeoidGrid::from_proj_egm96_gtx(&bytes).expect("parse sparse public PROJ GTX");

        for (index, record) in PROJ_EGM96_930_DENSE_BYTES[point_offset..]
            .as_chunks::<24>()
            .0
            .iter()
            .enumerate()
        {
            let lon_rad = f64::from_bits(u64::from_be_bytes(
                record[..8].try_into().expect("fixture longitude"),
            ));
            let lat_rad = f64::from_bits(u64::from_be_bytes(
                record[8..16].try_into().expect("fixture latitude"),
            ));
            let expected_bits =
                u64::from_be_bytes(record[16..24].try_into().expect("fixture undulation"));
            let got = grid
                .undulation_proj_rad(lat_rad, lon_rad, ProjVgridshiftArithmetic::FusedMultiplyAdd)
                .expect("fixture coordinate is inside the grid");
            assert_eq!(
                got.to_bits(),
                expected_bits,
                "contracted PROJ 9.3.0 EGM96 point {index}/{point_count}: lat={lat_rad}, lon={lon_rad}, got={got}"
            );
        }
        assert_eq!(point_count, 13_051);
    }

    #[test]
    fn proj_930_egm96_separate_multiply_add_bits_are_pinned() {
        let (bytes, _, _) = sparse_proj_egm96_gtx_from_dense_fixture();
        let grid = GeoidGrid::from_proj_egm96_gtx(&bytes).expect("parse sparse public PROJ GTX");

        // These cases are spread across the dense fixture and differ by one ULP
        // from contracted PROJ builds. Expected values come from the reviewed
        // PROJ 9.3.0 source sequence with contraction disabled.
        let cases = [
            (
                0xbff9_21e9_072f_0bff,
                0x3fb4_1b28_2494_7d44,
                0xc03d_88b2_3abe_f2f0,
            ),
            (
                0xbfd9_21e9_072f_0bfc,
                0x4006_eef9_c9b9_5ed9,
                0x4049_ff99_e8a7_6f47,
            ),
            (
                0x3fd9_21e9_072f_0c01,
                0x4004_6b94_c526_cf33,
                0x403d_018d_8044_d991,
            ),
            (
                0x3fef_6a63_48fa_cf01,
                0x3ffb_a557_324c_2c33,
                0xc044_4b8e_0e1f_a00c,
            ),
            (
                0x3ff9_21e9_072f_0bff,
                0x4009_21f2_2db9_9c8b,
                0x402b_362a_4459_31d0,
            ),
        ];
        for (lat_bits, lon_bits, expected_bits) in cases {
            let got = grid
                .undulation_proj_rad(
                    f64::from_bits(lat_bits),
                    f64::from_bits(lon_bits),
                    ProjVgridshiftArithmetic::SeparateMultiplyAdd,
                )
                .expect("fixture coordinate is inside the grid");
            assert_eq!(got.to_bits(), expected_bits);
        }
    }

    #[test]
    fn proj_lookup_rejects_invalid_coordinates_without_panicking() {
        let (bytes, _, _) = sparse_proj_egm96_gtx_from_dense_fixture();
        let grid = GeoidGrid::from_proj_egm96_gtx(&bytes).expect("parse sparse public PROJ GTX");
        let arithmetic = ProjVgridshiftArithmetic::FusedMultiplyAdd;

        assert_eq!(
            grid.undulation_proj_rad(f64::NAN, 0.0, arithmetic),
            Err(ProjVgridshiftError::NonFiniteCoordinate { field: "latitude" })
        );
        assert_eq!(
            grid.undulation_proj_rad(0.0, f64::INFINITY, arithmetic),
            Err(ProjVgridshiftError::NonFiniteCoordinate { field: "longitude" })
        );
        assert_eq!(
            grid.undulation_proj_rad(-91.0 * super::PROJ_DEG_TO_RAD, 0.0, arithmetic),
            Err(ProjVgridshiftError::CoordinateOutsideGrid { field: "latitude" })
        );
        assert_eq!(
            grid.undulation_proj_rad(91.0 * super::PROJ_DEG_TO_RAD, 0.0, arithmetic),
            Err(ProjVgridshiftError::CoordinateOutsideGrid { field: "latitude" })
        );
        assert!(grid
            .undulation_proj_rad(0.0, f64::MAX, arithmetic)
            .expect("full-world grids wrap every finite longitude")
            .is_finite());
    }

    #[test]
    fn egm96_dac_sparse_fixture_matches_proj_oracle() {
        let bytes = sparse_egm96_dac_bytes();
        let grid = GeoidGrid::from_egm96_dac(&bytes).expect("parse sparse EGM96 DAC fixture");
        for fixture in PROJ_EGM96_FIXTURES {
            let got = grid.undulation_deg(fixture.lat_deg, fixture.lon_deg);
            assert!(
                (got - fixture.undulation_m).abs() <= 0.005,
                "PROJ EGM96 fixture ({}, {}): got {got}, want {}",
                fixture.lat_deg,
                fixture.lon_deg,
                fixture.undulation_m
            );
        }
    }

    #[test]
    fn geoid_grid_height_conversions_pin_sign_convention() {
        let bytes = sparse_egm96_dac_bytes();
        let grid = GeoidGrid::from_egm96_dac(&bytes).expect("parse sparse EGM96 DAC fixture");
        for fixture in PROJ_EGM96_FIXTURES {
            let n = grid.undulation_deg(fixture.lat_deg, fixture.lon_deg);
            let h = 250.0;
            let orthometric = grid.orthometric_height_deg(h, fixture.lat_deg, fixture.lon_deg);
            assert_eq!(orthometric, h - n);
            assert_eq!(
                grid.ellipsoidal_height_deg(orthometric, fixture.lat_deg, fixture.lon_deg),
                orthometric + n
            );

            let lat_rad = fixture.lat_deg.to_radians();
            let lon_rad = fixture.lon_deg.to_radians();
            assert!((grid.orthometric_height_rad(h, lat_rad, lon_rad) - (h - n)).abs() <= 1.0e-12);
            assert!(
                (grid.ellipsoidal_height_rad(orthometric, lat_rad, lon_rad) - (orthometric + n))
                    .abs()
                    <= 1.0e-12
            );
        }
    }
}
