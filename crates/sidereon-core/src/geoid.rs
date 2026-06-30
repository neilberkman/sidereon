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
//! - [`geoid_undulation`], a zero-setup lookup against a small COARSE built-in
//!   global grid, plus [`orthometric_height_m`] / [`ellipsoidal_height_m`] height
//!   conversion helpers.
//!
//! ## Built-in grid vs. loading a real model
//!
//! Embedding a full-resolution EGM grid (EGM2008 is a 1-minute, ~2.3 GB grid)
//! is impractical to vendor into the crate, so the built-in grid is a COARSE
//! 30-degree global field. It reproduces the large-scale character of the geoid
//! (the Indian Ocean low, the North Atlantic / New Guinea highs, the polar
//! offsets) and is suitable for tests, sanity checks, and metre-scale fallback,
//! but it is NOT survey-grade. Production code should load a real model through
//! [`GeoidGrid::from_text`] (or build a [`GeoidGrid`] from any parsed source)
//! and call [`GeoidGrid::undulation_rad`] directly.

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
/// ellipsoidal height and a geodetic position in radians, using the built-in
/// grid's undulation. For a real model, subtract
/// [`GeoidGrid::undulation_rad`] directly.
pub fn orthometric_height_m(ellipsoidal_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    ellipsoidal_height_m - geoid_undulation(lat_rad, lon_rad)
}

/// Ellipsoidal height `h = H + N` (metres above the WGS84 ellipsoid) from an
/// orthometric height and a geodetic position in radians, using the built-in
/// grid's undulation. For a real model, add [`GeoidGrid::undulation_rad`]
/// directly.
pub fn ellipsoidal_height_m(orthometric_height_m: f64, lat_rad: f64, lon_rad: f64) -> f64 {
    orthometric_height_m + geoid_undulation(lat_rad, lon_rad)
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
}
