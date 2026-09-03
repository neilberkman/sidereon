//! Regular-grid TEC ionosphere delay variant.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use crate::astro::math::vec3::{
    dot3_fused_z_yx_ref as dot_three_fused, unit3_ref_unchecked as unit_vector,
};

use crate::constants::DEG_TO_RAD;
pub use crate::constants::MEAN_EARTH_RADIUS_M as EARTH_RADIUS_M;
use crate::frequencies::{self, CarrierBand};
use crate::validate;
use crate::GnssSystem;

pub const IONOSPHERE_HEIGHT_M: f64 = 450_000.0;
pub const IONOSPHERE_CONSTANT: f64 = 40.308193 * 1e16;

/// Error returned when a regular TEC grid or one of its queries is invalid.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TecGridError {
    /// One or more grid axes have fewer than two nodes.
    #[error("TEC grid axes must each contain at least two entries")]
    AxesTooShort,
    /// A grid axis is not strictly increasing.
    #[error("TEC grid axes must be strictly increasing")]
    AxesNotIncreasing,
    /// The product of the axis lengths overflowed `usize`.
    #[error("TEC grid dimensions overflow")]
    DimensionsOverflow,
    /// The number of values does not match the grid dimensions.
    #[error("TEC grid has {actual} values but expected {expected}")]
    ValueCountMismatch {
        /// Number of values supplied to [`TecGrid::new`] when it did not match the checked product of the three axis lengths.
        actual: usize,
        /// Checked epoch-by-latitude-by-longitude axis-length product required for the flat value vector.
        expected: usize,
    },
    /// A named input failed a shared validation rule.
    #[error("{field} {reason}")]
    InvalidField {
        /// Stable label returned by `validate::FieldError::field()` for the rejected input.
        field: &'static str,
        /// Short reason returned by `validate::FieldError::reason()` for the rejected input.
        reason: &'static str,
    },
    /// A query lies outside the grid's interpolation bounds.
    #[error("{name} {value} is out of TEC grid bounds")]
    OutOfBounds {
        /// Axis label passed by `TecGrid::interpolate_vtec` for the query that exceeded an axis endpoint.
        name: &'static str,
        /// Query coordinate passed by `TecGrid::interpolate_vtec` that exceeded the named axis endpoint.
        value: f64,
    },
}

#[cfg(test)]
mod error_display_tests {
    use super::TecGridError;

    #[test]
    fn tec_grid_error_display_preserves_parser_messages() {
        let cases = [
            (
                TecGridError::AxesTooShort,
                "TEC grid axes must each contain at least two entries",
            ),
            (
                TecGridError::AxesNotIncreasing,
                "TEC grid axes must be strictly increasing",
            ),
            (
                TecGridError::DimensionsOverflow,
                "TEC grid dimensions overflow",
            ),
            (
                TecGridError::ValueCountMismatch {
                    actual: 3,
                    expected: 8,
                },
                "TEC grid has 3 values but expected 8",
            ),
            (
                TecGridError::InvalidField {
                    field: "frequency_hz",
                    reason: "must be positive",
                },
                "frequency_hz must be positive",
            ),
            (
                TecGridError::OutOfBounds {
                    name: "latitude",
                    value: 95.0,
                },
                "latitude 95 is out of TEC grid bounds",
            ),
        ];

        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// [`TecGrid::vtec_at_pierce_point`] converts `unix_nanos` to the temporal
/// interpolation coordinate. `day_of_year` is retained for callers but is not
/// read by this regular-grid implementation.
pub struct TecGridEpoch {
    /// Unix-epoch timestamp in nanoseconds, matching the grid's epoch axis.
    pub unix_nanos: i64,
    /// Day-of-year companion carried with the timestamp but unused by grid interpolation.
    pub day_of_year: u16,
}

impl TecGridEpoch {
    /// Builds an epoch by copying the supplied timestamp and day-of-year value.
    ///
    /// Neither value is validated here; validation of the timestamp occurs
    /// when a grid query converts it to its floating-point axis coordinate.
    pub fn new(unix_nanos: i64, day_of_year: u16) -> Self {
        Self {
            unix_nanos,
            day_of_year,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// [`Self::shell_radius_m`] adds the shell height to the Earth radius, and
/// [`tec_xyz`] uses the resulting radius for the pierce-point intersection and
/// the slant-to-vertical mapping.
pub struct TecGridShellGeometry {
    /// Radius in meters used by the shell intersection and obliquity numerator;
    /// evaluation requires it to be finite and positive.
    pub earth_radius_m: f64,
    /// Height in meters added to the Earth radius for the shell intersection;
    /// evaluation requires it to be finite and nonnegative.
    pub shell_height_m: f64,
}

impl TecGridShellGeometry {
    /// Constructs shell geometry without validating its two distances.
    ///
    /// The distances are validated when the geometry is passed to [`tec_xyz`]
    /// or [`iono_delay_xyz`].
    pub const fn new(earth_radius_m: f64, shell_height_m: f64) -> Self {
        Self {
            earth_radius_m,
            shell_height_m,
        }
    }

    /// Returns the default Earth radius and ionospheric shell height.
    ///
    /// [`Default::default`] delegates to this constructor.
    pub const fn default_shell() -> Self {
        Self {
            earth_radius_m: EARTH_RADIUS_M,
            shell_height_m: IONOSPHERE_HEIGHT_M,
        }
    }

    /// Returns the spherical shell radius as `earth_radius_m + shell_height_m`.
    pub fn shell_radius_m(self) -> f64 {
        self.earth_radius_m + self.shell_height_m
    }
}

impl Default for TecGridShellGeometry {
    fn default() -> Self {
        Self::default_shell()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
/// [`tec_xyz`] consumes the epoch, elevation floor, fallback altitude, and
/// shell geometry; [`iono_delay_xyz`] additionally uses the carrier frequency.
pub struct TecGridEvalOptions {
    /// Epoch passed to [`TecGrid::vtec_at_pierce_point`].
    pub epoch: TecGridEpoch,
    /// Minimum elevation used by the thin-shell obliquity mapping, in radians.
    pub min_elevation_rad: f64,
    /// Fallback pierce-point altitude in meters when the coordinate callback returns any NaN.
    pub nan_pierce_point_height_m: f64,
    /// Carrier frequency used to convert slant TEC to group delay, in hertz.
    pub frequency_hz: f64,
    /// Earth radius and shell height used by the pierce-point and obliquity calculations.
    pub shell_geometry: TecGridShellGeometry,
}

impl TecGridEvalOptions {
    /// Creates options for the canonical GPS L1 frequency.
    ///
    /// The result uses a 5-degree minimum elevation, the default 450,000-meter
    /// fallback height, and [`TecGridShellGeometry::default`].
    pub fn l1(epoch: TecGridEpoch) -> Self {
        // invariant: the built-in GNSS frequency table always defines GPS L1.
        #[allow(clippy::expect_used)]
        let frequency_hz = frequencies::frequency_hz(GnssSystem::Gps, CarrierBand::L1)
            .expect("canonical GPS L1 carrier exists");
        Self {
            epoch,
            min_elevation_rad: 5.0 * DEG_TO_RAD,
            nan_pierce_point_height_m: IONOSPHERE_HEIGHT_M,
            frequency_hz,
            shell_geometry: TecGridShellGeometry::default(),
        }
    }

    /// Returns a copy using `shell_geometry` and its height as the NaN fallback altitude.
    pub fn with_shell_geometry(mut self, shell_geometry: TecGridShellGeometry) -> Self {
        self.nan_pierce_point_height_m = shell_geometry.shell_height_m;
        self.shell_geometry = shell_geometry;
        self
    }
}

#[derive(Clone, Debug)]
/// [`TecGrid::new`] stores finite TECU values on strictly increasing epoch,
/// latitude, and longitude axes in epoch-latitude-longitude order. Queries
/// interpolate the eight corners of the surrounding cell.
pub struct TecGrid {
    epochs_ns: Vec<f64>,
    latitudes_deg: Vec<f64>,
    longitudes_deg: Vec<f64>,
    values: Vec<f64>,
}

impl TecGrid {
    /// Builds a grid from epoch, latitude, and longitude axes and flat cell values.
    ///
    /// Each axis must contain at least two strictly increasing entries. The
    /// value count must equal the checked product of the axis lengths, and all
    /// values must be finite; otherwise the returned error describes the failed
    /// invariant.
    pub fn new(
        epochs_ns: Vec<f64>,
        latitudes_deg: Vec<f64>,
        longitudes_deg: Vec<f64>,
        values: Vec<f64>,
    ) -> Result<Self, TecGridError> {
        if epochs_ns.len() < 2 || latitudes_deg.len() < 2 || longitudes_deg.len() < 2 {
            return Err(TecGridError::AxesTooShort);
        }
        if !strictly_increasing(&epochs_ns)
            || !strictly_increasing(&latitudes_deg)
            || !strictly_increasing(&longitudes_deg)
        {
            return Err(TecGridError::AxesNotIncreasing);
        }
        let expected = epochs_ns
            .len()
            .checked_mul(latitudes_deg.len())
            .and_then(|v| v.checked_mul(longitudes_deg.len()))
            .ok_or(TecGridError::DimensionsOverflow)?;
        if values.len() != expected {
            return Err(TecGridError::ValueCountMismatch {
                actual: values.len(),
                expected,
            });
        }
        validate::finite_slice(&values, "TEC grid values").map_err(field_error_string)?;
        Ok(Self {
            epochs_ns,
            latitudes_deg,
            longitudes_deg,
            values,
        })
    }

    /// Returns VTEC interpolated at a pierce-point longitude and latitude.
    ///
    /// Latitude values outside `[-87.5, 87.5]` are clamped to that interval.
    /// The epoch's Unix-nanosecond timestamp is converted to the grid's
    /// floating-point epoch coordinate, and the effective epoch, latitude, and
    /// longitude query values must be finite and within their respective axes.
    pub fn vtec_at_pierce_point(
        &self,
        epoch: TecGridEpoch,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> Result<f64, TecGridError> {
        let latitude_deg = if latitude_deg.abs() > 87.5 {
            clamp(latitude_deg, -87.5, 87.5)
        } else {
            latitude_deg
        };
        self.interpolate_vtec(epoch.unix_nanos as f64, latitude_deg, longitude_deg)
    }

    pub(crate) fn interpolate_vtec(
        &self,
        epoch_ns: f64,
        latitude_deg: f64,
        longitude_deg: f64,
    ) -> Result<f64, TecGridError> {
        let epoch_ns = finite_query_value(epoch_ns, "timestamp")?;
        let latitude_deg = finite_query_value(latitude_deg, "latitude")?;
        let longitude_deg = finite_query_value(longitude_deg, "longitude")?;
        let (epoch_i, epoch_y) = interval(&self.epochs_ns, epoch_ns, "timestamp")?;
        let (lat_i, lat_y) = interval(&self.latitudes_deg, latitude_deg, "latitude")?;
        let (lon_i, lon_y) = interval(&self.longitudes_deg, longitude_deg, "longitude")?;

        let indices = [epoch_i, lat_i, lon_i];
        let norm_distances = [epoch_y, lat_y, lon_y];
        let shift_norm_distances = [
            1.0 - norm_distances[0],
            1.0 - norm_distances[1],
            1.0 - norm_distances[2],
        ];
        let shift_indices = [indices[0] + 1, indices[1] + 1, indices[2] + 1];

        let mut value = 0.0;
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    let i0 = if a == 0 { indices[0] } else { shift_indices[0] };
                    let i1 = if b == 0 { indices[1] } else { shift_indices[1] };
                    let i2 = if c == 0 { indices[2] } else { shift_indices[2] };
                    let w0 = if a == 0 {
                        shift_norm_distances[0]
                    } else {
                        norm_distances[0]
                    };
                    let w1 = if b == 0 {
                        shift_norm_distances[1]
                    } else {
                        norm_distances[1]
                    };
                    let w2 = if c == 0 {
                        shift_norm_distances[2]
                    } else {
                        norm_distances[2]
                    };

                    let mut weight = 1.0;
                    weight *= w0;
                    weight *= w1;
                    weight *= w2;
                    let term = self.value_at(i0, i1, i2) * weight;
                    value += term;
                }
            }
        }
        Ok(value)
    }

    fn value_at(&self, epoch_i: usize, lat_i: usize, lon_i: usize) -> f64 {
        let n_lat = self.latitudes_deg.len();
        let n_lon = self.longitudes_deg.len();
        self.values[(epoch_i * n_lat + lat_i) * n_lon + lon_i]
    }
}

/// Computes the ionospheric group delay for an ECEF satellite and receiver pair.
///
/// The callback receives the ECEF pierce point in meters and returns
/// `[longitude_deg, latitude_deg, altitude]`. This function validates the
/// carrier frequency, obtains slant TEC from [`tec_xyz`], and applies
/// `IONOSPHERE_CONSTANT * stec / frequency_hz^2`, returning the finite result in
/// meters. The altitude component is used only for the NaN check.
pub fn iono_delay_xyz<F>(
    grid: &TecGrid,
    options: TecGridEvalOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    ecef_to_lla: F,
) -> Result<f64, TecGridError>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    validate_frequency(options.frequency_hz)?;

    let (_vtec, stec) = tec_xyz(grid, options, sat_xyz, receiver_xyz, ecef_to_lla)?;
    let delay_m = IONOSPHERE_CONSTANT * stec / (options.frequency_hz * options.frequency_hz);
    validate::finite(delay_m, "ionosphere_delay_m")
        .map_err(field_error_string)
        .map(|_| delay_m)
}

/// Computes vertical and slant TEC for an ECEF satellite and receiver pair.
///
/// The callback receives the ECEF pierce point in meters and returns
/// `[longitude_deg, latitude_deg, altitude]`; if any returned component is
/// NaN, the receiver longitude/latitude and
/// `nan_pierce_point_height_m` are used instead. The result is
/// `(vtec_tecu, stec_tecu)`, with slant TEC obtained from the configured shell
/// geometry and the elevation after applying `min_elevation_rad`.
pub fn tec_xyz<F>(
    grid: &TecGrid,
    options: TecGridEvalOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    ecef_to_lla: F,
) -> Result<(f64, f64), TecGridError>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    let shell_radius_m = validate_tec_geometry_inputs(options, sat_xyz, receiver_xyz)?;
    let (_pp_xyz, pp_lonlatalt, mut elevation_rad) =
        pierce_point_with_shell_radius(sat_xyz, receiver_xyz, shell_radius_m, &ecef_to_lla);
    if elevation_rad < options.min_elevation_rad {
        elevation_rad = options.min_elevation_rad;
    }
    validate::finite(elevation_rad, "elevation_rad").map_err(field_error_string)?;

    let pp_lonlatalt = if pp_lonlatalt.iter().any(|v| v.is_nan()) {
        let receiver_lonlatalt = ecef_to_lla(receiver_xyz);
        [
            receiver_lonlatalt[0],
            receiver_lonlatalt[1],
            options.nan_pierce_point_height_m,
        ]
    } else {
        pp_lonlatalt
    };

    let vtec = grid.vtec_at_pierce_point(options.epoch, pp_lonlatalt[0], pp_lonlatalt[1])?;
    validate::finite(vtec, "vtec").map_err(field_error_string)?;
    let obliquity_arg =
        options.shell_geometry.earth_radius_m * libm::cos(elevation_rad) / shell_radius_m;
    validate::finite(obliquity_arg, "obliquity_arg").map_err(field_error_string)?;
    let mapping_denominator = 1.0 - obliquity_arg * obliquity_arg;
    validate::finite_positive(mapping_denominator, "TEC mapping denominator")
        .map_err(field_error_string)?;
    let stec = vtec / mapping_denominator.sqrt();
    validate::finite(stec, "stec").map_err(field_error_string)?;
    Ok((vtec, stec))
}

pub fn pierce_point_with_shell_radius<F>(
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    shell_radius_m: f64,
    ecef_to_lla: F,
) -> ([f64; 3], [f64; 3], f64)
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    let receiver_sat_vector = [
        sat_xyz[0] - receiver_xyz[0],
        sat_xyz[1] - receiver_xyz[1],
        sat_xyz[2] - receiver_xyz[2],
    ];

    let receiver_up = unit_vector(receiver_xyz);
    let sat_unit = unit_vector(&receiver_sat_vector);
    let elevation_rad = libm::asin(dot_three_fused(&sat_unit, &receiver_up));

    let a = 1.0;
    let b = 2.0 * dot_three_fused(receiver_xyz, &sat_unit);
    let c = dot_three_fused(receiver_xyz, receiver_xyz) - shell_radius_m * shell_radius_m;
    let t = (-b + (b * b - 4.0 * a * c).sqrt()) / (2.0 * a);

    let pp_xyz = [
        receiver_xyz[0] + t * sat_unit[0],
        receiver_xyz[1] + t * sat_unit[1],
        receiver_xyz[2] + t * sat_unit[2],
    ];
    let pp_lonlatalt = ecef_to_lla(&pp_xyz);
    (pp_xyz, pp_lonlatalt, elevation_rad)
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

fn strictly_increasing(values: &[f64]) -> bool {
    values.windows(2).all(|w| w[1] > w[0])
}

fn finite_query_value(value: f64, name: &'static str) -> Result<f64, TecGridError> {
    validate::finite(value, name).map_err(field_error_string)
}

fn field_error_string(error: validate::FieldError) -> TecGridError {
    TecGridError::InvalidField {
        field: error.field(),
        reason: error.reason(),
    }
}

fn validate_frequency(frequency_hz: f64) -> Result<(), TecGridError> {
    validate::finite_positive(frequency_hz, "frequency_hz")
        .map(|_| ())
        .map_err(field_error_string)
}

fn validate_tec_geometry_inputs(
    options: TecGridEvalOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
) -> Result<f64, TecGridError> {
    validate::finite_vec3(*sat_xyz, "satellite_xyz").map_err(field_error_string)?;
    validate::finite_vec3(*receiver_xyz, "receiver_xyz").map_err(field_error_string)?;
    validate::finite(options.min_elevation_rad, "min_elevation_rad").map_err(field_error_string)?;
    validate::finite(
        options.nan_pierce_point_height_m,
        "nan_pierce_point_height_m",
    )
    .map_err(field_error_string)?;
    validate::finite_positive(options.shell_geometry.earth_radius_m, "earth_radius_m")
        .map_err(field_error_string)?;
    validate::finite_nonneg(options.shell_geometry.shell_height_m, "shell_height_m")
        .map_err(field_error_string)?;

    let shell_radius_m = options.shell_geometry.shell_radius_m();
    validate::finite_positive(shell_radius_m, "shell_radius_m").map_err(field_error_string)?;

    let receiver_radius_m = dot_three_fused(receiver_xyz, receiver_xyz).sqrt();
    validate::finite_positive(receiver_radius_m, "receiver radius_m")
        .map_err(field_error_string)?;

    let line_of_sight_m = [
        sat_xyz[0] - receiver_xyz[0],
        sat_xyz[1] - receiver_xyz[1],
        sat_xyz[2] - receiver_xyz[2],
    ];
    validate::finite_vec3(line_of_sight_m, "line of sight_m").map_err(field_error_string)?;
    let line_of_sight_norm_m = dot_three_fused(&line_of_sight_m, &line_of_sight_m).sqrt();
    validate::finite_positive(line_of_sight_norm_m, "line of sight_m")
        .map_err(field_error_string)?;

    Ok(shell_radius_m)
}

fn interval(axis: &[f64], x: f64, name: &'static str) -> Result<(usize, f64), TecGridError> {
    if x < axis[0] || x > axis[axis.len() - 1] {
        return Err(TecGridError::OutOfBounds { name, value: x });
    }
    let upper = axis.partition_point(|v| *v <= x);
    let mut lower = upper.saturating_sub(1);
    if lower >= axis.len() - 1 {
        lower = axis.len() - 2;
    }
    let y = (x - axis[lower]) / (axis[lower + 1] - axis[lower]);
    Ok((lower, y))
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;

    fn small_grid() -> TecGrid {
        TecGrid::new(
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![20.0, 30.0],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .expect("small TEC grid")
    }

    #[test]
    fn interpolate_vtec_rejects_non_finite_query_coordinates() {
        let grid = small_grid();
        let cases = [
            (f64::NAN, 5.0, 25.0, "timestamp"),
            (f64::INFINITY, 5.0, 25.0, "timestamp"),
            (5.0, f64::NAN, 25.0, "latitude"),
            (5.0, 5.0, f64::NAN, "longitude"),
        ];

        for (epoch_ns, latitude_deg, longitude_deg, field) in cases {
            let error = grid
                .interpolate_vtec(epoch_ns, latitude_deg, longitude_deg)
                .expect_err("non-finite TEC coordinate must be rejected");
            assert!(error.to_string().contains(field), "{error}");
            assert!(error.to_string().contains("not finite"), "{error}");
        }
    }

    #[test]
    fn interpolate_vtec_valid_query_still_interpolates() {
        let grid = small_grid();

        assert_eq!(
            grid.interpolate_vtec(0.0, 0.0, 20.0)
                .expect("lower corner")
                .to_bits(),
            1.0f64.to_bits()
        );
        assert_eq!(
            grid.interpolate_vtec(5.0, 5.0, 25.0)
                .expect("center point")
                .to_bits(),
            4.5f64.to_bits()
        );
    }

    #[test]
    fn tec_grid_rejects_nonfinite_values() {
        let error = TecGrid::new(
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![20.0, 30.0],
            vec![1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .expect_err("nonfinite TEC grid cells must be rejected");

        assert!(error.to_string().contains("TEC grid values"), "{error}");
        assert!(error.to_string().contains("not finite"), "{error}");
    }

    #[test]
    fn tec_xyz_rejects_degenerate_geometry_without_nonfinite_success() {
        fn passthrough_lla(xyz: &[f64; 3]) -> [f64; 3] {
            [xyz[0], xyz[1], xyz[2]]
        }

        let grid = TecGrid::new(
            vec![0.0, 1.0],
            vec![-10.0, 10.0],
            vec![0.0, 20.0],
            vec![0.0; 8],
        )
        .expect("regular TEC grid");
        let mut options = TecGridEvalOptions::l1(TecGridEpoch::new(0, 0));
        options.min_elevation_rad = 0.0;
        options.nan_pierce_point_height_m = 0.0;

        let error = tec_xyz(
            &grid,
            options,
            &[0.0, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            passthrough_lla,
        )
        .expect_err("zero receiver and satellite vectors must be rejected");

        assert!(error.to_string().contains("receiver radius_m"), "{error}");
        assert!(error.to_string().contains("not positive"), "{error}");
    }

    #[test]
    fn iono_delay_xyz_rejects_invalid_frequency() {
        fn passthrough_lla(_: &[f64; 3]) -> [f64; 3] {
            [25.0, 5.0, IONOSPHERE_HEIGHT_M]
        }

        let grid = small_grid();
        let sat_xyz = [2.0, 0.0, 0.0];
        let receiver_xyz = [1.0, 0.0, 0.0];
        for (frequency_hz, reason) in [(0.0, "not positive"), (f64::NAN, "not finite")] {
            let mut options = TecGridEvalOptions::l1(TecGridEpoch::new(0, 1));
            options.frequency_hz = frequency_hz;

            let error = iono_delay_xyz(&grid, options, &sat_xyz, &receiver_xyz, passthrough_lla)
                .expect_err("invalid TEC-grid frequency must be rejected");
            assert!(error.to_string().contains("frequency_hz"), "{error}");
            assert!(error.to_string().contains(reason), "{error}");
        }
    }
}
