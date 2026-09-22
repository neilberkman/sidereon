//! Regular-grid TEC ionosphere delay variant.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

use crate::astro::math::vec3::{
    dot3_fused_z_yx_ref as dot_three_fused, unit3_ref_unchecked as unit_vector,
};

use super::{IonexMissingNodePolicy, IonexMissingNodes, IonexNodeGap};
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
    /// A query weights grid nodes that hold no value.
    ///
    /// Each [`IonexMissingNodes`] names the epoch by `map_number`, counting
    /// from 1, and the
    /// latitude and longitude axes by the cell's lower-index node.
    #[error("TEC grid nodes not available: {0}")]
    NodesNotAvailable(IonexNodeGap),
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
#[non_exhaustive]
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
    /// Build evaluation options for an explicit carrier frequency.
    ///
    /// The minimum elevation, NaN fallback height, and shell geometry use the
    /// engine defaults; assign those fields when a different shell is needed.
    #[must_use]
    pub fn new(epoch: TecGridEpoch, frequency_hz: f64) -> Self {
        Self {
            epoch,
            min_elevation_rad: 5.0 * DEG_TO_RAD,
            nan_pierce_point_height_m: IONOSPHERE_HEIGHT_M,
            frequency_hz,
            shell_geometry: TecGridShellGeometry::default(),
        }
    }

    /// Creates options for the canonical GPS L1 frequency.
    ///
    /// The result uses a 5-degree minimum elevation, the default 450,000-meter
    /// fallback height, and [`TecGridShellGeometry::default`].
    pub fn l1(epoch: TecGridEpoch) -> Self {
        // invariant: the built-in GNSS frequency table always defines GPS L1.
        #[allow(clippy::expect_used)]
        let frequency_hz = frequencies::frequency_hz(GnssSystem::Gps, CarrierBand::L1)
            .expect("canonical GPS L1 carrier exists");
        Self::new(epoch, frequency_hz)
    }

    /// Returns a copy using `shell_geometry` and its height as the NaN fallback altitude.
    pub fn with_shell_geometry(mut self, shell_geometry: TecGridShellGeometry) -> Self {
        self.nan_pierce_point_height_m = shell_geometry.shell_height_m;
        self.shell_geometry = shell_geometry;
        self
    }
}

/// A regular-grid TEC value, with the nodes a renormalizing fallback
/// interpolated around.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TecGridEvaluation<T> {
    /// The evaluated value.
    pub value: T,
    /// The weighted nodes that hold no value, where
    /// [`IonexMissingNodePolicy::Renormalize`] interpolated around them. A value
    /// with this set is degraded.
    pub degraded: Option<IonexNodeGap>,
}

#[derive(Clone, Debug)]
/// [`TecGrid::new`] stores TECU values on strictly increasing epoch, latitude,
/// and longitude axes in epoch-latitude-longitude order, `None` marking a node
/// without a value. Queries interpolate the eight corners of the surrounding
/// cell: bilinearly within each of the two bracketing epochs, then linearly in
/// time.
///
/// A corner is weighted when its weight is nonzero. When a weighted corner holds
/// no value, the strict query functions return
/// [`TecGridError::NodesNotAvailable`], and the `_with_policy` functions under
/// [`IonexMissingNodePolicy::Renormalize`] interpolate from the weighted corners
/// that hold values, the bilinear weights of each epoch and then the temporal
/// weights renormalized to sum to one, and mark the result degraded. With no
/// weighted corner holding a value there is no value under either policy.
pub struct TecGrid {
    epochs_ns: Vec<f64>,
    latitudes_deg: Vec<f64>,
    longitudes_deg: Vec<f64>,
    values: Vec<Option<f64>>,
}

impl TecGrid {
    /// Builds a grid from epoch, latitude, and longitude axes and flat cell values.
    ///
    /// Each axis must contain at least two strictly increasing entries. The
    /// value count must equal the checked product of the axis lengths, and every
    /// value present must be finite; otherwise the returned error describes the
    /// failed invariant. `None` marks a node without a value.
    pub fn new(
        epochs_ns: Vec<f64>,
        latitudes_deg: Vec<f64>,
        longitudes_deg: Vec<f64>,
        values: Vec<Option<f64>>,
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
        for value in values.iter().flatten() {
            validate::finite(*value, "TEC grid values").map_err(field_error_string)?;
        }
        Ok(Self {
            epochs_ns,
            latitudes_deg,
            longitudes_deg,
            values,
        })
    }

    /// Returns the grid epoch coordinates as an immutable slice.
    ///
    /// Epoch coordinates are `f64` Unix nanoseconds, and adjacent nanoseconds
    /// may not be distinguishable at large magnitudes.
    #[must_use]
    pub fn epochs_ns(&self) -> &[f64] {
        &self.epochs_ns
    }

    /// Returns the grid latitude coordinates as an immutable slice, with angles in degrees.
    #[must_use]
    pub fn latitudes_deg(&self) -> &[f64] {
        &self.latitudes_deg
    }

    /// Returns the grid longitude coordinates as an immutable slice, with angles in degrees.
    #[must_use]
    pub fn longitudes_deg(&self) -> &[f64] {
        &self.longitudes_deg
    }

    /// Returns the flat grid TEC values in TECU as an immutable slice in epoch-latitude-longitude order.
    ///
    /// Grid values are stored in units of TECU in flat epoch-latitude-longitude
    /// order with longitude varying fastest, where `None` indicates a missing
    /// node value and `Some(0.0)` indicates a valid zero-valued node.
    #[must_use]
    pub fn values(&self) -> &[Option<f64>] {
        &self.values
    }

    /// Returns VTEC interpolated at a pierce-point longitude and latitude.
    ///
    /// Latitude values outside `[-87.5, 87.5]` are clamped to that interval.
    /// The epoch's Unix-nanosecond timestamp is converted to the grid's
    /// floating-point epoch coordinate, and the effective epoch, latitude, and
    /// longitude query values must be finite and within their respective axes.
    /// A weighted node without a value returns
    /// [`TecGridError::NodesNotAvailable`].
    pub fn vtec_at_pierce_point(
        &self,
        epoch: TecGridEpoch,
        longitude_deg: f64,
        latitude_deg: f64,
    ) -> Result<f64, TecGridError> {
        self.vtec_at_pierce_point_with_policy(
            epoch,
            longitude_deg,
            latitude_deg,
            IonexMissingNodePolicy::Strict,
        )
        .map(|evaluation| evaluation.value)
    }

    /// [`TecGrid::vtec_at_pierce_point`] with an explicit policy for weighted
    /// nodes that hold no value.
    pub fn vtec_at_pierce_point_with_policy(
        &self,
        epoch: TecGridEpoch,
        longitude_deg: f64,
        latitude_deg: f64,
        policy: IonexMissingNodePolicy,
    ) -> Result<TecGridEvaluation<f64>, TecGridError> {
        let latitude_deg = if latitude_deg.abs() > 87.5 {
            clamp(latitude_deg, -87.5, 87.5)
        } else {
            latitude_deg
        };
        self.interpolate_vtec_with_policy(
            epoch.unix_nanos as f64,
            latitude_deg,
            longitude_deg,
            policy,
        )
    }

    #[cfg(test)]
    pub(crate) fn interpolate_vtec(
        &self,
        epoch_ns: f64,
        latitude_deg: f64,
        longitude_deg: f64,
    ) -> Result<f64, TecGridError> {
        self.interpolate_vtec_with_policy(
            epoch_ns,
            latitude_deg,
            longitude_deg,
            IonexMissingNodePolicy::Strict,
        )
        .map(|evaluation| evaluation.value)
    }

    pub(crate) fn interpolate_vtec_with_policy(
        &self,
        epoch_ns: f64,
        latitude_deg: f64,
        longitude_deg: f64,
        policy: IonexMissingNodePolicy,
    ) -> Result<TecGridEvaluation<f64>, TecGridError> {
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
        let weight_of = |axis: usize, upper: usize| {
            if upper == 0 {
                shift_norm_distances[axis]
            } else {
                norm_distances[axis]
            }
        };

        // The weighted corners of each bracketing epoch that hold no value.
        let mut missing = [[false; 4]; 2];
        for a in 0..2 {
            for b in 0..2 {
                for c in 0..2 {
                    let weight = weight_of(0, a) * weight_of(1, b) * weight_of(2, c);
                    let node = self.value_at(indices[0] + a, indices[1] + b, indices[2] + c);
                    missing[a][2 * b + c] = weight != 0.0 && node.is_none();
                }
            }
        }
        let gap_on = |a: usize| {
            missing[a].contains(&true).then_some(IonexMissingNodes {
                // The epoch axis is indexed from 0; a map is named from 1.
                map_number: indices[0] + a + 1,
                lat_index: indices[1],
                lon_index: indices[2],
                // A regular grid names both edges of its longitude range, so it
                // has no cell closing the circle and no column that wraps.
                lon_index_next: indices[2] + 1,
                missing: missing[a],
            })
        };
        let gap = IonexNodeGap {
            earlier: gap_on(0),
            later: gap_on(1),
        };
        if gap.earlier.is_some() || gap.later.is_some() {
            return match policy {
                IonexMissingNodePolicy::Strict => Err(TecGridError::NodesNotAvailable(gap)),
                IonexMissingNodePolicy::Renormalize => self
                    .renormalized_vtec(indices, &weight_of)
                    .map(|value| TecGridEvaluation {
                        value,
                        degraded: Some(gap),
                    })
                    .ok_or(TecGridError::NodesNotAvailable(gap)),
            };
        }

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
                    // Every corner with weight holds a value here; a corner
                    // without weight contributes nothing whatever it holds.
                    let term = self.value_at(i0, i1, i2).unwrap_or(0.0) * weight;
                    value += term;
                }
            }
        }
        Ok(TecGridEvaluation {
            value,
            degraded: None,
        })
    }

    /// Interpolate from the weighted corners that hold values: within each
    /// weighted epoch, bilinear weights renormalized over its corners with values,
    /// then temporal weights renormalized over the epochs with a value. `None`
    /// when no weighted corner holds a value.
    fn renormalized_vtec(
        &self,
        indices: [usize; 3],
        weight_of: &impl Fn(usize, usize) -> f64,
    ) -> Option<f64> {
        let mut time_weight = 0.0;
        let mut time_sum = 0.0;
        for a in 0..2 {
            let epoch_weight = weight_of(0, a);
            if epoch_weight == 0.0 {
                continue;
            }
            let mut cell_weight = 0.0;
            let mut cell_sum = 0.0;
            for b in 0..2 {
                for c in 0..2 {
                    let weight = weight_of(1, b) * weight_of(2, c);
                    let node = self.value_at(indices[0] + a, indices[1] + b, indices[2] + c);
                    if let (true, Some(value)) = (weight != 0.0, node) {
                        cell_weight += weight;
                        cell_sum += weight * value;
                    }
                }
            }
            if cell_weight != 0.0 {
                time_weight += epoch_weight;
                time_sum += epoch_weight * (cell_sum / cell_weight);
            }
        }
        (time_weight != 0.0).then(|| time_sum / time_weight)
    }

    fn value_at(&self, epoch_i: usize, lat_i: usize, lon_i: usize) -> Option<f64> {
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
///
/// [`TecGridDelayXyzConversion`] performs the same evaluation in steps for a
/// caller that cannot supply the conversion as a Rust closure.
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
    iono_delay_xyz_with_policy(
        grid,
        options,
        sat_xyz,
        receiver_xyz,
        ecef_to_lla,
        IonexMissingNodePolicy::Strict,
    )
    .map(|evaluation| evaluation.value)
}

/// [`iono_delay_xyz`] with an explicit policy for weighted grid nodes that hold
/// no value.
pub fn iono_delay_xyz_with_policy<F>(
    grid: &TecGrid,
    options: TecGridEvalOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    ecef_to_lla: F,
    policy: IonexMissingNodePolicy,
) -> Result<TecGridEvaluation<f64>, TecGridError>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    let frequency_hz = validate_frequency(options.frequency_hz)?;
    let stage = PiercePointStage::prepare(options, sat_xyz, receiver_xyz, policy)?;
    let tec = stage.drive(grid, &ecef_to_lla)?;
    group_delay_from_tec(tec, frequency_hz)
}

/// Computes vertical and slant TEC for an ECEF satellite and receiver pair.
///
/// The callback receives the ECEF pierce point in meters and returns
/// `[longitude_deg, latitude_deg, altitude]`; if any returned component is
/// NaN, the receiver longitude/latitude and
/// `nan_pierce_point_height_m` are used instead. The result is
/// `(vtec_tecu, stec_tecu)`, with slant TEC obtained from the configured shell
/// geometry and the elevation after applying `min_elevation_rad`.
///
/// [`TecGridXyzConversion`] performs the same evaluation in steps for a caller
/// that cannot supply the conversion as a Rust closure.
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
    tec_xyz_with_policy(
        grid,
        options,
        sat_xyz,
        receiver_xyz,
        ecef_to_lla,
        IonexMissingNodePolicy::Strict,
    )
    .map(|evaluation| evaluation.value)
}

/// [`tec_xyz`] with an explicit policy for weighted grid nodes that hold no
/// value.
pub fn tec_xyz_with_policy<F>(
    grid: &TecGrid,
    options: TecGridEvalOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    ecef_to_lla: F,
    policy: IonexMissingNodePolicy,
) -> Result<TecGridEvaluation<(f64, f64)>, TecGridError>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    PiercePointStage::prepare(options, sat_xyz, receiver_xyz, policy)?.drive(grid, &ecef_to_lla)
}

/// The ECEF position a [`TecGridXyzConversion`] or
/// [`TecGridDelayXyzConversion`] asks the caller to convert.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TecGridXyzTarget {
    /// The ionospheric pierce point, the first conversion of every evaluation.
    ///
    /// Its coordinates are whatever the shell intersection produced. A line of
    /// sight that misses the shell gives NaN components, and they are still
    /// requested, as [`tec_xyz`] passes them to its callback.
    PiercePoint,
    /// The receiver position, requested only after the pierce-point answer held
    /// a NaN component. Its coordinates are the receiver ECEF position exactly
    /// as supplied.
    Receiver,
}

/// A [`tec_xyz_with_policy`] evaluation waiting for one ECEF to geodetic
/// coordinate conversion.
///
/// The conversion is owned and holds no reference to a grid: it keeps the
/// validated options, the missing-node policy, the receiver position, the shell
/// radius and the geometry computed so far. [`Self::prepare`] validates the
/// inputs in the order [`tec_xyz_with_policy`] does and returns a conversion
/// for the pierce point. The caller converts [`Self::xyz`], ECEF meters, to
/// `[longitude_deg, latitude_deg, altitude]` and passes that answer with the
/// grid to [`Self::resume`], which consumes the conversion and returns either
/// the next conversion or the finished evaluation.
///
/// An evaluation requests at most two conversions: the pierce point, and the
/// receiver only when the pierce-point answer held a NaN component. Resuming a
/// receiver conversion always finishes, with a value or an error. Every step
/// runs the same code as [`tec_xyz_with_policy`], so a caller answering each
/// request as that function's callback would get the same result, error and
/// sequence of requested coordinates.
///
/// The type is not `Clone`: each conversion is answered once, and a caller that
/// has to answer again starts a new evaluation with [`Self::prepare`]. The grid
/// is checked only when a step reads it, so a timestamp outside the grid, a
/// pierce point outside it or a missing node is reported by [`Self::resume`]
/// after the conversions that precede that check, not by [`Self::prepare`].
/// The evaluated grid is the one passed to the step that finishes; keeping the
/// same grid across steps is up to the caller.
#[derive(Debug)]
#[must_use = "a conversion does nothing until it is resumed"]
pub struct TecGridXyzConversion {
    stage: ConversionStage,
}

/// The result of resuming a [`TecGridXyzConversion`].
#[derive(Debug)]
#[must_use = "a step either requests another conversion or holds the evaluation"]
pub enum TecGridXyzStep {
    /// The receiver conversion, requested after a NaN pierce-point answer.
    Convert(TecGridXyzConversion),
    /// The finished `(vtec_tecu, stec_tecu)` evaluation, as
    /// [`tec_xyz_with_policy`] returns it.
    Complete(TecGridEvaluation<(f64, f64)>),
}

impl TecGridXyzConversion {
    /// Validates a vertical and slant TEC evaluation and returns the conversion
    /// of its pierce point.
    ///
    /// The checks and their order are those of [`tec_xyz_with_policy`]: finite
    /// satellite and receiver positions, finite `min_elevation_rad` and
    /// `nan_pierce_point_height_m`, a finite positive Earth radius, a finite
    /// nonnegative shell height, a finite positive shell radius, a finite
    /// positive receiver radius and a finite nonzero line of sight. The carrier
    /// frequency is not read. Positions are ECEF meters. Nothing about the grid
    /// is checked here.
    pub fn prepare(
        options: TecGridEvalOptions,
        sat_xyz: &[f64; 3],
        receiver_xyz: &[f64; 3],
        policy: IonexMissingNodePolicy,
    ) -> Result<Self, TecGridError> {
        PiercePointStage::prepare(options, sat_xyz, receiver_xyz, policy).map(|stage| Self {
            stage: ConversionStage::PiercePoint(stage),
        })
    }

    /// Returns the ECEF position in meters to convert, bit for bit as
    /// [`tec_xyz`] passes it to its callback, including NaN components.
    #[must_use]
    pub fn xyz(&self) -> [f64; 3] {
        self.stage.xyz()
    }

    /// Returns which position [`Self::xyz`] is.
    #[must_use]
    pub fn target(&self) -> TecGridXyzTarget {
        self.stage.target()
    }

    /// Consumes the conversion with the answer for [`Self::xyz`],
    /// `[longitude_deg, latitude_deg, altitude]`, and continues the evaluation
    /// on `grid`.
    ///
    /// For the pierce point, the elevation is raised to `min_elevation_rad` and
    /// must be finite; then an answer with any NaN component, altitude included,
    /// returns [`TecGridXyzStep::Convert`] for the receiver. For the receiver,
    /// only the longitude and latitude are used, and the returned altitude is
    /// ignored. A finished evaluation interpolates `grid` at the selected
    /// longitude and latitude, where a latitude beyond ±87.5 degrees, infinite
    /// included, is clamped and every other query coordinate must be finite
    /// and inside the grid.
    pub fn resume(
        self,
        grid: &TecGrid,
        lonlatalt: [f64; 3],
    ) -> Result<TecGridXyzStep, TecGridError> {
        Ok(match self.stage.resume(grid, lonlatalt)? {
            StageOutcome::Receiver(stage) => TecGridXyzStep::Convert(Self {
                stage: ConversionStage::Receiver(stage),
            }),
            StageOutcome::Complete(evaluation) => TecGridXyzStep::Complete(evaluation),
        })
    }
}

/// An [`iono_delay_xyz_with_policy`] evaluation waiting for one ECEF to
/// geodetic coordinate conversion.
///
/// This is a [`TecGridXyzConversion`] that also holds the validated carrier
/// frequency and finishes with the group delay in meters. Its preparation,
/// requests, ownership and single-use semantics are those of
/// [`TecGridXyzConversion`].
#[derive(Debug)]
#[must_use = "a conversion does nothing until it is resumed"]
pub struct TecGridDelayXyzConversion {
    tec: TecGridXyzConversion,
    frequency_hz: f64,
}

/// The result of resuming a [`TecGridDelayXyzConversion`].
#[derive(Debug)]
#[must_use = "a step either requests another conversion or holds the evaluation"]
pub enum TecGridDelayXyzStep {
    /// The receiver conversion, requested after a NaN pierce-point answer.
    Convert(TecGridDelayXyzConversion),
    /// The finished group delay in meters, as [`iono_delay_xyz_with_policy`]
    /// returns it.
    Complete(TecGridEvaluation<f64>),
}

impl TecGridDelayXyzConversion {
    /// Validates an ionospheric group-delay evaluation and returns the
    /// conversion of its pierce point.
    ///
    /// The carrier frequency, in Hz, must be finite and positive, and is checked
    /// before the geometry, as [`iono_delay_xyz_with_policy`] checks it; the
    /// geometry is then checked as by [`TecGridXyzConversion::prepare`]. The
    /// square of the frequency is not checked here, as it is not in
    /// [`iono_delay_xyz_with_policy`]: a frequency whose square underflows to
    /// zero gives an infinite delay, refused when the evaluation finishes, and
    /// a frequency whose square overflows to infinity gives a delay of zero,
    /// which is returned.
    pub fn prepare(
        options: TecGridEvalOptions,
        sat_xyz: &[f64; 3],
        receiver_xyz: &[f64; 3],
        policy: IonexMissingNodePolicy,
    ) -> Result<Self, TecGridError> {
        let frequency_hz = validate_frequency(options.frequency_hz)?;
        let tec = TecGridXyzConversion::prepare(options, sat_xyz, receiver_xyz, policy)?;
        Ok(Self { tec, frequency_hz })
    }

    /// Returns the ECEF position in meters to convert, bit for bit as
    /// [`iono_delay_xyz`] passes it to its callback, including NaN components.
    #[must_use]
    pub fn xyz(&self) -> [f64; 3] {
        self.tec.xyz()
    }

    /// Returns which position [`Self::xyz`] is.
    #[must_use]
    pub fn target(&self) -> TecGridXyzTarget {
        self.tec.target()
    }

    /// Consumes the conversion with the answer for [`Self::xyz`],
    /// `[longitude_deg, latitude_deg, altitude]`, and continues the evaluation
    /// on `grid` as [`TecGridXyzConversion::resume`] does. A finished
    /// evaluation converts slant TEC to
    /// `IONOSPHERE_CONSTANT * stec / frequency_hz^2` meters, which must be
    /// finite.
    pub fn resume(
        self,
        grid: &TecGrid,
        lonlatalt: [f64; 3],
    ) -> Result<TecGridDelayXyzStep, TecGridError> {
        let frequency_hz = self.frequency_hz;
        match self.tec.resume(grid, lonlatalt)? {
            TecGridXyzStep::Convert(tec) => {
                Ok(TecGridDelayXyzStep::Convert(Self { tec, frequency_hz }))
            }
            TecGridXyzStep::Complete(tec) => {
                group_delay_from_tec(tec, frequency_hz).map(TecGridDelayXyzStep::Complete)
            }
        }
    }
}

/// The validated inputs every stage of one XYZ evaluation carries.
#[derive(Clone, Copy, Debug)]
struct XyzContext {
    options: TecGridEvalOptions,
    policy: IonexMissingNodePolicy,
    receiver_xyz: [f64; 3],
    shell_radius_m: f64,
}

/// Waiting for the pierce-point conversion, with the unclamped elevation.
#[derive(Debug)]
struct PiercePointStage {
    context: XyzContext,
    pp_xyz: [f64; 3],
    elevation_rad: f64,
}

/// Waiting for the receiver conversion, with the clamped, finite elevation.
#[derive(Debug)]
struct ReceiverStage {
    context: XyzContext,
    elevation_rad: f64,
}

#[derive(Debug)]
enum ConversionStage {
    PiercePoint(PiercePointStage),
    Receiver(ReceiverStage),
}

enum StageOutcome {
    Receiver(ReceiverStage),
    Complete(TecGridEvaluation<(f64, f64)>),
}

impl PiercePointStage {
    fn prepare(
        options: TecGridEvalOptions,
        sat_xyz: &[f64; 3],
        receiver_xyz: &[f64; 3],
        policy: IonexMissingNodePolicy,
    ) -> Result<Self, TecGridError> {
        let shell_radius_m = validate_tec_geometry_inputs(options, sat_xyz, receiver_xyz)?;
        let (pp_xyz, elevation_rad) = pierce_point_geometry(sat_xyz, receiver_xyz, shell_radius_m);
        Ok(Self {
            context: XyzContext {
                options,
                policy,
                receiver_xyz: *receiver_xyz,
                shell_radius_m,
            },
            pp_xyz,
            elevation_rad,
        })
    }

    fn resume(self, grid: &TecGrid, pp_lonlatalt: [f64; 3]) -> Result<StageOutcome, TecGridError> {
        let context = self.context;
        let mut elevation_rad = self.elevation_rad;
        if elevation_rad < context.options.min_elevation_rad {
            elevation_rad = context.options.min_elevation_rad;
        }
        validate::finite(elevation_rad, "elevation_rad").map_err(field_error_string)?;

        if pp_lonlatalt.iter().any(|v| v.is_nan()) {
            return Ok(StageOutcome::Receiver(ReceiverStage {
                context,
                elevation_rad,
            }));
        }
        context
            .complete(grid, pp_lonlatalt[0], pp_lonlatalt[1], elevation_rad)
            .map(StageOutcome::Complete)
    }

    /// Answers each request with `ecef_to_lla`, called once per request in
    /// request order.
    fn drive<F>(
        self,
        grid: &TecGrid,
        ecef_to_lla: &F,
    ) -> Result<TecGridEvaluation<(f64, f64)>, TecGridError>
    where
        F: Fn(&[f64; 3]) -> [f64; 3],
    {
        let pp_lonlatalt = ecef_to_lla(&self.pp_xyz);
        match self.resume(grid, pp_lonlatalt)? {
            StageOutcome::Complete(evaluation) => Ok(evaluation),
            StageOutcome::Receiver(stage) => {
                let receiver_lonlatalt = ecef_to_lla(&stage.context.receiver_xyz);
                stage.resume(grid, receiver_lonlatalt)
            }
        }
    }
}

impl ReceiverStage {
    /// Uses the receiver longitude and latitude. The returned altitude is not
    /// read, so a NaN there requests nothing further.
    fn resume(
        self,
        grid: &TecGrid,
        receiver_lonlatalt: [f64; 3],
    ) -> Result<TecGridEvaluation<(f64, f64)>, TecGridError> {
        self.context.complete(
            grid,
            receiver_lonlatalt[0],
            receiver_lonlatalt[1],
            self.elevation_rad,
        )
    }
}

impl ConversionStage {
    fn xyz(&self) -> [f64; 3] {
        match self {
            Self::PiercePoint(stage) => stage.pp_xyz,
            Self::Receiver(stage) => stage.context.receiver_xyz,
        }
    }

    fn target(&self) -> TecGridXyzTarget {
        match self {
            Self::PiercePoint(_) => TecGridXyzTarget::PiercePoint,
            Self::Receiver(_) => TecGridXyzTarget::Receiver,
        }
    }

    fn resume(self, grid: &TecGrid, lonlatalt: [f64; 3]) -> Result<StageOutcome, TecGridError> {
        match self {
            Self::PiercePoint(stage) => stage.resume(grid, lonlatalt),
            Self::Receiver(stage) => stage.resume(grid, lonlatalt).map(StageOutcome::Complete),
        }
    }
}

impl XyzContext {
    /// Interpolates VTEC at the selected longitude and latitude and maps it to
    /// slant TEC with the clamped elevation.
    fn complete(
        &self,
        grid: &TecGrid,
        longitude_deg: f64,
        latitude_deg: f64,
        elevation_rad: f64,
    ) -> Result<TecGridEvaluation<(f64, f64)>, TecGridError> {
        let options = self.options;
        let evaluation = grid.vtec_at_pierce_point_with_policy(
            options.epoch,
            longitude_deg,
            latitude_deg,
            self.policy,
        )?;
        let vtec = evaluation.value;
        validate::finite(vtec, "vtec").map_err(field_error_string)?;
        let obliquity_arg =
            options.shell_geometry.earth_radius_m * libm::cos(elevation_rad) / self.shell_radius_m;
        validate::finite(obliquity_arg, "obliquity_arg").map_err(field_error_string)?;
        let mapping_denominator = 1.0 - obliquity_arg * obliquity_arg;
        validate::finite_positive(mapping_denominator, "TEC mapping denominator")
            .map_err(field_error_string)?;
        let stec = vtec / mapping_denominator.sqrt();
        validate::finite(stec, "stec").map_err(field_error_string)?;
        Ok(TecGridEvaluation {
            value: (vtec, stec),
            degraded: evaluation.degraded,
        })
    }
}

fn group_delay_from_tec(
    tec: TecGridEvaluation<(f64, f64)>,
    frequency_hz: f64,
) -> Result<TecGridEvaluation<f64>, TecGridError> {
    let (_vtec, stec) = tec.value;
    let delay_m = IONOSPHERE_CONSTANT * stec / (frequency_hz * frequency_hz);
    validate::finite(delay_m, "ionosphere_delay_m").map_err(field_error_string)?;
    Ok(TecGridEvaluation {
        value: delay_m,
        degraded: tec.degraded,
    })
}

/// Returns the ECEF pierce point, `ecef_to_lla` applied to it once, and the
/// unclamped elevation.
#[cfg(all(test, sidereon_repo_tests))]
pub(crate) fn pierce_point_with_shell_radius<F>(
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    shell_radius_m: f64,
    ecef_to_lla: F,
) -> ([f64; 3], [f64; 3], f64)
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    let (pp_xyz, elevation_rad) = pierce_point_geometry(sat_xyz, receiver_xyz, shell_radius_m);
    let pp_lonlatalt = ecef_to_lla(&pp_xyz);
    (pp_xyz, pp_lonlatalt, elevation_rad)
}

/// Intersects the receiver-to-satellite ray with the shell and returns the
/// ECEF pierce point and the unclamped elevation. A ray that misses the shell
/// gives NaN pierce-point components.
fn pierce_point_geometry(
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    shell_radius_m: f64,
) -> ([f64; 3], f64) {
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
    (pp_xyz, elevation_rad)
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

fn validate_frequency(frequency_hz: f64) -> Result<f64, TecGridError> {
    validate::finite_positive(frequency_hz, "frequency_hz").map_err(field_error_string)
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
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(Some).to_vec(),
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
            [1.0, f64::NAN, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
                .map(Some)
                .to_vec(),
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
            vec![Some(0.0); 8],
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

    #[test]
    fn tec_grid_slice_accessors_preserve_stored_data_and_order() {
        let epochs = vec![0.0, 30_000_000_000.0];
        let latitudes = vec![-10.0, 0.0, 10.0];
        let longitudes = vec![100.0, 110.0];
        let values = vec![
            Some(1.5),
            Some(0.0),
            None,
            Some(3.0),
            Some(4.0),
            Some(5.0),
            None,
            Some(7.0),
            Some(8.0),
            Some(9.0),
            Some(10.0),
            Some(0.0),
        ];

        let grid = TecGrid::new(
            epochs.clone(),
            latitudes.clone(),
            longitudes.clone(),
            values.clone(),
        )
        .expect("valid TEC grid");

        assert_eq!(grid.epochs_ns(), epochs.as_slice());
        assert_eq!(grid.latitudes_deg(), latitudes.as_slice());
        assert_eq!(grid.longitudes_deg(), longitudes.as_slice());
        assert_eq!(grid.values(), values.as_slice());

        assert_eq!(grid.values()[1], Some(0.0));
        assert_eq!(grid.values()[2], None);
    }
}

/// Staged XYZ evaluation: request coordinates, order and count, error order
/// and values, each derived by hand from the geometry and the grid rather than
/// from the synchronous functions, which share the staged code.
#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod staged_xyz_tests {
    use super::*;
    use core::cell::RefCell;

    /// Mean Earth radius plus the default shell height, meters.
    const SHELL_RADIUS_M: f64 = 6_821_000.0;
    const L1_HZ: f64 = 1_575_420_000.0;

    /// Values `1 + (lon - 20) / 10 + 2 (lat / 10) + 4 (t / 10)` on the corners
    /// of one cell, so a dyadic query interpolates exactly.
    fn small_grid() -> TecGrid {
        TecGrid::new(
            vec![0.0, 10.0],
            vec![0.0, 10.0],
            vec![20.0, 30.0],
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(Some).to_vec(),
        )
        .expect("small TEC grid")
    }

    /// A binding keeps a conversion inside a native resource between steps,
    /// which requires owned, thread-safe data.
    #[test]
    fn staged_types_are_owned_and_thread_safe() {
        fn assert_owned<T: Send + Sync + 'static>() {}
        assert_owned::<TecGridXyzConversion>();
        assert_owned::<TecGridXyzStep>();
        assert_owned::<TecGridDelayXyzConversion>();
        assert_owned::<TecGridDelayXyzStep>();
    }

    /// Mid-cell epoch of [`small_grid`].
    fn options() -> TecGridEvalOptions {
        TecGridEvalOptions::new(TecGridEpoch::new(5, 1), L1_HZ)
    }

    /// A receiver on the mean sphere with the satellite straight overhead. The
    /// ray meets the default shell at exactly `[6_821_000, 0, 0]`, and the
    /// elevation is π/2, whose cosine leaves the obliquity factor at exactly 1.
    const RECEIVER: [f64; 3] = [6_371_000.0, 0.0, 0.0];
    const SATELLITE: [f64; 3] = [26_371_000.0, 0.0, 0.0];
    const PIERCE_POINT: [f64; 3] = [SHELL_RADIUS_M, 0.0, 0.0];

    fn bits(v: [f64; 3]) -> [u64; 3] {
        v.map(f64::to_bits)
    }

    /// A converter that records each request and answers from `answers` in
    /// order.
    fn recorder<'a>(
        calls: &'a RefCell<Vec<[f64; 3]>>,
        answers: &'a [[f64; 3]],
    ) -> impl Fn(&[f64; 3]) -> [f64; 3] + 'a {
        move |xyz: &[f64; 3]| {
            let mut calls = calls.borrow_mut();
            let answer = *answers
                .get(calls.len())
                .expect("no more conversions were expected");
            calls.push(*xyz);
            answer
        }
    }

    type Requests = Vec<(TecGridXyzTarget, [f64; 3])>;

    fn drive_tec(
        grid: &TecGrid,
        mut conversion: TecGridXyzConversion,
        answers: &[[f64; 3]],
    ) -> (
        Requests,
        Result<TecGridEvaluation<(f64, f64)>, TecGridError>,
    ) {
        let mut requests = Vec::new();
        loop {
            requests.push((conversion.target(), conversion.xyz()));
            let answer = *answers
                .get(requests.len() - 1)
                .expect("no more conversions were expected");
            match conversion.resume(grid, answer) {
                Err(error) => return (requests, Err(error)),
                Ok(TecGridXyzStep::Complete(evaluation)) => return (requests, Ok(evaluation)),
                Ok(TecGridXyzStep::Convert(next)) => conversion = next,
            }
        }
    }

    fn drive_delay(
        grid: &TecGrid,
        mut conversion: TecGridDelayXyzConversion,
        answers: &[[f64; 3]],
    ) -> (Requests, Result<TecGridEvaluation<f64>, TecGridError>) {
        let mut requests = Vec::new();
        loop {
            requests.push((conversion.target(), conversion.xyz()));
            let answer = *answers
                .get(requests.len() - 1)
                .expect("no more conversions were expected");
            match conversion.resume(grid, answer) {
                Err(error) => return (requests, Err(error)),
                Ok(TecGridDelayXyzStep::Complete(evaluation)) => return (requests, Ok(evaluation)),
                Ok(TecGridDelayXyzStep::Convert(next)) => conversion = next,
            }
        }
    }

    fn invalid(field: &'static str, reason: &'static str) -> TecGridError {
        TecGridError::InvalidField { field, reason }
    }

    /// Runs the synchronous and staged TEC paths with the same answers and
    /// checks both against the expected requests and result.
    #[allow(clippy::too_many_arguments)]
    fn check_tec(
        grid: &TecGrid,
        options: TecGridEvalOptions,
        sat: [f64; 3],
        receiver: [f64; 3],
        policy: IonexMissingNodePolicy,
        answers: &[[f64; 3]],
        expected_requests: &[(TecGridXyzTarget, [f64; 3])],
        expected: &Result<TecGridEvaluation<(f64, f64)>, TecGridError>,
    ) {
        let calls = RefCell::new(Vec::new());
        let sync = tec_xyz_with_policy(
            grid,
            options,
            &sat,
            &receiver,
            recorder(&calls, answers),
            policy,
        );
        let calls = calls.into_inner();
        assert_eq!(
            calls.iter().copied().map(bits).collect::<Vec<_>>(),
            expected_requests
                .iter()
                .map(|(_, xyz)| bits(*xyz))
                .collect::<Vec<_>>(),
            "callback coordinates, order and count"
        );
        assert_evaluation_eq(&sync, expected);

        let conversion = TecGridXyzConversion::prepare(options, &sat, &receiver, policy)
            .expect("the expected requests imply a valid preparation");
        let (requests, staged) = drive_tec(grid, conversion, answers);
        assert_eq!(
            requests
                .iter()
                .map(|(target, xyz)| (*target, bits(*xyz)))
                .collect::<Vec<_>>(),
            expected_requests
                .iter()
                .map(|(target, xyz)| (*target, bits(*xyz)))
                .collect::<Vec<_>>(),
            "staged request targets, coordinates, order and count"
        );
        assert_evaluation_eq(&staged, expected);
    }

    fn assert_evaluation_eq<T: PartialEq + core::fmt::Debug + Copy + ValueBits>(
        got: &Result<TecGridEvaluation<T>, TecGridError>,
        want: &Result<TecGridEvaluation<T>, TecGridError>,
    ) {
        match (got, want) {
            (Ok(got), Ok(want)) => {
                assert_eq!(got.value.value_bits(), want.value.value_bits(), "{got:?}");
                assert_eq!(got.degraded, want.degraded);
            }
            _ => assert_eq!(got, want),
        }
    }

    trait ValueBits {
        fn value_bits(self) -> Vec<u64>;
    }

    impl ValueBits for f64 {
        fn value_bits(self) -> Vec<u64> {
            vec![self.to_bits()]
        }
    }

    impl ValueBits for (f64, f64) {
        fn value_bits(self) -> Vec<u64> {
            vec![self.0.to_bits(), self.1.to_bits()]
        }
    }

    fn complete<T>(value: T) -> Result<TecGridEvaluation<T>, TecGridError> {
        Ok(TecGridEvaluation {
            value,
            degraded: None,
        })
    }

    #[test]
    fn finite_answer_requests_the_pierce_point_once() {
        // Answer (25, 5) is the cell center at the mid epoch: 1 + 0.5 + 1 + 2.
        check_tec(
            &small_grid(),
            options(),
            SATELLITE,
            RECEIVER,
            IonexMissingNodePolicy::Strict,
            &[[25.0, 5.0, 450_000.0]],
            &[(TecGridXyzTarget::PiercePoint, PIERCE_POINT)],
            &complete((4.5, 4.5)),
        );
    }

    #[test]
    fn nan_answer_requests_the_receiver_once_and_ignores_its_altitude() {
        // (22.5, 2.5) interpolates to 1 + 0.25 + 0.5 + 2. The NaN receiver
        // altitude is not read, so there is no third request.
        for first in [
            [f64::NAN, f64::NAN, f64::NAN],
            [25.0, 5.0, f64::NAN],
            [f64::NAN, 5.0, 450_000.0],
        ] {
            check_tec(
                &small_grid(),
                options(),
                SATELLITE,
                RECEIVER,
                IonexMissingNodePolicy::Strict,
                &[first, [22.5, 2.5, f64::NAN]],
                &[
                    (TecGridXyzTarget::PiercePoint, PIERCE_POINT),
                    (TecGridXyzTarget::Receiver, RECEIVER),
                ],
                &complete((3.75, 3.75)),
            );
        }
    }

    #[test]
    fn nan_receiver_answer_is_not_retried() {
        check_tec(
            &small_grid(),
            options(),
            SATELLITE,
            RECEIVER,
            IonexMissingNodePolicy::Strict,
            &[[f64::NAN; 3], [22.5, f64::NAN, 0.0]],
            &[
                (TecGridXyzTarget::PiercePoint, PIERCE_POINT),
                (TecGridXyzTarget::Receiver, RECEIVER),
            ],
            &Err(invalid("latitude", "not finite")),
        );
    }

    #[test]
    fn tangent_ray_missing_the_shell_still_requests_its_nan_pierce_point() {
        // The ray along +y from radius 7,000,000 m touches that sphere and
        // never meets the 6,821,000 m shell: b = 0, c > 0, so t and every
        // pierce-point component are NaN. The elevation is 0, raised to 5°.
        let receiver = [7_000_000.0, 0.0, 0.0];
        let sat = [7_000_000.0, 1_000_000.0, 0.0];
        let grid = small_grid();

        let conversion = TecGridXyzConversion::prepare(
            options(),
            &sat,
            &receiver,
            IonexMissingNodePolicy::Strict,
        )
        .expect("finite inputs with a nonzero line of sight are valid");
        assert_eq!(conversion.target(), TecGridXyzTarget::PiercePoint);
        assert!(conversion.xyz().iter().all(|v| v.is_nan()));

        let calls = RefCell::new(Vec::new());
        let (vtec, stec) = tec_xyz(
            &grid,
            options(),
            &sat,
            &receiver,
            recorder(&calls, &[[25.0, 5.0, 0.0]]),
        )
        .expect("a finite answer finishes");
        let calls = calls.into_inner();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].iter().all(|v| v.is_nan()));
        assert_eq!(vtec.to_bits(), 4.5f64.to_bits());
        // Thin-shell mapping 4.5 / sqrt(1 - (R cos 5° / (R + h))^2).
        let expected_stec = 12.282_985_570_258_32;
        assert!(
            ((stec - expected_stec) / expected_stec).abs() < 1e-12,
            "{stec}"
        );

        // A NaN answer to that request falls back to the receiver.
        let (requests, staged) = drive_tec(&grid, conversion, &[[f64::NAN; 3], [25.0, 5.0, 0.0]]);
        assert_eq!(requests.len(), 2);
        assert!(requests[0].1.iter().all(|v| v.is_nan()));
        assert_eq!(requests[1], (TecGridXyzTarget::Receiver, receiver));
        let staged = staged.expect("receiver fallback finishes");
        assert_eq!(staged.value.0.to_bits(), vtec.to_bits());
        assert_eq!(staged.value.1.to_bits(), stec.to_bits());
    }

    #[test]
    fn invalid_elevation_fails_after_the_first_request_without_fallback() {
        // |x|^2 = 1e-320 is subnormal, so the receiver norm rounds to
        // 9.99994e-161 and the receiver unit vector has x = 1.0000056: the
        // elevation is asin of a value above 1, NaN. The pierce point is still
        // exactly the shell radius on +x.
        check_tec(
            &small_grid(),
            options(),
            [20_000_000.0, 0.0, 0.0],
            [1e-160, 0.0, 0.0],
            IonexMissingNodePolicy::Strict,
            &[[f64::NAN; 3]],
            &[(TecGridXyzTarget::PiercePoint, PIERCE_POINT)],
            &Err(invalid("elevation_rad", "not finite")),
        );
    }

    #[test]
    fn invalid_geometry_requests_nothing() {
        let calls = RefCell::new(Vec::new());
        let error = tec_xyz(
            &small_grid(),
            options(),
            &SATELLITE,
            &[0.0, 0.0, 0.0],
            recorder(&calls, &[]),
        )
        .expect_err("zero receiver");
        assert_eq!(error, invalid("receiver radius_m", "not positive"));
        assert!(calls.into_inner().is_empty());
        let error = TecGridXyzConversion::prepare(
            options(),
            &SATELLITE,
            &[0.0, 0.0, 0.0],
            IonexMissingNodePolicy::Strict,
        )
        .expect_err("zero receiver");
        assert_eq!(error, invalid("receiver radius_m", "not positive"));

        let error = TecGridXyzConversion::prepare(
            options(),
            &RECEIVER,
            &RECEIVER,
            IonexMissingNodePolicy::Strict,
        )
        .expect_err("zero line of sight");
        assert_eq!(error, invalid("line of sight_m", "not positive"));
    }

    #[test]
    fn delay_checks_frequency_before_geometry() {
        let mut options = options();
        options.frequency_hz = 0.0;
        let calls = RefCell::new(Vec::new());
        let error = iono_delay_xyz(
            &small_grid(),
            options,
            &[f64::NAN, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            recorder(&calls, &[]),
        )
        .expect_err("invalid frequency and geometry");
        assert_eq!(error, invalid("frequency_hz", "not positive"));
        assert!(calls.into_inner().is_empty());

        let error = TecGridDelayXyzConversion::prepare(
            options,
            &[f64::NAN, 0.0, 0.0],
            &[0.0, 0.0, 0.0],
            IonexMissingNodePolicy::Strict,
        )
        .expect_err("invalid frequency and geometry");
        assert_eq!(error, invalid("frequency_hz", "not positive"));
    }

    #[test]
    fn tec_ignores_the_frequency() {
        let mut options = options();
        options.frequency_hz = f64::NAN;
        check_tec(
            &small_grid(),
            options,
            SATELLITE,
            RECEIVER,
            IonexMissingNodePolicy::Strict,
            &[[25.0, 5.0, 0.0]],
            &[(TecGridXyzTarget::PiercePoint, PIERCE_POINT)],
            &complete((4.5, 4.5)),
        );
    }

    #[test]
    fn delay_scales_slant_tec_on_both_paths() {
        // 40.308193e16 * stec / 1575.42e6^2 meters.
        let grid = small_grid();
        for (answers, targets, expected) in [
            (
                vec![[25.0, 5.0, 0.0]],
                vec![TecGridXyzTarget::PiercePoint],
                0.730_824_560_418_891_8,
            ),
            (
                vec![[f64::NAN; 3], [22.5, 2.5, 0.0]],
                vec![TecGridXyzTarget::PiercePoint, TecGridXyzTarget::Receiver],
                0.609_020_467_015_743_1,
            ),
        ] {
            let calls = RefCell::new(Vec::new());
            let sync = iono_delay_xyz(
                &grid,
                options(),
                &SATELLITE,
                &RECEIVER,
                recorder(&calls, &answers),
            )
            .expect("delay");
            assert_eq!(sync.to_bits(), f64::to_bits(expected));
            assert_eq!(calls.into_inner().len(), answers.len());

            let conversion = TecGridDelayXyzConversion::prepare(
                options(),
                &SATELLITE,
                &RECEIVER,
                IonexMissingNodePolicy::Strict,
            )
            .expect("valid delay inputs");
            let (requests, staged) = drive_delay(&grid, conversion, &answers);
            assert_eq!(
                requests.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
                targets
            );
            assert_evaluation_eq(&staged, &complete(expected));
        }
    }

    #[test]
    fn delay_frequency_whose_square_underflows_fails_after_the_request() {
        // 1e-200 is finite and positive; its square is 0 and the delay is
        // infinite, which only the final delay check refuses.
        let mut options = options();
        options.frequency_hz = 1e-200;
        let calls = RefCell::new(Vec::new());
        let error = iono_delay_xyz(
            &small_grid(),
            options,
            &SATELLITE,
            &RECEIVER,
            recorder(&calls, &[[25.0, 5.0, 0.0]]),
        )
        .expect_err("infinite delay");
        assert_eq!(error, invalid("ionosphere_delay_m", "not finite"));
        assert_eq!(calls.into_inner().len(), 1);

        let conversion = TecGridDelayXyzConversion::prepare(
            options,
            &SATELLITE,
            &RECEIVER,
            IonexMissingNodePolicy::Strict,
        )
        .expect("the frequency itself is valid");
        let (requests, staged) = drive_delay(&small_grid(), conversion, &[[25.0, 5.0, 0.0]]);
        assert_eq!(requests.len(), 1);
        assert_eq!(staged, Err(invalid("ionosphere_delay_m", "not finite")));
    }

    #[test]
    fn timestamp_outside_the_grid_is_reported_after_the_conversion() {
        let mut options = options();
        options.epoch = TecGridEpoch::new(100, 1);
        check_tec(
            &small_grid(),
            options,
            SATELLITE,
            RECEIVER,
            IonexMissingNodePolicy::Strict,
            &[[25.0, 5.0, 0.0]],
            &[(TecGridXyzTarget::PiercePoint, PIERCE_POINT)],
            &Err(TecGridError::OutOfBounds {
                name: "timestamp",
                value: 100.0,
            }),
        );
    }

    #[test]
    fn missing_node_is_strict_or_renormalized() {
        let mut values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0].map(Some).to_vec();
        values[0] = None;
        let grid = TecGrid::new(vec![0.0, 10.0], vec![0.0, 10.0], vec![20.0, 30.0], values)
            .expect("grid with one missing node");
        let gap = IonexNodeGap {
            earlier: Some(IonexMissingNodes {
                map_number: 1,
                lat_index: 0,
                lon_index: 0,
                lon_index_next: 1,
                missing: [true, false, false, false],
            }),
            later: None,
        };
        let requests = [(TecGridXyzTarget::PiercePoint, PIERCE_POINT)];
        let answers = [[25.0, 5.0, 0.0]];

        check_tec(
            &grid,
            options(),
            SATELLITE,
            RECEIVER,
            IonexMissingNodePolicy::Strict,
            &answers,
            &requests,
            &Err(TecGridError::NodesNotAvailable(gap)),
        );
        // Earlier map: mean of 2, 3 and 4 is 3; later map: mean of 5..8 is
        // 6.5; equal temporal weights give 4.75.
        check_tec(
            &grid,
            options(),
            SATELLITE,
            RECEIVER,
            IonexMissingNodePolicy::Renormalize,
            &answers,
            &requests,
            &Ok(TecGridEvaluation {
                value: (4.75, 4.75),
                degraded: Some(gap),
            }),
        );
    }

    #[test]
    fn infinite_latitude_clamps_and_infinite_longitude_is_refused() {
        // Latitude rows -87.5 and 87.5 hold 1 and 3 on both maps.
        let grid = TecGrid::new(
            vec![0.0, 10.0],
            vec![-87.5, 87.5],
            vec![20.0, 30.0],
            [1.0, 1.0, 3.0, 3.0, 1.0, 1.0, 3.0, 3.0].map(Some).to_vec(),
        )
        .expect("polar grid");
        let requests = [(TecGridXyzTarget::PiercePoint, PIERCE_POINT)];
        for (answer, expected) in [
            ([25.0, f64::INFINITY, f64::INFINITY], complete((3.0, 3.0))),
            ([25.0, f64::NEG_INFINITY, 0.0], complete((1.0, 1.0))),
            (
                [f64::INFINITY, 0.0, 0.0],
                Err(invalid("longitude", "not finite")),
            ),
            (
                [f64::NEG_INFINITY, 0.0, 0.0],
                Err(invalid("longitude", "not finite")),
            ),
        ] {
            check_tec(
                &grid,
                options(),
                SATELLITE,
                RECEIVER,
                IonexMissingNodePolicy::Strict,
                &[answer],
                &requests,
                &expected,
            );
        }
    }

    #[test]
    fn conversion_holds_no_grid_between_steps() {
        let conversion = TecGridXyzConversion::prepare(
            options(),
            &SATELLITE,
            &RECEIVER,
            IonexMissingNodePolicy::Strict,
        )
        .expect("valid inputs");
        let step = {
            // A grid that lives only for the first step.
            let first = small_grid();
            conversion
                .resume(&first, [f64::NAN; 3])
                .expect("fallback request")
        };
        let TecGridXyzStep::Convert(receiver) = step else {
            panic!("a NaN answer requests the receiver");
        };
        assert_eq!(receiver.target(), TecGridXyzTarget::Receiver);
        assert_eq!(bits(receiver.xyz()), bits(RECEIVER));
        let second = small_grid();
        let TecGridXyzStep::Complete(evaluation) = receiver
            .resume(&second, [22.5, 2.5, 0.0])
            .expect("receiver answer finishes")
        else {
            panic!("a receiver answer always finishes");
        };
        assert_eq!(evaluation.value, (3.75, 3.75));
    }
}
