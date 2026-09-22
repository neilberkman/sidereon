//! Ionospheric delay models.
//!
//! This module exposes the single-frequency ionospheric group-delay models used
//! to correct GNSS pseudoranges. The GPS broadcast Klobuchar model and the IONEX
//! vertical-TEC grid path are both reached through the same [`ionosphere_delay`]
//! entry, and the IONEX grid parser is exposed directly as [`Ionex`].
//!
//! All delays returned are group delays and are positive: they increase the
//! measured pseudorange (the carrier-phase advance is the negation of this
//! value). The ionosphere is dispersive, so a delay reported on a carrier other
//! than the model's native L1 is the L1 delay scaled by `(f_l1 / f)^2`.

#![warn(clippy::expect_used, clippy::unwrap_used, clippy::panic)]

mod grid;
mod header;
mod klobuchar;
mod nequick_g;
mod nequick_g_data;
mod samples;
mod slant;
mod tec_grid;
mod write;

#[cfg(all(test, sidereon_repo_tests))]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests;

use crate::astro::constants::time::{DAYS_PER_JULIAN_YEAR, SECONDS_PER_DAY, SECONDS_PER_HOUR};
use crate::astro::time::civil::{
    fractional_day_of_year_from_instant, second_of_day_from_instant,
    split_julian_date_from_j2000_seconds, J2000_JULIAN_DAY_NUMBER,
};
use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};

use crate::constants::{DEG_TO_RAD, MEAN_EARTH_RADIUS_M, RAD_TO_DEG};
use crate::error::{Error, Result};
use crate::frame::Wgs84Geodetic;
use crate::frequencies::{self, CarrierBand};
use crate::GnssSystem;

pub use grid::Ionex;
pub use header::{
    IonexAssumedMapping, IonexHeader, IonexMappingDeclaration, IonexMappingFunction, IonexWarning,
};
pub use nequick_g::{nequick_g_delay_m, nequick_g_stec_tecu, NequickGRayEval};
pub use samples::{TecGridSamples, TecSample, TecSamplesError};
pub use tec_grid::{
    iono_delay_xyz as regular_tec_grid_delay_xyz,
    iono_delay_xyz_with_policy as regular_tec_grid_delay_xyz_with_policy,
    tec_xyz as regular_tec_xyz, tec_xyz_with_policy as regular_tec_xyz_with_policy, TecGrid,
    TecGridDelayXyzConversion, TecGridDelayXyzStep, TecGridEpoch, TecGridError, TecGridEvalOptions,
    TecGridEvaluation, TecGridShellGeometry, TecGridXyzConversion, TecGridXyzStep,
    TecGridXyzTarget,
};

/// Policy applied when an IONEX query lands outside the product's coverage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IonexCoveragePolicy {
    /// Return a typed error before exposing a held value.
    #[default]
    Strict,
    /// Hold the nearest map or grid edge and return a status marker.
    Hold,
}

/// IONEX coverage miss detected during slant-delay evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IonexCoverageError {
    /// Query epoch precedes the first map epoch.
    EpochBeforeFirstMap,
    /// Query epoch follows the last map epoch.
    EpochAfterLastMap,
    /// Pierce-point latitude is outside the latitude nodes.
    LatitudeOutOfRange,
    /// Pierce-point longitude is outside the longitude nodes.
    LongitudeOutOfRange,
}

impl core::fmt::Display for IonexCoverageError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let message = match self {
            Self::EpochBeforeFirstMap => "epoch precedes first map",
            Self::EpochAfterLastMap => "epoch follows last map",
            Self::LatitudeOutOfRange => "latitude outside grid",
            Self::LongitudeOutOfRange => "longitude outside grid",
        };
        f.write_str(message)
    }
}

/// Policy applied when an IONEX interpolation weights grid nodes the product
/// gives as non-available.
///
/// A node is weighted when its bilinear weight is nonzero, and a map when its
/// temporal weight is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IonexMissingNodePolicy {
    /// Return [`Error::IonexNodesNotAvailable`], naming the map, the cell and the
    /// missing nodes.
    #[default]
    Strict,
    /// Interpolate from the weighted nodes that hold values, their bilinear
    /// weights renormalized to sum to one, and from the weighted maps that give
    /// a value, their temporal weights renormalized the same way, and mark the
    /// value degraded. With no weighted node holding a value on any weighted map
    /// there is no value, and the evaluation returns
    /// [`Error::IonexNodesNotAvailable`].
    Renormalize,
}

/// The factor an IONEX slant delay maps vertical TEC to the line of sight with.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IonexMappingPolicy {
    /// The factor the product's `MAPPING FUNCTION` defines. `COSZ` defines the
    /// single-layer `1/cos(z')` at the shell height. `NONE` says no mapping
    /// function was used, `QFAC` names one the spec gives no formula for, and
    /// another code or no code defines none, so the evaluation returns
    /// [`Error::IonexSlantUnavailable`].
    Declared,
    /// The single-layer `1/cos(z')` at the shell height, whatever the product
    /// declares, with [`IonexSlantDelayStatus::assumed_mapping`] naming what a
    /// product that declares anything but `COSZ` declares. This is the default:
    /// most global products declare `NONE` while their descriptions name the
    /// mapping function their maps were determined with, and a vertical TEC map
    /// is mapped to the line of sight this way whatever determined it.
    #[default]
    SingleLayer,
}

/// The policies an IONEX slant-delay evaluation applies.
///
/// The default refuses a query the product's grid cannot answer as it stands, a
/// query outside its coverage and one whose interpolation weights a
/// non-available node, and maps vertical TEC to the line of sight with the
/// single-layer `1/cos(z')`, naming what the product declares in
/// [`IonexSlantDelayStatus::assumed_mapping`] where that is not `COSZ`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub struct IonexSlantPolicy {
    /// Queries outside the product's epochs, latitudes or longitudes.
    pub coverage: IonexCoveragePolicy,
    /// Queries whose interpolation weights nodes the product gives as
    /// non-available.
    pub missing_nodes: IonexMissingNodePolicy,
    /// The factor that maps vertical TEC to the line of sight.
    pub mapping: IonexMappingPolicy,
}

impl IonexSlantPolicy {
    /// This policy with `coverage`.
    #[must_use]
    pub const fn with_coverage(mut self, coverage: IonexCoveragePolicy) -> Self {
        self.coverage = coverage;
        self
    }

    /// This policy with `missing_nodes`.
    #[must_use]
    pub const fn with_missing_nodes(mut self, missing_nodes: IonexMissingNodePolicy) -> Self {
        self.missing_nodes = missing_nodes;
        self
    }

    /// This policy with `mapping`.
    #[must_use]
    pub const fn with_mapping(mut self, mapping: IonexMappingPolicy) -> Self {
        self.mapping = mapping;
        self
    }
}

impl From<IonexCoveragePolicy> for IonexSlantPolicy {
    fn from(coverage: IonexCoveragePolicy) -> Self {
        Self::default().with_coverage(coverage)
    }
}

/// Why an IONEX product gives no slant delay under the requested policy.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum IonexSlantRefusal {
    /// The product's height maps give nodes different single-layer heights, and
    /// the slant delay uses one shell height. The indices name the first node
    /// whose height differs from the first height map's first node.
    VaryingHeights {
        /// Number of the height map, counting from 1 as the file numbers its
        /// maps.
        map_number: usize,
        /// Latitude index of the node.
        lat_index: usize,
        /// Longitude index of the node.
        lon_index: usize,
    },
    /// A height map gives a node's height as non-available, so the single-layer
    /// height there is unknown.
    HeightNotAvailable {
        /// Number of the height map, counting from 1 as the file numbers its
        /// maps.
        map_number: usize,
        /// Latitude index of the node.
        lat_index: usize,
        /// Longitude index of the node.
        lon_index: usize,
    },
    /// Under [`IonexMappingPolicy::Declared`], the product's `MAPPING FUNCTION`
    /// defines no factor the slant delay applies.
    MappingFunction(IonexMappingDeclaration),
}

impl core::fmt::Display for IonexSlantRefusal {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::VaryingHeights {
                map_number,
                lat_index,
                lon_index,
            } => write!(
                f,
                "height map {map_number} gives node [{lat_index}][{lon_index}] another \
                 single-layer height than the first node; the slant delay uses one shell height"
            ),
            Self::HeightNotAvailable {
                map_number,
                lat_index,
                lon_index,
            } => write!(
                f,
                "height map {map_number} gives the height of node [{lat_index}][{lon_index}] as \
                 non-available"
            ),
            Self::MappingFunction(IonexMappingDeclaration::Declared(function)) => write!(
                f,
                "MAPPING FUNCTION {} defines no factor the slant delay applies; \
                 IonexMappingPolicy::SingleLayer, the default, applies 1/cos(z')",
                function.code()
            ),
            Self::MappingFunction(IonexMappingDeclaration::Absent) => f.write_str(
                "the product gives no MAPPING FUNCTION; IonexMappingPolicy::SingleLayer, the \
                 default, applies 1/cos(z')",
            ),
        }
    }
}

/// Status of a successful IONEX slant-delay value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IonexSlantDelayStatus {
    /// The coverage miss [`IonexCoveragePolicy::Hold`] held the value through,
    /// if any.
    pub held: Option<IonexCoverageError>,
    /// The non-available nodes [`IonexMissingNodePolicy::Renormalize`]
    /// interpolated around, if any. A value with this set is degraded.
    pub degraded: Option<IonexNodeGap>,
    /// Which mapping function the product declares, where the single-layer
    /// `1/cos(z')` mapped a product that declares anything else. `None` where
    /// the product declares `COSZ`, which is the factor applied, so the value
    /// rests on nothing the product does not state. The code's text, for an
    /// [`IonexAssumedMapping::Other`], is in [`IonexHeader::mapping_function`].
    ///
    /// This is not a coverage or node failure: the value is a nominal one, and
    /// [`Self::is_valid`] stays true with it set.
    pub assumed_mapping: Option<IonexAssumedMapping>,
}

impl IonexSlantDelayStatus {
    /// A value inside the product's coverage, interpolated from nodes that all
    /// hold values, mapped with the factor the product declares.
    pub const VALID: Self = Self {
        held: None,
        degraded: None,
        assumed_mapping: None,
    };

    /// Whether the value was neither held through a coverage miss nor degraded
    /// by a non-available node.
    ///
    /// [`Self::assumed_mapping`] is not part of this: most published products
    /// declare something other than `COSZ`, so a caller asking whether a value
    /// is inside coverage with every node present would read false on nominal
    /// results from the CODE, ESA, JPL and UPC GIMs. A caller that cares which
    /// factor was applied reads that field.
    pub fn is_valid(&self) -> bool {
        self.held.is_none() && self.degraded.is_none()
    }
}

/// IONEX slant-delay value with its coverage status.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IonexSlantDelayEvaluation {
    /// Ionospheric group delay, meters.
    pub delay_m: f64,
    /// Coverage status for `delay_m`.
    pub status: IonexSlantDelayStatus,
}

/// Nodes of one map's interpolation cell that carry weight in a slant-delay
/// query and that the product gives as non-available.
///
/// A node carries weight when its bilinear weight is nonzero, so a query on a
/// node, or on the edge between two nodes, uses only the nodes it lies on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IonexMissingNodes {
    /// The map's number, counting from 1 as a file numbers its maps.
    pub map_number: usize,
    /// Index in [`Ionex::lat_nodes_deg`] of the cell's first node row. A cell
    /// index is a position in the node axes, which count from 0, where a map
    /// number counts from 1.
    pub lat_index: usize,
    /// Index in [`Ionex::lon_nodes_deg`] of the cell's first node column.
    pub lon_index: usize,
    /// Index in [`Ionex::lon_nodes_deg`] of the cell's other node column.
    ///
    /// Within the grid this is `lon_index + 1`. On the cell that closes the
    /// circle it is `0`: a longitude axis such as 0 to 355 by 5 covers every
    /// longitude without naming the seam twice, so that cell runs from the last
    /// column to the first, and the grid has no column 72. The latitude axis
    /// does not wrap, so the cell's other node row is always `lat_index + 1`.
    pub lon_index_next: usize,
    /// Which weighted nodes are non-available, in the order
    /// `[lat_index][lon_index]`, `[lat_index][lon_index_next]`,
    /// `[lat_index + 1][lon_index]`, `[lat_index + 1][lon_index_next]`.
    pub missing: [bool; 4],
}

impl core::fmt::Display for IonexMissingNodes {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let Self {
            map_number,
            lat_index,
            lon_index,
            lon_index_next,
            missing,
        } = *self;
        // The map counts from 1 as a file numbers its maps; the cell and node
        // indices are positions in the node axes, which count from 0.
        write!(
            f,
            "map {map_number} cell [{lat_index}][{lon_index}] missing"
        )?;
        // The cell's other column wraps to the first on the cell that closes
        // the circle, so it is carried rather than added to.
        let corners = [
            (0, lon_index),
            (0, lon_index_next),
            (1, lon_index),
            (1, lon_index_next),
        ];
        for ((lat_offset, lon), _) in corners
            .into_iter()
            .zip(missing)
            .filter(|(_, is_missing)| *is_missing)
        {
            write!(f, " [{}][{}]", lat_index + lat_offset, lon)?;
        }
        Ok(())
    }
}

/// The non-available nodes a slant-delay query weights, on each weighted map
/// that has any.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IonexNodeGap {
    /// On the earlier of the two maps bracketing the query epoch, or the only
    /// map.
    pub earlier: Option<IonexMissingNodes>,
    /// On the later of the two maps bracketing the query epoch.
    pub later: Option<IonexMissingNodes>,
}

impl core::fmt::Display for IonexNodeGap {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut separator = "";
        for nodes in [self.earlier, self.later].into_iter().flatten() {
            write!(f, "{separator}{nodes}")?;
            separator = "; ";
        }
        Ok(())
    }
}

pub(crate) use klobuchar::klobuchar_l1_components;
pub(crate) use slant::pierce_point;

pub(crate) fn ionex_epoch_from_j2000_seconds(seconds: i64) -> Instant {
    instant_from_j2000_seconds(TimeScale::Utc, seconds)
}

// invariant: split_julian_date_from_j2000_seconds returns a valid normalized split.
#[allow(clippy::expect_used)]
pub(crate) fn instant_from_j2000_seconds(scale: TimeScale, seconds: i64) -> Instant {
    let (jd_whole, fraction) = split_julian_date_from_j2000_seconds(seconds);
    Instant::from_julian_date(
        scale,
        JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date"),
    )
}

/// Nanoseconds per second, for the integer-nanosecond instant representation.
const NANOS_PER_SECOND_I128: i128 = 1_000_000_000;

/// `86_400 = 2^7 * 675`. A Julian-date offset in days is a whole number of
/// seconds exactly when it is a whole multiple of `2^-7` days, and the second
/// it names is `675` times that multiple. Scaling by the power of two is an
/// exponent shift, so it is exact for every finite double.
const DAY_SCALE: f64 = 128.0;
const DAY_SCALE_I128: i128 = 128;
const SECONDS_PER_SCALED_DAY: i128 = 675;

/// The J2000 Julian-date origin on the `2^-7`-day scale. JD 2451545.0 is noon
/// of Julian day number [`J2000_JULIAN_DAY_NUMBER`], so the origin is a whole
/// number of days and scales without a remainder.
const J2000_SCALED_DAYS: i128 = J2000_JULIAN_DAY_NUMBER as i128 * DAY_SCALE_I128;

/// `2^47` days, the magnitude bound on a split Julian date's recombined value.
///
/// A recombined value this far from zero is more than `i64::MAX` seconds from
/// the J2000 origin in either direction - `(2^47 - 2_451_546) * 86_400` already
/// exceeds `2^63` - so the bound refuses only epochs the whole-second axis
/// could not hold anyway. Below it, the scaled value stays under `2^54` and
/// every remaining step is exact in `i128`.
const JD_SUM_LIMIT: f64 = 140_737_488_355_328.0;

/// The exact whole J2000 second an IONEX map epoch names, or `None` where the
/// instant names no whole second the epoch axis can hold.
///
/// An IONEX file states every map epoch as six whole civil fields, so the epoch
/// axis is an axis of whole seconds. Every place that reads a second off a
/// stored [`Instant`] - sample validation and grouping, the
/// strictly-increasing check, the whole-day diurnal shift, the
/// [`Ionex::map_epochs_s`] compatibility view, slant map-time evaluation and
/// the writer's epoch record - goes through this one contract, so no two of
/// them can disagree about which second an epoch is, and none of them can
/// silently move an epoch onto a different second. The instant is read as it
/// stands: its scale tag is not consulted and no time system is shifted.
///
/// [`InstantRepr::Nanos`] counts nanoseconds from the J2000 origin in the
/// instant's own scale, the convention
/// [`crate::astro::time::civil::julian_date_from_instant`] documents. The count
/// is divided in `i128`, so a count past the 53-bit integers an `f64` holds
/// keeps every second it states. A count that is not a whole number of seconds,
/// or whose seconds fall outside `i64`, is refused rather than rounded or
/// saturated.
///
/// [`InstantRepr::JulianDate`] carries a day boundary and a residual day
/// fraction. Two readings are accepted, in this order.
///
/// 1. *The value the two parts sum to.* Both parts are binary floats, so their
///    exact real sum is a dyadic rational, and `86_400 = 2^7 * 675` makes that
///    sum a whole number of seconds exactly when it is a whole multiple of
///    `2^-7` days. [`two_sum`] recovers the sum and its rounding error exactly,
///    both are scaled by `2^7`, and the reading holds only when both are
///    integral; the second is then `675` times the scaled day count. This is
///    exact real arithmetic, so it does not care which boundary `jd_whole`
///    names: `2_451_545.25` with a zero fraction is J2000 + 21_600 s and is
///    read as that, and a `jd_whole` one bit above `2_451_545.0` with a
///    fraction of exactly the negative of that bit sums to the origin and is
///    read as second 0.
/// 2. *The whole second the two parts encode.* A day fraction is a binary
///    approximation: `11 / 86_400` is not a dyadic rational, so the reader's
///    own J2000 + 11 s epoch has no whole-second value in exact real arithmetic
///    and reading 1 alone would refuse it. Where the boundary alone names a
///    whole second by reading 1, the fraction is therefore also read as an
///    encoding: the candidate residual `(fraction * 86_400).round()` is taken
///    only when `residual as f64 / 86_400.0` - the expression both
///    [`split_julian_date_from_j2000_seconds`] and
///    [`crate::astro::time::split_julian_date`] form - reproduces `fraction`
///    bit for bit.
///
/// The two readings cannot disagree. Reading 2 only adds epochs whose fraction
/// is not exactly `residual / 86_400`, and for those the exact sum is not a
/// whole multiple of `2^-7` days at all, so reading 1 has already declined
/// them; where the fraction is exact - a residual that is a whole multiple of
/// 675 s - both readings give the same second.
///
/// Reading 2 is the one place this conversion is not exact real arithmetic, and
/// the imprecision is bounded rather than hidden. An accepted encoding lies
/// within `86_400 * 2^-53` s of the second it is read as, under a hundredth of
/// a nanosecond, while distinct residuals are `1 / 86_400` apart in day
/// fraction, eleven orders of magnitude wider. No two seconds can therefore
/// share an encoding, and the one `f64` either side of an accepted fraction is
/// refused by both readings. No tolerance is applied anywhere: every acceptance
/// is an equality.
///
/// One case is left out deliberately, and it is a limit rather than a
/// restriction on any epoch that can be shown to name a second. A split whose
/// boundary names no whole second on its own and whose fraction is not exactly
/// a whole number of seconds carries no exact evidence of which second it
/// means: its exact value is not one, and the pair is not a form either encoder
/// produces, so reading it would mean picking a nearby second on no better
/// ground than proximity. Such a split is refused. Every split whose parts do
/// state a second - by their exact sum, or as a boundary that names one plus an
/// encoded residual - is read, whatever boundary it uses.
///
/// Representability here is this converter's, not the file grammar's: a second
/// returned here may still have a civil year no `I6` epoch field can print,
/// which the writer refuses by name when it formats the record.
pub(crate) fn exact_j2000_second(epoch: Instant) -> Option<i64> {
    let seconds = match epoch.repr {
        InstantRepr::Nanos(nanos) => {
            if nanos.rem_euclid(NANOS_PER_SECOND_I128) != 0 {
                return None;
            }
            nanos.div_euclid(NANOS_PER_SECOND_I128)
        }
        InstantRepr::JulianDate(split) => split_j2000_second(split.jd_whole, split.fraction)?,
    };
    i64::try_from(seconds).ok()
}

/// The whole J2000 second a split Julian date names, held as `i128` so the one
/// narrowing onto the epoch axis happens in [`exact_j2000_second`] and cannot
/// wrap on the way.
fn split_j2000_second(jd_whole: f64, fraction: f64) -> Option<i128> {
    summed_split_second(jd_whole, fraction).or_else(|| encoded_split_second(jd_whole, fraction))
}

/// Reading 1: the whole second the two parts sum to in exact real arithmetic.
fn summed_split_second(jd_whole: f64, fraction: f64) -> Option<i128> {
    let (sum, residue) = two_sum(jd_whole, fraction);
    // A non-finite part leaves `sum` non-finite, so the finiteness test refuses
    // NaN and both infinities before the magnitude bound is consulted.
    if !sum.is_finite() || sum.abs() >= JD_SUM_LIMIT {
        return None;
    }
    // Exact: multiplying by `2^7` only moves the exponent, and neither product
    // can overflow below the bound above.
    let scaled_sum = sum * DAY_SCALE;
    let scaled_residue = residue * DAY_SCALE;
    // `sum + residue` is the exact real value, and `residue` is the rounding
    // error of `sum`, so `|scaled_residue| <= ulp(scaled_sum) / 2`. A double
    // that is not an integer is a whole multiple of its own ulp and so sits at
    // least one ulp from every integer, which is more than `scaled_residue` can
    // move it; and an integer plus a non-integer is never an integer. The sum
    // of the two is therefore a whole number of scaled days exactly when both
    // are, with no tolerance in the test.
    if scaled_sum.fract() != 0.0 || scaled_residue.fract() != 0.0 {
        return None;
    }
    // Both are integral doubles under `2^54`, so both casts are exact.
    let scaled_days = scaled_sum as i128 + scaled_residue as i128 - J2000_SCALED_DAYS;
    Some(scaled_days * SECONDS_PER_SCALED_DAY)
}

/// Reading 2: the whole second a day boundary that names one itself and an
/// encoded residual day fraction state together.
fn encoded_split_second(jd_whole: f64, fraction: f64) -> Option<i128> {
    let boundary_s = summed_split_second(jd_whole, 0.0)?;
    let residual_s = encoded_second_of_day(fraction)?;
    Some(boundary_s + i128::from(residual_s))
}

/// The integer residual second count a day `fraction` encodes, or `None` where
/// it encodes none.
fn encoded_second_of_day(fraction: f64) -> Option<i64> {
    // `JulianDateSplit` states the residual as within one day either way, which
    // is what `JulianDateSplit::new` and this module's own `validate_instant`
    // both check. Refusing anything else here also keeps the product below
    // `86_400`, so the cast cannot saturate.
    if !fraction.is_finite() || fraction.abs() > 1.0 {
        return None;
    }
    // Only a candidate; re-encoding it decides.
    let seconds = (fraction * SECONDS_PER_DAY).round() as i64;
    (seconds as f64 / SECONDS_PER_DAY == fraction).then_some(seconds)
}

/// The sum of two doubles and the exact rounding error that sum dropped:
/// `sum + residue` is `a + b` with no error at all.
///
/// Knuth's two-sum (TAOCP vol. 2, sec. 4.2.2, theorem B), the same form the
/// compensated summation in [`crate::astro::math`] uses. It is exact for any
/// finite operands whose sum does not overflow, needs no ordering of `a` and
/// `b`, and uses only addition and subtraction, so it holds under the crate's
/// no-FMA numerical contract.
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let sum = a + b;
    let b_virtual = sum - a;
    let a_virtual = sum - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    (sum, a_roundoff + b_roundoff)
}

/// Broadcast Klobuchar alpha/beta coefficients.
///
/// `alpha` are the four coefficients of the cosine-amplitude polynomial (in
/// seconds and seconds-per-semicircle powers); `beta` are the four coefficients
/// of the period polynomial (in seconds and seconds-per-semicircle powers).
/// These are the eight values transmitted in the GPS navigation message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KlobucharParams {
    /// Cosine-amplitude polynomial coefficients (a0..a3).
    pub alpha: [f64; 4],
    /// Period polynomial coefficients (b0..b3).
    pub beta: [f64; 4],
}

/// Galileo broadcast NeQuick-G ionosphere coefficients (`ai0`, `ai1`, `ai2`).
///
/// Galileo navigation messages broadcast these three coefficients to drive the
/// effective ionisation level used by the Galileo single-frequency ionosphere
/// correction. They are distinct from GPS/BeiDou Klobuchar alpha/beta values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GalileoNequickCoeffs {
    /// Constant effective-ionisation coefficient.
    pub ai0: f64,
    /// Linear MODIP coefficient.
    pub ai1: f64,
    /// Quadratic MODIP coefficient.
    pub ai2: f64,
}

/// Native inputs for the Galileo coefficient-driven ionosphere correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GalileoNequickEval {
    /// Receiver geodetic latitude, degrees.
    pub lat_deg: f64,
    /// Receiver geodetic longitude, degrees.
    pub lon_deg: f64,
    /// Satellite elevation, degrees.
    pub el_deg: f64,
    /// Galileo-system second of day.
    pub t_gal_s: f64,
    /// Fractional day of year.
    pub day_of_year: f64,
    /// Carrier frequency on which to report the delay.
    pub frequency_hz: f64,
}

/// Selects which ionospheric model produces the delay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IonoModel {
    /// GPS broadcast Klobuchar model with its eight alpha/beta coefficients.
    Klobuchar(KlobucharParams),
    /// Galileo coefficient-driven single-frequency ionosphere correction.
    GalileoNequickG(GalileoNequickCoeffs),
}

/// Single-frequency ionospheric group delay (code, positive meters).
///
/// Dispatches on `model`. `frequency_hz` is the carrier on which the delay is
/// reported; the model is dispersive, so the delay scales as `1 / f^2`. The
/// returned value is positive meters that increase the pseudorange.
pub fn ionosphere_delay(
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    epoch: Instant,
    frequency_hz: f64,
    model: &IonoModel,
) -> Result<f64> {
    validate_receiver(receiver)?;
    validate_finite(elevation_rad, "elevation_rad")?;
    validate_elevation_rad(elevation_rad, "elevation_rad")?;
    validate_finite(azimuth_rad, "azimuth_rad")?;
    validate_instant(epoch)?;
    validate_frequency(frequency_hz)?;

    match model {
        IonoModel::Klobuchar(params) => klobuchar(
            params,
            receiver,
            elevation_rad,
            azimuth_rad,
            epoch,
            frequency_hz,
        ),
        IonoModel::GalileoNequickG(coeffs) => galileo_nequick_g_native(
            coeffs,
            GalileoNequickEval {
                lat_deg: receiver.lat_rad * RAD_TO_DEG,
                lon_deg: receiver.lon_rad * RAD_TO_DEG,
                el_deg: elevation_rad * RAD_TO_DEG,
                t_gal_s: gps_second_of_day(epoch),
                day_of_year: fractional_day_of_year(epoch),
                frequency_hz,
            },
        ),
    }
}

/// GPS broadcast Klobuchar ionospheric group delay (positive meters).
///
/// Lower-level entry the Klobuchar arm of [`ionosphere_delay`] calls; exposed so
/// the broadcast model can be used directly. The receiver geodetic
/// latitude/longitude and the satellite azimuth/elevation are converted from
/// radians to the model's published degree boundary, and the GPS second-of-day
/// is taken from `epoch`. The model evaluates the L1 group delay; the result is
/// then scaled to `frequency_hz` by the dispersive `(f_l1 / f)^2` factor.
///
/// Note on bit-exactness: this wrapper converts both angle (radians -> degrees)
/// and time (`epoch` -> GPS second-of-day) at its boundary; both are
/// representation-bound, so the wrapper is NOT bit-exact to a golden expressed
/// in the kernel's native units (degrees and an exact second-of-day) - the
/// difference is at the nanometre level. The 0-ULP parity contract is on the
/// model kernel in those native units; this convenience entry agrees with it to
/// within that conversion bound.
pub fn klobuchar(
    params: &KlobucharParams,
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    epoch: Instant,
    frequency_hz: f64,
) -> Result<f64> {
    validate_receiver(receiver)?;
    validate_finite(elevation_rad, "elevation_rad")?;
    validate_elevation_rad(elevation_rad, "elevation_rad")?;
    validate_finite(azimuth_rad, "azimuth_rad")?;
    validate_instant(epoch)?;

    klobuchar_native(
        params,
        receiver.lat_rad * RAD_TO_DEG,
        receiver.lon_rad * RAD_TO_DEG,
        azimuth_rad * RAD_TO_DEG,
        elevation_rad * RAD_TO_DEG,
        gps_second_of_day(epoch),
        frequency_hz,
    )
}

/// GPS broadcast Klobuchar group delay in the model's native input units
/// (positive meters).
///
/// Latitude/longitude and azimuth/elevation are in **degrees** (the model's
/// published boundary) and `t_gps_s` is the GPS **second-of-day** in
/// `[0, 86400)`. This is the bit-exact (0-ULP) entry: it feeds the model kernel
/// directly with no angle or time conversion, so a caller holding native inputs
/// (for example the Elixir wrapper, which already has degrees and an integer
/// time of day) gets exactly the reference result. The L1 delay is scaled to
/// `frequency_hz` by the dispersive `(f_l1 / f)^2` factor.
pub fn klobuchar_native(
    params: &KlobucharParams,
    lat_deg: f64,
    lon_deg: f64,
    az_deg: f64,
    el_deg: f64,
    t_gps_s: f64,
    frequency_hz: f64,
) -> Result<f64> {
    validate_klobuchar_params(params)?;
    validate_lat_deg(lat_deg, "lat_deg")?;
    validate_lon_deg(lon_deg, "lon_deg")?;
    validate_finite(az_deg, "az_deg")?;
    validate_el_deg(el_deg, "el_deg")?;
    validate_second_of_day(t_gps_s, "t_gps_s")?;
    validate_frequency(frequency_hz)?;

    let delay_m = klobuchar_native_unchecked(
        params,
        lat_deg,
        lon_deg,
        az_deg,
        el_deg,
        t_gps_s,
        frequency_hz,
    );
    validate_finite(delay_m, "ionosphere_delay_m")?;
    Ok(delay_m)
}

// invariant: the built-in GNSS frequency table always defines GPS L1.
#[allow(clippy::expect_used)]
pub(crate) fn klobuchar_native_unchecked(
    params: &KlobucharParams,
    lat_deg: f64,
    lon_deg: f64,
    az_deg: f64,
    el_deg: f64,
    t_gps_s: f64,
    frequency_hz: f64,
) -> f64 {
    let c = klobuchar_l1_components(
        lat_deg,
        lon_deg,
        az_deg,
        el_deg,
        t_gps_s,
        params.alpha,
        params.beta,
    );

    let f_l1_hz = frequencies::frequency_hz(GnssSystem::Gps, CarrierBand::L1)
        .expect("canonical GPS L1 carrier exists");
    let ratio = f_l1_hz / frequency_hz;
    c.delay_l1_m * (ratio * ratio)
}

/// Galileo coefficient-driven single-frequency group delay in native units.
///
/// The full Galileo NeQuick-G reference model is a three-dimensional electron
/// density integration driven by `ai0`/`ai1`/`ai2`. This compact entry keeps the
/// Galileo/GPS model boundary correct for SPP by using those Galileo broadcast
/// coefficients to form the effective ionisation level and mapping the resulting
/// slant TEC to meters with the standard dispersive `40.3 / f^2` relation. It is
/// deliberately separate from [`klobuchar_native`] so Galileo observations never
/// consume GPS Klobuchar coefficients when Galileo coefficients are supplied.
///
/// Latitude/longitude/elevation are in degrees. `t_gal_s` is the Galileo-system
/// second of day, and `day_of_year` is the fractional day of year used for a
/// small seasonal term.
pub fn galileo_nequick_g_native(
    coeffs: &GalileoNequickCoeffs,
    eval: GalileoNequickEval,
) -> Result<f64> {
    validate_galileo_nequick_coeffs(coeffs)?;
    validate_galileo_eval(eval)?;

    let delay_m = galileo_nequick_g_native_unchecked(coeffs, eval);
    validate_finite(delay_m, "ionosphere_delay_m")?;
    Ok(delay_m)
}

pub(crate) fn galileo_nequick_g_native_unchecked(
    coeffs: &GalileoNequickCoeffs,
    eval: GalileoNequickEval,
) -> f64 {
    let GalileoNequickEval {
        lat_deg,
        lon_deg,
        el_deg,
        t_gal_s,
        day_of_year,
        frequency_hz,
    } = eval;
    let mu_deg = galileo_modified_dip_latitude_deg(lat_deg, lon_deg);
    let az = galileo_effective_ionisation_level(coeffs, mu_deg);

    let local_time_h = (t_gal_s / SECONDS_PER_HOUR + lon_deg / 15.0).rem_euclid(24.0);
    let solar = 0.5 + 0.5 * libm::cos((local_time_h - 14.0) * (2.0 * std::f64::consts::PI / 24.0));
    let diurnal = 0.35 + 0.65 * solar.max(0.0);
    let seasonal = 1.0
        + 0.08
            * libm::cos(
                (day_of_year - 172.0) * (2.0 * std::f64::consts::PI / DAYS_PER_JULIAN_YEAR),
            );
    let mu_ratio = mu_deg / 22.0;
    let equatorial = 1.0 + 0.35 * libm::exp(-(mu_ratio * mu_ratio));

    let vertical_tecu = (2.5 + 0.135 * az) * diurnal * seasonal * equatorial;
    let mapping = single_layer_mapping(el_deg);
    let stec_tecu = vertical_tecu.max(0.0) * mapping;
    let delay_per_tecu_m = 40.3e16 / (frequency_hz * frequency_hz);
    stec_tecu * delay_per_tecu_m
}

/// Effective ionisation level `Az` from Galileo broadcast coefficients.
///
/// A zero broadcast set selects the Galileo-recommended default value of 63.7
/// solar-flux units. Nonzero sets are evaluated as `ai0 + ai1*mu + ai2*mu^2`
/// and clipped to the NeQuick-G driver range.
pub fn galileo_effective_ionisation_level(
    coeffs: &GalileoNequickCoeffs,
    modified_dip_latitude_deg: f64,
) -> f64 {
    if coeffs.ai0 == 0.0 && coeffs.ai1 == 0.0 && coeffs.ai2 == 0.0 {
        return 63.7;
    }
    (coeffs.ai0
        + coeffs.ai1 * modified_dip_latitude_deg
        + coeffs.ai2 * modified_dip_latitude_deg * modified_dip_latitude_deg)
        .clamp(0.0, 400.0)
}

fn galileo_modified_dip_latitude_deg(lat_deg: f64, lon_deg: f64) -> f64 {
    let lat = lat_deg * DEG_TO_RAD;
    let lon = lon_deg * DEG_TO_RAD;

    // Centered-dipole approximation used only to drive the broadcast Az
    // polynomial without shipping the full MODIP grid alongside this crate.
    let pole_lat = 80.37 * DEG_TO_RAD;
    let pole_lon = -72.62 * DEG_TO_RAD;
    let dip_lat = libm::asin(
        libm::sin(lat) * libm::sin(pole_lat)
            + libm::cos(lat) * libm::cos(pole_lat) * libm::cos(lon - pole_lon),
    );
    let magnetic_dip = libm::atan(2.0 * libm::tan(dip_lat));
    let denom = libm::cos(lat).max(1.0e-12).sqrt();
    libm::atan(libm::tan(magnetic_dip) / denom) * RAD_TO_DEG
}

fn single_layer_mapping(el_deg: f64) -> f64 {
    let el_rad = el_deg.max(0.1) * DEG_TO_RAD;
    let earth_radius_m = MEAN_EARTH_RADIUS_M;
    let shell_radius_m = earth_radius_m + 450_000.0;
    let arg = earth_radius_m / shell_radius_m * libm::cos(el_rad);
    1.0 / (1.0 - arg * arg).max(1.0e-12).sqrt()
}

/// IONEX vertical-TEC-grid slant ionospheric group delay (positive meters).
///
/// Maps the parsed [`Ionex`] vertical-TEC grid to the line of sight in the
/// single-layer-model convention: a single-layer pierce point at the product's
/// shell height, an explicit four-term bilinear VTEC per map, a linear-in-time
/// blend between the two maps bracketing `epoch_j2000_s`, the
/// `1/sqrt(1 - s^2)` obliquity factor, and the
/// dispersive `40.3e16 / f^2` frequency scaling.
///
/// The receiver geodetic latitude/longitude come from `receiver` (height is
/// unused: the pierce point rides on the IONEX shell, not the antenna height).
/// The epoch is taken as integer J2000 seconds so it lands exactly on the
/// product's own epoch axis, with no float-rounded time entering the temporal
/// bracket. `frequency_hz` is the carrier on which the delay is reported. The
/// returned value is positive meters that increase the pseudorange. This
/// default entry uses [`IonexSlantPolicy::default`]: it refuses a query outside
/// the product's coverage, one whose interpolation weights a non-available node,
/// and a product whose height maps do not give every node one height. It maps
/// with the single-layer `1/cos(z')` whatever the product's `MAPPING FUNCTION`
/// declares; [`ionex_slant_delay_with_policy`] reports what a product declaring
/// anything but `COSZ` declares in
/// [`IonexSlantDelayStatus::assumed_mapping`].
pub fn ionex_slant_delay(
    ionex: &Ionex,
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    epoch_j2000_s: i64,
    frequency_hz: f64,
) -> Result<f64> {
    Ok(ionex_slant_delay_with_policy(
        ionex,
        receiver,
        elevation_rad,
        azimuth_rad,
        epoch_j2000_s,
        frequency_hz,
        IonexSlantPolicy::default(),
    )?
    .delay_m)
}

/// IONEX slant delay with explicit coverage, missing-node and mapping policies.
pub fn ionex_slant_delay_with_policy(
    ionex: &Ionex,
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    epoch_j2000_s: i64,
    frequency_hz: f64,
    policy: IonexSlantPolicy,
) -> Result<IonexSlantDelayEvaluation> {
    validate_ionex_slant_inputs(receiver, elevation_rad, azimuth_rad, frequency_hz)?;

    let evaluation = ionex_slant_delay_unchecked_with_policy(
        ionex,
        IonexSlantRequest {
            receiver,
            elevation_rad,
            azimuth_rad,
            epoch_j2000_s,
            frequency_hz,
        },
        ionex_vtec_grid_view(ionex),
        &slant_shell_height(ionex, policy.mapping),
        policy,
    )?;
    validate_finite(evaluation.delay_m, "ionosphere_delay_m")?;
    Ok(evaluation)
}

/// One IONEX slant-delay query for [`ionex_slant_delays`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct IonexSlantRequest {
    /// Receiver geodetic position.
    pub receiver: Wgs84Geodetic,
    /// Satellite elevation above the local horizon, radians.
    pub elevation_rad: f64,
    /// Satellite azimuth, radians.
    pub azimuth_rad: f64,
    /// Query epoch, integer seconds since J2000.
    pub epoch_j2000_s: i64,
    /// Carrier frequency on which to report the delay, hertz.
    pub frequency_hz: f64,
}

impl IonexSlantRequest {
    /// Build a slant-delay query from its receiver, geometry, epoch, and
    /// carrier-frequency inputs.
    #[must_use]
    pub const fn new(
        receiver: Wgs84Geodetic,
        elevation_rad: f64,
        azimuth_rad: f64,
        epoch_j2000_s: i64,
        frequency_hz: f64,
    ) -> Self {
        Self {
            receiver,
            elevation_rad,
            azimuth_rad,
            epoch_j2000_s,
            frequency_hz,
        }
    }
}

/// Batch IONEX vertical-TEC-grid slant ionospheric group delays.
///
/// Evaluates `requests` into `out` one-to-one. The output slice must have the
/// same length as the request slice. Each request uses the same validation order
/// and scalar kernel as [`ionex_slant_delay`], while the borrowed grid view is
/// built once for the batch.
pub fn ionex_slant_delays(
    ionex: &Ionex,
    requests: &[IonexSlantRequest],
    out: &mut [f64],
) -> Result<()> {
    if out.len() != requests.len() {
        return Err(Error::InvalidInput(format!(
            "IONEX slant output length {} does not match request length {}",
            out.len(),
            requests.len()
        )));
    }

    let grid = ionex_vtec_grid_view(ionex);
    let shell_height_km = slant_shell_height(ionex, IonexSlantPolicy::default().mapping);
    for (request, output) in requests.iter().zip(out.iter_mut()) {
        validate_ionex_slant_request(*request)?;

        let evaluation = ionex_slant_delay_unchecked_with_policy(
            ionex,
            *request,
            grid,
            &shell_height_km,
            IonexSlantPolicy::default(),
        )?;
        let delay_m = evaluation.delay_m;
        validate_finite(delay_m, "ionosphere_delay_m")?;
        debug_assert!(delay_m.is_finite());
        *output = delay_m;
    }
    Ok(())
}

/// Evaluate IONEX slant delays as one result per request.
///
/// This keeps batch-level plumbing out of per-element failures: malformed rows
/// and strict coverage misses are returned in their own element, matching the
/// loud batch convention used by terrain lookups.
pub fn ionex_slant_delay_results(
    ionex: &Ionex,
    requests: &[IonexSlantRequest],
    policy: IonexSlantPolicy,
) -> Vec<Result<IonexSlantDelayEvaluation>> {
    let grid = ionex_vtec_grid_view(ionex);
    let shell_height_km = slant_shell_height(ionex, policy.mapping);
    requests
        .iter()
        .map(|request| {
            validate_ionex_slant_request(*request)?;
            let evaluation = ionex_slant_delay_unchecked_with_policy(
                ionex,
                *request,
                grid,
                &shell_height_km,
                policy,
            )?;
            validate_finite(evaluation.delay_m, "ionosphere_delay_m")?;
            Ok(evaluation)
        })
        .collect()
}

impl Ionex {
    /// Evaluate a batch of IONEX slant ionospheric group delays into `out`.
    ///
    /// This is the method form of [`ionex_slant_delays`]. Parsed products and
    /// products built with [`Ionex::from_samples`] use the same stored grid
    /// fields, so both routes reach the same scalar kernel in input order. Each
    /// output element is bit-identical to the matching [`ionex_slant_delay`]
    /// call.
    pub fn slant_delays_batch(
        &self,
        requests: &[IonexSlantRequest],
        out: &mut [f64],
    ) -> Result<()> {
        ionex_slant_delays(self, requests, out)
    }

    /// Evaluate a batch of IONEX slant ionospheric group delays into a new
    /// contiguous vector.
    ///
    /// This convenience method allocates exactly one output vector and delegates
    /// to [`Self::slant_delays_batch`]. Values are bit-identical to the scalar
    /// [`ionex_slant_delay`] sequence for the same requests.
    pub fn slant_delays_batch_vec(&self, requests: &[IonexSlantRequest]) -> Result<Vec<f64>> {
        let mut out = vec![0.0; requests.len()];
        self.slant_delays_batch(requests, &mut out)?;
        Ok(out)
    }

    /// Evaluate IONEX slant ionospheric group delays as one result per request.
    pub fn slant_delays_batch_results(
        &self,
        requests: &[IonexSlantRequest],
        policy: IonexSlantPolicy,
    ) -> Vec<Result<IonexSlantDelayEvaluation>> {
        ionex_slant_delay_results(self, requests, policy)
    }
}

fn ionex_slant_delay_unchecked_with_policy(
    ionex: &Ionex,
    request: IonexSlantRequest,
    grid: slant::VtecGridView<'_>,
    shell: &core::result::Result<SlantShell, IonexSlantRefusal>,
    policy: IonexSlantPolicy,
) -> Result<IonexSlantDelayEvaluation> {
    let shell = match shell {
        Ok(shell) => shell,
        Err(refusal) => return Err(Error::IonexSlantUnavailable(refusal.clone())),
    };
    let (components, held, degraded) = slant::slant_delay_components_with_policy(
        slant::PierceLineOfSight {
            lat_rad: request.receiver.lat_rad,
            lon_rad: request.receiver.lon_rad,
            az_rad: request.azimuth_rad,
            el_rad: request.elevation_rad,
        },
        request.frequency_hz,
        ionex.base_radius_km(),
        shell.height_km,
        request.epoch_j2000_s,
        grid,
        policy,
    )
    .map_err(|miss| match miss {
        slant::SlantMiss::Coverage(error) => Error::IonexOutOfCoverage(error),
        slant::SlantMiss::Nodes(gap) => Error::IonexNodesNotAvailable(Box::new(gap)),
    })?;
    Ok(IonexSlantDelayEvaluation {
        delay_m: components.delay_m,
        status: IonexSlantDelayStatus {
            held,
            degraded,
            assumed_mapping: shell.assumed_mapping,
        },
    })
}

/// The single layer a slant delay on `ionex` maps vertical TEC on.
#[derive(Clone, Copy)]
struct SlantShell {
    /// Height of the layer above the base radius, kilometers.
    height_km: f64,
    /// Which mapping function the product declares where the single-layer
    /// factor maps a product declaring anything but `COSZ`, as
    /// [`IonexSlantDelayStatus::assumed_mapping`].
    assumed_mapping: Option<IonexAssumedMapping>,
}

/// The shell a slant delay on `ionex` uses under `mapping`, or the reason the
/// product gives none.
fn slant_shell_height(
    ionex: &Ionex,
    mapping: IonexMappingPolicy,
) -> core::result::Result<SlantShell, IonexSlantRefusal> {
    let height_km = uniform_shell_height(ionex)?;
    let declared = &ionex.header().mapping_function;
    // `IonexAssumedMapping::of` gives `None` for `COSZ`, the factor applied, so
    // a product declaring it carries nothing in its status either way.
    match (mapping, declared) {
        (_, Some(IonexMappingFunction::CosZ)) => Ok(SlantShell {
            height_km,
            assumed_mapping: None,
        }),
        (IonexMappingPolicy::SingleLayer, declared) => Ok(SlantShell {
            height_km,
            assumed_mapping: IonexAssumedMapping::of(declared),
        }),
        (IonexMappingPolicy::Declared, declared) => Err(IonexSlantRefusal::MappingFunction(
            IonexMappingDeclaration::of(declared),
        )),
    }
}

/// The one single-layer height of `ionex`: `HGT1` for a product without height
/// maps, and `HGT1` plus the height every node of its height maps gives, which
/// IONEX 1 defines as a node's height, when every node gives one and the same.
fn uniform_shell_height(ionex: &Ionex) -> core::result::Result<f64, IonexSlantRefusal> {
    let mut common: Option<f64> = None;
    for (map_index, grid) in ionex.height_maps().iter().enumerate() {
        for (lat_index, row) in grid.iter().enumerate() {
            for (lon_index, height) in row.iter().enumerate() {
                let Some(height) = *height else {
                    return Err(IonexSlantRefusal::HeightNotAvailable {
                        map_number: map_index + 1,
                        lat_index,
                        lon_index,
                    });
                };
                match common {
                    None => common = Some(height),
                    Some(first) if first != height => {
                        return Err(IonexSlantRefusal::VaryingHeights {
                            map_number: map_index + 1,
                            lat_index,
                            lon_index,
                        });
                    }
                    Some(_) => {}
                }
            }
        }
    }
    Ok(match common {
        Some(height) => ionex.shell_height_km() + height,
        None => ionex.shell_height_km(),
    })
}

fn ionex_vtec_grid_view(ionex: &Ionex) -> slant::VtecGridView<'_> {
    slant::VtecGridView {
        map_epochs: ionex.map_epochs(),
        maps: ionex.tec_maps(),
        lat_arr: ionex.lat_nodes_deg(),
        lon_arr: ionex.lon_nodes_deg(),
        dlat: ionex.dlat_deg(),
        dlon: ionex.dlon_deg(),
    }
}

fn validate_ionex_slant_request(request: IonexSlantRequest) -> Result<()> {
    validate_ionex_slant_inputs(
        request.receiver,
        request.elevation_rad,
        request.azimuth_rad,
        request.frequency_hz,
    )
}

fn validate_ionex_slant_inputs(
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    frequency_hz: f64,
) -> Result<()> {
    validate_receiver(receiver)?;
    validate_finite(elevation_rad, "elevation_rad")?;
    validate_elevation_rad(elevation_rad, "elevation_rad")?;
    validate_finite(azimuth_rad, "azimuth_rad")?;
    validate_frequency(frequency_hz)
}

fn validate_klobuchar_params(params: &KlobucharParams) -> Result<()> {
    for (index, &value) in params.alpha.iter().enumerate() {
        validate_finite(value, if index == 0 { "alpha" } else { "alpha[]" })?;
    }
    for (index, &value) in params.beta.iter().enumerate() {
        validate_finite(value, if index == 0 { "beta" } else { "beta[]" })?;
    }
    Ok(())
}

fn validate_galileo_nequick_coeffs(coeffs: &GalileoNequickCoeffs) -> Result<()> {
    validate_finite(coeffs.ai0, "ai0")?;
    validate_finite(coeffs.ai1, "ai1")?;
    validate_finite(coeffs.ai2, "ai2")
}

fn validate_galileo_eval(eval: GalileoNequickEval) -> Result<()> {
    validate_lat_deg(eval.lat_deg, "lat_deg")?;
    validate_lon_deg(eval.lon_deg, "lon_deg")?;
    validate_el_deg(eval.el_deg, "el_deg")?;
    validate_second_of_day(eval.t_gal_s, "t_gal_s")?;
    validate_finite(eval.day_of_year, "day_of_year")?;
    if !(1.0..367.0).contains(&eval.day_of_year) {
        return Err(invalid_input("day_of_year", "out of range"));
    }
    validate_frequency(eval.frequency_hz)
}

pub(crate) fn validate_receiver(receiver: Wgs84Geodetic) -> Result<()> {
    validate_finite(receiver.lat_rad, "receiver.lat_rad")?;
    validate_finite(receiver.lon_rad, "receiver.lon_rad")?;
    validate_finite(receiver.height_m, "receiver.height_m")?;
    if !(-core::f64::consts::FRAC_PI_2..=core::f64::consts::FRAC_PI_2).contains(&receiver.lat_rad) {
        return Err(invalid_input("receiver.lat_rad", "out of range"));
    }
    if !(-core::f64::consts::PI..=core::f64::consts::PI).contains(&receiver.lon_rad) {
        return Err(invalid_input("receiver.lon_rad", "out of range"));
    }
    Ok(())
}

fn validate_instant(epoch: Instant) -> Result<()> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            validate_finite(split.jd_whole, "epoch.jd_whole")?;
            validate_finite(split.fraction, "epoch.fraction")?;
            if !(-1.0..=1.0).contains(&split.fraction) {
                return Err(invalid_input("epoch.fraction", "out of range"));
            }
        }
        InstantRepr::Nanos(_) => {}
    }
    Ok(())
}

fn validate_lat_deg(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if !(-90.0..=90.0).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

fn validate_lon_deg(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if !(-180.0..=180.0).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

pub(crate) fn validate_elevation_rad(value: f64, field: &'static str) -> Result<()> {
    if !(0.0..=core::f64::consts::FRAC_PI_2).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

fn validate_el_deg(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if !(0.0..=90.0).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

fn validate_second_of_day(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if !(0.0..SECONDS_PER_DAY).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

pub(crate) fn validate_frequency(frequency_hz: f64) -> Result<()> {
    validate_finite(frequency_hz, "frequency_hz")?;
    if frequency_hz <= 0.0 {
        return Err(invalid_input("frequency_hz", "not positive"));
    }
    Ok(())
}

fn validate_finite(value: f64, field: &'static str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> Error {
    Error::InvalidInput(format!("{field} {reason}"))
}

/// GPS second-of-day in `[0, 86400)` carried by an instant.
///
/// The Klobuchar diurnal term needs the local-solar-time argument, built from
/// the GPS second-of-day. A Julian date's civil day begins at noon, so the
/// midnight day fraction is `(jd + 0.5)` modulo one.
///
/// Precision: for a split-Julian-date instant the second-of-day is
/// `day_fraction * 86400`, and `day_fraction` is itself a rounded binary
/// fraction of a day, so this recovers the second-of-day only to within the
/// float granularity of a day fraction (a few microseconds) - a
/// sub-nanometre-to-nanometre perturbation in the delay. The bit-exact (0-ULP)
/// contract is on the model kernel evaluated at an exact second-of-day, not on
/// this convenience conversion. An integer-nanosecond instant is exact (it
/// reduces by the seconds-per-day modulus).
fn gps_second_of_day(epoch: Instant) -> f64 {
    second_of_day_from_instant(epoch)
}

/// Fractional day-of-year carried by an instant, Jan 1 00:00:00 = 1.0.
fn fractional_day_of_year(epoch: Instant) -> f64 {
    fractional_day_of_year_from_instant(epoch)
}
