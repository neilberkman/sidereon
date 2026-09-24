//! Single-layer-model slant ionospheric delay from a vertical-TEC grid.
//!
//! Converts an IONEX vertical-TEC grid into a slant ionospheric group delay in
//! meters. The receiver geodetic latitude/longitude, the satellite
//! azimuth/elevation, the shell height, and the carrier frequency define a
//! single-layer pierce point; the vertical TEC there is read from the grid by an
//! explicit four-term bilinear interpolation per map and a linear-in-time blend
//! between the two bracketing maps; the obliquity factor maps vertical to slant
//! TEC; and the dispersive frequency scaling turns slant TEC into meters.
//!
//! A node the product gives as non-available has no value to interpolate. A
//! node or map is weighted when its bilinear or temporal weight is nonzero.
//! When a weighted node is non-available, the strict policy gives no value and
//! names the map, the cell and the missing nodes; the renormalizing policy
//! interpolates from the weighted nodes and maps that hold values, their weights
//! renormalized to sum to one, and reports the missing nodes with the value. A
//! node or map without weight contributes nothing, so a query on an available
//! node, or at the epoch of a map, uses no neighbour.
//!
//! The delay returned is a group delay and is positive: it increases the
//! measured pseudorange (the carrier-phase advance is the negation of this
//! value).
//!
//! The grid is stored in the order the file writes each axis, which the sign of
//! its step gives; the bracketing and the
//! signed-step fractional offsets follow that ordering directly. There is no
//! fused multiply-add anywhere: every product and sum is a plain operator, so
//! the operation tree is identical to the reference recipe and the result is
//! bit-stable.

use super::{
    exact_j2000_second, IonexCoverageError, IonexCoveragePolicy, IonexMissingNodePolicy,
    IonexMissingNodes, IonexNodeGap, IonexSlantPolicy, UtcQueryTime,
};
use crate::astro::time::model::Instant;

/// Ionospheric frequency-scaling constant `40.3 * 1e16`.
///
/// The dispersive ionospheric delay is `40.3 / f^2` per electron column density;
/// the `1e16` factor lets the slant TEC be carried in TECU (`1e16`
/// electrons/m^2) rather than electrons/m^2.
pub(crate) const K_IONO: f64 = 40.3e16;

/// Single-layer pierce-point geometry.
///
/// `s = Re/(Re+H) * cos(E)` is the shell-scaled cosine, formed once and reused
/// by the earth-central angle and the obliquity factor. The pierce-point
/// latitude and longitude are returned in degrees for the grid lookup.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PiercePoint {
    /// Shell-scaled cosine `Re/(Re+H) * cos(E)` (dimensionless).
    pub s: f64,
    /// Earth-central angle from receiver to pierce point (radians).
    pub psi: f64,
    /// Pierce-point geodetic latitude (degrees).
    pub phi_ipp_deg: f64,
    /// Pierce-point geodetic longitude (degrees), before grid normalization.
    pub lambda_ipp_deg: f64,
}

/// Compute the single-layer pierce-point geometry.
///
/// Inputs are the receiver geodetic latitude/longitude and the satellite
/// azimuth/elevation in radians, plus the base radius and shell height in
/// kilometers. The earth-central angle uses the full spherical-trig `asin` form
/// (not a small-angle approximation).
pub(crate) fn pierce_point(
    lat_rad: f64,
    lon_rad: f64,
    az_rad: f64,
    el_rad: f64,
    re_km: f64,
    h_km: f64,
) -> PiercePoint {
    use crate::constants::RAD_TO_DEG;
    use core::f64::consts::PI;

    let s = re_km / (re_km + h_km) * libm::cos(el_rad);

    // Earth-central angle from receiver to pierce point.
    let psi = PI / 2.0 - el_rad - libm::asin(sine_argument(s));

    let phi_ipp = libm::asin(sine_argument(
        libm::sin(lat_rad) * libm::cos(psi)
            + libm::cos(lat_rad) * libm::sin(psi) * libm::cos(az_rad),
    ));
    // A pierce point at a pole lies on every meridian, and the quotient that
    // names its longitude divides by the cosine of its latitude, which is that
    // close to zero there. `cos` never returns zero for a finite argument -- at
    // `pi/2` it is about 6.1e-17 -- so the test is a tolerance, not equality:
    // dividing by that value instead sends the quotient past the domain, and
    // the clamp below would turn it into the receiver's meridian a quarter turn
    // off. The receiver's meridian is one of the meridians through the pole,
    // and the grid rows meet there, so the value read does not depend on which
    // is taken.
    let cos_phi_ipp = libm::cos(phi_ipp);
    let lambda_ipp = if cos_phi_ipp.abs() <= POLE_COS_TOLERANCE {
        lon_rad
    } else {
        lon_rad
            + libm::asin(sine_argument(
                libm::sin(psi) * libm::sin(az_rad) / cos_phi_ipp,
            ))
    };

    PiercePoint {
        s,
        psi,
        phi_ipp_deg: phi_ipp * RAD_TO_DEG,
        lambda_ipp_deg: lambda_ipp * RAD_TO_DEG,
    }
}

/// How near the cosine of a pierce-point latitude comes to zero before the
/// pierce point counts as being at a pole.
///
/// `1e-12` is a latitude within about `1e-12` radians of `pi/2`, which is a few
/// micrometres of ground distance; every meridian meets at the pole, so no
/// longitude is more right than another there.
const POLE_COS_TOLERANCE: f64 = 1.0e-12;

/// An argument for `asin`, held inside `[-1, 1]`.
///
/// The spherical-trig quotients reach `1` at the limit and can round just past
/// it, or past `-1`, where `asin` gives NaN. Two explicit comparisons, not a
/// clamp call, so a value already inside the interval is returned unchanged and
/// the operation order of the reference recipe is kept.
#[allow(clippy::manual_clamp)]
fn sine_argument(value: f64) -> f64 {
    if value > 1.0 {
        1.0
    } else if value < -1.0 {
        -1.0
    } else {
        value
    }
}

/// Bring a longitude into the grid's usable longitude range.
///
/// Full 360-degree grids wrap by `+-360` steps. Regional grids cover only an
/// interval on the longitude circle, so out-of-coverage pierce points are held
/// at the nearest interval edge instead of extrapolating past the edge cell.
fn normalize_lon_deg_with_coverage(
    mut lon_deg: f64,
    lon1: f64,
    lon2: f64,
    dlon_abs: f64,
) -> (f64, Option<IonexCoverageError>) {
    if !lon_deg.is_finite() {
        return (lon_deg, None);
    }

    // A grid whose ends are a full turn apart names the seam twice, so every
    // longitude lies between them.
    if lon2 - lon1 >= 360.0 {
        while lon_deg < lon1 {
            lon_deg += 360.0;
        }
        while lon_deg > lon2 {
            lon_deg -= 360.0;
        }
        return (lon_deg, None);
    }

    // A grid that closes the circle without naming the seam twice, as the
    // 0 to 355 by 5 of IONEX 1's example 1 does, also covers every longitude:
    // the cell between its last node and its first carries the rest of the
    // turn.
    if lon2 - lon1 + dlon_abs >= 360.0 {
        while lon_deg < lon1 {
            lon_deg += 360.0;
        }
        while lon_deg >= lon1 + 360.0 {
            lon_deg -= 360.0;
        }
        return (lon_deg, None);
    }

    let base_turn = ((lon_deg - lon1) / 360.0).floor();
    let mut best_lon = lon_deg;
    let mut best_distance = f64::INFINITY;
    for turn in [base_turn - 1.0, base_turn, base_turn + 1.0] {
        let offset = turn * 360.0;
        let lo = lon1 + offset;
        let hi = lon2 + offset;
        let clamped = if lon_deg < lo {
            lo
        } else if lon_deg > hi {
            hi
        } else {
            lon_deg
        };
        let distance = (lon_deg - clamped).abs();
        if distance < best_distance {
            best_distance = distance;
            best_lon = clamped - offset;
        }
    }
    let coverage = if best_distance == 0.0 {
        None
    } else {
        Some(IonexCoverageError::LongitudeOutOfRange)
    };
    (best_lon, coverage)
}

/// The lowest and highest node of an axis, whichever end the file writes first.
///
/// IONEX 1 gives an axis as its two bounds and a signed step, so a file may run
/// its latitudes north to south or south to north, and its longitudes either
/// way.
fn axis_bounds(nodes: &[f64]) -> (f64, f64) {
    let first = nodes[0];
    let last = nodes[nodes.len() - 1];
    if first <= last {
        (first, last)
    } else {
        (last, first)
    }
}

/// Whether the longitude axis closes the circle without naming the seam twice,
/// so the cell between its last node and its first covers the rest of the turn.
fn closes_circle(lon_arr: &[f64], dlon: f64) -> bool {
    let (lon1, lon2) = axis_bounds(lon_arr);
    let span = lon2 - lon1;
    span < 360.0 && span + dlon.abs() >= 360.0
}

/// Lower bracket index on a longitude axis that closes the circle, where the
/// last node in step order brackets with the first.
fn seam_bracket(value: f64, v1: f64, step: f64, n: usize) -> usize {
    let units = ((value - v1) / step).floor();
    if !units.is_finite() {
        return 0;
    }
    (units as i64).rem_euclid(n as i64) as usize
}

/// Lower bracket index along an axis with signed `step` and `n` nodes.
///
/// Returns `i` such that nodes `i` and `i+1` bracket `value`, clamped so both
/// indices are valid (an edge clamp for an out-of-grid query).
fn bracket(value: f64, v1: f64, step: f64, n: usize) -> usize {
    let idx = ((value - v1) / step) as i64;
    if idx < 0 {
        0
    } else if idx > (n as i64) - 2 {
        n - 2
    } else {
        idx as usize
    }
}

/// One map's bilinear VTEC at a pierce point, with the interpolation weights.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct BilinearVtec {
    /// Interpolated vertical TEC at the pierce point (TECU), when every node
    /// with a nonzero weight holds a value.
    pub vtec: Option<f64>,
    /// Vertical TEC from the nodes with a nonzero weight that hold values,
    /// their weights renormalized to sum to one (TECU), when some such node
    /// holds no value and another does.
    pub renormalized: Option<f64>,
    /// Longitude-direction fractional offset within the cell.
    pub p: f64,
    /// Latitude-direction fractional offset within the cell (signed step).
    pub q: f64,
    /// Latitude index of the cell's first node row.
    pub lat_index: usize,
    /// Longitude index of the cell's first node column.
    pub lon_index: usize,
    /// Longitude index of the cell's other node column, which is `0` on the
    /// cell that closes the circle.
    pub lon_index_next: usize,
    /// Nodes with a nonzero weight that hold no value, in the order `E00`,
    /// `E01`, `E10`, `E11`.
    pub missing: [bool; 4],
}

/// Explicit four-term bilinear VTEC at `(phi_deg, lam_deg)` on one map.
///
/// `vtec_map` is indexed `[i_lat][i_lon]` matching `lat_arr` and `lon_arr`, each
/// in the order its signed step gives. The pierce-point latitude is clamped to
/// the grid edge before bracketing, and so is the longitude unless the grid
/// closes the circle, where the cell between the last node and the first
/// carries the seam. The weighted sum is the explicit four-term form
/// `(1-p)(1-q)E00 + p(1-q)E01 + (1-p)q E10 + p q E11` with `(1-p)`/`(1-q)`
/// formed once.
pub(crate) fn bilinear_vtec(
    vtec_map: &[Vec<Option<f64>>],
    lat_arr: &[f64],
    lon_arr: &[f64],
    dlat: f64,
    dlon: f64,
    phi_deg: f64,
    lam_deg: f64,
) -> BilinearVtec {
    let nlat = lat_arr.len();
    let nlon = lon_arr.len();

    // Clamp the pierce-point latitude to the grid extent, whichever end the
    // file writes first.
    let (lat_lo, lat_hi) = axis_bounds(lat_arr);
    let mut phi = phi_deg;
    if phi > lat_hi {
        phi = lat_hi;
    }
    if phi < lat_lo {
        phi = lat_lo;
    }
    // A grid that closes the circle interpolates across the seam; one that does
    // not holds the query at its nearest edge, as before.
    let wraps = closes_circle(lon_arr, dlon);
    let (lon_lo, lon_hi) = axis_bounds(lon_arr);
    let mut lam = lam_deg;
    if !wraps {
        if lam < lon_lo {
            lam = lon_lo;
        }
        if lam > lon_hi {
            lam = lon_hi;
        }
    }

    let i = bracket(phi, lat_arr[0], dlat, nlat);
    let j = if wraps {
        seam_bracket(lam, lon_arr[0], dlon, nlon)
    } else {
        bracket(lam, lon_arr[0], dlon, nlon)
    };
    // The node after the last one in step order is the first, a turn away.
    let j1 = if j + 1 == nlon { 0 } else { j + 1 };

    let lat0 = lat_arr[i];
    let lon0 = lon_arr[j];

    // Signed-step fractional offsets: both land in [0, 1].
    let q = (phi - lat0) / dlat;
    let mut p = (lam - lon0) / dlon;
    if wraps {
        // A grid that closes the circle has a cell that runs past the turn, and
        // a query can sit any number of turns from the node that opens it. A
        // turn is `360 / |dlon|` cells whichever way the axis runs, so moving
        // by whole turns in the direction that shortens the offset brings the
        // query into the cell, leaving `p` in `[0, 1]`. A query already there,
        // the closing node at exactly 1 included, is left alone.
        let turn = (360.0 / dlon).abs();
        if turn > 0.0 && turn.is_finite() {
            while p < 0.0 {
                p += turn;
            }
            while p > 1.0 {
                p -= turn;
            }
        }
    }

    let nodes = [
        vtec_map[i][j],
        vtec_map[i][j1],
        vtec_map[i + 1][j],
        vtec_map[i + 1][j1],
    ];

    let one_p = 1.0 - p;
    let one_q = 1.0 - q;
    let weights = [one_p * one_q, p * one_q, one_p * q, p * q];
    let missing = [0, 1, 2, 3].map(|k| nodes[k].is_none() && weights[k] != 0.0);
    let (vtec, renormalized) = if missing.contains(&true) {
        let mut weight = 0.0;
        let mut sum = 0.0;
        for (node, node_weight) in nodes.into_iter().zip(weights) {
            if let (Some(value), true) = (node, node_weight != 0.0) {
                weight += node_weight;
                sum += node_weight * value;
            }
        }
        (None, (weight != 0.0).then(|| sum / weight))
    } else {
        // Only a node without weight can hold no value here, and its term is
        // zero whatever it holds.
        let [e00, e01, e10, e11] = nodes.map(|node| node.unwrap_or(0.0));
        let vtec = one_p * one_q * e00 + p * one_q * e01 + one_p * q * e10 + p * q * e11;
        (Some(vtec), None)
    };

    BilinearVtec {
        vtec,
        renormalized,
        p,
        q,
        lat_index: i,
        lon_index: j,
        lon_index_next: j1,
        missing,
    }
}

/// All intermediate quantities of one slant-delay evaluation.
///
/// Carrying every intermediate (not just the final delay) lets the parity test
/// localise any divergence to a single algorithm step rather than only seeing
/// the end result move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SlantComponents {
    /// Shell-scaled cosine `Re/(Re+H) * cos(E)` (dimensionless).
    pub s: f64,
    /// Earth-central angle from receiver to pierce point (radians).
    pub psi: f64,
    /// Pierce-point geodetic latitude (degrees).
    pub phi_ipp_deg: f64,
    /// Pierce-point geodetic longitude before grid normalization (degrees).
    pub lambda_ipp_deg_raw: f64,
    /// Pierce-point geodetic longitude normalized into the grid range (degrees).
    pub lambda_ipp_deg: f64,
    /// Index of the lower bracketing map in the epoch axis.
    pub map_index: usize,
    /// Temporal blend weight in `[0, 1]` toward the upper bracketing map.
    pub w: f64,
    /// Bilinear VTEC on the lower bracketing map (TECU), where it has one.
    pub vtec0: Option<f64>,
    /// Bilinear VTEC on the upper bracketing map (TECU), where it has one.
    pub vtec1: Option<f64>,
    /// Longitude fractional offset on the lower map.
    pub p0: f64,
    /// Latitude fractional offset on the lower map.
    pub q0: f64,
    /// Time-blended vertical TEC at the pierce point (TECU).
    pub vtec: f64,
    /// Obliquity (slant) factor mapping vertical to slant TEC (dimensionless).
    pub m: f64,
    /// Slant TEC (TECU).
    pub stec: f64,
    /// Slant ionospheric group delay (meters).
    pub delay_m: f64,
}

/// Receiver-to-satellite line of sight for the single-layer pierce point: the
/// receiver geodetic latitude/longitude and the satellite azimuth/elevation, all
/// in radians.
pub(crate) struct PierceLineOfSight {
    pub lat_rad: f64,
    pub lon_rad: f64,
    pub az_rad: f64,
    pub el_rad: f64,
}

/// Borrowed view of the IONEX vertical-TEC grid: the per-epoch maps on their
/// instant time axis and the latitude/longitude node arrays with their
/// signed steps. Bundles the six grid quantities the bilinear/temporal
/// interpolation reads so the entry point takes one grid argument.
#[derive(Clone, Copy)]
pub(crate) struct VtecGridView<'a> {
    pub map_epochs: &'a [Instant],
    pub maps: &'a [Vec<Vec<Option<f64>>>],
    pub lat_arr: &'a [f64],
    pub lon_arr: &'a [f64],
    pub dlat: f64,
    pub dlon: f64,
}

/// Why a slant-delay evaluation gives no value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SlantMiss {
    /// The query is outside the product's coverage and the policy is strict.
    Coverage(IonexCoverageError),
    /// A node the interpolation weights holds no value.
    Nodes(IonexNodeGap),
}

/// Full IONEX slant group delay in meters, with all intermediates.
///
/// `maps` is one VTEC grid per epoch in `map_epochs` (canonical instants), each a
/// 2-D array indexed `[i_lat][i_lon]`. The pierce-point VTEC is bilinearly
/// interpolated on the two maps bracketing `epoch_s` and then blended
/// linearly in time, holding the endpoint map outside coverage. The obliquity
/// factor `m = 1/cos(z') = 1/sqrt(1 - s^2)` maps vertical to slant TEC, and the
/// dispersive scaling `(40.3e16 / f^2) * STEC` gives the positive group delay.
#[cfg(all(test, sidereon_repo_tests))]
pub(crate) fn slant_delay_components(
    los: PierceLineOfSight,
    frequency_hz: f64,
    re_km: f64,
    h_km: f64,
    epoch_s: i64,
    grid: VtecGridView,
) -> Result<SlantComponents, IonexNodeGap> {
    slant_delay_components_with_coverage(
        los,
        frequency_hz,
        re_km,
        h_km,
        UtcQueryTime {
            seconds: epoch_s,
            fraction: 0.0,
        },
        grid,
        IonexMissingNodePolicy::Strict,
    )
    .0
    .map(|(components, _)| components)
}

/// One slant-delay evaluation under `policy`: its components, the coverage miss
/// a hold policy held it through, and the non-available nodes a renormalizing
/// policy interpolated around.
pub(crate) fn slant_delay_components_with_policy(
    los: PierceLineOfSight,
    frequency_hz: f64,
    re_km: f64,
    h_km: f64,
    epoch: UtcQueryTime,
    grid: VtecGridView,
    policy: IonexSlantPolicy,
) -> Result<
    (
        SlantComponents,
        Option<IonexCoverageError>,
        Option<IonexNodeGap>,
    ),
    SlantMiss,
> {
    let (evaluation, coverage) = slant_delay_components_with_coverage(
        los,
        frequency_hz,
        re_km,
        h_km,
        epoch,
        grid,
        policy.missing_nodes,
    );
    if let (IonexCoveragePolicy::Strict, Some(error)) = (policy.coverage, coverage) {
        return Err(SlantMiss::Coverage(error));
    }
    let (components, degraded) = evaluation.map_err(SlantMiss::Nodes)?;
    Ok((components, coverage, degraded))
}

/// The components with the non-available nodes interpolated around, or the
/// nodes that left the evaluation without a value.
type NodeEvaluation = Result<(SlantComponents, Option<IonexNodeGap>), IonexNodeGap>;

fn slant_delay_components_with_coverage(
    los: PierceLineOfSight,
    frequency_hz: f64,
    re_km: f64,
    h_km: f64,
    epoch: UtcQueryTime,
    grid: VtecGridView,
    missing_nodes: IonexMissingNodePolicy,
) -> (NodeEvaluation, Option<IonexCoverageError>) {
    let PierceLineOfSight {
        lat_rad,
        lon_rad,
        az_rad,
        el_rad,
    } = los;
    let VtecGridView {
        map_epochs,
        maps,
        lat_arr,
        lon_arr,
        dlat,
        dlon,
    } = grid;
    let geom = pierce_point(lat_rad, lon_rad, az_rad, el_rad, re_km, h_km);
    let s = geom.s;

    let (lon1, lon2) = axis_bounds(lon_arr);
    let (lam_deg, lon_coverage) =
        normalize_lon_deg_with_coverage(geom.lambda_ipp_deg, lon1, lon2, dlon.abs());
    let phi_deg = geom.phi_ipp_deg;
    let (lat_lo, lat_hi) = axis_bounds(lat_arr);
    let lat_coverage = if phi_deg > lat_hi || phi_deg < lat_lo {
        Some(IonexCoverageError::LatitudeOutOfRange)
    } else {
        None
    };

    // Temporal bracket (hold the endpoint map outside coverage). A single-map
    // product has no interval to interpolate across, so it holds that one map
    // (weight 0); the second sample index is held at `ti` so it can never read
    // past the end. Multi-map products keep the original bracketing exactly.
    let nmaps = map_epochs.len();
    let first_epoch_s = map_epoch_j2000_s(map_epochs, 0);
    let last_epoch_s = map_epoch_j2000_s(map_epochs, nmaps - 1);
    // Map epochs are whole seconds, so the query's fraction decides only
    // against the last map, and only when it sits on that map's second.
    let UtcQueryTime {
        seconds: epoch_s,
        fraction: epoch_fraction,
    } = epoch;
    let time_coverage = if epoch_s < first_epoch_s {
        Some(IonexCoverageError::EpochBeforeFirstMap)
    } else if epoch_s > last_epoch_s || (epoch_s == last_epoch_s && epoch_fraction > 0.0) {
        Some(IonexCoverageError::EpochAfterLastMap)
    } else {
        None
    };
    let coverage = time_coverage.or(lat_coverage).or(lon_coverage);
    let (ti, ti1, w) = if nmaps <= 1 {
        (0usize, 0usize, 0.0)
    } else {
        let mut ti = 0usize;
        while ti < nmaps - 2 && epoch_s >= map_epoch_j2000_s(map_epochs, ti + 1) {
            ti += 1;
        }
        let t0 = map_epoch_j2000_s(map_epochs, ti);
        let t1 = map_epoch_j2000_s(map_epochs, ti + 1);
        // Both differences are formed in `i128` and projected once. The epoch
        // axis is a whole-second axis over the entire `i64` range, and a query
        // may sit at the far end of it, so a difference taken in `i64` can
        // overflow and a difference taken between two `f64` projections of the
        // endpoints can collapse: past the 53-bit integers, `t1 as f64 - t0 as
        // f64` is zero for adjacent map times and the numerator rounds onto an
        // endpoint for a query between them. In `i128` neither can happen -
        // `t1 > t0` holds exactly, so the span is at least one second - and for
        // any epoch and bracket inside the 53-bit integers both differences are
        // exact integers either way, so ordinary products give bit-identical
        // weights.
        let span_s = i128::from(t1) - i128::from(t0);
        let offset_s = i128::from(epoch_s) - i128::from(t0);
        // A query on a whole second divides the exact integers as before; a
        // fraction is added to the whole-second offset, which is exact for
        // any offset of the 53-bit integers, and rounded with it once.
        let mut w = if epoch_fraction == 0.0 {
            offset_s as f64 / span_s as f64
        } else {
            (offset_s as f64 + epoch_fraction) / span_s as f64
        };
        // Two explicit comparisons, not a clamp call: this reproduces the
        // reference recipe's operation order and NaN handling exactly so the
        // result is bit-stable.
        #[allow(clippy::manual_clamp)]
        if w < 0.0 {
            w = 0.0;
        }
        if w > 1.0 {
            w = 1.0;
        }
        (ti, ti + 1, w)
    };

    let b0 = bilinear_vtec(&maps[ti], lat_arr, lon_arr, dlat, dlon, phi_deg, lam_deg);
    let b1 = bilinear_vtec(&maps[ti1], lat_arr, lon_arr, dlat, dlon, phi_deg, lam_deg);

    let earlier_weighted = 1.0 - w != 0.0;
    let later_weighted = ti1 != ti && w != 0.0;
    // The epoch axis is indexed from 0; a map is named by its number, from 1.
    let missing_on = |b: &BilinearVtec, map_index: usize, weighted: bool| {
        (weighted && b.vtec.is_none()).then_some(IonexMissingNodes {
            map_number: map_index + 1,
            lat_index: b.lat_index,
            lon_index: b.lon_index,
            lon_index_next: b.lon_index_next,
            missing: b.missing,
        })
    };
    let earlier = missing_on(&b0, ti, earlier_weighted);
    let later = missing_on(&b1, ti1, later_weighted);
    let gap = (earlier.is_some() || later.is_some()).then_some(IonexNodeGap { earlier, later });

    let vtec = match (gap, missing_nodes) {
        // A map without temporal weight contributes nothing, whatever it holds.
        (None, _) => (1.0 - w) * b0.vtec.unwrap_or(0.0) + w * b1.vtec.unwrap_or(0.0),
        (Some(gap), IonexMissingNodePolicy::Strict) => return (Err(gap), coverage),
        (Some(gap), IonexMissingNodePolicy::Renormalize) => {
            let mut weight = 0.0;
            let mut sum = 0.0;
            for (b, map_weight, weighted) in
                [(&b0, 1.0 - w, earlier_weighted), (&b1, w, later_weighted)]
            {
                if let (true, Some(value)) = (weighted, b.vtec.or(b.renormalized)) {
                    weight += map_weight;
                    sum += map_weight * value;
                }
            }
            if weight == 0.0 {
                return (Err(gap), coverage);
            }
            sum / weight
        }
    };

    // Obliquity (slant) factor m(E) = 1 / cos(z') = 1 / sqrt(1 - s^2).
    let m = 1.0 / (1.0 - s * s).sqrt();
    let stec = m * vtec;

    let delay_m = (K_IONO / (frequency_hz * frequency_hz)) * stec;

    let components = SlantComponents {
        s,
        psi: geom.psi,
        phi_ipp_deg: phi_deg,
        lambda_ipp_deg_raw: geom.lambda_ipp_deg,
        lambda_ipp_deg: lam_deg,
        map_index: ti,
        w,
        vtec0: b0.vtec,
        vtec1: b1.vtec,
        p0: b0.p,
        q0: b0.q,
        vtec,
        m,
        stec,
        delay_m,
    };
    (Ok((components, gap)), coverage)
}

// invariant: map epochs come from the validated IONEX product axis.
#[allow(clippy::expect_used)]
fn map_epoch_j2000_s(map_epochs: &[Instant], index: usize) -> i64 {
    exact_j2000_second(map_epochs[index]).expect("IONEX map epoch is convertible to J2000 seconds")
}
