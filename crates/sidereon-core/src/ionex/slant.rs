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
//! The delay returned is a group delay and is positive: it increases the
//! measured pseudorange (the carrier-phase advance is the negation of this
//! value).
//!
//! The grid is stored north-to-south in latitude (negative latitude step) and
//! west-to-east in longitude (positive longitude step); the bracketing and the
//! signed-step fractional offsets follow that ordering directly. There is no
//! fused multiply-add anywhere: every product and sum is a plain operator, so
//! the operation tree is identical to the reference recipe and the result is
//! bit-stable.

use super::j2000_seconds_from_instant;
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

    let s = re_km / (re_km + h_km) * el_rad.cos();

    // Earth-central angle from receiver to pierce point.
    let psi = PI / 2.0 - el_rad - s.asin();

    let phi_ipp = (lat_rad.sin() * psi.cos() + lat_rad.cos() * psi.sin() * az_rad.cos()).asin();
    let lambda_ipp = lon_rad + (psi.sin() * az_rad.sin() / phi_ipp.cos()).asin();

    PiercePoint {
        s,
        psi,
        phi_ipp_deg: phi_ipp * RAD_TO_DEG,
        lambda_ipp_deg: lambda_ipp * RAD_TO_DEG,
    }
}

/// Bring a longitude into the grid's usable longitude range.
///
/// Full 360-degree grids wrap by `+-360` steps. Regional grids cover only an
/// interval on the longitude circle, so out-of-coverage pierce points are held
/// at the nearest interval edge instead of extrapolating past the edge cell.
fn normalize_lon_deg(mut lon_deg: f64, lon1: f64, lon2: f64) -> f64 {
    if !lon_deg.is_finite() {
        return lon_deg;
    }

    if lon2 - lon1 >= 360.0 {
        while lon_deg < lon1 {
            lon_deg += 360.0;
        }
        while lon_deg > lon2 {
            lon_deg -= 360.0;
        }
        return lon_deg;
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
    best_lon
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
    /// Interpolated vertical TEC at the pierce point (TECU).
    pub vtec: f64,
    /// Longitude-direction fractional offset within the cell.
    pub p: f64,
    /// Latitude-direction fractional offset within the cell (signed step).
    pub q: f64,
}

/// Explicit four-term bilinear VTEC at `(phi_deg, lam_deg)` on one map.
///
/// `vtec_map` is indexed `[i_lat][i_lon]` matching `lat_arr` (descending) and
/// `lon_arr` (ascending). The pierce-point latitude and longitude are clamped to
/// the grid edge before bracketing. The weighted sum is the explicit four-term form
/// `(1-p)(1-q)E00 + p(1-q)E01 + (1-p)q E10 + p q E11` with `(1-p)`/`(1-q)`
/// formed once.
pub(crate) fn bilinear_vtec(
    vtec_map: &[Vec<f64>],
    lat_arr: &[f64],
    lon_arr: &[f64],
    dlat: f64,
    dlon: f64,
    phi_deg: f64,
    lam_deg: f64,
) -> BilinearVtec {
    let nlat = lat_arr.len();
    let nlon = lon_arr.len();

    // Clamp the pierce-point latitude to the grid extent (descending lat).
    let lat_hi = lat_arr[0];
    let lat_lo = lat_arr[nlat - 1];
    let mut phi = phi_deg;
    if phi > lat_hi {
        phi = lat_hi;
    }
    if phi < lat_lo {
        phi = lat_lo;
    }
    let lon_lo = lon_arr[0];
    let lon_hi = lon_arr[nlon - 1];
    let mut lam = lam_deg;
    if lam < lon_lo {
        lam = lon_lo;
    }
    if lam > lon_hi {
        lam = lon_hi;
    }

    let i = bracket(phi, lat_arr[0], dlat, nlat);
    let j = bracket(lam, lon_arr[0], dlon, nlon);

    let lat0 = lat_arr[i];
    let lon0 = lon_arr[j];

    // Signed-step fractional offsets: both land in [0, 1].
    let q = (phi - lat0) / dlat;
    let p = (lam - lon0) / dlon;

    let e00 = vtec_map[i][j];
    let e01 = vtec_map[i][j + 1];
    let e10 = vtec_map[i + 1][j];
    let e11 = vtec_map[i + 1][j + 1];

    let one_p = 1.0 - p;
    let one_q = 1.0 - q;
    let vtec = one_p * one_q * e00 + p * one_q * e01 + one_p * q * e10 + p * q * e11;

    BilinearVtec { vtec, p, q }
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
    /// Bilinear VTEC on the lower bracketing map (TECU).
    pub vtec0: f64,
    /// Bilinear VTEC on the upper bracketing map (TECU).
    pub vtec1: f64,
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

/// Full IONEX slant group delay in meters, with all intermediates.
///
/// `maps` is one VTEC grid per epoch in `map_epochs` (canonical instants), each a
/// 2-D array indexed `[i_lat][i_lon]`. The pierce-point VTEC is bilinearly
/// interpolated on the two maps bracketing `epoch_s` and then blended
/// linearly in time, holding the endpoint map outside coverage. The obliquity
/// factor `m = 1/cos(z') = 1/sqrt(1 - s^2)` maps vertical to slant TEC, and the
/// dispersive scaling `(40.3e16 / f^2) * STEC` gives the positive group delay.
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
pub(crate) struct VtecGridView<'a> {
    pub map_epochs: &'a [Instant],
    pub maps: &'a [Vec<Vec<f64>>],
    pub lat_arr: &'a [f64],
    pub lon_arr: &'a [f64],
    pub dlat: f64,
    pub dlon: f64,
}

pub(crate) fn slant_delay_components(
    los: PierceLineOfSight,
    frequency_hz: f64,
    re_km: f64,
    h_km: f64,
    epoch_s: i64,
    grid: VtecGridView,
) -> SlantComponents {
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

    let lon1 = lon_arr[0];
    let lon2 = lon_arr[lon_arr.len() - 1];
    let lam_deg = normalize_lon_deg(geom.lambda_ipp_deg, lon1, lon2);
    let phi_deg = geom.phi_ipp_deg;

    // Temporal bracket (hold the endpoint map outside coverage). A single-map
    // product has no interval to interpolate across, so it holds that one map
    // (weight 0); the second sample index is held at `ti` so it can never read
    // past the end. Multi-map products keep the original bracketing exactly.
    let nmaps = map_epochs.len();
    let (ti, ti1, w) = if nmaps <= 1 {
        (0usize, 0usize, 0.0)
    } else {
        let mut ti = 0usize;
        while ti < nmaps - 2 && epoch_s >= map_epoch_j2000_s(map_epochs, ti + 1) {
            ti += 1;
        }
        let t0 = map_epoch_j2000_s(map_epochs, ti);
        let t1 = map_epoch_j2000_s(map_epochs, ti + 1);
        let mut w = (epoch_s as f64 - t0 as f64) / (t1 as f64 - t0 as f64);
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
    let vtec0 = b0.vtec;
    let vtec1 = b1.vtec;
    let vtec = (1.0 - w) * vtec0 + w * vtec1;

    // Obliquity (slant) factor m(E) = 1 / cos(z') = 1 / sqrt(1 - s^2).
    let m = 1.0 / (1.0 - s * s).sqrt();
    let stec = m * vtec;

    let delay_m = (K_IONO / (frequency_hz * frequency_hz)) * stec;

    SlantComponents {
        s,
        psi: geom.psi,
        phi_ipp_deg: phi_deg,
        lambda_ipp_deg_raw: geom.lambda_ipp_deg,
        lambda_ipp_deg: lam_deg,
        map_index: ti,
        w,
        vtec0,
        vtec1,
        p0: b0.p,
        q0: b0.q,
        vtec,
        m,
        stec,
        delay_m,
    }
}

fn map_epoch_j2000_s(map_epochs: &[Instant], index: usize) -> i64 {
    j2000_seconds_from_instant(map_epochs[index])
        .expect("IONEX map epoch is convertible to J2000 seconds")
}
