//! Saastamoinen zenith delays + Niell (NMF) mapping, the bit-exact core recipe.
//!
//! This module is the operation-order-pinned heart of the troposphere model. It
//! takes angles already reduced to radians and a fractional day-of-year, and
//! returns every intermediate quantity (water-vapour partial pressure, the two
//! zenith delays, the two mapping factors, the height correction, the seasonal
//! phase) alongside the final slant delay. Carrying every intermediate lets the
//! parity test localise any divergence to a single algorithm step rather than
//! only seeing the end result move.
//!
//! The zenith delays follow the Saastamoinen (1972) hydrostatic and wet forms,
//! driven by supplied surface meteorology (pressure, temperature, relative
//! humidity) rather than a synthesised standard atmosphere. The slant mapping is
//! the Niell (1996) continued-fraction NMF, with the seasonal hydrostatic
//! coefficients interpolated across the node latitudes 15/30/45/60/75 degrees
//! and a height correction applied to the hydrostatic mapping only. The term
//! groupings match the de-facto standard implementation exactly.
//!
//! Determinism: the transcendentals used are `cos`, `sin`, `exp`, and `pow`
//! (the last only in the standard-atmosphere helper). There is no fused
//! multiply-add anywhere; every product and sum is a plain operator, so the
//! operation tree is identical to the reference recipe and the result is
//! bit-stable on the certified target.

use core::f64::consts::PI;

use crate::astro::constants::units::{KM_TO_M, RAD_TO_DEG};

use crate::frame::Wgs84Geodetic;

/// Niell continued-fraction height-correction coefficients `a, b, c`.
const AHT: [f64; 3] = [2.53e-5, 5.49e-3, 1.14e-3];

/// Niell (1996) table 3 coefficients at node latitudes 15/30/45/60/75 degrees.
///
/// Rows, in order: hydrostatic-average `a, b, c`; hydrostatic-amplitude
/// `a, b, c`; wet `a, b, c`. The hydrostatic mapping uses the average minus the
/// amplitude scaled by the seasonal cosine; the wet mapping uses the wet row
/// directly with no seasonal term.
const COEF: [[f64; 5]; 9] = [
    [
        1.2769934e-3,
        1.2683230e-3,
        1.2465397e-3,
        1.2196049e-3,
        1.2045996e-3,
    ], // hyd ave a
    [
        2.9153695e-3,
        2.9152299e-3,
        2.9288445e-3,
        2.9022565e-3,
        2.9024912e-3,
    ], // hyd ave b
    [
        62.610505e-3,
        62.837393e-3,
        63.721774e-3,
        63.824265e-3,
        64.258455e-3,
    ], // hyd ave c
    [
        0.0000000e-0,
        1.2709626e-5,
        2.6523662e-5,
        3.4000452e-5,
        4.1202191e-5,
    ], // hyd amp a
    [
        0.0000000e-0,
        2.1414979e-5,
        3.0160779e-5,
        7.2562722e-5,
        11.723375e-5,
    ], // hyd amp b
    [
        0.0000000e-0,
        9.0128400e-5,
        4.3497037e-5,
        84.795348e-5,
        170.37206e-5,
    ], // hyd amp c
    [
        5.8021897e-4,
        5.6794847e-4,
        5.8118019e-4,
        5.9727542e-4,
        6.1641693e-4,
    ], // wet a
    [
        1.4275268e-3,
        1.5138625e-3,
        1.4572752e-3,
        1.5007428e-3,
        1.7599082e-3,
    ], // wet b
    [
        4.3472961e-2,
        4.6729510e-2,
        4.3908931e-2,
        4.4626982e-2,
        5.4736038e-2,
    ], // wet c
];

/// Met-path height validity gate, inclusive lower bound in meters.
pub(crate) const MET_GATE_LOW_M: f64 = -100.0;

/// Met-path height validity gate, inclusive upper bound in meters.
pub(crate) const MET_GATE_HI_M: f64 = 1.0e4;

/// Zenith hydrostatic and wet delays plus the water-vapour partial pressure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ZenithComponents {
    /// Water-vapour partial pressure (hPa).
    pub e: f64,
    /// Zenith hydrostatic (dry) delay (meters).
    pub zhd_m: f64,
    /// Zenith wet delay (meters).
    pub zwd_m: f64,
}

/// Niell mapping factors plus the seasonal-phase intermediates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct MappingComponents {
    /// Hydrostatic mapping factor (dimensionless), including height correction.
    pub mh: f64,
    /// Wet mapping factor (dimensionless).
    pub mw: f64,
    /// Hydrostatic height correction term (dimensionless).
    pub dm: f64,
    /// Cosine of the seasonal phase.
    pub cosy: f64,
    /// Seasonal phase fraction (years from the day-28 reference).
    pub y: f64,
}

/// Every intermediate of one slant-delay evaluation, plus the final meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct SlantComponents {
    /// Water-vapour partial pressure (hPa).
    pub e: f64,
    /// Zenith hydrostatic (dry) delay (meters).
    pub zhd_m: f64,
    /// Zenith wet delay (meters).
    pub zwd_m: f64,
    /// Hydrostatic mapping factor (dimensionless).
    pub mh: f64,
    /// Wet mapping factor (dimensionless).
    pub mw: f64,
    /// Hydrostatic height correction term (dimensionless).
    pub dm: f64,
    /// Cosine of the seasonal phase.
    pub cosy: f64,
    /// Seasonal phase fraction.
    pub y: f64,
    /// Slant tropospheric delay (meters), positive.
    pub slant_m: f64,
}

/// Synthesised standard-atmosphere pressure/temperature for a height.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct StandardAtmosphere {
    /// Total pressure (hPa).
    pub pressure_hpa: f64,
    /// Temperature (kelvin).
    pub temperature_k: f64,
    /// Relative humidity carried through from the caller (fraction in `[0, 1]`).
    pub relative_humidity: f64,
}

/// Latitude interpolation over the node latitudes 15/30/45/60/75 degrees.
///
/// Below 15 degrees clamps to the 15-degree column and above 75 degrees clamps
/// to the 75-degree column (flat, no extrapolation); between nodes it is a
/// linear blend of the two bracketing columns.
fn interpc(coef: &[f64; 5], lat_deg: f64) -> f64 {
    let i = (lat_deg / 15.0) as i32;
    if i < 1 {
        return coef[0];
    }
    if i > 4 {
        return coef[4];
    }
    let iu = i as usize;
    let fi = i as f64;
    coef[iu - 1] * (1.0 - lat_deg / 15.0 + fi) + coef[iu] * (lat_deg / 15.0 - fi)
}

/// Three-term continued-fraction mapping function for coefficients `a, b, c`.
///
/// This is the Marini/Herring continued fraction shared by the Niell (NMF) and
/// Vienna (VMF) mapping functions: the families differ only in where `a, b, c`
/// come from, not in this evaluation, so the VMF module reuses it verbatim.
pub(crate) fn mapf(el_rad: f64, a: f64, b: f64, c: f64) -> f64 {
    let sinel = el_rad.sin();
    (1.0 + a / (1.0 + b / (1.0 + c))) / (sinel + (a / (sinel + b / (sinel + c))))
}

/// Saastamoinen zenith hydrostatic and wet delays from supplied meteorology.
///
/// `pressure_hpa` is total pressure in hectopascals, `temperature_k` is absolute
/// temperature in kelvin, `relative_humidity` is a fraction in `[0, 1]`,
/// `lat_rad` is the geodetic latitude in radians, and `height_m` is the WGS84
/// ellipsoidal height in meters (used with its sign; not clamped on this path).
/// These are the zenith parts of the Saastamoinen forms with the slant mapping
/// removed, since the mapping is applied separately by Niell.
pub(crate) fn zenith_delays(
    pressure_hpa: f64,
    temperature_k: f64,
    relative_humidity: f64,
    lat_rad: f64,
    height_m: f64,
) -> ZenithComponents {
    let t = temperature_k;
    // Water-vapour partial pressure (hPa): saturation pressure times humidity.
    let e = 6.108 * relative_humidity * ((17.15 * t - 4684.0) / (t - 38.45)).exp();

    let h_km = height_m / KM_TO_M;
    let denom = 1.0 - 0.00266 * (2.0 * lat_rad).cos() - 0.00028 * h_km;
    let zhd_m = 0.0022768 * pressure_hpa / denom;

    let zwd_m = 0.002277 * (1255.0 / t + 0.05) * e;

    ZenithComponents { e, zhd_m, zwd_m }
}

/// Standard-atmosphere pressure/temperature synthesis for a height.
///
/// A convenience for callers without live meteorology. The height is clamped to
/// be non-negative before the pressure lapse and temperature lapse are applied;
/// the supplied relative humidity is carried through unchanged. The pressure
/// lapse uses `pow`, the one transcendental specific to this helper.
pub(crate) fn standard_atmosphere(height_m: f64, relative_humidity: f64) -> StandardAtmosphere {
    let hgt = if height_m > 0.0 { height_m } else { 0.0 };
    let pressure_hpa = 1013.25 * (1.0 - 2.2557e-5 * hgt).powf(5.2568);
    let temperature_k = 15.0 - 6.5e-3 * hgt + 273.16;
    StandardAtmosphere {
        pressure_hpa,
        temperature_k,
        relative_humidity,
    }
}

/// Niell (1996) hydrostatic and wet mapping factors and seasonal intermediates.
///
/// `el_rad` is the elevation angle in radians, `lat_rad` is the geodetic
/// latitude in radians, `height_m` is the WGS84 ellipsoidal height in meters,
/// and `doy` is the fractional day-of-year (Jan 1 00:00:00 = 1.0). The seasonal
/// phase references day 28 and adds half a year for southern latitudes. The
/// hydrostatic mapping carries the height correction; the wet mapping does not.
pub(crate) fn niell_mapping(
    el_rad: f64,
    lat_rad: f64,
    height_m: f64,
    doy: f64,
) -> MappingComponents {
    let lat_deg = lat_rad * RAD_TO_DEG;
    // Seasonal phase: day-28 reference, plus half a year for southern latitudes.
    let y = (doy - 28.0) / 365.25 + if lat_deg < 0.0 { 0.5 } else { 0.0 };
    let cosy = (2.0 * PI * y).cos();
    let alat = lat_deg.abs();

    let mut ah = [0.0f64; 3];
    let mut aw = [0.0f64; 3];
    for k in 0..3 {
        ah[k] = interpc(&COEF[k], alat) - interpc(&COEF[k + 3], alat) * cosy;
        aw[k] = interpc(&COEF[k + 6], alat);
    }

    // Height correction (km), ellipsoidal height, applied to hydrostatic only.
    let dm = (1.0 / el_rad.sin() - mapf(el_rad, AHT[0], AHT[1], AHT[2])) * (height_m / KM_TO_M);
    let mh = mapf(el_rad, ah[0], ah[1], ah[2]) + dm;
    let mw = mapf(el_rad, aw[0], aw[1], aw[2]);

    MappingComponents {
        mh,
        mw,
        dm,
        cosy,
        y,
    }
}

/// Full slant tropospheric delay (meters) with every intermediate.
///
/// `el_rad`, `lat_rad`, and `lon_rad` are radians; `height_m` is the WGS84
/// ellipsoidal height in meters; `pressure_hpa`/`temperature_k`/
/// `relative_humidity` are the surface meteorology; `doy` is the fractional
/// day-of-year. Longitude is accepted for signature symmetry with the geometry
/// boundary but does not enter the delay.
///
/// Validity gates (permissive: out-of-range returns all-zero with `slant_m = 0`):
/// elevation `el_rad <= 0` yields zero, and a Met-path height outside
/// `[-100, 1e4]` meters yields zero. Inside the gates the possibly-negative
/// ellipsoidal height flows through with its sign. The slant combination is the
/// hydrostatic term first: `zhd_m * mh + zwd_m * mw`.
#[allow(clippy::manual_range_contains)]
pub(crate) fn slant_components(
    el_rad: f64,
    receiver: Wgs84Geodetic,
    pressure_hpa: f64,
    temperature_k: f64,
    relative_humidity: f64,
    doy: f64,
) -> SlantComponents {
    // Longitude does not enter the delay; the geodetic carries it for the
    // shared receiver-position type but only latitude and height are read here.
    let Wgs84Geodetic {
        lat_rad,
        lon_rad: _,
        height_m,
    } = receiver;
    if el_rad <= 0.0 || height_m < MET_GATE_LOW_M || height_m > MET_GATE_HI_M {
        return SlantComponents {
            e: 0.0,
            zhd_m: 0.0,
            zwd_m: 0.0,
            mh: 0.0,
            mw: 0.0,
            dm: 0.0,
            cosy: 0.0,
            y: 0.0,
            slant_m: 0.0,
        };
    }

    let z = zenith_delays(
        pressure_hpa,
        temperature_k,
        relative_humidity,
        lat_rad,
        height_m,
    );
    let m = niell_mapping(el_rad, lat_rad, height_m, doy);
    let slant_m = z.zhd_m * m.mh + z.zwd_m * m.mw;

    SlantComponents {
        e: z.e,
        zhd_m: z.zhd_m,
        zwd_m: z.zwd_m,
        mh: m.mh,
        mw: m.mw,
        dm: m.dm,
        cosy: m.cosy,
        y: m.y,
        slant_m,
    }
}
