//! Vienna Mapping Function 1 (VMF1) tropospheric mapping.
//!
//! VMF1 (Böhm, Werl & Schuh 2006, "Troposphere mapping functions for GPS and
//! very long baseline interferometry from European Centre for Medium-Range
//! Weather Forecasts operational analysis data", J. Geophys. Res. 111, B02406,
//! doi:10.1029/2005JB003629) maps the zenith hydrostatic and wet tropospheric
//! delays to the line of sight with the SAME three-term continued fraction as
//! Niell (see [`crate::tropo::saastamoinen::mapf`]). The families differ only in
//! the coefficients: VMF1's hydrostatic and wet `a` coefficients are
//! ray-traced, epoch-wise (00/06/12/18 UT) from a numerical weather model and
//! supplied as a data product, where Niell's come from a climatological table;
//! the `b` and `c` coefficients are VMF1's own (below).
//!
//! This is the SITE-WISE form, matching the TU Wien reference `vmf1.f`
//! (`https://vmf.geo.tuwien.ac.at/codes/vmf1.f`). The site-wise `a` coefficients
//! are interpolated from the surrounding grid to the station, so NO height
//! correction is applied. Per the TU Wien products documentation: "Hydrostatic
//! and wet 'a' coefficients of [site-wise] VMF1 can be used as input for the
//! routines vmf1.f/vmf1.m." (The Niell-style height correction lives only in the
//! grid routine `vmf1_ht.f`, which adjusts grid-height `a` coefficients to an
//! arbitrary station height; applying it to site-wise coefficients would
//! double-count the height adjustment.)
//!
//! b, c source (vmf1.f / Böhm 2006, corrected-coefficient errata):
//!   * hydrostatic: `bh = 0.0029`; `ch` is the seasonal expression
//!     `ch = c0h + ((cos(doy/DAYS_PER_JULIAN_YEAR*2π + φ) + 1)*c11h/2 + c10h)*(1 - cos φ_lat)`
//!     with `c0h = 0.062` and, for the northern hemisphere, `φ = 0`,
//!     `c10h = 0.001`, `c11h = 0.005`; for the southern hemisphere `φ = π`,
//!     `c10h = 0.002`, `c11h = 0.007`. `φ_lat` is the ellipsoidal latitude.
//!   * wet: `bw = 0.00146`, `cw = 0.04391` (constants).
//!
//! The seasonal reference is 28 January (Niell 1996 convention), entered as
//! `doy = mjd - 44239 + 1 - 28` exactly as in `vmf1.f` (where MJD 44239 is
//! 1980-01-01); `mjd` is the modified Julian date of the epoch.
//!
//! The seasonal phase and the zenith-distance complement use the TU Wien
//! constant `3.14159265359`.

use crate::astro::constants::time::DAYS_PER_JULIAN_YEAR;

use super::saastamoinen::mapf_from_sine;

#[allow(clippy::approx_constant)]
pub(super) const VMF1_PI: f64 = 3.14159265359;

/// VMF1 mapping factors at one elevation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Vmf1Components {
    /// Hydrostatic mapping factor (dimensionless), no height correction.
    pub mh: f64,
    /// Wet mapping factor (dimensionless).
    pub mw: f64,
    /// Hydrostatic `c` coefficient from the seasonal expression (intermediate).
    pub ch: f64,
}

/// VMF1 hydrostatic and wet mapping factors for site-wise `a` coefficients.
///
/// `zenith_distance_rad` is the zenith distance in radians, `lat_rad` the ellipsoidal
/// latitude in radians, `mjd` the modified Julian date of the epoch, and
/// `ah`/`aw` the hydrostatic/wet `a` coefficients from the VMF1 site-wise data
/// product at that station and epoch. Mirrors the TU Wien `vmf1.f` term order.
pub(crate) fn vmf1_mapping(
    zenith_distance_rad: f64,
    lat_rad: f64,
    mjd: f64,
    ah: f64,
    aw: f64,
) -> Vmf1Components {
    // Reference day 28 January (Niell 1996), via vmf1.f's MJD expression.
    let doy = mjd - 44239.0 + 1.0 - 28.0;

    // The reference coefficients are binary32 values promoted to binary64.
    let bh = 0.0029_f32 as f64;
    let c0h = 0.062_f32 as f64;
    let (phh, c11h, c10h) = if lat_rad < 0.0 {
        // Southern hemisphere.
        (VMF1_PI, 0.007_f32 as f64, 0.002_f32 as f64)
    } else {
        // Northern hemisphere.
        (0.0, 0.005_f32 as f64, 0.001_f32 as f64)
    };
    let ch = c0h
        + ((libm::cos(doy / DAYS_PER_JULIAN_YEAR * 2.0 * VMF1_PI + phh) + 1.0) * c11h / 2.0 + c10h)
            * (1.0 - libm::cos(lat_rad));
    let sine = libm::sin(VMF1_PI / 2.0 - zenith_distance_rad);
    let mh = mapf_from_sine(sine, ah, bh, ch);

    let bw = 0.00146_f32 as f64;
    let cw = 0.04391_f32 as f64;
    let mw = mapf_from_sine(sine, aw, bw, cw);

    Vmf1Components { mh, mw, ch }
}
