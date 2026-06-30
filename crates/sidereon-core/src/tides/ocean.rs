//! Ocean tide loading station displacement (IERS Conventions 2010, §6.2; the
//! Bos-Scherneck BLQ convention), via the IERS `ARG2` 11-constituent
//! astronomical-argument method.
//!
//! Scope: this is the ARG2 main-constituent method (the 11 BLQ constituents
//! below), **not** the full HARDISP admittance scheme. It does not apply the
//! 18.6-yr nodal modulation or interpolate the minor side constituents that
//! HARDISP (e.g. RTKLIB's 342-constituent spline) carries - see the
//! "Astronomical arguments" note. For inland stations the difference is sub-mm
//! (validated against RTKLIB below), but this is deliberately the ARG2
//! approximation, not a HARDISP reimplementation.
//!
//! [`ocean_tide_loading`] computes the displacement of an Earth-fixed (ITRF)
//! station caused by the elastic deformation of the solid Earth under the
//! periodic load of the ocean tide. It is the sibling of
//! [`super::solid_earth_tide`] and [`super::solid_earth_pole_tide`] and is wired
//! into the PPP correction stack in the identical way: a per-epoch station
//! displacement vector projected onto the line of sight in
//! `precise_positioning/model.rs`.
//!
//! Physics (IERS Conventions 2010, §6.2; the HARDISP / BLQ convention). The
//! site displacement in each of the three BLQ components is the sum over 11
//! tidal constituents (in BLQ column order M2, S2, N2, K2, K1, O1, P1, Q1, Mf,
//! Mm, Ssa) of
//!
//! ```text
//! dc(t) = sum_j  A_cj * cos( arg_j(t) - phi_cj )          (per component c)
//! ```
//!
//! where `A_cj` (m) and `phi_cj` (rad) are the per-station BLQ amplitude and
//! Greenwich phase lag for component `c` and constituent `j`, and `arg_j(t)` is
//! the astronomical (equilibrium) argument of constituent `j` at the epoch.
//! This is the displacement formula the Bos-Scherneck BLQ tables are designed
//! for; RTKLIB's `tide_oload`/`hardisp` is used as the validation oracle, and
//! agreement holds to sub-mm for inland stations (it is not claimed to be
//! bit-identical to HARDISP - the constituent sets differ, see below).
//!
//! Astronomical arguments. `arg_j(t)` is the IERS `ARG2` argument (IERS
//! Conventions 2010 Chapter 7 reference software `ARG2.F`):
//!
//! ```text
//! arg_j = SPEED_j * FDAY + n1_j*h0 + n2_j*s0 + n3_j*p0 + n4_j*2pi   (mod 2pi)
//! ```
//!
//! with `FDAY` the UT seconds of the day, `(h0, s0, p0)` the mean longitudes of
//! the Sun, the Moon, and the lunar perigee at 0h of the day (`ARG2.F` cubic
//! polynomials in `CAPT`, Julian centuries from the 1975 reference epoch),
//! `SPEED_j` the constituent angular speed (rad/s), and `(n1..n4)_j` the
//! `ANGFAC` multipliers. The quarter-cycle `n4_j` entries (`+/-0.25`) are the
//! Schwiderski phase corrections the `cos(arg - phi)` convention requires for
//! the diurnal band. `ARG2` deliberately omits the 18.6-yr nodal modulation and
//! the minor side constituents that the full HARDISP admittance method (e.g.
//! RTKLIB's 342-constituent spline) interpolates; for an inland station the
//! resulting difference is well below the millimetre (verified against RTKLIB in
//! `tests/ocean_loading_oracle.rs`).
//!
//! BLQ components are radial (positive up), tangential EW (positive west), and
//! tangential NS (positive south); the returned vector is the geodetic ENU
//! displacement (east = -west, north = -south, up = radial) rotated to ECEF on
//! the WGS84 ellipsoid, matching RTKLIB's `ecef2pos`/`xyz2enu`.
//!
//! The per-station BLQ coefficients are a data dependency the caller supplies
//! from an ocean-loading provider (Bos-Scherneck / OSO Chalmers, or equivalent);
//! the engine does not embed them and they must not be fabricated.

#[cfg(test)]
mod tests;

use crate::astro::constants::units::{DEG_TO_RAD, KM_TO_M};
use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use crate::astro::math::vec3::norm3_ref as norm;
use crate::validate;

use super::{cal2jd, invalid_tide_input, TideError};

/// Number of BLQ tidal constituents (M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa).
pub const NUM_OCEAN_CONSTITUENTS: usize = 11;

/// Two pi (cycle of an astronomical argument).
const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// IERS `ARG2.F` constituent angular speeds (rad/s), BLQ column order
/// M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa.
const SPEED_RAD_S: [f64; NUM_OCEAN_CONSTITUENTS] = [
    1.405_19e-4,
    1.454_44e-4,
    1.378_80e-4,
    1.458_42e-4,
    0.729_21e-4,
    0.675_98e-4,
    0.725_23e-4,
    0.649_59e-4,
    0.053_234e-4,
    0.026_392e-4,
    0.003_982e-4,
];

/// IERS `ARG2.F` `ANGFAC` multipliers `(h0, s0, p0, 2pi)` per constituent. The
/// fourth column is the quarter-cycle Schwiderski phase correction.
#[rustfmt::skip]
const ANGFAC: [[f64; 4]; NUM_OCEAN_CONSTITUENTS] = [
    [ 2.0, -2.0,  0.0,  0.00], // M2
    [ 0.0,  0.0,  0.0,  0.00], // S2
    [ 2.0, -3.0,  1.0,  0.00], // N2
    [ 2.0,  0.0,  0.0,  0.00], // K2
    [ 1.0,  0.0,  0.0,  0.25], // K1
    [ 1.0, -2.0,  0.0, -0.25], // O1
    [-1.0,  0.0,  0.0, -0.25], // P1
    [ 1.0, -3.0,  1.0, -0.25], // Q1
    [ 0.0,  2.0,  0.0,  0.00], // Mf
    [ 0.0,  1.0, -1.0,  0.00], // Mm
    [ 2.0,  0.0,  0.0,  0.00], // Ssa
];

/// Per-station ocean-loading BLQ coefficients (Bos-Scherneck / HARDISP format).
///
/// Both arrays are indexed `[component][constituent]`. The component order is
/// the BLQ row order: radial / up-positive (0), tangential EW / west-positive
/// (1), tangential NS / south-positive (2). The constituent order is the BLQ
/// column order M2 S2 N2 K2 K1 O1 P1 Q1 Mf Mm Ssa.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OceanLoadingBlq {
    /// Constituent amplitudes (m).
    pub amplitude_m: [[f64; NUM_OCEAN_CONSTITUENTS]; 3],
    /// Constituent Greenwich phase lags (degrees, positive lag).
    pub phase_deg: [[f64; NUM_OCEAN_CONSTITUENTS]; 3],
}

/// Ocean tide loading displacement of an ITRF station, in metres (ECEF).
///
/// Arguments:
/// * `xsta` - geocentric station position (m, ITRF).
/// * `year`, `month`, `day` - UTC calendar date (selects the day of year).
/// * `fhr` - UTC fractional hour of the day (`hour + min/60 + sec/3600`).
/// * `blq` - the station's BLQ ocean-loading coefficients (a data dependency the
///   caller supplies; the engine does not embed them).
///
/// Returns the displacement vector (m, geocentric ITRF), to be projected onto
/// the line of sight identically to [`super::solid_earth_tide`].
///
/// Returns [`TideError`] when inputs are non-finite, the date/hour is invalid,
/// the BLQ coefficients are non-finite, or the station vector is degenerate
/// (zero radius).
pub fn ocean_tide_loading(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    blq: &OceanLoadingBlq,
) -> Result<[f64; 3], TideError> {
    validate_ocean_loading_domain(xsta, year, month, day, fhr, blq)?;
    Ok(ocean_tide_loading_unchecked(
        xsta, year, month, day, fhr, blq,
    ))
}

fn validate_ocean_loading_domain(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    blq: &OceanLoadingBlq,
) -> Result<(), TideError> {
    validate::finite_vec3(*xsta, "station position").map_err(invalid_tide_input)?;
    validate::civil_datetime_with_second_policy(
        i64::from(year),
        i64::from(month),
        i64::from(day),
        0,
        0,
        0.0,
        validate::CivilSecondPolicy::Continuous,
    )
    .map_err(invalid_tide_input)?;
    validate::finite_in_range_exclusive_upper(fhr, 0.0, 24.0, "fractional hour")
        .map_err(invalid_tide_input)?;

    for component in &blq.amplitude_m {
        for &amplitude in component {
            validate::finite(amplitude, "ocean loading amplitude").map_err(invalid_tide_input)?;
        }
    }
    for component in &blq.phase_deg {
        for &phase in component {
            validate::finite(phase, "ocean loading phase").map_err(invalid_tide_input)?;
        }
    }

    validate::finite_positive(norm(xsta), "station radius").map_err(invalid_tide_input)?;

    Ok(())
}

fn ocean_tide_loading_unchecked(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    blq: &OceanLoadingBlq,
) -> [f64; 3] {
    let arg = arg2_angles(year, month, day, fhr);

    // BLQ component sums: 0 = radial (up), 1 = EW (west), 2 = NS (south).
    let mut component = [0.0_f64; 3];
    for (slot, (amplitudes, phases)) in component
        .iter_mut()
        .zip(blq.amplitude_m.iter().zip(blq.phase_deg.iter()))
    {
        let mut sum = 0.0;
        for ((&amplitude, &phase_deg), &a) in amplitudes.iter().zip(phases).zip(&arg) {
            sum += amplitude * (a - phase_deg * DEG_TO_RAD).cos();
        }
        *slot = sum;
    }
    let up = component[0];
    let west = component[1];
    let south = component[2];
    let east = -west;
    let north = -south;

    // Geodetic (WGS84) ENU -> ECEF, matching RTKLIB ecef2pos/xyz2enu.
    let (lat_deg, lon_deg, _height_km) =
        itrs_to_geodetic_compute(xsta[0] / KM_TO_M, xsta[1] / KM_TO_M, xsta[2] / KM_TO_M)
            .expect("validated station position yields geodetic coordinates");
    let (sinlat, coslat) = (lat_deg * DEG_TO_RAD).sin_cos();
    let (sinlon, coslon) = (lon_deg * DEG_TO_RAD).sin_cos();

    // ENU basis vectors expressed in ECEF (geodetic topocentric frame):
    //   e = [-sinlon, coslon, 0]
    //   n = [-sinlat coslon, -sinlat sinlon, coslat]
    //   u = [ coslat coslon,  coslat sinlon, sinlat]
    [
        east * (-sinlon) + north * (-sinlat * coslon) + up * (coslat * coslon),
        east * coslon + north * (-sinlat * sinlon) + up * (coslat * sinlon),
        north * coslat + up * sinlat,
    ]
}

/// IERS `ARG2.F` astronomical arguments (radians) of the 11 BLQ constituents at
/// the given UTC epoch.
fn arg2_angles(year: i32, month: i32, day: i32, fhr: f64) -> [f64; NUM_OCEAN_CONSTITUENTS] {
    let doy = day_of_year(year, month, day);
    // `DAY` of ARG2 is the fractional day of year; `ID` its integer part and
    // `FDAY` the seconds into the day, i.e. `(DAY - ID) * 86400 = fhr * 3600`.
    let fday = fhr * 3600.0;

    // ARG2.F day count and Julian centuries from the 1975 reference epoch.
    // Fortran integer division (truncating toward zero, == floor for years
    // >= 1973, the supported range) is reproduced by Rust's `/` on i32.
    let icapd = doy + 365 * (year - 1975) + (year - 1973) / 4;
    let capt = (27_392.500_528 + 1.000_000_035 * f64::from(icapd)) / 36_525.0;

    // Mean longitudes (rad). ARG2.F uses a truncated DTR; the exact PI/180 used
    // here is sub-femtometre different and is closer to the rigorous argument.
    let h0 = (279.696_68 + (36_000.768_930_485 + 3.03e-4 * capt) * capt) * DEG_TO_RAD;
    let s0 = (((1.9e-6 * capt - 0.001_133) * capt + 481_267.883_141_37) * capt + 270.434_358)
        * DEG_TO_RAD;
    let p0 = (((-1.2e-5 * capt - 0.010_325) * capt + 4_069.034_032_957_7) * capt + 334.329_653)
        * DEG_TO_RAD;

    let mut angle = [0.0_f64; NUM_OCEAN_CONSTITUENTS];
    for (j, slot) in angle.iter_mut().enumerate() {
        let a = SPEED_RAD_S[j] * fday
            + ANGFAC[j][0] * h0
            + ANGFAC[j][1] * s0
            + ANGFAC[j][2] * p0
            + ANGFAC[j][3] * TWO_PI;
        *slot = a.rem_euclid(TWO_PI);
    }
    angle
}

/// 1-based UTC day of year (ARG2 `ID`), from the IERS/SOFA `CAL2JD` MJD diff
/// (the `djm` return is in days, so the difference is the day-of-year minus 1).
fn day_of_year(year: i32, month: i32, day: i32) -> i32 {
    let (_, mjd) = cal2jd(year, month, day);
    let (_, mjd_jan1) = cal2jd(year, 1, 1);
    (mjd - mjd_jan1).round() as i32 + 1
}
