//! Solid-Earth pole tide station displacement (IERS Conventions 2010, §7.1.4).
//!
//! [`solid_earth_pole_tide`] computes the displacement of an Earth-fixed (ITRF)
//! station caused by the centrifugal-potential change of polar motion (the
//! "rotational deformation due to polar motion"). It is the sibling of
//! [`super::solid_earth_tide`] and is wired into the PPP correction stack in the
//! identical way: a per-epoch station displacement vector projected onto the
//! line of sight.
//!
//! Physics (IERS Conventions 2010, Chapter 7, §7.1.4):
//!
//! * The wobble variables are formed from the polar motion `(xp, yp)` and the
//!   IERS conventional mean pole `(x_bar, y_bar)`:
//!
//! ```text
//! m1 =  xp - x_bar   (arcsec)
//! m2 = -(yp - y_bar) (arcsec)
//! ```
//!
//! * The displacement, in the local radial / colatitude / longitude triad, is
//!   IERS Conventions (2010) Eq. (7.24):
//!
//! ```text
//! S_r      = -33 sin(2 theta) (m1 cos lambda + m2 sin lambda)  mm
//! S_theta  =  -9 cos(2 theta) (m1 cos lambda + m2 sin lambda)  mm
//! S_lambda =   9 cos(theta)   (m1 sin lambda - m2 cos lambda)  mm
//! ```
//!
//!   where `theta` is the geocentric colatitude and `lambda` the longitude. The
//!   integer mm coefficients embed the nominal degree-2 Love/Shida numbers
//!   h2 = 0.6078 and l2 = 0.0847 (radial scales with h2; the horizontal terms
//!   with l2). `S_theta` points to increasing colatitude (south), so the local
//!   north component is `-S_theta`; `S_lambda` is east; `S_r` is up.
//!
//! Mean pole model. The IERS 2010 conventional mean pole (a cubic until 2010.0,
//! linear thereafter; IERS Conventions 2010 Table 7.7) was revised in 2018 to a
//! single **linear secular pole**, because the original post-2010 linear
//! extrapolation diverges from the observed secular drift at the multi-cm level
//! within a decade. This module uses the **revised IERS (2018) linear secular
//! pole** (updated Chapter 7), which is the conventional model for the modern
//! (post-2010) epochs Sidereon targets and the one adopted for ITRF2014/2020:
//!
//! ```text
//! x_bar(t) =  55.0 + 1.677 (t - 2000.0)  mas
//! y_bar(t) = 320.5 + 3.460 (t - 2000.0)  mas
//! ```
//!
//! with `t` the epoch in years. (The widely used RTKLIB still applies the
//! superseded 2010 cubic+linear model; at a 2026 epoch the two mean poles differ
//! by >0.1 arcsec in x, which is why the choice is documented here.)
//!
//! Polar motion `(xp, yp)` is **not** available from the engine's embedded EOP
//! table (it carries UT1-UTC only; see
//! `crate::astro::frames::transforms::PolarMotion`). The caller therefore
//! supplies `xp`/`yp` in arcseconds, sourced from IERS EOP exactly like the
//! other Earth-orientation inputs. They must not be fabricated.

#[cfg(test)]
mod tests;

use crate::astro::constants::time::J2000_JD;
use crate::astro::constants::units::MM_PER_M;
use crate::astro::math::vec3::norm3_ref as norm;
use crate::validate;

use super::{cal2jd, invalid_tide_input, TideError};

/// Julian days per Julian year, used to form the mean-pole epoch argument.
const DAYS_PER_JULIAN_YEAR: f64 = 365.25;

/// IERS (2018) linear secular pole coefficients, in milliarcseconds and
/// milliarcseconds per year (revised IERS Conventions 2010, updated Chapter 7).
const MEAN_POLE_X0_MAS: f64 = 55.0;
const MEAN_POLE_X_RATE_MAS_PER_YR: f64 = 1.677;
const MEAN_POLE_Y0_MAS: f64 = 320.5;
const MEAN_POLE_Y_RATE_MAS_PER_YR: f64 = 3.460;

/// Eq. (7.24) displacement coefficients in millimetres per arcsecond of wobble.
/// The radial coefficient carries Love number h2 = 0.6078; the horizontal
/// coefficients carry Shida number l2 = 0.0847.
const RADIAL_MM_PER_ARCSEC: f64 = -33.0;
const COLATITUDE_MM_PER_ARCSEC: f64 = -9.0;
const LONGITUDE_MM_PER_ARCSEC: f64 = 9.0;

/// Solid-Earth pole tide displacement of an ITRF station, in metres (ECEF).
///
/// Arguments:
/// * `xsta` - geocentric station position (m, ITRF).
/// * `year`, `month`, `day` - UTC calendar date (selects the mean pole epoch).
/// * `fhr` - UTC fractional hour of the day (`hour + min/60 + sec/3600`).
/// * `xp_arcsec`, `yp_arcsec` - IERS polar motion of the date (arcsec). These
///   are a data dependency the caller supplies from IERS EOP; the engine does
///   not embed them.
///
/// Returns the displacement vector (m, geocentric ITRF), to be projected onto
/// the line of sight identically to [`super::solid_earth_tide`].
///
/// Returns [`TideError`] when inputs are non-finite, the date/hour is invalid,
/// or the station vector is degenerate (zero radius or on the polar axis).
pub fn solid_earth_pole_tide(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
) -> Result<[f64; 3], TideError> {
    validate_pole_tide_domain(xsta, year, month, day, fhr, xp_arcsec, yp_arcsec)?;
    Ok(solid_earth_pole_tide_unchecked(
        xsta, year, month, day, fhr, xp_arcsec, yp_arcsec,
    ))
}

fn validate_pole_tide_domain(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
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
    validate::finite(xp_arcsec, "polar motion xp").map_err(invalid_tide_input)?;
    validate::finite(yp_arcsec, "polar motion yp").map_err(invalid_tide_input)?;

    validate::finite_positive(norm(xsta), "station radius").map_err(invalid_tide_input)?;
    let station_horizontal_radius = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt();
    validate::finite_positive(station_horizontal_radius, "station horizontal radius")
        .map_err(invalid_tide_input)?;

    Ok(())
}

fn solid_earth_pole_tide_unchecked(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    xp_arcsec: f64,
    yp_arcsec: f64,
) -> [f64; 3] {
    let (x_bar_arcsec, y_bar_arcsec) = mean_pole_arcsec(year, month, day, fhr);

    // Wobble variables (arcsec), IERS Conventions (2010) §7.1.4.
    let m1 = xp_arcsec - x_bar_arcsec;
    let m2 = -(yp_arcsec - y_bar_arcsec);

    // Geocentric station triad: theta is colatitude, lambda longitude.
    let rsta = norm(xsta);
    let sinphi = xsta[2] / rsta; // cos(theta)
    let cosphi = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt() / rsta; // sin(theta)
    let sinla = xsta[1] / cosphi / rsta;
    let cosla = xsta[0] / cosphi / rsta;

    let sin2theta = 2.0 * cosphi * sinphi;
    let cos2theta = sinphi * sinphi - cosphi * cosphi;
    let costheta = sinphi;

    let radial_factor = m1 * cosla + m2 * sinla;
    let longitude_factor = m1 * sinla - m2 * cosla;

    // Eq. (7.24), in millimetres.
    let s_r = RADIAL_MM_PER_ARCSEC * sin2theta * radial_factor;
    let s_theta = COLATITUDE_MM_PER_ARCSEC * cos2theta * radial_factor;
    let s_lambda = LONGITUDE_MM_PER_ARCSEC * costheta * longitude_factor;

    // Local radial/north/east displacement in metres (north = -S_theta).
    let up = s_r / MM_PER_M;
    let north = -s_theta / MM_PER_M;
    let east = s_lambda / MM_PER_M;

    // Geocentric ENU -> ECEF, matching the solid-earth tide convention.
    [
        up * cosla * cosphi - east * sinla - north * sinphi * cosla,
        up * sinla * cosphi + east * cosla - north * sinphi * sinla,
        up * sinphi + north * cosphi,
    ]
}

/// IERS (2018) linear secular mean pole at the given UTC epoch, in arcseconds.
fn mean_pole_arcsec(year: i32, month: i32, day: i32, fhr: f64) -> (f64, f64) {
    // Epoch in years since J2000.0 (RTKLIB-compatible Julian-year measure).
    let (jd0, jd1) = cal2jd(year, month, day);
    let years = ((jd0 - J2000_JD) + jd1 + fhr / 24.0) / DAYS_PER_JULIAN_YEAR;

    let x_bar_mas = MEAN_POLE_X0_MAS + MEAN_POLE_X_RATE_MAS_PER_YR * years;
    let y_bar_mas = MEAN_POLE_Y0_MAS + MEAN_POLE_Y_RATE_MAS_PER_YR * years;

    // Milliarcseconds -> arcseconds.
    (x_bar_mas / 1000.0, y_bar_mas / 1000.0)
}
