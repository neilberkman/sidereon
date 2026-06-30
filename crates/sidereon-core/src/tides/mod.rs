//! Solid-earth tide station displacement (IERS Conventions, Chapter 7).
//!
//! [`solid_earth_tide`] computes the tidal displacement of an Earth-fixed (ITRF)
//! GNSS station caused by the lunar and solar gravitational attraction. It is a
//! derived work of the IERS Conventions (2010) reference routine
//! `DEHANTTIDEINEL.F` (and its companion routines `ST1IDIU`, `ST1ISEM`,
//! `ST1L1`, `STEP2DIU`, `STEP2LON`, `CAL2JD`, `DAT`), reproduced here in Rust.
//!
//! How this derived work is based upon and differs from the original Software:
//!
//! * It is a line-for-line Rust translation of the in-phase degree-2/degree-3
//!   displacement, the out-of-phase corrections (`ST1IDIU`, `ST1ISEM`), the
//!   latitude-dependence correction (`ST1L1`), and the frequency-dependent
//!   step-2 diurnal/long-period band corrections (`STEP2DIU`, `STEP2LON`),
//!   evaluating the identical Love/Shida numbers, Doodson/argument tables, and
//!   leap-second table.
//! * It keeps the permanent (mean) tide deformation: the original routine's
//!   commented-out "Step 3" permanent-tide removal is left disabled, matching
//!   the ITRF/IGS conform-to-mean-tide convention.
//! * The routine names are changed from the IERS originals (per the IERS
//!   Conventions Software License), and the Fortran subroutine structure is
//!   inlined into private helpers.
//!
//! The Sun and Moon geocentric positions are inputs (metres, ECEF/ITRF); the
//! caller supplies them, e.g. from [`crate::astro::bodies::sun_moon_ecef`].
//!
//! IERS Conventions Software License: permission is granted to use this software
//! for any purpose, including commercial applications, free of charge, and to
//! distribute derived works subject to the conditions reproduced above. Results
//! obtained with this software acknowledge use of the IERS Conventions software.

#[cfg(all(test, sidereon_repo_tests))]
mod tests;

mod ocean;
mod pole;
pub use ocean::{ocean_tide_loading, OceanLoadingBlq, NUM_OCEAN_CONSTITUENTS};
pub use pole::solid_earth_pole_tide;

use crate::astro::constants::models::iers::SOLID_TIDE_EARTH_RADIUS_M;
use crate::astro::constants::time::{
    DAYS_PER_JULIAN_CENTURY, J2000_JD, SECONDS_PER_DAY, TT_MINUS_TAI_S,
};
use crate::astro::constants::units::{DEG_TO_RAD, KM_TO_M};
use crate::astro::math::vec3::{dot3_ref as dot, norm3_ref as norm8};
use crate::validate::{self, FieldError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TideInputErrorKind {
    Missing,
    NonFinite,
    NotPositive,
    Negative,
    OutOfRange,
    FloatParse,
    IntParse,
    InvalidCivilDate,
    InvalidCivilTime,
}

impl core::fmt::Display for TideInputErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Missing => "missing",
            Self::NonFinite => "not finite",
            Self::NotPositive => "not positive",
            Self::Negative => "negative",
            Self::OutOfRange => "out of range",
            Self::FloatParse => "invalid float",
            Self::IntParse => "invalid integer",
            Self::InvalidCivilDate => "invalid civil date",
            Self::InvalidCivilTime => "invalid civil time",
        })
    }
}

impl From<&FieldError> for TideInputErrorKind {
    fn from(error: &FieldError) -> Self {
        match error {
            FieldError::Missing { .. } => Self::Missing,
            FieldError::NonFinite { .. } => Self::NonFinite,
            FieldError::NotPositive { .. } => Self::NotPositive,
            FieldError::Negative { .. } => Self::Negative,
            FieldError::OutOfRange { .. } => Self::OutOfRange,
            FieldError::FloatParse { .. } => Self::FloatParse,
            FieldError::IntParse { .. } => Self::IntParse,
            FieldError::InvalidCivilDate { .. } => Self::InvalidCivilDate,
            FieldError::InvalidCivilTime { .. } => Self::InvalidCivilTime,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TideError {
    #[error("invalid solid-earth tide input {field}: {kind}")]
    InvalidInput {
        field: &'static str,
        kind: TideInputErrorKind,
    },
}

fn invalid_tide_input(error: FieldError) -> TideError {
    TideError::InvalidInput {
        field: error.field(),
        kind: (&error).into(),
    }
}

/// Solid-earth tide displacement of an ITRF station, in metres (ECEF).
///
/// Arguments mirror the IERS reference routine:
/// * `xsta` - geocentric station position (m, ITRF).
/// * `year`, `month`, `day` - UTC calendar date.
/// * `fhr` - UTC fractional hour of the day (hour + min/60 + sec/3600).
/// * `xsun` - geocentric Sun position (m, ECEF).
/// * `xmon` - geocentric Moon position (m, ECEF).
///
/// Returns the displacement vector `dxtide` (m, geocentric ITRF). The permanent
/// (mean) tide deformation is retained (ITRF/IGS convention).
///
/// Returns [`TideError`] when inputs are non-finite or geometrically
/// degenerate: the station vector must be non-zero and non-polar, and Sun/Moon
/// vectors must be non-zero.
pub fn solid_earth_tide(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    xsun: &[f64; 3],
    xmon: &[f64; 3],
) -> Result<[f64; 3], TideError> {
    validate_tide_domain(xsta, year, month, day, fhr, xsun, xmon)?;
    Ok(solid_earth_tide_unchecked(
        xsta, year, month, day, fhr, xsun, xmon,
    ))
}

fn validate_tide_domain(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    xsun: &[f64; 3],
    xmon: &[f64; 3],
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
    validate::finite_vec3(*xsun, "sun position").map_err(invalid_tide_input)?;
    validate::finite_vec3(*xmon, "moon position").map_err(invalid_tide_input)?;

    validate::finite_positive(norm8(xsta), "station radius").map_err(invalid_tide_input)?;
    let station_horizontal_radius = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt();
    validate::finite_positive(station_horizontal_radius, "station horizontal radius")
        .map_err(invalid_tide_input)?;
    validate::finite_positive(norm8(xsun), "sun radius").map_err(invalid_tide_input)?;
    validate::finite_positive(norm8(xmon), "moon radius").map_err(invalid_tide_input)?;

    Ok(())
}

fn solid_earth_tide_unchecked(
    xsta: &[f64; 3],
    year: i32,
    month: i32,
    day: i32,
    fhr: f64,
    xsun: &[f64; 3],
    xmon: &[f64; 3],
) -> [f64; 3] {
    // Nominal second- and third-degree Love and Shida numbers.
    const H20: f64 = 0.6078;
    const L20: f64 = 0.0847;
    const H3: f64 = 0.292;
    const L3: f64 = 0.015;

    // Scalar product of station vector with Sun/Moon vector (SPROD).
    let rsta = norm8(xsta);
    let rsun = norm8(xsun);
    let rmon = norm8(xmon);
    let scs = dot(xsta, xsun);
    let scm = dot(xsta, xmon);
    let scsun = scs / rsta / rsun;
    let scmon = scm / rsta / rmon;

    // Latitude-corrected H2 and L2.
    let cosphi = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt() / rsta;
    let h2 = H20 - 0.0006 * (1.0 - 3.0 / 2.0 * cosphi * cosphi);
    let l2 = L20 + 0.0002 * (1.0 - 3.0 / 2.0 * cosphi * cosphi);

    // P2 term.
    let p2sun = 3.0 * (h2 / 2.0 - l2) * scsun * scsun - h2 / 2.0;
    let p2mon = 3.0 * (h2 / 2.0 - l2) * scmon * scmon - h2 / 2.0;

    // P3 term.
    let p3sun = 5.0 / 2.0 * (H3 - 3.0 * L3) * scsun.powi(3) + 3.0 / 2.0 * (L3 - H3) * scsun;
    let p3mon = 5.0 / 2.0 * (H3 - 3.0 * L3) * scmon.powi(3) + 3.0 / 2.0 * (L3 - H3) * scmon;

    // Term in direction of Sun/Moon vector.
    let x2sun = 3.0 * l2 * scsun;
    let x2mon = 3.0 * l2 * scmon;
    let x3sun = 3.0 * L3 / 2.0 * (5.0 * scsun * scsun - 1.0);
    let x3mon = 3.0 * L3 / 2.0 * (5.0 * scmon * scmon - 1.0);

    // Factors for Sun/Moon (IAU current best estimates).
    const MASS_RATIO_SUN: f64 = 332946.0482;
    const MASS_RATIO_MOON: f64 = 0.0123000371;
    const RE: f64 = SOLID_TIDE_EARTH_RADIUS_M;
    let fac2sun = MASS_RATIO_SUN * RE * (RE / rsun).powi(3);
    let fac2mon = MASS_RATIO_MOON * RE * (RE / rmon).powi(3);
    let fac3sun = fac2sun * (RE / rsun);
    let fac3mon = fac2mon * (RE / rmon);

    // Total in-phase degree-2/degree-3 displacement.
    let mut dxtide = [0.0_f64; 3];
    for i in 0..3 {
        dxtide[i] = fac2sun * (x2sun * xsun[i] / rsun + p2sun * xsta[i] / rsta)
            + fac2mon * (x2mon * xmon[i] / rmon + p2mon * xsta[i] / rsta)
            + fac3sun * (x3sun * xsun[i] / rsun + p3sun * xsta[i] / rsta)
            + fac3mon * (x3mon * xmon[i] / rmon + p3mon * xsta[i] / rsta);
    }

    // Out-of-phase corrections (diurnal, semi-diurnal) and latitude dependence.
    let c = st1idiu(xsta, xsun, xmon, fac2sun, fac2mon);
    for i in 0..3 {
        dxtide[i] += c[i];
    }
    let c = st1isem(xsta, xsun, xmon, fac2sun, fac2mon);
    for i in 0..3 {
        dxtide[i] += c[i];
    }
    let c = st1l1(xsta, xsun, xmon, fac2sun, fac2mon);
    for i in 0..3 {
        dxtide[i] += c[i];
    }

    // Step 2 corrections need the date in Julian centuries (TT).
    let (jjm0, jjm1) = cal2jd(year, month, day);
    let fhrd = fhr / 24.0;
    let mut t = ((jjm0 - J2000_JD) + jjm1 + fhrd) / DAYS_PER_JULIAN_CENTURY;
    let dtt = dat(year, month, day) + TT_MINUS_TAI_S;
    t += dtt / (SECONDS_PER_DAY * DAYS_PER_JULIAN_CENTURY);

    let c = step2diu(xsta, fhr, t);
    for i in 0..3 {
        dxtide[i] += c[i];
    }
    let c = step2lon(xsta, t);
    for i in 0..3 {
        dxtide[i] += c[i];
    }

    // Step 3 of the IERS routine, the permanent (zero-frequency) tide removal,
    // is intentionally not applied, so the permanent (mean) tide deformation is
    // retained (the ITRF/IGS conform-to-mean-tide convention; see module docs).
    dxtide
}

/// Out-of-phase part of the Love numbers, diurnal band (ST1IDIU).
fn st1idiu(
    xsta: &[f64; 3],
    xsun: &[f64; 3],
    xmon: &[f64; 3],
    fac2sun: f64,
    fac2mon: f64,
) -> [f64; 3] {
    const DHI: f64 = -0.0025;
    const DLI: f64 = -0.0007;
    let rsta = norm8(xsta);
    let sinphi = xsta[2] / rsta;
    let cosphi = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt() / rsta;
    let cos2phi = cosphi * cosphi - sinphi * sinphi;
    let sinla = xsta[1] / cosphi / rsta;
    let cosla = xsta[0] / cosphi / rsta;
    let rmon = norm8(xmon);
    let rsun = norm8(xsun);

    let drsun =
        -3.0 * DHI * sinphi * cosphi * fac2sun * xsun[2] * (xsun[0] * sinla - xsun[1] * cosla)
            / (rsun * rsun);
    let drmon =
        -3.0 * DHI * sinphi * cosphi * fac2mon * xmon[2] * (xmon[0] * sinla - xmon[1] * cosla)
            / (rmon * rmon);
    let dnsun = -3.0 * DLI * cos2phi * fac2sun * xsun[2] * (xsun[0] * sinla - xsun[1] * cosla)
        / (rsun * rsun);
    let dnmon = -3.0 * DLI * cos2phi * fac2mon * xmon[2] * (xmon[0] * sinla - xmon[1] * cosla)
        / (rmon * rmon);
    let desun = -3.0 * DLI * sinphi * fac2sun * xsun[2] * (xsun[0] * cosla + xsun[1] * sinla)
        / (rsun * rsun);
    let demon = -3.0 * DLI * sinphi * fac2mon * xmon[2] * (xmon[0] * cosla + xmon[1] * sinla)
        / (rmon * rmon);

    let dr = drsun + drmon;
    let dn = dnsun + dnmon;
    let de = desun + demon;

    [
        dr * cosla * cosphi - de * sinla - dn * sinphi * cosla,
        dr * sinla * cosphi + de * cosla - dn * sinphi * sinla,
        dr * sinphi + dn * cosphi,
    ]
}

/// Out-of-phase part of the Love numbers, semi-diurnal band (ST1ISEM).
fn st1isem(
    xsta: &[f64; 3],
    xsun: &[f64; 3],
    xmon: &[f64; 3],
    fac2sun: f64,
    fac2mon: f64,
) -> [f64; 3] {
    const DHI: f64 = -0.0022;
    const DLI: f64 = -0.0007;
    let rsta = norm8(xsta);
    let sinphi = xsta[2] / rsta;
    let cosphi = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt() / rsta;
    let sinla = xsta[1] / cosphi / rsta;
    let cosla = xsta[0] / cosphi / rsta;
    let costwola = cosla * cosla - sinla * sinla;
    let sintwola = 2.0 * cosla * sinla;
    let rmon = norm8(xmon);
    let rsun = norm8(xsun);

    let drsun = -3.0 / 4.0
        * DHI
        * cosphi
        * cosphi
        * fac2sun
        * ((xsun[0] * xsun[0] - xsun[1] * xsun[1]) * sintwola - 2.0 * xsun[0] * xsun[1] * costwola)
        / (rsun * rsun);
    let drmon = -3.0 / 4.0
        * DHI
        * cosphi
        * cosphi
        * fac2mon
        * ((xmon[0] * xmon[0] - xmon[1] * xmon[1]) * sintwola - 2.0 * xmon[0] * xmon[1] * costwola)
        / (rmon * rmon);
    let dnsun = 3.0 / 2.0
        * DLI
        * sinphi
        * cosphi
        * fac2sun
        * ((xsun[0] * xsun[0] - xsun[1] * xsun[1]) * sintwola - 2.0 * xsun[0] * xsun[1] * costwola)
        / (rsun * rsun);
    let dnmon = 3.0 / 2.0
        * DLI
        * sinphi
        * cosphi
        * fac2mon
        * ((xmon[0] * xmon[0] - xmon[1] * xmon[1]) * sintwola - 2.0 * xmon[0] * xmon[1] * costwola)
        / (rmon * rmon);
    let desun = -3.0 / 2.0
        * DLI
        * cosphi
        * fac2sun
        * ((xsun[0] * xsun[0] - xsun[1] * xsun[1]) * costwola + 2.0 * xsun[0] * xsun[1] * sintwola)
        / (rsun * rsun);
    let demon = -3.0 / 2.0
        * DLI
        * cosphi
        * fac2mon
        * ((xmon[0] * xmon[0] - xmon[1] * xmon[1]) * costwola + 2.0 * xmon[0] * xmon[1] * sintwola)
        / (rmon * rmon);

    let dr = drsun + drmon;
    let dn = dnsun + dnmon;
    let de = desun + demon;

    [
        dr * cosla * cosphi - de * sinla - dn * sinphi * cosla,
        dr * sinla * cosphi + de * cosla - dn * sinphi * sinla,
        dr * sinphi + dn * cosphi,
    ]
}

/// Latitude dependence of the Love numbers, part L^(1) (ST1L1).
fn st1l1(
    xsta: &[f64; 3],
    xsun: &[f64; 3],
    xmon: &[f64; 3],
    fac2sun: f64,
    fac2mon: f64,
) -> [f64; 3] {
    const L1D: f64 = 0.0012;
    const L1SD: f64 = 0.0024;
    let rsta = norm8(xsta);
    let sinphi = xsta[2] / rsta;
    let cosphi = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt() / rsta;
    let sinla = xsta[1] / cosphi / rsta;
    let cosla = xsta[0] / cosphi / rsta;
    let rmon = norm8(xmon);
    let rsun = norm8(xsun);

    // Diurnal band.
    let mut l1 = L1D;
    let dnsun = -l1 * sinphi * sinphi * fac2sun * xsun[2] * (xsun[0] * cosla + xsun[1] * sinla)
        / (rsun * rsun);
    let dnmon = -l1 * sinphi * sinphi * fac2mon * xmon[2] * (xmon[0] * cosla + xmon[1] * sinla)
        / (rmon * rmon);
    let desun = l1
        * sinphi
        * (cosphi * cosphi - sinphi * sinphi)
        * fac2sun
        * xsun[2]
        * (xsun[0] * sinla - xsun[1] * cosla)
        / (rsun * rsun);
    let demon = l1
        * sinphi
        * (cosphi * cosphi - sinphi * sinphi)
        * fac2mon
        * xmon[2]
        * (xmon[0] * sinla - xmon[1] * cosla)
        / (rmon * rmon);

    let de = 3.0 * (desun + demon);
    let dn = 3.0 * (dnsun + dnmon);

    let mut xcorsta = [
        -de * sinla - dn * sinphi * cosla,
        de * cosla - dn * sinphi * sinla,
        dn * cosphi,
    ];

    // Semi-diurnal band.
    l1 = L1SD;
    let costwola = cosla * cosla - sinla * sinla;
    let sintwola = 2.0 * cosla * sinla;

    let dnsun = -l1 / 2.0
        * sinphi
        * cosphi
        * fac2sun
        * ((xsun[0] * xsun[0] - xsun[1] * xsun[1]) * costwola + 2.0 * xsun[0] * xsun[1] * sintwola)
        / (rsun * rsun);
    let dnmon = -l1 / 2.0
        * sinphi
        * cosphi
        * fac2mon
        * ((xmon[0] * xmon[0] - xmon[1] * xmon[1]) * costwola + 2.0 * xmon[0] * xmon[1] * sintwola)
        / (rmon * rmon);
    let desun = -l1 / 2.0
        * sinphi
        * sinphi
        * cosphi
        * fac2sun
        * ((xsun[0] * xsun[0] - xsun[1] * xsun[1]) * sintwola - 2.0 * xsun[0] * xsun[1] * costwola)
        / (rsun * rsun);
    let demon = -l1 / 2.0
        * sinphi
        * sinphi
        * cosphi
        * fac2mon
        * ((xmon[0] * xmon[0] - xmon[1] * xmon[1]) * sintwola - 2.0 * xmon[0] * xmon[1] * costwola)
        / (rmon * rmon);

    let de = 3.0 * (desun + demon);
    let dn = 3.0 * (dnsun + dnmon);

    xcorsta[0] += -de * sinla - dn * sinphi * cosla;
    xcorsta[1] += de * cosla - dn * sinphi * sinla;
    xcorsta[2] += dn * cosphi;
    xcorsta
}

/// In-phase / out-of-phase frequency-dependent corrections, diurnal band
/// (STEP2DIU). `fhr` is UTC fractional hour, `t` is Julian centuries (TT).
fn step2diu(xsta: &[f64; 3], fhr: f64, t: f64) -> [f64; 3] {
    // DATDI(9,31): {l, l', F, D, Omega(Ps), Adr, Adi, Anr, Ani} per wave.
    #[rustfmt::skip]
    const DATDI: [[f64; 9]; 31] = [
        [-3.0, 0.0, 2.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
        [-3.0, 2.0, 0.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
        [-2.0, 0.0, 1.0, -1.0, 0.0, -0.02, 0.0, 0.0, 0.0],
        [-2.0, 0.0, 1.0, 0.0, 0.0, -0.08, 0.0, -0.01, 0.01],
        [-2.0, 2.0, -1.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, -1.0, 0.0, -0.10, 0.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0, 0.0, -0.51, 0.0, -0.02, 0.03],
        [-1.0, 2.0, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
        [0.0, -2.0, 1.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.06, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0, 0.0, 0.01, 0.0, 0.0, 0.0],
        [0.0, 2.0, -1.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
        [1.0, -3.0, 0.0, 0.0, 1.0, -0.06, 0.0, 0.0, 0.0],
        [1.0, -2.0, 0.0, -1.0, 0.0, 0.01, 0.0, 0.0, 0.0],
        [1.0, -2.0, 0.0, 0.0, 0.0, -1.23, -0.07, 0.06, 0.01],
        [1.0, -1.0, 0.0, 0.0, -1.0, 0.02, 0.0, 0.0, 0.0],
        [1.0, -1.0, 0.0, 0.0, 1.0, 0.04, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, -1.0, 0.0, -0.22, 0.01, 0.01, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 12.00, -0.80, -0.67, -0.03],
        [1.0, 0.0, 0.0, 1.0, 0.0, 1.73, -0.12, -0.10, 0.0],
        [1.0, 0.0, 0.0, 2.0, 0.0, -0.04, 0.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, -1.0, -0.50, -0.01, 0.03, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, 0.01, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.0, -1.0, -0.01, 0.0, 0.0, 0.0],
        [1.0, 2.0, -2.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 0.0, 0.0, -0.11, 0.01, 0.01, 0.0],
        [2.0, -2.0, 1.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0],
        [2.0, 0.0, -1.0, 0.0, 0.0, -0.02, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let mut s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * t) * t) * t;
    let mut tau = fhr * 15.0
        + 280.4606184
        + (36000.7700536 + (0.00038793 + -0.0000000258 * t) * t) * t
        + (-s);
    let pr = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007 * t) * t) * t) * t;
    s += pr;
    let mut h = 280.46645
        + (36000.7697489 + (0.00030322222 + (0.000000020 + -0.00000000654 * t) * t) * t) * t;
    let mut p = 83.35324312
        + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * t) * t) * t) * t;
    let mut zns = 234.95544499
        + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * t) * t) * t) * t;
    let mut ps = 282.93734098
        + (1.71945766667 + (0.00045688889 + (-0.00000001778 + -0.00000000334 * t) * t) * t) * t;

    s = s.rem_euclid(360.0);
    tau = tau.rem_euclid(360.0);
    h = h.rem_euclid(360.0);
    p = p.rem_euclid(360.0);
    zns = zns.rem_euclid(360.0);
    ps = ps.rem_euclid(360.0);

    let rsta = (xsta[0] * xsta[0] + xsta[1] * xsta[1] + xsta[2] * xsta[2]).sqrt();
    let sinphi = xsta[2] / rsta;
    let cosphi = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt() / rsta;
    let cosla = xsta[0] / cosphi / rsta;
    let sinla = xsta[1] / cosphi / rsta;
    let zla = xsta[1].atan2(xsta[0]);

    let mut xcorsta = [0.0_f64; 3];
    for w in &DATDI {
        let thetaf = (tau + w[0] * s + w[1] * h + w[2] * p + w[3] * zns + w[4] * ps) * DEG_TO_RAD;
        let dr = w[5] * 2.0 * sinphi * cosphi * (thetaf + zla).sin()
            + w[6] * 2.0 * sinphi * cosphi * (thetaf + zla).cos();
        let dn = w[7] * (cosphi * cosphi - sinphi * sinphi) * (thetaf + zla).sin()
            + w[8] * (cosphi * cosphi - sinphi * sinphi) * (thetaf + zla).cos();
        let de = w[7] * sinphi * (thetaf + zla).cos() - w[8] * sinphi * (thetaf + zla).sin();

        xcorsta[0] += dr * cosla * cosphi - de * sinla - dn * sinphi * cosla;
        xcorsta[1] += dr * sinla * cosphi + de * cosla - dn * sinphi * sinla;
        xcorsta[2] += dr * sinphi + dn * cosphi;
    }
    for v in &mut xcorsta {
        *v /= KM_TO_M;
    }
    xcorsta
}

/// In-phase / out-of-phase frequency-dependent corrections, long-period band
/// (STEP2LON). `t` is Julian centuries (TT).
fn step2lon(xsta: &[f64; 3], t: f64) -> [f64; 3] {
    #[rustfmt::skip]
    const DATDI: [[f64; 9]; 5] = [
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.47, 0.23, 0.16, 0.07],
        [0.0, 2.0, 0.0, 0.0, 0.0, -0.20, -0.12, -0.11, -0.05],
        [1.0, 0.0, -1.0, 0.0, 0.0, -0.11, -0.08, -0.09, -0.04],
        [2.0, 0.0, 0.0, 0.0, 0.0, -0.13, -0.11, -0.15, -0.07],
        [2.0, 0.0, 0.0, 1.0, 0.0, -0.05, -0.05, -0.06, -0.03],
    ];
    let mut s = 218.31664563 + (481267.88194 + (-0.0014663889 + 0.00000185139 * t) * t) * t;
    let pr = (1.396971278 + (0.000308889 + (0.000000021 + 0.000000007 * t) * t) * t) * t;
    s += pr;
    let mut h = 280.46645
        + (36000.7697489 + (0.00030322222 + (0.000000020 + -0.00000000654 * t) * t) * t) * t;
    let mut p = 83.35324312
        + (4069.01363525 + (-0.01032172222 + (-0.0000124991 + 0.00000005263 * t) * t) * t) * t;
    let mut zns = 234.95544499
        + (1934.13626197 + (-0.00207561111 + (-0.00000213944 + 0.00000001650 * t) * t) * t) * t;
    let mut ps = 282.93734098
        + (1.71945766667 + (0.00045688889 + (-0.00000001778 + -0.00000000334 * t) * t) * t) * t;

    let rsta = (xsta[0] * xsta[0] + xsta[1] * xsta[1] + xsta[2] * xsta[2]).sqrt();
    let sinphi = xsta[2] / rsta;
    let cosphi = (xsta[0] * xsta[0] + xsta[1] * xsta[1]).sqrt() / rsta;
    let cosla = xsta[0] / cosphi / rsta;
    let sinla = xsta[1] / cosphi / rsta;

    s = s.rem_euclid(360.0);
    h = h.rem_euclid(360.0);
    p = p.rem_euclid(360.0);
    zns = zns.rem_euclid(360.0);
    ps = ps.rem_euclid(360.0);

    let mut xcorsta = [0.0_f64; 3];
    for w in &DATDI {
        let thetaf = (w[0] * s + w[1] * h + w[2] * p + w[3] * zns + w[4] * ps) * DEG_TO_RAD;
        let dr = w[5] * (3.0 * sinphi * sinphi - 1.0) / 2.0 * thetaf.cos()
            + w[7] * (3.0 * sinphi * sinphi - 1.0) / 2.0 * thetaf.sin();
        let dn = w[6] * (cosphi * sinphi * 2.0) * thetaf.cos()
            + w[8] * (cosphi * sinphi * 2.0) * thetaf.sin();
        let de = 0.0;

        xcorsta[0] += dr * cosla * cosphi - de * sinla - dn * sinphi * cosla;
        xcorsta[1] += dr * sinla * cosphi + de * cosla - dn * sinphi * sinla;
        xcorsta[2] += dr * sinphi + dn * cosphi;
    }
    for v in &mut xcorsta {
        *v /= KM_TO_M;
    }
    xcorsta
}

/// Gregorian calendar date -> (MJD epoch 2400000.5, MJD) (SOFA CAL2JD).
///
/// This is a SOFA parity adapter, deliberately NOT routed through
/// [`crate::astro::time::civil`]: the solid-Earth/ocean/pole tide models are
/// validated bit-for-bit against the IERS/SOFA reference (the
/// `ocean_loading_oracle` test), so the calendar-to-MJD step must reproduce
/// SOFA's `iauCal2jd` exactly. It is kept local under this tides-specific name
/// so it is not mistaken for a duplicate of the canonical civil conversions and
/// is never consolidated into them.
fn cal2jd(iy: i32, im: i32, id: i32) -> (f64, f64) {
    let my = (im - 14) / 12;
    let iypmy = iy + my;
    let djm0 = 2400000.5;
    let djm = ((1461 * (iypmy + 4800)) / 4 + (367 * (im - 2 - 12 * my)) / 12
        - (3 * ((iypmy + 4900) / 100)) / 4
        + id
        - 2432076) as f64;
    (djm0, djm)
}

/// TAI-UTC (Delta(AT)) in seconds for the given date (SOFA DAT, leap-second
/// table only; the four golden dates are all post-1972 so the pre-1972 drift
/// branch is not exercised, but it is retained for completeness).
fn dat(iy: i32, im: i32, _id: i32) -> f64 {
    // Post-1972 leap-second table: (year, month, Delta(AT) seconds).
    const IDAT: [(i32, i32, f64); 28] = [
        (1972, 1, 10.0),
        (1972, 7, 11.0),
        (1973, 1, 12.0),
        (1974, 1, 13.0),
        (1975, 1, 14.0),
        (1976, 1, 15.0),
        (1977, 1, 16.0),
        (1978, 1, 17.0),
        (1979, 1, 18.0),
        (1980, 1, 19.0),
        (1981, 7, 20.0),
        (1982, 7, 21.0),
        (1983, 7, 22.0),
        (1985, 7, 23.0),
        (1988, 1, 24.0),
        (1990, 1, 25.0),
        (1991, 1, 26.0),
        (1992, 7, 27.0),
        (1993, 7, 28.0),
        (1994, 7, 29.0),
        (1996, 1, 30.0),
        (1997, 7, 31.0),
        (1999, 1, 32.0),
        (2006, 1, 33.0),
        (2009, 1, 34.0),
        (2012, 7, 35.0),
        (2015, 7, 36.0),
        (2017, 1, 37.0),
    ];
    let m = 12 * iy + im;
    let mut da = IDAT[0].2;
    for &(y, mo, d) in &IDAT {
        if m >= 12 * y + mo {
            da = d;
        }
    }
    da
}
