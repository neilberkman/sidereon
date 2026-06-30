//! Analytic low-precision Sun and Moon positions in ECI and ECEF.
//!
//! The Earth-centered-inertial series below are the standard low-precision
//! analytic solar/lunar formulae (Montenbruck & Gill, "Satellite Orbits",
//! sec. 3.3.2 / 3.3.3) as carried by common open GNSS code. They take the time
//! argument `t` in Julian centuries of Terrestrial Time since the J2000.0 epoch
//! (2000-01-01T12:00:00 TT). The five fundamental (Delaunay) arguments
//! `{l, l', F, D, Omega}` are evaluated by [`ast_args`] from the IAU 1980
//! nutation series coefficients.
//!
//! The series are referred to the mean equator and equinox of date (their mean
//! longitude and obliquity are of-date), so the resulting vectors are rotated
//! into the Earth-fixed (ITRS) frame with
//! [`crate::astro::frames::transforms::mean_of_date_to_itrs_matrix`] (IAU 2000A
//! nutation + GAST). Precession is already implicit in the of-date series and is
//! deliberately NOT re-applied; the full GCRS->ITRS transform would double-count
//! it (about 0.37 deg of Sun-direction error at epoch 2026). This mirrors the
//! GMST/GAST + nutation rotation the analytic references consume the series with.

use crate::astro::constants::astro::MONTENBRUCK_AU_M;
use crate::astro::constants::earth::WGS84_A_M;
use crate::astro::constants::time::{DAYS_PER_JULIAN_CENTURY, J2000_JD};
use crate::astro::constants::units::{ARCSEC_TO_RAD, DEG_TO_RAD};
use crate::astro::frames::transforms::{
    mat3_vec3_mul_unchecked, mean_of_date_to_itrs_matrix, FrameTransformError,
};
use crate::astro::time::scales::TimeScales;
use crate::validate;

/// Sun and Moon positions (metres) in a single frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SunMoon {
    /// Sun position vector (m).
    pub sun: [f64; 3],
    /// Moon position vector (m).
    pub moon: [f64; 3],
}

/// Error returned by Sun/Moon ephemeris helpers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SunMoonError {
    /// A public ephemeris input was non-finite or outside its valid domain.
    #[error("invalid Sun/Moon input {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    /// The ECI-to-ECEF frame rotation rejected the supplied time scales.
    #[error("Sun/Moon frame transform failed: {0}")]
    FrameTransform(#[from] FrameTransformError),
}

/// Fundamental (Delaunay) arguments `f = {l, l', F, D, Omega}` in radians for
/// the given time `t` (Julian centuries TT since J2000). IAU 1980 series.
fn ast_args(t: f64) -> [f64; 5] {
    // Coefficients for IAU 1980 nutation: l, l', F, D, OMG.
    const FC: [[f64; 5]; 5] = [
        [
            134.96340251,
            1717915923.2178,
            31.8792,
            0.051635,
            -0.00024470,
        ],
        [357.52910918, 129596581.0481, -0.5532, 0.000136, -0.00001149],
        [
            93.27209062,
            1739527262.8478,
            -12.7512,
            -0.001037,
            0.00000417,
        ],
        [
            297.85019547,
            1602961601.2090,
            -6.3706,
            0.006593,
            -0.00003169,
        ],
        [125.04455501, -6962890.2665, 7.4722, 0.007702, -0.00005939],
    ];

    let mut tt = [0.0_f64; 4];
    tt[0] = t;
    for i in 1..4 {
        tt[i] = tt[i - 1] * t;
    }

    let mut f = [0.0_f64; 5];
    for i in 0..5 {
        let mut v = FC[i][0] * 3600.0;
        for j in 0..4 {
            v += FC[i][j + 1] * tt[j];
        }
        f[i] = (v * ARCSEC_TO_RAD).rem_euclid(2.0 * std::f64::consts::PI);
    }
    f
}

/// Analytic Sun and Moon position in ECI (metres) for time `t` in Julian
/// centuries of TT since J2000.0.
pub fn sun_moon_eci(t: f64) -> Result<SunMoon, SunMoonError> {
    validate::finite(t, "t").map_err(map_sun_moon_input)?;
    validate_sun_moon(sun_moon_eci_unchecked(t))
}

fn sun_moon_eci_unchecked(t: f64) -> SunMoon {
    let f = ast_args(t);

    // Obliquity of the ecliptic (deg -> rad).
    let eps = 23.439291 - 0.0130042 * t;
    let sine = (eps * DEG_TO_RAD).sin();
    let cose = (eps * DEG_TO_RAD).cos();

    // Sun position in ECI.
    let ms = 357.5277233 + 35999.05034 * t;
    let ls = 280.460
        + 36000.770 * t
        + 1.914666471 * (ms * DEG_TO_RAD).sin()
        + 0.019994643 * (2.0 * ms * DEG_TO_RAD).sin();
    let rs = MONTENBRUCK_AU_M
        * (1.000140612
            - 0.016708617 * (ms * DEG_TO_RAD).cos()
            - 0.000139589 * (2.0 * ms * DEG_TO_RAD).cos());
    let sinl = (ls * DEG_TO_RAD).sin();
    let cosl = (ls * DEG_TO_RAD).cos();
    let sun = [rs * cosl, rs * cose * sinl, rs * sine * sinl];

    // Moon position in ECI.
    let lm = 218.32 + 481267.883 * t + 6.29 * f[0].sin() - 1.27 * (f[0] - 2.0 * f[3]).sin()
        + 0.66 * (2.0 * f[3]).sin()
        + 0.21 * (2.0 * f[0]).sin()
        - 0.19 * f[1].sin()
        - 0.11 * (2.0 * f[2]).sin();
    let pm = 5.13 * f[2].sin() + 0.28 * (f[0] + f[2]).sin()
        - 0.28 * (f[2] - f[0]).sin()
        - 0.17 * (f[2] - 2.0 * f[3]).sin();
    let rm = WGS84_A_M
        / ((0.9508
            + 0.0518 * f[0].cos()
            + 0.0095 * (f[0] - 2.0 * f[3]).cos()
            + 0.0078 * (2.0 * f[3]).cos()
            + 0.0028 * (2.0 * f[0]).cos())
            * DEG_TO_RAD)
            .sin();
    let sinlm = (lm * DEG_TO_RAD).sin();
    let coslm = (lm * DEG_TO_RAD).cos();
    let sinp = (pm * DEG_TO_RAD).sin();
    let cosp = (pm * DEG_TO_RAD).cos();
    let moon = [
        rm * cosp * coslm,
        rm * (cose * cosp * sinlm - sine * sinp),
        rm * (sine * cosp * sinlm + cose * sinp),
    ];

    SunMoon { sun, moon }
}

/// Analytic Sun and Moon ECI positions (metres) for a resolved time, forming the
/// Julian-century TT argument [`sun_moon_eci`] expects from the precise
/// [`TimeScales`].
///
/// This is the time-tagged entry point [`sun_moon_ecef`] rotates into ITRS. It is
/// public so a caller holding a [`TimeScales`] (e.g. one per epoch of a grid) can
/// obtain the ECI vectors without re-deriving the century argument; the
/// `jd_tt -> t -> sun_moon_eci` chain is identical to the one `sun_moon_ecef`
/// consumes, so the two stay consistent.
pub fn sun_moon_eci_at(ts: &TimeScales) -> Result<SunMoon, SunMoonError> {
    validate::finite(ts.jd_tt, "jd_tt").map_err(map_sun_moon_input)?;
    let t = (ts.jd_tt - J2000_JD) / DAYS_PER_JULIAN_CENTURY;
    validate_sun_moon(sun_moon_eci_unchecked(t))
}

/// Analytic Sun and Moon geocentric positions in the Earth-fixed (ITRS) frame
/// (metres) for the given UTC instant.
///
/// The analytic series ([`sun_moon_eci`]) are referred to the **mean equator and
/// equinox of date** (of-date mean longitude and obliquity), so they are rotated
/// to ITRS with [`mean_of_date_to_itrs_matrix`] (nutation + GAST). Precession is
/// already implicit in the of-date series and must NOT be applied a second time;
/// using the full GCRS->ITRS transform here would double-count precession (about
/// 0.37 deg of Sun-direction error at epoch 2026).
pub fn sun_moon_ecef(ts: &TimeScales) -> Result<SunMoon, SunMoonError> {
    let eci = sun_moon_eci_at(ts)?;
    let r = mean_of_date_to_itrs_matrix(ts)?;
    let sun = mat3_vec3_mul_unchecked(&r, &eci.sun);
    let moon = mat3_vec3_mul_unchecked(&r, &eci.moon);
    validate_sun_moon(SunMoon { sun, moon })
}

fn validate_sun_moon(value: SunMoon) -> Result<SunMoon, SunMoonError> {
    for (name, vector) in [("sun", &value.sun), ("moon", &value.moon)] {
        for (idx, component) in vector.iter().enumerate() {
            validate::finite(*component, vector_field(name, idx)).map_err(map_sun_moon_input)?;
        }
    }
    Ok(value)
}

fn vector_field(name: &str, idx: usize) -> &'static str {
    match (name, idx) {
        ("sun", 0) => "sun[0]",
        ("sun", 1) => "sun[1]",
        ("sun", 2) => "sun[2]",
        ("moon", 0) => "moon[0]",
        ("moon", 1) => "moon[1]",
        ("moon", 2) => "moon[2]",
        _ => "component",
    }
}

fn map_sun_moon_input(error: validate::FieldError) -> SunMoonError {
    SunMoonError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_moon_eci_rejects_non_finite_time() {
        for t in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                sun_moon_eci(t),
                Err(SunMoonError::InvalidInput {
                    field: "t",
                    reason: "not finite"
                })
            );
        }
    }

    #[test]
    fn sun_moon_eci_at_rejects_non_finite_jd_tt() {
        let mut ts = TimeScales::from_utc(2026, 4, 30, 9, 45, 0.0).expect("valid UTC instant");
        ts.jd_tt = f64::NAN;

        assert_eq!(
            sun_moon_eci_at(&ts),
            Err(SunMoonError::InvalidInput {
                field: "jd_tt",
                reason: "not finite"
            })
        );
    }

    #[test]
    fn sun_moon_eci_rejects_non_finite_outputs() {
        assert!(matches!(
            sun_moon_eci(f64::MAX),
            Err(SunMoonError::InvalidInput {
                field: "sun[0]" | "sun[1]" | "sun[2]" | "moon[0]" | "moon[1]" | "moon[2]",
                reason: "not finite"
            })
        ));
    }
}
