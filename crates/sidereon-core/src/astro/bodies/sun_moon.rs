//! Analytic low-precision Sun and Moon positions: mean of date, GCRS and ECEF.
//!
//! The Earth-centered-inertial series below are the standard low-precision
//! analytic solar/lunar formulae (Montenbruck & Gill, "Satellite Orbits",
//! sec. 3.3.2 / 3.3.3) as carried by common open GNSS code. They take the time
//! argument `t` in Julian centuries of Terrestrial Time since the J2000.0 epoch
//! (2000-01-01T12:00:00 TT). The five fundamental (Delaunay) arguments
//! `{l, l', F, D, Omega}` are evaluated by `ast_args` from the IAU 1980
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
//!
//! [`sun_moon_eci`] and [`sun_moon_eci_at`] return the series in that mean of
//! date frame. They are not GCRS vectors: GCRS (the frame of the GCRF states of
//! [`crate::astro::state::CartesianState`]) differs from the mean equator and
//! equinox of date by the precession accumulated since J2000.0 plus the frame
//! bias, about 0.37 deg at 2026, growing by about 0.014 deg per year. For use
//! with GCRF states, [`sun_moon_gcrs`], [`sun_moon_gcrs_at`] and
//! [`sun_moon_gcrs_at_tdb_seconds`] rotate the series to GCRS with
//! [`crate::astro::frames::transforms::mean_of_date_to_gcrs_matrix`], the
//! transpose of the IAU 2006 precession (Capitaine et al. 2003; Wallace &
//! Capitaine 2006) times the ICRS frame bias. The solar radiation pressure,
//! Earth radiation pressure and third-body force models use these GCRS vectors.
//!
//! Against the JPL DE440 Sun and Moon in GCRF (Orekit 13.1.10,
//! `tests/sun_moon_gcrs_orekit_oracle.rs`) the GCRS vectors stay within the
//! accuracy the low-precision series are published with, from 1990 to 2050;
//! read as GCRS without the rotation, the mean of date Sun is about 0.37 deg off
//! at 2026.

use crate::astro::constants::astro::MONTENBRUCK_AU_M;
use crate::astro::constants::earth::WGS84_A_M;
use crate::astro::constants::time::{DAYS_PER_JULIAN_CENTURY, J2000_JD, SECONDS_PER_DAY};
use crate::astro::constants::units::{ARCSEC_TO_RAD, DEG_TO_RAD};
use crate::astro::frames::transforms::{
    mat3_vec3_mul_unchecked, mean_of_date_to_gcrs_matrix_unchecked, mean_of_date_to_itrs_matrix,
    mean_of_date_to_itrs_matrix_with_polar_motion, FrameTransformError, PolarMotion,
};
use crate::astro::time::scales::{
    tdb_minus_tt_seconds_at_tt_jd, tt_from_tdb_j2000_seconds, TimeScales,
};
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
        /// Label from validation: `"t"`, `"jd_tt"`, `"jd_tdb"` or
        /// `"epoch_tdb_seconds"` for a non-finite time input, or `"sun[0]"`
        /// through `"sun[2]"` / `"moon[0]"` through `"moon[2]"` for a
        /// non-finite computed vector component.
        field: &'static str,
        /// Static reason copied from validation; this module's finite checks
        /// report `"not finite"` for a non-finite input or computed component.
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

/// Analytic Sun and Moon geocentric positions (metres) referred to the **mean
/// equator and equinox of date**, for time `t` in Julian centuries of TT since
/// J2000.0.
///
/// The vectors are in the frame the series are written in (Montenbruck & Gill,
/// "Satellite Orbits", 3.3.2 and 3.3.3), not GCRS: read as GCRS they are off by
/// the precession since J2000.0 plus the frame bias (about 0.37 deg at 2026).
/// Use [`sun_moon_gcrs`] for vectors to combine with GCRF states, and
/// [`sun_moon_ecef`] for Earth-fixed vectors.
pub fn sun_moon_eci(t: f64) -> Result<SunMoon, SunMoonError> {
    validate::finite(t, "t").map_err(map_sun_moon_input)?;
    validate_sun_moon(sun_moon_eci_unchecked(t))
}

fn sun_moon_eci_unchecked(t: f64) -> SunMoon {
    let f = ast_args(t);

    // Obliquity of the ecliptic (deg -> rad).
    let eps = 23.439291 - 0.0130042 * t;
    let sine = libm::sin(eps * DEG_TO_RAD);
    let cose = libm::cos(eps * DEG_TO_RAD);

    // Sun position in ECI.
    let ms = 357.5277233 + 35999.05034 * t;
    let ls = 280.460
        + 36000.770 * t
        + 1.914666471 * libm::sin(ms * DEG_TO_RAD)
        + 0.019994643 * libm::sin(2.0 * ms * DEG_TO_RAD);
    let rs = MONTENBRUCK_AU_M
        * (1.000140612
            - 0.016708617 * libm::cos(ms * DEG_TO_RAD)
            - 0.000139589 * libm::cos(2.0 * ms * DEG_TO_RAD));
    let sinl = libm::sin(ls * DEG_TO_RAD);
    let cosl = libm::cos(ls * DEG_TO_RAD);
    let sun = [rs * cosl, rs * cose * sinl, rs * sine * sinl];

    // Moon position in ECI.
    let lm = 218.32 + 481267.883 * t + 6.29 * libm::sin(f[0]) - 1.27 * libm::sin(f[0] - 2.0 * f[3])
        + 0.66 * libm::sin(2.0 * f[3])
        + 0.21 * libm::sin(2.0 * f[0])
        - 0.19 * libm::sin(f[1])
        - 0.11 * libm::sin(2.0 * f[2]);
    let pm = 5.13 * libm::sin(f[2]) + 0.28 * libm::sin(f[0] + f[2])
        - 0.28 * libm::sin(f[2] - f[0])
        - 0.17 * libm::sin(f[2] - 2.0 * f[3]);
    // Geocentric distance is the equatorial radius over the sine of the
    // horizontal parallax. The parallax is under a degree, so sin(x) is
    // within 4e-5 of x and dropping the sine would move the Moon ~17 km
    // while still passing any coarse range check.
    let rm = WGS84_A_M
        / libm::sin(
            (0.9508
                + 0.0518 * libm::cos(f[0])
                + 0.0095 * libm::cos(f[0] - 2.0 * f[3])
                + 0.0078 * libm::cos(2.0 * f[3])
                + 0.0028 * libm::cos(2.0 * f[0]))
                * DEG_TO_RAD,
        );
    let sinlm = libm::sin(lm * DEG_TO_RAD);
    let coslm = libm::cos(lm * DEG_TO_RAD);
    let sinp = libm::sin(pm * DEG_TO_RAD);
    let cosp = libm::cos(pm * DEG_TO_RAD);
    let moon = [
        rm * cosp * coslm,
        rm * (cose * cosp * sinlm - sine * sinp),
        rm * (sine * cosp * sinlm + cose * sinp),
    ];

    SunMoon { sun, moon }
}

/// Analytic Sun and Moon positions (metres) referred to the **mean equator and
/// equinox of date** for a resolved time, forming the Julian-century TT
/// argument [`sun_moon_eci`] expects from the precise [`TimeScales`].
///
/// This is the time-tagged entry point [`sun_moon_ecef`] rotates into ITRS. It is
/// public so a caller holding a [`TimeScales`] (e.g. one per epoch of a grid) can
/// obtain the mean of date vectors without re-deriving the century argument; the
/// `jd_tt -> t -> sun_moon_eci` chain is identical to the one `sun_moon_ecef`
/// consumes, so the two stay consistent.
///
/// Like [`sun_moon_eci`], the vectors are mean of date, not GCRS; use
/// [`sun_moon_gcrs_at`] for vectors to combine with GCRF states.
pub fn sun_moon_eci_at(ts: &TimeScales) -> Result<SunMoon, SunMoonError> {
    validate::finite(ts.jd_tt, "jd_tt").map_err(map_sun_moon_input)?;
    let t = (ts.jd_tt - J2000_JD) / DAYS_PER_JULIAN_CENTURY;
    validate_sun_moon(sun_moon_eci_unchecked(t))
}

/// Analytic Sun and Moon geocentric positions (metres) in GCRS, for time
/// `t_tt_centuries` in Julian centuries of TT since J2000.0.
///
/// These are the vectors to use with GCRF states
/// ([`crate::astro::state::CartesianState`]): the [`sun_moon_eci`] series
/// rotated from the mean equator and equinox of date to GCRS by
/// [`crate::astro::frames::transforms::mean_of_date_to_gcrs_matrix`], the
/// transpose of the IAU 2006 precession times the ICRS frame bias (Capitaine et
/// al. 2003; Wallace & Capitaine 2006).
///
/// The series takes TT. The precession takes TDB, formed here from
/// `t_tt_centuries` with the TDB - TT series [`TimeScales`] uses; that offset
/// stays under 1.71 ms, over which the precession turns by at most 1.4e-14 rad.
///
/// The positions keep the accuracy of the low-precision series (Montenbruck &
/// Gill, "Satellite Orbits", 3.3.2 and 3.3.3), 0.01 deg for the Sun and a few
/// tenths of a degree for the Moon; the rotation agrees with Orekit's to under
/// 1e-12 rad. Use [`sun_moon_gcrs_at_tdb_seconds`] for an epoch in TDB seconds
/// since J2000.0, as propagation states carry it.
pub fn sun_moon_gcrs(t_tt_centuries: f64) -> Result<SunMoon, SunMoonError> {
    validate::finite(t_tt_centuries, "t").map_err(map_sun_moon_input)?;
    let jd_tt = J2000_JD + t_tt_centuries * DAYS_PER_JULIAN_CENTURY;
    let jd_tdb = jd_tt + tdb_minus_tt_seconds_at_tt_jd(jd_tt) / SECONDS_PER_DAY;
    validate_sun_moon(rotate_to_gcrs(
        sun_moon_eci_unchecked(t_tt_centuries),
        jd_tdb,
    ))
}

/// Analytic Sun and Moon geocentric positions (metres) in GCRS for a resolved
/// time.
///
/// The series is evaluated at `ts.jd_tt` exactly as in [`sun_moon_eci_at`], and
/// rotated to GCRS with the precession at `ts.jd_tdb`, so the result is
/// [`crate::astro::frames::transforms::mean_of_date_to_gcrs_matrix`]`(ts)`
/// applied to [`sun_moon_eci_at`]`(ts)`. Only `jd_tt` and `jd_tdb` are read, so
/// time scales whose UT1 lies outside the UT1 table are accepted. These are
/// the vectors to use with GCRF states; see [`sun_moon_gcrs`].
pub fn sun_moon_gcrs_at(ts: &TimeScales) -> Result<SunMoon, SunMoonError> {
    validate::finite(ts.jd_tt, "jd_tt").map_err(map_sun_moon_input)?;
    validate::finite(ts.jd_tdb, "jd_tdb").map_err(map_sun_moon_input)?;
    let t = (ts.jd_tt - J2000_JD) / DAYS_PER_JULIAN_CENTURY;
    validate_sun_moon(rotate_to_gcrs(sun_moon_eci_unchecked(t), ts.jd_tdb))
}

/// Rotate mean-of-date Sun/Moon positions to GCRS at `ts.jd_tdb`.
pub(crate) fn sun_moon_mean_to_gcrs(
    mean_of_date: SunMoon,
    ts: &TimeScales,
) -> Result<SunMoon, SunMoonError> {
    validate::finite(ts.jd_tdb, "jd_tdb").map_err(map_sun_moon_input)?;
    validate_sun_moon(rotate_to_gcrs(mean_of_date, ts.jd_tdb))
}

/// Analytic Sun and Moon geocentric positions (metres) in GCRS at an epoch in
/// TDB seconds since J2000.0, the time tag of propagation states
/// ([`crate::astro::state::CartesianState::epoch_tdb_seconds`]).
///
/// The series is evaluated at the TT of that instant (TDB converted with the
/// inverse of the TDB - TT series [`TimeScales`] uses) and rotated to GCRS with
/// the precession at the given TDB. The solar radiation pressure, Earth
/// radiation pressure and third-body force models take their Sun and Moon from
/// this function, so passing its vectors to their `acceleration_with_sun_moon`
/// methods reproduces `acceleration` bit for bit. See [`sun_moon_gcrs`].
pub fn sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds: f64) -> Result<SunMoon, SunMoonError> {
    validate::finite(epoch_tdb_seconds, "epoch_tdb_seconds").map_err(map_sun_moon_input)?;
    let t_tt_centuries =
        tt_from_tdb_j2000_seconds(epoch_tdb_seconds) / (DAYS_PER_JULIAN_CENTURY * SECONDS_PER_DAY);
    let jd_tdb = J2000_JD + epoch_tdb_seconds / SECONDS_PER_DAY;
    validate_sun_moon(rotate_to_gcrs(
        sun_moon_eci_unchecked(t_tt_centuries),
        jd_tdb,
    ))
}

fn rotate_to_gcrs(mean_of_date: SunMoon, jd_tdb: f64) -> SunMoon {
    let r = mean_of_date_to_gcrs_matrix_unchecked(jd_tdb);
    SunMoon {
        sun: mat3_vec3_mul_unchecked(&r, &mean_of_date.sun),
        moon: mat3_vec3_mul_unchecked(&r, &mean_of_date.moon),
    }
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
    sun_moon_ecef_with_polar_motion(ts, PolarMotion::ZERO)
}

/// Analytic Sun and Moon geocentric positions in the Earth-fixed (ITRS) frame
/// (metres), applying caller-supplied IERS polar motion in the same
/// mean-of-date-to-ITRS rotation used by the solid-tide force lane.
pub fn sun_moon_ecef_with_polar_motion(
    ts: &TimeScales,
    polar_motion: PolarMotion,
) -> Result<SunMoon, SunMoonError> {
    let eci = sun_moon_eci_at(ts)?;
    sun_moon_mean_to_ecef_with_polar_motion(eci, ts, polar_motion)
}

/// Rotate mean-of-date Sun/Moon positions to ITRS with zero polar motion.
pub(crate) fn sun_moon_mean_to_ecef(
    mean_of_date: SunMoon,
    ts: &TimeScales,
) -> Result<SunMoon, SunMoonError> {
    sun_moon_mean_to_ecef_with_polar_motion(mean_of_date, ts, PolarMotion::ZERO)
}

fn sun_moon_mean_to_ecef_with_polar_motion(
    mean_of_date: SunMoon,
    ts: &TimeScales,
    polar_motion: PolarMotion,
) -> Result<SunMoon, SunMoonError> {
    let r = if polar_motion == PolarMotion::ZERO {
        mean_of_date_to_itrs_matrix(ts)?
    } else {
        mean_of_date_to_itrs_matrix_with_polar_motion(ts, polar_motion)?
    };
    let sun = mat3_vec3_mul_unchecked(&r, &mean_of_date.sun);
    let moon = mat3_vec3_mul_unchecked(&r, &mean_of_date.moon);
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

    fn angle_deg(a: [f64; 3], b: [f64; 3]) -> f64 {
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        libm::atan2(
            libm::sqrt(cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]),
            dot,
        )
        .to_degrees()
    }

    fn relative_difference(a: [f64; 3], b: [f64; 3]) -> f64 {
        let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        libm::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
            / libm::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    }

    #[test]
    fn sun_moon_gcrs_at_is_the_mean_of_date_series_rotated_by_the_public_matrix() {
        use crate::astro::frames::transforms::{mat3_vec3_mul, mean_of_date_to_gcrs_matrix};
        for (year, month, day) in [(1990, 3, 1), (2026, 11, 23), (2049, 7, 15)] {
            let ts = TimeScales::from_utc(year, month, day, 18, 0, 0.0).expect("valid UTC");
            let mean = sun_moon_eci_at(&ts).expect("mean of date");
            let r = mean_of_date_to_gcrs_matrix(&ts).expect("rotation");
            let gcrs = sun_moon_gcrs_at(&ts).expect("GCRS");
            assert_eq!(gcrs.sun, mat3_vec3_mul(&r, &mean.sun).expect("sun"));
            assert_eq!(gcrs.moon, mat3_vec3_mul(&r, &mean.moon).expect("moon"));
        }
    }

    #[test]
    fn gcrs_entry_points_agree_on_one_instant() {
        let century_s = DAYS_PER_JULIAN_CENTURY * SECONDS_PER_DAY;
        for epoch_tdb_seconds in [-3.0e8, 0.0, 10_001.0, 8.17e8, 1.57e9] {
            // Same TT argument for the series, so the two differ only in how the
            // TDB of the precession is formed (TDB - TT from TT here, the given
            // TDB there): Julian-date rounding, under 1e-15 rad of rotation.
            let t_tt = tt_from_tdb_j2000_seconds(epoch_tdb_seconds) / century_s;
            let from_tdb = sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds).expect("TDB seconds");
            let from_tt = sun_moon_gcrs(t_tt).expect("TT centuries");
            assert!(relative_difference(from_tdb.sun, from_tt.sun) <= 1.0e-14);
            assert!(relative_difference(from_tdb.moon, from_tt.moon) <= 1.0e-14);
            // The series is evaluated at the TT of the instant (not the TDB read
            // as TT) and the precession at the given TDB.
            let mean = sun_moon_eci(t_tt).expect("mean of date");
            let r = mean_of_date_to_gcrs_matrix_unchecked(
                J2000_JD + epoch_tdb_seconds / SECONDS_PER_DAY,
            );
            assert_eq!(from_tdb.sun, mat3_vec3_mul_unchecked(&r, &mean.sun));
            assert_eq!(from_tdb.moon, mat3_vec3_mul_unchecked(&r, &mean.moon));
        }

        let ts = TimeScales::from_utc(2026, 11, 23, 0, 0, 0.0).expect("valid UTC");
        let t = (ts.jd_tt - J2000_JD) / DAYS_PER_JULIAN_CENTURY;
        let at = sun_moon_gcrs_at(&ts).expect("time scales");
        let from_t = sun_moon_gcrs(t).expect("TT centuries");
        assert!(relative_difference(at.sun, from_t.sun) <= 1.0e-14);
        assert!(relative_difference(at.moon, from_t.moon) <= 1.0e-14);
    }

    #[test]
    fn gcrs_and_mean_of_date_differ_by_the_accumulated_precession() {
        // The Sun lies in the ecliptic, about 90 deg from the precession axis,
        // so its direction turns by nearly the full rotation angle: 0.3757 deg
        // at 2026-11-23 (Orekit 13.1.10 MOD to GCRF, IERS 2010 conventions).
        let ts = TimeScales::from_utc(2026, 11, 23, 0, 0, 0.0).expect("valid UTC");
        let mean = sun_moon_eci_at(&ts).expect("mean of date");
        let gcrs = sun_moon_gcrs_at(&ts).expect("GCRS");
        let sun_deg = angle_deg(mean.sun, gcrs.sun);
        assert!(
            (0.374..0.377).contains(&sun_deg),
            "Sun turned {sun_deg} deg"
        );
        let moon_deg = angle_deg(mean.moon, gcrs.moon);
        assert!(
            (0.33..0.377).contains(&moon_deg),
            "Moon turned {moon_deg} deg"
        );

        // At J2000.0 only the frame bias (23 mas) separates the frames.
        let at_j2000 = sun_moon_gcrs(0.0).expect("GCRS");
        let mean_j2000 = sun_moon_eci(0.0).expect("mean of date");
        assert!(angle_deg(mean_j2000.sun, at_j2000.sun) < 25.0 / 3.6e6);
    }

    #[test]
    fn sun_moon_gcrs_accepts_time_scales_outside_the_ut1_table() {
        let ts = TimeScales::from_utc(2100, 1, 1, 0, 0, 0.0).expect("valid UTC");
        assert!(ts.ut1_degraded.is_some());
        assert!(sun_moon_gcrs_at(&ts).is_ok());
    }

    #[test]
    fn sun_moon_gcrs_rejects_non_finite_inputs_and_outputs() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                sun_moon_gcrs(value),
                Err(SunMoonError::InvalidInput {
                    field: "t",
                    reason: "not finite"
                })
            );
            assert_eq!(
                sun_moon_gcrs_at_tdb_seconds(value),
                Err(SunMoonError::InvalidInput {
                    field: "epoch_tdb_seconds",
                    reason: "not finite"
                })
            );
        }
        let mut ts = TimeScales::from_utc(2026, 4, 30, 9, 45, 0.0).expect("valid UTC instant");
        ts.jd_tdb = f64::NAN;
        assert_eq!(
            sun_moon_gcrs_at(&ts),
            Err(SunMoonError::InvalidInput {
                field: "jd_tdb",
                reason: "not finite"
            })
        );
        ts.jd_tt = f64::NAN;
        assert_eq!(
            sun_moon_gcrs_at(&ts),
            Err(SunMoonError::InvalidInput {
                field: "jd_tt",
                reason: "not finite"
            })
        );
        for result in [
            sun_moon_gcrs(f64::MAX),
            sun_moon_gcrs_at_tdb_seconds(f64::MAX),
        ] {
            assert!(matches!(
                result,
                Err(SunMoonError::InvalidInput {
                    field: "sun[0]" | "sun[1]" | "sun[2]" | "moon[0]" | "moon[1]" | "moon[2]",
                    reason: "not finite"
                })
            ));
        }
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
