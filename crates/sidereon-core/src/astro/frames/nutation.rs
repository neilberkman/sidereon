//! IAU 2000A nutation computation, ported from the C++ Skyfield-compatible
//! implementation.
//!
//! Originally `pub(crate)` inside `orbis_nif`; now public in the core crate so
//! a Rust-only consumer can reach the transform substrate without Rustler or
//! the BEAM. The numerics, summation order, and transcendental sequence are
//! preserved exactly so the existing Skyfield 0-ULP parity holds.
//!
//! All arithmetic uses plain operators (no `f64::mul_add`) so that
//! rounding matches CPython / Skyfield compiled without FMA contraction.

use crate::astro::constants::time::{DAYS_PER_JULIAN_CENTURY, J2000_JD};
use crate::astro::constants::units::ARCSEC_TO_RAD;
use crate::astro::data::iau2000a::*;
use crate::astro::math::mat3::Mat3;

use std::f64::consts::PI;

const TAU: f64 = 2.0 * PI;
const ASEC360: f64 = 1_296_000.0;
const TENTH_USEC_2_RAD: f64 = ARCSEC_TO_RAD / 1.0e7;

/// Error returned when public nutation inputs are outside the valid domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum NutationError {
    /// A nutation input was non-finite or otherwise invalid.
    #[error("invalid nutation {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

fn invalid_input(field: &'static str, reason: &'static str) -> NutationError {
    NutationError::InvalidInput { field, reason }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), NutationError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

fn validate_finite_values<const N: usize>(
    field: &'static str,
    values: [f64; N],
) -> Result<[f64; N], NutationError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(values)
    } else {
        Err(invalid_input(field, "components must be finite"))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn positive_fmod(value: f64, modulus: f64) -> f64 {
    let mut result = value % modulus;
    if result < 0.0 {
        result += modulus;
    }
    result
}

// ---------------------------------------------------------------------------
// Fundamental arguments (lunisolar)
// ---------------------------------------------------------------------------

/// Compute the five Delaunay fundamental arguments of nutation theory
/// for a given Julian-century offset `t` from J2000.0 TDB.
pub fn skyfield_fundamental_arguments(t: f64) -> Result<[f64; 5], NutationError> {
    validate_finite("t", t)?;
    validate_finite_values(
        "fundamental_arguments",
        skyfield_fundamental_arguments_unchecked(t),
    )
}

fn skyfield_fundamental_arguments_unchecked(t: f64) -> [f64; 5] {
    const FA0: [f64; 5] = [
        485868.249036,
        1287104.79305,
        335779.526232,
        1072260.70369,
        450160.398036,
    ];
    const FA1: [f64; 5] = [
        1717915923.2178,
        129596581.0481,
        1739527262.8478,
        1602961601.2090,
        -6962890.5431,
    ];
    const FA2: [f64; 5] = [31.8792, -0.5532, -12.7512, -6.3706, 7.4722];
    const FA3: [f64; 5] = [0.051635, 0.000136, -0.001037, 0.006593, 0.007702];
    const FA4: [f64; 5] = [
        -0.00024470,
        -0.00001149,
        0.00000417,
        -0.00003169,
        -0.00005939,
    ];

    let mut args = [0.0_f64; 5];
    for i in 0..5 {
        let mut value = FA4[i] * t;
        value += FA3[i];
        value *= t;
        value += FA2[i];
        value *= t;
        value += FA1[i];
        value *= t;
        value += FA0[i];
        value %= ASEC360;
        args[i] = value * ARCSEC_TO_RAD;
    }
    args
}

// ---------------------------------------------------------------------------
// IAU 2000A nutation in longitude and obliquity
// ---------------------------------------------------------------------------

/// Compute nutation in longitude (`dpsi`) and obliquity (`deps`) in radians
/// using the full IAU 2000A model.
///
/// `jd_tt` is the Julian date on the TT time scale.
pub fn skyfield_iau2000a_radians(jd_tt: f64) -> Result<(f64, f64), NutationError> {
    validate_finite("jd_tt", jd_tt)?;
    let values = skyfield_iau2000a_radians_unchecked(jd_tt);
    validate_finite("dpsi", values.0)?;
    validate_finite("deps", values.1)?;
    Ok(values)
}

pub(crate) fn skyfield_iau2000a_radians_unchecked(jd_tt: f64) -> (f64, f64) {
    const ANOMALY_CONSTANT: [f64; 14] = [
        2.35555598,
        6.24006013,
        1.627905234,
        5.198466741,
        2.18243920,
        4.402608842,
        3.176146697,
        1.753470314,
        6.203480913,
        0.599546497,
        0.874016757,
        5.481293871,
        5.321159000,
        0.02438175,
    ];
    const ANOMALY_COEFFICIENT: [f64; 14] = [
        8328.6914269554,
        628.301955,
        8433.466158131,
        7771.3771468121,
        -33.757045,
        2608.7903141574,
        1021.3285546211,
        628.3075849991,
        334.0612426700,
        52.9690962641,
        21.3299104960,
        7.4781598567,
        3.8127774000,
        0.00000538691,
    ];

    let t = (jd_tt - J2000_JD) / DAYS_PER_JULIAN_CENTURY;

    let fundamental_args = skyfield_fundamental_arguments_unchecked(t);

    let mut dpsi = 0.0_f64;
    let mut deps = 0.0_f64;

    // Lunisolar nutation (678 terms)
    for row in 0..678 {
        let mut arg = 0.0_f64;
        for i in 0..5 {
            arg += (NALS_T[row][i] as f64) * fundamental_args[i];
        }

        let sarg = arg.sin();
        let carg = arg.cos();

        dpsi += sarg * LUNISOLAR_LONGITUDE_COEFFICIENTS[row][0];
        dpsi += sarg * LUNISOLAR_LONGITUDE_COEFFICIENTS[row][1] * t;
        dpsi += carg * LUNISOLAR_LONGITUDE_COEFFICIENTS[row][2];

        deps += carg * LUNISOLAR_OBLIQUITY_COEFFICIENTS[row][0];
        deps += carg * LUNISOLAR_OBLIQUITY_COEFFICIENTS[row][1] * t;
        deps += sarg * LUNISOLAR_OBLIQUITY_COEFFICIENTS[row][2];
    }

    // Planetary arguments
    let mut planetary_args = [0.0_f64; 14];
    for i in 0..14 {
        planetary_args[i] = ANOMALY_CONSTANT[i] + ANOMALY_COEFFICIENT[i] * t;
    }
    planetary_args[13] *= t;

    // Planetary nutation (687 terms)
    for row in 0..687 {
        let mut arg = 0.0_f64;
        for i in 0..14 {
            arg += (NAPL_T[row][i] as f64) * planetary_args[i];
        }

        let sarg = arg.sin();
        let carg = arg.cos();

        dpsi += sarg * NUTATION_COEFFICIENTS_LONGITUDE[row][0];
        dpsi += carg * NUTATION_COEFFICIENTS_LONGITUDE[row][1];

        deps += sarg * NUTATION_COEFFICIENTS_OBLIQUITY[row][0];
        deps += carg * NUTATION_COEFFICIENTS_OBLIQUITY[row][1];
    }

    (dpsi * TENTH_USEC_2_RAD, deps * TENTH_USEC_2_RAD)
}

// ---------------------------------------------------------------------------
// Mean obliquity
// ---------------------------------------------------------------------------

/// Mean obliquity of the ecliptic in radians for the given TDB Julian date.
pub fn skyfield_mean_obliquity_radians(jd_tdb: f64) -> Result<f64, NutationError> {
    validate_finite("jd_tdb", jd_tdb)?;
    let value = skyfield_mean_obliquity_radians_unchecked(jd_tdb);
    validate_finite("mean_obliquity_radians", value)?;
    Ok(value)
}

pub(crate) fn skyfield_mean_obliquity_radians_unchecked(jd_tdb: f64) -> f64 {
    let t = (jd_tdb - J2000_JD) / DAYS_PER_JULIAN_CENTURY;
    let epsilon = ((((-0.0000000434 * t - 0.000000576) * t + 0.00200340) * t - 0.0001831) * t
        - 46.836769)
        * t
        + 84381.406;
    epsilon * ARCSEC_TO_RAD
}

// ---------------------------------------------------------------------------
// Nutation matrix
// ---------------------------------------------------------------------------

/// Build the 3x3 nutation rotation matrix from mean obliquity, true obliquity,
/// and the nutation in longitude (psi).
pub fn build_skyfield_nutation_matrix(
    mean_obliquity_radians: f64,
    true_obliquity_radians: f64,
    psi_radians: f64,
) -> Result<Mat3, NutationError> {
    validate_finite("mean_obliquity_radians", mean_obliquity_radians)?;
    validate_finite("true_obliquity_radians", true_obliquity_radians)?;
    validate_finite("psi_radians", psi_radians)?;
    validate_mat3(
        "nutation_matrix",
        build_skyfield_nutation_matrix_unchecked(
            mean_obliquity_radians,
            true_obliquity_radians,
            psi_radians,
        ),
    )
}

pub(crate) fn build_skyfield_nutation_matrix_unchecked(
    mean_obliquity_radians: f64,
    true_obliquity_radians: f64,
    psi_radians: f64,
) -> Mat3 {
    let cobm = mean_obliquity_radians.cos();
    let sobm = mean_obliquity_radians.sin();
    let cobt = true_obliquity_radians.cos();
    let sobt = true_obliquity_radians.sin();
    let cpsi = psi_radians.cos();
    let spsi = psi_radians.sin();

    [
        [cpsi, -spsi * cobm, -spsi * sobm],
        [
            spsi * cobt,
            cpsi * cobm * cobt + sobm * sobt,
            cpsi * sobm * cobt - cobm * sobt,
        ],
        [
            spsi * sobt,
            cpsi * cobm * sobt - sobm * cobt,
            cpsi * sobm * sobt + cobm * cobt,
        ],
    ]
}

fn validate_mat3(field: &'static str, mat: Mat3) -> Result<Mat3, NutationError> {
    if mat.iter().flatten().all(|value| value.is_finite()) {
        Ok(mat)
    } else {
        Err(invalid_input(field, "components must be finite"))
    }
}

// ---------------------------------------------------------------------------
// Equation of the equinoxes -- complementary terms
// ---------------------------------------------------------------------------

/// Complementary terms of the equation of the equinoxes, in radians.
///
/// `jd_tt` is the Julian date on the TT time scale.
pub fn skyfield_equation_of_the_equinoxes_complimentary_terms(
    jd_tt: f64,
) -> Result<f64, NutationError> {
    validate_finite("jd_tt", jd_tt)?;
    let value = skyfield_equation_of_the_equinoxes_complimentary_terms_unchecked(jd_tt);
    validate_finite("equation_of_the_equinoxes_terms", value)?;
    Ok(value)
}

pub(crate) fn skyfield_equation_of_the_equinoxes_complimentary_terms_unchecked(jd_tt: f64) -> f64 {
    const KE1: [i32; 14] = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    #[rustfmt::skip]
    const KE0_T: [[i32; 14]; 33] = [
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, -2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, -2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, -4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, 1, 0, -8, 12, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -2, 2, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, -2, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 8, -13, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, -2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 2, -2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, -2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, -2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, -2, 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, -2, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ];

    #[rustfmt::skip]
    const SE0_T_0: [f64; 33] = [
        0.00264096, 0.00006352, 0.00001175, 0.00001121, -0.00000455, 0.00000202,
        0.00000198, -0.00000172, -0.00000141, -0.00000126, -0.00000063, -0.00000063,
        0.00000046, 0.00000045, 0.00000036, -0.00000024, 0.00000032, 0.00000028,
        0.00000027, 0.00000026, -0.00000021, 0.00000019, 0.00000018, -0.00000010,
        0.00000015, -0.00000014, 0.00000014, -0.00000014, 0.00000014, 0.00000013,
        -0.00000011, 0.00000011, 0.00000011,
    ];

    #[rustfmt::skip]
    const SE0_T_1: [f64; 33] = [
        -0.00000039, -0.00000002, 0.00000001, 0.00000001, 0.0, 0.0,
        0.0, 0.0, -0.00000001, -0.00000001, 0.0, 0.0,
        0.0, 0.0, 0.0, -0.00000012, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.00000005,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ];

    const SE1_0: f64 = -0.00000087;

    let t = (jd_tt - J2000_JD) / DAYS_PER_JULIAN_CENTURY;

    // Compute the 14 fundamental arguments (the same ones used for the
    // equation-of-equinoxes complementary terms in the C++ code).
    let mut fa = [0.0_f64; 14];

    fa[0] = (485868.249036
        + (715923.2178 + (31.8792 + (0.051635 + (-0.00024470) * t) * t) * t) * t)
        * ARCSEC_TO_RAD
        + positive_fmod(1325.0 * t, 1.0) * TAU;

    fa[1] = (1287104.793048
        + (1292581.0481 + (-0.5532 + (0.000136 + (-0.00001149) * t) * t) * t) * t)
        * ARCSEC_TO_RAD
        + positive_fmod(99.0 * t, 1.0) * TAU;

    fa[2] = (335779.526232
        + (295262.8478 + (-12.7512 + (-0.001037 + (0.00000417) * t) * t) * t) * t)
        * ARCSEC_TO_RAD
        + positive_fmod(1342.0 * t, 1.0) * TAU;

    fa[3] = (1072260.703692
        + (1105601.2090 + (-6.3706 + (0.006593 + (-0.00003169) * t) * t) * t) * t)
        * ARCSEC_TO_RAD
        + positive_fmod(1236.0 * t, 1.0) * TAU;

    fa[4] = (450160.398036
        + (-482890.5431 + (7.4722 + (0.007702 + (-0.00005939) * t) * t) * t) * t)
        * ARCSEC_TO_RAD
        + positive_fmod(-5.0 * t, 1.0) * TAU;

    fa[5] = 4.402608842 + 2608.7903141574 * t;
    fa[6] = 3.176146697 + 1021.3285546211 * t;
    fa[7] = 1.753470314 + 628.3075849991 * t;
    fa[8] = 6.203480913 + 334.0612426700 * t;
    fa[9] = 0.599546497 + 52.9690962641 * t;
    fa[10] = 0.874016757 + 21.3299104960 * t;
    fa[11] = 5.481293872 + 7.4781598567 * t;
    fa[12] = 5.311886287 + 3.8133035638 * t;
    fa[13] = (0.024381750 + 0.00000538691 * t) * t;

    for angle in &mut fa {
        *angle = positive_fmod(*angle, TAU);
    }

    // Single t-dependent term
    let mut a = 0.0_f64;
    for i in 0..14 {
        a += (KE1[i] as f64) * fa[i];
    }
    let mut c_terms = SE1_0 * a.sin();
    c_terms *= t;

    // Constant terms (33 rows)
    for row in 0..33 {
        let mut arg = 0.0_f64;
        for i in 0..14 {
            arg += (KE0_T[row][i] as f64) * fa[i];
        }
        c_terms += SE0_T_0[row] * arg.sin();
        c_terms += SE0_T_1[row] * arg.cos();
    }

    c_terms * ARCSEC_TO_RAD
}

#[cfg(test)]
mod tests {
    use super::{
        skyfield_fundamental_arguments, skyfield_iau2000a_radians, skyfield_mean_obliquity_radians,
        NutationError,
    };

    #[test]
    fn public_nutation_helpers_reject_non_finite_times() {
        assert_eq!(
            skyfield_fundamental_arguments(f64::NAN),
            Err(NutationError::InvalidInput {
                field: "t",
                reason: "must be finite"
            })
        );
        assert_eq!(
            skyfield_iau2000a_radians(f64::INFINITY),
            Err(NutationError::InvalidInput {
                field: "jd_tt",
                reason: "must be finite"
            })
        );
        assert_eq!(
            skyfield_mean_obliquity_radians(f64::NEG_INFINITY),
            Err(NutationError::InvalidInput {
                field: "jd_tdb",
                reason: "must be finite"
            })
        );
    }
}
