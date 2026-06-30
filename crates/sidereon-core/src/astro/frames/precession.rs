//! IAU 2006 precession matrix and ICRS-to-J2000 frame bias, ported from the
//! C++ Skyfield-compatible implementation.
//!
//! Originally `pub(crate)` inside `orbis_nif`; now public in the core crate so
//! a Rust-only consumer can reach it without Rustler or the BEAM. The numerics,
//! summation order, and transcendental sequence are preserved exactly so the
//! existing Skyfield 0-ULP parity holds.
//!
//! All arithmetic uses plain operators (no `f64::mul_add`) so that
//! rounding matches CPython / Skyfield compiled without FMA contraction.

use crate::astro::constants::time::{DAYS_PER_JULIAN_CENTURY, J2000_JD};
use crate::astro::constants::units::ARCSEC_TO_RAD;
use crate::astro::math::mat3::Mat3;

/// Error returned when public precession inputs are outside the valid domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum PrecessionError {
    /// A precession input was non-finite or otherwise invalid.
    #[error("invalid precession {field}: {reason}")]
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
}

fn invalid_input(field: &'static str, reason: &'static str) -> PrecessionError {
    PrecessionError::InvalidInput { field, reason }
}

fn validate_finite(field: &'static str, value: f64) -> Result<(), PrecessionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

fn validate_mat3(field: &'static str, mat: Mat3) -> Result<Mat3, PrecessionError> {
    if mat.iter().flatten().all(|value| value.is_finite()) {
        Ok(mat)
    } else {
        Err(invalid_input(field, "components must be finite"))
    }
}

// ---------------------------------------------------------------------------
// Precession matrix (IAU 2006)
// ---------------------------------------------------------------------------

/// Compute the 3x3 precession rotation matrix for the given TDB Julian date,
/// using the IAU 2006 Fukushima-Williams parameterisation.
pub fn compute_skyfield_precession_matrix(jd_tdb: f64) -> Result<Mat3, PrecessionError> {
    validate_finite("jd_tdb", jd_tdb)?;
    validate_mat3(
        "precession_matrix",
        compute_skyfield_precession_matrix_unchecked(jd_tdb),
    )
}

pub(crate) fn compute_skyfield_precession_matrix_unchecked(jd_tdb: f64) -> Mat3 {
    const EPS0_ARCSEC: f64 = 84381.406;

    let t = (jd_tdb - J2000_JD) / DAYS_PER_JULIAN_CENTURY;

    let psia = ((((-0.0000000951 * t + 0.000132851) * t - 0.00114045) * t - 1.0790069) * t
        + 5038.481507)
        * t;

    let omegaa =
        ((((0.0000003337 * t - 0.000000467) * t - 0.00772503) * t + 0.0512623) * t - 0.025754) * t
            + EPS0_ARCSEC;

    let chia = ((((-0.0000000560 * t + 0.000170663) * t - 0.00121197) * t - 2.3814292) * t
        + 10.556403)
        * t;

    let eps0 = EPS0_ARCSEC * ARCSEC_TO_RAD;
    let psia_rad = psia * ARCSEC_TO_RAD;
    let omegaa_rad = omegaa * ARCSEC_TO_RAD;
    let chia_rad = chia * ARCSEC_TO_RAD;

    let sa = eps0.sin();
    let ca = eps0.cos();
    let sb = (-psia_rad).sin();
    let cb = (-psia_rad).cos();
    let sc = (-omegaa_rad).sin();
    let cc = (-omegaa_rad).cos();
    let sd = chia_rad.sin();
    let cd = chia_rad.cos();

    [
        [
            cd * cb - sb * sd * cc,
            cd * sb * ca + sd * cc * cb * ca - sa * sd * sc,
            cd * sb * sa + sd * cc * cb * sa + ca * sd * sc,
        ],
        [
            -sd * cb - sb * cd * cc,
            -sd * sb * ca + cd * cc * cb * ca - sa * cd * sc,
            -sd * sb * sa + cd * cc * cb * sa + ca * cd * sc,
        ],
        [sb * sc, -sc * cb * ca - sa * cc, -sc * cb * sa + cc * ca],
    ]
}

// ---------------------------------------------------------------------------
// ICRS-to-J2000 frame bias matrix
// ---------------------------------------------------------------------------

/// Build the ICRS-to-J2000 frame bias rotation matrix.
///
/// This accounts for the small offset between the ICRS axes and the
/// mean J2000.0 dynamical frame, parameterised by the three bias angles
/// xi_0, eta_0, and da_0 from the IAU 2006 conventions.
pub fn build_icrs_to_j2000() -> Mat3 {
    let xi0 = -0.0166170 * ARCSEC_TO_RAD;
    let eta0 = -0.0068192 * ARCSEC_TO_RAD;
    let da0 = -0.01460 * ARCSEC_TO_RAD;

    let yx = -da0;
    let zx = xi0;
    let xy = da0;
    let zy = eta0;
    let xz = -xi0;
    let yz = -eta0;

    [
        [1.0 - 0.5 * (yx * yx + zx * zx), xy, xz],
        [yx, 1.0 - 0.5 * (yx * yx + zy * zy), yz],
        [zx, zy, 1.0 - 0.5 * (zy * zy + zx * zx)],
    ]
}
