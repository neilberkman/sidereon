//! Equinoctial and modified equinoctial orbital elements.
//!
//! This module is pure element algebra layered on the existing classical
//! conversion kernels. Cartesian conversions compose through [`rv2coe`] and
//! [`coe2rv`], and the mean-longitude equinoctial path uses the shared anomaly
//! conversions.

use crate::astro::anomaly::{mean_to_true, true_to_mean, AnomalyError};
use crate::astro::elements::{
    clamp_acos, classify, coe2rv, rv2coe, ClassicalElements, ElementsError, OrbitType,
};
use crate::astro::math::{normalize_angle, wrap_to_pi, SMALL, TWO_PI};

const PI: f64 = std::f64::consts::PI;
const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;
const STABLE_HALF_ANGLE_MIN: f64 = 1.0e-4;
const STABLE_HALF_ANGLE_MAX: f64 = 1.0e4;

/// Retrograde-factor selector for equinoctial-family element sets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetrogradeFactor {
    /// I = +1. Default. Singular only at i = pi.
    Prograde,
    /// I = -1. Singular only at i = 0. Use for orbits at or near i = pi.
    Retrograde,
}

impl RetrogradeFactor {
    #[inline]
    fn sign(self) -> f64 {
        match self {
            Self::Prograde => 1.0,
            Self::Retrograde => -1.0,
        }
    }
}

/// Equinoctial elements `(a, h, k, p, q, lambda)`, mean-longitude form.
///
/// The `p` and `q` fields are inclination components, not the semi-latus
/// rectum.
///
/// ```
/// use sidereon_core::astro::equinoctial::{EquinoctialElements, RetrogradeFactor};
///
/// let eq = EquinoctialElements {
///     a: 7000.0,
///     h: 0.0,
///     k: 0.001,
///     p: 0.0,
///     q: 0.0,
///     lambda: 0.2,
///     retrograde: RetrogradeFactor::Prograde,
/// };
/// assert_eq!(eq.p, 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EquinoctialElements {
    /// Semi-major axis (km), finite for all supported equinoctial conics.
    pub a: f64,
    /// Eccentricity component `e sin(varpi)`.
    pub h: f64,
    /// Eccentricity component `e cos(varpi)`.
    pub k: f64,
    /// Inclination component `tan(i/2)^I sin(raan)`.
    pub p: f64,
    /// Inclination component `tan(i/2)^I cos(raan)`.
    pub q: f64,
    /// Mean longitude (rad).
    pub lambda: f64,
    /// Retrograde factor used to build the components.
    pub retrograde: RetrogradeFactor,
}

/// Modified equinoctial elements `(p, f, g, h, k, L)`, true-longitude form.
///
/// The `p` field is the semi-latus rectum. The `h` and `k` fields are
/// inclination components.
///
/// ```
/// use sidereon_core::astro::equinoctial::{
///     ModifiedEquinoctialElements, RetrogradeFactor,
/// };
///
/// let mee = ModifiedEquinoctialElements {
///     p: 6999.0,
///     f: 0.001,
///     g: 0.0,
///     h: 0.0,
///     k: 0.0,
///     l: 0.2,
///     retrograde: RetrogradeFactor::Prograde,
/// };
/// assert!(mee.p > 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModifiedEquinoctialElements {
    /// Semi-latus rectum (km), the same size element carried by
    /// [`ClassicalElements::p`].
    pub p: f64,
    /// Eccentricity component `e cos(varpi)`.
    pub f: f64,
    /// Eccentricity component `e sin(varpi)`.
    pub g: f64,
    /// Inclination component `tan(i/2)^I cos(raan)`.
    pub h: f64,
    /// Inclination component `tan(i/2)^I sin(raan)`.
    pub k: f64,
    /// True longitude `L` (rad).
    pub l: f64,
    /// Retrograde factor used to build the components.
    pub retrograde: RetrogradeFactor,
}

/// Error returned by equinoctial-family conversions.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum EquinoctialError {
    /// A classical-element or state conversion failed.
    #[error(transparent)]
    Elements(#[from] ElementsError),
    /// The mean-longitude anomaly step failed.
    #[error(transparent)]
    Anomaly(#[from] AnomalyError),
    /// The selected retrograde factor is singular at this inclination.
    #[error("inclination at the singular pole for the selected retrograde factor")]
    RetrogradePole,
    /// The mean-longitude equinoctial set cannot represent a parabolic orbit.
    #[error("parabolic orbit is not representable in the equinoctial set; use MEE")]
    ParabolicEquinoctial,
}

/// Convert classical elements to mean-longitude equinoctial elements.
pub fn coe2eq(
    coe: &ClassicalElements,
    factor: RetrogradeFactor,
) -> Result<EquinoctialElements, EquinoctialError> {
    validate_classical_common(coe)?;
    if is_parabolic(coe.ecc) {
        return Err(EquinoctialError::ParabolicEquinoctial);
    }
    check_finite(coe.a, "a")?;
    check_retrograde_pole(coe.incl, factor)?;

    let resolved = resolve_classical_angles(coe)?;
    let varpi = wrap_to_pi(resolved.argp + factor.sign() * resolved.raan);
    let mean = true_to_mean(resolved.nu, coe.ecc)?;
    let lambda = mean_longitude(mean, varpi, coe.ecc);
    let scale = inclination_scale(coe.incl, factor)?;

    Ok(EquinoctialElements {
        a: coe.a,
        h: coe.ecc * libm::sin(varpi),
        k: coe.ecc * libm::cos(varpi),
        p: scale * libm::sin(resolved.raan),
        q: scale * libm::cos(resolved.raan),
        lambda,
        retrograde: factor,
    })
}

/// Convert mean-longitude equinoctial elements to classical elements.
pub fn eq2coe(eq: &EquinoctialElements) -> Result<ClassicalElements, EquinoctialError> {
    let recovered = recover_equinoctial(eq)?;
    Ok(recovered_to_classical(&recovered))
}

/// Convert classical elements to modified equinoctial elements.
pub fn coe2mee(
    coe: &ClassicalElements,
    factor: RetrogradeFactor,
) -> Result<ModifiedEquinoctialElements, EquinoctialError> {
    validate_classical_common(coe)?;
    validate_classical_a_for_mee(coe)?;
    check_retrograde_pole(coe.incl, factor)?;

    let resolved = resolve_classical_angles(coe)?;
    let varpi = wrap_to_pi(resolved.argp + factor.sign() * resolved.raan);
    let l = normalize_angle(varpi + resolved.nu);
    let scale = inclination_scale(coe.incl, factor)?;

    Ok(ModifiedEquinoctialElements {
        p: coe.p,
        f: coe.ecc * libm::cos(varpi),
        g: coe.ecc * libm::sin(varpi),
        h: scale * libm::cos(resolved.raan),
        k: scale * libm::sin(resolved.raan),
        l,
        retrograde: factor,
    })
}

/// Convert modified equinoctial elements to classical elements.
pub fn mee2coe(mee: &ModifiedEquinoctialElements) -> Result<ClassicalElements, EquinoctialError> {
    let recovered = recover_modified_equinoctial(mee)?;
    Ok(recovered_to_classical(&recovered))
}

/// Convert an inertial Cartesian state to mean-longitude equinoctial elements.
pub fn rv2eq(
    r: [f64; 3],
    v: [f64; 3],
    mu: f64,
    factor: RetrogradeFactor,
) -> Result<EquinoctialElements, EquinoctialError> {
    let coe = rv2coe(r, v, mu)?;
    coe2eq(&coe, factor)
}

/// Convert mean-longitude equinoctial elements to an inertial Cartesian state.
pub fn eq2rv(eq: &EquinoctialElements, mu: f64) -> Result<([f64; 3], [f64; 3]), EquinoctialError> {
    let coe = eq2coe(eq)?;
    Ok(coe2rv(&coe, mu)?)
}

/// Convert an inertial Cartesian state to modified equinoctial elements.
pub fn rv2mee(
    r: [f64; 3],
    v: [f64; 3],
    mu: f64,
    factor: RetrogradeFactor,
) -> Result<ModifiedEquinoctialElements, EquinoctialError> {
    let coe = rv2coe(r, v, mu)?;
    coe2mee(&coe, factor)
}

/// Convert modified equinoctial elements to an inertial Cartesian state.
pub fn mee2rv(
    mee: &ModifiedEquinoctialElements,
    mu: f64,
) -> Result<([f64; 3], [f64; 3]), EquinoctialError> {
    let coe = mee2coe(mee)?;
    Ok(coe2rv(&coe, mu)?)
}

/// Convert mean-longitude equinoctial elements to modified equinoctial elements.
pub fn eq2mee(eq: &EquinoctialElements) -> Result<ModifiedEquinoctialElements, EquinoctialError> {
    let recovered = recover_equinoctial(eq)?;
    let scale = inclination_scale(recovered.incl, recovered.retrograde)?;

    Ok(ModifiedEquinoctialElements {
        p: recovered.p,
        f: recovered.ecc * libm::cos(recovered.varpi),
        g: recovered.ecc * libm::sin(recovered.varpi),
        h: scale * libm::cos(recovered.raan),
        k: scale * libm::sin(recovered.raan),
        l: recovered.l,
        retrograde: recovered.retrograde,
    })
}

/// Convert modified equinoctial elements to mean-longitude equinoctial elements.
pub fn mee2eq(mee: &ModifiedEquinoctialElements) -> Result<EquinoctialElements, EquinoctialError> {
    let recovered = recover_modified_equinoctial(mee)?;
    if is_parabolic(recovered.ecc) {
        return Err(EquinoctialError::ParabolicEquinoctial);
    }
    let mean = true_to_mean(recovered.nu, recovered.ecc)?;
    let lambda = mean_longitude(mean, recovered.varpi, recovered.ecc);
    let scale = inclination_scale(recovered.incl, recovered.retrograde)?;

    Ok(EquinoctialElements {
        a: recovered.a,
        h: recovered.ecc * libm::sin(recovered.varpi),
        k: recovered.ecc * libm::cos(recovered.varpi),
        p: scale * libm::sin(recovered.raan),
        q: scale * libm::cos(recovered.raan),
        lambda,
        retrograde: recovered.retrograde,
    })
}

#[derive(Debug, Clone, Copy)]
struct ResolvedClassicalAngles {
    raan: f64,
    argp: f64,
    nu: f64,
}

#[derive(Debug, Clone, Copy)]
struct RecoveredConic {
    p: f64,
    a: f64,
    ecc: f64,
    incl: f64,
    raan: f64,
    varpi: f64,
    nu: f64,
    l: f64,
    retrograde: RetrogradeFactor,
}

fn recover_equinoctial(eq: &EquinoctialElements) -> Result<RecoveredConic, EquinoctialError> {
    validate_equinoctial(eq)?;

    let ecc = libm::hypot(eq.h, eq.k);
    if is_parabolic(ecc) {
        return Err(EquinoctialError::ParabolicEquinoctial);
    }
    let varpi = libm::atan2(eq.h, eq.k);
    let incl = recover_inclination(eq.p, eq.q, eq.retrograde)?;
    let raan = libm::atan2(eq.p, eq.q);
    let mean = eq.lambda - varpi;
    let nu = mean_to_true(mean, ecc)?;
    let l = normalize_angle(varpi + nu);
    let p = eq.a * (1.0 - ecc * ecc);
    if !p.is_finite() {
        return Err(ElementsError::NonFinite { field: "p" }.into());
    }
    if p <= 0.0 {
        return Err(ElementsError::NonPositiveSemiLatus.into());
    }

    Ok(RecoveredConic {
        p,
        a: eq.a,
        ecc,
        incl,
        raan,
        varpi,
        nu,
        l,
        retrograde: eq.retrograde,
    })
}

fn recover_modified_equinoctial(
    mee: &ModifiedEquinoctialElements,
) -> Result<RecoveredConic, EquinoctialError> {
    validate_modified_equinoctial(mee)?;

    let ecc = libm::hypot(mee.f, mee.g);
    let varpi = libm::atan2(mee.g, mee.f);
    let incl = recover_inclination(mee.h, mee.k, mee.retrograde)?;
    let raan = libm::atan2(mee.k, mee.h);
    let nu = mee.l - varpi;
    let a = if is_parabolic(ecc) {
        f64::INFINITY
    } else {
        mee.p / (1.0 - ecc * ecc)
    };
    if !is_parabolic(ecc) && !a.is_finite() {
        return Err(ElementsError::NonFinite { field: "a" }.into());
    }

    Ok(RecoveredConic {
        p: mee.p,
        a,
        ecc,
        incl,
        raan,
        varpi,
        nu,
        l: normalize_angle(mee.l),
        retrograde: mee.retrograde,
    })
}

fn recovered_to_classical(recovered: &RecoveredConic) -> ClassicalElements {
    let orbit_type = classify(recovered.ecc, recovered.incl);
    let circular = matches!(
        orbit_type,
        OrbitType::CircularInclined | OrbitType::CircularEquatorial
    );
    let equatorial = matches!(
        orbit_type,
        OrbitType::EllipticalEquatorial | OrbitType::CircularEquatorial
    );
    let ecc = if circular { 0.0 } else { recovered.ecc };
    let incl = if equatorial {
        if recovered.incl > HALF_PI {
            PI
        } else {
            0.0
        }
    } else {
        recovered.incl
    };
    let (p, a) = if circular && recovered.a.is_finite() {
        (recovered.a, recovered.a)
    } else {
        (recovered.p, recovered.a)
    };

    let raan = normalize_angle(recovered.raan);
    let argp = normalize_angle(recovered.varpi - recovered.retrograde.sign() * raan);
    let nu = normalize_angle(recovered.nu);
    let l = normalize_angle(recovered.l);

    match orbit_type {
        OrbitType::EllipticalInclined => ClassicalElements {
            p,
            a,
            ecc,
            incl,
            raan,
            argp,
            nu,
            arglat: f64::NAN,
            truelon: f64::NAN,
            lonper: f64::NAN,
            orbit_type,
        },
        OrbitType::CircularInclined => ClassicalElements {
            p,
            a,
            ecc,
            incl,
            raan,
            argp: f64::NAN,
            nu: f64::NAN,
            arglat: normalize_angle(l - recovered.retrograde.sign() * raan),
            truelon: f64::NAN,
            lonper: f64::NAN,
            orbit_type,
        },
        OrbitType::EllipticalEquatorial => ClassicalElements {
            p,
            a,
            ecc,
            incl,
            raan: f64::NAN,
            argp: f64::NAN,
            nu,
            arglat: f64::NAN,
            truelon: f64::NAN,
            lonper: folded_equatorial_angle(recovered.varpi, incl),
            orbit_type,
        },
        OrbitType::CircularEquatorial => ClassicalElements {
            p,
            a,
            ecc,
            incl,
            raan: f64::NAN,
            argp: f64::NAN,
            nu: f64::NAN,
            arglat: f64::NAN,
            truelon: folded_equatorial_angle(l, incl),
            lonper: f64::NAN,
            orbit_type,
        },
    }
}

fn resolve_classical_angles(
    coe: &ClassicalElements,
) -> Result<ResolvedClassicalAngles, EquinoctialError> {
    match coe.orbit_type {
        OrbitType::EllipticalInclined => {
            check_finite(coe.raan, "raan")?;
            check_finite(coe.argp, "argp")?;
            check_finite(coe.nu, "nu")?;
            Ok(ResolvedClassicalAngles {
                raan: coe.raan,
                argp: coe.argp,
                nu: coe.nu,
            })
        }
        OrbitType::CircularInclined => {
            check_finite(coe.raan, "raan")?;
            check_finite(coe.arglat, "arglat")?;
            Ok(ResolvedClassicalAngles {
                raan: coe.raan,
                argp: 0.0,
                nu: coe.arglat,
            })
        }
        OrbitType::EllipticalEquatorial => {
            check_finite(coe.lonper, "lonper")?;
            check_finite(coe.nu, "nu")?;
            Ok(ResolvedClassicalAngles {
                raan: 0.0,
                argp: unfolded_equatorial_angle(coe.lonper, coe.incl),
                nu: coe.nu,
            })
        }
        OrbitType::CircularEquatorial => {
            check_finite(coe.truelon, "truelon")?;
            Ok(ResolvedClassicalAngles {
                raan: 0.0,
                argp: 0.0,
                nu: unfolded_equatorial_angle(coe.truelon, coe.incl),
            })
        }
    }
}

fn validate_classical_common(coe: &ClassicalElements) -> Result<(), EquinoctialError> {
    check_finite(coe.p, "p")?;
    if coe.p <= 0.0 {
        return Err(ElementsError::NonPositiveSemiLatus.into());
    }
    check_finite(coe.ecc, "ecc")?;
    check_finite(coe.incl, "incl")?;
    Ok(())
}

fn validate_classical_a_for_mee(coe: &ClassicalElements) -> Result<(), EquinoctialError> {
    if is_parabolic(coe.ecc) {
        Ok(())
    } else {
        check_finite(coe.a, "a")
    }
}

fn validate_equinoctial(eq: &EquinoctialElements) -> Result<(), EquinoctialError> {
    check_finite(eq.a, "a")?;
    check_finite(eq.h, "h")?;
    check_finite(eq.k, "k")?;
    check_finite(eq.p, "p")?;
    check_finite(eq.q, "q")?;
    check_finite(eq.lambda, "lambda")
}

fn validate_modified_equinoctial(
    mee: &ModifiedEquinoctialElements,
) -> Result<(), EquinoctialError> {
    check_finite(mee.p, "p")?;
    if mee.p <= 0.0 {
        return Err(ElementsError::NonPositiveSemiLatus.into());
    }
    check_finite(mee.f, "f")?;
    check_finite(mee.g, "g")?;
    check_finite(mee.h, "h")?;
    check_finite(mee.k, "k")?;
    check_finite(mee.l, "l")
}

fn check_finite(value: f64, field: &'static str) -> Result<(), EquinoctialError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(ElementsError::NonFinite { field }.into())
    }
}

fn inclination_scale(incl: f64, factor: RetrogradeFactor) -> Result<f64, EquinoctialError> {
    check_retrograde_pole(incl, factor)?;
    let tan_half = libm::tan(0.5 * incl);
    let scale = match factor {
        RetrogradeFactor::Prograde => tan_half,
        RetrogradeFactor::Retrograde => 1.0 / tan_half,
    };
    if scale.is_finite() {
        Ok(scale)
    } else {
        Err(EquinoctialError::RetrogradePole)
    }
}

fn recover_inclination(x: f64, y: f64, factor: RetrogradeFactor) -> Result<f64, EquinoctialError> {
    let scale = libm::hypot(x, y);
    if !scale.is_finite() {
        return Err(EquinoctialError::RetrogradePole);
    }

    let atan_incl = match factor {
        RetrogradeFactor::Prograde => 2.0 * libm::atan(scale),
        RetrogradeFactor::Retrograde => PI - 2.0 * libm::atan(scale),
    };
    let incl = if (STABLE_HALF_ANGLE_MIN..=STABLE_HALF_ANGLE_MAX).contains(&scale) {
        let scale_sq = scale * scale;
        let cos_incl = match factor {
            RetrogradeFactor::Prograde => (1.0 - scale_sq) / (1.0 + scale_sq),
            RetrogradeFactor::Retrograde => (scale_sq - 1.0) / (1.0 + scale_sq),
        };
        clamp_acos(cos_incl)
    } else {
        atan_incl
    };
    check_retrograde_pole(incl, factor)?;
    Ok(incl)
}

fn check_retrograde_pole(incl: f64, factor: RetrogradeFactor) -> Result<(), EquinoctialError> {
    match factor {
        RetrogradeFactor::Prograde if (incl - PI).abs() < SMALL => {
            Err(EquinoctialError::RetrogradePole)
        }
        RetrogradeFactor::Retrograde if incl.abs() < SMALL => Err(EquinoctialError::RetrogradePole),
        _ => Ok(()),
    }
}

fn folded_equatorial_angle(angle: f64, incl: f64) -> f64 {
    let angle = normalize_angle(angle);
    if incl > HALF_PI {
        normalize_angle(TWO_PI - angle)
    } else {
        angle
    }
}

fn unfolded_equatorial_angle(angle: f64, incl: f64) -> f64 {
    folded_equatorial_angle(angle, incl)
}

fn mean_longitude(mean: f64, varpi: f64, ecc: f64) -> f64 {
    if ecc < 1.0 {
        normalize_angle(mean + varpi)
    } else {
        mean + varpi
    }
}

fn is_parabolic(ecc: f64) -> bool {
    (ecc - 1.0).abs() < SMALL
}

#[cfg(test)]
mod tests {
    use super::*;

    const MU: f64 = 398600.4418;
    const DEG: f64 = std::f64::consts::PI / 180.0;

    #[derive(Debug, Clone, Copy)]
    struct Case {
        name: &'static str,
        coe: ClassicalElements,
        factor: RetrogradeFactor,
    }

    fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
        assert!(
            (got - want).abs() <= tol,
            "{label}: got {got}, want {want}, diff {}",
            (got - want).abs()
        );
    }

    fn assert_scaled_close(got: f64, want: f64, rel: f64, label: &str) {
        let scale = 1.0_f64.max(want.abs());
        assert!(
            (got - want).abs() <= rel * scale,
            "{label}: got {got}, want {want}, diff {}",
            (got - want).abs()
        );
    }

    fn angle_diff(a: f64, b: f64) -> f64 {
        let diff = normalize_angle(a - b);
        diff.min(TWO_PI - diff)
    }

    fn assert_angle_close(got: f64, want: f64, tol: f64, label: &str) {
        let diff = angle_diff(got, want);
        assert!(diff <= tol, "{label}: got {got}, want {want}, diff {diff}");
    }

    fn assert_vec_close(got: [f64; 3], want: [f64; 3], tol: f64, label: &str) {
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() <= tol,
                "{label}[{i}]: got {}, want {}, diff {}",
                got[i],
                want[i],
                (got[i] - want[i]).abs()
            );
        }
    }

    fn assert_state_close(got: ([f64; 3], [f64; 3]), want: ([f64; 3], [f64; 3]), label: &str) {
        assert_vec_close(got.0, want.0, 1.0e-7, &format!("{label} r"));
        assert_vec_close(got.1, want.1, 1.0e-10, &format!("{label} v"));
    }

    fn classical(p: f64, ecc: f64, incl: f64, raan: f64, argp: f64, nu: f64) -> ClassicalElements {
        ClassicalElements::new(p, ecc, incl, raan, argp, nu)
    }

    fn circular_inclined(p: f64, incl: f64, raan: f64, arglat: f64) -> ClassicalElements {
        let mut coe = ClassicalElements::new(p, 0.0, incl, raan, 0.0, 0.0);
        coe.orbit_type = OrbitType::CircularInclined;
        coe.argp = f64::NAN;
        coe.nu = f64::NAN;
        coe.arglat = arglat;
        coe
    }

    fn elliptical_equatorial(
        p: f64,
        ecc: f64,
        incl: f64,
        lonper: f64,
        nu: f64,
    ) -> ClassicalElements {
        let mut coe = ClassicalElements::new(p, ecc, incl, 0.0, 0.0, nu);
        coe.orbit_type = OrbitType::EllipticalEquatorial;
        coe.raan = f64::NAN;
        coe.argp = f64::NAN;
        coe.lonper = lonper;
        coe
    }

    fn regimes() -> Vec<Case> {
        vec![
            Case {
                name: "leo circular inclined",
                coe: circular_inclined(7000.0, 51.6 * DEG, 80.0 * DEG, 135.0 * DEG),
                factor: RetrogradeFactor::Prograde,
            },
            Case {
                name: "geo near equatorial",
                coe: classical(
                    42164.0 * (1.0 - 0.001_f64.powi(2)),
                    0.001,
                    0.01 * DEG,
                    40.0 * DEG,
                    20.0 * DEG,
                    10.0 * DEG,
                ),
                factor: RetrogradeFactor::Prograde,
            },
            Case {
                name: "molniya",
                coe: classical(
                    26600.0 * (1.0 - 0.74_f64.powi(2)),
                    0.74,
                    63.4 * DEG,
                    30.0 * DEG,
                    270.0 * DEG,
                    15.0 * DEG,
                ),
                factor: RetrogradeFactor::Prograde,
            },
            Case {
                name: "near circular",
                coe: classical(
                    7200.0 * (1.0 - 1.0e-6_f64.powi(2)),
                    1.0e-6,
                    40.0 * DEG,
                    12.0 * DEG,
                    25.0 * DEG,
                    80.0 * DEG,
                ),
                factor: RetrogradeFactor::Prograde,
            },
            Case {
                name: "near equatorial",
                coe: classical(
                    9000.0 * (1.0 - 0.2_f64.powi(2)),
                    0.2,
                    1.0e-4 * DEG,
                    90.0 * DEG,
                    80.0 * DEG,
                    70.0 * DEG,
                ),
                factor: RetrogradeFactor::Prograde,
            },
            Case {
                name: "combined near circular near equatorial",
                coe: classical(
                    7100.0 * (1.0 - 1.0e-6_f64.powi(2)),
                    1.0e-6,
                    1.0e-4 * DEG,
                    5.0 * DEG,
                    5.0 * DEG,
                    45.0 * DEG,
                ),
                factor: RetrogradeFactor::Prograde,
            },
            Case {
                name: "retrograde prograde factor",
                coe: classical(
                    12000.0 * (1.0 - 0.1_f64.powi(2)),
                    0.1,
                    175.0 * DEG,
                    45.0 * DEG,
                    10.0 * DEG,
                    250.0 * DEG,
                ),
                factor: RetrogradeFactor::Prograde,
            },
            Case {
                name: "true retrograde",
                coe: elliptical_equatorial(
                    11000.0 * (1.0 - 0.2_f64.powi(2)),
                    0.2,
                    PI,
                    40.0 * DEG,
                    80.0 * DEG,
                ),
                factor: RetrogradeFactor::Retrograde,
            },
        ]
    }

    fn assert_classical_recombined_close(
        got: &ClassicalElements,
        want: &ClassicalElements,
        factor: RetrogradeFactor,
        label: &str,
    ) {
        assert_eq!(got.orbit_type, want.orbit_type, "{label} orbit_type");
        assert_scaled_close(got.p, want.p, 1.0e-10, &format!("{label} p"));
        if got.a.is_finite() && want.a.is_finite() {
            assert_scaled_close(got.a, want.a, 1.0e-10, &format!("{label} a"));
        } else {
            assert_eq!(got.a.is_infinite(), want.a.is_infinite(), "{label} a inf");
        }
        assert_close(got.ecc, want.ecc, 1.0e-10, &format!("{label} ecc"));
        assert_angle_close(got.incl, want.incl, 1.0e-10, &format!("{label} incl"));

        let got_resolved = resolve_classical_angles(got).unwrap();
        let want_resolved = resolve_classical_angles(want).unwrap();
        let got_varpi = wrap_to_pi(got_resolved.argp + factor.sign() * got_resolved.raan);
        let want_varpi = wrap_to_pi(want_resolved.argp + factor.sign() * want_resolved.raan);
        assert_angle_close(got_varpi, want_varpi, 1.0e-10, &format!("{label} varpi"));
        assert_angle_close(
            normalize_angle(got_varpi + got_resolved.nu),
            normalize_angle(want_varpi + want_resolved.nu),
            1.0e-10,
            &format!("{label} L"),
        );
    }

    fn assert_eq_close(
        got: &EquinoctialElements,
        want: &EquinoctialElements,
        ecc: f64,
        label: &str,
    ) {
        assert_scaled_close(got.a, want.a, 1.0e-10, &format!("{label} a"));
        assert_close(got.h, want.h, 1.0e-10, &format!("{label} h"));
        assert_close(got.k, want.k, 1.0e-10, &format!("{label} k"));
        assert_close(got.p, want.p, 1.0e-10, &format!("{label} p"));
        assert_close(got.q, want.q, 1.0e-10, &format!("{label} q"));
        if ecc > 1.0 {
            assert_scaled_close(got.lambda, want.lambda, 1.0e-9, &format!("{label} lambda"));
        } else {
            assert_angle_close(got.lambda, want.lambda, 1.0e-10, &format!("{label} lambda"));
        }
        assert_eq!(got.retrograde, want.retrograde, "{label} factor");
    }

    fn assert_mee_close(
        got: &ModifiedEquinoctialElements,
        want: &ModifiedEquinoctialElements,
        label: &str,
    ) {
        assert_scaled_close(got.p, want.p, 1.0e-10, &format!("{label} p"));
        assert_close(got.f, want.f, 1.0e-10, &format!("{label} f"));
        assert_close(got.g, want.g, 1.0e-10, &format!("{label} g"));
        assert_close(got.h, want.h, 1.0e-10, &format!("{label} h"));
        assert_close(got.k, want.k, 1.0e-10, &format!("{label} k"));
        assert_angle_close(got.l, want.l, 1.0e-10, &format!("{label} l"));
        assert_eq!(got.retrograde, want.retrograde, "{label} factor");
    }

    #[test]
    fn round_trip_state_across_regimes() {
        for case in regimes() {
            let state = coe2rv(&case.coe, MU).unwrap();

            let eq = rv2eq(state.0, state.1, MU, case.factor).unwrap();
            let eq_state = eq2rv(&eq, MU).unwrap();
            assert_state_close(eq_state, state, case.name);

            let mee = rv2mee(state.0, state.1, MU, case.factor).unwrap();
            let mee_state = mee2rv(&mee, MU).unwrap();
            assert_state_close(mee_state, state, case.name);
        }
    }

    #[test]
    fn round_trip_element_algebra_across_regimes() {
        for case in regimes() {
            let eq = coe2eq(&case.coe, case.factor).unwrap();
            let eq_back = eq2coe(&eq).unwrap();
            assert_classical_recombined_close(&eq_back, &case.coe, case.factor, case.name);

            let mee = coe2mee(&case.coe, case.factor).unwrap();
            let mee_back = mee2coe(&mee).unwrap();
            assert_classical_recombined_close(&mee_back, &case.coe, case.factor, case.name);

            let converted_mee = eq2mee(&eq).unwrap();
            assert_mee_close(&converted_mee, &mee, case.name);

            let converted_eq = mee2eq(&mee).unwrap();
            assert_eq_close(&converted_eq, &eq, case.coe.ecc, case.name);
        }
    }

    #[test]
    fn vallado_reference_components_match_full_precision_and_rounded_values() {
        let r = [6524.834, 6862.875, 6448.296];
        let v = [4.901327, 5.533756, -1.976341];
        let coe = rv2coe(r, v, MU).unwrap();

        let eq = coe2eq(&coe, RetrogradeFactor::Prograde).unwrap();
        let mee = coe2mee(&coe, RetrogradeFactor::Prograde).unwrap();

        // Derivation: rv2coe gives the full-precision classical oracle for this
        // state. The expected components below are recomputed from those values
        // with the definitions in this module. The rounded check uses Vallado's
        // printed Example 2-5 elements only as a loose external anchor.
        let varpi = wrap_to_pi(coe.argp + coe.raan);
        let mean = true_to_mean(coe.nu, coe.ecc).unwrap();
        let scale = libm::tan(0.5 * coe.incl);
        assert_close(eq.a, coe.a, 1.0e-12, "eq a full");
        assert_close(eq.h, coe.ecc * libm::sin(varpi), 1.0e-12, "eq h full");
        assert_close(eq.k, coe.ecc * libm::cos(varpi), 1.0e-12, "eq k full");
        assert_close(eq.p, scale * libm::sin(coe.raan), 1.0e-12, "eq p full");
        assert_close(eq.q, scale * libm::cos(coe.raan), 1.0e-12, "eq q full");
        assert_angle_close(eq.lambda, mean + varpi, 1.0e-12, "eq lambda full");

        assert_close(mee.p, coe.p, 1.0e-12, "mee p full");
        assert_close(mee.f, coe.ecc * libm::cos(varpi), 1.0e-12, "mee f full");
        assert_close(mee.g, coe.ecc * libm::sin(varpi), 1.0e-12, "mee g full");
        assert_close(mee.h, scale * libm::cos(coe.raan), 1.0e-12, "mee h full");
        assert_close(mee.k, scale * libm::sin(coe.raan), 1.0e-12, "mee k full");
        assert_angle_close(mee.l, varpi + coe.nu, 1.0e-12, "mee l full");

        let e = 0.832853_f64;
        let incl = 87.870 * DEG;
        let raan = 227.898 * DEG;
        let argp = 53.38 * DEG;
        let nu = 92.335 * DEG;
        let varpi = wrap_to_pi(argp + raan);
        let mean = true_to_mean(nu, e).unwrap();
        let scale = libm::tan(0.5 * incl);

        assert_close(eq.a, 36127.343, 1.0e-2, "eq a rounded");
        assert_close(eq.h, e * libm::sin(varpi), 1.0e-4, "eq h rounded");
        assert_close(eq.k, e * libm::cos(varpi), 1.0e-4, "eq k rounded");
        assert_close(eq.p, scale * libm::sin(raan), 1.0e-4, "eq p rounded");
        assert_close(eq.q, scale * libm::cos(raan), 1.0e-4, "eq q rounded");
        assert_angle_close(eq.lambda, mean + varpi, 1.0e-4, "eq lambda rounded");

        assert_close(mee.p, 11067.790, 1.0e-2, "mee p rounded");
        assert_close(mee.f, e * libm::cos(varpi), 1.0e-4, "mee f rounded");
        assert_close(mee.g, e * libm::sin(varpi), 1.0e-4, "mee g rounded");
        assert_close(mee.h, scale * libm::cos(raan), 1.0e-4, "mee h rounded");
        assert_close(mee.k, scale * libm::sin(raan), 1.0e-4, "mee k rounded");
        assert_angle_close(mee.l, varpi + nu, 1.0e-4, "mee l rounded");
    }

    #[test]
    fn mee2rv_near_equatorial_low_eccentricity_regression() {
        let coe = classical(
            7000.0 * (1.0 - 1.0e-4_f64.powi(2)),
            1.0e-4,
            1.0e-3 * DEG,
            15.0 * DEG,
            25.0 * DEG,
            35.0 * DEG,
        );
        let state = coe2rv(&coe, MU).unwrap();
        let mee = coe2mee(&coe, RetrogradeFactor::Prograde).unwrap();
        assert_state_close(mee2rv(&mee, MU).unwrap(), state, "near equatorial mee2rv");
    }

    #[test]
    fn retrograde_pole_errors_and_opposite_factor_round_trips() {
        let coe = elliptical_equatorial(
            11000.0 * (1.0 - 0.2_f64.powi(2)),
            0.2,
            PI,
            40.0 * DEG,
            80.0 * DEG,
        );
        assert_eq!(
            coe2eq(&coe, RetrogradeFactor::Prograde),
            Err(EquinoctialError::RetrogradePole)
        );
        assert_eq!(
            coe2mee(&coe, RetrogradeFactor::Prograde),
            Err(EquinoctialError::RetrogradePole)
        );

        let state = coe2rv(&coe, MU).unwrap();
        let eq = coe2eq(&coe, RetrogradeFactor::Retrograde).unwrap();
        assert_state_close(eq2rv(&eq, MU).unwrap(), state, "retrograde eq");
        let mee = coe2mee(&coe, RetrogradeFactor::Retrograde).unwrap();
        assert_state_close(mee2rv(&mee, MU).unwrap(), state, "retrograde mee");
    }

    #[test]
    fn parabolic_equinoctial_rejected_and_mee_round_trips() {
        let coe = classical(12000.0, 1.0, 30.0 * DEG, 40.0 * DEG, 50.0 * DEG, 60.0 * DEG);
        assert_eq!(
            coe2eq(&coe, RetrogradeFactor::Prograde),
            Err(EquinoctialError::ParabolicEquinoctial)
        );

        let state = coe2rv(&coe, MU).unwrap();
        assert_eq!(
            rv2eq(state.0, state.1, MU, RetrogradeFactor::Prograde),
            Err(EquinoctialError::ParabolicEquinoctial)
        );

        let mee = coe2mee(&coe, RetrogradeFactor::Prograde).unwrap();
        assert_close(mee.p, coe.p, 1.0e-12, "mee parabolic p");
        let back = mee2coe(&mee).unwrap();
        assert!(back.a.is_infinite());
        assert_close(back.p, coe.p, 1.0e-12, "coe parabolic p");
        assert_state_close(mee2rv(&mee, MU).unwrap(), state, "parabolic mee");
    }

    #[test]
    fn input_rejections_are_typed() {
        let coe = classical(
            7200.0 * (1.0 - 0.01_f64.powi(2)),
            0.01,
            40.0 * DEG,
            12.0 * DEG,
            25.0 * DEG,
            80.0 * DEG,
        );
        let state = coe2rv(&coe, MU).unwrap();
        let mut eq = coe2eq(&coe, RetrogradeFactor::Prograde).unwrap();
        eq.h = f64::NAN;
        assert!(matches!(
            eq2coe(&eq),
            Err(EquinoctialError::Elements(ElementsError::NonFinite {
                field: "h"
            }))
        ));

        let mut mee = coe2mee(&coe, RetrogradeFactor::Prograde).unwrap();
        mee.l = f64::INFINITY;
        assert!(matches!(
            mee2coe(&mee),
            Err(EquinoctialError::Elements(ElementsError::NonFinite {
                field: "l"
            }))
        ));

        let eq = coe2eq(&coe, RetrogradeFactor::Prograde).unwrap();
        assert_eq!(
            eq2rv(&eq, 0.0),
            Err(EquinoctialError::Elements(ElementsError::NonPositiveMu))
        );
        assert_eq!(
            rv2eq(state.0, state.1, 0.0, RetrogradeFactor::Prograde),
            Err(EquinoctialError::Elements(ElementsError::NonPositiveMu))
        );
        assert_eq!(
            rv2eq([0.0, 0.0, 0.0], state.1, MU, RetrogradeFactor::Prograde),
            Err(EquinoctialError::Elements(ElementsError::ZeroPosition))
        );
        assert_eq!(
            rv2mee(
                [7000.0, 0.0, 0.0],
                [7.5, 0.0, 0.0],
                MU,
                RetrogradeFactor::Prograde
            ),
            Err(EquinoctialError::Elements(ElementsError::DegenerateOrbit))
        );

        let hyperbolic = classical(10000.0, 1.5, 40.0 * DEG, 20.0 * DEG, 30.0 * DEG, PI);
        assert!(matches!(
            coe2eq(&hyperbolic, RetrogradeFactor::Prograde),
            Err(EquinoctialError::Anomaly(
                AnomalyError::BeyondAsymptote { .. }
            ))
        ));
    }
}
