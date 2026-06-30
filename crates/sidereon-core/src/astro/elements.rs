//! State-vector to classical orbital element conversions.
//!
//! Authoritative implementations of the standard two-body conversions between an
//! inertial Cartesian state (ECI position and velocity) and the classical
//! Keplerian elements; language bindings are thin marshaling layers over these
//! functions.
//!
//! - [`rv2coe`] - position/velocity to classical elements (Algorithm 9,
//!   Vallado 2022, pp. 113-116).
//! - [`coe2rv`] - classical elements to position/velocity (Algorithm 10,
//!   Vallado 2022, pp. 118-120).
//!
//! ## Angle conventions
//!
//! All angles in [`ClassicalElements`] are radians, normalized to `[0, 2*pi)`
//! except inclination which lies in `[0, pi]`. The right ascension of the
//! ascending node, argument of perigee, and true anomaly follow the Vallado
//! reference conventions, including the canonical auxiliary outputs that resolve
//! the degenerate elements of circular and equatorial orbits:
//!
//! - argument of latitude (`u = argp + nu`) for circular inclined orbits, where
//!   the argument of perigee is undefined;
//! - longitude of perigee (`lonper`) for elliptical equatorial orbits, where the
//!   node is undefined;
//! - true longitude (`truelon`) for circular equatorial orbits, where both the
//!   node and the argument of perigee are undefined.
//!
//! The auxiliary outputs that do not apply to a given orbit are returned as
//! `f64::NAN`. [`coe2rv`] reads back whichever auxiliary angle the orbit type
//! requires, so a [`ClassicalElements`] produced by [`rv2coe`] round-trips
//! through [`coe2rv`] regardless of orbit type.

use crate::astro::math::vec3;

/// Vallado degeneracy threshold (`small`) for treating eccentricity or
/// inclination as zero when classifying the orbit type (Algorithm 9).
const SMALL: f64 = 1.0e-8;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
const PI: f64 = std::f64::consts::PI;
const HALF_PI: f64 = std::f64::consts::FRAC_PI_2;

/// Geometric classification of a two-body orbit, which determines which of the
/// classical elements are defined and which auxiliary angle resolves the
/// degenerate ones.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrbitType {
    /// Eccentric and inclined: all six classical elements are defined.
    EllipticalInclined,
    /// Eccentric but equatorial: the ascending node is undefined; the longitude
    /// of perigee ([`ClassicalElements::lonper`]) replaces it.
    EllipticalEquatorial,
    /// Circular but inclined: the argument of perigee is undefined; the argument
    /// of latitude ([`ClassicalElements::arglat`]) replaces it.
    CircularInclined,
    /// Circular and equatorial: both the node and the argument of perigee are
    /// undefined; the true longitude ([`ClassicalElements::truelon`]) replaces
    /// them.
    CircularEquatorial,
}

/// Classical (Keplerian) orbital elements in the Vallado convention.
///
/// Angles are radians. [`semi_latus_rectum`](Self::p) is the primary size
/// element used by [`coe2rv`] so the representation stays well defined for
/// parabolic orbits, where the semi-major axis is infinite.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClassicalElements {
    /// Semi-latus rectum `p = h^2 / mu` (km).
    pub p: f64,
    /// Semi-major axis `a` (km). `f64::INFINITY` for a parabolic orbit
    /// (`ecc == 1`).
    pub a: f64,
    /// Eccentricity (dimensionless).
    pub ecc: f64,
    /// Inclination in `[0, pi]` (rad).
    pub incl: f64,
    /// Right ascension of the ascending node in `[0, 2*pi)` (rad). Undefined
    /// (`f64::NAN`) for equatorial orbits.
    pub raan: f64,
    /// Argument of perigee in `[0, 2*pi)` (rad). Undefined (`f64::NAN`) for
    /// circular orbits.
    pub argp: f64,
    /// True anomaly in `[0, 2*pi)` (rad). Undefined (`f64::NAN`) for circular
    /// orbits.
    pub nu: f64,
    /// Argument of latitude `u = argp + nu` in `[0, 2*pi)` (rad). Defined for
    /// circular inclined orbits, `f64::NAN` otherwise.
    pub arglat: f64,
    /// True longitude in `[0, 2*pi)` (rad). Defined for circular equatorial
    /// orbits, `f64::NAN` otherwise.
    pub truelon: f64,
    /// Longitude of perigee in `[0, 2*pi)` (rad). Defined for elliptical
    /// equatorial orbits, `f64::NAN` otherwise.
    pub lonper: f64,
    /// Geometric classification of the orbit.
    pub orbit_type: OrbitType,
}

impl ClassicalElements {
    /// Build a non-degenerate (elliptical inclined) element set from the six
    /// primary elements, leaving the auxiliary special-case angles undefined.
    ///
    /// This is the convenience constructor for ordinary orbits fed to
    /// [`coe2rv`]. For circular or equatorial orbits, populate the relevant
    /// auxiliary angle and [`orbit_type`](Self::orbit_type) directly, or obtain
    /// the element set from [`rv2coe`].
    ///
    /// `p` is the semi-latus rectum (km); the semi-major axis is derived as
    /// `p / (1 - ecc^2)`.
    pub fn new(p: f64, ecc: f64, incl: f64, raan: f64, argp: f64, nu: f64) -> Self {
        let a = if (ecc - 1.0).abs() < SMALL {
            f64::INFINITY
        } else {
            p / (1.0 - ecc * ecc)
        };
        Self {
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
            orbit_type: OrbitType::EllipticalInclined,
        }
    }
}

/// Error returned by the classical-element conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum ElementsError {
    /// An input position, velocity, element, or `mu` was not finite.
    #[error("non-finite input {field}")]
    NonFinite {
        /// The offending input.
        field: &'static str,
    },
    /// The gravitational parameter was not strictly positive.
    #[error("mu must be positive")]
    NonPositiveMu,
    /// The position vector has (near) zero magnitude, so it cannot be
    /// normalized.
    #[error("position vector has near-zero magnitude")]
    ZeroPosition,
    /// The specific angular momentum is (near) zero (rectilinear trajectory), so
    /// no orbit plane is defined.
    #[error("angular momentum is near zero (degenerate orbit)")]
    DegenerateOrbit,
    /// The semi-latus rectum is non-positive, so the orbit is not defined.
    #[error("semi-latus rectum must be positive")]
    NonPositiveSemiLatus,
}

/// Convert an inertial Cartesian state to classical orbital elements.
///
/// `r` is the ECI position (km), `v` the ECI velocity (km/s), and `mu` the
/// gravitational parameter (km^3/s^2). Implements Vallado Algorithm 9 (RV2COE)
/// including the canonical special-case handling for circular and equatorial
/// orbits.
pub fn rv2coe(r: [f64; 3], v: [f64; 3], mu: f64) -> Result<ClassicalElements, ElementsError> {
    validate_finite(&r, "r")?;
    validate_finite(&v, "v")?;
    if !mu.is_finite() {
        return Err(ElementsError::NonFinite { field: "mu" });
    }
    if mu <= 0.0 {
        return Err(ElementsError::NonPositiveMu);
    }

    let magr = vec3::norm3(r);
    let magv = vec3::norm3(v);
    if magr < SMALL {
        return Err(ElementsError::ZeroPosition);
    }

    // Angular momentum h = r x v.
    let hbar = vec3::cross3(r, v);
    let magh = vec3::norm3(hbar);
    if magh < SMALL {
        return Err(ElementsError::DegenerateOrbit);
    }

    // Node vector n = k x h, pointing toward the ascending node.
    let nbar = [-hbar[1], hbar[0], 0.0];
    let magn = vec3::norm3(nbar);

    // Eccentricity vector e = ((v^2 - mu/r) r - (r.v) v) / mu.
    let rdotv = vec3::dot3(r, v);
    let c1 = magv * magv - mu / magr;
    let ebar = [
        (c1 * r[0] - rdotv * v[0]) / mu,
        (c1 * r[1] - rdotv * v[1]) / mu,
        (c1 * r[2] - rdotv * v[2]) / mu,
    ];
    let ecc = vec3::norm3(ebar);

    // Specific mechanical energy and the size elements.
    let sme = magv * magv * 0.5 - mu / magr;
    let a = if sme.abs() > SMALL {
        -mu / (2.0 * sme)
    } else {
        f64::INFINITY
    };
    let p = magh * magh / mu;

    let incl = clamp_acos(hbar[2] / magh);

    let orbit_type = classify(ecc, incl);

    // Right ascension of the ascending node.
    let raan = if magn > SMALL {
        let mut omega = clamp_acos(nbar[0] / magn);
        if nbar[1] < 0.0 {
            omega = TWO_PI - omega;
        }
        omega
    } else {
        f64::NAN
    };

    // Argument of perigee (elliptical inclined only).
    let argp = if orbit_type == OrbitType::EllipticalInclined {
        let mut ap = angle_between(nbar, ebar);
        if ebar[2] < 0.0 {
            ap = TWO_PI - ap;
        }
        ap
    } else {
        f64::NAN
    };

    // True anomaly (any eccentric orbit).
    let nu = if ecc > SMALL {
        let mut ta = angle_between(ebar, r);
        if rdotv < 0.0 {
            ta = TWO_PI - ta;
        }
        ta
    } else {
        f64::NAN
    };

    // Argument of latitude (circular inclined).
    let arglat = if orbit_type == OrbitType::CircularInclined {
        let mut u = angle_between(nbar, r);
        if r[2] < 0.0 {
            u = TWO_PI - u;
        }
        u
    } else {
        f64::NAN
    };

    // Longitude of perigee (elliptical equatorial).
    let lonper = if orbit_type == OrbitType::EllipticalEquatorial {
        let mut lp = clamp_acos(ebar[0] / ecc);
        if ebar[1] < 0.0 {
            lp = TWO_PI - lp;
        }
        if incl > HALF_PI {
            lp = TWO_PI - lp;
        }
        normalize_angle(lp)
    } else {
        f64::NAN
    };

    // True longitude (circular equatorial).
    let truelon = if orbit_type == OrbitType::CircularEquatorial {
        let mut tl = clamp_acos(r[0] / magr);
        if r[1] < 0.0 {
            tl = TWO_PI - tl;
        }
        if incl > HALF_PI {
            tl = TWO_PI - tl;
        }
        normalize_angle(tl)
    } else {
        f64::NAN
    };

    // Canonicalize the shape and orientation elements that the classification
    // collapsed, so coe2rv reconstructs a self-consistent state that round-trips.
    // A circular orbit carries exactly zero eccentricity (its true anomaly folds
    // into the auxiliary angle); an equatorial orbit carries exactly zero or pi
    // inclination (its node folds into the auxiliary angle). Keeping the tiny
    // residual ecc/incl while zeroing the dropped angles would not round-trip.
    let circular = matches!(
        orbit_type,
        OrbitType::CircularInclined | OrbitType::CircularEquatorial
    );
    let equatorial = matches!(
        orbit_type,
        OrbitType::EllipticalEquatorial | OrbitType::CircularEquatorial
    );
    let ecc = if circular { 0.0 } else { ecc };
    let incl = if equatorial {
        if incl > HALF_PI {
            PI
        } else {
            0.0
        }
    } else {
        incl
    };

    Ok(ClassicalElements {
        p,
        a,
        ecc,
        incl,
        raan,
        argp,
        nu,
        arglat,
        truelon,
        lonper,
        orbit_type,
    })
}

/// Convert classical orbital elements to an inertial Cartesian state.
///
/// Returns the ECI position (km) and velocity (km/s) for `mu` (km^3/s^2).
/// Implements Vallado Algorithm 10 (COE2RV), reading whichever auxiliary angle
/// ([`ClassicalElements::arglat`], [`truelon`](ClassicalElements::truelon), or
/// [`lonper`](ClassicalElements::lonper)) the orbit type requires.
pub fn coe2rv(coe: &ClassicalElements, mu: f64) -> Result<([f64; 3], [f64; 3]), ElementsError> {
    if !mu.is_finite() {
        return Err(ElementsError::NonFinite { field: "mu" });
    }
    if mu <= 0.0 {
        return Err(ElementsError::NonPositiveMu);
    }
    if !coe.p.is_finite() {
        return Err(ElementsError::NonFinite { field: "p" });
    }
    if coe.p <= 0.0 {
        return Err(ElementsError::NonPositiveSemiLatus);
    }
    if !coe.ecc.is_finite() {
        return Err(ElementsError::NonFinite { field: "ecc" });
    }
    if !coe.incl.is_finite() {
        return Err(ElementsError::NonFinite { field: "incl" });
    }

    // Resolve the in-plane and orientation angles, substituting the auxiliary
    // special-case angle for any element that is undefined for this orbit type.
    let (raan, argp, nu) = match coe.orbit_type {
        OrbitType::EllipticalInclined => {
            check_angle(coe.raan, "raan")?;
            check_angle(coe.argp, "argp")?;
            check_angle(coe.nu, "nu")?;
            (coe.raan, coe.argp, coe.nu)
        }
        OrbitType::CircularInclined => {
            check_angle(coe.raan, "raan")?;
            check_angle(coe.arglat, "arglat")?;
            (coe.raan, 0.0, coe.arglat)
        }
        OrbitType::EllipticalEquatorial => {
            check_angle(coe.lonper, "lonper")?;
            check_angle(coe.nu, "nu")?;
            (0.0, coe.lonper, coe.nu)
        }
        OrbitType::CircularEquatorial => {
            check_angle(coe.truelon, "truelon")?;
            (0.0, 0.0, coe.truelon)
        }
    };

    let ecc = coe.ecc;
    let p = coe.p;
    let incl = coe.incl;

    let (sin_nu, cos_nu) = nu.sin_cos();

    // Perifocal (PQW) position and velocity.
    let temp = p / (1.0 + ecc * cos_nu);
    let r_pqw = [temp * cos_nu, temp * sin_nu, 0.0];
    let sqrt_mu_p = (mu / p).sqrt();
    let v_pqw = [-sin_nu * sqrt_mu_p, (ecc + cos_nu) * sqrt_mu_p, 0.0];

    // Perifocal-to-ECI rotation R = ROT3(-raan) ROT1(-incl) ROT3(-argp).
    let (sin_raan, cos_raan) = raan.sin_cos();
    let (sin_argp, cos_argp) = argp.sin_cos();
    let (sin_incl, cos_incl) = incl.sin_cos();

    let m11 = cos_raan * cos_argp - sin_raan * sin_argp * cos_incl;
    let m12 = -cos_raan * sin_argp - sin_raan * cos_argp * cos_incl;
    let m21 = sin_raan * cos_argp + cos_raan * sin_argp * cos_incl;
    let m22 = -sin_raan * sin_argp + cos_raan * cos_argp * cos_incl;
    let m31 = sin_argp * sin_incl;
    let m32 = cos_argp * sin_incl;

    let r = [
        m11 * r_pqw[0] + m12 * r_pqw[1],
        m21 * r_pqw[0] + m22 * r_pqw[1],
        m31 * r_pqw[0] + m32 * r_pqw[1],
    ];
    let v = [
        m11 * v_pqw[0] + m12 * v_pqw[1],
        m21 * v_pqw[0] + m22 * v_pqw[1],
        m31 * v_pqw[0] + m32 * v_pqw[1],
    ];

    Ok((r, v))
}

/// Classify an orbit from its eccentricity and inclination per the Vallado
/// degeneracy thresholds.
fn classify(ecc: f64, incl: f64) -> OrbitType {
    let equatorial = incl < SMALL || (incl - PI).abs() < SMALL;
    let circular = ecc < SMALL;
    match (circular, equatorial) {
        (true, true) => OrbitType::CircularEquatorial,
        (true, false) => OrbitType::CircularInclined,
        (false, true) => OrbitType::EllipticalEquatorial,
        (false, false) => OrbitType::EllipticalInclined,
    }
}

/// Normalize an angle to the documented `[0, 2*pi)` range. Applied after the
/// retrograde and quadrant corrections so a value driven to exactly `2*pi` at the
/// boundary wraps back to `0` rather than escaping the half-open range.
#[inline]
fn normalize_angle(x: f64) -> f64 {
    x.rem_euclid(TWO_PI)
}

/// `acos` with the argument clamped to `[-1, 1]` to absorb round-off that would
/// otherwise produce a NaN at the poles of the conversion.
#[inline]
fn clamp_acos(x: f64) -> f64 {
    x.clamp(-1.0, 1.0).acos()
}

/// Unsigned angle between two vectors in `[0, pi]`, clamped against round-off.
#[inline]
fn angle_between(a: [f64; 3], b: [f64; 3]) -> f64 {
    let denom = vec3::norm3(a) * vec3::norm3(b);
    clamp_acos(vec3::dot3(a, b) / denom)
}

#[inline]
fn validate_finite(v: &[f64; 3], field: &'static str) -> Result<(), ElementsError> {
    if v.iter().all(|x| x.is_finite()) {
        Ok(())
    } else {
        Err(ElementsError::NonFinite { field })
    }
}

#[inline]
fn check_angle(x: f64, field: &'static str) -> Result<(), ElementsError> {
    if x.is_finite() {
        Ok(())
    } else {
        Err(ElementsError::NonFinite { field })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Vallado reference suite gravitational parameter (km^3/s^2).
    const MU: f64 = 398600.4418;
    const DEG: f64 = std::f64::consts::PI / 180.0;

    fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
        assert!(
            (got - want).abs() < tol,
            "{what}: got {got}, want {want}, diff {}",
            (got - want).abs()
        );
    }

    fn assert_vec_close(got: [f64; 3], want: [f64; 3], tol: f64, what: &str) {
        for i in 0..3 {
            assert!(
                (got[i] - want[i]).abs() < tol,
                "{what}[{i}]: got {}, want {}, diff {}",
                got[i],
                want[i],
                (got[i] - want[i]).abs()
            );
        }
    }

    /// Vallado 2022 Example 2-5 (RV2COE): the canonical worked example.
    #[test]
    fn rv2coe_vallado_example_2_5() {
        let r = [6524.834, 6862.875, 6448.296];
        let v = [4.901327, 5.533756, -1.976341];

        let coe = rv2coe(r, v, MU).unwrap();

        // Published results (Vallado 2022, Example 2-5).
        assert_close(coe.p, 11067.790, 1.0e-2, "p");
        assert_close(coe.a, 36127.343, 1.0e-2, "a");
        assert_close(coe.ecc, 0.832853, 1.0e-5, "ecc");
        assert_close(coe.incl, 87.870 * DEG, 1.0e-4, "incl");
        assert_close(coe.raan, 227.898 * DEG, 1.0e-4, "raan");
        assert_close(coe.argp, 53.38 * DEG, 1.0e-3, "argp");
        assert_close(coe.nu, 92.335 * DEG, 1.0e-4, "nu");
        assert_eq!(coe.orbit_type, OrbitType::EllipticalInclined);
    }

    /// Vallado Example 2-6 (COE2RV) is the inverse of 2-5: the published
    /// elements must reproduce the state vector.
    #[test]
    fn coe2rv_vallado_example_2_6() {
        // Vallado 2022 Example 2-6 (COE2RV) uses the rounded elements published
        // for that worked example, p = 11067.790 km, e = 0.83285, i = 87.87 deg,
        // raan = 227.89 deg, argp = 53.38 deg, nu = 92.335 deg. These are not bit
        // identical to the higher-precision Example 2-5 outputs, so they must be
        // fed in exactly as published to reproduce the published state.
        let coe = ClassicalElements::new(
            11067.790,
            0.83285,
            87.87 * DEG,
            227.89 * DEG,
            53.38 * DEG,
            92.335 * DEG,
        );

        let (r, v) = coe2rv(&coe, MU).unwrap();

        // The published state is r = [6525.344, 6861.535, 6449.125] km,
        // v = [4.902276, 5.533124, -1.975709] km/s. Tolerance reflects the
        // rounding of the published elements fed back in.
        assert_vec_close(r, [6525.344, 6861.535, 6449.125], 5.0e-2, "r");
        assert_vec_close(v, [4.902276, 5.533124, -1.975709], 1.0e-3, "v");
    }

    #[test]
    fn round_trip_elliptical_inclined() {
        let r = [6524.834, 6862.875, 6448.296];
        let v = [4.901327, 5.533756, -1.976341];

        let coe = rv2coe(r, v, MU).unwrap();
        let (r2, v2) = coe2rv(&coe, MU).unwrap();

        assert_vec_close(r2, r, 1.0e-7, "r");
        assert_vec_close(v2, v, 1.0e-10, "v");
    }

    #[test]
    fn round_trip_circular_inclined() {
        // Circular inclined orbit: argument of perigee is undefined, so the
        // round trip must travel through the argument of latitude. Build the
        // state from elements so it is exactly circular.
        let raan0 = 80.0 * DEG;
        let incl0 = 51.6 * DEG;
        let arglat0 = 135.0 * DEG;

        let mut coe = ClassicalElements::new(7000.0, 0.0, incl0, raan0, 0.0, 0.0);
        coe.orbit_type = OrbitType::CircularInclined;
        coe.argp = f64::NAN;
        coe.nu = f64::NAN;
        coe.arglat = arglat0;

        let (r, v) = coe2rv(&coe, MU).unwrap();
        let back = rv2coe(r, v, MU).unwrap();

        assert_eq!(back.orbit_type, OrbitType::CircularInclined);
        assert!(back.argp.is_nan());
        assert!(back.arglat.is_finite());
        assert_close(back.incl, incl0, 1.0e-10, "incl");
        assert_close(back.raan, raan0, 1.0e-10, "raan");
        assert_close(back.arglat, arglat0, 1.0e-10, "arglat");

        let (r2, v2) = coe2rv(&back, MU).unwrap();
        assert_vec_close(r2, r, 1.0e-8, "r");
        assert_vec_close(v2, v, 1.0e-11, "v");
    }

    #[test]
    fn round_trip_circular_equatorial() {
        // Construct a circular equatorial state directly, then round-trip it.
        let radius = 7000.0_f64;
        let speed = (MU / radius).sqrt();
        let truelon0 = 35.0 * DEG;
        let r = [radius * truelon0.cos(), radius * truelon0.sin(), 0.0];
        let v = [-speed * truelon0.sin(), speed * truelon0.cos(), 0.0];

        let coe = rv2coe(r, v, MU).unwrap();
        assert_eq!(coe.orbit_type, OrbitType::CircularEquatorial);
        assert!(coe.raan.is_nan());
        assert!(coe.argp.is_nan());
        assert!(coe.nu.is_nan());
        assert_close(coe.truelon, truelon0, 1.0e-9, "truelon");

        let (r2, v2) = coe2rv(&coe, MU).unwrap();
        assert_vec_close(r2, r, 1.0e-8, "r");
        assert_vec_close(v2, v, 1.0e-11, "v");
    }

    #[test]
    fn round_trip_elliptical_equatorial() {
        // Eccentric orbit in the equatorial plane: the node is undefined, so the
        // round trip must travel through the longitude of perigee.
        let p = 11067.790_f64;
        let ecc = 0.4_f64;
        let lonper0 = 110.0 * DEG;
        let nu0 = 40.0 * DEG;

        let mut coe = ClassicalElements::new(p, ecc, 0.0, 0.0, 0.0, nu0);
        coe.orbit_type = OrbitType::EllipticalEquatorial;
        coe.raan = f64::NAN;
        coe.argp = f64::NAN;
        coe.lonper = lonper0;

        let (r, v) = coe2rv(&coe, MU).unwrap();
        let back = rv2coe(r, v, MU).unwrap();

        assert_eq!(back.orbit_type, OrbitType::EllipticalEquatorial);
        assert_close(back.ecc, ecc, 1.0e-12, "ecc");
        assert_close(back.incl, 0.0, 1.0e-12, "incl");
        assert_close(back.lonper, lonper0, 1.0e-10, "lonper");
        assert_close(back.nu, nu0, 1.0e-10, "nu");

        let (r2, v2) = coe2rv(&back, MU).unwrap();
        assert_vec_close(r2, r, 1.0e-8, "r");
        assert_vec_close(v2, v, 1.0e-11, "v");
    }

    #[test]
    fn rejects_bad_inputs() {
        let r = [7000.0, 0.0, 0.0];
        let v = [0.0, 7.5, 0.0];

        assert_eq!(rv2coe(r, v, -1.0), Err(ElementsError::NonPositiveMu));
        assert_eq!(
            rv2coe([0.0, 0.0, 0.0], v, MU),
            Err(ElementsError::ZeroPosition)
        );
        // Rectilinear radial motion has zero angular momentum.
        assert_eq!(
            rv2coe(r, [7.5, 0.0, 0.0], MU),
            Err(ElementsError::DegenerateOrbit)
        );
        assert!(matches!(
            rv2coe([f64::NAN, 0.0, 0.0], v, MU),
            Err(ElementsError::NonFinite { field: "r" })
        ));

        let coe = ClassicalElements::new(11067.79, 0.1, 0.5, 0.1, 0.2, 0.3);
        assert_eq!(coe2rv(&coe, 0.0), Err(ElementsError::NonPositiveMu));
        let mut bad = coe;
        bad.p = -1.0;
        assert_eq!(coe2rv(&bad, MU), Err(ElementsError::NonPositiveSemiLatus));
    }
}
