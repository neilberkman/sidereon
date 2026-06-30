//! Initial orbit determination (IOD) from position or angle observations.
//!
//! Authoritative implementations of the classical IOD methods; language
//! bindings are thin marshaling layers over these functions.
//!
//! - [`gibbs`] - velocity at the middle of three coplanar position vectors
//!   (Algorithm 54, Vallado 2022, pp. 460-467).
//! - [`hgibbs`] - Herrick-Gibbs velocity from three closely-spaced timed
//!   positions (Algorithm 55, Vallado 2022, pp. 467-472).
//! - [`gauss_angles`] - angles-only orbit from three optical sightings
//!   (Algorithm 52, Vallado 2022, pp. 448-459).
//!
//! ## Reference constants
//!
//! The constants prefixed `VALLADO_` below are reference-suite values, NOT the
//! WGS84/EGM datum. They match the Vallado worked examples and the `valladopy`
//! reference suite the unit tests validate against (`VALLADO_MU = 398600.4415`,
//! `VALLADO_RE = 6378.1363`), and are kept local so the methods stay bit-exact
//! with that published reference rather than drifting to the WGS84/GM values in
//! [`crate::astro::constants`]. Callers needing the WGS84/GM datum must use the
//! constants module, not these.

/// Earth gravitational parameter (km^3/s^2), Vallado reference suite value (not
/// the WGS84/GM datum in [`crate::astro::constants`]).
const VALLADO_MU: f64 = 398600.4415;
/// Earth equatorial radius (km), Vallado reference suite value (not the WGS84
/// value in [`crate::astro::constants`]).
const VALLADO_RE: f64 = 6378.1363;
/// Canonical time unit (seconds) for the Gauss canonical-unit formulation,
/// Vallado reference suite value.
const VALLADO_TUSEC: f64 = 806.8109913067327;
// Seconds per day; the canonical core value (bit-identical to the Vallado 86400)
// under the local `DAY2SEC` name the epoch-difference factors below read.
use crate::astro::math::linear::invert_3x3_adjugate;
use crate::astro::math::vec3;
use crate::constants::SECONDS_PER_DAY as DAY2SEC;
const SMALL: f64 = 1e-10;
/// Maximum coplanarity deviation (radians) tolerated by the Gibbs methods. The
/// three position vectors must lie in a common plane to define an orbit; this
/// is the standard few-degree IOD acceptance bound (here 5 degrees).
const COPLANAR_TOL_RAD: f64 = 5.0 * std::f64::consts::PI / 180.0;

/// Error returned by the initial-orbit-determination methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum IodError {
    /// The line-of-sight matrix determinant is too small to invert (degenerate
    /// geometry).
    #[error("line-of-sight determinant too small")]
    DeterminantTooSmall,
    /// The position vectors do not admit an orbit solution (degenerate D or N
    /// vector in the Gibbs construction).
    #[error("orbit determination not possible from the given geometry")]
    OrbitNotPossible,
    /// A supplied position vector has (near) zero magnitude, so it cannot be
    /// normalized.
    #[error("position vector has near-zero magnitude")]
    ZeroVector,
    /// The two position vectors whose cross product is normalized for the
    /// coplanarity check are collinear, leaving the coplanarity angle undefined.
    #[error("position vectors are collinear")]
    CollinearVectors,
    /// The three position vectors are not sufficiently coplanar to define a
    /// single orbit.
    #[error("position vectors are not coplanar")]
    NotCoplanar,
    /// The observation times are equal or near-equal, so the time geometry is
    /// degenerate (zero denominators).
    #[error("observation times are equal or near-equal")]
    InvalidTimeGeometry,
    /// The Gauss radius polynomial's root is non-positive or outside the
    /// supported geocentric-radius range (see [`gauss_angles`]).
    #[error("no positive real root for the slant-range polynomial")]
    NoPositiveRoot,
    /// The Gauss radius root solver failed numerically (degenerate Halley
    /// denominator, non-finite iterate, or no convergence within the iteration
    /// cap), so no trustworthy root is available.
    #[error("slant-range root solver did not converge")]
    RootSolveFailed,
    /// An intermediate or output value was not finite (NaN or infinity).
    #[error("non-finite value encountered")]
    NonFiniteValue,
}

/// True when every component of a 3-vector is finite.
fn all_finite(a: &[f64; 3]) -> bool {
    a.iter().all(|x| x.is_finite())
}

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    vec3::cross3_ref(a, b)
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    vec3::dot3_ref(a, b)
}

fn mag(a: &[f64; 3]) -> f64 {
    vec3::norm3_ref(a)
}

fn unit(a: &[f64; 3]) -> [f64; 3] {
    vec3::unit3_ref_unchecked(a)
}

// Kept local: the shared `vec3::add3` debug-asserts finiteness, but the
// Gibbs/Herrick-Gibbs overflow guards intentionally let a non-finite
// intermediate sum form and then reject it (returning `NonFiniteValue`), so a
// finiteness assertion here would change that behavior.
fn vadd(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

// `s * a[i]` and `a[i] * s` are bitwise identical (IEEE multiplication is
// commutative), so the shared `scale3` preserves the prior operation order.
fn smul(s: f64, a: &[f64; 3]) -> [f64; 3] {
    vec3::scale3(*a, s)
}

/// 3x3 matrix determinant.
fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// 3x3 matrix-vector multiply.
fn mat3_vec3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

// IOD-local 3x3 product. Kept distinct from `astro::math::mat3::inline_rxr`
// (which accumulates from 0.0): this evaluates each entry as the single
// `t0 + t1 + t2` expression the IOD goldens were captured against, so it stays
// bit-identical (the two differ only on signed-zero / NaN intermediates).
fn mat3_mat3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

/// Line-of-sight unit vector from right ascension and declination (radians).
fn los(ra: f64, dec: f64) -> [f64; 3] {
    [dec.cos() * ra.cos(), dec.cos() * ra.sin(), dec.sin()]
}

/// Gibbs method: determine the velocity at `r2` from three coplanar position
/// vectors (km).
///
/// Algorithm 54, Vallado 2022, pp. 460-467.
///
/// Returns `(v2, theta12_rad, theta23_rad, copa_rad)`: the velocity at `r2`
/// (km/s), the angles between successive position vectors, and the coplanarity
/// angle.
pub fn gibbs(
    r1: &[f64; 3],
    r2: &[f64; 3],
    r3: &[f64; 3],
) -> Result<([f64; 3], f64, f64, f64), IodError> {
    // Validate inputs before any normalization or division so a degenerate
    // input surfaces as a typed error instead of a NaN inside `Ok`.
    if !all_finite(r1) || !all_finite(r2) || !all_finite(r3) {
        return Err(IodError::NonFiniteValue);
    }

    let magr1 = mag(r1);
    let magr2 = mag(r2);
    let magr3 = mag(r3);

    // A finite input whose magnitude overflows to infinity would silently
    // collapse later normalizations to zero and fabricate a finite result, so
    // reject non-finite magnitudes outright.
    if !magr1.is_finite() || !magr2.is_finite() || !magr3.is_finite() {
        return Err(IodError::NonFiniteValue);
    }

    if magr1 < SMALL || magr2 < SMALL || magr3 < SMALL {
        return Err(IodError::ZeroVector);
    }

    // Cross products
    let p = cross(r2, r3);
    let q = cross(r3, r1);
    let w = cross(r1, r2);

    // Only `p` is normalized (for the coplanarity angle), so only r2/r3
    // collinearity is fatal here. `q` or `w` vanishing on its own (e.g.
    // anti-parallel r1/r2 at the apsides) is a legitimate geometry; the
    // degenerate-orbit case is caught by the `D`/`N` magnitude check below.
    let magp = mag(&p);
    if !magp.is_finite() {
        return Err(IodError::NonFiniteValue);
    }
    if magp < SMALL {
        return Err(IodError::CollinearVectors);
    }

    // Coplanarity angle (now safe: `p` and `r1` are nonzero). Clamp the dot
    // product into asin's domain so floating-point spill past +-1 cannot yield
    // a NaN that slips through the `> tol` test below.
    let copa = dot(&unit(&p), &unit(r1)).clamp(-1.0, 1.0).asin();
    if copa.abs() > COPLANAR_TOL_RAD {
        return Err(IodError::NotCoplanar);
    }

    // D = P + Q + W
    let d = vadd(&vadd(&p, &q), &w);
    let magd = mag(&d);

    // N = |r1|*P + |r2|*Q + |r3|*W
    let n = vadd(&vadd(&smul(magr1, &p), &smul(magr2, &q)), &smul(magr3, &w));
    let magn = mag(&n);

    if !magd.is_finite() || !magn.is_finite() {
        return Err(IodError::NonFiniteValue);
    }
    if magd < 1e-6 || magn < 1e-6 {
        return Err(IodError::OrbitNotPossible);
    }

    // Angles between position vectors
    let theta12 = (dot(r1, r2) / (magr1 * magr2)).clamp(-1.0, 1.0).acos();
    let theta23 = (dot(r2, r3) / (magr2 * magr3)).clamp(-1.0, 1.0).acos();

    // S vector
    let r1mr2 = magr1 - magr2;
    let r3mr1 = magr3 - magr1;
    let r2mr3 = magr2 - magr3;
    let s = vadd(&vadd(&smul(r1mr2, r3), &smul(r3mr1, r2)), &smul(r2mr3, r1));

    // B = D x r2
    let b = cross(&d, r2);

    // Scaling factor
    let lg = (VALLADO_MU / (magd * magn)).sqrt();

    // v2 = (lg / |r2|) * B + lg * S
    let v2 = vadd(&smul(lg / magr2, &b), &smul(lg, &s));

    // Guard against non-finite results that can arise from finite-but-extreme
    // inputs (e.g. magnitudes that overflow to infinity).
    if !all_finite(&v2) || !theta12.is_finite() || !theta23.is_finite() || !copa.is_finite() {
        return Err(IodError::NonFiniteValue);
    }

    Ok((v2, theta12, theta23, copa))
}

/// Herrick-Gibbs method: determine the velocity at `r2` from three
/// closely-spaced position vectors (km) with timestamps.
///
/// Algorithm 55, Vallado 2022, pp. 467-472.
///
/// `jd1`, `jd2`, `jd3` are the observation epochs in Julian days. The method
/// converts epoch differences to seconds via the `* DAY2SEC` factor below, so
/// the inputs must be Julian days (not seconds or an arbitrary unit); the
/// epochs must be distinct.
///
/// Returns `(v2, theta12_rad, theta23_rad, copa_rad)`.
pub fn hgibbs(
    r1: &[f64; 3],
    r2: &[f64; 3],
    r3: &[f64; 3],
    jd1: f64,
    jd2: f64,
    jd3: f64,
) -> Result<([f64; 3], f64, f64, f64), IodError> {
    // Validate inputs before any normalization or division so a degenerate
    // input surfaces as a typed error instead of a NaN inside `Ok`.
    if !all_finite(r1)
        || !all_finite(r2)
        || !all_finite(r3)
        || !jd1.is_finite()
        || !jd2.is_finite()
        || !jd3.is_finite()
    {
        return Err(IodError::NonFiniteValue);
    }

    let magr1 = mag(r1);
    let magr2 = mag(r2);
    let magr3 = mag(r3);

    // Reject magnitudes that overflowed to infinity (finite but huge inputs),
    // which would otherwise collapse the normalization below to a fake result.
    if !magr1.is_finite() || !magr2.is_finite() || !magr3.is_finite() {
        return Err(IodError::NonFiniteValue);
    }

    if magr1 < SMALL || magr2 < SMALL || magr3 < SMALL {
        return Err(IodError::ZeroVector);
    }

    // Time differences (seconds; inputs are Julian days).
    let dt21 = (jd2 - jd1) * DAY2SEC;
    let dt31 = (jd3 - jd1) * DAY2SEC;
    let dt32 = (jd3 - jd2) * DAY2SEC;

    // Equal or near-equal epochs make the divisors below blow up.
    if dt21.abs() < SMALL || dt31.abs() < SMALL || dt32.abs() < SMALL {
        return Err(IodError::InvalidTimeGeometry);
    }

    // Cross product for coplanarity check; must be finite and nonzero to
    // normalize.
    let p = cross(r2, r3);
    let magp = mag(&p);
    if !magp.is_finite() {
        return Err(IodError::NonFiniteValue);
    }
    if magp < SMALL {
        return Err(IodError::CollinearVectors);
    }

    // Coplanarity angle (now safe: `p` and `r1` are nonzero). Clamp the dot
    // product into asin's domain so floating-point spill past +-1 cannot yield
    // a NaN that slips through the `> tol` test below.
    let copa = dot(&unit(&p), &unit(r1)).clamp(-1.0, 1.0).asin();
    if copa.abs() > COPLANAR_TOL_RAD {
        return Err(IodError::NotCoplanar);
    }

    // Angles between position vectors
    let theta12 = (dot(r1, r2) / (magr1 * magr2)).clamp(-1.0, 1.0).acos();
    let theta23 = (dot(r2, r3) / (magr2 * magr3)).clamp(-1.0, 1.0).acos();

    // Herrick-Gibbs velocity approximation
    let term1 = smul(
        -dt32 * (1.0 / (dt21 * dt31) + VALLADO_MU / (12.0 * magr1.powi(3))),
        r1,
    );
    let term2 = smul(
        (dt32 - dt21) * (1.0 / (dt21 * dt32) + VALLADO_MU / (12.0 * magr2.powi(3))),
        r2,
    );
    let term3 = smul(
        dt21 * (1.0 / (dt32 * dt31) + VALLADO_MU / (12.0 * magr3.powi(3))),
        r3,
    );

    let v2 = vadd(&vadd(&term1, &term2), &term3);

    // Guard against non-finite results from finite-but-extreme inputs (e.g.
    // epoch differences large enough to overflow the second-scale divisors).
    if !all_finite(&v2) || !theta12.is_finite() || !theta23.is_finite() || !copa.is_finite() {
        return Err(IodError::NonFiniteValue);
    }

    Ok((v2, theta12, theta23, copa))
}

/// Halley iteration to refine the 8th-order Gauss polynomial root.
///
/// Returns `None` if the Halley denominator vanishes, the iterate becomes
/// non-finite, or the iteration does not converge within the cap, so the caller
/// can surface a typed error rather than propagate a NaN/Inf or non-root value.
fn halley_iteration(poly: &[f64; 9]) -> Option<f64> {
    // Initial guess at roughly GPS altitude in canonical (Earth-radii) units.
    let mut bigr2c = 20000.0 / VALLADO_RE;
    let mut bigr2 = 100.0;
    let mut converged = false;

    for _ in 0..15 {
        if (bigr2 - bigr2c).abs() < 8e-5 {
            converged = true;
            break;
        }
        bigr2 = bigr2c;
        let x = bigr2;
        let f = x.powi(8) + poly[2] * x.powi(6) + poly[5] * x.powi(3) + poly[8];
        let f1 = 8.0 * x.powi(7) + 6.0 * poly[2] * x.powi(5) + 3.0 * poly[5] * x.powi(2);
        let f2 = 56.0 * x.powi(6) + 30.0 * poly[2] * x.powi(4) + 6.0 * poly[5] * x;
        let denom = 2.0 * f1 * f1 - f * f2;
        if denom.abs() < SMALL {
            return None;
        }
        bigr2c = bigr2 - (2.0 * f * f1) / denom;
        if !bigr2c.is_finite() {
            return None;
        }
    }

    // Convergence is decided by the final step size, not the loop count, so an
    // iterate that converges on the last allowed pass still counts; an iterate
    // that merely ran out of iterations without settling is rejected rather than
    // returned as a fabricated root.
    if !converged {
        converged = (bigr2 - bigr2c).abs() < 8e-5;
    }

    if !converged || !bigr2c.is_finite() {
        return None;
    }

    // A zero Halley step also occurs at a stationary point of f (f1 == 0) that
    // is not a root, which the step-size test alone would accept. Confirm the
    // polynomial actually vanishes there via a scale-relative residual.
    let x = bigr2c;
    let t0 = x.powi(8);
    let t2 = poly[2] * x.powi(6);
    let t5 = poly[5] * x.powi(3);
    let t8 = poly[8];
    let residual = t0 + t2 + t5 + t8;
    let scale = t0.abs() + t2.abs() + t5.abs() + t8.abs();
    if !residual.is_finite() || !scale.is_finite() || residual.abs() > 1e-9 * scale {
        return None;
    }

    Some(bigr2c)
}

/// Gauss angles-only orbit determination.
///
/// Given three angular observations (right ascension / declination, radians)
/// with split Julian dates (`jd` whole part, `jdf` fraction) and the observer
/// site ECI positions (km), determine the orbit at the middle observation.
///
/// Algorithm 52, Vallado 2022, pp. 448-459. Returns `(r2, v2)`: the position
/// (km) and velocity (km/s) at the middle epoch.
///
/// Domain: the radius root solver is seeded near GPS altitude and accepts a
/// geocentric radius in the near-Earth-through-GEO regime (positive and up to
/// ~50,000 km). A converged root outside that range yields
/// [`IodError::NoPositiveRoot`]; a numerical failure of the solver yields
/// [`IodError::RootSolveFailed`].
pub fn gauss_angles(
    decl: &[f64; 3],
    rtasc: &[f64; 3],
    jd: &[f64; 3],
    jdf: &[f64; 3],
    rseci: &[[f64; 3]; 3],
) -> Result<([f64; 3], [f64; 3]), IodError> {
    // Reject non-finite inputs up front: a NaN angle would slip past the
    // determinant guard below (NaN comparisons are always false).
    if !decl.iter().all(|x| x.is_finite())
        || !rtasc.iter().all(|x| x.is_finite())
        || !jd.iter().all(|x| x.is_finite())
        || !jdf.iter().all(|x| x.is_finite())
        || !rseci.iter().all(all_finite)
    {
        return Err(IodError::NonFiniteValue);
    }

    // Time intervals (seconds)
    let tau12 = ((jd[0] - jd[1]) + (jdf[0] - jdf[1])) * DAY2SEC;
    let _tau13 = ((jd[0] - jd[2]) + (jdf[0] - jdf[2])) * DAY2SEC;
    let tau32 = ((jd[2] - jd[1]) + (jdf[2] - jdf[1])) * DAY2SEC;

    // Equal or near-equal observation times make the polynomial-coefficient
    // divisors below degenerate.
    if tau12.abs() < SMALL || tau32.abs() < SMALL || (tau32 - tau12).abs() < SMALL {
        return Err(IodError::InvalidTimeGeometry);
    }

    // Line-of-sight vectors
    let l1 = los(rtasc[0], decl[0]);
    let l2 = los(rtasc[1], decl[1]);
    let l3 = los(rtasc[2], decl[2]);

    // Canonical units
    let tau12c = tau12 / VALLADO_TUSEC;
    let tau32c = tau32 / VALLADO_TUSEC;
    let rseci1c = smul(1.0 / VALLADO_RE, &rseci[0]);
    let rseci2c = smul(1.0 / VALLADO_RE, &rseci[1]);
    let rseci3c = smul(1.0 / VALLADO_RE, &rseci[2]);

    // L-matrix (columns = LOS vectors)
    let lmat = [
        [l1[0], l2[0], l3[0]],
        [l1[1], l2[1], l3[1]],
        [l1[2], l2[2], l3[2]],
    ];

    let d = det3(&lmat);
    if d.abs() < SMALL {
        return Err(IodError::DeterminantTooSmall);
    }

    // The determinant guard above keeps `|d| >= SMALL`, which exceeds the
    // adjugate inverter's `PIVOT_EPSILON` floor, so this never yields `None` on
    // reachable geometry; the `?` simply re-maps the degenerate case to the same
    // typed error.
    let lmati = invert_3x3_adjugate(&lmat).ok_or(IodError::DeterminantTooSmall)?;

    // Range-site matrix (columns = site vectors in canonical units)
    let rsmatc = [
        [rseci1c[0], rseci2c[0], rseci3c[0]],
        [rseci1c[1], rseci2c[1], rseci3c[1]],
        [rseci1c[2], rseci2c[2], rseci3c[2]],
    ];

    let lir = mat3_mat3(&lmati, &rsmatc);

    // Polynomial coefficients
    let a1 = tau32c / (tau32c - tau12c);
    let a1u = (tau32c * ((tau32c - tau12c).powi(2) - tau32c.powi(2))) / (6.0 * (tau32c - tau12c));
    let a3 = -tau12c / (tau32c - tau12c);
    let a3u = -(tau12c * ((tau32c - tau12c).powi(2) - tau12c.powi(2))) / (6.0 * (tau32c - tau12c));

    let d1c = lir[1][0] * a1 - lir[1][1] + lir[1][2] * a3;
    let d2c = lir[1][0] * a1u + lir[1][2] * a3u;
    let magrs2 = mag(&rseci2c);
    let l2dotrs = dot(&l2, &rseci2c);

    // 8th-order polynomial
    let mut poly = [0.0; 9];
    poly[0] = 1.0;
    poly[2] = -(d1c.powi(2) + 2.0 * d1c * l2dotrs + magrs2.powi(2));
    poly[5] = -2.0 * (l2dotrs * d2c + d1c * d2c);
    poly[8] = -(d2c.powi(2));

    // Solve for radius. Accept only a converged root in the supported physical
    // range; surface a typed error rather than substitute a fabricated orbit.
    // Distinguish a solver failure (`None`) from a converged-but-out-of-range
    // root so callers can tell the two apart.
    let bigr2c = match halley_iteration(&poly) {
        Some(r) if r > 0.0 && r * VALLADO_RE <= 50000.0 => r,
        Some(_) => return Err(IodError::NoPositiveRoot),
        None => return Err(IodError::RootSolveFailed),
    };

    let bigr2 = bigr2c * VALLADO_RE;
    let a1u_sec = a1u * VALLADO_TUSEC.powi(2);
    let a3u_sec = a3u * VALLADO_TUSEC.powi(2);

    // Solve for f and g series
    let u = VALLADO_MU / bigr2.powi(3);
    let c1 = a1 + a1u_sec * u;
    let c2 = -1.0;
    let c3 = a3 + a3u_sec * u;

    // The reconstructed positions divide by c1 and c3; a vanishing coefficient
    // means the geometry does not yield an orbit.
    if c1.abs() < SMALL || c3.abs() < SMALL {
        return Err(IodError::OrbitNotPossible);
    }

    // Range-site matrix (non-canonical)
    let rsmat = [
        [rseci[0][0], rseci[1][0], rseci[2][0]],
        [rseci[0][1], rseci[1][1], rseci[2][1]],
        [rseci[0][2], rseci[1][2], rseci[2][2]],
    ];
    let lir_full = mat3_mat3(&lmati, &rsmat);
    let cmat = [-c1, -c2, -c3];
    let rhomat = mat3_vec3(&lir_full, &cmat);

    // Form position vectors
    let r1 = vadd(&smul(rhomat[0] / c1, &l1), &rseci[0]);
    let r2 = vadd(&smul(rhomat[1] / c2, &l2), &rseci[1]);
    let r3 = vadd(&smul(rhomat[2] / c3, &l3), &rseci[2]);

    // Use Gibbs to recover the velocity at the middle epoch.
    let (v2, _, _, _) = gibbs(&r1, &r2, &r3)?;

    if !all_finite(&r2) || !all_finite(&v2) {
        return Err(IodError::NonFiniteValue);
    }

    Ok((r2, v2))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn assert_rel(actual: f64, expected: f64, label: &str) {
        let rel = ((actual - expected) / expected).abs();
        assert!(rel < 1e-12, "{label}: relative error {rel:e} exceeds 1e-12");
    }

    // Bit-for-bit reference: valladopy. Gibbs/Herrick-Gibbs velocities are
    // exact; the reported angles match to a few ULP.
    fn assert_ulp(actual: f64, expected: f64, max_ulps: i64, label: &str) {
        if actual == expected {
            return;
        }
        let a = actual.to_bits() as i64;
        let b = expected.to_bits() as i64;
        let ulps = (a - b).abs();
        assert!(
            ulps <= max_ulps,
            "{label}: {ulps} ulps exceeds {max_ulps} (got {actual}, expected {expected})"
        );
    }

    #[test]
    fn gibbs_example_7_3() {
        let r1 = [0.0, 0.0, 6378.1363];
        let r2 = [0.0, -4464.696, -5102.509];
        let r3 = [0.0, 5740.323, 3189.068];

        let (v2, theta12, theta23, copa) = gibbs(&r1, &r2, &r3).unwrap();
        assert_ulp(v2[0], 0.0, 0, "v2_x");
        assert_ulp(v2[1], 5.5311472050176125, 0, "v2_y");
        assert_ulp(v2[2], -5.191806413494606, 0, "v2_z");
        assert_ulp(theta12 * 180.0 / PI, 138.81407085944375, 2, "theta12");
        assert_ulp(theta23 * 180.0 / PI, 160.24053069723146, 2, "theta23");
        assert_ulp(copa, 0.0, 0, "copa");
    }

    #[test]
    fn hgibbs_example_7_4() {
        let r1 = [3419.85564, 6019.82602, 2784.60022];
        let r2 = [2935.91195, 6326.18324, 2660.59584];
        let r3 = [2434.95202, 6597.38674, 2521.52311];
        let jd1 = 0.0;
        let jd2 = (60.0 + 16.48) / 86400.0;
        let jd3 = (120.0 + 33.04) / 86400.0;

        let (v2, theta12, theta23, _copa) = hgibbs(&r1, &r2, &r3, jd1, jd2, jd3).unwrap();
        assert_ulp(v2[0], -6.441557227511062, 0, "v2_x");
        assert_ulp(v2[1], 3.777559606719521, 0, "v2_y");
        assert_ulp(v2[2], -1.7205675602414345, 0, "v2_z");
        assert_ulp(theta12 * 180.0 / PI, 4.499996147374992, 2, "theta12");
        assert_ulp(theta23 * 180.0 / PI, 4.499998402168982, 2, "theta23");
    }

    #[test]
    fn gauss_example_7_2() {
        let d2r = |d: f64| d * PI / 180.0;
        let decl = [d2r(18.667717), d2r(35.664741), d2r(36.996583)];
        let rtasc = [d2r(0.939913), d2r(45.025748), d2r(67.886655)];
        let jd = [2_456_159.5, 2_456_159.5, 2_456_159.5];
        let jdf = [0.4864351851851852, 0.49199074074074073, 0.4947685185185185];
        let rseci = [
            [4054.881, 2748.195, 4074.237],
            [3956.224, 2888.232, 4074.364],
            [3905.073, 2956.935, 4074.430],
        ];

        let (r2, v2) = gauss_angles(&decl, &rtasc, &jd, &jdf, &rseci).unwrap();
        assert_rel(r2[0], 6313.378130210396, "r2_x");
        assert_rel(r2[1], 5247.50563344895, "r2_y");
        assert_rel(r2[2], 6467.707164431651, "r2_z");
        assert_rel(v2[0], -4.185488280436629, "v2_x");
        assert_rel(v2[1], 4.7884929168898145, "v2_y");
        assert_rel(v2[2], 1.721714659663034, "v2_z");
    }

    // --- Degenerate-input rejection (no NaN/Inf or fabricated value in Ok). ---

    #[test]
    fn gibbs_rejects_zero_vector() {
        let r2 = [0.0, -4464.696, -5102.509];
        let r3 = [0.0, 5740.323, 3189.068];
        assert_eq!(
            gibbs(&[0.0, 0.0, 0.0], &r2, &r3).unwrap_err(),
            IodError::ZeroVector
        );
    }

    #[test]
    fn gibbs_rejects_collinear_vectors() {
        // r2 and r3 are parallel: cross(r2, r3) vanishes.
        let r1 = [0.0, 0.0, 6378.1363];
        let r2 = [0.0, 1000.0, 0.0];
        let r3 = [0.0, 2000.0, 0.0];
        assert_eq!(
            gibbs(&r1, &r2, &r3).unwrap_err(),
            IodError::CollinearVectors
        );
    }

    #[test]
    fn gibbs_rejects_noncoplanar_vectors() {
        // Push r1 well out of the r2/r3 plane.
        let r1 = [6378.1363, 6378.1363, 6378.1363];
        let r2 = [0.0, -4464.696, -5102.509];
        let r3 = [0.0, 5740.323, 3189.068];
        assert_eq!(gibbs(&r1, &r2, &r3).unwrap_err(), IodError::NotCoplanar);
    }

    #[test]
    fn gibbs_rejects_nonfinite_input() {
        let r2 = [0.0, -4464.696, -5102.509];
        let r3 = [0.0, 5740.323, 3189.068];
        assert_eq!(
            gibbs(&[f64::NAN, 0.0, 6378.1363], &r2, &r3).unwrap_err(),
            IodError::NonFiniteValue
        );
    }

    #[test]
    fn gibbs_accepts_antiparallel_endpoints() {
        // r1 and r3 are anti-parallel (q = r3 x r1 = 0), but this is a valid
        // coplanar geometry: three points 90 degrees apart on a circular orbit.
        // The solver must not reject it as collinear; the middle velocity is the
        // circular speed, tangent to r2.
        let r1 = [7000.0, 0.0, 0.0];
        let r2 = [0.0, 7000.0, 0.0];
        let r3 = [-7000.0, 0.0, 0.0];
        let (v2, _, _, copa) = gibbs(&r1, &r2, &r3).unwrap();
        let vcirc = (VALLADO_MU / 7000.0).sqrt();
        assert!((v2[0] + vcirc).abs() < 1e-9, "v2_x {} vs {}", v2[0], -vcirc);
        assert!(v2[1].abs() < 1e-9 && v2[2].abs() < 1e-9);
        assert!(copa.abs() < 1e-12);
    }

    #[test]
    fn gibbs_rejects_overflowing_magnitude() {
        // Finite but astronomically large inputs whose magnitudes/cross products
        // overflow to infinity must not collapse into a fabricated finite Ok.
        let r1 = [1e160, 0.0, 0.0];
        let r2 = [0.0, 1e160, 0.0];
        let r3 = [0.0, 0.0, 1e160];
        assert_eq!(gibbs(&r1, &r2, &r3).unwrap_err(), IodError::NonFiniteValue);
    }

    #[test]
    fn halley_iteration_reports_stationary_nonroot() {
        // Coefficients chosen so f'(x0) == 0 at the seed x0 = 20000/RE while
        // f(x0) != 0: the Halley step is zero, which the step-size test alone
        // would accept. The residual check must reject this stationary non-root.
        let x0 = 20000.0 / VALLADO_RE;
        let mut poly = [0.0; 9];
        poly[0] = 1.0;
        poly[2] = -x0.powi(2);
        poly[5] = -(2.0 / 3.0) * x0.powi(5);
        poly[8] = -x0.powi(8) / 9.0;
        assert_eq!(halley_iteration(&poly), None);
    }

    #[test]
    fn halley_iteration_reports_nonconvergence() {
        // f(x) = x^8 has its only root at 0; from the GPS-altitude seed the
        // Halley step shrinks geometrically and does not reach the tolerance
        // within the iteration cap, so the solver reports failure (None) rather
        // than returning a non-root iterate.
        let poly = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert_eq!(halley_iteration(&poly), None);
    }

    #[test]
    fn hgibbs_rejects_equal_times() {
        let r1 = [3419.85564, 6019.82602, 2784.60022];
        let r2 = [2935.91195, 6326.18324, 2660.59584];
        let r3 = [2434.95202, 6597.38674, 2521.52311];
        // jd1 == jd2 -> zero time difference.
        let jd = 0.0;
        assert_eq!(
            hgibbs(&r1, &r2, &r3, jd, jd, (120.0 + 33.04) / 86400.0).unwrap_err(),
            IodError::InvalidTimeGeometry
        );
    }

    #[test]
    fn hgibbs_rejects_zero_vector() {
        let r2 = [2935.91195, 6326.18324, 2660.59584];
        let r3 = [2434.95202, 6597.38674, 2521.52311];
        let jd1 = 0.0;
        let jd2 = (60.0 + 16.48) / 86400.0;
        let jd3 = (120.0 + 33.04) / 86400.0;
        assert_eq!(
            hgibbs(&[0.0, 0.0, 0.0], &r2, &r3, jd1, jd2, jd3).unwrap_err(),
            IodError::ZeroVector
        );
    }

    #[test]
    fn hgibbs_rejects_nonfinite_output() {
        // Epoch differences large enough to overflow the second-scale divisors
        // would yield a non-finite velocity; surface it instead of returning
        // NaN/Inf inside Ok.
        let r1 = [3419.85564, 6019.82602, 2784.60022];
        let r2 = [2935.91195, 6326.18324, 2660.59584];
        let r3 = [2434.95202, 6597.38674, 2521.52311];
        let err = hgibbs(&r1, &r2, &r3, 0.0, 1e306, 2e306).unwrap_err();
        assert_eq!(err, IodError::NonFiniteValue);
    }

    #[test]
    fn gauss_rejects_equal_times() {
        let d2r = |d: f64| d * PI / 180.0;
        let decl = [d2r(18.667717), d2r(35.664741), d2r(36.996583)];
        let rtasc = [d2r(0.939913), d2r(45.025748), d2r(67.886655)];
        let jd = [2_456_159.5, 2_456_159.5, 2_456_159.5];
        // First two epochs identical -> tau12 == 0.
        let jdf = [0.49199074074074073, 0.49199074074074073, 0.4947685185185185];
        let rseci = [
            [4054.881, 2748.195, 4074.237],
            [3956.224, 2888.232, 4074.364],
            [3905.073, 2956.935, 4074.430],
        ];
        assert_eq!(
            gauss_angles(&decl, &rtasc, &jd, &jdf, &rseci).unwrap_err(),
            IodError::InvalidTimeGeometry
        );
    }

    #[test]
    fn gauss_rejects_out_of_range_root() {
        let d2r = |d: f64| d * PI / 180.0;
        let decl = [d2r(18.667717), d2r(35.664741), d2r(36.996583)];
        let rtasc = [d2r(0.939913), d2r(45.025748), d2r(67.886655)];
        let jd = [2_456_159.5, 2_456_159.5, 2_456_159.5];
        let rseci = [
            [4054.881, 2748.195, 4074.237],
            [3956.224, 2888.232, 4074.364],
            [3905.073, 2956.935, 4074.430],
        ];
        // Stretch the time gaps 100x past the method's validity so the implied
        // middle range leaves the physical bracket: no positive real root.
        let base = [0.4864351851851852, 0.49199074074074073, 0.4947685185185185];
        let mid = base[1];
        let jdf = [
            mid + (base[0] - mid) * 100.0,
            mid,
            mid + (base[2] - mid) * 100.0,
        ];
        assert_eq!(
            gauss_angles(&decl, &rtasc, &jd, &jdf, &rseci).unwrap_err(),
            IodError::NoPositiveRoot
        );
    }

    #[test]
    fn gauss_rejects_nonfinite_input() {
        let d2r = |d: f64| d * PI / 180.0;
        let decl = [f64::NAN, d2r(35.664741), d2r(36.996583)];
        let rtasc = [d2r(0.939913), d2r(45.025748), d2r(67.886655)];
        let jd = [2_456_159.5, 2_456_159.5, 2_456_159.5];
        let jdf = [0.4864351851851852, 0.49199074074074073, 0.4947685185185185];
        let rseci = [
            [4054.881, 2748.195, 4074.237],
            [3956.224, 2888.232, 4074.364],
            [3905.073, 2956.935, 4074.430],
        ];
        assert_eq!(
            gauss_angles(&decl, &rtasc, &jd, &jdf, &rseci).unwrap_err(),
            IodError::NonFiniteValue
        );
    }
}
