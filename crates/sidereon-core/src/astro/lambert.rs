//! Lambert two-point boundary-value orbit solver (Battin's method).
//!
//! Given two position vectors and a time of flight, solve for the transfer
//! orbit velocities at each endpoint. This is the authoritative implementation;
//! language bindings are thin marshaling layers over it.
//!
//! Algorithm 61, Vallado, *Fundamentals of Astrodynamics and Applications*
//! (2022), pp. 505-510. The velocity transfer uses the Thompson (2013/2018)
//! hodograph formulation.
//!
//! ## Reference constants
//!
//! `VALLADO_MU` below is a reference-suite value, NOT the WGS84/EGM datum. It
//! matches the Vallado worked examples and the `valladopy` reference suite that
//! the unit tests validate against (`VALLADO_MU = 398600.4415`, not the
//! WGS84/GM value in [`crate::astro::constants`]). It is kept local so the
//! solver stays bit-exact with that published reference rather than drifting to
//! a different datum. Callers needing the WGS84/GM datum must use the constants
//! module, not this value.

use std::f64::consts::PI;

/// Earth gravitational parameter (km^3/s^2), Vallado reference suite value (not
/// the WGS84/GM datum in [`crate::astro::constants`]).
const VALLADO_MU: f64 = 398600.4415;
const TWOPI: f64 = 2.0 * PI;
const SMALL: f64 = 1e-10;

use crate::astro::math::vec3;

fn cross(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    vec3::cross3_ref(a, b)
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    vec3::dot3_ref(a, b)
}

fn mag(a: &[f64; 3]) -> f64 {
    vec3::norm3_ref(a)
}

// `s * a[i]` and `a[i] * s` are bitwise identical (IEEE multiplication is
// commutative), so the shared `scale3` preserves the prior operation order.
fn smul(s: f64, a: &[f64; 3]) -> [f64; 3] {
    vec3::scale3(*a, s)
}

// Kept local: the shared `vec3::add3` debug-asserts finiteness, but the Battin
// solver's overflow guards intentionally let a non-finite intermediate sum form
// and then reject it, so a finiteness assertion here would change that behavior.
fn vadd(a: &[f64; 3], b: &[f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

/// Continued fraction for Battin's method (Vallado Eq. 7-69).
fn seebatt(v: f64) -> f64 {
    let c = [
        9.0 / 7.0,
        16.0 / 63.0,
        25.0 / 99.0,
        36.0 / 143.0,
        49.0 / 195.0,
        64.0 / 255.0,
        81.0 / 323.0,
        100.0 / 399.0,
        121.0 / 483.0,
        144.0 / 575.0,
        169.0 / 675.0,
        196.0 / 783.0,
        225.0 / 899.0,
        256.0 / 1023.0,
        289.0 / 1155.0,
        324.0 / 1295.0,
        361.0 / 1443.0,
        400.0 / 1599.0,
        441.0 / 1763.0,
        484.0 / 1935.0,
    ];

    let sqrtopv = (1.0 + v).sqrt();
    let eta = v / (1.0 + sqrtopv).powi(2);

    let mut term2 = 1.0 + c[19] * eta;
    for j in (0..19).rev() {
        term2 = 1.0 + c[j] * eta / term2;
    }

    8.0 * (1.0 + sqrtopv) / (3.0 + 1.0 / (5.0 + eta + (9.0 / 7.0) * eta / term2))
}

/// Continued fraction for Battin's method (Vallado Eq. 7-70).
fn kbatt(v: f64) -> f64 {
    let d = [
        1.0 / 3.0,
        4.0 / 27.0,
        8.0 / 27.0,
        2.0 / 9.0,
        22.0 / 81.0,
        208.0 / 891.0,
        340.0 / 1287.0,
        418.0 / 1755.0,
        598.0 / 2295.0,
        700.0 / 2907.0,
        928.0 / 3591.0,
        1054.0 / 4347.0,
        1330.0 / 5175.0,
        1480.0 / 6075.0,
        1804.0 / 7047.0,
        1978.0 / 8091.0,
        2350.0 / 9207.0,
        2548.0 / 10395.0,
        2968.0 / 11655.0,
        3190.0 / 12987.0,
        3658.0 / 14391.0,
    ];

    // Forward pass
    let mut sum1: f64 = d[0];
    let mut delold: f64 = 1.0;
    let mut termold: f64 = d[0];
    let ktr = 21;

    for di in d.iter().take(ktr).skip(1) {
        if termold.abs() <= 1e-8 {
            break;
        }
        let del = 1.0 / (1.0 + di * v * delold);
        let term = termold * (del - 1.0);
        sum1 += term;
        delold = del;
        termold = term;
    }
    let _ = sum1; // forward pass result not used in final; backward pass is

    // Backward pass
    let mut term2 = 1.0 + d[ktr - 1] * v;
    for i in 0..(ktr - 2) {
        let sum2 = d[ktr - i - 2] * v / term2;
        term2 = 1.0 + sum2;
    }

    d[0] / term2
}

/// Hodograph velocity transfer (Thompson 2013/2018).
fn hodograph(
    r1: &[f64; 3],
    r2: &[f64; 3],
    v1: &[f64; 3],
    p: f64,
    ecc: f64,
    dnu: f64,
    dtsec: f64,
) -> ([f64; 3], [f64; 3]) {
    let magr1 = mag(r1);
    let magr2 = mag(r2);

    let a = VALLADO_MU * (1.0 / magr1 - 1.0 / p);
    let b = (VALLADO_MU * ecc / p).powi(2) - a * a;
    let x1_abs = if b <= 0.0 { 0.0 } else { b.sqrt() };
    let mut x1 = -x1_abs;

    let nvec;
    if dnu.sin().abs() < SMALL {
        // 180-degree transfer
        let cp = cross(r1, v1);
        let ncp = mag(&cp);
        nvec = smul(1.0 / ncp, &cp);

        if ecc < 1.0 {
            let ptx = TWOPI * (p.powi(3) / (VALLADO_MU * (1.0 - ecc * ecc).powi(3))).sqrt();
            if dtsec % ptx > ptx * 0.5 {
                x1 = x1_abs;
            }
        }
    } else {
        // Common path
        let y2a = VALLADO_MU / p - x1 * dnu.sin() + a * dnu.cos();
        let y2b = VALLADO_MU / p + x1 * dnu.sin() + a * dnu.cos();
        if (VALLADO_MU / magr2 - y2b).abs() < (VALLADO_MU / magr2 - y2a).abs() {
            x1 = x1_abs;
        }

        let cp = cross(r1, r2);
        let ncp = mag(&cp);
        nvec = if dnu % TWOPI > PI {
            smul(-1.0 / ncp, &cp)
        } else {
            smul(1.0 / ncp, &cp)
        };
    }

    let sqrtmup = (VALLADO_MU * p).sqrt();
    let nr1 = cross(&nvec, r1);
    let v1t = smul(
        sqrtmup / magr1,
        &vadd(&smul(x1 / VALLADO_MU, r1), &smul(1.0 / magr1, &nr1)),
    );

    let x2 = x1 * dnu.cos() + a * dnu.sin();
    let nr2 = cross(&nvec, r2);
    let v2t = smul(
        sqrtmup / magr2,
        &vadd(&smul(x2 / VALLADO_MU, r2), &smul(1.0 / magr2, &nr2)),
    );

    (v1t, v2t)
}

/// Direction of motion for the Lambert transfer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DirectionOfMotion {
    /// Short-way transfer (transfer angle < 180 degrees).
    Short,
    /// Long-way transfer (transfer angle > 180 degrees).
    Long,
}

/// Direction of energy for the Lambert transfer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DirectionOfEnergy {
    /// Low-energy branch.
    Low,
    /// High-energy branch.
    High,
}

/// Error returned when the Lambert solver cannot produce a valid transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum LambertError {
    /// The Battin iteration did not reach the convergence tolerance.
    #[error("Lambert solver did not converge")]
    NoConvergence,
    /// A supplied position vector has (near) zero magnitude.
    #[error("position vector has near-zero magnitude")]
    ZeroVector,
    /// The time of flight is not strictly positive (or not finite).
    #[error("time of flight must be strictly positive")]
    NonPositiveTimeOfFlight,
    /// The transfer-plane normal is undefined: the endpoints are collinear or
    /// near-collinear (a near-zero-degree transfer, or a near-180-degree
    /// transfer whose plane cannot be fixed from `v1`). Near-collinearity is
    /// judged by `|sin(dnu)| < SMALL`, which is exactly when the endpoint cross
    /// product `r1 x r2` degenerates.
    #[error("transfer-plane geometry is degenerate")]
    DegenerateGeometry,
    /// An intermediate or output value was not finite (NaN or infinity).
    #[error("non-finite value encountered")]
    NonFiniteValue,
}

/// True when every component of a 3-vector is finite.
fn all_finite(a: &[f64; 3]) -> bool {
    a.iter().all(|x| x.is_finite())
}

/// Solve Lambert's problem using Battin's method.
///
/// Given two position vectors (km) and a time of flight (seconds), return the
/// transfer velocity vectors at `r1` and `r2` (km/s). `v1` is only consulted for
/// the degenerate 180-degree transfer where the transfer-plane normal is
/// otherwise undefined.
///
/// Algorithm 61, Vallado 2022, pp. 505-510.
pub fn battin(
    r1: &[f64; 3],
    r2: &[f64; 3],
    v1: &[f64; 3],
    dm: DirectionOfMotion,
    de: DirectionOfEnergy,
    nrev: i32,
    dtsec: f64,
) -> Result<([f64; 3], [f64; 3]), LambertError> {
    // Validate inputs before any division or normalization so degenerate input
    // surfaces as a typed error instead of an Ok(NaN/Inf) transfer.
    if !all_finite(r1) || !all_finite(r2) || !all_finite(v1) {
        return Err(LambertError::NonFiniteValue);
    }

    let magr1 = mag(r1);
    let magr2 = mag(r2);
    // Reject magnitudes that overflowed to infinity (finite but huge inputs);
    // otherwise the plane normalization below collapses to zero and fabricates a
    // finite transfer.
    if !magr1.is_finite() || !magr2.is_finite() {
        return Err(LambertError::NonFiniteValue);
    }
    if magr1 < SMALL || magr2 < SMALL {
        return Err(LambertError::ZeroVector);
    }

    if !dtsec.is_finite() || dtsec <= 0.0 {
        return Err(LambertError::NonPositiveTimeOfFlight);
    }

    // Angle between position vectors
    let cosdeltanu = dot(r1, r2) / (magr1 * magr2);
    let magrcrossr = mag(&cross(r1, r2));
    // The endpoint normal `r1 x r2` is normalized in the hodograph; if its
    // magnitude overflowed, that normalization would collapse to zero.
    if !magrcrossr.is_finite() {
        return Err(LambertError::NonFiniteValue);
    }
    let sign = if dm == DirectionOfMotion::Short {
        1.0
    } else {
        -1.0
    };
    let sindeltanu = sign * magrcrossr / (magr1 * magr2);
    let mut dnu = sindeltanu.atan2(cosdeltanu);
    if dnu < 0.0 {
        dnu += TWOPI;
    }

    // The transfer-plane normal must be well-defined. `|sin(dnu)| < SMALL` means
    // the endpoints are (near-)collinear, so the endpoint normal `r1 x r2` is
    // degenerate. That is recoverable only for a near-180-degree transfer
    // (cosdeltanu < 0) whose plane can instead be fixed from `v1`. The `v1`
    // normal must be finite and nonzero, since the hodograph normalizes it.
    if dnu.sin().abs() < SMALL {
        let n_v1 = mag(&cross(r1, v1));
        let plane_from_v1 = cosdeltanu < 0.0 && n_v1.is_finite() && n_v1 >= SMALL;
        if !plane_from_v1 {
            return Err(LambertError::DegenerateGeometry);
        }
    }

    // Chord and semiperimeter
    let chord = (magr1 * magr1 + magr2 * magr2 - 2.0 * magr1 * magr2 * cosdeltanu).sqrt();
    let s = (magr1 + magr2 + chord) * 0.5;
    let eps = magr2 / magr1 - 1.0;

    // Lambda, L, m
    let lam = (magr1 * magr2).sqrt() / s * (dnu * 0.5).cos();
    let l_ = ((1.0 - lam) / (1.0 + lam)).powi(2);
    let m = 8.0 * VALLADO_MU * dtsec * dtsec / (s.powi(3) * (1.0 + lam).powi(6));

    // Initial guess
    let mut xn = if nrev > 0 { 1.0 + 4.0 * l_ } else { l_ };

    if de == DirectionOfEnergy::High && nrev > 0 {
        // High energy multi-rev case
        xn = 1e-20;
        let mut x = 10.0;
        for _ in 0..20 {
            if (xn - x).abs() < SMALL {
                break;
            }
            x = xn;
            let temp = 1.0 / (2.0 * (l_ - x * x));
            let temp1 = x.sqrt();
            let temp2 = (nrev as f64 * PI * 0.5 + temp1.atan()) / temp1;
            let h1 = temp * (l_ + x) * (1.0 + 2.0 * x + l_);
            let h2 = temp * m * temp1 * ((l_ - x * x) * temp2 - (l_ + x));

            let b = 0.25 * 27.0 * h2 / (temp1 * (1.0 + h1)).powi(3);
            let f = if b < 0.0 {
                2.0 * ((b + 1.0).sqrt().acos() / 3.0).cos()
            } else {
                let a_ = (b.sqrt() + (b + 1.0).sqrt()).powf(1.0 / 3.0);
                a_ + 1.0 / a_
            };

            let y = 2.0 / 3.0 * temp1 * (1.0 + h1) * ((b + 1.0).sqrt() / f + 1.0);
            xn = 0.5
                * ((m / (y * y) - (1.0 + l_))
                    - ((m / (y * y) - (1.0 + l_)).powi(2) - 4.0 * l_).sqrt());
        }

        // Convergence is decided by the final step size, not the loop count, so
        // an iterate that converges on the last allowed pass is not falsely
        // reported as non-convergent. A diverging (NaN) step fails this test and
        // is reported as non-convergence rather than propagated.
        let converged = (xn - x).abs() < SMALL;
        if !converged {
            return Err(LambertError::NoConvergence);
        }

        let a_orbit = s * (1.0 + lam).powi(2) * (1.0 + xn) * (l_ + xn) / (8.0 * xn);
        let p = 2.0 * magr1 * magr2 * (1.0 + xn) * (dnu * 0.5).sin().powi(2)
            / (s * (1.0 + lam).powi(2) * (l_ + xn));
        let ecc = (1.0 - p / a_orbit).abs().sqrt();
        finite_or_err(hodograph(r1, r2, v1, p, ecc, dnu, dtsec))
    } else {
        // Standard / low energy case
        let mut x = 10.0;
        let max_loops = 30;
        let mut loops = 0;
        let mut y = 0.0;

        while (xn - x).abs() >= SMALL && loops < max_loops {
            x = xn;
            loops += 1;

            let (h1, h2) = if nrev > 0 {
                let temp = 1.0 / ((1.0 + 2.0 * x + l_) * (4.0 * x * x));
                let temp1 = (nrev as f64 * PI * 0.5 + x.sqrt().atan()) / x.sqrt();
                let h1 =
                    temp * (l_ + x).powi(2) * (3.0 * (1.0 + x).powi(2) * temp1 - (3.0 + 5.0 * x));
                let h2 = temp * m * ((x * x - x * (1.0 + l_) - 3.0 * l_) * temp1 + (3.0 * l_ + x));
                (h1, h2)
            } else {
                let tempx = seebatt(x);
                let denom = 1.0 / ((1.0 + 2.0 * x + l_) * (4.0 * x + tempx * (3.0 + x)));
                let h1 = (l_ + x).powi(2) * (1.0 + 3.0 * x + tempx) * denom;
                let h2 = m * (x - l_ + tempx) * denom;
                (h1, h2)
            };

            let b = 0.25 * 27.0 * h2 / (1.0 + h1).powi(3);
            let u = 0.5 * b / (1.0 + (1.0 + b).sqrt());
            let k2 = kbatt(u);
            y = (1.0 + h1) / 3.0 * (2.0 + (1.0 + b).sqrt() / (1.0 + 2.0 * u * k2 * k2));
            xn = (((1.0 - l_) * 0.5).powi(2) + m / (y * y)).sqrt() - (1.0 + l_) * 0.5;
        }

        // Decide convergence by the final step size, not the loop count, so an
        // iterate that settles on the last allowed pass is not falsely rejected;
        // a diverging (NaN) step also fails this test and reports NoConvergence.
        let converged = (xn - x).abs() < SMALL;
        if !converged {
            return Err(LambertError::NoConvergence);
        }

        let p = 2.0 * magr1 * magr2 * y * y * (1.0 + x).powi(2) * (dnu * 0.5).sin().powi(2)
            / (m * s * (1.0 + lam).powi(2));
        let ecc = (eps * eps
            + 4.0 * magr2 / magr1 * (dnu * 0.5).sin().powi(2) * ((l_ - x) / (l_ + x)).powi(2))
            / (eps * eps + 4.0 * magr2 / magr1 * (dnu * 0.5).sin().powi(2));
        let ecc = ecc.sqrt();

        finite_or_err(hodograph(r1, r2, v1, p, ecc, dnu, dtsec))
    }
}

/// Pass the transfer velocities through only if every component is finite,
/// otherwise report a non-finite result as a typed error.
fn finite_or_err(vels: ([f64; 3], [f64; 3])) -> Result<([f64; 3], [f64; 3]), LambertError> {
    let (v1t, v2t) = vels;
    if all_finite(&v1t) && all_finite(&v2t) {
        Ok((v1t, v2t))
    } else {
        Err(LambertError::NonFiniteValue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vallado reference: RE = 6378.1363 km, values via the valladopy suite.
    const RE: f64 = 6378.1363;
    const DTSEC: f64 = 92854.234;
    const NREV: i32 = 1;

    fn inputs() -> ([f64; 3], [f64; 3], [f64; 3]) {
        let r1 = [2.5 * RE, 0.0, 0.0];
        let r2 = [1.9151111 * RE, 1.6069690 * RE, 0.0];
        let v1 = [0.0, 4.999792554221911, 0.0];
        (r1, r2, v1)
    }

    fn assert_close(actual: f64, expected: f64, label: &str) {
        if expected == 0.0 {
            assert!(actual.abs() < 1e-10, "{label}: expected ~0, got {actual}");
        } else {
            let rel = ((actual - expected) / expected).abs();
            assert!(rel < 1e-12, "{label}: relative error {rel:e} exceeds 1e-12");
        }
    }

    #[test]
    fn battin_short_high() {
        let (r1, r2, v1) = inputs();
        let (v1t, v2t) = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Short,
            DirectionOfEnergy::High,
            NREV,
            DTSEC,
        )
        .unwrap();
        assert_close(v1t[0], -0.8696153795282852, "v1t_x");
        assert_close(v1t[1], 6.3351545812502374, "v1t_y");
        assert_close(v1t[2], 0.0, "v1t_z");
        assert_close(v2t[0], -3.405994961791248, "v2t_x");
        assert_close(v2t[1], 5.41198791828363, "v2t_y");
        assert_close(v2t[2], 0.0, "v2t_z");
    }

    #[test]
    fn battin_short_low() {
        let (r1, r2, v1) = inputs();
        let (v1t, v2t) = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Short,
            DirectionOfEnergy::Low,
            NREV,
            DTSEC,
        )
        .unwrap();
        assert_close(v1t[0], 5.832522716212579, "v1t_x");
        assert_close(v1t[1], 1.4319944881331306, "v1t_y");
        assert_close(v2t[0], -5.388439978490882, "v2t_x");
        assert_close(v2t[1], -2.652101898141935, "v2t_y");
    }

    #[test]
    fn battin_long_high() {
        let (r1, r2, v1) = inputs();
        let (v1t, v2t) = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Long,
            DirectionOfEnergy::High,
            NREV,
            DTSEC,
        )
        .unwrap();
        assert_close(v1t[0], -6.241103309400493, "v1t_x");
        assert_close(v1t[1], -1.351339299630816, "v1t_y");
        assert_close(v2t[0], 5.649586715490154, "v2t_x");
        assert_close(v2t[1], 2.976517897853268, "v2t_y");
    }

    #[test]
    fn battin_long_low() {
        let (r1, r2, v1) = inputs();
        let (v1t, v2t) = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Long,
            DirectionOfEnergy::Low,
            NREV,
            DTSEC,
        )
        .unwrap();
        assert_close(v1t[0], 0.641119158614630, "v1t_x");
        assert_close(v1t[1], -5.957501823796459, "v1t_y");
        assert_close(v2t[0], 3.33828270226307, "v2t_x");
        assert_close(v2t[1], -4.975814585231199, "v2t_y");
    }

    #[test]
    fn battin_rejects_zero_position() {
        let (_, r2, v1) = inputs();
        let r1 = [0.0, 0.0, 0.0];
        let err = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Short,
            DirectionOfEnergy::Low,
            NREV,
            DTSEC,
        )
        .unwrap_err();
        assert_eq!(err, LambertError::ZeroVector);
    }

    #[test]
    fn battin_rejects_nonpositive_dtsec() {
        let (r1, r2, v1) = inputs();
        for bad in [0.0, -1.0] {
            let err = battin(
                &r1,
                &r2,
                &v1,
                DirectionOfMotion::Short,
                DirectionOfEnergy::Low,
                NREV,
                bad,
            )
            .unwrap_err();
            assert_eq!(err, LambertError::NonPositiveTimeOfFlight);
        }
    }

    #[test]
    fn battin_rejects_overflowing_magnitude() {
        // Finite but astronomically large inputs whose magnitudes/cross products
        // overflow to infinity must not collapse into a fabricated finite Ok.
        let r1 = [1e160, 0.0, 0.0];
        let r2 = [0.0, 1e160, 0.0];
        let v1 = [0.0, 5.0, 0.0];
        let err = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Short,
            DirectionOfEnergy::Low,
            NREV,
            DTSEC,
        )
        .unwrap_err();
        assert_eq!(err, LambertError::NonFiniteValue);
    }

    #[test]
    fn battin_rejects_collinear_endpoints() {
        // r2 parallel to r1 (zero-degree transfer): transfer plane undefined.
        let r1 = [2.5 * RE, 0.0, 0.0];
        let r2 = [5.0 * RE, 0.0, 0.0];
        let v1 = [0.0, 4.999792554221911, 0.0];
        let err = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Short,
            DirectionOfEnergy::Low,
            NREV,
            DTSEC,
        )
        .unwrap_err();
        assert_eq!(err, LambertError::DegenerateGeometry);
    }

    #[test]
    fn battin_rejects_bad_180_transfer() {
        // 180-degree transfer (r2 anti-parallel to r1) with v1 collinear to r1,
        // so the plane cannot be fixed from v1 either.
        let r1 = [2.5 * RE, 0.0, 0.0];
        let r2 = [-5.0 * RE, 0.0, 0.0];
        let v1 = [3.0, 0.0, 0.0];
        let err = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Short,
            DirectionOfEnergy::Low,
            NREV,
            DTSEC,
        )
        .unwrap_err();
        assert_eq!(err, LambertError::DegenerateGeometry);
    }

    #[test]
    fn battin_rejects_nonfinite_input() {
        let (_, r2, v1) = inputs();
        let r1 = [f64::NAN, 0.0, 0.0];
        let err = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Short,
            DirectionOfEnergy::Low,
            NREV,
            DTSEC,
        )
        .unwrap_err();
        assert_eq!(err, LambertError::NonFiniteValue);
    }

    #[test]
    fn battin_high_energy_reports_nonconvergence() {
        // A very short time of flight on the high-energy multi-rev branch does
        // not converge within the iteration cap; surface it instead of
        // returning the last iterate.
        let (r1, r2, v1) = inputs();
        let err = battin(
            &r1,
            &r2,
            &v1,
            DirectionOfMotion::Short,
            DirectionOfEnergy::High,
            5,
            1.0,
        )
        .unwrap_err();
        assert_eq!(err, LambertError::NoConvergence);
    }
}
