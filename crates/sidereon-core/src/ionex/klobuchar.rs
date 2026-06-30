//! GPS broadcast Klobuchar single-frequency ionospheric delay.
//!
//! Implements the eight-coefficient half-cosine ionospheric group-delay model
//! defined in IS-GPS-200 (the L1 broadcast correction). The model takes the
//! receiver geodetic latitude/longitude, the satellite azimuth/elevation, the
//! GPS second-of-day, and the eight broadcast alpha/beta coefficients, and
//! returns the L1 ionospheric group delay in meters.
//!
//! The delay returned is a group delay and is positive: it increases the
//! measured pseudorange (the carrier-phase advance is the negation of this
//! value). The model is dispersive, so the delay on a carrier other than L1 is
//! the L1 delay scaled by `(f_l1 / f)^2`.
//!
//! The published algorithm works in semicircles (one semicircle = 180 degrees).
//! Internally the latitude, longitude, and elevation are reduced to semicircles
//! (`deg / 180`), while the azimuth is reduced to radians because it enters
//! `sin`/`cos` directly. The diurnal term is the truncated Taylor expansion
//! `1 - x^2/2 + x^4/24`, used (not the library cosine) when the phase magnitude
//! is below the cutoff; otherwise the night-time floor term applies. Integer
//! powers are written as explicit repeated multiplies and there is no fused
//! multiply-add: every product and sum is a plain operator, so the operation
//! tree is identical to the reference recipe and the result is bit-stable.

use core::f64::consts::PI;

/// Speed of light in vacuum (m/s), the IS-GPS-200 defined value.
use crate::constants::C_M_S as C;

/// All intermediate quantities of one Klobuchar evaluation.
///
/// Carrying every intermediate (not just the final delay) lets the parity test
/// localise any divergence to a single algorithm step rather than only seeing
/// the end result move.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct KlobucharComponents {
    /// Earth-central angle between receiver and ionospheric pierce point (semicircles).
    pub psi: f64,
    /// Subionospheric (pierce-point) geodetic latitude, clamped (semicircles).
    pub phi_i: f64,
    /// Subionospheric (pierce-point) geodetic longitude (semicircles).
    pub lambda_i: f64,
    /// Geomagnetic latitude of the pierce point (semicircles).
    pub phi_m: f64,
    /// Local time at the pierce point, wrapped to `[0, 86400)` (seconds).
    pub t: f64,
    /// Obliquity (slant) factor mapping vertical to slant delay (dimensionless).
    pub f: f64,
    /// Cosine-term amplitude, floored at zero (seconds).
    pub amp: f64,
    /// Cosine-term period, floored at 72000 (seconds).
    pub per: f64,
    /// Diurnal phase (radians).
    pub x: f64,
    /// Slant ionospheric time delay on L1 (seconds).
    pub t_iono: f64,
    /// Slant ionospheric group delay on L1 (meters).
    pub delay_l1_m: f64,
}

/// Compute the Klobuchar L1 ionospheric group delay and every intermediate.
///
/// Inputs are in the model's published boundary units: receiver latitude and
/// longitude in degrees, satellite azimuth and elevation in degrees, the GPS
/// second-of-day in `[0, 86400)`, and the eight broadcast coefficients (`alpha`
/// in the amplitude polynomial, `beta` in the period polynomial). The returned
/// `delay_l1_m` is the L1 group delay in positive meters.
#[allow(clippy::manual_clamp)]
pub(crate) fn klobuchar_l1_components(
    phi_u_deg: f64,
    lambda_u_deg: f64,
    azimuth_deg: f64,
    elevation_deg: f64,
    t_gps_s: f64,
    alpha: [f64; 4],
    beta: [f64; 4],
) -> KlobucharComponents {
    let [a0, a1, a2, a3] = alpha;
    let [b0, b1, b2, b3] = beta;

    // Reduce the public-degree inputs to the units each quantity uses. These
    // literal operations are pinned by the SPP 0-ULP trace oracle.
    let phi_u = phi_u_deg / 180.0; // semicircles
    let lambda_u = lambda_u_deg / 180.0; // semicircles
    let a = azimuth_deg * PI / 180.0; // radians (enters sin/cos directly)
    let e = elevation_deg / 180.0; // semicircles

    // 1. Earth-central angle (semicircles).
    let psi = 0.0137 / (e + 0.11) - 0.022;

    // 2. Subionospheric latitude (semicircles), clamped to +-0.416.
    let mut phi_i = phi_u + psi * a.cos();
    if phi_i > 0.416 {
        phi_i = 0.416;
    }
    if phi_i < -0.416 {
        phi_i = -0.416;
    }

    // 3. Subionospheric longitude (semicircles). cos takes phi_i in radians.
    let lambda_i = lambda_u + psi * a.sin() / (phi_i * PI).cos();

    // 4. Geomagnetic latitude (semicircles).
    let phi_m = phi_i + 0.064 * ((lambda_i - 1.617) * PI).cos();

    // 5. Local time at the pierce point (seconds), wrapped to [0, 86400).
    let mut t = 43200.0 * lambda_i + t_gps_s;
    if t >= 86400.0 {
        t -= 86400.0;
    }
    if t < 0.0 {
        t += 86400.0;
    }

    // 6. Obliquity (slant) factor. Integer cube as explicit multiply.
    let d = 0.53 - e;
    let cube = d * d * d;
    let f = 1.0 + 16.0 * cube;

    // 7. Amplitude (seconds), Horner, floored at 0.
    let mut amp = a0 + phi_m * (a1 + phi_m * (a2 + phi_m * a3));
    if amp < 0.0 {
        amp = 0.0;
    }

    // 8. Period (seconds), Horner, floored at 72000.
    let mut per = b0 + phi_m * (b1 + phi_m * (b2 + phi_m * b3));
    if per < 72000.0 {
        per = 72000.0;
    }

    // 9. Phase (radians).
    let x = 2.0 * PI * (t - 50400.0) / per;

    // 10. Slant time delay (seconds). Truncated cosine series while |x| < 1.57,
    //     else the night-time floor term only.
    let abs_x = if x < 0.0 { -x } else { x };
    let t_iono = if abs_x < 1.57 {
        let x2 = x * x;
        let x4 = x2 * x2;
        f * (5.0e-9 + amp * (1.0 - x2 / 2.0 + x4 / 24.0))
    } else {
        f * 5.0e-9
    };

    // 11. Convert L1 group delay to meters.
    let delay_l1_m = C * t_iono;

    KlobucharComponents {
        psi,
        phi_i,
        lambda_i,
        phi_m,
        t,
        f,
        amp,
        per,
        x,
        t_iono,
        delay_l1_m,
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
