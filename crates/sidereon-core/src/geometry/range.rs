//! Shared GNSS range geometry: Sagnac rotation and geometric-range forms.
//!
//! Centralizes the Earth-rotation (Sagnac) correction and geometric-range
//! computations that the observable predictor, the SPP residual model, and the
//! RTK baseline filter previously each implemented inline. Each named function
//! pins the specific floating-point operation order required for 0-ULP parity,
//! so callers reuse the order rather than copy-pasting it.

use crate::astro::math::vec3::{norm3, sub3};

/// Closed-form Sagnac (Earth-rotation) rotation of an ECEF satellite position
/// over the signal flight time `tau_s`, applied as a `+theta` rotation about
/// the Z axis (`theta = omega_rad_s * tau_s`).
///
/// `cos` and `sin` are evaluated separately (not via `sin_cos`) and the matrix
/// is applied row-wise, matching the order pinned by the SPP trace oracle and
/// the observable-prediction goldens.
#[inline]
pub(crate) fn sagnac_rotate_exact(pos: [f64; 3], tau_s: f64, omega_rad_s: f64) -> [f64; 3] {
    let theta = omega_rad_s * tau_s;
    let c = theta.cos();
    let s = theta.sin();
    [c * pos[0] + s * pos[1], -s * pos[0] + c * pos[1], pos[2]]
}

/// Geometric range with the first-order Sagnac scalar correction used on
/// transmit-time satellite positions (RTK double differences): the Euclidean
/// range plus `omega * (sat_x*recv_y - sat_y*recv_x) / c`. Unlike
/// [`sagnac_rotate_exact`] this corrects the scalar range only, not the vector.
#[inline]
pub(crate) fn sagnac_range_first_order(
    sat: [f64; 3],
    recv: [f64; 3],
    omega_rad_s: f64,
    c_m_s: f64,
) -> f64 {
    norm3(sub3(sat, recv)) + omega_rad_s * (sat[0] * recv[1] - sat[1] * recv[0]) / c_m_s
}

#[cfg(test)]
mod tests {
    use super::*;

    // Representative ECEF satellite/receiver geometry and a 0.07 s flight time.
    const SAT: [f64; 3] = [15_600_000.0, -20_400_000.0, 9_800_000.0];
    const RECV: [f64; 3] = [4_027_894.0, 307_046.0, 4_919_474.0];
    const TAU_S: f64 = 0.072_345;
    const OMEGA: f64 = 7.292_115_146_7e-5;
    const C: f64 = 299_792_458.0;

    #[test]
    fn sagnac_rotate_exact_matches_explicit_recipe_bits() {
        let theta = OMEGA * TAU_S;
        let c = theta.cos();
        let s = theta.sin();
        let want = [c * SAT[0] + s * SAT[1], -s * SAT[0] + c * SAT[1], SAT[2]];
        let got = sagnac_rotate_exact(SAT, TAU_S, OMEGA);
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits());
        }
    }

    #[test]
    fn sagnac_range_first_order_matches_explicit_recipe_bits() {
        let dx = SAT[0] - RECV[0];
        let dy = SAT[1] - RECV[1];
        let dz = SAT[2] - RECV[2];
        let r = (dx * dx + dy * dy + dz * dz).sqrt();
        let want = r + OMEGA * (SAT[0] * RECV[1] - SAT[1] * RECV[0]) / C;
        let got = sagnac_range_first_order(SAT, RECV, OMEGA, C);
        assert_eq!(got.to_bits(), want.to_bits());
    }
}
