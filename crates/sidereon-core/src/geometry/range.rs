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
    let c = libm::cos(theta);
    let s = libm::sin(theta);
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

/// First-order Sagnac term of the range rate, m/s: the time derivative of the
/// [`sagnac_range_first_order`] term, `omega / c · (vs_x·rr_y + rs_x·vr_y − vs_y·rr_x −
/// rs_y·vr_x)`, with the satellite position `sat` and velocity `sat_vel` in the
/// transmission-epoch frame, unrotated, and the receiver position `recv` and velocity
/// `recv_vel`, so a range-rate row predicts the rate of the range the code row predicts.
///
/// RTKLIB `resdop` adds `OMGE/CLIGHT·(vs_y·rr_x + rs_y·vr_x − vs_x·rr_y − rs_x·vr_y)`,
/// the same magnitude with the opposite sign, which is not the derivative of the
/// `geodist` term its code rows use; this follows `geodist`.
#[inline]
pub(crate) fn sagnac_range_rate_first_order(
    sat: [f64; 3],
    sat_vel: [f64; 3],
    recv: [f64; 3],
    recv_vel: [f64; 3],
    omega_rad_s: f64,
    c_m_s: f64,
) -> f64 {
    omega_rad_s / c_m_s
        * (sat_vel[0] * recv[1] + sat[0] * recv_vel[1]
            - sat_vel[1] * recv[0]
            - sat[1] * recv_vel[0])
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
        let c = libm::cos(theta);
        let s = libm::sin(theta);
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

    /// The Sagnac rate term against a value worked by hand, and against the numerical
    /// derivative of the Sagnac range term it differentiates.
    #[test]
    fn sagnac_range_rate_first_order_matches_a_hand_computed_value() {
        let sat = [15_600_000.0, -20_400_000.0, 9_800_000.0];
        let sat_vel = [1_200.0, 900.0, -2_800.0];
        let recv = [4_075_580.0, 931_854.0, 4_801_568.0];
        let recv_vel = [10.0, -20.0, 5.0];
        // vs_x rr_y + rs_x vr_y - vs_y rr_x - rs_y vr_x
        //   = 1200 * 931854 + 15600000 * (-20) - 900 * 4075580 - (-20400000) * 10
        //   = 1118224800 - 312000000 - 3668022000 + 204000000 = -2657797200 m^2/s,
        // times 7.2921151467e-5 / 299792458 = -6.46479e-4 m/s.
        let got = sagnac_range_rate_first_order(sat, sat_vel, recv, recv_vel, OMEGA, C);
        let hand = -2_657_797_200.0 * (OMEGA / C);
        assert!((got - hand).abs() <= 1.0e-18, "{got} vs {hand}");
        assert!((got + 6.464_79e-4).abs() < 1.0e-9, "{got}");

        let term = |t: f64| {
            let s = [0, 1, 2].map(|i| sat[i] + sat_vel[i] * t);
            let r = [0, 1, 2].map(|i| recv[i] + recv_vel[i] * t);
            OMEGA * (s[0] * r[1] - s[1] * r[0]) / C
        };
        let numeric = term(0.5) - term(-0.5);
        assert!((got - numeric).abs() < 1.0e-12, "{got} vs {numeric}");
    }
}
