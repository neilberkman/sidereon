//! Satellite-relative RSW, RTN, RIC, LVLH frames and Clohessy-Wiltshire motion.
//!
//! The RSW, RTN, RIC, and LVLH names all use the same right-handed chief
//! triad in this module. The first axis is radial and points outward from the
//! central body. The third axis is the positive orbit normal, along `r x v`.
//! The second axis is `W x R`, the transverse in-plane direction. For an
//! eccentric orbit this transverse axis is not exactly the velocity direction.
//!
//! LVLH is pinned to the same axes: `x` radial outward, `y` transverse
//! in-plane, and `z` positive orbit normal. This is the frame used by the
//! Clohessy-Wiltshire state-transition matrix below. The nadir-pointing LVLH
//! convention with `z = -R`, `x = S`, and `y = -W` is not used here.
//!
//! Relative states are carried in [`CartesianState`]. For relative-returning
//! APIs, `position_km` is the deputy position relative to the chief expressed
//! in the rotating chief frame, and `velocity_km_s` is the relative velocity
//! seen by that rotating frame. The epoch is copied as documented on each
//! function.
//!
//! The rotating-frame rate used by the state transforms is the osculating
//! two-body RSW rate `[0, 0, |r x v| / |r|^2]`. This is exact for unperturbed
//! Keplerian motion and is consistent with the inverse transform and the
//! Clohessy-Wiltshire model. Perturbed chiefs can have additional basis-vector
//! rate terms that are outside this API.
//!
//! The Clohessy-Wiltshire equations linearize relative motion about a circular
//! chief orbit. They are appropriate for short arcs and separations small
//! relative to the chief orbit radius. Error grows for eccentric chiefs,
//! hundreds-of-kilometers separations in LEO, and long arcs where J2, drag,
//! solar radiation pressure, or third-body effects dominate. Eccentric-chief
//! models such as Tschauner-Hempel or Yamanaka-Ankersen are future work.
//!
//! References: W. H. Clohessy and R. S. Wiltshire, "Terminal Guidance System
//! for Satellite Rendezvous," Journal of the Aerospace Sciences, 27(9),
//! pp. 653-658, 1960, DOI 10.2514/8.8704. G. W. Hill, "Researches in the Lunar
//! Theory," American Journal of Mathematics, 1(1), pp. 5-26, 1878,
//! DOI 10.2307/2369430. D. A. Vallado, "Fundamentals of Astrodynamics and
//! Applications," 4th ed., 2013, Section 6.7 "Clohessy-Wiltshire".

use crate::astro::constants::MU_EARTH;
use crate::astro::covariance::{rtn_to_eci_rotation, Mat6, RtnFrameError};
use crate::astro::elements::{rv2coe, ElementsError};
use crate::astro::math::mat3::{self, Mat3};
use crate::astro::math::vec3;
use crate::astro::state::CartesianState;

/// Return the chief RSW-to-inertial rotation.
///
/// The returned matrix has the chief radial, transverse, and orbit-normal unit
/// vectors as columns. Its transpose maps inertial vector components into the
/// chief frame.
pub fn rsw_to_inertial_rotation(chief: &CartesianState) -> Result<Mat3, RtnFrameError> {
    rtn_to_eci_rotation(chief.position_array(), chief.velocity_array())
}

/// Return the chief RTN-to-inertial rotation.
///
/// This is an alias of [`rsw_to_inertial_rotation`].
pub fn rtn_to_inertial_rotation(chief: &CartesianState) -> Result<Mat3, RtnFrameError> {
    rsw_to_inertial_rotation(chief)
}

/// Return the chief RIC-to-inertial rotation.
///
/// This is an alias of [`rsw_to_inertial_rotation`].
pub fn ric_to_inertial_rotation(chief: &CartesianState) -> Result<Mat3, RtnFrameError> {
    rsw_to_inertial_rotation(chief)
}

/// Return the chief LVLH-to-inertial rotation.
///
/// This module defines LVLH with the same axes as RSW: radial outward,
/// transverse in-plane, and positive orbit normal.
pub fn lvlh_to_inertial_rotation(chief: &CartesianState) -> Result<Mat3, RtnFrameError> {
    rsw_to_inertial_rotation(chief)
}

/// Express an absolute deputy state relative to a chief in the chief frame.
///
/// The chief and deputy states must be synchronized by the caller. No epoch
/// tolerance is applied and no propagation is performed. The returned
/// [`CartesianState`] carries the deputy epoch. Its position and velocity are
/// relative quantities in the rotating RSW, RTN, RIC, LVLH frame.
pub fn relative_state(
    chief: &CartesianState,
    deputy: &CartesianState,
) -> Result<CartesianState, RtnFrameError> {
    let r = rsw_to_inertial_rotation(chief)?;
    let rt = mat3::inline_tr(&r);
    validate_vec3("deputy.position", deputy.position_array())?;
    validate_vec3("deputy.velocity", deputy.velocity_array())?;

    let dr = vec3::sub3(deputy.position_array(), chief.position_array());
    let dv = vec3::sub3(deputy.velocity_array(), chief.velocity_array());
    let rho = mat3::mul_vec3(&rt, dr);
    validate_vec3("relative.position", rho)?;

    let omega = chief_frame_omega(chief);
    let inertial_rate_in_frame = mat3::mul_vec3(&rt, dv);
    let rho_dot = vec3::sub3(inertial_rate_in_frame, vec3::cross3(omega, rho));
    validate_vec3("relative.velocity", rho_dot)?;

    Ok(CartesianState::new(deputy.epoch_tdb_seconds, rho, rho_dot))
}

/// Convert a chief-relative state in the chief frame back to absolute ECI.
///
/// This is the inverse of [`relative_state`] for the same chief. The returned
/// [`CartesianState`] carries the chief epoch.
pub fn absolute_from_relative(
    chief: &CartesianState,
    rel: &CartesianState,
) -> Result<CartesianState, RtnFrameError> {
    let r = rsw_to_inertial_rotation(chief)?;
    let rho = rel.position_array();
    let rho_dot = rel.velocity_array();
    validate_vec3("rel.position", rho)?;
    validate_vec3("rel.velocity", rho_dot)?;

    let omega = chief_frame_omega(chief);
    let dr = mat3::mul_vec3(&r, rho);
    let velocity_in_frame = vec3::add3(rho_dot, vec3::cross3(omega, rho));
    let dv = mat3::mul_vec3(&r, velocity_in_frame);

    let position = vec3::add3(chief.position_array(), dr);
    let velocity = vec3::add3(chief.velocity_array(), dv);
    validate_vec3("deputy.position", position)?;
    validate_vec3("deputy.velocity", velocity)?;

    Ok(CartesianState::new(
        chief.epoch_tdb_seconds,
        position,
        velocity,
    ))
}

/// Build the Clohessy-Wiltshire state-transition matrix.
///
/// The state order is `[x, y, z, xdot, ydot, zdot]` in the LVLH convention
/// documented at module level. `n` is the chief mean motion in rad/s and `dt`
/// is seconds. `n` must be finite and positive. `dt` must be finite.
pub fn cw_stm(n: f64, dt: f64) -> Result<Mat6, RtnFrameError> {
    if !n.is_finite() || n <= 0.0 {
        return Err(invalid_input("n", "must be finite and positive"));
    }
    if !dt.is_finite() {
        return Err(invalid_input("dt", "must be finite"));
    }

    let nt = n * dt;
    let s = libm::sin(nt);
    let c = libm::cos(nt);
    let one_minus_c = 1.0 - c;

    Ok([
        [4.0 - 3.0 * c, 0.0, 0.0, s / n, 2.0 * one_minus_c / n, 0.0],
        [
            6.0 * (s - nt),
            1.0,
            0.0,
            -2.0 * one_minus_c / n,
            (4.0 * s - 3.0 * nt) / n,
            0.0,
        ],
        [0.0, 0.0, c, 0.0, 0.0, s / n],
        [3.0 * n * s, 0.0, 0.0, c, 2.0 * s, 0.0],
        [
            -6.0 * n * one_minus_c,
            0.0,
            0.0,
            -2.0 * s,
            4.0 * c - 3.0,
            0.0,
        ],
        [0.0, 0.0, -n * s, 0.0, 0.0, c],
    ])
}

/// Propagate a relative state with the Clohessy-Wiltshire equations.
///
/// The input and output states are relative states carried in [`CartesianState`]
/// as documented at module level. The output epoch is
/// `rel.epoch_tdb_seconds + dt`.
pub fn cw_propagate(
    rel: &CartesianState,
    n: f64,
    dt: f64,
) -> Result<CartesianState, RtnFrameError> {
    let stm = cw_stm(n, dt)?;
    validate_vec3("rel.position", rel.position_array())?;
    validate_vec3("rel.velocity", rel.velocity_array())?;

    let epoch = rel.epoch_tdb_seconds + dt;
    if !epoch.is_finite() {
        return Err(invalid_input(
            "epoch_tdb_seconds",
            "rel epoch plus dt must be finite",
        ));
    }

    let propagated = mat6_vec_mul(
        &stm,
        [
            rel.position_km.x,
            rel.position_km.y,
            rel.position_km.z,
            rel.velocity_km_s.x,
            rel.velocity_km_s.y,
            rel.velocity_km_s.z,
        ],
    );

    Ok(CartesianState::new(
        epoch,
        [propagated[0], propagated[1], propagated[2]],
        [propagated[3], propagated[4], propagated[5]],
    ))
}

/// Mean motion for a circular Earth orbit of `radius_km`.
pub fn mean_motion_circular(radius_km: f64) -> Result<f64, RtnFrameError> {
    if !radius_km.is_finite() || radius_km <= 0.0 {
        return Err(invalid_input("radius_km", "must be finite and positive"));
    }

    let n = (MU_EARTH / radius_km.powi(3)).sqrt();
    if !n.is_finite() {
        return Err(invalid_input("mean_motion", "not finite"));
    }
    Ok(n)
}

/// Mean motion from the semi-major axis of an osculating Earth orbit.
///
/// The semi-major axis is obtained from [`rv2coe`]. This is appropriate for the
/// near-circular chief orbits assumed by Clohessy-Wiltshire and degrades as the
/// chief eccentricity grows.
pub fn mean_motion_from_state(chief: &CartesianState) -> Result<f64, RtnFrameError> {
    let elements =
        rv2coe(chief.position_array(), chief.velocity_array(), MU_EARTH).map_err(|error| {
            match error {
                ElementsError::ZeroPosition | ElementsError::DegenerateOrbit => {
                    invalid_input("state", "degenerate orbit geometry")
                }
                ElementsError::NonFinite { field: "r" } => {
                    invalid_input("position", "components must be finite")
                }
                ElementsError::NonFinite { field: "v" } => {
                    invalid_input("velocity", "components must be finite")
                }
                ElementsError::NonFinite { field } => invalid_input(field, "not finite"),
                ElementsError::NonPositiveMu => invalid_input("mu", "must be positive"),
                ElementsError::NonPositiveSemiLatus => {
                    invalid_input("semi_latus_rectum", "must be positive")
                }
            }
        })?;

    if !elements.a.is_finite() || elements.a <= 0.0 {
        return Err(invalid_input(
            "semi_major_axis",
            "orbit is not elliptical (a must be finite and positive)",
        ));
    }

    let n = (MU_EARTH / elements.a.powi(3)).sqrt();
    if !n.is_finite() {
        return Err(invalid_input("mean_motion", "not finite"));
    }
    Ok(n)
}

fn invalid_input(field: &'static str, reason: &'static str) -> RtnFrameError {
    RtnFrameError::InvalidInput { field, reason }
}

fn validate_vec3(field: &'static str, values: [f64; 3]) -> Result<(), RtnFrameError> {
    if values.iter().all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(invalid_input(field, "components must be finite"))
    }
}

fn chief_frame_omega(chief: &CartesianState) -> [f64; 3] {
    let r = chief.position_array();
    let v = chief.velocity_array();
    let h = vec3::cross3(r, v);
    let r_norm = vec3::norm3(r);
    [0.0, 0.0, vec3::norm3(h) / (r_norm * r_norm)]
}

fn mat6_vec_mul(m: &Mat6, v: [f64; 6]) -> [f64; 6] {
    let mut out = [0.0_f64; 6];
    for (idx, row) in m.iter().enumerate() {
        out[idx] = row.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
    }
    out
}

#[cfg(test)]
mod tests {
    //! Clohessy-Wiltshire analytic tests use these references:
    //! W. H. Clohessy and R. S. Wiltshire, "Terminal Guidance System for
    //! Satellite Rendezvous," Journal of the Aerospace Sciences, 27(9),
    //! pp. 653-658, 1960, DOI 10.2514/8.8704. D. A. Vallado,
    //! "Fundamentals of Astrodynamics and Applications," 4th ed., 2013,
    //! Section 6.7 "Clohessy-Wiltshire".

    use super::*;

    const N: f64 = 0.001_078_007_612_872_506;

    #[test]
    fn cw_closed_orbit_is_periodic_after_one_chief_period() {
        let x0 = 1.25;
        let initial = CartesianState::new(10.0, [x0, 0.0, 0.0], [0.0, -2.0 * N * x0, 0.0]);
        let period = 2.0 * std::f64::consts::PI / N;

        let propagated = cw_propagate(&initial, N, period).expect("valid CW state");

        assert_vec_close(
            propagated.position_array(),
            initial.position_array(),
            1.0e-6,
        );
        assert_vec_close(
            propagated.velocity_array(),
            initial.velocity_array(),
            1.0e-9,
        );
        assert!(
            (propagated.epoch_tdb_seconds - (initial.epoch_tdb_seconds + period)).abs() < 1.0e-12
        );
    }

    #[test]
    fn cw_out_of_plane_motion_matches_harmonic_solution() {
        let z0 = 2.5;
        let dt = 1234.5;
        let initial = CartesianState::new(0.0, [0.0, 0.0, z0], [0.0, 0.0, 0.0]);
        let propagated = cw_propagate(&initial, N, dt).expect("valid CW state");
        let nt = N * dt;

        assert!((propagated.position_km.z - z0 * libm::cos(nt)).abs() < 1.0e-6);
        assert!((propagated.velocity_km_s.z - (-N * z0 * libm::sin(nt))).abs() < 1.0e-9);
    }

    #[test]
    fn cw_along_track_offset_and_radial_drift_match_closed_form() {
        let dt = 2222.0;
        let y0 = 3.0;
        let along_track = CartesianState::new(0.0, [0.0, y0, 0.0], [0.0, 0.0, 0.0]);
        let propagated = cw_propagate(&along_track, N, dt).expect("valid CW state");
        assert_vec_close(propagated.position_array(), [0.0, y0, 0.0], 1.0e-12);
        assert_vec_close(propagated.velocity_array(), [0.0, 0.0, 0.0], 1.0e-12);

        let x0 = 1.1;
        let radial = CartesianState::new(0.0, [x0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let propagated = cw_propagate(&radial, N, dt).expect("valid CW state");
        let nt = N * dt;
        assert!((propagated.position_km.y - 6.0 * (libm::sin(nt) - nt) * x0).abs() < 1.0e-6);
    }

    #[test]
    fn cw_stm_zero_dt_is_identity() {
        let stm = cw_stm(N, 0.0).expect("valid STM");
        assert_mat6_close(&stm, &identity6(), 1.0e-12);
    }

    #[test]
    fn cw_stm_inverse_and_semigroup_are_consistent() {
        let a = 400.0;
        let b = 850.0;
        let stm_a = cw_stm(N, a).expect("valid STM");
        let stm_b = cw_stm(N, b).expect("valid STM");
        let stm_neg_a = cw_stm(N, -a).expect("valid STM");
        let stm_a_b = cw_stm(N, a + b).expect("valid STM");

        assert_mat6_close(&mat6_mul(&stm_a, &stm_neg_a), &identity6(), 1.0e-9);
        assert_mat6_close(&mat6_mul(&stm_b, &stm_a), &stm_a_b, 1.0e-9);
    }

    #[test]
    fn cw_rejects_invalid_inputs() {
        assert!(matches!(
            cw_stm(0.0, 1.0),
            Err(RtnFrameError::InvalidInput { field: "n", .. })
        ));
        assert!(matches!(
            cw_stm(f64::NAN, 1.0),
            Err(RtnFrameError::InvalidInput { field: "n", .. })
        ));
        assert!(matches!(
            cw_stm(N, f64::INFINITY),
            Err(RtnFrameError::InvalidInput { field: "dt", .. })
        ));

        let bad_position = CartesianState::new(0.0, [f64::NAN, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(matches!(
            cw_propagate(&bad_position, N, 1.0),
            Err(RtnFrameError::InvalidInput {
                field: "rel.position",
                ..
            })
        ));

        let bad_epoch = CartesianState::new(f64::MAX, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        assert!(matches!(
            cw_propagate(&bad_epoch, N, f64::MAX),
            Err(RtnFrameError::InvalidInput {
                field: "epoch_tdb_seconds",
                ..
            })
        ));
    }

    #[test]
    fn mean_motion_rejects_invalid_inputs() {
        assert!(matches!(
            mean_motion_circular(f64::NAN),
            Err(RtnFrameError::InvalidInput {
                field: "radius_km",
                ..
            })
        ));
        assert!(matches!(
            mean_motion_circular(0.0),
            Err(RtnFrameError::InvalidInput {
                field: "radius_km",
                ..
            })
        ));
        assert!(matches!(
            mean_motion_circular(f64::MIN_POSITIVE),
            Err(RtnFrameError::InvalidInput {
                field: "mean_motion",
                ..
            })
        ));

        let degenerate = CartesianState::new(0.0, [0.0, 0.0, 0.0], [0.0, 7.5, 0.0]);
        assert!(matches!(
            mean_motion_from_state(&degenerate),
            Err(RtnFrameError::InvalidInput { field: "state", .. })
        ));

        let hyperbolic = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, 11.0, 0.0]);
        assert!(matches!(
            mean_motion_from_state(&hyperbolic),
            Err(RtnFrameError::InvalidInput {
                field: "semi_major_axis",
                ..
            })
        ));
    }

    fn assert_vec_close(actual: [f64; 3], expected: [f64; 3], tolerance: f64) {
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (*actual - *expected).abs() <= tolerance,
                "component {idx}: actual {}, expected {}",
                actual,
                expected
            );
        }
    }

    fn assert_mat6_close(actual: &Mat6, expected: &Mat6, tolerance: f64) {
        for (i, (actual_row, expected_row)) in actual.iter().zip(expected.iter()).enumerate() {
            for (j, (actual, expected)) in actual_row.iter().zip(expected_row.iter()).enumerate() {
                assert!(
                    (*actual - *expected).abs() <= tolerance,
                    "entry ({i},{j}): actual {}, expected {}",
                    actual,
                    expected
                );
            }
        }
    }

    fn identity6() -> Mat6 {
        let mut matrix = [[0.0_f64; 6]; 6];
        for (idx, row) in matrix.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        matrix
    }

    fn mat6_mul(a: &Mat6, b: &Mat6) -> Mat6 {
        let mut out = [[0.0_f64; 6]; 6];
        let b_t = transpose6(b);
        for (i, row) in a.iter().enumerate() {
            for (j, col) in b_t.iter().enumerate() {
                out[i][j] = row.iter().zip(col.iter()).map(|(lhs, rhs)| lhs * rhs).sum();
            }
        }
        out
    }

    fn transpose6(m: &Mat6) -> Mat6 {
        let mut out = [[0.0_f64; 6]; 6];
        for (i, row) in m.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                out[j][i] = *value;
            }
        }
        out
    }
}
