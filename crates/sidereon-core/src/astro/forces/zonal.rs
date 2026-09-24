//! Axisymmetric Earth zonal harmonic gravity through degree 6.
//!
//! The acceleration is formed from the zonal part of the geopotential
//! `U_n = -mu * J_n * Re^n * P_n(z/r) / r^(n+1)`, using the closed-form
//! Legendre polynomials listed by Montenbruck & Gill, "Satellite Orbits",
//! section 3.2, and Vallado, "Fundamentals of Astrodynamics and Applications",
//! section 8.7. The Earth pole is the inertial +Z axis, matching the existing
//! J2 force model.

use crate::astro::constants::{
    J2_EARTH, J3_EARTH, J4_EARTH, J5_EARTH, J6_EARTH, MU_EARTH, RE_EARTH,
};
use crate::astro::error::PropagationError;
use crate::astro::forces::geopotential::TideSystem;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

/// Active zonal harmonic degrees for [`ZonalGravity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZonalDegrees {
    /// Include the J2 oblateness term.
    pub j2: bool,
    /// Include the J3 odd zonal term.
    pub j3: bool,
    /// Include the J4 zonal term.
    pub j4: bool,
    /// Include the J5 zonal term.
    pub j5: bool,
    /// Include the J6 zonal term.
    pub j6: bool,
}

impl ZonalDegrees {
    /// No zonal degrees.
    pub const NONE: Self = Self {
        j2: false,
        j3: false,
        j4: false,
        j5: false,
        j6: false,
    };

    /// J2 only.
    pub const J2_ONLY: Self = Self {
        j2: true,
        j3: false,
        j4: false,
        j5: false,
        j6: false,
    };

    /// J2 through J6.
    pub const J2_THROUGH_J6: Self = Self {
        j2: true,
        j3: true,
        j4: true,
        j5: true,
        j6: true,
    };

    /// J3 through J6, useful when layering on the legacy [`crate::astro::forces::J2Gravity`].
    pub const J3_THROUGH_J6: Self = Self {
        j2: false,
        j3: true,
        j4: true,
        j5: true,
        j6: true,
    };

    /// Build a consecutive active set from J2 through `max_degree`.
    pub fn through(max_degree: u8) -> Result<Self, PropagationError> {
        if !(2..=6).contains(&max_degree) {
            return Err(PropagationError::InvalidInput(
                "zonal max_degree must be in 2..=6".to_string(),
            ));
        }
        Ok(Self {
            j2: max_degree >= 2,
            j3: max_degree >= 3,
            j4: max_degree >= 4,
            j5: max_degree >= 5,
            j6: max_degree >= 6,
        })
    }

    fn is_active(self, degree: u8) -> bool {
        match degree {
            2 => self.j2,
            3 => self.j3,
            4 => self.j4,
            5 => self.j5,
            6 => self.j6,
            _ => false,
        }
    }
}

/// Unnormalized zonal harmonic coefficients through degree 6.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZonalCoefficients {
    /// J2 coefficient, dimensionless.
    pub j2: f64,
    /// J3 coefficient, dimensionless.
    pub j3: f64,
    /// J4 coefficient, dimensionless.
    pub j4: f64,
    /// J5 coefficient, dimensionless.
    pub j5: f64,
    /// J6 coefficient, dimensionless.
    pub j6: f64,
    /// Tide system of `j2`. `j2 = -sqrt(5) C20`, so it carries whatever
    /// permanent tide `C20` holds; the default coefficients are EGM96's, which
    /// are tide-free.
    pub tide_system: TideSystem,
}

impl Default for ZonalCoefficients {
    fn default() -> Self {
        Self {
            j2: J2_EARTH,
            j3: J3_EARTH,
            j4: J4_EARTH,
            j5: J5_EARTH,
            j6: J6_EARTH,
            tide_system: TideSystem::TideFree,
        }
    }
}

impl ZonalCoefficients {
    fn get(self, degree: u8) -> f64 {
        match degree {
            2 => self.j2,
            3 => self.j3,
            4 => self.j4,
            5 => self.j5,
            6 => self.j6,
            _ => 0.0,
        }
    }
}

/// Closed-form zonal gravity perturbation through degree 6.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZonalGravity {
    /// Gravitational parameter, km^3/s^2.
    pub mu_km3_s2: f64,
    /// Reference equatorial radius, km.
    pub re_km: f64,
    /// Active zonal degrees.
    pub degrees: ZonalDegrees,
    /// Unnormalized zonal coefficients.
    pub coefficients: ZonalCoefficients,
}

impl Default for ZonalGravity {
    fn default() -> Self {
        Self {
            mu_km3_s2: MU_EARTH,
            re_km: RE_EARTH,
            degrees: ZonalDegrees::J2_THROUGH_J6,
            coefficients: ZonalCoefficients::default(),
        }
    }
}

impl ZonalGravity {
    /// Build zonal gravity with explicit parameters.
    pub fn new(
        mu_km3_s2: f64,
        re_km: f64,
        degrees: ZonalDegrees,
        coefficients: ZonalCoefficients,
    ) -> Self {
        Self {
            mu_km3_s2,
            re_km,
            degrees,
            coefficients,
        }
    }

    /// Earth J2 through J6 zonal gravity using the crate constants.
    pub fn earth_j2_through_j6() -> Self {
        Self::default()
    }

    /// Earth J3 through J6 zonal gravity using the crate constants.
    pub fn earth_j3_through_j6() -> Self {
        Self {
            degrees: ZonalDegrees::J3_THROUGH_J6,
            ..Self::default()
        }
    }
}

impl ForceModel for ZonalGravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let r_mag2 = state.position_km.norm_squared();
        if r_mag2 == 0.0 {
            return Err(PropagationError::NumericalFailure(
                "Zero position magnitude".to_string(),
            ));
        }
        let r_mag = r_mag2.sqrt();
        let s = state.position_km.z / r_mag;

        let mut accel = Vector3::zeros();
        for degree in 2_u8..=6 {
            if self.degrees.is_active(degree) {
                accel += zonal_degree_acceleration(
                    state.position_km,
                    r_mag,
                    s,
                    degree,
                    self.mu_km3_s2,
                    self.re_km,
                    self.coefficients.get(degree),
                );
            }
        }
        Ok(accel)
    }
}

fn zonal_degree_acceleration(
    position_km: Vector3<f64>,
    r_mag: f64,
    s: f64,
    degree: u8,
    mu_km3_s2: f64,
    re_km: f64,
    coefficient: f64,
) -> Vector3<f64> {
    let degree_i32 = i32::from(degree);
    let degree_f64 = f64::from(degree);
    let (p, dp) = legendre_and_derivative(degree, s);
    let common = mu_km3_s2 * coefficient * re_km.powi(degree_i32);
    let xy_bracket = (degree_f64 + 1.0) * p + s * dp;
    let z_bracket = (degree_f64 + 1.0) * s * p - (1.0 - s * s) * dp;
    let xy_scale = common / r_mag.powi(degree_i32 + 3);
    let z_scale = common / r_mag.powi(degree_i32 + 2);

    Vector3::new(
        xy_scale * position_km.x * xy_bracket,
        xy_scale * position_km.y * xy_bracket,
        z_scale * z_bracket,
    )
}

fn legendre_and_derivative(degree: u8, s: f64) -> (f64, f64) {
    let s2 = s * s;
    match degree {
        2 => ((3.0 * s2 - 1.0) / 2.0, 3.0 * s),
        3 => ((5.0 * s2 * s - 3.0 * s) / 2.0, (15.0 * s2 - 3.0) / 2.0),
        4 => (
            (35.0 * s2 * s2 - 30.0 * s2 + 3.0) / 8.0,
            (35.0 * s2 * s - 15.0 * s) / 2.0,
        ),
        5 => (
            (63.0 * s2 * s2 * s - 70.0 * s2 * s + 15.0 * s) / 8.0,
            (315.0 * s2 * s2 - 210.0 * s2 + 15.0) / 8.0,
        ),
        6 => (
            (231.0 * s2 * s2 * s2 - 315.0 * s2 * s2 + 105.0 * s2 - 5.0) / 16.0,
            (693.0 * s2 * s2 * s - 630.0 * s2 * s + 105.0 * s) / 8.0,
        ),
        _ => (0.0, 0.0),
    }
}

#[cfg(test)]
mod tests {
    //! Clean-room validation anchors:
    //!
    //! The expected accelerations below are fixed values evaluated from the
    //! closed-form zonal equations in Montenbruck & Gill, section 3.2, and
    //! Vallado, section 8.7, at the stated Cartesian point. They are not
    //! generated from the implementation under test. The J3..J6 constants are
    //! unnormalized EGM96 values from the published model coefficients.

    use super::*;

    fn one_degree(degree: u8) -> ZonalGravity {
        let mut degrees = ZonalDegrees::NONE;
        match degree {
            2 => degrees.j2 = true,
            3 => degrees.j3 = true,
            4 => degrees.j4 = true,
            5 => degrees.j5 = true,
            6 => degrees.j6 = true,
            _ => unreachable!("test degree is in 2..=6"),
        }
        ZonalGravity {
            degrees,
            ..ZonalGravity::default()
        }
    }

    fn test_state() -> CartesianState {
        CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [0.0, 0.0, 0.0])
    }

    fn assert_close_vector(actual: Vector3<f64>, expected: [f64; 3], tolerance: f64) {
        for axis in 0..3 {
            assert!(
                (actual[axis] - expected[axis]).abs() <= tolerance,
                "axis {axis}: actual={} expected={}",
                actual[axis],
                expected[axis]
            );
        }
    }

    #[test]
    fn coefficients_pin_published_egm96_values() {
        let coefficients = ZonalCoefficients::default();
        assert_eq!(
            coefficients.j3.to_bits(),
            (-2.532_656_485_332_235_5e-6_f64).to_bits()
        );
        assert_eq!(
            coefficients.j4.to_bits(),
            (-1.619_621_591_367e-6_f64).to_bits()
        );
        assert_eq!(
            coefficients.j5.to_bits(),
            (-2.272_960_828_686_982e-7_f64).to_bits()
        );
        assert_eq!(
            coefficients.j6.to_bits(),
            5.406_812_391_070_848e-7_f64.to_bits()
        );
    }

    #[test]
    fn each_degree_matches_closed_form_reference_value() {
        let expected = [
            [
                -7.863_322_262_469_1e-6,
                1.359_231_419_655_373e-6,
                -4.945_691_390_909_161e-6,
            ],
            [
                1.613_036_665_014_226_3e-8,
                -2.788_249_092_381_734e-9,
                -1.376_534_176_925_069_8e-8,
            ],
            [
                -7.779_765_177_631_29e-9,
                1.344_787_980_704_837e-9,
                -1.084_371_965_506_775_4e-8,
            ],
            [
                -1.736_874_053_306_558_4e-9,
                3.002_310_863_572_765e-10,
                6.722_489_204_891_498e-10,
            ],
            [
                -9.402_407_445_840_708e-10,
                1.625_273_287_066_751e-10,
                -3.939_169_694_358_007e-9,
            ],
        ];

        for (idx, degree) in (2_u8..=6).enumerate() {
            let actual = one_degree(degree)
                .acceleration(&test_state(), &PropagationContext::default())
                .expect("zonal acceleration");
            assert_close_vector(actual, expected[idx], 1.0e-20);
        }
    }

    #[test]
    fn j3_has_odd_zonal_north_south_signature() {
        let gravity = one_degree(3);
        let north = CartesianState::new(0.0, [7000.0, 0.0, 1300.0], [0.0, 0.0, 0.0]);
        let south = CartesianState::new(0.0, [7000.0, 0.0, -1300.0], [0.0, 0.0, 0.0]);
        let north_accel = gravity
            .acceleration(&north, &PropagationContext::default())
            .expect("north acceleration");
        let south_accel = gravity
            .acceleration(&south, &PropagationContext::default())
            .expect("south acceleration");

        assert!((north_accel.x + south_accel.x).abs() < 1.0e-22);
        assert_eq!(north_accel.y, 0.0);
        assert_eq!(south_accel.y, 0.0);
        assert!((north_accel.z - south_accel.z).abs() < 1.0e-22);
        assert!(north_accel.z < 0.0);
        assert!(south_accel.z < 0.0);
    }
}
