//! Sun and Moon third-body point-mass perturbations.
//!
//! The acceleration uses the standard direct-minus-indirect expression
//! `mu_b * ((r_b - r) / |r_b - r|^3 - r_b / |r_b|^3)`, as given by
//! Montenbruck & Gill, "Satellite Orbits", section 3.2, and Vallado,
//! "Fundamentals of Astrodynamics and Applications", section 8.6. Body
//! positions come from the crate's analytic Sun/Moon series at the state epoch,
//! in GCRS ([`sun_moon_gcrs_at_tdb_seconds`]), the frame of the propagation
//! state.

use crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds;
use crate::astro::constants::astro::{GM_MOON_KM3_S2, GM_SUN_KM3_S2};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::ForceModel;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::Vector3;

/// Enabled third bodies for [`ThirdBodyGravity`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThirdBodyBodies {
    /// Include the Sun perturbation.
    pub sun: bool,
    /// Include the Moon perturbation.
    pub moon: bool,
}

impl ThirdBodyBodies {
    /// No third bodies.
    pub const NONE: Self = Self {
        sun: false,
        moon: false,
    };

    /// Sun only.
    pub const SUN: Self = Self {
        sun: true,
        moon: false,
    };

    /// Moon only.
    pub const MOON: Self = Self {
        sun: false,
        moon: true,
    };

    /// Sun and Moon.
    pub const SUN_AND_MOON: Self = Self {
        sun: true,
        moon: true,
    };
}

/// Third-body gravity from analytic Sun and Moon positions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThirdBodyGravity {
    /// Enabled bodies.
    pub bodies: ThirdBodyBodies,
    /// Solar gravitational parameter, km^3/s^2.
    pub gm_sun_km3_s2: f64,
    /// Lunar gravitational parameter, km^3/s^2.
    pub gm_moon_km3_s2: f64,
}

impl Default for ThirdBodyGravity {
    fn default() -> Self {
        Self {
            bodies: ThirdBodyBodies::SUN_AND_MOON,
            gm_sun_km3_s2: GM_SUN_KM3_S2,
            gm_moon_km3_s2: GM_MOON_KM3_S2,
        }
    }
}

impl ThirdBodyGravity {
    /// Build a third-body model with explicit bodies and gravitational parameters.
    pub fn new(bodies: ThirdBodyBodies, gm_sun_km3_s2: f64, gm_moon_km3_s2: f64) -> Self {
        Self {
            bodies,
            gm_sun_km3_s2,
            gm_moon_km3_s2,
        }
    }

    /// Sun-only third-body gravity.
    pub fn sun() -> Self {
        Self {
            bodies: ThirdBodyBodies::SUN,
            ..Self::default()
        }
    }

    /// Moon-only third-body gravity.
    pub fn moon() -> Self {
        Self {
            bodies: ThirdBodyBodies::MOON,
            ..Self::default()
        }
    }
}

impl ForceModel for ThirdBodyGravity {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let bodies = sun_moon_positions_km(state.epoch_tdb_seconds)?;
        let mut accel = Vector3::zeros();
        if self.bodies.sun {
            accel += third_body_acceleration(state.position_km, bodies.sun_km, self.gm_sun_km3_s2)?;
        }
        if self.bodies.moon {
            accel +=
                third_body_acceleration(state.position_km, bodies.moon_km, self.gm_moon_km3_s2)?;
        }
        Ok(accel)
    }
}

struct SunMoonKm {
    sun_km: Vector3<f64>,
    moon_km: Vector3<f64>,
}

fn sun_moon_positions_km(epoch_tdb_seconds: f64) -> Result<SunMoonKm, PropagationError> {
    let bodies = sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds)
        .map_err(|error| PropagationError::ForceModelFailure(format!("Sun/Moon: {error}")))?;
    Ok(SunMoonKm {
        sun_km: meters_to_km(bodies.sun),
        moon_km: meters_to_km(bodies.moon),
    })
}

fn meters_to_km(position_m: [f64; 3]) -> Vector3<f64> {
    Vector3::new(
        position_m[0] / M_PER_KM,
        position_m[1] / M_PER_KM,
        position_m[2] / M_PER_KM,
    )
}

fn third_body_acceleration(
    sat_pos_km: Vector3<f64>,
    body_pos_km: Vector3<f64>,
    gm_body_km3_s2: f64,
) -> Result<Vector3<f64>, PropagationError> {
    let body_r2 = body_pos_km.norm_squared();
    if body_r2 == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Zero third-body position magnitude".to_string(),
        ));
    }
    let delta = body_pos_km - sat_pos_km;
    let delta_r2 = delta.norm_squared();
    if delta_r2 == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Satellite coincident with third body".to_string(),
        ));
    }

    let body_r = body_r2.sqrt();
    let delta_r = delta_r2.sqrt();
    Ok(gm_body_km3_s2 * (delta / (delta_r2 * delta_r) - body_pos_km / (body_r2 * body_r)))
}

#[cfg(test)]
mod tests {
    //! Clean-room validation anchors:
    //!
    //! The exact vector checks use fixed Sun and Moon stand-ins and numeric
    //! values evaluated independently from the published direct-minus-indirect
    //! third-body equation in Montenbruck & Gill, section 3.2. The magnitude
    //! checks are the textbook LEO order of `1e-7..1e-6 m/s^2` for solar and
    //! lunar third-body perturbations.

    use super::*;
    use crate::astro::constants::astro::AU_KM;

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
    fn fixed_sun_and_moon_vectors_match_independent_closed_form() {
        let sat = Vector3::new(7000.0, -1210.0, 1300.0);
        let sun = third_body_acceleration(sat, Vector3::new(AU_KM, 0.0, 0.0), GM_SUN_KM3_S2)
            .expect("sun acceleration");
        let moon = third_body_acceleration(sat, Vector3::new(384_400.0, 0.0, 0.0), GM_MOON_KM3_S2)
            .expect("moon acceleration");

        assert_close_vector(
            sun,
            [
                5.549_999_393_729_882e-10,
                4.797_132_723_126_184e-11,
                -5.153_944_247_986_81e-11,
            ],
            1.0e-20,
        );
        assert_close_vector(
            moon,
            [
                1.241_117_020_105_779_8e-9,
                1.103_594_278_427_440_5e-10,
                -1.185_679_803_269_151e-10,
            ],
            1.0e-20,
        );
    }

    #[test]
    fn indirect_term_cancels_at_geocenter() {
        let state = CartesianState::new(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]);
        let accel = ThirdBodyGravity::default()
            .acceleration(&state, &PropagationContext::default())
            .expect("third-body acceleration");
        assert_eq!(accel, Vector3::zeros());
    }

    #[test]
    fn leo_magnitudes_match_textbook_order() {
        let state = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]);
        let sun = ThirdBodyGravity::sun()
            .acceleration(&state, &PropagationContext::default())
            .expect("sun acceleration")
            .norm()
            * M_PER_KM;
        let moon = ThirdBodyGravity::moon()
            .acceleration(&state, &PropagationContext::default())
            .expect("moon acceleration")
            .norm()
            * M_PER_KM;

        assert!(
            (1.0e-7..1.0e-6).contains(&sun),
            "sun perturbation magnitude {sun} m/s^2"
        );
        assert!(
            (1.0e-7..2.0e-6).contains(&moon),
            "moon perturbation magnitude {moon} m/s^2"
        );
    }

    /// TDB epochs from 1990 to 2050, one at the 2024-04-08 solar eclipse.
    fn epochs_1990_to_2050() -> [f64; 4] {
        use crate::astro::time::civil::j2000_seconds;
        [
            j2000_seconds(1990, 2, 1, 0, 0, 0.0),
            j2000_seconds(2024, 4, 8, 18, 17, 0.0),
            j2000_seconds(2026, 11, 23, 0, 0, 0.0),
            j2000_seconds(2050, 6, 30, 12, 0, 0.0),
        ]
    }

    fn gcrs_sun_moon_km(epoch_tdb_seconds: f64) -> (Vector3<f64>, Vector3<f64>) {
        let bodies =
            crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds)
                .expect("GCRS Sun/Moon");
        let km = |v: [f64; 3]| Vector3::new(v[0] / M_PER_KM, v[1] / M_PER_KM, v[2] / M_PER_KM);
        (km(bodies.sun), km(bodies.moon))
    }

    fn mean_of_date_sun_moon_km(epoch_tdb_seconds: f64) -> (Vector3<f64>, Vector3<f64>) {
        let t = crate::astro::time::scales::tt_seconds_from_tdb_seconds(epoch_tdb_seconds)
            / (36_525.0 * 86_400.0);
        let bodies = crate::astro::bodies::sun_moon::sun_moon_eci(t).expect("mean of date");
        let km = |v: [f64; 3]| Vector3::new(v[0] / M_PER_KM, v[1] / M_PER_KM, v[2] / M_PER_KM);
        (km(bodies.sun), km(bodies.moon))
    }

    #[test]
    fn acceleration_equals_the_point_mass_terms_of_the_gcrs_sun_and_moon() {
        let ctx = PropagationContext::default();
        for epoch in epochs_1990_to_2050() {
            let (sun, moon) = gcrs_sun_moon_km(epoch);
            for position in [
                Vector3::new(7_000.0, -1_210.0, 1_300.0),
                Vector3::new(26_560.0, 1_200.0, -3_400.0),
                Vector3::new(-42_164.0, 0.0, 0.0),
            ] {
                let state = CartesianState::new(epoch, position.into(), [0.0, 7.5, 0.0]);
                let sun_term =
                    third_body_acceleration(position, sun, GM_SUN_KM3_S2).expect("Sun term");
                let moon_term =
                    third_body_acceleration(position, moon, GM_MOON_KM3_S2).expect("Moon term");
                let both = ThirdBodyGravity::default()
                    .acceleration(&state, &ctx)
                    .expect("Sun and Moon");
                let mut expected = Vector3::zeros();
                expected += sun_term;
                expected += moon_term;
                assert_eq!(both, expected, "epoch {epoch}");
                let mut sun_only = Vector3::zeros();
                sun_only += sun_term;
                assert_eq!(
                    ThirdBodyGravity::sun()
                        .acceleration(&state, &ctx)
                        .expect("Sun"),
                    sun_only
                );
                let mut moon_only = Vector3::zeros();
                moon_only += moon_term;
                assert_eq!(
                    ThirdBodyGravity::moon()
                        .acceleration(&state, &ctx)
                        .expect("Moon"),
                    moon_only
                );
            }
        }
    }

    #[test]
    fn acceleration_no_longer_uses_the_mean_of_date_bodies() {
        let epoch = crate::astro::time::civil::j2000_seconds(2026, 11, 23, 0, 0, 0.0);
        let (mean_sun, mean_moon) = mean_of_date_sun_moon_km(epoch);
        let position = Vector3::new(26_560.0, 1_200.0, -3_400.0);
        let state = CartesianState::new(epoch, position.into(), [0.0, 7.5, 0.0]);
        let used = ThirdBodyGravity::default()
            .acceleration(&state, &PropagationContext::default())
            .expect("Sun and Moon");
        let with_mean = third_body_acceleration(position, mean_sun, GM_SUN_KM3_S2)
            .expect("Sun term")
            + third_body_acceleration(position, mean_moon, GM_MOON_KM3_S2).expect("Moon term");
        // Turning the bodies by 0.37 deg (0.0065 rad) changes the tidal
        // acceleration by a fraction of that order.
        assert!((used - with_mean).norm() > 1.0e-3 * used.norm());
    }
}
