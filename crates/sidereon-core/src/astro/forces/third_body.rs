//! Sun and Moon third-body point-mass perturbations.
//!
//! The acceleration uses the standard direct-minus-indirect expression
//! `mu_b * ((r_b - r) / |r_b - r|^3 - r_b / |r_b|^3)`, as given by
//! Montenbruck & Gill, "Satellite Orbits", section 3.2, and Vallado,
//! "Fundamentals of Astrodynamics and Applications", section 8.6. Body
//! positions come from the context's [DE ephemeris](PropagationContext::de_ephemeris)
//! or the default analytic Sun/Moon series in GCRS
//! ([`sun_moon_gcrs_at_tdb_seconds`]). Only the direct term depends on the satellite position, so
//! the position partials are `-mu_b / d^3 * (I - 3 d̂ d̂ᵀ)` with `d = r_b - r`
//! (Montenbruck & Gill, section 7.3).

use super::scaled::{wide_norm_squared, Wide};
use crate::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds;
use crate::astro::constants::astro::{GM_MOON_KM3_S2, GM_SUN_KM3_S2};
use crate::astro::constants::units::M_PER_KM;
use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_partials, validate_finite,
    validate_position, AccelerationPartials, ForceModel,
};
use crate::astro::forces::two_body::{binary64_point_mass_gradient, scaled_point_mass_gradient};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};

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
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        if self.bodies == ThirdBodyBodies::NONE {
            return Ok(Vector3::zeros());
        }
        self.validate(state)?;
        let bodies = sun_moon_positions_km(state.epoch_tdb_seconds, ctx)?;
        let active = self.active_bodies(&bodies);
        let mut acceleration = Vector3::zeros();
        let mut recover = false;
        for (body, gm) in active.clone() {
            match binary64_third_body_acceleration(state.position_km, body, gm)? {
                Some(contribution) => acceleration += contribution,
                None => recover = true,
            }
        }
        if !recover && acceleration.iter().all(|value| value.is_finite()) {
            return Ok(acceleration);
        }
        let mut sum = [Wide::from(0.0); 3];
        for (body, gm) in active {
            let contribution = scaled_third_body_acceleration(state.position_km, body, gm);
            for (total, value) in sum.iter_mut().zip(contribution) {
                *total = *total + value;
            }
        }
        finite_acceleration(Vector3::from(sum.map(Wide::value)))
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        expect_parameter_count(parameters, 0)?;
        if self.bodies == ThirdBodyBodies::NONE {
            return Ok(AccelerationPartials::zeros());
        }
        self.validate(state)?;
        let bodies = sun_moon_positions_km(state.epoch_tdb_seconds, ctx)?;
        let active = self.active_bodies(&bodies);
        let mut gradient = Matrix3::zeros();
        let mut recover = false;
        for (body, gm) in active.clone() {
            let separation = body - state.position_km;
            if separation.iter().all(|&value| value == 0.0) {
                return Err(coincident_with_third_body());
            }
            match binary64_point_mass_gradient(separation, gm) {
                Some(contribution) => gradient -= contribution,
                None => recover = true,
            }
        }
        if !recover && gradient.iter().all(|value| value.is_finite()) {
            return Ok(AccelerationPartials::position_only(gradient));
        }
        let mut sum = [Wide::from(0.0); 9];
        let satellite = state.position_array().map(Wide::from);
        for (body, gm) in active {
            let body = <[f64; 3]>::from(body).map(Wide::from);
            let separation = core::array::from_fn(|i| body[i] - satellite[i]);
            let contribution = scaled_point_mass_gradient(separation, gm)
                .map_err(|_| coincident_with_third_body())?;
            for (total, value) in sum.iter_mut().zip(contribution) {
                *total = *total - value;
            }
        }
        let gradient = Matrix3::from_column_slice(&sum.map(Wide::value));
        finite_partials(AccelerationPartials::position_only(gradient))
    }
}

impl ThirdBodyGravity {
    fn active_bodies(
        &self,
        bodies: &SunMoonKm,
    ) -> impl Iterator<Item = (Vector3<f64>, f64)> + Clone {
        [
            self.bodies
                .sun
                .then_some((bodies.sun_km, self.gm_sun_km3_s2)),
            self.bodies
                .moon
                .then_some((bodies.moon_km, self.gm_moon_km3_s2)),
        ]
        .into_iter()
        .flatten()
    }

    fn validate(&self, state: &CartesianState) -> Result<(), PropagationError> {
        if self.bodies.sun {
            validate_finite([self.gm_sun_km3_s2], "solar gravitational parameter")?;
        }
        if self.bodies.moon {
            validate_finite([self.gm_moon_km3_s2], "lunar gravitational parameter")?;
        }
        validate_position(state)
    }
}

fn coincident_with_third_body() -> PropagationError {
    PropagationError::NumericalFailure("Satellite coincident with third body".to_string())
}

/// Geocentric Sun and Moon positions in GCRS, km.
pub(crate) struct SunMoonKm {
    pub(crate) sun_km: Vector3<f64>,
    pub(crate) moon_km: Vector3<f64>,
}

/// The Sun and Moon positions in GCRS at a TDB epoch (seconds past J2000),
/// from the attached DE ephemeris or the analytic series.
pub(crate) fn sun_moon_positions_km(
    epoch_tdb_seconds: f64,
    ctx: &PropagationContext,
) -> Result<SunMoonKm, PropagationError> {
    validate_finite([epoch_tdb_seconds], "epoch")?;
    let body_error = |error: &dyn core::fmt::Display| {
        PropagationError::ForceModelFailure(format!("Sun/Moon: {error}"))
    };
    let bodies = match ctx.de_ephemeris() {
        Some(ephemeris) => ephemeris
            .sun_moon_positions_at_tdb_seconds(epoch_tdb_seconds)
            .map_err(|error| body_error(&error))?,
        None => {
            sun_moon_gcrs_at_tdb_seconds(epoch_tdb_seconds).map_err(|error| body_error(&error))?
        }
    };
    if !bodies
        .sun
        .iter()
        .chain(&bodies.moon)
        .all(|value| value.is_finite())
    {
        return Err(body_error(&"non-finite position at epoch"));
    }
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

fn binary64_third_body_acceleration(
    sat_pos_km: Vector3<f64>,
    body_pos_km: Vector3<f64>,
    gm_body_km3_s2: f64,
) -> Result<Option<Vector3<f64>>, PropagationError> {
    let body_r2 = body_pos_km.norm_squared();
    if body_pos_km.iter().all(|&value| value == 0.0) {
        return Err(PropagationError::NumericalFailure(
            "Zero third-body position magnitude".to_string(),
        ));
    }
    let delta = body_pos_km - sat_pos_km;
    let delta_r2 = delta.norm_squared();
    if delta.iter().all(|&value| value == 0.0) {
        return Err(PropagationError::NumericalFailure(
            "Satellite coincident with third body".to_string(),
        ));
    }

    let body_r = body_r2.sqrt();
    let delta_r = delta_r2.sqrt();
    let body_r3 = body_r2 * body_r;
    let delta_r3 = delta_r2 * delta_r;
    let acceleration = gm_body_km3_s2 * (delta / delta_r3 - body_pos_km / body_r3);
    if body_r3.is_finite()
        && delta_r3.is_finite()
        && acceleration.iter().all(|value| value.is_finite())
    {
        return Ok(Some(acceleration));
    }

    Ok(None)
}

#[cfg(test)]
fn third_body_acceleration(
    satellite: Vector3<f64>,
    body: Vector3<f64>,
    gm: f64,
) -> Result<Vector3<f64>, PropagationError> {
    Ok(
        binary64_third_body_acceleration(satellite, body, gm)?.unwrap_or_else(|| {
            Vector3::from(scaled_third_body_acceleration(satellite, body, gm).map(Wide::value))
        }),
    )
}

fn scaled_third_body_acceleration(
    sat_pos_km: Vector3<f64>,
    body_pos_km: Vector3<f64>,
    gm_body_km3_s2: f64,
) -> [Wide; 3] {
    let body = <[f64; 3]>::from(body_pos_km).map(Wide::from);
    let satellite = <[f64; 3]>::from(sat_pos_km).map(Wide::from);
    let delta = core::array::from_fn(|i| body[i] - satellite[i]);
    let body_r2 = wide_norm_squared(body);
    let delta_r2 = wide_norm_squared(delta);
    let body_r3 = body_r2 * body_r2.sqrt();
    let delta_r3 = delta_r2 * delta_r2.sqrt();
    core::array::from_fn(|i| Wide::from(gm_body_km3_s2) * (delta[i] / delta_r3 - body[i] / body_r3))
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
    use crate::astro::forces::two_body::point_mass_gradient;

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
    fn tidal_term_recovers_when_both_radius_cubes_underflow() {
        let body = Vector3::new(2.0, 0.0, 0.0);
        let satellite = Vector3::x();
        let gm = libm::ldexp(1.0, -800);
        let reference = third_body_acceleration(satellite, body, gm).unwrap();
        let scale = libm::ldexp(1.0, -400);
        let actual = third_body_acceleration(satellite * scale, body * scale, gm).unwrap();
        assert_ne!(reference.x, 0.0);
        assert_eq!(actual.x.to_bits(), libm::ldexp(reference.x, 800).to_bits());
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

    use crate::astro::forces::test_helpers::{
        epochs_1990_to_2050, gcrs_sun_moon_km, mean_of_date_sun_moon_km,
    };

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
                let mut expected_partials = Matrix3::zeros();
                expected_partials -=
                    point_mass_gradient(sun - position, GM_SUN_KM3_S2).expect("Sun gradient");
                expected_partials -=
                    point_mass_gradient(moon - position, GM_MOON_KM3_S2).expect("Moon gradient");
                let both_partials = ThirdBodyGravity::default()
                    .acceleration_partials(&[], &state, &ctx)
                    .expect("Sun and Moon partials");
                assert_eq!(both_partials.wrt_position, expected_partials);
                assert_eq!(both_partials.wrt_velocity, Matrix3::zeros());
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
