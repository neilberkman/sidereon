use super::*;
use crate::astro::error::PropagationError;
use crate::astro::frames::orientation::TdbEarthOrientationProvider;
use crate::astro::propagator::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};
use std::sync::Arc;

fn state() -> CartesianState {
    CartesianState::new(0.0, [6_800.0, -390.0, 250.0], [0.1, 7.5, 0.8])
}

fn context() -> PropagationContext {
    PropagationContext::default()
        .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::default()))
}

fn invalid<T: std::fmt::Debug>(result: Result<T, PropagationError>) {
    assert!(
        matches!(result, Err(PropagationError::InvalidInput(_))),
        "{result:?}"
    );
}

fn parameters(model: &dyn ForceModel) -> Vec<f64> {
    model
        .parameters()
        .into_iter()
        .map(|parameter| parameter.value)
        .collect()
}

fn invalid_state(model: &dyn ForceModel, bad: &CartesianState, ctx: &PropagationContext) {
    let parameters = parameters(model);
    invalid(model.acceleration(bad, ctx));
    invalid(model.acceleration_with_parameters(&parameters, bad, ctx));
    invalid(model.acceleration_partials(&parameters, bad, ctx));
    if parameters.is_empty() {
        assert!(model.parameter_partials(&[], bad, ctx).unwrap().is_empty());
    } else {
        invalid(model.parameter_partials(&parameters, bad, ctx));
    }
}

fn ignores_state(model: &dyn ForceModel, bad: &CartesianState, ctx: &PropagationContext) {
    let parameters = parameters(model);
    assert_eq!(
        model.acceleration(bad, ctx).unwrap(),
        model.acceleration(&state(), ctx).unwrap()
    );
    assert_eq!(
        model
            .acceleration_with_parameters(&parameters, bad, ctx)
            .unwrap(),
        model
            .acceleration_with_parameters(&parameters, &state(), ctx)
            .unwrap()
    );
    assert_eq!(
        model.acceleration_partials(&parameters, bad, ctx).unwrap(),
        model
            .acceleration_partials(&parameters, &state(), ctx)
            .unwrap()
    );
    assert_eq!(
        model.parameter_partials(&parameters, bad, ctx).unwrap(),
        model
            .parameter_partials(&parameters, &state(), ctx)
            .unwrap()
    );
}

#[test]
fn position_only_forces_validate_their_inputs() {
    let inertial = PropagationContext::default();
    let fixed = context();
    let nominal = state();
    let mut bad_position = nominal;
    bad_position.position_km.x = f64::NAN;
    let mut bad_velocity = nominal;
    bad_velocity.velocity_km_s.y = f64::INFINITY;
    let mut bad_epoch = nominal;
    bad_epoch.epoch_tdb_seconds = f64::NAN;

    let harmonics = SphericalHarmonicGravity::from_normalized_coefficients(
        4.0e5,
        6_378.0,
        2,
        0,
        &[SphericalHarmonicCoefficient {
            degree: 2,
            order: 0,
            c: -0.001,
            s: 0.0,
        }],
        TideSystem::TideFree,
    )
    .unwrap();
    for model in [
        &TwoBodyGravity::default() as &dyn ForceModel,
        &J2Gravity::default(),
        &ZonalGravity::default(),
        &harmonics,
    ] {
        invalid_state(model, &bad_position, &inertial);
        ignores_state(model, &bad_velocity, &inertial);
        ignores_state(model, &bad_epoch, &inertial);
    }

    let sun = SolarRadiationPressure::new(1.3, 0.02).unwrap();
    let earth = EarthRadiationPressure::new(1.3, 0.02).unwrap();
    for model in [
        &ThirdBodyGravity::default() as &dyn ForceModel,
        &sun,
        &earth,
        &harmonics,
        &SolidEarthTideGravity::default(),
        &SolidEarthPoleTideGravity::default(),
    ] {
        invalid_state(model, &bad_position, &fixed);
        invalid_state(model, &bad_epoch, &fixed);
        ignores_state(model, &bad_velocity, &fixed);
    }
}

#[test]
fn velocity_dependent_forces_validate_their_inputs() {
    let ctx = context();
    let nominal = state();
    let mut bad_position = nominal;
    bad_position.position_km.y = f64::INFINITY;
    let mut bad_velocity = nominal;
    bad_velocity.velocity_km_s.z = f64::NAN;
    let mut bad_epoch = nominal;
    bad_epoch.epoch_tdb_seconds = f64::NAN;
    let rt = EmpiricalRtnAcceleration::constant_terms();
    let relativity = SchwarzschildRelativity::default();
    for model in [&rt as &dyn ForceModel, &relativity] {
        invalid_state(model, &bad_position, &ctx);
        invalid_state(model, &bad_velocity, &ctx);
        ignores_state(model, &bad_epoch, &ctx);
    }

    let radial = EmpiricalRtnAcceleration::new()
        .with_term(RtnAxis::Radial, EmpiricalHarmonic::Constant, 1.0e-8)
        .unwrap();
    ignores_state(&radial, &bad_velocity, &ctx);

    let ecom1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        dc: 1.0e-8,
        ..Ecom1Coefficients::default()
    })
    .unwrap();
    let ecom2 = Ecom2SolarRadiationPressure::new(1, 0, vec![0.0, 0.0, 1.0e-8, 0.0, 0.0]).unwrap();
    let drag = DragParameters::from_bc_factor_m2_kg(0.02, SpaceWeather::default(), 0.0).unwrap();
    let sourced = SourcedDragForce::new(drag, SpaceWeatherSource::Fixed(SpaceWeather::default()));
    for model in [
        &ecom1 as &dyn ForceModel,
        &ecom2,
        &drag.to_force(),
        &sourced,
    ] {
        invalid_state(model, &bad_position, &ctx);
        invalid_state(model, &bad_velocity, &ctx);
        invalid_state(model, &bad_epoch, &ctx);
    }

    let mut above_atmosphere = bad_velocity;
    above_atmosphere.position_km *= 2.0;
    for model in [&drag.to_force() as &dyn ForceModel, &sourced] {
        assert_eq!(
            model.acceleration(&above_atmosphere, &ctx).unwrap(),
            Vector3::zeros()
        );
        assert_eq!(
            model
                .acceleration_partials(&[], &above_atmosphere, &ctx)
                .unwrap(),
            AccelerationPartials::zeros()
        );
    }
    let constant = Ecom2SolarRadiationPressure::new(0, 0, vec![0.0, 1.0e-8, 0.0]).unwrap();
    ignores_state(&constant, &bad_velocity, &ctx);
    let constant = Ecom1SolarRadiationPressure::new(Ecom1Coefficients {
        d0: 1.0e-8,
        ..Ecom1Coefficients::default()
    })
    .unwrap();
    assert_eq!(
        constant.acceleration(&bad_velocity, &ctx).unwrap(),
        constant.acceleration(&nominal, &ctx).unwrap()
    );
    assert_eq!(
        constant
            .acceleration_partials(&parameters(&constant), &bad_velocity, &ctx)
            .unwrap(),
        constant
            .acceleration_partials(&parameters(&constant), &nominal, &ctx)
            .unwrap()
    );
    invalid(constant.parameter_partials(&parameters(&constant), &bad_velocity, &ctx));
}

#[test]
fn stored_constants_and_supplied_parameters_are_validated() {
    let ctx = context();
    let state = state();
    let nan = f64::NAN;
    let zonal = ZonalGravity::default();
    let third = ThirdBodyGravity::default();
    let tide = SolidEarthTideGravity::default();
    let bad_constants: &[&dyn ForceModel] = &[
        &TwoBodyGravity { mu: nan },
        &J2Gravity {
            mu: nan,
            ..J2Gravity::default()
        },
        &J2Gravity {
            re: nan,
            ..J2Gravity::default()
        },
        &J2Gravity {
            j2: nan,
            ..J2Gravity::default()
        },
        &ZonalGravity {
            mu_km3_s2: nan,
            ..zonal
        },
        &ZonalGravity {
            re_km: nan,
            ..zonal
        },
        &ZonalGravity {
            coefficients: ZonalCoefficients {
                j2: nan,
                ..zonal.coefficients
            },
            ..zonal
        },
        &SchwarzschildRelativity::new(nan, 3.0e5),
        &SchwarzschildRelativity::new(4.0e5, f64::NEG_INFINITY),
        &ThirdBodyGravity {
            gm_sun_km3_s2: nan,
            ..third
        },
        &ThirdBodyGravity {
            gm_moon_km3_s2: nan,
            ..third
        },
        &SolidEarthTideGravity {
            mu_earth_km3_s2: nan,
            ..tide
        },
        &SolidEarthTideGravity {
            reference_radius_km: nan,
            ..tide
        },
        &SolidEarthTideGravity {
            gm_sun_km3_s2: nan,
            ..tide
        },
        &SolidEarthTideGravity {
            gm_moon_km3_s2: nan,
            ..tide
        },
        &SolidEarthPoleTideGravity::new(nan, 6_378.0),
        &SolidEarthPoleTideGravity::new(4.0e5, nan),
    ];
    for model in bad_constants {
        invalid_state(*model, &state, &ctx);
    }
    let sun_only = ThirdBodyGravity::new(ThirdBodyBodies::SUN, third.gm_sun_km3_s2, nan);
    assert_eq!(
        sun_only.acceleration(&state, &ctx).unwrap(),
        ThirdBodyGravity::sun().acceleration(&state, &ctx).unwrap()
    );
    let drag = DragParameters::from_bc_factor_m2_kg(0.02, SpaceWeather::default(), 0.0).unwrap();
    for space_weather in [
        SpaceWeather {
            f107: nan,
            ..SpaceWeather::default()
        },
        SpaceWeather {
            f107a: nan,
            ..SpaceWeather::default()
        },
        SpaceWeather {
            ap: nan,
            ..SpaceWeather::default()
        },
    ] {
        invalid(DragParameters::from_bc_factor_m2_kg(
            0.02,
            space_weather,
            0.0,
        ));
        invalid_state(
            &SourcedDragForce::new(drag, SpaceWeatherSource::Fixed(space_weather)),
            &state,
            &ctx,
        );
    }
    invalid(DragParameters::from_bc_factor_m2_kg(
        nan,
        SpaceWeather::default(),
        0.0,
    ));
    invalid(DragParameters::from_bc_factor_m2_kg(
        0.02,
        SpaceWeather::default(),
        nan,
    ));
    invalid(EarthRadiationPressure::new(nan, 0.02));
    invalid(EarthRadiationPressure::new(1.3, nan));
    for values in [
        [nan, 1.0, 1.0, 1.0],
        [1.0, nan, 1.0, 1.0],
        [1.0, 1.0, nan, 1.0],
        [1.0, 1.0, 1.0, nan],
    ] {
        let [cr, area, pressure, au] = values;
        invalid(SolarRadiationPressure::with_pressure(
            cr, area, pressure, au,
        ));
    }
    let ecom1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients::default()).unwrap();
    let ecom2 = Ecom2SolarRadiationPressure::new(1, 1, vec![0.0; 7]).unwrap();
    let srp = SolarRadiationPressure::new(1.3, 0.02).unwrap();
    let rt = EmpiricalRtnAcceleration::all_terms();
    for model in [&ecom1 as &dyn ForceModel, &ecom2, &srp, &rt] {
        for index in 0..model.parameter_count() {
            let mut values = parameters(model);
            values[index] = nan;
            invalid(model.acceleration_with_parameters(&values, &state, &ctx));
            invalid(model.acceleration_partials(&values, &state, &ctx));
            invalid(model.parameter_partials(&values, &state, &ctx));
        }
    }
}

fn pow2(exponent: i32) -> f64 {
    libm::ldexp(1.0, exponent)
}

fn axial(exponent: i32) -> CartesianState {
    CartesianState::new(f64::NAN, [pow2(exponent), 0.0, 0.0], [0.0; 3])
}

#[test]
fn point_mass_exponent_recovery_preserves_homogeneity() {
    let ctx = PropagationContext::default();
    for (mu, k) in [(pow2(-600), -400), (1.0, 530), (6.0, 538), (pow2(700), 360)] {
        let model = TwoBodyGravity { mu };
        let reference = model.acceleration(&axial(0), &ctx).unwrap().x;
        let reference_gradient = model
            .acceleration_partials(&[], &axial(0), &ctx)
            .unwrap()
            .wrt_position;
        let position = axial(k);
        let actual = model.acceleration(&position, &ctx).unwrap().x;
        let actual_gradient = model
            .acceleration_partials(&[], &position, &ctx)
            .unwrap()
            .wrt_position;
        let expected = libm::ldexp(reference, -2 * k);
        assert_ne!(expected, 0.0);
        assert_eq!(actual.to_bits(), expected.to_bits());
        assert_eq!(
            actual_gradient[(0, 0)].to_bits(),
            libm::ldexp(reference_gradient[(0, 0)], -3 * k).to_bits()
        );
    }
}

#[test]
fn high_degree_and_relativistic_recovery_preserve_homogeneity() {
    let ctx = PropagationContext::default();
    let j2 = J2Gravity {
        mu: pow2(-650),
        re: 1.0,
        j2: 1.0,
    };
    let zonal = ZonalGravity::new(
        pow2(-650),
        1.0,
        ZonalDegrees::through(2).unwrap(),
        ZonalCoefficients {
            j2: 1.0,
            ..ZonalCoefficients::default()
        },
    );
    let rel = SchwarzschildRelativity::new(pow2(-290), 1.0);
    for (model, acceleration_degree, exponent) in [
        (&j2 as &dyn ForceModel, 4, -170),
        (&zonal, 4, -230),
        (&rel, 3, -380),
    ] {
        let base = model.acceleration(&axial(0), &ctx).unwrap().x;
        let base_gradient = model
            .acceleration_partials(&[], &axial(0), &ctx)
            .unwrap()
            .wrt_position[(0, 0)];
        let state = axial(exponent);
        let actual = model.acceleration(&state, &ctx).unwrap().x;
        let gradient = model
            .acceleration_partials(&[], &state, &ctx)
            .unwrap()
            .wrt_position[(0, 0)];
        assert_eq!(
            actual.to_bits(),
            libm::ldexp(base, -acceleration_degree * exponent).to_bits()
        );
        assert_eq!(
            gradient.to_bits(),
            libm::ldexp(base_gradient, -(acceleration_degree + 1) * exponent).to_bits()
        );
    }
}

#[test]
fn zonal_and_relativistic_gravity_recover_overflowing_radius_powers() {
    let zonal = ZonalGravity::new(
        pow2(500),
        pow2(100),
        ZonalDegrees::through(2).unwrap(),
        ZonalCoefficients {
            j2: 1.0,
            ..ZonalCoefficients::default()
        },
    );
    let relativity = SchwarzschildRelativity::new(pow2(225), pow2(260));
    let ctx = PropagationContext::default();
    for (model, radius_exponent, acceleration, gradient) in [
        (
            &zonal as &dyn ForceModel,
            210,
            -1.5 * pow2(-140),
            6.0 * pow2(-350),
        ),
        (&relativity, 180, pow2(-608), -3.0 * pow2(-788)),
    ] {
        let state = axial(radius_exponent);
        assert_eq!(
            model.acceleration(&state, &ctx).unwrap().x.to_bits(),
            acceleration.to_bits()
        );
        assert_eq!(
            model
                .acceleration_partials(&[], &state, &ctx)
                .unwrap()
                .wrt_position[(0, 0)]
                .to_bits(),
            gradient.to_bits()
        );
    }
}

#[test]
fn j2_keeps_signed_zeros_when_the_numerator_underflows() {
    let ctx = PropagationContext::default();
    let j2 = J2Gravity {
        mu: pow2(-950),
        re: pow2(-150),
        j2: 1.0,
    };
    let acceleration = j2.acceleration(&axial(-60), &ctx).unwrap();
    assert_eq!(
        acceleration.map(f64::to_bits),
        Vector3::repeat((-0.0_f64).to_bits())
    );
    let fallback = j2.acceleration(&axial(-545), &ctx).unwrap().x;
    assert_eq!(fallback.to_bits(), (-1.5 * pow2(930)).to_bits());
}

#[test]
fn j2_recovers_when_the_cubed_radius_overflows() {
    let model = J2Gravity {
        mu: pow2(600),
        re: pow2(200),
        j2: 0.125,
    };
    let state = axial(350);
    let ctx = PropagationContext::default();
    // On the x axis, a_x = -(3/2) J2 mu Re² / x⁴ and da_x/dx = -4 a_x / x.
    let acceleration = model.acceleration(&state, &ctx).unwrap();
    let partials = model.acceleration_partials(&[], &state, &ctx).unwrap();
    assert_eq!(
        acceleration.map(f64::to_bits),
        Vector3::new(-0.1875 * pow2(-400), -0.0, -0.0).map(f64::to_bits)
    );
    let diagonal = Vector3::new(0.75, -0.1875, -0.5625) * pow2(-750);
    assert_eq!(
        partials.wrt_position.map(f64::to_bits),
        Matrix3::from_diagonal(&diagonal).map(f64::to_bits)
    );
}

#[test]
fn newtonian_limit_returns_zero_acceleration_and_partials() {
    let model = SchwarzschildRelativity::new(4.0e5, f64::INFINITY);
    let ctx = PropagationContext::default();
    for state in [
        state(),
        CartesianState::new(f64::NAN, [1.0, 0.0, 0.0], [f64::MAX; 3]),
    ] {
        assert_eq!(
            model.acceleration(&state, &ctx).unwrap().map(f64::to_bits),
            Vector3::zeros()
        );
        assert_eq!(
            model.acceleration_partials(&[], &state, &ctx).unwrap(),
            AccelerationPartials::zeros()
        );
    }
}

#[test]
fn newtonian_limit_refuses_the_origin() {
    let model = SchwarzschildRelativity::new(4.0e5, f64::INFINITY);
    let ctx = PropagationContext::default();
    let state = CartesianState::new(0.0, [0.0; 3], [0.0; 3]);
    for error in [
        model.acceleration(&state, &ctx).unwrap_err(),
        model.acceleration_partials(&[], &state, &ctx).unwrap_err(),
    ] {
        assert!(matches!(
            error,
            PropagationError::NumericalFailure(message) if message == "Zero position magnitude"
        ));
    }
}

#[test]
fn newtonian_limit_validates_mass_position_and_velocity() {
    let ctx = PropagationContext::default();
    let nominal = state();
    for value in [f64::NAN, f64::INFINITY] {
        let model = SchwarzschildRelativity::new(4.0e5, f64::INFINITY);
        let mut bad_position = nominal;
        bad_position.position_km.x = value;
        let mut bad_velocity = nominal;
        bad_velocity.velocity_km_s.x = value;
        invalid_state(&model, &bad_position, &ctx);
        invalid_state(&model, &bad_velocity, &ctx);
        invalid_state(
            &SchwarzschildRelativity::new(value, f64::INFINITY),
            &nominal,
            &ctx,
        );
    }
}

#[test]
fn empirical_harmonics_refuse_below_the_inclination_cutoff() {
    let model = EmpiricalRtnAcceleration::all_terms();
    for scale in [1.0, pow2(600)] {
        let state = CartesianState::new(
            0.0,
            [7000.0 * scale, 0.0, 0.0],
            [0.0, 7.5 * scale, 7.5e-13 * scale],
        );
        assert_empirical_geometry_error(
            &model,
            &state,
            "equatorial orbit: the argument of latitude is undefined",
        );
    }
}

#[test]
fn empirical_frame_refuses_a_cross_product_that_cancels_to_zero() {
    let model = EmpiricalRtnAcceleration::constant_terms();
    let state = CartesianState::new(
        0.0,
        [1.0 + pow2(-52), 1.0, 0.0],
        [1.0 + pow2(-51), 1.0 + pow2(-52), 0.0],
    );
    assert_empirical_geometry_error(
        &model,
        &state,
        "position and velocity are parallel: the orbit plane is undefined",
    );
}

fn assert_empirical_geometry_error(
    model: &EmpiricalRtnAcceleration,
    state: &CartesianState,
    reason: &str,
) {
    let ctx = PropagationContext::default();
    let values = parameters(model);
    for error in [
        model.acceleration(state, &ctx).unwrap_err(),
        model
            .acceleration_partials(&values, state, &ctx)
            .unwrap_err(),
        model.parameter_partials(&values, state, &ctx).unwrap_err(),
    ] {
        assert!(matches!(error, PropagationError::NumericalFailure(message) if message == reason));
    }
}

#[test]
fn empirical_sum_starts_at_positive_zero() {
    let model = EmpiricalRtnAcceleration::constant_terms();
    let ctx = PropagationContext::default();
    let values = [0.0, -1.0, -1.0];
    for scale in [1.0, pow2(600)] {
        let state = CartesianState::new(0.0, [-scale, 0.0, 0.0], [0.0, -scale, 0.0]);
        let acceleration = model
            .acceleration_with_parameters(&values, &state, &ctx)
            .unwrap();
        assert_eq!(
            acceleration.map(f64::to_bits),
            Vector3::new(0.0, 1.0, -1.0).map(f64::to_bits)
        );
    }
}

#[test]
fn inactive_zonal_gravity_ignores_the_state_and_constants() {
    let model = ZonalGravity::new(
        f64::NAN,
        f64::INFINITY,
        ZonalDegrees::NONE,
        ZonalCoefficients {
            j2: f64::NAN,
            ..ZonalCoefficients::default()
        },
    );
    let ctx = PropagationContext::default();
    for position in [[f64::NAN, 0.0, 0.0], [0.0; 3]] {
        let state = CartesianState::new(f64::NAN, position, [f64::INFINITY; 3]);
        assert_eq!(model.acceleration(&state, &ctx).unwrap(), Vector3::zeros());
        assert_eq!(
            model
                .acceleration_with_parameters(&[], &state, &ctx)
                .unwrap(),
            Vector3::zeros()
        );
        assert_eq!(
            model.acceleration_partials(&[], &state, &ctx).unwrap(),
            AccelerationPartials::zeros()
        );
    }
}

#[test]
fn spherical_harmonic_partials_require_both_central_samples() {
    let model = SphericalHarmonicGravity::from_normalized_coefficients(
        4.0e5,
        6_378.0,
        2,
        0,
        &[SphericalHarmonicCoefficient {
            degree: 2,
            order: 0,
            c: -0.001,
            s: 0.0,
        }],
        TideSystem::TideFree,
    )
    .unwrap();
    let ctx = PropagationContext::default();
    for x in [f64::MAX, -f64::MAX] {
        let state = CartesianState::new(0.0, [x, 0.0, 0.0], [0.0; 3]);
        assert!(matches!(
            model.acceleration_partials(&[], &state, &ctx),
            Err(PropagationError::NumericalFailure(message)) if message == "state too large for difference quotients"
        ));
    }
}

#[test]
fn earth_radiation_vanishes_when_the_apparent_disk_weight_rounds_to_zero() {
    let model = EarthRadiationPressure::new(1.2, 0.011).unwrap();
    let state = CartesianState::new(0.0, [2.0e154, 0.0, 0.0], [0.0; 3]);
    let ctx = PropagationContext::default();
    assert_eq!(
        model.acceleration(&state, &ctx).unwrap().map(f64::to_bits),
        Vector3::zeros()
    );
}

#[test]
fn empirical_frame_recovers_an_overflowing_norm() {
    let model = EmpiricalRtnAcceleration::constant_terms();
    let values = [1.0, 2.0, 3.0];
    let ctx = PropagationContext::default();
    let base = CartesianState::new(0.0, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    let large = CartesianState::new(0.0, [pow2(600), 0.0, 0.0], [0.0, pow2(600), 0.0]);
    assert_eq!(
        model
            .acceleration_with_parameters(&values, &large, &ctx)
            .unwrap(),
        Vector3::from(values)
    );
    assert_eq!(
        model.parameter_partials(&values, &large, &ctx).unwrap(),
        model.parameter_partials(&values, &base, &ctx).unwrap()
    );
    let reference = model.acceleration_partials(&values, &base, &ctx).unwrap();
    let actual = model.acceleration_partials(&values, &large, &ctx).unwrap();
    assert_eq!(actual.wrt_position, reference.wrt_position * pow2(-600));
    assert_eq!(actual.wrt_velocity, reference.wrt_velocity * pow2(-600));
}

#[test]
fn empirical_frame_recovers_underflowing_momentum_and_norms() {
    let model = EmpiricalRtnAcceleration::constant_terms();
    let values = [1.0, 2.0, 3.0];
    let ctx = PropagationContext::default();
    for (reference_radius, position_scale, velocity_scale) in [
        (1.0, pow2(-600), pow2(-600)),
        (1.0, pow2(-500), pow2(-600)),
        (7000.0, 1.0, pow2(-1000)),
    ] {
        let reference = CartesianState::new(0.0, [reference_radius, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let state = CartesianState::new(
            0.0,
            [reference_radius * position_scale, 0.0, 0.0],
            [0.0, velocity_scale, 0.0],
        );
        assert_eq!(model.acceleration(&state, &ctx).unwrap(), Vector3::zeros());
        assert_eq!(
            model
                .acceleration_with_parameters(&values, &state, &ctx)
                .unwrap(),
            Vector3::from(values)
        );
        assert_eq!(
            model.parameter_partials(&values, &state, &ctx).unwrap(),
            model.parameter_partials(&values, &reference, &ctx).unwrap()
        );
        let expected = model
            .acceleration_partials(&values, &reference, &ctx)
            .unwrap();
        let actual = model.acceleration_partials(&values, &state, &ctx).unwrap();
        assert_eq!(actual.wrt_position, expected.wrt_position / position_scale);
        assert_eq!(actual.wrt_velocity, expected.wrt_velocity / velocity_scale);
    }
}

#[test]
fn empirical_radial_constant_ignores_velocity() {
    let ctx = PropagationContext::default();
    for (position, coefficient) in [
        (Vector3::<f64>::new(7000.0, 0.0, 0.0), 1.0e-8),
        (Vector3::new(3.0, -4.0, -0.0), f64::from_bits(7)),
    ] {
        let model = EmpiricalRtnAcceleration::new()
            .with_term(RtnAxis::Radial, EmpiricalHarmonic::Constant, coefficient)
            .unwrap();
        let radius = position.norm();
        let radial = position / radius;
        let acceleration = Vector3::zeros() + radial * coefficient;
        let gradient = Matrix3::zeros()
            + (Matrix3::identity() - radial * radial.transpose()) / radius * coefficient;
        for velocity in [
            [0.0; 3],
            [1.0, 0.0, 0.0],
            position.into(),
            [f64::NAN, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ] {
            let state = CartesianState::new(0.0, position.into(), velocity);
            assert_eq!(
                model.acceleration(&state, &ctx).unwrap().map(f64::to_bits),
                acceleration.map(f64::to_bits)
            );
            assert_eq!(
                model
                    .acceleration_with_parameters(&[coefficient], &state, &ctx)
                    .unwrap()
                    .map(f64::to_bits),
                acceleration.map(f64::to_bits)
            );
            let partials = model
                .acceleration_partials(&[coefficient], &state, &ctx)
                .unwrap();
            assert_eq!(
                partials.wrt_position.map(f64::to_bits),
                gradient.map(f64::to_bits)
            );
            assert_eq!(partials.wrt_velocity.map(f64::to_bits), Matrix3::zeros());
            assert_eq!(
                model
                    .parameter_partials(&[coefficient], &state, &ctx)
                    .unwrap()[0]
                    .map(f64::to_bits),
                radial.map(f64::to_bits)
            );
        }
    }
}

struct ConstantForce(f64);

impl ForceModel for ConstantForce {
    fn acceleration(
        &self,
        _: &CartesianState,
        _: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        Ok(Vector3::repeat(self.0))
    }

    fn acceleration_partials(
        &self,
        _: &[f64],
        _: &CartesianState,
        _: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        Ok(AccelerationPartials::position_only(Matrix3::repeat(self.0)))
    }

    fn acceleration_with_parameters(
        &self,
        _: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.acceleration(state, ctx)
    }

    fn parameter_count(&self) -> usize {
        1
    }

    fn parameter_partials(
        &self,
        _: &[f64],
        _: &CartesianState,
        _: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        Ok(vec![Vector3::repeat(self.0)])
    }
}

#[test]
fn composite_and_forwarders_enforce_finite_results() {
    let ctx = PropagationContext::default();
    let state = state();
    let model = CompositeForceModel {
        models: vec![
            Box::new(ConstantForce(f64::MAX)),
            Box::new(ConstantForce(f64::MAX)),
            Box::new(ConstantForce(-f64::MAX)),
        ],
    };
    assert_eq!(
        model.acceleration(&state, &ctx).unwrap(),
        Vector3::repeat(f64::MAX)
    );
    assert_eq!(
        model
            .acceleration_with_parameters(&[1.0; 3], &state, &ctx)
            .unwrap(),
        Vector3::repeat(f64::MAX)
    );
    assert_eq!(
        model
            .acceleration_partials(&[1.0; 3], &state, &ctx)
            .unwrap()
            .wrt_position,
        Matrix3::repeat(f64::MAX)
    );

    for model in [
        &Arc::new(ConstantForce(f64::NAN)) as &dyn ForceModel,
        &CompositeForceModel {
            models: vec![Box::new(ConstantForce(f64::NAN))],
        },
    ] {
        assert!(matches!(
            model.acceleration(&state, &ctx),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            model.acceleration_with_parameters(&[1.0], &state, &ctx),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            model.acceleration_partials(&[1.0], &state, &ctx),
            Err(PropagationError::NumericalFailure(_))
        ));
        assert!(matches!(
            model.parameter_partials(&[1.0], &state, &ctx),
            Err(PropagationError::NumericalFailure(_))
        ));
    }
    let mut bad_position = state;
    bad_position.position_km.fill(f64::NAN);
    for model in [
        &Arc::new(TwoBodyGravity::default()) as &dyn ForceModel,
        &CompositeForceModel {
            models: vec![Box::new(TwoBodyGravity::default())],
        },
    ] {
        invalid_state(model, &bad_position, &ctx);
    }
}

#[test]
fn spherical_harmonics_recover_a_radius_whose_square_underflows() {
    let model = SphericalHarmonicGravity::from_normalized_coefficients(
        pow2(-900),
        pow2(-250),
        2,
        0,
        &[SphericalHarmonicCoefficient {
            degree: 2,
            order: 0,
            c: 1.0,
            s: 0.0,
        }],
        TideSystem::TideFree,
    )
    .unwrap();
    let ctx = PropagationContext::default();
    let normal = model.acceleration(&axial(-250), &ctx).unwrap();
    let actual = model.acceleration(&axial(-550), &ctx).unwrap();
    assert_ne!(normal.x, 0.0);
    assert_eq!(actual.x.to_bits(), libm::ldexp(normal.x, 1200).to_bits());
}

#[test]
fn earth_radiation_recovers_an_overflowing_spacecraft_scale() {
    let ctx = PropagationContext::default();
    let normal = EarthRadiationPressure::new(1.0, 1.0).unwrap();
    let large = EarthRadiationPressure::new(pow2(520), pow2(520)).unwrap();
    let state = state();
    let reference = normal.acceleration(&state, &ctx).unwrap();
    assert_ne!(reference, Vector3::zeros());
    let actual = large.acceleration(&state, &ctx).unwrap();
    assert_eq!(
        actual.map(f64::to_bits),
        reference.map(|value| libm::ldexp(value, 1040).to_bits())
    );
}

#[test]
fn a_failed_body_evaluation_returns_the_epoch_error() {
    let mut state = state();
    state.epoch_tdb_seconds = f64::MAX;
    let ctx = PropagationContext::default();
    let srp = SolarRadiationPressure::new(1.3, 0.02).unwrap();
    let earth = EarthRadiationPressure::new(1.3, 0.02).unwrap();
    let ecom1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients::default()).unwrap();
    let ecom2 = Ecom2SolarRadiationPressure::new(0, 0, vec![0.0; 3]).unwrap();
    for model in [
        &ThirdBodyGravity::sun() as &dyn ForceModel,
        &srp,
        &earth,
        &ecom1,
        &ecom2,
    ] {
        assert!(matches!(
            model.acceleration(&state, &ctx),
            Err(PropagationError::ForceModelFailure(_))
        ));
    }
}

#[test]
fn tide_results_recover_extreme_radius_and_coefficient_powers() {
    let ctx = context();
    let mut nominal = axial(-250);
    nominal.epoch_tdb_seconds = 0.0;
    let mut tiny = nominal;
    tiny.position_km.x = pow2(-550);
    let pole = SolidEarthPoleTideGravity::new(pow2(-900), pow2(-250));
    let reference = pole.acceleration(&nominal, &ctx).unwrap();
    let actual = pole.acceleration(&tiny, &ctx).unwrap();
    assert_ne!(reference, Vector3::zeros());
    assert_eq!(
        actual.map(f64::to_bits),
        reference.map(|value| libm::ldexp(value, 1200).to_bits())
    );

    let mut tide = SolidEarthTideGravity {
        frequency_dependent: false,
        mu_earth_km3_s2: pow2(-740),
        gm_sun_km3_s2: pow2(225),
        gm_moon_km3_s2: pow2(225),
        ..SolidEarthTideGravity::default()
    };
    let state = state();
    let reference = tide.acceleration(&state, &ctx).unwrap();
    tide.mu_earth_km3_s2 = pow2(-1000);
    assert!(tide
        .coefficient_corrections_at_epoch(state.epoch_tdb_seconds, &ctx)
        .is_err());
    let actual = tide.acceleration(&state, &ctx).unwrap();
    for (expected, value) in reference.iter().zip(&actual) {
        assert_ne!(*expected, 0.0);
        assert!(
            (value / expected - 1.0).abs() <= 8.0 * f64::EPSILON,
            "{value} vs {expected}"
        );
    }
}
