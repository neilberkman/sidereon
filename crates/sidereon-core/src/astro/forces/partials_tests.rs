//! Closed-form acceleration partials against central difference quotients.
//!
//! Every model with a closed-form [`ForceModel::acceleration_partials`] is
//! differentiated numerically here with a step chosen for it, on states that
//! cover a GNSS orbit, a low orbit at high latitude and an eccentric orbit
//! near perigee. The variational equations take these blocks (Montenbruck &
//! Gill, "Satellite Orbits", section 7.3). Solar radiation pressure and the
//! empirical accelerations are checked in their own modules.

use super::r#trait::{difference_quotient_partials, AccelerationPartials, ForceModel};
use super::{
    CompositeForceModel, Ecom1Coefficients, Ecom1SolarRadiationPressure, Ecom2D4b1Coefficients,
    Ecom2SolarRadiationPressure, EmpiricalHarmonic, EmpiricalRtnAcceleration, J2Gravity, RtnAxis,
    SchwarzschildRelativity, SolarRadiationPressure, ThirdBodyBodies, ThirdBodyGravity,
    TwoBodyGravity, ZonalGravity,
};
use crate::astro::error::PropagationError;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};
use std::sync::Arc;

/// 2026-05-13 00:00 TT, in seconds past J2000.
const EPOCH: f64 = 831_902_400.0;

fn states() -> [CartesianState; 3] {
    [
        // GNSS: 26560 km, 55 degrees.
        CartesianState::new(EPOCH, [15_934.0, 21_245.0, 1_200.0], [-2.235, 1.680, 3.170]),
        // Low orbit over high latitude.
        CartesianState::new(
            EPOCH + 3_600.0,
            [1_200.0, -900.0, 6_850.0],
            [6.9, 3.1, -0.8],
        ),
        // Eccentric orbit near a 7000 km perigee.
        CartesianState::new(
            EPOCH + 7_200.0,
            [-4_100.0, 5_300.0, 2_600.0],
            [-6.2, -5.8, 4.1],
        ),
    ]
}

/// Central difference quotients with explicit position and velocity steps.
fn quotients(
    model: &dyn ForceModel,
    state: &CartesianState,
    position_step_km: f64,
    velocity_step_km_s: f64,
) -> AccelerationPartials {
    let ctx = PropagationContext::default();
    let accel = |state: &CartesianState| model.acceleration(state, &ctx).expect("acceleration");
    let mut partials = AccelerationPartials::zeros();
    for axis in 0..3 {
        let mut plus = *state;
        let mut minus = *state;
        plus.position_km[axis] += position_step_km;
        minus.position_km[axis] -= position_step_km;
        partials.wrt_position.set_column(
            axis,
            &((accel(&plus) - accel(&minus)) / (plus.position_km[axis] - minus.position_km[axis])),
        );
        let mut plus = *state;
        let mut minus = *state;
        plus.velocity_km_s[axis] += velocity_step_km_s;
        minus.velocity_km_s[axis] -= velocity_step_km_s;
        partials.wrt_velocity.set_column(
            axis,
            &((accel(&plus) - accel(&minus))
                / (plus.velocity_km_s[axis] - minus.velocity_km_s[axis])),
        );
    }
    partials
}

fn assert_close(label: &str, analytic: &Matrix3<f64>, numeric: &Matrix3<f64>, tolerance: f64) {
    let scale = analytic.norm();
    assert!(
        (analytic - numeric).norm() <= tolerance * scale,
        "{label}: analytic {analytic:?} vs quotients {numeric:?}"
    );
}

/// Position steps of 1e-5 of the radius and velocity steps of 1e-5 of the
/// speed: for an acceleration varying on the scale of the radius the
/// truncation error is (1e-5)^2 = 1e-10 relative and the rounding error
/// eps / 1e-5 = 2e-11 relative; 1e-8 covers both.
#[test]
fn central_gravity_zonals_and_relativity_match_difference_quotients() {
    let ctx = PropagationContext::default();
    let models: [(&str, Box<dyn ForceModel>); 4] = [
        ("two-body", Box::new(TwoBodyGravity::default())),
        ("J2", Box::new(J2Gravity::default())),
        ("J2-J6", Box::new(ZonalGravity::earth_j2_through_j6())),
        ("relativity", Box::new(SchwarzschildRelativity::default())),
    ];
    for state in states() {
        let position_step = state.position_km.norm() * 1.0e-5;
        let velocity_step = state.velocity_km_s.norm() * 1.0e-5;
        for (name, model) in &models {
            let analytic = model
                .acceleration_partials(&[], &state, &ctx)
                .expect("partials");
            let numeric = quotients(model.as_ref(), &state, position_step, velocity_step);
            assert_close(name, &analytic.wrt_position, &numeric.wrt_position, 1.0e-8);
            if *name == "relativity" {
                assert_close(name, &analytic.wrt_velocity, &numeric.wrt_velocity, 1.0e-8);
            } else {
                assert_eq!(analytic.wrt_velocity, Matrix3::zeros(), "{name}");
            }
        }
    }
}

/// The zonal gradient is the Hessian of a potential that is harmonic outside
/// the Earth: symmetric and trace-free.
#[test]
fn the_zonal_gradient_is_symmetric_and_trace_free() {
    let ctx = PropagationContext::default();
    let model = ZonalGravity::earth_j2_through_j6();
    for state in states() {
        let gradient = model
            .acceleration_partials(&[], &state, &ctx)
            .expect("partials")
            .wrt_position;
        let scale = gradient.norm();
        // Rounding of the nine entries: a few ulps of the largest.
        assert!((gradient - gradient.transpose()).norm() <= 1.0e-14 * scale);
        assert!(gradient.trace().abs() <= 1.0e-14 * scale);
    }
}

/// A 10 km step: the bodies are 3.8e5 km and 1.5e8 km away, so the
/// truncation error is at most (10 / 3.6e5)^2 = 8e-10 relative. The tidal
/// gradient is 8.7e-14 1/s^2 for the Moon and 3.9e-14 1/s^2 for the Sun; the
/// rounding of the direct and indirect terms (3.3e-8 and 5.9e-9 km/s^2, a few
/// ulps each) divided by the 20 km of the quotient is below 1e-11 of it.
#[test]
fn third_body_partials_match_difference_quotients() {
    let ctx = PropagationContext::default();
    for bodies in [
        ThirdBodyBodies {
            sun: true,
            moon: false,
        },
        ThirdBodyBodies {
            sun: false,
            moon: true,
        },
    ] {
        let model = ThirdBodyGravity {
            bodies,
            ..ThirdBodyGravity::default()
        };
        for state in states() {
            let analytic = model
                .acceleration_partials(&[], &state, &ctx)
                .expect("partials");
            let numeric = quotients(&model, &state, 10.0, 1.0e-3);
            assert_close(
                "third body",
                &analytic.wrt_position,
                &numeric.wrt_position,
                1.0e-8,
            );
            assert_eq!(analytic.wrt_velocity, Matrix3::zeros());
        }
    }
}

/// `a = -k r`: the difference quotients of a linear field are exact up to
/// rounding, so the trait's default partials must return `-k I`.
struct Spring;

impl ForceModel for Spring {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        Ok(state.position_km * -2.5e-6)
    }
}

#[test]
fn the_default_partials_are_difference_quotients() {
    let ctx = PropagationContext::default();
    for state in states() {
        let partials = Spring
            .acceleration_partials(&[], &state, &ctx)
            .expect("default partials");
        assert_eq!(
            partials,
            difference_quotient_partials(&Spring, &[], &state, &ctx).expect("quotients")
        );
        // A difference of two accelerations of order 2.5e-6 * 26560 km over a
        // step of 0.27 km: rounding below 1e-10 of the 2.5e-6 slope.
        assert!((partials.wrt_position + Matrix3::identity() * 2.5e-6).norm() <= 1.0e-15);
        assert_eq!(partials.wrt_velocity, Matrix3::zeros());
    }
    let mut bad = states()[0];
    bad.position_km.x = f64::NAN;
    assert!(matches!(
        difference_quotient_partials(&Spring, &[], &bad, &ctx),
        Err(PropagationError::InvalidInput(_))
    ));
}

#[test]
fn a_composite_concatenates_parameters_and_sums_partials() {
    let ctx = PropagationContext::default();
    let srp = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let empirical = EmpiricalRtnAcceleration::with_zero_terms(&[
        (RtnAxis::AlongTrack, EmpiricalHarmonic::Constant),
        (RtnAxis::CrossTrack, EmpiricalHarmonic::Sine),
    ]);
    let mut composite = CompositeForceModel::new();
    composite.add(Box::new(TwoBodyGravity::default()));
    composite.add(Box::new(srp));
    composite.add(Box::new(Arc::new(empirical)));
    let names: Vec<String> = composite.parameters().into_iter().map(|p| p.name).collect();
    assert_eq!(
        names,
        ["srp.cr", "empirical.along.constant", "empirical.cross.sin"]
    );
    assert_eq!(composite.parameter_count(), 3);

    let parameters = [1.1, 2.0e-12, -1.0e-12];
    let state = states()[0];
    let partials = composite
        .acceleration_partials(&parameters, &state, &ctx)
        .expect("partials");
    let mut expected = TwoBodyGravity::default()
        .acceleration_partials(&[], &state, &ctx)
        .expect("two-body");
    for (model, own) in [
        (&srp as &dyn ForceModel, &parameters[..1]),
        (&empirical, &parameters[1..]),
    ] {
        let component = model
            .acceleration_partials(own, &state, &ctx)
            .expect("component");
        expected.wrt_position += component.wrt_position;
        expected.wrt_velocity += component.wrt_velocity;
    }
    assert_eq!(partials, expected);

    let parameter_partials = composite
        .parameter_partials(&parameters, &state, &ctx)
        .expect("parameter partials");
    let mut expected = srp
        .parameter_partials(&parameters[..1], &state, &ctx)
        .expect("SRP");
    expected.extend(
        empirical
            .parameter_partials(&parameters[1..], &state, &ctx)
            .expect("empirical"),
    );
    assert_eq!(parameter_partials, expected);

    let acceleration = composite
        .acceleration_with_parameters(&parameters, &state, &ctx)
        .expect("acceleration");
    let expected = TwoBodyGravity::default()
        .acceleration(&state, &ctx)
        .expect("two-body")
        + srp
            .acceleration_with_parameters(&parameters[..1], &state, &ctx)
            .expect("SRP")
        + empirical
            .acceleration_with_parameters(&parameters[1..], &state, &ctx)
            .expect("empirical");
    assert_eq!(acceleration, expected);

    for wrong in [&parameters[..2], &[1.0, 0.0, 0.0, 0.0][..]] {
        assert!(matches!(
            composite.acceleration_with_parameters(wrong, &state, &ctx),
            Err(PropagationError::InvalidInput(_))
        ));
        assert!(matches!(
            composite.parameter_partials(wrong, &state, &ctx),
            Err(PropagationError::InvalidInput(_))
        ));
    }
    // Models without parameters refuse a non-empty slice.
    assert!(matches!(
        TwoBodyGravity::default().acceleration_partials(&[1.0], &state, &ctx),
        Err(PropagationError::InvalidInput(_))
    ));
}

#[test]
fn non_finite_components_are_refused_when_used_by_partials() {
    let ctx = PropagationContext::default();
    let srp = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
    let empirical = EmpiricalRtnAcceleration::all_terms();
    let ecom2 = Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients::default()).expect("ECOM2");
    let ecom1 = Ecom1SolarRadiationPressure::new(Ecom1Coefficients::default()).expect("ECOM1");
    let models: [(&dyn ForceModel, usize); 10] = [
        (&TwoBodyGravity::default(), 0),
        (&J2Gravity::default(), 0),
        (&ZonalGravity::earth_j2_through_j6(), 0),
        (&ThirdBodyGravity::default(), 0),
        (&SchwarzschildRelativity::default(), 0),
        (&srp, 1),
        (&empirical, 9),
        (&ecom2, 9),
        (&ecom1, 9),
        (&Spring, 0),
    ];
    for component in 0..6 {
        let mut state = states()[0];
        if component < 3 {
            state.position_km[component] = f64::NAN;
        } else {
            state.velocity_km_s[component - 3] = f64::INFINITY;
        }
        for (index, (model, count)) in models.iter().enumerate() {
            let parameters = vec![1.0e-12; *count];
            let result = model.acceleration_partials(&parameters, &state, &ctx);
            if component >= 3 && [0, 1, 2, 3, 5].contains(&index) {
                assert_eq!(
                    result.unwrap(),
                    model
                        .acceleration_partials(&parameters, &states()[0], &ctx)
                        .unwrap()
                );
            } else {
                assert!(
                    matches!(result, Err(PropagationError::InvalidInput(_))),
                    "model {index}, component {component}: {result:?}"
                );
            }
        }
    }
}

/// A component with fixed outputs: one parameter, an acceleration, a
/// position partial in every entry and a chosen number of parameter
/// partials.
struct Fixed {
    acceleration: f64,
    partial: f64,
    parameter_partials: usize,
}

impl ForceModel for Fixed {
    fn acceleration(
        &self,
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        Ok(Vector3::new(self.acceleration, 0.0, 0.0))
    }

    fn parameters(&self) -> Vec<super::ForceParameter> {
        vec![super::ForceParameter::new("test.fixed", 0.0)]
    }

    fn acceleration_with_parameters(
        &self,
        _parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.acceleration(state, ctx)
    }

    fn acceleration_partials(
        &self,
        _parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        Ok(AccelerationPartials::position_only(Matrix3::from_element(
            self.partial,
        )))
    }

    fn parameter_partials(
        &self,
        _parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        Ok(vec![Vector3::zeros(); self.parameter_partials])
    }
}

#[test]
fn a_composite_sum_that_overflows_is_a_numerical_failure() {
    let large = || {
        Box::new(Fixed {
            acceleration: 1.0e308,
            partial: 1.0e308,
            parameter_partials: 1,
        })
    };
    let mut composite = CompositeForceModel::new();
    composite.add(large());
    composite.add(large());
    let ctx = PropagationContext::default();
    let state = states()[0];
    let failed = |result: Result<Vector3<f64>, PropagationError>| {
        matches!(result, Err(PropagationError::NumericalFailure(_)))
    };
    assert!(failed(composite.acceleration(&state, &ctx)));
    assert!(failed(composite.acceleration_with_parameters(
        &[0.0, 0.0],
        &state,
        &ctx
    )));
    let partials = composite.acceleration_partials(&[0.0, 0.0], &state, &ctx);
    assert!(
        matches!(partials, Err(PropagationError::NumericalFailure(_))),
        "{partials:?}"
    );
}

#[test]
fn a_component_returning_the_wrong_number_of_parameter_partials_is_named() {
    let mut composite = CompositeForceModel::new();
    composite.add(Box::new(Fixed {
        acceleration: 0.0,
        partial: 0.0,
        parameter_partials: 2,
    }));
    let result = composite.parameter_partials(&[0.0], &states()[0], &PropagationContext::default());
    match result {
        Err(PropagationError::InvalidInput(message)) => assert!(
            message.contains("returned 2 parameter partials for 1 parameter"),
            "{message}"
        ),
        other => panic!("expected invalid input, got {other:?}"),
    }
}
