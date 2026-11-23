#![no_main]

use std::sync::Arc;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::events::eclipse::EarthShadowModel;
use sidereon_core::astro::forces::{
    CompositeForceModel, EmpiricalHarmonic, EmpiricalRtnAcceleration, ForceModel, RtnAxis,
    SolarRadiationPressure, ThirdBodyGravity, TwoBodyGravity, ZonalGravity,
};
use sidereon_core::astro::propagator::{
    propagate_variational, ForceModelComponents, ForceModelKind, IntegratorKind,
    IntegratorOptions, PropagationContext, VariationalEpoch, VariationalEquations,
};
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::time::model::{Instant, TimeScale};
use sidereon_core::orbit_determination::{
    fit_inertial_orbit_parameters, InertialPositionSample, OrbitParameterCovariance,
    OrbitParameterFitOptions,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

#[derive(Debug, Arbitrary)]
struct Input {
    epoch_s: f64,
    position: [f64; 3],
    velocity: [f64; 3],
    term_bits: u16,
    coefficients: [f64; 9],
    cr: f64,
    area_to_mass: f64,
    span_s: f64,
    integrator: bool,
    options: [f64; 3],
    selection_bits: u16,
    samples: Vec<([f64; 3], i16)>,
    fit_options: [f64; 3],
    iterations: u8,
}

fn finite3(values: impl IntoIterator<Item = f64>) -> bool {
    values.into_iter().all(f64::is_finite)
}

/// Bit `index` of `bits`, false past its width.
fn bit(bits: u16, index: usize) -> bool {
    u32::try_from(index)
        .ok()
        .and_then(|shift| bits.checked_shr(shift))
        .is_some_and(|value| value & 1 == 1)
}

// Force-model partials, variational propagation and the force-parameter fit
// on arbitrary states, parameters, options and observations: finite results
// or typed errors, never a panic, a non-finite output or a long run.
fuzz_target!(|input: Input| {
    let state = CartesianState::new(input.epoch_s, input.position, input.velocity);
    let ctx = PropagationContext::default();

    let mut empirical = EmpiricalRtnAcceleration::new();
    for (index, (axis, harmonic)) in RtnAxis::ALL
        .into_iter()
        .flat_map(|axis| EmpiricalHarmonic::ALL.into_iter().map(move |h| (axis, h)))
        .enumerate()
    {
        if bit(input.term_bits, index) {
            if let Ok(model) = empirical.with_term(axis, harmonic, input.coefficients[index]) {
                empirical = model;
            }
        }
    }
    // Bits 9 and 10, past the nine empirical terms, select the Moon's shadow
    // and the oblate Earth.
    let earth_shadow_model = if bit(input.term_bits, 10) {
        EarthShadowModel::Wgs84Oblate
    } else {
        EarthShadowModel::Spherical
    };
    let srp = SolarRadiationPressure::new(input.cr, input.area_to_mass)
        .ok()
        .map(|srp| {
            srp.with_moon_eclipse(bit(input.term_bits, 9))
                .with_earth_shadow_model(earth_shadow_model)
        });

    // Partials of every closed-form model at caller-supplied parameters.
    let mut models: Vec<Box<dyn ForceModel>> = vec![
        Box::new(TwoBodyGravity::default()),
        Box::new(ZonalGravity::earth_j2_through_j6()),
        Box::new(ThirdBodyGravity::default()),
        Box::new(empirical),
    ];
    if let Some(srp) = srp {
        models.push(Box::new(srp));
    }
    for model in &models {
        if let Ok(acceleration) = model.acceleration(&state, &ctx) {
            assert!(finite3(acceleration.iter().copied()), "{acceleration:?}");
        }
        let parameters: Vec<f64> = (0..model.parameter_count())
            .map(|index| input.coefficients[index % 9] * if index == 0 { 1.0e12 } else { 1.0 })
            .collect();
        if let Ok(acceleration) = model.acceleration_with_parameters(&parameters, &state, &ctx) {
            assert!(finite3(acceleration.iter().copied()), "{acceleration:?}");
        }
        if let Ok(partials) = model.acceleration_partials(&parameters, &state, &ctx) {
            assert!(finite3(partials.wrt_position.iter().copied()), "{partials:?}");
            assert!(finite3(partials.wrt_velocity.iter().copied()), "{partials:?}");
        }
        if let Ok(partials) = model.parameter_partials(&parameters, &state, &ctx) {
            assert_eq!(partials.len(), parameters.len());
            assert!(partials.iter().all(|p| finite3(p.iter().copied())), "{partials:?}");
        }
    }

    // Variational propagation over a bounded number of steps.
    let mut composite = CompositeForceModel::new();
    composite.add(Box::new(TwoBodyGravity::default()));
    composite.add(Box::new(empirical));
    let values: Vec<f64> = composite.parameters().iter().map(|p| p.value).collect();
    let selected: Vec<usize> = (0..values.len())
        .filter(|&index| bit(input.selection_bits, index))
        .collect();
    let mut options = IntegratorOptions::default();
    options.abs_tol = input.options[0];
    options.rel_tol = input.options[1];
    options.initial_step = input.options[2];
    options.max_steps = 64;
    let integrator = if input.integrator {
        IntegratorKind::Rk4
    } else {
        IntegratorKind::Dp54
    };
    let equations = VariationalEquations {
        force: &composite,
        parameters: &values,
        sensitivity_parameters: &selected,
    };
    // An epoch from the TDB seconds, and the same TDB with the span taken as
    // the elapsed TT, which the propagator refuses when the two disagree.
    let epochs = [
        VariationalEpoch::from_tdb_seconds(input.epoch_s, input.epoch_s + input.span_s),
        VariationalEpoch {
            tdb_j2000_s: input.epoch_s + input.span_s,
            elapsed_tt_s: input.span_s,
        },
    ];
    if let Ok(states) = propagate_variational(
        state,
        &epochs,
        &equations,
        integrator,
        &options,
        &ctx,
    ) {
        for propagated in &states {
            assert!(finite3(propagated.state.position_km.iter().copied()));
            assert!(finite3(propagated.state.velocity_km_s.iter().copied()));
            assert!(finite3(propagated.state_transition.iter().copied()));
            assert!(finite3(propagated.sensitivity.iter().copied()));
        }
    }

    // A fit of a few arbitrary observations with bounded work.
    let samples: Vec<InertialPositionSample> = input
        .samples
        .iter()
        .take(6)
        .map(|(position, offset)| InertialPositionSample {
            epoch: Instant::from_nanos(
                TimeScale::Tt,
                (831_902_400 + i128::from(*offset) * 60) * 1_000_000_000,
            ),
            position_gcrf_m: *position,
        })
        .collect();
    let mut fit_options = OrbitParameterFitOptions::default();
    fit_options.force_model = ForceModelKind::composite(ForceModelComponents {
        two_body_mu_km3_s2: Some(398_600.4418),
        solar_radiation_pressure: srp,
        ..ForceModelComponents::default()
    });
    fit_options.additional_forces = vec![Arc::new(empirical)];
    fit_options.estimated_parameters = composite
        .parameters()
        .into_iter()
        .zip(0_usize..)
        .filter(|&(_, index)| bit(input.selection_bits, index + 7))
        .map(|(parameter, _)| parameter.name)
        .collect();
    if srp.is_some() && bit(input.selection_bits, 0) {
        fit_options
            .estimated_parameters
            .push(SolarRadiationPressure::CR_PARAMETER.to_string());
    }
    fit_options.observation_sigma_m = input.fit_options[0];
    fit_options.convergence_threshold = input.fit_options[1];
    fit_options.max_condition_number = input.fit_options[2];
    fit_options.max_iterations = usize::from(input.iterations % 4);
    fit_options.integrator_options.max_steps = 200;
    let satellite = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite");
    if let Ok(fit) = fit_inertial_orbit_parameters(&samples, satellite, None, &fit_options) {
        assert!(finite3(fit.initial_state.position_km.iter().copied()));
        assert!(finite3(fit.initial_state.velocity_km_s.iter().copied()));
        assert!(fit.force_parameters.iter().all(|p| p.value.is_finite()));
        assert!(fit.residual_stats.rms_3d_m.is_finite());
        assert!(fit
            .residuals
            .iter()
            .all(|r| finite3([r.radial_m, r.along_m, r.cross_m, r.elapsed_tt_s])));
        if let OrbitParameterCovariance::Estimated {
            matrix,
            correlation,
            variance_factor,
            ..
        } = &fit.covariance
        {
            assert!(variance_factor.is_finite());
            assert!(matrix.iter().all(|value| value.is_finite()));
            assert!(correlation.iter().all(|value| value.is_finite()));
        }
    }
});
