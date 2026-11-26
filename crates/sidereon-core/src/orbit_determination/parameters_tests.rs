//! Synthetic checks of the force-parameter fit: noise-free observations made
//! by propagating known parameters are fitted back from a displaced guess,
//! and every refusal is a typed error. The agreement with an independent
//! estimator is in `tests/orbit_parameter_fit_orekit.rs`.

use super::*;
use crate::astro::constants::MU_EARTH;
use crate::astro::error::PropagationError;
use crate::astro::forces::{
    AccelerationPartials, CompositeForceModel, EmpiricalHarmonic, EmpiricalRtnAcceleration,
    RtnAxis, SolarRadiationPressure,
};
use crate::astro::propagator::ForceModelComponents;
use crate::astro::time::model::TimeScale;
use crate::GnssSystem;

/// 2026-05-13 00:00 TT, in seconds past J2000.
const EPOCH_TT: f64 = 831_902_400.0;
/// Observation spacing and count: 12 hours, one GPS revolution.
const STEP_S: i128 = 900;
const SAMPLES: i128 = 49;
const AREA_TO_MASS: f64 = 0.02;

fn satellite() -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, 7).expect("valid satellite")
}

fn tight_integrator() -> IntegratorOptions {
    IntegratorOptions {
        abs_tol: 1.0e-12,
        rel_tol: 1.0e-13,
        initial_step: 30.0,
        max_step: 120.0,
        ..IntegratorOptions::default()
    }
}

/// A GPS-like state: 26560 km, 55 degrees inclination, slightly eccentric.
fn truth_state() -> CartesianState {
    let radius = 26_560.0_f64;
    let speed = (MU_EARTH / radius).sqrt() * 1.003;
    let (sin_i, cos_i) = libm::sincos(55.0_f64.to_radians());
    CartesianState::new(
        EPOCH_TT,
        [radius * 0.6, radius * 0.8, 0.0],
        [
            -speed * 0.8 * cos_i + 0.01,
            speed * 0.6 * cos_i,
            speed * sin_i,
        ],
    )
}

fn truth_empirical() -> EmpiricalRtnAcceleration {
    EmpiricalRtnAcceleration::new()
        .with_term(RtnAxis::Radial, EmpiricalHarmonic::Constant, 1.5e-12)
        .and_then(|model| {
            model.with_term(RtnAxis::AlongTrack, EmpiricalHarmonic::Constant, -2.0e-12)
        })
        .and_then(|model| model.with_term(RtnAxis::CrossTrack, EmpiricalHarmonic::Cosine, 3.0e-12))
        .and_then(|model| model.with_term(RtnAxis::CrossTrack, EmpiricalHarmonic::Sine, -1.0e-12))
        .expect("finite")
}

fn force_kind(cr: f64) -> ForceModelKind {
    ForceModelKind::composite(ForceModelComponents::earth_phase_a(Some(
        SolarRadiationPressure::new(cr, AREA_TO_MASS).expect("valid SRP"),
    )))
}

fn tt_instant(seconds_past_j2000: i128) -> Instant {
    Instant::from_nanos(TimeScale::Tt, seconds_past_j2000 * 1_000_000_000)
}

/// The epochs of `instants` as propagation epochs from the first, as the fit
/// forms them: each instant's TDB, and its TT less the first one's.
fn propagation_epochs(instants: &[Instant]) -> Vec<VariationalEpoch> {
    let seconds: Vec<(f64, f64)> = instants
        .iter()
        .map(|&instant| dynamical_seconds_from_instant(instant, satellite()).expect("epoch"))
        .collect();
    seconds
        .iter()
        .map(|&(tdb, tt)| VariationalEpoch {
            tdb_j2000_s: tdb,
            elapsed_tt_s: tt - seconds[0].1,
        })
        .collect()
}

/// Positions from the truth model at the observation epochs.
fn synthetic_samples(cr: f64, empirical: EmpiricalRtnAcceleration) -> Vec<InertialPositionSample> {
    let mut composite = CompositeForceModel::new();
    composite.add(
        StatePropagator {
            initial: truth_state(),
            force_model: force_kind(cr),
            integrator: IntegratorKind::Dp54,
            options: tight_integrator(),
            drag: None,
            space_weather: None,
        }
        .force_model_with_drag()
        .expect("force"),
    );
    composite.add(Box::new(empirical));
    let values: Vec<f64> = composite.parameters().iter().map(|p| p.value).collect();
    let instants: Vec<Instant> = (0..SAMPLES)
        .map(|index| tt_instant(EPOCH_TT as i128 + index * STEP_S))
        .collect();
    let epochs = propagation_epochs(&instants);
    let initial = CartesianState {
        epoch_tdb_seconds: epochs[0].tdb_j2000_s,
        ..truth_state()
    };
    let states = propagate_variational(
        initial,
        &epochs,
        &VariationalEquations {
            force: &composite,
            parameters: &values,
            sensitivity_parameters: &[],
        },
        IntegratorKind::Dp54,
        &tight_integrator(),
        &PropagationContext::default(),
    )
    .expect("truth propagation");
    states
        .iter()
        .zip(instants)
        .map(|(state, epoch)| InertialPositionSample {
            epoch,
            position_gcrf_m: (state.state.position_km * M_PER_KM).into(),
        })
        .collect()
}

/// Options estimating C_R and the four empirical terms of
/// [`truth_empirical`], all starting from the model values 1.0 and zero.
fn estimation_options() -> OrbitParameterFitOptions {
    let empirical = EmpiricalRtnAcceleration::with_zero_terms(&[
        (RtnAxis::Radial, EmpiricalHarmonic::Constant),
        (RtnAxis::AlongTrack, EmpiricalHarmonic::Constant),
        (RtnAxis::CrossTrack, EmpiricalHarmonic::Cosine),
        (RtnAxis::CrossTrack, EmpiricalHarmonic::Sine),
    ]);
    let estimated = [SolarRadiationPressure::CR_PARAMETER.to_string()]
        .into_iter()
        .chain(empirical.parameters().into_iter().map(|p| p.name))
        .collect();
    OrbitParameterFitOptions {
        force_model: force_kind(1.0),
        additional_forces: vec![Arc::new(empirical)],
        estimated_parameters: estimated,
        integrator_options: tight_integrator(),
        convergence_threshold: 1.0e-6,
        ..OrbitParameterFitOptions::default()
    }
}

/// The truth state at the first epoch, labelled with its TDB epoch.
fn seed_label(samples: &[InertialPositionSample]) -> f64 {
    dynamical_seconds_from_instant(samples[0].epoch, satellite())
        .expect("epoch")
        .0
}

fn displaced_seed(samples: &[InertialPositionSample]) -> CartesianState {
    let truth = truth_state();
    CartesianState {
        epoch_tdb_seconds: seed_label(samples),
        position_km: truth.position_km + Vector3::new(0.3, -0.2, 0.15),
        velocity_km_s: truth.velocity_km_s + Vector3::new(-2.0e-5, 1.0e-5, 3.0e-5),
    }
}

#[test]
fn recovers_the_state_cr_and_empirical_terms_from_noise_free_positions() {
    let samples = synthetic_samples(1.25, truth_empirical());
    let options = estimation_options();
    let fit = fit_inertial_orbit_parameters(
        &samples,
        satellite(),
        Some(displaced_seed(&samples)),
        &options,
    )
    .expect("fit");

    // The observations are the truth trajectory itself, so the only error is
    // the integration error of the two propagations (tolerance 2.7 um per
    // step); the residuals stay below 1e-5 m and the estimates match the
    // truth to the precision those residuals allow.
    assert!(fit.seed_rms_3d_m > 100.0, "seed rms {}", fit.seed_rms_3d_m);
    assert!(
        fit.residual_stats.rms_3d_m < 1.0e-5,
        "post-fit rms {}",
        fit.residual_stats.rms_3d_m
    );
    let truth = truth_state();
    assert!((fit.initial_state.position_km - truth.position_km).norm() < 1.0e-8);
    assert!((fit.initial_state.velocity_km_s - truth.velocity_km_s).norm() < 1.0e-11);
    assert_eq!(fit.initial_state.epoch_tdb_seconds, seed_label(&samples));
    let cr = fit
        .force_parameter(SolarRadiationPressure::CR_PARAMETER)
        .expect("C_R estimated");
    assert_eq!(cr.initial_value, 1.0);
    assert!((cr.value - 1.25).abs() < 1.0e-6, "C_R {}", cr.value);
    for (axis, harmonic, truth_value) in truth_empirical().terms() {
        let name = EmpiricalRtnAcceleration::parameter_name(axis, harmonic);
        let estimate = fit.force_parameter(name).expect("estimated").value;
        assert!(
            (estimate - truth_value).abs() < 1.0e-17,
            "{name}: {estimate} vs {truth_value}"
        );
    }
    assert_eq!(fit.parameter_names.len(), 11);
    assert_eq!(&fit.parameter_names[..6], &STATE_PARAMETER_NAMES);
    assert_eq!(fit.residuals.len(), samples.len());
    assert!(fit.iterations >= 2 && fit.iterations <= 10);
    assert!(fit.condition_number > 1.0 && fit.condition_number.is_finite());
    match &fit.covariance {
        OrbitParameterCovariance::Estimated {
            matrix,
            correlation,
            variance_factor,
            degrees_of_freedom,
        } => {
            assert_eq!(*degrees_of_freedom, 3 * samples.len() - 11);
            // Residuals of 1e-5 m at a 1 m a-priori sigma.
            assert!(*variance_factor < 1.0e-10);
            for index in 0..11 {
                assert!(matrix[(index, index)] > 0.0);
                assert_eq!(correlation[(index, index)], 1.0);
                for other in 0..11 {
                    assert!(correlation[(index, other)].abs() <= 1.0 + 1.0e-12);
                    assert_eq!(correlation[(index, other)], correlation[(other, index)]);
                }
            }
        }
        OrbitParameterCovariance::Unbounded => panic!("covariance expected"),
    }
}

/// The synthetic arc with a deterministic pattern of centimetre-level offsets
/// standing in for observation noise, so the variance factor is well above
/// rounding.
fn noisy_samples() -> Vec<InertialPositionSample> {
    let mut samples = synthetic_samples(1.25, truth_empirical());
    add_offsets(&mut samples);
    samples
}

/// Add the centimetre-level offsets of [`noisy_samples`].
fn add_offsets(samples: &mut [InertialPositionSample]) {
    for (index, sample) in samples.iter_mut().enumerate() {
        let phase = index as f64 * 2.399;
        sample.position_gcrf_m[0] += 0.03 * libm::sin(phase);
        sample.position_gcrf_m[1] += 0.02 * libm::cos(1.7 * phase);
        sample.position_gcrf_m[2] -= 0.025 * libm::sin(0.6 * phase + 1.0);
    }
}

#[test]
fn the_scaled_covariance_does_not_depend_on_the_a_priori_sigma() {
    let samples = noisy_samples();
    let seed = displaced_seed(&samples);
    // A step's length in formal standard deviations scales with 1 / sigma,
    // so the convergence threshold is scaled with sigma; both are above the
    // integration noise of the steps (about 1e-6 formal standard deviations
    // at 1 m).
    let unit_options = OrbitParameterFitOptions {
        convergence_threshold: 1.0e-5,
        ..estimation_options()
    };
    let unit = fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &unit_options)
        .expect("fit at 1 m");
    let scaled_options = OrbitParameterFitOptions {
        observation_sigma_m: 0.01,
        convergence_threshold: 1.0e-3,
        ..estimation_options()
    };
    let centimetre =
        fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &scaled_options)
            .expect("fit at 1 cm");
    let (
        OrbitParameterCovariance::Estimated {
            matrix: unit_matrix,
            variance_factor: unit_factor,
            ..
        },
        OrbitParameterCovariance::Estimated {
            matrix: centimetre_matrix,
            variance_factor: centimetre_factor,
            ..
        },
    ) = (&unit.covariance, &centimetre.covariance)
    else {
        panic!("covariances expected");
    };
    // The weights scale by 1e4 and the variance factor by the same. Each fit
    // stops within 1e-5 formal standard deviations (at 1 m) of the minimum,
    // which leaves its sum of squares above the minimum by at most 1e-10 of
    // the a-priori variance, 2e-9 of the 0.047 sum; the sums also carry the
    // integration error of the trajectory, which moves them by about 1e-8 of
    // themselves between nearby estimates. 1e-6 relative covers both.
    assert!(
        (centimetre_factor / unit_factor / 1.0e4 - 1.0).abs() < 1.0e-6,
        "{unit_factor:e} vs {centimetre_factor:e}"
    );
    for row in 0..11 {
        for column in 0..11 {
            let a = unit_matrix[(row, column)];
            let b = centimetre_matrix[(row, column)];
            let scale = (unit_matrix[(row, row)] * unit_matrix[(column, column)]).sqrt();
            // The scaled covariance is the variance factor times a cofactor
            // that changes only to second order with the estimate.
            assert!(
                (a - b).abs() <= 1.0e-6 * scale,
                "C[{row}][{column}] {a} vs {b}"
            );
        }
    }
    // The pattern is about 2.6 cm per axis: the variance factor at a 1 m
    // a-priori sigma is of order (0.026)^2.
    assert!(
        *unit_factor > 1.0e-4 && *unit_factor < 2.0e-3,
        "{unit_factor}"
    );
    assert!(unit.sigma("srp.cr").expect("sigma") > 0.0);
    assert!(
        unit.correlation("srp.cr", "x_km")
            .expect("correlation")
            .abs()
            < 1.0
    );
    assert_eq!(unit.sigma("no.such"), None);
}

#[test]
fn the_state_alone_is_estimated_when_no_parameter_is_selected() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let options = OrbitParameterFitOptions {
        force_model: force_kind(1.0),
        integrator_options: tight_integrator(),
        ..OrbitParameterFitOptions::default()
    };
    // No seed: Herrick-Gibbs on the first three positions.
    let fit = fit_inertial_orbit_parameters(&samples, satellite(), None, &options)
        .expect("state-only fit");
    assert!(fit.force_parameters.is_empty());
    assert_eq!(fit.parameter_names.len(), 6);
    assert!(fit.residual_stats.rms_3d_m < 1.0e-5);
    assert!((fit.initial_state.position_km - truth_state().position_km).norm() < 1.0e-8);
}

/// A model with one parameter that has no effect.
struct Inert;

impl ForceModel for Inert {
    fn acceleration(
        &self,
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        Ok(Vector3::zeros())
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        vec![ForceParameter::new("test.inert", 0.0)]
    }

    fn acceleration_with_parameters(
        &self,
        _parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        Ok(Vector3::zeros())
    }

    fn acceleration_partials(
        &self,
        _parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        Ok(AccelerationPartials::zeros())
    }

    fn parameter_partials(
        &self,
        _parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        Ok(vec![Vector3::zeros()])
    }
}

/// A second solar radiation pressure scale, `k * a_srp(C_R = 1)` with
/// `k = 0` held, listed under the given name: its parameter partial is the
/// C_R partial exactly.
struct SrpCopy(SolarRadiationPressure, &'static str);

impl ForceModel for SrpCopy {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.acceleration_with_parameters(&[0.0], state, ctx)
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        vec![ForceParameter::new(self.1, 0.0)]
    }

    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        Ok(self.0.acceleration_with_parameters(&[1.0], state, ctx)? * parameters[0])
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        let unit = self.0.acceleration_partials(&[1.0], state, ctx)?;
        Ok(AccelerationPartials {
            wrt_position: unit.wrt_position * parameters[0],
            wrt_velocity: unit.wrt_velocity * parameters[0],
        })
    }

    fn parameter_partials(
        &self,
        _parameters: &[f64],
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        self.0.parameter_partials(&[1.0], state, ctx)
    }
}

fn state_cr_options() -> OrbitParameterFitOptions {
    OrbitParameterFitOptions {
        force_model: force_kind(1.0),
        estimated_parameters: vec![SolarRadiationPressure::CR_PARAMETER.to_string()],
        integrator_options: tight_integrator(),
        ..OrbitParameterFitOptions::default()
    }
}

#[test]
fn a_parameter_without_effect_is_unobservable() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let mut options = state_cr_options();
    options.additional_forces = vec![Arc::new(Inert)];
    options.estimated_parameters.push("test.inert".to_string());
    let error = fit_inertial_orbit_parameters(&samples, satellite(), None, &options)
        .expect_err("unobservable");
    assert!(
        matches!(&error, OrbitParameterFitError::UnobservableParameter { name, .. } if name == "test.inert"),
        "{error}"
    );
}

#[test]
fn collinear_parameters_make_the_normal_equations_singular() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let mut options = state_cr_options();
    options.additional_forces = vec![Arc::new(SrpCopy(
        SolarRadiationPressure::new(1.0, AREA_TO_MASS).expect("valid SRP"),
        "test.srp_copy",
    ))];
    options
        .estimated_parameters
        .push("test.srp_copy".to_string());
    let error =
        fit_inertial_orbit_parameters(&samples, satellite(), None, &options).expect_err("singular");
    match error {
        OrbitParameterFitError::SingularNormalEquations {
            condition_number,
            weakest,
            ..
        } => {
            assert!(condition_number > 1.0e25, "{condition_number:e}");
            // The weakest combination is C_R minus the copy, each 1/sqrt(2)
            // of the unit vector.
            let names: Vec<&str> = weakest.iter().map(|(name, _)| name.as_str()).collect();
            assert!(names.contains(&"srp.cr") && names.contains(&"test.srp_copy"));
            for (_, component) in &weakest[..2] {
                assert!((component.abs() - std::f64::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
            }
        }
        other => panic!("expected singular normal equations, got {other}"),
    }
}

#[test]
fn a_condition_number_above_the_limit_is_refused_and_reported() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let mut options = state_cr_options();
    options.max_condition_number = 1.5;
    let error = fit_inertial_orbit_parameters(&samples, satellite(), None, &options)
        .expect_err("ill-conditioned");
    match error {
        OrbitParameterFitError::IllConditionedNormalEquations {
            condition_number,
            limit,
            weakest,
            ..
        } => {
            assert_eq!(limit, 1.5);
            assert!(condition_number > 1.5 && condition_number.is_finite());
            assert!(!weakest.is_empty());
        }
        other => panic!("expected ill-conditioned normal equations, got {other}"),
    }
}

#[test]
fn too_few_iterations_is_non_convergence() {
    let samples = synthetic_samples(1.25, truth_empirical());
    let options = OrbitParameterFitOptions {
        max_iterations: 1,
        ..estimation_options()
    };
    let error = fit_inertial_orbit_parameters(
        &samples,
        satellite(),
        Some(displaced_seed(&samples)),
        &options,
    )
    .expect_err("one iteration is not enough from a displaced guess");
    match error {
        OrbitParameterFitError::DidNotConverge {
            iterations,
            last_step_sigmas,
            ..
        } => {
            assert_eq!(iterations, 1);
            assert!(last_step_sigmas > 1.0e-6);
        }
        other => panic!("expected non-convergence, got {other}"),
    }
}

#[test]
fn invalid_parameter_selections_are_refused() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let run = |estimated: Vec<&str>, additional: Vec<Arc<dyn ForceModel>>| {
        let options = OrbitParameterFitOptions {
            estimated_parameters: estimated.into_iter().map(str::to_string).collect(),
            additional_forces: additional,
            ..state_cr_options()
        };
        fit_inertial_orbit_parameters(&samples, satellite(), None, &options)
    };
    let refused = |result: Result<OrbitParameterFit, OrbitParameterFitError>, expected: &str| matches!(result, Err(OrbitParameterFitError::InvalidParameter { reason, .. }) if reason.contains(expected));
    assert!(refused(run(vec!["srp.nope"], vec![]), "not a parameter"));
    assert!(refused(
        run(vec!["srp.cr", "srp.cr"], vec![]),
        "more than once"
    ));
    let second_srp: Arc<dyn ForceModel> =
        Arc::new(SolarRadiationPressure::new(1.0, AREA_TO_MASS).expect("valid SRP"));
    assert!(refused(
        run(vec!["srp.cr"], vec![second_srp]),
        "more than one force model"
    ));
}

#[test]
fn invalid_options_and_inputs_are_typed_errors() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let invalid_option = |options: OrbitParameterFitOptions, field: &str| {
        matches!(
            fit_inertial_orbit_parameters(&samples, satellite(), None, &options),
            Err(OrbitParameterFitError::Fit(OrbitFitError::InvalidOption { field: f, .. })) if f == field
        )
    };
    let base = state_cr_options;
    assert!(invalid_option(
        OrbitParameterFitOptions {
            observation_sigma_m: 0.0,
            ..base()
        },
        "observation_sigma_m"
    ));
    assert!(invalid_option(
        OrbitParameterFitOptions {
            observation_sigma_m: f64::NAN,
            ..base()
        },
        "observation_sigma_m"
    ));
    assert!(invalid_option(
        OrbitParameterFitOptions {
            max_iterations: 0,
            ..base()
        },
        "max_iterations"
    ));
    assert!(invalid_option(
        OrbitParameterFitOptions {
            convergence_threshold: f64::INFINITY,
            ..base()
        },
        "convergence_threshold"
    ));
    assert!(invalid_option(
        OrbitParameterFitOptions {
            max_condition_number: 1.0,
            ..base()
        },
        "max_condition_number"
    ));
    assert!(invalid_option(
        OrbitParameterFitOptions {
            max_condition_number: f64::NAN,
            ..base()
        },
        "max_condition_number"
    ));
    assert!(invalid_option(
        OrbitParameterFitOptions {
            min_ledger_samples: 0,
            ..base()
        },
        "min_ledger_samples"
    ));

    let options = base();
    let mut seed = displaced_seed(&samples);
    seed.epoch_tdb_seconds += 1.0;
    assert!(matches!(
        fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &options),
        Err(OrbitParameterFitError::Fit(
            OrbitFitError::InvalidEpoch { .. }
        ))
    ));
    let mut seed = displaced_seed(&samples);
    seed.velocity_km_s.y = f64::NAN;
    assert!(matches!(
        fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &options),
        Err(OrbitParameterFitError::Fit(
            OrbitFitError::InvalidObservation { .. }
        ))
    ));
    assert!(matches!(
        fit_inertial_orbit_parameters(&samples[..1], satellite(), None, &options),
        Err(OrbitParameterFitError::Fit(OrbitFitError::TooFewSamples {
            got: 1,
            ..
        }))
    ));
    let mut repeated = samples.clone();
    repeated[3].epoch = repeated[2].epoch;
    assert!(matches!(
        fit_inertial_orbit_parameters(&repeated, satellite(), None, &options),
        Err(OrbitParameterFitError::Fit(
            OrbitFitError::NonMonotonicEpochs { .. }
        ))
    ));
    let mut non_finite = samples.clone();
    non_finite[5].position_gcrf_m[2] = f64::INFINITY;
    assert!(matches!(
        fit_inertial_orbit_parameters(&non_finite, satellite(), None, &options),
        Err(OrbitParameterFitError::Fit(
            OrbitFitError::InvalidObservation { .. }
        ))
    ));
    let starved = OrbitParameterFitOptions {
        integrator_options: IntegratorOptions {
            max_steps: 3,
            ..tight_integrator()
        },
        ..base()
    };
    assert!(matches!(
        fit_inertial_orbit_parameters(&samples, satellite(), None, &starved),
        Err(OrbitParameterFitError::Fit(OrbitFitError::Propagation {
            source: PropagationError::MaxStepsExceeded,
            ..
        }))
    ));
}

#[test]
fn unsorted_samples_are_ordered_by_epoch() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let mut shuffled = samples.clone();
    shuffled.reverse();
    shuffled.swap(3, 17);
    let options = state_cr_options();
    let sorted = fit_inertial_orbit_parameters(&samples, satellite(), None, &options).expect("fit");
    let unsorted =
        fit_inertial_orbit_parameters(&shuffled, satellite(), None, &options).expect("fit");
    assert_eq!(sorted, unsorted);
}

/// The scaled covariance, its correlations and the variance factor of a fit.
fn estimated_covariance(fit: &OrbitParameterFit) -> (&DMatrix<f64>, &DMatrix<f64>, f64) {
    match &fit.covariance {
        OrbitParameterCovariance::Estimated {
            matrix,
            correlation,
            variance_factor,
            ..
        } => (matrix, correlation, *variance_factor),
        OrbitParameterCovariance::Unbounded => panic!("covariance expected"),
    }
}

/// The estimate in the order of [`OrbitParameterFit::parameter_names`].
fn estimate_vector(fit: &OrbitParameterFit) -> Vec<f64> {
    let state = fit.initial_state;
    state
        .position_km
        .iter()
        .chain(state.velocity_km_s.iter())
        .copied()
        .chain(fit.force_parameters.iter().map(|parameter| parameter.value))
        .collect()
}

/// Options estimating the state, C_R and the four terms of
/// [`truth_empirical`], with the force models holding the true values.
fn truth_options() -> OrbitParameterFitOptions {
    let empirical = truth_empirical();
    let estimated = [SolarRadiationPressure::CR_PARAMETER.to_string()]
        .into_iter()
        .chain(empirical.parameters().into_iter().map(|p| p.name))
        .collect();
    OrbitParameterFitOptions {
        force_model: force_kind(1.25),
        additional_forces: vec![Arc::new(empirical)],
        estimated_parameters: estimated,
        integrator_options: tight_integrator(),
        ..OrbitParameterFitOptions::default()
    }
}

/// Assert that two fits of the same observations agree: estimates within
/// `sigmas` formal standard deviations of the first, covariances within
/// `relative` of their scale and correlations within `correlation_bound`.
fn assert_fits_agree(
    reference: &OrbitParameterFit,
    other: &OrbitParameterFit,
    sigmas: f64,
    relative: f64,
    correlation_bound: f64,
) {
    assert_eq!(reference.parameter_names, other.parameter_names);
    let (reference_matrix, reference_correlation, reference_factor) =
        estimated_covariance(reference);
    let (matrix, correlation, _) = estimated_covariance(other);
    let a = estimate_vector(reference);
    let b = estimate_vector(other);
    let n = a.len();
    for k in 0..n {
        let formal = (reference_matrix[(k, k)] / reference_factor).sqrt();
        assert!(
            (a[k] - b[k]).abs() <= sigmas * formal,
            "{}: {:e} vs {:e}, formal sigma {formal:e}",
            reference.parameter_names[k],
            a[k],
            b[k]
        );
    }
    for row in 0..n {
        for column in 0..n {
            let scale = (reference_matrix[(row, row)] * reference_matrix[(column, column)]).sqrt();
            assert!(
                (reference_matrix[(row, column)] - matrix[(row, column)]).abs() <= relative * scale,
                "C[{row}][{column}] {:e} vs {:e}",
                reference_matrix[(row, column)],
                matrix[(row, column)]
            );
            assert!(
                (reference_correlation[(row, column)] - correlation[(row, column)]).abs()
                    <= correlation_bound,
                "rho[{row}][{column}] {} vs {}",
                reference_correlation[(row, column)],
                correlation[(row, column)]
            );
        }
    }
}

#[test]
fn a_weighted_sum_of_squares_below_the_binary64_range_keeps_the_covariance() {
    // Noise-free positions fitted with the true force parameters leave the
    // integration error, under 1e-8 km, as residuals. At an a-priori sigma
    // of 1e160 m each weighted square is below 1e-330 and rounds to zero, so
    // the weighted sum and the variance factor are zero; the covariance is
    // the variance factor times a cofactor that grows with the sigma squared,
    // so it does not depend on the sigma, and the correlations depend on
    // neither.
    let samples = synthetic_samples(1.25, truth_empirical());
    let seed = displaced_seed(&samples);
    let at = |sigma_m: f64| OrbitParameterFitOptions {
        observation_sigma_m: sigma_m,
        // A step's length in formal standard deviations scales with
        // 1 / sigma: the same threshold in metres.
        convergence_threshold: 1.0e-5 / sigma_m,
        ..truth_options()
    };
    let unit = fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &at(1.0))
        .expect("fit at 1 m");
    let huge = fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &at(1.0e160))
        .expect("fit at 1e160 m");
    let (_, _, unit_factor) = estimated_covariance(&unit);
    let (_, _, huge_factor) = estimated_covariance(&huge);
    assert!(
        unit_factor > 0.0 && unit_factor < 1.0e-10,
        "{unit_factor:e}"
    );
    assert_eq!(huge_factor, 0.0);
    // Both fits take the same steps: the step does not depend on the sigma.
    assert_fits_agree(&unit, &huge, 1.0e-6, 1.0e-9, 1.0e-12);
}

#[test]
fn an_extreme_a_priori_sigma_leaves_the_estimate_and_covariance_unchanged() {
    // At 1e-145 m the weighted design of an empirical acceleration reaches
    // 4e157, whose square overflows; at 1e170 m the weighted design of a
    // position is near 1e-167, whose square underflows. Neither makes a
    // parameter unobservable, and the estimate and the scaled covariance do
    // not depend on the sigma.
    let samples = noisy_samples();
    let seed = displaced_seed(&samples);
    let at = |sigma_m: f64| OrbitParameterFitOptions {
        observation_sigma_m: sigma_m,
        convergence_threshold: 1.0e-5 / sigma_m,
        ..estimation_options()
    };
    let unit = fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &at(1.0))
        .expect("fit at 1 m");
    let (_, _, unit_factor) = estimated_covariance(&unit);
    for sigma_m in [1.0e-145, 1.0e170] {
        let fit = fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &at(sigma_m))
            .unwrap_or_else(|error| panic!("fit at {sigma_m:e} m: {error}"));
        assert_fits_agree(&unit, &fit, 1.0e-6, 1.0e-9, 1.0e-12);
        let (_, _, factor) = estimated_covariance(&fit);
        if sigma_m < 1.0 {
            // The variance factor scales with 1 / sigma^2.
            assert!(
                (factor * sigma_m * sigma_m / unit_factor - 1.0).abs() < 1.0e-9,
                "{factor:e} vs {unit_factor:e}"
            );
        } else {
            // Its value, 7e-4 * 1e-340, is below the smallest subnormal.
            assert_eq!(factor, 0.0);
        }
    }
}

#[test]
fn epochs_outside_the_calendar_are_invalid_epochs() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let options = state_cr_options();
    // i64::MAX seconds, where the civil date's noon offset overflows, and
    // 2^32 Gregorian years past J2000, whose year wraps into 0..=9999 when
    // narrowed to 32 bits.
    for start in [
        i128::from(i64::MAX),
        i128::from(i64::MIN) + 86_400,
        135_537_140_211_712_000,
        -135_537_140_211_712_000,
    ] {
        let far: Vec<InertialPositionSample> = samples
            .iter()
            .enumerate()
            .map(|(index, sample)| InertialPositionSample {
                epoch: tt_instant(start + index as i128 * STEP_S),
                ..*sample
            })
            .collect();
        let result = fit_inertial_orbit_parameters(&far, satellite(), None, &options);
        assert!(
            matches!(
                result,
                Err(OrbitParameterFitError::Fit(
                    OrbitFitError::InvalidEpoch { .. }
                ))
            ),
            "{start}: {result:?}"
        );
    }
}

/// A force without effect that records the epoch of every evaluation.
#[derive(Default)]
struct EpochRecorder(std::sync::Mutex<Vec<f64>>);

impl ForceModel for EpochRecorder {
    fn acceleration(
        &self,
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.0
            .lock()
            .expect("recorder")
            .push(state.epoch_tdb_seconds);
        Ok(Vector3::zeros())
    }

    fn acceleration_partials(
        &self,
        _parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        Ok(AccelerationPartials::zeros())
    }
}

fn tdb_labels(samples: &[InertialPositionSample]) -> Vec<f64> {
    samples
        .iter()
        .map(|sample| {
            dynamical_seconds_from_instant(sample.epoch, satellite())
                .expect("epoch")
                .0
        })
        .collect()
}

#[test]
fn the_force_models_receive_tdb_epochs() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    let recorder = Arc::new(EpochRecorder::default());
    let options = OrbitParameterFitOptions {
        additional_forces: vec![Arc::clone(&recorder) as Arc<dyn ForceModel>],
        ..state_cr_options()
    };
    fit_inertial_orbit_parameters(&samples, satellite(), None, &options).expect("fit");
    let labels = tdb_labels(&samples);
    let recorded = recorder.0.lock().expect("recorder");
    // Every observation's own TDB epoch is evaluated, and nothing outside
    // the arc's TDB span.
    for label in &labels {
        assert!(recorded.contains(label), "{label} not evaluated");
    }
    let first = labels[0];
    let last = labels[labels.len() - 1];
    assert!(
        recorded
            .iter()
            .all(|&epoch| (first..=last).contains(&epoch)),
        "an evaluation outside [{first}, {last}]"
    );
}

/// 2020-04-01 00:00 TT, in seconds past J2000, when TDB runs 1.65 ms ahead
/// of TT.
const APRIL_2020_TT: i128 = 638_971_200;

/// Positions of the truth state propagated with `kind` under `ctx` to the
/// epochs of `instants`, the first of which is the truth state's epoch.
fn propagated_samples(
    kind: ForceModelKind,
    ctx: &PropagationContext,
    instants: &[Instant],
) -> Vec<InertialPositionSample> {
    let force = StatePropagator {
        initial: truth_state(),
        force_model: kind,
        integrator: IntegratorKind::Dp54,
        options: tight_integrator(),
        drag: None,
        space_weather: None,
    }
    .force_model_with_drag()
    .expect("force");
    let values: Vec<f64> = force.parameters().iter().map(|p| p.value).collect();
    let epochs = propagation_epochs(instants);
    let initial = CartesianState {
        epoch_tdb_seconds: epochs[0].tdb_j2000_s,
        ..truth_state()
    };
    let states = propagate_variational(
        initial,
        &epochs,
        &VariationalEquations {
            force: force.as_ref(),
            parameters: &values,
            sensitivity_parameters: &[],
        },
        IntegratorKind::Dp54,
        &tight_integrator(),
        ctx,
    )
    .expect("truth propagation");
    states
        .iter()
        .zip(instants)
        .map(|(state, &epoch)| InertialPositionSample {
            epoch,
            position_gcrf_m: (state.state.position_km * M_PER_KM).into(),
        })
        .collect()
}

#[test]
fn a_polar_motion_series_spanning_exactly_the_arc_covers_the_fit() {
    use crate::astro::frames::orientation::{
        PolarMotionSample, PolarMotionSeriesEarthOrientationProvider,
    };
    let instants: Vec<Instant> = (0..9)
        .map(|index| tt_instant(APRIL_2020_TT + index * STEP_S))
        .collect();
    let labels: Vec<f64> = instants
        .iter()
        .map(|&instant| {
            dynamical_seconds_from_instant(instant, satellite())
                .expect("epoch")
                .0
        })
        .collect();
    let (first, last) = (labels[0], labels[8]);
    let pole = |epoch: f64| PolarMotionSample::from_arcseconds(epoch, 0.12, 0.31).expect("pole");
    let series = |from: f64, to: f64| -> Arc<dyn EarthOrientationProvider> {
        Arc::new(
            PolarMotionSeriesEarthOrientationProvider::new(vec![pole(from), pole(to)])
                .expect("series"),
        )
    };
    let kind = ForceModelKind::earth_phase_b(4, 4, None).expect("phase B");
    let wide = PropagationContext::new()
        .with_body_fixed_frame_provider(series(first - 86_400.0, last + 86_400.0));
    let samples = propagated_samples(kind, &wide, &instants);
    let options = OrbitParameterFitOptions {
        force_model: kind,
        integrator_options: tight_integrator(),
        propagation_context: PropagationContext::new()
            .with_body_fixed_frame_provider(series(first, last)),
        ..OrbitParameterFitOptions::default()
    };
    let seed = CartesianState {
        epoch_tdb_seconds: first,
        ..truth_state()
    };
    let fit = fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &options)
        .expect("the series covers every epoch the force models read");
    // Both propagations read the same constant pole.
    assert!(
        fit.residual_stats.rms_3d_m < 1.0e-5,
        "post-fit rms {}",
        fit.residual_stats.rms_3d_m
    );
}

/// The fit's force model with its estimated parameters in place.
fn estimated_force(
    fit: &OrbitParameterFit,
    options: &OrbitParameterFitOptions,
) -> (Box<dyn ForceModel>, Vec<f64>) {
    let force = build_force(options).expect("force");
    let listed = force.parameters();
    let mut values: Vec<f64> = listed.iter().map(|p| p.value).collect();
    for estimated in &fit.force_parameters {
        let index = listed
            .iter()
            .position(|p| p.name == estimated.name)
            .expect("listed");
        values[index] = estimated.value;
    }
    (force, values)
}

/// A post-fit residual recomputed from a propagated state.
fn rtn_residual_m(sample: &InertialPositionSample, state: &CartesianState) -> [f64; 3] {
    let rotation =
        rtn_to_eci_rotation(state.position_array(), state.velocity_array()).expect("RTN frame");
    let difference_m =
        (Vector3::from(sample.position_gcrf_m) / M_PER_KM - state.position_km) * M_PER_KM;
    let project = |axis: usize| {
        (0..3)
            .map(|component| difference_m[component] * rotation[component][axis])
            .sum::<f64>()
    };
    [project(0), project(1), project(2)]
}

#[test]
fn propagating_the_estimate_reproduces_the_fitted_residuals() {
    let samples = noisy_samples();
    let options = estimation_options();
    let fit = fit_inertial_orbit_parameters(
        &samples,
        satellite(),
        Some(displaced_seed(&samples)),
        &options,
    )
    .expect("fit");
    let (force, values) = estimated_force(&fit, &options);
    let propagate = |epochs: &[VariationalEpoch]| {
        propagate_variational(
            fit.initial_state,
            epochs,
            &VariationalEquations {
                force: force.as_ref(),
                parameters: &values,
                sensitivity_parameters: &[],
            },
            options.integrator,
            &options.integrator_options,
            &options.propagation_context,
        )
        .expect("propagation")
    };
    // The fit's own epochs reproduce its trajectory exactly.
    let epochs: Vec<VariationalEpoch> = fit
        .residuals
        .iter()
        .map(OrbitRtnResidual::variational_epoch)
        .collect();
    for ((state, residual), sample) in propagate(&epochs).iter().zip(&fit.residuals).zip(&samples) {
        let recomputed = rtn_residual_m(sample, &state.state);
        let reported = [residual.radial_m, residual.along_m, residual.cross_m];
        assert_eq!(recomputed, reported, "at {}", residual.epoch_tdb_j2000_s);
    }
    // Epochs formed from TDB seconds alone still carry the TDB rounding of the
    // input timestamps themselves, well below 1 mm at GNSS speed.
    let from_tdb: Vec<VariationalEpoch> = fit
        .residuals
        .iter()
        .map(|residual| {
            VariationalEpoch::from_tdb_seconds(
                fit.initial_state.epoch_tdb_seconds,
                residual.epoch_tdb_j2000_s,
            )
        })
        .collect();
    for ((state, residual), sample) in propagate(&from_tdb)
        .iter()
        .zip(&fit.residuals)
        .zip(&samples)
    {
        let recomputed = rtn_residual_m(sample, &state.state);
        let reported = [residual.radial_m, residual.along_m, residual.cross_m];
        for axis in 0..3 {
            assert!(
                (recomputed[axis] - reported[axis]).abs() <= 1.0e-3,
                "residual at {}: {recomputed:?} vs {reported:?}",
                residual.epoch_tdb_j2000_s
            );
        }
    }
}

#[test]
fn epochs_equal_in_tt_are_not_strictly_increasing() {
    use crate::astro::time::civil::j2000_seconds_from_split;
    use crate::astro::time::scales::TimeScales;
    // Two epochs one unit in the last place apart in TDB seconds whose TT
    // seconds are equal: the fit integrates in TT.
    let scales = TimeScales::from_scale(TimeScale::Tt, 2020, 6, 25, 0, 0, 0.0).expect("TT");
    let tt = j2000_seconds_from_split(scales.jd_whole, scales.tt_fraction);
    let tdb = j2000_seconds_from_split(scales.jd_whole, scales.tdb_fraction);
    let observation = |epoch_tdb: f64, y_km: f64| super::super::OrbitObservation {
        label_j2000_s: epoch_tdb,
        epoch_tdb_j2000_s: epoch_tdb,
        epoch_tt_j2000_s: tt,
        time_scale: TimeScale::Tdb,
        time_scales: scales,
        orientation: None,
        observed_itrs_km: [26_560.0, y_km, 0.0],
        observed_gcrs_km: [26_560.0, y_km, 0.0],
        observed_gcrs_velocity_km_s: None,
    };
    let observations = vec![observation(tdb, 0.0), observation(tdb.next_up(), 1.0e-4)];
    let validated = super::super::validate_observations(satellite(), observations.clone());
    assert!(
        matches!(validated, Err(OrbitFitError::NonMonotonicEpochs { .. })),
        "{validated:?}"
    );
    // The seed refuses a zero time step rather than dividing by it.
    let arc: Vec<ArcObservation> = observations.iter().map(ArcObservation::from).collect();
    let options = OrbitParameterFitOptions {
        integrator_options: tight_integrator(),
        ..OrbitParameterFitOptions::default()
    };
    let result = fit_arc(satellite(), &arc, None, &options);
    assert!(
        matches!(
            result,
            Err(OrbitParameterFitError::Fit(
                OrbitFitError::NonMonotonicEpochs { .. }
            ))
        ),
        "{result:?}"
    );
}

#[test]
fn a_force_parameter_named_like_a_state_component_is_refused() {
    let samples = synthetic_samples(1.0, EmpiricalRtnAcceleration::new());
    for name in STATE_PARAMETER_NAMES {
        let options = OrbitParameterFitOptions {
            additional_forces: vec![Arc::new(SrpCopy(
                SolarRadiationPressure::new(1.0, AREA_TO_MASS).expect("valid SRP"),
                name,
            ))],
            estimated_parameters: vec![name.to_string()],
            ..state_cr_options()
        };
        let result = fit_inertial_orbit_parameters(&samples, satellite(), None, &options);
        assert!(
            matches!(
                &result,
                Err(OrbitParameterFitError::InvalidParameter { name: refused, reason })
                    if refused == name && reason.contains("initial-state")
            ),
            "{name}: {result:?}"
        );
    }
}

#[test]
fn a_singular_value_that_is_not_a_number_is_refused() {
    assert_eq!(singular_value_extremes(&[1.0, 0.5]), Some((1, 0.5, 1.0)));
    for singular in [
        [1.0, f64::NAN, 0.5].as_slice(),
        &[f64::NAN, 1.0],
        &[1.0, f64::INFINITY],
        &[],
    ] {
        assert_eq!(singular_value_extremes(singular), None, "{singular:?}");
    }
}

/// A uniform acceleration `[k p, 0, 0]` km/s^2 with one parameter,
/// `test.uniform` = `p`, held at zero. Two models whose `k` differ by a power
/// of two give the same acceleration at values that differ by its inverse,
/// and their sensitivity columns differ by that power.
struct UniformAcceleration(f64);

impl ForceModel for UniformAcceleration {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.acceleration_with_parameters(&[0.0], state, ctx)
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        vec![ForceParameter::new("test.uniform", 0.0)]
    }

    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        Ok(Vector3::new(self.0 * parameters[0], 0.0, 0.0))
    }

    fn acceleration_partials(
        &self,
        _parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        Ok(AccelerationPartials::zeros())
    }

    fn parameter_partials(
        &self,
        _parameters: &[f64],
        _state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        Ok(vec![Vector3::new(self.0, 0.0, 0.0)])
    }
}

#[test]
fn a_sensitivity_column_whose_norm_exceeds_the_binary64_range_is_estimated() {
    // With k = 1.5e302 the x sensitivity grows as k t^2 / 2 to 1.1e308 at
    // 1200 s: every entry of the column is finite and its norm is not. The
    // same fit with k scaled by 2^-SHIFT has a column norm well inside the
    // range. Scaling the columns removes the factor, so both fits agree on
    // the state, its covariance and the correlations, and the parameter's
    // estimate and its rows of the covariance carry the factor. The
    // estimate, near -1.2e-312, is subnormal: its spacing, 5e-324, moves the
    // trajectory by under 1e-15 km.
    const SHIFT: i32 = 200;
    let k = 1.5e302;
    let instants: Vec<Instant> = [0, 1000, 1050, 1100, 1150, 1200]
        .iter()
        .map(|offset| tt_instant(EPOCH_TT as i128 + offset))
        .collect();
    let mut samples = propagated_samples(
        ForceModelKind::two_body(),
        &PropagationContext::default(),
        &instants,
    );
    add_offsets(&mut samples);
    let seed = CartesianState {
        epoch_tdb_seconds: seed_label(&samples),
        ..truth_state()
    };
    let options = |partial: f64| OrbitParameterFitOptions {
        force_model: ForceModelKind::two_body(),
        additional_forces: vec![Arc::new(UniformAcceleration(partial))],
        estimated_parameters: vec!["test.uniform".to_string()],
        integrator_options: tight_integrator(),
        max_condition_number: f64::INFINITY,
        ..OrbitParameterFitOptions::default()
    };

    // The column at the seed: finite entries whose norm, computed directly
    // in binary64, is infinite.
    let force = build_force(&options(k)).expect("force");
    let listed = force.parameters();
    let values: Vec<f64> = listed.iter().map(|parameter| parameter.value).collect();
    let uniform = listed
        .iter()
        .position(|parameter| parameter.name == "test.uniform")
        .expect("listed");
    let states = propagate_variational(
        seed,
        &propagation_epochs(&instants),
        &VariationalEquations {
            force: force.as_ref(),
            parameters: &values,
            sensitivity_parameters: &[uniform],
        },
        IntegratorKind::Dp54,
        &tight_integrator(),
        &PropagationContext::default(),
    )
    .expect("variational propagation");
    let column: Vec<f64> = states
        .iter()
        .flat_map(|state| (0..3).map(move |axis| state.sensitivity[(axis, 0)]))
        .collect();
    let largest = column
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs()));
    assert!(largest.is_finite() && largest > 1.0e308, "{largest:e}");
    let direct = largest
        * column
            .iter()
            .map(|value| (value / largest).powi(2))
            .sum::<f64>()
            .sqrt();
    assert!(direct.is_infinite(), "{direct:e}");

    let overflowing = fit_inertial_orbit_parameters(&samples, satellite(), Some(seed), &options(k))
        .expect("fit with a column norm beyond the binary64 range");
    let reference = fit_inertial_orbit_parameters(
        &samples,
        satellite(),
        Some(seed),
        &options(libm::scalbn(k, -SHIFT)),
    )
    .expect("fit with the scaled partial");
    assert_eq!(overflowing.parameter_names, reference.parameter_names);
    assert!(
        (overflowing.condition_number / reference.condition_number - 1.0).abs() < 1.0e-9,
        "{:e} vs {:e}",
        overflowing.condition_number,
        reference.condition_number
    );

    let (matrix, correlation, factor) = estimated_covariance(&overflowing);
    let (reference_matrix, reference_correlation, reference_factor) =
        estimated_covariance(&reference);
    assert!(
        reference_factor > 0.0 && (factor / reference_factor - 1.0).abs() < 1.0e-9,
        "{factor:e} vs {reference_factor:e}"
    );
    // Values of the reference fit scaled to the overflowing one: 2^-SHIFT
    // per index of the parameter.
    let parameter = STATE_PARAMETER_COUNT;
    let scale = |value: f64, row: usize, column: usize| {
        let shift = SHIFT * (i32::from(row == parameter) + i32::from(column == parameter));
        libm::scalbn(value, -shift)
    };
    let spacing = f64::from_bits(1);
    let estimate = estimate_vector(&overflowing);
    let reference_estimate = estimate_vector(&reference);
    for index in 0..=parameter {
        let formal = scale(
            (reference_matrix[(index, index)] / reference_factor).sqrt(),
            index,
            parameter + 1,
        );
        let expected = scale(reference_estimate[index], index, parameter + 1);
        assert!(
            (estimate[index] - expected).abs() <= 1.0e-6 * formal + 2.0 * spacing,
            "{}: {:e} vs {expected:e}, formal sigma {formal:e}",
            overflowing.parameter_names[index],
            estimate[index]
        );
    }
    assert!(estimate[parameter] != 0.0 && !estimate[parameter].is_normal());
    for row in 0..=parameter {
        for column in 0..=parameter {
            let bound = scale(
                (reference_matrix[(row, row)] * reference_matrix[(column, column)]).sqrt(),
                row,
                column,
            );
            let expected = scale(reference_matrix[(row, column)], row, column);
            assert!(
                (matrix[(row, column)] - expected).abs() <= 1.0e-9 * bound + 2.0 * spacing,
                "C[{row}][{column}] {:e} vs {expected:e}",
                matrix[(row, column)]
            );
            assert!(
                (correlation[(row, column)] - reference_correlation[(row, column)]).abs()
                    <= 1.0e-12,
                "rho[{row}][{column}] {} vs {}",
                correlation[(row, column)],
                reference_correlation[(row, column)]
            );
        }
    }
}

#[test]
fn a_column_norm_outside_the_binary64_range_divides_its_entries() {
    // sqrt(2) * f64::MAX and sqrt(2) * 2^-1074 are not binary64 values; the
    // entries divided by them are 1 / sqrt(2).
    for entry in [f64::MAX, f64::from_bits(1)] {
        let norm = column_norm([entry, -entry].into_iter()).expect("finite entries");
        let unit = norm.divide(entry);
        assert!(
            (unit - std::f64::consts::FRAC_1_SQRT_2).abs() <= f64::EPSILON,
            "{entry:e}: {unit}"
        );
        assert_eq!(norm.divide(-entry), -unit);
    }
    assert_eq!(
        column_norm([0.0, -0.0].into_iter()).map(|norm| norm.fraction),
        Some(0.0)
    );
    assert_eq!(column_norm([1.0, f64::NAN].into_iter()), None);
    assert_eq!(column_norm([1.0, f64::INFINITY].into_iter()), None);
}

/// Observations at the positions propagated from `initial` with the force
/// model and integrator of `options`, in km and without rounding, so a fit
/// from `initial` has residuals of exactly zero.
fn exact_arc(
    options: &OrbitParameterFitOptions,
    initial: CartesianState,
    instants: &[Instant],
) -> Vec<ArcObservation> {
    let force = build_force(options).expect("force");
    let values: Vec<f64> = force.parameters().iter().map(|p| p.value).collect();
    let epochs = propagation_epochs(instants);
    let states = propagate_variational(
        CartesianState {
            epoch_tdb_seconds: epochs[0].tdb_j2000_s,
            ..initial
        },
        &epochs,
        &VariationalEquations {
            force: force.as_ref(),
            parameters: &values,
            sensitivity_parameters: &[],
        },
        options.integrator,
        &options.integrator_options,
        &options.propagation_context,
    )
    .expect("propagation");
    instants
        .iter()
        .zip(&states)
        .map(|(&instant, state)| {
            let (epoch_tdb_s, epoch_tt_s) =
                dynamical_seconds_from_instant(instant, satellite()).expect("epoch");
            ArcObservation {
                epoch_tdb_s,
                epoch_tt_s,
                position_km: state.state.position_km,
                velocity_km_s: None,
            }
        })
        .collect()
}

#[test]
fn a_subnormal_a_priori_sigma_reports_the_zero_variance_factor_of_an_exact_fit() {
    // In km a sigma of 1e-322 m is zero. Fitted from the state they were
    // propagated from, the observations leave every residual exactly zero:
    // the variance factor and the covariance are zero, and the estimate and
    // the correlations are those at any other sigma.
    let sigma_m = 1.0e-322;
    assert_eq!(sigma_m / M_PER_KM, 0.0);
    let at = |sigma_m: f64| OrbitParameterFitOptions {
        force_model: ForceModelKind::two_body(),
        integrator_options: tight_integrator(),
        observation_sigma_m: sigma_m,
        ..OrbitParameterFitOptions::default()
    };
    let instants: Vec<Instant> = (0..5)
        .map(|index| tt_instant(EPOCH_TT as i128 + index * STEP_S))
        .collect();
    let arc = exact_arc(&at(1.0), truth_state(), &instants);
    let seed = CartesianState {
        epoch_tdb_seconds: arc[0].epoch_tdb_s,
        ..truth_state()
    };
    let unit = fit_arc(satellite(), &arc, Some(seed), &at(1.0)).expect("fit at 1 m");
    let fit = fit_arc(satellite(), &arc, Some(seed), &at(sigma_m)).expect("fit at 1e-322 m");
    assert_eq!(fit.residual_stats.rms_3d_m, 0.0);
    let (matrix, correlation, factor) = estimated_covariance(&fit);
    assert_eq!(factor, 0.0);
    assert!(matrix.iter().all(|value| *value == 0.0), "{matrix}");
    let (_, unit_correlation, _) = estimated_covariance(&unit);
    assert_eq!(correlation, unit_correlation);
    assert_eq!(estimate_vector(&fit), estimate_vector(&unit));
}

#[test]
fn a_subnormal_a_priori_sigma_measures_the_step_in_formal_standard_deviations() {
    // A free particle a micrometre from the origin, observed twice, the
    // second x one 1e-21 km off its trajectory: six residual components for
    // six parameters, so the first step removes the offset. That step, under
    // 1e-18 m, is about 1e303 formal standard deviations of 2e-321 m, a
    // sigma that is zero in km.
    let sigma_m = 2.0e-321;
    assert_eq!(sigma_m / M_PER_KM, 0.0);
    let free = |sigma_m: f64, convergence_threshold: f64| OrbitParameterFitOptions {
        force_model: ForceModelKind::composite(ForceModelComponents::EMPTY),
        integrator_options: tight_integrator(),
        observation_sigma_m: sigma_m,
        convergence_threshold,
        max_iterations: 1,
        ..OrbitParameterFitOptions::default()
    };
    let truth = CartesianState::new(0.0, [6.0e-10, 8.0e-10, 0.0], [-8.0e-13, 6.0e-13, 1.0e-13]);
    let instants = [
        tt_instant(EPOCH_TT as i128),
        tt_instant(EPOCH_TT as i128 + 60),
    ];
    let mut arc = exact_arc(&free(1.0, 1.0e-3), truth, &instants);
    arc[1].position_km.x += 1.0e-21;
    let seed = CartesianState {
        epoch_tdb_seconds: arc[0].epoch_tdb_s,
        ..truth
    };

    // Within a threshold of 1e308 the one step allowed converges.
    let fit = fit_arc(satellite(), &arc, Some(seed), &free(sigma_m, 1.0e308))
        .expect("converged in one step");
    assert_eq!(fit.iterations, 1);
    assert!(matches!(
        fit.covariance,
        OrbitParameterCovariance::Unbounded
    ));

    // The reported length is the length at 1 m divided by the sigma.
    let last_step = |sigma_m: f64, threshold: f64| match fit_arc(
        satellite(),
        &arc,
        Some(seed),
        &free(sigma_m, threshold),
    ) {
        Err(OrbitParameterFitError::DidNotConverge {
            last_step_sigmas, ..
        }) => last_step_sigmas,
        other => panic!("expected non-convergence at {sigma_m:e} m, got {other:?}"),
    };
    let at_unit = last_step(1.0, f64::MIN_POSITIVE);
    let at_sigma = last_step(sigma_m, 1.0e-3);
    assert!(at_unit > 0.0 && at_unit < 1.0e-18, "{at_unit:e}");
    assert!(
        (at_sigma * sigma_m / at_unit - 1.0).abs() < 4.0 * f64::EPSILON,
        "{at_sigma:e} vs {at_unit:e}"
    );
}
