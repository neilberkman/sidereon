//! Clean-room validation for the SP3-style orbit fit.
//!
//! Oracles are generated from the crate's published numerical propagators and
//! public frame transforms. No external source code or non-public data is used.
//! The fit integrates with TT as the time argument and labels states in TDB;
//! where a test needs the fit's own trajectory to the bit it propagates with
//! [`propagate_variational`], which does the same.

use std::sync::Arc;

use sidereon_core::astro::forces::{CompositeForceModel, ForceModel, J2Gravity, TwoBodyGravity};
use sidereon_core::astro::frames::orientation::{
    EarthOrientationProvider, PolarMotionSample, PolarMotionSeriesEarthOrientationProvider,
};
use sidereon_core::astro::frames::transforms::gcrs_to_itrs_compute;
use sidereon_core::astro::math::least_squares::SolveOptions;
use sidereon_core::astro::propagator::{
    propagate_variational, ForceModelKind, IntegratorKind, IntegratorOptions, PropagationContext,
    StatePropagator, VariationalEpoch, VariationalEquations,
};
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::time::civil::{
    civil_from_j2000_seconds, j2000_seconds, j2000_seconds_from_split,
    split_julian_date_from_j2000_seconds,
};
use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::astro::time::scales::TimeScales;
use sidereon_core::ephemeris::{
    fit_precise_ephemeris_sample_orbit, fit_precise_ephemeris_sample_orbit_with_initial_state,
    OrbitFitCovariance, OrbitFitError, OrbitFitOptions, PreciseEphemerisSample,
};
use sidereon_core::geometry_quality::ObservabilityTier;
use sidereon_core::{GnssSatelliteId, GnssSystem};

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS satellite")
}

fn instant_at(scale: TimeScale, epoch_j2000_s: i64) -> Instant {
    let (jd_whole, fraction) = split_julian_date_from_j2000_seconds(epoch_j2000_s);
    Instant::from_julian_date(
        scale,
        JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date"),
    )
}

fn time_scales_at(scale: TimeScale, epoch_j2000_s: i64) -> TimeScales {
    let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(epoch_j2000_s);
    TimeScales::from_scale(
        scale,
        year as i32,
        month as i32,
        day as i32,
        hour as i32,
        minute as i32,
        second as f64,
    )
    .expect("time scales in embedded coverage")
}

fn fit_options(force_model: ForceModelKind) -> OrbitFitOptions {
    let mut integrator_options = IntegratorOptions::default();
    integrator_options.abs_tol = 1.0e-12;
    integrator_options.rel_tol = 1.0e-13;
    integrator_options.initial_step = 10.0;
    integrator_options.max_step = 60.0;
    let mut solver_options = SolveOptions::default();
    solver_options.gtol = 1.0e-15;
    solver_options.ftol = 1.0e-15;
    solver_options.xtol = 1.0e-15;
    solver_options.max_nfev = 1200;
    let mut options = OrbitFitOptions::default();
    options.force_model = force_model;
    options.integrator = IntegratorKind::Dp54;
    options.integrator_options = integrator_options;
    options.solver_options = solver_options;
    options
}

/// Samples of `initial`, at the first of `epochs_j2000_s`, propagated with
/// `force` as the fit propagates: TT elapsed between the samples' own
/// epochs, states at their TDB.
fn generated_samples(
    sat: GnssSatelliteId,
    scale: TimeScale,
    initial: CartesianState,
    force: &dyn ForceModel,
    epochs_j2000_s: &[i64],
    options: IntegratorOptions,
) -> Vec<PreciseEphemerisSample> {
    let tt = |epoch: i64| {
        let ts = time_scales_at(scale, epoch);
        j2000_seconds_from_split(ts.jd_whole, ts.tt_fraction)
    };
    let tt0 = tt(epochs_j2000_s[0]);
    let epochs: Vec<VariationalEpoch> = epochs_j2000_s
        .iter()
        .map(|&epoch| VariationalEpoch {
            tdb_j2000_s: tdb_seconds_at(scale, epoch),
            elapsed_tt_s: tt(epoch) - tt0,
        })
        .collect();
    let states: Vec<CartesianState> = propagate_variational(
        initial,
        &epochs,
        &VariationalEquations {
            force,
            parameters: &[],
            sensitivity_parameters: &[],
        },
        IntegratorKind::Dp54,
        &options,
        &PropagationContext::default(),
    )
    .expect("generated truth arc")
    .into_iter()
    .map(|propagated| propagated.state)
    .collect();
    to_samples(sat, scale, &states, epochs_j2000_s)
}

/// Two-body gravity plus J2, as [`ForceModelKind::two_body_j2`] builds it.
fn two_body_j2_force() -> CompositeForceModel {
    let mut force = CompositeForceModel::new();
    force.add(Box::new(TwoBodyGravity::default()));
    force.add(Box::new(J2Gravity::default()));
    force
}

/// Samples of `initial` propagated with the force model, integrator and
/// propagation context of `options` to whole-second epochs in `scale`, the
/// first of which is the initial state's epoch. Both [`StatePropagator`] and
/// the fit integrate with elapsed TT as the independent variable and pass TDB
/// epochs to the force models.
fn generated_samples_with_context(
    sat: GnssSatelliteId,
    scale: TimeScale,
    initial: CartesianState,
    epochs_j2000_s: &[i64],
    options: &OrbitFitOptions,
) -> Vec<PreciseEphemerisSample> {
    let propagator = StatePropagator {
        initial,
        force_model: options.force_model,
        integrator: options.integrator,
        options: options.integrator_options,
        drag: None,
        space_weather: None,
    };
    let epochs: Vec<f64> = epochs_j2000_s
        .iter()
        .map(|&epoch| tdb_seconds_at(scale, epoch))
        .collect();
    let states = propagator
        .ephemeris_with_context(&epochs, &options.propagation_context)
        .expect("generated truth arc");
    to_samples(sat, scale, &states, epochs_j2000_s)
}

/// ITRS samples of GCRS states at whole-second epochs in `scale`.
fn to_samples(
    sat: GnssSatelliteId,
    scale: TimeScale,
    states: &[CartesianState],
    epochs_j2000_s: &[i64],
) -> Vec<PreciseEphemerisSample> {
    states
        .iter()
        .zip(epochs_j2000_s)
        .map(|(state, &epoch)| {
            let ts = time_scales_at(scale, epoch);
            let (x_km, y_km, z_km) = gcrs_to_itrs_compute(
                state.position_km.x,
                state.position_km.y,
                state.position_km.z,
                &ts,
                false,
            )
            .expect("GCRS to ITRS");
            PreciseEphemerisSample::new(
                sat,
                instant_at(scale, epoch),
                [x_km * 1000.0, y_km * 1000.0, z_km * 1000.0],
                None,
            )
        })
        .collect()
}

fn state_position_error_km(a: CartesianState, b: CartesianState) -> f64 {
    let dx = a.position_km.x - b.position_km.x;
    let dy = a.position_km.y - b.position_km.y;
    let dz = a.position_km.z - b.position_km.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

fn state_velocity_error_km_s(a: CartesianState, b: CartesianState) -> f64 {
    let dx = a.velocity_km_s.x - b.velocity_km_s.x;
    let dy = a.velocity_km_s.y - b.velocity_km_s.y;
    let dz = a.velocity_km_s.z - b.velocity_km_s.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[test]
fn self_consistency_recovers_own_dynamics_to_sub_micrometer_residuals() {
    let sat = gps(3);
    let start = j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64;
    let epochs: Vec<i64> = (0..=12).map(|step| start + step * 60).collect();
    let initial = CartesianState::new(
        tdb_seconds_at(TimeScale::Gpst, start),
        [7078.0, -30.0, 820.0],
        [0.20, 7.35, 1.05],
    );
    let options = fit_options(ForceModelKind::two_body());
    let samples = generated_samples(
        sat,
        TimeScale::Gpst,
        initial,
        &TwoBodyGravity::default(),
        &epochs,
        options.integrator_options,
    );

    let report =
        fit_precise_ephemeris_sample_orbit_with_initial_state(&samples, sat, initial, &options)
            .expect("self fit succeeds");
    let fit = report.fits.get(&sat).expect("fit for satellite");
    let stats = report
        .ledger
        .per_sat
        .get(&sat)
        .expect("ledger for satellite");

    assert_eq!(fit.geometry_quality.tier, ObservabilityTier::Nominal);
    assert!(
        stats.rms_3d_m < 1.0e-9,
        "self-consistency RMS was {:.17e} m",
        stats.rms_3d_m
    );
    assert!(
        state_position_error_km(fit.initial_state, initial) < 1.0e-12,
        "position error was {:.17e} km",
        state_position_error_km(fit.initial_state, initial)
    );
    assert!(
        state_velocity_error_km_s(fit.initial_state, initial) < 1.0e-12,
        "velocity error was {:.17e} km/s",
        state_velocity_error_km_s(fit.initial_state, initial)
    );
}

#[test]
fn perturbed_truth_fit_beats_two_body_fit_strictly() {
    let sat = gps(7);
    let start = j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64;
    let epochs: Vec<i64> = (0..=24).map(|step| start + step * 300).collect();
    let initial = CartesianState::new(
        tdb_seconds_at(TimeScale::Gpst, start),
        [7078.0, 0.0, 900.0],
        [0.0, 7.32, 1.10],
    );
    let j2_options = fit_options(ForceModelKind::two_body_j2());
    let samples = generated_samples(
        sat,
        TimeScale::Gpst,
        initial,
        &two_body_j2_force(),
        &epochs,
        j2_options.integrator_options,
    );

    let perturbed_seed = CartesianState::new(
        tdb_seconds_at(TimeScale::Gpst, start),
        [
            initial.position_km.x + 0.050,
            initial.position_km.y - 0.025,
            initial.position_km.z + 0.020,
        ],
        [
            initial.velocity_km_s.x + 2.0e-5,
            initial.velocity_km_s.y - 1.0e-5,
            initial.velocity_km_s.z + 1.5e-5,
        ],
    );
    let two_body = fit_precise_ephemeris_sample_orbit_with_initial_state(
        &samples,
        sat,
        perturbed_seed,
        &fit_options(ForceModelKind::two_body()),
    )
    .expect("two-body fit succeeds");
    let j2 = fit_precise_ephemeris_sample_orbit_with_initial_state(
        &samples,
        sat,
        perturbed_seed,
        &j2_options,
    )
    .expect("matching-force fit succeeds");
    let two_body_fit = two_body.fits.get(&sat).expect("two-body fit");
    let j2_fit = j2.fits.get(&sat).expect("J2 fit");
    let two_body_rms = two_body.ledger.per_sat.get(&sat).unwrap().rms_3d_m;
    let j2_rms = j2.ledger.per_sat.get(&sat).unwrap().rms_3d_m;

    assert!(
        two_body_fit.seed_rms_3d_m > two_body_rms,
        "two-body solve did not reduce seed RMS"
    );
    assert!(
        two_body_rms > j2_rms,
        "matching force RMS {:.17e} m did not beat two-body {:.17e} m",
        j2_rms,
        two_body_rms
    );
    assert!(j2_rms < 2.5e-3, "matching-force RMS was {:.17e} m", j2_rms);
    assert_eq!(j2_fit.geometry_quality.tier, ObservabilityTier::Nominal);
}

#[test]
fn two_epoch_arc_returns_unbounded_covariance_and_low_n_ledger() {
    let sat = gps(11);
    let start = j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64;
    let epochs = [start, start + 600];
    let initial = CartesianState::new(
        tdb_seconds_at(TimeScale::Gpst, start),
        [7078.0, 0.0, 820.0],
        [0.15, 7.35, 1.00],
    );
    let options = fit_options(ForceModelKind::two_body());
    let samples = generated_samples(
        sat,
        TimeScale::Gpst,
        initial,
        &TwoBodyGravity::default(),
        &epochs,
        options.integrator_options,
    );

    let report =
        fit_precise_ephemeris_sample_orbit(&samples, sat, &options).expect("short fit succeeds");
    let fit = report.fits.get(&sat).expect("fit for satellite");
    let stats = report
        .ledger
        .per_sat
        .get(&sat)
        .expect("ledger for satellite");

    assert_eq!(fit.geometry_quality.tier, ObservabilityTier::ZeroRedundancy);
    assert_eq!(fit.covariance, OrbitFitCovariance::Unbounded);
    assert_eq!(stats.n, 2);
    assert!(stats.low_sample_count);
}

/// TDB seconds past J2000 of a whole-second epoch in `scale`.
fn tdb_seconds_at(scale: TimeScale, epoch_j2000_s: i64) -> f64 {
    let ts = time_scales_at(scale, epoch_j2000_s);
    j2000_seconds_from_split(ts.jd_whole, ts.tdb_fraction)
}

#[test]
fn the_fitted_initial_state_carries_the_tdb_epoch_of_the_first_sample() {
    let sat = gps(3);
    let start = j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64;
    let epochs: Vec<i64> = (0..=12).map(|step| start + step * 60).collect();
    let initial = CartesianState::new(
        tdb_seconds_at(TimeScale::Gpst, start),
        [7078.0, -30.0, 820.0],
        [0.20, 7.35, 1.05],
    );
    let options = fit_options(ForceModelKind::two_body());
    let samples = generated_samples(
        sat,
        TimeScale::Gpst,
        initial,
        &TwoBodyGravity::default(),
        &epochs,
        options.integrator_options,
    );
    let report = fit_precise_ephemeris_sample_orbit(&samples, sat, &options).expect("fit");
    let fit = report.fits.get(&sat).expect("fit for satellite");
    // GPS time runs 51.184 s behind TT; the state is labelled in TDB.
    let first_tdb = tdb_seconds_at(TimeScale::Gpst, start);
    assert!((first_tdb - start as f64 - 51.184).abs() < 2.0e-3);
    assert_eq!(fit.initial_state.epoch_tdb_seconds, first_tdb);
    // A seed is given at the same epoch; the samples' own seconds are not it.
    assert!(fit_precise_ephemeris_sample_orbit_with_initial_state(
        &samples,
        sat,
        fit.initial_state,
        &options
    )
    .is_ok());
    let mut gpst_labelled = fit.initial_state;
    gpst_labelled.epoch_tdb_seconds = start as f64;
    assert!(matches!(
        fit_precise_ephemeris_sample_orbit_with_initial_state(
            &samples,
            sat,
            gpst_labelled,
            &options
        ),
        Err(OrbitFitError::InvalidEpoch { .. })
    ));
    // The residual ledger keeps the samples' own time scale and seconds.
    assert_eq!(report.ledger.arc_span.time_scale, TimeScale::Gpst);
    assert_eq!(report.ledger.arc_span.start_j2000_s, start as f64);
}

#[test]
fn a_polar_motion_series_spanning_exactly_the_arc_covers_the_state_only_fit() {
    let sat = gps(5);
    let start = j2000_seconds(2020, 4, 1, 0, 0, 0.0) as i64;
    let epochs: Vec<i64> = (0..=8).map(|step| start + step * 900).collect();
    let first = tdb_seconds_at(TimeScale::Gpst, epochs[0]);
    let last = tdb_seconds_at(TimeScale::Gpst, epochs[8]);
    let pole = |epoch: f64| PolarMotionSample::from_arcseconds(epoch, 0.12, 0.31).expect("pole");
    let series = |from: f64, to: f64| -> Arc<dyn EarthOrientationProvider> {
        Arc::new(
            PolarMotionSeriesEarthOrientationProvider::new(vec![pole(from), pole(to)])
                .expect("series"),
        )
    };
    let kind = ForceModelKind::earth_phase_b(4, 4, None).expect("phase B");
    let initial = CartesianState::new(first, [15_934.0, 21_245.0, 1_200.0], [-2.235, 1.680, 3.170]);
    let mut options = fit_options(kind);
    options.propagation_context = PropagationContext::new()
        .with_body_fixed_frame_provider(series(first - 86_400.0, last + 86_400.0));
    let samples = generated_samples_with_context(sat, TimeScale::Gpst, initial, &epochs, &options);
    options.propagation_context =
        PropagationContext::new().with_body_fixed_frame_provider(series(first, last));
    let report = fit_precise_ephemeris_sample_orbit(&samples, sat, &options)
        .expect("the series covers every epoch the force models read");
    let fit = report.fits.get(&sat).expect("fit for satellite");
    assert_eq!(fit.initial_state.epoch_tdb_seconds, first);
    assert!(
        fit.fit_rms_3d_m < 1.0e-3,
        "fit RMS {:e} m",
        fit.fit_rms_3d_m
    );
}
