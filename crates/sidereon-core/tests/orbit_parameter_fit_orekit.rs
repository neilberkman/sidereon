//! Force-parameter orbit fit against Orekit's `BatchLSEstimator`.
//!
//! The fixture `tests/fixtures/orbit_determination/orbit_parameter_fit_orekit.json`
//! comes from `fixtures-generators/orekit_orbit_parameter_oracle/`
//! (`OrbitParameterOracle.java`, Orekit 13.1.10 and Hipparchus 4.0.3; the
//! versions and the orekit-data commit are in its `provenance`). The arc is
//! GPS G03 in `tests/fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3`, the
//! CODE MGEX final orbit of 2020-06-25: 289 positions at 5-minute spacing (see
//! `tests/fixtures/sp3/PROVENANCE.md`). The satellite stays in sunlight all
//! day. Orekit rotated the positions to GCRF and fitted them with
//! Gauss-Newton batch least squares, estimating the initial state with
//! ParameterDrivers selected on the cannonball SRP coefficient, on empirical
//! RTN accelerations (constant and once per revolution), and on Orekit's
//! `ECOM2` model (used unchanged because G03 stays in full sunlight across the
//! entire arc: D0/Y0/B0, D2B1, and D4B1). Force models and constants are those
//! of sidereon's Phase A composite (see the generator for the list). The method
//! is that of Montenbruck & Gill, "Satellite Orbits", chapters 7 and 8.
//!
//! The comparisons, in order:
//! 1. the analytic Sun and Moon positions and every force model at the initial
//!    guess, so that a disagreement of the fits is not a disagreement of the
//!    models;
//! 2. the fit of the fixture's GCRF observations from the fixture's guess
//!    ([`fit_inertial_orbit_parameters`]): initial state, parameters,
//!    variance factor, covariance scaled by the variance factor, correlations
//!    and post-fit RTN residuals;
//! 3. the fit of the SP3 file itself through sidereon's own frame chain
//!    ([`fit_sp3_ecef_orbit_parameters`]), whose observations differ from
//!    Orekit's GCRF positions by the difference of the two frame chains
//!    (up to 0.14 m here, see the test); the estimates must agree within the
//!    first-order bound that difference allows.
//!
//! The largest disagreements with Orekit on the same observations, in formal
//! standard deviations for a 1 m a-priori sigma: 4.6e-6 (`vz_km_s`, state and
//! C_R), 6.6e-6 (along-track constant, state, C_R and RTN constants), 3.4e-6
//! (along-track constant, state and nine RTN terms), 1.2e-6 (`ecom2.y0`, state
//! and ECOM2 D0/Y0/B0), 8.6e-7 (`vz_km_s`, state and ECOM2 D2B1), and 3.4e-7
//! (`ecom2.d2c`, state and ECOM2 D4B1); per-epoch residuals within 3.3 um
//! (within 2.7 um on ECOM2 cases); variance factors within 1.1e-8 relative;
//! formal standard deviations within 9.3e-13 relative and correlations within
//! 3.31e-12.

#![cfg(sidereon_repo_tests)]

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::Value;
use sidereon_core::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds;
use sidereon_core::astro::forces::zonal::ZonalGravity;
use sidereon_core::astro::forces::{
    Ecom2D2b1Coefficients, Ecom2D4b1Coefficients, Ecom2SolarRadiationPressure, EmpiricalHarmonic,
    EmpiricalRtnAcceleration, ForceModel, RtnAxis, SchwarzschildRelativity, SolarRadiationPressure,
    ThirdBodyBodies, ThirdBodyGravity, TwoBodyGravity,
};
use sidereon_core::astro::frames::orientation::{
    EarthOrientationProvider, PolarMotionSample, PolarMotionSeriesEarthOrientationProvider,
};
use sidereon_core::astro::propagator::{ForceModelKind, IntegratorOptions, PropagationContext};
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::time::civil::j2000_seconds_from_split;
use sidereon_core::astro::time::model::TimeScale;
use sidereon_core::astro::time::scales::{
    tdb_from_tt_j2000_seconds, tt_from_tdb_j2000_seconds, TimeScales,
};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::orbit_determination::{
    fit_inertial_orbit_parameters, fit_sp3_ecef_orbit_parameters, InertialPositionSample,
    OrbitParameterCovariance, OrbitParameterFit, OrbitParameterFitOptions,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const SP3: &[u8] = include_bytes!("fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3");

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/orbit_determination/orbit_parameter_fit_orekit.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    serde_json::from_str(&text).expect("fixture JSON")
}

fn f(value: &Value) -> f64 {
    value.as_f64().expect("number")
}

fn vector(value: &Value) -> [f64; 3] {
    let items = value.as_array().expect("array");
    [f(&items[0]), f(&items[1]), f(&items[2])]
}

fn g03() -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, 3).expect("valid satellite")
}

fn product() -> Sp3 {
    Sp3::parse(SP3).expect("parse SP3 fixture")
}

/// The fixture's observations paired with the SP3 epochs of G03.
fn inertial_samples(fixture: &Value) -> Vec<InertialPositionSample> {
    let observations = fixture["observations"].as_array().expect("observations");
    let samples: Vec<_> = product()
        .precise_ephemeris_samples()
        .into_iter()
        .filter(|sample| sample.sat == g03())
        .collect();
    assert_eq!(samples.len(), observations.len());
    samples
        .iter()
        .zip(observations)
        .map(|(sample, observation)| {
            // Both read the same SP3 records.
            assert_eq!(sample.position_ecef_m, vector(&observation["itrf_m"]));
            InertialPositionSample {
                epoch: sample.epoch,
                position_gcrf_m: vector(&observation["gcrf_m"]),
            }
        })
        .collect()
}

/// sidereon's names of a case's empirical terms.
fn empirical_terms(estimated: &[String]) -> Vec<(RtnAxis, EmpiricalHarmonic)> {
    let mut terms = Vec::new();
    for axis in RtnAxis::ALL {
        for harmonic in EmpiricalHarmonic::ALL {
            let name = EmpiricalRtnAcceleration::parameter_name(axis, harmonic);
            if estimated.iter().any(|entry| entry == name) {
                terms.push((axis, harmonic));
            }
        }
    }
    terms
}

fn options(estimated: &[String]) -> OrbitParameterFitOptions {
    let mut options = OrbitParameterFitOptions::default();
    let is_ecom2 = estimated.iter().any(|entry| entry.starts_with("ecom2."));
    if is_ecom2 {
        let ecom2 = if estimated.iter().any(|entry| entry == "ecom2.d4c") {
            Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients::default()).expect("valid D4B1")
        } else if estimated.iter().any(|entry| entry == "ecom2.d2c") {
            Ecom2SolarRadiationPressure::d2b1(Ecom2D2b1Coefficients::default()).expect("valid D2B1")
        } else {
            Ecom2SolarRadiationPressure::new(0, 0, vec![0.0; 3]).expect("valid D0B0")
        };
        options.force_model = ForceModelKind::earth_phase_a(None);
        options.additional_forces = vec![Arc::new(ecom2)];
    } else {
        let srp = SolarRadiationPressure::new(1.0, 0.02).expect("valid SRP");
        options.force_model = ForceModelKind::earth_phase_a(Some(srp));
        let terms = empirical_terms(estimated);
        if !terms.is_empty() {
            options.additional_forces =
                vec![Arc::new(EmpiricalRtnAcceleration::with_zero_terms(&terms))];
        }
    }
    options.estimated_parameters = estimated.to_vec();
    let mut integrator = IntegratorOptions::default();
    integrator.abs_tol = 1.0e-13;
    integrator.rel_tol = 1.0e-15;
    integrator.initial_step = 30.0;
    integrator.max_step = 60.0;
    options.integrator_options = integrator;
    options.observation_sigma_m = 1.0;
    options.convergence_threshold = 1.0e-4;
    options
}

/// Metres (or m/s, m/s^2) per sidereon unit of a parameter.
fn metres_per_unit(name: &str) -> f64 {
    if name == "srp.cr" || name.starts_with("ecom2.") {
        1.0
    } else {
        1.0e3
    }
}

/// The fixture's parameter name for one of sidereon's.
fn orekit_name(name: &str) -> &str {
    match name {
        "x_km" => "x",
        "y_km" => "y",
        "z_km" => "z",
        "vx_km_s" => "vx",
        "vy_km_s" => "vy",
        "vz_km_s" => "vz",
        other => other,
    }
}

/// One fixture case read into sidereon's parameter order and units of
/// metres.
struct OrekitCase {
    name: String,
    estimated: Vec<String>,
    /// Estimate in sidereon's order, SI units.
    estimate: Vec<f64>,
    /// Covariance scaled by Orekit's variance factor, sidereon's order, SI.
    covariance: Vec<Vec<f64>>,
    variance_factor: f64,
    rtn_rms_m: [f64; 3],
    rtn_residuals_m: Vec<[f64; 3]>,
}

fn orekit_case(case: &Value, names: &[String]) -> OrekitCase {
    let order: Vec<&str> = case["parameter_order"]
        .as_array()
        .expect("order")
        .iter()
        .map(|value| value.as_str().expect("name"))
        .collect();
    let index = |name: &str| {
        order
            .iter()
            .position(|entry| *entry == orekit_name(name))
            .unwrap_or_else(|| panic!("{name} not in the Orekit case"))
    };
    let position = vector(&case["initial_position_m"]);
    let velocity = vector(&case["initial_velocity_m_s"]);
    let estimate = names
        .iter()
        .enumerate()
        .map(|(k, name)| match k {
            0..=2 => position[k],
            3..=5 => velocity[k - 3],
            _ => f(&case["parameters"][name.as_str()]),
        })
        .collect();
    let variance_factor = f(&case["variance_factor"]);
    let unscaled = &case["covariance_unscaled"];
    let covariance = names
        .iter()
        .map(|row| {
            names
                .iter()
                .map(|column| f(&unscaled[index(row)][index(column)]) * variance_factor)
                .collect()
        })
        .collect();
    OrekitCase {
        name: case["name"].as_str().expect("name").to_string(),
        estimated: case["estimated"]
            .as_array()
            .expect("estimated")
            .iter()
            .map(|value| value.as_str().expect("name").to_string())
            .collect(),
        estimate,
        covariance,
        variance_factor,
        rtn_rms_m: vector(&case["rtn_rms_m"]),
        rtn_residuals_m: case["rtn_residuals_m"]
            .as_array()
            .expect("residuals")
            .iter()
            .map(vector)
            .collect(),
    }
}

/// sidereon's estimate in SI units, in its own parameter order.
fn sidereon_estimate(fit: &OrbitParameterFit) -> Vec<f64> {
    let state = &fit.initial_state;
    state
        .position_km
        .iter()
        .chain(state.velocity_km_s.iter())
        .map(|value| value * 1.0e3)
        .chain(
            fit.force_parameters
                .iter()
                .map(|parameter| parameter.value * metres_per_unit(&parameter.name)),
        )
        .collect()
}

fn sidereon_covariance(fit: &OrbitParameterFit) -> (Vec<Vec<f64>>, f64) {
    let OrbitParameterCovariance::Estimated {
        matrix,
        variance_factor,
        ..
    } = &fit.covariance
    else {
        panic!("covariance expected");
    };
    let unit = |k: usize| {
        if k < 6 {
            1.0e3
        } else {
            metres_per_unit(&fit.parameter_names[k])
        }
    };
    let n = fit.parameter_names.len();
    let si = (0..n)
        .map(|row| {
            (0..n)
                .map(|column| matrix[(row, column)] * unit(row) * unit(column))
                .collect()
        })
        .collect();
    (si, *variance_factor)
}

/// Largest difference of a post-fit residual between the two estimators at
/// one epoch, m. Both integrate the trajectory with the partials as
/// secondary equations on the trajectory's steps: Orekit with an
/// eighth-order Dormand-Prince to 1e-6 m per step, sidereon with the
/// fifth-order Dormand-Prince to 1e-15 of the 26560 km radius (2.7e-8 m) per
/// step. At these tolerances the trajectories differ by a few micrometres
/// over the day. The bound belongs to these tolerances: at a relative
/// tolerance of 1e-13 the fifth-order method drifts along track by 0.2 mm
/// over the day.
const RESIDUAL_DIFFERENCE_M: f64 = 1.0e-5;

/// Largest difference of an estimate, in its formal standard deviation for a
/// 1 m a-priori sigma. By the Cauchy-Schwarz inequality a change `d` of the
/// observations or of the modelled trajectory moves parameter `j` of a
/// weighted least-squares estimate by at most `sigma_j |d|_W`; with the
/// residual difference bound at each of the 289 epochs,
/// `|d|_W <= sqrt(289) * 1e-5 m / 1 m = 1.7e-4`. Each fit also stops short
/// of its fixed point: sidereon within 1e-4 formal standard deviations (its
/// convergence threshold), Orekit within 1e-3 of its driver scales (1 um,
/// 1e-8 in C_R and 1e-15 m/s^2, at most 1.3e-5 of the formal deviations).
const ESTIMATE_DIFFERENCE_SIGMAS: f64 = 1.7e-4 + 1.0e-4 + 1.3e-5;

/// Largest relative difference of a formal standard deviation and largest
/// difference of a correlation coefficient. Both come from the cofactor
/// `(H^T W H)^-1`, which depends only on the partial derivatives; these agree
/// to the relative accuracy of the integration, of order 1e-12, which the
/// inversion amplifies by at most the condition number of the normal
/// equations (up to 3.537e5 for `state_ecom2_d4b1`, below 3.6e5 in every
/// case). Across all six cases the measured maxima are 9.30e-13 relative in
/// formal standard deviations and 3.31e-12 in correlation coefficients (both
/// on `state_ecom2_d4b1`).
const COFACTOR_DIFFERENCE: f64 = 1.0e-12 * 3.6e5;

#[test]
fn analytic_bodies_and_force_models_match_the_oracle_inputs() {
    let fixture = fixture();
    for body in fixture["bodies"].as_array().expect("bodies") {
        let tdb = tdb_from_tt_j2000_seconds(f(&body["tt_j2000_s"]));
        let positions = sun_moon_gcrs_at_tdb_seconds(tdb).expect("GCRS series");
        // The generator ports the series and the mean-of-date to GCRS rotation
        // `(P * B)^T` at TDB with the same operations in the same order and
        // fdlibm's StrictMath, the algorithm of the libm crate, so the GCRS
        // positions are identical.
        assert_eq!(positions.sun, vector(&body["sun_m"]));
        assert_eq!(positions.moon, vector(&body["moon_m"]));
    }

    let guess = &fixture["initial_guess"];
    let tt0 = f(&fixture["observations"][0]["tt_j2000_s"]);
    // The force models receive TDB and evaluate the Sun and Moon at its TT;
    // this TDB converts back to the fixture's TT exactly.
    let tdb0 = tdb_from_tt_j2000_seconds(tt0);
    assert_eq!(tt_from_tdb_j2000_seconds(tdb0), tt0);
    let km = |m: [f64; 3]| [m[0] / 1.0e3, m[1] / 1.0e3, m[2] / 1.0e3];
    let state = CartesianState::new(
        tdb0,
        km(vector(&guess["position_m"])),
        km(vector(&guess["velocity_m_s"])),
    );
    let ctx = PropagationContext::default();
    let accelerations = &fixture["accelerations_at_guess_m_s2"];
    let mut models: Vec<(&str, Box<dyn ForceModel>)> = vec![
        ("central", Box::new(TwoBodyGravity::default())),
        ("zonal", Box::new(ZonalGravity::earth_j2_through_j6())),
        (
            "sun",
            Box::new(ThirdBodyGravity {
                bodies: ThirdBodyBodies {
                    sun: true,
                    moon: false,
                },
                ..ThirdBodyGravity::default()
            }),
        ),
        (
            "moon",
            Box::new(ThirdBodyGravity {
                bodies: ThirdBodyBodies {
                    sun: false,
                    moon: true,
                },
                ..ThirdBodyGravity::default()
            }),
        ),
        (
            "srp_unit_cr",
            Box::new(SolarRadiationPressure::new(1.0, 0.02).expect("valid SRP")),
        ),
        ("relativity", Box::new(SchwarzschildRelativity::default())),
    ];
    for axis in RtnAxis::ALL {
        for harmonic in EmpiricalHarmonic::ALL {
            let model = EmpiricalRtnAcceleration::new()
                .with_term(axis, harmonic, 1.0e-12)
                .expect("finite");
            models.push((
                EmpiricalRtnAcceleration::parameter_name(axis, harmonic),
                Box::new(model),
            ));
        }
    }
    models.push((
        "ecom2_d4b1",
        Box::new(
            Ecom2SolarRadiationPressure::d4b1(Ecom2D4b1Coefficients {
                b0: -1.0e-9,
                b1c: 3.0e-10,
                b1s: -4.0e-10,
                d0: -1.0e-7,
                d2c: 2.0e-9,
                d4c: 1.0e-10,
                d2s: -5.0e-10,
                d4s: 2.0e-10,
                y0: 5.0e-10,
            })
            .expect("valid ECOM2"),
        ),
    ));
    for (name, model) in &models {
        let sidereon = model.acceleration(&state, &ctx).expect("acceleration") * 1.0e3;
        let orekit = vector(&accelerations[*name]);
        let difference = (0..3)
            .map(|k| (sidereon[k] - orekit[k]).powi(2))
            .sum::<f64>()
            .sqrt();
        let norm = orekit.iter().map(|v| v * v).sum::<f64>().sqrt();
        // Each model is the same formula evaluated in another order: a few
        // ulps, 1e-15 relative. The Sun's tidal acceleration is the
        // difference of two terms 5300 times larger than itself (the direct
        // and the indirect attraction, 5.9e-3 m/s^2 each), so its rounding is
        // that much larger: 1.2e-12 relative, 1.2e-18 m/s^2, which moves the
        // satellite by 5e-9 m over the day.
        let tolerance = if *name == "sun" { 2.0e-12 } else { 4.0e-15 };
        assert!(
            difference <= tolerance * norm,
            "{name}: sidereon {sidereon:?} vs Orekit {orekit:?}, relative {:e}",
            difference / norm
        );
    }
}

/// One fit of the fixture's GCRF observations from the fixture's guess.
fn inertial_fit(
    fixture: &Value,
    samples: &[InertialPositionSample],
    estimated: &[String],
) -> OrbitParameterFit {
    let guess = &fixture["initial_guess"];
    let km = |m: [f64; 3]| [m[0] / 1.0e3, m[1] / 1.0e3, m[2] / 1.0e3];
    let seed = CartesianState::new(
        first_epoch_tdb(),
        km(vector(&guess["position_m"])),
        km(vector(&guess["velocity_m_s"])),
    );
    fit_inertial_orbit_parameters(samples, g03(), Some(seed), &options(estimated)).expect("fit")
}

fn case_names(case: &Value) -> Vec<String> {
    case["estimated"]
        .as_array()
        .expect("estimated")
        .iter()
        .map(|value| value.as_str().expect("name").to_string())
        .collect()
}

/// Formal standard deviations for the 1 m a-priori sigma: the scaled
/// covariance over the variance factor.
fn formal_sigmas(covariance: &[Vec<f64>], variance_factor: f64) -> Vec<f64> {
    (0..covariance.len())
        .map(|k| (covariance[k][k] / variance_factor).sqrt())
        .collect()
}

#[test]
fn fits_of_the_same_observations_agree_with_orekit() {
    let fixture = fixture();
    let samples = inertial_samples(&fixture);
    for case in fixture["cases"].as_array().expect("cases") {
        let estimated = case_names(case);
        let fit = inertial_fit(&fixture, &samples, &estimated);
        let orekit = orekit_case(case, &fit.parameter_names);
        assert_eq!(orekit.estimated, estimated);
        let label = &orekit.name;
        let estimate = sidereon_estimate(&fit);
        let (covariance, variance_factor) = sidereon_covariance(&fit);
        let sidereon_sigmas = formal_sigmas(&covariance, variance_factor);
        let orekit_sigmas = formal_sigmas(&orekit.covariance, orekit.variance_factor);
        let n = fit.parameter_names.len();
        assert_eq!(n, 6 + estimated.len());
        assert!(
            fit.iterations <= 10,
            "{label}: {} iterations",
            fit.iterations
        );

        let mut max_sigma_diff = 0.0_f64;
        let mut max_sigma_name = "";
        let mut max_sigma_rel = 0.0_f64;
        let mut max_corr_diff = 0.0_f64;
        for (k, name) in fit.parameter_names.iter().enumerate() {
            let difference = (estimate[k] - orekit.estimate[k]) / orekit_sigmas[k];
            if difference.abs() > max_sigma_diff {
                max_sigma_diff = difference.abs();
                max_sigma_name = name.as_str();
            }
            assert!(
                difference.abs() <= ESTIMATE_DIFFERENCE_SIGMAS,
                "{label} {name}: sidereon {:e}, Orekit {:e}, {difference:e} formal sigma",
                estimate[k],
                orekit.estimate[k]
            );
            let sigma_rel = (sidereon_sigmas[k] / orekit_sigmas[k] - 1.0).abs();
            max_sigma_rel = max_sigma_rel.max(sigma_rel);
            assert!(
                sigma_rel <= COFACTOR_DIFFERENCE,
                "{label} {name}: formal sigma {:e} vs {:e}",
                sidereon_sigmas[k],
                orekit_sigmas[k]
            );
        }
        for row in 0..n {
            for column in 0..n {
                let correlation =
                    |c: &[Vec<f64>]| c[row][column] / (c[row][row] * c[column][column]).sqrt();
                let sidereon = correlation(&covariance);
                let oracle = correlation(&orekit.covariance);
                let corr_diff = (sidereon - oracle).abs();
                max_corr_diff = max_corr_diff.max(corr_diff);
                assert!(
                    corr_diff <= COFACTOR_DIFFERENCE,
                    "{label} correlation {row},{column}: {sidereon} vs {oracle}"
                );
                if let OrbitParameterCovariance::Estimated { correlation, .. } = &fit.covariance {
                    assert!((correlation[(row, column)] - sidereon).abs() <= 1.0e-14);
                }
            }
        }

        // The weighted sums of squares of two minima whose residual functions
        // differ by d differ by at most |d| (|r_1| + |r_2|): relative
        // 2 |d| / |r|, with |d| <= sqrt(289) * RESIDUAL_DIFFERENCE_M.
        let residual_norm = (variance_factor * (3 * samples.len() - n) as f64).sqrt();
        let difference_norm = (samples.len() as f64).sqrt() * RESIDUAL_DIFFERENCE_M;
        let vf_rel = (variance_factor / orekit.variance_factor - 1.0).abs();
        assert!(
            vf_rel <= 2.0 * difference_norm / residual_norm,
            "{label}: variance factor {variance_factor} vs {}",
            orekit.variance_factor
        );

        assert_eq!(fit.residuals.len(), orekit.rtn_residuals_m.len());
        let mut max_res_diff_m = 0.0_f64;
        for (sidereon, oracle) in fit.residuals.iter().zip(&orekit.rtn_residuals_m) {
            let difference = ((sidereon.radial_m - oracle[0]).powi(2)
                + (sidereon.along_m - oracle[1]).powi(2)
                + (sidereon.cross_m - oracle[2]).powi(2))
            .sqrt();
            max_res_diff_m = max_res_diff_m.max(difference);
            assert!(
                difference <= RESIDUAL_DIFFERENCE_M,
                "{label}: residual at {} differs by {difference:e} m",
                sidereon.epoch_tdb_j2000_s
            );
        }
        eprintln!(
            "OD case {label}: iters={}, cond={:.3e}, max_est_diff={:.2e} sigma ({max_sigma_name}), max_res_diff={:.2e} m, vf_rel={:.2e}, max_sigma_rel={:.2e}, max_corr_diff={:.2e}, rms_3d={:.4} m",
            fit.iterations,
            fit.condition_number,
            max_sigma_diff,
            max_res_diff_m,
            vf_rel,
            max_sigma_rel,
            max_corr_diff,
            fit.residual_stats.rms_3d_m
        );
        // The RMS of two residual series differs by at most the RMS of their
        // difference, which the per-epoch bound bounds.
        let stats = fit.residual_stats;
        for (sidereon, oracle) in [stats.radial_rms_m, stats.along_rms_m, stats.cross_rms_m]
            .into_iter()
            .zip(orekit.rtn_rms_m)
        {
            assert!(
                (sidereon - oracle).abs() <= RESIDUAL_DIFFERENCE_M,
                "{label}: RTN RMS {sidereon} vs {oracle}"
            );
        }
    }
}

/// TDB seconds past J2000 of an SP3 epoch on the 5-minute grid of
/// 2020-06-25 GPST, converted as the fit converts it.
fn grid_epoch_tdb(index: usize) -> f64 {
    let seconds = 300 * index as i32;
    let (day, second_of_day) = (seconds / 86_400, seconds % 86_400);
    let scales = TimeScales::from_scale(
        TimeScale::Gpst,
        2020,
        6,
        25 + day,
        second_of_day / 3600,
        (second_of_day % 3600) / 60,
        0.0,
    )
    .expect("GPST");
    j2000_seconds_from_split(scales.jd_whole, scales.tdb_fraction)
}

fn first_epoch_tdb() -> f64 {
    grid_epoch_tdb(0)
}

fn utc_tdb_seconds(utc: &str) -> f64 {
    let date: Vec<i32> = utc[..10]
        .split('-')
        .map(|part| part.parse().expect("date"))
        .collect();
    let scales = TimeScales::from_utc(date[0], date[1], date[2], 0, 0, 0.0).expect("UTC");
    j2000_seconds_from_split(scales.jd_whole, scales.tdb_fraction)
}

#[test]
fn the_sp3_path_agrees_within_the_frame_difference() {
    let fixture = fixture();
    let samples = inertial_samples(&fixture);
    // sidereon's frame chain with the pole coordinates Orekit read from the
    // same orekit-data; UT1 comes from sidereon's own table.
    let polar_motion = fixture["earth_orientation"]
        .as_array()
        .expect("earth orientation")
        .iter()
        .map(|entry| {
            PolarMotionSample::from_arcseconds(
                utc_tdb_seconds(entry["utc"].as_str().expect("utc")),
                f(&entry["xp_arcsec"]),
                f(&entry["yp_arcsec"]),
            )
            .expect("polar motion")
        })
        .collect();
    let provider = PolarMotionSeriesEarthOrientationProvider::new(polar_motion).expect("series");

    // The observations of the SP3 path differ from the fixture's by the
    // difference of the two frame chains.
    let observations = fixture["observations"].as_array().expect("observations");
    let mut sum_squares = 0.0;
    let mut largest: f64 = 0.0;
    for (index, observation) in observations.iter().enumerate() {
        let orientation = provider
            .orientation_at_tdb_seconds(grid_epoch_tdb(index))
            .expect("orientation");
        let itrf = vector(&observation["itrf_m"]);
        let gcrf_km = orientation
            .itrf_to_gcrf_position_km([itrf[0] / 1.0e3, itrf[1] / 1.0e3, itrf[2] / 1.0e3])
            .expect("rotation");
        let orekit = vector(&observation["gcrf_m"]);
        let difference = (0..3)
            .map(|k| (gcrf_km[k] * 1.0e3 - orekit[k]).powi(2))
            .sum::<f64>();
        sum_squares += difference;
        largest = largest.max(difference.sqrt());
    }
    let frame_norm = sum_squares.sqrt();
    // Orekit's IERS 2010 ITRF adds the sub-daily ocean-tide and libration
    // terms of the pole and UT1 and the celestial pole offsets dX, dY, which
    // sidereon's chain does not apply: on this arc the two rotations of an SP3
    // position differ by up to 0.14 m, up to 0.12 m of it the sub-daily terms;
    // without them the chains agree within 6 cm. The bound below holds for
    // whatever the difference is; these figures keep the test from passing
    // with a frame chain gone wrong.
    assert!(largest < 0.2, "frame difference {largest} m");
    assert!(frame_norm < 2.0, "frame difference norm {frame_norm} m");

    let product = product();
    for case in fixture["cases"].as_array().expect("cases") {
        let estimated = case_names(case);
        let inertial = inertial_fit(&fixture, &samples, &estimated);
        let guess = &fixture["initial_guess"];
        let km = |m: [f64; 3]| [m[0] / 1.0e3, m[1] / 1.0e3, m[2] / 1.0e3];
        let seed = CartesianState::new(
            first_epoch_tdb(),
            km(vector(&guess["position_m"])),
            km(vector(&guess["velocity_m_s"])),
        );
        let sp3 = fit_sp3_ecef_orbit_parameters(
            &product,
            g03(),
            &provider,
            Some(seed),
            &options(&estimated),
        )
        .expect("SP3 fit");
        assert_eq!(sp3.parameter_names, inertial.parameter_names);
        let (covariance, variance_factor) = sidereon_covariance(&inertial);
        let sigmas = formal_sigmas(&covariance, variance_factor);
        let a = sidereon_estimate(&sp3);
        let b = sidereon_estimate(&inertial);
        let orekit = orekit_case(case, &inertial.parameter_names);
        for (k, name) in inertial.parameter_names.iter().enumerate() {
            // First-order bound of the move an observation change of norm
            // `frame_norm` (1 m a-priori sigma) causes, plus both fits'
            // convergence threshold of 1e-4 formal standard deviations. The
            // second-order term is smaller by the ratio of the change to the
            // scale on which the model is nonlinear, kilometres.
            let bound = sigmas[k] * (frame_norm + 2.0e-4);
            let moved = a[k] - b[k];
            assert!(
                moved.abs() <= bound,
                "{name}: SP3 path moved {moved:e}, bound {bound:e}"
            );
            assert!(
                (a[k] - orekit.estimate[k]).abs() <= bound + ESTIMATE_DIFFERENCE_SIGMAS * sigmas[k],
                "{name}: SP3 path {:e} vs Orekit {:e}",
                a[k],
                orekit.estimate[k]
            );
        }
        // The minimum residual norm changes by at most the norm of the
        // observation change: the RMS by at most its RMS.
        let frame_rms = frame_norm / (samples.len() as f64).sqrt();
        assert!(
            (sp3.residual_stats.rms_3d_m - inertial.residual_stats.rms_3d_m).abs() <= frame_rms,
            "RMS {} vs {}",
            sp3.residual_stats.rms_3d_m,
            inertial.residual_stats.rms_3d_m
        );
    }
}
