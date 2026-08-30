//! Real IGS SP3 ECEF orbit-fit validation.
//!
//! Source fixture: `IGS0OPSFIN_20261330000_03H_15M_ORB.SP3`, an in-repo IGS
//! final SP3-c orbit product parsed by the core SP3 reader. The validation uses
//! the published SP3 ECEF convention and the landed IERS/IAU frame chain through
//! `TdbEarthOrientationProvider`.
//! Additional synthetic SP3 text in this file is constructed from the public
//! two-body propagator and public frame API to guard provider semantics that the
//! real fixture does not exercise: nonzero polar motion and mixed position plus
//! partial-velocity products.
//!
//! The absolute RMS against the agency orbit is a regression measure for the
//! current force-model fidelity and EOP source, not a claim of agency-grade
//! orbit reproduction. The test pins the achieved RMS with `<=` bounds so model
//! improvements pass while regressions fail. On x86-64-linux canonical runs
//! this file pins the listed RMS values directly.

use std::sync::Arc;

use sidereon_core::astro::forces::{
    SolarRadiationPressure, SolidEarthPoleTideGravity, SolidEarthTideGravity,
};
use sidereon_core::astro::frames::orientation::{
    EarthOrientationProvider, PolarMotionSample, PolarMotionSeriesEarthOrientationProvider,
};
use sidereon_core::astro::frames::transforms::PolarMotion;
use sidereon_core::astro::math::least_squares::SolveOptions;
use sidereon_core::astro::propagator::{
    ForceModelComponents, ForceModelKind, IntegratorKind, IntegratorOptions, PropagationContext,
};
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::time::civil::{
    civil_from_j2000_seconds, j2000_seconds, j2000_seconds_from_split,
};
use sidereon_core::astro::time::model::TimeScale;
use sidereon_core::astro::time::scales::TimeScales;
use sidereon_core::ephemeris::{fit_sp3_ecef_precise_orbit, OrbitFitOptions, Sp3};
use sidereon_core::geometry_quality::ObservabilityTier;
use sidereon_core::{GnssSatelliteId, GnssSystem, TdbEarthOrientationProvider};

const IGS_FINAL_SP3: &[u8] = include_bytes!("fixtures/sp3/IGS0OPSFIN_20261330000_03H_15M_ORB.SP3");
const PINNED_ARC_START_TDB_J2000_S: f64 = 831_902_451.185_288_4;
// Re-pinned after replacing host libm calls in the SP3 frame/force path with
// portable Rust libm kernels; the independent agency SP3 regression remains
// within the same fixed headroom.
const ACHIEVED_PHASE_A_RMS_3D_M: f64 = 2.547_885_426_393_437;
const ACHIEVED_PHASE_A_FINALS2000A_RMS_3D_M: f64 = 1.194_618_055_240_168_2;
const ACHIEVED_PHASE_A_WITH_TIDES_FINALS2000A_RMS_3D_M: f64 = 1.193_361_555_626_073_7;
// 0.051 mm headroom keeps the regression bound boundary-sensitive while
// allowing ordinary floating-point and solver-path variation.
const REAL_SP3_RMS_HEADROOM_M: f64 = 5.1e-5;
// Re-pinned 2026-07-04: the TCG/TCB time-scale refinement shifted the frame
// rotation by 78 micrometers on this 311 m residual (2.5e-7 relative).
const PINNED_TWO_BODY_RMS_3D_M: f64 = 311.171_650_553_354_1;
// Six fractional SP3 kilometers give a 0.5 mm half-cell per axis.
const SYNTHETIC_SP3_POSITION_QUANTIZATION_3D_BOUND_M: f64 = 8.660_254_037_844_386e-4;

fn gps(prn: u8) -> GnssSatelliteId {
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS satellite")
}

fn fit_options(force_model: ForceModelKind) -> OrbitFitOptions {
    OrbitFitOptions {
        force_model,
        integrator: IntegratorKind::Dp54,
        integrator_options: IntegratorOptions {
            abs_tol: 1.0e-11,
            rel_tol: 1.0e-13,
            initial_step: 30.0,
            max_step: 180.0,
            ..IntegratorOptions::default()
        },
        solver_options: SolveOptions {
            gtol: 1.0e-13,
            ftol: 1.0e-13,
            xtol: 1.0e-13,
            max_nfev: 900,
        },
        ..OrbitFitOptions::default()
    }
}

fn fit_options_with_provider(
    force_model: ForceModelKind,
    provider: Arc<dyn EarthOrientationProvider>,
) -> OrbitFitOptions {
    let mut options = fit_options(force_model);
    options.propagation_context =
        PropagationContext::new().with_body_fixed_frame_provider(provider);
    options
}

fn phase_a_with_tides(srp: SolarRadiationPressure) -> ForceModelKind {
    ForceModelKind::composite(
        ForceModelComponents::earth_phase_a(Some(srp))
            .with_solid_earth_tide(SolidEarthTideGravity::default())
            .with_solid_earth_pole_tide(SolidEarthPoleTideGravity::default()),
    )
}

fn gpst_tdb_seconds(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
    let scales = TimeScales::from_scale(TimeScale::Gpst, year, month, day, hour, minute, second)
        .expect("valid GPST time scales");
    j2000_seconds_from_split(scales.jd_whole, scales.tdb_fraction)
}

fn utc_tdb_seconds(year: i32, month: i32, day: i32, hour: i32, minute: i32, second: f64) -> f64 {
    let scales = TimeScales::from_utc(year, month, day, hour, minute, second)
        .expect("valid UTC time scales");
    j2000_seconds_from_split(scales.jd_whole, scales.tdb_fraction)
}

fn finals2000a_polar_motion_provider() -> Arc<dyn EarthOrientationProvider> {
    // Source rows: IERS finals2000A.all, MJD 61172-61176, columns x and y
    // polar motion in arcseconds. These rows cover the 2026-05-13 03H SP3 arc.
    let samples = vec![
        PolarMotionSample::from_arcseconds(
            utc_tdb_seconds(2026, 5, 12, 0, 0, 0.0),
            0.168_095,
            0.412_462,
        )
        .expect("polar motion sample"),
        PolarMotionSample::from_arcseconds(
            utc_tdb_seconds(2026, 5, 13, 0, 0, 0.0),
            0.169_051,
            0.411_759,
        )
        .expect("polar motion sample"),
        PolarMotionSample::from_arcseconds(
            utc_tdb_seconds(2026, 5, 14, 0, 0, 0.0),
            0.169_841,
            0.411_639,
        )
        .expect("polar motion sample"),
        PolarMotionSample::from_arcseconds(
            utc_tdb_seconds(2026, 5, 15, 0, 0, 0.0),
            0.170_524,
            0.411_749,
        )
        .expect("polar motion sample"),
        PolarMotionSample::from_arcseconds(
            utc_tdb_seconds(2026, 5, 16, 0, 0, 0.0),
            0.171_241,
            0.412_116,
        )
        .expect("polar motion sample"),
    ];
    Arc::new(PolarMotionSeriesEarthOrientationProvider::new(samples).expect("polar motion series"))
}

fn synthetic_sp3_with_partial_velocity(
    satellite: GnssSatelliteId,
    provider: &TdbEarthOrientationProvider,
) -> Sp3 {
    let gpst_start = j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64;
    let tdb_start = gpst_tdb_seconds(2026, 6, 1, 0, 0, 0.0);
    let initial = CartesianState::new(
        tdb_start,
        [20_200.0, -11_300.0, 14_400.0],
        [1.25, 2.65, 2.05],
    );
    let options = fit_options(ForceModelKind::two_body());
    let propagator = sidereon_core::astro::propagator::StatePropagator {
        initial,
        force_model: ForceModelKind::two_body(),
        integrator: IntegratorKind::Dp54,
        options: options.integrator_options,
        drag: None,
        space_weather: None,
    };

    let mut epochs = Vec::new();
    for step in 0..9 {
        let gpst_epoch = gpst_start + step * 60;
        let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(gpst_epoch);
        let tdb = gpst_tdb_seconds(
            year as i32,
            month as i32,
            day as i32,
            hour as i32,
            minute as i32,
            second as f64,
        );
        epochs.push((year, month, day, hour, minute, second, tdb));
    }
    let query_epochs: Vec<f64> = epochs.iter().map(|epoch| epoch.6).collect();
    let states = propagator
        .ephemeris(&query_epochs)
        .expect("truth ephemeris");

    let mut text = "\
#dV2026  6  1  0  0  0.00000000       9 ORBIT IGS20 FIT  TST
## 2421  86400.00000000    60.00000000 61192 0.0000000000000
+    1   G09  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* PUBLIC TWO-BODY AND FRAME API REGRESSION
"
    .to_string();

    for (index, (state, &(year, month, day, hour, minute, second, tdb))) in
        states.iter().zip(&epochs).enumerate()
    {
        let orientation = provider
            .orientation_at_tdb_seconds(tdb)
            .expect("orientation for synthetic epoch");
        let (position_itrf_km, velocity_itrf_km_s) = orientation
            .gcrf_to_itrf_state_km(state.position_array(), state.velocity_array())
            .expect("state to ITRF");
        text.push_str(&format!(
            "*  {year:4} {month:2} {day:2} {hour:2} {minute:2} {second:11.8}\n"
        ));
        text.push_str(&format!(
            "P{sat}{x:14.6}{y:14.6}{z:14.6}{clock:14.6}\n",
            sat = satellite,
            x = position_itrf_km[0],
            y = position_itrf_km[1],
            z = position_itrf_km[2],
            clock = 0.0,
        ));
        if index == 0 {
            text.push_str(&format!(
                "V{sat}{vx:14.6}{vy:14.6}{vz:14.6}{rate:14.6}\n",
                sat = satellite,
                vx = velocity_itrf_km_s[0] * 10_000.0,
                vy = velocity_itrf_km_s[1] * 10_000.0,
                vz = velocity_itrf_km_s[2] * 10_000.0,
                rate = 1.0,
            ));
        }
    }
    text.push_str("EOF\n");
    Sp3::parse(text.as_bytes()).expect("parse synthetic mixed SP3")
}

#[test]
fn real_igs_sp3_ecef_fit_converges_and_phase_a_improves_rms() {
    let product = Sp3::parse(IGS_FINAL_SP3).expect("parse IGS SP3 fixture");
    let satellite = gps(1);
    let provider = TdbEarthOrientationProvider::default();
    let srp = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP parameters");

    let two_body = fit_sp3_ecef_precise_orbit(
        &product,
        satellite,
        &provider,
        &fit_options(ForceModelKind::two_body()),
    )
    .expect("two-body real SP3 fit converges");
    let phase_a = fit_sp3_ecef_precise_orbit(
        &product,
        satellite,
        &provider,
        &fit_options(ForceModelKind::earth_phase_a(Some(srp))),
    )
    .expect("Phase A real SP3 fit converges");

    let two_body_fit = two_body.fits.get(&satellite).expect("two-body fit");
    let phase_a_fit = phase_a.fits.get(&satellite).expect("Phase A fit");
    let two_body_stats = two_body
        .ledger
        .per_sat
        .get(&satellite)
        .expect("two-body satellite ledger");
    let phase_a_stats = phase_a
        .ledger
        .per_sat
        .get(&satellite)
        .expect("Phase A satellite ledger");
    let constellation_stats = phase_a
        .ledger
        .per_constellation
        .get(&GnssSystem::Gps)
        .expect("GPS constellation ledger");

    assert_eq!(phase_a_stats.n, 13);
    assert_eq!(constellation_stats.n, phase_a_stats.n);
    assert_eq!(phase_a.ledger.arc_span.time_scale, TimeScale::Tdb);
    assert_eq!(
        phase_a.ledger.arc_span.start_j2000_s.to_bits(),
        PINNED_ARC_START_TDB_J2000_S.to_bits()
    );
    assert!(!phase_a_stats.low_sample_count);
    assert!(!constellation_stats.low_sample_count);
    assert_eq!(
        phase_a_fit.geometry_quality.tier,
        ObservabilityTier::Nominal
    );
    assert!(
        two_body_stats.rms_3d_m > phase_a_stats.rms_3d_m,
        "Phase A RMS {:.17e} m did not improve on two-body {:.17e} m",
        phase_a_stats.rms_3d_m,
        two_body_stats.rms_3d_m
    );
    assert!(
        phase_a_fit.fit_rms_3d_m <= ACHIEVED_PHASE_A_RMS_3D_M + REAL_SP3_RMS_HEADROOM_M,
        "Phase A fit RMS was {:.17e} m",
        phase_a_fit.fit_rms_3d_m
    );
    assert!(
        phase_a_stats.rms_3d_m <= ACHIEVED_PHASE_A_RMS_3D_M + REAL_SP3_RMS_HEADROOM_M,
        "Phase A ledger RMS was {:.17e} m",
        phase_a_stats.rms_3d_m
    );
    assert!(
        two_body_fit.fit_rms_3d_m <= PINNED_TWO_BODY_RMS_3D_M,
        "two-body fit RMS was {:.17e} m",
        two_body_fit.fit_rms_3d_m
    );
    assert!(
        two_body_stats.rms_3d_m <= PINNED_TWO_BODY_RMS_3D_M,
        "two-body ledger RMS was {:.17e} m",
        two_body_stats.rms_3d_m
    );
}

#[test]
fn real_igs_sp3_ecef_fit_with_tides_does_not_worsen_phase_a_rms() {
    let product = Sp3::parse(IGS_FINAL_SP3).expect("parse IGS SP3 fixture");
    let satellite = gps(1);
    let provider = finals2000a_polar_motion_provider();
    let srp = SolarRadiationPressure::new(1.2, 0.02).expect("valid SRP parameters");
    let phase_a_options = fit_options_with_provider(
        ForceModelKind::earth_phase_a(Some(srp)),
        Arc::clone(&provider),
    );
    let tide_options = fit_options_with_provider(phase_a_with_tides(srp), Arc::clone(&provider));

    let phase_a =
        fit_sp3_ecef_precise_orbit(&product, satellite, provider.as_ref(), &phase_a_options)
            .expect("Phase A real SP3 fit converges");
    let with_tides =
        fit_sp3_ecef_precise_orbit(&product, satellite, provider.as_ref(), &tide_options)
            .expect("tide-enabled real SP3 fit converges");

    let phase_a_stats = phase_a
        .ledger
        .per_sat
        .get(&satellite)
        .expect("Phase A satellite ledger");
    let tide_stats = with_tides
        .ledger
        .per_sat
        .get(&satellite)
        .expect("tide satellite ledger");
    let tide_fit = with_tides.fits.get(&satellite).expect("tide fit");

    assert_eq!(phase_a_stats.n, 13);
    assert_eq!(tide_stats.n, phase_a_stats.n);
    assert_eq!(tide_fit.geometry_quality.tier, ObservabilityTier::Nominal);
    assert!(
        tide_stats.rms_3d_m <= phase_a_stats.rms_3d_m,
        "tide RMS {:.17e} m worsened Phase A {:.17e} m",
        tide_stats.rms_3d_m,
        phase_a_stats.rms_3d_m
    );
    assert!(
        phase_a_stats.rms_3d_m <= ACHIEVED_PHASE_A_FINALS2000A_RMS_3D_M + REAL_SP3_RMS_HEADROOM_M,
        "finals2000A Phase A ledger RMS was {:.17e} m",
        phase_a_stats.rms_3d_m
    );
    assert!(
        tide_stats.rms_3d_m
            <= ACHIEVED_PHASE_A_WITH_TIDES_FINALS2000A_RMS_3D_M + REAL_SP3_RMS_HEADROOM_M,
        "tide ledger RMS was {:.17e} m",
        tide_stats.rms_3d_m
    );
    assert!(
        tide_fit.fit_rms_3d_m
            <= ACHIEVED_PHASE_A_WITH_TIDES_FINALS2000A_RMS_3D_M + REAL_SP3_RMS_HEADROOM_M,
        "tide fit RMS was {:.17e} m",
        tide_fit.fit_rms_3d_m
    );
}

#[test]
fn provider_fit_keeps_mixed_position_arc_with_polar_motion() {
    let satellite = gps(9);
    let pole = PolarMotion::from_arcseconds(0.1, -0.2).expect("valid polar motion");
    let provider = TdbEarthOrientationProvider::with_polar_motion(pole);
    let product = synthetic_sp3_with_partial_velocity(satellite, &provider);

    assert_eq!(
        product
            .precise_ephemeris_samples()
            .iter()
            .filter(|sample| sample.sat == satellite)
            .count(),
        9
    );
    assert_eq!(
        product
            .precise_ephemeris_state_samples()
            .iter()
            .filter(|sample| sample.sat == satellite)
            .count(),
        1
    );

    let report = fit_sp3_ecef_precise_orbit(
        &product,
        satellite,
        &provider,
        &fit_options(ForceModelKind::two_body()),
    )
    .expect("provider fit converges on mixed SP3 arc");
    let fit = report.fits.get(&satellite).expect("fit");
    let stats = report.ledger.per_sat.get(&satellite).expect("ledger");

    assert_eq!(stats.n, 9);
    assert_eq!(report.ledger.arc_span.time_scale, TimeScale::Tdb);
    assert_eq!(fit.geometry_quality.tier, ObservabilityTier::Nominal);
    assert!(
        stats.rms_3d_m <= SYNTHETIC_SP3_POSITION_QUANTIZATION_3D_BOUND_M,
        "synthetic mixed SP3 RMS was {:.17e} m",
        stats.rms_3d_m
    );
}
