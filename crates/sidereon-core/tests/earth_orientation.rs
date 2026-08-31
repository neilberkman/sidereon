//! Phase B frame-chain validation.
//!
//! Oracles and references:
//! - GCRF/ITRF matrices reuse the pinned Skyfield/ERFA frame constants already
//!   committed in the frame observation tests, sourced from IERS Conventions
//!   (2010) TN36 and IAU 2006/2000A through the existing transform substrate.
//! - The SP3 bridge oracle is an independently assembled public transform path:
//!   existing ITRF->GCRS position transform plus the same evaluated DCM and the
//!   IERS Earth rotation transport term.
//! - The velocity transport check finite-differences rotated positions about
//!   the Earth rotation axis with the evaluated DCM held fixed.

#![allow(clippy::excessive_precision)]

use std::sync::Arc;

use sidereon_core::astro::frames::orientation::{EarthOrientation, TdbEarthOrientationProvider};
use sidereon_core::astro::frames::transforms::{
    gcrs_to_itrs_matrix, itrs_to_gcrs_compute, mat3_vec3_mul,
};
use sidereon_core::astro::math::least_squares::SolveOptions;
use sidereon_core::astro::math::mat3::Mat3;
use sidereon_core::astro::propagator::{
    ForceModelKind, IntegratorKind, IntegratorOptions, PropagationContext, StatePropagator,
};
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::time::civil::{j2000_seconds, split_julian_date_from_j2000_seconds};
use sidereon_core::astro::time::model::{Instant, JulianDateSplit, TimeScale};
use sidereon_core::astro::time::scales::TimeScales;
use sidereon_core::ephemeris::{
    fit_precise_ephemeris_state_sample_orbit, sp3_ecef_state_to_eci, OrbitFitOptions,
    OrientedPreciseEphemerisStateSample, PreciseEphemerisStateSample,
};
use sidereon_core::geometry_quality::ObservabilityTier;
use sidereon_core::{GnssSatelliteId, GnssSystem};

type FrameGoldenCase = ((i32, i32, i32, i32, i32, f64), Mat3);

const FRAME_GOLDEN: &[FrameGoldenCase] = &[
    (
        (2000, 1, 1, 12, 0, 0.0),
        [
            [
                1.81585122847680802e-1,
                -9.83375229832204600e-1,
                -2.26462012731456990e-5,
            ],
            [
                9.83375229723779998e-1,
                1.81585122100325458e-1,
                3.15834393603663929e-5,
            ],
            [
                -2.69461587142590570e-5,
                -2.80047961012289743e-5,
                9.99999999244818083e-1,
            ],
        ],
    ),
    (
        (2020, 6, 24, 12, 34, 56.0),
        [
            [
                -2.01022163818197153e-1,
                9.79586611856021472e-1,
                3.99407422332429831e-4,
            ],
            [
                -9.79584737922584714e-1,
                -2.01022560513757581e-1,
                1.91608810716750451e-3,
            ],
            [
                1.95726415964563851e-3,
                -6.07723776638428277e-6,
                9.99998084538203713e-1,
            ],
        ],
    ),
    (
        (2024, 1, 1, 0, 0, 0.0),
        [
            [
                -1.70985862949848105e-1,
                9.85273414718201956e-1,
                3.64583089080899262e-4,
            ],
            [
                -9.85270747182341422e-1,
                -1.70986248483862069e-1,
                2.29294050648503125e-3,
            ],
            [
                2.32151201723513039e-3,
                3.28473585994561317e-5,
                9.99997304747870186e-1,
            ],
        ],
    ),
    (
        (2026, 7, 1, 0, 0, 0.0),
        [
            [
                1.51667472875174736e-1,
                -9.88431507364607498e-1,
                -3.64582757404158340e-4,
            ],
            [
                9.88428180113040566e-1,
                1.51667908368473991e-1,
                -2.56482544345893642e-3,
            ],
            [
                2.59044978345025550e-3,
                2.86367219681365483e-5,
                9.99996644369298804e-1,
            ],
        ],
    ),
];

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

fn fit_options(force_model: ForceModelKind) -> OrbitFitOptions {
    OrbitFitOptions {
        force_model,
        integrator: IntegratorKind::Dp54,
        integrator_options: IntegratorOptions {
            abs_tol: 1.0e-12,
            rel_tol: 1.0e-13,
            initial_step: 10.0,
            max_step: 60.0,
            ..IntegratorOptions::default()
        },
        solver_options: SolveOptions {
            gtol: 1.0e-15,
            ftol: 1.0e-15,
            xtol: 1.0e-15,
            max_nfev: 1200,
        },
        ..OrbitFitOptions::default()
    }
}

fn assert_matrix_near(label: &str, actual: &Mat3, expected: &Mat3, tolerance: f64) {
    for i in 0..3 {
        for j in 0..3 {
            let error = (actual[i][j] - expected[i][j]).abs();
            assert!(
                error <= tolerance,
                "{label}[{i}][{j}] error {error:.17e} exceeds {tolerance:.17e}"
            );
        }
    }
}

fn assert_vec_near(label: &str, actual: [f64; 3], expected: [f64; 3], tolerance: f64) {
    for i in 0..3 {
        let error = (actual[i] - expected[i]).abs();
        assert!(
            error <= tolerance,
            "{label}[{i}] error {error:.17e} exceeds {tolerance:.17e}"
        );
    }
}

fn assert_state_bits(left: CartesianState, right: CartesianState) {
    assert_eq!(
        left.epoch_tdb_seconds.to_bits(),
        right.epoch_tdb_seconds.to_bits()
    );
    for i in 0..3 {
        assert_eq!(
            left.position_array()[i].to_bits(),
            right.position_array()[i].to_bits(),
            "position axis {i}"
        );
        assert_eq!(
            left.velocity_array()[i].to_bits(),
            right.velocity_array()[i].to_bits(),
            "velocity axis {i}"
        );
    }
}

fn rotate_z(position: [f64; 3], angle: f64) -> [f64; 3] {
    let c = libm::cos(angle);
    let s = libm::sin(angle);
    [
        c * position[0] - s * position[1],
        s * position[0] + c * position[1],
        position[2],
    ]
}

#[test]
fn earth_orientation_matches_pinned_frame_matrices() {
    for (utc, expected) in FRAME_GOLDEN {
        let ts = TimeScales::from_utc(utc.0, utc.1, utc.2, utc.3, utc.4, utc.5).expect("valid UTC");
        let orientation = EarthOrientation::from_time_scales(&ts).expect("orientation");
        let matrix = orientation.gcrf_to_itrf_matrix();
        assert_matrix_near("skyfield", &matrix, expected, 1.0e-12);

        let existing = gcrs_to_itrs_matrix(&ts).expect("existing transform");
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(
                    matrix[i][j].to_bits(),
                    existing[i][j].to_bits(),
                    "matrix bit mismatch [{i}][{j}]"
                );
            }
        }
    }
}

#[test]
fn earth_orientation_round_trips_unit_vectors_to_epsilon() {
    let vectors = [
        [1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [
            0.267_261_241_912_424_4,
            -0.534_522_483_824_848_8,
            0.801_783_725_737_273_2,
        ],
    ];
    for (utc, _) in FRAME_GOLDEN {
        let orientation = EarthOrientation::from_utc(utc.0, utc.1, utc.2, utc.3, utc.4, utc.5)
            .expect("orientation");
        for vector in vectors {
            let fixed = orientation
                .gcrf_to_itrf_position_km(vector)
                .expect("GCRF to ITRF");
            let inertial = orientation
                .itrf_to_gcrf_position_km(fixed)
                .expect("ITRF to GCRF");
            assert_vec_near("round trip", inertial, vector, 6.0e-15);
        }
    }
}

#[test]
fn sp3_bridge_matches_existing_position_path_and_transport_bits() {
    let epoch = instant_at(TimeScale::Gpst, j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64);
    let orientation = EarthOrientation::from_instant(epoch).expect("orientation");
    let sample = PreciseEphemerisStateSample::new(
        gps(5),
        epoch,
        [20_156_123.25, -14_201_337.5, 21_780_044.75],
        [-122.5, 2411.25, 188.125],
        Some(3.25e-6),
        Some(-1.5e-12),
    );

    let got = sp3_ecef_state_to_eci(&sample, &orientation).expect("bridge state");
    let position_itrf_km = [
        sample.position_ecef_m[0] / 1000.0,
        sample.position_ecef_m[1] / 1000.0,
        sample.position_ecef_m[2] / 1000.0,
    ];
    let velocity_itrf_km_s = [
        sample.velocity_ecef_m_s[0] / 1000.0,
        sample.velocity_ecef_m_s[1] / 1000.0,
        sample.velocity_ecef_m_s[2] / 1000.0,
    ];
    let expected_position = itrs_to_gcrs_compute(
        position_itrf_km[0],
        position_itrf_km[1],
        position_itrf_km[2],
        &orientation.time_scales(),
    )
    .expect("existing position transform");
    let omega = orientation.earth_rotation_vector_itrf_rad_s();
    let transport = [
        omega[1] * position_itrf_km[2] - omega[2] * position_itrf_km[1],
        omega[2] * position_itrf_km[0] - omega[0] * position_itrf_km[2],
        omega[0] * position_itrf_km[1] - omega[1] * position_itrf_km[0],
    ];
    let expected_velocity = mat3_vec3_mul(
        &orientation.itrf_to_gcrf_matrix(),
        &[
            velocity_itrf_km_s[0] + transport[0],
            velocity_itrf_km_s[1] + transport[1],
            velocity_itrf_km_s[2] + transport[2],
        ],
    )
    .expect("velocity rotation");

    assert_eq!(
        got.epoch_tdb_seconds.to_bits(),
        j2000_seconds(2026, 6, 1, 0, 0, 0.0).to_bits()
    );
    for i in 0..3 {
        assert_eq!(
            got.position_array()[i].to_bits(),
            [
                expected_position.0,
                expected_position.1,
                expected_position.2
            ][i]
                .to_bits(),
            "position axis {i}"
        );
        assert_eq!(
            got.velocity_array()[i].to_bits(),
            expected_velocity[i].to_bits(),
            "velocity axis {i}"
        );
    }
}

#[test]
fn velocity_transport_matches_finite_difference_of_rotated_positions() {
    let orientation = EarthOrientation::from_utc(2026, 7, 1, 0, 0, 0.0).expect("orientation");
    let position_itrf_km = [20_156.123_25, -14_201.337_5, 21_780.044_75];
    let (_, velocity_gcrf_km_s) = orientation
        .itrf_to_gcrf_state_km(position_itrf_km, [0.0, 0.0, 0.0])
        .expect("state transform");
    let omega_z = orientation.earth_rotation_vector_itrf_rad_s()[2];
    let central_difference = |h: f64| {
        let plus = orientation
            .itrf_to_gcrf_position_km(rotate_z(position_itrf_km, omega_z * h))
            .expect("plus position");
        let minus = orientation
            .itrf_to_gcrf_position_km(rotate_z(position_itrf_km, -omega_z * h))
            .expect("minus position");
        [
            (plus[0] - minus[0]) / (2.0 * h),
            (plus[1] - minus[1]) / (2.0 * h),
            (plus[2] - minus[2]) / (2.0 * h),
        ]
    };
    let coarse = central_difference(0.2);
    let fine = central_difference(0.1);
    let finite_difference = [
        (4.0 * fine[0] - coarse[0]) / 3.0,
        (4.0 * fine[1] - coarse[1]) / 3.0,
        (4.0 * fine[2] - coarse[2]) / 3.0,
    ];
    assert_vec_near(
        "velocity finite difference",
        velocity_gcrf_km_s,
        finite_difference,
        1.0e-10,
    );
}

#[test]
fn propagation_default_bits_are_unchanged_with_unused_frame_provider() {
    let state = CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [1.0, 7.2, 0.5]);
    let propagator = StatePropagator {
        initial: state,
        force_model: ForceModelKind::two_body(),
        integrator: IntegratorKind::Rk4,
        options: IntegratorOptions {
            initial_step: 10.0,
            ..IntegratorOptions::default()
        },
        drag: None,
        space_weather: None,
    };
    let default_result = propagator.propagate_to(120.0).expect("default").final_state;
    let ctx = PropagationContext::new()
        .with_body_fixed_frame_provider(Arc::new(TdbEarthOrientationProvider::default()));
    let with_provider = propagator
        .propagate_to_with_context(120.0, &ctx)
        .expect("with provider")
        .final_state;
    assert_state_bits(default_result, with_provider);
}

#[test]
fn oriented_state_samples_feed_orbit_fit() {
    let sat = gps(9);
    let start = j2000_seconds(2026, 6, 1, 0, 0, 0.0) as i64;
    let epochs: Vec<i64> = (0..=8).map(|step| start + step * 60).collect();
    let initial = CartesianState::new(start as f64, [7078.0, -30.0, 820.0], [0.20, 7.35, 1.05]);
    let options = fit_options(ForceModelKind::two_body());
    let propagator = StatePropagator {
        initial,
        force_model: ForceModelKind::two_body(),
        integrator: IntegratorKind::Dp54,
        options: options.integrator_options,
        drag: None,
        space_weather: None,
    };
    let query_epochs: Vec<f64> = epochs.iter().map(|&epoch| epoch as f64).collect();
    let states = propagator.ephemeris(&query_epochs).expect("truth arc");
    let mut oriented = Vec::new();
    for (state, &epoch) in states.iter().zip(&epochs) {
        let instant = instant_at(TimeScale::Gpst, epoch);
        let orientation = EarthOrientation::from_instant(instant).expect("orientation");
        let (position_itrf_km, velocity_itrf_km_s) = orientation
            .gcrf_to_itrf_state_km(state.position_array(), state.velocity_array())
            .expect("GCRF to ITRF");
        let sample = PreciseEphemerisStateSample::new(
            sat,
            instant,
            [
                position_itrf_km[0] * 1000.0,
                position_itrf_km[1] * 1000.0,
                position_itrf_km[2] * 1000.0,
            ],
            [
                velocity_itrf_km_s[0] * 1000.0,
                velocity_itrf_km_s[1] * 1000.0,
                velocity_itrf_km_s[2] * 1000.0,
            ],
            None,
            None,
        );
        oriented.push(OrientedPreciseEphemerisStateSample::new(
            sample,
            orientation,
        ));
    }

    let report = fit_precise_ephemeris_state_sample_orbit(&oriented, sat, &options)
        .expect("state-sample fit");
    let fit = report.fits.get(&sat).expect("fit");
    let stats = report.ledger.per_sat.get(&sat).expect("ledger");
    assert_eq!(fit.geometry_quality.tier, ObservabilityTier::Nominal);
    assert!(
        stats.rms_3d_m < 5.0e-4,
        "state-sample RMS was {:.17e} m",
        stats.rms_3d_m
    );

    let pos_error = (fit.initial_state.position_km - initial.position_km).norm();
    let vel_error = (fit.initial_state.velocity_km_s - initial.velocity_km_s).norm();
    assert!(pos_error < 2.0e-7, "position error {pos_error:.17e} km");
    assert!(vel_error < 3.0e-9, "velocity error {vel_error:.17e} km/s");
}
