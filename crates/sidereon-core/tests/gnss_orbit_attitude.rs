#![cfg(sidereon_repo_tests)]

use sidereon_core::astro::attitude::gnss::{
    gnss_attitude, AttitudeSample, AttitudeTrajectory, GnssAttitudeError, GnssAttitudeLaw,
    OrbitAttitudeTrajectory,
};
use sidereon_core::astro::bodies::sun_moon::sun_moon_gcrs_at_tdb_seconds;
use sidereon_core::astro::de::DeEphemeris;
use sidereon_core::astro::frames::{
    EarthOrientationProvider, Iers2010EarthOrientationProvider, Iers2010EopSample,
    TdbEarthOrientationProvider,
};
use sidereon_core::astro::propagator::analytic::{
    BrouwerLyddanePropagator, CircularElements, EcksteinHechlerPropagator, KeplerianElements,
    ZonalHarmonics,
};
use sidereon_core::astro::propagator::{
    ForceModelKind, IntegratorKind, IntegratorOptions, StatePropagator,
};
use sidereon_core::astro::state::CartesianState;
use sidereon_core::astro::time::{
    j2000_seconds, j2000_seconds_from_split, tdb_from_tt_j2000_seconds, tt_from_tdb_j2000_seconds,
    Instant, TimeScale, TimeScales,
};
use sidereon_core::ephemeris::{BroadcastEphemeris, Sp3};
use sidereon_core::observables::ObservableEphemerisSource;
use sidereon_core::orbit_scoring::{
    BroadcastOrbit, OrbitTrajectory, PredictedSp3, TrajectorySample, MIN_VELOCITY_HALF_STEP_S,
};
use sidereon_core::GnssSatelliteId;

fn gps_03() -> GnssSatelliteId {
    "G03".parse().unwrap()
}

fn gps_to_tdb(gps: f64) -> f64 {
    tdb_from_tt_j2000_seconds(gps + 51.184)
}

fn assert_sample_bits(actual: AttitudeSample, expected: AttitudeSample) {
    assert_eq!(
        actual.position_m.map(f64::to_bits),
        expected.position_m.map(f64::to_bits)
    );
    assert_eq!(
        actual.velocity_m_s.map(f64::to_bits),
        expected.velocity_m_s.map(f64::to_bits)
    );
    assert_eq!(
        actual.sun_position_m.map(f64::to_bits),
        expected.sun_position_m.map(f64::to_bits)
    );
}

fn inertial_sample(state: CartesianState, tdb: f64) -> AttitudeSample {
    AttitudeSample {
        position_m: <[f64; 3]>::from(state.position_km).map(|x| x * 1000.0),
        velocity_m_s: <[f64; 3]>::from(state.velocity_km_s).map(|x| x * 1000.0),
        sun_position_m: sun_moon_gcrs_at_tdb_seconds(tdb).unwrap().sun,
    }
}

fn eckstein_hechler() -> EcksteinHechlerPropagator {
    EcksteinHechlerPropagator::from_mean_elements(
        0.0,
        CircularElements {
            a_km: 26_560.0,
            ex: 0.001,
            ey: 0.002,
            i_rad: 0.95,
            raan_rad: 0.3,
            mean_latitude_argument_rad: 1.7,
        },
        &ZonalHarmonics::earth(),
    )
    .unwrap()
}

#[test]
fn analytic_samples_match_propagated_states() {
    let eckstein = eckstein_hechler();
    let brouwer = BrouwerLyddanePropagator::from_mean_elements(
        0.0,
        KeplerianElements {
            a_km: 26_560.0,
            e: 0.03,
            i_rad: 0.95,
            raan_rad: 0.6,
            arg_perigee_rad: 0.9,
            mean_anomaly_rad: 1.7,
        },
        &ZonalHarmonics::earth(),
        0.0,
    )
    .unwrap();
    let eckstein_trajectory = OrbitAttitudeTrajectory::from_eckstein_hechler(&eckstein);
    let brouwer_trajectory = OrbitAttitudeTrajectory::from_brouwer_lyddane(&brouwer);
    for tdb in [-1800.75, 8641.25] {
        assert_sample_bits(
            eckstein_trajectory.sample(tdb).unwrap(),
            inertial_sample(eckstein.propagate(tdb).unwrap(), tdb),
        );
        assert_sample_bits(
            brouwer_trajectory.sample(tdb).unwrap(),
            inertial_sample(brouwer.propagate(tdb).unwrap(), tdb),
        );
    }
    assert!(eckstein_trajectory.sample(f64::NAN).is_none());
}

#[test]
fn numerical_samples_use_the_continuous_propagation_arc() {
    let start = 646_315_251.184;
    let mut options = IntegratorOptions::default();
    options.dense_output = true;
    let integrator = StatePropagator::new(
        start,
        [26_560.0, 0.0, 0.0],
        [0.0, 2.5, 2.9],
        ForceModelKind::two_body(),
        IntegratorKind::Dp54,
    )
    .with_options(options);
    for end in [start - 1800.0, start + 1800.0] {
        let result = integrator.propagate_to(end).unwrap();
        let dense = result.dense.as_ref().unwrap();
        let trajectory = OrbitAttitudeTrajectory::from_dense_output(dense);
        let segment = &dense.segments[0];
        let interior = segment.t_start + 0.375 * segment.h;
        assert!(interior > start.min(end) && interior < start.max(end));
        assert!(dense.segments.iter().all(|s| interior != s.t_end()));
        for tdb in [start, interior, end] {
            assert_sample_bits(
                trajectory.sample(tdb).unwrap(),
                inertial_sample(dense.eval(tdb).unwrap(), tdb),
            );
        }
        assert!(trajectory.sample(start.min(end).next_down()).is_none());
        assert!(trajectory.sample(start.max(end).next_up()).is_none());
    }
}

fn earth_fixed_sample(
    provider: &dyn EarthOrientationProvider,
    tdb: f64,
    position_m: [f64; 3],
    velocity_m_s: [f64; 3],
) -> AttitudeSample {
    let orientation = provider.orientation_at_tdb_seconds(tdb).unwrap();
    let (r, v) = orientation
        .itrf_to_gcrf_state_km(
            position_m.map(|x| x / 1000.0),
            velocity_m_s.map(|x| x / 1000.0),
        )
        .unwrap();
    AttitudeSample {
        position_m: r.map(|x| x * 1000.0),
        velocity_m_s: v.map(|x| x * 1000.0),
        sun_position_m: sun_moon_gcrs_at_tdb_seconds(tdb).unwrap().sun,
    }
}

#[test]
fn broadcast_samples_match_navigation_positions_and_orientation() {
    let nav = BroadcastEphemeris::from_nav(include_str!(
        "fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
    ))
    .unwrap();
    let sat = gps_03();
    let gps = j2000_seconds(2020, 6, 25, 18, 10, 0.25);
    let tdb = gps_to_tdb(gps);
    let orientation = TdbEarthOrientationProvider::new();
    let position = |t| {
        nav.observable_state_at_j2000_s(sat, t)
            .unwrap()
            .position_ecef_m
    };
    let before = position(gps - 0.5);
    let after = position(gps + 0.5);
    let velocity = std::array::from_fn(|i| after[i] - before[i]);
    let expected = earth_fixed_sample(&orientation, tdb, position(gps), velocity);
    let broadcast = BroadcastOrbit::new(&nav, Instant::from_nanos(TimeScale::Gpst, 0)).unwrap();
    for trajectory in [
        OrbitAttitudeTrajectory::from_broadcast(&nav, sat, &orientation),
        OrbitAttitudeTrajectory::from_orbit_source(&broadcast, sat, &orientation),
    ] {
        assert_sample_bits(trajectory.sample(tdb).unwrap(), expected);
        assert!(trajectory.sample(tdb + 14.0 * 86_400.0).is_none());
    }
}

#[test]
fn discrete_utc_samples_use_the_source_axis_and_neighbours() {
    let utc = j2000_seconds(2020, 6, 25, 12, 0, 0.0);
    let satellite = gps_03();
    let tdb = gps_to_tdb(utc + 18.0);
    let epoch =
        |shift| Instant::from_nanos(TimeScale::Utc, (646_358_400_i128 + shift) * 1_000_000_000);
    let positions = [
        [21_002_000.0, -15_002_000.0, 8_001_000.0],
        [21_000_000.0, -15_000_000.0, 8_000_000.0],
        [20_996_000.0, -14_997_000.0, 7_998_000.0],
    ];
    let source = OrbitTrajectory::new(
        "ITRF",
        epoch(0),
        [-2, 0, 2]
            .into_iter()
            .zip(positions)
            .map(|(shift, position_ecef_m)| TrajectorySample {
                satellite,
                epoch: epoch(shift),
                position_ecef_m,
                clock_s: None,
            }),
    )
    .unwrap();
    let provider = TdbEarthOrientationProvider::new();
    let with_default_step =
        OrbitAttitudeTrajectory::from_orbit_source(&source, satellite, &provider);
    assert!(with_default_step.sample(tdb).is_none());
    assert!(with_default_step
        .with_velocity_half_step_s(MIN_VELOCITY_HALF_STEP_S / 2.0)
        .is_err());
    let trajectory = OrbitAttitudeTrajectory::from_orbit_source(&source, satellite, &provider)
        .with_velocity_half_step_s(2.0)
        .unwrap();
    assert_sample_bits(
        trajectory.sample(tdb).unwrap(),
        earth_fixed_sample(&provider, tdb, positions[1], [-1500.0, 1250.0, -750.0]),
    );
    let first = gps_to_tdb(utc + 16.0);
    assert_sample_bits(
        trajectory.sample(first).unwrap(),
        earth_fixed_sample(&provider, first, positions[0], [-1000.0, 1000.0, -500.0]),
    );
    assert!(trajectory.sample(tdb + 1.0).is_none());
}

#[test]
fn sp3_samples_and_attitude_match_an_independent_trajectory() {
    let product = Sp3::parse(include_bytes!(
        "fixtures/sp3/COD0MGXFIN_20201770000_01D_05M_ORB.SP3"
    ))
    .unwrap();
    let sat = gps_03();
    let provider = Iers2010EarthOrientationProvider::new(vec![
        Iers2010EopSample {
            mjd_utc: 59024,
            xp_arcsec: 0.153959,
            yp_arcsec: 0.435032,
            ut1_minus_utc_s: -0.2435776,
            lod_s: -0.0010683,
            dx_arcsec: 0.000204,
            dy_arcsec: -0.000124,
            pole_rates_rad_s: Some([8.03159701652918e-14, -3.5977589336964334e-14]),
        },
        Iers2010EopSample {
            mjd_utc: 59025,
            xp_arcsec: 0.155398,
            yp_arcsec: 0.434469,
            ut1_minus_utc_s: -0.2426081,
            lod_s: -0.00086,
            dx_arcsec: 0.000224,
            dy_arcsec: -0.000140,
            pole_rates_rad_s: Some([7.995123765056706e-14, -3.3415109618138886e-14]),
        },
        Iers2010EopSample {
            mjd_utc: 59026,
            xp_arcsec: 0.156921,
            yp_arcsec: 0.433871,
            ut1_minus_utc_s: -0.2418658,
            lod_s: -0.0006103,
            dx_arcsec: 0.000245,
            dy_arcsec: -0.000157,
            pole_rates_rad_s: Some([1.0083451214743489e-13, -3.148857377113773e-14]),
        },
    ])
    .unwrap();
    let position = |gps| {
        Some(
            product
                .position_at_j2000_seconds(sat, gps)
                .ok()?
                .position
                .as_array(),
        )
    };
    let closure = |tdb| {
        let gps = tt_from_tdb_j2000_seconds(tdb) - 51.184;
        let before = position(gps - 0.5)?;
        let after = position(gps + 0.5)?;
        let velocity =
            std::array::from_fn(|i| (after[i] - before[i]) / ((gps + 0.5) - (gps - 0.5)));
        Some(earth_fixed_sample(&provider, tdb, position(gps)?, velocity))
    };
    let gps = j2000_seconds(2020, 6, 25, 12, 0, 0.0);
    let tdb = gps_to_tdb(gps);
    let scored = PredictedSp3::with_cutoff(&product, product.epochs[0]).unwrap();
    let trajectory = OrbitAttitudeTrajectory::from_sp3(&product, sat, &provider);
    let scored_trajectory = OrbitAttitudeTrajectory::from_orbit_source(&scored, sat, &provider);
    for t in [tdb, tdb + 6.0 * 3600.0 + 123.25] {
        let expected = closure(t).unwrap();
        assert_sample_bits(trajectory.sample(t).unwrap(), expected);
        assert_sample_bits(scored_trajectory.sample(t).unwrap(), expected);
        assert_eq!(
            gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &trajectory, t).unwrap(),
            gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &closure, t).unwrap()
        );
    }

    let t_turn = tdb + 6.0 * 3600.0;
    let law = GnssAttitudeLaw::GpsIIR {
        yaw_rate_rad_s: 0.2_f64.to_radians(),
    };
    assert_eq!(
        gnss_attitude(&law, &trajectory, t_turn).unwrap(),
        gnss_attitude(&law, &closure, t_turn).unwrap()
    );

    // G03 at 12:00 GPST. The SP3 record gives the terrestrial position.
    for (actual, expected) in
        position(gps)
            .unwrap()
            .into_iter()
            .zip([1_812_402.264, -15_421_072.636, -21_621_927.82])
    {
        assert!((actual - expected).abs() < 1.0e-8);
    }
    // The inertial references are the GCRF position and the five-point
    // derivative of the 5-minute GCRF positions in the Orekit reference arc.
    let sample = trajectory.sample(tdb).unwrap();
    let reference_position = [15_227_209.467735, 2_817_233.821536, -21_651_728.136229];
    let reference_velocity = [-1381.92698, 3575.08041, -497.86548];
    for i in 0..3 {
        assert!((sample.position_m[i] - reference_position[i]).abs() < 0.001);
        assert!((sample.velocity_m_s[i] - reference_velocity[i]).abs() < 0.02);
    }

    let outside = gps_to_tdb(product.epochs_j2000_seconds()[0] - 600.0);
    assert!(provider.orientation_at_tdb_seconds(outside).is_ok());
    assert!(trajectory.sample(outside).is_none());
    assert_eq!(
        gnss_attitude(&GnssAttitudeLaw::OrbitNormal, &trajectory, outside),
        Err(GnssAttitudeError::TrajectoryUnavailable { t_s: outside })
    );
}

#[test]
fn de_sun_matches_ephemeris_and_requires_coverage() {
    let de = DeEphemeris::from_bytes(include_bytes!("fixtures/bodies/de440/lnxp1990.440")).unwrap();
    let propagator = eckstein_hechler();
    let analytic = OrbitAttitudeTrajectory::from_eckstein_hechler(&propagator);
    let trajectory =
        OrbitAttitudeTrajectory::from_eckstein_hechler(&propagator).with_de_ephemeris(&de);
    let tdb = 646_358_451.25;
    let mut reference = inertial_sample(propagator.propagate(tdb).unwrap(), tdb);
    let sun = de.sun_moon_at_tdb_seconds(tdb).unwrap().sun.position_m;
    assert_ne!(reference.sun_position_m, sun);
    reference.sun_position_m = sun;
    assert_sample_bits(trajectory.sample(tdb).unwrap(), reference);
    let outside = de.coverage_tdb_seconds().0 - 1.0;
    assert!(analytic.sample(outside).is_some());
    assert!(trajectory.sample(outside).is_none());
}

#[test]
fn sp3_samples_require_earth_orientation_coverage() {
    let product = Sp3::parse(include_bytes!(
        "fixtures/sp3/IGS0OPSFIN_20261330000_03H_15M_ORB.SP3"
    ))
    .unwrap();
    let gpst = product.epochs_j2000_seconds()[5];
    let tdb = gps_to_tdb(gpst);
    let orientation = TdbEarthOrientationProvider::new();
    let sat = gps_03();
    let covered = OrbitAttitudeTrajectory::from_sp3(&product, sat, &orientation);
    assert!(covered.sample(tdb).is_some());

    let scales = TimeScales::from_utc(2020, 6, 25, 12, 0, 0.0).unwrap();
    let old = j2000_seconds_from_split(scales.jd_whole, scales.tdb_fraction);
    let samples = [old, old + 86_400.0]
        .into_iter()
        .map(|time| {
            sidereon_core::astro::frames::PolarMotionSample::from_radians(time, 0.0, 0.0).unwrap()
        })
        .collect();
    let orientation =
        sidereon_core::astro::frames::PolarMotionSeriesEarthOrientationProvider::new(samples)
            .unwrap();
    let uncovered = OrbitAttitudeTrajectory::from_sp3(&product, sat, &orientation);
    assert!(uncovered.sample(tdb).is_none());
}
