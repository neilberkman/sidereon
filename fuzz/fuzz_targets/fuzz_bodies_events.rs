#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::{
    bodies::sun_moon,
    conjunction::{self, ConjunctionState, PcMethod},
    events::{self, eclipse, root, EventFinder},
    passes::{self, GroundStation, UtcInstant},
    sgp4::ElementSet,
    tca::{self, TcaCandidate, TcaPcOptions},
    time::TimeScales,
};

#[derive(Debug, Arbitrary)]
struct Input {
    ts: [f64; 7],
    sat: [f64; 3],
    sun: [f64; 3],
    r1: [f64; 3],
    v1: [f64; 3],
    r2: [f64; 3],
    v2: [f64; 3],
    cov1: [[f64; 3]; 3],
    cov2: [[f64; 3]; 3],
    scalars: [f64; 12],
    ints: [i64; 4],
    bytes: [u8; 4],
    sgp4: [f64; 9],
}

fn time_scales(raw: [f64; 7]) -> TimeScales {
    TimeScales {
        jd_whole: raw[0],
        ut1_fraction: raw[1],
        tt_fraction: raw[2],
        tdb_fraction: raw[3],
        jd_ut1: raw[4],
        jd_tt: raw[5],
        jd_tdb: raw[6],
    }
}

fn pc_method(byte: u8) -> PcMethod {
    match byte % 3 {
        0 => PcMethod::FosterEqualArea,
        1 => PcMethod::FosterNumerical,
        _ => PcMethod::Alfano2005,
    }
}

fn element_set(input: &Input) -> ElementSet {
    ElementSet {
        epoch: sidereon_core::astro::sgp4::JulianDate(input.ints[0] as f64, input.scalars[0]),
        bstar: input.sgp4[0],
        mean_motion_dot: input.sgp4[1],
        mean_motion_double_dot: input.sgp4[2],
        eccentricity: input.sgp4[3],
        argument_of_perigee_deg: input.sgp4[4],
        inclination_deg: input.sgp4[5],
        mean_anomaly_deg: input.sgp4[6],
        mean_motion_rev_per_day: input.sgp4[7],
        right_ascension_deg: input.sgp4[8],
        catalog_number: input.ints[1] as u32,
    }
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };
    let ts = time_scales(input.ts);

    assert_ok_finite_or_err(
        "bodies::sun_moon_eci",
        sun_moon::sun_moon_eci(input.scalars[0]),
    );
    assert_ok_finite_or_err("bodies::sun_moon_eci_at", sun_moon::sun_moon_eci_at(&ts));
    assert_ok_finite_or_err("bodies::sun_moon_ecef", sun_moon::sun_moon_ecef(&ts));

    assert_ok_finite_or_err(
        "events::eclipse::shadow_fraction",
        eclipse::shadow_fraction(input.sat, input.sun),
    );
    let _ = eclipse::status(input.sat, input.sun);

    let start = bounded_abs_or_raw(input.scalars[1], 1_000.0);
    let span = bounded_positive_or_raw(input.scalars[2], 1.0, 1_000.0);
    let step = bounded_positive_or_raw(input.scalars[3], 1.0, 120.0);
    let tol = bounded_positive_or_raw(input.scalars[4], 1.0e-6, 10.0);
    if let Ok(finder) = EventFinder::new(start, start + span, step, tol) {
        let freq = bounded_abs_or_raw(input.scalars[5], 10.0);
        let phase = input.scalars[6];
        let threshold = input.scalars[7];
        let predicate = |t: f64| (t * freq + phase).sin();
        assert_ok_finite_or_err(
            "EventFinder::find_crossings",
            finder.find_crossings(predicate, threshold),
        );
        assert_ok_finite_or_err("EventFinder::find_extrema", finder.find_extrema(predicate));
        assert_ok_finite_or_err(
            "EventFinder::find_state_changes",
            finder.find_state_changes(|t| predicate(t) >= threshold),
        );
    }
    let root_low = input.scalars[8];
    let root_high = input.scalars[9];
    let root_shift = input.scalars[10];
    let _ = events::root::sign_change_bracketed(root_low, root_high);
    assert_ok_finite_or_err(
        "events::root::bisect_crossing_by_iterations",
        root::bisect_crossing_by_iterations(
            root_low,
            root_high,
            bounded_usize(input.bytes[0], 0, 16),
            |t| t - root_shift,
            |a, b| (a + b) * 0.5,
        ),
    );
    assert_ok_finite_or_err(
        "events::root::bisect_crossing_until",
        root::bisect_crossing_until(
            root_low,
            root_high,
            |t| t - root_shift,
            |a, b| (a + b) * 0.5,
            |a, b| (b - a).abs() <= tol,
        ),
    );
    assert_ok_finite_or_err(
        "events::root::try_bisect_crossing_until",
        root::try_bisect_crossing_until(
            root_low,
            root_high,
            |t| Ok::<f64, ()>(t - root_shift),
            |a, b| (a + b) * 0.5,
            |a, b| (b - a).abs() <= tol,
        ),
    );

    let frame = conjunction::encounter_frame(input.r1, input.v1, input.r2, input.v2);
    assert_ok_finite_or_err("conjunction::encounter_frame", frame);
    if let Ok(frame) = conjunction::encounter_frame(input.r1, input.v1, input.r2, input.v2) {
        assert_ok_finite_or_err(
            "conjunction::encounter_plane_covariance",
            conjunction::encounter_plane_covariance(&frame, &input.cov1),
        );
    }
    let object1 = ConjunctionState {
        position_km: input.r1,
        velocity_km_s: input.v1,
        covariance_km2: input.cov1,
    };
    let object2 = ConjunctionState {
        position_km: input.r2,
        velocity_km_s: input.v2,
        covariance_km2: input.cov2,
    };
    assert_ok_finite_or_err(
        "conjunction::collision_probability",
        conjunction::collision_probability(
            &object1,
            &object2,
            input.scalars[11],
            pc_method(input.bytes[1]),
        ),
    );

    let candidate = TcaCandidate {
        tca_time: sidereon_core::astro::sgp4::JulianDate(input.scalars[0], input.scalars[1]),
        tca_seconds_since_window_start: input.scalars[2],
        miss_distance_km: input.scalars[3],
        relative_position_km: input.r1,
        relative_velocity_km_s: input.v1,
    };
    assert_ok_finite_or_err(
        "tca::tca_collision_probability",
        tca::tca_collision_probability(
            candidate,
            TcaPcOptions::with_covariances(
                input.scalars[11],
                pc_method(input.bytes[2]),
                input.cov1,
                input.cov2,
            ),
        ),
    );

    let ground = GroundStation {
        latitude_deg: input.scalars[5],
        longitude_deg: input.scalars[6],
        altitude_m: input.scalars[7],
    };
    let instant = UtcInstant::from_unix_microseconds(input.ints[2]);
    assert_ok_finite_or_err(
        "passes::look_angle",
        passes::look_angle(&element_set(&input), ground, instant),
    );
});
