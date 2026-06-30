#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::{
    covariance::Covariance6,
    forces::TwoBodyGravity,
    integrators::{Integrator, DP54, RK4},
    propagator::{
        api::{IntegratorOptions, PropagationContext},
        ForceModelKind, IntegratorKind, OrbitalDynamics, StatePropagator,
    },
    sgp4::{ElementSet, JulianDate, MinutesSinceEpoch, OpsMode, Satellite},
    CartesianState,
};
use sidereon_core::orbit::{self, CalendarEpoch, EcefSample, Elements, Frame, Model};

#[derive(Debug, Arbitrary)]
struct Input {
    epoch_s: f64,
    pos: [f64; 3],
    vel: [f64; 3],
    t_end: f64,
    opts: [f64; 5],
    opt_bits: [u8; 5],
    cov_diag: [f64; 6],
    sgp4: [f64; 11],
    year: i32,
    day: f64,
    catalog: u32,
    orbit_scalars: [f64; 12],
    samples_xyz: Vec<[f64; 3]>,
    samples_sec: Vec<f64>,
    scale: u8,
    frame: u8,
}

fn integrator_options(input: &Input) -> IntegratorOptions {
    IntegratorOptions {
        abs_tol: input.opts[0],
        rel_tol: input.opts[1],
        min_step: input.opts[2],
        max_step: bounded_positive_or_raw(input.opts[3], 1.0, 600.0),
        initial_step: bounded_positive_or_raw(input.opts[4], 1.0, 120.0),
        max_steps: bounded_usize(input.opt_bits[0], 1, 32) as u32,
        dense_output: input.opt_bits[1] & 1 == 1,
    }
}

fn elements(input: &Input) -> ElementSet {
    ElementSet {
        epoch: JulianDate(input.year as f64, input.day),
        bstar: input.sgp4[0],
        mean_motion_dot: input.sgp4[1],
        mean_motion_double_dot: input.sgp4[2],
        eccentricity: input.sgp4[3],
        argument_of_perigee_deg: input.sgp4[4],
        inclination_deg: input.sgp4[5],
        mean_anomaly_deg: input.sgp4[6],
        mean_motion_rev_per_day: input.sgp4[7],
        right_ascension_deg: input.sgp4[8],
        catalog_number: input.catalog,
    }
}

fn calendar(input: &Input, idx: usize) -> CalendarEpoch {
    CalendarEpoch::new(
        2020 + (idx as i32),
        1,
        1,
        0,
        0,
        input
            .samples_sec
            .get(idx)
            .copied()
            .unwrap_or(input.orbit_scalars[0]),
    )
}

fn reduced_elements(input: &Input) -> Elements {
    Elements {
        model: if input.opt_bits[2] & 1 == 0 {
            Model::CircularSecular
        } else {
            Model::EccentricSecular
        },
        epoch: calendar(input, 0),
        a_m: input.orbit_scalars[0],
        e: input.orbit_scalars[1],
        i_rad: input.orbit_scalars[2],
        raan_rad: input.orbit_scalars[3],
        raan_rate_rad_s: input.orbit_scalars[4],
        raan_rate_j2_rad_s: input.orbit_scalars[5],
        arg_lat_rad: input.orbit_scalars[6],
        mean_motion_rad_s: input.orbit_scalars[7],
        h: input.orbit_scalars[8],
        k: input.orbit_scalars[9],
        arg_perigee_rad: input.orbit_scalars[10],
    }
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };

    let opts = integrator_options(&input);
    let state = CartesianState::new(input.epoch_s, input.pos, input.vel);
    let force_model = if input.opt_bits[3] & 1 == 0 {
        ForceModelKind::TwoBody {
            mu_km3_s2: input.sgp4[9],
        }
    } else {
        ForceModelKind::TwoBodyJ2 {
            mu_km3_s2: input.sgp4[9],
            re_km: input.sgp4[10],
            j2: input.orbit_scalars[11],
        }
    };
    let integrator = if input.opt_bits[4] & 1 == 0 {
        IntegratorKind::Rk4
    } else {
        IntegratorKind::Dp54
    };
    let propagator = StatePropagator {
        initial: state,
        force_model,
        integrator,
        options: opts,
    };
    assert_ok_finite_or_err(
        "StatePropagator::propagate_to",
        propagator.propagate_to(input.t_end),
    );
    assert_ok_finite_or_err(
        "StatePropagator::state_transition_matrix_for_span",
        propagator.state_transition_matrix_for_span(bounded_abs_or_raw(input.t_end, 60.0)),
    );
    let epochs = [
        input.epoch_s,
        input.epoch_s + bounded_abs_or_raw(input.t_end, 60.0),
    ];
    assert_ok_finite_or_err("StatePropagator::ephemeris", propagator.ephemeris(&epochs));
    if let Ok(cov) = Covariance6::from_diagonal(input.cov_diag) {
        assert_ok_finite_or_err(
            "StatePropagator::propagate_state_with_covariance",
            propagator.propagate_state_with_covariance(cov, bounded_abs_or_raw(input.t_end, 30.0)),
        );
    }

    let force = TwoBodyGravity { mu: input.sgp4[9] };
    let dynamics = OrbitalDynamics {
        force_model: &force,
    };
    let ctx = PropagationContext::default();
    assert_ok_finite_or_err(
        "RK4::propagate",
        RK4.propagate(state, input.t_end, &dynamics, &ctx, &opts),
    );
    assert_ok_finite_or_err(
        "DP54::propagate",
        DP54.propagate(state, input.t_end, &dynamics, &ctx, &opts),
    );

    let elset = elements(&input);
    assert_ok_finite_or_err(
        "sgp4::propagate_elements",
        sidereon_core::astro::sgp4::propagate_elements(
            &elset,
            MinutesSinceEpoch(input.t_end),
        ),
    );
    assert_ok_finite_or_err(
        "sgp4::propagate_elements_with_opsmode",
        sidereon_core::astro::sgp4::propagate_elements_with_opsmode(
            &elset,
            MinutesSinceEpoch(input.t_end),
            OpsMode::Afspc,
        ),
    );
    if let Ok(sat) = Satellite::from_elements(&elset) {
        assert_ok_finite_or_err(
            "Satellite::propagate",
            sat.propagate(MinutesSinceEpoch(input.t_end)),
        );
        assert_ok_finite_or_err(
            "Satellite::propagate_jd",
            sat.propagate_jd(JulianDate(input.sgp4[9], input.sgp4[10])),
        );
    }

    let reduced = reduced_elements(&input);
    let scale = time_scale(input.scale);
    let frame = if input.frame & 1 == 0 {
        Frame::Gcrs
    } else {
        Frame::Ecef
    };
    let query_epoch = calendar(&input, 1);
    assert_ok_finite_or_err(
        "orbit::position",
        orbit::position(&reduced, query_epoch, scale, frame),
    );
    assert_ok_finite_or_err(
        "orbit::position_velocity",
        orbit::position_velocity(&reduced, query_epoch, scale, frame),
    );
    let samples: Vec<EcefSample> = cap_vec(input.samples_xyz.clone(), 6)
        .into_iter()
        .enumerate()
        .map(|(idx, xyz)| EcefSample::new(calendar(&input, idx), xyz[0], xyz[1], xyz[2]))
        .collect();
    assert_ok_finite_or_err(
        "orbit::drift",
        orbit::drift(&reduced, &samples, scale, input.orbit_scalars[11]),
    );
    assert_ok_finite_or_err("orbit::fit", orbit::fit(&samples, scale));
    assert_ok_finite_or_err(
        "orbit::fit_with_model",
        orbit::fit_with_model(&samples, scale, reduced.model),
    );
});
