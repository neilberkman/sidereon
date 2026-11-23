#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::propagator::analytic::{
    brouwer_lyddane_mean_elements, brouwer_lyddane_osculating_elements,
    eckstein_hechler_mean_elements, eckstein_hechler_osculating_elements,
    BrouwerLyddanePropagator, CircularElements, EcksteinHechlerPropagator, KeplerianElements,
    MeanConversionOptions, ZonalHarmonics,
};
use sidereon_core::astro::CartesianState;

#[derive(Debug, Arbitrary)]
struct Input {
    epoch_s: f64,
    pos: [f64; 3],
    vel: [f64; 3],
    targets: [f64; 3],
    gravity: [f64; 7],
    use_raw_gravity: bool,
    tolerance: f64,
    max_iterations: u8,
    m2: f64,
    circular: [f64; 6],
    keplerian: [f64; 6],
}

fn gravity(input: &Input) -> ZonalHarmonics {
    if input.use_raw_gravity {
        let g = input.gravity;
        ZonalHarmonics {
            mu_km3_s2: g[0],
            re_km: g[1],
            c20: g[2],
            c30: g[3],
            c40: g[4],
            c50: g[5],
            c60: g[6],
        }
    } else {
        ZonalHarmonics::earth()
    }
}

fn state_values(s: &CartesianState) -> [f64; 7] {
    let p = s.position_array();
    let v = s.velocity_array();
    [s.epoch_tdb_seconds, p[0], p[1], p[2], v[0], v[1], v[2]]
}

fn circular_values(c: &CircularElements) -> [f64; 6] {
    [
        c.a_km,
        c.ex,
        c.ey,
        c.i_rad,
        c.raan_rad,
        c.mean_latitude_argument_rad,
    ]
}

fn keplerian_values(k: &KeplerianElements) -> [f64; 6] {
    [
        k.a_km,
        k.e,
        k.i_rad,
        k.arg_perigee_rad,
        k.raan_rad,
        k.mean_anomaly_rad,
    ]
}

/// Every element set returned has its inclination in `[0, pi]`.
fn assert_inclination(label: &str, i_rad: f64) {
    assert!(
        (0.0..=core::f64::consts::PI).contains(&i_rad),
        "{label}: inclination {i_rad}"
    );
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };
    let g = gravity(&input);
    let state = CartesianState::new(input.epoch_s, input.pos, input.vel);
    // Small iteration limits keep each input fast; the error paths are the same.
    let options = MeanConversionOptions::new(
        input.tolerance,
        u32::from(input.max_iterations % 16),
    );
    let c = input.circular;
    let circular = CircularElements {
        a_km: c[0],
        ex: c[1],
        ey: c[2],
        i_rad: c[3],
        raan_rad: c[4],
        mean_latitude_argument_rad: c[5],
    };
    let k = input.keplerian;
    let keplerian = KeplerianElements {
        a_km: k[0],
        e: k[1],
        i_rad: k[2],
        arg_perigee_rad: k[3],
        raan_rad: k[4],
        mean_anomaly_rad: k[5],
    };

    if let Ok(mean) = eckstein_hechler_mean_elements(&state, &g, options) {
        assert_all_finite("eh_mean", circular_values(&mean.elements));
        assert_inclination("eh_mean", mean.elements.i_rad);
    }
    if let Ok(mean) = brouwer_lyddane_mean_elements(&state, &g, options) {
        assert_all_finite("bl_mean", keplerian_values(&mean.elements));
        assert_inclination("bl_mean", mean.elements.i_rad);
    }
    if let Ok(osc) = eckstein_hechler_osculating_elements(&circular, &g) {
        assert_all_finite("eh_osc", circular_values(&osc));
        assert_inclination("eh_osc", osc.i_rad);
    }
    if let Ok(osc) = brouwer_lyddane_osculating_elements(&keplerian, &g) {
        assert_all_finite("bl_osc", keplerian_values(&osc));
        assert_inclination("bl_osc", osc.i_rad);
    }
    if let Ok(pv) = circular.to_cartesian(g.mu_km3_s2) {
        assert_all_finite("circular_to_cartesian", pv.0.into_iter().chain(pv.1));
    }
    if let Ok(pv) = keplerian.to_cartesian(g.mu_km3_s2) {
        assert_all_finite("keplerian_to_cartesian", pv.0.into_iter().chain(pv.1));
    }
    if let Ok(c) = CircularElements::from_cartesian(input.pos, input.vel, g.mu_km3_s2) {
        assert_all_finite("circular_from_cartesian", circular_values(&c));
        assert_inclination("circular_from_cartesian", c.i_rad);
    }

    let eh = [
        EcksteinHechlerPropagator::new(&state, &g, options),
        EcksteinHechlerPropagator::from_mean_elements(input.epoch_s, circular, &g),
    ];
    for prop in eh.into_iter().flatten() {
        assert_inclination("eh_mean_elements", prop.mean_elements().i_rad);
        for t in input.targets {
            if let Ok(s) = prop.propagate(t) {
                assert_all_finite("eh_propagate", state_values(&s));
            }
            if let Ok(o) = prop.osculating_elements(t) {
                assert_all_finite("eh_osculating", circular_values(&o));
                assert_inclination("eh_osculating", o.i_rad);
            }
        }
        let _ = prop.ephemeris(&input.targets);
    }
    let bl = [
        BrouwerLyddanePropagator::new(&state, &g, input.m2, options),
        BrouwerLyddanePropagator::from_mean_elements(input.epoch_s, keplerian, &g, input.m2),
    ];
    for prop in bl.into_iter().flatten() {
        assert_inclination("bl_mean_elements", prop.mean_elements().i_rad);
        for t in input.targets {
            if let Ok(s) = prop.propagate(t) {
                assert_all_finite("bl_propagate", state_values(&s));
            }
            if let Ok(o) = prop.osculating_elements(t) {
                assert_all_finite("bl_osculating", keplerian_values(&o));
                assert_inclination("bl_osculating", o.i_rad);
            }
        }
        let _ = prop.ephemeris(&input.targets);
    }
});
