#![no_main]

mod compute_common;

use std::sync::OnceLock;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::time::model::{Instant, TimeScale};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::orbit_scoring::{
    inertial_velocity_from_earth_fixed, rtn_components, score_orbit, ClockDatum, ErrorStats,
    GapPolicy, OrbitRegime, OrbitScoreOptions, OrbitTrajectory, RtnComponents, ScoreEpochs,
    SisreWeights, TrajectorySample, TriadVelocity,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

/// IGS final GPS orbits, 2026-05-13 00:00 to 03:00 GPST.
const REFERENCE_SP3: &[u8] = include_bytes!(
    "../../crates/sidereon-core/tests/fixtures/sp3/IGS0OPSFIN_20261330000_03H_15M_ORB.SP3"
);
/// Seconds since J2000 of the reference's first record.
const REFERENCE_START_S: i64 = 831_902_400;

fn reference() -> &'static Sp3 {
    static REFERENCE: OnceLock<Sp3> = OnceLock::new();
    REFERENCE.get_or_init(|| Sp3::parse(REFERENCE_SP3).expect("reference SP3 fixture"))
}

#[derive(Debug, Arbitrary)]
struct Sample {
    prn: u8,
    /// Seconds after the reference's first record.
    offset_s: i16,
    position_ecef_m: [f64; 3],
    clock_s: Option<f64>,
}

#[derive(Debug, Arbitrary)]
struct Input {
    difference_m: [f64; 3],
    position_m: [f64; 3],
    velocity_m_s: [f64; 3],
    weights: [f64; 2],
    components: [f64; 3],
    clock_m: f64,
    edges_s: Vec<f64>,
    velocity_half_step_s: f64,
    origin_offset_s: i16,
    samples: Vec<Sample>,
    satellites: Vec<u8>,
    earth_fixed_triad: bool,
    unadjusted_clock: bool,
    refuse_gaps: bool,
    trajectory_epochs: bool,
}

fn gpst(offset_s: i16) -> Instant {
    let seconds = i128::from(REFERENCE_START_S) + i128::from(offset_s);
    Instant::from_nanos(TimeScale::Gpst, seconds * 1_000_000_000)
}

fn gps(prn: u8) -> Option<GnssSatelliteId> {
    GnssSatelliteId::new(GnssSystem::Gps, prn % 40).ok()
}

fn stats_values(stats: Option<ErrorStats>) -> Vec<f64> {
    stats.map_or_else(Vec::new, |s| vec![s.rms_m, s.mean_m, s.max_abs_m])
}

fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };

    if let Ok(rtn) = rtn_components(input.difference_m, input.position_m, input.velocity_m_s) {
        let components = [rtn.radial_m, rtn.transverse_m, rtn.normal_m];
        assert_all_finite("orbit_scoring::rtn_components", components);
        // The frame is a rotation, so the components keep the length of the
        // difference. Both vectors are divided by the largest magnitude of the
        // difference before their lengths are formed: a length near or above
        // f64::MAX is then compared like any other, where an unscaled length
        // of accurate components can round past f64::MAX. The bound is the
        // rounding of the products plus two units of the subnormal spacing,
        // the most the scaling back of three components into the subnormal
        // range moves the length (sqrt(3) / 2 units).
        let largest = input
            .difference_m
            .iter()
            .fold(0.0_f64, |m, c| m.max(c.abs()));
        if largest == 0.0 {
            assert!(
                components.iter().all(|&c| c == 0.0),
                "orbit_scoring::rtn_components of a zero difference {components:?}"
            );
        } else {
            let length = |v: [f64; 3]| {
                let unit = v.map(|c| c / largest);
                (unit[0] * unit[0] + unit[1] * unit[1] + unit[2] * unit[2]).sqrt()
            };
            let (before, after) = (length(input.difference_m), length(components));
            let floor = 2.0 * f64::from_bits(1) / largest;
            assert!(
                (after - before).abs() <= 1.0e-12 * before + floor,
                "orbit_scoring::rtn_components length {before:e} -> {after:e} in units of {largest:e}"
            );
        }
    }
    if let Ok(velocity) = inertial_velocity_from_earth_fixed(input.position_m, input.velocity_m_s) {
        assert_all_finite(
            "orbit_scoring::inertial_velocity_from_earth_fixed",
            velocity,
        );
    }
    let _ = OrbitRegime::from_inertial_state(input.position_m, input.velocity_m_s);
    if let Ok(weights) = SisreWeights::new(input.weights[0], input.weights[1]) {
        let components = RtnComponents {
            radial_m: input.components[0],
            transverse_m: input.components[1],
            normal_m: input.components[2],
        };
        if let Ok(value) = weights.orbit_only_m(components) {
            assert_all_finite("orbit_scoring::SisreWeights::orbit_only_m", [value]);
        }
        if let Ok(value) = weights.with_clock_m(components, input.clock_m) {
            assert_all_finite("orbit_scoring::SisreWeights::with_clock_m", [value]);
        }
    }

    let samples: Vec<TrajectorySample> = cap_vec(input.samples, MAX_OBS)
        .into_iter()
        .filter_map(|sample| {
            Some(TrajectorySample {
                satellite: gps(sample.prn)?,
                epoch: gpst(sample.offset_s),
                position_ecef_m: sample.position_ecef_m,
                clock_s: sample.clock_s,
            })
        })
        .collect();
    let Ok(trajectory) = OrbitTrajectory::new("IGc20", gpst(input.origin_offset_s), samples) else {
        return;
    };
    let satellites: Vec<GnssSatelliteId> = cap_vec(input.satellites, MAX_EPOCHS)
        .into_iter()
        .filter_map(gps)
        .collect();

    let mut options = OrbitScoreOptions::new(cap_vec(input.edges_s, MAX_VEC));
    options.velocity_half_step_s = input.velocity_half_step_s;
    if input.earth_fixed_triad {
        options.triad = TriadVelocity::EarthFixed;
    }
    if input.unadjusted_clock {
        options.clock_datum = ClockDatum::Unadjusted;
    }
    if input.refuse_gaps {
        options.gaps = GapPolicy::Refuse;
    }
    if input.trajectory_epochs {
        options.epochs = ScoreEpochs::Instants(trajectory.epochs());
    }

    if let Ok(report) = score_orbit(&trajectory, reference(), &satellites, &options) {
        let coverage = report.coverage;
        assert_eq!(
            coverage.epochs,
            coverage.scored
                + coverage.outside_horizon_bins
                + coverage.no_prediction_origin
                + coverage.source_satellite_missing
                + coverage.source_epoch_missing
                + coverage.reference_satellite_missing
                + coverage.reference_epoch_missing,
            "orbit_scoring::score_orbit coverage"
        );
        let bins = report
            .per_satellite
            .iter()
            .flat_map(|score| score.bins.iter())
            .chain(report.per_system.iter().flat_map(|score| score.bins.iter()));
        for bin in bins {
            for stats in [
                bin.radial,
                bin.transverse,
                bin.normal,
                bin.orbit_3d,
                bin.sisre_orbit,
                bin.clock,
                bin.sisre,
            ] {
                assert_all_finite("orbit_scoring::score_orbit", stats_values(stats));
            }
        }
    }
});
