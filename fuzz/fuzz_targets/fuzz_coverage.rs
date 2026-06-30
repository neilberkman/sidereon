#![no_main]

mod compute_common;

use arbitrary::Arbitrary;
use compute_common::*;
use libfuzzer_sys::fuzz_target;
use sidereon_core::astro::coverage;
use sidereon_core::astro::passes::{GroundStation, UtcInstant};
use sidereon_core::astro::sgp4::{ElementSet, JulianDate, Satellite};

#[derive(Debug, Arbitrary)]
struct Input {
    sats: Vec<([f64; 11], i64, u32)>,
    stations: Vec<[f64; 3]>,
    epoch_us: i64,
    min_elevation_deg: f64,
}

fn element_set(raw: &([f64; 11], i64, u32)) -> ElementSet {
    let (doubles, year, catalog) = raw;
    ElementSet {
        epoch: JulianDate(*year as f64, doubles[10]),
        bstar: doubles[0],
        mean_motion_dot: doubles[1],
        mean_motion_double_dot: doubles[2],
        eccentricity: doubles[3],
        argument_of_perigee_deg: doubles[4],
        inclination_deg: doubles[5],
        mean_anomaly_deg: doubles[6],
        mean_motion_rev_per_day: doubles[7],
        right_ascension_deg: doubles[8],
        catalog_number: *catalog,
    }
}

// The coverage grid is a batch wrapper over the look-angle kernel. Feed a small
// set of arbitrary satellites/stations and reduce the grid; the batch builder and
// the column reductions must not panic regardless of per-cell errors.
fuzz_target!(|data: &[u8]| {
    let Some(input) = fuzz_input::<Input>(data) else {
        return;
    };

    let satellites: Vec<Satellite> = cap_vec(input.sats, 4)
        .iter()
        .filter_map(|raw| Satellite::from_elements(&element_set(raw)).ok())
        .collect();
    let stations: Vec<GroundStation> = cap_vec(input.stations, 4)
        .into_iter()
        .map(|s| GroundStation {
            latitude_deg: s[0],
            longitude_deg: s[1],
            altitude_m: s[2],
        })
        .collect();
    let instant = UtcInstant::from_unix_microseconds(input.epoch_us);

    let grid = coverage::look_angles_batch(&satellites, &stations, instant);
    let _ = coverage::visible_mask(&grid, input.min_elevation_deg);
    let _ = coverage::access_counts(&grid, input.min_elevation_deg);
    let _ = coverage::max_elevation(&grid);
});
