//! Batch frame costs in milliseconds per call over shared epoch grids.

use sidereon_core::astro::passes::{
    find_passes_batch_serial, ground_track_batch_serial, look_angle_arc, look_angle_batch_serial,
    visible_from_satellites, GroundStation, PassFinderOptions, UtcInstant,
};
use sidereon_core::astro::sgp4::{ElementSet, JulianDate, Satellite};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

fn main() {
    let epoch = UtcInstant::from_utc(2026, 4, 5, 0, 0, 0, 0).unwrap();
    let station = GroundStation {
        latitude_deg: 51.5074,
        longitude_deg: -0.1278,
        altitude_m: 11.0,
    };
    let elements: Vec<_> = (0..143)
        .map(|i| ElementSet {
            epoch: JulianDate(2_461_135.0, 0.5),
            bstar: 0.0,
            mean_motion_dot: None,
            mean_motion_double_dot: None,
            eccentricity: 0.001,
            argument_of_perigee_deg: 30.0,
            inclination_deg: 55.0,
            mean_anomaly_deg: (i as f64 * 137.5) % 360.0,
            mean_motion_rev_per_day: 2.0,
            right_ascension_deg: (i as f64 * 23.0) % 360.0,
            catalog_number: Some(i + 1),
            omm_epoch_days: None,
        })
        .collect();
    let satellites: Vec<_> = elements
        .iter()
        .map(|e| Satellite::from_elements(e).unwrap())
        .collect();
    if let Some(case) = std::env::args().nth(1) {
        match case.as_str() {
            "pass" => {
                let end =
                    UtcInstant::from_unix_microseconds(epoch.unix_microseconds() + 86_400_000_000);
                let start = Instant::now();
                let rows = find_passes_batch_serial(
                    &elements,
                    station,
                    epoch,
                    end,
                    PassFinderOptions::default(),
                );
                println!(
                    "pass {} ms {} events",
                    start.elapsed().as_secs_f64() * 1000.0,
                    rows.into_iter().map(|r| r.unwrap().len()).sum::<usize>()
                );
            }
            _ => {
                let count = if case == "arc" { 604_800 } else { 10_080 };
                let epochs: Vec<_> = (0..count)
                    .map(|i| {
                        UtcInstant::from_unix_microseconds(
                            epoch.unix_microseconds()
                                + i * if case == "arc" { 1_000_000 } else { 60_000_000 },
                        )
                    })
                    .collect();
                if case == "arc" {
                    println!(
                        "{}",
                        black_box(look_angle_arc(&satellites[0], station, &epochs).unwrap()).len()
                    );
                } else {
                    println!(
                        "{}",
                        black_box(look_angle_batch_serial(&satellites, station, &epochs))
                            .iter()
                            .map(|r| r.as_ref().unwrap().len())
                            .sum::<usize>()
                    );
                }
            }
        }
        return;
    }
    let ids: Vec<_> = (0..143).map(|i| i.to_string()).collect();
    println!("satellites,epochs,visible_ms,look_angle_arcs_ms,ground_tracks_ms");
    for count in [1, 10, 143] {
        for epoch_count in [1, 51] {
            let epochs: Vec<_> = (0..epoch_count)
                .map(|i| {
                    UtcInstant::from_unix_microseconds(epoch.unix_microseconds() + i * 60_000_000)
                })
                .collect();
            let satellites = &satellites[..count];
            let ids = &ids[..count];
            let mut timings = Vec::new();
            for path in 0..3 {
                let evaluate = || match path {
                    0 => {
                        for &epoch in &epochs {
                            black_box(
                                visible_from_satellites(
                                    black_box(satellites),
                                    ids,
                                    station,
                                    epoch,
                                    0.0,
                                )
                                .unwrap(),
                            );
                        }
                    }
                    1 => {
                        black_box(look_angle_batch_serial(
                            black_box(satellites),
                            station,
                            &epochs,
                        ));
                    }
                    _ => {
                        black_box(ground_track_batch_serial(black_box(satellites), &epochs));
                    }
                };
                let warmup = Instant::now();
                let mut repetitions = 0;
                while warmup.elapsed() < Duration::from_millis(100) {
                    evaluate();
                    repetitions += 1;
                }
                let repetitions = repetitions.max(20);
                let start = Instant::now();
                for _ in 0..repetitions {
                    evaluate();
                }
                timings.push(start.elapsed().as_secs_f64() * 1000.0 / f64::from(repetitions));
            }
            println!(
                "{count},{epoch_count},{:.6},{:.6},{:.6}",
                timings[0], timings[1], timings[2]
            );
        }
    }
}
