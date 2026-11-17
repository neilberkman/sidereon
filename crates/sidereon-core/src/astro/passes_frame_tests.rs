use super::*;
use crate::astro::coverage;
use crate::frame::Wgs84Geodetic;

trait BitSnapshot {
    fn snapshot(&self) -> String;
}

impl<T: BitSnapshot> BitSnapshot for Vec<T> {
    fn snapshot(&self) -> String {
        format!(
            "{:?}",
            self.iter().map(BitSnapshot::snapshot).collect::<Vec<_>>()
        )
    }
}
impl<T: BitSnapshot, E: std::fmt::Debug> BitSnapshot for Result<T, E> {
    fn snapshot(&self) -> String {
        match self {
            Ok(value) => format!("Ok({})", value.snapshot()),
            Err(error) => format!("Err({error:?})"),
        }
    }
}
impl<T: BitSnapshot> BitSnapshot for Validated<T> {
    fn snapshot(&self) -> String {
        format!("{:?}:{}", self.degraded, self.value.snapshot())
    }
}
impl BitSnapshot for LookAngle {
    fn snapshot(&self) -> String {
        format!(
            "{:?}",
            [
                self.azimuth_deg.to_bits(),
                self.elevation_deg.to_bits(),
                self.range_km.to_bits()
            ]
        )
    }
}
impl BitSnapshot for Wgs84Geodetic {
    fn snapshot(&self) -> String {
        format!(
            "{:?}",
            [
                self.lat_rad.to_bits(),
                self.lon_rad.to_bits(),
                self.height_m.to_bits()
            ]
        )
    }
}
impl BitSnapshot for VisibleSatellite {
    fn snapshot(&self) -> String {
        format!(
            "{:?}:{:?}:{:?}",
            self.catalog_number,
            [
                self.azimuth_deg.to_bits(),
                self.elevation_deg.to_bits(),
                self.range_km.to_bits()
            ],
            self.position_km.map(f64::to_bits)
        )
    }
}
impl BitSnapshot for SatellitePass {
    fn snapshot(&self) -> String {
        format!(
            "{:?}:{:?}:{:?}:{}",
            self.aos,
            self.los,
            self.culmination,
            self.max_elevation_deg.to_bits()
        )
    }
}
fn identical<T: BitSnapshot>(actual: T, expected: &T) {
    assert_eq!(actual.snapshot(), expected.snapshot());
}

fn elements_at(base: UtcInstant, count: usize) -> Vec<ElementSet> {
    (0..count)
        .map(|i| {
            let mut elements = tests::iss_fixture_elements();
            elements.epoch = base.sgp4_julian_date();
            elements.bstar = 0.0;
            elements.mean_motion_rev_per_day = if i % 3 == 0 { 15.0 } else { 2.0 };
            elements.eccentricity = if i % 5 == 0 && i % 3 != 0 { 0.6 } else { 0.001 };
            elements.mean_anomaly_deg = (i as f64 * 137.5) % 360.0;
            elements.right_ascension_deg = (i as f64 * 23.0) % 360.0;
            elements
        })
        .collect()
}

fn stations() -> [GroundStation; 3] {
    [
        GroundStation {
            latitude_deg: 51.5074,
            longitude_deg: -0.1278,
            altitude_m: 11.0,
        },
        GroundStation {
            latitude_deg: 0.0,
            longitude_deg: 32.58,
            altitude_m: 1200.0,
        },
        GroundStation {
            latitude_deg: -89.9,
            longitude_deg: 179.9,
            altitude_m: -30.0,
        },
    ]
}

#[test]
fn frame_grids_match_single_satellite_bits() {
    for base in [
        UtcInstant::from_utc(1970, 1, 1, 0, 0, 0, 0).unwrap(),
        UtcInstant::from_utc(2016, 12, 31, 23, 59, 58, 0).unwrap(),
        UtcInstant::from_utc(2050, 1, 1, 0, 0, 0, 0).unwrap(),
    ] {
        let mut epochs: Vec<_> = (0..FRAME_CHUNK_SIZE as i64 + 7)
            .map(|i| base.add_microseconds(i * 500_000))
            .collect();
        epochs[FRAME_CHUNK_SIZE + 1] = base;
        let elements = elements_at(base, 16);
        let satellites: Vec<_> = elements
            .iter()
            .map(|e| Satellite::from_elements(e).unwrap())
            .collect();
        for mode in [ValidityMode::Strict, ValidityMode::Permissive] {
            let tracks: Vec<_> = satellites
                .iter()
                .map(|sat| ground_track_with_validity(sat, &epochs, mode))
                .collect();
            identical(
                ground_track_batch_serial_with_validity(&satellites, &epochs, mode),
                &tracks,
            );
            identical(
                ground_track_batch_parallel_with_validity(&satellites, &epochs, mode),
                &tracks,
            );
            if mode == ValidityMode::Strict {
                let tracks: Vec<_> = satellites
                    .iter()
                    .map(|sat| ground_track(sat, &epochs))
                    .collect();
                identical(ground_track_batch_serial(&satellites, &epochs), &tracks);
                identical(ground_track_batch_parallel(&satellites, &epochs), &tracks);
            }
            for station in stations() {
                let looks: Vec<_> = satellites
                    .iter()
                    .map(|sat| look_angle_arc_with_validity(sat, station, &epochs, mode))
                    .collect();
                identical(
                    look_angle_batch_serial_with_validity(&satellites, station, &epochs, mode),
                    &looks,
                );
                identical(
                    look_angle_batch_parallel_with_validity(&satellites, station, &epochs, mode),
                    &looks,
                );
                if mode == ValidityMode::Strict {
                    let looks: Vec<_> = satellites
                        .iter()
                        .map(|sat| look_angle_arc(sat, station, &epochs))
                        .collect();
                    identical(
                        look_angle_batch_serial(&satellites, station, &epochs),
                        &looks,
                    );
                    identical(
                        look_angle_batch_parallel(&satellites, station, &epochs),
                        &looks,
                    );
                }
            }
            let invalid = GroundStation {
                latitude_deg: f64::NAN,
                ..stations()[0]
            };
            let mut sites = stations().to_vec();
            sites.push(invalid);
            for epoch in [epochs[0], epochs[5], epochs[FRAME_CHUNK_SIZE]] {
                let grid: Vec<Vec<_>> = satellites
                    .iter()
                    .map(|sat| {
                        sites
                            .iter()
                            .map(|&site| {
                                look_angle_arc_with_validity(sat, site, &[epoch], mode)
                                    .map(|arc| arc.value[0])
                            })
                            .collect()
                    })
                    .collect();
                let actual =
                    coverage::look_angles_batch_with_validity(&satellites, &sites, epoch, mode);
                assert_eq!(
                    actual.degraded,
                    if mode == ValidityMode::Permissive {
                        epoch.time_scales().ut1_degraded
                    } else {
                        None
                    }
                );
                identical(actual.value, &grid);
                if mode == ValidityMode::Strict {
                    identical(
                        coverage::look_angles_batch(&satellites, &sites, epoch),
                        &grid,
                    );
                }
                let members: Vec<_> = elements
                    .iter()
                    .enumerate()
                    .map(|(i, e)| ConstellationMember {
                        catalog_number: i.to_string(),
                        elements: e.clone(),
                    })
                    .collect();
                let ids: Vec<_> = members.iter().map(|m| m.catalog_number.clone()).collect();
                for site in stations() {
                    let expected = (|| {
                        let gate = Ut1Gate::new(mode);
                        time_scales_for_pass(epoch, &gate)?;
                        let mut values = Vec::new();
                        for (sat, id) in satellites.iter().zip(&ids) {
                            if let Ok(arc) = look_angle_arc_with_validity(sat, site, &[epoch], mode)
                            {
                                let look = arc.value[0];
                                let position =
                                    propagate_teme_arc(sat, &[epoch]).unwrap()[0].position;
                                values.push(VisibleSatellite {
                                    catalog_number: id.clone(),
                                    azimuth_deg: look.azimuth_deg,
                                    elevation_deg: look.elevation_deg,
                                    range_km: look.range_km,
                                    position_km: position,
                                });
                            }
                        }
                        values
                            .sort_by(|a, b| b.elevation_deg.partial_cmp(&a.elevation_deg).unwrap());
                        finish_pass(&gate, values)
                    })();
                    identical(
                        visible_from_constellation_with_validity(
                            &members, site, epoch, -90.0, mode,
                        ),
                        &expected,
                    );
                    identical(
                        visible_from_satellites_with_validity(
                            &satellites,
                            &ids,
                            site,
                            epoch,
                            -90.0,
                            mode,
                        ),
                        &expected,
                    );
                    if mode == ValidityMode::Strict {
                        let expected = expected.map(|v| v.value);
                        identical(
                            visible_from_constellation(&members, site, epoch, -90.0),
                            &expected,
                        );
                        identical(
                            visible_from_satellites(&satellites, &ids, site, epoch, -90.0),
                            &expected,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn pass_batches_match_individual_searches() {
    for year in [2016, 2050] {
        let base = UtcInstant::from_utc(year, 12, 31, 23, 50, 0, 0).unwrap();
        let end = base.add_microseconds(1800 * MICROSECONDS_PER_SECOND_I64);
        let elements = elements_at(base, 4);
        for site in stations() {
            for mode in [ValidityMode::Strict, ValidityMode::Permissive] {
                for opsmode in [OpsMode::Improved, OpsMode::Afspc] {
                    let opts = PassFinderOptions::default();
                    let expected: Vec<_> = elements
                        .iter()
                        .map(|e| find_passes_with_validity(e, site, base, end, opts, opsmode, mode))
                        .collect();
                    identical(
                        find_passes_batch_serial_with_validity(
                            &elements, site, base, end, opts, opsmode, mode,
                        ),
                        &expected,
                    );
                    identical(
                        find_passes_batch_parallel_with_validity(
                            &elements, site, base, end, opts, opsmode, mode,
                        ),
                        &expected,
                    );
                }
            }
        }
    }
}

#[test]
fn grid_errors_follow_epoch_order() {
    let base = UtcInstant::from_utc(2016, 12, 31, 23, 59, 58, 0).unwrap();
    let satellites: Vec<_> = elements_at(base, 6)
        .iter()
        .map(|e| Satellite::from_elements(e).unwrap())
        .collect();
    let mut epochs = vec![base; FRAME_CHUNK_SIZE + 9];
    epochs[FRAME_CHUNK_SIZE + 1] = UtcInstant::from_utc(2050, 1, 1, 0, 0, 0, 0).unwrap();
    epochs[FRAME_CHUNK_SIZE + 2] = UtcInstant::from_unix_microseconds(i64::MAX);
    for mode in [ValidityMode::Strict, ValidityMode::Permissive] {
        let site = stations()[0];
        let expected: Vec<_> = satellites
            .iter()
            .map(|sat| look_angle_arc_with_validity(sat, site, &epochs, mode))
            .collect();
        assert!(expected.iter().all(Result::is_err));
        assert_eq!(
            look_angle_batch_serial_with_validity(&satellites, site, &epochs, mode),
            expected
        );
        assert_eq!(
            look_angle_batch_parallel_with_validity(&satellites, site, &epochs, mode),
            expected
        );
        let expected: Vec<_> = satellites
            .iter()
            .map(|sat| ground_track_with_validity(sat, &epochs, mode))
            .collect();
        assert_eq!(
            ground_track_batch_serial_with_validity(&satellites, &epochs, mode),
            expected
        );
        assert_eq!(
            ground_track_batch_parallel_with_validity(&satellites, &epochs, mode),
            expected
        );
    }
}
