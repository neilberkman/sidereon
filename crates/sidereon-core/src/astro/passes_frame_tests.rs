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
            for station in stations() {
                assert_grid_matches_single_satellite(&satellites, station, &epochs, mode);
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
                        epoch.try_time_scales().unwrap().ut1_degraded
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

#[test]
#[allow(deprecated)]
fn checked_time_scales_match_legacy_across_leap_seconds() {
    for micros in [i64::MIN, i64::MAX] {
        assert!(UtcInstant::from_unix_microseconds(micros)
            .try_time_scales()
            .is_none());
    }
    let bits = |ts: TimeScales| {
        (
            [
                ts.jd_whole,
                ts.ut1_fraction,
                ts.tt_fraction,
                ts.tdb_fraction,
                ts.jd_ut1,
                ts.jd_tt,
                ts.jd_tdb,
            ]
            .map(f64::to_bits),
            ts.ut1_degraded,
        )
    };
    for (year, month, day, next_year, next_month) in
        [(2015, 6, 30, 2015, 7), (2016, 12, 31, 2017, 1)]
    {
        let start = UtcInstant::from_utc(year, month, day, 23, 59, 59, 875_000).unwrap();
        for offset in [0, 124_999, 125_000, 125_001, 1_125_000] {
            let epoch = start.add_microseconds(offset);
            let calendar = if offset < 125_000 {
                TimeScales::from_utc(
                    year,
                    month,
                    day,
                    23,
                    59,
                    59.0 + (875_000 + offset) as f64 / 1_000_000.0,
                )
            } else {
                TimeScales::from_utc(
                    next_year,
                    next_month,
                    1,
                    0,
                    0,
                    (offset - 125_000) as f64 / 1_000_000.0,
                )
            };
            let checked = bits(epoch.try_time_scales().unwrap());
            assert_eq!(checked, bits(epoch.time_scales()));
            assert_eq!(checked, bits(calendar.unwrap()));
        }
    }
}

fn assert_grid_matches_single_satellite(
    satellites: &[Satellite],
    station: GroundStation,
    epochs: &[UtcInstant],
    mode: ValidityMode,
) {
    let looks: Vec<_> = satellites
        .iter()
        .map(|sat| look_angle_arc_with_validity(sat, station, epochs, mode))
        .collect();
    identical(
        look_angle_batch_serial_with_validity(satellites, station, epochs, mode),
        &looks,
    );
    identical(
        look_angle_batch_parallel_with_validity(satellites, station, epochs, mode),
        &looks,
    );
    let tracks: Vec<_> = satellites
        .iter()
        .map(|sat| ground_track_with_validity(sat, epochs, mode))
        .collect();
    identical(
        ground_track_batch_serial_with_validity(satellites, epochs, mode),
        &tracks,
    );
    identical(
        ground_track_batch_parallel_with_validity(satellites, epochs, mode),
        &tracks,
    );
    if mode == ValidityMode::Strict {
        let looks: Vec<_> = satellites
            .iter()
            .map(|sat| look_angle_arc(sat, station, epochs))
            .collect();
        identical(look_angle_batch_serial(satellites, station, epochs), &looks);
        identical(
            look_angle_batch_parallel(satellites, station, epochs),
            &looks,
        );
        let tracks: Vec<_> = satellites
            .iter()
            .map(|sat| ground_track(sat, epochs))
            .collect();
        identical(ground_track_batch_serial(satellites, epochs), &tracks);
        identical(ground_track_batch_parallel(satellites, epochs), &tracks);
    }
}

#[test]
fn invalid_epochs_and_stations_preserve_error_precedence() {
    let base = UtcInstant::from_utc(2016, 12, 31, 23, 59, 58, 0).unwrap();
    let satellites: Vec<_> = elements_at(base, 2)
        .iter()
        .map(|e| Satellite::from_elements(e).unwrap())
        .collect();
    let invalid_site = GroundStation {
        latitude_deg: f64::NAN,
        ..stations()[0]
    };
    for micros in [i64::MIN, i64::MAX] {
        let epoch = UtcInstant::from_unix_microseconds(micros);
        for mode in [ValidityMode::Strict, ValidityMode::Permissive] {
            let invalid_datetime = LookAngleError::InvalidInput {
                field: "datetime",
                reason: "invalid UTC instant",
            };
            for sites in [stations().to_vec(), vec![invalid_site, stations()[0]]] {
                let expected: Vec<Vec<_>> = satellites
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
                for (row, sat) in expected.iter().zip(&satellites) {
                    for (&site, result) in sites.iter().zip(row) {
                        if site.latitude_deg.is_finite() {
                            assert_eq!(*result, Err(invalid_datetime.clone()));
                        } else {
                            assert_ne!(*result, Err(invalid_datetime.clone()));
                            assert_eq!(
                                *result,
                                look_angle_arc_with_validity(sat, site, &[base], mode)
                                    .map(|arc| arc.value[0])
                            );
                        }
                    }
                }
                let actual =
                    coverage::look_angles_batch_with_validity(&satellites, &sites, epoch, mode);
                assert_eq!(actual.degraded, None);
                identical(actual.value, &expected);
                if mode == ValidityMode::Strict {
                    identical(
                        coverage::look_angles_batch(&satellites, &sites, epoch),
                        &expected,
                    );
                }
            }
            assert_grid_matches_single_satellite(&satellites, invalid_site, &[epoch], mode);
        }
    }
}

#[test]
fn propagation_error_precedes_later_invalid_epoch() {
    let base = UtcInstant::from_utc(2005, 11, 29, 0, 30, 0, 0).unwrap();
    let mut satellites: Vec<_> = elements_at(base, 2)
        .iter()
        .map(|e| Satellite::from_elements(e).unwrap())
        .collect();
    satellites.insert(
        1,
        Satellite::from_tle(
            "1 28872U 05037B   05333.02012661  .25992681  00000-0  24476-3 0  1534",
            "2 28872  96.4736 157.9986 0303955 244.0492 110.6523 16.46015938 10708",
        )
        .unwrap(),
    );
    let mut epochs = vec![base; FRAME_CHUNK_SIZE + 1];
    epochs.push(base.add_microseconds(MICROSECONDS_PER_DAY_I64));
    let station = stations()[0];
    assert!(look_angle_arc(&satellites[1], station, &epochs[..FRAME_CHUNK_SIZE + 1]).is_ok());
    assert!(matches!(
        look_angle_arc(&satellites[1], station, &epochs),
        Err(LookAngleError::Propagate(_))
    ));
    for index in [0, 2] {
        assert!(look_angle_arc(&satellites[index], station, &epochs).is_ok());
        assert!(ground_track(&satellites[index], &epochs).is_ok());
    }
    let failure_len = epochs.len();
    epochs.push(UtcInstant::from_unix_microseconds(i64::MAX));
    for mode in [ValidityMode::Strict, ValidityMode::Permissive] {
        assert_grid_matches_single_satellite(&satellites, station, &epochs, mode);
        let decaying = &satellites[1];
        let look_error =
            look_angle_arc_with_validity(decaying, station, &epochs[..failure_len], mode)
                .unwrap_err();
        let track_error =
            ground_track_with_validity(decaying, &epochs[..failure_len], mode).unwrap_err();
        assert!(matches!(look_error, LookAngleError::Propagate(_)));
        assert_eq!(track_error, look_error);
        assert_eq!(
            look_angle_arc_with_validity(decaying, station, &epochs, mode).unwrap_err(),
            look_error
        );
        assert_eq!(
            ground_track_with_validity(decaying, &epochs, mode).unwrap_err(),
            track_error
        );
        for index in [0, 2] {
            let invalid_datetime = LookAngleError::InvalidInput {
                field: "datetime",
                reason: "invalid UTC instant",
            };
            assert_eq!(
                look_angle_arc_with_validity(&satellites[index], station, &epochs, mode)
                    .unwrap_err(),
                invalid_datetime
            );
            assert_eq!(
                ground_track_with_validity(&satellites[index], &epochs, mode).unwrap_err(),
                invalid_datetime
            );
        }
    }
}

#[test]
fn empty_and_exact_chunks_preserve_late_ut1_departure() {
    let base = UtcInstant::from_utc(2027, 7, 2, 23, 59, 55, 0).unwrap();
    let boundary = UtcInstant::from_utc(2027, 7, 3, 0, 0, 0, 0).unwrap();
    let satellites: Vec<_> = elements_at(base, 2)
        .iter()
        .map(|e| Satellite::from_elements(e).unwrap())
        .collect();
    let epochs: Vec<_> = (0..512)
        .map(|i| {
            if i < 256 {
                base.add_microseconds(i * 10_000)
            } else {
                boundary.add_microseconds((i - 255) * 10_000)
            }
        })
        .collect();
    assert!(epochs[..256]
        .iter()
        .all(|e| e.try_time_scales().unwrap().ut1_degraded.is_none()));
    let departure = Some(crate::astro::time::DegradeReason::AfterCoverage);
    assert_eq!(
        epochs[256].try_time_scales().unwrap().ut1_degraded,
        departure
    );
    let station = stations()[0];
    for sat in &satellites {
        assert_eq!(
            look_angle_arc_with_validity(sat, station, &epochs, ValidityMode::Permissive)
                .unwrap()
                .degraded,
            departure
        );
        assert_eq!(
            ground_track_with_validity(sat, &epochs, ValidityMode::Permissive)
                .unwrap()
                .degraded,
            departure
        );
    }
    let covered: Vec<_> = (0..512).map(|i| base.add_microseconds(i * 5_000)).collect();
    assert_grid_matches_single_satellite(&satellites, station, &covered, ValidityMode::Strict);
    for len in [0, 256, 512] {
        for mode in [ValidityMode::Strict, ValidityMode::Permissive] {
            assert_grid_matches_single_satellite(&satellites, station, &epochs[..len], mode);
        }
    }
}
