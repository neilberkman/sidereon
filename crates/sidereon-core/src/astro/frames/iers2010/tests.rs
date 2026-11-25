use std::collections::BTreeMap;

use super::*;

const FIXTURE: &str = include_str!("../../../../tests/fixtures/earth_orientation/iers2010.txt");

type State = ([f64; 3], [f64; 3]);

fn state(row: &[f64]) -> State {
    (row[..3].try_into().unwrap(), row[3..].try_into().unwrap())
}

fn ordered_bits(value: f64) -> u64 {
    let bits = value.to_bits();
    if value.is_sign_negative() {
        !bits
    } else {
        bits | (1_u64 << 63)
    }
}

#[test]
fn raw_orekit_earth_orientation() {
    let rows: Vec<_> = FIXTURE
        .lines()
        .map(|line| {
            let mut columns = line.split_whitespace();
            let name = columns.next().unwrap();
            let values: Vec<f64> = columns.map(|value| value.parse().unwrap()).collect();
            (name, values)
        })
        .collect();
    let samples = rows
        .iter()
        .filter(|(name, _)| *name == "sample")
        .map(|(_, row)| Iers2010EopSample {
            mjd_utc: row[0] as i32,
            xp_arcsec: row[1],
            yp_arcsec: row[2],
            ut1_minus_utc_s: row[3],
            lod_s: row[4],
            dx_arcsec: row[5],
            dy_arcsec: row[6],
            pole_rates_rad_s: row[7].is_finite().then_some([row[7], row[8]]),
        })
        .collect();
    let strict = Iers2010EarthOrientationProvider::new(samples).unwrap();
    let provider = strict.clone().with_validity(ValidityMode::Permissive);
    let states: Vec<_> = rows
        .iter()
        .filter(|(name, _)| *name == "state")
        .map(|(_, row)| state(row))
        .collect();
    let blocks = rows.split(|(label, _)| *label == "epoch");
    let mut epochs = rows.iter().filter(|(name, _)| *name == "epoch");
    let mut counts = BTreeMap::<_, (usize, u64)>::new();
    let mut failures = BTreeMap::new();
    for block in blocks.skip(1) {
        let (_, date) = epochs.next().unwrap();
        let mjd = date[0] as i32;
        let seconds = date[1];
        let epoch = date::utc(mjd, Offset::from_seconds(seconds).unwrap());
        let evaluation = provider.evaluate(epoch).unwrap();
        let orientation = provider.orientation_at_utc_mjd(mjd, seconds).unwrap();
        assert_eq!(
            strict.orientation_at_utc_mjd(mjd, seconds).is_err(),
            evaluation.degraded.is_some()
        );
        assert_eq!(orientation.ut1_degraded(), evaluation.degraded);
        let mut pv = states.iter();
        let mut inverse_pv = states.iter();
        for (name, expected) in block {
            let actual = match *name {
                "elapsed" => vec![
                    epoch.plus(TT_MINUS_TAI).as_seconds(),
                    epoch.as_seconds(),
                    -date::tai_minus_utc(epoch).as_seconds(),
                    epoch.minus(date::utc(mjd, Offset::seconds(0))).as_seconds(),
                ],
                "matrix" => orientation
                    .gcrf_to_itrf_matrix()
                    .into_iter()
                    .flatten()
                    .collect(),
                "daily" => evaluation.daily.to_vec(),
                "corrected" => evaluation.corrected.to_vec(),
                "cip" => evaluation.xys.to_vec(),
                "corrected_cip" => evaluation.corrected_xys.to_vec(),
                "angles" => vec![evaluation.era, evaluation.sp],
                "rotation" => orientation.iers2010_quaternion().unwrap().to_vec(),
                "rate" => orientation.earth_rotation_vector_itrf_rad_s().to_vec(),
                "pv" => {
                    let &(p, v) = pv.next().unwrap();
                    let (p, v) = orientation.gcrf_to_itrf_state_km(p, v).unwrap();
                    [p, v].concat()
                }
                "inverse_pv" => {
                    let &(p, v) = inverse_pv.next().unwrap();
                    let (p, v) = orientation.itrf_to_gcrf_state_km(p, v).unwrap();
                    [p, v].concat()
                }
                "factory_difference" => {
                    assert!(expected[0] > 0.0 && expected[0] < 1.4e-13);
                    assert!(expected[1] > 0.0 && expected[1] < 7.2e-12);
                    continue;
                }
                _ => panic!("unexpected oracle row {name}"),
            };
            assert_eq!(actual.len(), expected.len());
            for (i, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                assert!(
                    a.is_finite() && b.is_finite(),
                    "{name}[{i}] at {mjd}, {seconds}"
                );
                let ulp = ordered_bits(a).abs_diff(ordered_bits(b));
                let count = counts.entry(*name).or_default();
                count.0 += 1;
                count.1 = count.1.max(ulp);
                if ulp > 0 {
                    failures
                        .entry((*name, i))
                        .or_insert((mjd, seconds, a, b, ulp));
                }
            }
        }
    }
    assert!(epochs.next().is_none());
    let empty = Iers2010EarthOrientationProvider::new(Vec::new()).unwrap();
    let otherwise_covered = date::utc(59025, Offset::seconds(0));
    assert!(empty.orientation(otherwise_covered).is_err());
    let fallback = empty
        .with_validity(ValidityMode::Permissive)
        .orientation(otherwise_covered)
        .unwrap();
    assert_eq!(fallback.ut1_degraded(), Some(DegradeReason::BeforeCoverage));
    assert_eq!(counts.values().map(|&(n, _)| n).sum::<usize>(), 4408);
    assert!(failures.is_empty(), "{counts:#?}\n{failures:#?}");
}

#[test]
fn a_single_daily_entry_has_defined_orientation() {
    let sample = Iers2010EopSample {
        mjd_utc: 59025,
        xp_arcsec: 0.15,
        yp_arcsec: 0.43,
        ut1_minus_utc_s: -0.24,
        lod_s: -0.003,
        dx_arcsec: 0.0,
        dy_arcsec: 0.0,
        pole_rates_rad_s: None,
    };
    let provider = Iers2010EarthOrientationProvider::new(vec![sample]).unwrap();
    let epoch = date::utc(sample.mjd_utc, Offset::seconds(0));
    let tt = epoch.plus(TT_MINUS_TAI).as_seconds();
    let tdb = tt + tdb_minus_tt_seconds(tt);
    assert_eq!(
        provider.orientation_at_tdb_seconds(tdb).unwrap(),
        provider
            .orientation_at_utc_mjd(sample.mjd_utc, 0.0)
            .unwrap(),
    );
    assert!(provider
        .orientation_at_utc_mjd(sample.mjd_utc, 86400.0)
        .is_err());
}

#[test]
fn celestial_rotation_at_its_removable_singularities() {
    let s = 5e-7;
    for x in [0.0, -0.003] {
        let limit = Rotation::celestial([x, 0.0, s]);
        // The squared normalization factor is nonzero at these offsets.
        for y in [-1e-150, 1e-150] {
            let regular = Rotation::celestial([x, y, s]);
            for (actual, expected) in limit.0.into_iter().zip(regular.0).chain(
                limit
                    .matrix()
                    .into_iter()
                    .flatten()
                    .zip(regular.matrix().into_iter().flatten()),
            ) {
                assert!(
                    (actual - expected).abs() < 1e-15,
                    "{x}, {y}: {actual} vs {expected}"
                );
            }
        }
    }
}

#[test]
fn tdb_epochs_near_the_signed_second_limits() {
    let provider = Iers2010EarthOrientationProvider::new(Vec::new())
        .unwrap()
        .with_validity(ValidityMode::Permissive);
    let min = i64::MIN as f64;
    let max_exclusive = i64::MAX as f64;

    assert!(matches!(
        provider.orientation_at_tdb_seconds(max_exclusive),
        Err(FrameTransformError::InvalidInput {
            field: "epoch_tdb_seconds",
            ..
        })
    ));
    assert!(matches!(
        provider.orientation_at_tdb_seconds(max_exclusive - 1024.0),
        Err(FrameTransformError::InvalidInput { field: "TT", .. })
    ));
    for epoch in [min, min + 32_768.0, max_exclusive - 32_768.0] {
        assert!(matches!(
            provider.orientation_at_tdb_seconds(epoch),
            Err(FrameTransformError::InvalidInput {
                field: "celestial pole",
                ..
            })
        ));
    }

    let (day, start) = Offset::seconds(i64::MIN).day_and_start().unwrap();
    assert_eq!(day, i64::MIN.div_euclid(86_400));
    assert_eq!(start.whole_seconds(), None);
    assert_eq!(
        Offset::seconds(i64::MIN).minus(start),
        Offset::seconds(i64::MIN.rem_euclid(86_400))
    );
}

#[test]
fn ut1_date_must_fit_the_signed_second_range() {
    let lower = i64::MIN as f64;
    let upper = i64::MAX as f64;
    for (mjd_utc, ut1_minus_utc_s, in_range) in [
        (50_000, lower, false),
        (60_000, upper, false),
        (60_000, f64::MAX, false),
        (50_000, upper, true),
        (60_000, lower, true),
    ] {
        let provider = Iers2010EarthOrientationProvider::new(vec![Iers2010EopSample {
            mjd_utc,
            xp_arcsec: 0.0,
            yp_arcsec: 0.0,
            ut1_minus_utc_s,
            lod_s: 0.0,
            dx_arcsec: 0.0,
            dy_arcsec: 0.0,
            pole_rates_rad_s: None,
        }])
        .unwrap();
        let result = provider.orientation_at_utc_mjd(mjd_utc, 0.0);
        if in_range {
            assert!(result.is_ok(), "{result:?}");
        } else {
            assert!(matches!(
                result,
                Err(FrameTransformError::InvalidInput { field: "UT1", .. })
            ));
        }
    }
}

#[test]
fn rotation_preserves_a_finite_state_when_intermediates_overflow() {
    let state = Rotation([1.0, 0.0, 0.0, 0.0])
        .state([f64::MAX, 0.0, 0.0], [0.0, -f64::MAX, 0.0], [0.0; 3], false)
        .unwrap();
    assert_eq!(state, ([f64::MAX, 0.0, 0.0], [0.0, -f64::MAX, 0.0]));
}
