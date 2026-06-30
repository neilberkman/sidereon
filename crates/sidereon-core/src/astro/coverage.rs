//! One-epoch satellite/station coverage grid.
//!
//! The public helpers here build row-major `[satellite][station]` products by
//! wrapping the scalar look-angle kernel. Every cell equals the corresponding
//! per-pair [`crate::astro::passes::look_angle_arc`] result.

use crate::astro::passes::{look_angle_arc, GroundStation, LookAngle, LookAngleError, UtcInstant};
use crate::astro::sgp4::Satellite;

/// Row-major look-angle grid indexed as `[satellite][station]`.
pub type LookAngleGrid = Vec<Vec<Result<LookAngle, LookAngleError>>>;

/// Compute topocentric look angles for all satellite/station pairs at one epoch.
///
/// Each cell is produced by calling [`look_angle_arc`] for exactly that
/// satellite/station pair with a one-element epoch slice, so the cell is
/// element-wise identical to the scalar kernel.
pub fn look_angles_batch(
    satellites: &[Satellite],
    stations: &[GroundStation],
    datetime: UtcInstant,
) -> LookAngleGrid {
    satellites
        .iter()
        .map(|satellite| {
            stations
                .iter()
                .map(|&station| {
                    look_angle_arc(satellite, station, std::slice::from_ref(&datetime))
                        .map(|arc| arc[0])
                })
                .collect()
        })
        .collect()
}

/// Return true for every successful look angle at or above `min_elevation_deg`.
///
/// Error cells are treated as not visible.
pub fn visible_mask(
    grid: &[Vec<Result<LookAngle, LookAngleError>>],
    min_elevation_deg: f64,
) -> Vec<Vec<bool>> {
    grid.iter()
        .map(|row| {
            row.iter()
                .map(|cell| matches!(cell, Ok(look) if look.elevation_deg >= min_elevation_deg))
                .collect()
        })
        .collect()
}

/// Count visible satellites per station for a look-angle grid.
pub fn access_counts(
    grid: &[Vec<Result<LookAngle, LookAngleError>>],
    min_elevation_deg: f64,
) -> Vec<usize> {
    let Some(first_row) = grid.first() else {
        return Vec::new();
    };
    let mut counts = vec![0; first_row.len()];

    for row in grid {
        for (count, cell) in counts.iter_mut().zip(row) {
            if matches!(cell, Ok(look) if look.elevation_deg >= min_elevation_deg) {
                *count += 1;
            }
        }
    }

    counts
}

/// Return the maximum successful elevation per station.
///
/// Error cells are ignored. A station with no successful cells returns `None`.
pub fn max_elevation(grid: &[Vec<Result<LookAngle, LookAngleError>>]) -> Vec<Option<f64>> {
    let Some(first_row) = grid.first() else {
        return Vec::new();
    };
    let mut elevations: Vec<Option<f64>> = vec![None; first_row.len()];

    for row in grid {
        for (elevation, cell) in elevations.iter_mut().zip(row) {
            if let Ok(look) = cell {
                *elevation = Some(match *elevation {
                    Some(current) => current.max(look.elevation_deg),
                    None => look.elevation_deg,
                });
            }
        }
    }

    elevations
}

#[cfg(test)]
mod tests {
    use super::*;

    const ISS_L1: &str = "1 25544U 98067A   24001.50000000  .00016717  00000-0  10270-3 0  9009";
    const ISS_L2: &str = "2 25544  51.6400 208.8657 0002644 250.3037 109.7782 15.49560812999990";

    #[test]
    fn look_angles_batch_equals_scalar_per_pair() {
        let sats = satellites();
        let stations = stations();
        let datetime = datetime();

        let grid = look_angles_batch(&sats, &stations, datetime);

        assert_eq!(grid.len(), sats.len());
        for (sat_idx, row) in grid.iter().enumerate() {
            assert_eq!(row.len(), stations.len());
            for (station_idx, cell) in row.iter().enumerate() {
                let expected = look_angle_arc(&sats[sat_idx], stations[station_idx], &[datetime])
                    .map(|arc| arc[0]);
                assert_look_angle_result_bits_eq(cell, &expected);
            }
        }
    }

    #[test]
    fn visible_mask_matches_threshold() {
        let mut grid = sample_grid();
        grid[0][0] = Err(LookAngleError::InvalidInput {
            field: "test",
            reason: "forced error",
        });

        for min_elevation_deg in [0.0, 80.0] {
            let mask = visible_mask(&grid, min_elevation_deg);

            assert_eq!(mask.len(), grid.len());
            for (mask_row, grid_row) in mask.iter().zip(&grid) {
                assert_eq!(mask_row.len(), grid_row.len());
                for (visible, cell) in mask_row.iter().zip(grid_row) {
                    let expected =
                        matches!(cell, Ok(look) if look.elevation_deg >= min_elevation_deg);
                    assert_eq!(*visible, expected);
                }
            }
        }
    }

    #[test]
    fn access_counts_sums_mask() {
        let mut grid = sample_grid();
        grid[0][0] = Err(LookAngleError::InvalidInput {
            field: "test",
            reason: "forced error",
        });
        let min_elevation_deg = 0.0;

        let mask = visible_mask(&grid, min_elevation_deg);
        let counts = access_counts(&grid, min_elevation_deg);

        assert_eq!(counts.len(), grid[0].len());
        for station_idx in 0..counts.len() {
            let expected = mask.iter().filter(|row| row[station_idx]).count();
            assert_eq!(counts[station_idx], expected);
        }
    }

    #[test]
    fn max_elevation_reduces_columns() {
        let mut grid = sample_grid();
        grid[0][0] = Err(LookAngleError::InvalidInput {
            field: "test",
            reason: "forced error",
        });

        let reduced = max_elevation(&grid);

        assert_eq!(reduced.len(), grid[0].len());
        for station_idx in 0..reduced.len() {
            let mut expected = None;
            for row in &grid {
                if let Ok(look) = &row[station_idx] {
                    expected = Some(match expected {
                        Some(current) => f64::max(current, look.elevation_deg),
                        None => look.elevation_deg,
                    });
                }
            }
            assert_optional_f64_bits_eq(reduced[station_idx], expected);
        }
    }

    fn sample_grid() -> LookAngleGrid {
        let sats = satellites();
        let stations = stations();
        look_angles_batch(&sats, &stations, datetime())
    }

    fn satellites() -> Vec<Satellite> {
        vec![
            Satellite::from_tle(ISS_L1, ISS_L2).expect("ISS TLE parses"),
            Satellite::from_tle(ISS_L1, ISS_L2).expect("ISS TLE parses"),
        ]
    }

    fn stations() -> Vec<GroundStation> {
        vec![
            GroundStation {
                latitude_deg: 51.5,
                longitude_deg: -0.1,
                altitude_m: 11.0,
            },
            GroundStation {
                latitude_deg: 40.7,
                longitude_deg: -74.0,
                altitude_m: 10.0,
            },
        ]
    }

    fn datetime() -> UtcInstant {
        UtcInstant::from_utc(2024, 1, 1, 12, 0, 0, 0).unwrap()
    }

    fn assert_look_angle_result_bits_eq(
        actual: &Result<LookAngle, LookAngleError>,
        expected: &Result<LookAngle, LookAngleError>,
    ) {
        match (actual, expected) {
            (Ok(actual), Ok(expected)) => {
                assert_eq!(actual.azimuth_deg.to_bits(), expected.azimuth_deg.to_bits());
                assert_eq!(
                    actual.elevation_deg.to_bits(),
                    expected.elevation_deg.to_bits()
                );
                assert_eq!(actual.range_km.to_bits(), expected.range_km.to_bits());
            }
            (Err(actual), Err(expected)) => assert_eq!(actual, expected),
            _ => panic!("actual {actual:?} did not match expected {expected:?}"),
        }
    }

    fn assert_optional_f64_bits_eq(actual: Option<f64>, expected: Option<f64>) {
        match (actual, expected) {
            (Some(actual), Some(expected)) => assert_eq!(actual.to_bits(), expected.to_bits()),
            (None, None) => {}
            _ => panic!("actual {actual:?} did not match expected {expected:?}"),
        }
    }
}
