#![no_main]

use libfuzzer_sys::fuzz_target;
use sidereon_core::terrain::{DtedInterpolation, DtedLookupOptions};
use sidereon_core::terrain_store::MmapTerrain;

fuzz_target!(|data: &[u8]| {
    if let Ok(store) = MmapTerrain::from_bytes(data) {
        let round_trip = store.to_bytes();
        assert_eq!(round_trip, data);
        MmapTerrain::from_bytes(&round_trip).expect("round-trip terrain store parses");

        // Query every accepted tile at its corners, centre and an interior
        // point with bits below one ulp of 1, so index metadata that passed
        // parsing is exercised by the lookup arithmetic.
        let mut nearest = DtedLookupOptions::default();
        nearest.interpolation = DtedInterpolation::NearestPosting;
        let bilinear = DtedLookupOptions::default();
        for tile in store.tile_index() {
            let west = tile.min_longitude_deg;
            let south = tile.min_latitude_deg;
            let points = [
                (west, south),
                (tile.max_longitude_deg, tile.max_latitude_deg),
                (west + 0.5, south + 0.5),
                (
                    west + 0.123_456_789_012_345_6,
                    south + 0.987_654_321_098_765_4,
                ),
            ];
            for (longitude_deg, latitude_deg) in points {
                for options in [nearest, bilinear] {
                    let _ = store.orthometric_height_m_with_options(
                        longitude_deg,
                        latitude_deg,
                        options,
                    );
                }
            }
        }
    }
});
