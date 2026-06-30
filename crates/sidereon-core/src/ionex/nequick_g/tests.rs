//! NeQuick-G validation against the official Galileo reference vectors.
//!
//! All expected values are taken from the European Commission Joint Research
//! Centre NeQuick-G reference implementation test suite (release 2019-12-10,
//! algorithm spec *Ionospheric Correction Algorithm for Galileo Single Frequency
//! Users*, issue 1.2, September 2016):
//!
//! - MODIP grid points: `src/lib/UT/NeQuickG_JRC_MODIP_test.c` (tolerance 1e-5).
//! - Effective ionisation level `Az`: `src/lib/UT/NeQuickG_JRC_Az_test.c`
//!   (tolerance 1e-6).
//! - Ray-perigee geometry: `src/lib/UT/NeQuickG_JRC_ray_test.c` (tolerance 1e-5).
//! - End-to-end slant TEC: the reference `test/benchmark/benchmark{High,Mid,Low}`
//!   files, whose first line is the `Az` coefficient triple and whose rows are
//!   `month UT staLon staLat staHeight_m satLon satLat satHeight_m STEC_TECU`.
//!   The reference prints STEC to five decimals, so the documented tolerance is
//!   1e-4 TECU.

use super::*;
use crate::atmosphere::ionosphere::{
    galileo_nequick_g_native, klobuchar_native, GalileoNequickCoeffs,
};

const AZ_HIGH: GalileoNequickCoeffs = GalileoNequickCoeffs {
    ai0: 236.831641,
    ai1: -0.39362878,
    ai2: 0.00402826613,
};
const AZ_MEDIUM: GalileoNequickCoeffs = GalileoNequickCoeffs {
    ai0: 121.129893,
    ai1: 0.351254133,
    ai2: 0.0134635348,
};
const AZ_LOW: GalileoNequickCoeffs = GalileoNequickCoeffs {
    ai0: 2.580271,
    ai1: 0.127628236,
    ai2: 0.0252748384,
};

const GALILEO_E1_HZ: f64 = 1_575_420_000.0;

#[test]
fn modip_grid_matches_reference() {
    // (longitude_deg, latitude_deg, expected_modip_deg) from MODIP_test.c.
    let vectors = [
        (297.65954, 82.49429, 76.28407),
        (307.19404, 5.25218, 19.52877),
        (355.75034, 40.42916, 47.85769),
        (40.19439, -2.99591, -23.31631),
        (166.66933, -77.83835, -71.81130),
        (141.13283, 39.13517, 46.48742),
        (204.54366, 19.80135, 33.05457),
        (115.88525, -31.80197, -51.37982),
    ];
    for (lon, lat, expected) in vectors {
        let got = modip_degree(lon, lat);
        assert!(
            (got - expected).abs() < 1.0e-5,
            "modip(lon={lon}, lat={lat}) = {got}, expected {expected}"
        );
    }
}

#[test]
fn effective_ionisation_level_matches_reference() {
    // (coeffs, modip_deg, expected_Az_sfu) from Az_test.c.
    let vectors = [
        (AZ_HIGH, 76.284073, 230.245562),
        (AZ_MEDIUM, 76.284073, 226.272795),
        (AZ_LOW, 76.284073, 159.397123),
        (AZ_HIGH, 19.528774, 230.680826),
        (AZ_MEDIUM, 19.528774, 133.124084),
        (AZ_LOW, 19.528774, 14.711835),
        (AZ_HIGH, -71.811295, 285.871846),
        (AZ_MEDIUM, -71.811295, 165.335471),
        (AZ_LOW, -71.811295, 123.753978),
    ];
    for (coeffs, modip, expected) in vectors {
        let got = effective_ionisation_level_sfu(&coeffs, modip);
        assert!(
            (got - expected).abs() < 1.0e-6,
            "Az(modip={modip}) = {got}, expected {expected}"
        );
    }
}

#[test]
fn ray_perigee_geometry_matches_reference() {
    // ray_test.c: station and satellite heights are in km in the reference vector.
    let station = Position::new(297.659539798, 82.494293510, 0.078107446 * 1.0e3);
    let satellite = Position::new(241.529931024, 54.445029416, 20370.730845002 * 1.0e3);
    let geometry = RayGeometry::new(&station, &satellite).expect("valid ray");

    assert!(!geometry.is_vertical);
    assert!((geometry.perigee_radius_km - 4169.486317342).abs() < 1.0e-5);
    assert!((geometry.azimuth_sin - (-0.164640718)).abs() < 1.0e-5);
    assert!((geometry.azimuth_cos - 0.986353605).abs() < 1.0e-5);
    assert!((geometry.perigee_lat.deg - 43.550617197).abs() < 1.0e-5);
    assert!((geometry.perigee_lon.deg - 405.289045373).abs() < 1.0e-5);
}

struct Bench {
    coeffs: GalileoNequickCoeffs,
    ray: NequickGRayEval,
    stec: f64,
}

#[allow(clippy::too_many_arguments)]
fn bench(
    coeffs: GalileoNequickCoeffs,
    month: u8,
    utc: f64,
    s_lon: f64,
    s_lat: f64,
    s_h: f64,
    v_lon: f64,
    v_lat: f64,
    v_h: f64,
    stec: f64,
) -> Bench {
    Bench {
        coeffs,
        ray: NequickGRayEval {
            month,
            utc_hours: utc,
            station_lon_deg: s_lon,
            station_lat_deg: s_lat,
            station_height_m: s_h,
            satellite_lon_deg: v_lon,
            satellite_lat_deg: v_lat,
            satellite_height_m: v_h,
        },
        stec,
    }
}

fn reference_benchmarks() -> Vec<Bench> {
    vec![
        // test/benchmark/benchmarkHigh
        bench(
            AZ_HIGH,
            4,
            0.0,
            297.66,
            82.49,
            78.11,
            8.23,
            54.29,
            20281546.18,
            20.40224,
        ),
        bench(
            AZ_HIGH,
            4,
            4.0,
            297.66,
            82.49,
            78.11,
            -85.72,
            53.69,
            20544786.65,
            18.77379,
        ),
        bench(
            AZ_HIGH,
            4,
            16.0,
            297.66,
            82.49,
            78.11,
            -70.26,
            50.63,
            20043030.82,
            23.92627,
        ),
        bench(
            AZ_HIGH,
            4,
            4.0,
            307.19,
            5.25,
            -25.76,
            -18.13,
            14.17,
            20267783.18,
            95.52199,
        ),
        bench(
            AZ_HIGH,
            4,
            20.0,
            307.19,
            5.25,
            -25.76,
            10.94,
            44.72,
            20450566.19,
            336.73204,
        ),
        // test/benchmark/benchmarkMid
        bench(
            AZ_MEDIUM,
            4,
            0.0,
            40.19,
            -3.00,
            -23.32,
            76.65,
            -41.43,
            20157673.93,
            18.26001,
        ),
        bench(
            AZ_MEDIUM,
            4,
            8.0,
            40.19,
            -3.00,
            -23.32,
            89.22,
            -40.56,
            20055109.63,
            101.24016,
        ),
        bench(
            AZ_MEDIUM,
            4,
            0.0,
            115.89,
            -31.80,
            12.78,
            119.90,
            -8.76,
            19941513.27,
            24.84680,
        ),
        bench(
            AZ_MEDIUM,
            4,
            12.0,
            115.89,
            -31.80,
            12.78,
            133.47,
            -24.87,
            19975574.41,
            13.63080,
        ),
        // test/benchmark/benchmarkLow
        bench(
            AZ_LOW,
            4,
            0.0,
            141.13,
            39.14,
            117.00,
            165.14,
            -13.93,
            20181976.50,
            36.44498,
        ),
        bench(
            AZ_LOW,
            4,
            12.0,
            141.13,
            39.14,
            117.00,
            115.63,
            -1.28,
            20165065.92,
            11.05962,
        ),
        bench(
            AZ_LOW,
            4,
            0.0,
            204.54,
            19.80,
            3754.69,
            -144.16,
            -15.44,
            20007317.84,
            72.82501,
        ),
        bench(
            AZ_LOW,
            4,
            16.0,
            204.54,
            19.80,
            3754.69,
            -167.50,
            -43.24,
            20095343.11,
            3.11261,
        ),
    ]
}

#[test]
fn slant_tec_matches_reference_benchmarks() {
    for b in reference_benchmarks() {
        let got = nequick_g_stec_tecu(&b.coeffs, &b.ray).expect("valid NeQuick-G ray");
        assert!(
            (got - b.stec).abs() < 1.0e-4,
            "STEC for ray {:?} = {got}, expected {}",
            b.ray,
            b.stec
        );
    }
}

#[test]
fn delay_is_tec_scaled_by_dispersive_factor() {
    let b = &reference_benchmarks()[0];
    let stec = nequick_g_stec_tecu(&b.coeffs, &b.ray).expect("valid ray");
    let delay = nequick_g_delay_m(&b.coeffs, &b.ray, GALILEO_E1_HZ).expect("valid ray");
    let expected = stec * (40.3e16 / (GALILEO_E1_HZ * GALILEO_E1_HZ));
    assert!((delay - expected).abs() < 1.0e-9);
    // a positive, physically plausible E1 slant delay (a few metres at ~20 TECU)
    assert!(delay > 0.0 && delay < 100.0);
}

#[test]
fn satellite_at_receiver_gives_zero_tec() {
    // API_test.c: receiver and satellite co-located returns ~0 TEC.
    let coeffs = GalileoNequickCoeffs {
        ai0: 0.0,
        ai1: 0.0,
        ai2: 0.0,
    };
    let ray = NequickGRayEval {
        month: 1,
        utc_hours: 0.0,
        station_lon_deg: 0.0,
        station_lat_deg: 0.0,
        station_height_m: 0.0,
        satellite_lon_deg: 0.0,
        satellite_lat_deg: 0.0,
        satellite_height_m: 0.0,
    };
    let stec = nequick_g_stec_tecu(&coeffs, &ray).expect("valid ray");
    assert!(stec.abs() < 1.0e-10, "co-located STEC = {stec}");
}

#[test]
fn sub_surface_ray_is_rejected() {
    // API_test.c test_bad_ray: receiver below the surface, grazing satellite.
    let coeffs = GalileoNequickCoeffs {
        ai0: 0.0,
        ai1: 0.0,
        ai2: 0.0,
    };
    let ray = NequickGRayEval {
        month: 1,
        utc_hours: 0.0,
        station_lon_deg: 0.0,
        station_lat_deg: 0.0,
        station_height_m: -3_000_000.0,
        satellite_lon_deg: 0.0,
        satellite_lat_deg: 90.0,
        satellite_height_m: 2_000_000.0,
    };
    assert!(nequick_g_stec_tecu(&coeffs, &ray).is_err());
}

#[test]
fn rejects_invalid_inputs() {
    let coeffs = AZ_HIGH;
    let base = reference_benchmarks()[0].ray;

    let bad_month = NequickGRayEval { month: 13, ..base };
    assert!(nequick_g_stec_tecu(&coeffs, &bad_month).is_err());

    let bad_utc = NequickGRayEval {
        utc_hours: 25.0,
        ..base
    };
    assert!(nequick_g_stec_tecu(&coeffs, &bad_utc).is_err());

    let bad_lat = NequickGRayEval {
        station_lat_deg: 91.0,
        ..base
    };
    assert!(nequick_g_stec_tecu(&coeffs, &bad_lat).is_err());

    assert!(nequick_g_delay_m(&coeffs, &base, -1.0).is_err());
}

/// Regression: the Galileo single-frequency correction path must use NeQuick-G
/// and never fall back to the GPS Klobuchar model.
#[test]
fn galileo_single_frequency_path_uses_nequick_g_not_klobuchar() {
    let b = &reference_benchmarks()[0];

    // The full 3-D NeQuick-G model reproduces the reference STEC for this ray,
    // so the Galileo path is genuinely running NeQuick-G (benchmarkHigh row 1).
    let nequick_delay = nequick_g_delay_m(&b.coeffs, &b.ray, GALILEO_E1_HZ).expect("valid ray");
    let nequick_stec = nequick_g_stec_tecu(&b.coeffs, &b.ray).expect("valid ray");
    assert!((nequick_stec - b.stec).abs() < 1.0e-4);
    assert!(nequick_delay > 0.0);

    // The compact broadcast-driven Galileo entry also consumes the Galileo
    // coefficients (not GPS alpha/beta): a non-zero broadcast set yields a delay
    // distinct from the all-zero-default case.
    let galileo_compact = galileo_nequick_g_native(
        &b.coeffs,
        super::super::GalileoNequickEval {
            lat_deg: b.ray.station_lat_deg,
            // the compact helper takes longitude in [-180, 180]
            lon_deg: b.ray.station_lon_deg - 360.0,
            el_deg: 40.0,
            t_gal_s: 3600.0 * b.ray.utc_hours,
            day_of_year: 101.0,
            frequency_hz: GALILEO_E1_HZ,
        },
    )
    .expect("valid compact Galileo eval");

    // A GPS Klobuchar delay for a comparable observation uses entirely different
    // coefficients; the Galileo correction must not collapse onto it.
    let klobuchar = klobuchar_native(
        &super::super::KlobucharParams {
            alpha: [1.0e-8, 0.0, 0.0, 0.0],
            beta: [90_000.0, 0.0, 0.0, 0.0],
        },
        b.ray.station_lat_deg,
        b.ray.station_lon_deg - 360.0,
        0.0,
        40.0,
        3600.0 * b.ray.utc_hours,
        GALILEO_E1_HZ,
    )
    .expect("valid Klobuchar eval");

    assert!(galileo_compact > 0.0);
    assert!(
        (nequick_delay - klobuchar).abs() > 1.0e-3,
        "NeQuick-G delay {nequick_delay} must differ from Klobuchar {klobuchar}"
    );
}
