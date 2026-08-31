//! Provenance: `tests/fixtures/geodesic/geodtest_subset.dat` is a 90-row
//! subset of Charles F. F. Karney's public `GeodTest.dat` WGS84 geodesic test
//! set, selected as the first ten rows from each published category block:
//! random, near-antipodal, short, one endpoint near a pole, endpoints near
//! opposite poles, near-meridional, near-equatorial, vertex-to-vertex, and
//! endpoint-near-vertex. Source:
//! <https://sourceforge.net/projects/geographiclib/files/testdata/GeodTest.dat.gz/download>.
//! Reference paper: Karney (2013), "Algorithms for geodesics", Journal of
//! Geodesy 87(1), 43-55, DOI 10.1007/s00190-012-0578-z. The public dataset
//! gives `s12` as an exact decimal meter value for the generated case and gives
//! `lat2`, `lon2`, and `azi2` to 1e-18 degrees.

use sidereon_core::constants::{DEG_TO_RAD, WGS84_A_M};
use sidereon_core::{geodesic_direct, geodesic_inverse};

const GEODTEST_SUBSET: &str = include_str!("fixtures/geodesic/geodtest_subset.dat");
const INVERSE_DISTANCE_TOL_M: f64 = 1.0e-8;
const AZIMUTH_TOL_DEG: f64 = 5.0e-13;
// Several GeodTest categories print direct-problem azimuth inputs for endpoints
// where the inverse azimuth is ill-conditioned. These categories still assert
// numeric value bounds; they are not finiteness escapes.
const NEAR_ANTIPODAL_AZIMUTH_TOL_DEG: f64 = 5.0e-12;
const SHORT_LINE_AZIMUTH_TOL_DEG: f64 = 1.0e-9;
const OPPOSITE_POLE_AZIMUTH_TOL_DEG: f64 = 5.0e-11;
const STRESS_AZIMUTH_TOL_DEG: f64 = 1.0e-9;
const VERTEX_TO_VERTEX_AZIMUTH_TOL_DEG: f64 = 5.0e-5;
const ENDPOINT_NEAR_VERTEX_AZIMUTH_TOL_DEG: f64 = 5.0e-5;
const DIRECT_POSITION_TOL_M: f64 = 1.0e-8;
const GEODTEST_CATEGORIES: [&str; 9] = [
    "random",
    "near-antipodal",
    "short",
    "one-near-pole",
    "opposite-poles",
    "near-meridional",
    "near-equatorial",
    "vertex-to-vertex",
    "endpoint-near-vertex",
];

#[derive(Debug, Clone, Copy)]
struct GeodTestCase {
    lat1_deg: f64,
    lon1_deg: f64,
    azi1_deg: f64,
    lat2_deg: f64,
    lon2_deg: f64,
    azi2_deg: f64,
    s12_m: f64,
}

fn geodtest_cases() -> Vec<GeodTestCase> {
    GEODTEST_SUBSET
        .lines()
        .enumerate()
        .map(|(idx, line)| {
            let fields: Vec<f64> = line
                .split_whitespace()
                .take(7)
                .map(|field| {
                    field.parse::<f64>().unwrap_or_else(|err| {
                        panic!("line {} invalid float {field:?}: {err}", idx + 1)
                    })
                })
                .collect();
            assert_eq!(fields.len(), 7, "line {} field count", idx + 1);
            GeodTestCase {
                lat1_deg: fields[0],
                lon1_deg: fields[1],
                azi1_deg: fields[2],
                lat2_deg: fields[3],
                lon2_deg: fields[4],
                azi2_deg: fields[5],
                s12_m: fields[6],
            }
        })
        .collect()
}

fn angle_diff_deg(actual: f64, expected: f64) -> f64 {
    let mut diff = (actual - expected).rem_euclid(360.0);
    if diff > 180.0 {
        diff -= 360.0;
    }
    diff
}

fn local_position_error_m(actual_lat_deg: f64, actual_lon_deg: f64, case: GeodTestCase) -> f64 {
    let north_m = (actual_lat_deg - case.lat2_deg) * DEG_TO_RAD * WGS84_A_M;
    let east_m = angle_diff_deg(actual_lon_deg, case.lon2_deg)
        * DEG_TO_RAD
        * WGS84_A_M
        * libm::cos(case.lat2_deg * DEG_TO_RAD).abs();
    libm::hypot(north_m, east_m)
}

fn inverse_azimuth_tolerance_deg(index: usize) -> f64 {
    if (10..20).contains(&index) {
        NEAR_ANTIPODAL_AZIMUTH_TOL_DEG
    } else if (20..30).contains(&index) {
        SHORT_LINE_AZIMUTH_TOL_DEG
    } else if (40..50).contains(&index) {
        OPPOSITE_POLE_AZIMUTH_TOL_DEG
    } else if (70..80).contains(&index) {
        VERTEX_TO_VERTEX_AZIMUTH_TOL_DEG
    } else if (80..90).contains(&index) {
        ENDPOINT_NEAR_VERTEX_AZIMUTH_TOL_DEG
    } else {
        AZIMUTH_TOL_DEG
    }
}

fn direct_azimuth_tolerance_deg(case: GeodTestCase) -> f64 {
    if case.s12_m > 19_000_000.0 || case.s12_m < 1_000.0 {
        STRESS_AZIMUTH_TOL_DEG
    } else {
        AZIMUTH_TOL_DEG
    }
}

fn category_index(case_index: usize) -> usize {
    case_index / 10
}

#[test]
fn inverse_matches_public_geodtest_subset() {
    let mut checked = 0usize;
    let mut max_s12_err_m = 0.0_f64;
    let mut max_azi_err_deg = 0.0_f64;
    let mut max_stress_azi_err_deg = 0.0_f64;
    let mut category_s12_err_m = [0.0_f64; GEODTEST_CATEGORIES.len()];
    let mut category_azi_err_deg = [0.0_f64; GEODTEST_CATEGORIES.len()];
    for case in geodtest_cases() {
        let (s12_m, azi1_deg, azi2_deg) =
            geodesic_inverse(case.lat1_deg, case.lon1_deg, case.lat2_deg, case.lon2_deg)
                .expect("inverse geodesic");
        let azi_tol_deg = inverse_azimuth_tolerance_deg(checked);
        max_s12_err_m = max_s12_err_m.max((s12_m - case.s12_m).abs());
        let azi_err_deg = angle_diff_deg(azi1_deg, case.azi1_deg)
            .abs()
            .max(angle_diff_deg(azi2_deg, case.azi2_deg).abs());
        let category = category_index(checked);
        category_s12_err_m[category] = category_s12_err_m[category].max((s12_m - case.s12_m).abs());
        category_azi_err_deg[category] = category_azi_err_deg[category].max(azi_err_deg);
        if azi_tol_deg == AZIMUTH_TOL_DEG {
            max_azi_err_deg = max_azi_err_deg.max(azi_err_deg);
        } else {
            max_stress_azi_err_deg = max_stress_azi_err_deg.max(azi_err_deg);
        }

        assert!(
            (s12_m - case.s12_m).abs() <= INVERSE_DISTANCE_TOL_M,
            "case {checked} s12 actual={s12_m:.17e} expected={:.17e} err={:.17e}",
            case.s12_m,
            (s12_m - case.s12_m).abs()
        );
        if azi_tol_deg == AZIMUTH_TOL_DEG {
            assert!(
                angle_diff_deg(azi1_deg, case.azi1_deg).abs() <= azi_tol_deg,
                "case {checked} azi1 actual={azi1_deg:.17e} expected={:.17e} err={:.17e}",
                case.azi1_deg,
                angle_diff_deg(azi1_deg, case.azi1_deg).abs()
            );
            assert!(
                angle_diff_deg(azi2_deg, case.azi2_deg).abs() <= azi_tol_deg,
                "case {checked} azi2 actual={azi2_deg:.17e} expected={:.17e} err={:.17e}",
                case.azi2_deg,
                angle_diff_deg(azi2_deg, case.azi2_deg).abs()
            );
        } else {
            assert!(
                angle_diff_deg(azi1_deg, case.azi1_deg).abs() <= azi_tol_deg,
                "case {checked} stress azi1 actual={azi1_deg:.17e} expected={:.17e} err={:.17e}",
                case.azi1_deg,
                angle_diff_deg(azi1_deg, case.azi1_deg).abs()
            );
            assert!(
                angle_diff_deg(azi2_deg, case.azi2_deg).abs() <= azi_tol_deg,
                "case {checked} stress azi2 actual={azi2_deg:.17e} expected={:.17e} err={:.17e}",
                case.azi2_deg,
                angle_diff_deg(azi2_deg, case.azi2_deg).abs()
            );
        }
        checked += 1;
    }
    eprintln!(
        "max inverse errors: s12={max_s12_err_m:.17e} m, azimuth={max_azi_err_deg:.17e} deg, stress azimuth={max_stress_azi_err_deg:.17e} deg"
    );
    for (idx, name) in GEODTEST_CATEGORIES.iter().enumerate() {
        eprintln!(
            "inverse {name}: s12={:.17e} m, azimuth={:.17e} deg",
            category_s12_err_m[idx], category_azi_err_deg[idx]
        );
    }
    assert_eq!(checked, 90);
}

#[test]
fn direct_matches_public_geodtest_subset() {
    let mut checked = 0usize;
    let mut max_position_err_m = 0.0_f64;
    let mut max_azi_err_deg = 0.0_f64;
    let mut max_stress_azi_err_deg = 0.0_f64;
    let mut category_position_err_m = [0.0_f64; GEODTEST_CATEGORIES.len()];
    let mut category_azi_err_deg = [0.0_f64; GEODTEST_CATEGORIES.len()];
    for case in geodtest_cases() {
        let (lat2_deg, lon2_deg, azi2_deg) =
            geodesic_direct(case.lat1_deg, case.lon1_deg, case.azi1_deg, case.s12_m)
                .expect("direct geodesic");
        let position_error_m = local_position_error_m(lat2_deg, lon2_deg, case);
        let azi_tol_deg = direct_azimuth_tolerance_deg(case);
        max_position_err_m = max_position_err_m.max(position_error_m);
        let azi_err_deg = angle_diff_deg(azi2_deg, case.azi2_deg).abs();
        let category = category_index(checked);
        category_position_err_m[category] = category_position_err_m[category].max(position_error_m);
        category_azi_err_deg[category] = category_azi_err_deg[category].max(azi_err_deg);
        if azi_tol_deg == AZIMUTH_TOL_DEG {
            max_azi_err_deg = max_azi_err_deg.max(azi_err_deg);
        } else {
            max_stress_azi_err_deg = max_stress_azi_err_deg.max(azi_err_deg);
        }

        assert!(
            position_error_m <= DIRECT_POSITION_TOL_M,
            "case {checked} endpoint position error {position_error_m:.17e} m"
        );
        assert!(
            angle_diff_deg(azi2_deg, case.azi2_deg).abs() <= azi_tol_deg,
            "case {checked} azi2 actual={azi2_deg:.17e} expected={:.17e} err={:.17e}",
            case.azi2_deg,
            angle_diff_deg(azi2_deg, case.azi2_deg).abs()
        );
        checked += 1;
    }
    eprintln!(
        "max direct errors: endpoint={max_position_err_m:.17e} m, azimuth={max_azi_err_deg:.17e} deg, stress azimuth={max_stress_azi_err_deg:.17e} deg"
    );
    for (idx, name) in GEODTEST_CATEGORIES.iter().enumerate() {
        eprintln!(
            "direct {name}: endpoint={:.17e} m, azimuth={:.17e} deg",
            category_position_err_m[idx], category_azi_err_deg[idx]
        );
    }
    assert_eq!(checked, 90);
}

#[test]
fn direct_inverse_closes_to_nanometers() {
    let mut checked = 0usize;
    let mut max_position_err_m = 0.0_f64;
    let mut category_position_err_m = [0.0_f64; GEODTEST_CATEGORIES.len()];
    for case in geodtest_cases() {
        let (s12_m, azi1_deg, _azi2_deg) =
            geodesic_inverse(case.lat1_deg, case.lon1_deg, case.lat2_deg, case.lon2_deg)
                .expect("inverse geodesic");
        let (lat2_deg, lon2_deg, _azi2_deg) =
            geodesic_direct(case.lat1_deg, case.lon1_deg, azi1_deg, s12_m)
                .expect("direct geodesic");
        let position_error_m = local_position_error_m(lat2_deg, lon2_deg, case);
        max_position_err_m = max_position_err_m.max(position_error_m);
        let category = category_index(checked);
        category_position_err_m[category] = category_position_err_m[category].max(position_error_m);

        assert!(
            position_error_m <= DIRECT_POSITION_TOL_M,
            "case {checked} closure position error {position_error_m:.17e} m"
        );
        checked += 1;
    }
    eprintln!("max closure endpoint error: {max_position_err_m:.17e} m");
    for (idx, name) in GEODTEST_CATEGORIES.iter().enumerate() {
        eprintln!(
            "closure {name}: endpoint={:.17e} m",
            category_position_err_m[idx]
        );
    }
    assert_eq!(checked, 90);
}

#[test]
fn direct_accepts_negative_longitude_start() {
    let (lat2_deg, lon2_deg, azi2_deg) =
        geodesic_direct(40.64, -73.78, 45.0, 10_000_000.0).expect("direct geodesic");

    assert!((lat2_deg - 32.621_100_463_725_796).abs() <= 1.0e-14);
    assert!((lon2_deg - 49.052_487_092_959_82).abs() <= 1.0e-14);
    assert!((azi2_deg - 140.405_985_876_800_7).abs() <= 1.0e-13);
}
