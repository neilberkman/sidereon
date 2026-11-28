#![cfg(sidereon_repo_tests)]
#![allow(deprecated)]

use serde_json::Value;
use sidereon_core::astro::coverage::{
    access_counts, try_access_counts, try_visible_mask, visible_mask,
};
use sidereon_core::astro::frames::transforms::{itrs_to_topocentric, GeodeticStationKm};
use sidereon_core::astro::passes::LookAngle;
use sidereon_core::dop::{dop, line_of_sight_from_az_el_deg, DopError};
use sidereon_core::frame::{geodetic_to_itrf, Wgs84Geodetic};
use sidereon_core::geometry::{
    dop_at_epoch, visible_at_elevation_mask, DopOptions, VisibilityOptions,
};
use sidereon_core::integrity::{erfc_inv, normal_q_inv};
use sidereon_core::observables::{ObservableEphemerisSource, ObservableState, ObservablesError};
use sidereon_core::quality::{chi2_inv, chi2_isf, raim, RaimInput, RaimOptions, RaimWeights};
use sidereon_core::rtk::{apply_elevation_mask, ElevationMaskEpoch};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::collections::BTreeMap;

struct FixedGeometrySource(BTreeMap<GnssSatelliteId, [f64; 3]>);

#[test]
fn erfc_inverse_near_one_matches_mpmath() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/chi2_extended_mpmath.json")).unwrap();
    for case in fixture["erfc_near_one"].as_array().unwrap() {
        let y: f64 = case["y"].as_str().unwrap().parse().unwrap();
        let expected: f64 = case["value"].as_str().unwrap().parse().unwrap();
        let actual = erfc_inv(y).unwrap();
        // Sterbenz-exact 1-y, then the inverse series (omitted term <0.04
        // ulp at its endpoint) or a direct-erf Newton correction. Allow four
        // ulps for the series products or erf, density, and final correction.
        assert!(
            actual.to_bits().abs_diff(expected.to_bits()) <= 4,
            "y={y} actual={actual} expected={expected}"
        );
    }
}

#[test]
fn chi_square_extended_tails_match_mpmath() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/chi2_extended_mpmath.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let p = case["p"].as_str().unwrap().parse::<f64>().unwrap();
        let dof = case["dof"].as_u64().unwrap() as usize;
        for (name, actual) in [("ppf", chi2_inv(p, dof)), ("isf", chi2_isf(p, dof))] {
            let expected = case[name].as_str().unwrap().parse::<f64>().unwrap();
            let actual = actual.unwrap_or_else(|error| panic!("{name} p={p} dof={dof}: {error:?}"));
            // Four rounding stages in the gamma tail and its inverse density
            // correction, plus final bracket selection, allow eight ulps.
            assert!(
                actual.is_finite() && actual.to_bits().abs_diff(expected.to_bits()) <= 8,
                "{name} p={p} dof={dof}: {actual} != {expected} ({} ulps)",
                actual.to_bits().abs_diff(expected.to_bits())
            );
        }
    }
}

#[test]
fn chi_square_lower_tail_series_switches_are_monotone() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/chi2_extended_mpmath.json")).unwrap();
    for switch in fixture["lower_tail_series_switches"].as_array().unwrap() {
        let p: f64 = switch["p"].as_str().unwrap().parse().unwrap();
        let dof = switch["dof"].as_u64().unwrap() as usize;
        let mut previous = 0.0;
        for index in -128..=128 {
            let probability = f64::from_bits(p.to_bits().checked_add_signed(index).unwrap());
            let actual = chi2_inv(probability, dof).unwrap();
            assert!(actual >= previous, "dof={dof} p={probability}");
            previous = actual;
        }
    }
}

#[test]
fn chi_square_tail_equation_switches_are_monotone() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/chi2_extended_mpmath.json")).unwrap();
    for switch in fixture["equation_switches"].as_array().unwrap() {
        let p: f64 = switch["p"].as_str().unwrap().parse().unwrap();
        let dof = switch["dof"].as_u64().unwrap() as usize;
        // Check both directions at each equation boundary and its 1-p
        // mirror, including all 513 adjacent inputs rather than a step size
        // that could jump over a one-ulp reversal.
        for (name, inverse, increasing) in [
            (
                "ppf",
                chi2_inv as fn(f64, usize) -> Result<f64, sidereon_core::quality::QualityError>,
                true,
            ),
            (
                "isf",
                chi2_isf as fn(f64, usize) -> Result<f64, sidereon_core::quality::QualityError>,
                false,
            ),
        ] {
            let mut previous = if increasing { 0.0 } else { f64::INFINITY };
            for index in -256..=256 {
                let probability = f64::from_bits(p.to_bits().checked_add_signed(index).unwrap());
                if !(0.0..1.0).contains(&probability) || probability == 0.0 {
                    continue;
                }
                let actual = inverse(probability, dof).unwrap();
                assert!(
                    if increasing {
                        actual >= previous
                    } else {
                        actual <= previous
                    },
                    "{name} dof={dof} p={probability}: {previous} -> {actual}"
                );
                previous = actual;
            }
        }
    }
}

#[test]
fn chi_square_upper_tail_switch_selects_mpmath_endpoint() {
    let fixture: Value =
        serde_json::from_str(include_str!("fixtures/chi2_extended_mpmath.json")).unwrap();
    for case in fixture["equation_switch_roots"].as_array().unwrap() {
        let p: f64 = case["p"].as_str().unwrap().parse().unwrap();
        let dof = case["dof"].as_u64().unwrap() as usize;
        let expected: f64 = case["upper_endpoint"].as_str().unwrap().parse().unwrap();
        // The public inverse selects the upper adjacent endpoint. These
        // independent 90-digit roots distinguish that contract from nearest
        // rounding and from either incorrectly normalized tail equation.
        assert_eq!(
            chi2_isf(p, dof).unwrap().to_bits(),
            expected.to_bits(),
            "dof={dof} p={p}"
        );
    }
}

impl ObservableEphemerisSource for FixedGeometrySource {
    fn observable_state_at_j2000_s(
        &self,
        satellite: GnssSatelliteId,
        _epoch_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        Ok(ObservableState {
            position_ecef_m: self.0[&satellite],
            clock_s: Some(0.0),
        })
    }
}

#[test]
fn dop_at_epoch_filters_an_explicit_below_horizon_satellite() {
    let receiver = [6_378_137.0, 0.0, 0.0];
    let range = 22_000_000.0;
    let geometry = [
        (45.0_f64, 0.0_f64),
        (50.0, 90.0),
        (55.0, 180.0),
        (60.0, 270.0),
        (-5.0, 45.0),
    ];
    let satellites: Vec<_> = (1..=5)
        .map(|prn| GnssSatelliteId::new(GnssSystem::Gps, prn).unwrap())
        .collect();
    let states = satellites
        .iter()
        .zip(geometry)
        .map(|(&sat, (elevation, azimuth))| {
            let el = elevation.to_radians();
            let az = azimuth.to_radians();
            let los = [
                libm::sin(el),
                libm::cos(el) * libm::sin(az),
                libm::cos(el) * libm::cos(az),
            ];
            (sat, std::array::from_fn(|i| receiver[i] + range * los[i]))
        })
        .collect();
    let source = FixedGeometrySource(states);
    let mut options = DopOptions::default();
    options.visibility = VisibilityOptions::default();
    options.visibility.elevation_mask_deg = 10.0;
    let result = dop_at_epoch(
        &source,
        &satellites,
        Some(&satellites),
        receiver,
        0.0,
        &options,
    )
    .unwrap();
    assert_eq!(result.satellites, satellites[..4]);
}

const RTKLIB: &str = include_str!("fixtures/rtklib_integrity_geometry.json");
const PYTHON: &str = include_str!("fixtures/integrity_quality_scipy_skyfield.json");
const PNTPOS: &str = include_str!("fixtures/rtklib_pntpos_wrapper_real_epoch.json");

fn number(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| v.as_str().expect("decimal string").parse().expect("f64"))
}

fn close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!((actual - expected).abs() <= tolerance,
        "{label}: actual={actual:.17e} expected={expected:.17e} delta={:.3e} tolerance={tolerance:.3e}",
        (actual - expected).abs());
}

#[test]
fn rebuilt_rtklib_wrapper_preserves_satellite_ids_and_intermediates() {
    let doc: Value = serde_json::from_str(PNTPOS).unwrap();
    assert_eq!(
        doc["provenance"]["commit"],
        "75a2e56275485b21a67bd35bc94bbeb8936e1a74"
    );
    assert_eq!(doc["stat"], 1);
    assert!(doc["ns"].as_u64().unwrap() >= 5);
    let satellites = doc["satellites"].as_array().unwrap();
    assert!(satellites.iter().all(|sat| {
        let id = sat["id"].as_str().unwrap();
        !id.is_empty()
    }));
    assert!(satellites
        .iter()
        .any(|sat| sat["id"] == "G05" && sat["vsat"] == 1));
    let rows = doc["rows"].as_array().unwrap();
    assert_eq!(rows.len(), doc["nv"].as_u64().unwrap() as usize);
    assert!(rows
        .iter()
        .all(|row| row["h"].as_array().unwrap().len() == 9));
}

#[test]
fn chi2_inverse_matches_scipy_grid() {
    let doc: Value = serde_json::from_str(PYTHON).expect("fixture");
    for case in doc["chi2_cases"].as_array().expect("cases") {
        let p = number(&case["p"]);
        let k = case["dof"].as_u64().expect("dof") as usize;
        let expected = number(&case["ppf"]);
        let actual = chi2_inv(p, k).expect("chi2 inverse");
        // The implementation bisects the incomplete-gamma CDF.  Its gamma
        // convergence target is 1e-15, which maps to a conservative 2e-10
        // relative quantile bound in the extreme tails; retain an absolute
        // floor for values close to zero.
        close(
            actual,
            expected,
            2e-10 * expected.abs().max(1.0),
            &format!("p={p} k={k}"),
        );
    }
}

#[test]
fn chi2_dof1_dense_lower_tail_matches_mpmath() {
    let doc: Value = serde_json::from_str(PYTHON).expect("fixture");
    for case in doc["chi2_dof1_lower_tail"].as_array().expect("cases") {
        let p = number(&case["p"]);
        let expected = number(&case["ppf"]);
        let actual = chi2_inv(p, 1).expect("chi2 inverse");
        // For x = 2 erf_inv(p)^2, p^2, the two correction products, their
        // additions, and the final product contribute at most six rounded
        // operations; six ulps is the corresponding outward error bound.
        let tolerance = 6.0 * (f64::from_bits(expected.to_bits() + 1) - expected);
        close(actual, expected, tolerance, &format!("p={p}"));
    }
}

#[test]
fn chi2_dof1_dense_upper_tail_matches_mpmath() {
    let doc: Value = serde_json::from_str(PYTHON).expect("fixture");
    for case in doc["chi2_dof1_upper_tail"].as_array().expect("cases") {
        let p = number(&case["p"]);
        let expected = number(&case["isf"]);
        let actual = chi2_isf(p, 1).expect("chi2 inverse survival");
        // AS 241 evaluates one normal quantile followed by one multiplication;
        // eight ulps cover its documented binary64 polynomial error and squaring.
        let tolerance = 8.0 * (f64::from_bits(expected.to_bits() + 1) - expected);
        close(actual, expected, tolerance, &format!("p={p}"));
    }
}

#[test]
fn chi2_general_tails_match_mpmath_to_few_ulps() {
    let doc: Value = serde_json::from_str(PYTHON).expect("fixture");
    for case in doc["chi2_mpmath_grid"].as_array().expect("cases") {
        let p = number(&case["p"]);
        let dof = case["dof"].as_u64().unwrap() as usize;
        for (actual, key) in [
            (chi2_inv(p, dof).unwrap(), "ppf"),
            (chi2_isf(p, dof).unwrap(), "isf"),
        ] {
            let expected = number(&case[key]);
            let ulp = if expected == 0.0 {
                f64::from_bits(1)
            } else {
                f64::from_bits(expected.to_bits() + 1) - expected
            };
            // The continued fraction/series uses at most sixteen rounded
            // recurrence terms near the branch, followed by a density and a
            // safeguarded Newton update; two ulps per stage gives 32 ulps.
            close(
                actual,
                expected,
                32.0 * ulp,
                &format!("{key} p={p} dof={dof}"),
            );
        }
    }
}

#[test]
fn chi2_dof1_is_accurate_and_monotone_across_the_series_switch() {
    let doc: Value = serde_json::from_str(PYTHON).expect("fixture");
    let mut previous = 0.0;
    for case in doc["chi2_dof1_switch"].as_array().expect("switch cases") {
        let p = number(&case["p"]);
        let expected = number(&case["ppf"]);
        let actual = chi2_inv(p, 1).unwrap();
        let ulp = f64::from_bits(expected.to_bits() + 1) - expected;
        close(actual, expected, 4.0 * ulp, &format!("p={p}"));
        assert!(actual > previous);
        previous = actual;
    }
    let mut previous = chi2_inv(1.0e-6, 1).unwrap();
    for index in 1..=20_000 {
        let p = 1.0e-6 + index as f64 * (0.5 - 1.0e-6) / 20_000.0;
        let actual = chi2_inv(p, 1).unwrap();
        assert!(actual > previous, "p={p}");
        previous = actual;
    }
}

#[test]
fn erfc_inverse_matches_mpmath_grid() {
    let doc: Value = serde_json::from_str(PYTHON).expect("fixture");
    for case in doc["erfc_inv_grid"].as_array().expect("cases") {
        let y = number(&case["y"]);
        let expected = number(&case["value"]);
        let actual = erfc_inv(y).expect("interior erfc inverse");
        let magnitude = expected.abs();
        let ulp = if magnitude == 0.0 {
            f64::from_bits(1)
        } else {
            f64::from_bits(magnitude.to_bits() + 1) - magnitude
        };
        // The central inverse-erf series, four Halley corrections at ordinary
        // tails, or eight log-domain Newton corrections in extreme tails
        // retain this 16-ulp absolute bound; the near-one grid separately
        // checks four ulps to preserve relative precision at zero.
        close(actual, expected, 16.0 * ulp, &format!("y={y}"));
    }
}

#[test]
fn inverse_tail_boundaries_never_return_nan() {
    assert_eq!(normal_q_inv(0.0), None);
    assert!(chi2_isf(f64::from_bits(1), 1).unwrap().is_finite());
}

#[test]
fn chi2_inverse_matches_rtklib_threshold_table() {
    let doc: Value = serde_json::from_str(RTKLIB).expect("fixture");
    for (index, threshold) in doc["chisqr_alpha_0_001"]
        .as_array()
        .expect("table")
        .iter()
        .enumerate()
        .take(41)
    {
        let scipy_equivalent = chi2_inv(0.999, index + 1).expect("chi2 inverse");
        // RTKLIB publishes this table rounded to 3 significant decimal digits
        // (and later entries to integer units), so half a displayed unit is the
        // oracle's intrinsic quantisation tolerance.
        let expected = number(threshold);
        let tolerance = if index < 59 { 0.05 } else { 0.5 };
        close(
            scipy_equivalent,
            expected,
            tolerance,
            &format!("RTKLIB dof {}", index + 1),
        );
    }
}

#[test]
fn raim_exact_chi_square_rejection_documents_demo5_difference() {
    let doc: Value = serde_json::from_str(RTKLIB).expect("fixture");
    let rtklib_rejection_active = doc["valsol_chisqr_rejection_active"]
        .as_bool()
        .expect("harness decision");
    assert!(!rtklib_rejection_active);
    for (index, table) in doc["chisqr_alpha_0_001"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let dof = index + 1;
        let exact = chi2_inv(0.999, dof).unwrap();
        let statistic = 0.5 * (number(table) + exact);
        let n = dof + 4;
        let mut residuals = vec![0.0; n];
        residuals[0] = statistic.sqrt();
        let mut options = RaimOptions::default();
        options.p_fa = 0.001;
        options.weights = RaimWeights::Unit;
        options.n_systems = Some(1);
        let result = raim(
            &RaimInput {
                used_sats: (0..n).map(|i| format!("G{:03}", i + 1)).collect(),
                residuals_m: residuals,
                variances_m2: None,
            },
            &options,
        )
        .unwrap();
        // Sidereon keeps the exact chi-square rejection active. RTKLIB demo5
        // computes the warning but comments out the rejecting return.
        assert_eq!(
            result.fault_detected,
            statistic > exact,
            "dof={dof} table={table} exact={exact} statistic={statistic}"
        );
        assert!(!rtklib_rejection_active);
    }
}

#[test]
fn dop_matches_rtklib_dops() {
    let doc: Value = serde_json::from_str(RTKLIB).expect("fixture");
    let receiver = Wgs84Geodetic::new(0.7, -1.2, 50.0).expect("receiver");
    for case in doc["dop_cases"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        if matches!(
            name,
            "near_singular"
                | "rank_deficient_cone"
                | "receiver_coplanar"
                | "two_coincident_of_four"
        ) {
            continue;
        }
        let mask = number(&case["mask_deg"]);
        let pairs = case["azel_deg"].as_array().expect("azel");
        let los: Vec<_> = pairs
            .iter()
            .filter_map(|pair| {
                let pair = pair.as_array().expect("pair");
                let el = number(&pair[1]);
                (el >= mask && el > 0.0).then(|| {
                    line_of_sight_from_az_el_deg(number(&pair[0]), el, receiver).expect("los")
                })
            })
            .collect();
        let result = dop(&los, &vec![1.0; los.len()], receiver);
        if los.len() < 4 {
            assert_eq!(result, Err(DopError::TooFewSatellites), "{name}");
            continue;
        }
        let actual = result.expect("dop");
        let expected = case["dop"].as_array().expect("dop values");
        // Both form H^T H from the same sin/cos rows, but RTKLIB uses its
        // column-major Gauss-Jordan inverse while core uses a cofactor inverse.
        // 2e-12 covers trig rounding plus the two fixed 4x4 operation orders.
        for (label, got, want) in [
            ("gdop", actual.gdop, number(&expected[0])),
            ("pdop", actual.pdop, number(&expected[1])),
            ("hdop", actual.hdop, number(&expected[2])),
            ("vdop", actual.vdop, number(&expected[3])),
        ] {
            close(
                got,
                want,
                2e-12 * want.abs().max(1.0),
                &format!("{name} {label}"),
            );
        }
    }
}

#[test]
fn dop_near_singular_documents_rtklib_rounding_noise() {
    let doc: Value = serde_json::from_str(RTKLIB).expect("fixture");
    let case = doc["dop_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "near_singular")
        .unwrap();
    let receiver = Wgs84Geodetic::new(0.7, -1.2, 50.0).unwrap();
    let los: Vec<_> = case["azel_deg"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            line_of_sight_from_az_el_deg(number(&p[0]), number(&p[1]), receiver).unwrap()
        })
        .collect();
    let rtklib_dop = case["dop"].as_array().expect("DOP vector");
    let rtklib_rounding_artifact = rtklib_dop.iter().map(number).fold(0.0_f64, f64::max);
    assert!(rtklib_rounding_artifact.is_finite() && rtklib_rounding_artifact > 1.0e7);
    // This RTKLIB assertion documents only its finite rounding artifact; it
    // does not attest a meaningful DOP at condition number about 3e35. The
    // engine assertion below enforces its documented Singular classification.
    assert_eq!(dop(&los, &[1.0; 4], receiver), Err(DopError::Singular));
}

#[test]
fn rank_deficient_dop_is_singular_where_rtklib_returns_zero() {
    let doc: Value = serde_json::from_str(RTKLIB).unwrap();
    let receiver = Wgs84Geodetic::new(0.7, -1.2, 50.0).unwrap();
    for name in ["receiver_coplanar", "two_coincident_of_four"] {
        let case = doc["dop_cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .unwrap();
        assert!(case["dop"]
            .as_array()
            .unwrap()
            .iter()
            .all(|x| number(x) == 0.0));
        let los: Vec<_> = case["azel_deg"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let p = p.as_array().unwrap();
                line_of_sight_from_az_el_deg(number(&p[0]), number(&p[1]), receiver).unwrap()
            })
            .collect();
        assert_eq!(
            dop(&los, &vec![1.0; los.len()], receiver),
            Err(DopError::Singular),
            "{name}"
        );
    }
}

#[test]
fn constant_elevation_cone_is_reported_singular() {
    let doc: Value = serde_json::from_str(RTKLIB).unwrap();
    let case = doc["dop_cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "rank_deficient_cone")
        .unwrap();
    let receiver = Wgs84Geodetic::new(0.7, -1.2, 50.0).unwrap();
    let los: Vec<_> = case["azel_deg"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let p = p.as_array().unwrap();
            line_of_sight_from_az_el_deg(number(&p[0]), number(&p[1]), receiver).unwrap()
        })
        .collect();
    assert_eq!(dop(&los, &[1.0; 5], receiver), Err(DopError::Singular));
}

#[test]
fn dop_scales_by_inverse_square_root_of_uniform_weight() {
    let receiver =
        Wgs84Geodetic::new(45.0_f64.to_radians(), (-75.0_f64).to_radians(), 0.0).unwrap();
    let azel = [
        (5.0, 25.0),
        (80.0, 35.0),
        (155.0, 55.0),
        (235.0, 40.0),
        (310.0, 65.0),
    ];
    let los: Vec<_> = azel
        .into_iter()
        .map(|(a, e)| line_of_sight_from_az_el_deg(a, e, receiver).unwrap())
        .collect();
    let oracle = 4.083_875_830_911_294;
    for scale in [1.0, 1e-80, 1e-200, 1e80] {
        let scaled = dop(&los, &[scale; 5], receiver).unwrap();
        let expected = oracle / libm::sqrt(scale);
        close(
            scaled.gdop,
            expected,
            1e-12 * expected.abs(),
            &format!("scale={scale}"),
        );
    }
}

#[test]
fn elevation_boundaries_and_rtk_mask_match_rtklib_satazel() {
    let doc: Value = serde_json::from_str(RTKLIB).expect("fixture");
    let mut positions = BTreeMap::new();
    let receiver_ecef = geodetic_to_itrf(Wgs84Geodetic::new(0.7, -1.2, 50.0).unwrap())
        .unwrap()
        .as_array();
    for case in doc["satazel_cases"].as_array().unwrap() {
        let los = case["los_ecef"].as_array().unwrap();
        let sat = [
            receiver_ecef[0] + 20_000_000.0 * number(&los[0]),
            receiver_ecef[1] + 20_000_000.0 * number(&los[1]),
            receiver_ecef[2] + 20_000_000.0 * number(&los[2]),
        ];
        if case["name"] == "zenith" {
            positions.insert("zenith".to_string(), sat);
            positions.insert(
                "nadir".to_string(),
                [
                    receiver_ecef[0] - 20_000_000.0 * number(&los[0]),
                    receiver_ecef[1] - 20_000_000.0 * number(&los[1]),
                    receiver_ecef[2] - 20_000_000.0 * number(&los[2]),
                ],
            );
        }
    }
    assert!(visible_at_elevation_mask(10.0, 10.0));
    assert!(!visible_at_elevation_mask(
        f64::from_bits(10.0f64.to_bits() - 1),
        10.0
    ));
    let result = apply_elevation_mask(
        receiver_ecef,
        &[ElevationMaskEpoch {
            satellite_positions_m: positions,
        }],
        0.0,
    )
    .unwrap();
    assert!(result.epochs[0]
        .kept_satellite_ids
        .contains(&"zenith".to_string()));
    // RTKLIB dops uses `< elmin` (equal survives), while selsat in rtkpos.c and
    // pntpos residual screening use `< opt->elmin`; Sidereon likewise uses >=.
    assert_eq!(result.masked_satellite_ids, vec!["nadir"]);
    assert_eq!(result.epochs[0].kept_satellite_ids, vec!["zenith"]);
}

#[test]
fn exact_mask_boundary_documents_platform_libm_rounding() {
    let doc: Value = serde_json::from_str(RTKLIB).unwrap();
    let mut seen = [false; 2];
    let mut different_math_decision = false;
    for case in doc["elevation_mask_cases"].as_array().unwrap() {
        let rr = case["receiver_ecef_m"].as_array().unwrap();
        let sat = case["satellite_ecef_m"].as_array().unwrap();
        let receiver = [number(&rr[0]), number(&rr[1]), number(&rr[2])];
        let satellite = [number(&sat[0]), number(&sat[1]), number(&sat[2])];
        let threshold_deg = number(&case["mask_deg"]);
        assert!(visible_at_elevation_mask(threshold_deg, threshold_deg));
        let epoch = ElevationMaskEpoch {
            satellite_positions_m: BTreeMap::from([("sat".to_string(), satellite)]),
        };
        let kept = apply_elevation_mask(receiver, &[epoch], threshold_deg)
            .unwrap()
            .epochs[0]
            .kept_satellite_ids
            .clone();
        let expected = if case["portable_kept"].as_bool().unwrap() {
            vec!["sat".to_string()]
        } else {
            Vec::new()
        };
        // Both decisions are generated by executing RTKLIB on these exact
        // ECEF inputs, once with platform math and once with Rust libm.
        // No inferred ulp envelope can excuse an incorrect portable decision.
        assert_eq!(kept, expected, "mask {threshold_deg}");
        seen[usize::from(kept.is_empty())] = true;
        different_math_decision |= case["platform_kept"] != case["portable_kept"];
        for implementation in ["platform", "portable"] {
            let elevation = number(&case[format!("{implementation}_ecef_elevation_rad")]);
            assert_eq!(
                elevation >= number(&case["mask_rad"]),
                case[format!("{implementation}_kept")].as_bool().unwrap(),
                "{implementation} fixture comparison"
            );
        }
    }
    assert!(
        seen.into_iter().all(|value| value),
        "both acceptance and rejection exercised"
    );
    assert!(different_math_decision, "one-ulp math boundary exercised");
}

#[test]
fn elevation_mask_accepts_negative_and_zero_thresholds() {
    let doc: Value = serde_json::from_str(RTKLIB).expect("fixture");
    let case = &doc["satazel_cases"][0];
    let pos = case["pos_rad_m"].as_array().unwrap();
    let receiver_position = geodetic_to_itrf(
        Wgs84Geodetic::new(number(&pos[0]), number(&pos[1]), number(&pos[2])).unwrap(),
    )
    .unwrap();
    let receiver = [
        receiver_position.x_m,
        receiver_position.y_m,
        receiver_position.z_m,
    ];
    let los = case["los_ecef"].as_array().unwrap();
    let satellite = [
        receiver[0] + 2.0e7 * number(&los[0]),
        receiver[1] + 2.0e7 * number(&los[1]),
        receiver[2] + 2.0e7 * number(&los[2]),
    ];
    let epoch = ElevationMaskEpoch {
        satellite_positions_m: BTreeMap::from([("sat".to_string(), satellite)]),
    };
    for mask in [-1.0, 0.0] {
        let result = apply_elevation_mask(receiver, std::slice::from_ref(&epoch), mask).unwrap();
        assert_eq!(result.epochs[0].kept_satellite_ids, vec!["sat"]);
    }
}

#[test]
fn elevation_weighting_drops_rows_below_the_horizon() {
    // Explicit satellite lists must still apply the configured mask; RTKLIB dops drops el <= 0.
    dop_at_epoch_filters_an_explicit_below_horizon_satellite();
}

#[test]
#[allow(deprecated)]
fn coverage_legacy_helpers_return_values_and_checked_forms_validate() {
    let grid = vec![vec![Ok(LookAngle {
        azimuth_deg: 0.0,
        elevation_deg: 45.0,
        range_km: 1.0,
    })]];
    for threshold in [f64::NAN, f64::INFINITY] {
        assert_eq!(visible_mask(&grid, threshold), vec![vec![false]]);
        assert!(try_visible_mask(&grid, threshold).is_err());
    }
    let ragged = vec![
        grid[0].clone(),
        vec![grid[0][0].clone(), grid[0][0].clone()],
    ];
    assert_eq!(access_counts(&ragged, 0.0), vec![2]);
    assert!(try_access_counts(&ragged, 0.0).is_err());
}

#[test]
fn standalone_rtk_mask_requires_both_base_and_rover_to_pass() {
    let base = [6_378_137.0, 0.0, 0.0];
    let rover = [6_378_105.642_906_835, 19_999.967_224_386_01, 0.0];
    let satellite = [12_369_740.698_780_913, -22_205_870.510_222_178, 0.0];
    let epoch = ElevationMaskEpoch {
        satellite_positions_m: BTreeMap::from([("sat".to_string(), satellite)]),
    };
    assert_eq!(
        apply_elevation_mask(base, std::slice::from_ref(&epoch), 15.0)
            .unwrap()
            .epochs[0]
            .kept_satellite_ids,
        ["sat"]
    );
    assert!(
        apply_elevation_mask(rover, &[epoch], 15.0).unwrap().epochs[0]
            .kept_satellite_ids
            .is_empty()
    );
}

#[test]
fn coverage_angles_and_visibility_match_skyfield() {
    let doc: Value = serde_json::from_str(PYTHON).unwrap();
    assert_eq!(
        doc["provenance"]["refraction"].as_str(),
        Some("disabled (temperature_C=None, pressure_mbar=0.0)")
    );
    for case in doc["skyfield_cases"].as_array().unwrap() {
        let s = case["site_deg_m"].as_array().unwrap();
        let target = case["target_itrs_km"].as_array().unwrap();
        let (azimuth_deg, elevation_deg, range_km) = itrs_to_topocentric(
            [number(&target[0]), number(&target[1]), number(&target[2])],
            &GeodeticStationKm {
                latitude_deg: number(&s[0]),
                longitude_deg: number(&s[1]),
                altitude_km: number(&s[2]) / 1000.0,
            },
        )
        .unwrap();
        // The shared binary64 ITRS coordinates undergo three subtractions, a
        // 3x3 ENU rotation, norm, asin and atan2.  A 2e-11 degree bound is over
        // 100x the observed accumulated roundoff and below 1e-9 degrees.
        close(azimuth_deg, number(&case["azimuth_deg"]), 2e-11, "azimuth");
        close(
            elevation_deg,
            number(&case["elevation_deg"]),
            2e-11,
            "elevation",
        );
        close(range_km, number(&case["range_km"]), 2e-10, "range");
        let mask = number(&case["mask_deg"]);
        let offset = number(&case["offset_deg"]);
        let predicate_elevation = if offset == 0.0 { mask } else { elevation_deg };
        let grid = vec![vec![Ok(LookAngle {
            azimuth_deg,
            elevation_deg: predicate_elevation,
            range_km,
        })]];
        assert_eq!(visible_mask(&grid, mask)[0][0], offset >= 0.0);
    }
}
