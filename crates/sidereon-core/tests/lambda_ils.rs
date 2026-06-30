#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::ils::{bounded_ils_search, lambda_ils_search, IlsError};

const GOLDEN: &str = include_str!("fixtures/lambda_golden.json");
const RADIUS_CYCLES: i64 = 1;
const CANDIDATE_LIMIT: usize = 200_000;
const RATIO_THRESHOLD: f64 = 3.0;
const SCORE_TOL: f64 = 1.0e-6;

#[derive(Debug, Clone, Copy)]
struct SearchBits {
    best: u64,
    second: u64,
    ratio: u64,
    status: bool,
    candidates: usize,
}

#[derive(Debug, Clone, Copy)]
struct CoreBits {
    name: &'static str,
    lambda: SearchBits,
    // `None` when a ±1 bounded box is infeasible at this dimension (3^n exceeds
    // `CANDIDATE_LIMIT`) - only LAMBDA can solve those, so there are no bounded
    // bits to freeze.
    bounded: Option<SearchBits>,
}

const CORE_BITS: &[CoreBits] = &[
    CoreBits {
        name: "rtklib_utest1",
        lambda: SearchBits {
            best: 4_615_081_697_568_339_738,
            second: 4_615_533_119_754_070_617,
            ratio: 4_607_439_787_206_307_510,
            status: false,
            candidates: 2,
        },
        bounded: Some(SearchBits {
            best: 4_615_081_697_568_339_738,
            second: 4_615_533_119_754_070_617,
            ratio: 4_607_439_787_206_307_510,
            status: false,
            candidates: 729,
        }),
    },
    CoreBits {
        name: "rtklib_utest2",
        lambda: SearchBits {
            best: 4_654_340_190_279_434_240,
            second: 4_654_808_036_696_129_536,
            ratio: 4_607_500_437_478_353_122,
            status: false,
            candidates: 2,
        },
        bounded: Some(SearchBits {
            best: 4_671_550_238_517_840_040,
            second: 4_673_228_389_141_085_448,
            ratio: 4_608_480_767_302_246_517,
            status: false,
            candidates: 59_049,
        }),
    },
    CoreBits {
        name: "synthetic_diag3",
        lambda: SearchBits {
            best: 4_611_926_210_407_514_330,
            second: 4_631_122_803_819_181_068,
            ratio: 4_626_319_154_241_953_272,
            status: true,
            candidates: 2,
        },
        bounded: Some(SearchBits {
            best: 4_611_926_210_407_514_330,
            second: 4_631_122_803_819_181_068,
            ratio: 4_626_319_154_241_953_272,
            status: true,
            candidates: 27,
        }),
    },
    CoreBits {
        name: "synthetic_corr3",
        lambda: SearchBits {
            best: 4_605_004_207_215_537_561,
            second: 4_605_357_430_715_723_482,
            ratio: 4_607_415_363_608_329_675,
            status: false,
            candidates: 2,
        },
        bounded: Some(SearchBits {
            best: 4_605_004_207_215_537_561,
            second: 4_605_357_430_715_723_482,
            ratio: 4_607_415_363_608_329_675,
            status: false,
            candidates: 27,
        }),
    },
    CoreBits {
        name: "synthetic_corr4",
        lambda: SearchBits {
            best: 4_607_217_402_530_968_994,
            second: 4_609_316_772_818_804_779,
            ratio: 4_609_265_606_988_832_640,
            status: false,
            candidates: 2,
        },
        bounded: Some(SearchBits {
            best: 4_607_217_402_530_968_994,
            second: 4_609_316_772_818_804_779,
            ratio: 4_609_265_606_988_832_640,
            status: false,
            candidates: 81,
        }),
    },
    // n=16 multi-constellation PPP/RTK arc (decaying-correlation + common-mode
    // SPD covariance). 3^16 exceeds the ±1 box limit, so LAMBDA only.
    CoreBits {
        name: "ppp_arc16",
        lambda: SearchBits {
            best: 4_621_277_653_433_667_077,
            second: 4_621_288_762_916_491_697,
            ratio: 4_607_192_252_195_224_650,
            status: false,
            candidates: 2,
        },
        bounded: None,
    },
    // Near-tie / low-ratio case (RTKLIB ratio ~2.0): exercises the ratio test.
    // The ±1 box reaches the same optimum and runner-up here.
    CoreBits {
        name: "near_tie5",
        lambda: SearchBits {
            best: 4_607_660_474_530_029_357,
            second: 4_612_164_099_115_450_853,
            ratio: 4_611_686_040_990_383_760,
            status: false,
            candidates: 2,
        },
        bounded: Some(SearchBits {
            best: 4_607_660_474_530_029_357,
            second: 4_612_164_099_115_450_853,
            ratio: 4_611_686_040_990_383_760,
            status: false,
            candidates: 243,
        }),
    },
    // Well-conditioned near-diagonal sanity anchor (RTKLIB ratio ~249).
    CoreBits {
        name: "easy_diag4",
        lambda: SearchBits {
            best: 4_593_816_240_120_848_181,
            second: 4_629_723_294_799_908_436,
            ratio: 4_642_975_306_685_316_894,
            status: true,
            candidates: 2,
        },
        bounded: Some(SearchBits {
            best: 4_593_816_240_120_848_181,
            second: 4_629_723_294_799_908_436,
            ratio: 4_642_975_306_685_316_894,
            status: true,
            candidates: 81,
        }),
    },
];

fn golden() -> Value {
    serde_json::from_str(GOLDEN).expect("parse lambda golden")
}

fn cases(doc: &Value) -> &[Value] {
    doc["cases"].as_array().expect("golden cases array")
}

fn case_name(case: &Value) -> &str {
    case["name"].as_str().expect("case name")
}

fn floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("float array")
        .iter()
        .map(|v| v.as_f64().expect("float value"))
        .collect()
}

fn matrix(value: &Value) -> Vec<Vec<f64>> {
    value
        .as_array()
        .expect("matrix rows")
        .iter()
        .map(floats)
        .collect()
}

fn fixed_vectors(value: &Value) -> Vec<Vec<i64>> {
    value
        .as_array()
        .expect("fixed vector array")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("fixed row")
                .iter()
                .map(|v| v.as_i64().expect("fixed integer"))
                .collect()
        })
        .collect()
}

fn residuals(case: &Value) -> [f64; 2] {
    let values = floats(&case["lambda_residuals"]);
    [values[0], values[1]]
}

fn core_bits_for(name: &str) -> CoreBits {
    CORE_BITS
        .iter()
        .copied()
        .find(|bits| bits.name == name)
        .unwrap_or_else(|| panic!("missing frozen core bits for {name}"))
}

fn tol_for(score: f64) -> f64 {
    SCORE_TOL.max(score.abs() * 1.0e-9 + 1.0e-4)
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{label}: actual={actual:?} expected={expected:?} tolerance={tolerance:?}"
    );
}

fn assert_search_bits(result: &sidereon_core::ils::IlsResult, expected: SearchBits) {
    assert_eq!(result.best_score.to_bits(), expected.best, "best score");
    assert_eq!(
        result.second_best_score.expect("runner-up score").to_bits(),
        expected.second,
        "runner-up score"
    );
    assert_eq!(result.ratio.to_bits(), expected.ratio, "ratio");
    assert_eq!(result.fixed_status, expected.status, "ratio status");
    assert_eq!(
        result.candidates_evaluated, expected.candidates,
        "candidate count"
    );
}

#[test]
fn lambda_search_matches_rtklib_golden_cases() {
    let doc = golden();

    for case in cases(&doc) {
        let name = case_name(case);
        let a = floats(&case["a"]);
        let q = matrix(&case["Q"]);
        let expected_fixed = fixed_vectors(&case["lambda_fixed"]);
        let expected_residuals = residuals(case);
        let expected_ratio = case["lambda_ratio"].as_f64().expect("lambda ratio");

        let result = lambda_ils_search(&a, &q, RATIO_THRESHOLD)
            .unwrap_or_else(|err| panic!("{name}: LAMBDA search failed: {err:?}"));

        assert_eq!(
            result.fixed, expected_fixed[0],
            "{name}: LAMBDA selected a different integer vector than RTKLIB lambda()"
        );
        assert_close(
            result.best_score,
            expected_residuals[0],
            tol_for(expected_residuals[0]),
            &format!("{name} best score"),
        );
        assert_close(
            result.second_best_score.expect("runner-up score"),
            expected_residuals[1],
            tol_for(expected_residuals[1]),
            &format!("{name} runner-up score"),
        );
        assert_close(
            result.ratio,
            expected_ratio,
            SCORE_TOL,
            &format!("{name} ratio"),
        );
        assert_eq!(
            result.fixed_status,
            expected_ratio >= RATIO_THRESHOLD,
            "{name}: ratio-test status"
        );
    }
}

#[test]
fn bounded_search_matches_rtklib_only_in_regime() {
    let doc = golden();

    for case in cases(&doc).iter().filter(|case| case["in_regime"] == true) {
        let name = case_name(case);
        let a = floats(&case["a"]);
        let q = matrix(&case["Q"]);
        let expected_fixed = fixed_vectors(&case["lambda_fixed"]);
        let expected_residuals = residuals(case);
        let expected_ratio = case["lambda_ratio"].as_f64().expect("lambda ratio");

        let result = bounded_ils_search(&a, &q, RADIUS_CYCLES, CANDIDATE_LIMIT, RATIO_THRESHOLD)
            .unwrap_or_else(|err| panic!("{name}: bounded search failed: {err:?}"));

        assert_eq!(result.fixed, expected_fixed[0], "{name}: fixed vector");
        assert_close(
            result.best_score,
            expected_residuals[0],
            SCORE_TOL,
            &format!("{name} bounded best score"),
        );
        assert_close(
            result.second_best_score.expect("runner-up score"),
            expected_residuals[1],
            SCORE_TOL,
            &format!("{name} bounded runner-up score"),
        );
        assert_close(
            result.ratio,
            expected_ratio,
            SCORE_TOL,
            &format!("{name} bounded ratio"),
        );
    }
}

#[test]
fn bounded_search_cannot_reach_the_strongly_correlated_rtklib_optimum() {
    let doc = golden();
    let case = cases(&doc)
        .iter()
        .find(|case| case_name(case) == "rtklib_utest2")
        .expect("rtklib_utest2 case");
    let a = floats(&case["a"]);
    let q = matrix(&case["Q"]);
    let expected_fixed = fixed_vectors(&case["lambda_fixed"]);
    let expected_residuals = residuals(case);

    let result = bounded_ils_search(&a, &q, RADIUS_CYCLES, CANDIDATE_LIMIT, RATIO_THRESHOLD)
        .expect("bounded search returns an in-box candidate");
    assert_ne!(
        result.fixed, expected_fixed[0],
        "the ±1 box must not claim the RTKLIB optimum"
    );
    assert!(result.best_score > expected_residuals[0]);

    let wide = bounded_ils_search(&a, &q, 14, CANDIDATE_LIMIT, RATIO_THRESHOLD);
    assert!(matches!(
        wide,
        Err(IlsError::TooManyCandidates {
            evaluated: _,
            limit: CANDIDATE_LIMIT
        })
    ));
}

#[test]
fn core_solver_outputs_are_frozen_to_exact_bits() {
    let doc = golden();

    for case in cases(&doc) {
        let name = case_name(case);
        let a = floats(&case["a"]);
        let q = matrix(&case["Q"]);
        let expected = core_bits_for(name);

        let lambda = lambda_ils_search(&a, &q, RATIO_THRESHOLD).unwrap();
        assert_search_bits(&lambda, expected.lambda);

        if let Some(expected_bounded) = expected.bounded {
            let bounded =
                bounded_ils_search(&a, &q, RADIUS_CYCLES, CANDIDATE_LIMIT, RATIO_THRESHOLD)
                    .unwrap();
            assert_search_bits(&bounded, expected_bounded);
        }
    }
}
