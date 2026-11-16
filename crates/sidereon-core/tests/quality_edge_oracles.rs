#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::ephemeris::BroadcastEphemeris;
use sidereon_core::positioning::{
    solve_spp_from_rinex_obs, Corrections, Observation, RinexSppOptions, SolvePolicy,
};
use sidereon_core::quality::{
    chi2_inv, fde, raim, residual_diagnostics, validate_receiver_solution, FdeError, FdeOptions,
    FdeSolveFailure, QualityError, RaimInput, RaimOptions, RaimSolution, RaimWeights,
    SolutionValidationError, SolutionValidationOptions,
};
use sidereon_core::rinex::observations::{ObservationFile, SignalPolicy};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::path::PathBuf;

const SCIPY: &str = include_str!("fixtures/integrity_quality_scipy_skyfield.json");

fn number(value: &Value) -> f64 {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|text| text.parse().ok()))
        .expect("numeric fixture value")
}

fn fixture(parts: &[&str]) -> PathBuf {
    parts.iter().fold(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        |mut path, part| {
            path.push(part);
            path
        },
    )
}

fn real_solution() -> sidereon_core::positioning::ReceiverSolution {
    let obs = ObservationFile::parse(
        &std::fs::read_to_string(fixture(&[
            "obs",
            "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx",
        ]))
        .unwrap(),
    )
    .unwrap();
    let nav = BroadcastEphemeris::from_nav(
        &std::fs::read_to_string(fixture(&["nav", "ESBC00DNK_R_20201770000_01D_MN.rnx"])).unwrap(),
    )
    .unwrap();
    let options = RinexSppOptions::new(SignalPolicy {
        codes: [(GnssSystem::Gps, vec!["C1C".to_string()])].into(),
    })
    .with_corrections(Corrections {
        ionosphere: true,
        troposphere: true,
    });
    solve_spp_from_rinex_obs(&nav, &obs, &options, false, SolvePolicy::default())
        .unwrap()
        .into_iter()
        .find_map(|epoch| epoch.solution.ok())
        .expect("real SPP solution")
}

#[test]
fn validation_rejects_nan_position() {
    let mut solution = real_solution();
    solution.position.x_m = f64::NAN;
    assert!(
        matches!(validate_receiver_solution(&solution,SolutionValidationOptions::default()),Err(SolutionValidationError::ImplausiblePosition(value)) if value.is_nan())
    );
}

#[test]
fn validation_rejects_nan_pdop() {
    let mut solution = real_solution();
    solution.dop.as_mut().unwrap().pdop = f64::NAN;
    let mut options = SolutionValidationOptions::default();
    options.max_pdop = Some(6.0);
    assert!(
        matches!(validate_receiver_solution(&solution,options),Err(SolutionValidationError::DegenerateGeometryPdop(value)) if value.is_nan())
    );
}

#[test]
fn overflowing_finite_residual_is_non_convergence() {
    let mut solution = real_solution();
    solution.residuals_m[0] = 1e200;
    assert!(
        matches!(validate_receiver_solution(&solution, SolutionValidationOptions::default()), Err(SolutionValidationError::NoConvergence(value)) if value.is_infinite())
    );
}

#[test]
fn all_zero_raim_residuals_have_no_worst_satellite() {
    let input = RaimInput {
        used_sats: (1..=6).map(|i| format!("G{i:02}")).collect(),
        residuals_m: vec![0.0; 6],
        variances_m2: Some(vec![1.0; 6]),
    };
    assert_eq!(
        raim(&input, &RaimOptions::default()).unwrap().worst_sat,
        None
    );
}

#[test]
fn raim_rejects_repeated_satellite_tokens_like_rtklib_rescode() {
    let input = RaimInput {
        used_sats: ["G01", "G01", "G02", "G03", "G04", "G05"]
            .map(str::to_string)
            .into(),
        residuals_m: vec![0.0; 6],
        variances_m2: Some(vec![1.0; 6]),
    };
    assert_eq!(
        raim(&input, &RaimOptions::default()),
        Err(QualityError::InvalidResiduals)
    );
}

#[test]
fn raim_refuses_overflowing_clock_count() {
    let input = RaimInput {
        used_sats: (1..=6).map(|i| format!("G{i:02}")).collect(),
        residuals_m: vec![0.0; 6],
        variances_m2: None,
    };
    let mut options = RaimOptions::default();
    options.weights = RaimWeights::Unit;
    options.n_systems = Some(isize::MAX);
    let result = std::panic::catch_unwind(|| raim(&input, &options));
    assert_eq!(
        result.expect("must not panic"),
        Err(QualityError::InvalidSystemCount)
    );
}

#[test]
fn residual_diagnostics_refuses_overflowing_parameter_count() {
    let result = residual_diagnostics(&[], None, usize::MAX, Some(1e-3)).unwrap();
    assert!(result.degrees_of_freedom < 0);
    assert_eq!(result.chi_square_threshold, None);
}

#[test]
fn raim_supports_integrity_probability_below_one_minus_rounding() {
    let fixture: Value = serde_json::from_str(SCIPY).expect("SciPy fixture");
    let oracle = &fixture["quality_edge_chi2"]["raim_tiny_tail"];
    let input = RaimInput {
        used_sats: (1..=6).map(|i| format!("G{i:02}")).collect(),
        residuals_m: vec![0.0; 6],
        variances_m2: Some(vec![1.0; 6]),
    };
    let mut options = RaimOptions::default();
    options.p_fa = number(&oracle["p_fa"]);
    let result = raim(&input, &options).expect("SciPy isf is finite");
    assert_eq!(result.dof, oracle["dof"].as_i64().unwrap() as isize);
    assert!((result.threshold.unwrap() - number(&oracle["isf"])).abs() < 1e-12);
}

#[test]
fn chi2_lower_tail_matches_scipy() {
    let fixture: Value = serde_json::from_str(SCIPY).expect("SciPy fixture");
    let cases = fixture["quality_edge_chi2"]["lower_tail"]
        .as_array()
        .unwrap();
    for case in cases {
        let expected = number(&case["ppf"]);
        let actual = chi2_inv(number(&case["p"]), case["dof"].as_u64().unwrap() as usize).unwrap();
        assert!((actual - expected).abs() < expected.abs() * 1e-12 + 1e-312);
    }
}

#[test]
fn chi2_million_dof_matches_scipy() {
    let fixture: Value = serde_json::from_str(SCIPY).expect("SciPy fixture");
    for case in fixture["quality_edge_chi2"]["million_dof"]
        .as_array()
        .unwrap()
    {
        let expected = number(&case["ppf"]);
        let actual = chi2_inv(number(&case["p"]), case["dof"].as_u64().unwrap() as usize).unwrap();
        assert!((actual - expected).abs() / expected < 1e-12);
    }
}

#[derive(Clone, Debug)]
struct ThreeClockSolution {
    sats: Vec<String>,
    residuals: Vec<f64>,
    variances: Vec<f64>,
}
impl RaimSolution for ThreeClockSolution {
    fn raim_used_sats(&self) -> Vec<String> {
        self.sats.clone()
    }
    fn raim_residuals_m(&self) -> &[f64] {
        &self.residuals
    }
    fn raim_variances_m2(&self) -> Option<&[f64]> {
        Some(&self.variances)
    }
    fn raim_clock_count(&self) -> Option<usize> {
        Some(3)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Never;
impl FdeSolveFailure for Never {
    fn admits_exclusion_search(&self) -> bool {
        false
    }
}

#[test]
fn multi_clock_fde_requires_redundancy_after_exclusion() {
    let ids = [
        ("G", 1),
        ("G", 2),
        ("G", 3),
        ("E", 1),
        ("E", 2),
        ("C", 1),
        ("C", 2),
    ];
    let observations: Vec<_> = ids
        .iter()
        .map(|(s, p)| Observation {
            satellite_id: GnssSatelliteId::new(
                GnssSystem::from_letter(s.chars().next().unwrap()).unwrap(),
                *p,
            )
            .unwrap(),
            pseudorange_m: 0.0,
        })
        .collect();
    let solve = |remaining: &[Observation]| -> Result<ThreeClockSolution, Never> {
        let sats: Vec<_> = remaining
            .iter()
            .map(|o| o.satellite_id.to_string())
            .collect();
        let residuals = if remaining.len() == 7 {
            vec![100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        } else {
            vec![1e-9; remaining.len()]
        };
        Ok(ThreeClockSolution {
            sats,
            variances: vec![1.0; remaining.len()],
            residuals,
        })
    };
    assert!(matches!(
        fde(&observations, &FdeOptions::default(), solve),
        Err(FdeError::FaultUnresolved(_))
    ));
}
