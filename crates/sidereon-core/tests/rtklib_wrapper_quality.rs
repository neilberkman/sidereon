#![cfg(sidereon_repo_tests)]

use serde_json::Value;
use sidereon_core::ephemeris::BroadcastEphemeris;
use sidereon_core::positioning::{
    spp_inputs_from_rinex_obs, Corrections, Observation, RinexSppOptions, SolveInputs,
};
use sidereon_core::quality::{
    fde, fde_spp, raim, FdeError, FdeOptions, FdeSolveFailure, FdeSppOptions, RaimInput,
    RaimOptions, RaimSolution,
};
use sidereon_core::rinex::observations::{ObservationFile, SignalPolicy};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::collections::BTreeMap;

const ORACLE: &str = include_str!("fixtures/rtk/rtklib_quality_fde_100m.json");
const MULTI_ORACLE: &str = include_str!("fixtures/rtk/rtklib_multiconstellation_fde_100m.json");
const EDGE_ORACLE: &str = include_str!("fixtures/rtk/rtklib_spp_edge_cases.json");

#[derive(Clone, Debug)]
struct Candidate {
    satellites: Vec<String>,
    residuals_m: Vec<f64>,
    variances_m2: Vec<f64>,
    position_m: [f64; 3],
}

impl RaimSolution for Candidate {
    fn raim_used_sats(&self) -> Vec<String> {
        self.satellites.clone()
    }

    fn raim_residuals_m(&self) -> &[f64] {
        &self.residuals_m
    }

    fn raim_variances_m2(&self) -> Option<&[f64]> {
        Some(&self.variances_m2)
    }
}

#[derive(Clone, Copy, Debug)]
struct Failed;

impl FdeSolveFailure for Failed {
    fn admits_exclusion_search(&self) -> bool {
        true
    }
}

fn position(value: &Value) -> [f64; 3] {
    std::array::from_fn(|index| value[index].as_f64().unwrap())
}

fn edge_context() -> (BroadcastEphemeris, SolveInputs, Value) {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let obs = ObservationFile::parse(
        &std::fs::read_to_string(
            manifest.join("tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx"),
        )
        .unwrap(),
    )
    .unwrap();
    let store = BroadcastEphemeris::from_nav(
        &std::fs::read_to_string(
            manifest.join("tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"),
        )
        .unwrap(),
    )
    .unwrap();
    let oracle: Value = serde_json::from_str(EDGE_ORACLE).unwrap();
    let guess = position(&oracle["guess"]);
    let policy = SignalPolicy {
        codes: [
            (GnssSystem::Gps, vec!["C1C".to_owned()]),
            (GnssSystem::Glonass, vec!["C1C".to_owned()]),
            (GnssSystem::Galileo, vec!["C1C".to_owned()]),
            (GnssSystem::BeiDou, vec!["C2I".to_owned()]),
        ]
        .into(),
    };
    let options = RinexSppOptions::new(policy)
        .with_corrections(Corrections::IONO_TROPO)
        .with_initial_guess([guess[0], guess[1], guess[2], 0.0]);
    let inputs = spp_inputs_from_rinex_obs(&obs, &store, &options).unwrap()[0]
        .inputs
        .clone();
    (store, inputs, oracle)
}

fn edge_inputs(all: &SolveInputs, case: &Value) -> SolveInputs {
    let mode = case["mode"].as_str().unwrap();
    let selected: std::collections::BTreeSet<GnssSatelliteId> = case["candidates"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|candidate| {
            let sat: GnssSatelliteId = candidate["sat"].as_str().unwrap().parse().unwrap();
            (mode != "clean_tie" || sat.system == GnssSystem::Gps).then_some(sat)
        })
        .collect();
    let mut inputs = all.clone();
    inputs
        .observations
        .retain(|observation| selected.contains(&observation.satellite_id));
    let invalid = match mode {
        "nan_pseudorange" => Some(f64::NAN),
        "infinite_pseudorange" => Some(f64::INFINITY),
        "zero_pseudorange" | "two_zero_pseudoranges" => Some(0.0),
        "negative_pseudorange" => Some(-10.0),
        _ => None,
    };
    if let Some(value) = invalid {
        for invalid_observation in case["invalid_observations"].as_array().unwrap() {
            inputs
                .observations
                .iter_mut()
                .find(|observation| {
                    observation.satellite_id.to_string()
                        == invalid_observation["sat"].as_str().unwrap()
                })
                .unwrap()
                .pseudorange_m = value;
        }
    }
    inputs
}

#[test]
fn fde_candidate_order_rms_cap_exclusion_and_position_match_rtklib() {
    let document: Value = serde_json::from_str(ORACLE).unwrap();
    assert_eq!(
        document["provenance"]["commit"],
        "75a2e56275485b21a67bd35bc94bbeb8936e1a74"
    );
    for case in document["cases"].as_array().unwrap() {
        let candidates = case["candidates"].as_array().unwrap();
        let observations: Vec<_> = candidates
            .iter()
            .map(|candidate| Observation {
                satellite_id: candidate["sat"]
                    .as_str()
                    .unwrap()
                    .parse::<GnssSatelliteId>()
                    .unwrap(),
                pseudorange_m: 20_000_000.0,
            })
            .collect();
        let all_tokens: Vec<_> = observations
            .iter()
            .map(|observation| observation.satellite_id.to_string())
            .collect();
        let solve = |remaining: &[Observation]| -> Result<Candidate, Failed> {
            if remaining.len() == observations.len() {
                return Err(Failed);
            }
            let omitted = observations
                .iter()
                .find(|observation| {
                    !remaining
                        .iter()
                        .any(|kept| kept.satellite_id == observation.satellite_id)
                })
                .unwrap()
                .satellite_id
                .to_string();
            let record = candidates
                .iter()
                .find(|candidate| candidate["sat"] == omitted)
                .unwrap();
            if record["status"] != "solved" {
                return Err(Failed);
            }
            let used = record["nvsat"].as_u64().unwrap() as usize;
            let rms = record["rms_m"].as_f64().unwrap();
            let satellites: Vec<_> = all_tokens
                .iter()
                .filter(|token| **token != omitted)
                .take(used)
                .cloned()
                .collect();
            Ok(Candidate {
                residuals_m: vec![rms; used],
                variances_m2: vec![1.0; used],
                satellites,
                position_m: position(&record["position_m"]),
            })
        };
        let result = fde(&observations, &FdeOptions::default(), solve);
        let unresolved = match result {
            Err(FdeError::FaultUnresolved(unresolved)) => unresolved,
            Ok(result) => {
                assert_eq!(result.excluded, vec![case["excluded"].as_str().unwrap()]);
                assert_eq!(result.solution.position_m, position(&case["position_m"]));
                continue;
            }
            other => panic!("unexpected FDE result: {other:?}"),
        };
        assert_eq!(
            unresolved.excluded,
            vec![case["excluded"].as_str().unwrap()]
        );
        assert_eq!(
            unresolved.solution.position_m,
            position(&case["position_m"])
        );
    }
}

#[test]
fn rtklib_wrapper_contains_an_exact_candidate_rms_tie() {
    let document: Value = serde_json::from_str(MULTI_ORACLE).unwrap();
    let tie = document["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find_map(|case| {
            let mut by_bits: BTreeMap<u64, Vec<&str>> = BTreeMap::new();
            for candidate in case["candidates"].as_array().unwrap() {
                if candidate["status"] == "solved" {
                    by_bits
                        .entry(candidate["rms_m"].as_f64().unwrap().to_bits())
                        .or_default()
                        .push(candidate["sat"].as_str().unwrap());
                }
            }
            by_bits
                .into_values()
                .find(|satellites| satellites.len() >= 2)
        });
    assert!(
        tie.is_some(),
        "RTKLIB corpus must exercise an exact RMS tie"
    );
}

#[test]
fn raim_postfit_statistic_accounts_for_rtklib_pre_update_reverse_dot() {
    let document: Value = serde_json::from_str(ORACLE).unwrap();
    for case in document["cases"].as_array().unwrap() {
        let states = case["satellite_states"].as_array().unwrap();
        let residuals: Vec<_> = states
            .iter()
            .map(|state| state["residual_m"].as_f64().unwrap())
            .collect();
        let variances: Vec<_> = states
            .iter()
            .map(|state| state["reference_variance_m2"].as_f64().unwrap())
            .collect();
        let used_sats = states
            .iter()
            .map(|state| state["sat"].as_str().unwrap().to_owned())
            .collect();
        let result = raim(
            &RaimInput {
                used_sats,
                residuals_m: residuals.clone(),
                variances_m2: Some(variances.clone()),
            },
            &RaimOptions::default(),
        )
        .unwrap();
        let postfit_reverse = residuals
            .iter()
            .zip(&variances)
            .rev()
            .fold(0.0, |sum, (&residual, &variance)| {
                sum + residual * residual / variance
            });
        assert!(
            (result.test_statistic - postfit_reverse).abs()
                <= 8.0 * f64::EPSILON * postfit_reverse.max(1.0)
        );

        // RTKLIB validates the residual vector just before its final update. Bound the
        // resulting statistic change by ||W^.5 H dx|| and Cauchy--Schwarz; this is an
        // arithmetic bound, not a tolerance fitted to the observed difference.
        let step = case["lsq_step"].as_array().unwrap();
        let delta_sq = states
            .iter()
            .zip(&variances)
            .fold(0.0, |sum, (state, variance)| {
                let row = state["design_row"].as_array().unwrap();
                let delta = (0..4)
                    .map(|index| row[index].as_f64().unwrap() * step[index].as_f64().unwrap())
                    .sum::<f64>();
                sum + delta * delta / variance
            });
        let bound = 2.0 * libm::sqrt(postfit_reverse * delta_sq)
            + delta_sq
            + 32.0 * f64::EPSILON * postfit_reverse.max(1.0);
        let rtklib_pre_update = case["pre_update_chi_square"].as_f64().unwrap();
        assert!((rtklib_pre_update - result.test_statistic).abs() <= bound);
    }
}

#[test]
fn galileo_variance_uses_nequick_while_rtklib_uses_gps_klobuchar() {
    use sidereon_core::atmosphere::{
        galileo_nequick_g_native, klobuchar_native, GalileoNequickEval, KlobucharParams,
    };
    use sidereon_core::constants::F_L1_HZ;
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let obs = ObservationFile::parse(
        &std::fs::read_to_string(
            manifest.join("tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx"),
        )
        .unwrap(),
    )
    .unwrap();
    let store = BroadcastEphemeris::from_nav(
        &std::fs::read_to_string(
            manifest.join("tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"),
        )
        .unwrap(),
    )
    .unwrap();
    let oracle: Value = serde_json::from_str(MULTI_ORACLE).unwrap();
    let guess = position(&oracle["guess"]);
    let policy = SignalPolicy {
        codes: [
            (GnssSystem::Gps, vec!["C1C".to_owned()]),
            (GnssSystem::Galileo, vec!["C1C".to_owned()]),
            (GnssSystem::BeiDou, vec!["C2I".to_owned()]),
        ]
        .into(),
    };
    let options = RinexSppOptions::new(policy)
        .with_corrections(Corrections::IONO_TROPO)
        .with_initial_guess([guess[0], guess[1], guess[2], 0.0]);
    let mut inputs = spp_inputs_from_rinex_obs(&obs, &store, &options).unwrap()[0]
        .inputs
        .clone();
    let case = &oracle["cases"][0];
    for fault in case["faults"].as_array().unwrap() {
        let sat: GnssSatelliteId = fault["sat"].as_str().unwrap().parse().unwrap();
        let observation = inputs
            .observations
            .iter_mut()
            .find(|observation| observation.satellite_id == sat)
            .unwrap();
        observation.pseudorange_m += fault["bias_m"].as_f64().unwrap();
    }
    let result = fde_spp(&store, &inputs, true, &FdeSppOptions::default()).unwrap();
    assert_eq!(result.excluded, vec![case["excluded"].as_str().unwrap()]);
    let expected: BTreeMap<_, _> = case["satellite_states"]
        .as_array()
        .unwrap()
        .iter()
        .map(|state| (state["sat"].as_str().unwrap(), state))
        .collect();
    let receiver = case["geodetic_rad_m"].as_array().unwrap();
    let lat_deg = receiver[0].as_f64().unwrap().to_degrees();
    let lon_deg = receiver[1].as_f64().unwrap().to_degrees();
    let nequick = inputs.galileo_nequick.expect("Galileo coefficients");
    let klobuchar = KlobucharParams {
        alpha: inputs.klobuchar.alpha,
        beta: inputs.klobuchar.beta,
    };
    for (sat, &variance) in result
        .solution
        .used_sats
        .iter()
        .zip(&result.solution.pseudorange_variances_m2)
    {
        let state = expected[sat.to_string().as_str()];
        let rtklib_variance = state["reference_variance_m2"].as_f64().unwrap();
        if sat.system == GnssSystem::Galileo {
            let az_deg = state["az_rad"].as_f64().unwrap().to_degrees();
            let el_deg = state["el_rad"].as_f64().unwrap().to_degrees();
            let nequick_delay = galileo_nequick_g_native(
                &nequick,
                GalileoNequickEval {
                    lat_deg,
                    lon_deg,
                    el_deg,
                    t_gal_s: inputs.t_rx_second_of_day_s,
                    day_of_year: inputs.day_of_year,
                    frequency_hz: F_L1_HZ,
                },
            )
            .unwrap();
            let klobuchar_delay = klobuchar_native(
                &klobuchar,
                lat_deg,
                lon_deg,
                az_deg,
                el_deg,
                inputs.t_rx_second_of_day_s,
                F_L1_HZ,
            )
            .unwrap();
            let ionosphere_variance_delta =
                0.25 * (nequick_delay * nequick_delay - klobuchar_delay * klobuchar_delay);
            let independent_nequick_variance = rtklib_variance + ionosphere_variance_delta;
            // The independent atmosphere calculation uses RTKLIB's returned geometry,
            // while the Sidereon solve differs by less than 0.1 m. Propagating that
            // displacement through the smooth elevation mapping bounds the variance
            // change by 2e-5 m²; binary64 evaluation is negligible beside that bound.
            assert!(
                (variance - independent_nequick_variance).abs() <= 2.0e-5,
                "{sat}: Sidereon {variance}, independent NeQuick-G {independent_nequick_variance}"
            );
            assert!(
                (variance - rtklib_variance).abs() > 0.1,
                "{sat}: the NeQuick-G and RTKLIB Klobuchar terms must differ"
            );
        } else {
            assert!(
                (variance - rtklib_variance).abs() <= 2.0e-7 * rtklib_variance.max(1.0),
                "{sat}: Sidereon {variance}, RTKLIB {rtklib_variance}"
            );
        }
    }
    let systems: std::collections::BTreeSet<_> = result
        .solution
        .used_sats
        .iter()
        .map(|sat| sat.system)
        .collect();
    assert_eq!(
        systems,
        [GnssSystem::Gps, GnssSystem::Galileo, GnssSystem::BeiDou].into()
    );
}

#[test]
fn public_spp_fde_and_validation_cover_rtklib_edge_cases() {
    use sidereon_core::quality::validate_receiver_solution;
    let (store, all, oracle) = edge_context();

    for case in oracle["cases"].as_array().unwrap() {
        let mode = case["mode"].as_str().unwrap();
        if matches!(
            mode,
            "gdop_failure"
                | "nan_pseudorange"
                | "infinite_pseudorange"
                | "zero_pseudorange"
                | "negative_pseudorange"
                | "two_zero_pseudoranges"
        ) {
            continue;
        }
        let inputs = edge_inputs(&all, case);
        let expected_count = match mode {
            "gps_minimum" => 4,
            "gps_minimum_plus_one" => 5,
            "multi_minimum" => 6,
            "multi_minimum_plus_one" => 7,
            "clean_tie" => 8,
            _ => unreachable!(),
        };
        assert_eq!(inputs.observations.len(), expected_count, "{mode}");
        let result = fde_spp(&store, &inputs, false, &FdeSppOptions::default());
        let result = result.unwrap_or_else(|error| panic!("{mode}: {error:?}"));
        validate_receiver_solution(&result.solution, Default::default()).unwrap();
        assert_eq!(case["full_estpos_stat"], 1, "{mode}: RTKLIB valsol");
        assert!(result.excluded.is_empty(), "{mode}: public FDE exclusion");
    }
}

#[test]
fn public_spp_validation_matches_rtklib_gdop_failure() {
    use sidereon_core::quality::validate_receiver_solution;
    let (store, all, oracle) = edge_context();
    let case = oracle["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["mode"] == "gdop_failure")
        .unwrap();
    assert_eq!(case["full_estpos_stat"], 0);
    let mut result = fde_spp(
        &store,
        &edge_inputs(&all, case),
        false,
        &FdeSppOptions::default(),
    )
    .unwrap();
    let mut validation = sidereon_core::quality::SolutionValidationOptions::default();
    assert!(validate_receiver_solution(&result.solution, validation).is_ok());
    validation.max_gdop = Some(30.0);
    assert!(matches!(
        validate_receiver_solution(&result.solution, validation),
        Err(sidereon_core::quality::SolutionValidationError::DegenerateGeometryGdop(value))
            if value > 30.0
    ));
    result.solution.rtklib_single_clock_dop = None;
    assert!(matches!(
        validate_receiver_solution(&result.solution, validation),
        Err(sidereon_core::quality::SolutionValidationError::DegenerateGeometryGdop(value))
            if value == 0.0
    ));
}

#[test]
fn public_spp_pre_rejection_documents_rtklib_nan_fde_difference() {
    let (store, all, oracle) = edge_context();
    let case = oracle["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|case| case["mode"] == "nan_pseudorange")
        .unwrap();
    assert_eq!(case["pntpos_stat"], 1);
    let result = fde_spp(
        &store,
        &edge_inputs(&all, case),
        false,
        &FdeSppOptions::default(),
    )
    .expect("RTKLIB invalid-pseudorange exclusion solves");
    let rtklib_excluded = case["excluded"].as_str().unwrap();
    assert!(result.excluded.is_empty());
    let expected_position = position(&case["position_m"]);
    let step = position(&case["lsq_step"]);
    let final_step_norm = step.iter().map(|value| value * value).sum::<f64>().sqrt();
    for (actual, expected) in result
        .solution
        .position
        .as_array()
        .into_iter()
        .zip(expected_position)
    {
        assert!((actual - expected).abs() <= final_step_norm);
    }
    let rejected = result
        .solution
        .rejected_sats
        .iter()
        .find(|rejected| {
            !edge_inputs(&all, case)
                .observations
                .iter()
                .find(|observation| observation.satellite_id == rejected.satellite_id)
                .expect("rejected input")
                .pseudorange_m
                .is_finite()
        })
        .expect("non-finite observation is reported");
    assert_eq!(
        rejected.reason,
        sidereon_core::RejectionReason::NonFinitePseudorange
    );
    assert_eq!(rejected.satellite_id.to_string(), rtklib_excluded);
}

#[test]
fn public_spp_invalid_pseudoranges_match_corresponding_rtklib_solve_stage() {
    let (store, all, oracle) = edge_context();
    for (mode, expected_reason) in [
        (
            "infinite_pseudorange",
            sidereon_core::RejectionReason::NonFinitePseudorange,
        ),
        (
            "zero_pseudorange",
            sidereon_core::RejectionReason::NonPositivePseudorange,
        ),
        (
            "negative_pseudorange",
            sidereon_core::RejectionReason::NonPositivePseudorange,
        ),
        (
            "two_zero_pseudoranges",
            sidereon_core::RejectionReason::NonPositivePseudorange,
        ),
    ] {
        let case = oracle["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["mode"] == mode)
            .expect("generated edge case");
        assert!(case["excluded"].as_str().is_some());
        let result = fde_spp(
            &store,
            &edge_inputs(&all, case),
            false,
            &FdeSppOptions::default(),
        );
        let full_succeeded = case["full_estpos_stat"].as_u64().unwrap() == 1;
        let fde_succeeded = case["stat"].as_u64().unwrap() == 1;
        assert_eq!(result.is_ok(), full_succeeded || fde_succeeded, "{mode}");
        let result = result.unwrap_or_else(|error| panic!("{mode}: {error:?}"));
        if full_succeeded {
            // -1 is this wrapper's "pntpos not called" sentinel, not a
            // failing RTKLIB status. Its forced raim_fde diagnostic can spend
            // an exclusion (G27 in the zero cases) even though estpos already
            // succeeds. Compare this successful full solve with full estpos,
            // not with that independently forced exclusion diagnostic.
            assert_eq!(case["pntpos_stat"].as_i64().unwrap(), -1);
            let plain =
                sidereon_core::positioning::solve(&store, &edge_inputs(&all, case), false).unwrap();
            assert_eq!(plain.used_sats, result.solution.used_sats);
            assert_eq!(plain.position, result.solution.position);
        }
        assert!(result.excluded.is_empty(), "{mode}");
        let rejected: Vec<_> = result
            .solution
            .rejected_sats
            .iter()
            .filter(|rejection| rejection.reason == expected_reason)
            .collect();
        assert_eq!(
            rejected.len(),
            case["invalid_observations"].as_array().unwrap().len(),
            "{mode}"
        );
        // RTKLIB skips zero in rescode; the other classes enter residuals and
        // require raim_fde to remove the observation. The engine reports its
        // stricter pre-rejection without spending that exclusion budget.
        let reference_used = if full_succeeded {
            &case["full_used"]
        } else {
            &case["used"]
        };
        let expected_used: Vec<_> = reference_used
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap().to_string())
            .collect();
        let actual_used: Vec<_> = result
            .solution
            .used_sats
            .iter()
            .map(ToString::to_string)
            .collect();
        assert_eq!(actual_used, expected_used, "{mode}");
        let reference_position = if full_succeeded {
            position(&case["full_position_m"])
        } else {
            position(&case["position_m"])
        };
        let step = position(if full_succeeded {
            &case["full_lsq_step"]
        } else {
            &case["lsq_step"]
        });
        let step_norm = step.iter().map(|x| x * x).sum::<f64>().sqrt();
        for (actual, expected) in result
            .solution
            .position
            .as_array()
            .into_iter()
            .zip(reference_position)
        {
            assert!(
                (actual - expected).abs() <= step_norm,
                "{mode}: {actual} != {expected}"
            );
        }
    }
}
