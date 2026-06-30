#![cfg(sidereon_repo_tests)]

use sidereon_core::astro::time::model::JulianDateSplit;
use sidereon_core::astro::time::split_julian_date;
use sidereon_core::atmosphere::troposphere::Met;
use sidereon_core::combinations::{ionosphere_free, ionosphere_free_phase_cycles};
use sidereon_core::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::observables::j2000_seconds_from_split;
use sidereon_core::ppp_corrections::CivilDateTime;
use sidereon_core::precise_positioning::{
    prepare_widelane_fixed_epochs, solve_fixed_from_float, solve_float_epochs, CycleSlipPolicy,
    DualFrequencyEpoch, DualFrequencyObservation, FixedAmbiguityOptions, FixedSolveConfig,
    FloatEpoch, FloatObservation, FloatSolveConfig, FloatSolveOptions, FloatState, IntegerStatus,
    MeasurementWeights, RangeCorrections, TroposphereOptions, WideLanePrepError,
    WideLanePrepOptions,
};
use sidereon_core::rinex::observations::{
    observation_values, ObsEpoch, ObsEpochTime, ObservationFilter, RinexObs,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

fn fixture_path(parts: &[&str]) -> PathBuf {
    parts.iter().fold(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures"),
        |path, part| path.join(part),
    )
}

fn load_text(parts: &[&str]) -> String {
    let path = fixture_path(parts);
    std::fs::read_to_string(&path).unwrap_or_else(|err| panic!("read fixture {path:?}: {err}"))
}

fn load_sp3() -> Sp3 {
    let path = fixture_path(&["sp3", "GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3"]);
    let bytes = std::fs::read(&path).unwrap_or_else(|err| panic!("read fixture {path:?}: {err}"));
    Sp3::parse(&bytes).unwrap_or_else(|err| panic!("parse SP3 {path:?}: {err}"))
}

fn load_obs() -> RinexObs {
    RinexObs::parse(&load_text(&[
        "obs",
        "ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx",
    ]))
    .expect("parse ESBC observation fixture")
}

fn civil_to_julian_split(epoch: ObsEpochTime) -> JulianDateSplit {
    let (jd_whole, fraction) = split_julian_date(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    );
    JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date")
}

fn civil_datetime(epoch: ObsEpochTime) -> CivilDateTime {
    CivilDateTime {
        year: epoch.year,
        month: epoch.month,
        day: epoch.day,
        hour: epoch.hour,
        minute: epoch.minute,
        second: epoch.second,
    }
}

fn j2000_seconds(epoch: ObsEpochTime) -> f64 {
    let split = civil_to_julian_split(epoch);
    j2000_seconds_from_split(split.jd_whole, split.fraction).expect("valid split Julian date")
}

fn gps_l1_l2_filter() -> ObservationFilter {
    ObservationFilter::from_entries([(
        GnssSystem::Gps,
        vec![
            "C1C".to_string(),
            "C2W".to_string(),
            "L1C".to_string(),
            "L2W".to_string(),
        ],
    )])
}

fn gps_float_epochs(obs: &RinexObs, count: usize, excluded: &[&str]) -> Vec<FloatEpoch> {
    let excluded = excluded.iter().copied().collect::<BTreeSet<_>>();
    obs.epochs()
        .iter()
        .take(count)
        .map(|epoch| {
            let observations = float_observations(epoch, obs, &excluded);
            assert!(
                observations.len() >= 6,
                "fixture epoch {:?} has only {} complete GPS L1/L2 rows",
                epoch.epoch,
                observations.len()
            );
            float_epoch(epoch.epoch, observations)
        })
        .collect()
}

fn float_observations(
    epoch: &ObsEpoch,
    obs: &RinexObs,
    excluded: &BTreeSet<&str>,
) -> Vec<FloatObservation> {
    let mut out = observation_values(obs, epoch, &gps_l1_l2_filter())
        .expect("valid observation values")
        .into_iter()
        .filter_map(|(sat, rows)| {
            let token = sat.to_string();
            if excluded.contains(token.as_str()) {
                return None;
            }
            let mut values = BTreeMap::new();
            for row in rows {
                values.insert(row.code, row.value);
            }
            let code_m = ionosphere_free(
                values.get("C1C").and_then(|v| *v)?,
                values.get("C2W").and_then(|v| *v)?,
                F_L1_HZ,
                F_L2_HZ,
            )
            .expect("ionosphere-free code");
            let phase_m = ionosphere_free_phase_cycles(
                values.get("L1C").and_then(|v| *v)?,
                values.get("L2W").and_then(|v| *v)?,
                F_L1_HZ,
                F_L2_HZ,
            )
            .expect("ionosphere-free carrier phase");
            Some(FloatObservation {
                sat,
                satellite_id: token.clone(),
                ambiguity_id: token,
                code_m,
                phase_m,
                freq1_hz: 0.0,
                freq2_hz: 0.0,
            })
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));
    out
}

fn float_epoch(epoch: ObsEpochTime, observations: Vec<FloatObservation>) -> FloatEpoch {
    let split = civil_to_julian_split(epoch);
    FloatEpoch {
        epoch: civil_datetime(epoch),
        jd_whole: split.jd_whole,
        jd_fraction: split.fraction,
        t_rx_j2000_s: j2000_seconds_from_split(split.jd_whole, split.fraction)
            .expect("valid split Julian date"),
        observations,
    }
}

fn gps_dual_epochs(obs: &RinexObs, count: usize) -> Vec<DualFrequencyEpoch> {
    obs.epochs()
        .iter()
        .take(count)
        .map(|epoch| {
            let observations = dual_observations(epoch, obs);
            assert!(
                observations.len() >= 6,
                "fixture epoch {:?} has only {} complete GPS L1/L2 rows",
                epoch.epoch,
                observations.len()
            );
            DualFrequencyEpoch {
                gap_time_s: Some(j2000_seconds(epoch.epoch)),
                observations,
            }
        })
        .collect()
}

fn dual_observations(epoch: &ObsEpoch, obs: &RinexObs) -> Vec<DualFrequencyObservation> {
    let mut out = observation_values(obs, epoch, &gps_l1_l2_filter())
        .expect("valid observation values")
        .into_iter()
        .filter_map(|(sat, rows)| {
            let mut c1 = None;
            let mut c2 = None;
            let mut l1 = None;
            let mut l2 = None;
            let mut lli1 = None;
            let mut lli2 = None;
            for row in rows {
                match row.code.as_str() {
                    "C1C" => c1 = row.value,
                    "C2W" => c2 = row.value,
                    "L1C" => {
                        l1 = row.value;
                        lli1 = row.lli.map(i64::from);
                    }
                    "L2W" => {
                        l2 = row.value;
                        lli2 = row.lli.map(i64::from);
                    }
                    _ => {}
                }
            }
            let token = sat.to_string();
            Some(DualFrequencyObservation {
                satellite_id: token.clone(),
                ambiguity_id: token,
                p1_m: c1?,
                p2_m: c2?,
                phi1_cyc: l1?,
                phi2_cyc: l2?,
                f1_hz: F_L1_HZ,
                f2_hz: F_L2_HZ,
                lli1,
                lli2,
            })
        })
        .collect::<Vec<_>>();
    out.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));
    out
}

fn initial_state(epochs: &[FloatEpoch], truth: [f64; 3]) -> FloatState {
    FloatState {
        position_m: [truth[0] + 100.0, truth[1] - 100.0, truth[2] + 100.0],
        clocks_m: vec![0.0; epochs.len()],
        ambiguities_m: initial_ambiguities(epochs),
        ztd_m: 0.0,
    }
}

fn initial_ambiguities(epochs: &[FloatEpoch]) -> BTreeMap<String, f64> {
    let mut out = BTreeMap::new();
    for obs in epochs.iter().flat_map(|epoch| &epoch.observations) {
        out.entry(obs.ambiguity_id.clone())
            .or_insert(obs.phase_m - obs.code_m);
    }
    out
}

fn float_config(
    elevation_weighting: bool,
    tropo: TroposphereOptions,
    max_iterations: usize,
) -> FloatSolveConfig {
    FloatSolveConfig {
        weights: MeasurementWeights {
            code: 1.0,
            phase: 100.0,
            elevation_weighting,
        },
        tropo,
        corrections: RangeCorrections::disabled(),
        opts: FloatSolveOptions {
            max_iterations,
            position_tolerance_m: 1.0e-4,
            clock_tolerance_m: 1.0e-4,
            ambiguity_tolerance_m: 1.0e-4,
            ztd_tolerance_m: 1.0e-4,
        },
        residual_screen: false,
    }
}

fn fixed_config(
    elevation_weighting: bool,
    tropo: TroposphereOptions,
    max_iterations: usize,
    wavelengths_m: BTreeMap<String, f64>,
    offsets_m: BTreeMap<String, f64>,
) -> FixedSolveConfig {
    FixedSolveConfig {
        weights: MeasurementWeights {
            code: 1.0,
            phase: 100.0,
            elevation_weighting,
        },
        tropo,
        corrections: RangeCorrections::disabled(),
        opts: FloatSolveOptions {
            max_iterations,
            position_tolerance_m: 1.0e-4,
            clock_tolerance_m: 1.0e-4,
            ambiguity_tolerance_m: 1.0e-4,
            ztd_tolerance_m: 1.0e-4,
        },
        ambiguity: FixedAmbiguityOptions {
            wavelengths_m,
            offsets_m,
            ratio_threshold: 3.0,
        },
    }
}

fn tropo(enabled: bool, estimate_ztd: bool) -> TroposphereOptions {
    if enabled {
        TroposphereOptions {
            enabled: true,
            estimate_ztd,
            met: Met::new(1013.25, 288.15, 0.5).expect("valid PPP troposphere met"),
            ..TroposphereOptions::disabled()
        }
    } else {
        TroposphereOptions::disabled()
    }
}

fn scalar_wavelengths(epochs: &[FloatEpoch], wavelength_m: f64) -> BTreeMap<String, f64> {
    satellite_ids(epochs)
        .into_iter()
        .map(|sat| (sat, wavelength_m))
        .collect()
}

fn zero_offsets(epochs: &[FloatEpoch]) -> BTreeMap<String, f64> {
    satellite_ids(epochs)
        .into_iter()
        .map(|sat| (sat, 0.0))
        .collect()
}

fn satellite_ids(epochs: &[FloatEpoch]) -> Vec<String> {
    epochs
        .iter()
        .flat_map(|epoch| {
            epoch
                .observations
                .iter()
                .map(|obs| obs.ambiguity_id.clone())
        })
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn gps_id(token: &str) -> GnssSatelliteId {
    let prn = token
        .strip_prefix('G')
        .unwrap_or_else(|| panic!("expected GPS token, got {token:?}"))
        .parse::<u8>()
        .unwrap_or_else(|_| panic!("bad GPS token {token:?}"));
    GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
}

fn position_error_m(position_m: [f64; 3], truth_m: [f64; 3]) -> f64 {
    ((position_m[0] - truth_m[0]).powi(2)
        + (position_m[1] - truth_m[1]).powi(2)
        + (position_m[2] - truth_m[2]).powi(2))
    .sqrt()
}

fn wide_lane_options(tolerance_cycles: f64) -> WideLanePrepOptions {
    WideLanePrepOptions {
        min_epochs: 2,
        tolerance_cycles,
    }
}

fn cycle_slip_options() -> sidereon_core::carrier_phase::CycleSlipOptions {
    sidereon_core::carrier_phase::CycleSlipOptions {
        gf_threshold_m: 0.05,
        mw_threshold_cycles: 4.0,
        min_arc_gap_s: 300.0,
    }
}

/// Bounded-tolerance band (m) for canonical PPP vs the PPP-oracle-faithful
/// reference PPP float position on the shared troposphere-corrected ESBC arc.
/// Canonical and reference solve the SAME dense SPD weighted normal system
/// `AᵀWA x = AᵀWy` (the same undifferenced rows, the same Gauss-Newton iteration
/// to the same 1e-4 m convergence tolerance) and differ only in factorization:
/// the reference by dense last-tie Gaussian elimination, canonical by the owned
/// Cholesky (square-root-information) factorization. The two converged solutions
/// therefore agree to the f64 roundoff cluster of two factorizations of one
/// ~155-unknown SPD system, observed at ~1.9e-9 m -- far inside the solver's own
/// 1e-4 m convergence tolerance. The band holds ~500x above that floor; a
/// divergence beyond it is a canonical bug to root-cause, not a tolerance to
/// widen.
const CANONICAL_VS_REFERENCE_PPP_TOL_M: f64 = 1.0e-6;

/// Coarse model-regression bound (m): the canonical PPP float position vs the
/// surveyed ESBC00DNK RINEX APPROX POSITION XYZ. This is a 5 m bound, NOT a PPP
/// accuracy signal: this arc runs with [`RangeCorrections::disabled`] and only
/// troposphere on, so the dominant relativistic-clock term and the antenna /
/// tide / wind-up stack are absent (the periodic relativity term alone is
/// meters-level). What it guards is that the undifferenced code/phase model and
/// the dense weighted solve stay self-consistent end to end on a real arc; it
/// deliberately leaves the metre-level correction error in place.
///
/// The decimeter PPP truth signal -- the FULL correction stack on a real IGS
/// arc fed by a 30 s CLK, asserted within a decimeter of published ITRF2020
/// truth and cross-validated against RTKLIB ppp-static -- lives in
/// `tests/ppp_decimeter_arc.rs`. That is the test to read for PPP accuracy.
const CANONICAL_PPP_TRUTH_BOUND_M: f64 = 5.0;

/// P6 increment 3: the canonical PPP strategy, an ADDITIVE selectable strategy
/// whose canonical divergence is the numerically rigorous square-root-information
/// solve (the same owned Cholesky factorization as canonical RTK) on the dense
/// SPD weighted PPP normal system. It keeps the PPP-oracle-faithful undifferenced
/// ionosphere-free measurement model unchanged; only the linear solve differs.
/// Both canonical bars are checked on the real troposphere-corrected ESBC PPP
/// float arc:
///   1. DETERMINISM: canonical is bit-reproducible run-to-run on this pinned
///      build (the frozen-bits position golden below, re-asserted on a second
///      canonical solve). Scope (calibrated, not overstated): the owned Cholesky
///      solve is bit-portable, but the PPP measurement model that builds the rows
///      evaluates troposphere/antenna/geodetic transcendentals through the
///      platform math library, so these pinned bits are THIS build's reproducible
///      output, with the cross-platform guarantee scoped to the solve.
///   2. BOUNDED-TOLERANCE + TRUTH: canonical lands within
///      [`CANONICAL_VS_REFERENCE_PPP_TOL_M`] of the PPP-oracle-faithful reference
///      PPP float position on the shared case, and within
///      [`CANONICAL_PPP_TRUTH_BOUND_M`] of the surveyed ESBC APPROX POSITION.
#[test]
fn canonical_ppp_is_deterministic_bounded_and_truthful() {
    use sidereon_core::estimation::{
        estimate, EstimateInput, EstimateOptions, EstimateOutput, StrategyId, Technique,
    };

    let sp3 = load_sp3();
    let obs = load_obs();
    let truth = obs
        .header()
        .approx_position_m
        .expect("ESBC approx position");
    let epochs = gps_float_epochs(&obs, 120, &[]);
    assert_eq!(epochs.len(), 120);

    // Reference PPP float (PPP-oracle-faithful), the unchanged reference path.
    let reference = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(false, tropo(true, false), 8),
    )
    .expect("reference PPP float solve");

    let run_canonical = || -> sidereon_core::precise_positioning::FloatSolution {
        match estimate(
            EstimateInput::PppFloat {
                source: &sp3,
                epochs: &epochs,
                initial_state: initial_state(&epochs, truth),
                config: float_config(false, tropo(true, false), 8),
            },
            EstimateOptions::new(StrategyId::Canonical {
                technique: Technique::Ppp,
            }),
        )
        .expect("canonical PPP float solves")
        {
            EstimateOutput::PppFloat(solution) => *solution,
            other => panic!("canonical PPP must yield a PPP float solution, got {other:?}"),
        }
    };
    let canonical = run_canonical();

    let dpos = position_error_m(canonical.position_m, reference.position_m);
    let terr = position_error_m(canonical.position_m, truth);

    // BAR 2a: bounded tolerance vs the PPP-oracle-faithful reference.
    assert!(
        dpos < CANONICAL_VS_REFERENCE_PPP_TOL_M,
        "canonical PPP diverged from reference by {dpos} m (> {CANONICAL_VS_REFERENCE_PPP_TOL_M} m); root-cause, do not widen"
    );
    // BAR 2b: surveyed-truth bound.
    assert!(
        terr < CANONICAL_PPP_TRUTH_BOUND_M,
        "canonical PPP truth error was {terr} m (> {CANONICAL_PPP_TRUTH_BOUND_M} m)"
    );

    // BAR 1: frozen-bits determinism golden (this-build reproducible; the solve is
    // owned scalar, the surrounding measurement model rides platform libm).
    assert_eq!(canonical.position_m[0].to_bits(), 0x414b544c30f74f9a);
    assert_eq!(canonical.position_m[1].to_bits(), 0x412040d68a0054f5);
    assert_eq!(canonical.position_m[2].to_bits(), 0x4153f61c555f9818);

    // Determinism: a second canonical solve is bit-identical.
    let again = run_canonical();
    assert_eq!(
        canonical.position_m[0].to_bits(),
        again.position_m[0].to_bits()
    );
    assert_eq!(
        canonical.position_m[1].to_bits(),
        again.position_m[1].to_bits()
    );
    assert_eq!(
        canonical.position_m[2].to_bits(),
        again.position_m[2].to_bits()
    );
}

#[test]
fn esbc_real_float_ppp_arc_improves_with_troposphere_correction() {
    let sp3 = load_sp3();
    let obs = load_obs();
    let truth = obs
        .header()
        .approx_position_m
        .expect("ESBC approx position");
    let epochs = gps_float_epochs(&obs, 120, &[]);
    assert_eq!(epochs.len(), 120);

    let uncorrected = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(false, tropo(false, false), 8),
    )
    .expect("uncorrected PPP solve");
    let corrected = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(false, tropo(true, false), 8),
    )
    .expect("troposphere-corrected PPP solve");

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        let wavelength_m = C_M_S / (F_L1_HZ + F_L2_HZ);
        let fixed_cfg = fixed_config(
            false,
            tropo(true, false),
            8,
            scalar_wavelengths(&epochs, wavelength_m),
            zero_offsets(&epochs),
        );
        let fixed = solve_fixed_from_float(&sp3, &epochs, corrected.clone(), fixed_cfg.clone())
            .expect("fixed PPP fixture solve");
        dump_ppp_fixture(
            &epochs,
            &initial_state(&epochs, truth),
            corrected.position_m,
            &fixed_cfg,
            &fixed,
        );
    }

    let ztd_estimated = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(false, tropo(true, true), 8),
    )
    .expect("ZTD-estimated PPP solve");

    let uncorrected_error_m = position_error_m(uncorrected.position_m, truth);
    let corrected_error_m = position_error_m(corrected.position_m, truth);

    assert_eq!(corrected.residuals_m.len(), 1282);
    assert!(ztd_estimated.ztd_residual_m.expect("ZTD residual") > 0.0);
    assert!(ztd_estimated.ztd_residual_m.expect("ZTD residual") < 1.0);
    assert!(uncorrected_error_m > 20.0);
    assert!(corrected_error_m < 5.0);
    assert!(uncorrected_error_m - corrected_error_m > 18.0);
    assert!(ztd_estimated.weighted_rms_m < corrected.weighted_rms_m);
    assert!(ztd_estimated.phase_rms_m < corrected.phase_rms_m);

    let elevation_weighted = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(true, tropo(false, false), 8),
    )
    .expect("elevation-weighted PPP solve");
    let elevation_weighted_error_m = position_error_m(elevation_weighted.position_m, truth);

    assert!(elevation_weighted_error_m < uncorrected_error_m / 3.0);
    assert!(elevation_weighted.weighted_rms_m < uncorrected.weighted_rms_m / 5.0);
}

#[test]
fn esbc_real_noisy_narrow_lane_fix_refuses_unsafe_integer_solution() {
    let sp3 = load_sp3();
    let obs = load_obs();
    let truth = obs
        .header()
        .approx_position_m
        .expect("ESBC approx position");
    let epochs = gps_float_epochs(&obs, 120, &["G21"]);
    let wavelength_m = C_M_S / (F_L1_HZ + F_L2_HZ);

    let float_solution = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(false, tropo(true, false), 8),
    )
    .expect("float PPP solve");
    let fixed = solve_fixed_from_float(
        &sp3,
        &epochs,
        float_solution,
        fixed_config(
            false,
            tropo(true, false),
            8,
            scalar_wavelengths(&epochs, wavelength_m),
            zero_offsets(&epochs),
        ),
    )
    .expect("fixed PPP solve");

    assert_eq!(fixed.integer.integer_status, IntegerStatus::NotFixed);
    assert!(fixed.integer.integer_ratio < 3.0);
    assert_eq!(fixed.integer.integer_candidates, 2);
    assert!(position_error_m(fixed.position_m, truth) < 6.0);
}

#[test]
fn esbc_real_elevation_weighted_narrow_lane_fix_still_refuses_unsafe_fix() {
    let sp3 = load_sp3();
    let obs = load_obs();
    let truth = obs
        .header()
        .approx_position_m
        .expect("ESBC approx position");
    let epochs = gps_float_epochs(&obs, 120, &["G21"]);
    let wavelength_m = C_M_S / (F_L1_HZ + F_L2_HZ);

    let unweighted_float = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(false, tropo(false, false), 8),
    )
    .expect("unweighted float PPP solve");
    let unweighted = solve_fixed_from_float(
        &sp3,
        &epochs,
        unweighted_float,
        fixed_config(
            false,
            tropo(false, false),
            8,
            scalar_wavelengths(&epochs, wavelength_m),
            zero_offsets(&epochs),
        ),
    )
    .expect("unweighted fixed PPP solve");
    let weighted_float = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(true, tropo(false, false), 8),
    )
    .expect("weighted float PPP solve");
    let weighted = solve_fixed_from_float(
        &sp3,
        &epochs,
        weighted_float,
        fixed_config(
            true,
            tropo(false, false),
            8,
            scalar_wavelengths(&epochs, wavelength_m),
            zero_offsets(&epochs),
        ),
    )
    .expect("weighted fixed PPP solve");

    assert_eq!(unweighted.integer.integer_status, IntegerStatus::NotFixed);
    assert_eq!(weighted.integer.integer_status, IntegerStatus::NotFixed);
    assert!(weighted.integer.integer_ratio < 3.0);
    assert_eq!(weighted.integer.integer_candidates, 2);
    assert!(
        position_error_m(weighted.position_m, truth)
            < position_error_m(unweighted.position_m, truth) / 2.0
    );
}

#[test]
fn esbc_real_slipped_arcs_can_be_split_before_narrow_lane_search() {
    let sp3 = load_sp3();
    let obs = load_obs();
    let truth = obs
        .header()
        .approx_position_m
        .expect("ESBC approx position");
    let dual_epochs = gps_dual_epochs(&obs, 30);

    let slip = prepare_widelane_fixed_epochs(
        &dual_epochs,
        wide_lane_options(0.5),
        CycleSlipPolicy::Error,
        cycle_slip_options(),
    )
    .expect_err("default policy rejects G21 slip");
    assert!(matches!(
        slip,
        WideLanePrepError::CycleSlipDetected {
            ref satellite_id,
            ..
        } if satellite_id == "G21"
    ));

    let prep = prepare_widelane_fixed_epochs(
        &dual_epochs,
        wide_lane_options(2.0),
        CycleSlipPolicy::SplitArc,
        cycle_slip_options(),
    )
    .expect("split slipped wide-lane arcs");
    let epochs = prep
        .epochs
        .iter()
        .map(|prepared| {
            let raw_epoch = obs.epochs()[prepared.epoch_index].epoch;
            let observations = prepared
                .observations
                .iter()
                .map(|row| FloatObservation {
                    sat: gps_id(&row.satellite_id),
                    satellite_id: row.satellite_id.clone(),
                    ambiguity_id: row.ambiguity_id.clone(),
                    code_m: row.code_m,
                    phase_m: row.phase_m,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                })
                .collect();
            float_epoch(raw_epoch, observations)
        })
        .collect::<Vec<_>>();
    let float_solution = solve_float_epochs(
        &sp3,
        &epochs,
        initial_state(&epochs, truth),
        float_config(false, tropo(true, false), 5),
    )
    .expect("split-arc float PPP solve");
    let fixed = solve_fixed_from_float(
        &sp3,
        &epochs,
        float_solution,
        fixed_config(
            false,
            tropo(true, false),
            5,
            prep.wavelengths_m,
            prep.offsets_m,
        ),
    )
    .expect("split-arc fixed PPP solve");

    assert_eq!(fixed.integer.integer_status, IntegerStatus::NotFixed);
    assert!(fixed.integer.integer_ratio < 3.0);
    assert_eq!(fixed.integer.integer_candidates, 2);
    assert_eq!(prep.split_arcs.len(), 2);
    assert!(position_error_m(fixed.position_m, truth) < 9.0);
}

/// Env-gated emitter (`SIDEREON_DUMP_FIXTURES=1`) that serializes the fully
/// built ESBC static float-PPP arc (epochs, initial state, the troposphere-
/// corrected config) plus the engine's reference position to a JSON fixture
/// consumed by the Python binding's pytest. Reuses this validated harness
/// verbatim; changes no assertion and never runs in a normal `cargo test`.
fn dump_ppp_fixture(
    epochs: &[FloatEpoch],
    initial: &FloatState,
    expected_position_m: [f64; 3],
    fixed_config: &FixedSolveConfig,
    fixed: &sidereon_core::precise_positioning::FixedSolution,
) {
    use serde_json::{json, Value};

    let obs_json = |o: &FloatObservation| -> Value {
        json!({
            "satellite_id": o.satellite_id,
            "ambiguity_id": o.ambiguity_id,
            "code_m": o.code_m,
            "phase_m": o.phase_m,
            "freq1_hz": o.freq1_hz,
            "freq2_hz": o.freq2_hz,
        })
    };
    let epochs_json: Vec<Value> = epochs
        .iter()
        .map(|e| {
            json!({
                "civil": {
                    "year": e.epoch.year,
                    "month": e.epoch.month,
                    "day": e.epoch.day,
                    "hour": e.epoch.hour,
                    "minute": e.epoch.minute,
                    "second": e.epoch.second,
                },
                "jd_whole": e.jd_whole,
                "jd_fraction": e.jd_fraction,
                "t_rx_j2000_s": e.t_rx_j2000_s,
                "observations": e.observations.iter().map(obs_json).collect::<Vec<_>>(),
            })
        })
        .collect();

    let ambiguities: Vec<Value> = initial
        .ambiguities_m
        .iter()
        .map(|(id, v)| json!([id, v]))
        .collect();

    let opts_json = |opts: FloatSolveOptions| -> Value {
        json!({
            "max_iterations": opts.max_iterations,
            "position_tolerance_m": opts.position_tolerance_m,
            "clock_tolerance_m": opts.clock_tolerance_m,
            "ambiguity_tolerance_m": opts.ambiguity_tolerance_m,
            "ztd_tolerance_m": opts.ztd_tolerance_m,
        })
    };
    let weights_json = |weights: MeasurementWeights| -> Value {
        json!({
            "code": weights.code,
            "phase": weights.phase,
            "elevation_weighting": weights.elevation_weighting,
        })
    };
    let tropo_json = |tropo: TroposphereOptions| -> Value {
        json!({
            "enabled": tropo.enabled,
            "estimate_ztd": tropo.estimate_ztd,
            "pressure_hpa": tropo.met.pressure_hpa,
            "temperature_k": tropo.met.temperature_k,
            "relative_humidity": tropo.met.relative_humidity,
        })
    };

    let doc = json!({
        "source": "esbc_real_float_ppp_arc_improves_with_troposphere_correction",
        "sp3_file": "GBM0MGXRAP_20201770000_01D_05M_ORB_120epoch.sp3",
        "epochs": epochs_json,
        "initial_state": {
            "position_m": initial.position_m,
            "clocks_m": initial.clocks_m,
            "ambiguities_m": ambiguities,
            "ztd_m": initial.ztd_m,
        },
        "config": {
            "weights": { "code": 1.0, "phase": 100.0, "elevation_weighting": false },
            "tropo": {
                "enabled": true,
                "estimate_ztd": false,
                "pressure_hpa": 1013.25,
                "temperature_k": 288.15,
                "relative_humidity": 0.5,
            },
            "opts": {
                "max_iterations": 8,
                "position_tolerance_m": 1.0e-4,
                "clock_tolerance_m": 1.0e-4,
                "ambiguity_tolerance_m": 1.0e-4,
                "ztd_tolerance_m": 1.0e-4,
            },
            "residual_screen": false,
        },
        "fixed_config": {
            "weights": weights_json(fixed_config.weights),
            "tropo": tropo_json(fixed_config.tropo),
            "opts": opts_json(fixed_config.opts),
            "ambiguity": {
                "wavelengths_m": fixed_config.ambiguity.wavelengths_m.clone(),
                "offsets_m": fixed_config.ambiguity.offsets_m.clone(),
                "ratio_threshold": fixed_config.ambiguity.ratio_threshold,
            },
        },
        "expected": {
            "position_m": expected_position_m,
            "fixed_position_m": fixed.position_m,
            "fixed_float_position_m": fixed.float_solution.position_m,
            "fixed_integer_status": format!("{:?}", fixed.integer.integer_status),
            "fixed_integer_ratio": fixed.integer.integer_ratio,
            "fixed_integer_candidates": fixed.integer.integer_candidates,
            "fixed_ambiguities_cycles": fixed.fixed_ambiguities_cycles.clone(),
            "fixed_ambiguities_m": fixed.fixed_ambiguities_m.clone(),
        },
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/ppp_esbc.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped PPP fixture to {out:?}");
}
