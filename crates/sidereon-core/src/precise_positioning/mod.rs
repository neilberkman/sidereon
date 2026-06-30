//! Static multi-epoch PPP float/fixed positioning.
//!
//! This module owns the language-independent static PPP orchestration: float
//! carrier ambiguities, integer ambiguity resolution, and the fixed-ambiguity
//! re-solve from SP3-backed ionosphere-free code and phase observations. The
//! observation-only wide-lane/narrow-lane and cycle-slip preparation that runs
//! ahead of the solve lives in the [`prep`] leaf submodule.
//!
//! ```
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use std::collections::BTreeMap;
//! # use sidereon_core::constants::F_L1_HZ;
//! # use sidereon_core::observables::{
//! #     predict, ObservableEphemerisSource, ObservableState, ObservablesError, PredictOptions,
//! # };
//! # use sidereon_core::ppp_corrections::CivilDateTime;
//! # use sidereon_core::precise_positioning::{
//! #     solve_kinematic_ppp, FloatEpoch, FloatObservation, KinematicConfig, KinematicState,
//! # };
//! # use sidereon_core::{GnssSatelliteId, GnssSystem};
//! #
//! # struct Source {
//! #     states: BTreeMap<GnssSatelliteId, [f64; 3]>,
//! # }
//! #
//! # impl ObservableEphemerisSource for Source {
//! #     fn observable_state_at_j2000_s(
//! #         &self,
//! #         sat: GnssSatelliteId,
//! #         _t_j2000_s: f64,
//! #     ) -> Result<ObservableState, ObservablesError> {
//! #         Ok(ObservableState {
//! #             position_ecef_m: *self.states.get(&sat).ok_or(ObservablesError::NoEphemeris)?,
//! #             clock_s: Some(0.0),
//! #         })
//! #     }
//! # }
//! #
//! # fn diagonal_covariance(dimension: usize, variance_m2: f64) -> Vec<Vec<f64>> {
//! #     let mut covariance = vec![vec![0.0; dimension]; dimension];
//! #     for (idx, row) in covariance.iter_mut().enumerate() {
//! #         row[idx] = variance_m2;
//! #     }
//! #     covariance
//! # }
//! #
//! # let sats = [
//! #     (1u8, [20_200_000.0, 13_000_000.0, 21_500_000.0]),
//! #     (2, [-21_300_000.0, 14_500_000.0, 20_700_000.0]),
//! #     (3, [15_200_000.0, -22_000_000.0, 19_500_000.0]),
//! #     (4, [-18_200_000.0, -16_000_000.0, 21_000_000.0]),
//! #     (5, [22_000_000.0, -12_000_000.0, 20_200_000.0]),
//! # ];
//! # let ids = sats
//! #     .iter()
//! #     .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn))
//! #     .collect::<Result<Vec<_>, _>>()?;
//! # let source = Source {
//! #     states: ids
//! #         .iter()
//! #         .zip(sats.iter())
//! #         .map(|(id, (_, position))| (*id, *position))
//! #         .collect(),
//! # };
//! # let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
//! # let clock_m = 12.5;
//! # let ambiguities_m = ids
//! #     .iter()
//! #     .enumerate()
//! #     .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
//! #     .collect::<BTreeMap<_, _>>();
//! # let observations = ids
//! #     .iter()
//! #     .map(|id| {
//! #         let prediction = predict(
//! #             &source,
//! #             *id,
//! #             truth,
//! #             0.0,
//! #             PredictOptions {
//! #                 carrier_hz: F_L1_HZ,
//! #                 light_time: true,
//! #                 sagnac: true,
//! #             },
//! #         )?;
//! #         let code_m = prediction.geometric_range_m + clock_m;
//! #         let ambiguity_m = ambiguities_m.get(&id.to_string()).copied().unwrap();
//! #         Ok(FloatObservation {
//! #             sat: *id,
//! #             satellite_id: id.to_string(),
//! #             ambiguity_id: id.to_string(),
//! #             code_m,
//! #             phase_m: code_m + ambiguity_m,
//! #             freq1_hz: 0.0,
//! #             freq2_hz: 0.0,
//! #         })
//! #     })
//! #     .collect::<Result<Vec<_>, ObservablesError>>()?;
//! # let epoch = FloatEpoch {
//! #     epoch: CivilDateTime {
//! #         year: 2020,
//! #         month: 6,
//! #         day: 24,
//! #         hour: 12,
//! #         minute: 0,
//! #         second: 0.0,
//! #     },
//! #     jd_whole: 2_459_024.5,
//! #     jd_fraction: 0.5,
//! #     t_rx_j2000_s: 0.0,
//! #     observations,
//! # };
//! # let initial_state = KinematicState {
//! #     position_m: [truth[0] + 5.0, truth[1] - 4.0, truth[2] + 3.0],
//! #     clock_m: 0.0,
//! #     ztd_residual_m: 0.0,
//! #     ambiguities_m,
//! # };
//! # let config = KinematicConfig {
//! #     initial_covariance_m2: diagonal_covariance(initial_state.dimension(), 1.0e8),
//! #     initial_state,
//! #     ..KinematicConfig::default()
//! # };
//! let solutions = solve_kinematic_ppp(&source, &[epoch], config)?;
//! assert_eq!(solutions.len(), 1);
//! assert!(solutions[0].innovation_rms_m.is_finite());
//! # Ok(())
//! # }
//! ```

pub mod auto_init;
pub mod cycle_slip;
mod fixed;
mod float;
mod kinematic;
mod model;
mod normal;
mod prep;
pub mod raim;
mod rows;
pub mod tec;
mod types;
pub mod velocity;

pub use auto_init::{
    solve_ppp_auto_init_fixed, solve_ppp_auto_init_fixed_with_strategy, solve_ppp_auto_init_float,
    solve_ppp_auto_init_float_with_strategy, PppAutoInitError, PppAutoInitOptions,
    PppAutoInitStrategy, PppInitialGuess,
};
pub use cycle_slip::{
    detect_cycle_slips, geometry_free_m, melbourne_wubbena_cycles, update_geometry_free,
    update_melbourne_wubbena, CycleSlipConfig, CycleSlipConfigError, CycleSlipDetectorState,
    CycleSlipError, CycleSlipFlagEpoch, CycleSlipFlagObservation, CycleSlipStateKey,
    GeometryFreeUpdate, MelbourneWubbenaUpdate, RunningMeanVariance, SatelliteCycleSlipState,
    DEFAULT_MINIMUM_ARC_LENGTH, DEFAULT_RUNNING_STATISTIC_K_FACTOR,
};
pub(crate) use fixed::run_fixed_from_float;
pub use fixed::solve_fixed_from_float;
#[cfg(test)]
use float::initial_ambiguities;
pub(crate) use float::run_float_epochs;
pub use float::{solve_float_epoch, solve_float_epochs};
pub use kinematic::{
    correct_kinematic_state, predict_kinematic_state, solve_kinematic_ppp, KinematicConfig,
    KinematicEpochSolution, KinematicEpochStatus, KinematicMotionModel,
    KinematicPositionProcessNoise, KinematicProcessNoise, KinematicSolveError, KinematicState,
    KinematicUpdateSummary,
};
pub use prep::{
    prepare_widelane_fixed_epochs, split_float_cycle_slip_epochs, DualFrequencyEpoch,
    DualFrequencyObservation, FloatCycleSlipEpoch, FloatCycleSlipObservation,
    FloatCycleSlipTaggedEpoch, FloatCycleSlipTaggedObservation, PppSplitArc, PreparedFloatEpoch,
    PreparedFloatObservation, WideLanePrepError, WideLanePrepOptions, WideLanePrepResult,
};
pub use raim::{
    solve_float_epoch_with_raim, ProtectionLevels, RaimConfig, RaimError, RaimFdeError,
    RaimFdeResult, RaimFdeStatus, RaimGeometryRow, RaimIdentification, RaimResult, RaimStatus,
    SatelliteTestStatistic,
};
pub use tec::{
    code_geometry_free_m, estimate_code_slant_tec, estimate_phase_slant_tec, estimate_tec,
    ionospheric_pierce_point, level_slant_tec_arc, phase_geometry_free_m,
    slant_tec_from_code_geometry_free_m, slant_tec_from_phase_geometry_free_m,
    thin_shell_mapping_function, vertical_tec_from_slant_tec, CodeSlantTecEstimate,
    IonosphericPiercePoint, LeveledTecSample, PhaseSlantTecEstimate, TecConfig, TecEpoch, TecError,
    TecEstimate, TecEstimateSample, TecLevelingResult, TecLevelingSample, TecObservation,
    TecSatelliteArc, DEFAULT_IONOSPHERIC_SHELL_HEIGHT_M, ELECTRONS_PER_TECU_M2,
    TEC_GROUP_DELAY_COEFFICIENT,
};
pub use types::*;
pub use velocity::{
    predict_range_rate_m_s, solve_velocity, RangeRatePrediction, ReceiverVelocityState,
    VelocityConfig, VelocityObservation, VelocityRobustConfig, VelocitySolution,
    VelocitySolveError,
};

pub use crate::ambiguity::CycleSlipPolicy;

/// Canonical static-PPP solver defaults.
///
/// Single source of truth for the per-binding static-PPP option literals that
/// previously drifted across the Elixir, Python, and WASM bindings. A thin
/// binding builds a [`FloatSolveOptions`] / [`FixedAmbiguityOptions`] from these
/// constants instead of carrying its own numbers (or uses
/// [`FloatSolveOptions::default`], which reads them).
///
/// The values are the ones the core's PPP tests run with: every state tolerance
/// is `1.0e-4` m and the iterated float/fixed solve runs `8` iterations.
/// `RATIO_THRESHOLD` is the LAMBDA acceptance ratio (RTKLIB-demo5 `pos2-arthres`
/// default `3.0`). Exposing these does not change any solve; the solvers still
/// read the values from the caller's config.
pub mod defaults {
    /// Canonical receiver-position state-step convergence tolerance, metres.
    ///
    /// Feeds [`super::FloatSolveOptions::position_tolerance_m`].
    pub const POSITION_TOLERANCE_M: f64 = 1.0e-4;

    /// Canonical receiver-clock state-step convergence tolerance, metres.
    ///
    /// Feeds [`super::FloatSolveOptions::clock_tolerance_m`].
    pub const CLOCK_TOLERANCE_M: f64 = 1.0e-4;

    /// Canonical ambiguity state-step convergence tolerance, metres.
    ///
    /// Feeds [`super::FloatSolveOptions::ambiguity_tolerance_m`].
    pub const AMBIGUITY_TOLERANCE_M: f64 = 1.0e-4;

    /// Canonical zenith-total-delay state-step convergence tolerance, metres.
    ///
    /// Feeds [`super::FloatSolveOptions::ztd_tolerance_m`].
    pub const ZTD_TOLERANCE_M: f64 = 1.0e-4;

    /// Canonical maximum iterations for the static-PPP float/fixed solve.
    ///
    /// Feeds [`super::FloatSolveOptions::max_iterations`]. This is the value the
    /// core PPP goldens run with and the bindings hardcode.
    pub const MAX_ITERATIONS: usize = 8;

    /// Canonical LAMBDA acceptance ratio threshold for fixed PPP (dimensionless).
    ///
    /// Feeds [`super::FixedAmbiguityOptions::ratio_threshold`]. This is the
    /// RTKLIB-demo5 `pos2-arthres` default of 3.0, which the bindings hardcode
    /// and the core fixed-PPP goldens use.
    pub const RATIO_THRESHOLD: f64 = 3.0;
}

use std::collections::BTreeMap;

use crate::constants::F_L1_HZ;
use crate::estimation::recipe::NormalRecipe;
use crate::observables::{ObservableEphemerisSource, ObservablesError, PredictOptions};
use crate::ppp_corrections::{
    self, PppCorrectionEpoch, PppCorrectionObservation, PppCorrectionsError, PppCorrectionsOptions,
};
use crate::sp3::Sp3;
use crate::validate::{self, FieldError};

const MAX_PPP_ITERATIONS: usize = 10_000;

/// Build indexed PPP correction lookups at the static-arc seed position.
pub fn build_ppp_lookup(
    sp3: &Sp3,
    epochs: &[FloatEpoch],
    receiver_ecef_m: [f64; 3],
    options: &PppCorrectionsOptions,
) -> Result<PppCorrectionLookup, PppCorrectionsError> {
    let ppp_epochs: Vec<PppCorrectionEpoch> = epochs
        .iter()
        .map(|epoch| PppCorrectionEpoch {
            epoch: epoch.epoch,
            t_rx_j2000_s: epoch.t_rx_j2000_s,
            observations: epoch
                .observations
                .iter()
                .map(|obs| PppCorrectionObservation {
                    sat: obs.sat,
                    freq1_hz: obs.freq1_hz,
                    freq2_hz: obs.freq2_hz,
                })
                .collect(),
        })
        .collect();
    let corrections = ppp_corrections::build(sp3, &ppp_epochs, receiver_ecef_m, options)?;
    Ok(PppCorrectionLookup::from_options(corrections, options))
}

impl FloatState {
    fn default_for_epochs(epochs: &[FloatEpoch]) -> Self {
        Self {
            position_m: [0.0; 3],
            clocks_m: vec![0.0; epochs.len()],
            ambiguities_m: BTreeMap::new(),
            ztd_m: 0.0,
        }
    }
}

/// Measurement-model invariants shared across a whole static PPP solve: the
/// ephemeris source, measurement weighting, troposphere controls, the precomputed
/// range corrections, and the normal-equation operation-order recipe. Bundled so
/// the iterated solve, residual, and finalize helpers take one context argument
/// instead of repeating the same parameters everywhere; carrying `normal` here
/// lets the resolved recipe reach the solve seam without threading a parameter
/// through every iterate/screen helper.
#[derive(Clone, Copy)]
struct ModelContext<'a> {
    source: &'a dyn ObservableEphemerisSource,
    weights: MeasurementWeights,
    tropo: TroposphereOptions,
    corrections: &'a RangeCorrections,
    normal: NormalRecipe,
}

fn predict_default(
    _source: &dyn ObservableEphemerisSource,
    _obs: &FloatObservation,
) -> Result<PredictOptions, FloatSolveError> {
    Ok(PredictOptions {
        carrier_hz: F_L1_HZ,
        light_time: true,
        sagnac: true,
    })
}

fn no_ephemeris(obs: &FloatObservation, error: ObservablesError) -> FloatSolveError {
    FloatSolveError::NoEphemeris {
        satellite_id: obs.satellite_id.clone(),
        reason: match error {
            ObservablesError::NoEphemeris => NoEphemerisReason::NoEphemeris,
            ObservablesError::InvalidInput { .. } => NoEphemerisReason::Reason(error.to_string()),
            ObservablesError::Ephemeris(err) => NoEphemerisReason::Reason(err.to_string()),
        },
    }
}

fn missing_satellite_clock(obs: &FloatObservation) -> FloatSolveError {
    FloatSolveError::NoEphemeris {
        satellite_id: obs.satellite_id.clone(),
        reason: NoEphemerisReason::MissingSatelliteClock,
    }
}

fn missing_correction(obs: &FloatObservation, correction: MissingCorrection) -> FloatSolveError {
    FloatSolveError::MissingCorrection {
        satellite_id: obs.satellite_id.clone(),
        correction,
    }
}

fn invalid_clock_count(expected: usize, actual: usize) -> FloatSolveError {
    FloatSolveError::InvalidClockCount { expected, actual }
}

fn invalid_solve_option(field: &'static str, reason: &'static str) -> FloatSolveError {
    FloatSolveError::InvalidSolveOption { field, reason }
}

pub(super) fn invalid_input(error: FieldError) -> FloatSolveError {
    invalid_input_field(error.field(), error.reason())
}

fn invalid_input_field(field: &'static str, reason: &'static str) -> FloatSolveError {
    FloatSolveError::InvalidInput { field, reason }
}

fn invalid_fixed_input(error: FieldError) -> FixedSolveError {
    FixedSolveError::Float(invalid_input(error))
}

pub(super) fn validate_float_solve_boundary(
    epochs: &[FloatEpoch],
    state: &FloatState,
    config: &FloatSolveConfig,
) -> Result<(), FloatSolveError> {
    validate_epochs(epochs)?;
    validate_float_state(state, epochs.len())?;
    validate_float_config(config)
}

pub(super) fn validate_fixed_solve_boundary(
    epochs: &[FloatEpoch],
    solution: &FloatSolution,
    config: &FixedSolveConfig,
) -> Result<(), FixedSolveError> {
    validate_epochs(epochs).map_err(FixedSolveError::Float)?;
    validate_float_solution(solution, epochs.len())?;
    validate_fixed_config(config)
}

fn validate_epochs(epochs: &[FloatEpoch]) -> Result<(), FloatSolveError> {
    for epoch in epochs {
        validate_epoch(epoch)?;
    }
    Ok(())
}

fn validate_epoch(epoch: &FloatEpoch) -> Result<(), FloatSolveError> {
    validate::civil_datetime_with_second_policy(
        epoch.epoch.year as i64,
        epoch.epoch.month as i64,
        epoch.epoch.day as i64,
        epoch.epoch.hour as i64,
        epoch.epoch.minute as i64,
        epoch.epoch.second,
        validate::CivilSecondPolicy::Continuous,
    )
    .map_err(invalid_input)?;
    validate::finite(epoch.jd_whole, "ppp epoch jd_whole").map_err(invalid_input)?;
    validate::finite(epoch.jd_fraction, "ppp epoch jd_fraction").map_err(invalid_input)?;
    validate::finite(epoch.t_rx_j2000_s, "ppp epoch t_rx_j2000_s").map_err(invalid_input)?;
    for obs in &epoch.observations {
        validate_observation(obs)?;
    }
    Ok(())
}

fn validate_observation(obs: &FloatObservation) -> Result<(), FloatSolveError> {
    validate::finite(obs.code_m, "ppp observation code_m").map_err(invalid_input)?;
    validate::finite(obs.phase_m, "ppp observation phase_m").map_err(invalid_input)?;
    validate::finite(obs.freq1_hz, "ppp observation freq1_hz").map_err(invalid_input)?;
    validate::finite(obs.freq2_hz, "ppp observation freq2_hz").map_err(invalid_input)?;
    Ok(())
}

fn validate_float_state(state: &FloatState, n_epochs: usize) -> Result<(), FloatSolveError> {
    validate_state_clock_count(state, n_epochs)?;
    validate::finite_vec3(state.position_m, "ppp state position_m").map_err(invalid_input)?;
    validate::finite_slice(&state.clocks_m, "ppp state clocks_m").map_err(invalid_input)?;
    for value in state.ambiguities_m.values() {
        validate::finite(*value, "ppp state ambiguities_m").map_err(invalid_input)?;
    }
    validate::finite(state.ztd_m, "ppp state ztd_m").map_err(invalid_input)?;
    Ok(())
}

fn validate_float_solution(
    solution: &FloatSolution,
    n_epochs: usize,
) -> Result<(), FixedSolveError> {
    validate_solution_clock_count(solution, n_epochs)?;
    validate::finite_vec3(solution.position_m, "ppp float_solution position_m")
        .map_err(invalid_fixed_input)?;
    validate::finite_slice(
        &solution.epoch_clocks_m,
        "ppp float_solution epoch_clocks_m",
    )
    .map_err(invalid_fixed_input)?;
    for value in solution.ambiguities_m.values() {
        validate::finite(*value, "ppp float_solution ambiguities_m")
            .map_err(invalid_fixed_input)?;
    }
    if let Some(ztd_m) = solution.ztd_residual_m {
        validate::finite(ztd_m, "ppp float_solution ztd_residual_m")
            .map_err(invalid_fixed_input)?;
    }
    for residual in &solution.residuals_m {
        validate::finite(residual.code_m, "ppp float_solution residual code_m")
            .map_err(invalid_fixed_input)?;
        validate::finite(residual.phase_m, "ppp float_solution residual phase_m")
            .map_err(invalid_fixed_input)?;
        validate::finite(
            residual.code_weight,
            "ppp float_solution residual code_weight",
        )
        .map_err(invalid_fixed_input)?;
        validate::finite(
            residual.phase_weight,
            "ppp float_solution residual phase_weight",
        )
        .map_err(invalid_fixed_input)?;
    }
    validate::finite_nonneg(solution.code_rms_m, "ppp float_solution code_rms_m")
        .map_err(invalid_fixed_input)?;
    validate::finite_nonneg(solution.phase_rms_m, "ppp float_solution phase_rms_m")
        .map_err(invalid_fixed_input)?;
    validate::finite_nonneg(solution.weighted_rms_m, "ppp float_solution weighted_rms_m")
        .map_err(invalid_fixed_input)?;
    Ok(())
}

pub(super) fn validate_float_solution_output(
    solution: &FloatSolution,
    n_epochs: usize,
) -> Result<(), FloatSolveError> {
    validate_float_solution_clock_count(solution, n_epochs)?;
    validate::finite_vec3(solution.position_m, "ppp float_solution position_m")
        .map_err(invalid_input)?;
    validate::finite_slice(
        &solution.epoch_clocks_m,
        "ppp float_solution epoch_clocks_m",
    )
    .map_err(invalid_input)?;
    for value in solution.ambiguities_m.values() {
        validate::finite(*value, "ppp float_solution ambiguities_m").map_err(invalid_input)?;
    }
    if let Some(ztd_m) = solution.ztd_residual_m {
        validate::finite(ztd_m, "ppp float_solution ztd_residual_m").map_err(invalid_input)?;
    }
    for residual in &solution.residuals_m {
        validate::finite(residual.code_m, "ppp float_solution residual code_m")
            .map_err(invalid_input)?;
        validate::finite(residual.phase_m, "ppp float_solution residual phase_m")
            .map_err(invalid_input)?;
        validate::finite(
            residual.code_weight,
            "ppp float_solution residual code_weight",
        )
        .map_err(invalid_input)?;
        validate::finite(
            residual.phase_weight,
            "ppp float_solution residual phase_weight",
        )
        .map_err(invalid_input)?;
    }
    validate::finite_nonneg(solution.code_rms_m, "ppp float_solution code_rms_m")
        .map_err(invalid_input)?;
    validate::finite_nonneg(solution.phase_rms_m, "ppp float_solution phase_rms_m")
        .map_err(invalid_input)?;
    validate::finite_nonneg(solution.weighted_rms_m, "ppp float_solution weighted_rms_m")
        .map_err(invalid_input)?;
    Ok(())
}

fn validate_float_config(config: &FloatSolveConfig) -> Result<(), FloatSolveError> {
    validate_common_config(
        config.weights,
        config.tropo,
        &config.corrections,
        config.opts,
    )
}

fn validate_fixed_config(config: &FixedSolveConfig) -> Result<(), FixedSolveError> {
    validate_common_config(
        config.weights,
        config.tropo,
        &config.corrections,
        config.opts,
    )
    .map_err(FixedSolveError::Float)?;
    validate_fixed_ambiguity_options(&config.ambiguity)
}

fn validate_common_config(
    weights: MeasurementWeights,
    tropo: TroposphereOptions,
    corrections: &RangeCorrections,
    opts: FloatSolveOptions,
) -> Result<(), FloatSolveError> {
    validate_measurement_weights(weights)?;
    validate_troposphere_options(tropo)?;
    validate_range_corrections(corrections)?;
    validate_float_solve_options(opts)
}

fn validate_measurement_weights(weights: MeasurementWeights) -> Result<(), FloatSolveError> {
    validate::finite_positive(weights.code, "ppp measurement weight code")
        .map_err(invalid_input)?;
    validate::finite_positive(weights.phase, "ppp measurement weight phase")
        .map_err(invalid_input)?;
    Ok(())
}

fn validate_troposphere_options(tropo: TroposphereOptions) -> Result<(), FloatSolveError> {
    if !tropo.enabled {
        return Ok(());
    }
    validate::finite_positive(tropo.met.pressure_hpa, "ppp tropo pressure_hpa")
        .map_err(invalid_input)?;
    validate::finite_positive(tropo.met.temperature_k, "ppp tropo temperature_k")
        .map_err(invalid_input)?;
    validate::fraction(tropo.met.relative_humidity, "ppp tropo relative_humidity")
        .map_err(invalid_input)?;
    Ok(())
}

fn validate_range_corrections(corrections: &RangeCorrections) -> Result<(), FloatSolveError> {
    if let Some(receiver) = &corrections.receiver_antenna {
        validate::finite_positive(receiver.freq1_hz, "ppp receiver antenna freq1_hz")
            .map_err(invalid_input)?;
        validate::finite_positive(receiver.freq2_hz, "ppp receiver antenna freq2_hz")
            .map_err(invalid_input)?;
        if receiver.freq1_hz == receiver.freq2_hz {
            return Err(invalid_input_field(
                "ppp receiver antenna frequency pair",
                "must differ",
            ));
        }
        for frequency in &receiver.frequencies {
            validate_receiver_antenna_frequency(frequency)?;
        }
    }
    if let Some(clock) = &corrections.satellite_clock {
        for records in clock.series.values() {
            validate::require_strictly_increasing(
                records.iter().map(|&(t_gps_s, _)| t_gps_s),
                "ppp satellite clock epoch_s",
            )
            .map_err(invalid_input)?;
            for &(t_gps_s, bias_s) in records {
                validate::finite(t_gps_s, "ppp satellite clock epoch_s").map_err(invalid_input)?;
                validate::finite(bias_s, "ppp satellite clock bias_s").map_err(invalid_input)?;
            }
        }
    }
    for vector in corrections.ppp.tide.values() {
        validate::finite_vec3(*vector, "ppp correction tide vector_m").map_err(invalid_input)?;
    }
    for vector in corrections.ppp.pole_tide.values() {
        validate::finite_vec3(*vector, "ppp correction pole_tide vector_m")
            .map_err(invalid_input)?;
    }
    for vector in corrections.ppp.ocean_loading.values() {
        validate::finite_vec3(*vector, "ppp correction ocean_loading vector_m")
            .map_err(invalid_input)?;
    }
    for value in corrections.ppp.windup_m.values() {
        validate::finite(*value, "ppp correction windup_m").map_err(invalid_input)?;
    }
    for vector in corrections.ppp.sat_pco_ecef.values() {
        validate::finite_vec3(*vector, "ppp correction sat_pco_ecef").map_err(invalid_input)?;
    }
    for value in corrections.ppp.sat_pcv_m.values() {
        validate::finite(*value, "ppp correction sat_pcv_m").map_err(invalid_input)?;
    }
    Ok(())
}

fn validate_receiver_antenna_frequency(
    frequency: &ReceiverAntennaFrequency,
) -> Result<(), FloatSolveError> {
    validate::finite_vec3(frequency.pco_m, "ppp receiver antenna pco_m").map_err(invalid_input)?;
    for sample in &frequency.pcv_samples {
        validate_pcv_sample(sample)?;
    }
    Ok(())
}

fn validate_pcv_sample(sample: &PcvSample) -> Result<(), FloatSolveError> {
    if let Some(azimuth_deg) = sample.azimuth_deg {
        validate::finite(azimuth_deg, "ppp receiver antenna pcv azimuth_deg")
            .map_err(invalid_input)?;
    }
    validate::finite_in_range(
        sample.zenith_deg,
        0.0,
        180.0,
        "ppp receiver antenna pcv zenith_deg",
    )
    .map_err(invalid_input)?;
    validate::finite(sample.value_m, "ppp receiver antenna pcv value_m").map_err(invalid_input)?;
    Ok(())
}

fn validate_fixed_ambiguity_options(
    ambiguity: &FixedAmbiguityOptions,
) -> Result<(), FixedSolveError> {
    validate::finite_nonneg(
        ambiguity.ratio_threshold,
        "ppp fixed ambiguity ratio_threshold",
    )
    .map_err(invalid_fixed_input)?;
    for value in ambiguity.wavelengths_m.values() {
        validate::finite_positive(*value, "ppp fixed ambiguity wavelength_m")
            .map_err(invalid_fixed_input)?;
    }
    for value in ambiguity.offsets_m.values() {
        validate::finite(*value, "ppp fixed ambiguity offset_m").map_err(invalid_fixed_input)?;
    }
    Ok(())
}

fn validate_float_solve_options(opts: FloatSolveOptions) -> Result<(), FloatSolveError> {
    if opts.max_iterations == 0 {
        return Err(invalid_solve_option("max_iterations", "must be positive"));
    }
    if opts.max_iterations > MAX_PPP_ITERATIONS {
        return Err(invalid_solve_option(
            "max_iterations",
            "exceeds the PPP iteration cap",
        ));
    }
    validate_tolerance("position_tolerance_m", opts.position_tolerance_m)?;
    validate_tolerance("clock_tolerance_m", opts.clock_tolerance_m)?;
    validate_tolerance("ambiguity_tolerance_m", opts.ambiguity_tolerance_m)?;
    validate_tolerance("ztd_tolerance_m", opts.ztd_tolerance_m)
}

fn validate_tolerance(field: &'static str, value: f64) -> Result<(), FloatSolveError> {
    if validate::finite(value, field).is_err() {
        return Err(invalid_solve_option(field, "must be finite"));
    }
    if value <= 0.0 {
        return Err(invalid_solve_option(field, "must be positive"));
    }
    Ok(())
}

fn validate_state_clock_count(state: &FloatState, n_epochs: usize) -> Result<(), FloatSolveError> {
    if state.clocks_m.len() == n_epochs {
        Ok(())
    } else {
        Err(invalid_clock_count(n_epochs, state.clocks_m.len()))
    }
}

fn validate_solution_clock_count(
    solution: &FloatSolution,
    n_epochs: usize,
) -> Result<(), FixedSolveError> {
    if solution.epoch_clocks_m.len() == n_epochs {
        Ok(())
    } else {
        Err(FixedSolveError::Float(invalid_clock_count(
            n_epochs,
            solution.epoch_clocks_m.len(),
        )))
    }
}

fn validate_float_solution_clock_count(
    solution: &FloatSolution,
    n_epochs: usize,
) -> Result<(), FloatSolveError> {
    if solution.epoch_clocks_m.len() == n_epochs {
        Ok(())
    } else {
        Err(invalid_clock_count(n_epochs, solution.epoch_clocks_m.len()))
    }
}

fn state_from_solution(solution: &FloatSolution, prior: &FloatState) -> FloatState {
    FloatState {
        position_m: solution.position_m,
        clocks_m: solution.epoch_clocks_m.clone(),
        ambiguities_m: solution.ambiguities_m.clone(),
        ztd_m: solution.ztd_residual_m.unwrap_or(prior.ztd_m),
    }
}

fn estimates_ztd(tropo: TroposphereOptions) -> bool {
    tropo.enabled && tropo.estimate_ztd
}

fn ztd_unknown_count(tropo: TroposphereOptions) -> usize {
    usize::from(estimates_ztd(tropo))
}

fn rms(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|v| v * v).sum::<f64>() / values.len() as f64).sqrt()
}

fn weighted_rms(rows: &[FloatResidual], weights: MeasurementWeights) -> f64 {
    let mut values = Vec::with_capacity(rows.len() * 2);
    for row in rows {
        values.push(row.code_m * row.code_weight);
        values.push(row.phase_m * row.phase_weight);
    }
    if values.is_empty() {
        rms(&[0.0 * weights.code, 0.0 * weights.phase])
    } else {
        rms(&values)
    }
}

fn max_abs(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

#[cfg(test)]
mod tests;
