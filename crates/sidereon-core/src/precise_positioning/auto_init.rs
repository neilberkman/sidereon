//! SPP-seeded auto-initialization driver for static multi-epoch PPP arcs.
//!
//! This leaf owns the high-level "raw epochs in, solved arc out" orchestration
//! that previously lived only in the Elixir binding. It seeds the static PPP
//! float state from the existing single-point-positioning solver and the raw
//! observations, then runs the existing static float solve ([`solve_float_epochs`])
//! and, for the fixed driver, the existing integer-fixed re-solve
//! ([`solve_fixed_from_float`]). It re-implements no solve numerics; only the
//! seeding policy moves here so every binding can delegate to one driver.
//!
//! The seeding policy reproduces the Elixir reference
//! (`Sidereon.GNSS.PrecisePositioning.solve_float_epochs/3`) exactly so a later
//! binding delegation is bit-for-bit:
//!
//! 1. **SPP seed** - per epoch, a code-only single-point solve with the
//!    ionosphere correction off (the troposphere optional), each epoch using the
//!    caller's cold-start guess. The first SPP failure aborts the whole driver.
//! 2. **Mean-position seed** - the static position seed is the unweighted
//!    arithmetic mean of every epoch's SPP position, summed in reverse-epoch
//!    order to match the reference's floating-point reduction.
//! 3. **Per-epoch clock seed** - each epoch keeps its own SPP receiver clock
//!    (seconds times the speed of light), in arc order.
//! 4. **Phase-minus-code ambiguity seed** - each ambiguity id is seeded with
//!    `phase_m - code_m` (metres) from the first epoch, in sorted-observation
//!    order, where that id is seen.
//! 5. **ZTD seed** - the zenith-total-delay residual seeds to zero.
//!
//! When the caller supplies an explicit initial guess, the SPP/mean stages are
//! skipped and that position with its clock (duplicated across epochs) is the
//! seed, again matching the reference.

use std::collections::BTreeMap;

use crate::astro::time::{day_of_year, second_of_day};
use crate::constants::C_M_S;
use crate::estimation::recipe::{StrategyId, Technique};
use crate::estimation::strategies::{
    estimate, EstimateError, EstimateInput, EstimateOptions, EstimateOutput,
};
use crate::observables::ObservableEphemerisSource;
use crate::spp::{
    self, Corrections, EphemerisSource, KlobucharCoeffs, Observation as SppObservation, SppError,
    SurfaceMet,
};

use super::{
    solve_fixed_from_float, solve_float_epochs, FixedSolution, FixedSolveConfig, FixedSolveError,
    FloatEpoch, FloatSolution, FloatSolveConfig, FloatSolveError, FloatState,
};

/// Explicit static-position/clock seed that bypasses the SPP auto-init stages.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PppInitialGuess {
    /// Static receiver position seed (ECEF metres).
    pub position_m: [f64; 3],
    /// Receiver clock seed (metres), duplicated across every epoch.
    pub clock_m: f64,
}

/// Auto-initialization policy for the raw-epochs PPP driver.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PppAutoInitOptions {
    /// Explicit seed. `Some` skips the SPP/mean stages entirely (the Elixir
    /// `:initial_guess`); `None` runs the per-epoch SPP auto-init.
    pub initial_guess: Option<PppInitialGuess>,
    /// SPP cold-start guess `[x_m, y_m, z_m, b_m]` for every per-epoch seed solve
    /// (the Elixir `:spp_initial_guess`, default all-zero).
    pub spp_initial_guess: [f64; 4],
    /// Apply the troposphere correction in the SPP seed solve (the Elixir
    /// `:troposphere`, default off). The ionosphere is always off in the seed.
    pub spp_troposphere: bool,
    /// Surface meteorology used by the SPP seed troposphere (when enabled).
    pub spp_met: SurfaceMet,
}

impl Default for PppAutoInitOptions {
    /// Canonical auto-init defaults mirroring the Elixir reference: no explicit
    /// guess, an all-zero SPP cold start, the troposphere off, and standard
    /// surface meteorology (1013.25 hPa, 288.15 K, 0.5 relative humidity).
    fn default() -> Self {
        Self {
            initial_guess: None,
            spp_initial_guess: [0.0; 4],
            spp_troposphere: false,
            spp_met: SurfaceMet {
                pressure_hpa: 1013.25,
                temperature_k: 288.15,
                relative_humidity: 0.5,
            },
        }
    }
}

/// Runtime strategy selector for the PPP auto-init drivers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PppAutoInitStrategy {
    /// Reference static PPP solve path.
    #[default]
    Reference,
    /// Canonical static PPP solve path.
    Canonical,
}

impl PppAutoInitStrategy {
    const fn strategy_id(self) -> StrategyId {
        match self {
            Self::Reference => StrategyId::ppp_reference(),
            Self::Canonical => StrategyId::Canonical {
                technique: Technique::Ppp,
            },
        }
    }
}

/// Why the raw-epochs PPP driver could not complete.
#[derive(Debug, Clone)]
pub enum PppAutoInitError {
    /// The arc has no epochs.
    EmptyEpochs,
    /// A per-epoch SPP seed solve failed; the whole driver aborts on the first
    /// failure (the Elixir `:code_seed_failed`).
    CodeSeedFailed {
        /// Index of the failing epoch in the input arc.
        epoch_index: usize,
        /// The SPP failure.
        source: SppError,
    },
    /// The seeded static float solve failed.
    Float(FloatSolveError),
    /// The integer-fixed re-solve failed.
    Fixed(FixedSolveError),
}

impl core::fmt::Display for PppAutoInitError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyEpochs => write!(f, "PPP auto-init requires at least one epoch"),
            Self::CodeSeedFailed {
                epoch_index,
                source,
            } => write!(f, "PPP code seed failed at epoch {epoch_index}: {source}"),
            Self::Float(error) => write!(f, "{error}"),
            Self::Fixed(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for PppAutoInitError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::CodeSeedFailed { source, .. } => Some(source),
            Self::Float(error) => Some(error),
            Self::Fixed(error) => Some(error),
            Self::EmptyEpochs => None,
        }
    }
}

/// Solve a static multi-epoch float PPP arc from raw epochs, auto-initializing
/// the float state from the SPP seed described on the module.
///
/// This is the float raw-epochs driver: it seeds the state and then calls the
/// existing [`solve_float_epochs`]. The `source` is used both as the SPP seed
/// ephemeris and as the PPP observable ephemeris (`Sp3` and the broadcast
/// ephemeris implement both traits).
pub fn solve_ppp_auto_init_float<S>(
    source: &S,
    epochs: &[FloatEpoch],
    options: PppAutoInitOptions,
    config: FloatSolveConfig,
) -> Result<FloatSolution, PppAutoInitError>
where
    S: EphemerisSource + ObservableEphemerisSource,
{
    let initial_state = seed_state(source, epochs, options)?;
    solve_float_epochs(source, epochs, initial_state, config).map_err(PppAutoInitError::Float)
}

/// Solve a static multi-epoch float PPP arc from raw epochs, selecting the PPP
/// strategy after the auto-init seed is built.
pub fn solve_ppp_auto_init_float_with_strategy<S>(
    source: &S,
    epochs: &[FloatEpoch],
    options: PppAutoInitOptions,
    config: FloatSolveConfig,
    strategy: PppAutoInitStrategy,
) -> Result<FloatSolution, PppAutoInitError>
where
    S: EphemerisSource + ObservableEphemerisSource,
{
    let initial_state = seed_state(source, epochs, options)?;
    solve_float_with_strategy(source, epochs, initial_state, config, strategy)
}

/// Solve a static integer-fixed PPP arc from raw epochs: auto-init seed, the
/// float solve, then the LAMBDA integer fix and ambiguity-conditioned re-solve.
///
/// This reproduces the Elixir `solve_fixed_epochs/3` order: the float arc is
/// solved first (from the same auto-init seed as [`solve_ppp_auto_init_float`]),
/// then [`solve_fixed_from_float`] runs the integer search and fixed re-solve.
pub fn solve_ppp_auto_init_fixed<S>(
    source: &S,
    epochs: &[FloatEpoch],
    options: PppAutoInitOptions,
    float_config: FloatSolveConfig,
    fixed_config: FixedSolveConfig,
) -> Result<FixedSolution, PppAutoInitError>
where
    S: EphemerisSource + ObservableEphemerisSource,
{
    let float_solution = solve_ppp_auto_init_float(source, epochs, options, float_config)?;
    solve_fixed_from_float(source, epochs, float_solution, fixed_config)
        .map_err(PppAutoInitError::Fixed)
}

/// Solve a static integer-fixed PPP arc from raw epochs, selecting the PPP
/// strategy for both the float seed solve and the fixed re-solve.
pub fn solve_ppp_auto_init_fixed_with_strategy<S>(
    source: &S,
    epochs: &[FloatEpoch],
    options: PppAutoInitOptions,
    float_config: FloatSolveConfig,
    fixed_config: FixedSolveConfig,
    strategy: PppAutoInitStrategy,
) -> Result<FixedSolution, PppAutoInitError>
where
    S: EphemerisSource + ObservableEphemerisSource,
{
    let float_solution =
        solve_ppp_auto_init_float_with_strategy(source, epochs, options, float_config, strategy)?;
    solve_fixed_with_strategy(source, epochs, float_solution, fixed_config, strategy)
}

fn solve_float_with_strategy(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    initial_state: FloatState,
    config: FloatSolveConfig,
    strategy: PppAutoInitStrategy,
) -> Result<FloatSolution, PppAutoInitError> {
    match estimate(
        EstimateInput::PppFloat {
            source,
            epochs,
            initial_state,
            config,
        },
        EstimateOptions::new(strategy.strategy_id()),
    ) {
        Ok(EstimateOutput::PppFloat(solution)) => Ok(*solution),
        Err(EstimateError::PppFloat(error)) => Err(PppAutoInitError::Float(error)),
        Ok(_) | Err(_) => {
            unreachable!("PPP float strategy produces a PPP float result or error")
        }
    }
}

fn solve_fixed_with_strategy(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    float_solution: FloatSolution,
    config: FixedSolveConfig,
    strategy: PppAutoInitStrategy,
) -> Result<FixedSolution, PppAutoInitError> {
    match estimate(
        EstimateInput::PppFixed {
            source,
            epochs,
            float_solution,
            config,
        },
        EstimateOptions::new(strategy.strategy_id()),
    ) {
        Ok(EstimateOutput::PppFixed(solution)) => Ok(*solution),
        Err(EstimateError::PppFixed(error)) => Err(PppAutoInitError::Fixed(error)),
        Ok(_) | Err(_) => {
            unreachable!("PPP fixed strategy produces a PPP fixed result or error")
        }
    }
}

/// Build the auto-initialized float state (position, per-epoch clocks, float
/// ambiguities, zero ZTD residual) from the raw epochs.
fn seed_state<S>(
    source: &S,
    epochs: &[FloatEpoch],
    options: PppAutoInitOptions,
) -> Result<FloatState, PppAutoInitError>
where
    S: EphemerisSource + ObservableEphemerisSource,
{
    if epochs.is_empty() {
        return Err(PppAutoInitError::EmptyEpochs);
    }
    let (position_m, clocks_m) = match options.initial_guess {
        Some(guess) => (guess.position_m, vec![guess.clock_m; epochs.len()]),
        None => spp_seed(source, epochs, options)?,
    };
    Ok(FloatState {
        position_m,
        clocks_m,
        ambiguities_m: initial_ambiguities(epochs),
        ztd_m: 0.0,
    })
}

/// Per-epoch SPP code seed: returns the mean static position (summed in
/// reverse-epoch order, matching the Elixir reduction) and the per-epoch clock
/// seeds in arc order. Aborts on the first SPP failure.
fn spp_seed<S>(
    source: &S,
    epochs: &[FloatEpoch],
    options: PppAutoInitOptions,
) -> Result<([f64; 3], Vec<f64>), PppAutoInitError>
where
    S: EphemerisSource + ObservableEphemerisSource,
{
    let mut positions = Vec::with_capacity(epochs.len());
    let mut clocks = Vec::with_capacity(epochs.len());
    for (epoch_index, epoch) in epochs.iter().enumerate() {
        let inputs = spp_seed_inputs(epoch, options);
        let solution = spp::solve(source, &inputs, false).map_err(|source| {
            PppAutoInitError::CodeSeedFailed {
                epoch_index,
                source,
            }
        })?;
        positions.push(solution.position.as_array());
        clocks.push(solution.rx_clock_s * C_M_S);
    }
    Ok((mean_position(&positions), clocks))
}

/// Build the SPP seed inputs for one epoch: code-only pseudoranges, ionosphere
/// off, the optional troposphere, and the caller's cold-start guess.
fn spp_seed_inputs(epoch: &FloatEpoch, options: PppAutoInitOptions) -> spp::SolveInputs {
    let observations = epoch
        .observations
        .iter()
        .map(|obs| SppObservation {
            satellite_id: obs.sat,
            pseudorange_m: obs.code_m,
        })
        .collect();
    spp::SolveInputs {
        observations,
        t_rx_j2000_s: epoch.t_rx_j2000_s,
        t_rx_second_of_day_s: second_of_day(
            i32::from(epoch.epoch.hour),
            i32::from(epoch.epoch.minute),
            epoch.epoch.second,
        ),
        day_of_year: day_of_year(
            epoch.epoch.year,
            i32::from(epoch.epoch.month),
            i32::from(epoch.epoch.day),
            i32::from(epoch.epoch.hour),
            i32::from(epoch.epoch.minute),
            epoch.epoch.second,
        ),
        initial_guess: options.spp_initial_guess,
        corrections: Corrections {
            ionosphere: false,
            troposphere: options.spp_troposphere,
        },
        klobuchar: KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        glonass_channels: BTreeMap::new(),
        met: options.spp_met,
        robust: None,
    }
}

/// Unweighted arithmetic mean of the per-epoch SPP positions.
///
/// The sum walks the positions in reverse-epoch order: the Elixir reference
/// accumulates SPP positions onto a reversed list and reduces that list, so the
/// floating-point addition order is last-epoch-first. Replicating it keeps the
/// seed bit-for-bit.
fn mean_position(positions: &[[f64; 3]]) -> [f64; 3] {
    let mut sum = [0.0_f64; 3];
    for position in positions.iter().rev() {
        sum[0] += position[0];
        sum[1] += position[1];
        sum[2] += position[2];
    }
    let n = positions.len() as f64;
    [sum[0] / n, sum[1] / n, sum[2] / n]
}

/// Phase-minus-code float ambiguity seed per ambiguity id.
///
/// Folds over every observation in arc-then-observation order and keeps the
/// first-sighting `phase_m - code_m` (metres) for each ambiguity id, matching
/// the Elixir `Map.put_new` fold. The [`BTreeMap`] result is the column key set
/// the float solve indexes by.
fn initial_ambiguities(epochs: &[FloatEpoch]) -> BTreeMap<String, f64> {
    let mut ambiguities = BTreeMap::new();
    for epoch in epochs {
        for obs in &epoch.observations {
            ambiguities
                .entry(obs.ambiguity_id.clone())
                .or_insert(obs.phase_m - obs.code_m);
        }
    }
    ambiguities
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::math::vec3::{norm3, sub3};
    use crate::constants::F_L1_HZ;
    use crate::estimation::strategies::{
        estimate as estimate_with_strategy, EstimateInput, EstimateOptions, EstimateOutput,
    };
    use crate::observables::{predict, ObservableState, ObservablesError, PredictOptions};
    use crate::ppp_corrections::CivilDateTime;
    use crate::precise_positioning::{
        FixedAmbiguityOptions, FixedSolveConfig, FloatObservation, FloatSolution,
        FloatSolveOptions, MeasurementWeights, RangeCorrections, TroposphereOptions,
    };
    use crate::{GnssSatelliteId, GnssSystem};
    use std::collections::BTreeMap;

    /// Time-invariant ephemeris implementing both the SPP and observable traits.
    struct SeedSource {
        states: BTreeMap<GnssSatelliteId, [f64; 3]>,
    }

    impl ObservableEphemerisSource for SeedSource {
        fn observable_state_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(ObservableState {
                position_ecef_m: self
                    .states
                    .get(&sat)
                    .copied()
                    .ok_or(ObservablesError::NoEphemeris)?,
                clock_s: Some(0.0),
            })
        }
    }

    impl EphemerisSource for SeedSource {
        fn position_clock_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Option<([f64; 3], f64)> {
            self.states
                .get(&sat)
                .copied()
                .map(|position| (position, 0.0))
        }
    }

    fn sat_layout() -> [(u8, [f64; 3]); 6] {
        // Six satellites all well above the SPP elevation mask at the truth
        // receiver (elevations ~66-90 deg), so the per-epoch SPP seed keeps all
        // six and the geometry is well conditioned.
        [
            (1, [14_350_000.0, 3_190_000.0, 21_440_000.0]),
            (2, [20_000_000.0, 3_000_000.0, 18_000_000.0]),
            (3, [9_000_000.0, 9_000_000.0, 22_000_000.0]),
            (4, [16_000_000.0, -4_000_000.0, 21_000_000.0]),
            (5, [10_000_000.0, -2_000_000.0, 24_000_000.0]),
            (6, [19_000_000.0, 8_000_000.0, 17_000_000.0]),
        ]
    }

    fn source_and_ids() -> (SeedSource, Vec<GnssSatelliteId>) {
        let layout = sat_layout();
        let ids: Vec<GnssSatelliteId> = layout
            .iter()
            .map(|(prn, _)| GnssSatelliteId::new(GnssSystem::Gps, *prn).expect("valid prn"))
            .collect();
        let states = ids
            .iter()
            .zip(layout.iter())
            .map(|(id, (_, position))| (*id, *position))
            .collect();
        (SeedSource { states }, ids)
    }

    fn make_epoch(
        source: &SeedSource,
        ids: &[GnssSatelliteId],
        truth: [f64; 3],
        clock_m: f64,
        ambiguities_m: &BTreeMap<String, f64>,
        t_rx_j2000_s: f64,
    ) -> FloatEpoch {
        let observations = ids
            .iter()
            .map(|id| {
                let prediction = predict(
                    source,
                    *id,
                    truth,
                    t_rx_j2000_s,
                    PredictOptions {
                        carrier_hz: F_L1_HZ,
                        light_time: true,
                        sagnac: true,
                    },
                )
                .expect("prediction");
                let code_m = prediction.geometric_range_m + clock_m;
                let ambiguity_m = ambiguities_m[&id.to_string()];
                FloatObservation {
                    sat: *id,
                    satellite_id: id.to_string(),
                    ambiguity_id: id.to_string(),
                    code_m,
                    phase_m: code_m + ambiguity_m,
                    freq1_hz: 0.0,
                    freq2_hz: 0.0,
                }
            })
            .collect();
        FloatEpoch {
            epoch: CivilDateTime {
                year: 2020,
                month: 6,
                day: 24,
                hour: 12,
                minute: 0,
                second: 0.0,
            },
            jd_whole: 2_459_024.5,
            jd_fraction: 0.5,
            t_rx_j2000_s,
            observations,
        }
    }

    fn float_config() -> FloatSolveConfig {
        FloatSolveConfig {
            weights: MeasurementWeights {
                code: 1.0,
                phase: 100.0,
                elevation_weighting: false,
            },
            tropo: TroposphereOptions::disabled(),
            corrections: RangeCorrections::disabled(),
            opts: FloatSolveOptions::default(),
            residual_screen: false,
        }
    }

    fn fixed_config(ids: &[GnssSatelliteId], wavelength_m: f64) -> FixedSolveConfig {
        let wavelengths_m: BTreeMap<String, f64> = ids
            .iter()
            .map(|id| (id.to_string(), wavelength_m))
            .collect();
        let offsets_m: BTreeMap<String, f64> = ids.iter().map(|id| (id.to_string(), 0.0)).collect();
        FixedSolveConfig {
            weights: float_config().weights,
            tropo: float_config().tropo,
            corrections: float_config().corrections,
            opts: float_config().opts,
            ambiguity: FixedAmbiguityOptions {
                wavelengths_m,
                offsets_m,
                ratio_threshold: super::super::defaults::RATIO_THRESHOLD,
            },
        }
    }

    fn manual_float_with_strategy(
        source: &SeedSource,
        epochs: &[FloatEpoch],
        strategy: PppAutoInitStrategy,
    ) -> FloatSolution {
        let initial_state =
            seed_state(source, epochs, PppAutoInitOptions::default()).expect("seed builds");
        match estimate_with_strategy(
            EstimateInput::PppFloat {
                source,
                epochs,
                initial_state,
                config: float_config(),
            },
            EstimateOptions::new(strategy.strategy_id()),
        )
        .expect("manual float strategy")
        {
            EstimateOutput::PppFloat(solution) => *solution,
            _ => unreachable!("PPP float estimate returns PPP float output"),
        }
    }

    #[test]
    fn auto_init_float_recovers_truth() {
        let (source, ids) = source_and_ids();
        let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
        let ambiguities_m: BTreeMap<String, f64> = ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.to_string(), 0.25 + idx as f64 * 0.1))
            .collect();
        let clocks = [12.5, 13.0, 11.8];
        let epochs: Vec<FloatEpoch> = clocks
            .iter()
            .enumerate()
            .map(|(idx, &clock_m)| {
                make_epoch(&source, &ids, truth, clock_m, &ambiguities_m, idx as f64)
            })
            .collect();

        let solution = solve_ppp_auto_init_float(
            &source,
            &epochs,
            PppAutoInitOptions::default(),
            float_config(),
        )
        .expect("float arc solves");

        let error_m = norm3(sub3(solution.position_m, truth));
        assert!(error_m < 1.0e-3, "position error {error_m} m too large");
        for (idx, id) in ids.iter().enumerate() {
            let recovered = solution.ambiguities_m[&id.to_string()];
            let expected = 0.25 + idx as f64 * 0.1;
            assert!(
                (recovered - expected).abs() < 1.0e-3,
                "ambiguity {id} recovered {recovered} expected {expected}"
            );
        }
    }

    #[test]
    fn auto_init_matches_explicit_float_solve() {
        // The driver is a thin seed in front of `solve_float_epochs`: seeding by
        // hand from the same SPP policy and calling the existing solver must give
        // the identical solution, proving the driver adds no solve behavior.
        let (source, ids) = source_and_ids();
        let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
        let ambiguities_m: BTreeMap<String, f64> = ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.to_string(), 0.4 + idx as f64 * 0.05))
            .collect();
        let clocks = [9.5, 10.25];
        let epochs: Vec<FloatEpoch> = clocks
            .iter()
            .enumerate()
            .map(|(idx, &clock_m)| {
                make_epoch(&source, &ids, truth, clock_m, &ambiguities_m, idx as f64)
            })
            .collect();

        let driven = solve_ppp_auto_init_float(
            &source,
            &epochs,
            PppAutoInitOptions::default(),
            float_config(),
        )
        .expect("driver solves");

        let hand_state =
            seed_state(&source, &epochs, PppAutoInitOptions::default()).expect("seed builds");
        let by_hand =
            solve_float_epochs(&source, &epochs, hand_state, float_config()).expect("hand solve");
        assert_eq!(driven, by_hand);
    }

    #[test]
    fn auto_init_float_with_strategy_matches_manual_strategy_composition() {
        let (source, ids) = source_and_ids();
        let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
        let ambiguities_m: BTreeMap<String, f64> = ids
            .iter()
            .enumerate()
            .map(|(idx, id)| (id.to_string(), 0.35 + idx as f64 * 0.07))
            .collect();
        let epochs: Vec<FloatEpoch> = [8.5, 9.25, 8.9]
            .iter()
            .enumerate()
            .map(|(idx, &clock_m)| {
                make_epoch(&source, &ids, truth, clock_m, &ambiguities_m, idx as f64)
            })
            .collect();

        for strategy in [
            PppAutoInitStrategy::Reference,
            PppAutoInitStrategy::Canonical,
        ] {
            let driven = solve_ppp_auto_init_float_with_strategy(
                &source,
                &epochs,
                PppAutoInitOptions::default(),
                float_config(),
                strategy,
            )
            .expect("strategy driver solves");
            let manual = manual_float_with_strategy(&source, &epochs, strategy);
            assert_eq!(driven, manual);
        }
    }

    #[test]
    fn auto_init_fixed_holds_integers() {
        let (source, ids) = source_and_ids();
        let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
        let wavelength_m = C_M_S / F_L1_HZ;
        // Integer-cycle ambiguities so the LAMBDA fix has an exact lattice point.
        let cycles = [5i64, -3, 8, 2, -6, 4];
        let ambiguities_m: BTreeMap<String, f64> = ids
            .iter()
            .zip(cycles.iter())
            .map(|(id, &n)| (id.to_string(), n as f64 * wavelength_m))
            .collect();
        let clocks = [12.5, 12.7, 12.6];
        let epochs: Vec<FloatEpoch> = clocks
            .iter()
            .enumerate()
            .map(|(idx, &clock_m)| {
                make_epoch(&source, &ids, truth, clock_m, &ambiguities_m, idx as f64)
            })
            .collect();

        let fixed_config = fixed_config(&ids, wavelength_m);

        let fixed = solve_ppp_auto_init_fixed(
            &source,
            &epochs,
            PppAutoInitOptions::default(),
            float_config(),
            fixed_config,
        )
        .expect("fixed arc solves");

        let error_m = norm3(sub3(fixed.position_m, truth));
        assert!(
            error_m < 1.0e-3,
            "fixed position error {error_m} m too large"
        );
        for (id, &n) in ids.iter().zip(cycles.iter()) {
            let held = fixed.fixed_ambiguities_cycles[&id.to_string()];
            assert_eq!(held, n, "satellite {id} integer cycle");
        }
    }

    #[test]
    fn auto_init_fixed_with_strategy_matches_manual_strategy_composition() {
        let (source, ids) = source_and_ids();
        let truth = [3_512_900.0, 780_500.0, 5_248_700.0];
        let wavelength_m = C_M_S / F_L1_HZ;
        let cycles = [4i64, -2, 6, 1, -5, 3];
        let ambiguities_m: BTreeMap<String, f64> = ids
            .iter()
            .zip(cycles.iter())
            .map(|(id, &n)| (id.to_string(), n as f64 * wavelength_m))
            .collect();
        let epochs: Vec<FloatEpoch> = [11.5, 11.7, 11.6]
            .iter()
            .enumerate()
            .map(|(idx, &clock_m)| {
                make_epoch(&source, &ids, truth, clock_m, &ambiguities_m, idx as f64)
            })
            .collect();
        let strategy = PppAutoInitStrategy::Canonical;
        let fixed_config = fixed_config(&ids, wavelength_m);

        let driven = solve_ppp_auto_init_fixed_with_strategy(
            &source,
            &epochs,
            PppAutoInitOptions::default(),
            float_config(),
            fixed_config.clone(),
            strategy,
        )
        .expect("strategy fixed driver solves");
        let manual_float = manual_float_with_strategy(&source, &epochs, strategy);
        let manual = match estimate_with_strategy(
            EstimateInput::PppFixed {
                source: &source,
                epochs: &epochs,
                float_solution: manual_float,
                config: fixed_config,
            },
            EstimateOptions::new(strategy.strategy_id()),
        )
        .expect("manual fixed strategy")
        {
            EstimateOutput::PppFixed(solution) => *solution,
            _ => unreachable!("PPP fixed estimate returns PPP fixed output"),
        };

        assert_eq!(driven, manual);
    }
}
