//! Private, causal PPC RTK runner adapter.
//!
//! The current public arc driver chooses references from the complete arc. PPC
//! must not let future satellite availability influence an earlier solution, so
//! this module keeps the route policy in the unpublished CLI crate and feeds the
//! existing streaming filter one epoch at a time. This Phase 2 adapter is kept
//! deliberately single-frequency because restoring the historical dual-frequency
//! control would require estimator interfaces that are not present on current main.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, bail, Context, Result};
use sidereon_core::astro::math::linear::invert_matrix_first_tie;
use sidereon_core::astro::time::{j2000_seconds, TimeScale};
use sidereon_core::constants::C_M_S;
use sidereon_core::ils::IlsError;
use sidereon_core::observables::{
    is_observable_state_gap, ObservableEphemerisSource, ObservablesError,
};
use sidereon_core::rinex::observations::{
    observation_frequency_hz, observation_values, ObsEpoch, ObservationFilter, RinexObs,
};
use sidereon_core::rtk::{
    apply_elevation_mask, baseline_reference_satellites, BaselineReferenceEpoch,
    BaselineReferenceSelection, ElevationMaskEpoch,
};
use sidereon_core::rtk_filter::{
    update_epoch_with_scratch, AmbiguityScale, DynamicsModel, Epoch, EpochUpdate, FilterState,
    MeasModel, RtkArcEpoch, RtkArcObservation, RtkFilterScratch, SatMeas, SearchOpts,
    StochasticModel, UpdateError, UpdateOpts,
};
use sidereon_core::velocity::{
    self, VelocityObservable, VelocityObservation, VelocitySolveOptions,
};
use sidereon_core::{GnssSatelliteId, GnssSystem};

const MINIMUM_SATELLITES: usize = 4;
pub(crate) const DEFAULT_AMBIGUITY_RETIREMENT_AGE_S: f64 = 15.0;
pub(crate) const DEFAULT_MAX_AMBIGUITY_COLUMNS: usize = 128;
// Match the established private receiver-position contract in rinex_qc. PPC is
// a terrestrial benchmark; accepting an algebraically finite position outside
// this shell would serialize a corrupt filter state as a benchmark solution.
const EARTH_FIXED_RADIUS_MIN_M: f64 = 6_300_000.0;
const EARTH_FIXED_RADIUS_MAX_M: f64 = 6_400_000.0;

#[derive(Debug, Clone, Copy)]
pub(crate) struct PpcRunnerOptions {
    pub max_base_age_s: f64,
    pub max_epochs: Option<usize>,
    pub elevation_mask_deg: f64,
    pub hold_sigma_m: f64,
    pub process_noise_sigma_m: f64,
    pub velocity_dynamics: bool,
    pub ambiguity_retirement_age_s: f64,
    pub max_ambiguity_columns: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PpcSolutionEpoch {
    pub time_j2000_s: f64,
    pub position_ecef_m: [f64; 3],
    pub integer_fixed: bool,
    pub satellites_used: usize,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct PpcRunnerStats {
    pub rover_epochs_scanned: usize,
    pub arc_epochs: usize,
    pub missing_base_epochs: usize,
    pub stale_base_epochs: usize,
    pub unusable_epochs: usize,
    pub coasted_epochs: usize,
    pub fixed_epochs: usize,
    pub peak_ambiguity_columns: usize,
    pub retired_ambiguity_columns: usize,
    pub float_fallback_epochs: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PpcRunnerResult {
    pub epochs: Vec<PpcSolutionEpoch>,
    pub stats: PpcRunnerStats,
}

#[derive(Debug, Clone, Copy)]
struct SignalPair {
    system: GnssSystem,
    code: &'static str,
    phase: &'static str,
}

#[derive(Debug, Clone)]
struct SignalSelector {
    by_system: BTreeMap<GnssSystem, Vec<SignalPair>>,
    filter: ObservationFilter,
}

impl SignalSelector {
    fn ppc_l1() -> Self {
        // Keep every fallback on the same carrier band. In particular, the PPC
        // Galileo base uses the X tracking label while the rover uses C; the
        // receiver-local first match intentionally pairs those two E1 signals.
        let pairs = vec![
            pair(GnssSystem::Gps, "C1C", "L1C"),
            pair(GnssSystem::Gps, "C1X", "L1X"),
            pair(GnssSystem::Glonass, "C1C", "L1C"),
            pair(GnssSystem::Galileo, "C1C", "L1C"),
            pair(GnssSystem::Galileo, "C1X", "L1X"),
            pair(GnssSystem::BeiDou, "C2I", "L2I"),
            pair(GnssSystem::Qzss, "C1C", "L1C"),
            pair(GnssSystem::Qzss, "C1X", "L1X"),
        ];
        let mut by_system = BTreeMap::<GnssSystem, Vec<SignalPair>>::new();
        let mut codes = BTreeMap::<GnssSystem, BTreeSet<String>>::new();
        for pair in pairs {
            by_system.entry(pair.system).or_default().push(pair);
            let system_codes = codes.entry(pair.system).or_default();
            system_codes.insert(pair.code.to_string());
            system_codes.insert(pair.phase.to_string());
            system_codes.insert(doppler_code(pair.phase));
        }
        let filter = ObservationFilter::from_entries(
            codes
                .into_iter()
                .map(|(system, codes)| (system, codes.into_iter().collect())),
        );
        Self { by_system, filter }
    }
}

const fn pair(system: GnssSystem, code: &'static str, phase: &'static str) -> SignalPair {
    SignalPair {
        system,
        code,
        phase,
    }
}

fn doppler_code(phase_code: &str) -> String {
    format!("D{}", &phase_code[1..])
}

#[derive(Debug, Clone, PartialEq)]
struct SelectedObservation {
    satellite_id: GnssSatelliteId,
    ambiguity_id: String,
    phase_code: &'static str,
    code_m: f64,
    phase_m: f64,
    wavelength_m: f64,
    lli: Option<i64>,
    doppler_hz: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TimedEpoch {
    index: usize,
    time_j2000_s: f64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CausalJoinStep {
    advanced_base_indices: Vec<usize>,
    latest_base_index: Option<usize>,
    fresh_base_index: Option<usize>,
}

#[derive(Debug)]
enum BaseEpochSource<'a> {
    Rinex(&'a [ObsEpoch]),
    #[cfg(test)]
    Timed(&'a [TimedEpoch]),
}

#[derive(Debug)]
struct CausalEpochJoin<'a> {
    base_epochs: BaseEpochSource<'a>,
    cursor: usize,
    pending_base: Option<TimedEpoch>,
    latest_base: Option<TimedEpoch>,
    previous_base_time_j2000_s: Option<f64>,
    previous_rover_time_j2000_s: Option<f64>,
    saw_base_epoch: bool,
}

impl<'a> CausalEpochJoin<'a> {
    fn new(base_epochs: &'a [ObsEpoch]) -> Self {
        Self {
            base_epochs: BaseEpochSource::Rinex(base_epochs),
            cursor: 0,
            pending_base: None,
            latest_base: None,
            previous_base_time_j2000_s: None,
            previous_rover_time_j2000_s: None,
            saw_base_epoch: false,
        }
    }

    #[cfg(test)]
    fn from_timed(base_epochs: &'a [TimedEpoch]) -> Self {
        Self {
            base_epochs: BaseEpochSource::Timed(base_epochs),
            cursor: 0,
            pending_base: None,
            latest_base: None,
            previous_base_time_j2000_s: None,
            previous_rover_time_j2000_s: None,
            saw_base_epoch: false,
        }
    }

    fn advance(&mut self, rover_time_j2000_s: f64, max_base_age_s: f64) -> Result<CausalJoinStep> {
        if !rover_time_j2000_s.is_finite() {
            bail!("rover observation timestamp must be finite");
        }
        if self
            .previous_rover_time_j2000_s
            .is_some_and(|previous| rover_time_j2000_s <= previous)
        {
            bail!("rover RINEX observation timestamps must be strictly increasing");
        }
        if !max_base_age_s.is_finite() || max_base_age_s < 0.0 {
            bail!("maximum causal base age must be finite and non-negative");
        }

        let mut advanced_base_indices = Vec::new();
        loop {
            let candidate = match self.pending_base.take() {
                Some(candidate) => Some(candidate),
                None => self.next_base_epoch()?,
            };
            let Some(candidate) = candidate else {
                break;
            };
            if candidate.time_j2000_s > rover_time_j2000_s {
                self.pending_base = Some(candidate);
                break;
            }
            advanced_base_indices.push(candidate.index);
            self.latest_base = Some(candidate);
        }
        self.previous_rover_time_j2000_s = Some(rover_time_j2000_s);
        let latest_base_index = self.latest_base.map(|epoch| epoch.index);
        let fresh_base_index = self.latest_base.and_then(|epoch| {
            (rover_time_j2000_s - epoch.time_j2000_s <= max_base_age_s).then_some(epoch.index)
        });
        Ok(CausalJoinStep {
            advanced_base_indices,
            latest_base_index,
            fresh_base_index,
        })
    }

    fn next_base_epoch(&mut self) -> Result<Option<TimedEpoch>> {
        loop {
            let epoch = match self.base_epochs {
                BaseEpochSource::Rinex(epochs) => {
                    let Some(epoch) = epochs.get(self.cursor) else {
                        return Ok(None);
                    };
                    let index = self.cursor;
                    self.cursor += 1;
                    if epoch.flag > 1 {
                        continue;
                    }
                    TimedEpoch {
                        index,
                        time_j2000_s: epoch_j2000_s(epoch),
                    }
                }
                #[cfg(test)]
                BaseEpochSource::Timed(epochs) => {
                    let Some(epoch) = epochs.get(self.cursor).copied() else {
                        return Ok(None);
                    };
                    self.cursor += 1;
                    epoch
                }
            };
            if !epoch.time_j2000_s.is_finite() {
                bail!(
                    "base RINEX epoch {} has a non-finite timestamp",
                    epoch.index
                );
            }
            if self
                .previous_base_time_j2000_s
                .is_some_and(|previous| epoch.time_j2000_s < previous)
            {
                bail!("base RINEX observation timestamps must be non-decreasing");
            }
            self.previous_base_time_j2000_s = Some(epoch.time_j2000_s);
            self.saw_base_epoch = true;
            return Ok(Some(epoch));
        }
    }
}

#[derive(Debug, Clone)]
struct LatestBase {
    time_j2000_s: f64,
    observations: BTreeMap<String, SelectedObservation>,
}

#[derive(Debug, Default)]
struct ArcTracker {
    generations: BTreeMap<String, usize>,
    seen: BTreeSet<String>,
    previous_present: BTreeSet<String>,
    previous_phase_codes: BTreeMap<String, &'static str>,
}

impl ArcTracker {
    fn segment(
        &mut self,
        observations: &mut BTreeMap<String, SelectedObservation>,
        receiver_power_failure: bool,
    ) {
        let current = observations.keys().cloned().collect::<BTreeSet<_>>();
        let mut current_phase_codes = BTreeMap::new();
        for (satellite_id, observation) in observations {
            let reacquired =
                self.seen.contains(satellite_id) && !self.previous_present.contains(satellite_id);
            let lli_slip = observation.lli.is_some_and(|lli| lli & 1 != 0);
            let signal_changed = self
                .previous_phase_codes
                .get(satellite_id)
                .is_some_and(|previous| *previous != observation.phase_code);
            let generation = self.generations.entry(satellite_id.clone()).or_default();
            if reacquired || lli_slip || signal_changed || receiver_power_failure {
                *generation += 1;
            }
            observation.ambiguity_id = if *generation == 0 {
                satellite_id.clone()
            } else {
                format!("{satellite_id}#{}", *generation + 1)
            };
            self.seen.insert(satellite_id.clone());
            current_phase_codes.insert(satellite_id.clone(), observation.phase_code);
        }
        self.previous_present = current;
        self.previous_phase_codes = current_phase_codes;
    }
}

#[derive(Debug)]
struct CausalArc {
    epochs: Vec<RtkArcEpoch>,
    wavelengths_m: BTreeMap<String, f64>,
    offsets_m: BTreeMap<String, f64>,
    stats: PpcRunnerStats,
}

pub(crate) fn solve_ppc_route(
    ephemeris: &dyn ObservableEphemerisSource,
    base_obs: &RinexObs,
    rover_obs: &RinexObs,
    options: PpcRunnerOptions,
) -> Result<PpcRunnerResult> {
    validate_options(options)?;
    validate_gpst(base_obs, "base")?;
    validate_gpst(rover_obs, "rover")?;
    let base_m = base_obs
        .header()
        .approx_position_m
        .ok_or_else(|| anyhow!("base RINEX has no APPROX POSITION XYZ header"))?;
    validate_earth_fixed_position(base_m, "base RINEX APPROX POSITION XYZ")?;
    let selector = SignalSelector::ppc_l1();
    let arc = build_causal_arc(ephemeris, base_obs, rover_obs, base_m, &selector, options)?;
    solve_causal_arc(base_m, arc, options)
}

pub(crate) fn validate_options(options: PpcRunnerOptions) -> Result<()> {
    if !options.max_base_age_s.is_finite() || options.max_base_age_s < 0.0 {
        bail!("--max-base-age-s must be finite and non-negative");
    }
    if options.max_epochs == Some(0) {
        bail!("--max-epochs must be positive");
    }
    if !options.elevation_mask_deg.is_finite() || !(0.0..90.0).contains(&options.elevation_mask_deg)
    {
        bail!("--elevation-mask-deg must be finite and in [0, 90)");
    }
    if !options.hold_sigma_m.is_finite() || options.hold_sigma_m <= 0.0 {
        bail!("--hold-sigma-m must be finite and positive");
    }
    if !options.process_noise_sigma_m.is_finite() || options.process_noise_sigma_m < 0.0 {
        bail!("--process-noise-sigma-m must be finite and non-negative");
    }
    if !options.ambiguity_retirement_age_s.is_finite() || options.ambiguity_retirement_age_s < 0.0 {
        bail!("--ambiguity-retirement-age-s must be finite and non-negative");
    }
    if options.max_ambiguity_columns == 0 {
        bail!("--max-ambiguity-columns must be positive");
    }
    Ok(())
}

fn validate_gpst(obs: &RinexObs, receiver: &str) -> Result<()> {
    if let Some((_, scale)) = obs.header().time_of_first_obs {
        if scale != TimeScale::Gpst {
            bail!("{receiver} RINEX observation time scale must be GPST, got {scale:?}");
        }
    }
    Ok(())
}

fn build_causal_arc(
    ephemeris: &dyn ObservableEphemerisSource,
    base_obs: &RinexObs,
    rover_obs: &RinexObs,
    base_m: [f64; 3],
    selector: &SignalSelector,
    options: PpcRunnerOptions,
) -> Result<CausalArc> {
    if base_obs.epochs().is_empty() {
        bail!("base RINEX has no observation epochs");
    }
    if rover_obs.epochs().is_empty() {
        bail!("rover RINEX has no observation epochs");
    }

    let mut stats = PpcRunnerStats::default();
    let mut epochs = Vec::new();
    let mut wavelengths_m = BTreeMap::new();
    let mut offsets_m = BTreeMap::new();
    let mut base_join = CausalEpochJoin::new(base_obs.epochs());
    let mut latest_base = None::<LatestBase>;
    let mut base_tracker = ArcTracker::default();
    let mut rover_tracker = ArcTracker::default();

    for (rover_index, rover_epoch) in rover_obs.epochs().iter().enumerate() {
        if options
            .max_epochs
            .is_some_and(|limit| epochs.len() >= limit)
        {
            break;
        }
        if rover_epoch.flag > 1 {
            continue;
        }
        let rover = TimedEpoch {
            index: rover_index,
            time_j2000_s: epoch_j2000_s(rover_epoch),
        };
        stats.rover_epochs_scanned += 1;

        let joined = base_join.advance(rover.time_j2000_s, options.max_base_age_s)?;
        // Advance each native base epoch exactly once. Reusing a 1 Hz base row
        // for five rover rows must not apply its LLI five times.
        for base_index in joined.advanced_base_indices {
            let base_epoch = &base_obs.epochs()[base_index];
            let mut observations = select_observations(base_obs, base_epoch, selector)?;
            base_tracker.segment(&mut observations, base_epoch.flag == 1);
            latest_base = Some(LatestBase {
                time_j2000_s: epoch_j2000_s(base_epoch),
                observations,
            });
        }

        let mut rover_observations = select_observations(rover_obs, rover_epoch, selector)?;
        rover_tracker.segment(&mut rover_observations, rover_epoch.flag == 1);
        let velocity_mps = options.velocity_dynamics.then(|| {
            solve_rover_velocity(ephemeris, &rover_observations, base_m, rover.time_j2000_s)
        });
        let velocity_mps = velocity_mps.flatten();

        if joined.latest_base_index.is_none() {
            stats.missing_base_epochs += 1;
            continue;
        }
        if joined.fresh_base_index.is_none() {
            stats.stale_base_epochs += 1;
            continue;
        }
        let base = latest_base
            .as_ref()
            .ok_or_else(|| anyhow!("causal base join lost its latest selected observation"))?;

        let Some(epoch) = build_paired_epoch(
            ephemeris,
            base.time_j2000_s,
            rover.time_j2000_s,
            &base.observations,
            &rover_observations,
            velocity_mps,
        )?
        else {
            stats.unusable_epochs += 1;
            continue;
        };
        extend_scales(
            &epoch,
            &mut wavelengths_m,
            &mut offsets_m,
            &base.observations,
        )?;
        epochs.push(epoch);
    }

    if stats.rover_epochs_scanned == 0 {
        bail!("rover RINEX has no observation epochs");
    }
    if !base_join.saw_base_epoch {
        bail!("base RINEX has no observation epochs");
    }
    if epochs.is_empty() {
        bail!("causal PPC builder produced no usable epochs");
    }
    stats.arc_epochs = epochs.len();
    Ok(CausalArc {
        epochs,
        wavelengths_m,
        offsets_m,
        stats,
    })
}

fn epoch_j2000_s(epoch: &ObsEpoch) -> f64 {
    let epoch = epoch.epoch;
    j2000_seconds(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    )
}

fn select_observations(
    obs: &RinexObs,
    epoch: &ObsEpoch,
    selector: &SignalSelector,
) -> Result<BTreeMap<String, SelectedObservation>> {
    let mut selected = BTreeMap::new();
    for (satellite_id, rows) in observation_values(obs, epoch, &selector.filter)
        .context("read RINEX observation values for causal PPC epoch")?
    {
        let Some(pairs) = selector.by_system.get(&satellite_id.system) else {
            continue;
        };
        let rows = rows
            .iter()
            .map(|row| (row.code.as_str(), row))
            .collect::<BTreeMap<_, _>>();
        for pair in pairs {
            let Some(code_m) = rows.get(pair.code).and_then(|row| row.value) else {
                continue;
            };
            let Some(phase_cycles) = rows.get(pair.phase).and_then(|row| row.value) else {
                continue;
            };
            let glonass_channel = (satellite_id.system == GnssSystem::Glonass)
                .then(|| obs.header().glonass_slots.get(&satellite_id.prn).copied())
                .flatten();
            let frequency_hz = observation_frequency_hz(
                satellite_id.system,
                pair.phase,
                obs.header().version,
                glonass_channel,
            )
            .context("resolve PPC carrier frequency")?
            .ok_or_else(|| {
                anyhow!(
                    "no carrier frequency for {satellite_id} observable {}",
                    pair.phase
                )
            })?;
            let wavelength_m = C_M_S / frequency_hz;
            let doppler_hz = rows
                .get(doppler_code(pair.phase).as_str())
                .and_then(|row| row.value);
            let satellite_token = satellite_id.to_string();
            selected.insert(
                satellite_token.clone(),
                SelectedObservation {
                    satellite_id,
                    ambiguity_id: satellite_token,
                    phase_code: pair.phase,
                    code_m,
                    phase_m: phase_cycles * wavelength_m,
                    wavelength_m,
                    lli: rows.get(pair.phase).and_then(|row| row.lli).map(i64::from),
                    doppler_hz,
                },
            );
            break;
        }
    }
    Ok(selected)
}

fn solve_rover_velocity(
    ephemeris: &dyn ObservableEphemerisSource,
    observations: &BTreeMap<String, SelectedObservation>,
    receiver_ecef_m: [f64; 3],
    time_j2000_s: f64,
) -> Option<[f64; 3]> {
    let observations = observations
        .values()
        .filter_map(|observation| {
            observation
                .doppler_hz
                .map(|doppler_hz| VelocityObservation {
                    satellite_id: observation.satellite_id,
                    value: doppler_hz,
                    carrier_hz: C_M_S / observation.wavelength_m,
                    sat_clock_drift_s_s: 0.0,
                })
        })
        .collect::<Vec<_>>();
    velocity::solve(
        ephemeris,
        &observations,
        receiver_ecef_m,
        time_j2000_s,
        VelocitySolveOptions {
            observable: VelocityObservable::Doppler,
            ..VelocitySolveOptions::default()
        },
    )
    .ok()
    .map(|solution| solution.velocity_m_s)
}

fn build_paired_epoch(
    ephemeris: &dyn ObservableEphemerisSource,
    base_time_j2000_s: f64,
    rover_time_j2000_s: f64,
    base: &BTreeMap<String, SelectedObservation>,
    rover: &BTreeMap<String, SelectedObservation>,
    velocity_mps: Option<[f64; 3]>,
) -> Result<Option<RtkArcEpoch>> {
    let common = base
        .keys()
        .filter(|satellite_id| rover.contains_key(*satellite_id))
        .cloned()
        .collect::<Vec<_>>();
    let mut base_rows = Vec::new();
    let mut rover_rows = Vec::new();
    let mut satellite_positions_m = BTreeMap::new();
    let mut base_satellite_positions_m = BTreeMap::new();
    let mut rover_satellite_positions_m = BTreeMap::new();

    for satellite_id in common {
        let base_observation = &base[&satellite_id];
        let rover_observation = &rover[&satellite_id];
        if (base_observation.wavelength_m - rover_observation.wavelength_m).abs() > 1.0e-12 {
            bail!("base and rover selected different carrier bands for {satellite_id}");
        }
        let Some(shared_position) =
            ephemeris_position(ephemeris, base_observation.satellite_id, rover_time_j2000_s)?
        else {
            continue;
        };
        let base_tx_time = transmit_time(base_time_j2000_s, base_observation.code_m);
        let rover_tx_time = transmit_time(rover_time_j2000_s, rover_observation.code_m);
        let Some(base_tx_position) =
            ephemeris_position(ephemeris, base_observation.satellite_id, base_tx_time)?
        else {
            continue;
        };
        let Some(rover_tx_position) =
            ephemeris_position(ephemeris, rover_observation.satellite_id, rover_tx_time)?
        else {
            continue;
        };

        base_rows.push(to_arc_observation(base_observation));
        rover_rows.push(to_arc_observation(rover_observation));
        satellite_positions_m.insert(satellite_id.clone(), shared_position);
        base_satellite_positions_m.insert(satellite_id.clone(), base_tx_position);
        rover_satellite_positions_m.insert(satellite_id, rover_tx_position);
    }
    if base_rows.len() < MINIMUM_SATELLITES {
        return Ok(None);
    }
    Ok(Some(RtkArcEpoch {
        base: base_rows,
        rover: rover_rows,
        satellite_positions_m,
        base_satellite_positions_m,
        rover_satellite_positions_m,
        velocity_mps,
        prediction_time_s: Some(rover_time_j2000_s),
    }))
}

fn ephemeris_position(
    ephemeris: &dyn ObservableEphemerisSource,
    satellite_id: GnssSatelliteId,
    time_j2000_s: f64,
) -> Result<Option<[f64; 3]>> {
    match ephemeris.observable_state_at_j2000_s(satellite_id, time_j2000_s) {
        Ok(state) => Ok(Some(state.position_ecef_m)),
        Err(error) if is_observable_state_gap(&error) => Ok(None),
        Err(error) => Err(ephemeris_error(satellite_id, time_j2000_s, error)),
    }
}

fn ephemeris_error(
    satellite_id: GnssSatelliteId,
    time_j2000_s: f64,
    error: ObservablesError,
) -> anyhow::Error {
    anyhow!("ephemeris lookup for {satellite_id} at {time_j2000_s} s failed: {error}")
}

fn transmit_time(receive_time_j2000_s: f64, code_m: f64) -> f64 {
    let transmit_offset_us = (code_m / C_M_S * 1_000_000.0).round();
    receive_time_j2000_s - transmit_offset_us / 1_000_000.0
}

fn to_arc_observation(observation: &SelectedObservation) -> RtkArcObservation {
    RtkArcObservation {
        satellite_id: observation.satellite_id.to_string(),
        ambiguity_id: observation.ambiguity_id.clone(),
        code_m: observation.code_m,
        phase_m: observation.phase_m,
        lli: observation.lli,
    }
}

fn extend_scales(
    epoch: &RtkArcEpoch,
    wavelengths_m: &mut BTreeMap<String, f64>,
    offsets_m: &mut BTreeMap<String, f64>,
    selected_base: &BTreeMap<String, SelectedObservation>,
) -> Result<()> {
    let rover_by_sat = epoch
        .rover
        .iter()
        .map(|observation| (observation.satellite_id.as_str(), observation))
        .collect::<BTreeMap<_, _>>();
    for base_observation in &epoch.base {
        let rover_observation = &rover_by_sat[base_observation.satellite_id.as_str()];
        let sd_id = sd_ambiguity_token(
            &base_observation.satellite_id,
            &base_observation.ambiguity_id,
            &rover_observation.ambiguity_id,
        );
        let wavelength_m = selected_base
            .get(&base_observation.satellite_id)
            .ok_or_else(|| anyhow!("missing selected base observation for scale"))?
            .wavelength_m;
        if let Some(previous) = wavelengths_m.get(&sd_id) {
            if (previous - wavelength_m).abs() > 1.0e-12 {
                bail!(
                    "causal PPC ambiguity {sd_id} changed carrier wavelength from {previous} to {wavelength_m} m"
                );
            }
        } else {
            wavelengths_m.insert(sd_id.clone(), wavelength_m);
            offsets_m.insert(sd_id, 0.0);
        }
    }
    Ok(())
}

fn solve_causal_arc(
    base_m: [f64; 3],
    mut arc: CausalArc,
    options: PpcRunnerOptions,
) -> Result<PpcRunnerResult> {
    let model = MeasModel {
        code_sigma_m: 0.3,
        phase_sigma_m: 0.003,
        sagnac: true,
        stochastic: StochasticModel::Rtklib,
    };
    let update_options = UpdateOpts {
        hold_sigma_m: options.hold_sigma_m,
        position_tol_m: 1.0e-4,
        ambiguity_tol_m: 1.0e-4,
        max_iterations: 10,
        process_noise_baseline_sigma_m: options.process_noise_sigma_m,
        dynamics_model: if options.velocity_dynamics {
            DynamicsModel::VelocityPropagated
        } else {
            DynamicsModel::ConstantPosition
        },
        float_only_systems: vec!["R".to_string()],
        report_residuals: false,
        receiver_antenna_corrections: None,
        ar_arming_sigma_m: None,
        search: SearchOpts {
            ratio_threshold: 3.0,
        },
    };
    let mut state = FilterState::new(BTreeMap::new(), [0.0; 3], 100.0, 1000.0)
        .context("initialize causal PPC filter")?;
    let mut scratch = RtkFilterScratch::new();
    let mut float_fallback_scratch = RtkFilterScratch::new();
    let mut ambiguity_last_seen_s = BTreeMap::<String, f64>::new();
    let mut previous_time = None::<f64>;
    let mut solutions = Vec::with_capacity(arc.epochs.len());

    for (epoch_index, raw_epoch) in arc.epochs.iter().enumerate() {
        let time_j2000_s = raw_epoch
            .prediction_time_s
            .ok_or_else(|| anyhow!("causal PPC epoch has no prediction timestamp"))?;
        let Some(prepared) =
            prepare_filter_epoch(raw_epoch, base_m, options.elevation_mask_deg, previous_time)?
        else {
            arc.stats.coasted_epochs += 1;
            if !state.fixed_cycles.is_empty() {
                arc.stats.fixed_epochs += 1;
            }
            solutions.push(checked_solution_epoch(
                epoch_index,
                time_j2000_s,
                base_m,
                state.baseline_m,
                !state.fixed_cycles.is_empty(),
                0,
                "coasted",
            )?);
            continue;
        };

        if state.references != prepared.reference_sd_ids {
            state.fixed_cycles.clear();
            state.fixed_m.clear();
            state.references = prepared.reference_sd_ids;
        }
        // The historical causal driver advances its prediction clock once an
        // epoch has enough reference geometry, before the update attempt. A
        // singular epoch therefore coasts the state but still bounds the next
        // Doppler propagation interval at this timestamp.
        previous_time = Some(time_j2000_s);
        match update_causal_epoch(
            state.clone(),
            &prepared.epoch,
            base_m,
            &model,
            &arc.wavelengths_m,
            &arc.offsets_m,
            &update_options,
            &mut scratch,
            &mut float_fallback_scratch,
        ) {
            Ok((update, used_float_fallback)) => {
                let solution = checked_update_solution(
                    epoch_index,
                    time_j2000_s,
                    base_m,
                    update.state.baseline_m,
                    update.reported_baseline_m,
                    update.integer_fixed,
                    prepared.satellites_used,
                )?;
                if update.integer_fixed {
                    arc.stats.fixed_epochs += 1;
                }
                if used_float_fallback {
                    arc.stats.float_fallback_epochs += 1;
                }
                state = update.state;
                arc.stats.peak_ambiguity_columns = arc
                    .stats
                    .peak_ambiguity_columns
                    .max(state.sd_ambiguity_ids.len());
                let retired = retire_causal_ambiguities(
                    &mut state,
                    &prepared.epoch,
                    time_j2000_s,
                    options.ambiguity_retirement_age_s,
                    options.max_ambiguity_columns,
                    &mut ambiguity_last_seen_s,
                )?;
                arc.stats.retired_ambiguity_columns += retired;
                solutions.push(solution);
            }
            Err(UpdateError::SingularGeometry) => {
                arc.stats.coasted_epochs += 1;
                if !state.fixed_cycles.is_empty() {
                    arc.stats.fixed_epochs += 1;
                }
                solutions.push(checked_solution_epoch(
                    epoch_index,
                    time_j2000_s,
                    base_m,
                    state.baseline_m,
                    !state.fixed_cycles.is_empty(),
                    prepared.satellites_used,
                    "coasted",
                )?);
            }
            Err(error) => {
                return Err(anyhow!(
                    "causal PPC filter update failed at arc epoch {} ({time_j2000_s} J2000 s): {error}",
                    epoch_index + 1
                ));
            }
        }
    }
    Ok(PpcRunnerResult {
        epochs: solutions,
        stats: arc.stats,
    })
}

#[allow(clippy::too_many_arguments)]
fn update_causal_epoch(
    state: FilterState,
    epoch: &Epoch,
    base_m: [f64; 3],
    model: &MeasModel,
    wavelengths_m: &BTreeMap<String, f64>,
    offsets_m: &BTreeMap<String, f64>,
    options: &UpdateOpts,
    scratch: &mut RtkFilterScratch,
    float_fallback_scratch: &mut RtkFilterScratch,
) -> Result<(EpochUpdate, bool), UpdateError> {
    let scale = AmbiguityScale {
        wavelengths_m,
        offsets_m,
    };
    match update_epoch_with_scratch(state.clone(), epoch, base_m, model, scale, options, scratch) {
        Ok(update) => Ok((update, false)),
        Err(UpdateError::Ils(error)) if recoverable_ils_error(&error) => {
            // A bounded/search-conditioning failure is environmental for a
            // sequential route, not malformed filter input. Preserve the float
            // measurement update and all existing holds, but do not attempt a
            // new fix for this epoch. The ordinary streaming kernel and its
            // defaults remain unchanged.
            let mut float_options = options.clone();
            float_options.float_only_systems = epoch
                .references
                .iter()
                .chain(&epoch.nonref)
                .map(|measurement| constellation_letter(&measurement.sat).to_string())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();
            update_epoch_with_scratch(
                state,
                epoch,
                base_m,
                model,
                AmbiguityScale {
                    wavelengths_m,
                    offsets_m,
                },
                &float_options,
                float_fallback_scratch,
            )
            .map(|update| (update, true))
        }
        Err(error) => Err(error),
    }
}

fn recoverable_ils_error(error: &IlsError) -> bool {
    matches!(
        error,
        IlsError::Singular
            | IlsError::NoCandidates(_)
            | IlsError::TooManyCandidates { .. }
            | IlsError::SearchLimitExceeded
    )
}

fn retire_causal_ambiguities(
    state: &mut FilterState,
    epoch: &Epoch,
    now_s: f64,
    retirement_age_s: f64,
    max_columns: usize,
    last_seen_s: &mut BTreeMap<String, f64>,
) -> Result<usize> {
    let observed = epoch
        .references
        .iter()
        .chain(&epoch.nonref)
        .map(|measurement| measurement.sd_ambiguity_id.clone())
        .collect::<BTreeSet<_>>();
    for id in &observed {
        last_seen_s.insert(id.clone(), now_s);
    }
    let protected = state.references.values().collect::<BTreeSet<_>>();
    let mut candidates = state
        .sd_ambiguity_ids
        .iter()
        .filter(|id| !observed.contains(*id) && !protected.contains(id))
        .map(|id| {
            let last_seen = last_seen_s.get(id).copied().unwrap_or(now_s);
            (id.clone(), last_seen, now_s - last_seen > retirement_age_s)
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then_with(|| a.0.cmp(&b.0)));
    let aged_count = candidates.iter().filter(|candidate| candidate.2).count();
    let excess = state.sd_ambiguity_ids.len().saturating_sub(max_columns);
    let retire_count = aged_count.max(excess).min(candidates.len());
    let retired = candidates
        .into_iter()
        .take(retire_count)
        .map(|(id, _, _)| id)
        .collect::<Vec<_>>();
    if retired.is_empty() {
        return Ok(0);
    }
    marginalize_ambiguities(state, &retired)?;
    for id in &retired {
        last_seen_s.remove(id);
    }
    Ok(retired.len())
}

fn marginalize_ambiguities(state: &mut FilterState, ids: &[String]) -> Result<()> {
    let retired = ids.iter().collect::<BTreeSet<_>>();
    if state
        .references
        .values()
        .any(|reference| retired.contains(reference))
    {
        bail!("causal PPC ambiguity retirement attempted to remove an active reference");
    }
    let remove_positions = state
        .sd_ambiguity_ids
        .iter()
        .enumerate()
        .filter_map(|(position, id)| retired.contains(id).then_some(position))
        .collect::<BTreeSet<_>>();
    if remove_positions.is_empty() {
        return Ok(());
    }

    let old_n = 3 + state.sd_ambiguity_ids.len();
    let expected_len = old_n
        .checked_mul(old_n)
        .ok_or_else(|| anyhow!("causal PPC filter dimension overflow during retirement"))?;
    if state.information.len() != expected_len
        || state.sd_ambiguities_m.len() != state.sd_ambiguity_ids.len()
    {
        bail!("causal PPC filter state has inconsistent ambiguity dimensions");
    }
    let remove = remove_positions
        .iter()
        .map(|position| position + 3)
        .collect::<Vec<_>>();
    let remove_indices = remove.iter().copied().collect::<BTreeSet<_>>();
    let kept = (0..old_n)
        .filter(|index| !remove_indices.contains(index))
        .collect::<Vec<_>>();
    let removed_information = remove
        .iter()
        .map(|&row| {
            remove
                .iter()
                .map(|&column| state.information[row * old_n + column])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let inverse = invert_matrix_first_tie(&removed_information)
        .ok_or_else(|| anyhow!("causal PPC retired ambiguity information block is singular"))?;

    let kept_n = kept.len();
    let mut marginalized = vec![0.0; kept_n * kept_n];
    for (a, &row) in kept.iter().enumerate() {
        for (b, &column) in kept.iter().enumerate().skip(a) {
            let mut value = state.information[row * old_n + column];
            for (r, &removed_row) in remove.iter().enumerate() {
                for (s, &removed_column) in remove.iter().enumerate() {
                    value -= state.information[row * old_n + removed_row]
                        * inverse[r][s]
                        * state.information[removed_column * old_n + column];
                }
            }
            if !value.is_finite() {
                bail!("causal PPC ambiguity retirement produced non-finite information");
            }
            marginalized[a * kept_n + b] = value;
            marginalized[b * kept_n + a] = value;
        }
    }

    state.sd_ambiguity_ids = state
        .sd_ambiguity_ids
        .iter()
        .enumerate()
        .filter(|(position, _)| !remove_positions.contains(position))
        .map(|(_, id)| id.clone())
        .collect();
    state.sd_ambiguities_m = state
        .sd_ambiguities_m
        .iter()
        .enumerate()
        .filter(|(position, _)| !remove_positions.contains(position))
        .map(|(_, value)| *value)
        .collect();
    state.information = marginalized;
    state.fixed_cycles.retain(|id, _| !retired.contains(id));
    state.fixed_m.retain(|id, _| !retired.contains(id));
    Ok(())
}

struct PreparedFilterEpoch {
    epoch: Epoch,
    reference_sd_ids: BTreeMap<String, String>,
    satellites_used: usize,
}

fn prepare_filter_epoch(
    raw: &RtkArcEpoch,
    base_m: [f64; 3],
    elevation_mask_deg: f64,
    previous_time: Option<f64>,
) -> Result<Option<PreparedFilterEpoch>> {
    let base_by_sat = raw
        .base
        .iter()
        .map(|observation| (observation.satellite_id.as_str(), observation))
        .collect::<BTreeMap<_, _>>();
    let rover_by_sat = raw
        .rover
        .iter()
        .map(|observation| (observation.satellite_id.as_str(), observation))
        .collect::<BTreeMap<_, _>>();
    let mask = apply_elevation_mask(
        base_m,
        &[ElevationMaskEpoch {
            satellite_positions_m: raw.satellite_positions_m.clone(),
        }],
        elevation_mask_deg,
    )
    .map_err(|error| anyhow!("apply PPC elevation mask: {error:?}"))?;
    let kept = mask.epochs[0]
        .kept_satellite_ids
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut by_system = BTreeMap::<String, Vec<String>>::new();
    for satellite_id in kept {
        if base_by_sat.contains_key(satellite_id)
            && rover_by_sat.contains_key(satellite_id)
            && raw.base_satellite_positions_m.contains_key(satellite_id)
            && raw.rover_satellite_positions_m.contains_key(satellite_id)
        {
            by_system
                .entry(constellation_letter(satellite_id).to_string())
                .or_default()
                .push(satellite_id.to_string());
        }
    }
    let eligible = by_system
        .values()
        .filter(|satellites| satellites.len() >= 2)
        .flatten()
        .cloned()
        .collect::<BTreeSet<_>>();
    if eligible.len() < MINIMUM_SATELLITES {
        return Ok(None);
    }
    let reference_epoch = BaselineReferenceEpoch {
        available_satellite_ids: eligible.iter().cloned().collect(),
        satellite_positions_m: raw
            .satellite_positions_m
            .iter()
            .filter(|(satellite_id, _)| eligible.contains(*satellite_id))
            .map(|(satellite_id, position)| (satellite_id.clone(), *position))
            .collect(),
    };
    let references_by_system =
        baseline_reference_satellites(base_m, &[reference_epoch], BaselineReferenceSelection::Auto)
            .map_err(|error| anyhow!("select causal PPC reference satellites: {error:?}"))?;
    let reference_sats = references_by_system
        .values()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut reference_sd_ids = BTreeMap::new();
    let mut references = Vec::new();
    for (system, satellite_id) in &references_by_system {
        let measurement = sat_measurement(raw, satellite_id, &base_by_sat, &rover_by_sat)?;
        reference_sd_ids.insert(system.clone(), measurement.sd_ambiguity_id.clone());
        references.push(measurement);
    }
    let mut nonref = Vec::new();
    for satellite_id in &eligible {
        if !reference_sats.contains(satellite_id.as_str()) {
            nonref.push(sat_measurement(
                raw,
                satellite_id,
                &base_by_sat,
                &rover_by_sat,
            )?);
        }
    }
    let current_time = raw
        .prediction_time_s
        .ok_or_else(|| anyhow!("causal PPC epoch has no prediction timestamp"))?;
    let dt_s = previous_time.map_or(0.0, |previous| current_time - previous);
    Ok(Some(PreparedFilterEpoch {
        epoch: Epoch {
            references,
            nonref,
            velocity_mps: raw.velocity_mps,
            dt_s,
        },
        reference_sd_ids,
        satellites_used: eligible.len(),
    }))
}

fn sat_measurement(
    raw: &RtkArcEpoch,
    satellite_id: &str,
    base_by_sat: &BTreeMap<&str, &RtkArcObservation>,
    rover_by_sat: &BTreeMap<&str, &RtkArcObservation>,
) -> Result<SatMeas> {
    let base = base_by_sat[satellite_id];
    let rover = rover_by_sat[satellite_id];
    Ok(SatMeas {
        sat: satellite_id.to_string(),
        sd_ambiguity_id: sd_ambiguity_token(satellite_id, &base.ambiguity_id, &rover.ambiguity_id),
        base_code_m: base.code_m,
        base_phase_m: base.phase_m,
        rover_code_m: rover.code_m,
        rover_phase_m: rover.phase_m,
        base_tx_pos: raw.base_satellite_positions_m[satellite_id],
        rover_tx_pos: raw.rover_satellite_positions_m[satellite_id],
        pos: raw.satellite_positions_m[satellite_id],
    })
}

fn checked_solution_epoch(
    epoch_index: usize,
    time_j2000_s: f64,
    base_m: [f64; 3],
    baseline_m: [f64; 3],
    integer_fixed: bool,
    satellites_used: usize,
    source: &str,
) -> Result<PpcSolutionEpoch> {
    let position_ecef_m = add3(base_m, baseline_m);
    validate_solution_position(epoch_index, time_j2000_s, position_ecef_m, source)?;
    Ok(PpcSolutionEpoch {
        time_j2000_s,
        position_ecef_m,
        integer_fixed,
        satellites_used,
    })
}

fn checked_update_solution(
    epoch_index: usize,
    time_j2000_s: f64,
    base_m: [f64; 3],
    carried_baseline_m: [f64; 3],
    reported_baseline_m: [f64; 3],
    integer_fixed: bool,
    satellites_used: usize,
) -> Result<PpcSolutionEpoch> {
    validate_solution_position(
        epoch_index,
        time_j2000_s,
        add3(base_m, carried_baseline_m),
        "carried float",
    )?;
    checked_solution_epoch(
        epoch_index,
        time_j2000_s,
        base_m,
        reported_baseline_m,
        integer_fixed,
        satellites_used,
        "reported",
    )
}

fn validate_solution_position(
    epoch_index: usize,
    time_j2000_s: f64,
    position_ecef_m: [f64; 3],
    source: &str,
) -> Result<()> {
    let continuous_gps_s = time_j2000_s + sidereon_core::constants::GPS_EPOCH_TO_J2000_S;
    let tow_s = continuous_gps_s.rem_euclid(sidereon_core::constants::SECONDS_PER_WEEK);
    validate_earth_fixed_position(
        position_ecef_m,
        &format!(
            "causal PPC {source} receiver position at arc epoch {} ({time_j2000_s:.3} J2000 s, GPS TOW {tow_s:.3} s)",
            epoch_index + 1
        ),
    )
}

fn validate_earth_fixed_position(position_ecef_m: [f64; 3], context: &str) -> Result<()> {
    if !position_ecef_m
        .iter()
        .all(|coordinate| coordinate.is_finite())
    {
        bail!(
            "{context} must have finite ECEF coordinates within [{EARTH_FIXED_RADIUS_MIN_M}, {EARTH_FIXED_RADIUS_MAX_M}] m radius"
        );
    }
    let radius_m = (position_ecef_m[0] * position_ecef_m[0]
        + position_ecef_m[1] * position_ecef_m[1]
        + position_ecef_m[2] * position_ecef_m[2])
        .sqrt();
    if !(EARTH_FIXED_RADIUS_MIN_M..=EARTH_FIXED_RADIUS_MAX_M).contains(&radius_m) {
        bail!(
            "{context} has nonphysical ECEF radius {radius_m:.3} m; expected [{EARTH_FIXED_RADIUS_MIN_M}, {EARTH_FIXED_RADIUS_MAX_M}] m"
        );
    }
    Ok(())
}

fn add3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn constellation_letter(satellite_id: &str) -> &str {
    satellite_id.get(..1).unwrap_or(satellite_id)
}

fn sd_ambiguity_token(satellite_id: &str, base_id: &str, rover_id: &str) -> String {
    if base_id == satellite_id && rover_id == satellite_id {
        satellite_id.to_string()
    } else if base_id == satellite_id {
        rover_id.to_string()
    } else if rover_id == satellite_id {
        base_id.to_string()
    } else {
        format!("{satellite_id}:base={base_id},rover={rover_id}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sidereon_core::observables::ObservableState;

    struct TimeCodedEphemeris;

    impl ObservableEphemerisSource for TimeCodedEphemeris {
        fn observable_state_at_j2000_s(
            &self,
            satellite_id: GnssSatelliteId,
            time_j2000_s: f64,
        ) -> std::result::Result<ObservableState, ObservablesError> {
            Ok(ObservableState {
                position_ecef_m: [time_j2000_s, f64::from(satellite_id.prn), 20_000_000.0],
                clock_s: None,
            })
        }
    }

    fn observation(prn: u8, code_m: f64) -> SelectedObservation {
        let satellite_id = GnssSatelliteId::new(GnssSystem::Gps, prn).expect("satellite id");
        SelectedObservation {
            satellite_id,
            ambiguity_id: satellite_id.to_string(),
            phase_code: "L1C",
            code_m,
            phase_m: code_m + 1.0,
            wavelength_m: 0.190_293_672_798,
            lli: None,
            doppler_hz: None,
        }
    }

    #[test]
    fn paired_geometry_uses_native_base_and_rover_times() {
        let code_m = C_M_S * 0.072_345_6;
        let base = (1..=4)
            .map(|prn| (format!("G{prn:02}"), observation(prn, code_m)))
            .collect();
        let rover = (1..=4)
            .map(|prn| (format!("G{prn:02}"), observation(prn, code_m)))
            .collect();
        let epoch = build_paired_epoch(&TimeCodedEphemeris, 100.0, 100.8, &base, &rover, None)
            .expect("build epoch")
            .expect("usable epoch");
        assert_eq!(epoch.satellite_positions_m["G01"][0], 100.8);
        assert_eq!(epoch.base_satellite_positions_m["G01"][0], 99.927_654);
        assert_eq!(epoch.rover_satellite_positions_m["G01"][0], 100.727_654);
    }

    #[test]
    fn tracker_applies_reused_native_lli_only_once() {
        let mut tracker = ArcTracker::default();
        let mut first = BTreeMap::from([("G01".to_string(), observation(1, 20_000_000.0))]);
        tracker.segment(&mut first, false);
        assert_eq!(first["G01"].ambiguity_id, "G01");

        let mut slipped = first.clone();
        slipped.get_mut("G01").expect("observation").lli = Some(1);
        tracker.segment(&mut slipped, false);
        assert_eq!(slipped["G01"].ambiguity_id, "G01#2");

        // A reused base row is cloned from the already-segmented native row;
        // the tracker is deliberately not invoked by a rover association.
        let reused = slipped.clone();
        assert_eq!(reused["G01"].ambiguity_id, "G01#2");
    }

    #[test]
    fn tracker_splits_reacquired_observation() {
        let mut tracker = ArcTracker::default();
        let mut present = BTreeMap::from([("G01".to_string(), observation(1, 20_000_000.0))]);
        tracker.segment(&mut present, false);
        tracker.segment(&mut BTreeMap::new(), false);
        let mut reacquired = present;
        tracker.segment(&mut reacquired, false);
        assert_eq!(reacquired["G01"].ambiguity_id, "G01#2");
    }

    #[test]
    fn tracker_splits_a_receiver_power_failure_once_per_native_epoch() {
        let mut tracker = ArcTracker::default();
        let mut observation = BTreeMap::from([("G01".to_string(), observation(1, 20_000_000.0))]);
        tracker.segment(&mut observation, false);
        tracker.segment(&mut observation, true);
        assert_eq!(observation["G01"].ambiguity_id, "G01#2");

        // Reusing the already segmented native base row at rover rate does not
        // invoke the tracker again. The next ordinary native epoch keeps the
        // same generation unless another lifecycle event occurs.
        let reused = observation.clone();
        assert_eq!(reused["G01"].ambiguity_id, "G01#2");
        tracker.segment(&mut observation, false);
        assert_eq!(observation["G01"].ambiguity_id, "G01#2");
    }

    #[test]
    fn tracker_splits_a_signal_fallback_transition() {
        let mut tracker = ArcTracker::default();
        let mut first = BTreeMap::from([("G01".to_string(), observation(1, 20_000_000.0))]);
        tracker.segment(&mut first, false);

        let mut changed = first;
        changed.get_mut("G01").expect("observation").phase_code = "L1X";
        tracker.segment(&mut changed, false);
        assert_eq!(changed["G01"].ambiguity_id, "G01#2");
    }

    #[test]
    fn sd_token_preserves_receiver_specific_segments() {
        assert_eq!(sd_ambiguity_token("G01", "G01", "G01"), "G01");
        assert_eq!(sd_ambiguity_token("G01", "G01#2", "G01"), "G01#2");
        assert_eq!(sd_ambiguity_token("G01", "G01", "G01#3"), "G01#3");
        assert_eq!(
            sd_ambiguity_token("G01", "G01#2", "G01#2"),
            "G01:base=G01#2,rover=G01#2"
        );
        assert_eq!(
            sd_ambiguity_token("G01", "G01#2", "G01#3"),
            "G01:base=G01#2,rover=G01#3"
        );
    }

    #[test]
    fn staggered_receiver_slips_cannot_reuse_an_sd_ambiguity_id() {
        let mut base_tracker = ArcTracker::default();
        let mut rover_tracker = ArcTracker::default();
        let mut base = BTreeMap::from([("G01".to_string(), observation(1, 20_000_000.0))]);
        let mut rover = base.clone();
        base_tracker.segment(&mut base, false);
        rover_tracker.segment(&mut rover, false);
        let original =
            sd_ambiguity_token("G01", &base["G01"].ambiguity_id, &rover["G01"].ambiguity_id);

        base.get_mut("G01").expect("base observation").lli = Some(1);
        base_tracker.segment(&mut base, false);
        base.get_mut("G01").expect("base observation").lli = None;
        let base_slipped =
            sd_ambiguity_token("G01", &base["G01"].ambiguity_id, &rover["G01"].ambiguity_id);

        rover.get_mut("G01").expect("rover observation").lli = Some(1);
        rover_tracker.segment(&mut rover, false);
        let both_slipped =
            sd_ambiguity_token("G01", &base["G01"].ambiguity_id, &rover["G01"].ambiguity_id);
        assert_ne!(original, base_slipped);
        assert_ne!(base_slipped, both_slipped);
    }

    #[test]
    fn exact_age_is_accepted_and_future_base_is_not() {
        let base_times = [10.0, 11.0, 11.0, 13.0];
        let rover_times = [9.9, 10.0, 11.4, 12.2, 13.0];
        let matches = joined_base_indices(&base_times, &rover_times, 1.2).expect("causal join");
        assert_eq!(matches, vec![None, Some(0), Some(2), Some(2), Some(3)]);
    }

    #[test]
    fn stale_gap_recovers_and_suffix_cannot_change_prefix() {
        let base_times = [10.0, 13.0, 20.0];
        let prefix = [10.1, 11.2, 11.3, 13.0];
        let mut with_future = prefix.to_vec();
        with_future.extend([19.0, 20.0]);
        let prefix_matches = joined_base_indices(&base_times, &prefix, 1.2).expect("prefix join");
        let full_matches = joined_base_indices(&base_times, &with_future, 1.2).expect("full join");
        assert_eq!(prefix_matches, vec![Some(0), Some(0), None, Some(1)]);
        assert_eq!(prefix_matches, full_matches[..prefix.len()]);
    }

    #[test]
    fn causal_join_rejects_out_of_order_inputs() {
        assert!(joined_base_indices(&[2.0, 1.0], &[2.0], 1.2).is_err());
        assert!(joined_base_indices(&[1.0], &[2.0, 1.0], 1.2).is_err());
    }

    #[test]
    fn bounded_prefix_does_not_validate_an_unconsumed_base_suffix() {
        let base_epochs = [10.0, 20.0, 5.0]
            .into_iter()
            .enumerate()
            .map(|(index, time_j2000_s)| TimedEpoch {
                index,
                time_j2000_s,
            })
            .collect::<Vec<_>>();
        let mut join = CausalEpochJoin::from_timed(&base_epochs);
        let prefix = join.advance(10.0, 1.2).expect("bounded prefix join");
        assert_eq!(prefix.fresh_base_index, Some(0));
        assert!(join.advance(20.0, 1.2).is_err());
    }

    #[test]
    fn paired_epoch_order_is_independent_of_insertion_order() {
        let code_m = C_M_S * 0.072_345_6;
        let ascending = (1..=6)
            .map(|prn| (format!("G{prn:02}"), observation(prn, code_m)))
            .collect::<BTreeMap<_, _>>();
        let descending = (1..=6)
            .rev()
            .map(|prn| (format!("G{prn:02}"), observation(prn, code_m)))
            .collect::<BTreeMap<_, _>>();
        let a = build_paired_epoch(
            &TimeCodedEphemeris,
            100.0,
            100.8,
            &ascending,
            &descending,
            None,
        )
        .expect("ascending epoch")
        .expect("usable ascending epoch");
        let b = build_paired_epoch(
            &TimeCodedEphemeris,
            100.0,
            100.8,
            &descending,
            &ascending,
            None,
        )
        .expect("descending epoch")
        .expect("usable descending epoch");
        assert_eq!(a, b);
        assert_eq!(
            a.base
                .iter()
                .map(|observation| observation.satellite_id.as_str())
                .collect::<Vec<_>>(),
            vec!["G01", "G02", "G03", "G04", "G05", "G06"]
        );
    }

    #[test]
    fn causal_filter_solution_prefix_is_suffix_invariant() {
        let (base_m, prefix_arc, options) = synthetic_arc(3);
        let (_, mut full_arc, _) = synthetic_arc(5);
        let initial_reference = prepare_filter_epoch(
            &full_arc.epochs[0],
            base_m,
            options.elevation_mask_deg,
            None,
        )
        .expect("prepare first epoch")
        .expect("first epoch is usable")
        .reference_sd_ids["G"]
            .clone();
        let reference_satellite = initial_reference
            .split(['#', ':'])
            .next()
            .expect("reference satellite token");
        for epoch in full_arc.epochs.iter_mut().skip(3) {
            epoch
                .base
                .retain(|observation| observation.satellite_id != reference_satellite);
            epoch
                .rover
                .retain(|observation| observation.satellite_id != reference_satellite);
            epoch.satellite_positions_m.remove(reference_satellite);
            epoch.base_satellite_positions_m.remove(reference_satellite);
            epoch
                .rover_satellite_positions_m
                .remove(reference_satellite);
            epoch.rover[0].ambiguity_id = format!("{}#2", epoch.rover[0].satellite_id);
        }
        let prefix = solve_causal_arc(base_m, prefix_arc, options).expect("solve prefix");
        let full = solve_causal_arc(base_m, full_arc, options).expect("solve full arc");
        assert_eq!(prefix.epochs, full.epochs[..prefix.epochs.len()]);
    }

    #[test]
    fn earth_fixed_position_guard_matches_project_receiver_contract() {
        for radius_m in [
            EARTH_FIXED_RADIUS_MIN_M,
            6_350_000.0,
            EARTH_FIXED_RADIUS_MAX_M,
        ] {
            validate_earth_fixed_position([radius_m, 0.0, 0.0], "test position")
                .expect("inclusive Earth-fixed radius");
        }

        for position in [
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [EARTH_FIXED_RADIUS_MIN_M - 1.0, 0.0, 0.0],
            [EARTH_FIXED_RADIUS_MAX_M + 1.0, 0.0, 0.0],
            [
                -1_032_333_435_907_834.0,
                -526_167_520_694_624.1,
                -5_219_109_660_661_722.0,
            ],
        ] {
            assert!(validate_earth_fixed_position(position, "test position").is_err());
        }
    }

    #[test]
    fn update_and_coast_positions_fail_closed_with_epoch_context() {
        let base_m = [6_350_000.0, 0.0, 0.0];
        let huge = [1.0e15, 0.0, 0.0];
        let carried = checked_update_solution(6, 123.5, base_m, huge, [0.0; 3], false, 8)
            .expect_err("invalid carried state must fail");
        let carried = carried.to_string();
        assert!(carried.contains("carried float receiver position"));
        assert!(carried.contains("arc epoch 7"));
        assert!(carried.contains("GPS TOW"));

        let reported = checked_update_solution(6, 123.5, base_m, [0.0; 3], huge, true, 8)
            .expect_err("invalid reported state must fail")
            .to_string();
        assert!(reported.contains("reported receiver position"));

        let coasted = checked_solution_epoch(6, 123.5, base_m, huge, false, 0, "coasted")
            .expect_err("invalid coast must fail")
            .to_string();
        assert!(coasted.contains("coasted receiver position"));
    }

    #[test]
    fn ambiguity_retirement_uses_a_symmetric_schur_marginal() {
        let mut state = FilterState::new(
            BTreeMap::from([("G".to_string(), "G01".to_string())]),
            [0.0; 3],
            1.0,
            1.0,
        )
        .expect("filter state");
        state.sd_ambiguity_ids = vec!["G01".to_string(), "G02".to_string()];
        state.sd_ambiguities_m = vec![1.0, 2.0];
        state.information = vec![0.0; 25];
        for index in 0..5 {
            state.information[index * 5 + index] = 1.0;
        }
        state.information[3 * 5 + 3] = 4.0;
        state.information[4 * 5 + 4] = 9.0;
        state.information[3 * 5 + 4] = 3.0;
        state.information[4 * 5 + 3] = 3.0;
        state.fixed_cycles.insert("G02".to_string(), 2);
        state.fixed_m.insert("G02".to_string(), 2.0);

        marginalize_ambiguities(&mut state, &["G02".to_string()]).expect("retire ambiguity");
        assert_eq!(state.sd_ambiguity_ids, vec!["G01"]);
        assert_eq!(state.sd_ambiguities_m, vec![1.0]);
        assert_eq!(state.information.len(), 16);
        assert!((state.information[3 * 4 + 3] - 3.0).abs() < 1.0e-12);
        for row in 0..4 {
            for column in 0..4 {
                assert_eq!(
                    state.information[row * 4 + column],
                    state.information[column * 4 + row]
                );
            }
        }
        assert!(!state.fixed_m.contains_key("G02"));
        assert!(!state.fixed_cycles.contains_key("G02"));
    }

    #[test]
    fn lifecycle_retires_only_unseen_nonreference_fragments() {
        let mut state = FilterState::new(
            BTreeMap::from([("G".to_string(), "G01".to_string())]),
            [0.0; 3],
            1.0,
            1.0,
        )
        .expect("filter state");
        state.sd_ambiguity_ids = vec!["G01".into(), "G02".into(), "G03".into()];
        state.sd_ambiguities_m = vec![1.0, 2.0, 3.0];
        state.information = vec![0.0; 36];
        for index in 0..6 {
            state.information[index * 6 + index] = 1.0;
        }
        state.fixed_cycles.insert("G02".to_string(), 2);
        state.fixed_m.insert("G02".to_string(), 2.0);
        let epoch = Epoch {
            references: vec![test_measurement("G01", "G01")],
            nonref: vec![test_measurement("G03", "G03")],
            velocity_mps: None,
            dt_s: 0.2,
        };
        let mut last_seen = BTreeMap::from([("G02".to_string(), 0.0)]);
        let count = retire_causal_ambiguities(&mut state, &epoch, 20.0, 15.0, 128, &mut last_seen)
            .expect("retire stale fragment");
        assert_eq!(count, 1);
        assert_eq!(state.sd_ambiguity_ids, vec!["G01", "G03"]);
        assert_eq!(state.references["G"], "G01");
        assert_eq!(
            last_seen,
            BTreeMap::from([("G01".to_string(), 20.0), ("G03".to_string(), 20.0)])
        );
    }

    #[test]
    fn float_fallback_accepts_only_environmental_integer_search_failures() {
        for error in [
            IlsError::Singular,
            IlsError::NoCandidates(0),
            IlsError::TooManyCandidates {
                evaluated: 101,
                limit: 100,
            },
            IlsError::SearchLimitExceeded,
        ] {
            assert!(recoverable_ils_error(&error), "{error:?}");
        }

        for error in [
            IlsError::InvalidDimensions { n: 2, rows: 1 },
            IlsError::NonFinite,
            IlsError::InvalidInput {
                field: "ratio_threshold",
                reason: "must be finite",
            },
        ] {
            assert!(!recoverable_ils_error(&error), "{error:?}");
        }
    }

    #[test]
    fn runner_options_validate_every_scientific_boundary() {
        let valid = PpcRunnerOptions {
            max_base_age_s: 1.2,
            max_epochs: Some(1),
            elevation_mask_deg: 10.0,
            hold_sigma_m: 0.15,
            process_noise_sigma_m: 0.0,
            velocity_dynamics: true,
            ambiguity_retirement_age_s: DEFAULT_AMBIGUITY_RETIREMENT_AGE_S,
            max_ambiguity_columns: DEFAULT_MAX_AMBIGUITY_COLUMNS,
        };
        validate_options(valid).expect("valid boundary options");

        let mut cases = Vec::new();
        let mut options = valid;
        options.max_base_age_s = f64::NAN;
        cases.push((options, "--max-base-age-s"));
        options = valid;
        options.max_base_age_s = -1.0;
        cases.push((options, "--max-base-age-s"));
        options = valid;
        options.max_epochs = Some(0);
        cases.push((options, "--max-epochs"));
        options = valid;
        options.elevation_mask_deg = -0.1;
        cases.push((options, "--elevation-mask-deg"));
        options = valid;
        options.elevation_mask_deg = 90.0;
        cases.push((options, "--elevation-mask-deg"));
        options = valid;
        options.elevation_mask_deg = f64::INFINITY;
        cases.push((options, "--elevation-mask-deg"));
        options = valid;
        options.hold_sigma_m = 0.0;
        cases.push((options, "--hold-sigma-m"));
        options = valid;
        options.hold_sigma_m = f64::NAN;
        cases.push((options, "--hold-sigma-m"));
        options = valid;
        options.process_noise_sigma_m = -f64::EPSILON;
        cases.push((options, "--process-noise-sigma-m"));
        options = valid;
        options.process_noise_sigma_m = f64::INFINITY;
        cases.push((options, "--process-noise-sigma-m"));
        options = valid;
        options.ambiguity_retirement_age_s = -f64::EPSILON;
        cases.push((options, "--ambiguity-retirement-age-s"));
        options = valid;
        options.ambiguity_retirement_age_s = f64::NAN;
        cases.push((options, "--ambiguity-retirement-age-s"));
        options = valid;
        options.max_ambiguity_columns = 0;
        cases.push((options, "--max-ambiguity-columns"));

        for (options, expected) in cases {
            let error = validate_options(options)
                .expect_err("invalid option must be rejected")
                .to_string();
            assert!(error.contains(expected), "unexpected error: {error}");
        }
    }

    fn test_measurement(satellite_id: &str, ambiguity_id: &str) -> SatMeas {
        SatMeas {
            sat: satellite_id.to_string(),
            sd_ambiguity_id: ambiguity_id.to_string(),
            base_code_m: 20_000_000.0,
            base_phase_m: 20_000_000.0,
            rover_code_m: 20_000_001.0,
            rover_phase_m: 20_000_001.0,
            base_tx_pos: [20_000_000.0, 0.0, 0.0],
            rover_tx_pos: [20_000_000.0, 0.0, 0.0],
            pos: [20_000_000.0, 0.0, 0.0],
        }
    }

    fn synthetic_arc(epoch_count: usize) -> ([f64; 3], CausalArc, PpcRunnerOptions) {
        let base_m = [3_512_900.0, 780_500.0, 5_248_700.0];
        let baseline_m = [12.0, -7.0, 9.0];
        let rover_m = add3(base_m, baseline_m);
        let positions = [
            (1, [14_350_000.0, 3_190_000.0, 21_440_000.0]),
            (2, [20_000_000.0, 3_000_000.0, 18_000_000.0]),
            (3, [9_000_000.0, 9_000_000.0, 22_000_000.0]),
            (4, [16_000_000.0, -4_000_000.0, 21_000_000.0]),
            (5, [10_000_000.0, -2_000_000.0, 24_000_000.0]),
            (6, [19_000_000.0, 8_000_000.0, 17_000_000.0]),
        ];
        let wavelength_m = 0.190_293_672_798;
        let ambiguity_cycles = [3.0, -2.0, 5.0, 1.0, -4.0, 6.0];
        let mut base = Vec::new();
        let mut rover = Vec::new();
        let mut satellite_positions_m = BTreeMap::new();
        let mut wavelengths_m = BTreeMap::new();
        let mut offsets_m = BTreeMap::new();
        for ((prn, position), ambiguity_cycles) in positions.into_iter().zip(ambiguity_cycles) {
            let satellite_id = format!("G{prn:02}");
            let base_range = distance(position, base_m);
            let rover_range = distance(position, rover_m);
            base.push(RtkArcObservation {
                satellite_id: satellite_id.clone(),
                ambiguity_id: satellite_id.clone(),
                code_m: base_range,
                phase_m: base_range,
                lli: None,
            });
            rover.push(RtkArcObservation {
                satellite_id: satellite_id.clone(),
                ambiguity_id: satellite_id.clone(),
                code_m: rover_range,
                phase_m: rover_range + ambiguity_cycles * wavelength_m,
                lli: None,
            });
            satellite_positions_m.insert(satellite_id.clone(), position);
            wavelengths_m.insert(satellite_id.clone(), wavelength_m);
            offsets_m.insert(satellite_id, 0.0);
        }
        let epochs = (0..epoch_count)
            .map(|index| RtkArcEpoch {
                base: base.clone(),
                rover: rover.clone(),
                satellite_positions_m: satellite_positions_m.clone(),
                base_satellite_positions_m: satellite_positions_m.clone(),
                rover_satellite_positions_m: satellite_positions_m.clone(),
                velocity_mps: None,
                prediction_time_s: Some(index as f64 * 0.2),
            })
            .collect();
        (
            base_m,
            CausalArc {
                epochs,
                wavelengths_m,
                offsets_m,
                stats: PpcRunnerStats {
                    arc_epochs: epoch_count,
                    ..PpcRunnerStats::default()
                },
            },
            PpcRunnerOptions {
                max_base_age_s: 1.2,
                max_epochs: None,
                elevation_mask_deg: 10.0,
                hold_sigma_m: 0.15,
                process_noise_sigma_m: 0.0,
                velocity_dynamics: false,
                ambiguity_retirement_age_s: DEFAULT_AMBIGUITY_RETIREMENT_AGE_S,
                max_ambiguity_columns: DEFAULT_MAX_AMBIGUITY_COLUMNS,
            },
        )
    }

    fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        let dz = a[2] - b[2];
        (dx * dx + dy * dy + dz * dz).sqrt()
    }

    fn joined_base_indices(
        base_times: &[f64],
        rover_times: &[f64],
        max_age_s: f64,
    ) -> Result<Vec<Option<usize>>> {
        let base_epochs = base_times
            .iter()
            .enumerate()
            .map(|(index, &time_j2000_s)| TimedEpoch {
                index,
                time_j2000_s,
            })
            .collect::<Vec<_>>();
        let mut join = CausalEpochJoin::from_timed(&base_epochs);
        let mut out = Vec::new();
        for &rover_time in rover_times {
            out.push(join.advance(rover_time, max_age_s)?.fresh_base_index);
        }
        Ok(out)
    }
}
