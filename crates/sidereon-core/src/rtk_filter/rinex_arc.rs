//! RINEX observation to RTK arc conversion.
//!
//! These builders turn paired base/rover RINEX observation epochs plus an
//! ephemeris source into the raw arc records consumed by the RTK filter drivers.
//! They own parsing, signal selection, shared-epoch matching, transmit-time
//! satellite lookup, and deterministic ordering only. Double-difference
//! reference selection and numeric solving stay in the existing RTK arc drivers.

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::time::{j2000_seconds_from_split, split_julian_date};
use crate::constants::C_M_S;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::observables::{is_observable_state_gap, ObservableEphemerisSource, ObservablesError};
use crate::rinex::observations::{
    observation_frequency_hz, observation_values, ObsEpoch, ObsEpochTime, ObservationFilter,
    ObservationValueRow, RinexObs,
};

use super::{
    RtkArcEpoch, RtkArcObservation, RtkDualFrequencyArcEpoch, RtkDualFrequencyObservation,
    RtkDualFrequencySatelliteObservation,
};

const DEFAULT_MIN_COMMON_SATELLITES: usize = 4;

/// One single-frequency code/carrier pair to extract from RINEX observations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtkRinexSignalPair {
    /// Constellation whose RINEX observations this pair can select. The builder
    /// groups pairs by this [`GnssSystem`] value and skips satellites from other
    /// constellations.
    pub system: GnssSystem,
    /// Full RINEX code observable whose present value supplies the pseudorange in
    /// meters and the satellite transmit-time correction.
    pub code_observable: String,
    /// Full RINEX carrier-phase observable whose present value is read in cycles
    /// and converted to meters using its carrier frequency; its LLI is retained.
    pub phase_observable: String,
}

impl RtkRinexSignalPair {
    /// GPS L1 C/A code and carrier (`C1C` plus `L1C`).
    pub fn gps_l1_c() -> Self {
        Self {
            system: GnssSystem::Gps,
            code_observable: "C1C".to_string(),
            phase_observable: "L1C".to_string(),
        }
    }
}

/// Options for building single-frequency RTK arc records from RINEX.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtkRinexArcOptions {
    /// Signal choices grouped by constellation and tried in vector order. For
    /// each satellite, the first pair with both values present is used; an empty
    /// vector returns [`RtkRinexArcError::NoSignalPairs`].
    pub signal_pairs: Vec<RtkRinexSignalPair>,
    /// Optional cap on base epochs considered, in file order.
    pub max_epochs: Option<usize>,
    /// Minimum common satellites with observations and ephemeris in an epoch.
    pub min_common_satellites: usize,
    /// Whether to fill `prediction_time_s` with seconds since J2000.
    pub include_prediction_time: bool,
}

impl RtkRinexArcOptions {
    /// Defaults for the GPS L1 C/A code and carrier path.
    pub fn gps_l1_c() -> Self {
        Self {
            signal_pairs: vec![RtkRinexSignalPair::gps_l1_c()],
            max_epochs: None,
            min_common_satellites: DEFAULT_MIN_COMMON_SATELLITES,
            include_prediction_time: true,
        }
    }
}

/// Single-frequency arc records plus the ambiguity scale maps needed by the
/// sequential and static RTK arc solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkRinexArc {
    /// Output [`RtkArcEpoch`] records in considered base-RINEX order. Each retains
    /// paired base/rover records only for satellites with receive-time,
    /// base-transmit-time, and rover-transmit-time positions.
    pub epochs: Vec<RtkArcEpoch>,
    /// Carrier wavelength per single-difference ambiguity id, metres.
    pub wavelengths_m: BTreeMap<String, f64>,
    /// Code-to-phase metre offsets per single-difference ambiguity id.
    pub offsets_m: BTreeMap<String, f64>,
    /// Number of considered base epochs omitted because the rover civil-time key
    /// was absent or too few usable satellites remained.
    pub skipped_epoch_count: usize,
}

/// One dual-frequency code/carrier selection for one constellation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtkRinexDualSignalPair {
    /// Constellation whose RINEX observations this pair can select. The builder
    /// groups pairs by this [`GnssSystem`] value and skips satellites from other
    /// constellations.
    pub system: GnssSystem,
    /// Full RINEX code observable supplying the first-frequency pseudorange in
    /// meters when its row has a value.
    pub code1_observable: String,
    /// Full RINEX carrier-phase observable supplying the first-frequency phase in
    /// cycles and its carrier frequency when both are available.
    pub phase1_observable: String,
    /// Full RINEX code observable supplying the second-frequency pseudorange in
    /// meters when its row has a value.
    pub code2_observable: String,
    /// Full RINEX carrier-phase observable supplying the second-frequency phase
    /// in cycles and its carrier frequency when both are available.
    pub phase2_observable: String,
}

impl RtkRinexDualSignalPair {
    /// GPS L1 C/A plus L2 P(Y) style code/carrier (`C1C`, `L1C`, `C2W`, `L2W`).
    pub fn gps_l1_l2_cw() -> Self {
        Self {
            system: GnssSystem::Gps,
            code1_observable: "C1C".to_string(),
            phase1_observable: "L1C".to_string(),
            code2_observable: "C2W".to_string(),
            phase2_observable: "L2W".to_string(),
        }
    }
}

/// Options for building dual-frequency RTK arc records from RINEX.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtkRinexDualArcOptions {
    /// Four-observable choices grouped by constellation and tried in vector order.
    /// For each satellite, the first pair with all four values present is used;
    /// an empty vector returns [`RtkRinexArcError::NoSignalPairs`].
    pub signal_pairs: Vec<RtkRinexDualSignalPair>,
    /// Optional cap on base epochs considered, in file order.
    pub max_epochs: Option<usize>,
    /// Minimum common satellites with observations and ephemeris in an epoch.
    pub min_common_satellites: usize,
    /// Whether to fill `prediction_time_s` with seconds since J2000.
    pub include_prediction_time: bool,
}

impl RtkRinexDualArcOptions {
    /// Defaults for the GPS L1/L2 path used by the real arc fixtures.
    pub fn gps_l1_l2_cw() -> Self {
        Self {
            signal_pairs: vec![RtkRinexDualSignalPair::gps_l1_l2_cw()],
            max_epochs: None,
            min_common_satellites: DEFAULT_MIN_COMMON_SATELLITES,
            include_prediction_time: true,
        }
    }
}

/// Dual-frequency arc records for wide-lane and ionosphere-free RTK paths.
#[derive(Debug, Clone, PartialEq)]
pub struct RtkRinexDualFrequencyArc {
    /// Output [`RtkDualFrequencyArcEpoch`] records in considered base-RINEX order,
    /// including their paired observations and receive/transmit-time satellite
    /// positions.
    pub epochs: Vec<RtkDualFrequencyArcEpoch>,
    /// Number of considered base epochs omitted because the rover civil-time key
    /// was absent or too few usable dual-frequency satellites remained.
    pub skipped_epoch_count: usize,
}

/// Failure while building RTK arc records from RINEX.
#[derive(Debug, Clone, PartialEq)]
pub enum RtkRinexArcError {
    /// An option or satellite identifier failed the builder's input checks.
    InvalidInput {
        /// Name of the rejected input: `min_common_satellites` or `satellite_id`.
        field: &'static str,
        /// Static reason for rejection: `must be positive` or `invalid satellite token`.
        reason: &'static str,
    },
    /// RINEX observation extraction or carrier-frequency lookup failed.
    Observation(crate::Error),
    /// An ephemeris lookup failed for a reason other than an unavailable state
    /// gap; gap results instead make the satellite unavailable for that epoch.
    Ephemeris {
        /// Satellite token used in the failed state lookup.
        satellite_id: String,
        /// Seconds since J2000 passed to the failed state lookup.
        epoch_j2000_s: f64,
        /// Display text from the ephemeris source error.
        reason: String,
    },
    /// No signal pair was supplied in the single- or dual-frequency options.
    NoSignalPairs,
    /// No considered base epoch met the configured usable-satellite threshold.
    NoUsableEpochs,
    /// A selected phase observable has no carrier frequency in its RINEX context.
    MissingFrequency {
        /// Satellite token for the missing carrier frequency.
        satellite_id: String,
        /// Full RINEX phase observable code whose frequency is missing.
        observable_code: String,
    },
}

impl core::fmt::Display for RtkRinexArcError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid RINEX RTK arc input {field}: {reason}")
            }
            Self::Observation(error) => write!(f, "{error}"),
            Self::Ephemeris {
                satellite_id,
                epoch_j2000_s,
                reason,
            } => write!(
                f,
                "RTK arc ephemeris lookup failed for {satellite_id} at {epoch_j2000_s} s: {reason}"
            ),
            Self::NoSignalPairs => write!(f, "RTK RINEX arc requires at least one signal pair"),
            Self::NoUsableEpochs => write!(f, "RTK RINEX arc produced no usable epochs"),
            Self::MissingFrequency {
                satellite_id,
                observable_code,
            } => write!(
                f,
                "RTK RINEX arc has no carrier frequency for {satellite_id} {observable_code}"
            ),
        }
    }
}

impl std::error::Error for RtkRinexArcError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Observation(error) => Some(error),
            _ => None,
        }
    }
}

impl From<crate::Error> for RtkRinexArcError {
    fn from(error: crate::Error) -> Self {
        Self::Observation(error)
    }
}

/// Build single-frequency RTK arc records from parsed RINEX observations.
///
/// Base and rover epochs are matched by exact civil epoch fields. Each output
/// epoch contains only satellites with complete base/rover code and carrier
/// observations and valid receive-time plus transmit-time ephemeris states.
/// Satellites or epochs without coverage are skipped; non-gap ephemeris errors
/// are returned.
pub fn build_rinex_rtk_arc(
    ephemeris: &dyn ObservableEphemerisSource,
    base_obs: &RinexObs,
    rover_obs: &RinexObs,
    options: &RtkRinexArcOptions,
) -> Result<RtkRinexArc, RtkRinexArcError> {
    validate_arc_options(
        options.min_common_satellites,
        options.signal_pairs.is_empty(),
    )?;

    let pair_by_system = single_pairs_by_system(&options.signal_pairs);
    let filter = single_observation_filter(&options.signal_pairs);
    let rover_by_epoch = rover_epoch_index(rover_obs);
    let mut epochs = Vec::new();
    let mut skipped_epoch_count = 0;
    let mut wavelengths_m = BTreeMap::new();

    for base_epoch in base_obs
        .epochs()
        .iter()
        .take(options.max_epochs.unwrap_or(usize::MAX))
    {
        let Some(rover_epoch) = rover_by_epoch.get(&epoch_key(base_epoch.epoch)).copied() else {
            skipped_epoch_count += 1;
            continue;
        };
        let epoch_j2000_s = j2000_seconds(base_epoch.epoch);
        let base_values =
            single_frequency_observations(base_obs, base_epoch, &filter, &pair_by_system)?;
        let rover_values =
            single_frequency_observations(rover_obs, rover_epoch, &filter, &pair_by_system)?;
        let common = common_keys(base_values.keys(), rover_values.keys());

        let mut satellite_positions_m = BTreeMap::new();
        let mut base_satellite_positions_m = BTreeMap::new();
        let mut rover_satellite_positions_m = BTreeMap::new();
        let mut usable = BTreeSet::new();

        for satellite_id in common {
            let sat = parse_satellite_id(&satellite_id)?;
            let Some(position) = ephemeris_position(ephemeris, sat, epoch_j2000_s)? else {
                continue;
            };
            let base_tx_epoch_s =
                transmit_epoch_j2000_s(epoch_j2000_s, base_values[&satellite_id].code_m);
            let rover_tx_epoch_s =
                transmit_epoch_j2000_s(epoch_j2000_s, rover_values[&satellite_id].code_m);
            let Some(base_tx) = ephemeris_position(ephemeris, sat, base_tx_epoch_s)? else {
                continue;
            };
            let Some(rover_tx) = ephemeris_position(ephemeris, sat, rover_tx_epoch_s)? else {
                continue;
            };
            satellite_positions_m.insert(satellite_id.clone(), position);
            base_satellite_positions_m.insert(satellite_id.clone(), base_tx);
            rover_satellite_positions_m.insert(satellite_id.clone(), rover_tx);
            wavelengths_m.insert(
                satellite_id.clone(),
                base_values[&satellite_id].wavelength_m,
            );
            usable.insert(satellite_id);
        }

        if usable.len() < options.min_common_satellites {
            skipped_epoch_count += 1;
            continue;
        }

        epochs.push(RtkArcEpoch {
            base: retain_single_observations(base_values, &usable),
            rover: retain_single_observations(rover_values, &usable),
            satellite_positions_m,
            base_satellite_positions_m,
            rover_satellite_positions_m,
            velocity_mps: None,
            prediction_time_s: options.include_prediction_time.then_some(epoch_j2000_s),
        });
    }

    if epochs.is_empty() {
        return Err(RtkRinexArcError::NoUsableEpochs);
    }
    let offsets_m = wavelengths_m
        .keys()
        .map(|id| (id.clone(), 0.0))
        .collect::<BTreeMap<_, _>>();
    Ok(RtkRinexArc {
        epochs,
        wavelengths_m,
        offsets_m,
        skipped_epoch_count,
    })
}

/// Build dual-frequency RTK arc records from parsed RINEX observations.
pub fn build_dual_frequency_rinex_rtk_arc(
    ephemeris: &dyn ObservableEphemerisSource,
    base_obs: &RinexObs,
    rover_obs: &RinexObs,
    options: &RtkRinexDualArcOptions,
) -> Result<RtkRinexDualFrequencyArc, RtkRinexArcError> {
    validate_arc_options(
        options.min_common_satellites,
        options.signal_pairs.is_empty(),
    )?;

    let pair_by_system = dual_pairs_by_system(&options.signal_pairs);
    let filter = dual_observation_filter(&options.signal_pairs);
    let rover_by_epoch = rover_epoch_index(rover_obs);
    let mut epochs = Vec::new();
    let mut skipped_epoch_count = 0;

    for base_epoch in base_obs
        .epochs()
        .iter()
        .take(options.max_epochs.unwrap_or(usize::MAX))
    {
        let Some(rover_epoch) = rover_by_epoch.get(&epoch_key(base_epoch.epoch)).copied() else {
            skipped_epoch_count += 1;
            continue;
        };
        let epoch_j2000_s = j2000_seconds(base_epoch.epoch);
        let base_values =
            dual_frequency_observations(base_obs, base_epoch, &filter, &pair_by_system)?;
        let rover_values =
            dual_frequency_observations(rover_obs, rover_epoch, &filter, &pair_by_system)?;
        let common = common_keys(base_values.keys(), rover_values.keys());

        let mut satellite_positions_m = BTreeMap::new();
        let mut base_satellite_positions_m = BTreeMap::new();
        let mut rover_satellite_positions_m = BTreeMap::new();
        let mut observations = Vec::new();

        for satellite_id in common {
            let sat = parse_satellite_id(&satellite_id)?;
            let Some(position) = ephemeris_position(ephemeris, sat, epoch_j2000_s)? else {
                continue;
            };
            let base_tx_epoch_s =
                transmit_epoch_j2000_s(epoch_j2000_s, base_values[&satellite_id].p1_m);
            let rover_tx_epoch_s =
                transmit_epoch_j2000_s(epoch_j2000_s, rover_values[&satellite_id].p1_m);
            let Some(base_tx) = ephemeris_position(ephemeris, sat, base_tx_epoch_s)? else {
                continue;
            };
            let Some(rover_tx) = ephemeris_position(ephemeris, sat, rover_tx_epoch_s)? else {
                continue;
            };
            satellite_positions_m.insert(satellite_id.clone(), position);
            base_satellite_positions_m.insert(satellite_id.clone(), base_tx);
            rover_satellite_positions_m.insert(satellite_id.clone(), rover_tx);
            observations.push(RtkDualFrequencySatelliteObservation {
                satellite_id: satellite_id.clone(),
                base: base_values[&satellite_id].clone(),
                rover: rover_values[&satellite_id].clone(),
            });
        }

        if observations.len() < options.min_common_satellites {
            skipped_epoch_count += 1;
            continue;
        }

        let (jd_whole, jd_fraction) = civil_to_julian_split(base_epoch.epoch);
        epochs.push(RtkDualFrequencyArcEpoch {
            jd_whole,
            jd_fraction,
            epoch_sort_key: Some(epoch_sort_key(base_epoch.epoch)),
            gap_time_s: Some(epoch_j2000_s),
            observations,
            satellite_positions_m,
            base_satellite_positions_m,
            rover_satellite_positions_m,
            velocity_mps: None,
            prediction_time_s: options.include_prediction_time.then_some(epoch_j2000_s),
        });
    }

    if epochs.is_empty() {
        return Err(RtkRinexArcError::NoUsableEpochs);
    }
    Ok(RtkRinexDualFrequencyArc {
        epochs,
        skipped_epoch_count,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct SingleObservation {
    code_m: f64,
    phase_m: f64,
    wavelength_m: f64,
    lli: Option<i64>,
}

fn single_frequency_observations(
    obs: &RinexObs,
    epoch: &ObsEpoch,
    filter: &ObservationFilter,
    pair_by_system: &BTreeMap<GnssSystem, Vec<RtkRinexSignalPair>>,
) -> Result<BTreeMap<String, SingleObservation>, RtkRinexArcError> {
    let mut out = BTreeMap::new();
    for (sat, rows) in observation_values(obs, epoch, filter)? {
        let Some(pairs) = pair_by_system.get(&sat.system) else {
            continue;
        };
        let rows_by_code = rows_by_code(rows);
        for pair in pairs {
            let Some(code_m) = row_value(&rows_by_code, &pair.code_observable) else {
                continue;
            };
            let Some(phase_cycles) = row_value(&rows_by_code, &pair.phase_observable) else {
                continue;
            };
            let frequency_hz = carrier_frequency_hz(obs, sat, &pair.phase_observable)?;
            let wavelength_m = C_M_S / frequency_hz;
            out.insert(
                sat.to_string(),
                SingleObservation {
                    code_m,
                    phase_m: phase_cycles * wavelength_m,
                    wavelength_m,
                    lli: rows_by_code
                        .get(&pair.phase_observable)
                        .and_then(|row| row.lli)
                        .map(i64::from),
                },
            );
            break;
        }
    }
    Ok(out)
}

fn dual_frequency_observations(
    obs: &RinexObs,
    epoch: &ObsEpoch,
    filter: &ObservationFilter,
    pair_by_system: &BTreeMap<GnssSystem, Vec<RtkRinexDualSignalPair>>,
) -> Result<BTreeMap<String, RtkDualFrequencyObservation>, RtkRinexArcError> {
    let mut out = BTreeMap::new();
    for (sat, rows) in observation_values(obs, epoch, filter)? {
        let Some(pairs) = pair_by_system.get(&sat.system) else {
            continue;
        };
        let rows_by_code = rows_by_code(rows);
        for pair in pairs {
            let Some(p1_m) = row_value(&rows_by_code, &pair.code1_observable) else {
                continue;
            };
            let Some(p2_m) = row_value(&rows_by_code, &pair.code2_observable) else {
                continue;
            };
            let Some(phi1_cycles) = row_value(&rows_by_code, &pair.phase1_observable) else {
                continue;
            };
            let Some(phi2_cycles) = row_value(&rows_by_code, &pair.phase2_observable) else {
                continue;
            };
            let f1_hz = carrier_frequency_hz(obs, sat, &pair.phase1_observable)?;
            let f2_hz = carrier_frequency_hz(obs, sat, &pair.phase2_observable)?;
            out.insert(
                sat.to_string(),
                RtkDualFrequencyObservation {
                    ambiguity_id: sat.to_string(),
                    p1_m,
                    p2_m,
                    phi1_cycles,
                    phi2_cycles,
                    f1_hz,
                    f2_hz,
                    lli1: rows_by_code
                        .get(&pair.phase1_observable)
                        .and_then(|row| row.lli)
                        .map(i64::from),
                    lli2: rows_by_code
                        .get(&pair.phase2_observable)
                        .and_then(|row| row.lli)
                        .map(i64::from),
                },
            );
            break;
        }
    }
    Ok(out)
}

fn validate_arc_options(
    min_common_satellites: usize,
    signal_pairs_empty: bool,
) -> Result<(), RtkRinexArcError> {
    if signal_pairs_empty {
        return Err(RtkRinexArcError::NoSignalPairs);
    }
    if min_common_satellites == 0 {
        return Err(RtkRinexArcError::InvalidInput {
            field: "min_common_satellites",
            reason: "must be positive",
        });
    }
    Ok(())
}

fn single_pairs_by_system(
    pairs: &[RtkRinexSignalPair],
) -> BTreeMap<GnssSystem, Vec<RtkRinexSignalPair>> {
    let mut out = BTreeMap::<GnssSystem, Vec<RtkRinexSignalPair>>::new();
    for pair in pairs {
        out.entry(pair.system).or_default().push(pair.clone());
    }
    out
}

fn dual_pairs_by_system(
    pairs: &[RtkRinexDualSignalPair],
) -> BTreeMap<GnssSystem, Vec<RtkRinexDualSignalPair>> {
    let mut out = BTreeMap::<GnssSystem, Vec<RtkRinexDualSignalPair>>::new();
    for pair in pairs {
        out.entry(pair.system).or_default().push(pair.clone());
    }
    out
}

fn single_observation_filter(pairs: &[RtkRinexSignalPair]) -> ObservationFilter {
    let mut by_system = BTreeMap::<GnssSystem, BTreeSet<String>>::new();
    for pair in pairs {
        by_system
            .entry(pair.system)
            .or_default()
            .extend([pair.code_observable.clone(), pair.phase_observable.clone()]);
    }
    ObservationFilter::from_entries(
        by_system
            .into_iter()
            .map(|(system, codes)| (system, codes.into_iter().collect())),
    )
}

fn dual_observation_filter(pairs: &[RtkRinexDualSignalPair]) -> ObservationFilter {
    let mut by_system = BTreeMap::<GnssSystem, BTreeSet<String>>::new();
    for pair in pairs {
        by_system.entry(pair.system).or_default().extend([
            pair.code1_observable.clone(),
            pair.phase1_observable.clone(),
            pair.code2_observable.clone(),
            pair.phase2_observable.clone(),
        ]);
    }
    ObservationFilter::from_entries(
        by_system
            .into_iter()
            .map(|(system, codes)| (system, codes.into_iter().collect())),
    )
}

fn rover_epoch_index(obs: &RinexObs) -> BTreeMap<(i32, u8, u8, u8, u8, u64), &ObsEpoch> {
    obs.epochs()
        .iter()
        .map(|epoch| (epoch_key(epoch.epoch), epoch))
        .collect()
}

fn rows_by_code(rows: Vec<ObservationValueRow>) -> BTreeMap<String, ObservationValueRow> {
    rows.into_iter()
        .map(|row| (row.code.clone(), row))
        .collect()
}

fn row_value(rows: &BTreeMap<String, ObservationValueRow>, code: &str) -> Option<f64> {
    rows.get(code).and_then(|row| row.value)
}

fn carrier_frequency_hz(
    obs: &RinexObs,
    sat: GnssSatelliteId,
    observable_code: &str,
) -> Result<f64, RtkRinexArcError> {
    let glonass_channel = (sat.system == GnssSystem::Glonass)
        .then(|| obs.header().glonass_slots.get(&sat.prn).copied())
        .flatten();
    observation_frequency_hz(
        sat.system,
        observable_code,
        obs.header().version,
        glonass_channel,
    )?
    .ok_or_else(|| RtkRinexArcError::MissingFrequency {
        satellite_id: sat.to_string(),
        observable_code: observable_code.to_string(),
    })
}

fn transmit_epoch_j2000_s(receive_epoch_j2000_s: f64, code_m: f64) -> f64 {
    let transmit_offset_us = (code_m / C_M_S * 1_000_000.0).round();
    receive_epoch_j2000_s - transmit_offset_us / 1_000_000.0
}

fn ephemeris_position(
    ephemeris: &dyn ObservableEphemerisSource,
    satellite_id: GnssSatelliteId,
    epoch_j2000_s: f64,
) -> Result<Option<[f64; 3]>, RtkRinexArcError> {
    match ephemeris.observable_state_at_j2000_s(satellite_id, epoch_j2000_s) {
        Ok(state) => Ok(Some(state.position_ecef_m)),
        Err(error) if is_observable_state_gap(&error) => Ok(None),
        Err(error) => Err(ephemeris_error(satellite_id, epoch_j2000_s, error)),
    }
}

fn ephemeris_error(
    satellite_id: GnssSatelliteId,
    epoch_j2000_s: f64,
    error: ObservablesError,
) -> RtkRinexArcError {
    RtkRinexArcError::Ephemeris {
        satellite_id: satellite_id.to_string(),
        epoch_j2000_s,
        reason: error.to_string(),
    }
}

fn retain_single_observations(
    observations: BTreeMap<String, SingleObservation>,
    keep: &BTreeSet<String>,
) -> Vec<RtkArcObservation> {
    observations
        .into_iter()
        .filter(|(satellite_id, _)| keep.contains(satellite_id))
        .map(|(satellite_id, observation)| RtkArcObservation {
            satellite_id: satellite_id.clone(),
            ambiguity_id: satellite_id,
            code_m: observation.code_m,
            phase_m: observation.phase_m,
            lli: observation.lli,
        })
        .collect()
}

fn common_keys<'a>(
    left: impl Iterator<Item = &'a String>,
    right: impl Iterator<Item = &'a String>,
) -> Vec<String> {
    let left = left.cloned().collect::<BTreeSet<_>>();
    let right = right.cloned().collect::<BTreeSet<_>>();
    left.intersection(&right).cloned().collect()
}

fn parse_satellite_id(token: &str) -> Result<GnssSatelliteId, RtkRinexArcError> {
    token.parse().map_err(|_| RtkRinexArcError::InvalidInput {
        field: "satellite_id",
        reason: "invalid satellite token",
    })
}

fn epoch_key(epoch: ObsEpochTime) -> (i32, u8, u8, u8, u8, u64) {
    (
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second.to_bits(),
    )
}

fn epoch_sort_key(epoch: ObsEpochTime) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:.9}",
        epoch.year, epoch.month, epoch.day, epoch.hour, epoch.minute, epoch.second
    )
}

fn civil_to_julian_split(epoch: ObsEpochTime) -> (f64, f64) {
    split_julian_date(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    )
}

fn j2000_seconds(epoch: ObsEpochTime) -> f64 {
    let (jd_whole, fraction) = civil_to_julian_split(epoch);
    j2000_seconds_from_split(jd_whole, fraction)
}
