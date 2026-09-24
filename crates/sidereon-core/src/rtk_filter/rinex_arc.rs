//! RINEX observation to RTK arc conversion.
//!
//! These builders turn paired base/rover RINEX observation epochs plus an
//! ephemeris source into the raw arc records consumed by the RTK filter drivers.
//! They own parsing, signal selection, shared-epoch matching, transmit-time
//! satellite lookup, and deterministic ordering only. Double-difference
//! reference selection and numeric solving stay in the existing RTK arc drivers.
//!
//! Each receiver's transmit-time satellite positions are placed from its own
//! pseudoranges as RTKLIB `rtkpos` places them (`satposs`): the transmission
//! epoch is `t_rx - P / c` less the satellite clock read there, and the clock and
//! the position both come from the record the source selects at the reception
//! epoch. A satellite with no clock there, or with a pseudorange that is not a
//! positive distance, is skipped for that epoch, as RTKLIB places none for it.

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::time::{split_julian_date, ExactEpoch};
use crate::constants::C_M_S;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::observables::{is_observable_state_gap, ObservableEphemerisSource, ObservablesError};
use crate::rinex::observations::{
    observation_frequency_hz, observation_values, ObsEpoch, ObsEpochTime, ObsHeader,
    ObservationFilter, ObservationValueRow, RinexObs,
};
use crate::rinex_obs::ObsHeaderTimeline;

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
#[non_exhaustive]
pub struct RtkRinexArcOptions {
    /// Signal choices grouped by constellation and tried in vector order. At
    /// each epoch, each satellite's measurement is formed from the first pair
    /// with both values present; an empty vector returns
    /// [`RtkRinexArcError::NoSignalPairs`]. A measurement on another carrier
    /// phase observable or frequency than the satellite's last one, including
    /// a return to an earlier one, starts a new ambiguity; a fallback to
    /// another code on the same carrier keeps it.
    pub signal_pairs: Vec<RtkRinexSignalPair>,
    /// Optional cap on base epochs considered, in file order.
    pub max_epochs: Option<usize>,
    /// Minimum common satellites with observations and ephemeris in an epoch.
    pub min_common_satellites: usize,
    /// Whether to fill `prediction_time_s` with seconds since J2000.
    pub include_prediction_time: bool,
}

impl RtkRinexArcOptions {
    /// Build single-frequency RINEX arc options from every selection and limit.
    #[must_use]
    pub fn new(
        signal_pairs: Vec<RtkRinexSignalPair>,
        max_epochs: Option<usize>,
        min_common_satellites: usize,
        include_prediction_time: bool,
    ) -> Self {
        Self {
            signal_pairs,
            max_epochs,
            min_common_satellites,
            include_prediction_time,
        }
    }

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
    /// Carrier wavelength per single-difference ambiguity id, metres. A
    /// satellite whose carrier changes, as a GLONASS slot re-declared on another
    /// channel changes it, starts a new ambiguity id, `<satellite>~freq<n>`, with
    /// its own wavelength.
    pub wavelengths_m: BTreeMap<String, f64>,
    /// Code-to-phase metre offsets per single-difference ambiguity id.
    pub offsets_m: BTreeMap<String, f64>,
    /// Number of considered base epochs omitted because the rover civil-time key
    /// was absent or too few usable satellites remained.
    pub skipped_epoch_count: usize,
    /// Carrier-phase measurements left out of an epoch because their observable
    /// has no carrier frequency in the file's context; see
    /// [`RtkRinexUnresolvedCarrier`].
    pub unresolved_carriers: Vec<RtkRinexUnresolvedCarrier>,
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
#[non_exhaustive]
pub struct RtkRinexDualArcOptions {
    /// Four-observable choices grouped by constellation and tried in vector order.
    /// At each epoch, each satellite's measurement is formed from the first
    /// pair with all four values present; an empty vector returns
    /// [`RtkRinexArcError::NoSignalPairs`]. A measurement on other carrier
    /// phase observables or frequencies than the satellite's last one starts a
    /// new ambiguity, as for one frequency.
    pub signal_pairs: Vec<RtkRinexDualSignalPair>,
    /// Optional cap on base epochs considered, in file order.
    pub max_epochs: Option<usize>,
    /// Minimum common satellites with observations and ephemeris in an epoch.
    pub min_common_satellites: usize,
    /// Whether to fill `prediction_time_s` with seconds since J2000.
    pub include_prediction_time: bool,
}

impl RtkRinexDualArcOptions {
    /// Build dual-frequency RINEX arc options from every selection and limit.
    #[must_use]
    pub fn new(
        signal_pairs: Vec<RtkRinexDualSignalPair>,
        max_epochs: Option<usize>,
        min_common_satellites: usize,
        include_prediction_time: bool,
    ) -> Self {
        Self {
            signal_pairs,
            max_epochs,
            min_common_satellites,
            include_prediction_time,
        }
    }

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
    /// Carrier-phase measurements left out of an epoch because an observable
    /// has no carrier frequency in the file's context; see
    /// [`RtkRinexUnresolvedCarrier`].
    pub unresolved_carriers: Vec<RtkRinexUnresolvedCarrier>,
}

/// The receiver whose observation file a reported measurement comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtkRinexReceiver {
    /// The base receiver's file.
    Base,
    /// The rover receiver's file.
    Rover,
}

/// A satellite's measurement left out of one epoch because a selected phase
/// observable has no carrier frequency in the file's context - a GLONASS slot
/// with no `GLONASS SLOT / FRQ #` channel, or one whose channel is outside the
/// `-7..=6` FDMA allocation, such as the `7` real IGS headers give `R28`.
///
/// Only that satellite is left out of that epoch, as RTKLIB leaves out a
/// measurement whose carrier frequency is zero; the rest of the epoch and the
/// arc are built as usual.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RtkRinexUnresolvedCarrier {
    /// The receiver whose file holds the measurement.
    pub receiver: RtkRinexReceiver,
    /// Index of the epoch in that receiver's file.
    pub epoch_index: usize,
    /// Satellite token.
    pub satellite_id: String,
    /// Full RINEX phase observable code with no carrier frequency.
    pub observable_code: String,
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
    /// The ephemeris source refused a satellite position because producing it
    /// reads UT1 outside the UT1 table under a strict UT1 policy. The arc build
    /// fails rather than leaving that satellite out.
    Ut1OutsideCoverage(crate::astro::time::DegradeReason),
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
            Self::Ut1OutsideCoverage(reason) => write!(
                f,
                "the ephemeris source refused an RTK arc satellite position: {reason}"
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
    // A GLONASS channel an event declares applies to the epochs after it.
    let base_timeline = base_obs.header_timeline()?;
    let rover_timeline = rover_obs.header_timeline()?;
    let mut epochs = Vec::new();
    let mut skipped_epoch_count = 0;
    let mut unresolved_carriers = Vec::new();
    let mut wavelengths_m = BTreeMap::new();
    let mut arcs = CarrierArcs::default();
    // Every carrier each receiver tracks, over every epoch whether or not a
    // measurement is formed at it.
    let phase_codes = phase_codes_by_system(
        options
            .signal_pairs
            .iter()
            .map(|pair| (pair.system, pair.phase_observable.as_str())),
    );
    let base_carriers = receiver_carriers(
        base_obs,
        &base_timeline,
        options.max_epochs,
        &filter,
        &phase_codes,
    );
    let rover_carriers = receiver_carriers(rover_obs, &rover_timeline, None, &filter, &phase_codes);
    let mut base_emitted = LastEmitted::new();
    let mut rover_emitted = LastEmitted::new();

    for (base_index, base_epoch) in base_obs
        .epochs()
        .iter()
        .enumerate()
        .take(options.max_epochs.unwrap_or(usize::MAX))
    {
        let Some(base_time) = observation_epoch_time(base_epoch) else {
            skipped_epoch_count += 1;
            continue;
        };
        let Some(&(rover_index, rover_epoch)) = rover_by_epoch.get(&epoch_key(base_time)) else {
            skipped_epoch_count += 1;
            continue;
        };
        let epoch_j2000_s = j2000_seconds(base_time);
        let exact_epoch = exact_epoch(base_time);
        let mut base_values = single_frequency_observations(
            base_obs,
            base_timeline.at(base_index),
            base_epoch,
            &filter,
            &pair_by_system,
            UnresolvedSink {
                receiver: RtkRinexReceiver::Base,
                epoch_index: base_index,
                out: &mut unresolved_carriers,
            },
        )?;
        let mut rover_values = single_frequency_observations(
            rover_obs,
            rover_timeline.at(rover_index),
            rover_epoch,
            &filter,
            &pair_by_system,
            UnresolvedSink {
                receiver: RtkRinexReceiver::Rover,
                epoch_index: rover_index,
                out: &mut unresolved_carriers,
            },
        )?;
        let common = common_keys(base_values.keys(), rover_values.keys());

        let mut satellite_positions_m = BTreeMap::new();
        let mut base_satellite_positions_m = BTreeMap::new();
        let mut rover_satellite_positions_m = BTreeMap::new();
        let mut usable = BTreeSet::new();

        for satellite_id in common {
            // A single difference holds an integer ambiguity only when both
            // receivers track the satellite on one carrier; a GLONASS channel
            // the two files give differently leaves none.
            if base_values[&satellite_id].wavelength_m.to_bits()
                != rover_values[&satellite_id].wavelength_m.to_bits()
            {
                continue;
            }
            let sat = parse_satellite_id(&satellite_id)?;
            let Some(position) = ephemeris_position(ephemeris, sat, epoch_j2000_s)? else {
                continue;
            };
            let Some(base_tx) = placed_position(
                ephemeris,
                sat,
                epoch_j2000_s,
                base_values[&satellite_id].code_m,
            )?
            else {
                continue;
            };
            let Some(rover_tx) = placed_position(
                ephemeris,
                sat,
                epoch_j2000_s,
                rover_values[&satellite_id].code_m,
            )?
            else {
                continue;
            };
            satellite_positions_m.insert(satellite_id.clone(), position);
            base_satellite_positions_m.insert(satellite_id.clone(), base_tx);
            rover_satellite_positions_m.insert(satellite_id.clone(), rover_tx);
            usable.insert(satellite_id);
        }

        if usable.len() < options.min_common_satellites {
            skipped_epoch_count += 1;
            continue;
        }

        // A satellite whose carrier changes, on either receiver, starts a new
        // ambiguity with its own scale, and a loss of lock at an epoch no
        // measurement was formed at reaches the next one on its carrier.
        let mut ambiguity_ids = BTreeMap::new();
        for satellite_id in &usable {
            let (Some(base), Some(rover)) = (
                base_values.get_mut(satellite_id),
                rover_values.get_mut(satellite_id),
            ) else {
                continue;
            };
            let ambiguity_id = arcs.ambiguity_id(
                satellite_id,
                vec![
                    carrier_identity(
                        &base_carriers,
                        satellite_id,
                        &base.phase_observable,
                        base_index,
                    ),
                    carrier_identity(
                        &rover_carriers,
                        satellite_id,
                        &rover.phase_observable,
                        rover_index,
                    ),
                ],
            );
            carry_loss_of_lock(
                &mut base.lli,
                &base_carriers,
                &mut base_emitted,
                satellite_id,
                &base.phase_observable,
                base_index,
            );
            carry_loss_of_lock(
                &mut rover.lli,
                &rover_carriers,
                &mut rover_emitted,
                satellite_id,
                &rover.phase_observable,
                rover_index,
            );
            wavelengths_m.insert(ambiguity_id.clone(), base.wavelength_m);
            ambiguity_ids.insert(satellite_id.clone(), ambiguity_id);
        }

        epochs.push(RtkArcEpoch {
            base: retain_single_observations(base_values, &ambiguity_ids),
            rover: retain_single_observations(rover_values, &ambiguity_ids),
            satellite_positions_m,
            base_satellite_positions_m,
            rover_satellite_positions_m,
            velocity_mps: None,
            prediction_time_s: options.include_prediction_time.then_some(epoch_j2000_s),
            prediction_epoch: exact_epoch.filter(|_| options.include_prediction_time),
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
        unresolved_carriers,
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
    // A GLONASS channel an event declares applies to the epochs after it.
    let base_timeline = base_obs.header_timeline()?;
    let rover_timeline = rover_obs.header_timeline()?;
    let mut epochs = Vec::new();
    let mut skipped_epoch_count = 0;
    let mut unresolved_carriers = Vec::new();
    let mut arcs = CarrierArcs::default();
    // Every carrier each receiver tracks, over every epoch.
    let phase_codes = phase_codes_by_system(options.signal_pairs.iter().flat_map(|pair| {
        [
            (pair.system, pair.phase1_observable.as_str()),
            (pair.system, pair.phase2_observable.as_str()),
        ]
    }));
    let base_carriers = receiver_carriers(
        base_obs,
        &base_timeline,
        options.max_epochs,
        &filter,
        &phase_codes,
    );
    let rover_carriers = receiver_carriers(rover_obs, &rover_timeline, None, &filter, &phase_codes);
    let mut base_emitted = LastEmitted::new();
    let mut rover_emitted = LastEmitted::new();

    for (base_index, base_epoch) in base_obs
        .epochs()
        .iter()
        .enumerate()
        .take(options.max_epochs.unwrap_or(usize::MAX))
    {
        let Some(base_time) = observation_epoch_time(base_epoch) else {
            skipped_epoch_count += 1;
            continue;
        };
        let Some(&(rover_index, rover_epoch)) = rover_by_epoch.get(&epoch_key(base_time)) else {
            skipped_epoch_count += 1;
            continue;
        };
        let epoch_j2000_s = j2000_seconds(base_time);
        let exact_epoch = exact_epoch(base_time);
        let base_values = dual_frequency_observations(
            base_obs,
            base_timeline.at(base_index),
            base_epoch,
            &filter,
            &pair_by_system,
            UnresolvedSink {
                receiver: RtkRinexReceiver::Base,
                epoch_index: base_index,
                out: &mut unresolved_carriers,
            },
        )?;
        let rover_values = dual_frequency_observations(
            rover_obs,
            rover_timeline.at(rover_index),
            rover_epoch,
            &filter,
            &pair_by_system,
            UnresolvedSink {
                receiver: RtkRinexReceiver::Rover,
                epoch_index: rover_index,
                out: &mut unresolved_carriers,
            },
        )?;
        let common = common_keys(base_values.keys(), rover_values.keys());

        let mut satellite_positions_m = BTreeMap::new();
        let mut base_satellite_positions_m = BTreeMap::new();
        let mut rover_satellite_positions_m = BTreeMap::new();
        let mut observations: Vec<RtkDualFrequencySatelliteObservation> = Vec::new();
        let mut selected_phases = BTreeMap::new();

        for satellite_id in common {
            // Both receivers have to track the satellite on the same carriers.
            let ((base, base_phases), (rover, rover_phases)) =
                (&base_values[&satellite_id], &rover_values[&satellite_id]);
            if base.f1_hz.to_bits() != rover.f1_hz.to_bits()
                || base.f2_hz.to_bits() != rover.f2_hz.to_bits()
            {
                continue;
            }
            let sat = parse_satellite_id(&satellite_id)?;
            let Some(position) = ephemeris_position(ephemeris, sat, epoch_j2000_s)? else {
                continue;
            };
            let Some(base_tx) = placed_position(ephemeris, sat, epoch_j2000_s, base.p1_m)? else {
                continue;
            };
            let Some(rover_tx) = placed_position(ephemeris, sat, epoch_j2000_s, rover.p1_m)? else {
                continue;
            };
            satellite_positions_m.insert(satellite_id.clone(), position);
            base_satellite_positions_m.insert(satellite_id.clone(), base_tx);
            rover_satellite_positions_m.insert(satellite_id.clone(), rover_tx);
            selected_phases.insert(
                satellite_id.clone(),
                (base_phases.clone(), rover_phases.clone()),
            );
            observations.push(RtkDualFrequencySatelliteObservation {
                satellite_id: satellite_id.clone(),
                base: base.clone(),
                rover: rover.clone(),
            });
        }

        if observations.len() < options.min_common_satellites {
            skipped_epoch_count += 1;
            continue;
        }
        // A satellite whose carriers change starts a new ambiguity, and a loss
        // of lock at an epoch left out reaches the next observation of its
        // carrier.
        for observation in &mut observations {
            let satellite_id = observation.satellite_id.clone();
            let Some(([base_phase1, base_phase2], [rover_phase1, rover_phase2])) =
                selected_phases.get(&satellite_id)
            else {
                continue;
            };
            let ambiguity_id = arcs.ambiguity_id(
                &satellite_id,
                vec![
                    carrier_identity(&base_carriers, &satellite_id, base_phase1, base_index),
                    carrier_identity(&base_carriers, &satellite_id, base_phase2, base_index),
                    carrier_identity(&rover_carriers, &satellite_id, rover_phase1, rover_index),
                    carrier_identity(&rover_carriers, &satellite_id, rover_phase2, rover_index),
                ],
            );
            carry_loss_of_lock(
                &mut observation.base.lli1,
                &base_carriers,
                &mut base_emitted,
                &satellite_id,
                base_phase1,
                base_index,
            );
            carry_loss_of_lock(
                &mut observation.base.lli2,
                &base_carriers,
                &mut base_emitted,
                &satellite_id,
                base_phase2,
                base_index,
            );
            carry_loss_of_lock(
                &mut observation.rover.lli1,
                &rover_carriers,
                &mut rover_emitted,
                &satellite_id,
                rover_phase1,
                rover_index,
            );
            carry_loss_of_lock(
                &mut observation.rover.lli2,
                &rover_carriers,
                &mut rover_emitted,
                &satellite_id,
                rover_phase2,
                rover_index,
            );
            observation.base.ambiguity_id.clone_from(&ambiguity_id);
            observation.rover.ambiguity_id = ambiguity_id;
        }

        let (jd_whole, jd_fraction) = civil_to_julian_split(base_time);
        epochs.push(RtkDualFrequencyArcEpoch {
            jd_whole,
            jd_fraction,
            epoch_sort_key: Some(epoch_sort_key(base_time)),
            gap_time_s: Some(epoch_j2000_s),
            gap_epoch: exact_epoch,
            observations,
            satellite_positions_m,
            base_satellite_positions_m,
            rover_satellite_positions_m,
            velocity_mps: None,
            prediction_time_s: options.include_prediction_time.then_some(epoch_j2000_s),
            prediction_epoch: exact_epoch.filter(|_| options.include_prediction_time),
        });
    }

    if epochs.is_empty() {
        return Err(RtkRinexArcError::NoUsableEpochs);
    }
    Ok(RtkRinexDualFrequencyArc {
        epochs,
        skipped_epoch_count,
        unresolved_carriers,
    })
}

#[derive(Debug, Clone, PartialEq)]
struct SingleObservation {
    code_m: f64,
    phase_m: f64,
    wavelength_m: f64,
    lli: Option<i64>,
    /// The carrier phase observable the measurement is on.
    phase_observable: String,
}

/// Where one receiver's epoch reports the measurements it leaves out for want
/// of a carrier frequency.
struct UnresolvedSink<'a> {
    receiver: RtkRinexReceiver,
    epoch_index: usize,
    out: &'a mut Vec<RtkRinexUnresolvedCarrier>,
}

impl UnresolvedSink<'_> {
    fn report(&mut self, sat: GnssSatelliteId, observable_code: &str) {
        self.out.push(RtkRinexUnresolvedCarrier {
            receiver: self.receiver,
            epoch_index: self.epoch_index,
            satellite_id: sat.to_string(),
            observable_code: observable_code.to_string(),
        });
    }
}

fn single_frequency_observations(
    obs: &RinexObs,
    header: &ObsHeader,
    epoch: &ObsEpoch,
    filter: &ObservationFilter,
    pair_by_system: &BTreeMap<GnssSystem, Vec<RtkRinexSignalPair>>,
    mut unresolved: UnresolvedSink<'_>,
) -> Result<BTreeMap<String, SingleObservation>, RtkRinexArcError> {
    let mut out = BTreeMap::new();
    for (sat, rows) in observation_values(obs, epoch, filter)? {
        let Some(pairs) = pair_by_system.get(&sat.system) else {
            continue;
        };
        let rows_by_code = rows_by_code(rows);
        let (pair, frequency_hz) = match selected_single_pair(header, sat, pairs, &rows_by_code)? {
            PairSelection::Selected(selected) => selected,
            PairSelection::NoneHeld => continue,
            PairSelection::NoneResolved(observables) => {
                for observable in observables {
                    unresolved.report(sat, observable);
                }
                continue;
            }
        };
        let (Some(code_m), Some(phase_cycles)) = (
            row_value(&rows_by_code, &pair.code_observable),
            row_value(&rows_by_code, &pair.phase_observable),
        ) else {
            continue;
        };
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
                phase_observable: pair.phase_observable.clone(),
            },
        );
    }
    Ok(out)
}

/// The outcome of choosing the pair a satellite's measurement is formed from.
enum PairSelection<'a, P, F> {
    /// The first configured pair whose values the epoch holds and whose
    /// carriers all resolve, with their frequencies.
    Selected((&'a P, F)),
    /// No configured pair's values are all present.
    NoneHeld,
    /// Some pair's values are present, but no such pair has every carrier
    /// resolved; the unresolved phase observables, in pair order.
    NoneResolved(Vec<&'a str>),
}

/// Note an unresolved phase observable once, however many pairs name it.
fn push_once<'a>(unresolved: &mut Vec<&'a str>, observable: &'a str) {
    if !unresolved.contains(&observable) {
        unresolved.push(observable);
    }
}

/// The pair a satellite's single-frequency measurement is formed from at an
/// epoch: the first configured pair whose code and carrier phase the epoch both
/// hold and whose carrier has a frequency in the header in effect. A pair whose
/// carrier does not resolve - GLONASS FDMA `L1C` on a channel outside the
/// allocation, say - gives way to a later one that does, such as the CDMA `L3Q`.
fn selected_single_pair<'a>(
    header: &ObsHeader,
    sat: GnssSatelliteId,
    pairs: &'a [RtkRinexSignalPair],
    rows_by_code: &BTreeMap<String, ObservationValueRow>,
) -> Result<PairSelection<'a, RtkRinexSignalPair, f64>, RtkRinexArcError> {
    let mut unresolved = Vec::new();
    for pair in pairs {
        if row_value(rows_by_code, &pair.code_observable).is_none()
            || row_value(rows_by_code, &pair.phase_observable).is_none()
        {
            continue;
        }
        match carrier_frequency_hz(header, sat, &pair.phase_observable)? {
            Some(frequency_hz) => return Ok(PairSelection::Selected((pair, frequency_hz))),
            None => push_once(&mut unresolved, &pair.phase_observable),
        }
    }
    Ok(if unresolved.is_empty() {
        PairSelection::NoneHeld
    } else {
        PairSelection::NoneResolved(unresolved)
    })
}

/// The pair a satellite's dual-frequency measurement is formed from at an
/// epoch: the first configured pair whose two codes and two carrier phases the
/// epoch all holds and whose two carriers both resolve.
fn selected_dual_pair<'a>(
    header: &ObsHeader,
    sat: GnssSatelliteId,
    pairs: &'a [RtkRinexDualSignalPair],
    rows_by_code: &BTreeMap<String, ObservationValueRow>,
) -> Result<PairSelection<'a, RtkRinexDualSignalPair, (f64, f64)>, RtkRinexArcError> {
    let mut unresolved = Vec::new();
    for pair in pairs {
        let held = [
            &pair.code1_observable,
            &pair.phase1_observable,
            &pair.code2_observable,
            &pair.phase2_observable,
        ]
        .iter()
        .all(|code| row_value(rows_by_code, code).is_some());
        if !held {
            continue;
        }
        let f1 = carrier_frequency_hz(header, sat, &pair.phase1_observable)?;
        let f2 = carrier_frequency_hz(header, sat, &pair.phase2_observable)?;
        match (f1, f2) {
            (Some(f1_hz), Some(f2_hz)) => {
                return Ok(PairSelection::Selected((pair, (f1_hz, f2_hz))));
            }
            _ => {
                if f1.is_none() {
                    push_once(&mut unresolved, &pair.phase1_observable);
                }
                if f2.is_none() {
                    push_once(&mut unresolved, &pair.phase2_observable);
                }
            }
        }
    }
    Ok(if unresolved.is_empty() {
        PairSelection::NoneHeld
    } else {
        PairSelection::NoneResolved(unresolved)
    })
}

/// A satellite's measurement with the two carrier phase observables it is on.
type DualObservationOnPhases = (RtkDualFrequencyObservation, [String; 2]);

fn dual_frequency_observations(
    obs: &RinexObs,
    header: &ObsHeader,
    epoch: &ObsEpoch,
    filter: &ObservationFilter,
    pair_by_system: &BTreeMap<GnssSystem, Vec<RtkRinexDualSignalPair>>,
    mut unresolved: UnresolvedSink<'_>,
) -> Result<BTreeMap<String, DualObservationOnPhases>, RtkRinexArcError> {
    let mut out = BTreeMap::new();
    for (sat, rows) in observation_values(obs, epoch, filter)? {
        let Some(pairs) = pair_by_system.get(&sat.system) else {
            continue;
        };
        let rows_by_code = rows_by_code(rows);
        let (pair, (f1_hz, f2_hz)) = match selected_dual_pair(header, sat, pairs, &rows_by_code)? {
            PairSelection::Selected(selected) => selected,
            PairSelection::NoneHeld => continue,
            PairSelection::NoneResolved(observables) => {
                for observable in observables {
                    unresolved.report(sat, observable);
                }
                continue;
            }
        };
        let (Some(p1_m), Some(p2_m), Some(phi1_cycles), Some(phi2_cycles)) = (
            row_value(&rows_by_code, &pair.code1_observable),
            row_value(&rows_by_code, &pair.code2_observable),
            row_value(&rows_by_code, &pair.phase1_observable),
            row_value(&rows_by_code, &pair.phase2_observable),
        ) else {
            continue;
        };
        out.insert(
            sat.to_string(),
            (
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
                [
                    pair.phase1_observable.clone(),
                    pair.phase2_observable.clone(),
                ],
            ),
        );
    }
    Ok(out)
}

/// The carrier phase observables configured for each constellation.
fn phase_codes_by_system<'a>(
    codes: impl Iterator<Item = (GnssSystem, &'a str)>,
) -> BTreeMap<GnssSystem, BTreeSet<String>> {
    let mut out = BTreeMap::<GnssSystem, BTreeSet<String>>::new();
    for (system, code) in codes {
        out.entry(system).or_default().insert(code.to_string());
    }
    out
}

/// One carrier a receiver tracks, a satellite's carrier phase observable, over
/// every observation epoch: where its frequency changes and where it loses
/// lock. It is followed wherever its phase is recorded, whether or not a code
/// was measured with it or a measurement was formed.
#[derive(Debug, Default)]
struct CarrierHistory {
    /// The epoch index each frequency starts at, with the frequency's bits, or
    /// `None` from an epoch whose phase is recorded but whose carrier has no
    /// frequency in the header in effect there. An unresolved stretch is an arc
    /// of its own, so a carrier that returns to its earlier frequency after one
    /// is not taken for the carrier it was before.
    frequency_starts: Vec<(usize, Option<u64>)>,
    /// The epoch indices whose loss of lock indicator is set.
    losses: Vec<usize>,
}

impl CarrierHistory {
    /// The carrier's frequency arc at an epoch: 1 from its first frequency and
    /// one more at each change.
    fn arc_at(&self, index: usize) -> usize {
        self.frequency_starts
            .partition_point(|(start, _)| *start <= index)
            .max(1)
    }

    /// Whether the carrier lost lock after epoch `after` and before `before`.
    fn lost_between(&self, after: usize, before: usize) -> bool {
        let first = self.losses.partition_point(|index| *index <= after);
        self.losses.get(first).is_some_and(|index| *index < before)
    }
}

/// A receiver's carrier histories by satellite and carrier phase observable.
type ReceiverCarriers = BTreeMap<(String, String), CarrierHistory>;

/// Each carrier a receiver tracks over every observation epoch, up to `limit`
/// epochs: its frequency from the header in effect wherever its phase is
/// recorded, and its loss of lock indicator wherever it is set.
fn receiver_carriers(
    obs: &RinexObs,
    timeline: &ObsHeaderTimeline,
    limit: Option<usize>,
    filter: &ObservationFilter,
    phase_codes: &BTreeMap<GnssSystem, BTreeSet<String>>,
) -> ReceiverCarriers {
    let mut carriers = ReceiverCarriers::new();
    for (index, epoch) in obs
        .epochs()
        .iter()
        .enumerate()
        .take(limit.unwrap_or(usize::MAX))
    {
        if observation_epoch_time(epoch).is_none() {
            continue;
        }
        let Ok(values) = observation_values(obs, epoch, filter) else {
            continue;
        };
        let header = timeline.at(index);
        for (sat, rows) in values {
            let Some(codes) = phase_codes.get(&sat.system) else {
                continue;
            };
            let rows = rows_by_code(rows);
            for code in codes {
                let Some(row) = rows.get(code) else {
                    continue;
                };
                let key = (sat.to_string(), code.clone());
                if row.value.is_some() {
                    if let Ok(frequency_hz) = carrier_frequency_hz(header, sat, code) {
                        let bits = frequency_hz.map(f64::to_bits);
                        let history = carriers.entry(key.clone()).or_default();
                        if history
                            .frequency_starts
                            .last()
                            .is_none_or(|(_, held)| *held != bits)
                        {
                            history.frequency_starts.push((index, bits));
                        }
                    }
                }
                if row.lli.is_some_and(|lli| lli & 1 == 1) {
                    carriers.entry(key).or_default().losses.push(index);
                }
            }
        }
    }
    carriers
}

/// What a measured carrier is at an epoch: its phase observable and its
/// frequency arc there.
fn carrier_identity(
    carriers: &ReceiverCarriers,
    satellite_id: &str,
    phase_observable: &str,
    index: usize,
) -> (String, usize) {
    let arc = carriers
        .get(&(satellite_id.to_string(), phase_observable.to_string()))
        .map_or(1, |history| history.arc_at(index));
    (phase_observable.to_string(), arc)
}

/// The epoch each of a receiver's carriers was last measured at, by satellite
/// and carrier phase observable.
type LastEmitted = BTreeMap<(String, String), usize>;

/// Set a measurement's loss of lock bit where its carrier lost lock since its
/// last measurement, at an epoch no measurement was formed at, and note this
/// one. Before a carrier's first measurement there is no ambiguity to break.
fn carry_loss_of_lock(
    lli: &mut Option<i64>,
    carriers: &ReceiverCarriers,
    emitted: &mut LastEmitted,
    satellite_id: &str,
    phase_observable: &str,
    index: usize,
) {
    let key = (satellite_id.to_string(), phase_observable.to_string());
    let lost = emitted.get(&key).is_some_and(|last| {
        carriers
            .get(&key)
            .is_some_and(|history| history.lost_between(*last, index))
    });
    if lost {
        *lli = Some(lli.unwrap_or(0) | 1);
    }
    emitted.insert(key, index);
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

/// The time of an epoch holding observations: flag 0 or 1, with an epoch time.
/// An event or cycle slip epoch can share an observation epoch's time, and
/// holds no observation to pair.
fn observation_epoch_time(epoch: &ObsEpoch) -> Option<ObsEpochTime> {
    epoch.epoch.filter(|_| epoch.flag <= 1)
}

/// Rover observation epochs by time, each with its index in the product.
type RoverEpochIndex<'a> = BTreeMap<(i32, u8, u8, u8, u8, u64), (usize, &'a ObsEpoch)>;

/// Each rover observation epoch by its time, with its index.
fn rover_epoch_index(obs: &RinexObs) -> RoverEpochIndex<'_> {
    obs.epochs()
        .iter()
        .enumerate()
        .filter_map(|(index, epoch)| {
            observation_epoch_time(epoch).map(|time| (epoch_key(time), (index, epoch)))
        })
        .collect()
}

fn rows_by_code(rows: Vec<ObservationValueRow>) -> BTreeMap<String, ObservationValueRow> {
    // A code a list declares twice is read by its first copy, as observation
    // QC and the pseudorange selection read it.
    let mut by_code = BTreeMap::new();
    for row in rows {
        by_code.entry(row.code.clone()).or_insert(row);
    }
    by_code
}

fn row_value(rows: &BTreeMap<String, ObservationValueRow>, code: &str) -> Option<f64> {
    rows.get(code).and_then(|row| row.value)
}

/// The carrier frequency of a phase observable in the file's context, or `None`
/// when the context gives it none (a GLONASS slot with no channel, or a
/// channel outside the FDMA allocation).
fn carrier_frequency_hz(
    header: &ObsHeader,
    sat: GnssSatelliteId,
    observable_code: &str,
) -> Result<Option<f64>, RtkRinexArcError> {
    let glonass_channel = (sat.system == GnssSystem::Glonass)
        .then(|| header.glonass_slots.get(&sat.prn).copied())
        .flatten();
    observation_frequency_hz(sat.system, observable_code, header.version, glonass_channel)
        .map_err(RtkRinexArcError::from)
}

/// Satellite ECEF position at the transmission epoch of one receiver's pseudorange
/// `code_m`, received at `receive_epoch_j2000_s`, as RTKLIB `rtkpos` forms it for each
/// receiver from its own pseudoranges (`satposs`): the epoch `t_rx - P / c` less the
/// satellite clock read there, and the state there, both from the record selected at the
/// reception epoch. `None` where the source cannot place the satellite: no ephemeris or no
/// clock there, or a pseudorange that is not a positive distance, which RTKLIB reads as
/// none and places no satellite for.
fn placed_position(
    ephemeris: &dyn ObservableEphemerisSource,
    satellite_id: GnssSatelliteId,
    receive_epoch_j2000_s: f64,
    code_m: f64,
) -> Result<Option<[f64; 3]>, RtkRinexArcError> {
    let transmit_epoch_j2000_s = match crate::observables::pseudorange_transmit_epoch_j2000_s(
        ephemeris,
        satellite_id,
        receive_epoch_j2000_s,
        code_m,
    ) {
        Ok(epoch) => epoch,
        Err(ObservablesError::InvalidInput {
            field: "pseudorange_m",
            ..
        }) => return Ok(None),
        Err(error) if is_observable_state_gap(&error) => return Ok(None),
        Err(error) => return Err(ephemeris_error(satellite_id, receive_epoch_j2000_s, error)),
    };
    match ephemeris.try_observable_state_group_delay_selected_at_j2000_s(
        satellite_id,
        transmit_epoch_j2000_s,
        receive_epoch_j2000_s,
    ) {
        Ok(state) => Ok(Some(state.value.0.position_ecef_m)),
        Err(error) if is_observable_state_gap(&error) => Ok(None),
        Err(error) => Err(ephemeris_error(satellite_id, transmit_epoch_j2000_s, error)),
    }
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
    if let ObservablesError::Ephemeris(crate::Error::Ut1OutsideCoverage(reason)) = error {
        return RtkRinexArcError::Ut1OutsideCoverage(reason);
    }
    RtkRinexArcError::Ephemeris {
        satellite_id: satellite_id.to_string(),
        epoch_j2000_s,
        reason: error.to_string(),
    }
}

/// The single-difference ambiguity arcs: a satellite starts a new arc whenever
/// the carriers its measurement is on have changed since its last single
/// difference, on either receiver, in phase observable or in frequency arc,
/// including a change back to an earlier carrier, and including changes at
/// epochs left out. The first arc is named by the satellite, and each later one
/// `<satellite>~freq<n>`.
#[derive(Default)]
struct CarrierArcs {
    held: BTreeMap<String, (Vec<(String, usize)>, usize)>,
}

impl CarrierArcs {
    fn ambiguity_id(&mut self, satellite_id: &str, carriers: Vec<(String, usize)>) -> String {
        let arc = match self.held.get_mut(satellite_id) {
            Some((last, arc)) => {
                if *last != carriers {
                    *last = carriers;
                    *arc += 1;
                }
                *arc
            }
            None => {
                self.held.insert(satellite_id.to_string(), (carriers, 1));
                1
            }
        };
        if arc == 1 {
            satellite_id.to_string()
        } else {
            format!("{satellite_id}~freq{arc}")
        }
    }
}

fn retain_single_observations(
    observations: BTreeMap<String, SingleObservation>,
    ambiguity_ids: &BTreeMap<String, String>,
) -> Vec<RtkArcObservation> {
    observations
        .into_iter()
        .filter_map(|(satellite_id, observation)| {
            ambiguity_ids
                .get(&satellite_id)
                .map(|ambiguity_id| (satellite_id, ambiguity_id.clone(), observation))
        })
        .map(
            |(satellite_id, ambiguity_id, observation)| RtkArcObservation {
                satellite_id,
                ambiguity_id,
                code_m: observation.code_m,
                phase_m: observation.phase_m,
                lli: observation.lli,
            },
        )
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

/// Seconds since J2000 of an observation epoch: the whole clock seconds plus
/// the second's fractional part in one `f64` addition, the time RTKLIB's
/// `epoch2time` holds for the epoch, rounded once.
fn j2000_seconds(epoch: ObsEpochTime) -> f64 {
    crate::astro::time::j2000_seconds(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    )
}

/// The observation epoch held exactly, for intervals between epochs.
fn exact_epoch(epoch: ObsEpochTime) -> Option<ExactEpoch> {
    ExactEpoch::from_civil(
        epoch.year,
        i32::from(epoch.month),
        i32::from(epoch.day),
        i32::from(epoch.hour),
        i32::from(epoch.minute),
        epoch.second,
    )
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::observables::ObservableState;

    /// Every satellite at one fixed position at every time.
    struct FixedSource;

    impl ObservableEphemerisSource for FixedSource {
        fn observable_state_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<crate::observables::ObservableState, ObservablesError> {
            Ok(ObservableState {
                position_ecef_m: [20_000_000.0, 10_000_000.0, 10_000_000.0],
                clock_s: Some(0.0),
            })
        }
    }

    /// Refuses every state, as an SSR source does outside the UT1 table under
    /// a strict UT1 policy.
    struct Ut1RefusingSource;

    impl ObservableEphemerisSource for Ut1RefusingSource {
        fn observable_state_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<crate::observables::ObservableState, ObservablesError> {
            Err(ObservablesError::Ephemeris(
                crate::Error::Ut1OutsideCoverage(crate::astro::time::DegradeReason::AfterCoverage),
            ))
        }
    }

    #[test]
    fn ephemeris_position_keeps_a_ut1_refusal_typed() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("G01");
        assert_eq!(
            ephemeris_position(&Ut1RefusingSource, sat, 0.0),
            Err(RtkRinexArcError::Ut1OutsideCoverage(
                crate::astro::time::DegradeReason::AfterCoverage
            ))
        );
        assert_eq!(
            ephemeris_position(&FixedSource, sat, 0.0),
            Ok(Some([20_000_000.0, 10_000_000.0, 10_000_000.0]))
        );
    }

    /// A satellite whose position records the epoch it was read at and the epoch the
    /// record was selected at, with the clock `clock_s`.
    struct RecordingSource {
        clock_s: Option<f64>,
        reads: std::cell::RefCell<Vec<(f64, f64)>>,
    }

    impl ObservableEphemerisSource for RecordingSource {
        fn observable_state_at_j2000_s(
            &self,
            sat: GnssSatelliteId,
            t_j2000_s: f64,
        ) -> Result<crate::observables::ObservableState, ObservablesError> {
            Ok(self
                .try_observable_state_group_delay_selected_at_j2000_s(sat, t_j2000_s, t_j2000_s)?
                .value
                .0)
        }

        fn try_observable_state_group_delay_selected_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            t_j2000_s: f64,
            selection_j2000_s: f64,
        ) -> Result<crate::astro::time::Validated<(ObservableState, Option<f64>)>, ObservablesError>
        {
            self.reads.borrow_mut().push((t_j2000_s, selection_j2000_s));
            Ok(crate::astro::time::Validated {
                value: (
                    ObservableState {
                        position_ecef_m: [t_j2000_s, selection_j2000_s, 1.0],
                        clock_s: self.clock_s,
                    },
                    None,
                ),
                degraded: None,
            })
        }
    }

    #[test]
    fn placed_position_is_rtklib_satposs_per_receiver() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("G01");
        let t_rx = 646_315_200.0;
        let code_m = 21_000_000.0;
        let clock_s = 1.5e-4;
        let source = RecordingSource {
            clock_s: Some(clock_s),
            reads: std::cell::RefCell::new(Vec::new()),
        };
        let clock_epoch = t_rx - code_m / C_M_S;
        let transmit_epoch = clock_epoch - clock_s;
        // The clock is read at t_rx - P/c and the state at that epoch less the clock,
        // both from the record selected at the reception epoch; nothing is rounded to
        // a whole microsecond.
        assert_eq!(
            placed_position(&source, sat, t_rx, code_m),
            Ok(Some([transmit_epoch, t_rx, 1.0]))
        );
        assert!(source
            .reads
            .borrow()
            .iter()
            .all(|(_, selection)| selection.to_bits() == t_rx.to_bits()));
        assert!(source
            .reads
            .borrow()
            .iter()
            .any(|(t, _)| t.to_bits() == clock_epoch.to_bits()));

        // No satellite clock, or a pseudorange that is not a positive distance,
        // places no satellite, as RTKLIB `satposs` places none.
        let clockless = RecordingSource {
            clock_s: None,
            reads: std::cell::RefCell::new(Vec::new()),
        };
        assert_eq!(placed_position(&clockless, sat, t_rx, code_m), Ok(None));
        assert_eq!(placed_position(&source, sat, t_rx, 0.0), Ok(None));
        assert_eq!(placed_position(&source, sat, t_rx, -1.0), Ok(None));
        assert_eq!(placed_position(&source, sat, t_rx, f64::NAN), Ok(None));

        assert_eq!(
            placed_position(&Ut1RefusingSource, sat, t_rx, code_m),
            Err(RtkRinexArcError::Ut1OutsideCoverage(
                crate::astro::time::DegradeReason::AfterCoverage
            ))
        );
    }

    fn header_line(content: &str, label: &str) -> String {
        format!("{content:<60}{label}")
    }

    /// R01 on channel -7, then, after a flag 4 event re-declaring its slot, on
    /// channel +6, with 100 carrier cycles at both epochs.
    fn channel_change_text(types: &str, record: &str) -> String {
        [
            header_line(
                "     3.05           OBSERVATION DATA    R (GLONASS)",
                "RINEX VERSION / TYPE",
            ),
            header_line(types, "SYS / # / OBS TYPES"),
            header_line("  1 R01 -7", "GLONASS SLOT / FRQ #"),
            header_line("", "END OF HEADER"),
            "> 2020 01 01 00 00  0.0000000  0  1".to_string(),
            record.to_string(),
            format!(">{:30}4  1", ""),
            header_line("  1 R01  6", "GLONASS SLOT / FRQ #"),
            "> 2020 01 01 00 00 30.0000000  0  1".to_string(),
            record.to_string(),
        ]
        .join("\n")
    }

    fn wavelength_m(channel: i8) -> f64 {
        C_M_S
            / observation_frequency_hz(GnssSystem::Glonass, "L1C", 3.05, Some(channel))
                .expect("frequency")
                .expect("GLONASS L1 frequency")
    }

    #[test]
    fn a_glonass_channel_change_starts_a_new_single_frequency_ambiguity() {
        let text = channel_change_text(
            "R    2 C1C L1C",
            &format!("R01{:14.3}  {:14.3}", 20_000_000.0, 100.0),
        );
        let obs = RinexObs::parse(&text).expect("parse");
        let options = RtkRinexArcOptions::new(
            vec![RtkRinexSignalPair {
                system: GnssSystem::Glonass,
                code_observable: "C1C".to_string(),
                phase_observable: "L1C".to_string(),
            }],
            None,
            1,
            false,
        );
        let arc = build_rinex_rtk_arc(&FixedSource, &obs, &obs, &options).expect("arc");
        assert_eq!(arc.epochs.len(), 2);
        let first = &arc.epochs[0].base[0];
        let second = &arc.epochs[1].base[0];
        assert_ne!(first.ambiguity_id, second.ambiguity_id);
        for (observation, channel) in [(first, -7), (second, 6)] {
            let scale = arc.wavelengths_m[&observation.ambiguity_id];
            assert!(
                (scale - wavelength_m(channel)).abs() < 1e-12,
                "{}: {scale} for channel {channel}",
                observation.ambiguity_id
            );
            assert!((observation.phase_m / scale - 100.0).abs() < 1e-9);
        }
        for epoch in &arc.epochs {
            assert_eq!(epoch.base[0].ambiguity_id, epoch.rover[0].ambiguity_id);
        }
    }

    #[test]
    fn a_glonass_channel_change_starts_a_new_dual_frequency_ambiguity() {
        let text = channel_change_text(
            "R    4 C1C L1C C2C L2C",
            &format!(
                "R01{:14.3}  {:14.3}  {:14.3}  {:14.3}",
                20_000_000.0, 100.0, 20_000_001.0, 90.0
            ),
        );
        let obs = RinexObs::parse(&text).expect("parse");
        let options = RtkRinexDualArcOptions::new(
            vec![RtkRinexDualSignalPair {
                system: GnssSystem::Glonass,
                code1_observable: "C1C".to_string(),
                phase1_observable: "L1C".to_string(),
                code2_observable: "C2C".to_string(),
                phase2_observable: "L2C".to_string(),
            }],
            None,
            1,
            false,
        );
        let arc =
            build_dual_frequency_rinex_rtk_arc(&FixedSource, &obs, &obs, &options).expect("arc");
        assert_eq!(arc.epochs.len(), 2);
        let first = &arc.epochs[0].observations[0];
        let second = &arc.epochs[1].observations[0];
        assert_ne!(first.base.ambiguity_id, second.base.ambiguity_id);
        assert_ne!(first.base.f1_hz, second.base.f1_hz);
        for observation in [first, second] {
            assert_eq!(
                observation.base.ambiguity_id,
                observation.rover.ambiguity_id
            );
        }
    }
    /// R01 on channel -7 and R28 on the channel 7 real IGS headers state for it,
    /// both observed at one epoch.
    fn r28_channel_7_text(types: &str, r01: &str, r28: &str) -> String {
        [
            header_line(
                "     3.05           OBSERVATION DATA    R (GLONASS)",
                "RINEX VERSION / TYPE",
            ),
            header_line(types, "SYS / # / OBS TYPES"),
            header_line("  2 R01 -7 R28  7", "GLONASS SLOT / FRQ #"),
            header_line("", "END OF HEADER"),
            "> 2020 01 01 00 00  0.0000000  0  2".to_string(),
            r01.to_string(),
            r28.to_string(),
        ]
        .join("\n")
    }

    /// A satellite whose phase observable has no carrier frequency - R28 on
    /// channel 7, outside the FDMA allocation - is left out of its epoch and
    /// reported for each receiver, and the arc is built from the other
    /// satellites. It used to fail the whole arc with `MissingFrequency`.
    #[test]
    fn an_unresolved_carrier_leaves_out_one_satellite_not_the_arc() {
        let record = |sat: &str| format!("{sat}{:14.3}  {:14.3}", 20_000_000.0, 100.0);
        let text = r28_channel_7_text("R    2 C1C L1C", &record("R01"), &record("R28"));
        let obs = RinexObs::parse(&text).expect("parse");
        assert_eq!(obs.header().glonass_slots.get(&28).copied(), Some(7));
        let options = RtkRinexArcOptions::new(
            vec![RtkRinexSignalPair {
                system: GnssSystem::Glonass,
                code_observable: "C1C".to_string(),
                phase_observable: "L1C".to_string(),
            }],
            None,
            1,
            false,
        );
        let arc = build_rinex_rtk_arc(&FixedSource, &obs, &obs, &options)
            .expect("the arc is built from R01");
        assert_eq!(arc.epochs.len(), 1);
        assert_eq!(arc.epochs[0].base.len(), 1);
        assert_eq!(arc.epochs[0].base[0].ambiguity_id, "R01");
        let expected: Vec<RtkRinexUnresolvedCarrier> =
            [RtkRinexReceiver::Base, RtkRinexReceiver::Rover]
                .into_iter()
                .map(|receiver| RtkRinexUnresolvedCarrier {
                    receiver,
                    epoch_index: 0,
                    satellite_id: "R28".to_string(),
                    observable_code: "L1C".to_string(),
                })
                .collect();
        assert_eq!(arc.unresolved_carriers, expected);

        let dual_record = |sat: &str| {
            format!(
                "{sat}{:14.3}  {:14.3}  {:14.3}  {:14.3}",
                20_000_000.0, 100.0, 20_000_001.0, 90.0
            )
        };
        let text = r28_channel_7_text(
            "R    4 C1C L1C C2C L2C",
            &dual_record("R01"),
            &dual_record("R28"),
        );
        let obs = RinexObs::parse(&text).expect("parse");
        let options = RtkRinexDualArcOptions::new(
            vec![RtkRinexDualSignalPair {
                system: GnssSystem::Glonass,
                code1_observable: "C1C".to_string(),
                phase1_observable: "L1C".to_string(),
                code2_observable: "C2C".to_string(),
                phase2_observable: "L2C".to_string(),
            }],
            None,
            1,
            false,
        );
        let arc = build_dual_frequency_rinex_rtk_arc(&FixedSource, &obs, &obs, &options)
            .expect("the dual arc is built from R01");
        assert_eq!(arc.epochs.len(), 1);
        assert_eq!(arc.epochs[0].observations.len(), 1);
        assert_eq!(arc.epochs[0].observations[0].base.ambiguity_id, "R01");
        let reported: Vec<(RtkRinexReceiver, &str, &str)> = arc
            .unresolved_carriers
            .iter()
            .map(|item| {
                (
                    item.receiver,
                    item.satellite_id.as_str(),
                    item.observable_code.as_str(),
                )
            })
            .collect();
        assert_eq!(
            reported,
            vec![
                (RtkRinexReceiver::Base, "R28", "L1C"),
                (RtkRinexReceiver::Base, "R28", "L2C"),
                (RtkRinexReceiver::Rover, "R28", "L1C"),
                (RtkRinexReceiver::Rover, "R28", "L2C"),
            ]
        );
    }

    /// A satellite whose first configured pair has no carrier frequency is
    /// formed from a later pair whose carrier resolves, and is not reported:
    /// R28 on channel 7 has no FDMA G1 carrier, but its CDMA G3 `L3Q` needs no
    /// channel. Only a satellite no configured pair resolves for is reported.
    #[test]
    fn a_later_pair_with_a_resolvable_carrier_is_used_before_reporting() {
        let record = |sat: &str| {
            format!(
                "{sat}{:14.3}  {:14.3}  {:14.3}  {:14.3}",
                20_000_000.0, 100.0, 20_000_002.0, 80.0
            )
        };
        let text = r28_channel_7_text("R    4 C1C L1C C3Q L3Q", &record("R01"), &record("R28"));
        let obs = RinexObs::parse(&text).expect("parse");
        let options = RtkRinexArcOptions::new(
            vec![
                RtkRinexSignalPair {
                    system: GnssSystem::Glonass,
                    code_observable: "C1C".to_string(),
                    phase_observable: "L1C".to_string(),
                },
                RtkRinexSignalPair {
                    system: GnssSystem::Glonass,
                    code_observable: "C3Q".to_string(),
                    phase_observable: "L3Q".to_string(),
                },
            ],
            None,
            1,
            false,
        );
        let arc = build_rinex_rtk_arc(&FixedSource, &obs, &obs, &options).expect("arc");
        assert!(
            arc.unresolved_carriers.is_empty(),
            "{:?}",
            arc.unresolved_carriers
        );
        let base = &arc.epochs[0].base;
        let satellites: Vec<&str> = base.iter().map(|o| o.satellite_id.as_str()).collect();
        assert_eq!(satellites, ["R01", "R28"]);
        let r01_scale = arc.wavelengths_m[&base[0].ambiguity_id];
        assert!((r01_scale - wavelength_m(-7)).abs() < 1e-12);
        let r28_scale = arc.wavelengths_m[&base[1].ambiguity_id];
        let g3_wavelength = C_M_S
            / observation_frequency_hz(GnssSystem::Glonass, "L3Q", 3.05, None)
                .expect("frequency")
                .expect("GLONASS G3 is a fixed CDMA carrier");
        assert!((r28_scale - g3_wavelength).abs() < 1e-12);
        assert!((base[1].phase_m / r28_scale - 80.0).abs() < 1e-9);
    }

    /// R01 on each listed channel in turn, one epoch every ten seconds, each
    /// epoch after a flag 4 event re-declaring the slot.
    fn channel_sequence_text(types: &str, record: &str, channels: &[i8]) -> String {
        let mut lines = vec![
            header_line(
                "     3.05           OBSERVATION DATA    R (GLONASS)",
                "RINEX VERSION / TYPE",
            ),
            header_line(types, "SYS / # / OBS TYPES"),
            header_line(
                &format!("  1 R01 {:2}", channels[0]),
                "GLONASS SLOT / FRQ #",
            ),
            header_line("", "END OF HEADER"),
        ];
        for (index, channel) in channels.iter().enumerate() {
            lines.push(format!(">{:30}4  1", ""));
            lines.push(header_line(
                &format!("  1 R01 {channel:2}"),
                "GLONASS SLOT / FRQ #",
            ));
            lines.push(format!("> 2020 01 01 00 00 {:2}.0000000  0  1", index * 10));
            lines.push(record.to_string());
        }
        lines.join("\n")
    }

    #[test]
    fn a_channel_change_in_an_epoch_left_out_still_starts_a_new_ambiguity() {
        // The base changes channel at the middle epoch and back; the rover does
        // not, so the middle epoch holds no single difference and is left out.
        // The ambiguity after it is not the one before it. Channel 7 is outside
        // the FDMA allocation, so the middle carrier has no frequency at all:
        // that unresolved stretch breaks the ambiguity as a channel change does,
        // rather than letting the return to -7 continue the first arc.
        for base_channels in [[-7i8, 6, -7], [-7, 7, -7]] {
            let single_record = format!("R01{:14.3}  {:14.3}", 20_000_000.0, 100.0);
            let base = RinexObs::parse(&channel_sequence_text(
                "R    2 C1C L1C",
                &single_record,
                &base_channels,
            ))
            .expect("parse base");
            let rover = RinexObs::parse(&channel_sequence_text(
                "R    2 C1C L1C",
                &single_record,
                &[-7, -7, -7],
            ))
            .expect("parse rover");
            let options = RtkRinexArcOptions::new(
                vec![RtkRinexSignalPair {
                    system: GnssSystem::Glonass,
                    code_observable: "C1C".to_string(),
                    phase_observable: "L1C".to_string(),
                }],
                None,
                1,
                false,
            );
            let arc = build_rinex_rtk_arc(&FixedSource, &base, &rover, &options).expect("arc");
            assert_eq!(arc.epochs.len(), 2, "{base_channels:?}");
            assert_ne!(
                arc.epochs[0].base[0].ambiguity_id, arc.epochs[1].base[0].ambiguity_id,
                "{base_channels:?}"
            );
            for epoch in &arc.epochs {
                let scale = arc.wavelengths_m[&epoch.base[0].ambiguity_id];
                assert!((scale - wavelength_m(-7)).abs() < 1e-12);
            }

            let dual_record = format!(
                "R01{:14.3}  {:14.3}  {:14.3}  {:14.3}",
                20_000_000.0, 100.0, 20_000_001.0, 90.0
            );
            let types = "R    4 C1C L1C C2C L2C";
            let base = RinexObs::parse(&channel_sequence_text(types, &dual_record, &base_channels))
                .expect("parse base");
            let rover = RinexObs::parse(&channel_sequence_text(types, &dual_record, &[-7, -7, -7]))
                .expect("parse rover");
            let options = RtkRinexDualArcOptions::new(
                vec![RtkRinexDualSignalPair {
                    system: GnssSystem::Glonass,
                    code1_observable: "C1C".to_string(),
                    phase1_observable: "L1C".to_string(),
                    code2_observable: "C2C".to_string(),
                    phase2_observable: "L2C".to_string(),
                }],
                None,
                1,
                false,
            );
            let arc = build_dual_frequency_rinex_rtk_arc(&FixedSource, &base, &rover, &options)
                .expect("arc");
            assert_eq!(arc.epochs.len(), 2, "{base_channels:?}");
            assert_ne!(
                arc.epochs[0].observations[0].base.ambiguity_id,
                arc.epochs[1].observations[0].base.ambiguity_id,
                "{base_channels:?}"
            );
        }
    }

    /// R01 on each listed channel in turn, one epoch every ten seconds, each
    /// after a flag 4 event re-declaring the slot, its C1C blank where `blank`
    /// says.
    fn channel_code_text(channels: &[i8], blank: &[bool]) -> String {
        let mut lines = vec![
            header_line(
                "     3.05           OBSERVATION DATA    R (GLONASS)",
                "RINEX VERSION / TYPE",
            ),
            header_line("R    4 C1C L1C C2C L2C", "SYS / # / OBS TYPES"),
            header_line("", "END OF HEADER"),
        ];
        for (index, (channel, blank)) in channels.iter().zip(blank).enumerate() {
            lines.push(format!(">{:30}4  1", ""));
            lines.push(header_line(
                &format!("  1 R01 {channel:2}"),
                "GLONASS SLOT / FRQ #",
            ));
            lines.push(format!("> 2020 01 01 00 00 {:2}.0000000  0  1", index * 10));
            let code = if *blank {
                format!("{:14}", "")
            } else {
                format!("{:14.3}", 20_000_000.0)
            };
            lines.push(format!(
                "R01{code}  {:14.3}  {:14.3}  {:14.3}",
                100.0, 20_000_001.0, 90.0
            ));
        }
        lines.join("\n")
    }

    fn glonass_options() -> (RtkRinexArcOptions, RtkRinexDualArcOptions) {
        (
            RtkRinexArcOptions::new(
                vec![RtkRinexSignalPair {
                    system: GnssSystem::Glonass,
                    code_observable: "C1C".to_string(),
                    phase_observable: "L1C".to_string(),
                }],
                None,
                1,
                false,
            ),
            RtkRinexDualArcOptions::new(
                vec![RtkRinexDualSignalPair {
                    system: GnssSystem::Glonass,
                    code1_observable: "C1C".to_string(),
                    phase1_observable: "L1C".to_string(),
                    code2_observable: "C2C".to_string(),
                    phase2_observable: "L2C".to_string(),
                }],
                None,
                1,
                false,
            ),
        )
    }

    #[test]
    fn a_channel_change_at_an_epoch_with_no_code_still_starts_a_new_ambiguity() {
        // The channel moves to +6 and back on the base (`side` 0), the rover
        // (1) or both (2), where that receiver's C1C is blank. The carrier phase
        // is there, so the carrier changed there, although no measurement can
        // be formed at that epoch.
        let (single, dual) = glonass_options();
        for side in 0..3 {
            let text = |changes: bool| {
                if changes {
                    channel_code_text(&[-7, 6, -7], &[false, true, false])
                } else {
                    channel_code_text(&[-7, -7, -7], &[false, false, false])
                }
            };
            let base = RinexObs::parse(&text(side != 1)).expect("parse base");
            let rover = RinexObs::parse(&text(side != 0)).expect("parse rover");
            let arc = build_rinex_rtk_arc(&FixedSource, &base, &rover, &single).expect("arc");
            assert_eq!(arc.epochs.len(), 2, "side {side}");
            assert_ne!(
                arc.epochs[0].base[0].ambiguity_id, arc.epochs[1].base[0].ambiguity_id,
                "side {side}"
            );
            for epoch in &arc.epochs {
                let scale = arc.wavelengths_m[&epoch.base[0].ambiguity_id];
                assert!((scale - wavelength_m(-7)).abs() < 1e-12, "side {side}");
            }
            let arc = build_dual_frequency_rinex_rtk_arc(&FixedSource, &base, &rover, &dual)
                .expect("arc");
            assert_eq!(arc.epochs.len(), 2, "side {side}");
            assert_ne!(
                arc.epochs[0].observations[0].base.ambiguity_id,
                arc.epochs[1].observations[0].base.ambiguity_id,
                "side {side}"
            );
        }
    }

    #[test]
    fn a_change_of_carrier_phase_observable_starts_a_new_ambiguity() {
        // An event replaces C1C/L1C with C1W/L1W, both pairs configured: the
        // frequency stays and the signal tracked changes.
        let types = |codes: &str| header_line(codes, "SYS / # / OBS TYPES");
        let epoch = |index: usize| {
            format!(
                "> 2020 01 01 00 00 {:2}.0000000  0  1\nG01{:14.3}  {:14.3}  {:14.3}  {:14.3}",
                index * 10,
                20_000_000.0,
                100.0 + index as f64 * 10.0,
                20_000_000.0,
                90.0
            )
        };
        let text = |changes: bool| {
            let mut lines = vec![
                header_line(
                    "     3.05           OBSERVATION DATA    G (GPS)",
                    "RINEX VERSION / TYPE",
                ),
                types("G    4 C1C L1C C2W L2W"),
                header_line("", "END OF HEADER"),
                epoch(0),
            ];
            if changes {
                lines.push(format!(">{:30}4  1", ""));
                lines.push(types("G    4 C1W L1W C2W L2W"));
            }
            lines.push(epoch(1));
            RinexObs::parse(&lines.join("\n")).expect("parse")
        };
        let single = RtkRinexArcOptions::new(
            [("C1C", "L1C"), ("C1W", "L1W")]
                .iter()
                .map(|(code, phase)| RtkRinexSignalPair {
                    system: GnssSystem::Gps,
                    code_observable: (*code).to_string(),
                    phase_observable: (*phase).to_string(),
                })
                .collect(),
            None,
            1,
            false,
        );
        let dual = RtkRinexDualArcOptions::new(
            [("C1C", "L1C"), ("C1W", "L1W")]
                .iter()
                .map(|(code, phase)| RtkRinexDualSignalPair {
                    system: GnssSystem::Gps,
                    code1_observable: (*code).to_string(),
                    phase1_observable: (*phase).to_string(),
                    code2_observable: "C2W".to_string(),
                    phase2_observable: "L2W".to_string(),
                })
                .collect(),
            None,
            1,
            false,
        );
        let (changed, steady) = (text(true), text(false));
        for (what, base, rover) in [
            ("base", &changed, &steady),
            ("rover", &steady, &changed),
            ("both", &changed, &changed),
        ] {
            let arc = build_rinex_rtk_arc(&FixedSource, base, rover, &single).expect("arc");
            assert_eq!(arc.epochs.len(), 2, "{what}");
            assert_ne!(
                arc.epochs[0].base[0].ambiguity_id, arc.epochs[1].base[0].ambiguity_id,
                "{what}"
            );
            let arc =
                build_dual_frequency_rinex_rtk_arc(&FixedSource, base, rover, &dual).expect("arc");
            assert_eq!(arc.epochs.len(), 2, "{what}");
            assert_ne!(
                arc.epochs[0].observations[0].base.ambiguity_id,
                arc.epochs[1].observations[0].base.ambiguity_id,
                "{what}"
            );
        }
        // Without the event the ambiguity continues.
        let arc = build_rinex_rtk_arc(&FixedSource, &steady, &steady, &single).expect("arc");
        assert_eq!(
            arc.epochs[0].base[0].ambiguity_id,
            arc.epochs[1].base[0].ambiguity_id
        );
    }

    /// One GPS epoch's `C1C L1C C1W L1W` values and whether L1C lost lock;
    /// `C2W` and `L2W` are always present.
    #[derive(Clone, Copy)]
    struct GpsValues {
        c1c: Option<f64>,
        l1c: Option<f64>,
        l1c_lost: bool,
        c1w: Option<f64>,
        l1w: Option<f64>,
    }

    const STEADY: GpsValues = GpsValues {
        c1c: Some(20_000_000.0),
        l1c: Some(100.0),
        l1c_lost: false,
        c1w: Some(20_000_000.0),
        l1w: Some(200.0),
    };

    /// G01 over the listed epochs, ten seconds apart.
    fn gps_values_obs(epochs: &[GpsValues]) -> RinexObs {
        let field = |value: Option<f64>, lost: bool| {
            value.map_or_else(
                || " ".repeat(16),
                |value| format!("{value:14.3}{} ", if lost { '1' } else { ' ' }),
            )
        };
        let mut lines = vec![
            header_line(
                "     3.05           OBSERVATION DATA    G (GPS)",
                "RINEX VERSION / TYPE",
            ),
            header_line("G    6 C1C L1C C1W L1W C2W L2W", "SYS / # / OBS TYPES"),
            header_line("", "END OF HEADER"),
        ];
        for (index, values) in epochs.iter().enumerate() {
            lines.push(format!("> 2020 01 01 00 00 {:2}.0000000  0  1", index * 10));
            lines.push(
                format!(
                    "G01{}{}{}{}{}{}",
                    field(values.c1c, false),
                    field(values.l1c, values.l1c_lost),
                    field(values.c1w, false),
                    field(values.l1w, false),
                    field(Some(20_000_000.0), false),
                    field(Some(90.0), false)
                )
                .trim_end()
                .to_string(),
            );
        }
        RinexObs::parse(&lines.join("\n")).expect("parse")
    }

    fn gps_options(pairs: &[(&str, &str)]) -> (RtkRinexArcOptions, RtkRinexDualArcOptions) {
        (
            RtkRinexArcOptions::new(
                pairs
                    .iter()
                    .map(|(code, phase)| RtkRinexSignalPair {
                        system: GnssSystem::Gps,
                        code_observable: (*code).to_string(),
                        phase_observable: (*phase).to_string(),
                    })
                    .collect(),
                None,
                1,
                false,
            ),
            RtkRinexDualArcOptions::new(
                pairs
                    .iter()
                    .map(|(code, phase)| RtkRinexDualSignalPair {
                        system: GnssSystem::Gps,
                        code1_observable: (*code).to_string(),
                        phase1_observable: (*phase).to_string(),
                        code2_observable: "C2W".to_string(),
                        phase2_observable: "L2W".to_string(),
                    })
                    .collect(),
                None,
                1,
                false,
            ),
        )
    }

    /// The ambiguity ids a single-frequency arc's epochs carry for G01 once
    /// its cycle slips split them.
    fn split_ids(arc: &RtkRinexArc) -> Vec<String> {
        let epochs: Vec<crate::rtk::CycleSlipEpoch> = arc
            .epochs
            .iter()
            .map(|epoch| {
                let convert = |observations: &[RtkArcObservation]| {
                    observations
                        .iter()
                        .map(|observation| crate::rtk::CycleSlipObservation {
                            satellite_id: observation.satellite_id.clone(),
                            ambiguity_id: observation.ambiguity_id.clone(),
                            code_m: observation.code_m,
                            phase_m: observation.phase_m,
                            lli: observation.lli,
                        })
                        .collect()
                };
                crate::rtk::CycleSlipEpoch {
                    base_observations: convert(&epoch.base),
                    rover_observations: convert(&epoch.rover),
                }
            })
            .collect();
        crate::rtk::prepare_cycle_slip_baseline_epochs(
            &epochs,
            crate::rtk::CycleSlipPolicy::SplitArc,
        )
        .expect("prepare")
        .epochs
        .iter()
        .map(|epoch| {
            crate::rtk::sd_ambiguity_token(
                "G01",
                &epoch.base_observations[0].ambiguity_id,
                &epoch.rover_observations[0].ambiguity_id,
            )
        })
        .collect()
    }

    #[test]
    fn a_loss_of_lock_at_an_epoch_left_out_reaches_the_next_observation() {
        // The rover's L1C loses lock at the middle epoch, where its C1C is
        // blank; the base tracks on.
        let base = gps_values_obs(&[STEADY; 3]);
        let lost = GpsValues {
            c1c: None,
            l1c: Some(103.0),
            l1c_lost: true,
            ..STEADY
        };
        let after = GpsValues {
            l1c: Some(103.0),
            ..STEADY
        };
        let rover = gps_values_obs(&[STEADY, lost, after]);

        // With only C1C/L1C configured the middle epoch holds no measurement
        // and is left out; the loss of lock reaches the next L1C observation.
        let (single, dual) = gps_options(&[("C1C", "L1C")]);
        let arc = build_rinex_rtk_arc(&FixedSource, &base, &rover, &single).expect("arc");
        assert_eq!(arc.epochs.len(), 2);
        assert!(
            arc.epochs[1].rover[0].lli.is_some_and(|lli| lli & 1 == 1),
            "{:?}",
            arc.epochs[1].rover[0].lli
        );
        let ids = split_ids(&arc);
        assert_ne!(ids[0], ids[1], "{ids:?}");
        let arc =
            build_dual_frequency_rinex_rtk_arc(&FixedSource, &base, &rover, &dual).expect("arc");
        assert_eq!(arc.epochs.len(), 2);
        assert!(
            arc.epochs[1].observations[0]
                .rover
                .lli1
                .is_some_and(|lli| lli & 1 == 1),
            "{:?}",
            arc.epochs[1].observations[0].rover.lli1
        );

        // With C1W/L1W configured after it, the middle epoch's rover measurement
        // is on L1W, another carrier, and the return to L1C does not continue
        // the first epoch's ambiguity.
        let (single, dual) = gps_options(&[("C1C", "L1C"), ("C1W", "L1W")]);
        let arc = build_rinex_rtk_arc(&FixedSource, &base, &rover, &single).expect("arc");
        assert_eq!(arc.epochs.len(), 3);
        let ids = split_ids(&arc);
        assert_ne!(ids[0], ids[1], "{ids:?}");
        assert_ne!(ids[1], ids[2], "{ids:?}");
        assert_ne!(ids[0], ids[2], "{ids:?}");
        let arc =
            build_dual_frequency_rinex_rtk_arc(&FixedSource, &base, &rover, &dual).expect("arc");
        assert_eq!(arc.epochs.len(), 3);
        let ids: Vec<&str> = arc
            .epochs
            .iter()
            .map(|epoch| epoch.observations[0].rover.ambiguity_id.as_str())
            .collect();
        assert_ne!(ids[0], ids[2], "{ids:?}");
    }

    #[test]
    fn a_configured_code_fallback_on_the_same_carrier_keeps_every_measurement() {
        // WTZR00DEU holds C1C and L1C for GPS and no C1W. A configuration that
        // prefers C1W and falls back to C1C, both with L1C, gives what C1C/L1C
        // alone gives.
        let obs = RinexObs::parse(include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/obs/WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx"
        )))
        .expect("parse");
        let (plain, plain_dual) = gps_options(&[("C1C", "L1C")]);
        let (fallback, fallback_dual) = gps_options(&[("C1W", "L1C"), ("C1C", "L1C")]);
        let expected = build_rinex_rtk_arc(&FixedSource, &obs, &obs, &plain).expect("arc");
        assert_eq!(expected.epochs.len(), 120);
        assert_eq!(
            build_rinex_rtk_arc(&FixedSource, &obs, &obs, &fallback).expect("arc"),
            expected
        );
        let expected =
            build_dual_frequency_rinex_rtk_arc(&FixedSource, &obs, &obs, &plain_dual).expect("arc");
        assert_eq!(expected.epochs.len(), 120);
        assert_eq!(
            build_dual_frequency_rinex_rtk_arc(&FixedSource, &obs, &obs, &fallback_dual)
                .expect("arc"),
            expected
        );

        // A code missing at one epoch falls back to the next pair on the same
        // carrier, and the ambiguity continues.
        let no_c1w = GpsValues {
            c1w: None,
            ..STEADY
        };
        let obs = gps_values_obs(&[STEADY, no_c1w, STEADY]);
        let (single, dual) = gps_options(&[("C1W", "L1C"), ("C1C", "L1C")]);
        let arc = build_rinex_rtk_arc(&FixedSource, &obs, &obs, &single).expect("arc");
        assert_eq!(arc.epochs.len(), 3);
        assert!(arc
            .epochs
            .iter()
            .all(|epoch| epoch.base[0].ambiguity_id == "G01"));
        let arc = build_dual_frequency_rinex_rtk_arc(&FixedSource, &obs, &obs, &dual).expect("arc");
        assert_eq!(arc.epochs.len(), 3);
        assert!(arc
            .epochs
            .iter()
            .all(|epoch| epoch.observations[0].base.ambiguity_id == "G01"));
    }

    #[test]
    fn a_fallback_to_another_carrier_and_back_starts_a_new_ambiguity_each_time() {
        let no_c1c = GpsValues {
            c1c: None,
            ..STEADY
        };
        let obs = gps_values_obs(&[STEADY, no_c1c, STEADY]);
        let (single, dual) = gps_options(&[("C1C", "L1C"), ("C1W", "L1W")]);
        let arc = build_rinex_rtk_arc(&FixedSource, &obs, &obs, &single).expect("arc");
        assert_eq!(arc.epochs.len(), 3);
        let ids: Vec<&str> = arc
            .epochs
            .iter()
            .map(|epoch| epoch.base[0].ambiguity_id.as_str())
            .collect();
        assert_eq!(ids, ["G01", "G01~freq2", "G01~freq3"]);
        // The middle epoch's measurement is L1W's.
        let wavelength = C_M_S / crate::constants::F_L1_HZ;
        assert!((arc.epochs[1].base[0].phase_m - 200.0 * wavelength).abs() < 1e-6);
        let arc = build_dual_frequency_rinex_rtk_arc(&FixedSource, &obs, &obs, &dual).expect("arc");
        let ids: Vec<&str> = arc
            .epochs
            .iter()
            .map(|epoch| epoch.observations[0].base.ambiguity_id.as_str())
            .collect();
        assert_eq!(ids, ["G01", "G01~freq2", "G01~freq3"]);
        assert_eq!(arc.epochs[1].observations[0].base.phi1_cycles, 200.0);
    }
}
