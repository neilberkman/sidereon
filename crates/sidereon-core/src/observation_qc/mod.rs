//! RINEX observation-file quality-control rollups.
//!
//! This module works from an already parsed [`RinexObs`] product. It does not
//! parse, repair, or resample files; it reports the completeness and signal
//! indicators a caller needs before choosing solver inputs.

use std::collections::{BTreeMap, BTreeSet};

mod multipath;
mod report_html;
mod report_text;

pub use multipath::{
    arc_multipath_rms, mp_combination, multipath_stats, MpStats, MultipathReport,
    SatelliteMultipathQc, SystemMultipathQc,
};
pub use report_html::render_html;
pub use report_text::render_text;

use crate::frequencies::{default_iono_free_pair, frequency_hz, rinex_observation_frequency_hz};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::precise_positioning::{
    detect_cycle_slips as detect_dual_frequency_cycle_slips, CycleSlipConfig, DualFrequencyEpoch,
    DualFrequencyObservation,
};
use crate::rinex::observations::{ObsEpochTime, RinexObs};
use crate::rinex_common::{
    dominant_obs_interval_s, obs_epoch_seconds, time_scale_rinex_label, usable_obs_interval_s,
};
use crate::rinex_qc::{lint_obs, Severity};

/// Default receiver-clock jump threshold, in seconds.
pub const DEFAULT_CLOCK_JUMP_THRESHOLD_S: f64 = 0.0005;

/// Options controlling RINEX observation QC aggregation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ObservationQcOptions {
    /// Override the header `INTERVAL` value when detecting missing epochs.
    pub interval_override_s: Option<f64>,
    /// Minimum `delta / interval` ratio that is treated as a data gap.
    pub gap_factor: f64,
    /// Minimum de-trended receiver-clock step treated as a clock jump.
    pub clock_jump_threshold_s: f64,
}

impl Default for ObservationQcOptions {
    fn default() -> Self {
        Self {
            interval_override_s: None,
            gap_factor: 1.5,
            clock_jump_threshold_s: DEFAULT_CLOCK_JUMP_THRESHOLD_S,
        }
    }
}

/// Error returned when QC options are invalid.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
pub enum ObservationQcError {
    /// The supplied nominal interval was zero, negative, or non-finite.
    #[error("invalid observation QC interval: must be finite and positive")]
    InvalidInterval,
    /// The supplied gap factor was not finite and greater than one.
    #[error("invalid observation QC gap factor: must be finite and greater than one")]
    InvalidGapFactor,
    /// The supplied clock-jump threshold was zero, negative, or non-finite.
    #[error("invalid observation QC clock-jump threshold: must be finite and positive")]
    InvalidClockJumpThreshold,
}

/// Source of the interval used for gap detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum IntervalSource {
    /// Caller override.
    Override,
    /// Header `INTERVAL`.
    Header,
    /// Modal positive epoch delta inferred from the body.
    Inferred,
    /// Not enough positive epoch deltas were available.
    Unresolved,
}

/// Non-fatal QC note.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum ObservationQcNote {
    /// Adjacent observation epochs were duplicate or out of order.
    NonMonotonicEpoch {
        /// Zero-based index of the later epoch in the observation sequence
        /// that failed strict monotonicity (`delta_t <= 0.0`).
        epoch_index: usize,
    },
    /// No interval could be resolved.
    IntervalUnresolved,
}

/// Aggregate QC report for one parsed RINEX observation file.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ObservationQcReport {
    /// Header metadata copied from the source observation product.
    pub header: ObservationQcHeader,
    /// Total number of epoch records retained by the parser, including events.
    pub total_epoch_records: usize,
    /// Count of normal observation epochs (`flag == 0`) and power-failure
    /// observation epochs (`flag == 1`).
    pub observation_epochs: usize,
    /// Count of non-observation event records (`flag > 1`).
    pub event_records: usize,
    /// Count of observation epochs marked as power-failure epochs (`flag == 1`).
    pub power_failure_epochs: usize,
    /// Count of malformed records skipped by the RINEX observation parser.
    pub skipped_records: usize,
    /// Interval used for gap detection.
    pub interval_s: Option<f64>,
    /// Where `interval_s` came from.
    pub interval_source: IntervalSource,
    /// Estimated number of missing nominal epochs across all detected gaps.
    pub missing_epochs: usize,
    /// Gaps detected from adjacent observation epochs and the nominal interval.
    pub data_gaps: Vec<ObservationDataGap>,
    /// Millisecond-scale receiver-clock jumps detected from epoch clock offsets.
    pub clock_jumps: Vec<ClockJump>,
    /// Aggregate dual-frequency cycle-slip counts.
    pub cycle_slips: CycleSlipQc,
    /// MP1/MP2 teqc moving-average multipath RMS.
    pub multipath: MultipathReport,
    /// Per-constellation observation completeness and gap rollups.
    pub systems: Vec<SystemObservationQc>,
    /// Per-satellite observation completeness.
    pub satellites: Vec<SatelliteObservationQc>,
    /// Per-satellite, per-code observation completeness and SSI statistics.
    pub satellite_signals: Vec<SatelliteSignalQc>,
    /// Per-system, per-code observation completeness and SSI statistics.
    pub system_signals: Vec<SystemSignalQc>,
    /// RINEX lint findings summarized for report rendering.
    pub lint_findings: Vec<ObservationQcFinding>,
    /// Non-fatal QC notes.
    pub notes: Vec<ObservationQcNote>,
}

/// Source observation header fields rendered in the QC report.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ObservationQcHeader {
    /// Marker name.
    pub marker_name: Option<String>,
    /// Marker number.
    pub marker_number: Option<String>,
    /// Marker type.
    pub marker_type: Option<String>,
    /// Receiver information.
    pub receiver: Option<ObservationQcReceiver>,
    /// Antenna information.
    pub antenna: Option<ObservationQcAntenna>,
    /// Approximate receiver ECEF position, meters.
    pub approx_position_m: Option<[f64; 3]>,
    /// Antenna height/east/north offset, meters.
    pub antenna_delta_hen_m: Option<[f64; 3]>,
    /// Time of first observation.
    pub time_of_first_obs: Option<ObservationQcTime>,
    /// Time of last observation.
    pub time_of_last_obs: Option<ObservationQcTime>,
    /// Duration covered by retained observation epochs, seconds.
    pub duration_s: Option<f64>,
}

/// Receiver identity fields copied from `REC # / TYPE / VERS`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ObservationQcReceiver {
    /// Receiver serial number.
    pub number: String,
    /// Receiver type.
    pub receiver_type: String,
    /// Receiver firmware/version.
    pub version: String,
}

/// Antenna identity fields copied from `ANT # / TYPE`.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ObservationQcAntenna {
    /// Antenna serial number.
    pub number: String,
    /// Antenna type.
    pub antenna_type: String,
}

/// Observation epoch with an optional RINEX time-system label.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ObservationQcTime {
    /// Civil epoch.
    pub epoch: ObsEpochTime,
    /// RINEX time-system label, when declared in the source header.
    pub time_scale: Option<String>,
}

/// Compact lint finding retained by the QC report.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ObservationQcFinding {
    /// Stable finding code.
    pub code: String,
    /// Finding severity.
    pub severity: Severity,
    /// RINEX specification or QC policy reference.
    pub spec_ref: String,
}

/// One detected gap between adjacent observation epochs.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct ObservationDataGap {
    /// Epoch immediately before the gap.
    pub start_epoch: ObsEpochTime,
    /// Epoch immediately after the gap.
    pub end_epoch: ObsEpochTime,
    /// Nominal interval used for the estimate.
    pub nominal_interval_s: f64,
    /// Observed delta between the two retained epochs.
    pub observed_delta_s: f64,
    /// Estimated missing nominal epochs between `start_epoch` and `end_epoch`.
    pub missing_epochs: usize,
}

/// One detected receiver-clock jump.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct ClockJump {
    /// Epoch index in the parsed observation stream where the jump appears.
    pub epoch_index: usize,
    /// Epoch where the jump appears.
    pub epoch: ObsEpochTime,
    /// De-trended receiver-clock step, in seconds.
    pub delta_s: f64,
}

/// Aggregate cycle-slip counts from the dual-frequency detector.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize)]
pub struct CycleSlipQc {
    /// Total dual-frequency observations passed to the detector.
    pub observations: usize,
    /// Total detector-flagged cycle slips.
    pub total_slips: usize,
    /// Dual-frequency observations per flagged slip.
    pub observations_per_slip: Option<f64>,
    /// Per-constellation detector counts.
    pub by_system: Vec<SystemCycleSlipQc>,
}

/// Per-constellation cycle-slip counts.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SystemCycleSlipQc {
    /// GNSS constellation.
    pub system: GnssSystem,
    /// Dual-frequency observations passed to the detector for this constellation.
    pub observations: usize,
    /// Detector-flagged cycle slips for this constellation.
    pub slips: usize,
    /// Dual-frequency observations per flagged slip for this constellation.
    pub observations_per_slip: Option<f64>,
}

/// Per-constellation observation completeness and gap counts.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SystemObservationQc {
    /// GNSS constellation.
    pub system: GnssSystem,
    /// Distinct satellites with at least one non-blank observation.
    pub satellites_seen: usize,
    /// Epochs where this constellation has at least one non-blank observation.
    pub epochs_with_observations: usize,
    /// Non-blank observation values.
    pub value_observations: usize,
    /// Retained observation slots in satellite records for this constellation.
    pub expected_observations: usize,
    /// `value_observations / expected_observations`, when defined.
    pub completeness_ratio: Option<f64>,
    /// Number of per-constellation data gaps.
    pub gap_count: usize,
    /// Missing nominal time across per-constellation gaps, seconds.
    pub total_gap_s: f64,
}

/// Per-satellite observation counts.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SatelliteObservationQc {
    /// Satellite id.
    pub satellite: GnssSatelliteId,
    /// Epochs where the satellite has at least one non-blank observation value.
    pub epochs_with_observations: usize,
    /// Non-blank observation values across all codes and observation epochs.
    pub value_observations: usize,
}

/// Per-satellite, per-observation-code counts.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SatelliteSignalQc {
    /// Satellite id.
    pub satellite: GnssSatelliteId,
    /// RINEX observation code, e.g. `C1C`, `L1C`, or `S1C`.
    pub code: String,
    /// Non-blank values for this satellite/code pair.
    pub value_observations: usize,
    /// Signal-strength indicator statistics for non-blank values that carried
    /// an SSI digit.
    pub ssi: Option<SsiHistogram>,
    /// Raw S-code statistics when this code is an `S*` observable.
    pub snr: Option<SnrStats>,
}

/// Per-system, per-observation-code counts.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SystemSignalQc {
    /// GNSS constellation.
    pub system: GnssSystem,
    /// RINEX observation code, e.g. `C1C`, `L1C`, or `S1C`.
    pub code: String,
    /// Non-blank values for this system/code pair across satellites.
    pub value_observations: usize,
    /// Signal-strength indicator statistics for non-blank values that carried
    /// an SSI digit.
    pub ssi: Option<SsiHistogram>,
    /// Raw S-code statistics when this code is an `S*` observable.
    pub snr: Option<SnrStats>,
}

/// Histogram over RINEX SSI digits. Index 0 is blank/unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct SsiHistogram {
    /// Counts indexed by SSI digit.
    pub counts: [u64; 10],
}

/// Summary statistics over raw numeric `S*` signal-strength observations.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct SnrStats {
    /// Number of samples.
    pub n: usize,
    /// Arithmetic mean.
    pub mean: f64,
    /// Minimum sample.
    pub min: f64,
    /// Maximum sample.
    pub max: f64,
    /// Sample standard deviation, absent for one sample.
    pub std: Option<f64>,
}

/// Build a QC report with default options.
///
/// A zero, negative, or non-finite source `INTERVAL` is never used as cadence.
/// The report carries the corresponding lint finding and either infers cadence
/// from finite positive epoch deltas or marks it [`IntervalSource::Unresolved`].
pub fn observation_qc(obs: &RinexObs) -> ObservationQcReport {
    observation_qc_validated(obs, ObservationQcOptions::default())
}

/// Build a QC report with explicit options.
///
/// Source metadata follows the same fail-visible behavior as
/// [`observation_qc`]. An unusable caller-supplied interval override is instead
/// rejected as [`ObservationQcError::InvalidInterval`].
pub fn observation_qc_with_options(
    obs: &RinexObs,
    options: ObservationQcOptions,
) -> Result<ObservationQcReport, ObservationQcError> {
    validate_options(options)?;
    Ok(observation_qc_validated(obs, options))
}

fn observation_qc_validated(obs: &RinexObs, options: ObservationQcOptions) -> ObservationQcReport {
    let mut satellites: BTreeMap<GnssSatelliteId, SatelliteAccum> = BTreeMap::new();
    let mut systems: BTreeMap<GnssSystem, SystemObservationAccum> = BTreeMap::new();
    let mut satellite_signals: BTreeMap<(GnssSatelliteId, String), SignalAccum> = BTreeMap::new();
    let mut system_signals: BTreeMap<(GnssSystem, String), SignalAccum> = BTreeMap::new();
    let mut observation_epoch_times = Vec::new();
    let mut system_epoch_times: BTreeMap<GnssSystem, Vec<ObsEpochTime>> = BTreeMap::new();

    let mut observation_epochs = 0;
    let mut event_records = 0;
    let mut power_failure_epochs = 0;

    for epoch in obs.epochs() {
        if epoch.flag > 1 {
            event_records += 1;
            continue;
        }

        observation_epochs += 1;
        if epoch.flag == 1 {
            power_failure_epochs += 1;
        }
        observation_epoch_times.push(epoch.epoch);

        let mut epoch_systems = BTreeSet::new();
        for (satellite, values) in &epoch.sats {
            let value_observations = values.iter().filter(|value| value.value.is_some()).count();
            let system_acc = systems.entry(satellite.system).or_default();
            system_acc.expected_observations += values.len();
            system_acc.value_observations += value_observations;

            if value_observations == 0 {
                continue;
            }

            system_acc.satellites.insert(*satellite);
            epoch_systems.insert(satellite.system);

            let satellite_acc = satellites.entry(*satellite).or_default();
            satellite_acc.epochs_with_observations += 1;
            satellite_acc.value_observations += value_observations;

            let Some(codes) = obs.header().obs_codes.get(&satellite.system) else {
                continue;
            };

            for (index, value) in values.iter().enumerate() {
                if value.value.is_none() {
                    continue;
                }

                let Some(code) = codes.get(index) else {
                    continue;
                };

                let sat_signal = satellite_signals
                    .entry((*satellite, code.clone()))
                    .or_default();
                sat_signal.add(code, value.value, value.ssi);

                let sys_signal = system_signals
                    .entry((satellite.system, code.clone()))
                    .or_default();
                sys_signal.add(code, value.value, value.ssi);
            }
        }

        for system in epoch_systems {
            systems.entry(system).or_default().epochs_with_observations += 1;
            system_epoch_times
                .entry(system)
                .or_default()
                .push(epoch.epoch);
        }
    }

    let mut notes = non_monotonic_notes(&observation_epoch_times);
    let (interval_s, interval_source) =
        resolve_interval(obs, options, &observation_epoch_times, &mut notes);
    let data_gaps = detect_gaps(options, &observation_epoch_times, interval_s);
    let missing_epochs = data_gaps
        .iter()
        .map(|gap| gap.missing_epochs)
        .fold(0_usize, usize::saturating_add);
    let clock_jumps = detect_clock_jumps(obs, options.clock_jump_threshold_s);
    let cycle_slips = aggregate_cycle_slips(obs);
    let multipath = multipath_stats(obs, &CycleSlipConfig::default());
    let systems = finish_system_observation_qc(systems, &system_epoch_times, options, interval_s);

    ObservationQcReport {
        header: observation_qc_header(obs, &observation_epoch_times),
        total_epoch_records: obs.epochs().len(),
        observation_epochs,
        event_records,
        power_failure_epochs,
        skipped_records: obs.skipped_records,
        interval_s,
        interval_source,
        missing_epochs,
        data_gaps,
        clock_jumps,
        cycle_slips,
        multipath,
        systems,
        satellites: satellites
            .into_iter()
            .map(|(satellite, acc)| SatelliteObservationQc {
                satellite,
                epochs_with_observations: acc.epochs_with_observations,
                value_observations: acc.value_observations,
            })
            .collect(),
        satellite_signals: satellite_signals
            .into_iter()
            .map(|((satellite, code), acc)| SatelliteSignalQc {
                satellite,
                code,
                value_observations: acc.value_observations,
                ssi: acc.ssi.finish(),
                snr: acc.snr.finish(),
            })
            .collect(),
        system_signals: system_signals
            .into_iter()
            .map(|((system, code), acc)| SystemSignalQc {
                system,
                code,
                value_observations: acc.value_observations,
                ssi: acc.ssi.finish(),
                snr: acc.snr.finish(),
            })
            .collect(),
        lint_findings: observation_qc_findings(obs),
        notes,
    }
}

fn validate_options(options: ObservationQcOptions) -> Result<(), ObservationQcError> {
    if !options.gap_factor.is_finite() || options.gap_factor <= 1.0 {
        return Err(ObservationQcError::InvalidGapFactor);
    }
    if !positive_finite(options.clock_jump_threshold_s) {
        return Err(ObservationQcError::InvalidClockJumpThreshold);
    }

    if let Some(interval_s) = options.interval_override_s {
        validate_interval(interval_s)?;
    }

    Ok(())
}

fn validate_interval(interval_s: f64) -> Result<(), ObservationQcError> {
    if usable_obs_interval_s(interval_s) {
        Ok(())
    } else {
        Err(ObservationQcError::InvalidInterval)
    }
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn observation_qc_header(
    obs: &RinexObs,
    observation_epoch_times: &[ObsEpochTime],
) -> ObservationQcHeader {
    let header = obs.header();
    let time_of_first_obs = header
        .time_of_first_obs
        .map(stamped_declared_time)
        .or_else(|| observation_epoch_times.first().copied().map(unstamped_time));
    let time_of_last_obs = header
        .time_of_last_obs
        .map(stamped_declared_time)
        .or_else(|| observation_epoch_times.last().copied().map(unstamped_time));
    let duration_s = observation_epoch_times
        .first()
        .zip(observation_epoch_times.last())
        .map(|(first, last)| obs_epoch_seconds(*last) - obs_epoch_seconds(*first))
        .filter(|duration_s| duration_s.is_finite() && *duration_s >= 0.0);

    ObservationQcHeader {
        marker_name: header.marker_name.clone(),
        marker_number: header.marker_number.clone(),
        marker_type: header.marker_type.clone(),
        receiver: header
            .receiver
            .as_ref()
            .map(|receiver| ObservationQcReceiver {
                number: receiver.number.clone(),
                receiver_type: receiver.receiver_type.clone(),
                version: receiver.version.clone(),
            }),
        antenna: header.antenna.as_ref().map(|antenna| ObservationQcAntenna {
            number: antenna.number.clone(),
            antenna_type: antenna.antenna_type.clone(),
        }),
        approx_position_m: header.approx_position_m,
        antenna_delta_hen_m: header.antenna_delta_hen_m,
        time_of_first_obs,
        time_of_last_obs,
        duration_s,
    }
}

fn stamped_declared_time(
    (epoch, scale): (ObsEpochTime, crate::astro::time::model::TimeScale),
) -> ObservationQcTime {
    ObservationQcTime {
        epoch,
        time_scale: time_scale_rinex_label(scale).map(str::to_string),
    }
}

fn unstamped_time(epoch: ObsEpochTime) -> ObservationQcTime {
    ObservationQcTime {
        epoch,
        time_scale: None,
    }
}

fn finish_system_observation_qc(
    systems: BTreeMap<GnssSystem, SystemObservationAccum>,
    system_epoch_times: &BTreeMap<GnssSystem, Vec<ObsEpochTime>>,
    options: ObservationQcOptions,
    interval_s: Option<f64>,
) -> Vec<SystemObservationQc> {
    systems
        .into_iter()
        .map(|(system, acc)| {
            let times = system_epoch_times
                .get(&system)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let gaps = detect_gaps(options, times, interval_s);
            let total_gap_s = gaps
                .iter()
                .map(|gap| gap.missing_epochs as f64 * gap.nominal_interval_s)
                .sum::<f64>();
            let total_gap_s = if total_gap_s == 0.0 { 0.0 } else { total_gap_s };
            SystemObservationQc {
                system,
                satellites_seen: acc.satellites.len(),
                epochs_with_observations: acc.epochs_with_observations,
                value_observations: acc.value_observations,
                expected_observations: acc.expected_observations,
                completeness_ratio: (acc.expected_observations > 0)
                    .then(|| acc.value_observations as f64 / acc.expected_observations as f64),
                gap_count: gaps.len(),
                total_gap_s,
            }
        })
        .collect()
}

fn observation_qc_findings(obs: &RinexObs) -> Vec<ObservationQcFinding> {
    lint_obs(obs)
        .findings
        .into_iter()
        .map(|finding| ObservationQcFinding {
            code: finding.code().to_string(),
            severity: finding.severity(),
            spec_ref: finding.spec_ref().to_string(),
        })
        .collect()
}

fn resolve_interval(
    obs: &RinexObs,
    options: ObservationQcOptions,
    observation_epoch_times: &[ObsEpochTime],
    notes: &mut Vec<ObservationQcNote>,
) -> (Option<f64>, IntervalSource) {
    let Some(interval_s) = options.interval_override_s else {
        if let Some(interval_s) = obs
            .header()
            .interval_s
            .filter(|interval_s| usable_obs_interval_s(*interval_s))
        {
            return (Some(interval_s), IntervalSource::Header);
        }
        if let Some(interval_s) = dominant_obs_interval_s(observation_epoch_times) {
            return (Some(interval_s), IntervalSource::Inferred);
        }
        notes.push(ObservationQcNote::IntervalUnresolved);
        return (None, IntervalSource::Unresolved);
    };
    (Some(interval_s), IntervalSource::Override)
}

fn detect_gaps(
    options: ObservationQcOptions,
    observation_epoch_times: &[ObsEpochTime],
    interval_s: Option<f64>,
) -> Vec<ObservationDataGap> {
    let Some(interval_s) = interval_s else {
        return Vec::new();
    };

    let mut gaps = Vec::new();
    for window in observation_epoch_times.windows(2) {
        let start_epoch = window[0];
        let end_epoch = window[1];
        let observed_delta_s = obs_epoch_seconds(end_epoch) - obs_epoch_seconds(start_epoch);
        if !observed_delta_s.is_finite()
            || observed_delta_s <= 0.0
            || observed_delta_s <= interval_s * options.gap_factor
        {
            continue;
        }

        let missing_epochs = ((observed_delta_s / interval_s).round() - 1.0).max(0.0) as usize;
        gaps.push(ObservationDataGap {
            start_epoch,
            end_epoch,
            nominal_interval_s: interval_s,
            observed_delta_s,
            missing_epochs,
        });
    }

    gaps
}

fn non_monotonic_notes(observation_epoch_times: &[ObsEpochTime]) -> Vec<ObservationQcNote> {
    let mut notes = Vec::new();
    for (idx, window) in observation_epoch_times.windows(2).enumerate() {
        if obs_epoch_seconds(window[1]) - obs_epoch_seconds(window[0]) <= 0.0 {
            notes.push(ObservationQcNote::NonMonotonicEpoch {
                epoch_index: idx + 1,
            });
        }
    }
    notes
}

/// Detect millisecond-scale receiver-clock jumps from RINEX epoch clock offsets.
pub fn detect_clock_jumps(obs: &RinexObs, threshold_s: f64) -> Vec<ClockJump> {
    if !positive_finite(threshold_s) {
        return Vec::new();
    }

    let deltas = clock_offset_deltas(obs);
    let nominal_drift_s_per_s = nominal_clock_drift_s_per_s(&deltas, threshold_s);

    deltas
        .into_iter()
        .filter_map(|delta| {
            let expected_delta_s = nominal_drift_s_per_s * delta.time_delta_s;
            let adjusted_delta_s = delta.raw_delta_s - expected_delta_s;
            millisecond_clock_step(adjusted_delta_s, threshold_s).then_some(ClockJump {
                epoch_index: delta.epoch_index,
                epoch: delta.epoch,
                delta_s: adjusted_delta_s,
            })
        })
        .collect()
}

/// Aggregate cycle-slip counts by reusing the dual-frequency detector.
pub fn aggregate_cycle_slips(obs: &RinexObs) -> CycleSlipQc {
    let epochs = dual_frequency_epochs(obs);
    let mut by_system = BTreeMap::<GnssSystem, CycleSlipAccum>::new();

    for epoch in &epochs {
        for observation in &epoch.observations {
            if let Some(system) = system_from_satellite_token(&observation.satellite_id) {
                by_system.entry(system).or_default().observations += 1;
            }
        }
    }

    let Ok(flags) = detect_dual_frequency_cycle_slips(&epochs, CycleSlipConfig::default()) else {
        return finish_cycle_slip_qc(by_system);
    };

    for epoch in flags {
        for observation in epoch.observations {
            if !observation.slip {
                continue;
            }
            if let Some(system) = system_from_satellite_token(&observation.satellite_id) {
                by_system.entry(system).or_default().slips += 1;
            }
        }
    }

    finish_cycle_slip_qc(by_system)
}

#[derive(Debug, Clone, Copy)]
struct ClockOffsetSample {
    epoch_index: usize,
    epoch: ObsEpochTime,
    epoch_time_s: f64,
    offset_s: f64,
}

#[derive(Debug, Clone, Copy)]
struct ClockOffsetDelta {
    epoch_index: usize,
    epoch: ObsEpochTime,
    time_delta_s: f64,
    raw_delta_s: f64,
}

fn clock_offset_deltas(obs: &RinexObs) -> Vec<ClockOffsetDelta> {
    let mut previous: Option<ClockOffsetSample> = None;
    let mut deltas = Vec::new();

    for (epoch_index, epoch) in obs.epochs().iter().enumerate() {
        if epoch.flag > 1 {
            continue;
        }
        let Some(offset_s) = epoch.rcv_clock_offset_s else {
            continue;
        };
        if !offset_s.is_finite() {
            continue;
        }

        let sample = ClockOffsetSample {
            epoch_index,
            epoch: epoch.epoch,
            epoch_time_s: obs_epoch_seconds(epoch.epoch),
            offset_s,
        };

        if let Some(prev) = previous {
            let time_delta_s = sample.epoch_time_s - prev.epoch_time_s;
            if time_delta_s > 0.0 {
                deltas.push(ClockOffsetDelta {
                    epoch_index: sample.epoch_index,
                    epoch: sample.epoch,
                    time_delta_s,
                    raw_delta_s: sample.offset_s - prev.offset_s,
                });
            }
        }

        previous = Some(sample);
    }

    deltas
}

fn nominal_clock_drift_s_per_s(deltas: &[ClockOffsetDelta], threshold_s: f64) -> f64 {
    let mut slopes = deltas
        .iter()
        .filter(|delta| delta.raw_delta_s.abs() < threshold_s)
        .map(|delta| delta.raw_delta_s / delta.time_delta_s)
        .filter(|slope| slope.is_finite())
        .collect::<Vec<_>>();
    median(&mut slopes).unwrap_or(0.0)
}

fn median(values: &mut [f64]) -> Option<f64> {
    crate::astro::math::robust::median_sorting_in_place(values)
}

fn millisecond_clock_step(delta_s: f64, threshold_s: f64) -> bool {
    if !delta_s.is_finite() || delta_s.abs() < threshold_s {
        return false;
    }

    let step_ms = delta_s.abs() * 1000.0;
    let nearest_ms = step_ms.round();
    if nearest_ms < 1.0 {
        return false;
    }

    let tolerance_ms = (threshold_s * 500.0).min(0.25);
    (step_ms - nearest_ms).abs() <= tolerance_ms
}

#[derive(Debug, Clone, Copy, Default)]
struct CycleSlipAccum {
    observations: usize,
    slips: usize,
}

fn finish_cycle_slip_qc(by_system: BTreeMap<GnssSystem, CycleSlipAccum>) -> CycleSlipQc {
    let observations = by_system.values().map(|acc| acc.observations).sum();
    let total_slips = by_system.values().map(|acc| acc.slips).sum();
    let by_system = by_system
        .into_iter()
        .map(|(system, acc)| SystemCycleSlipQc {
            system,
            observations: acc.observations,
            slips: acc.slips,
            observations_per_slip: observations_per_slip(acc.observations, acc.slips),
        })
        .collect();

    CycleSlipQc {
        observations,
        total_slips,
        observations_per_slip: observations_per_slip(observations, total_slips),
        by_system,
    }
}

fn observations_per_slip(observations: usize, slips: usize) -> Option<f64> {
    (slips > 0).then(|| observations as f64 / slips as f64)
}

fn dual_frequency_epochs(obs: &RinexObs) -> Vec<DualFrequencyEpoch> {
    obs.epochs()
        .iter()
        .filter(|epoch| epoch.flag <= 1)
        .map(|epoch| DualFrequencyEpoch {
            gap_time_s: Some(obs_epoch_seconds(epoch.epoch)),
            observations: epoch
                .sats
                .iter()
                .filter_map(|(satellite, values)| {
                    dual_frequency_observation(obs, *satellite, values)
                })
                .collect(),
        })
        .collect()
}

#[derive(Debug, Clone, Copy)]
struct DualFrequencyBand {
    first_index: usize,
    rinex_band: char,
    frequency_hz: f64,
    pseudorange_m: Option<f64>,
    pseudorange_rank: u8,
    carrier_phase_cyc: Option<f64>,
    lli: Option<i64>,
}

fn dual_frequency_observation(
    obs: &RinexObs,
    satellite: GnssSatelliteId,
    values: &[crate::rinex::observations::ObsValue],
) -> Option<DualFrequencyObservation> {
    dual_frequency_observation_with_pseudorange_selection(
        obs,
        satellite,
        values,
        PseudorangeSelection::HeaderOrder,
    )
}

fn multipath_dual_frequency_observation(
    obs: &RinexObs,
    satellite: GnssSatelliteId,
    values: &[crate::rinex::observations::ObsValue],
) -> Option<DualFrequencyObservation> {
    dual_frequency_observation_with_pseudorange_selection(
        obs,
        satellite,
        values,
        PseudorangeSelection::PreferPreciseCode,
    )
}

#[derive(Debug, Clone, Copy)]
enum PseudorangeSelection {
    HeaderOrder,
    PreferPreciseCode,
}

fn dual_frequency_observation_with_pseudorange_selection(
    obs: &RinexObs,
    satellite: GnssSatelliteId,
    values: &[crate::rinex::observations::ObsValue],
    pseudorange_selection: PseudorangeSelection,
) -> Option<DualFrequencyObservation> {
    let codes = obs.header().obs_codes.get(&satellite.system)?;
    let glonass_channel = obs.header().glonass_slots.get(&satellite.prn).copied();
    let mut bands = Vec::<DualFrequencyBand>::new();

    for (index, (code, value)) in codes.iter().zip(values.iter()).enumerate() {
        let kind = code.as_bytes().first().copied();
        if !matches!(kind, Some(b'C' | b'L')) {
            continue;
        }
        let rinex_band = code.chars().nth(1)?;
        let Some(raw_value) = value.value else {
            continue;
        };
        if !raw_value.is_finite() {
            continue;
        }
        let frequency_hz = rinex_observation_frequency_hz(
            satellite.system,
            code,
            obs.header().version,
            glonass_channel,
        )?;

        let band_index = if let Some(existing) = bands
            .iter()
            .position(|band| same_frequency_hz(band.frequency_hz, frequency_hz))
        {
            existing
        } else {
            bands.push(DualFrequencyBand {
                first_index: index,
                rinex_band,
                frequency_hz,
                pseudorange_m: None,
                pseudorange_rank: u8::MAX,
                carrier_phase_cyc: None,
                lli: None,
            });
            bands.len() - 1
        };

        let band = &mut bands[band_index];
        match kind {
            Some(b'C') if pseudorange_rank(code, pseudorange_selection) < band.pseudorange_rank => {
                band.pseudorange_rank = pseudorange_rank(code, pseudorange_selection);
                band.pseudorange_m = Some(raw_value);
            }
            Some(b'L') if band.carrier_phase_cyc.is_none() => {
                band.carrier_phase_cyc = Some(raw_value);
                band.lli = value.lli.map(i64::from);
            }
            _ => {}
        }
    }

    let mut usable = bands
        .into_iter()
        .filter(|band| band.pseudorange_m.is_some() && band.carrier_phase_cyc.is_some())
        .collect::<Vec<_>>();
    let (band1, band2) = select_dual_frequency_bands(satellite.system, &mut usable)?;

    Some(DualFrequencyObservation {
        satellite_id: satellite.to_string(),
        ambiguity_id: format!(
            "{}:{:.0}:{:.0}",
            satellite, band1.frequency_hz, band2.frequency_hz
        ),
        p1_m: band1.pseudorange_m?,
        p2_m: band2.pseudorange_m?,
        phi1_cyc: band1.carrier_phase_cyc?,
        phi2_cyc: band2.carrier_phase_cyc?,
        f1_hz: band1.frequency_hz,
        f2_hz: band2.frequency_hz,
        lli1: band1.lli,
        lli2: band2.lli,
    })
}

fn select_dual_frequency_bands(
    system: GnssSystem,
    usable: &mut [DualFrequencyBand],
) -> Option<(DualFrequencyBand, DualFrequencyBand)> {
    if let Some(pair) = default_iono_free_pair(system) {
        let pair_f1_hz = frequency_hz(system, pair.band1)?;
        let pair_f2_hz = frequency_hz(system, pair.band2)?;
        let band1 = usable
            .iter()
            .copied()
            .find(|band| same_frequency_hz(band.frequency_hz, pair_f1_hz));
        let band2 = usable
            .iter()
            .copied()
            .find(|band| same_frequency_hz(band.frequency_hz, pair_f2_hz));
        if let (Some(band1), Some(band2)) = (band1, band2) {
            return Some((band1, band2));
        }
    }

    usable.sort_by_key(|band| (rinex_band_sort_key(band.rinex_band), band.first_index));
    let band1 = *usable.first()?;
    let band2 = usable
        .iter()
        .copied()
        .find(|band| !same_frequency_hz(band.frequency_hz, band1.frequency_hz))?;
    Some((band1, band2))
}

fn rinex_band_sort_key(band: char) -> u32 {
    band.to_digit(10).unwrap_or(u32::MAX)
}

fn pseudorange_rank(code: &str, selection: PseudorangeSelection) -> u8 {
    match selection {
        PseudorangeSelection::HeaderOrder => 0,
        PseudorangeSelection::PreferPreciseCode => pseudorange_preference_rank(code),
    }
}

fn pseudorange_preference_rank(code: &str) -> u8 {
    match code.chars().nth(2) {
        Some('W' | 'P' | 'Y' | 'M' | 'N') => 0,
        Some('X') => 1,
        Some('C' | 'S' | 'L') => 2,
        Some(_) => 3,
        None => 4,
    }
}

fn same_frequency_hz(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1.0e-3
}

fn system_from_satellite_token(satellite_id: &str) -> Option<GnssSystem> {
    satellite_id
        .chars()
        .next()
        .and_then(GnssSystem::from_letter)
}

#[derive(Debug, Default)]
struct SystemObservationAccum {
    satellites: BTreeSet<GnssSatelliteId>,
    epochs_with_observations: usize,
    value_observations: usize,
    expected_observations: usize,
}

#[derive(Debug, Default)]
struct SatelliteAccum {
    epochs_with_observations: usize,
    value_observations: usize,
}

#[derive(Debug, Default)]
struct SignalAccum {
    value_observations: usize,
    ssi: SsiAccum,
    snr: SnrAccum,
}

impl SignalAccum {
    fn add(&mut self, code: &str, value: Option<f64>, ssi: Option<u8>) {
        self.value_observations += 1;
        self.ssi.add(ssi);
        if code.starts_with('S') {
            if let Some(value) = value {
                self.snr.add(value);
            }
        }
    }
}

#[derive(Debug, Default)]
struct SsiAccum {
    counts: [u64; 10],
}

impl SsiAccum {
    fn add(&mut self, value: Option<u8>) {
        let idx = value.unwrap_or(0).min(9) as usize;
        self.counts[idx] += 1;
    }

    fn finish(self) -> Option<SsiHistogram> {
        if self.counts.iter().all(|count| *count == 0) {
            return None;
        }

        Some(SsiHistogram {
            counts: self.counts,
        })
    }
}

#[derive(Debug, Default)]
struct SnrAccum {
    samples: Vec<f64>,
}

impl SnrAccum {
    fn add(&mut self, value: f64) {
        self.samples.push(value);
    }

    fn finish(self) -> Option<SnrStats> {
        if self.samples.is_empty() {
            return None;
        }
        let n = self.samples.len();
        let mean = self.samples.iter().sum::<f64>() / n as f64;
        let min = self.samples.iter().copied().fold(f64::INFINITY, f64::min);
        let max = self
            .samples
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let std = (n > 1).then(|| {
            let sum_sq = self
                .samples
                .iter()
                .map(|value| {
                    let residual = *value - mean;
                    residual * residual
                })
                .sum::<f64>();
            (sum_sq / (n - 1) as f64).sqrt()
        });
        Some(SnrStats {
            n,
            mean,
            min,
            max,
            std,
        })
    }
}

#[cfg(test)]
mod tests {
    //! Teqc oracle provenance is retained in `tests/fixtures/qc/teqc_algo0010_2015001_v1_trim.json`;
    //! the MP regression below uses teqc 2019Feb25 on the decoded RINEX 2 fixture.

    use super::*;
    use crate::constants::{C_M_S, F_L1_HZ, F_L2_HZ};
    use crate::crinex;
    use crate::rinex::observations::{ObsEpoch, ObsHeader, ObsValue};
    use serde_json::Value;
    use std::collections::BTreeMap;
    use std::path::PathBuf;

    #[test]
    fn observation_qc_counts_epochs_satellites_signals_and_ssi() {
        let g01 = sat(1);
        let g02 = sat(2);
        let obs = observation_file(vec![
            epoch(
                0,
                0.0,
                0,
                BTreeMap::from([
                    (
                        g01,
                        vec![
                            obs_value(Some(1.0), Some(5)),
                            obs_value(Some(2.0), Some(6)),
                            obs_value(None, None),
                        ],
                    ),
                    (
                        g02,
                        vec![
                            obs_value(Some(10.0), Some(4)),
                            obs_value(None, None),
                            obs_value(None, None),
                        ],
                    ),
                ]),
            ),
            epoch(
                0,
                30.0,
                1,
                BTreeMap::from([(
                    g01,
                    vec![
                        obs_value(Some(3.0), Some(7)),
                        obs_value(None, None),
                        obs_value(Some(9.0), Some(8)),
                    ],
                )]),
            ),
            epoch(1, 0.0, 2, BTreeMap::new()),
        ]);

        let report = observation_qc(&obs);

        assert_eq!(report.total_epoch_records, 3);
        assert_eq!(report.observation_epochs, 2);
        assert_eq!(report.event_records, 1);
        assert_eq!(report.power_failure_epochs, 1);
        assert_eq!(report.skipped_records, 0);
        assert!(report.clock_jumps.is_empty());
        assert_eq!(report.cycle_slips, CycleSlipQc::default());
        assert_eq!(report.multipath, MultipathReport::default());
        assert_eq!(report.satellites.len(), 2);
        assert_eq!(
            report.satellites[0],
            SatelliteObservationQc {
                satellite: g01,
                epochs_with_observations: 2,
                value_observations: 4,
            }
        );
        assert_eq!(
            report.satellites[1],
            SatelliteObservationQc {
                satellite: g02,
                epochs_with_observations: 1,
                value_observations: 1,
            }
        );

        let g01_c1c = report
            .satellite_signals
            .iter()
            .find(|signal| signal.satellite == g01 && signal.code == "C1C")
            .expect("G01 C1C signal");
        assert_eq!(g01_c1c.value_observations, 2);
        assert_eq!(
            g01_c1c.ssi,
            Some(SsiHistogram {
                counts: [0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
            })
        );
        assert_eq!(g01_c1c.snr, None);

        let gps_c1c = report
            .system_signals
            .iter()
            .find(|signal| signal.system == GnssSystem::Gps && signal.code == "C1C")
            .expect("GPS C1C signal");
        assert_eq!(gps_c1c.value_observations, 3);
        assert_eq!(
            gps_c1c.ssi,
            Some(SsiHistogram {
                counts: [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            })
        );

        let gps_s1c = report
            .system_signals
            .iter()
            .find(|signal| signal.system == GnssSystem::Gps && signal.code == "S1C")
            .expect("GPS S1C signal");
        assert_eq!(
            gps_s1c.snr,
            Some(SnrStats {
                n: 1,
                mean: 9.0,
                min: 9.0,
                max: 9.0,
                std: None,
            })
        );
    }

    #[test]
    fn observation_qc_detects_nominal_interval_gaps() {
        let g01 = sat(1);
        let obs = observation_file(vec![
            epoch(
                0,
                0.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(1.0), Some(5))])]),
            ),
            epoch(
                1,
                30.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(2.0), Some(6))])]),
            ),
        ]);

        let report = observation_qc(&obs);

        assert_eq!(report.missing_epochs, 2);
        assert_eq!(report.data_gaps.len(), 1);
        assert_eq!(report.data_gaps[0].nominal_interval_s, 30.0);
        assert_eq!(report.data_gaps[0].observed_delta_s, 90.0);
        assert_eq!(report.data_gaps[0].missing_epochs, 2);
    }

    #[test]
    fn observation_qc_infers_interval_when_header_is_absent() {
        let g01 = sat(1);
        let mut obs = observation_file(vec![
            epoch(
                0,
                0.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(1.0), Some(5))])]),
            ),
            epoch(
                0,
                30.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(2.0), Some(6))])]),
            ),
            epoch(
                2,
                0.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(3.0), Some(7))])]),
            ),
        ]);
        obs.header.interval_s = None;

        let report = observation_qc(&obs);

        assert_eq!(report.interval_s, Some(30.0));
        assert_eq!(report.interval_source, IntervalSource::Inferred);
        assert_eq!(report.missing_epochs, 2);
    }

    #[test]
    fn observation_qc_does_not_use_zero_header_interval_as_cadence() {
        let g01 = sat(1);
        let observations = BTreeMap::from([(g01, vec![obs_value(Some(1.0), Some(5))])]);
        let mut obs = observation_file(vec![
            epoch(0, 0.0, 0, observations.clone()),
            epoch(0, 30.0, 0, observations.clone()),
            epoch(1, 0.0, 0, observations.clone()),
            epoch(2, 30.0, 0, observations),
        ]);
        obs.header.interval_s = Some(0.0);

        let report = observation_qc(&obs);

        assert_eq!(report.interval_s, Some(30.0));
        assert_eq!(report.interval_source, IntervalSource::Inferred);
        assert_eq!(report.missing_epochs, 2);
        assert_eq!(report.data_gaps.len(), 1);
        assert!(report
            .lint_findings
            .iter()
            .any(|finding| { finding.code == "OBS-H19" && finding.severity == Severity::Info }));
    }

    #[test]
    fn observation_qc_reports_unresolved_zero_header_interval_without_calculating_gaps() {
        let mut obs = observation_file(Vec::new());
        obs.header.interval_s = Some(-0.0);

        let report = observation_qc(&obs);

        assert_eq!(report.interval_s, None);
        assert_eq!(report.interval_source, IntervalSource::Unresolved);
        assert!(report.data_gaps.is_empty());
        assert_eq!(report.missing_epochs, 0);
        assert!(report
            .notes
            .contains(&ObservationQcNote::IntervalUnresolved));
        assert!(report
            .lint_findings
            .iter()
            .any(|finding| { finding.code == "OBS-H19" && finding.severity == Severity::Info }));
    }

    #[test]
    fn observation_qc_ignores_and_reports_invalid_source_intervals() {
        let g01 = sat(1);
        let original = observation_file(vec![
            epoch(
                0,
                0.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(1.0), Some(5))])]),
            ),
            epoch(
                0,
                30.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(2.0), Some(6))])]),
            ),
        ]);

        for invalid in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut obs = original.clone();
            obs.header.interval_s = Some(invalid);

            let report = observation_qc(&obs);

            assert_eq!(report.interval_s, Some(30.0), "{invalid:?}");
            assert_eq!(
                report.interval_source,
                IntervalSource::Inferred,
                "{invalid:?}"
            );
            assert!(report.lint_findings.iter().any(|finding| {
                finding.code == "OBS-H20" && finding.severity == Severity::Error
            }));
        }
    }

    #[test]
    fn observation_qc_saturates_missing_epoch_counts_for_tiny_positive_interval() {
        let g01 = sat(1);
        let observations = BTreeMap::from([(g01, vec![obs_value(Some(1.0), Some(5))])]);
        let mut obs = observation_file(vec![
            epoch(0, 0.0, 0, observations.clone()),
            epoch(0, 30.0, 0, observations.clone()),
            epoch(1, 0.0, 0, observations.clone()),
            epoch(1, 30.0, 0, observations),
        ]);
        obs.header.interval_s = Some(f64::MIN_POSITIVE);

        let report = observation_qc(&obs);

        assert_eq!(report.data_gaps.len(), 3);
        assert!(report
            .data_gaps
            .iter()
            .all(|gap| gap.missing_epochs == usize::MAX));
        assert_eq!(report.missing_epochs, usize::MAX);
    }

    #[test]
    fn observation_qc_skips_nonfinite_public_epoch_deltas_without_panicking() {
        let g01 = sat(1);
        let original = observation_file(vec![
            epoch(
                0,
                0.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(1.0), Some(5))])]),
            ),
            epoch(
                0,
                30.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(2.0), Some(6))])]),
            ),
        ]);

        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut obs = original.clone();
            obs.header.interval_s = None;
            obs.epochs[1].epoch.second = nonfinite;
            let report = observation_qc(&obs);
            assert_eq!(report.interval_s, None, "{nonfinite:?}");
            assert_eq!(
                report.interval_source,
                IntervalSource::Unresolved,
                "{nonfinite:?}"
            );
            assert!(report.data_gaps.is_empty(), "{nonfinite:?}");
            assert_eq!(report.missing_epochs, 0, "{nonfinite:?}");
        }
    }

    #[test]
    fn observation_qc_notes_non_monotonic_epochs_and_excludes_them_from_gaps() {
        let g01 = sat(1);
        let obs = observation_file(vec![
            epoch(
                1,
                0.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(1.0), Some(5))])]),
            ),
            epoch(
                0,
                30.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(2.0), Some(6))])]),
            ),
        ]);

        let report = observation_qc(&obs);

        assert_eq!(
            report.notes,
            vec![ObservationQcNote::NonMonotonicEpoch { epoch_index: 1 }]
        );
        assert!(report.data_gaps.is_empty());
    }

    #[test]
    fn observation_qc_rejects_invalid_options() {
        let obs = observation_file(Vec::new());

        for invalid in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let err = observation_qc_with_options(
                &obs,
                ObservationQcOptions {
                    interval_override_s: Some(invalid),
                    ..ObservationQcOptions::default()
                },
            )
            .expect_err("invalid interval");
            assert_eq!(err, ObservationQcError::InvalidInterval, "{invalid:?}");
        }

        let err = observation_qc_with_options(
            &obs,
            ObservationQcOptions {
                interval_override_s: None,
                gap_factor: 1.0,
                ..ObservationQcOptions::default()
            },
        )
        .expect_err("invalid gap factor");
        assert_eq!(err, ObservationQcError::InvalidGapFactor);

        let err = observation_qc_with_options(
            &obs,
            ObservationQcOptions {
                clock_jump_threshold_s: 0.0,
                ..ObservationQcOptions::default()
            },
        )
        .expect_err("invalid clock-jump threshold");
        assert_eq!(err, ObservationQcError::InvalidClockJumpThreshold);
    }

    #[test]
    fn detect_clock_jumps_flags_injected_millisecond_step() {
        let g01 = sat(1);
        let mut obs = observation_file(vec![
            epoch(
                0,
                0.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(1.0), None)])]),
            ),
            epoch(
                0,
                30.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(2.0), None)])]),
            ),
            epoch(
                1,
                0.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(3.0), None)])]),
            ),
            epoch(
                1,
                30.0,
                0,
                BTreeMap::from([(g01, vec![obs_value(Some(4.0), None)])]),
            ),
        ]);
        let offsets_s = [0.0, 0.000_010, 0.001_020, 0.001_030];
        for (epoch, offset_s) in obs.epochs.iter_mut().zip(offsets_s) {
            epoch.rcv_clock_offset_s = Some(offset_s);
        }

        let jumps = detect_clock_jumps(&obs, DEFAULT_CLOCK_JUMP_THRESHOLD_S);

        assert_eq!(jumps.len(), 1);
        assert_eq!(jumps[0].epoch_index, 2);
        assert_close(jumps[0].delta_s, 0.001, "clock jump");

        let report = observation_qc(&obs);
        assert_eq!(report.clock_jumps, jumps);
    }

    #[test]
    fn detect_clock_jumps_ignores_linear_clock_drift() {
        let g01 = sat(1);
        let mut obs = observation_file(
            (0..4)
                .map(|idx| {
                    epoch(
                        idx / 2,
                        if idx % 2 == 0 { 0.0 } else { 30.0 },
                        0,
                        BTreeMap::from([(g01, vec![obs_value(Some(idx as f64), None)])]),
                    )
                })
                .collect(),
        );
        for (idx, epoch) in obs.epochs.iter_mut().enumerate() {
            epoch.rcv_clock_offset_s = Some(idx as f64 * 0.000_010);
        }

        assert!(detect_clock_jumps(&obs, DEFAULT_CLOCK_JUMP_THRESHOLD_S).is_empty());
        assert!(observation_qc(&obs).clock_jumps.is_empty());
    }

    #[test]
    fn observation_qc_tallies_synthetic_injected_cycle_slip() {
        let g01 = sat(1);
        let obs = observation_file(
            (0usize..5)
                .map(|epoch_index| {
                    let wide_lane_cycles = if epoch_index >= 3 { 14.0 } else { 8.0 };
                    epoch(
                        (epoch_index / 2) as u8,
                        if epoch_index % 2 == 0 { 0.0 } else { 30.0 },
                        0,
                        BTreeMap::from([(
                            g01,
                            dual_frequency_values(epoch_index, wide_lane_cycles, 0.0),
                        )]),
                    )
                })
                .collect(),
        );

        let report = observation_qc(&obs);

        assert_eq!(
            report.cycle_slips,
            CycleSlipQc {
                observations: 5,
                total_slips: 1,
                observations_per_slip: Some(5.0),
                by_system: vec![SystemCycleSlipQc {
                    system: GnssSystem::Gps,
                    observations: 5,
                    slips: 1,
                    observations_per_slip: Some(5.0),
                }],
            }
        );
    }

    #[test]
    fn multipath_combination_and_arc_rms_match_closed_form() {
        let p1_m = 20_000_010.0;
        let l1_m = 20_000_003.0;
        let l2_m = 20_000_001.0;
        let f2sq = F_L2_HZ * F_L2_HZ;
        let denom = F_L1_HZ * F_L1_HZ - f2sq;
        let expected = p1_m - l1_m - (2.0 * f2sq / denom) * (l1_m - l2_m);

        assert_close(
            mp_combination(p1_m, l1_m, l2_m, F_L1_HZ, F_L2_HZ),
            expected,
            "MP1 combination",
        );
        assert_close(
            arc_multipath_rms(&[1.0, 3.0, 5.0]),
            (8.0_f64 / 3.0).sqrt(),
            "arc RMS",
        );
    }

    #[test]
    fn multipath_stats_splits_arc_on_injected_lli_slip() {
        let g01 = sat(1);
        let high_threshold_config = CycleSlipConfig {
            melbourne_wubbena_threshold_cycles: 1.0e6,
            geometry_free_threshold_m: 1.0e6,
            minimum_arc_length: 10,
            maximum_gap_s: 1.0e6,
            ..CycleSlipConfig::default()
        };
        let with_slip = observation_file(
            (0usize..4)
                .map(|epoch_index| {
                    let bias_m = if epoch_index >= 2 { 10.0 } else { 0.0 };
                    epoch(
                        (epoch_index / 2) as u8,
                        if epoch_index % 2 == 0 { 0.0 } else { 30.0 },
                        0,
                        BTreeMap::from([(
                            g01,
                            dual_frequency_values_with_mp_bias(
                                epoch_index,
                                bias_m,
                                epoch_index == 2,
                            ),
                        )]),
                    )
                })
                .collect(),
        );
        let without_slip = observation_file(
            (0usize..4)
                .map(|epoch_index| {
                    let bias_m = if epoch_index >= 2 { 10.0 } else { 0.0 };
                    epoch(
                        (epoch_index / 2) as u8,
                        if epoch_index % 2 == 0 { 0.0 } else { 30.0 },
                        0,
                        BTreeMap::from([(
                            g01,
                            dual_frequency_values_with_mp_bias(epoch_index, bias_m, false),
                        )]),
                    )
                })
                .collect(),
        );

        let split = multipath_stats(&with_slip, &high_threshold_config);
        let unsplit = multipath_stats(&without_slip, &high_threshold_config);
        let split_mp1 = split.satellites[0].mp1.expect("split MP1");
        let unsplit_mp1 = unsplit.satellites[0].mp1.expect("unsplit MP1");

        assert_eq!(split_mp1.n, 4);
        assert_eq!(unsplit_mp1.n, 4);
        assert_close(split_mp1.rms_m, 0.0, "split MP1 RMS");
        assert_close(unsplit_mp1.rms_m, 25.0 / 6.0, "unsplit MP1 RMS");
    }

    #[test]
    fn multipath_matches_teqc_algo0010_oracle() {
        let oracle = read_json_fixture("qc/teqc_algo0010_2015001_v1_trim.json");
        let path = fixture_path("tests/fixtures/obs/algo0010_2015001_v1_trim.crx");
        let crx = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let decoded = crinex::decode(&crx).expect("decode CRINEX v1 fixture");
        let obs = RinexObs::parse(&decoded).expect("parse decoded RINEX 2 OBS");
        let report = observation_qc(&obs);
        let gps = report
            .multipath
            .systems
            .iter()
            .find(|system| system.system == GnssSystem::Gps)
            .expect("GPS multipath row");
        let mp1 = gps.mp1.expect("GPS MP1");
        let mp2 = gps.mp2.expect("GPS MP2");

        assert_eq!(mp1.n, 23);
        assert_eq!(mp2.n, 23);
        assert_close_tolerance(
            mp1.rms_m,
            oracle["summary"]["moving_average_mp12_m"].as_f64().unwrap(),
            1.0e-6,
            "teqc MP12",
        );
        assert_close_tolerance(
            mp2.rms_m,
            oracle["summary"]["moving_average_mp21_m"].as_f64().unwrap(),
            1.0e-6,
            "teqc MP21",
        );
    }

    #[test]
    fn observation_qc_accepts_crinex_v1_decoded_rinex2() {
        let path = fixture_path("tests/fixtures/obs/algo0010_2015001_v1_trim.crx");
        let crx = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        let decoded = crinex::decode(&crx).expect("decode CRINEX v1 fixture");
        let obs = RinexObs::parse(&decoded).expect("parse decoded RINEX 2 OBS");
        let report: ObservationQcReport = observation_qc(&obs);

        assert_eq!(report.total_epoch_records, 2);
        assert_eq!(report.observation_epochs, 2);
        assert_eq!(report.event_records, 0);
        assert_eq!(report.skipped_records, 0);
        assert_eq!(report.interval_s, Some(30.0));
        assert_eq!(report.interval_source, IntervalSource::Header);
        assert_eq!(report.missing_epochs, 0);
        assert_eq!(report.satellites.len(), 20);

        let g08 = GnssSatelliteId::new(GnssSystem::Gps, 8).expect("valid GPS PRN");
        let g08_report = report
            .satellites
            .iter()
            .find(|sat| sat.satellite == g08)
            .expect("G08 QC row");
        assert_eq!(g08_report.epochs_with_observations, 2);
        assert_eq!(g08_report.value_observations, 14);
    }

    #[test]
    fn observation_qc_matches_independent_real_fixture_oracles() {
        let doc = read_json_fixture("qc/observation_qc_real_oracles.json");
        assert_eq!(
            doc["provenance"]["generator"],
            "crates/sidereon-core/fixtures-generators/generate_observation_qc_oracles.py"
        );
        for fixture in doc["fixtures"].as_array().expect("fixtures array") {
            let rel = fixture["path"].as_str().expect("fixture path");
            let text = std::fs::read_to_string(fixture_path(rel))
                .unwrap_or_else(|e| panic!("read {rel}: {e}"));
            let obs = RinexObs::parse(&text).unwrap_or_else(|e| panic!("parse {rel}: {e}"));
            let report = observation_qc(&obs);

            assert_eq!(
                report.total_epoch_records,
                fixture["total_epoch_records"].as_u64().unwrap() as usize,
                "{rel}"
            );
            assert_eq!(
                report.observation_epochs,
                fixture["observation_epochs"].as_u64().unwrap() as usize,
                "{rel}"
            );
            assert_eq!(
                report.event_records,
                fixture["event_records"].as_u64().unwrap() as usize,
                "{rel}"
            );
            assert_eq!(
                report.power_failure_epochs,
                fixture["power_failure_epochs"].as_u64().unwrap() as usize,
                "{rel}"
            );
            assert_eq!(
                report.skipped_records,
                fixture["skipped_records"].as_u64().unwrap() as usize,
                "{rel}"
            );
            assert_close(
                report.interval_s.expect("oracle interval"),
                fixture["interval_s"].as_f64().unwrap(),
                rel,
            );
            assert_eq!(
                report.missing_epochs,
                fixture["missing_epochs"].as_u64().unwrap() as usize,
                "{rel}"
            );
            assert_gaps(&report.data_gaps, &fixture["data_gaps"], rel);
            assert_satellites(&report.satellites, &fixture["satellites"], rel);
            assert_satellite_signals(
                &report.satellite_signals,
                &fixture["satellite_signals"],
                rel,
            );
            assert_system_signals(&report.system_signals, &fixture["system_signals"], rel);
        }
    }

    #[test]
    fn observation_qc_pins_real_fixture_cycle_slip_tally() {
        let rel = "tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx";
        let text = std::fs::read_to_string(fixture_path(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"));
        let obs = RinexObs::parse(&text).unwrap_or_else(|e| panic!("parse {rel}: {e}"));
        let report = observation_qc(&obs);

        assert_eq!(report.cycle_slips.observations, 4135);
        assert_eq!(report.cycle_slips.total_slips, 27);
        assert_close(
            report
                .cycle_slips
                .observations_per_slip
                .expect("observations per slip"),
            4135.0 / 27.0,
            rel,
        );

        let by_system = report
            .cycle_slips
            .by_system
            .iter()
            .map(|row| {
                (
                    row.system,
                    (
                        row.observations,
                        row.slips,
                        row.observations_per_slip
                            .expect("system observations per slip"),
                    ),
                )
            })
            .collect::<BTreeMap<_, _>>();
        assert_eq!(by_system[&GnssSystem::Gps].0, 1282);
        assert_eq!(by_system[&GnssSystem::Gps].1, 4);
        assert_close(by_system[&GnssSystem::Gps].2, 1282.0 / 4.0, rel);
        assert_eq!(by_system[&GnssSystem::Glonass].0, 784);
        assert_eq!(by_system[&GnssSystem::Glonass].1, 10);
        assert_close(by_system[&GnssSystem::Glonass].2, 784.0 / 10.0, rel);
        assert_eq!(by_system[&GnssSystem::Galileo].0, 1023);
        assert_eq!(by_system[&GnssSystem::Galileo].1, 9);
        assert_close(by_system[&GnssSystem::Galileo].2, 1023.0 / 9.0, rel);
        assert_eq!(by_system[&GnssSystem::BeiDou].0, 1046);
        assert_eq!(by_system[&GnssSystem::BeiDou].1, 4);
        assert_close(by_system[&GnssSystem::BeiDou].2, 1046.0 / 4.0, rel);
    }

    #[test]
    fn observation_qc_report_text_snapshot_esbc00dnk() {
        let rel = "tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx";
        let text = std::fs::read_to_string(fixture_path(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"));
        let obs = RinexObs::parse(&text).unwrap_or_else(|e| panic!("parse {rel}: {e}"));
        let report = observation_qc(&obs);
        let rendered = render_text(&report);

        assert_eq!(rendered, ESBC_QC_REPORT_TEXT);
        assert!(rendered.contains("G   GPS"));
        assert!(rendered.contains("R   GLONASS"));
        assert!(rendered.contains("E   Galileo"));
        assert!(rendered.contains("C   BeiDou"));
        assert!(rendered.contains("S   SBAS"));
        assert!(rendered.contains("0.292"));
        assert!(rendered.contains("1.174"));
        assert!(rendered.contains("FINDINGS"));
        assert!(rendered.contains("CODE     SEVERITY SPEC REF"));
    }

    #[test]
    fn observation_qc_report_json_contains_expected_fields_esbc00dnk() {
        let rel = "tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx";
        let text = std::fs::read_to_string(fixture_path(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"));
        let obs = RinexObs::parse(&text).unwrap_or_else(|e| panic!("parse {rel}: {e}"));
        let report = observation_qc(&obs);

        let encoded = serde_json::to_string(&report).expect("serialize QC report");
        let doc: Value = serde_json::from_str(&encoded).expect("parse serialized QC report");

        assert_eq!(doc["header"]["marker_name"], "ESBC00DNK");
        assert_eq!(doc["header"]["receiver"]["receiver_type"], "SEPT POLARX5");
        assert_eq!(doc["interval_s"], 30.0);

        let gps = json_system(&doc, "Gps");
        assert_eq!(gps["satellites_seen"], 13);
        assert_eq!(gps["epochs_with_observations"], 120);
        assert_eq!(gps["value_observations"], 18645);
        assert_close(
            gps["completeness_ratio"].as_f64().unwrap(),
            0.800489438433797,
            "GPS JSON completeness",
        );

        let gps_mp = json_multipath_system(&doc, "Gps");
        assert_close(
            gps_mp["mp1"]["rms_m"].as_f64().unwrap(),
            0.29240479301672934,
            "GPS JSON MP1",
        );
        let beidou_mp = json_multipath_system(&doc, "BeiDou");
        assert_close(
            beidou_mp["mp2"]["rms_m"].as_f64().unwrap(),
            1.1736185873490712,
            "BeiDou JSON MP2",
        );

        let galileo_slips = doc["cycle_slips"]["by_system"]
            .as_array()
            .expect("cycle slip systems")
            .iter()
            .find(|row| row["system"] == "Galileo")
            .expect("Galileo cycle slip row");
        assert_eq!(galileo_slips["slips"], 9);
        assert_eq!(doc["lint_findings"].as_array().unwrap().len(), 0);
    }

    #[test]
    fn observation_qc_report_html_contains_rows_without_external_assets() {
        let rel = "tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx";
        let text = std::fs::read_to_string(fixture_path(rel))
            .unwrap_or_else(|e| panic!("read {rel}: {e}"));
        let obs = RinexObs::parse(&text).unwrap_or_else(|e| panic!("parse {rel}: {e}"));
        let report = observation_qc(&obs);
        let html = render_html(&report);

        assert!(html.contains("<td class=\"text\">G</td>"));
        assert!(html.contains("<td class=\"text\">GPS</td>"));
        assert!(html.contains("<td>0.292</td>"));
        assert!(html.contains("<td>1.174</td>"));
        assert!(html.contains("<h2>Findings</h2>"));
        assert!(!html.contains("http"));
    }

    const ESBC_QC_REPORT_TEXT: &str = r#"RINEX OBSERVATION QC +QC SUMMARY

HEADER
  MARKER NAME        ESBC00DNK
  MARKER NUMBER      10118M001
  MARKER TYPE        GEODETIC
  RECEIVER           3047937 / SEPT POLARX5 / 5.2.0
  ANTENNA            CR5200327016 / ASH701945E_M    SCIS
  POSITION XYZ M     3582105.2910 532589.7313 5232754.8054
  ANTENNA HEN M      0.2160 0.0000 0.0000
  TIME FIRST         2020-06-25 00:00:00.0000000 GPS
  TIME LAST          2020-06-25 00:59:30.0000000 GPS
  INTERVAL S         30.000 (header)
  DURATION S         3570.0

PER-CONSTELLATION
SYS NAME     SATS EPOCHS      OBS   EXPECT     COMP SNR MEAN/MIN BY BAND                                                                              MP1 RMS  MP2 RMS  SLIPS GAPS     GAP S
--- -------- ---- ------ -------- -------- -------- ------------------------------------------------------------------------------------------------ -------- -------- ------ ---- ---------
G   GPS        13    120    18645    23292    0.800 1:39.0/5.5 2:37.7/5.5 5:36.0/23.8                                                                   0.292    0.281      4    0       0.0
R   GLONASS    12    120    16323    22600    0.722 1:42.9/20.5 2:42.0/21.5 3:35.4/25.2                                                                 0.519    0.314     10    0       0.0
E   Galileo     9    120    19147    20540    0.932 1:42.2/18.8 5:36.5/20.5 6:34.8/24.0 7:45.2/25.5 8:45.2/26.0                                         0.386    0.483      9    0       0.0
C   BeiDou     12    120    11213    15708    0.714 2:42.6/32.5 6:35.6/26.8 7:41.2/36.0                                                                 1.017    1.174      4    0       0.0
S   SBAS        5    120     3032     4144    0.732 1:38.2/30.8 5:33.5/31.5                                                                                 -        -      0    0       0.0

FINDINGS
CODE     SEVERITY SPEC REF
-------- -------- ------------------------------------------------
NONE
"#;

    fn observation_file(epochs: Vec<ObsEpoch>) -> RinexObs {
        RinexObs {
            header: ObsHeader {
                version: 3.05,
                approx_position_m: None,
                antenna_delta_hen_m: None,
                obs_codes: BTreeMap::from([(
                    GnssSystem::Gps,
                    vec![
                        "C1C".to_string(),
                        "L1C".to_string(),
                        "S1C".to_string(),
                        "C2W".to_string(),
                        "L2W".to_string(),
                    ],
                )]),
                program_run_by_date: None,
                comments: Vec::new(),
                marker_number: None,
                marker_type: None,
                observer: None,
                agency: None,
                receiver: None,
                antenna: None,
                interval_s: Some(30.0),
                time_of_first_obs: None,
                time_of_last_obs: None,
                n_satellites: None,
                prn_obs_counts: BTreeMap::new(),
                phase_shifts: Vec::new(),
                scale_factors: Vec::new(),
                glonass_slots: BTreeMap::new(),
                glonass_cod_phs_bis: None,
                signal_strength_unit: None,
                leap_seconds: None,
                marker_name: None,
                unretained_header_labels: Vec::new(),
            },
            epochs,
            skipped_records: 0,
        }
    }

    fn epoch(
        minute: u8,
        second: f64,
        flag: u8,
        sats: BTreeMap<GnssSatelliteId, Vec<ObsValue>>,
    ) -> ObsEpoch {
        ObsEpoch {
            epoch: ObsEpochTime {
                year: 2024,
                month: 1,
                day: 1,
                hour: 0,
                minute,
                second,
            },
            flag,
            rcv_clock_offset_s: None,
            epoch_picoseconds: None,
            declared_record_count: sats.len(),
            special_record_count: if flag > 1 { sats.len() } else { 0 },
            sats,
        }
    }

    fn obs_value(value: Option<f64>, ssi: Option<u8>) -> ObsValue {
        ObsValue {
            value,
            lli: None,
            ssi,
        }
    }

    fn dual_frequency_values(
        epoch_index: usize,
        melbourne_wubbena_cycles: f64,
        geometry_free_m: f64,
    ) -> Vec<ObsValue> {
        let geometric_m = 23_000_000.0 + epoch_index as f64 * 100.0;
        let lambda1 = C_M_S / F_L1_HZ;
        let lambda2 = C_M_S / F_L2_HZ;
        let lambda_wl = C_M_S / (F_L1_HZ - F_L2_HZ);
        let l2_m = geometric_m + lambda_wl * (melbourne_wubbena_cycles - geometry_free_m / lambda1);
        let l1_m = l2_m + geometry_free_m;

        vec![
            obs_value(Some(geometric_m), None),
            obs_value(Some(l1_m / lambda1), None),
            obs_value(None, None),
            obs_value(Some(geometric_m), None),
            obs_value(Some(l2_m / lambda2), None),
        ]
    }

    fn dual_frequency_values_with_mp_bias(
        epoch_index: usize,
        p_bias_m: f64,
        lli_slip: bool,
    ) -> Vec<ObsValue> {
        let mut values = dual_frequency_values(epoch_index, 8.0, 0.0);
        values[0].value = values[0].value.map(|value| value + p_bias_m);
        values[3].value = values[3].value.map(|value| value + p_bias_m);
        if lli_slip {
            values[1].lli = Some(1);
        }
        values
    }

    fn sat(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS PRN")
    }

    fn fixture_path(rel: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(rel)
    }

    fn read_json_fixture(rel: &str) -> Value {
        let path = fixture_path(&format!("tests/fixtures/{rel}"));
        let raw = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
    }

    fn json_system<'a>(doc: &'a Value, system: &str) -> &'a Value {
        doc["systems"]
            .as_array()
            .expect("systems")
            .iter()
            .find(|row| row["system"] == system)
            .unwrap_or_else(|| panic!("missing JSON system {system}"))
    }

    fn json_multipath_system<'a>(doc: &'a Value, system: &str) -> &'a Value {
        doc["multipath"]["systems"]
            .as_array()
            .expect("multipath systems")
            .iter()
            .find(|row| row["system"] == system)
            .unwrap_or_else(|| panic!("missing JSON multipath system {system}"))
    }

    fn assert_close(actual: f64, expected: f64, context: &str) {
        assert_close_tolerance(actual, expected, 1.0e-9, context);
    }

    fn assert_close_tolerance(actual: f64, expected: f64, tolerance: f64, context: &str) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{context}: actual {actual:?}, expected {expected:?}"
        );
    }

    fn assert_gaps(actual: &[ObservationDataGap], expected: &Value, context: &str) {
        let expected = expected.as_array().expect("gap array");
        assert_eq!(actual.len(), expected.len(), "{context}");
        for (actual, expected) in actual.iter().zip(expected) {
            assert_epoch(&actual.start_epoch, &expected["start_epoch"], context);
            assert_epoch(&actual.end_epoch, &expected["end_epoch"], context);
            assert_close(
                actual.nominal_interval_s,
                expected["nominal_interval_s"].as_f64().unwrap(),
                context,
            );
            assert_close(
                actual.observed_delta_s,
                expected["observed_delta_s"].as_f64().unwrap(),
                context,
            );
            assert_eq!(
                actual.missing_epochs,
                expected["missing_epochs"].as_u64().unwrap() as usize,
                "{context}"
            );
        }
    }

    fn assert_epoch(actual: &ObsEpochTime, expected: &Value, context: &str) {
        assert_eq!(
            actual.year,
            expected["year"].as_i64().unwrap() as i32,
            "{context}"
        );
        assert_eq!(
            actual.month,
            expected["month"].as_u64().unwrap() as u8,
            "{context}"
        );
        assert_eq!(
            actual.day,
            expected["day"].as_u64().unwrap() as u8,
            "{context}"
        );
        assert_eq!(
            actual.hour,
            expected["hour"].as_u64().unwrap() as u8,
            "{context}"
        );
        assert_eq!(
            actual.minute,
            expected["minute"].as_u64().unwrap() as u8,
            "{context}"
        );
        assert_close(actual.second, expected["second"].as_f64().unwrap(), context);
    }

    fn assert_satellites(actual: &[SatelliteObservationQc], expected: &Value, context: &str) {
        let expected = expected.as_array().expect("satellites array");
        assert_eq!(actual.len(), expected.len(), "{context}");
        let actual = actual
            .iter()
            .map(|sat| {
                (
                    sat.satellite.to_string(),
                    (sat.epochs_with_observations, sat.value_observations),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for expected in expected {
            let satellite = expected["satellite"].as_str().unwrap();
            let actual = actual
                .get(satellite)
                .unwrap_or_else(|| panic!("{context}: missing satellite {satellite}"));
            assert_eq!(
                actual.0,
                expected["epochs_with_observations"].as_u64().unwrap() as usize,
                "{context} {satellite}"
            );
            assert_eq!(
                actual.1,
                expected["value_observations"].as_u64().unwrap() as usize,
                "{context} {satellite}"
            );
        }
    }

    fn assert_satellite_signals(actual: &[SatelliteSignalQc], expected: &Value, context: &str) {
        let expected = expected.as_array().expect("satellite signals array");
        assert_eq!(actual.len(), expected.len(), "{context}");
        let actual = actual
            .iter()
            .map(|signal| {
                (
                    (signal.satellite.to_string(), signal.code.as_str()),
                    (signal.value_observations, signal.ssi, signal.snr),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for expected in expected {
            let satellite = expected["satellite"].as_str().unwrap();
            let code = expected["code"].as_str().unwrap();
            let actual = actual
                .get(&(satellite.to_string(), code))
                .unwrap_or_else(|| panic!("{context}: missing {satellite} {code}"));
            assert_eq!(
                actual.0,
                expected["value_observations"].as_u64().unwrap() as usize,
                "{context} {satellite} {code}"
            );
            assert_ssi(actual.1, &expected["ssi"], context);
            assert_snr(actual.2, &expected["snr"], context);
        }
    }

    fn assert_system_signals(actual: &[SystemSignalQc], expected: &Value, context: &str) {
        let expected = expected.as_array().expect("system signals array");
        assert_eq!(actual.len(), expected.len(), "{context}");
        let actual = actual
            .iter()
            .map(|signal| {
                (
                    (signal.system.letter().to_string(), signal.code.as_str()),
                    (signal.value_observations, signal.ssi, signal.snr),
                )
            })
            .collect::<BTreeMap<_, _>>();
        for expected in expected {
            let system = expected["system"].as_str().unwrap();
            let code = expected["code"].as_str().unwrap();
            let actual = actual
                .get(&(system.to_string(), code))
                .unwrap_or_else(|| panic!("{context}: missing {system} {code}"));
            assert_eq!(
                actual.0,
                expected["value_observations"].as_u64().unwrap() as usize,
                "{context} {system} {code}"
            );
            assert_ssi(actual.1, &expected["ssi"], context);
            assert_snr(actual.2, &expected["snr"], context);
        }
    }

    fn assert_ssi(actual: Option<SsiHistogram>, expected: &Value, context: &str) {
        if expected.is_null() {
            assert_eq!(actual, None, "{context}");
            return;
        }
        let expected = expected
            .as_array()
            .expect("ssi array")
            .iter()
            .map(|value| value.as_u64().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(actual.expect("ssi").counts.to_vec(), expected, "{context}");
    }

    fn assert_snr(actual: Option<SnrStats>, expected: &Value, context: &str) {
        if expected.is_null() {
            assert_eq!(actual, None, "{context}");
            return;
        }
        let actual = actual.expect("snr");
        assert_eq!(
            actual.n,
            expected["n"].as_u64().unwrap() as usize,
            "{context}"
        );
        assert_close(actual.mean, expected["mean"].as_f64().unwrap(), context);
        assert_close(actual.min, expected["min"].as_f64().unwrap(), context);
        assert_close(actual.max, expected["max"].as_f64().unwrap(), context);
        if expected["std"].is_null() {
            assert_eq!(actual.std, None, "{context}");
        } else {
            assert_close(
                actual.std.expect("std"),
                expected["std"].as_f64().unwrap(),
                context,
            );
        }
    }
}
