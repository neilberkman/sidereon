//! MP1/MP2 code multipath RMS for observation QC.

use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::carrier_phase::phase_meters;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::precise_positioning::{detect_cycle_slips, CycleSlipConfig, DualFrequencyObservation};
use crate::rinex::observations::{ObsEpochTime, RinexObs};
use crate::rinex_common::dominant_obs_interval_s;

use super::{
    detect_gaps, dual_frequency_epochs, multipath_dual_frequency_observation, ObservationQcOptions,
};

const TEQC_MP_MOVING_AVERAGE_POINTS: usize = 50;

/// MP1/MP2 multipath RMS report.
#[derive(Debug, Clone, PartialEq, Default, serde::Serialize)]
pub struct MultipathReport {
    /// Per-satellite MP1/MP2 RMS rows.
    pub satellites: Vec<SatelliteMultipathQc>,
    /// Per-constellation MP1/MP2 RMS rows.
    pub systems: Vec<SystemMultipathQc>,
}

/// Per-satellite multipath RMS after per-arc teqc moving-average filtering.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SatelliteMultipathQc {
    /// Satellite id.
    pub satellite: GnssSatelliteId,
    /// MP1 RMS, when at least one complete sample was available.
    pub mp1: Option<MpStats>,
    /// MP2 RMS, when at least one complete sample was available.
    pub mp2: Option<MpStats>,
}

/// Per-constellation multipath RMS after per-arc teqc moving-average filtering.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct SystemMultipathQc {
    /// GNSS constellation.
    pub system: GnssSystem,
    /// MP1 RMS across satellites in this constellation.
    pub mp1: Option<MpStats>,
    /// MP2 RMS across satellites in this constellation.
    pub mp2: Option<MpStats>,
}

/// RMS statistics for one multipath series.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct MpStats {
    /// Number of multipath samples.
    pub n: usize,
    /// Root-mean-square multipath residual, in meters.
    pub rms_m: f64,
}

#[derive(Debug, Clone, Copy)]
struct MultipathSample {
    mp1_m: f64,
    mp2_m: f64,
}

#[derive(Debug, Default)]
struct SatelliteMultipathState {
    mp1_arc_m: Vec<f64>,
    mp2_arc_m: Vec<f64>,
    mp1_residuals: MovingAverageAccum,
    mp2_residuals: MovingAverageAccum,
}

impl SatelliteMultipathState {
    fn push(&mut self, sample: MultipathSample) {
        debug_assert!(sample.mp1_m.is_finite());
        debug_assert!(sample.mp2_m.is_finite());
        self.mp1_arc_m.push(sample.mp1_m);
        self.mp2_arc_m.push(sample.mp2_m);
    }

    fn close_arc(&mut self) {
        debug_assert_eq!(self.mp1_arc_m.len(), self.mp2_arc_m.len());
        self.mp1_residuals.add_arc(&self.mp1_arc_m);
        self.mp2_residuals.add_arc(&self.mp2_arc_m);
        self.mp1_arc_m.clear();
        self.mp2_arc_m.clear();
    }

    fn finish(mut self) -> (Option<MpStats>, Option<MpStats>) {
        self.close_arc();
        (self.mp1_residuals.finish(), self.mp2_residuals.finish())
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct MovingAverageAccum {
    n: usize,
    sum_sq_m2: f64,
}

impl MovingAverageAccum {
    fn add_arc(&mut self, arc_m: &[f64]) {
        if arc_m.is_empty() {
            return;
        }

        let mut window =
            VecDeque::<f64>::with_capacity(TEQC_MP_MOVING_AVERAGE_POINTS.min(arc_m.len()));
        for value_m in arc_m {
            window.push_back(*value_m);
            if window.len() > TEQC_MP_MOVING_AVERAGE_POINTS {
                window.pop_front();
            }
            let residual_m = *value_m - mean_window(&window);
            self.n += 1;
            self.sum_sq_m2 += residual_m * residual_m;
        }
    }

    fn finish(self) -> Option<MpStats> {
        (self.n > 0).then(|| MpStats {
            n: self.n,
            rms_m: (self.sum_sq_m2 / self.n as f64).sqrt(),
        })
    }
}

#[derive(Debug, Default)]
struct SystemMultipathAccum {
    mp1: SystemMovingAverageAccum,
    mp2: SystemMovingAverageAccum,
}

#[derive(Debug, Clone, Copy, Default)]
struct SystemMovingAverageAccum {
    samples: usize,
    satellite_rms_sum_m: f64,
}

impl SystemMovingAverageAccum {
    fn add_stats(&mut self, stats: MpStats) {
        self.samples += stats.n;
        if stats.n > 1 {
            self.satellite_rms_sum_m += stats.rms_m;
        }
    }

    fn finish(self, epoch_count: usize) -> Option<MpStats> {
        if self.samples == 0 || epoch_count == 0 {
            return None;
        }
        let average_observations_per_epoch = self.samples as f64 / epoch_count as f64;
        Some(MpStats {
            n: self.samples,
            rms_m: self.satellite_rms_sum_m / average_observations_per_epoch,
        })
    }
}

/// Compute the meter-valued MP combination for one band.
///
/// Inputs are pseudorange and phase in meters. The operation order intentionally
/// squares each carrier first, forms the signed scale factor, then combines.
pub fn mp_combination(p_i_m: f64, l_i_m: f64, l_j_m: f64, f_i_hz: f64, f_j_hz: f64) -> f64 {
    debug_assert!(p_i_m.is_finite());
    debug_assert!(l_i_m.is_finite());
    debug_assert!(l_j_m.is_finite());
    debug_assert!(f_i_hz.is_finite() && f_i_hz > 0.0);
    debug_assert!(f_j_hz.is_finite() && f_j_hz > 0.0);

    let fi_sq = f_i_hz * f_i_hz;
    let fj_sq = f_j_hz * f_j_hz;
    let denominator = fi_sq - fj_sq;
    debug_assert!(denominator != 0.0);
    let factor = 2.0 * fj_sq / denominator;
    p_i_m - l_i_m - factor * (l_i_m - l_j_m)
}

/// RMS of a single continuous-phase MP arc after subtracting that arc's mean.
pub fn arc_multipath_rms(series_m: &[f64]) -> f64 {
    if series_m.is_empty() {
        return 0.0;
    }
    let mean_m = mean(series_m);
    let sum_sq_m2 = series_m
        .iter()
        .map(|value_m| {
            let residual_m = *value_m - mean_m;
            residual_m * residual_m
        })
        .sum::<f64>();
    (sum_sq_m2 / series_m.len() as f64).sqrt()
}

/// Compute MP1/MP2 RMS stats from a parsed observation file.
pub fn multipath_stats(obs: &RinexObs, cycle_slip_config: &CycleSlipConfig) -> MultipathReport {
    let slip_boundaries = slip_boundaries(obs, cycle_slip_config);
    let gap_starts = gap_start_epoch_indices(obs);
    let (timeline, _) = super::header_timeline_or_file(obs);
    let mut states = BTreeMap::<GnssSatelliteId, SatelliteMultipathState>::new();
    let epoch_count = obs
        .epochs()
        .iter()
        .filter(|epoch| super::is_observation_epoch(epoch))
        .count();

    for (epoch_index, (file_index, epoch)) in obs
        .epochs()
        .iter()
        .enumerate()
        .filter(|(_, epoch)| super::is_observation_epoch(epoch))
        .enumerate()
    {
        let header = timeline.at(file_index);
        if gap_starts.contains(&epoch_index) {
            close_all_arcs(&mut states);
        }

        let mut samples =
            BTreeMap::<GnssSatelliteId, (DualFrequencyObservation, MultipathSample)>::new();
        for (satellite, values) in &epoch.sats {
            let Some(observation) =
                multipath_dual_frequency_observation(header, *satellite, values)
            else {
                continue;
            };
            let Some(sample) = multipath_sample(&observation) else {
                continue;
            };
            samples.insert(*satellite, (observation, sample));
        }

        let active_satellites = states.keys().copied().collect::<Vec<_>>();
        for satellite in active_satellites {
            if !samples.contains_key(&satellite) {
                if let Some(state) = states.get_mut(&satellite) {
                    state.close_arc();
                }
            }
        }

        for (satellite, (observation, sample)) in samples {
            let state = states.entry(satellite).or_default();
            if slip_boundaries.contains(&(epoch_index, observation.satellite_id)) {
                state.close_arc();
            }
            state.push(sample);
        }
    }

    finish_report(states, epoch_count)
}

fn close_all_arcs(states: &mut BTreeMap<GnssSatelliteId, SatelliteMultipathState>) {
    for state in states.values_mut() {
        state.close_arc();
    }
}

fn finish_report(
    states: BTreeMap<GnssSatelliteId, SatelliteMultipathState>,
    epoch_count: usize,
) -> MultipathReport {
    let mut satellite_rows = Vec::new();
    let mut system_accums = BTreeMap::<GnssSystem, SystemMultipathAccum>::new();

    for (satellite, state) in states {
        let (mp1, mp2) = state.finish();
        if mp1.is_none() && mp2.is_none() {
            continue;
        }
        let system_acc = system_accums.entry(satellite.system).or_default();
        if let Some(stats) = mp1 {
            system_acc.mp1.add_stats(stats);
        }
        if let Some(stats) = mp2 {
            system_acc.mp2.add_stats(stats);
        }
        satellite_rows.push(SatelliteMultipathQc {
            satellite,
            mp1,
            mp2,
        });
    }

    let systems = system_accums
        .into_iter()
        .map(|(system, acc)| SystemMultipathQc {
            system,
            mp1: acc.mp1.finish(epoch_count),
            mp2: acc.mp2.finish(epoch_count),
        })
        .collect();

    MultipathReport {
        satellites: satellite_rows,
        systems,
    }
}

fn multipath_sample(observation: &DualFrequencyObservation) -> Option<MultipathSample> {
    let l1_m = phase_meters(observation.phi1_cyc, observation.f1_hz).ok()?;
    let l2_m = phase_meters(observation.phi2_cyc, observation.f2_hz).ok()?;
    let mp1_m = mp_combination(
        observation.p1_m,
        l1_m,
        l2_m,
        observation.f1_hz,
        observation.f2_hz,
    );
    let mp2_m = mp_combination(
        observation.p2_m,
        l2_m,
        l1_m,
        observation.f2_hz,
        observation.f1_hz,
    );
    (mp1_m.is_finite() && mp2_m.is_finite()).then_some(MultipathSample { mp1_m, mp2_m })
}

fn slip_boundaries(
    obs: &RinexObs,
    cycle_slip_config: &CycleSlipConfig,
) -> BTreeSet<(usize, String)> {
    let epochs = dual_frequency_epochs(obs);
    let Ok(flags) = detect_cycle_slips(&epochs, *cycle_slip_config) else {
        return BTreeSet::new();
    };

    flags
        .into_iter()
        .enumerate()
        .flat_map(|(epoch_index, epoch)| {
            epoch
                .observations
                .into_iter()
                .filter(|observation| observation.slip)
                .map(move |observation| (epoch_index, observation.satellite_id))
        })
        .collect()
}

fn gap_start_epoch_indices(obs: &RinexObs) -> BTreeSet<usize> {
    let epochs = observation_epochs_with_interval(obs);
    let epoch_times: Vec<ObsEpochTime> = epochs.iter().map(|(time, _)| *time).collect();
    let nominal = super::NominalInterval {
        override_s: None,
        inferred_s: dominant_obs_interval_s(&epoch_times),
    };
    let gaps = detect_gaps(ObservationQcOptions::default(), &epochs, nominal);

    gaps.into_iter()
        .filter_map(|gap| epoch_times.iter().position(|epoch| *epoch == gap.end_epoch))
        .collect()
}

/// Each observation epoch's time, with the `INTERVAL` of the header in effect
/// at it when that interval is usable.
fn observation_epochs_with_interval(obs: &RinexObs) -> Vec<(ObsEpochTime, Option<f64>)> {
    let (timeline, _) = super::header_timeline_or_file(obs);
    obs.epochs()
        .iter()
        .enumerate()
        .filter(|(_, epoch)| super::is_observation_epoch(epoch))
        .filter_map(|(index, epoch)| {
            epoch.epoch.map(|time| {
                let interval_s = timeline
                    .at(index)
                    .interval_s
                    .filter(|interval_s| interval_s.is_finite() && *interval_s > 0.0);
                (time, interval_s)
            })
        })
        .collect()
}

fn mean(series_m: &[f64]) -> f64 {
    series_m.iter().sum::<f64>() / series_m.len() as f64
}

fn mean_window(series_m: &VecDeque<f64>) -> f64 {
    series_m.iter().sum::<f64>() / series_m.len() as f64
}
