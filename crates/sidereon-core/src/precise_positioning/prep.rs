//! Wide-lane/narrow-lane and cycle-slip preparation for static PPP.
//!
//! Owns the dual-frequency ambiguity prep that runs ahead of the float/fixed
//! solve: wide-lane integer estimation, ionosphere-free narrow-lane epoch
//! construction, and cycle-slip arc segmentation (both the dual-frequency
//! wide-lane path and the float ambiguity-id tagging path). These steps depend
//! only on the raw observations, not on the least-squares solve, so they form a
//! self-contained leaf that the parent module re-exports.

use std::collections::{BTreeMap, BTreeSet};

use crate::ambiguity::{self, AmbiguityId, CycleSlipPolicy, NarrowLaneParams};
use crate::carrier_phase::{
    detect_cycle_slips, ArcEpoch, CarrierPhaseError, CycleSlipOptions, SlipReason,
};
use crate::combinations::{self, IonosphereFreeError};

/// Raw dual-frequency PPP observation used by wide-lane/narrow-lane prep.
#[derive(Debug, Clone, PartialEq)]
pub struct DualFrequencyObservation {
    pub satellite_id: String,
    pub ambiguity_id: String,
    pub p1_m: f64,
    pub p2_m: f64,
    pub phi1_cyc: f64,
    pub phi2_cyc: f64,
    pub f1_hz: f64,
    pub f2_hz: f64,
    pub lli1: Option<i64>,
    pub lli2: Option<i64>,
}

/// One raw dual-frequency PPP epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct DualFrequencyEpoch {
    /// Comparable epoch coordinate in seconds for data-gap cycle-slip checks.
    pub gap_time_s: Option<f64>,
    pub observations: Vec<DualFrequencyObservation>,
}

/// One ionosphere-free PPP observation emitted by dual-frequency prep.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedFloatObservation {
    pub satellite_id: String,
    pub ambiguity_id: String,
    pub code_m: f64,
    pub phase_m: f64,
}

/// One prepared ionosphere-free epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct PreparedFloatEpoch {
    pub epoch_index: usize,
    pub observations: Vec<PreparedFloatObservation>,
}

/// Wide-lane and narrow-lane prep controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WideLanePrepOptions {
    pub min_epochs: usize,
    pub tolerance_cycles: f64,
}

/// Public split-arc metadata for PPP ambiguity segmentation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PppSplitArc {
    pub satellite_id: String,
    pub ambiguity_id: String,
    pub start_epoch_index: usize,
    pub end_epoch_index: usize,
    pub n_epochs: usize,
}

/// Prepared dual-frequency PPP arc for the fixed wide-lane/narrow-lane path.
#[derive(Debug, Clone, PartialEq)]
pub struct WideLanePrepResult {
    pub epochs: Vec<PreparedFloatEpoch>,
    pub wavelengths_m: BTreeMap<String, f64>,
    pub offsets_m: BTreeMap<String, f64>,
    pub wide_lane_cycles: BTreeMap<String, i64>,
    pub dropped_sats: Vec<String>,
    pub split_arcs: Vec<PppSplitArc>,
}

/// Error from PPP wide-lane/narrow-lane prep.
#[derive(Debug, Clone, PartialEq)]
pub enum WideLanePrepError {
    CycleSlipDetected {
        satellite_id: String,
        epoch_index: usize,
        reasons: Vec<SlipReason>,
    },
    WideLaneFailed {
        ambiguity_id: String,
        reason: CarrierPhaseError,
    },
    TooFewWideLaneEpochs {
        ambiguity_id: String,
        count: usize,
        minimum: usize,
    },
    WideLaneNotInteger {
        ambiguity_id: String,
        mean_cycles: f64,
        fixed_cycles: i64,
    },
    MissingWideLaneAmbiguity(String),
    InconsistentFrequencies(String),
    IonosphereFreeFailed {
        satellite_id: String,
        reason: IonosphereFreeError,
    },
}

/// One float PPP observation with optional raw dual-frequency fields for
/// cycle-slip ambiguity-tagging.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatCycleSlipObservation {
    pub satellite_id: String,
    pub ambiguity_id: String,
    pub raw: Option<DualFrequencyObservation>,
}

/// One float PPP epoch for cycle-slip ambiguity-tagging.
#[derive(Debug, Clone, PartialEq)]
pub struct FloatCycleSlipEpoch {
    pub gap_time_s: Option<f64>,
    pub observations: Vec<FloatCycleSlipObservation>,
}

/// One tagged float PPP observation returned by cycle-slip prep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloatCycleSlipTaggedObservation {
    pub satellite_id: String,
    pub ambiguity_id: String,
}

/// One tagged float PPP epoch returned by cycle-slip prep.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FloatCycleSlipTaggedEpoch {
    pub observations: Vec<FloatCycleSlipTaggedObservation>,
}
/// Prepare raw dual-frequency PPP observations for the wide-lane then
/// narrow-lane fixed ambiguity path.
pub fn prepare_widelane_fixed_epochs(
    epochs: &[DualFrequencyEpoch],
    wide_lane: WideLanePrepOptions,
    cycle_slip_policy: CycleSlipPolicy,
    cycle_slip_options: CycleSlipOptions,
) -> Result<WideLanePrepResult, WideLanePrepError> {
    let (prepared_dual_epochs, wide_lane_cycles, dropped_sats, split_arcs) =
        wide_lane_ambiguities(epochs, wide_lane, cycle_slip_policy, cycle_slip_options)?;
    let filtered_dual_epochs =
        filter_dual_epochs_by_wide_lanes(&prepared_dual_epochs, &wide_lane_cycles);
    let (if_epochs, wavelengths_m, offsets_m) =
        ionosphere_free_narrow_lane_epochs(&filtered_dual_epochs, &wide_lane_cycles)?;
    Ok(WideLanePrepResult {
        epochs: if_epochs,
        wavelengths_m,
        offsets_m,
        wide_lane_cycles,
        dropped_sats,
        split_arcs,
    })
}
/// Rewrite float PPP ambiguity ids at detected dual-frequency cycle slips.
pub fn split_float_cycle_slip_epochs(
    epochs: &[FloatCycleSlipEpoch],
    cycle_slip_options: CycleSlipOptions,
) -> Vec<FloatCycleSlipTaggedEpoch> {
    let tags = float_cycle_slip_tags(epochs, cycle_slip_options);
    epochs
        .iter()
        .enumerate()
        .map(|(epoch_index, epoch)| {
            let mut observations = epoch
                .observations
                .iter()
                .map(|obs| {
                    let ambiguity_id = tags
                        .get(&(epoch_index, obs.satellite_id.clone()))
                        .map_or_else(|| obs.ambiguity_id.clone(), |id| id.as_str().to_string());
                    FloatCycleSlipTaggedObservation {
                        satellite_id: obs.satellite_id.clone(),
                        ambiguity_id,
                    }
                })
                .collect::<Vec<_>>();
            observations.sort_by(|a, b| {
                (a.satellite_id.as_str(), a.ambiguity_id.as_str())
                    .cmp(&(b.satellite_id.as_str(), b.ambiguity_id.as_str()))
            });
            FloatCycleSlipTaggedEpoch { observations }
        })
        .collect()
}
#[derive(Clone, Copy)]
struct DualArcSample<'a> {
    epoch_index: usize,
    gap_time_s: Option<f64>,
    observation: &'a DualFrequencyObservation,
}

#[derive(Clone)]
struct PreparedDualFrequencyEpoch {
    epoch_index: usize,
    observations: Vec<DualFrequencyObservation>,
}

struct DualSlipEvent {
    epoch_index: usize,
    reasons: Vec<SlipReason>,
}

type WideLanePrepPieces = (
    Vec<PreparedDualFrequencyEpoch>,
    BTreeMap<String, i64>,
    Vec<String>,
    Vec<PppSplitArc>,
);

type TaggedWideLaneArc = (
    Vec<(usize, DualFrequencyObservation)>,
    BTreeMap<String, i64>,
    Option<PppSplitArc>,
);

type WideLaneArcPrepared = (
    Vec<(usize, DualFrequencyObservation)>,
    BTreeMap<String, i64>,
    Vec<String>,
    Vec<PppSplitArc>,
);

fn wide_lane_ambiguities(
    epochs: &[DualFrequencyEpoch],
    wide_lane: WideLanePrepOptions,
    cycle_slip_policy: CycleSlipPolicy,
    cycle_slip_options: CycleSlipOptions,
) -> Result<WideLanePrepPieces, WideLanePrepError> {
    let mut arcs = BTreeMap::<String, Vec<DualArcSample<'_>>>::new();
    for (epoch_index, epoch) in epochs.iter().enumerate() {
        for observation in &epoch.observations {
            arcs.entry(observation.satellite_id.clone())
                .or_default()
                .push(DualArcSample {
                    epoch_index,
                    gap_time_s: epoch.gap_time_s,
                    observation,
                });
        }
    }

    let mut entries = Vec::new();
    let mut cycles = BTreeMap::new();
    let mut dropped = Vec::new();
    let mut split_arcs = Vec::new();
    for (satellite_id, mut arc) in arcs {
        arc.sort_by_key(|sample| sample.epoch_index);
        let (arc_entries, arc_cycles, arc_dropped, arc_splits) = prepare_wide_lane_arc(
            &satellite_id,
            &arc,
            wide_lane,
            cycle_slip_policy,
            cycle_slip_options,
        )?;
        entries.extend(arc_entries);
        cycles.extend(arc_cycles);
        dropped.extend(arc_dropped);
        split_arcs.extend(arc_splits);
    }

    dropped.sort();
    dropped.dedup();
    split_arcs.sort_by(|a, b| {
        (a.satellite_id.as_str(), a.ambiguity_id.as_str())
            .cmp(&(b.satellite_id.as_str(), b.ambiguity_id.as_str()))
    });

    Ok((
        dual_epochs_from_entries(entries),
        cycles,
        dropped,
        split_arcs,
    ))
}

fn prepare_wide_lane_arc(
    satellite_id: &str,
    arc: &[DualArcSample<'_>],
    wide_lane: WideLanePrepOptions,
    cycle_slip_policy: CycleSlipPolicy,
    cycle_slip_options: CycleSlipOptions,
) -> Result<WideLaneArcPrepared, WideLanePrepError> {
    let slips = cycle_slips_for_dual_arc(arc, cycle_slip_options);
    match cycle_slip_policy {
        CycleSlipPolicy::SplitArc if !slips.is_empty() => {
            prepare_split_wide_lane_arc(satellite_id, arc, wide_lane, &slips)
        }
        _ if slips.is_empty() => {
            // An unslipped arc's ambiguity id is the bare satellite token.
            let arc_id = AmbiguityId::new(satellite_id);
            estimate_tagged_wide_lane_arc(&arc_id, &arc_id, arc, wide_lane, None).map(
                |(entries, cycles, split_arc)| {
                    (entries, cycles, Vec::new(), split_arc.into_iter().collect())
                },
            )
        }
        CycleSlipPolicy::DropSatellite => Ok((
            Vec::new(),
            BTreeMap::new(),
            vec![satellite_id.to_string()],
            Vec::new(),
        )),
        CycleSlipPolicy::Error | CycleSlipPolicy::SplitArc => {
            let slip = &slips[0];
            Err(WideLanePrepError::CycleSlipDetected {
                satellite_id: satellite_id.to_string(),
                epoch_index: slip.epoch_index,
                reasons: slip.reasons.clone(),
            })
        }
    }
}

fn prepare_split_wide_lane_arc(
    satellite_id: &str,
    arc: &[DualArcSample<'_>],
    wide_lane: WideLanePrepOptions,
    slips: &[DualSlipEvent],
) -> Result<WideLaneArcPrepared, WideLanePrepError> {
    let slip_epochs = slips
        .iter()
        .map(|slip| slip.epoch_index)
        .collect::<BTreeSet<_>>();
    let segments = split_dual_arc(arc, &slip_epochs);
    let mut entries = Vec::new();
    let mut cycles = BTreeMap::new();
    let dropped = Vec::new();
    let mut split_arcs = Vec::new();

    for (segment_idx, segment) in segments {
        if segment.len() < wide_lane.min_epochs {
            continue;
        }
        let ambiguity_id = split_ambiguity_id(satellite_id, segment_idx);
        let split_arc = split_arc_metadata(satellite_id, &ambiguity_id, &segment);
        let (arc_entries, arc_cycles, arc_split) = estimate_tagged_wide_lane_arc(
            &ambiguity_id,
            &ambiguity_id,
            &segment,
            wide_lane,
            Some(split_arc),
        )?;
        entries.extend(arc_entries);
        cycles.extend(arc_cycles);
        split_arcs.extend(arc_split);
    }

    if cycles.is_empty() {
        Ok((
            Vec::new(),
            BTreeMap::new(),
            vec![satellite_id.to_string()],
            split_arcs,
        ))
    } else {
        Ok((entries, cycles, dropped, split_arcs))
    }
}

fn estimate_tagged_wide_lane_arc(
    error_id: &AmbiguityId,
    ambiguity_id: &AmbiguityId,
    arc: &[DualArcSample<'_>],
    wide_lane: WideLanePrepOptions,
    split_arc: Option<PppSplitArc>,
) -> Result<TaggedWideLaneArc, WideLanePrepError> {
    let fixed = estimate_wide_lane_integer(error_id, arc, wide_lane)?;
    let entries = arc
        .iter()
        .map(|sample| {
            let mut observation = sample.observation.clone();
            observation.ambiguity_id = ambiguity_id.to_string();
            (sample.epoch_index, observation)
        })
        .collect();
    Ok((
        entries,
        BTreeMap::from([(ambiguity_id.to_string(), fixed)]),
        split_arc,
    ))
}

fn estimate_wide_lane_integer(
    ambiguity_id: &AmbiguityId,
    arc: &[DualArcSample<'_>],
    wide_lane: WideLanePrepOptions,
) -> Result<i64, WideLanePrepError> {
    let mut cycles = Vec::with_capacity(arc.len());
    for sample in arc {
        let value = wide_lane_cycles(sample.observation).map_err(|reason| {
            WideLanePrepError::WideLaneFailed {
                ambiguity_id: ambiguity_id.to_string(),
                reason,
            }
        })?;
        cycles.push(value);
    }

    ambiguity::estimate_wide_lane_integer(&cycles, wide_lane.min_epochs, wide_lane.tolerance_cycles)
        .map_err(|err| match err {
            ambiguity::WideLaneEstimateError::TooFewEpochs { count, minimum } => {
                WideLanePrepError::TooFewWideLaneEpochs {
                    ambiguity_id: ambiguity_id.to_string(),
                    count,
                    minimum,
                }
            }
            ambiguity::WideLaneEstimateError::NotInteger {
                mean_cycles,
                fixed_cycles,
            } => WideLanePrepError::WideLaneNotInteger {
                ambiguity_id: ambiguity_id.to_string(),
                mean_cycles,
                fixed_cycles,
            },
        })
}

fn wide_lane_cycles(observation: &DualFrequencyObservation) -> Result<f64, CarrierPhaseError> {
    crate::carrier_phase::wide_lane_cycles(
        observation.phi1_cyc,
        observation.phi2_cyc,
        observation.p1_m,
        observation.p2_m,
        observation.f1_hz,
        observation.f2_hz,
    )
}

fn cycle_slips_for_dual_arc<'a>(
    arc: &'a [DualArcSample<'a>],
    options: CycleSlipOptions,
) -> Vec<DualSlipEvent> {
    let arc_epochs = arc
        .iter()
        .map(|sample| dual_arc_epoch(sample.observation, sample.gap_time_s))
        .collect::<Vec<_>>();
    let results = detect_cycle_slips(&arc_epochs, options).expect("validated cycle-slip arc");
    arc.iter()
        .zip(results)
        .filter_map(|(sample, result)| {
            if result.slip {
                Some(DualSlipEvent {
                    epoch_index: sample.epoch_index,
                    reasons: result.reasons,
                })
            } else {
                None
            }
        })
        .collect()
}

fn dual_arc_epoch(observation: &DualFrequencyObservation, gap_time_s: Option<f64>) -> ArcEpoch {
    ArcEpoch {
        phi1_cycles: Some(observation.phi1_cyc),
        phi2_cycles: Some(observation.phi2_cyc),
        p1_m: Some(observation.p1_m),
        p2_m: Some(observation.p2_m),
        lli1: observation.lli1,
        lli2: observation.lli2,
        f1_hz: Some(observation.f1_hz),
        f2_hz: Some(observation.f2_hz),
        gap_time_s,
    }
}

fn split_dual_arc<'a>(
    arc: &'a [DualArcSample<'a>],
    slip_epochs: &BTreeSet<usize>,
) -> Vec<(usize, Vec<DualArcSample<'a>>)> {
    let mut segments = Vec::new();
    let mut current = Vec::new();
    let mut current_idx = 1;
    for sample in arc {
        if slip_epochs.contains(&sample.epoch_index) {
            if !current.is_empty() {
                segments.push((current_idx, current));
            }
            current = vec![*sample];
            current_idx += 1;
        } else {
            current.push(*sample);
        }
    }
    if !current.is_empty() {
        segments.push((current_idx, current));
    }
    segments
}

fn split_ambiguity_id(satellite_id: &str, segment_idx: usize) -> AmbiguityId {
    AmbiguityId::new(format!("{satellite_id}#{segment_idx}"))
}

fn split_arc_metadata(
    satellite_id: &str,
    ambiguity_id: &AmbiguityId,
    segment: &[DualArcSample<'_>],
) -> PppSplitArc {
    PppSplitArc {
        satellite_id: satellite_id.to_string(),
        ambiguity_id: ambiguity_id.to_string(),
        start_epoch_index: segment.first().map(|s| s.epoch_index).unwrap_or(0),
        end_epoch_index: segment.last().map(|s| s.epoch_index).unwrap_or(0),
        n_epochs: segment.len(),
    }
}

fn dual_epochs_from_entries(
    entries: Vec<(usize, DualFrequencyObservation)>,
) -> Vec<PreparedDualFrequencyEpoch> {
    let mut by_epoch = BTreeMap::<usize, Vec<DualFrequencyObservation>>::new();
    for (epoch_index, observation) in entries {
        by_epoch.entry(epoch_index).or_default().push(observation);
    }
    by_epoch
        .into_iter()
        .map(|(epoch_index, mut observations)| {
            observations.sort_by(|a, b| {
                (a.satellite_id.as_str(), a.ambiguity_id.as_str())
                    .cmp(&(b.satellite_id.as_str(), b.ambiguity_id.as_str()))
            });
            PreparedDualFrequencyEpoch {
                epoch_index,
                observations,
            }
        })
        .collect()
}

fn filter_dual_epochs_by_wide_lanes(
    dual_epochs: &[PreparedDualFrequencyEpoch],
    wide_lane_cycles: &BTreeMap<String, i64>,
) -> Vec<PreparedDualFrequencyEpoch> {
    let keep = wide_lane_cycles.keys().cloned().collect::<BTreeSet<_>>();
    dual_epochs
        .iter()
        .filter_map(|epoch| {
            let observations = epoch
                .observations
                .iter()
                .filter(|observation| keep.contains(&observation.ambiguity_id))
                .cloned()
                .collect::<Vec<_>>();
            if observations.is_empty() {
                None
            } else {
                Some(PreparedDualFrequencyEpoch {
                    epoch_index: epoch.epoch_index,
                    observations,
                })
            }
        })
        .collect()
}

type IonosphereFreeNarrowLane = (
    Vec<PreparedFloatEpoch>,
    BTreeMap<String, f64>,
    BTreeMap<String, f64>,
);

fn ionosphere_free_narrow_lane_epochs(
    dual_epochs: &[PreparedDualFrequencyEpoch],
    wide_lane_cycles: &BTreeMap<String, i64>,
) -> Result<IonosphereFreeNarrowLane, WideLanePrepError> {
    let params = narrow_lane_params(dual_epochs, wide_lane_cycles)?;
    let if_epochs = ionosphere_free_epochs(dual_epochs)?;
    let wavelengths_m = params
        .iter()
        .map(|(id, params)| (id.as_str().to_string(), params.wavelength_m))
        .collect();
    let offsets_m = params
        .iter()
        .map(|(id, params)| (id.as_str().to_string(), params.offset_m))
        .collect();
    Ok((if_epochs, wavelengths_m, offsets_m))
}

fn narrow_lane_params(
    dual_epochs: &[PreparedDualFrequencyEpoch],
    wide_lane_cycles: &BTreeMap<String, i64>,
) -> Result<BTreeMap<AmbiguityId, NarrowLaneParams>, WideLanePrepError> {
    let mut out = BTreeMap::new();
    for observation in dual_epochs.iter().flat_map(|epoch| &epoch.observations) {
        let ambiguity_id = AmbiguityId::new(observation.ambiguity_id.as_str());
        let wide_lane = wide_lane_cycles
            .get(ambiguity_id.as_str())
            .copied()
            .ok_or_else(|| {
                WideLanePrepError::MissingWideLaneAmbiguity(ambiguity_id.as_str().to_string())
            })?;
        let params = narrow_lane_param(observation.f1_hz, observation.f2_hz, wide_lane as f64)?;
        if let Some(prev) = out.get(&ambiguity_id) {
            ensure_consistent_narrow_lane_params(&ambiguity_id, params, *prev)?;
        } else {
            out.insert(ambiguity_id, params);
        }
    }
    Ok(out)
}

fn narrow_lane_param(
    f1_hz: f64,
    f2_hz: f64,
    wide_lane_cycles: f64,
) -> Result<NarrowLaneParams, WideLanePrepError> {
    ambiguity::narrow_lane_params(f1_hz, f2_hz, wide_lane_cycles).map_err(|reason| {
        WideLanePrepError::IonosphereFreeFailed {
            satellite_id: String::new(),
            reason,
        }
    })
}

fn ensure_consistent_narrow_lane_params(
    ambiguity_id: &AmbiguityId,
    params: NarrowLaneParams,
    prev: NarrowLaneParams,
) -> Result<(), WideLanePrepError> {
    if ambiguity::frequencies_match(params.f1_hz, prev.f1_hz)
        && ambiguity::frequencies_match(params.f2_hz, prev.f2_hz)
    {
        Ok(())
    } else {
        Err(WideLanePrepError::InconsistentFrequencies(
            ambiguity_id.to_string(),
        ))
    }
}

fn ionosphere_free_epochs(
    dual_epochs: &[PreparedDualFrequencyEpoch],
) -> Result<Vec<PreparedFloatEpoch>, WideLanePrepError> {
    dual_epochs
        .iter()
        .map(|epoch| {
            Ok(PreparedFloatEpoch {
                epoch_index: epoch.epoch_index,
                observations: ionosphere_free_observations(&epoch.observations)?,
            })
        })
        .collect()
}

fn ionosphere_free_observations(
    observations: &[DualFrequencyObservation],
) -> Result<Vec<PreparedFloatObservation>, WideLanePrepError> {
    observations
        .iter()
        .map(|observation| {
            let code_m = combinations::ionosphere_free(
                observation.p1_m,
                observation.p2_m,
                observation.f1_hz,
                observation.f2_hz,
            )
            .map_err(|reason| WideLanePrepError::IonosphereFreeFailed {
                satellite_id: observation.satellite_id.clone(),
                reason,
            })?;
            let phase_m = combinations::ionosphere_free_phase_cycles(
                observation.phi1_cyc,
                observation.phi2_cyc,
                observation.f1_hz,
                observation.f2_hz,
            )
            .map_err(|reason| WideLanePrepError::IonosphereFreeFailed {
                satellite_id: observation.satellite_id.clone(),
                reason,
            })?;
            Ok(PreparedFloatObservation {
                satellite_id: observation.satellite_id.clone(),
                ambiguity_id: observation.ambiguity_id.clone(),
                code_m,
                phase_m,
            })
        })
        .collect()
}

#[derive(Clone, Copy)]
struct FloatSlipSample<'a> {
    epoch_index: usize,
    gap_time_s: Option<f64>,
    observation: &'a FloatCycleSlipObservation,
}

fn float_cycle_slip_tags(
    epochs: &[FloatCycleSlipEpoch],
    options: CycleSlipOptions,
) -> BTreeMap<(usize, String), AmbiguityId> {
    let mut arcs = BTreeMap::<String, Vec<FloatSlipSample<'_>>>::new();
    for (epoch_index, epoch) in epochs.iter().enumerate() {
        for observation in &epoch.observations {
            arcs.entry(observation.satellite_id.clone())
                .or_default()
                .push(FloatSlipSample {
                    epoch_index,
                    gap_time_s: epoch.gap_time_s,
                    observation,
                });
        }
    }

    let mut tags = BTreeMap::new();
    for (satellite_id, mut arc) in arcs {
        arc.sort_by_key(|sample| sample.epoch_index);
        tags.extend(float_arc_tags(&satellite_id, &arc, options));
    }
    tags
}

fn float_arc_tags(
    satellite_id: &str,
    arc: &[FloatSlipSample<'_>],
    options: CycleSlipOptions,
) -> BTreeMap<(usize, String), AmbiguityId> {
    let carrier_phase_samples = float_carrier_phase_arc(arc);
    if carrier_phase_samples.is_empty() {
        return BTreeMap::new();
    }
    let arc_epochs = carrier_phase_samples
        .iter()
        .map(|(_, epoch)| *epoch)
        .collect::<Vec<_>>();
    let slip_epochs = detect_cycle_slips(&arc_epochs, options)
        .expect("validated cycle-slip arc")
        .into_iter()
        .zip(carrier_phase_samples.iter())
        .filter_map(|(result, (epoch_index, _))| result.slip.then_some(*epoch_index))
        .collect::<BTreeSet<_>>();
    if slip_epochs.is_empty() {
        return BTreeMap::new();
    }

    let mut out = BTreeMap::new();
    for (segment_idx, segment) in split_float_arc(arc, &slip_epochs) {
        let ambiguity_id = split_ambiguity_id(satellite_id, segment_idx);
        for sample in segment {
            out.insert(
                (sample.epoch_index, satellite_id.to_string()),
                ambiguity_id.clone(),
            );
        }
    }
    out
}

fn float_carrier_phase_arc(arc: &[FloatSlipSample<'_>]) -> Vec<(usize, ArcEpoch)> {
    arc.iter()
        .filter_map(|sample| {
            let raw = sample.observation.raw.as_ref()?;
            Some((sample.epoch_index, dual_arc_epoch(raw, sample.gap_time_s)))
        })
        .collect()
}

fn split_float_arc<'a>(
    arc: &'a [FloatSlipSample<'a>],
    slip_epochs: &BTreeSet<usize>,
) -> Vec<(usize, Vec<FloatSlipSample<'a>>)> {
    let mut segments = Vec::new();
    let mut current = Vec::new();
    let mut current_idx = 1;
    for sample in arc {
        if slip_epochs.contains(&sample.epoch_index) {
            if !current.is_empty() {
                segments.push((current_idx, current));
            }
            current = vec![*sample];
            current_idx += 1;
        } else {
            current.push(*sample);
        }
    }
    if !current.is_empty() {
        segments.push((current_idx, current));
    }
    segments
}
