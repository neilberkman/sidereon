//! Multi-source SP3 combination: clock-datum alignment across analysis centers.
//!
//! Precise clock products from different analysis centers are referenced to
//! different station/ensemble clocks, so their raw clock values differ by a
//! per-epoch common offset - the reference-clock difference - that drifts over
//! the day. Before clocks from two centers can be compared or combined, that
//! datum must be removed. [`clock_reference_offset`] estimates it robustly (the
//! median, over the satellites both products report at each epoch, of
//! `other - reference`); subtract it from `other`'s clocks to put both products
//! on `reference`'s datum.
//!
//! Orbit positions need no such treatment: every center reports ITRF
//! center-of-mass coordinates, so cross-center position differences are already
//! directly comparable.

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::math::vec3;
use crate::astro::time::civil::mjd_from_jd;
use crate::astro::time::gnss;
use crate::astro::time::model::Instant;

use super::interp::instant_to_j2000_seconds;
use super::{RawNode, Sp3, Sp3DataType, Sp3Flags, Sp3Header, Sp3State};
use crate::constants::{GPS_EPOCH_TO_J2000_S, KM_TO_M};
use crate::frame::ItrfPositionM;
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::tolerances::WHOLE_SECOND_EPS_S;
use crate::validate;
use crate::{Error, Result};

const MAX_EXACT_CLIQUE_NODES: usize = 32;

/// One epoch's reference-clock offset of `other` relative to `reference`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockReferenceOffset {
    /// The matched epoch.
    pub epoch: Instant,
    /// `other - reference` clock datum at this epoch, in seconds. Positive means
    /// `other`'s clock datum runs ahead of `reference`'s; subtract it from
    /// `other`'s clocks to align them to `reference`.
    pub offset_s: f64,
    /// Number of satellites that contributed to the (median) estimate.
    pub satellites: usize,
}

/// Estimate the per-epoch reference-clock offset of `other` relative to
/// `reference`.
///
/// For each epoch present in both products, the offset is the median over the
/// satellites both report (each with a finite clock) of
/// `other_clock - reference_clock`. The median makes the estimate robust to a
/// single satellite whose clock one center has wrong - but only with enough
/// satellites, so `min_common` is the minimum number of common clocked
/// satellites required to emit an offset for an epoch (a sound robust median
/// wants at least three, so one outlier can be outvoted). Epochs with fewer
/// common clocks are omitted rather than reported as a fragile one- or
/// two-satellite estimate.
///
/// Epochs are matched by their J2000 second floored to a whole second (the same
/// node-axis convention the interpolator uses). Non-finite clock differences are
/// skipped. Epochs present in only one product, or below `min_common`, are
/// omitted from the result.
///
/// The floored-whole-second key assumes the input cadence is at least one second,
/// which holds for every standard SP3 product (15 min, 5 min, 1 min, ... down to
/// 1 s). Two distinct epochs less than a second apart would collapse onto the
/// same key and be matched as one; the same applies to the floored key in
/// [`MergeReport::per_epoch_agreement`]. This is kept deliberately aligned with
/// the interpolator's node axis rather than refined to sub-second resolution, so
/// that matching here and interpolation downstream use one consistent grid.
pub fn clock_reference_offset(
    reference: &Sp3,
    other: &Sp3,
    min_common: usize,
) -> Vec<ClockReferenceOffset> {
    let mut other_index: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for (idx, epoch) in other.epochs.iter().enumerate() {
        if let Some(seconds) = instant_to_j2000_seconds(epoch) {
            other_index.insert(seconds.floor() as i64, idx);
        }
    }

    let mut offsets = Vec::new();

    for (ref_idx, epoch) in reference.epochs.iter().enumerate() {
        let Some(ref_seconds) = instant_to_j2000_seconds(epoch) else {
            continue;
        };
        let Some(&other_idx) = other_index.get(&(ref_seconds.floor() as i64)) else {
            continue;
        };

        let (Ok(ref_states), Ok(other_states)) =
            (reference.states_at(ref_idx), other.states_at(other_idx))
        else {
            continue;
        };

        let mut diffs: Vec<f64> = Vec::new();
        for (sat, ref_state) in ref_states.iter() {
            let Some(ref_clock) = ref_state.clock_s else {
                continue;
            };
            if let Some(other_state) = other_states.get(sat) {
                if let Some(other_clock) = other_state.clock_s {
                    let diff = other_clock - ref_clock;
                    // SP3 should not carry NaN/inf clocks, but the parser can
                    // accept them; merge infrastructure must not panic on data.
                    if diff.is_finite() {
                        diffs.push(diff);
                    }
                }
            }
        }

        if diffs.len() >= min_common.max(1) {
            if let Some(offset_s) = median(&mut diffs) {
                offsets.push(ClockReferenceOffset {
                    epoch: *epoch,
                    offset_s,
                    satellites: diffs.len(),
                });
            }
        }
    }

    offsets
}

fn median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    // Inputs are pre-filtered to finite values; total_cmp never panics regardless.
    values.sort_by(f64::total_cmp);

    let n = values.len();
    if n % 2 == 1 {
        Some(values[n / 2])
    } else {
        Some((values[n / 2 - 1] + values[n / 2]) / 2.0)
    }
}

// ===========================================================================
// Multi-source merge
// ===========================================================================

/// How the agreeing (consensus) sources for a cell are combined into the merged
/// value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeCombine {
    /// Arithmetic mean of the consensus sources. The clustering step has already
    /// removed outliers, so the mean uses every agreeing measurement. Default.
    Mean,
    /// Component-wise median of the consensus sources.
    Median,
    /// The value from the highest-precedence (earliest-listed) consensus source.
    Precedence,
}

/// Options for [`merge`].
#[derive(Debug, Clone, PartialEq)]
pub struct MergeOptions {
    /// Maximum 3D position difference (meters) for two sources to be in
    /// agreement.
    pub position_tolerance_m: f64,
    /// Maximum clock difference (seconds, after datum alignment) for two sources
    /// to be in agreement.
    pub clock_tolerance_s: f64,
    /// Minimum number of mutually-agreeing sources required to accept a cell that
    /// has two or more sources. A cell with a single source is always carried
    /// through (gap fill, recorded as `single_source`); a cell with several
    /// sources but no agreeing subset this large is quarantined rather than
    /// averaged across disagreeing centers.
    pub min_agree: usize,
    /// Minimum common clocked satellites for the per-epoch clock-datum estimate
    /// between two sources (see [`clock_reference_offset`]).
    pub clock_min_common: usize,
    /// How to combine the agreeing sources.
    pub combine: MergeCombine,
    /// Optional target epoch interval, in seconds. When unset the coarsest input
    /// interval is used. Finer inputs are decimated onto this grid by exact
    /// subset selection (never interpolated); inputs whose interval does not
    /// evenly divide it are rejected.
    pub target_epoch_interval_s: Option<f64>,
    /// Optional constellation/system filter. When set, only satellites whose
    /// system is in this set are considered for the merged product.
    pub systems: Option<BTreeSet<GnssSystem>>,
}

impl Default for MergeOptions {
    /// Defaults tuned for the common case of ~3 analysis centers: agreement is a
    /// 2-of-3 majority (`min_agree = 2`); combine the agreeing subset by mean.
    fn default() -> Self {
        Self {
            position_tolerance_m: 0.5,
            clock_tolerance_s: 5.0e-9,
            min_agree: 2,
            clock_min_common: 5,
            combine: MergeCombine::Mean,
            target_epoch_interval_s: None,
            systems: None,
        }
    }
}

/// One (epoch, satellite) cell the merge handled with a caveat. Nothing is
/// dropped or averaged silently - every such cell is recorded here.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeFlag {
    /// The epoch.
    pub epoch: Instant,
    /// The satellite.
    pub satellite: GnssSatelliteId,
    /// The source indices (into the input slice) this flag refers to: for
    /// `single_source`, the lone contributor; for `quarantined`, all sources
    /// that disagreed; for `position_outliers`, the sources rejected from an
    /// otherwise-accepted consensus.
    pub sources: Vec<usize>,
}

/// Per-(epoch, satellite) agreement statistics for one accepted consensus cell:
/// how tightly the consensus member values cluster about the combined value that
/// was actually written to the merged product.
///
/// The dispersion is measured about the *combined* value (the mean, median, or
/// precedence pick - whatever the strategy wrote), not about the cluster centroid,
/// so it reflects the agreement of the product the merge emitted. A single-source
/// cell has one member and zero dispersion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AgreementMetric {
    /// The epoch.
    pub epoch: Instant,
    /// The satellite.
    pub satellite: GnssSatelliteId,
    /// Number of sources in the accepted position consensus (>= 1).
    pub position_members: usize,
    /// RMS, over the position-consensus members, of the 3D distance from the
    /// combined position, meters. Zero for a single-source cell.
    pub position_rms_m: f64,
    /// Largest 3D distance of any position-consensus member from the combined
    /// position, meters.
    pub position_max_m: f64,
    /// Number of sources in the accepted clock consensus (0 when the cell carries
    /// no clock).
    pub clock_members: usize,
    /// RMS, over the clock-consensus members, of the deviation from the combined
    /// clock, seconds; `None` when the cell carries no clock.
    pub clock_rms_s: Option<f64>,
    /// Largest absolute clock deviation from the combined clock, seconds; `None`
    /// when the cell carries no clock.
    pub clock_max_s: Option<f64>,
}

/// Per-epoch aggregate of [`AgreementMetric`] over the satellites combined at that
/// epoch, restricted to cells with a *multi-source* consensus (a single source
/// has no measurable dispersion, so it is excluded from the aggregate spread).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochAgreement {
    /// The epoch.
    pub epoch: Instant,
    /// Satellites at this epoch with a multi-source position consensus.
    pub satellites: usize,
    /// Member-count-weighted pooled RMS of the per-cell position dispersion over
    /// those satellites, meters (i.e. the RMS of every member-to-combined 3D
    /// distance pooled across the epoch).
    pub position_rms_m: f64,
    /// Worst per-cell position dispersion at this epoch, meters.
    pub position_max_m: f64,
    /// As `position_rms_m` for the clock channel; `None` when no multi-source
    /// clock consensus existed at this epoch.
    pub clock_rms_s: Option<f64>,
    /// Worst per-cell clock dispersion at this epoch, seconds; `None` as above.
    pub clock_max_s: Option<f64>,
}

/// Audit trail for a [`merge`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MergeReport {
    /// Cells where two or more sources disagreed beyond tolerance with no
    /// agreeing subset of `min_agree` - omitted from the merged product.
    pub quarantined: Vec<MergeFlag>,
    /// Cells carried from a single source (no cross-check was possible).
    pub single_source: Vec<MergeFlag>,
    /// Cells accepted by consensus where one or more sources were rejected as
    /// position outliers.
    pub position_outliers: Vec<MergeFlag>,
    /// Per-(epoch, satellite) agreement statistics for every accepted cell, in
    /// output (epoch, then satellite) order - one entry per cell written to the
    /// merged product. Quantifies how tightly the consensus sources clustered
    /// about the combined value (Gap: per-epoch quality metrics).
    pub agreement: Vec<AgreementMetric>,
}

impl MergeReport {
    /// Fraction of accepted cells that were carried from a single source, in
    /// `0.0..=1.0`; `None` when no cells were accepted.
    ///
    /// This is the blind-spot companion to the agreement-RMS accessors, which
    /// quantify dispersion only over *multi-source* cells. A product can show a
    /// tight (or `None`) agreement RMS yet be largely un-cross-checked: those
    /// gap-fill cells (also enumerated in [`MergeReport::single_source`]) had no
    /// second source to compare against. Read this alongside the RMS so a clean
    /// dispersion is not mistaken for a fully corroborated product.
    pub fn single_source_fraction(&self) -> Option<f64> {
        let accepted = self.agreement.len();
        (accepted > 0).then(|| self.single_source.len() as f64 / accepted as f64)
    }

    /// Member-count-weighted pooled RMS of the per-cell position dispersion over
    /// every accepted cell with a multi-source consensus, meters. `None` when no
    /// cell had two or more position-consensus members.
    ///
    /// The pool is exact: each cell contributes its summed squared member-to-
    /// combined distances (`position_rms_m^2 * position_members`), normalised by
    /// the total member count, so the result is the RMS of all member-to-combined
    /// distances across the whole product.
    ///
    /// This covers only multi-source cells; single-source gap-fill cells are
    /// excluded (they have no dispersion). A small or `None` result therefore does
    /// not by itself mean the whole product was corroborated - check
    /// [`MergeReport::single_source_fraction`] for the un-cross-checked share.
    pub fn position_agreement_rms_m(&self) -> Option<f64> {
        pooled_rms(
            self.agreement
                .iter()
                .filter(|m| m.position_members >= 2)
                .map(|m| (m.position_rms_m, m.position_members)),
        )
    }

    /// Largest single-cell position dispersion over all accepted cells, meters.
    /// `None` when there are no accepted cells.
    pub fn position_agreement_max_m(&self) -> Option<f64> {
        self.agreement
            .iter()
            .map(|m| m.position_max_m)
            .fold(None, |acc, v| Some(fold_max(acc, v)))
    }

    /// As [`Self::position_agreement_rms_m`] for the clock channel, seconds.
    pub fn clock_agreement_rms_s(&self) -> Option<f64> {
        pooled_rms(self.agreement.iter().filter_map(|m| {
            m.clock_rms_s
                .filter(|_| m.clock_members >= 2)
                .map(|rms| (rms, m.clock_members))
        }))
    }

    /// Largest single-cell clock dispersion over all accepted cells, seconds.
    pub fn clock_agreement_max_s(&self) -> Option<f64> {
        self.agreement
            .iter()
            .filter_map(|m| m.clock_max_s)
            .fold(None, |acc, v| Some(fold_max(acc, v)))
    }

    /// Per-epoch aggregate agreement, in output-epoch order. Each entry pools the
    /// multi-source cells at that epoch (see [`EpochAgreement`]); epochs whose
    /// cells were all single-source are still listed with `satellites == 0` and a
    /// zero position spread so the caller sees every output epoch.
    pub fn per_epoch_agreement(&self) -> Vec<EpochAgreement> {
        let mut out: Vec<EpochAgreement> = Vec::new();
        let mut current_key: Option<i64> = None;
        for m in &self.agreement {
            let key = instant_to_j2000_seconds(&m.epoch).map(|s| s.floor() as i64);
            if current_key != key || out.is_empty() {
                out.push(EpochAgreement {
                    epoch: m.epoch,
                    satellites: 0,
                    position_rms_m: 0.0,
                    position_max_m: 0.0,
                    clock_rms_s: None,
                    clock_max_s: None,
                });
                current_key = key;
            }
            let agg = out.last_mut().expect("just pushed");
            agg.position_max_m = agg.position_max_m.max(m.position_max_m);
            if m.position_members >= 2 {
                agg.satellites += 1;
            }
            // Only multi-source clock cells contribute to the epoch clock max,
            // matching the RMS path: a single-member cell has zero dispersion and
            // must not leave clock_max_s = Some(0.0) while clock_rms_s is None.
            if let Some(max) = m.clock_max_s.filter(|_| m.clock_members >= 2) {
                agg.clock_max_s = Some(fold_max(agg.clock_max_s, max));
            }
        }

        // Pooled RMS per epoch needs the sum of squared distances, which the per
        // entry RMS encodes; recompute it in a second pass grouped by epoch key.
        for agg in &mut out {
            let key = instant_to_j2000_seconds(&agg.epoch).map(|s| s.floor() as i64);
            agg.position_rms_m = pooled_rms(
                self.agreement
                    .iter()
                    .filter(|m| {
                        m.position_members >= 2
                            && instant_to_j2000_seconds(&m.epoch).map(|s| s.floor() as i64) == key
                    })
                    .map(|m| (m.position_rms_m, m.position_members)),
            )
            .unwrap_or(0.0);
            agg.clock_rms_s = pooled_rms(
                self.agreement
                    .iter()
                    .filter(|m| instant_to_j2000_seconds(&m.epoch).map(|s| s.floor() as i64) == key)
                    .filter_map(|m| {
                        m.clock_rms_s
                            .filter(|_| m.clock_members >= 2)
                            .map(|rms| (rms, m.clock_members))
                    }),
            );
        }

        out
    }
}

/// Pool per-cell RMS values weighted by member count into one RMS:
/// `sqrt(sum(rms_i^2 * n_i) / sum(n_i))`. `None` when the iterator is empty.
fn pooled_rms(cells: impl Iterator<Item = (f64, usize)>) -> Option<f64> {
    let mut sumsq = 0.0_f64;
    let mut total = 0_usize;
    for (rms, n) in cells {
        sumsq += rms * rms * n as f64;
        total += n;
    }
    (total > 0).then(|| (sumsq / total as f64).sqrt())
}

/// `max` reduction over an `Option` accumulator (`None` is the empty identity).
fn fold_max(acc: Option<f64>, value: f64) -> f64 {
    match acc {
        Some(current) if current >= value => current,
        _ => value,
    }
}

/// Merge several SP3 products from different analysis centers into one
/// consistent precise-ephemeris dataset.
///
/// Orthogonal to time-stitching: this combines providers at the **same** epochs.
/// Inputs must be on one common, uniform epoch grid. Mixed-cadence products are
/// rejected rather than unioned onto a finer grid; callers that need that must
/// resample first. For every (epoch, satellite) cell on the common grid:
///
/// - **Union satellite coverage.** A satellite present in any input may appear
///   in the output, but only on the shared grid and only when doing so preserves
///   a coherent source/consensus arc.
/// - **Position consensus.** With one source the value is carried through
///   (`single_source`). With several, the largest subset of sources mutually
///   within `position_tolerance_m` is found; if it has at least `min_agree`
///   members it is combined per `combine` and any sources outside it are recorded
///   as `position_outliers`. If no such subset exists the cell is `quarantined`
///   (omitted) - never averaged across disagreeing centers.
/// - **Clock consensus.** Clocks are first put on a common datum (each source
///   aligned to the first via [`clock_reference_offset`]), then combined by the
///   same agreement rule; a cell with no clock consensus carries no clock. A
///   non-reference source whose datum cannot be estimated at an epoch (below
///   `clock_min_common` common clocks) contributes **no** clock there rather than
///   an unaligned one - its position is still merged.
///
/// `Precedence` is resolved per satellite arc: once a satellite is assigned to
/// the highest-precedence source that carries it, that satellite never switches
/// centers at adjacent epochs. If that source is missing a cell, the cell is
/// omitted rather than filled from a lower-precedence source.
///
/// All inputs must share an exact SP3 time-system label and exact
/// coordinate-system label (epochs are matched across products in that scale);
/// mismatches are rejected. The merged record flags are the union (OR) of the
/// contributing sources' flags - in particular a `clock_event` on any
/// clock-consensus member is preserved, so the interpolator still splits the
/// clock arc. The merged header is **synthetic**: its first-epoch fields
/// describe the union's first epoch and its data type is position-only.
///
/// Pure and deterministic: order the inputs by center precedence and ties (equal
/// cluster sizes, `Precedence` combine) resolve to the earliest-listed source.
/// The merged product's interpolation nodes are the consensus values, so it
/// samples and interpolates like any other [`Sp3`] (it is a derived combination,
/// not a byte-faithful copy of any one center). Consensus is exact max-clique for
/// normal source counts and uses a deterministic greedy fallback above the exact
/// search cap, so hostile disagreement graphs remain bounded.
pub fn merge(sources: &[Sp3], opts: &MergeOptions) -> Result<(Sp3, MergeReport)> {
    if sources.is_empty() {
        return Err(Error::InvalidInput(
            "merge requires at least one SP3 product".into(),
        ));
    }

    // Inputs must be combinable: epochs are matched in one exact product time
    // system, and positions are only comparable in an exactly common coordinate
    // system / frame. Do not silently alias labels such as QZS/GPS or
    // IGS20/IGc20 here: without an explicit transform, a differently labeled
    // product contract is a different product contract.
    let base = &sources[0].header;
    for s in &sources[1..] {
        if s.header.time_system != base.time_system {
            return Err(Error::InvalidInput(format!(
                "merge inputs have mismatched SP3 time systems ({:?} vs {:?})",
                base.time_system, s.header.time_system
            )));
        }
        if s.header.coordinate_system.trim() != base.coordinate_system.trim() {
            return Err(Error::InvalidInput(format!(
                "merge inputs have mismatched coordinate systems ({:?} vs {:?})",
                base.coordinate_system, s.header.coordinate_system
            )));
        }
    }

    // floored-J2000-second -> epoch index, per source.
    let epoch_index: Vec<BTreeMap<i64, usize>> = sources
        .iter()
        .map(|s| {
            s.epochs
                .iter()
                .enumerate()
                .filter_map(|(i, ep)| {
                    instant_to_j2000_seconds(ep).map(|sec| (sec.floor() as i64, i))
                })
                .collect()
        })
        .collect();

    let epoch_interval_s = resolve_common_epoch_interval(sources, opts.target_epoch_interval_s)?;

    // Per-source per-epoch clock-datum offset relative to source 0. Source 0 is
    // the datum, so its offset is identically zero.
    let clock_offset: Vec<BTreeMap<i64, f64>> = sources
        .iter()
        .enumerate()
        .map(|(idx, s)| {
            if idx == 0 {
                BTreeMap::new()
            } else {
                clock_reference_offset(&sources[0], s, opts.clock_min_common)
                    .into_iter()
                    .filter_map(|o| {
                        instant_to_j2000_seconds(&o.epoch)
                            .map(|sec| (sec.floor() as i64, o.offset_s))
                    })
                    .collect()
            }
        })
        .collect();

    // Intersection of epochs (by floored second), keeping source 0's
    // representative Instant. Mixing a 15-minute product with 5-minute products
    // must not emit the union grid; if all inputs share a cadence but differ in
    // coverage, only the epochs present in every product are combined.
    let mut epoch_keys: BTreeMap<i64, Instant> = sources[0]
        .epochs
        .iter()
        .filter_map(|ep| instant_to_j2000_seconds(ep).map(|sec| (sec.floor() as i64, *ep)))
        .collect();

    for index in epoch_index.iter().skip(1) {
        epoch_keys.retain(|key, _| index.contains_key(key));
    }

    // Decimate onto the resolved common-interval grid (anchored at the earliest
    // common epoch): keep only epochs that land on the grid, dropping off-grid
    // epochs by exact subset selection (never interpolation). A no-op when the
    // inputs already share the interval; the real decimation when finer inputs
    // are mixed with a coarser one, or an explicit coarser target is requested.
    if let Some((&anchor, _)) = epoch_keys.iter().next() {
        let step = epoch_interval_s.round() as i64;
        if step > 0 {
            epoch_keys.retain(|&key, _| (key - anchor).rem_euclid(step) == 0);
        }
    }

    if epoch_keys.is_empty() {
        return Err(Error::InvalidInput(
            "merge inputs have no common epochs on a shared time grid".into(),
        ));
    }

    let precedence_source_for_sat = if opts.combine == MergeCombine::Precedence {
        Some(precedence_sources_for_satellites(
            sources,
            &epoch_index,
            &epoch_keys,
            opts.systems.as_ref(),
        ))
    } else {
        None
    };

    let allowed_system = |sat: &GnssSatelliteId| {
        opts.systems
            .as_ref()
            .is_none_or(|systems| systems.contains(&sat.system))
    };

    if let Some(systems) = &opts.systems {
        if systems.is_empty() {
            return Err(Error::InvalidInput(
                "merge systems filter must not be empty".into(),
            ));
        }
    }

    let mut out_epochs: Vec<Instant> = Vec::with_capacity(epoch_keys.len());
    let mut out_states: Vec<BTreeMap<GnssSatelliteId, Sp3State>> =
        Vec::with_capacity(epoch_keys.len());
    let mut out_raw: Vec<BTreeMap<GnssSatelliteId, RawNode>> = Vec::with_capacity(epoch_keys.len());
    let mut report = MergeReport::default();
    let mut all_sats: BTreeSet<GnssSatelliteId> = BTreeSet::new();

    for (&key, &epoch) in &epoch_keys {
        out_epochs.push(epoch);
        let mut states: BTreeMap<GnssSatelliteId, Sp3State> = BTreeMap::new();
        let mut raws: BTreeMap<GnssSatelliteId, RawNode> = BTreeMap::new();

        // Satellites present at this epoch in any source, after any requested
        // constellation filter.
        let mut sats: BTreeSet<GnssSatelliteId> = BTreeSet::new();
        for (idx, s) in sources.iter().enumerate() {
            if let Some(&ei) = epoch_index[idx].get(&key) {
                if let Ok(map) = s.states_at(ei) {
                    sats.extend(map.keys().copied().filter(|sat| allowed_system(sat)));
                }
            }
        }

        for sat in sats {
            // (source_idx, position_m, flags) and (source_idx, datum-aligned
            // clock_s, flags). A non-reference source contributes a clock only
            // when its datum offset could be estimated at this epoch; otherwise
            // its clock would be unaligned, so it is omitted (the position is
            // still gathered).
            let preferred_source = precedence_source_for_sat
                .as_ref()
                .and_then(|by_sat| by_sat.get(&sat).copied());

            let mut pos: Vec<(usize, [f64; 3], Sp3Flags)> = Vec::new();
            let mut clk: Vec<(usize, f64, Sp3Flags)> = Vec::new();
            for (idx, s) in sources.iter().enumerate() {
                let Some(&ei) = epoch_index[idx].get(&key) else {
                    continue;
                };
                let Ok(map) = s.states_at(ei) else { continue };
                let Some(state) = map.get(&sat) else { continue };
                pos.push((idx, state.position.as_array(), state.flags));
                if let Some(c) = state.clock_s {
                    let offset = if idx == 0 {
                        Some(0.0)
                    } else {
                        clock_offset[idx].get(&key).copied()
                    };
                    if let Some(off) = offset {
                        let aligned = c - off;
                        if aligned.is_finite() {
                            clk.push((idx, aligned, state.flags));
                        }
                    }
                }
            }

            let flag = |srcs: Vec<usize>| MergeFlag {
                epoch,
                satellite: sat,
                sources: srcs,
            };

            // Position consensus -> the merged position and the indices (into
            // `pos`) of the sources that contributed it. In precedence mode the
            // preferred source is fixed per satellite arc; never switch to a
            // lower-precedence source just because the preferred source is
            // missing or outside a different consensus cluster at this epoch.
            let (position_m, pos_members) = if opts.combine == MergeCombine::Precedence {
                let Some(preferred_source) = preferred_source else {
                    continue;
                };
                let Some(preferred_idx) =
                    pos.iter().position(|(src, _, _)| *src == preferred_source)
                else {
                    continue;
                };

                if pos.len() == 1 {
                    report.single_source.push(flag(vec![pos[preferred_idx].0]));
                    (pos[preferred_idx].1, vec![preferred_idx])
                } else {
                    let pts: Vec<[f64; 3]> = pos.iter().map(|(_, p, _)| *p).collect();
                    let cluster = largest_within_containing(&pts, preferred_idx, |a, b| {
                        dist3(a, b) <= opts.position_tolerance_m
                    });
                    if cluster.len() >= opts.min_agree {
                        let rejected: Vec<usize> = (0..pos.len())
                            .filter(|i| !cluster.contains(i))
                            .map(|i| pos[i].0)
                            .collect();
                        if !rejected.is_empty() {
                            report.position_outliers.push(flag(rejected));
                        }
                        (pos[preferred_idx].1, cluster)
                    } else {
                        report
                            .quarantined
                            .push(flag(pos.iter().map(|(i, _, _)| *i).collect()));
                        continue;
                    }
                }
            } else if pos.len() == 1 {
                report.single_source.push(flag(vec![pos[0].0]));
                (pos[0].1, vec![0usize])
            } else {
                let pts: Vec<[f64; 3]> = pos.iter().map(|(_, p, _)| *p).collect();
                let cluster = largest_within(&pts, |a, b| dist3(a, b) <= opts.position_tolerance_m);
                if cluster.len() >= opts.min_agree {
                    let rejected: Vec<usize> = (0..pos.len())
                        .filter(|i| !cluster.contains(i))
                        .map(|i| pos[i].0)
                        .collect();
                    if !rejected.is_empty() {
                        report.position_outliers.push(flag(rejected));
                    }
                    let members: Vec<(usize, [f64; 3])> =
                        cluster.iter().map(|&i| (pos[i].0, pos[i].1)).collect();
                    (combine3(&members, opts.combine), cluster)
                } else {
                    report
                        .quarantined
                        .push(flag(pos.iter().map(|(i, _, _)| *i).collect()));
                    continue;
                }
            };

            // Clock consensus, independent of position -> the merged clock and the
            // indices (into `clk`) of the sources that contributed it.
            let (clock_s, clk_members): (Option<f64>, Vec<usize>) = if clk.is_empty() {
                (None, Vec::new())
            } else if opts.combine == MergeCombine::Precedence {
                match preferred_source
                    .and_then(|src| clk.iter().position(|(clock_src, _, _)| *clock_src == src))
                {
                    None => (None, Vec::new()),
                    Some(preferred_idx) if clk.len() == 1 => {
                        (Some(clk[preferred_idx].1), vec![preferred_idx])
                    }
                    Some(preferred_idx) => {
                        let vals: Vec<f64> = clk.iter().map(|(_, c, _)| *c).collect();
                        let cluster = largest_within_containing(&vals, preferred_idx, |a, b| {
                            (a - b).abs() <= opts.clock_tolerance_s
                        });
                        if cluster.len() >= opts.min_agree {
                            (Some(clk[preferred_idx].1), cluster)
                        } else {
                            (None, Vec::new())
                        }
                    }
                }
            } else if clk.len() == 1 {
                (Some(clk[0].1), vec![0usize])
            } else {
                let vals: Vec<f64> = clk.iter().map(|(_, c, _)| *c).collect();
                let cluster = largest_within(&vals, |a, b| (a - b).abs() <= opts.clock_tolerance_s);
                if cluster.len() >= opts.min_agree {
                    let members: Vec<(usize, f64)> =
                        cluster.iter().map(|&i| (clk[i].0, clk[i].1)).collect();
                    (Some(combine_axis(&members, opts.combine)), cluster)
                } else {
                    (None, Vec::new())
                }
            };

            // Preserve record flags: OR the orbit flags across the position
            // members and the clock flags across the clock members, so a
            // `clock_event` (clock reset) or maneuver on any contributing source
            // survives into the merged product.
            let mut flags = Sp3Flags::default();
            for &i in &pos_members {
                flags.maneuver |= pos[i].2.maneuver;
                flags.orbit_predicted |= pos[i].2.orbit_predicted;
            }
            for &i in &clk_members {
                flags.clock_event |= clk[i].2.clock_event;
                flags.clock_predicted |= clk[i].2.clock_predicted;
            }

            // Per-cell agreement: dispersion of the accepted consensus members
            // about the combined value actually written below.
            let (position_rms_m, position_max_m) =
                position_dispersion(&pos, &pos_members, &position_m);
            let (clock_members_n, clock_rms_s, clock_max_s) = match clock_s {
                Some(c) => {
                    let (rms, max) = clock_dispersion(&clk, &clk_members, c);
                    (clk_members.len(), Some(rms), Some(max))
                }
                None => (0, None, None),
            };
            report.agreement.push(AgreementMetric {
                epoch,
                satellite: sat,
                position_members: pos_members.len(),
                position_rms_m,
                position_max_m,
                clock_members: clock_members_n,
                clock_rms_s,
                clock_max_s,
            });

            all_sats.insert(sat);
            states.insert(
                sat,
                Sp3State {
                    position: ItrfPositionM::new(position_m[0], position_m[1], position_m[2])
                        .expect("valid ITRF position"),
                    clock_s,
                    velocity: None,
                    clock_rate_s_s: None,
                    flags,
                },
            );
            raws.insert(
                sat,
                RawNode {
                    km: [
                        position_m[0] / KM_TO_M,
                        position_m[1] / KM_TO_M,
                        position_m[2] / KM_TO_M,
                    ],
                    clock_us: clock_s.map(|c| c * 1.0e6),
                    clock_event: flags.clock_event,
                },
            );
        }

        out_states.push(states);
        out_raw.push(raws);
    }

    // Base the non-epoch metadata on a source product, but derive the first-epoch
    // header fields from the merged grid itself. Mixed cadence / coverage can make
    // the merged first epoch later than every input's first epoch, so cloning
    // those fields from any input would make the `##` line stale.
    let first_key = instant_to_j2000_seconds(&out_epochs[0]).map(|s| s.floor() as i64);
    let base_idx = sources
        .iter()
        .position(|s| {
            s.epochs
                .first()
                .and_then(instant_to_j2000_seconds)
                .map(|s| s.floor() as i64)
                == first_key
        })
        .or_else(|| {
            sources
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    s.epochs
                        .first()
                        .and_then(instant_to_j2000_seconds)
                        .map(|sec| (sec, i))
                })
                .min_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)))
                .map(|(_, i)| i)
        })
        .unwrap_or(0);
    let first_epoch_header = first_epoch_header_fields(&out_epochs[0]).ok_or_else(|| {
        Error::InvalidInput("merged SP3 first epoch cannot be represented in header fields".into())
    })?;

    let satellites: Vec<_> = all_sats.into_iter().collect();
    let satellite_accuracy_codes = satellites
        .iter()
        .map(|sat| {
            sources[base_idx]
                .header
                .satellites
                .iter()
                .position(|base_sat| base_sat == sat)
                .and_then(|idx| {
                    sources[base_idx]
                        .header
                        .satellite_accuracy_codes
                        .get(idx)
                        .copied()
                })
                .unwrap_or(0)
        })
        .collect();

    let header = Sp3Header {
        num_epochs: out_epochs.len() as u64,
        satellites,
        satellite_accuracy_codes,
        data_type: Sp3DataType::Position,
        gnss_week: first_epoch_header.gnss_week,
        seconds_of_week: first_epoch_header.seconds_of_week,
        epoch_interval_s,
        mjd: first_epoch_header.mjd,
        mjd_fraction: first_epoch_header.mjd_fraction,
        ..sources[base_idx].header.clone()
    };

    let merged = Sp3 {
        header,
        epochs: out_epochs,
        states: out_states,
        interp_raw: out_raw,
        comments: vec![format!("MERGED from {} SP3 products", sources.len())],
        skipped_records: sources.iter().map(|s| s.skipped_records).sum(),
    };

    Ok((merged, report))
}

#[derive(Debug, Clone, Copy)]
struct FirstEpochHeaderFields {
    gnss_week: u32,
    seconds_of_week: f64,
    mjd: u32,
    mjd_fraction: f64,
}

fn first_epoch_header_fields(epoch: &Instant) -> Option<FirstEpochHeaderFields> {
    let split = epoch.julian_date()?;

    let mjd_day = mjd_from_jd(split.jd_whole);
    let mut mjd = mjd_day.floor();
    let mut mjd_fraction = split.fraction + (mjd_day - mjd);
    let fraction_days = mjd_fraction.floor();
    if fraction_days != 0.0 {
        mjd += fraction_days;
        mjd_fraction -= fraction_days;
    }
    if !(0.0..=u32::MAX as f64).contains(&mjd) {
        return None;
    }

    let gps_seconds = instant_to_j2000_seconds(epoch)? + GPS_EPOCH_TO_J2000_S;
    let (gnss_week, seconds_of_week) = gnss::week_and_seconds_of_week(gps_seconds);
    if !(0.0..=u32::MAX as f64).contains(&gnss_week) {
        return None;
    }

    Some(FirstEpochHeaderFields {
        gnss_week: gnss_week as u32,
        seconds_of_week,
        mjd: mjd as u32,
        mjd_fraction,
    })
}

fn dist3(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    vec3::norm3(vec3::sub3(*a, *b))
}

/// RMS and max of the 3D distance of each `members` position (indices into `pos`)
/// from `combined`. `members` is the accepted consensus, always non-empty.
fn position_dispersion(
    pos: &[(usize, [f64; 3], Sp3Flags)],
    members: &[usize],
    combined: &[f64; 3],
) -> (f64, f64) {
    let mut sumsq = 0.0;
    let mut max = 0.0_f64;
    for &i in members {
        let d = dist3(&pos[i].1, combined);
        sumsq += d * d;
        max = max.max(d);
    }
    ((sumsq / members.len().max(1) as f64).sqrt(), max)
}

/// RMS and max of the absolute deviation of each `members` clock (indices into
/// `clk`) from `combined`. `members` is the accepted consensus, always non-empty.
fn clock_dispersion(
    clk: &[(usize, f64, Sp3Flags)],
    members: &[usize],
    combined: f64,
) -> (f64, f64) {
    let mut sumsq = 0.0;
    let mut max = 0.0_f64;
    for &i in members {
        let d = (clk[i].1 - combined).abs();
        sumsq += d * d;
        max = max.max(d);
    }
    ((sumsq / members.len().max(1) as f64).sqrt(), max)
}

fn precedence_sources_for_satellites(
    sources: &[Sp3],
    epoch_index: &[BTreeMap<i64, usize>],
    epoch_keys: &BTreeMap<i64, Instant>,
    systems: Option<&BTreeSet<GnssSystem>>,
) -> BTreeMap<GnssSatelliteId, usize> {
    let mut by_sat = BTreeMap::new();

    for (idx, source) in sources.iter().enumerate() {
        for key in epoch_keys.keys() {
            let Some(&epoch_idx) = epoch_index[idx].get(key) else {
                continue;
            };
            let Ok(states) = source.states_at(epoch_idx) else {
                continue;
            };

            for sat in states.keys() {
                if systems.is_none_or(|allowed| allowed.contains(&sat.system)) {
                    by_sat.entry(*sat).or_insert(idx);
                }
            }
        }
    }

    by_sat
}

/// Resolve the common (output) epoch interval and validate that every input can
/// be decimated onto it without interpolation.
///
/// The common interval is the caller's `target` if given, otherwise the
/// **coarsest** native interval among the inputs (the finest grid every input
/// can supply). An input is compatible only when the common interval is a
/// positive-integer multiple of that input's native interval: then the
/// common-grid epochs are an exact subset of the input's epochs, and the merge's
/// epoch intersection performs the decimation (e.g. a 5-minute product
/// contributes its :00/:15/:30/:45 epochs to a 15-minute merge - no orbit/clock
/// interpolation is introduced). Inputs whose interval does not evenly divide the
/// common interval - a coarser input than the requested grid, or a non-divisible
/// cadence - are rejected as incompatible. Equal-interval inputs (multiple 1) are
/// the same-interval fast path and behave exactly as before.
fn resolve_common_epoch_interval(sources: &[Sp3], target: Option<f64>) -> Result<f64> {
    let intervals: Vec<f64> = sources
        .iter()
        .enumerate()
        .map(|(idx, source)| {
            effective_epoch_interval_s(source)?.ok_or_else(|| {
                Error::InvalidInput(format!(
                    "merge input {idx} has no usable positive epoch interval"
                ))
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let common = match target {
        Some(t) if t.is_finite() && t > 0.0 => t,
        Some(t) => {
            return Err(Error::InvalidInput(format!(
                "merge target epoch interval must be positive and finite, got {t}"
            )))
        }
        None => intervals.iter().copied().fold(0.0_f64, f64::max),
    };

    // The merge matches and decimates epochs on whole-second J2000 keys, so the
    // common grid must fall on whole seconds for the decimation lattice to be
    // exact. SP3 grids are integer-second; reject a fractional common interval
    // rather than decimate on a mismatched (rounded) lattice.
    if (common - common.round()).abs() > WHOLE_SECOND_EPS_S || common.round() < 1.0 {
        return Err(Error::InvalidInput(format!(
            "merge common epoch interval {common:.6} s must be a positive whole number of seconds"
        )));
    }

    for (idx, interval) in intervals.iter().copied().enumerate() {
        if !divides_evenly(interval, common) {
            return Err(Error::InvalidInput(format!(
                "merge inputs have mismatched epoch intervals: common {common:.6} s is not an integer multiple of input {idx} {interval:.6} s (no exact-subset decimation; positional interpolation is not performed)"
            )));
        }
    }

    Ok(common)
}

/// True when `common` is a positive-integer multiple of `interval` (within the
/// interval tolerance), i.e. `interval`'s grid is a superset of the common grid.
fn divides_evenly(interval: f64, common: f64) -> bool {
    if !(interval.is_finite() && interval > 0.0 && common.is_finite() && common > 0.0) {
        return false;
    }
    let k = (common / interval).round();
    k >= 1.0 && same_interval(k * interval, common)
}

fn effective_epoch_interval_s(source: &Sp3) -> Result<Option<f64>> {
    let secs: Vec<f64> = source
        .epochs
        .iter()
        .filter_map(instant_to_j2000_seconds)
        .collect();
    validate::require_strictly_increasing(secs.iter().copied(), "merge input epochs").map_err(
        |error| Error::InvalidInput(format!("{} must be strictly increasing", error.field())),
    )?;
    let gaps: Vec<f64> = secs.windows(2).map(|w| w[1] - w[0]).collect();

    if gaps.is_empty() {
        let header = source.header.epoch_interval_s;
        return Ok((header.is_finite() && header > 0.0).then_some(header));
    }

    let interval = gaps[0];
    if gaps.iter().all(|g| same_interval(*g, interval)) {
        Ok(Some(interval))
    } else {
        Ok(None)
    }
}

fn same_interval(a: f64, b: f64) -> bool {
    (a - b).abs() <= WHOLE_SECOND_EPS_S
}

/// Indices of the largest subset of `items` whose members are *mutually* within
/// `within`. Exact max-clique over normal source counts; deterministic greedy
/// fallback above [`MAX_EXACT_CLIQUE_NODES`] keeps hostile overlap graphs bounded.
/// Ties resolve to the lowest-indexed subset (precedence).
fn largest_within<T>(items: &[T], within: impl Fn(&T, &T) -> bool) -> Vec<usize> {
    let n = items.len();
    if n <= 1 {
        return (0..n).collect();
    }
    let graph = agreement_graph(items, within);
    if n > MAX_EXACT_CLIQUE_NODES {
        return greedy_largest_clique(&graph);
    }
    let mut best = vec![0];
    let mut current = Vec::new();
    max_clique_search(&graph, &mut current, (0..n).collect(), &mut best);
    best
}

fn largest_within_containing<T>(
    items: &[T],
    required: usize,
    within: impl Fn(&T, &T) -> bool,
) -> Vec<usize> {
    let n = items.len();
    if n == 0 || required >= n {
        return Vec::new();
    }
    if n == 1 {
        return vec![required];
    }

    let graph = agreement_graph(items, within);
    if n > MAX_EXACT_CLIQUE_NODES {
        return greedy_largest_clique_containing(&graph, required);
    }
    let candidates = (0..n)
        .filter(|&idx| idx != required && graph[required][idx])
        .collect();
    let mut best = vec![required];
    let mut current = vec![required];
    max_clique_search(&graph, &mut current, candidates, &mut best);
    best
}

fn agreement_graph<T>(items: &[T], within: impl Fn(&T, &T) -> bool) -> Vec<Vec<bool>> {
    let n = items.len();
    let mut graph = vec![vec![false; n]; n];
    for i in 0..n {
        graph[i][i] = true;
        for j in i + 1..n {
            let agrees = within(&items[i], &items[j]);
            graph[i][j] = agrees;
            graph[j][i] = agrees;
        }
    }
    graph
}

fn greedy_largest_clique(graph: &[Vec<bool>]) -> Vec<usize> {
    let mut best = Vec::new();
    for seed in 0..graph.len() {
        let candidate = greedy_clique_from_seed(graph, seed);
        update_best_clique(&candidate, &mut best);
    }
    best
}

fn greedy_largest_clique_containing(graph: &[Vec<bool>], required: usize) -> Vec<usize> {
    if required >= graph.len() {
        return Vec::new();
    }
    greedy_clique_from_seed(graph, required)
}

fn greedy_clique_from_seed(graph: &[Vec<bool>], seed: usize) -> Vec<usize> {
    let mut clique = vec![seed];
    for (idx, _) in graph.iter().enumerate() {
        if idx == seed {
            continue;
        }
        if clique.iter().all(|&member| graph[member][idx]) {
            clique.push(idx);
        }
    }
    clique.sort_unstable();
    clique
}

fn max_clique_search(
    graph: &[Vec<bool>],
    current: &mut Vec<usize>,
    mut candidates: Vec<usize>,
    best: &mut Vec<usize>,
) {
    candidates.sort_unstable();
    for (pos, &candidate) in candidates.iter().enumerate() {
        let remaining = candidates.len() - pos;
        if current.len() + remaining < best.len() {
            break;
        }

        let next_candidates = candidates[pos + 1..]
            .iter()
            .copied()
            .filter(|&idx| graph[candidate][idx])
            .collect();

        current.push(candidate);
        update_best_clique(current, best);
        max_clique_search(graph, current, next_candidates, best);
        current.pop();
    }
}

fn update_best_clique(current: &[usize], best: &mut Vec<usize>) {
    let mut candidate = current.to_vec();
    candidate.sort_unstable();
    if candidate.len() > best.len()
        || (candidate.len() == best.len() && candidate.as_slice() < best.as_slice())
    {
        *best = candidate;
    }
}

fn combine3(members: &[(usize, [f64; 3])], how: MergeCombine) -> [f64; 3] {
    [0usize, 1, 2].map(|axis| {
        let axis_members: Vec<(usize, f64)> = members.iter().map(|(s, v)| (*s, v[axis])).collect();
        combine_axis(&axis_members, how)
    })
}

fn combine_axis(members: &[(usize, f64)], how: MergeCombine) -> f64 {
    match how {
        MergeCombine::Mean => members.iter().map(|(_, v)| *v).sum::<f64>() / members.len() as f64,
        MergeCombine::Median => {
            let mut vals: Vec<f64> = members.iter().map(|(_, v)| *v).collect();
            median(&mut vals).expect("consensus cluster is non-empty")
        }
        MergeCombine::Precedence => members
            .iter()
            .min_by_key(|(s, _)| *s)
            .map(|(_, v)| *v)
            .expect("consensus cluster is non-empty"),
    }
}

/// Return a copy of `other` with its clocks shifted onto `reference`'s clock
/// datum.
///
/// This applies the per-epoch reference-clock offset from
/// [`clock_reference_offset`]: at each epoch where the offset could be estimated
/// (at least `min_common` common clocked satellites), every clocked satellite's
/// offset has the datum subtracted, so the result's clocks are directly
/// comparable to `reference`'s. Positions are untouched (already comparable).
///
/// Epochs where the offset could not be estimated are left unchanged - they are
/// *not* on `reference`'s datum, so a caller mixing aligned and unaligned epochs
/// should consult [`clock_reference_offset`] to see which epochs were aligned.
/// The returned product interpolates like any other [`Sp3`].
pub fn align_clock_reference(reference: &Sp3, other: &Sp3, min_common: usize) -> Sp3 {
    let offsets: BTreeMap<i64, f64> = clock_reference_offset(reference, other, min_common)
        .into_iter()
        .filter_map(|o| {
            instant_to_j2000_seconds(&o.epoch).map(|sec| (sec.floor() as i64, o.offset_s))
        })
        .collect();

    let mut aligned = other.clone();
    for ei in 0..aligned.epochs.len() {
        let Some(sec) = instant_to_j2000_seconds(&aligned.epochs[ei]) else {
            continue;
        };
        let Some(&off) = offsets.get(&(sec.floor() as i64)) else {
            continue;
        };
        for state in aligned.states[ei].values_mut() {
            if let Some(c) = state.clock_s.as_mut() {
                *c -= off;
            }
        }
        for node in aligned.interp_raw[ei].values_mut() {
            if let Some(us) = node.clock_us.as_mut() {
                *us -= off * 1.0e6;
            }
        }
    }
    aligned
}

#[cfg(test)]
mod tests {
    use super::super::Sp3;
    use super::{
        align_clock_reference, clock_reference_offset, merge, MergeCombine, MergeOptions,
        MergeReport,
    };
    use crate::constants::SECONDS_PER_DAY;
    use crate::id::{GnssSatelliteId, GnssSystem};
    use std::collections::BTreeSet;

    /// One satellite sample in a synthetic SP3 epoch: token, ECEF position
    /// (km), and optional clock (microseconds).
    type SatSample<'a> = (&'a str, [f64; 3], Option<f64>);

    fn gps(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
    }

    // Single-epoch SP3-c from explicit `(satellite, [x,y,z] km, clock us, flag
    // suffix)` records under coordinate system `cs` (5 chars, e.g. `"IGS14"`).
    // `flags` is appended verbatim after the 60-column record body, so a test can
    // place an SP3 flag (e.g. `"              E"` -> the `E` clock-event flag at
    // column 75). A `None` clock writes the SP3 bad-clock sentinel.
    fn sp3_build(records: &[(&str, [f64; 3], Option<f64>, &str)], cs: &str) -> Sp3 {
        let n = records.len();
        let mut sats = String::new();
        for (sat, _, _, _) in records {
            sats.push_str(sat);
        }
        for _ in n..17 {
            sats.push_str("  0");
        }
        let mut body = String::new();
        body.push_str(&format!(
            "#cP2020  6 25  0  0  0.00000000       1 ORBIT {cs} FIT  TST\n"
        ));
        body.push_str("## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n");
        body.push_str(&format!("+   {n:2}   {sats}\n"));
        body.push_str("++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n");
        body.push_str("%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
        body.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
        body.push_str("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n");
        body.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
        body.push_str("%i    0    0    0    0      0      0      0      0         0\n");
        body.push_str("%i    0    0    0    0      0      0      0      0         0\n");
        body.push_str("/* TEST SP3-c FIXTURE\n");
        body.push_str("*  2020  6 25  0  0  0.00000000\n");
        for (sat, p, clk, flags) in records {
            let c = clk.unwrap_or(999_999.999_999);
            body.push_str(&format!(
                "P{sat}{:14.6}{:14.6}{:14.6}{c:14.6}{flags}\n",
                p[0], p[1], p[2]
            ));
        }
        body.push_str("EOF\n");
        Sp3::parse(body.as_bytes()).expect("parse test sp3")
    }

    // The common case: IGS14, no flags.
    fn sp3_records(records: &[(&str, [f64; 3], Option<f64>)]) -> Sp3 {
        let full: Vec<(&str, [f64; 3], Option<f64>, &str)> =
            records.iter().map(|(s, p, c)| (*s, *p, *c, "")).collect();
        sp3_build(&full, "IGS14")
    }

    fn sp3_two_epochs(
        epoch0: &[(&str, [f64; 3], Option<f64>)],
        epoch1: &[(&str, [f64; 3], Option<f64>)],
        interval_s: f64,
        cs: &str,
    ) -> Sp3 {
        let mut sats: Vec<&str> = epoch0
            .iter()
            .chain(epoch1.iter())
            .map(|(sat, _, _)| *sat)
            .collect();
        sats.sort_unstable();
        sats.dedup();
        let n = sats.len();
        let mut sat_field = String::new();
        for sat in &sats {
            sat_field.push_str(sat);
        }
        for _ in n..17 {
            sat_field.push_str("  0");
        }

        let mut body = String::new();
        body.push_str(&format!(
            "#cP2020  6 25  0  0  0.00000000       2 ORBIT {cs} FIT  TST\n"
        ));
        body.push_str(&format!(
            "## 2111 432000.00000000 {interval_s:14.8} 59025 0.0000000000000\n"
        ));
        body.push_str(&format!("+   {n:2}   {sat_field}\n"));
        body.push_str("++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n");
        body.push_str("%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
        body.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
        body.push_str("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n");
        body.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
        body.push_str("%i    0    0    0    0      0      0      0      0         0\n");
        body.push_str("%i    0    0    0    0      0      0      0      0         0\n");
        body.push_str("/* TEST SP3-c FIXTURE\n");
        body.push_str("*  2020  6 25  0  0  0.00000000\n");
        for (sat, p, clk) in epoch0 {
            let c = clk.unwrap_or(999_999.999_999);
            body.push_str(&format!(
                "P{sat}{:14.6}{:14.6}{:14.6}{c:14.6}\n",
                p[0], p[1], p[2]
            ));
        }
        let second_hour = (interval_s as i64) / 3600;
        let second_minute = ((interval_s as i64) % 3600) / 60;
        let second_second = (interval_s as i64) % 60;
        body.push_str(&format!(
            "*  2020  6 25 {second_hour:2} {second_minute:2} {second_second:2}.00000000\n"
        ));
        for (sat, p, clk) in epoch1 {
            let c = clk.unwrap_or(999_999.999_999);
            body.push_str(&format!(
                "P{sat}{:14.6}{:14.6}{:14.6}{c:14.6}\n",
                p[0], p[1], p[2]
            ));
        }
        body.push_str("EOF\n");
        Sp3::parse(body.as_bytes()).expect("parse test sp3")
    }

    // N consecutive epochs spaced `interval_s` apart from 2020-06-25 00:00:00.
    fn sp3_epochs(
        start_offset_s: f64,
        epochs: &[&[SatSample<'_>]],
        interval_s: f64,
        cs: &str,
    ) -> Sp3 {
        let mut sats: Vec<&str> = epochs
            .iter()
            .flat_map(|e| e.iter().map(|(sat, _, _)| *sat))
            .collect();
        sats.sort_unstable();
        sats.dedup();
        let n = sats.len();
        let mut sat_field = String::new();
        for sat in &sats {
            sat_field.push_str(sat);
        }
        for _ in n..17 {
            sat_field.push_str("  0");
        }

        let hms = |t: i64| (t / 3600, (t % 3600) / 60, t % 60);
        let start = start_offset_s as i64;
        let (sh, sm, ss0) = hms(start);

        let mut body = String::new();
        body.push_str(&format!(
            "#cP2020  6 25 {sh:2} {sm:2} {ss0:2}.00000000      {:2} ORBIT {cs} FIT  TST\n",
            epochs.len()
        ));
        // Seconds-of-week and MJD fraction of the first epoch shift with the start.
        let sow = 432_000.0 + start_offset_s;
        let mjd_frac = start_offset_s / SECONDS_PER_DAY;
        body.push_str(&format!(
            "## 2111 {sow:15.8} {interval_s:14.8} 59025 {mjd_frac:.13}\n"
        ));
        body.push_str(&format!("+   {n:2}   {sat_field}\n"));
        body.push_str("++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n");
        body.push_str("%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
        body.push_str("%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n");
        body.push_str("%f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n");
        body.push_str("%f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n");
        body.push_str("%i    0    0    0    0      0      0      0      0         0\n");
        body.push_str("%i    0    0    0    0      0      0      0      0         0\n");
        body.push_str("/* TEST SP3-c FIXTURE\n");
        for (k, recs) in epochs.iter().enumerate() {
            let (hh, mm, ss) = hms(start + (k as i64) * (interval_s as i64));
            body.push_str(&format!("*  2020  6 25 {hh:2} {mm:2} {ss:2}.00000000\n"));
            for (sat, p, clk) in recs.iter() {
                let c = clk.unwrap_or(999_999.999_999);
                body.push_str(&format!(
                    "P{sat}{:14.6}{:14.6}{:14.6}{c:14.6}\n",
                    p[0], p[1], p[2]
                ));
            }
        }
        body.push_str("EOF\n");
        Sp3::parse(body.as_bytes()).expect("parse test sp3")
    }

    #[test]
    fn merge_unions_coverage_when_one_center_misses_a_satellite() {
        // Center A reports G01/G02/G03; center B is missing G03. The merged
        // product must still cover G03 at that epoch (filled from A).
        let a = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
            ("G03", [17000.0, -22000.0, 7000.0], Some(300.0)),
        ]);
        let b = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
        ]);

        let (merged, report) = merge(&[a, b], &MergeOptions::default()).expect("merge");

        let states = merged.states_at(0).expect("epoch 0");
        assert!(
            states.contains_key(&gps(3)),
            "merged output must cover G03 from the center that has it"
        );
        assert_eq!(states.len(), 3, "union is G01/G02/G03");
        // G01 agreed across both centers -> consensus clock is their value.
        let g01 = states[&gps(1)];
        assert!((g01.clock_s.unwrap() - 100.0e-6).abs() < 1.0e-15);
        // G03 had a single source -> carried through, recorded, not quarantined.
        assert!(report.quarantined.is_empty());
        assert_eq!(report.single_source.len(), 1);
        assert_eq!(report.single_source[0].satellite, gps(3));

        // The un-cross-checked share is surfaced: 1 of 3 accepted cells (G03) was
        // single-source, so a clean multi-source agreement RMS is not the whole
        // story. An empty report reports None.
        let frac = report
            .single_source_fraction()
            .expect("accepted cells present");
        assert!(
            (frac - 1.0 / 3.0).abs() < 1.0e-12,
            "single-source fraction {frac}"
        );
        assert_eq!(MergeReport::default().single_source_fraction(), None);
    }

    #[test]
    fn merge_combines_two_of_three_agreeing_sources_and_rejects_the_outlier() {
        // A and B agree on G01; C is 10 m off in X (> the default 0.5 m tolerance).
        let a = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))]);
        let b = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))]);
        let c = sp3_records(&[("G01", [15000.010, -20000.0, 5000.0], Some(100.0))]);

        let (merged, report) = merge(&[a, b, c], &MergeOptions::default()).expect("merge");

        let states = merged.states_at(0).expect("epoch 0");
        let g01 = states[&gps(1)];
        // Consensus is A/B (15000 km == 1.5e7 m); not dragged toward C.
        assert!(
            (g01.position.as_array()[0] - 15_000_000.0).abs() < 1.0e-3,
            "got {}",
            g01.position.as_array()[0]
        );
        // C is source index 2 -> recorded as the rejected position outlier.
        assert_eq!(report.position_outliers.len(), 1);
        assert_eq!(report.position_outliers[0].sources, vec![2]);
        assert!(report.quarantined.is_empty());
    }

    #[test]
    fn merge_consensus_handles_more_than_u32_mask_bits() {
        // Thirty-two centers agree and the 33rd is 10 m off in X. This used to
        // overflow the u32 subset mask before any consensus could be found.
        let sources: Vec<Sp3> = (0..33)
            .map(|idx| {
                let x_km = if idx < 32 { 15000.0 } else { 15000.010 };
                sp3_records(&[("G01", [x_km, -20000.0, 5000.0], Some(100.0))])
            })
            .collect();

        for combine in [MergeCombine::Mean, MergeCombine::Precedence] {
            let opts = MergeOptions {
                combine,
                min_agree: 32,
                ..MergeOptions::default()
            };

            let (merged, report) = merge(&sources, &opts).expect("33-source merge");

            let states = merged.states_at(0).expect("epoch 0");
            let g01 = states[&gps(1)];
            assert!(
                (g01.position.as_array()[0] - 15_000_000.0).abs() < 1.0e-3,
                "{combine:?}: got {}",
                g01.position.as_array()[0]
            );
            assert_eq!(
                report.position_outliers.len(),
                1,
                "{combine:?}: expected one outlier report"
            );
            assert_eq!(report.position_outliers[0].sources, vec![32]);
            assert!(report.quarantined.is_empty(), "{combine:?}");
        }
    }

    #[test]
    fn merge_bounds_large_overlap_clique_search() {
        let sources: Vec<Sp3> = (0..40)
            .map(|idx| {
                let x_km = if idx % 2 == 0 { 15000.0 } else { 15000.010 };
                sp3_records(&[("G01", [x_km, -20000.0, 5000.0], Some(100.0))])
            })
            .collect();
        let opts = MergeOptions {
            min_agree: 20,
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&sources, &opts).expect("bounded large-source merge");

        let states = merged.states_at(0).expect("epoch 0");
        let g01 = states[&gps(1)];
        assert!(
            (g01.position.as_array()[0] - 15_000_000.0).abs() < 1.0e-3,
            "got {}",
            g01.position.as_array()[0]
        );
        assert_eq!(report.position_outliers.len(), 1);
        assert_eq!(
            report.position_outliers[0].sources,
            (1..40).step_by(2).collect::<Vec<_>>()
        );
        assert!(report.quarantined.is_empty());
    }

    #[test]
    fn merge_quarantines_a_satellite_all_centers_disagree_on() {
        // Three sources, mutually beyond tolerance on G01: no 2-of-3 consensus.
        let a = sp3_records(&[("G01", [15000.000, -20000.0, 5000.0], Some(100.0))]);
        let b = sp3_records(&[("G01", [15000.010, -20000.0, 5000.0], Some(100.0))]);
        let c = sp3_records(&[("G01", [15000.020, -20000.0, 5000.0], Some(100.0))]);

        let (merged, report) = merge(&[a, b, c], &MergeOptions::default()).expect("merge");

        assert!(
            merged.states_at(0).expect("epoch 0").is_empty(),
            "no consensus -> G01 omitted, not averaged across disagreeing centers"
        );
        assert_eq!(report.quarantined.len(), 1);
        assert_eq!(report.quarantined[0].satellite, gps(1));
    }

    #[test]
    fn merge_rejects_an_empty_input() {
        assert!(merge(&[], &MergeOptions::default()).is_err());
    }

    #[test]
    fn merge_omits_an_unalignable_secondary_clock() {
        // Only 3 common satellites, but the default clock datum needs 5, so
        // center B's clocks cannot be put on A's datum. They must be dropped
        // rather than emitted raw, and a B-only satellite gets a position but no
        // clock.
        let a = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
            ("G03", [17000.0, -22000.0, 7000.0], Some(300.0)),
        ]);
        let b = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(150.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(250.0)),
            ("G03", [17000.0, -22000.0, 7000.0], Some(350.0)),
            ("G04", [18000.0, -23000.0, 8000.0], Some(450.0)),
        ]);

        let (merged, _) = merge(&[a, b], &MergeOptions::default()).expect("merge");
        let states = merged.states_at(0).expect("epoch 0");

        // G04 is B-only (gap fill): position carried, clock unalignable -> dropped.
        assert!(states.contains_key(&gps(4)));
        assert!(
            states[&gps(4)].clock_s.is_none(),
            "an unalignable secondary clock must be dropped, not emitted raw"
        );
        // G01's clock comes from the reference (source 0), which is on its own datum.
        let g01_clock = states[&gps(1)]
            .clock_s
            .expect("G01 carries the reference clock");
        assert!((g01_clock - 100.0e-6).abs() < 1.0e-12, "got {g01_clock}");
    }

    #[test]
    fn merge_rejects_mismatched_coordinate_systems() {
        let a = sp3_build(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0), "")],
            "IGS14",
        );
        let b = sp3_build(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0), "")],
            "IGS20",
        );

        assert!(merge(&[a, b], &MergeOptions::default()).is_err());
    }

    #[test]
    fn merge_rejects_different_igs_frame_labels_without_a_transform() {
        let a = sp3_build(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0), "")],
            "IGS20",
        );
        let b = sp3_build(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0), "")],
            "IGc20",
        );

        let err = merge(&[a, b], &MergeOptions::default()).expect_err("frame mismatch");
        assert!(
            err.to_string().contains("mismatched coordinate systems"),
            "{err}"
        );
    }

    #[test]
    fn merge_decimates_finer_interval_onto_coarse_common_grid() {
        // 15-min (900 s) center A and 5-min (300 s) center B over the same span.
        // The merge must decimate B onto the 900 s grid (exact subset of B's :00
        // and :15 epochs; the :05/:10 epochs are dropped, not interpolated),
        // output at 900 s. Under precedence, A (source 0) wins G01's whole arc and
        // B's distinct values must never be substituted mid-arc.
        let a = sp3_two_epochs(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))],
            &[("G01", [15003.0, -20003.0, 5003.0], Some(103.0))],
            900.0,
            "IGS14",
        );
        let b = sp3_epochs(
            0.0,
            &[
                &[("G01", [26000.0, -20000.0, 5000.0], Some(200.0))],
                &[("G01", [26001.0, -20001.0, 5001.0], Some(201.0))],
                &[("G01", [26002.0, -20002.0, 5002.0], Some(202.0))],
                &[("G01", [26003.0, -20003.0, 5003.0], Some(203.0))],
            ],
            300.0,
            "IGS14",
        );

        let opts = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 1,
            ..MergeOptions::default()
        };
        let (merged, _report) =
            merge(&[a, b], &opts).expect("mixed-interval merge decimates onto the coarse grid");

        assert_eq!(
            merged.header.epoch_interval_s, 900.0,
            "output is on the coarse (900 s) common grid"
        );
        assert_eq!(
            merged.epochs.len(),
            2,
            "only the two aligned epochs (:00, :15), not B's four"
        );
        // Per-arc precedence intact across the decimated grid: A (source 0) wins
        // both epochs; B's :00/:15 values (26000xxx km) are never substituted.
        for idx in 0..2 {
            let g01 = merged.states_at(idx).expect("epoch")[&gps(1)];
            assert!(
                (g01.position.as_array()[0] - 15_000_000.0 - (idx as f64) * 3000.0).abs() < 1.0,
                "epoch {idx}: expected A's value, got {}",
                g01.position.as_array()[0]
            );
        }
        assert!(merged.states_at(0).expect("epoch 0").contains_key(&gps(1)));
        assert!(merged.states_at(1).expect("epoch 1").contains_key(&gps(1)));
    }

    #[test]
    fn merge_decimates_with_explicit_coarser_target_interval() {
        // Two 5-min inputs, explicit 900 s target: both decimate to the 15-min grid.
        let recs = |x: f64| vec![("G01", [x, -20000.0, 5000.0], Some(100.0))];
        let make = || {
            sp3_epochs(
                0.0,
                &[
                    &recs(15000.0),
                    &recs(15001.0),
                    &recs(15002.0),
                    &recs(15003.0),
                ],
                300.0,
                "IGS14",
            )
        };
        let opts = MergeOptions {
            min_agree: 1,
            target_epoch_interval_s: Some(900.0),
            ..MergeOptions::default()
        };
        let (merged, _) = merge(&[make(), make()], &opts).expect("explicit coarse target");
        assert_eq!(merged.header.epoch_interval_s, 900.0);
        assert_eq!(
            merged.epochs.len(),
            2,
            "decimated 5-min inputs to the 900 s target"
        );
    }

    #[test]
    fn merge_rejects_non_divisible_epoch_intervals() {
        // 900 s and 400 s: 900 is not an integer multiple of 400, so no exact
        // subset of the 400 s grid lands on the 900 s grid -> still rejected
        // (positional interpolation is never performed).
        let a = sp3_two_epochs(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))],
            &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
            900.0,
            "IGS14",
        );
        let b = sp3_two_epochs(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))],
            &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
            400.0,
            "IGS14",
        );

        let err = merge(&[a, b], &MergeOptions::default()).expect_err("non-divisible intervals");
        assert!(
            err.to_string().contains("mismatched epoch intervals"),
            "{err}"
        );
    }

    #[test]
    fn merge_rejects_a_non_whole_second_common_interval() {
        // The decimation lattice is whole-second J2000 keys, so a fractional
        // common interval must be rejected rather than silently rounded.
        let mk = || {
            sp3_two_epochs(
                &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))],
                &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
                900.0,
                "IGS14",
            )
        };
        let opts = MergeOptions {
            target_epoch_interval_s: Some(450.5),
            ..MergeOptions::default()
        };
        let err = merge(&[mk(), mk()], &opts).expect_err("fractional target");
        assert!(err.to_string().contains("whole number of seconds"), "{err}");
    }

    #[test]
    fn merge_header_first_epoch_describes_the_decimated_grid_start() {
        // Source A starts at 00:00, source B at 00:15 (both 15-min). The merged
        // grid's first epoch is the first COMMON epoch, 00:15, so the output
        // header's seconds-of-week / MJD fraction must describe 00:15 (source B),
        // not source A's earlier 00:00 start.
        let a = sp3_epochs(
            0.0,
            &[
                &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))],
                &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
                &[("G01", [15002.0, -20002.0, 5002.0], Some(102.0))],
            ],
            900.0,
            "IGS14",
        );
        let b = sp3_epochs(
            900.0,
            &[
                &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
                &[("G01", [15002.0, -20002.0, 5002.0], Some(102.0))],
                &[("G01", [15003.0, -20003.0, 5003.0], Some(103.0))],
            ],
            900.0,
            "IGS14",
        );

        let opts = MergeOptions {
            min_agree: 1,
            ..MergeOptions::default()
        };
        let (merged, _) = merge(&[a, b], &opts).expect("merge");

        assert_eq!(merged.epochs.len(), 2, "common epochs are 00:15 and 00:30");
        assert!(
            (merged.header.seconds_of_week - 346_500.0).abs() < 1.0e-6,
            "header sow must describe the merged first epoch 00:15 (346500 s), got {}",
            merged.header.seconds_of_week
        );
        assert!(
            (merged.header.mjd_fraction - 900.0 / 86_400.0).abs() < 1.0e-9,
            "header MJD fraction must describe 00:15, got {}",
            merged.header.mjd_fraction
        );
    }

    #[test]
    fn merge_writer_recomputes_header_when_common_grid_starts_after_all_inputs() {
        // A starts on the 15-minute grid at 00:00. B starts on a 7.5-minute grid
        // at 00:07:30. Their coarsened common grid starts at 00:15, which is not
        // the first epoch of either input, so the merged `##` header must be
        // derived from the output epoch rather than cloned from a source header.
        let a = sp3_epochs(
            0.0,
            &[
                &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))],
                &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
                &[("G01", [15002.0, -20002.0, 5002.0], Some(102.0))],
            ],
            900.0,
            "IGS14",
        );
        let b = sp3_epochs(
            450.0,
            &[
                &[("G01", [15010.0, -20010.0, 5010.0], Some(110.0))],
                &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
                &[("G01", [15011.0, -20011.0, 5011.0], Some(111.0))],
                &[("G01", [15002.0, -20002.0, 5002.0], Some(102.0))],
            ],
            450.0,
            "IGS14",
        );

        let opts = MergeOptions {
            min_agree: 1,
            ..MergeOptions::default()
        };
        let (merged, _) = merge(&[a, b], &opts).expect("mixed-cadence merge");

        assert_eq!(merged.epochs.len(), 2, "common epochs are 00:15 and 00:30");
        let text = merged.to_sp3_string();
        let header = text
            .lines()
            .find(|line| line.starts_with("## "))
            .expect("written ## header");
        let first_epoch = text
            .lines()
            .find(|line| line.starts_with("*  "))
            .expect("written first epoch");

        assert_eq!(first_epoch, "*  2020  6 25  0 15  0.00000000");
        assert_eq!(
            header,
            "## 2111 346500.00000000   900.00000000 59025 0.0104166666667"
        );
    }

    #[test]
    fn precedence_merge_never_switches_source_within_one_satellite_arc() {
        let a = sp3_two_epochs(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))],
            &[],
            900.0,
            "IGS14",
        );
        let b = sp3_two_epochs(
            &[("G01", [15000.001, -20000.0, 5000.0], Some(100.0))],
            &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
            900.0,
            "IGS14",
        );
        let opts = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 1,
            ..MergeOptions::default()
        };

        let (merged, _report) = merge(&[a, b], &opts).expect("merge");
        let epoch0 = merged.states_at(0).expect("epoch 0");
        let epoch1 = merged.states_at(1).expect("epoch 1");

        assert!(epoch0.contains_key(&gps(1)));
        assert!(
            !epoch1.contains_key(&gps(1)),
            "G01 must not switch from source 0 at epoch 0 to source 1 at epoch 1"
        );
        assert_eq!(merged.header.epoch_interval_s, 900.0);
    }

    #[test]
    fn merge_filters_requested_constellations_and_header_satellites() {
        let a = sp3_two_epochs(
            &[
                ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
                ("E01", [21000.0, -1000.0, 13000.0], Some(120.0)),
            ],
            &[
                ("G01", [15001.0, -20001.0, 5001.0], Some(101.0)),
                ("E01", [21001.0, -1001.0, 13001.0], Some(121.0)),
            ],
            900.0,
            "IGS14",
        );
        let systems = BTreeSet::from([GnssSystem::Gps]);
        let opts = MergeOptions {
            systems: Some(systems),
            ..MergeOptions::default()
        };

        let (merged, _report) = merge(&[a], &opts).expect("merge");

        assert_eq!(merged.header.satellites, vec![gps(1)]);
        for idx in 0..merged.epochs.len() {
            let states = merged.states_at(idx).expect("epoch");
            assert_eq!(states.keys().copied().collect::<Vec<_>>(), vec![gps(1)]);
        }
    }

    #[test]
    fn merge_preserves_a_clock_event_flag() {
        // Source A carries an `E` clock-event flag on G01 (column 75); the merged
        // product must keep it so the interpolator still splits the clock arc.
        let a = sp3_build(
            &[(
                "G01",
                [15000.0, -20000.0, 5000.0],
                Some(100.0),
                "              E",
            )],
            "IGS14",
        );
        let b = sp3_build(
            &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0), "")],
            "IGS14",
        );

        let (merged, _) = merge(&[a, b], &MergeOptions::default()).expect("merge");
        let g01 = merged.states_at(0).expect("epoch 0")[&gps(1)];

        assert!(
            g01.flags.clock_event,
            "merged cell must preserve a contributing source's clock-event flag"
        );
    }

    #[test]
    fn merge_reports_effective_epoch_interval_from_actual_epochs() {
        // The header DECLARES a 300 s interval, but the two epochs are 15 min
        // (900 s) apart. The synthetic merged header must report the spacing of
        // the actual merged epochs, not inherit the stale declared value.
        let body = "#cP2020  6 25  0  0  0.00000000       2 ORBIT IGS14 FIT  TST\n\
            ## 2111 432000.00000000   300.00000000 59025 0.0000000000000\n\
            +    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
            ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
            %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
            %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
            %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
            %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
            %i    0    0    0    0      0      0      0      0         0\n\
            %i    0    0    0    0      0      0      0      0         0\n\
            /* TEST SP3-c FIXTURE\n\
            *  2020  6 25  0  0  0.00000000\n\
            PG01  15000.000000 -20000.000000   5000.000000    100.000000\n\
            *  2020  6 25  0 15  0.00000000\n\
            PG01  15001.000000 -20001.000000   5001.000000    101.000000\n\
            EOF\n";
        let a = Sp3::parse(body.as_bytes()).expect("parse test sp3");

        let (merged, _) = merge(&[a], &MergeOptions::default()).expect("merge");

        assert!(
            (merged.header.epoch_interval_s - 900.0).abs() < 1.0e-6,
            "got {}",
            merged.header.epoch_interval_s
        );
    }

    #[test]
    fn merge_rejects_unsorted_input_epochs_before_cadence_inference() {
        let body = "#cP2020  6 25  0  0  0.00000000       2 ORBIT IGS14 FIT  TST\n\
            ## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n\
            +    1   G01  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
            ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
            %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
            %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
            %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
            %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
            %i    0    0    0    0      0      0      0      0         0\n\
            %i    0    0    0    0      0      0      0      0         0\n\
            /* TEST SP3-c FIXTURE\n\
            *  2020  6 25  0 15  0.00000000\n\
            PG01  15001.000000 -20001.000000   5001.000000    101.000000\n\
            *  2020  6 25  0  0  0.00000000\n\
            PG01  15000.000000 -20000.000000   5000.000000    100.000000\n\
            EOF\n";
        let source = Sp3::parse(body.as_bytes()).expect("parse unsorted test sp3");

        let err = merge(&[source], &MergeOptions::default()).expect_err("unsorted epochs");

        assert!(
            err.to_string()
                .contains("merge input epochs must be strictly increasing"),
            "{err}"
        );
    }

    #[test]
    fn align_clock_reference_puts_other_on_the_reference_datum() {
        // `other`'s clocks all run +50 us ahead; after alignment they should sit
        // on `reference`'s datum (G01: 150 us - 50 us = 100 us = 1e-4 s).
        let reference = sp3([100.0, 200.0, 300.0]);
        let other = sp3([150.0, 250.0, 350.0]);

        let aligned = align_clock_reference(&reference, &other, 3);

        let g01 = aligned.states_at(0).expect("epoch 0")[&gps(1)];
        assert!(
            (g01.clock_s.unwrap() - 100.0e-6).abs() < 1.0e-15,
            "got {}",
            g01.clock_s.unwrap()
        );
        // Positions are untouched by clock alignment.
        let original = other.states_at(0).expect("epoch 0")[&gps(1)];
        assert_eq!(g01.position.as_array(), original.position.as_array());
    }

    // Minimal single-epoch SP3-c with three satellites; each `clocks_us` entry is
    // that satellite's clock in microseconds (positions are arbitrary but non-zero
    // so they parse as valid records).
    fn sp3(clocks_us: [f64; 3]) -> Sp3 {
        let body = format!(
            "#cP2020  6 25  0  0  0.00000000       1 ORBIT IGS14 FIT  TST\n\
             ## 2111 432000.00000000   900.00000000 59025 0.0000000000000\n\
             +    3   G01G02G03  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
             ++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0\n\
             %c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
             %c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc\n\
             %f  1.2500000  1.025000000  0.00000000000  0.000000000000000\n\
             %f  0.0000000  0.000000000  0.00000000000  0.000000000000000\n\
             %i    0    0    0    0      0      0      0      0         0\n\
             %i    0    0    0    0      0      0      0      0         0\n\
             /* TEST SP3-c FIXTURE\n\
             *  2020  6 25  0  0  0.00000000\n\
             PG01  15000.000000 -20000.000000   5000.000000 {:13.6}\n\
             PG02  -1234.567890   2345.678901  -3456.789012 {:13.6}\n\
             PG03   8000.000000  12000.000000 -19000.000000 {:13.6}\n\
             EOF\n",
            clocks_us[0], clocks_us[1], clocks_us[2]
        );
        Sp3::parse(body.as_bytes()).expect("parse test sp3")
    }

    #[test]
    fn recovers_a_uniform_datum_shift() {
        // every `other` clock is +50 us (= 5e-5 s) from `reference`.
        let reference = sp3([100.0, 200.0, 300.0]);
        let other = sp3([150.0, 250.0, 350.0]);

        let offsets = clock_reference_offset(&reference, &other, 3);

        assert_eq!(offsets.len(), 1);
        assert_eq!(offsets[0].satellites, 3);
        assert!(
            (offsets[0].offset_s - 5.0e-5).abs() < 1.0e-12,
            "got {}",
            offsets[0].offset_s
        );
    }

    #[test]
    fn median_rejects_a_single_outlier_clock() {
        // Two satellites agree (+50 us); one is a wild outlier (+9000 us). The
        // median over the three tracks the consensus instead of being dragged out.
        let reference = sp3([100.0, 200.0, 300.0]);
        let other = sp3([150.0, 250.0, 9_300.0]);

        let offsets = clock_reference_offset(&reference, &other, 3);

        assert_eq!(offsets.len(), 1);
        assert!(
            (offsets[0].offset_s - 5.0e-5).abs() < 1.0e-12,
            "got {}",
            offsets[0].offset_s
        );
    }

    #[test]
    fn omits_epochs_below_min_common() {
        // Three common clocked satellites, but require four: the fragile estimate
        // is omitted rather than reported.
        let reference = sp3([100.0, 200.0, 300.0]);
        let other = sp3([150.0, 250.0, 350.0]);

        assert!(clock_reference_offset(&reference, &other, 4).is_empty());
    }

    #[test]
    fn merge_agreement_metric_reports_known_position_dispersion() {
        // Three centers place G01 on a line, 0 / +3 m / +6 m in X, all within a
        // wide consensus tolerance. The mean combine writes +3 m, so the member
        // distances from the combined value are {3, 0, 3} m:
        //   RMS = sqrt((9 + 0 + 9) / 3) = sqrt(6) m,  max = 3 m.
        let a = sp3_records(&[("G01", [15000.000, -20000.0, 5000.0], Some(100.0))]);
        let b = sp3_records(&[("G01", [15000.003, -20000.0, 5000.0], Some(100.0))]);
        let c = sp3_records(&[("G01", [15000.006, -20000.0, 5000.0], Some(100.0))]);
        let opts = MergeOptions {
            position_tolerance_m: 10.0,
            min_agree: 3,
            combine: MergeCombine::Mean,
            ..MergeOptions::default()
        };

        let (_merged, report) = merge(&[a, b, c], &opts).expect("merge");

        assert_eq!(report.agreement.len(), 1, "one accepted cell");
        let m = report.agreement[0];
        assert_eq!(m.satellite, gps(1));
        assert_eq!(m.position_members, 3);
        assert!(
            (m.position_rms_m - 6.0_f64.sqrt()).abs() < 1.0e-6,
            "got rms {}",
            m.position_rms_m
        );
        assert!(
            (m.position_max_m - 3.0).abs() < 1.0e-6,
            "got max {}",
            m.position_max_m
        );

        // The pooled summaries over the single cell reproduce the cell values.
        assert!((report.position_agreement_rms_m().unwrap() - 6.0_f64.sqrt()).abs() < 1.0e-6);
        assert!((report.position_agreement_max_m().unwrap() - 3.0).abs() < 1.0e-6);

        // Per-epoch aggregate: one epoch, one multi-source satellite.
        let per_epoch = report.per_epoch_agreement();
        assert_eq!(per_epoch.len(), 1);
        assert_eq!(per_epoch[0].satellites, 1);
        assert!((per_epoch[0].position_rms_m - 6.0_f64.sqrt()).abs() < 1.0e-6);
        assert!((per_epoch[0].position_max_m - 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn merge_agreement_metric_reports_known_clock_dispersion() {
        // Same positions across A/B/C (zero position spread); the three centers
        // share a clock datum (G01/G02 identical) so the per-epoch datum offset is
        // zero and G03's clocks stay as authored: 300 / 330 / 270 us. The mean
        // combine writes 300 us, so the deviations are {0, +30, -30} us:
        //   RMS = sqrt((0 + 30^2 + 30^2)/3) us = sqrt(600) us,  max = 30 us.
        let a = sp3([100.0, 200.0, 300.0]);
        let b = sp3([100.0, 200.0, 330.0]);
        let c = sp3([100.0, 200.0, 270.0]);
        let opts = MergeOptions {
            clock_min_common: 1,
            clock_tolerance_s: 1.0e-3,
            min_agree: 3,
            combine: MergeCombine::Mean,
            ..MergeOptions::default()
        };

        let (_merged, report) = merge(&[a, b, c], &opts).expect("merge");

        let g03 = report
            .agreement
            .iter()
            .find(|m| m.satellite == gps(3))
            .expect("G03 agreement metric");
        assert_eq!(g03.clock_members, 3);
        let expected_rms_s = 600.0_f64.sqrt() * 1.0e-6;
        assert!(
            (g03.clock_rms_s.unwrap() - expected_rms_s).abs() < 1.0e-15,
            "got clock rms {:?}",
            g03.clock_rms_s
        );
        assert!(
            (g03.clock_max_s.unwrap() - 30.0e-6).abs() < 1.0e-15,
            "got clock max {:?}",
            g03.clock_max_s
        );
        // G01/G02 agree exactly -> zero clock dispersion.
        for prn in [1u8, 2] {
            let m = report
                .agreement
                .iter()
                .find(|m| m.satellite == gps(prn))
                .expect("metric");
            assert!(m.clock_rms_s.unwrap().abs() < 1.0e-18, "prn {prn}");
            // Positions identical across centers -> zero position dispersion too.
            assert!(m.position_rms_m.abs() < 1.0e-9, "prn {prn}");
        }

        // The clock pooled summary is the RMS over the three multi-source cells
        // (G01=0, G02=0, G03), each with 3 members:
        //   sqrt((0 + 0 + 3*expected^2) / 9) = expected / sqrt(3).
        let pooled = report.clock_agreement_rms_s().expect("clock pool");
        assert!(
            (pooled - expected_rms_s / 3.0_f64.sqrt()).abs() < 1.0e-15,
            "got pooled {pooled}"
        );
        assert!((report.clock_agreement_max_s().unwrap() - 30.0e-6).abs() < 1.0e-15);
    }

    // Real-data oracle: combine published individual analysis-center final
    // products (COD/GFZ/JPL, 2026-04-30, GPS week 2416 DOY 120) and compare to the
    // published IGS official combined for the same day. The IGS combination is a
    // specific weighted algorithm, so the crate's mean combine is not a bit-match;
    // the gate is agreement at the inter-center spread level (cm-level bound), gated
    // at RMS < 2 cm and max < 5 cm (observed RMS ~0.7 cm, max ~1.6 cm over 88 cells).
    //
    // Fixture provenance: the COD/GFZ/JPL `_trim.SP3` files are the final precise
    // orbit products of CODE (AIUB Bern), GFZ Potsdam, and JPL, all frame IGc20 /
    // time system GPS (ESA/GRG excluded for IGS20 frame labelling). From the Wuhan
    // University IGS mirror `ftp://igs.gnsswhu.cn/pub/gps/products/2416/`, full-day
    // `.gz`: COD0OPSFIN_20261200000_01D_05M_ORB.SP3.gz (634569 B, sha256
    // 90393acaed691cd4d19cd4ade7153873eb41ef38585df177d9d540eac6316112);
    // GFZ0OPSFIN…05M_ORB.SP3.gz (647028 B, sha256
    // a51a04ab283a981ddec20ae77d575cd05f4f8249202e0ee4f73e7243b7817e88);
    // JPL0OPSFIN…05M_ORB.SP3.gz (482973 B, sha256
    // 3a39ccb2d097eddb139047532b2b93c5d538abc39255fc779278ac64f10cd185). Each trim
    // keeps the verbatim header and only the 11 epochs 09:45..12:15 landing on the
    // combined's 900 s grid plus the 8-sat subset common to all three centers and
    // the combined (G02,G03,G04,G05,G09,G17,G25,G31); velocity/correlation records
    // dropped, no values altered. Trim sha256: COD…_trim.SP3 (7227 B)
    // f3ad3f637134651d086815345f3e5f531a9dbacb6f739b7dddf664e0ab3a1795;
    // GFZ…_trim.SP3 (9805 B)
    // 9e50edc53ac42791923fd71c39b49a97bf516084f1d2b1dcb260685d2a8f11cc;
    // JPL…_trim.SP3 (8210 B)
    // 9ac5aafdabed38679892f57b42864cc3716d997400280f29ee8049a37057adf4. The oracle
    // IGS0OPSFIN combined product provenance is in `sp3/tests.rs`.
    #[cfg(sidereon_repo_tests)]
    #[test]
    fn merge_agrees_with_published_igs_combined_within_cm() {
        fn load(name: &str) -> Sp3 {
            let path = format!("{}/tests/fixtures/sp3/{}", env!("CARGO_MANIFEST_DIR"), name);
            let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
            Sp3::parse(&bytes).unwrap_or_else(|e| panic!("parse {name}: {e}"))
        }

        let cod = load("COD0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3");
        let gfz = load("GFZ0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3");
        let jpl = load("JPL0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3");
        let igs = load("IGS0OPSFIN_20261200945_02H30M_15M_ORB.SP3");

        let (merged, report) =
            merge(&[cod, gfz, jpl], &MergeOptions::default()).expect("multi-center merge");

        // All three centers agree at the 0.5 m position tolerance: nothing
        // quarantined, every cell a 3-source consensus.
        assert!(
            report.quarantined.is_empty(),
            "centers should agree: {:?}",
            report.quarantined
        );
        // A clean 3-source consensus everywhere: no gap-fills, no rejected
        // outliers, and every accepted cell backed by all three centers.
        assert!(
            report.single_source.is_empty(),
            "{:?}",
            report.single_source
        );
        assert!(
            report.position_outliers.is_empty(),
            "{:?}",
            report.position_outliers
        );
        assert!(
            report.agreement.iter().all(|a| a.position_members == 3),
            "every agreement cell should be a 3-source consensus"
        );

        let mut igs_idx: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
        for (i, ep) in igs.epochs.iter().enumerate() {
            if let Some(s) = super::instant_to_j2000_seconds(ep) {
                igs_idx.insert(s.floor() as i64, i);
            }
        }

        let mut sumsq = 0.0_f64;
        let mut max = 0.0_f64;
        let mut n = 0usize;
        for (mi, ep) in merged.epochs.iter().enumerate() {
            let key = super::instant_to_j2000_seconds(ep)
                .expect("merged epoch key")
                .floor() as i64;
            let ii = *igs_idx.get(&key).expect("IGS combined covers merged epoch");
            let merged_states = merged.states_at(mi).expect("merged states");
            let igs_states = igs.states_at(ii).expect("IGS states");
            for (sat, mst) in merged_states.iter() {
                let ist = igs_states
                    .get(sat)
                    .unwrap_or_else(|| panic!("merged sat {sat} missing from IGS combined"));
                let d = super::dist3(&mst.position.as_array(), &ist.position.as_array());
                sumsq += d * d;
                max = max.max(d);
                n += 1;
            }
        }

        // Exact coverage: 8 satellites x 11 epochs, every merged cell present in
        // the IGS combined (proves same epochs/sats, not a lucky subset).
        assert_eq!(n, 88, "expected exactly 88 compared cells, got {n}");
        let rms = (sumsq / n as f64).sqrt();
        // Observed on this day: RMS ~0.7 cm, max ~1.6 cm. Gate at a cm-level bound.
        assert!(
            rms < 0.02,
            "combine-vs-IGS RMS {:.4} m ({} cells) exceeds the 2 cm gate",
            rms,
            n
        );
        assert!(
            max < 0.05,
            "combine-vs-IGS max {max:.4} m exceeds the 5 cm gate"
        );

        // The internal inter-center agreement metric is also cm-level.
        let dispersion = report
            .position_agreement_rms_m()
            .expect("multi-source cells present");
        assert!(
            dispersion < 0.05,
            "inter-center position dispersion {dispersion:.4} m"
        );
    }
}
