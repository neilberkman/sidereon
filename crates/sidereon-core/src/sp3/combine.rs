#![warn(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

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
//! Orbit positions are directly comparable only when the SP3 coordinate-system
//! labels match, or when the caller explicitly opts into an audited label
//! assertion or terrestrial Helmert reconciliation.

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::math::vec3;
use crate::astro::time::civil::{
    civil_from_julian_day_number, fractional_day_of_year_from_instant, is_leap_year,
    julian_date_from_instant,
};
use crate::astro::time::model::Instant;

use super::grid::{
    gcd, interval_seconds, interval_ticks, product_grid, product_ticks, tick_seconds,
    TICKS_PER_SECOND,
};
use super::interp::{precise_node_j2000_seconds_from_instant, Sp3InterpolationOptions};
use super::{
    RawNode, Sp3, Sp3ClockRecord, Sp3DataType, Sp3Flags, Sp3Header, Sp3State, TerminalRecordState,
};
use crate::constants::{DAYS_PER_JULIAN_YEAR, GPS_EPOCH_TO_J2000_S, KM_TO_M, SECONDS_PER_DAY};
use crate::frame::{ItrfPositionM, ItrfVelocityMS};
use crate::frame_catalog::{
    self, HelmertParameters, HelmertRates, TerrestrialFrame, TerrestrialPositionM,
    TerrestrialVelocityMPerYear,
};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::sp3::continuity::{
    check_continuity, ContinuityDefect, ContinuityOptions, ContinuityReport, EpochWindow,
    InterpolationNodes, WindowContinuityDecision, WindowContinuityVerdict,
};
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
/// Epochs are matched exactly, on the 10-nanosecond tick axis an SP3 epoch
/// record states (`super::grid`): two epochs are matched only when they are the
/// same instant, so distinct epochs never collapse onto one, however close.
/// Non-finite clock differences are skipped. Epochs present in only one
/// product, epochs no SP3 record states exactly, and epochs below `min_common`
/// are omitted from the result.
pub fn clock_reference_offset(
    reference: &Sp3,
    other: &Sp3,
    min_common: usize,
) -> Vec<ClockReferenceOffset> {
    clock_offsets_by_tick(
        reference,
        &product_ticks(reference),
        other,
        &product_ticks(other),
        min_common,
    )
    .into_iter()
    .map(|(_, offset)| offset)
    .collect()
}

/// [`clock_reference_offset`] keyed by the tick of each matched epoch, given
/// both products' epochs on the tick axis.
fn clock_offsets_by_tick(
    reference: &Sp3,
    reference_ticks: &[Option<i128>],
    other: &Sp3,
    other_ticks: &[Option<i128>],
    min_common: usize,
) -> Vec<(i128, ClockReferenceOffset)> {
    let other_index: BTreeMap<i128, usize> = other_ticks
        .iter()
        .enumerate()
        .filter_map(|(index, tick)| tick.map(|tick| (tick, index)))
        .collect();

    let mut offsets = Vec::new();

    for (ref_idx, epoch) in reference.epochs.iter().enumerate() {
        let Some(tick) = reference_ticks.get(ref_idx).copied().flatten() else {
            continue;
        };
        let Some(&other_idx) = other_index.get(&tick) else {
            continue;
        };

        let get_clock = |sp3: &Sp3, ei: usize, sat: &GnssSatelliteId| -> Option<f64> {
            sp3.states_at(ei)
                .ok()
                .and_then(|m| m.get(sat))
                .and_then(|s| s.clock_s)
                .or_else(|| {
                    sp3.clock_records_at(ei)
                        .ok()
                        .and_then(|m| m.get(sat))
                        .map(|r| r.clock_s)
                })
        };

        let mut ref_sats = Vec::new();
        if let Ok(states) = reference.states_at(ref_idx) {
            ref_sats.extend(states.keys().copied());
        }
        if let Ok(clocks) = reference.clock_records_at(ref_idx) {
            ref_sats.extend(clocks.keys().copied());
        }
        ref_sats.sort_unstable();
        ref_sats.dedup();

        let mut diffs: Vec<f64> = Vec::new();
        for sat in ref_sats {
            if let (Some(ref_clock), Some(other_clock)) = (
                get_clock(reference, ref_idx, &sat),
                get_clock(other, other_idx, &sat),
            ) {
                let diff = other_clock - ref_clock;
                // SP3 should not carry NaN/inf clocks, but the parser can
                // accept them; merge infrastructure must not panic on data.
                if diff.is_finite() {
                    diffs.push(diff);
                }
            }
        }

        if diffs.len() >= min_common.max(1) {
            if let Some(offset_s) = median(&mut diffs) {
                offsets.push((
                    tick,
                    ClockReferenceOffset {
                        epoch: *epoch,
                        offset_s,
                        satellites: diffs.len(),
                    },
                ));
            }
        }
    }

    offsets
}

fn median(values: &mut [f64]) -> Option<f64> {
    // Inputs are pre-filtered to finite values; total_cmp never panics regardless.
    crate::astro::math::robust::median_sorting_in_place(values)
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

/// Scope used by [`MergeCombine::Precedence`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergePrecedenceScope {
    /// Select the earliest-listed source that actually carries each individual
    /// `(epoch, satellite)` cell. This maximizes coverage and is the default.
    Cell,
    /// Select one earliest-listed source for the whole satellite arc. Missing
    /// cells in that source remain holes even when a later source has them;
    /// each such withheld position is listed in [`MergeReport::arc_withheld`],
    /// and an epoch left with no cell at all is omitted from the product and
    /// listed in [`MergeReport::omitted_epochs`].
    SatelliteArc,
}

/// Optional consensus guard for precedence selection.
///
/// With this guard disabled, precedence retains its historical behavior: the
/// preferred source wins a contested cell whenever `min_agree` permits it. With
/// it enabled, contested positions and clocks must contain a mutually agreeing
/// cluster of at least `max(min_agree, 2)` sources. The preferred value is kept
/// when it belongs to that cluster; otherwise the earliest-listed member of the
/// deterministic largest cluster replaces it and the rejected source is
/// recorded in the merge report.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct OutlierRejectOptions {
    /// Maximum 3D position separation inside the accepted cluster, meters.
    pub position_tolerance_m: f64,
    /// Maximum aligned-clock separation inside the accepted cluster, seconds.
    pub clock_tolerance_s: f64,
}

impl OutlierRejectOptions {
    /// Build outlier-rejection settings from both required tolerances.
    #[must_use]
    pub const fn new(position_tolerance_m: f64, clock_tolerance_s: f64) -> Self {
        Self {
            position_tolerance_m,
            clock_tolerance_s,
        }
    }
}

/// Options for [`merge`].
///
/// Non-exhaustive: construct with struct-update syntax over
/// [`MergeOptions::default`] (`MergeOptions { min_agree: 3, ..Default::default() }`).
/// This struct gains a field whenever the merge learns a new policy - 0.37.0
/// alone added two - and each addition used to be source-breaking for every
/// consumer holding an exhaustive literal, including all four language
/// bindings. Struct-update construction makes future options non-breaking.
#[non_exhaustive]
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
    /// Whether precedence is selected independently for each cell or fixed for
    /// a whole satellite arc. Ignored for mean and median combination.
    pub precedence_scope: MergePrecedenceScope,
    /// Optional consensus guard for precedence-selected values. `None` preserves
    /// the historical contested-cell behavior.
    pub outlier_reject: Option<OutlierRejectOptions>,
    /// Optional target epoch interval, in seconds; it must be a whole number of
    /// the 10-nanosecond ticks an SP3 interval states. When unset the output
    /// grid is the greatest common divisor of the inputs' grid steps and epoch
    /// offsets, which holds every input epoch. When set, the grid is anchored
    /// at the earliest input epoch, inputs contribute at the grid epochs they
    /// actually carry, and every input epoch off the grid is listed in
    /// [`MergeReport::dropped_input_epochs`]. Values are never interpolated.
    pub target_epoch_interval_s: Option<f64>,
    /// Optional constellation/system filter. When set, only satellites whose
    /// system is in this set are considered for the merged product.
    pub systems: Option<BTreeSet<GnssSystem>>,
    /// Explicit coordinate-label reconciliation rules. Default is disabled, so
    /// mismatched coordinate-system labels are rejected.
    pub frame_reconciliation: Sp3FrameReconciliationOptions,
    /// Record per-epoch provenance as the merge decides. `None` (the default)
    /// records nothing: a full record costs one entry per accepted cell.
    ///
    /// This never changes the merged product - the SP3 output is byte-identical
    /// whether or not provenance is enabled, and a test pins that.
    pub provenance: Option<ProvenanceMode>,
    /// Verify the continuity of the merged product as a post-condition, and
    /// attribute each violation to the contributors on both sides.
    ///
    /// `None` (the default) runs no check. Enabling it never changes the merged
    /// product and never fails the merge: violations are reported on
    /// [`MergeReport::continuity`] and refusing is the caller's decision.
    pub verify_continuity: Option<ContinuityOptions>,
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
            precedence_scope: MergePrecedenceScope::Cell,
            outlier_reject: None,
            target_epoch_interval_s: None,
            systems: None,
            frame_reconciliation: Sp3FrameReconciliationOptions::default(),
            provenance: None,
            verify_continuity: None,
        }
    }
}

/// Explicit opt-in rules for reconciling mismatched SP3 coordinate labels.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Sp3FrameReconciliationOptions {
    /// Caller-asserted label sets that may be treated as physically equivalent
    /// without applying any coordinate transform.
    pub asserted_equivalent_label_sets: Vec<Sp3FrameLabelSet>,
    /// Whether to apply catalog Helmert transforms between known ITRF/IGS
    /// realizations when labels differ and no assertion covers the pair.
    pub helmert: bool,
}

impl Sp3FrameReconciliationOptions {
    /// Construct disabled reconciliation options.
    pub const fn disabled() -> Self {
        Self {
            asserted_equivalent_label_sets: Vec::new(),
            helmert: false,
        }
    }

    /// Construct options that enable catalog Helmert reconciliation.
    pub const fn helmert() -> Self {
        Self {
            asserted_equivalent_label_sets: Vec::new(),
            helmert: true,
        }
    }
}

/// A caller-asserted set of SP3 coordinate labels that may be merged as one
/// physical frame with no coordinate math.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sp3FrameLabelSet {
    /// Exact trimmed labels in this asserted-equivalent set.
    pub labels: BTreeSet<String>,
}

impl Sp3FrameLabelSet {
    /// Construct an asserted-equivalent label set from an iterator of labels.
    pub fn new(labels: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            labels: labels
                .into_iter()
                .map(|label| label.into().trim().to_string())
                .collect(),
        }
    }

    /// Construct a two-label asserted-equivalent set.
    pub fn pair(a: impl Into<String>, b: impl Into<String>) -> Self {
        Self::new([a.into(), b.into()])
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
    /// that disagreed; for `position_outliers` or `clock_outliers`, the sources
    /// rejected from an otherwise-accepted consensus; for `arc_withheld`, the
    /// sources whose positions were withheld.
    pub sources: Vec<usize>,
}

/// Why a source's clock for a cell was not written.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClockOmissionReason {
    /// The source's datum offset to source 0 could not be estimated at this
    /// epoch: no bracketing epochs with at least
    /// [`MergeOptions::clock_min_common`] common clocks. The offset is never
    /// extrapolated, so the clock takes no part in the consensus, whether or not
    /// the cell gets a clock from other sources.
    DatumNotObservable,
    /// Precedence writes a clock only from the preferred source - under
    /// [`MergePrecedenceScope::SatelliteArc`] the satellite's arc owner - and
    /// that source had no clock on the reference datum for the cell, so this
    /// source's clock was not written either.
    PreferredSourceWithoutClock {
        /// The preferred source, when the merge had one for the satellite.
        preferred: Option<usize>,
    },
    /// Clocks on the reference datum disagreed and no agreeing subset met the
    /// consensus rule, so no clock was written. When an outlier guard refused
    /// the cell, the same sources are also in [`MergeReport::clock_outliers`].
    NoConsensus,
}

/// One source's clock for a cell that the merge did not write.
///
/// One entry per source and cell. A clock rejected as an outlier from a clock
/// consensus that was accepted is in [`MergeReport::clock_outliers`] instead.
/// Cells for which no source carried a clock are not listed;
/// [`Sp3::satellite_coverage`] on the merged product shows every clock the
/// product lacks.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockOmission {
    /// The epoch.
    pub epoch: Instant,
    /// The satellite.
    pub satellite: GnssSatelliteId,
    /// Index into the input slice of the source whose clock was not written.
    pub source: usize,
    /// Why it was not written.
    pub reason: ClockOmissionReason,
    /// Whether the merged cell carries a clock from other sources.
    pub cell_has_clock: bool,
}

/// Why an input epoch took no part in a merge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DroppedEpochReason {
    /// The epoch is not on the explicit [`MergeOptions::target_epoch_interval_s`]
    /// grid. The default grid holds every input epoch, so this reason arises
    /// only from an explicit target.
    OffTargetGrid,
    /// No SP3 record states the epoch exactly: its instant is not a whole
    /// number of the 10-nanosecond ticks an epoch record resolves, so it has no
    /// place on the merge's exact epoch axis.
    NotOnTickAxis,
}

/// An input epoch that took no part in a merge, with the reason.
#[derive(Debug, Clone, PartialEq)]
pub struct DroppedInputEpoch {
    /// Index into the input slice.
    pub source: usize,
    /// Index into that source's [`Sp3::epochs`].
    pub epoch_index: usize,
    /// The epoch.
    pub epoch: Instant,
    /// Why it took no part.
    pub reason: DroppedEpochReason,
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
    /// Number of sources in the accepted position consensus (0 when the cell carries
    /// no position).
    pub position_members: usize,
    /// RMS, over the position-consensus members, of the 3D distance from the
    /// combined position, meters. Zero for a single-source cell; `None` when the
    /// cell carries no position.
    pub position_rms_m: Option<f64>,
    /// Largest 3D distance of any position-consensus member from the combined
    /// position, meters; `None` when the cell carries no position.
    pub position_max_m: Option<f64>,
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
    /// those satellites, meters; `None` when no multi-source position consensus
    /// existed at this epoch.
    pub position_rms_m: Option<f64>,
    /// Worst per-cell position dispersion at this epoch, meters; `None` as above.
    pub position_max_m: Option<f64>,
    /// As `position_rms_m` for the clock channel; `None` when no multi-source
    /// clock consensus existed at this epoch.
    pub clock_rms_s: Option<f64>,
    /// Worst per-cell clock dispersion at this epoch, seconds; `None` as above.
    pub clock_max_s: Option<f64>,
}

/// Mechanism used to reconcile one non-reference source's coordinate label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sp3FrameReconciliationMethod {
    /// Caller asserted the labels are physically equivalent; no coordinate math
    /// was applied.
    AssertedEquivalence,
    /// A catalog Helmert transform, or exact identity for the same resolved
    /// realization, reconciled the source to the target label.
    Helmert,
}

/// Audit record for one reconciled SP3 source coordinate label.
#[derive(Debug, Clone, PartialEq)]
pub struct Sp3FrameReconciliation {
    /// Source index in the input slice.
    pub source_index: usize,
    /// Original coordinate-system label on that source.
    pub source_label: String,
    /// Target coordinate-system label, taken from source 0.
    pub target_label: String,
    /// Mechanism selected by the explicit caller options.
    pub method: Sp3FrameReconciliationMethod,
    /// Caller-asserted label set used for [`Sp3FrameReconciliationMethod::AssertedEquivalence`].
    pub asserted_label_set: Option<Vec<String>>,
    /// Resolved source terrestrial realization for Helmert reconciliation.
    pub source_frame: Option<TerrestrialFrame>,
    /// Resolved target terrestrial realization for Helmert reconciliation.
    pub target_frame: Option<TerrestrialFrame>,
    /// Source realization of the published catalog row used for Helmert
    /// reconciliation.
    pub catalog_source_frame: Option<TerrestrialFrame>,
    /// Target realization of the published catalog row used for Helmert
    /// reconciliation.
    pub catalog_target_frame: Option<TerrestrialFrame>,
    /// Whether the published catalog row was applied in reverse.
    pub catalog_inverse: bool,
    /// Published transform reference epoch, when a non-identity catalog entry was
    /// used.
    pub reference_epoch_year: Option<f64>,
    /// Published seven Helmert parameters at the reference epoch, when a
    /// non-identity catalog entry was used.
    pub parameters: Option<HelmertParameters>,
    /// Published parameter rates, when a non-identity catalog entry was used.
    pub rates: Option<HelmertRates>,
    /// Published-table provenance for the catalog entry, when available.
    pub provenance: Option<String>,
    /// Decimal-year span of transformed records, inclusive, when Helmert
    /// reconciliation was applied.
    pub epoch_year_span: Option<[f64; 2]>,
    /// Number of satellite position records covered by the reconciliation.
    pub records_affected: usize,
    /// Whether the resolved source and target realizations were identical, so
    /// the Helmert path left coordinates bit-equal.
    pub identity: bool,
}

/// How much per-epoch provenance [`merge`] records.
///
/// Off entirely unless [`MergeOptions::provenance`] asks for it: a full record
/// is one entry per accepted `(epoch, satellite)` cell, which for a day of
/// 5-minute GNSS data is tens of thousands of entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceMode {
    /// Transitions and per-contributor coverage only. Bounded by the number of
    /// selection changes rather than by the number of cells.
    Summary,
    /// Everything in [`ProvenanceMode::Summary`], plus one
    /// [`CellProvenance`] per accepted cell.
    Full,
}

/// How the merge arrived at the value it wrote for one channel of one cell.
///
/// The distinction between [`CellSelection::Precedence`] and
/// [`CellSelection::Combined`] is load-bearing rather than cosmetic: under
/// [`MergeCombine::Mean`] or [`MergeCombine::Median`] - and `Mean` is the
/// default - the emitted value is a combination of the agreeing members, so
/// "which contributor supplied this cell" has no answer. The honest record there
/// is the member set and the rule that combined them, and this type refuses to
/// pretend otherwise.
#[derive(Debug, Clone, PartialEq)]
pub enum CellSelection {
    /// One source carried the cell; it was carried through as gap fill. Also
    /// recorded in [`MergeReport::single_source`].
    SingleSource {
        /// Index into the input slice.
        source: usize,
    },
    /// Precedence picked one source out of an agreeing set.
    Precedence {
        /// Index into the input slice of the source whose value was written.
        source: usize,
        /// Every source in the accepted consensus, ascending.
        members: Vec<usize>,
    },
    /// The written value is a combination of the members; no single source
    /// supplied it.
    Combined {
        /// The rule that produced the written value.
        rule: MergeCombine,
        /// Every source in the accepted consensus, ascending.
        members: Vec<usize>,
    },
}

impl CellSelection {
    /// The single source whose value was written, when one exists.
    ///
    /// `None` for [`CellSelection::Combined`], where no single contributor
    /// supplied the value.
    pub fn selected_source(&self) -> Option<usize> {
        match self {
            Self::SingleSource { source } | Self::Precedence { source, .. } => Some(*source),
            Self::Combined { .. } => None,
        }
    }

    /// Every source in the accepted consensus.
    pub fn members(&self) -> Vec<usize> {
        match self {
            Self::SingleSource { source } => vec![*source],
            Self::Precedence { members, .. } | Self::Combined { members, .. } => members.clone(),
        }
    }
}

/// Provenance of one accepted `(epoch, satellite)` cell, recorded by the merge
/// as it decided - never reconstructed afterwards.
#[derive(Debug, Clone, PartialEq)]
pub struct CellProvenance {
    /// The epoch.
    pub epoch: Instant,
    /// The satellite.
    pub satellite: GnssSatelliteId,
    /// How the written position was arrived at; `None` when the cell carries no
    /// position (e.g. clock-only record).
    pub position: Option<CellSelection>,
    /// How the written clock was arrived at; `None` when the cell carries no
    /// clock.
    pub clock: Option<CellSelection>,
}

/// Why the source supplying a satellite's position changed at an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransitionReason {
    /// The previously selected source no longer carried the cell.
    SoleAvailability,
    /// Precedence order chose a different source that was already available.
    Precedence,
    /// The previously selected source was rejected from the consensus as an
    /// outlier.
    OutlierRejection,
    /// The cell moved between a single-source carry and a multi-source
    /// consensus, or between combined and single-source selection.
    ConsensusChange,
}

/// One change in which source supplied a satellite's position.
#[derive(Debug, Clone, PartialEq)]
pub struct PrecedenceTransition {
    /// The satellite.
    pub satellite: GnssSatelliteId,
    /// The epoch at which the new source took over.
    pub epoch: Instant,
    /// The source supplying the previous accepted cell; `None` at a satellite's
    /// first accepted cell.
    pub from_source: Option<usize>,
    /// The source supplying this cell; `None` when the new cell is combined and
    /// so has no single supplier.
    pub to_source: Option<usize>,
    /// Why selection changed.
    pub reason: TransitionReason,
}

/// What one contributor supplied to the merged product.
#[derive(Debug, Clone, PartialEq)]
pub struct ContributorCoverage {
    /// Index into the input slice.
    pub source: usize,
    /// Accepted cells where this source was in the position or clock consensus.
    pub cells_contributed: usize,
    /// Accepted cells whose written position came from this source alone
    /// (single-source carry or precedence pick). Always zero under a combining
    /// rule, where no cell has a single supplier.
    pub cells_selected: usize,
    /// First accepted cell this source contributed to.
    pub first_epoch: Option<Instant>,
    /// Last accepted cell this source contributed to.
    pub last_epoch: Option<Instant>,
    /// Accepted cells this source contributed nothing to - the complement of
    /// `cells_contributed` over the merged product.
    pub cells_absent: usize,
}

/// Per-epoch merge provenance, recorded as the merge decided.
///
/// Present on [`MergeReport::provenance`] only when [`MergeOptions::provenance`]
/// requested it. The `Option` is deliberate: a caller must be able to tell
/// "provenance was not requested" from "provenance says one contributor".
#[derive(Debug, Clone, PartialEq)]
pub struct MergeProvenance {
    /// The mode that produced this record.
    pub mode: ProvenanceMode,
    /// One entry per accepted cell, in output order. Empty under
    /// [`ProvenanceMode::Summary`].
    pub cells: Vec<CellProvenance>,
    /// Every change of supplying source, in output order. Identical under both
    /// modes.
    pub transitions: Vec<PrecedenceTransition>,
    /// What each input contributed, indexed by source order.
    pub coverage: Vec<ContributorCoverage>,
}

/// The part one merged cell plays in a continuity finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeContinuityCellRole {
    /// The sample a hold-out residual compares against its prediction.
    HeldOut,
    /// A retained node the hold-out prediction was formed from.
    InterpolationNode,
    /// One end of the adjacent pair the speed gate measured.
    PairEnd,
    /// The epoch a duplicate-epoch finding names.
    RepeatedEpoch,
}

/// One merged cell a continuity finding rests on, with the selection the merge
/// recorded for it when it decided the cell.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeContinuityCell {
    /// Epoch of the cell, seconds since J2000.
    pub epoch_j2000_s: f64,
    /// The part the cell plays in the finding.
    pub role: MergeContinuityCellRole,
    /// How the merge arrived at the value it wrote there. The selected source
    /// ([`CellSelection::selected_source`]) and the rest of the agreeing
    /// members stay distinct. `None` when the merge recorded no selection for
    /// the cell.
    pub selection: Option<CellSelection>,
}

/// One continuity violation in the merged product, attributed to the
/// contributors whose records it rests on.
///
/// At a splice the actionable fact is not that the arc jumped but *between
/// which contributors* it jumped, which is why this exists rather than a bare
/// [`ContinuityDefect`]. A hold-out residual rests on more than its bracketing
/// pair: the prediction is formed from up to eleven retained nodes, and near a
/// run end the window slides inward, so a residual reported between two
/// records of one source can be caused by another source's nodes several
/// epochs away. [`Self::cells`] names every one of those records.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeContinuityViolation {
    /// The violation as the continuity check reported it.
    pub defect: ContinuityDefect,
    /// Sources whose value was written to the earlier record of the offending
    /// epoch pair, as recorded when the merge chose that cell: the selected
    /// source of a precedence or single-source cell, every member of a combined
    /// cell - the same meaning as [`Self::sources`]. Empty when the merged
    /// product carries no cell there (the check reached past the merge's own
    /// coverage).
    pub from_sources: Vec<usize>,
    /// Sources whose value was written to the later record of the offending
    /// epoch pair, in the same meaning.
    pub to_sources: Vec<usize>,
    /// Every merged cell the finding rests on, ascending by epoch: for a
    /// hold-out residual the held-out sample and each node its prediction used;
    /// for a speed-bound finding both ends of the pair; for a duplicate epoch
    /// that epoch. Empty for a single-sample series, which names no epoch.
    pub cells: Vec<MergeContinuityCell>,
    /// Every source whose value was written to a cell in [`Self::cells`],
    /// ascending: the selected source of a precedence or single-source cell,
    /// and every member of a combined cell.
    pub sources: Vec<usize>,
    /// Whether the cells in [`Self::cells`] were written from different
    /// contributors: different selected sources, or a combination over a
    /// different member set. A violation across a contributor change is a
    /// splice; one inside a single contributor's arc is that contributor's own
    /// discontinuity, and the two call for different follow-up.
    pub crosses_contributors: bool,
}

/// Continuity verification of the merged product, run as a merge
/// post-condition.
///
/// Reporting, never refusing: the merge still returns the product. Whether a
/// product with continuity defects is acceptable is the caller's decision.
#[derive(Debug, Clone, PartialEq)]
pub struct MergeContinuityReport {
    /// The full continuity report over the merged product.
    pub report: ContinuityReport,
    /// Each violation, attributed to the contributors whose records it rests
    /// on.
    pub violations: Vec<MergeContinuityViolation>,
    /// The merged product's position nodes, which window-scoped verdicts read
    /// to find exactly the nodes a window's interpolations use.
    pub nodes: InterpolationNodes,
}

impl MergeContinuityReport {
    /// Whether the merged product is attested continuous.
    pub fn attested(&self) -> bool {
        self.report.attested()
    }

    /// Violations that sit across a change of contributor - the splices.
    pub fn splices(&self) -> impl Iterator<Item = &MergeContinuityViolation> {
        self.violations
            .iter()
            .filter(|violation| violation.crosses_contributors)
    }

    /// Violations that the interpolations of an evaluation window rest on.
    ///
    /// The window's nodes are found exactly, per satellite, with the
    /// interpolator's own rule ([`InterpolationNodes::selected_nodes`]). A
    /// violation influences the window when those nodes include its held-out,
    /// repeated or pair-end record, or straddle a handover between its
    /// records: two consecutive records written from different sources, with a
    /// selected node at or before the earlier and one at or after the later. A
    /// window whose nodes include only records on one side of the handover, and
    /// not the record at fault, interpolates records the finding does not
    /// implicate. A single-sample series names no epoch and influences every
    /// window.
    pub fn violations_influencing(&self, window: EpochWindow) -> Vec<&MergeContinuityViolation> {
        self.violations
            .iter()
            .filter(|violation| violation.influences(window, &self.nodes))
            .collect()
    }

    /// Contributor-changing violations that influence an evaluation window;
    /// see [`Self::violations_influencing`].
    pub fn splices_influencing(&self, window: EpochWindow) -> Vec<&MergeContinuityViolation> {
        self.violations_influencing(window)
            .into_iter()
            .filter(|violation| violation.crosses_contributors)
            .collect()
    }

    /// One window-scoped decision over every violation.
    ///
    /// The decision refuses when any violation influences the window (see
    /// [`Self::violations_influencing`]). Both the influencing subset and
    /// complete finding lists remain available on the returned verdict.
    pub fn verdict_for_window(&self, window: EpochWindow) -> WindowContinuityVerdict<'_> {
        let influencing = self.violations_influencing(window);
        let decision = if influencing.is_empty() {
            WindowContinuityDecision::Accept
        } else {
            WindowContinuityDecision::Refuse
        };
        WindowContinuityVerdict {
            decision,
            influencing_defects: influencing
                .iter()
                .map(|violation| &violation.defect)
                .collect(),
            influencing_splices: influencing
                .into_iter()
                .filter(|violation| violation.crosses_contributors)
                .collect(),
            all_defects: &self.report.defects,
            all_splices: self.splices().collect(),
        }
    }
}

impl MergeContinuityViolation {
    /// Whether the interpolations of `window` rest on this violation; see
    /// [`MergeContinuityReport::violations_influencing`].
    fn influences(&self, window: EpochWindow, nodes: &InterpolationNodes) -> bool {
        if matches!(self.defect, ContinuityDefect::SingleSampleSeries { .. }) {
            return true;
        }
        let selected = nodes.selected_nodes(self.defect.satellite(), window);
        let (Some(&first), Some(&last)) = (selected.first(), selected.last()) else {
            return false;
        };
        let at_fault = self.cells.iter().any(|cell| {
            cell.role != MergeContinuityCellRole::InterpolationNode
                && selected
                    .binary_search_by(|node| node.total_cmp(&cell.epoch_j2000_s))
                    .is_ok()
        });
        let straddles_handover = self.cells.windows(2).any(|pair| {
            let (Some(earlier), Some(later)) = (&pair[0].selection, &pair[1].selection) else {
                return false;
            };
            written_sources(earlier) != written_sources(later)
                && first <= pair[0].epoch_j2000_s
                && last >= pair[1].epoch_j2000_s
        });
        at_fault || straddles_handover
    }
}

/// Audit trail for a [`merge`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MergeReport {
    /// Coordinate-label reconciliations applied before source consensus.
    pub frame_reconciliations: Vec<Sp3FrameReconciliation>,
    /// Cells where two or more sources disagreed beyond tolerance with no
    /// agreeing subset of `min_agree` - omitted from the merged product.
    pub quarantined: Vec<MergeFlag>,
    /// Cells carried from a single source (no cross-check was possible).
    pub single_source: Vec<MergeFlag>,
    /// Cells accepted by consensus where one or more sources were rejected as
    /// position outliers.
    pub position_outliers: Vec<MergeFlag>,
    /// Clock contributors rejected from an accepted clock consensus, or every
    /// clock contributor when an enabled consensus guard found no cluster.
    pub clock_outliers: Vec<MergeFlag>,
    /// Per-(epoch, satellite) agreement statistics for every accepted cell, in
    /// output (epoch, then satellite) order - one entry per cell written to the
    /// merged product. Quantifies how tightly the consensus sources clustered
    /// about the combined value (Gap: per-epoch quality metrics).
    pub agreement: Vec<AgreementMetric>,
    /// Per-epoch provenance, present only when [`MergeOptions::provenance`]
    /// requested it. `None` means "not requested", which a caller must be able
    /// to distinguish from a record naming one contributor.
    pub provenance: Option<MergeProvenance>,
    /// Continuity verification of the merged product, present only when
    /// [`MergeOptions::verify_continuity`] requested it.
    pub continuity: Option<MergeContinuityReport>,
    /// Union-grid epochs at which the merge accepted no cell, in time order.
    /// None of them is written to the product: an epoch block whose every
    /// record is the missing sentinel states nothing and would stretch the
    /// product's apparent coverage past its data. Each position some source
    /// carried there is in [`Self::quarantined`] or [`Self::arc_withheld`], and
    /// each source's clock in [`Self::clock_omissions`].
    pub omitted_epochs: Vec<Instant>,
    /// Cells whose position some source carried but precedence did not write,
    /// because the preferred source carried none there. Under
    /// [`MergePrecedenceScope::SatelliteArc`] this is each cell outside the arc
    /// owner's coverage that a later source fills; cell-scoped precedence
    /// always prefers a source that is present.
    pub arc_withheld: Vec<MergeFlag>,
    /// Each source's clock that the merge did not write, with the reason, in
    /// (epoch, satellite, source) order - including a clock left out of a
    /// cell that got its clock from other sources.
    pub clock_omissions: Vec<ClockOmission>,
    /// Input epochs that took no part in the merge, with the reason, in
    /// (source, epoch) order.
    pub dropped_input_epochs: Vec<DroppedInputEpoch>,
}

impl MergeReport {
    /// Contributor-changing violations that influence an evaluation window,
    /// when merge continuity verification was requested; see
    /// [`MergeContinuityReport::violations_influencing`].
    ///
    /// `None` means verification was not requested. `Some(Vec::new())` means it
    /// ran and no recorded splice influences the window.
    pub fn splices_influencing(
        &self,
        window: EpochWindow,
    ) -> Option<Vec<&MergeContinuityViolation>> {
        self.continuity
            .as_ref()
            .map(|report| report.splices_influencing(window))
    }

    /// Decide whether the optional continuity post-condition influences an
    /// evaluation window.
    ///
    /// `None` preserves the distinction that merge continuity verification was
    /// not requested. When present, the verdict includes the complete defect and
    /// splice lists even when it accepts the window.
    pub fn continuity_verdict_for_window(
        &self,
        window: EpochWindow,
    ) -> Option<WindowContinuityVerdict<'_>> {
        self.continuity
            .as_ref()
            .map(|report| report.verdict_for_window(window))
    }

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
        pooled_rms(self.agreement.iter().filter_map(|m| {
            m.position_rms_m
                .filter(|_| m.position_members >= 2)
                .map(|rms| (rms, m.position_members))
        }))
    }

    /// Largest single-cell position dispersion over all accepted cells, meters.
    /// `None` when there are no accepted cells with position.
    pub fn position_agreement_max_m(&self) -> Option<f64> {
        self.agreement
            .iter()
            .filter_map(|m| m.position_max_m)
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
    /// cells were all single-source are still listed with `satellites == 0` and
    /// `None` position spread so the caller sees every output epoch.
    pub fn per_epoch_agreement(&self) -> Vec<EpochAgreement> {
        let mut out: Vec<EpochAgreement> = Vec::new();
        for m in &self.agreement {
            // Cells of one output epoch are consecutive and carry that epoch's
            // one instant, so grouping compares the instant itself.
            if out.last().is_none_or(|agg| agg.epoch != m.epoch) {
                out.push(EpochAgreement {
                    epoch: m.epoch,
                    satellites: 0,
                    position_rms_m: None,
                    position_max_m: None,
                    clock_rms_s: None,
                    clock_max_s: None,
                });
            }
            // invariant: the branch above pushes the first aggregate before this
            // lookup, and every later aggregate remains in `out`.
            #[allow(clippy::expect_used)]
            let agg = out.last_mut().expect("just pushed");
            if m.position_members >= 2 {
                agg.satellites += 1;
                if let Some(max) = m.position_max_m {
                    agg.position_max_m = Some(fold_max(agg.position_max_m, max));
                }
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
            agg.position_rms_m = pooled_rms(
                self.agreement
                    .iter()
                    .filter(|m| m.position_members >= 2 && m.epoch == agg.epoch)
                    .filter_map(|m| m.position_rms_m.map(|rms| (rms, m.position_members))),
            );
            agg.clock_rms_s = pooled_rms(
                self.agreement
                    .iter()
                    .filter(|m| m.epoch == agg.epoch)
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
/// Every epoch is placed on the exact 10-nanosecond tick axis an SP3 epoch
/// record states, so two inputs share an epoch only when they state the same
/// instant. Each input must lie on a grid: equal steps, or steps that are all
/// whole multiples of its declared interval (a product that skips epochs of
/// its grid); one whose steps are neither is refused, as are epochs out of
/// order. By default the output grid is the greatest common divisor of every
/// input's grid step and every input epoch's offset from the earliest, so it
/// holds every input epoch and the header states its step. An explicit
/// [`MergeOptions::target_epoch_interval_s`], anchored at the earliest epoch,
/// keeps only the input epochs on its grid and lists every other one in
/// [`MergeReport::dropped_input_epochs`]. Only epochs actually present in an
/// input are used, and nothing is interpolated. For every (epoch, satellite)
/// cell on that union grid:
///
/// - **Union satellite coverage.** A satellite present in any input may appear
///   in the output at every union-grid epoch where an input carries that cell.
/// - **Position consensus.** With one source the value is carried through
///   (`single_source`). With several, the largest subset of sources mutually
///   within `position_tolerance_m` is found; if it has at least `min_agree`
///   members it is combined per `combine` and any sources outside it are recorded
///   as `position_outliers`. If no such subset exists the cell is `quarantined`
///   (omitted) - never averaged across disagreeing centers.
/// - **Clock consensus.** Clocks are first put on a common datum (each source
///   aligned to the first via [`clock_reference_offset`]), then combined by the
///   same agreement rule; a cell with no clock consensus carries no clock. A
///   non-reference source's datum offset is linearly interpolated between
///   bracketing epochs where at least `clock_min_common` common clocks made it
///   observable. Outside that bracket, or when no bracket exists, the source
///   contributes **no** clock rather than an unaligned one; its position is
///   still merged. Every source clock the merge does not write is listed in
///   [`MergeReport::clock_omissions`] with the reason, including one left out
///   of a cell that got its clock from other sources.
/// - **No empty epochs.** A union-grid epoch at which no cell was accepted is
///   not written, and is listed in [`MergeReport::omitted_epochs`]. A merge
///   that accepts no cell at any epoch returns a product with no epochs, whose
///   header describes the first union-grid epoch, together with the report
///   that says why. SP3 sets no minimum epoch count, so that product writes as
///   a header whose line 1 states that start.
///
/// The union epoch grid is not a statement of what each satellite carries. Use
/// [`Sp3::satellite_coverage`] on the merged product for each satellite's
/// position and clock coverage and its gaps. A continuity check with no
/// findings attests that the records present are mutually consistent; it does
/// not say that every epoch of the grid can be interpolated for every
/// satellite.
///
/// `Precedence` is resolved per cell by default, so a lower-precedence source
/// fills a cell missing from all earlier sources. Whole-satellite-arc ownership
/// remains available through [`MergePrecedenceScope::SatelliteArc`]. The
/// optional [`OutlierRejectOptions`] independently guards contested precedence
/// cells: the deterministic largest mutually-agreeing cluster must contain at
/// least `max(min_agree, 2)` sources. The preferred source is retained when it
/// belongs to that cluster; otherwise the earliest-listed cluster member wins.
///
/// All inputs must share an exact SP3 time-system label. Coordinate-system
/// labels must also match unless [`MergeOptions::frame_reconciliation`] opts
/// into a caller assertion or catalog Helmert reconciliation; every such
/// reconciliation is recorded in [`MergeReport::frame_reconciliations`].
/// Otherwise coordinate-label mismatches are rejected. The merged record flags
/// are the union (OR) of the contributing sources' flags - in particular a
/// `clock_event` on any clock-consensus member is preserved, so the interpolator
/// still splits the clock arc. The merged header is **synthetic**: its
/// first-epoch fields describe the first epoch written, each the value its
/// field states, and its data type is position-only. The MJD fraction is the within-day tick count rounded once to
/// the thirteen decimals its field holds, so an off-midnight start such as
/// 05:00 writes and reads back as the value a real file holds.
///
/// Preservation of velocities and rates: merge policy intentionally drops
/// velocities and clock rates for all records (both position states and clock-only
/// records) in the merged product. Merged states retain combined positions and
/// clocks, but have `velocity: None` and `clock_rate_s_s: None`. Clock-only records
/// retain combined clocks, but have `velocity: None`, `clock_rate_s_s: None`, and
/// `clock_rate_raw: None`.
///
/// Pure and deterministic: order the inputs by center precedence and ties (equal
/// cluster sizes, `Precedence` combine) resolve to the earliest-listed source.
/// The merged product's interpolation nodes are the consensus values, so it
/// samples and interpolates like any other [`Sp3`] (it is a derived combination,
/// not a byte-faithful copy of any one center). Consensus is exact max-clique for
/// normal source counts and uses a deterministic greedy fallback above the exact
/// search cap, so hostile disagreement graphs remain bounded.
pub fn merge(sources: &[Sp3], opts: &MergeOptions) -> Result<(Sp3, MergeReport)> {
    let prepared = prepare_merge_inputs(sources, opts)?;
    let prepared_sources = prepared.sources;
    let frame_reconciliations = prepared.frame_reconciliations;
    let sources = prepared_sources.as_slice();

    let timing = prepare_merge_timing(sources, opts)?;
    let epoch_interval_s = interval_seconds(timing.step_ticks);
    let cells = emit_merge_cells(sources, opts, &timing, frame_reconciliations)?;

    // The header's first-epoch fields describe the first epoch written. A merge
    // that accepted no cell writes no epoch; its header then describes the first
    // union-grid epoch, the one every omission is counted from.
    let header_tick = match cells.out_ticks.first() {
        Some(&tick) => tick,
        None => {
            // invariant: `prepare_merge_timing` refuses an empty union grid.
            #[allow(clippy::expect_used)]
            let (&tick, _) = timing
                .epoch_keys
                .iter()
                .next()
                .expect("non-empty union grid");
            tick
        }
    };
    let synthesized = synthesize_merge_header(
        sources,
        &timing.first_ticks,
        header_tick,
        cells.out_ticks.len(),
        &cells.all_sats,
        epoch_interval_s,
    )?;
    let (merged, mut report, continuity_selection) =
        emit_merged_product(sources, synthesized, header_tick, cells);

    if let Some(continuity_options) = opts.verify_continuity.as_ref() {
        report.continuity = Some(verify_merged_continuity(
            &merged,
            continuity_options,
            &continuity_selection,
        ));
    }

    Ok((merged, report))
}

struct PreparedMergeInputs {
    sources: Vec<Sp3>,
    frame_reconciliations: Vec<Sp3FrameReconciliation>,
}

struct MergeCellOutput {
    out_epochs: Vec<Instant>,
    /// Each written epoch on the tick axis, parallel to `out_epochs`.
    out_ticks: Vec<i128>,
    out_states: Vec<BTreeMap<GnssSatelliteId, Sp3State>>,
    out_clock_records: Vec<BTreeMap<GnssSatelliteId, Sp3ClockRecord>>,
    out_raw: Vec<BTreeMap<GnssSatelliteId, RawNode>>,
    all_sats: BTreeSet<GnssSatelliteId>,
    report: MergeReport,
    continuity_selection: BTreeMap<(GnssSatelliteId, i64), CellSelection>,
}

struct MergeHeaderOutput {
    header: Sp3Header,
    mandatory_header_lines: usize,
    declared_satellite_tokens: Vec<String>,
    epoch_position_tokens: Vec<Vec<String>>,
    epoch_state_record_sequence: Vec<Vec<(char, String)>>,
}

/// Assemble the canonical `Sp3` product from the ordered merge stages without
/// changing the serializer-facing header counts, records, or source comment.
fn emit_merged_product(
    sources: &[Sp3],
    synthesized: MergeHeaderOutput,
    header_tick: i128,
    cells: MergeCellOutput,
) -> (
    Sp3,
    MergeReport,
    BTreeMap<(GnssSatelliteId, i64), CellSelection>,
) {
    let MergeHeaderOutput {
        header,
        mandatory_header_lines,
        declared_satellite_tokens,
        epoch_position_tokens,
        epoch_state_record_sequence,
    } = synthesized;
    let MergeCellOutput {
        out_epochs,
        out_ticks,
        out_states,
        out_clock_records,
        out_raw,
        report,
        continuity_selection,
        ..
    } = cells;
    let out_epoch_j2000_s: Vec<f64> = out_ticks.iter().map(|&tick| tick_seconds(tick)).collect();
    let merged = Sp3 {
        header,
        epochs: out_epochs,
        declared_num_epochs: out_epoch_j2000_s.len() as u64,
        // Line 1 states the header's start: the first epoch written, or for a
        // merge that wrote none, the first union-grid epoch.
        declared_start_j2000_s: Some(tick_seconds(header_tick)),
        terminal_record: TerminalRecordState::valid(),
        satellite_header_lines: mandatory_header_lines,
        accuracy_header_lines: mandatory_header_lines,
        time_system_header_lines: 2,
        float_header_lines: 2,
        integer_header_lines: 2,
        header_comment_lines: 4,
        declared_satellite_count: Some(declared_satellite_tokens.len()),
        declared_satellite_tokens,
        epoch_velocity_tokens: vec![Vec::new(); epoch_position_tokens.len()],
        epoch_position_tokens,
        epoch_state_record_sequence,
        epoch_j2000_s: out_epoch_j2000_s,
        states: out_states,
        clock_records: out_clock_records,
        interp_raw: out_raw,
        interpolation: Sp3InterpolationOptions::default(),
        comments: vec![format!("MERGED from {} SP3 products", sources.len())],
        skipped_records: sources.iter().map(|s| s.skipped_records).sum(),
    };
    (merged, report, continuity_selection)
}

/// Consume ordered merged cells and source metadata, and produce the synthetic
/// SP3 header plus the canonical output record-token sequences.
fn synthesize_merge_header(
    sources: &[Sp3],
    first_ticks: &[Option<i128>],
    first_tick: i128,
    epoch_count: usize,
    all_sats: &BTreeSet<GnssSatelliteId>,
    epoch_interval_s: f64,
) -> Result<MergeHeaderOutput> {
    // Base the non-epoch metadata on a source product, but derive the first-epoch
    // header fields from the merged grid itself. Mixed cadence / coverage can make
    // the merged first epoch later than every input's first epoch, so cloning
    // those fields from any input would make the `##` line stale.
    let base_idx = first_ticks
        .iter()
        .position(|tick| *tick == Some(first_tick))
        .or_else(|| {
            first_ticks
                .iter()
                .enumerate()
                .filter_map(|(index, tick)| tick.map(|tick| (tick, index)))
                .min()
                .map(|(_, index)| index)
        })
        .unwrap_or(0);
    let first_epoch_header = first_epoch_header_fields(first_tick).ok_or_else(|| {
        Error::InvalidInput("merged SP3 first epoch cannot be represented in header fields".into())
    })?;

    let satellites: Vec<_> = all_sats.iter().copied().collect();
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
        num_epochs: epoch_count as u64,
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

    let mandatory_header_lines = 5.max(header.satellites.len().div_ceil(17));
    let declared_satellite_tokens = header
        .satellites
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    let epoch_position_tokens = vec![declared_satellite_tokens.clone(); epoch_count];
    let epoch_state_record_sequence = epoch_position_tokens
        .iter()
        .map(|tokens| {
            tokens
                .iter()
                .cloned()
                .map(|token| ('P', token))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();

    Ok(MergeHeaderOutput {
        header,
        mandatory_header_lines,
        declared_satellite_tokens,
        epoch_position_tokens,
        epoch_state_record_sequence,
    })
}

/// Consume ordered merge timing data and emit every accepted epoch/satellite cell,
/// preserving consensus, provenance, report, and native interpolation values.
fn emit_merge_cells(
    sources: &[Sp3],
    opts: &MergeOptions,
    timing: &MergeTiming,
    frame_reconciliations: Vec<Sp3FrameReconciliation>,
) -> Result<MergeCellOutput> {
    let epoch_index = timing.epoch_index.as_slice();
    let epoch_keys = &timing.epoch_keys;
    let clock_offset = timing.clock_offset.as_slice();

    let precedence_source_for_sat = if opts.combine == MergeCombine::Precedence
        && opts.precedence_scope == MergePrecedenceScope::SatelliteArc
    {
        Some(precedence_sources_for_satellites(
            sources,
            epoch_index,
            epoch_keys,
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

    let mut out_epochs: Vec<Instant> = Vec::with_capacity(epoch_keys.len());
    // Provenance accumulators. Every entry is written at the moment the merge
    // decides the cell; nothing here is reconstructed from the merged product.
    let mut prov_cells: Vec<CellProvenance> = Vec::new();
    let mut prov_transitions: Vec<PrecedenceTransition> = Vec::new();
    let mut prov_contributed: Vec<usize> = vec![0; sources.len()];
    let mut prov_selected: Vec<usize> = vec![0; sources.len()];
    let mut prov_first: Vec<Option<Instant>> = vec![None; sources.len()];
    let mut prov_last: Vec<Option<Instant>> = vec![None; sources.len()];
    let mut prov_accepted_cells: usize = 0;
    let mut prov_previous: BTreeMap<GnssSatelliteId, CellSelection> = BTreeMap::new();
    let mut continuity_selection: BTreeMap<(GnssSatelliteId, i64), CellSelection> = BTreeMap::new();

    let mut out_ticks: Vec<i128> = Vec::with_capacity(epoch_keys.len());
    let mut out_states: Vec<BTreeMap<GnssSatelliteId, Sp3State>> =
        Vec::with_capacity(epoch_keys.len());
    let mut out_clock_records: Vec<BTreeMap<GnssSatelliteId, Sp3ClockRecord>> =
        Vec::with_capacity(epoch_keys.len());
    let mut out_raw: Vec<BTreeMap<GnssSatelliteId, RawNode>> = Vec::with_capacity(epoch_keys.len());
    let mut report = MergeReport {
        frame_reconciliations,
        dropped_input_epochs: timing.dropped.clone(),
        ..MergeReport::default()
    };
    let mut all_sats: BTreeSet<GnssSatelliteId> = BTreeSet::new();

    for (&key, &epoch) in epoch_keys {
        // The continuity check reads the product on the interpolation node
        // axis, so its findings are keyed back to cells by that axis.
        let node_key = precise_node_j2000_seconds_from_instant(&epoch).map(|node| node as i64);
        let mut states: BTreeMap<GnssSatelliteId, Sp3State> = BTreeMap::new();
        let mut clock_records: BTreeMap<GnssSatelliteId, Sp3ClockRecord> = BTreeMap::new();
        let mut raws: BTreeMap<GnssSatelliteId, RawNode> = BTreeMap::new();

        // Satellites present at this epoch in any source, after any requested
        // constellation filter.
        let mut sats: BTreeSet<GnssSatelliteId> = BTreeSet::new();
        for (idx, s) in sources.iter().enumerate() {
            if let Some(&ei) = epoch_index[idx].get(&key) {
                if let Ok(map) = s.states_at(ei) {
                    sats.extend(map.keys().copied().filter(|sat| allowed_system(sat)));
                }
                if let Ok(map) = s.clock_records_at(ei) {
                    sats.extend(map.keys().copied().filter(|sat| allowed_system(sat)));
                }
            }
        }

        for sat in sats {
            // (source_idx, position_m, flags) and (source_idx, datum-aligned
            // clock_s, flags). A non-reference source contributes a clock only
            // when its datum offset can be estimated exactly or between
            // bracketing estimates; otherwise its clock would be unaligned, so
            // it is omitted (the position is still gathered).
            let arc_preferred_source = precedence_source_for_sat
                .as_ref()
                .and_then(|by_sat| by_sat.get(&sat).copied());

            let mut pos: Vec<(usize, [f64; 3], Sp3Flags)> = Vec::new();
            let mut clk: Vec<(usize, f64, Sp3Flags)> = Vec::new();
            // Sources that carried a clock here whose datum offset to source 0
            // could not be estimated at this epoch.
            let mut unaligned_clk: Vec<usize> = Vec::new();
            for (idx, s) in sources.iter().enumerate() {
                let Some(&ei) = epoch_index[idx].get(&key) else {
                    continue;
                };
                if let Ok(map) = s.states_at(ei) {
                    if let Some(state) = map.get(&sat) {
                        pos.push((idx, state.position.as_array(), state.flags));
                        if let Some(c) = state.clock_s {
                            let offset = if idx == 0 {
                                Some(0.0)
                            } else {
                                clock_offset_at(&clock_offset[idx], key)
                            };
                            if let Some(off) = offset {
                                let aligned = c - off;
                                if aligned.is_finite() {
                                    clk.push((idx, aligned, state.flags));
                                }
                            } else {
                                unaligned_clk.push(idx);
                            }
                        }
                    }
                }
                if let Ok(map) = s.clock_records_at(ei) {
                    if let Some(clock_rec) = map.get(&sat) {
                        let offset = if idx == 0 {
                            Some(0.0)
                        } else {
                            clock_offset_at(&clock_offset[idx], key)
                        };
                        if let Some(off) = offset {
                            let aligned = clock_rec.clock_s - off;
                            if aligned.is_finite() {
                                clk.push((idx, aligned, clock_rec.flags));
                            }
                        } else {
                            unaligned_clk.push(idx);
                        }
                    }
                }
            }

            let position_preferred_source = match opts.precedence_scope {
                MergePrecedenceScope::Cell => pos.first().map(|(source, _, _)| *source),
                MergePrecedenceScope::SatelliteArc => arc_preferred_source,
            };
            let clock_preferred_source = match opts.precedence_scope {
                MergePrecedenceScope::Cell => clk.first().map(|(source, _, _)| *source),
                MergePrecedenceScope::SatelliteArc => arc_preferred_source,
            };

            let flag = |srcs: Vec<usize>| MergeFlag {
                epoch,
                satellite: sat,
                sources: srcs,
            };

            // Precedence writes a position only from the preferred source. Under
            // satellite-arc scope that is the arc owner, which can be missing
            // from a cell other sources carry; those positions are withheld,
            // and recorded rather than dropped without trace.
            if opts.combine == MergeCombine::Precedence
                && !pos.is_empty()
                && !position_preferred_source
                    .is_some_and(|preferred| pos.iter().any(|(src, _, _)| *src == preferred))
            {
                report
                    .arc_withheld
                    .push(flag(pos.iter().map(|(src, _, _)| *src).collect()));
            }

            // Position consensus -> the merged position and the indices (into
            // `pos`) of the sources that contributed it. Cell precedence selects
            // the first source present here; satellite-arc precedence can leave
            // a deliberate hole when the arc owner is missing.
            let pos_result = if opts.combine == MergeCombine::Precedence {
                if let Some(preferred_source) = position_preferred_source {
                    if let Some(preferred_idx) =
                        pos.iter().position(|(src, _, _)| *src == preferred_source)
                    {
                        if pos.len() == 1 {
                            report.single_source.push(flag(vec![pos[preferred_idx].0]));
                            Some((
                                pos[preferred_idx].1,
                                vec![preferred_idx],
                                CellSelection::SingleSource {
                                    source: pos[preferred_idx].0,
                                },
                            ))
                        } else if let Some(reject) = opts.outlier_reject {
                            let pts: Vec<[f64; 3]> = pos.iter().map(|(_, p, _)| *p).collect();
                            let cluster = largest_within(&pts, |a, b| {
                                dist3(a, b) <= reject.position_tolerance_m
                            });
                            if cluster.len() >= opts.min_agree.max(2) {
                                let selected_idx = if cluster.contains(&preferred_idx) {
                                    preferred_idx
                                } else {
                                    cluster[0]
                                };
                                let rejected: Vec<usize> = (0..pos.len())
                                    .filter(|i| !cluster.contains(i))
                                    .map(|i| pos[i].0)
                                    .collect();
                                let rejected_selection = !rejected.is_empty();
                                if rejected_selection {
                                    report.position_outliers.push(flag(rejected));
                                }
                                let selection = CellSelection::Precedence {
                                    source: pos[selected_idx].0,
                                    members: cluster.iter().map(|&i| pos[i].0).collect(),
                                };
                                Some((pos[selected_idx].1, cluster, selection))
                            } else {
                                report
                                    .quarantined
                                    .push(flag(pos.iter().map(|(i, _, _)| *i).collect()));
                                None
                            }
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
                                let selection = CellSelection::Precedence {
                                    source: pos[preferred_idx].0,
                                    members: cluster.iter().map(|&i| pos[i].0).collect(),
                                };
                                Some((pos[preferred_idx].1, cluster, selection))
                            } else {
                                report
                                    .quarantined
                                    .push(flag(pos.iter().map(|(i, _, _)| *i).collect()));
                                None
                            }
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else if pos.len() == 1 {
                report.single_source.push(flag(vec![pos[0].0]));
                Some((
                    pos[0].1,
                    vec![0usize],
                    CellSelection::SingleSource { source: pos[0].0 },
                ))
            } else if !pos.is_empty() {
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
                    let selection = CellSelection::Combined {
                        rule: opts.combine,
                        members: members.iter().map(|(source, _)| *source).collect(),
                    };
                    Some((combine3(&members, opts.combine), cluster, selection))
                } else {
                    report
                        .quarantined
                        .push(flag(pos.iter().map(|(i, _, _)| *i).collect()));
                    None
                }
            } else {
                None
            };

            // Clock consensus, independent of position -> the merged clock and the
            // indices (into `clk`) of the sources that contributed it.
            let mut clk_selection: Option<CellSelection> = None;
            let (clock_s, clk_members): (Option<f64>, Vec<usize>) = if clk.is_empty() {
                (None, Vec::new())
            } else if opts.combine == MergeCombine::Precedence {
                match clock_preferred_source
                    .and_then(|src| clk.iter().position(|(clock_src, _, _)| *clock_src == src))
                {
                    None => (None, Vec::new()),
                    Some(preferred_idx) if clk.len() == 1 => {
                        clk_selection = Some(CellSelection::SingleSource {
                            source: clk[preferred_idx].0,
                        });
                        (Some(clk[preferred_idx].1), vec![preferred_idx])
                    }
                    Some(preferred_idx) if opts.outlier_reject.is_some() => {
                        // invariant: this match arm is guarded by
                        // `outlier_reject.is_some()` immediately above.
                        #[allow(clippy::expect_used)]
                        let reject = opts.outlier_reject.expect("checked above");
                        let vals: Vec<f64> = clk.iter().map(|(_, c, _)| *c).collect();
                        let cluster =
                            largest_within(&vals, |a, b| (a - b).abs() <= reject.clock_tolerance_s);
                        if cluster.len() >= opts.min_agree.max(2) {
                            let selected_idx = if cluster.contains(&preferred_idx) {
                                preferred_idx
                            } else {
                                cluster[0]
                            };
                            let rejected: Vec<usize> = (0..clk.len())
                                .filter(|i| !cluster.contains(i))
                                .map(|i| clk[i].0)
                                .collect();
                            if !rejected.is_empty() {
                                report.clock_outliers.push(flag(rejected));
                            }
                            clk_selection = Some(CellSelection::Precedence {
                                source: clk[selected_idx].0,
                                members: cluster.iter().map(|&i| clk[i].0).collect(),
                            });
                            (Some(clk[selected_idx].1), cluster)
                        } else {
                            report
                                .clock_outliers
                                .push(flag(clk.iter().map(|(source, _, _)| *source).collect()));
                            (None, Vec::new())
                        }
                    }
                    Some(preferred_idx) => {
                        let vals: Vec<f64> = clk.iter().map(|(_, c, _)| *c).collect();
                        let cluster = largest_within_containing(&vals, preferred_idx, |a, b| {
                            (a - b).abs() <= opts.clock_tolerance_s
                        });
                        if cluster.len() >= opts.min_agree {
                            let rejected: Vec<usize> = (0..clk.len())
                                .filter(|i| !cluster.contains(i))
                                .map(|i| clk[i].0)
                                .collect();
                            if !rejected.is_empty() {
                                report.clock_outliers.push(flag(rejected));
                            }
                            clk_selection = Some(CellSelection::Precedence {
                                source: clk[preferred_idx].0,
                                members: cluster.iter().map(|&i| clk[i].0).collect(),
                            });
                            (Some(clk[preferred_idx].1), cluster)
                        } else {
                            (None, Vec::new())
                        }
                    }
                }
            } else if clk.len() == 1 {
                clk_selection = Some(CellSelection::SingleSource { source: clk[0].0 });
                (Some(clk[0].1), vec![0usize])
            } else {
                let vals: Vec<f64> = clk.iter().map(|(_, c, _)| *c).collect();
                let cluster = largest_within(&vals, |a, b| (a - b).abs() <= opts.clock_tolerance_s);
                if cluster.len() >= opts.min_agree {
                    let rejected: Vec<usize> = (0..clk.len())
                        .filter(|i| !cluster.contains(i))
                        .map(|i| clk[i].0)
                        .collect();
                    if !rejected.is_empty() {
                        report.clock_outliers.push(flag(rejected));
                    }
                    let members: Vec<(usize, f64)> =
                        cluster.iter().map(|&i| (clk[i].0, clk[i].1)).collect();
                    clk_selection = Some(CellSelection::Combined {
                        rule: opts.combine,
                        members: members.iter().map(|(source, _)| *source).collect(),
                    });
                    (Some(combine_axis(&members, opts.combine)), cluster)
                } else {
                    (None, Vec::new())
                }
            };

            // Every source clock the cell does not write, per source: a clock
            // off the reference datum takes no part in the consensus whether or
            // not the cell gets a clock from other sources; aligned clocks are
            // left out only when the cell gets none.
            let mut omitted: Vec<(usize, ClockOmissionReason)> = unaligned_clk
                .iter()
                .map(|&source| (source, ClockOmissionReason::DatumNotObservable))
                .collect();
            if clock_s.is_none() && !clk.is_empty() {
                let reason = if opts.combine == MergeCombine::Precedence
                    && !clock_preferred_source
                        .is_some_and(|preferred| clk.iter().any(|(src, _, _)| *src == preferred))
                {
                    ClockOmissionReason::PreferredSourceWithoutClock {
                        preferred: clock_preferred_source,
                    }
                } else {
                    ClockOmissionReason::NoConsensus
                };
                omitted.extend(clk.iter().map(|(source, _, _)| (*source, reason)));
            }
            omitted.sort_by_key(|(source, _)| *source);
            report
                .clock_omissions
                .extend(omitted.into_iter().map(|(source, reason)| ClockOmission {
                    epoch,
                    satellite: sat,
                    source,
                    reason,
                    cell_has_clock: clock_s.is_some(),
                }));

            // If neither position nor clock was agreed upon, satellite is absent.
            if pos_result.is_none() && clock_s.is_none() {
                continue;
            }

            // Preserve record flags: OR the orbit flags across the position
            // members and the clock flags across the clock members, so a
            // `clock_event` (clock reset) or maneuver on any contributing source
            // survives into the merged product.
            let mut flags = Sp3Flags::default();
            if let Some((_, ref pos_members, _)) = pos_result {
                for &i in pos_members {
                    flags.maneuver |= pos[i].2.maneuver;
                    flags.orbit_predicted |= pos[i].2.orbit_predicted;
                }
            }
            for &i in &clk_members {
                flags.clock_event |= clk[i].2.clock_event;
                flags.clock_predicted |= clk[i].2.clock_predicted;
            }

            if let Some((position_m, pos_members, pos_selection)) = pos_result {
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
                    position_rms_m: Some(position_rms_m),
                    position_max_m: Some(position_max_m),
                    clock_members: clock_members_n,
                    clock_rms_s,
                    clock_max_s,
                });

                if let (true, Some(node_key)) = (opts.verify_continuity.is_some(), node_key) {
                    continuity_selection.insert((sat, node_key), pos_selection.clone());
                }

                if let Some(provenance_mode) = opts.provenance {
                    record_cell_provenance(
                        RecordCellProvenance {
                            epoch,
                            sat,
                            position: Some(&pos_selection),
                            clock: clk_selection.as_ref(),
                            candidates: &pos.iter().map(|(src, _, _)| *src).collect::<Vec<_>>(),
                            mode: provenance_mode,
                        },
                        &mut ProvenanceAccumulator {
                            cells: &mut prov_cells,
                            transitions: &mut prov_transitions,
                            contributed: &mut prov_contributed,
                            selected: &mut prov_selected,
                            first: &mut prov_first,
                            last: &mut prov_last,
                            accepted_cells: &mut prov_accepted_cells,
                            previous: &mut prov_previous,
                        },
                    );
                }

                all_sats.insert(sat);
                states.insert(
                    sat,
                    Sp3State {
                        position: ItrfPositionM::new(position_m[0], position_m[1], position_m[2])
                            .map_err(|error| Error::InvalidInput(error.to_string()))?,
                        clock_s,
                        velocity: None,
                        clock_rate_s_s: None,
                        flags,
                    },
                );
                let clock_us = match clock_s {
                    Some(c) => match preserved_source_raw_clock_us(
                        sources,
                        epoch_index,
                        key,
                        &sat,
                        clk_selection.as_ref(),
                        clock_offset,
                        c,
                    ) {
                        Some(raw_opt) => raw_opt,
                        None => Some(c * 1.0e6),
                    },
                    None => None,
                };
                raws.insert(
                    sat,
                    RawNode {
                        km: [
                            position_m[0] / KM_TO_M,
                            position_m[1] / KM_TO_M,
                            position_m[2] / KM_TO_M,
                        ],
                        clock_us,
                        clock_event: flags.clock_event,
                    },
                );
            } else if let Some(c) = clock_s {
                // Position is absent or quarantined, but valid clock consensus exists:
                // retain as a clock-only record.
                let (clock_members_n, clock_rms_s, clock_max_s) = {
                    let (rms, max) = clock_dispersion(&clk, &clk_members, c);
                    (clk_members.len(), Some(rms), Some(max))
                };
                report.agreement.push(AgreementMetric {
                    epoch,
                    satellite: sat,
                    position_members: 0,
                    position_rms_m: None,
                    position_max_m: None,
                    clock_members: clock_members_n,
                    clock_rms_s,
                    clock_max_s,
                });

                if clk.len() == 1 {
                    report.single_source.push(flag(vec![clk[0].0]));
                }

                if let (true, Some(node_key)) = (opts.verify_continuity.is_some(), node_key) {
                    if let Some(ref clk_sel) = clk_selection {
                        continuity_selection.insert((sat, node_key), clk_sel.clone());
                    }
                }

                if let Some(provenance_mode) = opts.provenance {
                    record_cell_provenance(
                        RecordCellProvenance {
                            epoch,
                            sat,
                            position: None,
                            clock: clk_selection.as_ref(),
                            candidates: &clk.iter().map(|(src, _, _)| *src).collect::<Vec<_>>(),
                            mode: provenance_mode,
                        },
                        &mut ProvenanceAccumulator {
                            cells: &mut prov_cells,
                            transitions: &mut prov_transitions,
                            contributed: &mut prov_contributed,
                            selected: &mut prov_selected,
                            first: &mut prov_first,
                            last: &mut prov_last,
                            accepted_cells: &mut prov_accepted_cells,
                            previous: &mut prov_previous,
                        },
                    );
                }

                all_sats.insert(sat);
                let clock_us = match preserved_source_raw_clock_us(
                    sources,
                    epoch_index,
                    key,
                    &sat,
                    clk_selection.as_ref(),
                    clock_offset,
                    c,
                ) {
                    Some(Some(raw_us)) => raw_us,
                    _ => c * 1.0e6,
                };
                clock_records.insert(
                    sat,
                    Sp3ClockRecord {
                        clock_s: c,
                        clock_us,
                        velocity: None,
                        clock_rate_s_s: None,
                        clock_rate_raw: None,
                        flags,
                    },
                );
            }
        }

        // An epoch at which no cell was accepted is not written: an SP3 epoch
        // block whose every record is the missing sentinel states nothing, and
        // it would stretch the product's apparent coverage past the data.
        if states.is_empty() && clock_records.is_empty() {
            report.omitted_epochs.push(epoch);
            continue;
        }
        out_epochs.push(epoch);
        out_ticks.push(key);
        out_states.push(states);
        out_clock_records.push(clock_records);
        out_raw.push(raws);
    }

    report.provenance = opts.provenance.map(|mode| MergeProvenance {
        mode,
        cells: prov_cells,
        transitions: prov_transitions,
        coverage: (0..sources.len())
            .map(|source| ContributorCoverage {
                source,
                cells_contributed: prov_contributed[source],
                cells_selected: prov_selected[source],
                first_epoch: prov_first[source],
                last_epoch: prov_last[source],
                cells_absent: prov_accepted_cells - prov_contributed[source],
            })
            .collect(),
    });

    Ok(MergeCellOutput {
        out_epochs,
        out_ticks,
        out_states,
        out_clock_records,
        out_raw,
        all_sats,
        report,
        continuity_selection,
    })
}

struct MergeTiming {
    /// Per source, tick -> index into its epochs.
    epoch_index: Vec<BTreeMap<i128, usize>>,
    /// Per source, the tick of its first placed epoch.
    first_ticks: Vec<Option<i128>>,
    /// The output grid step in ticks.
    step_ticks: i128,
    clock_offset: Vec<BTreeMap<i128, f64>>,
    /// The union grid: tick -> representative epoch from the earliest-listed
    /// source carrying it.
    epoch_keys: BTreeMap<i128, Instant>,
    dropped: Vec<DroppedInputEpoch>,
}

/// Consume raw SP3 sources and merge options, validate their combinability, and
/// produce frame-reconciled sources plus the reconciliation audit trail.
fn prepare_merge_inputs(sources: &[Sp3], opts: &MergeOptions) -> Result<PreparedMergeInputs> {
    if sources.is_empty() {
        return Err(Error::InvalidInput(
            "merge requires at least one SP3 product".into(),
        ));
    }

    validate_merge_options(opts)?;

    // Inputs must be combinable: epochs are matched in one exact product time
    // system, and positions are only comparable in an exactly common coordinate
    // system / frame unless the caller explicitly opted into one of the audited
    // reconciliation mechanisms below.
    let base = &sources[0].header;
    for s in &sources[1..] {
        if s.header.time_system != base.time_system {
            return Err(Error::InvalidInput(format!(
                "merge inputs have mismatched SP3 time systems ({:?} vs {:?})",
                base.time_system, s.header.time_system
            )));
        }
    }

    let (sources, frame_reconciliations) = reconcile_sp3_coordinate_labels(sources, opts)?;
    Ok(PreparedMergeInputs {
        sources,
        frame_reconciliations,
    })
}

/// Consume frame-reconciled sources and merge timing options, and produce the
/// exact epoch index of every input, the clock-datum offsets, the output grid
/// step and the union grid.
///
/// Every epoch is placed on the exact 10-nanosecond tick axis
/// ([`super::grid`]), so two inputs share an epoch only when they state the
/// same instant, and epochs never fuse by rounding. Each input must lie on a
/// grid by the rule [`Sp3::satellite_coverage`] also reports
/// ([`super::grid::product_grid`]): equal steps, or steps that are all whole
/// multiples of its declared interval. The default output grid is the greatest
/// common divisor of every input's grid step and every input epoch's offset
/// from the earliest one, so it holds every input epoch and none is dropped. An
/// explicit target grid, anchored at the earliest epoch, drops the input
/// epochs it does not hold, and reports each.
fn prepare_merge_timing(sources: &[Sp3], opts: &MergeOptions) -> Result<MergeTiming> {
    let source_ticks: Vec<Vec<Option<i128>>> = sources.iter().map(product_ticks).collect();
    let mut dropped: Vec<DroppedInputEpoch> = Vec::new();
    let mut steps: Vec<i128> = Vec::new();
    for (idx, (source, ticks)) in sources.iter().zip(&source_ticks).enumerate() {
        let facts = product_grid(ticks, source.header.epoch_interval_s);
        if !facts.out_of_order.is_empty() {
            return Err(Error::InvalidInput(
                "merge input epochs must be strictly increasing".into(),
            ));
        }
        let placed = ticks.iter().flatten().count();
        match facts.step {
            Some(step) => steps.push(step),
            None if placed >= 2 => {
                return Err(Error::InvalidInput(format!(
                    "merge input {idx} epochs lie on no grid: its steps differ and are not all whole multiples of its declared {} s interval",
                    source.header.epoch_interval_s
                )))
            }
            None => {}
        }
        for &epoch_index in &facts.unplaced {
            dropped.push(DroppedInputEpoch {
                source: idx,
                epoch_index,
                epoch: source.epochs[epoch_index],
                reason: DroppedEpochReason::NotOnTickAxis,
            });
        }
    }

    let epoch_index: Vec<BTreeMap<i128, usize>> = source_ticks
        .iter()
        .map(|ticks| {
            ticks
                .iter()
                .enumerate()
                .filter_map(|(index, tick)| tick.map(|tick| (tick, index)))
                .collect()
        })
        .collect();
    let first_ticks: Vec<Option<i128>> = source_ticks
        .iter()
        .map(|ticks| ticks.iter().flatten().copied().min())
        .collect();
    let Some(anchor) = first_ticks.iter().flatten().copied().min() else {
        return Err(Error::InvalidInput(
            "merge inputs have no epochs on the requested time grid".into(),
        ));
    };

    let step_ticks = match opts.target_epoch_interval_s {
        Some(target) => {
            if !target.is_finite() || target <= 0.0 {
                return Err(Error::InvalidInput(format!(
                    "merge target epoch interval must be positive and finite, got {target}"
                )));
            }
            interval_ticks(target).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "merge target epoch interval {target} s is not a whole number of the 10-nanosecond ticks an SP3 interval states"
                ))
            })?
        }
        None => {
            let mut step = steps.iter().fold(0, |acc, &step| gcd(acc, step));
            for ticks in &source_ticks {
                for tick in ticks.iter().flatten() {
                    step = gcd(step, tick - anchor);
                }
            }
            if step == 0 {
                return Err(Error::InvalidInput(
                    "merge inputs hold one instant and declare no positive epoch interval".into(),
                ));
            }
            step
        }
    };

    // Union of epochs by exact tick, retaining the representative Instant from
    // the earliest-listed source on duplicate ticks. This is what lets a dense
    // source fill cells absent from a sparse preferred source.
    let mut epoch_keys: BTreeMap<i128, Instant> = BTreeMap::new();
    for (idx, (source, ticks)) in sources.iter().zip(&source_ticks).enumerate() {
        for (epoch_index, tick) in ticks.iter().enumerate() {
            let Some(tick) = *tick else {
                continue;
            };
            if (tick - anchor).rem_euclid(step_ticks) == 0 {
                epoch_keys.entry(tick).or_insert(source.epochs[epoch_index]);
            } else {
                dropped.push(DroppedInputEpoch {
                    source: idx,
                    epoch_index,
                    epoch: source.epochs[epoch_index],
                    reason: DroppedEpochReason::OffTargetGrid,
                });
            }
        }
    }
    dropped.sort_by_key(|entry| (entry.source, entry.epoch_index));

    // Per-source per-epoch clock-datum offset relative to source 0. Source 0 is
    // the datum, so its offset is identically zero.
    let clock_offset: Vec<BTreeMap<i128, f64>> = sources
        .iter()
        .enumerate()
        .map(|(idx, source)| {
            if idx == 0 {
                BTreeMap::new()
            } else {
                clock_offsets_by_tick(
                    &sources[0],
                    &source_ticks[0],
                    source,
                    &source_ticks[idx],
                    opts.clock_min_common,
                )
                .into_iter()
                .map(|(tick, offset)| (tick, offset.offset_s))
                .collect()
            }
        })
        .collect();

    Ok(MergeTiming {
        epoch_index,
        first_ticks,
        step_ticks,
        clock_offset,
        epoch_keys,
        dropped,
    })
}

/// Run the continuity check over the merged product and attribute each
/// violation to the contributors whose records it rests on.
///
/// The check runs on the product the merge actually emitted, so it verifies the
/// output rather than re-deriving it from the inputs. Attribution reads the
/// selections recorded while the merge decided, so a splice names the real
/// contributors rather than a guess reconstructed afterwards. A hold-out
/// residual is attributed to the held-out cell and to every node its
/// prediction used, which the continuity check reports from the interpolator's
/// own node selection.
fn verify_merged_continuity(
    merged: &Sp3,
    options: &ContinuityOptions,
    selection: &BTreeMap<(GnssSatelliteId, i64), CellSelection>,
) -> MergeContinuityReport {
    let report = check_continuity(&merged.precise_ephemeris_samples(), options);

    let violations = report
        .defects
        .iter()
        .map(|defect| {
            let sat = defect.satellite();
            let selection_at = |epoch_j2000_s: f64| selection.get(&(sat, epoch_j2000_s as i64));
            let (from_epoch, to_epoch) = defect_epoch_pair(defect);
            let from_sources = from_epoch
                .and_then(|epoch| selection.get(&(sat, epoch)))
                .map(written_sources)
                .unwrap_or_default();
            let to_sources = to_epoch
                .and_then(|epoch| selection.get(&(sat, epoch)))
                .map(written_sources)
                .unwrap_or_default();

            let mut cells: Vec<MergeContinuityCell> = defect_cells(defect)
                .into_iter()
                .map(|(epoch_j2000_s, role)| MergeContinuityCell {
                    epoch_j2000_s,
                    role,
                    selection: selection_at(epoch_j2000_s).cloned(),
                })
                .collect();
            cells.sort_by(|a, b| a.epoch_j2000_s.total_cmp(&b.epoch_j2000_s));

            let written: Vec<Vec<usize>> = cells
                .iter()
                .filter_map(|cell| cell.selection.as_ref().map(written_sources))
                .collect();
            let crosses_contributors = written.windows(2).any(|pair| pair[0] != pair[1]);
            let sources: Vec<usize> = written
                .iter()
                .flatten()
                .copied()
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect();

            MergeContinuityViolation {
                defect: defect.clone(),
                from_sources,
                to_sources,
                cells,
                sources,
                crosses_contributors,
            }
        })
        .collect();

    MergeContinuityReport {
        report,
        violations,
        nodes: InterpolationNodes::for_sp3(merged),
    }
}

/// The sources whose values a cell's written value came from: the selected
/// source where one supplied it, otherwise every member of the combination.
fn written_sources(selection: &CellSelection) -> Vec<usize> {
    selection
        .selected_source()
        .map_or_else(|| selection.members(), |source| vec![source])
}

/// Every record a defect rests on, with the part it plays.
fn defect_cells(defect: &ContinuityDefect) -> Vec<(f64, MergeContinuityCellRole)> {
    match defect {
        ContinuityDefect::HoldOutResidual {
            epoch_j2000_s,
            node_epochs_j2000_s,
            ..
        } => core::iter::once((*epoch_j2000_s, MergeContinuityCellRole::HeldOut))
            .chain(
                node_epochs_j2000_s
                    .iter()
                    .map(|&node| (node, MergeContinuityCellRole::InterpolationNode)),
            )
            .collect(),
        ContinuityDefect::SpeedBound {
            from_j2000_s,
            to_j2000_s,
            ..
        } => vec![
            (*from_j2000_s, MergeContinuityCellRole::PairEnd),
            (*to_j2000_s, MergeContinuityCellRole::PairEnd),
        ],
        ContinuityDefect::DuplicateEpoch { epoch_j2000_s, .. } => {
            vec![(*epoch_j2000_s, MergeContinuityCellRole::RepeatedEpoch)]
        }
        ContinuityDefect::SingleSampleSeries { .. } => Vec::new(),
    }
}

/// The epoch pair a defect brackets, as integer epoch keys.
fn defect_epoch_pair(defect: &ContinuityDefect) -> (Option<i64>, Option<i64>) {
    match defect {
        ContinuityDefect::SpeedBound {
            from_j2000_s,
            to_j2000_s,
            ..
        } => (Some(*from_j2000_s as i64), Some(*to_j2000_s as i64)),
        ContinuityDefect::HoldOutResidual {
            preceding_j2000_s,
            epoch_j2000_s,
            ..
        } => (Some(*preceding_j2000_s as i64), Some(*epoch_j2000_s as i64)),
        ContinuityDefect::DuplicateEpoch { epoch_j2000_s, .. } => {
            (Some(*epoch_j2000_s as i64), Some(*epoch_j2000_s as i64))
        }
        ContinuityDefect::SingleSampleSeries { .. } => (None, None),
    }
}

fn reconcile_sp3_coordinate_labels(
    sources: &[Sp3],
    opts: &MergeOptions,
) -> Result<(Vec<Sp3>, Vec<Sp3FrameReconciliation>)> {
    let target_label = normalized_sp3_frame_label(&sources[0].header.coordinate_system);
    let mut prepared = sources.to_vec();
    let mut report = Vec::new();

    for idx in 1..sources.len() {
        let source_label = normalized_sp3_frame_label(&sources[idx].header.coordinate_system);
        if source_label == target_label {
            continue;
        }

        if let Some(asserted) = asserted_frame_label_set(
            &source_label,
            &target_label,
            &opts.frame_reconciliation.asserted_equivalent_label_sets,
        ) {
            prepared[idx].header.coordinate_system = target_label.clone();
            report.push(Sp3FrameReconciliation {
                source_index: idx,
                source_label,
                target_label: target_label.clone(),
                method: Sp3FrameReconciliationMethod::AssertedEquivalence,
                asserted_label_set: Some(asserted),
                source_frame: None,
                target_frame: None,
                catalog_source_frame: None,
                catalog_target_frame: None,
                catalog_inverse: false,
                reference_epoch_year: None,
                parameters: None,
                rates: None,
                provenance: None,
                epoch_year_span: None,
                records_affected: count_position_records(&sources[idx]),
                identity: true,
            });
            continue;
        }

        if opts.frame_reconciliation.helmert {
            let from = sp3_coordinate_label_frame(&source_label).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "merge inputs have mismatched coordinate systems ({:?} vs {:?}); source label {:?} is not a known ITRF/IGS realization",
                    sources[0].header.coordinate_system,
                    sources[idx].header.coordinate_system,
                    sources[idx].header.coordinate_system
                ))
            })?;
            let to = sp3_coordinate_label_frame(&target_label).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "merge inputs have mismatched coordinate systems ({:?} vs {:?}); target label {:?} is not a known ITRF/IGS realization",
                    sources[0].header.coordinate_system,
                    sources[idx].header.coordinate_system,
                    sources[0].header.coordinate_system
                ))
            })?;

            let transform_report = reconcile_source_by_helmert(
                &mut prepared[idx],
                idx,
                source_label,
                target_label.clone(),
                from,
                to,
            )?;
            report.push(transform_report);
            continue;
        }

        return Err(Error::InvalidInput(format!(
            "merge inputs have mismatched coordinate systems ({:?} vs {:?})",
            sources[0].header.coordinate_system, sources[idx].header.coordinate_system
        )));
    }

    Ok((prepared, report))
}

fn asserted_frame_label_set(
    source_label: &str,
    target_label: &str,
    label_sets: &[Sp3FrameLabelSet],
) -> Option<Vec<String>> {
    label_sets.iter().find_map(|set| {
        if set.labels.contains(source_label) && set.labels.contains(target_label) {
            Some(set.labels.iter().cloned().collect())
        } else {
            None
        }
    })
}

fn reconcile_source_by_helmert(
    source: &mut Sp3,
    source_index: usize,
    source_label: String,
    target_label: String,
    from: TerrestrialFrame,
    to: TerrestrialFrame,
) -> Result<Sp3FrameReconciliation> {
    let records_affected = count_position_records(source);
    let epoch_year_span = epoch_year_span(source);
    let identity = from == to;

    if !identity {
        transform_sp3_positions(source, from, to)?;
    }
    source.header.coordinate_system = target_label.clone();

    let published = published_transform_for_report(from, to);
    Ok(Sp3FrameReconciliation {
        source_index,
        source_label,
        target_label,
        method: Sp3FrameReconciliationMethod::Helmert,
        asserted_label_set: None,
        source_frame: Some(from),
        target_frame: Some(to),
        catalog_source_frame: published.map(|published| published.entry.from),
        catalog_target_frame: published.map(|published| published.entry.to),
        catalog_inverse: published.is_some_and(|published| published.inverse),
        reference_epoch_year: published.map(|published| published.entry.reference_epoch_year),
        parameters: published.map(|published| published.entry.parameters),
        rates: published.map(|published| published.entry.rates),
        provenance: published.map(|published| published.entry.provenance.to_string()),
        epoch_year_span,
        records_affected,
        identity,
    })
}

fn transform_sp3_positions(
    source: &mut Sp3,
    from: TerrestrialFrame,
    to: TerrestrialFrame,
) -> Result<()> {
    let seconds_per_julian_year = DAYS_PER_JULIAN_YEAR * SECONDS_PER_DAY;
    for epoch_idx in 0..source.epochs.len() {
        let epoch_year = decimal_year(source.epochs[epoch_idx]);
        let states = &mut source.states[epoch_idx];
        let raw_nodes = &mut source.interp_raw[epoch_idx];
        for (sat, state) in states.iter_mut() {
            let position = TerrestrialPositionM::from_itrf(state.position);
            let velocity = state
                .velocity
                .map(|velocity| {
                    let [vx, vy, vz] = velocity.as_array();
                    TerrestrialVelocityMPerYear::new(
                        vx * seconds_per_julian_year,
                        vy * seconds_per_julian_year,
                        vz * seconds_per_julian_year,
                    )
                })
                .transpose()
                .map_err(|error| Error::InvalidInput(error.to_string()))?;
            let transformed = frame_catalog::transform(position, velocity, from, to, epoch_year)
                .map_err(|error| Error::InvalidInput(error.to_string()))?;
            let [x, y, z] = transformed.position.as_array();
            state.position = ItrfPositionM::new(x, y, z)
                .map_err(|error| Error::InvalidInput(error.to_string()))?;
            state.velocity = transformed
                .velocity
                .map(|velocity| {
                    let [vx, vy, vz] = velocity.as_array();
                    ItrfVelocityMS::new(
                        vx / seconds_per_julian_year,
                        vy / seconds_per_julian_year,
                        vz / seconds_per_julian_year,
                    )
                })
                .transpose()
                .map_err(|error| Error::InvalidInput(error.to_string()))?;
            if let Some(raw) = raw_nodes.get_mut(sat) {
                raw.km = [x / KM_TO_M, y / KM_TO_M, z / KM_TO_M];
            }
        }
    }
    Ok(())
}

fn count_position_records(source: &Sp3) -> usize {
    source.states.iter().map(BTreeMap::len).sum()
}

fn epoch_year_span(source: &Sp3) -> Option<[f64; 2]> {
    let first = source.epochs.first().copied().map(decimal_year)?;
    let last = source.epochs.last().copied().map(decimal_year)?;
    Some([first, last])
}

fn decimal_year(epoch: Instant) -> f64 {
    let jd_midnight = julian_date_from_instant(epoch) + 0.5;
    let (year, _, _) = civil_from_julian_day_number(jd_midnight.floor() as i64);
    let days = if is_leap_year(year) { 366.0 } else { 365.0 };
    year as f64 + (fractional_day_of_year_from_instant(epoch) - 1.0) / days
}

fn normalized_sp3_frame_label(label: &str) -> String {
    label.trim().to_string()
}

fn sp3_coordinate_label_frame(label: &str) -> Option<TerrestrialFrame> {
    match label.trim() {
        "ITRF2020" | "ITRF20" | "IGS20" | "IGc20" => Some(TerrestrialFrame::Itrf2020),
        "ITRF2014" | "ITRF14" | "IGS14" | "IGb14" => Some(TerrestrialFrame::Itrf2014),
        "ITRF2008" | "ITRF08" | "IGS08" | "IGb08" => Some(TerrestrialFrame::Itrf2008),
        _ => None,
    }
}

fn published_transform_for_report(
    from: TerrestrialFrame,
    to: TerrestrialFrame,
) -> Option<PublishedTransformForReport> {
    frame_catalog::catalog_entry(from, to)
        .map(|entry| PublishedTransformForReport {
            entry,
            inverse: false,
        })
        .or_else(|| {
            frame_catalog::catalog_entry(to, from).map(|entry| PublishedTransformForReport {
                entry,
                inverse: true,
            })
        })
}

#[derive(Debug, Clone, Copy)]
struct PublishedTransformForReport {
    entry: &'static frame_catalog::HelmertTransform,
    inverse: bool,
}

#[derive(Debug, Clone, Copy)]
struct FirstEpochHeaderFields {
    gnss_week: u32,
    seconds_of_week: f64,
    mjd: u32,
    mjd_fraction: f64,
}

/// The line-2 fields for a first epoch `tick` ticks from the J2000 origin in
/// the product's own scale, each the value its field states and the reader
/// holds for a real file.
///
/// The GPS week and seconds of week count from the GPS epoch (1980-01-06
/// 00:00:00) in whole ticks, so the seconds of week is exact in its eight
/// decimals. The MJD is the civil day's, and the MJD fraction is the within-day
/// tick count as a fraction of a day rounded once, half to even, to the
/// thirteen decimals its field states; a fraction that rounds to a whole day
/// is carried into the next MJD. Both come from the exact count, so no
/// intermediate `f64` rounding reaches a field.
fn first_epoch_header_fields(tick: i128) -> Option<FirstEpochHeaderFields> {
    const TICKS_PER_DAY: i128 = 86_400 * TICKS_PER_SECOND;
    const TICKS_PER_WEEK: i128 = 7 * TICKS_PER_DAY;
    const J2000_NOON_TICKS: i128 = TICKS_PER_DAY / 2;
    // MJD 51544 is the civil day of the J2000 origin (MJD 51544.5).
    const J2000_MJD_DAY: i128 = 51_544;
    const FRACTION_UNITS: i128 = 10_000_000_000_000;

    let from_midnight = tick.checked_add(J2000_NOON_TICKS)?;
    let mut day = J2000_MJD_DAY + from_midnight.div_euclid(TICKS_PER_DAY);
    let within_day = from_midnight.rem_euclid(TICKS_PER_DAY);
    let scaled = within_day.checked_mul(FRACTION_UNITS)?;
    let mut units = scaled / TICKS_PER_DAY;
    let remainder = scaled % TICKS_PER_DAY;
    if remainder * 2 > TICKS_PER_DAY || (remainder * 2 == TICKS_PER_DAY && units % 2 == 1) {
        units += 1;
    }
    if units == FRACTION_UNITS {
        day += 1;
        units = 0;
    }
    let mjd = u32::try_from(day).ok()?;
    let mjd_fraction = units as f64 / FRACTION_UNITS as f64;

    let gps_ticks = tick.checked_add(GPS_EPOCH_TO_J2000_S as i128 * TICKS_PER_SECOND)?;
    let gnss_week = u32::try_from(gps_ticks.div_euclid(TICKS_PER_WEEK)).ok()?;
    let seconds_of_week = interval_seconds(gps_ticks.rem_euclid(TICKS_PER_WEEK));

    Some(FirstEpochHeaderFields {
        gnss_week,
        seconds_of_week,
        mjd,
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

/// Datum offset at `key`, using an exact estimate when available or linear
/// interpolation between the nearest bracketing estimates. Never extrapolates
/// beyond the observed offset interval.
fn clock_offset_at(offsets: &BTreeMap<i128, f64>, key: i128) -> Option<f64> {
    if let Some(offset) = offsets.get(&key) {
        return Some(*offset);
    }
    let (&before_key, &before) = offsets.range(..key).next_back()?;
    let (&after_key, &after) = offsets.range(key..).next()?;
    if after_key <= before_key {
        return None;
    }
    let fraction = (key - before_key) as f64 / (after_key - before_key) as f64;
    Some(before + fraction * (after - before))
}

fn source_clock_raw_at(
    source: &Sp3,
    ei: usize,
    sat: &GnssSatelliteId,
) -> Option<(f64, Option<f64>)> {
    if let Ok(clock_records) = source.clock_records_at(ei) {
        if let Some(clock_rec) = clock_records.get(sat) {
            return Some((clock_rec.clock_s, Some(clock_rec.clock_us)));
        }
    }

    if let Ok(states) = source.states_at(ei) {
        if let Some(state) = states.get(sat) {
            if let Some(state_clock_s) = state.clock_s {
                let raw_us = source
                    .interp_raw
                    .get(ei)
                    .and_then(|map| map.get(sat))
                    .and_then(|node| node.clock_us);
                return Some((state_clock_s, raw_us));
            }
        }
    }

    None
}

fn source_has_zero_applied_offset(
    clock_offset: &[BTreeMap<i128, f64>],
    source_idx: usize,
    key: i128,
) -> bool {
    let offset = if source_idx == 0 {
        Some(0.0)
    } else {
        clock_offset
            .get(source_idx)
            .and_then(|offsets| clock_offset_at(offsets, key))
    };
    offset.is_some_and(|off| off.to_bits() == 0.0_f64.to_bits())
}

/// Retrieve the unperturbed raw `clock_us` value from the selected source(s) for
/// a single-source, precedence-selected, or uniformly agreeing combined clock
/// when the applied datum offset is exactly zero and the selected seconds value
/// is unchanged.
///
/// Returns:
/// - `Some(Some(raw_us))` when a raw value exists in the source and was preserved.
/// - `Some(None)` when the selected source explicitly lacked a raw node value
///   (no raw value should be fabricated).
/// - `None` when preservation is not eligible (e.g. diverging combined clock,
///   non-zero applied offset, missing raw values, or modified clock seconds).
fn preserved_source_raw_clock_us(
    sources: &[Sp3],
    epoch_index: &[BTreeMap<i128, usize>],
    key: i128,
    sat: &GnssSatelliteId,
    clock_selection: Option<&CellSelection>,
    clock_offset: &[BTreeMap<i128, f64>],
    selected_s: f64,
) -> Option<Option<f64>> {
    match clock_selection {
        Some(CellSelection::SingleSource { source })
        | Some(CellSelection::Precedence { source, .. }) => {
            if !source_has_zero_applied_offset(clock_offset, *source, key) {
                return None;
            }
            let &ei = epoch_index.get(*source).and_then(|idx| idx.get(&key))?;
            let src = sources.get(*source)?;
            let (src_s, raw_us) = source_clock_raw_at(src, ei, sat)?;
            if selected_s.to_bits() != src_s.to_bits() {
                return None;
            }
            Some(raw_us)
        }
        Some(CellSelection::Combined { members, .. }) => {
            if members.is_empty() {
                return None;
            }
            let mut common_raw_bits: Option<u64> = None;
            let mut preserved_raw_us = 0.0;

            for &member in members {
                if !source_has_zero_applied_offset(clock_offset, member, key) {
                    return None;
                }
                let &ei = epoch_index.get(member).and_then(|idx| idx.get(&key))?;
                let src = sources.get(member)?;
                let (src_s, raw_us_opt) = source_clock_raw_at(src, ei, sat)?;
                if selected_s.to_bits() != src_s.to_bits() {
                    return None;
                }
                let raw_us = raw_us_opt?;
                let raw_bits = raw_us.to_bits();
                if let Some(expected_bits) = common_raw_bits {
                    if raw_bits != expected_bits {
                        return None;
                    }
                } else {
                    common_raw_bits = Some(raw_bits);
                    preserved_raw_us = raw_us;
                }
            }

            Some(Some(preserved_raw_us))
        }
        _ => None,
    }
}

fn precedence_sources_for_satellites(
    sources: &[Sp3],
    epoch_index: &[BTreeMap<i128, usize>],
    epoch_keys: &BTreeMap<i128, Instant>,
    systems: Option<&BTreeSet<GnssSystem>>,
) -> BTreeMap<GnssSatelliteId, usize> {
    let mut by_sat = BTreeMap::new();

    for (idx, source) in sources.iter().enumerate() {
        for key in epoch_keys.keys() {
            let Some(&epoch_idx) = epoch_index[idx].get(key) else {
                continue;
            };
            if let Ok(states) = source.states_at(epoch_idx) {
                for sat in states.keys() {
                    if systems.is_none_or(|allowed| allowed.contains(&sat.system)) {
                        by_sat.entry(*sat).or_insert(idx);
                    }
                }
            }
            if let Ok(clock_records) = source.clock_records_at(epoch_idx) {
                for sat in clock_records.keys() {
                    if systems.is_none_or(|allowed| allowed.contains(&sat.system)) {
                        by_sat.entry(*sat).or_insert(idx);
                    }
                }
            }
        }
    }

    by_sat
}

fn validate_merge_options(opts: &MergeOptions) -> Result<()> {
    validate::finite_nonneg(opts.position_tolerance_m, "merge position tolerance meters")
        .map_err(|error| Error::InvalidInput(error.to_string()))?;
    validate::finite_nonneg(opts.clock_tolerance_s, "merge clock tolerance seconds")
        .map_err(|error| Error::InvalidInput(error.to_string()))?;
    if opts.min_agree == 0 {
        return Err(Error::InvalidInput(
            "merge minimum agreement must be at least one".into(),
        ));
    }
    if opts.clock_min_common == 0 {
        return Err(Error::InvalidInput(
            "merge minimum common clock satellites must be at least one".into(),
        ));
    }
    if let Some(reject) = opts.outlier_reject {
        validate::finite_nonneg(
            reject.position_tolerance_m,
            "merge outlier position tolerance meters",
        )
        .map_err(|error| Error::InvalidInput(error.to_string()))?;
        validate::finite_nonneg(
            reject.clock_tolerance_s,
            "merge outlier clock tolerance seconds",
        )
        .map_err(|error| Error::InvalidInput(error.to_string()))?;
    }
    if opts
        .systems
        .as_ref()
        .is_some_and(|systems| systems.is_empty())
    {
        return Err(Error::InvalidInput(
            "merge systems filter must not be empty".into(),
        ));
    }
    for labels in &opts.frame_reconciliation.asserted_equivalent_label_sets {
        if labels.labels.len() < 2 || labels.labels.iter().any(|label| label.trim().is_empty()) {
            return Err(Error::InvalidInput(
                "merge asserted frame label sets require at least two non-empty labels".into(),
            ));
        }
    }
    Ok(())
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
            // invariant: combine_axis is called only with a non-empty consensus
            // cluster selected by the preceding merge stage.
            #[allow(clippy::expect_used)]
            median(&mut vals).expect("consensus cluster is non-empty")
        }
        MergeCombine::Precedence => {
            let preferred = members.iter().min_by_key(|(s, _)| *s).map(|(_, v)| *v);
            // invariant: combine_axis is called only with a non-empty consensus
            // cluster selected by the preceding merge stage.
            #[allow(clippy::expect_used)]
            preferred.expect("consensus cluster is non-empty")
        }
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
    let other_ticks = product_ticks(other);
    let offsets: BTreeMap<i128, f64> = clock_offsets_by_tick(
        reference,
        &product_ticks(reference),
        other,
        &other_ticks,
        min_common,
    )
    .into_iter()
    .map(|(tick, offset)| (tick, offset.offset_s))
    .collect();

    let mut aligned = other.clone();
    for ei in 0..aligned.epochs.len() {
        let Some(tick) = other_ticks.get(ei).copied().flatten() else {
            continue;
        };
        let Some(&off) = offsets.get(&tick) else {
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
        for clock_rec in aligned.clock_records[ei].values_mut() {
            clock_rec.clock_s -= off;
            clock_rec.clock_us -= off * 1.0e6;
        }
    }
    aligned
}

/// One cell's decision, as the merge made it.
struct RecordCellProvenance<'a> {
    epoch: Instant,
    sat: GnssSatelliteId,
    position: Option<&'a CellSelection>,
    clock: Option<&'a CellSelection>,
    /// Every source that offered a position or clock for this cell, whether or not it
    /// survived into the consensus. Distinguishing "did not offer" from
    /// "offered and was rejected" is the whole difference between a transition
    /// caused by availability and one caused by outlier rejection, and the
    /// accepted member set alone cannot tell them apart.
    candidates: &'a [usize],
    mode: ProvenanceMode,
}

/// Running provenance state threaded through the epoch loop.
struct ProvenanceAccumulator<'a> {
    cells: &'a mut Vec<CellProvenance>,
    transitions: &'a mut Vec<PrecedenceTransition>,
    contributed: &'a mut [usize],
    selected: &'a mut [usize],
    first: &'a mut [Option<Instant>],
    last: &'a mut [Option<Instant>],
    accepted_cells: &'a mut usize,
    previous: &'a mut BTreeMap<GnssSatelliteId, CellSelection>,
}

/// Record one accepted cell: its selection, any transition it represents, and
/// its effect on per-contributor coverage.
///
/// Called only from the point in [`merge`] where a cell is known to have been
/// accepted and written, so the record is an attestation of the decision rather
/// than a later reconstruction of it.
fn record_cell_provenance(cell: RecordCellProvenance<'_>, acc: &mut ProvenanceAccumulator<'_>) {
    *acc.accepted_cells += 1;

    let selection = cell.position.or(cell.clock);

    let mut contributing_sources = Vec::new();
    if let Some(pos) = cell.position {
        contributing_sources.extend(pos.members());
    }
    if let Some(clk) = cell.clock {
        contributing_sources.extend(clk.members());
    }
    contributing_sources.sort_unstable();
    contributing_sources.dedup();

    for source in contributing_sources {
        acc.contributed[source] += 1;
        if acc.first[source].is_none() {
            acc.first[source] = Some(cell.epoch);
        }
        acc.last[source] = Some(cell.epoch);
    }

    if let Some(pos) = cell.position {
        if let Some(source) = pos.selected_source() {
            acc.selected[source] += 1;
        }
    } else if let Some(clk) = cell.clock {
        if let Some(source) = clk.selected_source() {
            acc.selected[source] += 1;
        }
    }

    if let Some(sel) = selection {
        if let Some(transition) = transition_between(
            cell.sat,
            cell.epoch,
            acc.previous.get(&cell.sat),
            sel,
            cell.candidates,
        ) {
            acc.transitions.push(transition);
        }
        acc.previous.insert(cell.sat, sel.clone());
    }

    if cell.mode == ProvenanceMode::Full {
        acc.cells.push(CellProvenance {
            epoch: cell.epoch,
            satellite: cell.sat,
            position: cell.position.cloned(),
            clock: cell.clock.cloned(),
        });
    }
}

/// The transition, if any, between a satellite's previous accepted cell and this
/// one.
///
/// A satellite's first accepted cell is a transition from `None`: a consumer
/// reading the transition list as a timeline needs the arc's opening entry, not
/// an implicit one it has to infer.
fn transition_between(
    sat: GnssSatelliteId,
    epoch: Instant,
    previous: Option<&CellSelection>,
    current: &CellSelection,
    candidates: &[usize],
) -> Option<PrecedenceTransition> {
    let Some(previous) = previous else {
        return Some(PrecedenceTransition {
            satellite: sat,
            epoch,
            from_source: None,
            to_source: current.selected_source(),
            reason: TransitionReason::SoleAvailability,
        });
    };

    let from = previous.selected_source();
    let to = current.selected_source();
    if from == to && std::mem::discriminant(previous) == std::mem::discriminant(current) {
        return None;
    }

    // Why selection moved. The candidate set separates the two cases the
    // accepted member set cannot: a previous supplier that did not offer this
    // cell at all left the product (availability), while one that offered it and
    // did not survive the consensus was rejected (outlier).
    let current_members = current.members();
    let reason = match from {
        Some(from_source) if !candidates.contains(&from_source) => {
            TransitionReason::SoleAvailability
        }
        Some(from_source) if !current_members.contains(&from_source) => {
            TransitionReason::OutlierRejection
        }
        Some(_) if std::mem::discriminant(previous) != std::mem::discriminant(current) => {
            TransitionReason::ConsensusChange
        }
        Some(_) => TransitionReason::Precedence,
        None => TransitionReason::ConsensusChange,
    };

    Some(PrecedenceTransition {
        satellite: sat,
        epoch,
        from_source: from,
        to_source: to,
        reason,
    })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::super::Sp3;
    use super::{
        align_clock_reference, clock_reference_offset, merge, CellSelection, ClockOmission,
        ClockOmissionReason, MergeCombine, MergeFlag, MergeOptions, MergePrecedenceScope,
        MergeReport, OutlierRejectOptions, ProvenanceMode, Sp3FrameLabelSet,
        Sp3FrameReconciliationMethod, Sp3FrameReconciliationOptions, TransitionReason,
    };
    use crate::constants::SECONDS_PER_DAY;
    use crate::error::Error;
    use crate::frame::ItrfPositionM;
    use crate::id::{GnssSatelliteId, GnssSystem};
    use crate::sp3::Sp3DataType;
    use sha2::{Digest, Sha256};
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

    fn sha256_hex(bytes: &[u8]) -> String {
        format!("{:x}", Sha256::digest(bytes))
    }

    #[test]
    fn frozen_merge_output_hashes() {
        let union_a = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
            ("G03", [17000.0, -22000.0, 7000.0], Some(300.0)),
        ]);
        let union_b = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
        ]);

        let mean_a = sp3_records(&[("G01", [15000.000, -20000.0, 5000.0], Some(100.0))]);
        let mean_b = sp3_records(&[("G01", [15000.003, -20000.0, 5000.0], Some(100.0))]);
        let mean_c = sp3_records(&[("G01", [15000.006, -20000.0, 5000.0], Some(100.0))]);
        let mean_options = MergeOptions {
            position_tolerance_m: 10.0,
            min_agree: 3,
            combine: MergeCombine::Mean,
            ..MergeOptions::default()
        };

        let preferred = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))]);
        let agreeing = sp3_records(&[("G01", [15000.001, -20000.0, 5000.0], Some(100.0))]);
        let outlier = sp3_records(&[("G01", [15002.0, -20000.0, 5000.0], Some(100.0))]);
        let guarded_precedence_options = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 2,
            outlier_reject: Some(OutlierRejectOptions {
                position_tolerance_m: 2.0,
                clock_tolerance_s: 1.0e-6,
            }),
            ..MergeOptions::default()
        };

        let mixed_a = sp3_epochs(
            0.0,
            &[
                &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))],
                &[("G01", [15001.0, -20001.0, 5001.0], Some(101.0))],
                &[("G01", [15002.0, -20002.0, 5002.0], Some(102.0))],
            ],
            900.0,
            "IGS14",
        );
        let mixed_b = sp3_epochs(
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
        let mixed_options = MergeOptions {
            min_agree: 1,
            ..MergeOptions::default()
        };

        let cases = [
            (
                "union coverage",
                vec![union_a, union_b],
                MergeOptions::default(),
                "6462f7f385f8141cc728c152a0c90276cae3727f3031482a95781ea6665f8d28",
            ),
            (
                "mean consensus",
                vec![mean_a, mean_b, mean_c],
                mean_options,
                "9fd9021f4262ea0303984974fa26f4fdb132bc7ffec4ac06ecf1e2a735eafe40",
            ),
            (
                "guarded precedence",
                vec![preferred, agreeing, outlier],
                guarded_precedence_options,
                "e3f498d13e3fadbf3d3e8b029417a8f45ca4efd19778e7b0bcef2218f281f960",
            ),
            (
                "mixed cadence union",
                vec![mixed_a, mixed_b],
                mixed_options,
                "0923299915c137218a2ab5c8648425a5952f42d8e06734e5d3caa5beae1b5c1d",
            ),
        ];

        // Each synthetic case is still an exact-byte comparison, but the four
        // comparisons are made together below rather than one at a time inside
        // the loop. Asserting per case stopped the loop at the first stale
        // golden, so a regeneration only ever learned one hash per run and the
        // remaining cases had not been merged at all; collecting them first
        // means one failure reports every case's actual hash beside the value
        // frozen for it.
        let mut frozen_hashes: Vec<(&str, String, &str)> = Vec::new();

        for (name, sources, options, expected) in cases {
            let (merged, _) = merge(&sources, &options).expect("frozen merge case");
            let text = merged.to_sp3_string().expect("serialize SP3 product");
            // The bytes under these hashes now carry header descriptors the
            // reader retains and the writer states back, where it used to write
            // constants of its own: the data-used field, the `%c` file type and
            // the two `%f` bases all come from the base source, and the agency
            // sits in the four columns the layout gives it rather than starting
            // one column early. Pinning those columns here is what makes a hash
            // regeneration reviewable - a new hash has to be this content.
            let lines: Vec<&str> = text.lines().collect();
            assert_eq!(&lines[0][40..45], "ORBIT", "{name}: data used");
            assert_eq!(&lines[0][56..60], " TST", "{name}: agency columns 57-60");
            let pc = lines
                .iter()
                .find(|line| line.starts_with("%c"))
                .expect("first %c line");
            assert_eq!(&pc[3..5], "G ", "{name}: %c file type");
            let pf = lines
                .iter()
                .find(|line| line.starts_with("%f"))
                .expect("first %f line");
            assert_eq!(&pf[3..13], " 1.2500000", "{name}: %f pos/vel base");
            assert_eq!(&pf[14..26], " 1.025000000", "{name}: %f clock/rate base");

            frozen_hashes.push((name, sha256_hex(text.as_bytes()), expected));
        }

        let actual_hashes: Vec<(&str, &str)> = frozen_hashes
            .iter()
            .map(|(name, actual, _)| (*name, actual.as_str()))
            .collect();
        let expected_hashes: Vec<(&str, &str)> = frozen_hashes
            .iter()
            .map(|(name, _, expected)| (*name, *expected))
            .collect();
        assert_eq!(
            actual_hashes, expected_hashes,
            "frozen synthetic merge output hashes"
        );

        #[cfg(sidereon_repo_tests)]
        {
            fn load(name: &str) -> Sp3 {
                let path = format!("{}/tests/fixtures/sp3/{name}", env!("CARGO_MANIFEST_DIR"));
                let bytes = std::fs::read(&path).unwrap_or_else(|error| panic!("{path}: {error}"));
                Sp3::parse(&bytes).unwrap_or_else(|error| panic!("{name}: {error}"))
            }

            let sources = [
                load("COD0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3"),
                load("GFZ0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3"),
                load("JPL0OPSFIN_20261200945_02H30M_15M_ORB_trim.SP3"),
            ];
            let (merged, _) = merge(&sources, &MergeOptions::default()).expect("real frozen merge");

            // This case froze the bytes of a *mean-combined* merge, which meant
            // freezing numbers the format cannot state: the default combine
            // averages the agreeing centers, and the mean of millimetre-
            // resolution positions is not a millimetre. The writer now reports
            // that by name instead of rounding it away, so what is asserted
            // here is the refusal and the value behind it. A byte golden for
            // real data needs a merge whose values SP3 can state - precedence,
            // which keeps each contributor's own numbers.
            let refusal = merged
                .to_sp3_string()
                .expect_err("a mean-combined merge is finer than the record columns");
            let crate::sp3::Sp3WriteError::RecordValueNotRepresentable {
                field,
                sat,
                epoch_index,
                stored,
                column_value,
                ..
            } = refusal
            else {
                panic!("expected a record-value refusal; got {refusal:?}");
            };

            // The refusal names a value the merged product really holds ...
            let (held, scale) = match merged.state(sat, epoch_index) {
                Ok(state) => match field {
                    "position x" => (state.position.x_m, crate::constants::KM_TO_M),
                    "position y" => (state.position.y_m, crate::constants::KM_TO_M),
                    "position z" => (state.position.z_m, crate::constants::KM_TO_M),
                    "clock" => (
                        state.clock_s.expect("a refused clock column holds a clock"),
                        crate::constants::US_TO_S,
                    ),
                    other => panic!("unexpected refused record field {other:?}"),
                },
                Err(_) => {
                    // A cell with no orbit is a clock-only record, which has
                    // only the one column.
                    assert_eq!(field, "clock", "a clock-only record has one column");
                    let record = merged
                        .clock_record(sat, epoch_index)
                        .expect("the refused cell holds a record");
                    (record.clock_s, crate::constants::US_TO_S)
                }
            };
            assert_eq!(
                held.to_bits(),
                stored.to_bits(),
                "the refusal states the value the product holds"
            );
            // ... and a column that really cannot state it: read the column
            // back the way the parser does and the value does not return.
            let read: f64 = format!("{column_value:.6}")
                .parse()
                .expect("a formatted column re-reads as a number");
            assert_ne!(
                (read * scale).to_bits(),
                stored.to_bits(),
                "the column the writer refused would in fact have restated the value"
            );

            // The positive half of the same case. The refusal above is only
            // the right answer if a real three-center merge whose values the
            // format *can* state still serializes - and comes back from the
            // text value for value. Precedence keeps each cell's numbers from
            // the contributor that supplied them, and those came out of an
            // `F14.6` column, so they go back into one.
            let precedence = MergeOptions {
                combine: MergeCombine::Precedence,
                precedence_scope: MergePrecedenceScope::Cell,
                ..MergeOptions::default()
            };
            let (merged, report) = merge(&sources, &precedence).expect("real precedence merge");
            assert!(
                report
                    .agreement
                    .iter()
                    .any(|metric| metric.position_members == 3),
                "this is only a three-center case if some cell was carried by three"
            );

            let text = merged
                .to_sp3_string()
                .expect("a precedence merge holds values the record columns state");
            let reparsed = Sp3::parse(text.as_bytes()).expect("the merged product reparses");

            assert_eq!(
                reparsed.epochs, merged.epochs,
                "every merged epoch comes back as the instant it was, bit for bit"
            );
            assert_eq!(reparsed.epoch_count(), merged.epoch_count());
            assert_eq!(
                reparsed.header.satellites, merged.header.satellites,
                "the merged satellite list survives the round trip"
            );
            for idx in 0..merged.epoch_count() {
                assert_eq!(
                    reparsed.states_at(idx).ok(),
                    merged.states_at(idx).ok(),
                    "merged states differ at epoch {idx} after a write/read cycle"
                );
                assert_eq!(
                    reparsed.clock_records_at(idx).ok(),
                    merged.clock_records_at(idx).ok(),
                    "merged clock-only records differ at epoch {idx} after a write/read cycle"
                );
            }
            // Written text this writer accepted holds only values it accepts,
            // so writing the reparsed product again is the same bytes. No hash
            // is pinned here: the assertions above say what the bytes have to
            // mean, which is what a regenerated golden would have to be checked
            // against anyway.
            assert_eq!(
                reparsed
                    .to_sp3_string()
                    .expect("a product parsed from written SP3 re-encodes"),
                text,
                "serializing the merged product is idempotent"
            );
        }
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
    fn merge_rejects_non_executable_system_and_frame_policies() {
        let source = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))]);

        let empty_systems = MergeOptions {
            systems: Some(BTreeSet::new()),
            ..MergeOptions::default()
        };
        let error = merge(std::slice::from_ref(&source), &empty_systems).unwrap_err();
        assert!(error
            .to_string()
            .contains("systems filter must not be empty"));

        let incomplete_frame_set = MergeOptions {
            frame_reconciliation: Sp3FrameReconciliationOptions {
                asserted_equivalent_label_sets: vec![Sp3FrameLabelSet::new(["IGS20"])],
                helmert: false,
            },
            ..MergeOptions::default()
        };
        let error = merge(&[source], &incomplete_frame_set).unwrap_err();
        assert!(error.to_string().contains("at least two non-empty labels"));
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
    fn guarded_precedence_replaces_a_corrupt_preferred_position() {
        let preferred = sp3_records(&[("G01", [16000.0, -20000.0, 5000.0], None)]);
        let agreeing_a = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], None)]);
        let agreeing_b = sp3_records(&[("G01", [15000.0002, -20000.0, 5000.0], None)]);
        let opts = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 1,
            outlier_reject: Some(OutlierRejectOptions {
                position_tolerance_m: 0.5,
                clock_tolerance_s: 5.0e-9,
            }),
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[preferred, agreeing_a, agreeing_b], &opts).expect("merge");

        let x = merged.states_at(0).expect("epoch")[&gps(1)]
            .position
            .as_array()[0];
        assert_eq!(
            x, 15_000_000.0,
            "earliest member of the 2-source cluster wins"
        );
        assert_eq!(report.position_outliers.len(), 1);
        assert_eq!(report.position_outliers[0].sources, vec![0]);
    }

    #[test]
    fn unguarded_precedence_preserves_the_existing_preferred_value_behavior() {
        let preferred = sp3_records(&[("G01", [16000.0, -20000.0, 5000.0], None)]);
        let agreeing_a = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], None)]);
        let agreeing_b = sp3_records(&[("G01", [15000.0002, -20000.0, 5000.0], None)]);
        let opts = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 1,
            outlier_reject: None,
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[preferred, agreeing_a, agreeing_b], &opts).expect("merge");

        let x = merged.states_at(0).expect("epoch")[&gps(1)]
            .position
            .as_array()[0];
        assert_eq!(x, 16_000_000.0);
        assert_eq!(report.position_outliers[0].sources, vec![1, 2]);
    }

    #[test]
    fn guarded_precedence_keeps_a_preferred_member_of_the_majority() {
        let preferred = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], None)]);
        let agreeing = sp3_records(&[("G01", [15000.0002, -20000.0, 5000.0], None)]);
        let outlier = sp3_records(&[("G01", [16000.0, -20000.0, 5000.0], None)]);
        let opts = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 1,
            outlier_reject: Some(OutlierRejectOptions {
                position_tolerance_m: 0.5,
                clock_tolerance_s: 5.0e-9,
            }),
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[preferred, agreeing, outlier], &opts).expect("merge");

        let x = merged.states_at(0).expect("epoch")[&gps(1)]
            .position
            .as_array()[0];
        assert_eq!(x, 15_000_000.0);
        assert_eq!(report.position_outliers[0].sources, vec![2]);
    }

    #[test]
    fn guarded_precedence_keeps_a_single_source_cell() {
        let only = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], None)]);
        let opts = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 1,
            outlier_reject: Some(OutlierRejectOptions {
                position_tolerance_m: 0.5,
                clock_tolerance_s: 5.0e-9,
            }),
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[only], &opts).expect("merge");

        assert!(merged.states_at(0).expect("epoch").contains_key(&gps(1)));
        assert_eq!(report.single_source.len(), 1);
        assert!(report.quarantined.is_empty());
    }

    #[test]
    fn guarded_precedence_position_tolerance_is_inclusive() {
        for (delta_km, accepted) in [(0.000_499, true), (0.000_501, false)] {
            let a = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], None)]);
            let b = sp3_records(&[("G01", [15000.0 + delta_km, -20000.0, 5000.0], None)]);
            let opts = MergeOptions {
                combine: MergeCombine::Precedence,
                min_agree: 1,
                outlier_reject: Some(OutlierRejectOptions {
                    position_tolerance_m: 0.5,
                    clock_tolerance_s: 5.0e-9,
                }),
                ..MergeOptions::default()
            };

            let (merged, report) = merge(&[a, b], &opts).expect("merge");
            // The lone cell is the lone epoch: refusing it leaves the epoch with
            // nothing to write, so it is omitted and reported.
            assert_eq!(
                merged
                    .states_at(0)
                    .is_ok_and(|states| states.contains_key(&gps(1))),
                accepted,
                "delta {delta_km} km"
            );
            assert_eq!(merged.epochs.is_empty(), !accepted);
            assert_eq!(report.omitted_epochs.len(), usize::from(!accepted));
            assert_eq!(report.quarantined.is_empty(), accepted);
        }

        assert_eq!(
            super::largest_within(&[0.0_f64, 0.5_f64], |a, b| (*a - *b).abs() <= 0.5).len(),
            2,
            "the tolerance boundary itself is accepted"
        );
    }

    #[test]
    fn guarded_precedence_replaces_a_corrupt_preferred_clock() {
        let positions = |clock_g01: f64| {
            sp3_records(&[
                ("G01", [15000.0, -20000.0, 5000.0], Some(clock_g01)),
                ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
                ("G03", [17000.0, -22000.0, 7000.0], Some(300.0)),
                ("G04", [18000.0, -23000.0, 8000.0], Some(400.0)),
                ("G05", [19000.0, -24000.0, 9000.0], Some(500.0)),
            ])
        };
        let opts = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 1,
            outlier_reject: Some(OutlierRejectOptions {
                position_tolerance_m: 0.5,
                clock_tolerance_s: 5.0e-9,
            }),
            ..MergeOptions::default()
        };

        let (merged, report) = merge(
            &[positions(1100.0), positions(100.0), positions(100.0)],
            &opts,
        )
        .expect("merge");

        let clock = merged.states_at(0).expect("epoch")[&gps(1)]
            .clock_s
            .expect("consensus clock");
        assert!((clock - 100.0e-6).abs() < 1.0e-15, "clock {clock}");
        let rejected = report
            .clock_outliers
            .iter()
            .find(|entry| entry.satellite == gps(1))
            .expect("clock outlier provenance");
        assert_eq!(rejected.sources, vec![0]);
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
    fn merge_rejects_nonfinite_consensus_position_without_panicking() {
        let mut a = sp3_records(&[("G01", [15_000.0, -20_000.0, 5_000.0], None)]);
        let mut b = sp3_records(&[("G01", [15_000.0, -20_000.0, 5_000.0], None)]);
        let extreme = ItrfPositionM::new(f64::MAX, 0.0, 0.0).unwrap();
        a.states[0].get_mut(&gps(1)).unwrap().position = extreme;
        b.states[0].get_mut(&gps(1)).unwrap().position = extreme;

        let error = match merge(&[a, b], &MergeOptions::default()) {
            Ok(_) => panic!("non-finite consensus position must be rejected"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("must be finite"));
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
    fn merge_accepts_asserted_equivalent_labels_and_reports_assertion() {
        for (a_label, b_label) in [("IGS14", "ITRF2"), ("ITRF2", "IGS14")] {
            let a = sp3_build(
                &[("G01", [15000.0, -20000.0, 5000.0], Some(100.0), "")],
                a_label,
            );
            let b = sp3_build(
                &[("G02", [16000.0, -21000.0, 6000.0], Some(200.0), "")],
                b_label,
            );
            let opts = MergeOptions {
                frame_reconciliation: super::Sp3FrameReconciliationOptions {
                    asserted_equivalent_label_sets: vec![Sp3FrameLabelSet::pair("IGS14", "ITRF2")],
                    helmert: false,
                },
                ..MergeOptions::default()
            };

            let (merged, report) = merge(&[a, b], &opts).expect("asserted frame merge");

            let states = merged.states_at(0).expect("epoch 0");
            assert!(states.contains_key(&gps(1)));
            assert!(states.contains_key(&gps(2)));
            assert_eq!(merged.header.coordinate_system, a_label);
            assert_eq!(report.frame_reconciliations.len(), 1);
            let reconciliation = &report.frame_reconciliations[0];
            assert_eq!(
                reconciliation.method,
                Sp3FrameReconciliationMethod::AssertedEquivalence
            );
            assert_eq!(reconciliation.source_index, 1);
            assert_eq!(reconciliation.source_label, b_label);
            assert_eq!(reconciliation.target_label, a_label);
            assert_eq!(reconciliation.records_affected, 1);
            assert!(reconciliation.parameters.is_none());
            assert!(reconciliation.rates.is_none());
            assert_eq!(
                reconciliation
                    .asserted_label_set
                    .as_ref()
                    .expect("assertion set"),
                &vec!["IGS14".to_string(), "ITRF2".to_string()]
            );
        }
    }

    #[test]
    fn merge_applies_helmert_reconciliation_to_resolved_labels() {
        // Source 0 sets the target label. Source 1 is IGS20, which resolves to
        // ITRF2020 and is transformed into IGS14/ITRF2014 at the record epoch.
        // Expected coordinates duplicate the ITRF/IGN 2020->2014 table values:
        // T=(-1.4,-0.9,1.4) mm, dT=(0,-0.1,0.2) mm/year, D=-0.42 ppb.
        let a = sp3_build(
            &[("G01", [14000.0, -19000.0, 4000.0], Some(100.0), "")],
            "IGS14",
        );
        let b = sp3_build(
            &[("G02", [15000.0, -20000.0, 5000.0], Some(200.0), "")],
            "IGS20",
        );
        let opts = MergeOptions {
            min_agree: 1,
            frame_reconciliation: super::Sp3FrameReconciliationOptions::helmert(),
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[a, b], &opts).expect("helmert frame merge");

        let g02 = merged.states_at(0).expect("epoch 0")[&gps(2)];
        let got = g02.position.as_array();
        let expected = [
            14_999_999.992_3,
            -19_999_999.993_048_087,
            5_000_000.000_396_175,
        ];
        for axis in 0..3 {
            assert!(
                (got[axis] - expected[axis]).abs() < 2.0e-9,
                "axis {axis}: got {}, expected {}",
                got[axis],
                expected[axis]
            );
        }
        assert_eq!(merged.header.coordinate_system, "IGS14");
        assert_eq!(report.frame_reconciliations.len(), 1);
        let reconciliation = &report.frame_reconciliations[0];
        assert_eq!(reconciliation.method, Sp3FrameReconciliationMethod::Helmert);
        assert_eq!(reconciliation.source_label, "IGS20");
        assert_eq!(reconciliation.target_label, "IGS14");
        assert_eq!(reconciliation.records_affected, 1);
        assert_eq!(
            reconciliation
                .parameters
                .expect("published parameters")
                .translation_mm,
            [-1.4, -0.9, 1.4]
        );
        assert_eq!(
            reconciliation.catalog_source_frame,
            Some(crate::frame_catalog::TerrestrialFrame::Itrf2020)
        );
        assert_eq!(
            reconciliation.catalog_target_frame,
            Some(crate::frame_catalog::TerrestrialFrame::Itrf2014)
        );
        assert!(!reconciliation.catalog_inverse);
        assert_eq!(
            reconciliation
                .rates
                .expect("published rates")
                .translation_mm_per_year,
            [0.0, -0.1, 0.2]
        );
        assert!(reconciliation
            .provenance
            .as_ref()
            .expect("provenance")
            .contains("ITRF2020 to past ITRFs"));
    }

    #[test]
    fn merge_reports_inverse_helmert_catalog_direction() {
        let a = sp3_build(
            &[("G01", [14000.0, -19000.0, 4000.0], Some(100.0), "")],
            "IGS20",
        );
        let b = sp3_build(
            &[("G02", [15000.0, -20000.0, 5000.0], Some(200.0), "")],
            "IGS14",
        );
        let opts = MergeOptions {
            min_agree: 1,
            frame_reconciliation: super::Sp3FrameReconciliationOptions::helmert(),
            ..MergeOptions::default()
        };

        let (_merged, report) = merge(&[a, b], &opts).expect("inverse helmert frame merge");

        let reconciliation = &report.frame_reconciliations[0];
        assert_eq!(reconciliation.method, Sp3FrameReconciliationMethod::Helmert);
        assert_eq!(
            reconciliation.source_frame,
            Some(crate::frame_catalog::TerrestrialFrame::Itrf2014)
        );
        assert_eq!(
            reconciliation.target_frame,
            Some(crate::frame_catalog::TerrestrialFrame::Itrf2020)
        );
        assert_eq!(
            reconciliation.catalog_source_frame,
            Some(crate::frame_catalog::TerrestrialFrame::Itrf2020)
        );
        assert_eq!(
            reconciliation.catalog_target_frame,
            Some(crate::frame_catalog::TerrestrialFrame::Itrf2014)
        );
        assert!(reconciliation.catalog_inverse);
        assert_eq!(
            reconciliation
                .parameters
                .expect("published parameters")
                .translation_mm,
            [-1.4, -0.9, 1.4]
        );
    }

    #[test]
    fn helmert_identity_label_reconciliation_is_bit_equal() {
        let a = sp3_build(
            &[("G01", [14000.0, -19000.0, 4000.0], Some(100.0), "")],
            "IGS20",
        );
        let b = sp3_build(
            &[("G02", [15000.125, -20000.5, 5000.25], Some(200.0), "")],
            "IGc20",
        );
        let original = b.states_at(0).expect("epoch 0")[&gps(2)].position;
        let opts = MergeOptions {
            min_agree: 1,
            frame_reconciliation: super::Sp3FrameReconciliationOptions::helmert(),
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[a, b], &opts).expect("identity frame merge");

        let g02 = merged.states_at(0).expect("epoch 0")[&gps(2)].position;
        for axis in 0..3 {
            assert_eq!(
                g02.as_array()[axis].to_bits(),
                original.as_array()[axis].to_bits()
            );
        }
        assert_eq!(report.frame_reconciliations.len(), 1);
        assert!(report.frame_reconciliations[0].identity);
        assert!(report.frame_reconciliations[0].parameters.is_none());
    }

    #[test]
    fn helmert_reconciliation_rejects_unknown_labels() {
        let a = sp3_build(
            &[("G01", [14000.0, -19000.0, 4000.0], Some(100.0), "")],
            "ITRF2",
        );
        let b = sp3_build(
            &[("G02", [15000.0, -20000.0, 5000.0], Some(200.0), "")],
            "IGS20",
        );
        let opts = MergeOptions {
            frame_reconciliation: super::Sp3FrameReconciliationOptions::helmert(),
            ..MergeOptions::default()
        };

        let err = merge(&[a, b], &opts).expect_err("unknown frame label");

        assert!(
            err.to_string().contains("target label"),
            "unknown labels must not be guessed: {err}"
        );
    }

    #[test]
    fn merge_uses_finest_union_grid_and_fills_sparse_precedence_cells() {
        // 15-min (900 s) center A and 5-min (300 s) center B over the same span.
        // The default output uses the 5-min union grid. Under cell precedence A
        // wins the epochs it carries, and B fills A's :05/:10 holes.
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
        let (merged, _report) = merge(&[a, b], &opts).expect("mixed-interval union merge");

        assert_eq!(
            merged.header.epoch_interval_s, 300.0,
            "output is on the finest (300 s) input grid"
        );
        assert_eq!(
            merged.epochs.len(),
            4,
            "B fills the :05 and :10 epochs between A's samples"
        );
        let xs: Vec<f64> = (0..4)
            .map(|idx| {
                merged.states_at(idx).expect("epoch")[&gps(1)]
                    .position
                    .as_array()[0]
            })
            .collect();
        assert_eq!(
            xs,
            vec![15_000_000.0, 26_001_000.0, 26_002_000.0, 15_003_000.0]
        );
    }

    #[test]
    fn mixed_cadence_interpolates_only_the_clock_datum_for_filled_cells() {
        let reference_epoch: Vec<SatSample<'_>> = vec![
            ("G01", [15_001.0, -20_000.0, 5_000.0], Some(100.0)),
            ("G02", [15_002.0, -20_000.0, 5_000.0], Some(200.0)),
            ("G03", [15_003.0, -20_000.0, 5_000.0], Some(300.0)),
            ("G04", [15_004.0, -20_000.0, 5_000.0], Some(400.0)),
            ("G05", [15_005.0, -20_000.0, 5_000.0], Some(500.0)),
        ];
        let shifted_epoch: Vec<SatSample<'_>> = reference_epoch
            .iter()
            .map(|(sat, position, clock)| (*sat, *position, clock.map(|value| value + 50.0)))
            .collect();
        let a = sp3_epochs(
            0.0,
            &[reference_epoch.as_slice(), reference_epoch.as_slice()],
            900.0,
            "IGS14",
        );
        let b = sp3_epochs(
            0.0,
            &[
                shifted_epoch.as_slice(),
                shifted_epoch.as_slice(),
                shifted_epoch.as_slice(),
                shifted_epoch.as_slice(),
            ],
            300.0,
            "IGS14",
        );
        let opts = MergeOptions {
            combine: MergeCombine::Precedence,
            min_agree: 1,
            ..MergeOptions::default()
        };

        let (merged, _) = merge(&[a, b], &opts).expect("mixed-cadence clock merge");

        assert_eq!(merged.epochs.len(), 4);
        for epoch_index in 0..4 {
            let clock = merged.states_at(epoch_index).expect("epoch")[&gps(1)]
                .clock_s
                .expect("aligned clock");
            assert!(
                (clock - 100.0e-6).abs() < 1.0e-15,
                "epoch {epoch_index}: {clock}"
            );
        }
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
    fn merge_unions_non_commensurate_cadences_on_their_common_grid() {
        // 900 s and 400 s: no input is decimated and nothing is interpolated.
        // The default grid is the greatest common divisor of the steps and the
        // offsets, 100 s, so every input epoch has a place on it.
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

        let (merged, report) =
            merge(&[a, b], &MergeOptions::default()).expect("non-commensurate cadences");
        assert_eq!(merged.header.epoch_interval_s, 100.0);
        assert_eq!(merged.epochs.len(), 3);
        assert!(report.dropped_input_epochs.is_empty());
        let seconds = merged.epochs_j2000_seconds();
        assert_eq!(seconds[1] - seconds[0], 400.0);
        assert_eq!(seconds[2] - seconds[0], 900.0);
    }

    #[test]
    fn merge_reports_the_input_epochs_an_explicit_target_grid_drops() {
        // A 450.5 s target is a whole number of 10 ns ticks, so it is a grid an
        // SP3 header states. Anchored at the first epoch it holds 00:00 but not
        // 00:15, and each input's 00:15 epoch is reported as dropped.
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
        let (merged, report) = merge(&[mk(), mk()], &opts).expect("explicit target");
        assert_eq!(merged.header.epoch_interval_s, 450.5);
        assert_eq!(merged.epochs.len(), 1);
        let dropped: Vec<(usize, usize, super::DroppedEpochReason)> = report
            .dropped_input_epochs
            .iter()
            .map(|entry| (entry.source, entry.epoch_index, entry.reason))
            .collect();
        assert_eq!(
            dropped,
            vec![
                (0, 1, super::DroppedEpochReason::OffTargetGrid),
                (1, 1, super::DroppedEpochReason::OffTargetGrid),
            ]
        );

        // A target no whole number of ticks states is refused by name.
        let finer = MergeOptions {
            target_epoch_interval_s: Some(1.0e-9),
            ..MergeOptions::default()
        };
        let err = merge(&[mk()], &finer).expect_err("sub-tick target");
        assert!(err.to_string().contains("10-nanosecond ticks"), "{err}");
    }

    #[test]
    fn merge_header_first_epoch_describes_the_union_grid_start() {
        // Source A starts at 00:00, source B at 00:15 (both 15-min). The union
        // begins at 00:00 and ends at 00:45, and the synthetic header must agree.
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

        assert_eq!(
            merged.epochs.len(),
            4,
            "union epochs run from 00:00 to 00:45"
        );
        assert!(
            (merged.header.seconds_of_week - 345_600.0).abs() < 1.0e-6,
            "header sow must describe the union's first epoch 00:00 (345600 s), got {}",
            merged.header.seconds_of_week
        );
        assert!(
            merged.header.mjd_fraction.abs() < 1.0e-9,
            "header MJD fraction must describe 00:00, got {}",
            merged.header.mjd_fraction
        );
    }

    #[test]
    fn merge_writer_recomputes_header_for_a_fine_union_grid() {
        // A starts on a 15-minute grid at 00:00. B starts on a 7.5-minute grid at
        // 00:07:30. The output is the 7.5-minute union grid, and the writer must
        // use that derived interval and first epoch in its `##` header.
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

        assert_eq!(merged.epochs.len(), 5, "union epochs run every 7.5 minutes");
        let text = merged.to_sp3_string().expect("serialize SP3 product");
        let header = text
            .lines()
            .find(|line| line.starts_with("## "))
            .expect("written ## header");
        let first_epoch = text
            .lines()
            .find(|line| line.starts_with("*  "))
            .expect("written first epoch");

        assert_eq!(first_epoch, "*  2020  6 25  0  0  0.00000000");
        assert_eq!(
            header,
            "## 2111 345600.00000000   450.00000000 59025 0.0000000000000"
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
            precedence_scope: MergePrecedenceScope::SatelliteArc,
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[a, b], &opts).expect("merge");
        let epoch0 = merged.states_at(0).expect("epoch 0");

        assert!(epoch0.contains_key(&gps(1)));
        // G01 must not switch from source 0 at epoch 0 to source 1 at epoch 1.
        // That leaves epoch 1 with no cell at all, so it is not written; the
        // withheld cell and the omitted epoch are both reported.
        assert_eq!(merged.epochs.len(), 1);
        assert_eq!(report.omitted_epochs.len(), 1);
        assert_eq!(
            report.arc_withheld,
            vec![MergeFlag {
                epoch: report.omitted_epochs[0],
                satellite: gps(1),
                sources: vec![1],
            }]
        );
        // Source 1's clocks cannot be put on source 0's datum (one common
        // clock, below `clock_min_common`), so each is reported rather than
        // lost: at epoch 0, where the cell has source 0's clock, and at epoch 1,
        // where nothing is written.
        assert_eq!(
            report.clock_omissions,
            vec![
                ClockOmission {
                    epoch: merged.epochs[0],
                    satellite: gps(1),
                    source: 1,
                    reason: ClockOmissionReason::DatumNotObservable,
                    cell_has_clock: true,
                },
                ClockOmission {
                    epoch: report.omitted_epochs[0],
                    satellite: gps(1),
                    source: 1,
                    reason: ClockOmissionReason::DatumNotObservable,
                    cell_has_clock: false,
                },
            ]
        );
        assert_eq!(merged.header.epoch_interval_s, 900.0);
    }

    #[test]
    fn cell_precedence_fills_a_preferred_source_dropout() {
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

        let (merged, report) = merge(&[a, b], &opts).expect("merge");

        assert!(merged.states_at(0).expect("epoch 0").contains_key(&gps(1)));
        let epoch1 = merged.states_at(1).expect("epoch 1");
        assert!(
            epoch1.contains_key(&gps(1)),
            "source 1 must fill source 0's dropout"
        );
        assert_eq!(epoch1[&gps(1)].position.as_array()[0], 15_001_000.0);
        assert!(report
            .single_source
            .iter()
            .any(|entry| entry.satellite == gps(1) && entry.sources == vec![1]));
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
            (m.position_rms_m.unwrap() - 6.0_f64.sqrt()).abs() < 1.0e-6,
            "got rms {:?}",
            m.position_rms_m
        );
        assert!(
            (m.position_max_m.unwrap() - 3.0).abs() < 1.0e-6,
            "got max {:?}",
            m.position_max_m
        );

        // The pooled summaries over the single cell reproduce the cell values.
        assert!((report.position_agreement_rms_m().unwrap() - 6.0_f64.sqrt()).abs() < 1.0e-6);
        assert!((report.position_agreement_max_m().unwrap() - 3.0).abs() < 1.0e-6);

        // Per-epoch aggregate: one epoch, one multi-source satellite.
        let per_epoch = report.per_epoch_agreement();
        assert_eq!(per_epoch.len(), 1);
        assert_eq!(per_epoch[0].satellites, 1);
        assert!((per_epoch[0].position_rms_m.unwrap() - 6.0_f64.sqrt()).abs() < 1.0e-6);
        assert!((per_epoch[0].position_max_m.unwrap() - 3.0).abs() < 1.0e-6);
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
            assert!(m.position_rms_m.unwrap().abs() < 1.0e-9, "prn {prn}");
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
            if let Some(s) = super::super::interp::instant_to_j2000_seconds(ep) {
                igs_idx.insert(s.floor() as i64, i);
            }
        }

        let mut sumsq = 0.0_f64;
        let mut max = 0.0_f64;
        let mut n = 0usize;
        for (mi, ep) in merged.epochs.iter().enumerate() {
            let key = super::super::interp::instant_to_j2000_seconds(ep)
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

    #[test]
    fn merge_retains_clock_only_single_source() {
        let a = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))]);
        let b = sp3_records(&[
            ("G01", [15000.0002, -20000.0, 5000.0], Some(100.0)),
            ("G02", [0.0, 0.0, 0.0], Some(250.0)),
        ]);
        let opts = MergeOptions {
            clock_min_common: 1,
            provenance: Some(ProvenanceMode::Full),
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[a, b], &opts).expect("merge");

        let g01 = gps(1);
        let g02 = gps(2);

        // G01 has a merged state with position and clock
        assert!(merged.state(g01, 0).is_ok());
        let g01_state = merged.state(g01, 0).unwrap();
        assert!((g01_state.clock_s.unwrap() - 100.0e-6).abs() < 1.0e-12);

        // G02 has absent orbit: state() returns UnknownSatellite
        assert!(matches!(
            merged.state(g02, 0),
            Err(Error::UnknownSatellite(sat)) if sat == g02
        ));
        assert!(!merged.states_at(0).unwrap().contains_key(&g02));

        // G02 is retained as clock-only record
        let g02_clock = merged
            .clock_record(g02, 0)
            .expect("G02 clock record retained");
        assert_eq!(g02_clock.clock_us, 250.0);
        assert!((g02_clock.clock_s - 250.0e-6).abs() < 1.0e-12);
        assert!(g02_clock.velocity.is_none());
        assert!(g02_clock.clock_rate_s_s.is_none());
        assert_eq!(
            merged.clock_records_at(0).unwrap().get(&g02),
            Some(&g02_clock)
        );

        // Single-source report tracks G02 from source 1
        assert!(
            report
                .single_source
                .iter()
                .any(|f| f.satellite == g02 && f.sources == vec![1]),
            "single source report must track G02"
        );

        // Agreement metric for G02 represents clock-only without invented position metrics
        let g02_agreement = report
            .agreement
            .iter()
            .find(|m| m.satellite == g02)
            .expect("G02 agreement metric");
        assert_eq!(g02_agreement.position_members, 0);
        assert_eq!(g02_agreement.position_rms_m, None);
        assert_eq!(g02_agreement.position_max_m, None);
        assert_eq!(g02_agreement.clock_members, 1);
        assert_eq!(g02_agreement.clock_rms_s, Some(0.0));
        assert_eq!(g02_agreement.clock_max_s, Some(0.0));

        // Provenance in Full mode describes G02 cell with position: None
        let prov = report.provenance.as_ref().expect("provenance recorded");
        let g02_prov = prov
            .cells
            .iter()
            .find(|c| c.satellite == g02)
            .expect("G02 cell provenance");
        assert_eq!(g02_prov.position, None);
        assert_eq!(
            g02_prov.clock,
            Some(CellSelection::SingleSource { source: 1 })
        );

        // Contributor coverage tracks G02 under source 1
        assert_eq!(prov.coverage[1].cells_selected, 1); // G02 (G01 was combined)
        assert_eq!(prov.coverage[1].cells_contributed, 2);
    }

    #[test]
    fn align_clock_reference_shifts_clock_only_satellite() {
        let source0 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
        ]);
        let source1 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(105.0)),
            ("G03", [0.0, 0.0, 0.0], Some(305.0)),
        ]);

        let aligned = align_clock_reference(&source0, &source1, 1);

        let g01 = gps(1);
        let g03 = gps(3);

        // Offset is +5 us (105 - 100); aligned clocks shift by -5 us
        let g01_clock = aligned.state(g01, 0).unwrap().clock_s.unwrap();
        assert!((g01_clock - 100.0e-6).abs() < 1.0e-12);

        let g03_rec = aligned.clock_record(g03, 0).expect("G03 clock record");
        assert!((g03_rec.clock_us - 300.0).abs() < 1.0e-9);
        assert!((g03_rec.clock_s - 300.0e-6).abs() < 1.0e-15);
    }

    #[test]
    fn merge_shared_clocks_alignment_with_clock_only_satellite() {
        let source0 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
        ]);
        let source1 = sp3_records(&[
            ("G01", [15000.0002, -20000.0, 5000.0], Some(105.0)),
            ("G03", [0.0, 0.0, 0.0], Some(305.0)),
        ]);
        let opts = MergeOptions {
            clock_min_common: 1,
            ..MergeOptions::default()
        };

        let g01 = gps(1);
        let g02 = gps(2);
        let g03 = gps(3);

        let input_delta_m = super::dist3(
            &source0.state(g01, 0).unwrap().position.as_array(),
            &source1.state(g01, 0).unwrap().position.as_array(),
        );
        assert!(
            input_delta_m > 0.0 && input_delta_m <= opts.position_tolerance_m,
            "input delta {input_delta_m} m must be distinct and within configured tolerance {}",
            opts.position_tolerance_m
        );

        let (merged, report) = merge(&[source0, source1], &opts).expect("merge");

        // G01 agreed on position and datum-aligned clock
        assert!(merged.state(g01, 0).is_ok());
        let g01_clk = merged.state(g01, 0).unwrap().clock_s.unwrap();
        assert!((g01_clk - 100.0e-6).abs() < 1.0e-12);

        // G02 single-source from source 0
        assert!(merged.state(g02, 0).is_ok());

        // G03 clock-only record from source 1, aligned onto source 0 datum (-5 us)
        let g03_clk = merged.clock_record(g03, 0).expect("G03 clock-only record");
        assert!((g03_clk.clock_us - 300.0).abs() < 1.0e-9);
        assert!((g03_clk.clock_s - 300.0e-6).abs() < 1.0e-15);

        assert_eq!(report.single_source.len(), 2); // G02 and G03
    }

    #[test]
    fn merge_output_write_read_retention_round_trip() {
        // Verify retention round-trip fidelity using distinct within-tolerance
        // source positions whose mean is exactly representable in both binary
        // metres and F14.6 kilometres (0.25 m separation -> 0.125 m mean).
        let source0 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
        ]);
        let source1 = sp3_records(&[
            ("G01", [15000.00025, -20000.0, 5000.0], Some(100.0)),
            ("G03", [0.0, 0.0, 0.0], Some(300.0)),
        ]);
        let opts = MergeOptions {
            clock_min_common: 1,
            ..MergeOptions::default()
        };
        let (merged, _) = merge(&[source0, source1], &opts).expect("merge");

        let g01 = gps(1);
        assert_eq!(
            merged.state(g01, 0).unwrap().position.as_array()[0],
            15000000.125,
            "computed mean must match intended exact binary/wire mean"
        );

        let text = merged.to_sp3_string().expect("serialize SP3 product");
        // Serialized output must contain 0,0,0 sentinel for G03 with numeric clock
        let g03_line = text
            .lines()
            .find(|l| l.starts_with("PG03"))
            .expect("PG03 line in serialized merged output");
        assert!(
            g03_line.contains("0.000000      0.000000      0.000000    300.000000"),
            "got line: {g03_line}"
        );

        let reparsed = Sp3::parse(text.as_bytes()).expect("reparse serialized merged SP3");
        assert_eq!(reparsed.clock_records, merged.clock_records);
        assert_eq!(reparsed.states, merged.states);
        assert_eq!(reparsed.header.satellites, merged.header.satellites);
        assert_eq!(
            reparsed, merged,
            "full product equality across write/read round trip"
        );
    }

    #[test]
    fn merge_raw_clock_preservation_in_selected_orbit_and_clock_only_sources() {
        // Source 0: G01 anchor (100.0 us), G02 orbit record with clock_us = 200.0
        let source0 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(200.0)),
        ]);
        // Source 1: G01 anchor (100.0 us -> 0.0 us offset), G03 clock-only record with clock_us = 200.0
        let source1 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G03", [0.0, 0.0, 0.0], Some(200.0)),
        ]);
        let opts = MergeOptions {
            clock_min_common: 1,
            ..MergeOptions::default()
        };
        let (merged, _) = merge(&[source0, source1], &opts).expect("merge");

        let g01 = gps(1);
        let g02 = gps(2);
        let g03 = gps(3);

        // Combined identical G01 raw clock preserved bit-for-bit from sources (100.0 us)
        let g01_raw = merged.interp_raw[0].get(&g01).expect("G01 raw node");
        assert_eq!(g01_raw.clock_us, Some(100.0));

        // Orbit clock raw node preserved from source 0 without float drift (200.0 us)
        let g02_raw = merged.interp_raw[0].get(&g02).expect("G02 raw node");
        assert_eq!(g02_raw.clock_us, Some(200.0));

        // Clock-only raw record preserved from source 1 (200.0 us)
        let g03_clk = merged.clock_record(g03, 0).expect("G03 clock record");
        assert_eq!(g03_clk.clock_us, 200.0);

        // Differing clock consensus falls back to calculated clock arithmetic
        let diff_source0 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(100.0)),
            ("G03", [17000.0, -22000.0, 7000.0], Some(200.0)),
        ]);
        let diff_source1 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(100.000002)),
            ("G03", [17000.0, -22000.0, 7000.0], Some(200.0)),
        ]);
        let (diff_merged, _) = merge(&[diff_source0, diff_source1], &opts).expect("merge");
        let g02_diff_raw = diff_merged.interp_raw[0]
            .get(&g02)
            .expect("G02 diff raw node");
        let g02_diff_state = diff_merged.state(g02, 0).expect("G02 state");
        let computed_us = g02_diff_state.clock_s.expect("G02 clock_s") * 1.0e6;
        assert_eq!(g02_diff_raw.clock_us, Some(computed_us));
        assert_ne!(g02_diff_raw.clock_us, Some(100.0));
    }

    #[test]
    fn merge_clock_only_satellite_never_creates_interp_nodes() {
        let source0 = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))]);
        let source1 = sp3_records(&[
            ("G01", [15000.0002, -20000.0, 5000.0], Some(100.0)),
            ("G02", [0.0, 0.0, 0.0], Some(200.0)),
        ]);
        let opts = MergeOptions {
            clock_min_common: 1,
            ..MergeOptions::default()
        };
        let (merged, _) = merge(&[source0, source1], &opts).expect("merge");

        let g02 = gps(2);
        // Verify actual retained clock before asserting absence of orbit nodes
        let g02_clk = merged
            .clock_record(g02, 0)
            .expect("G02 clock record retained");
        assert_eq!(g02_clk.clock_us, 200.0);

        // Interp raw node must not exist for G02
        assert!(
            !merged.interp_raw[0].contains_key(&g02),
            "interp_raw must never contain clock-only satellite"
        );

        // Position evaluation must fail with UnknownSatellite
        let err = merged.position(g02, merged.epochs[0]);
        assert!(
            matches!(err, Err(Error::UnknownSatellite(sat)) if sat == g02),
            "expected UnknownSatellite, got {err:?}"
        );
    }

    #[test]
    fn merge_satellite_arc_precedence_selects_clock_only_owner() {
        let source0 = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))]);
        let source1 = sp3_records(&[
            ("G01", [15000.0002, -20000.0, 5000.0], Some(100.0)),
            ("G02", [0.0, 0.0, 0.0], Some(200.0)),
        ]);
        let opts = MergeOptions {
            clock_min_common: 1,
            combine: MergeCombine::Precedence,
            precedence_scope: MergePrecedenceScope::SatelliteArc,
            ..MergeOptions::default()
        };

        let (merged, _) = merge(&[source0, source1], &opts).expect("merge arc precedence");
        let g02 = gps(2);
        // Source 1 is identified as arc owner for G02 despite it being clock-only
        assert!(merged.clock_record(g02, 0).is_ok());
        assert_eq!(merged.clock_record(g02, 0).unwrap().clock_us, 200.0);
    }

    #[test]
    fn merge_provenance_tracks_clock_only_transitions_and_coverage() {
        let source0 = sp3_records(&[("G01", [15000.0, -20000.0, 5000.0], Some(100.0))]);
        let source1 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], Some(100.0)),
            ("G02", [0.0, 0.0, 0.0], Some(200.0)),
        ]);
        let opts = MergeOptions {
            clock_min_common: 1,
            provenance: Some(ProvenanceMode::Full),
            ..MergeOptions::default()
        };

        let (_merged, report) = merge(&[source0, source1], &opts).expect("merge provenance");
        let prov = report.provenance.expect("provenance");

        let g02 = gps(2);
        // Transition for G02 exists as SoleAvailability
        let g02_trans = prov
            .transitions
            .iter()
            .find(|t| t.satellite == g02)
            .expect("G02 transition");
        assert_eq!(g02_trans.from_source, None);
        assert_eq!(g02_trans.to_source, Some(1));
        assert_eq!(g02_trans.reason, TransitionReason::SoleAvailability);

        // Coverage accurately counts accepted cells without dropping G02
        assert_eq!(prov.coverage[0].cells_contributed, 1);
        assert_eq!(prov.coverage[1].cells_contributed, 2);
        assert_eq!(prov.coverage[0].cells_absent, 1); // missed G02
        assert_eq!(prov.coverage[1].cells_absent, 0);
    }

    #[test]
    fn merge_provenance_mixed_availability_tracks_distinct_position_and_clock_contributors() {
        // Source 0 supplies position for G01 but has no clock (sentinel).
        // Source 1 supplies clock for G01 but has no orbit (sentinel 0,0,0).
        // Both share G02 with valid orbit and clock to anchor the clock datum offset (0.0 us).
        let source0 = sp3_records(&[
            ("G01", [15000.0, -20000.0, 5000.0], None),
            ("G02", [16000.0, -21000.0, 6000.0], Some(100.0)),
        ]);
        let source1 = sp3_records(&[
            ("G01", [0.0, 0.0, 0.0], Some(200.0)),
            ("G02", [16000.0, -21000.0, 6000.0], Some(100.0)),
        ]);
        let opts = MergeOptions {
            clock_min_common: 1,
            provenance: Some(ProvenanceMode::Full),
            ..MergeOptions::default()
        };

        let (merged, report) = merge(&[source0, source1], &opts).expect("merge mixed availability");
        let g01 = gps(1);

        // G01 has position from source 0 and clock from source 1
        let g01_state = merged.state(g01, 0).expect("G01 merged state");
        assert!((g01_state.clock_s.unwrap() - 200.0e-6).abs() < 1.0e-12);

        let prov = report.provenance.expect("provenance recorded");
        let g01_prov = prov
            .cells
            .iter()
            .find(|c| c.satellite == g01)
            .expect("G01 cell provenance");
        assert_eq!(
            g01_prov.position,
            Some(CellSelection::SingleSource { source: 0 })
        );
        assert_eq!(
            g01_prov.clock,
            Some(CellSelection::SingleSource { source: 1 })
        );

        // Both sources contributed to G01 (source 0 position, source 1 clock),
        // and both contributed to G02. Thus both sources contributed to 2 accepted cells.
        assert_eq!(prov.coverage[0].cells_contributed, 2);
        assert_eq!(prov.coverage[1].cells_contributed, 2);
        assert_eq!(prov.coverage[0].cells_absent, 0);
        assert_eq!(prov.coverage[1].cells_absent, 0);
    }

    #[test]
    fn merge_drops_velocities_and_rates_for_all_records() {
        // Construct SP3-c Velocity product (#cV) with V records
        let v_file = "\
#cV2020  6 25  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59025 0.0000000000000
+    2   G01G02  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         5  5  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3-c FIXTURE
*  2020  6 25  0  0  0.00000000
PG01  15000.000000 -20000.000000   5000.000000    100.000000
VG01   1000.000000  -2000.000000   3000.000000      1.500000
PG02      0.000000      0.000000      0.000000    200.000000
VG02   1000.000000  -2000.000000   3000.000000      2.500000
EOF
";
        let sp3_input = Sp3::parse(v_file.as_bytes()).expect("parse velocity product");
        assert_eq!(sp3_input.header.data_type, Sp3DataType::Velocity);
        let g01 = gps(1);
        let g02 = gps(2);
        assert!(sp3_input.state(g01, 0).unwrap().velocity.is_some());
        assert!(sp3_input.state(g01, 0).unwrap().clock_rate_s_s.is_some());
        assert!(sp3_input.clock_record(g02, 0).unwrap().velocity.is_some());
        assert!(sp3_input
            .clock_record(g02, 0)
            .unwrap()
            .clock_rate_raw
            .is_some());

        let (merged, _) = merge(&[sp3_input], &MergeOptions::default()).expect("merge");

        // Merged header is position-only
        assert_eq!(merged.header.data_type, Sp3DataType::Position);

        // Position state drops velocity and clock rate
        let m_g01 = merged.state(g01, 0).expect("merged state G01");
        assert!(
            m_g01.velocity.is_none(),
            "merged state velocity must be None"
        );
        assert!(
            m_g01.clock_rate_s_s.is_none(),
            "merged state clock_rate_s_s must be None"
        );

        // Clock record drops velocity and clock rate
        let m_g02 = merged.clock_record(g02, 0).expect("merged clock G02");
        assert!(
            m_g02.velocity.is_none(),
            "merged clock velocity must be None"
        );
        assert!(
            m_g02.clock_rate_s_s.is_none(),
            "merged clock clock_rate_s_s must be None"
        );
        assert!(
            m_g02.clock_rate_raw.is_none(),
            "merged clock clock_rate_raw must be None"
        );
    }
}
