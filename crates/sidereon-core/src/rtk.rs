//! RTK double-difference primitives.
//!
//! This module owns the language-independent carrier/code double-difference
//! construction used by Sidereon' public RTK API.

use crate::astro::angles::normalize_geodetic_lon_rad;
use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use crate::astro::math::vec3::{dot3, norm3, sub3};
use crate::astro::time::model::{Instant, JulianDateSplit, TimeScale};

use crate::ambiguity::{self, AmbiguityId, NarrowLaneParams};
use crate::carrier_phase::{
    detect_cycle_slips, validate_hatch_window_cap, ArcEpoch, CarrierPhaseError, CycleSlipOptions,
    SlipReason,
};
use crate::combinations::{self, IonosphereFreeError};
use crate::constants::{DEG_TO_RAD, KM_TO_M, RAD_TO_DEG};
use crate::tropo::{tropo_slant, Met};
use crate::validate;
use crate::Wgs84Geodetic;
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

/// One single-frequency code/carrier observation at a receiver.
#[derive(Debug, Clone, PartialEq)]
pub struct Observation {
    pub satellite_id: String,
    pub ambiguity_id: String,
    pub code_m: f64,
    pub phase_m: f64,
}

/// One single-frequency RTK observation for code-smoothing preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeSmoothingObservation {
    pub satellite_id: String,
    pub ambiguity_id: String,
    pub code_m: f64,
    pub phase_m: f64,
    pub lli: Option<i64>,
}

/// One RTK epoch for base/rover code-smoothing preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub struct CodeSmoothingEpoch {
    pub base_observations: Vec<CodeSmoothingObservation>,
    pub rover_observations: Vec<CodeSmoothingObservation>,
}

/// One single-frequency RTK observation for cycle-slip preprocessing.
pub type CycleSlipObservation = CodeSmoothingObservation;

/// One single-frequency RTK epoch for cycle-slip preprocessing.
pub type CycleSlipEpoch = CodeSmoothingEpoch;

/// One receiver's dual-frequency code/carrier observation used by the
/// wide-lane RTK pre-step.
#[derive(Debug, Clone, PartialEq)]
pub struct DualObservation {
    pub ambiguity_id: String,
    pub p1_m: f64,
    pub p2_m: f64,
    pub phi1_cycles: f64,
    pub phi2_cycles: f64,
    pub f1_hz: f64,
    pub f2_hz: f64,
}

/// Paired base/rover dual-frequency observation for one satellite.
#[derive(Debug, Clone, PartialEq)]
pub struct DualSatelliteObservation {
    pub satellite_id: String,
    pub base: DualObservation,
    pub rover: DualObservation,
}

/// One dual-frequency RTK epoch, already normalized to satellites usable by the
/// caller's baseline epoch contract.
#[derive(Debug, Clone, PartialEq)]
pub struct DualEpoch {
    pub observations: Vec<DualSatelliteObservation>,
}

/// One receiver's dual-frequency observation for cycle-slip preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub struct DualCycleSlipObservation {
    pub satellite_id: String,
    pub ambiguity_id: String,
    pub p1_m: f64,
    pub p2_m: f64,
    pub phi1_cycles: f64,
    pub phi2_cycles: f64,
    pub f1_hz: f64,
    pub f2_hz: f64,
    pub lli1: Option<i64>,
    pub lli2: Option<i64>,
}

/// One dual-frequency RTK epoch for cycle-slip preprocessing.
#[derive(Debug, Clone, PartialEq)]
pub struct DualCycleSlipEpoch {
    /// Caller-provided deterministic epoch ordering key.
    pub epoch_sort_key: String,
    /// Comparable epoch coordinate in seconds, when the caller can supply one.
    pub gap_time_s: Option<f64>,
    pub base_observations: Vec<DualCycleSlipObservation>,
    pub rover_observations: Vec<DualCycleSlipObservation>,
}

/// One receiver's dual-frequency observation plus a precomputed non-dispersive
/// range correction removed from the ionosphere-free observable.
#[derive(Debug, Clone, PartialEq)]
pub struct DualIonosphereFreeObservation {
    pub ambiguity_id: String,
    pub p1_m: f64,
    pub p2_m: f64,
    pub phi1_cycles: f64,
    pub phi2_cycles: f64,
    pub f1_hz: f64,
    pub f2_hz: f64,
    pub tropo_m: f64,
}

/// Paired base/rover dual-frequency observation for IF/narrow-lane conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct DualIonosphereFreeSatelliteObservation {
    pub satellite_id: String,
    pub base: DualIonosphereFreeObservation,
    pub rover: DualIonosphereFreeObservation,
}

/// One dual-frequency RTK epoch for IF/narrow-lane conversion.
#[derive(Debug, Clone, PartialEq)]
pub struct DualIonosphereFreeEpoch {
    pub observations: Vec<DualIonosphereFreeSatelliteObservation>,
}

/// One normalized dual-frequency RTK epoch before IF/narrow-lane conversion.
///
/// The core owns the optional troposphere setup, so this shape carries the
/// satellite positions and split Julian epoch needed to form per-receiver
/// slant delays before the IF conversion is applied.
#[derive(Debug, Clone, PartialEq)]
pub struct DualIonosphereFreeSetupEpoch {
    pub jd_whole: f64,
    pub jd_fraction: f64,
    pub observations: Vec<DualSatelliteObservation>,
    pub base_satellite_positions_m: BTreeMap<String, [f64; 3]>,
    pub rover_satellite_positions_m: BTreeMap<String, [f64; 3]>,
}

/// One converted single-observable epoch.
#[derive(Debug, Clone, PartialEq)]
pub struct IonosphereFreeBaselineEpoch {
    pub epoch_index: usize,
    pub satellite_ids: Vec<String>,
    pub base_observations: Vec<Observation>,
    pub rover_observations: Vec<Observation>,
}

/// Converted IF epochs plus per-DD narrow-lane ambiguity parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct IonosphereFreeBaselineResult {
    pub epochs: Vec<IonosphereFreeBaselineEpoch>,
    pub wavelengths_m: BTreeMap<String, f64>,
    pub offsets_m: BTreeMap<String, f64>,
}

/// One normalized RTK baseline epoch for reference-satellite selection.
#[derive(Debug, Clone, PartialEq)]
pub struct BaselineReferenceEpoch {
    /// Satellites available in both receivers and in the position maps.
    pub available_satellite_ids: Vec<String>,
    /// Satellite ECEF positions in metres, keyed by satellite id.
    pub satellite_positions_m: BTreeMap<String, [f64; 3]>,
}

/// One RTK baseline epoch's satellite positions for elevation masking.
#[derive(Debug, Clone, PartialEq)]
pub struct ElevationMaskEpoch {
    /// Satellite ECEF positions in metres, keyed by satellite id.
    pub satellite_positions_m: BTreeMap<String, [f64; 3]>,
}

/// Per-epoch elevation-mask decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElevationMaskEpochResult {
    /// Satellites at or above the mask in this epoch, sorted by id.
    pub kept_satellite_ids: Vec<String>,
}

/// Elevation-mask result for a baseline arc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElevationMaskResult {
    pub epochs: Vec<ElevationMaskEpochResult>,
    /// Satellites below the mask in any epoch, sorted by id.
    pub masked_satellite_ids: Vec<String>,
}

/// Wide-lane integer estimation controls.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WideLaneOptions {
    pub min_epochs: usize,
    pub tolerance_cycles: f64,
    /// When true, short ambiguity fragments are omitted instead of failing. This
    /// is the `:split_arc` policy used by Sidereon after cycle-slip segmentation.
    pub skip_short_fragments: bool,
}

/// Error from dual-frequency wide-lane integer estimation.
#[derive(Debug, Clone, PartialEq)]
pub enum WideLaneError {
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    ReferenceSatelliteMissing(String),
    WideLaneFailed {
        satellite_id: String,
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
}

/// Error from dual-frequency IF/narrow-lane conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IonosphereFreeBaselineError {
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    NoEpochs,
    InconsistentFrequencies(String),
    NarrowLaneFailed(IonosphereFreeError),
    IonosphereFreeFailed {
        satellite_id: String,
        reason: IonosphereFreeError,
    },
}

/// Error from RTK code-smoothing preprocessing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeSmoothingError {
    InvalidWindowCap,
}

/// Base/rover receiver side for RTK preprocessing diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CycleSlipReceiver {
    Base,
    Rover,
}

pub use crate::ambiguity::CycleSlipPolicy;

/// Public split-arc metadata, with epoch indices for callers to remap.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleSlipSplitArc {
    pub receiver: CycleSlipReceiver,
    pub satellite_id: String,
    pub ambiguity_id: String,
    pub start_epoch_index: usize,
    pub end_epoch_index: usize,
    pub n_epochs: usize,
}

/// Prepared single-frequency RTK epochs and policy metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct CycleSlipPrepResult {
    pub epochs: Vec<CycleSlipEpoch>,
    pub dropped_sats: Vec<String>,
    pub split_arcs: Vec<CycleSlipSplitArc>,
}

/// Prepared dual-frequency RTK epochs and policy metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct DualCycleSlipPrepResult {
    pub epochs: Vec<DualCycleSlipEpoch>,
    pub dropped_sats: Vec<String>,
    pub split_arcs: Vec<CycleSlipSplitArc>,
}

/// Error from RTK cycle-slip preprocessing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CycleSlipPrepError {
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    CycleSlipDetected {
        receiver: CycleSlipReceiver,
        satellite_id: String,
        epoch_index: usize,
        reasons: Vec<SlipReason>,
    },
}

/// Reference-satellite option for double-difference construction.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum ReferenceSelection {
    /// Pick the lexicographically first common satellite per constellation.
    #[default]
    Auto,
    /// Use one fixed reference satellite. Valid only for single-system data.
    Satellite(String),
    /// Use one fixed reference satellite per constellation letter.
    PerSystem(BTreeMap<String, String>),
}

/// Reference report shape matching the Sidereon public API.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceReport {
    Satellite(String),
    PerSystem(BTreeMap<String, String>),
}

/// Baseline-solver reference-satellite option.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum BaselineReferenceSelection {
    /// Pick the highest-average-elevation satellite per constellation.
    #[default]
    Auto,
    /// Use one fixed reference satellite. Valid only for single-system data.
    Satellite(String),
    /// Use one fixed reference satellite per constellation letter.
    PerSystem(BTreeMap<String, String>),
}

/// One non-reference satellite's double-difference measurement.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDifference {
    pub satellite_id: String,
    pub reference_satellite_id: String,
    pub ambiguity_id: String,
    pub code_m: f64,
    pub phase_m: f64,
}

/// Result of double-difference construction.
#[derive(Debug, Clone, PartialEq)]
pub struct DoubleDifferenceResult {
    pub reference_satellite_id: ReferenceReport,
    pub double_differences: Vec<DoubleDifference>,
    pub dropped_sats: Vec<String>,
}

/// Error from double-difference construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DoubleDifferenceError {
    InvalidInput {
        field: &'static str,
        reason: &'static str,
    },
    DuplicateObservation(String),
    TooFewCommonSatellites {
        count: usize,
        minimum: usize,
    },
    NoCommonReferenceSatellite(String),
    MissingSatellitePosition(String),
    ReferenceSatelliteMissing(String),
    ReferenceSatelliteSingleSystem(String),
    ReferenceSatelliteMissingSystem(String),
    InvalidReferenceOption,
}

#[derive(Debug, Clone)]
struct SingleDifference {
    satellite_id: String,
    ambiguity_id: AmbiguityId,
    code_m: f64,
    phase_m: f64,
}

#[derive(Debug, Clone, Copy)]
struct CodeSmoothingState {
    p_smooth_m: f64,
    phase_m: f64,
    window: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CycleSlipEvent {
    receiver: CycleSlipReceiver,
    satellite_id: String,
    epoch_index: usize,
    reasons: Vec<SlipReason>,
}

#[derive(Debug, Clone)]
struct DualSingleDifference {
    satellite_id: String,
    ambiguity_id: AmbiguityId,
    wide_lane_cycles: f64,
}

#[derive(Debug, Clone)]
struct WideLaneSample {
    ambiguity_id: AmbiguityId,
    cycles: f64,
}

#[derive(Debug, Clone, Copy)]
struct DualTropoReceiver {
    position_m: [f64; 3],
    geodetic: Wgs84Geodetic,
}

#[derive(Debug, Clone, Copy)]
struct DualTropoConfig {
    base: DualTropoReceiver,
    rover: DualTropoReceiver,
}

/// Build code and carrier-phase double differences from base and rover observations.
pub fn double_differences(
    base_observations: &[Observation],
    rover_observations: &[Observation],
    reference: ReferenceSelection,
) -> Result<DoubleDifferenceResult, DoubleDifferenceError> {
    let base = observations_by_satellite(base_observations)?;
    let rover = observations_by_satellite(rover_observations)?;
    let (common, dropped_sats) = common_observations(&base, &rover)?;
    let refs = reference_satellites(&common, &reference)?;
    ensure_non_reference_satellites(&common, &refs)?;
    let ref_set = refs.values().cloned().collect::<BTreeSet<_>>();

    let mut ref_data = BTreeMap::new();
    for (system, reference_sat) in &refs {
        let ref_base = base.get(reference_sat).expect("reference base observation");
        let ref_rover = rover
            .get(reference_sat)
            .expect("reference rover observation");
        ref_data.insert(
            system.clone(),
            SingleDifference {
                satellite_id: reference_sat.clone(),
                ambiguity_id: single_difference_ambiguity_id(reference_sat, ref_base, ref_rover),
                code_m: finite_double_difference_value(
                    ref_rover.code_m - ref_base.code_m,
                    "rtk single difference code_m",
                )?,
                phase_m: finite_double_difference_value(
                    ref_rover.phase_m - ref_base.phase_m,
                    "rtk single difference phase_m",
                )?,
            },
        );
    }

    let double_differences = common
        .iter()
        .filter(|sat| !ref_set.contains(*sat))
        .map(|sat| {
            let system = satellite_system(sat);
            let reference = ref_data
                .get(&system)
                .expect("reference for satellite system");
            let base_obs = base.get(sat).expect("base observation");
            let rover_obs = rover.get(sat).expect("rover observation");
            let sat_sd_id = single_difference_ambiguity_id(sat, base_obs, rover_obs);
            let code_m = finite_double_difference_value(
                rover_obs.code_m - base_obs.code_m - reference.code_m,
                "rtk double difference code_m",
            )?;
            let phase_m = finite_double_difference_value(
                rover_obs.phase_m - base_obs.phase_m - reference.phase_m,
                "rtk double difference phase_m",
            )?;

            Ok(DoubleDifference {
                satellite_id: sat.clone(),
                reference_satellite_id: reference.satellite_id.clone(),
                ambiguity_id: double_difference_ambiguity_id(sat, &sat_sd_id, reference)
                    .into_string(),
                code_m,
                phase_m,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(DoubleDifferenceResult {
        reference_satellite_id: reference_report(refs),
        double_differences,
        dropped_sats,
    })
}

/// Hatch-smooth code observations independently for base and rover receivers.
///
/// State is keyed by ambiguity id, reset when LLI bit 0 is set, and advanced in
/// satellite-id order within each epoch to match the Sidereon public RTK path.
pub fn hatch_smooth_baseline_code_epochs(
    epochs: &[CodeSmoothingEpoch],
    hatch_window_cap: usize,
) -> Result<Vec<CodeSmoothingEpoch>, CodeSmoothingError> {
    let hatch_window_cap = validate_hatch_window_cap(hatch_window_cap)
        .map_err(|_| CodeSmoothingError::InvalidWindowCap)?;

    let mut smoothed = epochs.to_vec();
    smooth_receiver_code_epochs(&mut smoothed, Receiver::Base, hatch_window_cap);
    smooth_receiver_code_epochs(&mut smoothed, Receiver::Rover, hatch_window_cap);
    Ok(smoothed)
}

/// Prepare single-frequency RTK epochs according to the configured cycle-slip policy.
///
/// The core owns the language-independent LLI event ordering, drop/split policy
/// behavior, split-arc ambiguity ids, and reacquired-satellite ambiguity ids.
pub fn prepare_cycle_slip_baseline_epochs(
    epochs: &[CycleSlipEpoch],
    policy: CycleSlipPolicy,
) -> Result<CycleSlipPrepResult, CycleSlipPrepError> {
    validate_cycle_slip_baseline_epochs(epochs)?;
    let slips = cycle_slip_events(epochs);

    let (prepared, dropped_sats, split_arcs) = match (policy, slips.as_slice()) {
        (_, []) => (epochs.to_vec(), Vec::new(), Vec::new()),
        (CycleSlipPolicy::Error, [slip, ..]) => {
            return Err(CycleSlipPrepError::CycleSlipDetected {
                receiver: slip.receiver,
                satellite_id: slip.satellite_id.clone(),
                epoch_index: slip.epoch_index,
                reasons: slip.reasons.clone(),
            });
        }
        (CycleSlipPolicy::DropSatellite, slips) => {
            let dropped_sats = dropped_cycle_slip_sats(slips);
            (
                drop_cycle_slip_satellites(epochs, &dropped_sats),
                dropped_sats,
                Vec::new(),
            )
        }
        (CycleSlipPolicy::SplitArc, slips) => {
            let split_sides = cycle_slip_split_sides(slips);
            let split_epochs = split_cycle_slip_arcs(epochs, &split_sides, slips);
            let split_arcs = cycle_slip_split_metadata(&split_epochs, &split_sides);
            (split_epochs, Vec::new(), split_arcs)
        }
    };

    Ok(CycleSlipPrepResult {
        epochs: segment_reacquired_arcs(prepared),
        dropped_sats,
        split_arcs,
    })
}

/// Prepare dual-frequency RTK epochs according to the configured cycle-slip policy.
///
/// The core owns the LLI, data-gap, geometry-free, and Melbourne-Wubbena
/// classification used before wide-lane estimation, plus the drop/split policy,
/// split-arc ambiguity ids, and reacquired-satellite ambiguity ids.
pub fn prepare_dual_cycle_slip_baseline_epochs(
    epochs: &[DualCycleSlipEpoch],
    policy: CycleSlipPolicy,
    options: CycleSlipOptions,
) -> Result<DualCycleSlipPrepResult, CycleSlipPrepError> {
    validate_cycle_slip_options(options)?;
    validate_dual_cycle_slip_baseline_epochs(epochs)?;
    let slips = dual_cycle_slip_events(epochs, options)?;

    let (prepared, dropped_sats, split_arcs) = match (policy, slips.as_slice()) {
        (_, []) => (epochs.to_vec(), Vec::new(), Vec::new()),
        (CycleSlipPolicy::Error, [slip, ..]) => {
            return Err(CycleSlipPrepError::CycleSlipDetected {
                receiver: slip.receiver,
                satellite_id: slip.satellite_id.clone(),
                epoch_index: slip.epoch_index,
                reasons: slip.reasons.clone(),
            });
        }
        (CycleSlipPolicy::DropSatellite, slips) => {
            let dropped_sats = dropped_cycle_slip_sats(slips);
            (
                drop_dual_cycle_slip_satellites(epochs, &dropped_sats),
                dropped_sats,
                Vec::new(),
            )
        }
        (CycleSlipPolicy::SplitArc, slips) => {
            let split_sides = cycle_slip_split_sides(slips);
            let split_epochs = split_dual_cycle_slip_arcs(epochs, &split_sides, slips);
            let split_arcs = dual_cycle_slip_split_metadata(&split_epochs, &split_sides);
            (split_epochs, Vec::new(), split_arcs)
        }
    };

    Ok(DualCycleSlipPrepResult {
        epochs: segment_reacquired_dual_arcs(prepared),
        dropped_sats,
        split_arcs,
    })
}

/// Select per-system RTK baseline reference satellites.
///
/// This is the baseline-solver rule: automatic references are the
/// highest-average-elevation satellites within each constellation's per-system
/// common set. It is intentionally separate from [`double_differences`], whose
/// public helper keeps the older lexicographic default.
pub fn baseline_reference_satellites(
    base_m: [f64; 3],
    epochs: &[BaselineReferenceEpoch],
    selection: BaselineReferenceSelection,
) -> Result<BTreeMap<String, String>, DoubleDifferenceError> {
    validate_rtk_receiver_position(base_m)?;
    validate_baseline_reference_positions(base_m, epochs)?;

    let all_sats = baseline_all_satellites(epochs);
    let systems = all_sats
        .iter()
        .map(|sat| satellite_system(sat))
        .collect::<BTreeSet<_>>();
    let common_by_system = baseline_common_by_system(epochs);

    match selection {
        BaselineReferenceSelection::Auto => systems
            .into_iter()
            .map(|system| {
                let common = common_by_system.get(&system).cloned().unwrap_or_default();
                if common.is_empty() {
                    Err(DoubleDifferenceError::NoCommonReferenceSatellite(system))
                } else {
                    let system_epochs = baseline_epochs_for_system(epochs, &system);
                    let reference = highest_elevation_reference(base_m, &system_epochs, &common)?;
                    Ok((system, reference))
                }
            })
            .collect(),
        BaselineReferenceSelection::Satellite(sat) => {
            let systems = systems.into_iter().collect::<Vec<_>>();
            match systems.as_slice() {
                [system] => {
                    if common_by_system
                        .get(system)
                        .is_some_and(|sats| sats.contains(&sat))
                    {
                        Ok(BTreeMap::from([(system.clone(), sat)]))
                    } else {
                        Err(DoubleDifferenceError::ReferenceSatelliteMissing(sat))
                    }
                }
                _ => Err(DoubleDifferenceError::ReferenceSatelliteSingleSystem(sat)),
            }
        }
        BaselineReferenceSelection::PerSystem(refs) => {
            let mut out = BTreeMap::new();
            for system in systems {
                let Some(sat) = refs.get(&system) else {
                    return Err(DoubleDifferenceError::ReferenceSatelliteMissingSystem(
                        system,
                    ));
                };
                if common_by_system
                    .get(&system)
                    .is_some_and(|sats| sats.contains(sat))
                {
                    out.insert(system, sat.clone());
                } else {
                    return Err(DoubleDifferenceError::ReferenceSatelliteMissing(
                        sat.clone(),
                    ));
                }
            }
            Ok(out)
        }
    }
}

fn validate_baseline_reference_positions(
    base_m: [f64; 3],
    epochs: &[BaselineReferenceEpoch],
) -> Result<(), DoubleDifferenceError> {
    for epoch in epochs {
        for sat in &epoch.available_satellite_ids {
            let sat_pos = *baseline_satellite_position(epoch, sat)?;
            validate_rtk_satellite_geometry(base_m, sat_pos)?;
        }
    }
    Ok(())
}

fn baseline_satellite_position<'a>(
    epoch: &'a BaselineReferenceEpoch,
    sat: &str,
) -> Result<&'a [f64; 3], DoubleDifferenceError> {
    validate::present(
        epoch.satellite_positions_m.get(sat),
        "satellite_positions_m",
    )
    .map_err(|_| DoubleDifferenceError::MissingSatellitePosition(sat.to_string()))
}

fn validate_rtk_receiver_position(base_m: [f64; 3]) -> Result<(), DoubleDifferenceError> {
    validate::finite_vec3(base_m, "rtk base position_m")
        .map_err(double_difference_invalid_input)?;
    let norm = norm3(base_m);
    if !norm.is_finite() {
        return Err(invalid_double_difference_input(
            "rtk base position_m",
            "out of range",
        ));
    }
    if norm <= 0.0 {
        return Err(invalid_double_difference_input(
            "rtk base position_m",
            "degenerate geometry",
        ));
    }
    Ok(())
}

fn validate_rtk_satellite_geometry(
    base_m: [f64; 3],
    sat_pos_m: [f64; 3],
) -> Result<(), DoubleDifferenceError> {
    validate::finite_vec3(sat_pos_m, "rtk satellite position_m")
        .map_err(double_difference_invalid_input)?;
    rtk_line_of_sight(base_m, sat_pos_m).map(|_| ())
}

/// Apply an RTK elevation mask at the base receiver.
///
/// A satellite is kept in an epoch when the sine of its geocentric-up elevation
/// is at least `sin(mask_deg)`. The caller owns receiver observation maps and
/// uses the returned keep lists to thin each epoch consistently.
pub fn apply_elevation_mask(
    base_m: [f64; 3],
    epochs: &[ElevationMaskEpoch],
    mask_deg: f64,
) -> Result<ElevationMaskResult, DoubleDifferenceError> {
    validate_rtk_receiver_position(base_m)?;
    let mask_deg = validate::finite_in_range(mask_deg, -90.0, 90.0, "rtk elevation mask_deg")
        .map_err(double_difference_invalid_input)?;
    let min_sin = (mask_deg * DEG_TO_RAD).sin();
    let up = local_up(base_m);
    let mut masked = BTreeSet::new();
    let mut results = Vec::with_capacity(epochs.len());

    for epoch in epochs {
        let mut kept = Vec::new();
        for (sat, sat_pos) in &epoch.satellite_positions_m {
            if elevation_score_with_up(base_m, up, *sat_pos)? >= min_sin {
                kept.push(sat.clone());
            } else {
                masked.insert(sat.clone());
            }
        }
        results.push(ElevationMaskEpochResult {
            kept_satellite_ids: kept,
        });
    }

    Ok(ElevationMaskResult {
        epochs: results,
        masked_satellite_ids: masked.into_iter().collect(),
    })
}

/// Estimate arc-level double-difference wide-lane integers from dual-frequency
/// base/rover observations.
pub fn estimate_wide_lane_ambiguities(
    epochs: &[DualEpoch],
    reference_satellite_id: &str,
    options: WideLaneOptions,
) -> Result<BTreeMap<String, i64>, WideLaneError> {
    validate_wide_lane_options(options)?;
    validate_wide_lane_epochs(epochs)?;
    let mut samples = BTreeMap::<AmbiguityId, Vec<f64>>::new();

    for epoch in epochs {
        let reference = epoch
            .observations
            .iter()
            .find(|obs| obs.satellite_id == reference_satellite_id)
            .ok_or_else(|| {
                WideLaneError::ReferenceSatelliteMissing(reference_satellite_id.to_string())
            })?;
        let reference = dual_single_difference(reference)?;

        for observation in epoch
            .observations
            .iter()
            .filter(|obs| obs.satellite_id != reference_satellite_id)
        {
            let sample = dual_wide_lane_double_difference(observation, &reference)?;
            samples
                .entry(sample.ambiguity_id)
                .or_default()
                .push(sample.cycles);
        }
    }

    let mut fixed = BTreeMap::new();
    for (ambiguity_id, cycles) in samples {
        match estimate_wide_lane_integer(ambiguity_id.as_str(), &cycles, options) {
            Ok(value) => {
                fixed.insert(ambiguity_id.into_string(), value);
            }
            Err(WideLaneError::TooFewWideLaneEpochs { .. }) if options.skip_short_fragments => {}
            Err(err) => return Err(err),
        }
    }
    Ok(fixed)
}

/// Build ionosphere-free single-observable epochs and the corresponding
/// narrow-lane ambiguity wavelength/offset maps.
pub fn build_ionosphere_free_baseline_epochs(
    epochs: &[DualIonosphereFreeEpoch],
    reference_satellite_id: &str,
    wide_lane_cycles: &BTreeMap<String, i64>,
) -> Result<IonosphereFreeBaselineResult, IonosphereFreeBaselineError> {
    validate_ionosphere_free_epochs(epochs)?;
    let params = dual_narrow_lane_params(epochs, reference_satellite_id, wide_lane_cycles)?;
    let mut if_epochs = Vec::new();

    for (epoch_index, epoch) in epochs.iter().enumerate() {
        let keep_sats =
            dual_ionosphere_free_keep_sats(epoch, reference_satellite_id, wide_lane_cycles);
        if keep_sats.len() < 2 {
            continue;
        }

        if_epochs.push(IonosphereFreeBaselineEpoch {
            epoch_index,
            base_observations: dual_ionosphere_free_observations(
                epoch,
                &keep_sats,
                Receiver::Base,
            )?,
            rover_observations: dual_ionosphere_free_observations(
                epoch,
                &keep_sats,
                Receiver::Rover,
            )?,
            satellite_ids: keep_sats,
        });
    }

    if if_epochs.is_empty() {
        return Err(IonosphereFreeBaselineError::NoEpochs);
    }

    Ok(IonosphereFreeBaselineResult {
        epochs: if_epochs,
        wavelengths_m: params
            .iter()
            .map(|(id, param)| (id.as_str().to_string(), param.wavelength_m))
            .collect(),
        offsets_m: params
            .into_iter()
            .map(|(id, param)| (id.into_string(), param.offset_m))
            .collect(),
    })
}

/// Build IF/narrow-lane baseline epochs from normalized dual-frequency data.
///
/// This composes the language-independent dual-frequency setup that Sidereon used
/// to perform in Elixir: receiver geodetic conversion from the base plus
/// initial baseline, RTKLIB-style standard-atmosphere meteorology, per-satellite
/// geodetic elevation, optional slant troposphere subtraction, then the existing
/// IF/narrow-lane conversion.
pub fn prepare_ionosphere_free_baseline_epochs(
    base_m: [f64; 3],
    initial_baseline_m: [f64; 3],
    epochs: &[DualIonosphereFreeSetupEpoch],
    reference_satellite_id: &str,
    wide_lane_cycles: &BTreeMap<String, i64>,
    apply_troposphere: bool,
) -> Result<IonosphereFreeBaselineResult, IonosphereFreeBaselineError> {
    validate_ionosphere_free_setup_epochs(base_m, initial_baseline_m, epochs, apply_troposphere)?;
    let tropo = apply_troposphere
        .then(|| dual_tropo_config(base_m, initial_baseline_m))
        .transpose()?;
    let if_epochs = epochs
        .iter()
        .map(|epoch| dual_setup_ionosphere_free_epoch(epoch, tropo.as_ref()))
        .collect::<Result<Vec<_>, _>>()?;

    build_ionosphere_free_baseline_epochs(&if_epochs, reference_satellite_id, wide_lane_cycles)
}

fn observations_by_satellite(
    observations: &[Observation],
) -> Result<BTreeMap<String, Observation>, DoubleDifferenceError> {
    let mut by_sat = BTreeMap::new();
    for observation in observations {
        validate_double_difference_observation(observation)?;
        if by_sat
            .insert(observation.satellite_id.clone(), observation.clone())
            .is_some()
        {
            return Err(DoubleDifferenceError::DuplicateObservation(
                observation.satellite_id.clone(),
            ));
        }
    }
    Ok(by_sat)
}

fn validate_double_difference_observation(
    observation: &Observation,
) -> Result<(), DoubleDifferenceError> {
    validate::finite(observation.code_m, "rtk observation code_m")
        .map_err(double_difference_invalid_input)?;
    validate::finite(observation.phase_m, "rtk observation phase_m")
        .map_err(double_difference_invalid_input)?;
    Ok(())
}

fn finite_double_difference_value(
    value: f64,
    field: &'static str,
) -> Result<f64, DoubleDifferenceError> {
    validate::finite(value, field).map_err(double_difference_invalid_input)
}

fn double_difference_invalid_input(error: validate::FieldError) -> DoubleDifferenceError {
    DoubleDifferenceError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn invalid_double_difference_input(
    field: &'static str,
    reason: &'static str,
) -> DoubleDifferenceError {
    DoubleDifferenceError::InvalidInput { field, reason }
}

fn common_observations(
    base: &BTreeMap<String, Observation>,
    rover: &BTreeMap<String, Observation>,
) -> Result<(Vec<String>, Vec<String>), DoubleDifferenceError> {
    let base_sats = base.keys().cloned().collect::<BTreeSet<_>>();
    let rover_sats = rover.keys().cloned().collect::<BTreeSet<_>>();
    let common = base_sats
        .intersection(&rover_sats)
        .cloned()
        .collect::<Vec<_>>();

    if common.len() < 2 {
        return Err(DoubleDifferenceError::TooFewCommonSatellites {
            count: common.len(),
            minimum: 2,
        });
    }

    let common_set = common.iter().cloned().collect::<BTreeSet<_>>();
    let dropped = base_sats
        .union(&rover_sats)
        .filter(|sat| !common_set.contains(*sat))
        .cloned()
        .collect();
    Ok((common, dropped))
}

fn reference_satellites(
    common: &[String],
    selection: &ReferenceSelection,
) -> Result<BTreeMap<String, String>, DoubleDifferenceError> {
    let mut common_by_system = BTreeMap::<String, Vec<String>>::new();
    for sat in common {
        common_by_system
            .entry(satellite_system(sat))
            .or_default()
            .push(sat.clone());
    }
    let systems = common_by_system.keys().cloned().collect::<Vec<_>>();

    match selection {
        ReferenceSelection::Auto => Ok(common_by_system
            .into_iter()
            .map(|(system, sats)| (system, sats[0].clone()))
            .collect()),
        ReferenceSelection::Satellite(sat) => match systems.as_slice() {
            [system] => {
                if common_by_system
                    .get(system)
                    .is_some_and(|sats| sats.contains(sat))
                {
                    Ok(BTreeMap::from([(system.clone(), sat.clone())]))
                } else {
                    Err(DoubleDifferenceError::ReferenceSatelliteMissing(
                        sat.clone(),
                    ))
                }
            }
            _ => Err(DoubleDifferenceError::ReferenceSatelliteSingleSystem(
                sat.clone(),
            )),
        },
        ReferenceSelection::PerSystem(refs) => {
            let mut out = BTreeMap::new();
            for system in systems {
                let Some(sat) = refs.get(&system) else {
                    return Err(DoubleDifferenceError::ReferenceSatelliteMissingSystem(
                        system,
                    ));
                };
                if common_by_system
                    .get(&system)
                    .is_some_and(|sats| sats.contains(sat))
                {
                    out.insert(system, sat.clone());
                } else {
                    return Err(DoubleDifferenceError::ReferenceSatelliteMissing(
                        sat.clone(),
                    ));
                }
            }
            Ok(out)
        }
    }
}

fn ensure_non_reference_satellites(
    common: &[String],
    refs: &BTreeMap<String, String>,
) -> Result<(), DoubleDifferenceError> {
    for (system, reference_sat) in refs {
        let common_count = common
            .iter()
            .filter(|sat| satellite_system(sat) == system.as_str())
            .count();
        let non_reference_count = common
            .iter()
            .filter(|sat| {
                satellite_system(sat) == system.as_str() && sat.as_str() != reference_sat.as_str()
            })
            .count();
        if non_reference_count == 0 {
            return Err(DoubleDifferenceError::TooFewCommonSatellites {
                count: common_count,
                minimum: 2,
            });
        }
    }
    Ok(())
}

fn baseline_all_satellites(epochs: &[BaselineReferenceEpoch]) -> Vec<String> {
    epochs
        .iter()
        .flat_map(|epoch| epoch.available_satellite_ids.iter().cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn baseline_common_by_system(
    epochs: &[BaselineReferenceEpoch],
) -> BTreeMap<String, BTreeSet<String>> {
    let mut common_by_system = BTreeMap::<String, BTreeSet<String>>::new();

    for epoch in epochs {
        let mut sats_by_system = BTreeMap::<String, BTreeSet<String>>::new();
        for sat in &epoch.available_satellite_ids {
            sats_by_system
                .entry(satellite_system(sat))
                .or_default()
                .insert(sat.clone());
        }

        for (system, sats) in sats_by_system {
            common_by_system
                .entry(system)
                .and_modify(|common| {
                    *common = common.intersection(&sats).cloned().collect();
                })
                .or_insert(sats);
        }
    }

    common_by_system
}

fn baseline_epochs_for_system<'a>(
    epochs: &'a [BaselineReferenceEpoch],
    system: &str,
) -> Vec<&'a BaselineReferenceEpoch> {
    epochs
        .iter()
        .filter(|epoch| {
            epoch
                .available_satellite_ids
                .iter()
                .any(|sat| satellite_system(sat) == system)
        })
        .collect()
}

fn highest_elevation_reference(
    base_m: [f64; 3],
    epochs: &[&BaselineReferenceEpoch],
    common: &BTreeSet<String>,
) -> Result<String, DoubleDifferenceError> {
    let mut scores = common
        .iter()
        .map(|sat| average_elevation_score(base_m, epochs, sat).map(|score| (sat.clone(), score)))
        .collect::<Result<Vec<_>, _>>()?;

    scores.sort_by(|(sat_a, score_a), (sat_b, score_b)| {
        score_b
            .partial_cmp(score_a)
            .unwrap_or(Ordering::Equal)
            .then_with(|| sat_a.cmp(sat_b))
    });

    Ok(scores[0].0.clone())
}

fn average_elevation_score(
    base_m: [f64; 3],
    epochs: &[&BaselineReferenceEpoch],
    sat: &str,
) -> Result<f64, DoubleDifferenceError> {
    let up = local_up(base_m);
    let mut sum = 0.0;

    for epoch in epochs {
        let sat_pos = *baseline_satellite_position(epoch, sat)?;
        sum += elevation_score_with_up(base_m, up, sat_pos)?;
    }

    Ok(sum / epochs.len() as f64)
}

fn elevation_score_with_up(
    base_m: [f64; 3],
    up: [f64; 3],
    sat_pos_m: [f64; 3],
) -> Result<f64, DoubleDifferenceError> {
    validate::finite_vec3(up, "rtk local up").map_err(double_difference_invalid_input)?;
    let (los, n) = rtk_line_of_sight(base_m, sat_pos_m)?;
    let inv = 1.0 / n;
    let los = [los[0] * inv, los[1] * inv, los[2] * inv];
    let score = dot3(los, up);
    validate_elevation_score(score)
}

fn rtk_line_of_sight(
    base_m: [f64; 3],
    sat_pos_m: [f64; 3],
) -> Result<([f64; 3], f64), DoubleDifferenceError> {
    validate::finite_vec3(base_m, "rtk base position_m")
        .map_err(double_difference_invalid_input)?;
    validate::finite_vec3(sat_pos_m, "rtk satellite position_m")
        .map_err(double_difference_invalid_input)?;
    let los = sub3(sat_pos_m, base_m);
    validate::finite_vec3(los, "rtk line of sight_m").map_err(double_difference_invalid_input)?;
    let n = norm3(los);
    if !n.is_finite() {
        return Err(invalid_double_difference_input(
            "rtk line of sight range_m",
            "out of range",
        ));
    }
    if n <= 0.0 {
        return Err(invalid_double_difference_input(
            "rtk line of sight_m",
            "degenerate geometry",
        ));
    }
    Ok((los, n))
}

fn validate_elevation_score(score: f64) -> Result<f64, DoubleDifferenceError> {
    validate::finite(score, "rtk elevation score").map_err(double_difference_invalid_input)?;
    if !(-1.0 - 1.0e-12..=1.0 + 1.0e-12).contains(&score) {
        return Err(invalid_double_difference_input(
            "rtk elevation score",
            "out of range",
        ));
    }
    Ok(score.clamp(-1.0, 1.0))
}

fn local_up(base_m: [f64; 3]) -> [f64; 3] {
    crate::estimation::substrate::frames::local_up(
        crate::estimation::recipe::FrameRecipe::GeocentricUpRtkReference,
        base_m,
    )
}

fn validate_wide_lane_options(options: WideLaneOptions) -> Result<(), WideLaneError> {
    if options.min_epochs == 0 {
        return Err(invalid_wide_lane_input(
            "rtk wide lane min_epochs",
            "not positive",
        ));
    }
    validate::finite_positive(options.tolerance_cycles, "rtk wide lane tolerance_cycles")
        .map_err(wide_lane_invalid_input)?;
    Ok(())
}

fn validate_wide_lane_epochs(epochs: &[DualEpoch]) -> Result<(), WideLaneError> {
    for epoch in epochs {
        for observation in &epoch.observations {
            validate_wide_lane_observation(&observation.base)?;
            validate_wide_lane_observation(&observation.rover)?;
        }
    }
    Ok(())
}

fn validate_wide_lane_observation(observation: &DualObservation) -> Result<(), WideLaneError> {
    validate::finite(observation.p1_m, "rtk wide lane p1_m").map_err(wide_lane_invalid_input)?;
    validate::finite(observation.p2_m, "rtk wide lane p2_m").map_err(wide_lane_invalid_input)?;
    validate::finite(observation.phi1_cycles, "rtk wide lane phi1_cycles")
        .map_err(wide_lane_invalid_input)?;
    validate::finite(observation.phi2_cycles, "rtk wide lane phi2_cycles")
        .map_err(wide_lane_invalid_input)?;
    validate::finite_positive(observation.f1_hz, "rtk wide lane f1_hz")
        .map_err(wide_lane_invalid_input)?;
    validate::finite_positive(observation.f2_hz, "rtk wide lane f2_hz")
        .map_err(wide_lane_invalid_input)?;
    Ok(())
}

fn finite_wide_lane_value(value: f64, field: &'static str) -> Result<f64, WideLaneError> {
    validate::finite(value, field).map_err(wide_lane_invalid_input)
}

fn wide_lane_invalid_input(error: validate::FieldError) -> WideLaneError {
    WideLaneError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn invalid_wide_lane_input(field: &'static str, reason: &'static str) -> WideLaneError {
    WideLaneError::InvalidInput { field, reason }
}

fn dual_single_difference(
    observation: &DualSatelliteObservation,
) -> Result<DualSingleDifference, WideLaneError> {
    let rover_wide_lane = dual_observation_wide_lane_cycles(&observation.rover).map_err(|err| {
        WideLaneError::WideLaneFailed {
            satellite_id: observation.satellite_id.clone(),
            reason: err,
        }
    })?;
    let base_wide_lane = dual_observation_wide_lane_cycles(&observation.base).map_err(|err| {
        WideLaneError::WideLaneFailed {
            satellite_id: observation.satellite_id.clone(),
            reason: err,
        }
    })?;

    let wide_lane_cycles = finite_wide_lane_value(
        rover_wide_lane - base_wide_lane,
        "rtk wide lane single difference cycles",
    )?;

    Ok(DualSingleDifference {
        satellite_id: observation.satellite_id.clone(),
        ambiguity_id: single_difference_dual_ambiguity_id(
            &observation.satellite_id,
            &observation.base,
            &observation.rover,
        ),
        wide_lane_cycles,
    })
}

fn dual_wide_lane_double_difference(
    observation: &DualSatelliteObservation,
    reference: &DualSingleDifference,
) -> Result<WideLaneSample, WideLaneError> {
    let sd = dual_single_difference(observation)?;
    let cycles = finite_wide_lane_value(
        sd.wide_lane_cycles - reference.wide_lane_cycles,
        "rtk wide lane double difference cycles",
    )?;
    Ok(WideLaneSample {
        ambiguity_id: double_difference_dual_ambiguity_id(
            &observation.satellite_id,
            &sd.ambiguity_id,
            reference,
        ),
        cycles,
    })
}

fn dual_observation_wide_lane_cycles(
    observation: &DualObservation,
) -> Result<f64, CarrierPhaseError> {
    crate::carrier_phase::wide_lane_cycles(
        observation.phi1_cycles,
        observation.phi2_cycles,
        observation.p1_m,
        observation.p2_m,
        observation.f1_hz,
        observation.f2_hz,
    )
}

fn estimate_wide_lane_integer(
    ambiguity_id: &str,
    cycles: &[f64],
    options: WideLaneOptions,
) -> Result<i64, WideLaneError> {
    ambiguity::estimate_wide_lane_integer(cycles, options.min_epochs, options.tolerance_cycles)
        .map_err(|err| match err {
            ambiguity::WideLaneEstimateError::TooFewEpochs { count, minimum } => {
                WideLaneError::TooFewWideLaneEpochs {
                    ambiguity_id: ambiguity_id.to_string(),
                    count,
                    minimum,
                }
            }
            ambiguity::WideLaneEstimateError::NotInteger {
                mean_cycles,
                fixed_cycles,
            } => WideLaneError::WideLaneNotInteger {
                ambiguity_id: ambiguity_id.to_string(),
                mean_cycles,
                fixed_cycles,
            },
        })
}

fn validate_ionosphere_free_epochs(
    epochs: &[DualIonosphereFreeEpoch],
) -> Result<(), IonosphereFreeBaselineError> {
    for epoch in epochs {
        for observation in &epoch.observations {
            validate_ionosphere_free_observation(&observation.base)?;
            validate_ionosphere_free_observation(&observation.rover)?;
        }
    }
    Ok(())
}

fn validate_ionosphere_free_observation(
    observation: &DualIonosphereFreeObservation,
) -> Result<(), IonosphereFreeBaselineError> {
    validate::finite(observation.p1_m, "rtk if p1_m").map_err(ionosphere_free_invalid_input)?;
    validate::finite(observation.p2_m, "rtk if p2_m").map_err(ionosphere_free_invalid_input)?;
    validate::finite(observation.phi1_cycles, "rtk if phi1_cycles")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite(observation.phi2_cycles, "rtk if phi2_cycles")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite_positive(observation.f1_hz, "rtk if f1_hz")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite_positive(observation.f2_hz, "rtk if f2_hz")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite(observation.tropo_m, "rtk if tropo_m")
        .map_err(ionosphere_free_invalid_input)?;
    Ok(())
}

fn validate_ionosphere_free_setup_epochs(
    base_m: [f64; 3],
    initial_baseline_m: [f64; 3],
    epochs: &[DualIonosphereFreeSetupEpoch],
    apply_troposphere: bool,
) -> Result<(), IonosphereFreeBaselineError> {
    validate_tropo_receiver_position(base_m, "rtk tropo base position_m")?;
    validate::finite_vec3(initial_baseline_m, "rtk tropo initial_baseline_m")
        .map_err(ionosphere_free_invalid_input)?;
    let rover_m = [
        base_m[0] + initial_baseline_m[0],
        base_m[1] + initial_baseline_m[1],
        base_m[2] + initial_baseline_m[2],
    ];
    validate_tropo_receiver_position(rover_m, "rtk tropo rover position_m")?;

    for epoch in epochs {
        validate::finite(epoch.jd_whole, "rtk if setup jd_whole")
            .map_err(ionosphere_free_invalid_input)?;
        validate::finite_in_range(epoch.jd_fraction, -1.0, 1.0, "rtk if setup jd_fraction")
            .map_err(ionosphere_free_invalid_input)?;
        for observation in &epoch.observations {
            validate_setup_dual_observation(&observation.base)?;
            validate_setup_dual_observation(&observation.rover)?;
            if apply_troposphere {
                let base_sat = *validate::present(
                    epoch
                        .base_satellite_positions_m
                        .get(&observation.satellite_id),
                    "rtk tropo base satellite position_m",
                )
                .map_err(ionosphere_free_invalid_input)?;
                let rover_sat = *validate::present(
                    epoch
                        .rover_satellite_positions_m
                        .get(&observation.satellite_id),
                    "rtk tropo rover satellite position_m",
                )
                .map_err(ionosphere_free_invalid_input)?;
                validate_tropo_satellite_geometry(
                    base_m,
                    base_sat,
                    "rtk tropo base satellite position_m",
                )?;
                validate_tropo_satellite_geometry(
                    rover_m,
                    rover_sat,
                    "rtk tropo rover satellite position_m",
                )?;
            }
        }
    }
    Ok(())
}

fn validate_setup_dual_observation(
    observation: &DualObservation,
) -> Result<(), IonosphereFreeBaselineError> {
    validate::finite(observation.p1_m, "rtk if setup p1_m")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite(observation.p2_m, "rtk if setup p2_m")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite(observation.phi1_cycles, "rtk if setup phi1_cycles")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite(observation.phi2_cycles, "rtk if setup phi2_cycles")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite_positive(observation.f1_hz, "rtk if setup f1_hz")
        .map_err(ionosphere_free_invalid_input)?;
    validate::finite_positive(observation.f2_hz, "rtk if setup f2_hz")
        .map_err(ionosphere_free_invalid_input)?;
    Ok(())
}

fn validate_tropo_receiver_position(
    position_m: [f64; 3],
    field: &'static str,
) -> Result<(), IonosphereFreeBaselineError> {
    validate::finite_vec3(position_m, field).map_err(ionosphere_free_invalid_input)?;
    let norm = norm3(position_m);
    if !norm.is_finite() {
        return Err(invalid_ionosphere_free_input(field, "out of range"));
    }
    if norm <= 0.0 {
        return Err(invalid_ionosphere_free_input(field, "degenerate geometry"));
    }
    Ok(())
}

fn validate_tropo_satellite_geometry(
    receiver_m: [f64; 3],
    sat_pos_m: [f64; 3],
    field: &'static str,
) -> Result<(), IonosphereFreeBaselineError> {
    validate::finite_vec3(sat_pos_m, field).map_err(ionosphere_free_invalid_input)?;
    let los = sub3(sat_pos_m, receiver_m);
    validate::finite_vec3(los, "rtk tropo line of sight_m")
        .map_err(ionosphere_free_invalid_input)?;
    let range = norm3(los);
    if !range.is_finite() {
        return Err(invalid_ionosphere_free_input(
            "rtk tropo line of sight range_m",
            "out of range",
        ));
    }
    if range <= 0.0 {
        return Err(invalid_ionosphere_free_input(
            "rtk tropo line of sight_m",
            "degenerate geometry",
        ));
    }
    Ok(())
}

fn finite_ionosphere_free_value(
    value: f64,
    field: &'static str,
) -> Result<f64, IonosphereFreeBaselineError> {
    validate::finite(value, field).map_err(ionosphere_free_invalid_input)
}

fn ionosphere_free_invalid_input(error: validate::FieldError) -> IonosphereFreeBaselineError {
    IonosphereFreeBaselineError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn invalid_ionosphere_free_input(
    field: &'static str,
    reason: &'static str,
) -> IonosphereFreeBaselineError {
    IonosphereFreeBaselineError::InvalidInput { field, reason }
}

fn dual_narrow_lane_params(
    epochs: &[DualIonosphereFreeEpoch],
    reference_satellite_id: &str,
    wide_lane_cycles: &BTreeMap<String, i64>,
) -> Result<BTreeMap<AmbiguityId, NarrowLaneParams>, IonosphereFreeBaselineError> {
    let mut params = BTreeMap::new();
    for epoch in epochs {
        let Some(ref_sd) = dual_if_single_difference_ambiguity(epoch, reference_satellite_id)
        else {
            continue;
        };
        for sat in dual_if_epoch_common_sats(epoch)
            .into_iter()
            .filter(|sat| sat != reference_satellite_id)
        {
            let Some(ambiguity_id) = dual_if_wide_lane_ambiguity_id(epoch, &sat, &ref_sd) else {
                continue;
            };
            let Some(&wide_lane) = wide_lane_cycles.get(ambiguity_id.as_str()) else {
                continue;
            };
            let param = dual_narrow_lane_param_from_epoch(
                epoch,
                &sat,
                reference_satellite_id,
                ambiguity_id.as_str(),
                wide_lane as f64,
            )?;
            ensure_consistent_dual_narrow_lane_params(
                ambiguity_id.as_str(),
                param,
                params.get(&ambiguity_id).copied(),
            )?;
            params.entry(ambiguity_id).or_insert(param);
        }
    }
    Ok(params)
}

#[derive(Debug, Clone, Copy)]
enum Receiver {
    Base,
    Rover,
}

fn dual_narrow_lane_param_from_epoch(
    epoch: &DualIonosphereFreeEpoch,
    sat: &str,
    reference_satellite_id: &str,
    ambiguity_id: &str,
    wide_lane_cycles: f64,
) -> Result<NarrowLaneParams, IonosphereFreeBaselineError> {
    let sat_obs = dual_if_satellite(epoch, sat).expect("satellite from epoch common set");
    let ref_obs =
        dual_if_satellite(epoch, reference_satellite_id).expect("reference from epoch common set");
    ensure_same_dual_frequencies(
        ambiguity_id,
        [&sat_obs.base, &sat_obs.rover, &ref_obs.base, &ref_obs.rover],
    )?;
    dual_narrow_lane_param(sat_obs.base.f1_hz, sat_obs.base.f2_hz, wide_lane_cycles)
}

fn ensure_same_dual_frequencies(
    ambiguity_id: &str,
    observations: [&DualIonosphereFreeObservation; 4],
) -> Result<(), IonosphereFreeBaselineError> {
    let first = observations[0];
    if observations[1..].iter().all(|obs| {
        ambiguity::frequencies_match(obs.f1_hz, first.f1_hz)
            && ambiguity::frequencies_match(obs.f2_hz, first.f2_hz)
    }) {
        Ok(())
    } else {
        Err(IonosphereFreeBaselineError::InconsistentFrequencies(
            ambiguity_id.to_string(),
        ))
    }
}

fn dual_narrow_lane_param(
    f1_hz: f64,
    f2_hz: f64,
    wide_lane_cycles: f64,
) -> Result<NarrowLaneParams, IonosphereFreeBaselineError> {
    ambiguity::narrow_lane_params(f1_hz, f2_hz, wide_lane_cycles)
        .map_err(IonosphereFreeBaselineError::NarrowLaneFailed)
}

fn ensure_consistent_dual_narrow_lane_params(
    ambiguity_id: &str,
    params: NarrowLaneParams,
    prev: Option<NarrowLaneParams>,
) -> Result<(), IonosphereFreeBaselineError> {
    if prev.is_none_or(|prev| {
        ambiguity::frequencies_match(params.f1_hz, prev.f1_hz)
            && ambiguity::frequencies_match(params.f2_hz, prev.f2_hz)
    }) {
        Ok(())
    } else {
        Err(IonosphereFreeBaselineError::InconsistentFrequencies(
            ambiguity_id.to_string(),
        ))
    }
}

fn dual_ionosphere_free_keep_sats(
    epoch: &DualIonosphereFreeEpoch,
    reference_satellite_id: &str,
    wide_lane_cycles: &BTreeMap<String, i64>,
) -> Vec<String> {
    let common = dual_if_epoch_common_sats(epoch);
    if !common.iter().any(|sat| sat == reference_satellite_id) {
        return Vec::new();
    }
    let Some(ref_sd) = dual_if_single_difference_ambiguity(epoch, reference_satellite_id) else {
        return Vec::new();
    };
    let kept_nonrefs = common
        .into_iter()
        .filter(|sat| sat != reference_satellite_id)
        .filter(|sat| {
            dual_if_wide_lane_ambiguity_id(epoch, sat, &ref_sd)
                .is_some_and(|id| wide_lane_cycles.contains_key(id.as_str()))
        })
        .collect::<Vec<_>>();
    if kept_nonrefs.is_empty() {
        Vec::new()
    } else {
        let mut out = Vec::with_capacity(kept_nonrefs.len() + 1);
        out.push(reference_satellite_id.to_string());
        out.extend(kept_nonrefs);
        out
    }
}

fn dual_ionosphere_free_observations(
    epoch: &DualIonosphereFreeEpoch,
    keep_sats: &[String],
    receiver: Receiver,
) -> Result<Vec<Observation>, IonosphereFreeBaselineError> {
    let keep = keep_sats.iter().collect::<BTreeSet<_>>();
    let mut out = Vec::new();
    for sat_obs in &epoch.observations {
        if keep.contains(&sat_obs.satellite_id) {
            let obs = match receiver {
                Receiver::Base => &sat_obs.base,
                Receiver::Rover => &sat_obs.rover,
            };
            out.push(dual_ionosphere_free_observation(
                &sat_obs.satellite_id,
                obs,
            )?);
        }
    }
    Ok(out)
}

fn dual_ionosphere_free_observation(
    satellite_id: &str,
    obs: &DualIonosphereFreeObservation,
) -> Result<Observation, IonosphereFreeBaselineError> {
    let code_m = combinations::ionosphere_free(obs.p1_m, obs.p2_m, obs.f1_hz, obs.f2_hz).map_err(
        |reason| IonosphereFreeBaselineError::IonosphereFreeFailed {
            satellite_id: satellite_id.to_string(),
            reason,
        },
    )?;
    let phase_m = combinations::ionosphere_free_phase_cycles(
        obs.phi1_cycles,
        obs.phi2_cycles,
        obs.f1_hz,
        obs.f2_hz,
    )
    .map_err(|reason| IonosphereFreeBaselineError::IonosphereFreeFailed {
        satellite_id: satellite_id.to_string(),
        reason,
    })?;
    let code_m = finite_ionosphere_free_value(code_m - obs.tropo_m, "rtk if code_m")?;
    let phase_m = finite_ionosphere_free_value(phase_m - obs.tropo_m, "rtk if phase_m")?;
    Ok(Observation {
        satellite_id: satellite_id.to_string(),
        ambiguity_id: obs.ambiguity_id.clone(),
        code_m,
        phase_m,
    })
}

fn dual_setup_ionosphere_free_epoch(
    epoch: &DualIonosphereFreeSetupEpoch,
    tropo: Option<&DualTropoConfig>,
) -> Result<DualIonosphereFreeEpoch, IonosphereFreeBaselineError> {
    let observations = epoch
        .observations
        .iter()
        .map(|obs| {
            let base_tropo_m = dual_slant_tropo_m(
                tropo,
                Receiver::Base,
                epoch
                    .base_satellite_positions_m
                    .get(&obs.satellite_id)
                    .copied(),
                epoch.jd_whole,
                epoch.jd_fraction,
            )?;
            let rover_tropo_m = dual_slant_tropo_m(
                tropo,
                Receiver::Rover,
                epoch
                    .rover_satellite_positions_m
                    .get(&obs.satellite_id)
                    .copied(),
                epoch.jd_whole,
                epoch.jd_fraction,
            )?;

            Ok(DualIonosphereFreeSatelliteObservation {
                satellite_id: obs.satellite_id.clone(),
                base: dual_if_observation_from_dual(&obs.base, base_tropo_m),
                rover: dual_if_observation_from_dual(&obs.rover, rover_tropo_m),
            })
        })
        .collect::<Result<Vec<_>, IonosphereFreeBaselineError>>()?;
    Ok(DualIonosphereFreeEpoch { observations })
}

fn dual_if_observation_from_dual(
    obs: &DualObservation,
    tropo_m: f64,
) -> DualIonosphereFreeObservation {
    DualIonosphereFreeObservation {
        ambiguity_id: obs.ambiguity_id.clone(),
        p1_m: obs.p1_m,
        p2_m: obs.p2_m,
        phi1_cycles: obs.phi1_cycles,
        phi2_cycles: obs.phi2_cycles,
        f1_hz: obs.f1_hz,
        f2_hz: obs.f2_hz,
        tropo_m,
    }
}

fn dual_slant_tropo_m(
    tropo: Option<&DualTropoConfig>,
    receiver: Receiver,
    sat_pos_m: Option<[f64; 3]>,
    jd_whole: f64,
    jd_fraction: f64,
) -> Result<f64, IonosphereFreeBaselineError> {
    let (Some(tropo), Some(sat_pos_m)) = (tropo, sat_pos_m) else {
        return Ok(0.0);
    };
    let receiver = match receiver {
        Receiver::Base => tropo.base,
        Receiver::Rover => tropo.rover,
    };
    let elevation_rad = geodetic_elevation_rad(receiver.geodetic, receiver.position_m, sat_pos_m);
    let met = Met::standard(receiver.geodetic.height_m, 0.0)
        .map_err(map_ionosphere_free_setup_tropo_error)?;
    let split = JulianDateSplit::new(jd_whole, jd_fraction)
        .map_err(map_ionosphere_free_setup_julian_split_error)?;
    let epoch = Instant::from_julian_date(TimeScale::Gpst, split);
    tropo_slant(elevation_rad, receiver.geodetic, met, epoch)
        .map_err(map_ionosphere_free_setup_tropo_error)
}

fn map_ionosphere_free_setup_julian_split_error(
    error: crate::astro::time::model::TimeModelError,
) -> IonosphereFreeBaselineError {
    let crate::astro::time::model::TimeModelError::InvalidInput { field, reason } = error;
    let field = match field {
        "jd_whole" => "rtk if setup jd_whole",
        "fraction" => "rtk if setup jd_fraction",
        _ => "rtk if setup JulianDateSplit",
    };
    invalid_ionosphere_free_input(field, reason)
}

fn map_ionosphere_free_setup_tropo_error(
    error: crate::error::Error,
) -> IonosphereFreeBaselineError {
    match error {
        crate::error::Error::InvalidInput(message) => {
            let (field, reason) = map_ionosphere_free_setup_tropo_message(&message);
            invalid_ionosphere_free_input(field, reason)
        }
        _ => invalid_ionosphere_free_input("rtk if setup tropo", "invalid input"),
    }
}

fn map_ionosphere_free_setup_tropo_message(message: &str) -> (&'static str, &'static str) {
    match message {
        "height_m not finite" => ("rtk tropo receiver height_m", "not finite"),
        "relative_humidity not finite" => ("rtk tropo relative_humidity", "not finite"),
        "relative_humidity out of range" => ("rtk tropo relative_humidity", "out of range"),
        "elevation_rad not finite" => ("rtk tropo elevation_rad", "not finite"),
        "elevation_rad out of range" => ("rtk tropo elevation_rad", "out of range"),
        "receiver.lat_rad not finite" => ("rtk tropo receiver.lat_rad", "not finite"),
        "receiver.lat_rad out of range" => ("rtk tropo receiver.lat_rad", "out of range"),
        "receiver.lon_rad not finite" => ("rtk tropo receiver.lon_rad", "not finite"),
        "receiver.lon_rad out of range" => ("rtk tropo receiver.lon_rad", "out of range"),
        "receiver.height_m not finite" => ("rtk tropo receiver.height_m", "not finite"),
        "receiver.height_m out of range" => ("rtk tropo receiver.height_m", "out of range"),
        "pressure_hpa not finite" => ("rtk tropo pressure_hpa", "not finite"),
        "pressure_hpa not positive" => ("rtk tropo pressure_hpa", "not positive"),
        "temperature_k not finite" => ("rtk tropo temperature_k", "not finite"),
        "temperature_k not positive" => ("rtk tropo temperature_k", "not positive"),
        "epoch.jd_whole not finite" => ("rtk if setup jd_whole", "not finite"),
        "epoch.fraction not finite" => ("rtk if setup jd_fraction", "not finite"),
        "epoch.fraction out of range" => ("rtk if setup jd_fraction", "out of range"),
        _ => ("rtk if setup tropo", "invalid input"),
    }
}

fn dual_tropo_config(
    base_m: [f64; 3],
    initial_baseline_m: [f64; 3],
) -> Result<DualTropoConfig, IonosphereFreeBaselineError> {
    let rover_m = [
        base_m[0] + initial_baseline_m[0],
        base_m[1] + initial_baseline_m[1],
        base_m[2] + initial_baseline_m[2],
    ];
    Ok(DualTropoConfig {
        base: DualTropoReceiver {
            position_m: base_m,
            geodetic: receiver_geodetic_for_rtk_tropo(base_m, "rtk tropo base position_m")?,
        },
        rover: DualTropoReceiver {
            position_m: rover_m,
            geodetic: receiver_geodetic_for_rtk_tropo(rover_m, "rtk tropo rover position_m")?,
        },
    })
}

fn receiver_geodetic_for_rtk_tropo(
    position_m: [f64; 3],
    field: &'static str,
) -> Result<Wgs84Geodetic, IonosphereFreeBaselineError> {
    let (lat_deg, lon_deg, height_km) = itrs_to_geodetic_compute(
        position_m[0] / KM_TO_M,
        position_m[1] / KM_TO_M,
        position_m[2] / KM_TO_M,
    )
    .map_err(|_| invalid_ionosphere_free_input(field, "invalid geodetic"))?;
    Wgs84Geodetic::new(
        lat_deg * DEG_TO_RAD,
        normalize_geodetic_lon_rad(lon_deg * DEG_TO_RAD),
        (height_km * KM_TO_M).max(0.0),
    )
    .map_err(|_| invalid_ionosphere_free_input(field, "invalid geodetic"))
}

fn geodetic_elevation_rad(
    geodetic: Wgs84Geodetic,
    receiver_m: [f64; 3],
    sat_pos_m: [f64; 3],
) -> f64 {
    let dx = sat_pos_m[0] - receiver_m[0];
    let dy = sat_pos_m[1] - receiver_m[1];
    let dz = sat_pos_m[2] - receiver_m[2];
    let range = (dx * dx + dy * dy + dz * dz).sqrt();

    if range <= 0.0 {
        0.0
    } else {
        let lat = geodetic.lat_rad;
        let lon = geodetic.lon_rad;
        let u = lat.cos() * lon.cos() * dx + lat.cos() * lon.sin() * dy + lat.sin() * dz;
        let elevation_deg = (u / range).clamp(-1.0, 1.0).asin() * RAD_TO_DEG;
        elevation_deg * DEG_TO_RAD
    }
}

fn dual_if_satellite<'a>(
    epoch: &'a DualIonosphereFreeEpoch,
    sat: &str,
) -> Option<&'a DualIonosphereFreeSatelliteObservation> {
    epoch
        .observations
        .iter()
        .find(|obs| obs.satellite_id == sat)
}

fn dual_if_epoch_common_sats(epoch: &DualIonosphereFreeEpoch) -> Vec<String> {
    epoch
        .observations
        .iter()
        .map(|obs| obs.satellite_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn dual_if_single_difference_ambiguity(
    epoch: &DualIonosphereFreeEpoch,
    sat: &str,
) -> Option<DualSingleDifference> {
    let obs = dual_if_satellite(epoch, sat)?;
    Some(DualSingleDifference {
        satellite_id: sat.to_string(),
        ambiguity_id: single_difference_if_ambiguity_id(sat, &obs.base, &obs.rover),
        wide_lane_cycles: 0.0,
    })
}

fn dual_if_wide_lane_ambiguity_id(
    epoch: &DualIonosphereFreeEpoch,
    sat: &str,
    ref_sd: &DualSingleDifference,
) -> Option<AmbiguityId> {
    let obs = dual_if_satellite(epoch, sat)?;
    let sat_sd_id = single_difference_if_ambiguity_id(sat, &obs.base, &obs.rover);
    Some(double_difference_dual_ambiguity_id(sat, &sat_sd_id, ref_sd))
}

fn single_difference_if_ambiguity_id(
    sat: &str,
    base_obs: &DualIonosphereFreeObservation,
    rover_obs: &DualIonosphereFreeObservation,
) -> AmbiguityId {
    let token = match (
        base_obs.ambiguity_id.as_str(),
        rover_obs.ambiguity_id.as_str(),
    ) {
        (base_id, rover_id) if base_id == sat && rover_id == sat => sat.to_string(),
        (base_id, rover_id) if base_id == sat => rover_id.to_string(),
        (base_id, rover_id) if rover_id == sat => base_id.to_string(),
        (base_id, rover_id) if base_id == rover_id => base_id.to_string(),
        (base_id, rover_id) => format!("{sat}:base={base_id},rover={rover_id}"),
    };
    AmbiguityId::new(token)
}

fn reference_report(refs: BTreeMap<String, String>) -> ReferenceReport {
    if refs.len() == 1 {
        ReferenceReport::Satellite(refs.into_values().next().expect("single reference"))
    } else {
        ReferenceReport::PerSystem(refs)
    }
}

fn smooth_receiver_code_epochs(
    epochs: &mut [CodeSmoothingEpoch],
    receiver: Receiver,
    hatch_window_cap: usize,
) {
    let mut states = BTreeMap::<String, CodeSmoothingState>::new();

    for epoch in epochs {
        let observations = match receiver {
            Receiver::Base => &mut epoch.base_observations,
            Receiver::Rover => &mut epoch.rover_observations,
        };
        observations.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));

        for observation in observations {
            let state = states.get(&observation.ambiguity_id).copied();
            let (code_m, next_state) =
                smooth_observation_code(observation, state, hatch_window_cap);
            observation.code_m = code_m;
            states.insert(observation.ambiguity_id.clone(), next_state);
        }
    }
}

fn smooth_observation_code(
    observation: &CodeSmoothingObservation,
    state: Option<CodeSmoothingState>,
    hatch_window_cap: usize,
) -> (f64, CodeSmoothingState) {
    if state.is_none() || rtk_lli_set(observation.lli) {
        return (
            observation.code_m,
            CodeSmoothingState {
                p_smooth_m: observation.code_m,
                phase_m: observation.phase_m,
                window: 1,
            },
        );
    }

    let state = state.expect("checked above");
    let window = (state.window + 1).min(hatch_window_cap);
    let n = window as f64;
    let p_smooth_m = observation.code_m / n
        + (n - 1.0) / n * (state.p_smooth_m + (observation.phase_m - state.phase_m));
    (
        p_smooth_m,
        CodeSmoothingState {
            p_smooth_m,
            phase_m: observation.phase_m,
            window,
        },
    )
}

fn rtk_lli_set(lli: Option<i64>) -> bool {
    lli.is_some_and(|value| (value & 1) == 1)
}

fn validate_cycle_slip_baseline_epochs(
    epochs: &[CycleSlipEpoch],
) -> Result<(), CycleSlipPrepError> {
    for epoch in epochs {
        for observation in &epoch.base_observations {
            validate_cycle_slip_observation(observation)?;
        }
        for observation in &epoch.rover_observations {
            validate_cycle_slip_observation(observation)?;
        }
    }
    Ok(())
}

fn validate_cycle_slip_observation(
    observation: &CycleSlipObservation,
) -> Result<(), CycleSlipPrepError> {
    validate::finite(observation.code_m, "rtk cycle slip code_m")
        .map_err(cycle_slip_invalid_input)?;
    validate::finite(observation.phase_m, "rtk cycle slip phase_m")
        .map_err(cycle_slip_invalid_input)?;
    Ok(())
}

fn validate_cycle_slip_options(options: CycleSlipOptions) -> Result<(), CycleSlipPrepError> {
    validate::finite_positive(options.gf_threshold_m, "rtk cycle slip gf_threshold_m")
        .map_err(cycle_slip_invalid_input)?;
    validate::finite_positive(
        options.mw_threshold_cycles,
        "rtk cycle slip mw_threshold_cycles",
    )
    .map_err(cycle_slip_invalid_input)?;
    validate::finite_positive(options.min_arc_gap_s, "rtk cycle slip min_arc_gap_s")
        .map_err(cycle_slip_invalid_input)?;
    Ok(())
}

fn validate_dual_cycle_slip_baseline_epochs(
    epochs: &[DualCycleSlipEpoch],
) -> Result<(), CycleSlipPrepError> {
    for epoch in epochs {
        if let Some(gap_time_s) = epoch.gap_time_s {
            validate::finite(gap_time_s, "rtk cycle slip gap_time_s")
                .map_err(cycle_slip_invalid_input)?;
        }
        for observation in &epoch.base_observations {
            validate_dual_cycle_slip_observation(observation)?;
        }
        for observation in &epoch.rover_observations {
            validate_dual_cycle_slip_observation(observation)?;
        }
    }
    Ok(())
}

fn validate_dual_cycle_slip_observation(
    observation: &DualCycleSlipObservation,
) -> Result<(), CycleSlipPrepError> {
    validate::finite(observation.p1_m, "rtk cycle slip p1_m").map_err(cycle_slip_invalid_input)?;
    validate::finite(observation.p2_m, "rtk cycle slip p2_m").map_err(cycle_slip_invalid_input)?;
    validate::finite(observation.phi1_cycles, "rtk cycle slip phi1_cycles")
        .map_err(cycle_slip_invalid_input)?;
    validate::finite(observation.phi2_cycles, "rtk cycle slip phi2_cycles")
        .map_err(cycle_slip_invalid_input)?;
    validate::finite_positive(observation.f1_hz, "rtk cycle slip f1_hz")
        .map_err(cycle_slip_invalid_input)?;
    validate::finite_positive(observation.f2_hz, "rtk cycle slip f2_hz")
        .map_err(cycle_slip_invalid_input)?;
    if (observation.f1_hz - observation.f2_hz).abs() < crate::carrier_phase::FREQ_EPSILON_HZ {
        return Err(invalid_cycle_slip_input(
            "rtk cycle slip frequencies_hz",
            "degenerate frequencies",
        ));
    }
    Ok(())
}

fn cycle_slip_invalid_input(error: validate::FieldError) -> CycleSlipPrepError {
    CycleSlipPrepError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn invalid_cycle_slip_input(field: &'static str, reason: &'static str) -> CycleSlipPrepError {
    CycleSlipPrepError::InvalidInput { field, reason }
}

fn cycle_slip_events(epochs: &[CycleSlipEpoch]) -> Vec<CycleSlipEvent> {
    let mut events = Vec::new();
    for (epoch_index, epoch) in epochs.iter().enumerate() {
        cycle_slip_events_for_receiver(
            CycleSlipReceiver::Base,
            epoch_index,
            &epoch.base_observations,
            &mut events,
        );
        cycle_slip_events_for_receiver(
            CycleSlipReceiver::Rover,
            epoch_index,
            &epoch.rover_observations,
            &mut events,
        );
    }
    events
}

fn cycle_slip_events_for_receiver(
    receiver: CycleSlipReceiver,
    epoch_index: usize,
    observations: &[CycleSlipObservation],
    events: &mut Vec<CycleSlipEvent>,
) {
    let mut observations = observations.iter().collect::<Vec<_>>();
    observations.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));

    for obs in observations {
        if rtk_lli_set(obs.lli) {
            events.push(CycleSlipEvent {
                receiver,
                satellite_id: obs.satellite_id.clone(),
                epoch_index,
                reasons: vec![SlipReason::Lli],
            });
        }
    }
}

fn dual_cycle_slip_events(
    epochs: &[DualCycleSlipEpoch],
    options: CycleSlipOptions,
) -> Result<Vec<CycleSlipEvent>, CycleSlipPrepError> {
    let mut events = Vec::new();
    dual_cycle_slip_events_for_receiver(CycleSlipReceiver::Base, epochs, options, &mut events)?;
    dual_cycle_slip_events_for_receiver(CycleSlipReceiver::Rover, epochs, options, &mut events)?;
    Ok(events)
}

#[derive(Clone, Copy)]
struct DualCycleSlipSample<'a> {
    epoch_index: usize,
    epoch_sort_key: &'a str,
    gap_time_s: Option<f64>,
    observation: &'a DualCycleSlipObservation,
}

fn dual_cycle_slip_events_for_receiver(
    receiver: CycleSlipReceiver,
    epochs: &[DualCycleSlipEpoch],
    options: CycleSlipOptions,
    events: &mut Vec<CycleSlipEvent>,
) -> Result<(), CycleSlipPrepError> {
    let mut arcs = BTreeMap::<String, Vec<DualCycleSlipSample<'_>>>::new();

    for (epoch_index, epoch) in epochs.iter().enumerate() {
        let observations = match receiver {
            CycleSlipReceiver::Base => &epoch.base_observations,
            CycleSlipReceiver::Rover => &epoch.rover_observations,
        };

        for observation in observations {
            arcs.entry(observation.satellite_id.clone())
                .or_default()
                .push(DualCycleSlipSample {
                    epoch_index,
                    epoch_sort_key: &epoch.epoch_sort_key,
                    gap_time_s: epoch.gap_time_s,
                    observation,
                });
        }
    }

    for (satellite_id, mut samples) in arcs {
        samples.sort_by(|a, b| a.epoch_sort_key.cmp(b.epoch_sort_key));
        let arc = samples
            .iter()
            .map(|sample| dual_arc_epoch(sample.observation, sample.gap_time_s))
            .collect::<Vec<_>>();
        let results = detect_cycle_slips(&arc, options).map_err(cycle_slip_detector_error)?;

        for (sample, result) in samples.iter().zip(results) {
            if result.slip {
                events.push(CycleSlipEvent {
                    receiver,
                    satellite_id: satellite_id.clone(),
                    epoch_index: sample.epoch_index,
                    reasons: result.reasons,
                });
            }
        }
    }
    Ok(())
}

fn cycle_slip_detector_error(error: CarrierPhaseError) -> CycleSlipPrepError {
    let (field, reason) = match error {
        CarrierPhaseError::EqualFrequencies => {
            ("rtk cycle slip frequencies_hz", "degenerate frequencies")
        }
        CarrierPhaseError::InvalidFrequency => ("rtk cycle slip frequency_hz", "not positive"),
        CarrierPhaseError::InvalidObservation => ("rtk cycle slip observation", "not finite"),
        CarrierPhaseError::InvalidThreshold => ("rtk cycle slip threshold", "invalid"),
    };
    invalid_cycle_slip_input(field, reason)
}

fn dual_arc_epoch(observation: &DualCycleSlipObservation, gap_time_s: Option<f64>) -> ArcEpoch {
    ArcEpoch {
        phi1_cycles: Some(observation.phi1_cycles),
        phi2_cycles: Some(observation.phi2_cycles),
        p1_m: Some(observation.p1_m),
        p2_m: Some(observation.p2_m),
        lli1: observation.lli1,
        lli2: observation.lli2,
        f1_hz: Some(observation.f1_hz),
        f2_hz: Some(observation.f2_hz),
        gap_time_s,
    }
}

fn dropped_cycle_slip_sats(slips: &[CycleSlipEvent]) -> Vec<String> {
    slips
        .iter()
        .map(|slip| slip.satellite_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn drop_cycle_slip_satellites(
    epochs: &[CycleSlipEpoch],
    dropped_sats: &[String],
) -> Vec<CycleSlipEpoch> {
    let dropped = dropped_sats.iter().collect::<BTreeSet<_>>();
    epochs
        .iter()
        .map(|epoch| CycleSlipEpoch {
            base_observations: epoch
                .base_observations
                .iter()
                .filter(|obs| !dropped.contains(&obs.satellite_id))
                .cloned()
                .collect(),
            rover_observations: epoch
                .rover_observations
                .iter()
                .filter(|obs| !dropped.contains(&obs.satellite_id))
                .cloned()
                .collect(),
        })
        .collect()
}

fn drop_dual_cycle_slip_satellites(
    epochs: &[DualCycleSlipEpoch],
    dropped_sats: &[String],
) -> Vec<DualCycleSlipEpoch> {
    let dropped = dropped_sats.iter().collect::<BTreeSet<_>>();
    epochs
        .iter()
        .map(|epoch| DualCycleSlipEpoch {
            epoch_sort_key: epoch.epoch_sort_key.clone(),
            gap_time_s: epoch.gap_time_s,
            base_observations: epoch
                .base_observations
                .iter()
                .filter(|obs| !dropped.contains(&obs.satellite_id))
                .cloned()
                .collect(),
            rover_observations: epoch
                .rover_observations
                .iter()
                .filter(|obs| !dropped.contains(&obs.satellite_id))
                .cloned()
                .collect(),
        })
        .collect()
}

fn cycle_slip_split_sides(slips: &[CycleSlipEvent]) -> BTreeSet<(CycleSlipReceiver, String)> {
    slips
        .iter()
        .map(|slip| (slip.receiver, slip.satellite_id.clone()))
        .collect()
}

fn split_cycle_slip_arcs(
    epochs: &[CycleSlipEpoch],
    split_sides: &BTreeSet<(CycleSlipReceiver, String)>,
    slips: &[CycleSlipEvent],
) -> Vec<CycleSlipEpoch> {
    let slip_epochs = slips
        .iter()
        .map(|slip| (slip.receiver, slip.satellite_id.clone(), slip.epoch_index))
        .collect::<BTreeSet<_>>();
    let mut split_epochs = epochs.to_vec();
    let mut segments = BTreeMap::<(CycleSlipReceiver, String), usize>::new();

    for (epoch_index, epoch) in split_epochs.iter_mut().enumerate() {
        split_receiver_cycle_slip_arcs(
            CycleSlipReceiver::Base,
            epoch_index,
            &mut epoch.base_observations,
            split_sides,
            &slip_epochs,
            &mut segments,
        );
        split_receiver_cycle_slip_arcs(
            CycleSlipReceiver::Rover,
            epoch_index,
            &mut epoch.rover_observations,
            split_sides,
            &slip_epochs,
            &mut segments,
        );
    }

    split_epochs
}

fn split_dual_cycle_slip_arcs(
    epochs: &[DualCycleSlipEpoch],
    split_sides: &BTreeSet<(CycleSlipReceiver, String)>,
    slips: &[CycleSlipEvent],
) -> Vec<DualCycleSlipEpoch> {
    let slip_epochs = slips
        .iter()
        .map(|slip| (slip.receiver, slip.satellite_id.clone(), slip.epoch_index))
        .collect::<BTreeSet<_>>();
    let mut split_epochs = epochs.to_vec();
    let mut segments = BTreeMap::<(CycleSlipReceiver, String), usize>::new();

    for (epoch_index, epoch) in split_epochs.iter_mut().enumerate() {
        split_dual_receiver_cycle_slip_arcs(
            CycleSlipReceiver::Base,
            epoch_index,
            &mut epoch.base_observations,
            split_sides,
            &slip_epochs,
            &mut segments,
        );
        split_dual_receiver_cycle_slip_arcs(
            CycleSlipReceiver::Rover,
            epoch_index,
            &mut epoch.rover_observations,
            split_sides,
            &slip_epochs,
            &mut segments,
        );
    }

    split_epochs
}

fn split_receiver_cycle_slip_arcs(
    receiver: CycleSlipReceiver,
    epoch_index: usize,
    observations: &mut [CycleSlipObservation],
    split_sides: &BTreeSet<(CycleSlipReceiver, String)>,
    slip_epochs: &BTreeSet<(CycleSlipReceiver, String, usize)>,
    segments: &mut BTreeMap<(CycleSlipReceiver, String), usize>,
) {
    observations.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));

    for obs in observations {
        let key = (receiver, obs.satellite_id.clone());
        if split_sides.contains(&key) {
            let current_segment = segments.get(&key).copied().unwrap_or(1);
            let segment =
                if slip_epochs.contains(&(receiver, obs.satellite_id.clone(), epoch_index)) {
                    current_segment + 1
                } else {
                    current_segment
                };
            obs.ambiguity_id = split_side_ambiguity_id_core(&obs.satellite_id, receiver, segment);
            segments.insert(key, segment);
        }
    }
}

fn split_dual_receiver_cycle_slip_arcs(
    receiver: CycleSlipReceiver,
    epoch_index: usize,
    observations: &mut [DualCycleSlipObservation],
    split_sides: &BTreeSet<(CycleSlipReceiver, String)>,
    slip_epochs: &BTreeSet<(CycleSlipReceiver, String, usize)>,
    segments: &mut BTreeMap<(CycleSlipReceiver, String), usize>,
) {
    observations.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));

    for obs in observations {
        let key = (receiver, obs.satellite_id.clone());
        if split_sides.contains(&key) {
            let current_segment = segments.get(&key).copied().unwrap_or(1);
            let segment =
                if slip_epochs.contains(&(receiver, obs.satellite_id.clone(), epoch_index)) {
                    current_segment + 1
                } else {
                    current_segment
                };
            obs.ambiguity_id = split_side_ambiguity_id_core(&obs.satellite_id, receiver, segment);
            segments.insert(key, segment);
        }
    }
}

fn cycle_slip_split_metadata(
    epochs: &[CycleSlipEpoch],
    split_sides: &BTreeSet<(CycleSlipReceiver, String)>,
) -> Vec<CycleSlipSplitArc> {
    let mut grouped = BTreeMap::<(CycleSlipReceiver, String, String), Vec<usize>>::new();

    for (epoch_index, epoch) in epochs.iter().enumerate() {
        split_metadata_entries(
            CycleSlipReceiver::Base,
            epoch_index,
            &epoch.base_observations,
            split_sides,
            &mut grouped,
        );
        split_metadata_entries(
            CycleSlipReceiver::Rover,
            epoch_index,
            &epoch.rover_observations,
            split_sides,
            &mut grouped,
        );
    }

    grouped
        .into_iter()
        .map(|((receiver, satellite_id, ambiguity_id), epoch_indices)| {
            let start_epoch_index = *epoch_indices
                .first()
                .expect("metadata has at least one epoch");
            let end_epoch_index = *epoch_indices
                .last()
                .expect("metadata has at least one epoch");
            CycleSlipSplitArc {
                receiver,
                satellite_id,
                ambiguity_id,
                start_epoch_index,
                end_epoch_index,
                n_epochs: epoch_indices.len(),
            }
        })
        .collect()
}

fn dual_cycle_slip_split_metadata(
    epochs: &[DualCycleSlipEpoch],
    split_sides: &BTreeSet<(CycleSlipReceiver, String)>,
) -> Vec<CycleSlipSplitArc> {
    let mut grouped = BTreeMap::<(CycleSlipReceiver, String, String), Vec<usize>>::new();

    for (epoch_index, epoch) in epochs.iter().enumerate() {
        dual_split_metadata_entries(
            CycleSlipReceiver::Base,
            epoch_index,
            &epoch.base_observations,
            split_sides,
            &mut grouped,
        );
        dual_split_metadata_entries(
            CycleSlipReceiver::Rover,
            epoch_index,
            &epoch.rover_observations,
            split_sides,
            &mut grouped,
        );
    }

    grouped
        .into_iter()
        .map(|((receiver, satellite_id, ambiguity_id), epoch_indices)| {
            let start_epoch_index = *epoch_indices
                .first()
                .expect("metadata has at least one epoch");
            let end_epoch_index = *epoch_indices
                .last()
                .expect("metadata has at least one epoch");
            CycleSlipSplitArc {
                receiver,
                satellite_id,
                ambiguity_id,
                start_epoch_index,
                end_epoch_index,
                n_epochs: epoch_indices.len(),
            }
        })
        .collect()
}

fn split_metadata_entries(
    receiver: CycleSlipReceiver,
    epoch_index: usize,
    observations: &[CycleSlipObservation],
    split_sides: &BTreeSet<(CycleSlipReceiver, String)>,
    grouped: &mut BTreeMap<(CycleSlipReceiver, String, String), Vec<usize>>,
) {
    let mut observations = observations.iter().collect::<Vec<_>>();
    observations.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));

    for obs in observations {
        if split_sides.contains(&(receiver, obs.satellite_id.clone())) {
            grouped
                .entry((receiver, obs.satellite_id.clone(), obs.ambiguity_id.clone()))
                .or_default()
                .push(epoch_index);
        }
    }
}

fn dual_split_metadata_entries(
    receiver: CycleSlipReceiver,
    epoch_index: usize,
    observations: &[DualCycleSlipObservation],
    split_sides: &BTreeSet<(CycleSlipReceiver, String)>,
    grouped: &mut BTreeMap<(CycleSlipReceiver, String, String), Vec<usize>>,
) {
    let mut observations = observations.iter().collect::<Vec<_>>();
    observations.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));

    for obs in observations {
        if split_sides.contains(&(receiver, obs.satellite_id.clone())) {
            grouped
                .entry((receiver, obs.satellite_id.clone(), obs.ambiguity_id.clone()))
                .or_default()
                .push(epoch_index);
        }
    }
}

fn segment_reacquired_arcs(mut epochs: Vec<CycleSlipEpoch>) -> Vec<CycleSlipEpoch> {
    segment_receiver_reacquisitions(&mut epochs, CycleSlipReceiver::Base);
    segment_receiver_reacquisitions(&mut epochs, CycleSlipReceiver::Rover);
    epochs
}

fn segment_receiver_reacquisitions(epochs: &mut [CycleSlipEpoch], receiver: CycleSlipReceiver) {
    let mut present_last = BTreeSet::<String>::new();
    let mut arcs = BTreeMap::<String, usize>::new();

    for epoch in epochs {
        let observations = match receiver {
            CycleSlipReceiver::Base => &mut epoch.base_observations,
            CycleSlipReceiver::Rover => &mut epoch.rover_observations,
        };
        observations.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));
        let present = observations
            .iter()
            .map(|obs| obs.satellite_id.clone())
            .collect::<BTreeSet<_>>();

        for obs in observations {
            let reacquired =
                arcs.contains_key(&obs.satellite_id) && !present_last.contains(&obs.satellite_id);
            let arc = arcs.get(&obs.satellite_id).copied().unwrap_or(0) + usize::from(reacquired);
            arcs.insert(obs.satellite_id.clone(), arc);

            if arc > 0 {
                obs.ambiguity_id = reacquired_ambiguity_id_core(&obs.ambiguity_id, arc);
            }
        }

        present_last = present;
    }
}

fn segment_reacquired_dual_arcs(mut epochs: Vec<DualCycleSlipEpoch>) -> Vec<DualCycleSlipEpoch> {
    segment_dual_receiver_reacquisitions(&mut epochs, CycleSlipReceiver::Base);
    segment_dual_receiver_reacquisitions(&mut epochs, CycleSlipReceiver::Rover);
    epochs
}

fn segment_dual_receiver_reacquisitions(
    epochs: &mut [DualCycleSlipEpoch],
    receiver: CycleSlipReceiver,
) {
    let mut present_last = BTreeSet::<String>::new();
    let mut arcs = BTreeMap::<String, usize>::new();

    for epoch in epochs {
        let observations = match receiver {
            CycleSlipReceiver::Base => &mut epoch.base_observations,
            CycleSlipReceiver::Rover => &mut epoch.rover_observations,
        };
        observations.sort_by(|a, b| a.satellite_id.cmp(&b.satellite_id));
        let present = observations
            .iter()
            .map(|obs| obs.satellite_id.clone())
            .collect::<BTreeSet<_>>();

        for obs in observations {
            let reacquired =
                arcs.contains_key(&obs.satellite_id) && !present_last.contains(&obs.satellite_id);
            let arc = arcs.get(&obs.satellite_id).copied().unwrap_or(0) + usize::from(reacquired);
            arcs.insert(obs.satellite_id.clone(), arc);

            if arc > 0 {
                obs.ambiguity_id = reacquired_ambiguity_id_core(&obs.ambiguity_id, arc);
            }
        }

        present_last = present;
    }
}

fn split_side_ambiguity_id_core(
    satellite_id: &str,
    receiver: CycleSlipReceiver,
    segment: usize,
) -> String {
    format!(
        "{satellite_id}@{}#{segment}",
        cycle_slip_receiver_tag(receiver)
    )
}

fn reacquired_ambiguity_id_core(ambiguity_id: &str, arc: usize) -> String {
    let base = strip_reacquired_suffix(ambiguity_id);
    format!("{base}~ra{arc}")
}

fn strip_reacquired_suffix(ambiguity_id: &str) -> &str {
    if let Some(index) = ambiguity_id.rfind("~ra") {
        let suffix = &ambiguity_id[index + 3..];
        if !suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_digit()) {
            return &ambiguity_id[..index];
        }
    }
    ambiguity_id
}

fn cycle_slip_receiver_tag(receiver: CycleSlipReceiver) -> &'static str {
    match receiver {
        CycleSlipReceiver::Base => "base",
        CycleSlipReceiver::Rover => "rover",
    }
}

fn satellite_system(satellite_id: &str) -> String {
    crate::id::constellation_letter(satellite_id).to_string()
}

/// Single-difference ambiguity-id token from the per-receiver ambiguity ids.
///
/// Clean arcs (both receivers carry the bare satellite id) yield the satellite
/// id; a split on one side carries that side's id, a shared split id carries it,
/// and a divergent split records both. Shared by the single- and dual-frequency
/// single-difference builders and by the sequential RTK arc driver so the SD
/// column naming is defined in exactly one place.
pub(crate) fn sd_ambiguity_token(sat: &str, base_id: &str, rover_id: &str) -> String {
    match (base_id, rover_id) {
        (base_id, rover_id) if base_id == sat && rover_id == sat => sat.to_string(),
        (base_id, rover_id) if base_id == sat => rover_id.to_string(),
        (base_id, rover_id) if rover_id == sat => base_id.to_string(),
        (base_id, rover_id) if base_id == rover_id => base_id.to_string(),
        (base_id, rover_id) => format!("{sat}:base={base_id},rover={rover_id}"),
    }
}

/// Double-difference ambiguity-id token from the satellite SD id and its own
/// system's reference SD id/satellite. Clean arcs (the satellite carries its
/// bare id and the reference SD is the reference satellite's bare id) yield the
/// satellite id; otherwise the reference is recorded explicitly. Shared by the
/// single- and dual-frequency double-difference builders and the arc driver.
pub(crate) fn dd_ambiguity_token(
    sat: &str,
    sat_sd_id: &str,
    ref_sd_id: &str,
    ref_sat: &str,
) -> String {
    if sat_sd_id == sat && ref_sd_id == ref_sat {
        sat.to_string()
    } else {
        format!("{sat_sd_id}|ref={ref_sd_id}")
    }
}

fn single_difference_ambiguity_id(
    sat: &str,
    base_obs: &Observation,
    rover_obs: &Observation,
) -> AmbiguityId {
    AmbiguityId::new(sd_ambiguity_token(
        sat,
        base_obs.ambiguity_id.as_str(),
        rover_obs.ambiguity_id.as_str(),
    ))
}

fn single_difference_dual_ambiguity_id(
    sat: &str,
    base_obs: &DualObservation,
    rover_obs: &DualObservation,
) -> AmbiguityId {
    AmbiguityId::new(sd_ambiguity_token(
        sat,
        base_obs.ambiguity_id.as_str(),
        rover_obs.ambiguity_id.as_str(),
    ))
}

fn double_difference_ambiguity_id(
    sat: &str,
    sat_sd_id: &AmbiguityId,
    ref_sd: &SingleDifference,
) -> AmbiguityId {
    AmbiguityId::new(dd_ambiguity_token(
        sat,
        sat_sd_id.as_str(),
        ref_sd.ambiguity_id.as_str(),
        &ref_sd.satellite_id,
    ))
}

fn double_difference_dual_ambiguity_id(
    sat: &str,
    sat_sd_id: &AmbiguityId,
    ref_sd: &DualSingleDifference,
) -> AmbiguityId {
    AmbiguityId::new(dd_ambiguity_token(
        sat,
        sat_sd_id.as_str(),
        ref_sd.ambiguity_id.as_str(),
        &ref_sd.satellite_id,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gps_l1_hz() -> f64 {
        crate::frequencies::frequency_hz(
            crate::GnssSystem::Gps,
            crate::frequencies::CarrierBand::L1,
        )
        .expect("canonical GPS L1 carrier exists")
    }

    fn gps_l2_hz() -> f64 {
        crate::frequencies::frequency_hz(
            crate::GnssSystem::Gps,
            crate::frequencies::CarrierBand::L2,
        )
        .expect("canonical GPS L2 carrier exists")
    }

    fn obs(sat: &str, code_m: f64, phase_m: f64) -> Observation {
        Observation {
            satellite_id: sat.to_string(),
            ambiguity_id: sat.to_string(),
            code_m,
            phase_m,
        }
    }

    fn dual_observation(ambiguity_id: &str, wide_lane_phase_cycles: f64) -> DualObservation {
        DualObservation {
            ambiguity_id: ambiguity_id.to_string(),
            p1_m: 0.0,
            p2_m: 0.0,
            phi1_cycles: wide_lane_phase_cycles,
            phi2_cycles: 0.0,
            f1_hz: gps_l1_hz(),
            f2_hz: gps_l2_hz(),
        }
    }

    fn dual_pair(sat: &str, base_wide_lane: f64, rover_wide_lane: f64) -> DualSatelliteObservation {
        DualSatelliteObservation {
            satellite_id: sat.to_string(),
            base: dual_observation(sat, base_wide_lane),
            rover: dual_observation(sat, rover_wide_lane),
        }
    }

    fn split_dual_pair(
        sat: &str,
        rover_id: &str,
        base_wide_lane: f64,
        rover_wide_lane: f64,
    ) -> DualSatelliteObservation {
        DualSatelliteObservation {
            satellite_id: sat.to_string(),
            base: dual_observation(sat, base_wide_lane),
            rover: dual_observation(rover_id, rover_wide_lane),
        }
    }

    fn if_observation(
        ambiguity_id: &str,
        p1_m: f64,
        p2_m: f64,
        phi1_cycles: f64,
        phi2_cycles: f64,
        tropo_m: f64,
    ) -> DualIonosphereFreeObservation {
        DualIonosphereFreeObservation {
            ambiguity_id: ambiguity_id.to_string(),
            p1_m,
            p2_m,
            phi1_cycles,
            phi2_cycles,
            f1_hz: gps_l1_hz(),
            f2_hz: gps_l2_hz(),
            tropo_m,
        }
    }

    fn if_pair(
        sat: &str,
        base_code: f64,
        rover_code: f64,
        base_phase_cycles: f64,
        rover_phase_cycles: f64,
    ) -> DualIonosphereFreeSatelliteObservation {
        DualIonosphereFreeSatelliteObservation {
            satellite_id: sat.to_string(),
            base: if_observation(
                sat,
                base_code,
                base_code + 2.0,
                base_phase_cycles,
                base_phase_cycles - 4.0,
                0.25,
            ),
            rover: if_observation(
                sat,
                rover_code,
                rover_code + 2.5,
                rover_phase_cycles,
                rover_phase_cycles - 3.0,
                0.5,
            ),
        }
    }

    fn arc_obs(sat: &str, ambiguity_id: &str, code_m: f64, phase_m: f64) -> Observation {
        Observation {
            satellite_id: sat.to_string(),
            ambiguity_id: ambiguity_id.to_string(),
            code_m,
            phase_m,
        }
    }

    fn smooth_obs(
        sat: &str,
        ambiguity_id: &str,
        code_m: f64,
        phase_m: f64,
        lli: Option<i64>,
    ) -> CodeSmoothingObservation {
        CodeSmoothingObservation {
            satellite_id: sat.to_string(),
            ambiguity_id: ambiguity_id.to_string(),
            code_m,
            phase_m,
            lli,
        }
    }

    fn dual_slip_obs(
        sat: &str,
        ambiguity_id: &str,
        phi1_cycles: f64,
        phi2_cycles: f64,
        lli1: Option<i64>,
        lli2: Option<i64>,
    ) -> DualCycleSlipObservation {
        DualCycleSlipObservation {
            satellite_id: sat.to_string(),
            ambiguity_id: ambiguity_id.to_string(),
            p1_m: 20.0,
            p2_m: 21.0,
            phi1_cycles,
            phi2_cycles,
            f1_hz: gps_l1_hz(),
            f2_hz: gps_l2_hz(),
            lli1,
            lli2,
        }
    }

    fn dual_slip_epoch(
        epoch_sort_key: &str,
        gap_time_s: f64,
        base_observations: Vec<DualCycleSlipObservation>,
        rover_observations: Vec<DualCycleSlipObservation>,
    ) -> DualCycleSlipEpoch {
        DualCycleSlipEpoch {
            epoch_sort_key: epoch_sort_key.to_string(),
            gap_time_s: Some(gap_time_s),
            base_observations,
            rover_observations,
        }
    }

    fn code_bits(observations: &[CodeSmoothingObservation]) -> Vec<(&str, u64)> {
        observations
            .iter()
            .map(|obs| (obs.satellite_id.as_str(), obs.code_m.to_bits()))
            .collect()
    }

    fn ambiguity_id<'a>(
        observations: &'a [CycleSlipObservation],
        satellite_id: &str,
    ) -> Option<&'a str> {
        observations
            .iter()
            .find(|obs| obs.satellite_id == satellite_id)
            .map(|obs| obs.ambiguity_id.as_str())
    }

    fn dual_ambiguity_id<'a>(
        observations: &'a [DualCycleSlipObservation],
        satellite_id: &str,
    ) -> Option<&'a str> {
        observations
            .iter()
            .find(|obs| obs.satellite_id == satellite_id)
            .map(|obs| obs.ambiguity_id.as_str())
    }

    fn baseline_reference_epoch(entries: &[(&str, [f64; 3])]) -> BaselineReferenceEpoch {
        BaselineReferenceEpoch {
            available_satellite_ids: entries.iter().map(|(sat, _)| sat.to_string()).collect(),
            satellite_positions_m: entries
                .iter()
                .map(|(sat, pos)| (sat.to_string(), *pos))
                .collect(),
        }
    }

    #[test]
    fn double_differences_cancel_receiver_and_common_terms() {
        let sats = ["G01", "G02", "G03", "G04"];
        let reference = "G01";
        let base_clock_m = 125.0;
        let rover_clock_m = -42.0;
        let base_ranges = BTreeMap::from([
            ("G01", 20_000.0),
            ("G02", 21_000.0),
            ("G03", 22_500.0),
            ("G04", 23_100.0),
        ]);
        let rover_ranges = BTreeMap::from([
            ("G01", 20_010.0),
            ("G02", 21_025.0),
            ("G03", 22_480.0),
            ("G04", 23_150.0),
        ]);
        let common_errors =
            BTreeMap::from([("G01", 3.25), ("G02", -12.0), ("G03", 8.5), ("G04", 1.0)]);
        let base_ambiguities =
            BTreeMap::from([("G01", 2.0), ("G02", -3.0), ("G03", 7.0), ("G04", 11.0)]);
        let rover_ambiguities =
            BTreeMap::from([("G01", 5.0), ("G02", 4.0), ("G03", 1.0), ("G04", 19.0)]);

        let base = sats
            .iter()
            .map(|sat| {
                obs(
                    sat,
                    base_ranges[sat] + base_clock_m + common_errors[sat],
                    base_ranges[sat] + base_clock_m + common_errors[sat] + base_ambiguities[sat],
                )
            })
            .collect::<Vec<_>>();
        let rover = sats
            .iter()
            .map(|sat| {
                obs(
                    sat,
                    rover_ranges[sat] + rover_clock_m + common_errors[sat],
                    rover_ranges[sat] + rover_clock_m + common_errors[sat] + rover_ambiguities[sat],
                )
            })
            .collect::<Vec<_>>();

        let result = double_differences(
            &base,
            &rover,
            ReferenceSelection::Satellite(reference.to_string()),
        )
        .unwrap();

        assert_eq!(
            result.reference_satellite_id,
            ReferenceReport::Satellite(reference.to_string())
        );
        assert!(result.dropped_sats.is_empty());

        let by_sat = result
            .double_differences
            .iter()
            .map(|dd| (dd.satellite_id.as_str(), dd))
            .collect::<BTreeMap<_, _>>();

        for sat in sats.into_iter().filter(|sat| *sat != reference) {
            let expected_code = rover_ranges[sat]
                - base_ranges[sat]
                - (rover_ranges[reference] - base_ranges[reference]);
            let expected_phase = expected_code + (rover_ambiguities[sat] - base_ambiguities[sat])
                - (rover_ambiguities[reference] - base_ambiguities[reference]);
            let dd = by_sat[sat];
            assert_eq!(dd.reference_satellite_id, reference);
            assert_eq!(dd.code_m, expected_code);
            assert_eq!(dd.phase_m, expected_phase);
        }
    }

    #[test]
    fn auto_reference_reports_dropped_satellites() {
        let base = vec![
            obs("G02", 210.0, 211.0),
            obs("G01", 100.0, 101.0),
            obs("G09", 900.0, 901.0),
        ];
        let rover = vec![
            obs("G02", 230.0, 233.0),
            obs("G01", 105.0, 108.0),
            obs("G10", 1000.0, 1001.0),
        ];

        let result = double_differences(&base, &rover, ReferenceSelection::Auto).unwrap();

        assert_eq!(
            result.reference_satellite_id,
            ReferenceReport::Satellite("G01".to_string())
        );
        assert_eq!(result.dropped_sats, ["G09".to_string(), "G10".to_string()]);
        assert_eq!(
            result.double_differences,
            vec![DoubleDifference {
                satellite_id: "G02".to_string(),
                reference_satellite_id: "G01".to_string(),
                ambiguity_id: "G02".to_string(),
                code_m: 15.0,
                phase_m: 15.0,
            }]
        );
    }

    #[test]
    fn explicit_arc_ids_feed_double_difference_ambiguity_id() {
        let base = vec![obs("G01", 100.0, 101.0), obs("G02", 210.0, 211.0)];
        let rover = vec![
            arc_obs("G01", "G01#2", 105.0, 108.0),
            arc_obs("G02", "G02#2", 230.0, 233.0),
        ];

        let result = double_differences(
            &base,
            &rover,
            ReferenceSelection::Satellite("G01".to_string()),
        )
        .unwrap();

        assert_eq!(result.double_differences[0].ambiguity_id, "G02#2|ref=G01#2");
    }

    #[test]
    fn multi_system_uses_reference_per_system() {
        let base = vec![
            obs("G01", 10.0, 11.0),
            obs("G02", 20.0, 21.0),
            obs("E11", 30.0, 31.0),
            obs("E19", 40.0, 41.0),
        ];
        let rover = vec![
            obs("G01", 12.0, 14.0),
            obs("G02", 24.0, 27.0),
            obs("E11", 33.0, 35.0),
            obs("E19", 46.0, 49.0),
        ];
        let refs = BTreeMap::from([
            ("E".to_string(), "E11".to_string()),
            ("G".to_string(), "G01".to_string()),
        ]);

        let result =
            double_differences(&base, &rover, ReferenceSelection::PerSystem(refs.clone())).unwrap();

        assert_eq!(
            result.reference_satellite_id,
            ReferenceReport::PerSystem(refs)
        );
        assert_eq!(
            result
                .double_differences
                .iter()
                .map(|dd| (&dd.satellite_id, &dd.reference_satellite_id))
                .collect::<Vec<_>>(),
            vec![
                (&"E19".to_string(), &"E11".to_string()),
                (&"G02".to_string(), &"G01".to_string()),
            ]
        );
    }

    #[test]
    fn multi_system_requires_non_reference_satellite_per_system() {
        let base = vec![obs("G01", 10.0, 11.0), obs("E01", 30.0, 31.0)];
        let rover = vec![obs("G01", 12.0, 14.0), obs("E01", 33.0, 35.0)];
        let expected = Err(DoubleDifferenceError::TooFewCommonSatellites {
            count: 1,
            minimum: 2,
        });

        assert_eq!(
            double_differences(&base, &rover, ReferenceSelection::Auto),
            expected
        );
        assert_eq!(
            double_differences(
                &base,
                &rover,
                ReferenceSelection::PerSystem(BTreeMap::from([
                    ("E".to_string(), "E01".to_string()),
                    ("G".to_string(), "G01".to_string()),
                ])),
            ),
            expected
        );
    }

    #[test]
    fn baseline_auto_reference_uses_highest_average_elevation_per_system() {
        let base = [10.0, 0.0, 0.0];
        let epochs = vec![
            baseline_reference_epoch(&[
                ("G01", [20.0, 0.0, 0.0]),
                ("G02", [20.0, 0.0, 0.0]),
                ("G03", [20.0, 0.0, 0.0]),
                ("E01", [10.0, 10.0, 0.0]),
                ("E02", [20.0, 0.0, 0.0]),
            ]),
            baseline_reference_epoch(&[
                ("G01", [10.0, 10.0, 0.0]),
                ("G03", [20.0, 0.0, 0.0]),
                ("E01", [10.0, 10.0, 0.0]),
                ("E02", [20.0, 0.0, 0.0]),
            ]),
        ];

        let refs =
            baseline_reference_satellites(base, &epochs, BaselineReferenceSelection::Auto).unwrap();

        assert_eq!(
            refs,
            BTreeMap::from([
                ("E".to_string(), "E02".to_string()),
                ("G".to_string(), "G03".to_string()),
            ])
        );

        let g_epochs = baseline_epochs_for_system(&epochs, "G");
        assert_eq!(
            average_elevation_score(base, &g_epochs, "G01")
                .unwrap()
                .to_bits(),
            0x3fe0_0000_0000_0000
        );
        assert_eq!(
            average_elevation_score(base, &g_epochs, "G03")
                .unwrap()
                .to_bits(),
            0x3ff0_0000_0000_0000
        );
    }

    #[test]
    fn baseline_auto_reference_ties_by_satellite_id() {
        let base = [10.0, 0.0, 0.0];
        let epochs = vec![baseline_reference_epoch(&[
            ("G01", [20.0, 0.0, 0.0]),
            ("G02", [20.0, 0.0, 0.0]),
        ])];

        let refs =
            baseline_reference_satellites(base, &epochs, BaselineReferenceSelection::Auto).unwrap();

        assert_eq!(refs, BTreeMap::from([("G".to_string(), "G01".to_string())]));
    }

    #[test]
    fn baseline_reference_errors_when_available_satellite_position_missing() {
        let mut epoch = baseline_reference_epoch(&[("G01", [20.0, 0.0, 0.0])]);
        epoch.available_satellite_ids.push("G02".to_string());

        assert_eq!(
            baseline_reference_satellites(
                [10.0, 0.0, 0.0],
                &[epoch],
                BaselineReferenceSelection::Auto
            ),
            Err(DoubleDifferenceError::MissingSatellitePosition(
                "G02".to_string()
            ))
        );
    }

    #[test]
    fn elevation_mask_keeps_epoch_satellites_above_threshold() {
        let base = [10.0, 0.0, 0.0];
        let up = local_up(base);

        assert_eq!(
            elevation_score_with_up(base, up, [20.0, 0.0, 0.0])
                .unwrap()
                .to_bits(),
            0x3ff0_0000_0000_0000
        );
        assert_eq!(
            elevation_score_with_up(base, up, [10.0, 10.0, 0.0])
                .unwrap()
                .to_bits(),
            0x0000_0000_0000_0000
        );
        assert_eq!(
            elevation_score_with_up(base, up, [0.0, 0.0, 0.0])
                .unwrap()
                .to_bits(),
            0xbff0_0000_0000_0000
        );

        let epochs = vec![
            ElevationMaskEpoch {
                satellite_positions_m: BTreeMap::from([
                    ("G01".to_string(), [20.0, 0.0, 0.0]),
                    ("G02".to_string(), [10.0, 10.0, 0.0]),
                    ("G03".to_string(), [0.0, 0.0, 0.0]),
                ]),
            },
            ElevationMaskEpoch {
                satellite_positions_m: BTreeMap::from([
                    ("G01".to_string(), [20.0, 0.0, 0.0]),
                    ("G02".to_string(), [20.0, 0.0, 0.0]),
                    ("G04".to_string(), [0.0, 0.0, 0.0]),
                ]),
            },
        ];

        let result = apply_elevation_mask(base, &epochs, 30.0).unwrap();

        assert_eq!(
            result.epochs,
            vec![
                ElevationMaskEpochResult {
                    kept_satellite_ids: vec!["G01".to_string()],
                },
                ElevationMaskEpochResult {
                    kept_satellite_ids: vec!["G01".to_string(), "G02".to_string()],
                },
            ]
        );
        assert_eq!(
            result.masked_satellite_ids,
            vec!["G02".to_string(), "G03".to_string(), "G04".to_string()]
        );
    }

    #[test]
    fn baseline_reference_rejects_invalid_geometry() {
        let epochs = vec![baseline_reference_epoch(&[
            ("G01", [20.0, 0.0, 0.0]),
            ("G02", [30.0, 0.0, 0.0]),
        ])];

        assert_eq!(
            baseline_reference_satellites(
                [f64::NAN, 0.0, 0.0],
                &epochs,
                BaselineReferenceSelection::Auto,
            ),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk base position_m",
                reason: "not finite",
            })
        );
        assert_eq!(
            baseline_reference_satellites(
                [0.0, 0.0, 0.0],
                &epochs,
                BaselineReferenceSelection::Auto,
            ),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk base position_m",
                reason: "degenerate geometry",
            })
        );

        let invalid_sat = vec![baseline_reference_epoch(&[
            ("G01", [20.0, 0.0, 0.0]),
            ("G02", [f64::INFINITY, 0.0, 0.0]),
        ])];
        assert_eq!(
            baseline_reference_satellites(
                [10.0, 0.0, 0.0],
                &invalid_sat,
                BaselineReferenceSelection::Auto,
            ),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk satellite position_m",
                reason: "not finite",
            })
        );

        let coincident_sat = vec![baseline_reference_epoch(&[
            ("G01", [10.0, 0.0, 0.0]),
            ("G02", [20.0, 0.0, 0.0]),
        ])];
        assert_eq!(
            baseline_reference_satellites(
                [10.0, 0.0, 0.0],
                &coincident_sat,
                BaselineReferenceSelection::Auto,
            ),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk line of sight_m",
                reason: "degenerate geometry",
            })
        );
    }

    #[test]
    fn elevation_mask_rejects_invalid_geometry_and_mask() {
        let epochs = vec![ElevationMaskEpoch {
            satellite_positions_m: BTreeMap::from([("G01".to_string(), [20.0, 0.0, 0.0])]),
        }];

        assert_eq!(
            apply_elevation_mask([10.0, 0.0, 0.0], &epochs, f64::NAN),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk elevation mask_deg",
                reason: "not finite",
            })
        );
        assert_eq!(
            apply_elevation_mask([10.0, 0.0, 0.0], &epochs, 91.0),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk elevation mask_deg",
                reason: "out of range",
            })
        );

        let coincident_sat = vec![ElevationMaskEpoch {
            satellite_positions_m: BTreeMap::from([("G01".to_string(), [10.0, 0.0, 0.0])]),
        }];
        assert_eq!(
            apply_elevation_mask([10.0, 0.0, 0.0], &coincident_sat, 0.0),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk line of sight_m",
                reason: "degenerate geometry",
            })
        );
    }

    #[test]
    fn code_smoothing_has_frozen_bits_and_receiver_state() {
        let epochs = vec![
            CodeSmoothingEpoch {
                base_observations: vec![
                    smooth_obs("G02", "G02", 100.0, 10.0, None),
                    smooth_obs("G01", "G01", 200.0, 20.0, None),
                ],
                rover_observations: vec![
                    smooth_obs("G03", "G03", 600.0, 60.0, None),
                    smooth_obs("G02", "G02", 500.0, 50.0, None),
                ],
            },
            CodeSmoothingEpoch {
                base_observations: vec![
                    smooth_obs("G01", "G01", 201.0, 21.0, Some(1)),
                    smooth_obs("G02", "G02", 102.0, 11.0, None),
                ],
                rover_observations: vec![smooth_obs("G02", "G02", 504.0, 51.0, None)],
            },
            CodeSmoothingEpoch {
                base_observations: vec![
                    smooth_obs("G02", "G02", 103.0, 13.0, None),
                    smooth_obs("G01", "G01", 202.0, 22.0, None),
                ],
                rover_observations: vec![
                    smooth_obs("G03", "G03", 604.0, 62.0, None),
                    smooth_obs("G02", "G02", 506.0, 53.0, None),
                ],
            },
        ];

        let result = hatch_smooth_baseline_code_epochs(&epochs, 2).unwrap();

        assert_eq!(
            code_bits(&result[0].base_observations),
            vec![
                ("G01", 0x4069_0000_0000_0000),
                ("G02", 0x4059_0000_0000_0000),
            ]
        );
        assert_eq!(
            code_bits(&result[1].base_observations),
            vec![
                ("G01", 0x4069_2000_0000_0000),
                ("G02", 0x4059_6000_0000_0000),
            ]
        );
        assert_eq!(
            code_bits(&result[2].base_observations),
            vec![
                ("G01", 0x4069_4000_0000_0000),
                ("G02", 0x4059_d000_0000_0000),
            ]
        );
        assert_eq!(
            code_bits(&result[0].rover_observations),
            vec![
                ("G02", 0x407f_4000_0000_0000),
                ("G03", 0x4082_c000_0000_0000),
            ]
        );
        assert_eq!(
            code_bits(&result[1].rover_observations),
            vec![("G02", 0x407f_6800_0000_0000)]
        );
        assert_eq!(
            code_bits(&result[2].rover_observations),
            vec![
                ("G02", 0x407f_9400_0000_0000),
                ("G03", 0x4082_d800_0000_0000),
            ]
        );
        assert_eq!(result[1].base_observations[0].lli, Some(1));
        assert_eq!(
            hatch_smooth_baseline_code_epochs(&epochs, 0),
            Err(CodeSmoothingError::InvalidWindowCap)
        );
    }

    #[test]
    fn cycle_slip_prep_pins_policy_and_reacquisition_behavior() {
        let epochs = vec![
            CycleSlipEpoch {
                base_observations: vec![
                    smooth_obs("G02", "G02", 100.0, 10.0, None),
                    smooth_obs("G01", "G01", 200.0, 20.0, None),
                ],
                rover_observations: vec![
                    smooth_obs("G02", "G02", 101.0, 11.0, None),
                    smooth_obs("G01", "G01", 201.0, 21.0, None),
                ],
            },
            CycleSlipEpoch {
                base_observations: vec![
                    smooth_obs("G01", "G01", 210.0, 30.0, None),
                    smooth_obs("G02", "G02", 110.0, 15.0, None),
                ],
                rover_observations: vec![
                    smooth_obs("G01", "G01", 211.0, 31.0, None),
                    smooth_obs("G02", "G02", 111.0, 16.0, Some(1)),
                ],
            },
            CycleSlipEpoch {
                base_observations: vec![smooth_obs("G01", "G01", 220.0, 40.0, None)],
                rover_observations: vec![smooth_obs("G01", "G01", 221.0, 41.0, None)],
            },
            CycleSlipEpoch {
                base_observations: vec![
                    smooth_obs("G02", "G02", 130.0, 18.0, None),
                    smooth_obs("G01", "G01", 230.0, 50.0, None),
                ],
                rover_observations: vec![
                    smooth_obs("G02", "G02", 131.0, 19.0, None),
                    smooth_obs("G01", "G01", 231.0, 51.0, None),
                ],
            },
        ];

        assert_eq!(
            prepare_cycle_slip_baseline_epochs(&epochs, CycleSlipPolicy::Error),
            Err(CycleSlipPrepError::CycleSlipDetected {
                receiver: CycleSlipReceiver::Rover,
                satellite_id: "G02".to_string(),
                epoch_index: 1,
                reasons: vec![SlipReason::Lli],
            })
        );

        let dropped =
            prepare_cycle_slip_baseline_epochs(&epochs, CycleSlipPolicy::DropSatellite).unwrap();
        assert_eq!(dropped.dropped_sats, vec!["G02".to_string()]);
        assert!(dropped.split_arcs.is_empty());
        assert!(dropped
            .epochs
            .iter()
            .all(
                |epoch| ambiguity_id(&epoch.base_observations, "G02").is_none()
                    && ambiguity_id(&epoch.rover_observations, "G02").is_none()
            ));

        let split = prepare_cycle_slip_baseline_epochs(&epochs, CycleSlipPolicy::SplitArc).unwrap();
        assert!(split.dropped_sats.is_empty());
        assert_eq!(
            split.split_arcs,
            vec![
                CycleSlipSplitArc {
                    receiver: CycleSlipReceiver::Rover,
                    satellite_id: "G02".to_string(),
                    ambiguity_id: "G02@rover#1".to_string(),
                    start_epoch_index: 0,
                    end_epoch_index: 0,
                    n_epochs: 1,
                },
                CycleSlipSplitArc {
                    receiver: CycleSlipReceiver::Rover,
                    satellite_id: "G02".to_string(),
                    ambiguity_id: "G02@rover#2".to_string(),
                    start_epoch_index: 1,
                    end_epoch_index: 3,
                    n_epochs: 2,
                },
            ]
        );
        assert_eq!(
            ambiguity_id(&split.epochs[0].rover_observations, "G02"),
            Some("G02@rover#1")
        );
        assert_eq!(
            ambiguity_id(&split.epochs[1].rover_observations, "G02"),
            Some("G02@rover#2")
        );
        assert_eq!(
            ambiguity_id(&split.epochs[2].rover_observations, "G02"),
            None
        );
        assert_eq!(
            ambiguity_id(&split.epochs[3].rover_observations, "G02"),
            Some("G02@rover#2~ra1")
        );
        assert_eq!(
            ambiguity_id(&split.epochs[3].base_observations, "G02"),
            Some("G02~ra1")
        );
    }

    #[test]
    fn cycle_slip_prep_rejects_non_finite_arc_data() {
        assert_eq!(
            prepare_cycle_slip_baseline_epochs(
                &[CycleSlipEpoch {
                    base_observations: vec![smooth_obs("G01", "G01", f64::NAN, 10.0, None)],
                    rover_observations: Vec::new(),
                }],
                CycleSlipPolicy::DropSatellite,
            ),
            Err(CycleSlipPrepError::InvalidInput {
                field: "rtk cycle slip code_m",
                reason: "not finite",
            })
        );
        assert_eq!(
            prepare_cycle_slip_baseline_epochs(
                &[CycleSlipEpoch {
                    base_observations: Vec::new(),
                    rover_observations: vec![smooth_obs("G01", "G01", 100.0, f64::INFINITY, None,)],
                }],
                CycleSlipPolicy::DropSatellite,
            ),
            Err(CycleSlipPrepError::InvalidInput {
                field: "rtk cycle slip phase_m",
                reason: "not finite",
            })
        );
    }

    #[test]
    fn dual_cycle_slip_prep_rejects_invalid_arc_data_and_options() {
        let epochs = vec![dual_slip_epoch(
            "0",
            0.0,
            vec![dual_slip_obs("G01", "G01", 10.0, 8.0, None, None)],
            Vec::new(),
        )];

        assert_eq!(
            prepare_dual_cycle_slip_baseline_epochs(
                &epochs,
                CycleSlipPolicy::DropSatellite,
                CycleSlipOptions {
                    gf_threshold_m: f64::NAN,
                    ..CycleSlipOptions::default()
                },
            ),
            Err(CycleSlipPrepError::InvalidInput {
                field: "rtk cycle slip gf_threshold_m",
                reason: "not finite",
            })
        );
        assert_eq!(
            prepare_dual_cycle_slip_baseline_epochs(
                &epochs,
                CycleSlipPolicy::DropSatellite,
                CycleSlipOptions {
                    min_arc_gap_s: 0.0,
                    ..CycleSlipOptions::default()
                },
            ),
            Err(CycleSlipPrepError::InvalidInput {
                field: "rtk cycle slip min_arc_gap_s",
                reason: "not positive",
            })
        );

        let mut bad_gap = epochs.clone();
        bad_gap[0].gap_time_s = Some(f64::INFINITY);
        assert_eq!(
            prepare_dual_cycle_slip_baseline_epochs(
                &bad_gap,
                CycleSlipPolicy::DropSatellite,
                CycleSlipOptions::default(),
            ),
            Err(CycleSlipPrepError::InvalidInput {
                field: "rtk cycle slip gap_time_s",
                reason: "not finite",
            })
        );

        let mut bad_phase = epochs.clone();
        bad_phase[0].base_observations[0].phi1_cycles = f64::NAN;
        assert_eq!(
            prepare_dual_cycle_slip_baseline_epochs(
                &bad_phase,
                CycleSlipPolicy::DropSatellite,
                CycleSlipOptions::default(),
            ),
            Err(CycleSlipPrepError::InvalidInput {
                field: "rtk cycle slip phi1_cycles",
                reason: "not finite",
            })
        );

        let mut equal_frequency = epochs;
        equal_frequency[0].base_observations[0].f2_hz =
            equal_frequency[0].base_observations[0].f1_hz;
        assert_eq!(
            prepare_dual_cycle_slip_baseline_epochs(
                &equal_frequency,
                CycleSlipPolicy::DropSatellite,
                CycleSlipOptions::default(),
            ),
            Err(CycleSlipPrepError::InvalidInput {
                field: "rtk cycle slip frequencies_hz",
                reason: "degenerate frequencies",
            })
        );

        let overflow_epoch = dual_slip_epoch(
            "0",
            0.0,
            vec![dual_slip_obs("G01", "G01", f64::MAX, -f64::MAX, None, None)],
            Vec::new(),
        );
        assert_eq!(
            prepare_dual_cycle_slip_baseline_epochs(
                &[overflow_epoch],
                CycleSlipPolicy::DropSatellite,
                CycleSlipOptions::default(),
            ),
            Err(CycleSlipPrepError::InvalidInput {
                field: "rtk cycle slip observation",
                reason: "not finite",
            })
        );
    }

    #[test]
    fn dual_cycle_slip_prep_pins_policy_and_reacquisition_behavior() {
        let epochs = vec![
            dual_slip_epoch(
                "0",
                0.0,
                vec![
                    dual_slip_obs("G02", "G02", 10.0, 8.0, None, None),
                    dual_slip_obs("G01", "G01", 20.0, 18.0, None, None),
                ],
                vec![
                    dual_slip_obs("G02", "G02", 11.0, 9.0, None, None),
                    dual_slip_obs("G01", "G01", 21.0, 19.0, None, None),
                ],
            ),
            dual_slip_epoch(
                "1",
                1.0,
                vec![
                    dual_slip_obs("G01", "G01", 20.0, 18.0, None, None),
                    dual_slip_obs("G02", "G02", 10.0, 8.0, None, None),
                ],
                vec![
                    dual_slip_obs("G01", "G01", 21.0, 19.0, None, None),
                    dual_slip_obs("G02", "G02", 11.0, 9.0, Some(1), None),
                ],
            ),
            dual_slip_epoch(
                "2",
                2.0,
                vec![dual_slip_obs("G01", "G01", 20.0, 18.0, None, None)],
                vec![dual_slip_obs("G01", "G01", 21.0, 19.0, None, None)],
            ),
            dual_slip_epoch(
                "3",
                3.0,
                vec![
                    dual_slip_obs("G02", "G02", 10.0, 8.0, None, None),
                    dual_slip_obs("G01", "G01", 20.0, 18.0, None, None),
                ],
                vec![
                    dual_slip_obs("G02", "G02", 11.0, 9.0, None, None),
                    dual_slip_obs("G01", "G01", 21.0, 19.0, None, None),
                ],
            ),
        ];
        let options = CycleSlipOptions {
            gf_threshold_m: 1.0e9,
            mw_threshold_cycles: 1.0e9,
            min_arc_gap_s: 300.0,
        };

        assert_eq!(
            prepare_dual_cycle_slip_baseline_epochs(&epochs, CycleSlipPolicy::Error, options),
            Err(CycleSlipPrepError::CycleSlipDetected {
                receiver: CycleSlipReceiver::Rover,
                satellite_id: "G02".to_string(),
                epoch_index: 1,
                reasons: vec![SlipReason::Lli],
            })
        );

        let dropped = prepare_dual_cycle_slip_baseline_epochs(
            &epochs,
            CycleSlipPolicy::DropSatellite,
            options,
        )
        .unwrap();
        assert_eq!(dropped.dropped_sats, vec!["G02".to_string()]);
        assert!(dropped.split_arcs.is_empty());
        assert!(dropped.epochs.iter().all(|epoch| dual_ambiguity_id(
            &epoch.base_observations,
            "G02"
        )
        .is_none()
            && dual_ambiguity_id(&epoch.rover_observations, "G02").is_none()));

        let split =
            prepare_dual_cycle_slip_baseline_epochs(&epochs, CycleSlipPolicy::SplitArc, options)
                .unwrap();
        assert!(split.dropped_sats.is_empty());
        assert_eq!(
            split.split_arcs,
            vec![
                CycleSlipSplitArc {
                    receiver: CycleSlipReceiver::Rover,
                    satellite_id: "G02".to_string(),
                    ambiguity_id: "G02@rover#1".to_string(),
                    start_epoch_index: 0,
                    end_epoch_index: 0,
                    n_epochs: 1,
                },
                CycleSlipSplitArc {
                    receiver: CycleSlipReceiver::Rover,
                    satellite_id: "G02".to_string(),
                    ambiguity_id: "G02@rover#2".to_string(),
                    start_epoch_index: 1,
                    end_epoch_index: 3,
                    n_epochs: 2,
                },
            ]
        );
        assert_eq!(
            dual_ambiguity_id(&split.epochs[0].rover_observations, "G02"),
            Some("G02@rover#1")
        );
        assert_eq!(
            dual_ambiguity_id(&split.epochs[1].rover_observations, "G02"),
            Some("G02@rover#2")
        );
        assert_eq!(
            dual_ambiguity_id(&split.epochs[2].rover_observations, "G02"),
            None
        );
        assert_eq!(
            dual_ambiguity_id(&split.epochs[3].rover_observations, "G02"),
            Some("G02@rover#2~ra1")
        );
        assert_eq!(
            dual_ambiguity_id(&split.epochs[3].base_observations, "G02"),
            Some("G02~ra1")
        );
    }

    #[test]
    fn dual_cycle_slip_prep_uses_threshold_and_gap_classification() {
        let epochs = vec![
            dual_slip_epoch(
                "0",
                0.0,
                vec![dual_slip_obs("G03", "G03", 10.0, 8.0, None, None)],
                vec![dual_slip_obs("G03", "G03", 11.0, 9.0, None, None)],
            ),
            dual_slip_epoch(
                "1",
                20.0,
                vec![dual_slip_obs("G03", "G03", 11.0, 8.0, None, None)],
                vec![dual_slip_obs("G03", "G03", 11.0, 9.0, None, None)],
            ),
        ];
        let options = CycleSlipOptions {
            gf_threshold_m: 0.05,
            mw_threshold_cycles: 0.5,
            min_arc_gap_s: 10.0,
        };

        assert_eq!(
            prepare_dual_cycle_slip_baseline_epochs(&epochs, CycleSlipPolicy::Error, options),
            Err(CycleSlipPrepError::CycleSlipDetected {
                receiver: CycleSlipReceiver::Base,
                satellite_id: "G03".to_string(),
                epoch_index: 1,
                reasons: vec![
                    SlipReason::DataGap,
                    SlipReason::GeometryFree,
                    SlipReason::MelbourneWubbena,
                ],
            })
        );
    }

    #[test]
    fn baseline_reference_errors_match_public_tags() {
        let base = [10.0, 0.0, 0.0];
        let multi = vec![baseline_reference_epoch(&[
            ("G01", [20.0, 0.0, 0.0]),
            ("G02", [20.0, 0.0, 0.0]),
            ("E01", [20.0, 0.0, 0.0]),
            ("E02", [20.0, 0.0, 0.0]),
        ])];
        assert_eq!(
            baseline_reference_satellites(
                base,
                &multi,
                BaselineReferenceSelection::Satellite("G01".to_string()),
            ),
            Err(DoubleDifferenceError::ReferenceSatelliteSingleSystem(
                "G01".to_string()
            ))
        );
        assert_eq!(
            baseline_reference_satellites(
                base,
                &multi,
                BaselineReferenceSelection::PerSystem(BTreeMap::from([(
                    "G".to_string(),
                    "G01".to_string(),
                )])),
            ),
            Err(DoubleDifferenceError::ReferenceSatelliteMissingSystem(
                "E".to_string()
            ))
        );

        let split = vec![
            baseline_reference_epoch(&[("G01", [20.0, 0.0, 0.0])]),
            baseline_reference_epoch(&[("G02", [20.0, 0.0, 0.0])]),
        ];
        assert_eq!(
            baseline_reference_satellites(base, &split, BaselineReferenceSelection::Auto),
            Err(DoubleDifferenceError::NoCommonReferenceSatellite(
                "G".to_string()
            ))
        );
    }

    #[test]
    fn errors_are_tagged() {
        assert_eq!(
            double_differences(
                &[obs("G01", 1.0, 2.0)],
                &[obs("G01", 1.0, 2.0)],
                ReferenceSelection::Auto
            ),
            Err(DoubleDifferenceError::TooFewCommonSatellites {
                count: 1,
                minimum: 2,
            })
        );
        assert_eq!(
            double_differences(
                &[obs("G01", 1.0, 2.0), obs("G01", 3.0, 4.0)],
                &[obs("G01", 1.0, 2.0), obs("G02", 3.0, 4.0)],
                ReferenceSelection::Auto,
            ),
            Err(DoubleDifferenceError::DuplicateObservation(
                "G01".to_string()
            ))
        );
        assert_eq!(
            double_differences(
                &[obs("G01", 1.0, 2.0), obs("G02", 3.0, 4.0)],
                &[obs("G01", 1.0, 2.0), obs("G02", 3.0, 4.0)],
                ReferenceSelection::Satellite("G99".to_string()),
            ),
            Err(DoubleDifferenceError::ReferenceSatelliteMissing(
                "G99".to_string()
            ))
        );
    }

    #[test]
    fn double_differences_reject_non_finite_observations() {
        let rover = vec![obs("G01", 10.0, 11.0), obs("G02", 20.0, 21.0)];

        assert_eq!(
            double_differences(
                &[obs("G01", 1.0, 2.0), obs("G02", f64::NAN, 4.0)],
                &rover,
                ReferenceSelection::Auto,
            ),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk observation code_m",
                reason: "not finite",
            })
        );

        assert_eq!(
            double_differences(
                &[obs("G01", 1.0, 2.0), obs("G02", 3.0, 4.0)],
                &[obs("G01", 10.0, f64::INFINITY), obs("G02", 20.0, 21.0)],
                ReferenceSelection::Auto,
            ),
            Err(DoubleDifferenceError::InvalidInput {
                field: "rtk observation phase_m",
                reason: "not finite",
            })
        );
    }

    #[test]
    fn estimates_dual_frequency_wide_lane_integers() {
        let epochs = vec![
            DualEpoch {
                observations: vec![
                    dual_pair("G01", 0.0, 1.0),
                    dual_pair("G02", 0.0, 4.0),
                    dual_pair("G03", 0.0, -1.0),
                ],
            },
            DualEpoch {
                observations: vec![
                    dual_pair("G01", 0.0, 2.0),
                    dual_pair("G02", 0.0, 5.0),
                    dual_pair("G03", 0.0, 0.0),
                ],
            },
        ];

        let fixed = estimate_wide_lane_ambiguities(
            &epochs,
            "G01",
            WideLaneOptions {
                min_epochs: 2,
                tolerance_cycles: 1.0e-9,
                skip_short_fragments: false,
            },
        )
        .unwrap();

        assert_eq!(
            fixed,
            BTreeMap::from([("G02".to_string(), 3), ("G03".to_string(), -2)])
        );
    }

    #[test]
    fn split_arc_wide_lane_skips_short_fragments() {
        let epochs = vec![
            DualEpoch {
                observations: vec![
                    dual_pair("G01", 0.0, 1.0),
                    split_dual_pair("G02", "G02@rover#1", 0.0, 4.0),
                    dual_pair("G03", 0.0, -1.0),
                ],
            },
            DualEpoch {
                observations: vec![dual_pair("G01", 0.0, 2.0), dual_pair("G03", 0.0, 0.0)],
            },
        ];

        let options = WideLaneOptions {
            min_epochs: 2,
            tolerance_cycles: 1.0e-9,
            skip_short_fragments: true,
        };
        let fixed = estimate_wide_lane_ambiguities(&epochs, "G01", options).unwrap();
        assert_eq!(fixed, BTreeMap::from([("G03".to_string(), -2)]));

        let err = estimate_wide_lane_ambiguities(
            &epochs,
            "G01",
            WideLaneOptions {
                skip_short_fragments: false,
                ..options
            },
        )
        .unwrap_err();
        assert_eq!(
            err,
            WideLaneError::TooFewWideLaneEpochs {
                ambiguity_id: "G02@rover#1|ref=G01".to_string(),
                count: 1,
                minimum: 2,
            }
        );
    }

    #[test]
    fn wide_lane_rejects_invalid_inputs_and_options() {
        let epochs = vec![DualEpoch {
            observations: vec![dual_pair("G01", 0.0, 1.0), dual_pair("G02", 0.0, 4.0)],
        }];

        assert_eq!(
            estimate_wide_lane_ambiguities(
                &epochs,
                "G01",
                WideLaneOptions {
                    min_epochs: 0,
                    tolerance_cycles: 1.0e-9,
                    skip_short_fragments: false,
                },
            ),
            Err(WideLaneError::InvalidInput {
                field: "rtk wide lane min_epochs",
                reason: "not positive",
            })
        );
        assert_eq!(
            estimate_wide_lane_ambiguities(
                &epochs,
                "G01",
                WideLaneOptions {
                    min_epochs: 1,
                    tolerance_cycles: f64::NAN,
                    skip_short_fragments: false,
                },
            ),
            Err(WideLaneError::InvalidInput {
                field: "rtk wide lane tolerance_cycles",
                reason: "not finite",
            })
        );

        let mut bad_epoch = epochs;
        bad_epoch[0].observations[1].rover.p1_m = f64::INFINITY;
        assert_eq!(
            estimate_wide_lane_ambiguities(
                &bad_epoch,
                "G01",
                WideLaneOptions {
                    min_epochs: 1,
                    tolerance_cycles: 1.0e-9,
                    skip_short_fragments: false,
                },
            ),
            Err(WideLaneError::InvalidInput {
                field: "rtk wide lane p1_m",
                reason: "not finite",
            })
        );
    }

    #[test]
    fn builds_ionosphere_free_epochs_and_narrow_lane_params() {
        let epochs = vec![
            DualIonosphereFreeEpoch {
                observations: vec![
                    if_pair(
                        "G01",
                        20_000_000.0,
                        20_000_020.0,
                        105_100_000.0,
                        105_100_040.0,
                    ),
                    if_pair(
                        "G02",
                        21_000_000.0,
                        21_000_035.0,
                        110_200_000.0,
                        110_200_090.0,
                    ),
                    if_pair(
                        "G03",
                        22_000_000.0,
                        22_000_055.0,
                        115_300_000.0,
                        115_300_120.0,
                    ),
                ],
            },
            DualIonosphereFreeEpoch {
                observations: vec![
                    if_pair(
                        "G01",
                        20_000_100.0,
                        20_000_120.0,
                        105_100_500.0,
                        105_100_540.0,
                    ),
                    if_pair(
                        "G02",
                        21_000_100.0,
                        21_000_135.0,
                        110_200_500.0,
                        110_200_590.0,
                    ),
                    if_pair(
                        "G03",
                        22_000_100.0,
                        22_000_155.0,
                        115_300_500.0,
                        115_300_620.0,
                    ),
                ],
            },
        ];
        let wide_lanes = BTreeMap::from([("G02".to_string(), 3), ("G03".to_string(), -5)]);

        let result = build_ionosphere_free_baseline_epochs(&epochs, "G01", &wide_lanes).unwrap();

        assert_eq!(
            result
                .wavelengths_m
                .iter()
                .map(|(id, value)| (id.as_str(), value.to_bits()))
                .collect::<Vec<_>>(),
            [("G02", 0x3fbb614bed5136b9), ("G03", 0x3fbb614bed5136b9),]
        );
        assert_eq!(
            result
                .offsets_m
                .iter()
                .map(|(id, value)| (id.as_str(), value.to_bits()))
                .collect::<Vec<_>>(),
            [("G02", 0x3ff21e814dfd4618), ("G03", 0xbffe32d781fb74d4),]
        );
        assert_eq!(result.epochs.len(), 2);
        assert_eq!(
            result
                .epochs
                .iter()
                .map(|epoch| (
                    epoch.epoch_index,
                    epoch.satellite_ids.clone(),
                    epoch
                        .base_observations
                        .iter()
                        .map(|obs| (
                            obs.satellite_id.as_str(),
                            obs.ambiguity_id.as_str(),
                            obs.code_m.to_bits(),
                            obs.phase_m.to_bits()
                        ))
                        .collect::<Vec<_>>(),
                    epoch
                        .rover_observations
                        .iter()
                        .map(|obs| (
                            obs.satellite_id.as_str(),
                            obs.ambiguity_id.as_str(),
                            obs.code_m.to_bits(),
                            obs.phase_m.to_bits()
                        ))
                        .collect::<Vec<_>>()
                ))
                .collect::<Vec<_>>(),
            vec![
                (
                    0,
                    vec!["G01".to_string(), "G02".to_string(), "G03".to_string()],
                    vec![
                        ("G01", "G01", 0x417312cfca8965e4, 0x416570ac29af7848),
                        ("G02", "G02", 0x417406f3ca8965e4, 0x41667b02f0ff8bd8),
                        ("G03", "G03", 0x4174fb17ca8965e4, 0x41678559b84f9f64),
                    ],
                    vec![
                        ("G01", "G01", 0x417312d0fa2bbf5f, 0x416570ac9e819db4),
                        ("G02", "G02", 0x417406f5ea2bbf5e, 0x41667b0410f1cbd0),
                        ("G03", "G03", 0x4174fb1b2a2bbf5e, 0x4167855b3eeebc18),
                    ],
                ),
                (
                    1,
                    vec!["G01".to_string(), "G02".to_string(), "G03".to_string()],
                    vec![
                        ("G01", "G01", 0x417312d60a8965e4, 0x416570b2d8f081bc),
                        ("G02", "G02", 0x417406fa0a8965e4, 0x41667b09a0409544),
                        ("G03", "G03", 0x4174fb1e0a8965e4, 0x416785606790a8d4),
                    ],
                    vec![
                        ("G01", "G01", 0x417312d73a2bbf5f, 0x416570b34dc2a728),
                        ("G02", "G02", 0x417406fc2a2bbf5e, 0x41667b0ac032d540),
                        ("G03", "G03", 0x4174fb216a2bbf5e, 0x41678561ee2fc588),
                    ],
                ),
            ]
        );
    }

    #[test]
    fn ionosphere_free_builders_reject_invalid_tropo_and_setup_inputs() {
        let mut epochs = vec![DualIonosphereFreeEpoch {
            observations: vec![
                if_pair(
                    "G01",
                    20_000_000.0,
                    20_000_020.0,
                    105_100_000.0,
                    105_100_040.0,
                ),
                if_pair(
                    "G02",
                    21_000_000.0,
                    21_000_035.0,
                    110_200_000.0,
                    110_200_090.0,
                ),
            ],
        }];
        epochs[0].observations[1].rover.tropo_m = f64::NAN;
        assert_eq!(
            build_ionosphere_free_baseline_epochs(
                &epochs,
                "G01",
                &BTreeMap::from([("G02".to_string(), 3)]),
            ),
            Err(IonosphereFreeBaselineError::InvalidInput {
                field: "rtk if tropo_m",
                reason: "not finite",
            })
        );

        let setup_epochs = vec![DualIonosphereFreeSetupEpoch {
            jd_whole: 2_460_100.5,
            jd_fraction: 2.0,
            observations: vec![dual_pair("G01", 0.0, 1.0), dual_pair("G02", 0.0, 4.0)],
            base_satellite_positions_m: BTreeMap::from([
                ("G01".to_string(), [20.0, 0.0, 0.0]),
                ("G02".to_string(), [30.0, 0.0, 0.0]),
            ]),
            rover_satellite_positions_m: BTreeMap::from([
                ("G01".to_string(), [20.0, 0.0, 0.0]),
                ("G02".to_string(), [30.0, 0.0, 0.0]),
            ]),
        }];
        assert_eq!(
            prepare_ionosphere_free_baseline_epochs(
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                &setup_epochs,
                "G01",
                &BTreeMap::from([("G02".to_string(), 3)]),
                false,
            ),
            Err(IonosphereFreeBaselineError::InvalidInput {
                field: "rtk if setup jd_fraction",
                reason: "out of range",
            })
        );

        let tropo_epochs = vec![DualIonosphereFreeSetupEpoch {
            jd_whole: 2_460_100.5,
            jd_fraction: 0.0,
            observations: vec![dual_pair("G01", 0.0, 1.0), dual_pair("G02", 0.0, 4.0)],
            base_satellite_positions_m: BTreeMap::from([
                ("G01".to_string(), [10.0, 0.0, 0.0]),
                ("G02".to_string(), [30.0, 0.0, 0.0]),
            ]),
            rover_satellite_positions_m: BTreeMap::from([
                ("G01".to_string(), [20.0, 0.0, 0.0]),
                ("G02".to_string(), [30.0, 0.0, 0.0]),
            ]),
        }];
        assert_eq!(
            prepare_ionosphere_free_baseline_epochs(
                [10.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                &tropo_epochs,
                "G01",
                &BTreeMap::from([("G02".to_string(), 3)]),
                true,
            ),
            Err(IonosphereFreeBaselineError::InvalidInput {
                field: "rtk tropo line of sight_m",
                reason: "degenerate geometry",
            })
        );
    }

    #[test]
    fn ionosphere_free_setup_rejects_invalid_julian_split_without_panic() {
        let setup_epochs = vec![DualIonosphereFreeSetupEpoch {
            jd_whole: f64::NAN,
            jd_fraction: 0.0,
            observations: vec![dual_pair("G01", 0.0, 1.0), dual_pair("G02", 0.0, 4.0)],
            base_satellite_positions_m: BTreeMap::from([
                (
                    "G01".to_string(),
                    [20_200_000.0, 14_000_000.0, 21_700_000.0],
                ),
                (
                    "G02".to_string(),
                    [21_200_000.0, 13_000_000.0, 20_700_000.0],
                ),
            ]),
            rover_satellite_positions_m: BTreeMap::from([
                (
                    "G01".to_string(),
                    [20_200_100.0, 14_000_000.0, 21_700_000.0],
                ),
                (
                    "G02".to_string(),
                    [21_200_100.0, 13_000_000.0, 20_700_000.0],
                ),
            ]),
        }];

        let result = std::panic::catch_unwind(|| {
            prepare_ionosphere_free_baseline_epochs(
                [6_378_137.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                &setup_epochs,
                "G01",
                &BTreeMap::from([("G02".to_string(), 3)]),
                true,
            )
        });

        assert!(result.is_ok(), "invalid Julian split must not panic");
        assert_eq!(
            result.expect("invalid Julian split should not unwind"),
            Err(IonosphereFreeBaselineError::InvalidInput {
                field: "rtk if setup jd_whole",
                reason: "not finite",
            })
        );
    }

    #[test]
    fn ionosphere_free_setup_rejects_nonfinite_tropo_receiver_without_panic() {
        let setup_epochs = vec![DualIonosphereFreeSetupEpoch {
            jd_whole: 2_460_100.5,
            jd_fraction: 0.0,
            observations: vec![dual_pair("G01", 0.0, 1.0), dual_pair("G02", 0.0, 4.0)],
            base_satellite_positions_m: BTreeMap::from([
                (
                    "G01".to_string(),
                    [20_200_000.0, 14_000_000.0, 21_700_000.0],
                ),
                (
                    "G02".to_string(),
                    [21_200_000.0, 13_000_000.0, 20_700_000.0],
                ),
            ]),
            rover_satellite_positions_m: BTreeMap::from([
                (
                    "G01".to_string(),
                    [20_200_100.0, 14_000_000.0, 21_700_000.0],
                ),
                (
                    "G02".to_string(),
                    [21_200_100.0, 13_000_000.0, 20_700_000.0],
                ),
            ]),
        }];

        let result = std::panic::catch_unwind(|| {
            prepare_ionosphere_free_baseline_epochs(
                [f64::NAN, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                &setup_epochs,
                "G01",
                &BTreeMap::from([("G02".to_string(), 3)]),
                true,
            )
        });

        assert!(result.is_ok(), "non-finite tropo receiver must not panic");
        assert_eq!(
            result.expect("non-finite tropo receiver should not unwind"),
            Err(IonosphereFreeBaselineError::InvalidInput {
                field: "rtk tropo base position_m",
                reason: "not finite",
            })
        );
    }

    #[test]
    fn ionosphere_free_setup_handles_antimeridian_tropo_receiver_without_panic() {
        let setup_epochs = vec![DualIonosphereFreeSetupEpoch {
            jd_whole: 2_460_100.5,
            jd_fraction: 0.0,
            observations: vec![dual_pair("G01", 0.0, 1.0), dual_pair("G02", 0.0, 4.0)],
            base_satellite_positions_m: BTreeMap::from([
                (
                    "G01".to_string(),
                    [-20_200_000.0, -14_000_000.0, 21_700_000.0],
                ),
                (
                    "G02".to_string(),
                    [-21_200_000.0, -13_000_000.0, 20_700_000.0],
                ),
            ]),
            rover_satellite_positions_m: BTreeMap::from([
                (
                    "G01".to_string(),
                    [-20_200_100.0, -14_000_000.0, 21_700_000.0],
                ),
                (
                    "G02".to_string(),
                    [-21_200_100.0, -13_000_000.0, 20_700_000.0],
                ),
            ]),
        }];

        let result = std::panic::catch_unwind(|| {
            prepare_ionosphere_free_baseline_epochs(
                [-6_378_137.0, -0.0, 0.0],
                [1.0, 0.0, 0.0],
                &setup_epochs,
                "G01",
                &BTreeMap::from([("G02".to_string(), 3)]),
                true,
            )
        });

        assert!(result.is_ok(), "antimeridian tropo receiver must not panic");
        result
            .expect("antimeridian tropo receiver should not unwind")
            .expect("antimeridian tropo receiver should prepare IF epochs");
    }

    #[test]
    fn ionosphere_free_epoch_builder_skips_missing_wide_lane_fragments() {
        let epochs = vec![DualIonosphereFreeEpoch {
            observations: vec![
                if_pair(
                    "G01",
                    20_000_000.0,
                    20_000_020.0,
                    105_100_000.0,
                    105_100_040.0,
                ),
                if_pair(
                    "G02",
                    21_000_000.0,
                    21_000_035.0,
                    110_200_000.0,
                    110_200_090.0,
                ),
                if_pair(
                    "G03",
                    22_000_000.0,
                    22_000_055.0,
                    115_300_000.0,
                    115_300_120.0,
                ),
            ],
        }];
        let wide_lanes = BTreeMap::from([("G02".to_string(), 3)]);

        let result = build_ionosphere_free_baseline_epochs(&epochs, "G01", &wide_lanes).unwrap();

        assert_eq!(result.epochs[0].satellite_ids, ["G01", "G02"]);
        assert_eq!(
            result.wavelengths_m.keys().collect::<Vec<_>>(),
            [&"G02".to_string()]
        );
    }

    #[test]
    fn wide_lane_errors_on_equal_frequencies() {
        let mut bad = dual_pair("G02", 0.0, 4.0);
        bad.base.f2_hz = gps_l1_hz();
        let epochs = vec![DualEpoch {
            observations: vec![dual_pair("G01", 0.0, 1.0), bad],
        }];

        let err = estimate_wide_lane_ambiguities(
            &epochs,
            "G01",
            WideLaneOptions {
                min_epochs: 1,
                tolerance_cycles: 0.5,
                skip_short_fragments: false,
            },
        )
        .unwrap_err();

        assert_eq!(
            err,
            WideLaneError::WideLaneFailed {
                satellite_id: "G02".to_string(),
                reason: CarrierPhaseError::EqualFrequencies,
            }
        );
    }
}
