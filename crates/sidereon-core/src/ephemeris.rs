//! Ephemeris products and satellite orbit/clock evaluation.
//!
//! This is the main public home for loaded GNSS ephemeris data. Use
//! [`Sp3`] for precise SP3 products and [`BroadcastEphemeris`] for broadcast
//! navigation products parsed from RINEX NAV files (or built from records decoded
//! off the air with [`BroadcastRecord::from_lnav`]). Both implement
//! [`EphemerisSource`] so they can feed [`crate::positioning::solve`]; the
//! broadcast-only real-time/offline path is
//! [`solve_broadcast`](crate::positioning::solve_broadcast) and the
//! precise-with-broadcast-fallback path is
//! [`solve_with_fallback`](crate::positioning::solve_with_fallback).

pub use crate::artifact_bytes::DigestProvenance;
pub use crate::broadcast::{
    eccentric_anomaly, relativistic_clock_correction_s, satellite_clock_offset_s,
    satellite_position_ecef, satellite_position_ecef_cnav, satellite_state, satellite_state_cnav,
    ClockOffset, ClockPolynomial, CnavRates, ConstellationConstants, EccentricAnomaly,
    KeplerianElements, OrbitState, SatelliteState,
};
pub use crate::observables::{
    is_observable_state_gap, observable_states_at_j2000_s, observable_states_at_shared_j2000_s,
    ObservableEphemerisSource, ObservableStateBatch, ObservableStateElementStatus,
    ObservablesError, OBSERVABLE_STATE_MISSING_POSITION_ECEF_M,
};
use crate::observables::{ObservableState, ObservablesInputErrorKind};
pub use crate::orbit_determination::{
    fit_all_sp3_ecef_precise_orbits, fit_all_sp3_precise_orbits,
    fit_precise_ephemeris_sample_orbit, fit_precise_ephemeris_sample_orbit_with_initial_state,
    fit_precise_ephemeris_sample_orbits, fit_precise_ephemeris_state_sample_orbit,
    fit_precise_ephemeris_state_sample_orbits, fit_sp3_ecef_precise_orbit,
    fit_sp3_ecef_precise_orbits, fit_sp3_precise_orbit, fit_sp3_precise_orbit_with_initial_state,
    fit_sp3_precise_orbits, OrbitArcSpan, OrbitFitCovariance, OrbitFitError, OrbitFitOptions,
    OrbitFitReport, OrbitFitSolution, OrbitResidualLedger, OrbitResidualStats,
    OrientedPreciseEphemerisStateSample,
};
pub use crate::rinex_nav::{
    cnav_ura_ned_m, cnav_ura_nominal_m, is_beidou_geo, BroadcastGroupDelayTerm,
    BroadcastGroupDelays, BroadcastIssue, BroadcastRecord, CnavParameters, CnavSignal,
    GlonassRecord, IonoCorrections, KlobucharAlphaBeta, LnavRecordError, NavMessage,
    NavMessagePreference,
};
pub use crate::sp3::{
    align_clock_reference, clock_reference_offset, compare_position_series, merge, parse_exact_sp3,
    precise_interpolant_store_checksum64, sp3_ecef_state_to_eci, validate_exact_sp3,
    AgreementMetric, ClockReferenceOffset, EpochAgreement, ExactSp3Coverage, ExactSp3Request,
    ExactSp3ValidationError, InterpolationComparison, InterpolationDivergence, MergeCombine,
    MergeFlag, MergeOptions, MergePrecedenceScope, MergeReport, MmapPreciseEphemerisInterpolant,
    OutlierRejectOptions, PreciseEphemerisInterpolant, PreciseEphemerisSample,
    PreciseEphemerisSamples, PreciseEphemerisStateSample, PreciseInterpolantError,
    PreciseInterpolantStoreError, PreciseSamplesError, ReferenceState, Sp3, Sp3ArtifactIdentity,
    Sp3DataType, Sp3EpochPrediction, Sp3Flags, Sp3FrameLabelSet, Sp3FrameReconciliation,
    Sp3FrameReconciliationMethod, Sp3FrameReconciliationOptions, Sp3Header, Sp3MergeInputIdentity,
    Sp3MergeInputIdentityError, Sp3PredictionSummary, Sp3State, Sp3TimeSystem, Sp3Version,
    SP3_MERGE_INPUT_ID_PREFIX, SP3_MERGE_INPUT_SCHEMA_VERSION,
};
pub use crate::sp3::{
    check_continuity, ContinuityCheck, ContinuityDefect, ContinuityOptions, ContinuityReport,
    OrbitClass, SpeedBound,
};
pub use crate::sp3::{
    CellProvenance, CellSelection, ContributorCoverage, MergeContinuityReport,
    MergeContinuityViolation, MergeProvenance, PrecedenceTransition, ProvenanceMode,
    TransitionReason,
};
pub use crate::spp::EphemerisSource;
use crate::{validate, GnssSatelliteId, GnssSystem};

/// Broadcast navigation ephemeris store selected by satellite and query epoch.
///
/// The underlying implementation type is `BroadcastStore`; the public alias
/// names the role rather than the storage detail.
pub type BroadcastEphemeris = crate::rinex_nav::BroadcastStore;

/// Acronym-preserving alias for users who prefer the format name spelling.
///
/// Rust item names normally use `Sp3`; this alias keeps `SP3` available without
/// making the implementation type fight Rust naming conventions.
#[allow(clippy::upper_case_acronyms)]
pub type SP3 = Sp3;

/// Status for one source-agnostic ephemeris sample row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EphemerisSampleStatus {
    /// The source returned a usable ECEF position for this satellite and epoch.
    Valid,
    /// The source had no data, or the query fell outside a valid fit interval.
    Gap,
}

/// One satellite and epoch in a regular ephemeris sample grid.
///
/// For [`EphemerisSampleStatus::Valid`], `position_ecef_m` is `Some` and
/// `clock_s` carries the source clock when available. For
/// [`EphemerisSampleStatus::Gap`], both value fields are `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EphemerisSampleRow {
    /// The sampled satellite.
    pub sat: GnssSatelliteId,
    /// Query epoch, seconds since J2000 in the source time scale.
    pub epoch_j2000_s: f64,
    /// Whether this row carries data or a gap marker.
    pub status: EphemerisSampleStatus,
    /// Satellite ECEF position, meters, when `status` is `Valid`.
    pub position_ecef_m: Option<[f64; 3]>,
    /// Satellite clock offset, seconds, when supplied by the source.
    pub clock_s: Option<f64>,
}

impl EphemerisSampleRow {
    /// Build a valid ephemeris sample row.
    pub const fn valid(
        sat: GnssSatelliteId,
        epoch_j2000_s: f64,
        position_ecef_m: [f64; 3],
        clock_s: Option<f64>,
    ) -> Self {
        Self {
            sat,
            epoch_j2000_s,
            status: EphemerisSampleStatus::Valid,
            position_ecef_m: Some(position_ecef_m),
            clock_s,
        }
    }

    /// Build a gap ephemeris sample row.
    pub const fn gap(sat: GnssSatelliteId, epoch_j2000_s: f64) -> Self {
        Self {
            sat,
            epoch_j2000_s,
            status: EphemerisSampleStatus::Gap,
            position_ecef_m: None,
            clock_s: None,
        }
    }

    /// Whether this row is a gap marker.
    pub const fn is_gap(&self) -> bool {
        matches!(self.status, EphemerisSampleStatus::Gap)
    }
}

/// Sample an ephemeris source on a regular inclusive time grid.
///
/// Rows are returned in satellite-major order: for each satellite in
/// `satellites`, the function emits `start_j2000_s`, `start_j2000_s + step_s`,
/// and so on up to `stop_j2000_s`, with a final row snapped to `stop_j2000_s`
/// when the step does not land exactly on the end. If `start_j2000_s >
/// stop_j2000_s`, the returned grid is empty.
///
/// Missing satellite data and out-of-fit-interval epochs are represented as
/// [`EphemerisSampleStatus::Gap`] rows, not as errors. Invalid inputs and invalid
/// source states remain errors.
pub fn sample(
    source: &dyn ObservableEphemerisSource,
    satellites: &[GnssSatelliteId],
    start_j2000_s: f64,
    stop_j2000_s: f64,
    step_s: f64,
) -> core::result::Result<Vec<EphemerisSampleRow>, ObservablesError> {
    let epochs = sample_epochs(start_j2000_s, stop_j2000_s, step_s)?;
    let mut rows = Vec::with_capacity(satellites.len().saturating_mul(epochs.len()));
    for &sat in satellites {
        for &epoch_j2000_s in &epochs {
            rows.push(sample_one(source, sat, epoch_j2000_s)?);
        }
    }
    Ok(rows)
}

/// Select a broadcast TGD/BGD value from a parsed delay set.
///
/// This is a function-form accessor for [`BroadcastGroupDelays::get`], useful
/// when composing the group-delay term independently from a full broadcast clock
/// evaluation.
pub const fn broadcast_group_delay_s(
    delays: &BroadcastGroupDelays,
    term: BroadcastGroupDelayTerm,
) -> Option<f64> {
    delays.get(term)
}

/// Select the broadcast group delay used by a navigation message's clock model.
///
/// This is a function-form accessor for [`BroadcastGroupDelays::for_message`].
/// It returns `None` when the supplied delay set does not carry a term for the
/// requested constellation/message pair.
pub const fn broadcast_message_group_delay_s(
    delays: BroadcastGroupDelays,
    system: GnssSystem,
    message: NavMessage,
) -> Option<f64> {
    delays.for_message(system, message)
}

/// Group delay selected by a parsed broadcast navigation record's clock model.
///
/// This is a function-form accessor for
/// [`BroadcastRecord::broadcast_clock_group_delay_s`]. It returns zero when the
/// record carries no applicable TGD/BGD term, matching the broadcast clock
/// evaluator.
pub fn broadcast_record_group_delay_s(record: &BroadcastRecord) -> f64 {
    record.broadcast_clock_group_delay_s()
}

fn sample_epochs(
    start_j2000_s: f64,
    stop_j2000_s: f64,
    step_s: f64,
) -> core::result::Result<Vec<f64>, ObservablesError> {
    validate::finite(start_j2000_s, "start_j2000_s").map_err(map_sample_input_error)?;
    validate::finite(stop_j2000_s, "stop_j2000_s").map_err(map_sample_input_error)?;
    validate::finite_positive(step_s, "step_s").map_err(map_sample_input_error)?;

    if start_j2000_s > stop_j2000_s {
        return Ok(Vec::new());
    }

    let mut epochs = Vec::new();
    let mut step_index = 0usize;
    loop {
        let epoch = start_j2000_s + step_s * step_index as f64;
        if epoch > stop_j2000_s {
            break;
        }
        if epochs.last().is_some_and(|last| epoch <= *last) {
            return Err(sample_input_error(
                "step_s",
                ObservablesInputErrorKind::OutOfRange,
            ));
        }
        epochs.push(epoch);
        step_index = step_index
            .checked_add(1)
            .ok_or_else(|| sample_input_error("step_s", ObservablesInputErrorKind::OutOfRange))?;
    }

    if let Some(&last) = epochs.last() {
        if last < stop_j2000_s {
            epochs.push(stop_j2000_s);
        }
    }

    Ok(epochs)
}

fn sample_one(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    epoch_j2000_s: f64,
) -> core::result::Result<EphemerisSampleRow, ObservablesError> {
    match source.observable_state_at_j2000_s(sat, epoch_j2000_s) {
        Ok(state) => sample_row_from_state(sat, epoch_j2000_s, state),
        Err(error) if is_observable_state_gap(&error) => {
            Ok(EphemerisSampleRow::gap(sat, epoch_j2000_s))
        }
        Err(error) => Err(error),
    }
}

fn sample_row_from_state(
    sat: GnssSatelliteId,
    epoch_j2000_s: f64,
    state: ObservableState,
) -> core::result::Result<EphemerisSampleRow, ObservablesError> {
    validate::finite_vec3(state.position_ecef_m, "observable state position_ecef_m")
        .map_err(map_sample_input_error)?;
    if let Some(clock_s) = state.clock_s {
        validate::finite(clock_s, "observable state clock_s").map_err(map_sample_input_error)?;
    }
    Ok(EphemerisSampleRow::valid(
        sat,
        epoch_j2000_s,
        state.position_ecef_m,
        state.clock_s,
    ))
}

fn map_sample_input_error(error: validate::FieldError) -> ObservablesError {
    sample_input_error(error.field(), ObservablesInputErrorKind::from(&error))
}

fn sample_input_error(field: &'static str, kind: ObservablesInputErrorKind) -> ObservablesError {
    ObservablesError::InvalidInput { field, kind }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_group_delay_free_functions_delegate_to_delay_set() {
        let gps = BroadcastGroupDelays::gps_lnav(-2.3e-9);
        assert_eq!(
            broadcast_group_delay_s(&gps, BroadcastGroupDelayTerm::GpsTgd),
            Some(-2.3e-9)
        );
        assert_eq!(
            broadcast_message_group_delay_s(gps, GnssSystem::Gps, NavMessage::GpsLnav),
            Some(-2.3e-9)
        );

        let galileo = BroadcastGroupDelays::galileo(1.0e-9, 2.0e-9);
        assert_eq!(
            broadcast_message_group_delay_s(galileo, GnssSystem::Galileo, NavMessage::GalileoFnav),
            Some(1.0e-9)
        );
        assert_eq!(
            broadcast_message_group_delay_s(galileo, GnssSystem::Galileo, NavMessage::GalileoInav),
            Some(2.0e-9)
        );
    }

    #[test]
    fn broadcast_record_group_delay_free_function_delegates_to_record() {
        let record = BroadcastRecord {
            satellite_id: crate::GnssSatelliteId::new(GnssSystem::Galileo, 1)
                .expect("valid satellite"),
            message: NavMessage::GalileoInav,
            issue_of_data: BroadcastIssue {
                issue: 0,
                message: NavMessage::GalileoInav,
            },
            week: 2_400,
            toe: crate::astro::time::model::GnssWeekTow::new(
                crate::astro::time::model::TimeScale::Gst,
                2_400,
                100_000.0,
            )
            .expect("valid toe"),
            toc: crate::astro::time::model::GnssWeekTow::new(
                crate::astro::time::model::TimeScale::Gst,
                2_400,
                100_000.0,
            )
            .expect("valid toc"),
            elements: KeplerianElements {
                sqrt_a: 5_440.0,
                e: 0.01,
                m0: 0.1,
                delta_n: 0.0,
                omega0: 0.2,
                i0: 0.94,
                omega: 0.3,
                omega_dot: -8.0e-9,
                idot: 0.0,
                cuc: 0.0,
                cus: 0.0,
                crc: 0.0,
                crs: 0.0,
                cic: 0.0,
                cis: 0.0,
                toe_sow: 100_000.0,
            },
            clock: ClockPolynomial {
                af0: 0.0,
                af1: 0.0,
                af2: 0.0,
                toc_sow: 100_000.0,
            },
            group_delays: BroadcastGroupDelays::galileo(1.0e-9, 2.5e-9),
            cnav: None,
            sv_health: 0.0,
            sv_accuracy_m: 1.0,
            fit_interval_s: None,
        };

        assert_eq!(
            broadcast_record_group_delay_s(&record).to_bits(),
            record.broadcast_clock_group_delay_s().to_bits()
        );
        assert_eq!(
            broadcast_record_group_delay_s(&record).to_bits(),
            2.5e-9_f64.to_bits()
        );
    }
}
