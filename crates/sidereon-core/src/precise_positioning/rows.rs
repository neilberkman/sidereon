//! Shared undifferenced PPP row and residual assembly.
//!
//! The float and fixed static PPP solves build the SAME undifferenced code/phase
//! model: the forward-predicted geometric range, satellite-clock and range
//! corrections, the prefit code/phase residuals, and the elevation-weighted
//! measurement weights. They differ only in how the carrier ambiguity is bound.
//! The float solve ESTIMATES one ambiguity per satellite (a design column the
//! solve adjusts); the fixed solve HOLDS the integer ambiguity (folded into the
//! phase prefit, no column). That single degree of freedom is the
//! [`AmbiguityBinding`] recipe.
//!
//! This module owns the one shared assembly that both clusters route through:
//! [`build_rows`] emits the weighted design rows the normal-equation solve
//! consumes, [`residual_rows`] emits the post-fit residual rows the finalize and
//! screening paths consume, and both share [`UndiffModel`] so the measurement
//! model is evaluated by exactly one piece of code. The dense design-row column
//! layout itself lives in the substrate
//! ([`crate::estimation::substrate::parameters::undifferenced_design_row`]).

use std::collections::BTreeMap;

use crate::ambiguity::AmbiguityId;
use crate::estimation::substrate::parameters::{
    undifferenced_design_row, UndifferencedDesignOptions,
};
use crate::observables::{
    flight_time_seed_s, transmit_epoch_j2000_s, transmit_velocity_m_s, ObservableEphemerisSource,
    ObservablesError, TransmitGeometry, TransmitTimeOptions,
};
use crate::ssr::{SsrBiasStatus, SsrSolution};
use crate::validate::{self, FieldError};

use super::model::{
    measurement_weight, model_troposphere, phase_bias_m, phase_windup_m, range_corrections_m,
    satellite_clock_m, ssr_code_bias_m, CorrectedObservation,
};
use super::normal::Row;
use super::{
    estimates_tropo_gradients, estimates_ztd, invalid_clock_count, invalid_input, no_ephemeris,
    observation_geometry, predict_default, validate_state_clock_count, FixedSolveError, FloatEpoch,
    FloatObservation, FloatResidual, FloatSolveError, FloatState, MissingCorrection, ModelContext,
    PppCorrectionLookup, RangeCorrections, SsrBiasExclusion, SsrBiasExclusionStage, SsrBiasRecord,
    SsrIfCombinationStatus, SsrTransmitTimeFailure,
};

/// Drop every epoch with no observations left, returning the remaining epochs and, for
/// each, its index in `epochs`.
///
/// A static solve estimates one receiver clock per epoch; an epoch with no observations
/// has no rows for its clock, which would leave the normal equations singular. The
/// returned indices key the remaining epochs' corrections
/// (`ModelContext::correction_epoch_indices`) and are reported with the solution.
pub(super) fn drop_empty_epochs(epochs: Vec<FloatEpoch>) -> (Vec<FloatEpoch>, Vec<usize>) {
    epochs
        .into_iter()
        .enumerate()
        .filter(|(_, epoch)| !epoch.observations.is_empty())
        .map(|(epoch_index, epoch)| (epoch, epoch_index))
        .unzip()
}

/// Leave out every observation whose required SSR/HAS bias is absent from `lookup`, or
/// whose recorded biases do not hold at its transmission time
/// ([`ssr_bias_records_hold`]), returning the retained epochs and one
/// [`SsrBiasExclusion`] per observation left out. `epochs[i]` is the caller's epoch
/// `first_epoch_index + i`, the index that keys its corrections and is reported in the
/// exclusion. The transmission time is predicted from `source` and
/// `receiver_position_m`, the solve's starting position.
///
/// Epochs keep their positions, even when all their observations are left out, because
/// the correction lookups are keyed by the caller's epoch index. The retained observations
/// are solved as if the excluded ones had not been supplied. [`build_rows`] and
/// [`residual_rows`] still refuse an observation whose required bias is absent or no
/// longer holds at the transmission time of the iteration, so a caller that skips this
/// step fails closed instead of solving without the bias.
///
/// A satellite state the source refuses because producing it reads UT1 outside the UT1
/// table under a strict UT1 policy is not an exclusion: the pass returns
/// [`FloatSolveError::Ut1OutsideCoverage`], whether the refusal was met here, predicting
/// the transmission time or checking the recorded biases, or when `lookup` was built
/// ([`SsrIfCombinationStatus::Ut1OutsideCoverage`] in its application report).
pub(super) fn exclude_unresolved_ssr_bias_observations(
    source: &dyn ObservableEphemerisSource,
    epochs: &[FloatEpoch],
    first_epoch_index: usize,
    receiver_position_m: [f64; 3],
    lookup: &PppCorrectionLookup,
    pass: usize,
    stage: SsrBiasExclusionStage,
) -> Result<(Vec<FloatEpoch>, Vec<SsrBiasExclusion>), FloatSolveError> {
    let mut exclusions = Vec::new();
    let mut refusal = None;
    let retained = epochs
        .iter()
        .enumerate()
        .map(|(offset, epoch)| {
            let epoch_index = first_epoch_index + offset;
            let mut epoch_out = epoch.clone();
            epoch_out.observations.retain(|obs| {
                let key = (obs.sat, epoch_index, obs.ambiguity_id.clone());
                let code_bias_missing =
                    lookup.ssr_code_bias_enabled && !lookup.ssr_code_bias_m.contains_key(&key);
                let phase_bias_missing =
                    lookup.phase_bias_enabled && !lookup.phase_bias_m.contains_key(&key);
                if let Some(reason) = lookup_ut1_refusal(lookup, obs, epoch_index) {
                    refusal = refusal.or(Some(reason));
                    return true;
                }
                let transmit_time_failure = if code_bias_missing || phase_bias_missing {
                    None
                } else {
                    let transmit_time = match predict_default(source, obs) {
                        Ok(options) => match transmit_epoch_j2000_s(
                            source,
                            obs.sat,
                            receiver_position_m,
                            epoch.t_rx_j2000_s,
                            TransmitTimeOptions {
                                light_time: options.light_time,
                                sagnac: options.sagnac,
                            },
                            flight_time_seed_s(obs.code_m),
                        ) {
                            Ok(t_tx) => Some(t_tx),
                            Err(ObservablesError::Ephemeris(crate::Error::Ut1OutsideCoverage(
                                reason,
                            ))) => {
                                refusal = refusal.or(Some(reason));
                                return true;
                            }
                            Err(_) => None,
                        },
                        Err(_) => None,
                    };
                    match ssr_bias_records_hold(source, obs, epoch_index, lookup, transmit_time) {
                        Ok(held) => held.err(),
                        Err(FloatSolveError::Ut1OutsideCoverage(reason)) => {
                            refusal = refusal.or(Some(reason));
                            return true;
                        }
                        Err(_) => None,
                    }
                };
                if !code_bias_missing && !phase_bias_missing && transmit_time_failure.is_none() {
                    return true;
                }
                exclusions.push(ssr_bias_exclusion(
                    lookup,
                    obs,
                    epoch_index,
                    SsrBiasShortfall {
                        code_bias_missing,
                        phase_bias_missing,
                        transmit_time_failure,
                    },
                    pass,
                    stage,
                ));
                false
            });
            epoch_out
        })
        .collect();
    match refusal {
        Some(reason) => Err(FloatSolveError::Ut1OutsideCoverage(reason)),
        None => Ok((retained, exclusions)),
    }
}

/// The UT1 refusal the application report of `lookup` records for `obs` in caller epoch
/// `epoch_index`, when [`PppCorrectionLookup::with_ssr_biases`] met one building it.
fn lookup_ut1_refusal(
    lookup: &PppCorrectionLookup,
    obs: &FloatObservation,
    epoch_index: usize,
) -> Option<crate::astro::time::DegradeReason> {
    let report = lookup.ssr_bias_report.as_ref()?;
    report
        .observation_reports
        .iter()
        .filter(|row| {
            row.epoch_index == epoch_index
                && row.sat == obs.sat
                && row.ambiguity_id == obs.ambiguity_id
        })
        .find_map(|row| {
            [row.code_status, row.phase_status]
                .into_iter()
                .find_map(|status| match status {
                    SsrIfCombinationStatus::Ut1OutsideCoverage(reason) => Some(reason),
                    _ => None,
                })
        })
}

/// Why an observation's required SSR/HAS biases do not hold.
pub(super) struct SsrBiasShortfall {
    pub(super) code_bias_missing: bool,
    pub(super) phase_bias_missing: bool,
    pub(super) transmit_time_failure: Option<SsrTransmitTimeFailure>,
}

/// The exclusion of `obs` in caller epoch `epoch_index`, with the application report row
/// `lookup` holds for it, found on solve pass `pass`.
pub(super) fn ssr_bias_exclusion(
    lookup: &PppCorrectionLookup,
    obs: &FloatObservation,
    epoch_index: usize,
    shortfall: SsrBiasShortfall,
    pass: usize,
    stage: SsrBiasExclusionStage,
) -> SsrBiasExclusion {
    let application = lookup.ssr_bias_report.as_ref().and_then(|report| {
        report
            .observation_reports
            .iter()
            .find(|row| {
                row.epoch_index == epoch_index
                    && row.sat == obs.sat
                    && row.ambiguity_id == obs.ambiguity_id
            })
            .cloned()
    });
    SsrBiasExclusion {
        epoch_index,
        satellite_id: obs.satellite_id.clone(),
        ambiguity_id: obs.ambiguity_id.clone(),
        code_bias_missing: shortfall.code_bias_missing,
        phase_bias_missing: shortfall.phase_bias_missing,
        transmit_time_failure: shortfall.transmit_time_failure,
        pass,
        stage,
        application,
    }
}

/// Whether the SSR/HAS bias records `lookup` holds for `obs` in caller epoch
/// `epoch_index` still hold at the transmission time `transmit_time_j2000_s`.
///
/// For every required bias with records, `source` has to apply SSR orbit and clock
/// corrections of the records' solution to the satellite at that time, and each record
/// has to be the one its store holds available there, with the same solution, IOD SSR and
/// reference epoch. A bias without records, as in a lookup filled directly, is not
/// checked.
///
/// The outer `Err` is a solve failure rather than a failure of the records:
/// [`FloatSolveError::Ut1OutsideCoverage`] when the source refuses to say which solution
/// it applies because producing the state reads UT1 outside the UT1 table.
pub(super) fn ssr_bias_records_hold(
    source: &dyn ObservableEphemerisSource,
    obs: &FloatObservation,
    epoch_index: usize,
    lookup: &PppCorrectionLookup,
    transmit_time_j2000_s: Option<f64>,
) -> Result<Result<(), SsrTransmitTimeFailure>, FloatSolveError> {
    match lookup_ut1_refusal(lookup, obs, epoch_index) {
        Some(reason) => Err(FloatSolveError::Ut1OutsideCoverage(reason)),
        None => records_hold(source, obs, epoch_index, lookup, transmit_time_j2000_s),
    }
}

fn records_hold(
    source: &dyn ObservableEphemerisSource,
    obs: &FloatObservation,
    epoch_index: usize,
    lookup: &PppCorrectionLookup,
    transmit_time_j2000_s: Option<f64>,
) -> Result<Result<(), SsrTransmitTimeFailure>, FloatSolveError> {
    let key = (obs.sat, epoch_index, obs.ambiguity_id.clone());
    let code_records = lookup
        .ssr_code_bias_enabled
        .then(|| lookup.ssr_code_bias_records.get(&key))
        .flatten();
    let phase_records = lookup
        .phase_bias_enabled
        .then(|| lookup.phase_bias_records.get(&key))
        .flatten();
    if code_records.is_none() && phase_records.is_none() {
        return Ok(Ok(()));
    }
    let Some(ssr) = source.ssr_corrections() else {
        return Ok(Err(SsrTransmitTimeFailure::SourceWithoutSsrCorrections));
    };
    let Some(t_tx) = transmit_time_j2000_s.filter(|t| t.is_finite()) else {
        return Ok(Err(SsrTransmitTimeFailure::TransmitTimeUnavailable));
    };
    let applied = match ssr.try_applied_orbit_clock_solution(obs.sat, t_tx) {
        Ok(applied) => applied,
        Err(crate::Error::Ut1OutsideCoverage(reason)) => {
            return Err(FloatSolveError::Ut1OutsideCoverage(reason))
        }
        Err(error) => {
            return Ok(Err(SsrTransmitTimeFailure::Source {
                transmit_time_j2000_s: t_tx,
                error,
            }))
        }
    };
    let store = ssr.ssr_store();
    let holds = |record: &SsrBiasRecord,
                 status: SsrBiasStatus,
                 solution: Option<SsrSolution>,
                 iod_ssr: Option<u8>,
                 ref_epoch: Option<f64>| {
        status == SsrBiasStatus::Available
            && solution == Some(record.solution)
            && iod_ssr == Some(record.iod_ssr)
            && ref_epoch == Some(record.ref_epoch_j2000_s)
    };
    for record in code_records.into_iter().chain(phase_records).flatten() {
        if applied != Some(record.solution) {
            return Ok(Err(SsrTransmitTimeFailure::OrbitClockSolution {
                transmit_time_j2000_s: t_tx,
                applied,
            }));
        }
    }
    for record in code_records.into_iter().flatten() {
        let query = store.query_code_bias(obs.sat, record.signal, t_tx);
        if !holds(
            record,
            query.status,
            query.solution,
            query.iod_ssr,
            query.ref_epoch_j2000_s,
        ) {
            return Ok(Err(SsrTransmitTimeFailure::BiasRecord {
                transmit_time_j2000_s: t_tx,
                signal: record.signal,
                status: query.status,
            }));
        }
    }
    for record in phase_records.into_iter().flatten() {
        // Continuity was settled against the caller's token when the bias was resolved;
        // this query only confirms the record.
        let query = store.query_phase_bias(obs.sat, record.signal, t_tx, None);
        if !holds(
            record,
            query.status,
            query.solution,
            query.iod_ssr,
            query.ref_epoch_j2000_s,
        ) {
            return Ok(Err(SsrTransmitTimeFailure::BiasRecord {
                transmit_time_j2000_s: t_tx,
                signal: record.signal,
                status: query.status,
            }));
        }
    }
    Ok(Ok(()))
}

/// Missing correction to report when the recorded biases fail at the transmission time:
/// the phase bias for a record failure on a signal only the phase records hold, otherwise
/// the code bias, which the row model reads first.
fn failed_ssr_bias(
    failure: &SsrTransmitTimeFailure,
    obs: &FloatObservation,
    epoch_index: usize,
    lookup: &PppCorrectionLookup,
) -> MissingCorrection {
    let key = (obs.sat, epoch_index, obs.ambiguity_id.clone());
    let code_has = |signal: Option<u8>| {
        lookup.ssr_code_bias_enabled
            && lookup
                .ssr_code_bias_records
                .get(&key)
                .is_some_and(|records| signal.is_none_or(|s| records.iter().any(|r| r.signal == s)))
    };
    let signal = match failure {
        SsrTransmitTimeFailure::BiasRecord { signal, .. } => Some(*signal),
        _ => None,
    };
    if code_has(signal) {
        MissingCorrection::SsrCodeBias
    } else {
        MissingCorrection::PhaseBias
    }
}

/// How the carrier ambiguity is bound for this assembly: the single degree of
/// freedom separating the float and fixed undifferenced PPP rows.
pub(super) enum AmbiguityBinding<'a> {
    /// Float: each satellite's ambiguity is estimated. The metre value seeds the
    /// phase prefit; the design row carries one column per id in `ids` (the phase
    /// row's own column set to `1.0`).
    Estimated {
        ids: &'a [AmbiguityId],
        values: &'a BTreeMap<String, f64>,
    },
    /// Fixed: each satellite's ambiguity is held at `values`. The held metre value
    /// is folded into the phase prefit; the design row carries no ambiguity column.
    Held { values: &'a BTreeMap<String, f64> },
}

impl<'a> AmbiguityBinding<'a> {
    /// The ambiguity metre values this binding reads from (the float state's
    /// estimates or the held integers). The value lookup is identical for both;
    /// only the design columns differ.
    fn values(&self) -> &'a BTreeMap<String, f64> {
        match self {
            Self::Estimated { values, .. } | Self::Held { values } => values,
        }
    }

    /// Number of estimated ambiguity columns (zero when the integers are held).
    fn ambiguity_columns(&self) -> usize {
        match self {
            Self::Estimated { ids, .. } => ids.len(),
            Self::Held { .. } => 0,
        }
    }

    /// Number of residual ionosphere columns. Unlike ambiguity columns, these
    /// are still estimated when carrier ambiguities are held fixed.
    fn residual_ionosphere_columns(&self) -> usize {
        match self {
            Self::Estimated { ids, .. } => ids.len(),
            Self::Held { values } => values.len(),
        }
    }

    /// The design column index of `obs`'s own ambiguity on the phase row, or
    /// `None` for the code row / a held-integer solve.
    fn active_column(&self, obs: &FloatObservation) -> Option<usize> {
        match self {
            Self::Estimated { ids, .. } => {
                ids.iter().position(|id| id.as_str() == obs.ambiguity_id)
            }
            Self::Held { .. } => None,
        }
    }

    fn active_residual_ionosphere_column(&self, obs: &FloatObservation) -> Option<usize> {
        match self {
            Self::Estimated { ids, .. } => {
                ids.iter().position(|id| id.as_str() == obs.ambiguity_id)
            }
            Self::Held { values } => values.keys().position(|id| id == &obs.ambiguity_id),
        }
    }
}

/// Whether the rows add the satellite clock relativity term `2 r·v / c`.
///
/// Only when it is enabled and the satellite clock in use lacks it. With an external CLK
/// series the rows use the series' clock, which lacks it whatever the source. Without
/// one they use the source's clock, which already carries it when the source says so
/// ([`ObservableEphemerisSource::clock_includes_relativity`]): adding it again would
/// count it twice. RTKLIB `satposs` likewise returns one clock per ephemeris option,
/// with the relativistic term in it, and `ppp.c` adds none.
pub(super) fn adds_sat_clock_relativity(
    source: &dyn ObservableEphemerisSource,
    corrections: &RangeCorrections,
) -> bool {
    corrections.sat_clock_relativity
        && (corrections.satellite_clock.is_some() || !source.clock_includes_relativity())
}

fn residual_ionosphere_m(state: &FloatState, obs: &FloatObservation, enabled: bool) -> f64 {
    if enabled {
        state
            .residual_ionosphere_m
            .get(&obs.ambiguity_id)
            .copied()
            .unwrap_or(0.0)
    } else {
        0.0
    }
}

/// The bound ambiguity metres for `obs`, or the missing-ambiguity error.
fn bound_ambiguity(
    values: &BTreeMap<String, f64>,
    obs: &FloatObservation,
) -> Result<f64, PppRowError> {
    values
        .get(&obs.ambiguity_id)
        .copied()
        .ok_or_else(|| PppRowError::MissingAmbiguity(obs.ambiguity_id.clone()))
}

/// Row/residual assembly error, neutral between the float and fixed callers so
/// each maps it onto its own error surface ([`into_float`](Self::into_float) /
/// [`into_fixed`](Self::into_fixed)).
#[derive(Debug)]
pub(super) enum PppRowError {
    Model(FloatSolveError),
    MissingAmbiguity(String),
    /// SSR/HAS bias records the lookup holds for an observation do not hold at the
    /// transmission time of the state the rows were built at. A solve that iterates to a
    /// fixed point turns this into an exclusion; elsewhere it is the missing correction.
    SsrBiasFlip {
        exclusion: Box<SsrBiasExclusion>,
        missing: MissingCorrection,
    },
}

fn row_invalid(error: FieldError) -> PppRowError {
    PppRowError::Model(invalid_input(error))
}

impl PppRowError {
    /// Map onto the float solve's error surface.
    pub(super) fn into_float(self) -> FloatSolveError {
        match self {
            Self::Model(error) => error,
            Self::MissingAmbiguity(id) => FloatSolveError::MissingAmbiguity(id),
            Self::SsrBiasFlip { exclusion, missing } => FloatSolveError::MissingCorrection {
                satellite_id: exclusion.satellite_id,
                correction: missing,
            },
        }
    }

    /// Map onto the fixed solve's error surface (a missing held integer is the
    /// fixed-specific variant; an ephemeris gap wraps the float error).
    pub(super) fn into_fixed(self) -> FixedSolveError {
        match self {
            Self::Model(error) => FixedSolveError::Float(error),
            Self::MissingAmbiguity(id) => FixedSolveError::MissingFixedAmbiguity(id),
            flip @ Self::SsrBiasFlip { .. } => FixedSolveError::Float(flip.into_float()),
        }
    }
}

/// The undifferenced code/phase model for one observation: the prefit residuals,
/// the measurement weights, and the geometry the design row needs. Evaluated once
/// and consumed by both the design-row and residual-row builders.
struct UndiffModel {
    code_prefit: f64,
    phase_prefit: f64,
    code_weight: f64,
    phase_weight: f64,
    los_base: [f64; 3],
    ztd_mapping: f64,
    tropo_gradient_mapping: [f64; 2],
    residual_ionosphere_m: f64,
}

fn undifferenced_model(
    ctx: ModelContext,
    epoch: &FloatEpoch,
    epoch_idx: usize,
    obs: &FloatObservation,
    state: &FloatState,
    ambiguity_m: f64,
) -> Result<UndiffModel, PppRowError> {
    let pred = observation_geometry(ctx.source, obs, state.position_m, epoch.t_rx_j2000_s)
        .map_err(PppRowError::Model)?;
    validate_transmit_geometry(&pred)?;
    // Only the satellite clock relativity term uses the satellite velocity.
    let sat_velocity_m_s = if adds_sat_clock_relativity(ctx.source, ctx.corrections) {
        let options = predict_default(ctx.source, obs).map_err(PppRowError::Model)?;
        let velocity = transmit_velocity_m_s(ctx.source, obs.sat, &pred, options.sagnac)
            .map_err(|e| PppRowError::Model(no_ephemeris(obs, e)))?;
        validate::finite_vec3(velocity, "ppp predicted sat_velocity_m_s").map_err(row_invalid)?;
        Some(velocity)
    } else {
        None
    };
    let clock_m = state.clocks_m.get(epoch_idx).copied().ok_or_else(|| {
        PppRowError::Model(invalid_clock_count(epoch_idx + 1, state.clocks_m.len()))
    })?;
    validate::finite(clock_m, "ppp row receiver clock_m").map_err(row_invalid)?;
    let tropo_model =
        model_troposphere(&pred, state.position_m, epoch, ctx.tropo).map_err(PppRowError::Model)?;
    validate_tropo_model(&tropo_model)?;
    let sat_clock_m = satellite_clock_m(&pred, obs, ctx.corrections.satellite_clock.as_ref())
        .map_err(PppRowError::Model)?;
    validate::finite(sat_clock_m, "ppp row satellite clock_m").map_err(row_invalid)?;
    // The receiver clock is indexed by the solved slice; the corrections by the caller's
    // epochs.
    let correction_idx = ctx.correction_epoch_index(epoch_idx);
    let corrections_m = range_corrections_m(
        CorrectedObservation {
            pred: &pred,
            sat_velocity_m_s,
            rx_pos: state.position_m,
            epoch_idx: correction_idx,
            obs,
        },
        &tropo_model,
        state,
        ctx.corrections,
    )
    .map_err(PppRowError::Model)?;
    validate::finite(corrections_m, "ppp row corrections_m").map_err(row_invalid)?;
    let ssr_code_bias_m =
        ssr_code_bias_m(obs, correction_idx, &ctx.corrections.ppp).map_err(PppRowError::Model)?;
    validate::finite(ssr_code_bias_m, "ppp row ssr_code_bias_m").map_err(row_invalid)?;
    let phase_windup_m =
        phase_windup_m(obs, correction_idx, ctx.corrections).map_err(PppRowError::Model)?;
    validate::finite(phase_windup_m, "ppp row phase_windup_m").map_err(row_invalid)?;
    let phase_bias_m =
        phase_bias_m(obs, correction_idx, ctx.corrections).map_err(PppRowError::Model)?;
    validate::finite(phase_bias_m, "ppp row phase_bias_m").map_err(row_invalid)?;
    // A recorded SSR bias that does not hold at this iteration's transmission time is
    // missing here rather than applied to other clocks. An observation admitted again
    // after an exclusion is judged only at a converged position, by the solve.
    let deferred = ctx
        .ssr_bias_deferred
        .iter()
        .any(|(epoch_index, ambiguity_id)| {
            *epoch_index == correction_idx && *ambiguity_id == obs.ambiguity_id
        });
    let records_hold = if deferred {
        Ok(())
    } else {
        ssr_bias_records_hold(
            ctx.source,
            obs,
            correction_idx,
            &ctx.corrections.ppp,
            Some(pred.transmit_time_j2000_s),
        )
        .map_err(PppRowError::Model)?
    };
    if let Err(failure) = records_hold {
        let missing = failed_ssr_bias(&failure, obs, correction_idx, &ctx.corrections.ppp);
        return Err(PppRowError::SsrBiasFlip {
            exclusion: Box::new(ssr_bias_exclusion(
                &ctx.corrections.ppp,
                obs,
                correction_idx,
                SsrBiasShortfall {
                    code_bias_missing: false,
                    phase_bias_missing: false,
                    transmit_time_failure: Some(failure),
                },
                ctx.ssr_bias_pass,
                ctx.ssr_bias_stage,
            )),
            missing,
        });
    }
    let residual_ionosphere_m = residual_ionosphere_m(state, obs, ctx.estimate_residual_ionosphere);
    validate::finite(residual_ionosphere_m, "ppp row residual_ionosphere_m")
        .map_err(row_invalid)?;
    let model_range = pred.geometric_range_m + clock_m - sat_clock_m + corrections_m;
    let model_code = model_range + ssr_code_bias_m + residual_ionosphere_m;
    validate::finite(model_code, "ppp row model_code_m").map_err(row_invalid)?;
    validate::finite(model_range, "ppp row model_range_m").map_err(row_invalid)?;
    let model = UndiffModel {
        code_prefit: obs.code_m - model_code,
        phase_prefit: obs.phase_m + phase_bias_m
            - phase_windup_m
            - (model_range + ambiguity_m - residual_ionosphere_m),
        code_weight: measurement_weight(ctx.weights, true, pred.elevation_deg),
        phase_weight: measurement_weight(ctx.weights, false, pred.elevation_deg),
        los_base: [-pred.los_unit[0], -pred.los_unit[1], -pred.los_unit[2]],
        ztd_mapping: tropo_model.ztd_mapping,
        tropo_gradient_mapping: tropo_model.gradient_mapping,
        residual_ionosphere_m,
    };
    validate_undifferenced_model(&model)?;
    Ok(model)
}

fn validate_transmit_geometry(pred: &TransmitGeometry) -> Result<(), PppRowError> {
    validate::finite(pred.geometric_range_m, "ppp predicted geometric_range_m")
        .map_err(row_invalid)?;
    if let Some(sat_clock_s) = pred.sat_clock_s {
        validate::finite(sat_clock_s, "ppp predicted sat_clock_s").map_err(row_invalid)?;
    }
    validate::finite(pred.elevation_deg, "ppp predicted elevation_deg").map_err(row_invalid)?;
    validate::finite(pred.azimuth_deg, "ppp predicted azimuth_deg").map_err(row_invalid)?;
    validate::finite(
        pred.transmit_time_j2000_s,
        "ppp predicted transmit_time_j2000_s",
    )
    .map_err(row_invalid)?;
    validate::finite_vec3(pred.los_unit, "ppp predicted los_unit").map_err(row_invalid)?;
    validate::finite_vec3(pred.sat_pos_ecef_m, "ppp predicted sat_pos_ecef_m")
        .map_err(row_invalid)?;
    Ok(())
}

fn validate_tropo_model(tropo_model: &super::model::TropoModelState) -> Result<(), PppRowError> {
    validate::finite(tropo_model.ztd_mapping, "ppp row ztd_mapping").map_err(row_invalid)?;
    Ok(())
}

fn validate_undifferenced_model(model: &UndiffModel) -> Result<(), PppRowError> {
    validate::finite(model.code_prefit, "ppp row code_prefit_m").map_err(row_invalid)?;
    validate::finite(model.phase_prefit, "ppp row phase_prefit_m").map_err(row_invalid)?;
    validate::finite_positive(model.code_weight, "ppp row code_weight").map_err(row_invalid)?;
    validate::finite_positive(model.phase_weight, "ppp row phase_weight").map_err(row_invalid)?;
    validate::finite_vec3(model.los_base, "ppp row los_base").map_err(row_invalid)?;
    validate::finite(model.ztd_mapping, "ppp row ztd_mapping").map_err(row_invalid)?;
    validate::finite_slice(
        &model.tropo_gradient_mapping,
        "ppp row tropo_gradient_mapping",
    )
    .map_err(row_invalid)?;
    validate::finite(model.residual_ionosphere_m, "ppp row residual_ionosphere_m")
        .map_err(row_invalid)?;
    Ok(())
}

/// Assemble the weighted code/phase design rows for the whole arc. The float and
/// fixed solves differ only in `binding`: float carries an estimated-ambiguity
/// column per satellite, fixed holds the integers and carries none.
pub(super) fn build_rows(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    binding: &AmbiguityBinding,
    state: &FloatState,
) -> Result<Vec<Row>, PppRowError> {
    validate_state_clock_count(state, epochs.len()).map_err(PppRowError::Model)?;
    let n_ambiguities = binding.ambiguity_columns();
    let mut rows = Vec::new();
    for (epoch_idx, epoch) in epochs.iter().enumerate() {
        for obs in &epoch.observations {
            let ambiguity_m = bound_ambiguity(binding.values(), obs)?;
            let model = undifferenced_model(ctx, epoch, epoch_idx, obs, state, ambiguity_m)?;
            let ztd_mapping = estimates_ztd(ctx.tropo).then_some(model.ztd_mapping);
            let tropo_gradient_mapping =
                estimates_tropo_gradients(ctx.tropo).then_some(model.tropo_gradient_mapping);
            let n_residual_ionosphere = if ctx.estimate_residual_ionosphere {
                binding.residual_ionosphere_columns()
            } else {
                0
            };
            let active_ionosphere = ctx
                .estimate_residual_ionosphere
                .then(|| {
                    binding
                        .active_residual_ionosphere_column(obs)
                        .map(|idx| (idx, 1.0))
                })
                .flatten();
            rows.push(Row {
                h: undifferenced_design_row(
                    model.los_base,
                    epoch_idx,
                    epochs.len(),
                    UndifferencedDesignOptions {
                        ztd_mapping,
                        tropo_gradient_mapping,
                        residual_ionosphere_columns: n_residual_ionosphere,
                        active_residual_ionosphere: active_ionosphere,
                        ambiguity_columns: n_ambiguities,
                        active_ambiguity: None,
                    },
                ),
                y: model.code_prefit,
                weight: model.code_weight,
            });
            rows.push(Row {
                h: undifferenced_design_row(
                    model.los_base,
                    epoch_idx,
                    epochs.len(),
                    UndifferencedDesignOptions {
                        ztd_mapping,
                        tropo_gradient_mapping,
                        residual_ionosphere_columns: n_residual_ionosphere,
                        active_residual_ionosphere: active_ionosphere.map(|(idx, _)| (idx, -1.0)),
                        ambiguity_columns: n_ambiguities,
                        active_ambiguity: binding.active_column(obs),
                    },
                ),
                y: model.phase_prefit,
                weight: model.phase_weight,
            });
        }
    }
    Ok(rows)
}

/// Assemble the post-fit residual rows for the whole arc. Identical undifferenced
/// model to [`build_rows`]; the residual carries the prefit code/phase values and
/// their weights without a design vector, so it needs only the ambiguity `values`
/// (the column layout is a design-row concern that does not reach the residuals).
pub(super) fn residual_rows(
    ctx: ModelContext,
    epochs: &[FloatEpoch],
    values: &BTreeMap<String, f64>,
    state: &FloatState,
) -> Result<Vec<FloatResidual>, PppRowError> {
    validate_state_clock_count(state, epochs.len()).map_err(PppRowError::Model)?;
    let mut rows = Vec::new();
    for (epoch_idx, epoch) in epochs.iter().enumerate() {
        for obs in &epoch.observations {
            let ambiguity_m = bound_ambiguity(values, obs)?;
            let model = undifferenced_model(ctx, epoch, epoch_idx, obs, state, ambiguity_m)?;
            rows.push(FloatResidual {
                epoch_index: ctx.correction_epoch_index(epoch_idx),
                satellite_id: obs.satellite_id.clone(),
                ambiguity_id: obs.ambiguity_id.clone(),
                code_m: model.code_prefit,
                phase_m: model.phase_prefit,
                code_weight: model.code_weight,
                phase_weight: model.phase_weight,
            });
        }
    }
    Ok(rows)
}
