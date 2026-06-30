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
use crate::estimation::substrate::parameters::undifferenced_design_row;
use crate::observables::{predict, PredictedObservables};
use crate::validate::{self, FieldError};

use super::model::{
    measurement_weight, model_troposphere, phase_windup_m, range_corrections_m, satellite_clock_m,
};
use super::normal::Row;
use super::{
    estimates_ztd, invalid_clock_count, invalid_input, no_ephemeris, predict_default,
    validate_state_clock_count, FixedSolveError, FloatEpoch, FloatObservation, FloatResidual,
    FloatSolveError, FloatState, ModelContext,
};

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
        }
    }

    /// Map onto the fixed solve's error surface (a missing held integer is the
    /// fixed-specific variant; an ephemeris gap wraps the float error).
    pub(super) fn into_fixed(self) -> FixedSolveError {
        match self {
            Self::Model(error) => FixedSolveError::Float(error),
            Self::MissingAmbiguity(id) => FixedSolveError::MissingFixedAmbiguity(id),
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
}

fn undifferenced_model(
    ctx: ModelContext,
    epoch: &FloatEpoch,
    epoch_idx: usize,
    obs: &FloatObservation,
    state: &FloatState,
    ambiguity_m: f64,
) -> Result<UndiffModel, PppRowError> {
    let options = predict_default(ctx.source, obs).map_err(PppRowError::Model)?;
    let pred = predict(
        ctx.source,
        obs.sat,
        state.position_m,
        epoch.t_rx_j2000_s,
        options,
    )
    .map_err(|e| PppRowError::Model(no_ephemeris(obs, e)))?;
    validate_predicted_observables(&pred)?;
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
    let corrections_m = range_corrections_m(
        &pred,
        state.position_m,
        epoch_idx,
        obs,
        &tropo_model,
        state,
        ctx.corrections,
    )
    .map_err(PppRowError::Model)?;
    validate::finite(corrections_m, "ppp row corrections_m").map_err(row_invalid)?;
    let phase_windup_m =
        phase_windup_m(obs, epoch_idx, ctx.corrections).map_err(PppRowError::Model)?;
    validate::finite(phase_windup_m, "ppp row phase_windup_m").map_err(row_invalid)?;
    let model_code = pred.geometric_range_m + clock_m - sat_clock_m + corrections_m;
    validate::finite(model_code, "ppp row model_code_m").map_err(row_invalid)?;
    let model = UndiffModel {
        code_prefit: obs.code_m - model_code,
        phase_prefit: obs.phase_m - phase_windup_m - (model_code + ambiguity_m),
        code_weight: measurement_weight(ctx.weights, true, pred.elevation_deg),
        phase_weight: measurement_weight(ctx.weights, false, pred.elevation_deg),
        los_base: [-pred.los_unit[0], -pred.los_unit[1], -pred.los_unit[2]],
        ztd_mapping: tropo_model.ztd_mapping,
    };
    validate_undifferenced_model(&model)?;
    Ok(model)
}

fn validate_predicted_observables(pred: &PredictedObservables) -> Result<(), PppRowError> {
    validate::finite(pred.geometric_range_m, "ppp predicted geometric_range_m")
        .map_err(row_invalid)?;
    validate::finite(pred.range_rate_m_s, "ppp predicted range_rate_m_s").map_err(row_invalid)?;
    validate::finite(pred.doppler_hz, "ppp predicted doppler_hz").map_err(row_invalid)?;
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
    validate::finite_vec3(pred.sat_velocity_m_s, "ppp predicted sat_velocity_m_s")
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
            rows.push(Row {
                h: undifferenced_design_row(
                    model.los_base,
                    epoch_idx,
                    epochs.len(),
                    ztd_mapping,
                    n_ambiguities,
                    None,
                ),
                y: model.code_prefit,
                weight: model.code_weight,
            });
            rows.push(Row {
                h: undifferenced_design_row(
                    model.los_base,
                    epoch_idx,
                    epochs.len(),
                    ztd_mapping,
                    n_ambiguities,
                    binding.active_column(obs),
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
                epoch_index: epoch_idx,
                satellite_id: obs.satellite_id.clone(),
                code_m: model.code_prefit,
                phase_m: model.phase_prefit,
                code_weight: model.code_weight,
                phase_weight: model.phase_weight,
            });
        }
    }
    Ok(rows)
}
