//! Double-difference row buffers shared by the RTK baseline and sequential
//! filter solvers.
//!
//! `DdRowScratch`/`RefCtxScratch`/`EpochRowsScratch` are the reusable per-epoch
//! staging buffers that the float, fixed, and filter row builders fill in place;
//! `DdRow` is the owned, test-only row consumed by the reference fold wrappers.
//! These are pure data containers plus the string-assignment helpers used to
//! populate them without reallocating.
//!
//! [`dd_epoch_rows_into`] is the single double-difference row builder the three
//! RTK paths share. The static float baseline, the static fixed baseline (with
//! held integers), and the sequential information filter all linearize the same
//! per-system double-difference geometry; they differed only in how the carrier
//! ambiguity column is parameterized and how a row is weighted. Those two
//! degrees of freedom are named in [`DdRowRecipe`], so the geometry is written
//! once and every reference golden stays bit-exact.

use std::collections::BTreeMap;

use crate::astro::math::vec3::{add3, sub3};

use super::antenna::{
    double_difference_receiver_antenna_correction, DoubleDifferenceAntennaGeometry,
    ReceiverAntennaError, ReceiverAntennaScratch,
};
use super::model::{
    geometric_range_m, range_derivative, satellite_system, single_difference_variance, system_of,
    Epoch, RowKind, SatMeas,
};
use super::{MeasContext, RtkInputErrorKind};
use crate::ambiguity::AmbiguityId;
use crate::estimation::substrate::parameters::ParameterLayout;
use crate::id::GnssSystem;
use crate::validate;

/// A double-difference row: design vector `h` (length `state.dim()`), prefit
/// residual `y`, the satellite and reference single-difference variances, and
/// the diagonal information weight `1/(sd_variance + ref_sd_variance)` retained
/// for diagnostics/back-compat tests. Measurement updates use the full
/// block covariance in the epoch fold path. Test-only owned row; production
/// folds through the `DdRowScratch` index variants.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct DdRow {
    pub kind: RowKind,
    pub sat: String,
    /// Reference satellite this row differenced against (its own system's
    /// reference). The covariance blocking key - rows correlate only through a
    /// shared reference single difference.
    pub ref_sat: String,
    pub sd_ambiguity_id: String,
    pub h: Vec<f64>,
    pub y: f64,
    pub sd_variance_m2: f64,
    pub ref_sd_variance_m2: f64,
    pub weight: f64,
}

#[cfg(test)]
impl DdRow {
    /// Owned snapshot of a production scratch row. The `epoch_index` and typed
    /// `ambiguity_id` scratch fields are filter bookkeeping, not part of the
    /// measurement row, so they are dropped here. Used by the test-only owned
    /// row wrappers (`update::epoch_dd_rows`, `float::float_epoch_rows`,
    /// `fixed::fixed_epoch_rows`) that drive the row-level golden traces.
    pub(super) fn from_scratch(r: &DdRowScratch) -> Self {
        Self {
            kind: r.kind,
            sat: r.sat.clone(),
            ref_sat: r.ref_sat.clone(),
            sd_ambiguity_id: r.sd_ambiguity_id.clone(),
            h: r.h.clone(),
            y: r.y,
            sd_variance_m2: r.sd_variance_m2,
            ref_sd_variance_m2: r.ref_sd_variance_m2,
            weight: r.weight,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(super) struct DdRowScratch {
    pub(super) epoch_index: usize,
    pub(super) kind: RowKind,
    pub(super) sat: String,
    pub(super) ref_sat: String,
    /// Composed double-difference ambiguity id for this row (`sat_sd|ref=ref_sd`,
    /// or the bare satellite token for the trivial single-system case). Reused in
    /// place per solve via [`assign_double_difference_ambiguity_id`].
    pub(super) ambiguity_id: AmbiguityId,
    pub(super) sd_ambiguity_id: String,
    pub(super) h: Vec<f64>,
    pub(super) y: f64,
    pub(super) sd_variance_m2: f64,
    pub(super) ref_sd_variance_m2: f64,
    pub(super) weight: f64,
}

#[derive(Debug, Default)]
pub(super) struct RefCtxScratch {
    /// Constellation of this reference, the per-system double-difference
    /// grouping key. `None` marks an unused pooled slot (`0..refs_len` slots are
    /// always populated). A typed [`GnssSystem`] is `Copy`, so unlike the other
    /// fields it needs no buffer-reusing string assignment.
    pub(super) system: Option<GnssSystem>,
    pub(super) sat: String,
    pub(super) sd_ambiguity_id: String,
    pub(super) pos: [f64; 3],
    pub(super) sd_code: f64,
    pub(super) sd_phase: f64,
    pub(super) sd_geom: f64,
    pub(super) los: [f64; 3],
    pub(super) code_var: f64,
    pub(super) phase_var: f64,
}

#[derive(Debug, Default)]
pub(super) struct EpochRowsScratch {
    pub(super) refs: Vec<RefCtxScratch>,
    pub(super) refs_len: usize,
    pub(super) rows: Vec<DdRowScratch>,
    pub(super) rows_len: usize,
    pub(super) receiver_antenna: ReceiverAntennaScratch,
}

pub(super) fn assign_str(dst: &mut String, src: &str) {
    dst.clear();
    dst.push_str(src);
}

pub(super) fn assign_double_difference_ambiguity_id(
    dst: &mut AmbiguityId,
    sat: &str,
    sat_sd_id: &str,
    ref_sat: &str,
    ref_sd_id: &str,
) {
    if sat_sd_id == sat && ref_sd_id == ref_sat {
        dst.assign(sat);
    } else {
        dst.clear();
        dst.push_str(sat_sd_id);
        dst.push_str("|ref=");
        dst.push_str(ref_sd_id);
    }
}

/// The two ways the three RTK double-difference paths differ, named so the shared
/// builder reproduces each path's exact operation order (see [`dd_epoch_rows_into`]).
///
/// Every variant shares the per-system double-difference geometry; they part only
/// at the carrier phase ambiguity column (how the design row carries the integer
/// state) and at the row weight (inverse double-difference variance for the
/// information filter, inverse sigma for the static least-squares baselines).
#[derive(Clone, Copy)]
pub(super) enum DdRowRecipe<'a> {
    /// Sequential information filter (`update`). The estimated state holds the
    /// per-satellite single-difference ambiguities, so a phase row carries `+1`
    /// at the satellite's SD column and `-1` at the reference's SD column; the
    /// prefit subtracts `SD(sat) - SD(ref)`. Rows weight by inverse
    /// double-difference variance (`1/(σ²_sat + σ²_ref)`); the correlated block
    /// covariance is applied later in the fold.
    SequentialFilter {
        /// Single-difference ambiguity ids in information-matrix column order.
        sd_ambiguity_ids: &'a [String],
        /// Current single-difference ambiguity iterate (metres), column-parallel
        /// to `sd_ambiguity_ids`.
        sd_ambiguities_m: &'a [f64],
    },
    /// Static float baseline (`float`). The estimated state holds the
    /// double-difference ambiguities directly, so a phase row carries `+1` at the
    /// single composed DD column and the prefit subtracts the float DD ambiguity.
    /// Rows weight by inverse sigma (`1/√(σ²_sat + σ²_ref)`).
    FloatBaseline {
        /// Composed double-difference ambiguity ids in column order.
        ambiguity_ids: &'a [String],
        /// Float double-difference ambiguities (metres), column-parallel to
        /// `ambiguity_ids`.
        ambiguities_m: &'a [f64],
    },
    /// Static fixed baseline (`fixed`). Held integers leave their column out of
    /// the design (the fixed value is folded into the prefit); still-free
    /// ambiguities carry `+1` at their free-column index. Rows weight by inverse
    /// sigma, like the float baseline.
    FixedBaseline {
        /// Still-estimated (free) double-difference ambiguity ids in column order.
        free_ids: &'a [AmbiguityId],
        /// Held double-difference ambiguities (id -> metres).
        fixed_m: &'a BTreeMap<AmbiguityId, f64>,
        /// Free double-difference ambiguities (metres), column-parallel to
        /// `free_ids`.
        ambiguities_m: &'a [f64],
    },
}

/// A double-difference row could not be built. Each RTK path maps this into its
/// own error surface (the filter to `None`/`SingularGeometry`, the static
/// baselines to their `FloatSolveError`/`FixedSolveError` variants).
#[derive(Debug, Clone, PartialEq)]
pub(super) enum DdRowError {
    /// No reference satellite present for a non-reference satellite's
    /// constellation; the payload is the constellation token.
    MissingReference(String),
    /// The satellite's ambiguity has no column in the estimated state; the
    /// payload is the composed double-difference ambiguity id.
    MissingAmbiguity(String),
    /// A provided receiver-antenna calibration could not be applied.
    ReceiverAntenna(ReceiverAntennaError),
    /// A row-builder boundary input was malformed, non-finite, or outside its
    /// physical domain.
    InvalidInput {
        field: &'static str,
        kind: RtkInputErrorKind,
    },
}

fn row_input_error(error: validate::FieldError) -> DdRowError {
    DdRowError::InvalidInput {
        field: error.field(),
        kind: RtkInputErrorKind::from(&error),
    }
}

fn validate_row_boundary(
    ctx: MeasContext<'_>,
    epoch: &Epoch,
    baseline_m: [f64; 3],
) -> Result<(), DdRowError> {
    validate::finite_vec3(ctx.base, "rtk.base_pos").map_err(row_input_error)?;
    validate::finite_vec3(baseline_m, "rtk.baseline_m").map_err(row_input_error)?;
    validate::finite_positive(ctx.model.code_sigma_m, "rtk.code_sigma_m")
        .map_err(row_input_error)?;
    validate::finite_positive(ctx.model.phase_sigma_m, "rtk.phase_sigma_m")
        .map_err(row_input_error)?;
    for meas in epoch.references.iter().chain(&epoch.nonref) {
        validate_sat_meas(meas)?;
    }
    Ok(())
}

fn validate_sat_meas(meas: &SatMeas) -> Result<(), DdRowError> {
    validate::finite(meas.base_code_m, "rtk.base_code_m").map_err(row_input_error)?;
    validate::finite(meas.base_phase_m, "rtk.base_phase_m").map_err(row_input_error)?;
    validate::finite(meas.rover_code_m, "rtk.rover_code_m").map_err(row_input_error)?;
    validate::finite(meas.rover_phase_m, "rtk.rover_phase_m").map_err(row_input_error)?;
    validate::finite_vec3(meas.base_tx_pos, "rtk.base_tx_pos").map_err(row_input_error)?;
    validate::finite_vec3(meas.rover_tx_pos, "rtk.rover_tx_pos").map_err(row_input_error)?;
    validate::finite_vec3(meas.pos, "rtk.pos").map_err(row_input_error)?;
    Ok(())
}

fn validate_variance(value: f64, field: &'static str) -> Result<f64, DdRowError> {
    validate::finite_positive(value, field).map_err(row_input_error)
}

fn validate_built_row(row: &DdRowScratch) -> Result<(), DdRowError> {
    validate::finite_slice(&row.h, "rtk.design_row").map_err(row_input_error)?;
    validate::finite(row.y, "rtk.prefit_residual_m").map_err(row_input_error)?;
    validate::finite_positive(row.weight, "rtk.row_weight").map_err(row_input_error)?;
    Ok(())
}

impl DdRowRecipe<'_> {
    /// Parameter dimension `n = 3 + ambiguity columns` for the design rows.
    fn dim(&self) -> usize {
        let ambiguities = match self {
            DdRowRecipe::SequentialFilter {
                sd_ambiguity_ids, ..
            } => sd_ambiguity_ids.len(),
            DdRowRecipe::FloatBaseline { ambiguity_ids, .. } => ambiguity_ids.len(),
            DdRowRecipe::FixedBaseline { free_ids, .. } => free_ids.len(),
        };
        ParameterLayout::rtk(ambiguities).dim()
    }

    /// Diagonal row weight from the satellite and reference single-difference
    /// variances: inverse double-difference variance for the information filter,
    /// inverse sigma for the static least-squares baselines.
    #[inline]
    fn weight(&self, sat_variance: f64, ref_variance: f64) -> f64 {
        match self {
            DdRowRecipe::SequentialFilter { .. } => 1.0 / (sat_variance + ref_variance),
            DdRowRecipe::FloatBaseline { .. } | DdRowRecipe::FixedBaseline { .. } => {
                1.0 / (sat_variance + ref_variance).sqrt()
            }
        }
    }

    /// Populate the phase row's ambiguity column(s) in `h` and return the prefit
    /// ambiguity value subtracted from the phase double difference. `ambiguity_id`
    /// is the composed DD id; `sat_sd_id`/`ref_sd_id` are the satellite and
    /// reference single-difference ids the sequential filter indexes by.
    fn phase_ambiguity(
        &self,
        ambiguity_id: &AmbiguityId,
        sat_sd_id: &str,
        ref_sd_id: &str,
        h: &mut [f64],
    ) -> Result<f64, DdRowError> {
        match self {
            DdRowRecipe::SequentialFilter {
                sd_ambiguity_ids,
                sd_ambiguities_m,
            } => {
                let sat_pos = sd_ambiguity_ids
                    .iter()
                    .position(|x| x == sat_sd_id)
                    .ok_or_else(|| {
                        DdRowError::MissingAmbiguity(ambiguity_id.as_str().to_string())
                    })?;
                let ref_pos = sd_ambiguity_ids
                    .iter()
                    .position(|x| x == ref_sd_id)
                    .ok_or_else(|| {
                        DdRowError::MissingAmbiguity(ambiguity_id.as_str().to_string())
                    })?;
                h[3 + sat_pos] = 1.0;
                h[3 + ref_pos] = -1.0;
                Ok(sd_ambiguities_m[sat_pos] - sd_ambiguities_m[ref_pos])
            }
            DdRowRecipe::FloatBaseline {
                ambiguity_ids,
                ambiguities_m,
            } => {
                let pos = ambiguity_ids
                    .iter()
                    .position(|id| id.as_str() == ambiguity_id.as_str())
                    .ok_or_else(|| {
                        DdRowError::MissingAmbiguity(ambiguity_id.as_str().to_string())
                    })?;
                h[3 + pos] = 1.0;
                Ok(ambiguities_m[pos])
            }
            DdRowRecipe::FixedBaseline {
                free_ids,
                fixed_m,
                ambiguities_m,
            } => match fixed_m.get(ambiguity_id) {
                Some(&fixed) => Ok(fixed),
                None => {
                    let pos = free_ids
                        .iter()
                        .position(|id| id == ambiguity_id)
                        .ok_or_else(|| {
                            DdRowError::MissingAmbiguity(ambiguity_id.as_str().to_string())
                        })?;
                    h[3 + pos] = 1.0;
                    Ok(ambiguities_m[pos])
                }
            },
        }
    }
}

/// Build the double-difference code+phase rows for one epoch, linearized at
/// `baseline_m` (rover = `base + baseline_m`), filling `scratch` in place and
/// returning the populated prefix. Each non-reference satellite contributes a
/// code row then a phase row (Elixir order). The per-system reference geometry,
/// the single-difference variances, the geometric double difference (with the
/// receiver-antenna correction), and the line-of-sight design columns are
/// identical across the three RTK paths; `recipe` selects the carrier ambiguity
/// column parameterization and the row weight so each path stays bit-exact.
pub(super) fn dd_epoch_rows_into<'a>(
    ctx: MeasContext,
    epoch: &Epoch,
    epoch_index: usize,
    baseline_m: [f64; 3],
    recipe: DdRowRecipe<'_>,
    scratch: &'a mut EpochRowsScratch,
) -> Result<&'a [DdRowScratch], DdRowError> {
    validate_row_boundary(ctx, epoch, baseline_m)?;

    let MeasContext {
        base,
        model,
        antenna: receiver_antenna_corrections,
    } = ctx;
    let rover = add3(base, baseline_m);
    let n = recipe.dim();

    // Per-system reference context, computed once per epoch (Elixir
    // `epoch_reference_data/2`): the reference's observation single differences,
    // SD geometry + LOS at the current linearization point, and SD variances from
    // the shared receive-time position (Elixir `row_variance/5`: satellite and
    // reference evaluated separately).
    scratch.refs_len = 0;
    for r in &epoch.references {
        let Some(system) = system_of(&r.sat) else {
            continue;
        };
        let existing = (0..scratch.refs_len).find(|&i| scratch.refs[i].system == Some(system));
        let idx = if let Some(i) = existing {
            i
        } else {
            if scratch.refs_len == scratch.refs.len() {
                scratch.refs.push(RefCtxScratch::default());
            }
            let i = scratch.refs_len;
            scratch.refs_len += 1;
            i
        };
        let rc = &mut scratch.refs[idx];
        rc.system = Some(system);
        assign_str(&mut rc.sat, &r.sat);
        assign_str(&mut rc.sd_ambiguity_id, &r.sd_ambiguity_id);
        rc.pos = r.pos;
        rc.sd_code = r.rover_code_m - r.base_code_m;
        rc.sd_phase = r.rover_phase_m - r.base_phase_m;
        rc.sd_geom = geometric_range_m(r.rover_tx_pos, rover, model.sagnac)
            - geometric_range_m(r.base_tx_pos, base, model.sagnac);
        rc.los = range_derivative(rover, r.rover_tx_pos);
        rc.code_var = validate_variance(
            single_difference_variance(model.code_sigma_m, model.stochastic, base, r.pos),
            "rtk.code_variance_m2",
        )?;
        rc.phase_var = validate_variance(
            single_difference_variance(model.phase_sigma_m, model.stochastic, base, r.pos),
            "rtk.phase_variance_m2",
        )?;
    }

    scratch.rows_len = 0;
    for m in &epoch.nonref {
        // Each non-reference satellite pairs with its OWN system's reference.
        let rc_idx = (0..scratch.refs_len)
            .find(|&i| scratch.refs[i].system == system_of(&m.sat))
            .ok_or_else(|| DdRowError::MissingReference(satellite_system(&m.sat).to_string()))?;
        let rc = &scratch.refs[rc_idx];

        let code_sd_var = validate_variance(
            single_difference_variance(model.code_sigma_m, model.stochastic, base, m.pos),
            "rtk.code_variance_m2",
        )?;
        let phase_sd_var = validate_variance(
            single_difference_variance(model.phase_sigma_m, model.stochastic, base, m.pos),
            "rtk.phase_variance_m2",
        )?;
        let sat_sd_code = m.rover_code_m - m.base_code_m;
        let sat_sd_phase = m.rover_phase_m - m.base_phase_m;
        let obs_dd_code = sat_sd_code - rc.sd_code;
        let obs_dd_phase = sat_sd_phase - rc.sd_phase;
        let sat_sd_geom = geometric_range_m(m.rover_tx_pos, rover, model.sagnac)
            - geometric_range_m(m.base_tx_pos, base, model.sagnac);
        let geom_dd = sat_sd_geom - rc.sd_geom;
        let dd_receiver_correction = double_difference_receiver_antenna_correction(
            DoubleDifferenceAntennaGeometry {
                sat_pos: m.pos,
                ref_pos: rc.pos,
                base_pos: base,
                rover_pos: rover,
            },
            receiver_antenna_corrections,
            &mut scratch.receiver_antenna,
        )
        .map_err(DdRowError::ReceiverAntenna)?;
        let modeled_geom_dd = geom_dd - dd_receiver_correction;
        let sat_los = range_derivative(rover, m.rover_tx_pos);
        let dd_deriv = sub3(sat_los, rc.los);
        let code_weight = recipe.weight(code_sd_var, rc.code_var);
        let phase_weight = recipe.weight(phase_sd_var, rc.phase_var);

        // Code design row: [dx, dy, dz | 0...] (code carries no ambiguity).
        if scratch.rows_len == scratch.rows.len() {
            scratch.rows.push(DdRowScratch::default());
        }
        let row = &mut scratch.rows[scratch.rows_len];
        scratch.rows_len += 1;
        row.epoch_index = epoch_index;
        row.kind = RowKind::Code;
        assign_str(&mut row.sat, &m.sat);
        assign_str(&mut row.ref_sat, &rc.sat);
        assign_double_difference_ambiguity_id(
            &mut row.ambiguity_id,
            &m.sat,
            &m.sd_ambiguity_id,
            &rc.sat,
            &rc.sd_ambiguity_id,
        );
        assign_str(&mut row.sd_ambiguity_id, &m.sd_ambiguity_id);
        row.h.resize(n, 0.0);
        row.h.fill(0.0);
        row.h[0] = dd_deriv[0];
        row.h[1] = dd_deriv[1];
        row.h[2] = dd_deriv[2];
        row.y = obs_dd_code - modeled_geom_dd;
        row.sd_variance_m2 = code_sd_var;
        row.ref_sd_variance_m2 = rc.code_var;
        row.weight = code_weight;
        validate_built_row(row)?;

        // Phase design row: [dx, dy, dz | ambiguity column(s) per recipe].
        if scratch.rows_len == scratch.rows.len() {
            scratch.rows.push(DdRowScratch::default());
        }
        let row = &mut scratch.rows[scratch.rows_len];
        scratch.rows_len += 1;
        row.epoch_index = epoch_index;
        row.kind = RowKind::Phase;
        assign_str(&mut row.sat, &m.sat);
        assign_str(&mut row.ref_sat, &rc.sat);
        assign_double_difference_ambiguity_id(
            &mut row.ambiguity_id,
            &m.sat,
            &m.sd_ambiguity_id,
            &rc.sat,
            &rc.sd_ambiguity_id,
        );
        assign_str(&mut row.sd_ambiguity_id, &m.sd_ambiguity_id);
        row.h.resize(n, 0.0);
        row.h.fill(0.0);
        row.h[0] = dd_deriv[0];
        row.h[1] = dd_deriv[1];
        row.h[2] = dd_deriv[2];
        // The phase ambiguity column(s) and prefit value follow the recipe; the
        // composed double-difference id was just written into `row.ambiguity_id`,
        // so the float/fixed column lookup borrows it without a fresh allocation.
        let ambiguity_dd = recipe.phase_ambiguity(
            &row.ambiguity_id,
            &m.sd_ambiguity_id,
            &rc.sd_ambiguity_id,
            &mut row.h,
        )?;
        row.y = obs_dd_phase - (modeled_geom_dd + ambiguity_dd);
        row.sd_variance_m2 = phase_sd_var;
        row.ref_sd_variance_m2 = rc.phase_var;
        row.weight = phase_weight;
        validate_built_row(row)?;
    }

    Ok(&scratch.rows[..scratch.rows_len])
}
