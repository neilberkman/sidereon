//! Sequential RTK baseline filter update cluster.
//!
//! This is the streaming hot path: the pure `update(state, epoch) -> (state',
//! solution)` measurement update that consumes the [`FilterState`] ABI from the
//! `state` submodule. It drives the shared double-difference measurement model
//! ([`super::rows::dd_epoch_rows_into`]), the Gauss-Newton iterated information filter
//! ([`iterate_epoch_into`]), the LAMBDA search-and-hold step
//! ([`search_and_hold`]), the kinematic predict step, the predicted-residual
//! screen, and the public option/result/error types ([`UpdateOpts`],
//! [`EpochUpdate`], [`UpdateError`]). The row buffers, normal-equation folds,
//! integer search, and measurement-model primitives live in the sibling
//! `rows`/`normal`/`search`/`model`/`antenna` submodules and are reused here.
//!
//! Parity reference: `Sidereon.GNSS.RTK.run_sequential_baseline_filter` /
//! `sequential_initial_information`. The prior is a diagonal information matrix
//! (`1/σ²` on the baseline x/y/z and on each ambiguity diagonal, zero
//! cross-terms). Adding an ambiguity column on first sighting (a streaming
//! filter cannot see the future) is mathematically identical to the Elixir batch
//! pre-sizing every column at epoch 0: an unobserved ambiguity contributes no
//! measurement information, so it carries only its prior.

use std::collections::BTreeMap;

use crate::astro::math::linear::{
    invert_3x3_adjugate, FlatNormalSolveScratch as SolveNormalScratch,
};

use super::antenna::ReceiverAntennaCorrections;
use super::float::FloatResidual;
use super::model::{
    float_only_set, is_float_only_system, satellite_system, system_of, Epoch, MeasModel, RowKind,
    SatMeas,
};
use super::normal::{fold_hold_block_with_ambiguities, fold_measurement_block_indices};
#[cfg(test)]
use super::rows::DdRow;
use super::rows::{
    assign_str, dd_epoch_rows_into, DdRowError, DdRowRecipe, DdRowScratch, EpochRowsScratch,
};
use super::search::{search_result_from_ils, IntegerSearchMeta, IntegerStatus};
use super::state::{FilterState, FilterStateValidationError, FilterStateValidationKind};
use super::BlockFoldScratch;
use super::{AmbiguityScale, MeasContext, ReceiverAntennaError};
use crate::estimation::recipe::ResidualNormRecipe;
use crate::estimation::substrate::ambiguity::resolve_integer_lattice;
use crate::estimation::substrate::qc::normalized_residual;
use crate::id::GnssSystem;
use crate::validate;

/// Gauss-Newton iterate controls shared by the sequential filter's float and
/// fixed-report epoch solves: the fix-and-hold pseudo-measurement sigma, the
/// baseline/ambiguity step convergence tolerances, and the iteration cap.
/// Carried as one struct so the iterate argument lists stay small.
#[derive(Clone, Copy)]
pub(crate) struct IterateControls {
    pub hold_sigma_m: f64,
    pub position_tol_m: f64,
    pub ambiguity_tol_m: f64,
    pub max_iterations: usize,
}

/// Ambiguity-resolution policy for one [`search_and_hold`] step: the
/// constellations excluded from the LAMBDA search set, the optional AR arming
/// sigma gate, and the ratio-test options. Carried as one struct so the search
/// argument list stays small.
#[derive(Clone, Copy)]
pub(crate) struct SearchPolicy<'a> {
    pub float_only_systems: &'a [GnssSystem],
    pub ar_arming_sigma_m: Option<f64>,
    pub ratio: SearchOpts,
}

// =========================================================================
// Double-difference measurement model (kernel slice 2a)
// -------------------------------------------------------------------------
// Port of the Elixir `build_epoch_sequential_baseline_rows` +
// `geometry_double_difference` + `design_*_baseline_row`. The geometry is now the
// single shared builder [`super::rows::dd_epoch_rows_into`]; the sequential filter
// selects it with [`DdRowRecipe::SequentialFilter`] (single-difference ambiguity
// columns `+1`/`-1`, inverse double-difference variance weight), the same builder
// the static `float`/`fixed` baselines drive with their own recipe variants.

/// Build the double-difference code+phase rows for an epoch, linearized at
/// `state.baseline_m`. `base` is the base-station ECEF; rover = base + baseline.
/// Each satellite contributes a code row then a phase row (Elixir order).
/// Returns `None` if a satellite's SD ambiguity column is not yet in the state
/// (caller must `ensure_ambiguity` first). Test-only owned-allocation wrapper
/// over the shared [`super::rows::dd_epoch_rows_into`] builder, which the solve
/// paths drive directly.
#[cfg(test)]
pub(super) fn epoch_dd_rows(
    epoch: &Epoch,
    base: [f64; 3],
    state: &FilterState,
    model: &MeasModel,
) -> Option<Vec<DdRow>> {
    let mut scratch = EpochRowsScratch::default();
    let ctx = MeasContext {
        base,
        model,
        antenna: None,
    };
    let rows = dd_epoch_rows_into(
        ctx,
        epoch,
        0,
        state.baseline_m,
        DdRowRecipe::SequentialFilter {
            sd_ambiguity_ids: &state.sd_ambiguity_ids,
            sd_ambiguities_m: &state.sd_ambiguities_m,
        },
        &mut scratch,
    )
    .ok()?;
    Some(rows.iter().map(DdRow::from_scratch).collect())
}

// =========================================================================
// Iterated information-filter measurement update (kernel slice 2b)
// -------------------------------------------------------------------------
// Port of the Elixir `iterate_sequential_filter_epoch` (Gauss-Newton iterated
// information filter). Per iteration: relinearize the DD geometry at the current
// iterate, accumulate the measurement normal equations (+ fix-and-hold pseudo-
// measurements), add the prior contribution `Λ_prior·(center - current)` to the
// rhs, solve δx, apply, and repeat until `‖δbaseline‖ ≤ pos_tol` and
// `max|δambiguity| ≤ amb_tol` (or `max_iterations`). The posterior information is
// `Λ_prior + Λ_measurement + Λ_hold` and is carried to the next epoch.

/// A held (fixed) double-difference ambiguity, as a hold pseudo-measurement:
/// constrains `SD(sat) - SD(ref)` toward `fixed_m`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Hold {
    pub sat_sd_id: String,
    pub ref_sd_id: String,
    pub fixed_m: f64,
}

/// Posterior of one epoch's iterated update. Test-only owned form returned by
/// the `iterate_epoch` wrapper; the solve paths carry `EpochPosteriorRef`.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq)]
pub(super) struct EpochPosterior {
    pub baseline_m: [f64; 3],
    pub sd_ambiguities_m: Vec<f64>,
    /// Posterior information matrix (row-major n×n), carried to the next epoch.
    pub information: Vec<f64>,
}

#[derive(Debug)]
struct EpochPosteriorRef<'a> {
    baseline_m: [f64; 3],
    sd_ambiguities_m: &'a [f64],
    information: &'a [f64],
}

#[derive(Debug, Default)]
struct IterateScratch {
    epoch_rows: EpochRowsScratch,
    fold_block: BlockFoldScratch,
    solve: SolveNormalScratch,
    prior_center: Vec<f64>,
    work_ambiguities: Vec<f64>,
    measurement_lambda: Vec<f64>,
    measurement_eta: Vec<f64>,
    hold_lambda: Vec<f64>,
    hold_eta: Vec<f64>,
    lambda: Vec<f64>,
    eta: Vec<f64>,
    delta_center: Vec<f64>,
    prior_rhs: Vec<f64>,
    block_indices: Vec<usize>,
    block_rows: Vec<usize>,
}

#[derive(Debug, Default)]
struct HoldPool {
    holds: Vec<Hold>,
    len: usize,
}

/// Reusable buffers for the RTK filter hot path.
///
/// Create one per arc/track and pass it to [`update_epoch_with_scratch`] to keep
/// row, normal-equation, and solver workspaces allocated across epochs. The
/// existing [`update_epoch`] entry point remains available for callers that do
/// not manage scratch explicitly.
#[derive(Debug, Default)]
pub struct RtkFilterScratch {
    iterate: IterateScratch,
    report_iterate: IterateScratch,
    screen_rows: EpochRowsScratch,
    residual_rows: EpochRowsScratch,
    screen_mask: Vec<bool>,
    held: HoldPool,
}

impl RtkFilterScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

fn matvec_into(m: &[f64], v: &[f64], out: &mut [f64]) {
    let n = v.len();
    for (i, out_i) in out.iter_mut().enumerate().take(n) {
        let row = i * n;
        let mut acc = 0.0;
        for j in 0..n {
            acc += m[row + j] * v[j];
        }
        *out_i = acc;
    }
}

/// Iterated information-filter update for one epoch. `state` is the carried-in
/// prior (its `baseline_m`/`sd_ambiguities_m` are the prior center, `information`
/// the prior information). All ambiguity columns referenced by the epoch and the
/// holds must already be present in `state` (call `ensure_ambiguity` first).
/// `held` is empty for a pure float update.
///
/// PERF: clones the prior state once for relinearization and allocates the
/// per-iteration normal-equations buffers. Test-only owned-allocation wrapper
/// over `iterate_epoch_into`, which the streaming update path uses directly.
#[cfg(test)]
pub(super) fn iterate_epoch(
    ctx: MeasContext,
    state: &FilterState,
    epoch: &Epoch,
    held: &[Hold],
    controls: IterateControls,
) -> Option<EpochPosterior> {
    let mut scratch = IterateScratch::default();
    let posterior = iterate_epoch_into(ctx, state, epoch, held, controls, None, &mut scratch)
        .ok()
        .flatten()?;
    Some(EpochPosterior {
        baseline_m: posterior.baseline_m,
        sd_ambiguities_m: posterior.sd_ambiguities_m.to_vec(),
        information: posterior.information.to_vec(),
    })
}

#[allow(clippy::needless_range_loop)]
fn iterate_epoch_into<'a>(
    ctx: MeasContext,
    state: &FilterState,
    epoch: &Epoch,
    held: &[Hold],
    controls: IterateControls,
    screen_mask: Option<&[bool]>,
    scratch: &'a mut IterateScratch,
) -> Result<Option<EpochPosteriorRef<'a>>, UpdateError> {
    let IterateControls {
        hold_sigma_m,
        position_tol_m,
        ambiguity_tol_m,
        max_iterations,
    } = controls;
    let n = state.dim();
    let prior_information = &state.information;

    // Prior center (fixed across iterations): [bx, by, bz | sd ambiguities].
    scratch.prior_center.resize(n, 0.0);
    scratch.prior_center[..3].copy_from_slice(&state.baseline_m);
    scratch.prior_center[3..].copy_from_slice(&state.sd_ambiguities_m);

    // Current linearization point (starts at the prior center).
    let mut work_baseline = state.baseline_m;
    scratch
        .work_ambiguities
        .resize(state.sd_ambiguities_m.len(), 0.0);
    scratch
        .work_ambiguities
        .copy_from_slice(&state.sd_ambiguities_m);
    let hold_weight = 1.0 / (hold_sigma_m * hold_sigma_m);
    let nn = n * n;
    scratch.measurement_lambda.resize(nn, 0.0);
    scratch.measurement_eta.resize(n, 0.0);
    scratch.hold_lambda.resize(nn, 0.0);
    scratch.hold_eta.resize(n, 0.0);
    scratch.lambda.resize(nn, 0.0);
    scratch.eta.resize(n, 0.0);
    scratch.delta_center.resize(n, 0.0);
    scratch.prior_rhs.resize(n, 0.0);

    for iter in 1..=max_iterations.max(1) {
        let rows = match dd_epoch_rows_into(
            ctx,
            epoch,
            0,
            work_baseline,
            DdRowRecipe::SequentialFilter {
                sd_ambiguity_ids: &state.sd_ambiguity_ids,
                sd_ambiguities_m: &scratch.work_ambiguities,
            },
            &mut scratch.epoch_rows,
        ) {
            Ok(rows) => rows,
            Err(DdRowError::MissingAmbiguity(_)) => return Ok(None),
            Err(error) => return Err(update_row_error(error)),
        };

        scratch.measurement_lambda.fill(0.0);
        scratch.measurement_eta.fill(0.0);
        scratch.hold_lambda.fill(0.0);
        scratch.hold_eta.fill(0.0);

        // Measurement normal equations. Covariance blocks are keyed
        // {kind, reference satellite}: rows correlate only through their shared
        // reference single difference, so each system's code (and phase) double
        // differences form their own block with no cross-system correlation -
        // Elixir groups by `{epoch_idx, kind, ref_sat}` and folds the blocks in
        // sorted key order (`:code < :phase`, then reference id). With one
        // system the key is constant per kind, so this degenerates to the
        // historical two-block code-then-phase fold bit-for-bit. Within each
        // block, fold rows in satellite order - the Elixir reference sorts each
        // block (`Enum.sort_by(block_rows, & &1.sat)`) before assembling, so the
        // kernel must accumulate Σ_a Σ_b in the same order.
        scratch.block_indices.clear();
        match screen_mask {
            Some(mask) => scratch
                .block_indices
                .extend((0..rows.len()).filter(|&idx| mask.get(idx).copied().unwrap_or(false))),
            None => scratch.block_indices.extend(0..rows.len()),
        }
        scratch.block_indices.sort_by(|&a, &b| {
            (rows[a].kind, rows[a].ref_sat.as_str()).cmp(&(rows[b].kind, rows[b].ref_sat.as_str()))
        });
        let mut start = 0;
        while start < scratch.block_indices.len() {
            let first = scratch.block_indices[start];
            let kind = rows[first].kind;
            let ref_sat = rows[first].ref_sat.as_str();
            let mut end = start + 1;
            while end < scratch.block_indices.len() {
                let idx = scratch.block_indices[end];
                if rows[idx].kind != kind || rows[idx].ref_sat != ref_sat {
                    break;
                }
                end += 1;
            }
            scratch.block_rows.clear();
            scratch
                .block_rows
                .extend_from_slice(&scratch.block_indices[start..end]);
            scratch
                .block_rows
                .sort_by(|&a, &b| rows[a].sat.cmp(&rows[b].sat));
            fold_measurement_block_indices(
                &mut scratch.measurement_lambda,
                &mut scratch.measurement_eta,
                rows,
                &scratch.block_rows,
                &mut scratch.fold_block,
            )
            .ok_or(UpdateError::SingularGeometry)?;
            start = end;
        }

        // Fix-and-hold pseudo-measurements: SD(sat) - SD(ref) pulled to fixed_m.
        fold_hold_block_with_ambiguities(
            &mut scratch.hold_lambda,
            &mut scratch.hold_eta,
            state,
            &scratch.work_ambiguities,
            held,
            hold_weight,
        )
        .ok_or(UpdateError::SingularGeometry)?;

        // Prior contribution: rhs += Λ_prior·(center - current). Assemble the
        // matrix/vector in Elixir's exact grouping:
        //   information = (prior + measurement) + hold
        //   rhs         = (measurement + hold) + prior_rhs
        scratch.delta_center[0] = scratch.prior_center[0] - work_baseline[0];
        scratch.delta_center[1] = scratch.prior_center[1] - work_baseline[1];
        scratch.delta_center[2] = scratch.prior_center[2] - work_baseline[2];
        for i in 0..scratch.work_ambiguities.len() {
            scratch.delta_center[3 + i] = scratch.prior_center[3 + i] - scratch.work_ambiguities[i];
        }
        matvec_into(
            prior_information,
            &scratch.delta_center,
            &mut scratch.prior_rhs,
        );
        for i in 0..(n * n) {
            scratch.lambda[i] =
                (prior_information[i] + scratch.measurement_lambda[i]) + scratch.hold_lambda[i];
        }
        for i in 0..n {
            scratch.eta[i] =
                (scratch.measurement_eta[i] + scratch.hold_eta[i]) + scratch.prior_rhs[i];
        }

        // Per-system SD gauge fixing (op-for-op `apply_reference_sd_gauge/8`).
        // Every measurement row, hold, and reported quantity depends only on
        // within-system single-difference DIFFERENCES, so the common-mode level
        // of each system's SD ambiguity block is unobservable; once per-epoch
        // information accumulates, the 1/σ² initial prior carrying that gauge
        // direction falls below float precision and the pivot cancels exactly to
        // zero. Pin each per-system reference SD ambiguity at its prior-center
        // value with the hold weight - a pure gauge constraint (double
        // differences and the baseline are invariant). Applied to single-system
        // arcs too: the one system's reference SD is equally a gauge DOF and its
        // pivot cancels on a long tight-hold arc (the epoch-124 singularity)
        // without this. Applied AFTER the (prior + measurement) + hold assembly
        // and BEFORE the solve, exactly where the Elixir iterate folds it; the
        // gauge terms stay in `lambda`, so the carried posterior information
        // includes them.
        if !state.references.is_empty() {
            // Iterate the run-level references in sorted-system order (Elixir
            // `refs |> Enum.sort()`), skipping systems absent this epoch.
            for system in state.references.keys() {
                // Type the per-system match through GnssSystem. The references
                // map keys stay String at the NIF boundary; an unrecognized key
                // letter parses to None and matches no valid epoch reference,
                // exactly as the prior raw leading-letter string compare.
                let Some(system) = system_of(system) else {
                    continue;
                };
                let Some(r) = epoch
                    .references
                    .iter()
                    .find(|m| system_of(&m.sat) == Some(system))
                else {
                    continue;
                };
                let Some(pos) = state.ambiguity_pos(&r.sd_ambiguity_id) else {
                    return Ok(None);
                };
                let col = 3 + pos;
                let residual = scratch.prior_center[col] - scratch.work_ambiguities[col - 3];
                scratch.lambda[col * n + col] += hold_weight;
                scratch.eta[col] += hold_weight * residual;
            }
        }

        let dx = super::RTK_ASSEMBLER
            .solve_flat_first_tie(&scratch.lambda, &scratch.eta, &mut scratch.solve)
            .ok_or(UpdateError::SingularGeometry)?;

        // Apply: x += δx (baseline first, then ambiguities).
        for (k, b) in work_baseline.iter_mut().enumerate() {
            *b += dx[k];
        }
        for (k, a) in scratch.work_ambiguities.iter_mut().enumerate() {
            *a += dx[3 + k];
        }

        let baseline_step = (dx[0] * dx[0] + dx[1] * dx[1] + dx[2] * dx[2]).sqrt();
        let ambiguity_step = dx[3..].iter().fold(0.0_f64, |m, &v| m.max(v.abs()));

        if (baseline_step <= position_tol_m && ambiguity_step <= ambiguity_tol_m)
            || iter >= max_iterations.max(1)
        {
            return Ok(Some(EpochPosteriorRef {
                baseline_m: work_baseline,
                sd_ambiguities_m: &scratch.work_ambiguities,
                information: &scratch.lambda,
            }));
        }
    }
    Ok(None)
}

// =========================================================================
// Ambiguity search inputs (kernel slice 2c-i): SD -> DD covariance in cycles
// -------------------------------------------------------------------------
// Port of the Elixir `dd_covariance_m_to_cycles` / `dd_covariance_m`. Given the
// single-difference ambiguity covariance block (metres²) from the inverted
// posterior information, build the double-difference ambiguity covariance in
// cycles for the LAMBDA search. For DD targets i,j whose SD positions are
// (sat_i, ref_i): the DD = sat_SD - ref_SD, so
//   DD_cov[i][j] = (C[si][sj] - C[si][rj] - C[ri][sj] + C[ri][rj]) / (λ_i λ_j).

/// Double-difference ambiguity covariance (cycles) from the `k x k` SD ambiguity
/// covariance block `sd_cov` (row-major, metres²). `dd[i] = (sat_pos, ref_pos)`
/// are SD-block indices; `wavelengths[i]` is the carrier wavelength (metres) of
/// DD target `i`.
#[allow(clippy::needless_range_loop)]
pub(crate) fn dd_covariance_cycles(
    sd_cov: &[f64],
    k: usize,
    dd: &[(usize, usize)],
    wavelengths: &[f64],
) -> Vec<f64> {
    let m = dd.len();
    let mut out = vec![0.0; m * m];
    for i in 0..m {
        let (si, ri) = dd[i];
        for j in 0..m {
            let (sj, rj) = dd[j];
            // Op-for-op with Elixir `dd_covariance_m/3`: two inner reductions
            // `(A - B)` and `(-C + D)`, then the outer addition. The flat
            // left-associated form `((A - B) - C) + D` is algebraically
            // equivalent but changes real-arc LAMBDA ratios at ~1e-11.
            let sat_terms = sd_cov[si * k + sj] + -sd_cov[si * k + rj];
            let ref_terms = -sd_cov[ri * k + sj] + sd_cov[ri * k + rj];
            let c = sat_terms + ref_terms;
            out[i * m + j] = c / (wavelengths[i] * wavelengths[j]);
        }
    }
    out
}

// =========================================================================
// Ambiguity search-and-hold (kernel slice 2c-ii)
// -------------------------------------------------------------------------
// Port of the Elixir `sequential_search_and_hold`. Invert the posterior
// information to covariance, take the SD ambiguity block, transform the
// not-yet-held DD ambiguities to cycles, compute the float DD cycles, run the
// LAMBDA kernel (ils.rs), and on a passing ratio test merge the accepted
// integers into the held fixes (each recording its (sat,ref) SD pair so the
// hold pseudo-measurement is correct even if the reference satellite changes).

/// Ratio-test options for the ambiguity search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchOpts {
    pub ratio_threshold: f64,
}

/// Baseline dynamics used by the prediction step before each measurement update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DynamicsModel {
    /// Keep the carried baseline mean fixed. This is the historical RTK filter.
    ConstantPosition,
    /// Advance the baseline mean by the epoch velocity times elapsed seconds.
    /// Process-noise meaning is unchanged.
    VelocityPropagated,
}

/// Optional predicted-residual screen for one epoch update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InnovationScreenOpts {
    pub threshold_sigma: f64,
    pub min_rows: usize,
}

/// Diagnostics from the optional predicted-residual screen.
#[derive(Debug, Clone, PartialEq)]
pub struct InnovationScreen {
    pub threshold_sigma: f64,
    pub min_rows: usize,
    pub input_rows: usize,
    pub accepted_rows: usize,
    pub rejected_rows: usize,
    pub rejected_code_rows: usize,
    pub rejected_phase_rows: usize,
    pub max_abs_normalized_innovation: Option<f64>,
    pub max_rejected_abs_normalized_innovation: Option<f64>,
    pub coasted: bool,
}

/// Options for one streaming filter epoch update.
#[derive(Debug, Clone, PartialEq)]
pub struct UpdateOpts {
    pub hold_sigma_m: f64,
    pub position_tol_m: f64,
    pub ambiguity_tol_m: f64,
    pub max_iterations: usize,
    /// Kinematic process-noise sigma (metres) for the baseline. `0.0` (default)
    /// is the static filter: the carried information is propagated forward
    /// untouched. When `> 0`, a between-epoch predict step inflates the baseline
    /// covariance block by `sigma^2` (the `Λ⁻¹ + Q` round-trip), skipped on the
    /// first epoch. Mirrors the Elixir `:process_noise_baseline_sigma_m`.
    pub process_noise_baseline_sigma_m: f64,
    /// Prediction mean dynamics. Defaults to [`DynamicsModel::ConstantPosition`]
    /// in public callers.
    pub dynamics_model: DynamicsModel,
    /// Constellation letters whose double-difference ambiguities never enter
    /// the LAMBDA search set - they contribute float measurement rows only
    /// (GLONASS FDMA is the canonical use). Mirrors the Elixir
    /// `:float_only_systems`.
    pub float_only_systems: Vec<String>,
    /// Optional predicted-residual screen for the Rust kernel update.
    pub innovation_screen: Option<InnovationScreenOpts>,
    /// Emit public residual diagnostics in [`EpochUpdate`]. Keep this disabled
    /// for pure state-carry hot-path callers.
    pub report_residuals: bool,
    /// Test hook used to verify newly fixed epochs surface report solve failures.
    #[cfg(test)]
    pub force_report_iterate_failure: bool,
    /// Optional receiver-antenna PCO/PCV corrections applied inside DD row construction.
    pub receiver_antenna_corrections: Option<ReceiverAntennaCorrections>,
    /// AR commitment arming gate (mirrors the Elixir `:ar_arming_sigma_m`).
    /// When set, the ambiguity search is attempted only once the baseline-block
    /// posterior standard deviation has converged to at most this many metres;
    /// below the gate the epoch carries its existing held set and stays float.
    /// `None` (default) keeps the always-armed behaviour.
    pub ar_arming_sigma_m: Option<f64>,
    pub search: SearchOpts,
}

/// Result of one streaming filter epoch update.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochUpdate {
    /// Updated serializable filter state. Its `baseline_m` is the FLOAT posterior
    /// (the carried Kalman/information state); use [`Self::reported_baseline_m`]
    /// for the reported solution.
    pub state: FilterState,
    /// Ambiguity-conditioned ("fixed") baseline for this epoch's reported
    /// solution. When AR newly succeeds, this re-solves the epoch with the new
    /// integers held so the reported baseline reflects the fix in the same epoch
    /// (matching RTKLIB's fixed solution); otherwise it equals `state.baseline_m`.
    /// The carried `state` keeps the float baseline so the next epoch linearizes
    /// from the Kalman posterior, not the conditioned point.
    pub reported_baseline_m: [f64; 3],
    /// Single-difference ambiguity vector at the reported solution when it differs
    /// from the carried state, i.e. a same-epoch fixed report re-solve succeeded.
    /// `None` means the carried state's ambiguity vector is also the reported one.
    pub reported_sd_ambiguities_m: Option<Vec<f64>>,
    /// Integer ratio from this epoch's ambiguity search (`0.0` when all observed
    /// ambiguities were already held).
    pub integer_ratio: f64,
    /// LAMBDA search diagnostics for this epoch. `None` means no search ran
    /// (all observed ambiguities were already held, or an AR arming gate withheld
    /// the search).
    pub search: Option<IntegerSearchMeta>,
    /// Whether the state has any held integer ambiguity after this update.
    pub integer_fixed: bool,
    /// Ambiguity ids newly fixed during this epoch.
    pub newly_fixed: Vec<String>,
    /// All held ambiguity ids after this epoch.
    pub fixed_ids: Vec<String>,
    /// Public residual rows computed at the reported solution for this epoch.
    pub residuals: Vec<FloatResidual>,
    /// Optional predicted-residual screen metrics.
    pub innovation_screen: Option<InnovationScreen>,
}

/// Why a carried [`FilterState`] is not structurally valid for an update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvalidStateKind {
    /// A state vector or matrix length did not match the state dimension.
    Length { expected: usize, actual: usize },
    /// A state dimension was too large to form the row-major information shape.
    DimensionOverflow,
    /// A floating-point state field was NaN or infinite.
    NonFinite,
    /// A state sigma was zero or negative.
    NotPositive,
    /// The row-major state information matrix was not symmetric.
    NotSymmetric,
    /// The row-major state information matrix was not positive semidefinite.
    NotPositiveSemidefinite,
}

impl core::fmt::Display for InvalidStateKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Length { expected, actual } => {
                write!(f, "length is {actual}, expected {expected}")
            }
            Self::DimensionOverflow => f.write_str("dimension overflow"),
            Self::NonFinite => f.write_str("not finite"),
            Self::NotPositive => f.write_str("not positive"),
            Self::NotSymmetric => f.write_str("not symmetric"),
            Self::NotPositiveSemidefinite => f.write_str("not positive semidefinite"),
        }
    }
}

/// Why one streaming RTK filter update could not complete.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UpdateError {
    /// The carried serialized filter state is malformed and cannot be indexed safely.
    InvalidState {
        field: &'static str,
        kind: InvalidStateKind,
    },
    /// The carried filter state is tied to one reference ambiguity arc per
    /// system; applying held DDs to a different reference would reinterpret
    /// fixed integers.
    ReferenceChanged {
        system: String,
        expected: String,
        actual: String,
    },
    /// The epoch carries a reference for a constellation the state does not
    /// track (the per-system reference set is fixed at filter construction).
    UnknownReferenceSystem(String),
    /// A non-reference satellite's constellation has no reference this epoch
    /// (or a held ambiguity's constellation has no reference in the state).
    MissingSystemReference(String),
    /// An ambiguity needed by the epoch is not present in the state column set.
    MissingAmbiguityColumn(String),
    /// A target ambiguity has no wavelength entry.
    MissingWavelength(String),
    /// A target ambiguity has no metre offset entry.
    MissingOffset(String),
    /// A row-builder boundary input was malformed, non-finite, or outside its
    /// physical domain.
    InvalidInput {
        field: &'static str,
        kind: super::RtkInputErrorKind,
    },
    /// The measurement/update normal equations or posterior covariance were singular.
    SingularGeometry,
    /// A provided receiver-antenna calibration could not be applied.
    ReceiverAntenna(ReceiverAntennaError),
    /// The integer least-squares search rejected the ambiguity covariance/input.
    Ils(crate::ils::IlsError),
}

impl core::fmt::Display for UpdateError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidState { field, kind } => {
                write!(f, "invalid RTK filter state {field}: {kind}")
            }
            Self::ReferenceChanged {
                system,
                expected,
                actual,
            } => write!(
                f,
                "RTK reference changed for constellation {system}: expected {expected}, got {actual}"
            ),
            Self::UnknownReferenceSystem(system) => {
                write!(f, "unknown RTK reference constellation {system}")
            }
            Self::MissingSystemReference(system) => {
                write!(f, "missing RTK reference satellite for constellation {system}")
            }
            Self::MissingAmbiguityColumn(id) => {
                write!(f, "missing RTK ambiguity column {id}")
            }
            Self::MissingWavelength(id) => write!(f, "missing RTK wavelength for ambiguity {id}"),
            Self::MissingOffset(id) => write!(f, "missing RTK offset for ambiguity {id}"),
            Self::InvalidInput { field, kind } => {
                write!(f, "invalid RTK update input {field}: {kind}")
            }
            Self::SingularGeometry => write!(f, "RTK update geometry is singular"),
            Self::ReceiverAntenna(error) => write!(f, "{error}"),
            Self::Ils(error) => write!(f, "RTK integer ambiguity search failed: {error}"),
        }
    }
}

impl std::error::Error for UpdateError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ReceiverAntenna(error) => Some(error),
            Self::Ils(error) => Some(error),
            _ => None,
        }
    }
}

fn invalid_state(error: FilterStateValidationError) -> UpdateError {
    let kind = match error.kind {
        FilterStateValidationKind::Length { expected, actual } => {
            InvalidStateKind::Length { expected, actual }
        }
        FilterStateValidationKind::DimensionOverflow => InvalidStateKind::DimensionOverflow,
        FilterStateValidationKind::NonFinite => InvalidStateKind::NonFinite,
        FilterStateValidationKind::NotPositive => InvalidStateKind::NotPositive,
        FilterStateValidationKind::NotSymmetric => InvalidStateKind::NotSymmetric,
        FilterStateValidationKind::NotPositiveSemidefinite => {
            InvalidStateKind::NotPositiveSemidefinite
        }
    };
    UpdateError::InvalidState {
        field: error.field,
        kind,
    }
}

fn update_row_error(error: DdRowError) -> UpdateError {
    match error {
        DdRowError::MissingReference(system) => UpdateError::MissingSystemReference(system),
        DdRowError::MissingAmbiguity(id) => UpdateError::MissingAmbiguityColumn(id),
        DdRowError::ReceiverAntenna(error) => UpdateError::ReceiverAntenna(error),
        DdRowError::InvalidInput { field, kind } => UpdateError::InvalidInput { field, kind },
    }
}

fn invalid_update_input(error: validate::FieldError) -> UpdateError {
    UpdateError::InvalidInput {
        field: error.field(),
        kind: super::RtkInputErrorKind::from(&error),
    }
}

fn invalid_update_option(field: &'static str, kind: super::RtkInputErrorKind) -> UpdateError {
    UpdateError::InvalidInput { field, kind }
}

fn validate_inverse_variance_sigma(sigma_m: f64, field: &'static str) -> Result<(), UpdateError> {
    validate::finite_positive(sigma_m, field).map_err(invalid_update_input)?;
    let weight = 1.0 / (sigma_m * sigma_m);
    if !weight.is_finite() {
        return Err(invalid_update_option(
            field,
            super::RtkInputErrorKind::NonFinite,
        ));
    }
    if weight <= 0.0 {
        return Err(invalid_update_option(
            field,
            super::RtkInputErrorKind::NotPositive,
        ));
    }
    Ok(())
}

fn validate_update_opts(opts: &UpdateOpts) -> Result<(), UpdateError> {
    validate_inverse_variance_sigma(opts.hold_sigma_m, "rtk.update.hold_sigma_m")?;
    validate::finite_positive(opts.position_tol_m, "rtk.update.position_tol_m")
        .map_err(invalid_update_input)?;
    validate::finite_positive(opts.ambiguity_tol_m, "rtk.update.ambiguity_tol_m")
        .map_err(invalid_update_input)?;
    if opts.max_iterations == 0 {
        return Err(invalid_update_option(
            "rtk.update.max_iterations",
            super::RtkInputErrorKind::NotPositive,
        ));
    }
    validate::finite_nonneg(
        opts.process_noise_baseline_sigma_m,
        "rtk.update.process_noise_baseline_sigma_m",
    )
    .map_err(invalid_update_input)?;

    if let Some(screen) = opts.innovation_screen {
        validate::finite_positive(
            screen.threshold_sigma,
            "rtk.update.innovation_screen.threshold_sigma",
        )
        .map_err(invalid_update_input)?;
        if screen.min_rows == 0 {
            return Err(invalid_update_option(
                "rtk.update.innovation_screen.min_rows",
                super::RtkInputErrorKind::NotPositive,
            ));
        }
    }

    if let Some(ar_arming_sigma_m) = opts.ar_arming_sigma_m {
        validate::finite_positive(ar_arming_sigma_m, "rtk.update.ar_arming_sigma_m")
            .map_err(invalid_update_input)?;
    }

    validate::finite_positive(
        opts.search.ratio_threshold,
        "rtk.update.search.ratio_threshold",
    )
    .map_err(invalid_update_input)?;
    Ok(())
}

fn empty_filter_residuals() -> Vec<FloatResidual> {
    Vec::new()
}

fn filter_residuals(rows: &[DdRowScratch]) -> Result<Vec<FloatResidual>, UpdateError> {
    let mut residuals = Vec::with_capacity(rows.len() / 2);
    let mut idx = 0;
    while idx < rows.len() {
        let code = rows.get(idx).ok_or(UpdateError::SingularGeometry)?;
        let phase = rows.get(idx + 1).ok_or(UpdateError::SingularGeometry)?;
        if code.kind != RowKind::Code
            || phase.kind != RowKind::Phase
            || code.sat != phase.sat
            || code.ref_sat != phase.ref_sat
            || code.ambiguity_id != phase.ambiguity_id
        {
            return Err(UpdateError::SingularGeometry);
        }
        let code_sigma_m = (code.sd_variance_m2 + code.ref_sd_variance_m2).sqrt();
        let phase_sigma_m = (phase.sd_variance_m2 + phase.ref_sd_variance_m2).sqrt();
        residuals.push(FloatResidual {
            epoch_index: 0,
            satellite_id: code.sat.clone(),
            reference_satellite_id: code.ref_sat.clone(),
            ambiguity_id: code.ambiguity_id.as_str().to_string(),
            code_m: code.y,
            phase_m: phase.y,
            code_sigma_m,
            phase_sigma_m,
            code_normalized: code.y / code_sigma_m,
            phase_normalized: phase.y / phase_sigma_m,
        });
        idx += 2;
    }
    Ok(residuals)
}

fn prepare_innovation_screen(
    state: &FilterState,
    epoch: &Epoch,
    base: [f64; 3],
    model: &MeasModel,
    opts: &UpdateOpts,
    rows_scratch: &mut EpochRowsScratch,
    mask: &mut Vec<bool>,
) -> Result<Option<InnovationScreen>, UpdateError> {
    let Some(screen) = opts.innovation_screen else {
        mask.clear();
        return Ok(None);
    };

    let ctx = MeasContext {
        base,
        model,
        antenna: opts.receiver_antenna_corrections.as_ref(),
    };
    let rows = dd_epoch_rows_into(
        ctx,
        epoch,
        0,
        state.baseline_m,
        DdRowRecipe::SequentialFilter {
            sd_ambiguity_ids: &state.sd_ambiguity_ids,
            sd_ambiguities_m: &state.sd_ambiguities_m,
        },
        rows_scratch,
    )
    .map_err(update_row_error)?;

    mask.clear();
    mask.reserve(rows.len());

    let mut accepted_rows = 0usize;
    let mut rejected_rows = 0usize;
    let mut rejected_code_rows = 0usize;
    let mut rejected_phase_rows = 0usize;
    let mut max_abs_normalized_innovation = None;
    let mut max_rejected_abs_normalized_innovation = None;

    for row in rows {
        let normalized = normalized_residual(
            ResidualNormRecipe::RtkInverseVarianceInnovation,
            row.y,
            row.weight,
        )
        .abs();
        max_abs_normalized_innovation = Some(
            max_abs_normalized_innovation
                .map_or(normalized, |current: f64| current.max(normalized)),
        );

        let rejected = normalized > screen.threshold_sigma;
        mask.push(!rejected);
        if rejected {
            rejected_rows += 1;
            match row.kind {
                RowKind::Code => rejected_code_rows += 1,
                RowKind::Phase => rejected_phase_rows += 1,
            }
            max_rejected_abs_normalized_innovation = Some(
                max_rejected_abs_normalized_innovation
                    .map_or(normalized, |current: f64| current.max(normalized)),
            );
        } else {
            accepted_rows += 1;
        }
    }

    Ok(Some(InnovationScreen {
        threshold_sigma: screen.threshold_sigma,
        min_rows: screen.min_rows,
        input_rows: rows.len(),
        accepted_rows,
        rejected_rows,
        rejected_code_rows,
        rejected_phase_rows,
        max_abs_normalized_innovation,
        max_rejected_abs_normalized_innovation,
        coasted: accepted_rows < screen.min_rows,
    }))
}

fn reported_epoch_residuals(
    ctx: MeasContext,
    state: &FilterState,
    epoch: &Epoch,
    reported_baseline_m: [f64; 3],
    reported_sd_ambiguities_m: &[f64],
    scratch: &mut EpochRowsScratch,
) -> Result<Vec<FloatResidual>, UpdateError> {
    let rows = dd_epoch_rows_into(
        ctx,
        epoch,
        0,
        reported_baseline_m,
        DdRowRecipe::SequentialFilter {
            sd_ambiguity_ids: &state.sd_ambiguity_ids,
            sd_ambiguities_m: reported_sd_ambiguities_m,
        },
        scratch,
    )
    .map_err(update_row_error)?;

    filter_residuals(rows)
}

fn coasted_update(
    mut state: FilterState,
    epoch: &Epoch,
    base: [f64; 3],
    model: &MeasModel,
    opts: &UpdateOpts,
    innovation_screen: Option<InnovationScreen>,
    residual_scratch: &mut EpochRowsScratch,
) -> Result<EpochUpdate, UpdateError> {
    let residuals = if opts.report_residuals {
        let ctx = MeasContext {
            base,
            model,
            antenna: opts.receiver_antenna_corrections.as_ref(),
        };
        reported_epoch_residuals(
            ctx,
            &state,
            epoch,
            state.baseline_m,
            &state.sd_ambiguities_m,
            residual_scratch,
        )?
    } else {
        empty_filter_residuals()
    };
    let fixed_ids: Vec<String> = state.fixed_cycles.keys().cloned().collect();
    state.epoch_count += 1;
    Ok(EpochUpdate {
        reported_baseline_m: state.baseline_m,
        reported_sd_ambiguities_m: None,
        integer_ratio: 0.0,
        search: None,
        integer_fixed: !fixed_ids.is_empty(),
        newly_fixed: Vec::new(),
        fixed_ids,
        residuals,
        innovation_screen,
        state,
    })
}

/// Run the LAMBDA search on the epoch's not-yet-held double differences and hold
/// any that fix. `posterior_information` is the `n x n` matrix from the iterated
/// epoch update; `wavelengths`/`offsets` are keyed by the non-reference
/// satellite's SD ambiguity id. Double differences whose satellite belongs to a
/// system in `float_only_systems` never enter the search set (Elixir
/// `float_only_ambiguity_ids` rejection in `sequential_search_and_hold`).
/// Returns the updated held set and the ratio (`0.0` when there was nothing to
/// search).
#[allow(clippy::needless_range_loop)]
pub(crate) fn search_and_hold(
    state: &FilterState,
    posterior_information: &[f64],
    epoch: &Epoch,
    scale: AmbiguityScale,
    held: &[Hold],
    policy: SearchPolicy,
) -> Result<(Vec<Hold>, Option<IntegerSearchMeta>), UpdateError> {
    let AmbiguityScale {
        wavelengths_m: wavelengths,
        offsets_m: offsets,
    } = scale;
    let SearchPolicy {
        float_only_systems,
        ar_arming_sigma_m,
        ratio: opts,
    } = policy;
    let n = state.dim();
    let held_sats: std::collections::BTreeSet<&str> =
        held.iter().map(|h| h.sat_sd_id.as_str()).collect();

    // DD candidates: non-reference sats observed this epoch, not already held,
    // not in a float-only system.
    let targets: Vec<&SatMeas> = epoch
        .nonref
        .iter()
        .filter(|m| {
            !held_sats.contains(m.sd_ambiguity_id.as_str())
                && !is_float_only_system(&m.sat, float_only_systems)
        })
        .collect();
    if targets.is_empty() {
        return Ok((held.to_vec(), None));
    }

    // Posterior covariance; take the SD ambiguity block (indices 3..n).
    let info_rows: Vec<Vec<f64>> = (0..n)
        .map(|i| posterior_information[i * n..i * n + n].to_vec())
        .collect();
    let cov = crate::ils::invert(&info_rows).map_err(|_| UpdateError::SingularGeometry)?;

    // AR commitment arming gate: skip the search (carry the held set, ratio 0.0)
    // while the baseline-block posterior sigma is above the threshold. Op order
    // matches the Elixir `ar_armed?` reference exactly.
    if let Some(threshold_m) = ar_arming_sigma_m {
        let base_sigma_m = (cov[0][0].max(0.0) + cov[1][1].max(0.0) + cov[2][2].max(0.0)).sqrt();

        if base_sigma_m > threshold_m {
            return Ok((held.to_vec(), None));
        }
    }

    let k = n - 3;
    let mut sd_block = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            sd_block[i * k + j] = cov[3 + i][3 + j];
        }
    }

    // DD target SD-block indices, wavelengths, and float DD cycles. Each DD
    // pairs the satellite's SD ambiguity with its OWN system's reference.
    let mut dd_idx = Vec::with_capacity(targets.len());
    let mut wl = Vec::with_capacity(targets.len());
    let mut float_cycles = Vec::with_capacity(targets.len());
    let mut target_ids = Vec::with_capacity(targets.len());
    let mut float_cycle_pairs = Vec::with_capacity(targets.len());
    let mut target_offsets = BTreeMap::new();
    for m in &targets {
        let system = satellite_system(&m.sat);
        let ref_sd = state
            .references
            .get(system)
            .ok_or_else(|| UpdateError::MissingSystemReference(system.to_string()))?
            .as_str();
        let ref_pos = state
            .ambiguity_pos(ref_sd)
            .ok_or_else(|| UpdateError::MissingAmbiguityColumn(ref_sd.to_string()))?;
        let sat_pos = state
            .ambiguity_pos(&m.sd_ambiguity_id)
            .ok_or_else(|| UpdateError::MissingAmbiguityColumn(m.sd_ambiguity_id.clone()))?;
        dd_idx.push((sat_pos, ref_pos));
        let lambda = *wavelengths
            .get(&m.sd_ambiguity_id)
            .ok_or_else(|| UpdateError::MissingWavelength(m.sd_ambiguity_id.clone()))?;
        let offset = *offsets
            .get(&m.sd_ambiguity_id)
            .ok_or_else(|| UpdateError::MissingOffset(m.sd_ambiguity_id.clone()))?;
        wl.push(lambda);
        let amb_m = state
            .dd_ambiguity_m(&m.sd_ambiguity_id, ref_sd)
            .ok_or_else(|| UpdateError::MissingAmbiguityColumn(m.sd_ambiguity_id.clone()))?;
        let float_cycle = (amb_m - offset) / lambda;
        target_ids.push(m.sd_ambiguity_id.clone());
        float_cycles.push(float_cycle);
        float_cycle_pairs.push((m.sd_ambiguity_id.clone(), float_cycle));
        target_offsets.insert(m.sd_ambiguity_id.clone(), offset);
    }

    let m_dd = targets.len();
    let dd_cov = dd_covariance_cycles(&sd_block, k, &dd_idx, &wl);
    let dd_cov_rows: Vec<Vec<f64>> = (0..m_dd)
        .map(|i| dd_cov[i * m_dd..i * m_dd + m_dd].to_vec())
        .collect();

    let result = resolve_integer_lattice(&float_cycles, &dd_cov_rows, opts.ratio_threshold)
        .map_err(UpdateError::Ils)?;
    let mut search_result = search_result_from_ils(&target_ids, &float_cycle_pairs, result);
    search_result.meta.ambiguity_offsets_m = target_offsets.into_iter().collect();

    if search_result.meta.integer_status == IntegerStatus::Fixed {
        let mut updated = held.to_vec();
        for t in &targets {
            let cycles = *search_result
                .fixed_cycles
                .get(&t.sd_ambiguity_id)
                .expect("search result contains target id");
            let lambda = *wavelengths
                .get(&t.sd_ambiguity_id)
                .ok_or_else(|| UpdateError::MissingWavelength(t.sd_ambiguity_id.clone()))?;
            let offset = *offsets
                .get(&t.sd_ambiguity_id)
                .ok_or_else(|| UpdateError::MissingOffset(t.sd_ambiguity_id.clone()))?;
            let system = satellite_system(&t.sat);
            let ref_sd = state
                .references
                .get(system)
                .ok_or_else(|| UpdateError::MissingSystemReference(system.to_string()))?;
            updated.push(Hold {
                sat_sd_id: t.sd_ambiguity_id.clone(),
                ref_sd_id: ref_sd.clone(),
                fixed_m: cycles as f64 * lambda + offset,
            });
        }
        Ok((updated, Some(search_result.meta)))
    } else {
        Ok((held.to_vec(), Some(search_result.meta)))
    }
}

fn phase_code_seed_m(meas: &SatMeas) -> f64 {
    (meas.rover_phase_m - meas.base_phase_m) - (meas.rover_code_m - meas.base_code_m)
}

fn held_from_state(state: &FilterState) -> Result<Vec<Hold>, UpdateError> {
    state
        .fixed_m
        .iter()
        .map(|(sat_sd_id, &fixed_m)| {
            let system = satellite_system(sat_sd_id);
            let ref_sd_id = state
                .references
                .get(system)
                .ok_or_else(|| UpdateError::MissingSystemReference(system.to_string()))?;
            Ok(Hold {
                sat_sd_id: sat_sd_id.clone(),
                ref_sd_id: ref_sd_id.clone(),
                fixed_m,
            })
        })
        .collect()
}

fn held_from_state_into<'a>(
    state: &FilterState,
    pool: &'a mut HoldPool,
) -> Result<&'a [Hold], UpdateError> {
    pool.len = 0;
    for (sat_sd_id, &fixed_m) in &state.fixed_m {
        let system = satellite_system(sat_sd_id);
        let ref_sd_id = state
            .references
            .get(system)
            .ok_or_else(|| UpdateError::MissingSystemReference(system.to_string()))?;
        if pool.len == pool.holds.len() {
            pool.holds.push(Hold {
                sat_sd_id: String::new(),
                ref_sd_id: String::new(),
                fixed_m: 0.0,
            });
        }
        let hold = &mut pool.holds[pool.len];
        pool.len += 1;
        assign_str(&mut hold.sat_sd_id, sat_sd_id);
        assign_str(&mut hold.ref_sd_id, ref_sd_id);
        hold.fixed_m = fixed_m;
    }
    Ok(&pool.holds[..pool.len])
}

fn held_contains_sat(held: &[Hold], sat_sd_id: &str) -> bool {
    held.iter().any(|h| h.sat_sd_id == sat_sd_id)
}

fn has_search_targets(epoch: &Epoch, held: &[Hold], float_only_systems: &[GnssSystem]) -> bool {
    epoch.nonref.iter().any(|m| {
        !held_contains_sat(held, &m.sd_ambiguity_id)
            && !is_float_only_system(&m.sat, float_only_systems)
    })
}

type FixedMaps = (BTreeMap<String, i64>, BTreeMap<String, f64>);

fn fixed_maps_from_holds(
    holds: &[Hold],
    wavelengths: &BTreeMap<String, f64>,
    offsets: &BTreeMap<String, f64>,
) -> Result<FixedMaps, UpdateError> {
    let mut fixed_cycles = BTreeMap::new();
    let mut fixed_m = BTreeMap::new();
    for hold in holds {
        let lambda = *wavelengths
            .get(&hold.sat_sd_id)
            .ok_or_else(|| UpdateError::MissingWavelength(hold.sat_sd_id.clone()))?;
        let offset = *offsets
            .get(&hold.sat_sd_id)
            .ok_or_else(|| UpdateError::MissingOffset(hold.sat_sd_id.clone()))?;
        let cycles = ((hold.fixed_m - offset) / lambda).round() as i64;
        fixed_cycles.insert(hold.sat_sd_id.clone(), cycles);
        fixed_m.insert(hold.sat_sd_id.clone(), hold.fixed_m);
    }
    Ok((fixed_cycles, fixed_m))
}

/// Kinematic predict: rank-3 (Woodbury) inflation of the baseline (indices 0..3)
/// covariance block of an `n x n` row-major information matrix by `q = sigma^2`:
///
///   Λ' = Λ - Λ[:,0..3] · (Q⁻¹ + Λ₀₀)⁻¹ · Λ[0..3,:],   Q = q·I₃
///
/// Mathematically equal to adding `q` to the baseline covariance diagonal and
/// re-inverting, but via a single 3×3 inverse instead of two full `n x n`
/// inversions. The double full-invert corrupts near-singular ambiguity
/// directions when `q` does not dominate (the filter then goes singular at small
/// process noise); this rank-3 form is stable for any `q` and preserves the
/// ambiguity block exactly. Returns `None` on a singular 3×3 system (caller keeps
/// the un-inflated prior). Same operation order as the Elixir
/// `inflate_baseline_information` for bit-for-bit trace parity.
fn time_update_information(information: &[f64], n: usize, sigma: f64) -> Option<Vec<f64>> {
    let inv_q = 1.0 / (sigma * sigma);
    // M = Q⁻¹ + Λ₀₀ (3×3 top-left block of the information matrix).
    let mut m = [[0.0f64; 3]; 3];
    for (i, mi) in m.iter_mut().enumerate() {
        for (j, mij) in mi.iter_mut().enumerate() {
            *mij = information[i * n + j] + if i == j { inv_q } else { 0.0 };
        }
    }
    let m_inv = invert_3x3_adjugate(&m)?;
    // w = Λ[:,0..3] · M⁻¹ (n×3).
    let mut w = vec![[0.0f64; 3]; n];
    for (i, wi) in w.iter_mut().enumerate() {
        for (a, wia) in wi.iter_mut().enumerate() {
            let mut s = 0.0;
            for b in 0..3 {
                s += information[i * n + b] * m_inv[b][a];
            }
            *wia = s;
        }
    }
    // Λ' = Λ - w · Λ[0..3,:]  (Λ symmetric ⇒ Λ[0..3,:][a][j] = Λ[j][a]).
    let mut out = information.to_vec();
    for (i, wi) in w.iter().enumerate() {
        for j in 0..n {
            let mut corr = 0.0;
            for (a, &wia) in wi.iter().enumerate() {
                corr += wia * information[j * n + a];
            }
            out[i * n + j] -= corr;
        }
    }
    Some(out)
}

pub(super) fn propagate_baseline_mean(state: &mut FilterState, epoch: &Epoch, opts: &UpdateOpts) {
    if state.epoch_count == 0 || opts.dynamics_model != DynamicsModel::VelocityPropagated {
        return;
    }

    let Some(velocity) = epoch.velocity_mps else {
        return;
    };

    let dt = epoch.dt_s;
    if !dt.is_finite() || dt <= 0.0 || !velocity.iter().all(|v| v.is_finite()) {
        return;
    }

    for (k, v) in velocity.iter().enumerate() {
        state.baseline_m[k] += v * dt;
    }
}

/// Streaming RTK filter update for one epoch.
///
/// This is the crate-level `update(state, epoch) -> (state', solution)` entry
/// point for the kernel lane. It dynamically adds newly observed
/// single-difference ambiguity columns, seeds them from phase-code, runs the
/// iterated information update, searches unheld DD ambiguities with LAMBDA, and
/// persists accepted fixed ambiguities into the returned [`FilterState`].
pub fn update_epoch(
    state: FilterState,
    epoch: &Epoch,
    base: [f64; 3],
    model: &MeasModel,
    wavelengths: &BTreeMap<String, f64>,
    offsets: &BTreeMap<String, f64>,
    opts: &UpdateOpts,
) -> Result<EpochUpdate, UpdateError> {
    let mut scratch = RtkFilterScratch::default();
    let scale = AmbiguityScale {
        wavelengths_m: wavelengths,
        offsets_m: offsets,
    };
    update_epoch_with_scratch(state, epoch, base, model, scale, opts, &mut scratch)
}

/// Streaming RTK filter update using caller-owned reusable buffers.
pub fn update_epoch_with_scratch(
    mut state: FilterState,
    epoch: &Epoch,
    base: [f64; 3],
    model: &MeasModel,
    scale: AmbiguityScale,
    opts: &UpdateOpts,
    scratch: &mut RtkFilterScratch,
) -> Result<EpochUpdate, UpdateError> {
    state.validate_for_update().map_err(invalid_state)?;
    validate_update_opts(opts)?;
    let AmbiguityScale {
        wavelengths_m: wavelengths,
        offsets_m: offsets,
    } = scale;
    let ctx = MeasContext {
        base,
        model,
        antenna: opts.receiver_antenna_corrections.as_ref(),
    };
    let controls = IterateControls {
        hold_sigma_m: opts.hold_sigma_m,
        position_tol_m: opts.position_tol_m,
        ambiguity_tol_m: opts.ambiguity_tol_m,
        max_iterations: opts.max_iterations,
    };
    // Per-system reference guard: each epoch reference must match the state's
    // reference ambiguity arc for its constellation. A different arc (or an
    // untracked constellation) would reinterpret held integers.
    for r in &epoch.references {
        let system = satellite_system(&r.sat);
        match state.references.get(system) {
            None => return Err(UpdateError::UnknownReferenceSystem(system.to_string())),
            Some(expected) if expected != &r.sd_ambiguity_id => {
                return Err(UpdateError::ReferenceChanged {
                    system: system.to_string(),
                    expected: expected.clone(),
                    actual: r.sd_ambiguity_id.clone(),
                });
            }
            Some(_) => {}
        }
    }
    // Every non-reference satellite needs its own system's reference this epoch
    // (the Elixir per-system common invariant guarantees this upstream).
    for m in &epoch.nonref {
        // Type the per-system match through GnssSystem; the boundary error
        // string keeps the raw constellation letter via satellite_system.
        let system = system_of(&m.sat);
        if !epoch.references.iter().any(|r| system_of(&r.sat) == system) {
            return Err(UpdateError::MissingSystemReference(
                satellite_system(&m.sat).to_string(),
            ));
        }
    }

    // First epoch has no prior motion to propagate (mirrors the Elixir guard
    // `acc.epochs != []`). Do not infer this from ambiguity columns: the Sidereon
    // trace path pre-sizes columns to match Elixir's global ordering.
    let first_epoch = state.epoch_count == 0;

    for r in &epoch.references {
        state.ensure_ambiguity(&r.sd_ambiguity_id, phase_code_seed_m(r));
    }
    for meas in &epoch.nonref {
        state.ensure_ambiguity(&meas.sd_ambiguity_id, phase_code_seed_m(meas));
    }

    // Predict step: optionally advance the baseline mean from an external
    // velocity, then inflate baseline covariance before the measurement update.
    // On a singular round-trip, keep the un-inflated prior (matches the Elixir
    // fallback). If the inflated prior later makes the measurement solve singular
    // at an unlucky sigma, retry the same epoch with the un-inflated prior; this
    // is the same fallback point extended to the first operation that can prove
    // the predicted information unusable.
    propagate_baseline_mean(&mut state, epoch, opts);

    let mut uninflated_state = None;
    if !first_epoch && opts.process_noise_baseline_sigma_m > 0.0 {
        if let Some(updated) = time_update_information(
            &state.information,
            state.dim(),
            opts.process_noise_baseline_sigma_m,
        ) {
            uninflated_state = Some(state.clone());
            state.information = updated;
        }
    }

    let mut innovation_screen = prepare_innovation_screen(
        &state,
        epoch,
        base,
        model,
        opts,
        &mut scratch.screen_rows,
        &mut scratch.screen_mask,
    )?;
    if innovation_screen
        .as_ref()
        .is_some_and(|screen| screen.coasted)
    {
        return coasted_update(
            state,
            epoch,
            base,
            model,
            opts,
            innovation_screen,
            &mut scratch.residual_rows,
        );
    }
    let screen_mask = innovation_screen
        .as_ref()
        .map(|_| scratch.screen_mask.as_slice());

    let mut held = held_from_state_into(&state, &mut scratch.held)?;
    let mut posterior = iterate_epoch_into(
        ctx,
        &state,
        epoch,
        held,
        controls,
        screen_mask,
        &mut scratch.iterate,
    )?;

    if posterior.is_none() {
        if let Some(fallback_state) = uninflated_state {
            state = fallback_state;
            innovation_screen = prepare_innovation_screen(
                &state,
                epoch,
                base,
                model,
                opts,
                &mut scratch.screen_rows,
                &mut scratch.screen_mask,
            )?;
            if innovation_screen
                .as_ref()
                .is_some_and(|screen| screen.coasted)
            {
                return coasted_update(
                    state,
                    epoch,
                    base,
                    model,
                    opts,
                    innovation_screen,
                    &mut scratch.residual_rows,
                );
            }
            let screen_mask = innovation_screen
                .as_ref()
                .map(|_| scratch.screen_mask.as_slice());
            held = held_from_state_into(&state, &mut scratch.held)?;
            posterior = iterate_epoch_into(
                ctx,
                &state,
                epoch,
                held,
                controls,
                screen_mask,
                &mut scratch.iterate,
            )?;
        }
    }

    let float_only = float_only_set(&opts.float_only_systems);
    let has_targets = has_search_targets(epoch, held, &float_only);
    let prior_state_for_report = if has_targets {
        Some(state.clone())
    } else {
        None
    };

    {
        let posterior = posterior.ok_or(UpdateError::SingularGeometry)?;
        state.baseline_m = posterior.baseline_m;
        state
            .sd_ambiguities_m
            .copy_from_slice(posterior.sd_ambiguities_m);
        state.information.copy_from_slice(posterior.information);
    }

    if !has_targets {
        let fixed_ids: Vec<String> = state.fixed_cycles.keys().cloned().collect();
        let residuals = if opts.report_residuals {
            reported_epoch_residuals(
                ctx,
                &state,
                epoch,
                state.baseline_m,
                &state.sd_ambiguities_m,
                &mut scratch.residual_rows,
            )?
        } else {
            empty_filter_residuals()
        };
        state.epoch_count += 1;
        return Ok(EpochUpdate {
            reported_baseline_m: state.baseline_m,
            reported_sd_ambiguities_m: None,
            integer_ratio: 0.0,
            search: None,
            integer_fixed: !fixed_ids.is_empty(),
            newly_fixed: Vec::new(),
            fixed_ids,
            residuals,
            innovation_screen,
            state,
        });
    }

    let (updated_holds, search) = search_and_hold(
        &state,
        &state.information,
        epoch,
        scale,
        held,
        SearchPolicy {
            float_only_systems: &float_only,
            ar_arming_sigma_m: opts.ar_arming_sigma_m,
            ratio: opts.search,
        },
    )?;
    let integer_ratio = search
        .as_ref()
        .and_then(|meta| meta.integer_ratio)
        .unwrap_or(0.0);

    let previous_fixed: std::collections::BTreeSet<String> =
        state.fixed_cycles.keys().cloned().collect();
    let (fixed_cycles, fixed_m) = fixed_maps_from_holds(&updated_holds, wavelengths, offsets)?;
    let fixed_ids: Vec<String> = fixed_cycles.keys().cloned().collect();
    let newly_fixed: Vec<String> = fixed_ids
        .iter()
        .filter(|id| !previous_fixed.contains(*id))
        .cloned()
        .collect();

    state.fixed_cycles = fixed_cycles;
    state.fixed_m = fixed_m;
    state.epoch_count += 1;

    // Reported (fixed) baseline: when AR newly succeeds, re-solve THIS epoch with
    // the new integers held so the reported baseline reflects the fix in the same
    // epoch. Re-uses the same prior (`state`) as the float iterate, swapping the
    // held set for the post-fix one. Without this the first fixed epoch reports
    // its float baseline while claiming a fix (the cold-start false confidence).
    let (reported_baseline_m, reported_sd_ambiguities_m) = if newly_fixed.is_empty() {
        (state.baseline_m, None)
    } else {
        let prior_state = prior_state_for_report.expect("search targets clone prior state");
        let new_held = held_from_state(&state)?;
        let screen_mask = innovation_screen
            .as_ref()
            .map(|_| scratch.screen_mask.as_slice());
        let report_posterior = {
            #[cfg(test)]
            {
                if opts.force_report_iterate_failure {
                    Ok(None)
                } else {
                    iterate_epoch_into(
                        ctx,
                        &prior_state,
                        epoch,
                        &new_held,
                        controls,
                        screen_mask,
                        &mut scratch.report_iterate,
                    )
                }
            }
            #[cfg(not(test))]
            {
                iterate_epoch_into(
                    ctx,
                    &prior_state,
                    epoch,
                    &new_held,
                    controls,
                    screen_mask,
                    &mut scratch.report_iterate,
                )
            }
        };
        let conditioned = report_posterior?.ok_or(UpdateError::SingularGeometry)?;
        (
            conditioned.baseline_m,
            Some(conditioned.sd_ambiguities_m.to_vec()),
        )
    };
    let residuals = if opts.report_residuals {
        let residual_ambiguities_m = reported_sd_ambiguities_m
            .as_deref()
            .unwrap_or(&state.sd_ambiguities_m);
        reported_epoch_residuals(
            ctx,
            &state,
            epoch,
            reported_baseline_m,
            residual_ambiguities_m,
            &mut scratch.residual_rows,
        )?
    } else {
        empty_filter_residuals()
    };

    Ok(EpochUpdate {
        state,
        reported_baseline_m,
        reported_sd_ambiguities_m,
        integer_ratio,
        search,
        integer_fixed: !fixed_ids.is_empty(),
        newly_fixed,
        fixed_ids,
        residuals,
        innovation_screen,
    })
}
