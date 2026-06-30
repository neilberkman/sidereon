//! Sequential and static RTK baseline filtering, split into focused submodules.
//!
//! The kernel is composed from leaf submodules that the sidereon NIF, the crate
//! integration tests, and each other consume:
//! - `state` - the serializable [`FilterState`] ABI (baseline, single-difference
//!   float ambiguities, information matrix, held integers) carried epoch to epoch.
//! - `model` - the double-difference measurement-model primitives (geometric
//!   range, Sagnac, line-of-sight derivative, single-difference variance).
//! - `rows` - the reusable per-epoch double-difference row staging buffers.
//! - `normal` - the information-form measurement-fold and hold kernels.
//! - `search` - the LAMBDA integer ambiguity search and its result metadata.
//! - `antenna` - receiver-antenna PCO/PCV calibration and local-frame geometry.
//! - `float` - the static batch float-baseline solver.
//! - `fixed` - the static batch LAMBDA-fixed baseline solver.
//! - `update` - the streaming `update(state, epoch) -> (state', solution)`
//!   sequential filter: iterated information update, search-and-hold, predict
//!   step, and the public option/result/error types.
//!
//! [`rms`] is shared by the static float and fixed solvers and stays here.

mod antenna;
mod arc;
mod fixed;
mod float;
mod model;
mod moving_baseline;
mod normal;
mod rows;
mod search;
mod state;
mod update;

use std::collections::BTreeMap;

use crate::estimation::recipe::NormalRecipe;
use crate::estimation::substrate::normal::{BlockFoldScratch, NormalAssembler};
use crate::validate;

pub use crate::carrier_phase::CycleSlipOptions;
pub use crate::rtk::{
    CycleSlipPolicy, CycleSlipSplitArc, IonosphereFreeBaselineError, WideLaneError, WideLaneOptions,
};
#[cfg(test)]
use antenna::{pcv_m, receiver_antenna_correction};
pub use antenna::{ReceiverAntennaCalibration, ReceiverAntennaCorrections, ReceiverAntennaError};
pub use arc::{
    fix_wide_lane_rtk_arc, prepare_ionosphere_free_rtk_arc, solve_rtk_arc, solve_static_rtk_arc,
    solve_wide_lane_fixed_rtk_arc, RtkArcConfig, RtkArcEpoch, RtkArcEpochSolution, RtkArcError,
    RtkArcObservation, RtkArcPreprocessing, RtkArcSolution, RtkDualCycleSlipConfig,
    RtkDualFrequencyArcEpoch, RtkDualFrequencyObservation, RtkDualFrequencySatelliteObservation,
    RtkIonosphereFreeArcConfig, RtkIonosphereFreeArcError, RtkIonosphereFreeArcSolution,
    RtkStaticArcConfig, RtkStaticArcError, RtkStaticArcSolution, RtkWideLaneArcConfig,
    RtkWideLaneArcError, RtkWideLaneArcSolution, RtkWideLaneFixedArcConfig,
    RtkWideLaneFixedArcError, RtkWideLaneFixedArcIntegerMethod, RtkWideLaneFixedArcMetadata,
    RtkWideLaneFixedArcSolution, RtkWideLaneFixedArcSolveConfig,
    RtkWideLaneFixedSequentialArcSolution, RtkWideLaneFixedStaticArcSolution,
};
#[cfg(test)]
use fixed::fixed_epoch_rows;
pub(crate) use fixed::{baseline_ambiguity_index_core, run_fixed_validated};
pub use fixed::{
    solve_fixed_baseline, solve_fixed_baseline_validated, AmbiguitySet, FixedBaselineSolution,
    FixedSolveError, FixedSolveOpts, FloatPrior, ResidualComponentKind, ResidualValidationMeta,
    ResidualValidationOpts, ResidualValidationOutlier, ValidatedFixedBaselineSolution,
    ValidatedFixedSolveError, ValidatedFixedSolveOpts,
};
#[cfg(test)]
use float::float_epoch_rows;
pub(crate) use float::run_float;
pub use float::{
    solve_float_baseline, FloatBaselineSolution, FloatResidual, FloatSolveError, FloatSolveOpts,
    FloatSolveStatus,
};
use model::RowKind;
#[cfg(test)]
use model::{elevation_sin, range_derivative, range_m, single_difference_variance};
pub use model::{Epoch, MeasModel, SatMeas, StochasticModel};
pub use moving_baseline::{
    solve_moving_baseline, solve_moving_baseline_epoch, MovingBaselineEpoch,
    MovingBaselineEpochSolution, MovingBaselineError, MovingBaselineOpts,
    MovingBaselineSequenceError, MovingBaselineStatus,
};
use normal::fold_measurement_block_indices;
#[cfg(test)]
use normal::{
    double_difference_inverse_covariance, fold_measurement, fold_measurement_block, solve_normal,
};
#[cfg(test)]
use rows::DdRow;
use rows::{
    assign_double_difference_ambiguity_id, dd_epoch_rows_into, DdRowError, DdRowRecipe,
    DdRowScratch, EpochRowsScratch,
};
pub use search::{
    AmbiguitySearch, FullSetIntegerSummary, IntegerSearchMeta, IntegerStatus, PartialSearchMeta,
};
pub use state::{
    FilterState, FilterStateValidationError, FilterStateValidationKind, FILTER_STATE_VERSION,
};
use update::Hold;
#[cfg(test)]
use update::{
    dd_covariance_cycles, epoch_dd_rows, iterate_epoch, propagate_baseline_mean, search_and_hold,
    IterateControls, SearchPolicy,
};
pub use update::{
    update_epoch, update_epoch_with_scratch, DynamicsModel, EpochUpdate, InnovationScreen,
    InnovationScreenOpts, InvalidStateKind, RtkFilterScratch, SearchOpts, UpdateError, UpdateOpts,
};

/// Canonical RTK solver defaults.
///
/// These are the single source of truth for the per-binding RTK defaults that
/// previously drifted (Elixir, Python, WASM, and C each hardcoded their own).
/// A thin binding constructs a [`MeasModel`] / [`FloatSolveOpts`] /
/// [`FixedSolveOpts`] from these constants instead of carrying its own literals,
/// so every interface starts from the same numbers.
///
/// The values are the ones the core's own RTK tests use and the ones the
/// RTKLIB-demo5 reference fixes: `pos1-errphase = 0.003 m` carrier-phase sigma
/// and a code-to-phase error ratio of 100, giving the `0.3 m` code sigma. The
/// iteration cap is the value the majority of the core RTK goldens run with
/// (the Python/WASM bindings already used it); the Elixir binding's `8` is the
/// odd one out and should adopt this.
///
/// Exposing these does not change any solver behavior: the solvers still read
/// the values from the caller's config. These constants only give bindings one
/// place to read the canonical defaults from.
pub mod defaults {
    /// Canonical code (pseudorange) measurement sigma, metres.
    ///
    /// RTKLIB-demo5 reference value: `pos1-errphase (0.003 m) * code/phase
    /// error ratio (100) = 0.3 m`. Feeds [`super::MeasModel::code_sigma_m`].
    pub const CODE_SIGMA_M: f64 = 0.3;

    /// Canonical carrier-phase measurement sigma, metres.
    ///
    /// RTKLIB-demo5 reference value `pos1-errphase = 0.003 m`. Feeds
    /// [`super::MeasModel::phase_sigma_m`].
    pub const PHASE_SIGMA_M: f64 = 0.003;

    /// Canonical maximum iterations for the RTK float/fixed least-squares solve.
    ///
    /// Feeds [`super::FloatSolveOpts::max_iterations`] and
    /// [`super::FixedSolveOpts::max_iterations`]. This is the value the bulk of
    /// the core RTK goldens run with and the Python/WASM bindings already use.
    pub const MAX_ITERATIONS: usize = 10;

    /// Canonical baseline-step convergence tolerance, metres.
    ///
    /// The iterated double-difference solve stops once the L2 baseline step
    /// drops below this. Feeds [`super::FloatSolveOpts::position_tol_m`] and
    /// [`super::FixedSolveOpts::position_tol_m`]. This is the value the bindings
    /// hardcode and the core fixed/float goldens converge under (see the
    /// `rtk_filter` tests' `position_tol_m: 1.0e-4`).
    pub const POSITION_TOL_M: f64 = 1.0e-4;

    /// Canonical ambiguity-step convergence tolerance, metres.
    ///
    /// The iterated solve stops once the L2 single-difference ambiguity step
    /// drops below this. Feeds [`super::FloatSolveOpts::ambiguity_tol_m`] and
    /// [`super::FixedSolveOpts::ambiguity_tol_m`]. The bindings hardcode this
    /// value (core `rtk_filter` tests' `ambiguity_tol_m: 1.0e-4`).
    pub const AMBIGUITY_TOL_M: f64 = 1.0e-4;

    /// Canonical LAMBDA acceptance ratio threshold (dimensionless).
    ///
    /// A fixed integer candidate is accepted only when the second-best to best
    /// residual ratio meets or exceeds this. Feeds
    /// [`super::FixedSolveOpts::ratio_threshold`]. This is the long-standing
    /// RTKLIB-demo5 `pos2-arthres` default of 3.0, which the bindings hardcode
    /// and the core fixed-solve goldens use.
    pub const RATIO_THRESHOLD: f64 = 3.0;

    /// Canonical minimum ambiguity count for partial ambiguity resolution.
    ///
    /// When partial ambiguity resolution is enabled, the search retains at least
    /// this many integer ambiguities. Feeds
    /// [`super::FixedSolveOpts::partial_min_ambiguities`]. This is the value the
    /// bindings hardcode and the core partial-resolution goldens use
    /// (`partial_min_ambiguities: 4`).
    pub const PARTIAL_MIN_AMBIGUITIES: usize = 4;
}

/// Input-validation failure category for RTK row-builder entry points.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtkInputErrorKind {
    /// A floating-point input was NaN or infinite.
    NonFinite,
    /// A positive physical input was zero or negative.
    NotPositive,
    /// A non-negative physical input was negative.
    Negative,
    /// A finite numeric input was outside its accepted range.
    OutOfRange,
    /// A required input field was absent.
    Missing,
    /// A text field could not be parsed as a float.
    FloatParse,
    /// A text field could not be parsed as an integer.
    IntParse,
    /// A civil date field was out of range.
    InvalidCivilDate,
    /// A civil time field was out of range.
    InvalidCivilTime,
}

impl core::fmt::Display for RtkInputErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let label = match self {
            Self::NonFinite => "not finite",
            Self::NotPositive => "not positive",
            Self::Negative => "negative",
            Self::OutOfRange => "out of range",
            Self::Missing => "missing",
            Self::FloatParse => "invalid float",
            Self::IntParse => "invalid integer",
            Self::InvalidCivilDate => "invalid civil date",
            Self::InvalidCivilTime => "invalid civil time",
        };
        f.write_str(label)
    }
}

impl From<&validate::FieldError> for RtkInputErrorKind {
    fn from(error: &validate::FieldError) -> Self {
        match error {
            validate::FieldError::Missing { .. } => Self::Missing,
            validate::FieldError::NonFinite { .. } => Self::NonFinite,
            validate::FieldError::NotPositive { .. } => Self::NotPositive,
            validate::FieldError::Negative { .. } => Self::Negative,
            validate::FieldError::OutOfRange { .. } => Self::OutOfRange,
            validate::FieldError::FloatParse { .. } => Self::FloatParse,
            validate::FieldError::IntParse { .. } => Self::IntParse,
            validate::FieldError::InvalidCivilDate { .. } => Self::InvalidCivilDate,
            validate::FieldError::InvalidCivilTime { .. } => Self::InvalidCivilTime,
        }
    }
}

/// The RTK filter folds correlated double-difference blocks into a flat
/// information system and solves it first-tie; the static `float`/`fixed`
/// baseline solves and the sequential `update` filter all reduce through this
/// assembler ([`NormalRecipe::RtkDoubleDifferenceBlockFirstTie`]).
const RTK_ASSEMBLER: NormalAssembler =
    NormalAssembler::new(NormalRecipe::RtkDoubleDifferenceBlockFirstTie);

/// Per-solve measurement-model invariants shared by every double-difference row
/// builder (static `float`/`fixed` and the sequential `update` filter): the
/// base-station ECEF position, the sigma/stochastic/Sagnac measurement model,
/// and the optional receiver-antenna calibration. Carrying the trio as one
/// context keeps the row-builder and finalize argument lists small instead of
/// threading `base`/`model`/`receiver_antenna_corrections` through every call.
#[derive(Clone, Copy)]
pub(crate) struct MeasContext<'a> {
    base: [f64; 3],
    model: &'a MeasModel,
    antenna: Option<&'a ReceiverAntennaCorrections>,
}

impl<'a> MeasContext<'a> {
    /// Bundle the per-solve measurement environment so the runtime selector can
    /// hand the static runners one context argument instead of the base/model/
    /// antenna trio (keeping their parameter lists small).
    pub(crate) const fn new(
        base: [f64; 3],
        model: &'a MeasModel,
        antenna: Option<&'a ReceiverAntennaCorrections>,
    ) -> Self {
        Self {
            base,
            model,
            antenna,
        }
    }
}

/// Cycle-to-metre ambiguity scaling shared by every double-difference solver: the
/// per-ambiguity carrier wavelengths and the code-to-phase offsets (both metres).
/// Carried together so the static `fixed` reconstruction and the sequential
/// `update`/`search_and_hold` argument lists stay small instead of threading the
/// two maps separately.
#[derive(Clone, Copy)]
pub struct AmbiguityScale<'a> {
    pub wavelengths_m: &'a BTreeMap<String, f64>,
    pub offsets_m: &'a BTreeMap<String, f64>,
}

/// Root-mean-square of an iterator of values (`0.0` for an empty iterator).
/// Shared by the static batch float ([`float`]) and fixed ([`fixed`]) solvers
/// to summarise post-fit residual and weighted-residual rows.
fn rms(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value * value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64).sqrt()
    }
}

#[cfg(test)]
mod tests;
