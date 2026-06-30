//! Sequential RTK baseline-filter state value type: the serializable
//! [`FilterState`] a streaming processor owns epoch-to-epoch (baseline,
//! single-difference float ambiguities, information matrix, held integers) plus
//! the prior-seeded, dynamically-grown ambiguity-column bookkeeping. The
//! measurement-update numerics in the parent module operate on top of it.

use std::collections::BTreeMap;

use crate::estimation::substrate::parameters::ParameterLayout;
use crate::validate;

/// Schema version of the serialized filter state. Bump on any layout change so a
/// keyed stream processor can migrate persisted state.
///
/// v3: `reference_sat: String` became `references: BTreeMap<String, String>`
/// (per-system double-difference references, keyed by constellation letter).
pub const FILTER_STATE_VERSION: u16 = 3;

/// Sequential RTK baseline-filter state. The column order is the ABI: indices
/// 0..3 of the information matrix are baseline x/y/z, then one column per id in
/// `sd_ambiguity_ids`.
///
/// ABI forward-compatibility (decide-before-calcify; [`FILTER_STATE_VERSION`] is
/// the migration mechanism, and the update loop must keep these assumptions
/// LOCALIZED so they can be widened without a rewrite):
/// - **Multi-constellation:** DONE (v3). Double differences form within a GNSS
///   against that system's own reference; `references` carries one DD reference
///   per constellation letter. A single-entry map is the historical
///   single-system filter, bit-for-bit.
/// - **Kinematic (process noise):** baseline indices `0..3` may grow to `0..6`
///   (position+velocity). A time update with process noise is not expressible by
///   leaving the prior information unchanged - it runs in covariance (or
///   square-root) space (`Λ⁻¹ + Q`, re-invert), so the update loop must not
///   assume the carried information is propagated forward untouched.
#[derive(Debug, Clone, PartialEq)]
pub struct FilterState {
    /// Serialization schema version.
    pub version: u16,
    /// Per-system double-difference reference single-difference ambiguity ids,
    /// keyed by constellation letter (`"G" -> "G04"`). Each non-reference
    /// satellite differences against its own system's reference; there are no
    /// cross-system double differences.
    pub references: BTreeMap<String, String>,
    /// Single-difference ambiguity ids, in information-matrix column order.
    pub sd_ambiguity_ids: Vec<String>,
    /// Baseline estimate (metres, ECEF delta rover - base).
    pub baseline_m: [f64; 3],
    /// Float single-difference ambiguities (metres), parallel to `sd_ambiguity_ids`.
    pub sd_ambiguities_m: Vec<f64>,
    /// Row-major `n x n` information matrix, `n = 3 + sd_ambiguity_ids.len()`.
    pub information: Vec<f64>,
    /// Prior sigma (metres) applied to each new ambiguity column's diagonal.
    pub ambiguity_prior_sigma_m: f64,
    /// Number of epochs already incorporated into this state. This is the
    /// process-noise sentinel: pre-sized ambiguity columns mean "no ambiguity
    /// columns" no longer identifies the first epoch.
    pub epoch_count: usize,
    /// Held fixed integer double-difference ambiguities (id -> cycles).
    pub fixed_cycles: BTreeMap<String, i64>,
    /// Held fixed double-difference ambiguities (id -> metres). Redundant with
    /// `fixed_cycles` (metres = cycles·λ + offset); both are carried to mirror
    /// the Elixir filter state for 1:1 parity tracing. The metres form is the
    /// derived one - keep them consistent on every hold.
    pub fixed_m: BTreeMap<String, f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterStateValidationKind {
    Length { expected: usize, actual: usize },
    NonFinite,
    NotPositive,
    NotSymmetric,
    NotPositiveSemidefinite,
    DimensionOverflow,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FilterStateValidationError {
    pub field: &'static str,
    pub kind: FilterStateValidationKind,
}

impl core::fmt::Display for FilterStateValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let kind = match &self.kind {
            FilterStateValidationKind::Length { expected, actual } => {
                return write!(
                    f,
                    "invalid filter state {}: length is {actual}, expected {expected}",
                    self.field
                );
            }
            FilterStateValidationKind::NonFinite => "not finite",
            FilterStateValidationKind::NotPositive => "not positive",
            FilterStateValidationKind::NotSymmetric => "not symmetric",
            FilterStateValidationKind::NotPositiveSemidefinite => "not positive semidefinite",
            FilterStateValidationKind::DimensionOverflow => "dimension overflow",
        };
        write!(f, "invalid filter state {}: {kind}", self.field)
    }
}

impl std::error::Error for FilterStateValidationError {}

impl FilterState {
    /// New filter seeded with a baseline guess and diagonal priors (mirrors the
    /// Elixir `sequential_initial_information`). `references` maps each
    /// constellation letter to its reference single-difference ambiguity id.
    pub fn new(
        references: BTreeMap<String, String>,
        baseline_m: [f64; 3],
        baseline_prior_sigma_m: f64,
        ambiguity_prior_sigma_m: f64,
    ) -> Result<Self, FilterStateValidationError> {
        validate::finite_vec3(baseline_m, "state.baseline_m").map_err(map_state_field_error)?;
        let b = prior_information(baseline_prior_sigma_m, "state.baseline_prior_sigma_m")?;
        prior_information(ambiguity_prior_sigma_m, "state.ambiguity_prior_sigma_m")?;
        let information = vec![b, 0.0, 0.0, 0.0, b, 0.0, 0.0, 0.0, b];
        validate_information_matrix(&information, 3)?;
        Ok(Self {
            version: FILTER_STATE_VERSION,
            references,
            sd_ambiguity_ids: Vec::new(),
            baseline_m,
            sd_ambiguities_m: Vec::new(),
            information,
            ambiguity_prior_sigma_m,
            epoch_count: 0,
            fixed_cycles: BTreeMap::new(),
            fixed_m: BTreeMap::new(),
        })
    }

    /// State dimension `n = 3 + number of single-difference ambiguities`.
    pub(crate) fn dim(&self) -> usize {
        ParameterLayout::rtk(self.sd_ambiguity_ids.len()).dim()
    }

    /// Validate persisted-state shape and finite numeric fields before the
    /// update path indexes the row-major information matrix or parallel SD
    /// ambiguity arrays.
    pub(crate) fn validate_for_update(&self) -> Result<(), FilterStateValidationError> {
        let n = self.dim();
        let expected_information_len = n.checked_mul(n).ok_or(FilterStateValidationError {
            field: "state.information",
            kind: FilterStateValidationKind::DimensionOverflow,
        })?;
        validate::exact_len(
            &self.sd_ambiguities_m,
            self.sd_ambiguity_ids.len(),
            "state.sd_ambiguities_m",
        )
        .map_err(|error| FilterStateValidationError {
            field: error.field,
            kind: FilterStateValidationKind::Length {
                expected: error.expected,
                actual: error.actual,
            },
        })?;
        validate::exact_len(
            &self.information,
            expected_information_len,
            "state.information",
        )
        .map_err(|error| FilterStateValidationError {
            field: error.field,
            kind: FilterStateValidationKind::Length {
                expected: error.expected,
                actual: error.actual,
            },
        })?;
        validate::finite_vec3(self.baseline_m, "state.baseline_m")
            .map_err(map_state_field_error)?;
        validate::finite_slice(&self.sd_ambiguities_m, "state.sd_ambiguities_m")
            .map_err(map_state_field_error)?;
        validate::finite_slice(&self.information, "state.information")
            .map_err(map_state_field_error)?;
        if self.fixed_cycles.is_empty() && self.fixed_m.is_empty() {
            validate_information_matrix(&self.information, n)?;
        }
        for fixed_m in self.fixed_m.values().copied() {
            validate::finite(fixed_m, "state.fixed_m").map_err(map_state_field_error)?;
        }
        validate::finite_positive(
            self.ambiguity_prior_sigma_m,
            "state.ambiguity_prior_sigma_m",
        )
        .map_err(map_state_field_error)?;
        prior_information(
            self.ambiguity_prior_sigma_m,
            "state.ambiguity_prior_sigma_m",
        )?;
        Ok(())
    }

    /// Information-matrix column index of an ambiguity id (`3 + position`), if tracked.
    pub(crate) fn index_of(&self, id: &str) -> Option<usize> {
        self.sd_ambiguity_ids
            .iter()
            .position(|x| x == id)
            .map(|p| 3 + p)
    }

    /// Read `information[i][j]`. Test-only accessor for the information matrix.
    #[cfg(test)]
    pub(super) fn info(&self, i: usize, j: usize) -> f64 {
        let n = self.dim();
        self.information[i * n + j]
    }

    /// Add a single-difference ambiguity column if absent, seeded with `initial_m`
    /// and the prior (`1/σ²` on its new diagonal, zero cross-terms). No-op if the
    /// id is already tracked.
    pub(crate) fn ensure_ambiguity(&mut self, id: &str, initial_m: f64) {
        if self.index_of(id).is_some() {
            return;
        }
        let old_n = self.dim();
        let new_n = old_n + 1;
        let mut grown = vec![0.0f64; new_n * new_n];
        for i in 0..old_n {
            for j in 0..old_n {
                grown[i * new_n + j] = self.information[i * old_n + j];
            }
        }
        let prior = 1.0 / (self.ambiguity_prior_sigma_m * self.ambiguity_prior_sigma_m);
        grown[(new_n - 1) * new_n + (new_n - 1)] = prior;
        self.information = grown;
        self.sd_ambiguity_ids.push(id.to_string());
        self.sd_ambiguities_m.push(initial_m);
    }

    /// Position index (0..len) of an SD ambiguity within `sd_ambiguities_m`.
    pub(super) fn ambiguity_pos(&self, id: &str) -> Option<usize> {
        self.sd_ambiguity_ids.iter().position(|x| x == id)
    }

    /// State double-difference ambiguity (metres): SD(sat) - SD(ref).
    pub(super) fn dd_ambiguity_m(&self, sat_sd_id: &str, ref_sd_id: &str) -> Option<f64> {
        let s = self.sd_ambiguities_m[self.ambiguity_pos(sat_sd_id)?];
        let r = self.sd_ambiguities_m[self.ambiguity_pos(ref_sd_id)?];
        Some(s - r)
    }
}

fn prior_information(sigma_m: f64, field: &'static str) -> Result<f64, FilterStateValidationError> {
    validate::finite_positive(sigma_m, field).map_err(map_state_field_error)?;
    let information = 1.0 / (sigma_m * sigma_m);
    if !information.is_finite() {
        return Err(FilterStateValidationError {
            field,
            kind: FilterStateValidationKind::NonFinite,
        });
    }
    if information <= 0.0 {
        return Err(FilterStateValidationError {
            field,
            kind: FilterStateValidationKind::NotPositive,
        });
    }
    Ok(information)
}

fn validate_information_matrix(
    information: &[f64],
    n: usize,
) -> Result<(), FilterStateValidationError> {
    validate_information_symmetry(information, n)?;
    validate_information_psd(information, n)
}

fn validate_information_symmetry(
    information: &[f64],
    n: usize,
) -> Result<(), FilterStateValidationError> {
    let tol = information_tolerance(information);
    for i in 0..n {
        for j in (i + 1)..n {
            let a = information[i * n + j];
            let b = information[j * n + i];
            if (a - b).abs() > tol {
                return Err(FilterStateValidationError {
                    field: "state.information",
                    kind: FilterStateValidationKind::NotSymmetric,
                });
            }
        }
    }
    Ok(())
}

fn validate_information_psd(
    information: &[f64],
    n: usize,
) -> Result<(), FilterStateValidationError> {
    let tol = information_tolerance(information);
    let mut symmetric = vec![0.0f64; n * n];
    for i in 0..n {
        symmetric[i * n + i] = information[i * n + i];
        for j in (i + 1)..n {
            let value = 0.5 * (information[i * n + j] + information[j * n + i]);
            symmetric[i * n + j] = value;
            symmetric[j * n + i] = value;
        }
    }
    if symmetric_min_eigenvalue(&mut symmetric, n, tol) < -tol {
        return Err(FilterStateValidationError {
            field: "state.information",
            kind: FilterStateValidationKind::NotPositiveSemidefinite,
        });
    }
    Ok(())
}

fn symmetric_min_eigenvalue(matrix: &mut [f64], n: usize, tol: f64) -> f64 {
    let max_sweeps = (16 * n * n).max(32);
    for _ in 0..max_sweeps {
        let mut p = 0usize;
        let mut q = 0usize;
        let mut max_offdiag = 0.0_f64;
        for i in 0..n {
            for j in (i + 1)..n {
                let offdiag = matrix[i * n + j].abs();
                if offdiag > max_offdiag {
                    max_offdiag = offdiag;
                    p = i;
                    q = j;
                }
            }
        }

        if max_offdiag <= tol {
            break;
        }

        let app = matrix[p * n + p];
        let aqq = matrix[q * n + q];
        let apq = matrix[p * n + q];
        if apq == 0.0 {
            break;
        }

        let tau = (aqq - app) / (2.0 * apq);
        let t = if tau >= 0.0 {
            1.0 / (tau + (1.0 + tau * tau).sqrt())
        } else {
            -1.0 / (-tau + (1.0 + tau * tau).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;

        for k in 0..n {
            if k != p && k != q {
                let akp = matrix[k * n + p];
                let akq = matrix[k * n + q];
                let new_kp = c * akp - s * akq;
                let new_kq = s * akp + c * akq;
                matrix[k * n + p] = new_kp;
                matrix[p * n + k] = new_kp;
                matrix[k * n + q] = new_kq;
                matrix[q * n + k] = new_kq;
            }
        }

        matrix[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        matrix[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        matrix[p * n + q] = 0.0;
        matrix[q * n + p] = 0.0;
    }

    let mut min = f64::INFINITY;
    for i in 0..n {
        min = min.min(matrix[i * n + i]);
    }
    min
}

fn information_tolerance(information: &[f64]) -> f64 {
    let scale = information
        .iter()
        .fold(1.0_f64, |acc, &value| acc.max(value.abs()));
    (scale.max(1.0) * 1.0e-6).max(0.25)
}

fn map_state_field_error(error: validate::FieldError) -> FilterStateValidationError {
    let kind = match error {
        validate::FieldError::NonFinite { .. } => FilterStateValidationKind::NonFinite,
        validate::FieldError::NotPositive { .. } => FilterStateValidationKind::NotPositive,
        validate::FieldError::Missing { .. }
        | validate::FieldError::Negative { .. }
        | validate::FieldError::OutOfRange { .. }
        | validate::FieldError::FloatParse { .. }
        | validate::FieldError::IntParse { .. }
        | validate::FieldError::InvalidCivilDate { .. }
        | validate::FieldError::InvalidCivilTime { .. } => {
            unreachable!("filter-state validation only uses finite/positive numeric validators")
        }
    };
    FilterStateValidationError {
        field: error.field(),
        kind,
    }
}
