//! Shared LAMBDA integer ambiguity-search cluster for the RTK baseline filter.
//!
//! Both the static batch fixed-baseline solver (parent module) and the
//! sequential filter's `search_and_hold` consume this layer: it turns a float
//! ambiguity covariance (metres) into cycles, runs the LAMBDA integer
//! least-squares kernel (`crate::ils`), and builds the public
//! [`IntegerSearchMeta`] diagnostics, including the partial ambiguity-resolution
//! ranking and exhaustive subset fallback. The arithmetic and operation order
//! are unchanged from the original in-module implementation, so the frozen-bits
//! goldens and real-arc LAMBDA ratios are identical.

use std::collections::BTreeMap;

use super::model::is_float_only_system;
use super::{FixedSolveError, FixedSolveOpts};
use crate::id::GnssSystem;

/// Integer ambiguity-fix verdict for a static fixed RTK solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegerStatus {
    Fixed,
    NotFixed,
}

/// Diagnostic payload for one integer ambiguity search.
#[derive(Debug, Clone, PartialEq)]
pub struct AmbiguitySearch {
    pub order: Vec<String>,
    pub float_cycles: Vec<(String, f64)>,
    pub covariance_cycles: Vec<f64>,
    pub covariance_inverse_cycles: Vec<f64>,
}

/// Summary of a rejected full-set search when partial AR is enabled.
#[derive(Debug, Clone, PartialEq)]
pub struct FullSetIntegerSummary {
    pub integer_status: IntegerStatus,
    pub integer_ratio: Option<f64>,
    pub integer_best_score: Option<f64>,
    pub integer_second_best_score: Option<f64>,
    pub integer_candidates: usize,
    pub order: Vec<String>,
}

/// Partial ambiguity-resolution diagnostics for a static fixed RTK solve.
#[derive(Debug, Clone, PartialEq)]
pub struct PartialSearchMeta {
    pub enabled: bool,
    pub fixed: bool,
    pub fixed_ambiguities: Vec<String>,
    pub free_ambiguities: Vec<String>,
    pub full_set: Option<FullSetIntegerSummary>,
    pub exhaustive_subsets_evaluated: Option<usize>,
}

/// Integer-search metadata for a static fixed RTK solve.
#[derive(Debug, Clone, PartialEq)]
pub struct IntegerSearchMeta {
    pub integer_status: IntegerStatus,
    pub integer_method: &'static str,
    pub integer_ratio: Option<f64>,
    pub integer_best_score: Option<f64>,
    pub integer_second_best_score: Option<f64>,
    pub integer_candidates: usize,
    pub ambiguity_search: AmbiguitySearch,
    pub ambiguity_offsets_m: Vec<(String, f64)>,
    pub partial: PartialSearchMeta,
}

#[derive(Debug, Clone)]
pub(super) struct FixedSearchResult {
    pub(super) fixed_cycles: BTreeMap<String, i64>,
    pub(super) meta: IntegerSearchMeta,
}

pub(super) fn empty_integer_search_meta(offsets_m: &BTreeMap<String, f64>) -> IntegerSearchMeta {
    IntegerSearchMeta {
        integer_status: IntegerStatus::NotFixed,
        integer_method: "lambda",
        integer_ratio: None,
        integer_best_score: None,
        integer_second_best_score: None,
        integer_candidates: 0,
        ambiguity_search: AmbiguitySearch {
            order: Vec::new(),
            float_cycles: Vec::new(),
            covariance_cycles: Vec::new(),
            covariance_inverse_cycles: Vec::new(),
        },
        ambiguity_offsets_m: offsets_m.iter().map(|(k, &v)| (k.clone(), v)).collect(),
        partial: partial_meta(false, false, &BTreeMap::new(), &[], None, None),
    }
}

pub(super) fn float_only_ambiguity_ids(
    ambiguity_satellites: &BTreeMap<String, String>,
    float_only: &[GnssSystem],
) -> std::collections::BTreeSet<String> {
    ambiguity_satellites
        .iter()
        .filter(|(_, sat)| is_float_only_system(sat, float_only))
        .map(|(id, _)| id.clone())
        .collect()
}

pub(super) fn covariance_m_to_cycles(
    covariance_m: &[f64],
    ambiguity_ids: &[String],
    wavelengths_m: &BTreeMap<String, f64>,
) -> Result<Vec<f64>, FixedSolveError> {
    let n = ambiguity_ids.len();
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        let lambda_i = *wavelengths_m
            .get(&ambiguity_ids[i])
            .ok_or_else(|| FixedSolveError::MissingWavelength(ambiguity_ids[i].clone()))?;
        for j in 0..n {
            let lambda_j = *wavelengths_m
                .get(&ambiguity_ids[j])
                .ok_or_else(|| FixedSolveError::MissingWavelength(ambiguity_ids[j].clone()))?;
            out[i * n + j] = covariance_m[i * n + j] / (lambda_i * lambda_j);
        }
    }
    Ok(out)
}

pub(super) fn covariance_submatrix(
    all_ids: &[String],
    covariance: &[f64],
    subset_ids: &[String],
) -> Vec<f64> {
    let n = all_ids.len();
    let indices = subset_ids
        .iter()
        .map(|id| {
            all_ids
                .iter()
                .position(|candidate| candidate == id)
                .expect("known id")
        })
        .collect::<Vec<_>>();
    let m = indices.len();
    let mut out = vec![0.0; m * m];
    for (ii, &i) in indices.iter().enumerate() {
        for (jj, &j) in indices.iter().enumerate() {
            out[ii * m + jj] = covariance[i * n + j];
        }
    }
    out
}

pub(super) fn float_cycles_for_ids(
    ids: &[String],
    float_ambiguities_m: &BTreeMap<String, f64>,
    wavelengths_m: &BTreeMap<String, f64>,
    offsets_m: &BTreeMap<String, f64>,
) -> Result<Vec<(String, f64)>, FixedSolveError> {
    ids.iter()
        .map(|id| {
            let ambiguity_m = *float_ambiguities_m
                .get(id)
                .ok_or_else(|| FixedSolveError::MissingAmbiguity(id.clone()))?;
            let offset_m = *offsets_m
                .get(id)
                .ok_or_else(|| FixedSolveError::MissingOffset(id.clone()))?;
            let wavelength_m = *wavelengths_m
                .get(id)
                .ok_or_else(|| FixedSolveError::MissingWavelength(id.clone()))?;
            Ok((id.clone(), (ambiguity_m - offset_m) / wavelength_m))
        })
        .collect()
}

pub(super) fn search_ambiguity_ids(
    ids: &[String],
    float_cycles: &[(String, f64)],
    covariance_cycles: &[f64],
    opts: FixedSolveOpts,
) -> Result<FixedSearchResult, FixedSolveError> {
    let n = ids.len();
    let floats = ids
        .iter()
        .map(|id| {
            float_cycles
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, v)| *v)
                .expect("known float cycle")
        })
        .collect::<Vec<_>>();
    let covariance_rows = matrix_rows_from_flat(covariance_cycles, n);
    let result = crate::estimation::substrate::ambiguity::resolve_integer_lattice(
        &floats,
        &covariance_rows,
        opts.ratio_threshold,
    )
    .map_err(FixedSolveError::Ils)?;
    Ok(search_result_from_ils(ids, float_cycles, result))
}

pub(super) fn search_result_from_ils(
    ids: &[String],
    float_cycles: &[(String, f64)],
    result: crate::ils::IlsResult,
) -> FixedSearchResult {
    let fixed_cycles = ids
        .iter()
        .cloned()
        .zip(result.fixed.iter().copied())
        .collect::<BTreeMap<_, _>>();
    let status = if result.fixed_status {
        IntegerStatus::Fixed
    } else {
        IntegerStatus::NotFixed
    };
    FixedSearchResult {
        fixed_cycles: fixed_cycles.clone(),
        meta: IntegerSearchMeta {
            integer_status: status,
            integer_method: "lambda",
            integer_ratio: Some(result.ratio),
            integer_best_score: Some(result.best_score),
            integer_second_best_score: result.second_best_score,
            integer_candidates: result.candidates_evaluated,
            ambiguity_search: AmbiguitySearch {
                order: ids.to_vec(),
                float_cycles: float_cycles.to_vec(),
                covariance_cycles: flatten_matrix(&result.covariance),
                covariance_inverse_cycles: flatten_matrix(&result.covariance_inverse),
            },
            ambiguity_offsets_m: Vec::new(),
            partial: partial_meta(false, false, &fixed_cycles, &[], None, None),
        },
    }
}

/// Shared inputs for the partial ambiguity-resolution search recursion: the full
/// ambiguity id list, the float cycles, the cycle covariance, the metre offsets,
/// and the solve options. Carried as one context so the partial/exhaustive/subset
/// search helpers keep small argument lists instead of threading the same five
/// values through every call.
#[derive(Clone, Copy)]
pub(super) struct PartialSearchInputs<'a> {
    pub(super) all_ids: &'a [String],
    pub(super) float_cycles: &'a [(String, f64)],
    pub(super) covariance_cycles: &'a [f64],
    pub(super) offsets_m: &'a BTreeMap<String, f64>,
    pub(super) opts: FixedSolveOpts,
}

pub(super) fn search_partial_fixed_ambiguities(
    inputs: PartialSearchInputs,
    full_fixed_cycles: BTreeMap<String, i64>,
    full_meta: IntegerSearchMeta,
) -> Result<FixedSearchResult, FixedSolveError> {
    let ranked_ids = ambiguity_ids_ranked_by_integer_confidence(
        inputs.all_ids,
        inputs.float_cycles,
        inputs.covariance_cycles,
    );
    let min_size = inputs
        .opts
        .partial_min_ambiguities
        .min(inputs.all_ids.len() - 1);
    let full_summary = Some(full_set_integer_summary(&full_meta));

    if min_size < 1 {
        let mut meta = full_meta;
        meta.ambiguity_offsets_m = inputs
            .offsets_m
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        meta.partial = partial_meta(true, false, &full_fixed_cycles, &[], full_summary, None);
        return Ok(FixedSearchResult {
            fixed_cycles: full_fixed_cycles,
            meta,
        });
    }

    for subset_size in (min_size..inputs.all_ids.len()).rev() {
        let mut subset_ids = ranked_ids
            .iter()
            .take(subset_size)
            .cloned()
            .collect::<Vec<_>>();
        subset_ids.sort();
        let result = search_ambiguity_subset(inputs, &subset_ids, &full_meta)?;
        if result.meta.integer_status == IntegerStatus::Fixed {
            return Ok(result);
        }
    }

    search_partial_fixed_ambiguities_exhaustive(inputs, full_fixed_cycles, full_meta, min_size)
}

const PARTIAL_EXHAUSTIVE_MAX_AMBIGUITIES: usize = 14;
const PARTIAL_EXHAUSTIVE_MAX_SUBSETS: usize = 20_000;

fn search_partial_fixed_ambiguities_exhaustive(
    inputs: PartialSearchInputs,
    full_fixed_cycles: BTreeMap<String, i64>,
    full_meta: IntegerSearchMeta,
    min_size: usize,
) -> Result<FixedSearchResult, FixedSolveError> {
    let not_fixed = |evaluated: usize| {
        let mut meta = full_meta.clone();
        meta.ambiguity_offsets_m = inputs
            .offsets_m
            .iter()
            .map(|(k, &v)| (k.clone(), v))
            .collect();
        meta.partial = partial_meta(
            true,
            false,
            &full_fixed_cycles,
            &[],
            Some(full_set_integer_summary(&full_meta)),
            Some(evaluated),
        );
        FixedSearchResult {
            fixed_cycles: full_fixed_cycles.clone(),
            meta,
        }
    };

    if inputs.all_ids.len() > PARTIAL_EXHAUSTIVE_MAX_AMBIGUITIES {
        return Ok(not_fixed(0));
    }

    let mut evaluated = 0usize;
    for subset_size in (min_size..inputs.all_ids.len()).rev() {
        let combos = ambiguity_subset_combinations(inputs.all_ids, subset_size);
        if evaluated + combos.len() > PARTIAL_EXHAUSTIVE_MAX_SUBSETS {
            return Ok(not_fixed(evaluated));
        }

        let mut best: Option<(f64, FixedSearchResult)> = None;
        for subset_ids in combos.iter() {
            let mut subset_ids = subset_ids.clone();
            subset_ids.sort();
            let result = search_ambiguity_subset(inputs, &subset_ids, &full_meta)?;
            if result.meta.integer_status == IntegerStatus::Fixed {
                let ratio = result.meta.integer_ratio.unwrap_or(0.0);
                match &best {
                    Some((best_ratio, _)) if *best_ratio >= ratio => {}
                    _ => best = Some((ratio, result)),
                }
            }
        }
        evaluated += combos.len();

        if let Some((_ratio, mut result)) = best {
            result.meta.partial.exhaustive_subsets_evaluated = Some(evaluated);
            return Ok(result);
        }
    }

    Ok(not_fixed(evaluated))
}

fn search_ambiguity_subset(
    inputs: PartialSearchInputs,
    subset_ids: &[String],
    full_meta: &IntegerSearchMeta,
) -> Result<FixedSearchResult, FixedSolveError> {
    let subset_cycles = subset_ids
        .iter()
        .map(|id| {
            let value = inputs
                .float_cycles
                .iter()
                .find(|(candidate, _)| candidate == id)
                .map(|(_, v)| *v)
                .expect("known subset id");
            (id.clone(), value)
        })
        .collect::<Vec<_>>();
    let subset_covariance =
        covariance_submatrix(inputs.all_ids, inputs.covariance_cycles, subset_ids);
    let mut result =
        search_ambiguity_ids(subset_ids, &subset_cycles, &subset_covariance, inputs.opts)?;
    let free_ids = inputs
        .all_ids
        .iter()
        .filter(|id| !subset_ids.contains(id))
        .cloned()
        .collect::<Vec<_>>();
    result.meta.ambiguity_offsets_m = subset_ids
        .iter()
        .map(|id| (id.clone(), inputs.offsets_m[id]))
        .collect();
    result.meta.partial = partial_meta(
        true,
        result.meta.integer_status == IntegerStatus::Fixed,
        &result.fixed_cycles,
        &free_ids,
        Some(full_set_integer_summary(full_meta)),
        None,
    );
    Ok(result)
}

fn ambiguity_subset_combinations(ids: &[String], size: usize) -> Vec<Vec<String>> {
    if size == 0 {
        return vec![Vec::new()];
    }
    if ids.is_empty() {
        return Vec::new();
    }

    let head = ids[0].clone();
    let tail = &ids[1..];
    let mut with_head = ambiguity_subset_combinations(tail, size - 1)
        .into_iter()
        .map(|mut combo| {
            combo.insert(0, head.clone());
            combo
        })
        .collect::<Vec<_>>();
    with_head.extend(ambiguity_subset_combinations(tail, size));
    with_head
}

fn ambiguity_ids_ranked_by_integer_confidence(
    ambiguity_ids: &[String],
    float_cycles: &[(String, f64)],
    covariance_cycles: &[f64],
) -> Vec<String> {
    let n = ambiguity_ids.len();
    let float_map = float_cycles.iter().cloned().collect::<BTreeMap<_, _>>();
    let mut ranked = ambiguity_ids
        .iter()
        .enumerate()
        .map(|(idx, id)| {
            let variance = covariance_cycles[idx * n + idx];
            let value = float_map[id];
            let distance_to_integer = (value - value.round()).abs();
            let sigma = variance.max(0.0).sqrt();
            let normalized_margin = (0.5 - distance_to_integer) / sigma;
            (id.clone(), -normalized_margin, variance)
        })
        .collect::<Vec<_>>();
    ranked.sort_by(
        |(id_a, margin_a, variance_a), (id_b, margin_b, variance_b)| {
            margin_a
                .partial_cmp(margin_b)
                .unwrap_or(core::cmp::Ordering::Equal)
                .then_with(|| {
                    variance_a
                        .partial_cmp(variance_b)
                        .unwrap_or(core::cmp::Ordering::Equal)
                })
                .then_with(|| id_a.cmp(id_b))
        },
    );
    ranked.into_iter().map(|(id, _, _)| id).collect()
}

fn full_set_integer_summary(meta: &IntegerSearchMeta) -> FullSetIntegerSummary {
    FullSetIntegerSummary {
        integer_status: meta.integer_status,
        integer_ratio: meta.integer_ratio,
        integer_best_score: meta.integer_best_score,
        integer_second_best_score: meta.integer_second_best_score,
        integer_candidates: meta.integer_candidates,
        order: meta.ambiguity_search.order.clone(),
    }
}

pub(super) fn partial_meta(
    enabled: bool,
    fixed: bool,
    fixed_cycles: &BTreeMap<String, i64>,
    free_ambiguities: &[String],
    full_set: Option<FullSetIntegerSummary>,
    exhaustive_subsets_evaluated: Option<usize>,
) -> PartialSearchMeta {
    PartialSearchMeta {
        enabled,
        fixed,
        fixed_ambiguities: fixed_cycles.keys().cloned().collect(),
        free_ambiguities: {
            let mut ids = free_ambiguities.to_vec();
            ids.sort();
            ids
        },
        full_set,
        exhaustive_subsets_evaluated,
    }
}

fn matrix_rows_from_flat(values: &[f64], n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| values[i * n..(i + 1) * n].to_vec())
        .collect()
}

fn flatten_matrix(values: &[Vec<f64>]) -> Vec<f64> {
    values.iter().flat_map(|row| row.iter().copied()).collect()
}
