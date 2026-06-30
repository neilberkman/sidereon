//! Measurement-quality control for GNSS positioning.
//!
//! This module owns the language-independent RAIM/FDE decision logic and the
//! standard pseudorange weighting primitives used by Sidereon' QC surface.

use std::collections::{BTreeMap, BTreeSet};

use crate::astro::math::linear::{invert_symmetric_pd, normal_equations_weighted};
use crate::constants::DEG_TO_RAD;
use crate::spp::{solve, EphemerisSource, Observation, ReceiverSolution, SolveInputs, SppError};
use crate::validate;

/// Default zenith-floor term for pseudorange variance, meters.
pub const DEFAULT_VARIANCE_A_M: f64 = 0.3;
/// Default elevation-scaled term for pseudorange variance, meters.
pub const DEFAULT_VARIANCE_B_M: f64 = 0.3;
/// Default false-alarm probability for RAIM.
pub const DEFAULT_P_FA: f64 = 1.0e-3;

/// Pseudorange variance model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PseudorangeVarianceModel {
    /// Elevation-only `a^2 + b^2 / sin(el)^2`.
    Elevation,
    /// Elevation plus a C/N0 variance contribution.
    ElevationCn0,
}

/// Options for [`pseudorange_variance`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PseudorangeVarianceOptions {
    /// Zenith-floor term, meters.
    pub a_m: f64,
    /// Elevation-scaled term, meters.
    pub b_m: f64,
    /// Selected variance model.
    pub model: PseudorangeVarianceModel,
    /// Carrier-to-noise density, dB-Hz, required by
    /// [`PseudorangeVarianceModel::ElevationCn0`].
    pub cn0_dbhz: Option<f64>,
    /// C/N0 variance scale, square meters.
    pub cn0_scale_m2: f64,
}

impl Default for PseudorangeVarianceOptions {
    fn default() -> Self {
        Self {
            a_m: DEFAULT_VARIANCE_A_M,
            b_m: DEFAULT_VARIANCE_B_M,
            model: PseudorangeVarianceModel::Elevation,
            cn0_dbhz: None,
            cn0_scale_m2: 1.0,
        }
    }
}

impl PseudorangeVarianceOptions {
    fn with_entry_cn0(self, cn0_dbhz: f64) -> Self {
        Self {
            model: PseudorangeVarianceModel::ElevationCn0,
            cn0_dbhz: Some(cn0_dbhz),
            ..self
        }
    }
}

/// One satellite/elevation entry used to build sigma or weight maps.
#[derive(Debug, Clone, PartialEq)]
pub struct WeightEntry {
    /// Satellite token at the binding boundary, e.g. `"G01"`.
    pub satellite_id: String,
    /// Topocentric elevation, degrees.
    pub elevation_deg: f64,
    /// Optional C/N0 for this observation. When present, it selects the C/N0
    /// model for this entry.
    pub cn0_dbhz: Option<f64>,
}

/// Error from quality-control primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityError {
    /// Elevation must be finite, inside `[-90, 90]`, and yield finite variance.
    InvalidElevation,
    /// The C/N0 model was selected without a C/N0 value.
    MissingCn0,
    /// Variance-model parameters must be finite and non-negative.
    InvalidParameter,
    /// Probability must be strictly inside `(0, 1)`.
    InvalidProbability,
    /// RAIM system-count override must be positive.
    InvalidSystemCount,
    /// Chi-square degrees of freedom must be positive.
    InvalidDof,
    /// RAIM weights must be positive finite values.
    InvalidWeight,
    /// RAIM residuals must be finite and aligned with used satellites.
    InvalidResiduals,
    /// A linearized measurement set was empty, ragged, non-finite, or carried
    /// fewer measurements than estimated state parameters.
    InvalidDesign,
    /// The weighted normal matrix `H^T W H` was singular or rank deficient, so
    /// no protected state correction exists.
    SingularGeometry,
}

impl core::fmt::Display for QualityError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidElevation => write!(f, "invalid elevation"),
            Self::MissingCn0 => write!(f, "missing C/N0"),
            Self::InvalidParameter => write!(f, "invalid quality parameter"),
            Self::InvalidProbability => write!(f, "invalid probability"),
            Self::InvalidSystemCount => write!(f, "invalid RAIM system count"),
            Self::InvalidDof => write!(f, "invalid degrees of freedom"),
            Self::InvalidWeight => write!(f, "invalid RAIM weight"),
            Self::InvalidResiduals => write!(f, "invalid RAIM residuals"),
            Self::InvalidDesign => write!(f, "invalid linearized measurement design"),
            Self::SingularGeometry => write!(f, "singular or rank-deficient geometry"),
        }
    }
}

impl std::error::Error for QualityError {}

/// Pseudorange measurement variance, square meters.
pub fn pseudorange_variance(
    elevation_deg: f64,
    options: PseudorangeVarianceOptions,
) -> Result<f64, QualityError> {
    validate_elevation_deg(elevation_deg)?;
    validate_variance_options(options)?;

    let mut elevation_var = options.a_m * options.a_m;
    if options.b_m != 0.0 {
        let sin_el = (elevation_deg * DEG_TO_RAD).sin();
        let scaled = options.b_m * options.b_m / (sin_el * sin_el);
        if !scaled.is_finite() {
            return Err(QualityError::InvalidElevation);
        }
        elevation_var += scaled;
    }

    let variance = match options.model {
        PseudorangeVarianceModel::Elevation => elevation_var,
        PseudorangeVarianceModel::ElevationCn0 => {
            let Some(cn0) = options.cn0_dbhz else {
                return Err(QualityError::MissingCn0);
            };
            validate_nonneg_parameter(cn0, "cn0_dbhz")?;
            elevation_var + options.cn0_scale_m2 * 10.0_f64.powf(-cn0 / 10.0)
        }
    };

    validate_positive_variance(variance)?;
    Ok(variance)
}

fn validate_elevation_deg(elevation_deg: f64) -> Result<(), QualityError> {
    validate::finite(elevation_deg, "elevation_deg").map_err(|_| QualityError::InvalidElevation)?;
    if (-90.0..=90.0).contains(&elevation_deg) {
        Ok(())
    } else {
        Err(QualityError::InvalidElevation)
    }
}

fn validate_variance_options(options: PseudorangeVarianceOptions) -> Result<(), QualityError> {
    validate_nonneg_parameter(options.a_m, "variance a_m")?;
    validate_nonneg_parameter(options.b_m, "variance b_m")?;
    validate_nonneg_parameter(options.cn0_scale_m2, "variance cn0_scale_m2")
}

fn validate_nonneg_parameter(value: f64, field: &'static str) -> Result<(), QualityError> {
    validate::finite_nonneg(value, field)
        .map(|_| ())
        .map_err(map_parameter_error)
}

fn validate_positive_variance(value: f64) -> Result<(), QualityError> {
    validate::finite_positive(value, "pseudorange variance")
        .map(|_| ())
        .map_err(map_parameter_error)
}

fn map_parameter_error(_error: validate::FieldError) -> QualityError {
    QualityError::InvalidParameter
}

/// Build a satellite-to-sigma map. Entries whose variance cannot be computed are
/// dropped, matching the Sidereon public API.
pub fn sigmas(
    entries: &[WeightEntry],
    options: PseudorangeVarianceOptions,
) -> BTreeMap<String, f64> {
    entries
        .iter()
        .filter_map(|entry| {
            let opts = match entry.cn0_dbhz {
                Some(cn0) => options.with_entry_cn0(cn0),
                None => options,
            };
            pseudorange_variance(entry.elevation_deg, opts)
                .ok()
                .map(|var| (entry.satellite_id.clone(), var.sqrt()))
        })
        .collect()
}

/// Build a satellite-to-inverse-variance-weight map. Entries whose variance
/// cannot be computed are dropped, matching the Sidereon public API.
pub fn weight_vector(
    entries: &[WeightEntry],
    options: PseudorangeVarianceOptions,
) -> BTreeMap<String, f64> {
    entries
        .iter()
        .filter_map(|entry| {
            let opts = match entry.cn0_dbhz {
                Some(cn0) => options.with_entry_cn0(cn0),
                None => options,
            };
            pseudorange_variance(entry.elevation_deg, opts)
                .ok()
                .map(|var| (entry.satellite_id.clone(), 1.0 / var))
        })
        .collect()
}

/// RAIM weighting mode.
#[derive(Debug, Clone, PartialEq)]
pub enum RaimWeights {
    /// Unit weights, equivalent to sigma = 1 m for every satellite.
    Unit,
    /// Per-satellite inverse variance weights. Missing satellites default to
    /// unit weight.
    BySatellite(BTreeMap<String, f64>),
}

impl RaimWeights {
    fn validate(&self) -> Result<(), QualityError> {
        match self {
            Self::Unit => Ok(()),
            Self::BySatellite(weights) => weights
                .values()
                .try_for_each(|w| validate::finite_positive(*w, "raim weight").map(|_| ()))
                .map_err(|_| QualityError::InvalidWeight),
        }
    }

    fn weight_for(&self, satellite_id: &str) -> f64 {
        match self {
            Self::Unit => 1.0,
            Self::BySatellite(weights) => weights.get(satellite_id).copied().unwrap_or(1.0),
        }
    }
}

/// Options for [`raim`].
#[derive(Debug, Clone, PartialEq)]
pub struct RaimOptions {
    /// False-alarm probability.
    pub p_fa: f64,
    /// RAIM residual weights.
    pub weights: RaimWeights,
    /// Optional override for the number of distinct GNSS clock systems.
    pub n_systems: Option<isize>,
}

impl Default for RaimOptions {
    fn default() -> Self {
        Self {
            p_fa: DEFAULT_P_FA,
            weights: RaimWeights::Unit,
            n_systems: None,
        }
    }
}

/// Minimal solution view needed by RAIM.
#[derive(Debug, Clone, PartialEq)]
pub struct RaimInput {
    /// Used satellite tokens, in residual order.
    pub used_sats: Vec<String>,
    /// Post-fit pseudorange residuals, meters.
    pub residuals_m: Vec<f64>,
}

/// A solution that can feed the RAIM test.
pub trait RaimSolution {
    /// Used satellite tokens, in residual order.
    fn raim_used_sats(&self) -> Vec<String>;
    /// Post-fit residuals, meters, in used-satellite order.
    fn raim_residuals_m(&self) -> &[f64];
}

impl RaimSolution for ReceiverSolution {
    fn raim_used_sats(&self) -> Vec<String> {
        self.used_sats.iter().map(ToString::to_string).collect()
    }

    fn raim_residuals_m(&self) -> &[f64] {
        &self.residuals_m
    }
}

/// Result of a residual chi-square RAIM test.
#[derive(Debug, Clone, PartialEq)]
pub struct RaimResult {
    /// True when the test statistic exceeds the chi-square threshold.
    pub fault_detected: bool,
    /// Weighted residual sum of squares.
    pub test_statistic: f64,
    /// Chi-square threshold, absent when the geometry is not testable.
    pub threshold: Option<f64>,
    /// Degrees of freedom, `n_used - (3 + n_systems)`.
    pub dof: isize,
    /// False when `dof <= 0`.
    pub testable: bool,
    /// Per-satellite standardized residuals.
    pub normalized_residuals: BTreeMap<String, f64>,
    /// Satellite with the largest absolute standardized residual.
    pub worst_sat: Option<String>,
}

/// Standalone post-fit residual diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualDiagnostics {
    /// Number of residuals.
    pub n_residuals: usize,
    /// Number of fitted parameters used to compute redundancy.
    pub n_parameters: usize,
    /// Redundancy / degrees of freedom: `n_residuals - n_parameters`.
    pub degrees_of_freedom: isize,
    /// Weighted residual sum of squares.
    pub weighted_sum_squares: f64,
    /// Root-mean-square residual in metres, unweighted.
    pub rms_m: f64,
    /// Residuals scaled by `sqrt(weight)`; unit weights when no weights are given.
    pub normalized_residuals: Vec<f64>,
    /// Index of the largest absolute normalized residual.
    pub worst_index: Option<usize>,
    /// Reduced chi-square, `weighted_sum_squares / degrees_of_freedom`, when
    /// degrees of freedom are positive.
    pub reduced_chi_square: Option<f64>,
    /// Chi-square threshold for the requested false-alarm probability, when
    /// requested and degrees of freedom are positive.
    pub chi_square_threshold: Option<f64>,
    /// Whether `weighted_sum_squares <= chi_square_threshold`, when a threshold
    /// was requested and degrees of freedom are positive.
    pub chi_square_consistent: Option<bool>,
}

/// Post-fit residual diagnostics from residuals and optional inverse-variance
/// weights.
///
/// `n_parameters` is the number of estimated state parameters in the fit that
/// produced `residuals_m`. `p_fa`, when supplied, requests a global chi-square
/// consistency threshold at probability `1 - p_fa`.
pub fn residual_diagnostics(
    residuals_m: &[f64],
    weights: Option<&[f64]>,
    n_parameters: usize,
    p_fa: Option<f64>,
) -> Result<ResidualDiagnostics, QualityError> {
    validate::finite_slice(residuals_m, "diagnostic residuals")
        .map_err(|_| QualityError::InvalidResiduals)?;
    let weights = match weights {
        Some(weights) => {
            if weights.len() != residuals_m.len() {
                return Err(QualityError::InvalidWeight);
            }
            validate_weights_slice(weights)?;
            Some(weights)
        }
        None => None,
    };
    if let Some(p_fa) = p_fa {
        validate_probability(p_fa)?;
    }

    let degrees_of_freedom = residuals_m.len() as isize - n_parameters as isize;
    let mut weighted_sum_squares = 0.0;
    let mut normalized_residuals = Vec::with_capacity(residuals_m.len());
    let mut worst_index = None;
    let mut worst_abs = f64::NEG_INFINITY;
    for (idx, residual_m) in residuals_m.iter().enumerate() {
        let weight = weights.map(|w| w[idx]).unwrap_or(1.0);
        let normalized = residual_m * weight.sqrt();
        weighted_sum_squares += residual_m * residual_m * weight;
        normalized_residuals.push(normalized);
        let abs_normalized = normalized.abs();
        if abs_normalized > worst_abs {
            worst_abs = abs_normalized;
            worst_index = Some(idx);
        }
    }

    let rms_m = residual_rms(residuals_m);
    let reduced_chi_square = if degrees_of_freedom > 0 {
        Some(weighted_sum_squares / degrees_of_freedom as f64)
    } else {
        None
    };
    let chi_square_threshold = match (p_fa, degrees_of_freedom > 0) {
        (Some(p_fa), true) => Some(chi2_inv(1.0 - p_fa, degrees_of_freedom as usize)?),
        _ => None,
    };
    let chi_square_consistent =
        chi_square_threshold.map(|threshold| weighted_sum_squares <= threshold);

    Ok(ResidualDiagnostics {
        n_residuals: residuals_m.len(),
        n_parameters,
        degrees_of_freedom,
        weighted_sum_squares,
        rms_m,
        normalized_residuals,
        worst_index,
        reduced_chi_square,
        chi_square_threshold,
        chi_square_consistent,
    })
}

/// Run RAIM over a generic solution.
pub fn raim_for_solution<S: RaimSolution>(
    solution: &S,
    options: &RaimOptions,
) -> Result<RaimResult, QualityError> {
    raim(
        &RaimInput {
            used_sats: solution.raim_used_sats(),
            residuals_m: solution.raim_residuals_m().to_vec(),
        },
        options,
    )
}

/// Residual-based chi-square RAIM.
pub fn raim(input: &RaimInput, options: &RaimOptions) -> Result<RaimResult, QualityError> {
    validate_probability(options.p_fa)?;
    options.weights.validate()?;
    validate_raim_input(input)?;

    let n_used = input.used_sats.len() as isize;
    let n_systems = raim_system_count(input, options)?;
    let dof = n_used - (3 + n_systems);

    let mut test_statistic = 0.0;
    let mut normalized_residuals = BTreeMap::new();
    let mut worst_sat = None::<String>;
    let mut worst_abs = f64::NEG_INFINITY;

    for (satellite_id, residual_m) in input.used_sats.iter().zip(input.residuals_m.iter()) {
        let weight = options.weights.weight_for(satellite_id);
        let normalized = residual_m * weight.sqrt();
        test_statistic += residual_m * residual_m * weight;
        normalized_residuals.insert(satellite_id.clone(), normalized);
        let abs_normalized = normalized.abs();
        if abs_normalized > worst_abs {
            worst_abs = abs_normalized;
            worst_sat = Some(satellite_id.clone());
        }
    }

    if dof <= 0 {
        return Ok(RaimResult {
            fault_detected: false,
            test_statistic,
            threshold: None,
            dof,
            testable: false,
            normalized_residuals,
            worst_sat,
        });
    }

    let threshold = chi2_inv(1.0 - options.p_fa, dof as usize)?;
    Ok(RaimResult {
        fault_detected: test_statistic > threshold,
        test_statistic,
        threshold: Some(threshold),
        dof,
        testable: true,
        normalized_residuals,
        worst_sat,
    })
}

fn validate_probability(p: f64) -> Result<(), QualityError> {
    let p = validate::finite(p, "probability").map_err(|_| QualityError::InvalidProbability)?;
    if p > 0.0 && p < 1.0 {
        Ok(())
    } else {
        Err(QualityError::InvalidProbability)
    }
}

fn validate_raim_input(input: &RaimInput) -> Result<(), QualityError> {
    if input.used_sats.len() != input.residuals_m.len() {
        return Err(QualityError::InvalidResiduals);
    }
    validate::finite_slice(&input.residuals_m, "raim residuals")
        .map_err(|_| QualityError::InvalidResiduals)
}

fn validate_weights_slice(weights: &[f64]) -> Result<(), QualityError> {
    weights
        .iter()
        .try_for_each(|w| validate::finite_positive(*w, "diagnostic weight").map(|_| ()))
        .map_err(|_| QualityError::InvalidWeight)
}

fn raim_system_count(input: &RaimInput, options: &RaimOptions) -> Result<isize, QualityError> {
    match options.n_systems {
        Some(n_systems) if n_systems >= 1 => Ok(n_systems),
        Some(_) => Err(QualityError::InvalidSystemCount),
        None => Ok(distinct_systems(&input.used_sats)),
    }
}

fn distinct_systems(used_sats: &[String]) -> isize {
    used_sats
        .iter()
        .filter_map(|sat| sat.chars().next())
        .collect::<BTreeSet<_>>()
        .len() as isize
}

/// Result of a fault-detection-and-exclusion loop.
#[derive(Debug, Clone, PartialEq)]
pub struct FdeResult<S> {
    /// Final accepted solution.
    pub solution: S,
    /// Excluded satellites in exclusion order.
    pub excluded: Vec<String>,
    /// Number of exclusions performed.
    pub iterations: usize,
}

/// Error from [`fde`].
#[derive(Debug, Clone, PartialEq)]
pub enum FdeError<E> {
    /// RAIM still flagged the set when the exclusion budget was exhausted.
    FaultUnresolved(f64),
    /// The supplied solve callback failed.
    Solve(E),
    /// RAIM configuration was invalid.
    Raim(QualityError),
}

/// Options for [`fde`].
#[derive(Debug, Clone, PartialEq)]
pub struct FdeOptions {
    /// RAIM options used after each solve.
    pub raim: RaimOptions,
    /// Maximum number of exclusions to attempt.
    pub max_iterations: usize,
}

/// Fault detection and exclusion over a caller-supplied SPP solver.
pub fn fde<S, E, F>(
    observations: &[Observation],
    options: &FdeOptions,
    mut solve: F,
) -> Result<FdeResult<S>, FdeError<E>>
where
    S: RaimSolution,
    F: FnMut(&[Observation]) -> Result<S, E>,
{
    let mut remaining = observations.to_vec();
    let mut excluded = Vec::new();
    let mut iter = 0usize;

    loop {
        let solution = solve(&remaining).map_err(FdeError::Solve)?;
        let result = raim_for_solution(&solution, &options.raim).map_err(FdeError::Raim)?;

        if !result.fault_detected {
            return Ok(FdeResult {
                solution,
                excluded,
                iterations: iter,
            });
        }

        let Some(worst) = result.worst_sat else {
            return Err(FdeError::FaultUnresolved(result.test_statistic));
        };

        if iter >= options.max_iterations {
            return Err(FdeError::FaultUnresolved(result.test_statistic));
        }

        remaining.retain(|ob| ob.satellite_id.to_string() != worst);
        excluded.push(worst);
        iter += 1;
    }
}

// --- single-point-positioning FDE driver ----------------------------------

/// Per-iteration failure carried out of the [`fde_spp`] solve closure: either
/// the SPP [`solve`] failed for the current observation set, or the converged
/// candidate failed [`validate_receiver_solution`].
#[derive(Debug, Clone)]
pub enum FdeSppError {
    /// The SPP solve failed for the current observation set.
    Spp(SppError),
    /// The converged candidate failed solution validation.
    Validation(SolutionValidationError),
}

impl core::fmt::Display for FdeSppError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Spp(err) => write!(f, "SPP solve failed: {err}"),
            Self::Validation(err) => write!(f, "solution validation failed: {err}"),
        }
    }
}

impl std::error::Error for FdeSppError {}

/// Options for [`fde_spp`]: the RAIM-gated exclusion loop plus the per-iteration
/// solution-validation gates applied to each candidate solve.
#[derive(Debug, Clone, PartialEq)]
pub struct FdeSppOptions {
    /// FDE loop options: the RAIM configuration and the exclusion budget.
    pub fde: FdeOptions,
    /// Per-iteration solution-validation gates (PDOP ceiling and plausibility
    /// band) applied to each candidate solution.
    pub validation: SolutionValidationOptions,
}

/// Run single-point positioning with RAIM fault detection and exclusion.
///
/// Solves [`solve`] over the input observation set, applies residual chi-square
/// RAIM via [`fde`], and on a detected fault excludes the worst satellite and
/// re-solves, repeating until the set is self-consistent or the exclusion budget
/// in [`FdeSppOptions::fde`] is exhausted. Every candidate solution is screened
/// with [`validate_receiver_solution`] using [`FdeSppOptions::validation`]. On
/// success returns the protected [`FdeResult`]: the surviving
/// [`ReceiverSolution`], the excluded satellite tokens in exclusion order, and
/// the exclusion count.
///
/// This is the single core driver the language bindings reduce to. It chains the
/// existing [`solve`], [`validate_receiver_solution`], and [`fde`] primitives and
/// adds no detection, exclusion, or solve math of its own, so it is bit-for-bit
/// identical to assembling that loop by hand around the same primitives.
pub fn fde_spp(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    options: &FdeSppOptions,
) -> Result<FdeResult<ReceiverSolution>, FdeError<FdeSppError>> {
    let observations = inputs.observations.clone();
    fde(&observations, &options.fde, |remaining| {
        let mut next = inputs.clone();
        next.observations = remaining.to_vec();
        let solution = solve(eph, &next, with_geodetic).map_err(FdeSppError::Spp)?;
        validate_receiver_solution(&solution, options.validation)
            .map_err(FdeSppError::Validation)?;
        Ok(solution)
    })
}

// --- generic range RAIM/FDE over a linearized measurement set -------------

/// One linearized range measurement for [`raim_fde_design`].
///
/// The set `{ (design_row, residual_m, weight) }` is a single linearization of a
/// range solve about a nominal state: `residual_m` is the observed-minus-computed
/// range, `design_row` is that measurement's row of the design (geometry) matrix
/// `H` (the partials of the predicted range with respect to the estimated state),
/// and `weight` is the measurement's inverse-variance weight `1 / sigma^2`. Every
/// row must carry the same `design_row` length, which is the number of estimated
/// state parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeFdeRow {
    /// Stable measurement identifier, e.g. a satellite token `"G01"`.
    pub id: String,
    /// Observed-minus-computed range residual, metres.
    pub residual_m: f64,
    /// Design-matrix row: partials of the predicted range with respect to each
    /// estimated state parameter. Length equals the state dimension.
    pub design_row: Vec<f64>,
    /// Inverse-variance weight `1 / sigma^2`, square metres reciprocal. Must be
    /// finite and strictly positive.
    pub weight: f64,
}

/// Options for [`raim_fde_design`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeFdeOptions {
    /// False-alarm probability for the global chi-square test. The detection
    /// threshold is the `1 - p_fa` chi-square quantile at the redundancy
    /// (degrees of freedom). RTKLIB demo5 uses `p_fa = 1.0e-3`.
    pub p_fa: f64,
    /// Maximum number of measurements the exclusion loop may remove.
    pub max_exclusions: usize,
    /// Minimum redundancy (degrees of freedom) that an exclusion must leave
    /// behind. An exclusion is only attempted when the surviving set still has
    /// at least `min_redundancy` more measurements than state parameters, so the
    /// protected set stays testable. RTKLIB demo5's `nvsat >= 5` floor for a
    /// four-state solve is `min_redundancy == 1`.
    pub min_redundancy: usize,
}

impl Default for RangeFdeOptions {
    fn default() -> Self {
        Self {
            p_fa: DEFAULT_P_FA,
            max_exclusions: usize::MAX,
            min_redundancy: 1,
        }
    }
}

/// Global chi-square consistency test over a protected measurement set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RangeChiSquareTest {
    /// Weighted sum of squared post-fit residuals, `v^T W v`.
    pub weighted_sum_squares: f64,
    /// Redundancy: `n_used - n_state`.
    pub dof: isize,
    /// Chi-square threshold `chi2_inv(1 - p_fa, dof)`, absent when `dof <= 0`.
    pub threshold: Option<f64>,
    /// False when `dof <= 0` (no redundancy to test against).
    pub testable: bool,
    /// True when the test statistic exceeds the threshold (a fault remains).
    pub fault_detected: bool,
}

/// Per-measurement diagnostics, in the caller's input order.
#[derive(Debug, Clone, PartialEq)]
pub struct RangeMeasurementDiagnostic {
    /// Measurement identifier, echoed from the input row.
    pub id: String,
    /// Whether the FDE loop excluded this measurement from the protected solve.
    pub excluded: bool,
    /// Post-fit residual against the protected state correction, metres
    /// (`residual_m - design_row . dx`). Computed for every input row, including
    /// excluded ones, so a true outlier shows a large value here.
    pub post_fit_residual_m: f64,
    /// Standardized post-fit residual `post_fit_residual_m * sqrt(weight)`.
    pub normalized_residual: f64,
}

/// Result of [`raim_fde_design`].
#[derive(Debug, Clone, PartialEq)]
pub struct RangeFdeResult {
    /// Protected weighted-least-squares state correction `dx`, length `n_state`.
    pub state_correction: Vec<f64>,
    /// Protected state covariance `(H^T W H)^-1` for the accepted set.
    pub state_covariance: Vec<Vec<f64>>,
    /// Global chi-square consistency test for the accepted set.
    pub global_test: RangeChiSquareTest,
    /// Excluded measurement identifiers, in exclusion order.
    pub excluded: Vec<String>,
    /// Per-measurement diagnostics, in input order.
    pub diagnostics: Vec<RangeMeasurementDiagnostic>,
    /// Number of exclusions performed.
    pub iterations: usize,
}

/// A weighted-least-squares fit of a linearized range set.
struct WlsFit {
    dx: Vec<f64>,
    covariance: Vec<Vec<f64>>,
}

/// Standalone, composable range RAIM/FDE over a generic linearized measurement
/// set, independent of any full positioning solve.
///
/// Given rows `{ (design_row, residual_m, weight) }` that linearize a range solve
/// about a nominal state, this solves the protected weighted least squares
/// `dx = (H^T W H)^-1 H^T W r` with covariance `(H^T W H)^-1`, runs the global
/// chi-square consistency test, and, when a fault is detected, runs the fault
/// detection and exclusion (FDE) loop.
///
/// # Algorithm
///
/// 1. Weighted least squares on the active set yields `dx`, the covariance, and
///    post-fit residuals `v = r - H dx`. The test statistic is the weighted sum
///    of squares `WSSR = v^T W v = sum_k w_k v_k^2`.
/// 2. Global chi-square test: with redundancy `dof = n_used - n_state`, a fault
///    is declared when `WSSR > chi2_inv(1 - p_fa, dof)`. This is the standard
///    snapshot residual-based RAIM test and matches RTKLIB demo5's `valsol`
///    chi-square gate (`pntpos.c`).
/// 3. FDE exclusion loop (the RTKLIB demo5 `raim_fde` leave-one-out pattern,
///    `pntpos.c`): while a fault is detected and the exclusion budget and
///    redundancy floor allow, each active measurement is removed in turn, the set
///    is re-solved, and the candidate whose removal yields the smallest reduced
///    weighted post-fit residual RMS is excluded. The loop repeats so multiple
///    outliers can be removed, stopping when the test passes, the budget is
///    exhausted, or no further exclusion keeps the set testable.
///
/// The returned [`RangeChiSquareTest`] reports whether a fault still remains
/// after the loop, so a caller can detect an unresolved fault without an error
/// path. An error is returned only when the input is malformed or the initial
/// geometry is rank deficient.
///
/// # References
///
/// - RTKLIB demo5, `pntpos.c` (`valsol` chi-square residual gate and `raim_fde`
///   leave-one-out exclusion) and `rtkcmn.c` (`chisqr` table, `alpha = 0.001`).
/// - Parkinson & Spilker, *Global Positioning System: Theory and Applications*,
///   Vol. II, Ch. 5 (RAIM, integrity monitoring).
/// - Kaplan & Hegarty, *Understanding GPS/GNSS: Principles and Applications*,
///   3rd ed., receiver-autonomous-integrity-monitoring section.
pub fn raim_fde_design(
    rows: &[RangeFdeRow],
    options: &RangeFdeOptions,
) -> Result<RangeFdeResult, QualityError> {
    validate_probability(options.p_fa)?;
    let n_state = validate_range_rows(rows)?;

    let mut active: Vec<usize> = (0..rows.len()).collect();
    let mut excluded: Vec<String> = Vec::new();
    let mut iterations = 0usize;

    let mut fit = solve_range_wls(rows, &active, n_state)?;
    loop {
        let test = range_chi_square_test(rows, &active, &fit, n_state, options.p_fa)?;

        if !test.fault_detected || excluded.len() >= options.max_exclusions {
            return Ok(finish_range_fde(
                rows, &active, &excluded, fit, test, iterations,
            ));
        }

        // Leave-one-out: pick the exclusion that minimises the reduced weighted
        // post-fit residual RMS while keeping the surviving set testable.
        let Some((slot, candidate_fit)) =
            best_range_exclusion(rows, &active, n_state, options.min_redundancy)
        else {
            return Ok(finish_range_fde(
                rows, &active, &excluded, fit, test, iterations,
            ));
        };

        excluded.push(rows[active[slot]].id.clone());
        active.remove(slot);
        fit = candidate_fit;
        iterations += 1;
    }
}

fn finish_range_fde(
    rows: &[RangeFdeRow],
    active: &[usize],
    excluded: &[String],
    fit: WlsFit,
    test: RangeChiSquareTest,
    iterations: usize,
) -> RangeFdeResult {
    let active_set: BTreeSet<usize> = active.iter().copied().collect();
    let diagnostics = rows
        .iter()
        .enumerate()
        .map(|(idx, row)| {
            let post_fit = row.residual_m - dot(&row.design_row, &fit.dx);
            RangeMeasurementDiagnostic {
                id: row.id.clone(),
                excluded: !active_set.contains(&idx),
                post_fit_residual_m: post_fit,
                normalized_residual: post_fit * row.weight.sqrt(),
            }
        })
        .collect();

    RangeFdeResult {
        state_correction: fit.dx,
        state_covariance: fit.covariance,
        global_test: test,
        excluded: excluded.to_vec(),
        diagnostics,
        iterations,
    }
}

/// Find the single exclusion that minimises the reduced weighted post-fit RMS.
/// Returns the slot (index into `active`) and the re-solved fit, or `None` when
/// no exclusion keeps the surviving set solvable and testable.
fn best_range_exclusion(
    rows: &[RangeFdeRow],
    active: &[usize],
    n_state: usize,
    min_redundancy: usize,
) -> Option<(usize, WlsFit)> {
    // The surviving set must keep at least `min_redundancy` redundancy.
    if active.len() < n_state + min_redundancy + 1 {
        return None;
    }

    let mut best: Option<(usize, WlsFit, f64)> = None;
    let mut remaining: Vec<usize> = Vec::with_capacity(active.len() - 1);
    for slot in 0..active.len() {
        remaining.clear();
        remaining.extend(active.iter().enumerate().filter_map(|(s, &idx)| {
            if s == slot {
                None
            } else {
                Some(idx)
            }
        }));

        let Ok(candidate) = solve_range_wls(rows, &remaining, n_state) else {
            continue;
        };
        let rms = reduced_weighted_rms(rows, &remaining, &candidate);

        let better = match &best {
            Some((_, _, best_rms)) => rms < *best_rms,
            None => true,
        };
        if better {
            best = Some((slot, candidate, rms));
        }
    }

    best.map(|(slot, fit, _)| (slot, fit))
}

/// Reduced weighted post-fit residual RMS, `sqrt(WSSR / n)`. This is the RTKLIB
/// demo5 `raim_fde` selection statistic in standardized (weighted) form.
fn reduced_weighted_rms(rows: &[RangeFdeRow], active: &[usize], fit: &WlsFit) -> f64 {
    if active.is_empty() {
        return 0.0;
    }
    let mut wss = 0.0;
    for &idx in active {
        let row = &rows[idx];
        let v = row.residual_m - dot(&row.design_row, &fit.dx);
        wss += row.weight * v * v;
    }
    (wss / active.len() as f64).sqrt()
}

fn range_chi_square_test(
    rows: &[RangeFdeRow],
    active: &[usize],
    fit: &WlsFit,
    n_state: usize,
    p_fa: f64,
) -> Result<RangeChiSquareTest, QualityError> {
    let mut weighted_sum_squares = 0.0;
    for &idx in active {
        let row = &rows[idx];
        let v = row.residual_m - dot(&row.design_row, &fit.dx);
        weighted_sum_squares += row.weight * v * v;
    }

    let dof = active.len() as isize - n_state as isize;
    if dof <= 0 {
        return Ok(RangeChiSquareTest {
            weighted_sum_squares,
            dof,
            threshold: None,
            testable: false,
            fault_detected: false,
        });
    }

    let threshold = chi2_inv(1.0 - p_fa, dof as usize)?;
    Ok(RangeChiSquareTest {
        weighted_sum_squares,
        dof,
        threshold: Some(threshold),
        testable: true,
        fault_detected: weighted_sum_squares > threshold,
    })
}

/// Solve the protected weighted least squares over the active rows.
///
/// Reuses the shared weighted normal-equation accumulator and symmetric
/// positive-definite inverse: the row weight handed to
/// [`normal_equations_weighted`] is `sqrt(weight)`, so the normal matrix is
/// exactly `H^T W H` and the right-hand side `H^T W r`.
fn solve_range_wls(
    rows: &[RangeFdeRow],
    active: &[usize],
    n_state: usize,
) -> Result<WlsFit, QualityError> {
    let (ata, aty) = normal_equations_weighted(
        active.iter().map(|&idx| {
            let row = &rows[idx];
            (row.design_row.as_slice(), row.residual_m, row.weight.sqrt())
        }),
        n_state,
    )
    .ok_or(QualityError::InvalidDesign)?;

    let covariance = invert_symmetric_pd(&ata).ok_or(QualityError::SingularGeometry)?;
    let dx = (0..n_state)
        .map(|i| (0..n_state).map(|j| covariance[i][j] * aty[j]).sum())
        .collect();
    Ok(WlsFit { dx, covariance })
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn validate_range_rows(rows: &[RangeFdeRow]) -> Result<usize, QualityError> {
    let first = rows.first().ok_or(QualityError::InvalidDesign)?;
    let n_state = first.design_row.len();
    if n_state == 0 || rows.len() < n_state {
        return Err(QualityError::InvalidDesign);
    }
    for row in rows {
        if row.design_row.len() != n_state {
            return Err(QualityError::InvalidDesign);
        }
        validate::finite_slice(&row.design_row, "design row")
            .map_err(|_| QualityError::InvalidDesign)?;
        validate::finite(row.residual_m, "design residual")
            .map_err(|_| QualityError::InvalidResiduals)?;
        validate::finite_positive(row.weight, "design weight")
            .map_err(|_| QualityError::InvalidWeight)?;
    }
    Ok(n_state)
}

/// Validation policy for receiver solutions returned by SPP.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolutionValidationOptions {
    /// Optional PDOP ceiling.
    pub max_pdop: Option<f64>,
    /// Minimum plausible geocentric radius, meters.
    pub min_plausible_radius_m: f64,
    /// Maximum plausible geocentric radius, meters.
    pub max_plausible_radius_m: f64,
    /// Maximum plausible RMS for a solution flagged converged, meters.
    pub max_converged_residual_rms_m: f64,
}

impl Default for SolutionValidationOptions {
    fn default() -> Self {
        Self {
            max_pdop: None,
            min_plausible_radius_m: 6_344_752.0,
            max_plausible_radius_m: 8_378_137.0,
            max_converged_residual_rms_m: 1.0e4,
        }
    }
}

/// Error from [`validate_receiver_solution`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SolutionValidationError {
    /// Validation gate options were malformed or degenerate.
    InvalidOptions {
        /// The invalid option field.
        field: &'static str,
        /// The validation failure category.
        reason: &'static str,
    },
    /// DOP could not be computed because the geometry was rank deficient.
    DegenerateGeometryRankDeficient,
    /// PDOP exceeded the caller's configured ceiling.
    DegenerateGeometryPdop(f64),
    /// Position geocentric radius was outside the physical receiver band.
    ImplausiblePosition(f64),
    /// Converged solution residuals were non-finite or produced non-finite RMS.
    InvalidResiduals,
    /// Converged solution had physically implausible post-fit residual RMS.
    NoConvergence(f64),
}

impl core::fmt::Display for SolutionValidationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidOptions { field, reason } => {
                write!(f, "invalid receiver validation option {field}: {reason}")
            }
            Self::DegenerateGeometryRankDeficient => {
                write!(f, "receiver geometry is rank deficient")
            }
            Self::DegenerateGeometryPdop(pdop) => {
                write!(
                    f,
                    "receiver geometry PDOP {pdop} exceeds the configured limit"
                )
            }
            Self::ImplausiblePosition(radius_m) => write!(
                f,
                "receiver geocentric radius {radius_m} m is outside the plausible range"
            ),
            Self::InvalidResiduals => {
                write!(f, "converged solution residuals must be finite")
            }
            Self::NoConvergence(rms_m) => write!(
                f,
                "converged solution residual RMS {rms_m} m is implausibly large"
            ),
        }
    }
}

impl std::error::Error for SolutionValidationError {}

/// Apply the receiver-solution plausibility gates used by the Sidereon SPP API.
pub fn validate_receiver_solution(
    solution: &ReceiverSolution,
    options: SolutionValidationOptions,
) -> Result<(), SolutionValidationError> {
    validate_solution_validation_options(options)?;

    let Some(dop) = solution.dop.as_ref() else {
        return Err(SolutionValidationError::DegenerateGeometryRankDeficient);
    };

    if let Some(max_pdop) = options.max_pdop {
        if dop.pdop > max_pdop {
            return Err(SolutionValidationError::DegenerateGeometryPdop(dop.pdop));
        }
    }

    let p = solution.position.as_array();
    let radius_m = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    if radius_m < options.min_plausible_radius_m || radius_m > options.max_plausible_radius_m {
        return Err(SolutionValidationError::ImplausiblePosition(radius_m));
    }

    if solution.metadata.converged {
        if validate::finite_slice(&solution.residuals_m, "solution residuals").is_err() {
            return Err(SolutionValidationError::InvalidResiduals);
        }
        let rms = residual_rms(&solution.residuals_m);
        if !rms.is_finite() {
            return Err(SolutionValidationError::InvalidResiduals);
        }
        if rms > options.max_converged_residual_rms_m {
            return Err(SolutionValidationError::NoConvergence(rms));
        }
    }

    Ok(())
}

fn validate_solution_validation_options(
    options: SolutionValidationOptions,
) -> Result<(), SolutionValidationError> {
    if let Some(max_pdop) = options.max_pdop {
        validate::finite_positive(max_pdop, "max_pdop").map_err(validation_option_error)?;
    }
    validate::finite_positive(options.min_plausible_radius_m, "min_plausible_radius_m")
        .map_err(validation_option_error)?;
    validate::finite_positive(options.max_plausible_radius_m, "max_plausible_radius_m")
        .map_err(validation_option_error)?;
    if options.min_plausible_radius_m >= options.max_plausible_radius_m {
        return Err(invalid_validation_option(
            "plausible_radius_m",
            "must be increasing",
        ));
    }
    validate::finite_positive(
        options.max_converged_residual_rms_m,
        "max_converged_residual_rms_m",
    )
    .map_err(validation_option_error)?;
    Ok(())
}

fn validation_option_error(error: validate::FieldError) -> SolutionValidationError {
    invalid_validation_option(error.field(), error.reason())
}

fn invalid_validation_option(field: &'static str, reason: &'static str) -> SolutionValidationError {
    SolutionValidationError::InvalidOptions { field, reason }
}

fn residual_rms(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 0.0;
    }
    let sum_sq = residuals.iter().map(|r| r * r).sum::<f64>();
    (sum_sq / residuals.len() as f64).sqrt()
}

/// Chi-square inverse CDF.
pub fn chi2_inv(p: f64, k: usize) -> Result<f64, QualityError> {
    validate_probability(p)?;
    if k == 0 {
        return Err(QualityError::InvalidDof);
    }
    let a = 0.5 * k as f64;
    let hi0 = (k as f64 + 10.0 * (2.0 * k as f64).sqrt()).max(1.0);
    let hi = chi2_bracket_hi(p, a, hi0);
    Ok(chi2_bisect(p, a, 0.0, hi, 0))
}

fn chi2_bracket_hi(p: f64, a: f64, hi: f64) -> f64 {
    if chi2_cdf(hi, a) >= p {
        hi
    } else {
        chi2_bracket_hi(p, a, hi * 2.0)
    }
}

fn chi2_bisect(p: f64, a: f64, lo: f64, hi: f64, iter: usize) -> f64 {
    if iter >= 120 {
        return 0.5 * (lo + hi);
    }
    let mid = 0.5 * (lo + hi);
    if chi2_cdf(mid, a) < p {
        chi2_bisect(p, a, mid, hi, iter + 1)
    } else {
        chi2_bisect(p, a, lo, mid, iter + 1)
    }
}

fn chi2_cdf(x: f64, a: f64) -> f64 {
    regularized_gamma_p(a, 0.5 * x)
}

const GAMMA_EPS: f64 = 1.0e-15;
const GAMMA_FPMIN: f64 = 1.0e-300;
const GAMMA_ITMAX: usize = 1_000;

fn regularized_gamma_p(a: f64, x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }

    if x < a + 1.0 {
        let gln = log_gamma(a);
        let sum = gamma_series(x, 1.0 / a, 1.0 / a, a, 1);
        sum * (-x + a * x.ln() - gln).exp()
    } else {
        let gln = log_gamma(a);
        let q = gamma_continued_fraction(a, x) * (-x + a * x.ln() - gln).exp();
        1.0 - q
    }
}

fn gamma_series(x: f64, sum: f64, del: f64, ap: f64, n: usize) -> f64 {
    if n > GAMMA_ITMAX {
        return sum;
    }
    let ap = ap + 1.0;
    let del = del * x / ap;
    let sum = sum + del;
    if del.abs() < sum.abs() * GAMMA_EPS {
        sum
    } else {
        gamma_series(x, sum, del, ap, n + 1)
    }
}

fn gamma_continued_fraction(a: f64, x: f64) -> f64 {
    let b = x + 1.0 - a;
    let c = 1.0 / GAMMA_FPMIN;
    let d = 1.0 / safe_denominator(b);
    gamma_cf_iter(a, b, c, d, d, 1)
}

fn gamma_cf_iter(a: f64, b: f64, c: f64, d: f64, h: f64, n: usize) -> f64 {
    if n > GAMMA_ITMAX {
        return h;
    }

    let an = -(n as f64) * (n as f64 - a);
    let b = b + 2.0;
    let d = 1.0 / safe_denominator(an * d + b);
    let c = safe_denominator(b + an / c);
    let delta = d * c;
    let h = h * delta;

    if (delta - 1.0).abs() < GAMMA_EPS {
        h
    } else {
        gamma_cf_iter(a, b, c, d, h, n + 1)
    }
}

fn safe_denominator(x: f64) -> f64 {
    if x.abs() < GAMMA_FPMIN {
        GAMMA_FPMIN
    } else {
        x
    }
}

const LANCZOS: [f64; 9] = [
    0.9999999999998099,
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
];
const SQRT_2PI: f64 = 2.5066282746310002;

fn log_gamma(z: f64) -> f64 {
    if z < 0.5 {
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - log_gamma(1.0 - z)
    } else {
        let z = z - 1.0;
        let mut x = LANCZOS[0];
        for (i, coef) in LANCZOS.iter().enumerate().skip(1) {
            x += coef / (z + i as f64);
        }
        let t = z + 7.5;
        SQRT_2PI.ln() + (z + 0.5) * t.ln() - t + x.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GnssSatelliteId, GnssSystem};

    use std::path::PathBuf;

    use crate::rinex_nav::BroadcastStore;
    use crate::rinex_obs::{pseudoranges, RinexObs, SignalPolicy};
    use crate::spp::{Corrections, KlobucharCoeffs, SurfaceMet};

    fn fixture_path(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name)
    }

    /// The real ESBC broadcast navigation store (day 177, GPS) used as a live,
    /// converging FDE ephemeris source.
    fn esbc_broadcast_store() -> BroadcastStore {
        let nav = std::fs::read_to_string(fixture_path("nav/ESBC00DNK_R_20201770000_01D_MN.rnx"))
            .expect("read ESBC broadcast NAV fixture");
        BroadcastStore::from_nav(&nav).expect("parse ESBC broadcast NAV")
    }

    /// The real ESBC first-epoch GPS L1 pseudorange solve inputs.
    fn esbc_first_epoch_inputs() -> SolveInputs {
        let obs_text = std::fs::read_to_string(fixture_path(
            "obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx",
        ))
        .expect("read ESBC OBS fixture");
        let obs = RinexObs::parse(&obs_text).expect("parse ESBC OBS fixture");
        let policy = SignalPolicy {
            codes: [(GnssSystem::Gps, vec!["C1C".to_string()])]
                .into_iter()
                .collect(),
        };
        let observations = pseudoranges(&obs, &obs.epochs()[0], &policy)
            .expect("valid pseudoranges")
            .into_iter()
            .map(|(satellite_id, pseudorange_m)| Observation {
                satellite_id,
                pseudorange_m,
            })
            .collect();

        SolveInputs {
            observations,
            t_rx_j2000_s: 646_315_200.0,
            t_rx_second_of_day_s: 0.0,
            day_of_year: 177.0,
            initial_guess: [3_582_135.0, 532_569.0, 5_232_779.0, 0.0],
            corrections: Corrections {
                ionosphere: false,
                troposphere: true,
            },
            klobuchar: KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            },
            beidou_klobuchar: None,
            galileo_nequick: None,
            glonass_channels: std::collections::BTreeMap::new(),
            met: SurfaceMet {
                pressure_hpa: 1013.25,
                temperature_k: 288.15,
                relative_humidity: 0.5,
            },
            robust: None,
        }
    }

    fn assert_receiver_solution_bits_eq(left: &ReceiverSolution, right: &ReceiverSolution) {
        assert_eq!(left.position.x_m.to_bits(), right.position.x_m.to_bits());
        assert_eq!(left.position.y_m.to_bits(), right.position.y_m.to_bits());
        assert_eq!(left.position.z_m.to_bits(), right.position.z_m.to_bits());
        assert_eq!(left.geodetic, right.geodetic);
        assert_eq!(left.rx_clock_s.to_bits(), right.rx_clock_s.to_bits());
        assert_eq!(left.dop, right.dop);
        assert_eq!(
            left.residuals_m
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            right
                .residuals_m
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(left.used_sats, right.used_sats);
        assert_eq!(left.rejected_sats, right.rejected_sats);
        assert_eq!(left.metadata, right.metadata);
    }

    /// The `fde_spp` driver must equal the hand-assembled solve + validate + FDE
    /// loop the bindings each spell out, bit-for-bit, on a real converging
    /// scenario with one injected outlier that the loop detects and excludes.
    #[test]
    fn fde_spp_matches_manual_composition_bit_for_bit() {
        let store = esbc_broadcast_store();
        let with_geodetic = true;

        // Solve the clean set first so the outlier is injected on a satellite the
        // solver actually uses (the real epoch drops three low-elevation GPS
        // satellites before RAIM ever sees them).
        let clean_inputs = esbc_first_epoch_inputs();
        let clean = solve(&store, &clean_inputs, with_geodetic).expect("clean solve converges");
        assert!(
            clean.used_sats.len() >= 6,
            "scenario needs redundancy for a testable RAIM exclusion"
        );
        let outlier_sat = *clean.used_sats.last().expect("a used satellite");

        // Inject a gross 1 km bias on that used satellite so RAIM has a clear
        // worst residual to exclude.
        let mut inputs = clean_inputs;
        let outlier_obs = inputs
            .observations
            .iter_mut()
            .find(|obs| obs.satellite_id == outlier_sat)
            .expect("outlier satellite is present in the observation set");
        outlier_obs.pseudorange_m += 1000.0;

        let options = FdeSppOptions {
            fde: FdeOptions {
                raim: RaimOptions::default(),
                max_iterations: inputs.observations.len().saturating_sub(4),
            },
            validation: SolutionValidationOptions::default(),
        };

        // Driver path.
        let driver = fde_spp(&store, &inputs, with_geodetic, &options)
            .expect("driver FDE resolves the fault");

        // Hand-assembled reference: exactly the loop the bindings reduce to.
        let observations = inputs.observations.clone();
        let reference = fde(&observations, &options.fde, |remaining| {
            let mut next = inputs.clone();
            next.observations = remaining.to_vec();
            let solution = solve(&store, &next, with_geodetic).map_err(FdeSppError::Spp)?;
            validate_receiver_solution(&solution, options.validation)
                .map_err(FdeSppError::Validation)?;
            Ok::<_, FdeSppError>(solution)
        })
        .expect("reference FDE resolves the fault");

        // Bit-for-bit parity: the driver IS the hand-assembled loop.
        assert_eq!(driver.excluded, reference.excluded);
        assert_eq!(driver.iterations, reference.iterations);
        assert_receiver_solution_bits_eq(&driver.solution, &reference.solution);

        // The fault drove the loop to detect, exclude, and re-solve: the
        // protected solution dropped at least one satellite and the surviving
        // set is self-consistent under RAIM. (The injected blunder is smeared by
        // unit-weight RAIM onto other residuals, so the excluded set is the
        // engine's own RAIM decision; the driver mirrors it exactly rather than
        // imposing any exclusion policy of its own.)
        assert!(driver.iterations >= 1, "the fault must drive an exclusion");
        assert!(!driver.excluded.is_empty());
        assert_eq!(driver.excluded.len(), driver.iterations);
        let surviving = raim_for_solution(&driver.solution, &options.fde.raim).expect("raim");
        assert!(
            !surviving.fault_detected,
            "the protected set must pass RAIM (or be untestable)"
        );
    }

    /// A clean set converges with no exclusion, and the driver still equals the
    /// hand-assembled composition bit-for-bit.
    #[test]
    fn fde_spp_clean_set_takes_no_exclusion_and_matches_manual() {
        let store = esbc_broadcast_store();
        let inputs = esbc_first_epoch_inputs();
        let options = FdeSppOptions {
            fde: FdeOptions {
                raim: RaimOptions::default(),
                max_iterations: inputs.observations.len().saturating_sub(4),
            },
            validation: SolutionValidationOptions::default(),
        };

        let driver = fde_spp(&store, &inputs, false, &options).expect("driver solves clean set");

        let observations = inputs.observations.clone();
        let reference = fde(&observations, &options.fde, |remaining| {
            let mut next = inputs.clone();
            next.observations = remaining.to_vec();
            let solution = solve(&store, &next, false).map_err(FdeSppError::Spp)?;
            validate_receiver_solution(&solution, options.validation)
                .map_err(FdeSppError::Validation)?;
            Ok::<_, FdeSppError>(solution)
        })
        .expect("reference solves clean set");

        assert_eq!(driver.iterations, 0);
        assert!(driver.excluded.is_empty());
        assert_eq!(driver.iterations, reference.iterations);
        assert_eq!(driver.excluded, reference.excluded);
        assert_receiver_solution_bits_eq(&driver.solution, &reference.solution);
    }

    #[derive(Debug, Clone)]
    struct TestSolution {
        used_sats: Vec<String>,
        residuals_m: Vec<f64>,
    }

    impl RaimSolution for TestSolution {
        fn raim_used_sats(&self) -> Vec<String> {
            self.used_sats.clone()
        }

        fn raim_residuals_m(&self) -> &[f64] {
            &self.residuals_m
        }
    }

    fn gps(prn: u8) -> GnssSatelliteId {
        GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid satellite id")
    }

    fn valid_receiver_solution() -> ReceiverSolution {
        ReceiverSolution {
            position: crate::frame::ItrfPositionM::new(6_378_137.0, 0.0, 0.0).unwrap(),
            geodetic: None,
            rx_clock_s: 0.0,
            system_clocks_s: vec![(GnssSystem::Gps, 0.0)],
            dop: Some(crate::dop::Dop {
                gdop: 2.5,
                pdop: 2.0,
                hdop: 1.5,
                vdop: 1.0,
                tdop: 0.5,
                system_tdops: vec![(GnssSystem::Gps, 0.5)],
            }),
            system_tdops: vec![(GnssSystem::Gps, 0.5)],
            residuals_m: vec![0.1, -0.1, 0.0, 0.05, -0.05],
            used_sats: (1..=5).map(gps).collect(),
            rejected_sats: Vec::new(),
            metadata: crate::spp::SolutionMetadata {
                iterations: 3,
                converged: true,
                status: crate::astro::math::least_squares::Status::StepTolerance,
                ionosphere_applied: false,
                troposphere_applied: false,
                outer_iterations: 0,
                final_robust_scale_m: None,
                used_count: 5,
                systems: vec![GnssSystem::Gps],
                redundancy: 1,
                raim_checkable: true,
            },
        }
    }

    #[test]
    fn pseudorange_variance_matches_elevation_model() {
        let opts = PseudorangeVarianceOptions::default();
        let variance = pseudorange_variance(30.0, opts).unwrap();
        assert!((variance - 0.45).abs() < 1.0e-15);
        assert_eq!(
            pseudorange_variance(0.0, opts),
            Err(QualityError::InvalidElevation)
        );
        let horizon_opts = PseudorangeVarianceOptions { b_m: 0.0, ..opts };
        assert_eq!(
            pseudorange_variance(0.0, horizon_opts),
            Ok(horizon_opts.a_m * horizon_opts.a_m)
        );
        assert_eq!(
            pseudorange_variance(-90.0, horizon_opts),
            Ok(horizon_opts.a_m * horizon_opts.a_m)
        );
        assert_eq!(
            pseudorange_variance(90.1, horizon_opts),
            Err(QualityError::InvalidElevation)
        );
        assert_eq!(
            pseudorange_variance(f64::NAN, opts),
            Err(QualityError::InvalidElevation)
        );
    }

    #[test]
    fn cn0_model_requires_cn0_and_adds_noise_term() {
        let opts = PseudorangeVarianceOptions {
            model: PseudorangeVarianceModel::ElevationCn0,
            cn0_dbhz: None,
            ..Default::default()
        };
        assert_eq!(
            pseudorange_variance(30.0, opts),
            Err(QualityError::MissingCn0)
        );

        let weak = pseudorange_variance(
            30.0,
            PseudorangeVarianceOptions {
                cn0_dbhz: Some(30.0),
                ..opts
            },
        )
        .unwrap();
        let strong = pseudorange_variance(
            30.0,
            PseudorangeVarianceOptions {
                cn0_dbhz: Some(50.0),
                ..opts
            },
        )
        .unwrap();
        assert!(strong < weak);
    }

    #[test]
    fn pseudorange_variance_rejects_nonfinite_and_negative_parameters() {
        let invalid_a = PseudorangeVarianceOptions {
            a_m: f64::NAN,
            ..Default::default()
        };
        assert_eq!(
            pseudorange_variance(30.0, invalid_a),
            Err(QualityError::InvalidParameter)
        );

        let invalid_b = PseudorangeVarianceOptions {
            b_m: -1.0,
            ..Default::default()
        };
        assert_eq!(
            pseudorange_variance(30.0, invalid_b),
            Err(QualityError::InvalidParameter)
        );

        let invalid_cn0_scale = PseudorangeVarianceOptions {
            cn0_scale_m2: f64::INFINITY,
            ..Default::default()
        };
        assert_eq!(
            pseudorange_variance(30.0, invalid_cn0_scale),
            Err(QualityError::InvalidParameter)
        );

        let invalid_cn0 = PseudorangeVarianceOptions {
            model: PseudorangeVarianceModel::ElevationCn0,
            cn0_dbhz: Some(f64::NAN),
            ..Default::default()
        };
        assert_eq!(
            pseudorange_variance(30.0, invalid_cn0),
            Err(QualityError::InvalidParameter)
        );
    }

    #[test]
    fn pseudorange_variance_rejects_zero_total_variance() {
        let zero_variance = PseudorangeVarianceOptions {
            a_m: 0.0,
            b_m: 0.0,
            ..Default::default()
        };
        assert_eq!(
            pseudorange_variance(30.0, zero_variance),
            Err(QualityError::InvalidParameter)
        );

        let entries = vec![WeightEntry {
            satellite_id: "G01".to_string(),
            elevation_deg: 30.0,
            cn0_dbhz: None,
        }];
        let weights = weight_vector(&entries, zero_variance);
        assert!(
            !weights.contains_key("G01"),
            "zero variance must not produce an infinite inverse-variance weight"
        );
    }

    #[test]
    fn sigma_and_weight_maps_drop_invalid_entries() {
        let entries = vec![
            WeightEntry {
                satellite_id: "G01".to_string(),
                elevation_deg: 90.0,
                cn0_dbhz: None,
            },
            WeightEntry {
                satellite_id: "G02".to_string(),
                elevation_deg: -91.0,
                cn0_dbhz: None,
            },
        ];
        let sigmas = sigmas(&entries, Default::default());
        let weights = weight_vector(&entries, Default::default());
        assert!(sigmas.contains_key("G01"));
        assert!(!sigmas.contains_key("G02"));
        assert_eq!(weights["G01"], 1.0 / (sigmas["G01"] * sigmas["G01"]));
    }

    #[test]
    fn sigma_and_weight_maps_retain_horizon_entries_without_elevation_term() {
        let entries = vec![
            WeightEntry {
                satellite_id: "G01".to_string(),
                elevation_deg: 0.0,
                cn0_dbhz: None,
            },
            WeightEntry {
                satellite_id: "G02".to_string(),
                elevation_deg: f64::NAN,
                cn0_dbhz: None,
            },
        ];
        let options = PseudorangeVarianceOptions {
            b_m: 0.0,
            ..Default::default()
        };
        let sigmas = sigmas(&entries, options);
        let weights = weight_vector(&entries, options);
        assert_eq!(sigmas["G01"], options.a_m);
        assert_eq!(weights["G01"], 1.0 / (options.a_m * options.a_m));
        assert!(!sigmas.contains_key("G02"));
        assert!(!weights.contains_key("G02"));
    }

    #[test]
    fn chi_square_inverse_matches_reference_values() {
        let refs = [
            (1, 10.828),
            (2, 13.816),
            (3, 16.266),
            (4, 18.467),
            (5, 20.515),
        ];
        for (dof, expected) in refs {
            let got = chi2_inv(0.999, dof).unwrap();
            assert!((got - expected).abs() < 1.0e-3);
        }
        assert_eq!(chi2_inv(1.0, 1), Err(QualityError::InvalidProbability));
        assert_eq!(chi2_inv(0.95, 0), Err(QualityError::InvalidDof));
    }

    #[test]
    fn residual_diagnostics_reports_weighted_redundancy_and_reduced_chi_square() {
        let residuals = [1.0, -2.0, 0.5, 3.0, -1.5];
        let weights = [1.0, 0.25, 4.0, 1.0, 0.5];
        let diagnostics =
            residual_diagnostics(&residuals, Some(&weights), 3, Some(1.0e-3)).expect("diagnostics");

        let wss = residuals
            .iter()
            .zip(weights)
            .map(|(r, w)| r * r * w)
            .sum::<f64>();
        assert_eq!(diagnostics.n_residuals, 5);
        assert_eq!(diagnostics.n_parameters, 3);
        assert_eq!(diagnostics.degrees_of_freedom, 2);
        assert_eq!(diagnostics.weighted_sum_squares.to_bits(), wss.to_bits());
        assert_eq!(
            diagnostics.reduced_chi_square.unwrap().to_bits(),
            (wss / 2.0).to_bits()
        );
        assert_eq!(
            diagnostics.normalized_residuals[1].to_bits(),
            (-1.0f64).to_bits()
        );
        assert_eq!(diagnostics.worst_index, Some(3));
        assert!(diagnostics.chi_square_threshold.unwrap().is_finite());
        assert_eq!(diagnostics.chi_square_consistent, Some(true));
    }

    #[test]
    fn residual_diagnostics_handles_no_redundancy_and_rejects_bad_inputs() {
        let residuals = [1.0, -1.0];
        let diagnostics =
            residual_diagnostics(&residuals, None, 2, Some(1.0e-3)).expect("diagnostics");
        assert_eq!(diagnostics.degrees_of_freedom, 0);
        assert_eq!(diagnostics.reduced_chi_square, None);
        assert_eq!(diagnostics.chi_square_threshold, None);
        assert_eq!(diagnostics.chi_square_consistent, None);

        assert_eq!(
            residual_diagnostics(&[1.0, f64::NAN], None, 1, None),
            Err(QualityError::InvalidResiduals)
        );
        assert_eq!(
            residual_diagnostics(&[1.0], Some(&[0.0]), 0, None),
            Err(QualityError::InvalidWeight)
        );
        assert_eq!(
            residual_diagnostics(&[1.0], None, 0, Some(1.0)),
            Err(QualityError::InvalidProbability)
        );
    }

    #[test]
    fn raim_reports_fault_and_worst_satellite() {
        let input = RaimInput {
            used_sats: ["G01", "G02", "G03", "G04", "G05"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            residuals_m: vec![0.0, 0.0, 0.0, 0.0, 5.0],
        };
        let result = raim(&input, &RaimOptions::default()).unwrap();
        assert!(result.fault_detected);
        assert!(result.testable);
        assert_eq!(result.dof, 1);
        assert_eq!(result.test_statistic, 25.0);
        assert_eq!(result.worst_sat.as_deref(), Some("G05"));
    }

    #[test]
    fn raim_dof_zero_is_not_testable() {
        let input = RaimInput {
            used_sats: ["G01", "G02", "G03", "G04"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            residuals_m: vec![0.0, 0.0, 0.0, 0.0],
        };
        let result = raim(&input, &RaimOptions::default()).unwrap();
        assert!(!result.fault_detected);
        assert!(!result.testable);
        assert_eq!(result.threshold, None);
        assert_eq!(result.dof, 0);
    }

    #[test]
    fn raim_rejects_nonpositive_system_overrides() {
        let input = RaimInput {
            used_sats: ["G01", "G02", "G03", "G04", "G05"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            residuals_m: vec![0.0; 5],
        };

        for n_systems in [0, -1] {
            let options = RaimOptions {
                n_systems: Some(n_systems),
                ..Default::default()
            };
            assert_eq!(
                raim(&input, &options),
                Err(QualityError::InvalidSystemCount)
            );
        }
    }

    #[test]
    fn raim_positive_system_override_controls_dof() {
        let input = RaimInput {
            used_sats: ["G01", "G02", "G03", "G04", "G05", "G06"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            residuals_m: vec![0.0; 6],
        };
        let options = RaimOptions {
            n_systems: Some(2),
            ..Default::default()
        };

        let result = raim(&input, &options).unwrap();
        assert!(result.testable);
        assert_eq!(result.dof, 1);
    }

    #[test]
    fn raim_rejects_misaligned_or_nonfinite_residuals() {
        let input = RaimInput {
            used_sats: ["G01", "G02"].into_iter().map(str::to_string).collect(),
            residuals_m: vec![1.0],
        };
        assert_eq!(
            raim(&input, &RaimOptions::default()),
            Err(QualityError::InvalidResiduals)
        );

        let input = RaimInput {
            used_sats: ["G01", "G02"].into_iter().map(str::to_string).collect(),
            residuals_m: vec![1.0, f64::NAN],
        };
        assert_eq!(
            raim(&input, &RaimOptions::default()),
            Err(QualityError::InvalidResiduals)
        );
    }

    #[test]
    fn raim_rejects_nonfinite_weights_and_probability() {
        let input = RaimInput {
            used_sats: ["G01", "G02", "G03", "G04", "G05"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            residuals_m: vec![0.0; 5],
        };
        let mut weights = BTreeMap::new();
        weights.insert("G01".to_string(), f64::NAN);
        let options = RaimOptions {
            weights: RaimWeights::BySatellite(weights),
            ..Default::default()
        };
        assert_eq!(raim(&input, &options), Err(QualityError::InvalidWeight));

        let options = RaimOptions {
            p_fa: f64::NAN,
            ..Default::default()
        };
        assert_eq!(
            raim(&input, &options),
            Err(QualityError::InvalidProbability)
        );
    }

    #[test]
    fn fde_excludes_largest_normalized_residual() {
        let observations: Vec<Observation> = (1..=5)
            .map(|prn| Observation {
                satellite_id: gps(prn),
                pseudorange_m: prn as f64,
            })
            .collect();

        let options = FdeOptions {
            raim: RaimOptions::default(),
            max_iterations: 1,
        };
        let result = fde(&observations, &options, |remaining| {
            let used_sats = remaining
                .iter()
                .map(|ob| ob.satellite_id.to_string())
                .collect::<Vec<_>>();
            let residuals_m = remaining
                .iter()
                .map(|ob| if ob.satellite_id == gps(5) { 5.0 } else { 0.0 })
                .collect::<Vec<_>>();
            Ok::<_, ()>(TestSolution {
                used_sats,
                residuals_m,
            })
        })
        .unwrap();

        assert_eq!(result.excluded, vec!["G05".to_string()]);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.solution.used_sats.len(), 4);
    }

    #[test]
    fn fde_refuses_fault_when_budget_is_exhausted() {
        let observations: Vec<Observation> = (1..=5)
            .map(|prn| Observation {
                satellite_id: gps(prn),
                pseudorange_m: prn as f64,
            })
            .collect();
        let options = FdeOptions {
            raim: RaimOptions::default(),
            max_iterations: 0,
        };
        let err = fde(&observations, &options, |remaining| {
            Ok::<_, ()>(TestSolution {
                used_sats: remaining
                    .iter()
                    .map(|ob| ob.satellite_id.to_string())
                    .collect(),
                residuals_m: vec![0.0, 0.0, 0.0, 0.0, 5.0],
            })
        })
        .unwrap_err();

        assert_eq!(err, FdeError::FaultUnresolved(25.0));
    }

    #[test]
    fn receiver_solution_validation_rejects_invalid_gate_options() {
        let solution = valid_receiver_solution();
        for (options, field, reason) in [
            (
                SolutionValidationOptions {
                    max_pdop: Some(f64::NAN),
                    ..Default::default()
                },
                "max_pdop",
                "not finite",
            ),
            (
                SolutionValidationOptions {
                    max_pdop: Some(0.0),
                    ..Default::default()
                },
                "max_pdop",
                "not positive",
            ),
            (
                SolutionValidationOptions {
                    min_plausible_radius_m: 0.0,
                    ..Default::default()
                },
                "min_plausible_radius_m",
                "not positive",
            ),
            (
                SolutionValidationOptions {
                    max_plausible_radius_m: f64::INFINITY,
                    ..Default::default()
                },
                "max_plausible_radius_m",
                "not finite",
            ),
            (
                SolutionValidationOptions {
                    max_converged_residual_rms_m: f64::NAN,
                    ..Default::default()
                },
                "max_converged_residual_rms_m",
                "not finite",
            ),
        ] {
            assert_eq!(
                validate_receiver_solution(&solution, options),
                Err(SolutionValidationError::InvalidOptions { field, reason })
            );
        }

        let inverted_radius = SolutionValidationOptions {
            min_plausible_radius_m: 8_000_000.0,
            max_plausible_radius_m: 7_000_000.0,
            ..Default::default()
        };
        assert_eq!(
            validate_receiver_solution(&solution, inverted_radius),
            Err(SolutionValidationError::InvalidOptions {
                field: "plausible_radius_m",
                reason: "must be increasing",
            })
        );
    }

    #[test]
    fn receiver_solution_validation_rejects_nonfinite_residuals() {
        let mut solution = valid_receiver_solution();
        solution.residuals_m[1] = f64::NAN;
        assert_eq!(
            validate_receiver_solution(&solution, SolutionValidationOptions::default()),
            Err(SolutionValidationError::InvalidResiduals)
        );
    }

    // --- generic range RAIM/FDE -------------------------------------------

    fn range_design_rows() -> Vec<[f64; 4]> {
        vec![
            [-0.10, -0.20, -0.97, 1.0],
            [0.50, -0.30, -0.81, 1.0],
            [-0.60, 0.40, -0.69, 1.0],
            [0.20, 0.80, -0.56, 1.0],
            [0.70, 0.50, -0.51, 1.0],
            [-0.50, -0.70, -0.51, 1.0],
            [0.30, -0.60, -0.74, 1.0],
            [-0.80, 0.10, -0.59, 1.0],
        ]
    }

    fn range_rows(dx_true: [f64; 4]) -> Vec<RangeFdeRow> {
        range_design_rows()
            .iter()
            .enumerate()
            .map(|(i, h)| RangeFdeRow {
                id: format!("S{:02}", i + 1),
                residual_m: h.iter().zip(dx_true).map(|(a, b)| a * b).sum(),
                design_row: h.to_vec(),
                weight: 1.0,
            })
            .collect()
    }

    fn assert_close(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() < tol, "got {g}, want {w}");
        }
    }

    #[test]
    fn range_fde_clean_set_recovers_state_without_exclusions() {
        let dx_true = [1.0, -2.0, 0.5, 3.0];
        let rows = range_rows(dx_true);
        let result = raim_fde_design(&rows, &RangeFdeOptions::default()).expect("fde");

        assert!(!result.global_test.fault_detected);
        assert!(result.global_test.testable);
        assert_eq!(result.global_test.dof, 4);
        assert!(result.excluded.is_empty());
        assert_eq!(result.iterations, 0);
        assert!(result.global_test.weighted_sum_squares < 1.0e-12);
        assert_close(&result.state_correction, &dx_true, 1.0e-9);
        assert_eq!(result.state_covariance.len(), 4);
    }

    #[test]
    fn range_fde_detects_and_excludes_a_single_outlier() {
        let dx_true = [1.0, -2.0, 0.5, 3.0];
        let mut rows = range_rows(dx_true);
        rows[2].residual_m += 50.0; // inject a fault on S03

        let result = raim_fde_design(&rows, &RangeFdeOptions::default()).expect("fde");

        assert_eq!(result.excluded, vec!["S03".to_string()]);
        assert_eq!(result.iterations, 1);
        assert!(!result.global_test.fault_detected);
        assert_close(&result.state_correction, &dx_true, 1.0e-9);

        let s03 = result
            .diagnostics
            .iter()
            .find(|d| d.id == "S03")
            .expect("S03 diagnostic");
        assert!(s03.excluded);
        // The excluded fault is large against the clean protected solution.
        assert!(s03.post_fit_residual_m.abs() > 40.0);
        // Surviving measurements are consistent.
        for d in result.diagnostics.iter().filter(|d| !d.excluded) {
            assert!(d.normalized_residual.abs() < 1.0e-6);
        }
    }

    #[test]
    fn range_fde_excludes_multiple_outliers() {
        let dx_true = [0.5, 1.5, -1.0, 2.0];
        let mut rows = range_rows(dx_true);
        rows[2].residual_m += 50.0; // S03
        rows[5].residual_m -= 40.0; // S06

        let result = raim_fde_design(&rows, &RangeFdeOptions::default()).expect("fde");

        assert_eq!(result.iterations, 2);
        let mut excluded = result.excluded.clone();
        excluded.sort();
        assert_eq!(excluded, vec!["S03".to_string(), "S06".to_string()]);
        assert!(!result.global_test.fault_detected);
        assert_close(&result.state_correction, &dx_true, 1.0e-9);
    }

    #[test]
    fn range_fde_respects_the_exclusion_budget() {
        let dx_true = [0.5, 1.5, -1.0, 2.0];
        let mut rows = range_rows(dx_true);
        rows[2].residual_m += 50.0;
        rows[5].residual_m -= 40.0;

        let options = RangeFdeOptions {
            max_exclusions: 1,
            ..Default::default()
        };
        let result = raim_fde_design(&rows, &options).expect("fde");

        // One exclusion used; the second fault is still flagged.
        assert_eq!(result.iterations, 1);
        assert_eq!(result.excluded.len(), 1);
        assert!(result.global_test.fault_detected);
    }

    #[test]
    fn range_fde_rejects_rank_deficient_geometry() {
        let rows: Vec<RangeFdeRow> = (0..5)
            .map(|i| RangeFdeRow {
                id: format!("S{:02}", i + 1),
                residual_m: 1.0,
                design_row: vec![1.0, 0.0, 0.0, 1.0], // collinear: rank 2 of 4
                weight: 1.0,
            })
            .collect();
        assert_eq!(
            raim_fde_design(&rows, &RangeFdeOptions::default()),
            Err(QualityError::SingularGeometry)
        );
    }

    #[test]
    fn range_fde_rejects_malformed_inputs() {
        assert_eq!(
            raim_fde_design(&[], &RangeFdeOptions::default()),
            Err(QualityError::InvalidDesign)
        );

        // Fewer measurements than state parameters.
        let too_few = vec![RangeFdeRow {
            id: "S01".to_string(),
            residual_m: 0.0,
            design_row: vec![1.0, 0.0, 0.0, 1.0],
            weight: 1.0,
        }];
        assert_eq!(
            raim_fde_design(&too_few, &RangeFdeOptions::default()),
            Err(QualityError::InvalidDesign)
        );

        // Ragged design rows.
        let mut ragged = range_rows([1.0, 0.0, 0.0, 0.0]);
        ragged[1].design_row.pop();
        assert_eq!(
            raim_fde_design(&ragged, &RangeFdeOptions::default()),
            Err(QualityError::InvalidDesign)
        );

        // Non-positive weight and non-finite residual.
        let mut bad_weight = range_rows([1.0, 0.0, 0.0, 0.0]);
        bad_weight[0].weight = 0.0;
        assert_eq!(
            raim_fde_design(&bad_weight, &RangeFdeOptions::default()),
            Err(QualityError::InvalidWeight)
        );

        let mut bad_residual = range_rows([1.0, 0.0, 0.0, 0.0]);
        bad_residual[0].residual_m = f64::NAN;
        assert_eq!(
            raim_fde_design(&bad_residual, &RangeFdeOptions::default()),
            Err(QualityError::InvalidResiduals)
        );

        let rows = range_rows([1.0, 0.0, 0.0, 0.0]);
        let bad_p = RangeFdeOptions {
            p_fa: 1.0,
            ..Default::default()
        };
        assert_eq!(
            raim_fde_design(&rows, &bad_p),
            Err(QualityError::InvalidProbability)
        );
    }

    #[test]
    fn chi_square_threshold_matches_rtklib_demo5_chisqr_table() {
        // RTKLIB demo5 chi-square detection thresholds, alpha = 0.001
        // (p_fa = 1e-3), from `rtkcmn.c:192` `chisqr[]`, dof 1..=20. The global
        // RAIM test compares the weighted residual sum of squares against this
        // quantile, so reproducing the table is the demo5 oracle for the
        // threshold side of the test.
        let table: [f64; 20] = [
            10.8, 13.8, 16.3, 18.5, 20.5, 22.5, 24.3, 26.1, 27.9, 29.6, 31.3, 32.9, 34.5, 36.1,
            37.7, 39.3, 40.8, 42.3, 43.8, 45.3,
        ];
        for (i, &expected) in table.iter().enumerate() {
            let dof = i + 1;
            let got = chi2_inv(0.999, dof).expect("chi2 quantile");
            let tol = (0.01 * expected).max(0.05);
            assert!(
                (got - expected).abs() < tol,
                "dof {dof}: got {got}, demo5 chisqr {expected}"
            );
        }
    }
}
