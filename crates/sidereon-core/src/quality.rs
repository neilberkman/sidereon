//! Measurement-quality control for GNSS positioning.
//!
//! This module owns the language-independent RAIM/FDE decision logic and the
//! standard pseudorange weighting primitives used by Sidereon' QC surface.

use std::collections::{BTreeMap, BTreeSet};

pub mod normality;

pub use crate::araim::reliability::{
    reliability_araim, reliability_design, wtest_noncentrality, wtest_noncentrality_components,
    ObservationReliability, RangeReliabilityRow, ReliabilityOptions, ReliabilityReport,
    ReliabilitySummary, WtestNoncentralityComponents,
};

use crate::astro::math::linear::{invert_symmetric_pd, normal_equations_weighted};
use crate::constants::DEG_TO_RAD;
use crate::spp::{
    solve, EphemerisSource, Observation, ReceiverSolution, RobustConfig, SolveInputs, SppError,
};
use crate::validate;
use crate::{GnssSatelliteId, GnssSystem};

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
#[non_exhaustive]
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
    /// Reliability parameter must be positive finite or inside its valid interval.
    InvalidReliabilityParameter,
    /// RAIM residuals must be finite and aligned with used satellites.
    InvalidResiduals,
    /// A linearized measurement set was empty, ragged, non-finite, or carried
    /// fewer measurements than estimated state parameters.
    InvalidDesign,
    /// The weighted normal matrix `H^T W H` was singular or rank deficient, so
    /// no protected state correction exists.
    SingularGeometry,
    /// [`RaimWeights::Solution`] was selected for residuals that carry no
    /// variances.
    MissingVariances,
    /// Residual variances must be finite, strictly positive, and aligned with
    /// the used satellites.
    InvalidVariance,
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
            Self::InvalidReliabilityParameter => write!(f, "invalid reliability parameter"),
            Self::InvalidResiduals => write!(f, "invalid RAIM residuals"),
            Self::InvalidDesign => write!(f, "invalid linearized measurement design"),
            Self::SingularGeometry => write!(f, "singular or rank-deficient geometry"),
            Self::MissingVariances => write!(f, "residual variances are required"),
            Self::InvalidVariance => write!(f, "invalid residual variance"),
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
        let sin_el = libm::sin(elevation_deg * DEG_TO_RAD);
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
            elevation_var + options.cn0_scale_m2 * libm::pow(10.0_f64, -cn0 / 10.0)
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
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RaimWeights {
    /// The variances the estimator weighted each residual by
    /// ([`RaimInput::variances_m2`], [`RaimSolution::raim_variances_m2`]). The
    /// test statistic is `sum (r / sigma)^2`, each residual divided by its own
    /// standard deviation, as RTKLIB demo5 `valsol` forms it. Residuals without
    /// variances are refused with [`QualityError::MissingVariances`].
    #[default]
    Solution,
    /// Unit weights, equivalent to sigma = 1 m for every satellite.
    Unit,
    /// Per-satellite inverse variance weights. Missing satellites default to
    /// unit weight.
    BySatellite(BTreeMap<String, f64>),
}

impl RaimWeights {
    fn validate(&self) -> Result<(), QualityError> {
        match self {
            Self::Solution | Self::Unit => Ok(()),
            Self::BySatellite(weights) => weights
                .values()
                .try_for_each(|w| validate::finite_positive(*w, "raim weight").map(|_| ()))
                .map_err(|_| QualityError::InvalidWeight),
        }
    }
}

/// Options for [`raim`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RaimOptions {
    /// False-alarm probability. The default, `1.0e-3`, is the RTKLIB demo5
    /// `chisqr` table's `alpha = 0.001`.
    pub p_fa: f64,
    /// RAIM residual weights. The default is [`RaimWeights::Solution`], the
    /// estimator's own variances.
    pub weights: RaimWeights,
    /// Optional override for the number of receiver clock parameters. Without
    /// it, [`raim_for_solution`] takes the count the solution reports
    /// ([`RaimSolution::raim_clock_count`]) and [`raim`] counts the distinct
    /// system letters of the satellite tokens.
    pub n_systems: Option<isize>,
}

impl Default for RaimOptions {
    fn default() -> Self {
        Self {
            p_fa: DEFAULT_P_FA,
            weights: RaimWeights::Solution,
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
    /// The variance of each residual the estimator weighted it by, square
    /// metres, in residual order; read by [`RaimWeights::Solution`].
    pub variances_m2: Option<Vec<f64>>,
}

/// A solution that can feed the RAIM test.
pub trait RaimSolution {
    /// Used satellite tokens, in residual order.
    fn raim_used_sats(&self) -> Vec<String>;
    /// Post-fit residuals, meters, in used-satellite order.
    fn raim_residuals_m(&self) -> &[f64];
    /// The variance of each residual the estimator weighted it by, square
    /// metres, in used-satellite order, or `None` when the estimator reports
    /// none.
    fn raim_variances_m2(&self) -> Option<&[f64]>;
    /// The number of receiver clock parameters the solution estimated, or
    /// `None` to count the distinct system letters of the satellite tokens.
    fn raim_clock_count(&self) -> Option<usize> {
        None
    }
}

impl RaimSolution for ReceiverSolution {
    fn raim_used_sats(&self) -> Vec<String> {
        self.used_sats.iter().map(ToString::to_string).collect()
    }

    fn raim_residuals_m(&self) -> &[f64] {
        &self.residuals_m
    }

    fn raim_variances_m2(&self) -> Option<&[f64]> {
        Some(&self.pseudorange_variances_m2)
    }

    /// The clock systems of the solve: QZSS and SBAS ranges on the GPS clock add
    /// no clock parameter, so the letters of the tokens would overcount them.
    fn raim_clock_count(&self) -> Option<usize> {
        Some(self.metadata.systems.len())
    }
}

/// Result of a residual chi-square RAIM test.
#[derive(Debug, Clone, PartialEq)]
pub struct RaimResult {
    /// True when the test statistic exceeds the chi-square threshold.
    pub fault_detected: bool,
    /// Weighted residual sum of squares, `sum (r / sigma)^2` under
    /// [`RaimWeights::Solution`] and `sum w r^2` under the other modes.
    pub test_statistic: f64,
    /// Chi-square threshold, absent when the geometry is not testable.
    pub threshold: Option<f64>,
    /// Degrees of freedom, `n_used - (3 + n_systems)`.
    pub dof: isize,
    /// False when `dof <= 0`.
    pub testable: bool,
    /// Per-satellite weighted residuals: `r / sigma` under
    /// [`RaimWeights::Solution`], `r sqrt(w)` under the other modes. These are
    /// not standardized by each residual's redundancy, so a satellite the
    /// geometry leans on shows a small value here even when it carries the
    /// fault. The standardized (w-test) residual is `r / (sigma sqrt(r_i))`
    /// with the redundancy number `r_i` that [`reliability_design`] computes
    /// from the design.
    pub normalized_residuals: BTreeMap<String, f64>,
    /// Satellite with the largest absolute weighted residual in
    /// `normalized_residuals`. This is not a fault identification: a fault on
    /// a low-redundancy satellite is spread onto the others' residuals and can
    /// leave a healthy satellite here, which is why [`fde`] chooses exclusions
    /// by re-solving without each candidate instead.
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
///
/// The residuals, their variances and the clock count come from the solution.
/// An explicit [`RaimOptions::n_systems`] overrides the clock count.
pub fn raim_for_solution<S: RaimSolution + ?Sized>(
    solution: &S,
    options: &RaimOptions,
) -> Result<RaimResult, QualityError> {
    let used_sats = solution.raim_used_sats();
    let n_systems = match (options.n_systems, solution.raim_clock_count()) {
        (Some(n_systems), _) => Some(n_systems),
        (None, Some(count)) => Some(isize::try_from(count).unwrap_or(isize::MAX)),
        (None, None) => None,
    };
    raim_checked(
        &used_sats,
        solution.raim_residuals_m(),
        solution.raim_variances_m2(),
        options,
        n_systems,
    )
}

/// Residual-based chi-square RAIM.
///
/// With `dof = n_used - (3 + n_systems)` redundancy, a fault is declared when
/// the weighted residual sum of squares exceeds the `1 - p_fa` chi-square
/// quantile at `dof`. Under the default [`RaimWeights::Solution`] the statistic
/// is RTKLIB demo5 `valsol`'s `sum (v / sigma)^2` over the estimator's own
/// variances. The threshold is the exact quantile, which RTKLIB's `chisqr`
/// table lists rounded to one decimal.
pub fn raim(input: &RaimInput, options: &RaimOptions) -> Result<RaimResult, QualityError> {
    raim_checked(
        &input.used_sats,
        &input.residuals_m,
        input.variances_m2.as_deref(),
        options,
        options.n_systems,
    )
}

fn raim_checked(
    used_sats: &[String],
    residuals_m: &[f64],
    variances_m2: Option<&[f64]>,
    options: &RaimOptions,
    n_systems: Option<isize>,
) -> Result<RaimResult, QualityError> {
    validate_probability(options.p_fa)?;
    options.weights.validate()?;
    validate_raim_input(used_sats, residuals_m, None)?;
    // Variances are read, and so validated, only under `Solution`.
    let variances_m2 = match options.weights {
        RaimWeights::Solution => Some(variances_m2.ok_or(QualityError::MissingVariances)?),
        RaimWeights::Unit | RaimWeights::BySatellite(_) => None,
    };
    validate_raim_input(used_sats, residuals_m, variances_m2)?;

    let n_used = used_sats.len() as isize;
    let n_systems = match n_systems {
        Some(n_systems) if n_systems >= 1 => n_systems,
        Some(_) => return Err(QualityError::InvalidSystemCount),
        None => distinct_systems(used_sats),
    };
    let dof = n_used - (3 + n_systems);

    let mut test_statistic = 0.0;
    let mut normalized_residuals = BTreeMap::new();
    let mut worst_sat = None::<String>;
    let mut worst_abs = f64::NEG_INFINITY;

    for (index, (satellite_id, residual_m)) in used_sats.iter().zip(residuals_m).enumerate() {
        let normalized = match (&options.weights, variances_m2) {
            (RaimWeights::Solution, Some(variances_m2)) => {
                // RTKLIB `estpos` divides each residual by `sqrt(var)` and
                // `valsol` sums the squares of the quotients.
                let normalized = residual_m / variances_m2[index].sqrt();
                test_statistic += normalized * normalized;
                normalized
            }
            (weights, _) => {
                let weight = match weights {
                    RaimWeights::BySatellite(weights) => {
                        weights.get(satellite_id).copied().unwrap_or(1.0)
                    }
                    RaimWeights::Solution | RaimWeights::Unit => 1.0,
                };
                test_statistic += residual_m * residual_m * weight;
                residual_m * weight.sqrt()
            }
        };
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

fn validate_raim_input(
    used_sats: &[String],
    residuals_m: &[f64],
    variances_m2: Option<&[f64]>,
) -> Result<(), QualityError> {
    if used_sats.len() != residuals_m.len() {
        return Err(QualityError::InvalidResiduals);
    }
    validate::finite_slice(residuals_m, "raim residuals")
        .map_err(|_| QualityError::InvalidResiduals)?;
    if let Some(variances_m2) = variances_m2 {
        if variances_m2.len() != residuals_m.len() {
            return Err(QualityError::InvalidVariance);
        }
        variances_m2
            .iter()
            .try_for_each(|v| validate::finite_positive(*v, "raim variance").map(|_| ()))
            .map_err(|_| QualityError::InvalidVariance)?;
    }
    Ok(())
}

fn validate_weights_slice(weights: &[f64]) -> Result<(), QualityError> {
    weights
        .iter()
        .try_for_each(|w| validate::finite_positive(*w, "diagnostic weight").map(|_| ()))
        .map_err(|_| QualityError::InvalidWeight)
}

fn distinct_systems(used_sats: &[String]) -> isize {
    used_sats
        .iter()
        .filter_map(|sat| sat.chars().next())
        .collect::<BTreeSet<_>>()
        .len() as isize
}

/// The fewest observations for which a failed full-set solve is searched for an
/// exclusion, RTKLIB demo5 `pntpos`'s `n >= 6` condition on `raim_fde`.
pub const FDE_MIN_OBSERVATIONS: usize = 6;

/// The fewest satellites a leave-one-out re-solve may use and still be kept as
/// an exclusion candidate, RTKLIB demo5 `raim_fde`'s `nvsat < 5` floor.
pub const FDE_MIN_CANDIDATE_SATELLITES: usize = 5;

/// The largest residual RMS, metres, an exclusion may leave: RTKLIB demo5
/// `raim_fde` starts its search from `rms = 100.0` and keeps a candidate only
/// when its RMS does not exceed the best so far.
pub const DEFAULT_FDE_MAX_EXCLUSION_RMS_M: f64 = 100.0;

/// Result of a fault-detection-and-exclusion loop.
#[derive(Debug, Clone, PartialEq)]
pub struct FdeResult<S> {
    /// Final accepted solution.
    pub solution: S,
    /// Excluded satellites in exclusion order.
    pub excluded: Vec<String>,
    /// Number of exclusions performed, `excluded.len()`.
    pub iterations: usize,
    /// The detection test of the accepted solution. `testable` is false when
    /// the accepted set has no redundancy left to test.
    pub raim: RaimResult,
}

/// Why [`fde`] ended with a fault still detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum FdeUnresolvedReason {
    /// The exclusion budget ([`FdeOptions::max_exclusions`]) was spent.
    ExclusionBudgetExhausted,
    /// No leave-one-out re-solve was admissible: every candidate failed to
    /// solve, used fewer than [`FDE_MIN_CANDIDATE_SATELLITES`] satellites, or
    /// left a residual RMS above [`FdeOptions::max_exclusion_rms_m`].
    NoAdmissibleExclusion,
}

/// The state [`fde`] stopped in with a fault still detected: the last solution,
/// the exclusions made to reach it and its detection test.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FdeUnresolved<S> {
    /// Why the loop stopped.
    pub reason: FdeUnresolvedReason,
    /// The last solution, which still fails the detection test.
    pub solution: S,
    /// Excluded satellites in exclusion order.
    pub excluded: Vec<String>,
    /// The detection test of `solution`, with `fault_detected` true.
    pub raim: RaimResult,
}

/// Error from [`fde`].
#[derive(Debug, Clone, PartialEq)]
pub enum FdeError<S, E> {
    /// A fault was still detected when the loop stopped. The payload carries
    /// the last solution, the exclusions made and its detection test.
    FaultUnresolved(Box<FdeUnresolved<S>>),
    /// The supplied solve callback failed on the full observation set, and
    /// the failure was an input error or no leave-one-out re-solve cured it.
    Solve(E),
    /// RAIM configuration was invalid.
    Raim(QualityError),
}

/// Options for [`fde`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FdeOptions {
    /// RAIM options used to test each accepted solution.
    pub raim: RaimOptions,
    /// Maximum number of exclusions. The default, 1, is RTKLIB demo5's single
    /// `raim_fde` exclusion. A larger budget repeats detection and a fresh
    /// leave-one-out search after each exclusion, removing one satellite per
    /// round from the set that remains.
    pub max_exclusions: usize,
    /// The largest residual RMS, metres, an exclusion may leave; default
    /// [`DEFAULT_FDE_MAX_EXCLUSION_RMS_M`], RTKLIB demo5's initial `rms`.
    pub max_exclusion_rms_m: f64,
}

impl FdeOptions {
    /// Build FDE options from the RAIM policy and exclusion budget, with the
    /// RTKLIB exclusion RMS cap.
    #[must_use]
    pub const fn new(raim: RaimOptions, max_exclusions: usize) -> Self {
        Self {
            raim,
            max_exclusions,
            max_exclusion_rms_m: DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
        }
    }
}

impl Default for FdeOptions {
    /// Default RAIM options and RTKLIB demo5's single exclusion.
    fn default() -> Self {
        Self::new(RaimOptions::default(), 1)
    }
}

/// How a failed solve of the full observation set is treated by [`fde`].
///
/// RTKLIB demo5 `pntpos` calls `raim_fde` when `estpos` fails: when the
/// iteration diverges, the least-squares step is singular, or `valsol` refuses
/// the solution's geometry. A solve error of that kind describes the
/// measurement set, and removing a satellite can cure it; an input error
/// cannot be cured that way and is returned as [`FdeError::Solve`].
pub trait FdeSolveFailure {
    /// True when the failure is one a leave-one-out search may cure: a
    /// non-converging or singular solve, or a solution refused for its
    /// geometry or plausibility.
    fn admits_exclusion_search(&self) -> bool;
}

/// Fault detection and exclusion over a caller-supplied SPP solver.
///
/// # Algorithm
///
/// 1. Solve the full observation set and test it with [`raim_for_solution`]
///    under [`FdeOptions::raim`] (by default the chi-square of the residuals
///    over the estimator's own variances, RTKLIB demo5 `valsol`). A set that
///    passes, or has no redundancy to test, is accepted.
/// 2. On a detected fault, choose the exclusion by RTKLIB demo5 `raim_fde`'s
///    leave-one-out rule (`pntpos.c`): each satellite of the flagged solution is
///    left out in turn, in RTKLIB's satellite-number order (GPS, GLONASS,
///    Galileo, QZSS, BeiDou, NavIC, SBAS, each by PRN), and the rest are
///    re-solved with `solve`. A candidate whose re-solve fails or uses fewer
///    than [`FDE_MIN_CANDIDATE_SATELLITES`] satellites is skipped. Of the
///    others, the one whose unweighted post-fit residual RMS
///    `sqrt(sum r^2 / n)`, summed in that satellite order, is smallest is
///    excluded, provided it is no larger than
///    [`FdeOptions::max_exclusion_rms_m`]; equal RMS values go to the later
///    candidate, as RTKLIB's `if (rms_e > rms) continue;` does.
/// 3. When the full-set solve fails in a way [`FdeSolveFailure`] admits and at
///    least [`FDE_MIN_OBSERVATIONS`] observations were supplied, which is when
///    demo5 `pntpos` calls `raim_fde`, the same search runs over every observed
///    satellite. When no candidate is admissible, or the failure is an input
///    error, the failure is returned as [`FdeError::Solve`].
/// 4. Test the chosen re-solve. It is accepted when it passes. Otherwise, while
///    [`FdeOptions::max_exclusions`] allows, step 2 repeats on the remaining
///    set; when the budget is spent, or no candidate is admissible, the loop
///    ends with [`FdeError::FaultUnresolved`] carrying the last solution.
///
/// The residual with the largest weighted value is never used to pick the
/// exclusion: a fault on a satellite the geometry leans on is spread onto the
/// other residuals, so that choice can remove healthy satellites and keep the
/// faulty one. Re-solving without each candidate removes the fault whenever
/// the geometry can identify it.
///
/// # Differences from RTKLIB demo5
///
/// - demo5 has `valsol`'s chi-square rejection commented out, so `pntpos` calls
///   `raim_fde` only when `estpos` fails. This function is an explicit
///   fault-detection API: it also runs the chi-square test demo5 computes and
///   reports, and excludes when that test fails. The threshold is the exact
///   `1 - p_fa` chi-square quantile; demo5's `chisqr` table rounds it to one
///   decimal, repeats values past 60 degrees of freedom and ends at 100.
/// - demo5 starts its first candidate from the geocentre (`sol_e` is
///   zero-initialized) and every later candidate from the position the last
///   converged candidate reached, including one `valsol` then refused for its
///   GDOP, since `estpos` writes the position before validating it. `solve`
///   starts wherever the caller's solver starts. Those starts can reach
///   different solutions; the RTKLIB oracle records the candidate RMS and state,
///   and the comparison bounds RMS differences from the measured state distance
///   and RTKLIB's final least-squares step.
/// - After a detected fault only the satellites the flagged solution used are
///   candidates. Leaving out a satellite it did not use returns the same
///   flagged solution, which is no exclusion at all; in demo5's own path the
///   full-set `estpos` failed, so that candidate repeats the failure and is
///   skipped. After a failed full-set solve every observed satellite is a
///   candidate, as in demo5.
pub fn fde<S, E, F>(
    observations: &[Observation],
    options: &FdeOptions,
    mut solve: F,
) -> Result<FdeResult<S>, FdeError<S, E>>
where
    S: RaimSolution,
    E: FdeSolveFailure,
    F: FnMut(&[Observation]) -> Result<S, E>,
{
    validate_exclusion_rms_cap(options.max_exclusion_rms_m).map_err(FdeError::Raim)?;
    let mut remaining = observations.to_vec();
    let mut excluded = Vec::new();
    let mut solution = match solve(&remaining) {
        Ok(solution) => solution,
        Err(error) => {
            if !error.admits_exclusion_search()
                || remaining.len() < FDE_MIN_OBSERVATIONS
                || options.max_exclusions == 0
            {
                return Err(FdeError::Solve(error));
            }
            let candidates = rtklib_candidates(&remaining, |_| true);
            let Some((left_out, candidate)) =
                leave_one_out_exclusion(&remaining, &candidates, options, &mut solve)
            else {
                return Err(FdeError::Solve(error));
            };
            remaining.retain(|ob| ob.satellite_id != left_out);
            excluded.push(left_out.to_string());
            candidate
        }
    };

    loop {
        let raim = raim_for_solution(&solution, &options.raim).map_err(FdeError::Raim)?;
        if !raim.fault_detected {
            return Ok(FdeResult {
                solution,
                iterations: excluded.len(),
                excluded,
                raim,
            });
        }

        if excluded.len() >= options.max_exclusions {
            return Err(fault_unresolved(
                FdeUnresolvedReason::ExclusionBudgetExhausted,
                solution,
                excluded,
                raim,
            ));
        }

        let used: BTreeSet<String> = solution.raim_used_sats().into_iter().collect();
        let candidates = rtklib_candidates(&remaining, |satellite| {
            used.contains(&satellite.to_string())
        });
        let Some((left_out, candidate)) =
            leave_one_out_exclusion(&remaining, &candidates, options, &mut solve)
        else {
            return Err(fault_unresolved(
                FdeUnresolvedReason::NoAdmissibleExclusion,
                solution,
                excluded,
                raim,
            ));
        };

        remaining.retain(|ob| ob.satellite_id != left_out);
        excluded.push(left_out.to_string());
        solution = candidate;
    }
}

/// The satellites of `remaining` that `keep` admits, once each, in RTKLIB's
/// satellite-number order.
fn rtklib_candidates(
    remaining: &[Observation],
    keep: impl Fn(GnssSatelliteId) -> bool,
) -> Vec<GnssSatelliteId> {
    let mut candidates: Vec<GnssSatelliteId> = remaining
        .iter()
        .map(|ob| ob.satellite_id)
        .filter(|satellite| keep(*satellite))
        .collect();
    candidates.sort_by_key(|satellite| rtklib_satellite_order(*satellite));
    candidates.dedup();
    candidates
}

/// RTKLIB's leave-one-out choice over `candidates`: each is left out of
/// `remaining` in turn and the rest re-solved with `solve`.
fn leave_one_out_exclusion<S, E, F>(
    remaining: &[Observation],
    candidates: &[GnssSatelliteId],
    options: &FdeOptions,
    solve: &mut F,
) -> Option<(GnssSatelliteId, S)>
where
    S: RaimSolution,
    F: FnMut(&[Observation]) -> Result<S, E>,
{
    let mut subset = Vec::with_capacity(remaining.len());
    rtklib_leave_one_out(
        candidates.len(),
        FDE_MIN_CANDIDATE_SATELLITES,
        options.max_exclusion_rms_m,
        |index| {
            let left_out = candidates[index];
            subset.clear();
            subset.extend(
                remaining
                    .iter()
                    .filter(|ob| ob.satellite_id != left_out)
                    .cloned(),
            );
            let candidate = solve(&subset).ok()?;
            let used_sats = candidate.raim_used_sats();
            Some(LeaveOneOutFit {
                used: used_sats.len(),
                rms_m: rtklib_order_rms_m(&used_sats, candidate.raim_residuals_m()),
                fit: candidate,
            })
        },
    )
    .map(|(index, candidate)| (candidates[index], candidate))
}

/// RTKLIB `raim_fde`'s residual RMS, `sqrt(sum r^2 / n)`, with the squares
/// summed in RTKLIB's satellite-number order, the order `raim_fde` sums them.
/// Tokens that do not name a satellite are summed in the given order.
fn rtklib_order_rms_m(used_sats: &[String], residuals_m: &[f64]) -> f64 {
    let ids: Option<Vec<GnssSatelliteId>> = used_sats
        .iter()
        .map(|token| token.parse::<GnssSatelliteId>().ok())
        .collect();
    let mut order: Vec<usize> = (0..residuals_m.len()).collect();
    if let Some(ids) = ids.filter(|ids| ids.len() == residuals_m.len()) {
        order.sort_by_key(|&index| rtklib_satellite_order(ids[index]));
    }
    if order.is_empty() {
        return 0.0;
    }
    let mut sum_sq = 0.0;
    for index in order {
        sum_sq += residuals_m[index] * residuals_m[index];
    }
    (sum_sq / residuals_m.len() as f64).sqrt()
}

fn fault_unresolved<S, E>(
    reason: FdeUnresolvedReason,
    solution: S,
    excluded: Vec<String>,
    raim: RaimResult,
) -> FdeError<S, E> {
    FdeError::FaultUnresolved(Box::new(FdeUnresolved {
        reason,
        solution,
        excluded,
        raim,
    }))
}

/// A leave-one-out re-solve: the fit, how many measurements it used and the
/// RMS of its unweighted post-fit residuals.
struct LeaveOneOutFit<C> {
    fit: C,
    used: usize,
    rms_m: f64,
}

/// RTKLIB demo5 `raim_fde`'s choice among leave-one-out re-solves (`pntpos.c`).
///
/// `evaluate(i)` re-solves without candidate `i`, in the caller's candidate
/// order, returning `None` when that solve fails. A fit that used fewer than
/// `min_used` measurements is skipped (`nvsat < 5`). The kept fit is the one with
/// the smallest RMS, starting from `max_rms_m` (RTKLIB's `rms = 100.0`); a
/// candidate whose RMS equals the best so far replaces it, as RTKLIB's
/// `if (rms_e > rms) continue;` gives ties to the later candidate. Returns the
/// chosen candidate's index and fit.
fn rtklib_leave_one_out<C>(
    candidate_count: usize,
    min_used: usize,
    max_rms_m: f64,
    mut evaluate: impl FnMut(usize) -> Option<LeaveOneOutFit<C>>,
) -> Option<(usize, C)> {
    let mut best = None;
    let mut best_rms_m = max_rms_m;
    for index in 0..candidate_count {
        let Some(candidate) = evaluate(index) else {
            continue;
        };
        if candidate.used < min_used {
            continue;
        }
        // A NaN RMS never qualifies (RTKLIB's `>` test would keep it).
        if candidate.rms_m.is_nan() || candidate.rms_m > best_rms_m {
            continue;
        }
        best_rms_m = candidate.rms_m;
        best = Some((index, candidate.fit));
    }
    best
}

/// RTKLIB's satellite-number order (`satno`): GPS, GLONASS, Galileo, QZSS,
/// BeiDou, NavIC, SBAS, each by PRN. `raim_fde` leaves satellites out in this
/// order, which decides ties.
fn rtklib_satellite_order(satellite: GnssSatelliteId) -> (u8, u8) {
    let system = match satellite.system {
        GnssSystem::Gps => 0,
        GnssSystem::Glonass => 1,
        GnssSystem::Galileo => 2,
        GnssSystem::Qzss => 3,
        GnssSystem::BeiDou => 4,
        GnssSystem::Navic => 5,
        GnssSystem::Sbas => 6,
    };
    (system, satellite.prn)
}

fn validate_exclusion_rms_cap(max_rms_m: f64) -> Result<(), QualityError> {
    if max_rms_m.is_nan() || max_rms_m <= 0.0 {
        Err(QualityError::InvalidParameter)
    } else {
        Ok(())
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

impl FdeSolveFailure for FdeSppError {
    /// A solve that did not settle or hit singular geometry, and a solution
    /// refused as rank deficient, over the PDOP ceiling, implausibly placed or
    /// with an implausible residual RMS, admit the search; input, duplicate,
    /// ephemeris, UT1 and too-few-satellite failures, and invalid options, do
    /// not.
    fn admits_exclusion_search(&self) -> bool {
        match self {
            Self::Spp(error) => matches!(
                error,
                SppError::Singular(crate::astro::math::least_squares::SolveError::SingularJacobian)
                    | SppError::SelectionUnsettled { .. }
            ),
            Self::Validation(error) => matches!(
                error,
                SolutionValidationError::DegenerateGeometryRankDeficient
                    | SolutionValidationError::DegenerateGeometryPdop(_)
                    | SolutionValidationError::ImplausiblePosition(_)
                    | SolutionValidationError::NoConvergence(_)
            ),
        }
    }
}

/// Options for [`fde_spp`]: the RAIM-gated exclusion loop plus the per-iteration
/// solution-validation gates applied to each candidate solve.
#[derive(Debug, Clone, PartialEq, Default)]
#[non_exhaustive]
pub struct FdeSppOptions {
    /// FDE loop options: the RAIM configuration and the exclusion budget.
    pub fde: FdeOptions,
    /// Per-iteration solution-validation gates (PDOP ceiling and plausibility
    /// band) applied to each candidate solution.
    pub validation: SolutionValidationOptions,
}

impl FdeSppOptions {
    /// Build SPP FDE options from the exclusion and validation policies.
    #[must_use]
    pub const fn new(fde: FdeOptions, validation: SolutionValidationOptions) -> Self {
        Self { fde, validation }
    }
}

/// Run single-point positioning with RAIM fault detection and exclusion.
///
/// Solves [`solve`] over the input observation set and runs [`fde`] over it:
/// the residual chi-square test over the solve's own pseudorange variances
/// ([`ReceiverSolution::pseudorange_variances_m2`]) and, on a detected fault,
/// RTKLIB demo5 `raim_fde`'s leave-one-out exclusion, within the budget in
/// [`FdeSppOptions::fde`] (one exclusion by default). Every solve, the first and
/// each leave-one-out re-solve, is screened with [`validate_receiver_solution`]
/// using [`FdeSppOptions::validation`]; a re-solve that fails the screen is not
/// an exclusion candidate, as RTKLIB skips a candidate whose `estpos` fails. On
/// success returns the protected [`FdeResult`]: the surviving
/// [`ReceiverSolution`], the excluded satellite tokens in exclusion order, the
/// exclusion count and the accepted solution's detection test.
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
) -> Result<FdeResult<ReceiverSolution>, FdeError<ReceiverSolution, FdeSppError>> {
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

/// Run robust-reweighted SPP under the RAIM/FDE exclusion loop.
///
/// This is the robust-specific composition of [`RobustConfig`] and [`fde_spp`].
/// It clones the inputs, installs `robust`, then delegates to [`fde_spp`], which
/// delegates each candidate solve to [`solve`] and each exclusion step to
/// [`fde`]. Detection standardizes the residuals by the pseudorange variances,
/// not by the Huber-reduced weights of the robust solve, so a satellite the
/// reweighting has discounted still counts at its modelled variance.
pub fn spp_robust_fde_driver(
    eph: &dyn EphemerisSource,
    inputs: &SolveInputs,
    with_geodetic: bool,
    robust: RobustConfig,
    options: &FdeSppOptions,
) -> Result<FdeResult<ReceiverSolution>, FdeError<ReceiverSolution, FdeSppError>> {
    let mut robust_inputs = inputs.clone();
    robust_inputs.robust = Some(robust);
    fde_spp(eph, &robust_inputs, with_geodetic, options)
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
#[non_exhaustive]
pub struct RangeFdeOptions {
    /// False-alarm probability for the global chi-square test. The detection
    /// threshold is the `1 - p_fa` chi-square quantile at the redundancy
    /// (degrees of freedom). RTKLIB demo5 uses `p_fa = 1.0e-3`.
    pub p_fa: f64,
    /// Maximum number of measurements the exclusion loop may remove. The
    /// default, 1, is RTKLIB demo5's single `raim_fde` exclusion; a larger budget
    /// repeats the test and a fresh leave-one-out search after each exclusion.
    pub max_exclusions: usize,
    /// Minimum redundancy (degrees of freedom) that an exclusion must leave
    /// behind. A leave-one-out candidate is kept only when the surviving set
    /// still has at least `min_redundancy` more measurements than state
    /// parameters, so the protected set stays testable. RTKLIB demo5's
    /// `nvsat >= 5` floor for a four-state solve is `min_redundancy == 1`.
    pub min_redundancy: usize,
    /// The largest unweighted post-fit residual RMS, metres, an exclusion may
    /// leave; default [`DEFAULT_FDE_MAX_EXCLUSION_RMS_M`], RTKLIB demo5's initial
    /// `rms`.
    pub max_exclusion_rms_m: f64,
}

impl Default for RangeFdeOptions {
    fn default() -> Self {
        Self {
            p_fa: DEFAULT_P_FA,
            max_exclusions: 1,
            min_redundancy: 1,
            max_exclusion_rms_m: DEFAULT_FDE_MAX_EXCLUSION_RMS_M,
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
/// 3. FDE exclusion (RTKLIB demo5 `raim_fde`, `pntpos.c`, the same rule
///    [`fde`] applies): while a fault is detected and the exclusion budget
///    allows, each active measurement is removed in turn, in input order, and
///    the set is re-solved. A candidate that leaves fewer than `n_state +
///    min_redundancy` measurements, or whose re-solve is singular, is skipped.
///    Of the others, the one whose unweighted post-fit residual RMS
///    `sqrt(sum v^2 / n)` is smallest is excluded, provided it does not exceed
///    [`RangeFdeOptions::max_exclusion_rms_m`]; equal RMS values go to the later
///    measurement. The default budget is one exclusion, as in RTKLIB; a larger
///    budget repeats the test and the search on the remaining set, stopping when
///    the test passes, the budget is spent, or no candidate is admissible.
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
    validate_exclusion_rms_cap(options.max_exclusion_rms_m)?;
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

        let Some((slot, candidate_fit)) = best_range_exclusion(
            rows,
            &active,
            n_state,
            options.min_redundancy,
            options.max_exclusion_rms_m,
        ) else {
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

/// The RTKLIB demo5 `raim_fde` exclusion over the active rows. Returns the slot
/// (index into `active`) and the re-solved fit, or `None` when no candidate is
/// admissible.
fn best_range_exclusion(
    rows: &[RangeFdeRow],
    active: &[usize],
    n_state: usize,
    min_redundancy: usize,
    max_rms_m: f64,
) -> Option<(usize, WlsFit)> {
    let mut remaining: Vec<usize> = Vec::with_capacity(active.len());
    rtklib_leave_one_out(
        active.len(),
        n_state.saturating_add(min_redundancy),
        max_rms_m,
        |slot| {
            remaining.clear();
            remaining.extend(
                active
                    .iter()
                    .enumerate()
                    .filter(|&(s, _)| s != slot)
                    .map(|(_, &idx)| idx),
            );
            let fit = solve_range_wls(rows, &remaining, n_state).ok()?;
            let residuals_m: Vec<f64> = remaining
                .iter()
                .map(|&idx| rows[idx].residual_m - dot(&rows[idx].design_row, &fit.dx))
                .collect();
            Some(LeaveOneOutFit {
                used: remaining.len(),
                rms_m: residual_rms(&residuals_m),
                fit,
            })
        },
    )
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
#[non_exhaustive]
pub struct SolutionValidationOptions {
    /// Optional PDOP ceiling.
    pub max_pdop: Option<f64>,
    /// Minimum plausible geocentric radius, meters.
    pub min_plausible_radius_m: f64,
    /// Maximum plausible geocentric radius, meters.
    pub max_plausible_radius_m: f64,
    /// Maximum plausible post-fit residual RMS, meters, checked on every
    /// solution whether or not its solve converged (the name is kept from when
    /// only converged solutions were checked).
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
    /// Solution residuals were non-finite or produced non-finite RMS.
    InvalidResiduals,
    /// The solution had physically implausible post-fit residual RMS.
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
                write!(f, "solution residuals must be finite")
            }
            Self::NoConvergence(rms_m) => {
                write!(f, "solution residual RMS {rms_m} m is implausibly large")
            }
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

    // The residuals are checked whether or not the solve converged: a solve that
    // did not is no more plausible for it.
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
        sum * libm::exp(-x + a * libm::log(x) - gln)
    } else {
        let gln = log_gamma(a);
        let q = gamma_continued_fraction(a, x) * libm::exp(-x + a * libm::log(x) - gln);
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
        libm::log(std::f64::consts::PI)
            - libm::log(libm::sin(std::f64::consts::PI * z))
            - log_gamma(1.0 - z)
    } else {
        let z = z - 1.0;
        let mut x = LANCZOS[0];
        for (i, coef) in LANCZOS.iter().enumerate().skip(1) {
            x += coef / (z + i as f64);
        }
        let t = z + 7.5;
        libm::log(SQRT_2PI) + (z + 0.5) * libm::log(t) - t + libm::log(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GnssSatelliteId, GnssSystem};

    use std::path::PathBuf;

    use crate::rinex_nav::BroadcastStore;
    use crate::rinex_obs::{pseudoranges, RinexObs, SignalPolicy};
    use crate::spp::{Corrections, KlobucharCoeffs, RobustConfig, SurfaceMet};

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
            sbas_iono: None,
            glonass_channels: std::collections::BTreeMap::new(),
            met: SurfaceMet {
                pressure_hpa: 1013.25,
                temperature_k: 288.15,
                relative_humidity: 0.5,
            },
            robust: None,
            pseudorange_code: crate::spp::PseudorangeCode::SingleFrequency,
            qzss_clock: crate::spp::QzssClock::Gps,
            troposphere_model: crate::spp::TroposphereModel::Rtklib,
        }
    }

    fn assert_receiver_solution_bits_eq(left: &ReceiverSolution, right: &ReceiverSolution) {
        assert_eq!(left.position.x_m.to_bits(), right.position.x_m.to_bits());
        assert_eq!(left.position.y_m.to_bits(), right.position.y_m.to_bits());
        assert_eq!(left.position.z_m.to_bits(), right.position.z_m.to_bits());
        assert_eq!(left.geodetic, right.geodetic);
        assert_eq!(left.rx_clock_s.to_bits(), right.rx_clock_s.to_bits());
        assert_eq!(left.rx_clock_drift_s_s, right.rx_clock_drift_s_s);
        assert_eq!(left.dop, right.dop);
        assert_eq!(
            left.position_covariance
                .ecef_m2
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            right
                .position_covariance
                .ecef_m2
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            left.position_covariance
                .enu_m2
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            right
                .position_covariance
                .enu_m2
                .iter()
                .flatten()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
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
        for (left, right) in [
            (
                &left.pseudorange_variances_m2,
                &right.pseudorange_variances_m2,
            ),
            (&left.weights, &right.weights),
        ] {
            assert_eq!(
                left.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                right.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
        assert_eq!(left.used_sats, right.used_sats);
        assert_eq!(left.rejected_sats, right.rejected_sats);
        assert_eq!(left.metadata, right.metadata);
    }

    /// The default single-exclusion options: RAIM over the solve's own
    /// variances at `p_fa = 1e-3` and RTKLIB's one exclusion.
    fn fde_spp_options() -> FdeSppOptions {
        FdeSppOptions::default()
    }

    /// Solve `inputs` without `satellite`, as the leave-one-out search does.
    fn solve_without(
        eph: &dyn EphemerisSource,
        inputs: &SolveInputs,
        satellite: GnssSatelliteId,
        with_geodetic: bool,
    ) -> ReceiverSolution {
        let mut subset = inputs.clone();
        subset
            .observations
            .retain(|ob| ob.satellite_id != satellite);
        solve(eph, &subset, with_geodetic).expect("leave-one-out solve")
    }

    fn position_delta_m(left: &ReceiverSolution, right: &ReceiverSolution) -> f64 {
        ((left.position.x_m - right.position.x_m).powi(2)
            + (left.position.y_m - right.position.y_m).powi(2)
            + (left.position.z_m - right.position.z_m).powi(2))
        .sqrt()
    }

    /// The `fde_spp` driver must equal the hand-assembled solve + validate + FDE
    /// loop the bindings each spell out, bit-for-bit, on a real converging
    /// scenario with one injected outlier, and that loop must remove exactly the
    /// faulted satellite: the protected solution is the solve without it.
    #[test]
    fn fde_spp_matches_manual_composition_and_removes_the_faulted_satellite() {
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

        let mut inputs = clean_inputs;
        inputs
            .observations
            .iter_mut()
            .find(|obs| obs.satellite_id == outlier_sat)
            .expect("outlier satellite is present in the observation set")
            .pseudorange_m += 1000.0;

        let options = fde_spp_options();

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

        assert_eq!(driver.excluded, reference.excluded);
        assert_eq!(driver.iterations, reference.iterations);
        assert_eq!(driver.raim, reference.raim);
        assert_receiver_solution_bits_eq(&driver.solution, &reference.solution);

        assert_eq!(driver.excluded, vec![outlier_sat.to_string()]);
        assert_eq!(driver.iterations, 1);
        assert!(!driver.raim.fault_detected);
        assert!(driver.raim.testable);
        assert_receiver_solution_bits_eq(
            &driver.solution,
            &solve_without(&store, &inputs, outlier_sat, with_geodetic),
        );
    }

    /// A clean set converges with no exclusion, and the driver still equals the
    /// hand-assembled composition bit-for-bit.
    #[test]
    fn fde_spp_clean_set_takes_no_exclusion_and_matches_manual() {
        let store = esbc_broadcast_store();
        let inputs = esbc_first_epoch_inputs();
        let options = fde_spp_options();

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
        assert!(!driver.raim.fault_detected);
        assert_eq!(driver.iterations, reference.iterations);
        assert_eq!(driver.excluded, reference.excluded);
        assert_receiver_solution_bits_eq(&driver.solution, &reference.solution);
        assert_receiver_solution_bits_eq(
            &driver.solution,
            &solve(&store, &inputs, false).expect("clean solve"),
        );
    }

    #[test]
    fn spp_robust_fde_driver_clean_set_uses_robust_solve_without_exclusion() {
        let store = esbc_broadcast_store();
        let inputs = esbc_first_epoch_inputs();
        let options = fde_spp_options();

        let driver =
            spp_robust_fde_driver(&store, &inputs, false, RobustConfig::default(), &options)
                .expect("robust FDE solves clean set");

        assert_eq!(driver.iterations, 0);
        assert!(driver.excluded.is_empty());
        assert!(driver.solution.metadata.outer_iterations > 0);
        assert!(driver.solution.metadata.final_robust_scale_m.is_some());
        let surviving = raim_for_solution(&driver.solution, &options.fde.raim).expect("raim");
        assert!(!surviving.fault_detected);
    }

    #[test]
    fn spp_robust_fde_driver_excludes_fault_and_recovers_solution() {
        let store = esbc_broadcast_store();
        let clean_inputs = esbc_first_epoch_inputs();
        let clean_options = fde_spp_options();
        let robust = RobustConfig::default();
        let clean = spp_robust_fde_driver(&store, &clean_inputs, false, robust, &clean_options)
            .expect("clean robust FDE solve");
        let outlier_sat = gps(15);
        assert!(clean.solution.used_sats.contains(&outlier_sat));

        let mut faulty_inputs = clean_inputs.clone();
        let outlier_obs = faulty_inputs
            .observations
            .iter_mut()
            .find(|obs| obs.satellite_id == outlier_sat)
            .expect("outlier satellite is observed");
        outlier_obs.pseudorange_m += 1000.0;
        let faulty_options = fde_spp_options();

        let driver = spp_robust_fde_driver(&store, &faulty_inputs, false, robust, &faulty_options)
            .expect("robust FDE resolves fault");

        assert_eq!(driver.iterations, 1);
        assert_eq!(driver.iterations, driver.excluded.len());
        assert_eq!(driver.excluded, vec![outlier_sat.to_string()]);
        assert!(driver.solution.metadata.outer_iterations > 0);
        assert!(driver.solution.metadata.final_robust_scale_m.is_some());
        let surviving = raim_for_solution(&driver.solution, &faulty_options.fde.raim)
            .expect("surviving set RAIM");
        assert!(!surviving.fault_detected);
        let recovered_delta_m = position_delta_m(&driver.solution, &clean.solution);
        assert!(
            recovered_delta_m < 1.0,
            "protected solution should stay close to the clean robust solution, got {recovered_delta_m} m with exclusions {:?}",
            driver.excluded
        );
    }

    /// A solve failure for the synthetic FDE tests: an input error, or a solve
    /// that did not settle.
    #[derive(Debug, Clone, PartialEq)]
    enum TestSolveError {
        Input,
        Unsettled,
    }

    impl FdeSolveFailure for TestSolveError {
        fn admits_exclusion_search(&self) -> bool {
            matches!(self, Self::Unsettled)
        }
    }

    #[derive(Debug, Clone, PartialEq)]
    struct TestSolution {
        used_sats: Vec<String>,
        residuals_m: Vec<f64>,
        variances_m2: Vec<f64>,
    }

    impl TestSolution {
        /// A solution over `remaining` whose residuals `residual_m` assigns,
        /// each with unit variance.
        fn over(remaining: &[Observation], residual_m: impl Fn(GnssSatelliteId) -> f64) -> Self {
            Self {
                used_sats: remaining
                    .iter()
                    .map(|ob| ob.satellite_id.to_string())
                    .collect(),
                residuals_m: remaining
                    .iter()
                    .map(|ob| residual_m(ob.satellite_id))
                    .collect(),
                variances_m2: vec![1.0; remaining.len()],
            }
        }
    }

    impl RaimSolution for TestSolution {
        fn raim_used_sats(&self) -> Vec<String> {
            self.used_sats.clone()
        }

        fn raim_residuals_m(&self) -> &[f64] {
            &self.residuals_m
        }

        fn raim_variances_m2(&self) -> Option<&[f64]> {
            Some(&self.variances_m2)
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
            rx_clock_drift_s_s: None,
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
            position_covariance: crate::dop::PositionCovariance {
                ecef_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                enu_m2: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            },
            residuals_m: vec![0.1, -0.1, 0.0, 0.05, -0.05],
            pseudorange_variances_m2: vec![1.0; 5],
            weights: vec![1.0; 5],
            used_sats: (1..=5).map(gps).collect(),
            rejected_sats: Vec::new(),
            geometry_quality: crate::geometry_quality::GeometryQuality {
                tier: crate::geometry_quality::ObservabilityTier::Nominal,
                redundancy: 1,
                rank: 4,
                condition_number: 1.0,
                gdop: 2.5,
                raim_checkable: true,
                covariance_validated: true,
            },
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
                ut1_degraded: None,
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
            variances_m2: Some(vec![1.0; 5]),
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
            variances_m2: Some(vec![1.0; 4]),
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
            variances_m2: Some(vec![1.0; 5]),
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
            variances_m2: Some(vec![1.0; 6]),
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
            variances_m2: None,
        };
        assert_eq!(
            raim(&input, &RaimOptions::default()),
            Err(QualityError::InvalidResiduals)
        );

        let input = RaimInput {
            used_sats: ["G01", "G02"].into_iter().map(str::to_string).collect(),
            residuals_m: vec![1.0, f64::NAN],
            variances_m2: None,
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
            variances_m2: Some(vec![1.0; 5]),
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
    fn raim_solution_weights_divide_each_residual_by_its_own_sigma() {
        let variances_m2 = vec![4.0, 0.25, 9.0, 1.0, 16.0];
        let residuals_m = vec![2.0, -1.0, 6.0, 0.5, -8.0];
        let input = RaimInput {
            used_sats: ["G01", "G02", "G03", "G04", "G05"]
                .into_iter()
                .map(str::to_string)
                .collect(),
            residuals_m: residuals_m.clone(),
            variances_m2: Some(variances_m2.clone()),
        };
        let result = raim(&input, &RaimOptions::default()).unwrap();

        // RTKLIB `estpos` divides by `sqrt(var)`, `valsol` sums the squares.
        let mut expected = 0.0;
        for (r, v) in residuals_m.iter().zip(&variances_m2) {
            let n = r / v.sqrt();
            expected += n * n;
        }
        assert_eq!(result.test_statistic.to_bits(), expected.to_bits());
        assert_eq!(result.test_statistic, 1.0 + 4.0 + 4.0 + 0.25 + 4.0);
        assert_eq!(result.normalized_residuals["G02"], -2.0);
        assert_eq!(result.dof, 1);
        assert!(result.fault_detected == (expected > result.threshold.unwrap()));

        // The unit mode reads the raw residuals.
        let unit = raim(
            &input,
            &RaimOptions {
                weights: RaimWeights::Unit,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(unit.test_statistic, 4.0 + 1.0 + 36.0 + 0.25 + 64.0);
    }

    #[test]
    fn raim_solution_weights_refuse_missing_or_invalid_variances() {
        let used_sats: Vec<String> = ["G01", "G02", "G03", "G04", "G05"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let input = |variances_m2| RaimInput {
            used_sats: used_sats.clone(),
            residuals_m: vec![0.0; 5],
            variances_m2,
        };
        assert_eq!(
            raim(&input(None), &RaimOptions::default()),
            Err(QualityError::MissingVariances)
        );
        for variances in [
            vec![1.0; 4],
            vec![1.0, 1.0, 0.0, 1.0, 1.0],
            vec![1.0, -1.0, 1.0, 1.0, 1.0],
            vec![1.0, 1.0, 1.0, f64::NAN, 1.0],
            vec![1.0, 1.0, 1.0, 1.0, f64::INFINITY],
        ] {
            assert_eq!(
                raim(&input(Some(variances)), &RaimOptions::default()),
                Err(QualityError::InvalidVariance)
            );
        }
        // Explicit weights need no variances.
        let unit = RaimOptions {
            weights: RaimWeights::Unit,
            ..Default::default()
        };
        assert!(raim(&input(None), &unit).is_ok());
        // ... and are not validated when they are not read.
        assert!(raim(&input(Some(vec![f64::NAN; 2])), &unit).is_ok());
    }

    #[test]
    fn raim_for_solution_counts_the_clocks_the_solve_estimated() {
        // A QZSS range on the GPS clock adds no clock parameter: six
        // satellites and one clock leave two degrees of freedom, where the two
        // system letters would leave one.
        let mut solution = valid_receiver_solution();
        solution.used_sats = (1..=5)
            .map(gps)
            .chain([GnssSatelliteId::new(GnssSystem::Qzss, 1).unwrap()])
            .collect();
        solution.residuals_m = vec![0.0; 6];
        solution.pseudorange_variances_m2 = vec![1.0; 6];
        solution.weights = vec![1.0; 6];
        solution.metadata.systems = vec![GnssSystem::Gps];
        let result = raim_for_solution(&solution, &RaimOptions::default()).unwrap();
        assert_eq!(result.dof, 2);

        let overridden = raim_for_solution(
            &solution,
            &RaimOptions {
                n_systems: Some(2),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(overridden.dof, 1);
    }

    fn gps_observations(prns: std::ops::RangeInclusive<u8>) -> Vec<Observation> {
        prns.map(|prn| Observation {
            satellite_id: gps(prn),
            pseudorange_m: f64::from(prn),
        })
        .collect()
    }

    /// The leave-one-out rule, not the largest residual, picks the exclusion.
    /// The fault on G04 shows up in the full solve as the largest residual on
    /// G01 (a satellite the geometry leans on hides its own fault); leaving
    /// G04 out removes it, so the re-solve's RMS is smallest there.
    #[test]
    fn fde_excludes_the_leave_one_out_minimum_not_the_largest_residual() {
        let observations = gps_observations(1..=7);
        let result = fde(&observations, &FdeOptions::default(), |remaining| {
            let faulted = remaining.iter().any(|ob| ob.satellite_id == gps(4));
            Ok::<_, TestSolveError>(TestSolution::over(remaining, |satellite| {
                match (faulted, satellite.prn) {
                    (true, 1) => 9.0,
                    (true, 4) => 1.0,
                    (true, _) => -2.0,
                    (false, _) => 0.0,
                }
            }))
        })
        .unwrap();

        assert_eq!(result.excluded, vec!["G04".to_string()]);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.solution.used_sats.len(), 6);
        assert!(!result.raim.fault_detected);
    }

    /// Equal RMS values go to the later satellite in RTKLIB's order, candidates
    /// using fewer than five satellites are skipped, and the 100 m cap bounds
    /// the kept RMS.
    #[test]
    fn fde_leave_one_out_ties_floor_and_cap_follow_rtklib() {
        // Leaving out G02 or G05 gives the same residuals (RMS 1 m); every other
        // exclusion leaves 3 m residuals.
        let observations = gps_observations(1..=6);
        let tied = |remaining: &[Observation]| {
            let without = |prn| !remaining.iter().any(|ob| ob.satellite_id == gps(prn));
            let level = if without(2) || without(5) { 1.0 } else { 3.0 };
            Ok::<_, TestSolveError>(TestSolution::over(remaining, |_| {
                if remaining.len() == 6 {
                    50.0
                } else {
                    level
                }
            }))
        };
        let options = FdeOptions {
            raim: RaimOptions {
                weights: RaimWeights::Unit,
                p_fa: 0.5,
                ..Default::default()
            },
            ..FdeOptions::default()
        };
        let err = fde(&observations, &options, tied).unwrap_err();
        let FdeError::FaultUnresolved(unresolved) = err else {
            panic!("the 1 m set still fails a p_fa = 0.5 test");
        };
        assert_eq!(
            unresolved.reason,
            FdeUnresolvedReason::ExclusionBudgetExhausted
        );
        assert_eq!(unresolved.excluded, vec!["G05".to_string()]);
        assert_eq!(unresolved.solution.used_sats.len(), 5);
        assert!(unresolved.raim.fault_detected);

        // Five observations: every candidate re-solve uses four satellites.
        let five = gps_observations(1..=5);
        let err = fde(&five, &FdeOptions::default(), |remaining| {
            Ok::<_, TestSolveError>(TestSolution::over(remaining, |_| {
                if remaining.len() == 5 {
                    50.0
                } else {
                    0.0
                }
            }))
        })
        .unwrap_err();
        let FdeError::FaultUnresolved(unresolved) = err else {
            panic!("no candidate keeps five satellites");
        };
        assert_eq!(
            unresolved.reason,
            FdeUnresolvedReason::NoAdmissibleExclusion
        );
        assert!(unresolved.excluded.is_empty());

        // Every exclusion leaves more than 100 m RMS.
        let err = fde(&observations, &FdeOptions::default(), |remaining| {
            Ok::<_, TestSolveError>(TestSolution::over(remaining, |_| {
                if remaining.len() == 6 {
                    500.0
                } else {
                    100.5
                }
            }))
        })
        .unwrap_err();
        let FdeError::FaultUnresolved(unresolved) = err else {
            panic!("the cap refuses every candidate");
        };
        assert_eq!(
            unresolved.reason,
            FdeUnresolvedReason::NoAdmissibleExclusion
        );

        // Exactly 100 m is kept, as RTKLIB's `rms_e > rms` test keeps it.
        let err = fde(&observations, &FdeOptions::default(), |remaining| {
            Ok::<_, TestSolveError>(TestSolution::over(remaining, |_| {
                if remaining.len() == 6 {
                    500.0
                } else {
                    100.0
                }
            }))
        })
        .unwrap_err();
        let FdeError::FaultUnresolved(unresolved) = err else {
            panic!("a 100 m set still fails detection");
        };
        assert_eq!(
            unresolved.reason,
            FdeUnresolvedReason::ExclusionBudgetExhausted
        );
        assert_eq!(unresolved.excluded, vec!["G06".to_string()]);
    }

    /// A candidate whose re-solve fails is skipped, as RTKLIB skips a
    /// candidate whose `estpos` fails, and the search goes on.
    #[test]
    fn fde_skips_candidates_whose_resolve_fails() {
        let observations = gps_observations(1..=7);
        let result = fde(&observations, &FdeOptions::default(), |remaining| {
            let without = |prn| !remaining.iter().any(|ob| ob.satellite_id == gps(prn));
            if remaining.len() == 6 && without(3) {
                return Err(TestSolveError::Unsettled);
            }
            Ok(TestSolution::over(remaining, |_| {
                if remaining.len() == 7 {
                    40.0
                } else if without(6) {
                    0.0
                } else {
                    5.0
                }
            }))
        })
        .unwrap();
        assert_eq!(result.excluded, vec!["G06".to_string()]);
    }

    /// A full-set solve that fails the way demo5's `estpos` does is searched
    /// over every observed satellite, as `pntpos` calls `raim_fde`; an input
    /// error, fewer than six observations, a zero budget, or no admissible
    /// candidate returns the failure.
    #[test]
    fn fde_searches_a_failed_full_set_solve_as_rtklib_pntpos_does() {
        let observations = gps_observations(1..=7);
        let unsettled_unless_g03_is_out = |remaining: &[Observation]| {
            let with_g03 = remaining.iter().any(|ob| ob.satellite_id == gps(3));
            if remaining.len() == 7 {
                return Err(TestSolveError::Unsettled);
            }
            Ok(TestSolution::over(remaining, |_| {
                if with_g03 {
                    5.0
                } else {
                    0.0
                }
            }))
        };
        let result = fde(
            &observations,
            &FdeOptions::default(),
            unsettled_unless_g03_is_out,
        )
        .unwrap();
        assert_eq!(result.excluded, vec!["G03".to_string()]);
        assert_eq!(result.iterations, 1);
        assert!(!result.raim.fault_detected);

        let input_error = fde(&observations, &FdeOptions::default(), |_| {
            Err::<TestSolution, _>(TestSolveError::Input)
        })
        .unwrap_err();
        assert_eq!(input_error, FdeError::Solve(TestSolveError::Input));

        let five = gps_observations(1..=5);
        let too_few = fde(&five, &FdeOptions::default(), |remaining| {
            if remaining.len() == 5 {
                return Err(TestSolveError::Unsettled);
            }
            Ok(TestSolution::over(remaining, |_| 0.0))
        })
        .unwrap_err();
        assert_eq!(too_few, FdeError::Solve(TestSolveError::Unsettled));

        let no_budget = fde(
            &observations,
            &FdeOptions::new(RaimOptions::default(), 0),
            unsettled_unless_g03_is_out,
        )
        .unwrap_err();
        assert_eq!(no_budget, FdeError::Solve(TestSolveError::Unsettled));

        let nothing_cures = fde(&observations, &FdeOptions::default(), |_| {
            Err::<TestSolution, _>(TestSolveError::Unsettled)
        })
        .unwrap_err();
        assert_eq!(nothing_cures, FdeError::Solve(TestSolveError::Unsettled));
    }

    #[test]
    fn fde_spp_errors_admit_the_search_only_for_estpos_failures() {
        use crate::astro::math::least_squares::SolveError;
        for (error, admits) in [
            (
                FdeSppError::Spp(SppError::Singular(SolveError::SingularJacobian)),
                true,
            ),
            (
                FdeSppError::Spp(SppError::SelectionUnsettled { passes: 10 }),
                true,
            ),
            (
                FdeSppError::Spp(SppError::TooFewSatellites {
                    used: 3,
                    required: 4,
                }),
                false,
            ),
            (
                FdeSppError::Spp(SppError::DuplicateObservation { satellite: gps(1) }),
                false,
            ),
            (
                FdeSppError::Spp(SppError::EphemerisLost { satellite: gps(1) }),
                false,
            ),
            (
                FdeSppError::Validation(SolutionValidationError::DegenerateGeometryRankDeficient),
                true,
            ),
            (
                FdeSppError::Validation(SolutionValidationError::DegenerateGeometryPdop(30.0)),
                true,
            ),
            (
                FdeSppError::Validation(SolutionValidationError::ImplausiblePosition(1.0)),
                true,
            ),
            (
                FdeSppError::Validation(SolutionValidationError::NoConvergence(2.0e4)),
                true,
            ),
            (
                FdeSppError::Validation(SolutionValidationError::InvalidResiduals),
                false,
            ),
            (
                FdeSppError::Validation(SolutionValidationError::InvalidOptions {
                    field: "max_pdop",
                    reason: "not finite",
                }),
                false,
            ),
        ] {
            assert_eq!(error.admits_exclusion_search(), admits, "{error}");
        }
    }

    /// The candidate RMS sums the squares in RTKLIB's satellite-number order
    /// (QZSS before BeiDou), whatever order the solution lists them in.
    #[test]
    fn leave_one_out_rms_sums_in_rtklib_satellite_order() {
        let used: Vec<String> = ["C01", "J01", "G01"]
            .into_iter()
            .map(str::to_string)
            .collect();
        let residuals = [1.0, 1.0, 1.0e8];
        // G01 first: 1e16 + 1 rounds to 1e16, and so does the second + 1.
        let expected = (((1.0e16_f64 + 1.0) + 1.0) / 3.0).sqrt();
        let listed_order = (((1.0_f64 + 1.0) + 1.0e16) / 3.0).sqrt();
        assert_ne!(expected.to_bits(), listed_order.to_bits());
        assert_eq!(
            rtklib_order_rms_m(&used, &residuals).to_bits(),
            expected.to_bits()
        );
        // Tokens that name no satellite keep the listed order.
        let opaque: Vec<String> = ["a", "b", "c"].into_iter().map(str::to_string).collect();
        assert_eq!(
            rtklib_order_rms_m(&opaque, &residuals).to_bits(),
            listed_order.to_bits()
        );
    }

    /// The iterative mode repeats detection and the leave-one-out search.
    #[test]
    fn fde_multiple_exclusions_repeat_the_search_on_the_remaining_set() {
        let observations = gps_observations(1..=8);
        let solve = |remaining: &[Observation]| {
            let with = |prn| remaining.iter().any(|ob| ob.satellite_id == gps(prn));
            Ok::<_, TestSolveError>(TestSolution::over(remaining, |satellite| {
                let mut r = 0.0;
                if with(2) {
                    r += if satellite.prn == 2 { 30.0 } else { -3.0 };
                }
                if with(7) {
                    r += if satellite.prn == 7 { -20.0 } else { 2.0 };
                }
                r
            }))
        };

        let single = fde(&observations, &FdeOptions::default(), solve).unwrap_err();
        let FdeError::FaultUnresolved(single) = single else {
            panic!("one exclusion leaves the second fault");
        };
        assert_eq!(single.excluded, vec!["G02".to_string()]);

        let options = FdeOptions {
            max_exclusions: 2,
            ..FdeOptions::default()
        };
        let result = fde(&observations, &options, solve).unwrap();
        assert_eq!(result.excluded, vec!["G02".to_string(), "G07".to_string()]);
        assert_eq!(result.iterations, 2);
        assert!(!result.raim.fault_detected);
    }

    #[test]
    fn fde_refuses_fault_when_budget_is_exhausted() {
        let observations = gps_observations(1..=5);
        let options = FdeOptions::new(RaimOptions::default(), 0);
        let err = fde(&observations, &options, |remaining| {
            Ok::<_, TestSolveError>(TestSolution::over(remaining, |satellite| {
                if satellite.prn == 5 {
                    5.0
                } else {
                    0.0
                }
            }))
        })
        .unwrap_err();

        let FdeError::FaultUnresolved(unresolved) = err else {
            panic!("expected an unresolved fault, got {err:?}");
        };
        assert_eq!(
            unresolved.reason,
            FdeUnresolvedReason::ExclusionBudgetExhausted
        );
        assert_eq!(unresolved.raim.test_statistic, 25.0);
        assert!(unresolved.raim.fault_detected);
        assert!(unresolved.excluded.is_empty());
        assert_eq!(unresolved.solution.used_sats.len(), 5);
    }

    #[test]
    fn fde_rejects_an_invalid_exclusion_rms_cap() {
        let observations = gps_observations(1..=6);
        for cap in [0.0, -1.0, f64::NAN] {
            let options = FdeOptions {
                max_exclusion_rms_m: cap,
                ..FdeOptions::default()
            };
            let err = fde(&observations, &options, |remaining| {
                Ok::<_, TestSolveError>(TestSolution::over(remaining, |_| 0.0))
            })
            .unwrap_err();
            assert_eq!(err, FdeError::Raim(QualityError::InvalidParameter));
        }
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

        let options = RangeFdeOptions {
            max_exclusions: usize::MAX,
            ..Default::default()
        };
        let result = raim_fde_design(&rows, &options).expect("fde");

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
    fn range_fde_default_makes_one_exclusion_as_rtklib() {
        let dx_true = [0.5, 1.5, -1.0, 2.0];
        let mut rows = range_rows(dx_true);
        rows[2].residual_m += 50.0;
        rows[5].residual_m -= 40.0;

        let default = raim_fde_design(&rows, &RangeFdeOptions::default()).expect("fde");
        let single = raim_fde_design(
            &rows,
            &RangeFdeOptions {
                max_exclusions: 1,
                ..Default::default()
            },
        )
        .expect("fde");
        assert_eq!(default, single);
        assert_eq!(default.iterations, 1);
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
