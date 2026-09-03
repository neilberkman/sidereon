//! Classical Baarda/Teunissen reliability design.
//!
//! Clean-room validation provenance for the tests in this module:
//! Baarda, W. (1968), "A testing procedure for use in geodetic networks",
//! Netherlands Geodetic Commission, Publications on Geodesy 2(5).
//! Teunissen, P. J. G. (2024), "Network Quality Control", TU Delft OPEN.

use crate::araim::ism::Ism;
use crate::araim::protection::{design_row, gain_matrix_enu, ProtectionModel};
use crate::astro::math::linear::{invert_symmetric_pd, normal_equations_weighted};
use crate::astro::math::special::normal_q_inv;
use crate::quality::{QualityError, DEFAULT_P_FA};
use crate::validate;

use super::{AraimError, AraimGeometry};

const DEFAULT_BETA: f64 = 0.20;
const DEFAULT_MIN_REDUNDANCY: f64 = 1.0e-9;
const REDUNDANCY_TOL: f64 = 1.0e-10;

/// One geometry row for pre-data reliability design.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RangeReliabilityRow {
    /// Observation identifier echoed into the report.
    pub id: String,
    /// Linearized design row for this range observation.
    pub design_row: Vec<f64>,
    /// Externally supplied one-sigma range model, meters.
    pub sigma_m: f64,
}

/// Options for Baarda/Teunissen reliability design.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[non_exhaustive]
pub struct ReliabilityOptions {
    /// Two-sided false-alarm probability for the one-dimensional data-snooping w-test.
    pub alpha: f64,
    /// Missed-detection probability for the target bias.
    pub beta: f64,
    /// Optional precomputed noncentrality parameter.
    pub lambda0_override: Option<f64>,
    /// Redundancy below which an observation is reported as uncheckable.
    pub min_redundancy: f64,
}

impl Default for ReliabilityOptions {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_P_FA,
            beta: DEFAULT_BETA,
            lambda0_override: None,
            min_redundancy: DEFAULT_MIN_REDUNDANCY,
        }
    }
}

/// Reliability diagnostics for one observation.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ObservationReliability {
    /// Observation identifier echoed from the input row.
    pub id: String,
    /// Redundancy number `r_i`, the geometry-checked fraction of this observation.
    pub redundancy: f64,
    /// Minimal detectable bias in this observation, meters.
    pub mdb_m: Option<f64>,
    /// External effect for the just-detectable bias, meters.
    ///
    /// [`reliability_araim`] reports local `[east, north, up]`. [`reliability_design`]
    /// reports the first three state-coordinate entries of the same gain-vector
    /// imprint, or `None` when the state has fewer than three coordinates.
    pub external_enu_m: Option<[f64; 3]>,
    /// Bias-to-noise ratio in state space.
    pub bias_to_noise: Option<f64>,
    /// True when redundancy is below the configured reporting floor.
    pub uncheckable: bool,
}

/// Aggregate reliability diagnostics for a design.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReliabilitySummary {
    /// Number of observations in the design.
    pub n_obs: usize,
    /// Number of estimated parameters in the design.
    pub n_params: usize,
    /// Algebraic degrees of freedom, `n_obs - n_params`.
    pub dof: usize,
    /// Sum of per-observation redundancy numbers.
    pub sum_redundancy: f64,
    /// Noncentrality parameter used for MDB calculations.
    pub lambda0: f64,
    /// Largest finite MDB in the design, if any observation is checkable.
    pub max_mdb_m: Option<(String, f64)>,
    /// Smallest redundancy number and its observation identifier.
    pub min_redundancy: (String, f64),
    /// Count of observations reported as uncheckable.
    pub n_uncheckable: usize,
}

/// Full reliability design report.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ReliabilityReport {
    /// Per-observation reliability diagnostics, in input order.
    pub per_observation: Vec<ObservationReliability>,
    /// Aggregate design diagnostics.
    pub summary: ReliabilitySummary,
}

/// Compute Baarda's one-dimensional data-snooping noncentrality parameter.
///
/// The returned value is
/// `(normal_q_inv(alpha / 2) + normal_q_inv(beta))^2`, with `alpha` treated as a
/// two-sided false-alarm probability and `beta` as missed-detection probability.
pub fn wtest_noncentrality(alpha: f64, beta: f64) -> Result<f64, QualityError> {
    Ok(wtest_noncentrality_components(alpha, beta)?.lambda0)
}

/// Both Baarda w-test noncentrality forms from a single derivation: the
/// noncentrality parameter `delta0 = normal_q_inv(alpha / 2) + normal_q_inv(beta)`
/// and its square `lambda0`. Consumers that report both must take them from
/// here rather than re-deriving one from the other.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WtestNoncentralityComponents {
    /// The noncentrality parameter `delta0`.
    pub delta0: f64,
    /// The squared noncentrality `lambda0 = delta0 * delta0`.
    pub lambda0: f64,
}

/// Compute [`WtestNoncentralityComponents`] for a two-sided false-alarm
/// probability `alpha` and a missed-detection probability `beta`.
pub fn wtest_noncentrality_components(
    alpha: f64,
    beta: f64,
) -> Result<WtestNoncentralityComponents, QualityError> {
    validate_probability(alpha)?;
    if !(0.0..1.0).contains(&beta) || !beta.is_finite() {
        return Err(QualityError::InvalidReliabilityParameter);
    }
    let k_alpha = normal_q_inv(0.5 * alpha).ok_or(QualityError::InvalidProbability)?;
    let k_beta = normal_q_inv(beta).ok_or(QualityError::InvalidReliabilityParameter)?;
    let delta0 = k_alpha + k_beta;
    let lambda0 = delta0 * delta0;
    if lambda0.is_finite() && lambda0 > 0.0 {
        Ok(WtestNoncentralityComponents { delta0, lambda0 })
    } else {
        Err(QualityError::InvalidReliabilityParameter)
    }
}

/// Compute internal and external reliability from supplied range geometry.
///
/// This is a pre-data design calculation. It consumes only the design matrix
/// and externally supplied range sigmas, never residuals or measured ranges.
pub fn reliability_design(
    rows: &[RangeReliabilityRow],
    options: &ReliabilityOptions,
) -> Result<ReliabilityReport, QualityError> {
    let n_params = validate_reliability_rows(rows)?;
    let lambda0 = reliability_lambda0(options)?;
    let q_xx = design_covariance(rows, n_params)?;

    finish_reliability_report(
        rows,
        n_params,
        lambda0,
        options.min_redundancy,
        |idx, mdb_m| state_external(rows, &q_xx, idx, mdb_m),
    )
}

/// Compute reliability for ARAIM geometry using the same ENU gain matrix as MHSS.
///
/// The protection model supplies the range sigmas. The calculation is pre-data
/// and does not consume residuals or measured pseudoranges.
pub fn reliability_araim(
    geometry: &AraimGeometry,
    model: &dyn ProtectionModel,
    options: &ReliabilityOptions,
) -> Result<ReliabilityReport, AraimError> {
    let rows = araim_rows(geometry, model)?;
    let lambda0 = reliability_lambda0(options).map_err(map_quality_error)?;
    let weights = rows
        .iter()
        .map(|row| 1.0 / (row.sigma_m * row.sigma_m))
        .collect::<Vec<_>>();
    let gain = gain_matrix_enu(geometry, &weights)?;
    let n_params = geometry_state_len(geometry)?;

    finish_reliability_report(
        &rows,
        n_params,
        lambda0,
        options.min_redundancy,
        |idx, mdb_m| {
            Some([
                gain.enu_rows[0][idx] * mdb_m,
                gain.enu_rows[1][idx] * mdb_m,
                gain.enu_rows[2][idx] * mdb_m,
            ])
        },
    )
    .map_err(map_quality_error)
}

impl ProtectionModel for Ism {
    fn sigma_int_m(&self, row: &super::AraimRow) -> Result<f64, AraimError> {
        Ok(self.effective_for(row)?.sigma_int_m)
    }

    fn sigma_acc_m(&self, row: &super::AraimRow) -> Result<f64, AraimError> {
        Ok(self.effective_for(row)?.sigma_acc_m)
    }
}

fn araim_rows(
    geometry: &AraimGeometry,
    model: &dyn ProtectionModel,
) -> Result<Vec<RangeReliabilityRow>, AraimError> {
    let n_params = geometry_state_len(geometry)?;
    if geometry.rows.len() < n_params {
        return Err(AraimError::InsufficientGeometry);
    }
    let mut rows = Vec::with_capacity(geometry.rows.len());
    for row in &geometry.rows {
        let sigma_int_m = model.sigma_int_m(row)?;
        let sigma_acc_m = model.sigma_acc_m(row)?;
        if !valid_positive_finite(sigma_int_m) || !valid_positive_finite(sigma_acc_m) {
            return Err(AraimError::InvalidIsm);
        }
        rows.push(RangeReliabilityRow {
            id: row.id.to_string(),
            design_row: design_row(&geometry.clock_systems, row)?,
            sigma_m: sigma_int_m,
        });
    }
    Ok(rows)
}

fn reliability_lambda0(options: &ReliabilityOptions) -> Result<f64, QualityError> {
    validate_reliability_options(options)?;
    match options.lambda0_override {
        Some(lambda0) => Ok(lambda0),
        None => wtest_noncentrality(options.alpha, options.beta),
    }
}

fn validate_reliability_options(options: &ReliabilityOptions) -> Result<(), QualityError> {
    validate_probability(options.alpha)?;
    if !(0.0..1.0).contains(&options.beta) || !options.beta.is_finite() {
        return Err(QualityError::InvalidReliabilityParameter);
    }
    if let Some(lambda0) = options.lambda0_override {
        if !lambda0.is_finite() || lambda0 <= 0.0 {
            return Err(QualityError::InvalidReliabilityParameter);
        }
    }
    if !options.min_redundancy.is_finite() || options.min_redundancy <= 0.0 {
        return Err(QualityError::InvalidReliabilityParameter);
    }
    Ok(())
}

fn validate_reliability_rows(rows: &[RangeReliabilityRow]) -> Result<usize, QualityError> {
    let first = rows.first().ok_or(QualityError::InvalidDesign)?;
    let n_params = first.design_row.len();
    if n_params == 0 || rows.len() < n_params {
        return Err(QualityError::InvalidDesign);
    }
    for row in rows {
        if row.design_row.len() != n_params {
            return Err(QualityError::InvalidDesign);
        }
        validate::finite_slice(&row.design_row, "reliability design row")
            .map_err(|_| QualityError::InvalidDesign)?;
        validate::finite_positive(row.sigma_m, "reliability sigma")
            .map_err(|_| QualityError::InvalidWeight)?;
        let weight = 1.0 / (row.sigma_m * row.sigma_m);
        if !weight.is_finite() || weight <= 0.0 {
            return Err(QualityError::InvalidWeight);
        }
    }
    Ok(n_params)
}

fn design_covariance(
    rows: &[RangeReliabilityRow],
    n_params: usize,
) -> Result<Vec<Vec<f64>>, QualityError> {
    let (normal, _) = normal_equations_weighted(
        rows.iter()
            .map(|row| (row.design_row.as_slice(), 0.0, 1.0 / row.sigma_m)),
        n_params,
    )
    .ok_or(QualityError::InvalidDesign)?;
    invert_symmetric_pd(&normal).ok_or(QualityError::SingularGeometry)
}

fn finish_reliability_report<F>(
    rows: &[RangeReliabilityRow],
    n_params: usize,
    lambda0: f64,
    min_redundancy: f64,
    mut external: F,
) -> Result<ReliabilityReport, QualityError>
where
    F: FnMut(usize, f64) -> Option<[f64; 3]>,
{
    let q_xx = design_covariance(rows, n_params)?;
    let mut per_observation = Vec::with_capacity(rows.len());
    let mut sum_redundancy = 0.0;
    let mut max_mdb_m: Option<(String, f64)> = None;
    let mut min_pair: Option<(String, f64)> = None;
    let mut n_uncheckable = 0usize;

    for (idx, row) in rows.iter().enumerate() {
        let redundancy = redundancy_number(row, &q_xx)?;
        sum_redundancy += redundancy;
        update_min_redundancy(&mut min_pair, &row.id, redundancy);

        let uncheckable = redundancy < min_redundancy;
        let (mdb_m, external_enu_m, bias_to_noise) = if uncheckable {
            n_uncheckable += 1;
            (None, None, None)
        } else {
            let mdb = row.sigma_m * (lambda0 / redundancy).sqrt();
            let bnr = (lambda0 * (1.0 - redundancy) / redundancy).sqrt();
            update_max_mdb(&mut max_mdb_m, &row.id, mdb);
            (Some(mdb), external(idx, mdb), Some(bnr))
        };

        per_observation.push(ObservationReliability {
            id: row.id.clone(),
            redundancy,
            mdb_m,
            external_enu_m,
            bias_to_noise,
            uncheckable,
        });
    }

    Ok(ReliabilityReport {
        per_observation,
        summary: ReliabilitySummary {
            n_obs: rows.len(),
            n_params,
            dof: rows.len().saturating_sub(n_params),
            sum_redundancy,
            lambda0,
            max_mdb_m,
            min_redundancy: min_pair.unwrap_or_else(|| (String::new(), 0.0)),
            n_uncheckable,
        },
    })
}

fn redundancy_number(row: &RangeReliabilityRow, q_xx: &[Vec<f64>]) -> Result<f64, QualityError> {
    let qh = mat_vec(q_xx, &row.design_row);
    let leverage = dot(&row.design_row, &qh) / (row.sigma_m * row.sigma_m);
    if !leverage.is_finite() {
        return Err(QualityError::SingularGeometry);
    }
    let raw = 1.0 - leverage;
    if raw >= 0.0 {
        Ok(raw.min(1.0))
    } else if raw.abs() <= REDUNDANCY_TOL {
        Ok(0.0)
    } else {
        Err(QualityError::SingularGeometry)
    }
}

fn state_external(
    rows: &[RangeReliabilityRow],
    q_xx: &[Vec<f64>],
    idx: usize,
    mdb_m: f64,
) -> Option<[f64; 3]> {
    if q_xx.len() < 3 {
        return None;
    }
    let row = &rows[idx];
    let scale = mdb_m / (row.sigma_m * row.sigma_m);
    let qh = mat_vec(q_xx, &row.design_row);
    Some([qh[0] * scale, qh[1] * scale, qh[2] * scale])
}

fn update_min_redundancy(pair: &mut Option<(String, f64)>, id: &str, value: f64) {
    if match pair.as_ref() {
        Some((_, current)) => value < *current,
        None => true,
    } {
        *pair = Some((id.to_owned(), value));
    }
}

fn update_max_mdb(pair: &mut Option<(String, f64)>, id: &str, value: f64) {
    if match pair.as_ref() {
        Some((_, current)) => value > *current,
        None => true,
    } {
        *pair = Some((id.to_owned(), value));
    }
}

fn geometry_state_len(geometry: &AraimGeometry) -> Result<usize, AraimError> {
    if geometry.clock_systems.is_empty() {
        return Err(AraimError::InsufficientGeometry);
    }
    Ok(3 + geometry.clock_systems.len())
}

fn map_quality_error(err: QualityError) -> AraimError {
    match err {
        QualityError::InvalidProbability | QualityError::InvalidReliabilityParameter => {
            AraimError::InvalidAllocation
        }
        QualityError::InvalidWeight => AraimError::InvalidIsm,
        QualityError::InvalidDesign | QualityError::SingularGeometry => {
            AraimError::InsufficientGeometry
        }
        _ => AraimError::NumericalFailure,
    }
}

fn validate_probability(value: f64) -> Result<(), QualityError> {
    if (0.0..1.0).contains(&value) && value.is_finite() {
        Ok(())
    } else {
        Err(QualityError::InvalidProbability)
    }
}

fn valid_positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter().map(|row| dot(row, x)).collect()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::araim::protection::gain_matrix_enu;
    use crate::araim::{AraimGeometry, AraimRow};
    use crate::dop::{ecef_to_enu_rotation, LineOfSight};
    use crate::frame::Wgs84Geodetic;
    use crate::id::{GnssSatelliteId, GnssSystem};

    const LAMBDA_BAARDA_4DP: f64 = 17.075;

    #[test]
    fn algebraic_identities_hold_for_weighted_design() {
        let rows = algebraic_rows();
        let options = options_with_lambda(LAMBDA_BAARDA_4DP);
        let report = reliability_design(&rows, &options).expect("reliability report");
        let n = rows.len();
        let u = rows[0].design_row.len();
        let q_xx = design_covariance(&rows, u).expect("covariance");
        let r = residual_projector(&rows, &q_xx);
        let h = design_matrix(&rows);
        let w = weight_matrix(&rows);

        assert_close(report.summary.sum_redundancy, (n - u) as f64, 2.0e-14);
        assert_close(trace(&r), (n - u) as f64, 2.0e-14);
        assert!(inf_norm(&mat_sub(&mat_mul(&r, &r), &r)) < 1.0e-9);
        assert!(inf_norm(&mat_mul(&r, &h)) < 1.0e-9);
        assert!(inf_norm(&mat_mul(&mat_mul(&transpose(&r), &w), &h)) < 1.0e-9);

        let q_v = q_v_matrix(&rows, &q_xx);
        let q_v_w = mat_mul(&q_v, &w);
        let normal = normal_matrix(&rows);
        for (idx, obs) in report.per_observation.iter().enumerate() {
            assert_close(obs.redundancy, q_v_w[idx][idx], 2.0e-14);
            let mdb = obs.mdb_m.expect("checkable MDB");
            let dx = full_state_external(&rows[idx], &q_xx, mdb);
            let bnr_full = quadratic_form(&normal, &dx).sqrt();
            assert_close(bnr_full, obs.bias_to_noise.expect("checkable BNR"), 2.0e-12);
        }
    }

    #[test]
    fn baarda_wtest_constant_and_monotonicity_are_pinned() {
        let lambda0 = wtest_noncentrality(1.0e-3, 0.20).expect("canonical lambda");
        let delta0 = lambda0.sqrt();

        assert_close(delta0, 4.132147965064809, 2.0e-15);
        assert_close(lambda0, 17.074646805189243, 4.0e-15);
        assert_eq!((delta0 * 100.0).round() / 100.0, 4.13);
        assert_eq!((lambda0 * 100.0).round() / 100.0, 17.07);

        let smaller_alpha = wtest_noncentrality(1.0e-4, 0.20).expect("smaller alpha");
        let smaller_beta = wtest_noncentrality(1.0e-3, 0.10).expect("smaller beta");
        assert!(smaller_alpha > lambda0);
        assert!(smaller_beta > lambda0);
    }

    #[test]
    fn equal_precision_single_unknown_closed_form_is_exact_to_roundoff() {
        let lambda0 = wtest_noncentrality(1.0e-3, 0.20).expect("lambda");
        let sigma_m = 1.0;
        let rows = equal_precision_rows(2, sigma_m);
        let report = reliability_design(&rows, &ReliabilityOptions::default()).expect("report");

        for obs in &report.per_observation {
            assert_close(obs.redundancy, 0.5, 1.0e-15);
            assert_close(
                obs.mdb_m.expect("MDB"),
                sigma_m * (2.0 * lambda0).sqrt(),
                1.0e-14,
            );
            assert_close(obs.bias_to_noise.expect("BNR"), lambda0.sqrt(), 1.0e-14);
        }
        assert_eq!(
            (report.per_observation[0].mdb_m.expect("MDB") * 100.0).round() / 100.0,
            5.84
        );
        assert_eq!(
            (report.per_observation[0].bias_to_noise.expect("BNR") * 100.0).round() / 100.0,
            4.13
        );
        assert_close(report.summary.sum_redundancy, 1.0, 1.0e-15);

        let report = reliability_design(
            &equal_precision_rows(1, sigma_m),
            &ReliabilityOptions::default(),
        )
        .expect("one-row report");
        assert_eq!(report.summary.n_uncheckable, 1);
        assert!(report.per_observation[0].uncheckable);
        assert_eq!(report.per_observation[0].mdb_m, None);
        assert_eq!(report.per_observation[0].bias_to_noise, None);
        assert_eq!(report.per_observation[0].external_enu_m, None);
    }

    #[test]
    fn teunissen_leveling_connection_matches_published_formula() {
        let n = 4usize;
        let sigma_h = 2.0;
        let sigma_h_cap = 3.0;
        let rows = leveling_connection_rows(n, sigma_h, sigma_h_cap);
        let report = reliability_design(&rows, &options_with_lambda(LAMBDA_BAARDA_4DP))
            .expect("leveling report");

        let variance_sum = sigma_h * sigma_h + sigma_h_cap * sigma_h_cap;
        let common = 1.0 - 1.0 / n as f64;
        let expected_r_h = sigma_h * sigma_h * common / variance_sum;
        let expected_r_h_cap = sigma_h_cap * sigma_h_cap * common / variance_sum;
        let expected_mdb = (LAMBDA_BAARDA_4DP * variance_sum / common).sqrt();

        for obs in &report.per_observation[..n] {
            assert_close(obs.redundancy, expected_r_h, 2.0e-14);
            assert_close(obs.mdb_m.expect("MDB"), expected_mdb, 2.0e-13);
        }
        for obs in &report.per_observation[n..] {
            assert_close(obs.redundancy, expected_r_h_cap, 2.0e-14);
            assert_close(obs.mdb_m.expect("MDB"), expected_mdb, 2.0e-13);
        }
        assert_close(report.summary.sum_redundancy, n as f64 - 1.0, 2.0e-13);

        let one = reliability_design(
            &leveling_connection_rows(1, sigma_h, sigma_h_cap),
            &options_with_lambda(LAMBDA_BAARDA_4DP),
        )
        .expect("one station report");
        assert_eq!(one.summary.n_uncheckable, 2);
        assert!(one.per_observation.iter().all(|obs| obs.mdb_m.is_none()));
    }

    #[test]
    fn araim_external_reliability_reuses_gain_matrix_columns() {
        let geometry = simple_araim_geometry();
        let model = UnitProtectionModel;
        let options = options_with_lambda(LAMBDA_BAARDA_4DP);
        let report = reliability_araim(&geometry, &model, &options).expect("ARAIM report");
        let weights = vec![1.0; geometry.rows.len()];
        let gain = gain_matrix_enu(&geometry, &weights).expect("gain");

        for (idx, obs) in report.per_observation.iter().enumerate() {
            let mdb = obs.mdb_m.expect("MDB");
            let external = obs.external_enu_m.expect("external ENU");
            for (coord, value) in external.iter().enumerate() {
                assert_close(*value / mdb, gain.enu_rows[coord][idx], 2.0e-12);
            }
        }

        let design_rows = geometry
            .rows
            .iter()
            .map(|row| RangeReliabilityRow {
                id: row.id.to_string(),
                design_row: design_row(&geometry.clock_systems, row).expect("design row"),
                sigma_m: 1.0,
            })
            .collect::<Vec<_>>();
        let design_report = reliability_design(&design_rows, &options).expect("design report");
        let enu = ecef_to_enu_rotation(geometry.receiver.lat_rad, geometry.receiver.lon_rad);
        for (idx, obs) in design_report.per_observation.iter().enumerate() {
            let mdb = obs.mdb_m.expect("MDB");
            let state = obs.external_enu_m.expect("state external");
            let rotated = mat3_vec(enu, state);
            for (coord, value) in rotated.iter().enumerate() {
                assert_close(*value / mdb, gain.enu_rows[coord][idx], 2.0e-12);
            }
        }
    }

    #[test]
    fn invalid_inputs_use_quality_error_domains() {
        let mut rows = equal_precision_rows(2, 1.0);
        rows[0].sigma_m = 0.0;
        assert_eq!(
            reliability_design(&rows, &ReliabilityOptions::default()),
            Err(QualityError::InvalidWeight)
        );

        rows = equal_precision_rows(2, 1.0);
        let mut options = ReliabilityOptions {
            beta: 0.0,
            ..ReliabilityOptions::default()
        };
        assert_eq!(
            reliability_design(&rows, &options),
            Err(QualityError::InvalidReliabilityParameter)
        );
        options = ReliabilityOptions {
            lambda0_override: Some(f64::INFINITY),
            ..ReliabilityOptions::default()
        };
        assert_eq!(
            reliability_design(&rows, &options),
            Err(QualityError::InvalidReliabilityParameter)
        );
        options = ReliabilityOptions {
            min_redundancy: 0.0,
            ..ReliabilityOptions::default()
        };
        assert_eq!(
            reliability_design(&rows, &options),
            Err(QualityError::InvalidReliabilityParameter)
        );
        options = ReliabilityOptions {
            alpha: 1.0,
            ..ReliabilityOptions::default()
        };
        assert_eq!(
            reliability_design(&rows, &options),
            Err(QualityError::InvalidProbability)
        );

        let singular = vec![
            RangeReliabilityRow {
                id: "a".to_owned(),
                design_row: vec![1.0, 1.0],
                sigma_m: 1.0,
            },
            RangeReliabilityRow {
                id: "b".to_owned(),
                design_row: vec![2.0, 2.0],
                sigma_m: 1.0,
            },
        ];
        assert_eq!(
            reliability_design(&singular, &ReliabilityOptions::default()),
            Err(QualityError::SingularGeometry)
        );
    }

    struct UnitProtectionModel;

    impl ProtectionModel for UnitProtectionModel {
        fn sigma_int_m(&self, _row: &AraimRow) -> Result<f64, AraimError> {
            Ok(1.0)
        }

        fn sigma_acc_m(&self, _row: &AraimRow) -> Result<f64, AraimError> {
            Ok(1.0)
        }
    }

    fn options_with_lambda(lambda0: f64) -> ReliabilityOptions {
        ReliabilityOptions {
            lambda0_override: Some(lambda0),
            ..ReliabilityOptions::default()
        }
    }

    fn algebraic_rows() -> Vec<RangeReliabilityRow> {
        vec![
            row("r1", &[1.0, 0.3, -0.2], 0.9),
            row("r2", &[0.2, 1.1, 0.4], 1.1),
            row("r3", &[-0.5, 0.7, 1.0], 1.3),
            row("r4", &[1.2, -0.4, 0.6], 0.8),
            row("r5", &[-0.7, -0.9, 0.3], 1.5),
            row("r6", &[0.4, -0.2, -1.1], 1.2),
        ]
    }

    fn equal_precision_rows(n: usize, sigma_m: f64) -> Vec<RangeReliabilityRow> {
        (0..n)
            .map(|idx| RangeReliabilityRow {
                id: format!("obs{}", idx + 1),
                design_row: vec![1.0],
                sigma_m,
            })
            .collect()
    }

    fn leveling_connection_rows(
        n: usize,
        sigma_h: f64,
        sigma_h_cap: f64,
    ) -> Vec<RangeReliabilityRow> {
        let mut rows = Vec::with_capacity(2 * n);
        for idx in 0..n {
            let mut design = vec![0.0; n + 1];
            design[idx] = 1.0;
            design[n] = 1.0;
            rows.push(RangeReliabilityRow {
                id: format!("h{}", idx + 1),
                design_row: design,
                sigma_m: sigma_h,
            });
        }
        for idx in 0..n {
            let mut design = vec![0.0; n + 1];
            design[idx] = 1.0;
            rows.push(RangeReliabilityRow {
                id: format!("H{}", idx + 1),
                design_row: design,
                sigma_m: sigma_h_cap,
            });
        }
        rows
    }

    fn row(id: &str, design_row: &[f64], sigma_m: f64) -> RangeReliabilityRow {
        RangeReliabilityRow {
            id: id.to_owned(),
            design_row: design_row.to_vec(),
            sigma_m,
        }
    }

    fn simple_araim_geometry() -> AraimGeometry {
        const INV_SQRT_3: f64 = 0.5773502691896258;
        let receiver = Wgs84Geodetic::new(0.5, 0.2, 0.0).expect("valid receiver");
        let vectors = [
            [INV_SQRT_3, INV_SQRT_3, INV_SQRT_3],
            [INV_SQRT_3, -INV_SQRT_3, -INV_SQRT_3],
            [-INV_SQRT_3, INV_SQRT_3, -INV_SQRT_3],
            [-INV_SQRT_3, -INV_SQRT_3, INV_SQRT_3],
            [0.2, 0.6, 0.7745966692414834],
        ];
        let rows = vectors
            .iter()
            .enumerate()
            .map(|(idx, los)| AraimRow {
                id: GnssSatelliteId::new(GnssSystem::Gps, (idx + 1) as u8)
                    .expect("valid satellite"),
                line_of_sight: LineOfSight::new(los[0], los[1], los[2]),
                system: GnssSystem::Gps,
                elevation_rad: 0.8,
            })
            .collect();
        AraimGeometry {
            rows,
            receiver,
            clock_systems: vec![GnssSystem::Gps],
        }
    }

    fn design_matrix(rows: &[RangeReliabilityRow]) -> Vec<Vec<f64>> {
        rows.iter().map(|row| row.design_row.clone()).collect()
    }

    fn weight_matrix(rows: &[RangeReliabilityRow]) -> Vec<Vec<f64>> {
        let mut w = vec![vec![0.0; rows.len()]; rows.len()];
        for (idx, row) in rows.iter().enumerate() {
            w[idx][idx] = 1.0 / (row.sigma_m * row.sigma_m);
        }
        w
    }

    fn residual_projector(rows: &[RangeReliabilityRow], q_xx: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let h = design_matrix(rows);
        let h_t = transpose(&h);
        let w = weight_matrix(rows);
        let h_q = mat_mul(&h, q_xx);
        let h_q_h_t_w = mat_mul(&mat_mul(&h_q, &h_t), &w);
        let mut out = identity(rows.len());
        for i in 0..rows.len() {
            for j in 0..rows.len() {
                out[i][j] -= h_q_h_t_w[i][j];
            }
        }
        out
    }

    fn q_v_matrix(rows: &[RangeReliabilityRow], q_xx: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let h = design_matrix(rows);
        let h_t = transpose(&h);
        let h_q_h_t = mat_mul(&mat_mul(&h, q_xx), &h_t);
        let mut q_v = vec![vec![0.0; rows.len()]; rows.len()];
        for i in 0..rows.len() {
            q_v[i][i] = rows[i].sigma_m * rows[i].sigma_m;
            for j in 0..rows.len() {
                q_v[i][j] -= h_q_h_t[i][j];
            }
        }
        q_v
    }

    fn normal_matrix(rows: &[RangeReliabilityRow]) -> Vec<Vec<f64>> {
        let n_params = rows[0].design_row.len();
        normal_equations_weighted(
            rows.iter()
                .map(|row| (row.design_row.as_slice(), 0.0, 1.0 / row.sigma_m)),
            n_params,
        )
        .expect("normal matrix")
        .0
    }

    fn full_state_external(row: &RangeReliabilityRow, q_xx: &[Vec<f64>], mdb_m: f64) -> Vec<f64> {
        let scale = mdb_m / (row.sigma_m * row.sigma_m);
        mat_vec(q_xx, &row.design_row)
            .iter()
            .map(|value| value * scale)
            .collect()
    }

    fn identity(n: usize) -> Vec<Vec<f64>> {
        let mut out = vec![vec![0.0; n]; n];
        for (idx, row) in out.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        out
    }

    fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let b_t = transpose(b);
        a.iter()
            .map(|row| b_t.iter().map(|col| dot(row, col)).collect())
            .collect()
    }

    fn mat_sub(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
        a.iter()
            .zip(b)
            .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x - y).collect())
            .collect()
    }

    fn transpose(a: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let cols = a[0].len();
        (0..cols)
            .map(|col| a.iter().map(|row| row[col]).collect())
            .collect()
    }

    fn trace(a: &[Vec<f64>]) -> f64 {
        a.iter().enumerate().map(|(idx, row)| row[idx]).sum()
    }

    fn inf_norm(a: &[Vec<f64>]) -> f64 {
        a.iter()
            .map(|row| row.iter().map(|value| value.abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    fn quadratic_form(a: &[Vec<f64>], x: &[f64]) -> f64 {
        dot(x, &mat_vec(a, x))
    }

    fn mat3_vec(a: [[f64; 3]; 3], x: [f64; 3]) -> [f64; 3] {
        [
            a[0][0] * x[0] + a[0][1] * x[1] + a[0][2] * x[2],
            a[1][0] * x[0] + a[1][1] * x[1] + a[1][2] * x[2],
            a[2][0] * x[0] + a[2][1] * x[1] + a[2][2] * x[2],
        ]
    }

    fn assert_close(actual: f64, expected: f64, tol: f64) {
        assert!(
            (actual - expected).abs() <= tol,
            "actual={actual:.17e} expected={expected:.17e} tol={tol:.17e}"
        );
    }
}
