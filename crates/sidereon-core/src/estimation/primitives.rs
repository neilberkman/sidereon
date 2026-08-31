//! Estimation and detection primitives.
//!
//! The primitives here are intentionally scalar, deterministic, and dependency-light.
//! They are a direct implementation point for embedded and desktop callers.
//!
//! ## Provenance
//!
//! - Alpha-beta and equivalent scalar 2-state constant-velocity Kalman gains use the
//!   Bar-Shalom steady-state family relations.
//! - Innovation gating and NIS monitoring follow the standard chi-square gate
//!   construction.
//! - CFAR formulas use the cell-averaging exponential-tail model from Richards.
//! - MAD and EWMA are the canonical scalar definitions used in this repository's
//!   robust-statistics layer, with the Gaussian consistency constant for MAD
//!   `1.482602...`.

use crate::quality;

/// Public error for estimation/detection primitive validation failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveError {
    /// Input field could not be used as given.
    InvalidInput {
        /// Field name in the failing input.
        field: &'static str,
        /// Human-readable reason for the failure.
        reason: &'static str,
    },
}

impl core::fmt::Display for PrimitiveError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, reason } => {
                write!(f, "invalid input {field}: {reason}")
            }
        }
    }
}

impl std::error::Error for PrimitiveError {}

/// State of a scalar level + rate estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaBetaState {
    /// Level estimate (e.g., position).
    pub level: f64,
    /// Rate estimate (e.g., velocity).
    pub rate: f64,
}

/// Alpha-beta steady-state gain set for one scalar channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaBetaGains {
    /// Level gain `α`.
    pub alpha: f64,
    /// Rate gain `β`, which maps innovation by `β * innovation / dt`.
    pub beta: f64,
}

/// One alpha-beta update result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AlphaBetaStep {
    /// Predicted state before the measurement is applied.
    pub predicted: AlphaBetaState,
    /// Corrected state after applying the measurement.
    pub updated: AlphaBetaState,
    /// Innovation `z - level_pred`.
    pub innovation: f64,
}

/// Steady-state gains of the equivalent 2-state scalar Kalman model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScalarKalmanGains {
    /// Position gain `K_x`.
    pub position_gain: f64,
    /// Rate gain `K_v` in 1/time units.
    pub rate_gain: f64,
}

/// Result of an NIS gate test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NisGate {
    /// Normalized innovation squared statistic.
    pub nis: f64,
    /// Chi-square threshold for the selected false-alarm probability and DOF.
    pub threshold: f64,
    /// `true` when `nis <= threshold`.
    pub in_gate: bool,
    /// Measurement dimension used by the gate.
    pub dof: usize,
}

/// MAD-Gaussian consistency factor `1 / Φ^{-1}(3/4)`.
pub const MAD_GAUSSIAN_CONSISTENCY: f64 = 1.482_602_218_505_602;

/// Compute steady-state alpha-beta gains from the tracking index `λ`.
///
/// Inputs:
///
/// - `tracking_index > 0`, dimensionless.
///
/// Output:
///
/// - `α` and `β` from the Bar-Shalom steady-state equations.
///
/// The pair is linked by `β^2 = λ (1 - α)` and the closure equation
/// `(2 - α) sqrt(λ(1 - α)) = α^2 + λ(1 - α)/6`.
///
/// Embedded consumers can keep the output as fixed-point coefficients for the
/// same predict/update form.
pub fn alpha_beta_steady_state_gains(
    tracking_index: f64,
) -> Result<AlphaBetaGains, PrimitiveError> {
    validate_finite_positive(tracking_index, "tracking_index")?;
    let alpha = alpha_beta_steady_state_alpha(tracking_index)?;
    let beta_sq = tracking_index * (1.0 - alpha);
    let beta = beta_sq.sqrt();
    Ok(AlphaBetaGains { alpha, beta })
}

/// Predict step for scalar alpha-beta.
///
/// `x⁻ = x + dt * v`, `v⁻ = v`.
///
/// The predict step is the same as a constant-rate model projection.
pub fn alpha_beta_predict(
    state: AlphaBetaState,
    dt: f64,
) -> Result<AlphaBetaState, PrimitiveError> {
    validate_finite_positive(dt, "dt")?;
    Ok(AlphaBetaState {
        level: state.level + dt * state.rate,
        rate: state.rate,
    })
}

/// Apply one measurement update on a predicted scalar alpha-beta state.
///
/// Update equations:
///
/// `x = x⁻ + α (z - x⁻)`  
/// `v = v⁻ + β/dt (z - x⁻)`.
pub fn alpha_beta_apply_measurement(
    predicted: AlphaBetaState,
    measurement: f64,
    dt: f64,
    gains: AlphaBetaGains,
) -> Result<AlphaBetaState, PrimitiveError> {
    validate_finite(predicted.level, "predicted_level")?;
    validate_finite(predicted.rate, "predicted_rate")?;
    validate_finite(measurement, "measurement")?;
    validate_finite_positive(gains.alpha, "alpha")?;
    validate_finite_nonnegative(gains.beta, "beta")?;
    if gains.alpha > 1.0 {
        return Err(invalid_input("alpha", "must be <= 1"));
    }
    validate_finite_positive(dt, "dt")?;

    let innovation = measurement - predicted.level;
    Ok(AlphaBetaState {
        level: predicted.level + gains.alpha * innovation,
        rate: predicted.rate + (gains.beta / dt) * innovation,
    })
}

/// Run one alpha-beta predict/update step from prior state to posterior state.
pub fn alpha_beta_filter_step(
    state: AlphaBetaState,
    measurement: f64,
    dt: f64,
    gains: AlphaBetaGains,
) -> Result<AlphaBetaStep, PrimitiveError> {
    let predicted = alpha_beta_predict(state, dt)?;
    let updated = alpha_beta_apply_measurement(predicted, measurement, dt, gains)?;
    let innovation = measurement - predicted.level;
    Ok(AlphaBetaStep {
        predicted,
        updated,
        innovation,
    })
}

/// Steady-state gains for a 2-state scalar constant-velocity Kalman filter.
///
/// The process model is
///
/// `x_{k+1} = x_k + dt * v_k + 0.5 dt^2 w_k`  
/// `v_{k+1} = v_k + dt w_k`  
///
/// with
/// `Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]` and `q = λ * R / dt^3`.
///
/// The equivalent relation to alpha-beta is `position_gain = α` and
/// `rate_gain * dt = β`.
pub fn kalman_cv_steady_state_gains(
    tracking_index: f64,
    dt: f64,
    measurement_variance: f64,
) -> Result<ScalarKalmanGains, PrimitiveError> {
    validate_finite_positive(tracking_index, "tracking_index")?;
    validate_finite_positive(dt, "dt")?;
    validate_finite_positive(measurement_variance, "measurement_variance")?;

    let q = tracking_index * measurement_variance / dt.powi(3);
    let q11 = q * dt;
    let q10 = q * dt * dt / 2.0;
    let q00 = q * dt * dt * dt / 3.0;

    let mut p00 = measurement_variance;
    let mut p01 = 0.0;
    let mut p11 = measurement_variance;
    for _ in 0..5_000 {
        let p00_pred = p00 + 2.0 * dt * p01 + dt * dt * p11 + q00;
        let p01_pred = p01 + dt * p11 + q10;
        let p11_pred = p11 + q11;

        let s = p00_pred + measurement_variance;
        validate_finite_positive(s, "measurement_gate_denominator")?;

        let k0 = p00_pred / s;
        let k1 = p01_pred / s;

        let p00_next = (1.0 - k0) * p00_pred;
        let p01_next = p01_pred - k1 * p00_pred;
        let p11_next = p11_pred - k1 * p01_pred;

        let delta = (p00_next - p00)
            .abs()
            .max((p01_next - p01).abs())
            .max((p11_next - p11).abs());
        p00 = p00_next;
        p01 = p01_next;
        p11 = p11_next;

        if delta <= 1.0e-15 * measurement_variance {
            break;
        }
    }

    let p00_pred = p00 + 2.0 * dt * p01 + dt * dt * p11 + q00;
    let p01_pred = p01 + dt * p11 + q10;
    let s = p00_pred + measurement_variance;
    let position_gain = p00_pred / s;
    let rate_gain = p01_pred / s;
    Ok(ScalarKalmanGains {
        position_gain,
        rate_gain,
    })
}

/// Scalar normalized innovation `ν / sqrt(S)`.
pub fn normalized_innovation(
    innovation: f64,
    innovation_variance: f64,
) -> Result<f64, PrimitiveError> {
    validate_finite(innovation, "innovation")?;
    validate_finite_positive(innovation_variance, "innovation_variance")?;
    Ok(innovation / innovation_variance.sqrt())
}

/// Scalar normalized innovation squared statistic.
pub fn nis_statistic(innovation: f64, innovation_variance: f64) -> Result<f64, PrimitiveError> {
    let normalized = normalized_innovation(innovation, innovation_variance)?;
    Ok(normalized * normalized)
}

/// Bar-Shalom chi-square expectation `E[NIS] = m`.
pub fn nis_expected_value(dof: usize) -> Result<f64, PrimitiveError> {
    if dof == 0 {
        return Err(invalid_input("dof", "must be positive"));
    }
    Ok(dof as f64)
}

/// Chi-square gate threshold for the requested confidence and DOF.
pub fn nis_gate_threshold(dof: usize, confidence: f64) -> Result<f64, PrimitiveError> {
    if dof == 0 {
        return Err(invalid_input("dof", "must be positive"));
    }
    validate_probability(confidence, "confidence")?;
    quality::chi2_inv(confidence, dof).map_err(|_| invalid_input("confidence", "no chi2 inverse"))
}

/// Test a measurement residual against a chi-square gate.
pub fn nis_gate_test(
    innovation: f64,
    innovation_variance: f64,
    dof: usize,
    confidence: f64,
) -> Result<NisGate, PrimitiveError> {
    let nis = nis_statistic(innovation, innovation_variance)?;
    let threshold = nis_gate_threshold(dof, confidence)?;
    Ok(NisGate {
        nis,
        threshold,
        in_gate: nis <= threshold,
        dof,
    })
}

/// EWMA update with generic gain `alpha`.
pub fn ewma_update(previous: f64, sample: f64, alpha: f64) -> Result<f64, PrimitiveError> {
    validate_finite(previous, "previous")?;
    validate_finite(sample, "sample")?;
    validate_fraction_or_one(alpha, "alpha")?;
    Ok(previous + alpha * (sample - previous))
}

/// EWMA update with power-of-two denominator, `alpha = 1 / 2^shift`.
///
/// Integer-friendly form:
/// `(1 - alpha) y[n-1] + alpha x[n]`
/// -> `((2^shift - 1) y[n-1] + x[n]) / 2^shift`.
pub fn ewma_update_power_of_two(
    previous: f64,
    sample: f64,
    shift: u32,
) -> Result<f64, PrimitiveError> {
    validate_finite(previous, "previous")?;
    validate_finite(sample, "sample")?;
    if shift > 52 {
        return Err(invalid_input("shift", "must be <= 52"));
    }
    let denominator = 1u64 << shift;
    let denominator = denominator as f64;
    let gain = 1.0 / denominator;
    if (denominator - 1.0).abs() < f64::EPSILON {
        Ok(sample)
    } else {
        Ok(previous + gain * (sample - previous))
    }
}

/// Median absolute deviation spread estimate with Gaussian consistency scaling.
pub fn mad_spread(values: &[f64], scale_floor: f64) -> Result<f64, PrimitiveError> {
    validate_finite_nonnegative(scale_floor, "scale_floor")?;
    let median_all = median(values)?;
    let mut deviations = values
        .iter()
        .map(|value| (value - median_all).abs())
        .collect::<Vec<_>>();
    if deviations.is_empty() {
        return Err(invalid_input("values", "must not be empty"));
    }
    deviations.sort_by(|a, b| a.total_cmp(b));
    let n = deviations.len();
    let mad = if n % 2 == 1 {
        deviations[n / 2]
    } else {
        (deviations[n / 2 - 1] + deviations[n / 2]) * 0.5
    };
    let scaled = MAD_GAUSSIAN_CONSISTENCY * mad;
    if scaled > scale_floor {
        Ok(scaled)
    } else {
        Ok(scale_floor)
    }
}

/// CA-CFAR multiplier from target false-alarm probability.
///
/// For exponential noise samples and a search window `N`:
/// `Pfa = (1 + T/N)^(-N)` where `T` is the threshold multiplier on mean noise.
pub fn cfar_ca_multiplier_from_pfa(
    searched_cells: usize,
    false_alarm_probability: f64,
) -> Result<f64, PrimitiveError> {
    if searched_cells == 0 {
        return Err(invalid_input("searched_cells", "must be positive"));
    }
    validate_probability(false_alarm_probability, "false_alarm_probability")?;
    let n = searched_cells as f64;
    let inv = 1.0 / libm::pow(false_alarm_probability, 1.0 / n);
    Ok(n * (inv - 1.0))
}

/// CA-CFAR inverse map from threshold multiplier to false-alarm probability.
pub fn cfar_ca_pfa_from_multiplier(
    searched_cells: usize,
    multiplier: f64,
) -> Result<f64, PrimitiveError> {
    if searched_cells == 0 {
        return Err(invalid_input("searched_cells", "must be positive"));
    }
    validate_finite_positive(multiplier, "multiplier")?;
    let n = searched_cells as f64;
    let one_plus = 1.0 + multiplier / n;
    if !one_plus.is_finite() {
        return Err(invalid_input("multiplier", "not finite"));
    }
    Ok(libm::pow(one_plus, -n))
}

/// CA-CFAR absolute threshold from noise level, search window, and false-alarm target.
pub fn cfar_ca_threshold(
    searched_cells: usize,
    false_alarm_probability: f64,
    noise_level: f64,
) -> Result<f64, PrimitiveError> {
    validate_finite_positive(noise_level, "noise_level")?;
    let multiplier = cfar_ca_multiplier_from_pfa(searched_cells, false_alarm_probability)?;
    Ok(noise_level * multiplier)
}

/// CA-CFAR inverse map from absolute threshold to false-alarm probability.
pub fn cfar_ca_false_alarm_probability(
    searched_cells: usize,
    threshold: f64,
    noise_level: f64,
) -> Result<f64, PrimitiveError> {
    validate_finite_positive(noise_level, "noise_level")?;
    validate_finite_positive(threshold, "threshold")?;
    let multiplier = threshold / noise_level;
    cfar_ca_pfa_from_multiplier(searched_cells, multiplier)
}

fn alpha_beta_steady_state_alpha(tracking_index: f64) -> Result<f64, PrimitiveError> {
    let mut roots = Vec::new();
    let start = f64::from_bits(1);
    let end = 1.0 - start;
    let steps = 16_384usize;

    let mut previous_alpha = start;
    let mut previous_value = alpha_beta_equation(tracking_index, previous_alpha);
    if previous_value == 0.0 {
        return Ok(previous_alpha);
    }

    for idx in 1..=steps {
        let current_alpha = start + (end - start) * idx as f64 / steps as f64;
        let current_value = alpha_beta_equation(tracking_index, current_alpha);
        if current_value == 0.0 {
            roots.push(current_alpha);
        }
        if previous_value * current_value < 0.0 {
            let root = bisect_alpha_root(tracking_index, previous_alpha, current_alpha);
            roots.push(root);
        }
        previous_alpha = current_alpha;
        previous_value = current_value;
    }

    let alpha = roots.last().copied().ok_or_else(|| {
        invalid_input(
            "tracking_index",
            "no valid steady-state alpha root in (0,1)",
        )
    })?;
    Ok(alpha)
}

fn alpha_beta_equation(tracking_index: f64, alpha: f64) -> f64 {
    let one_minus_alpha = 1.0 - alpha;
    let spread = (tracking_index * one_minus_alpha).sqrt();
    (2.0 - alpha) * spread - (alpha * alpha + tracking_index * one_minus_alpha / 6.0)
}

fn bisect_alpha_root(tracking_index: f64, mut left: f64, mut right: f64) -> f64 {
    let mut f_left = alpha_beta_equation(tracking_index, left);
    let mut f_right = alpha_beta_equation(tracking_index, right);
    for _ in 0..80 {
        let mid = 0.5 * (left + right);
        let f_mid = alpha_beta_equation(tracking_index, mid);
        if f_left * f_mid <= 0.0 {
            right = mid;
            f_right = f_mid;
        } else {
            left = mid;
            f_left = f_mid;
        }
        if f_left == 0.0 || f_right == 0.0 {
            return 0.5 * (left + right);
        }
    }
    0.5 * (left + right)
}

fn median(values: &[f64]) -> Result<f64, PrimitiveError> {
    if values.is_empty() {
        return Err(invalid_input("values", "must not be empty"));
    }
    for value in values {
        validate_finite(*value, "values")?;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    Ok(if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    })
}

fn validate_finite(value: f64, field: &'static str) -> Result<(), PrimitiveError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn validate_finite_positive(value: f64, field: &'static str) -> Result<(), PrimitiveError> {
    validate_finite(value, field)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be positive"))
    }
}

fn validate_finite_nonnegative(value: f64, field: &'static str) -> Result<(), PrimitiveError> {
    validate_finite(value, field)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "must be non-negative"))
    }
}

fn validate_fraction_or_one(value: f64, field: &'static str) -> Result<(), PrimitiveError> {
    validate_finite(value, field)?;
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid_input(field, "must be in [0,1]"))
    }
}

fn validate_probability(value: f64, field: &'static str) -> Result<(), PrimitiveError> {
    validate_finite(value, field)?;
    if (0.0..1.0).contains(&value) {
        Ok(())
    } else {
        Err(invalid_input(field, "must be in (0,1)"))
    }
}

const fn invalid_input(field: &'static str, reason: &'static str) -> PrimitiveError {
    PrimitiveError::InvalidInput { field, reason }
}

#[cfg(test)]
mod tests {
    //! Test oracles in the order: alpha-beta and scalar Kalman equivalence, gating/NIS,
    //! MAD/EWMA behavior, and CA-CFAR threshold inversion.
    //!
    //! - Bar-Shalom alpha-beta + Kalman: fixed-point and closed-form residual equation.
    //! - Bar-Shalom NIS: chi-square thresholding and expectation.
    //! - Richards CA-CFAR: exact analytic target false-alarm law.

    use super::*;

    const KALMAN_TOLERANCE: f64 = 1.0e-12;

    #[test]
    fn alpha_beta_steady_state_matches_reference_and_kalman() {
        let gains = alpha_beta_steady_state_gains(4.0).expect("alpha-beta gains");
        assert!((gains.alpha - 0.864_145_399_682_717_8).abs() < KALMAN_TOLERANCE);
        assert!((gains.beta - 0.737_169_180_900_238_8).abs() < KALMAN_TOLERANCE);

        let kalman =
            kalman_cv_steady_state_gains(4.0, 1.0, 1.0).expect("kalman steady-state gains");
        assert!((kalman.position_gain - gains.alpha).abs() < KALMAN_TOLERANCE);
        assert!((kalman.rate_gain - gains.beta).abs() < KALMAN_TOLERANCE);
        assert!((kalman.position_gain * 1.0 - gains.alpha).abs() < KALMAN_TOLERANCE);
    }

    #[test]
    fn alpha_beta_step_response_settles_to_the_new_level() {
        // Step response (Bar-Shalom): a stable alpha-beta filter at rest, fed a
        // constant level, converges to that level with the rate estimate
        // returning to zero and the innovation decaying to zero.
        let gains = alpha_beta_steady_state_gains(4.0).expect("alpha-beta gains");
        let step_level = 10.0;
        let dt = 1.0;
        let mut state = AlphaBetaState {
            level: 0.0,
            rate: 0.0,
        };
        let mut last_innovation = f64::INFINITY;
        for _ in 0..300 {
            let step = alpha_beta_filter_step(state, step_level, dt, gains).expect("step");
            state = step.updated;
            last_innovation = step.innovation;
        }
        assert!(
            (state.level - step_level).abs() < 1e-9,
            "level did not settle: {}",
            state.level
        );
        assert!(
            state.rate.abs() < 1e-9,
            "rate did not settle: {}",
            state.rate
        );
        assert!(
            last_innovation.abs() < 1e-9,
            "innovation did not decay: {}",
            last_innovation
        );
    }

    #[test]
    fn alpha_beta_predict_and_update_are_dt_aware() {
        let state = AlphaBetaState {
            level: 5.0,
            rate: 2.0,
        };
        let gains = AlphaBetaGains {
            alpha: 0.6,
            beta: 0.8,
        };
        let step = alpha_beta_filter_step(state, 8.0, 2.0, gains).expect("step");

        assert_eq!(step.predicted.level, 9.0);
        assert_eq!(step.predicted.rate, 2.0);
        assert_eq!(step.innovation, -1.0);
        assert_eq!(step.updated.level, 8.4);
        assert_eq!(step.updated.rate, 1.6);
    }

    #[test]
    fn nis_gate_threshold_and_expectation() {
        let gate = nis_gate_test(1.0, 1.0, 1, 0.95).expect("nis gate");
        assert!((gate.threshold - 3.841_458_820_694_124).abs() < 1.0e-12);
        assert_eq!(gate.dof, 1);
        assert!(gate.in_gate);
        assert_eq!(nis_expected_value(3).expect("dof"), 3.0);
    }

    #[test]
    fn mad_gives_expected_gaussian_scaling_and_ewma_power_of_two_matches_alpha_form() {
        const Q75: f64 = 0.674_489_750_196_081_7;
        let sigma_est =
            mad_spread(&[-2.0 * Q75, -Q75, 0.0, Q75, 2.0 * Q75], 1.0e-12).expect("mad spread");
        assert!((sigma_est - 1.0).abs() < 1.0e-12);
        assert!((mad_spread(&[0.0, 0.0, 0.0], 1.2).expect("floored mad") - 1.2).abs() < 1.0e-12);

        let previous = 16.0;
        let sample = 2.0;
        let alpha = 1.0 / 16.0;
        let ewma_alpha = ewma_update(previous, sample, alpha).expect("ewma alpha");
        let ewma_pow2 = ewma_update_power_of_two(previous, sample, 4).expect("ewma pow2");
        let ewma_int = ((16.0_f64 - 1.0) * previous + sample) / 16.0;
        assert!((ewma_alpha - ewma_pow2).abs() < 1.0e-15);
        assert!((ewma_pow2 - ewma_int).abs() < 1.0e-15);
        assert!((ewma_alpha - 15.125).abs() < 1.0e-12);
    }

    #[test]
    fn cfar_threshold_and_probability_roundtrip() {
        let noise = 5.0;
        let pfa = 1.0e-3;
        let searched_cells = 4usize;
        let multiplier = cfar_ca_multiplier_from_pfa(searched_cells, pfa).expect("cfar multiplier");
        let threshold = cfar_ca_threshold(searched_cells, pfa, noise).expect("cfar threshold");
        let pfa_back = cfar_ca_false_alarm_probability(searched_cells, threshold, noise)
            .expect("cfar pfa back");
        assert!((multiplier - 18.493_653_007_613_965).abs() < 1.0e-12);
        assert_eq!(threshold, noise * multiplier);
        assert!((pfa_back - pfa).abs() < 1.0e-12);
    }
}
