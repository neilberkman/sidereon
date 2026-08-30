//! Temporal-correlation covariance deflation for static PPP.

use std::collections::BTreeMap;

use crate::astro::math::special::portable_log;
use crate::dop::PositionCovariance;

use super::{FloatEpoch, FloatResidual, TemporalCorrelationSummary};

const MAX_LAG1_AUTOCORRELATION: f64 = 0.99;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ObservableKind {
    Code,
    Phase,
}

pub(super) fn temporal_position_covariance(
    formal: PositionCovariance,
    posterior_variance_factor: f64,
    temporal: TemporalCorrelationSummary,
) -> (PositionCovariance, f64) {
    let scale_factor =
        posterior_variance_factor.max(1.0) * temporal.variance_inflation_factor.max(1.0);
    (
        PositionCovariance {
            ecef_m2: scale_3x3(formal.ecef_m2, scale_factor),
            enu_m2: scale_3x3(formal.enu_m2, scale_factor),
        },
        scale_factor,
    )
}

pub(super) fn estimate_temporal_correlation(
    residuals: &[FloatResidual],
    epochs: &[FloatEpoch],
) -> TemporalCorrelationSummary {
    let epoch_interval_s = regular_epoch_interval_s(epochs);
    let mut grouped: BTreeMap<(String, ObservableKind), Vec<(usize, f64)>> = BTreeMap::new();
    let mut nominal_sample_count = 0_usize;
    for residual in residuals {
        let code = residual.code_m * residual.code_weight;
        if code.is_finite() {
            grouped
                .entry((residual.satellite_id.clone(), ObservableKind::Code))
                .or_default()
                .push((residual.epoch_index, code));
            nominal_sample_count += 1;
        }
        let phase = residual.phase_m * residual.phase_weight;
        if phase.is_finite() {
            grouped
                .entry((residual.satellite_id.clone(), ObservableKind::Phase))
                .or_default()
                .push((residual.epoch_index, phase));
            nominal_sample_count += 1;
        }
    }

    let mut numerator = 0.0_f64;
    let mut previous_norm = 0.0_f64;
    let mut next_norm = 0.0_f64;
    let mut arc_lengths = Vec::new();
    for samples in grouped.values_mut() {
        samples.sort_by_key(|(epoch_index, _)| *epoch_index);
        let mut start = 0;
        while start < samples.len() {
            let mut end = start + 1;
            while end < samples.len() && samples[end].0 == samples[end - 1].0 + 1 {
                end += 1;
            }
            if let Some(arc) = analyze_arc(&samples[start..end]) {
                numerator += arc.numerator;
                previous_norm += arc.previous_norm;
                next_norm += arc.next_norm;
                arc_lengths.push(end - start);
            }
            start = end;
        }
    }

    if nominal_sample_count == 0 {
        return TemporalCorrelationSummary {
            lag1_autocorrelation: 0.0,
            decorrelation_time_epochs: 0.0,
            decorrelation_time_s: None,
            nominal_sample_count: 0,
            effective_sample_count: 0.0,
            variance_inflation_factor: 1.0,
            arcs_used: 0,
        };
    }

    let denominator = (previous_norm * next_norm).sqrt();
    let lag1_autocorrelation = if denominator > 0.0 {
        (numerator / denominator).clamp(0.0, MAX_LAG1_AUTOCORRELATION)
    } else {
        0.0
    };
    let variance_inflation_factor =
        pooled_ar1_variance_inflation(lag1_autocorrelation, &arc_lengths, nominal_sample_count);
    let effective_sample_count = nominal_sample_count as f64 / variance_inflation_factor;
    let decorrelation_time_epochs = decorrelation_time_epochs(lag1_autocorrelation);
    TemporalCorrelationSummary {
        lag1_autocorrelation,
        decorrelation_time_epochs,
        decorrelation_time_s: epoch_interval_s.map(|dt_s| decorrelation_time_epochs * dt_s),
        nominal_sample_count,
        effective_sample_count,
        variance_inflation_factor,
        arcs_used: arc_lengths.len(),
    }
}

fn regular_epoch_interval_s(epochs: &[FloatEpoch]) -> Option<f64> {
    if epochs.len() < 2 {
        return None;
    }
    let mut deltas = Vec::with_capacity(epochs.len() - 1);
    for pair in epochs.windows(2) {
        let delta = pair[1].t_rx_j2000_s - pair[0].t_rx_j2000_s;
        if !(delta.is_finite() && delta > 0.0) {
            return None;
        }
        deltas.push(delta);
    }
    let mean = deltas.iter().sum::<f64>() / deltas.len() as f64;
    let tolerance = mean.abs().max(1.0) * 1.0e-9;
    if deltas
        .iter()
        .all(|delta| (*delta - mean).abs() <= tolerance)
    {
        Some(mean)
    } else {
        None
    }
}

struct ArcAutocorrelation {
    numerator: f64,
    previous_norm: f64,
    next_norm: f64,
}

fn analyze_arc(samples: &[(usize, f64)]) -> Option<ArcAutocorrelation> {
    if samples.len() < 3 {
        return None;
    }
    let mean = samples.iter().map(|(_, value)| *value).sum::<f64>() / samples.len() as f64;
    let mut numerator = 0.0_f64;
    let mut previous_norm = 0.0_f64;
    let mut next_norm = 0.0_f64;
    for pair in samples.windows(2) {
        let previous = pair[0].1 - mean;
        let next = pair[1].1 - mean;
        numerator += previous * next;
        previous_norm += previous * previous;
        next_norm += next * next;
    }
    if previous_norm > 0.0 && next_norm > 0.0 {
        Some(ArcAutocorrelation {
            numerator,
            previous_norm,
            next_norm,
        })
    } else {
        None
    }
}

fn pooled_ar1_variance_inflation(
    rho: f64,
    arc_lengths: &[usize],
    nominal_sample_count: usize,
) -> f64 {
    if rho <= 0.0 || arc_lengths.is_empty() || nominal_sample_count == 0 {
        return 1.0;
    }
    let weighted_sum = arc_lengths
        .iter()
        .map(|&len| len as f64 * finite_ar1_variance_inflation(rho, len))
        .sum::<f64>();
    let factor = weighted_sum / nominal_sample_count as f64;
    if factor.is_finite() && factor >= 1.0 {
        factor
    } else {
        1.0
    }
}

fn finite_ar1_variance_inflation(rho: f64, n: usize) -> f64 {
    if n <= 1 || rho <= 0.0 {
        return 1.0;
    }
    let mut sum = 0.0_f64;
    let mut rho_k = rho;
    for lag in 1..n {
        sum += (1.0 - lag as f64 / n as f64) * rho_k;
        rho_k *= rho;
    }
    1.0 + 2.0 * sum
}

fn decorrelation_time_epochs(rho: f64) -> f64 {
    if rho > 0.0 {
        -1.0 / portable_log(rho)
    } else {
        0.0
    }
}

fn scale_3x3(mut matrix: [[f64; 3]; 3], factor: f64) -> [[f64; 3]; 3] {
    for row in &mut matrix {
        for value in row {
            *value *= factor;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ppp_corrections::CivilDateTime;

    const EPOCH_INTERVAL_S: f64 = 30.0;

    #[test]
    fn ar1_temporal_estimator_recovers_injected_decorrelation_time() {
        let injected_tau_s = 180.0;
        let rho = libm::exp(-EPOCH_INTERVAL_S / injected_tau_s);
        let residuals = synthetic_ar1_residuals(180, 8, rho, 0);
        let epochs = synthetic_epochs(180);

        let estimate = estimate_temporal_correlation(&residuals, &epochs);

        assert!(
            (0.78..=0.89).contains(&estimate.lag1_autocorrelation),
            "estimated rho {} did not recover injected rho {rho}",
            estimate.lag1_autocorrelation
        );
        assert!(
            (120.0..=260.0).contains(&estimate.decorrelation_time_s.unwrap()),
            "estimated decorrelation time {} s did not recover injected {injected_tau_s} s",
            estimate.decorrelation_time_s.unwrap()
        );
        assert!(estimate.variance_inflation_factor > 7.0);
        assert!(estimate.effective_sample_count < estimate.nominal_sample_count as f64 / 7.0);
    }

    #[test]
    fn white_temporal_estimator_does_not_distort_covariance() {
        let residuals = synthetic_ar1_residuals(180, 8, 0.0, 11);
        let epochs = synthetic_epochs(180);

        let estimate = estimate_temporal_correlation(&residuals, &epochs);

        assert!(
            estimate.lag1_autocorrelation <= 0.08,
            "white residuals estimated rho {}",
            estimate.lag1_autocorrelation
        );
        assert!(
            estimate.variance_inflation_factor <= 1.2,
            "white residuals inflated by {}",
            estimate.variance_inflation_factor
        );
    }

    #[test]
    fn ar1_coverage_is_calibrated_where_independent_covariance_undercovers() {
        let injected_tau_s = 180.0;
        let rho = libm::exp(-EPOCH_INTERVAL_S / injected_tau_s);
        let n = 240;
        let trials = 320;
        let mut independent_covered = 0_usize;
        let mut inflated_covered = 0_usize;
        let epochs = synthetic_epochs(n);
        for trial in 0..trials {
            let residuals = synthetic_ar1_residuals(n, 1, rho, trial as u64 + 101);
            let estimate = estimate_temporal_correlation(&residuals, &epochs);
            let mean = residuals.iter().map(|r| r.code_m).sum::<f64>() / n as f64;
            let independent_sigma = 1.0 / (n as f64).sqrt();
            let inflated_sigma = (estimate.variance_inflation_factor / n as f64).sqrt();
            if mean.abs() <= 2.0 * independent_sigma {
                independent_covered += 1;
            }
            if mean.abs() <= 2.0 * inflated_sigma {
                inflated_covered += 1;
            }
        }
        let independent_rate = independent_covered as f64 / trials as f64;
        let inflated_rate = inflated_covered as f64 / trials as f64;

        assert!(
            independent_rate < 0.75,
            "independent covariance coverage {independent_rate} was not undercovered"
        );
        assert!(
            (0.88..=0.99).contains(&inflated_rate),
            "inflated covariance coverage {inflated_rate} was not near nominal 2-sigma coverage"
        );
    }

    fn synthetic_epochs(n: usize) -> Vec<FloatEpoch> {
        (0..n)
            .map(|idx| {
                let total_s = idx * EPOCH_INTERVAL_S as usize;
                FloatEpoch {
                    epoch: CivilDateTime {
                        year: 2020,
                        month: 6,
                        day: 24,
                        hour: ((total_s / 3600) % 24) as u8,
                        minute: ((total_s % 3600) / 60) as u8,
                        second: (total_s % 60) as f64,
                    },
                    jd_whole: 2_459_024.5,
                    jd_fraction: 0.5
                        + idx as f64 * EPOCH_INTERVAL_S / crate::constants::SECONDS_PER_DAY,
                    t_rx_j2000_s: idx as f64 * EPOCH_INTERVAL_S,
                    observations: Vec::new(),
                }
            })
            .collect()
    }

    fn synthetic_ar1_residuals(
        n_epochs: usize,
        n_sats: usize,
        rho: f64,
        seed_offset: u64,
    ) -> Vec<FloatResidual> {
        let innovation_scale = (1.0 - rho * rho).sqrt();
        let mut residuals = Vec::with_capacity(n_epochs * n_sats);
        for sat_idx in 0..n_sats {
            let mut rng = Lcg::new(0x5eed_1234_u64 + seed_offset + sat_idx as u64 * 997);
            let mut code = gaussian_pair(&mut rng).0;
            let mut phase = gaussian_pair(&mut rng).1;
            for epoch_index in 0..n_epochs {
                let (code_innovation, phase_innovation) = gaussian_pair(&mut rng);
                code = rho * code + innovation_scale * code_innovation;
                phase = rho * phase + innovation_scale * phase_innovation;
                residuals.push(FloatResidual {
                    epoch_index,
                    satellite_id: format!("G{:02}", sat_idx + 1),
                    code_m: code,
                    phase_m: phase,
                    code_weight: 1.0,
                    phase_weight: 1.0,
                });
            }
        }
        residuals
    }

    fn gaussian_pair(rng: &mut Lcg) -> (f64, f64) {
        let u1 = rng.next_open01();
        let u2 = rng.next_open01();
        let radius = (-2.0 * portable_log(u1)).sqrt();
        let angle = 2.0 * std::f64::consts::PI * u2;
        (radius * libm::cos(angle), radius * libm::sin(angle))
    }

    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            self.state
        }

        fn next_open01(&mut self) -> f64 {
            let value = self.next_u64() >> 11;
            ((value as f64) + 0.5) / ((1_u64 << 53) as f64)
        }
    }
}
