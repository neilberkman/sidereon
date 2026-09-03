//! Repeating-geometry residual filtering.
//!
//! This module is a pure post-processing kernel for residual arrays. It models
//! local-environment repeat signatures as a phase-stacked template at a supplied
//! repeat period, then subtracts only corrections backed by the requested prior
//! coverage.

use std::f64::consts::PI;

use crate::astro::time::Duration;
use crate::constants::SECONDS_PER_DAY;
use crate::ephemeris::BroadcastEphemeris;
use crate::estimation::{ewma_update, mad_spread, PrimitiveError};
use crate::{GnssSatelliteId, GnssSystem};

/// Length of the mean sidereal day used for constellation-default GPS phasing.
///
/// The value is 23 h 56 min 4.0905 s, stored as an integer nanosecond duration
/// so [`repeat_period`] does not depend on floating-point rounding.
pub const SIDEREAL_DAY_NANOS: i128 = 86_164_090_500_000;

/// Length of the mean sidereal day in SI seconds.
pub const SIDEREAL_DAY_SECONDS: f64 = 86_164.090_5;

/// Errors returned by sidereal filtering and repeat-period helpers.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SiderealFilterError {
    /// A public input was not usable.
    #[error("invalid sidereal filter {field}: {reason}")]
    InvalidInput {
        /// Field name for the failing input.
        field: &'static str,
        /// Human-readable reason for the failure.
        reason: &'static str,
    },
    /// No broadcast record covered the requested satellite and epoch.
    #[error("no broadcast record for {sat} at requested epoch")]
    NoBroadcastRecord {
        /// Satellite without a selected broadcast record.
        sat: GnssSatelliteId,
    },
    /// The requested constellation has no Keplerian broadcast-repeat path.
    #[error("unsupported orbit repeat constellation {system}")]
    UnsupportedConstellation {
        /// Unsupported constellation.
        system: GnssSystem,
    },
}

/// Template estimator used by [`sidereal_filter`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SiderealTemplateMethod {
    /// Arithmetic mean of the prior values in the same phase bin.
    #[default]
    Mean,
    /// MAD-gated mean of the prior values in the same phase bin.
    RobustMad,
    /// Exponentially weighted mean of prior values in the same phase bin.
    Ewma {
        /// EWMA gain in `[0, 1]`.
        alpha: f64,
    },
}

/// Options controlling residual template stacking.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct SiderealFilterOptions {
    /// Sampling interval of the residual array.
    pub sample_interval: Duration,
    /// Maximum number of prior repeats retained per phase bin.
    pub prior_periods: usize,
    /// Minimum prior samples required before a correction is applied.
    pub min_coverage: usize,
    /// Template estimator applied to prior samples.
    pub template_method: SiderealTemplateMethod,
}

impl Default for SiderealFilterOptions {
    fn default() -> Self {
        Self {
            sample_interval: Duration::from_nanos(1_000_000_000),
            prior_periods: 1,
            min_coverage: 1,
            template_method: SiderealTemplateMethod::Mean,
        }
    }
}

/// Output of [`sidereal_filter`].
#[derive(Debug, Clone, PartialEq)]
pub struct SiderealFilterOutput {
    /// Residuals after subtracting covered repeat-template corrections.
    pub filtered: Vec<f64>,
    /// Last correction template applied for each phase bin.
    ///
    /// Bins that never reached [`SiderealFilterOptions::min_coverage`] are
    /// stored as `NaN` and marked in [`Self::under_covered`].
    pub template: Vec<f64>,
    /// Prior-sample count backing the last template decision for each phase bin.
    pub coverage: Vec<usize>,
    /// `true` for phase bins whose last decision had insufficient coverage.
    pub under_covered: Vec<bool>,
}

/// Return the default ground-track repeat period for a GNSS constellation.
///
/// GPS, QZSS, NavIC, and SBAS default to one mean sidereal day. GLONASS,
/// Galileo, and BeiDou use the design repeat cycles of 8, 10, and 7 sidereal
/// days respectively. Use [`orbit_repeat_lag`] when a GPS-like per-satellite
/// broadcast-orbit refinement is available.
pub fn repeat_period(system: GnssSystem) -> Duration {
    let sidereal_days = match system {
        GnssSystem::Gps | GnssSystem::Qzss | GnssSystem::Navic | GnssSystem::Sbas => 1,
        GnssSystem::Glonass => 8,
        GnssSystem::Galileo => 10,
        GnssSystem::BeiDou => 7,
    };
    Duration::from_nanos(SIDEREAL_DAY_NANOS * sidereal_days)
}

/// Compute a per-satellite orbital repeat lag from a broadcast ephemeris.
///
/// The selected broadcast record supplies the square root of semi-major axis
/// and the mean-motion correction. The returned lag is `4*pi/n`, where
/// `n = sqrt(GM / a^3) + delta_n`, matching the orbit-repeat construction used
/// for modified sidereal filtering.
pub fn orbit_repeat_lag(
    ephemeris: &BroadcastEphemeris,
    sat: GnssSatelliteId,
    near_epoch_j2000_s: f64,
) -> Result<Duration, SiderealFilterError> {
    validate_finite(near_epoch_j2000_s, "near_epoch_j2000_s")?;
    if !matches!(
        sat.system,
        GnssSystem::Gps | GnssSystem::Galileo | GnssSystem::BeiDou | GnssSystem::Qzss
    ) {
        return Err(SiderealFilterError::UnsupportedConstellation { system: sat.system });
    }
    let record = ephemeris
        .select_record_at(sat, near_epoch_j2000_s)
        .ok_or(SiderealFilterError::NoBroadcastRecord { sat })?;
    let repeat_s = orbit_repeat_seconds(
        record.elements.sqrt_a,
        record.elements.delta_n,
        record.constants().gm_m3_s2,
    )?;
    Duration::from_seconds(repeat_s).map_err(|_| invalid_input("orbit_repeat_lag", "out of range"))
}

/// Remove a repeating residual component by phase-stacking prior repeats.
///
/// Samples are assigned to phase bins by `(index * sample_interval) mod period`.
/// A correction is subtracted only after a phase bin has at least
/// [`SiderealFilterOptions::min_coverage`] prior samples. Under-covered samples
/// are returned unchanged and the corresponding phase bins are flagged.
pub fn sidereal_filter(
    series: &[f64],
    period: Duration,
    options: SiderealFilterOptions,
) -> Result<SiderealFilterOutput, SiderealFilterError> {
    validate_series(series)?;
    validate_options(options)?;
    let period_s = validate_positive_duration(period, "period")?;
    let sample_interval_s =
        validate_positive_duration(options.sample_interval, "options.sample_interval")?;
    let bins = phase_bin_count(period_s, sample_interval_s)?;

    let mut histories = vec![Vec::<f64>::new(); bins];
    let mut filtered = Vec::with_capacity(series.len());
    let mut template = vec![f64::NAN; bins];
    let mut coverage = vec![0usize; bins];
    let mut under_covered = vec![true; bins];

    for (sample_index, &sample) in series.iter().enumerate() {
        let bin = phase_bin(sample_index, period_s, sample_interval_s, bins);
        let history = &histories[bin];
        let prior_count = history.len();
        coverage[bin] = prior_count;
        under_covered[bin] = prior_count < options.min_coverage;

        if prior_count >= options.min_coverage {
            let correction = estimate_template(history, options.template_method)?;
            template[bin] = correction;
            filtered.push(sample - correction);
        } else {
            filtered.push(sample);
        }

        let history = &mut histories[bin];
        history.push(sample);
        while history.len() > options.prior_periods {
            history.remove(0);
        }
    }

    Ok(SiderealFilterOutput {
        filtered,
        template,
        coverage,
        under_covered,
    })
}

/// Score repeating components at candidate periods for 1 Hz samples.
///
/// The returned strength is a robust variance-reduction score in `[0, 1]`,
/// where `1` means the period template removes all robust spread and `0` means
/// no robust spread reduction was measured. Use
/// [`periodicity_strength_with_sample_interval`] for other sample cadences.
pub fn periodicity_strength(
    series: &[f64],
    candidate_periods: &[Duration],
) -> Result<Vec<(Duration, f64)>, SiderealFilterError> {
    periodicity_strength_with_sample_interval(
        series,
        candidate_periods,
        Duration::from_nanos(1_000_000_000),
    )
}

/// Score repeating components at candidate periods for an explicit cadence.
///
/// Each candidate period is converted to phase bins at `sample_interval`.
/// Bins with fewer than two samples do not contribute a confident template.
/// The score uses [`mad_spread`] for both the input and period-template
/// residual spread.
pub fn periodicity_strength_with_sample_interval(
    series: &[f64],
    candidate_periods: &[Duration],
    sample_interval: Duration,
) -> Result<Vec<(Duration, f64)>, SiderealFilterError> {
    validate_series(series)?;
    if series.len() < 2 {
        return Err(invalid_input("series", "must contain at least two samples"));
    }
    let sample_interval_s = validate_positive_duration(sample_interval, "sample_interval")?;
    let baseline = mad_spread(series, 0.0).map_err(map_primitive_error)?;
    let mut scores = Vec::with_capacity(candidate_periods.len());

    for &period in candidate_periods {
        let period_s = validate_positive_duration(period, "candidate_periods")?;
        let bins = phase_bin_count(period_s, sample_interval_s)?;
        let strength = if baseline == 0.0 {
            0.0
        } else {
            periodicity_strength_one(series, period_s, sample_interval_s, bins, baseline)?
        };
        scores.push((period, strength));
    }

    Ok(scores)
}

fn periodicity_strength_one(
    series: &[f64],
    period_s: f64,
    sample_interval_s: f64,
    bins: usize,
    baseline: f64,
) -> Result<f64, SiderealFilterError> {
    let mut phases = vec![Vec::<f64>::new(); bins];
    for (sample_index, &sample) in series.iter().enumerate() {
        let bin = phase_bin(sample_index, period_s, sample_interval_s, bins);
        phases[bin].push(sample);
    }

    let mut templates = vec![f64::NAN; bins];
    let mut covered_bins = 0usize;
    for (bin, values) in phases.iter().enumerate() {
        if values.len() >= 2 {
            templates[bin] = robust_mad_template(values)?;
            covered_bins += 1;
        }
    }
    if covered_bins == 0 {
        return Ok(0.0);
    }

    let residuals = series
        .iter()
        .enumerate()
        .map(|(sample_index, &sample)| {
            let bin = phase_bin(sample_index, period_s, sample_interval_s, bins);
            let correction = templates[bin];
            if correction.is_finite() {
                sample - correction
            } else {
                sample
            }
        })
        .collect::<Vec<_>>();
    let residual_spread = mad_spread(&residuals, 0.0).map_err(map_primitive_error)?;
    let raw = 1.0 - (residual_spread / baseline).powi(2);
    Ok(raw.clamp(0.0, 1.0))
}

fn orbit_repeat_seconds(
    sqrt_a: f64,
    delta_n: f64,
    gm_m3_s2: f64,
) -> Result<f64, SiderealFilterError> {
    validate_positive(sqrt_a, "record.elements.sqrt_a")?;
    validate_finite(delta_n, "record.elements.delta_n")?;
    validate_positive(gm_m3_s2, "record.constants.gm_m3_s2")?;
    let a = sqrt_a * sqrt_a;
    let n0 = (gm_m3_s2 / (a * a * a)).sqrt();
    let n = n0 + delta_n;
    validate_positive(n, "record.mean_motion")?;
    let repeat_s = 4.0 * PI / n;
    validate_positive(repeat_s, "orbit_repeat_lag")
}

fn validate_options(options: SiderealFilterOptions) -> Result<(), SiderealFilterError> {
    if options.prior_periods == 0 {
        return Err(invalid_input("options.prior_periods", "must be positive"));
    }
    if options.min_coverage == 0 {
        return Err(invalid_input("options.min_coverage", "must be positive"));
    }
    match options.template_method {
        SiderealTemplateMethod::Mean | SiderealTemplateMethod::RobustMad => {}
        SiderealTemplateMethod::Ewma { alpha } => {
            validate_finite(alpha, "options.template_method.alpha")?;
            if !(0.0..=1.0).contains(&alpha) {
                return Err(invalid_input(
                    "options.template_method.alpha",
                    "must be in [0, 1]",
                ));
            }
        }
    }
    Ok(())
}

fn validate_series(series: &[f64]) -> Result<(), SiderealFilterError> {
    for &sample in series {
        validate_finite(sample, "series")?;
    }
    Ok(())
}

fn validate_positive_duration(
    duration: Duration,
    field: &'static str,
) -> Result<f64, SiderealFilterError> {
    if duration.nanos <= 0 {
        return Err(invalid_input(field, "must be positive"));
    }
    let seconds = duration.as_seconds();
    validate_positive(seconds, field)
}

fn validate_finite(value: f64, field: &'static str) -> Result<f64, SiderealFilterError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid_input(field, "must be finite"))
    }
}

fn validate_positive(value: f64, field: &'static str) -> Result<f64, SiderealFilterError> {
    validate_finite(value, field)?;
    if value > 0.0 {
        Ok(value)
    } else {
        Err(invalid_input(field, "must be positive"))
    }
}

fn phase_bin_count(period_s: f64, sample_interval_s: f64) -> Result<usize, SiderealFilterError> {
    let bins = (period_s / sample_interval_s).ceil();
    if !bins.is_finite() || bins < 1.0 || bins > usize::MAX as f64 {
        return Err(invalid_input("period", "phase-bin count is out of range"));
    }
    Ok(bins as usize)
}

fn phase_bin(sample_index: usize, period_s: f64, sample_interval_s: f64, bins: usize) -> usize {
    let phase_s = (sample_index as f64 * sample_interval_s).rem_euclid(period_s);
    let bin = (phase_s / sample_interval_s).floor() as usize;
    bin.min(bins - 1)
}

fn estimate_template(
    values: &[f64],
    method: SiderealTemplateMethod,
) -> Result<f64, SiderealFilterError> {
    match method {
        SiderealTemplateMethod::Mean => mean(values),
        SiderealTemplateMethod::RobustMad => robust_mad_template(values),
        SiderealTemplateMethod::Ewma { alpha } => ewma_template(values, alpha),
    }
}

fn mean(values: &[f64]) -> Result<f64, SiderealFilterError> {
    if values.is_empty() {
        return Err(invalid_input("values", "must not be empty"));
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

fn ewma_template(values: &[f64], alpha: f64) -> Result<f64, SiderealFilterError> {
    let Some((&first, rest)) = values.split_first() else {
        return Err(invalid_input("values", "must not be empty"));
    };
    let mut template = first;
    for &sample in rest {
        template = ewma_update(template, sample, alpha).map_err(map_primitive_error)?;
    }
    Ok(template)
}

fn robust_mad_template(values: &[f64]) -> Result<f64, SiderealFilterError> {
    let median = median(values)?;
    let spread = mad_spread(values, 0.0).map_err(map_primitive_error)?;
    if spread == 0.0 {
        return Ok(median);
    }
    let gate = 3.0 * spread;
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        if (value - median).abs() <= gate {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        Ok(median)
    } else {
        Ok(sum / count as f64)
    }
}

fn median(values: &[f64]) -> Result<f64, SiderealFilterError> {
    let mut sorted = values.to_vec();
    crate::astro::math::robust::median_sorting_in_place(&mut sorted)
        .ok_or_else(|| invalid_input("values", "must not be empty"))
}

fn map_primitive_error(error: PrimitiveError) -> SiderealFilterError {
    match error {
        PrimitiveError::InvalidInput { field, reason } => {
            SiderealFilterError::InvalidInput { field, reason }
        }
    }
}

const fn invalid_input(field: &'static str, reason: &'static str) -> SiderealFilterError {
    SiderealFilterError::InvalidInput { field, reason }
}

/// Return the 24-hour solar day as a duration for diagnostic callers.
pub fn solar_day_period() -> Duration {
    Duration::from_nanos((SECONDS_PER_DAY as i128) * 1_000_000_000)
}

#[cfg(test)]
mod tests {
    //! Validation provenance:
    //!
    //! - Sidereal day: the mean sidereal day constant is checked at the exact
    //!   integer nanosecond value for 23 h 56 min 4.0905 s.
    //! - Agnew and Larson, "Finding the Repeat Times of the GPS Constellation":
    //!   the orbit-repeat formula `n = sqrt(GM / a^3) + delta_n`,
    //!   `To = 4*pi/n` is checked against the Table 1 SV 1 orbit repeat time
    //!   of 86152.95 s.
    //! - Synthetic filtering tests use exact binary residual templates so the
    //!   recovered template and current-cycle noise floor are bit-exact.

    use super::*;
    use crate::astro::time::GnssWeekTow;
    use crate::broadcast::{ClockPolynomial, ConstellationConstants, KeplerianElements};
    use crate::constants::{GPS_EPOCH_TO_J2000_S, SECONDS_PER_HOUR, SECONDS_PER_WEEK};
    use crate::rinex_nav::{BroadcastGroupDelays, BroadcastIssue, BroadcastRecord, NavMessage};

    #[test]
    fn repeat_periods_use_documented_sidereal_day_multiples() {
        assert_eq!(
            repeat_period(GnssSystem::Gps),
            Duration::from_nanos(SIDEREAL_DAY_NANOS)
        );
        assert_eq!(
            repeat_period(GnssSystem::Glonass),
            Duration::from_nanos(SIDEREAL_DAY_NANOS * 8)
        );
        assert_eq!(
            repeat_period(GnssSystem::Galileo),
            Duration::from_nanos(SIDEREAL_DAY_NANOS * 10)
        );
        assert_eq!(
            repeat_period(GnssSystem::BeiDou),
            Duration::from_nanos(SIDEREAL_DAY_NANOS * 7)
        );
    }

    #[test]
    fn orbit_repeat_lag_matches_agnew_larson_sv1_table_value() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("valid satellite");
        let target_repeat_s = 86_152.95;
        let record = agnew_larson_table_record(sat, target_repeat_s);
        let week = record.toe.week;
        let toe = record.toe.tow_s;
        let store = BroadcastEphemeris::new(vec![record]).expect("valid broadcast store");
        let near_epoch_j2000_s = f64::from(week) * SECONDS_PER_WEEK + toe - GPS_EPOCH_TO_J2000_S;

        let lag = orbit_repeat_lag(&store, sat, near_epoch_j2000_s)
            .expect("orbit repeat lag from selected record");
        let repeat_s = lag.as_seconds();

        assert!(
            (repeat_s - target_repeat_s).abs() <= 1.0e-9,
            "repeat_s={repeat_s:.12}"
        );
    }

    #[test]
    fn sidereal_filter_recovers_exact_template_and_noise_floor() {
        let injected = [1.0, 0.5, -0.25, -1.25, 2.0, -0.75, 0.125, 0.875];
        let noise = [0.125, -0.25, 0.0625, 0.0, -0.125, 0.25, -0.0625, 0.1875];
        let mut series = injected.to_vec();
        series.extend(
            injected
                .iter()
                .zip(noise)
                .map(|(&signal, noise)| signal + noise),
        );

        let output = sidereal_filter(
            &series,
            Duration::from_nanos(8_000_000_000),
            SiderealFilterOptions::default(),
        )
        .expect("sidereal filter");

        assert_eq!(&output.template, &injected);
        assert_eq!(&output.filtered[8..], &noise);
        assert_eq!(variance(&output.filtered[8..]), variance(&noise));
        assert_eq!(output.coverage, vec![1; injected.len()]);
        assert_eq!(output.under_covered, vec![false; injected.len()]);
    }

    #[test]
    fn short_or_undercovered_series_is_flagged_and_left_unchanged() {
        let series = [0.25, -0.5, 0.75, -1.0, 1.25];
        let output = sidereal_filter(
            &series,
            Duration::from_nanos(4_000_000_000),
            SiderealFilterOptions {
                min_coverage: 2,
                ..SiderealFilterOptions::default()
            },
        )
        .expect("sidereal filter");

        assert_eq!(output.filtered, series);
        assert_eq!(output.coverage, vec![1, 0, 0, 0]);
        assert_eq!(output.under_covered, vec![true, true, true, true]);
        assert!(output.template.iter().all(|value| value.is_nan()));
    }

    #[test]
    fn periodicity_strength_separates_solar_and_sidereal_components() {
        let solar = solar_day_period();
        let sidereal = repeat_period(GnssSystem::Gps);
        let cadence = Duration::from_nanos(60_000_000_000);
        let samples = (5.0 * SECONDS_PER_DAY / cadence.as_seconds()) as usize;

        let solar_only = patterned_period_series(samples, solar, cadence, 1.0);
        let sidereal_only = patterned_period_series(samples, sidereal, cadence, 1.0);
        let mixed = solar_only
            .iter()
            .zip(&sidereal_only)
            .map(|(&solar, &sidereal)| solar + 0.5 * sidereal)
            .collect::<Vec<_>>();
        let candidates = [solar, sidereal];

        let solar_scores =
            periodicity_strength_with_sample_interval(&solar_only, &candidates, cadence)
                .expect("solar scores");
        let sidereal_scores =
            periodicity_strength_with_sample_interval(&sidereal_only, &candidates, cadence)
                .expect("sidereal scores");
        let mixed_scores = periodicity_strength_with_sample_interval(&mixed, &candidates, cadence)
            .expect("mixed scores");

        assert_eq!(solar_scores[0].1.to_bits(), 1.0f64.to_bits());
        assert_eq!(solar_scores[1].1.to_bits(), 0x3fd7_0a3d_70a3_d708);
        assert_eq!(sidereal_scores[0].1.to_bits(), 0x3fcf_db97_530e_ca84);
        assert_eq!(sidereal_scores[1].1.to_bits(), 1.0f64.to_bits());
        assert_eq!(mixed_scores[0].1.to_bits(), 0x3fe9_fdb9_7530_eca9);
        assert_eq!(mixed_scores[1].1.to_bits(), 0x3fd7_0a3d_70a3_d708);
    }

    fn agnew_larson_table_record(sat: GnssSatelliteId, repeat_s: f64) -> BroadcastRecord {
        let sqrt_a = 5_153.795_477_5;
        let a = sqrt_a * sqrt_a;
        let n0 = (ConstellationConstants::GPS.gm_m3_s2 / (a * a * a)).sqrt();
        let delta_n = 4.0 * PI / repeat_s - n0;
        let toe_sow = 100_000.0;
        let toe = GnssWeekTow::new(crate::astro::time::TimeScale::Gpst, 2_295, toe_sow)
            .expect("valid toe");
        BroadcastRecord {
            satellite_id: sat,
            message: NavMessage::GpsLnav,
            issue_of_data: BroadcastIssue {
                issue: 0,
                message: NavMessage::GpsLnav,
            },
            week: toe.week,
            toe,
            toc: toe,
            elements: KeplerianElements {
                sqrt_a,
                e: 0.01,
                m0: 0.0,
                delta_n,
                omega0: 1.0,
                i0: 0.94,
                omega: 0.25,
                omega_dot: -8.0e-9,
                idot: 0.0,
                cuc: 0.0,
                cus: 0.0,
                crc: 100.0,
                crs: -50.0,
                cic: 0.0,
                cis: 0.0,
                toe_sow,
            },
            clock: ClockPolynomial {
                af0: 0.0,
                af1: 0.0,
                af2: 0.0,
                toc_sow: toe_sow,
            },
            group_delays: BroadcastGroupDelays::default(),
            cnav: None,
            sv_health: 0.0,
            sv_accuracy_m: 0.0,
            fit_interval_s: Some(4.0 * SECONDS_PER_HOUR),
        }
    }

    fn variance(values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        values
            .iter()
            .map(|value| {
                let residual = value - mean;
                residual * residual
            })
            .sum::<f64>()
            / values.len() as f64
    }

    fn patterned_period_series(
        samples: usize,
        period: Duration,
        cadence: Duration,
        amplitude: f64,
    ) -> Vec<f64> {
        let pattern = [1.0, -0.5, 0.25, -1.0, 0.75];
        let period_s = period.as_seconds();
        let cadence_s = cadence.as_seconds();
        let bins = phase_bin_count(period_s, cadence_s).expect("phase bins");
        (0..samples)
            .map(|sample_index| {
                let bin = phase_bin(sample_index, period_s, cadence_s, bins);
                amplitude * pattern[bin % pattern.len()]
            })
            .collect()
    }
}
