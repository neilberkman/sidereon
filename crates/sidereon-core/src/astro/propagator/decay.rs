//! Orbital decay helper built on the numerical propagator and drag force.
//!
//! The scan samples geodetic altitude on a coarse elapsed-time grid and refines
//! the first endpoint that falls at or below the reentry altitude. A coarse step
//! longer than the time spent below the threshold at perigee can miss a brief
//! crossing; callers should keep `scan_step_s` well below the orbital period,
//! for example no more than about one twentieth of the period in near-circular
//! decay cases.

use crate::astro::atmosphere::MAX_ALTITUDE_KM;
use crate::astro::constants::{J2_EARTH, MU_EARTH, RE_EARTH};
use crate::astro::error::PropagationError;
use crate::astro::events::root::{try_bisect_crossing_until, RootError};
use crate::astro::forces::drag::geodetic_altitude_km;
use crate::astro::forces::{DragForce, DragParameters, SpaceWeatherSource};
use crate::astro::propagator::api::IntegratorOptions;
use crate::astro::propagator::driver::PropagationForceModel;
use crate::astro::propagator::numerical::{ForceModelKind, IntegratorKind, StatePropagator};
use crate::astro::state::CartesianState;

/// Result of an orbital-lifetime estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecayEstimate {
    /// Seconds from the initial epoch to reentry.
    pub time_to_decay_s: f64,
    /// State at reentry.
    pub reentry_state: CartesianState,
    /// Geodetic altitude at the reported state, km.
    pub reentry_altitude_km: f64,
}

/// Configuration for a decay run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DecayConfig {
    /// Gravity choice layered under drag.
    pub force_model: PropagationForceModel,
    /// Gravitational-parameter override, km^3/s^2.
    pub mu_km3_s2: Option<f64>,
    /// Numerical integrator.
    pub integrator: IntegratorKind,
    /// Integrator options.
    pub options: IntegratorOptions,
    /// Drag parameters.
    pub drag: DragParameters,
    /// Reentry threshold altitude, km.
    pub reentry_altitude_km: f64,
    /// Coarse scan step, s.
    pub scan_step_s: f64,
    /// Bisection time tolerance, s.
    pub crossing_tolerance_s: f64,
    /// Maximum elapsed scan horizon, s.
    pub max_duration_s: f64,
    /// Maximum number of coarse scan samples.
    pub max_scan_samples: u32,
}

impl DecayConfig {
    /// Default reentry altitude, km.
    pub const DEFAULT_REENTRY_ALTITUDE_KM: f64 = DragForce::DEFAULT_REENTRY_ALTITUDE_KM;
    /// Default coarse scan step, s.
    pub const DEFAULT_SCAN_STEP_S: f64 = 60.0;
    /// Default bisection time tolerance, s.
    pub const DEFAULT_CROSSING_TOLERANCE_S: f64 = 1.0;
    /// Default maximum coarse scan samples.
    pub const DEFAULT_MAX_SCAN_SAMPLES: u32 = 200_000;
    /// Default maximum elapsed scan horizon, s.
    pub const DEFAULT_MAX_DURATION_S: f64 =
        Self::DEFAULT_MAX_SCAN_SAMPLES as f64 * Self::DEFAULT_SCAN_STEP_S;

    /// Build a decay config around required drag parameters.
    pub fn new(drag: DragParameters) -> Self {
        Self {
            force_model: PropagationForceModel::default(),
            mu_km3_s2: None,
            integrator: IntegratorKind::Dp54,
            options: IntegratorOptions::default(),
            drag,
            reentry_altitude_km: Self::DEFAULT_REENTRY_ALTITUDE_KM,
            scan_step_s: Self::DEFAULT_SCAN_STEP_S,
            crossing_tolerance_s: Self::DEFAULT_CROSSING_TOLERANCE_S,
            max_duration_s: Self::DEFAULT_MAX_DURATION_S,
            max_scan_samples: Self::DEFAULT_MAX_SCAN_SAMPLES,
        }
    }

    /// Replace the gravity choice.
    pub fn with_force_model(mut self, force_model: PropagationForceModel) -> Self {
        self.force_model = force_model;
        self
    }

    /// Replace the gravitational parameter override.
    pub fn with_mu_km3_s2(mut self, mu_km3_s2: Option<f64>) -> Self {
        self.mu_km3_s2 = mu_km3_s2;
        self
    }

    /// Replace the integrator.
    pub fn with_integrator(mut self, integrator: IntegratorKind) -> Self {
        self.integrator = integrator;
        self
    }

    /// Replace the integrator options.
    pub fn with_options(mut self, options: IntegratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Replace the reentry altitude, km.
    pub fn with_reentry_altitude_km(mut self, reentry_altitude_km: f64) -> Self {
        self.reentry_altitude_km = reentry_altitude_km;
        self
    }

    /// Replace the coarse scan step, s.
    pub fn with_scan_step_s(mut self, scan_step_s: f64) -> Self {
        self.scan_step_s = scan_step_s;
        self
    }

    /// Replace the crossing tolerance, s.
    pub fn with_crossing_tolerance_s(mut self, crossing_tolerance_s: f64) -> Self {
        self.crossing_tolerance_s = crossing_tolerance_s;
        self
    }

    /// Replace the maximum scan horizon, s.
    pub fn with_max_duration_s(mut self, max_duration_s: f64) -> Self {
        self.max_duration_s = max_duration_s;
        self
    }

    /// Replace the maximum coarse sample count.
    pub fn with_max_scan_samples(mut self, max_scan_samples: u32) -> Self {
        self.max_scan_samples = max_scan_samples;
        self
    }
}

/// Error returned by [`estimate_decay`].
#[derive(Debug, Clone, thiserror::Error)]
pub enum DecayError {
    /// Propagation or altitude evaluation failed.
    #[error("propagation failed: {0}")]
    Propagation(PropagationError),
    /// A config field was out of domain.
    #[error("invalid decay config field: {0}")]
    InvalidConfig(&'static str),
    /// The orbit did not decay within the requested horizon.
    #[error("no decay within horizon {horizon_s} s")]
    NoDecayWithinHorizon {
        /// Configured maximum elapsed scan time, in seconds, reached before a crossing was found.
        horizon_s: f64,
    },
    /// The coarse scan hit its sample budget before the horizon.
    #[error("scan budget exhausted after {samples} samples and {scanned_s} s")]
    ScanBudgetExhausted {
        /// Elapsed time covered by completed coarse samples, in seconds, when the budget check failed.
        scanned_s: f64,
        /// Number of completed coarse propagation samples reported at the budget check.
        samples: u32,
    },
}

/// Estimate time to reentry using drag-perturbed numerical propagation.
pub fn estimate_decay(
    initial: CartesianState,
    config: &DecayConfig,
) -> Result<DecayEstimate, DecayError> {
    estimate_decay_driver(initial, config, None)
}

/// Estimate time to reentry using per-epoch space weather from `source`.
///
/// `config.drag` supplies the ballistic coefficient and cutoff altitude; its
/// fixed [`crate::astro::forces::SpaceWeather`] value is not consulted.
pub fn estimate_decay_with_source(
    initial: CartesianState,
    config: &DecayConfig,
    source: &SpaceWeatherSource,
) -> Result<DecayEstimate, DecayError> {
    estimate_decay_driver(initial, config, Some(source))
}

fn estimate_decay_driver(
    initial: CartesianState,
    config: &DecayConfig,
    source: Option<&SpaceWeatherSource>,
) -> Result<DecayEstimate, DecayError> {
    validate_config(config)?;

    let initial_altitude = geodetic_altitude_km(&initial).map_err(DecayError::Propagation)?;
    if initial_altitude <= config.reentry_altitude_km {
        return Ok(DecayEstimate {
            time_to_decay_s: 0.0,
            reentry_state: initial,
            reentry_altitude_km: initial_altitude,
        });
    }

    let initial_epoch = initial.epoch_tdb_seconds;
    let mut elapsed_s = 0.0;
    let mut samples = 0_u32;
    let mut current = initial;

    loop {
        if elapsed_s >= config.max_duration_s {
            return Err(DecayError::NoDecayWithinHorizon {
                horizon_s: config.max_duration_s,
            });
        }
        if samples >= config.max_scan_samples {
            return Err(DecayError::ScanBudgetExhausted {
                scanned_s: elapsed_s,
                samples,
            });
        }

        let next_elapsed_s = (elapsed_s + config.scan_step_s).min(config.max_duration_s);
        let next_state =
            propagate_segment(current, initial_epoch + next_elapsed_s, config, source)?;
        samples += 1;
        let next_altitude = geodetic_altitude_km(&next_state).map_err(DecayError::Propagation)?;

        if next_altitude <= config.reentry_altitude_km {
            return refine_reentry(
                initial_epoch,
                current,
                elapsed_s,
                next_elapsed_s,
                config,
                source,
            );
        }

        elapsed_s = next_elapsed_s;
        current = next_state;
    }
}

fn validate_config(config: &DecayConfig) -> Result<(), DecayError> {
    validate_positive("scan_step_s", config.scan_step_s)?;
    validate_positive("crossing_tolerance_s", config.crossing_tolerance_s)?;
    validate_positive("max_duration_s", config.max_duration_s)?;
    if config.max_scan_samples == 0 {
        return Err(DecayError::InvalidConfig("max_scan_samples"));
    }
    if !config.reentry_altitude_km.is_finite() {
        return Err(DecayError::InvalidConfig("reentry_altitude_km"));
    }
    if !(0.0..=MAX_ALTITUDE_KM).contains(&config.reentry_altitude_km) {
        return Err(DecayError::InvalidConfig("reentry_altitude_km"));
    }
    if config.reentry_altitude_km < config.drag.cutoff_altitude_km() {
        return Err(DecayError::InvalidConfig("reentry_altitude_km"));
    }
    Ok(())
}

fn validate_positive(field: &'static str, value: f64) -> Result<(), DecayError> {
    if !value.is_finite() || value <= 0.0 {
        return Err(DecayError::InvalidConfig(field));
    }
    Ok(())
}

fn refine_reentry(
    initial_epoch: f64,
    low_state: CartesianState,
    low_elapsed_s: f64,
    high_elapsed_s: f64,
    config: &DecayConfig,
    source: Option<&SpaceWeatherSource>,
) -> Result<DecayEstimate, DecayError> {
    let threshold = config.reentry_altitude_km;
    let root_elapsed_s = try_bisect_crossing_until(
        low_elapsed_s,
        high_elapsed_s,
        |elapsed_s| {
            let state = propagate_segment(low_state, initial_epoch + elapsed_s, config, source)?;
            let altitude = geodetic_altitude_km(&state).map_err(DecayError::Propagation)?;
            Ok(altitude - threshold)
        },
        |lo, hi| (lo + hi) * 0.5,
        |lo, hi| (hi - lo).abs() <= config.crossing_tolerance_s,
    )
    .map_err(map_root_error)?;

    let reentry_state =
        propagate_segment(low_state, initial_epoch + root_elapsed_s, config, source)?;
    let reentry_altitude_km =
        geodetic_altitude_km(&reentry_state).map_err(DecayError::Propagation)?;
    Ok(DecayEstimate {
        time_to_decay_s: reentry_state.epoch_tdb_seconds - initial_epoch,
        reentry_state,
        reentry_altitude_km,
    })
}

fn map_root_error(error: RootError<DecayError>) -> DecayError {
    match error {
        RootError::Predicate(error) => error,
        RootError::InvalidInput { field, reason } => DecayError::Propagation(
            PropagationError::NumericalFailure(format!("drag decay bisection {field} {reason}")),
        ),
    }
}

fn propagate_segment(
    initial: CartesianState,
    t_end_tdb_seconds: f64,
    config: &DecayConfig,
    source: Option<&SpaceWeatherSource>,
) -> Result<CartesianState, DecayError> {
    let propagator = StatePropagator {
        initial,
        force_model: force_model_kind(config),
        integrator: config.integrator,
        options: config.options,
        drag: Some(config.drag),
        space_weather: source.cloned(),
    };
    Ok(propagator
        .propagate_to(t_end_tdb_seconds)
        .map_err(DecayError::Propagation)?
        .final_state)
}

fn force_model_kind(config: &DecayConfig) -> ForceModelKind {
    let mu_km3_s2 = config.mu_km3_s2.unwrap_or(MU_EARTH);
    match config.force_model {
        PropagationForceModel::TwoBody => ForceModelKind::TwoBody { mu_km3_s2 },
        PropagationForceModel::TwoBodyJ2 => ForceModelKind::TwoBodyJ2 {
            mu_km3_s2,
            re_km: RE_EARTH,
            j2: J2_EARTH,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::constants::RE_EARTH;
    use crate::astro::forces::SpaceWeather;
    use crate::astro::frames::transforms::{
        geodetic_to_itrs, greenwich_mean_sidereal_time_radians_from_j2000_seconds,
    };

    const TEST_EPOCH_S: f64 = 646_315_200.0;

    fn state_from_geodetic_alt(epoch: f64, altitude_km: f64) -> CartesianState {
        let (x_ecef, y_ecef, z_ecef) =
            geodetic_to_itrs(0.0, 0.0, altitude_km).expect("valid geodetic");
        let theta =
            greenwich_mean_sidereal_time_radians_from_j2000_seconds(epoch).expect("valid gmst");
        let c = libm::cos(theta);
        let s = libm::sin(theta);
        let x_eci = c * x_ecef - s * y_ecef;
        let y_eci = s * x_ecef + c * y_ecef;
        let r = RE_EARTH + altitude_km;
        let v = (MU_EARTH / r).sqrt();
        CartesianState::new(
            epoch,
            [x_eci, y_eci, z_ecef],
            [-v * y_eci / r, v * x_eci / r, 0.0],
        )
    }

    fn decay_drag() -> DragParameters {
        DragParameters::from_bc_factor_m2_kg(
            0.8,
            SpaceWeather::default(),
            DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
        )
        .expect("valid drag")
    }

    fn decay_options() -> IntegratorOptions {
        IntegratorOptions {
            abs_tol: 1.0e-8,
            rel_tol: 1.0e-10,
            initial_step: 5.0,
            min_step: 1.0e-6,
            max_step: 30.0,
            max_steps: 200_000,
            dense_output: false,
        }
    }

    fn base_config() -> DecayConfig {
        DecayConfig::new(decay_drag())
            .with_options(decay_options())
            .with_scan_step_s(60.0)
            .with_crossing_tolerance_s(2.0)
            .with_max_duration_s(50_000.0)
            .with_max_scan_samples(2_000)
    }

    #[test]
    fn estimate_decay_brackets_and_refines_reentry() {
        let initial = state_from_geodetic_alt(TEST_EPOCH_S, 125.0);
        let config = base_config();
        let estimate = estimate_decay(initial, &config).expect("decays within horizon");

        assert!(estimate.time_to_decay_s > 0.0);
        assert_eq!(
            estimate.time_to_decay_s.to_bits(),
            (estimate.reentry_state.epoch_tdb_seconds - initial.epoch_tdb_seconds).to_bits()
        );
        let before = propagate_segment(
            initial,
            initial.epoch_tdb_seconds + estimate.time_to_decay_s - config.crossing_tolerance_s,
            &config,
            None,
        )
        .expect("before crossing");
        let after = propagate_segment(
            initial,
            initial.epoch_tdb_seconds + estimate.time_to_decay_s + config.crossing_tolerance_s,
            &config,
            None,
        )
        .expect("after crossing");
        let before_alt = geodetic_altitude_km(&before).expect("before altitude");
        let after_alt = geodetic_altitude_km(&after).expect("after altitude");
        let altitude_window_km = (before_alt - after_alt).abs().max(1.0e-6);
        assert!(
            (estimate.reentry_altitude_km - config.reentry_altitude_km).abs() <= altitude_window_km
        );

        let high = state_from_geodetic_alt(TEST_EPOCH_S, 700.0);
        let no_decay = base_config()
            .with_max_duration_s(120.0)
            .with_max_scan_samples(10);
        match estimate_decay(high, &no_decay).expect_err("short horizon should stop") {
            DecayError::NoDecayWithinHorizon { horizon_s } => assert_eq!(horizon_s, 120.0),
            other => panic!("expected horizon stop, got {other:?}"),
        }

        let budget = base_config()
            .with_max_duration_s(10_000.0)
            .with_max_scan_samples(2);
        match estimate_decay(high, &budget).expect_err("sample budget should stop") {
            DecayError::ScanBudgetExhausted { scanned_s, samples } => {
                assert_eq!(scanned_s, 120.0);
                assert_eq!(samples, 2);
            }
            other => panic!("expected scan budget stop, got {other:?}"),
        }
    }

    #[test]
    fn estimate_decay_zero_when_initial_below_threshold() {
        let initial = state_from_geodetic_alt(TEST_EPOCH_S, 90.0);
        let estimate = estimate_decay(initial, &base_config()).expect("initial is below threshold");

        assert_eq!(estimate.time_to_decay_s, 0.0);
        assert_eq!(estimate.reentry_state, initial);
        assert!(estimate.reentry_altitude_km <= DecayConfig::DEFAULT_REENTRY_ALTITUDE_KM);
    }
}
