//! High-level numerical-propagation driver.
//!
//! A single callable entry that turns a high-level configuration (force-model
//! choice, integrator, integrator options, initial state, and an output epoch
//! grid) into a sampled trajectory. It composes the existing
//! [`StatePropagator`] / [`ForceModelKind`] / [`IntegratorOptions`] primitives
//! with the canonical Earth constants from [`crate::astro::constants`]; it adds
//! no integration math of its own. A language binding reduces to: marshal user
//! options into a [`PropagationConfig`], call [`propagate_states`], then marshal
//! the returned states back out, with no force-model or integrator policy left
//! in the binding.
//!
//! The defaults come from [`IntegratorOptions::default`] and
//! [`crate::astro::constants::MU_EARTH`], so the driver's defaults are exactly
//! the engine's defaults rather than independently chosen numbers.

use crate::astro::constants::{J2_EARTH, MU_EARTH, RE_EARTH};
use crate::astro::error::PropagationError;
use crate::astro::propagator::api::IntegratorOptions;
use crate::astro::propagator::numerical::{ForceModelKind, IntegratorKind, StatePropagator};
use crate::astro::state::CartesianState;

/// High-level force-model choice for [`PropagationConfig`].
///
/// Mirrors the selector every language binding exposes: a point-mass two-body
/// central force, or two-body plus the Earth J2 oblateness perturbation. The
/// concrete [`ForceModelKind`] is composed by the driver from this choice and
/// the configured gravitational parameter, filling the canonical Earth
/// reference radius and J2 coefficient from [`crate::astro::constants`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PropagationForceModel {
    /// Point-mass two-body gravity.
    #[default]
    TwoBody,
    /// Two-body gravity plus the Earth J2 oblateness perturbation.
    TwoBodyJ2,
}

/// High-level configuration for the numerical-propagation driver.
///
/// Construct with [`PropagationConfig::new`] and override fields; the defaults
/// match every binding's defaults: two-body gravity, the adaptive DP54
/// integrator, the canonical [`MU_EARTH`] gravitational parameter, and the
/// engine-default [`IntegratorOptions`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PropagationConfig {
    /// Initial ECI Cartesian state; its epoch is the propagation start epoch.
    pub initial: CartesianState,
    /// Force-model choice.
    pub force_model: PropagationForceModel,
    /// Gravitational-parameter override, km^3/s^2. `None` uses the canonical
    /// [`MU_EARTH`].
    pub mu_km3_s2: Option<f64>,
    /// Integrator choice.
    pub integrator: IntegratorKind,
    /// Step-size / tolerance controls forwarded to the integrator.
    pub options: IntegratorOptions,
}

impl PropagationConfig {
    /// Build a config from a raw epoch (TDB seconds), ECI position (km), and ECI
    /// velocity (km/s) with the binding defaults: two-body gravity, the DP54
    /// integrator, the canonical [`MU_EARTH`] gravitational parameter, and
    /// [`IntegratorOptions::default`].
    pub fn new(epoch_tdb_seconds: f64, position_km: [f64; 3], velocity_km_s: [f64; 3]) -> Self {
        Self {
            initial: CartesianState::new(epoch_tdb_seconds, position_km, velocity_km_s),
            force_model: PropagationForceModel::TwoBody,
            mu_km3_s2: None,
            integrator: IntegratorKind::Dp54,
            options: IntegratorOptions::default(),
        }
    }

    /// The gravitational parameter the driver will use: the configured override,
    /// or the canonical [`MU_EARTH`] when none is set.
    pub fn gravitational_parameter(&self) -> f64 {
        self.mu_km3_s2.unwrap_or(MU_EARTH)
    }

    /// Compose the concrete [`ForceModelKind`] from the high-level choice and the
    /// effective gravitational parameter, filling [`RE_EARTH`] / [`J2_EARTH`] for
    /// the J2 variant. This is the force-model composition policy the bindings
    /// each duplicated.
    pub fn force_model_kind(&self) -> ForceModelKind {
        let mu_km3_s2 = self.gravitational_parameter();
        match self.force_model {
            PropagationForceModel::TwoBody => ForceModelKind::TwoBody { mu_km3_s2 },
            PropagationForceModel::TwoBodyJ2 => ForceModelKind::TwoBodyJ2 {
                mu_km3_s2,
                re_km: RE_EARTH,
                j2: J2_EARTH,
            },
        }
    }

    /// Assemble the [`StatePropagator`] this config describes. Equivalent to the
    /// per-binding hand assembly, so a downstream `ephemeris` call is bit-for-bit
    /// identical to the binding's own.
    pub fn to_propagator(&self) -> StatePropagator {
        StatePropagator {
            initial: self.initial,
            force_model: self.force_model_kind(),
            integrator: self.integrator,
            options: self.options,
        }
    }
}

/// Run the numerical propagator described by `config`, sampling the trajectory
/// at `epochs_tdb_seconds` (absolute TDB seconds, monotonic in the propagation
/// direction).
///
/// Composes [`PropagationConfig::to_propagator`] with the existing
/// [`StatePropagator::ephemeris`] sampler; the returned states are exactly what
/// the engine produces, so the driver is bit-for-bit identical to assembling the
/// propagator and calling `ephemeris` by hand.
pub fn propagate_states(
    config: &PropagationConfig,
    epochs_tdb_seconds: &[f64],
) -> Result<Vec<CartesianState>, PropagationError> {
    config.to_propagator().ephemeris(epochs_tdb_seconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn circular_state() -> (f64, [f64; 3], [f64; 3]) {
        let r: f64 = 7000.0;
        let v = (MU_EARTH / r).sqrt();
        (0.0, [r, 0.0, 0.0], [0.0, v, 0.0])
    }

    fn assert_states_bit_for_bit(left: &[CartesianState], right: &[CartesianState]) {
        assert_eq!(left.len(), right.len(), "state-count mismatch");
        for (a, b) in left.iter().zip(right.iter()) {
            assert_eq!(a.epoch_tdb_seconds.to_bits(), b.epoch_tdb_seconds.to_bits());
            for axis in 0..3 {
                assert_eq!(
                    a.position_array()[axis].to_bits(),
                    b.position_array()[axis].to_bits(),
                    "position axis {axis}"
                );
                assert_eq!(
                    a.velocity_array()[axis].to_bits(),
                    b.velocity_array()[axis].to_bits(),
                    "velocity axis {axis}"
                );
            }
        }
    }

    #[test]
    fn config_defaults_come_from_core_constants() {
        let (epoch, pos, vel) = circular_state();
        let config = PropagationConfig::new(epoch, pos, vel);

        assert_eq!(config.force_model, PropagationForceModel::TwoBody);
        assert_eq!(config.mu_km3_s2, None);
        assert_eq!(
            config.gravitational_parameter().to_bits(),
            MU_EARTH.to_bits()
        );
        assert_eq!(config.integrator, IntegratorKind::Dp54);
        assert_eq!(config.options, IntegratorOptions::default());
        assert_eq!(
            config.force_model_kind(),
            ForceModelKind::TwoBody {
                mu_km3_s2: MU_EARTH
            }
        );
    }

    #[test]
    fn force_model_kind_composes_j2_from_canonical_constants() {
        let (epoch, pos, vel) = circular_state();
        let config = PropagationConfig {
            force_model: PropagationForceModel::TwoBodyJ2,
            ..PropagationConfig::new(epoch, pos, vel)
        };

        assert_eq!(
            config.force_model_kind(),
            ForceModelKind::TwoBodyJ2 {
                mu_km3_s2: MU_EARTH,
                re_km: RE_EARTH,
                j2: J2_EARTH,
            }
        );
    }

    #[test]
    fn driver_two_body_default_matches_manual_composition_bit_for_bit() {
        // The driver path: high-level config -> propagate_states.
        let (epoch, pos, vel) = circular_state();
        let config = PropagationConfig::new(epoch, pos, vel);
        let epochs = [0.0, 600.0, 1800.0, 3600.0];
        let via_driver = propagate_states(&config, &epochs).expect("driver propagation");

        // The hand-assembled path the Python / WASM / C bindings each spell out:
        // default mu = MU_EARTH, two-body force, DP54, default options.
        let via_manual = StatePropagator {
            initial: CartesianState::new(epoch, pos, vel),
            force_model: ForceModelKind::TwoBody {
                mu_km3_s2: MU_EARTH,
            },
            integrator: IntegratorKind::Dp54,
            options: IntegratorOptions {
                abs_tol: 1.0e-9,
                rel_tol: 1.0e-12,
                initial_step: 60.0,
                min_step: 1.0e-6,
                max_step: 3600.0,
                max_steps: 1_000_000,
                dense_output: false,
            },
        }
        .ephemeris(&epochs)
        .expect("manual propagation");

        assert_states_bit_for_bit(&via_driver, &via_manual);
    }

    #[test]
    fn driver_two_body_j2_custom_mu_rk4_matches_manual_composition_bit_for_bit() {
        let (epoch, pos, vel) = circular_state();
        let mu = 398_600.5;
        let options = IntegratorOptions {
            abs_tol: 1.0e-11,
            rel_tol: 1.0e-13,
            initial_step: 30.0,
            min_step: 1.0e-5,
            max_step: 120.0,
            max_steps: 500_000,
            dense_output: false,
        };
        let config = PropagationConfig {
            force_model: PropagationForceModel::TwoBodyJ2,
            mu_km3_s2: Some(mu),
            integrator: IntegratorKind::Rk4,
            options,
            ..PropagationConfig::new(epoch, pos, vel)
        };
        let epochs = [0.0, 300.0, 900.0];
        let via_driver = propagate_states(&config, &epochs).expect("driver propagation");

        let via_manual = StatePropagator {
            initial: CartesianState::new(epoch, pos, vel),
            force_model: ForceModelKind::TwoBodyJ2 {
                mu_km3_s2: mu,
                re_km: RE_EARTH,
                j2: J2_EARTH,
            },
            integrator: IntegratorKind::Rk4,
            options,
        }
        .ephemeris(&epochs)
        .expect("manual propagation");

        assert_states_bit_for_bit(&via_driver, &via_manual);
    }

    #[test]
    fn driver_surfaces_the_integrator_error_unchanged() {
        // A non-positive initial step is rejected by the integrator itself; the
        // driver forwards that error verbatim rather than masking it.
        let (epoch, pos, vel) = circular_state();
        let config = PropagationConfig {
            options: IntegratorOptions {
                initial_step: 0.0,
                ..IntegratorOptions::default()
            },
            ..PropagationConfig::new(epoch, pos, vel)
        };

        let err = propagate_states(&config, &[0.0, 60.0]).expect_err("non-positive step rejected");
        match err {
            PropagationError::InvalidInput(message) => {
                assert!(message.contains("initial_step"), "{message}");
                assert!(message.contains("not positive"), "{message}");
            }
            other => panic!("expected invalid-input error, got {other:?}"),
        }
    }
}
