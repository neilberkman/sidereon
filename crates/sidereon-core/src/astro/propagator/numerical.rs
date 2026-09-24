//! High-level numerical state-vector propagation entry point.
//!
//! Builds a propagatable object from a raw epoch plus an ECI Cartesian state
//! (position + velocity) and propagates it forward (or backward) with the
//! existing integrators ([`RK4`], [`DP54`]) and the existing force models
//! (two-body, two-body + J2, and additive perturbation sets). This is a thin
//! orchestration layer over
//! [`crate::astro::integrators`] and [`crate::astro::forces`]: it constructs the
//! force model, wraps it in [`OrbitalDynamics`], and drives the chosen
//! integrator. It adds no integration math of its own, so a single-shot
//! [`StatePropagator::propagate_to`] is bit-for-bit identical to assembling the
//! force model, dynamics, integrator, and options by hand. State-transition
//! matrices are formed by finite-differencing this same entry point, so they
//! reflect the selected force model and integrator without duplicating dynamics.

use crate::astro::constants::{J2_EARTH, MU_EARTH, RE_EARTH};
use crate::astro::covariance::{Covariance6, Covariance6Error};
use crate::astro::error::PropagationError;
use crate::astro::forces::{
    CompositeForceModel, DragParameters, EarthRadiationPressure, ForceModel, J2Gravity,
    SchwarzschildRelativity, SolarRadiationPressure, SolidEarthPoleTideGravity,
    SolidEarthTideGravity, SourcedDragForce, SpaceWeatherSource, SphericalHarmonicGravityConfig,
    ThirdBodyGravity, TideSystem, TwoBodyGravity, ZonalGravity,
};
use crate::astro::integrators::{Integrator, DP54, RK4};
use crate::astro::propagator::api::{IntegratorOptions, PropagationContext};
use crate::astro::propagator::covariance::{
    CovarianceFrame, CovariancePropagationOptions, LabeledCovariance6,
};
use crate::astro::propagator::dynamics::OrbitalDynamics;
use crate::astro::propagator::result::PropagationResult;
use crate::astro::state::CartesianState;

/// Row-major 6x6 state-transition matrix for `[r_x, r_y, r_z, v_x, v_y, v_z]`.
///
/// Entry `[i][j]` is the finite-difference derivative of final-state component
/// `i` with respect to initial-state component `j`.
pub type StateTransitionMatrix = [[f64; 6]; 6];

const STM_RELATIVE_PERTURBATION: f64 = 1.0e-6;
const STM_MIN_PERTURBATION: f64 = 1.0e-6;

/// Which numerical integrator drives the propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntegratorKind {
    /// Fixed-step classical Runge-Kutta 4 (step taken from
    /// [`IntegratorOptions::initial_step`]; tolerances are ignored).
    Rk4,
    /// Adaptive Dormand-Prince 5(4) with PI step control (honors the absolute
    /// and relative tolerances).
    Dp54,
}

/// Which force model supplies the acceleration during propagation.
///
/// The legacy variants carry their own physical parameters so a caller can
/// propagate a non-Earth central body by supplying a different gravitational
/// parameter. The [`Self::two_body`] / [`Self::two_body_j2`] constructors fill
/// in the canonical Earth values from [`crate::astro::constants`]. The
/// [`Self::Composite`] variant adds optional force components without changing
/// those legacy branches.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ForceModelKind {
    /// Pure two-body (Keplerian) gravity.
    TwoBody {
        /// Gravitational parameter, km^3/s^2.
        mu_km3_s2: f64,
    },
    /// Two-body gravity plus the J2 oblateness perturbation.
    TwoBodyJ2 {
        /// Gravitational parameter, km^3/s^2.
        mu_km3_s2: f64,
        /// Reference equatorial radius, km.
        re_km: f64,
        /// J2 zonal harmonic coefficient (dimensionless).
        j2: f64,
    },
    /// Additive composition of central gravity and optional perturbations.
    Composite {
        /// Force components to add.
        components: ForceModelComponents,
    },
}

/// Additive force components for [`ForceModelKind::Composite`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForceModelComponents {
    /// Optional central two-body gravitational parameter, km^3/s^2.
    pub two_body_mu_km3_s2: Option<f64>,
    /// Optional zonal gravity perturbation.
    pub zonal: Option<ZonalGravity>,
    /// Optional embedded EGM96 spherical-harmonic perturbation.
    pub spherical_harmonic: Option<SphericalHarmonicGravityConfig>,
    /// Optional Sun and Moon third-body perturbation.
    pub third_body: Option<ThirdBodyGravity>,
    /// Optional solid Earth tide geopotential perturbation. Its
    /// [`SolidEarthTideGravity::tide_system`] must be the tide system of the
    /// zonal or spherical-harmonic field when one is selected.
    pub solid_earth_tide: Option<SolidEarthTideGravity>,
    /// Optional solid Earth pole tide geopotential perturbation.
    pub solid_earth_pole_tide: Option<SolidEarthPoleTideGravity>,
    /// Optional cannonball solar radiation pressure perturbation.
    pub solar_radiation_pressure: Option<SolarRadiationPressure>,
    /// Optional cannonball Earth albedo and infrared radiation pressure perturbation.
    pub earth_radiation_pressure: Option<EarthRadiationPressure>,
    /// Optional geocentric Schwarzschild relativistic correction.
    pub relativity: Option<SchwarzschildRelativity>,
}

impl Default for ForceModelComponents {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl ForceModelComponents {
    /// No force components.
    pub const EMPTY: Self = Self {
        two_body_mu_km3_s2: None,
        zonal: None,
        spherical_harmonic: None,
        third_body: None,
        solid_earth_tide: None,
        solid_earth_pole_tide: None,
        solar_radiation_pressure: None,
        earth_radiation_pressure: None,
        relativity: None,
    };

    /// Canonical Earth two-body central gravity.
    pub fn earth_two_body() -> Self {
        Self {
            two_body_mu_km3_s2: Some(MU_EARTH),
            ..Self::EMPTY
        }
    }

    /// Canonical Earth Phase A force set, with optional spacecraft SRP parameters.
    pub fn earth_phase_a(solar_radiation_pressure: Option<SolarRadiationPressure>) -> Self {
        Self {
            two_body_mu_km3_s2: Some(MU_EARTH),
            zonal: Some(ZonalGravity::earth_j2_through_j6()),
            spherical_harmonic: None,
            third_body: Some(ThirdBodyGravity::default()),
            solid_earth_tide: None,
            solid_earth_pole_tide: None,
            solar_radiation_pressure,
            earth_radiation_pressure: None,
            relativity: Some(SchwarzschildRelativity::default()),
        }
    }

    /// Canonical Earth Phase B force set with embedded EGM96 harmonics.
    pub fn earth_phase_b(
        max_degree: u16,
        max_order: u16,
        solar_radiation_pressure: Option<SolarRadiationPressure>,
    ) -> Result<Self, PropagationError> {
        Ok(Self {
            two_body_mu_km3_s2: Some(MU_EARTH),
            zonal: None,
            spherical_harmonic: Some(SphericalHarmonicGravityConfig::earth(
                max_degree, max_order,
            )?),
            third_body: Some(ThirdBodyGravity::default()),
            solid_earth_tide: None,
            solid_earth_pole_tide: None,
            solar_radiation_pressure,
            earth_radiation_pressure: None,
            relativity: Some(SchwarzschildRelativity::default()),
        })
    }

    /// Set or replace central two-body gravity.
    pub fn with_two_body_mu(mut self, mu_km3_s2: f64) -> Self {
        self.two_body_mu_km3_s2 = Some(mu_km3_s2);
        self
    }

    /// Set or replace zonal gravity.
    pub fn with_zonal(mut self, zonal: ZonalGravity) -> Self {
        self.zonal = Some(zonal);
        self
    }

    /// Set or replace embedded EGM96 spherical-harmonic gravity.
    pub fn with_spherical_harmonic(mut self, gravity: SphericalHarmonicGravityConfig) -> Self {
        self.spherical_harmonic = Some(gravity);
        self
    }

    /// Set or replace third-body gravity.
    pub fn with_third_body(mut self, third_body: ThirdBodyGravity) -> Self {
        self.third_body = Some(third_body);
        self
    }

    /// Set or replace solid Earth tide geopotential perturbation.
    pub fn with_solid_earth_tide(mut self, tide: SolidEarthTideGravity) -> Self {
        self.solid_earth_tide = Some(tide);
        self
    }

    /// Set or replace solid Earth pole tide geopotential perturbation.
    pub fn with_solid_earth_pole_tide(mut self, tide: SolidEarthPoleTideGravity) -> Self {
        self.solid_earth_pole_tide = Some(tide);
        self
    }

    /// Set or replace solar radiation pressure.
    pub fn with_solar_radiation_pressure(mut self, srp: SolarRadiationPressure) -> Self {
        self.solar_radiation_pressure = Some(srp);
        self
    }

    /// Set or replace Earth albedo and infrared radiation pressure.
    pub fn with_earth_radiation_pressure(mut self, pressure: EarthRadiationPressure) -> Self {
        self.earth_radiation_pressure = Some(pressure);
        self
    }

    /// Set or replace relativity.
    pub fn with_relativity(mut self, relativity: SchwarzschildRelativity) -> Self {
        self.relativity = Some(relativity);
        self
    }
}

impl ForceModelKind {
    /// Earth two-body gravity using the canonical [`MU_EARTH`].
    pub fn two_body() -> Self {
        Self::TwoBody {
            mu_km3_s2: MU_EARTH,
        }
    }

    /// Earth two-body + J2 using the canonical [`MU_EARTH`], [`RE_EARTH`], and
    /// [`J2_EARTH`].
    pub fn two_body_j2() -> Self {
        Self::TwoBodyJ2 {
            mu_km3_s2: MU_EARTH,
            re_km: RE_EARTH,
            j2: J2_EARTH,
        }
    }

    /// Build an additive force model from components.
    pub fn composite(components: ForceModelComponents) -> Self {
        Self::Composite { components }
    }

    /// Earth two-body plus Phase A perturbations.
    pub fn earth_phase_a(solar_radiation_pressure: Option<SolarRadiationPressure>) -> Self {
        Self::Composite {
            components: ForceModelComponents::earth_phase_a(solar_radiation_pressure),
        }
    }

    /// Earth two-body plus Phase B perturbations.
    pub fn earth_phase_b(
        max_degree: u16,
        max_order: u16,
        solar_radiation_pressure: Option<SolarRadiationPressure>,
    ) -> Result<Self, PropagationError> {
        Ok(Self::Composite {
            components: ForceModelComponents::earth_phase_b(
                max_degree,
                max_order,
                solar_radiation_pressure,
            )?,
        })
    }

    /// Build the boxed [`ForceModel`] this variant describes. Reuses the force
    /// model implementations; no acceleration math is duplicated here.
    fn build(self) -> Result<Box<dyn ForceModel>, PropagationError> {
        match self {
            ForceModelKind::TwoBody { mu_km3_s2 } => Ok(Box::new(TwoBodyGravity { mu: mu_km3_s2 })),
            ForceModelKind::TwoBodyJ2 {
                mu_km3_s2,
                re_km,
                j2,
            } => {
                let mut composite = CompositeForceModel::new();
                composite.add(Box::new(TwoBodyGravity { mu: mu_km3_s2 }));
                composite.add(Box::new(J2Gravity {
                    mu: mu_km3_s2,
                    re: re_km,
                    j2,
                }));
                Ok(Box::new(composite))
            }
            ForceModelKind::Composite { components } => {
                if components.zonal.is_some() && components.spherical_harmonic.is_some() {
                    return Err(PropagationError::InvalidInput(
                        "zonal and spherical harmonic gravity cannot both be selected".to_string(),
                    ));
                }
                // The tide force's Step 3 removes the permanent tide the field's
                // C20 holds; told another tide system, it would count that part
                // twice or not at all.
                // A zonal field without J2 has no C20, so it holds no permanent
                // tide whatever its coefficients' tide system says.
                let field_tide_system = components
                    .zonal
                    .map(|zonal| {
                        if zonal.degrees.j2 {
                            zonal.coefficients.tide_system
                        } else {
                            TideSystem::TideFree
                        }
                    })
                    .or(components
                        .spherical_harmonic
                        .map(|gravity| gravity.tide_system()));
                if let (Some(tide), Some(field)) = (components.solid_earth_tide, field_tide_system)
                {
                    if tide.tide_system != field {
                        return Err(PropagationError::InvalidInput(format!(
                            "solid Earth tide is set for a {:?} geopotential but the selected field is {:?}",
                            tide.tide_system, field
                        )));
                    }
                }
                let mut composite = CompositeForceModel::new();
                if let Some(mu_km3_s2) = components.two_body_mu_km3_s2 {
                    composite.add(Box::new(TwoBodyGravity { mu: mu_km3_s2 }));
                }
                if let Some(zonal) = components.zonal {
                    composite.add(Box::new(zonal));
                }
                if let Some(spherical_harmonic) = components.spherical_harmonic {
                    composite.add(Box::new(spherical_harmonic.build()?));
                }
                if let Some(third_body) = components.third_body {
                    composite.add(Box::new(third_body));
                }
                if let Some(tide) = components.solid_earth_tide {
                    composite.add(Box::new(tide));
                }
                if let Some(tide) = components.solid_earth_pole_tide {
                    composite.add(Box::new(tide));
                }
                if let Some(srp) = components.solar_radiation_pressure {
                    composite.add(Box::new(srp));
                }
                if let Some(pressure) = components.earth_radiation_pressure {
                    composite.add(Box::new(pressure));
                }
                if let Some(relativity) = components.relativity {
                    composite.add(Box::new(relativity));
                }
                Ok(Box::new(composite))
            }
        }
    }
}

/// A propagatable object built from a raw initial state.
///
/// Construct it with [`StatePropagator::new`] (or by filling the public fields),
/// then call [`StatePropagator::propagate_to`] for a single end epoch or
/// [`StatePropagator::ephemeris`] to sample the trajectory at a sequence of
/// epochs.
pub struct StatePropagator {
    /// Initial ECI Cartesian state (its `epoch_tdb_seconds` is the start epoch).
    pub initial: CartesianState,
    /// Force model used to compute the acceleration.
    pub force_model: ForceModelKind,
    /// Integrator that advances the state.
    pub integrator: IntegratorKind,
    /// Step-size / tolerance controls passed to the integrator.
    pub options: IntegratorOptions,
    /// Optional atmospheric drag perturbation layered on the gravity model.
    pub drag: Option<DragParameters>,
    /// Optional per-epoch space-weather source for atmospheric drag.
    pub space_weather: Option<SpaceWeatherSource>,
}

impl StatePropagator {
    /// Build a propagator from a raw epoch (TDB seconds), ECI position (km), and
    /// ECI velocity (km/s), with the given force model and integrator and the
    /// default [`IntegratorOptions`].
    pub fn new(
        epoch_tdb_seconds: f64,
        position_km: [f64; 3],
        velocity_km_s: [f64; 3],
        force_model: ForceModelKind,
        integrator: IntegratorKind,
    ) -> Self {
        Self {
            initial: CartesianState::new(epoch_tdb_seconds, position_km, velocity_km_s),
            force_model,
            integrator,
            options: IntegratorOptions::default(),
            drag: None,
            space_weather: None,
        }
    }

    /// Replace the integrator options (builder-style).
    pub fn with_options(mut self, options: IntegratorOptions) -> Self {
        self.options = options;
        self
    }

    /// Enable atmospheric drag (builder-style).
    pub fn with_drag(mut self, drag: DragParameters) -> Self {
        self.drag = Some(drag);
        self
    }

    /// Use a per-epoch space-weather source for the drag perturbation.
    pub fn with_space_weather(mut self, source: SpaceWeatherSource) -> Self {
        self.space_weather = Some(source);
        self
    }

    /// Propagate from the initial epoch to `t_end_tdb_seconds` (an absolute TDB
    /// epoch), returning the underlying integrator's full
    /// [`PropagationResult`]. Bit-for-bit identical to building the force model,
    /// [`OrbitalDynamics`], integrator, and options by hand.
    pub fn propagate_to(
        &self,
        t_end_tdb_seconds: f64,
    ) -> Result<PropagationResult, PropagationError> {
        let ctx = PropagationContext::default();
        self.propagate_to_with_context(t_end_tdb_seconds, &ctx)
    }

    /// Propagate to `t_end_tdb_seconds` with an explicit propagation context.
    ///
    /// This is the opt-in path for force models that need shared per-evaluation
    /// services such as a body-fixed frame provider. Passing
    /// [`PropagationContext::default`] is equivalent to [`Self::propagate_to`].
    pub fn propagate_to_with_context(
        &self,
        t_end_tdb_seconds: f64,
        ctx: &PropagationContext,
    ) -> Result<PropagationResult, PropagationError> {
        let force = self.build_force()?;
        let dynamics = OrbitalDynamics {
            force_model: force.as_ref(),
        };
        self.run(self.initial, t_end_tdb_seconds, &dynamics, ctx)
    }

    /// Propagate the initial state and a 6x6 state covariance over a relative
    /// span in seconds.
    ///
    /// This is the single-segment, no-process-noise convenience wrapper over
    /// [`Self::propagate_covariance`].
    pub fn propagate_state_with_covariance(
        &self,
        covariance0: Covariance6,
        span_seconds: f64,
    ) -> Result<(CartesianState, Covariance6), PropagationError> {
        validate_initial_state(self.initial)?;
        crate::validate::finite(span_seconds, "span_seconds").map_err(map_field_error)?;
        let t_end_tdb_seconds = self.initial.epoch_tdb_seconds + span_seconds;
        crate::validate::finite(t_end_tdb_seconds, "t_end_tdb_seconds").map_err(map_field_error)?;

        if span_seconds == 0.0 {
            return Ok((self.initial, covariance0));
        }

        let ephemeris = self.propagate_covariance(
            LabeledCovariance6 {
                covariance: covariance0,
                frame: CovarianceFrame::Inertial,
            },
            &[t_end_tdb_seconds],
            &CovariancePropagationOptions::default(),
        )?;
        let node = ephemeris.nodes()[0];
        Ok((node.state, node.covariance))
    }

    /// Build the finite-difference state-transition matrix over a relative
    /// propagation span in seconds.
    ///
    /// Columns perturb the initial state in `[r_x, r_y, r_z, v_x, v_y, v_z]`
    /// order. Each plus/minus leg is propagated through [`Self::propagate_to`]'s
    /// same force-model and integrator assembly path.
    pub fn state_transition_matrix_for_span(
        &self,
        span_seconds: f64,
    ) -> Result<StateTransitionMatrix, PropagationError> {
        crate::validate::finite(span_seconds, "span_seconds").map_err(map_field_error)?;
        self.state_transition_matrix_to(self.initial.epoch_tdb_seconds + span_seconds)
    }

    /// Build the finite-difference state-transition matrix to an absolute TDB
    /// epoch in seconds.
    ///
    /// The zero-span STM is exactly identity and does not call the propagator.
    /// Non-zero spans propagate twelve perturbed initial states with central
    /// differences and the existing numerical propagator.
    pub fn state_transition_matrix_to(
        &self,
        t_end_tdb_seconds: f64,
    ) -> Result<StateTransitionMatrix, PropagationError> {
        let ctx = PropagationContext::default();
        self.state_transition_matrix_to_with_context(t_end_tdb_seconds, &ctx)
    }

    /// Build the finite-difference state-transition matrix with an explicit
    /// propagation context.
    ///
    /// Passing [`PropagationContext::default`] is equivalent to
    /// [`Self::state_transition_matrix_to`].
    pub fn state_transition_matrix_to_with_context(
        &self,
        t_end_tdb_seconds: f64,
        ctx: &PropagationContext,
    ) -> Result<StateTransitionMatrix, PropagationError> {
        crate::validate::finite(t_end_tdb_seconds, "t_end_tdb_seconds").map_err(map_field_error)?;
        let force = self.build_force()?;
        let dynamics = OrbitalDynamics {
            force_model: force.as_ref(),
        };
        self.state_transition_matrix_between(self.initial, t_end_tdb_seconds, &dynamics, ctx)
    }

    pub(super) fn state_transition_matrix_between(
        &self,
        initial: CartesianState,
        t_end_tdb_seconds: f64,
        dynamics: &OrbitalDynamics,
        ctx: &PropagationContext,
    ) -> Result<StateTransitionMatrix, PropagationError> {
        crate::validate::finite(t_end_tdb_seconds, "t_end_tdb_seconds").map_err(map_field_error)?;
        if t_end_tdb_seconds == initial.epoch_tdb_seconds {
            return Ok(identity_stm());
        }

        let mut stm = [[0.0_f64; 6]; 6];
        let initial_vector = state_vector(&initial);

        for (column, &component) in initial_vector.iter().enumerate() {
            let delta = finite_difference_step(component);
            let plus = perturb_state(initial, column, delta);
            let minus = perturb_state(initial, column, -delta);

            let plus_final = self
                .run(plus, t_end_tdb_seconds, dynamics, ctx)?
                .final_state;
            let minus_final = self
                .run(minus, t_end_tdb_seconds, dynamics, ctx)?
                .final_state;
            let plus_vector = state_vector(&plus_final);
            let minus_vector = state_vector(&minus_final);
            let denom = 2.0 * delta;

            for (row, stm_row) in stm.iter_mut().enumerate() {
                stm_row[column] = (plus_vector[row] - minus_vector[row]) / denom;
            }
        }

        validate_stm(&stm)?;
        Ok(stm)
    }

    /// Sample the trajectory at a sequence of absolute TDB epochs (seconds),
    /// returning the Cartesian state at each. The epochs must be monotonic in
    /// the propagation direction; the satellite is stepped from one requested
    /// epoch to the next (sequential segments), so the cost is linear in the
    /// number of epochs. An epoch equal to the current epoch returns the current
    /// state without re-integrating.
    ///
    /// The force model is built once and reused across every segment.
    pub fn ephemeris(
        &self,
        epochs_tdb_seconds: &[f64],
    ) -> Result<Vec<CartesianState>, PropagationError> {
        let ctx = PropagationContext::default();
        self.ephemeris_with_context(epochs_tdb_seconds, &ctx)
    }

    /// Sample the trajectory with an explicit propagation context.
    ///
    /// Passing [`PropagationContext::default`] is equivalent to
    /// [`Self::ephemeris`].
    pub fn ephemeris_with_context(
        &self,
        epochs_tdb_seconds: &[f64],
        ctx: &PropagationContext,
    ) -> Result<Vec<CartesianState>, PropagationError> {
        validate_initial_state(self.initial)?;
        validate_epoch_finite(self.initial.epoch_tdb_seconds, "initial.epoch_tdb_seconds")?;
        validate_ephemeris_epochs(epochs_tdb_seconds)?;

        let force = self.build_force()?;
        let dynamics = OrbitalDynamics {
            force_model: force.as_ref(),
        };

        let mut states = Vec::with_capacity(epochs_tdb_seconds.len());
        let mut current = self.initial;
        for &t in epochs_tdb_seconds {
            if t != current.epoch_tdb_seconds {
                current = self.run(current, t, &dynamics, ctx)?.final_state;
            }
            states.push(current);
        }
        Ok(states)
    }

    /// Dispatch to the selected integrator. Kept private so the public surface
    /// stays `propagate_to` / `ephemeris`.
    pub(super) fn run(
        &self,
        initial: CartesianState,
        t_end_tdb_seconds: f64,
        dynamics: &OrbitalDynamics,
        ctx: &PropagationContext,
    ) -> Result<PropagationResult, PropagationError> {
        validate_epoch_finite(initial.epoch_tdb_seconds, "initial.epoch_tdb_seconds")?;
        validate_epoch_finite(t_end_tdb_seconds, "t_end_tdb_seconds")?;
        validate_initial_state(initial)?;

        // The integrators record this run's UT1 departure in the result and
        // pass it back to `ctx`.
        match self.integrator {
            IntegratorKind::Rk4 => {
                RK4.propagate(initial, t_end_tdb_seconds, dynamics, ctx, &self.options)
            }
            IntegratorKind::Dp54 => {
                DP54.propagate(initial, t_end_tdb_seconds, dynamics, ctx, &self.options)
            }
        }
    }

    pub(super) fn build_force(&self) -> Result<Box<dyn ForceModel>, PropagationError> {
        let gravity = self.force_model.build()?;
        match (self.drag, self.space_weather.clone()) {
            (Some(drag), Some(source)) => {
                let mut composite = CompositeForceModel::new();
                composite.add(gravity);
                composite.add(Box::new(SourcedDragForce::new(drag, source)));
                Ok(Box::new(composite))
            }
            (Some(drag), None) => {
                let mut composite = CompositeForceModel::new();
                composite.add(gravity);
                composite.add(Box::new(drag.to_force()));
                Ok(Box::new(composite))
            }
            (None, Some(_)) => Err(PropagationError::InvalidInput(
                "space weather source without drag".to_string(),
            )),
            (None, None) => Ok(gravity),
        }
    }
}

fn map_field_error(error: crate::validate::FieldError) -> PropagationError {
    PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
}

pub(super) fn map_covariance6_error(error: Covariance6Error) -> PropagationError {
    let reason = match error {
        Covariance6Error::NonFinite => "not finite",
        Covariance6Error::Asymmetric => "not symmetric",
        Covariance6Error::NotPositiveSemidefinite => "not positive semidefinite",
        Covariance6Error::NotFactorizable => "not factorizable",
        Covariance6Error::InvalidInterpolationParameter => "invalid interpolation parameter",
    };
    PropagationError::NumericalFailure(format!("covariance {reason}"))
}

fn identity_stm() -> StateTransitionMatrix {
    let mut matrix = [[0.0_f64; 6]; 6];
    for (idx, row) in matrix.iter_mut().enumerate() {
        row[idx] = 1.0;
    }
    matrix
}

fn finite_difference_step(component: f64) -> f64 {
    (component.abs().max(1.0) * STM_RELATIVE_PERTURBATION).max(STM_MIN_PERTURBATION)
}

fn perturb_state(state: CartesianState, component: usize, delta: f64) -> CartesianState {
    let mut perturbed = state;
    match component {
        0 => perturbed.position_km.x += delta,
        1 => perturbed.position_km.y += delta,
        2 => perturbed.position_km.z += delta,
        3 => perturbed.velocity_km_s.x += delta,
        4 => perturbed.velocity_km_s.y += delta,
        5 => perturbed.velocity_km_s.z += delta,
        _ => unreachable!("state-transition matrix component index is in 0..6"),
    }
    perturbed
}

fn state_vector(state: &CartesianState) -> [f64; 6] {
    [
        state.position_km.x,
        state.position_km.y,
        state.position_km.z,
        state.velocity_km_s.x,
        state.velocity_km_s.y,
        state.velocity_km_s.z,
    ]
}

fn validate_ephemeris_epochs(epochs_tdb_seconds: &[f64]) -> Result<(), PropagationError> {
    for &epoch_tdb_seconds in epochs_tdb_seconds {
        validate_epoch_finite(epoch_tdb_seconds, "epochs_tdb_seconds")?;
    }
    Ok(())
}

fn validate_initial_state(initial: CartesianState) -> Result<(), PropagationError> {
    validate_state_vector(initial.position_array(), "initial.position_km")?;
    validate_state_vector(initial.velocity_array(), "initial.velocity_km_s")
}

fn validate_stm(stm: &StateTransitionMatrix) -> Result<(), PropagationError> {
    for row in stm {
        crate::validate::finite_slice(row, "state_transition_matrix").map_err(|error| {
            PropagationError::NumericalFailure(format!("{} {}", error.field(), error.reason()))
        })?;
    }
    Ok(())
}

fn validate_state_vector(values: [f64; 3], field: &'static str) -> Result<(), PropagationError> {
    crate::validate::finite_slice(&values, field).map_err(|error| {
        PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
    })
}

fn validate_epoch_finite(value: f64, field: &'static str) -> Result<(), PropagationError> {
    crate::validate::finite(value, field)
        .map(|_| ())
        .map_err(|error| {
            PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::forces::{DragParameters, SpaceWeather, SpaceWeatherSource};
    use crate::astro::integrators::Integrator;
    use nalgebra::Vector3;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    struct CountingForce<'a> {
        calls: &'a AtomicUsize,
    }

    impl ForceModel for CountingForce<'_> {
        fn acceleration(
            &self,
            _state: &CartesianState,
            _ctx: &PropagationContext,
        ) -> Result<Vector3<f64>, PropagationError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(Vector3::zeros())
        }
    }

    #[derive(Default)]
    struct EpochRecordingForce {
        epochs: Mutex<Vec<f64>>,
    }

    impl ForceModel for EpochRecordingForce {
        fn acceleration(
            &self,
            state: &CartesianState,
            _ctx: &PropagationContext,
        ) -> Result<Vector3<f64>, PropagationError> {
            self.epochs
                .lock()
                .expect("epoch recorder mutex")
                .push(state.epoch_tdb_seconds);
            Ok(Vector3::zeros())
        }
    }

    fn circular_state() -> ([f64; 3], [f64; 3], f64) {
        let r: f64 = 7000.0;
        let v = (MU_EARTH / r).sqrt();
        ([r, 0.0, 0.0], [0.0, v, 0.0], r)
    }

    fn leo_state(altitude_km: f64) -> CartesianState {
        let r = RE_EARTH + altitude_km;
        let v = (MU_EARTH / r).sqrt();
        CartesianState::new(0.0, [r, 0.0, 0.0], [0.0, v, 0.0])
    }

    fn test_drag_parameters(bc_factor_m2_kg: f64) -> DragParameters {
        DragParameters::from_bc_factor_m2_kg(
            bc_factor_m2_kg,
            SpaceWeather::default(),
            crate::astro::forces::DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
        )
        .expect("valid drag")
    }

    fn assert_states_bit_for_bit(left: &[CartesianState], right: &[CartesianState]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right.iter()) {
            assert_eq!(
                left.epoch_tdb_seconds.to_bits(),
                right.epoch_tdb_seconds.to_bits()
            );
            for idx in 0..3 {
                assert_eq!(
                    left.position_array()[idx].to_bits(),
                    right.position_array()[idx].to_bits()
                );
                assert_eq!(
                    left.velocity_array()[idx].to_bits(),
                    right.velocity_array()[idx].to_bits()
                );
            }
        }
    }

    fn rk4_test_options() -> IntegratorOptions {
        IntegratorOptions {
            initial_step: 1.0,
            ..IntegratorOptions::default()
        }
    }

    fn dp54_drag_options() -> IntegratorOptions {
        IntegratorOptions {
            abs_tol: 1.0e-9,
            rel_tol: 1.0e-11,
            initial_step: 30.0,
            min_step: 1.0e-6,
            max_step: 120.0,
            max_steps: 200_000,
            dense_output: false,
        }
    }

    fn rk4_two_body_propagator(initial: CartesianState) -> StatePropagator {
        StatePropagator {
            initial,
            force_model: ForceModelKind::two_body(),
            integrator: IntegratorKind::Rk4,
            options: rk4_test_options(),
            drag: None,
            space_weather: None,
        }
    }

    fn circular_rk4_two_body_propagator() -> StatePropagator {
        let (pos, vel, _) = circular_state();
        rk4_two_body_propagator(CartesianState::new(0.0, pos, vel))
    }

    #[test]
    fn two_body_and_j2_bits_unchanged_when_drag_none() {
        let state = CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [1.0, 7.2, 0.5]);
        let options = IntegratorOptions {
            initial_step: 10.0,
            ..IntegratorOptions::default()
        };
        let two_body = StatePropagator {
            initial: state,
            force_model: ForceModelKind::two_body(),
            integrator: IntegratorKind::Rk4,
            options,
            drag: None,
            space_weather: None,
        }
        .propagate_to(120.0)
        .expect("two-body propagation")
        .final_state;
        let j2 = StatePropagator {
            initial: state,
            force_model: ForceModelKind::two_body_j2(),
            integrator: IntegratorKind::Rk4,
            options,
            drag: None,
            space_weather: None,
        }
        .propagate_to(120.0)
        .expect("J2 propagation")
        .final_state;

        assert_eq!(
            [
                two_body.position_km.x.to_bits(),
                two_body.position_km.y.to_bits(),
                two_body.position_km.z.to_bits(),
                two_body.velocity_km_s.x.to_bits(),
                two_body.velocity_km_s.y.to_bits(),
                two_body.velocity_km_s.z.to_bits(),
            ],
            [
                4_664_491_478_405_259_647,
                13_868_042_866_614_843_437,
                4_653_651_863_611_153_978,
                4_592_023_743_103_375_898,
                4_619_903_732_185_607_459,
                4_599_631_655_498_578_350,
            ]
        );
        assert_eq!(
            [
                j2.position_km.x.to_bits(),
                j2.position_km.y.to_bits(),
                j2.position_km.z.to_bits(),
                j2.velocity_km_s.x.to_bits(),
                j2.velocity_km_s.y.to_bits(),
                j2.velocity_km_s.z.to_bits(),
            ],
            [
                4_664_491_415_796_994_328,
                13_868_042_735_578_520_184,
                4_653_651_704_383_841_626,
                4_591_955_084_532_107_724,
                4_619_903_849_988_711_109,
                4_599_620_700_962_266_984,
            ]
        );
    }

    #[test]
    fn composite_with_phase_a_terms_off_matches_two_body_bit_for_bit() {
        let state = CartesianState::new(0.0, [7000.0, -1210.0, 1300.0], [1.0, 7.2, 0.5]);
        let options = IntegratorOptions {
            initial_step: 10.0,
            ..IntegratorOptions::default()
        };
        let legacy = StatePropagator {
            initial: state,
            force_model: ForceModelKind::two_body(),
            integrator: IntegratorKind::Rk4,
            options,
            drag: None,
            space_weather: None,
        }
        .ephemeris(&[0.0, 60.0, 120.0])
        .expect("legacy two-body ephemeris");
        let composite = StatePropagator {
            initial: state,
            force_model: ForceModelKind::composite(ForceModelComponents::earth_two_body()),
            integrator: IntegratorKind::Rk4,
            options,
            drag: None,
            space_weather: None,
        }
        .ephemeris(&[0.0, 60.0, 120.0])
        .expect("composite two-body ephemeris");

        assert_states_bit_for_bit(&legacy, &composite);
    }

    #[test]
    fn solid_earth_tide_must_share_the_field_tide_system() {
        let tide_free = SolidEarthTideGravity::default();
        let zero_tide = SolidEarthTideGravity {
            tide_system: TideSystem::ZeroTide,
            ..tide_free
        };
        let zonal = ForceModelComponents::earth_phase_a(None);
        let harmonic = ForceModelComponents::earth_phase_b(4, 4, None).expect("phase B");

        // The default zonal coefficients and the embedded EGM96 table are
        // tide-free, as is the default tide force.
        for components in [zonal, harmonic] {
            ForceModelKind::composite(components.with_solid_earth_tide(tide_free))
                .build()
                .expect("matching tide systems");
            let error = ForceModelKind::composite(components.with_solid_earth_tide(zero_tide))
                .build()
                .err()
                .expect("mismatched tide systems are refused");
            assert!(
                format!("{error}")
                    .contains("ZeroTide geopotential but the selected field is TideFree"),
                "{error}"
            );
        }

        // A zero-tide zonal field takes a zero-tide tide force.
        let mut zero_tide_zonal = ZonalGravity::earth_j2_through_j6();
        zero_tide_zonal.coefficients.tide_system = TideSystem::ZeroTide;
        let components = ForceModelComponents::earth_two_body().with_zonal(zero_tide_zonal);
        ForceModelKind::composite(components.with_solid_earth_tide(zero_tide))
            .build()
            .expect("matching zero-tide systems");
        ForceModelKind::composite(components.with_solid_earth_tide(tide_free))
            .build()
            .err()
            .expect("a tide-free tide force on a zero-tide field is refused");

        // Without J2 the zonal field holds no C20, so it counts as tide-free
        // whatever its coefficients say: a zero-tide tide force would remove a
        // permanent tide the field does not hold.
        let mut without_j2 = zero_tide_zonal;
        without_j2.degrees.j2 = false;
        let components = ForceModelComponents::earth_two_body().with_zonal(without_j2);
        ForceModelKind::composite(components.with_solid_earth_tide(tide_free))
            .build()
            .expect("a zonal field without J2 pairs with a tide-free tide force");
        ForceModelKind::composite(components.with_solid_earth_tide(zero_tide))
            .build()
            .err()
            .expect("a zero-tide tide force on a zonal field without J2 is refused");

        // Without a zonal or spherical-harmonic field the tide force's own
        // setting stands.
        ForceModelKind::composite(
            ForceModelComponents::earth_two_body().with_solid_earth_tide(zero_tide),
        )
        .build()
        .expect("no field to disagree with");
    }

    #[test]
    fn earth_radiation_pressure_component_is_opt_in() {
        let components = ForceModelComponents::earth_phase_a(None);
        assert_eq!(components.earth_radiation_pressure, None);
        assert_eq!(components.solid_earth_tide, None);
        assert_eq!(components.solid_earth_pole_tide, None);

        let pressure = EarthRadiationPressure::new(1.2, 0.011).expect("valid model");
        let with_pressure = components.with_earth_radiation_pressure(pressure);
        assert_eq!(components.earth_radiation_pressure, None);
        assert_eq!(with_pressure.earth_radiation_pressure, Some(pressure));

        let with_tides = components
            .with_solid_earth_tide(SolidEarthTideGravity::default())
            .with_solid_earth_pole_tide(SolidEarthPoleTideGravity::default());
        assert_eq!(components.solid_earth_tide, None);
        assert_eq!(components.solid_earth_pole_tide, None);
        assert_eq!(
            with_tides.solid_earth_tide,
            Some(SolidEarthTideGravity::default())
        );
        assert_eq!(
            with_tides.solid_earth_pole_tide,
            Some(SolidEarthPoleTideGravity::default())
        );
    }

    #[test]
    fn phase_a_all_forces_leo_smoke_stays_bounded() {
        let initial = leo_state(500.0);
        let srp = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
        let propagator = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::earth_phase_a(Some(srp)),
            IntegratorKind::Dp54,
        )
        .with_options(IntegratorOptions {
            abs_tol: 1.0e-10,
            rel_tol: 1.0e-12,
            initial_step: 30.0,
            max_step: 120.0,
            ..IntegratorOptions::default()
        })
        .with_drag(test_drag_parameters(0.02));

        let result = propagator
            .propagate_to(initial.epoch_tdb_seconds + 1800.0)
            .expect("Phase A propagation");
        let final_state = result.final_state;

        assert!(final_state
            .position_km
            .iter()
            .all(|value| value.is_finite()));
        assert!(final_state
            .velocity_km_s
            .iter()
            .all(|value| value.is_finite()));
        assert!((6500.0..8000.0).contains(&final_state.position_km.norm()));
        assert!((6.0..9.0).contains(&final_state.velocity_km_s.norm()));
    }

    #[test]
    fn fixed_space_weather_source_matches_fixed_drag_ephemeris_bit_for_bit() {
        let initial = leo_state(250.0);
        let drag = test_drag_parameters(0.15);
        let fixed = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Dp54,
        )
        .with_options(dp54_drag_options())
        .with_drag(drag);
        let sourced = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Dp54,
        )
        .with_options(dp54_drag_options())
        .with_drag(drag)
        .with_space_weather(SpaceWeatherSource::Fixed(drag.space_weather()));
        let epochs: Vec<f64> = (0..=6)
            .map(|i| initial.epoch_tdb_seconds + i as f64 * 300.0)
            .collect();

        let fixed_states = fixed.ephemeris(&epochs).expect("fixed ephemeris");
        let sourced_states = sourced.ephemeris(&epochs).expect("sourced ephemeris");
        assert_states_bit_for_bit(&fixed_states, &sourced_states);
    }

    #[test]
    fn space_weather_source_without_drag_is_invalid_input() {
        let initial = leo_state(250.0);
        let err = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Rk4,
        )
        .with_space_weather(SpaceWeatherSource::Fixed(SpaceWeather::default()))
        .propagate_to(initial.epoch_tdb_seconds + 60.0)
        .expect_err("source without drag fails");
        match err {
            PropagationError::InvalidInput(message) => {
                assert!(message.contains("space weather source without drag"));
            }
            other => panic!("expected invalid input, got {other:?}"),
        }
    }

    #[test]
    fn entry_point_matches_integrator_called_directly_bit_for_bit() {
        let (pos, vel, _) = circular_state();
        let opts = IntegratorOptions {
            abs_tol: 1e-12,
            rel_tol: 1e-12,
            ..IntegratorOptions::default()
        };

        // Entry point.
        let propagator = StatePropagator::new(
            0.0,
            pos,
            vel,
            ForceModelKind::two_body(),
            IntegratorKind::Dp54,
        )
        .with_options(IntegratorOptions {
            abs_tol: 1e-12,
            rel_tol: 1e-12,
            ..IntegratorOptions::default()
        });
        let via_entry = propagator
            .propagate_to(crate::constants::SECONDS_PER_HOUR)
            .unwrap()
            .final_state;

        // Integrator called directly, the way the entry point assembles it.
        let force = TwoBodyGravity::default();
        let dynamics = OrbitalDynamics {
            force_model: &force,
        };
        let ctx = PropagationContext::default();
        let via_direct = DP54
            .propagate(
                CartesianState::new(0.0, pos, vel),
                crate::constants::SECONDS_PER_HOUR,
                &dynamics,
                &ctx,
                &opts,
            )
            .unwrap()
            .final_state;

        assert_eq!(
            via_entry.position_km.x.to_bits(),
            via_direct.position_km.x.to_bits()
        );
        assert_eq!(
            via_entry.position_km.y.to_bits(),
            via_direct.position_km.y.to_bits()
        );
        assert_eq!(
            via_entry.position_km.z.to_bits(),
            via_direct.position_km.z.to_bits()
        );
        assert_eq!(
            via_entry.velocity_km_s.x.to_bits(),
            via_direct.velocity_km_s.x.to_bits()
        );
        assert_eq!(
            via_entry.velocity_km_s.y.to_bits(),
            via_direct.velocity_km_s.y.to_bits()
        );
        assert_eq!(
            via_entry.velocity_km_s.z.to_bits(),
            via_direct.velocity_km_s.z.to_bits()
        );
    }

    #[test]
    fn ephemeris_last_sample_matches_single_shot_propagation() {
        // ephemeris() segments from epoch->t1->t2; the final sample must equal a
        // single propagate_to() over the same span only when the intermediate
        // node lands on the same point. We assert the cheaper invariant: the
        // first sample is the initial state, and the sample count matches.
        let (pos, vel, _) = circular_state();
        let propagator = StatePropagator::new(
            100.0,
            pos,
            vel,
            ForceModelKind::two_body(),
            IntegratorKind::Dp54,
        );
        let epochs = [100.0, 700.0, 1300.0];
        let states = propagator.ephemeris(&epochs).unwrap();

        assert_eq!(states.len(), 3);
        // First sample is the initial state, untouched.
        assert_eq!(states[0].position_km.x.to_bits(), pos[0].to_bits());
        assert_eq!(states[0].velocity_km_s.y.to_bits(), vel[1].to_bits());
        for (state, &t) in states.iter().zip(epochs.iter()) {
            assert_eq!(state.epoch_tdb_seconds, t);
        }
    }

    #[test]
    fn state_transition_matrix_zero_span_is_identity() {
        let propagator = circular_rk4_two_body_propagator();
        let stm = propagator.state_transition_matrix_for_span(0.0).unwrap();

        for (i, row) in stm.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                let expected = if i == j { 1.0_f64 } else { 0.0_f64 };
                assert_eq!(value.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn state_transition_matrix_has_short_span_two_body_structure() {
        let propagator = circular_rk4_two_body_propagator();
        let span = 10.0;
        let stm = propagator.state_transition_matrix_for_span(span).unwrap();

        for axis in 0..3 {
            assert_close(stm[axis][axis], 1.0, 2.0e-4);
            assert_close(stm[axis][axis + 3], span, 2.0e-3);
            assert_close(stm[axis + 3][axis + 3], 1.0, 2.0e-4);
        }
    }

    #[test]
    fn state_transition_matrix_matches_independent_perturbation() {
        let propagator = circular_rk4_two_body_propagator();
        let span = 60.0;
        let stm = propagator.state_transition_matrix_for_span(span).unwrap();
        let base_final = propagator.propagate_to(span).unwrap().final_state;

        let delta = [2.0e-4, -1.5e-4, 1.0e-4, 2.0e-7, -1.0e-7, 1.5e-7];
        let mut perturbed_initial = propagator.initial;
        for (component, &value) in delta.iter().enumerate() {
            perturbed_initial = perturb_state(perturbed_initial, component, value);
        }
        let perturbed_final = rk4_two_body_propagator(perturbed_initial)
            .propagate_to(span)
            .unwrap()
            .final_state;

        let base_vector = state_vector(&base_final);
        let perturbed_vector = state_vector(&perturbed_final);
        let predicted = mat6_vec6(&stm, &delta);

        for row in 0..6 {
            let observed = perturbed_vector[row] - base_vector[row];
            let tolerance = if row < 3 { 2.0e-8 } else { 2.0e-10 };
            assert_close(predicted[row], observed, tolerance);
        }
    }

    #[test]
    fn state_transition_matrix_is_symplectic_for_short_two_body_span() {
        let propagator = circular_rk4_two_body_propagator();
        let stm = propagator.state_transition_matrix_for_span(30.0).unwrap();

        assert!(max_symplectic_residual(&stm) < 1.0e-5);
    }

    #[test]
    fn stm_with_drag_test() {
        let initial = leo_state(300.0);
        let propagator = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Rk4,
        )
        .with_options(IntegratorOptions {
            initial_step: 2.0,
            ..IntegratorOptions::default()
        })
        .with_drag(test_drag_parameters(0.2));
        let stm = propagator
            .state_transition_matrix_for_span(8.0)
            .expect("drag STM");

        for row in stm {
            for value in row {
                assert!(value.is_finite());
            }
        }
        assert!(stm[0][3] > 0.0);
        assert!(stm[1][4] > 0.0);
        assert!(stm[2][5] > 0.0);
    }

    #[test]
    fn integrator_presents_advancing_substep_epoch() {
        let initial = CartesianState::new(10.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]);
        let options = IntegratorOptions {
            initial_step: 10.0,
            max_step: 10.0,
            ..IntegratorOptions::default()
        };

        for integrator in [IntegratorKind::Rk4, IntegratorKind::Dp54] {
            let force = EpochRecordingForce::default();
            let dynamics = OrbitalDynamics {
                force_model: &force,
            };
            let ctx = PropagationContext::default();
            match integrator {
                IntegratorKind::Rk4 => {
                    RK4.propagate(initial, 20.0, &dynamics, &ctx, &options)
                        .expect("RK4 propagation");
                }
                IntegratorKind::Dp54 => {
                    DP54.propagate(initial, 20.0, &dynamics, &ctx, &options)
                        .expect("DP54 propagation");
                }
            }
            let epochs = force.epochs.lock().expect("epoch recorder mutex");
            assert!(epochs
                .iter()
                .any(|&epoch| epoch > initial.epoch_tdb_seconds));
            assert!(epochs.contains(&20.0));
        }
    }

    #[test]
    fn stm_with_drag_differs_from_no_drag() {
        let initial = leo_state(300.0);
        let options = IntegratorOptions {
            initial_step: 2.0,
            ..IntegratorOptions::default()
        };
        let no_drag = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Rk4,
        )
        .with_options(options)
        .state_transition_matrix_for_span(20.0)
        .expect("no-drag STM");
        let with_drag = StatePropagator::new(
            initial.epoch_tdb_seconds,
            initial.position_array(),
            initial.velocity_array(),
            ForceModelKind::two_body(),
            IntegratorKind::Rk4,
        )
        .with_options(options)
        .with_drag(test_drag_parameters(0.4))
        .state_transition_matrix_for_span(20.0)
        .expect("drag STM");

        let mut max_diff = 0.0_f64;
        for row in 0..6 {
            for col in 0..6 {
                max_diff = max_diff.max((with_drag[row][col] - no_drag[row][col]).abs());
            }
        }
        assert!(max_diff > 1.0e-10, "STM diff {max_diff}");
    }

    #[test]
    fn propagate_state_with_covariance_zero_span_returns_initial_inputs() {
        let propagator = circular_rk4_two_body_propagator();
        let covariance = test_covariance();

        let (state, propagated_covariance) = propagator
            .propagate_state_with_covariance(covariance, 0.0)
            .unwrap();

        assert_eq!(state, propagator.initial);
        assert_eq!(propagated_covariance, covariance);
    }

    #[test]
    fn propagate_state_with_covariance_keeps_covariance_psd_and_coupled() {
        let propagator = circular_rk4_two_body_propagator();
        let covariance0 = test_covariance();
        let span = 120.0;

        let (state, covariance_f) = propagator
            .propagate_state_with_covariance(covariance0, span)
            .unwrap();

        assert_eq!(state.epoch_tdb_seconds, span);
        assert!(covariance_f.is_symmetric());
        assert!(covariance_f.is_positive_semidefinite());

        let p0 = covariance0.as_matrix();
        let pf = covariance_f.as_matrix();
        let initial_position_trace = p0[0][0] + p0[1][1] + p0[2][2];
        let final_position_trace = pf[0][0] + pf[1][1] + pf[2][2];
        assert!(final_position_trace > initial_position_trace);

        let max_position_velocity_coupling = (0..3)
            .flat_map(|i| (3..6).map(move |j| pf[i][j].abs()))
            .fold(0.0_f64, f64::max);
        assert!(max_position_velocity_coupling > 1.0e-8);
    }

    #[test]
    fn propagator_rejects_zero_initial_step() {
        let (pos, vel, _) = circular_state();
        let propagator = StatePropagator::new(
            0.0,
            pos,
            vel,
            ForceModelKind::two_body(),
            IntegratorKind::Rk4,
        )
        .with_options(IntegratorOptions {
            initial_step: 0.0,
            ..IntegratorOptions::default()
        });

        assert_invalid_propagation_field(
            propagator.propagate_to(60.0).unwrap_err(),
            "initial_step",
        );
    }

    #[test]
    fn rejects_non_finite_epochs_before_running_integrator() {
        let (pos, vel, _) = circular_state();
        let calls = AtomicUsize::new(0);
        let force = CountingForce { calls: &calls };
        let dynamics = OrbitalDynamics {
            force_model: &force,
        };
        let ctx = PropagationContext::default();
        let propagator = StatePropagator::new(
            0.0,
            pos,
            vel,
            ForceModelKind::two_body(),
            IntegratorKind::Dp54,
        );

        let cases = [
            (
                CartesianState::new(f64::NAN, pos, vel),
                60.0,
                "initial.epoch_tdb_seconds",
            ),
            (
                CartesianState::new(0.0, pos, vel),
                f64::INFINITY,
                "t_end_tdb_seconds",
            ),
        ];

        for (initial, t_end, field) in cases {
            calls.store(0, Ordering::SeqCst);
            let err = propagator
                .run(initial, t_end, &dynamics, &ctx)
                .expect_err("non-finite epoch should be rejected");

            assert_non_finite_epoch_error(err, field);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "non-finite {field} must not enter the integrator"
            );
        }
    }

    #[test]
    fn ephemeris_rejects_non_finite_query_epochs_before_first_segment() {
        let (pos, vel, _) = circular_state();
        let propagator = StatePropagator::new(
            0.0,
            pos,
            vel,
            ForceModelKind::two_body(),
            IntegratorKind::Rk4,
        )
        .with_options(IntegratorOptions {
            initial_step: 0.0,
            ..IntegratorOptions::default()
        });

        let err = propagator
            .ephemeris(&[60.0, f64::NAN])
            .expect_err("non-finite query epoch should be rejected");

        assert_non_finite_epoch_error(err, "epochs_tdb_seconds");
    }

    #[test]
    fn rejects_non_finite_initial_state_vectors_before_running_integrator() {
        let (pos, vel, _) = circular_state();
        let calls = AtomicUsize::new(0);
        let force = CountingForce { calls: &calls };
        let dynamics = OrbitalDynamics {
            force_model: &force,
        };
        let ctx = PropagationContext::default();
        let propagator = StatePropagator::new(
            0.0,
            pos,
            vel,
            ForceModelKind::two_body(),
            IntegratorKind::Rk4,
        );

        let cases = [
            (
                CartesianState::new(0.0, [f64::NAN, pos[1], pos[2]], vel),
                "initial.position_km",
            ),
            (
                CartesianState::new(0.0, [pos[0], f64::INFINITY, pos[2]], vel),
                "initial.position_km",
            ),
            (
                CartesianState::new(0.0, pos, [vel[0], f64::NAN, vel[2]]),
                "initial.velocity_km_s",
            ),
            (
                CartesianState::new(0.0, pos, [vel[0], vel[1], f64::NEG_INFINITY]),
                "initial.velocity_km_s",
            ),
        ];

        for (initial, field) in cases {
            calls.store(0, Ordering::SeqCst);
            let err = propagator
                .run(initial, 60.0, &dynamics, &ctx)
                .expect_err("non-finite state vector should be rejected");

            assert_non_finite_state_error(err, field);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "non-finite {field} must not enter the integrator"
            );
        }
    }

    #[test]
    fn propagate_to_rejects_non_finite_integrator_outputs() {
        let (pos, vel, _) = circular_state();
        let propagator = StatePropagator::new(
            0.0,
            pos,
            vel,
            ForceModelKind::TwoBody {
                mu_km3_s2: f64::INFINITY,
            },
            IntegratorKind::Rk4,
        )
        .with_options(rk4_test_options());

        let err = propagator
            .propagate_to(1.0)
            .expect_err("non-finite integration result should be rejected");

        assert_output_non_finite_error(err, "final_state");
    }

    #[test]
    fn state_transition_matrix_rejects_non_finite_propagation_legs() {
        let (pos, vel, _) = circular_state();
        let propagator = StatePropagator::new(
            0.0,
            pos,
            vel,
            ForceModelKind::TwoBody {
                mu_km3_s2: f64::INFINITY,
            },
            IntegratorKind::Rk4,
        )
        .with_options(rk4_test_options());

        let err = propagator
            .state_transition_matrix_for_span(1.0)
            .expect_err("non-finite STM propagation leg should be rejected");

        assert_output_non_finite_error(err, "final_state");
    }

    fn assert_invalid_propagation_field(error: PropagationError, expected: &str) {
        match error {
            PropagationError::InvalidInput(message) => {
                assert!(message.contains(expected), "{message}");
                assert!(message.contains("not positive"), "{message}");
            }
            other => panic!("expected invalid propagation input for {expected}, got {other:?}"),
        }
    }

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "{actual} differs from {expected} by more than {tolerance}"
        );
    }

    fn test_covariance() -> Covariance6 {
        Covariance6::from_diagonal([1.0e-6, 2.0e-6, 3.0e-6, 1.0e-8, 2.0e-8, 3.0e-8]).unwrap()
    }

    fn mat6_vec6(matrix: &StateTransitionMatrix, vector: &[f64; 6]) -> [f64; 6] {
        let mut out = [0.0_f64; 6];
        for (i, row) in matrix.iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                out[i] += value * vector[j];
            }
        }
        out
    }

    fn max_symplectic_residual(phi: &StateTransitionMatrix) -> f64 {
        let mut max = 0.0_f64;
        for i in 0..6 {
            for j in 0..6 {
                let mut value = 0.0_f64;
                for k in 0..6 {
                    for l in 0..6 {
                        value += phi[k][i] * canonical_j(k, l) * phi[l][j];
                    }
                }
                let residual = (value - canonical_j(i, j)).abs();
                max = max.max(residual);
            }
        }
        max
    }

    fn canonical_j(row: usize, col: usize) -> f64 {
        if row < 3 && col == row + 3 {
            1.0
        } else if row >= 3 && col + 3 == row {
            -1.0
        } else {
            0.0
        }
    }

    fn assert_non_finite_epoch_error(error: PropagationError, expected: &str) {
        match error {
            PropagationError::InvalidInput(message) => {
                assert!(message.contains(expected), "{message}");
                assert!(message.contains("not finite"), "{message}");
            }
            other => panic!("expected invalid epoch input for {expected}, got {other:?}"),
        }
    }

    fn assert_non_finite_state_error(error: PropagationError, expected: &str) {
        match error {
            PropagationError::InvalidInput(message) => {
                assert!(message.contains(expected), "{message}");
                assert!(message.contains("not finite"), "{message}");
            }
            other => panic!("expected invalid state input for {expected}, got {other:?}"),
        }
    }

    fn assert_output_non_finite_error(error: PropagationError, expected: &str) {
        match error {
            PropagationError::InvalidInput(message)
            | PropagationError::NumericalFailure(message) => {
                assert!(message.contains(expected), "{message}");
                assert!(message.contains("not finite"), "{message}");
            }
            other => panic!("expected non-finite output for {expected}, got {other:?}"),
        }
    }
}
