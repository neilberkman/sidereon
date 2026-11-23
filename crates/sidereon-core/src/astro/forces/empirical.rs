//! Empirical accelerations in the radial, along-track and cross-track frame.
//!
//! The acceleration is, per axis, a constant plus a once-per-revolution pair:
//!
//! ```text
//! a = sum over axis k in {R, T, N} of (c_k0 + c_kc cos u + c_ks sin u) e_k
//! ```
//!
//! with `e_R = r / |r|`, `e_N = (r x v) / |r x v|`, `e_T = e_N x e_R`, and `u`
//! the argument of latitude of the osculating state: the angle in the orbit
//! plane from the ascending node on the inertial equator to the satellite,
//! `cos u = e_R . n`, `sin u = e_R . (e_N x n)`, `n = (z x e_N) / |z x e_N|`.
//! Each of the nine terms is present or absent individually; the present
//! terms are the model's estimable parameters, in km/s^2.
//!
//! Empirical accelerations of this form absorb unmodelled forces in orbit
//! determination (Montenbruck & Gill, "Satellite Orbits", chapters 3 and 8).
//! The partial derivatives with respect to position and velocity, which the
//! variational equations of chapter 7 take, follow from differentiating the
//! unit vectors and `u`:
//!
//! ```text
//! d e_R = (I - e_R e_Rᵀ) dr / |r|
//! d e_N = (I - e_N e_Nᵀ) (dr x v + r x dv) / |r x v|
//! d e_T = d e_N x e_R + e_N x d e_R
//! d u   = e_T . dr / |r| - cot(i) n . (dr x v + r x dv) / |r x v|
//! ```
//!
//! The constant terms are Orekit's `ParametricAcceleration` with a degree-0
//! `PolynomialAccelerationModel` along the axes of its `QSW` local orbital
//! frame. Orekit's `HarmonicAccelerationModel` is a harmonic of elapsed time
//! with a phase parameter, not of the argument of latitude, so the
//! once-per-revolution terms are compared with Orekit through an acceleration
//! model written for the purpose (`tests/orbit_parameter_fit_orekit.rs`).

use crate::astro::error::PropagationError;
use crate::astro::forces::r#trait::{
    expect_parameter_count, finite_acceleration, finite_partials, outer, validate_finite_state,
    AccelerationPartials, ForceModel, ForceParameter,
};
use crate::astro::propagator::api::PropagationContext;
use crate::astro::state::CartesianState;
use nalgebra::{Matrix3, Vector3};

/// `|z x e_N|`, the sine of the inclination, below which the ascending node
/// and with it the argument of latitude are undefined.
const EQUATORIAL_SIN_INCLINATION: f64 = 1.0e-12;

/// An axis of the radial, along-track, cross-track frame of an orbit state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum RtnAxis {
    /// Along the position vector.
    Radial,
    /// In the orbit plane, perpendicular to the position vector, on the side
    /// of the velocity.
    AlongTrack,
    /// Along the orbital angular momentum `r x v`.
    CrossTrack,
}

impl RtnAxis {
    /// The three axes in the order radial, along-track, cross-track.
    pub const ALL: [Self; 3] = [Self::Radial, Self::AlongTrack, Self::CrossTrack];

    const fn index(self) -> usize {
        match self {
            Self::Radial => 0,
            Self::AlongTrack => 1,
            Self::CrossTrack => 2,
        }
    }
}

/// The dependence of an empirical term on the argument of latitude `u`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EmpiricalHarmonic {
    /// Constant.
    Constant,
    /// Proportional to `cos u`.
    Cosine,
    /// Proportional to `sin u`.
    Sine,
}

impl EmpiricalHarmonic {
    /// The three dependences in the order constant, cosine, sine.
    pub const ALL: [Self; 3] = [Self::Constant, Self::Cosine, Self::Sine];

    const fn index(self) -> usize {
        match self {
            Self::Constant => 0,
            Self::Cosine => 1,
            Self::Sine => 2,
        }
    }
}

/// Empirical acceleration in the radial, along-track, cross-track frame: per
/// axis a constant and a cosine and sine of the argument of latitude, each
/// present or absent.
///
/// The present terms are the model's [`ForceModel::parameters`], ordered by
/// axis and then by [`EmpiricalHarmonic`], named by
/// [`EmpiricalRtnAcceleration::parameter_name`]. A model with no cosine or
/// sine term does not evaluate the argument of latitude and so is defined for
/// equatorial orbits too.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EmpiricalRtnAcceleration {
    /// `[axis][harmonic]` coefficient in km/s^2, `None` for an absent term.
    coefficients_km_s2: [[Option<f64>; 3]; 3],
}

impl EmpiricalRtnAcceleration {
    /// A model with no terms; it contributes no acceleration.
    pub const fn new() -> Self {
        Self {
            coefficients_km_s2: [[None; 3]; 3],
        }
    }

    /// A model with every listed term present at coefficient zero, the usual
    /// starting point of an estimation.
    pub fn with_zero_terms(terms: &[(RtnAxis, EmpiricalHarmonic)]) -> Self {
        let mut model = Self::new();
        for &(axis, harmonic) in terms {
            model.coefficients_km_s2[axis.index()][harmonic.index()] = Some(0.0);
        }
        model
    }

    /// The constant terms of all three axes at coefficient zero.
    pub fn constant_terms() -> Self {
        Self::with_zero_terms(&[
            (RtnAxis::Radial, EmpiricalHarmonic::Constant),
            (RtnAxis::AlongTrack, EmpiricalHarmonic::Constant),
            (RtnAxis::CrossTrack, EmpiricalHarmonic::Constant),
        ])
    }

    /// All nine terms at coefficient zero.
    pub fn all_terms() -> Self {
        Self {
            coefficients_km_s2: [[Some(0.0); 3]; 3],
        }
    }

    /// Add a term, or replace its coefficient. `coefficient_km_s2` is in
    /// km/s^2 and must be finite.
    pub fn with_term(
        mut self,
        axis: RtnAxis,
        harmonic: EmpiricalHarmonic,
        coefficient_km_s2: f64,
    ) -> Result<Self, PropagationError> {
        if !coefficient_km_s2.is_finite() {
            return Err(PropagationError::InvalidInput(format!(
                "{} must be finite",
                Self::parameter_name(axis, harmonic)
            )));
        }
        self.coefficients_km_s2[axis.index()][harmonic.index()] = Some(coefficient_km_s2);
        Ok(self)
    }

    /// The coefficient of a term in km/s^2, or `None` when the term is absent.
    pub fn coefficient_km_s2(&self, axis: RtnAxis, harmonic: EmpiricalHarmonic) -> Option<f64> {
        self.coefficients_km_s2[axis.index()][harmonic.index()]
    }

    /// The present terms with their coefficients in km/s^2, in parameter order.
    pub fn terms(&self) -> Vec<(RtnAxis, EmpiricalHarmonic, f64)> {
        let mut terms = Vec::new();
        for axis in RtnAxis::ALL {
            for harmonic in EmpiricalHarmonic::ALL {
                if let Some(coefficient) = self.coefficient_km_s2(axis, harmonic) {
                    terms.push((axis, harmonic, coefficient));
                }
            }
        }
        terms
    }

    /// The name of a term in [`ForceModel::parameters`], for example
    /// `"empirical.along.cos"`.
    pub const fn parameter_name(axis: RtnAxis, harmonic: EmpiricalHarmonic) -> &'static str {
        match (axis, harmonic) {
            (RtnAxis::Radial, EmpiricalHarmonic::Constant) => "empirical.radial.constant",
            (RtnAxis::Radial, EmpiricalHarmonic::Cosine) => "empirical.radial.cos",
            (RtnAxis::Radial, EmpiricalHarmonic::Sine) => "empirical.radial.sin",
            (RtnAxis::AlongTrack, EmpiricalHarmonic::Constant) => "empirical.along.constant",
            (RtnAxis::AlongTrack, EmpiricalHarmonic::Cosine) => "empirical.along.cos",
            (RtnAxis::AlongTrack, EmpiricalHarmonic::Sine) => "empirical.along.sin",
            (RtnAxis::CrossTrack, EmpiricalHarmonic::Constant) => "empirical.cross.constant",
            (RtnAxis::CrossTrack, EmpiricalHarmonic::Cosine) => "empirical.cross.cos",
            (RtnAxis::CrossTrack, EmpiricalHarmonic::Sine) => "empirical.cross.sin",
        }
    }

    fn term_count(&self) -> usize {
        self.coefficients_km_s2
            .iter()
            .flatten()
            .filter(|coefficient| coefficient.is_some())
            .count()
    }

    fn has_harmonic_term(&self) -> bool {
        self.coefficients_km_s2.iter().any(|axis| {
            axis[EmpiricalHarmonic::Cosine.index()].is_some()
                || axis[EmpiricalHarmonic::Sine.index()].is_some()
        })
    }

    /// The `[axis][harmonic]` coefficients with the present terms taken from
    /// `parameters` in parameter order and the absent terms zero.
    fn coefficient_table(&self, parameters: &[f64]) -> Result<[[f64; 3]; 3], PropagationError> {
        expect_parameter_count(parameters, self.term_count())?;
        let mut table = [[0.0; 3]; 3];
        let mut values = parameters.iter();
        for axis in RtnAxis::ALL {
            for harmonic in EmpiricalHarmonic::ALL {
                if self.coefficient_km_s2(axis, harmonic).is_some() {
                    let value = values.next().copied().unwrap_or(0.0);
                    if !value.is_finite() {
                        return Err(PropagationError::InvalidInput(format!(
                            "{} must be finite",
                            Self::parameter_name(axis, harmonic)
                        )));
                    }
                    table[axis.index()][harmonic.index()] = value;
                }
            }
        }
        Ok(table)
    }

    fn own_parameters(&self) -> Vec<f64> {
        self.terms()
            .into_iter()
            .map(|(_, _, coefficient)| coefficient)
            .collect()
    }
}

/// The frame and the argument of latitude of one state.
struct RtnGeometry {
    /// `e_R`, `e_T`, `e_N`.
    axes: [Vector3<f64>; 3],
    position_norm: f64,
    momentum_norm: f64,
    /// `[1, cos u, sin u]`; the last two are zero when not evaluated.
    harmonics: [f64; 3],
    /// The unit ascending-node vector and `cot(i)`, when evaluated.
    node: Option<(Vector3<f64>, f64)>,
}

fn rtn_geometry(
    state: &CartesianState,
    with_argument_of_latitude: bool,
) -> Result<RtnGeometry, PropagationError> {
    validate_finite_state(state)?;
    let position = state.position_km;
    let velocity = state.velocity_km_s;
    let position_norm = position.norm();
    let momentum = position.cross(&velocity);
    let momentum_norm = momentum.norm();
    if !position_norm.is_finite() || !momentum_norm.is_finite() {
        return Err(PropagationError::NumericalFailure(
            "state too large for the radial, along-track, cross-track frame".to_string(),
        ));
    }
    if position_norm == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "Zero position magnitude".to_string(),
        ));
    }
    if momentum_norm == 0.0 {
        return Err(PropagationError::NumericalFailure(
            "position and velocity are parallel: the orbit plane is undefined".to_string(),
        ));
    }
    let radial = position / position_norm;
    let cross = momentum / momentum_norm;
    let along = cross.cross(&radial);

    let (harmonics, node) = if with_argument_of_latitude {
        // z x e_N lies along the ascending node; its length is sin(i).
        let node = Vector3::new(-cross.y, cross.x, 0.0);
        let sin_inclination = node.norm();
        if sin_inclination < EQUATORIAL_SIN_INCLINATION {
            return Err(PropagationError::NumericalFailure(
                "equatorial orbit: the argument of latitude is undefined".to_string(),
            ));
        }
        let node = node / sin_inclination;
        let cos_u = radial.dot(&node);
        let sin_u = radial.dot(&cross.cross(&node));
        ([1.0, cos_u, sin_u], Some((node, cross.z / sin_inclination)))
    } else {
        ([1.0, 0.0, 0.0], None)
    };

    Ok(RtnGeometry {
        axes: [radial, along, cross],
        position_norm,
        momentum_norm,
        harmonics,
        node,
    })
}

/// `[a]x`, the matrix with `[a]x b = a x b`.
fn cross_matrix(a: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -a.z, a.y, a.z, 0.0, -a.x, -a.y, a.x, 0.0)
}

impl ForceModel for EmpiricalRtnAcceleration {
    fn acceleration(
        &self,
        state: &CartesianState,
        ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        self.acceleration_with_parameters(&self.own_parameters(), state, ctx)
    }

    fn parameters(&self) -> Vec<ForceParameter> {
        self.terms()
            .into_iter()
            .map(|(axis, harmonic, coefficient)| {
                ForceParameter::new(Self::parameter_name(axis, harmonic), coefficient)
            })
            .collect()
    }

    fn parameter_count(&self) -> usize {
        self.term_count()
    }

    fn acceleration_with_parameters(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vector3<f64>, PropagationError> {
        let table = self.coefficient_table(parameters)?;
        if parameters.is_empty() {
            return Ok(Vector3::zeros());
        }
        let geometry = rtn_geometry(state, self.has_harmonic_term())?;
        let mut accel = Vector3::zeros();
        for (axis, coefficients) in geometry.axes.iter().zip(&table) {
            let amplitude: f64 = coefficients
                .iter()
                .zip(&geometry.harmonics)
                .map(|(coefficient, harmonic)| coefficient * harmonic)
                .sum();
            accel += axis * amplitude;
        }
        finite_acceleration(accel)
    }

    fn acceleration_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<AccelerationPartials, PropagationError> {
        let table = self.coefficient_table(parameters)?;
        if parameters.is_empty() {
            return Ok(AccelerationPartials::zeros());
        }
        let geometry = rtn_geometry(state, self.has_harmonic_term())?;
        let [radial, along, cross] = geometry.axes;
        let identity = Matrix3::identity();

        // Derivatives of the unit vectors.
        let radial_wrt_position = (identity - outer(&radial, &radial)) / geometry.position_norm;
        let momentum_projector = (identity - outer(&cross, &cross)) / geometry.momentum_norm;
        // d(r x v) = dr x v + r x dv = -[v]x dr + [r]x dv.
        let cross_wrt_position = -momentum_projector * cross_matrix(&state.velocity_km_s);
        let cross_wrt_velocity = momentum_projector * cross_matrix(&state.position_km);
        let along_wrt_position =
            cross_matrix(&cross) * radial_wrt_position - cross_matrix(&radial) * cross_wrt_position;
        let along_wrt_velocity = -cross_matrix(&radial) * cross_wrt_velocity;
        let axes_wrt_position = [radial_wrt_position, along_wrt_position, cross_wrt_position];
        let axes_wrt_velocity = [Matrix3::zeros(), along_wrt_velocity, cross_wrt_velocity];

        // Gradient of the argument of latitude.
        let (u_wrt_position, u_wrt_velocity) = match geometry.node {
            Some((node, cot_inclination)) => {
                let scale = cot_inclination / geometry.momentum_norm;
                (
                    along / geometry.position_norm - state.velocity_km_s.cross(&node) * scale,
                    -node.cross(&state.position_km) * scale,
                )
            }
            None => (Vector3::zeros(), Vector3::zeros()),
        };

        let [_, cos_u, sin_u] = geometry.harmonics;
        let mut partials = AccelerationPartials::zeros();
        for (index, coefficients) in table.iter().enumerate() {
            let [constant, cosine, sine] = *coefficients;
            let amplitude = constant + cosine * cos_u + sine * sin_u;
            let amplitude_per_u = -cosine * sin_u + sine * cos_u;
            partials.wrt_position += axes_wrt_position[index] * amplitude
                + outer(&geometry.axes[index], &u_wrt_position) * amplitude_per_u;
            partials.wrt_velocity += axes_wrt_velocity[index] * amplitude
                + outer(&geometry.axes[index], &u_wrt_velocity) * amplitude_per_u;
        }
        finite_partials(partials)
    }

    fn parameter_partials(
        &self,
        parameters: &[f64],
        state: &CartesianState,
        _ctx: &PropagationContext,
    ) -> Result<Vec<Vector3<f64>>, PropagationError> {
        self.coefficient_table(parameters)?;
        if parameters.is_empty() {
            return Ok(Vec::new());
        }
        let geometry = rtn_geometry(state, self.has_harmonic_term())?;
        Ok(self
            .terms()
            .into_iter()
            .map(|(axis, harmonic, _)| {
                geometry.axes[axis.index()] * geometry.harmonics[harmonic.index()]
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    //! The frame and the argument of latitude are checked on a state built
    //! from its inclination, node and argument of latitude with the standard
    //! perifocal-to-inertial rotation (Montenbruck & Gill, chapter 2), not
    //! from the model's own geometry. The partial derivatives are checked
    //! against central difference quotients of the acceleration.

    use super::*;
    use crate::astro::forces::r#trait::difference_quotient_partials;

    /// A state at radius `radius_km` and argument of latitude `u`, moving at
    /// `speed_km_s` with flight-path angle `gamma` above the local horizontal,
    /// on an orbit of inclination `i` and node `raan`.
    fn state_from_orbit_angles(
        radius_km: f64,
        speed_km_s: f64,
        gamma: f64,
        inclination: f64,
        raan: f64,
        u: f64,
    ) -> (CartesianState, [Vector3<f64>; 3]) {
        let (sin_o, cos_o) = libm::sincos(raan);
        let (sin_i, cos_i) = libm::sincos(inclination);
        let (sin_u, cos_u) = libm::sincos(u);
        // Unit vectors of the orbit plane: toward the node, and 90 degrees
        // ahead of it.
        let p = Vector3::new(cos_o, sin_o, 0.0);
        let q = Vector3::new(-cos_i * sin_o, cos_i * cos_o, sin_i);
        let radial = p * cos_u + q * sin_u;
        let along = q * cos_u - p * sin_u;
        let cross = Vector3::new(sin_i * sin_o, -sin_i * cos_o, cos_i);
        let velocity = (radial * libm::sin(gamma) + along * libm::cos(gamma)) * speed_km_s;
        let position = radial * radius_km;
        (
            CartesianState {
                epoch_tdb_seconds: 0.0,
                position_km: position,
                velocity_km_s: velocity,
            },
            [radial, along, cross],
        )
    }

    fn all_terms_model() -> EmpiricalRtnAcceleration {
        let mut model = EmpiricalRtnAcceleration::new();
        let mut value = 1.0e-12;
        for axis in RtnAxis::ALL {
            for harmonic in EmpiricalHarmonic::ALL {
                model = model.with_term(axis, harmonic, value).expect("finite");
                value *= -1.7;
            }
        }
        model
    }

    #[test]
    fn acceleration_follows_the_frame_and_argument_of_latitude() {
        let u = 0.7;
        let (state, [radial, along, cross]) =
            state_from_orbit_angles(26_560.0, 3.9, 0.02, 0.96, 0.5, u);
        let ctx = PropagationContext::default();
        let (sin_u, cos_u) = libm::sincos(u);
        for (axis, unit) in RtnAxis::ALL.into_iter().zip([radial, along, cross]) {
            for (harmonic, factor) in EmpiricalHarmonic::ALL.into_iter().zip([1.0, cos_u, sin_u]) {
                let model = EmpiricalRtnAcceleration::new()
                    .with_term(axis, harmonic, 2.0e-12)
                    .expect("finite");
                let accel = model.acceleration(&state, &ctx).expect("acceleration");
                let expected = unit * (2.0e-12 * factor);
                // Rounding of the unit vectors and of cos u, sin u: a few ulps
                // of the 2e-12 km/s^2 amplitude.
                assert!(
                    (accel - expected).norm() <= 2.0e-12 * 8.0 * f64::EPSILON,
                    "{axis:?} {harmonic:?}: {accel:?} vs {expected:?}"
                );
            }
        }
    }

    #[test]
    fn state_partials_match_difference_quotients() {
        let model = all_terms_model();
        let parameters = model.own_parameters();
        let ctx = PropagationContext::default();
        for (gamma, inclination, u) in [(0.02, 0.96, 0.7), (-0.15, 1.9, 4.0), (0.0, 0.3, 2.4)] {
            let (state, _) = state_from_orbit_angles(26_560.0, 3.9, gamma, inclination, 0.5, u);
            let analytic = model
                .acceleration_partials(&parameters, &state, &ctx)
                .expect("partials");
            let numeric = difference_quotient_partials(&model, &parameters, &state, &ctx)
                .expect("difference quotients");
            // The coefficients are of order 1e-11 km/s^2, so the derivatives
            // are of order 1e-11 / 26560 km = 4e-16 1/s^2 and 1e-11 / 3.9 km/s
            // = 3e-12 1/s. A central difference with a relative step of 1e-6
            // has a truncation error of 1e-12 relative and a rounding error of
            // eps / 1e-6 = 2e-10 relative; 1e-8 relative covers both.
            let position_scale = analytic.wrt_position.norm();
            let velocity_scale = analytic.wrt_velocity.norm();
            assert!(position_scale > 1.0e-16 && velocity_scale > 1.0e-12);
            assert!(
                (analytic.wrt_position - numeric.wrt_position).norm() <= 1.0e-8 * position_scale,
                "d a / d r: {:?} vs {:?}",
                analytic.wrt_position,
                numeric.wrt_position
            );
            assert!(
                (analytic.wrt_velocity - numeric.wrt_velocity).norm() <= 1.0e-8 * velocity_scale,
                "d a / d v: {:?} vs {:?}",
                analytic.wrt_velocity,
                numeric.wrt_velocity
            );
        }
    }

    #[test]
    fn parameter_partials_are_the_acceleration_per_unit_coefficient() {
        let model = all_terms_model();
        let parameters = model.own_parameters();
        let (state, _) = state_from_orbit_angles(26_560.0, 3.9, 0.02, 0.96, 0.5, 0.7);
        let ctx = PropagationContext::default();
        let partials = model
            .parameter_partials(&parameters, &state, &ctx)
            .expect("parameter partials");
        assert_eq!(partials.len(), 9);
        assert_eq!(model.parameters().len(), 9);
        let base = model
            .acceleration_with_parameters(&parameters, &state, &ctx)
            .expect("acceleration");
        for (index, partial) in partials.iter().enumerate() {
            let mut shifted = parameters.clone();
            shifted[index] += 1.0e-9;
            let moved = model
                .acceleration_with_parameters(&shifted, &state, &ctx)
                .expect("acceleration");
            // The acceleration is linear in each coefficient, so the quotient
            // is exact up to rounding of the 1e-9 km/s^2 sum: 1e-6 relative.
            assert!(
                ((moved - base) / 1.0e-9 - partial).norm() <= 1.0e-6,
                "parameter {index}"
            );
        }
    }

    #[test]
    fn parameters_are_named_and_ordered_by_axis_then_harmonic() {
        let model = EmpiricalRtnAcceleration::new()
            .with_term(RtnAxis::CrossTrack, EmpiricalHarmonic::Sine, 3.0e-12)
            .and_then(|model| {
                model.with_term(RtnAxis::AlongTrack, EmpiricalHarmonic::Constant, 1.0e-12)
            })
            .expect("finite");
        let parameters = model.parameters();
        assert_eq!(
            parameters,
            vec![
                ForceParameter::new("empirical.along.constant", 1.0e-12),
                ForceParameter::new("empirical.cross.sin", 3.0e-12),
            ]
        );
        assert_eq!(model.parameter_count(), 2);
        assert_eq!(
            model.coefficient_km_s2(RtnAxis::Radial, EmpiricalHarmonic::Constant),
            None
        );
    }

    #[test]
    fn constant_terms_are_defined_on_an_equatorial_orbit_and_harmonics_are_not() {
        let state = CartesianState::new(0.0, [42_164.0, 0.0, 0.0], [0.0, 3.07, 0.0]);
        let ctx = PropagationContext::default();
        let constant = EmpiricalRtnAcceleration::constant_terms()
            .with_term(RtnAxis::AlongTrack, EmpiricalHarmonic::Constant, 1.0e-12)
            .expect("finite");
        let accel = constant.acceleration(&state, &ctx).expect("constant terms");
        assert_eq!(accel, Vector3::new(0.0, 1.0e-12, 0.0));

        let harmonic = EmpiricalRtnAcceleration::all_terms();
        assert!(matches!(
            harmonic.acceleration(&state, &ctx),
            Err(PropagationError::NumericalFailure(message)) if message.contains("equatorial")
        ));
    }

    #[test]
    fn invalid_input_is_a_typed_error() {
        let ctx = PropagationContext::default();
        let model = EmpiricalRtnAcceleration::constant_terms();
        assert!(EmpiricalRtnAcceleration::new()
            .with_term(RtnAxis::Radial, EmpiricalHarmonic::Constant, f64::NAN)
            .is_err());
        let state = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 1.0]);
        assert!(matches!(
            model.acceleration_with_parameters(&[0.0, 0.0], &state, &ctx),
            Err(PropagationError::InvalidInput(_))
        ));
        assert!(matches!(
            model.acceleration_with_parameters(&[0.0, f64::INFINITY, 0.0], &state, &ctx),
            Err(PropagationError::InvalidInput(_))
        ));
        let radial_fall = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(matches!(
            model.acceleration(&radial_fall, &ctx),
            Err(PropagationError::NumericalFailure(_))
        ));
        let at_origin = CartesianState::new(0.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
        assert!(matches!(
            model.acceleration(&at_origin, &ctx),
            Err(PropagationError::NumericalFailure(_))
        ));
        let empty = EmpiricalRtnAcceleration::new();
        assert_eq!(
            empty.acceleration(&at_origin, &ctx).expect("no terms"),
            Vector3::zeros()
        );
    }

    #[test]
    fn an_acceleration_that_overflows_is_a_numerical_failure() {
        // Two finite coefficients whose sum along the radial axis, 2e308,
        // overflows.
        let model = EmpiricalRtnAcceleration::all_terms();
        let ctx = PropagationContext::default();
        let state = CartesianState::new(0.0, [26_560.0, 0.0, 0.0], [0.0, 3.0, 3.0]);
        let mut parameters = [0.0; 9];
        parameters[0] = 1.0e308;
        parameters[1] = 1.0e308;
        let result = model.acceleration_with_parameters(&parameters, &state, &ctx);
        assert!(
            matches!(result, Err(PropagationError::NumericalFailure(_))),
            "{result:?}"
        );
    }
}
