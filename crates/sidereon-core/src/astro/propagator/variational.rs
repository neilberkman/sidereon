//! State transition and sensitivity matrices by variational equations.
//!
//! With the state `y = [r, v]` and force-model parameters `p`, the state
//! transition matrix `Phi(t) = d y(t) / d y(t0)` and the sensitivity matrix
//! `S(t) = d y(t) / d p` obey (Montenbruck & Gill, "Satellite Orbits",
//! section 7.2)
//!
//! ```text
//! d Phi / dt = A Phi,        Phi(t0) = I
//! d S / dt   = A S + B,      S(t0)   = 0
//!
//!     | 0        I       |        | 0       |
//! A = | da/dr    da/dv   |    B = | da/dp   |
//! ```
//!
//! where `a` is the acceleration. [`propagate_variational`] integrates these
//! equations together with the trajectory, taking `da/dr` and `da/dv` from
//! [`ForceModel::acceleration_partials`] and `da/dp` from
//! [`ForceModel::parameter_partials`]. No perturbed trajectory is propagated.
//!
//! # Integration
//!
//! The trajectory and the matrices advance with the same Runge-Kutta stages
//! and the same step. With [`IntegratorKind::Dp54`] the step is controlled by
//! the error estimate of the position and velocity alone, with the tolerances
//! and step limits [`StatePropagator`](crate::astro::propagator::StatePropagator)
//! uses, so the matrices do not change the steps the trajectory takes. Orekit
//! integrates its state transition and Jacobian columns the same way, as
//! secondary equations outside step control.
//!
//! # Time argument
//!
//! The equations of motion of an Earth satellite are integrated with
//! Terrestrial Time as the time argument (IERS Conventions 2010, chapter 10).
//! Each requested [`VariationalEpoch`] gives the TT elapsed since the initial
//! epoch, and the integration covers exactly that many SI seconds, carried as
//! seconds elapsed so the length of every step is exact. The force models
//! receive TDB, as [`ForceModel::acceleration`] specifies: at a requested
//! epoch its own TDB, and between two requested epochs the earlier one's TDB
//! advanced by the elapsed TT and by the change of TDB - TT
//! ([`tdb_minus_tt_seconds`]), kept within the two epochs' TDB. A force model
//! that reads Earth orientation therefore never receives an epoch outside the
//! TDB span of the initial and requested epochs.

use crate::astro::error::PropagationError;
use crate::astro::forces::ForceModel;
use crate::astro::integrators::tableau::DP54Tableau;
use crate::astro::propagator::api::{
    validate_adaptive_integrator_options, validate_integrator_epoch, validate_integrator_options,
    IntegratorOptions, PropagationContext,
};
use crate::astro::propagator::controller::PIController;
use crate::astro::propagator::numerical::IntegratorKind;
use crate::astro::state::CartesianState;
use crate::astro::time::scales::{tdb_minus_tt_seconds, tt_from_tdb_j2000_seconds};
use nalgebra::{Matrix6, Matrix6xX, Vector3};

/// Largest accepted difference, s, between a [`VariationalEpoch`]'s TDB
/// advance from the initial epoch and its elapsed TT once the change of
/// TDB - TT is removed, beyond the rounding of the TDB seconds. Epochs taken
/// from one instant on both scales differ by the rounding alone; TDB from
/// another realisation of TDB - TT differs by microseconds. A larger
/// difference means the two do not name the same instant.
const EPOCH_CONSISTENCY_S: f64 = 1.0e-3;

/// A force model with the parameter values to evaluate it at and the
/// parameters whose sensitivity is integrated.
#[derive(Clone, Copy)]
pub struct VariationalEquations<'a> {
    /// Force model supplying the acceleration and its partial derivatives.
    pub force: &'a dyn ForceModel,
    /// One value per entry of [`ForceModel::parameters`] of `force`, in the
    /// same order.
    pub parameters: &'a [f64],
    /// Indices into `parameters` of the parameters whose columns form
    /// [`VariationalState::sensitivity`], in column order.
    pub sensitivity_parameters: &'a [usize],
}

/// A requested epoch of [`propagate_variational`], on both time scales the
/// propagation uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VariationalEpoch {
    /// TDB seconds past J2000: the epoch of the returned state, and the
    /// epoch the force models receive there.
    pub tdb_j2000_s: f64,
    /// TT seconds elapsed since the initial epoch: how far the equations of
    /// motion are integrated.
    pub elapsed_tt_s: f64,
}

impl VariationalEpoch {
    /// The TDB epoch `tdb_j2000_s` of a propagation from
    /// `initial_tdb_j2000_s`, with the elapsed TT by
    /// [`tt_from_tdb_j2000_seconds`].
    ///
    /// Each conversion rounds to the nearest `f64` of TT seconds past J2000,
    /// about 0.1 microseconds in this century, so the elapsed TT carries up
    /// to two such roundings; the difference of the two instants' own TT
    /// seconds, where they are known, carries none on a regular grid.
    pub fn from_tdb_seconds(initial_tdb_j2000_s: f64, tdb_j2000_s: f64) -> Self {
        Self {
            tdb_j2000_s,
            elapsed_tt_s: tt_from_tdb_j2000_seconds(tdb_j2000_s)
                - tt_from_tdb_j2000_seconds(initial_tdb_j2000_s),
        }
    }
}

/// A propagated state with its partial derivatives.
#[derive(Debug, Clone, PartialEq)]
pub struct VariationalState {
    /// The propagated state, at the requested TDB epoch.
    pub state: CartesianState,
    /// `d [r, v](t) / d [r, v](t0)`: rows and columns in the order
    /// `[r_x, r_y, r_z, v_x, v_y, v_z]`, km and km/s.
    pub state_transition: Matrix6<f64>,
    /// `d [r, v](t) / d p`: one column per entry of
    /// [`VariationalEquations::sensitivity_parameters`], km and km/s per unit
    /// of the parameter.
    pub sensitivity: Matrix6xX<f64>,
}

/// Propagate a state, its state transition matrix and its parameter
/// sensitivities to each of `epochs`.
///
/// `initial.epoch_tdb_seconds` is the initial epoch in TDB. The state is
/// integrated for each epoch's [`VariationalEpoch::elapsed_tt_s`] (see the
/// module documentation, "Time argument") and returned at its
/// [`VariationalEpoch::tdb_j2000_s`]. The state is stepped from one
/// requested epoch to the next, as
/// [`StatePropagator::ephemeris_with_context`](crate::astro::propagator::StatePropagator::ephemeris_with_context)
/// does, and the matrices are accumulated from the initial epoch throughout:
/// every returned matrix is relative to `initial`. An epoch at the current
/// elapsed time returns the current values.
///
/// Force models that read Earth orientation record a departure from the UT1
/// table in `ctx`, as in any propagation.
///
/// # Errors
///
/// [`PropagationError::InvalidInput`] for a non-finite initial state, epoch
/// or parameter value, an epoch whose TDB and elapsed TT name different
/// instants, invalid integrator options, a parameter slice of the wrong
/// length or a sensitivity index outside it; the force model's own errors;
/// and [`PropagationError::NumericalFailure`] when a state, an intermediate
/// state of a step or a partial derivative is not finite.
pub fn propagate_variational(
    initial: CartesianState,
    epochs: &[VariationalEpoch],
    equations: &VariationalEquations<'_>,
    integrator: IntegratorKind,
    options: &IntegratorOptions,
    ctx: &PropagationContext,
) -> Result<Vec<VariationalState>, PropagationError> {
    let columns = 6 + equations.sensitivity_parameters.len();
    let mut states = Vec::with_capacity(epochs.len());
    propagate(
        initial,
        epochs,
        equations,
        integrator,
        options,
        ctx,
        columns,
        |augmented, epoch| {
            states.push(augmented.to_variational_state(epoch)?);
            Ok(())
        },
    )?;
    Ok(states)
}

/// The trajectory of [`propagate_variational`] without its matrices: the
/// same steps and the same states, at the cost of the trajectory alone.
pub(crate) fn propagate_trajectory(
    initial: CartesianState,
    epochs: &[VariationalEpoch],
    equations: &VariationalEquations<'_>,
    integrator: IntegratorKind,
    options: &IntegratorOptions,
    ctx: &PropagationContext,
) -> Result<Vec<CartesianState>, PropagationError> {
    let mut states = Vec::with_capacity(epochs.len());
    propagate(
        initial,
        epochs,
        equations,
        integrator,
        options,
        ctx,
        0,
        |augmented, epoch| {
            states.push(augmented.to_state(epoch)?);
            Ok(())
        },
    )?;
    Ok(states)
}

/// Validate the inputs, then step through `epochs` with `columns` matrix
/// columns (zero for the trajectory alone), handing each requested state to
/// `emit`.
#[allow(clippy::too_many_arguments)]
fn propagate(
    initial: CartesianState,
    epochs: &[VariationalEpoch],
    equations: &VariationalEquations<'_>,
    integrator: IntegratorKind,
    options: &IntegratorOptions,
    ctx: &PropagationContext,
    columns: usize,
    mut emit: impl FnMut(&Augmented, f64) -> Result<(), PropagationError>,
) -> Result<(), PropagationError> {
    match integrator {
        IntegratorKind::Rk4 => validate_integrator_options(options)?,
        IntegratorKind::Dp54 => validate_adaptive_integrator_options(options)?,
    }
    validate_integrator_epoch(initial.epoch_tdb_seconds, "initial.epoch_tdb_seconds")?;
    for value in initial
        .position_km
        .iter()
        .chain(initial.velocity_km_s.iter())
    {
        if !value.is_finite() {
            return Err(PropagationError::InvalidInput(
                "initial state not finite".to_string(),
            ));
        }
    }
    let parameter_count = equations.force.parameter_count();
    if equations.parameters.len() != parameter_count {
        return Err(PropagationError::InvalidInput(format!(
            "force model takes {parameter_count} parameter values, got {}",
            equations.parameters.len()
        )));
    }
    if !equations.parameters.iter().all(|value| value.is_finite()) {
        return Err(PropagationError::InvalidInput(
            "force-model parameter values must be finite".to_string(),
        ));
    }
    if let Some(&index) = equations
        .sensitivity_parameters
        .iter()
        .find(|&&index| index >= parameter_count)
    {
        return Err(PropagationError::InvalidInput(format!(
            "sensitivity parameter index {index} is outside the {parameter_count} parameters"
        )));
    }
    // TT of the initial epoch, as the argument of TDB - TT only: an error of
    // a few rounding units moves TDB - TT by less than 1e-16 s.
    let tt_reference = tt_from_tdb_j2000_seconds(initial.epoch_tdb_seconds);
    let start = Anchor {
        tdb: initial.epoch_tdb_seconds,
        elapsed: 0.0,
    };
    for epoch in epochs {
        validate_integrator_epoch(epoch.tdb_j2000_s, "epochs.tdb_j2000_s")?;
        validate_integrator_epoch(epoch.elapsed_tt_s, "epochs.elapsed_tt_s")?;
        let anchor = Anchor {
            tdb: epoch.tdb_j2000_s,
            elapsed: epoch.elapsed_tt_s,
        };
        let advance = epoch.tdb_j2000_s - initial.epoch_tdb_seconds;
        let mismatch = advance
            - epoch.elapsed_tt_s
            - tdb_minus_tt_change(tt_reference, start.elapsed, anchor.elapsed);
        let rounding =
            4.0 * f64::EPSILON * initial.epoch_tdb_seconds.abs().max(epoch.tdb_j2000_s.abs());
        if !mismatch.is_finite() || mismatch.abs() > EPOCH_CONSISTENCY_S + rounding {
            return Err(PropagationError::InvalidInput(format!(
                "epoch at TDB {} s and elapsed TT {} s do not name one instant",
                epoch.tdb_j2000_s, epoch.elapsed_tt_s
            )));
        }
    }

    let system = System {
        equations,
        tt_reference,
        ctx,
    };
    let mut current = Augmented::initial(&initial, columns);
    let mut from = start;
    for epoch in epochs {
        let to = Anchor {
            tdb: epoch.tdb_j2000_s,
            elapsed: epoch.elapsed_tt_s,
        };
        if to.elapsed != from.elapsed {
            let segment = Segment { from, to };
            current = match integrator {
                IntegratorKind::Rk4 => system.rk4(current, &segment, options)?,
                IntegratorKind::Dp54 => system.dp54(current, &segment, options)?,
            };
        }
        emit(&current, epoch.tdb_j2000_s)?;
        from = to;
    }
    Ok(())
}

/// An epoch on both scales: TDB seconds past J2000 and TT seconds elapsed.
#[derive(Clone, Copy)]
struct Anchor {
    tdb: f64,
    elapsed: f64,
}

/// The change of TDB - TT from `from` to `to` TT seconds elapsed after the
/// TT instant `tt_reference`.
fn tdb_minus_tt_change(tt_reference: f64, from: f64, to: f64) -> f64 {
    tdb_minus_tt_seconds(tt_reference + to) - tdb_minus_tt_seconds(tt_reference + from)
}

/// One integration from a requested epoch (or the initial one) to the next.
struct Segment {
    from: Anchor,
    to: Anchor,
}

impl Segment {
    fn start(&self) -> f64 {
        self.from.elapsed
    }

    fn target(&self) -> f64 {
        self.to.elapsed
    }
}

/// The trajectory with the columns of `[Phi | S]`.
#[derive(Clone)]
struct Augmented {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    /// Six columns of `Phi` followed by the columns of `S`, each
    /// `[d r, d v]`; none for the trajectory alone.
    columns: Vec<[f64; 6]>,
}

/// Time derivative of an [`Augmented`].
struct Rate {
    position: Vector3<f64>,
    velocity: Vector3<f64>,
    columns: Vec<[f64; 6]>,
}

impl Augmented {
    fn initial(state: &CartesianState, columns: usize) -> Self {
        let mut matrix = vec![[0.0; 6]; columns];
        for (index, column) in matrix.iter_mut().take(6).enumerate() {
            column[index] = 1.0;
        }
        Self {
            position: state.position_km,
            velocity: state.velocity_km_s,
            columns: matrix,
        }
    }

    /// `self + h * sum(weight_i * rate_i)`, summed in the order given.
    fn advanced(&self, rates: &[(&Rate, f64)], h: f64) -> Self {
        let mut position = Vector3::zeros();
        let mut velocity = Vector3::zeros();
        let mut columns = vec![[0.0; 6]; self.columns.len()];
        for &(rate, weight) in rates {
            position += rate.position * weight;
            velocity += rate.velocity * weight;
            for (sum, column) in columns.iter_mut().zip(&rate.columns) {
                for (sum, value) in sum.iter_mut().zip(column) {
                    *sum += value * weight;
                }
            }
        }
        for (sum, base) in columns.iter_mut().zip(&self.columns) {
            for (sum, base) in sum.iter_mut().zip(base) {
                *sum = base + *sum * h;
            }
        }
        Self {
            position: self.position + position * h,
            velocity: self.velocity + velocity * h,
            columns,
        }
    }

    fn trajectory_finite(&self) -> bool {
        self.position
            .iter()
            .chain(self.velocity.iter())
            .all(|value| value.is_finite())
    }

    fn to_state(&self, epoch: f64) -> Result<CartesianState, PropagationError> {
        if !self.trajectory_finite() {
            return Err(PropagationError::NumericalFailure(
                "propagated state not finite".to_string(),
            ));
        }
        Ok(CartesianState {
            epoch_tdb_seconds: epoch,
            position_km: self.position,
            velocity_km_s: self.velocity,
        })
    }

    fn to_variational_state(&self, epoch: f64) -> Result<VariationalState, PropagationError> {
        let finite = self.trajectory_finite()
            && self.columns.iter().flatten().all(|value| value.is_finite());
        if !finite {
            return Err(PropagationError::NumericalFailure(
                "variational state not finite".to_string(),
            ));
        }
        let state_transition = Matrix6::from_fn(|row, column| self.columns[column][row]);
        let sensitivity = Matrix6xX::from_fn(self.columns.len() - 6, |row, column| {
            self.columns[6 + column][row]
        });
        Ok(VariationalState {
            state: CartesianState {
                epoch_tdb_seconds: epoch,
                position_km: self.position,
                velocity_km_s: self.velocity,
            },
            state_transition,
            sensitivity,
        })
    }
}

struct System<'a> {
    equations: &'a VariationalEquations<'a>,
    /// TT of the initial epoch, the argument of TDB - TT.
    tt_reference: f64,
    ctx: &'a PropagationContext,
}

impl System<'_> {
    /// The TDB epoch the force models receive `elapsed` TT seconds into the
    /// propagation, within `segment` (see the module documentation).
    fn tdb_at(&self, segment: &Segment, elapsed: f64) -> f64 {
        let Segment { from, to } = segment;
        if elapsed == to.elapsed {
            return to.tdb;
        }
        if elapsed == from.elapsed {
            return from.tdb;
        }
        let tdb = from.tdb
            + ((elapsed - from.elapsed)
                + tdb_minus_tt_change(self.tt_reference, from.elapsed, elapsed));
        tdb.max(from.tdb.min(to.tdb)).min(from.tdb.max(to.tdb))
    }

    fn rate(
        &self,
        segment: &Segment,
        elapsed: f64,
        y: &Augmented,
    ) -> Result<Rate, PropagationError> {
        if !y.trajectory_finite() {
            return Err(PropagationError::NumericalFailure(
                "state not finite within a step".to_string(),
            ));
        }
        let state = CartesianState {
            epoch_tdb_seconds: self.tdb_at(segment, elapsed),
            position_km: y.position,
            velocity_km_s: y.velocity,
        };
        let VariationalEquations {
            force,
            parameters,
            sensitivity_parameters,
        } = *self.equations;
        let acceleration = force.acceleration_with_parameters(parameters, &state, self.ctx)?;
        if y.columns.is_empty() {
            return Ok(Rate {
                position: y.velocity,
                velocity: acceleration,
                columns: Vec::new(),
            });
        }
        let partials = force.acceleration_partials(parameters, &state, self.ctx)?;
        let parameter_partials = if sensitivity_parameters.is_empty() {
            Vec::new()
        } else {
            let all = force.parameter_partials(parameters, &state, self.ctx)?;
            if all.len() != parameters.len() {
                return Err(PropagationError::InvalidInput(format!(
                    "force model returned {} parameter partials for {} parameters",
                    all.len(),
                    parameters.len()
                )));
            }
            sensitivity_parameters
                .iter()
                .map(|&index| all[index])
                .collect()
        };

        let mut columns = Vec::with_capacity(y.columns.len());
        for (index, column) in y.columns.iter().enumerate() {
            let d_position = Vector3::new(column[0], column[1], column[2]);
            let d_velocity = Vector3::new(column[3], column[4], column[5]);
            let mut d_acceleration =
                partials.wrt_position * d_position + partials.wrt_velocity * d_velocity;
            if index >= 6 {
                d_acceleration += parameter_partials[index - 6];
            }
            columns.push([
                d_velocity.x,
                d_velocity.y,
                d_velocity.z,
                d_acceleration.x,
                d_acceleration.y,
                d_acceleration.z,
            ]);
        }
        Ok(Rate {
            position: y.velocity,
            velocity: acceleration,
            columns,
        })
    }

    /// Classical fourth-order Runge-Kutta across `segment`, in steps of
    /// [`IntegratorOptions::initial_step`].
    fn rk4(
        &self,
        mut y: Augmented,
        segment: &Segment,
        options: &IntegratorOptions,
    ) -> Result<Augmented, PropagationError> {
        let (start, target) = (segment.start(), segment.target());
        let span = target - start;
        let span_abs = span.abs();
        let mut elapsed = start;
        let mut h = options.initial_step.min(span_abs) * span.signum();
        let mut steps = 0_u32;
        while (elapsed - start).abs() < span_abs {
            if steps >= options.max_steps {
                return Err(PropagationError::MaxStepsExceeded);
            }
            let last = (elapsed + h - start).abs() >= span_abs;
            if last {
                h = target - elapsed;
            }
            let next_elapsed = if last { target } else { elapsed + h };
            let k1 = self.rate(segment, elapsed, &y)?;
            let k2 = self.rate(
                segment,
                elapsed + h / 2.0,
                &y.advanced(&[(&k1, 1.0)], h / 2.0),
            )?;
            let k3 = self.rate(
                segment,
                elapsed + h / 2.0,
                &y.advanced(&[(&k2, 1.0)], h / 2.0),
            )?;
            let k4 = self.rate(segment, next_elapsed, &y.advanced(&[(&k3, 1.0)], h))?;
            y = y.advanced(&[(&k1, 1.0), (&k2, 2.0), (&k3, 2.0), (&k4, 1.0)], h / 6.0);
            elapsed = next_elapsed;
            steps += 1;
        }
        Ok(y)
    }

    /// Dormand-Prince 5(4) across `segment`, with the step controlled by the
    /// position and velocity error estimates.
    fn dp54(
        &self,
        mut y: Augmented,
        segment: &Segment,
        options: &IntegratorOptions,
    ) -> Result<Augmented, PropagationError> {
        let tableau = DP54Tableau::default();
        let controller = PIController {
            order: 5.0,
            ..PIController::default()
        };
        let (start, target) = (segment.start(), segment.target());
        let span = target - start;
        let span_abs = span.abs();
        let mut elapsed = start;
        let mut h = crate::validate::clamp_magnitude(
            options.initial_step.min(span_abs) * span.signum(),
            options.max_step,
        );
        let mut steps = 0_u32;
        // First-same-as-last: the rate at the end of an accepted step opens
        // the next one.
        let mut k1 = self.rate(segment, elapsed, &y)?;

        while (elapsed - start).abs() < span_abs {
            if steps >= options.max_steps {
                return Err(PropagationError::MaxStepsExceeded);
            }
            let last = (elapsed + h - start).abs() >= span_abs;
            let h_step = if last { target - elapsed } else { h };

            let mut stages: Vec<Rate> = Vec::with_capacity(7);
            stages.push(k1);
            for stage in 1..6 {
                let weighted: Vec<(&Rate, f64)> = stages
                    .iter()
                    .zip(&tableau.a[stage])
                    .map(|(rate, &weight)| (rate, weight))
                    .collect();
                let stage_state = y.advanced(&weighted, h_step);
                let rate = self.rate(segment, elapsed + h_step * tableau.c[stage], &stage_state)?;
                stages.push(rate);
            }
            let fifth: Vec<(&Rate, f64)> = stages
                .iter()
                .zip(&tableau.b5)
                .map(|(rate, &weight)| (rate, weight))
                .collect();
            let next = y.advanced(&fifth, h_step);
            let next_elapsed = if last { target } else { elapsed + h_step };
            let k_next = self.rate(segment, next_elapsed, &next)?;
            stages.push(k_next);

            // Embedded fourth-order solution, for the position and velocity
            // error estimate only.
            let mut position_error = Vector3::zeros();
            let mut velocity_error = Vector3::zeros();
            for (index, rate) in stages.iter().enumerate() {
                let weight = if index < 6 { tableau.b5[index] } else { 0.0 } - tableau.b4[index];
                position_error += rate.position * weight;
                velocity_error += rate.velocity * weight;
            }
            let position_scale =
                options.abs_tol + y.position.norm().max(next.position.norm()) * options.rel_tol;
            let velocity_scale =
                options.abs_tol + y.velocity.norm().max(next.velocity.norm()) * options.rel_tol;
            let error = ((position_error * h_step).norm() / position_scale)
                .max((velocity_error * h_step).norm() / velocity_scale);
            if !error.is_finite() {
                return Err(PropagationError::NumericalFailure(
                    "variational step error not finite".to_string(),
                ));
            }

            steps += 1;
            h = crate::validate::clamp_magnitude(
                controller.next_step(h_step, error),
                options.max_step,
            );
            let mut stages = stages.into_iter();
            if error <= 1.0 {
                y = next;
                elapsed = next_elapsed;
                k1 = stages.next_back().ok_or_else(missing_stage)?;
            } else {
                if h.abs() < options.min_step {
                    return Err(PropagationError::NumericalFailure(
                        "Step size too small".to_string(),
                    ));
                }
                k1 = stages.next().ok_or_else(missing_stage)?;
            }
        }
        Ok(y)
    }
}

fn missing_stage() -> PropagationError {
    PropagationError::NumericalFailure("Runge-Kutta stage missing".to_string())
}

#[cfg(test)]
mod tests {
    //! The state transition and sensitivity matrices are checked against
    //! central finite differences of trajectories propagated by the plain
    //! [`StatePropagator`], which shares no partial-derivative code with the
    //! variational equations.

    use super::*;
    use crate::astro::constants::MU_EARTH;
    use crate::astro::forces::{
        CompositeForceModel, EmpiricalHarmonic, EmpiricalRtnAcceleration, RtnAxis,
        SolarRadiationPressure, TwoBodyGravity,
    };
    use crate::astro::propagator::numerical::{
        ForceModelComponents, ForceModelKind, StatePropagator,
    };

    /// 2026-05-13 00:00 TDB, in seconds past J2000.
    const EPOCH: f64 = 831_902_400.0;

    /// TDB epochs as propagation epochs from [`EPOCH`].
    fn from_epoch(epochs: &[f64]) -> Vec<VariationalEpoch> {
        epochs
            .iter()
            .map(|&epoch| VariationalEpoch::from_tdb_seconds(EPOCH, epoch))
            .collect()
    }

    fn tight_options() -> IntegratorOptions {
        IntegratorOptions {
            abs_tol: 1.0e-13,
            rel_tol: 1.0e-13,
            initial_step: 30.0,
            max_step: 120.0,
            ..IntegratorOptions::default()
        }
    }

    /// A GPS-like state: 26560 km, 55 degrees inclination, slightly eccentric.
    fn gnss_state() -> CartesianState {
        let radius = 26_560.0_f64;
        let speed = (MU_EARTH / radius).sqrt() * 1.003;
        let (sin_i, cos_i) = libm::sincos(55.0_f64.to_radians());
        CartesianState::new(
            EPOCH,
            [radius * 0.6, radius * 0.8, 0.0],
            [
                -speed * 0.8 * cos_i + 0.01,
                speed * 0.6 * cos_i,
                speed * sin_i,
            ],
        )
    }

    fn empirical_model() -> EmpiricalRtnAcceleration {
        let mut model = EmpiricalRtnAcceleration::new();
        let mut value = 2.0e-12;
        for axis in RtnAxis::ALL {
            for harmonic in EmpiricalHarmonic::ALL {
                model = model.with_term(axis, harmonic, value).expect("finite");
                value *= -1.3;
            }
        }
        model
    }

    /// The Phase A force set with solar radiation pressure, built by the
    /// plain propagator, plus the empirical accelerations.
    fn force(cr: f64, empirical: EmpiricalRtnAcceleration) -> CompositeForceModel {
        let srp = SolarRadiationPressure::new(cr, 0.02).expect("valid SRP");
        let base = StatePropagator {
            initial: gnss_state(),
            force_model: ForceModelKind::composite(ForceModelComponents::earth_phase_a(Some(srp))),
            integrator: IntegratorKind::Dp54,
            options: tight_options(),
            drag: None,
            space_weather: None,
        }
        .build_force()
        .expect("force");
        let mut composite = CompositeForceModel::new();
        composite.add(base);
        composite.add(Box::new(empirical));
        composite
    }

    /// The position and velocity at `epoch`, propagated without variational
    /// equations, for a perturbed initial state or parameter vector.
    fn plain_final(
        initial: CartesianState,
        force: &CompositeForceModel,
        parameters: &[f64],
        epoch: f64,
    ) -> [f64; 6] {
        let equations = VariationalEquations {
            force,
            parameters,
            sensitivity_parameters: &[],
        };
        // Only the trajectory is read; the state transition matrix that comes
        // with it is not.
        let state = propagate_variational(
            initial,
            &from_epoch(&[epoch]),
            &equations,
            IntegratorKind::Dp54,
            &tight_options(),
            &PropagationContext::default(),
        )
        .expect("propagation")[0]
            .state;
        [
            state.position_km.x,
            state.position_km.y,
            state.position_km.z,
            state.velocity_km_s.x,
            state.velocity_km_s.y,
            state.velocity_km_s.z,
        ]
    }

    fn perturbed(state: CartesianState, component: usize, delta: f64) -> CartesianState {
        let mut state = state;
        if component < 3 {
            state.position_km[component] += delta;
        } else {
            state.velocity_km_s[component - 3] += delta;
        }
        state
    }

    #[test]
    fn trajectory_matches_the_plain_propagator() {
        let srp = SolarRadiationPressure::new(1.3, 0.02).expect("valid SRP");
        let kind = ForceModelKind::composite(ForceModelComponents::earth_phase_a(Some(srp)));
        let initial = gnss_state();
        let tdb: Vec<f64> = (0..=8)
            .map(|step| EPOCH + 900.0 * f64::from(step))
            .collect();
        let epochs = from_epoch(&tdb);
        // The plain propagator's time argument is TDB itself: it is asked for
        // the epochs that span the same elapsed time. The force models then
        // see epochs that differ by the change of TDB - TT, under 2
        // microseconds over these two hours, which moves the Sun and Moon by
        // centimetres.
        let plain_epochs: Vec<f64> = epochs
            .iter()
            .map(|epoch| EPOCH + epoch.elapsed_tt_s)
            .collect();
        let plain = StatePropagator {
            initial,
            force_model: kind,
            integrator: IntegratorKind::Dp54,
            options: tight_options(),
            drag: None,
            space_weather: None,
        };
        let expected = plain.ephemeris(&plain_epochs).expect("plain ephemeris");
        let force = plain.build_force().expect("force");
        let parameters: Vec<f64> = force.parameters().iter().map(|p| p.value).collect();
        let equations = VariationalEquations {
            force: force.as_ref(),
            parameters: &parameters,
            sensitivity_parameters: &[],
        };
        let states = propagate_variational(
            initial,
            &epochs,
            &equations,
            IntegratorKind::Dp54,
            &tight_options(),
            &PropagationContext::default(),
        )
        .expect("variational propagation");
        assert_eq!(states[0].state, initial);
        assert_eq!(states[0].state_transition, Matrix6::identity());
        for ((state, expected), epoch) in states.iter().zip(&expected).zip(&tdb) {
            assert_eq!(state.state.epoch_tdb_seconds, *epoch);
            // Both integrate the same equations with the same tableau and
            // tolerances. The plain propagator advances its epoch as a f64 of
            // TDB seconds past J2000, which rounds each step's end to 2^-23 s
            // at this epoch, while the variational run keeps elapsed seconds;
            // 60 or so steps of at most 6e-8 s each at 3.9 km/s bound the
            // difference by 1.4e-5 km.
            assert!(
                (state.state.position_km - expected.position_km).norm() <= 1.4e-5,
                "position differs by {} km",
                (state.state.position_km - expected.position_km).norm()
            );
        }
    }

    #[test]
    fn state_transition_matrix_matches_central_differences() {
        let force = force(1.3, empirical_model());
        let parameters: Vec<f64> = force.parameters().iter().map(|p| p.value).collect();
        let initial = gnss_state();
        // Half a revolution.
        let epoch = EPOCH + 21_600.0;
        let equations = VariationalEquations {
            force: &force,
            parameters: &parameters,
            sensitivity_parameters: &[],
        };
        let result = propagate_variational(
            initial,
            &from_epoch(&[epoch]),
            &equations,
            IntegratorKind::Dp54,
            &tight_options(),
            &PropagationContext::default(),
        )
        .expect("variational propagation");
        let stm = result[0].state_transition;

        for column in 0..6 {
            // 1 km and 1e-4 km/s: the second-order term of a central
            // difference is of relative size (delta / scale)^2 = 1.4e-9 for
            // position and 6.6e-10 for velocity.
            let delta = if column < 3 { 1.0 } else { 1.0e-4 };
            let plus = plain_final(
                perturbed(initial, column, delta),
                &force,
                &parameters,
                epoch,
            );
            let minus = plain_final(
                perturbed(initial, column, -delta),
                &force,
                &parameters,
                epoch,
            );
            let scale = (0..6)
                .map(|row| stm[(row, column)].abs())
                .fold(0.0_f64, f64::max);
            for row in 0..6 {
                let numeric = (plus[row] - minus[row]) / (2.0 * delta);
                // Truncation of the difference quotient as above, plus the
                // integration error of the two perturbed trajectories
                // (tolerance 1e-13 relative per step, a few hundred steps)
                // divided by the perturbation; 2e-7 of the column's largest
                // entry covers both with the propagated rounding error.
                assert!(
                    (stm[(row, column)] - numeric).abs() <= 2.0e-7 * scale,
                    "Phi[{row}][{column}] = {} vs {numeric}",
                    stm[(row, column)]
                );
            }
        }
    }

    #[test]
    fn sensitivity_matrix_matches_central_differences() {
        let force = force(1.3, empirical_model());
        let listed = force.parameters();
        let parameters: Vec<f64> = listed.iter().map(|p| p.value).collect();
        assert_eq!(listed.len(), 10);
        assert_eq!(listed[0].name, SolarRadiationPressure::CR_PARAMETER);
        let initial = gnss_state();
        let epoch = EPOCH + 21_600.0;
        let all: Vec<usize> = (0..parameters.len()).collect();
        let equations = VariationalEquations {
            force: &force,
            parameters: &parameters,
            sensitivity_parameters: &all,
        };
        let result = propagate_variational(
            initial,
            &from_epoch(&[epoch]),
            &equations,
            IntegratorKind::Dp54,
            &tight_options(),
            &PropagationContext::default(),
        )
        .expect("variational propagation");
        let sensitivity = &result[0].sensitivity;
        assert_eq!(sensitivity.ncols(), parameters.len());

        for (column, name) in listed.iter().map(|p| &p.name).enumerate() {
            // The trajectory is close to linear in every parameter, so a
            // large step keeps the quotient above the integration noise:
            // 0.1 in C_R moves the satellite by metres, 1e-10 km/s^2 by
            // tens of metres.
            let delta = if column == 0 { 0.1 } else { 1.0e-10 };
            let mut plus = parameters.clone();
            let mut minus = parameters.clone();
            plus[column] += delta;
            minus[column] -= delta;
            let plus = plain_final(initial, &force, &plus, epoch);
            let minus = plain_final(initial, &force, &minus, epoch);
            let scale = (0..6)
                .map(|row| sensitivity[(row, column)].abs())
                .fold(0.0_f64, f64::max);
            assert!(scale > 0.0, "{name} has no effect");
            for row in 0..6 {
                let numeric = (plus[row] - minus[row]) / (2.0 * delta);
                // The perturbed trajectories carry an integration and
                // rounding error of about 1e-9 km over half a revolution; the
                // quotient divides it by a displacement of 1e-3 km or more,
                // so the entries agree to about 1e-6 of the column's largest.
                assert!(
                    (sensitivity[(row, column)] - numeric).abs() <= 2.0e-6 * scale,
                    "S[{row}][{name}] = {} vs {numeric}",
                    sensitivity[(row, column)]
                );
            }
        }
    }

    #[test]
    fn two_body_state_transition_is_symplectic_and_rk4_agrees() {
        let force = TwoBodyGravity::default();
        let initial = gnss_state();
        let equations = VariationalEquations {
            force: &force,
            parameters: &[],
            sensitivity_parameters: &[],
        };
        let epoch = EPOCH + 7_200.0;
        let ctx = PropagationContext::default();
        let dp54 = propagate_variational(
            initial,
            &from_epoch(&[epoch]),
            &equations,
            IntegratorKind::Dp54,
            &tight_options(),
            &ctx,
        )
        .expect("DP54")[0]
            .clone();
        let rk4_options = IntegratorOptions {
            initial_step: 10.0,
            ..IntegratorOptions::default()
        };
        let rk4 = propagate_variational(
            initial,
            &from_epoch(&[epoch]),
            &equations,
            IntegratorKind::Rk4,
            &rk4_options,
            &ctx,
        )
        .expect("RK4")[0]
            .clone();

        // A conservative force has a symplectic state transition matrix, so
        // its determinant is 1 (Liouville's theorem). The
        // entries reach 1e4, so the determinant's rounding error is far below
        // the 1e-6 allowed here.
        let stm = dp54.state_transition;
        let determinant = crate::astro::math::portable::matrix_from_f64(
            &nalgebra::DMatrix::from_fn(6, 6, |row, column| stm[(row, column)]),
        )
        .determinant()
        .0;
        assert!(
            (determinant - 1.0).abs() <= 1.0e-6,
            "determinant {determinant}"
        );

        // RK4 at a 10 s step has a local error of about (n h)^5 = 1e-15 per
        // step for the mean motion n = 1.5e-4 rad/s and takes 720 steps; the
        // two integrators agree to 1e-9 of the largest entry.
        let scale = stm.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        assert!((rk4.state_transition - stm)
            .iter()
            .all(|d| d.abs() <= 1.0e-9 * scale));
        assert!((rk4.state.position_km - dp54.state.position_km).norm() <= 1.0e-7);
    }

    #[test]
    fn invalid_input_is_a_typed_error() {
        let force = force(1.3, EmpiricalRtnAcceleration::new());
        let parameters = [1.3];
        let ctx = PropagationContext::default();
        let run = |initial: CartesianState,
                   epochs: &[f64],
                   parameters: &[f64],
                   selected: &[usize],
                   options: &IntegratorOptions| {
            propagate_variational(
                initial,
                &from_epoch(epochs),
                &VariationalEquations {
                    force: &force,
                    parameters,
                    sensitivity_parameters: selected,
                },
                IntegratorKind::Dp54,
                options,
                &ctx,
            )
        };
        let initial = gnss_state();
        let options = tight_options();
        let invalid = |result: Result<Vec<VariationalState>, PropagationError>| {
            matches!(result, Err(PropagationError::InvalidInput(_)))
        };
        assert!(invalid(run(initial, &[EPOCH + 60.0], &[], &[], &options)));
        assert!(invalid(run(
            initial,
            &[EPOCH + 60.0],
            &parameters,
            &[1],
            &options
        )));
        assert!(invalid(run(
            initial,
            &[f64::NAN],
            &parameters,
            &[],
            &options
        )));
        let mut bad = initial;
        bad.position_km.x = f64::INFINITY;
        assert!(invalid(run(
            bad,
            &[EPOCH + 60.0],
            &parameters,
            &[],
            &options
        )));
        let bad_options = IntegratorOptions {
            rel_tol: 0.0,
            ..tight_options()
        };
        assert!(invalid(run(
            initial,
            &[EPOCH + 60.0],
            &parameters,
            &[],
            &bad_options
        )));
        assert!(invalid(run(
            initial,
            &[EPOCH + 60.0],
            &[f64::NAN],
            &[],
            &options
        )));
        let few_steps = IntegratorOptions {
            max_steps: 2,
            ..tight_options()
        };
        assert!(matches!(
            run(initial, &[EPOCH + 86_400.0], &parameters, &[0], &few_steps),
            Err(PropagationError::MaxStepsExceeded)
        ));
        // No epochs: nothing to do.
        assert!(run(initial, &[], &parameters, &[0], &options)
            .expect("empty")
            .is_empty());
        // A TDB epoch and an elapsed TT that are a second apart.
        let mismatched = VariationalEpoch {
            tdb_j2000_s: EPOCH + 60.0,
            elapsed_tt_s: 61.0,
        };
        assert!(invalid(propagate_variational(
            initial,
            &[mismatched],
            &VariationalEquations {
                force: &force,
                parameters: &parameters,
                sensitivity_parameters: &[],
            },
            IntegratorKind::Dp54,
            &options,
            &ctx,
        )));
        let non_finite = VariationalEpoch {
            tdb_j2000_s: EPOCH + 60.0,
            elapsed_tt_s: f64::NAN,
        };
        assert!(invalid(propagate_variational(
            initial,
            &[non_finite],
            &VariationalEquations {
                force: &force,
                parameters: &parameters,
                sensitivity_parameters: &[],
            },
            IntegratorKind::Dp54,
            &options,
            &ctx,
        )));
    }

    #[test]
    fn non_finite_parameters_are_refused_without_a_step() {
        let force = force(1.3, EmpiricalRtnAcceleration::new());
        let ctx = PropagationContext::default();
        for epochs in [[EPOCH].as_slice(), &[]] {
            let result = propagate_variational(
                gnss_state(),
                &from_epoch(epochs),
                &VariationalEquations {
                    force: &force,
                    parameters: &[f64::NAN],
                    sensitivity_parameters: &[],
                },
                IntegratorKind::Dp54,
                &tight_options(),
                &ctx,
            );
            assert!(
                matches!(result, Err(PropagationError::InvalidInput(_))),
                "{result:?}"
            );
        }
    }

    #[test]
    fn a_state_that_overflows_within_a_step_is_a_numerical_failure() {
        // A finite initial state whose first half step reaches 5e309 km.
        let force = TwoBodyGravity::default();
        let initial = CartesianState::new(EPOCH, [1.0e100, 0.0, 0.0], [1.0e305, 0.0, 0.0]);
        let options = IntegratorOptions {
            initial_step: 1.0e5,
            ..IntegratorOptions::default()
        };
        let result = propagate_variational(
            initial,
            &from_epoch(&[EPOCH + 1.0e5]),
            &VariationalEquations {
                force: &force,
                parameters: &[],
                sensitivity_parameters: &[],
            },
            IntegratorKind::Rk4,
            &options,
            &PropagationContext::default(),
        );
        assert!(
            matches!(result, Err(PropagationError::NumericalFailure(_))),
            "{result:?}"
        );
    }

    /// A force without effect that records the epoch of every evaluation.
    #[derive(Default)]
    struct EpochRecorder(std::sync::Mutex<Vec<f64>>);

    impl ForceModel for EpochRecorder {
        fn acceleration(
            &self,
            state: &CartesianState,
            _ctx: &PropagationContext,
        ) -> Result<Vector3<f64>, PropagationError> {
            self.0
                .lock()
                .expect("recorder")
                .push(state.epoch_tdb_seconds);
            Ok(Vector3::zeros())
        }

        fn acceleration_partials(
            &self,
            _parameters: &[f64],
            _state: &CartesianState,
            _ctx: &PropagationContext,
        ) -> Result<crate::astro::forces::AccelerationPartials, PropagationError> {
            Ok(crate::astro::forces::AccelerationPartials::zeros())
        }
    }

    #[test]
    fn the_force_models_receive_each_requested_tdb_and_nothing_outside_the_span() {
        let recorder = std::sync::Arc::new(EpochRecorder::default());
        let mut composite = CompositeForceModel::new();
        composite.add(Box::new(TwoBodyGravity::default()));
        composite.add(Box::new(std::sync::Arc::clone(&recorder)));
        let tdb: Vec<f64> = (1..=6)
            .map(|step| EPOCH + 600.0 * f64::from(step))
            .collect();
        let mut epochs = from_epoch(&tdb);
        // A label one unit in the last place off the series is used as given.
        epochs[2].tdb_j2000_s = epochs[2].tdb_j2000_s.next_up();
        propagate_variational(
            gnss_state(),
            &epochs,
            &VariationalEquations {
                force: &composite,
                parameters: &[],
                sensitivity_parameters: &[],
            },
            IntegratorKind::Dp54,
            &tight_options(),
            &PropagationContext::default(),
        )
        .expect("propagation");
        let recorded = recorder.0.lock().expect("recorder");
        for epoch in &epochs {
            assert!(recorded.contains(&epoch.tdb_j2000_s), "{epoch:?}");
        }
        let last = epochs[epochs.len() - 1].tdb_j2000_s;
        assert!(recorded
            .iter()
            .all(|&epoch| (EPOCH..=last).contains(&epoch)));
        assert!(recorded.contains(&EPOCH));
    }
}
