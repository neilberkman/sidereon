use crate::astro::error::PropagationError;
use crate::astro::integrators::{DynamicsModel, Integrator};
use crate::astro::propagator::api::{
    validate_integrator_epoch, validate_integrator_options, IntegratorOptions, PropagationContext,
};
use crate::astro::propagator::result::{
    validate_propagation_result, PropagationPoint, PropagationResult, PropagationStats,
};
use crate::astro::state::{CartesianState, StateDerivative};

pub struct RK4;

impl Integrator for RK4 {
    fn propagate(
        &self,
        initial: CartesianState,
        t_end_seconds: f64,
        rhs: &dyn DynamicsModel,
        ctx: &PropagationContext,
        opts: &IntegratorOptions,
    ) -> Result<PropagationResult, PropagationError> {
        validate_integrator_options(opts)?;
        validate_integrator_epoch(initial.epoch_tdb_seconds, "initial.epoch_tdb_seconds")?;
        validate_integrator_epoch(t_end_seconds, "t_end_seconds")?;

        let mut state = initial;
        let mut t = initial.epoch_tdb_seconds;
        let dt_target = t_end_seconds - t;
        let sign = dt_target.signum();
        let target_abs = dt_target.abs();

        let h_initial = opts.initial_step.min(target_abs) * sign;
        let mut h = h_initial;
        let mut steps = 0;
        let mut points = Vec::new();

        points.push(PropagationPoint {
            epoch_tdb_seconds: t,
            position_km: state.position_array(),
            velocity_km_s: state.velocity_array(),
        });

        while (t - initial.epoch_tdb_seconds).abs() < target_abs {
            if steps >= opts.max_steps {
                return Err(PropagationError::MaxStepsExceeded);
            }

            if (t + h - initial.epoch_tdb_seconds).abs() > target_abs {
                h = t_end_seconds - t;
            }

            let next_state = self.step(state, h, rhs, ctx)?;
            state = next_state;
            t += h;
            steps += 1;

            if opts.dense_output {
                points.push(PropagationPoint {
                    epoch_tdb_seconds: t,
                    position_km: state.position_array(),
                    velocity_km_s: state.velocity_array(),
                });
            }
        }

        if !opts.dense_output {
            points.push(PropagationPoint {
                epoch_tdb_seconds: t,
                position_km: state.position_array(),
                velocity_km_s: state.velocity_array(),
            });
        }

        validate_propagation_result(PropagationResult {
            final_state: state,
            points,
            events: Vec::new(),
            stats: PropagationStats {
                accepted_steps: steps,
                rejected_steps: 0,
                evaluations: steps * 4,
            },
            dense: None,
        })
    }
}

impl RK4 {
    fn step(
        &self,
        state: CartesianState,
        h: f64,
        rhs: &dyn DynamicsModel,
        ctx: &PropagationContext,
    ) -> Result<CartesianState, PropagationError> {
        let k1 = rhs.derivative(&state, ctx)?;

        let s2 = self.advance(&state, &k1, h / 2.0);
        let k2 = rhs.derivative(&s2, ctx)?;

        let s3 = self.advance(&state, &k2, h / 2.0);
        let k3 = rhs.derivative(&s3, ctx)?;

        let s4 = self.advance(&state, &k3, h);
        let k4 = rhs.derivative(&s4, ctx)?;

        let dpos =
            (k1.dpos_km_s + k2.dpos_km_s * 2.0 + k3.dpos_km_s * 2.0 + k4.dpos_km_s) * (h / 6.0);
        let dvel =
            (k1.dvel_km_s2 + k2.dvel_km_s2 * 2.0 + k3.dvel_km_s2 * 2.0 + k4.dvel_km_s2) * (h / 6.0);

        Ok(CartesianState {
            epoch_tdb_seconds: state.epoch_tdb_seconds + h,
            position_km: state.position_km + dpos,
            velocity_km_s: state.velocity_km_s + dvel,
        })
    }

    fn advance(&self, state: &CartesianState, deriv: &StateDerivative, h: f64) -> CartesianState {
        CartesianState {
            epoch_tdb_seconds: state.epoch_tdb_seconds + h,
            position_km: state.position_km + deriv.dpos_km_s * h,
            velocity_km_s: state.velocity_km_s + deriv.dvel_km_s2 * h,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct CountingDynamics<'a> {
        calls: &'a AtomicUsize,
    }

    impl DynamicsModel for CountingDynamics<'_> {
        fn derivative(
            &self,
            state: &CartesianState,
            _ctx: &PropagationContext,
        ) -> Result<StateDerivative, PropagationError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(StateDerivative {
                dpos_km_s: state.velocity_km_s,
                dvel_km_s2: Vector3::zeros(),
            })
        }
    }

    struct InfiniteAcceleration;

    impl DynamicsModel for InfiniteAcceleration {
        fn derivative(
            &self,
            state: &CartesianState,
            _ctx: &PropagationContext,
        ) -> Result<StateDerivative, PropagationError> {
            Ok(StateDerivative {
                dpos_km_s: state.velocity_km_s,
                dvel_km_s2: Vector3::new(f64::INFINITY, 0.0, 0.0),
            })
        }
    }

    fn initial_state() -> CartesianState {
        CartesianState {
            epoch_tdb_seconds: 0.0,
            position_km: Vector3::new(7000.0, 0.0, 0.0),
            velocity_km_s: Vector3::new(0.0, 7.5, 0.0),
        }
    }

    #[test]
    fn rejects_non_finite_epochs_before_derivative_evaluation() {
        let base = initial_state();
        let mut nan_initial = base;
        nan_initial.epoch_tdb_seconds = f64::NAN;
        let mut infinite_initial = base;
        infinite_initial.epoch_tdb_seconds = f64::INFINITY;
        let cases = [
            (nan_initial, 60.0, "initial.epoch_tdb_seconds"),
            (infinite_initial, 60.0, "initial.epoch_tdb_seconds"),
            (base, f64::NAN, "t_end_seconds"),
            (base, f64::INFINITY, "t_end_seconds"),
        ];

        for (initial, t_end_seconds, field) in cases {
            let calls = AtomicUsize::new(0);
            let dynamics = CountingDynamics { calls: &calls };
            let ctx = PropagationContext::default();
            let opts = IntegratorOptions::default();

            let error = RK4
                .propagate(initial, t_end_seconds, &dynamics, &ctx, &opts)
                .expect_err("non-finite RK4 epoch must fail validation");

            assert_invalid_input(error, field, "not finite");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "non-finite {field} must be rejected before integration starts"
            );
        }
    }

    #[test]
    fn finite_epochs_integrate_as_before() {
        let calls = AtomicUsize::new(0);
        let dynamics = CountingDynamics { calls: &calls };
        let ctx = PropagationContext::default();
        let opts = IntegratorOptions {
            initial_step: 10.0,
            ..IntegratorOptions::default()
        };

        let result = RK4
            .propagate(initial_state(), 60.0, &dynamics, &ctx, &opts)
            .expect("finite RK4 epochs must remain valid");

        assert_eq!(result.final_state.epoch_tdb_seconds, 60.0);
        assert_eq!(
            result.final_state.position_km.x.to_bits(),
            7000.0f64.to_bits()
        );
        assert_eq!(
            result.final_state.position_km.y.to_bits(),
            450.0f64.to_bits()
        );
        assert_eq!(
            result.final_state.velocity_km_s.y.to_bits(),
            7.5f64.to_bits()
        );
        assert_eq!(result.stats.accepted_steps, 6);
        assert_eq!(calls.load(Ordering::SeqCst), 24);
    }

    #[test]
    fn rejects_non_finite_outputs() {
        let ctx = PropagationContext::default();
        let opts = IntegratorOptions {
            initial_step: 1.0,
            ..IntegratorOptions::default()
        };

        let error = RK4
            .propagate(initial_state(), 1.0, &InfiniteAcceleration, &ctx, &opts)
            .expect_err("non-finite RK4 result must be rejected");

        assert_numerical_failure(error, "final_state.position_km", "not finite");
    }

    fn assert_invalid_input(error: PropagationError, field: &str, reason: &str) {
        match error {
            PropagationError::InvalidInput(message) => {
                assert!(message.contains(field), "{message}");
                assert!(message.contains(reason), "{message}");
            }
            other => panic!("expected invalid propagation input for {field}, got {other:?}"),
        }
    }

    fn assert_numerical_failure(error: PropagationError, field: &str, reason: &str) {
        match error {
            PropagationError::NumericalFailure(message) => {
                assert!(message.contains(field), "{message}");
                assert!(message.contains(reason), "{message}");
            }
            other => panic!("expected numerical failure for {field}, got {other:?}"),
        }
    }
}
