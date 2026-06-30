use crate::astro::error::PropagationError;
use crate::astro::integrators::tableau::DP54Tableau;
use crate::astro::integrators::{DynamicsModel, Integrator};
use crate::astro::propagator::api::{
    validate_adaptive_integrator_options, validate_integrator_epoch, IntegratorOptions,
    PropagationContext,
};
use crate::astro::propagator::controller::PIController;
use crate::astro::propagator::dense_output::{DenseOutput, DenseSegment};
use crate::astro::propagator::result::{
    validate_propagation_result, PropagationPoint, PropagationResult, PropagationStats,
};
use crate::astro::state::{CartesianState, StateDerivative};
use nalgebra::Vector3;

pub struct DP54;

impl Integrator for DP54 {
    fn propagate(
        &self,
        initial: CartesianState,
        t_end_seconds: f64,
        rhs: &dyn DynamicsModel,
        ctx: &PropagationContext,
        opts: &IntegratorOptions,
    ) -> Result<PropagationResult, PropagationError> {
        validate_adaptive_integrator_options(opts)?;
        validate_integrator_epoch(initial.epoch_tdb_seconds, "initial.epoch_tdb_seconds")?;
        validate_integrator_epoch(t_end_seconds, "t_end_seconds")?;

        let dt_target = t_end_seconds - initial.epoch_tdb_seconds;
        let target_abs = dt_target.abs();
        if target_abs == 0.0 {
            let point = PropagationPoint {
                epoch_tdb_seconds: initial.epoch_tdb_seconds,
                position_km: initial.position_array(),
                velocity_km_s: initial.velocity_array(),
            };
            let mut points = vec![point.clone()];
            if !opts.dense_output {
                points.push(point);
            }
            let dense = if opts.dense_output {
                Some(DenseOutput {
                    segments: Vec::new(),
                })
            } else {
                None
            };

            return validate_propagation_result(PropagationResult {
                final_state: initial,
                points,
                events: Vec::new(),
                stats: PropagationStats {
                    accepted_steps: 0,
                    rejected_steps: 0,
                    evaluations: 0,
                },
                dense,
            });
        }

        let tableau = DP54Tableau::default();
        let controller = PIController {
            order: 5.0,
            ..PIController::default()
        };

        let mut state = initial;
        let mut t = initial.epoch_tdb_seconds;
        let sign = dt_target.signum();

        let mut h = crate::validate::clamp_magnitude(
            opts.initial_step.min(target_abs) * sign,
            opts.max_step,
        );
        let mut steps_accepted = 0;
        let mut steps_rejected = 0;
        let mut evals = 0;
        let mut points = Vec::new();
        let mut dense_segments = Vec::new();

        points.push(PropagationPoint {
            epoch_tdb_seconds: t,
            position_km: state.position_array(),
            velocity_km_s: state.velocity_array(),
        });

        // FSAL: k1
        let mut k1 = rhs.derivative(&state, ctx)?;
        evals += 1;

        while (t - initial.epoch_tdb_seconds).abs() < target_abs {
            if steps_accepted + steps_rejected >= opts.max_steps {
                return Err(PropagationError::MaxStepsExceeded);
            }

            let mut h_step = h;
            if (t + h_step - initial.epoch_tdb_seconds).abs() > target_abs {
                h_step = t_end_seconds - t;
            }

            // Step using DP54
            let step_ctx = DP54StepContext {
                rhs,
                ctx,
                tableau: &tableau,
                capture_stages: opts.dense_output,
            };
            let step_res = self.step(state, h_step, k1, &step_ctx)?;

            // Error estimation
            let r_scale = opts.abs_tol
                + state
                    .position_km
                    .norm()
                    .max(step_res.next_state.position_km.norm())
                    * opts.rel_tol;
            let v_scale = opts.abs_tol
                + state
                    .velocity_km_s
                    .norm()
                    .max(step_res.next_state.velocity_km_s.norm())
                    * opts.rel_tol;

            let err_r = step_res.r_err.norm() / r_scale;
            let err_v = step_res.v_err.norm() / v_scale;
            let err = err_r.max(err_v);

            if err <= 1.0 {
                // Accepted
                if opts.dense_output {
                    if let Some(stages) = step_res.stages {
                        let ks_array: [StateDerivative; 7] = stages.try_into().map_err(|_| {
                            PropagationError::NumericalFailure(
                                "Failed to capture RK stages".to_string(),
                            )
                        })?;
                        dense_segments.push(DenseSegment::from_dp54_stages(
                            t,
                            h_step,
                            state,
                            step_res.next_state,
                            &ks_array,
                        ));
                    }
                }

                state = step_res.next_state;
                t += h_step;
                k1 = step_res.k_fsal; // FSAL
                steps_accepted += 1;
                evals += step_res.evals;

                if opts.dense_output {
                    points.push(PropagationPoint {
                        epoch_tdb_seconds: t,
                        position_km: state.position_array(),
                        velocity_km_s: state.velocity_array(),
                    });
                }

                h = crate::validate::clamp_magnitude(
                    controller.next_step(h_step, err),
                    opts.max_step,
                );
            } else {
                steps_rejected += 1;
                evals += step_res.evals;
                h = crate::validate::clamp_magnitude(
                    controller.next_step(h_step, err),
                    opts.max_step,
                );

                if h.abs() < opts.min_step {
                    return Err(PropagationError::NumericalFailure(
                        "Step size too small".to_string(),
                    ));
                }
            }
        }

        if !opts.dense_output {
            points.push(PropagationPoint {
                epoch_tdb_seconds: t,
                position_km: state.position_array(),
                velocity_km_s: state.velocity_array(),
            });
        }

        let dense = if opts.dense_output {
            Some(DenseOutput {
                segments: dense_segments,
            })
        } else {
            None
        };

        validate_propagation_result(PropagationResult {
            final_state: state,
            points,
            events: Vec::new(),
            stats: PropagationStats {
                accepted_steps: steps_accepted,
                rejected_steps: steps_rejected,
                evaluations: evals,
            },
            dense,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    struct CountingOscillator<'a> {
        calls: &'a AtomicUsize,
    }

    impl DynamicsModel for CountingOscillator<'_> {
        fn derivative(
            &self,
            state: &CartesianState,
            _ctx: &PropagationContext,
        ) -> Result<StateDerivative, PropagationError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(StateDerivative {
                dpos_km_s: state.velocity_km_s,
                dvel_km_s2: -state.position_km,
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
    fn rejects_invalid_tolerances_before_derivative_evaluation() {
        let cases = [
            ("abs_tol", "not positive", -1.0, 1.0e-12),
            ("abs_tol", "not positive", 0.0, 1.0e-12),
            ("abs_tol", "not finite", f64::NAN, 1.0e-12),
            ("rel_tol", "not positive", 1.0e-9, -1.0),
            ("rel_tol", "not positive", 1.0e-9, 0.0),
            ("rel_tol", "not finite", 1.0e-9, f64::NAN),
        ];

        for (field, reason, abs_tol, rel_tol) in cases {
            let calls = AtomicUsize::new(0);
            let dynamics = CountingDynamics { calls: &calls };
            let ctx = PropagationContext::default();
            let opts = IntegratorOptions {
                abs_tol,
                rel_tol,
                ..IntegratorOptions::default()
            };

            let error = DP54
                .propagate(initial_state(), 60.0, &dynamics, &ctx, &opts)
                .expect_err("invalid DP54 tolerance must fail validation");

            assert_invalid_input(error, field, reason);
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "invalid {field} must be rejected before integration starts"
            );
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

            let error = DP54
                .propagate(initial, t_end_seconds, &dynamics, &ctx, &opts)
                .expect_err("non-finite DP54 epoch must fail validation");

            assert_invalid_input(error, field, "not finite");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                0,
                "non-finite {field} must be rejected before integration starts"
            );
        }
    }

    #[test]
    fn accepts_positive_tolerances() {
        let calls = AtomicUsize::new(0);
        let dynamics = CountingDynamics { calls: &calls };
        let ctx = PropagationContext::default();
        let opts = IntegratorOptions {
            abs_tol: 1.0e-9,
            rel_tol: 1.0e-12,
            initial_step: 10.0,
            ..IntegratorOptions::default()
        };

        let result = DP54
            .propagate(initial_state(), 60.0, &dynamics, &ctx, &opts)
            .expect("positive DP54 tolerances must remain valid");

        assert_eq!(result.final_state.epoch_tdb_seconds, 60.0);
        assert!(calls.load(Ordering::SeqCst) > 0);
    }

    #[test]
    fn zero_duration_returns_initial_state_without_derivative_evaluation() {
        let calls = AtomicUsize::new(0);
        let dynamics = CountingDynamics { calls: &calls };
        let ctx = PropagationContext::default();
        let opts = IntegratorOptions::default();
        let initial = initial_state();

        let result = DP54
            .propagate(initial, initial.epoch_tdb_seconds, &dynamics, &ctx, &opts)
            .expect("zero-duration propagation should return the initial state");

        assert_eq!(result.final_state, initial);
        assert_eq!(result.stats.accepted_steps, 0);
        assert_eq!(result.stats.rejected_steps, 0);
        assert_eq!(result.stats.evaluations, 0);
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn zero_duration_rejects_non_finite_initial_state_output() {
        let calls = AtomicUsize::new(0);
        let dynamics = CountingDynamics { calls: &calls };
        let ctx = PropagationContext::default();
        let opts = IntegratorOptions::default();
        let mut initial = initial_state();
        initial.position_km.x = f64::INFINITY;

        let error = DP54
            .propagate(initial, initial.epoch_tdb_seconds, &dynamics, &ctx, &opts)
            .expect_err("zero-duration non-finite output must be rejected");

        assert_numerical_failure(error, "final_state.position_km", "not finite");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn rejected_steps_count_every_derivative_evaluation() {
        let calls = AtomicUsize::new(0);
        let dynamics = CountingOscillator { calls: &calls };
        let ctx = PropagationContext::default();
        let opts = IntegratorOptions {
            abs_tol: 1.0e-12,
            rel_tol: 1.0e-12,
            initial_step: 1.0,
            max_step: 1.0,
            min_step: 1.0e-15,
            ..IntegratorOptions::default()
        };
        let initial = CartesianState {
            epoch_tdb_seconds: 0.0,
            position_km: Vector3::new(1.0, 0.0, 0.0),
            velocity_km_s: Vector3::new(0.0, 1.0, 0.0),
        };

        let result = DP54
            .propagate(initial, 1.0, &dynamics, &ctx, &opts)
            .expect("tight oscillator propagation should recover after rejected steps");

        assert!(
            result.stats.rejected_steps > 0,
            "test setup must force at least one rejected step"
        );
        assert_eq!(
            result.stats.evaluations,
            calls.load(Ordering::SeqCst) as u32
        );
        assert_eq!(
            result.stats.evaluations,
            1 + 6 * (result.stats.accepted_steps + result.stats.rejected_steps)
        );
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

struct DP54Step {
    next_state: CartesianState,
    k_fsal: StateDerivative,
    r_err: Vector3<f64>,
    v_err: Vector3<f64>,
    evals: u32,
    stages: Option<Vec<StateDerivative>>,
}

/// Per-step invariants shared across every Dormand-Prince stage evaluation:
/// the dynamics model, propagation context, Butcher tableau, and whether to
/// retain the intermediate stages for dense output.
#[derive(Clone, Copy)]
struct DP54StepContext<'a> {
    rhs: &'a dyn DynamicsModel,
    ctx: &'a PropagationContext,
    tableau: &'a DP54Tableau,
    capture_stages: bool,
}

impl DP54 {
    fn step(
        &self,
        state: CartesianState,
        h: f64,
        k1: StateDerivative,
        step_ctx: &DP54StepContext,
    ) -> Result<DP54Step, PropagationError> {
        let DP54StepContext {
            rhs,
            ctx,
            tableau,
            capture_stages,
        } = *step_ctx;
        let mut ks = Vec::with_capacity(7);
        ks.push(k1);

        for i in 1..6 {
            let mut dpos = Vector3::zeros();
            let mut dvel = Vector3::zeros();
            for (j, k) in ks.iter().enumerate().take(i) {
                dpos += k.dpos_km_s * tableau.a[i][j];
                dvel += k.dvel_km_s2 * tableau.a[i][j];
            }

            let stage_state = CartesianState {
                epoch_tdb_seconds: state.epoch_tdb_seconds + h * tableau.c[i],
                position_km: state.position_km + dpos * h,
                velocity_km_s: state.velocity_km_s + dvel * h,
            };
            ks.push(rhs.derivative(&stage_state, ctx)?);
        }

        // 5th order solution
        let mut dpos5 = Vector3::zeros();
        let mut dvel5 = Vector3::zeros();
        for (i, k) in ks.iter().enumerate().take(6) {
            dpos5 += k.dpos_km_s * tableau.b5[i];
            dvel5 += k.dvel_km_s2 * tableau.b5[i];
        }

        let next_state = CartesianState {
            epoch_tdb_seconds: state.epoch_tdb_seconds + h,
            position_km: state.position_km + dpos5 * h,
            velocity_km_s: state.velocity_km_s + dvel5 * h,
        };

        // FSAL
        let k_fsal = rhs.derivative(&next_state, ctx)?;
        ks.push(k_fsal);

        // 4th order for error estimate
        let mut dpos4 = Vector3::zeros();
        let mut dvel4 = Vector3::zeros();
        for (i, k) in ks.iter().enumerate().take(7) {
            dpos4 += k.dpos_km_s * tableau.b4[i];
            dvel4 += k.dvel_km_s2 * tableau.b4[i];
        }

        let r_err = (dpos5 - dpos4) * h;
        let v_err = (dvel5 - dvel4) * h;

        let stages = if capture_stages { Some(ks) } else { None };

        Ok(DP54Step {
            next_state,
            k_fsal,
            r_err,
            v_err,
            evals: 6,
            stages,
        })
    }
}
