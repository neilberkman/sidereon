//! Covariance transport along numerical propagation segments.

use crate::astro::covariance::{
    covariance_congruence6_checked, finite6, rtn_to_eci_rotation, symmetrize6, Covariance6,
    RtnFrameError,
};
use crate::astro::error::PropagationError;
use crate::astro::math::mat3;
use crate::astro::propagator::api::PropagationContext;
use crate::astro::propagator::dynamics::OrbitalDynamics;
use crate::astro::propagator::numerical::{map_covariance6_error, StatePropagator};
use crate::astro::propagator::StateTransitionMatrix;
use crate::astro::state::CartesianState;

/// Frame a 6x6 state covariance is expressed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovarianceFrame {
    /// The propagator's inertial frame.
    Inertial,
    /// Satellite-relative radial, transverse, normal axes.
    Rtn,
}

/// A covariance plus the frame label it is expressed in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LabeledCovariance6 {
    pub covariance: Covariance6,
    pub frame: CovarianceFrame,
}

/// Process-noise model for covariance transport.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ProcessNoise {
    /// Pure Phi P Phi^T transport.
    #[default]
    None,
    /// Per-axis white acceleration PSD in RTN, km^2/s^3.
    RtnAccelerationPsd {
        q_radial_km2_s3: f64,
        q_transverse_km2_s3: f64,
        q_normal_km2_s3: f64,
    },
}

/// Options for a covariance propagation run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovariancePropagationOptions {
    pub process_noise: ProcessNoise,
    pub output_frame: CovarianceFrame,
}

impl Default for CovariancePropagationOptions {
    fn default() -> Self {
        Self {
            process_noise: ProcessNoise::None,
            output_frame: CovarianceFrame::Inertial,
        }
    }
}

/// State plus covariance at one requested epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovarianceNode {
    /// The propagated Cartesian state, always in the inertial frame.
    pub state: CartesianState,
    /// The covariance expressed in `frame`.
    pub covariance: Covariance6,
    /// Frame used by `covariance`.
    pub frame: CovarianceFrame,
}

/// Ordered result of a covariance propagation run.
#[derive(Debug, Clone, PartialEq)]
pub struct CovarianceEphemeris {
    nodes: Vec<CovarianceNode>,
}

impl CovarianceEphemeris {
    /// Borrow the propagated nodes in request order.
    pub fn nodes(&self) -> &[CovarianceNode] {
        &self.nodes
    }

    /// Number of propagated nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Whether this result contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Interpolate an inertial covariance inside the node span.
    ///
    /// Exact node hits return the stored covariance without a factorization
    /// round trip. RTN ephemerides are rejected because the RTN frame at a query
    /// epoch is undefined unless that epoch is propagated as a node.
    pub fn covariance_at(&self, epoch_tdb_seconds: f64) -> Result<Covariance6, PropagationError> {
        crate::validate::finite(epoch_tdb_seconds, "epoch_tdb_seconds").map_err(|error| {
            PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
        })?;
        if self.nodes.is_empty() {
            return Err(PropagationError::InvalidInput(
                "covariance ephemeris is empty".to_string(),
            ));
        }
        if self
            .nodes
            .iter()
            .any(|node| node.frame != CovarianceFrame::Inertial)
        {
            return Err(PropagationError::InvalidInput(
                "covariance_at requires inertial covariance nodes; request the epoch as a propagation node for RTN output".to_string(),
            ));
        }

        for node in &self.nodes {
            if node.state.epoch_tdb_seconds == epoch_tdb_seconds {
                return Ok(node.covariance);
            }
        }

        for pair in self.nodes.windows(2) {
            let a = pair[0];
            let b = pair[1];
            let ta = a.state.epoch_tdb_seconds;
            let tb = b.state.epoch_tdb_seconds;
            if ta == tb {
                continue;
            }
            let inside = (ta <= epoch_tdb_seconds && epoch_tdb_seconds <= tb)
                || (tb <= epoch_tdb_seconds && epoch_tdb_seconds <= ta);
            if inside {
                let u = (epoch_tdb_seconds - ta) / (tb - ta);
                return crate::astro::covariance::interpolate_covariance_psd(
                    &a.covariance,
                    &b.covariance,
                    u,
                )
                .map_err(map_covariance6_error);
            }
        }

        Err(PropagationError::InvalidInput(
            "epoch_tdb_seconds outside covariance ephemeris span".to_string(),
        ))
    }
}

/// One caller-supplied covariance transport segment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CovarianceSegment {
    /// Segment state-transition matrix.
    pub stm: StateTransitionMatrix,
    /// Segment duration in seconds. Process noise uses the absolute value.
    pub dt_seconds: f64,
    /// State whose RTN axes rotate segment process noise into inertial axes.
    pub q_rotation_state: CartesianState,
}

impl StatePropagator {
    /// Propagate the initial state and covariance to each requested epoch.
    ///
    /// A RTN-labeled input is first rotated to inertial axes at the initial
    /// state. Epochs must be monotonic from the initial epoch, with repeated
    /// epochs allowed. The output covariance is expressed in the requested
    /// frame at each node.
    pub fn propagate_covariance(
        &self,
        initial: LabeledCovariance6,
        epochs_tdb_seconds: &[f64],
        options: &CovariancePropagationOptions,
    ) -> Result<CovarianceEphemeris, PropagationError> {
        validate_initial_state(self.initial)?;
        validate_epochs_monotonic(self.initial.epoch_tdb_seconds, epochs_tdb_seconds)?;
        validate_process_noise(options.process_noise)?;

        let force = self.build_force()?;
        let dynamics = OrbitalDynamics {
            force_model: force.as_ref(),
        };
        let ctx = PropagationContext::default();

        let mut current_state = self.initial;
        let mut current_covariance = match initial.frame {
            CovarianceFrame::Inertial => initial.covariance,
            CovarianceFrame::Rtn => {
                covariance_rtn_to_eci_for_propagation(&initial.covariance, &self.initial)?
            }
        };
        let mut nodes = Vec::with_capacity(epochs_tdb_seconds.len());

        for &epoch in epochs_tdb_seconds {
            if epoch != current_state.epoch_tdb_seconds {
                let dt = epoch - current_state.epoch_tdb_seconds;
                let q_rotation_state =
                    segment_q_rotation_state(self, current_state, epoch, options, &dynamics, &ctx)?;
                let stm =
                    self.state_transition_matrix_between(current_state, epoch, &dynamics, &ctx)?;
                let final_state = self.run(current_state, epoch, &dynamics, &ctx)?.final_state;
                let segments = [CovarianceSegment {
                    stm,
                    dt_seconds: dt,
                    q_rotation_state,
                }];
                let transported =
                    transport_covariance(current_covariance, &segments, options.process_noise)?;
                current_covariance = transported[1];
                current_state = final_state;
            }

            nodes.push(CovarianceNode {
                state: current_state,
                covariance: express_output_covariance(
                    current_covariance,
                    current_state,
                    options.output_frame,
                )?,
                frame: options.output_frame,
            });
        }

        Ok(CovarianceEphemeris { nodes })
    }
}

/// Apply covariance transport over caller-supplied segments.
///
/// The returned vector contains the initial covariance followed by one
/// covariance per segment.
pub fn transport_covariance(
    covariance0: Covariance6,
    segments: &[CovarianceSegment],
    process_noise: ProcessNoise,
) -> Result<Vec<Covariance6>, PropagationError> {
    validate_process_noise(process_noise)?;
    let mut covariances = Vec::with_capacity(segments.len() + 1);
    let mut current = covariance0;
    covariances.push(current);

    for segment in segments {
        if !finite6(&segment.stm) {
            return Err(PropagationError::NumericalFailure(
                "state_transition_matrix not finite".to_string(),
            ));
        }
        crate::validate::finite(segment.dt_seconds, "dt_seconds").map_err(|error| {
            PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
        })?;

        let mut matrix = current
            .propagate_with_stm(&segment.stm)
            .map_err(map_covariance6_error)?
            .into_matrix();
        if let Some(q) = process_noise_increment(segment, process_noise)? {
            let q_matrix = q.as_matrix();
            for (i, row) in matrix.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell += q_matrix[i][j];
                }
            }
            symmetrize6(&mut matrix);
        }
        current = Covariance6::try_from_matrix(matrix).map_err(map_covariance6_error)?;
        covariances.push(current);
    }

    Ok(covariances)
}

fn segment_q_rotation_state(
    propagator: &StatePropagator,
    current_state: CartesianState,
    epoch: f64,
    options: &CovariancePropagationOptions,
    dynamics: &OrbitalDynamics,
    ctx: &PropagationContext,
) -> Result<CartesianState, PropagationError> {
    if matches!(options.process_noise, ProcessNoise::None) {
        return Ok(current_state);
    }
    let midpoint = 0.5 * (current_state.epoch_tdb_seconds + epoch);
    if midpoint == current_state.epoch_tdb_seconds {
        Ok(current_state)
    } else {
        Ok(propagator
            .run(current_state, midpoint, dynamics, ctx)?
            .final_state)
    }
}

fn express_output_covariance(
    covariance: Covariance6,
    state: CartesianState,
    frame: CovarianceFrame,
) -> Result<Covariance6, PropagationError> {
    match frame {
        CovarianceFrame::Inertial => Ok(covariance),
        CovarianceFrame::Rtn => covariance_eci_to_rtn_for_propagation(&covariance, &state),
    }
}

fn process_noise_increment(
    segment: &CovarianceSegment,
    process_noise: ProcessNoise,
) -> Result<Option<Covariance6>, PropagationError> {
    let ProcessNoise::RtnAccelerationPsd {
        q_radial_km2_s3,
        q_transverse_km2_s3,
        q_normal_km2_s3,
    } = process_noise
    else {
        return Ok(None);
    };

    let dt = segment.dt_seconds.abs();
    if q_radial_km2_s3 == 0.0 && q_transverse_km2_s3 == 0.0 && q_normal_km2_s3 == 0.0 {
        return Ok(None);
    }
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    let qs = [q_radial_km2_s3, q_transverse_km2_s3, q_normal_km2_s3];
    let mut matrix = [[0.0_f64; 6]; 6];
    for axis in 0..3 {
        matrix[axis][axis] = qs[axis] * dt3 / 3.0;
        matrix[axis][axis + 3] = qs[axis] * dt2 / 2.0;
        matrix[axis + 3][axis] = qs[axis] * dt2 / 2.0;
        matrix[axis + 3][axis + 3] = qs[axis] * dt;
    }
    let q_rtn = Covariance6::try_from_matrix(matrix).map_err(map_covariance6_error)?;
    covariance_rtn_to_eci_for_propagation(&q_rtn, &segment.q_rotation_state).map(Some)
}

fn validate_epochs_monotonic(
    initial_epoch: f64,
    epochs_tdb_seconds: &[f64],
) -> Result<(), PropagationError> {
    crate::validate::finite(initial_epoch, "initial.epoch_tdb_seconds").map_err(|error| {
        PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
    })?;
    let mut current = initial_epoch;
    let mut direction = 0.0_f64;
    for &epoch in epochs_tdb_seconds {
        crate::validate::finite(epoch, "epochs_tdb_seconds").map_err(|error| {
            PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
        })?;
        let dt = epoch - current;
        if dt != 0.0 {
            let sign = dt.signum();
            if direction == 0.0 {
                direction = sign;
            } else if sign != direction {
                return Err(PropagationError::InvalidInput(
                    "epochs_tdb_seconds direction reversal".to_string(),
                ));
            }
        }
        current = epoch;
    }
    Ok(())
}

fn validate_initial_state(initial: CartesianState) -> Result<(), PropagationError> {
    validate_state_vector(initial.position_array(), "initial.position_km")?;
    validate_state_vector(initial.velocity_array(), "initial.velocity_km_s")
}

fn validate_state_vector(values: [f64; 3], field: &'static str) -> Result<(), PropagationError> {
    crate::validate::finite_slice(&values, field).map_err(|error| {
        PropagationError::InvalidInput(format!("{} {}", error.field(), error.reason()))
    })
}

fn validate_process_noise(process_noise: ProcessNoise) -> Result<(), PropagationError> {
    let ProcessNoise::RtnAccelerationPsd {
        q_radial_km2_s3,
        q_transverse_km2_s3,
        q_normal_km2_s3,
    } = process_noise
    else {
        return Ok(());
    };

    validate_q("q_radial_km2_s3", q_radial_km2_s3)?;
    validate_q("q_transverse_km2_s3", q_transverse_km2_s3)?;
    validate_q("q_normal_km2_s3", q_normal_km2_s3)
}

fn validate_q(field: &'static str, value: f64) -> Result<(), PropagationError> {
    if !value.is_finite() {
        return Err(PropagationError::InvalidInput(format!(
            "{field} not finite"
        )));
    }
    if value < 0.0 {
        return Err(PropagationError::InvalidInput(format!("{field} negative")));
    }
    Ok(())
}

fn map_rtn_frame_error(error: RtnFrameError) -> PropagationError {
    PropagationError::InvalidInput(format!("covariance frame {}", error.message()))
}

fn covariance_rtn_to_eci_for_propagation(
    covariance: &Covariance6,
    state: &CartesianState,
) -> Result<Covariance6, PropagationError> {
    let rot = rtn_to_eci_rotation(state.position_array(), state.velocity_array())
        .map_err(map_rtn_frame_error)?;
    covariance_congruence6_checked(covariance, &rot).map_err(map_covariance6_error)
}

fn covariance_eci_to_rtn_for_propagation(
    covariance: &Covariance6,
    state: &CartesianState,
) -> Result<Covariance6, PropagationError> {
    let rot = rtn_to_eci_rotation(state.position_array(), state.velocity_array())
        .map_err(map_rtn_frame_error)?;
    let rot_t = mat3::inline_tr(&rot);
    covariance_congruence6_checked(covariance, &rot_t).map_err(map_covariance6_error)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::astro::constants::MU_EARTH;
    use crate::astro::covariance::{
        covariance_congruence6_checked, eci_to_rtn_covariance6, interpolate_covariance_psd,
        rtn_to_eci_covariance6, Mat6,
    };
    use crate::astro::forces::{DragForce, DragParameters, SpaceWeather};
    use crate::astro::math::mat3::Mat3;
    use crate::astro::propagator::{ForceModelKind, IntegratorKind, IntegratorOptions};
    use crate::astro::relative::cw_stm;
    use nalgebra::SMatrix;
    use std::f64::consts::PI;

    fn circular_propagator() -> StatePropagator {
        let r: f64 = 7000.0;
        let v = (MU_EARTH / r).sqrt();
        StatePropagator::new(
            0.0,
            [r, 0.0, 0.0],
            [0.0, v, 0.0],
            ForceModelKind::two_body(),
            IntegratorKind::Rk4,
        )
        .with_options(IntegratorOptions {
            initial_step: 2.0,
            ..IntegratorOptions::default()
        })
    }

    fn test_covariance() -> Covariance6 {
        Covariance6::from_diagonal([1.0e-4, 2.0e-4, 3.0e-4, 1.0e-8, 2.0e-8, 3.0e-8]).unwrap()
    }

    fn coupled_covariance() -> Covariance6 {
        covariance_from_lower([
            [0.4, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.02, 0.5, 0.0, 0.0, 0.0, 0.0],
            [-0.01, 0.03, 0.3, 0.0, 0.0, 0.0],
            [1.0e-5, 2.0e-5, 0.0, 4.0e-4, 0.0, 0.0],
            [0.0, -2.0e-5, 3.0e-5, 1.0e-5, 5.0e-4, 0.0],
            [1.0e-5, 0.0, -1.0e-5, 0.0, 2.0e-5, 3.0e-4],
        ])
    }

    #[test]
    fn multi_epoch_no_noise_matches_segment_chaining() {
        let propagator = circular_propagator();
        let covariance0 = test_covariance();
        let epochs = [0.0, 60.0, 120.0];

        let ephemeris = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &CovariancePropagationOptions::default(),
            )
            .expect("covariance ephemeris");

        assert_eq!(ephemeris.len(), 3);
        assert_eq!(ephemeris.nodes()[0].covariance, covariance0);
        assert_eq!(ephemeris.nodes()[2].state.epoch_tdb_seconds, 120.0);
        assert!(ephemeris.nodes()[2].covariance.is_positive_semidefinite());

        let (_, single_segment) = propagator
            .propagate_state_with_covariance(covariance0, 60.0)
            .expect("single segment covariance");
        assert_eq!(ephemeris.nodes()[1].covariance, single_segment);
    }

    #[test]
    fn zero_span_and_repeated_epochs_reuse_current_node() {
        let propagator = circular_propagator();
        let covariance0 = coupled_covariance();
        let ephemeris = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &[0.0, 60.0, 60.0],
                &CovariancePropagationOptions::default(),
            )
            .expect("covariance ephemeris");

        assert_eq!(ephemeris.nodes()[0].state, propagator.initial);
        assert_eq!(ephemeris.nodes()[0].covariance, covariance0);
        assert_eq!(ephemeris.nodes()[1], ephemeris.nodes()[2]);
    }

    #[test]
    fn rtn_output_is_labeled_and_round_trips_to_inertial() {
        let propagator = circular_propagator();
        let covariance0 = test_covariance();
        let epochs = [120.0];

        let inertial = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &CovariancePropagationOptions::default(),
            )
            .expect("inertial covariance ephemeris");
        let rtn = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &CovariancePropagationOptions {
                    output_frame: CovarianceFrame::Rtn,
                    ..CovariancePropagationOptions::default()
                },
            )
            .expect("RTN covariance ephemeris");

        assert_eq!(rtn.nodes()[0].frame, CovarianceFrame::Rtn);
        let round_trip =
            rtn_to_eci_covariance6(&rtn.nodes()[0].covariance, &rtn.nodes()[0].state).unwrap();
        for i in 0..6 {
            for j in 0..6 {
                let expected = inertial.nodes()[0].covariance.as_matrix()[i][j];
                let actual = round_trip.as_matrix()[i][j];
                assert!((actual - expected).abs() <= 1.0e-12 * expected.abs().max(1.0));
            }
        }
    }

    #[test]
    fn process_noise_increases_velocity_variance() {
        let propagator = circular_propagator();
        let covariance0 = test_covariance();
        let epochs = [60.0, 120.0];
        let options = CovariancePropagationOptions {
            process_noise: ProcessNoise::RtnAccelerationPsd {
                q_radial_km2_s3: 1.0e-12,
                q_transverse_km2_s3: 2.0e-12,
                q_normal_km2_s3: 3.0e-12,
            },
            output_frame: CovarianceFrame::Inertial,
        };

        let no_noise = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &CovariancePropagationOptions::default(),
            )
            .expect("no-noise covariance ephemeris");
        let with_noise = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &options,
            )
            .expect("noise covariance ephemeris");

        let no_noise_trace = velocity_trace(no_noise.nodes()[1].covariance);
        let with_noise_trace = velocity_trace(with_noise.nodes()[1].covariance);
        assert!(with_noise_trace > no_noise_trace);
        assert!(with_noise.nodes()[1].covariance.is_positive_semidefinite());
    }

    #[test]
    fn zero_process_noise_matches_none_bit_for_bit() {
        let state = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]);
        let segment = CovarianceSegment {
            stm: identity6(),
            dt_seconds: 120.0,
            q_rotation_state: state,
        };
        let covariance0 = coupled_covariance();

        let none = transport_covariance(covariance0, &[segment], ProcessNoise::None)
            .expect("no-noise transport");
        let zero = transport_covariance(
            covariance0,
            &[segment],
            ProcessNoise::RtnAccelerationPsd {
                q_radial_km2_s3: 0.0,
                q_transverse_km2_s3: 0.0,
                q_normal_km2_s3: 0.0,
            },
        )
        .expect("zero-noise transport");

        assert_eq!(zero, none);
    }

    #[test]
    fn backward_process_noise_uses_positive_time_span() {
        let state = CartesianState::new(0.0, [7000.0, 0.0, 0.0], [0.0, 7.5, 0.0]);
        let segment = CovarianceSegment {
            stm: identity6(),
            dt_seconds: -10.0,
            q_rotation_state: state,
        };
        let covariance0 = Covariance6::from_diagonal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();

        let covariances = transport_covariance(
            covariance0,
            &[segment],
            ProcessNoise::RtnAccelerationPsd {
                q_radial_km2_s3: 1.0e-9,
                q_transverse_km2_s3: 0.0,
                q_normal_km2_s3: 0.0,
            },
        )
        .expect("backward process noise");

        assert!(covariances[1].as_matrix()[0][0] > covariance0.as_matrix()[0][0]);
        assert!(covariances[1].as_matrix()[3][3] > covariance0.as_matrix()[3][3]);
    }

    #[test]
    fn rtn_labeled_input_matches_manual_pre_rotation() {
        let propagator = circular_propagator();
        let inertial0 = coupled_covariance();
        let rtn0 = eci_to_rtn_covariance6(&inertial0, &propagator.initial).expect("initial RTN");
        let manually_inertial =
            rtn_to_eci_covariance6(&rtn0, &propagator.initial).expect("manual RTN to ECI");
        let epochs = [240.0];

        let from_rtn = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: rtn0,
                    frame: CovarianceFrame::Rtn,
                },
                &epochs,
                &CovariancePropagationOptions::default(),
            )
            .expect("RTN-labeled propagation");
        let from_manual = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: manually_inertial,
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &CovariancePropagationOptions::default(),
            )
            .expect("manual inertial propagation");

        assert_eq!(
            from_rtn.nodes()[0].covariance,
            from_manual.nodes()[0].covariance
        );
    }

    #[test]
    fn transport_covariance_matches_orchestrated_single_segment() {
        let propagator = circular_propagator();
        let covariance0 = coupled_covariance();
        let epoch = 180.0;
        let stm = propagator
            .state_transition_matrix_to(epoch)
            .expect("single-segment STM");
        let direct = transport_covariance(
            covariance0,
            &[CovarianceSegment {
                stm,
                dt_seconds: epoch,
                q_rotation_state: propagator.initial,
            }],
            ProcessNoise::None,
        )
        .expect("direct transport");
        let ephemeris = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &[epoch],
                &CovariancePropagationOptions::default(),
            )
            .expect("orchestrated transport");

        assert_eq!(direct[1], ephemeris.nodes()[0].covariance);
    }

    #[test]
    fn covariance_ephemeris_interpolates_inertial_nodes_only() {
        let propagator = circular_propagator();
        let covariance0 = coupled_covariance();
        let inertial = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &[0.0, 120.0, 240.0],
                &CovariancePropagationOptions::default(),
            )
            .expect("inertial covariance ephemeris");
        let exact = inertial
            .covariance_at(120.0)
            .expect("exact covariance node");
        assert_eq!(exact, inertial.nodes()[1].covariance);

        let interpolated = inertial
            .covariance_at(60.0)
            .expect("interpolated covariance");
        assert!(interpolated.is_symmetric());
        assert!(interpolated.is_positive_semidefinite());

        let rtn = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &[0.0, 120.0],
                &CovariancePropagationOptions {
                    output_frame: CovarianceFrame::Rtn,
                    ..CovariancePropagationOptions::default()
                },
            )
            .expect("RTN covariance ephemeris");
        assert!(matches!(
            rtn.covariance_at(60.0),
            Err(PropagationError::InvalidInput(message)) if message.contains("inertial")
        ));
    }

    #[test]
    fn cw_analytic_oracle_matches_two_body_transport() {
        let propagator = circular_propagator();
        let covariance0 = coupled_covariance();
        let radius = propagator.initial.position_km.norm();
        let mean_motion = (MU_EARTH / radius.powi(3)).sqrt();
        let dt = 0.25 * 2.0 * PI / mean_motion;

        let ephemeris = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &[dt],
                &CovariancePropagationOptions::default(),
            )
            .expect("finite-difference covariance propagation");
        let actual = ephemeris.nodes()[0].covariance;
        let final_state = ephemeris.nodes()[0].state;

        let j0 = hill_jacobian(&propagator.initial, mean_motion);
        let hill0 = covariance_through_jacobian(covariance0, &j0);
        let cw = cw_stm(mean_motion, dt).expect("CW STM");
        let hill_f = covariance_through_jacobian(hill0, &cw);
        let inertial_j = hill_inverse_jacobian(&final_state, mean_motion);
        let expected = covariance_through_jacobian(hill_f, &inertial_j);

        assert!(
            relative_frobenius(actual.as_matrix(), expected.as_matrix()) <= 1.0e-4,
            "CW oracle mismatch"
        );
    }

    #[test]
    fn two_body_logdet_anchors_hold_without_and_with_process_noise() {
        let propagator = circular_propagator();
        let covariance0 = coupled_covariance();
        let radius = propagator.initial.position_km.norm();
        let period = 2.0 * PI * (radius.powi(3) / MU_EARTH).sqrt();
        let epochs: Vec<f64> = (1..=10).map(|idx| period * f64::from(idx) / 10.0).collect();
        let initial_logdet = logdet(covariance0);

        let no_noise = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &CovariancePropagationOptions::default(),
            )
            .expect("Q-free covariance ephemeris");
        let final_logdet = logdet(no_noise.nodes().last().unwrap().covariance);
        assert!((final_logdet - initial_logdet).abs() <= 1.0e-5);

        let noisy = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: covariance0,
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &CovariancePropagationOptions {
                    process_noise: ProcessNoise::RtnAccelerationPsd {
                        q_radial_km2_s3: 1.0e-13,
                        q_transverse_km2_s3: 1.0e-13,
                        q_normal_km2_s3: 1.0e-13,
                    },
                    output_frame: CovarianceFrame::Inertial,
                },
            )
            .expect("Q covariance ephemeris");
        let mut previous = initial_logdet;
        for node in noisy.nodes() {
            let current = logdet(node.covariance);
            assert!(current + 1.0e-6 >= previous);
            previous = current;
        }
    }

    #[test]
    fn seeded_psd_property_suite_preserves_psd_and_interpolation_bounds() {
        let mut rng = SplitMix64::new(0x9876_5432_10fe_dcba);
        for _ in 0..1000 {
            let initial_state = random_leo_state(&mut rng);
            let covariance0 = random_spd_covariance(&mut rng);
            let span = rng.range_f64(1.0, 3600.0);
            let process_noise = ProcessNoise::RtnAccelerationPsd {
                q_radial_km2_s3: rng.range_f64(0.0, 1.0e-12),
                q_transverse_km2_s3: rng.range_f64(0.0, 1.0e-12),
                q_normal_km2_s3: rng.range_f64(0.0, 1.0e-12),
            };
            let propagator = StatePropagator {
                initial: initial_state,
                force_model: ForceModelKind::two_body(),
                integrator: IntegratorKind::Rk4,
                options: IntegratorOptions {
                    initial_step: 900.0,
                    ..IntegratorOptions::default()
                },
                drag: None,
                space_weather: None,
            };
            for output_frame in [CovarianceFrame::Inertial, CovarianceFrame::Rtn] {
                let ephemeris = propagator
                    .propagate_covariance(
                        LabeledCovariance6 {
                            covariance: covariance0,
                            frame: CovarianceFrame::Inertial,
                        },
                        &[initial_state.epoch_tdb_seconds + span],
                        &CovariancePropagationOptions {
                            process_noise,
                            output_frame,
                        },
                    )
                    .expect("random covariance propagation");
                assert!(ephemeris.nodes()[0].covariance.is_symmetric());
                assert!(ephemeris.nodes()[0].covariance.is_positive_semidefinite());
            }

            let b = random_spd_covariance(&mut rng);
            let u = rng.range_f64(0.0, 1.0);
            let interpolated =
                interpolate_covariance_psd(&covariance0, &b, u).expect("random interpolation");
            assert!(interpolated.is_symmetric());
            assert!(interpolated.is_positive_semidefinite());
            assert_eq!(
                interpolate_covariance_psd(&covariance0, &b, 0.0).unwrap(),
                covariance0
            );
            assert_eq!(
                interpolate_covariance_psd(&covariance0, &b, 1.0).unwrap(),
                b
            );
        }

        let (a, b) = rotated_anisotropic_pair();
        let geodesic = interpolate_covariance_psd(&a, &b, 0.5).expect("geodesic midpoint");
        let linear = linear_covariance_midpoint(&a, &b);
        assert!(logdet(geodesic) < logdet(linear));
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn long_arc_j2_drag_covariance_stays_symmetric_psd_and_bounded() {
        let radius = 6878.137_f64;
        let velocity = (MU_EARTH / radius).sqrt();
        let period = 2.0 * PI * (radius.powi(3) / MU_EARTH).sqrt();
        let drag = DragParameters::from_bc_factor_m2_kg(
            0.02,
            SpaceWeather::default(),
            DragForce::DEFAULT_REENTRY_ALTITUDE_KM,
        )
        .expect("valid drag parameters");
        let propagator = StatePropagator::new(
            0.0,
            [radius, 0.0, 0.0],
            [0.0, velocity, 0.0],
            ForceModelKind::two_body_j2(),
            IntegratorKind::Rk4,
        )
        .with_options(IntegratorOptions {
            initial_step: 120.0,
            ..IntegratorOptions::default()
        })
        .with_drag(drag);
        let epochs: Vec<f64> = (0..=100)
            .map(|idx| 10.0 * period * f64::from(idx) / 100.0)
            .collect();
        let ephemeris = propagator
            .propagate_covariance(
                LabeledCovariance6 {
                    covariance: test_covariance(),
                    frame: CovarianceFrame::Inertial,
                },
                &epochs,
                &CovariancePropagationOptions {
                    process_noise: ProcessNoise::RtnAccelerationPsd {
                        q_radial_km2_s3: 1.0e-13,
                        q_transverse_km2_s3: 1.0e-13,
                        q_normal_km2_s3: 1.0e-13,
                    },
                    output_frame: CovarianceFrame::Rtn,
                },
            )
            .expect("long-arc covariance propagation");

        for node in ephemeris.nodes() {
            let matrix = node.covariance.as_matrix();
            assert!(finite6(matrix));
            assert!(node.covariance.is_positive_semidefinite());
            for i in 0..6 {
                for j in (i + 1)..6 {
                    assert_eq!(matrix[i][j].to_bits(), matrix[j][i].to_bits());
                }
            }
            for &idx in &[0_usize, 1, 2] {
                assert!(matrix[idx][idx] >= 0.0);
                assert!(matrix[idx][idx].sqrt() < 100.0);
            }
            for &idx in &[3_usize, 4, 5] {
                assert!(matrix[idx][idx] >= 0.0);
                assert!(matrix[idx][idx].sqrt() < 1.0);
            }
        }
    }

    #[test]
    fn rejects_direction_reversal_and_bad_noise() {
        let propagator = circular_propagator();
        let covariance0 = test_covariance();
        let initial = LabeledCovariance6 {
            covariance: covariance0,
            frame: CovarianceFrame::Inertial,
        };

        let err = propagator
            .propagate_covariance(
                initial,
                &[60.0, 30.0],
                &CovariancePropagationOptions::default(),
            )
            .unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidInput(message) if message.contains("epochs_tdb_seconds"))
        );

        let err = propagator
            .propagate_covariance(
                initial,
                &[60.0],
                &CovariancePropagationOptions {
                    process_noise: ProcessNoise::RtnAccelerationPsd {
                        q_radial_km2_s3: -1.0,
                        q_transverse_km2_s3: 0.0,
                        q_normal_km2_s3: 0.0,
                    },
                    output_frame: CovarianceFrame::Inertial,
                },
            )
            .unwrap_err();
        assert!(
            matches!(err, PropagationError::InvalidInput(message) if message.contains("q_radial_km2_s3"))
        );
    }

    fn velocity_trace(covariance: Covariance6) -> f64 {
        covariance.as_matrix()[3][3] + covariance.as_matrix()[4][4] + covariance.as_matrix()[5][5]
    }

    fn identity6() -> StateTransitionMatrix {
        let mut matrix = [[0.0_f64; 6]; 6];
        for (idx, row) in matrix.iter_mut().enumerate() {
            row[idx] = 1.0;
        }
        matrix
    }

    #[allow(clippy::needless_range_loop)]
    fn covariance_from_lower(lower: Mat6) -> Covariance6 {
        let mut matrix = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let mut value = 0.0_f64;
                for k in 0..=j {
                    value += lower[i][k] * lower[j][k];
                }
                matrix[i][j] = value;
                matrix[j][i] = value;
            }
        }
        Covariance6::try_from_matrix(matrix).expect("test covariance is SPD")
    }

    fn cross_matrix(v: [f64; 3]) -> Mat3 {
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    }

    fn hill_jacobian(state: &CartesianState, mean_motion: f64) -> Mat6 {
        let rot = rtn_to_eci_rotation(state.position_array(), state.velocity_array())
            .expect("non-degenerate circular state");
        let rot_t = mat3::inline_tr(&rot);
        let omega_cross = cross_matrix([0.0, 0.0, mean_motion]);
        let mut neg_omega_cross = omega_cross;
        for row in &mut neg_omega_cross {
            for value in row {
                *value = -*value;
            }
        }
        let bottom_left = mat3::inline_rxr(&neg_omega_cross, &rot_t);
        let mut jacobian = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                jacobian[i][j] = rot_t[i][j];
                jacobian[i + 3][j] = bottom_left[i][j];
                jacobian[i + 3][j + 3] = rot_t[i][j];
            }
        }
        jacobian
    }

    fn hill_inverse_jacobian(state: &CartesianState, mean_motion: f64) -> Mat6 {
        let rot = rtn_to_eci_rotation(state.position_array(), state.velocity_array())
            .expect("non-degenerate circular state");
        let omega_cross = cross_matrix([0.0, 0.0, mean_motion]);
        let bottom_left = mat3::inline_rxr(&rot, &omega_cross);
        let mut jacobian = [[0.0_f64; 6]; 6];
        for i in 0..3 {
            for j in 0..3 {
                jacobian[i][j] = rot[i][j];
                jacobian[i + 3][j] = bottom_left[i][j];
                jacobian[i + 3][j + 3] = rot[i][j];
            }
        }
        jacobian
    }

    #[allow(clippy::needless_range_loop)]
    fn covariance_through_jacobian(covariance: Covariance6, jacobian: &Mat6) -> Covariance6 {
        let mut temp = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    temp[i][j] += jacobian[i][k] * covariance.as_matrix()[k][j];
                }
            }
        }
        let mut out = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    out[i][j] += temp[i][k] * jacobian[j][k];
                }
            }
        }
        symmetrize6(&mut out);
        Covariance6::try_from_matrix(out).expect("jacobian congruence remains PSD")
    }

    fn relative_frobenius(actual: &Mat6, expected: &Mat6) -> f64 {
        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        for i in 0..6 {
            for j in 0..6 {
                numerator += (actual[i][j] - expected[i][j]).powi(2);
                denominator += expected[i][j].powi(2);
            }
        }
        numerator.sqrt() / denominator.sqrt().max(1.0)
    }

    fn logdet(covariance: Covariance6) -> f64 {
        let matrix = SMatrix::<f64, 6, 6>::from_fn(|i, j| covariance.as_matrix()[i][j]);
        let cholesky = matrix.cholesky().expect("test covariance is SPD");
        2.0 * (0..6)
            .map(|idx| libm::log(cholesky.l()[(idx, idx)]))
            .sum::<f64>()
    }

    struct SplitMix64 {
        state: u64,
    }

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.state;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn unit_f64(&mut self) -> f64 {
            let bits = 0x3ff0_0000_0000_0000 | (self.next_u64() >> 12);
            f64::from_bits(bits) - 1.0
        }

        fn range_f64(&mut self, min: f64, max: f64) -> f64 {
            min + (max - min) * self.unit_f64()
        }
    }

    fn random_leo_state(rng: &mut SplitMix64) -> CartesianState {
        let radius = rng.range_f64(6800.0, 7600.0);
        let theta = rng.range_f64(0.0, 2.0 * PI);
        let z = rng.range_f64(-50.0, 50.0);
        let xy_radius = (radius * radius - z * z).sqrt();
        let position = [
            xy_radius * libm::cos(theta),
            xy_radius * libm::sin(theta),
            z,
        ];
        let speed = (MU_EARTH / radius).sqrt();
        let velocity = [
            -speed * libm::sin(theta),
            speed * libm::cos(theta),
            rng.range_f64(-0.02, 0.02),
        ];
        CartesianState::new(rng.range_f64(-100.0, 100.0), position, velocity)
    }

    #[allow(clippy::needless_range_loop)]
    fn random_spd_covariance(rng: &mut SplitMix64) -> Covariance6 {
        let mut lower = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                lower[i][j] = rng.range_f64(-1.0, 1.0);
            }
            lower[i][i] += if i < 3 { 1.0 } else { 1.0e-3 };
        }
        let mut covariance = covariance_from_lower(lower).into_matrix();
        for (idx, row) in covariance.iter_mut().enumerate() {
            row[idx] += 1.0e-9;
        }
        Covariance6::try_from_matrix(covariance).expect("random SPD covariance")
    }

    fn rotated_anisotropic_pair() -> (Covariance6, Covariance6) {
        let a = Covariance6::from_diagonal([100.0, 1.0, 4.0, 1.0e-6, 2.0e-6, 3.0e-6]).unwrap();
        let rot = [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]];
        let b = covariance_congruence6_checked(&a, &rot).expect("rotated covariance");
        (a, b)
    }

    #[allow(clippy::needless_range_loop)]
    fn linear_covariance_midpoint(a: &Covariance6, b: &Covariance6) -> Covariance6 {
        let mut matrix = [[0.0_f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                matrix[i][j] = 0.5 * (a.as_matrix()[i][j] + b.as_matrix()[i][j]);
            }
        }
        Covariance6::try_from_matrix(matrix).expect("linear midpoint remains PSD")
    }
}
