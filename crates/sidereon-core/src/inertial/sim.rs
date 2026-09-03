//! Synthetic IMU generation from truth states or truth increments.

use crate::astro::math::mat3::{inline_rxr, inline_tr};
use crate::astro::math::vec3::{add3, scale3, sub3};

use super::config::{gauss_markov_bias_decay, gauss_markov_bias_variance_increment, ImuSpec};
use super::frames::gravity_ecef_mps2;
use super::imu::{
    apply_calibration_error_vector, CorrectedImuIncrement, ImuBias, ImuCalibration, ImuSample,
};
use super::mechanization::{
    earth_rate_cross, earth_rotation_first_order, mid_interval_dcm, validate_increment,
};
use super::state::{mat3_mul_vec, reorthonormalize_dcm, NavState};
use super::{
    invalid_input, validate_finite, validate_nonnegative, validate_positive, InertialError,
};

/// Default deterministic seed used by [`ImuSimulationOptions::default`].
pub const DEFAULT_IMU_SIM_SEED: u64 = 0x4d59_5df4_d0f3_3173;

/// Output representation for generated IMU samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImuSimulationOutput {
    /// Return specific force and angular rate samples.
    Rate,
    /// Return integrated delta-velocity and delta-angle samples.
    #[default]
    Increment,
}

/// Optional rate-random-walk densities for the generated IMU.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuRateRandomWalk {
    /// Accelerometer rate-error random-walk density in `(m/s^2)/sqrt(s)`.
    pub accel_mps2_sqrt_s: f64,
    /// Gyroscope rate-error random-walk density in `(rad/s)/sqrt(s)`.
    pub gyro_rps_sqrt_s: f64,
}

impl ImuRateRandomWalk {
    /// Build rate-random-walk densities from SI values.
    pub const fn new(accel_mps2_sqrt_s: f64, gyro_rps_sqrt_s: f64) -> Self {
        Self {
            accel_mps2_sqrt_s,
            gyro_rps_sqrt_s,
        }
    }

    /// Validate finite, non-negative densities.
    pub fn validate(&self) -> Result<(), InertialError> {
        validate_nonnegative(self.accel_mps2_sqrt_s, "accel_mps2_sqrt_s")?;
        validate_nonnegative(self.gyro_rps_sqrt_s, "gyro_rps_sqrt_s")
    }
}

/// Options for synthetic IMU generation.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct ImuSimulationOptions {
    /// Sample representation returned by the simulator.
    pub output: ImuSimulationOutput,
    /// Seed for the deterministic SplitMix64-backed random stream.
    pub seed: u64,
    /// Initial accelerometer and gyroscope bias states.
    pub initial_bias: ImuBias,
    /// Sensor scale-factor and misalignment error applied to raw samples.
    pub calibration: ImuCalibration,
    /// Optional additive rate-random-walk error state.
    pub rate_random_walk: Option<ImuRateRandomWalk>,
}

impl Default for ImuSimulationOptions {
    fn default() -> Self {
        Self {
            output: ImuSimulationOutput::Increment,
            seed: DEFAULT_IMU_SIM_SEED,
            initial_bias: ImuBias::default(),
            calibration: ImuCalibration::default(),
            rate_random_walk: None,
        }
    }
}

impl ImuSimulationOptions {
    /// Validate option fields and calibration matrices.
    pub fn validate(&self) -> Result<(), InertialError> {
        self.initial_bias.validate()?;
        self.calibration.validate()?;
        if let Some(rate_random_walk) = self.rate_random_walk {
            rate_random_walk.validate()?;
        }
        Ok(())
    }
}

/// Batch output from synthetic IMU generation.
#[derive(Debug, Clone, PartialEq)]
pub struct SimulatedImuSequence {
    /// Generated IMU samples.
    pub samples: Vec<ImuSample>,
    /// Bias state used for each generated sample.
    pub bias_history: Vec<ImuBias>,
    /// Rate-random-walk state used for each generated sample.
    pub rate_random_walk_history: Vec<ImuBias>,
}

/// Stateful deterministic synthetic IMU generator.
#[derive(Debug, Clone, PartialEq)]
pub struct ImuSimulator {
    spec: ImuSpec,
    options: ImuSimulationOptions,
    rng: SplitMix64,
    bias: ImuBias,
    rate_random_walk: ImuBias,
}

impl ImuSimulator {
    /// Build a simulator from stochastic parameters and generation options.
    pub fn new(spec: ImuSpec, options: ImuSimulationOptions) -> Result<Self, InertialError> {
        spec.validate()?;
        options.validate()?;
        Ok(Self {
            spec,
            options,
            rng: SplitMix64::new(options.seed),
            bias: options.initial_bias,
            rate_random_walk: ImuBias::default(),
        })
    }

    /// Current Gauss-Markov bias state.
    pub const fn bias(&self) -> ImuBias {
        self.bias
    }

    /// Current rate-random-walk state.
    pub const fn rate_random_walk(&self) -> ImuBias {
        self.rate_random_walk
    }

    /// Generate one sample from a truth increment.
    pub fn sample_increment(
        &mut self,
        truth: &CorrectedImuIncrement,
    ) -> Result<ImuSample, InertialError> {
        validate_increment(truth)?;
        let dt_s = truth.dt_s;
        self.bias = propagate_bias(self.bias, &self.spec, dt_s, &mut self.rng)?;
        self.rate_random_walk = propagate_rate_random_walk(
            self.rate_random_walk,
            self.options.rate_random_walk,
            dt_s,
            &mut self.rng,
        )?;

        match self.options.output {
            ImuSimulationOutput::Rate => Ok(self.rate_sample(truth)),
            ImuSimulationOutput::Increment => Ok(self.increment_sample(truth)),
        }
    }

    fn rate_sample(&mut self, truth: &CorrectedImuIncrement) -> ImuSample {
        let dt_s = truth.dt_s;
        let inv_dt = 1.0 / dt_s;
        let accel_noise = noise_vec(self.spec.accel_vrw_mps_sqrt_s / dt_s.sqrt(), &mut self.rng);
        let gyro_noise = noise_vec(self.spec.gyro_arw_rad_sqrt_s / dt_s.sqrt(), &mut self.rng);
        let accel_rate_truth = scale3(truth.delta_velocity_mps, inv_dt);
        let gyro_rate_truth = scale3(truth.delta_theta_rad, inv_dt);
        let accel_rate = add_output_errors(
            apply_accel_calibration(accel_rate_truth, self.options.calibration),
            self.rate_random_walk.accel_mps2,
            accel_noise,
            self.bias.accel_mps2,
        );
        let gyro_rate = add_output_errors(
            apply_gyro_calibration(gyro_rate_truth, self.options.calibration),
            self.rate_random_walk.gyro_rps,
            gyro_noise,
            self.bias.gyro_rps,
        );
        ImuSample::rate(truth.t_j2000_s, accel_rate, gyro_rate)
    }

    fn increment_sample(&mut self, truth: &CorrectedImuIncrement) -> ImuSample {
        let dt_s = truth.dt_s;
        let accel_noise = noise_vec(self.spec.accel_vrw_mps_sqrt_s * dt_s.sqrt(), &mut self.rng);
        let gyro_noise = noise_vec(self.spec.gyro_arw_rad_sqrt_s * dt_s.sqrt(), &mut self.rng);
        let delta_velocity = add_output_errors(
            apply_accel_calibration(truth.delta_velocity_mps, self.options.calibration),
            scale3(self.rate_random_walk.accel_mps2, dt_s),
            accel_noise,
            scale3(self.bias.accel_mps2, dt_s),
        );
        let delta_theta = add_output_errors(
            apply_gyro_calibration(truth.delta_theta_rad, self.options.calibration),
            scale3(self.rate_random_walk.gyro_rps, dt_s),
            gyro_noise,
            scale3(self.bias.gyro_rps, dt_s),
        );
        ImuSample::increment(truth.t_j2000_s, delta_velocity, delta_theta, dt_s)
    }
}

/// Generate IMU samples from a navigation-state truth trajectory.
pub fn simulate_imu_samples(
    trajectory: &[NavState],
    spec: ImuSpec,
    options: ImuSimulationOptions,
) -> Result<SimulatedImuSequence, InertialError> {
    if trajectory.len() < 2 {
        return Err(invalid_input(
            "trajectory",
            "must contain at least two states",
        ));
    }
    for state in trajectory {
        state.validate()?;
    }
    let mut increments = Vec::with_capacity(trajectory.len() - 1);
    for pair in trajectory.windows(2) {
        increments.push(true_imu_increment_between(&pair[0], &pair[1])?);
    }
    simulate_imu_samples_from_increments(&increments, spec, options)
}

/// Generate IMU samples from coning/sculling truth increments.
pub fn simulate_imu_samples_from_increments(
    truth_increments: &[CorrectedImuIncrement],
    spec: ImuSpec,
    options: ImuSimulationOptions,
) -> Result<SimulatedImuSequence, InertialError> {
    validate_increment_sequence(truth_increments)?;
    let mut simulator = ImuSimulator::new(spec, options)?;
    let mut samples = Vec::with_capacity(truth_increments.len());
    let mut bias_history = Vec::with_capacity(truth_increments.len());
    let mut rate_random_walk_history = Vec::with_capacity(truth_increments.len());
    for truth in truth_increments {
        let sample = simulator.sample_increment(truth)?;
        samples.push(sample);
        bias_history.push(simulator.bias());
        rate_random_walk_history.push(simulator.rate_random_walk());
    }
    Ok(SimulatedImuSequence {
        samples,
        bias_history,
        rate_random_walk_history,
    })
}

/// Reconstruct the mechanization-consistent truth increment between two states.
pub fn true_imu_increment_between(
    start: &NavState,
    end: &NavState,
) -> Result<CorrectedImuIncrement, InertialError> {
    start.validate()?;
    end.validate()?;
    let dt_s = end.t_j2000_s - start.t_j2000_s;
    validate_positive(dt_s, "dt_s")?;
    validate_position_consistency(start, end, dt_s)?;
    let delta_theta_rad = delta_theta_from_attitude_change(start, end, dt_s)?;
    let c_avg = mid_interval_dcm(&start.attitude_body_to_ecef, delta_theta_rad, dt_s)?;
    let c_avg_t = inline_tr(&c_avg);
    let gravity = gravity_ecef_mps2(start.position_ecef_m)?;
    let coriolis = scale3(earth_rate_cross(start.velocity_ecef_mps), -2.0);
    let acceleration = add3(gravity, coriolis);
    let required_ecef = sub3(
        sub3(end.velocity_ecef_mps, start.velocity_ecef_mps),
        scale3(acceleration, dt_s),
    );
    let increment = CorrectedImuIncrement {
        t_j2000_s: end.t_j2000_s,
        delta_velocity_mps: mat3_mul_vec(&c_avg_t, required_ecef),
        delta_theta_rad,
        dt_s,
    };
    validate_increment(&increment)?;
    Ok(increment)
}

fn validate_position_consistency(
    start: &NavState,
    end: &NavState,
    dt_s: f64,
) -> Result<(), InertialError> {
    let average_velocity = scale3(add3(start.velocity_ecef_mps, end.velocity_ecef_mps), 0.5);
    let expected_position = add3(start.position_ecef_m, scale3(average_velocity, dt_s));
    for (axis, expected) in expected_position.iter().enumerate() {
        let actual = end.position_ecef_m[axis];
        let scale = expected.abs().max(actual.abs()).max(1.0);
        let tolerance = 64.0 * f64::EPSILON * scale;
        if (actual - expected).abs() > tolerance {
            return Err(invalid_input(
                "end.position_ecef_m",
                "must match trapezoidal position implied by endpoint velocities",
            ));
        }
    }
    Ok(())
}

fn validate_increment_sequence(
    truth_increments: &[CorrectedImuIncrement],
) -> Result<(), InertialError> {
    let mut previous_end_t: Option<f64> = None;
    for increment in truth_increments {
        validate_increment(increment)?;
        let start_t = increment.t_j2000_s - increment.dt_s;
        validate_finite(start_t, "increment.start_t_j2000_s")?;
        if let Some(previous) = previous_end_t {
            let tolerance = 1.0e-12_f64.max(1.0e-12 * increment.dt_s.abs());
            if (start_t - previous).abs() > tolerance {
                return Err(invalid_input(
                    "truth_increments",
                    "must be contiguous in time",
                ));
            }
        }
        previous_end_t = Some(increment.t_j2000_s);
    }
    Ok(())
}

fn propagate_bias(
    bias: ImuBias,
    spec: &ImuSpec,
    dt_s: f64,
    rng: &mut SplitMix64,
) -> Result<ImuBias, InertialError> {
    Ok(ImuBias {
        accel_mps2: [
            propagate_gauss_markov_axis(
                bias.accel_mps2[0],
                spec.accel_bias_instab_mps2,
                dt_s,
                spec.accel_bias_tau_s,
                rng,
            )?,
            propagate_gauss_markov_axis(
                bias.accel_mps2[1],
                spec.accel_bias_instab_mps2,
                dt_s,
                spec.accel_bias_tau_s,
                rng,
            )?,
            propagate_gauss_markov_axis(
                bias.accel_mps2[2],
                spec.accel_bias_instab_mps2,
                dt_s,
                spec.accel_bias_tau_s,
                rng,
            )?,
        ],
        gyro_rps: [
            propagate_gauss_markov_axis(
                bias.gyro_rps[0],
                spec.gyro_bias_instab_rps,
                dt_s,
                spec.gyro_bias_tau_s,
                rng,
            )?,
            propagate_gauss_markov_axis(
                bias.gyro_rps[1],
                spec.gyro_bias_instab_rps,
                dt_s,
                spec.gyro_bias_tau_s,
                rng,
            )?,
            propagate_gauss_markov_axis(
                bias.gyro_rps[2],
                spec.gyro_bias_instab_rps,
                dt_s,
                spec.gyro_bias_tau_s,
                rng,
            )?,
        ],
    })
}

fn propagate_gauss_markov_axis(
    previous: f64,
    instability: f64,
    dt_s: f64,
    tau_s: f64,
    rng: &mut SplitMix64,
) -> Result<f64, InertialError> {
    let phi = gauss_markov_bias_decay(dt_s, tau_s)?;
    let variance = gauss_markov_bias_variance_increment(instability, dt_s, tau_s)?;
    if variance == 0.0 {
        Ok(phi * previous)
    } else {
        Ok(phi * previous + variance.sqrt() * rng.standard_normal())
    }
}

fn propagate_rate_random_walk(
    current: ImuBias,
    model: Option<ImuRateRandomWalk>,
    dt_s: f64,
    rng: &mut SplitMix64,
) -> Result<ImuBias, InertialError> {
    let Some(model) = model else {
        return Ok(current);
    };
    model.validate()?;
    let sqrt_dt = dt_s.sqrt();
    Ok(ImuBias {
        accel_mps2: [
            current.accel_mps2[0] + noise_scalar(model.accel_mps2_sqrt_s * sqrt_dt, rng),
            current.accel_mps2[1] + noise_scalar(model.accel_mps2_sqrt_s * sqrt_dt, rng),
            current.accel_mps2[2] + noise_scalar(model.accel_mps2_sqrt_s * sqrt_dt, rng),
        ],
        gyro_rps: [
            current.gyro_rps[0] + noise_scalar(model.gyro_rps_sqrt_s * sqrt_dt, rng),
            current.gyro_rps[1] + noise_scalar(model.gyro_rps_sqrt_s * sqrt_dt, rng),
            current.gyro_rps[2] + noise_scalar(model.gyro_rps_sqrt_s * sqrt_dt, rng),
        ],
    })
}

fn noise_vec(std: f64, rng: &mut SplitMix64) -> [f64; 3] {
    if std == 0.0 {
        [0.0; 3]
    } else {
        [
            noise_scalar(std, rng),
            noise_scalar(std, rng),
            noise_scalar(std, rng),
        ]
    }
}

fn noise_scalar(std: f64, rng: &mut SplitMix64) -> f64 {
    if std == 0.0 {
        0.0
    } else {
        std * rng.standard_normal()
    }
}

fn add_output_errors(
    calibrated_truth: [f64; 3],
    rate_random_walk: [f64; 3],
    white_noise: [f64; 3],
    bias: [f64; 3],
) -> [f64; 3] {
    let mut value = calibrated_truth;
    if !is_zero3(rate_random_walk) {
        value = add3(value, rate_random_walk);
    }
    if !is_zero3(white_noise) {
        value = add3(value, white_noise);
    }
    if !is_zero3(bias) {
        value = add3(value, bias);
    }
    value
}

fn is_zero3(value: [f64; 3]) -> bool {
    value
        .iter()
        .all(|component| component.to_bits() == 0.0_f64.to_bits())
}

fn apply_accel_calibration(value: [f64; 3], calibration: ImuCalibration) -> [f64; 3] {
    if calibration.accel_scale_misalignment == [[0.0; 3]; 3] {
        value
    } else {
        apply_calibration_error_vector(value, &calibration.accel_scale_misalignment)
    }
}

fn apply_gyro_calibration(value: [f64; 3], calibration: ImuCalibration) -> [f64; 3] {
    if calibration.gyro_scale_misalignment == [[0.0; 3]; 3] {
        value
    } else {
        apply_calibration_error_vector(value, &calibration.gyro_scale_misalignment)
    }
}

fn delta_theta_from_attitude_change(
    start: &NavState,
    end: &NavState,
    dt_s: f64,
) -> Result<[f64; 3], InertialError> {
    let earth_inv = earth_rotation_first_order_inverse(dt_s)?;
    let previous_t = inline_tr(&start.attitude_body_to_ecef);
    let relative_raw = inline_rxr(
        &inline_rxr(&previous_t, &earth_inv),
        &end.attitude_body_to_ecef,
    );
    let relative = reorthonormalize_dcm(&relative_raw)?;
    rotation_vector_from_dcm(&relative)
}

fn earth_rotation_first_order_inverse(dt_s: f64) -> Result<[[f64; 3]; 3], InertialError> {
    let earth = earth_rotation_first_order(dt_s);
    let a = earth[0][1];
    let denominator = 1.0 + a * a;
    validate_positive(denominator, "earth_rotation_denominator")?;
    Ok([
        [1.0 / denominator, -a / denominator, 0.0],
        [a / denominator, 1.0 / denominator, 0.0],
        [0.0, 0.0, 1.0],
    ])
}

fn rotation_vector_from_dcm(dcm: &[[f64; 3]; 3]) -> Result<[f64; 3], InertialError> {
    let trace = dcm[0][0] + dcm[1][1] + dcm[2][2];
    let cos_angle = bounded_rotation_cosine(0.5 * (trace - 1.0))?;
    let angle = libm::acos(cos_angle);
    let skew_part = [
        dcm[2][1] - dcm[1][2],
        dcm[0][2] - dcm[2][0],
        dcm[1][0] - dcm[0][1],
    ];
    if angle < 1.0e-8 {
        Ok(scale3(skew_part, 0.5))
    } else if core::f64::consts::PI - angle < 1.0e-6 {
        rotation_vector_near_pi(dcm, angle)
    } else {
        Ok(scale3(skew_part, angle / (2.0 * libm::sin(angle))))
    }
}

fn bounded_rotation_cosine(value: f64) -> Result<f64, InertialError> {
    validate_finite(value, "rotation_cosine")?;
    let tolerance = 16.0 * f64::EPSILON;
    if value > 1.0 {
        if value - 1.0 <= tolerance {
            Ok(1.0)
        } else {
            Err(InertialError::DegenerateAttitude)
        }
    } else if value < -1.0 {
        if -1.0 - value <= tolerance {
            Ok(-1.0)
        } else {
            Err(InertialError::DegenerateAttitude)
        }
    } else {
        Ok(value)
    }
}

fn rotation_vector_near_pi(dcm: &[[f64; 3]; 3], angle: f64) -> Result<[f64; 3], InertialError> {
    let x2 = nonnegative_roundoff(0.5 * (dcm[0][0] + 1.0))?;
    let y2 = nonnegative_roundoff(0.5 * (dcm[1][1] + 1.0))?;
    let z2 = nonnegative_roundoff(0.5 * (dcm[2][2] + 1.0))?;
    let axis = if x2 >= y2 && x2 >= z2 {
        let x = x2.sqrt();
        if x == 0.0 {
            return Err(InertialError::DegenerateAttitude);
        }
        [
            x,
            (dcm[0][1] + dcm[1][0]) / (4.0 * x),
            (dcm[0][2] + dcm[2][0]) / (4.0 * x),
        ]
    } else if y2 >= z2 {
        let y = y2.sqrt();
        if y == 0.0 {
            return Err(InertialError::DegenerateAttitude);
        }
        [
            (dcm[0][1] + dcm[1][0]) / (4.0 * y),
            y,
            (dcm[1][2] + dcm[2][1]) / (4.0 * y),
        ]
    } else {
        let z = z2.sqrt();
        if z == 0.0 {
            return Err(InertialError::DegenerateAttitude);
        }
        [
            (dcm[0][2] + dcm[2][0]) / (4.0 * z),
            (dcm[1][2] + dcm[2][1]) / (4.0 * z),
            z,
        ]
    };
    Ok(scale3(axis, angle))
}

fn nonnegative_roundoff(value: f64) -> Result<f64, InertialError> {
    validate_finite(value, "rotation_axis_component")?;
    let tolerance = 16.0 * f64::EPSILON;
    if value >= 0.0 {
        Ok(value)
    } else if value.abs() <= tolerance {
        Ok(0.0)
    } else {
        Err(InertialError::DegenerateAttitude)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare_normal: None,
        }
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

    fn standard_normal(&mut self) -> f64 {
        if let Some(value) = self.spare_normal.take() {
            return value;
        }
        loop {
            let u = 2.0 * self.unit_f64() - 1.0;
            let v = 2.0 * self.unit_f64() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let scale = (-2.0 * libm::log(s) / s).sqrt();
                self.spare_normal = Some(v * scale);
                return u * scale;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Provenance: IMU simulation equations follow Groves, Principles of GNSS,
    //! Inertial, and Multisensor Integrated Navigation Systems, 2nd ed.,
    //! Chapter 4.4 for white noise, bias instability, and scale-factor
    //! modeling. First-order Gauss-Markov validation uses the closed-form
    //! AR(1) transition `phi = exp(-dt/tau)` and stationary variance
    //! `sigma^2`. Allan deviation validation uses the crate's IEEE 1139 and
    //! NIST SP 1065 estimators as a cross-module oracle. Increment round-trip
    //! validation uses the landed ECEF mechanization as the independent
    //! consumer of coning/sculling truth increments.

    use super::*;
    use crate::astro::constants::earth::WGS84_A_M;
    use crate::clock_stability::{overlapping_adev, AllanSeries};
    use crate::inertial::config::RANDOM_WALK_BIAS_TAU_S;
    use crate::inertial::mechanization::mechanize_ecef;
    use crate::inertial::state::mat3_identity;
    use crate::inertial::{
        ImuBias, ImuErrorModel, ImuSampleKind, MechanizationConfig, StrapdownMechanizer,
    };

    fn zero_truth_increments(count: usize, dt_s: f64) -> Vec<CorrectedImuIncrement> {
        (1..=count)
            .map(|idx| CorrectedImuIncrement {
                t_j2000_s: idx as f64 * dt_s,
                delta_velocity_mps: [0.0; 3],
                delta_theta_rad: [0.0; 3],
                dt_s,
            })
            .collect()
    }

    fn bit_hash_samples(samples: &[ImuSample]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for sample in samples {
            hash_bits(&mut hash, sample.t_j2000_s.to_bits());
            match sample.kind {
                ImuSampleKind::Rate {
                    specific_force_mps2,
                    angular_rate_rps,
                } => {
                    hash_bits(&mut hash, 0);
                    for value in specific_force_mps2 {
                        hash_bits(&mut hash, value.to_bits());
                    }
                    for value in angular_rate_rps {
                        hash_bits(&mut hash, value.to_bits());
                    }
                }
                ImuSampleKind::Increment {
                    delta_velocity_mps,
                    delta_theta_rad,
                    dt_s,
                } => {
                    hash_bits(&mut hash, 1);
                    for value in delta_velocity_mps {
                        hash_bits(&mut hash, value.to_bits());
                    }
                    for value in delta_theta_rad {
                        hash_bits(&mut hash, value.to_bits());
                    }
                    hash_bits(&mut hash, dt_s.to_bits());
                }
            }
        }
        hash
    }

    fn hash_bits(hash: &mut u64, bits: u64) {
        *hash ^= bits;
        *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }

    fn autocorrelation_lag1(values: &[f64], burn_in: usize) -> f64 {
        let retained = &values[burn_in..];
        let mean = retained.iter().sum::<f64>() / retained.len() as f64;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for pair in retained.windows(2) {
            numerator += (pair[0] - mean) * (pair[1] - mean);
            denominator += (pair[0] - mean) * (pair[0] - mean);
        }
        numerator / denominator
    }

    fn collect_bias_axis(dt_s: f64, count: usize) -> Vec<f64> {
        let spec = ImuSpec::datasheet(0.0, 0.0, 0.2, 0.0, 0.75, 0.75, None, None);
        let mut simulator = ImuSimulator::new(
            spec,
            ImuSimulationOptions {
                output: ImuSimulationOutput::Rate,
                seed: 0x0123_4567_89ab_cdef,
                ..ImuSimulationOptions::default()
            },
        )
        .expect("simulator");
        let truth = CorrectedImuIncrement {
            t_j2000_s: dt_s,
            delta_velocity_mps: [0.0; 3],
            delta_theta_rad: [0.0; 3],
            dt_s,
        };
        let mut values = Vec::with_capacity(count);
        for idx in 1..=count {
            let mut step = truth;
            step.t_j2000_s = idx as f64 * dt_s;
            simulator.sample_increment(&step).expect("sample");
            values.push(simulator.bias().accel_mps2[0]);
        }
        values
    }

    #[test]
    fn gauss_markov_discretization_tracks_closed_form_at_two_rates() {
        let count = 450_000;
        let burn_in = 2_000;
        for dt_s in [0.25_f64, 0.5_f64] {
            let values = collect_bias_axis(dt_s, count);
            let rho = autocorrelation_lag1(&values, burn_in);
            let expected = libm::exp(-dt_s / 0.75_f64);
            assert!(
                (rho - expected).abs() <= 1.2e-3,
                "dt {dt_s}, rho {rho:.17e}, expected {expected:.17e}"
            );
            let phi = gauss_markov_bias_decay(dt_s, 0.75).expect("phi");
            assert_eq!(phi.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn white_noise_density_matches_allan_oracle() {
        let dt_s = 0.02;
        let count = 1_000_000;
        let accel_density = 0.032;
        let gyro_density = 0.0045;
        let spec = ImuSpec::datasheet(accel_density, gyro_density, 0.0, 0.0, 1.0, 1.0, None, None);
        let options = ImuSimulationOptions {
            output: ImuSimulationOutput::Rate,
            seed: 0xfeed_cafe_1357_2468,
            ..ImuSimulationOptions::default()
        };
        let mut simulator = ImuSimulator::new(spec, options).expect("simulator");
        let mut accel = Vec::with_capacity(count);
        let mut gyro = Vec::with_capacity(count);
        for idx in 1..=count {
            let truth = CorrectedImuIncrement {
                t_j2000_s: idx as f64 * dt_s,
                delta_velocity_mps: [0.0; 3],
                delta_theta_rad: [0.0; 3],
                dt_s,
            };
            let sample = simulator.sample_increment(&truth).expect("sample");
            match sample.kind {
                ImuSampleKind::Rate {
                    specific_force_mps2,
                    angular_rate_rps,
                } => {
                    accel.push(specific_force_mps2[0]);
                    gyro.push(angular_rate_rps[0]);
                }
                ImuSampleKind::Increment { .. } => unreachable!(),
            }
        }
        let m = [1usize];
        let accel_adev =
            overlapping_adev(AllanSeries::FractionalFrequency(&accel), dt_s, &m).expect("accel");
        let gyro_adev =
            overlapping_adev(AllanSeries::FractionalFrequency(&gyro), dt_s, &m).expect("gyro");
        let accel_recovered = accel_adev.deviation[0] * dt_s.sqrt();
        let gyro_recovered = gyro_adev.deviation[0] * dt_s.sqrt();
        assert!(
            (accel_recovered - accel_density).abs() <= 7.0e-5,
            "accel {accel_recovered:.17e}, expected {accel_density:.17e}"
        );
        assert!(
            (gyro_recovered - gyro_density).abs() <= 1.1e-5,
            "gyro {gyro_recovered:.17e}, expected {gyro_density:.17e}"
        );
    }

    #[test]
    fn zero_noise_increments_round_trip_through_mechanization_bits() {
        let initial = NavState::new(
            0.0,
            [WGS84_A_M, 0.0, 0.0],
            [0.0, 7.0, 0.25],
            mat3_identity(),
        )
        .expect("initial");
        let truth = [
            CorrectedImuIncrement {
                t_j2000_s: 0.125,
                delta_velocity_mps: [0.001, -0.002, 0.0005],
                delta_theta_rad: [0.0002, -0.0003, 0.0004],
                dt_s: 0.125,
            },
            CorrectedImuIncrement {
                t_j2000_s: 0.25,
                delta_velocity_mps: [-0.0007, 0.0015, -0.0002],
                delta_theta_rad: [-0.0001, 0.00025, -0.00035],
                dt_s: 0.125,
            },
        ];
        let mut expected = Vec::new();
        let mut state = initial;
        for increment in truth {
            state =
                mechanize_ecef(&state, &increment, MechanizationConfig::default()).expect("truth");
            expected.push(state);
        }
        let generated = simulate_imu_samples_from_increments(
            &truth,
            ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, None, None),
            ImuSimulationOptions::default(),
        )
        .expect("generated");
        let mut mechanizer = StrapdownMechanizer::new(initial).expect("mechanizer");
        for (sample, expected_state) in generated.samples.into_iter().zip(expected) {
            let actual = *mechanizer.propagate(sample).expect("propagated");
            assert_eq!(actual, expected_state);
        }
    }

    #[test]
    fn trajectory_position_must_match_mechanization_position_update() {
        let start = NavState::new(
            10.0,
            [WGS84_A_M, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            mat3_identity(),
        )
        .expect("start");
        let end = NavState::new(
            11.0,
            [WGS84_A_M + 2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            mat3_identity(),
        )
        .expect("end");
        assert_eq!(
            true_imu_increment_between(&start, &end),
            Err(InertialError::InvalidInput {
                field: "end.position_ecef_m",
                reason: "must match trapezoidal position implied by endpoint velocities",
            })
        );
    }

    #[test]
    fn scale_misalignment_is_applied_before_correction() {
        let calibration = ImuCalibration {
            accel_scale_misalignment: [[100.0e-6, 20.0e-6, 0.0], [0.0, -50.0e-6, 0.0], [0.0; 3]],
            gyro_scale_misalignment: [[10.0e-6, 0.0, 0.0], [30.0e-6, -20.0e-6, 0.0], [0.0; 3]],
        };
        let truth = [CorrectedImuIncrement {
            t_j2000_s: 2.0,
            delta_velocity_mps: [1.0, -2.0, 0.5],
            delta_theta_rad: [0.1, -0.2, 0.05],
            dt_s: 2.0,
        }];
        let generated = simulate_imu_samples_from_increments(
            &truth,
            ImuSpec::datasheet(0.0, 0.0, 0.0, 0.0, 1.0, 1.0, None, None),
            ImuSimulationOptions {
                calibration,
                ..ImuSimulationOptions::default()
            },
        )
        .expect("generated");
        match generated.samples[0].kind {
            ImuSampleKind::Increment {
                delta_velocity_mps,
                delta_theta_rad,
                ..
            } => {
                assert_eq!(
                    delta_velocity_mps[0].to_bits(),
                    (1.0_f64 * (1.0 + 100.0e-6) + -2.0 * 20.0e-6).to_bits()
                );
                assert_eq!(
                    delta_velocity_mps[1].to_bits(),
                    (-2.0_f64 * (1.0 - 50.0e-6)).to_bits()
                );
                assert_eq!(
                    delta_theta_rad[0].to_bits(),
                    (0.1_f64 * (1.0 + 10.0e-6)).to_bits()
                );
                assert_eq!(
                    delta_theta_rad[1].to_bits(),
                    (0.1_f64 * 30.0e-6 + -0.2 * (1.0 - 20.0e-6)).to_bits()
                );
            }
            ImuSampleKind::Rate { .. } => unreachable!(),
        }
    }

    #[test]
    fn stochastic_output_errors_are_not_scale_misaligned() {
        let calibration = ImuCalibration {
            accel_scale_misalignment: [
                [200.0e-6, -40.0e-6, 15.0e-6],
                [25.0e-6, -80.0e-6, 0.0],
                [0.0; 3],
            ],
            gyro_scale_misalignment: [
                [-30.0e-6, 12.0e-6, 0.0],
                [50.0e-6, 90.0e-6, -8.0e-6],
                [0.0; 3],
            ],
        };
        let truth = [CorrectedImuIncrement {
            t_j2000_s: 8.0,
            delta_velocity_mps: [1.25, -0.75, 0.2],
            delta_theta_rad: [0.03, -0.015, 0.004],
            dt_s: 0.5,
        }];
        let spec = ImuSpec::datasheet(
            0.04,
            0.005,
            0.0,
            0.0,
            RANDOM_WALK_BIAS_TAU_S,
            RANDOM_WALK_BIAS_TAU_S,
            None,
            None,
        );
        let options = ImuSimulationOptions {
            seed: 0x77aa_19f0_cafe_babe,
            rate_random_walk: Some(ImuRateRandomWalk::new(0.002, 0.0003)),
            ..ImuSimulationOptions::default()
        };
        let base = simulate_imu_samples_from_increments(&truth, spec, options).expect("base");
        let calibrated = simulate_imu_samples_from_increments(
            &truth,
            spec,
            ImuSimulationOptions {
                calibration,
                ..options
            },
        )
        .expect("calibrated");
        let (
            ImuSampleKind::Increment {
                delta_velocity_mps: base_delta_velocity,
                delta_theta_rad: base_delta_theta,
                ..
            },
            ImuSampleKind::Increment {
                delta_velocity_mps: calibrated_delta_velocity,
                delta_theta_rad: calibrated_delta_theta,
                ..
            },
        ) = (base.samples[0].kind, calibrated.samples[0].kind)
        else {
            unreachable!()
        };
        let expected_accel_delta = [
            200.0e-6 * truth[0].delta_velocity_mps[0]
                + -40.0e-6 * truth[0].delta_velocity_mps[1]
                + 15.0e-6 * truth[0].delta_velocity_mps[2],
            25.0e-6 * truth[0].delta_velocity_mps[0] + -80.0e-6 * truth[0].delta_velocity_mps[1],
            0.0,
        ];
        let expected_gyro_delta = [
            -30.0e-6 * truth[0].delta_theta_rad[0] + 12.0e-6 * truth[0].delta_theta_rad[1],
            50.0e-6 * truth[0].delta_theta_rad[0]
                + 90.0e-6 * truth[0].delta_theta_rad[1]
                + -8.0e-6 * truth[0].delta_theta_rad[2],
            0.0,
        ];
        for axis in 0..3 {
            assert!(
                (calibrated_delta_velocity[axis]
                    - base_delta_velocity[axis]
                    - expected_accel_delta[axis])
                    .abs()
                    <= 3.0e-16,
                "axis {axis} accel actual {:.17e}, expected {:.17e}",
                calibrated_delta_velocity[axis] - base_delta_velocity[axis],
                expected_accel_delta[axis]
            );
            assert!(
                (calibrated_delta_theta[axis] - base_delta_theta[axis] - expected_gyro_delta[axis])
                    .abs()
                    <= 3.0e-17,
                "axis {axis} gyro actual {:.17e}, expected {:.17e}",
                calibrated_delta_theta[axis] - base_delta_theta[axis],
                expected_gyro_delta[axis]
            );
        }
    }

    #[test]
    fn calibration_and_bias_are_inverse_of_existing_correction_model() {
        let calibration = ImuCalibration {
            accel_scale_misalignment: [[100.0e-6, 0.0, 0.0], [0.0; 3], [0.0; 3]],
            gyro_scale_misalignment: [[50.0e-6, 0.0, 0.0], [0.0; 3], [0.0; 3]],
        };
        let bias = ImuBias {
            accel_mps2: [0.1, 0.0, 0.0],
            gyro_rps: [0.01, 0.0, 0.0],
        };
        let truth = [CorrectedImuIncrement {
            t_j2000_s: 2.0,
            delta_velocity_mps: [1.0, -2.0, 0.5],
            delta_theta_rad: [0.1, -0.2, 0.05],
            dt_s: 2.0,
        }];
        let generated = simulate_imu_samples_from_increments(
            &truth,
            ImuSpec::datasheet(
                0.0,
                0.0,
                0.0,
                0.0,
                RANDOM_WALK_BIAS_TAU_S,
                RANDOM_WALK_BIAS_TAU_S,
                None,
                None,
            ),
            ImuSimulationOptions {
                initial_bias: bias,
                calibration,
                ..ImuSimulationOptions::default()
            },
        )
        .expect("generated");
        let model = ImuErrorModel { bias, calibration };
        let corrected = model
            .correct_sample(&generated.samples[0], 0.0)
            .expect("corrected");
        for axis in 0..3 {
            assert!(
                (corrected.delta_velocity_mps[axis] - truth[0].delta_velocity_mps[axis]).abs()
                    <= 2.0e-15,
                "axis {axis} accel actual {:.17e}, expected {:.17e}",
                corrected.delta_velocity_mps[axis],
                truth[0].delta_velocity_mps[axis]
            );
            assert!(
                (corrected.delta_theta_rad[axis] - truth[0].delta_theta_rad[axis]).abs() <= 2.0e-16,
                "axis {axis} gyro actual {:.17e}, expected {:.17e}",
                corrected.delta_theta_rad[axis],
                truth[0].delta_theta_rad[axis]
            );
        }
    }

    #[test]
    fn seeded_generation_has_exact_golden_hash() {
        let truth = zero_truth_increments(32, 0.01);
        let spec = ImuSpec::datasheet(0.04, 0.005, 0.003, 0.0002, 12.0, 18.0, None, None);
        let options = ImuSimulationOptions {
            output: ImuSimulationOutput::Increment,
            seed: 0x1111_2222_3333_4444,
            rate_random_walk: Some(ImuRateRandomWalk::new(0.0003, 0.00004)),
            ..ImuSimulationOptions::default()
        };
        let first =
            simulate_imu_samples_from_increments(&truth, spec, options).expect("first sequence");
        let second =
            simulate_imu_samples_from_increments(&truth, spec, options).expect("second sequence");
        assert_eq!(first.samples, second.samples);
        assert_eq!(bit_hash_samples(&first.samples), 0x1595_8f34_f021_b56d);
    }
}
