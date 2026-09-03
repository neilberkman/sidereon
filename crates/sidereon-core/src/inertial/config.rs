//! Configuration and stochastic error parameters for inertial propagation.

use super::{invalid_input, validate_nonnegative, validate_positive, InertialError};

/// Sentinel time constant for a random-walk bias model.
pub const RANDOM_WALK_BIAS_TAU_S: f64 = f64::INFINITY;

/// Coarse IMU class used to select a built-in [`ImuSpec`] preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImuGrade {
    /// Low-cost MEMS class.
    Mems,
    /// Tactical class.
    Tactical,
    /// Navigation class.
    Navigation,
}

/// Datasheet-level IMU stochastic parameters.
///
/// Noise densities are one-sigma values in SI units per square-root second.
/// Bias instabilities are one-sigma bias magnitudes. A bias time constant of
/// [`RANDOM_WALK_BIAS_TAU_S`] selects the random-walk limit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImuSpec {
    /// Accelerometer velocity random walk in m/s per square-root second.
    pub accel_vrw_mps_sqrt_s: f64,
    /// Gyroscope angular random walk in rad per square-root second.
    pub gyro_arw_rad_sqrt_s: f64,
    /// Accelerometer bias instability in m/s^2.
    pub accel_bias_instab_mps2: f64,
    /// Gyroscope bias instability in rad/s.
    pub gyro_bias_instab_rps: f64,
    /// Accelerometer first-order Gauss-Markov bias time constant in seconds.
    pub accel_bias_tau_s: f64,
    /// Gyroscope first-order Gauss-Markov bias time constant in seconds.
    pub gyro_bias_tau_s: f64,
    /// Optional accelerometer scale instability in parts per million.
    pub accel_scale_instab_ppm: Option<f64>,
    /// Optional gyroscope scale instability in parts per million.
    pub gyro_scale_instab_ppm: Option<f64>,
}

impl ImuSpec {
    /// Build an IMU specification from datasheet values.
    #[allow(clippy::too_many_arguments)]
    pub const fn datasheet(
        accel_vrw_mps_sqrt_s: f64,
        gyro_arw_rad_sqrt_s: f64,
        accel_bias_instab_mps2: f64,
        gyro_bias_instab_rps: f64,
        accel_bias_tau_s: f64,
        gyro_bias_tau_s: f64,
        accel_scale_instab_ppm: Option<f64>,
        gyro_scale_instab_ppm: Option<f64>,
    ) -> Self {
        Self {
            accel_vrw_mps_sqrt_s,
            gyro_arw_rad_sqrt_s,
            accel_bias_instab_mps2,
            gyro_bias_instab_rps,
            accel_bias_tau_s,
            gyro_bias_tau_s,
            accel_scale_instab_ppm,
            gyro_scale_instab_ppm,
        }
    }

    /// Return the built-in preset for an IMU grade.
    pub const fn preset(grade: ImuGrade) -> Self {
        match grade {
            ImuGrade::Mems => Self::mems(),
            ImuGrade::Tactical => Self::tactical(),
            ImuGrade::Navigation => Self::navigation(),
        }
    }

    /// Representative low-cost MEMS preset.
    pub const fn mems() -> Self {
        Self::datasheet(
            5.0e-2,
            5.0e-3,
            5.0e-2,
            5.0e-4,
            3_600.0,
            3_600.0,
            Some(500.0),
            Some(500.0),
        )
    }

    /// Representative tactical preset.
    pub const fn tactical() -> Self {
        Self::datasheet(
            5.0e-3,
            2.0e-4,
            5.0e-3,
            2.0e-5,
            7_200.0,
            7_200.0,
            Some(50.0),
            Some(50.0),
        )
    }

    /// Representative navigation preset.
    pub const fn navigation() -> Self {
        Self::datasheet(
            5.0e-4,
            2.0e-5,
            2.0e-4,
            1.0e-6,
            14_400.0,
            14_400.0,
            Some(5.0),
            Some(5.0),
        )
    }

    /// Validate finite, non-negative stochastic parameters and valid bias time constants.
    pub fn validate(&self) -> Result<(), InertialError> {
        validate_nonnegative(self.accel_vrw_mps_sqrt_s, "accel_vrw_mps_sqrt_s")?;
        validate_nonnegative(self.gyro_arw_rad_sqrt_s, "gyro_arw_rad_sqrt_s")?;
        validate_nonnegative(self.accel_bias_instab_mps2, "accel_bias_instab_mps2")?;
        validate_nonnegative(self.gyro_bias_instab_rps, "gyro_bias_instab_rps")?;
        validate_bias_tau(self.accel_bias_tau_s, "accel_bias_tau_s")?;
        validate_bias_tau(self.gyro_bias_tau_s, "gyro_bias_tau_s")?;
        if let Some(scale) = self.accel_scale_instab_ppm {
            validate_nonnegative(scale, "accel_scale_instab_ppm")?;
        }
        if let Some(scale) = self.gyro_scale_instab_ppm {
            validate_nonnegative(scale, "gyro_scale_instab_ppm")?;
        }
        Ok(())
    }

    /// Accelerometer bias decay over `dt_s`.
    pub fn accel_bias_decay(&self, dt_s: f64) -> Result<f64, InertialError> {
        gauss_markov_bias_decay(dt_s, self.accel_bias_tau_s)
    }

    /// Gyroscope bias decay over `dt_s`.
    pub fn gyro_bias_decay(&self, dt_s: f64) -> Result<f64, InertialError> {
        gauss_markov_bias_decay(dt_s, self.gyro_bias_tau_s)
    }

    /// Accelerometer bias variance increment over `dt_s`.
    pub fn accel_bias_variance_increment(&self, dt_s: f64) -> Result<f64, InertialError> {
        gauss_markov_bias_variance_increment(
            self.accel_bias_instab_mps2,
            dt_s,
            self.accel_bias_tau_s,
        )
    }

    /// Gyroscope bias variance increment over `dt_s`.
    pub fn gyro_bias_variance_increment(&self, dt_s: f64) -> Result<f64, InertialError> {
        gauss_markov_bias_variance_increment(self.gyro_bias_instab_rps, dt_s, self.gyro_bias_tau_s)
    }
}

/// Strapdown coning-correction setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConingCorrection {
    /// Do not apply coning correction.
    Off,
}

/// Configuration for ECEF strapdown mechanization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct MechanizationConfig {
    /// Coning correction mode. The default is off.
    pub coning_correction: ConingCorrection,
}

impl Default for MechanizationConfig {
    fn default() -> Self {
        Self {
            coning_correction: ConingCorrection::Off,
        }
    }
}

/// First-order Gauss-Markov bias decay factor over `dt_s`.
///
/// Infinite `tau_s` returns the random-walk limit decay of `1.0`.
pub fn gauss_markov_bias_decay(dt_s: f64, tau_s: f64) -> Result<f64, InertialError> {
    validate_nonnegative(dt_s, "dt_s")?;
    validate_bias_tau(tau_s, "tau_s")?;
    if tau_s.is_infinite() {
        Ok(1.0)
    } else {
        Ok(libm::exp(-dt_s / tau_s))
    }
}

/// Bias process variance increment for a Gauss-Markov bias over `dt_s`.
///
/// For finite `tau_s`, `instability` is treated as the stationary one-sigma
/// bias. For infinite `tau_s`, it is treated as the random-walk one-sigma
/// density, giving `instability^2 * dt_s`.
pub fn gauss_markov_bias_variance_increment(
    instability: f64,
    dt_s: f64,
    tau_s: f64,
) -> Result<f64, InertialError> {
    validate_nonnegative(instability, "instability")?;
    validate_nonnegative(dt_s, "dt_s")?;
    validate_bias_tau(tau_s, "tau_s")?;
    let variance = instability * instability;
    if tau_s.is_infinite() {
        Ok(variance * dt_s)
    } else {
        Ok(variance * (1.0 - libm::exp(-2.0 * dt_s / tau_s)))
    }
}

fn validate_bias_tau(value: f64, field: &'static str) -> Result<(), InertialError> {
    if value.is_infinite() && value.is_sign_positive() {
        Ok(())
    } else if value.is_finite() {
        validate_positive(value, field)
    } else {
        Err(invalid_input(field, "must be positive or infinite"))
    }
}

#[cfg(test)]
mod tests {
    //! Provenance: IMU stochastic model follows Groves, Principles of GNSS,
    //! Inertial, and Multisensor Integrated Navigation Systems, 2nd ed.,
    //! Chapter 4.4. The test pins the closed-form Gauss-Markov transition and
    //! the random-walk limit used by the public helpers.

    use super::*;

    #[test]
    fn gauss_markov_decay_matches_closed_form() {
        let decay = gauss_markov_bias_decay(12.5, 100.0).expect("decay");
        assert_eq!(decay.to_bits(), libm::exp(-0.125_f64).to_bits());
        assert_eq!(
            gauss_markov_bias_decay(12.5, RANDOM_WALK_BIAS_TAU_S)
                .expect("random walk")
                .to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn gauss_markov_variance_matches_closed_form() {
        let increment = gauss_markov_bias_variance_increment(0.02, 10.0, 200.0).expect("variance");
        let expected = 0.02_f64 * 0.02 * (1.0 - libm::exp(-0.1_f64));
        assert_eq!(increment.to_bits(), expected.to_bits());

        let random_walk = gauss_markov_bias_variance_increment(0.02, 10.0, RANDOM_WALK_BIAS_TAU_S)
            .expect("random-walk variance");
        assert_eq!(random_walk.to_bits(), (0.02_f64 * 0.02 * 10.0).to_bits());
    }

    #[test]
    fn presets_validate() {
        ImuSpec::preset(ImuGrade::Mems).validate().expect("mems");
        ImuSpec::preset(ImuGrade::Tactical)
            .validate()
            .expect("tactical");
        ImuSpec::preset(ImuGrade::Navigation)
            .validate()
            .expect("navigation");
    }
}
