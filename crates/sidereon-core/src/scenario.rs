//! Scenario-driven synthetic GNSS observable generation.
//!
//! The module is sans I/O: callers pass a typed scenario and, when needed, an
//! already loaded ephemeris source. The output is a set of contiguous arrays
//! plus a ground-truth term ledger. RINEX OBS text is an explicit serialization
//! step through the existing in-core RINEX observation writer.

use std::cell::Cell;
use std::collections::BTreeMap;

use crate::astro::constants::earth::GM_EARTH_M3_S2;
use crate::astro::math::vec3::{norm3, sub3};
use crate::astro::time::civil::{civil_from_j2000_seconds, day_of_year, second_of_day};
use crate::astro::time::model::TimeScale;
use crate::atmosphere::{ionex_slant_delay, Ionex};
use crate::clock_stability::PowerLawNoiseType;
use crate::constants::{C_M_S, F_L1_HZ, OMEGA_E_DOT_RAD_S};
use crate::frame::{geodetic_to_itrf, itrf_to_geodetic, ItrfPositionM, Wgs84Geodetic};
use crate::id::{GnssSatelliteId, GnssSystem};
use crate::observables::{predict, ObservableEphemerisSource, ObservableState, ObservablesError};
use crate::rinex_obs::{ObsEpoch, ObsEpochTime, ObsHeader, ObsValue, PgmRunByDate, RinexObs};
use crate::spp::{
    sat_model, Corrections, EphemerisSource, KlobucharCoeffs, SatModelEnv, SppIonosphere,
    SppModelRecipe, SurfaceMet,
};
use crate::validate;

/// Version of the serialized scenario schema accepted by this module.
pub const SCENARIO_SCHEMA_VERSION: u32 = 1;

/// Engine version string that participates in the determinism contract.
pub const SCENARIO_ENGINE_VERSION: &str =
    concat!(env!("CARGO_PKG_VERSION"), ":scenario-observables-v1");

/// Default deterministic seed for examples and tests.
pub const DEFAULT_SCENARIO_SEED: u64 = 0x515c_1e7e_0b5e_a11d;

const RINEX_QUANTIZATION: f64 = 1000.0;
const FIXED_POINT_ITERS: usize = 8;
const NOISE_STREAM_OBSERVABLE: u64 = 0x0b5e_a11d_f00d_0001;
const NOISE_STREAM_RECEIVER_CLOCK: u64 = 0x0b5e_a11d_f00d_0002;
const NOISE_STREAM_SAT_CLOCK: u64 = 0x0b5e_a11d_f00d_0003;

/// Error returned by scenario validation or generation.
#[derive(Debug, thiserror::Error)]
pub enum ScenarioError {
    /// A scenario field was malformed, non-finite, or outside its domain.
    #[error("invalid scenario {field}: {reason}")]
    InvalidInput {
        /// The invalid field name.
        field: &'static str,
        /// The validation failure reason.
        reason: &'static str,
    },
    /// The scenario names external products but no source was supplied.
    #[error("scenario requires an external ephemeris source")]
    ExternalSourceRequired,
    /// The supplied external source identity does not match the scenario.
    #[error("external source identity mismatch for {field}: expected {expected}, got {actual}")]
    ExternalSourceMismatch {
        /// Field carrying the mismatched source identity.
        field: &'static str,
        /// Identity declared by the scenario.
        expected: String,
        /// Identity supplied with the loaded source.
        actual: String,
    },
    /// The scenario names an external ionosphere product but none was supplied.
    #[error("scenario requires a declared IONEX source")]
    ExternalIonosphereRequired,
    /// Supplied ionosphere product evaluation failed.
    #[error("ionosphere evaluation failed: {0}")]
    Ionosphere(String),
    /// The ephemeris source had no usable state for the satellite.
    #[error("no ephemeris state for {satellite}")]
    NoEphemeris {
        /// Satellite whose state was unavailable.
        satellite: GnssSatelliteId,
    },
    /// Observable prediction failed while forming Doppler or geometry metadata.
    #[error("observable prediction failed: {0}")]
    Observable(ObservablesError),
    /// Receiver frame conversion failed.
    #[error("frame conversion failed: {0}")]
    Frame(String),
}

/// Versioned synthetic-observable scenario.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Scenario {
    /// Schema version. Must equal [`SCENARIO_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Seed for every deterministic random stream.
    pub seed: u64,
    /// Epoch grid to simulate.
    pub epochs: ScenarioEpochRange,
    /// Receiver truth model.
    pub receiver: ScenarioReceiver,
    /// Satellite source selection.
    pub constellation: ScenarioConstellation,
    /// Per-constellation observable and carrier selection.
    pub signals: Vec<ScenarioSignal>,
    /// Independent error-budget switches and parameters.
    pub error_budget: ScenarioErrorBudget,
}

impl Scenario {
    /// Validate the schema and numeric domains.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        if self.schema_version != SCENARIO_SCHEMA_VERSION {
            return Err(invalid("schema_version", "unsupported schema version"));
        }
        self.epochs.validate()?;
        self.receiver.validate()?;
        self.constellation.validate()?;
        if self.signals.is_empty() {
            return Err(invalid("signals", "must not be empty"));
        }
        for signal in &self.signals {
            signal.validate()?;
        }
        self.error_budget.validate()?;
        Ok(())
    }

    /// Satellites named by the scenario, in scenario order.
    pub fn satellites(&self) -> Vec<GnssSatelliteId> {
        match &self.constellation {
            ScenarioConstellation::ExternalProducts { satellites, .. } => satellites.clone(),
            ScenarioConstellation::SyntheticKeplerian { satellites } => {
                satellites.iter().map(|sat| sat.satellite_id).collect()
            }
        }
    }
}

/// Inclusive-start regular epoch grid.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioEpochRange {
    /// First receive epoch, seconds since J2000 in the selected GNSS time scale.
    pub start_j2000_s: f64,
    /// Number of epochs to generate.
    pub count: usize,
    /// Spacing between epochs in seconds.
    pub cadence_s: f64,
}

impl ScenarioEpochRange {
    /// Validate finite positive cadence and non-empty count.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        validate::finite(self.start_j2000_s, "epochs.start_j2000_s").map_err(map_field)?;
        if self.count == 0 {
            return Err(invalid("epochs.count", "must be positive"));
        }
        validate::finite_positive(self.cadence_s, "epochs.cadence_s").map_err(map_field)?;
        Ok(())
    }

    /// Epoch seconds since J2000 in generation order.
    pub fn epochs_j2000_s(&self) -> Vec<f64> {
        (0..self.count)
            .map(|idx| self.start_j2000_s + self.cadence_s * idx as f64)
            .collect()
    }
}

/// Receiver truth model.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScenarioReceiver {
    /// Receiver fixed at one WGS84 geodetic position.
    StaticGeodetic {
        /// Fixed geodetic position.
        position: ScenarioGeodeticPosition,
    },
    /// Receiver position interpolated between geodetic waypoints.
    KinematicWaypoints {
        /// Waypoints keyed by offset from the first scenario epoch.
        waypoints: Vec<ScenarioReceiverWaypoint>,
    },
}

impl ScenarioReceiver {
    /// Validate receiver fields.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        match self {
            Self::StaticGeodetic { position } => position.validate(),
            Self::KinematicWaypoints { waypoints } => {
                if waypoints.len() < 2 {
                    return Err(invalid("receiver.waypoints", "must contain at least two"));
                }
                let mut previous = None;
                for waypoint in waypoints {
                    waypoint.validate()?;
                    if previous.is_some_and(|t| waypoint.offset_s <= t) {
                        return Err(invalid(
                            "receiver.waypoints.offset_s",
                            "must increase strictly",
                        ));
                    }
                    previous = Some(waypoint.offset_s);
                }
                Ok(())
            }
        }
    }
}

/// WGS84 geodetic position in radians and meters.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioGeodeticPosition {
    /// Geodetic latitude in radians.
    pub lat_rad: f64,
    /// Geodetic longitude in radians, positive east.
    pub lon_rad: f64,
    /// Ellipsoidal height in meters.
    pub height_m: f64,
}

impl ScenarioGeodeticPosition {
    /// Convert to the frame-typed geodetic value.
    pub fn to_wgs84(self) -> Result<Wgs84Geodetic, ScenarioError> {
        Wgs84Geodetic::new(self.lat_rad, self.lon_rad, self.height_m)
            .map_err(|error| ScenarioError::Frame(error.to_string()))
    }

    /// Validate the geodetic value.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        self.to_wgs84().map(|_| ())
    }
}

/// Kinematic receiver waypoint.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioReceiverWaypoint {
    /// Seconds after the scenario start epoch.
    pub offset_s: f64,
    /// Receiver position at this waypoint.
    pub position: ScenarioGeodeticPosition,
    /// Optional ECEF velocity to report at the waypoint.
    pub velocity_ecef_m_s: Option<[f64; 3]>,
}

impl ScenarioReceiverWaypoint {
    /// Validate finite time, position, and optional velocity.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        validate::finite(self.offset_s, "receiver.waypoint.offset_s").map_err(map_field)?;
        self.position.validate()?;
        if let Some(velocity) = self.velocity_ecef_m_s {
            validate::finite_vec3(velocity, "receiver.waypoint.velocity_ecef_m_s")
                .map_err(map_field)?;
        }
        Ok(())
    }
}

/// Product class for a declared external scenario input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ScenarioExternalProductKind {
    /// Precise SP3 orbit and clock product.
    Sp3,
    /// Broadcast navigation product.
    Broadcast,
    /// Two-line element product evaluated through the crate's TLE path.
    Tle,
    /// IONEX vertical-TEC grid product.
    Ionex,
}

/// Stable identity of an external product named by a scenario.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioExternalProduct {
    /// External product class.
    pub kind: ScenarioExternalProductKind,
    /// Caller-controlled stable product identifier, such as a catalog path or name.
    pub product_id: String,
    /// Core-verified content fingerprint string for the supplied product.
    pub content_digest: String,
}

impl ScenarioExternalProduct {
    /// Validate non-empty product identity fields.
    pub fn validate(&self, field: &'static str) -> Result<(), ScenarioError> {
        if self.product_id.is_empty() {
            return Err(invalid(field, "product_id must not be empty"));
        }
        if self.content_digest.is_empty() {
            return Err(invalid(field, "content_digest must not be empty"));
        }
        if !self.product_id.is_ascii() || !self.content_digest.is_ascii() {
            return Err(invalid(field, "identity fields must be ASCII"));
        }
        Ok(())
    }

    fn label(&self) -> String {
        format!(
            "{:?}:{}:{}",
            self.kind, self.product_id, self.content_digest
        )
    }
}

/// Satellite source selection for a scenario.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScenarioConstellation {
    /// Satellites are evaluated from an external loaded source.
    ExternalProducts {
        /// Declared external orbit or navigation product identity.
        source: ScenarioExternalProduct,
        /// Satellites to request from the external source.
        satellites: Vec<GnssSatelliteId>,
    },
    /// Satellites are evaluated from in-scenario Keplerian elements.
    SyntheticKeplerian {
        /// Synthetic Keplerian orbit records.
        satellites: Vec<SyntheticKeplerOrbit>,
    },
}

impl ScenarioConstellation {
    /// Validate that at least one satellite is present and all orbit records are valid.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        match self {
            Self::ExternalProducts { source, satellites } => {
                source.validate("constellation.source")?;
                if source.kind == ScenarioExternalProductKind::Ionex {
                    return Err(invalid(
                        "constellation.source.kind",
                        "must be sp3, broadcast, or tle",
                    ));
                }
                if satellites.is_empty() {
                    return Err(invalid("constellation.satellites", "must not be empty"));
                }
                Ok(())
            }
            Self::SyntheticKeplerian { satellites } => {
                if satellites.is_empty() {
                    return Err(invalid("constellation.satellites", "must not be empty"));
                }
                for satellite in satellites {
                    satellite.validate()?;
                }
                Ok(())
            }
        }
    }
}

/// One synthetic two-body Keplerian satellite.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SyntheticKeplerOrbit {
    /// Satellite identifier used in output arrays and RINEX.
    pub satellite_id: GnssSatelliteId,
    /// Semi-major axis in meters.
    pub semi_major_axis_m: f64,
    /// Eccentricity, in `[0, 1)`.
    pub eccentricity: f64,
    /// Inclination in radians.
    pub inclination_rad: f64,
    /// Right ascension of ascending node in radians.
    pub raan_rad: f64,
    /// Argument of perigee in radians.
    pub arg_perigee_rad: f64,
    /// Mean anomaly at `epoch_j2000_s`, radians.
    pub mean_anomaly_rad: f64,
    /// Element epoch, seconds since J2000.
    pub epoch_j2000_s: f64,
    /// Nominal satellite clock offset at the element epoch, seconds.
    pub clock_bias_s: f64,
    /// Nominal satellite clock drift, seconds per second.
    pub clock_drift_s_s: f64,
}

impl SyntheticKeplerOrbit {
    /// Validate orbit fields.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        validate::finite_positive(self.semi_major_axis_m, "orbit.semi_major_axis_m")
            .map_err(map_field)?;
        validate::finite(self.eccentricity, "orbit.eccentricity").map_err(map_field)?;
        if !(0.0..1.0).contains(&self.eccentricity) {
            return Err(invalid("orbit.eccentricity", "must be in [0, 1)"));
        }
        for (field, value) in [
            ("orbit.inclination_rad", self.inclination_rad),
            ("orbit.raan_rad", self.raan_rad),
            ("orbit.arg_perigee_rad", self.arg_perigee_rad),
            ("orbit.mean_anomaly_rad", self.mean_anomaly_rad),
            ("orbit.epoch_j2000_s", self.epoch_j2000_s),
            ("orbit.clock_bias_s", self.clock_bias_s),
            ("orbit.clock_drift_s_s", self.clock_drift_s_s),
        ] {
            validate::finite(value, field).map_err(map_field)?;
        }
        Ok(())
    }
}

/// Per-constellation observable and carrier selection.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioSignal {
    /// Constellation this row applies to.
    pub system: GnssSystem,
    /// RINEX code observable, for example `C1C`.
    pub code_observable: String,
    /// RINEX carrier phase observable, for example `L1C`.
    pub phase_observable: String,
    /// RINEX Doppler observable, for example `D1C`.
    pub doppler_observable: String,
    /// Carrier frequency in hertz.
    pub carrier_hz: f64,
    /// Constant carrier-phase ambiguity in cycles.
    pub carrier_phase_bias_cycles: f64,
}

impl ScenarioSignal {
    /// Build a GPS L1 C/A style signal row for one constellation.
    pub fn l1_ca(system: GnssSystem) -> Self {
        Self {
            system,
            code_observable: "C1C".to_string(),
            phase_observable: "L1C".to_string(),
            doppler_observable: "D1C".to_string(),
            carrier_hz: F_L1_HZ,
            carrier_phase_bias_cycles: 0.0,
        }
    }

    /// Validate the signal row.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        validate_code(&self.code_observable, "signal.code_observable", b'C')?;
        validate_code(&self.phase_observable, "signal.phase_observable", b'L')?;
        validate_code(&self.doppler_observable, "signal.doppler_observable", b'D')?;
        validate::finite_positive(self.carrier_hz, "signal.carrier_hz").map_err(map_field)?;
        validate::finite(
            self.carrier_phase_bias_cycles,
            "signal.carrier_phase_bias_cycles",
        )
        .map_err(map_field)?;
        Ok(())
    }
}

/// Independent error-budget switches and parameters.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioErrorBudget {
    /// Receiver clock model. Its output contributes with positive range sign.
    pub receiver_clock: ScenarioClockModel,
    /// Satellite clock model. Its output contributes with negative range sign.
    pub satellite_clock: ScenarioClockModel,
    /// Ionosphere range-delay model.
    pub ionosphere: ScenarioIonosphereModel,
    /// Troposphere range-delay model.
    pub troposphere: ScenarioTroposphereModel,
    /// Thermal noise by observable.
    pub thermal_noise: ScenarioThermalNoise,
    /// One-path specular multipath model.
    pub multipath: ScenarioSpecularMultipath,
    /// Minimum topocentric elevation in degrees.
    pub elevation_mask_deg: f64,
}

impl Default for ScenarioErrorBudget {
    fn default() -> Self {
        Self {
            receiver_clock: ScenarioClockModel::disabled(),
            satellite_clock: ScenarioClockModel::disabled(),
            ionosphere: ScenarioIonosphereModel::Off,
            troposphere: ScenarioTroposphereModel::Off,
            thermal_noise: ScenarioThermalNoise::disabled(),
            multipath: ScenarioSpecularMultipath::disabled(),
            elevation_mask_deg: 0.0,
        }
    }
}

impl ScenarioErrorBudget {
    /// Validate all budget fields.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        self.receiver_clock
            .validate("error_budget.receiver_clock")?;
        self.satellite_clock
            .validate("error_budget.satellite_clock")?;
        self.ionosphere.validate()?;
        self.troposphere.validate()?;
        self.thermal_noise.validate()?;
        self.multipath.validate()?;
        validate::finite(self.elevation_mask_deg, "error_budget.elevation_mask_deg")
            .map_err(map_field)?;
        if !(-90.0..=90.0).contains(&self.elevation_mask_deg) {
            return Err(invalid(
                "error_budget.elevation_mask_deg",
                "must be in [-90, 90]",
            ));
        }
        Ok(())
    }
}

/// Receiver or satellite clock model with IEEE-1139 coefficient order.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioClockModel {
    /// Whether this clock term is active.
    pub enabled: bool,
    /// Constant clock offset, seconds.
    pub bias_s: f64,
    /// Linear clock drift, seconds per second.
    pub drift_s_s: f64,
    /// Power-law coefficients `[h_-2, h_-1, h_0, h_1, h_2]`.
    pub power_law_coefficients: [f64; 5],
}

impl ScenarioClockModel {
    /// Disabled zero clock model.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            bias_s: 0.0,
            drift_s_s: 0.0,
            power_law_coefficients: [0.0; 5],
        }
    }

    /// Validate finite non-negative coefficient fields.
    pub fn validate(&self, prefix: &'static str) -> Result<(), ScenarioError> {
        validate::finite(self.bias_s, prefix).map_err(map_field)?;
        validate::finite(self.drift_s_s, prefix).map_err(map_field)?;
        for &coefficient in &self.power_law_coefficients {
            validate::finite(coefficient, prefix).map_err(map_field)?;
            if coefficient < 0.0 {
                return Err(invalid(
                    prefix,
                    "power-law coefficients must be non-negative",
                ));
            }
        }
        Ok(())
    }
}

/// Ionosphere model selection.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScenarioIonosphereModel {
    /// No ionosphere delay.
    Off,
    /// Broadcast Klobuchar model, in the SPP operation order.
    Klobuchar {
        /// Alpha coefficients.
        alpha: [f64; 4],
        /// Beta coefficients.
        beta: [f64; 4],
    },
    /// Supplied IONEX grid product evaluated by the existing IONEX machinery.
    SuppliedIonex {
        /// Declared IONEX product identity.
        source: ScenarioExternalProduct,
    },
}

impl ScenarioIonosphereModel {
    /// Validate model coefficients.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        match self {
            Self::Off => Ok(()),
            Self::Klobuchar { alpha, beta } => {
                validate::finite_slice(alpha, "error_budget.ionosphere.alpha")
                    .map_err(map_field)?;
                validate::finite_slice(beta, "error_budget.ionosphere.beta").map_err(map_field)?;
                Ok(())
            }
            Self::SuppliedIonex { source } => {
                source.validate("error_budget.ionosphere.source")?;
                if source.kind != ScenarioExternalProductKind::Ionex {
                    return Err(invalid(
                        "error_budget.ionosphere.source.kind",
                        "must be ionex",
                    ));
                }
                Ok(())
            }
        }
    }

    fn corrections(&self) -> Corrections {
        Corrections {
            ionosphere: matches!(self, Self::Klobuchar { .. }),
            troposphere: false,
        }
    }

    fn coefficients(&self) -> KlobucharCoeffs {
        match self {
            Self::Off => KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            },
            Self::Klobuchar { alpha, beta } => KlobucharCoeffs {
                alpha: *alpha,
                beta: *beta,
            },
            Self::SuppliedIonex { .. } => KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            },
        }
    }
}

/// Troposphere model selection.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ScenarioTroposphereModel {
    /// No troposphere delay.
    Off,
    /// Saastamoinen zenith delay with Niell mapping through the SPP path.
    SaastamoinenNiell {
        /// Surface pressure in hPa.
        pressure_hpa: f64,
        /// Surface temperature in kelvin.
        temperature_k: f64,
        /// Relative humidity fraction in `[0, 1]`.
        relative_humidity: f64,
    },
}

impl ScenarioTroposphereModel {
    /// Validate model parameters.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        match self {
            Self::Off => Ok(()),
            Self::SaastamoinenNiell {
                pressure_hpa,
                temperature_k,
                relative_humidity,
            } => {
                validate::finite_positive(*pressure_hpa, "error_budget.troposphere.pressure_hpa")
                    .map_err(map_field)?;
                validate::finite_positive(*temperature_k, "error_budget.troposphere.temperature_k")
                    .map_err(map_field)?;
                validate::finite(
                    *relative_humidity,
                    "error_budget.troposphere.relative_humidity",
                )
                .map_err(map_field)?;
                if !(0.0..=1.0).contains(relative_humidity) {
                    return Err(invalid(
                        "error_budget.troposphere.relative_humidity",
                        "must be in [0, 1]",
                    ));
                }
                Ok(())
            }
        }
    }

    fn corrections(self) -> Corrections {
        Corrections {
            ionosphere: false,
            troposphere: matches!(self, Self::SaastamoinenNiell { .. }),
        }
    }

    fn surface_met(self) -> SurfaceMet {
        match self {
            Self::Off => SurfaceMet::default(),
            Self::SaastamoinenNiell {
                pressure_hpa,
                temperature_k,
                relative_humidity,
            } => SurfaceMet {
                pressure_hpa,
                temperature_k,
                relative_humidity,
            },
        }
    }
}

/// Thermal noise standard deviations.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioThermalNoise {
    /// Whether thermal noise is active.
    pub enabled: bool,
    /// Pseudorange standard deviation in meters.
    pub pseudorange_sigma_m: f64,
    /// Carrier phase standard deviation in meters.
    pub carrier_phase_sigma_m: f64,
    /// Doppler standard deviation in hertz.
    pub doppler_sigma_hz: f64,
}

impl ScenarioThermalNoise {
    /// Disabled zero-noise model.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            pseudorange_sigma_m: 0.0,
            carrier_phase_sigma_m: 0.0,
            doppler_sigma_hz: 0.0,
        }
    }

    /// Validate non-negative finite standard deviations.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        for (field, value) in [
            (
                "error_budget.thermal_noise.pseudorange_sigma_m",
                self.pseudorange_sigma_m,
            ),
            (
                "error_budget.thermal_noise.carrier_phase_sigma_m",
                self.carrier_phase_sigma_m,
            ),
            (
                "error_budget.thermal_noise.doppler_sigma_hz",
                self.doppler_sigma_hz,
            ),
        ] {
            validate::finite(value, field).map_err(map_field)?;
            if value < 0.0 {
                return Err(invalid(field, "must be non-negative"));
            }
        }
        Ok(())
    }
}

/// One-path specular multipath model.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScenarioSpecularMultipath {
    /// Whether multipath is active.
    pub enabled: bool,
    /// Code-range amplitude in meters.
    pub amplitude_m: f64,
    /// Reflector height above the antenna phase center, meters.
    pub reflector_height_m: f64,
    /// Phase offset in radians.
    pub phase_rad: f64,
}

impl ScenarioSpecularMultipath {
    /// Disabled zero-multipath model.
    pub const fn disabled() -> Self {
        Self {
            enabled: false,
            amplitude_m: 0.0,
            reflector_height_m: 0.0,
            phase_rad: 0.0,
        }
    }

    /// Validate finite non-negative amplitude and height.
    pub fn validate(&self) -> Result<(), ScenarioError> {
        for (field, value) in [
            ("error_budget.multipath.amplitude_m", self.amplitude_m),
            (
                "error_budget.multipath.reflector_height_m",
                self.reflector_height_m,
            ),
            ("error_budget.multipath.phase_rad", self.phase_rad),
        ] {
            validate::finite(value, field).map_err(map_field)?;
        }
        if self.amplitude_m < 0.0 || self.reflector_height_m < 0.0 {
            return Err(invalid(
                "error_budget.multipath",
                "amplitude and reflector height must be non-negative",
            ));
        }
        Ok(())
    }
}

/// Receiver truth state for one epoch.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SyntheticReceiverTruth {
    /// Receive epoch, seconds since J2000.
    pub t_rx_j2000_s: f64,
    /// Receiver ECEF position in meters.
    pub position_ecef_m: [f64; 3],
    /// Receiver ECEF velocity in meters per second.
    pub velocity_ecef_m_s: [f64; 3],
    /// Receiver clock contribution in meters.
    pub clock_m: f64,
    /// Receiver clock range-rate contribution in meters per second.
    pub clock_rate_m_s: f64,
}

/// Contiguous synthetic observable arrays.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SyntheticObservableArrays {
    /// Start index of each epoch in every per-observation array.
    pub epoch_offsets: Vec<usize>,
    /// Epoch index for each observation.
    pub epoch_index: Vec<usize>,
    /// Satellite id for each observation.
    pub satellite_id: Vec<GnssSatelliteId>,
    /// Code observable label for each observation.
    pub code_observable: Vec<String>,
    /// Carrier phase observable label for each observation.
    pub phase_observable: Vec<String>,
    /// Doppler observable label for each observation.
    pub doppler_observable: Vec<String>,
    /// Carrier frequency in hertz for each observation.
    pub carrier_hz: Vec<f64>,
    /// Synthetic code pseudorange in meters.
    pub pseudorange_m: Vec<f64>,
    /// Synthetic carrier phase in cycles.
    pub carrier_phase_cycles: Vec<f64>,
    /// Synthetic Doppler shift in hertz.
    pub doppler_hz: Vec<f64>,
}

impl SyntheticObservableArrays {
    fn new(epoch_count: usize) -> Self {
        Self {
            epoch_offsets: Vec::with_capacity(epoch_count + 1),
            epoch_index: Vec::new(),
            satellite_id: Vec::new(),
            code_observable: Vec::new(),
            phase_observable: Vec::new(),
            doppler_observable: Vec::new(),
            carrier_hz: Vec::new(),
            pseudorange_m: Vec::new(),
            carrier_phase_cycles: Vec::new(),
            doppler_hz: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.pseudorange_m.len()
    }
}

/// Per-observation ground-truth term arrays.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SyntheticTermArrays {
    /// Geometric range in meters.
    pub geometric_range_m: Vec<f64>,
    /// Nominal ephemeris satellite-clock contribution in meters.
    pub satellite_clock_m: Vec<f64>,
    /// Receiver-clock contribution in meters.
    pub receiver_clock_m: Vec<f64>,
    /// Injected satellite-clock contribution in meters.
    pub satellite_clock_error_m: Vec<f64>,
    /// Ionospheric code delay in meters.
    pub ionosphere_m: Vec<f64>,
    /// Tropospheric delay in meters.
    pub troposphere_m: Vec<f64>,
    /// Thermal code noise in meters.
    pub thermal_noise_m: Vec<f64>,
    /// Specular code multipath in meters.
    pub multipath_m: Vec<f64>,
    /// Core quantization contribution in meters. This is zero before explicit serialization.
    pub quantization_m: Vec<f64>,
    /// Carrier geometric range contribution in cycles.
    pub carrier_phase_geometric_cycles: Vec<f64>,
    /// Carrier receiver-clock contribution in cycles.
    pub carrier_phase_receiver_clock_cycles: Vec<f64>,
    /// Carrier nominal satellite-clock contribution in cycles.
    pub carrier_phase_satellite_clock_cycles: Vec<f64>,
    /// Carrier injected satellite-clock contribution in cycles.
    pub carrier_phase_satellite_clock_error_cycles: Vec<f64>,
    /// Carrier ionosphere contribution in cycles.
    pub carrier_phase_ionosphere_cycles: Vec<f64>,
    /// Carrier troposphere contribution in cycles.
    pub carrier_phase_troposphere_cycles: Vec<f64>,
    /// Carrier thermal-noise contribution in cycles.
    pub carrier_phase_thermal_noise_cycles: Vec<f64>,
    /// Constant carrier-phase ambiguity contribution in cycles.
    pub carrier_phase_bias_cycles: Vec<f64>,
    /// Core carrier quantization contribution in cycles. This is zero before explicit serialization.
    pub carrier_phase_quantization_cycles: Vec<f64>,
    /// Doppler contribution from satellite line-of-sight motion in hertz.
    pub doppler_satellite_motion_hz: Vec<f64>,
    /// Doppler contribution from receiver line-of-sight motion in hertz.
    pub doppler_receiver_motion_hz: Vec<f64>,
    /// Doppler contribution from nominal ephemeris satellite-clock rate in hertz.
    pub doppler_satellite_clock_hz: Vec<f64>,
    /// Doppler contribution from receiver-clock rate in hertz.
    pub doppler_receiver_clock_hz: Vec<f64>,
    /// Doppler contribution from injected satellite-clock rate in hertz.
    pub doppler_satellite_clock_error_hz: Vec<f64>,
    /// Doppler thermal-noise contribution in hertz.
    pub doppler_thermal_noise_hz: Vec<f64>,
    /// Core Doppler quantization contribution in hertz. This is zero before explicit serialization.
    pub doppler_quantization_hz: Vec<f64>,
}

impl SyntheticTermArrays {
    fn new() -> Self {
        Self {
            geometric_range_m: Vec::new(),
            satellite_clock_m: Vec::new(),
            receiver_clock_m: Vec::new(),
            satellite_clock_error_m: Vec::new(),
            ionosphere_m: Vec::new(),
            troposphere_m: Vec::new(),
            thermal_noise_m: Vec::new(),
            multipath_m: Vec::new(),
            quantization_m: Vec::new(),
            carrier_phase_geometric_cycles: Vec::new(),
            carrier_phase_receiver_clock_cycles: Vec::new(),
            carrier_phase_satellite_clock_cycles: Vec::new(),
            carrier_phase_satellite_clock_error_cycles: Vec::new(),
            carrier_phase_ionosphere_cycles: Vec::new(),
            carrier_phase_troposphere_cycles: Vec::new(),
            carrier_phase_thermal_noise_cycles: Vec::new(),
            carrier_phase_bias_cycles: Vec::new(),
            carrier_phase_quantization_cycles: Vec::new(),
            doppler_satellite_motion_hz: Vec::new(),
            doppler_receiver_motion_hz: Vec::new(),
            doppler_satellite_clock_hz: Vec::new(),
            doppler_receiver_clock_hz: Vec::new(),
            doppler_satellite_clock_error_hz: Vec::new(),
            doppler_thermal_noise_hz: Vec::new(),
            doppler_quantization_hz: Vec::new(),
        }
    }

    fn push(&mut self, terms: ObservationTerms) {
        self.geometric_range_m.push(terms.geometric_range_m);
        self.satellite_clock_m.push(terms.satellite_clock_m);
        self.receiver_clock_m.push(terms.receiver_clock_m);
        self.satellite_clock_error_m
            .push(terms.satellite_clock_error_m);
        self.ionosphere_m.push(terms.ionosphere_m);
        self.troposphere_m.push(terms.troposphere_m);
        self.thermal_noise_m.push(terms.thermal_noise_m);
        self.multipath_m.push(terms.multipath_m);
        self.quantization_m.push(terms.quantization_m);
        self.carrier_phase_geometric_cycles
            .push(terms.carrier_phase_geometric_cycles);
        self.carrier_phase_receiver_clock_cycles
            .push(terms.carrier_phase_receiver_clock_cycles);
        self.carrier_phase_satellite_clock_cycles
            .push(terms.carrier_phase_satellite_clock_cycles);
        self.carrier_phase_satellite_clock_error_cycles
            .push(terms.carrier_phase_satellite_clock_error_cycles);
        self.carrier_phase_ionosphere_cycles
            .push(terms.carrier_phase_ionosphere_cycles);
        self.carrier_phase_troposphere_cycles
            .push(terms.carrier_phase_troposphere_cycles);
        self.carrier_phase_thermal_noise_cycles
            .push(terms.carrier_phase_thermal_noise_cycles);
        self.carrier_phase_bias_cycles
            .push(terms.carrier_phase_bias_cycles);
        self.carrier_phase_quantization_cycles
            .push(terms.carrier_phase_quantization_cycles);
        self.doppler_satellite_motion_hz
            .push(terms.doppler_satellite_motion_hz);
        self.doppler_receiver_motion_hz
            .push(terms.doppler_receiver_motion_hz);
        self.doppler_satellite_clock_hz
            .push(terms.doppler_satellite_clock_hz);
        self.doppler_receiver_clock_hz
            .push(terms.doppler_receiver_clock_hz);
        self.doppler_satellite_clock_error_hz
            .push(terms.doppler_satellite_clock_error_hz);
        self.doppler_thermal_noise_hz
            .push(terms.doppler_thermal_noise_hz);
        self.doppler_quantization_hz
            .push(terms.doppler_quantization_hz);
    }

    /// Sum the pseudorange terms for one observation in the generator order.
    pub fn pseudorange_sum_m(&self, index: usize) -> Option<f64> {
        let mut value = *self.geometric_range_m.get(index)?;
        value += *self.receiver_clock_m.get(index)?;
        value += *self.satellite_clock_m.get(index)?;
        value += *self.satellite_clock_error_m.get(index)?;
        value += *self.ionosphere_m.get(index)?;
        value += *self.troposphere_m.get(index)?;
        value += *self.thermal_noise_m.get(index)?;
        value += *self.multipath_m.get(index)?;
        value += *self.quantization_m.get(index)?;
        Some(value)
    }

    /// Sum the carrier-phase terms for one observation in the generator order.
    pub fn carrier_phase_sum_cycles(&self, index: usize) -> Option<f64> {
        let mut value = *self.carrier_phase_geometric_cycles.get(index)?;
        value += *self.carrier_phase_receiver_clock_cycles.get(index)?;
        value += *self.carrier_phase_satellite_clock_cycles.get(index)?;
        value += *self.carrier_phase_satellite_clock_error_cycles.get(index)?;
        value += *self.carrier_phase_ionosphere_cycles.get(index)?;
        value += *self.carrier_phase_troposphere_cycles.get(index)?;
        value += *self.carrier_phase_thermal_noise_cycles.get(index)?;
        value += *self.carrier_phase_bias_cycles.get(index)?;
        value += *self.carrier_phase_quantization_cycles.get(index)?;
        Some(value)
    }

    /// Sum the Doppler terms for one observation in the generator order.
    pub fn doppler_sum_hz(&self, index: usize) -> Option<f64> {
        let mut value = *self.doppler_satellite_motion_hz.get(index)?;
        value += *self.doppler_receiver_motion_hz.get(index)?;
        value += *self.doppler_satellite_clock_hz.get(index)?;
        value += *self.doppler_receiver_clock_hz.get(index)?;
        value += *self.doppler_satellite_clock_error_hz.get(index)?;
        value += *self.doppler_thermal_noise_hz.get(index)?;
        value += *self.doppler_quantization_hz.get(index)?;
        Some(value)
    }
}

/// Complete synthetic observation output.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct SyntheticObservationSet {
    /// Scenario schema version used to produce this output.
    pub schema_version: u32,
    /// Engine version used to produce this output.
    pub engine_version: String,
    /// Seed used to produce this output.
    pub seed: u64,
    /// Receiver truth, one row per epoch.
    pub receiver_truth: Vec<SyntheticReceiverTruth>,
    /// Contiguous observable arrays.
    pub observations: SyntheticObservableArrays,
    /// Per-observation term decomposition.
    pub truth_terms: SyntheticTermArrays,
}

impl SyntheticObservationSet {
    /// Number of synthetic observations.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Deterministic FNV-1a fingerprint over output bits.
    pub fn determinism_fingerprint(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        hash_u64(&mut hash, self.schema_version as u64);
        hash_u64(&mut hash, self.seed);
        for byte in self.engine_version.as_bytes() {
            hash_u64(&mut hash, u64::from(*byte));
        }
        for state in &self.receiver_truth {
            hash_f64(&mut hash, state.t_rx_j2000_s);
            for value in state.position_ecef_m {
                hash_f64(&mut hash, value);
            }
            for value in state.velocity_ecef_m_s {
                hash_f64(&mut hash, value);
            }
            hash_f64(&mut hash, state.clock_m);
            hash_f64(&mut hash, state.clock_rate_m_s);
        }
        for &epoch_offset in &self.observations.epoch_offsets {
            hash_u64(&mut hash, epoch_offset as u64);
        }
        for index in 0..self.observation_count() {
            hash_u64(&mut hash, self.observations.epoch_index[index] as u64);
            hash_u64(
                &mut hash,
                satellite_hash(self.observations.satellite_id[index]),
            );
            hash_str(&mut hash, &self.observations.code_observable[index]);
            hash_str(&mut hash, &self.observations.phase_observable[index]);
            hash_str(&mut hash, &self.observations.doppler_observable[index]);
            hash_f64(&mut hash, self.observations.carrier_hz[index]);
            hash_f64(&mut hash, self.observations.pseudorange_m[index]);
            hash_f64(&mut hash, self.observations.carrier_phase_cycles[index]);
            hash_f64(&mut hash, self.observations.doppler_hz[index]);
            hash_f64(&mut hash, self.truth_terms.geometric_range_m[index]);
            hash_f64(&mut hash, self.truth_terms.satellite_clock_m[index]);
            hash_f64(&mut hash, self.truth_terms.receiver_clock_m[index]);
            hash_f64(&mut hash, self.truth_terms.satellite_clock_error_m[index]);
            hash_f64(&mut hash, self.truth_terms.ionosphere_m[index]);
            hash_f64(&mut hash, self.truth_terms.troposphere_m[index]);
            hash_f64(&mut hash, self.truth_terms.thermal_noise_m[index]);
            hash_f64(&mut hash, self.truth_terms.multipath_m[index]);
            hash_f64(&mut hash, self.truth_terms.quantization_m[index]);
            hash_f64(
                &mut hash,
                self.truth_terms.carrier_phase_geometric_cycles[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.carrier_phase_receiver_clock_cycles[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.carrier_phase_satellite_clock_cycles[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.carrier_phase_satellite_clock_error_cycles[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.carrier_phase_ionosphere_cycles[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.carrier_phase_troposphere_cycles[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.carrier_phase_thermal_noise_cycles[index],
            );
            hash_f64(&mut hash, self.truth_terms.carrier_phase_bias_cycles[index]);
            hash_f64(
                &mut hash,
                self.truth_terms.carrier_phase_quantization_cycles[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.doppler_satellite_motion_hz[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.doppler_receiver_motion_hz[index],
            );
            hash_f64(
                &mut hash,
                self.truth_terms.doppler_satellite_clock_hz[index],
            );
            hash_f64(&mut hash, self.truth_terms.doppler_receiver_clock_hz[index]);
            hash_f64(
                &mut hash,
                self.truth_terms.doppler_satellite_clock_error_hz[index],
            );
            hash_f64(&mut hash, self.truth_terms.doppler_thermal_noise_hz[index]);
            hash_f64(&mut hash, self.truth_terms.doppler_quantization_hz[index]);
        }
        hash
    }

    /// Build an in-core RINEX observation product from the synthetic arrays.
    pub fn to_rinex_observation_file(&self) -> RinexObs {
        let obs_codes = self.rinex_obs_codes();
        let mut epochs = Vec::with_capacity(self.receiver_truth.len());
        for epoch_index in 0..self.receiver_truth.len() {
            let start = self.observations.epoch_offsets[epoch_index];
            let end = self.observations.epoch_offsets[epoch_index + 1];
            let mut sats = BTreeMap::new();
            for obs_index in start..end {
                let sat = self.observations.satellite_id[obs_index];
                let codes = obs_codes.get(&sat.system).expect("system code list exists");
                let values = sats.entry(sat).or_insert_with(|| {
                    vec![
                        ObsValue {
                            value: None,
                            lli: None,
                            ssi: None,
                        };
                        codes.len()
                    ]
                });
                set_obs_value(
                    values,
                    codes,
                    &self.observations.code_observable[obs_index],
                    round_rinex(self.observations.pseudorange_m[obs_index]),
                );
                set_obs_value(
                    values,
                    codes,
                    &self.observations.phase_observable[obs_index],
                    round_rinex(self.observations.carrier_phase_cycles[obs_index]),
                );
                set_obs_value(
                    values,
                    codes,
                    &self.observations.doppler_observable[obs_index],
                    round_rinex(self.observations.doppler_hz[obs_index]),
                );
            }
            epochs.push(ObsEpoch {
                epoch: obs_epoch_time(self.receiver_truth[epoch_index].t_rx_j2000_s),
                flag: 0,
                rcv_clock_offset_s: None,
                epoch_picoseconds: None,
                declared_record_count: sats.len(),
                special_record_count: 0,
                sats,
            });
        }

        let first = self
            .receiver_truth
            .first()
            .map(|state| (obs_epoch_time(state.t_rx_j2000_s), TimeScale::Gpst));
        let last = self
            .receiver_truth
            .last()
            .map(|state| (obs_epoch_time(state.t_rx_j2000_s), TimeScale::Gpst));
        let approx_position_m = self
            .receiver_truth
            .first()
            .map(|state| state.position_ecef_m);

        RinexObs {
            header: ObsHeader {
                version: 3.05,
                approx_position_m,
                antenna_delta_hen_m: None,
                obs_codes,
                program_run_by_date: Some(PgmRunByDate {
                    program: "SIDEREON-SCENARIO".to_string(),
                    run_by: "sidereon-core".to_string(),
                    date: "SCENARIO-V1".to_string(),
                }),
                comments: vec![
                    "Synthetic observations from sidereon-core scenario engine".to_string()
                ],
                marker_number: None,
                marker_type: Some("SIMULATED".to_string()),
                observer: None,
                agency: None,
                receiver: None,
                antenna: None,
                interval_s: self
                    .receiver_truth
                    .windows(2)
                    .next()
                    .map(|pair| pair[1].t_rx_j2000_s - pair[0].t_rx_j2000_s),
                time_of_first_obs: first,
                time_of_last_obs: last,
                n_satellites: Some(
                    self.observations
                        .satellite_id
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>()
                        .len(),
                ),
                prn_obs_counts: BTreeMap::new(),
                phase_shifts: Vec::new(),
                scale_factors: Vec::new(),
                glonass_slots: BTreeMap::new(),
                glonass_cod_phs_bis: None,
                signal_strength_unit: None,
                leap_seconds: None,
                marker_name: Some("SYNTHETIC".to_string()),
                unretained_header_labels: Vec::new(),
            },
            epochs,
            skipped_records: 0,
        }
    }

    /// Serialize the synthetic observations to RINEX OBS text.
    pub fn to_rinex_string(&self) -> String {
        self.to_rinex_observation_file().to_rinex_string()
    }

    /// Build SPP observations for one epoch from the pseudorange arrays.
    pub fn spp_observations_for_epoch(&self, epoch_index: usize) -> Vec<crate::spp::Observation> {
        let Some((&start, &end)) = self
            .observations
            .epoch_offsets
            .get(epoch_index)
            .zip(self.observations.epoch_offsets.get(epoch_index + 1))
        else {
            return Vec::new();
        };
        let mut by_sat = BTreeMap::new();
        for index in start..end {
            by_sat
                .entry(self.observations.satellite_id[index])
                .or_insert(self.observations.pseudorange_m[index]);
        }
        by_sat
            .into_iter()
            .map(|(satellite_id, pseudorange_m)| crate::spp::Observation {
                satellite_id,
                pseudorange_m,
            })
            .collect()
    }

    fn rinex_obs_codes(&self) -> BTreeMap<GnssSystem, Vec<String>> {
        let mut codes: BTreeMap<GnssSystem, Vec<String>> = BTreeMap::new();
        for index in 0..self.observation_count() {
            let system = self.observations.satellite_id[index].system;
            let list = codes.entry(system).or_default();
            push_unique_code(list, &self.observations.code_observable[index]);
            push_unique_code(list, &self.observations.phase_observable[index]);
            push_unique_code(list, &self.observations.doppler_observable[index]);
        }
        codes
    }
}

/// Already loaded ephemeris source paired with its scenario identity.
#[derive(Debug)]
pub struct DeclaredScenarioSource<'a, E: ?Sized> {
    source: &'a E,
    identity: ScenarioExternalProduct,
}

impl<'a, E: ?Sized> DeclaredScenarioSource<'a, E> {
    /// Pair an in-memory ephemeris source with the identity declared by a scenario.
    pub fn new(source: &'a E, identity: ScenarioExternalProduct) -> Self {
        Self { source, identity }
    }

    /// Declared identity for this source.
    pub fn identity(&self) -> &ScenarioExternalProduct {
        &self.identity
    }

    /// Wrapped ephemeris source.
    pub fn source(&self) -> &'a E {
        self.source
    }
}

impl<E: EphemerisSource + ?Sized> EphemerisSource for DeclaredScenarioSource<'_, E> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        self.source.position_clock_at_j2000_s(sat, t_j2000_s)
    }
}

impl<E: ObservableEphemerisSource + ?Sized> ObservableEphemerisSource
    for DeclaredScenarioSource<'_, E>
{
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        self.source.observable_state_at_j2000_s(sat, t_j2000_s)
    }
}

#[derive(Debug)]
struct SourceTranscript<'a, E> {
    source: &'a E,
    hash: Cell<u64>,
}

impl<'a, E> SourceTranscript<'a, E> {
    fn new(source: &'a E) -> Self {
        Self {
            source,
            hash: Cell::new(0xcbf2_9ce4_8422_2325),
        }
    }

    fn digest(&self) -> String {
        fingerprint_label(self.hash.get())
    }

    fn hash_query(&self, tag: u64, sat: GnssSatelliteId, t_j2000_s: f64) -> u64 {
        let mut hash = self.hash.get();
        hash_u64(&mut hash, tag);
        hash_u64(&mut hash, satellite_hash(sat));
        hash_f64(&mut hash, t_j2000_s);
        hash
    }

    fn store_hash(&self, hash: u64) {
        self.hash.set(hash);
    }
}

impl<E: EphemerisSource> EphemerisSource for SourceTranscript<'_, E> {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let result = self.source.position_clock_at_j2000_s(sat, t_j2000_s);
        let mut hash = self.hash_query(0x4550_4845_4d45_5249, sat, t_j2000_s);
        match result {
            Some((position, clock)) => {
                hash_u64(&mut hash, 1);
                for value in position {
                    hash_f64(&mut hash, value);
                }
                hash_f64(&mut hash, clock);
            }
            None => hash_u64(&mut hash, 0),
        }
        self.store_hash(hash);
        result
    }
}

impl<E: ObservableEphemerisSource> ObservableEphemerisSource for SourceTranscript<'_, E> {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let result = self.source.observable_state_at_j2000_s(sat, t_j2000_s);
        let mut hash = self.hash_query(0x4f42_5345_5256_4552, sat, t_j2000_s);
        match &result {
            Ok(state) => {
                hash_u64(&mut hash, 1);
                for value in state.position_ecef_m {
                    hash_f64(&mut hash, value);
                }
                match state.clock_s {
                    Some(clock) => {
                        hash_u64(&mut hash, 1);
                        hash_f64(&mut hash, clock);
                    }
                    None => hash_u64(&mut hash, 0),
                }
            }
            Err(_) => hash_u64(&mut hash, 0),
        }
        self.store_hash(hash);
        result
    }
}

/// Parsed IONEX product paired with its scenario identity.
#[derive(Debug, Clone, Copy)]
pub struct DeclaredIonexSource<'a> {
    product: &'a Ionex,
    identity: &'a ScenarioExternalProduct,
}

impl<'a> DeclaredIonexSource<'a> {
    /// Pair an in-memory IONEX product with the identity declared by a scenario.
    pub const fn new(product: &'a Ionex, identity: &'a ScenarioExternalProduct) -> Self {
        Self { product, identity }
    }

    /// Declared identity for this product.
    pub const fn identity(&self) -> &'a ScenarioExternalProduct {
        self.identity
    }

    /// Wrapped IONEX product.
    pub const fn product(&self) -> &'a Ionex {
        self.product
    }
}

/// Optional external media products needed by a scenario.
#[derive(Debug, Clone, Copy, Default)]
pub struct ScenarioMediaSources<'a> {
    /// Declared IONEX grid, required when the ionosphere model is `supplied_ionex`.
    pub ionex: Option<DeclaredIonexSource<'a>>,
}

/// Synthetic Keplerian ephemeris source.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticKeplerSource {
    satellites: Vec<SyntheticKeplerOrbit>,
}

impl SyntheticKeplerSource {
    /// Build a source from validated orbit records.
    pub fn new(mut satellites: Vec<SyntheticKeplerOrbit>) -> Result<Self, ScenarioError> {
        if satellites.is_empty() {
            return Err(invalid("satellites", "must not be empty"));
        }
        satellites.sort_by_key(|orbit| orbit.satellite_id);
        for satellite in &satellites {
            satellite.validate()?;
        }
        Ok(Self { satellites })
    }

    /// Orbit records in source order.
    pub fn satellites(&self) -> &[SyntheticKeplerOrbit] {
        &self.satellites
    }

    /// Evaluate one satellite state at seconds since J2000.
    pub fn state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<ObservableState> {
        let orbit = self
            .satellites
            .iter()
            .find(|orbit| orbit.satellite_id == sat)?;
        Some(kepler_state(*orbit, t_j2000_s))
    }
}

impl EphemerisSource for SyntheticKeplerSource {
    fn position_clock_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Option<([f64; 3], f64)> {
        let state = self.state_at_j2000_s(sat, t_j2000_s)?;
        Some((state.position_ecef_m, state.clock_s.unwrap_or(0.0)))
    }
}

impl ObservableEphemerisSource for SyntheticKeplerSource {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        self.state_at_j2000_s(sat, t_j2000_s)
            .ok_or(ObservablesError::NoEphemeris)
    }
}

/// Simulate a scenario that carries synthetic Keplerian satellite records.
pub fn simulate_scenario(scenario: &Scenario) -> Result<SyntheticObservationSet, ScenarioError> {
    simulate_scenario_with_media(scenario, &ScenarioMediaSources::default())
}

/// Simulate a synthetic-Keplerian scenario with declared external media products.
pub fn simulate_scenario_with_media(
    scenario: &Scenario,
    media: &ScenarioMediaSources<'_>,
) -> Result<SyntheticObservationSet, ScenarioError> {
    scenario.validate()?;
    let ScenarioConstellation::SyntheticKeplerian { satellites } = &scenario.constellation else {
        return Err(ScenarioError::ExternalSourceRequired);
    };
    let source = SyntheticKeplerSource::new(satellites.clone())?;
    simulate_resolved_scenario(scenario, &source, media)
}

/// Simulate an external-product scenario against a declared loaded ephemeris source.
pub fn simulate_scenario_with_source<E>(
    scenario: &Scenario,
    source: &DeclaredScenarioSource<'_, E>,
) -> Result<SyntheticObservationSet, ScenarioError>
where
    E: EphemerisSource + ObservableEphemerisSource,
{
    simulate_scenario_with_source_and_media(scenario, source, &ScenarioMediaSources::default())
}

/// Simulate an external-product scenario with declared ephemeris and media sources.
pub fn simulate_scenario_with_source_and_media<E>(
    scenario: &Scenario,
    source: &DeclaredScenarioSource<'_, E>,
    media: &ScenarioMediaSources<'_>,
) -> Result<SyntheticObservationSet, ScenarioError>
where
    E: EphemerisSource + ObservableEphemerisSource,
{
    scenario.validate()?;
    let ScenarioConstellation::ExternalProducts {
        source: expected, ..
    } = &scenario.constellation
    else {
        return Err(invalid(
            "constellation.kind",
            "external source supplied for synthetic constellation",
        ));
    };
    validate_identity("constellation.source", expected, source.identity())?;
    let transcript = SourceTranscript::new(source.source());
    let set = simulate_resolved_scenario(scenario, &transcript, media)?;
    let actual = transcript.digest();
    if actual != expected.content_digest {
        return Err(ScenarioError::ExternalSourceMismatch {
            field: "constellation.source.content_digest",
            expected: expected.content_digest.clone(),
            actual,
        });
    }
    Ok(set)
}

/// Compute the core-verified ephemeris transcript fingerprint for a scenario and source.
pub fn scenario_source_transcript_fingerprint<E>(
    scenario: &Scenario,
    source: &DeclaredScenarioSource<'_, E>,
    media: &ScenarioMediaSources<'_>,
) -> Result<String, ScenarioError>
where
    E: EphemerisSource + ObservableEphemerisSource,
{
    scenario.validate()?;
    let ScenarioConstellation::ExternalProducts {
        source: expected, ..
    } = &scenario.constellation
    else {
        return Err(invalid(
            "constellation.kind",
            "external source fingerprint requires external products",
        ));
    };
    if expected.kind != source.identity().kind
        || expected.product_id != source.identity().product_id
    {
        return Err(ScenarioError::ExternalSourceMismatch {
            field: "constellation.source",
            expected: expected.label(),
            actual: source.identity().label(),
        });
    }
    let transcript = SourceTranscript::new(source.source());
    let _ = simulate_resolved_scenario(scenario, &transcript, media)?;
    Ok(transcript.digest())
}

/// Compute the core-verified fingerprint for a parsed IONEX product.
pub fn ionex_content_fingerprint(ionex: &Ionex) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    hash_str(&mut hash, &ionex.to_ionex_string());
    fingerprint_label(hash)
}

fn simulate_resolved_scenario<E>(
    scenario: &Scenario,
    source: &E,
    media: &ScenarioMediaSources<'_>,
) -> Result<SyntheticObservationSet, ScenarioError>
where
    E: EphemerisSource + ObservableEphemerisSource,
{
    let satellites = scenario.satellites();
    let signal_by_system = signal_map(&scenario.signals);
    let epochs = scenario.epochs.epochs_j2000_s();
    let mut receiver_clock = ClockSynth::new(
        scenario.error_budget.receiver_clock,
        mix_seed(scenario.seed, NOISE_STREAM_RECEIVER_CLOCK),
    );
    let mut satellite_clocks: BTreeMap<GnssSatelliteId, ClockSynth> = satellites
        .iter()
        .copied()
        .map(|sat| {
            (
                sat,
                ClockSynth::new(
                    scenario.error_budget.satellite_clock,
                    mix_seed(scenario.seed ^ satellite_hash(sat), NOISE_STREAM_SAT_CLOCK),
                ),
            )
        })
        .collect();
    let mut noise = SplitMix64::new(mix_seed(scenario.seed, NOISE_STREAM_OBSERVABLE));

    let mut receiver_truth = Vec::with_capacity(epochs.len());
    let mut observations = SyntheticObservableArrays::new(epochs.len());
    let mut truth_terms = SyntheticTermArrays::new();

    for (epoch_index, &t_rx_j2000_s) in epochs.iter().enumerate() {
        let receiver = receiver_state(scenario, t_rx_j2000_s, &mut receiver_clock)?;
        receiver_truth.push(receiver);
        observations.epoch_offsets.push(observations.len());
        let epoch_context = EpochContext::new(scenario, t_rx_j2000_s);

        for &sat in &satellites {
            let Some(signals) = signal_by_system.get(&sat.system) else {
                continue;
            };
            let Some(sat_clock) = satellite_clocks.get_mut(&sat) else {
                continue;
            };
            let sat_clock_error = sat_clock.sample(t_rx_j2000_s - scenario.epochs.start_j2000_s);
            for signal in signals {
                let thermal_code_m =
                    thermal_noise_m(scenario.error_budget.thermal_noise, &mut noise);
                let thermal_phase_m =
                    thermal_phase_noise_m(scenario.error_budget.thermal_noise, &mut noise);
                let thermal_doppler_hz =
                    thermal_doppler_noise_hz(scenario.error_budget.thermal_noise, &mut noise);

                let Some((pseudorange_m, mut terms)) = synthesize_pseudorange(
                    source,
                    PseudorangeRequest {
                        sat,
                        receiver,
                        signal,
                        scenario,
                        epoch_context: &epoch_context,
                        media,
                        sat_clock_error_s: sat_clock_error.offset_s,
                        thermal_noise_m: thermal_code_m,
                    },
                )?
                else {
                    continue;
                };

                let phase_cycles = synthesize_phase_terms(&mut terms, signal, thermal_phase_m);
                let doppler_hz = synthesize_doppler_terms(
                    source,
                    sat,
                    receiver,
                    signal.carrier_hz,
                    sat_clock_error.rate_s_s,
                    thermal_doppler_hz,
                    &mut terms,
                )?;

                observations.epoch_index.push(epoch_index);
                observations.satellite_id.push(sat);
                observations
                    .code_observable
                    .push(signal.code_observable.clone());
                observations
                    .phase_observable
                    .push(signal.phase_observable.clone());
                observations
                    .doppler_observable
                    .push(signal.doppler_observable.clone());
                observations.carrier_hz.push(signal.carrier_hz);
                observations.pseudorange_m.push(pseudorange_m);
                observations.carrier_phase_cycles.push(phase_cycles);
                observations.doppler_hz.push(doppler_hz);
                truth_terms.push(terms);
            }
        }
    }
    observations.epoch_offsets.push(observations.len());

    Ok(SyntheticObservationSet {
        schema_version: scenario.schema_version,
        engine_version: SCENARIO_ENGINE_VERSION.to_string(),
        seed: scenario.seed,
        receiver_truth,
        observations,
        truth_terms,
    })
}

#[derive(Debug, Clone, Copy)]
struct EpochContext {
    t_rx_second_of_day_s: f64,
    day_of_year: f64,
    corrections: Corrections,
    klobuchar: KlobucharCoeffs,
    met: SurfaceMet,
}

impl EpochContext {
    fn new(scenario: &Scenario, t_rx_j2000_s: f64) -> Self {
        let time = obs_epoch_time(t_rx_j2000_s);
        let corrections = Corrections {
            ionosphere: scenario.error_budget.ionosphere.corrections().ionosphere,
            troposphere: scenario.error_budget.troposphere.corrections().troposphere,
        };
        Self {
            t_rx_second_of_day_s: second_of_day(
                i32::from(time.hour),
                i32::from(time.minute),
                time.second,
            ),
            day_of_year: day_of_year(
                time.year,
                i32::from(time.month),
                i32::from(time.day),
                i32::from(time.hour),
                i32::from(time.minute),
                time.second,
            ),
            corrections,
            klobuchar: scenario.error_budget.ionosphere.coefficients(),
            met: scenario.error_budget.troposphere.surface_met(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ObservationTerms {
    geometric_range_m: f64,
    satellite_clock_m: f64,
    receiver_clock_m: f64,
    satellite_clock_error_m: f64,
    ionosphere_m: f64,
    troposphere_m: f64,
    thermal_noise_m: f64,
    multipath_m: f64,
    quantization_m: f64,
    carrier_phase_geometric_cycles: f64,
    carrier_phase_receiver_clock_cycles: f64,
    carrier_phase_satellite_clock_cycles: f64,
    carrier_phase_satellite_clock_error_cycles: f64,
    carrier_phase_ionosphere_cycles: f64,
    carrier_phase_troposphere_cycles: f64,
    carrier_phase_thermal_noise_cycles: f64,
    carrier_phase_bias_cycles: f64,
    carrier_phase_quantization_cycles: f64,
    doppler_satellite_motion_hz: f64,
    doppler_receiver_motion_hz: f64,
    doppler_satellite_clock_hz: f64,
    doppler_receiver_clock_hz: f64,
    doppler_satellite_clock_error_hz: f64,
    doppler_thermal_noise_hz: f64,
    doppler_quantization_hz: f64,
}

#[derive(Debug, Clone, Copy)]
struct PseudorangeRequest<'a, 'm> {
    sat: GnssSatelliteId,
    receiver: SyntheticReceiverTruth,
    signal: &'a ScenarioSignal,
    scenario: &'a Scenario,
    epoch_context: &'a EpochContext,
    media: &'a ScenarioMediaSources<'m>,
    sat_clock_error_s: f64,
    thermal_noise_m: f64,
}

fn synthesize_pseudorange<E>(
    source: &E,
    request: PseudorangeRequest<'_, '_>,
) -> Result<Option<(f64, ObservationTerms)>, ScenarioError>
where
    E: EphemerisSource + ObservableEphemerisSource,
{
    let PseudorangeRequest {
        sat,
        receiver,
        signal,
        scenario,
        epoch_context,
        media,
        sat_clock_error_s,
        thermal_noise_m,
    } = request;
    let mask_rad = scenario.error_budget.elevation_mask_deg.to_radians();
    let mut p_meas_m = rough_range(source, sat, receiver.position_ecef_m, receiver.t_rx_j2000_s)?;
    let mut out = None;
    for _ in 0..FIXED_POINT_ITERS {
        let model = spp_model(source, sat, receiver, p_meas_m, scenario, epoch_context)?;
        if model.el_rad < mask_rad {
            return Ok(None);
        }
        let ionosphere_m =
            ionosphere_delay_m(source, sat, receiver, signal, scenario, media, model.iono_m)?;
        let multipath_m = multipath_m(
            scenario.error_budget.multipath,
            model.el_rad,
            signal.carrier_hz,
        );
        let satellite_clock_error_m = -C_M_S * sat_clock_error_s;
        let mut unrounded = model.rho_m;
        unrounded += receiver.clock_m;
        unrounded += -C_M_S * model.dt_sat_s;
        unrounded += satellite_clock_error_m;
        unrounded += ionosphere_m;
        unrounded += model.tropo_m;
        unrounded += thermal_noise_m;
        unrounded += multipath_m;
        let terms = ObservationTerms {
            geometric_range_m: model.rho_m,
            satellite_clock_m: -C_M_S * model.dt_sat_s,
            receiver_clock_m: receiver.clock_m,
            satellite_clock_error_m,
            ionosphere_m,
            troposphere_m: model.tropo_m,
            thermal_noise_m,
            multipath_m,
            quantization_m: 0.0,
            carrier_phase_geometric_cycles: 0.0,
            carrier_phase_receiver_clock_cycles: 0.0,
            carrier_phase_satellite_clock_cycles: 0.0,
            carrier_phase_satellite_clock_error_cycles: 0.0,
            carrier_phase_ionosphere_cycles: 0.0,
            carrier_phase_troposphere_cycles: 0.0,
            carrier_phase_thermal_noise_cycles: 0.0,
            carrier_phase_bias_cycles: 0.0,
            carrier_phase_quantization_cycles: 0.0,
            doppler_satellite_motion_hz: 0.0,
            doppler_receiver_motion_hz: 0.0,
            doppler_satellite_clock_hz: 0.0,
            doppler_receiver_clock_hz: 0.0,
            doppler_satellite_clock_error_hz: 0.0,
            doppler_thermal_noise_hz: 0.0,
            doppler_quantization_hz: 0.0,
        };
        p_meas_m = unrounded;
        out = Some((unrounded, terms, model.el_rad));
    }
    Ok(out.map(|(pseudorange_m, terms, _)| (pseudorange_m, terms)))
}

fn spp_model<E>(
    source: &E,
    sat: GnssSatelliteId,
    receiver: SyntheticReceiverTruth,
    p_meas_m: f64,
    scenario: &Scenario,
    epoch_context: &EpochContext,
) -> Result<crate::spp::SatModel, ScenarioError>
where
    E: EphemerisSource,
{
    let glonass_channels = BTreeMap::new();
    let env = SatModelEnv {
        eph: source,
        t_rx_j2000_s: receiver.t_rx_j2000_s,
        t_rx_second_of_day_s: epoch_context.t_rx_second_of_day_s,
        day_of_year: epoch_context.day_of_year,
        corrections: epoch_context.corrections,
        met: &epoch_context.met,
        glonass_channels: &glonass_channels,
        model: SppModelRecipe::reference(),
    };
    let ionosphere = match &scenario.error_budget.ionosphere {
        ScenarioIonosphereModel::Off => SppIonosphere::Klobuchar(KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        }),
        ScenarioIonosphereModel::Klobuchar { .. } => {
            SppIonosphere::Klobuchar(epoch_context.klobuchar)
        }
        ScenarioIonosphereModel::SuppliedIonex { .. } => {
            SppIonosphere::Klobuchar(KlobucharCoeffs {
                alpha: [0.0; 4],
                beta: [0.0; 4],
            })
        }
    };
    sat_model(
        &env,
        sat,
        receiver.position_ecef_m,
        receiver.clock_m,
        p_meas_m,
        ionosphere,
    )
    .ok_or(ScenarioError::NoEphemeris { satellite: sat })
}

fn rough_range<E>(
    source: &E,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
) -> Result<f64, ScenarioError>
where
    E: EphemerisSource,
{
    let (sat_pos, sat_clock_s) = source
        .position_clock_at_j2000_s(sat, t_rx_j2000_s)
        .ok_or(ScenarioError::NoEphemeris { satellite: sat })?;
    Ok(norm3(sub3(sat_pos, receiver_ecef_m)) - C_M_S * sat_clock_s)
}

fn source_clock_rate_s_s<E>(
    source: &E,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
) -> Result<f64, ScenarioError>
where
    E: EphemerisSource,
{
    let (_, plus) = source
        .position_clock_at_j2000_s(sat, t_j2000_s + 0.5)
        .ok_or(ScenarioError::NoEphemeris { satellite: sat })?;
    let (_, minus) = source
        .position_clock_at_j2000_s(sat, t_j2000_s - 0.5)
        .ok_or(ScenarioError::NoEphemeris { satellite: sat })?;
    Ok((plus - minus) / 1.0)
}

fn ionosphere_delay_m<E>(
    source: &E,
    sat: GnssSatelliteId,
    receiver: SyntheticReceiverTruth,
    signal: &ScenarioSignal,
    scenario: &Scenario,
    media: &ScenarioMediaSources<'_>,
    klobuchar_iono_m: f64,
) -> Result<f64, ScenarioError>
where
    E: ObservableEphemerisSource,
{
    let ScenarioIonosphereModel::SuppliedIonex { source: expected } =
        &scenario.error_budget.ionosphere
    else {
        return Ok(klobuchar_iono_m);
    };
    let Some(ionex) = media.ionex else {
        return Err(ScenarioError::ExternalIonosphereRequired);
    };
    validate_identity("error_budget.ionosphere.source", expected, ionex.identity())?;
    let actual = ionex_content_fingerprint(ionex.product());
    if actual != expected.content_digest {
        return Err(ScenarioError::ExternalSourceMismatch {
            field: "error_budget.ionosphere.source.content_digest",
            expected: expected.content_digest.clone(),
            actual,
        });
    }
    let prediction = predict(
        source,
        sat,
        receiver.position_ecef_m,
        receiver.t_rx_j2000_s,
        crate::observables::PredictOptions {
            carrier_hz: signal.carrier_hz,
            ..crate::observables::PredictOptions::default()
        },
    )
    .map_err(ScenarioError::Observable)?;
    let receiver_geodetic = receiver_geodetic(receiver.position_ecef_m)?;
    let epoch_j2000_s = rounded_j2000_seconds(receiver.t_rx_j2000_s)?;
    ionex_slant_delay(
        ionex.product(),
        receiver_geodetic,
        prediction.elevation_deg.to_radians(),
        prediction.azimuth_deg.to_radians(),
        epoch_j2000_s,
        signal.carrier_hz,
    )
    .map_err(|error| ScenarioError::Ionosphere(error.to_string()))
}

fn synthesize_phase_terms(
    terms: &mut ObservationTerms,
    signal: &ScenarioSignal,
    thermal_phase_m: f64,
) -> f64 {
    let lambda = wavelength_m(signal.carrier_hz);
    let mut phase_cycles = terms.geometric_range_m / lambda;
    terms.carrier_phase_geometric_cycles = phase_cycles;
    let value = terms.receiver_clock_m / lambda;
    terms.carrier_phase_receiver_clock_cycles = value;
    phase_cycles += value;
    let value = terms.satellite_clock_m / lambda;
    terms.carrier_phase_satellite_clock_cycles = value;
    phase_cycles += value;
    let value = terms.satellite_clock_error_m / lambda;
    terms.carrier_phase_satellite_clock_error_cycles = value;
    phase_cycles += value;
    let value = -terms.ionosphere_m / lambda;
    terms.carrier_phase_ionosphere_cycles = value;
    phase_cycles += value;
    let value = terms.troposphere_m / lambda;
    terms.carrier_phase_troposphere_cycles = value;
    phase_cycles += value;
    let value = thermal_phase_m / lambda;
    terms.carrier_phase_thermal_noise_cycles = value;
    phase_cycles += value;
    terms.carrier_phase_bias_cycles = signal.carrier_phase_bias_cycles;
    phase_cycles += signal.carrier_phase_bias_cycles;
    terms.carrier_phase_quantization_cycles = 0.0;
    phase_cycles
}

fn synthesize_doppler_terms<E>(
    source: &E,
    sat: GnssSatelliteId,
    receiver: SyntheticReceiverTruth,
    carrier_hz: f64,
    sat_clock_error_rate_s_s: f64,
    thermal_doppler_hz: f64,
    terms: &mut ObservationTerms,
) -> Result<f64, ScenarioError>
where
    E: EphemerisSource + ObservableEphemerisSource,
{
    let prediction = predict(
        source,
        sat,
        receiver.position_ecef_m,
        receiver.t_rx_j2000_s,
        crate::observables::PredictOptions {
            carrier_hz,
            ..crate::observables::PredictOptions::default()
        },
    )
    .map_err(ScenarioError::Observable)?;
    let receiver_range_rate = dot3(prediction.los_unit, receiver.velocity_ecef_m_s);
    let mut doppler_hz = prediction.doppler_hz;
    terms.doppler_satellite_motion_hz = doppler_hz;
    let value = receiver_range_rate * carrier_hz / C_M_S;
    terms.doppler_receiver_motion_hz = value;
    doppler_hz += value;
    let value = source_clock_rate_s_s(source, sat, receiver.t_rx_j2000_s)? * carrier_hz;
    terms.doppler_satellite_clock_hz = value;
    doppler_hz += value;
    let value = -receiver.clock_rate_m_s * carrier_hz / C_M_S;
    terms.doppler_receiver_clock_hz = value;
    doppler_hz += value;
    let value = sat_clock_error_rate_s_s * carrier_hz;
    terms.doppler_satellite_clock_error_hz = value;
    doppler_hz += value;
    terms.doppler_thermal_noise_hz = thermal_doppler_hz;
    doppler_hz += thermal_doppler_hz;
    terms.doppler_quantization_hz = 0.0;
    Ok(doppler_hz)
}

fn receiver_state(
    scenario: &Scenario,
    t_rx_j2000_s: f64,
    clock: &mut ClockSynth,
) -> Result<SyntheticReceiverTruth, ScenarioError> {
    let offset_s = t_rx_j2000_s - scenario.epochs.start_j2000_s;
    let (position_ecef_m, velocity_ecef_m_s) = match &scenario.receiver {
        ScenarioReceiver::StaticGeodetic { position } => {
            let ecef = geodetic_to_itrf(position.to_wgs84()?)
                .map_err(|error| ScenarioError::Frame(error.to_string()))?
                .as_array();
            (ecef, [0.0; 3])
        }
        ScenarioReceiver::KinematicWaypoints { waypoints } => {
            interpolate_receiver_waypoints(waypoints, offset_s)?
        }
    };
    let clock_sample = clock.sample(offset_s);
    Ok(SyntheticReceiverTruth {
        t_rx_j2000_s,
        position_ecef_m,
        velocity_ecef_m_s,
        clock_m: C_M_S * clock_sample.offset_s,
        clock_rate_m_s: C_M_S * clock_sample.rate_s_s,
    })
}

fn interpolate_receiver_waypoints(
    waypoints: &[ScenarioReceiverWaypoint],
    offset_s: f64,
) -> Result<([f64; 3], [f64; 3]), ScenarioError> {
    let mut segment = waypoints
        .windows(2)
        .last()
        .expect("validated waypoint count");
    for pair in waypoints.windows(2) {
        if offset_s >= pair[0].offset_s && offset_s <= pair[1].offset_s {
            segment = pair;
            break;
        }
    }
    let start = segment[0];
    let end = segment[1];
    let p0 = geodetic_to_itrf(start.position.to_wgs84()?)
        .map_err(|error| ScenarioError::Frame(error.to_string()))?
        .as_array();
    let p1 = geodetic_to_itrf(end.position.to_wgs84()?)
        .map_err(|error| ScenarioError::Frame(error.to_string()))?
        .as_array();
    let dt = end.offset_s - start.offset_s;
    let u = ((offset_s - start.offset_s) / dt).clamp(0.0, 1.0);
    let position = [
        p0[0] + (p1[0] - p0[0]) * u,
        p0[1] + (p1[1] - p0[1]) * u,
        p0[2] + (p1[2] - p0[2]) * u,
    ];
    let velocity = start.velocity_ecef_m_s.unwrap_or([
        (p1[0] - p0[0]) / dt,
        (p1[1] - p0[1]) / dt,
        (p1[2] - p0[2]) / dt,
    ]);
    Ok((position, velocity))
}

fn kepler_state(orbit: SyntheticKeplerOrbit, t_j2000_s: f64) -> ObservableState {
    let dt = t_j2000_s - orbit.epoch_j2000_s;
    let n_rad_s = (GM_EARTH_M3_S2
        / (orbit.semi_major_axis_m * orbit.semi_major_axis_m * orbit.semi_major_axis_m))
        .sqrt();
    let mean_anomaly = orbit.mean_anomaly_rad + n_rad_s * dt;
    let eccentric_anomaly = solve_kepler(mean_anomaly, orbit.eccentricity);
    let cos_e = libm::cos(eccentric_anomaly);
    let sin_e = libm::sin(eccentric_anomaly);
    let a = orbit.semi_major_axis_m;
    let e = orbit.eccentricity;
    let one_minus_e2 = 1.0 - e * e;
    let x_orb = a * (cos_e - e);
    let y_orb = a * one_minus_e2.sqrt() * sin_e;
    let eci = perifocal_to_inertial(
        [x_orb, y_orb, 0.0],
        orbit.raan_rad,
        orbit.inclination_rad,
        orbit.arg_perigee_rad,
    );
    let theta = OMEGA_E_DOT_RAD_S * dt;
    let cos_t = libm::cos(theta);
    let sin_t = libm::sin(theta);
    let ecef = [
        cos_t * eci[0] + sin_t * eci[1],
        -sin_t * eci[0] + cos_t * eci[1],
        eci[2],
    ];
    ObservableState {
        position_ecef_m: ecef,
        clock_s: Some(orbit.clock_bias_s + orbit.clock_drift_s_s * dt),
    }
}

fn solve_kepler(mean_anomaly_rad: f64, eccentricity: f64) -> f64 {
    let mut e_anomaly = mean_anomaly_rad;
    for _ in 0..16 {
        let sin_e = libm::sin(e_anomaly);
        let cos_e = libm::cos(e_anomaly);
        let f = e_anomaly - eccentricity * sin_e - mean_anomaly_rad;
        let fp = 1.0 - eccentricity * cos_e;
        e_anomaly -= f / fp;
    }
    e_anomaly
}

fn perifocal_to_inertial(
    position: [f64; 3],
    raan_rad: f64,
    inclination_rad: f64,
    arg_perigee_rad: f64,
) -> [f64; 3] {
    let cos_o = libm::cos(raan_rad);
    let sin_o = libm::sin(raan_rad);
    let cos_i = libm::cos(inclination_rad);
    let sin_i = libm::sin(inclination_rad);
    let cos_w = libm::cos(arg_perigee_rad);
    let sin_w = libm::sin(arg_perigee_rad);
    let r11 = cos_o * cos_w - sin_o * sin_w * cos_i;
    let r12 = -cos_o * sin_w - sin_o * cos_w * cos_i;
    let r21 = sin_o * cos_w + cos_o * sin_w * cos_i;
    let r22 = -sin_o * sin_w + cos_o * cos_w * cos_i;
    let r31 = sin_w * sin_i;
    let r32 = cos_w * sin_i;
    [
        r11 * position[0] + r12 * position[1],
        r21 * position[0] + r22 * position[1],
        r31 * position[0] + r32 * position[1],
    ]
}

#[derive(Debug, Clone, Copy)]
struct ClockSample {
    offset_s: f64,
    rate_s_s: f64,
}

#[derive(Debug, Clone, Copy)]
struct ClockSynth {
    model: ScenarioClockModel,
    rng: SplitMix64,
    last_offset_s: Option<f64>,
    phase_noise_s: f64,
    frequency_noise: f64,
    flicker_frequency: f64,
}

impl ClockSynth {
    fn new(model: ScenarioClockModel, seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let flicker = if model.enabled {
            let coeff = coefficient(model, PowerLawNoiseType::FlickerFM);
            coeff.sqrt() * rng.standard_normal()
        } else {
            0.0
        };
        Self {
            model,
            rng,
            last_offset_s: None,
            phase_noise_s: 0.0,
            frequency_noise: 0.0,
            flicker_frequency: flicker,
        }
    }

    fn sample(&mut self, offset_s: f64) -> ClockSample {
        if !self.model.enabled {
            return ClockSample {
                offset_s: 0.0,
                rate_s_s: 0.0,
            };
        }
        let dt_s = self
            .last_offset_s
            .map_or(0.0, |previous| (offset_s - previous).max(0.0));
        self.last_offset_s = Some(offset_s);
        let mut rate_s_s = self.model.drift_s_s + self.flicker_frequency + self.frequency_noise;
        if dt_s > 0.0 {
            let rw_fm = coefficient(self.model, PowerLawNoiseType::RandomWalkFM);
            if rw_fm > 0.0 {
                self.frequency_noise += (rw_fm * dt_s).sqrt() * self.rng.standard_normal();
            }
            rate_s_s = self.model.drift_s_s + self.flicker_frequency + self.frequency_noise;
            let white_fm = coefficient(self.model, PowerLawNoiseType::WhiteFM);
            if white_fm > 0.0 {
                let phase_step_s = (white_fm * dt_s).sqrt() * self.rng.standard_normal();
                self.phase_noise_s += phase_step_s;
                rate_s_s += phase_step_s / dt_s;
            }
            self.phase_noise_s += self.frequency_noise * dt_s;
        }
        let pm_dt = dt_s.max(1.0);
        let flicker_pm = coefficient(self.model, PowerLawNoiseType::FlickerPM);
        let white_pm = coefficient(self.model, PowerLawNoiseType::WhitePM);
        let pm_noise = if flicker_pm == 0.0 && white_pm == 0.0 {
            0.0
        } else {
            (flicker_pm / pm_dt).sqrt() * self.rng.standard_normal()
                + (white_pm / (pm_dt * pm_dt)).sqrt() * self.rng.standard_normal()
        };
        let offset_s = self.model.bias_s
            + self.model.drift_s_s * offset_s
            + self.flicker_frequency * offset_s
            + self.phase_noise_s
            + pm_noise;
        ClockSample { offset_s, rate_s_s }
    }
}

fn coefficient(model: ScenarioClockModel, noise_type: PowerLawNoiseType) -> f64 {
    model.power_law_coefficients[noise_type.coefficient_index()]
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SplitMix64 {
    state: u64,
    spare_normal: Option<f64>,
}

impl SplitMix64 {
    const fn new(seed: u64) -> Self {
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

fn thermal_noise_m(model: ScenarioThermalNoise, rng: &mut SplitMix64) -> f64 {
    if model.enabled && model.pseudorange_sigma_m > 0.0 {
        model.pseudorange_sigma_m * rng.standard_normal()
    } else {
        0.0
    }
}

fn thermal_phase_noise_m(model: ScenarioThermalNoise, rng: &mut SplitMix64) -> f64 {
    if model.enabled && model.carrier_phase_sigma_m > 0.0 {
        model.carrier_phase_sigma_m * rng.standard_normal()
    } else {
        0.0
    }
}

fn thermal_doppler_noise_hz(model: ScenarioThermalNoise, rng: &mut SplitMix64) -> f64 {
    if model.enabled && model.doppler_sigma_hz > 0.0 {
        model.doppler_sigma_hz * rng.standard_normal()
    } else {
        0.0
    }
}

fn multipath_m(model: ScenarioSpecularMultipath, elevation_rad: f64, carrier_hz: f64) -> f64 {
    if !model.enabled || model.amplitude_m == 0.0 {
        return 0.0;
    }
    let lambda = wavelength_m(carrier_hz);
    let phase = 4.0 * core::f64::consts::PI * model.reflector_height_m * libm::sin(elevation_rad)
        / lambda
        + model.phase_rad;
    model.amplitude_m * libm::cos(phase)
}

fn wavelength_m(carrier_hz: f64) -> f64 {
    C_M_S / carrier_hz
}

fn round_rinex(value: f64) -> f64 {
    (value * RINEX_QUANTIZATION).round() / RINEX_QUANTIZATION
}

fn obs_epoch_time(t_j2000_s: f64) -> ObsEpochTime {
    let whole = t_j2000_s.floor() as i64;
    let frac = t_j2000_s - whole as f64;
    let (year, month, day, hour, minute, second) = civil_from_j2000_seconds(whole);
    ObsEpochTime {
        year: year as i32,
        month: month as u8,
        day: day as u8,
        hour: hour as u8,
        minute: minute as u8,
        second: second as f64 + frac,
    }
}

fn signal_map(signals: &[ScenarioSignal]) -> BTreeMap<GnssSystem, Vec<ScenarioSignal>> {
    let mut out: BTreeMap<GnssSystem, Vec<ScenarioSignal>> = BTreeMap::new();
    for signal in signals {
        out.entry(signal.system).or_default().push(signal.clone());
    }
    out
}

fn validate_identity(
    field: &'static str,
    expected: &ScenarioExternalProduct,
    actual: &ScenarioExternalProduct,
) -> Result<(), ScenarioError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ScenarioError::ExternalSourceMismatch {
            field,
            expected: expected.label(),
            actual: actual.label(),
        })
    }
}

fn receiver_geodetic(position_ecef_m: [f64; 3]) -> Result<Wgs84Geodetic, ScenarioError> {
    let position = ItrfPositionM::new(position_ecef_m[0], position_ecef_m[1], position_ecef_m[2])
        .map_err(|error| ScenarioError::Frame(error.to_string()))?;
    itrf_to_geodetic(position).map_err(|error| ScenarioError::Frame(error.to_string()))
}

fn rounded_j2000_seconds(t_rx_j2000_s: f64) -> Result<i64, ScenarioError> {
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_field)?;
    let rounded = t_rx_j2000_s.round();
    if !rounded.is_finite() || rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return Err(invalid("t_rx_j2000_s", "must round to i64 seconds"));
    }
    Ok(rounded as i64)
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn set_obs_value(values: &mut [ObsValue], codes: &[String], code: &str, value: f64) {
    if let Some(index) = codes.iter().position(|candidate| candidate == code) {
        values[index] = ObsValue {
            value: Some(value),
            lli: None,
            ssi: None,
        };
    }
}

fn push_unique_code(codes: &mut Vec<String>, code: &str) {
    if !codes.iter().any(|candidate| candidate == code) {
        codes.push(code.to_string());
    }
}

fn validate_code(code: &str, field: &'static str, first: u8) -> Result<(), ScenarioError> {
    if code.len() != 3 || code.as_bytes().first().copied() != Some(first) {
        return Err(invalid(field, "must be a three-character RINEX observable"));
    }
    if !code.bytes().all(|byte| byte.is_ascii_alphanumeric()) {
        return Err(invalid(field, "must be ASCII alphanumeric"));
    }
    Ok(())
}

fn map_field(error: validate::FieldError) -> ScenarioError {
    ScenarioError::InvalidInput {
        field: error.field(),
        reason: error.reason(),
    }
}

fn invalid(field: &'static str, reason: &'static str) -> ScenarioError {
    ScenarioError::InvalidInput { field, reason }
}

fn mix_seed(seed: u64, stream: u64) -> u64 {
    let mut value = seed ^ stream;
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn satellite_hash(sat: GnssSatelliteId) -> u64 {
    ((sat.system.letter() as u64) << 8) | u64::from(sat.prn)
}

fn hash_u64(hash: &mut u64, value: u64) {
    *hash ^= value;
    *hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
}

fn hash_str(hash: &mut u64, value: &str) {
    hash_u64(hash, value.len() as u64);
    for byte in value.as_bytes() {
        hash_u64(hash, u64::from(*byte));
    }
}

fn fingerprint_label(hash: u64) -> String {
    format!("sidereon-fnv64:{hash:016x}")
}

fn hash_f64(hash: &mut u64, value: f64) {
    hash_u64(hash, value.to_bits());
}

#[cfg(test)]
mod tests {
    //! Provenance: scenario observation equations use the standard GNSS code
    //! range model `P = rho + c(dt_r - dt_s) + I + T + errors` described in
    //! Kaplan and Hegarty plus Misra and Enge. Clock-noise coefficient labels
    //! follow IEEE Std 1139-2008 through the in-crate clock-stability module.
    //! The specular multipath fixture uses the standard one-reflector sinusoid
    //! in carrier wavelength and elevation. The closed-loop tests are
    //! consistency checks against this crate's SPP model, not truth validation.

    use super::*;
    use crate::astro::time::civil::j2000_seconds;
    use crate::atmosphere::TecGridSamples;
    use crate::clock_stability::{allan_deviation_power_law_slope, overlapping_adev, AllanSeries};
    use crate::positioning::{solve, SolveInputs};
    use crate::rinex::observations::{observation_values, ObservationFilter, ObservationKind};

    fn product(
        kind: ScenarioExternalProductKind,
        id: &str,
        digest: &str,
    ) -> ScenarioExternalProduct {
        ScenarioExternalProduct {
            kind,
            product_id: id.to_string(),
            content_digest: digest.to_string(),
        }
    }

    fn instant_from_j2000(seconds: i64) -> crate::astro::time::model::Instant {
        let (jd_whole, fraction) =
            crate::astro::time::civil::split_julian_date_from_j2000_seconds(seconds);
        crate::astro::time::model::Instant::from_julian_date(
            TimeScale::Gpst,
            crate::astro::time::model::JulianDateSplit::new(jd_whole, fraction)
                .expect("valid split Julian date"),
        )
    }

    fn constant_ionex(epoch_j2000_s: i64, tecu: f64) -> Ionex {
        let map = vec![
            vec![tecu, tecu, tecu],
            vec![tecu, tecu, tecu],
            vec![tecu, tecu, tecu],
        ];
        Ionex::from_samples(TecGridSamples {
            map_epochs: vec![instant_from_j2000(epoch_j2000_s)],
            lat_nodes_deg: vec![90.0, 0.0, -90.0],
            lon_nodes_deg: vec![-180.0, 0.0, 180.0],
            dlat_deg: -90.0,
            dlon_deg: 180.0,
            shell_height_km: 450.0,
            base_radius_km: 6371.0,
            exponent: 0,
            tec_maps: vec![map],
            rms_maps: Vec::new(),
        })
        .expect("valid IONEX samples")
    }

    fn base_scenario() -> Scenario {
        let start_j2000_s = j2000_seconds(2026, 1, 1, 0, 0, 0.0);
        Scenario {
            schema_version: SCENARIO_SCHEMA_VERSION,
            seed: DEFAULT_SCENARIO_SEED,
            epochs: ScenarioEpochRange {
                start_j2000_s,
                count: 2,
                cadence_s: 30.0,
            },
            receiver: ScenarioReceiver::StaticGeodetic {
                position: ScenarioGeodeticPosition {
                    lat_rad: 0.0,
                    lon_rad: 0.0,
                    height_m: 0.0,
                },
            },
            constellation: ScenarioConstellation::SyntheticKeplerian {
                satellites: gps_anchor_orbits(start_j2000_s),
            },
            signals: vec![ScenarioSignal::l1_ca(GnssSystem::Gps)],
            error_budget: ScenarioErrorBudget {
                elevation_mask_deg: -5.0,
                ..ScenarioErrorBudget::default()
            },
        }
    }

    fn gps_anchor_orbits(epoch_j2000_s: f64) -> Vec<SyntheticKeplerOrbit> {
        let a = 26_560_000.0;
        let u60 = core::f64::consts::PI / 3.0;
        [
            (1, 0.0, 0.0, 0.0),
            (2, 0.0, 0.0, u60),
            (3, 0.0, 0.0, -u60),
            (4, 0.0, core::f64::consts::FRAC_PI_2, u60),
            (5, 0.0, core::f64::consts::FRAC_PI_2, -u60),
        ]
        .into_iter()
        .map(
            |(prn, raan_rad, inclination_rad, mean_anomaly_rad)| SyntheticKeplerOrbit {
                satellite_id: GnssSatelliteId::new(GnssSystem::Gps, prn).expect("valid GPS PRN"),
                semi_major_axis_m: a,
                eccentricity: 0.0,
                inclination_rad,
                raan_rad,
                arg_perigee_rad: 0.0,
                mean_anomaly_rad,
                epoch_j2000_s,
                clock_bias_s: 0.0,
                clock_drift_s_s: 0.0,
            },
        )
        .collect()
    }

    #[test]
    fn schema_round_trips_through_json() {
        let mut scenario = base_scenario();
        scenario.error_budget.ionosphere = ScenarioIonosphereModel::SuppliedIonex {
            source: product(
                ScenarioExternalProductKind::Ionex,
                "fixture.ionex",
                "sha256:ionex-fixture",
            ),
        };
        let json = serde_json::to_string(&scenario).expect("serialize scenario");
        let reparsed: Scenario = serde_json::from_str(&json).expect("parse scenario");
        assert_eq!(reparsed, scenario);
        reparsed.validate().expect("valid scenario");
    }

    #[test]
    fn deterministic_runs_match_and_terms_sum_to_composite_bits() {
        let mut scenario = base_scenario();
        scenario.error_budget.receiver_clock = ScenarioClockModel {
            enabled: true,
            bias_s: 1.0e-7,
            drift_s_s: 1.0e-10,
            power_law_coefficients: [1.0e-24, 1.0e-26, 1.0e-22, 1.0e-26, 1.0e-28],
        };
        scenario.error_budget.thermal_noise = ScenarioThermalNoise {
            enabled: true,
            pseudorange_sigma_m: 0.25,
            carrier_phase_sigma_m: 0.002,
            doppler_sigma_hz: 0.02,
        };
        scenario.error_budget.multipath = ScenarioSpecularMultipath {
            enabled: true,
            amplitude_m: 0.15,
            reflector_height_m: 1.25,
            phase_rad: 0.3,
        };

        let first = simulate_scenario(&scenario).expect("simulate first");
        let second = simulate_scenario(&scenario).expect("simulate second");
        assert_eq!(
            first.determinism_fingerprint(),
            second.determinism_fingerprint()
        );
        assert_eq!(first, second);

        for index in 0..first.observation_count() {
            let sum = first
                .truth_terms
                .pseudorange_sum_m(index)
                .expect("term index");
            assert_eq!(
                sum.to_bits(),
                first.observations.pseudorange_m[index].to_bits(),
                "term sum at observation {index}"
            );
            let sum = first
                .truth_terms
                .carrier_phase_sum_cycles(index)
                .expect("phase term index");
            assert_eq!(
                sum.to_bits(),
                first.observations.carrier_phase_cycles[index].to_bits(),
                "phase term sum at observation {index}"
            );
            let sum = first
                .truth_terms
                .doppler_sum_hz(index)
                .expect("doppler term index");
            assert_eq!(
                sum.to_bits(),
                first.observations.doppler_hz[index].to_bits(),
                "doppler term sum at observation {index}"
            );
        }
    }

    #[test]
    fn scenario_clock_white_fm_matches_in_repo_power_law_oracle() {
        let mut clock = ClockSynth::new(
            ScenarioClockModel {
                enabled: true,
                bias_s: 0.0,
                drift_s_s: 0.0,
                power_law_coefficients: [0.0, 0.0, 1.0e-20, 0.0, 0.0],
            },
            mix_seed(DEFAULT_SCENARIO_SEED, 0x1139),
        );
        let mut frequency = Vec::with_capacity(4096);
        for index in 0..4096 {
            frequency.push(clock.sample(index as f64).rate_s_s);
        }
        let factors = [1, 2, 4, 8, 16, 32, 64, 128];
        let adev = overlapping_adev(AllanSeries::FractionalFrequency(&frequency), 1.0, &factors)
            .expect("overlapping ADEV");
        let slope = (libm::log(adev.deviation[5]) - libm::log(adev.deviation[1]))
            / (libm::log(adev.tau_s[5]) - libm::log(adev.tau_s[1]));
        assert!(
            (slope - allan_deviation_power_law_slope(PowerLawNoiseType::WhiteFM)).abs() < 0.25,
            "white-FM Allan deviation slope {slope:e}"
        );
        let expected_tau1 = (1.0e-20_f64 / (2.0 * adev.tau_s[0])).sqrt();
        let ratio = adev.deviation[0] / expected_tau1;
        assert!(
            (0.5..1.5).contains(&ratio),
            "white-FM tau-1 Allan deviation ratio {ratio:e}"
        );
    }

    #[test]
    fn rinex_export_reparses_to_same_observables() {
        let set = simulate_scenario(&base_scenario()).expect("simulate");
        let rinex = set.to_rinex_observation_file();
        let text = rinex.to_rinex_string();
        let reparsed = RinexObs::parse(&text).expect("parse generated RINEX");
        assert_eq!(reparsed, rinex);
        let sat = set.observations.satellite_id[0];
        let codes = rinex.header.obs_codes.get(&sat.system).expect("codes");
        let code_index = codes
            .iter()
            .position(|code| code == &set.observations.code_observable[0])
            .expect("code index");
        let exported = rinex.epochs[0].sats[&sat][code_index]
            .value
            .expect("exported value");
        assert_eq!(
            exported.to_bits(),
            round_rinex(set.observations.pseudorange_m[0]).to_bits()
        );

        let rows = observation_values(&reparsed, &reparsed.epochs()[0], &ObservationFilter::all())
            .expect("observation rows");
        let pseudorange_count = rows
            .iter()
            .flat_map(|(_, rows)| rows)
            .filter(|row| row.kind == ObservationKind::Pseudorange && row.value.is_some())
            .count();
        assert_eq!(
            pseudorange_count,
            set.observations.epoch_offsets[1] - set.observations.epoch_offsets[0]
        );
    }

    #[test]
    fn multiple_signals_per_constellation_survive_arrays_and_rinex() {
        let mut scenario = base_scenario();
        scenario.epochs.count = 1;
        scenario.signals.push(ScenarioSignal {
            system: GnssSystem::Gps,
            code_observable: "C2W".to_string(),
            phase_observable: "L2W".to_string(),
            doppler_observable: "D2W".to_string(),
            carrier_hz: 1_227_600_000.0,
            carrier_phase_bias_cycles: 12.0,
        });
        let set = simulate_scenario(&scenario).expect("simulate");
        assert_eq!(set.observation_count(), scenario.satellites().len() * 2);

        let rinex = set.to_rinex_observation_file();
        let gps_codes = rinex
            .header
            .obs_codes
            .get(&GnssSystem::Gps)
            .expect("GPS codes");
        assert!(gps_codes.iter().any(|code| code == "C1C"));
        assert!(gps_codes.iter().any(|code| code == "C2W"));
        assert_eq!(rinex.epochs[0].sats.len(), scenario.satellites().len());
        for values in rinex.epochs[0].sats.values() {
            let filled_pseudorange = gps_codes
                .iter()
                .zip(values.iter())
                .filter(|(code, value)| code.starts_with('C') && value.value.is_some())
                .count();
            assert_eq!(filled_pseudorange, 2);
        }
    }

    #[test]
    fn kinematic_receiver_velocity_contributes_to_doppler_terms() {
        let mut moving = base_scenario();
        moving.epochs.count = 1;
        let position = ScenarioGeodeticPosition {
            lat_rad: 0.0,
            lon_rad: 0.0,
            height_m: 0.0,
        };
        moving.receiver = ScenarioReceiver::KinematicWaypoints {
            waypoints: vec![
                ScenarioReceiverWaypoint {
                    offset_s: 0.0,
                    position,
                    velocity_ecef_m_s: Some([100.0, 0.0, 0.0]),
                },
                ScenarioReceiverWaypoint {
                    offset_s: 30.0,
                    position,
                    velocity_ecef_m_s: Some([100.0, 0.0, 0.0]),
                },
            ],
        };
        let mut static_scenario = base_scenario();
        static_scenario.epochs.count = 1;

        let moving_set = simulate_scenario(&moving).expect("simulate moving");
        let static_set = simulate_scenario(&static_scenario).expect("simulate static");
        let delta_hz =
            moving_set.observations.doppler_hz[0] - static_set.observations.doppler_hz[0];
        assert_eq!(
            delta_hz.to_bits(),
            moving_set.truth_terms.doppler_receiver_motion_hz[0].to_bits()
        );
        assert!(delta_hz.abs() > 100.0);
    }

    #[test]
    fn receiver_clock_drift_contributes_to_doppler_terms() {
        let mut scenario = base_scenario();
        scenario.epochs.count = 1;
        scenario.error_budget.receiver_clock = ScenarioClockModel {
            enabled: true,
            bias_s: 0.0,
            drift_s_s: 1.0e-10,
            power_law_coefficients: [0.0; 5],
        };
        let set = simulate_scenario(&scenario).expect("simulate");
        let expected = -1.0e-10 * F_L1_HZ;
        assert_eq!(
            set.truth_terms.doppler_receiver_clock_hz[0].to_bits(),
            expected.to_bits()
        );
        assert_eq!(
            set.truth_terms
                .doppler_sum_hz(0)
                .expect("doppler sum")
                .to_bits(),
            set.observations.doppler_hz[0].to_bits()
        );
    }

    #[test]
    fn external_source_identity_is_checked() {
        let start_j2000_s = j2000_seconds(2026, 1, 1, 0, 0, 0.0);
        let satellites = gps_anchor_orbits(start_j2000_s);
        let source = SyntheticKeplerSource::new(satellites.clone()).expect("source");
        let mut identity = product(ScenarioExternalProductKind::Sp3, "synthetic-sp3", "pending");
        let mut scenario = base_scenario();
        scenario.constellation = ScenarioConstellation::ExternalProducts {
            source: identity.clone(),
            satellites: satellites.iter().map(|sat| sat.satellite_id).collect(),
        };
        let declared = DeclaredScenarioSource::new(&source, identity.clone());
        identity.content_digest = scenario_source_transcript_fingerprint(
            &scenario,
            &declared,
            &ScenarioMediaSources::default(),
        )
        .expect("source fingerprint");
        scenario.constellation = ScenarioConstellation::ExternalProducts {
            source: identity.clone(),
            satellites: satellites.iter().map(|sat| sat.satellite_id).collect(),
        };
        let declared = DeclaredScenarioSource::new(&source, identity.clone());
        simulate_scenario_with_source(&scenario, &declared).expect("matching identity");

        let mut changed_satellites = satellites.clone();
        changed_satellites[0].semi_major_axis_m += 10.0;
        let changed_source = SyntheticKeplerSource::new(changed_satellites).expect("source");
        let mismatched_data = DeclaredScenarioSource::new(&changed_source, identity.clone());
        let err =
            simulate_scenario_with_source(&scenario, &mismatched_data).expect_err("data mismatch");
        assert!(matches!(err, ScenarioError::ExternalSourceMismatch { .. }));

        let wrong = DeclaredScenarioSource::new(
            &source,
            product(
                ScenarioExternalProductKind::Broadcast,
                "other",
                "sha256:other",
            ),
        );
        let err = simulate_scenario_with_source(&scenario, &wrong).expect_err("mismatch");
        assert!(matches!(err, ScenarioError::ExternalSourceMismatch { .. }));
    }

    #[test]
    fn supplied_ionex_requires_matching_media_and_contributes_terms() {
        let mut scenario = base_scenario();
        scenario.epochs.count = 1;
        let epoch_s = scenario.epochs.start_j2000_s.round() as i64;
        let ionex = constant_ionex(epoch_s, 12.0);
        let identity = product(
            ScenarioExternalProductKind::Ionex,
            "synthetic-ionex",
            &ionex_content_fingerprint(&ionex),
        );
        scenario.error_budget.ionosphere = ScenarioIonosphereModel::SuppliedIonex {
            source: identity.clone(),
        };

        let missing = simulate_scenario(&scenario).expect_err("IONEX media required");
        assert!(matches!(missing, ScenarioError::ExternalIonosphereRequired));

        let media = ScenarioMediaSources {
            ionex: Some(DeclaredIonexSource::new(&ionex, &identity)),
        };
        let set = simulate_scenario_with_media(&scenario, &media).expect("simulate with IONEX");
        assert!(set.truth_terms.ionosphere_m[0] > 0.0);
        assert!(
            set.truth_terms.carrier_phase_ionosphere_cycles[0] < 0.0,
            "carrier phase ionosphere has opposite sign"
        );

        let wrong_identity = product(ScenarioExternalProductKind::Ionex, "other", "sha256:other");
        let wrong_media = ScenarioMediaSources {
            ionex: Some(DeclaredIonexSource::new(&ionex, &wrong_identity)),
        };
        let err = simulate_scenario_with_media(&scenario, &wrong_media).expect_err("mismatch");
        assert!(matches!(err, ScenarioError::ExternalSourceMismatch { .. }));
    }

    #[test]
    fn synthetic_kepler_source_has_fixed_geometry_anchor() {
        let scenario = base_scenario();
        let ScenarioConstellation::SyntheticKeplerian { satellites } = &scenario.constellation
        else {
            panic!("synthetic scenario expected");
        };
        let source = SyntheticKeplerSource::new(satellites.clone()).expect("source");
        let sat = satellites[0].satellite_id;
        let state = source
            .state_at_j2000_s(sat, scenario.epochs.start_j2000_s)
            .expect("state");
        assert!((state.position_ecef_m[0] - 26_560_000.0).abs() < 1.0e-8);
        assert!(state.position_ecef_m[1].abs() < 1.0e-8);
        assert!(state.position_ecef_m[2].abs() < 1.0e-8);

        let receiver = geodetic_to_itrf(
            ScenarioGeodeticPosition {
                lat_rad: 0.0,
                lon_rad: 0.0,
                height_m: 0.0,
            }
            .to_wgs84()
            .expect("geodetic"),
        )
        .expect("ecef")
        .as_array();
        let geometric = norm3(sub3(state.position_ecef_m, receiver));
        assert!((geometric - 20_181_863.0).abs() < 1.0e-6);
    }

    #[test]
    fn clean_scenario_spp_recovers_truth_to_numerical_precision() {
        let scenario = base_scenario();
        let set = simulate_scenario(&scenario).expect("simulate");
        let ScenarioConstellation::SyntheticKeplerian { satellites } = &scenario.constellation
        else {
            panic!("synthetic scenario expected");
        };
        let source = SyntheticKeplerSource::new(satellites.clone()).expect("source");
        let truth = set.receiver_truth[0];
        let inputs = SolveInputs {
            observations: set.spp_observations_for_epoch(0),
            t_rx_j2000_s: truth.t_rx_j2000_s,
            t_rx_second_of_day_s: 0.0,
            day_of_year: 1.0,
            initial_guess: [
                truth.position_ecef_m[0],
                truth.position_ecef_m[1],
                truth.position_ecef_m[2],
                truth.clock_m,
            ],
            corrections: Corrections::NONE,
            ..SolveInputs::default()
        };
        let solved = solve(&source, &inputs, false).expect("SPP solution");
        let delta = norm3(sub3(solved.position.as_array(), truth.position_ecef_m));
        assert!(
            delta < 1.0e-5,
            "closed-loop consistency position delta {delta}"
        );
        assert!(
            (solved.rx_clock_s - truth.clock_m / C_M_S).abs() < 1.0e-12,
            "closed-loop consistency clock"
        );
    }
}
