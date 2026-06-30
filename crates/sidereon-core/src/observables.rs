//! Forward GNSS observable prediction.
//!
//! This module owns the language-independent geometry behind Sidereon'
//! `Observables.predict`: transmit-time iteration, Sagnac rotation, line of
//! sight, range rate, Doppler, and topocentric azimuth/elevation. Ephemeris
//! parsing and interpolation stay with their existing SP3/broadcast products.

use crate::astro::frames::transforms::itrs_to_geodetic_compute;
use std::f64::consts::PI;

use crate::astro::time::civil;
use crate::constants::{
    C_M_S, DEGREES_PER_CIRCLE, DEGREES_PER_SEMICIRCLE, F_L1_HZ, KM_TO_M, MICROSECONDS_PER_SECOND,
    OBSERVABLE_TRANSMIT_TIME_ITERATIONS, OMEGA_E_DOT_RAD_S,
};
use crate::ephemeris::BroadcastEphemeris;
use crate::estimation::recipe::SagnacRecipe;
use crate::id::GnssSatelliteId;
use crate::sp3::Sp3;
use crate::spp::EphemerisSource;
use crate::validate;
use crate::Error;

const FD_HALF_S: f64 = 0.5;

/// Satellite state required by the observable predictor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObservableState {
    /// Satellite ECEF position in meters at the query epoch.
    pub position_ecef_m: [f64; 3],
    /// Satellite clock offset in seconds. SP3 clocks can be absent.
    pub clock_s: Option<f64>,
}

/// An ephemeris product usable by [`predict`].
pub trait ObservableEphemerisSource {
    /// ECEF position and optional satellite clock at seconds since J2000.
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError>;
}

impl ObservableEphemerisSource for Sp3 {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let state = self
            .position_at_j2000_seconds(sat, t_j2000_s)
            .map_err(ObservablesError::Ephemeris)?;
        Ok(ObservableState {
            position_ecef_m: state.position.as_array(),
            clock_s: state.clock_s,
        })
    }
}

impl ObservableEphemerisSource for BroadcastEphemeris {
    fn observable_state_at_j2000_s(
        &self,
        sat: GnssSatelliteId,
        t_j2000_s: f64,
    ) -> Result<ObservableState, ObservablesError> {
        let Some((position_ecef_m, clock_s)) =
            EphemerisSource::position_clock_at_j2000_s(self, sat, t_j2000_s)
        else {
            return Err(ObservablesError::NoEphemeris);
        };
        Ok(ObservableState {
            position_ecef_m,
            clock_s: Some(clock_s),
        })
    }
}

/// Input-validation failure category for observable prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservablesInputErrorKind {
    /// A floating-point input was NaN or infinite.
    NonFinite,
    /// A positive physical input was zero or negative.
    NotPositive,
    /// A non-negative physical input was negative.
    Negative,
    /// A finite numeric input was outside its accepted range.
    OutOfRange,
    /// A required input field was absent.
    Missing,
    /// A text field could not be parsed as a float.
    FloatParse,
    /// A text field could not be parsed as an integer.
    IntParse,
    /// A civil date field was out of range.
    InvalidCivilDate,
    /// A civil time field was out of range.
    InvalidCivilTime,
}

impl core::fmt::Display for ObservablesInputErrorKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let label = match self {
            Self::NonFinite => "not finite",
            Self::NotPositive => "not positive",
            Self::Negative => "negative",
            Self::OutOfRange => "out of range",
            Self::Missing => "missing",
            Self::FloatParse => "invalid float",
            Self::IntParse => "invalid integer",
            Self::InvalidCivilDate => "invalid civil date",
            Self::InvalidCivilTime => "invalid civil time",
        };
        f.write_str(label)
    }
}

impl From<&validate::FieldError> for ObservablesInputErrorKind {
    fn from(error: &validate::FieldError) -> Self {
        match error {
            validate::FieldError::Missing { .. } => Self::Missing,
            validate::FieldError::NonFinite { .. } => Self::NonFinite,
            validate::FieldError::NotPositive { .. } => Self::NotPositive,
            validate::FieldError::Negative { .. } => Self::Negative,
            validate::FieldError::OutOfRange { .. } => Self::OutOfRange,
            validate::FieldError::FloatParse { .. } => Self::FloatParse,
            validate::FieldError::IntParse { .. } => Self::IntParse,
            validate::FieldError::InvalidCivilDate { .. } => Self::InvalidCivilDate,
            validate::FieldError::InvalidCivilTime { .. } => Self::InvalidCivilTime,
        }
    }
}

/// Error returned by the observable predictor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservablesError {
    /// A public predictor input or ephemeris-source state was malformed,
    /// non-finite, or outside its physical domain.
    InvalidInput {
        /// The invalid input field.
        field: &'static str,
        /// The validation failure category.
        kind: ObservablesInputErrorKind,
    },
    /// The ephemeris product has no usable record for the satellite/epoch.
    NoEphemeris,
    /// The underlying ephemeris product returned a structured crate error.
    Ephemeris(Error),
}

impl core::fmt::Display for ObservablesError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInput { field, kind } => {
                write!(f, "invalid observable input {field}: {kind}")
            }
            Self::NoEphemeris => write!(f, "no ephemeris"),
            Self::Ephemeris(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for ObservablesError {}

/// Options controlling observable prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictOptions {
    /// Carrier frequency used to scale Doppler, hertz.
    pub carrier_hz: f64,
    /// Apply fixed-point light-time / transmit-time correction.
    pub light_time: bool,
    /// Apply Earth-rotation Sagnac correction.
    pub sagnac: bool,
}

/// Options controlling transmit-time satellite-state evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransmitTimeOptions {
    /// Apply fixed-point light-time / transmit-time correction.
    pub light_time: bool,
    /// Apply Earth-rotation Sagnac correction to the returned position/velocity.
    pub sagnac: bool,
}

impl Default for TransmitTimeOptions {
    fn default() -> Self {
        Self {
            light_time: true,
            sagnac: true,
        }
    }
}

impl Default for PredictOptions {
    fn default() -> Self {
        Self {
            carrier_hz: F_L1_HZ,
            light_time: true,
            sagnac: true,
        }
    }
}

/// Satellite state at its signal transmit time for one receive epoch.
///
/// `transmit_position_ecef_m` is the ephemeris position evaluated at
/// `transmit_time_j2000_s`. `position_ecef_m` is that position transported into
/// the receive-time ECEF frame when [`TransmitTimeOptions::sagnac`] is enabled.
/// `velocity_m_s` is the finite-difference ECEF velocity at transmit time with
/// the same transport applied.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransmitTimeSatelliteState {
    /// Signal flight time, seconds.
    pub signal_flight_time_s: f64,
    /// Transmit-time offset from receive time, rounded to microseconds.
    pub transmit_offset_us: i64,
    /// Transmit time as seconds since J2000.
    pub transmit_time_j2000_s: f64,
    /// Satellite clock offset at transmit time, seconds.
    pub clock_s: Option<f64>,
    /// Ephemeris ECEF satellite position at transmit time, metres.
    pub transmit_position_ecef_m: [f64; 3],
    /// Sagnac-transported ECEF satellite position, metres.
    pub position_ecef_m: [f64; 3],
    /// Sagnac-transported ECEF satellite velocity, metres per second.
    pub velocity_m_s: [f64; 3],
    /// Geometric range after optional Sagnac transport, metres.
    pub geometric_range_m: f64,
    /// Receiver-to-satellite line-of-sight unit vector in ECEF.
    pub los_unit: [f64; 3],
}

/// Predicted GNSS observables at one receive epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PredictedObservables {
    /// Geometric range after optional Sagnac rotation, meters.
    pub geometric_range_m: f64,
    /// Range-rate LOS projection, meters per second.
    pub range_rate_m_s: f64,
    /// Doppler shift at `PredictOptions::carrier_hz`, hertz.
    pub doppler_hz: f64,
    /// Satellite clock offset at transmit time, seconds.
    pub sat_clock_s: Option<f64>,
    /// Topocentric elevation, degrees.
    pub elevation_deg: f64,
    /// Topocentric azimuth in `[0, 360)`, degrees.
    pub azimuth_deg: f64,
    /// Transmit-time offset from receive time, rounded to microseconds.
    pub transmit_offset_us: i64,
    /// Transmit time as seconds since J2000.
    pub transmit_time_j2000_s: f64,
    /// Receiver-to-satellite line-of-sight unit vector in ECEF.
    pub los_unit: [f64; 3],
    /// Sagnac-rotated satellite ECEF position in meters.
    pub sat_pos_ecef_m: [f64; 3],
    /// Sagnac-rotated satellite ECEF velocity in meters per second.
    pub sat_velocity_m_s: [f64; 3],
}

/// Convert split Julian date to seconds since J2000.
pub fn j2000_seconds_from_split(jd_whole: f64, jd_fraction: f64) -> Result<f64, ObservablesError> {
    validate::finite(jd_whole, "jd_whole").map_err(map_input_error)?;
    validate::finite(jd_fraction, "jd_fraction").map_err(map_input_error)?;
    validate::finite(
        civil::j2000_seconds_from_split(jd_whole, jd_fraction),
        "j2000_seconds",
    )
    .map_err(map_input_error)
}

/// Evaluate a satellite's transmit-time ECEF state for one static receiver.
///
/// This is the per-satellite primitive underneath observable prediction: it
/// iterates light time, evaluates the ephemeris at the satellite's transmit
/// epoch, applies the Sagnac/Earth-rotation transport if requested, and returns
/// the transported position, velocity, clock, range, and line of sight without
/// constructing Doppler or topocentric observables.
pub fn transmit_time_satellite_state(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: TransmitTimeOptions,
) -> Result<TransmitTimeSatelliteState, ObservablesError> {
    validate_transmit_time_inputs(receiver_ecef_m, t_rx_j2000_s)?;
    let predict_options = PredictOptions {
        carrier_hz: F_L1_HZ,
        light_time: options.light_time,
        sagnac: options.sagnac,
    };
    let solved = solve_transmit_time(source, sat, receiver_ecef_m, t_rx_j2000_s, predict_options)?;

    let dx = solved.sat_rot_ecef_m[0] - receiver_ecef_m[0];
    let dy = solved.sat_rot_ecef_m[1] - receiver_ecef_m[1];
    let dz = solved.sat_rot_ecef_m[2] - receiver_ecef_m[2];
    let range = geometric_range_m([dx, dy, dz])?;
    let los = [dx / range, dy / range, dz / range];

    let velocity = satellite_velocity(source, sat, solved.transmit_time_j2000_s)?;
    let velocity_rot = sagnac_rotate(velocity, solved.tau_s, options.sagnac);
    validate::finite_vec3(velocity_rot, "satellite velocity_m_s").map_err(map_input_error)?;

    Ok(TransmitTimeSatelliteState {
        signal_flight_time_s: solved.tau_s,
        transmit_offset_us: solved.transmit_offset_us,
        transmit_time_j2000_s: solved.transmit_time_j2000_s,
        clock_s: solved.state.clock_s,
        transmit_position_ecef_m: solved.state.position_ecef_m,
        position_ecef_m: solved.sat_rot_ecef_m,
        velocity_m_s: velocity_rot,
        geometric_range_m: range,
        los_unit: los,
    })
}

/// Predict observables for `sat` from a static ECEF receiver.
pub fn predict(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
) -> Result<PredictedObservables, ObservablesError> {
    validate_predict_inputs(receiver_ecef_m, t_rx_j2000_s, options)?;
    let solved = solve_transmit_time(source, sat, receiver_ecef_m, t_rx_j2000_s, options)?;

    let dx = solved.sat_rot_ecef_m[0] - receiver_ecef_m[0];
    let dy = solved.sat_rot_ecef_m[1] - receiver_ecef_m[1];
    let dz = solved.sat_rot_ecef_m[2] - receiver_ecef_m[2];
    let range = geometric_range_m([dx, dy, dz])?;
    let los = [dx / range, dy / range, dz / range];

    let velocity = satellite_velocity(source, sat, solved.transmit_time_j2000_s)?;
    let velocity_rot = sagnac_rotate(velocity, solved.tau_s, options.sagnac);
    validate::finite_vec3(velocity_rot, "satellite velocity_m_s").map_err(map_input_error)?;
    let range_rate = los[0] * velocity_rot[0] + los[1] * velocity_rot[1] + los[2] * velocity_rot[2];
    validate::finite(range_rate, "range_rate_m_s").map_err(map_input_error)?;
    let doppler_hz = -range_rate * options.carrier_hz / C_M_S;
    validate::finite(doppler_hz, "doppler_hz").map_err(map_input_error)?;
    let (elevation_deg, azimuth_deg) = topocentric(receiver_ecef_m, [dx, dy, dz], range)?;

    Ok(PredictedObservables {
        geometric_range_m: range,
        range_rate_m_s: range_rate,
        doppler_hz,
        sat_clock_s: solved.state.clock_s,
        elevation_deg,
        azimuth_deg,
        transmit_offset_us: solved.transmit_offset_us,
        transmit_time_j2000_s: solved.transmit_time_j2000_s,
        los_unit: los,
        sat_pos_ecef_m: solved.sat_rot_ecef_m,
        sat_velocity_m_s: velocity_rot,
    })
}

#[derive(Debug, Clone, Copy)]
struct SolvedTransmitTime {
    tau_s: f64,
    transmit_offset_us: i64,
    transmit_time_j2000_s: f64,
    state: ObservableState,
    sat_rot_ecef_m: [f64; 3],
}

fn solve_transmit_time(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
) -> Result<SolvedTransmitTime, ObservablesError> {
    if !options.light_time {
        let state = validated_state_at_j2000_s(source, sat, t_rx_j2000_s)?;
        let sat_rot = sagnac_rotate(state.position_ecef_m, 0.0, options.sagnac);
        validate::finite_vec3(sat_rot, "satellite position_ecef_m").map_err(map_input_error)?;
        return Ok(SolvedTransmitTime {
            tau_s: 0.0,
            transmit_offset_us: 0,
            transmit_time_j2000_s: t_rx_j2000_s,
            state,
            sat_rot_ecef_m: sat_rot,
        });
    }

    let mut tau = 0.0;
    for iter in 0..OBSERVABLE_TRANSMIT_TIME_ITERATIONS {
        let transmit_offset_us = microseconds_from_tau(tau);
        let t_tx = t_rx_j2000_s - transmit_offset_us as f64 / MICROSECONDS_PER_SECOND;
        let state = validated_state_at_j2000_s(source, sat, t_tx)?;
        let sat_rot = sagnac_rotate(state.position_ecef_m, tau, options.sagnac);
        validate::finite_vec3(sat_rot, "satellite position_ecef_m").map_err(map_input_error)?;
        let dx = sat_rot[0] - receiver_ecef_m[0];
        let dy = sat_rot[1] - receiver_ecef_m[1];
        let dz = sat_rot[2] - receiver_ecef_m[2];
        let range = geometric_range_m([dx, dy, dz])?;
        let new_tau = range / C_M_S;

        if iter + 1 == OBSERVABLE_TRANSMIT_TIME_ITERATIONS {
            return finalize_transmit_time(source, sat, t_rx_j2000_s, new_tau, options.sagnac);
        }

        tau = new_tau;
    }

    unreachable!("fixed transmit-time loop always returns on its last iteration")
}

fn finalize_transmit_time(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_rx_j2000_s: f64,
    tau: f64,
    sagnac: bool,
) -> Result<SolvedTransmitTime, ObservablesError> {
    let transmit_offset_us = microseconds_from_tau(tau);
    let t_tx = t_rx_j2000_s - transmit_offset_us as f64 / MICROSECONDS_PER_SECOND;
    validate::finite(t_tx, "transmit_time_j2000_s").map_err(map_input_error)?;
    let state = validated_state_at_j2000_s(source, sat, t_tx)?;
    let sat_rot = sagnac_rotate(state.position_ecef_m, tau, sagnac);
    validate::finite_vec3(sat_rot, "satellite position_ecef_m").map_err(map_input_error)?;
    Ok(SolvedTransmitTime {
        tau_s: tau,
        transmit_offset_us,
        transmit_time_j2000_s: t_tx,
        state,
        sat_rot_ecef_m: sat_rot,
    })
}

fn microseconds_from_tau(tau_s: f64) -> i64 {
    (tau_s * MICROSECONDS_PER_SECOND).round() as i64
}

fn satellite_velocity(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_tx_j2000_s: f64,
) -> Result<[f64; 3], ObservablesError> {
    let plus = validated_state_at_j2000_s(source, sat, t_tx_j2000_s + FD_HALF_S)?;
    let minus = validated_state_at_j2000_s(source, sat, t_tx_j2000_s - FD_HALF_S)?;
    let denom = 2.0 * FD_HALF_S;
    let velocity = [
        (plus.position_ecef_m[0] - minus.position_ecef_m[0]) / denom,
        (plus.position_ecef_m[1] - minus.position_ecef_m[1]) / denom,
        (plus.position_ecef_m[2] - minus.position_ecef_m[2]) / denom,
    ];
    validate::finite_vec3(velocity, "satellite velocity_m_s").map_err(map_input_error)
}

fn validate_predict_inputs(
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
    options: PredictOptions,
) -> Result<(), ObservablesError> {
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(map_input_error)?;
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    validate::finite_positive(options.carrier_hz, "options.carrier_hz").map_err(map_input_error)?;
    Ok(())
}

fn validate_transmit_time_inputs(
    receiver_ecef_m: [f64; 3],
    t_rx_j2000_s: f64,
) -> Result<(), ObservablesError> {
    validate::finite_vec3(receiver_ecef_m, "receiver_ecef_m").map_err(map_input_error)?;
    validate::finite(t_rx_j2000_s, "t_rx_j2000_s").map_err(map_input_error)?;
    Ok(())
}

fn validated_state_at_j2000_s(
    source: &dyn ObservableEphemerisSource,
    sat: GnssSatelliteId,
    t_j2000_s: f64,
) -> Result<ObservableState, ObservablesError> {
    let state = source.observable_state_at_j2000_s(sat, t_j2000_s)?;
    validate_observable_state(&state)?;
    Ok(state)
}

fn validate_observable_state(state: &ObservableState) -> Result<(), ObservablesError> {
    validate::finite_vec3(state.position_ecef_m, "observable state position_ecef_m")
        .map_err(map_input_error)?;
    if let Some(clock_s) = state.clock_s {
        validate::finite(clock_s, "observable state clock_s").map_err(map_input_error)?;
    }
    Ok(())
}

fn geometric_range_m(delta_ecef_m: [f64; 3]) -> Result<f64, ObservablesError> {
    let range = (delta_ecef_m[0] * delta_ecef_m[0]
        + delta_ecef_m[1] * delta_ecef_m[1]
        + delta_ecef_m[2] * delta_ecef_m[2])
        .sqrt();
    validate::finite_positive(range, "geometric_range_m").map_err(map_input_error)
}

fn map_input_error(error: validate::FieldError) -> ObservablesError {
    ObservablesError::InvalidInput {
        field: error.field(),
        kind: ObservablesInputErrorKind::from(&error),
    }
}

fn sagnac_rotate(pos: [f64; 3], tau_s: f64, apply: bool) -> [f64; 3] {
    let sagnac = if apply {
        SagnacRecipe::ClosedFormZRotation
    } else {
        SagnacRecipe::Off
    };
    crate::estimation::substrate::range::rotate_transmit_satellite(
        sagnac,
        pos,
        tau_s,
        OMEGA_E_DOT_RAD_S,
    )
}

fn topocentric(
    receiver_ecef_m: [f64; 3],
    delta_ecef_m: [f64; 3],
    range_m: f64,
) -> Result<(f64, f64), ObservablesError> {
    let (lat_deg, lon_deg, _height_km) = itrs_to_geodetic_compute(
        receiver_ecef_m[0] / KM_TO_M,
        receiver_ecef_m[1] / KM_TO_M,
        receiver_ecef_m[2] / KM_TO_M,
    )
    .map_err(|_| ObservablesError::InvalidInput {
        field: "receiver_ecef_m",
        kind: ObservablesInputErrorKind::OutOfRange,
    })?;
    // Sidereon' application oracle pins this multiply-then-divide order.
    let lat = lat_deg * PI / DEGREES_PER_SEMICIRCLE;
    let lon = lon_deg * PI / DEGREES_PER_SEMICIRCLE;

    let sl = lat.sin();
    let cl = lat.cos();
    let so = lon.sin();
    let co = lon.cos();

    let dx = delta_ecef_m[0];
    let dy = delta_ecef_m[1];
    let dz = delta_ecef_m[2];

    let e = -so * dx + co * dy;
    let n = -sl * co * dx - sl * so * dy + cl * dz;
    let u = cl * co * dx + cl * so * dy + sl * dz;

    // Sidereon' application oracle pins this multiply-then-divide order.
    let mut azimuth_deg = e.atan2(n) * DEGREES_PER_SEMICIRCLE / PI;
    if azimuth_deg < 0.0 {
        azimuth_deg += DEGREES_PER_CIRCLE;
    }
    let elevation_deg = (u / range_m).asin() * DEGREES_PER_SEMICIRCLE / PI;

    validate::finite(elevation_deg, "elevation_deg").map_err(map_input_error)?;
    validate::finite(azimuth_deg, "azimuth_deg").map_err(map_input_error)?;
    Ok((elevation_deg, azimuth_deg))
}

#[cfg(test)]
mod public_api_tests {
    use super::*;
    use crate::{GnssSatelliteId, GnssSystem};

    #[derive(Debug, Clone, Copy)]
    struct StaticSource {
        state: ObservableState,
    }

    impl ObservableEphemerisSource for StaticSource {
        fn observable_state_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(self.state)
        }
    }

    #[test]
    fn transmit_time_state_matches_predict_substrate_with_no_light_time() {
        let source = StaticSource {
            state: ObservableState {
                position_ecef_m: [20_200_000.0, 14_000_000.0, 21_700_000.0],
                clock_s: Some(1.25e-6),
            },
        };
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let rx = [4_027_894.0, 307_046.0, 4_919_474.0];
        let state = transmit_time_satellite_state(
            &source,
            sat,
            rx,
            646_272_000.0,
            TransmitTimeOptions {
                light_time: false,
                sagnac: true,
            },
        )
        .expect("state");
        let prediction = predict(
            &source,
            sat,
            rx,
            646_272_000.0,
            PredictOptions {
                carrier_hz: F_L1_HZ,
                light_time: false,
                sagnac: true,
            },
        )
        .expect("prediction");

        assert_eq!(state.signal_flight_time_s.to_bits(), 0.0f64.to_bits());
        assert_eq!(state.transmit_offset_us, 0);
        assert_eq!(
            state.transmit_time_j2000_s.to_bits(),
            646_272_000.0f64.to_bits()
        );
        assert_eq!(state.clock_s.unwrap().to_bits(), 1.25e-6f64.to_bits());
        assert_eq!(
            state.transmit_position_ecef_m.map(f64::to_bits),
            source.state.position_ecef_m.map(f64::to_bits)
        );
        assert_eq!(
            state.position_ecef_m.map(f64::to_bits),
            prediction.sat_pos_ecef_m.map(f64::to_bits)
        );
        assert_eq!(
            state.velocity_m_s.map(f64::to_bits),
            prediction.sat_velocity_m_s.map(f64::to_bits)
        );
        assert_eq!(
            state.geometric_range_m.to_bits(),
            prediction.geometric_range_m.to_bits()
        );
        assert_eq!(
            state.los_unit.map(f64::to_bits),
            prediction.los_unit.map(f64::to_bits)
        );
    }
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use super::*;
    use crate::{GnssSatelliteId, GnssSystem};

    #[derive(Debug, Clone, Copy)]
    struct StaticSource {
        state: ObservableState,
    }

    impl ObservableEphemerisSource for StaticSource {
        fn observable_state_at_j2000_s(
            &self,
            _sat: GnssSatelliteId,
            _t_j2000_s: f64,
        ) -> Result<ObservableState, ObservablesError> {
            Ok(self.state)
        }
    }

    fn sp3_fixture() -> Sp3 {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/sp3/GRG0MGXFIN_20201760000_01D_15M_ORB.SP3"
        );
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read SP3 fixture {path}: {e}"));
        Sp3::parse(&bytes).expect("parse SP3 fixture")
    }

    fn static_source(position_ecef_m: [f64; 3]) -> StaticSource {
        StaticSource {
            state: ObservableState {
                position_ecef_m,
                clock_s: Some(0.0),
            },
        }
    }

    fn no_light_time_options() -> PredictOptions {
        PredictOptions {
            carrier_hz: F_L1_HZ,
            light_time: false,
            sagnac: true,
        }
    }

    fn assert_invalid_observables_input(
        err: ObservablesError,
        field: &'static str,
        kind: ObservablesInputErrorKind,
    ) {
        match err {
            ObservablesError::InvalidInput {
                field: got_field,
                kind: got_kind,
            } => {
                assert_eq!(got_field, field);
                assert_eq!(got_kind, kind);
            }
            other => panic!("expected InvalidInput({field}, {kind:?}), got {other:?}"),
        }
    }

    #[test]
    fn split_julian_to_j2000_seconds_matches_orbis_time() {
        let t = j2000_seconds_from_split(2_459_024.5, 0.5).expect("valid split Julian date");
        assert_eq!(t, 646_272_000.0);
    }

    #[test]
    fn split_julian_to_j2000_seconds_rejects_non_finite_parts() {
        for (jd_whole, jd_fraction, field) in [
            (f64::NAN, 0.5, "jd_whole"),
            (f64::INFINITY, 0.5, "jd_whole"),
            (2_459_024.5, f64::NAN, "jd_fraction"),
            (2_459_024.5, f64::NEG_INFINITY, "jd_fraction"),
        ] {
            let err = j2000_seconds_from_split(jd_whole, jd_fraction)
                .expect_err("non-finite split Julian date part must fail");
            assert_invalid_observables_input(err, field, ObservablesInputErrorKind::NonFinite);
        }
    }

    #[test]
    fn sp3_predict_reference_case() {
        let sp3 = sp3_fixture();
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");
        let rx = [3_512_900.0, 780_500.0, 5_248_700.0];
        let obs = predict(&sp3, sat, rx, 646_272_000.0, PredictOptions::default())
            .expect("predict observables");

        assert_eq!(obs.geometric_range_m.to_bits(), 0x4173cf438ba57358);
        assert_eq!(obs.range_rate_m_s.to_bits(), 0x402d7dd36f6b8980);
        assert_eq!(obs.doppler_hz.to_bits(), 0xc0535f534ba7c77d);
        assert_eq!(obs.sat_clock_s.unwrap().to_bits(), 0x3ef04d2d8279460c);
        assert_eq!(obs.elevation_deg.to_bits(), 0x4054590eed870f52);
        assert_eq!(obs.azimuth_deg.to_bits(), 0x40645ff5a090a131);
        assert_eq!(obs.transmit_offset_us, 69_288);
        assert_eq!(obs.transmit_time_j2000_s.to_bits(), 0x41c342a9fff72192);
        assert_eq!(
            obs.los_unit.map(f64::to_bits),
            [0x3fe4c70da9fa70dd, 0x3fc834429adb2bae, 0x3fe792a4f57fdcb1,]
        );
        assert_eq!(
            obs.sat_pos_ecef_m.map(f64::to_bits),
            [0x41703667d8c0eb8f, 0x4151f601b1d775f3, 0x4173992c0ec03dcd,]
        );
        assert_eq!(
            obs.sat_velocity_m_s.map(f64::to_bits),
            [0xc09c17d81e540ab6, 0x409a192982abbeb7, 0x40926013f2ae8000,]
        );
    }

    #[test]
    fn predict_rejects_invalid_entry_inputs() {
        let source = static_source([20_200_000.0, 14_000_000.0, 21_700_000.0]);
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");

        let err = predict(
            &source,
            sat,
            [f64::NAN, 0.0, 0.0],
            646_272_000.0,
            no_light_time_options(),
        )
        .expect_err("non-finite receiver position must fail");
        assert_invalid_observables_input(
            err,
            "receiver_ecef_m",
            ObservablesInputErrorKind::NonFinite,
        );

        let err = predict(
            &source,
            sat,
            [0.0, 0.0, 0.0],
            f64::INFINITY,
            no_light_time_options(),
        )
        .expect_err("non-finite receive time must fail");
        assert_invalid_observables_input(err, "t_rx_j2000_s", ObservablesInputErrorKind::NonFinite);

        let mut options = no_light_time_options();
        options.carrier_hz = 0.0;
        let err = predict(&source, sat, [0.0, 0.0, 0.0], 646_272_000.0, options)
            .expect_err("non-positive carrier must fail");
        assert_invalid_observables_input(
            err,
            "options.carrier_hz",
            ObservablesInputErrorKind::NotPositive,
        );
    }

    #[test]
    fn predict_rejects_invalid_source_state_and_zero_range() {
        let sat = GnssSatelliteId::new(GnssSystem::Gps, 21).expect("valid satellite id");

        let source = static_source([f64::NAN, 14_000_000.0, 21_700_000.0]);
        let err = predict(
            &source,
            sat,
            [0.0, 0.0, 0.0],
            646_272_000.0,
            no_light_time_options(),
        )
        .expect_err("non-finite ephemeris position must fail");
        assert_invalid_observables_input(
            err,
            "observable state position_ecef_m",
            ObservablesInputErrorKind::NonFinite,
        );

        let source = static_source([1_000.0, 2_000.0, 3_000.0]);
        let err = predict(
            &source,
            sat,
            [1_000.0, 2_000.0, 3_000.0],
            646_272_000.0,
            no_light_time_options(),
        )
        .expect_err("zero geometric range must fail");
        assert_invalid_observables_input(
            err,
            "geometric_range_m",
            ObservablesInputErrorKind::NotPositive,
        );
    }

    #[test]
    fn topocentric_rejects_invalid_receiver_geodetic_conversion() {
        let err = topocentric([f64::MAX, 0.0, 0.0], [1.0, 0.0, 0.0], 1.0)
            .expect_err("invalid receiver geodetic conversion must fail");

        assert_invalid_observables_input(
            err,
            "receiver_ecef_m",
            ObservablesInputErrorKind::OutOfRange,
        );
    }
}
