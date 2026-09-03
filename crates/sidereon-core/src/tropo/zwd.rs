//! Exponential-scale-height ZWD troposphere delay variant.

use crate::astro::constants::{time::DAYS_PER_JULIAN_YEAR, units::DEGREES_PER_SEMICIRCLE};
use crate::astro::math::vec3::{
    dot3_z_yx_ref as dot_three_reference, unit3_ref_unchecked as unit_vector,
};
use crate::error::{Error, Result};
use std::f64::consts::PI;

pub const TROPOSPHERE_ALTITUDE_CLAMP_M: (f64, f64) = (-500.0, 9000.0);

/// Closed altitude interval used before evaluating the ZWD profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AltitudeClamp {
    /// Lower endpoint, in meters, used when clamping a height for the wet profile.
    /// The checked ZWD helpers require this value to be finite and no greater than `max_m`.
    pub min_m: f64,
    /// Upper endpoint, in meters, used when clamping a height for the wet profile.
    /// The checked ZWD helpers require this value to be finite and no less than `min_m`.
    pub max_m: f64,
}

impl Default for AltitudeClamp {
    fn default() -> Self {
        Self {
            min_m: TROPOSPHERE_ALTITUDE_CLAMP_M.0,
            max_m: TROPOSPHERE_ALTITUDE_CLAMP_M.1,
        }
    }
}

/// Exponential wet-delay profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZwdProfile {
    /// Closed altitude interval in meters applied before evaluating the wet profile.
    pub altitude_clamp: AltitudeClamp,
    /// Zenith wet delay at zero altitude, in meters, before exponential scaling.
    /// The checked helpers require this value to be finite and nonnegative.
    pub sea_level_zenith_wet_delay_m: f64,
    /// Scale height in meters used as the denominator of the negative wet-delay exponent.
    /// The checked helpers require this value to be finite and positive.
    pub wet_scale_height_m: f64,
}

impl Default for ZwdProfile {
    fn default() -> Self {
        Self {
            altitude_clamp: AltitudeClamp::default(),
            sea_level_zenith_wet_delay_m: 0.25,
            wet_scale_height_m: 2000.0,
        }
    }
}

/// Time input needed by the ZWD mapping variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZwdEpoch {
    /// Unix timestamp in nanoseconds retained with the epoch; the XYZ ZWD helper does not read it.
    pub unix_nanos: i64,
    /// Day number used by the Niell seasonal term, accepted in the inclusive range `1..=366`.
    pub day_of_year: u16,
}

impl ZwdEpoch {
    /// Construct an epoch, rejecting a `day_of_year` outside `1..=366`.
    /// The Unix timestamp and accepted day number are stored unchanged.
    pub fn new(unix_nanos: i64, day_of_year: u16) -> Result<Self> {
        validate_day_of_year(day_of_year)?;
        Ok(Self {
            unix_nanos,
            day_of_year,
        })
    }
}

/// Inputs controlling the XYZ ZWD slant-delay helper.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ZwdSlantOptions {
    /// Epoch metadata for the XYZ calculation; its `day_of_year` drives the Niell seasonal term.
    pub epoch: ZwdEpoch,
    /// Validated altitude and exponential wet-delay settings for the XYZ calculation.
    pub profile: ZwdProfile,
}

impl ZwdSlantOptions {
    /// Construct XYZ ZWD options after validating the epoch day and wet-delay profile.
    /// Returns [`crate::Error::InvalidInput`] when either validation fails.
    pub fn new(epoch: ZwdEpoch, profile: ZwdProfile) -> Result<Self> {
        validate_day_of_year(epoch.day_of_year)?;
        validate_profile(profile)?;
        Ok(Self { epoch, profile })
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Mapping {
    pub m_hydrostatic: f64,
    pub m_wet: f64,
}

pub fn niell_mapping_function(
    elevation_rad: f64,
    latitude_deg: f64,
    day_of_year: u16,
    height_m: f64,
) -> Result<Mapping> {
    validate_finite(elevation_rad, "elevation_rad")?;
    validate_latitude_deg(latitude_deg)?;
    validate_day_of_year(day_of_year)?;
    validate_finite(height_m, "height_m")?;

    Ok(niell_mapping_function_unchecked(
        elevation_rad,
        latitude_deg,
        day_of_year,
        height_m,
    ))
}

pub(crate) fn niell_mapping_function_unchecked(
    elevation_rad: f64,
    latitude_deg: f64,
    day_of_year: u16,
    height_m: f64,
) -> Mapping {
    let mut sin_e = libm::sin(elevation_rad);
    if sin_e < 0.01 {
        sin_e = 0.01;
    }

    // The ZWD 0-ULP fixture pins this multiply-then-divide order.
    let lat_rad = latitude_deg * PI / DEGREES_PER_SEMICIRCLE;
    let sin_lat = libm::sin(lat_rad);
    let phi = 2.0 * PI * (day_of_year as f64 - 1.0) / DAYS_PER_JULIAN_YEAR;

    let a_h_base = 2.65e-3;
    let b_h_base = 2.3e-4;
    let c_h_base = 1.2e-4;

    let mut a_h = a_h_base * (1.0 - 0.0025 * sin_lat * sin_lat);
    let mut b_h = b_h_base * (1.0 - 0.0025 * sin_lat * sin_lat);
    let mut c_h = c_h_base * (1.0 - 0.0025 * sin_lat * sin_lat);

    a_h += 0.0005 * libm::cos(phi) * (1.0 - 0.5 * sin_lat * sin_lat);
    b_h += 0.0001 * libm::cos(phi) * (1.0 - 0.5 * sin_lat * sin_lat);
    c_h += 0.00005 * libm::cos(phi) * (1.0 - 0.5 * sin_lat * sin_lat);

    let a_w_base = 1.5e-2;
    let b_w_base = 8.3e-3;
    let c_w_base = 1.0e-3;

    let mut a_w = a_w_base * (1.0 - 0.01 * sin_lat.abs());
    let mut b_w = b_w_base * (1.0 - 0.01 * sin_lat.abs());
    let mut c_w = c_w_base * (1.0 - 0.01 * sin_lat.abs());

    a_w += 0.005 * libm::cos(phi) * (1.0 - sin_lat.abs());
    b_w += 0.003 * libm::cos(phi) * (1.0 - sin_lat.abs());
    c_w += 0.0005 * libm::cos(phi) * (1.0 - sin_lat.abs());

    if height_m > 0.0 {
        let height_factor = libm::exp(-height_m / 8000.0);
        a_h *= height_factor;
        b_h *= height_factor;
        c_h *= height_factor;
    }

    Mapping {
        m_hydrostatic: mapping_fraction(a_h, b_h, c_h, sin_e),
        m_wet: mapping_fraction(a_w, b_w, c_w, sin_e),
    }
}

/// Compute the total ZWD-variant slant delay from satellite and receiver ECEF positions.
///
/// The two XYZ arrays are in meters. `ecef_to_lla` is called for the receiver and must return
/// `[longitude_deg, latitude_deg, ellipsoidal_height_m]`; the height is clamped by the profile
/// before standard-atmosphere Saastamoinen hydrostatic and exponential wet delays are mapped by
/// the Niell factors. Invalid options, non-finite or degenerate vectors, or an invalid callback
/// result return [`crate::Error::InvalidInput`].
pub fn tropo_delay_xyz<F>(
    options: ZwdSlantOptions,
    sat_xyz: &[f64; 3],
    receiver_xyz: &[f64; 3],
    ecef_to_lla: F,
) -> Result<f64>
where
    F: Fn(&[f64; 3]) -> [f64; 3],
{
    validate_options(options)?;
    validate_vector(sat_xyz, "sat_xyz")?;
    validate_vector(receiver_xyz, "receiver_xyz")?;
    validate_nonzero_vector(receiver_xyz, "receiver_xyz")?;

    let receiver_sat_vector = [
        sat_xyz[0] - receiver_xyz[0],
        sat_xyz[1] - receiver_xyz[1],
        sat_xyz[2] - receiver_xyz[2],
    ];
    validate_vector(&receiver_sat_vector, "receiver_sat_vector")?;
    validate_nonzero_vector(&receiver_sat_vector, "receiver_sat_vector")?;

    let receiver_up = unit_vector(receiver_xyz);
    let sat_unit = unit_vector(&receiver_sat_vector);
    let elevation_rad = libm::asin(dot_three_reference(&sat_unit, &receiver_up));

    let lonlatalt = ecef_to_lla(receiver_xyz);
    validate_lonlatalt(lonlatalt)?;
    let latitude = lonlatalt[1];
    let altitude = clamp(
        lonlatalt[2],
        options.profile.altitude_clamp.min_m,
        options.profile.altitude_clamp.max_m,
    );

    let zhd_m = saastamoinen_zhd(standard_pressure_hpa(altitude), latitude, altitude);
    let zwd_m = zwd_altitude_scaled(options.profile, altitude);
    let mapping =
        niell_mapping_function(elevation_rad, latitude, options.epoch.day_of_year, altitude)?;

    let slant_hyd = zhd_m * mapping.m_hydrostatic;
    let slant_wet = zwd_m * mapping.m_wet;
    Ok(slant_hyd + slant_wet)
}

fn standard_pressure_hpa(altitude_m: f64) -> f64 {
    1013.25 * libm::pow(1.0 - 2.25577e-5 * altitude_m, 5.2559)
}

fn saastamoinen_zhd(pressure_hpa: f64, latitude_deg: f64, altitude_m: f64) -> f64 {
    // The ZWD 0-ULP fixture pins this multiply-then-divide order.
    let lat_rad = latitude_deg * PI / DEGREES_PER_SEMICIRCLE;
    0.0022768 * pressure_hpa / (1.0 - 0.00266 * libm::cos(2.0 * lat_rad) - 2.8e-7 * altitude_m)
}

/// Evaluate the exponential zenith wet delay at a height in meters.
///
/// The height is clamped to the interval in `profile.altitude_clamp` before applying
/// `sea_level_zenith_wet_delay_m * exp(-height_m / wet_scale_height_m)`. Invalid profile values
/// or a non-finite height return [`crate::Error::InvalidInput`].
pub fn zenith_wet_delay(profile: ZwdProfile, height_m: f64) -> Result<f64> {
    validate_profile(profile)?;
    validate_finite(height_m, "height_m")?;

    Ok(zenith_wet_delay_unchecked(profile, height_m))
}

pub(crate) fn zenith_wet_delay_unchecked(profile: ZwdProfile, height_m: f64) -> f64 {
    let altitude_m = clamp(
        height_m,
        profile.altitude_clamp.min_m,
        profile.altitude_clamp.max_m,
    );
    zwd_altitude_scaled(profile, altitude_m)
}

fn zwd_altitude_scaled(profile: ZwdProfile, altitude_m: f64) -> f64 {
    profile.sea_level_zenith_wet_delay_m * libm::exp(-altitude_m / profile.wet_scale_height_m)
}

pub(crate) fn validate_profile(profile: ZwdProfile) -> Result<()> {
    validate_finite(profile.altitude_clamp.min_m, "profile.altitude_clamp.min_m")?;
    validate_finite(profile.altitude_clamp.max_m, "profile.altitude_clamp.max_m")?;
    if profile.altitude_clamp.min_m > profile.altitude_clamp.max_m {
        return Err(invalid_input("profile.altitude_clamp", "out of range"));
    }
    validate_finite(
        profile.sea_level_zenith_wet_delay_m,
        "profile.sea_level_zenith_wet_delay_m",
    )?;
    if profile.sea_level_zenith_wet_delay_m < 0.0 {
        return Err(invalid_input(
            "profile.sea_level_zenith_wet_delay_m",
            "negative",
        ));
    }
    validate_finite(profile.wet_scale_height_m, "profile.wet_scale_height_m")?;
    if profile.wet_scale_height_m <= 0.0 {
        return Err(invalid_input("profile.wet_scale_height_m", "not positive"));
    }
    Ok(())
}

fn validate_options(options: ZwdSlantOptions) -> Result<()> {
    validate_day_of_year(options.epoch.day_of_year)?;
    validate_profile(options.profile)
}

fn validate_day_of_year(day_of_year: u16) -> Result<()> {
    if (1..=366).contains(&day_of_year) {
        Ok(())
    } else {
        Err(invalid_input("day_of_year", "out of range"))
    }
}

fn validate_vector(vector: &[f64; 3], field: &'static str) -> Result<()> {
    for &value in vector {
        validate_finite(value, field)?;
    }
    Ok(())
}

fn validate_nonzero_vector(vector: &[f64; 3], field: &'static str) -> Result<()> {
    let norm2 = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    validate_finite(norm2, field)?;
    if norm2 > 0.0 {
        Ok(())
    } else {
        Err(invalid_input(field, "degenerate"))
    }
}

fn validate_lonlatalt(lonlatalt: [f64; 3]) -> Result<()> {
    validate_finite(lonlatalt[0], "receiver_lon_deg")?;
    validate_latitude_deg(lonlatalt[1])?;
    validate_finite(lonlatalt[2], "receiver_altitude_m")
}

fn validate_latitude_deg(latitude_deg: f64) -> Result<()> {
    validate_finite(latitude_deg, "latitude_deg")?;
    if (-90.0..=90.0).contains(&latitude_deg) {
        Ok(())
    } else {
        Err(invalid_input("latitude_deg", "out of range"))
    }
}

fn validate_finite(value: f64, field: &'static str) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(invalid_input(field, "not finite"))
    }
}

fn invalid_input(field: &'static str, reason: &'static str) -> Error {
    Error::InvalidInput(format!("{field} {reason}"))
}

fn mapping_fraction(a: f64, b: f64, c: f64, sin_e: f64) -> f64 {
    let numerator = 1.0 + a / (1.0 + b / (1.0 + c));
    let denominator = sin_e + a / (sin_e + b / (sin_e + c));
    numerator / denominator
}

fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}
