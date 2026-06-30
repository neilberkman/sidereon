//! Ionospheric delay models.
//!
//! This module exposes the single-frequency ionospheric group-delay models used
//! to correct GNSS pseudoranges. The GPS broadcast Klobuchar model and the IONEX
//! vertical-TEC grid path are both reached through the same [`ionosphere_delay`]
//! entry, and the IONEX grid parser is exposed directly as [`Ionex`].
//!
//! All delays returned are group delays and are positive: they increase the
//! measured pseudorange (the carrier-phase advance is the negation of this
//! value). The ionosphere is dispersive, so a delay reported on a carrier other
//! than the model's native L1 is the L1 delay scaled by `(f_l1 / f)^2`.

mod grid;
mod klobuchar;
mod nequick_g;
mod nequick_g_data;
mod slant;
mod tec_grid;
mod write;

#[cfg(all(test, sidereon_repo_tests))]
mod tests;

use crate::astro::constants::time::SECONDS_PER_DAY;
use crate::astro::time::civil::{
    fractional_day_of_year_from_instant, j2000_seconds_from_split, second_of_day_from_instant,
    split_julian_date_from_j2000_seconds,
};
use crate::astro::time::model::{Instant, InstantRepr, JulianDateSplit, TimeScale};

use crate::constants::{DEG_TO_RAD, MEAN_EARTH_RADIUS_M, RAD_TO_DEG};
use crate::error::{Error, Result};
use crate::frame::Wgs84Geodetic;
use crate::frequencies::{self, CarrierBand};
use crate::GnssSystem;

pub use grid::Ionex;
pub use nequick_g::{nequick_g_delay_m, nequick_g_stec_tecu, NequickGRayEval};
pub use tec_grid::{
    iono_delay_xyz as regular_tec_grid_delay_xyz, tec_xyz as regular_tec_xyz, TecGrid,
    TecGridEpoch, TecGridEvalOptions, TecGridShellGeometry,
};

pub(crate) use klobuchar::klobuchar_l1_components;

pub(crate) fn ionex_epoch_from_j2000_seconds(seconds: i64) -> Instant {
    instant_from_j2000_seconds(TimeScale::Utc, seconds)
}

pub(crate) fn instant_from_j2000_seconds(scale: TimeScale, seconds: i64) -> Instant {
    let (jd_whole, fraction) = split_julian_date_from_j2000_seconds(seconds);
    Instant::from_julian_date(
        scale,
        JulianDateSplit::new(jd_whole, fraction).expect("valid split Julian date"),
    )
}

pub(crate) fn j2000_seconds_from_instant(epoch: Instant) -> Option<i64> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            let seconds = j2000_seconds_from_split(split.jd_whole, split.fraction);
            if seconds.is_finite() && seconds >= i64::MIN as f64 && seconds <= i64::MAX as f64 {
                Some(seconds.round() as i64)
            } else {
                None
            }
        }
        InstantRepr::Nanos(nanos) => {
            let seconds = (nanos as f64 / 1.0e9).round();
            if seconds.is_finite() && seconds >= i64::MIN as f64 && seconds <= i64::MAX as f64 {
                Some(seconds as i64)
            } else {
                None
            }
        }
    }
}

/// Broadcast Klobuchar alpha/beta coefficients.
///
/// `alpha` are the four coefficients of the cosine-amplitude polynomial (in
/// seconds and seconds-per-semicircle powers); `beta` are the four coefficients
/// of the period polynomial (in seconds and seconds-per-semicircle powers).
/// These are the eight values transmitted in the GPS navigation message.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KlobucharParams {
    /// Cosine-amplitude polynomial coefficients (a0..a3).
    pub alpha: [f64; 4],
    /// Period polynomial coefficients (b0..b3).
    pub beta: [f64; 4],
}

/// Galileo broadcast NeQuick-G ionosphere coefficients (`ai0`, `ai1`, `ai2`).
///
/// Galileo navigation messages broadcast these three coefficients to drive the
/// effective ionisation level used by the Galileo single-frequency ionosphere
/// correction. They are distinct from GPS/BeiDou Klobuchar alpha/beta values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GalileoNequickCoeffs {
    /// Constant effective-ionisation coefficient.
    pub ai0: f64,
    /// Linear MODIP coefficient.
    pub ai1: f64,
    /// Quadratic MODIP coefficient.
    pub ai2: f64,
}

/// Native inputs for the Galileo coefficient-driven ionosphere correction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GalileoNequickEval {
    /// Receiver geodetic latitude, degrees.
    pub lat_deg: f64,
    /// Receiver geodetic longitude, degrees.
    pub lon_deg: f64,
    /// Satellite elevation, degrees.
    pub el_deg: f64,
    /// Galileo-system second of day.
    pub t_gal_s: f64,
    /// Fractional day of year.
    pub day_of_year: f64,
    /// Carrier frequency on which to report the delay.
    pub frequency_hz: f64,
}

/// Selects which ionospheric model produces the delay.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IonoModel {
    /// GPS broadcast Klobuchar model with its eight alpha/beta coefficients.
    Klobuchar(KlobucharParams),
    /// Galileo coefficient-driven single-frequency ionosphere correction.
    GalileoNequickG(GalileoNequickCoeffs),
}

/// Single-frequency ionospheric group delay (code, positive meters).
///
/// Dispatches on `model`. `frequency_hz` is the carrier on which the delay is
/// reported; the model is dispersive, so the delay scales as `1 / f^2`. The
/// returned value is positive meters that increase the pseudorange.
pub fn ionosphere_delay(
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    epoch: Instant,
    frequency_hz: f64,
    model: &IonoModel,
) -> Result<f64> {
    validate_receiver(receiver)?;
    validate_finite(elevation_rad, "elevation_rad")?;
    validate_elevation_rad(elevation_rad, "elevation_rad")?;
    validate_finite(azimuth_rad, "azimuth_rad")?;
    validate_instant(epoch)?;
    validate_frequency(frequency_hz)?;

    match model {
        IonoModel::Klobuchar(params) => klobuchar(
            params,
            receiver,
            elevation_rad,
            azimuth_rad,
            epoch,
            frequency_hz,
        ),
        IonoModel::GalileoNequickG(coeffs) => galileo_nequick_g_native(
            coeffs,
            GalileoNequickEval {
                lat_deg: receiver.lat_rad * RAD_TO_DEG,
                lon_deg: receiver.lon_rad * RAD_TO_DEG,
                el_deg: elevation_rad * RAD_TO_DEG,
                t_gal_s: gps_second_of_day(epoch),
                day_of_year: fractional_day_of_year(epoch),
                frequency_hz,
            },
        ),
    }
}

/// GPS broadcast Klobuchar ionospheric group delay (positive meters).
///
/// Lower-level entry the Klobuchar arm of [`ionosphere_delay`] calls; exposed so
/// the broadcast model can be used directly. The receiver geodetic
/// latitude/longitude and the satellite azimuth/elevation are converted from
/// radians to the model's published degree boundary, and the GPS second-of-day
/// is taken from `epoch`. The model evaluates the L1 group delay; the result is
/// then scaled to `frequency_hz` by the dispersive `(f_l1 / f)^2` factor.
///
/// Note on bit-exactness: this wrapper converts both angle (radians -> degrees)
/// and time (`epoch` -> GPS second-of-day) at its boundary; both are
/// representation-bound, so the wrapper is NOT bit-exact to a golden expressed
/// in the kernel's native units (degrees and an exact second-of-day) - the
/// difference is at the nanometre level. The 0-ULP parity contract is on the
/// model kernel in those native units; this convenience entry agrees with it to
/// within that conversion bound.
pub fn klobuchar(
    params: &KlobucharParams,
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    epoch: Instant,
    frequency_hz: f64,
) -> Result<f64> {
    validate_receiver(receiver)?;
    validate_finite(elevation_rad, "elevation_rad")?;
    validate_elevation_rad(elevation_rad, "elevation_rad")?;
    validate_finite(azimuth_rad, "azimuth_rad")?;
    validate_instant(epoch)?;

    klobuchar_native(
        params,
        receiver.lat_rad * RAD_TO_DEG,
        receiver.lon_rad * RAD_TO_DEG,
        azimuth_rad * RAD_TO_DEG,
        elevation_rad * RAD_TO_DEG,
        gps_second_of_day(epoch),
        frequency_hz,
    )
}

/// GPS broadcast Klobuchar group delay in the model's native input units
/// (positive meters).
///
/// Latitude/longitude and azimuth/elevation are in **degrees** (the model's
/// published boundary) and `t_gps_s` is the GPS **second-of-day** in
/// `[0, 86400)`. This is the bit-exact (0-ULP) entry: it feeds the model kernel
/// directly with no angle or time conversion, so a caller holding native inputs
/// (for example the Elixir wrapper, which already has degrees and an integer
/// time of day) gets exactly the reference result. The L1 delay is scaled to
/// `frequency_hz` by the dispersive `(f_l1 / f)^2` factor.
pub fn klobuchar_native(
    params: &KlobucharParams,
    lat_deg: f64,
    lon_deg: f64,
    az_deg: f64,
    el_deg: f64,
    t_gps_s: f64,
    frequency_hz: f64,
) -> Result<f64> {
    validate_klobuchar_params(params)?;
    validate_lat_deg(lat_deg, "lat_deg")?;
    validate_lon_deg(lon_deg, "lon_deg")?;
    validate_finite(az_deg, "az_deg")?;
    validate_el_deg(el_deg, "el_deg")?;
    validate_second_of_day(t_gps_s, "t_gps_s")?;
    validate_frequency(frequency_hz)?;

    let delay_m = klobuchar_native_unchecked(
        params,
        lat_deg,
        lon_deg,
        az_deg,
        el_deg,
        t_gps_s,
        frequency_hz,
    );
    validate_finite(delay_m, "ionosphere_delay_m")?;
    Ok(delay_m)
}

pub(crate) fn klobuchar_native_unchecked(
    params: &KlobucharParams,
    lat_deg: f64,
    lon_deg: f64,
    az_deg: f64,
    el_deg: f64,
    t_gps_s: f64,
    frequency_hz: f64,
) -> f64 {
    let c = klobuchar_l1_components(
        lat_deg,
        lon_deg,
        az_deg,
        el_deg,
        t_gps_s,
        params.alpha,
        params.beta,
    );

    let f_l1_hz = frequencies::frequency_hz(GnssSystem::Gps, CarrierBand::L1)
        .expect("canonical GPS L1 carrier exists");
    let ratio = f_l1_hz / frequency_hz;
    c.delay_l1_m * (ratio * ratio)
}

/// Galileo coefficient-driven single-frequency group delay in native units.
///
/// The full Galileo NeQuick-G reference model is a three-dimensional electron
/// density integration driven by `ai0`/`ai1`/`ai2`. This compact entry keeps the
/// Galileo/GPS model boundary correct for SPP by using those Galileo broadcast
/// coefficients to form the effective ionisation level and mapping the resulting
/// slant TEC to meters with the standard dispersive `40.3 / f^2` relation. It is
/// deliberately separate from [`klobuchar_native`] so Galileo observations never
/// consume GPS Klobuchar coefficients when Galileo coefficients are supplied.
///
/// Latitude/longitude/elevation are in degrees. `t_gal_s` is the Galileo-system
/// second of day, and `day_of_year` is the fractional day of year used for a
/// small seasonal term.
pub fn galileo_nequick_g_native(
    coeffs: &GalileoNequickCoeffs,
    eval: GalileoNequickEval,
) -> Result<f64> {
    validate_galileo_nequick_coeffs(coeffs)?;
    validate_galileo_eval(eval)?;

    let delay_m = galileo_nequick_g_native_unchecked(coeffs, eval);
    validate_finite(delay_m, "ionosphere_delay_m")?;
    Ok(delay_m)
}

pub(crate) fn galileo_nequick_g_native_unchecked(
    coeffs: &GalileoNequickCoeffs,
    eval: GalileoNequickEval,
) -> f64 {
    let GalileoNequickEval {
        lat_deg,
        lon_deg,
        el_deg,
        t_gal_s,
        day_of_year,
        frequency_hz,
    } = eval;
    let mu_deg = galileo_modified_dip_latitude_deg(lat_deg, lon_deg);
    let az = galileo_effective_ionisation_level(coeffs, mu_deg);

    let local_time_h = (t_gal_s / 3600.0 + lon_deg / 15.0).rem_euclid(24.0);
    let solar = 0.5 + 0.5 * ((local_time_h - 14.0) * (2.0 * std::f64::consts::PI / 24.0)).cos();
    let diurnal = 0.35 + 0.65 * solar.max(0.0);
    let seasonal =
        1.0 + 0.08 * ((day_of_year - 172.0) * (2.0 * std::f64::consts::PI / 365.25)).cos();
    let equatorial = 1.0 + 0.35 * (-(mu_deg / 22.0).powi(2)).exp();

    let vertical_tecu = (2.5 + 0.135 * az) * diurnal * seasonal * equatorial;
    let mapping = single_layer_mapping(el_deg);
    let stec_tecu = vertical_tecu.max(0.0) * mapping;
    let delay_per_tecu_m = 40.3e16 / (frequency_hz * frequency_hz);
    stec_tecu * delay_per_tecu_m
}

/// Effective ionisation level `Az` from Galileo broadcast coefficients.
///
/// A zero broadcast set selects the Galileo-recommended default value of 63.7
/// solar-flux units. Nonzero sets are evaluated as `ai0 + ai1*mu + ai2*mu^2`
/// and clipped to the NeQuick-G driver range.
pub fn galileo_effective_ionisation_level(
    coeffs: &GalileoNequickCoeffs,
    modified_dip_latitude_deg: f64,
) -> f64 {
    if coeffs.ai0 == 0.0 && coeffs.ai1 == 0.0 && coeffs.ai2 == 0.0 {
        return 63.7;
    }
    (coeffs.ai0
        + coeffs.ai1 * modified_dip_latitude_deg
        + coeffs.ai2 * modified_dip_latitude_deg * modified_dip_latitude_deg)
        .clamp(0.0, 400.0)
}

fn galileo_modified_dip_latitude_deg(lat_deg: f64, lon_deg: f64) -> f64 {
    let lat = lat_deg * DEG_TO_RAD;
    let lon = lon_deg * DEG_TO_RAD;

    // Centered-dipole approximation used only to drive the broadcast Az
    // polynomial without shipping the full MODIP grid alongside this crate.
    let pole_lat = 80.37 * DEG_TO_RAD;
    let pole_lon = -72.62 * DEG_TO_RAD;
    let dip_lat =
        (lat.sin() * pole_lat.sin() + lat.cos() * pole_lat.cos() * (lon - pole_lon).cos()).asin();
    let magnetic_dip = (2.0 * dip_lat.tan()).atan();
    let denom = lat.cos().max(1.0e-12).sqrt();
    (magnetic_dip.tan() / denom).atan() * RAD_TO_DEG
}

fn single_layer_mapping(el_deg: f64) -> f64 {
    let el_rad = el_deg.max(0.1) * DEG_TO_RAD;
    let earth_radius_m = MEAN_EARTH_RADIUS_M;
    let shell_radius_m = earth_radius_m + 450_000.0;
    let arg = earth_radius_m / shell_radius_m * el_rad.cos();
    1.0 / (1.0 - arg * arg).max(1.0e-12).sqrt()
}

/// IONEX vertical-TEC-grid slant ionospheric group delay (positive meters).
///
/// Maps the parsed [`Ionex`] vertical-TEC grid to the line of sight in the
/// single-layer-model convention: a single-layer pierce point at the product's
/// shell height, an explicit four-term bilinear VTEC per map, a linear-in-time
/// blend between the two maps bracketing `epoch_j2000_s` (holding the endpoint
/// map outside coverage), the `1/sqrt(1 - s^2)` obliquity factor, and the
/// dispersive `40.3e16 / f^2` frequency scaling.
///
/// The receiver geodetic latitude/longitude come from `receiver` (height is
/// unused: the pierce point rides on the IONEX shell, not the antenna height).
/// The epoch is taken as integer J2000 seconds so it lands exactly on the
/// product's own epoch axis, with no float-rounded time entering the temporal
/// bracket. `frequency_hz` is the carrier on which the delay is reported. The
/// returned value is positive meters that increase the pseudorange.
pub fn ionex_slant_delay(
    ionex: &Ionex,
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    epoch_j2000_s: i64,
    frequency_hz: f64,
) -> Result<f64> {
    validate_receiver(receiver)?;
    validate_finite(elevation_rad, "elevation_rad")?;
    validate_elevation_rad(elevation_rad, "elevation_rad")?;
    validate_finite(azimuth_rad, "azimuth_rad")?;
    validate_frequency(frequency_hz)?;

    let delay_m = ionex_slant_delay_unchecked(
        ionex,
        receiver,
        elevation_rad,
        azimuth_rad,
        epoch_j2000_s,
        frequency_hz,
    );
    validate_finite(delay_m, "ionosphere_delay_m")?;
    Ok(delay_m)
}

fn ionex_slant_delay_unchecked(
    ionex: &Ionex,
    receiver: Wgs84Geodetic,
    elevation_rad: f64,
    azimuth_rad: f64,
    epoch_j2000_s: i64,
    frequency_hz: f64,
) -> f64 {
    slant::slant_delay_components(
        slant::PierceLineOfSight {
            lat_rad: receiver.lat_rad,
            lon_rad: receiver.lon_rad,
            az_rad: azimuth_rad,
            el_rad: elevation_rad,
        },
        frequency_hz,
        ionex.base_radius_km(),
        ionex.shell_height_km(),
        epoch_j2000_s,
        slant::VtecGridView {
            map_epochs: ionex.map_epochs(),
            maps: ionex.tec_maps(),
            lat_arr: ionex.lat_nodes_deg(),
            lon_arr: ionex.lon_nodes_deg(),
            dlat: ionex.dlat_deg(),
            dlon: ionex.dlon_deg(),
        },
    )
    .delay_m
}

fn validate_klobuchar_params(params: &KlobucharParams) -> Result<()> {
    for (index, &value) in params.alpha.iter().enumerate() {
        validate_finite(value, if index == 0 { "alpha" } else { "alpha[]" })?;
    }
    for (index, &value) in params.beta.iter().enumerate() {
        validate_finite(value, if index == 0 { "beta" } else { "beta[]" })?;
    }
    Ok(())
}

fn validate_galileo_nequick_coeffs(coeffs: &GalileoNequickCoeffs) -> Result<()> {
    validate_finite(coeffs.ai0, "ai0")?;
    validate_finite(coeffs.ai1, "ai1")?;
    validate_finite(coeffs.ai2, "ai2")
}

fn validate_galileo_eval(eval: GalileoNequickEval) -> Result<()> {
    validate_lat_deg(eval.lat_deg, "lat_deg")?;
    validate_lon_deg(eval.lon_deg, "lon_deg")?;
    validate_el_deg(eval.el_deg, "el_deg")?;
    validate_second_of_day(eval.t_gal_s, "t_gal_s")?;
    validate_finite(eval.day_of_year, "day_of_year")?;
    if !(1.0..367.0).contains(&eval.day_of_year) {
        return Err(invalid_input("day_of_year", "out of range"));
    }
    validate_frequency(eval.frequency_hz)
}

fn validate_receiver(receiver: Wgs84Geodetic) -> Result<()> {
    validate_finite(receiver.lat_rad, "receiver.lat_rad")?;
    validate_finite(receiver.lon_rad, "receiver.lon_rad")?;
    validate_finite(receiver.height_m, "receiver.height_m")?;
    if !(-core::f64::consts::FRAC_PI_2..=core::f64::consts::FRAC_PI_2).contains(&receiver.lat_rad) {
        return Err(invalid_input("receiver.lat_rad", "out of range"));
    }
    if !(-core::f64::consts::PI..=core::f64::consts::PI).contains(&receiver.lon_rad) {
        return Err(invalid_input("receiver.lon_rad", "out of range"));
    }
    Ok(())
}

fn validate_instant(epoch: Instant) -> Result<()> {
    match epoch.repr {
        InstantRepr::JulianDate(split) => {
            validate_finite(split.jd_whole, "epoch.jd_whole")?;
            validate_finite(split.fraction, "epoch.fraction")?;
            if !(-1.0..=1.0).contains(&split.fraction) {
                return Err(invalid_input("epoch.fraction", "out of range"));
            }
        }
        InstantRepr::Nanos(_) => {}
    }
    Ok(())
}

fn validate_lat_deg(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if !(-90.0..=90.0).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

fn validate_lon_deg(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if !(-180.0..=180.0).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

fn validate_elevation_rad(value: f64, field: &'static str) -> Result<()> {
    if !(0.0..=core::f64::consts::FRAC_PI_2).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

fn validate_el_deg(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if !(0.0..=90.0).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

fn validate_second_of_day(value: f64, field: &'static str) -> Result<()> {
    validate_finite(value, field)?;
    if !(0.0..SECONDS_PER_DAY).contains(&value) {
        return Err(invalid_input(field, "out of range"));
    }
    Ok(())
}

fn validate_frequency(frequency_hz: f64) -> Result<()> {
    validate_finite(frequency_hz, "frequency_hz")?;
    if frequency_hz <= 0.0 {
        return Err(invalid_input("frequency_hz", "not positive"));
    }
    Ok(())
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

/// GPS second-of-day in `[0, 86400)` carried by an instant.
///
/// The Klobuchar diurnal term needs the local-solar-time argument, built from
/// the GPS second-of-day. A Julian date's civil day begins at noon, so the
/// midnight day fraction is `(jd + 0.5)` modulo one.
///
/// Precision: for a split-Julian-date instant the second-of-day is
/// `day_fraction * 86400`, and `day_fraction` is itself a rounded binary
/// fraction of a day, so this recovers the second-of-day only to within the
/// float granularity of a day fraction (a few microseconds) - a
/// sub-nanometre-to-nanometre perturbation in the delay. The bit-exact (0-ULP)
/// contract is on the model kernel evaluated at an exact second-of-day, not on
/// this convenience conversion. An integer-nanosecond instant is exact (it
/// reduces by the seconds-per-day modulus).
fn gps_second_of_day(epoch: Instant) -> f64 {
    second_of_day_from_instant(epoch)
}

/// Fractional day-of-year carried by an instant, Jan 1 00:00:00 = 1.0.
fn fractional_day_of_year(epoch: Instant) -> f64 {
    fractional_day_of_year_from_instant(epoch)
}
