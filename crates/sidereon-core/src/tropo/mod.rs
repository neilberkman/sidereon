//! Tropospheric delay model.
//!
//! Computes the neutral-atmosphere (tropospheric) delay on a GNSS signal as a
//! Saastamoinen (1972) zenith hydrostatic and wet delay, driven by supplied
//! surface meteorology, mapped to the line of sight by the Niell (1996)
//! continued-fraction mapping functions (NMF). The zenith primitives and the
//! mapping primitives are exposed separately so a caller can apply the distinct
//! hydrostatic and wet mappings, and a convenience entry composes the full
//! slant delay.
//!
//! The tropospheric delay is non-dispersive: it has the same sign and magnitude
//! for code and carrier phase, and it is a positive additive range error. The
//! returned delays are positive meters that increase the measured pseudorange;
//! `delay_m > 0` means the signal arrived later and the pseudorange is too long
//! by `delay_m`.
//!
//! Angles are radians internally (`_rad`); height is the WGS84 ellipsoidal
//! height in meters carried by [`Wgs84Geodetic`]. Pressure is hectopascals,
//! temperature is kelvin, and relative humidity is a unit fraction in `[0, 1]`.

mod saastamoinen;
mod vmf;
mod zwd;

use crate::astro::time::civil::{
    fractional_day_of_year_from_instant, julian_date_from_instant, mjd_from_jd,
};
use crate::astro::time::model::{Instant, InstantRepr};

use crate::error::{Error, Result};
use crate::frame::Wgs84Geodetic;

pub(crate) use saastamoinen::slant_components;
pub use zwd::{
    tropo_delay_xyz as tropo_zwd_delay_xyz, zenith_wet_delay as zwd_zenith_wet_delay,
    AltitudeClamp, ZwdEpoch, ZwdProfile, ZwdSlantOptions,
};

const MIN_CALENDAR_JULIAN_DATE: f64 = 1_721_425.0;
const MAX_CALENDAR_JULIAN_DATE: f64 = 5_373_485.0;

/// Surface meteorological conditions at the receiver.
///
/// These are the inputs to the Saastamoinen zenith delays. Pressure is in
/// hectopascals (millibars) because that is the unit the troposphere formulas
/// and meteorological products use; temperature is absolute (kelvin) to avoid a
/// Celsius zero-point slip; relative humidity is a unit fraction in `[0, 1]`,
/// not a percentage.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Met {
    /// Total atmospheric pressure in hectopascals (= millibars).
    pub pressure_hpa: f64,
    /// Ambient temperature in kelvin.
    pub temperature_k: f64,
    /// Relative humidity as a unit fraction in `[0, 1]` (0.5 == 50%).
    pub relative_humidity: f64,
}

impl Met {
    /// Construct surface meteorology from explicit values.
    pub fn new(pressure_hpa: f64, temperature_k: f64, relative_humidity: f64) -> Result<Self> {
        validate_met_values(pressure_hpa, temperature_k, relative_humidity)?;
        Ok(Self {
            pressure_hpa,
            temperature_k,
            relative_humidity,
        })
    }

    pub(crate) const fn new_unchecked(
        pressure_hpa: f64,
        temperature_k: f64,
        relative_humidity: f64,
    ) -> Self {
        Self {
            pressure_hpa,
            temperature_k,
            relative_humidity,
        }
    }

    /// Standard-atmosphere pressure and temperature for an ellipsoidal height,
    /// carrying the supplied relative humidity through unchanged.
    ///
    /// A convenience for callers without live meteorology. The height is clamped
    /// to be non-negative before the standard pressure and temperature lapses
    /// are applied, so a below-sea-level height yields the sea-level values.
    pub fn standard(height_m: f64, relative_humidity: f64) -> Result<Self> {
        validate_finite(height_m, "height_m")?;
        validate_relative_humidity(relative_humidity)?;
        let met = Self::standard_unchecked(height_m, relative_humidity);
        validate_met(met)?;
        Ok(met)
    }

    pub(crate) fn standard_unchecked(height_m: f64, relative_humidity: f64) -> Self {
        let s = saastamoinen::standard_atmosphere(height_m, relative_humidity);
        Self {
            pressure_hpa: s.pressure_hpa,
            temperature_k: s.temperature_k,
            relative_humidity: s.relative_humidity,
        }
    }
}

/// Tropospheric zenith-delay model selector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TropoModel {
    /// Saastamoinen (1972) zenith hydrostatic + wet delays.
    Saastamoinen,
    /// Saastamoinen hydrostatic term plus an exponential ZWD wet profile.
    ZwdAltitudeScaled(ZwdProfile),
}

/// Tropospheric mapping-function selector.
///
/// Not `Eq`: the [`MappingModel::Vmf1`] variant carries the floating-point `a`
/// coefficients of the VMF1 data product, so only `PartialEq` is available.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MappingModel {
    /// Niell (1996) mapping functions (NMF), driven by a climatological table.
    Niell,
    /// Vienna Mapping Function 1 (VMF1, Böhm 2006), site-wise form: the
    /// continued fraction with the supplied hydrostatic/wet `a` coefficients
    /// from the VMF1 numerical-weather-model data product at the station and
    /// epoch, and VMF1's own `b`, `c` coefficients. No height correction (the
    /// site-wise `a` are already valid at the station). See [`vmf`].
    Vmf1 {
        /// Hydrostatic `a` coefficient from the VMF1 data product.
        ah: f64,
        /// Wet `a` coefficient from the VMF1 data product.
        aw: f64,
    },
}

/// Zenith tropospheric delay split into its hydrostatic and wet parts.
///
/// The two components are returned separately because the Niell mapping applies
/// a distinct hydrostatic and wet mapping factor; the total slant delay is
/// `dry_m * mapping.dry + wet_m * mapping.wet`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ZenithDelay {
    /// Zenith hydrostatic (dry) delay in positive meters.
    pub dry_m: f64,
    /// Zenith wet delay in positive meters.
    pub wet_m: f64,
}

/// Dimensionless mapping factors at a given elevation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MappingFactors {
    /// Hydrostatic mapping factor (includes the height correction).
    pub dry: f64,
    /// Wet mapping factor.
    pub wet: f64,
}

/// Zenith tropospheric delay (hydrostatic + wet) from supplied meteorology.
///
/// The receiver geodetic latitude and ellipsoidal height come from `receiver`;
/// the surface pressure, temperature, and humidity come from `met`. Both
/// components are returned as positive meters. The possibly-negative ellipsoidal
/// height is used with its sign.
pub fn tropo_zenith(model: TropoModel, receiver: Wgs84Geodetic, met: Met) -> Result<ZenithDelay> {
    validate_receiver(receiver)?;
    validate_met(met)?;
    validate_tropo_model(model)?;

    let delay = tropo_zenith_unchecked(model, receiver, met);
    validate_finite(delay.dry_m, "zenith_dry_m")?;
    validate_finite(delay.wet_m, "zenith_wet_m")?;
    Ok(delay)
}

pub(crate) fn tropo_zenith_unchecked(
    model: TropoModel,
    receiver: Wgs84Geodetic,
    met: Met,
) -> ZenithDelay {
    match model {
        TropoModel::Saastamoinen => {
            let z = saastamoinen::zenith_delays(
                met.pressure_hpa,
                met.temperature_k,
                met.relative_humidity,
                receiver.lat_rad,
                receiver.height_m,
            );
            ZenithDelay {
                dry_m: z.zhd_m,
                wet_m: z.zwd_m,
            }
        }
        TropoModel::ZwdAltitudeScaled(profile) => {
            let z = saastamoinen::zenith_delays(
                met.pressure_hpa,
                met.temperature_k,
                met.relative_humidity,
                receiver.lat_rad,
                receiver.height_m,
            );
            ZenithDelay {
                dry_m: z.zhd_m,
                wet_m: zwd::zenith_wet_delay_unchecked(profile, receiver.height_m),
            }
        }
    }
}

/// Niell hydrostatic and wet mapping factors at the given elevation.
///
/// The mapping depends on the receiver geodetic latitude and ellipsoidal height
/// (via `receiver`) and on the fractional day-of-year (derived from `epoch`),
/// hence both are arguments. The hydrostatic mapping carries the height
/// correction; the wet mapping does not.
pub fn tropo_mapping(
    model: MappingModel,
    elevation_rad: f64,
    receiver: Wgs84Geodetic,
    epoch: Instant,
) -> Result<MappingFactors> {
    validate_mapping_model(model)?;
    validate_elevation(elevation_rad)?;
    validate_receiver(receiver)?;
    validate_instant(epoch)?;

    let mapping = tropo_mapping_unchecked(model, elevation_rad, receiver, epoch);
    validate_finite(mapping.dry, "mapping_dry")?;
    validate_finite(mapping.wet, "mapping_wet")?;
    Ok(mapping)
}

pub(crate) fn tropo_mapping_unchecked(
    model: MappingModel,
    elevation_rad: f64,
    receiver: Wgs84Geodetic,
    epoch: Instant,
) -> MappingFactors {
    match model {
        MappingModel::Niell => {
            let doy = fractional_day_of_year(epoch);
            let m = saastamoinen::niell_mapping(
                elevation_rad,
                receiver.lat_rad,
                receiver.height_m,
                doy,
            );
            MappingFactors {
                dry: m.mh,
                wet: m.mw,
            }
        }
        MappingModel::Vmf1 { ah, aw } => {
            let mjd = modified_julian_date(epoch);
            let m = vmf::vmf1_mapping(elevation_rad, receiver.lat_rad, mjd, ah, aw);
            MappingFactors {
                dry: m.mh,
                wet: m.mw,
            }
        }
    }
}

/// Full slant tropospheric delay in positive meters.
///
/// Composes the Saastamoinen zenith delays and the Niell mapping into the total
/// line-of-sight delay `dry_m * mapping.dry + wet_m * mapping.wet`. The delay is
/// zero for an elevation at or below the horizon and for a height outside the
/// model's validity range; inside that range the result is positive.
///
/// Note on bit-exactness: the fractional day-of-year is derived from `epoch` at
/// this boundary. Its integer day is exact (from the calendar date); the
/// within-day fraction carries the float granularity of a split Julian date,
/// which enters the Niell seasonal term so weakly that it is below the last bit
/// in practice. The 0-ULP parity contract is on the kernel evaluated at an
/// exact day-of-year; this wrapper agrees with it to within that bound.
pub fn tropo_slant(
    elevation_rad: f64,
    receiver: Wgs84Geodetic,
    met: Met,
    epoch: Instant,
) -> Result<f64> {
    validate_elevation(elevation_rad)?;
    validate_receiver(receiver)?;
    validate_met(met)?;
    validate_instant(epoch)?;

    let delay_m = tropo_slant_unchecked(elevation_rad, receiver, met, epoch);
    validate_finite(delay_m, "tropo_slant_m")?;
    Ok(delay_m)
}

pub(crate) fn tropo_slant_unchecked(
    elevation_rad: f64,
    receiver: Wgs84Geodetic,
    met: Met,
    epoch: Instant,
) -> f64 {
    let doy = fractional_day_of_year(epoch);
    slant_components(
        elevation_rad,
        receiver,
        met.pressure_hpa,
        met.temperature_k,
        met.relative_humidity,
        doy,
    )
    .slant_m
}

/// Slant tropospheric delay (meters) with a selectable mapping function.
///
/// Composes the Saastamoinen zenith hydrostatic and wet delays with the chosen
/// mapping (`MappingModel::Niell` or `MappingModel::Vmf1`):
/// `zenith.dry * mapping.dry + zenith.wet * mapping.wet`. With `Niell` this is
/// bit-identical to [`tropo_slant_unchecked`] (same zenith and mapping
/// primitives, same combination order). The same permissive validity gate
/// applies: elevation at or below the horizon, or a height outside the Met-path
/// range, yields zero.
pub(crate) fn tropo_slant_with_mapping_unchecked(
    model: MappingModel,
    elevation_rad: f64,
    receiver: Wgs84Geodetic,
    met: Met,
    epoch: Instant,
) -> f64 {
    if elevation_rad <= 0.0
        || receiver.height_m < saastamoinen::MET_GATE_LOW_M
        || receiver.height_m > saastamoinen::MET_GATE_HI_M
    {
        return 0.0;
    }
    let zenith = tropo_zenith_unchecked(TropoModel::Saastamoinen, receiver, met);
    let mapping = tropo_mapping_unchecked(model, elevation_rad, receiver, epoch);
    zenith.dry_m * mapping.dry + zenith.wet_m * mapping.wet
}

fn validate_tropo_model(model: TropoModel) -> Result<()> {
    match model {
        TropoModel::Saastamoinen => Ok(()),
        TropoModel::ZwdAltitudeScaled(profile) => zwd::validate_profile(profile),
    }
}

fn validate_mapping_model(model: MappingModel) -> Result<()> {
    match model {
        MappingModel::Niell => Ok(()),
        MappingModel::Vmf1 { ah, aw } => {
            validate_finite(ah, "mapping.vmf1.ah")?;
            if ah <= 0.0 {
                return Err(invalid_input("mapping.vmf1.ah", "not positive"));
            }
            validate_finite(aw, "mapping.vmf1.aw")?;
            if aw <= 0.0 {
                return Err(invalid_input("mapping.vmf1.aw", "not positive"));
            }
            Ok(())
        }
    }
}

fn validate_met(met: Met) -> Result<()> {
    validate_met_values(met.pressure_hpa, met.temperature_k, met.relative_humidity)
}

fn validate_met_values(
    pressure_hpa: f64,
    temperature_k: f64,
    relative_humidity: f64,
) -> Result<()> {
    validate_finite(pressure_hpa, "pressure_hpa")?;
    if pressure_hpa <= 0.0 {
        return Err(invalid_input("pressure_hpa", "not positive"));
    }
    validate_finite(temperature_k, "temperature_k")?;
    if temperature_k <= 0.0 {
        return Err(invalid_input("temperature_k", "not positive"));
    }
    validate_relative_humidity(relative_humidity)
}

fn validate_relative_humidity(relative_humidity: f64) -> Result<()> {
    validate_finite(relative_humidity, "relative_humidity")?;
    if !(0.0..=1.0).contains(&relative_humidity) {
        return Err(invalid_input("relative_humidity", "out of range"));
    }
    Ok(())
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
    if !(saastamoinen::MET_GATE_LOW_M..=saastamoinen::MET_GATE_HI_M).contains(&receiver.height_m) {
        return Err(invalid_input("receiver.height_m", "out of range"));
    }
    Ok(())
}

fn validate_elevation(elevation_rad: f64) -> Result<()> {
    validate_finite(elevation_rad, "elevation_rad")?;
    if !(0.0..=core::f64::consts::FRAC_PI_2).contains(&elevation_rad) {
        return Err(invalid_input("elevation_rad", "out of range"));
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
            let jd = split.jd_whole + split.fraction;
            validate_finite(jd, "epoch.julian_date")?;
            if !(MIN_CALENDAR_JULIAN_DATE..=MAX_CALENDAR_JULIAN_DATE).contains(&jd) {
                return Err(invalid_input("epoch.julian_date", "out of range"));
            }
        }
        InstantRepr::Nanos(_) => {}
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

/// Fractional day-of-year carried by an instant, Jan 1 00:00:00 = 1.0.
///
/// The Niell seasonal term needs the continuous day-of-year (it carries the
/// fractional time of day). This converts the instant to a calendar date and
/// returns the day number plus the within-day fraction. The conversion is the
/// standard civil-date algorithm from the Julian day number; the within-day
/// fraction comes from the day part shifted from the noon Julian-date origin to
/// midnight.
fn fractional_day_of_year(epoch: Instant) -> f64 {
    fractional_day_of_year_from_instant(epoch)
}

/// Modified Julian date (`jd - 2400000.5`) carried by an instant.
///
/// The VMF1 seasonal `c` expression references the modified Julian date (the
/// TU Wien `vmf1.f` convention `doy = mjd - 44239 + 1 - 28`), so the VMF mapping
/// arm derives the MJD here rather than the Niell fractional day-of-year.
fn modified_julian_date(epoch: Instant) -> f64 {
    mjd_from_jd(julian_date_from_instant(epoch))
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests;
