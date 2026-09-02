use crate::nmea::{self, Gga, GgaQuality, NmeaTalker, NmeaTime};
use crate::{Error, Result, Wgs84Geodetic};

#[derive(Clone, Debug, PartialEq)]
/// Position and receiver metadata consumed by [`format_gga`].
///
/// `Default::default` supplies a zero-degree, zero-height position with fix quality 1, 10 satellites, and HDOP 1.0.
pub struct GgaPosition {
    /// WGS84 geodetic latitude in degrees. [`format_gga`] accepts only finite values in `[-90, 90]` and converts it to radians before GGA coordinate formatting.
    pub lat_deg: f64,
    /// WGS84 geodetic longitude in degrees, positive east. [`format_gga`] accepts only finite values in `[-180, 180]` and converts it to radians before GGA coordinate formatting.
    pub lon_deg: f64,
    /// Height passed to [`crate::Wgs84Geodetic::new`] and then to the GGA altitude field. [`format_gga`] requires it finite; the writer emits it in meters with one decimal place and an `M` unit marker.
    pub height_m: f64,
    /// Numeric NMEA GGA fix-quality code. [`format_gga`] maps 0 through 8 to the corresponding [`crate::nmea::GgaQuality`] variant and preserves any other `u8` as `Other(value)` before writing it.
    pub fix_quality: u8,
    /// Number passed to GGA's satellites-used field. [`format_gga`] does not otherwise constrain it, and the NMEA writer emits it zero-padded to two digits.
    pub num_satellites: u8,
    /// Horizontal dilution of precision passed into GGA. [`format_gga`] requires a finite, non-negative value, and the NMEA writer emits it with two fractional digits.
    pub hdop: f64,
}

impl Default for GgaPosition {
    fn default() -> Self {
        Self {
            lat_deg: 0.0,
            lon_deg: 0.0,
            height_m: 0.0,
            fix_quality: 1,
            num_satellites: 10,
            hdop: 1.0,
        }
    }
}

/// Formats `position` and `utc_seconds_of_day` as a GPS NMEA GGA sentence and returns its UTF-8 bytes.
///
/// The formatter converts the degree coordinates to a WGS84 geodetic position, floors the time to centiseconds, maps the numeric fix-quality code, uses seven coordinate decimals, and omits geoid separation before writing the sentence.
///
/// It returns [`crate::Error::InvalidInput`] when the coordinates, height, HDOP, or time are non-finite, when latitude or longitude is outside its accepted interval, when HDOP is negative, or when the time is outside `[0, 86400)`; errors from geodetic construction, GGA construction, and sentence writing are mapped to the same variant.
pub fn format_gga(position: &GgaPosition, utc_seconds_of_day: f64) -> Result<Vec<u8>> {
    validate(position, utc_seconds_of_day)?;
    let geodetic = Wgs84Geodetic::new(
        position.lat_deg.to_radians(),
        position.lon_deg.to_radians(),
        position.height_m,
    )
    .map_err(|err| Error::InvalidInput(err.to_string()))?;
    let mut gga = Gga::vrs_position(
        geodetic,
        format_time(utc_seconds_of_day)?,
        quality(position.fix_quality),
        position.num_satellites,
        position.hdop,
        7,
    )
    .map_err(|err| Error::InvalidInput(err.to_string()))?;
    gga.geoid_separation_m = None;
    nmea::write_gga(NmeaTalker::System(crate::GnssSystem::Gps), &gga)
        .map(|sentence| sentence.into_bytes())
        .map_err(|err| Error::InvalidInput(err.to_string()))
}

fn validate(position: &GgaPosition, utc_seconds_of_day: f64) -> Result<()> {
    if !position.lat_deg.is_finite()
        || !position.lon_deg.is_finite()
        || !position.height_m.is_finite()
        || !position.hdop.is_finite()
        || !utc_seconds_of_day.is_finite()
    {
        return Err(Error::InvalidInput("GGA inputs must be finite".into()));
    }
    if !(-90.0..=90.0).contains(&position.lat_deg) {
        return Err(Error::InvalidInput("GGA latitude outside [-90, 90]".into()));
    }
    if !(-180.0..=180.0).contains(&position.lon_deg) {
        return Err(Error::InvalidInput(
            "GGA longitude outside [-180, 180]".into(),
        ));
    }
    if position.hdop < 0.0 {
        return Err(Error::InvalidInput("GGA HDOP must be non-negative".into()));
    }
    if !(0.0..86400.0).contains(&utc_seconds_of_day) {
        return Err(Error::InvalidInput("GGA time must be in [0, 86400)".into()));
    }
    Ok(())
}

fn format_time(seconds: f64) -> Result<NmeaTime> {
    NmeaTime::from_seconds_of_day_floor_centis(seconds)
        .map_err(|err| Error::InvalidInput(err.to_string()))
}

fn quality(value: u8) -> GgaQuality {
    match value {
        0 => GgaQuality::Invalid,
        1 => GgaQuality::GpsSps,
        2 => GgaQuality::Differential,
        3 => GgaQuality::Pps,
        4 => GgaQuality::RtkFixed,
        5 => GgaQuality::RtkFloat,
        6 => GgaQuality::Estimated,
        7 => GgaQuality::Manual,
        8 => GgaQuality::Simulator,
        other => GgaQuality::Other(other),
    }
}
