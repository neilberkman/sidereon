use crate::frequencies::CarrierBand;
use crate::validate::{self, FieldError};
use crate::{GnssSatelliteId, GnssSystem, Wgs84Geodetic};
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// An NMEA UTC time split into validated clock components and nanoseconds.
/// The parser preserves the fractional digit count, which the GGA writer uses to enforce its two-decimal contract.
pub struct NmeaTime {
    /// The hour parsed from the first two digits; valid input limits it to 0 through 23.
    pub hour: u8,
    /// The minute parsed from the middle two digits; valid input limits it to 0 through 59.
    pub minute: u8,
    /// The whole second parsed from the final two digits; NMEA leap-second input may use 60.
    pub second: u8,
    /// Nanoseconds obtained by scaling the fractional time token; epoch keys and UTC conversion consume this value.
    pub nanos: u32,
    /// The number of fractional digits in the source token; GGA output requires this to be 2.
    pub decimals: u8,
}

impl NmeaTime {
    /// Parses an NMEA time with six whole-time digits and up to nine fractional digits.
    /// Empty, malformed, out-of-range, and over-precise tokens return the corresponding [`FieldError`].
    pub fn parse(token: &str) -> Result<Self, FieldError> {
        let token = token.trim();
        if token.is_empty() {
            return Err(FieldError::Missing { field: "nmea time" });
        }
        let (whole, frac) = token.split_once('.').unwrap_or((token, ""));
        if whole.len() != 6 || !whole.bytes().all(|b| b.is_ascii_digit()) {
            return Err(FieldError::IntParse {
                field: "nmea time",
                value: token.to_string(),
            });
        }
        if frac.len() > 9 || !frac.bytes().all(|b| b.is_ascii_digit()) {
            return Err(FieldError::IntParse {
                field: "nmea time fraction",
                value: token.to_string(),
            });
        }
        let hour = whole[0..2]
            .parse::<u8>()
            .map_err(|_| FieldError::IntParse {
                field: "nmea time hour",
                value: token.to_string(),
            })?;
        let minute = whole[2..4]
            .parse::<u8>()
            .map_err(|_| FieldError::IntParse {
                field: "nmea time minute",
                value: token.to_string(),
            })?;
        let second = whole[4..6]
            .parse::<u8>()
            .map_err(|_| FieldError::IntParse {
                field: "nmea time second",
                value: token.to_string(),
            })?;
        if hour > 23 || minute > 59 || second > 60 {
            return Err(FieldError::InvalidCivilTime {
                field: "nmea time",
                hour: i64::from(hour),
                minute: i64::from(minute),
                second: f64::from(second),
            });
        }
        let decimals = frac.len() as u8;
        let frac_value = if frac.is_empty() {
            0
        } else {
            frac.parse::<u32>().map_err(|_| FieldError::IntParse {
                field: "nmea time fraction",
                value: token.to_string(),
            })?
        };
        let nanos = frac_value * 10_u32.pow(9 - u32::from(decimals));
        Ok(Self {
            hour,
            minute,
            second,
            nanos,
            decimals,
        })
    }

    /// Returns the exact clock tuple used to compare epoch anchors, including fractional seconds.
    pub fn key(self) -> (u8, u8, u8, u32) {
        (self.hour, self.minute, self.second, self.nanos)
    }

    /// Converts finite seconds of day in `[0, 86400)` to a time truncated to centiseconds.
    /// The returned value always records two fractional digits for compatibility with GGA output.
    pub fn from_seconds_of_day_floor_centis(seconds: f64) -> Result<Self, crate::nmea::NmeaError> {
        if !seconds.is_finite() || !(0.0..86_400.0).contains(&seconds) {
            return Err(crate::nmea::NmeaError::InvalidInput {
                field: "time",
                reason: "must be finite and in [0, 86400)",
            });
        }
        let whole = seconds.floor() as u32;
        let fractional = (seconds - f64::from(whole)).clamp(0.0, 1.0);
        let centis = (Duration::from_secs_f64(fractional).as_nanos() / 10_000_000).min(99) as u32;
        Ok(Self {
            hour: (whole / 3600) as u8,
            minute: ((whole % 3600) / 60) as u8,
            second: (whole % 60) as u8,
            nanos: centis * 10_000_000,
            decimals: 2,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// An NMEA latitude or longitude retained as integer degrees and scaled minutes.
/// The sign comes from the hemisphere or source decimal value and conversion to an epoch position uses WGS-84 radians.
pub struct NmeaCoordinate {
    /// The fixed-width degree component, bounded by 90 for latitude or 180 for longitude.
    pub degrees: u16,
    /// Whole and fractional minutes multiplied by `10^decimals`, preserving the source precision.
    pub minutes_scaled: u64,
    /// The number of fractional minute digits, limited to nine by the constructors.
    pub decimals: u8,
    /// Whether the coordinate is south or west, or was supplied with a negative decimal sign.
    pub negative: bool,
}

impl NmeaCoordinate {
    /// Parses an NMEA `ddmm.m...` or `dddmm.m...` coordinate with its hemisphere.
    /// The parser enforces the latitude/longitude hemisphere pairing, minute bounds, and coordinate limit.
    pub fn parse(value: &str, hemisphere: &str, is_latitude: bool) -> Result<Self, FieldError> {
        let value = value.trim();
        let hemisphere = hemisphere.trim();
        if value.is_empty() || hemisphere.is_empty() {
            return Err(FieldError::Missing {
                field: if is_latitude { "latitude" } else { "longitude" },
            });
        }
        let (negative, valid_hemisphere) = match hemisphere {
            "N" => (false, is_latitude),
            "S" => (true, is_latitude),
            "E" => (false, !is_latitude),
            "W" => (true, !is_latitude),
            _ => (false, false),
        };
        if !valid_hemisphere {
            return Err(FieldError::OutOfRange {
                field: "hemisphere",
                min: 0.0,
                max: 0.0,
                upper_inclusive: true,
            });
        }
        let degree_digits = if is_latitude { 2 } else { 3 };
        if value.len() < degree_digits + 2
            || !value[..degree_digits + 2]
                .bytes()
                .all(|b| b.is_ascii_digit())
        {
            return Err(FieldError::FloatParse {
                field: if is_latitude { "latitude" } else { "longitude" },
                value: value.to_string(),
            });
        }
        let degrees = value[..degree_digits]
            .parse::<u16>()
            .map_err(|_| FieldError::IntParse {
                field: "coordinate degrees",
                value: value.to_string(),
            })?;
        let minute_token = &value[degree_digits..];
        let (whole_minutes, minute_frac) =
            minute_token.split_once('.').unwrap_or((minute_token, ""));
        if whole_minutes.len() != 2
            || !whole_minutes.bytes().all(|b| b.is_ascii_digit())
            || minute_frac.len() > 9
            || !minute_frac.bytes().all(|b| b.is_ascii_digit())
        {
            return Err(FieldError::FloatParse {
                field: "coordinate minutes",
                value: value.to_string(),
            });
        }
        let decimals = minute_frac.len() as u8;
        let scale = 10_u64.pow(u32::from(decimals));
        let minutes_whole = whole_minutes
            .parse::<u64>()
            .map_err(|_| FieldError::IntParse {
                field: "coordinate minutes",
                value: value.to_string(),
            })?;
        let frac_scaled = if minute_frac.is_empty() {
            0
        } else {
            minute_frac
                .parse::<u64>()
                .map_err(|_| FieldError::IntParse {
                    field: "coordinate minute fraction",
                    value: value.to_string(),
                })?
        };
        let minutes_scaled = minutes_whole * scale + frac_scaled;
        let degree_max = if is_latitude { 90 } else { 180 };
        if degrees > degree_max
            || minutes_whole > 59
            || (degrees == degree_max && minutes_scaled != 0)
        {
            return Err(FieldError::OutOfRange {
                field: if is_latitude { "latitude" } else { "longitude" },
                min: 0.0,
                max: f64::from(degree_max),
                upper_inclusive: true,
            });
        }
        Ok(Self {
            degrees,
            minutes_scaled,
            decimals,
            negative,
        })
    }

    /// Converts signed decimal degrees to rounded NMEA degrees and scaled minutes.
    /// Minute rounding uses half-away-from-zero behavior and carries 60 rounded minutes into the next degree.
    pub fn from_degrees(
        degrees: f64,
        is_latitude: bool,
        decimals: u8,
    ) -> Result<Self, crate::nmea::NmeaError> {
        if !degrees.is_finite() || decimals > 9 {
            return Err(crate::nmea::NmeaError::InvalidInput {
                field: "coordinate",
                reason: "must be finite with at most 9 decimals",
            });
        }
        let max = if is_latitude { 90.0 } else { 180.0 };
        if degrees.abs() > max {
            return Err(crate::nmea::NmeaError::InvalidInput {
                field: "coordinate",
                reason: "out of range",
            });
        }
        let negative = degrees.is_sign_negative();
        let abs = degrees.abs();
        let mut whole_degrees = abs.floor() as u16;
        let scale = 10_u64.pow(u32::from(decimals));
        let minutes = (abs - f64::from(whole_degrees)) * 60.0;
        let mut minutes_scaled = round_half_away_from_zero(minutes * scale as f64) as u64;
        if minutes_scaled >= 60 * scale {
            whole_degrees += 1;
            minutes_scaled -= 60 * scale;
        }
        if f64::from(whole_degrees) > max {
            return Err(crate::nmea::NmeaError::InvalidInput {
                field: "coordinate",
                reason: "rounding exceeded coordinate bound",
            });
        }
        Ok(Self {
            degrees: whole_degrees,
            minutes_scaled,
            decimals,
            negative,
        })
    }

    /// Reconstructs signed decimal degrees from the stored degree/minute representation.
    pub fn degrees_f64(&self) -> f64 {
        let sign = if self.negative { -1.0 } else { 1.0 };
        let scale = 10_f64.powi(i32::from(self.decimals));
        sign * (f64::from(self.degrees) + (self.minutes_scaled as f64 / scale) / 60.0)
    }

    /// Converts the reconstructed signed decimal degrees to radians for geodetic position construction.
    pub fn radians(&self) -> f64 {
        self.degrees_f64().to_radians()
    }
}

fn round_half_away_from_zero(value: f64) -> i64 {
    if value >= 0.0 {
        (value + 0.5).floor() as i64
    } else {
        (value - 0.5).ceil() as i64
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// A validated civil date used by RMC/ZDA records and epoch date carry-forward.
pub struct NmeaDate {
    /// The validated calendar year, with RMC two-digit years expanded by [`NmeaDate::parse_rmc`].
    pub year: u16,
    /// The validated calendar month used with the civil month-length table.
    pub month: u8,
    /// The validated day within `month`; month and year rollover is handled by [`NmeaDate::next_day`].
    pub day: u8,
}

impl NmeaDate {
    /// Parses an RMC `ddmmyy` date and applies the NMEA 80-year pivot before calendar validation.
    pub fn parse_rmc(token: &str) -> Result<Self, FieldError> {
        let token = token.trim();
        if token.len() != 6 || !token.bytes().all(|b| b.is_ascii_digit()) {
            return Err(FieldError::IntParse {
                field: "nmea date",
                value: token.to_string(),
            });
        }
        let day = parse_u8(&token[0..2], "nmea date day")?;
        let month = parse_u8(&token[2..4], "nmea date month")?;
        let yy = parse_u8(&token[4..6], "nmea date year")?;
        let year = if yy >= 80 {
            1900 + u16::from(yy)
        } else {
            2000 + u16::from(yy)
        };
        Self::new(year, month, day)
    }

    /// Constructs a date only when the month and day identify a valid civil calendar date.
    pub fn new(year: u16, month: u8, day: u8) -> Result<Self, FieldError> {
        let max_day = crate::astro::time::civil::days_in_month(i64::from(year), i64::from(month));
        if max_day == 0 || day == 0 || i64::from(day) > max_day {
            return Err(FieldError::InvalidCivilDate {
                field: "nmea date",
                year: i64::from(year),
                month: i64::from(month),
                day: i64::from(day),
            });
        }
        Ok(Self { year, month, day })
    }

    /// Advances a valid date by one day, including month and December-to-January rollover.
    pub fn next_day(self) -> Self {
        let max_day =
            crate::astro::time::civil::days_in_month(i64::from(self.year), i64::from(self.month))
                as u8;
        if self.day < max_day {
            Self {
                day: self.day + 1,
                ..self
            }
        } else if self.month < 12 {
            Self {
                month: self.month + 1,
                day: 1,
                ..self
            }
        } else {
            Self {
                year: self.year + 1,
                month: 1,
                day: 1,
            }
        }
    }
}

fn parse_u8(token: &str, field: &'static str) -> Result<u8, FieldError> {
    token.parse::<u8>().map_err(|_| FieldError::IntParse {
        field,
        value: token.to_string(),
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Classifies an NMEA two-byte talker prefix as a GNSS system, combined source, or raw address.
pub enum NmeaTalker {
    /// A recognized talker prefix carrying its corresponding [`GnssSystem`].
    System(GnssSystem),
    /// The `GN` prefix, which combines multiple GNSS systems and has no single system context.
    Combined,
    /// An unrecognized two-byte prefix retained as raw bytes for round-tripping when ASCII.
    Other([u8; 2]),
}

impl NmeaTalker {
    /// Parses the known NMEA talker aliases and retains other two-byte tokens.
    /// Tokens with any other length become `Other(*b"??")`.
    pub fn parse(token: &str) -> Self {
        match token.as_bytes() {
            b"GP" => Self::System(GnssSystem::Gps),
            b"GL" => Self::System(GnssSystem::Glonass),
            b"GA" => Self::System(GnssSystem::Galileo),
            b"GB" | b"BD" => Self::System(GnssSystem::BeiDou),
            b"GQ" | b"QZ" => Self::System(GnssSystem::Qzss),
            b"GI" => Self::System(GnssSystem::Navic),
            b"GN" => Self::Combined,
            [a, b] => Self::Other([*a, *b]),
            _ => Self::Other(*b"??"),
        }
    }

    /// Returns the canonical two-byte code used when writing a sentence.
    /// Raw codes are accepted only when both bytes are ASCII; SBAS uses the GPS `GP` code.
    pub fn code(self) -> Result<[u8; 2], crate::nmea::NmeaError> {
        match self {
            Self::System(GnssSystem::Gps) | Self::System(GnssSystem::Sbas) => Ok(*b"GP"),
            Self::System(GnssSystem::Glonass) => Ok(*b"GL"),
            Self::System(GnssSystem::Galileo) => Ok(*b"GA"),
            Self::System(GnssSystem::BeiDou) => Ok(*b"GB"),
            Self::System(GnssSystem::Qzss) => Ok(*b"GQ"),
            Self::System(GnssSystem::Navic) => Ok(*b"GI"),
            Self::Combined => Ok(*b"GN"),
            Self::Other(raw) if raw.iter().all(u8::is_ascii) => Ok(raw),
            Self::Other(_) => Err(crate::nmea::NmeaError::InvalidInput {
                field: "talker",
                reason: "must be ASCII",
            }),
        }
    }

    /// Returns the embedded system for [`NmeaTalker::System`] and `None` for combined or raw talkers.
    pub fn system(self) -> Option<GnssSystem> {
        match self {
            Self::System(system) => Some(system),
            Self::Combined | Self::Other(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Decodes the numeric quality field carried by a GGA sentence.
pub enum GgaQuality {
    /// Quality code 0.
    Invalid,
    /// Quality code 1.
    GpsSps,
    /// Quality code 2.
    Differential,
    /// Quality code 3.
    Pps,
    /// Quality code 4.
    RtkFixed,
    /// Quality code 5.
    RtkFloat,
    /// Quality code 6.
    Estimated,
    /// Quality code 7.
    Manual,
    /// Quality code 8.
    Simulator,
    /// A quality code other than 0 through 8, retained unchanged.
    Other(u8),
}

impl GgaQuality {
    /// Parses a strict unsigned quality code, retaining unsupported values in [`GgaQuality::Other`].
    pub fn parse(token: &str) -> Result<Self, FieldError> {
        let value = validate::strict_int::<u8>(token, "gga quality")?;
        Ok(match value {
            0 => Self::Invalid,
            1 => Self::GpsSps,
            2 => Self::Differential,
            3 => Self::Pps,
            4 => Self::RtkFixed,
            5 => Self::RtkFloat,
            6 => Self::Estimated,
            7 => Self::Manual,
            8 => Self::Simulator,
            other => Self::Other(other),
        })
    }

    /// Returns the protocol byte represented by this quality value.
    pub fn value(self) -> u8 {
        match self {
            Self::Invalid => 0,
            Self::GpsSps => 1,
            Self::Differential => 2,
            Self::Pps => 3,
            Self::RtkFixed => 4,
            Self::RtkFloat => 5,
            Self::Estimated => 6,
            Self::Manual => 7,
            Self::Simulator => 8,
            Self::Other(value) => value,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
/// The optional fields decoded from a GGA fix sentence.
/// Meter-tagged heights are kept separately so an epoch can combine MSL altitude with geoid separation.
pub struct Gga {
    /// The optional UTC time used to anchor an epoch.
    pub time: Option<NmeaTime>,
    /// The optional latitude coordinate; writing requires it to be paired with [`Gga::longitude`].
    pub latitude: Option<NmeaCoordinate>,
    /// The optional longitude coordinate; writing requires it to be paired with [`Gga::latitude`].
    pub longitude: Option<NmeaCoordinate>,
    /// The optional decoded GGA quality code.
    pub quality: Option<GgaQuality>,
    /// The optional number of satellites used, parsed in the inclusive range 0 through 99.
    pub satellites_used: Option<u8>,
    /// The optional finite horizontal dilution of precision.
    pub hdop: Option<f64>,
    /// The optional mean-sea-level altitude in meters, parsed with an `M` unit tag.
    pub altitude_msl_m: Option<f64>,
    /// The optional geoid separation in meters, parsed with an `M` unit tag.
    pub geoid_separation_m: Option<f64>,
    /// The optional differential-correction age in seconds.
    pub differential_age_s: Option<f64>,
    /// The optional differential station identifier, parsed in the inclusive range 0 through 9999.
    pub differential_station_id: Option<u16>,
}

impl Gga {
    /// Builds a GGA record from a WGS-84 position and fix metadata.
    /// Coordinates are rounded at `coordinate_decimals`; the supplied ellipsoidal height is stored as MSL altitude, geoid separation is zero, and differential fields are absent.
    pub fn vrs_position(
        position: Wgs84Geodetic,
        time: NmeaTime,
        quality: GgaQuality,
        satellites_used: u8,
        hdop: f64,
        coordinate_decimals: u8,
    ) -> Result<Self, crate::nmea::NmeaError> {
        if !hdop.is_finite() || hdop < 0.0 {
            return Err(crate::nmea::NmeaError::InvalidInput {
                field: "hdop",
                reason: "must be finite and non-negative",
            });
        }
        Ok(Self {
            time: Some(time),
            latitude: Some(NmeaCoordinate::from_degrees(
                position.lat_rad.to_degrees(),
                true,
                coordinate_decimals,
            )?),
            longitude: Some(NmeaCoordinate::from_degrees(
                position.lon_rad.to_degrees(),
                false,
                coordinate_decimals,
            )?),
            quality: Some(quality),
            satellites_used: Some(satellites_used),
            hdop: Some(hdop),
            altitude_msl_m: Some(position.height_m),
            geoid_separation_m: Some(0.0),
            differential_age_s: None,
            differential_station_id: None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// An NMEA satellite number together with its optional resolved GNSS identity.
pub struct NmeaSatNumber {
    /// The unsigned satellite number exactly as supplied in the NMEA field.
    pub raw: u16,
    /// The GNSS system/PRN resolved from sentence context, when its numbering range is known.
    pub resolved: Option<GnssSatelliteId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// An NMEA signal identifier paired with the talker-derived GNSS system context.
pub struct NmeaSignalId {
    /// The optional GNSS system used to interpret [`NmeaSignalId::id`].
    pub system: Option<GnssSystem>,
    /// The numeric signal code parsed from the GSV signal field.
    pub id: u8,
}

impl NmeaSignalId {
    /// Maps a known system and signal code to a [`CarrierBand`].
    /// Missing system context and codes without a table entry return `None`.
    pub fn carrier_band(&self) -> Option<CarrierBand> {
        let system = self.system?;
        match system {
            GnssSystem::Gps | GnssSystem::Sbas => match self.id {
                1..=3 => Some(CarrierBand::L1),
                4..=6 => Some(CarrierBand::L2),
                7 | 8 => Some(CarrierBand::L5),
                _ => None,
            },
            GnssSystem::Glonass => match self.id {
                1 | 2 => Some(CarrierBand::G1),
                3 | 4 => Some(CarrierBand::G2),
                _ => None,
            },
            GnssSystem::Galileo => match self.id {
                1 => Some(CarrierBand::E5a),
                2 => Some(CarrierBand::E5b),
                3 => Some(CarrierBand::E5),
                4 | 5 => Some(CarrierBand::E6),
                6 | 7 => Some(CarrierBand::E1),
                _ => None,
            },
            GnssSystem::BeiDou => match self.id {
                1 | 2 => Some(CarrierBand::B1i),
                3 | 4 => Some(CarrierBand::B1c),
                5 => Some(CarrierBand::B2a),
                6 => Some(CarrierBand::B2b),
                7 => Some(CarrierBand::B2),
                8 | 9 => Some(CarrierBand::B3i),
                _ => None,
            },
            GnssSystem::Qzss => match self.id {
                1..=4 => Some(CarrierBand::L1),
                5 | 6 => Some(CarrierBand::L2),
                7 | 8 => Some(CarrierBand::L5),
                _ => None,
            },
            GnssSystem::Navic => match self.id {
                1 | 3 => Some(CarrierBand::L5),
                _ => None,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Decodes the RMC/GLL status character while retaining unrecognized values.
pub enum RmcStatus {
    /// Status character `A`.
    Valid,
    /// Status character `V`.
    Warning,
    /// Any other nonempty status character.
    Other(char),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Decodes the GSA satellite-selection mode character.
pub enum GsaSelectionMode {
    /// Selection character `M`.
    Manual,
    /// Selection character `A`.
    Automatic,
    /// Any other nonempty selection character.
    Other(char),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Decodes the numeric GSA fix-mode field.
pub enum GsaFixMode {
    /// Fix-mode value 1.
    None,
    /// Fix-mode value 2.
    TwoD,
    /// Fix-mode value 3.
    ThreeD,
    /// Any fix-mode value other than 1 through 3.
    Other(u8),
}

#[derive(Debug, Clone, PartialEq)]
/// The optional navigation and position fields decoded from an RMC sentence.
pub struct Rmc {
    /// The optional UTC time used as an epoch anchor.
    pub time: Option<NmeaTime>,
    /// The optional status decoded from `A`, `V`, or another character.
    pub status: Option<RmcStatus>,
    /// The optional latitude coordinate from the RMC value/hemisphere pair.
    pub latitude: Option<NmeaCoordinate>,
    /// The optional longitude coordinate from the RMC value/hemisphere pair.
    pub longitude: Option<NmeaCoordinate>,
    /// The optional speed over ground in knots.
    pub speed_over_ground_kn: Option<f64>,
    /// The optional course over ground in degrees.
    pub course_over_ground_deg: Option<f64>,
    /// The optional date from the RMC `ddmmyy` field.
    pub date: Option<NmeaDate>,
    /// The optional magnetic variation in degrees after applying the parser's negative multiplier for direction `W`.
    pub magnetic_variation_deg: Option<f64>,
    /// The optional single-character FAA mode.
    pub faa_mode: Option<char>,
    /// The optional single-character navigational status.
    pub navigational_status: Option<char>,
}

#[derive(Debug, Clone, PartialEq)]
/// The fields decoded from a GSA satellite-selection and dilution record.
pub struct Gsa {
    /// The optional manual/automatic selection mode.
    pub selection_mode: Option<GsaSelectionMode>,
    /// The optional numeric fix mode.
    pub fix_mode: Option<GsaFixMode>,
    /// Nonempty satellite fields, retained in their sentence order.
    pub satellites: Vec<NmeaSatNumber>,
    /// The optional position dilution of precision.
    pub pdop: Option<f64>,
    /// The optional horizontal dilution of precision.
    pub hdop: Option<f64>,
    /// The optional vertical dilution of precision.
    pub vdop: Option<f64>,
    /// The optional numeric GNSS system identifier from the sentence.
    pub system_id: Option<u8>,
    /// The optional system context used to resolve the satellite numbers.
    pub system: Option<GnssSystem>,
}

#[derive(Debug, Clone, PartialEq)]
/// One four-field satellite group decoded from a GSV page.
pub struct GsvSatellite {
    /// The optional satellite number and resolved identity.
    pub sat_number: Option<NmeaSatNumber>,
    /// The optional elevation in degrees, accepted from -90 through 90.
    pub elevation_deg: Option<i16>,
    /// The optional azimuth in degrees, accepted from 0 through 359.
    pub azimuth_deg: Option<u16>,
    /// The optional carrier-to-noise density ratio in dB-Hz, accepted from 0 through 99.
    pub cn0_db_hz: Option<u8>,
}

#[derive(Debug, Clone, PartialEq)]
/// One GSV page and its decoded satellite groups.
/// The epoch accumulator uses the page count, number, talker, and signal to combine a multi-page view.
pub struct Gsv {
    /// The required total page count, accepted from 1 through 15.
    pub total_messages: u8,
    /// The required page number, accepted from 1 through [`Gsv::total_messages`].
    pub message_number: u8,
    /// The optional claimed number of satellites in view, accepted from 0 through 999.
    pub satellites_in_view: Option<u16>,
    /// Satellite groups in their page order, including groups with absent individual fields.
    pub satellites: Vec<GsvSatellite>,
    /// The optional trailing signal identifier when the page has one signal field.
    pub signal: Option<NmeaSignalId>,
}

#[derive(Debug, Clone, PartialEq)]
/// The optional error statistics decoded from a GST sentence.
pub struct Gst {
    /// The optional UTC time used as an epoch anchor.
    pub time: Option<NmeaTime>,
    /// The optional RMS range residual in meters.
    pub rms_range_residual_m: Option<f64>,
    /// The optional semi-major position error in meters.
    pub semi_major_error_m: Option<f64>,
    /// The optional semi-minor position error in meters.
    pub semi_minor_error_m: Option<f64>,
    /// The optional error-ellipse orientation in degrees.
    pub orientation_deg: Option<f64>,
    /// The optional latitude standard deviation in meters.
    pub latitude_sigma_m: Option<f64>,
    /// The optional longitude standard deviation in meters.
    pub longitude_sigma_m: Option<f64>,
    /// The optional altitude standard deviation in meters.
    pub altitude_sigma_m: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
/// The optional course, speed, and FAA fields decoded from a VTG sentence.
pub struct Vtg {
    /// The optional true course in degrees, paired with a `T` unit tag.
    pub course_true_deg: Option<f64>,
    /// The optional magnetic course in degrees, paired with an `M` unit tag.
    pub course_magnetic_deg: Option<f64>,
    /// The optional speed in knots, paired with an `N` unit tag.
    pub speed_kn: Option<f64>,
    /// The optional speed in kilometers per hour, paired with a `K` unit tag.
    pub speed_kmh: Option<f64>,
    /// The optional single-character FAA mode.
    pub faa_mode: Option<char>,
}

#[derive(Debug, Clone, PartialEq)]
/// The optional position, time, status, and FAA fields decoded from a GLL sentence.
pub struct Gll {
    /// The optional latitude coordinate from the GLL value/hemisphere pair.
    pub latitude: Option<NmeaCoordinate>,
    /// The optional longitude coordinate from the GLL value/hemisphere pair.
    pub longitude: Option<NmeaCoordinate>,
    /// The optional UTC time used as an epoch anchor.
    pub time: Option<NmeaTime>,
    /// The optional status decoded with the RMC `A`/`V`/other mapping.
    pub status: Option<RmcStatus>,
    /// The optional single-character FAA mode.
    pub faa_mode: Option<char>,
}

#[derive(Debug, Clone, PartialEq)]
/// The UTC time, civil date, and local offset fields decoded from a ZDA sentence.
pub struct Zda {
    /// The optional UTC time used as an epoch anchor.
    pub time: Option<NmeaTime>,
    /// The optional date, present only when all day/month/year fields are present and valid.
    pub date: Option<NmeaDate>,
    /// The optional local time-zone hour offset, accepted from -13 through 13.
    pub local_zone_hours: Option<i8>,
    /// The optional local time-zone minute offset, accepted from 0 through 59.
    pub local_zone_minutes: Option<u8>,
}

pub(crate) fn resolve_sat_number(context: Option<GnssSystem>, raw: u16) -> Option<GnssSatelliteId> {
    let candidate = match context {
        Some(GnssSystem::Gps) => match raw {
            1..=32 => Some((GnssSystem::Gps, raw)),
            33..=64 => Some((GnssSystem::Sbas, raw - 13)),
            _ => None,
        },
        Some(GnssSystem::Glonass) => match raw {
            65..=99 => Some((GnssSystem::Glonass, raw - 64)),
            1..=35 => Some((GnssSystem::Glonass, raw)),
            _ => None,
        },
        Some(GnssSystem::Galileo) => match raw {
            1..=36 => Some((GnssSystem::Galileo, raw)),
            _ => None,
        },
        Some(GnssSystem::BeiDou) => match raw {
            1..=64 => Some((GnssSystem::BeiDou, raw)),
            _ => None,
        },
        Some(GnssSystem::Qzss) => match raw {
            1..=10 => Some((GnssSystem::Qzss, raw)),
            193..=202 => Some((GnssSystem::Qzss, raw - 192)),
            _ => None,
        },
        Some(GnssSystem::Navic) => match raw {
            1..=15 => Some((GnssSystem::Navic, raw)),
            _ => None,
        },
        Some(GnssSystem::Sbas) => match raw {
            33..=64 => Some((GnssSystem::Sbas, raw - 13)),
            120..=158 => Some((GnssSystem::Sbas, raw - 100)),
            _ => None,
        },
        None => match raw {
            1..=32 => Some((GnssSystem::Gps, raw)),
            33..=64 => Some((GnssSystem::Sbas, raw - 13)),
            65..=99 => Some((GnssSystem::Glonass, raw - 64)),
            193..=202 => Some((GnssSystem::Qzss, raw - 192)),
            _ => None,
        },
    }?;
    let prn = u8::try_from(candidate.1).ok()?;
    GnssSatelliteId::new(candidate.0, prn).ok()
}
