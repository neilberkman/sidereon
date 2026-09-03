use crate::format::{Diagnostics, Parsed, RecordRef, Warning, WarningKind};
use crate::validate::{self, FieldError};
use crate::GnssSystem;

use super::{
    Gga, GgaQuality, Gll, Gsa, GsaFixMode, GsaSelectionMode, Gst, Gsv, GsvSatellite,
    NmeaCoordinate, NmeaDate, NmeaError, NmeaSatNumber, NmeaSignalId, NmeaTalker, NmeaTime, Rmc,
    RmcStatus, Vtg, Zda,
};

#[derive(Debug, Clone, PartialEq)]
/// A decoded sentence returned by [`crate::nmea::parse_sentence`], combining the parsed address prefix with the payload parser selected by the address suffix.
pub struct NmeaSentence {
    /// The address prefix parsed by [`NmeaTalker::parse`]. [`crate::nmea::NmeaAccumulator`] uses it to distinguish GSV groups.
    pub talker: NmeaTalker,
    /// The typed payload selected by the sentence's three-letter address suffix. [`crate::nmea::NmeaAccumulator`] dispatches it to the corresponding epoch record.
    pub body: NmeaBody,
}

#[derive(Debug, Clone, PartialEq)]
/// Parsed payloads for the sentence types handled by [`crate::nmea::parse_sentence`]: GGA, RMC, GSA, GSV, GST, VTG, GLL, and ZDA.
pub enum NmeaBody {
    /// The GGA payload produced by `parse_gga`; [`crate::nmea::NmeaAccumulator`] retains the first value in [`crate::nmea::EpochSnapshot::gga`], and [`crate::nmea::write_gga`] serializes a [`Gga`] value as GGA.
    Gga(Gga),
    /// The RMC payload produced by `parse_rmc`; [`crate::nmea::NmeaAccumulator`] retains the first value, and the accumulator uses its time and date for epoch metadata.
    Rmc(Rmc),
    /// The GSA payload produced by `parse_gsa`; [`crate::nmea::NmeaAccumulator`] stores one [`crate::nmea::GsaEntry`] per system and warns on duplicate or differing DOP data.
    Gsa(Gsa),
    /// The GSV payload produced by `parse_gsv`; [`crate::nmea::NmeaAccumulator`] groups messages by talker and signal, then extends satellites in message order.
    Gsv(Gsv),
    /// The GST payload produced by `parse_gst`; [`crate::nmea::NmeaAccumulator`] retains the first value and uses its time as an epoch anchor.
    Gst(Gst),
    /// The VTG payload produced by `parse_vtg`; [`crate::nmea::NmeaAccumulator`] retains the first value, but the accumulator does not use VTG as a timestamp source.
    Vtg(Vtg),
    /// The GLL payload produced by `parse_gll`; [`crate::nmea::NmeaAccumulator`] retains the first value and uses its time as an epoch anchor.
    Gll(Gll),
    /// The ZDA payload produced by `parse_zda`; [`crate::nmea::NmeaAccumulator`] retains the first value and uses its time and date for epoch metadata.
    Zda(Zda),
}

pub(crate) struct FramedSentence<'a> {
    pub delimiter: u8,
    pub body: &'a str,
    pub diagnostics: Diagnostics,
}

pub(crate) fn checksum_body(body: &str) -> u8 {
    body.bytes().fold(0, |acc, byte| acc ^ byte)
}

pub(crate) fn frame_sentence(line: &str) -> Result<FramedSentence<'_>, NmeaError> {
    let mut diagnostics = Diagnostics::new();
    let trimmed = line.trim_end_matches(['\r', '\n']);
    let start =
        trimmed
            .bytes()
            .position(|b| b == b'$' || b == b'!')
            .ok_or(NmeaError::NotFramed {
                reason: "no NMEA start delimiter",
            })?;
    if start > 0 {
        diagnostics.push_warning(Warning {
            at: RecordRef::default(),
            kind: WarningKind::Mismatch,
        });
    }
    let sentence = &trimmed[start..];
    if sentence.len() > 1024 {
        return Err(NmeaError::NotFramed {
            reason: "sentence over length cap",
        });
    }
    if !sentence.is_ascii() {
        return Err(NmeaError::NotFramed {
            reason: "non-ASCII byte",
        });
    }
    let delimiter = sentence.as_bytes()[0];
    let rest = &sentence[1..];
    let (body, checksum) = if let Some(star) = rest.rfind('*') {
        let checksum_token = rest.get(star + 1..star + 3).unwrap_or("");
        if checksum_token.len() != 2 || !checksum_token.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(NmeaError::NotFramed {
                reason: "malformed checksum",
            });
        }
        let trailing = &rest[star + 3..];
        if !trailing
            .bytes()
            .all(|b| b == b' ' || b == b'\r' || b == b'\n')
        {
            diagnostics.push_warning(Warning {
                at: RecordRef::default(),
                kind: WarningKind::Mismatch,
            });
        }
        let stated = u8::from_str_radix(checksum_token, 16).map_err(|_| NmeaError::NotFramed {
            reason: "malformed checksum",
        })?;
        (&rest[..star], Some(stated))
    } else {
        diagnostics.push_warning(Warning {
            at: RecordRef::default(),
            kind: WarningKind::MissingMetadata,
        });
        (rest, None)
    };
    if let Some(stated) = checksum {
        let computed = checksum_body(body);
        if computed != stated {
            return Err(NmeaError::ChecksumMismatch { computed, stated });
        }
    }
    Ok(FramedSentence {
        delimiter,
        body,
        diagnostics,
    })
}

pub(crate) fn parse_framed(framed: FramedSentence<'_>) -> Result<Parsed<NmeaSentence>, NmeaError> {
    if framed.delimiter == b'!' {
        return Err(NmeaError::UnsupportedType {
            address: "encapsulated sentence".to_string(),
        });
    }
    let mut diagnostics = framed.diagnostics;
    let mut parts = framed.body.split(',');
    let address = parts.next().unwrap_or_default();
    if address.starts_with('P') {
        return Err(NmeaError::Proprietary {
            address: address.to_string(),
        });
    }
    if address.len() != 5 {
        return Err(NmeaError::UnsupportedType {
            address: address.to_string(),
        });
    }
    let talker = NmeaTalker::parse(&address[..2]);
    let sentence_type = &address[2..];
    let fields: Vec<&str> = parts.collect();
    let body = match sentence_type {
        "GGA" => NmeaBody::Gga(parse_gga(&fields)?),
        "RMC" => NmeaBody::Rmc(parse_rmc(&fields)?),
        "GSA" => NmeaBody::Gsa(parse_gsa(talker, &fields, &mut diagnostics)?),
        "GSV" => NmeaBody::Gsv(parse_gsv(talker, &fields, &mut diagnostics)?),
        "GST" => NmeaBody::Gst(parse_gst(&fields)?),
        "VTG" => NmeaBody::Vtg(parse_vtg(&fields)?),
        "GLL" => NmeaBody::Gll(parse_gll(&fields)?),
        "ZDA" => NmeaBody::Zda(parse_zda(&fields)?),
        _ => {
            return Err(NmeaError::UnsupportedType {
                address: address.to_string(),
            })
        }
    };
    Ok(Parsed::new(NmeaSentence { talker, body }, diagnostics))
}

fn parse_gga(fields: &[&str]) -> Result<Gga, FieldError> {
    Ok(Gga {
        time: parse_opt_time(fields.first().copied())?,
        latitude: parse_opt_coordinate(fields.get(1).copied(), fields.get(2).copied(), true)?,
        longitude: parse_opt_coordinate(fields.get(3).copied(), fields.get(4).copied(), false)?,
        quality: parse_opt_quality(fields.get(5).copied())?,
        satellites_used: parse_opt_u8_range(fields.get(6).copied(), "satellites used", 0, 99)?,
        hdop: parse_opt_f64(fields.get(7).copied(), "hdop")?,
        altitude_msl_m: parse_opt_unit_f64(
            fields.get(8).copied(),
            fields.get(9).copied(),
            "altitude msl",
        )?,
        geoid_separation_m: parse_opt_unit_f64(
            fields.get(10).copied(),
            fields.get(11).copied(),
            "geoid separation",
        )?,
        differential_age_s: parse_opt_f64(fields.get(12).copied(), "differential age")?,
        differential_station_id: parse_opt_u16_range(
            fields.get(13).copied(),
            "differential station id",
            0,
            9999,
        )?,
    })
}

fn parse_rmc(fields: &[&str]) -> Result<Rmc, FieldError> {
    let magnetic = parse_opt_signed_pair(
        fields.get(9).copied(),
        fields.get(10).copied(),
        "magnetic variation",
        "EW",
    )?;
    Ok(Rmc {
        time: parse_opt_time(fields.first().copied())?,
        status: parse_opt_rmc_status(fields.get(1).copied())?,
        latitude: parse_opt_coordinate(fields.get(2).copied(), fields.get(3).copied(), true)?,
        longitude: parse_opt_coordinate(fields.get(4).copied(), fields.get(5).copied(), false)?,
        speed_over_ground_kn: parse_opt_f64(fields.get(6).copied(), "speed over ground")?,
        course_over_ground_deg: parse_opt_f64(fields.get(7).copied(), "course over ground")?,
        date: parse_opt_date_rmc(fields.get(8).copied())?,
        magnetic_variation_deg: magnetic,
        faa_mode: parse_opt_char(fields.get(11).copied(), "faa mode")?,
        navigational_status: parse_opt_char(fields.get(12).copied(), "navigation status")?,
    })
}

fn parse_gsa(
    talker: NmeaTalker,
    fields: &[&str],
    diagnostics: &mut Diagnostics,
) -> Result<Gsa, FieldError> {
    let system_id = parse_opt_u8_range(fields.get(17).copied(), "system id", 0, 9)?;
    let system = match system_id {
        Some(1) => Some(GnssSystem::Gps),
        Some(2) => Some(GnssSystem::Glonass),
        Some(3) => Some(GnssSystem::Galileo),
        Some(4) => Some(GnssSystem::BeiDou),
        Some(5) => Some(GnssSystem::Qzss),
        Some(6) => Some(GnssSystem::Navic),
        Some(_) => {
            diagnostics.push_warning(Warning {
                at: RecordRef::default(),
                kind: WarningKind::Degraded,
            });
            None
        }
        None => talker.system(),
    };
    let mut satellites = Vec::new();
    for field in fields.iter().skip(2).take(12) {
        if let Some(sat) = parse_opt_sat_number(Some(*field), system, diagnostics)? {
            satellites.push(sat);
        }
    }
    Ok(Gsa {
        selection_mode: parse_opt_gsa_selection(fields.first().copied())?,
        fix_mode: parse_opt_gsa_fix(fields.get(1).copied())?,
        satellites,
        pdop: parse_opt_f64(fields.get(14).copied(), "pdop")?,
        hdop: parse_opt_f64(fields.get(15).copied(), "hdop")?,
        vdop: parse_opt_f64(fields.get(16).copied(), "vdop")?,
        system_id,
        system,
    })
}

fn parse_gsv(
    talker: NmeaTalker,
    fields: &[&str],
    diagnostics: &mut Diagnostics,
) -> Result<Gsv, FieldError> {
    let total_messages = parse_required_u8_range(fields.first().copied(), "gsv total", 1, 15)?;
    let message_number = parse_required_u8_range(
        fields.get(1).copied(),
        "gsv message number",
        1,
        total_messages,
    )?;
    let satellites_in_view =
        parse_opt_u16_range(fields.get(2).copied(), "satellites in view", 0, 999)?;
    let tail = if fields.len() > 3 { &fields[3..] } else { &[] };
    let rem = tail.len() % 4;
    let (sat_fields, signal) = if rem == 1 {
        let signal = parse_opt_signal_id(tail.last().copied(), talker.system(), diagnostics)?;
        (&tail[..tail.len() - 1], signal)
    } else {
        if rem == 2 || rem == 3 {
            diagnostics.push_warning(Warning {
                at: RecordRef::default(),
                kind: WarningKind::Mismatch,
            });
        }
        (tail, None)
    };
    let mut satellites = Vec::new();
    for chunk in sat_fields.chunks(4) {
        let sat_number =
            parse_opt_sat_number(chunk.first().copied(), talker.system(), diagnostics)?;
        let elevation_deg = parse_opt_i16_range(chunk.get(1).copied(), "elevation", -90, 90)?;
        let azimuth_deg = parse_opt_u16_range(chunk.get(2).copied(), "azimuth", 0, 359)?;
        let cn0_db_hz = parse_opt_u8_range(chunk.get(3).copied(), "cn0", 0, 99)?;
        satellites.push(GsvSatellite {
            sat_number,
            elevation_deg,
            azimuth_deg,
            cn0_db_hz,
        });
    }
    Ok(Gsv {
        total_messages,
        message_number,
        satellites_in_view,
        satellites,
        signal,
    })
}

fn parse_gst(fields: &[&str]) -> Result<Gst, FieldError> {
    Ok(Gst {
        time: parse_opt_time(fields.first().copied())?,
        rms_range_residual_m: parse_opt_f64(fields.get(1).copied(), "range residual rms")?,
        semi_major_error_m: parse_opt_f64(fields.get(2).copied(), "semi major error")?,
        semi_minor_error_m: parse_opt_f64(fields.get(3).copied(), "semi minor error")?,
        orientation_deg: parse_opt_f64(fields.get(4).copied(), "error orientation")?,
        latitude_sigma_m: parse_opt_f64(fields.get(5).copied(), "latitude sigma")?,
        longitude_sigma_m: parse_opt_f64(fields.get(6).copied(), "longitude sigma")?,
        altitude_sigma_m: parse_opt_f64(fields.get(7).copied(), "altitude sigma")?,
    })
}

fn parse_vtg(fields: &[&str]) -> Result<Vtg, FieldError> {
    Ok(Vtg {
        course_true_deg: parse_opt_tagged_f64(
            fields.first().copied(),
            fields.get(1).copied(),
            "course true",
            "T",
        )?,
        course_magnetic_deg: parse_opt_tagged_f64(
            fields.get(2).copied(),
            fields.get(3).copied(),
            "course magnetic",
            "M",
        )?,
        speed_kn: parse_opt_tagged_f64(
            fields.get(4).copied(),
            fields.get(5).copied(),
            "speed knots",
            "N",
        )?,
        speed_kmh: parse_opt_tagged_f64(
            fields.get(6).copied(),
            fields.get(7).copied(),
            "speed kmh",
            "K",
        )?,
        faa_mode: parse_opt_char(fields.get(8).copied(), "faa mode")?,
    })
}

fn parse_gll(fields: &[&str]) -> Result<Gll, FieldError> {
    Ok(Gll {
        latitude: parse_opt_coordinate(fields.first().copied(), fields.get(1).copied(), true)?,
        longitude: parse_opt_coordinate(fields.get(2).copied(), fields.get(3).copied(), false)?,
        time: parse_opt_time(fields.get(4).copied())?,
        status: parse_opt_rmc_status(fields.get(5).copied())?,
        faa_mode: parse_opt_char(fields.get(6).copied(), "faa mode")?,
    })
}

fn parse_zda(fields: &[&str]) -> Result<Zda, FieldError> {
    let day = parse_opt_u8_range(fields.get(1).copied(), "zda day", 1, 31)?;
    let month = parse_opt_u8_range(fields.get(2).copied(), "zda month", 1, 12)?;
    let year = parse_opt_u16_range(fields.get(3).copied(), "zda year", 0, 9999)?;
    let date = match (year, month, day) {
        (Some(year), Some(month), Some(day)) => Some(NmeaDate::new(year, month, day)?),
        (None, None, None) => None,
        _ => return Err(FieldError::Missing { field: "zda date" }),
    };
    Ok(Zda {
        time: parse_opt_time(fields.first().copied())?,
        date,
        local_zone_hours: parse_opt_i8_range(fields.get(4).copied(), "zone hours", -13, 13)?,
        local_zone_minutes: parse_opt_u8_range(fields.get(5).copied(), "zone minutes", 0, 59)?,
    })
}

fn parse_opt_time(token: Option<&str>) -> Result<Option<NmeaTime>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => NmeaTime::parse(token).map(Some),
    }
}

fn parse_opt_date_rmc(token: Option<&str>) -> Result<Option<NmeaDate>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => NmeaDate::parse_rmc(token).map(Some),
    }
}

fn parse_opt_coordinate(
    value: Option<&str>,
    hemisphere: Option<&str>,
    is_latitude: bool,
) -> Result<Option<NmeaCoordinate>, FieldError> {
    let value = value.unwrap_or("").trim();
    let hemisphere = hemisphere.unwrap_or("").trim();
    match (value.is_empty(), hemisphere.is_empty()) {
        (true, true) => Ok(None),
        (false, false) => NmeaCoordinate::parse(value, hemisphere, is_latitude).map(Some),
        _ => Err(FieldError::Missing {
            field: if is_latitude {
                "latitude pair"
            } else {
                "longitude pair"
            },
        }),
    }
}

fn parse_opt_quality(token: Option<&str>) -> Result<Option<GgaQuality>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => GgaQuality::parse(token).map(Some),
    }
}

fn parse_opt_rmc_status(token: Option<&str>) -> Result<Option<RmcStatus>, FieldError> {
    match parse_opt_char(token, "rmc status")? {
        None => Ok(None),
        Some('A') => Ok(Some(RmcStatus::Valid)),
        Some('V') => Ok(Some(RmcStatus::Warning)),
        Some(ch) => Ok(Some(RmcStatus::Other(ch))),
    }
}

fn parse_opt_gsa_selection(token: Option<&str>) -> Result<Option<GsaSelectionMode>, FieldError> {
    match parse_opt_char(token, "gsa selection")? {
        None => Ok(None),
        Some('M') => Ok(Some(GsaSelectionMode::Manual)),
        Some('A') => Ok(Some(GsaSelectionMode::Automatic)),
        Some(ch) => Ok(Some(GsaSelectionMode::Other(ch))),
    }
}

fn parse_opt_gsa_fix(token: Option<&str>) -> Result<Option<GsaFixMode>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => {
            let value = validate::strict_int::<u8>(token, "gsa fix mode")?;
            Ok(Some(match value {
                1 => GsaFixMode::None,
                2 => GsaFixMode::TwoD,
                3 => GsaFixMode::ThreeD,
                other => GsaFixMode::Other(other),
            }))
        }
    }
}

fn parse_opt_char(token: Option<&str>, field: &'static str) -> Result<Option<char>, FieldError> {
    let token = token.unwrap_or("").trim();
    if token.is_empty() {
        return Ok(None);
    }
    let mut chars = token.chars();
    let Some(ch) = chars.next() else {
        return Ok(None);
    };
    if chars.next().is_some() {
        return Err(FieldError::IntParse {
            field,
            value: token.to_string(),
        });
    }
    Ok(Some(ch))
}

fn parse_opt_f64(token: Option<&str>, field: &'static str) -> Result<Option<f64>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => validate::strict_f64(token, field).map(Some),
    }
}

fn parse_opt_tagged_f64(
    value: Option<&str>,
    unit: Option<&str>,
    field: &'static str,
    expected_unit: &str,
) -> Result<Option<f64>, FieldError> {
    let value = value.unwrap_or("").trim();
    let unit = unit.unwrap_or("").trim();
    if value.is_empty() {
        return Ok(None);
    }
    if unit != expected_unit {
        return Err(FieldError::OutOfRange {
            field: "unit",
            min: 0.0,
            max: 0.0,
            upper_inclusive: true,
        });
    }
    validate::strict_f64(value, field).map(Some)
}

fn parse_opt_signed_pair(
    value: Option<&str>,
    direction: Option<&str>,
    field: &'static str,
    valid_dirs: &str,
) -> Result<Option<f64>, FieldError> {
    let Some(mut value) = parse_opt_f64(value, field)? else {
        return Ok(None);
    };
    let direction = direction.unwrap_or("").trim();
    if direction.len() != 1 || !valid_dirs.contains(direction) {
        return Err(FieldError::OutOfRange {
            field: "direction",
            min: 0.0,
            max: 0.0,
            upper_inclusive: true,
        });
    }
    if direction == "W" || direction == "S" {
        value = -value;
    }
    Ok(Some(value))
}

fn parse_opt_unit_f64(
    value: Option<&str>,
    unit: Option<&str>,
    field: &'static str,
) -> Result<Option<f64>, FieldError> {
    let value = value.unwrap_or("").trim();
    let unit = unit.unwrap_or("").trim();
    if value.is_empty() {
        if unit.is_empty() {
            return Ok(None);
        }
        return Err(FieldError::Missing { field });
    }
    if unit != "M" {
        return Err(FieldError::OutOfRange {
            field: "unit",
            min: 0.0,
            max: 0.0,
            upper_inclusive: true,
        });
    }
    validate::strict_f64(value, field).map(Some)
}

fn parse_required_u8_range(
    token: Option<&str>,
    field: &'static str,
    min: u8,
    max: u8,
) -> Result<u8, FieldError> {
    parse_opt_u8_range(token, field, min, max)?.ok_or(FieldError::Missing { field })
}

fn parse_opt_u8_range(
    token: Option<&str>,
    field: &'static str,
    min: u8,
    max: u8,
) -> Result<Option<u8>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => {
            let value = validate::strict_int::<u8>(token, field)?;
            if value < min || value > max {
                Err(FieldError::OutOfRange {
                    field,
                    min: f64::from(min),
                    max: f64::from(max),
                    upper_inclusive: true,
                })
            } else {
                Ok(Some(value))
            }
        }
    }
}

fn parse_opt_i8_range(
    token: Option<&str>,
    field: &'static str,
    min: i8,
    max: i8,
) -> Result<Option<i8>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => {
            let value = validate::strict_int::<i8>(token, field)?;
            if value < min || value > max {
                Err(FieldError::OutOfRange {
                    field,
                    min: f64::from(min),
                    max: f64::from(max),
                    upper_inclusive: true,
                })
            } else {
                Ok(Some(value))
            }
        }
    }
}

fn parse_opt_i16_range(
    token: Option<&str>,
    field: &'static str,
    min: i16,
    max: i16,
) -> Result<Option<i16>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => {
            let value = validate::strict_int::<i16>(token, field)?;
            if value < min || value > max {
                Err(FieldError::OutOfRange {
                    field,
                    min: f64::from(min),
                    max: f64::from(max),
                    upper_inclusive: true,
                })
            } else {
                Ok(Some(value))
            }
        }
    }
}

fn parse_opt_u16_range(
    token: Option<&str>,
    field: &'static str,
    min: u16,
    max: u16,
) -> Result<Option<u16>, FieldError> {
    match token.unwrap_or("").trim() {
        "" => Ok(None),
        token => {
            let value = validate::strict_int::<u16>(token, field)?;
            if value < min || value > max {
                Err(FieldError::OutOfRange {
                    field,
                    min: f64::from(min),
                    max: f64::from(max),
                    upper_inclusive: true,
                })
            } else {
                Ok(Some(value))
            }
        }
    }
}

fn parse_opt_sat_number(
    token: Option<&str>,
    context: Option<GnssSystem>,
    diagnostics: &mut Diagnostics,
) -> Result<Option<NmeaSatNumber>, FieldError> {
    let token = token.unwrap_or("").trim();
    if token.is_empty() {
        return Ok(None);
    }
    let raw = validate::strict_int::<u16>(token, "satellite number")?;
    let resolved = super::fields::resolve_sat_number(context, raw);
    if resolved.is_none() {
        diagnostics.push_warning(Warning {
            at: RecordRef::default(),
            kind: WarningKind::Degraded,
        });
    }
    Ok(Some(NmeaSatNumber { raw, resolved }))
}

fn parse_opt_signal_id(
    token: Option<&str>,
    system: Option<GnssSystem>,
    diagnostics: &mut Diagnostics,
) -> Result<Option<NmeaSignalId>, FieldError> {
    let token = token.unwrap_or("").trim();
    if token.is_empty() {
        return Ok(None);
    }
    let id = if token.len() == 1 {
        u8::from_str_radix(token, 16).map_err(|_| FieldError::IntParse {
            field: "signal id",
            value: token.to_string(),
        })?
    } else {
        validate::strict_int::<u8>(token, "signal id")?
    };
    let signal = NmeaSignalId { system, id };
    if id > 15 || signal.carrier_band().is_none() {
        diagnostics.push_warning(Warning {
            at: RecordRef::default(),
            kind: WarningKind::Degraded,
        });
    }
    Ok(Some(signal))
}
