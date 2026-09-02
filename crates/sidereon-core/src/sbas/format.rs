use crate::astro::time::gnss::{seconds_of_week_from_calendar, week_from_calendar};
use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::error::{Error, Result};
use crate::id::GnssSatelliteId;

use super::message::SbasWireForm;
use super::store::sbas_prn_to_sat;

#[derive(Clone, Debug, PartialEq)]
/// One SBAS block recovered from an EMS or RTKLIB log line.
///
/// The line parsers retain recognized records in input order and store the
/// converted satellite, GPST epoch, wire-form classification, and decoded
/// bytes.
pub struct SbasLogBlock {
    /// SBAS satellite returned by [`sbas_prn_to_sat`] for a supported broadcast
    /// PRN (120 through 158, inclusive).
    pub satellite_id: GnssSatelliteId,
    /// GPST week and seconds-of-week associated with the logged block.
    ///
    /// EMS calendar fields are converted using the Sunday-origin GPST week;
    /// RTKLIB lines provide the week and time-of-week directly.
    pub epoch: GnssWeekTow,
    /// Wire representation inferred from the decoded byte count: 29 bytes use
    /// [`SbasWireForm::Body226`] and 32 bytes use [`SbasWireForm::Framed250`].
    pub form: SbasWireForm,
    /// Hex-decoded bytes from the source line, with whitespace removed before
    /// decoding. An odd number of hexadecimal digits is completed with a zero
    /// digit before byte pairs are formed.
    pub bytes: Vec<u8>,
}

/// Parse newline-separated EMS records into [`SbasLogBlock`] values.
///
/// For each line, the first seven non-empty comma-separated fields provide the
/// broadcast PRN and calendar components, and the last non-empty field
/// provides the hexadecimal block. Years numerically below 100 are increased
/// by 2000 before the calendar time is converted to GPST. Lines that fail the
/// field, PRN, calendar-week, or hexadecimal-character checks are ignored; the
/// result preserves input order, while block-length and epoch-construction
/// errors are returned.
pub fn parse_ems_lines(text: &str) -> Result<Vec<SbasLogBlock>> {
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(block) = parse_ems_line(line)? {
            out.push(block);
        }
    }
    Ok(out)
}

/// Parse newline-separated RTKLIB records into [`SbasLogBlock`] values.
///
/// Each recognized record has at least four whitespace-separated header fields
/// before a colon: week, seconds-of-week, broadcast PRN, and an additional
/// header field. The text after the first colon is decoded as the block's
/// hexadecimal bytes. Lines without the required delimiter or parseable,
/// supported fields are ignored, recognized records retain input order, and
/// block-length or epoch-construction errors are returned.
pub fn parse_rtklib_lines(text: &str) -> Result<Vec<SbasLogBlock>> {
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(block) = parse_rtklib_line(line)? {
            out.push(block);
        }
    }
    Ok(out)
}

fn parse_ems_line(line: &str) -> Result<Option<SbasLogBlock>> {
    let parts: Vec<&str> = line
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() < 8 {
        return Ok(None);
    }
    let Some(hex) = parts.last().copied().filter(|s| looks_hex(s)) else {
        return Ok(None);
    };
    let Some(prn) = parse_u16(parts[0]) else {
        return Ok(None);
    };
    let Some(satellite_id) = sbas_prn_to_sat(prn) else {
        return Ok(None);
    };
    let Some(year) = parse_i64(parts[1]) else {
        return Ok(None);
    };
    let Some(month) = parse_i64(parts[2]) else {
        return Ok(None);
    };
    let Some(day) = parse_i64(parts[3]) else {
        return Ok(None);
    };
    let Some(hour) = parse_i64(parts[4]) else {
        return Ok(None);
    };
    let Some(minute) = parse_i64(parts[5]) else {
        return Ok(None);
    };
    let Some(second) = parse_i64(parts[6]) else {
        return Ok(None);
    };
    let year = if year < 100 { 2000 + year } else { year };
    let Some(week) = week_from_calendar(TimeScale::Gpst, year, month, day) else {
        return Ok(None);
    };
    let tow_s = seconds_of_week_from_calendar(year, month, day, hour, minute, second);
    let epoch = GnssWeekTow::new(TimeScale::Gpst, week, tow_s)
        .map_err(|e| Error::Parse(format!("invalid SBAS EMS epoch: {e}")))?;
    let (form, bytes) = decode_hex_block(hex)?;
    Ok(Some(SbasLogBlock {
        satellite_id,
        epoch,
        form,
        bytes,
    }))
}

fn parse_rtklib_line(line: &str) -> Result<Option<SbasLogBlock>> {
    let Some((head, hex)) = line.split_once(':') else {
        return Ok(None);
    };
    if !looks_hex(hex.trim()) {
        return Ok(None);
    }
    let fields: Vec<&str> = head.split_whitespace().collect();
    if fields.len() < 4 {
        return Ok(None);
    }
    let Some(week) = parse_u32(fields[0]) else {
        return Ok(None);
    };
    let Some(tow_s) = parse_f64(fields[1]) else {
        return Ok(None);
    };
    let Some(prn) = parse_u16(fields[2]) else {
        return Ok(None);
    };
    let Some(satellite_id) = sbas_prn_to_sat(prn) else {
        return Ok(None);
    };
    let epoch = GnssWeekTow::new(TimeScale::Gpst, week, tow_s)
        .map_err(|e| Error::Parse(format!("invalid SBAS RTKLIB epoch: {e}")))?;
    let (form, bytes) = decode_hex_block(hex.trim())?;
    Ok(Some(SbasLogBlock {
        satellite_id,
        epoch,
        form,
        bytes,
    }))
}

fn decode_hex_block(hex: &str) -> Result<(SbasWireForm, Vec<u8>)> {
    let mut clean: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
    if !clean.len().is_multiple_of(2) {
        clean.push('0');
    }
    let mut bytes = Vec::with_capacity(clean.len() / 2);
    for idx in (0..clean.len()).step_by(2) {
        let byte = u8::from_str_radix(&clean[idx..idx + 2], 16)
            .map_err(|e| Error::Parse(format!("invalid SBAS hex block: {e}")))?;
        bytes.push(byte);
    }
    let form = match bytes.len() {
        32 => SbasWireForm::Framed250,
        29 => SbasWireForm::Body226,
        _ => return Err(Error::Parse("invalid SBAS hex block length".to_string())),
    };
    Ok((form, bytes))
}

fn looks_hex(value: &str) -> bool {
    let trimmed = value.trim();
    !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c.is_ascii_hexdigit() || c.is_whitespace())
}

fn parse_u16(value: &str) -> Option<u16> {
    value.trim().parse().ok()
}

fn parse_u32(value: &str) -> Option<u32> {
    value.trim().parse().ok()
}

fn parse_i64(value: &str) -> Option<i64> {
    value.trim().parse().ok()
}

fn parse_f64(value: &str) -> Option<f64> {
    value.trim().parse().ok()
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::unwrap_used, clippy::panic)]
mod tests {
    use super::*;
    use crate::astro::time::model::{GnssWeekTow, TimeScale};
    use crate::sbas::message::{SbasBlock, SbasFastCorrections, SbasMessage, SpareBits};

    // The body bytes are the MT2 capture in tests/sbas_real_vectors.rs. The
    // Framed250 counterpart is derived once from those 226 body bits with the
    // public framing convention: CRC-24Q followed by six zero pad bits.
    const RTKLIB_MT2_BODY: [u8; 29] = [
        0x53, 0x08, 0xDF, 0xFC, 0x01, 0x00, 0x05, 0xFF, 0xC0, 0x0D, 0xFF, 0xC0, 0x09, 0xFF, 0xDF,
        0xFC, 0x00, 0x1F, 0xFD, 0xFF, 0xDF, 0xFF, 0xBA, 0xBB, 0xBB, 0xBB, 0x9B, 0xBB, 0x80,
    ];
    const RTKLIB_MT2_FRAMED: [u8; 32] = [
        0x53, 0x08, 0xDF, 0xFC, 0x01, 0x00, 0x05, 0xFF, 0xC0, 0x0D, 0xFF, 0xC0, 0x09, 0xFF, 0xDF,
        0xFC, 0x00, 0x1F, 0xFD, 0xFF, 0xDF, 0xFF, 0xBA, 0xBB, 0xBB, 0xBB, 0x9B, 0xBB, 0x83, 0xA9,
        0xCE, 0x00,
    ];

    fn block_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02X}")).collect()
    }

    #[test]
    fn rtklib_lines_parse_public_body_and_framed_blocks() {
        let expected_message = SbasMessage::FastCorrections(SbasFastCorrections {
            preamble: 0x53,
            message_type: 2,
            iodf: 0,
            iodp: 3,
            prc: [
                2047, 4, 1, 2047, 3, 2047, 2, 2047, 2047, 0, 2047, 2047, 2047,
            ],
            udrei: [14, 14, 10, 14, 14, 14, 14, 14, 14, 6, 14, 14, 14],
            reserved: SpareBits::new(),
        });
        let expected_epoch =
            GnssWeekTow::new(TimeScale::Gpst, 2360, 259_200.0).expect("valid RTKLIB epoch");

        for (form, expected_bytes) in [
            (SbasWireForm::Body226, RTKLIB_MT2_BODY.as_slice()),
            (SbasWireForm::Framed250, RTKLIB_MT2_FRAMED.as_slice()),
        ] {
            let text = format!(
                "bad line\n2360 259200 120 1 : {}\n",
                block_hex(expected_bytes)
            );
            let parsed = parse_rtklib_lines(&text).expect("parse RTKLIB lines");
            assert_eq!(parsed.len(), 1);
            assert_eq!(parsed[0].satellite_id.to_string(), "S20");
            assert_eq!(parsed[0].epoch, expected_epoch);
            assert_eq!(parsed[0].form, form);
            assert_eq!(parsed[0].bytes.len(), expected_bytes.len());
            assert_eq!(parsed[0].bytes, expected_bytes);

            let decoded = SbasBlock::decode(&parsed[0].bytes, parsed[0].form)
                .expect("public SBAS block decoder accepts parsed form");
            assert_eq!(decoded.form, form);
            assert_eq!(decoded.message, expected_message);
        }
    }

    #[test]
    fn ems_lines_parse_calendar_epochs() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        let text = format!("120,26,7,1,0,0,1,1,{hex}\nnot,enough\n");
        let parsed = parse_ems_lines(&text).expect("parse EMS lines");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].satellite_id.to_string(), "S20");
        assert_eq!(parsed[0].form, SbasWireForm::Body226);
    }
}
