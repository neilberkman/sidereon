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
    /// decoding.
    pub bytes: Vec<u8>,
}

/// Parse newline-separated EMS records into [`SbasLogBlock`] values.
///
/// A standard EMS data record separates its fields with blanks, as specified
/// by the EGNOS Message Server User Interface Document
/// (E-RD-SYS-E31-011-ESA Issue 2 Revision 0, section 3): broadcast PRN, year,
/// month, day, hour, minute, second, message type, and the message in
/// hexadecimal. Any run of spaces or tabs separates two fields. A line
/// without a comma whose first non-blank character is `#` is an explicit
/// comment and is skipped before any record recognition runs, so such a line
/// stays ignored however numeric the text after the `#` reads. Every other
/// line without a comma is read in that layout, and two independent signs each
/// recognize it as a record: a first field of decimal digits, in the position
/// a broadcast PRN occupies, or six fields of decimal digits in the calendar
/// positions. Either sign alone is enough, so a record stays a record when the
/// other part of it is damaged or missing. Any other line is treated as a
/// header or blank line and skipped; line length alone never makes a line a
/// record.
///
/// Apart from those `#` comments, this record format reserves lines beginning
/// with a bare decimal number for data records. Prose or a column header that
/// begins with one is therefore read as a malformed record and reported rather
/// than skipped, while the textual headers, `#` comments, and blank lines that
/// EMS listings actually carry are ignored as before. Only a leading `#` marks
/// a comment: a `#` written after the fields of a data record stays part of
/// that record and fails the hexadecimal check rather than being stripped off.
///
/// A recognized record that is truncated, or whose PRN, message type, calendar
/// fields, or hexadecimal message are malformed, returns an error rather than
/// disappearing. Fields beyond the ninth are appended to the hexadecimal
/// message, so trailing text fails the hexadecimal check instead of being
/// dropped.
///
/// A line containing a comma is instead split on commas, preserving the
/// earlier reader's compatibility layout: at least eight non-empty fields,
/// the first seven giving the PRN and calendar components, the last giving
/// the hexadecimal message, and any fields between them ignored. A comma line
/// with fewer than eight non-empty fields is skipped.
///
/// Both layouts accept a four-digit year and raise a year numerically below
/// 100 by 2000 before converting the calendar time to GPST. Issue 2
/// Revision 0 placed the EMS time stamp on GPS time; the UTC stamps of
/// earlier EMS issues are not recognized or converted here. The message-type
/// field is required to be an integer, but its value is not retained:
/// [`SbasLogBlock`] carries no declared message type and classifies the wire
/// form from the decoded byte count alone.
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
/// hexadecimal bytes. Lines without the colon delimiter are ignored as
/// non-record lines; record lines with invalid fields, unsupported PRNs, or
/// malformed hexadecimal blocks return an error.
pub fn parse_rtklib_lines(text: &str) -> Result<Vec<SbasLogBlock>> {
    let mut out = Vec::new();
    for line in text.lines() {
        if let Some(block) = parse_rtklib_line(line)? {
            out.push(block);
        }
    }
    Ok(out)
}

/// Blank-separated field count of a standard EMS data record: PRN, year,
/// month, day, hour, minute, second, message type, and the hexadecimal
/// message.
const EMS_FIELD_COUNT: usize = 9;
/// Index of the year field, the first calendar field, in a standard record.
const EMS_CALENDAR_START: usize = 1;
/// Count of calendar fields: year, month, day, hour, minute, and second.
const EMS_CALENDAR_COUNT: usize = 6;
/// Index of the message-type field in a standard record.
const EMS_MESSAGE_TYPE: usize = 7;

fn parse_ems_line(line: &str) -> Result<Option<SbasLogBlock>> {
    if line.contains(',') {
        parse_ems_comma_line(line)
    } else {
        parse_ems_blank_separated_line(line)
    }
}

/// Read one line in the blank-separated record layout of the EMS User
/// Interface Document.
///
/// An explicit `#` comment is recognized first, by [`is_ems_comment_line`],
/// and skipped before any record evidence is weighed, so the fields written
/// after the `#` never turn a comment into a damaged record. Recognition
/// otherwise asks [`is_ems_record_candidate`] for the two
/// kinds of record evidence, either of which alone keeps the line a record: a
/// decimal PRN field at the front, or a complete decimal calendar block. A
/// candidate that does not carry all nine fields fails as an incomplete record
/// instead of disappearing, and a line carrying neither kind of evidence is a
/// header or blank line whatever its length.
fn parse_ems_blank_separated_line(line: &str) -> Result<Option<SbasLogBlock>> {
    if is_ems_comment_line(line) {
        return Ok(None);
    }
    let fields: Vec<&str> = line.split_whitespace().collect();
    let calendar = ems_calendar_fields(&fields);
    if !is_ems_record_candidate(&fields, calendar) {
        return Ok(None);
    }
    let Some(calendar) = calendar.filter(|_| fields.len() >= EMS_FIELD_COUNT) else {
        return Err(Error::Parse(format!(
            "incomplete SBAS EMS record, expected {EMS_FIELD_COUNT} blank-separated fields: {line}"
        )));
    };
    if !is_decimal_digits(fields[EMS_MESSAGE_TYPE]) {
        return Err(Error::Parse(format!(
            "invalid message type integer in SBAS EMS record: {line}"
        )));
    }
    let hex: String = fields[EMS_FIELD_COUNT - 1..].concat();
    let block = ems_block_from_fields(line, fields[0], calendar, &hex)?;
    Ok(Some(block))
}

/// Read one line in the comma-separated compatibility layout.
///
/// This keeps the earlier reader's rules unchanged, including its minimum of
/// eight non-empty fields and its use of the last field as the hexadecimal
/// message, so nine-field lines carrying a message type and lines with other
/// intermediate fields both continue to parse.
fn parse_ems_comma_line(line: &str) -> Result<Option<SbasLogBlock>> {
    let parts: Vec<&str> = line
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    if parts.len() < 8 {
        return Ok(None);
    }
    let calendar = [parts[1], parts[2], parts[3], parts[4], parts[5], parts[6]];
    let hex = parts[parts.len() - 1];
    let block = ems_block_from_fields(line, parts[0], calendar, hex)?;
    Ok(Some(block))
}

/// Return the fields standing in the six calendar positions, if the line
/// reaches that far.
///
/// The contents are not inspected here: a line of at least seven fields
/// always yields the block it holds in those positions, malformed or not, so
/// that the caller can report the offending field by name.
fn ems_calendar_fields<'a>(fields: &[&'a str]) -> Option<[&'a str; EMS_CALENDAR_COUNT]> {
    fields
        .get(EMS_CALENDAR_START..EMS_CALENDAR_START + EMS_CALENDAR_COUNT)?
        .try_into()
        .ok()
}

/// Whether the line is a candidate blank-separated EMS data record.
///
/// Two independent kinds of evidence each make the line a record, and neither
/// depends on the other being intact:
///
/// - the leading field is decimal digits, as a broadcast PRN is written; or
/// - all six calendar positions are present and decimal digits.
///
/// The first keeps a record with a damaged calendar a record, whether a
/// calendar field is non-numeric or the line stops before the calendar ends,
/// so the parser names the bad field instead of silently dropping the record.
/// The second keeps a record with a damaged PRN a record, since its time stamp
/// still reads as one. Because the leading field alone suffices, this format
/// reserves numeric-leading lines for data records: a prose or header line
/// starting with a bare decimal number is reported as a malformed record. A
/// line satisfying neither - a textual column header however long, prose, or a
/// blank line - is skipped. Explicit `#` comments never reach this test:
/// [`parse_ems_blank_separated_line`] skips them first, so the digits a
/// comment happens to contain are never read as record evidence.
fn is_ems_record_candidate(fields: &[&str], calendar: Option<[&str; EMS_CALENDAR_COUNT]>) -> bool {
    let numeric_prn = fields.first().copied().is_some_and(is_decimal_digits);
    let numeric_calendar =
        calendar.is_some_and(|calendar| calendar.iter().copied().all(is_decimal_digits));
    numeric_prn || numeric_calendar
}

fn ems_block_from_fields(
    line: &str,
    prn_field: &str,
    calendar: [&str; EMS_CALENDAR_COUNT],
    hex: &str,
) -> Result<SbasLogBlock> {
    if !looks_hex(hex) {
        return Err(Error::Parse(format!(
            "invalid hex block in SBAS EMS record: {line}"
        )));
    }
    let prn = parse_u16(prn_field)
        .ok_or_else(|| Error::Parse(format!("invalid PRN integer in SBAS EMS record: {line}")))?;
    let satellite_id = sbas_prn_to_sat(prn)
        .ok_or_else(|| Error::Parse(format!("unsupported SBAS PRN {prn} in EMS record: {line}")))?;
    let year = parse_i64(calendar[0])
        .ok_or_else(|| Error::Parse(format!("invalid year integer in SBAS EMS record: {line}")))?;
    let month = parse_i64(calendar[1])
        .ok_or_else(|| Error::Parse(format!("invalid month integer in SBAS EMS record: {line}")))?;
    let day = parse_i64(calendar[2])
        .ok_or_else(|| Error::Parse(format!("invalid day integer in SBAS EMS record: {line}")))?;
    let hour = parse_i64(calendar[3])
        .ok_or_else(|| Error::Parse(format!("invalid hour integer in SBAS EMS record: {line}")))?;
    let minute = parse_i64(calendar[4]).ok_or_else(|| {
        Error::Parse(format!("invalid minute integer in SBAS EMS record: {line}"))
    })?;
    let second = parse_i64(calendar[5]).ok_or_else(|| {
        Error::Parse(format!("invalid second integer in SBAS EMS record: {line}"))
    })?;
    let year = if year < 100 { 2000 + year } else { year };
    if !(1..=12).contains(&month)
        || !(1..=crate::astro::time::civil::days_in_month(year, month)).contains(&day)
    {
        return Err(Error::Parse(format!(
            "invalid calendar date in SBAS EMS record: {line}"
        )));
    }
    if !(0..=23).contains(&hour) || !(0..=59).contains(&minute) || !(0..=60).contains(&second) {
        return Err(Error::Parse(format!(
            "invalid calendar time in SBAS EMS record: {line}"
        )));
    }
    let week = week_from_calendar(TimeScale::Gpst, year, month, day)
        .ok_or_else(|| Error::Parse(format!("invalid calendar date in SBAS EMS record: {line}")))?;
    let tow_s = seconds_of_week_from_calendar(year, month, day, hour, minute, second)
        .ok_or_else(|| Error::Parse(format!("invalid calendar time in SBAS EMS record: {line}")))?;
    let epoch = GnssWeekTow::new(TimeScale::Gpst, week, tow_s)
        .map_err(|e| Error::Parse(format!("invalid SBAS EMS epoch: {e}")))?;
    let (form, bytes) = decode_hex_block(hex)?;
    Ok(SbasLogBlock {
        satellite_id,
        epoch,
        form,
        bytes,
    })
}

fn parse_rtklib_line(line: &str) -> Result<Option<SbasLogBlock>> {
    let Some((head, hex)) = line.split_once(':') else {
        return Ok(None);
    };
    if !looks_hex(hex.trim()) {
        return Err(Error::Parse(format!(
            "invalid hex block in SBAS RTKLIB record: {line}"
        )));
    }
    let fields: Vec<&str> = head.split_whitespace().collect();
    if fields.len() < 4 {
        return Err(Error::Parse(format!(
            "too few header fields in SBAS RTKLIB record: {line}"
        )));
    }
    let week = parse_u32(fields[0]).ok_or_else(|| {
        Error::Parse(format!(
            "invalid week integer in SBAS RTKLIB record: {line}"
        ))
    })?;
    let tow_s = parse_f64(fields[1])
        .ok_or_else(|| Error::Parse(format!("invalid TOW float in SBAS RTKLIB record: {line}")))?;
    let prn = parse_u16(fields[2]).ok_or_else(|| {
        Error::Parse(format!("invalid PRN integer in SBAS RTKLIB record: {line}"))
    })?;
    let satellite_id = sbas_prn_to_sat(prn).ok_or_else(|| {
        Error::Parse(format!(
            "unsupported SBAS PRN {prn} in RTKLIB record: {line}"
        ))
    })?;
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
    let clean: String = hex.chars().filter(|c| !c.is_whitespace()).collect();
    if !clean.len().is_multiple_of(2) {
        return Err(Error::Parse(
            "odd number of hexadecimal digits in SBAS block".to_string(),
        ));
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

/// Whether the line is an explicit comment of an EMS listing.
///
/// A comment is marked by a `#` as the first non-blank character, so leading
/// blanks before the marker are allowed. The test looks at that first
/// character only: a `#` appearing later on a line belongs to the fields
/// around it and is never treated as the start of a trailing comment, leaving
/// a data record with `#` in its message a malformed record rather than a
/// silently shortened one.
fn is_ems_comment_line(line: &str) -> bool {
    line.trim_start().starts_with('#')
}

/// Whether `value` is a non-empty run of ASCII decimal digits.
///
/// EMS PRN, calendar and message-type fields are written as unsigned decimal
/// numbers, so a sign, a decimal point or any other character disqualifies
/// the field without consulting its numeric value. Recognition and the
/// message-type check both use this test; the values themselves are converted
/// and range-checked afterwards.
fn is_decimal_digits(value: &str) -> bool {
    !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())
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

    // The blank-separated layout of E-RD-SYS-E31-011-ESA Issue 2 Revision 0,
    // section 3: PRN, year, month, day, hour, minute, second, message type,
    // and the message in hexadecimal. The expected week 2425 and seconds of
    // week 259201 were computed from 2026-07-01 00:00:01 against the Sunday
    // 1980-01-06 GPS week origin, not read back from this parser.
    fn expected_unit_epoch() -> GnssWeekTow {
        GnssWeekTow::new(TimeScale::Gpst, 2425, 259_201.0).expect("valid EMS epoch")
    }

    #[test]
    fn ems_lines_parse_blank_separated_records() {
        for (form, expected_bytes) in [
            (SbasWireForm::Body226, RTKLIB_MT2_BODY.as_slice()),
            (SbasWireForm::Framed250, RTKLIB_MT2_FRAMED.as_slice()),
        ] {
            let hex = block_hex(expected_bytes);
            // Single blanks, tabs, and runs of blanks all separate fields; the
            // two-digit and four-digit years name the same calendar date.
            let text = format!(
                "120 26 07 01 00 00 01 2 {hex}\n\
                 120\t2026\t07\t01\t00\t00\t01\t2\t{hex}\n\
                 120   26   07   01   00   00   01   2   {hex}\n"
            );
            let parsed = parse_ems_lines(&text).expect("parse blank-separated EMS lines");
            assert_eq!(parsed.len(), 3);
            for block in &parsed {
                assert_eq!(block.satellite_id.to_string(), "S20");
                assert_eq!(block.epoch, expected_unit_epoch());
                assert_eq!(block.form, form);
                assert_eq!(block.bytes, expected_bytes);
            }
        }
    }

    #[test]
    fn ems_lines_parse_mixed_blank_separated_and_comma_records() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        // A blank-separated record, the nine-field comma variant carrying a
        // message type, and the eight-field comma variant without one.
        let text = format!(
            "120 26 07 01 00 00 01 2 {hex}\n\
             120,26,7,1,0,0,1,2,{hex}\n\
             not,enough\n\
             120,26,7,1,0,0,1,{hex}\n"
        );
        let parsed = parse_ems_lines(&text).expect("parse mixed EMS lines");
        assert_eq!(parsed.len(), 3);
        for block in &parsed {
            assert_eq!(block.epoch, expected_unit_epoch());
            assert_eq!(block.form, SbasWireForm::Body226);
            assert_eq!(block.bytes, RTKLIB_MT2_BODY.as_slice());
        }
    }

    #[test]
    fn ems_lines_skip_header_comment_and_prose_lines() {
        let text = concat!(
            "EGNOS Message Server data records, PRN120, 2026-07-01\n",
            "PRN  YY  MM  DD  hh  mm  ss  MT  EGNOS message in hexadecimal format\n",
            "# blank characters separate the data fields of each data record\n",
            "\n",
            "   \n",
            // Neither kind of record evidence: the leading field is not
            // decimal digits and the calendar positions hold words, so length
            // alone leaves the line a non-record.
            "This sentence is long enough to be mistaken for a data record but ",
            "carries no calendar block in the positions a record uses.\n",
        );
        let parsed = parse_ems_lines(text).expect("header, comment and prose lines are skipped");
        assert!(parsed.is_empty());
    }

    #[test]
    fn ems_lines_skip_numeric_looking_hash_comment_lines() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        // Comments whose text after the `#` fills the calendar positions with
        // digits, or glues a PRN to the marker, stand before, between and
        // after the records they annotate. The second record repeats the
        // first with a four-digit year, naming the same epoch.
        let text = format!(
            "# 26 07 01 00 00 01 2 {hex}\n\
             #120 26 07 01 00 00 01 2 {hex}\n\
             120 26 07 01 00 00 01 2 {hex}\n\
             \t  # 26 07 01 00 00 01 2 {hex}\n\
             #120 26 07 01 00 00 01\n\
             120 2026 07 01 00 00 01 2 {hex}\n\
             # 120\n"
        );
        let parsed = parse_ems_lines(&text).expect("numeric-looking # comments are skipped");
        assert_eq!(parsed.len(), 2);
        for block in &parsed {
            assert_eq!(block.satellite_id.to_string(), "S20");
            assert_eq!(block.epoch, expected_unit_epoch());
            assert_eq!(block.form, SbasWireForm::Body226);
            assert_eq!(block.bytes, RTKLIB_MT2_BODY.as_slice());
        }

        let comments_only = format!(
            "# 26 07 01 00 00 01 2 {hex}\n\
             #120 26 07 01 00 00 01 2 {hex}\n\
             #\n"
        );
        let parsed =
            parse_ems_lines(&comments_only).expect("a comment-only listing holds no records");
        assert!(parsed.is_empty());
    }

    #[test]
    fn ems_lines_reject_malformed_blank_separated_records() {
        let hex = block_hex(&RTKLIB_MT2_BODY);

        // A well-formed calendar block keeps the line a record, so a PRN that
        // is not an integer is reported instead of being reclassified as a
        // header line and dropped.
        let bad_prn_token = format!("12O 26 07 01 00 00 01 2 {hex}\n");
        let err = parse_ems_lines(&bad_prn_token).unwrap_err();
        assert!(
            matches!(err, Error::Parse(ref msg) if msg.contains("invalid PRN integer")),
            "expected an invalid PRN error, got {err:?}"
        );

        let unsupported_prn = format!("999 26 07 01 00 00 01 2 {hex}\n");
        assert!(parse_ems_lines(&unsupported_prn).is_err());

        let bad_month = format!("120 26 13 01 00 00 01 2 {hex}\n");
        assert!(parse_ems_lines(&bad_month).is_err());

        let bad_day = format!("120 26 02 30 00 00 01 2 {hex}\n");
        assert!(parse_ems_lines(&bad_day).is_err());

        let bad_hour = format!("120 26 07 01 24 00 01 2 {hex}\n");
        assert!(parse_ems_lines(&bad_hour).is_err());

        let bad_minute = format!("120 26 07 01 00 60 01 2 {hex}\n");
        assert!(parse_ems_lines(&bad_minute).is_err());

        let bad_message_type = format!("120 26 07 01 00 00 01 MT {hex}\n");
        let err = parse_ems_lines(&bad_message_type).unwrap_err();
        assert!(
            matches!(err, Error::Parse(ref msg) if msg.contains("invalid message type integer")),
            "expected an invalid message type error, got {err:?}"
        );

        let bad_hex = "120 26 07 01 00 00 01 2 NOT_HEX_BLOCK\n";
        assert!(parse_ems_lines(bad_hex).is_err());

        let short_block = format!("120 26 07 01 00 00 01 2 {}\n", &hex[..hex.len() - 2]);
        assert!(parse_ems_lines(&short_block).is_err());

        // Trailing text joins the message rather than being ignored.
        let trailing_text = format!("120 26 07 01 00 00 01 2 {hex} extra\n");
        assert!(parse_ems_lines(&trailing_text).is_err());

        let bad_second = format!("120 26 07 01 00 00 61 2 {hex}\n");
        assert!(parse_ems_lines(&bad_second).is_err());

        // A numeric PRN keeps every truncated prefix a record, so each one
        // fails by name instead of passing for a header line.
        for truncated in [
            "120\n",
            "120 26\n",
            "120 26 07\n",
            "120 26 07 01 00 00\n",
            "120 26 07 01 00 00 01\n",
            "120 26 07 01 00 00 01 2\n",
        ] {
            let err = parse_ems_lines(truncated).unwrap_err();
            assert!(
                matches!(err, Error::Parse(ref msg) if msg.contains("incomplete SBAS EMS record")),
                "expected an incomplete record error for {truncated:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn ems_lines_report_each_malformed_calendar_field_of_a_numeric_prn_record() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        // A numeric PRN is record evidence on its own, so a calendar field
        // that is not an integer is named rather than reclassifying the line.
        let names = ["year", "month", "day", "hour", "minute", "second"];
        for (index, field) in names.into_iter().enumerate() {
            let mut calendar = ["26", "07", "01", "00", "00", "01"];
            calendar[index] = "xx";
            let line = format!("120 {} 2 {hex}\n", calendar.join(" "));
            let err = parse_ems_lines(&line).unwrap_err();
            let expected = format!("invalid {field} integer");
            assert!(
                matches!(err, Error::Parse(ref msg) if msg.contains(&expected)),
                "expected {expected:?} for {line:?}, got {err:?}"
            );
        }
    }

    #[test]
    fn ems_lines_report_a_malformed_record_following_a_good_one() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        let good = format!("120 26 07 01 00 00 01 2 {hex}");
        // A malformed record must not disappear behind the record before it,
        // leaving a silent one-record result.
        for bad in [
            "120 26 07".to_string(),
            format!("120 26 xx 01 00 00 01 2 {hex}"),
            format!("12O 26 07 01 00 00 01 2 {hex}"),
        ] {
            let text = format!("{good}\n{bad}\n");
            assert!(
                parse_ems_lines(&text).is_err(),
                "expected an error rather than a one-record result for {bad:?}"
            );
        }
    }

    #[test]
    fn decode_hex_block_rejects_odd_hex_length() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        let truncated = &hex[..hex.len() - 1];
        let err = decode_hex_block(truncated).unwrap_err();
        assert!(
            matches!(err, Error::Parse(ref msg) if msg.contains("odd number of hexadecimal digits")),
            "expected Error::Parse with odd hex digits message, got {err:?}"
        );
    }

    #[test]
    fn parse_ems_lines_rejects_corrupted_record_fields() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        let text_bad_prn = format!("999,26,7,1,0,0,1,1,{hex}\n");
        assert!(parse_ems_lines(&text_bad_prn).is_err());

        let text_bad_hex = "120,26,7,1,0,0,1,1,NOT_HEX_BLOCK\n";
        assert!(parse_ems_lines(text_bad_hex).is_err());

        let text_bad_date = format!("120,26,13,1,0,0,1,1,{hex}\n");
        assert!(parse_ems_lines(&text_bad_date).is_err());

        let text_bad_day = format!("120,26,7,32,0,0,1,1,{hex}\n");
        assert!(parse_ems_lines(&text_bad_day).is_err());

        let text_bad_hour = format!("120,26,7,1,24,0,1,1,{hex}\n");
        assert!(parse_ems_lines(&text_bad_hour).is_err());

        let text_bad_minute = format!("120,26,7,1,0,60,1,1,{hex}\n");
        assert!(parse_ems_lines(&text_bad_minute).is_err());

        let text_bad_second = format!("120,26,7,1,0,0,61,1,{hex}\n");
        assert!(parse_ems_lines(&text_bad_second).is_err());
    }

    #[test]
    fn parse_rtklib_lines_rejects_corrupted_record_fields() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        let text_bad_prn = format!("2360 259200 999 1 : {hex}\n");
        assert!(parse_rtklib_lines(&text_bad_prn).is_err());

        let text_bad_week = format!("not_a_week 259200 120 1 : {hex}\n");
        assert!(parse_rtklib_lines(&text_bad_week).is_err());

        let text_bad_hex = "2360 259200 120 1 : NOT_HEX_BLOCK\n";
        assert!(parse_rtklib_lines(text_bad_hex).is_err());
    }
}
