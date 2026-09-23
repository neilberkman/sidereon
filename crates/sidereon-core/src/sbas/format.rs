use crate::astro::time::gnss::{seconds_of_week_from_calendar, week_from_calendar};
use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::error::{Error, Result};
use crate::id::GnssSatelliteId;

use super::message::{SbasDeparture, SbasMessageType, SbasPolicy, SbasWireForm};
use super::store::sbas_prn_to_sat;

#[derive(Clone, Debug, PartialEq)]
/// One SBAS block recovered from an EMS or RTKLIB log line.
///
/// The line parsers retain recognized records in input order and store the
/// converted satellite, GPST epoch, wire-form classification, the message
/// type the record states, and decoded bytes.
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
    /// The message type the record's own field states: the EMS message-type
    /// field, or the fourth header field RTKLIB `sbsoutmsg` writes. `None` for
    /// an eight-field comma line, which carries no such field.
    pub declared_message_type: Option<SbasMessageType>,
}

impl SbasLogBlock {
    /// The six-bit message type carried at message bits 8 through 13
    /// (zero-based) of [`SbasLogBlock::bytes`], or `None` when the bytes are
    /// shorter than two bytes.
    pub fn message_type(&self) -> Option<SbasMessageType> {
        self.bytes.get(1).map(|byte| byte >> 2)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Why a log line was read as no record.
pub enum SbasSkippedLineKind {
    /// A line holding only blanks.
    Blank,
    /// A line whose first non-blank character is `#` or, in an RTKLIB log,
    /// `%`.
    Comment,
    /// A line carrying no sign of a record, such as a textual column header or
    /// prose.
    NonRecord,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A log line read as no record.
pub struct SbasSkippedLine {
    /// One-based line number.
    pub line: usize,
    /// Why the line holds no record.
    pub kind: SbasSkippedLineKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
/// A departure read from one log line under [`SbasPolicy::Lenient`].
pub struct SbasLineDeparture {
    /// One-based line number.
    pub line: usize,
    /// The departure.
    pub departure: SbasDeparture,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
/// Why a record line was left unread while the rest of the log was read.
pub enum SbasLineRefusal {
    /// A NovAtel OEM3 `$FRMA` week below 1024 is a 10-bit week, which names
    /// one week in every 1024 and [`SbasLogOptions::reference_week`] was not
    /// given to choose among them.
    AmbiguousWeek {
        /// The week as written.
        week: u32,
    },
    /// The checksum after a NovAtel line's `*` is not the one its text gives:
    /// the NovAtel CRC-32 for an OEM4 line, the XOR of the text for an OEM3
    /// line. `written` is `None` when the checksum text is not hexadecimal
    /// of the checksum's width.
    ChecksumMismatch {
        /// The checksum as written, when it reads as one.
        written: Option<u32>,
        /// The checksum the line's text gives.
        computed: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// A record line left unread, with the reason.
pub struct SbasRefusedLine {
    /// One-based line number.
    pub line: usize,
    /// Why the line was not read.
    pub reason: SbasLineRefusal,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
/// Options for [`parse_ems_log`] and [`parse_rtklib_log`].
pub struct SbasLogOptions {
    /// How departures are treated; [`SbasPolicy::Strict`] by default.
    pub policy: SbasPolicy,
    /// A full GPS week near the log's time, used to resolve the 10-bit week
    /// of a NovAtel OEM3 `$FRMA` line to the full week closest to it; a week
    /// exactly 512 away resolves to the later one. Without it such a line is
    /// listed in [`SbasLog::refused_lines`].
    pub reference_week: Option<u32>,
}

impl SbasLogOptions {
    /// Return the options with `policy`.
    pub fn with_policy(mut self, policy: SbasPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Return the options with `reference_week`.
    pub fn with_reference_week(mut self, reference_week: u32) -> Self {
        self.reference_week = Some(reference_week);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
/// Everything an EMS or RTKLIB log reader read: the records, every line read
/// as no record, every record line left unread, and, under
/// [`SbasPolicy::Lenient`], every departure.
///
/// Each input line appears exactly once, as a block, a skipped line or a
/// refused line.
pub struct SbasLog {
    /// Records in input order.
    pub blocks: Vec<SbasLogBlock>,
    /// Lines read as no record, in input order.
    pub skipped_lines: Vec<SbasSkippedLine>,
    /// Record lines left unread, in input order, with the reason.
    pub refused_lines: Vec<SbasRefusedLine>,
    /// Departures read under [`SbasPolicy::Lenient`], in input order. Always
    /// empty under [`SbasPolicy::Strict`], which refuses the first one.
    pub departures: Vec<SbasLineDeparture>,
}

/// Parse newline-separated EMS records into [`SbasLogBlock`] values under
/// [`SbasPolicy::Strict`].
///
/// This is [`parse_ems_log`] with default options, keeping only the blocks;
/// the lines it reads as no record are listed by [`parse_ems_log`]. A record
/// line [`parse_ems_log`] would leave unread returns an error, since the
/// blocks alone cannot show it.
pub fn parse_ems_lines(text: &str) -> Result<Vec<SbasLogBlock>> {
    blocks_only(parse_ems_log(text, SbasLogOptions::default())?)
}

/// Parse newline-separated EMS records under `options`.
///
/// A standard EMS data record separates its fields with blanks, as specified
/// by the EGNOS Message Server User Interface Document
/// (E-RD-SYS-E31-011-ESA Issue 2 Revision 0, section 3): broadcast PRN, year,
/// month, day, hour, minute, second, message type, and the message in
/// hexadecimal. Any run of spaces or tabs separates two fields. A line
/// whose first non-blank character is `#` is an explicit comment and is
/// skipped before any record recognition runs, so such a line stays ignored
/// however numeric the text after the `#` reads. Every other line without a
/// comma is read in that layout, and two independent signs each
/// recognize it as a record: a first field of decimal digits, in the position
/// a broadcast PRN occupies, or six fields of decimal digits in the calendar
/// positions. Either sign alone is enough, so a record stays a record when the
/// other part of it is damaged or missing. Any other line is treated as a
/// header or blank line and listed in [`SbasLog::skipped_lines`]; line length
/// alone never makes a line a record.
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
/// A line containing a comma is instead split on commas, the earlier reader's
/// compatibility layout: eight fields, the PRN, the six calendar fields and
/// the hexadecimal message, or nine fields with the message type before the
/// message. The same two signs recognize a comma line as a record. A record
/// with another field count, or with an empty field before its last non-empty
/// one, is refused, since the positions of its fields are then unknown; empty
/// fields after the message, as a trailing comma leaves, carry nothing and are
/// accepted.
///
/// Both layouts accept a four-digit year. A two-digit year is read as RTKLIB
/// `readmsgs` reads it: below 70 in the 2000s, otherwise in the 1900s. Issue 2
/// Revision 0 placed the EMS time stamp on GPS time; the UTC stamps of earlier
/// EMS issues are not recognized or converted here. The message type must be
/// an integer from 0 through 63 and is kept in
/// [`SbasLogBlock::declared_message_type`]. The EMS User Interface Document
/// derives it from message bits 9 to 14, so a value that differs from the
/// type the message carries is a departure: refused under
/// [`SbasPolicy::Strict`], read and reported under [`SbasPolicy::Lenient`].
///
/// NovAtel OEM4 `#RAWWAASFRAMEA` and OEM3 `$FRMA` lines, which RTKLIB
/// `readmsgs` reads in any SBAS log, are read as [`parse_rtklib_log`]
/// describes.
pub fn parse_ems_log(text: &str, options: SbasLogOptions) -> Result<SbasLog> {
    read_log(text, options, parse_ems_line)
}

/// Parse newline-separated RTKLIB records into [`SbasLogBlock`] values under
/// [`SbasPolicy::Strict`].
///
/// This is [`parse_rtklib_log`] with default options, keeping only the
/// blocks; the lines it reads as no record are listed by
/// [`parse_rtklib_log`]. A record line [`parse_rtklib_log`] would leave unread
/// returns an error, since the blocks alone cannot show it.
pub fn parse_rtklib_lines(text: &str) -> Result<Vec<SbasLogBlock>> {
    blocks_only(parse_rtklib_log(text, SbasLogOptions::default())?)
}

fn blocks_only(log: SbasLog) -> Result<Vec<SbasLogBlock>> {
    match log.refused_lines.first() {
        Some(refused) => Err(Error::Parse(format!(
            "SBAS log line {} left unread: {:?}",
            refused.line, refused.reason
        ))),
        None => Ok(log.blocks),
    }
}

/// Parse newline-separated RTKLIB records under `options`.
///
/// A record is the line RTKLIB `sbsoutmsg` writes: four whitespace-separated
/// header fields - week, seconds-of-week, broadcast PRN and message type -
/// then a colon and the block in hexadecimal. The message type must be an
/// integer from 0 through 63 and is kept in
/// [`SbasLogBlock::declared_message_type`]; `sbsoutmsg` writes the type the
/// message carries, so a value that differs from it is a departure: refused
/// under [`SbasPolicy::Strict`], read and reported under
/// [`SbasPolicy::Lenient`]. A record with another header field count, invalid
/// fields, an unsupported PRN or a malformed hexadecimal block returns an
/// error.
///
/// Blank lines, lines whose first non-blank character is `#` or `%`, and
/// lines without a colon whose first field is not decimal digits are listed in
/// [`SbasLog::skipped_lines`]. A line without a colon whose first field is
/// decimal digits, as a week is written, is a record missing its separator
/// and is refused.
///
/// NovAtel lines are read in both this reader and [`parse_ems_log`], as RTKLIB
/// `readmsgs` reads them in any SBAS log:
///
/// - OEM4 `#RAWWAASFRAMEA`: the GPS week and seconds are the sixth and
///   seventh header fields; after the `;` come the channel, the PRN, the
///   message ID, kept as [`SbasLogBlock::declared_message_type`], and the
///   frame in hexadecimal, exactly four fields. A `*` checksum is the NovAtel
///   ASCII CRC-32 of the text between `#` and `*`, written as eight
///   hexadecimal digits.
/// - OEM3 `$FRMA`: the week, seconds and PRN are the second to fourth fields
///   and the frame is the seventh and last. A `*` checksum is the XOR of the
///   text between `$` and `*`, written as two hexadecimal digits.
/// - A checksum that differs from the one the text gives, or that is not
///   written as one, leaves the line unread: it is listed in
///   [`SbasLog::refused_lines`] as [`SbasLineRefusal::ChecksumMismatch`] and
///   the rest of the log is read.
/// - An OEM3 week below 1024 is a 10-bit week, resolved with
///   [`SbasLogOptions::reference_week`]; RTKLIB adds 1024 to it, which names
///   the right week only between 1999 and 2019. Without a reference week the
///   line is listed in [`SbasLog::refused_lines`] and the rest of the log is
///   read.
///
/// The channel of an OEM4 line and the fifth and sixth fields of an OEM3
/// line, which RTKLIB `readmsgs` does not read either, are not kept.
pub fn parse_rtklib_log(text: &str, options: SbasLogOptions) -> Result<SbasLog> {
    read_log(text, options, parse_rtklib_line)
}

/// What one log line holds.
enum LogLine {
    Block(SbasLogBlock),
    Skipped(SbasSkippedLineKind),
    Refused(SbasLineRefusal),
}

fn read_log(
    text: &str,
    options: SbasLogOptions,
    parse_line: fn(&str) -> Result<LogLine>,
) -> Result<SbasLog> {
    let policy = options.policy;
    let mut log = SbasLog::default();
    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        let outcome = match parse_novatel_line(line, options.reference_week)? {
            Some(outcome) => outcome,
            None => parse_line(line)?,
        };
        match outcome {
            LogLine::Block(block) => {
                if let Some(departure) = declared_type_departure(&block) {
                    match policy {
                        SbasPolicy::Strict => {
                            return Err(Error::Parse(format!(
                                "SBAS log line {line_number}: {departure}: {line}"
                            )));
                        }
                        SbasPolicy::Lenient => log.departures.push(SbasLineDeparture {
                            line: line_number,
                            departure,
                        }),
                    }
                }
                log.blocks.push(block);
            }
            LogLine::Skipped(kind) => log.skipped_lines.push(SbasSkippedLine {
                line: line_number,
                kind,
            }),
            LogLine::Refused(reason) => log.refused_lines.push(SbasRefusedLine {
                line: line_number,
                reason,
            }),
        }
    }
    Ok(log)
}

fn declared_type_departure(block: &SbasLogBlock) -> Option<SbasDeparture> {
    let declared = block.declared_message_type?;
    let carried = block.message_type()?;
    (declared != carried).then_some(SbasDeparture::DeclaredMessageType { declared, carried })
}

const NOVATEL_OEM4_SBAS: &str = "#RAWWAASFRAMEA";
const NOVATEL_OEM3_SBAS: &str = "$FRMA";
/// The weeks a 10-bit GPS week field repeats over.
const WEEK_ROLLOVER: i64 = 1024;

/// Read a NovAtel SBAS frame line, or return `None` for any other line.
fn parse_novatel_line(line: &str, reference_week: Option<u32>) -> Result<Option<LogLine>> {
    let trimmed = line.trim();
    if trimmed.starts_with(NOVATEL_OEM4_SBAS) {
        parse_novatel_oem4_line(line, trimmed).map(Some)
    } else if trimmed.starts_with(NOVATEL_OEM3_SBAS) {
        parse_novatel_oem3_line(line, trimmed, reference_week).map(Some)
    } else {
        Ok(None)
    }
}

/// Split a NovAtel line at its checksum delimiter, returning the text after
/// the one-character sync and the checksum text, if any.
fn novatel_content(trimmed: &str) -> (&str, Option<&str>) {
    let after_sync = &trimmed[1..];
    match after_sync.rsplit_once('*') {
        Some((content, checksum)) => (content, Some(checksum)),
        None => (after_sync, None),
    }
}

fn parse_novatel_oem4_line(line: &str, trimmed: &str) -> Result<LogLine> {
    let (content, checksum) = novatel_content(trimmed);
    if let Some(checksum) = checksum {
        let computed = novatel_crc32(content.as_bytes());
        let written = read_checksum(checksum, 8);
        if written != Some(computed) {
            return Ok(LogLine::Refused(SbasLineRefusal::ChecksumMismatch {
                written,
                computed,
            }));
        }
    }
    let Some((header, data)) = content.split_once(';') else {
        return Err(Error::Parse(format!(
            "RAWWAASFRAMEA record without the ';' after its header: {line}"
        )));
    };
    let header: Vec<&str> = header.split(',').map(str::trim).collect();
    let (Some(week), Some(tow)) = (header.get(5), header.get(6)) else {
        return Err(Error::Parse(format!(
            "RAWWAASFRAMEA header without its GPS week and seconds: {line}"
        )));
    };
    let week = parse_u32(week).ok_or_else(|| {
        Error::Parse(format!(
            "invalid week integer in RAWWAASFRAMEA record: {line}"
        ))
    })?;
    let tow_s = parse_f64(tow)
        .ok_or_else(|| Error::Parse(format!("invalid seconds in RAWWAASFRAMEA record: {line}")))?;
    let data: Vec<&str> = data.split(',').map(str::trim).collect();
    let [channel, prn, message_id, frame] = data.as_slice() else {
        return Err(Error::Parse(format!(
            "RAWWAASFRAMEA record has {} data fields, expected channel, PRN, message ID and frame: {line}",
            data.len()
        )));
    };
    if parse_u32(channel).is_none() {
        return Err(Error::Parse(format!(
            "invalid channel integer in RAWWAASFRAMEA record: {line}"
        )));
    }
    let declared_message_type = parse_message_type(message_id, "RAWWAASFRAMEA", line)?;
    novatel_block(
        line,
        "RAWWAASFRAMEA",
        week,
        tow_s,
        prn,
        frame,
        Some(declared_message_type),
    )
    .map(LogLine::Block)
}

fn parse_novatel_oem3_line(
    line: &str,
    trimmed: &str,
    reference_week: Option<u32>,
) -> Result<LogLine> {
    let (content, checksum) = novatel_content(trimmed);
    if let Some(checksum) = checksum {
        let computed = u32::from(content.bytes().fold(0u8, |sum, byte| sum ^ byte));
        let written = read_checksum(checksum, 2);
        if written != Some(computed) {
            return Ok(LogLine::Refused(SbasLineRefusal::ChecksumMismatch {
                written,
                computed,
            }));
        }
    }
    let fields: Vec<&str> = content.split(',').map(str::trim).collect();
    let [_, week, tow, prn, _, _, frame] = fields.as_slice() else {
        return Err(Error::Parse(format!(
            "FRMA record has {} fields, expected 7: {line}",
            fields.len()
        )));
    };
    let week = parse_u32(week)
        .ok_or_else(|| Error::Parse(format!("invalid week integer in FRMA record: {line}")))?;
    let tow_s = parse_f64(tow)
        .ok_or_else(|| Error::Parse(format!("invalid seconds in FRMA record: {line}")))?;
    let week = if i64::from(week) < WEEK_ROLLOVER {
        match reference_week {
            Some(reference_week) => resolve_ten_bit_week(week, reference_week),
            None => return Ok(LogLine::Refused(SbasLineRefusal::AmbiguousWeek { week })),
        }
    } else {
        week
    };
    novatel_block(line, "FRMA", week, tow_s, prn, frame, None).map(LogLine::Block)
}

/// Read a checksum written as exactly `digits` hexadecimal digits.
fn read_checksum(text: &str, digits: usize) -> Option<u32> {
    let text = text.trim();
    (text.len() == digits && text.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .then(|| u32::from_str_radix(text, 16).ok())
        .flatten()
}

/// The full week that `week` (below 1024) names closest to `reference_week`;
/// a week exactly 512 away resolves to the later one. When that candidate
/// falls outside 0 through `u32::MAX`, the other candidate 1024 weeks the
/// other way is taken, so the result is always a week the field names.
fn resolve_ten_bit_week(week: u32, reference_week: u32) -> u32 {
    let week = i64::from(week);
    let reference_week = i64::from(reference_week);
    let rollovers = (reference_week - week + WEEK_ROLLOVER / 2).div_euclid(WEEK_ROLLOVER);
    let full = week + rollovers * WEEK_ROLLOVER;
    let full = if full < 0 {
        full + WEEK_ROLLOVER
    } else if full > i64::from(u32::MAX) {
        full - WEEK_ROLLOVER
    } else {
        full
    };
    // Both adjustments land within 0..=u32::MAX: `week` is below 1024 and
    // u32::MAX + 1 is a multiple of 1024.
    u32::try_from(full).unwrap_or(0)
}

fn novatel_block(
    line: &str,
    format: &str,
    week: u32,
    tow_s: f64,
    prn: &str,
    frame: &str,
    declared_message_type: Option<SbasMessageType>,
) -> Result<SbasLogBlock> {
    let prn = parse_u16(prn)
        .ok_or_else(|| Error::Parse(format!("invalid PRN integer in {format} record: {line}")))?;
    let satellite_id = sbas_prn_to_sat(prn).ok_or_else(|| {
        Error::Parse(format!(
            "unsupported SBAS PRN {prn} in {format} record: {line}"
        ))
    })?;
    if !looks_hex(frame) {
        return Err(Error::Parse(format!(
            "invalid hex block in {format} record: {line}"
        )));
    }
    let epoch = GnssWeekTow::new(TimeScale::Gpst, week, tow_s)
        .map_err(|e| Error::Parse(format!("invalid SBAS {format} epoch: {e}")))?;
    let (form, bytes) = decode_hex_block(frame)?;
    Ok(SbasLogBlock {
        satellite_id,
        epoch,
        form,
        bytes,
        declared_message_type,
    })
}

/// The NovAtel ASCII CRC-32: reflected polynomial `0xEDB88320`, initial value
/// 0 and no final XOR, as NovAtel's `CalculateBlockCRC32` forms it.
fn novatel_crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0u32;
    for &byte in bytes {
        let mut value = (crc ^ u32::from(byte)) & 0xFF;
        for _ in 0..8 {
            value = if value & 1 != 0 {
                (value >> 1) ^ 0xEDB8_8320
            } else {
                value >> 1
            };
        }
        crc = ((crc >> 8) & 0x00FF_FFFF) ^ value;
    }
    crc
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
/// Field count of an eight-field comma line, which has no message type.
const EMS_COMMA_FIELDS_WITHOUT_TYPE: usize = 8;
/// The largest six-bit SBAS message type.
const MAX_MESSAGE_TYPE: u8 = 63;

fn parse_ems_line(line: &str) -> Result<LogLine> {
    if line.trim().is_empty() {
        return Ok(LogLine::Skipped(SbasSkippedLineKind::Blank));
    }
    if is_ems_comment_line(line) {
        return Ok(LogLine::Skipped(SbasSkippedLineKind::Comment));
    }
    if line.contains(',') {
        parse_ems_comma_line(line)
    } else {
        parse_ems_blank_separated_line(line)
    }
}

/// Read one line in the blank-separated record layout of the EMS User
/// Interface Document.
///
/// Blank lines and explicit `#` comments, recognized by
/// [`is_ems_comment_line`], are skipped by [`parse_ems_line`] before any
/// record evidence is weighed, so the fields written after the `#` never turn
/// a comment into a damaged record. Recognition asks
/// [`is_ems_record_candidate`] for the two kinds of record evidence, either of
/// which alone keeps the line a record: a decimal PRN field at the front, or a
/// complete decimal calendar block. A
/// candidate that does not carry all nine fields fails as an incomplete record
/// instead of disappearing, and a line carrying neither kind of evidence is a
/// header or blank line whatever its length.
fn parse_ems_blank_separated_line(line: &str) -> Result<LogLine> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let calendar = ems_calendar_fields(&fields);
    if !is_ems_record_candidate(&fields, calendar) {
        return Ok(LogLine::Skipped(SbasSkippedLineKind::NonRecord));
    }
    let Some(calendar) = calendar.filter(|_| fields.len() >= EMS_FIELD_COUNT) else {
        return Err(Error::Parse(format!(
            "incomplete SBAS EMS record, expected {EMS_FIELD_COUNT} blank-separated fields: {line}"
        )));
    };
    let message_type = parse_message_type(fields[EMS_MESSAGE_TYPE], "EMS", line)?;
    let hex: String = fields[EMS_FIELD_COUNT - 1..].concat();
    let block = ems_block_from_fields(line, fields[0], calendar, &hex, Some(message_type))?;
    Ok(LogLine::Block(block))
}

/// Read one line in the comma-separated compatibility layout.
///
/// The same two signs as the blank-separated layout recognize a record. A
/// record is eight fields, PRN, calendar and message, or nine, with the
/// message type before the message; empty fields after the last non-empty
/// one are dropped first, and any other empty field refuses the record.
fn parse_ems_comma_line(line: &str) -> Result<LogLine> {
    let mut parts: Vec<&str> = line.split(',').map(str::trim).collect();
    while parts.last().is_some_and(|part| part.is_empty()) {
        parts.pop();
    }
    let calendar = ems_calendar_fields(&parts);
    if !is_ems_record_candidate(&parts, calendar) {
        return Ok(LogLine::Skipped(SbasSkippedLineKind::NonRecord));
    }
    if parts.iter().any(|part| part.is_empty()) {
        return Err(Error::Parse(format!(
            "empty field in SBAS EMS comma record: {line}"
        )));
    }
    let (message_type, hex) = match parts.len() {
        EMS_COMMA_FIELDS_WITHOUT_TYPE => (None, parts[7]),
        EMS_FIELD_COUNT => (
            Some(parse_message_type(parts[EMS_MESSAGE_TYPE], "EMS", line)?),
            parts[8],
        ),
        count => {
            return Err(Error::Parse(format!(
                "SBAS EMS comma record has {count} fields, expected 8 or 9: {line}"
            )));
        }
    };
    let Some(calendar) = calendar else {
        return Err(Error::Parse(format!(
            "incomplete SBAS EMS comma record: {line}"
        )));
    };
    let block = ems_block_from_fields(line, parts[0], calendar, hex, message_type)?;
    Ok(LogLine::Block(block))
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

/// Whether the line is a candidate EMS data record.
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
/// [`parse_ems_line`] skips them first, so the digits a
/// comment happens to contain are never read as record evidence.
fn is_ems_record_candidate(fields: &[&str], calendar: Option<[&str; EMS_CALENDAR_COUNT]>) -> bool {
    let numeric_prn = fields.first().copied().is_some_and(is_decimal_digits);
    let numeric_calendar =
        calendar.is_some_and(|calendar| calendar.iter().copied().all(is_decimal_digits));
    numeric_prn || numeric_calendar
}

/// Read a record's message-type field: decimal digits naming a six-bit type.
fn parse_message_type(field: &str, format: &str, line: &str) -> Result<SbasMessageType> {
    field
        .parse::<u8>()
        .ok()
        .filter(|value| is_decimal_digits(field) && *value <= MAX_MESSAGE_TYPE)
        .ok_or_else(|| {
            Error::Parse(format!(
                "invalid message type integer in SBAS {format} record, expected 0 through 63: {line}"
            ))
        })
}

fn ems_block_from_fields(
    line: &str,
    prn_field: &str,
    calendar: [&str; EMS_CALENDAR_COUNT],
    hex: &str,
    declared_message_type: Option<SbasMessageType>,
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
    let year = full_year(year);
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
        declared_message_type,
    })
}

/// The full year of an EMS year field, as RTKLIB `readmsgs` forms it for a
/// two-digit year: below 70 in the 2000s, 70 through 99 in the 1900s. A year
/// of 100 or more is taken as written.
fn full_year(year: i64) -> i64 {
    match year {
        0..=69 => 2000 + year,
        70..=99 => 1900 + year,
        _ => year,
    }
}

fn parse_rtklib_line(line: &str) -> Result<LogLine> {
    let trimmed = line.trim_start();
    if trimmed.is_empty() {
        return Ok(LogLine::Skipped(SbasSkippedLineKind::Blank));
    }
    if trimmed.starts_with('#') || trimmed.starts_with('%') {
        return Ok(LogLine::Skipped(SbasSkippedLineKind::Comment));
    }
    let Some((head, hex)) = line.split_once(':') else {
        if line
            .split_whitespace()
            .next()
            .is_some_and(is_decimal_digits)
        {
            return Err(Error::Parse(format!(
                "SBAS RTKLIB record without the ':' before its block: {line}"
            )));
        }
        return Ok(LogLine::Skipped(SbasSkippedLineKind::NonRecord));
    };
    if !looks_hex(hex.trim()) {
        return Err(Error::Parse(format!(
            "invalid hex block in SBAS RTKLIB record: {line}"
        )));
    }
    let fields: Vec<&str> = head.split_whitespace().collect();
    if fields.len() != 4 {
        return Err(Error::Parse(format!(
            "SBAS RTKLIB record has {} header fields, expected week, time of week, PRN and message type: {line}",
            fields.len()
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
    let declared_message_type = parse_message_type(fields[3], "RTKLIB", line)?;
    let epoch = GnssWeekTow::new(TimeScale::Gpst, week, tow_s)
        .map_err(|e| Error::Parse(format!("invalid SBAS RTKLIB epoch: {e}")))?;
    let (form, bytes) = decode_hex_block(hex.trim())?;
    Ok(LogLine::Block(SbasLogBlock {
        satellite_id,
        epoch,
        form,
        bytes,
        declared_message_type: Some(declared_message_type),
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
                "bad line\n2360 259200 120  2 : {}\n",
                block_hex(expected_bytes)
            );
            let parsed = parse_rtklib_lines(&text).expect("parse RTKLIB lines");
            assert_eq!(parsed.len(), 1);
            assert_eq!(parsed[0].declared_message_type, Some(2));
            assert_eq!(parsed[0].message_type(), Some(2));
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
        // The ninth comma field before the message is its message type, 2.
        let text = format!("120,26,7,1,0,0,1,2,{hex}\nnot,enough\n");
        let parsed = parse_ems_lines(&text).expect("parse EMS lines");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].declared_message_type, Some(2));
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

    #[test]
    fn declared_message_type_mismatch_is_refused_strictly_and_reported_leniently() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        for (text, reader) in [
            (
                format!("120 26 07 01 00 00 01 2 {hex}\n120 26 07 01 00 00 01 4 {hex}\n"),
                "ems",
            ),
            (
                format!("120,26,7,1,0,0,1,2,{hex}\n120,26,7,1,0,0,1,4,{hex}\n"),
                "ems",
            ),
            (
                format!("2360 259200 120  2 : {hex}\n2360 259200 120  4 : {hex}\n"),
                "rtklib",
            ),
        ] {
            let read = |policy| {
                if reader == "ems" {
                    parse_ems_log(&text, SbasLogOptions::default().with_policy(policy))
                } else {
                    parse_rtklib_log(&text, SbasLogOptions::default().with_policy(policy))
                }
            };
            let err = read(SbasPolicy::Strict).unwrap_err();
            assert!(
                matches!(err, Error::Parse(ref msg)
                    if msg.contains("line 2") && msg.contains("declares SBAS message type 4")),
                "expected the line-2 mismatch to be named, got {err:?}"
            );
            let log = read(SbasPolicy::Lenient).expect("lenient read");
            assert_eq!(log.blocks.len(), 2);
            assert_eq!(log.blocks[1].declared_message_type, Some(4));
            assert_eq!(log.blocks[1].message_type(), Some(2));
            assert_eq!(
                log.departures,
                vec![SbasLineDeparture {
                    line: 2,
                    departure: SbasDeparture::DeclaredMessageType {
                        declared: 4,
                        carried: 2,
                    },
                }]
            );
        }

        // An eight-field comma line states no type, so nothing is compared.
        let text = format!("120,26,7,1,0,0,1,{hex}\n");
        let log = parse_ems_log(&text, SbasLogOptions::default()).expect("no declared type");
        assert_eq!(log.blocks[0].declared_message_type, None);
    }

    #[test]
    fn skipped_lines_are_listed_with_their_line_numbers() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        let text = format!(
            "PRN  YY  MM  DD  hh  mm  ss  MT  message\n\
             \n\
             # comment 120 26 07 01\n\
             120 26 07 01 00 00 01 2 {hex}\n\
             not,enough\n\
             #120,26,7,1,0,0,1,2,{hex}\n\
             120,26,7,1,0,0,1,2,{hex}\n"
        );
        let log = parse_ems_log(&text, SbasLogOptions::default()).expect("parse EMS log");
        assert_eq!(log.blocks.len(), 2);
        assert_eq!(
            log.skipped_lines,
            vec![
                SbasSkippedLine {
                    line: 1,
                    kind: SbasSkippedLineKind::NonRecord,
                },
                SbasSkippedLine {
                    line: 2,
                    kind: SbasSkippedLineKind::Blank,
                },
                SbasSkippedLine {
                    line: 3,
                    kind: SbasSkippedLineKind::Comment,
                },
                SbasSkippedLine {
                    line: 5,
                    kind: SbasSkippedLineKind::NonRecord,
                },
                SbasSkippedLine {
                    line: 6,
                    kind: SbasSkippedLineKind::Comment,
                },
            ]
        );
        assert!(log.departures.is_empty());

        let text = format!(
            "% program   : RTKLIB\n\
             \n\
             # 2360 259200 120  2 : {hex}\n\
             2360 259200 120  2 : {hex}\n\
             end of file\n"
        );
        let log = parse_rtklib_log(&text, SbasLogOptions::default()).expect("parse RTKLIB log");
        assert_eq!(log.blocks.len(), 1);
        let kinds: Vec<(usize, SbasSkippedLineKind)> = log
            .skipped_lines
            .iter()
            .map(|skipped| (skipped.line, skipped.kind))
            .collect();
        assert_eq!(
            kinds,
            vec![
                (1, SbasSkippedLineKind::Comment),
                (2, SbasSkippedLineKind::Blank),
                (3, SbasSkippedLineKind::Comment),
                (5, SbasSkippedLineKind::NonRecord),
            ]
        );
    }

    #[test]
    fn ems_comma_records_with_unknown_field_positions_are_refused() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        // Previously a numeric line under eight fields was skipped, an
        // intermediate field past the ninth was dropped, and an empty field
        // shifted every field after it.
        for line in [
            "120,26,7,1,0,0,1".to_string(),
            format!("120,26,7,1,0,0,1,2,EXTRA,{hex}"),
            format!("120,26,7,1,0,0,1,2,3,{hex}"),
            format!("120,26,,7,1,0,0,1,{hex}"),
            format!("120,26,7,1,0,0,1,,{hex}"),
            format!("120,26,7,1,0,0,1,MT,{hex}"),
            format!("120,26,7,1,0,0,1,64,{hex}"),
        ] {
            assert!(
                parse_ems_lines(&format!("{line}\n")).is_err(),
                "expected {line:?} to be refused"
            );
        }
        // Empty fields after the message carry nothing.
        let parsed =
            parse_ems_lines(&format!("120,26,7,1,0,0,1,2,{hex},\n")).expect("trailing comma");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].bytes, RTKLIB_MT2_BODY.as_slice());
    }

    #[test]
    fn rtklib_records_with_unknown_header_fields_are_refused() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        for line in [
            // No separator: RTKLIB `readmsgs` skips this line, which would
            // drop the record without a trace.
            format!("2360 259200 120  2 {hex}"),
            // Three and five header fields.
            format!("2360 259200 120 : {hex}"),
            format!("2360 259200 120  2 7 : {hex}"),
            // The fourth field is the six-bit message type.
            format!("2360 259200 120 64 : {hex}"),
            format!("2360 259200 120 MT : {hex}"),
        ] {
            assert!(
                parse_rtklib_lines(&format!("{line}\n")).is_err(),
                "expected {line:?} to be refused"
            );
        }
    }

    // NovAtel lines carrying the MT2 body above. The checksums were
    // computed by a script independent of this reader: the OEM4 value is the
    // CRC-32 with reflected polynomial 0xEDB88320, initial value 0 and no
    // final XOR over the text between `#` and `*`; the OEM3 value is the XOR
    // of the text between `$` and `*`.
    const OEM4_MT2: &str = "#RAWWAASFRAMEA,COM1,0,77.5,SATTIME,2360,259200.000,00000000,58e4,1984;62,120,2,5308DFFC010005FFC00DFFC009FFDFFC001FFDFFDFFFBABBBBBB9BBB80*868ad9c9";
    const OEM4_MT2_DECLARED_4: &str = "#RAWWAASFRAMEA,COM1,0,77.5,SATTIME,2360,259200.000,00000000,58e4,1984;62,120,4,5308DFFC010005FFC00DFFC009FFDFFC001FFDFFDFFFBABBBBBB9BBB80*55f0f428";
    const OEM3_MT2_WEEK_312: &str =
        "$FRMA,312,259200.0,120,0,0,5308DFFC010005FFC00DFFC009FFDFFC001FFDFFDFFFBABBBBBB9BBB80*79";
    const OEM3_MT2_WEEK_1336: &str =
        "$FRMA,1336,259200.0,120,0,0,5308DFFC010005FFC00DFFC009FFDFFC001FFDFFDFFFBABBBBBB9BBB80*4E";

    #[test]
    fn novatel_crc32_matches_the_check_value() {
        // The CRC-32 of "123456789" with initial value 0 and no final XOR.
        assert_eq!(novatel_crc32(b"123456789"), 0x2DFD_2D88);
    }

    #[test]
    fn novatel_oem4_frames_read_in_both_readers() {
        let expected_epoch =
            GnssWeekTow::new(TimeScale::Gpst, 2360, 259_200.0).expect("valid epoch");
        let text = format!("{OEM4_MT2}\n");
        for log in [
            parse_ems_log(&text, SbasLogOptions::default()).expect("EMS reader"),
            parse_rtklib_log(&text, SbasLogOptions::default()).expect("RTKLIB reader"),
        ] {
            assert_eq!(log.blocks.len(), 1);
            let block = &log.blocks[0];
            assert_eq!(block.satellite_id.to_string(), "S20");
            assert_eq!(block.epoch, expected_epoch);
            assert_eq!(block.form, SbasWireForm::Body226);
            assert_eq!(block.bytes, RTKLIB_MT2_BODY.as_slice());
            assert_eq!(block.declared_message_type, Some(2));
            assert!(log.skipped_lines.is_empty() && log.refused_lines.is_empty());
        }

        // The message ID is the declared type, compared like any other.
        let text = format!("{OEM4_MT2_DECLARED_4}\n");
        assert!(parse_rtklib_lines(&text).is_err());
        let log = parse_rtklib_log(
            &text,
            SbasLogOptions::default().with_policy(SbasPolicy::Lenient),
        )
        .expect("lenient read");
        assert_eq!(
            log.departures[0].departure,
            SbasDeparture::DeclaredMessageType {
                declared: 4,
                carried: 2,
            }
        );

        // A damaged line fails its CRC and is left unread; a wrong data field
        // count is refused.
        let damaged = OEM4_MT2.replace(";62,120,", ";62,121,");
        assert!(parse_rtklib_lines(&damaged).is_err());
        let unchecked = OEM4_MT2.split('*').next().expect("content");
        assert_eq!(parse_rtklib_lines(unchecked).expect("no CRC").len(), 1);
        let extra_field = unchecked.replace(",5308", ",0,5308");
        assert!(parse_rtklib_lines(&extra_field).is_err());
    }

    #[test]
    fn novatel_oem3_ten_bit_weeks_resolve_from_the_reference_week() {
        // 2360 = 2 * 1024 + 312.
        let expected_epoch =
            GnssWeekTow::new(TimeScale::Gpst, 2360, 259_200.0).expect("valid epoch");
        let text = format!("{OEM3_MT2_WEEK_312}\n{OEM3_MT2_WEEK_1336}\n");

        // Without a reference week only the 10-bit line is left unread.
        let log = parse_ems_log(&text, SbasLogOptions::default()).expect("read");
        assert_eq!(
            log.refused_lines,
            vec![SbasRefusedLine {
                line: 1,
                reason: SbasLineRefusal::AmbiguousWeek { week: 312 },
            }]
        );
        assert_eq!(log.blocks.len(), 1);
        assert_eq!(log.blocks[0].epoch.week, 1336);
        assert_eq!(log.blocks[0].declared_message_type, None);
        assert!(parse_ems_lines(&text).is_err());

        for reference_week in [2360, 2100, 2871, 1848] {
            let log = parse_rtklib_log(
                &text,
                SbasLogOptions::default().with_reference_week(reference_week),
            )
            .expect("read");
            assert!(log.refused_lines.is_empty());
            assert_eq!(
                log.blocks[0].epoch, expected_epoch,
                "reference {reference_week}"
            );
            assert_eq!(log.blocks[0].bytes, RTKLIB_MT2_BODY.as_slice());
            // A full week is taken as written whatever the reference.
            assert_eq!(log.blocks[1].epoch.week, 1336);
        }

        let damaged = OEM3_MT2_WEEK_1336.replace(",120,", ",121,");
        assert!(parse_rtklib_lines(&damaged).is_err());
    }

    /// A line whose checksum does not match, or is not a checksum at all, is
    /// left unread with its reason, and the lines around it are read.
    #[test]
    fn novatel_checksum_mismatch_leaves_only_that_line_unread() {
        let damaged_oem4 = OEM4_MT2.replace(";62,120,", ";62,121,");
        let damaged_oem3 = OEM3_MT2_WEEK_1336.replace(",120,", ",121,");
        let unreadable_crc = OEM4_MT2.replace("*868ad9c9", "*868ad9cz");
        let short_crc = OEM4_MT2.replace("*868ad9c9", "*868ad9c");
        let text = format!(
            "{OEM4_MT2}\n{damaged_oem4}\n{OEM3_MT2_WEEK_1336}\n{damaged_oem3}\n{unreadable_crc}\n{short_crc}\n{OEM4_MT2}\n"
        );
        for log in [
            parse_ems_log(&text, SbasLogOptions::default()).expect("EMS reader"),
            parse_rtklib_log(&text, SbasLogOptions::default()).expect("RTKLIB reader"),
        ] {
            assert_eq!(log.blocks.len(), 3);
            let refused: Vec<usize> = log.refused_lines.iter().map(|r| r.line).collect();
            assert_eq!(refused, vec![2, 4, 5, 6]);
            assert!(matches!(
                log.refused_lines[0].reason,
                SbasLineRefusal::ChecksumMismatch {
                    written: Some(0x868a_d9c9),
                    ..
                }
            ));
            assert!(matches!(
                log.refused_lines[1].reason,
                SbasLineRefusal::ChecksumMismatch {
                    written: Some(0x4E),
                    ..
                }
            ));
            for refused in &log.refused_lines[2..] {
                assert_eq!(
                    refused.reason,
                    SbasLineRefusal::ChecksumMismatch {
                        written: None,
                        computed: 0x868a_d9c9,
                    }
                );
            }
        }
        assert!(parse_rtklib_lines(&text).is_err());
    }

    #[test]
    fn ten_bit_week_resolves_to_the_closest_full_week() {
        assert_eq!(resolve_ten_bit_week(312, 2360), 2360);
        // 512 either side resolves to the later week.
        assert_eq!(resolve_ten_bit_week(312, 1848), 2360);
        assert_eq!(resolve_ten_bit_week(312, 2872), 2872 + 512);
        assert_eq!(resolve_ten_bit_week(312, 2871), 2360);
        // No week before week 0 and none past u32::MAX: the other candidate.
        assert_eq!(resolve_ten_bit_week(1000, 0), 1000);
        assert_eq!(resolve_ten_bit_week(0, 0), 0);
        // u32::MAX = 4_194_303 * 1024 + 1023; the closest candidate for week
        // 0 near u32::MAX is u32::MAX + 1, one past the range.
        assert_eq!(resolve_ten_bit_week(0, u32::MAX), u32::MAX - 1023);
        assert_eq!(resolve_ten_bit_week(1023, u32::MAX), u32::MAX);
    }

    /// RTKLIB `readmsgs`: `ep[0]+=ep[0]<70.0?2000.0:1900.0`. The expected
    /// weeks and seconds were computed from the Sunday 1980-01-06 GPS week
    /// origin, not read back from this parser.
    #[test]
    fn ems_two_digit_years_follow_rtklib() {
        let hex = block_hex(&RTKLIB_MT2_BODY);
        for (line, week, tow_s) in [
            (format!("120 99 12 31 23 59 59 2 {hex}"), 1042, 518_399.0),
            (format!("120 69 01 01 00 00 00 2 {hex}"), 4643, 172_800.0),
            (format!("120 1999 12 31 23 59 59 2 {hex}"), 1042, 518_399.0),
        ] {
            let parsed = parse_ems_lines(&format!("{line}\n")).expect("parse year");
            assert_eq!(
                parsed[0].epoch,
                GnssWeekTow::new(TimeScale::Gpst, week, tow_s).expect("valid epoch"),
                "{line}"
            );
        }
    }
}
