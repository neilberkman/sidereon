//! RINEX navigation serialization - the inverse of [`super::parse_nav_file`].
//!
//! Pure and deterministic: a given input always produces byte-identical text and
//! no I/O is performed.
//!
//! [`encode_nav_file`] writes a [`NavFile`]: the header and each entry are
//! restated from the text they were read from when that text still reads as the
//! value held, so a file read and written unchanged comes back byte for byte;
//! anything built or changed in code is formatted from its fields in the file's
//! version. [`encode_nav`] writes a set of Keplerian records as a RINEX 3.04 file,
//! or RINEX 4.02 frames when a CNAV-family record is present.
//!
//! Numeric fields use the RINEX `D19.12` width, the same 13-significant-figure grid
//! the files carry, so a value read from a real file re-encodes to the same `f64`.

use core::fmt::Write as _;

use crate::astro::constants::time::{SECONDS_PER_DAY_I64, SECONDS_PER_HOUR, SECONDS_PER_WEEK};
use crate::astro::time::civil::civil_from_julian_day_number;
use crate::astro::time::gnss::week_epoch_julian_day_number;
use crate::astro::time::model::{GnssWeekTow, TimeScale};
use crate::astro::time::scales::julian_day_number;
use crate::constants::KM_TO_M;
use crate::id::GnssSystem;
use crate::rinex_obs::PgmRunByDate;

use super::body::{decode_block, is_v4_frame_marker, NavEntry, NavEntryKind, NavFile, NavItem};
use super::frames::{EarthOrientation, IonosphereFrame, SystemTimeOffset};
use super::header::{self, HeaderIonoRow, NavHeader, TimeSystemCorrection};
use super::{
    galileo_message, gps_fit_interval_from_value, qzss_fit_interval_s, BroadcastRecord,
    CnavParameters, GlonassRecord, IonoCorrections, NavEpoch, NavMessage, NavVersion, SbasRecord,
};

/// Why a navigation file cannot be written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NavWriteError {
    /// An entry the file's version cannot hold: a CNAV-family record or a RINEX 4
    /// frame in RINEX 2 or 3, a record of another system than a RINEX 2 file's type,
    /// or a Galileo record whose message is unclassified in RINEX 4, whose frame
    /// marker has to name the message.
    NotRepresentable {
        /// 1-based line of the entry in the file it was read from; 0 for an entry
        /// built in code.
        line: usize,
        /// What the version cannot hold.
        reason: &'static str,
    },
}

impl core::fmt::Display for NavWriteError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NavWriteError::NotRepresentable { line, reason } => {
                write!(
                    f,
                    "navigation entry at line {line} cannot be written: {reason}"
                )
            }
        }
    }
}

impl std::error::Error for NavWriteError {}

/// Serialize broadcast navigation records to standard RINEX navigation text: RINEX
/// 3.04, or RINEX 4.02 frames when a CNAV-family record is present. The header
/// carries `RINEX VERSION / TYPE`, `PGM / RUN BY / DATE` (program `sidereon`, the
/// run-by and date fields blank, since no clock is read) and `END OF HEADER`.
///
/// Re-parsing the output with [`super::parse_nav`] yields the same records. A set
/// holding both a CNAV-family record, which only RINEX 4 holds, and a Galileo
/// record whose message is unclassified, which only RINEX 3 holds, is refused.
pub fn encode_nav(records: &[BroadcastRecord]) -> Result<String, NavWriteError> {
    let use_v4 = records.iter().any(|record| record.message.is_cnav_family());
    let version = if use_v4 {
        NavVersion::new(4, 2)
    } else {
        NavVersion::new(3, 4)
    };
    let mut header = NavHeader::new(version);
    header.program = Some(PgmRunByDate {
        program: "sidereon".to_string(),
        run_by: String::new(),
        date: String::new(),
    });
    let mut file = NavFile::new(header);
    file.entries = records
        .iter()
        .map(|record| NavEntry::new(NavItem::Ephemeris(*record)))
        .collect();
    encode_nav_file(&file)
}

/// Serialize a [`NavFile`]. The header and each entry are restated from their text
/// when it still reads, in the file's version, as the value held (for an entry, with no
/// departure from the format it did not already have); otherwise they are formatted
/// from their fields. Undecoded blocks and stray lines are restated from
/// their text.
pub fn encode_nav_file(file: &NavFile) -> Result<String, NavWriteError> {
    let version = file.header.version;
    let file_type = file.header.file_type;
    let mut lines: Vec<String> = Vec::new();

    if header_text_reads_as(&file.header) {
        lines.extend(file.header.text.iter().cloned());
    } else {
        lines.extend(format_header(&file.header));
    }

    for entry in &file.entries {
        if entry_text_reads_as(entry, version, file_type) {
            lines.extend(entry.text.iter().cloned());
            continue;
        }
        lines.extend(format_entry(entry, version, file_type)?);
    }

    let terminator = if file.crlf { "\r\n" } else { "\n" };
    let mut out = String::with_capacity(lines.iter().map(|l| l.len() + 2).sum());
    let count = lines.len();
    for (index, line) in lines.iter().enumerate() {
        out.push_str(line);
        if index + 1 < count || file.final_newline {
            out.push_str(terminator);
        }
    }
    Ok(out)
}

fn strip_cr(line: &str) -> &str {
    line.strip_suffix('\r').unwrap_or(line)
}

fn header_text_reads_as(header: &NavHeader) -> bool {
    if header.text.is_empty() {
        return false;
    }
    let lines: Vec<&str> = header.text.iter().map(|line| strip_cr(line)).collect();
    match header::read_header(&lines) {
        Ok(read) => {
            read.body_start == lines.len() && read.header.without_text() == header.without_text()
        }
        Err(_) => false,
    }
}

fn entry_text_reads_as(entry: &NavEntry, version: NavVersion, file_type: char) -> bool {
    if entry.text.is_empty() {
        return false;
    }
    if matches!(entry.item, NavItem::Undecoded(_)) || entry.kind == NavEntryKind::Stray {
        return true;
    }
    let lines: Vec<&str> = entry.text.iter().map(|line| strip_cr(line)).collect();
    // A v4 entry's first line is its frame marker; a v2/v3 entry's is a record start.
    if version.major >= 4 && !lines.first().is_some_and(|line| is_v4_frame_marker(line)) {
        return false;
    }
    // Restated only when the text, read in the file's version (which an edit may have
    // changed), is the held value and reads with no departure the entry did not
    // already have.
    let decoded = decode_block(&lines, version, file_type);
    decoded.item == entry.item
        && decoded
            .departures
            .iter()
            .all(|departure| entry.departures.contains(departure))
}

/// Format the header from its fields.
fn format_header(header: &NavHeader) -> Vec<String> {
    let version = header.version;
    let mut out = Vec::new();
    let version_number = format!("{}.{:02}", version.major, version.minor);
    let version_text = format!("{version_number:>9}");
    let type_text = match (version.major, header.file_type) {
        (2, 'N') => "N: GPS NAV DATA".to_string(),
        (2, 'G') => "G: GLONASS NAV DATA".to_string(),
        (2, 'H') => "H: GEO NAV MSG DATA".to_string(),
        (_, file_type) => format!("{file_type}: GNSS NAV DATA"),
    };
    let system_text = header
        .satellite_system
        .map(|system| system.to_string())
        .unwrap_or_default();
    out.push(label_line(
        &format!("{version_text:<20}{type_text:<20}{system_text:<20}"),
        "RINEX VERSION / TYPE",
    ));
    if let Some(program) = &header.program {
        out.push(label_line(
            &format!(
                "{:<20}{:<20}{:<20}",
                truncate(&program.program, 20),
                truncate(&program.run_by, 20),
                truncate(&program.date, 20)
            ),
            "PGM / RUN BY / DATE",
        ));
    }
    for comment in &header.comments {
        out.push(label_line(truncate(comment, 60), "COMMENT"));
    }
    let rows = if header.iono_rows.is_empty()
        || header::iono_from_rows(&header.iono_rows) != header.iono
    {
        rows_from_iono(&header.iono)
    } else {
        header.iono_rows.clone()
    };
    for row in &rows {
        out.push(format_iono_row(row, version));
    }
    for correction in &header.time_system_corrections {
        out.push(format_time_system_correction(correction, version));
    }
    if let Some(leap) = &header.leap_seconds {
        let int = |value: Option<i64>| {
            value
                .map(|v| format!("{v:>6}"))
                .unwrap_or_else(|| " ".repeat(6))
        };
        let system = leap.time_system.clone().unwrap_or_default();
        out.push(label_line(
            &format!(
                "{}{}{}{}{:<3}",
                int(Some(leap.current)),
                int(leap.delta_future),
                int(leap.week),
                int(leap.day),
                system
            ),
            "LEAP SECONDS",
        ));
    }
    out.extend(header.other_records.iter().cloned());
    out.push(label_line("", "END OF HEADER"));
    out
}

fn truncate(text: &str, width: usize) -> &str {
    match text.char_indices().nth(width) {
        Some((index, _)) => &text[..index],
        None => text,
    }
}

/// A header line: `content` in columns 1-60, the label from column 61.
fn label_line(content: &str, label: &str) -> String {
    format!("{:<60}{label}", truncate(content, 60))
}

/// Header rows for coefficient sets built in code, in the order RINEX lists them.
fn rows_from_iono(iono: &IonoCorrections) -> Vec<HeaderIonoRow> {
    let mut rows = Vec::new();
    let mut klobuchar = |a: &str, b: &str, set: Option<super::KlobucharAlphaBeta>| {
        if let Some(set) = set {
            for (label, values) in [(a, set.alpha), (b, set.beta)] {
                rows.push(HeaderIonoRow {
                    label: label.to_string(),
                    values: values.map(Some),
                    time_mark: None,
                    satellite: None,
                });
            }
        }
    };
    klobuchar("GPSA", "GPSB", iono.gps);
    klobuchar("QZSA", "QZSB", iono.qzss);
    klobuchar("BDSA", "BDSB", iono.beidou);
    klobuchar("IRNA", "IRNB", iono.navic);
    if let Some(gal) = iono.galileo {
        rows.push(HeaderIonoRow {
            label: "GAL".to_string(),
            values: [
                Some(gal.ai0),
                Some(gal.ai1),
                Some(gal.ai2),
                iono.galileo_disturbance_flags,
            ],
            time_mark: None,
            satellite: None,
        });
    }
    rows
}

/// A value in a `Dw.d` field: sign or blank, one digit, `decimals` fraction digits,
/// `e`, and a signed two-digit exponent, right-aligned in `width` columns.
fn exp_field(value: f64, decimals: usize, width: usize) -> String {
    let negative = value.is_sign_negative() && value != 0.0;
    let base = format!("{:.*e}", decimals, value.abs());
    let (mantissa, exponent) = base.split_once('e').unwrap_or((base.as_str(), "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let sign = if negative { "-" } else { "" };
    let text = format!("{sign}{mantissa}e{exponent:+03}");
    format!("{text:>width$}")
}

fn optional_exp(value: Option<f64>, decimals: usize, width: usize) -> String {
    match value {
        Some(value) => exp_field(value, decimals, width),
        None => " ".repeat(width),
    }
}

/// A coefficient row: RINEX 2 writes the GPS rows as `ION ALPHA`/`ION BETA`
/// (`2X,4D12.4`); every other row is an `IONOSPHERIC CORR` record, which the reader
/// takes in any version.
fn format_iono_row(row: &HeaderIonoRow, version: NavVersion) -> String {
    let v2_label = match row.label.as_str() {
        "GPSA" => Some("ION ALPHA"),
        "GPSB" => Some("ION BETA"),
        _ => None,
    };
    if let (2, Some(label)) = (version.major, v2_label) {
        let values: String = row
            .values
            .iter()
            .map(|value| optional_exp(*value, 4, 12))
            .collect();
        return label_line(&format!("  {values}"), label);
    }
    let values: String = row
        .values
        .iter()
        .map(|value| optional_exp(*value, 4, 12))
        .collect();
    let mut content = format!("{:<4} {values}", row.label);
    if row.time_mark.is_some() || row.satellite.is_some() {
        let mark = row
            .time_mark
            .map(String::from)
            .unwrap_or_else(|| " ".to_string());
        let satellite = row.satellite.clone().unwrap_or_default();
        content.push_str(&format!(" {mark} {satellite:<3}"));
    }
    label_line(&content, "IONOSPHERIC CORR")
}

fn format_time_system_correction(correction: &TimeSystemCorrection, version: NavVersion) -> String {
    let int = |value: Option<f64>, width: usize| match value {
        Some(value) if value.trunc() == value => format!("{:>width$}", value as i64),
        Some(value) => format!("{value:>width$}"),
        None => " ".repeat(width),
    };
    if version.major == 2 && correction.code == "GPUT" {
        return label_line(
            &format!(
                "   {}{}{}{}",
                exp_field(correction.a0_s, 12, 19),
                optional_exp(correction.a1_s_s, 12, 19),
                int(correction.reference_time_s, 9),
                int(correction.reference_week, 9)
            ),
            "DELTA-UTC: A0,A1,T,W",
        );
    }
    label_line(
        &format!(
            "{:<4} {}{}{}{} {:<5} {:<2}",
            correction.code,
            exp_field(correction.a0_s, 10, 17),
            optional_exp(correction.a1_s_s, 9, 16),
            int(correction.reference_time_s, 7),
            int(correction.reference_week, 5),
            correction.source.clone().unwrap_or_default(),
            correction.utc_id.clone().unwrap_or_default()
        ),
        "TIME SYSTEM CORR",
    )
}

fn not_representable(entry: &NavEntry, reason: &'static str) -> NavWriteError {
    NavWriteError::NotRepresentable {
        line: entry.line,
        reason,
    }
}

/// Format an entry from its fields in `version`.
fn format_entry(
    entry: &NavEntry,
    version: NavVersion,
    file_type: char,
) -> Result<Vec<String>, NavWriteError> {
    let mut out = Vec::new();
    match &entry.item {
        NavItem::Ephemeris(record) => {
            if version.major >= 4 {
                let token = message_token(record.message).ok_or_else(|| {
                    not_representable(
                        entry,
                        "an unclassified Galileo message has no RINEX 4 token",
                    )
                })?;
                out.push(format!("> EPH {} {token}", record.satellite_id));
                if record.message.is_cnav_family() {
                    let cnav = record.cnav.ok_or_else(|| {
                        not_representable(entry, "a CNAV-family record without CNAV parameters")
                    })?;
                    write_cnav_record(&mut out, record, &cnav);
                } else {
                    write_record(&mut out, record, version, RecordColumns::V3);
                }
            } else {
                if record.message.is_cnav_family() {
                    return Err(not_representable(
                        entry,
                        "a CNAV-family record needs RINEX 4",
                    ));
                }
                let columns = v2_columns(entry, version, file_type, record.satellite_id.system)?;
                write_record(&mut out, record, version, columns);
            }
        }
        NavItem::Glonass(record) => {
            let columns = if version.major >= 4 {
                out.push(format!("> EPH {} FDMA", record.satellite_id));
                RecordColumns::V3
            } else {
                v2_columns(entry, version, file_type, GnssSystem::Glonass)?
            };
            write_glonass_record(&mut out, record, version, columns);
        }
        NavItem::Sbas(record) => {
            let columns = if version.major >= 4 {
                out.push(format!("> EPH {} SBAS", record.satellite_id));
                RecordColumns::V3
            } else {
                v2_columns(entry, version, file_type, GnssSystem::Sbas)?
            };
            write_sbas_record(&mut out, record, columns);
        }
        NavItem::SystemTimeOffset(frame) => {
            require_v4(entry, version)?;
            write_sto_frame(&mut out, frame);
        }
        NavItem::EarthOrientation(frame) => {
            require_v4(entry, version)?;
            write_eop_frame(&mut out, frame);
        }
        NavItem::Ionosphere(frame) => {
            require_v4(entry, version)?;
            write_ion_frame(&mut out, frame);
        }
        NavItem::Undecoded(_) => out.extend(entry.text.iter().cloned()),
    }
    Ok(out)
}

fn require_v4(entry: &NavEntry, version: NavVersion) -> Result<(), NavWriteError> {
    if version.major >= 4 {
        Ok(())
    } else {
        Err(not_representable(entry, "a RINEX 4 frame needs RINEX 4"))
    }
}

/// The columns a record is written in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum RecordColumns {
    /// Version 3/4: the three-character satellite token, a four-digit year.
    V3,
    /// Version 2 with a numeric PRN.
    V2 { prn: u8 },
}

/// The version 2 columns for a record of `system` in a file of `file_type`: a
/// numeric PRN in `N` (GPS, and QZSS as PRN 93-97), `G` and `H` files, the version
/// 3 columns in the lettered `J` and `L` extensions.
fn v2_columns(
    entry: &NavEntry,
    version: NavVersion,
    file_type: char,
    system: GnssSystem,
) -> Result<RecordColumns, NavWriteError> {
    if version.major != 2 {
        return Ok(RecordColumns::V3);
    }
    let prn = match &entry.item {
        NavItem::Ephemeris(record) => record.satellite_id.prn,
        NavItem::Glonass(record) => record.satellite_id.prn,
        NavItem::Sbas(record) => record.satellite_id.prn,
        _ => 0,
    };
    match (file_type, system) {
        ('N', GnssSystem::Gps) | ('G', GnssSystem::Glonass) | ('H', GnssSystem::Sbas) => {
            Ok(RecordColumns::V2 { prn })
        }
        ('N', GnssSystem::Qzss) if (1..=5).contains(&prn) => {
            Ok(RecordColumns::V2 { prn: prn + 92 })
        }
        ('J', GnssSystem::Qzss) | ('L', GnssSystem::Galileo) => Ok(RecordColumns::V3),
        _ => Err(not_representable(
            entry,
            "a RINEX 2 file holds records of its own file type's system",
        )),
    }
}

/// The first line's satellite and epoch prefix.
fn epoch_prefix(
    columns: RecordColumns,
    satellite: &str,
    epoch: (i64, i64, i64, i64, i64, f64),
) -> String {
    let (year, month, day, hour, minute, second) = epoch;
    match columns {
        RecordColumns::V3 => format!(
            "{satellite:<3} {year:04} {month:02} {day:02} {hour:02} {minute:02} {:02}",
            second.floor() as i64
        ),
        RecordColumns::V2 { prn } => format!(
            "{prn:>2} {:02} {month:>2} {day:>2} {hour:>2} {minute:>2}{second:>5.1}",
            year.rem_euclid(100)
        ),
    }
}

fn indent(columns: RecordColumns) -> &'static str {
    match columns {
        RecordColumns::V3 => "    ",
        RecordColumns::V2 { .. } => "   ",
    }
}

fn nav_epoch_tuple(epoch: &NavEpoch) -> (i64, i64, i64, i64, i64, f64) {
    (
        i64::from(epoch.year),
        i64::from(epoch.month),
        i64::from(epoch.day),
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        epoch.second,
    )
}

fn write_record(
    out: &mut Vec<String>,
    record: &BroadcastRecord,
    version: NavVersion,
    columns: RecordColumns,
) {
    let sat = record.satellite_id;
    let system = sat.system;
    let (year, month, day, hour, minute, second) = clock_epoch_civil(record);
    let mut line = epoch_prefix(
        columns,
        &sat.to_string(),
        (year, month, day, hour, minute, second as f64),
    );
    push_d19_12(&mut line, record.clock.af0);
    push_d19_12(&mut line, record.clock.af1);
    push_d19_12(&mut line, record.clock.af2);
    out.push(line);

    let e = &record.elements;
    let stated = &record.stated;
    let pad = indent(columns);
    // ORBIT-1 .. ORBIT-7, matching the fixed-column reader's field positions.
    out.push(orbit_line(
        pad,
        [
            record.issue_of_data.map(|issue| f64::from(issue.issue)),
            Some(e.crs),
            Some(e.delta_n),
            Some(e.m0),
        ],
    ));
    out.push(orbit_line(
        pad,
        [Some(e.cuc), Some(e.e), Some(e.cus), Some(e.sqrt_a)],
    ));
    out.push(orbit_line(
        pad,
        [Some(e.toe_sow), Some(e.cic), Some(e.omega0), Some(e.cis)],
    ));
    out.push(orbit_line(
        pad,
        [Some(e.i0), Some(e.crc), Some(e.omega), Some(e.omega_dot)],
    ));
    let orbit5_field2 = if system == GnssSystem::Galileo {
        Some(galileo_data_sources(record, version))
    } else {
        stated.orbit5_field2
    };
    out.push(orbit_line(
        pad,
        [
            Some(e.idot),
            orbit5_field2,
            Some(f64::from(record.week)),
            stated.orbit5_field4,
        ],
    ));
    let gd = &record.group_delays;
    let (delay3, field4) = match system {
        GnssSystem::Galileo => (gd.galileo_bgd_e5a_e1_s, gd.galileo_bgd_e5b_e1_s),
        GnssSystem::BeiDou => (gd.beidou_tgd1_s, gd.beidou_tgd2_s),
        _ => (gd.gps_tgd_s, stated.orbit6_field4),
    };
    out.push(orbit_line(
        pad,
        [record.sv_accuracy_m, Some(record.sv_health), delay3, field4],
    ));
    let orbit7_field2 = match system {
        GnssSystem::Gps => gps_fit_field(record, version),
        GnssSystem::Qzss => qzss_fit_field(record),
        _ => stated.orbit7_field2,
    };
    out.push(orbit_line(
        pad,
        [
            stated.transmission_time_sow,
            orbit7_field2,
            stated.orbit7_field3,
            stated.orbit7_field4,
        ],
    ));
}

/// The Galileo data-source word to write: the stated word when it reads, in
/// `version`, as the record's message, else the source bit naming that message with the
/// clock bit it implies, as RTKLIB's RTCM decoder writes the words it forms: 517 (bits
/// 0, 2 and 9) for I/NAV and 258 (bits 1 and 8) for F/NAV. A RINEX 4 frame names the
/// message itself, so there the stated word is written as it is.
fn galileo_data_sources(record: &BroadcastRecord, version: NavVersion) -> f64 {
    if let Some(word) = record.stated.orbit5_field2 {
        if version.major >= 4
            || galileo_message(word, "").is_ok_and(|(message, _)| message == record.message)
        {
            return word;
        }
    }
    match record.message {
        NavMessage::GalileoFnav => f64::from(0b10 | (1 << 8)),
        NavMessage::GalileoInav => f64::from(0b101 | (1 << 9)),
        _ => 0.0,
    }
}

/// The GPS ORBIT-7 field 2 to write: the stated value when it reads, in `version`,
/// as the record's fit interval, else the fit interval in hours.
fn gps_fit_field(record: &BroadcastRecord, version: NavVersion) -> Option<f64> {
    if let Some(value) = record.stated.orbit7_field2 {
        // A negative value states no fit interval.
        let reads_as = if value < 0.0 {
            None
        } else {
            gps_fit_interval_from_value(value, version)
        };
        if reads_as == record.fit_interval_s {
            return Some(value);
        }
    }
    record
        .fit_interval_s
        .map(|seconds| seconds / SECONDS_PER_HOUR)
}

/// The QZSS fit flag to write: the stated flag when it reads as the record's fit
/// interval, else 0 for two hours and 1 for anything longer.
fn qzss_fit_field(record: &BroadcastRecord) -> Option<f64> {
    if let Some(value) = record.stated.orbit7_field2 {
        if Some(qzss_fit_interval_s(value)) == record.fit_interval_s {
            return Some(value);
        }
    }
    record.fit_interval_s.map(|seconds| {
        if seconds == super::QZSS_SHORT_FIT_INTERVAL_S {
            0.0
        } else {
            1.0
        }
    })
}

fn orbit_line(indent: &str, values: [Option<f64>; 4]) -> String {
    let mut line = String::with_capacity(80);
    line.push_str(indent);
    for value in values {
        match value {
            Some(value) => push_d19_12(&mut line, value),
            None => line.push_str("                   "),
        }
    }
    line
}

// invariant: write_cnav_record is dispatched only for a record carrying CNAV data.
#[allow(clippy::expect_used)]
fn write_cnav_record(out: &mut Vec<String>, record: &BroadcastRecord, cnav: &CnavParameters) {
    let top = serialized_week_tow(cnav.top);

    let (year, month, day, hour, minute, second) = clock_epoch_civil(record);
    let mut line = epoch_prefix(
        RecordColumns::V3,
        &record.satellite_id.to_string(),
        (year, month, day, hour, minute, second as f64),
    );
    push_d19_12(&mut line, record.clock.af0);
    push_d19_12(&mut line, record.clock.af1);
    push_d19_12(&mut line, record.clock.af2);
    out.push(line);

    let e = &record.elements;
    let gd = &record.group_delays;
    let pad = "    ";
    out.push(orbit_line(
        pad,
        [
            Some(cnav.adot_m_s),
            Some(e.crs),
            Some(e.delta_n),
            Some(e.m0),
        ],
    ));
    out.push(orbit_line(
        pad,
        [Some(e.cuc), Some(e.e), Some(e.cus), Some(e.sqrt_a)],
    ));
    out.push(orbit_line(
        pad,
        [Some(top.tow_s), Some(e.cic), Some(e.omega0), Some(e.cis)],
    ));
    out.push(orbit_line(
        pad,
        [Some(e.i0), Some(e.crc), Some(e.omega), Some(e.omega_dot)],
    ));
    out.push(orbit_line(
        pad,
        [
            Some(e.idot),
            Some(cnav.delta_n0_dot_rad_s2),
            Some(f64::from(cnav.ura_ned0_index)),
            Some(f64::from(cnav.ura_ned1_index)),
        ],
    ));
    out.push(orbit_line(
        pad,
        [
            Some(f64::from(cnav.ura_ed_index)),
            Some(record.sv_health),
            gd.gps_tgd_s,
            Some(f64::from(cnav.ura_ned2_index)),
        ],
    ));
    out.push(orbit_line(
        pad,
        [
            gd.cnav_isc_l1ca_s,
            gd.cnav_isc_l2c_s,
            gd.cnav_isc_l5i5_s,
            gd.cnav_isc_l5q5_s,
        ],
    ));
    if matches!(record.message, NavMessage::GpsCnav2 | NavMessage::QzssCnav2) {
        out.push(orbit_line(
            pad,
            [gd.cnav_isc_l1cd_s, gd.cnav_isc_l1cp_s, None, None],
        ));
    }
    out.push(orbit_line(
        pad,
        [
            Some(cnav.transmission_time_sow),
            Some(f64::from(top.week)),
            cnav.flags.map(f64::from),
            None,
        ],
    ));
}

fn write_glonass_record(
    out: &mut Vec<String>,
    record: &GlonassRecord,
    version: NavVersion,
    columns: RecordColumns,
) {
    let epoch = NavEpoch::from_j2000_s(record.epoch_utc_j2000_s);
    let mut line = epoch_prefix(
        columns,
        &record.satellite_id.to_string(),
        nav_epoch_tuple(&epoch),
    );
    push_d19_12(&mut line, record.clk_bias);
    push_d19_12(&mut line, record.gamma_n);
    push_optional_d19_12(&mut line, record.message_frame_time_s);
    out.push(line);
    let pad = indent(columns);
    let km = |value: f64| Some(value / KM_TO_M);
    out.push(orbit_line(
        pad,
        [
            km(record.pos_m[0]),
            km(record.vel_m_s[0]),
            km(record.acc_m_s2[0]),
            Some(record.sv_health),
        ],
    ));
    out.push(orbit_line(
        pad,
        [
            km(record.pos_m[1]),
            km(record.vel_m_s[1]),
            km(record.acc_m_s2[1]),
            Some(f64::from(
                if super::fold_glonass_channel(record.stated_freq_channel) == record.freq_channel {
                    record.stated_freq_channel
                } else {
                    record.freq_channel
                },
            )),
        ],
    ));
    out.push(orbit_line(
        pad,
        [
            km(record.pos_m[2]),
            km(record.vel_m_s[2]),
            km(record.acc_m_s2[2]),
            record.age_days,
        ],
    ));
    if version.glonass_has_fourth_orbit_line() {
        out.push(orbit_line(
            pad,
            [
                record.status_flags,
                record.l1_l2_group_delay_field_s,
                record.urai,
                record.health_flags,
            ],
        ));
    }
}

fn write_sbas_record(out: &mut Vec<String>, record: &SbasRecord, columns: RecordColumns) {
    let mut line = epoch_prefix(
        columns,
        &record.satellite_id.to_string(),
        nav_epoch_tuple(&record.epoch),
    );
    push_d19_12(&mut line, record.af0_s);
    push_d19_12(&mut line, record.af1_s_s);
    push_optional_d19_12(&mut line, record.message_frame_time_s);
    out.push(line);
    let pad = indent(columns);
    let km = |value: f64| Some(value / KM_TO_M);
    out.push(orbit_line(
        pad,
        [
            km(record.pos_m[0]),
            km(record.vel_m_s[0]),
            km(record.acc_m_s2[0]),
            Some(record.health),
        ],
    ));
    out.push(orbit_line(
        pad,
        [
            km(record.pos_m[1]),
            km(record.vel_m_s[1]),
            km(record.acc_m_s2[1]),
            record.ura_m,
        ],
    ));
    out.push(orbit_line(
        pad,
        [
            km(record.pos_m[2]),
            km(record.vel_m_s[2]),
            km(record.acc_m_s2[2]),
            record.iodn,
        ],
    ));
}

/// The 19-column epoch `yyyy mm dd hh mm ss` of a RINEX 4 frame's first body line.
fn frame_epoch(epoch: &NavEpoch) -> String {
    format!(
        "{:04} {:02} {:02} {:02} {:02} {:02}",
        epoch.year,
        epoch.month,
        epoch.day,
        epoch.hour,
        epoch.minute,
        epoch.second.floor() as i64
    )
}

fn write_sto_frame(out: &mut Vec<String>, frame: &SystemTimeOffset) {
    out.push(format!(
        "> STO {} {}",
        frame.satellite_id, frame.message_token
    ));
    out.push(format!(
        "    {} {:<18} {:<18} {:<18}",
        frame_epoch(&frame.reference_epoch),
        frame.offset_code,
        frame.sbas_id.clone().unwrap_or_default(),
        frame.utc_id.clone().unwrap_or_default()
    ));
    out.push(orbit_line(
        "    ",
        [
            Some(frame.transmission_time_sow),
            Some(frame.a0_s),
            frame.a1_s_s,
            frame.a2_s_s2,
        ],
    ));
}

fn write_eop_frame(out: &mut Vec<String>, frame: &EarthOrientation) {
    out.push(format!(
        "> EOP {} {}",
        frame.satellite_id, frame.message_token
    ));
    let mut line = format!("    {}", frame_epoch(&frame.reference_epoch));
    for value in frame.xp {
        push_optional_d19_12(&mut line, value);
    }
    out.push(line);
    let mut line = String::from("                       ");
    for value in frame.yp {
        push_optional_d19_12(&mut line, value);
    }
    out.push(line);
    out.push(orbit_line(
        "    ",
        [
            Some(frame.transmission_time_sow),
            frame.dut1[0],
            frame.dut1[1],
            frame.dut1[2],
        ],
    ));
}

fn write_ion_frame(out: &mut Vec<String>, frame: &IonosphereFrame) {
    out.push(format!(
        "> ION {} {}",
        frame.satellite_id, frame.message_token
    ));
    let mut line = format!("    {}", frame_epoch(&frame.transmission_epoch));
    let mut values = frame.values.iter();
    for _ in 0..3 {
        push_optional_d19_12(&mut line, values.next().copied().flatten());
    }
    out.push(line.trim_end().to_string());
    let rest: Vec<Option<f64>> = values.copied().collect();
    for chunk in rest.chunks(4) {
        let mut line = String::from("    ");
        for value in chunk {
            push_optional_d19_12(&mut line, *value);
        }
        out.push(line.trim_end().to_string());
    }
}

/// Reconstruct the civil toc epoch (integer seconds) from a record's `toc`
/// week / seconds-of-week, in the record's broadcast time scale.
fn clock_epoch_civil(record: &BroadcastRecord) -> (i64, i64, i64, i64, i64, i64) {
    let week = i64::from(record.toc.week);
    let sow = record.toc.tow_s.round() as i64;
    let base_jdn = week_epoch_jdn(record.toc.system);
    let total_jdn = base_jdn + week * 7 + sow.div_euclid(SECONDS_PER_DAY_I64);
    let tod = sow.rem_euclid(SECONDS_PER_DAY_I64);
    let (year, month, day) = civil_from_julian_day_number(total_jdn);
    let hour = tod / 3600;
    let minute = (tod % 3600) / 60;
    let second = tod % 60;
    (year, month, day, hour, minute, second)
}

/// Julian Day Number of the start of a constellation's week numbering, the
/// inverse-side companion of [`super::gnss::week_from_calendar`]. Any scale that
/// never reaches a Keplerian broadcast record falls back to the GPS epoch.
fn week_epoch_jdn(scale: TimeScale) -> i64 {
    week_epoch_julian_day_number(scale).unwrap_or_else(|| julian_day_number(1980, 1, 6))
}

fn message_token(message: NavMessage) -> Option<&'static str> {
    Some(match message {
        NavMessage::GpsLnav | NavMessage::QzssLnav | NavMessage::NavicLnav => "LNAV",
        NavMessage::GpsCnav | NavMessage::QzssCnav => "CNAV",
        NavMessage::GpsCnav2 | NavMessage::QzssCnav2 => "CNV2",
        NavMessage::GalileoInav => "INAV",
        NavMessage::GalileoFnav => "FNAV",
        NavMessage::BeidouD1 => "D1",
        NavMessage::BeidouD2 => "D2",
        NavMessage::GalileoUnclassified => return None,
    })
}

fn push_optional_d19_12(out: &mut String, value: Option<f64>) {
    match value {
        Some(value) => push_d19_12(out, value),
        None => out.push_str("                   "),
    }
}

/// Append a value in RINEX `D19.12` fixed-width form: a leading sign or space,
/// one mantissa digit, twelve fraction digits, and a signed two-digit exponent
/// (e.g. ` 1.000000000000e+00`, `-2.907656250000e+02`), always 19 columns.
// invariant: Rust's fixed scientific formatter always emits an `e` separator.
#[allow(clippy::expect_used)]
pub(super) fn push_d19_12(out: &mut String, value: f64) {
    let negative = value.is_sign_negative() && value != 0.0;
    let magnitude = value.abs();
    let base = format!("{magnitude:.12e}");
    let (mantissa, _) = base.split_once('e').expect("scientific form has 'e'");
    let exponent = d19_12_exponent(value);
    let sign = if negative { '-' } else { ' ' };
    let _ = write!(out, "{sign}{mantissa}e{exponent:+03}");
}

/// The `(week, TOW)` pair as it will read back after serialization.
///
/// [`push_d19_12`] keeps twelve fractional mantissa digits, so a TOW within a
/// tenth of a microsecond of the week boundary is written as exactly
/// `604800.000000000000`. The parser normalizes that into the next week, which
/// makes the stored pair a non-fixed point: encoding it once yields week `w`
/// and TOW 604800, and encoding what that parses to yields week `w + 1` and
/// TOW 0. Normalizing the pair as it will be written, rather than as stored,
/// keeps `encode(parse(encode(x))) == encode(x)`.
///
/// Normalizing can itself land back on the boundary: a stored `(9, -1e-8)`
/// normalizes to `(8, 604799.99999999)`, which the column again writes as a
/// full week. So the check repeats on the normalized value until the written
/// TOW is inside the week; two rounds are always enough because each round
/// moves the pair by a whole week.
///
/// Nothing is lost: the format cannot represent the sub-rounding difference
/// between the stored TOW and the week boundary it rounds to.
fn serialized_week_tow(week_tow: GnssWeekTow) -> GnssWeekTow {
    let written_tow = |tow_s: f64| {
        let mut written = String::new();
        push_d19_12(&mut written, tow_s);
        written.trim().parse::<f64>().ok()
    };
    let mut current = week_tow;
    for _ in 0..3 {
        let Some(rounded) = written_tow(current.tow_s) else {
            return current;
        };
        if (0.0..SECONDS_PER_WEEK).contains(&rounded) {
            return current;
        }
        match GnssWeekTow::new(current.system, current.week, rounded)
            .and_then(GnssWeekTow::normalized)
        {
            Ok(normalized) => current = normalized,
            // The week cannot carry any further (u32 range); the stored pair is
            // the best representable answer and is written as it is.
            Err(_) => return current,
        }
    }
    current
}

/// The base-10 exponent [`push_d19_12`] emits for `value` (the rounded
/// `{:.12e}` exponent). Factored out so the parser shares the exact predicate.
// invariant: Rust's fixed scientific formatter always emits a parseable exponent.
#[allow(clippy::expect_used)]
fn d19_12_exponent(value: f64) -> i32 {
    let base = format!("{:.12e}", value.abs());
    let (_, exponent) = base.split_once('e').expect("scientific form has 'e'");
    exponent.parse().expect("scientific exponent parses")
}

/// Whether `value` fits the RINEX `D19.12` fixed field. The field reserves a
/// two-digit exponent (`e+NN`); a value whose rounded base-10 exponent needs
/// three digits would widen the field to 20 columns and shift every later
/// fixed column on reparse. Such a value cannot be represented in this format,
/// so the parser rejects it to keep the parse/encode domains aligned. Real
/// broadcast values have small exponents and are always representable.
pub(super) fn d19_12_representable(value: f64) -> bool {
    d19_12_exponent(value).unsigned_abs() <= 99
}
