//! RINEX navigation header records.
//!
//! A header record's label is in columns 61-80; it is matched there, so a label
//! that appears in a comment or a value field is not taken for a record.

use crate::format::columns::raw_field;
use crate::ionex::GalileoNequickCoeffs;
use crate::rinex_obs::{ObsLeapSeconds, PgmRunByDate};
use crate::validate;

use super::{parse_rinex_version, IonoCorrections, KlobucharAlphaBeta, NavParseError, NavVersion};

/// One ionosphere-coefficient header record: RINEX 3/4 `IONOSPHERIC CORR`, or RINEX 2
/// `ION ALPHA` (read as `GPSA`) and `ION BETA` (read as `GPSB`).
#[derive(Debug, Clone, PartialEq)]
pub struct HeaderIonoRow {
    /// Correction type (`GPSA`, `GPSB`, `GAL`, `QZSA`, `QZSB`, `BDSA`, `BDSB`, `IRNA`,
    /// `IRNB`), trimmed.
    pub label: String,
    /// The four coefficient columns as stated, `None` when blank.
    pub values: [Option<f64>; 4],
    /// RINEX 3.04 transmission time mark (column 55), when stated.
    pub time_mark: Option<char>,
    /// RINEX 3.04 satellite identifier (columns 57-59), when stated.
    pub satellite: Option<String>,
}

/// One time system correction header record: RINEX 3 `TIME SYSTEM CORR`, or RINEX 2
/// `DELTA-UTC: A0,A1,T,W` (read as `GPUT`).
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSystemCorrection {
    /// Correction type (`GPUT`, `GLUT`, `GLGP`, `GAUT`, `GAGP`, `QZUT`, `QZGP`, `BDUT`,
    /// `SBUT`, `IRUT`, `IRGP`, ...), trimmed.
    pub code: String,
    /// `a0`, seconds.
    pub a0_s: f64,
    /// `a1`, seconds per second; `None` when blank.
    pub a1_s_s: Option<f64>,
    /// Reference time for the polynomial `T`, seconds into the week; `None` when blank.
    pub reference_time_s: Option<f64>,
    /// Reference week number `W`; `None` when blank.
    pub reference_week: Option<f64>,
    /// Source of the parameters (`S`, columns 52-56), when stated.
    pub source: Option<String>,
    /// UTC identifier (`U`, columns 58-59), when stated.
    pub utc_id: Option<String>,
}

/// The header of a RINEX navigation file.
#[derive(Debug, Clone, PartialEq)]
pub struct NavHeader {
    /// Format version.
    pub version: NavVersion,
    /// File type, column 21: `N` for navigation data; RINEX 2 `G` (GLONASS) and `H`
    /// (SBAS), and the `J`/`L` extensions.
    pub file_type: char,
    /// Satellite system, column 41, when stated.
    pub satellite_system: Option<char>,
    /// `PGM / RUN BY / DATE`.
    pub program: Option<PgmRunByDate>,
    /// `COMMENT` texts (columns 1-60, trailing blanks removed), in order.
    pub comments: Vec<String>,
    /// The ionosphere coefficient sets the rows form, as
    /// [`crate::rinex_nav::parse_iono_corrections`] reads them from the header.
    pub iono: IonoCorrections,
    /// Every ionosphere coefficient row, in order, including repeated labels.
    pub iono_rows: Vec<HeaderIonoRow>,
    /// Time system correction records, in order.
    pub time_system_corrections: Vec<TimeSystemCorrection>,
    /// `LEAP SECONDS`.
    pub leap_seconds: Option<ObsLeapSeconds>,
    /// Header records not listed above (`MERGED FILE`, `DOI`, `LICENSE OF USE`,
    /// `STATION INFORMATION`, RINEX 2 `CORR TO SYSTEM TIME`, ...), and records whose
    /// values could not be read, verbatim, in order.
    pub other_records: Vec<String>,
    /// The header's lines as read, through `END OF HEADER`; empty for a header built in
    /// code. The writer restates them when they still read as this header.
    pub text: Vec<String>,
}

impl NavHeader {
    /// A header for `version` with no records beyond the version line, written with
    /// the file type `N` and the satellite system `M` (mixed).
    pub fn new(version: NavVersion) -> Self {
        Self {
            version,
            file_type: 'N',
            satellite_system: Some('M'),
            program: None,
            comments: Vec::new(),
            iono: IonoCorrections::default(),
            iono_rows: Vec::new(),
            time_system_corrections: Vec::new(),
            leap_seconds: None,
            other_records: Vec::new(),
            text: Vec::new(),
        }
    }

    /// The header without its source text, the part the writer compares.
    pub(crate) fn without_text(&self) -> NavHeader {
        NavHeader {
            text: Vec::new(),
            ..self.clone()
        }
    }
}

/// Which header record a problem concerns, so a strict helper can report the ones it
/// reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeaderField {
    Ionosphere,
    TimeSystemCorrection,
    LeapSeconds,
}

/// A header record whose values could not be read: the record is kept verbatim and
/// its values are absent.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct HeaderIssue {
    pub(crate) line: usize,
    pub(crate) field: HeaderField,
    pub(crate) error: NavParseError,
}

/// A coefficient row and each column as read: `None` blank, `Some(Err(()))` a column
/// that does not read.
type IonoRowRead = (HeaderIonoRow, [Option<Result<f64, ()>>; 4]);
/// An [`IonoRowRead`] with its 1-based line number.
type NumberedIonoRow = (usize, IonoRowRead);

/// A read header and where the body starts.
pub(crate) struct HeaderRead {
    pub(crate) header: NavHeader,
    /// Index of the first body line.
    pub(crate) body_start: usize,
    pub(crate) issues: Vec<HeaderIssue>,
}

/// The label of a header line: columns 61-80 exactly, with the trailing blanks of a
/// label shorter than the field removed. Text past column 80 is not part of the label,
/// and a label that does not start in column 61 is not one.
pub(crate) fn header_label(line: &str) -> &str {
    line.get(60..line.len().min(80)).unwrap_or("").trim_end()
}

/// Read the header from `lines` (line terminators removed).
pub(crate) fn read_header(lines: &[&str]) -> Result<HeaderRead, NavParseError> {
    read_header_inner(lines, true)
}

/// Read the header records from `lines` as the header-only helpers
/// ([`crate::rinex_nav::parse_leap_seconds`],
/// [`crate::rinex_nav::parse_iono_corrections`]) read them: through `END OF HEADER`, or
/// every line when there is none, whether or not a supported `RINEX VERSION / TYPE`
/// record is present.
pub(crate) fn scan_header(lines: &[&str]) -> HeaderRead {
    match read_header_inner(lines, false) {
        Ok(read) => read,
        // Unreachable: without `require_version` the reader returns no error.
        Err(_) => HeaderRead {
            header: NavHeader::new(NavVersion::new(3, 4)),
            body_start: lines.len(),
            issues: Vec::new(),
        },
    }
}

fn read_header_inner(lines: &[&str], require_version: bool) -> Result<HeaderRead, NavParseError> {
    let mut version: Option<(NavVersion, char, Option<char>)> = None;
    let mut program = None;
    let mut comments = Vec::new();
    let mut iono_rows = Vec::new();
    let mut time_system_corrections = Vec::new();
    let mut leap_seconds = None;
    let mut other_records = Vec::new();
    let mut issues = Vec::new();
    let mut end = None;

    for (index, &line) in lines.iter().enumerate() {
        let line_no = index + 1;
        match header_label(line) {
            "RINEX VERSION / TYPE" => {
                let parsed = parse_rinex_version(raw_field(line, 0, 9).trim());
                let file_type = line.get(20..21).and_then(|s| s.chars().next());
                let system = line
                    .get(40..41)
                    .and_then(|s| s.chars().next())
                    .filter(|c| !c.is_whitespace());
                match (parsed, file_type) {
                    (Some(v), Some(t)) if file_type_supported(v, t) => {
                        version = Some((v, t, system));
                    }
                    _ if require_version => {
                        return Err(NavParseError::UnsupportedHeader(
                            line.trim_end().to_string(),
                        ))
                    }
                    _ => other_records.push(line.to_string()),
                }
            }
            "END OF HEADER" => {
                end = Some(index);
                break;
            }
            "PGM / RUN BY / DATE" => {
                program = Some(PgmRunByDate {
                    program: raw_field(line, 0, 20).trim().to_string(),
                    run_by: raw_field(line, 20, 40).trim().to_string(),
                    date: raw_field(line, 40, 60).trim().to_string(),
                });
            }
            "COMMENT" => comments.push(raw_field(line, 0, 60).trim_end().to_string()),
            "IONOSPHERIC CORR" => iono_rows.push((line_no, v3_iono_row(line))),
            "ION ALPHA" => iono_rows.push((line_no, v2_iono_row(line, "GPSA"))),
            "ION BETA" => iono_rows.push((line_no, v2_iono_row(line, "GPSB"))),
            "TIME SYSTEM CORR" => match v3_time_system_correction(line) {
                Ok(correction) => time_system_corrections.push(correction),
                Err(error) => {
                    issues.push(HeaderIssue {
                        line: line_no,
                        field: HeaderField::TimeSystemCorrection,
                        error,
                    });
                    other_records.push(line.to_string());
                }
            },
            "DELTA-UTC: A0,A1,T,W" => match v2_delta_utc(line) {
                Ok(correction) => time_system_corrections.push(correction),
                Err(error) => {
                    issues.push(HeaderIssue {
                        line: line_no,
                        field: HeaderField::TimeSystemCorrection,
                        error,
                    });
                    other_records.push(line.to_string());
                }
            },
            "LEAP SECONDS" => match leap_seconds_record(line) {
                Ok(leap) => leap_seconds = Some(leap),
                Err(error) => {
                    issues.push(HeaderIssue {
                        line: line_no,
                        field: HeaderField::LeapSeconds,
                        error,
                    });
                    other_records.push(line.to_string());
                }
            },
            _ => other_records.push(line.to_string()),
        }
    }

    let body_start = match end {
        Some(index) => index + 1,
        None if require_version => return Err(NavParseError::MissingHeaderEnd),
        None => lines.len(),
    };
    let (version, file_type, satellite_system) = match version {
        Some(version) => version,
        None if require_version => {
            return Err(NavParseError::UnsupportedHeader(
                "no RINEX VERSION / TYPE".to_string(),
            ))
        }
        None => (NavVersion::new(3, 4), 'N', None),
    };
    let iono = iono_sets(&iono_rows, &mut issues);
    let iono_rows = iono_rows.into_iter().map(|(_, (row, _))| row).collect();
    Ok(HeaderRead {
        header: NavHeader {
            version,
            file_type,
            satellite_system,
            program,
            comments,
            iono,
            iono_rows,
            time_system_corrections,
            leap_seconds,
            other_records,
            text: lines[..body_start]
                .iter()
                .map(|s| (*s).to_string())
                .collect(),
        },
        body_start,
        issues,
    })
}

/// The file types a version holds: `N` in every version, and RINEX 2 `G` (GLONASS),
/// `H` (SBAS) and the `J`/`L` extensions RTKLIB reads.
fn file_type_supported(version: NavVersion, file_type: char) -> bool {
    match version.major {
        2 => matches!(file_type, 'N' | 'G' | 'H' | 'J' | 'L'),
        _ => file_type == 'N',
    }
}

/// `A4,1X,4D12.4` plus the RINEX 3.04 time mark (column 55) and satellite (columns
/// 57-59). A value column that does not read is kept as `None` here; the set it
/// belongs to is refused in [`iono_sets`].
fn v3_iono_row(line: &str) -> IonoRowRead {
    let columns = [(5, 17), (17, 29), (29, 41), (41, 53)];
    iono_row(line, raw_field(line, 0, 4).trim(), columns, true)
}

/// RINEX 2 `ION ALPHA`/`ION BETA`: `2X,4D12.4`.
fn v2_iono_row(line: &str, label: &str) -> IonoRowRead {
    let columns = [(2, 14), (14, 26), (26, 38), (38, 50)];
    iono_row(line, label, columns, false)
}

fn iono_row(line: &str, label: &str, columns: [(usize, usize); 4], time_mark: bool) -> IonoRowRead {
    let mut read: [Option<Result<f64, ()>>; 4] = [None; 4];
    let mut values = [None; 4];
    for (slot, (start, end)) in columns.into_iter().enumerate() {
        let raw = raw_field(line, start, end);
        if raw.trim().is_empty() {
            continue;
        }
        let value = validate::strict_f64(raw, "ionospheric correction").map_err(|_| ());
        values[slot] = value.ok();
        read[slot] = Some(value);
    }
    let text = |start: usize, end: usize| {
        let value = raw_field(line, start, end).trim();
        (!value.is_empty()).then(|| value.to_string())
    };
    (
        HeaderIonoRow {
            label: label.to_string(),
            values,
            time_mark: if time_mark {
                text(54, 55).and_then(|s| s.chars().next())
            } else {
                None
            },
            satellite: if time_mark { text(56, 59) } else { None },
        },
        read,
    )
}

/// The coefficient sets the rows form. GPS, QZSS, BeiDou and NavIC Klobuchar rows need
/// all four columns; Galileo needs the first three, and the fourth is the disturbance
/// flags. A later row of a label replaces an earlier one, as RTKLIB `decode_navh`
/// overwrites. A row that does not read is reported and forms no set.
fn iono_sets(rows: &[NumberedIonoRow], issues: &mut Vec<HeaderIssue>) -> IonoCorrections {
    let mut gpsa = None;
    let mut gpsb = None;
    let mut qzsa = None;
    let mut qzsb = None;
    let mut bdsa = None;
    let mut bdsb = None;
    let mut irna = None;
    let mut irnb = None;
    let mut galileo = None;
    let mut galileo_flags = None;
    for (line, (row, read)) in rows {
        let bad = || HeaderIssue {
            line: *line,
            field: HeaderField::Ionosphere,
            error: NavParseError::BadHeaderField {
                field: "ionospheric correction",
            },
        };
        let klobuchar = || -> Option<[f64; 4]> {
            let mut out = [0.0; 4];
            for (slot, value) in read.iter().enumerate() {
                out[slot] = (*value)?.ok()?;
            }
            Some(out)
        };
        let target = match row.label.as_str() {
            "GPSA" => Some(&mut gpsa),
            "GPSB" => Some(&mut gpsb),
            "QZSA" => Some(&mut qzsa),
            "QZSB" => Some(&mut qzsb),
            "BDSA" => Some(&mut bdsa),
            "BDSB" => Some(&mut bdsb),
            "IRNA" => Some(&mut irna),
            "IRNB" => Some(&mut irnb),
            _ => None,
        };
        if let Some(target) = target {
            match klobuchar() {
                Some(values) => *target = Some(values),
                None => {
                    *target = None;
                    issues.push(bad());
                }
            }
            continue;
        }
        if row.label == "GAL" {
            let first_three = match (read[0], read[1], read[2]) {
                (Some(Ok(ai0)), Some(Ok(ai1)), Some(Ok(ai2))) => Some([ai0, ai1, ai2]),
                _ => None,
            };
            let flags_ok = !matches!(read[3], Some(Err(())));
            match first_three {
                Some(coeffs) if flags_ok => {
                    galileo = Some(GalileoNequickCoeffs {
                        ai0: coeffs[0],
                        ai1: coeffs[1],
                        ai2: coeffs[2],
                    });
                    galileo_flags = row.values[3];
                }
                _ => {
                    galileo = None;
                    galileo_flags = None;
                    issues.push(bad());
                }
            }
        }
    }
    let pair = |a: Option<[f64; 4]>, b: Option<[f64; 4]>| match (a, b) {
        (Some(alpha), Some(beta)) => Some(KlobucharAlphaBeta { alpha, beta }),
        _ => None,
    };
    IonoCorrections {
        gps: pair(gpsa, gpsb),
        beidou: pair(bdsa, bdsb),
        galileo,
        galileo_disturbance_flags: galileo_flags,
        qzss: pair(qzsa, qzsb),
        navic: pair(irna, irnb),
        beidou_bdgim: None,
    }
}

fn optional_header_f64(
    line: &str,
    start: usize,
    end: usize,
    field: &'static str,
) -> Result<Option<f64>, NavParseError> {
    let raw = raw_field(line, start, end);
    if raw.trim().is_empty() {
        return Ok(None);
    }
    validate::strict_f64(raw, field)
        .map(Some)
        .map_err(|_| NavParseError::BadHeaderField { field })
}

fn optional_header_text(line: &str, start: usize, end: usize) -> Option<String> {
    let value = raw_field(line, start, end).trim();
    (!value.is_empty()).then(|| value.to_string())
}

/// RINEX 3 `TIME SYSTEM CORR`: `A4,1X,D17.10,D16.9,I7,I5,1X,A5,1X,I2,1X`, the columns
/// RTKLIB `decode_navh` reads.
fn v3_time_system_correction(line: &str) -> Result<TimeSystemCorrection, NavParseError> {
    const FIELD: &str = "time system correction";
    let a0_s = optional_header_f64(line, 5, 22, FIELD)?
        .ok_or(NavParseError::BadHeaderField { field: FIELD })?;
    Ok(TimeSystemCorrection {
        code: raw_field(line, 0, 4).trim().to_string(),
        a0_s,
        a1_s_s: optional_header_f64(line, 22, 38, FIELD)?,
        reference_time_s: optional_header_f64(line, 38, 45, FIELD)?,
        reference_week: optional_header_f64(line, 45, 50, FIELD)?,
        source: optional_header_text(line, 51, 56),
        utc_id: optional_header_text(line, 57, 59),
    })
}

/// RINEX 2 `DELTA-UTC: A0,A1,T,W`: `3X,2D19.12,2I9`, read as `GPUT`.
fn v2_delta_utc(line: &str) -> Result<TimeSystemCorrection, NavParseError> {
    const FIELD: &str = "time system correction";
    let a0_s = optional_header_f64(line, 3, 22, FIELD)?
        .ok_or(NavParseError::BadHeaderField { field: FIELD })?;
    Ok(TimeSystemCorrection {
        code: "GPUT".to_string(),
        a0_s,
        a1_s_s: optional_header_f64(line, 22, 41, FIELD)?,
        reference_time_s: optional_header_f64(line, 41, 50, FIELD)?,
        reference_week: optional_header_f64(line, 50, 59, FIELD)?,
        source: None,
        utc_id: None,
    })
}

/// `LEAP SECONDS`: `4I6,A3`, current, future, week, day and time system.
fn leap_seconds_record(line: &str) -> Result<ObsLeapSeconds, NavParseError> {
    const FIELD: &str = "leap seconds";
    let int = |start: usize, end: usize| -> Result<Option<i64>, NavParseError> {
        let raw = raw_field(line, start, end);
        if raw.trim().is_empty() {
            return Ok(None);
        }
        let value = validate::strict_f64(raw, FIELD)
            .map_err(|_| NavParseError::BadHeaderField { field: FIELD })?;
        if value.trunc() != value || value.abs() > 1.0e15 {
            return Err(NavParseError::BadHeaderField { field: FIELD });
        }
        Ok(Some(value as i64))
    };
    let current = int(0, 6)?.ok_or(NavParseError::BadHeaderField { field: FIELD })?;
    Ok(ObsLeapSeconds {
        current,
        delta_future: int(6, 12)?,
        week: int(12, 18)?,
        day: int(18, 24)?,
        time_system: optional_header_text(line, 24, 27),
    })
}

/// Rebuild the rows' read results for a header built in code: every stated value reads.
fn rows_as_read(rows: &[HeaderIonoRow]) -> Vec<NumberedIonoRow> {
    rows.iter()
        .map(|row| {
            let read = row.values.map(|value| value.map(Ok));
            (0, (row.clone(), read))
        })
        .collect()
}

/// The ionosphere sets `rows` form, for a header built in code.
pub(crate) fn iono_from_rows(rows: &[HeaderIonoRow]) -> IonoCorrections {
    let mut issues = Vec::new();
    iono_sets(&rows_as_read(rows), &mut issues)
}
