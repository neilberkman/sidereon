//! RINEX clock header records: labels, typed fields and time-system resolution.
//!
//! Column windows follow Table A15 of the RINEX clock 3.00 and 3.04
//! specifications. A record that does not read at its version's columns is
//! read at the other version's columns and reported as such.
//!
//! The 3.04 specification contradicts itself in these places, and this reader
//! resolves each as stated:
//!
//! - Its examples print `SYS / DCBS APPLIED`, `SYS / PCVS APPLIED` and
//!   `# OF CLK REF` (Table A17) and `STATION NAME / NUM` (Table A18) in the 3.00
//!   columns rather than its own Table A15 columns. Both are read.
//! - `TIME SYSTEM ID` defines `GLO` as "aligned to UTC(SU) + 3 hours", while
//!   `LEAP SECONDS GNSS` in the same table gives "UTC ~ GLO". RINEX 3.05
//!   section 4.1.2 states that reported GLONASS time has the same hours as UTC
//!   and not UTC + 3 h, RINEX clock 3.00 describes `GLO` as steered to UTC, and
//!   RTKLIB reads `GLO` as UTC. `GLO` is read as UTC in every version.
//! - Table A16 places the bias sigma after two blanks (columns 67-85), while
//!   the IGS combination example after Table A17 places it after one (columns
//!   66-84), where RTKLIB reads it; RTKLIB truncates the exponent of a sigma
//!   written after two blanks. Both spacings are read; a record is written
//!   after one blank unless the product's own lines use two.
//! - The Table A17 example states `LEAP SECONDS` 10 in a 1994 file, which is
//!   GPS - UTC as 3.00 defines the record, not TAI - UTC as 3.04 defines it.
//! - The Table A18 example has no `TIME SYSTEM ID` record, which 3.04 requires.
//!   Such a file is read with the 3.00 default and reported.

use crate::astro::time::model::TimeScale;
use crate::validate;

use super::{ClockEpoch, RinexClockDiagnostic, RinexClockError, RinexClockNotice};

/// Column layout of a RINEX clock file.
///
/// Version 3.04 widened receiver names to nine characters and every record to
/// 85 columns (header labels in columns 66-85). Earlier versions, including
/// 2.00 and 3.02 files in circulation, use the 80-column layout of 3.00.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ClockLayout {
    /// The 80-column layout of versions before 3.04.
    V300,
    /// The 85-column layout of version 3.04 and later.
    V304,
}

impl ClockLayout {
    /// The layout a declared format version uses.
    pub fn for_version(version: f64) -> Self {
        if version >= 3.04 {
            Self::V304
        } else {
            Self::V300
        }
    }

    /// Zero-based column where a header label starts (60 or 65).
    pub fn label_column(self) -> usize {
        match self {
            Self::V300 => 60,
            Self::V304 => 65,
        }
    }

    /// Width of the receiver or satellite name field in a data record.
    pub fn name_width(self) -> usize {
        match self {
            Self::V300 => 4,
            Self::V304 => 9,
        }
    }

    pub(super) fn other(self) -> Self {
        match self {
            Self::V300 => Self::V304,
            Self::V304 => Self::V300,
        }
    }
}

/// A time system a RINEX clock `TIME SYSTEM ID` record names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ClockTimeSystem {
    /// `GPS`: GPS system time.
    Gps,
    /// `GLO`: GLONASS time as RINEX reports it, with the hours of UTC.
    Glo,
    /// `GAL`: Galileo system time.
    Gal,
    /// `QZS`: QZSS system time.
    Qzs,
    /// `BDS`: BeiDou system time (the observation-file spelling `BDT` is also
    /// read).
    Bds,
    /// `IRN`: IRNSS system time.
    Irn,
    /// `UTC`: Coordinated Universal Time.
    Utc,
    /// `TAI`: International Atomic Time.
    Tai,
}

impl ClockTimeSystem {
    /// Read a `TIME SYSTEM ID` label.
    pub fn from_label(label: &str) -> Option<Self> {
        match label.trim() {
            "GPS" => Some(Self::Gps),
            "GLO" => Some(Self::Glo),
            "GAL" => Some(Self::Gal),
            "QZS" => Some(Self::Qzs),
            "BDS" | "BDT" => Some(Self::Bds),
            "IRN" => Some(Self::Irn),
            "UTC" => Some(Self::Utc),
            "TAI" => Some(Self::Tai),
            _ => None,
        }
    }

    /// The label this reader writes for the system.
    pub fn label(self) -> &'static str {
        match self {
            Self::Gps => "GPS",
            Self::Glo => "GLO",
            Self::Gal => "GAL",
            Self::Qzs => "QZS",
            Self::Bds => "BDS",
            Self::Irn => "IRN",
            Self::Utc => "UTC",
            Self::Tai => "TAI",
        }
    }

    /// The core time scale the system's epochs are interpreted in.
    ///
    /// `GLO` is UTC in every version, so a `23:59:60` label on a leap-second
    /// day is an epoch (see the module notes on the 3.04 wording). `IRN` has no
    /// core time scale and returns `None`.
    pub fn time_scale(self) -> Option<TimeScale> {
        match self {
            Self::Gps => Some(TimeScale::Gpst),
            Self::Gal => Some(TimeScale::Gst),
            Self::Qzs => Some(TimeScale::Qzsst),
            Self::Bds => Some(TimeScale::Bdt),
            Self::Utc | Self::Glo => Some(TimeScale::Utc),
            Self::Tai => Some(TimeScale::Tai),
            Self::Irn => None,
        }
    }

    /// The system a product built in `scale` is written with, if any.
    /// GLONASS system time (UTC(SU) + 3 h) has none: `GLO` names UTC hours.
    pub fn for_time_scale(scale: TimeScale) -> Option<Self> {
        match scale {
            TimeScale::Gpst => Some(Self::Gps),
            TimeScale::Gst => Some(Self::Gal),
            TimeScale::Qzsst => Some(Self::Qzs),
            TimeScale::Bdt => Some(Self::Bds),
            TimeScale::Utc => Some(Self::Utc),
            TimeScale::Tai => Some(Self::Tai),
            TimeScale::Glonasst
            | TimeScale::Tt
            | TimeScale::Tcg
            | TimeScale::Tdb
            | TimeScale::Tcb => None,
        }
    }
}

/// How a product's time system was established.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClockTimeSystemStatus {
    /// A `TIME SYSTEM ID` record declares it.
    Declared,
    /// No `TIME SYSTEM ID` record is present. RINEX clock 3.00 Table A15
    /// defaults a pure GLONASS file (`R` in `RINEX VERSION / TYPE`) to `GLO` and
    /// a pure Galileo file (`E`) to `GAL`; every other file takes `GPS`, the
    /// time system 3.00 Table A16 states data-record epochs in and the one
    /// RTKLIB reads clock epochs in. A 3.04 file, which requires the record,
    /// is read the same way and also reported.
    Defaulted,
    /// A `TIME SYSTEM ID` label this reader does not know.
    Unrecognized {
        /// The label as written.
        label: String,
    },
    /// Several `TIME SYSTEM ID` records naming different systems.
    Conflicting {
        /// The distinct labels in file order.
        labels: Vec<String>,
    },
    /// The product was built from series rows in a stated time scale.
    Constructed,
}

/// A typed reading of one header record's fields.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum ClockHeaderField {
    /// `RINEX VERSION / TYPE`.
    VersionType {
        /// Format version.
        version: f64,
        /// File type field as written (`C`, or `CLOCK DATA` in many files).
        file_type: String,
        /// Satellite system field as written (`G`, `M`, `GPS` or blank).
        satellite_system: String,
    },
    /// `PGM / RUN BY / DATE`.
    ProgramRunByDate {
        /// Program that created the file.
        program: String,
        /// Agency that created the file.
        run_by: String,
        /// Creation date and time as written.
        date: String,
    },
    /// `COMMENT`.
    Comment(String),
    /// `SYS / # / OBS TYPES`, a first line or a continuation line.
    ObservationTypes {
        /// Satellite system code; `None` on a continuation line.
        system: Option<char>,
        /// Declared descriptor count; `None` on a continuation line.
        count: Option<usize>,
        /// Observation descriptors on this line.
        descriptors: Vec<String>,
    },
    /// `TIME SYSTEM ID`.
    TimeSystem {
        /// The label as written, trimmed.
        label: String,
    },
    /// `LEAP SECONDS`. RINEX clock 3.00 defines it as the GPS almanac count
    /// since 1980-01-06 (GPS - UTC); 3.04 defines it as TAI - UTC.
    LeapSeconds(i64),
    /// `LEAP SECONDS GNSS` (3.04): GNSS time - UTC.
    LeapSecondsGnss(i64),
    /// `SYS / DCBS APPLIED`.
    DcbsApplied {
        /// Satellite system field.
        system: String,
        /// Program that applied the corrections.
        program: String,
        /// Source of the corrections.
        source: String,
    },
    /// `SYS / PCVS APPLIED`.
    PcvsApplied {
        /// Satellite system field.
        system: String,
        /// Program that applied the corrections.
        program: String,
        /// Source of the corrections.
        source: String,
    },
    /// `# / TYPES OF DATA`.
    TypesOfData {
        /// Declared number of data types.
        count: usize,
        /// Data type codes on the line.
        types: Vec<String>,
    },
    /// `STATION NAME / NUM`.
    StationNameNum {
        /// Receiver name.
        name: String,
        /// Receiver identifier (DOMES number).
        identifier: String,
    },
    /// `STATION CLK REF`.
    StationClockRef(String),
    /// `ANALYSIS CENTER`.
    AnalysisCenter {
        /// Three-character AC designator.
        designator: String,
        /// Full AC name.
        name: String,
    },
    /// `# OF CLK REF`.
    ClockRefCount {
        /// Number of analysis clock references.
        count: usize,
        /// Start epoch, blank when the reference applies to the whole file.
        start: Option<ClockEpoch>,
        /// Stop epoch, blank when the reference applies to the whole file.
        stop: Option<ClockEpoch>,
    },
    /// `ANALYSIS CLK REF`.
    AnalysisClockRef {
        /// Receiver or satellite name.
        name: String,
        /// Reference clock identifier.
        identifier: String,
        /// Optional a priori clock constraint, seconds.
        constraint_s: Option<f64>,
    },
    /// `# OF SOLN STA / TRF`.
    SolutionStationCount {
        /// Number of receivers.
        count: usize,
        /// Terrestrial reference frame or SINEX solution.
        frame: String,
    },
    /// `SOLN STA NAME / NUM`.
    SolutionStation {
        /// Receiver name.
        name: String,
        /// Receiver identifier (DOMES number).
        identifier: String,
        /// Geocentric X, Y, Z in millimetres.
        xyz_mm: [i64; 3],
    },
    /// `# OF SOLN SATS`.
    SolutionSatelliteCount(usize),
    /// `PRN LIST`.
    PrnList(Vec<String>),
    /// `END OF HEADER`.
    EndOfHeader,
}

/// How a header record's fields were read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ClockHeaderReading {
    /// Read at the columns of the file's version.
    Columns,
    /// Read at the columns of the other layout.
    OtherVersionColumns,
    /// Read as whitespace-separated values.
    Whitespace,
    /// The label is known but the fields do not read in any supported way.
    Uninterpreted,
    /// The label is not a RINEX clock header label.
    UnknownLabel,
}

/// One header line with its exact text and its typed reading.
#[derive(Debug, Clone, PartialEq)]
pub struct ClockHeaderRecord {
    pub(super) line: Option<usize>,
    pub(super) text: String,
    pub(super) label: String,
    pub(super) label_column: usize,
    pub(super) field: Option<ClockHeaderField>,
    pub(super) reading: ClockHeaderReading,
}

impl ClockHeaderRecord {
    /// One-based line number in the text the product was read from; `None` for
    /// a line written by an edit.
    pub fn line(&self) -> Option<usize> {
        self.line
    }

    /// The complete line without its terminator, exactly as it is written.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The header label, or for an unknown label the text in the label columns.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Zero-based column where the label starts.
    pub fn label_column(&self) -> usize {
        self.label_column
    }

    /// The text before the label.
    pub fn payload(&self) -> &str {
        self.text.get(..self.label_column).unwrap_or(&self.text)
    }

    /// The typed reading, when the fields read.
    pub fn field(&self) -> Option<&ClockHeaderField> {
        self.field.as_ref()
    }

    /// How the fields were read.
    pub fn reading(&self) -> ClockHeaderReading {
        self.reading
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Label {
    VersionType,
    ProgramRunByDate,
    Comment,
    ObservationTypes,
    TimeSystem,
    LeapSeconds,
    LeapSecondsGnss,
    DcbsApplied,
    PcvsApplied,
    TypesOfData,
    StationNameNum,
    StationClockRef,
    AnalysisCenter,
    ClockRefCount,
    AnalysisClockRef,
    SolutionStationCount,
    SolutionStation,
    SolutionSatelliteCount,
    PrnList,
    EndOfHeader,
}

/// Labels in Table A15 order; the index is the record's rank in that order.
const LABELS: [(Label, &str); 20] = [
    (Label::VersionType, "RINEX VERSION / TYPE"),
    (Label::ProgramRunByDate, "PGM / RUN BY / DATE"),
    (Label::Comment, "COMMENT"),
    (Label::ObservationTypes, "SYS / # / OBS TYPES"),
    (Label::TimeSystem, "TIME SYSTEM ID"),
    (Label::LeapSeconds, "LEAP SECONDS"),
    (Label::LeapSecondsGnss, "LEAP SECONDS GNSS"),
    (Label::DcbsApplied, "SYS / DCBS APPLIED"),
    (Label::PcvsApplied, "SYS / PCVS APPLIED"),
    (Label::TypesOfData, "# / TYPES OF DATA"),
    (Label::StationNameNum, "STATION NAME / NUM"),
    (Label::StationClockRef, "STATION CLK REF"),
    (Label::AnalysisCenter, "ANALYSIS CENTER"),
    (Label::ClockRefCount, "# OF CLK REF"),
    (Label::AnalysisClockRef, "ANALYSIS CLK REF"),
    (Label::SolutionStationCount, "# OF SOLN STA / TRF"),
    (Label::SolutionStation, "SOLN STA NAME / NUM"),
    (Label::SolutionSatelliteCount, "# OF SOLN SATS"),
    (Label::PrnList, "PRN LIST"),
    (Label::EndOfHeader, "END OF HEADER"),
];

pub(super) fn label_text(label: Label) -> &'static str {
    LABELS
        .iter()
        .find(|(candidate, _)| *candidate == label)
        .map_or("", |(_, text)| *text)
}

pub(super) fn label_rank(label: Label) -> usize {
    LABELS
        .iter()
        .position(|(candidate, _)| *candidate == label)
        .unwrap_or(LABELS.len())
}

/// Identify a header line's label by the text it ends with, returning the
/// label and the column where it starts.
pub(super) fn identify_label(content: &str) -> Option<(Label, usize)> {
    let trimmed = content.trim_end();
    LABELS
        .iter()
        .filter(|(_, text)| trimmed.ends_with(text))
        .max_by_key(|(_, text)| text.len())
        .map(|(label, text)| (*label, trimmed.len() - text.len()))
}

/// Whether a line closes the header section.
pub(super) fn is_end_of_header(content: &str) -> bool {
    content.trim_end().ends_with("END OF HEADER")
}

/// Read one header line.
pub(super) fn read_header_line(
    line: Option<usize>,
    content: &str,
    layout: Option<ClockLayout>,
) -> ClockHeaderRecord {
    let primary = layout.unwrap_or(ClockLayout::V300);
    let Some((label, label_column)) = identify_label(content) else {
        let label_column = primary.label_column().min(content.len());
        let label = if content.is_char_boundary(label_column) {
            content[label_column..].trim().to_string()
        } else {
            String::new()
        };
        return ClockHeaderRecord {
            line,
            text: content.to_string(),
            label,
            label_column,
            field: None,
            reading: ClockHeaderReading::UnknownLabel,
        };
    };
    let payload = &content[..label_column];
    let (field, reading) = if let Some(field) = read_columns(label, payload, primary) {
        (Some(field), ClockHeaderReading::Columns)
    } else if let Some(field) = read_columns(label, payload, primary.other()) {
        (Some(field), ClockHeaderReading::OtherVersionColumns)
    } else if let Some(field) = read_whitespace(label, payload) {
        (Some(field), ClockHeaderReading::Whitespace)
    } else {
        (None, ClockHeaderReading::Uninterpreted)
    };
    ClockHeaderRecord {
        line,
        text: content.to_string(),
        label: label_text(label).to_string(),
        label_column,
        field,
        reading,
    }
}

/// Split a payload at fixed columns. `None` when any character other than a
/// space falls outside every field or the payload is not ASCII.
fn columns<'a>(payload: &'a str, fields: &[(usize, usize)]) -> Option<Vec<&'a str>> {
    if !payload.is_ascii() {
        return None;
    }
    let inside = |index: usize| fields.iter().any(|&(s, e)| index >= s && index < e);
    if payload
        .bytes()
        .enumerate()
        .any(|(index, byte)| byte != b' ' && !inside(index))
    {
        return None;
    }
    Some(
        fields
            .iter()
            .map(|&(s, e)| {
                let s = s.min(payload.len());
                let e = e.min(payload.len());
                payload[s..e].trim()
            })
            .collect(),
    )
}

fn parse_usize(text: &str) -> Option<usize> {
    validate::strict_int::<usize>(text, "count").ok()
}

fn parse_i64(text: &str) -> Option<i64> {
    validate::strict_int::<i64>(text, "value").ok()
}

fn single_char(text: &str) -> Option<Option<char>> {
    let mut chars = text.chars();
    match (chars.next(), chars.next()) {
        (None, _) => Some(None),
        (Some(c), None) => Some(Some(c)),
        _ => None,
    }
}

fn epoch_fields(fields: &[&str]) -> Option<Option<ClockEpoch>> {
    if fields.iter().all(|f| f.is_empty()) {
        return Some(None);
    }
    let [year, month, day, hour, minute, second] = fields else {
        return None;
    };
    Some(Some(ClockEpoch {
        year: validate::strict_int::<i32>(year, "year").ok()?,
        month: validate::strict_int::<u8>(month, "month").ok()?,
        day: validate::strict_int::<u8>(day, "day").ok()?,
        hour: validate::strict_int::<u8>(hour, "hour").ok()?,
        minute: validate::strict_int::<u8>(minute, "minute").ok()?,
        second: validate::strict_f64(second, "second").ok()?,
    }))
}

fn repeated(start: usize, count: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..count).map(move |k| (start + 4 * k, start + 4 * k + 3))
}

fn non_empty(fields: &[&str]) -> Vec<String> {
    fields
        .iter()
        .filter(|f| !f.is_empty())
        .map(|f| f.to_string())
        .collect()
}

fn read_columns(label: Label, payload: &str, layout: ClockLayout) -> Option<ClockHeaderField> {
    let v304 = layout == ClockLayout::V304;
    match label {
        Label::VersionType => {
            let spec: &[(usize, usize)] = if v304 {
                &[(0, 9), (9, 42), (42, 65)]
            } else {
                &[(0, 9), (9, 40), (40, 60)]
            };
            let f = columns(payload, spec)?;
            Some(ClockHeaderField::VersionType {
                version: validate::strict_f64(f[0], "version").ok()?,
                file_type: f[1].to_string(),
                satellite_system: f[2].to_string(),
            })
        }
        Label::ProgramRunByDate => {
            let spec: &[(usize, usize)] = if v304 {
                &[(0, 19), (21, 40), (42, 63)]
            } else {
                &[(0, 20), (20, 40), (40, 60)]
            };
            let f = columns(payload, spec)?;
            Some(ClockHeaderField::ProgramRunByDate {
                program: f[0].to_string(),
                run_by: f[1].to_string(),
                date: f[2].to_string(),
            })
        }
        Label::Comment => Some(ClockHeaderField::Comment(payload.trim_end().to_string())),
        Label::ObservationTypes => {
            let spec: Vec<(usize, usize)> = if v304 {
                [(0, 1), (3, 6)]
                    .into_iter()
                    .chain(repeated(8, 14))
                    .collect()
            } else {
                [(0, 1), (3, 6)]
                    .into_iter()
                    .chain(repeated(7, 13))
                    .collect()
            };
            let f = columns(payload, &spec)?;
            let system = single_char(f[0])?;
            let count = if f[1].is_empty() {
                None
            } else {
                Some(parse_usize(f[1])?)
            };
            Some(ClockHeaderField::ObservationTypes {
                system,
                count,
                descriptors: non_empty(&f[2..]),
            })
        }
        Label::TimeSystem => {
            let f = columns(payload, &[(3, 6)])?;
            Some(ClockHeaderField::TimeSystem {
                label: f[0].to_string(),
            })
        }
        Label::LeapSeconds => {
            let f = columns(payload, &[(0, 6)])?;
            Some(ClockHeaderField::LeapSeconds(parse_i64(f[0])?))
        }
        Label::LeapSecondsGnss => {
            let f = columns(payload, &[(0, 6)])?;
            Some(ClockHeaderField::LeapSecondsGnss(parse_i64(f[0])?))
        }
        Label::DcbsApplied | Label::PcvsApplied => {
            let spec: &[(usize, usize)] = if v304 {
                &[(0, 1), (3, 20), (22, 65)]
            } else {
                &[(0, 1), (2, 19), (20, 60)]
            };
            let f = columns(payload, spec)?;
            let (system, program, source) = (f[0].to_string(), f[1].to_string(), f[2].to_string());
            Some(if label == Label::DcbsApplied {
                ClockHeaderField::DcbsApplied {
                    system,
                    program,
                    source,
                }
            } else {
                ClockHeaderField::PcvsApplied {
                    system,
                    program,
                    source,
                }
            })
        }
        Label::TypesOfData => {
            let spec: Vec<(usize, usize)> = std::iter::once((0, 6))
                .chain((0..5).map(|k| (10 + 6 * k, 12 + 6 * k)))
                .collect();
            let f = columns(payload, &spec)?;
            Some(ClockHeaderField::TypesOfData {
                count: parse_usize(f[0])?,
                types: non_empty(&f[1..]),
            })
        }
        Label::StationNameNum => {
            let spec: &[(usize, usize)] = if v304 {
                &[(0, 9), (10, 30)]
            } else {
                &[(0, 4), (5, 25)]
            };
            let f = columns(payload, spec)?;
            Some(ClockHeaderField::StationNameNum {
                name: f[0].to_string(),
                identifier: f[1].to_string(),
            })
        }
        Label::StationClockRef => Some(ClockHeaderField::StationClockRef(
            payload.trim().to_string(),
        )),
        Label::AnalysisCenter => {
            let spec: &[(usize, usize)] = if v304 {
                &[(0, 3), (5, 65)]
            } else {
                &[(0, 3), (5, 60)]
            };
            let f = columns(payload, spec)?;
            Some(ClockHeaderField::AnalysisCenter {
                designator: f[0].to_string(),
                name: f[1].to_string(),
            })
        }
        Label::ClockRefCount => {
            // 3.04 prints the month-to-minute group as `4(2I,1X)`, read here as
            // `4(I2,1X)`, which sums to the 65-column record.
            let spec: &[(usize, usize)] = if v304 {
                &[
                    (0, 6),
                    (7, 11),
                    (12, 14),
                    (15, 17),
                    (18, 20),
                    (21, 23),
                    (24, 34),
                    (36, 40),
                    (41, 43),
                    (44, 46),
                    (47, 49),
                    (50, 52),
                    (53, 63),
                ]
            } else {
                &[
                    (0, 6),
                    (7, 11),
                    (11, 14),
                    (14, 17),
                    (17, 20),
                    (20, 23),
                    (23, 33),
                    (34, 38),
                    (38, 41),
                    (41, 44),
                    (44, 47),
                    (47, 50),
                    (50, 60),
                ]
            };
            let f = columns(payload, spec)?;
            Some(ClockHeaderField::ClockRefCount {
                count: parse_usize(f[0])?,
                start: epoch_fields(&f[1..7])?,
                stop: epoch_fields(&f[7..13])?,
            })
        }
        Label::AnalysisClockRef => {
            let spec: &[(usize, usize)] = if v304 {
                &[(0, 9), (10, 30), (45, 64)]
            } else {
                &[(0, 4), (5, 25), (40, 59)]
            };
            let f = columns(payload, spec)?;
            let constraint_s = if f[2].is_empty() {
                None
            } else {
                Some(validate::strict_f64(f[2], "constraint").ok()?)
            };
            Some(ClockHeaderField::AnalysisClockRef {
                name: f[0].to_string(),
                identifier: f[1].to_string(),
                constraint_s,
            })
        }
        Label::SolutionStationCount => {
            let spec: &[(usize, usize)] = if v304 {
                &[(0, 6), (10, 65)]
            } else {
                &[(0, 6), (10, 60)]
            };
            let f = columns(payload, spec)?;
            Some(ClockHeaderField::SolutionStationCount {
                count: parse_usize(f[0])?,
                frame: f[1].to_string(),
            })
        }
        Label::SolutionStation => {
            // 3.00: A4,1X,A20,I11,X,I11,X,I11; 3.04: A9,1X,A20,I11,1X,I11,1X,I11.
            let spec: &[(usize, usize)] = if v304 {
                &[(0, 9), (10, 30), (30, 41), (42, 53), (54, 65)]
            } else {
                &[(0, 4), (5, 25), (25, 36), (37, 48), (49, 60)]
            };
            let f = columns(payload, spec)?;
            Some(ClockHeaderField::SolutionStation {
                name: f[0].to_string(),
                identifier: f[1].to_string(),
                xyz_mm: [parse_i64(f[2])?, parse_i64(f[3])?, parse_i64(f[4])?],
            })
        }
        Label::SolutionSatelliteCount => {
            let f = columns(payload, &[(0, 6)])?;
            Some(ClockHeaderField::SolutionSatelliteCount(parse_usize(f[0])?))
        }
        Label::PrnList => {
            let spec: Vec<(usize, usize)> = repeated(0, if v304 { 16 } else { 15 }).collect();
            let f = columns(payload, &spec)?;
            Some(ClockHeaderField::PrnList(non_empty(&f)))
        }
        Label::EndOfHeader => Some(ClockHeaderField::EndOfHeader),
    }
}

fn read_whitespace(label: Label, payload: &str) -> Option<ClockHeaderField> {
    let tokens: Vec<&str> = payload.split_whitespace().collect();
    match label {
        Label::VersionType => Some(ClockHeaderField::VersionType {
            version: validate::strict_f64(tokens.first()?, "version").ok()?,
            file_type: tokens.get(1).map_or_else(String::new, |t| t.to_string()),
            satellite_system: tokens.get(2..).map_or_else(String::new, |t| t.join(" ")),
        }),
        Label::ObservationTypes => {
            let first_is_system = tokens
                .first()
                .is_some_and(|t| t.len() == 1 && t.chars().all(|c| c.is_ascii_uppercase()));
            if first_is_system {
                Some(ClockHeaderField::ObservationTypes {
                    system: tokens[0].chars().next(),
                    count: Some(parse_usize(tokens.get(1)?)?),
                    descriptors: tokens[2..].iter().map(|t| t.to_string()).collect(),
                })
            } else {
                Some(ClockHeaderField::ObservationTypes {
                    system: None,
                    count: None,
                    descriptors: tokens.iter().map(|t| t.to_string()).collect(),
                })
            }
        }
        Label::TimeSystem => Some(ClockHeaderField::TimeSystem {
            label: tokens.first().map_or_else(String::new, |t| t.to_string()),
        }),
        Label::LeapSeconds | Label::LeapSecondsGnss => {
            let [value] = tokens.as_slice() else {
                return None;
            };
            let value = parse_i64(value)?;
            Some(if label == Label::LeapSeconds {
                ClockHeaderField::LeapSeconds(value)
            } else {
                ClockHeaderField::LeapSecondsGnss(value)
            })
        }
        Label::TypesOfData => Some(ClockHeaderField::TypesOfData {
            count: parse_usize(tokens.first()?)?,
            types: tokens[1..].iter().map(|t| t.to_string()).collect(),
        }),
        Label::StationNameNum => match tokens.as_slice() {
            [name] => Some(ClockHeaderField::StationNameNum {
                name: name.to_string(),
                identifier: String::new(),
            }),
            [name, identifier] => Some(ClockHeaderField::StationNameNum {
                name: name.to_string(),
                identifier: identifier.to_string(),
            }),
            _ => None,
        },
        Label::AnalysisCenter => {
            let designator = *tokens.first()?;
            let rest = payload.trim_start()[designator.len()..].trim();
            Some(ClockHeaderField::AnalysisCenter {
                designator: designator.to_string(),
                name: rest.to_string(),
            })
        }
        Label::ClockRefCount => {
            let count = parse_usize(tokens.first()?)?;
            let (start, stop) = match tokens.len() {
                1 => (None, None),
                7 => (epoch_fields(&tokens[1..7])?, None),
                13 => (epoch_fields(&tokens[1..7])?, epoch_fields(&tokens[7..13])?),
                _ => return None,
            };
            Some(ClockHeaderField::ClockRefCount { count, start, stop })
        }
        Label::SolutionStationCount => {
            let count_token = *tokens.first()?;
            let rest = payload.trim_start()[count_token.len()..].trim();
            Some(ClockHeaderField::SolutionStationCount {
                count: parse_usize(count_token)?,
                frame: rest.to_string(),
            })
        }
        Label::SolutionStation => {
            let [name, identifier, x, y, z] = tokens.as_slice() else {
                return None;
            };
            Some(ClockHeaderField::SolutionStation {
                name: name.to_string(),
                identifier: identifier.to_string(),
                xyz_mm: [parse_i64(x)?, parse_i64(y)?, parse_i64(z)?],
            })
        }
        Label::SolutionSatelliteCount => {
            let [count] = tokens.as_slice() else {
                return None;
            };
            Some(ClockHeaderField::SolutionSatelliteCount(parse_usize(
                count,
            )?))
        }
        Label::PrnList => Some(ClockHeaderField::PrnList(
            tokens.iter().map(|t| t.to_string()).collect(),
        )),
        Label::ProgramRunByDate
        | Label::Comment
        | Label::DcbsApplied
        | Label::PcvsApplied
        | Label::StationClockRef
        | Label::AnalysisClockRef
        | Label::EndOfHeader => None,
    }
}

/// The resolved time system of a product.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct TimeResolution {
    pub(super) system: Option<ClockTimeSystem>,
    pub(super) status: ClockTimeSystemStatus,
    pub(super) scale: Option<TimeScale>,
}

/// Everything the body reader needs from the header.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct HeaderContext {
    pub(super) version: Option<f64>,
    pub(super) layout: Option<ClockLayout>,
    pub(super) satellite_system: Option<char>,
    pub(super) time: TimeResolution,
}

impl HeaderContext {
    pub(super) fn constructed(scale: TimeScale) -> Self {
        let system = ClockTimeSystem::for_time_scale(scale);
        let layout = system.map(|system| constructed_layout(system).0);
        Self {
            version: system.map(|system| constructed_layout(system).1),
            layout,
            satellite_system: None,
            time: TimeResolution {
                system,
                status: ClockTimeSystemStatus::Constructed,
                scale: Some(scale),
            },
        }
    }
}

/// The layout and version a product built in a time system is written in:
/// 3.00 for the systems its Table A15 names, 3.04 for `QZS`, `BDS` and `IRN`.
pub(super) fn constructed_layout(system: ClockTimeSystem) -> (ClockLayout, f64) {
    match system {
        ClockTimeSystem::Gps
        | ClockTimeSystem::Glo
        | ClockTimeSystem::Gal
        | ClockTimeSystem::Utc
        | ClockTimeSystem::Tai => (ClockLayout::V300, 3.00),
        ClockTimeSystem::Qzs | ClockTimeSystem::Bds | ClockTimeSystem::Irn => {
            (ClockLayout::V304, 3.04)
        }
    }
}

/// Read every header line, derive the reading context, and report the
/// time-system errors (as diagnostics) and interpretation notices.
pub(super) fn read_header(
    lines: &[(Option<usize>, &str)],
) -> (
    Vec<ClockHeaderRecord>,
    HeaderContext,
    Vec<RinexClockDiagnostic>,
    Vec<RinexClockNotice>,
) {
    let version_field = lines.iter().find_map(|(line, content)| {
        matches!(identify_label(content), Some((Label::VersionType, _)))
            .then(|| read_header_line(*line, content, None))
    });
    let (version, satellite_system) = match version_field.as_ref().and_then(|r| r.field.as_ref()) {
        Some(ClockHeaderField::VersionType {
            version,
            satellite_system,
            ..
        }) => (
            Some(*version),
            satellite_system
                .chars()
                .next()
                .filter(|c| c.is_ascii_uppercase()),
        ),
        _ => (None, None),
    };
    let layout = version.map(ClockLayout::for_version);
    let records: Vec<ClockHeaderRecord> = lines
        .iter()
        .map(|(line, content)| read_header_line(*line, content, layout))
        .collect();

    let mut diagnostics = Vec::new();
    let time = resolve_time_system(&records, satellite_system, &mut diagnostics);

    let mut notices = Vec::new();
    for record in &records {
        let Some(line) = record.line else {
            continue;
        };
        match record.reading {
            ClockHeaderReading::Columns => {}
            ClockHeaderReading::OtherVersionColumns | ClockHeaderReading::Whitespace => {
                notices.push(RinexClockNotice::HeaderRecordNonconforming { line });
            }
            ClockHeaderReading::Uninterpreted => {
                notices.push(RinexClockNotice::HeaderRecordUninterpreted { line });
            }
            ClockHeaderReading::UnknownLabel => {
                notices.push(RinexClockNotice::HeaderRecordUnknownLabel { line });
            }
        }
    }
    if let (ClockTimeSystemStatus::Defaulted, Some(system)) = (&time.status, time.system) {
        notices.push(RinexClockNotice::TimeSystemDefaulted { system });
        if layout == Some(ClockLayout::V304) {
            notices.push(RinexClockNotice::TimeSystemMissing);
        }
    }
    if let (Some(system), None) = (time.system, time.scale) {
        notices.push(RinexClockNotice::TimeSystemWithoutScale { system });
    }

    (
        records,
        HeaderContext {
            version,
            layout,
            satellite_system,
            time,
        },
        diagnostics,
        notices,
    )
}

fn resolve_time_system(
    records: &[ClockHeaderRecord],
    satellite_system: Option<char>,
    diagnostics: &mut Vec<RinexClockDiagnostic>,
) -> TimeResolution {
    let declared: Vec<(usize, &str)> = records
        .iter()
        .filter_map(|record| match &record.field {
            Some(ClockHeaderField::TimeSystem { label }) if !label.is_empty() => {
                Some((record.line.unwrap_or(0), label.as_str()))
            }
            _ => None,
        })
        .collect();

    let Some(&(first_line, first_label)) = declared.first() else {
        // RINEX clock 3.00 Table A15: "Defaults: GPS for pure GPS files, GLO for
        // pure GLONASS files, GAL for pure Galileo files"; Table A16 states
        // data-record epochs in GPS time.
        let system = match satellite_system {
            Some('R') => ClockTimeSystem::Glo,
            Some('E') => ClockTimeSystem::Gal,
            _ => ClockTimeSystem::Gps,
        };
        return TimeResolution {
            system: Some(system),
            status: ClockTimeSystemStatus::Defaulted,
            scale: system.time_scale(),
        };
    };

    if let Some(&(line, label)) = declared.iter().find(|(_, label)| *label != first_label) {
        let mut labels: Vec<String> = Vec::new();
        for &(_, label) in &declared {
            if !labels.iter().any(|seen| seen.as_str() == label) {
                labels.push(label.to_string());
            }
        }
        diagnostics.push(RinexClockDiagnostic::new(
            line,
            RinexClockError::BadField {
                line,
                field: "time_system",
                value: label.to_string(),
            },
        ));
        return TimeResolution {
            system: None,
            status: ClockTimeSystemStatus::Conflicting { labels },
            scale: None,
        };
    }

    match ClockTimeSystem::from_label(first_label) {
        Some(system) => TimeResolution {
            system: Some(system),
            status: ClockTimeSystemStatus::Declared,
            scale: system.time_scale(),
        },
        None => {
            diagnostics.push(RinexClockDiagnostic::new(
                first_line,
                RinexClockError::BadField {
                    line: first_line,
                    field: "time_system",
                    value: first_label.to_string(),
                },
            ));
            TimeResolution {
                system: None,
                status: ClockTimeSystemStatus::Unrecognized {
                    label: first_label.to_string(),
                },
                scale: None,
            }
        }
    }
}

/// A `TIME SYSTEM ID` line in a layout's columns (`3X,A3`).
pub(super) fn render_time_system_line(system: ClockTimeSystem, layout: ClockLayout) -> String {
    render_header_line(&format!("   {}", system.label()), "TIME SYSTEM ID", layout)
}

pub(super) fn render_header_line(payload: &str, label: &str, layout: ClockLayout) -> String {
    format!("{payload:<width$}{label}", width = layout.label_column())
}

/// The header a product built from series rows is written with.
pub(super) fn render_constructed_header(
    system: ClockTimeSystem,
    satellite_system: char,
    types: &[&str],
) -> Vec<String> {
    let (layout, _) = constructed_layout(system);
    let version_payload = match layout {
        // F9.2,11X,A1,19X,A1,19X
        ClockLayout::V300 => format!("{:>9}{:11}C{:19}{satellite_system}", "3.00", "", ""),
        // F4.2,17X,A1,20X,A1,22X
        ClockLayout::V304 => format!("3.04{:17}C{:20}{satellite_system}", "", ""),
    };
    let mut types_payload = format!("{:>6}", types.len());
    for code in types {
        types_payload.push_str(&format!("{:4}{code}", ""));
    }
    vec![
        render_header_line(&version_payload, "RINEX VERSION / TYPE", layout),
        render_time_system_line(system, layout),
        render_header_line(&types_payload, "# / TYPES OF DATA", layout),
        render_header_line("", "END OF HEADER", layout),
    ]
}
