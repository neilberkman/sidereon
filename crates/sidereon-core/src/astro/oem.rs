//! CCSDS Orbit Ephemeris Message (OEM) KVN and XML reader/writer.
//!
//! OEM date/time values are carried as raw strings. The reader does not resolve
//! time systems or normalize epochs, so state-vector lines round-trip through the
//! canonical IR without calendar rewriting.
//!
//! The readers retain every item of CCSDS 502.0-B-3 tables 5-2 to 5-4: the
//! optional header `CLASSIFICATION` and `MESSAGE_ID`, `REF_FRAME_EPOCH`, and the
//! comments of the header, of each metadata block, and of each segment's
//! ephemeris and covariance data, and the writers write them back. Header and
//! metadata comments are written at the start of their block (7.8.9); an
//! ephemeris or covariance comment keeps its position among the ephemeris lines
//! or covariance matrices.
//!
//! The KVN header is read from the lines before the first `META_START`, and
//! each metadata block from its own `META_START`/`META_STOP` pair, so a keyword
//! of one block is never taken for another. A keyword repeated in one block
//! with a different value is refused by name, and so is a keyword or element
//! the tables do not define that carries a value (5.2.2.2, 5.2.3.2). Header and
//! metadata values are kept verbatim; the OEM defines no units for them, and
//! ephemeris and covariance lines carry none (7.7.2).
//!
//! A malformed KVN ephemeris data line is skipped and reported in
//! [`Oem::skipped_states`] with its line number, text and reason; the XML
//! reader refuses a malformed state vector instead. A `keyword = value` line
//! among the ephemeris data lines is not a data line and is refused by name.

use crate::astro::covariance::{Covariance6, Covariance6Error};
use crate::astro::ndm::{
    self, covariance6_unit, mirror_lower_triangle6, read_lower_triangle6, FieldMap, KvnLine,
    UnitMismatch, COVARIANCE6_KEYS,
};
use crate::astro::xml;
use crate::format::fmtnum::fmt_num;
use crate::format::tokens::Tokenizer;
use crate::validate;
use roxmltree::{Document, Node};
use std::fmt;

pub use crate::astro::ndm::TextIssue;

const COMMENT: &str = "COMMENT";
const META_START: &str = "META_START";
const META_STOP: &str = "META_STOP";
const COVARIANCE_START: &str = "COVARIANCE_START";
const COVARIANCE_STOP: &str = "COVARIANCE_STOP";
const OEM_VERSION_KEY: &str = "CCSDS_OEM_VERS";
const COV_REF_FRAME: &str = "COV_REF_FRAME";

/// Header keywords, CCSDS 502.0-B-3 table 5-2 (`COMMENT` is handled apart).
const HEADER_KEYS: &[&str] = &[
    OEM_VERSION_KEY,
    "CLASSIFICATION",
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
];
/// Metadata keywords, table 5-3.
const METADATA_KEYS: &[&str] = &[
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
    "START_TIME",
    "USEABLE_START_TIME",
    "USEABLE_STOP_TIME",
    "STOP_TIME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE",
];

const STATE_NUMBER_KEYS: [&str; 9] = [
    "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT", "X_DDOT", "Y_DDOT", "Z_DDOT",
];

/// Canonical, format-agnostic OEM container.
#[derive(Debug, Clone, PartialEq)]
pub struct Oem {
    /// KVN readers copy `CCSDS_OEM_VERS`, while XML readers prefer the trimmed
    /// `<oem version>` attribute and otherwise use the `CCSDS_OEM_VERS`
    /// element. An empty KVN value is rejected, and both encoders write the
    /// stored version.
    pub ccsds_oem_vers: String,
    /// Header comments, written after `CCSDS_OEM_VERS` (502.0-B-3 7.8.9).
    pub comments: Vec<String>,
    /// Optional header `CLASSIFICATION` text (table 5-2), written only when
    /// present.
    pub classification: Option<String>,
    /// Optional `CREATION_DATE` header text copied by both readers and emitted
    /// by both encoders; a missing value becomes an empty KVN header value or
    /// XML element.
    pub creation_date: Option<String>,
    /// Optional `ORIGINATOR` header text copied by both readers and emitted by
    /// both encoders; a missing value becomes an empty KVN header value or XML
    /// element.
    pub originator: Option<String>,
    /// Optional header `MESSAGE_ID` text (table 5-2), written only when
    /// present.
    pub message_id: Option<String>,
    /// Segments occur in KVN `META_START` order or XML document order. Both
    /// encoders iterate this vector in order; a message with no segment returns
    /// [`OemError::Field`].
    pub segments: Vec<OemSegment>,
    /// KVN ephemeris data lines the forgiving reader skipped, in input order,
    /// each with its line number, text and reason. The encoders do not write
    /// them, and the XML reader leaves this empty because it refuses a
    /// malformed state vector.
    pub skipped_states: Vec<OemSkippedState>,
}

/// One KVN ephemeris data line the forgiving OEM reader skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OemSkippedState {
    /// One-based line number in the input.
    pub line: usize,
    /// Zero-based index of the segment whose data section holds the line.
    pub segment: usize,
    /// The line text with surrounding whitespace removed.
    pub text: String,
    /// Why the line was skipped.
    pub reason: OemStateLineError,
}

/// Why an OEM KVN ephemeris data line was skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OemStateLineError {
    /// The line holds a number of items other than 7 (epoch, position,
    /// velocity) or 10 (with acceleration) (502.0-B-3 5.2.4.1-5.2.4.2).
    ItemCount(usize),
    /// A numeric item failed validation.
    InvalidField {
        /// The item: `X`, `Y`, `Z`, `X_DOT`, ... `Z_DDOT`.
        field: &'static str,
        /// The validation failure category.
        kind: OemInputErrorKind,
    },
}

/// A comment among the ephemeris lines or covariance matrices of a segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OemComment {
    /// The number of items of its list that precede the comment: ephemeris
    /// lines for [`OemSegment::data_comments`], covariance matrices for
    /// [`OemSegment::covariance_comments`].
    pub position: usize,
    /// The comment text.
    pub text: String,
}

/// One OEM metadata/data segment.
#[derive(Debug, Clone, PartialEq)]
pub struct OemSegment {
    /// Metadata read between KVN markers or from the XML `<metadata>` child.
    /// Encoders write it before the segment data.
    pub metadata: OemMetadata,
    /// Comments of the ephemeris data, each at its position among the state
    /// lines, in source order.
    pub data_comments: Vec<OemComment>,
    /// State vectors are collected in input order; malformed KVN lines are
    /// skipped and reported in [`Oem::skipped_states`], while XML state-vector
    /// errors fail parsing. Both encoders write states before covariances.
    pub states: Vec<OemState>,
    /// Comments of the covariance data, each at its position among the
    /// covariance matrices, in source order. KVN writes them in the covariance
    /// section; XML writes a comment before matrix `n` inside that matrix and
    /// one after the last matrix after it.
    pub covariance_comments: Vec<OemComment>,
    /// Covariance matrices in input order, from the KVN covariance section or
    /// XML `<covarianceMatrix>` elements. Both encoders write them after the
    /// states.
    pub covariances: Vec<OemCovariance>,
}

/// OEM segment metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct OemMetadata {
    /// Metadata comments, written after `META_START` (502.0-B-3 7.8.9).
    pub comments: Vec<String>,
    /// Required `OBJECT_NAME` text copied by `parse_metadata` and emitted under
    /// the same key or element by both encoders.
    pub object_name: String,
    /// Required `OBJECT_ID` text copied by `parse_metadata` and emitted under
    /// the same key or element by both encoders.
    pub object_id: String,
    /// Required `CENTER_NAME` text copied by `parse_metadata` and emitted under
    /// the same key or element by both encoders.
    pub center_name: String,
    /// Required `REF_FRAME` text copied by `parse_metadata` and emitted under
    /// the same key or element by both encoders.
    pub ref_frame: String,
    /// Optional `REF_FRAME_EPOCH` text (table 5-3), retained without date
    /// conversion and written only when present.
    pub ref_frame_epoch: Option<String>,
    /// Required `TIME_SYSTEM` text copied by `parse_metadata` and emitted under
    /// the same key or element by both encoders.
    pub time_system: String,
    /// Required `START_TIME` text retained without date conversion and emitted
    /// under the same key or element by both encoders.
    pub start_time: String,
    /// Required `STOP_TIME` text retained without date conversion and emitted
    /// under the same key or element by both encoders.
    pub stop_time: String,
    /// Optional `USEABLE_START_TIME` text; absent input yields `None`, and both
    /// encoders omit the field when it is `None`.
    pub useable_start_time: Option<String>,
    /// Optional `USEABLE_STOP_TIME` text; absent input yields `None`, and both
    /// encoders omit the field when it is `None`.
    pub useable_stop_time: Option<String>,
    /// Optional `INTERPOLATION` text; absent input yields `None`, and both
    /// encoders omit the field when it is `None`.
    pub interpolation: Option<String>,
    /// Optional `INTERPOLATION_DEGREE` parsed as a strict `u32`; an invalid
    /// integer becomes an [`OemError::InvalidField`], and encoders write it in
    /// decimal form only when present.
    pub interpolation_degree: Option<u32>,
}

/// One OEM Cartesian state sample.
#[derive(Debug, Clone, PartialEq)]
pub struct OemState {
    /// The first KVN token or required XML `EPOCH` text, retained without time-
    /// system resolution or normalization and written back by both encoders.
    pub epoch: String,
    /// Strict numeric `X`, `Y`, and `Z` values stored in that array order and
    /// encoded back under those fields through `fmt_num`.
    pub position_km: [f64; 3],
    /// Strict numeric `X_DOT`, `Y_DOT`, and `Z_DOT` values stored in that array
    /// order and encoded back under those fields through `fmt_num`.
    pub velocity_km_s: [f64; 3],
    /// KVN supplies acceleration only for a ten-token state line, while XML
    /// supplies it only when all three acceleration elements are present. A
    /// partial XML triple is an error, and encoders omit these fields for
    /// `None`.
    pub acceleration_km_s2: Option<[f64; 3]>,
}

/// One OEM covariance matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct OemCovariance {
    /// Required `EPOCH` text retained in the covariance block and emitted under
    /// the same field or element without date conversion.
    pub epoch: String,
    /// Optional `COV_REF_FRAME` text retained when present and omitted by both
    /// encoders when absent.
    pub cov_ref_frame: Option<String>,
    /// The 21 lower-triangle values exactly as read, in keyword order `CX_X`,
    /// `CY_X`, `CY_Y` ... `CZ_DOT_Z_DOT` (row by row). Each is finite; the
    /// matrix is not checked for positive semidefiniteness on read, since one
    /// printed to a few digits, as the standard's own example is, can fall
    /// short of it only through that rounding. The KVN encoder writes the
    /// values as six rows of one to six (502.0-B-3 5.2.5.4); the XML encoder
    /// writes one element per value. [`OemCovariance::to_covariance6`]
    /// validates the matrix for a consumer that needs a covariance.
    pub lower_triangle: [f64; 21],
}

impl OemCovariance {
    /// The symmetric matrix as a validated [`Covariance6`]: refused when it is
    /// not symmetric positive semidefinite within the [`Covariance6`]
    /// tolerance.
    pub fn to_covariance6(&self) -> Result<Covariance6, Covariance6Error> {
        Covariance6::try_from_matrix(mirror_lower_triangle6(&self.lower_triangle))
    }
}

/// Failure modes of the OEM readers and writers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OemError {
    /// A required field was absent from the message.
    MissingField(&'static str),
    /// A decoded scalar field failed validation.
    InvalidField {
        /// Static source-field label retained from validation and included in
        /// the error's display text.
        field: &'static str,
        /// Validation category mapped from the shared [`validate::FieldError`].
        kind: OemInputErrorKind,
    },
    /// A structural or XML-level error.
    Field(String),
    /// A keyword occurred more than once in one block with different values.
    DuplicateField {
        /// The repeated keyword.
        field: String,
        /// The value of its first occurrence.
        first: String,
        /// The later value that differs.
        second: String,
    },
    /// An XML `units` attribute contradicts the unit CCSDS 502.0-B-3 gives the
    /// element (8.10.11, tables 8-6 and 8-7).
    UnitMismatch {
        /// The element whose value carried the unit.
        field: String,
        /// The stated unit.
        unit: String,
        /// The table unit, or `None` for a dimensionless or text element.
        expected: Option<&'static str>,
    },
    /// An XML document holds more than one OEM.
    MultipleMessages {
        /// The number of OEM messages in the document.
        count: usize,
    },
    /// A KVN keyword or XML element that tables 5-2 to 5-4 do not define at
    /// that position and that carries a value (5.2.2.2, 5.2.3.2). XML elements
    /// are named with their parent element.
    UnknownField(String),
    /// A KVN header or metadata line that is not blank, a comment, or a
    /// `keyword = value` assignment (502.0-B-3 7.4.1.1).
    MalformedLine {
        /// One-based line number.
        line: usize,
        /// The trimmed line text.
        text: String,
    },
    /// A writer cannot write a text value so that it reads back unchanged.
    UnwritableText {
        /// The keyword, `COMMENT`, or `EPOCH` of an ephemeris line.
        field: String,
        /// The text that cannot be written.
        value: String,
        /// Why the text would not read back unchanged.
        issue: TextIssue,
    },
}

/// OEM boundary-validation failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OemInputErrorKind {
    /// A required field was absent.
    Missing,
    /// A floating-point field was NaN or infinite.
    NonFinite,
    /// A floating-point field could not be parsed.
    FloatParse,
    /// An integer field could not be parsed.
    IntParse,
    /// A positive physical field was zero or negative.
    NotPositive,
    /// A non-negative physical field was negative.
    Negative,
    /// A finite numeric field was outside its accepted range.
    OutOfRange,
    /// A civil date field was out of range.
    InvalidCivilDate,
    /// A civil time field was out of range.
    InvalidCivilTime,
}

impl fmt::Display for OemInputErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Missing => "missing",
            Self::NonFinite => "not finite",
            Self::FloatParse => "invalid float",
            Self::IntParse => "invalid integer",
            Self::NotPositive => "not positive",
            Self::Negative => "negative",
            Self::OutOfRange => "out of range",
            Self::InvalidCivilDate => "invalid civil date",
            Self::InvalidCivilTime => "invalid civil time",
        };
        f.write_str(label)
    }
}

impl From<&validate::FieldError> for OemInputErrorKind {
    fn from(error: &validate::FieldError) -> Self {
        match error {
            validate::FieldError::Missing { .. } => Self::Missing,
            validate::FieldError::NonFinite { .. } => Self::NonFinite,
            validate::FieldError::FloatParse { .. } => Self::FloatParse,
            validate::FieldError::IntParse { .. } => Self::IntParse,
            validate::FieldError::NotPositive { .. } => Self::NotPositive,
            validate::FieldError::Negative { .. } => Self::Negative,
            validate::FieldError::OutOfRange { .. } => Self::OutOfRange,
            validate::FieldError::InvalidCivilDate { .. } => Self::InvalidCivilDate,
            validate::FieldError::InvalidCivilTime { .. } => Self::InvalidCivilTime,
        }
    }
}

impl fmt::Display for OemError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OemError::MissingField(name) => write!(f, "OEM missing required field {name}"),
            OemError::InvalidField { field, kind } => {
                write!(f, "invalid OEM field {field}: {kind}")
            }
            OemError::Field(msg) => write!(f, "OEM field error: {msg}"),
            OemError::DuplicateField {
                field,
                first,
                second,
            } => write!(
                f,
                "OEM keyword {field} occurs with different values {first:?} and {second:?}"
            ),
            OemError::UnitMismatch {
                field,
                unit,
                expected,
            } => write!(
                f,
                "OEM element {field} states unit [{unit}], expected {}",
                ndm::expected_unit_label(*expected)
            ),
            OemError::MultipleMessages { count } => {
                write!(f, "XML holds {count} OEM messages; the reader reads one")
            }
            OemError::UnknownField(name) => write!(f, "OEM has no keyword {name}"),
            OemError::MalformedLine { line, text } => write!(
                f,
                "OEM line {line} is not a comment or keyword assignment: {text:?}"
            ),
            OemError::UnwritableText {
                field,
                value,
                issue,
            } => write!(f, "OEM {field} value {value:?} {issue}"),
        }
    }
}

impl std::error::Error for OemError {}

/// Parse a CCSDS OEM in KVN encoding into an [`Oem`].
///
/// Lines end at CR, LF, CR LF or LF CR (502.0-B-3 7.3.7). The header is the
/// part before the first `META_START`; header and metadata lines must be
/// blank, comments or assignments (7.4.1.1). A covariance section holds one or
/// more matrices, each opened by `EPOCH`, optionally followed by
/// `COV_REF_FRAME`, and given as six rows of one to six lower-triangle values
/// (5.2.5, 7.4.1.3). A matrix given with the OPM/OMM keywords (`CX_X = ...`),
/// which earlier versions of this writer produced, is also read.
pub fn parse_kvn(text: &str) -> Result<Oem, OemError> {
    let lines = numbered_lines(text);
    let first_segment = lines
        .iter()
        .position(|(_, line)| line == META_START)
        .unwrap_or(lines.len());
    let (header_map, header_comments) = kvn_block(&lines[..first_segment], HEADER_KEYS)?;
    let ccsds_oem_vers = header_map
        .get(OEM_VERSION_KEY)
        .ok_or(OemError::MissingField(OEM_VERSION_KEY))?
        .to_string();

    let mut skipped_states = Vec::new();
    let mut segments = Vec::new();
    let mut idx = first_segment;
    while idx < lines.len() {
        let (segment, next_idx) =
            parse_kvn_segment(&lines, idx, segments.len(), &mut skipped_states)?;
        segments.push(segment);
        idx = next_idx;
    }

    if segments.is_empty() {
        return Err(OemError::Field("OEM contains no segment".to_string()));
    }

    Ok(Oem {
        ccsds_oem_vers,
        comments: header_comments,
        classification: opt_text(&header_map, "CLASSIFICATION"),
        creation_date: opt_text(&header_map, "CREATION_DATE"),
        originator: opt_text(&header_map, "ORIGINATOR"),
        message_id: opt_text(&header_map, "MESSAGE_ID"),
        segments,
        skipped_states,
    })
}

/// Read a KVN header or metadata block: its comments and its assignments, with
/// every value verbatim. A keyword outside `keys` that carries a value, a
/// conflicting repeat, and a line that is neither blank, a comment, nor an
/// assignment are refused by name.
fn kvn_block(
    lines: &[(usize, String)],
    keys: &[&str],
) -> Result<(FieldMap, Vec<String>), OemError> {
    let mut pairs = Vec::new();
    let mut comments = Vec::new();
    for (line_no, line) in lines {
        match ndm::classify(line) {
            KvnLine::Blank => {}
            KvnLine::Comment(comment) => comments.push(comment.to_string()),
            KvnLine::Assignment { key, value } => {
                if keys.contains(&key) {
                    pairs.push((key.to_string(), value.to_string()));
                } else if !value.is_empty() {
                    return Err(OemError::UnknownField(key.to_string()));
                }
            }
            KvnLine::Other(other) => {
                return Err(OemError::MalformedLine {
                    line: *line_no,
                    text: other.to_string(),
                })
            }
        }
    }
    let map = FieldMap::from_pairs(pairs);
    reject_conflict(&map)?;
    Ok((map, comments))
}

fn parse_kvn_segment(
    lines: &[(usize, String)],
    start_idx: usize,
    segment_index: usize,
    skipped_states: &mut Vec<OemSkippedState>,
) -> Result<(OemSegment, usize), OemError> {
    let mut idx = start_idx + 1;
    let metadata_start = idx;
    while idx < lines.len() {
        if lines[idx].1 == META_STOP {
            break;
        }
        idx += 1;
    }
    if idx >= lines.len() {
        return Err(OemError::Field("META_START without META_STOP".to_string()));
    }

    let (metadata_map, metadata_comments) = kvn_block(&lines[metadata_start..idx], METADATA_KEYS)?;
    let metadata = parse_metadata(&metadata_map, metadata_comments)?;
    idx += 1;

    let mut data_comments = Vec::new();
    let mut states = Vec::new();
    let mut covariance_comments = Vec::new();
    let mut covariances = Vec::new();
    while idx < lines.len() {
        let (line_no, line) = &lines[idx];

        if line == META_START {
            break;
        }
        match ndm::classify(line) {
            KvnLine::Blank => {
                idx += 1;
                continue;
            }
            KvnLine::Comment(comment) => {
                data_comments.push(OemComment {
                    position: states.len(),
                    text: comment.to_string(),
                });
                idx += 1;
                continue;
            }
            // An ephemeris data line holds no `=` (502.0-B-3 5.2.4.1), so an
            // assignment here is a keyword the data section does not define,
            // refused by name like one in the header or metadata; a blank one
            // holds nothing to keep.
            KvnLine::Assignment { key, value } => {
                if value.is_empty() {
                    idx += 1;
                    continue;
                }
                return Err(OemError::UnknownField(key.to_string()));
            }
            KvnLine::Other(_) => {}
        }
        if line == COVARIANCE_START {
            idx = parse_kvn_covariance(lines, idx, &mut covariances, &mut covariance_comments)?;
            continue;
        }
        if line == COVARIANCE_STOP {
            return Err(OemError::Field(
                "COVARIANCE_STOP without COVARIANCE_START".to_string(),
            ));
        }

        match parse_state_line(line) {
            Ok(state) => states.push(state),
            Err(reason) => skipped_states.push(OemSkippedState {
                line: *line_no,
                segment: segment_index,
                text: line.clone(),
                reason,
            }),
        }
        idx += 1;
    }

    Ok((
        OemSegment {
            metadata,
            data_comments,
            states,
            covariance_comments,
            covariances,
        },
        idx,
    ))
}

/// Read one covariance section (502.0-B-3 5.2.5, table 5-4) into
/// `covariances`, returning the index after its `COVARIANCE_STOP`. A comment
/// takes as its position the number of matrices completed before it; a
/// row-form matrix completes with its sixth row, a keyword-form matrix at the
/// next `EPOCH` or `COVARIANCE_STOP`.
fn parse_kvn_covariance(
    lines: &[(usize, String)],
    start_idx: usize,
    covariances: &mut Vec<OemCovariance>,
    comments: &mut Vec<OemComment>,
) -> Result<usize, OemError> {
    let mut idx = start_idx + 1;
    let mut current: Option<CovarianceLines> = None;
    while idx < lines.len() {
        let (line_no, line) = &lines[idx];
        if line == COVARIANCE_STOP {
            if let Some(matrix) = current.take() {
                covariances.push(matrix.finish()?);
            }
            return Ok(idx + 1);
        }
        match ndm::classify(line) {
            KvnLine::Blank => {}
            KvnLine::Comment(comment) => comments.push(OemComment {
                position: covariances.len(),
                text: comment.to_string(),
            }),
            KvnLine::Assignment {
                key: "EPOCH",
                value,
            } => {
                if let Some(matrix) = current.take() {
                    covariances.push(matrix.finish()?);
                }
                if value.is_empty() {
                    return Err(OemError::MissingField("EPOCH"));
                }
                current = Some(CovarianceLines::new(value));
            }
            KvnLine::Assignment { key, value } => match current.as_mut() {
                Some(matrix) => matrix.keyword(key, value)?,
                None if value.is_empty() && !is_covariance_keyword(key) => {}
                None => {
                    return Err(OemError::Field(format!(
                        "line {line_no}: {key} precedes the covariance EPOCH"
                    )))
                }
            },
            KvnLine::Other(row) => {
                let matrix = current.as_mut().ok_or_else(|| {
                    OemError::Field(format!(
                        "line {line_no}: covariance row precedes the covariance EPOCH"
                    ))
                })?;
                matrix.row(*line_no, row)?;
                // The sixth row completes the matrix, so a comment after it
                // follows the matrix.
                if matrix.rows == 6 {
                    if let Some(complete) = current.take() {
                        covariances.push(complete.finish()?);
                    }
                }
            }
        }
        idx += 1;
    }

    Err(OemError::Field(
        "COVARIANCE_START without COVARIANCE_STOP".to_string(),
    ))
}

fn is_covariance_keyword(key: &str) -> bool {
    key == COV_REF_FRAME || COVARIANCE6_KEYS.contains(&key)
}

/// The lines of one covariance matrix in a KVN covariance section.
struct CovarianceLines {
    epoch: String,
    cov_ref_frame: Option<String>,
    /// Lower-triangle values read from data rows, in row order.
    values: Vec<f64>,
    rows: usize,
    /// `CX_X = ...` assignments of the keyword form.
    keywords: Vec<(String, String)>,
}

impl CovarianceLines {
    fn new(epoch: &str) -> Self {
        Self {
            epoch: epoch.to_string(),
            cov_ref_frame: None,
            values: Vec::with_capacity(21),
            rows: 0,
            keywords: Vec::new(),
        }
    }

    fn keyword(&mut self, key: &str, value: &str) -> Result<(), OemError> {
        if key == COV_REF_FRAME {
            if let Some(first) = &self.cov_ref_frame {
                if first != value {
                    return Err(OemError::DuplicateField {
                        field: key.to_string(),
                        first: first.clone(),
                        second: value.to_string(),
                    });
                }
                return Ok(());
            }
            self.cov_ref_frame = Some(value.to_string());
            return Ok(());
        }
        if !COVARIANCE6_KEYS.contains(&key) {
            if value.is_empty() {
                return Ok(());
            }
            return Err(OemError::UnknownField(key.to_string()));
        }
        if self.rows > 0 {
            return Err(OemError::Field(format!(
                "{key} mixes keyword and row forms in one covariance matrix"
            )));
        }
        self.keywords.push((key.to_string(), value.to_string()));
        Ok(())
    }

    fn row(&mut self, line_no: usize, row: &str) -> Result<(), OemError> {
        if !self.keywords.is_empty() {
            return Err(OemError::Field(format!(
                "line {line_no}: covariance row mixes keyword and row forms in one covariance matrix"
            )));
        }
        let expected = self.rows + 1;
        let mut tokenizer = Tokenizer::new(row);
        let mut tokens = Vec::new();
        while let Some(token) = tokenizer.next_str() {
            tokens.push(token);
        }
        if expected > 6 || tokens.len() != expected {
            return Err(OemError::Field(format!(
                "line {line_no}: covariance row {expected} holds {} values; row n of the lower triangle holds n values for n = 1 to 6 (CCSDS 502.0-B-3 7.4.1.3)",
                tokens.len()
            )));
        }
        for token in tokens {
            let key = COVARIANCE6_KEYS[self.values.len()];
            self.values
                .push(validate::strict_f64(token, key).map_err(map_oem_field_error)?);
        }
        self.rows = expected;
        Ok(())
    }

    fn finish(self) -> Result<OemCovariance, OemError> {
        let cov_ref_frame = self.cov_ref_frame.filter(|frame| !frame.is_empty());
        let lower_triangle = if self.keywords.is_empty() {
            self.values.try_into().map_err(|values: Vec<f64>| {
                OemError::Field(format!(
                    "covariance matrix at EPOCH {} holds {} of its 21 values",
                    self.epoch,
                    values.len()
                ))
            })?
        } else {
            let map = FieldMap::from_pairs(self.keywords);
            reject_conflict(&map)?;
            read_lower_triangle6(&map).map_err(map_oem_field_error)?
        };
        Ok(OemCovariance {
            epoch: self.epoch,
            cov_ref_frame,
            lower_triangle,
        })
    }
}

/// Encode an [`Oem`] as CCSDS OEM KVN.
///
/// Header and metadata comments are written at the start of their block, and
/// ephemeris and covariance comments at their positions. Each segment's
/// covariance matrices are written in one covariance section (502.0-B-3
/// 5.2.5.1), each as `EPOCH`, `COV_REF_FRAME` when present, and six rows of
/// lower-triangle values. Text that would not read back unchanged is refused
/// with [`OemError::UnwritableText`]: a line break, whitespace the reader
/// trims, an empty required value, whitespace inside an ephemeris epoch, or a
/// comment positioned past the end of its list or out of order. A non-finite
/// number is refused with [`OemError::InvalidField`].
pub fn encode_kvn(oem: &Oem) -> Result<String, OemError> {
    let mut out = KvnWriter::default();
    out.required(OEM_VERSION_KEY, &oem.ccsds_oem_vers)?;
    out.comments(&oem.comments)?;
    out.present("CLASSIFICATION", oem.classification.as_deref())?;
    out.optional("CREATION_DATE", oem.creation_date.as_deref())?;
    out.optional("ORIGINATOR", oem.originator.as_deref())?;
    out.present("MESSAGE_ID", oem.message_id.as_deref())?;

    for segment in &oem.segments {
        out.raw(META_START);
        let metadata = &segment.metadata;
        out.comments(&metadata.comments)?;
        out.required("OBJECT_NAME", &metadata.object_name)?;
        out.required("OBJECT_ID", &metadata.object_id)?;
        out.required("CENTER_NAME", &metadata.center_name)?;
        out.required("REF_FRAME", &metadata.ref_frame)?;
        out.present("REF_FRAME_EPOCH", metadata.ref_frame_epoch.as_deref())?;
        out.required("TIME_SYSTEM", &metadata.time_system)?;
        out.required("START_TIME", &metadata.start_time)?;
        out.required("STOP_TIME", &metadata.stop_time)?;
        out.present("USEABLE_START_TIME", metadata.useable_start_time.as_deref())?;
        out.present("USEABLE_STOP_TIME", metadata.useable_stop_time.as_deref())?;
        out.present("INTERPOLATION", metadata.interpolation.as_deref())?;
        if let Some(value) = metadata.interpolation_degree {
            out.raw(&format!("INTERPOLATION_DEGREE = {value}"));
        }
        out.raw(META_STOP);

        check_positions(&segment.data_comments, segment.states.len())?;
        for (index, state) in segment.states.iter().enumerate() {
            out.positioned_comments(&segment.data_comments, index)?;
            out.raw(&encode_state_kvn(state)?);
        }
        out.positioned_comments(&segment.data_comments, segment.states.len())?;

        check_positions(&segment.covariance_comments, segment.covariances.len())?;
        if !segment.covariances.is_empty() || !segment.covariance_comments.is_empty() {
            out.raw(COVARIANCE_START);
            for (index, covariance) in segment.covariances.iter().enumerate() {
                out.positioned_comments(&segment.covariance_comments, index)?;
                out.required("EPOCH", &covariance.epoch)?;
                out.present(COV_REF_FRAME, covariance.cov_ref_frame.as_deref())?;
                for row in covariance_rows(&covariance.lower_triangle)? {
                    out.raw(&row);
                }
            }
            out.positioned_comments(&segment.covariance_comments, segment.covariances.len())?;
            out.raw(COVARIANCE_STOP);
        }
    }

    Ok(out.lines.join("\n"))
}

/// Refuse comments whose positions a writer cannot restate: past the end of
/// their list or out of source order.
fn check_positions(comments: &[OemComment], len: usize) -> Result<(), OemError> {
    let mut previous = 0usize;
    for comment in comments {
        if comment.position > len || comment.position < previous {
            return Err(unwritable(
                COMMENT,
                &comment.text,
                TextIssue::DetachedComment,
            ));
        }
        previous = comment.position;
    }
    Ok(())
}

/// Line assembly for [`encode_kvn`], checking that every value reads back.
#[derive(Default)]
struct KvnWriter {
    lines: Vec<String>,
}

impl KvnWriter {
    fn raw(&mut self, line: &str) {
        self.lines.push(line.to_string());
    }

    fn required(&mut self, key: &str, value: &str) -> Result<(), OemError> {
        if let Some(issue) = ndm::text::kvn_required_issue(value) {
            return Err(unwritable(key, value, issue));
        }
        self.lines.push(format!("{key} = {value}"));
        Ok(())
    }

    /// A keyword written blank when absent, which reads back absent.
    fn optional(&mut self, key: &str, value: Option<&str>) -> Result<(), OemError> {
        let value = value.unwrap_or_default();
        if let Some(issue) = ndm::text::kvn_value_issue(value) {
            return Err(unwritable(key, value, issue));
        }
        self.lines.push(format!("{key} = {value}"));
        Ok(())
    }

    /// A keyword written only when present.
    fn present(&mut self, key: &str, value: Option<&str>) -> Result<(), OemError> {
        match value {
            Some(value) => self.optional(key, Some(value)),
            None => Ok(()),
        }
    }

    fn comment(&mut self, comment: &str) -> Result<(), OemError> {
        if let Some(issue) = ndm::text::kvn_comment_issue(comment) {
            return Err(unwritable(COMMENT, comment, issue));
        }
        if comment.is_empty() {
            self.lines.push(COMMENT.to_string());
        } else {
            self.lines.push(format!("{COMMENT} {comment}"));
        }
        Ok(())
    }

    fn comments(&mut self, comments: &[String]) -> Result<(), OemError> {
        for comment in comments {
            self.comment(comment)?;
        }
        Ok(())
    }

    fn positioned_comments(
        &mut self,
        comments: &[OemComment],
        position: usize,
    ) -> Result<(), OemError> {
        for comment in comments
            .iter()
            .filter(|comment| comment.position == position)
        {
            self.comment(&comment.text)?;
        }
        Ok(())
    }
}

fn unwritable(field: &str, value: &str, issue: TextIssue) -> OemError {
    OemError::UnwritableText {
        field: field.to_string(),
        value: value.to_string(),
        issue,
    }
}

fn check_finite(field: &'static str, value: f64) -> Result<(), OemError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(OemError::InvalidField {
            field,
            kind: OemInputErrorKind::NonFinite,
        })
    }
}

/// Parse a CCSDS OEM in XML encoding into an [`Oem`].
///
/// The message may be the root element or sit inside an `<ndm>` combined
/// instantiation (505.0-B-3 4.11); a document holding more than one OEM is
/// refused with [`OemError::MultipleMessages`]. Header values are read from
/// `<header>` and each segment's values from its own elements. A `units`
/// attribute must match the unit 502.0-B-3 gives the element (8.10.11), and an
/// element the tables do not define at its position is refused by name when it
/// holds a value. A `COMMENT` in `<data>` before the first covariance matrix is
/// an ephemeris comment; one inside or after a covariance matrix is a
/// covariance comment.
pub fn parse_xml(text: &str) -> Result<Oem, OemError> {
    let doc = Document::parse(text).map_err(|e| OemError::Field(format!("malformed XML: {e}")))?;
    let messages = ndm::message_elements(&doc, "oem");
    let oem_node = match messages.as_slice() {
        [] => return Err(OemError::Field("missing oem element".to_string())),
        [message] => *message,
        _ => {
            return Err(OemError::MultipleMessages {
                count: messages.len(),
            })
        }
    };

    let mut header = (FieldMap::default(), Vec::new());
    let mut segments = Vec::new();
    for child in ndm::element_children(oem_node) {
        match child.tag_name().name() {
            "header" => header = xml_block(child, HEADER_KEYS)?,
            "body" => {
                for body_child in ndm::element_children(child) {
                    match body_child.tag_name().name() {
                        "segment" => segments.push(parse_xml_segment(body_child)?),
                        other => unknown_element("body", other, body_child)?,
                    }
                }
            }
            other => unknown_element("oem", other, child)?,
        }
    }
    let (header_map, header_comments) = header;
    let version = oem_node
        .attribute("version")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| opt_text(&header_map, OEM_VERSION_KEY))
        .ok_or(OemError::MissingField(OEM_VERSION_KEY))?;

    if segments.is_empty() {
        return Err(OemError::Field("OEM contains no segment".to_string()));
    }

    Ok(Oem {
        ccsds_oem_vers: version,
        comments: header_comments,
        classification: opt_text(&header_map, "CLASSIFICATION"),
        creation_date: opt_text(&header_map, "CREATION_DATE"),
        originator: opt_text(&header_map, "ORIGINATOR"),
        message_id: opt_text(&header_map, "MESSAGE_ID"),
        segments,
        skipped_states: Vec::new(),
    })
}

/// Refuse an element the tables do not define at its position when it holds a
/// value; an empty one holds nothing to keep.
fn unknown_element(parent: &str, name: &str, node: Node) -> Result<(), OemError> {
    if ndm::carries_data(node) {
        Err(OemError::UnknownField(format!("{parent}/{name}")))
    } else {
        Ok(())
    }
}

fn parse_xml_segment(segment: Node) -> Result<OemSegment, OemError> {
    let mut metadata = None;
    let mut data = None;
    for child in ndm::element_children(segment) {
        match child.tag_name().name() {
            "metadata" => {
                let (map, comments) = xml_block(child, METADATA_KEYS)?;
                metadata = Some(parse_metadata(&map, comments)?);
            }
            "data" => data = Some(child),
            other => unknown_element("segment", other, child)?,
        }
    }
    let metadata =
        metadata.ok_or_else(|| OemError::Field("segment missing metadata".to_string()))?;
    let data = data.ok_or_else(|| OemError::Field("segment missing data".to_string()))?;

    let mut data_comments = Vec::new();
    let mut states = Vec::new();
    let mut covariance_comments = Vec::new();
    let mut covariances = Vec::new();
    for child in ndm::element_children(data) {
        match child.tag_name().name() {
            COMMENT => {
                if let Some(path) = ndm::nested_element(child) {
                    return Err(OemError::UnknownField(path));
                }
                let text = ndm::comment_text(child);
                if covariances.is_empty() {
                    data_comments.push(OemComment {
                        position: states.len(),
                        text,
                    });
                } else {
                    covariance_comments.push(OemComment {
                        position: covariances.len(),
                        text,
                    });
                }
            }
            "stateVector" => {
                let (map, comments) = xml_block(child, &STATE_ELEMENT_KEYS)?;
                data_comments.extend(comments.into_iter().map(|text| OemComment {
                    position: states.len(),
                    text,
                }));
                states.push(parse_xml_state(&map)?);
            }
            "covarianceMatrix" => {
                let (map, comments) = xml_block(child, &COVARIANCE_ELEMENT_KEYS)?;
                covariance_comments.extend(comments.into_iter().map(|text| OemComment {
                    position: covariances.len(),
                    text,
                }));
                covariances.push(parse_covariance_map(&map)?);
            }
            other => unknown_element("data", other, child)?,
        }
    }

    Ok(OemSegment {
        metadata,
        data_comments,
        states,
        covariance_comments,
        covariances,
    })
}

/// Element names of an XML `stateVector` (502.0-B-3 table 8-6).
const STATE_ELEMENT_KEYS: [&str; 10] = [
    "EPOCH", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT", "X_DDOT", "Y_DDOT", "Z_DDOT",
];

/// Element names of an XML `covarianceMatrix` (502.0-B-3 table 8-7).
const COVARIANCE_ELEMENT_KEYS: [&str; 23] = [
    "EPOCH",
    COV_REF_FRAME,
    "CX_X",
    "CY_X",
    "CY_Y",
    "CZ_X",
    "CZ_Y",
    "CZ_Z",
    "CX_DOT_X",
    "CX_DOT_Y",
    "CX_DOT_Z",
    "CX_DOT_X_DOT",
    "CY_DOT_X",
    "CY_DOT_Y",
    "CY_DOT_Z",
    "CY_DOT_X_DOT",
    "CY_DOT_Y_DOT",
    "CZ_DOT_X",
    "CZ_DOT_Y",
    "CZ_DOT_Z",
    "CZ_DOT_X_DOT",
    "CZ_DOT_Y_DOT",
    "CZ_DOT_Z_DOT",
];

/// Read the `COMMENT` and keyword elements of one XML block. A `units`
/// attribute must match the unit the tables give the element, an element
/// outside `keys` that holds a value is refused, and a keyword repeated with a
/// different value is refused.
fn xml_block(node: Node, keys: &[&str]) -> Result<(FieldMap, Vec<String>), OemError> {
    const NO_UNIT: &[&str] = &[];
    let mut pairs = Vec::new();
    let mut comments = Vec::new();
    for child in ndm::element_children(node) {
        let name = child.tag_name().name();
        if let Some(nested) = ndm::element_children(child).first() {
            return Err(OemError::UnknownField(format!(
                "{name}/{}",
                nested.tag_name().name()
            )));
        }
        if name == COMMENT {
            comments.push(ndm::comment_text(child));
            continue;
        }
        if !keys.contains(&name) {
            unknown_element(node.tag_name().name(), name, child)?;
            continue;
        }
        if let Some(unit) = ndm::units_attribute(child) {
            ndm::check_unit(Some(unit.trim()), oem_xml_unit(name).unwrap_or(NO_UNIT))
                .map_err(|mismatch| unit_mismatch(name, mismatch))?;
        }
        pairs.push((name.to_string(), ndm::leaf_text(child)));
    }
    let map = FieldMap::from_pairs(pairs);
    reject_conflict(&map)?;
    Ok((map, comments))
}

fn parse_xml_state(map: &FieldMap) -> Result<OemState, OemError> {
    let epoch = req_text(map, "EPOCH")?;
    let x = req_num(map, "X")?;
    let y = req_num(map, "Y")?;
    let z = req_num(map, "Z")?;
    let xd = req_num(map, "X_DOT")?;
    let yd = req_num(map, "Y_DOT")?;
    let zd = req_num(map, "Z_DOT")?;

    let acceleration_km_s2 = match (map.get("X_DDOT"), map.get("Y_DDOT"), map.get("Z_DDOT")) {
        (None, None, None) => None,
        (Some(xdd), Some(ydd), Some(zdd)) => Some([
            parse_num(xdd, "X_DDOT")?,
            parse_num(ydd, "Y_DDOT")?,
            parse_num(zdd, "Z_DDOT")?,
        ]),
        _ => {
            return Err(OemError::Field(
                "stateVector acceleration must contain X_DDOT, Y_DDOT, and Z_DDOT".to_string(),
            ))
        }
    };

    Ok(OemState {
        epoch,
        position_km: [x, y, z],
        velocity_km_s: [xd, yd, zd],
        acceleration_km_s2,
    })
}

/// Encode an [`Oem`] as CCSDS OEM XML.
///
/// Text the reader would not return unchanged is refused with
/// [`OemError::UnwritableText`]: a value with surrounding whitespace (the
/// reader trims element text), an empty required value, a comment with
/// trailing whitespace, any character XML 1.0 cannot carry, a comment
/// positioned past the end of its list or out of order, and a covariance
/// comment in a segment without covariance matrices, which would read back as
/// an ephemeris comment. A non-finite number is refused with
/// [`OemError::InvalidField`].
pub fn encode_xml(oem: &Oem) -> Result<String, OemError> {
    let mut out = XmlWriter::default();
    out.raw(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    if let Some(issue) = ndm::text::xml_required_issue(&oem.ccsds_oem_vers) {
        return Err(unwritable(OEM_VERSION_KEY, &oem.ccsds_oem_vers, issue));
    }
    out.raw(&format!(
        r#"<oem id="CCSDS_OEM_VERS" version="{}">"#,
        ndm::text::escape_attribute(&oem.ccsds_oem_vers)
    ));
    out.raw("  <header>");
    out.comments(4, &oem.comments)?;
    out.present(4, "CLASSIFICATION", oem.classification.as_deref())?;
    out.optional(4, "CREATION_DATE", oem.creation_date.as_deref())?;
    out.optional(4, "ORIGINATOR", oem.originator.as_deref())?;
    out.present(4, "MESSAGE_ID", oem.message_id.as_deref())?;
    out.raw("  </header>");
    out.raw("  <body>");

    for segment in &oem.segments {
        out.raw("    <segment>");
        out.raw("      <metadata>");
        let metadata = &segment.metadata;
        out.comments(8, &metadata.comments)?;
        out.required(8, "OBJECT_NAME", &metadata.object_name)?;
        out.required(8, "OBJECT_ID", &metadata.object_id)?;
        out.required(8, "CENTER_NAME", &metadata.center_name)?;
        out.required(8, "REF_FRAME", &metadata.ref_frame)?;
        out.present(8, "REF_FRAME_EPOCH", metadata.ref_frame_epoch.as_deref())?;
        out.required(8, "TIME_SYSTEM", &metadata.time_system)?;
        out.required(8, "START_TIME", &metadata.start_time)?;
        out.required(8, "STOP_TIME", &metadata.stop_time)?;
        out.present(
            8,
            "USEABLE_START_TIME",
            metadata.useable_start_time.as_deref(),
        )?;
        out.present(
            8,
            "USEABLE_STOP_TIME",
            metadata.useable_stop_time.as_deref(),
        )?;
        out.present(8, "INTERPOLATION", metadata.interpolation.as_deref())?;
        if let Some(value) = metadata.interpolation_degree {
            out.element(8, "INTERPOLATION_DEGREE", &value.to_string());
        }
        out.raw("      </metadata>");
        out.raw("      <data>");

        check_positions(&segment.data_comments, segment.states.len())?;
        check_positions(&segment.covariance_comments, segment.covariances.len())?;
        if segment.covariances.is_empty() {
            if let Some(comment) = segment.covariance_comments.first() {
                return Err(unwritable(
                    COMMENT,
                    &comment.text,
                    TextIssue::DetachedComment,
                ));
            }
        }
        for (index, state) in segment.states.iter().enumerate() {
            out.positioned_comments(8, &segment.data_comments, index)?;
            out.raw("        <stateVector>");
            out.required(10, "EPOCH", &state.epoch)?;
            for (key, value) in state_values(state) {
                out.number(10, key, value)?;
            }
            out.raw("        </stateVector>");
        }
        out.positioned_comments(8, &segment.data_comments, segment.states.len())?;
        for (index, covariance) in segment.covariances.iter().enumerate() {
            out.raw("        <covarianceMatrix>");
            out.positioned_comments(10, &segment.covariance_comments, index)?;
            out.required(10, "EPOCH", &covariance.epoch)?;
            out.present(10, COV_REF_FRAME, covariance.cov_ref_frame.as_deref())?;
            for (key, value) in COVARIANCE6_KEYS.into_iter().zip(covariance.lower_triangle) {
                out.number(10, key, value)?;
            }
            out.raw("        </covarianceMatrix>");
        }
        out.positioned_comments(8, &segment.covariance_comments, segment.covariances.len())?;

        out.raw("      </data>");
        out.raw("    </segment>");
    }

    out.raw("  </body>");
    out.raw("</oem>");
    Ok(out.lines.join("\n"))
}

/// Element assembly for [`encode_xml`], checking that every value reads back.
#[derive(Default)]
struct XmlWriter {
    lines: Vec<String>,
}

impl XmlWriter {
    fn raw(&mut self, line: &str) {
        self.lines.push(line.to_string());
    }

    fn element(&mut self, indent: usize, name: &str, content: &str) {
        self.lines
            .push(format!("{:indent$}<{name}>{content}</{name}>", ""));
    }

    fn required(&mut self, indent: usize, name: &str, value: &str) -> Result<(), OemError> {
        if let Some(issue) = ndm::text::xml_required_issue(value) {
            return Err(unwritable(name, value, issue));
        }
        self.element(indent, name, &xml::escape(value));
        Ok(())
    }

    /// An element written empty when absent, which reads back absent.
    fn optional(&mut self, indent: usize, name: &str, value: Option<&str>) -> Result<(), OemError> {
        let value = value.unwrap_or_default();
        if let Some(issue) = ndm::text::xml_value_issue(value) {
            return Err(unwritable(name, value, issue));
        }
        self.element(indent, name, &xml::escape(value));
        Ok(())
    }

    fn present(&mut self, indent: usize, name: &str, value: Option<&str>) -> Result<(), OemError> {
        match value {
            Some(value) => self.optional(indent, name, Some(value)),
            None => Ok(()),
        }
    }

    fn number(&mut self, indent: usize, name: &'static str, value: f64) -> Result<(), OemError> {
        check_finite(name, value)?;
        self.element(indent, name, &fmt_num(value));
        Ok(())
    }

    fn comment(&mut self, indent: usize, comment: &str) -> Result<(), OemError> {
        if let Some(issue) = ndm::text::xml_comment_issue(comment) {
            return Err(unwritable(COMMENT, comment, issue));
        }
        self.element(indent, COMMENT, &xml::escape(comment));
        Ok(())
    }

    fn comments(&mut self, indent: usize, comments: &[String]) -> Result<(), OemError> {
        for comment in comments {
            self.comment(indent, comment)?;
        }
        Ok(())
    }

    fn positioned_comments(
        &mut self,
        indent: usize,
        comments: &[OemComment],
        position: usize,
    ) -> Result<(), OemError> {
        for comment in comments
            .iter()
            .filter(|comment| comment.position == position)
        {
            self.comment(indent, &comment.text)?;
        }
        Ok(())
    }
}

/// The state components with their element names, acceleration included when
/// present.
fn state_values(state: &OemState) -> Vec<(&'static str, f64)> {
    let mut values = vec![
        ("X", state.position_km[0]),
        ("Y", state.position_km[1]),
        ("Z", state.position_km[2]),
        ("X_DOT", state.velocity_km_s[0]),
        ("Y_DOT", state.velocity_km_s[1]),
        ("Z_DOT", state.velocity_km_s[2]),
    ];
    if let Some(accel) = state.acceleration_km_s2 {
        values.extend([
            ("X_DDOT", accel[0]),
            ("Y_DDOT", accel[1]),
            ("Z_DDOT", accel[2]),
        ]);
    }
    values
}

/// An ephemeris data line: the epoch and the state components separated by
/// blanks (502.0-B-3 5.2.4.1-5.2.4.3). The epoch must be one non-empty token.
fn encode_state_kvn(state: &OemState) -> Result<String, OemError> {
    let issue = if state.epoch.is_empty() {
        Some(TextIssue::Empty)
    } else if state.epoch.contains(['\n', '\r']) {
        Some(TextIssue::LineBreak)
    } else if state.epoch.trim() != state.epoch {
        Some(TextIssue::SurroundingWhitespace)
    } else if state.epoch.contains(char::is_whitespace) {
        Some(TextIssue::InteriorWhitespace)
    } else {
        None
    };
    if let Some(issue) = issue {
        return Err(unwritable("EPOCH", &state.epoch, issue));
    }
    let mut fields = vec![state.epoch.clone()];
    for (key, value) in state_values(state) {
        check_finite(key, value)?;
        fields.push(fmt_num(value));
    }
    Ok(fields.join(" "))
}

/// The lower triangle of a covariance as six rows of one to six values.
fn covariance_rows(lower: &[f64; 21]) -> Result<Vec<String>, OemError> {
    let mut rows = Vec::with_capacity(6);
    let mut start = 0usize;
    for length in 1..=6 {
        let mut row = Vec::with_capacity(length);
        for (offset, value) in lower[start..start + length].iter().enumerate() {
            check_finite(COVARIANCE6_KEYS[start + offset], *value)?;
            row.push(fmt_num(*value));
        }
        rows.push(row.join(" "));
        start += length;
    }
    Ok(rows)
}

fn parse_metadata(map: &FieldMap, comments: Vec<String>) -> Result<OemMetadata, OemError> {
    Ok(OemMetadata {
        comments,
        object_name: req_text(map, "OBJECT_NAME")?,
        object_id: req_text(map, "OBJECT_ID")?,
        center_name: req_text(map, "CENTER_NAME")?,
        ref_frame: req_text(map, "REF_FRAME")?,
        ref_frame_epoch: opt_text(map, "REF_FRAME_EPOCH"),
        time_system: req_text(map, "TIME_SYSTEM")?,
        start_time: req_text(map, "START_TIME")?,
        stop_time: req_text(map, "STOP_TIME")?,
        useable_start_time: opt_text(map, "USEABLE_START_TIME"),
        useable_stop_time: opt_text(map, "USEABLE_STOP_TIME"),
        interpolation: opt_text(map, "INTERPOLATION"),
        interpolation_degree: opt_u32(map, "INTERPOLATION_DEGREE")?,
    })
}

fn parse_state_line(line: &str) -> Result<OemState, OemStateLineError> {
    let mut tokenizer = Tokenizer::new(line);
    let mut tokens = Vec::new();
    while let Some(token) = tokenizer.next_str() {
        tokens.push(token);
    }

    if tokens.len() != 7 && tokens.len() != 10 {
        return Err(OemStateLineError::ItemCount(tokens.len()));
    }

    let epoch = tokens[0].to_string();
    let mut values = [0.0_f64; 9];
    for (idx, &key) in STATE_NUMBER_KEYS.iter().enumerate().take(tokens.len() - 1) {
        values[idx] = validate::strict_f64(tokens[idx + 1], key).map_err(|error| {
            OemStateLineError::InvalidField {
                field: key,
                kind: OemInputErrorKind::from(&error),
            }
        })?;
    }

    let acceleration_km_s2 = if tokens.len() == 10 {
        Some([values[6], values[7], values[8]])
    } else {
        None
    };

    Ok(OemState {
        epoch,
        position_km: [values[0], values[1], values[2]],
        velocity_km_s: [values[3], values[4], values[5]],
        acceleration_km_s2,
    })
}

fn parse_covariance_map(map: &FieldMap) -> Result<OemCovariance, OemError> {
    Ok(OemCovariance {
        epoch: req_text(map, "EPOCH")?,
        cov_ref_frame: opt_text(map, COV_REF_FRAME),
        lower_triangle: read_lower_triangle6(map).map_err(map_oem_field_error)?,
    })
}

/// Number the physical lines of a KVN message from one, with surrounding
/// whitespace removed. Lines end at CR, LF, CR LF or LF CR (502.0-B-3 7.3.7).
fn numbered_lines(text: &str) -> Vec<(usize, String)> {
    ndm::kvn_lines(text)
        .into_iter()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim().to_string()))
        .collect()
}

fn reject_conflict(map: &FieldMap) -> Result<(), OemError> {
    match map.first_conflict(|key| key != COMMENT) {
        Some(conflict) => Err(OemError::DuplicateField {
            field: conflict.key,
            first: conflict.first,
            second: conflict.second,
        }),
        None => Ok(()),
    }
}

/// The unit CCSDS 502.0-B-3 tables 8-6 and 8-7 give an OEM XML element: an
/// empty list for a dimensionless one, `None` for a text element.
fn oem_xml_unit(name: &str) -> Option<&'static [&'static str]> {
    const DIMENSIONLESS: &[&str] = &[];
    const KM: &[&str] = &["km"];
    const KM_PER_S: &[&str] = &["km/s"];
    const KM_PER_S2: &[&str] = &["km/s**2"];
    match name {
        "X" | "Y" | "Z" => Some(KM),
        "X_DOT" | "Y_DOT" | "Z_DOT" => Some(KM_PER_S),
        "X_DDOT" | "Y_DDOT" | "Z_DDOT" => Some(KM_PER_S2),
        "INTERPOLATION_DEGREE" => Some(DIMENSIONLESS),
        other => covariance6_unit(other),
    }
}

fn unit_mismatch(name: &str, mismatch: UnitMismatch) -> OemError {
    OemError::UnitMismatch {
        field: name.to_string(),
        unit: mismatch.unit,
        expected: mismatch.expected,
    }
}

fn req_text(map: &FieldMap, field: &'static str) -> Result<String, OemError> {
    map.get(field)
        .map(str::to_string)
        .ok_or(OemError::MissingField(field))
}

fn opt_text(map: &FieldMap, field: &'static str) -> Option<String> {
    map.get(field).map(str::to_string)
}

fn opt_u32(map: &FieldMap, field: &'static str) -> Result<Option<u32>, OemError> {
    map.get(field)
        .map(|value| validate::strict_int::<u32>(value, field).map_err(map_oem_field_error))
        .transpose()
}

fn req_num(map: &FieldMap, field: &'static str) -> Result<f64, OemError> {
    let value = map.get(field).ok_or(OemError::MissingField(field))?;
    parse_num(value, field)
}

fn parse_num(value: &str, field: &'static str) -> Result<f64, OemError> {
    validate::strict_f64(value, field).map_err(map_oem_field_error)
}

fn map_oem_field_error(error: validate::FieldError) -> OemError {
    OemError::InvalidField {
        field: error.field(),
        kind: OemInputErrorKind::from(&error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lower triangle of a diagonal covariance.
    fn diagonal_covariance() -> [f64; 21] {
        let mut lower = [0.0; 21];
        for (index, value) in [
            (0, 1.0),
            (2, 2.0),
            (5, 3.0),
            (9, 4.0e-6),
            (14, 5.0e-6),
            (20, 6.0e-6),
        ] {
            lower[index] = value;
        }
        lower
    }

    #[test]
    fn forgiving_kvn_skips_malformed_state_lines() {
        let kvn = "\
CCSDS_OEM_VERS = 2.0
CREATION_DATE = 2026-06-28T00:00:00
ORIGINATOR = SIDEREON
META_START
OBJECT_NAME = TEST
OBJECT_ID = 2026-001A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
START_TIME = 2026-06-28T00:00:00
STOP_TIME = 2026-06-28T00:10:00
META_STOP
2026-06-28T00:00:00 1 2 3 0.1 0.2 0.3
2026-06-28T00:05:00 1 2
2026-06-28T00:10:00 1 2 3 0.1 NaN 0.3
";
        let oem = parse_kvn(kvn).expect("forgiving OEM parse");
        assert_eq!(oem.segments[0].states.len(), 1);
        assert_eq!(
            oem.skipped_states,
            vec![
                OemSkippedState {
                    line: 14,
                    segment: 0,
                    text: "2026-06-28T00:05:00 1 2".to_string(),
                    reason: OemStateLineError::ItemCount(3),
                },
                OemSkippedState {
                    line: 15,
                    segment: 0,
                    text: "2026-06-28T00:10:00 1 2 3 0.1 NaN 0.3".to_string(),
                    reason: OemStateLineError::InvalidField {
                        field: "Y_DOT",
                        kind: OemInputErrorKind::NonFinite,
                    },
                },
            ]
        );
    }

    #[test]
    fn malformed_xml_is_an_error() {
        assert!(parse_xml("<oem></oem><oem></oem>").is_err());
    }

    #[test]
    fn covariance_round_trips_through_kvn_and_xml() {
        let original = Oem {
            ccsds_oem_vers: "2.0".to_string(),
            comments: vec!["header comment".to_string()],
            classification: Some("SBU".to_string()),
            creation_date: Some("2026-06-28T00:00:00".to_string()),
            originator: Some("SIDEREON".to_string()),
            message_id: Some("OEM 201113719185".to_string()),
            skipped_states: Vec::new(),
            segments: vec![OemSegment {
                metadata: OemMetadata {
                    comments: vec!["metadata comment".to_string()],
                    object_name: "TEST".to_string(),
                    object_id: "2026-001A".to_string(),
                    center_name: "EARTH".to_string(),
                    ref_frame: "EME2000".to_string(),
                    ref_frame_epoch: Some("2000-01-01T12:00:00".to_string()),
                    time_system: "UTC".to_string(),
                    start_time: "2026-06-28T00:00:00".to_string(),
                    stop_time: "2026-06-28T00:10:00".to_string(),
                    useable_start_time: None,
                    useable_stop_time: None,
                    interpolation: Some("LAGRANGE".to_string()),
                    interpolation_degree: Some(5),
                },
                data_comments: vec![
                    OemComment {
                        position: 0,
                        text: "before the first state".to_string(),
                    },
                    OemComment {
                        position: 1,
                        text: "after the last state".to_string(),
                    },
                ],
                states: vec![OemState {
                    epoch: "2026-06-28T00:00:00".to_string(),
                    position_km: [1.0, 2.0, 3.0],
                    velocity_km_s: [0.1, 0.2, 0.3],
                    acceleration_km_s2: None,
                }],
                covariance_comments: vec![
                    OemComment {
                        position: 0,
                        text: "before the matrix".to_string(),
                    },
                    OemComment {
                        position: 1,
                        text: "after the matrix".to_string(),
                    },
                ],
                covariances: vec![OemCovariance {
                    epoch: "2026-06-28T00:00:00".to_string(),
                    cov_ref_frame: Some("RTN".to_string()),
                    lower_triangle: diagonal_covariance(),
                }],
            }],
        };

        assert_eq!(
            parse_kvn(&encode_kvn(&original).unwrap()).unwrap(),
            original
        );
        assert_eq!(
            parse_xml(&encode_xml(&original).unwrap()).unwrap(),
            original
        );
    }

    /// CCSDS 502.0-B-3 annex G figure G-13 without its elision marker: two
    /// covariance matrices in one section, each an EPOCH, a COV_REF_FRAME and
    /// six rows of the lower triangle (5.2.5, 7.4.1.3).
    const STANDARD_COVARIANCE_KVN: &str = "\
CCSDS_OEM_VERS = 3.0
CREATION_DATE = 2019-11-04T17:22:31
ORIGINATOR = NASA/JPL
MESSAGE_ID = OEM 201113719185
META_START
OBJECT_NAME           = MARS GLOBAL SURVEYOR
OBJECT_ID             = 1996-062A
CENTER_NAME           = MARS BARYCENTER
REF_FRAME             = EME2000
TIME_SYSTEM           = UTC
START_TIME            = 2019-12-28T21:29:07.267
USEABLE_START_TIME    = 2019-12-28T22:08:02.5
USEABLE_STOP_TIME     = 2019-12-30T01:18:02.5
STOP_TIME             = 2019-12-30T01:28:02.267
INTERPOLATION         = HERMITE
INTERPOLATION_DEGREE = 7
META_STOP


COMMENT This block begins after trajectory correction maneuver TCM-3.

2019-12-28T21:29:07.267 -2432.166 -063.042 1742.754     7.33702 -3.495867 -1.041945
2019-12-28T21:59:02.267 -2445.234 -878.141 1873.073     1.86043 -3.421256 -0.996366
2019-12-28T22:00:02.267 -2458.079 -683.858 2007.684     6.36786 -3.339563 -0.946654
2019-12-30T01:28:02.267 2164.375 1115.811 -688.131     -3.53328 -2.88452 0.88535


COVARIANCE_START
EPOCH = 2019-12-28T21:29:07.267
COV_REF_FRAME = EME2000
 3.3313494e-04
 4.6189273e-04 6.7824216e-04
-3.0700078e-04 -4.2212341e-04 3.2319319e-04
-3.3493650e-07 -4.6860842e-07 2.4849495e-07       4.2960228e-10
-2.2118325e-07 -2.8641868e-07 1.7980986e-07       2.6088992e-10   1.7675147e-10
-3.0413460e-07 -4.9894969e-07 3.5403109e-07       1.8692631e-10   1.0088625e-10   6.2244443e-10

EPOCH = 2019-12-29T21:00:00
COV_REF_FRAME = EME2000
 3.4424505e-04
 4.5078162e-04 6.8935327e-04
-3.0600067e-04 -4.1101230e-04   3.3420420e-04
-3.2382549e-07 -4.5750731e-07   2.3738384e-07     4.3071339e-10
-2.1007214e-07 -2.7530757e-07   1.6870875e-07     2.5077881e-10   1.8786258e-10
-3.0302350e-07 -4.8783858e-07   3.4302008e-07     1.7581520e-10   1.0077514e-10   6.2244443e-10
COVARIANCE_STOP
";

    #[test]
    fn reads_the_standard_covariance_section_and_writes_it_back() {
        let oem = parse_kvn(STANDARD_COVARIANCE_KVN).expect("502.0-B-3 figure G-13 parses");
        assert!(oem.skipped_states.is_empty());
        let segment = &oem.segments[0];
        assert_eq!(segment.states.len(), 4);
        assert_eq!(
            segment.states[0].position_km,
            [-2432.166, -63.042, 1742.754]
        );
        assert_eq!(segment.covariances.len(), 2);
        let first = &segment.covariances[0];
        assert_eq!(first.epoch, "2019-12-28T21:29:07.267");
        assert_eq!(first.cov_ref_frame.as_deref(), Some("EME2000"));
        // Lower-triangle positions: [0][0] is 0, [2][1] is 4, [5][5] is 20 and
        // [4][3] is 13.
        assert_eq!(first.lower_triangle[0], 3.3313494e-04);
        assert_eq!(first.lower_triangle[4], -4.2212341e-04);
        assert_eq!(first.lower_triangle[20], 6.2244443e-10);
        assert_eq!(segment.covariances[1].epoch, "2019-12-29T21:00:00");
        assert_eq!(segment.covariances[1].lower_triangle[13], 2.5077881e-10);
        assert!(first.to_covariance6().is_ok());

        let encoded = encode_kvn(&oem).unwrap();
        assert_eq!(encoded.matches(COVARIANCE_START).count(), 1);
        assert!(!encoded.contains("CX_X"));
        assert_eq!(parse_kvn(&encoded).unwrap(), oem);
        assert_eq!(parse_xml(&encode_xml(&oem).unwrap()).unwrap(), oem);
    }

    #[test]
    fn covariance_is_held_as_read_and_validated_only_on_request() {
        // A matrix that is not positive semidefinite is read and written back
        // as stated; the validated covariance is refused.
        let kvn = STANDARD_COVARIANCE_KVN.replacen("6.2244443e-10", "-6.2244443e-10", 1);
        let oem = parse_kvn(&kvn).unwrap();
        let first = &oem.segments[0].covariances[0];
        assert_eq!(first.lower_triangle[20], -6.2244443e-10);
        assert_eq!(
            first.to_covariance6(),
            Err(Covariance6Error::NotPositiveSemidefinite)
        );
        assert_eq!(parse_kvn(&encode_kvn(&oem).unwrap()).unwrap(), oem);
        assert_eq!(parse_xml(&encode_xml(&oem).unwrap()).unwrap(), oem);
    }

    #[test]
    fn covariance_rows_must_hold_the_lower_triangle() {
        let short_row = STANDARD_COVARIANCE_KVN.replacen(
            " 4.6189273e-04 6.7824216e-04\n",
            " 4.6189273e-04\n",
            1,
        );
        assert!(
            matches!(parse_kvn(&short_row), Err(OemError::Field(message)) if message.contains("covariance row 2 holds 1 values"))
        );

        let before_epoch = STANDARD_COVARIANCE_KVN.replacen(
            "COVARIANCE_START\nEPOCH = 2019-12-28T21:29:07.267\n",
            "COVARIANCE_START\n",
            1,
        );
        assert!(
            matches!(parse_kvn(&before_epoch), Err(OemError::Field(message)) if message.contains("precedes the covariance EPOCH"))
        );
    }

    #[test]
    fn header_is_read_only_before_the_first_metadata_block() {
        let kvn = "\
CCSDS_OEM_VERS = 2.0
ORIGINATOR = SIDEREON
META_START
OBJECT_NAME = TEST
OBJECT_ID = 2026-001A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
START_TIME = 2026-06-28T00:00:00
STOP_TIME = 2026-06-28T00:10:00
CREATION_DATE = 2030-01-01T00:00:00
META_STOP
2026-06-28T00:00:00 1 2 3 0.1 0.2 0.3
";
        // CREATION_DATE is a table 5-2 header keyword, not a table 5-3 metadata
        // keyword, so inside a metadata block it neither fills the header nor
        // disappears: it is refused by name.
        assert_eq!(
            parse_kvn(kvn),
            Err(OemError::UnknownField("CREATION_DATE".to_string()))
        );
        let without = kvn.replacen("CREATION_DATE = 2030-01-01T00:00:00\n", "", 1);
        let oem = parse_kvn(&without).unwrap();
        assert_eq!(oem.creation_date, None);
        assert_eq!(oem.originator.as_deref(), Some("SIDEREON"));

        let version_in_metadata = kvn.replacen("CCSDS_OEM_VERS = 2.0\n", "", 1).replacen(
            "META_START\n",
            "META_START\nCCSDS_OEM_VERS = 2.0\n",
            1,
        );
        assert_eq!(
            parse_kvn(&version_in_metadata),
            Err(OemError::MissingField(OEM_VERSION_KEY))
        );
    }

    #[test]
    fn metadata_text_is_verbatim_and_conflicting_repeats_are_refused() {
        let kvn = "\
CCSDS_OEM_VERS = 2.0
META_START
OBJECT_NAME = TEST [BLOCK 2]
OBJECT_ID = 2026-001A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
START_TIME = 2026-06-28T00:00:00
STOP_TIME = 2026-06-28T00:10:00
META_STOP
2026-06-28T00:00:00 1 2 3 0.1 0.2 0.3
";
        let oem = parse_kvn(kvn).unwrap();
        assert_eq!(oem.segments[0].metadata.object_name, "TEST [BLOCK 2]");
        assert_eq!(parse_kvn(&encode_kvn(&oem).unwrap()).unwrap(), oem);

        let repeated = kvn.replacen(
            "OBJECT_ID = 2026-001A\n",
            "OBJECT_ID = 2026-001A\nOBJECT_ID = 2026-001B\n",
            1,
        );
        assert_eq!(
            parse_kvn(&repeated),
            Err(OemError::DuplicateField {
                field: "OBJECT_ID".to_string(),
                first: "2026-001A".to_string(),
                second: "2026-001B".to_string(),
            })
        );
    }

    #[test]
    fn xml_units_and_message_count_are_checked() {
        let oem = parse_kvn(STANDARD_COVARIANCE_KVN).unwrap();
        let xml = encode_xml(&oem).unwrap();
        let with_unit = xml.replacen("<X>", "<X units=\"km\">", 1);
        assert_eq!(parse_xml(&with_unit).unwrap(), oem);
        let wrong_unit = xml.replacen("<X>", "<X units=\"m\">", 1);
        assert_eq!(
            parse_xml(&wrong_unit),
            Err(OemError::UnitMismatch {
                field: "X".to_string(),
                unit: "m".to_string(),
                expected: Some("km"),
            })
        );

        let message = xml.trim_start_matches(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        assert_eq!(
            parse_xml(&format!("<ndm>{message}{message}</ndm>")),
            Err(OemError::MultipleMessages { count: 2 })
        );
    }

    #[test]
    fn standard_example_retains_message_id_and_data_comments() {
        let oem = parse_kvn(STANDARD_COVARIANCE_KVN).unwrap();
        assert_eq!(oem.message_id.as_deref(), Some("OEM 201113719185"));
        assert_eq!(
            oem.segments[0].data_comments,
            vec![OemComment {
                position: 0,
                text: "This block begins after trajectory correction maneuver TCM-3.".to_string(),
            }]
        );
    }

    #[test]
    fn comments_keep_their_positions_among_states_and_matrices() {
        let kvn = STANDARD_COVARIANCE_KVN
            .replacen(
                "2019-12-28T21:59:02.267",
                "COMMENT between the first and second state\n2019-12-28T21:59:02.267",
                1,
            )
            .replacen(
                "\nEPOCH = 2019-12-29T21:00:00",
                "COMMENT before the second matrix\nEPOCH = 2019-12-29T21:00:00",
                1,
            );
        let oem = parse_kvn(&kvn).unwrap();
        let segment = &oem.segments[0];
        assert_eq!(segment.data_comments[1].position, 1);
        assert_eq!(
            segment.covariance_comments,
            vec![OemComment {
                position: 1,
                text: "before the second matrix".to_string(),
            }]
        );
        assert_eq!(parse_kvn(&encode_kvn(&oem).unwrap()).unwrap(), oem);
        assert_eq!(parse_xml(&encode_xml(&oem).unwrap()).unwrap(), oem);
    }

    #[test]
    fn unknown_header_and_covariance_keywords_are_refused() {
        let kvn = STANDARD_COVARIANCE_KVN.replacen(
            "ORIGINATOR = NASA/JPL\n",
            "ORIGINATOR = NASA/JPL\nFOO = 1\n",
            1,
        );
        assert_eq!(
            parse_kvn(&kvn),
            Err(OemError::UnknownField("FOO".to_string()))
        );
        let kvn = STANDARD_COVARIANCE_KVN.replacen(
            "COV_REF_FRAME = EME2000\n",
            "COV_REF_FRAME = EME2000\nCOV_TYPE = LTM\n",
            1,
        );
        assert_eq!(
            parse_kvn(&kvn),
            Err(OemError::UnknownField("COV_TYPE".to_string()))
        );
        // A keyword among the ephemeris lines was skipped as a malformed data
        // line; it names no state and is refused by name.
        let kvn = STANDARD_COVARIANCE_KVN.replacen(
            "2019-12-28T21:59:02.267 ",
            "INTERPOLATION = LAGRANGE\n2019-12-28T21:59:02.267 ",
            1,
        );
        assert_eq!(
            parse_kvn(&kvn),
            Err(OemError::UnknownField("INTERPOLATION".to_string()))
        );
        // An element inside a COMMENT is refused rather than dropped with
        // only the comment's text kept.
        let oem = parse_kvn(STANDARD_COVARIANCE_KVN).unwrap();
        let xml = encode_xml(&oem).unwrap().replacen(
            "<stateVector>",
            "<COMMENT>note <b>1</b></COMMENT><stateVector>",
            1,
        );
        assert_eq!(
            parse_xml(&xml),
            Err(OemError::UnknownField("COMMENT/b".to_string()))
        );
        let kvn = STANDARD_COVARIANCE_KVN.replacen(
            "ORIGINATOR = NASA/JPL\n",
            "ORIGINATOR = NASA/JPL\nnot an assignment\n",
            1,
        );
        assert_eq!(
            parse_kvn(&kvn),
            Err(OemError::MalformedLine {
                line: 4,
                text: "not an assignment".to_string(),
            })
        );
    }

    #[test]
    fn writers_refuse_text_that_would_read_back_differently() {
        let base = parse_kvn(STANDARD_COVARIANCE_KVN).unwrap();

        let mut oem = base.clone();
        oem.segments[0].states[0].epoch = "2019-12-28 21:29:07".to_string();
        assert_eq!(
            encode_kvn(&oem),
            Err(OemError::UnwritableText {
                field: "EPOCH".to_string(),
                value: "2019-12-28 21:29:07".to_string(),
                issue: TextIssue::InteriorWhitespace,
            })
        );

        let mut oem = base.clone();
        oem.segments[0].metadata.object_name = "MGS\nOBJECT_ID = X".to_string();
        assert!(matches!(
            encode_kvn(&oem),
            Err(OemError::UnwritableText {
                issue: TextIssue::LineBreak,
                ..
            })
        ));

        let mut oem = base.clone();
        oem.segments[0].states[1].velocity_km_s[2] = f64::INFINITY;
        assert_eq!(
            encode_xml(&oem),
            Err(OemError::InvalidField {
                field: "Z_DOT",
                kind: OemInputErrorKind::NonFinite,
            })
        );

        let mut oem = base;
        oem.segments[0].data_comments[0].position = 99;
        assert!(matches!(
            encode_kvn(&oem),
            Err(OemError::UnwritableText {
                issue: TextIssue::DetachedComment,
                ..
            })
        ));
    }

    #[cfg(all(test, sidereon_repo_tests))]
    mod fixtures {
        use super::*;

        const GPS_KVN: &str = include_str!("../../tests/fixtures/oem/gps.kvn");
        const GPS_XML: &str = include_str!("../../tests/fixtures/oem/gps.xml");

        #[test]
        fn parses_gps_kvn_fixture() {
            let oem = parse_kvn(GPS_KVN).unwrap();
            assert_eq!(oem.ccsds_oem_vers, "2.0");
            assert_eq!(oem.originator.as_deref(), Some("SIDEREON TEST"));
            assert_eq!(oem.segments.len(), 1);
            assert_eq!(oem.segments[0].metadata.object_name, "GPS BIIRM-8");
            assert_eq!(oem.segments[0].states.len(), 3);
            assert_eq!(oem.segments[0].covariances.len(), 1);
            assert!(oem.skipped_states.is_empty());
        }

        #[test]
        fn parses_gps_xml_fixture() {
            let oem = parse_xml(GPS_XML).unwrap();
            assert_eq!(oem.ccsds_oem_vers, "2.0");
            assert_eq!(oem.segments[0].metadata.object_id, "2005-038A");
            assert_eq!(oem.segments[0].states[1].epoch, "2026-06-28T00:15:00.000");
            assert_eq!(
                oem.segments[0].covariances[0].cov_ref_frame.as_deref(),
                Some("RTN")
            );
        }

        #[test]
        fn fixture_kvn_round_trips() {
            let oem = parse_kvn(GPS_KVN).unwrap();
            assert_eq!(parse_kvn(&encode_kvn(&oem).unwrap()).unwrap(), oem);
        }

        #[test]
        fn fixture_xml_round_trips() {
            let oem = parse_xml(GPS_XML).unwrap();
            assert_eq!(parse_xml(&encode_xml(&oem).unwrap()).unwrap(), oem);
        }
    }
}
