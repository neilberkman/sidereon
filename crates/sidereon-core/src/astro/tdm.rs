//! CCSDS Tracking Data Message KVN reader and writer.
//!
//! This module implements the CCSDS 503.0-B-2 KVN form as a sans-IO parser and
//! serializer. Date/time fields remain raw strings, matching the other NDM
//! readers in this crate. Observable values are kept as both the parsed `f64`
//! and the exact decimal token read from the message, so frequency-domain
//! records such as `RECEIVE_FREQ` and `TRANSMIT_FREQ_n` re-emit without decimal
//! rewriting.

use std::fmt;

const VERSION_KEY: &str = "CCSDS_TDM_VERS";
const COMMENT_KEY: &str = "COMMENT";

/// A parsed CCSDS Tracking Data Message.
#[derive(Debug, Clone, PartialEq)]
pub struct Tdm {
    /// The `CCSDS_TDM_VERS` header value.
    pub version: String,
    /// Header comments in parse order.
    pub comments: Vec<String>,
    /// The optional `CREATION_DATE` header value.
    pub creation_date: Option<String>,
    /// The optional `ORIGINATOR` header value.
    pub originator: Option<String>,
    /// The optional `MESSAGE_ID` header value.
    pub message_id: Option<String>,
    /// Header fields that are not part of the common modeled header.
    pub header_fields: Vec<TdmField>,
    /// Metadata/data segments in message order.
    pub segments: Vec<TdmSegment>,
}

/// One TDM segment, consisting of one metadata block and one data block.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmSegment {
    /// Metadata describing the records in this segment.
    pub metadata: TdmMetadata,
    /// Tracking data records in this segment.
    pub data: TdmDataSection,
}

/// A KVN key/value field preserved in parse order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdmField {
    /// The KVN keyword.
    pub key: String,
    /// The trimmed KVN value.
    pub value: String,
}

/// Metadata extracted from a TDM `META_START` / `META_STOP` block.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmMetadata {
    /// Metadata comments in parse order.
    pub comments: Vec<String>,
    /// Raw metadata fields in parse order.
    pub fields: Vec<TdmField>,
    /// Parsed `PARTICIPANT_n` entries.
    pub participants: Vec<TdmParticipant>,
    /// The optional `MODE` metadata value.
    pub mode: Option<String>,
    /// Parsed `PATH`, `PATH_1`, and `PATH_2` entries.
    pub paths: Vec<TdmPath>,
    /// The optional `TIMETAG_REF` metadata value.
    pub timetag_ref: Option<String>,
    /// The optional `TIME_SYSTEM` metadata value.
    pub time_system: Option<String>,
    /// The range unit for `RANGE` records, defaulting to kilometers when absent.
    pub range_units: TdmUnit,
}

impl TdmMetadata {
    /// Return the last metadata value for `key`.
    pub fn get_last(&self, key: &str) -> Option<&str> {
        self.fields
            .iter()
            .rev()
            .find(|field| field.key == key)
            .map(|field| field.value.as_str())
            .filter(|value| !value.is_empty())
    }
}

/// One named tracking participant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdmParticipant {
    /// The numeric suffix from `PARTICIPANT_n`.
    pub index: u8,
    /// The participant name.
    pub name: String,
}

/// A parsed signal path from `PATH`, `PATH_1`, or `PATH_2`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TdmPath {
    /// The original path keyword.
    pub key: String,
    /// The path suffix for `PATH_n`, or `None` for the unindexed `PATH`.
    pub index: Option<u8>,
    /// Participant indices listed in path order.
    pub participants: Vec<u8>,
}

/// A TDM data block.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmDataSection {
    /// Data-section comments in parse order.
    pub comments: Vec<String>,
    /// Data records in parse order.
    pub records: Vec<TdmDataRecord>,
}

/// One time-tagged tracking data record.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmDataRecord {
    /// The parsed observable family.
    pub observable: TdmObservable,
    /// The original data keyword.
    pub keyword: String,
    /// The raw epoch string.
    pub epoch: String,
    /// The numeric observable value.
    pub value: TdmScalar,
    /// The unit assigned by CCSDS 503.0-B-2.
    pub unit: TdmUnit,
}

/// A numeric record value plus the exact decimal token used to encode it.
#[derive(Debug, Clone, PartialEq)]
pub struct TdmScalar {
    /// The exact decimal or scientific-notation token read from the KVN record.
    pub text: String,
    /// The parsed finite `f64` value.
    pub value: f64,
}

/// Observable families used by TDM tracking data records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TdmObservable {
    /// A `RANGE` record.
    Range,
    /// A `DOPPLER_INSTANTANEOUS` record.
    DopplerInstantaneous,
    /// A `DOPPLER_INTEGRATED` record.
    DopplerIntegrated,
    /// A `RECEIVE_FREQ` or `RECEIVE_FREQ_n` record.
    ReceiveFreq {
        /// The participant suffix from `RECEIVE_FREQ_n`, if present.
        participant: Option<u8>,
    },
    /// A `TRANSMIT_FREQ` or `TRANSMIT_FREQ_n` record.
    TransmitFreq {
        /// The participant suffix from `TRANSMIT_FREQ_n`, if present.
        participant: Option<u8>,
    },
    /// A `TRANSMIT_FREQ_RATE` or `TRANSMIT_FREQ_RATE_n` record.
    TransmitFreqRate {
        /// The participant suffix from `TRANSMIT_FREQ_RATE_n`, if present.
        participant: Option<u8>,
    },
    /// An `ANGLE_1` record.
    Angle1,
    /// An `ANGLE_2` record.
    Angle2,
    /// A TDM data keyword not modeled as a dedicated enum variant.
    Other(String),
}

/// Units attached to TDM data records.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TdmUnit {
    /// Kilometers.
    Kilometers,
    /// Seconds.
    Seconds,
    /// CCSDS range units.
    RangeUnits,
    /// Kilometers per second.
    KilometersPerSecond,
    /// Hertz.
    Hertz,
    /// Hertz per second.
    HertzPerSecond,
    /// Degrees.
    Degrees,
    /// Decibel watts.
    DecibelWatts,
    /// Decibel hertz.
    DecibelHertz,
    /// Square meters.
    SquareMeters,
    /// Meters.
    Meters,
    /// Seconds per second.
    SecondsPerSecond,
    /// Percent.
    Percent,
    /// Kelvin.
    Kelvin,
    /// Hectopascals.
    Hectopascals,
    /// Total electron content units.
    TotalElectronContentUnits,
    /// Dimensionless quantity.
    Dimensionless,
    /// A unit label not modeled by this enum.
    Unknown(String),
}

impl TdmUnit {
    /// Return the canonical unit label.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Kilometers => "km",
            Self::Seconds => "s",
            Self::RangeUnits => "RU",
            Self::KilometersPerSecond => "km/s",
            Self::Hertz => "Hz",
            Self::HertzPerSecond => "Hz/s",
            Self::Degrees => "deg",
            Self::DecibelWatts => "dBW",
            Self::DecibelHertz => "dBHz",
            Self::SquareMeters => "m**2",
            Self::Meters => "m",
            Self::SecondsPerSecond => "s/s",
            Self::Percent => "%",
            Self::Kelvin => "K",
            Self::Hectopascals => "hPa",
            Self::TotalElectronContentUnits => "TECU",
            Self::Dimensionless => "n/a",
            Self::Unknown(label) => label.as_str(),
        }
    }
}

/// Boundary validation failure category for TDM parsing and encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TdmInputErrorKind {
    /// A required field or token was absent.
    Missing,
    /// A floating-point value could not be parsed.
    FloatParse,
    /// A floating-point value was NaN or infinite.
    NonFinite,
    /// A positive field was zero or negative.
    NotPositive,
    /// A numeric value was outside the CCSDS domain for that keyword.
    OutOfRange,
    /// An indexed keyword or path component did not contain a valid integer.
    InvalidIndex,
    /// A TDM data keyword is not defined by CCSDS 503.0-B-2 table 3-5.
    UnknownKeyword,
    /// A displayed unit was present even though TDM KVN units are table-defined.
    UnexpectedUnit,
    /// An integer-valued field contained a fractional value.
    NonInteger,
    /// A non-negative field contained a negative value.
    Negative,
    /// A numeric token used a negative zero form.
    NegativeZero,
    /// A record unit does not match CCSDS 503.0-B-2 table 3-5.
    UnitMismatch,
    /// The stored decimal token and `f64` value do not parse to the same bits.
    DecimalMismatch,
}

impl fmt::Display for TdmInputErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let label = match self {
            Self::Missing => "missing",
            Self::FloatParse => "invalid float",
            Self::NonFinite => "not finite",
            Self::NotPositive => "not positive",
            Self::OutOfRange => "out of range",
            Self::InvalidIndex => "invalid index",
            Self::UnknownKeyword => "unknown keyword",
            Self::UnexpectedUnit => "unexpected unit",
            Self::NonInteger => "not an integer",
            Self::Negative => "negative",
            Self::NegativeZero => "negative zero",
            Self::UnitMismatch => "unit mismatch",
            Self::DecimalMismatch => "decimal mismatch",
        };
        f.write_str(label)
    }
}

/// Failure modes for TDM KVN parsing and encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TdmError {
    /// The `CCSDS_TDM_VERS` header value was missing.
    MissingVersion,
    /// The message contained no complete metadata/data segment.
    NoSegments,
    /// A section marker appeared in an invalid location.
    Section {
        /// One-based input line number.
        line: usize,
        /// The section validation detail.
        detail: &'static str,
    },
    /// A non-comment line was not a valid KVN assignment or section marker.
    ///
    /// [`encode_kvn`] returns this for a field the KVN form cannot carry, where
    /// `line` counts the encoding it refused to emit rather than an input line.
    MalformedLine {
        /// One-based line number, in the input when parsing and in the output
        /// when encoding.
        line: usize,
        /// The offending line.
        text: String,
    },
    /// A data record did not contain `epoch value`.
    MalformedRecord {
        /// One-based input line number.
        line: usize,
        /// The offending data keyword.
        keyword: String,
    },
    /// A field failed numeric or indexed-keyword validation.
    InvalidField {
        /// The offending field name.
        field: String,
        /// The validation failure category.
        kind: TdmInputErrorKind,
    },
}

impl fmt::Display for TdmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingVersion => write!(f, "missing {VERSION_KEY}"),
            Self::NoSegments => write!(f, "missing TDM segment"),
            Self::Section { line, detail } => {
                write!(f, "invalid TDM section at line {line}: {detail}")
            }
            Self::MalformedLine { line, text } => {
                write!(f, "malformed TDM KVN line {line}: {text}")
            }
            Self::MalformedRecord { line, keyword } => {
                write!(f, "malformed TDM data record {keyword} at line {line}")
            }
            Self::InvalidField { field, kind } => write!(f, "invalid TDM field {field}: {kind}"),
        }
    }
}

impl std::error::Error for TdmError {}

#[derive(Default)]
struct HeaderBuilder {
    version: Option<String>,
    comments: Vec<String>,
    creation_date: Option<String>,
    originator: Option<String>,
    message_id: Option<String>,
    fields: Vec<TdmField>,
}

#[derive(Default)]
struct MetadataBuilder {
    comments: Vec<String>,
    fields: Vec<TdmField>,
}

#[derive(Default)]
struct DataBuilder {
    comments: Vec<String>,
    records: Vec<TdmDataRecord>,
}

/// Parse a TDM in CCSDS KVN format.
///
/// The parser accepts flexible whitespace around `=` and between the epoch and
/// value tokens. It requires complete `META_START` / `META_STOP` and
/// `DATA_START` / `DATA_STOP` blocks, and every data record with a numeric
/// keyword must contain a finite value. Frequency records are not converted to
/// range rate and keep their original decimal token for later serialization.
///
/// A line whose keyword is `COMMENT` is a comment, never an assignment: CCSDS
/// 503.0-B-2 4.2.5 c) excepts `COMMENT` from the KVN syntax, and 4.5.3 requires
/// at least one space after the keyword. `COMMENT=value` satisfies neither form
/// and is refused as a malformed line.
pub fn parse_kvn(text: &str) -> Result<Tdm, TdmError> {
    let mut header = HeaderBuilder::default();
    let mut metadata: Option<MetadataBuilder> = None;
    let mut pending_metadata: Option<TdmMetadata> = None;
    let mut data: Option<DataBuilder> = None;
    let mut segments = Vec::new();

    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = idx + 1;
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(comment) = comment_text(line) {
            if let Some(builder) = data.as_mut() {
                builder.comments.push(comment);
            } else if let Some(builder) = metadata.as_mut() {
                builder.comments.push(comment);
            } else if pending_metadata.is_none() {
                header.comments.push(comment);
            } else {
                return Err(TdmError::Section {
                    line: line_no,
                    detail: "comment between metadata and data",
                });
            }
            continue;
        }

        match line {
            "META_START" => {
                if metadata.is_some() || data.is_some() || pending_metadata.is_some() {
                    return Err(TdmError::Section {
                        line: line_no,
                        detail: "nested metadata block",
                    });
                }
                metadata = Some(MetadataBuilder::default());
                continue;
            }
            "META_STOP" => {
                let builder = metadata.take().ok_or(TdmError::Section {
                    line: line_no,
                    detail: "metadata stop without metadata start",
                })?;
                pending_metadata = Some(build_metadata(builder)?);
                continue;
            }
            "DATA_START" => {
                if metadata.is_some() || data.is_some() || pending_metadata.is_none() {
                    return Err(TdmError::Section {
                        line: line_no,
                        detail: "data start without completed metadata",
                    });
                }
                data = Some(DataBuilder::default());
                continue;
            }
            "DATA_STOP" => {
                let builder = data.take().ok_or(TdmError::Section {
                    line: line_no,
                    detail: "data stop without data start",
                })?;
                let metadata = pending_metadata.take().ok_or(TdmError::Section {
                    line: line_no,
                    detail: "data stop without metadata",
                })?;
                segments.push(TdmSegment {
                    metadata,
                    data: TdmDataSection {
                        comments: builder.comments,
                        records: builder.records,
                    },
                });
                continue;
            }
            _ => {}
        }

        let (key, value) = parse_assignment(line).ok_or_else(|| TdmError::MalformedLine {
            line: line_no,
            text: line.to_string(),
        })?;

        // `COMMENT` is excepted from the KVN syntax (4.2.5 c)) and a comment
        // line needs a space after the keyword (4.5.3), so `COMMENT=value` is
        // neither a comment nor an assignment. Keeping it as a field produced a
        // value whose encoding, `COMMENT = value`, reparsed as a comment.
        if key == COMMENT_KEY {
            return Err(TdmError::MalformedLine {
                line: line_no,
                text: line.to_string(),
            });
        }

        if let Some(builder) = data.as_mut() {
            let range_units = pending_metadata
                .as_ref()
                .map(|metadata| metadata.range_units.clone())
                .unwrap_or(TdmUnit::Kilometers);
            builder
                .records
                .push(parse_record(line_no, &key, &value, &range_units)?);
        } else if let Some(builder) = metadata.as_mut() {
            builder.fields.push(TdmField { key, value });
        } else if pending_metadata.is_none() {
            parse_header_field(&mut header, key, value);
        } else {
            return Err(TdmError::Section {
                line: line_no,
                detail: "field between metadata and data",
            });
        }
    }

    if metadata.is_some() {
        return Err(TdmError::Section {
            line: text.lines().count().saturating_add(1),
            detail: "unclosed metadata block",
        });
    }
    if data.is_some() {
        return Err(TdmError::Section {
            line: text.lines().count().saturating_add(1),
            detail: "unclosed data block",
        });
    }
    if pending_metadata.is_some() {
        return Err(TdmError::Section {
            line: text.lines().count().saturating_add(1),
            detail: "metadata without data block",
        });
    }

    let version = header
        .version
        .filter(|value| !value.is_empty())
        .ok_or(TdmError::MissingVersion)?;
    if segments.is_empty() {
        return Err(TdmError::NoSegments);
    }

    Ok(Tdm {
        version,
        comments: header.comments,
        creation_date: header.creation_date,
        originator: header.originator,
        message_id: header.message_id,
        header_fields: header.fields,
        segments,
    })
}

/// Encode a TDM to canonical CCSDS KVN text.
///
/// The output uses `KEY = VALUE` assignments and emits each data record as
/// `KEY = epoch decimal-token`. Record decimals are not reformatted. Encoding
/// validates that every stored decimal token parses back to the stored `f64`
/// bits, which keeps `RECEIVE_FREQ` and `TRANSMIT_FREQ_n` values lossless.
///
/// A header or metadata field keyed `COMMENT` is refused rather than written:
/// `COMMENT = value` reparses as a comment, so the encoding would not read back
/// as the value it was given.
pub fn encode_kvn(tdm: &Tdm) -> Result<String, TdmError> {
    validate_tdm(tdm)?;

    let mut lines = Vec::new();
    lines.push(format!("{VERSION_KEY} = {}", tdm.version));
    lines.extend(tdm.comments.iter().map(comment_line));
    if let Some(creation_date) = &tdm.creation_date {
        lines.push(format!("CREATION_DATE = {creation_date}"));
    }
    if let Some(originator) = &tdm.originator {
        lines.push(format!("ORIGINATOR = {originator}"));
    }
    if let Some(message_id) = &tdm.message_id {
        lines.push(format!("MESSAGE_ID = {message_id}"));
    }
    for field in &tdm.header_fields {
        push_field(&mut lines, field)?;
    }

    for segment in &tdm.segments {
        lines.push("META_START".to_string());
        lines.extend(segment.metadata.comments.iter().map(comment_line));
        for field in &segment.metadata.fields {
            push_field(&mut lines, field)?;
        }
        lines.push("META_STOP".to_string());
        lines.push("DATA_START".to_string());
        lines.extend(segment.data.comments.iter().map(comment_line));
        for record in &segment.data.records {
            lines.push(format!(
                "{} = {} {}",
                record.keyword, record.epoch, record.value.text
            ));
        }
        lines.push("DATA_STOP".to_string());
    }

    Ok(lines.join("\n"))
}

fn parse_header_field(header: &mut HeaderBuilder, key: String, value: String) {
    match key.as_str() {
        VERSION_KEY => header.version = Some(value),
        "CREATION_DATE" => header.creation_date = empty_to_none(value),
        "ORIGINATOR" => header.originator = empty_to_none(value),
        "MESSAGE_ID" => header.message_id = empty_to_none(value),
        _ => header.fields.push(TdmField { key, value }),
    }
}

fn build_metadata(builder: MetadataBuilder) -> Result<TdmMetadata, TdmError> {
    let mut participants = Vec::new();
    let mut mode = None;
    let mut paths = Vec::new();
    let mut timetag_ref = None;
    let mut time_system = None;
    let mut range_units = TdmUnit::Kilometers;

    for field in &builder.fields {
        if let Some(index) = indexed_suffix(&field.key, "PARTICIPANT")? {
            participants.push(TdmParticipant {
                index,
                name: field.value.clone(),
            });
        } else if field.key == "MODE" {
            mode = empty_to_none(field.value.clone());
        } else if field.key == "PATH" || field.key.starts_with("PATH_") {
            paths.push(parse_path(field)?);
        } else if field.key == "TIMETAG_REF" {
            timetag_ref = empty_to_none(field.value.clone());
        } else if field.key == "TIME_SYSTEM" {
            time_system = empty_to_none(field.value.clone());
        } else if field.key == "RANGE_UNITS" && !field.value.is_empty() {
            range_units = range_unit_from_label(&field.value)?;
        }
    }

    Ok(TdmMetadata {
        comments: builder.comments,
        fields: builder.fields,
        participants,
        mode,
        paths,
        timetag_ref,
        time_system,
        range_units,
    })
}

fn parse_path(field: &TdmField) -> Result<TdmPath, TdmError> {
    let index = if field.key == "PATH" {
        None
    } else {
        Some(indexed_suffix(&field.key, "PATH")?.ok_or_else(|| invalid_index(&field.key))?)
    };
    let mut participants = Vec::new();
    for token in field.value.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            return Err(invalid_index(&field.key));
        }
        let value = trimmed
            .parse::<u8>()
            .map_err(|_| invalid_index(&field.key))?;
        participants.push(value);
    }
    if participants.is_empty() {
        return Err(invalid_index(&field.key));
    }
    Ok(TdmPath {
        key: field.key.clone(),
        index,
        participants,
    })
}

fn parse_record(
    line: usize,
    keyword: &str,
    value: &str,
    range_units: &TdmUnit,
) -> Result<TdmDataRecord, TdmError> {
    if has_displayed_unit(keyword) || has_displayed_unit(value) {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::UnexpectedUnit,
        });
    }

    let mut parts = value.split_whitespace();
    let epoch = parts
        .next()
        .ok_or_else(|| malformed_record(line, keyword))?;
    let value_text = parts
        .next()
        .ok_or_else(|| malformed_record(line, keyword))?;
    if parts.next().is_some() {
        return Err(malformed_record(line, keyword));
    }

    let observable = observable_from_keyword(keyword)?;
    let scalar = parse_scalar(keyword, value_text, &observable)?;
    validate_record_value(keyword, &observable, &scalar)?;
    let unit = unit_for_keyword(keyword, &observable, range_units);

    Ok(TdmDataRecord {
        observable,
        keyword: keyword.to_string(),
        epoch: epoch.to_string(),
        value: scalar,
        unit,
    })
}

fn parse_scalar(
    field: &str,
    text: &str,
    observable: &TdmObservable,
) -> Result<TdmScalar, TdmError> {
    if is_nonfinite_float_token(text) {
        return Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::NonFinite,
        });
    }
    validate_numeric_token(field, text, observable)?;
    let value = text.parse::<f64>().map_err(|_| TdmError::InvalidField {
        field: field.to_string(),
        kind: TdmInputErrorKind::FloatParse,
    })?;
    if !value.is_finite() {
        return Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::NonFinite,
        });
    }
    let lexical_zero = numeric_token_is_zero(text);
    if !lexical_zero && decimal_magnitude_below_minimum_positive_double(text) {
        return Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    if value == 0.0 && text.trim_start().starts_with('-') {
        return Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::NegativeZero,
        });
    }
    if value == 0.0 && !lexical_zero {
        return Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    Ok(TdmScalar {
        text: text.to_string(),
        value,
    })
}

fn is_nonfinite_float_token(text: &str) -> bool {
    matches!(
        text,
        "NaN" | "+NaN" | "-NaN" | "Inf" | "+Inf" | "-Inf" | "Infinity" | "+Infinity" | "-Infinity"
    )
}

fn validate_numeric_token(
    field: &str,
    text: &str,
    observable: &TdmObservable,
) -> Result<(), TdmError> {
    if matches!(observable, TdmObservable::Other(name) if name == "DOPPLER_COUNT") {
        validate_integer_token(field, text)
    } else if phase_count_keyword(field) {
        validate_phase_count_token(field, text)
    } else if is_ccsds_double_token(text) {
        Ok(())
    } else {
        Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::FloatParse,
        })
    }
}

fn validate_integer_token(field: &str, text: &str) -> Result<(), TdmError> {
    let digits = strip_ascii_sign(text);
    if digits.is_empty() {
        return Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::NonInteger,
        });
    }
    if !digits.chars().all(|character| character.is_ascii_digit()) {
        return Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::NonInteger,
        });
    }
    let value = text.parse::<i64>().map_err(|_| TdmError::InvalidField {
        field: field.to_string(),
        kind: TdmInputErrorKind::OutOfRange,
    })?;
    if !(i64::from(i32::MIN)..=i64::from(i32::MAX)).contains(&value) {
        return Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    Ok(())
}

fn validate_phase_count_token(field: &str, text: &str) -> Result<(), TdmError> {
    if is_phase_count_token(text) {
        Ok(())
    } else {
        Err(TdmError::InvalidField {
            field: field.to_string(),
            kind: TdmInputErrorKind::FloatParse,
        })
    }
}

fn is_ccsds_double_token(text: &str) -> bool {
    let Some(unsigned) = strip_optional_sign(text) else {
        return false;
    };
    is_fixed_point_token(unsigned, Some(16)) || is_floating_point_token(unsigned, Some(16))
}

fn is_phase_count_token(text: &str) -> bool {
    is_unsigned_integer(text) || is_fixed_point_token(text, None)
}

fn strip_optional_sign(text: &str) -> Option<&str> {
    let unsigned = strip_ascii_sign(text);
    (!unsigned.is_empty()).then_some(unsigned)
}

fn strip_ascii_sign(text: &str) -> &str {
    match text.as_bytes().first() {
        Some(b'+') | Some(b'-') => &text[1..],
        _ => text,
    }
}

fn is_unsigned_integer(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|character| character.is_ascii_digit())
}

fn is_fixed_point_token(text: &str, max_digits: Option<usize>) -> bool {
    let Some((integer, fraction)) = text.split_once('.') else {
        return false;
    };
    if integer.is_empty()
        || fraction.is_empty()
        || fraction.contains('.')
        || !is_unsigned_integer(integer)
        || !is_unsigned_integer(fraction)
    {
        return false;
    }
    match max_digits {
        Some(max) => integer.len() + fraction.len() <= max,
        None => true,
    }
}

fn is_floating_point_token(text: &str, max_digits: Option<usize>) -> bool {
    let Some(exponent_index) = text.find(['E', 'e']) else {
        return false;
    };
    let mantissa = &text[..exponent_index];
    let exponent = &text[exponent_index + 1..];
    if exponent.is_empty() || exponent.find(['E', 'e']).is_some() {
        return false;
    }
    let exponent_digits = strip_ascii_sign(exponent);
    if exponent_digits.is_empty()
        || !exponent_digits
            .chars()
            .all(|character| character.is_ascii_digit())
    {
        return false;
    }
    let mut mantissa_chars = mantissa.chars();
    let Some(integer) = mantissa_chars.next() else {
        return false;
    };
    if !integer.is_ascii_digit() || mantissa_chars.next() != Some('.') {
        return false;
    }
    let fraction = mantissa_chars.as_str();
    if fraction.is_empty() || !is_unsigned_integer(fraction) {
        return false;
    }
    match max_digits {
        Some(max) => fraction.len() < max,
        None => true,
    }
}

fn numeric_token_is_zero(text: &str) -> bool {
    let unsigned = strip_ascii_sign(text);
    let mantissa = unsigned
        .find(['E', 'e'])
        .map_or(unsigned, |exponent_index| &unsigned[..exponent_index]);
    !mantissa.is_empty()
        && mantissa
            .bytes()
            .filter(|byte| *byte != b'.')
            .all(|byte| byte == b'0')
}

fn decimal_magnitude_below_minimum_positive_double(text: &str) -> bool {
    const MIN_POSITIVE_EXPONENT: i32 = -324;
    const MIN_POSITIVE_SIGNIFICAND_16: &[u8; 16] = b"4940000000000000";

    let Some((exponent, significand)) = normalized_decimal_parts(text) else {
        return false;
    };
    if exponent < MIN_POSITIVE_EXPONENT {
        return true;
    }
    if exponent > MIN_POSITIVE_EXPONENT {
        return false;
    }
    let significand = significand.as_bytes();
    for (index, minimum) in MIN_POSITIVE_SIGNIFICAND_16.iter().enumerate() {
        let digit = significand.get(index).copied().unwrap_or(b'0');
        if digit != *minimum {
            return digit < *minimum;
        }
    }
    false
}

fn normalized_decimal_parts(text: &str) -> Option<(i32, String)> {
    let unsigned = strip_ascii_sign(text);
    let (mantissa, exponent_adjust) = if let Some(exponent_index) = unsigned.find(['E', 'e']) {
        (
            &unsigned[..exponent_index],
            parse_exponent_for_bound(&unsigned[exponent_index + 1..]),
        )
    } else {
        (unsigned, 0)
    };
    let (integer, fraction) = mantissa.split_once('.')?;
    let decimal_index = i32::try_from(integer.len()).ok()?;
    let mut digits = String::with_capacity(integer.len() + fraction.len());
    digits.push_str(integer);
    digits.push_str(fraction);
    let leading = digits.bytes().position(|byte| byte != b'0')?;
    let leading = i32::try_from(leading).ok()?;
    let exponent = exponent_adjust + decimal_index - leading - 1;
    Some((exponent, digits[leading as usize..].to_string()))
}

fn parse_exponent_for_bound(text: &str) -> i32 {
    let negative = text.starts_with('-');
    let digits = strip_ascii_sign(text);
    let digits = digits.trim_start_matches('0');
    if digits.len() > 4 {
        return if negative { -10_000 } else { 10_000 };
    }
    let value = digits.parse::<i32>().unwrap_or(0);
    if negative {
        -value
    } else {
        value
    }
}

fn observable_from_keyword(keyword: &str) -> Result<TdmObservable, TdmError> {
    match keyword {
        "RANGE" => Ok(TdmObservable::Range),
        "DOPPLER_INSTANTANEOUS" => Ok(TdmObservable::DopplerInstantaneous),
        "DOPPLER_INTEGRATED" => Ok(TdmObservable::DopplerIntegrated),
        "ANGLE_1" => Ok(TdmObservable::Angle1),
        "ANGLE_2" => Ok(TdmObservable::Angle2),
        "RECEIVE_FREQ" => Ok(TdmObservable::ReceiveFreq { participant: None }),
        _ => {
            if let Some(participant) = indexed_suffix_in_range(keyword, "RECEIVE_FREQ", 1, 5)? {
                Ok(TdmObservable::ReceiveFreq {
                    participant: Some(participant),
                })
            } else if let Some(participant) =
                indexed_suffix_in_range(keyword, "TRANSMIT_FREQ_RATE", 1, 5)?
            {
                Ok(TdmObservable::TransmitFreqRate {
                    participant: Some(participant),
                })
            } else if let Some(participant) =
                indexed_suffix_in_range(keyword, "TRANSMIT_FREQ", 1, 5)?
            {
                Ok(TdmObservable::TransmitFreq {
                    participant: Some(participant),
                })
            } else if known_table_3_5_other_keyword(keyword)? {
                Ok(TdmObservable::Other(keyword.to_string()))
            } else {
                Err(unknown_keyword(keyword))
            }
        }
    }
}

fn validate_record_value(
    keyword: &str,
    observable: &TdmObservable,
    scalar: &TdmScalar,
) -> Result<(), TdmError> {
    let value = scalar.value;
    if value == 0.0 && scalar.text.trim_start().starts_with('-') {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::NegativeZero,
        });
    }
    if matches!(observable, TdmObservable::TransmitFreq { .. }) && value <= 0.0 {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::NotPositive,
        });
    }
    if matches!(observable, TdmObservable::Other(name) if name == "DOPPLER_COUNT") {
        validate_doppler_count(keyword, scalar)?;
    }
    if matches!(observable, TdmObservable::Other(name) if name == "RCS" || name == "STEC" || name == "TEMPERATURE")
        && value <= 0.0
    {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::NotPositive,
        });
    }
    if matches!(observable, TdmObservable::Other(name) if name == "TROPO_DRY" || name == "TROPO_WET")
        && value < 0.0
    {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::Negative,
        });
    }
    if matches!(observable, TdmObservable::Other(name) if name == "RHUMIDITY")
        && !(0.0..=100.0).contains(&value)
    {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    if matches!(observable, TdmObservable::Angle1 | TdmObservable::Angle2)
        && !(-180.0..360.0).contains(&value)
    {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    Ok(())
}

fn validate_doppler_count(keyword: &str, scalar: &TdmScalar) -> Result<(), TdmError> {
    let text = scalar.text.trim_start();
    if text.starts_with('-') {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::Negative,
        });
    }
    let digits = text.strip_prefix('+').unwrap_or(text);
    if digits.is_empty() || !digits.chars().all(|character| character.is_ascii_digit()) {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::NonInteger,
        });
    }
    let count = digits.parse::<u64>().map_err(|_| TdmError::InvalidField {
        field: keyword.to_string(),
        kind: TdmInputErrorKind::OutOfRange,
    })?;
    if count > i32::MAX as u64 {
        return Err(TdmError::InvalidField {
            field: keyword.to_string(),
            kind: TdmInputErrorKind::OutOfRange,
        });
    }
    Ok(())
}

fn validate_tdm(tdm: &Tdm) -> Result<(), TdmError> {
    if tdm.version.is_empty() {
        return Err(TdmError::MissingVersion);
    }
    if tdm.segments.is_empty() {
        return Err(TdmError::NoSegments);
    }
    for segment in &tdm.segments {
        for record in &segment.data.records {
            if !record.value.value.is_finite() {
                return Err(TdmError::InvalidField {
                    field: record.keyword.clone(),
                    kind: TdmInputErrorKind::NonFinite,
                });
            }
            let observable = observable_from_keyword(&record.keyword)?;
            let parsed = parse_scalar(&record.keyword, &record.value.text, &observable)?;
            if parsed.value.to_bits() != record.value.value.to_bits() {
                return Err(TdmError::InvalidField {
                    field: record.keyword.clone(),
                    kind: TdmInputErrorKind::DecimalMismatch,
                });
            }
            if observable != record.observable {
                return Err(TdmError::InvalidField {
                    field: record.keyword.clone(),
                    kind: TdmInputErrorKind::UnknownKeyword,
                });
            }
            let expected_unit = unit_for_keyword(
                &record.keyword,
                &record.observable,
                &segment.metadata.range_units,
            );
            if expected_unit != record.unit {
                return Err(TdmError::InvalidField {
                    field: record.keyword.clone(),
                    kind: TdmInputErrorKind::UnitMismatch,
                });
            }
            validate_record_value(&record.keyword, &record.observable, &record.value)?;
        }
    }
    Ok(())
}

fn parse_assignment(line: &str) -> Option<(String, String)> {
    let (key, raw_value) = line.split_once('=')?;
    let key = key.trim().to_string();
    Some((key, raw_value.trim().to_string()))
}

fn comment_text(line: &str) -> Option<String> {
    if line == COMMENT_KEY {
        return Some(String::new());
    }
    let rest = line.strip_prefix(COMMENT_KEY)?;
    if rest
        .chars()
        .next()
        .is_some_and(|character| character.is_ascii_whitespace())
    {
        Some(rest.trim_start().to_string())
    } else {
        None
    }
}

fn comment_line(comment: &String) -> String {
    if comment.is_empty() {
        COMMENT_KEY.to_string()
    } else {
        format!("{COMMENT_KEY} {comment}")
    }
}

fn field_line(field: &TdmField) -> String {
    format!("{} = {}", field.key, field.value)
}

/// Append one `KEY = VALUE` line, refusing a field the KVN form cannot carry.
///
/// `COMMENT` is excepted from the KVN syntax by CCSDS 503.0-B-2 4.2.5 c), and
/// 4.5.3 reads any line whose keyword is followed by a space as a comment. A
/// field keyed `COMMENT` has no assignment form: writing `COMMENT = value`
/// emits a line that reads back as a comment rather than as the field.
fn push_field(lines: &mut Vec<String>, field: &TdmField) -> Result<(), TdmError> {
    let text = field_line(field);
    // The key is compared trimmed: a key of `"COMMENT "` writes the same
    // line, which reads back as a comment just as an untrimmed one does.
    if field.key.trim() == COMMENT_KEY {
        return Err(TdmError::MalformedLine {
            line: lines.len().saturating_add(1),
            text,
        });
    }
    lines.push(text);
    Ok(())
}

fn empty_to_none(value: String) -> Option<String> {
    (!value.is_empty()).then_some(value)
}

fn malformed_record(line: usize, keyword: &str) -> TdmError {
    TdmError::MalformedRecord {
        line,
        keyword: keyword.to_string(),
    }
}

fn has_displayed_unit(value: &str) -> bool {
    let trimmed = value.trim_end();
    trimmed.ends_with(']') && trimmed.rfind('[').is_some()
}

fn indexed_suffix(key: &str, base: &str) -> Result<Option<u8>, TdmError> {
    let Some(suffix) = key
        .strip_prefix(base)
        .and_then(|rest| rest.strip_prefix('_'))
    else {
        return Ok(None);
    };
    if suffix.is_empty() || !suffix.chars().all(|character| character.is_ascii_digit()) {
        return Err(invalid_index(key));
    }
    suffix
        .parse::<u8>()
        .map(Some)
        .map_err(|_| invalid_index(key))
}

fn indexed_suffix_in_range(
    key: &str,
    base: &str,
    min: u8,
    max: u8,
) -> Result<Option<u8>, TdmError> {
    let Some(index) = indexed_suffix(key, base)? else {
        return Ok(None);
    };
    if (min..=max).contains(&index) {
        Ok(Some(index))
    } else {
        Err(invalid_index(key))
    }
}

fn invalid_index(field: &str) -> TdmError {
    TdmError::InvalidField {
        field: field.to_string(),
        kind: TdmInputErrorKind::InvalidIndex,
    }
}

fn unknown_keyword(field: &str) -> TdmError {
    TdmError::InvalidField {
        field: field.to_string(),
        kind: TdmInputErrorKind::UnknownKeyword,
    }
}

fn range_unit_from_label(label: &str) -> Result<TdmUnit, TdmError> {
    match label {
        "km" => Ok(TdmUnit::Kilometers),
        "s" => Ok(TdmUnit::Seconds),
        "RU" => Ok(TdmUnit::RangeUnits),
        _ => Err(TdmError::InvalidField {
            field: "RANGE_UNITS".to_string(),
            kind: TdmInputErrorKind::UnitMismatch,
        }),
    }
}

fn unit_for_keyword(keyword: &str, observable: &TdmObservable, range_units: &TdmUnit) -> TdmUnit {
    match observable {
        TdmObservable::Range => range_units.clone(),
        TdmObservable::DopplerInstantaneous | TdmObservable::DopplerIntegrated => {
            TdmUnit::KilometersPerSecond
        }
        TdmObservable::ReceiveFreq { .. } | TdmObservable::TransmitFreq { .. } => TdmUnit::Hertz,
        TdmObservable::TransmitFreqRate { .. } => TdmUnit::HertzPerSecond,
        TdmObservable::Angle1 | TdmObservable::Angle2 => TdmUnit::Degrees,
        TdmObservable::Other(_) => unit_for_other_keyword(keyword),
    }
}

fn unit_for_other_keyword(keyword: &str) -> TdmUnit {
    if indexed_suffix_in_range(keyword, "RECEIVE_PHASE_CT", 1, 5).is_ok_and(|value| value.is_some())
        || indexed_suffix_in_range(keyword, "TRANSMIT_PHASE_CT", 1, 5)
            .is_ok_and(|value| value.is_some())
    {
        return TdmUnit::Dimensionless;
    }
    match keyword {
        "CARRIER_POWER" => TdmUnit::DecibelWatts,
        "CLOCK_BIAS" | "DOR" | "VLBI_DELAY" => TdmUnit::Seconds,
        "CLOCK_DRIFT" => TdmUnit::SecondsPerSecond,
        "DOPPLER_COUNT" | "MAG" => TdmUnit::Dimensionless,
        "PC_N0" | "PR_N0" => TdmUnit::DecibelHertz,
        "PRESSURE" => TdmUnit::Hectopascals,
        "RCS" => TdmUnit::SquareMeters,
        "RHUMIDITY" => TdmUnit::Percent,
        "STEC" => TdmUnit::TotalElectronContentUnits,
        "TEMPERATURE" => TdmUnit::Kelvin,
        "TROPO_DRY" | "TROPO_WET" => TdmUnit::Meters,
        _ => unreachable!("table 3-5 keyword checked before unit lookup"),
    }
}

fn phase_count_keyword(keyword: &str) -> bool {
    keyword.starts_with("RECEIVE_PHASE_CT_") || keyword.starts_with("TRANSMIT_PHASE_CT_")
}

fn known_table_3_5_other_keyword(keyword: &str) -> Result<bool, TdmError> {
    if indexed_suffix_in_range(keyword, "RECEIVE_PHASE_CT", 1, 5)?.is_some()
        || indexed_suffix_in_range(keyword, "TRANSMIT_PHASE_CT", 1, 5)?.is_some()
    {
        return Ok(true);
    }

    Ok(matches!(
        keyword,
        "CARRIER_POWER"
            | "CLOCK_BIAS"
            | "CLOCK_DRIFT"
            | "DOPPLER_COUNT"
            | "DOR"
            | "MAG"
            | "PC_N0"
            | "PR_N0"
            | "PRESSURE"
            | "RCS"
            | "RHUMIDITY"
            | "STEC"
            | "TEMPERATURE"
            | "TROPO_DRY"
            | "TROPO_WET"
            | "VLBI_DELAY"
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE: &str = "\
CCSDS_TDM_VERS = 2.0
COMMENT sample
CREATION_DATE = 2005-160T20:15:00Z
ORIGINATOR = NASA
META_START
TIME_SYSTEM = UTC
PARTICIPANT_1 = DSS-25
PARTICIPANT_2 = yyyy-nnnA
MODE = SEQUENTIAL
PATH = 2,1
RANGE_UNITS = km
META_STOP
DATA_START
TRANSMIT_FREQ_2 = 2005-159T17:41:00 32023442781.733
RECEIVE_FREQ_1 = 2005-159T17:41:00 32021034790.7265
RANGE = 2005-159T17:41:00 80452.7542
ANGLE_1 = 2005-159T17:41:00 256.64002393
ANGLE_2 = 2005-159T17:41:00 13.38100016
DATA_STOP";

    #[test]
    fn parses_frequency_records_without_reformatting_decimal_tokens() {
        let tdm = parse_kvn(SIMPLE).unwrap();
        let records = &tdm.segments[0].data.records;
        assert_eq!(records[0].keyword, "TRANSMIT_FREQ_2");
        assert_eq!(records[0].value.text, "32023442781.733");
        assert_eq!(records[0].value.value.to_bits(), 0x421d_d2fb_d576_ee98);
        assert_eq!(records[0].unit, TdmUnit::Hertz);
        assert_eq!(records[1].keyword, "RECEIVE_FREQ_1");
        assert_eq!(records[1].value.text, "32021034790.7265");
        assert_eq!(records[1].value.value.to_bits(), 0x421d_d268_dc9a_e7f0);
    }

    #[test]
    fn canonical_encode_is_stable() {
        let tdm = parse_kvn(SIMPLE).unwrap();
        let encoded = encode_kvn(&tdm).unwrap();
        let reparsed = parse_kvn(&encoded).unwrap();
        assert_eq!(encode_kvn(&reparsed).unwrap(), encoded);
        assert_eq!(reparsed, tdm);
    }

    #[test]
    fn malformed_data_record_is_typed_error() {
        let err = parse_kvn(
            "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RECEIVE_FREQ_1 = 2005-159T17:41:00
DATA_STOP",
        )
        .unwrap_err();
        assert_eq!(
            err,
            TdmError::MalformedRecord {
                line: 6,
                keyword: "RECEIVE_FREQ_1".to_string()
            }
        );
    }

    #[test]
    fn invalid_transmit_frequency_is_rejected() {
        let err = parse_kvn(
            "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
TRANSMIT_FREQ_1 = 2005-159T17:41:00 0.0
DATA_STOP",
        )
        .unwrap_err();
        assert_eq!(
            err,
            TdmError::InvalidField {
                field: "TRANSMIT_FREQ_1".to_string(),
                kind: TdmInputErrorKind::NotPositive,
            }
        );
    }

    #[test]
    fn comment_assignment_is_refused_in_header_and_metadata() {
        let header = "\
CCSDS_TDM_VERS = 2.0
COMMENT=
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP";
        assert_eq!(
            parse_kvn(header),
            Err(TdmError::MalformedLine {
                line: 2,
                text: "COMMENT=".to_string(),
            })
        );

        let metadata = "\
CCSDS_TDM_VERS = 2.0
META_START
TIME_SYSTEM = UTC
COMMENT=file = tdm.dat
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP";
        assert_eq!(
            parse_kvn(metadata),
            Err(TdmError::MalformedLine {
                line: 4,
                text: "COMMENT=file = tdm.dat".to_string(),
            })
        );
    }

    #[test]
    fn comment_keyword_followed_by_a_space_stays_a_comment() {
        // 4.5.3 takes the rest of the line as the comment value, so a comment
        // whose text opens with `=` is still a comment, not an assignment.
        let tdm = parse_kvn(
            "\
CCSDS_TDM_VERS = 2.0
COMMENT = file = tdm.dat
META_START
TIME_SYSTEM = UTC
META_STOP
DATA_START
RANGE = 2005-159T17:41:00 1.0
DATA_STOP",
        )
        .unwrap();
        assert_eq!(tdm.comments, vec!["= file = tdm.dat".to_string()]);
        assert!(tdm.header_fields.is_empty());
    }

    #[test]
    fn encode_refuses_a_comment_keyed_field() {
        let comment_field = TdmField {
            key: COMMENT_KEY.to_string(),
            value: String::new(),
        };

        let mut header_case = parse_kvn(SIMPLE).unwrap();
        header_case.header_fields.push(comment_field.clone());
        assert_eq!(
            encode_kvn(&header_case),
            Err(TdmError::MalformedLine {
                line: 5,
                text: "COMMENT = ".to_string(),
            })
        );

        let mut metadata_case = parse_kvn(SIMPLE).unwrap();
        metadata_case.segments[0]
            .metadata
            .fields
            .push(comment_field);
        let err = encode_kvn(&metadata_case).unwrap_err();
        assert!(matches!(
            err,
            TdmError::MalformedLine { ref text, .. } if text == "COMMENT = "
        ));
    }
}
