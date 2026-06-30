//! CCSDS Orbit Ephemeris Message (OEM) KVN and XML reader/writer.
//!
//! OEM date/time values are carried as raw strings. The reader does not resolve
//! time systems or normalize epochs, so state-vector lines round-trip through the
//! canonical IR without calendar rewriting.

use crate::astro::covariance::Covariance6;
use crate::astro::ndm::{read_covariance6, write_covariance6, FieldMap, NdmHeader};
use crate::astro::xml;
use crate::format::fmtnum::fmt_num;
use crate::format::tokens::Tokenizer;
use crate::format::{Diagnostics, RecordRef, Skip, SkipReason};
use crate::validate;
use roxmltree::{Document, Node};
use std::fmt;

const COMMENT_PREFIX: &str = "COMMENT";
const META_START: &str = "META_START";
const META_STOP: &str = "META_STOP";
const COVARIANCE_START: &str = "COVARIANCE_START";
const COVARIANCE_STOP: &str = "COVARIANCE_STOP";
const OEM_VERSION_KEY: &str = "CCSDS_OEM_VERS";

const METADATA_KEYS: [&str; 11] = [
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "TIME_SYSTEM",
    "START_TIME",
    "STOP_TIME",
    "USEABLE_START_TIME",
    "USEABLE_STOP_TIME",
    "INTERPOLATION",
    "INTERPOLATION_DEGREE",
];

const STATE_NUMBER_KEYS: [&str; 9] = [
    "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT", "X_DDOT", "Y_DDOT", "Z_DDOT",
];

/// Canonical, format-agnostic OEM container.
#[derive(Debug, Clone, PartialEq)]
pub struct Oem {
    pub ccsds_oem_vers: String,
    pub creation_date: Option<String>,
    pub originator: Option<String>,
    pub segments: Vec<OemSegment>,
    /// Forgiving-parse count of ephemeris data lines skipped as malformed.
    pub skipped_states: usize,
}

/// One OEM metadata/data segment.
#[derive(Debug, Clone, PartialEq)]
pub struct OemSegment {
    pub metadata: OemMetadata,
    pub states: Vec<OemState>,
    pub covariances: Vec<OemCovariance>,
}

/// OEM segment metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct OemMetadata {
    pub object_name: String,
    pub object_id: String,
    pub center_name: String,
    pub ref_frame: String,
    pub time_system: String,
    pub start_time: String,
    pub stop_time: String,
    pub useable_start_time: Option<String>,
    pub useable_stop_time: Option<String>,
    pub interpolation: Option<String>,
    pub interpolation_degree: Option<u32>,
}

/// One OEM Cartesian state sample.
#[derive(Debug, Clone, PartialEq)]
pub struct OemState {
    pub epoch: String,
    pub position_km: [f64; 3],
    pub velocity_km_s: [f64; 3],
    pub acceleration_km_s2: Option<[f64; 3]>,
}

/// One OEM covariance block.
#[derive(Debug, Clone, PartialEq)]
pub struct OemCovariance {
    pub epoch: String,
    pub cov_ref_frame: Option<String>,
    pub matrix: Covariance6,
}

/// Failure modes of the OEM readers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OemError {
    /// A required field was absent from the message.
    MissingField(&'static str),
    /// A decoded scalar field failed validation.
    InvalidField {
        field: &'static str,
        kind: OemInputErrorKind,
    },
    /// A structural or XML-level error.
    Field(String),
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
        }
    }
}

impl std::error::Error for OemError {}

/// Parse a CCSDS OEM in KVN encoding into an [`Oem`].
pub fn parse_kvn(text: &str) -> Result<Oem, OemError> {
    let lines = numbered_lines(text);
    let header_map = FieldMap::from_pairs(parse_kv_lines(
        &lines
            .iter()
            .map(|(_, line)| line.clone())
            .collect::<Vec<_>>(),
    ));
    let header = NdmHeader::read(&header_map, OEM_VERSION_KEY);
    if header.vers.is_empty() {
        return Err(OemError::MissingField(OEM_VERSION_KEY));
    }

    let mut diagnostics = Diagnostics::new();
    let mut segments = Vec::new();
    let mut idx = 0usize;

    while idx < lines.len() {
        let line = lines[idx].1.trim();
        if line == META_START {
            let (segment, next_idx) = parse_kvn_segment(&lines, idx, &mut diagnostics)?;
            segments.push(segment);
            idx = next_idx;
        } else {
            idx += 1;
        }
    }

    if segments.is_empty() {
        return Err(OemError::Field("OEM contains no segment".to_string()));
    }

    let skipped_states = diagnostics.skips.len();
    Ok(Oem {
        ccsds_oem_vers: header.vers,
        creation_date: header.creation_date,
        originator: header.originator,
        segments,
        skipped_states,
    })
}

/// Encode an [`Oem`] as CCSDS OEM KVN.
pub fn encode_kvn(oem: &Oem) -> String {
    let mut lines = NdmHeader {
        vers: oem.ccsds_oem_vers.clone(),
        creation_date: oem.creation_date.clone(),
        originator: oem.originator.clone(),
    }
    .write_kvn(OEM_VERSION_KEY);

    for segment in &oem.segments {
        lines.push(META_START.to_string());
        lines.extend(encode_metadata_kvn(&segment.metadata));
        lines.push(META_STOP.to_string());

        for state in &segment.states {
            lines.push(encode_state_kvn(state));
        }

        for covariance in &segment.covariances {
            lines.push(COVARIANCE_START.to_string());
            lines.push(format!("EPOCH = {}", covariance.epoch));
            if let Some(cov_ref_frame) = &covariance.cov_ref_frame {
                lines.push(format!("COV_REF_FRAME = {cov_ref_frame}"));
            }
            lines.extend(write_covariance6(&covariance.matrix));
            lines.push(COVARIANCE_STOP.to_string());
        }
    }

    lines.join("\n")
}

/// Parse a CCSDS OEM in XML encoding into an [`Oem`].
pub fn parse_xml(text: &str) -> Result<Oem, OemError> {
    let doc = Document::parse(text).map_err(|e| OemError::Field(format!("malformed XML: {e}")))?;
    let oem_node = doc
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "oem")
        .ok_or_else(|| OemError::Field("missing oem element".to_string()))?;

    let version = oem_node
        .attribute("version")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| node_text(oem_node, OEM_VERSION_KEY))
        .ok_or(OemError::MissingField(OEM_VERSION_KEY))?;

    let segments: Vec<OemSegment> = oem_node
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "segment")
        .map(parse_xml_segment)
        .collect::<Result<_, _>>()?;

    if segments.is_empty() {
        return Err(OemError::Field("OEM contains no segment".to_string()));
    }

    Ok(Oem {
        ccsds_oem_vers: version,
        creation_date: node_text(oem_node, "CREATION_DATE"),
        originator: node_text(oem_node, "ORIGINATOR"),
        segments,
        skipped_states: 0,
    })
}

/// Encode an [`Oem`] as CCSDS OEM XML.
pub fn encode_xml(oem: &Oem) -> String {
    let mut lines = vec![
        r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string(),
        format!(
            r#"<oem id="CCSDS_OEM_VERS" version="{}">"#,
            xml::escape(&oem.ccsds_oem_vers)
        ),
        "  <header>".to_string(),
        format!(
            "    <CREATION_DATE>{}</CREATION_DATE>",
            xml::escape_opt(&oem.creation_date)
        ),
        format!(
            "    <ORIGINATOR>{}</ORIGINATOR>",
            xml::escape_opt(&oem.originator)
        ),
        "  </header>".to_string(),
        "  <body>".to_string(),
    ];

    for segment in &oem.segments {
        lines.extend(encode_xml_segment(segment));
    }

    lines.push("  </body>".to_string());
    lines.push("</oem>".to_string());
    lines.join("\n")
}

fn parse_kvn_segment(
    lines: &[(usize, String)],
    start_idx: usize,
    diagnostics: &mut Diagnostics,
) -> Result<(OemSegment, usize), OemError> {
    let mut idx = start_idx + 1;
    let mut metadata_lines = Vec::new();
    while idx < lines.len() {
        let line = lines[idx].1.trim();
        if line == META_STOP {
            break;
        }
        metadata_lines.push(lines[idx].1.clone());
        idx += 1;
    }
    if idx >= lines.len() {
        return Err(OemError::Field("META_START without META_STOP".to_string()));
    }

    let metadata = parse_metadata(&FieldMap::from_pairs(parse_kv_lines(&metadata_lines)))?;
    idx += 1;

    let mut states = Vec::new();
    let mut covariances = Vec::new();
    while idx < lines.len() {
        let (line_no, raw) = &lines[idx];
        let line = raw.trim();

        if line == META_START {
            break;
        }
        if line.is_empty() || line.starts_with(COMMENT_PREFIX) {
            idx += 1;
            continue;
        }
        if line == COVARIANCE_START {
            let (covariance, next_idx) = parse_kvn_covariance(lines, idx)?;
            covariances.push(covariance);
            idx = next_idx;
            continue;
        }
        if line == COVARIANCE_STOP {
            return Err(OemError::Field(
                "COVARIANCE_STOP without COVARIANCE_START".to_string(),
            ));
        }

        match parse_state_line(line) {
            Ok(state) => states.push(state),
            Err(StateLineError::WrongTokenCount) => diagnostics.push_skip(Skip {
                at: RecordRef::at_line(*line_no),
                reason: SkipReason::Truncated,
            }),
            Err(StateLineError::Field(error)) => diagnostics.push_skip(Skip {
                at: RecordRef::at_line(*line_no),
                reason: SkipReason::MalformedField(error),
            }),
        }
        idx += 1;
    }

    Ok((
        OemSegment {
            metadata,
            states,
            covariances,
        },
        idx,
    ))
}

fn parse_kvn_covariance(
    lines: &[(usize, String)],
    start_idx: usize,
) -> Result<(OemCovariance, usize), OemError> {
    let mut idx = start_idx + 1;
    let mut covariance_lines = Vec::new();
    while idx < lines.len() {
        let line = lines[idx].1.trim();
        if line == COVARIANCE_STOP {
            let covariance =
                parse_covariance_map(&FieldMap::from_pairs(parse_kv_lines(&covariance_lines)))?;
            return Ok((covariance, idx + 1));
        }
        if !line.is_empty() && !line.starts_with(COMMENT_PREFIX) {
            covariance_lines.push(lines[idx].1.clone());
        }
        idx += 1;
    }

    Err(OemError::Field(
        "COVARIANCE_START without COVARIANCE_STOP".to_string(),
    ))
}

fn parse_metadata(map: &FieldMap) -> Result<OemMetadata, OemError> {
    Ok(OemMetadata {
        object_name: req_text(map, "OBJECT_NAME")?,
        object_id: req_text(map, "OBJECT_ID")?,
        center_name: req_text(map, "CENTER_NAME")?,
        ref_frame: req_text(map, "REF_FRAME")?,
        time_system: req_text(map, "TIME_SYSTEM")?,
        start_time: req_text(map, "START_TIME")?,
        stop_time: req_text(map, "STOP_TIME")?,
        useable_start_time: opt_text(map, "USEABLE_START_TIME"),
        useable_stop_time: opt_text(map, "USEABLE_STOP_TIME"),
        interpolation: opt_text(map, "INTERPOLATION"),
        interpolation_degree: opt_u32(map, "INTERPOLATION_DEGREE")?,
    })
}

fn parse_state_line(line: &str) -> Result<OemState, StateLineError> {
    let mut tokenizer = Tokenizer::new(line);
    let mut tokens = Vec::new();
    while let Some(token) = tokenizer.next_str() {
        tokens.push(token);
    }

    if tokens.len() != 7 && tokens.len() != 10 {
        return Err(StateLineError::WrongTokenCount);
    }

    let epoch = tokens[0].to_string();
    let mut values = [0.0_f64; 9];
    for (idx, key) in STATE_NUMBER_KEYS.iter().enumerate().take(tokens.len() - 1) {
        values[idx] = validate::strict_f64(tokens[idx + 1], key).map_err(StateLineError::Field)?;
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
        cov_ref_frame: opt_text(map, "COV_REF_FRAME"),
        matrix: read_covariance6(map).map_err(map_oem_field_error)?,
    })
}

fn parse_xml_segment(segment: Node) -> Result<OemSegment, OemError> {
    let metadata_node = child_element(segment, "metadata")
        .ok_or_else(|| OemError::Field("segment missing metadata".to_string()))?;
    let data_node = child_element(segment, "data")
        .ok_or_else(|| OemError::Field("segment missing data".to_string()))?;

    let metadata = parse_metadata(&FieldMap::from_pairs(xml_fields(
        metadata_node,
        &METADATA_KEYS,
    )))?;
    let states = data_node
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "stateVector")
        .map(parse_xml_state)
        .collect::<Result<Vec<_>, _>>()?;
    let covariances = data_node
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == "covarianceMatrix")
        .map(parse_xml_covariance)
        .collect::<Result<Vec<_>, _>>()?;

    Ok(OemSegment {
        metadata,
        states,
        covariances,
    })
}

fn parse_xml_state(node: Node) -> Result<OemState, OemError> {
    let epoch = node_text(node, "EPOCH").ok_or(OemError::MissingField("EPOCH"))?;
    let x = required_node_num(node, "X")?;
    let y = required_node_num(node, "Y")?;
    let z = required_node_num(node, "Z")?;
    let xd = required_node_num(node, "X_DOT")?;
    let yd = required_node_num(node, "Y_DOT")?;
    let zd = required_node_num(node, "Z_DOT")?;

    let acceleration_km_s2 = match (
        node_text(node, "X_DDOT"),
        node_text(node, "Y_DDOT"),
        node_text(node, "Z_DDOT"),
    ) {
        (None, None, None) => None,
        (Some(xdd), Some(ydd), Some(zdd)) => Some([
            parse_num(&xdd, "X_DDOT")?,
            parse_num(&ydd, "Y_DDOT")?,
            parse_num(&zdd, "Z_DDOT")?,
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

fn parse_xml_covariance(node: Node) -> Result<OemCovariance, OemError> {
    let fields = node
        .descendants()
        .filter(Node::is_element)
        .filter_map(|n| {
            let text = n.text()?.trim();
            if text.is_empty() {
                None
            } else {
                Some((n.tag_name().name().to_string(), text.to_string()))
            }
        })
        .collect();
    parse_covariance_map(&FieldMap::from_pairs(fields))
}

fn encode_metadata_kvn(metadata: &OemMetadata) -> Vec<String> {
    let mut lines = vec![
        format!("OBJECT_NAME = {}", metadata.object_name),
        format!("OBJECT_ID = {}", metadata.object_id),
        format!("CENTER_NAME = {}", metadata.center_name),
        format!("REF_FRAME = {}", metadata.ref_frame),
        format!("TIME_SYSTEM = {}", metadata.time_system),
        format!("START_TIME = {}", metadata.start_time),
        format!("STOP_TIME = {}", metadata.stop_time),
    ];
    if let Some(value) = &metadata.useable_start_time {
        lines.push(format!("USEABLE_START_TIME = {value}"));
    }
    if let Some(value) = &metadata.useable_stop_time {
        lines.push(format!("USEABLE_STOP_TIME = {value}"));
    }
    if let Some(value) = &metadata.interpolation {
        lines.push(format!("INTERPOLATION = {value}"));
    }
    if let Some(value) = metadata.interpolation_degree {
        lines.push(format!("INTERPOLATION_DEGREE = {value}"));
    }
    lines
}

fn encode_state_kvn(state: &OemState) -> String {
    let mut fields = vec![
        state.epoch.clone(),
        fmt_num(state.position_km[0]),
        fmt_num(state.position_km[1]),
        fmt_num(state.position_km[2]),
        fmt_num(state.velocity_km_s[0]),
        fmt_num(state.velocity_km_s[1]),
        fmt_num(state.velocity_km_s[2]),
    ];
    if let Some(accel) = state.acceleration_km_s2 {
        fields.extend([fmt_num(accel[0]), fmt_num(accel[1]), fmt_num(accel[2])]);
    }
    fields.join(" ")
}

fn encode_xml_segment(segment: &OemSegment) -> Vec<String> {
    let mut lines = vec![
        "    <segment>".to_string(),
        "      <metadata>".to_string(),
        elem_line(8, "OBJECT_NAME", &segment.metadata.object_name),
        elem_line(8, "OBJECT_ID", &segment.metadata.object_id),
        elem_line(8, "CENTER_NAME", &segment.metadata.center_name),
        elem_line(8, "REF_FRAME", &segment.metadata.ref_frame),
        elem_line(8, "TIME_SYSTEM", &segment.metadata.time_system),
        elem_line(8, "START_TIME", &segment.metadata.start_time),
        elem_line(8, "STOP_TIME", &segment.metadata.stop_time),
    ];
    if let Some(value) = &segment.metadata.useable_start_time {
        lines.push(elem_line(8, "USEABLE_START_TIME", value));
    }
    if let Some(value) = &segment.metadata.useable_stop_time {
        lines.push(elem_line(8, "USEABLE_STOP_TIME", value));
    }
    if let Some(value) = &segment.metadata.interpolation {
        lines.push(elem_line(8, "INTERPOLATION", value));
    }
    if let Some(value) = segment.metadata.interpolation_degree {
        lines.push(elem_line_raw(8, "INTERPOLATION_DEGREE", &value.to_string()));
    }
    lines.push("      </metadata>".to_string());
    lines.push("      <data>".to_string());

    for state in &segment.states {
        lines.extend(encode_xml_state(state));
    }
    for covariance in &segment.covariances {
        lines.extend(encode_xml_covariance(covariance));
    }

    lines.push("      </data>".to_string());
    lines.push("    </segment>".to_string());
    lines
}

fn encode_xml_state(state: &OemState) -> Vec<String> {
    let mut lines = vec![
        "        <stateVector>".to_string(),
        elem_line(10, "EPOCH", &state.epoch),
        elem_line_raw(10, "X", &fmt_num(state.position_km[0])),
        elem_line_raw(10, "Y", &fmt_num(state.position_km[1])),
        elem_line_raw(10, "Z", &fmt_num(state.position_km[2])),
        elem_line_raw(10, "X_DOT", &fmt_num(state.velocity_km_s[0])),
        elem_line_raw(10, "Y_DOT", &fmt_num(state.velocity_km_s[1])),
        elem_line_raw(10, "Z_DOT", &fmt_num(state.velocity_km_s[2])),
    ];
    if let Some(accel) = state.acceleration_km_s2 {
        lines.push(elem_line_raw(10, "X_DDOT", &fmt_num(accel[0])));
        lines.push(elem_line_raw(10, "Y_DDOT", &fmt_num(accel[1])));
        lines.push(elem_line_raw(10, "Z_DDOT", &fmt_num(accel[2])));
    }
    lines.push("        </stateVector>".to_string());
    lines
}

fn encode_xml_covariance(covariance: &OemCovariance) -> Vec<String> {
    let mut lines = vec![
        "        <covarianceMatrix>".to_string(),
        elem_line(10, "EPOCH", &covariance.epoch),
    ];
    if let Some(value) = &covariance.cov_ref_frame {
        lines.push(elem_line(10, "COV_REF_FRAME", value));
    }
    for line in write_covariance6(&covariance.matrix) {
        if let Some((key, value)) = line.split_once('=') {
            lines.push(elem_line_raw(10, key.trim(), value.trim()));
        }
    }
    lines.push("        </covarianceMatrix>".to_string());
    lines
}

fn numbered_lines(text: &str) -> Vec<(usize, String)> {
    text.lines()
        .enumerate()
        .map(|(idx, line)| (idx + 1, line.trim().to_string()))
        .collect()
}

fn parse_kv_lines(lines: &[String]) -> Vec<(String, String)> {
    lines
        .iter()
        .filter_map(|line| {
            line.split_once('=').map(|(key, value)| {
                (
                    key.trim().to_string(),
                    strip_units(value.trim()).to_string(),
                )
            })
        })
        .collect()
}

fn strip_units(value: &str) -> &str {
    let trimmed = value.trim_end();
    if let Some(open) = trimmed.rfind('[') {
        if trimmed.ends_with(']') {
            return trimmed[..open].trim_end();
        }
    }
    trimmed
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

fn required_node_num(node: Node, tag: &'static str) -> Result<f64, OemError> {
    let value = node_text(node, tag).ok_or(OemError::MissingField(tag))?;
    parse_num(&value, tag)
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

fn node_text(node: Node, tag: &str) -> Option<String> {
    let element = node
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == tag)?;
    let text = element.text()?.trim();
    if text.is_empty() {
        None
    } else {
        Some(text.to_string())
    }
}

fn child_element<'a>(node: Node<'a, 'a>, tag: &str) -> Option<Node<'a, 'a>> {
    node.children()
        .find(|n| n.is_element() && n.tag_name().name() == tag)
}

fn xml_fields(node: Node, keys: &[&str]) -> Vec<(String, String)> {
    keys.iter()
        .filter_map(|key| node_text(node, key).map(|value| ((*key).to_string(), value)))
        .collect()
}

fn elem_line(indent: usize, name: &str, value: &str) -> String {
    elem_line_raw(indent, name, &xml::escape(value))
}

fn elem_line_raw(indent: usize, name: &str, value: &str) -> String {
    format!("{:indent$}<{name}>{value}</{name}>", "")
}

enum StateLineError {
    WrongTokenCount,
    Field(validate::FieldError),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn diagonal_covariance() -> Covariance6 {
        Covariance6::from_diagonal([1.0, 2.0, 3.0, 4.0e-6, 5.0e-6, 6.0e-6]).unwrap()
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
        assert_eq!(oem.skipped_states, 2);
    }

    #[test]
    fn malformed_xml_is_an_error() {
        assert!(parse_xml("<oem></oem><oem></oem>").is_err());
    }

    #[test]
    fn covariance_round_trips_through_kvn_and_xml() {
        let original = Oem {
            ccsds_oem_vers: "2.0".to_string(),
            creation_date: Some("2026-06-28T00:00:00".to_string()),
            originator: Some("SIDEREON".to_string()),
            skipped_states: 0,
            segments: vec![OemSegment {
                metadata: OemMetadata {
                    object_name: "TEST".to_string(),
                    object_id: "2026-001A".to_string(),
                    center_name: "EARTH".to_string(),
                    ref_frame: "EME2000".to_string(),
                    time_system: "UTC".to_string(),
                    start_time: "2026-06-28T00:00:00".to_string(),
                    stop_time: "2026-06-28T00:10:00".to_string(),
                    useable_start_time: None,
                    useable_stop_time: None,
                    interpolation: Some("LAGRANGE".to_string()),
                    interpolation_degree: Some(5),
                },
                states: vec![OemState {
                    epoch: "2026-06-28T00:00:00".to_string(),
                    position_km: [1.0, 2.0, 3.0],
                    velocity_km_s: [0.1, 0.2, 0.3],
                    acceleration_km_s2: None,
                }],
                covariances: vec![OemCovariance {
                    epoch: "2026-06-28T00:00:00".to_string(),
                    cov_ref_frame: Some("RTN".to_string()),
                    matrix: diagonal_covariance(),
                }],
            }],
        };

        assert_eq!(parse_kvn(&encode_kvn(&original)).unwrap(), original);
        assert_eq!(parse_xml(&encode_xml(&original)).unwrap(), original);
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
            assert_eq!(oem.skipped_states, 0);
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
            assert_eq!(parse_kvn(&encode_kvn(&oem)).unwrap(), oem);
        }

        #[test]
        fn fixture_xml_round_trips() {
            let oem = parse_xml(GPS_XML).unwrap();
            assert_eq!(parse_xml(&encode_xml(&oem)).unwrap(), oem);
        }
    }
}
