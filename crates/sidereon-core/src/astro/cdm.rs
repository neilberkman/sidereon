//! CCSDS Conjunction Data Message (CDM) KVN and XML format reader and writer.
//!
//! CDM (CCSDS 508.0-B-1) describes a predicted close approach between two space
//! objects: the time of closest approach, miss geometry, collision probability,
//! the per-object metadata block, and per-object state vectors with RTN
//! position-and-velocity covariance. This module owns the language-independent
//! grammar for both serializations:
//!
//! - KVN (Keyword=Value Notation): line tokenization, the `KEY = VALUE [unit]`
//!   split, trailing-unit stripping, the `COMMENT HBR =` convention,
//!   object-block segmentation, the leading-float value parser, the
//!   state/covariance extraction, and the KVN line layout used on encode.
//! - XML: a real DOM parse via the `roxmltree` crate (which correctly handles
//!   the `<?xml?>` declaration, comments, namespaces, entity escaping, and
//!   encoding), then name-based leaf-element lookup over the document and the
//!   two `<segment>` subtrees for the same state/covariance extraction. Encoding
//!   emits the CCSDS document layout through a controlled serializer (see
//!   [`encode_xml`]).
//!
//! Both run identically regardless of the calling language, so they live in the
//! core.
//!
//! Date/time fields cross this boundary as raw strings: resolving `TCA` /
//! `CREATION_DATE` to a concrete instant (and formatting one back) is the host's
//! job using its native date/time type, exactly as the TLE epoch is handled. This
//! module deliberately does not depend on the time-scale machinery, and applies
//! no calendar validation; it only carries the textual value through.

use crate::astro::xml;
use crate::validate;
use roxmltree::{Document, Node};
use std::fmt;

/// Keys of the six-component state vector, in CCSDS order (position then
/// velocity). Used to pull the state out of a parsed object block.
const STATE_KEYS: [&str; 6] = ["X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"];
/// Keys of the RTN position covariance lower triangle, in CCSDS order.
const COVARIANCE_KEYS: [&str; 6] = ["CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N"];
/// Velocity rows of the RTN covariance lower triangle (`(key, units)` pairs) in
/// CCSDS order: the position-velocity cross terms in `m**2/s` and the
/// velocity-velocity terms in `m**2/s**2`. These complete the standard 6x6
/// covariance (the six [`COVARIANCE_KEYS`] are its position 3x3 block). They are
/// captured all-or-nothing: a complete block round-trips, a stray or partial
/// element is canonicalized away.
const VELOCITY_COVARIANCE_FIELDS: [(&str, &str); 15] = [
    ("CRDOT_R", "m**2/s"),
    ("CRDOT_T", "m**2/s"),
    ("CRDOT_N", "m**2/s"),
    ("CRDOT_RDOT", "m**2/s**2"),
    ("CTDOT_R", "m**2/s"),
    ("CTDOT_T", "m**2/s"),
    ("CTDOT_N", "m**2/s"),
    ("CTDOT_RDOT", "m**2/s**2"),
    ("CTDOT_TDOT", "m**2/s**2"),
    ("CNDOT_R", "m**2/s"),
    ("CNDOT_T", "m**2/s"),
    ("CNDOT_N", "m**2/s"),
    ("CNDOT_RDOT", "m**2/s**2"),
    ("CNDOT_TDOT", "m**2/s**2"),
    ("CNDOT_NDOT", "m**2/s**2"),
];
/// Marker keyword whose value (`OBJECT1` / `OBJECT2`) opens an object block.
const OBJECT_MARKER: &str = "OBJECT";
/// Comment prefix: lines beginning with this are dropped before key/value
/// parsing (the embedded `COMMENT HBR =` value is recovered separately).
const COMMENT_PREFIX: &str = "COMMENT";
/// Element name wrapping one object's metadata/data block in the CDM XML schema.
const SEGMENT_TAG: &str = "segment";

/// A two-object conjunction parsed from a CDM message (KVN or XML). Date/time
/// fields are the raw textual values; the host resolves them to its own instant
/// type.
#[derive(Debug, Clone, PartialEq)]
pub struct CdmKvn {
    pub creation_date: Option<String>,
    pub originator: Option<String>,
    pub message_id: Option<String>,
    pub tca: Option<String>,
    pub miss_distance_m: Option<f64>,
    pub relative_speed_m_s: Option<f64>,
    pub collision_probability: Option<f64>,
    pub collision_probability_method: Option<String>,
    pub hard_body_radius_m: Option<f64>,
    pub object1: CdmObject,
    pub object2: CdmObject,
}

/// One object's CCSDS metadata block, state vector, and RTN covariance. Every
/// metadata field is the verbatim textual value (CCSDS enum fields such as
/// `OBJECT_TYPE` and `MANEUVERABLE` are carried as strings); absent fields are
/// `None` and are not emitted on encode.
#[derive(Debug, Clone, PartialEq)]
pub struct CdmObject {
    pub object_designator: Option<String>,
    pub catalog_name: Option<String>,
    pub object_name: Option<String>,
    pub international_designator: Option<String>,
    pub object_type: Option<String>,
    pub operator_contact_position: Option<String>,
    pub operator_organization: Option<String>,
    pub operator_phone: Option<String>,
    pub operator_email: Option<String>,
    pub ephemeris_name: Option<String>,
    pub covariance_method: Option<String>,
    pub maneuverable: Option<String>,
    pub orbit_center: Option<String>,
    pub ref_frame: Option<String>,
    pub gravity_model: Option<String>,
    pub atmospheric_model: Option<String>,
    pub n_body_perturbations: Option<String>,
    pub solar_rad_pressure: Option<String>,
    pub earth_tides: Option<String>,
    pub intrack_thrust: Option<String>,
    /// Position `(x, y, z)` then velocity `(x_dot, y_dot, z_dot)`.
    pub state: ((f64, f64, f64), (f64, f64, f64)),
    /// RTN position covariance lower triangle: CR_R, CT_R, CT_T, CN_R, CN_T, CN_N.
    pub covariance_rtn: [f64; 6],
    /// RTN velocity covariance lower-triangle rows completing the 6x6 matrix, in
    /// [`VELOCITY_COVARIANCE_FIELDS`] order, or `None` when the producer carried
    /// only the position block. Present only when the full 15-element block is.
    pub velocity_covariance_rtn: Option<[f64; 15]>,
}

/// The ordered `(metadata key, field value)` pairs for `object`, in CCSDS
/// 508.0-B-1 metadata-block order. Used by both serializers to write the block
/// in canonical order, emitting only the fields that are present.
fn object_metadata_pairs(object: &CdmObject) -> [(&'static str, &Option<String>); 20] {
    [
        ("OBJECT_DESIGNATOR", &object.object_designator),
        ("CATALOG_NAME", &object.catalog_name),
        ("OBJECT_NAME", &object.object_name),
        ("INTERNATIONAL_DESIGNATOR", &object.international_designator),
        ("OBJECT_TYPE", &object.object_type),
        (
            "OPERATOR_CONTACT_POSITION",
            &object.operator_contact_position,
        ),
        ("OPERATOR_ORGANIZATION", &object.operator_organization),
        ("OPERATOR_PHONE", &object.operator_phone),
        ("OPERATOR_EMAIL", &object.operator_email),
        ("EPHEMERIS_NAME", &object.ephemeris_name),
        ("COVARIANCE_METHOD", &object.covariance_method),
        ("MANEUVERABLE", &object.maneuverable),
        ("ORBIT_CENTER", &object.orbit_center),
        ("REF_FRAME", &object.ref_frame),
        ("GRAVITY_MODEL", &object.gravity_model),
        ("ATMOSPHERIC_MODEL", &object.atmospheric_model),
        ("N_BODY_PERTURBATIONS", &object.n_body_perturbations),
        ("SOLAR_RAD_PRESSURE", &object.solar_rad_pressure),
        ("EARTH_TIDES", &object.earth_tides),
        ("INTRACK_THRUST", &object.intrack_thrust),
    ]
}

/// Assemble a [`CdmObject`] from a serialization-specific metadata getter and the
/// already-parsed state and covariance. `get` resolves a metadata key to its
/// textual value (KVN field lookup or XML leaf-element text), keeping the KVN and
/// XML readers on one shared field list.
fn assemble_object<F>(
    get: F,
    state: ((f64, f64, f64), (f64, f64, f64)),
    covariance_rtn: [f64; 6],
    velocity_covariance_rtn: Option<[f64; 15]>,
) -> CdmObject
where
    F: Fn(&str) -> Option<String>,
{
    CdmObject {
        object_designator: get("OBJECT_DESIGNATOR"),
        catalog_name: get("CATALOG_NAME"),
        object_name: get("OBJECT_NAME"),
        international_designator: get("INTERNATIONAL_DESIGNATOR"),
        object_type: get("OBJECT_TYPE"),
        operator_contact_position: get("OPERATOR_CONTACT_POSITION"),
        operator_organization: get("OPERATOR_ORGANIZATION"),
        operator_phone: get("OPERATOR_PHONE"),
        operator_email: get("OPERATOR_EMAIL"),
        ephemeris_name: get("EPHEMERIS_NAME"),
        covariance_method: get("COVARIANCE_METHOD"),
        maneuverable: get("MANEUVERABLE"),
        orbit_center: get("ORBIT_CENTER"),
        ref_frame: get("REF_FRAME"),
        gravity_model: get("GRAVITY_MODEL"),
        atmospheric_model: get("ATMOSPHERIC_MODEL"),
        n_body_perturbations: get("N_BODY_PERTURBATIONS"),
        solar_rad_pressure: get("SOLAR_RAD_PRESSURE"),
        earth_tides: get("EARTH_TIDES"),
        intrack_thrust: get("INTRACK_THRUST"),
        state,
        covariance_rtn,
        velocity_covariance_rtn,
    }
}

/// Failure modes of [`parse_kvn`]. The message strings are the historical public
/// contract surfaced by the Elixir binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CdmError {
    /// An object block was missing one or more state-vector components.
    IncompleteStateVector,
    /// A numeric field was absent, malformed, non-finite, or outside its domain.
    InvalidField {
        /// The invalid CDM field.
        field: &'static str,
        /// The validation failure category.
        kind: CdmInputErrorKind,
    },
    /// The XML reader was handed text that is not a well-formed XML document.
    MalformedXml(String),
}

/// CDM boundary-validation failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CdmInputErrorKind {
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

impl fmt::Display for CdmInputErrorKind {
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

impl From<&validate::FieldError> for CdmInputErrorKind {
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

impl fmt::Display for CdmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CdmError::IncompleteStateVector => write!(f, "incomplete state vector"),
            CdmError::InvalidField { field, kind } => {
                write!(f, "invalid CDM field {field}: {kind}")
            }
            CdmError::MalformedXml(detail) => write!(f, "malformed XML: {detail}"),
        }
    }
}

impl std::error::Error for CdmError {}

/// Parse a CDM in KVN format.
///
/// Tokenizes the message, recovers the header/relative-metadata fields, the
/// `COMMENT HBR =` hard-body radius, and the two object blocks. Date/time fields
/// are returned verbatim for the host to resolve; presence/format checks on them
/// (and on `MESSAGE_ID`) are the host's concern. An object block missing any
/// state component is rejected with [`CdmError::IncompleteStateVector`]. Every
/// covariance component is required and every accepted numeric state/covariance
/// value must be finite.
pub fn parse_kvn(text: &str) -> Result<CdmKvn, CdmError> {
    let lines = significant_lines(text);
    let kv = crate::format::kvn::FieldMap::from_pairs(parse_kv_lines(&lines));

    let (object1_kv, object2_kv) = split_object_blocks(&lines);
    let object1 = parse_object(&object1_kv)?;
    let object2 = parse_object(&object2_kv)?;

    Ok(CdmKvn {
        creation_date: kv_get(&kv, "CREATION_DATE"),
        originator: kv_get(&kv, "ORIGINATOR"),
        message_id: kv_get(&kv, "MESSAGE_ID"),
        tca: kv_get(&kv, "TCA"),
        miss_distance_m: optional_kv_num(&kv, "MISS_DISTANCE")?,
        relative_speed_m_s: optional_kv_num(&kv, "RELATIVE_SPEED")?,
        collision_probability: optional_kv_num(&kv, "COLLISION_PROBABILITY")?,
        collision_probability_method: kv_get(&kv, "COLLISION_PROBABILITY_METHOD"),
        hard_body_radius_m: parse_hbr(text)?,
        object1,
        object2,
    })
}

/// Encode a [`CdmKvn`] back to KVN text.
///
/// The date/time fields are taken as already-formatted strings (the host owns
/// the instant-to-string conversion). Numeric values are written with their
/// shortest round-tripping decimal form, so a re-parse recovers the exact same
/// bits; the output is therefore round-trip faithful rather than byte-identical
/// to any one producer.
pub fn encode_kvn(cdm: &CdmKvn) -> Result<String, CdmError> {
    validate_cdm(cdm)?;
    let header = crate::astro::ndm::NdmHeader {
        vers: "1.0".to_string(),
        creation_date: cdm.creation_date.clone(),
        originator: cdm.originator.clone(),
    };
    let mut lines: Vec<String> = header.write_kvn("CCSDS_CDM_VERS");
    lines.extend([
        format!("MESSAGE_ID = {}", opt_str(&cdm.message_id)),
        format!("TCA = {}", opt_str(&cdm.tca)),
        format!("MISS_DISTANCE = {} [m]", opt_num(cdm.miss_distance_m)),
        format!("RELATIVE_SPEED = {} [m/s]", opt_num(cdm.relative_speed_m_s)),
        format!(
            "COLLISION_PROBABILITY = {}",
            opt_num(cdm.collision_probability)
        ),
        format!(
            "COLLISION_PROBABILITY_METHOD = {}",
            opt_str(&cdm.collision_probability_method)
        ),
    ]);

    if let Some(hbr) = cdm.hard_body_radius_m {
        lines.push(format!("COMMENT HBR = {}", fmt_num(hbr)));
    }

    lines.extend(encode_object(&cdm.object1, "OBJECT1"));
    lines.extend(encode_object(&cdm.object2, "OBJECT2"));

    Ok(lines.join("\n"))
}

/// Parse a CDM in XML format.
///
/// Parses the document with `roxmltree` (a real XML DOM reader: the `<?xml?>`
/// declaration, comments, namespaces, entity escaping, and encoding are handled
/// by the library, not by string scanning), then reads the header and
/// relative-metadata leaf elements by name from the document and the per-object
/// state/covariance from the two `<segment>` subtrees. The CCSDS CDM XML schema
/// is flat, so a name match uniquely identifies each value. The full 6x6 RTN
/// covariance is recovered when its complete velocity block is present; a stray
/// or partial velocity element (e.g. a lone `CRDOT_R`) is canonicalized away.
/// Date/time
/// fields are returned verbatim for the host to resolve, matching [`parse_kvn`];
/// text that is not well-formed XML is rejected with [`CdmError::MalformedXml`]
/// and an object block missing any state component with
/// [`CdmError::IncompleteStateVector`]. Every covariance component is required
/// and every accepted numeric state/covariance value must be finite.
pub fn parse_xml(text: &str) -> Result<CdmKvn, CdmError> {
    let doc = Document::parse(text).map_err(|e| CdmError::MalformedXml(e.to_string()))?;
    let root = doc.root();

    let mut segments = root
        .descendants()
        .filter(|n| n.is_element() && n.tag_name().name() == SEGMENT_TAG);
    let object1 = parse_xml_object(segments.next())?;
    let object2 = parse_xml_object(segments.next())?;

    Ok(CdmKvn {
        creation_date: node_text(root, "CREATION_DATE"),
        originator: node_text(root, "ORIGINATOR"),
        message_id: node_text(root, "MESSAGE_ID"),
        tca: node_text(root, "TCA"),
        miss_distance_m: optional_node_num(root, "MISS_DISTANCE")?,
        relative_speed_m_s: optional_node_num(root, "RELATIVE_SPEED")?,
        collision_probability: optional_node_num(root, "COLLISION_PROBABILITY")?,
        collision_probability_method: node_text(root, "COLLISION_PROBABILITY_METHOD"),
        hard_body_radius_m: optional_node_num(root, "HBR")?,
        object1,
        object2,
    })
}

/// Encode a [`CdmKvn`] to a CCSDS 508.0-B-1 CDM XML document.
///
/// The date/time fields are taken as already-formatted strings (the host owns
/// the instant-to-string conversion), and numeric values use the shortest
/// round-tripping decimal form, so the output is round-trip faithful rather than
/// byte-identical to any one producer. String values are XML-escaped.
///
/// This is a controlled straight-line serializer (no parsing, no tag scanning)
/// rather than a generic streaming-writer dependency: the output is a fixed,
/// documented CCSDS layout (`cdm > header/body > segment > metadata/data`) whose
/// element nesting and `units` attributes are the inter-system exchange contract,
/// and every interpolated value is escaped via [`xml::escape`]. The matching
/// reader is the vetted `roxmltree` DOM parser in [`parse_xml`].
pub fn encode_xml(cdm: &CdmKvn) -> Result<String, CdmError> {
    validate_cdm(cdm)?;
    let mut lines: Vec<String> = vec![
        r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string(),
        r#"<cdm id="CCSDS_CDM_VERS" version="1.0">"#.to_string(),
        "  <header>".to_string(),
        "    <CCSDS_CDM_VERS>1.0</CCSDS_CDM_VERS>".to_string(),
        format!(
            "    <CREATION_DATE>{}</CREATION_DATE>",
            opt_str(&cdm.creation_date)
        ),
        format!(
            "    <ORIGINATOR>{}</ORIGINATOR>",
            xml::escape_opt(&cdm.originator)
        ),
        format!(
            "    <MESSAGE_ID>{}</MESSAGE_ID>",
            xml::escape_opt(&cdm.message_id)
        ),
        "  </header>".to_string(),
        "  <body>".to_string(),
        "    <relativeMetadataData>".to_string(),
        format!("      <TCA>{}</TCA>", opt_str(&cdm.tca)),
        format!(
            r#"      <MISS_DISTANCE units="m">{}</MISS_DISTANCE>"#,
            opt_num(cdm.miss_distance_m)
        ),
        format!(
            r#"      <RELATIVE_SPEED units="m/s">{}</RELATIVE_SPEED>"#,
            opt_num(cdm.relative_speed_m_s)
        ),
        format!(
            "      <COLLISION_PROBABILITY>{}</COLLISION_PROBABILITY>",
            opt_num(cdm.collision_probability)
        ),
        format!(
            "      <COLLISION_PROBABILITY_METHOD>{}</COLLISION_PROBABILITY_METHOD>",
            xml::escape_opt(&cdm.collision_probability_method)
        ),
        "    </relativeMetadataData>".to_string(),
    ];

    lines.extend(encode_xml_segment(&cdm.object1, "OBJECT1"));
    lines.extend(encode_xml_segment(&cdm.object2, "OBJECT2"));
    lines.push("  </body>".to_string());
    lines.push("</cdm>".to_string());

    Ok(lines.join("\n"))
}

// -- KVN tokenization --

/// Trim every line, then drop blanks and comment lines. Comment values that the
/// grammar needs (the HBR) are recovered from the raw text separately.
fn significant_lines(text: &str) -> Vec<String> {
    text.split('\n')
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with(COMMENT_PREFIX))
        .collect()
}

/// Build the key/value map from `KEY = VALUE` lines. Later duplicates win, which
/// only matters for header keys repeated across object blocks (those are read per
/// block instead). The value has any trailing `[unit]` removed.
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

fn kv_get(kv: &crate::format::kvn::FieldMap, key: &str) -> Option<String> {
    kv.get_last(key).map(str::to_string)
}

fn optional_kv_num(
    kv: &crate::format::kvn::FieldMap,
    key: &'static str,
) -> Result<Option<f64>, CdmError> {
    kv.get_last(key)
        .map(|value| validate::strict_f64(value, key).map_err(map_cdm_field_error))
        .transpose()
}

fn required_kv_num(kv: &crate::format::kvn::FieldMap, key: &'static str) -> Result<f64, CdmError> {
    let value = kv
        .get_last(key)
        .ok_or(validate::FieldError::Missing { field: key })
        .map_err(map_cdm_field_error)?;
    validate::strict_f64(value, key).map_err(map_cdm_field_error)
}

fn required_state_kv_num(
    kv: &crate::format::kvn::FieldMap,
    key: &'static str,
) -> Result<f64, CdmError> {
    let value = kv.get_last(key).ok_or(CdmError::IncompleteStateVector)?;
    validate::strict_f64(value, key).map_err(map_cdm_field_error)
}

fn map_cdm_field_error(error: validate::FieldError) -> CdmError {
    CdmError::InvalidField {
        field: error.field(),
        kind: CdmInputErrorKind::from(&error),
    }
}

/// Remove a trailing bracketed unit (` [m]`, `[m**2/kg]`, ...) and surrounding
/// whitespace, leaving the bare value.
fn strip_units(value: &str) -> &str {
    let trimmed = value.trim_end();
    if let Some(open) = trimmed.rfind('[') {
        if trimmed.ends_with(']') {
            return trimmed[..open].trim_end();
        }
    }
    trimmed
}

/// Split the line stream into the two object blocks at the `OBJECT =` markers.
/// CDM always carries exactly two; if fewer are present the blocks are empty and
/// the state-vector check rejects the message.
fn split_object_blocks(lines: &[String]) -> (Vec<String>, Vec<String>) {
    let markers: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| {
            line.split_once('=')
                .is_some_and(|(key, _)| key.trim() == OBJECT_MARKER)
        })
        .map(|(idx, _)| idx)
        .collect();

    match markers.as_slice() {
        [i1, i2, ..] => (lines[*i1..*i2].to_vec(), lines[*i2..].to_vec()),
        _ => (Vec::new(), Vec::new()),
    }
}

fn parse_object(lines: &[String]) -> Result<CdmObject, CdmError> {
    let kv = crate::format::kvn::FieldMap::from_pairs(parse_kv_lines(lines));

    let mut state = [0.0_f64; 6];
    for (slot, key) in state.iter_mut().zip(STATE_KEYS) {
        *slot = required_state_kv_num(&kv, key)?;
    }
    validate::finite_slice(&state, "state").map_err(map_cdm_field_error)?;

    let mut covariance_rtn = [0.0_f64; 6];
    for (slot, key) in covariance_rtn.iter_mut().zip(COVARIANCE_KEYS) {
        *slot = required_kv_num(&kv, key)?;
    }
    validate::finite_slice(&covariance_rtn, "covariance_rtn").map_err(map_cdm_field_error)?;
    validate_covariance_rtn(&covariance_rtn)?;

    let velocity_covariance_rtn = read_velocity_covariance(|key| optional_kv_num(&kv, key))?;

    Ok(assemble_object(
        |key| kv_get(&kv, key),
        (
            (state[0], state[1], state[2]),
            (state[3], state[4], state[5]),
        ),
        covariance_rtn,
        velocity_covariance_rtn,
    ))
}

/// Read the optional RTN velocity covariance block through a
/// serialization-specific numeric getter. The 15 lower-triangle velocity terms
/// are captured all-or-nothing: every element present yields the full block, an
/// absent or partial block yields `None` (a stray element is canonicalized away).
/// Any present element must be a finite number.
fn read_velocity_covariance<F>(get_num: F) -> Result<Option<[f64; 15]>, CdmError>
where
    F: Fn(&'static str) -> Result<Option<f64>, CdmError>,
{
    let mut values = [0.0_f64; 15];
    let mut present = 0_usize;
    for (slot, (key, _units)) in values.iter_mut().zip(VELOCITY_COVARIANCE_FIELDS) {
        if let Some(value) = get_num(key)? {
            *slot = value;
            present += 1;
        }
    }
    if present == VELOCITY_COVARIANCE_FIELDS.len() {
        Ok(Some(values))
    } else {
        Ok(None)
    }
}

/// Recover the hard-body radius from a `COMMENT HBR = <value>` line (NASA CARA
/// convention). Scans the raw text case-insensitively, taking the leading
/// digits-and-dot run as the value.
fn parse_hbr(text: &str) -> Result<Option<f64>, CdmError> {
    for line in text.split('\n') {
        let trimmed = line.trim();
        let mut rest = match strip_prefix_ci(trimmed, COMMENT_PREFIX) {
            Some(rest) if starts_with_ascii_ws(rest) => rest.trim_start(),
            _ => continue,
        };
        rest = match strip_prefix_ci(rest, "HBR") {
            Some(rest) => rest.trim_start(),
            None => continue,
        };
        let rest = match rest.strip_prefix('=') {
            Some(rest) => rest.trim_start(),
            None => continue,
        };
        let value = strip_units(rest).split_whitespace().next().unwrap_or("");
        if value.is_empty() {
            return Ok(None);
        }
        return validate::strict_f64(value, "HBR")
            .map(Some)
            .map_err(map_cdm_field_error);
    }
    Ok(None)
}

// -- KVN encoding --

fn encode_object(object: &CdmObject, name: &str) -> Vec<String> {
    let ((x, y, z), (xd, yd, zd)) = object.state;
    let [cr_r, ct_r, ct_t, cn_r, cn_t, cn_n] = object.covariance_rtn;

    let mut lines = vec![format!("OBJECT = {name}")];
    for (key, value) in object_metadata_pairs(object) {
        if let Some(text) = value {
            lines.push(format!("{key} = {text}"));
        }
    }
    lines.extend([
        format!("X = {} [km]", fmt_num(x)),
        format!("Y = {} [km]", fmt_num(y)),
        format!("Z = {} [km]", fmt_num(z)),
        format!("X_DOT = {} [km/s]", fmt_num(xd)),
        format!("Y_DOT = {} [km/s]", fmt_num(yd)),
        format!("Z_DOT = {} [km/s]", fmt_num(zd)),
        format!("CR_R = {} [m**2]", fmt_num(cr_r)),
        format!("CT_R = {} [m**2]", fmt_num(ct_r)),
        format!("CT_T = {} [m**2]", fmt_num(ct_t)),
        format!("CN_R = {} [m**2]", fmt_num(cn_r)),
        format!("CN_T = {} [m**2]", fmt_num(cn_t)),
        format!("CN_N = {} [m**2]", fmt_num(cn_n)),
    ]);
    if let Some(velocity) = &object.velocity_covariance_rtn {
        for (value, (key, units)) in velocity.iter().zip(VELOCITY_COVARIANCE_FIELDS) {
            lines.push(format!("{key} = {} [{units}]", fmt_num(*value)));
        }
    }
    lines
}

// -- Value helpers --

/// Shortest round-tripping decimal for a finite value.
fn fmt_num(value: f64) -> String {
    format!("{value}")
}

fn opt_str(value: &Option<String>) -> String {
    value.clone().unwrap_or_default()
}

fn opt_num(value: Option<f64>) -> String {
    value.map_or_else(String::new, fmt_num)
}

fn validate_cdm(cdm: &CdmKvn) -> Result<(), CdmError> {
    validate_optional_num(cdm.miss_distance_m, "MISS_DISTANCE")?;
    validate_optional_num(cdm.relative_speed_m_s, "RELATIVE_SPEED")?;
    validate_optional_num(cdm.collision_probability, "COLLISION_PROBABILITY")?;
    validate_optional_num(cdm.hard_body_radius_m, "HBR")?;
    validate_object(&cdm.object1)?;
    validate_object(&cdm.object2)?;
    Ok(())
}

fn validate_optional_num(value: Option<f64>, field: &'static str) -> Result<(), CdmError> {
    value.map_or(Ok(()), |value| {
        validate::finite(value, field)
            .map(|_| ())
            .map_err(map_cdm_field_error)
    })
}

fn validate_object(object: &CdmObject) -> Result<(), CdmError> {
    let ((x, y, z), (xd, yd, zd)) = object.state;
    validate::finite_slice(&[x, y, z, xd, yd, zd], "state").map_err(map_cdm_field_error)?;
    validate::finite_slice(&object.covariance_rtn, "covariance_rtn")
        .map_err(map_cdm_field_error)?;
    if let Some(velocity) = &object.velocity_covariance_rtn {
        validate::finite_slice(velocity, "velocity_covariance_rtn").map_err(map_cdm_field_error)?;
    }
    validate_covariance_rtn(&object.covariance_rtn)
}

fn validate_covariance_rtn(covariance_rtn: &[f64; 6]) -> Result<(), CdmError> {
    let [cr_r, ct_r, ct_t, cn_r, cn_t, cn_n] = *covariance_rtn;
    let covariance = [[cr_r, ct_r, cn_r], [ct_r, ct_t, cn_t], [cn_r, cn_t, cn_n]];
    validate::validate_covariance_psd(&covariance, "covariance_rtn").map_err(map_cdm_field_error)
}

/// Case-insensitive ASCII prefix strip.
fn strip_prefix_ci<'a>(text: &'a str, prefix: &str) -> Option<&'a str> {
    if text
        .get(..prefix.len())
        .is_some_and(|candidate| candidate.eq_ignore_ascii_case(prefix))
    {
        text.get(prefix.len()..)
    } else {
        None
    }
}

fn starts_with_ascii_ws(text: &str) -> bool {
    text.chars().next().is_some_and(|c| c.is_ascii_whitespace())
}

// -- XML parsing --

/// First descendant element of `node` (document order) whose local tag name is
/// `tag`, returning its trimmed text. `None` if the element is absent, empty, or
/// self-closing. roxmltree decodes entities and ignores attributes, so the value
/// is the element's resolved text content.
fn node_text(node: Node, tag: &str) -> Option<String> {
    let element = node
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == tag)?;
    let text = element.text()?.trim();
    (!text.is_empty()).then(|| text.to_string())
}

/// Leaf-element value parsed as a finite float, or `None`.
fn optional_node_num(node: Node, tag: &'static str) -> Result<Option<f64>, CdmError> {
    node_text(node, tag)
        .map(|value| validate::strict_f64(&value, tag).map_err(map_cdm_field_error))
        .transpose()
}

fn required_node_num(node: Node, tag: &'static str) -> Result<f64, CdmError> {
    let value = node_text(node, tag)
        .ok_or(validate::FieldError::Missing { field: tag })
        .map_err(map_cdm_field_error)?;
    validate::strict_f64(&value, tag).map_err(map_cdm_field_error)
}

fn required_state_node_num(node: Node, tag: &'static str) -> Result<f64, CdmError> {
    let value = node_text(node, tag).ok_or(CdmError::IncompleteStateVector)?;
    validate::strict_f64(&value, tag).map_err(map_cdm_field_error)
}

/// Build one [`CdmObject`] from its `<segment>` subtree. A missing segment (fewer
/// than two present) or any missing state component is rejected with
/// [`CdmError::IncompleteStateVector`]. Every covariance component is required.
fn parse_xml_object(segment: Option<Node>) -> Result<CdmObject, CdmError> {
    let segment = segment.ok_or(CdmError::IncompleteStateVector)?;

    let mut state = [0.0_f64; 6];
    for (slot, key) in state.iter_mut().zip(STATE_KEYS) {
        *slot = required_state_node_num(segment, key)?;
    }
    validate::finite_slice(&state, "state").map_err(map_cdm_field_error)?;

    let mut covariance_rtn = [0.0_f64; 6];
    for (slot, key) in covariance_rtn.iter_mut().zip(COVARIANCE_KEYS) {
        *slot = required_node_num(segment, key)?;
    }
    validate::finite_slice(&covariance_rtn, "covariance_rtn").map_err(map_cdm_field_error)?;
    validate_covariance_rtn(&covariance_rtn)?;

    let velocity_covariance_rtn = read_velocity_covariance(|key| optional_node_num(segment, key))?;

    Ok(assemble_object(
        |key| node_text(segment, key),
        (
            (state[0], state[1], state[2]),
            (state[3], state[4], state[5]),
        ),
        covariance_rtn,
        velocity_covariance_rtn,
    ))
}

// -- XML encoding --

fn encode_xml_segment(object: &CdmObject, name: &str) -> Vec<String> {
    let ((x, y, z), (xd, yd, zd)) = object.state;
    let [cr_r, ct_r, ct_t, cn_r, cn_t, cn_n] = object.covariance_rtn;

    let mut lines = vec![
        "    <segment>".to_string(),
        "      <metadata>".to_string(),
        format!("        <OBJECT>{name}</OBJECT>"),
    ];
    for (key, value) in object_metadata_pairs(object) {
        if let Some(text) = value {
            lines.push(format!("        <{key}>{}</{key}>", xml::escape(text)));
        }
    }
    lines.extend([
        "      </metadata>".to_string(),
        "      <data>".to_string(),
        "        <stateVector>".to_string(),
        format!(r#"          <X units="km">{}</X>"#, fmt_num(x)),
        format!(r#"          <Y units="km">{}</Y>"#, fmt_num(y)),
        format!(r#"          <Z units="km">{}</Z>"#, fmt_num(z)),
        format!(r#"          <X_DOT units="km/s">{}</X_DOT>"#, fmt_num(xd)),
        format!(r#"          <Y_DOT units="km/s">{}</Y_DOT>"#, fmt_num(yd)),
        format!(r#"          <Z_DOT units="km/s">{}</Z_DOT>"#, fmt_num(zd)),
        "        </stateVector>".to_string(),
        "        <covarianceMatrix>".to_string(),
        format!(r#"          <CR_R units="m**2">{}</CR_R>"#, fmt_num(cr_r)),
        format!(r#"          <CT_R units="m**2">{}</CT_R>"#, fmt_num(ct_r)),
        format!(r#"          <CT_T units="m**2">{}</CT_T>"#, fmt_num(ct_t)),
        format!(r#"          <CN_R units="m**2">{}</CN_R>"#, fmt_num(cn_r)),
        format!(r#"          <CN_T units="m**2">{}</CN_T>"#, fmt_num(cn_t)),
        format!(r#"          <CN_N units="m**2">{}</CN_N>"#, fmt_num(cn_n)),
    ]);
    if let Some(velocity) = &object.velocity_covariance_rtn {
        for (value, (key, units)) in velocity.iter().zip(VELOCITY_COVARIANCE_FIELDS) {
            lines.push(format!(
                r#"          <{key} units="{units}">{}</{key}>"#,
                fmt_num(*value)
            ));
        }
    }
    lines.extend([
        "        </covarianceMatrix>".to_string(),
        "      </data>".to_string(),
        "    </segment>".to_string(),
    ]);
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_units_removes_trailing_bracket() {
        assert_eq!(strip_units("7000.0 [km]"), "7000.0");
        assert_eq!(strip_units("4.835E-05"), "4.835E-05");
        assert_eq!(strip_units("0.045663 [m**2/kg]"), "0.045663");
        assert_eq!(strip_units("97.8 [%]"), "97.8");
    }

    #[test]
    fn cdm_covariance_rtn_validation_accepts_psd_lower_triangle() {
        assert_eq!(
            validate_covariance_rtn(&[1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            Ok(())
        );
    }

    #[test]
    fn cdm_covariance_rtn_validation_rejects_non_psd_lower_triangle() {
        let expected = Err(CdmError::InvalidField {
            field: "covariance_rtn",
            kind: CdmInputErrorKind::NotPositive,
        });

        assert_eq!(
            validate_covariance_rtn(&[-1.0, 0.0, 1.0, 0.0, 0.0, 1.0]),
            expected
        );
        assert_eq!(
            validate_covariance_rtn(&[1.0, 2.0, 1.0, 0.0, 0.0, 1.0]),
            expected
        );
    }

    #[test]
    fn incomplete_state_vector_is_rejected() {
        let kvn = "OBJECT = OBJECT1\nX = 7000.0 [km]\nOBJECT = OBJECT2\nX = 1.0 [km]\n";
        assert_eq!(parse_kvn(kvn), Err(CdmError::IncompleteStateVector));
    }

    #[test]
    fn hbr_is_recovered_from_comment_only() {
        let with_hbr = "COMMENT HBR = 15.5\n";
        assert_eq!(parse_hbr(with_hbr), Ok(Some(15.5)));
        assert_eq!(parse_hbr("COMMENT Relative Metadata/Data\n"), Ok(None));
    }

    #[test]
    fn kvn_hbr_comment_with_multibyte_leading_token_is_ignored() {
        let kvn = "\
CREATION_DATE = 2024-01-01T00:00:00.000
MESSAGE_ID = HBR_TEST
COMMENT \u{1f4a5}BR = 15.5
TCA = 2024-01-01T12:00:00.000
OBJECT = OBJECT1
X = 1.0 [km]
Y = 2.0 [km]
Z = 3.0 [km]
X_DOT = 0.1 [km/s]
Y_DOT = 0.2 [km/s]
Z_DOT = 0.3 [km/s]
CR_R = 1.0 [m**2]
CT_R = 0.0 [m**2]
CT_T = 1.0 [m**2]
CN_R = 0.0 [m**2]
CN_T = 0.0 [m**2]
CN_N = 1.0 [m**2]
OBJECT = OBJECT2
X = 4.0 [km]
Y = 5.0 [km]
Z = 6.0 [km]
X_DOT = 0.4 [km/s]
Y_DOT = 0.5 [km/s]
Z_DOT = 0.6 [km/s]
CR_R = 1.0 [m**2]
CT_R = 0.0 [m**2]
CT_T = 1.0 [m**2]
CN_R = 0.0 [m**2]
CN_T = 0.0 [m**2]
CN_N = 1.0 [m**2]
";
        let parsed = parse_kvn(kvn).expect("malformed HBR comment must not panic");
        assert_eq!(parsed.hard_body_radius_m, None);
    }

    #[test]
    fn node_text_reads_leaf_value_ignoring_attrs() {
        let doc = Document::parse(
            r#"<r><MESSAGE_ID>abc123</MESSAGE_ID><X units="km">2570.097065</X><ORIGINATOR></ORIGINATOR></r>"#,
        )
        .unwrap();
        let root = doc.root();
        assert_eq!(node_text(root, "MESSAGE_ID").as_deref(), Some("abc123"));
        // The `units` attribute is ignored; the element text is returned.
        assert_eq!(node_text(root, "X").as_deref(), Some("2570.097065"));
        // An empty leaf element yields None.
        assert_eq!(node_text(root, "ORIGINATOR"), None);

        // A distinct element name sharing a prefix must not match.
        let only_xdot = Document::parse(r#"<r><X_DOT units="km/s">4.4</X_DOT></r>"#).unwrap();
        assert_eq!(node_text(only_xdot.root(), "X"), None);
    }

    #[test]
    fn xml_parse_decodes_entities_and_ignores_extra_covariance_element() {
        let xml = r#"<cdm><body>
<segment><metadata><OBJECT_NAME>SAT A &amp; B</OBJECT_NAME></metadata>
<data><stateVector>
<X units="km">1.0</X><Y units="km">2.0</Y><Z units="km">3.0</Z>
<X_DOT units="km/s">0.1</X_DOT><Y_DOT units="km/s">0.2</Y_DOT><Z_DOT units="km/s">0.3</Z_DOT>
</stateVector><covarianceMatrix>
<CR_R units="m**2">41.42</CR_R><CT_R units="m**2">-8.579</CT_R><CT_T units="m**2">2533.0</CT_T>
<CN_R units="m**2">-23.13</CN_R><CN_T units="m**2">13.36</CN_T><CN_N units="m**2">70.98</CN_N>
<CRDOT_R units="m**2/s">2.52e-3</CRDOT_R>
</covarianceMatrix></data></segment>
<segment><data><stateVector>
<X units="km">4.0</X><Y units="km">5.0</Y><Z units="km">6.0</Z>
<X_DOT units="km/s">0.4</X_DOT><Y_DOT units="km/s">0.5</Y_DOT><Z_DOT units="km/s">0.6</Z_DOT>
</stateVector><covarianceMatrix>
<CR_R units="m**2">1.0</CR_R><CT_R units="m**2">0.0</CT_R><CT_T units="m**2">1.0</CT_T>
<CN_R units="m**2">0.0</CN_R><CN_T units="m**2">0.0</CN_T><CN_N units="m**2">1.0</CN_N>
</covarianceMatrix></data></segment>
</body></cdm>"#;
        let cdm = parse_xml(xml).unwrap();
        // The DOM reader decodes the `&amp;` entity.
        assert_eq!(cdm.object1.object_name.as_deref(), Some("SAT A & B"));
        // The trailing CRDOT_R element is not one of the six RTN keys, so the
        // covariance is exactly the six lower-triangle components.
        assert_eq!(
            cdm.object1.covariance_rtn,
            [41.42, -8.579, 2533.0, -23.13, 13.36, 70.98]
        );
    }

    #[test]
    fn xml_incomplete_state_vector_is_rejected() {
        let xml = "<cdm><body>\
<segment><data><stateVector><X units=\"km\">1.0</X></stateVector></data></segment>\
<segment><data><stateVector></stateVector></data></segment>\
</body></cdm>";
        assert_eq!(parse_xml(xml), Err(CdmError::IncompleteStateVector));
    }

    #[test]
    fn xml_malformed_document_is_rejected() {
        // Two root elements is not well-formed XML; the DOM reader rejects it
        // rather than silently scanning past the structure.
        assert!(matches!(
            parse_xml("<segment></segment><segment></segment>"),
            Err(CdmError::MalformedXml(_))
        ));
    }

    #[test]
    fn xml_round_trips_through_encode_and_parse() {
        let object = CdmObject {
            object_designator: Some("12345".to_string()),
            catalog_name: None,
            object_name: Some("SAT A & B".to_string()),
            international_designator: None,
            object_type: None,
            operator_contact_position: None,
            operator_organization: None,
            operator_phone: None,
            operator_email: None,
            ephemeris_name: None,
            covariance_method: None,
            maneuverable: None,
            orbit_center: None,
            ref_frame: Some("EME2000".to_string()),
            gravity_model: None,
            atmospheric_model: None,
            n_body_perturbations: None,
            solar_rad_pressure: None,
            earth_tides: None,
            intrack_thrust: None,
            state: ((1.5, 2.5, 3.5), (0.1, 0.2, 0.3)),
            covariance_rtn: [41.42, -8.579, 2533.0, -23.13, 13.36, 70.98],
            velocity_covariance_rtn: None,
        };
        let original = CdmKvn {
            creation_date: Some("2024-01-01T00:00:00.000".to_string()),
            originator: Some("TEST".to_string()),
            message_id: Some("ID-1".to_string()),
            tca: Some("2024-01-01T12:00:00.000".to_string()),
            miss_distance_m: Some(715.0),
            relative_speed_m_s: Some(14762.0),
            collision_probability: Some(4.835e-5),
            collision_probability_method: Some("FOSTER-1992".to_string()),
            hard_body_radius_m: None,
            object1: object.clone(),
            object2: object,
        };

        let encoded = encode_xml(&original).expect("valid CDM XML encode");
        assert!(encoded.starts_with("<?xml"));
        // The ampersand in the object name must be escaped on encode.
        assert!(encoded.contains("SAT A &amp; B"));

        let reparsed = parse_xml(&encoded).unwrap();
        assert_eq!(reparsed.object1.state, original.object1.state);
        assert_eq!(
            reparsed.object2.covariance_rtn,
            original.object2.covariance_rtn
        );
        assert_eq!(reparsed.miss_distance_m, original.miss_distance_m);
        assert_eq!(
            reparsed.collision_probability,
            original.collision_probability
        );
        assert_eq!(reparsed.message_id, original.message_id);
        assert_eq!(reparsed.tca, original.tca);
    }

    #[test]
    fn optional_non_finite_kvn_fields_are_rejected() {
        let kvn = "OBJECT = OBJECT1\n\
X = 1.0 [km]\nY = 2.0 [km]\nZ = 3.0 [km]\n\
X_DOT = 0.1 [km/s]\nY_DOT = 0.2 [km/s]\nZ_DOT = 0.3 [km/s]\n\
CR_R = 1.0 [m**2]\nCT_R = 0.0 [m**2]\nCT_T = 1.0 [m**2]\n\
CN_R = 0.0 [m**2]\nCN_T = 0.0 [m**2]\nCN_N = 1.0 [m**2]\n\
OBJECT = OBJECT2\n\
X = 4.0 [km]\nY = 5.0 [km]\nZ = 6.0 [km]\n\
X_DOT = 0.4 [km/s]\nY_DOT = 0.5 [km/s]\nZ_DOT = 0.6 [km/s]\n\
CR_R = 1.0 [m**2]\nCT_R = 0.0 [m**2]\nCT_T = 1.0 [m**2]\n\
CN_R = 0.0 [m**2]\nCN_T = 0.0 [m**2]\nCN_N = 1.0 [m**2]\n\
MISS_DISTANCE = NaN [m]\n";

        assert_eq!(
            parse_kvn(kvn),
            Err(CdmError::InvalidField {
                field: "MISS_DISTANCE",
                kind: CdmInputErrorKind::NonFinite,
            })
        );
    }

    #[test]
    fn optional_non_finite_xml_fields_are_rejected() {
        let xml = r#"<cdm><body>
<relativeMetadataData><COLLISION_PROBABILITY>inf</COLLISION_PROBABILITY></relativeMetadataData>
<segment><data><stateVector>
<X>1.0</X><Y>2.0</Y><Z>3.0</Z><X_DOT>0.1</X_DOT><Y_DOT>0.2</Y_DOT><Z_DOT>0.3</Z_DOT>
</stateVector><covarianceMatrix>
<CR_R>1.0</CR_R><CT_R>0.0</CT_R><CT_T>1.0</CT_T><CN_R>0.0</CN_R><CN_T>0.0</CN_T><CN_N>1.0</CN_N>
</covarianceMatrix></data></segment>
<segment><data><stateVector>
<X>4.0</X><Y>5.0</Y><Z>6.0</Z><X_DOT>0.4</X_DOT><Y_DOT>0.5</Y_DOT><Z_DOT>0.6</Z_DOT>
</stateVector><covarianceMatrix>
<CR_R>1.0</CR_R><CT_R>0.0</CT_R><CT_T>1.0</CT_T><CN_R>0.0</CN_R><CN_T>0.0</CN_T><CN_N>1.0</CN_N>
</covarianceMatrix></data></segment>
</body></cdm>"#;

        assert_eq!(
            parse_xml(xml),
            Err(CdmError::InvalidField {
                field: "COLLISION_PROBABILITY",
                kind: CdmInputErrorKind::NonFinite,
            })
        );
    }

    #[test]
    fn encode_rejects_non_finite_public_numeric_fields() {
        let object = CdmObject {
            object_designator: None,
            catalog_name: None,
            object_name: None,
            international_designator: None,
            object_type: None,
            operator_contact_position: None,
            operator_organization: None,
            operator_phone: None,
            operator_email: None,
            ephemeris_name: None,
            covariance_method: None,
            maneuverable: None,
            orbit_center: None,
            ref_frame: None,
            gravity_model: None,
            atmospheric_model: None,
            n_body_perturbations: None,
            solar_rad_pressure: None,
            earth_tides: None,
            intrack_thrust: None,
            state: ((1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
            covariance_rtn: [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            velocity_covariance_rtn: None,
        };
        let mut cdm = CdmKvn {
            creation_date: None,
            originator: None,
            message_id: None,
            tca: None,
            miss_distance_m: Some(f64::NAN),
            relative_speed_m_s: None,
            collision_probability: None,
            collision_probability_method: None,
            hard_body_radius_m: None,
            object1: object.clone(),
            object2: object,
        };

        assert_eq!(
            encode_kvn(&cdm),
            Err(CdmError::InvalidField {
                field: "MISS_DISTANCE",
                kind: CdmInputErrorKind::NonFinite,
            })
        );

        cdm.miss_distance_m = Some(1.0);
        cdm.object1.state.0 = (f64::INFINITY, 2.0, 3.0);
        assert_eq!(
            encode_xml(&cdm),
            Err(CdmError::InvalidField {
                field: "state",
                kind: CdmInputErrorKind::NonFinite,
            })
        );
    }

    /// A realistic two-object CDM (CCSDS 508.0-B-1 Example 2 shape) carrying the
    /// full metadata block and the complete 6x6 RTN covariance for both objects.
    /// A non-HBR `COMMENT` line is included to prove comments are canonicalized
    /// away by the round trip.
    const FULL_KVN: &str = "\
CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2010-03-12T22:31:12.000
ORIGINATOR = JSPOC
MESSAGE_ID = 201113719185
COMMENT Relative Metadata/Data
TCA = 2010-03-13T22:37:52.618
MISS_DISTANCE = 715 [m]
RELATIVE_SPEED = 14762 [m/s]
COLLISION_PROBABILITY = 4.835E-05
COLLISION_PROBABILITY_METHOD = FOSTER-1992
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 12345
CATALOG_NAME = SATCAT
OBJECT_NAME = SATELLITE A
INTERNATIONAL_DESIGNATOR = 1997-030E
OBJECT_TYPE = PAYLOAD
OPERATOR_ORGANIZATION = INTELSAT
EPHEMERIS_NAME = EPHEMERIS SATELLITE A
COVARIANCE_METHOD = CALCULATED
MANEUVERABLE = YES
REF_FRAME = EME2000
GRAVITY_MODEL = EGM-96: 36D 36O
ATMOSPHERIC_MODEL = JACCHIA 70 DCA
N_BODY_PERTURBATIONS = MOON, SUN
SOLAR_RAD_PRESSURE = NO
EARTH_TIDES = NO
INTRACK_THRUST = NO
X = 2570.097065 [km]
Y = 2244.654904 [km]
Z = 6281.497978 [km]
X_DOT = 4.418769571 [km/s]
Y_DOT = 4.833547743 [km/s]
Z_DOT = -3.526774282 [km/s]
CR_R = 4.142E+01 [m**2]
CT_R = -8.579E+00 [m**2]
CT_T = 2.533E+03 [m**2]
CN_R = -2.313E+01 [m**2]
CN_T = 1.336E+01 [m**2]
CN_N = 7.098E+01 [m**2]
CRDOT_R = 2.520E-03 [m**2/s]
CRDOT_T = -5.476E+00 [m**2/s]
CRDOT_N = 8.626E-04 [m**2/s]
CRDOT_RDOT = 5.744E-03 [m**2/s**2]
CTDOT_R = -1.006E-02 [m**2/s]
CTDOT_T = 4.041E-03 [m**2/s]
CTDOT_N = -1.359E-03 [m**2/s]
CTDOT_RDOT = -1.502E-05 [m**2/s**2]
CTDOT_TDOT = 1.049E-05 [m**2/s**2]
CNDOT_R = 1.053E-03 [m**2/s]
CNDOT_T = -3.412E-03 [m**2/s]
CNDOT_N = 1.213E-02 [m**2/s]
CNDOT_RDOT = -3.004E-06 [m**2/s**2]
CNDOT_TDOT = -1.091E-06 [m**2/s**2]
CNDOT_NDOT = 5.529E-05 [m**2/s**2]
OBJECT = OBJECT2
OBJECT_DESIGNATOR = 30337
CATALOG_NAME = SATCAT
OBJECT_NAME = FENGYUN 1C DEB
INTERNATIONAL_DESIGNATOR = 1999-025AA
OBJECT_TYPE = DEBRIS
EPHEMERIS_NAME = NONE
COVARIANCE_METHOD = CALCULATED
MANEUVERABLE = NO
REF_FRAME = EME2000
GRAVITY_MODEL = EGM-96: 36D 36O
ATMOSPHERIC_MODEL = JACCHIA 70 DCA
N_BODY_PERTURBATIONS = MOON, SUN
SOLAR_RAD_PRESSURE = YES
EARTH_TIDES = NO
INTRACK_THRUST = NO
X = 2569.540800 [km]
Y = 2245.093614 [km]
Z = 6281.599946 [km]
X_DOT = -2.888612500 [km/s]
Y_DOT = -6.007247516 [km/s]
Z_DOT = 3.328770172 [km/s]
CR_R = 1.337E+03 [m**2]
CT_R = -4.806E+04 [m**2]
CT_T = 2.492E+06 [m**2]
CN_R = -3.298E+01 [m**2]
CN_T = -7.5888E+02 [m**2]
CN_N = 7.105E+01 [m**2]
CRDOT_R = 2.591E-03 [m**2/s]
CRDOT_T = -4.152E-02 [m**2/s]
CRDOT_N = -1.784E-06 [m**2/s]
CRDOT_RDOT = 6.886E-05 [m**2/s**2]
CTDOT_R = -1.016E-02 [m**2/s]
CTDOT_T = -1.506E-04 [m**2/s]
CTDOT_N = 1.637E-03 [m**2/s]
CTDOT_RDOT = -2.987E-06 [m**2/s**2]
CTDOT_TDOT = 1.059E-05 [m**2/s**2]
CNDOT_R = 4.400E-03 [m**2/s]
CNDOT_T = 8.482E-03 [m**2/s]
CNDOT_N = 8.633E-05 [m**2/s]
CNDOT_RDOT = -1.903E-06 [m**2/s**2]
CNDOT_TDOT = -4.594E-06 [m**2/s**2]
CNDOT_NDOT = 5.178E-05 [m**2/s**2]
";

    /// Assert that the parsed CDM captured the metadata-block and velocity
    /// covariance fields, so a passing equality round trip is not a vacuous match
    /// on a struct full of `None`.
    fn assert_full_fields_captured(parsed: &CdmKvn) {
        let o1 = &parsed.object1;
        assert_eq!(o1.catalog_name.as_deref(), Some("SATCAT"));
        assert_eq!(o1.international_designator.as_deref(), Some("1997-030E"));
        assert_eq!(o1.object_type.as_deref(), Some("PAYLOAD"));
        assert_eq!(o1.operator_organization.as_deref(), Some("INTELSAT"));
        assert_eq!(o1.ephemeris_name.as_deref(), Some("EPHEMERIS SATELLITE A"));
        assert_eq!(o1.covariance_method.as_deref(), Some("CALCULATED"));
        assert_eq!(o1.maneuverable.as_deref(), Some("YES"));
        assert_eq!(o1.gravity_model.as_deref(), Some("EGM-96: 36D 36O"));
        assert_eq!(o1.n_body_perturbations.as_deref(), Some("MOON, SUN"));
        assert_eq!(o1.intrack_thrust.as_deref(), Some("NO"));
        assert_eq!(
            o1.velocity_covariance_rtn,
            Some([
                2.520e-3, -5.476e0, 8.626e-4, 5.744e-3, -1.006e-2, 4.041e-3, -1.359e-3, -1.502e-5,
                1.049e-5, 1.053e-3, -3.412e-3, 1.213e-2, -3.004e-6, -1.091e-6, 5.529e-5,
            ])
        );
        assert_eq!(parsed.object2.object_type.as_deref(), Some("DEBRIS"));
        assert!(parsed.object2.velocity_covariance_rtn.is_some());
    }

    #[test]
    fn kvn_round_trips_full_metadata_and_velocity_covariance() {
        let parsed = parse_kvn(FULL_KVN).expect("parse realistic CDM KVN");
        assert_full_fields_captured(&parsed);

        let encoded = encode_kvn(&parsed).expect("encode realistic CDM KVN");
        // The metadata block and the velocity covariance are emitted.
        assert!(encoded.contains("CATALOG_NAME = SATCAT"));
        assert!(encoded.contains("INTERNATIONAL_DESIGNATOR = 1997-030E"));
        assert!(encoded.contains("OBJECT_TYPE = PAYLOAD"));
        assert!(encoded.contains("GRAVITY_MODEL = EGM-96: 36D 36O"));
        assert!(encoded.contains("CRDOT_RDOT = "));
        assert!(encoded.contains("CNDOT_NDOT = "));
        // Comments are canonicalized away.
        assert!(!encoded.contains("COMMENT"));

        let reparsed = parse_kvn(&encoded).expect("re-parse encoded CDM KVN");
        // Every captured field survives parse -> encode -> parse, byte-for-bit.
        assert_eq!(reparsed, parsed);
    }

    /// The same physical message as [`FULL_KVN`], in the CDM XML serialization.
    const FULL_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<cdm id="CCSDS_CDM_VERS" version="1.0">
  <header>
    <CCSDS_CDM_VERS>1.0</CCSDS_CDM_VERS>
    <CREATION_DATE>2010-03-12T22:31:12.000</CREATION_DATE>
    <ORIGINATOR>JSPOC</ORIGINATOR>
    <MESSAGE_ID>201113719185</MESSAGE_ID>
  </header>
  <body>
    <relativeMetadataData>
      <COMMENT>Relative Metadata/Data</COMMENT>
      <TCA>2010-03-13T22:37:52.618</TCA>
      <MISS_DISTANCE units="m">715</MISS_DISTANCE>
      <RELATIVE_SPEED units="m/s">14762</RELATIVE_SPEED>
      <COLLISION_PROBABILITY>4.835E-05</COLLISION_PROBABILITY>
      <COLLISION_PROBABILITY_METHOD>FOSTER-1992</COLLISION_PROBABILITY_METHOD>
    </relativeMetadataData>
    <segment>
      <metadata>
        <OBJECT>OBJECT1</OBJECT>
        <OBJECT_DESIGNATOR>12345</OBJECT_DESIGNATOR>
        <CATALOG_NAME>SATCAT</CATALOG_NAME>
        <OBJECT_NAME>SATELLITE A</OBJECT_NAME>
        <INTERNATIONAL_DESIGNATOR>1997-030E</INTERNATIONAL_DESIGNATOR>
        <OBJECT_TYPE>PAYLOAD</OBJECT_TYPE>
        <OPERATOR_ORGANIZATION>INTELSAT</OPERATOR_ORGANIZATION>
        <EPHEMERIS_NAME>EPHEMERIS SATELLITE A</EPHEMERIS_NAME>
        <COVARIANCE_METHOD>CALCULATED</COVARIANCE_METHOD>
        <MANEUVERABLE>YES</MANEUVERABLE>
        <REF_FRAME>EME2000</REF_FRAME>
        <GRAVITY_MODEL>EGM-96: 36D 36O</GRAVITY_MODEL>
        <ATMOSPHERIC_MODEL>JACCHIA 70 DCA</ATMOSPHERIC_MODEL>
        <N_BODY_PERTURBATIONS>MOON, SUN</N_BODY_PERTURBATIONS>
        <SOLAR_RAD_PRESSURE>NO</SOLAR_RAD_PRESSURE>
        <EARTH_TIDES>NO</EARTH_TIDES>
        <INTRACK_THRUST>NO</INTRACK_THRUST>
      </metadata>
      <data>
        <stateVector>
          <X units="km">2570.097065</X>
          <Y units="km">2244.654904</Y>
          <Z units="km">6281.497978</Z>
          <X_DOT units="km/s">4.418769571</X_DOT>
          <Y_DOT units="km/s">4.833547743</Y_DOT>
          <Z_DOT units="km/s">-3.526774282</Z_DOT>
        </stateVector>
        <covarianceMatrix>
          <CR_R units="m**2">4.142E+01</CR_R>
          <CT_R units="m**2">-8.579E+00</CT_R>
          <CT_T units="m**2">2.533E+03</CT_T>
          <CN_R units="m**2">-2.313E+01</CN_R>
          <CN_T units="m**2">1.336E+01</CN_T>
          <CN_N units="m**2">7.098E+01</CN_N>
          <CRDOT_R units="m**2/s">2.520E-03</CRDOT_R>
          <CRDOT_T units="m**2/s">-5.476E+00</CRDOT_T>
          <CRDOT_N units="m**2/s">8.626E-04</CRDOT_N>
          <CRDOT_RDOT units="m**2/s**2">5.744E-03</CRDOT_RDOT>
          <CTDOT_R units="m**2/s">-1.006E-02</CTDOT_R>
          <CTDOT_T units="m**2/s">4.041E-03</CTDOT_T>
          <CTDOT_N units="m**2/s">-1.359E-03</CTDOT_N>
          <CTDOT_RDOT units="m**2/s**2">-1.502E-05</CTDOT_RDOT>
          <CTDOT_TDOT units="m**2/s**2">1.049E-05</CTDOT_TDOT>
          <CNDOT_R units="m**2/s">1.053E-03</CNDOT_R>
          <CNDOT_T units="m**2/s">-3.412E-03</CNDOT_T>
          <CNDOT_N units="m**2/s">1.213E-02</CNDOT_N>
          <CNDOT_RDOT units="m**2/s**2">-3.004E-06</CNDOT_RDOT>
          <CNDOT_TDOT units="m**2/s**2">-1.091E-06</CNDOT_TDOT>
          <CNDOT_NDOT units="m**2/s**2">5.529E-05</CNDOT_NDOT>
        </covarianceMatrix>
      </data>
    </segment>
    <segment>
      <metadata>
        <OBJECT>OBJECT2</OBJECT>
        <OBJECT_DESIGNATOR>30337</OBJECT_DESIGNATOR>
        <CATALOG_NAME>SATCAT</CATALOG_NAME>
        <OBJECT_NAME>FENGYUN 1C DEB</OBJECT_NAME>
        <INTERNATIONAL_DESIGNATOR>1999-025AA</INTERNATIONAL_DESIGNATOR>
        <OBJECT_TYPE>DEBRIS</OBJECT_TYPE>
        <EPHEMERIS_NAME>NONE</EPHEMERIS_NAME>
        <COVARIANCE_METHOD>CALCULATED</COVARIANCE_METHOD>
        <MANEUVERABLE>NO</MANEUVERABLE>
        <REF_FRAME>EME2000</REF_FRAME>
        <GRAVITY_MODEL>EGM-96: 36D 36O</GRAVITY_MODEL>
        <ATMOSPHERIC_MODEL>JACCHIA 70 DCA</ATMOSPHERIC_MODEL>
        <N_BODY_PERTURBATIONS>MOON, SUN</N_BODY_PERTURBATIONS>
        <SOLAR_RAD_PRESSURE>YES</SOLAR_RAD_PRESSURE>
        <EARTH_TIDES>NO</EARTH_TIDES>
        <INTRACK_THRUST>NO</INTRACK_THRUST>
      </metadata>
      <data>
        <stateVector>
          <X units="km">2569.540800</X>
          <Y units="km">2245.093614</Y>
          <Z units="km">6281.599946</Z>
          <X_DOT units="km/s">-2.888612500</X_DOT>
          <Y_DOT units="km/s">-6.007247516</Y_DOT>
          <Z_DOT units="km/s">3.328770172</Z_DOT>
        </stateVector>
        <covarianceMatrix>
          <CR_R units="m**2">1.337E+03</CR_R>
          <CT_R units="m**2">-4.806E+04</CT_R>
          <CT_T units="m**2">2.492E+06</CT_T>
          <CN_R units="m**2">-3.298E+01</CN_R>
          <CN_T units="m**2">-7.5888E+02</CN_T>
          <CN_N units="m**2">7.105E+01</CN_N>
          <CRDOT_R units="m**2/s">2.591E-03</CRDOT_R>
          <CRDOT_T units="m**2/s">-4.152E-02</CRDOT_T>
          <CRDOT_N units="m**2/s">-1.784E-06</CRDOT_N>
          <CRDOT_RDOT units="m**2/s**2">6.886E-05</CRDOT_RDOT>
          <CTDOT_R units="m**2/s">-1.016E-02</CTDOT_R>
          <CTDOT_T units="m**2/s">-1.506E-04</CTDOT_T>
          <CTDOT_N units="m**2/s">1.637E-03</CTDOT_N>
          <CTDOT_RDOT units="m**2/s**2">-2.987E-06</CTDOT_RDOT>
          <CTDOT_TDOT units="m**2/s**2">1.059E-05</CTDOT_TDOT>
          <CNDOT_R units="m**2/s">4.400E-03</CNDOT_R>
          <CNDOT_T units="m**2/s">8.482E-03</CNDOT_T>
          <CNDOT_N units="m**2/s">8.633E-05</CNDOT_N>
          <CNDOT_RDOT units="m**2/s**2">-1.903E-06</CNDOT_RDOT>
          <CNDOT_TDOT units="m**2/s**2">-4.594E-06</CNDOT_TDOT>
          <CNDOT_NDOT units="m**2/s**2">5.178E-05</CNDOT_NDOT>
        </covarianceMatrix>
      </data>
    </segment>
  </body>
</cdm>"#;

    #[test]
    fn xml_round_trips_full_metadata_and_velocity_covariance() {
        let parsed = parse_xml(FULL_XML).expect("parse realistic CDM XML");
        assert_full_fields_captured(&parsed);

        let encoded = encode_xml(&parsed).expect("encode realistic CDM XML");
        assert!(encoded.contains("<CATALOG_NAME>SATCAT</CATALOG_NAME>"));
        assert!(encoded.contains("<OBJECT_TYPE>PAYLOAD</OBJECT_TYPE>"));
        assert!(encoded.contains("<GRAVITY_MODEL>EGM-96: 36D 36O</GRAVITY_MODEL>"));
        assert!(encoded.contains("<CRDOT_RDOT units=\"m**2/s**2\">"));
        assert!(encoded.contains("<CNDOT_NDOT units=\"m**2/s**2\">"));
        // The relativeMetadataData COMMENT element is canonicalized away.
        assert!(!encoded.contains("<COMMENT>"));

        let reparsed = parse_xml(&encoded).expect("re-parse encoded CDM XML");
        assert_eq!(reparsed, parsed);
    }

    #[test]
    fn kvn_and_xml_parse_the_realistic_message_identically() {
        let from_kvn = parse_kvn(FULL_KVN).expect("parse KVN");
        let from_xml = parse_xml(FULL_XML).expect("parse XML");
        // The same physical message in either serialization parses field-for-field
        // to the same IR, including the full metadata block and 6x6 covariance.
        assert_eq!(from_kvn, from_xml);
    }
}
