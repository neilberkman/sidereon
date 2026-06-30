//! CCSDS Orbit Mean-Elements Message (OMM) parser, encoder, and SGP4 bridge.
//!
//! OMM (CCSDS 502.0-B) is the modern replacement for the TLE: it carries the
//! same SGP4/SDP4 mean elements (mean motion, eccentricity, inclination, RAAN,
//! argument of perigee, mean anomaly, B\*, epoch, ...) plus richer metadata
//! (object name, NORAD id, reference frame, time system, element-set theory).
//! CelesTrak and Space-Track serve general-perturbations (GP) data as OMM in
//! three interchangeable encodings: KVN (`KEY = VALUE` lines), XML, and JSON.
//!
//! This module follows the format-agnostic design used on the Elixir side: a
//! single canonical container ([`Omm`]) holds the CCSDS field set as plain data
//! with documented units, and the per-encoding readers/writers map onto it. The
//! element values flow into the validated SGP4 path through [`Omm::to_element_set`]
//! (consumed by [`Satellite::from_elements`]) so an OMM drives SGP4 without
//! downgrading its full epoch into the legacy TLE year/day representation.
//!
//! ## TLE-derived field quantization
//!
//! An OMM carries a full UTC calendar `EPOCH`, which is converted directly to
//! SGP4's split Julian date. B\* and the second mean-motion derivative are still
//! TLE-derived GP parameters:
//!
//! - **B\* and the second mean-motion derivative.** A TLE stores these in its
//!   "assumed decimal" field (five significant mantissa digits and a power-of-ten
//!   exponent), and that quantized value is what SGP4 actually receives. OMM
//!   prints the same quantities as plain decimals, so the bridge re-quantizes
//!   them onto the assumed-decimal grid via [`crate::astro::tle`].

use crate::astro::sgp4::{self, ElementSet, Error as Sgp4Error, Satellite, Sgp4InputErrorKind};
use crate::astro::tle;
use crate::astro::xml;
use crate::validate;
use roxmltree::Document;
use std::fmt::{self, Write as _};

/// Leaf CCSDS field element names carried in an OMM XML message (the format-set
/// version, `CCSDS_OMM_VERS`, is an attribute on `<omm>` and is handled
/// separately). Decoding maps each onto the shared `(key, value)` field set, the
/// same one the KVN tokenizer produces.
const FIELD_TAGS: &[&str] = &[
    "CREATION_DATE",
    "ORIGINATOR",
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "TIME_SYSTEM",
    "MEAN_ELEMENT_THEORY",
    "EPOCH",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
];

/// UTC calendar epoch as carried by an OMM, split into the components a KVN/XML
/// `EPOCH` (or JSON `EPOCH`) string spells out. Stored as integers so the epoch
/// re-encodes losslessly and converts directly to the SGP4 epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OmmEpoch {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
    /// Fractional second expressed in whole microseconds (0..=999_999).
    pub microsecond: u32,
}

/// Canonical, format-agnostic OMM container.
///
/// Pure data: it knows nothing about KVN/XML/JSON serialization. The numeric
/// element values use standard astrodynamic units (angles in degrees, mean
/// motion in revolutions/day, its derivatives in rev/day^2 and rev/day^3, B\*
/// in inverse earth-radii) and are stored as directly parsed `f64`s, so every
/// encoding decodes to the same value.
#[derive(Debug, Clone, PartialEq)]
pub struct Omm {
    // -- Header / metadata --
    pub ccsds_omm_vers: String,
    pub creation_date: Option<String>,
    pub originator: Option<String>,
    pub object_name: Option<String>,
    /// International designator, CCSDS form (e.g. `"1998-067A"`).
    pub object_id: Option<String>,
    pub center_name: Option<String>,
    pub ref_frame: Option<String>,
    pub time_system: Option<String>,
    pub mean_element_theory: Option<String>,

    // -- Mean elements --
    pub epoch: OmmEpoch,
    /// Mean motion, revolutions per day.
    pub mean_motion: f64,
    /// Eccentricity, dimensionless, in [0, 1).
    pub eccentricity: f64,
    /// Inclination, degrees.
    pub inclination_deg: f64,
    /// Right ascension of the ascending node, degrees.
    pub ra_of_asc_node_deg: f64,
    /// Argument of pericenter, degrees.
    pub arg_of_pericenter_deg: f64,
    /// Mean anomaly, degrees.
    pub mean_anomaly_deg: f64,

    // -- TLE-derived parameters --
    pub ephemeris_type: i32,
    pub classification_type: String,
    pub norad_cat_id: u32,
    pub element_set_no: i32,
    pub rev_at_epoch: i64,
    /// SGP4 drag term B\*, inverse earth-radii.
    pub bstar: f64,
    /// First derivative of mean motion, rev/day^2.
    pub mean_motion_dot: f64,
    /// Second derivative of mean motion, rev/day^3.
    pub mean_motion_ddot: f64,
}

/// Failure modes of the OMM readers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OmmError {
    /// A required field was absent from the message.
    MissingField(&'static str),
    /// A decoded scalar field failed boundary validation.
    InvalidField {
        field: &'static str,
        kind: OmmInputErrorKind,
    },
    /// A numeric/integer field could not be parsed.
    Field(String),
    /// The `EPOCH` value was malformed.
    Epoch(String),
}

/// OMM boundary-validation failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OmmInputErrorKind {
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

impl fmt::Display for OmmInputErrorKind {
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

impl From<&validate::FieldError> for OmmInputErrorKind {
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

impl fmt::Display for OmmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OmmError::MissingField(name) => write!(f, "OMM missing required field {name}"),
            OmmError::InvalidField { field, kind } => {
                write!(f, "invalid OMM field {field}: {kind}")
            }
            OmmError::Field(msg) => write!(f, "OMM field error: {msg}"),
            OmmError::Epoch(msg) => write!(f, "OMM epoch error: {msg}"),
        }
    }
}

impl std::error::Error for OmmError {}

// ── KVN ──────────────────────────────────────────────────────────────

/// Parse a CCSDS OMM in KVN (`KEY = VALUE`) encoding into an [`Omm`].
///
/// Blank lines and lines without an `=` are ignored; keys and values are
/// trimmed. Numeric values accept the CelesTrak forms, including a leading
/// decimal point (`.0004737`) and scientific notation (`.17172E-3`).
pub fn parse_kvn(text: &str) -> Result<Omm, OmmError> {
    let map = crate::format::kvn::FieldMap::parse(text);
    Omm::from_field_map(&map)
}

/// Encode an [`Omm`] as a CCSDS OMM KVN message.
///
/// Numeric values use their shortest round-tripping decimal form, so parsing the
/// output reproduces the same `f64`s. The epoch is emitted to microseconds.
pub fn encode_kvn(omm: &Omm) -> String {
    let mut out = String::new();
    let header = crate::astro::ndm::NdmHeader {
        vers: omm.ccsds_omm_vers.clone(),
        creation_date: omm.creation_date.clone(),
        originator: omm.originator.clone(),
    };
    for line in header.write_kvn("CCSDS_OMM_VERS") {
        out.push_str(&line);
        out.push('\n');
    }

    let mut kv = |key: &str, value: &str| {
        out.push_str(key);
        out.push_str(" = ");
        out.push_str(value);
        out.push('\n');
    };

    kv("OBJECT_NAME", omm.object_name.as_deref().unwrap_or(""));
    kv("OBJECT_ID", omm.object_id.as_deref().unwrap_or(""));
    kv("CENTER_NAME", omm.center_name.as_deref().unwrap_or(""));
    kv("REF_FRAME", omm.ref_frame.as_deref().unwrap_or(""));
    kv("TIME_SYSTEM", omm.time_system.as_deref().unwrap_or(""));
    kv(
        "MEAN_ELEMENT_THEORY",
        omm.mean_element_theory.as_deref().unwrap_or(""),
    );
    kv("EPOCH", &omm.epoch.to_iso8601());
    kv("MEAN_MOTION", &fmt_num(omm.mean_motion));
    kv("ECCENTRICITY", &fmt_num(omm.eccentricity));
    kv("INCLINATION", &fmt_num(omm.inclination_deg));
    kv("RA_OF_ASC_NODE", &fmt_num(omm.ra_of_asc_node_deg));
    kv("ARG_OF_PERICENTER", &fmt_num(omm.arg_of_pericenter_deg));
    kv("MEAN_ANOMALY", &fmt_num(omm.mean_anomaly_deg));
    kv("EPHEMERIS_TYPE", &omm.ephemeris_type.to_string());
    kv("CLASSIFICATION_TYPE", &omm.classification_type);
    kv("NORAD_CAT_ID", &omm.norad_cat_id.to_string());
    kv("ELEMENT_SET_NO", &omm.element_set_no.to_string());
    kv("REV_AT_EPOCH", &omm.rev_at_epoch.to_string());
    kv("BSTAR", &fmt_num(omm.bstar));
    kv("MEAN_MOTION_DOT", &fmt_num(omm.mean_motion_dot));
    kv("MEAN_MOTION_DDOT", &fmt_num(omm.mean_motion_ddot));
    out
}

// ── XML ──────────────────────────────────────────────────────────────

/// Parse a CCSDS OMM in XML encoding into an [`Omm`].
///
/// Uses the `roxmltree` DOM reader (which handles the `<?xml?>` declaration,
/// namespaces, comments, and entity decoding) and then reads the known CCSDS
/// leaf elements by name into the shared `(key, value)` field set, exactly the
/// representation the KVN tokenizer produces, so both encodings flow through the
/// single field mapping. The format-set version is taken from the `version`
/// attribute on `<omm>`.
pub fn parse_xml(text: &str) -> Result<Omm, OmmError> {
    let doc = Document::parse(text).map_err(|e| OmmError::Field(format!("malformed XML: {e}")))?;
    let mut fields: Vec<(String, String)> = Vec::new();

    if let Some(omm_el) = doc
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "omm")
    {
        if let Some(version) = omm_el.attribute("version") {
            fields.push(("CCSDS_OMM_VERS".to_string(), version.trim().to_string()));
        }
    }

    for node in doc.descendants().filter(roxmltree::Node::is_element) {
        let name = node.tag_name().name();
        if FIELD_TAGS.contains(&name) {
            let value = node.text().unwrap_or("").trim().to_string();
            fields.push((name.to_string(), value));
        }
    }

    let map = crate::format::kvn::FieldMap::from_pairs(fields);
    Omm::from_field_map(&map)
}

/// Encode an [`Omm`] as a CCSDS OMM XML message, following the CelesTrak/`ndm`
/// document layout. Numeric values use their shortest round-tripping form and
/// text values are XML-escaped, so parsing the output reproduces the same [`Omm`].
pub fn encode_xml(omm: &Omm) -> String {
    fn elem(name: &str, value: &str) -> String {
        format!("<{name}>{value}</{name}>")
    }
    let opt = |value: &Option<String>| xml::escape_opt(value);

    let mut s = String::new();
    s.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    s.push_str("<ndm>\n");
    let _ = writeln!(
        s,
        "<omm id=\"CCSDS_OMM_VERS\" version=\"{}\">",
        xml::escape(&omm.ccsds_omm_vers)
    );

    s.push_str("<header>");
    s.push_str(&elem("CREATION_DATE", &opt(&omm.creation_date)));
    s.push_str(&elem("ORIGINATOR", &opt(&omm.originator)));
    s.push_str("</header>\n");

    s.push_str("<body><segment>\n<metadata>");
    s.push_str(&elem("OBJECT_NAME", &opt(&omm.object_name)));
    s.push_str(&elem("OBJECT_ID", &opt(&omm.object_id)));
    s.push_str(&elem("CENTER_NAME", &opt(&omm.center_name)));
    s.push_str(&elem("REF_FRAME", &opt(&omm.ref_frame)));
    s.push_str(&elem("TIME_SYSTEM", &opt(&omm.time_system)));
    s.push_str(&elem("MEAN_ELEMENT_THEORY", &opt(&omm.mean_element_theory)));
    s.push_str("</metadata>\n<data>\n<meanElements>");

    s.push_str(&elem("EPOCH", &omm.epoch.to_iso8601()));
    s.push_str(&elem("MEAN_MOTION", &fmt_num(omm.mean_motion)));
    s.push_str(&elem("ECCENTRICITY", &fmt_num(omm.eccentricity)));
    s.push_str(&elem("INCLINATION", &fmt_num(omm.inclination_deg)));
    s.push_str(&elem("RA_OF_ASC_NODE", &fmt_num(omm.ra_of_asc_node_deg)));
    s.push_str(&elem(
        "ARG_OF_PERICENTER",
        &fmt_num(omm.arg_of_pericenter_deg),
    ));
    s.push_str(&elem("MEAN_ANOMALY", &fmt_num(omm.mean_anomaly_deg)));
    s.push_str("</meanElements>\n<tleParameters>");

    s.push_str(&elem("EPHEMERIS_TYPE", &omm.ephemeris_type.to_string()));
    s.push_str(&elem(
        "CLASSIFICATION_TYPE",
        &xml::escape(&omm.classification_type),
    ));
    s.push_str(&elem("NORAD_CAT_ID", &omm.norad_cat_id.to_string()));
    s.push_str(&elem("ELEMENT_SET_NO", &omm.element_set_no.to_string()));
    s.push_str(&elem("REV_AT_EPOCH", &omm.rev_at_epoch.to_string()));
    s.push_str(&elem("BSTAR", &fmt_num(omm.bstar)));
    s.push_str(&elem("MEAN_MOTION_DOT", &fmt_num(omm.mean_motion_dot)));
    s.push_str(&elem("MEAN_MOTION_DDOT", &fmt_num(omm.mean_motion_ddot)));
    s.push_str("</tleParameters>\n</data>\n</segment></body>\n</omm>\n</ndm>\n");
    s
}

// ── JSON ─────────────────────────────────────────────────────────────

/// Parse a CCSDS/CelesTrak OMM in JSON encoding into an [`Omm`].
///
/// Accepts either a single object or an array of objects (CelesTrak GP queries
/// return an array); the first record is taken. Each member is mapped onto the
/// shared `(key, value)` field set - numbers stringified, strings taken verbatim
/// (so the Space-Track quirk of quoting numeric values is handled) - then flows
/// through the single field mapping. Requires the `json` feature.
pub fn parse_json(text: &str) -> Result<Omm, OmmError> {
    use serde_json::Value;

    let value: Value =
        serde_json::from_str(text).map_err(|e| OmmError::Field(format!("malformed JSON: {e}")))?;
    let object = match &value {
        Value::Array(items) => items
            .first()
            .ok_or_else(|| OmmError::Field("empty JSON array".to_string()))?,
        Value::Object(_) => &value,
        _ => {
            return Err(OmmError::Field(
                "expected a JSON object or array".to_string(),
            ))
        }
    };
    omm_from_json_value(object)
}

/// Map a single JSON OMM object onto the shared `(key, value)` field set and
/// parse it. Used by both [`parse_json`] and [`parse_json_array`].
fn omm_from_json_value(object: &serde_json::Value) -> Result<Omm, OmmError> {
    let map = object
        .as_object()
        .ok_or_else(|| OmmError::Field("expected a JSON object".to_string()))?;
    let fields: Vec<(String, String)> = map
        .iter()
        .map(|(key, value)| (key.clone(), json_scalar_to_string(value)))
        .collect();
    let map = crate::format::kvn::FieldMap::from_pairs(fields);
    Omm::from_field_map(&map)
}

/// The result of parsing a CelesTrak OMM JSON array: every OMM that parsed,
/// plus a count of array elements that were skipped.
#[derive(Debug, Clone, PartialEq)]
pub struct OmmArray {
    /// The successfully parsed OMMs, in array order.
    pub omms: Vec<Omm>,
    /// How many array elements were skipped because they were not a parseable
    /// OMM object (a non-object element, or an object that failed field
    /// validation). Mirrors [`crate::astro::sgp4::TleFile::skipped`]: lets
    /// callers tell an empty array (`omms` empty, `skipped == 0`) apart from one
    /// whose every element was malformed (`skipped > 0`) without aborting the
    /// whole parse on one bad entry. No fabricated OMM is emitted in their place.
    pub skipped: usize,
}

/// Parse a CelesTrak OMM JSON array into every contained [`Omm`].
///
/// CelesTrak GP queries return a JSON array of OMM objects; this reads all of
/// them (a lone object is accepted as a one-element array) through the same
/// field mapping [`parse_json`] uses. An individual array element that is not a
/// valid OMM object is skipped and counted in [`OmmArray::skipped`] rather than
/// aborting the whole array. A malformed top-level document (not valid JSON, or
/// neither an object nor an array) is still an error. Requires the `json`
/// feature.
pub fn parse_json_array(text: &str) -> Result<OmmArray, OmmError> {
    use serde_json::Value;

    let value: Value =
        serde_json::from_str(text).map_err(|e| OmmError::Field(format!("malformed JSON: {e}")))?;
    let items: &[Value] = match &value {
        Value::Array(items) => items.as_slice(),
        Value::Object(_) => std::slice::from_ref(&value),
        _ => {
            return Err(OmmError::Field(
                "expected a JSON object or array".to_string(),
            ))
        }
    };

    let mut omms = Vec::with_capacity(items.len());
    let mut skipped = 0usize;
    for object in items {
        match omm_from_json_value(object) {
            Ok(omm) => omms.push(omm),
            Err(_) => skipped += 1,
        }
    }
    Ok(OmmArray { omms, skipped })
}

/// Encode an [`Omm`] as a CCSDS/CelesTrak OMM JSON object.
///
/// Numeric element values are emitted as JSON numbers (round-tripping the exact
/// `f64`), strings as JSON strings, and the epoch as an ISO-8601 string, so
/// parsing the output reproduces the same [`Omm`]. Requires the `json` feature.
pub fn encode_json(omm: &Omm) -> String {
    use serde_json::{Map, Number, Value};

    let num = |x: f64| Number::from_f64(x).map_or(Value::Null, Value::Number);
    let opt = |value: &Option<String>| value.clone().map_or(Value::Null, Value::String);

    let mut map = Map::new();
    map.insert(
        "CCSDS_OMM_VERS".into(),
        Value::String(omm.ccsds_omm_vers.clone()),
    );
    map.insert("CREATION_DATE".into(), opt(&omm.creation_date));
    map.insert("ORIGINATOR".into(), opt(&omm.originator));
    map.insert("OBJECT_NAME".into(), opt(&omm.object_name));
    map.insert("OBJECT_ID".into(), opt(&omm.object_id));
    map.insert("CENTER_NAME".into(), opt(&omm.center_name));
    map.insert("REF_FRAME".into(), opt(&omm.ref_frame));
    map.insert("TIME_SYSTEM".into(), opt(&omm.time_system));
    map.insert("MEAN_ELEMENT_THEORY".into(), opt(&omm.mean_element_theory));
    map.insert("EPOCH".into(), Value::String(omm.epoch.to_iso8601()));
    map.insert("MEAN_MOTION".into(), num(omm.mean_motion));
    map.insert("ECCENTRICITY".into(), num(omm.eccentricity));
    map.insert("INCLINATION".into(), num(omm.inclination_deg));
    map.insert("RA_OF_ASC_NODE".into(), num(omm.ra_of_asc_node_deg));
    map.insert("ARG_OF_PERICENTER".into(), num(omm.arg_of_pericenter_deg));
    map.insert("MEAN_ANOMALY".into(), num(omm.mean_anomaly_deg));
    map.insert(
        "EPHEMERIS_TYPE".into(),
        Value::Number(omm.ephemeris_type.into()),
    );
    map.insert(
        "CLASSIFICATION_TYPE".into(),
        Value::String(omm.classification_type.clone()),
    );
    map.insert(
        "NORAD_CAT_ID".into(),
        Value::Number(omm.norad_cat_id.into()),
    );
    map.insert(
        "ELEMENT_SET_NO".into(),
        Value::Number(omm.element_set_no.into()),
    );
    map.insert(
        "REV_AT_EPOCH".into(),
        Value::Number(omm.rev_at_epoch.into()),
    );
    map.insert("BSTAR".into(), num(omm.bstar));
    map.insert("MEAN_MOTION_DOT".into(), num(omm.mean_motion_dot));
    map.insert("MEAN_MOTION_DDOT".into(), num(omm.mean_motion_ddot));
    Value::Object(map).to_string()
}

/// Render a JSON scalar as the string the shared field mapping consumes. Numbers
/// use their canonical decimal form; strings pass through; null becomes empty.
fn json_scalar_to_string(value: &serde_json::Value) -> String {
    use serde_json::Value;
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

// ── Encoding auto-detect ─────────────────────────────────────────────

/// Parse an OMM in any supported encoding, detecting it from the leading
/// non-whitespace character: `<` is XML, `{` or `[` is JSON, anything else is
/// KVN. JSON requires the `json` feature; without it a JSON document returns an
/// error rather than being misread.
pub fn parse(text: &str) -> Result<Omm, OmmError> {
    match text.trim_start().chars().next() {
        Some('<') => parse_xml(text),
        Some('{') | Some('[') => parse_json_detected(text),
        _ => parse_kvn(text),
    }
}

/// Parse a single CCSDS `EPOCH` string field to the canonical [`OmmEpoch`].
///
/// The accepted form is `YYYY-MM-DDThh:mm:ss[.ffffff][Z]`, interpreted under the
/// UTC-like civil-second policy (the OMM default when no `TIME_SYSTEM` is
/// declared, matching how a CelesTrak GP `EPOCH` is read). This is the single
/// public entry point a thin binding (for example the Elixir constellation NIF)
/// delegates to instead of hand-rolling the split; it wraps the same
/// [`OmmEpoch`]/`NdmEpoch` parser the full OMM decode uses, so it produces
/// byte-identical components.
pub fn parse_epoch(text: &str) -> Result<OmmEpoch, OmmError> {
    OmmEpoch::parse(text, validate::CivilSecondPolicy::UtcLike)
}

fn parse_json_detected(text: &str) -> Result<Omm, OmmError> {
    parse_json(text)
}

// ── Field mapping (shared by every encoding) ─────────────────────────

impl Omm {
    /// Build an [`Omm`] from the decoded CCSDS field set: a list of
    /// `(key, value)` string pairs as produced by any of the encodings. This is
    /// the single place the CCSDS field names map onto the canonical container.
    pub(crate) fn from_field_map(map: &crate::format::kvn::FieldMap) -> Result<Omm, OmmError> {
        let get = |key: &str| map.get(key);

        let time_system = xml_text(get("TIME_SYSTEM"), "TIME_SYSTEM")?;
        let epoch = OmmEpoch::parse(
            get("EPOCH").ok_or(OmmError::MissingField("EPOCH"))?,
            omm_civil_second_policy(time_system.as_deref()),
        )?;

        Ok(Omm {
            ccsds_omm_vers: xml_text_or_default(get("CCSDS_OMM_VERS"), "CCSDS_OMM_VERS", "2.0")?,
            creation_date: xml_text(get("CREATION_DATE"), "CREATION_DATE")?,
            originator: xml_text(get("ORIGINATOR"), "ORIGINATOR")?,
            object_name: xml_text(get("OBJECT_NAME"), "OBJECT_NAME")?,
            object_id: xml_text(get("OBJECT_ID"), "OBJECT_ID")?,
            center_name: xml_text(get("CENTER_NAME"), "CENTER_NAME")?,
            ref_frame: xml_text(get("REF_FRAME"), "REF_FRAME")?,
            time_system,
            mean_element_theory: xml_text(get("MEAN_ELEMENT_THEORY"), "MEAN_ELEMENT_THEORY")?,
            epoch,
            mean_motion: req_num(get("MEAN_MOTION"), "MEAN_MOTION")?,
            eccentricity: req_num(get("ECCENTRICITY"), "ECCENTRICITY")?,
            inclination_deg: req_num(get("INCLINATION"), "INCLINATION")?,
            ra_of_asc_node_deg: req_num(get("RA_OF_ASC_NODE"), "RA_OF_ASC_NODE")?,
            arg_of_pericenter_deg: req_num(get("ARG_OF_PERICENTER"), "ARG_OF_PERICENTER")?,
            mean_anomaly_deg: req_num(get("MEAN_ANOMALY"), "MEAN_ANOMALY")?,
            ephemeris_type: opt_int(get("EPHEMERIS_TYPE"), "EPHEMERIS_TYPE")?.unwrap_or(0),
            classification_type: xml_text_or_default(
                get("CLASSIFICATION_TYPE"),
                "CLASSIFICATION_TYPE",
                "U",
            )?,
            norad_cat_id: req_int(get("NORAD_CAT_ID"), "NORAD_CAT_ID")?,
            element_set_no: opt_int(get("ELEMENT_SET_NO"), "ELEMENT_SET_NO")?.unwrap_or(999),
            rev_at_epoch: opt_int(get("REV_AT_EPOCH"), "REV_AT_EPOCH")?.unwrap_or(0),
            bstar: req_num(get("BSTAR"), "BSTAR")?,
            mean_motion_dot: req_num(get("MEAN_MOTION_DOT"), "MEAN_MOTION_DOT")?,
            mean_motion_ddot: req_num(get("MEAN_MOTION_DDOT"), "MEAN_MOTION_DDOT")?,
        })
    }
}

fn xml_text(value: Option<&str>, field: &'static str) -> Result<Option<String>, OmmError> {
    value
        .map(|value| xml_text_value(value, field).map(str::to_string))
        .transpose()
}

fn xml_text_or_default(
    value: Option<&str>,
    field: &'static str,
    default: &'static str,
) -> Result<String, OmmError> {
    xml_text_value(value.unwrap_or(default), field).map(str::to_string)
}

fn xml_text_value<'a>(value: &'a str, field: &'static str) -> Result<&'a str, OmmError> {
    if let Some(ch) = xml::first_illegal_xml_1_0_char(value) {
        return Err(OmmError::Field(format!(
            "field {field} contains XML-illegal character U+{:04X}",
            ch as u32
        )));
    }
    Ok(value)
}

// ── SGP4 bridge ──────────────────────────────────────────────────────

impl Omm {
    /// Convert the canonical OMM elements into the SGP4 [`ElementSet`] consumed
    /// by [`Satellite::from_elements`].
    ///
    /// The epoch is converted directly from the OMM calendar timestamp into
    /// SGP4's split Julian date, preserving years outside the TLE pivot range.
    /// B\* and the second mean-motion derivative are quantized onto the TLE
    /// assumed-decimal grid because those GP parameters originate in that field
    /// format.
    pub fn to_element_set(&self) -> Result<ElementSet, OmmError> {
        validate_omm_bridge(self)?;
        Ok(ElementSet {
            epoch: self.epoch.sgp4_julian_date(),
            bstar: tle::assumed_decimal_quantize(self.bstar),
            mean_motion_dot: self.mean_motion_dot,
            mean_motion_double_dot: tle::assumed_decimal_quantize(self.mean_motion_ddot),
            eccentricity: self.eccentricity,
            argument_of_perigee_deg: self.arg_of_pericenter_deg,
            inclination_deg: self.inclination_deg,
            mean_anomaly_deg: self.mean_anomaly_deg,
            mean_motion_rev_per_day: self.mean_motion,
            right_ascension_deg: self.ra_of_asc_node_deg,
            catalog_number: self.norad_cat_id,
        })
    }
}

impl Satellite {
    /// Build a propagation-ready [`Satellite`] from an [`Omm`].
    ///
    /// Bridges the OMM mean elements into the validated SGP4 element path via
    /// [`Omm::to_element_set`].
    pub fn from_omm(omm: &Omm) -> Result<Self, Sgp4Error> {
        let elements = omm.to_element_set().map_err(map_omm_bridge_to_sgp4)?;
        Self::from_elements(&elements)
    }
}

fn validate_omm_bridge(omm: &Omm) -> Result<(), OmmError> {
    validate::finite_positive(omm.mean_motion, "mean_motion").map_err(map_omm_field_error)?;
    validate::finite_in_range_exclusive_upper(omm.eccentricity, 0.0, 1.0, "eccentricity")
        .map_err(map_omm_field_error)?;
    validate::finite(omm.inclination_deg, "inclination_deg").map_err(map_omm_field_error)?;
    validate::finite(omm.ra_of_asc_node_deg, "ra_of_asc_node_deg").map_err(map_omm_field_error)?;
    validate::finite(omm.arg_of_pericenter_deg, "arg_of_pericenter_deg")
        .map_err(map_omm_field_error)?;
    validate::finite(omm.mean_anomaly_deg, "mean_anomaly_deg").map_err(map_omm_field_error)?;
    validate::finite(omm.bstar, "bstar").map_err(map_omm_field_error)?;
    validate::finite(omm.mean_motion_dot, "mean_motion_dot").map_err(map_omm_field_error)?;
    validate::finite(omm.mean_motion_ddot, "mean_motion_ddot").map_err(map_omm_field_error)?;
    Ok(())
}

fn map_omm_bridge_to_sgp4(error: OmmError) -> Sgp4Error {
    match error {
        OmmError::InvalidField { field, kind } => Sgp4Error::InvalidInput {
            field,
            kind: match kind {
                OmmInputErrorKind::NonFinite => Sgp4InputErrorKind::NonFinite,
                OmmInputErrorKind::NotPositive => Sgp4InputErrorKind::NotPositive,
                OmmInputErrorKind::Negative => Sgp4InputErrorKind::Negative,
                OmmInputErrorKind::OutOfRange => Sgp4InputErrorKind::OutOfRange,
                OmmInputErrorKind::Missing => Sgp4InputErrorKind::Missing,
                OmmInputErrorKind::FloatParse => Sgp4InputErrorKind::FloatParse,
                OmmInputErrorKind::IntParse => Sgp4InputErrorKind::IntParse,
                OmmInputErrorKind::InvalidCivilDate => Sgp4InputErrorKind::InvalidCivilDate,
                OmmInputErrorKind::InvalidCivilTime => Sgp4InputErrorKind::InvalidCivilTime,
            },
        },
        other => Sgp4Error::InvalidTle(other.to_string()),
    }
}

// ── Epoch ────────────────────────────────────────────────────────────

impl OmmEpoch {
    /// Parse a CCSDS `EPOCH` value (`YYYY-MM-DDThh:mm:ss[.ffffff][Z]`, UTC).
    fn parse(text: &str, second_policy: validate::CivilSecondPolicy) -> Result<OmmEpoch, OmmError> {
        let e = crate::astro::ndm::NdmEpoch::parse(text, second_policy)
            .map_err(|err| map_omm_epoch_field_error(err, text))?;
        Ok(OmmEpoch {
            year: e.year,
            month: e.month,
            day: e.day,
            hour: e.hour,
            minute: e.minute,
            second: e.second,
            microsecond: e.microsecond,
        })
    }

    /// Convert directly to the SGP4 split Julian date from the full OMM
    /// calendar timestamp.
    fn sgp4_julian_date(&self) -> sgp4::JulianDate {
        sgp4::sgp4_julian_date_from_calendar(
            self.year,
            self.month as i32,
            self.day as i32,
            self.hour as i32,
            self.minute as i32,
            self.second as f64 + self.microsecond as f64 / 1_000_000.0,
        )
    }

    /// Format as a CCSDS `EPOCH` string with microsecond precision.
    fn to_iso8601(&self) -> String {
        format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}",
            self.year, self.month, self.day, self.hour, self.minute, self.second, self.microsecond
        )
    }
}

// ── Numeric helpers ──────────────────────────────────────────────────

fn omm_civil_second_policy(time_system: Option<&str>) -> validate::CivilSecondPolicy {
    let Some(label) = time_system.map(str::trim).filter(|label| !label.is_empty()) else {
        return validate::CivilSecondPolicy::UtcLike;
    };
    if label.eq_ignore_ascii_case("UTC")
        || label.eq_ignore_ascii_case("GLO")
        || label.eq_ignore_ascii_case("GLONASS")
    {
        validate::CivilSecondPolicy::UtcLike
    } else {
        validate::CivilSecondPolicy::Continuous
    }
}

fn req_num(value: Option<&str>, field: &'static str) -> Result<f64, OmmError> {
    let value = value.ok_or(OmmError::MissingField(field))?;
    parse_num(value, field)
}

fn parse_num(value: &str, field: &'static str) -> Result<f64, OmmError> {
    validate::strict_f64(value, field).map_err(map_omm_field_error)
}

fn req_int<T>(value: Option<&str>, field: &'static str) -> Result<T, OmmError>
where
    T: std::str::FromStr,
{
    let value = value.ok_or(OmmError::MissingField(field))?;
    parse_int(value, field)
}

fn opt_int<T>(value: Option<&str>, field: &'static str) -> Result<Option<T>, OmmError>
where
    T: std::str::FromStr,
{
    value.map(|v| parse_int(v, field)).transpose()
}

fn parse_int<T>(value: &str, field: &'static str) -> Result<T, OmmError>
where
    T: std::str::FromStr,
{
    validate::strict_int::<T>(value, field).map_err(map_omm_field_error)
}

fn map_omm_field_error(error: validate::FieldError) -> OmmError {
    OmmError::InvalidField {
        field: error.field(),
        kind: OmmInputErrorKind::from(&error),
    }
}

fn map_omm_epoch_field_error(error: validate::FieldError, full: &str) -> OmmError {
    match error {
        validate::FieldError::Missing { .. }
        | validate::FieldError::FloatParse { .. }
        | validate::FieldError::IntParse { .. } => {
            OmmError::Epoch(format!("invalid seconds in {full:?}"))
        }
        _ => map_omm_field_error(error),
    }
}

/// Shortest decimal form of a value that round-trips back to the same `f64`.
fn fmt_num(value: f64) -> String {
    format!("{value}")
}

#[cfg(all(test, sidereon_repo_tests))]
mod tests {
    use super::*;

    const ISS_KVN: &str = include_str!("../../tests/fixtures/omm/25544.kvn");
    const ISS_XML: &str = include_str!("../../tests/fixtures/omm/25544.xml");

    /// Reduce an OMM to its canonical orbital + catalog content (the fields the
    /// Elixir `Sidereon.Elements` struct carries), blanking the free-text header
    /// metadata that CelesTrak emits inconsistently across encodings: it labels
    /// the element theory `SGP/SGP4` in KVN but `SGP4` in XML/JSON, and its JSON
    /// omits `CENTER_NAME`/`REF_FRAME`/`TIME_SYSTEM` entirely. Cross-encoding
    /// identity is asserted on this canonical content, which must match exactly.
    fn canonical(omm: &Omm) -> Omm {
        Omm {
            ccsds_omm_vers: String::new(),
            creation_date: None,
            originator: None,
            center_name: None,
            ref_frame: None,
            time_system: None,
            mean_element_theory: None,
            ..omm.clone()
        }
    }

    fn kvn_with_field(field: &str, value: &str) -> String {
        kvn_with_fields(&[(field, value)])
    }

    fn kvn_with_fields(fields: &[(&str, &str)]) -> String {
        ISS_KVN
            .lines()
            .map(|line| match line.split_once('=') {
                Some((key, _)) => fields
                    .iter()
                    .find(|(field, _)| key.trim() == *field)
                    .map_or_else(
                        || line.to_string(),
                        |(field, value)| format!("{field} = {value}"),
                    ),
                _ => line.to_string(),
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn kvn_without_field(field: &str) -> String {
        ISS_KVN
            .lines()
            .filter(|line| match line.split_once('=') {
                Some((key, _)) => key.trim() != field,
                None => true,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn parses_iss_kvn_fields() {
        let omm = parse_kvn(ISS_KVN).unwrap();
        assert_eq!(omm.ccsds_omm_vers, "2.0");
        assert_eq!(omm.object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(omm.object_id.as_deref(), Some("1998-067A"));
        assert_eq!(omm.norad_cat_id, 25544);
        assert_eq!(omm.mean_motion, 15.49273435);
        assert_eq!(omm.eccentricity, 0.0004737);
        assert_eq!(omm.inclination_deg, 51.6332);
        assert_eq!(omm.bstar, 0.00017172);
        assert_eq!(omm.mean_motion_dot, 9.113e-5);
        assert_eq!(omm.mean_motion_ddot, 0.0);
        assert_eq!(
            omm.epoch,
            OmmEpoch {
                year: 2026,
                month: 6,
                day: 17,
                hour: 4,
                minute: 32,
                second: 52,
                microsecond: 99296,
            }
        );
    }

    #[test]
    fn parse_kvn_requires_drag_terms() {
        for field in ["BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT"] {
            assert_eq!(
                parse_kvn(&kvn_without_field(field)),
                Err(OmmError::MissingField(field))
            );
        }
    }

    #[test]
    fn parse_kvn_rejects_non_finite_drag_terms() {
        for field in ["BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT"] {
            assert_eq!(
                parse_kvn(&kvn_with_field(field, "NaN")),
                Err(OmmError::InvalidField {
                    field,
                    kind: OmmInputErrorKind::NonFinite,
                })
            );
        }
    }

    #[test]
    fn parse_kvn_rejects_negative_norad_catalog_id() {
        assert_eq!(
            parse_kvn(&kvn_with_field("NORAD_CAT_ID", "-1")),
            Err(OmmError::InvalidField {
                field: "NORAD_CAT_ID",
                kind: OmmInputErrorKind::IntParse,
            })
        );
    }

    #[test]
    fn parse_kvn_rejects_oversized_norad_catalog_id() {
        assert_eq!(
            parse_kvn(&kvn_with_field("NORAD_CAT_ID", "4294967296")),
            Err(OmmError::InvalidField {
                field: "NORAD_CAT_ID",
                kind: OmmInputErrorKind::IntParse,
            })
        );
    }

    #[test]
    fn parse_kvn_rejects_invalid_civil_epoch() {
        assert_eq!(
            parse_kvn(&kvn_with_field("EPOCH", "2026-02-30T04:32:52.099296")),
            Err(OmmError::InvalidField {
                field: "civil datetime",
                kind: OmmInputErrorKind::InvalidCivilDate,
            })
        );
        assert_eq!(
            parse_kvn(&kvn_with_field("EPOCH", "2026-06-17T24:00:00.000000")),
            Err(OmmError::InvalidField {
                field: "civil datetime",
                kind: OmmInputErrorKind::InvalidCivilTime,
            })
        );
    }

    #[test]
    fn parse_kvn_accepts_utc_leap_second_epoch() {
        let omm = parse_kvn(&kvn_with_field("EPOCH", "2016-12-31T23:59:60.000000Z"))
            .expect("OMM leap-second epoch");
        assert_eq!(
            omm.epoch,
            OmmEpoch {
                year: 2016,
                month: 12,
                day: 31,
                hour: 23,
                minute: 59,
                second: 60,
                microsecond: 0,
            }
        );
    }

    #[test]
    fn parse_kvn_rejects_gps_time_leap_second_epoch() {
        assert_eq!(
            parse_kvn(&kvn_with_fields(&[
                ("TIME_SYSTEM", "GPS"),
                ("EPOCH", "2016-12-31T23:59:60.000000Z"),
            ])),
            Err(OmmError::InvalidField {
                field: "civil datetime",
                kind: OmmInputErrorKind::InvalidCivilTime,
            })
        );
    }

    #[test]
    fn parse_kvn_rejects_invalid_leap_second_range() {
        assert!(parse_kvn(&kvn_with_field("EPOCH", "2016-12-31T23:59:61.000000Z")).is_err());
        assert!(parse_kvn(&kvn_with_field("EPOCH", "2016-12-31T23:59:-1.000000Z")).is_err());
    }

    #[test]
    fn parse_kvn_requires_fractional_epoch_digits() {
        let omm = parse_kvn(&kvn_with_field("EPOCH", "2026-06-17T04:32:52.500"))
            .expect("fractional epoch");
        assert_eq!(omm.epoch.microsecond, 500_000);

        let omm = parse_kvn(&kvn_with_field("EPOCH", "2026-06-17T04:32:52.5Z"))
            .expect("fractional epoch with UTC suffix");
        assert_eq!(omm.epoch.microsecond, 500_000);

        for epoch in [
            "2026-06-17T04:32:52.abc",
            "2026-06-17T04:32:52.abcZ",
            "2026-06-17T04:32:52.5x",
            "2026-06-17T04:32:52.5xZ",
            "2026-06-17T04:32:52.",
        ] {
            assert!(
                matches!(
                    parse_kvn(&kvn_with_field("EPOCH", epoch)),
                    Err(OmmError::Epoch(_))
                ),
                "{epoch} must be rejected"
            );
        }
    }

    #[test]
    fn parse_kvn_carries_rounded_fractional_epoch_seconds() {
        let omm = parse_kvn(&kvn_with_field("EPOCH", "2026-06-17T04:32:52.9999995"))
            .expect("fractional epoch carry");
        assert_eq!(
            omm.epoch,
            OmmEpoch {
                year: 2026,
                month: 6,
                day: 17,
                hour: 4,
                minute: 32,
                second: 53,
                microsecond: 0,
            }
        );
        assert!(
            encode_kvn(&omm).contains("EPOCH = 2026-06-17T04:32:53.000000"),
            "carried epoch must encode with six fractional digits"
        );
    }

    #[test]
    fn parse_kvn_carries_continuous_time_fractional_epoch_across_day() {
        let omm = parse_kvn(&kvn_with_fields(&[
            ("TIME_SYSTEM", "GPS"),
            ("EPOCH", "2026-06-17T23:59:59.9999995"),
        ]))
        .expect("continuous-time fractional epoch carry");
        assert_eq!(
            omm.epoch,
            OmmEpoch {
                year: 2026,
                month: 6,
                day: 18,
                hour: 0,
                minute: 0,
                second: 0,
                microsecond: 0,
            }
        );
    }

    #[test]
    fn parse_kvn_carries_continuous_time_fractional_epoch_across_year() {
        let ordinary = parse_kvn(&kvn_with_fields(&[
            ("TIME_SYSTEM", "GPS"),
            ("EPOCH", "2026-12-31T23:59:58.123456"),
        ]))
        .expect("ordinary continuous-time epoch");
        assert_eq!(
            ordinary.epoch,
            OmmEpoch {
                year: 2026,
                month: 12,
                day: 31,
                hour: 23,
                minute: 59,
                second: 58,
                microsecond: 123_456,
            }
        );
        assert!(
            encode_kvn(&ordinary).contains("EPOCH = 2026-12-31T23:59:58.123456"),
            "ordinary epoch must encode unchanged"
        );

        let carried = parse_kvn(&kvn_with_fields(&[
            ("TIME_SYSTEM", "GPS"),
            ("EPOCH", "2026-12-31T23:59:59.9999995"),
        ]))
        .expect("continuous-time fractional epoch carry across year");
        assert_eq!(
            carried.epoch,
            OmmEpoch {
                year: 2027,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
                microsecond: 0,
            }
        );
        assert!(
            encode_kvn(&carried).contains("EPOCH = 2027-01-01T00:00:00.000000"),
            "carried epoch must encode with the next year"
        );
    }

    #[test]
    fn kvn_round_trips_through_struct() {
        let omm = parse_kvn(ISS_KVN).unwrap();
        let reparsed = parse_kvn(&encode_kvn(&omm)).unwrap();
        assert_eq!(omm, reparsed);
    }

    #[test]
    fn xml_matches_kvn_orbital_content() {
        let kvn = parse_kvn(ISS_KVN).unwrap();
        let xml = parse_xml(ISS_XML).unwrap();
        assert_eq!(canonical(&kvn), canonical(&xml));
    }

    #[test]
    fn xml_round_trips_through_struct() {
        let omm = parse_xml(ISS_XML).unwrap();
        let reparsed = parse_xml(&encode_xml(&omm)).unwrap();
        assert_eq!(omm, reparsed);
    }

    #[test]
    fn parse_kvn_rejects_xml_illegal_text_controls() {
        let err = parse_kvn(&kvn_with_field("MEAN_ELEMENT_THEORY", "SGP\u{0005}SGP4"))
            .expect_err("XML-illegal control characters must not enter OMM text fields");
        assert_eq!(
            err,
            OmmError::Field(
                "field MEAN_ELEMENT_THEORY contains XML-illegal character U+0005".to_string()
            )
        );

        let omm = parse_kvn(&kvn_with_field("MEAN_ELEMENT_THEORY", "SGP\tSGP4"))
            .expect("XML-legal text control must remain valid");
        let reparsed = parse_xml(&encode_xml(&omm)).expect("encoded OMM must remain valid XML");
        assert_eq!(omm, reparsed);
    }

    #[test]
    fn xml_round_trip_preserves_carriage_returns_in_text_values() {
        for value in ["SGP\rSGP4", "SGP\r\nSGP4"] {
            let mut omm = parse_kvn(ISS_KVN).expect("base OMM must parse");
            omm.mean_element_theory = Some(value.to_string());
            let encoded = encode_xml(&omm);
            assert!(encoded.contains("&#xD;"));
            assert!(!encoded.contains('\r'));
            let reparsed = parse_xml(&encoded).expect("encoded OMM must remain valid XML");
            assert_eq!(omm.mean_element_theory, reparsed.mean_element_theory);
            assert_eq!(omm, reparsed);
        }
    }

    #[test]
    fn json_matches_kvn_orbital_content() {
        const ISS_JSON: &str = include_str!("../../tests/fixtures/omm/25544.json");
        let kvn = parse_kvn(ISS_KVN).unwrap();
        let json = parse_json(ISS_JSON).unwrap();
        assert_eq!(canonical(&kvn), canonical(&json));
    }

    #[test]
    fn json_round_trips_through_struct() {
        const ISS_JSON: &str = include_str!("../../tests/fixtures/omm/25544.json");
        let omm = parse_json(ISS_JSON).unwrap();
        let reparsed = parse_json(&encode_json(&omm)).unwrap();
        assert_eq!(omm, reparsed);
    }

    #[test]
    fn parse_auto_detects_encoding() {
        let from_kvn = parse(ISS_KVN).unwrap();
        let from_xml = parse(ISS_XML).unwrap();
        assert_eq!(parse_kvn(ISS_KVN).unwrap(), from_kvn);
        assert_eq!(parse_xml(ISS_XML).unwrap(), from_xml);
        assert_eq!(canonical(&from_kvn), canonical(&from_xml));
    }

    #[test]
    fn parse_auto_detects_json_array() {
        const ISS_JSON: &str = include_str!("../../tests/fixtures/omm/25544.json");
        // CelesTrak JSON is a top-level array.
        assert_eq!(parse(ISS_JSON).unwrap(), parse_json(ISS_JSON).unwrap());
    }

    #[test]
    fn parse_json_array_skips_malformed_objects_and_counts_them() {
        // A CelesTrak-shaped array with two good OMMs interleaved with a
        // non-object element and a malformed object (missing the required drag
        // terms). One bad object must not reject the whole array: the good
        // records survive and the skips are surfaced in `skipped`.
        let good = |norad: u32, id: &str| {
            format!(
                r#"{{"OBJECT_NAME":"SAT","OBJECT_ID":"{id}","EPOCH":"2026-06-17T04:32:52.099296","MEAN_MOTION":15.49273435,"ECCENTRICITY":0.0004737,"INCLINATION":51.6332,"RA_OF_ASC_NODE":300.0813,"ARG_OF_PERICENTER":195.1146,"MEAN_ANOMALY":164.9702,"EPHEMERIS_TYPE":0,"CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":{norad},"ELEMENT_SET_NO":999,"REV_AT_EPOCH":57175,"BSTAR":0.00017172,"MEAN_MOTION_DOT":9.113e-5,"MEAN_MOTION_DDOT":0}}"#
            )
        };
        let text = format!(
            "[{}, \"not an object\", {{\"OBJECT_NAME\":\"BROKEN\",\"NORAD_CAT_ID\":99999}}, {}]",
            good(25544, "1998-067A"),
            good(25545, "1998-067B"),
        );

        let result = parse_json_array(&text).expect("array with bad entries must still parse");
        assert_eq!(result.skipped, 2, "the string and the malformed object");
        let norads: Vec<u32> = result.omms.iter().map(|o| o.norad_cat_id).collect();
        assert_eq!(norads, vec![25544, 25545], "both good OMMs must survive");
    }

    #[test]
    fn bstar_quantizes_onto_assumed_decimal_grid() {
        // OMM B* is the plain-decimal 0.00017172; the SGP4 element set must carry
        // the assumed-decimal value 0.17172e-3 the TLE actually feeds SGP4.
        let omm = parse_kvn(ISS_KVN).unwrap();
        let es = omm.to_element_set().expect("valid OMM bridge");
        assert_eq!(es.bstar, 0.17172 * 10.0_f64.powi(-3));
        assert_ne!(es.bstar, omm.bstar);
    }

    #[test]
    fn to_element_set_rejects_invalid_bridge_fields() {
        let mut omm = parse_kvn(ISS_KVN).unwrap();
        omm.mean_motion = f64::NAN;
        assert_eq!(
            omm.to_element_set(),
            Err(OmmError::InvalidField {
                field: "mean_motion",
                kind: OmmInputErrorKind::NonFinite
            })
        );

        let mut omm = parse_kvn(ISS_KVN).unwrap();
        omm.eccentricity = 1.0;
        assert_eq!(
            omm.to_element_set(),
            Err(OmmError::InvalidField {
                field: "eccentricity",
                kind: OmmInputErrorKind::OutOfRange
            })
        );
    }

    #[test]
    fn from_omm_preserves_epoch_year_outside_tle_pivot_range() {
        let omm = parse_kvn(&kvn_with_field("EPOCH", "2057-01-01T00:00:00.000000"))
            .expect("future OMM epoch");
        let sat = Satellite::from_omm(&omm).expect("OMM with full-year epoch must initialize");

        let epoch = sat.epoch_jd();
        let actual_jd = epoch.0 + epoch.1;
        let expected_jd = crate::astro::time::scales::julian_day_number(2057, 1, 1) as f64 - 0.5;
        let aliased_1957_jd =
            crate::astro::time::scales::julian_day_number(1957, 1, 1) as f64 - 0.5;

        assert!(
            (actual_jd - expected_jd).abs() < 1.0e-9,
            "OMM epoch JD {actual_jd} must match the true 2057 epoch {expected_jd}",
        );
        assert!(
            (actual_jd - aliased_1957_jd).abs() > 36_000.0,
            "OMM epoch JD {actual_jd} must not alias to 1957 {aliased_1957_jd}",
        );
    }

    #[test]
    fn from_omm_uses_parser_rounded_year_end_epoch_directly() {
        for (epoch, expected_year) in [
            ("2021-12-31T23:59:59.9999995", 2022),
            ("2020-12-31T23:59:59.9999995", 2021),
        ] {
            let omm = parse_kvn(&kvn_with_field("EPOCH", epoch)).expect("year-end OMM epoch");
            assert_eq!(omm.epoch.year, expected_year);
            assert_eq!(omm.epoch.month, 1);
            assert_eq!(omm.epoch.day, 1);
            assert_eq!(omm.epoch.hour, 0);
            assert_eq!(omm.epoch.minute, 0);
            assert_eq!(omm.epoch.second, 0);
            assert_eq!(omm.epoch.microsecond, 0);

            let sat = Satellite::from_omm(&omm).expect("rounded year-end OMM must initialize");
            let epoch_jd = sat.epoch_jd();
            let actual_jd = epoch_jd.0 + epoch_jd.1;
            let expected_jd =
                crate::astro::time::scales::julian_day_number(expected_year, 1, 1) as f64 - 0.5;

            assert!(
                (actual_jd - expected_jd).abs() < 1.0e-9,
                "{epoch} carried to JD {actual_jd}, expected {expected_jd}",
            );
        }
    }

    #[test]
    fn from_omm_rejects_invalid_sgp4_element_fields() {
        let mut omm = parse_kvn(ISS_KVN).unwrap();
        omm.mean_motion = f64::NAN;
        let err = Satellite::from_omm(&omm).expect_err("non-finite mean motion must error");
        assert_eq!(
            err,
            Sgp4Error::InvalidInput {
                field: "mean_motion",
                kind: crate::astro::sgp4::Sgp4InputErrorKind::NonFinite,
            }
        );

        let mut omm = parse_kvn(ISS_KVN).unwrap();
        omm.eccentricity = 1.0;
        let err = Satellite::from_omm(&omm).expect_err("eccentricity >= 1 must error");
        assert_eq!(
            err,
            Sgp4Error::InvalidInput {
                field: "eccentricity",
                kind: crate::astro::sgp4::Sgp4InputErrorKind::OutOfRange,
            }
        );
    }
}
