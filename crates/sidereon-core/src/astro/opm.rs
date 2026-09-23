//! CCSDS Orbit Parameter Message (OPM) KVN and XML reader/writer.
//!
//! OPM date/time values are preserved as text. The parser validates required
//! scalar presence and numeric fields, but it does not normalize epochs.
//!
//! The readers retain every item of CCSDS 502.0-B-3 tables 3-1 to 3-3: the
//! optional header `CLASSIFICATION` and `MESSAGE_ID`, `REF_FRAME_EPOCH`, the
//! comments of every block, and `USER_DEFINED_*` parameters, and the writers
//! write them back. In KVN a comment belongs to the block of the keyword that
//! follows it and is written at the start of that block (7.8.7); comments
//! after the last keyword belong to that keyword's block.
//!
//! A numeric KVN value may carry the unit table 3-3 gives its keyword in
//! brackets, and an XML element may state it as a `units` attribute (7.7.1.1,
//! 8.8.11); a unit that contradicts the table is refused by name. Text values,
//! such as object names, are kept verbatim. A keyword repeated in one block
//! with a different value is refused by name, and so is a keyword or element
//! the tables do not define that carries a value (3.2.2.2, 3.2.3.1, 3.2.4.2).

use crate::astro::covariance::{Covariance6, Covariance6Error};
use crate::astro::ndm::{
    self, covariance6_unit, mirror_lower_triangle6, read_lower_triangle6, FieldMap, KvnLine,
    UnitMismatch, COVARIANCE6_KEYS,
};
use crate::astro::xml;
use crate::format::fmtnum::fmt_num;
use crate::validate;
use roxmltree::{Document, Node};
use std::fmt;

pub use crate::astro::ndm::TextIssue;

const COMMENT: &str = "COMMENT";
const OPM_VERSION_KEY: &str = "CCSDS_OPM_VERS";
const USER_DEFINED_PREFIX: &str = "USER_DEFINED_";
const MAN_EPOCH_IGNITION: &str = "MAN_EPOCH_IGNITION";

/// Header keywords, CCSDS 502.0-B-3 table 3-1 (`COMMENT` is handled apart).
const HEADER_KEYS: &[&str] = &[
    OPM_VERSION_KEY,
    "CLASSIFICATION",
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
];
/// Metadata keywords, table 3-2.
const METADATA_KEYS: &[&str] = &[
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
];
/// State vector keywords, table 3-3.
const STATE_KEYS: &[&str] = &["EPOCH", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"];
/// Osculating Keplerian element keywords, table 3-3.
const KEPLERIAN_KEYS: &[&str] = &[
    "SEMI_MAJOR_AXIS",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "TRUE_ANOMALY",
    "MEAN_ANOMALY",
    "GM",
];
/// Spacecraft parameter keywords, table 3-3.
const SPACECRAFT_KEYS: &[&str] = &[
    "MASS",
    "SOLAR_RAD_AREA",
    "SOLAR_RAD_COEFF",
    "DRAG_AREA",
    "DRAG_COEFF",
];
/// Maneuver parameter keywords, table 3-3.
const MANEUVER_KEYS: &[&str] = &[
    MAN_EPOCH_IGNITION,
    "MAN_DURATION",
    "MAN_DELTA_MASS",
    "MAN_REF_FRAME",
    "MAN_DV_1",
    "MAN_DV_2",
    "MAN_DV_3",
];
/// Covariance reference-frame keyword; the matrix keywords are
/// [`COVARIANCE6_KEYS`].
const COV_REF_FRAME: &str = "COV_REF_FRAME";

/// Canonical, format-agnostic OPM container.
#[derive(Debug, Clone, PartialEq)]
pub struct Opm {
    /// Version copied from `CCSDS_OPM_VERS` in KVN or the `<opm version>`/
    /// `CCSDS_OPM_VERS` value in XML. [`parse_kvn`] rejects an empty KVN
    /// version, and both encoders write this value back.
    pub ccsds_opm_vers: String,
    /// Header comments, written after `CCSDS_OPM_VERS` (502.0-B-3 7.8.7).
    pub comments: Vec<String>,
    /// Optional header `CLASSIFICATION` text (table 3-1), written only when
    /// present.
    pub classification: Option<String>,
    /// Optional `CREATION_DATE` header text copied by both readers and emitted
    /// as text by both encoders; `None` becomes an empty header value.
    pub creation_date: Option<String>,
    /// Optional `ORIGINATOR` header text copied by both readers and emitted as
    /// text by both encoders; `None` becomes an empty header value.
    pub originator: Option<String>,
    /// Optional header `MESSAGE_ID` text (table 3-1), written only when
    /// present.
    pub message_id: Option<String>,
    /// Required metadata assembled by `parse_metadata` from the metadata
    /// fields and emitted before state data by both encoders.
    pub metadata: OpmMetadata,
    /// Required Cartesian state assembled by `parse_state` and emitted by both
    /// encoders after [`Opm::metadata`].
    pub state: OpmState,
    /// Optional Keplerian block. Present when a Keplerian keyword carries a
    /// value or the block has comments (KVN) or when `keplerianElements` is
    /// given (XML); a present block must contain one, and only one, anomaly
    /// field.
    pub keplerian: Option<OpmKeplerian>,
    /// Optional spacecraft-parameters block. Present when any of its keywords
    /// occurs, even with a blank value, or the block has comments (KVN), or
    /// when `spacecraftParameters` is given (XML); each numeric field may
    /// still be absent.
    pub spacecraft: Option<OpmSpacecraft>,
    /// Optional covariance block. Present when `COV_REF_FRAME` or a matrix
    /// keyword carries a value or the block has comments (KVN), or when
    /// `covarianceMatrix` is given (XML); a present block must provide all 21
    /// matrix entries, each a finite number.
    pub covariance: Option<OpmCovariance>,
    /// Maneuver blocks in source order. Both readers append them in encounter
    /// order, and both encoders iterate this vector in that same order.
    pub maneuvers: Vec<OpmManeuver>,
    /// `USER_DEFINED_*` parameters in source order (table 3-3, 505.0-B-3
    /// 4.10), with values kept as verbatim text.
    pub user_defined: Vec<OpmUserDefined>,
    /// Comments of the user-defined-parameters block, written before the first
    /// `USER_DEFINED_*` keyword.
    pub user_defined_comments: Vec<String>,
}

/// OPM metadata block.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmMetadata {
    /// Metadata comments, written before `OBJECT_NAME` (502.0-B-3 7.8.7).
    pub comments: Vec<String>,
    /// Required `OBJECT_NAME` text copied by `parse_metadata` and emitted under
    /// the same key by both encoders.
    pub object_name: String,
    /// Required `OBJECT_ID` text copied by `parse_metadata` and emitted under
    /// the same key by both encoders.
    pub object_id: String,
    /// Required `CENTER_NAME` text copied by `parse_metadata` and emitted under
    /// the same key by both encoders.
    pub center_name: String,
    /// Required `REF_FRAME` label copied by `parse_metadata` and emitted under
    /// the same key by both encoders.
    pub ref_frame: String,
    /// Optional `REF_FRAME_EPOCH` text (table 3-2), retained without date
    /// conversion and written only when present.
    pub ref_frame_epoch: Option<String>,
    /// Required `TIME_SYSTEM` label copied by `parse_metadata` and emitted under
    /// the same key by both encoders.
    pub time_system: String,
}

/// OPM Cartesian state vector.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmState {
    /// State vector comments, written before `EPOCH`.
    pub comments: Vec<String>,
    /// Required `EPOCH` text. The readers do not convert or normalize it, and
    /// both encoders write the stored text back.
    pub epoch: String,
    /// Strict numeric values from `X`, `Y`, and `Z`, in that order; encoders map
    /// the entries back to those keys without conversion.
    pub position_km: [f64; 3],
    /// Strict numeric values from `X_DOT`, `Y_DOT`, and `Z_DOT`, in that order;
    /// encoders map the entries back to those keys without conversion.
    pub velocity_km_s: [f64; 3],
}

/// Optional OPM Keplerian elements.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmKeplerian {
    /// Keplerian block comments, written before `SEMI_MAJOR_AXIS`.
    pub comments: Vec<String>,
    /// Strict numeric `SEMI_MAJOR_AXIS` value stored directly and emitted under
    /// the same field name.
    pub semi_major_axis_km: f64,
    /// Strict numeric `ECCENTRICITY` value stored directly and emitted under
    /// the same field name; no orbital-range check is applied here.
    pub eccentricity: f64,
    /// Strict numeric `INCLINATION` value stored directly and emitted under the
    /// same field name.
    pub inclination_deg: f64,
    /// Strict numeric `RA_OF_ASC_NODE` value stored directly and emitted under
    /// the same field name.
    pub ra_of_asc_node_deg: f64,
    /// Strict numeric `ARG_OF_PERICENTER` value stored directly and emitted
    /// under the same field name.
    pub arg_of_pericenter_deg: f64,
    /// Selected by exactly one of `TRUE_ANOMALY` and `MEAN_ANOMALY`; missing or
    /// simultaneous fields produce [`OpmError::Field`], and the selected key is
    /// emitted on encoding.
    pub anomaly: OpmAnomaly,
    /// Strict numeric `GM` value stored directly and emitted under the same
    /// field name.
    pub gm_km3_s2: f64,
}

/// OPM true or mean anomaly.
#[derive(Debug, Clone, PartialEq)]
pub enum OpmAnomaly {
    /// Value from `TRUE_ANOMALY`, accepted only when `MEAN_ANOMALY` is absent;
    /// encoders write it as `TRUE_ANOMALY`.
    True(f64),
    /// Value from `MEAN_ANOMALY`, accepted only when `TRUE_ANOMALY` is absent;
    /// encoders write it as `MEAN_ANOMALY`.
    Mean(f64),
}

/// Optional OPM spacecraft parameters.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OpmSpacecraft {
    /// Spacecraft block comments, written before the first parameter.
    pub comments: Vec<String>,
    /// Optional strict numeric `MASS` value; `None` causes the encoders to omit
    /// the `MASS` field.
    pub mass_kg: Option<f64>,
    /// Optional strict numeric `SOLAR_RAD_AREA` value; `None` causes the
    /// encoders to omit the field.
    pub solar_rad_area_m2: Option<f64>,
    /// Optional strict numeric `SOLAR_RAD_COEFF` value; `None` causes the
    /// encoders to omit the field.
    pub solar_rad_coeff: Option<f64>,
    /// Optional strict numeric `DRAG_AREA` value; `None` causes the encoders to
    /// omit the field.
    pub drag_area_m2: Option<f64>,
    /// Optional strict numeric `DRAG_COEFF` value; `None` causes the encoders to
    /// omit the field.
    pub drag_coeff: Option<f64>,
}

/// Optional OPM 6x6 covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmCovariance {
    /// Covariance block comments, written before `COV_REF_FRAME` or `CX_X`.
    pub comments: Vec<String>,
    /// Optional `COV_REF_FRAME` label retained on the covariance block and
    /// emitted only when present.
    pub cov_ref_frame: Option<String>,
    /// The 21 lower-triangle values exactly as read, in keyword order `CX_X`,
    /// `CY_X`, `CY_Y` ... `CZ_DOT_Z_DOT` (table 3-3): km^2 for two position
    /// components, km^2/s for one position and one velocity component, km^2/s^2
    /// for two velocity components. Each is finite; the matrix is not checked
    /// for positive semidefiniteness on read, since one printed to a few
    /// digits can fall short of it only through that rounding, and the writers
    /// write these values back. [`OpmCovariance::to_covariance6`] validates it
    /// for a consumer that needs a covariance.
    pub lower_triangle: [f64; 21],
}

impl OpmCovariance {
    /// The symmetric matrix as a validated [`Covariance6`]: refused when it is
    /// not symmetric positive semidefinite within the [`Covariance6`]
    /// tolerance.
    pub fn to_covariance6(&self) -> Result<Covariance6, Covariance6Error> {
        Covariance6::try_from_matrix(mirror_lower_triangle6(&self.lower_triangle))
    }
}

/// One OPM maneuver block. Every field is mandatory in CCSDS 502.0-B when a
/// maneuver is present, including `MAN_REF_FRAME`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmManeuver {
    /// Maneuver block comments, written before `MAN_EPOCH_IGNITION`.
    pub comments: Vec<String>,
    /// Required `MAN_EPOCH_IGNITION` text retained without date conversion and
    /// emitted under the same key.
    pub epoch_ignition: String,
    /// Strict numeric `MAN_DURATION` value stored directly and emitted under
    /// the same key.
    pub duration_s: f64,
    /// Strict numeric `MAN_DELTA_MASS` value, including its sign, stored
    /// directly and emitted under the same key.
    pub delta_mass_kg: f64,
    /// Required `MAN_REF_FRAME` text stored for this maneuver and emitted under
    /// the same key.
    pub ref_frame: String,
    /// Strict numeric values from `MAN_DV_1`, `MAN_DV_2`, and `MAN_DV_3`, in that
    /// order; encoders map the entries back to those keys.
    pub dv_km_s: [f64; 3],
}

/// One `USER_DEFINED_*` parameter (CCSDS 502.0-B-3 table 3-3, 505.0-B-3 4.10).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpmUserDefined {
    /// The text after `USER_DEFINED_` in the KVN keyword, which is the XML
    /// `parameter` attribute.
    pub parameter: String,
    /// The value text, verbatim, including any units it states.
    pub value: String,
}

/// Failure modes of the OPM readers and writers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpmError {
    /// A required field was absent from the message.
    MissingField(&'static str),
    /// A decoded scalar field failed validation.
    InvalidField {
        /// Static source-field label retained from validation and included in
        /// the error's display text.
        field: &'static str,
        /// Validation category mapped from the shared [`validate::FieldError`].
        kind: OpmInputErrorKind,
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
    /// A stated unit contradicts the unit CCSDS 502.0-B-3 table 3-3 defines
    /// for the keyword (7.7.1.1, 8.8.11).
    UnitMismatch {
        /// The keyword whose value carried the unit.
        field: String,
        /// The stated unit.
        unit: String,
        /// The table unit, or `None` for a dimensionless or text keyword.
        expected: Option<&'static str>,
    },
    /// An XML document holds more than one OPM; each OPM describes a single
    /// object (502.0-B-3 3.1.5).
    MultipleMessages {
        /// The number of OPM messages in the document.
        count: usize,
    },
    /// A KVN keyword or XML element that tables 3-1 to 3-3 do not define at
    /// that position and that carries a value (3.2.2.2, 3.2.3.1, 3.2.4.2). XML
    /// elements are named with their parent element.
    UnknownField(String),
    /// A KVN line that is not blank, a comment, or a `keyword = value`
    /// assignment (502.0-B-3 7.3.1, 7.4.1).
    MalformedLine {
        /// One-based line number.
        line: usize,
        /// The trimmed line text.
        text: String,
    },
    /// A writer cannot write a text value so that it reads back unchanged.
    UnwritableText {
        /// The keyword, or `COMMENT`.
        field: String,
        /// The text that cannot be written.
        value: String,
        /// Why the text would not read back unchanged.
        issue: TextIssue,
    },
}

/// OPM boundary-validation failure category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpmInputErrorKind {
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

impl fmt::Display for OpmInputErrorKind {
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

impl From<&validate::FieldError> for OpmInputErrorKind {
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

impl fmt::Display for OpmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OpmError::MissingField(name) => write!(f, "OPM missing required field {name}"),
            OpmError::InvalidField { field, kind } => {
                write!(f, "invalid OPM field {field}: {kind}")
            }
            OpmError::Field(msg) => write!(f, "OPM field error: {msg}"),
            OpmError::DuplicateField {
                field,
                first,
                second,
            } => write!(
                f,
                "OPM keyword {field} occurs with different values {first:?} and {second:?}"
            ),
            OpmError::UnitMismatch {
                field,
                unit,
                expected,
            } => write!(
                f,
                "OPM keyword {field} states unit [{unit}], expected {}",
                ndm::expected_unit_label(*expected)
            ),
            OpmError::MultipleMessages { count } => {
                write!(f, "XML holds {count} OPM messages; the reader reads one")
            }
            OpmError::UnknownField(name) => write!(f, "OPM has no keyword {name}"),
            OpmError::MalformedLine { line, text } => write!(
                f,
                "OPM line {line} is not a comment or keyword assignment: {text:?}"
            ),
            OpmError::UnwritableText {
                field,
                value,
                issue,
            } => write!(f, "OPM {field} value {value:?} {issue}"),
        }
    }
}

impl std::error::Error for OpmError {}

/// The logical groups of OPM keywords: the header (table 3-1), the metadata
/// (table 3-2), and the data blocks of table 3-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpmBlock {
    Header,
    Metadata,
    State,
    Keplerian,
    Spacecraft,
    Covariance,
    Maneuver,
    UserDefined,
}

impl OpmBlock {
    fn of_keyword(key: &str) -> Option<Self> {
        if HEADER_KEYS.contains(&key) {
            Some(Self::Header)
        } else if METADATA_KEYS.contains(&key) {
            Some(Self::Metadata)
        } else if STATE_KEYS.contains(&key) {
            Some(Self::State)
        } else if KEPLERIAN_KEYS.contains(&key) {
            Some(Self::Keplerian)
        } else if SPACECRAFT_KEYS.contains(&key) {
            Some(Self::Spacecraft)
        } else if key == COV_REF_FRAME || COVARIANCE6_KEYS.contains(&key) {
            Some(Self::Covariance)
        } else if MANEUVER_KEYS.contains(&key) {
            Some(Self::Maneuver)
        } else if key.starts_with(USER_DEFINED_PREFIX) {
            Some(Self::UserDefined)
        } else {
            None
        }
    }
}

/// Decoded OPM content before the typed mapping.
#[derive(Default)]
struct OpmFields {
    /// Keywords outside the maneuver blocks, values stripped of any unit.
    pairs: Vec<(String, String)>,
    header_comments: Vec<String>,
    metadata_comments: Vec<String>,
    state_comments: Vec<String>,
    keplerian_comments: Vec<String>,
    spacecraft_comments: Vec<String>,
    covariance_comments: Vec<String>,
    user_defined_comments: Vec<String>,
    /// XML: the block element is present.
    keplerian_block: bool,
    spacecraft_block: bool,
    covariance_block: bool,
    maneuvers: Vec<ManeuverFields>,
    user_defined: Vec<OpmUserDefined>,
}

/// One maneuver block before the typed mapping.
#[derive(Default)]
struct ManeuverFields {
    pairs: Vec<(String, String)>,
    comments: Vec<String>,
}

impl OpmFields {
    fn comments_mut(&mut self, block: OpmBlock) -> &mut Vec<String> {
        match block {
            OpmBlock::Header => &mut self.header_comments,
            OpmBlock::Metadata => &mut self.metadata_comments,
            OpmBlock::State => &mut self.state_comments,
            OpmBlock::Keplerian => &mut self.keplerian_comments,
            OpmBlock::Spacecraft => &mut self.spacecraft_comments,
            OpmBlock::Covariance => &mut self.covariance_comments,
            OpmBlock::UserDefined => &mut self.user_defined_comments,
            OpmBlock::Maneuver => match self.maneuvers.last_mut() {
                Some(maneuver) => &mut maneuver.comments,
                // A maneuver keyword always opens or follows a maneuver.
                None => &mut self.state_comments,
            },
        }
    }

    /// Record a keyword value, already stripped of any unit.
    fn push(&mut self, block: OpmBlock, key: &str, value: &str) -> Result<(), OpmError> {
        match block {
            OpmBlock::UserDefined => {
                let parameter = key.strip_prefix(USER_DEFINED_PREFIX).unwrap_or(key);
                self.push_user_defined(parameter, value)
            }
            OpmBlock::Maneuver => {
                if key == MAN_EPOCH_IGNITION {
                    self.maneuvers.push(ManeuverFields::default());
                }
                let maneuver = self.maneuvers.last_mut().ok_or_else(|| {
                    OpmError::Field(format!(
                        "{key} precedes the MAN_EPOCH_IGNITION that opens a maneuver (CCSDS 502.0-B-3 3.2.4.8)"
                    ))
                })?;
                maneuver.pairs.push((key.to_string(), value.to_string()));
                Ok(())
            }
            _ => {
                self.pairs.push((key.to_string(), value.to_string()));
                Ok(())
            }
        }
    }

    /// Record a user-defined parameter. An exact repeat is read once; a repeat
    /// with a different value is refused by name.
    fn push_user_defined(&mut self, parameter: &str, value: &str) -> Result<(), OpmError> {
        if let Some(existing) = self
            .user_defined
            .iter()
            .find(|existing| existing.parameter == parameter)
        {
            if existing.value == value {
                return Ok(());
            }
            return Err(OpmError::DuplicateField {
                field: format!("{USER_DEFINED_PREFIX}{parameter}"),
                first: existing.value.clone(),
                second: value.to_string(),
            });
        }
        self.user_defined.push(OpmUserDefined {
            parameter: parameter.to_string(),
            value: value.to_string(),
        });
        Ok(())
    }
}

/// Parse a CCSDS OPM in KVN encoding into an [`Opm`].
///
/// Lines end at CR, LF, CR LF or LF CR (502.0-B-3 7.3.7). Blank lines are
/// ignored; any other line must be a comment or an assignment. A maneuver
/// opens at `MAN_EPOCH_IGNITION` and holds the maneuver keywords that follow it
/// (3.2.4.8).
pub fn parse_kvn(text: &str) -> Result<Opm, OpmError> {
    let mut fields = OpmFields::default();
    let mut pending_comments: Vec<String> = Vec::new();
    let mut last_block = OpmBlock::Header;
    for (index, line) in ndm::kvn_lines(text).into_iter().enumerate() {
        match ndm::classify(line) {
            KvnLine::Blank => {}
            KvnLine::Comment(comment) => pending_comments.push(comment.to_string()),
            KvnLine::Assignment { key, value } => {
                let Some(block) = OpmBlock::of_keyword(key) else {
                    // A keyword the tables do not define is refused when it
                    // carries a value; a blank one holds nothing to keep.
                    if value.is_empty() {
                        continue;
                    }
                    return Err(OpmError::UnknownField(key.to_string()));
                };
                let value = kvn_value(key, value)?;
                fields.push(block, key, value)?;
                fields
                    .comments_mut(block)
                    .extend(std::mem::take(&mut pending_comments));
                last_block = block;
            }
            KvnLine::Other(other) => {
                return Err(OpmError::MalformedLine {
                    line: index + 1,
                    text: other.to_string(),
                })
            }
        }
    }
    fields.comments_mut(last_block).extend(pending_comments);
    opm_from_fields(fields)
}

/// A KVN value with any unit table 3-3 gives its keyword checked and removed;
/// a text value is kept verbatim.
fn kvn_value<'a>(key: &str, value: &'a str) -> Result<&'a str, OpmError> {
    match opm_unit(key) {
        Some(allowed) => {
            let (number, unit) = ndm::split_unit(value);
            ndm::check_unit(unit, allowed).map_err(|mismatch| unit_mismatch(key, mismatch))?;
            Ok(number)
        }
        None => Ok(value),
    }
}

fn opm_from_fields(fields: OpmFields) -> Result<Opm, OpmError> {
    let OpmFields {
        pairs,
        header_comments,
        metadata_comments,
        state_comments,
        keplerian_comments,
        spacecraft_comments,
        covariance_comments,
        user_defined_comments,
        keplerian_block,
        spacecraft_block,
        covariance_block,
        maneuvers,
        user_defined,
    } = fields;
    let map = FieldMap::from_pairs(pairs);
    reject_conflict(&map)?;

    let version = map
        .get(OPM_VERSION_KEY)
        .ok_or(OpmError::MissingField(OPM_VERSION_KEY))?
        .to_string();

    let keplerian = if keplerian_block
        || !keplerian_comments.is_empty()
        || KEPLERIAN_KEYS.iter().any(|key| map.get(key).is_some())
    {
        Some(parse_keplerian(&map, keplerian_comments)?)
    } else {
        None
    };
    let spacecraft = if spacecraft_block
        || !spacecraft_comments.is_empty()
        || keyword_occurs(&map, SPACECRAFT_KEYS)
    {
        Some(parse_spacecraft(&map, spacecraft_comments)?)
    } else {
        None
    };
    let covariance = if covariance_block
        || !covariance_comments.is_empty()
        || map.get(COV_REF_FRAME).is_some()
        || COVARIANCE6_KEYS.iter().any(|key| map.get(key).is_some())
    {
        Some(OpmCovariance {
            comments: covariance_comments,
            cov_ref_frame: opt_text(&map, COV_REF_FRAME),
            lower_triangle: read_lower_triangle6(&map).map_err(map_opm_field_error)?,
        })
    } else {
        None
    };

    Ok(Opm {
        ccsds_opm_vers: version,
        comments: header_comments,
        classification: opt_text(&map, "CLASSIFICATION"),
        creation_date: opt_text(&map, "CREATION_DATE"),
        originator: opt_text(&map, "ORIGINATOR"),
        message_id: opt_text(&map, "MESSAGE_ID"),
        metadata: parse_metadata(&map, metadata_comments)?,
        state: parse_state(&map, state_comments)?,
        keplerian,
        spacecraft,
        covariance,
        maneuvers: maneuvers
            .into_iter()
            .map(|maneuver| {
                let map = FieldMap::from_pairs(maneuver.pairs);
                reject_conflict(&map)?;
                parse_maneuver(&map, maneuver.comments)
            })
            .collect::<Result<_, _>>()?,
        user_defined,
        user_defined_comments,
    })
}

/// Encode an [`Opm`] as CCSDS OPM KVN.
///
/// Each block's comments are written at the start of the block (502.0-B-3
/// 7.8.7) and optional keywords only when present. Text that would not read
/// back unchanged is refused with [`OpmError::UnwritableText`]: a line break,
/// whitespace the reader trims, an empty required value, a `USER_DEFINED_*`
/// parameter name containing `=` or given more than once, or user-defined
/// comments with no parameter to precede. A non-finite number is refused with
/// [`OpmError::InvalidField`].
pub fn encode_kvn(opm: &Opm) -> Result<String, OpmError> {
    check_user_defined_names(opm)?;
    let mut out = KvnWriter::default();
    out.required(OPM_VERSION_KEY, &opm.ccsds_opm_vers)?;
    out.comments(&opm.comments)?;
    out.present("CLASSIFICATION", opm.classification.as_deref())?;
    out.optional("CREATION_DATE", opm.creation_date.as_deref())?;
    out.optional("ORIGINATOR", opm.originator.as_deref())?;
    out.present("MESSAGE_ID", opm.message_id.as_deref())?;

    let metadata = &opm.metadata;
    out.comments(&metadata.comments)?;
    out.required("OBJECT_NAME", &metadata.object_name)?;
    out.required("OBJECT_ID", &metadata.object_id)?;
    out.required("CENTER_NAME", &metadata.center_name)?;
    out.required("REF_FRAME", &metadata.ref_frame)?;
    out.present("REF_FRAME_EPOCH", metadata.ref_frame_epoch.as_deref())?;
    out.required("TIME_SYSTEM", &metadata.time_system)?;

    out.comments(&opm.state.comments)?;
    out.required("EPOCH", &opm.state.epoch)?;
    for (key, value) in state_values(&opm.state) {
        out.number(key, value)?;
    }

    if let Some(keplerian) = &opm.keplerian {
        out.comments(&keplerian.comments)?;
        for (key, value) in keplerian_values(keplerian) {
            out.number(key, value)?;
        }
    }
    if let Some(spacecraft) = &opm.spacecraft {
        out.comments(&spacecraft.comments)?;
        let values = spacecraft_values(spacecraft);
        for (key, value) in values {
            if let Some(value) = value {
                out.number(key, value)?;
            }
        }
        if values.iter().all(|(_, value)| value.is_none()) {
            // A blank MASS keeps the block, and any comments, present on read.
            out.line("MASS", "");
        }
    }
    if let Some(covariance) = &opm.covariance {
        out.comments(&covariance.comments)?;
        out.present(COV_REF_FRAME, covariance.cov_ref_frame.as_deref())?;
        for (key, value) in COVARIANCE6_KEYS.into_iter().zip(covariance.lower_triangle) {
            out.number(key, value)?;
        }
    }
    for maneuver in &opm.maneuvers {
        out.comments(&maneuver.comments)?;
        out.required(MAN_EPOCH_IGNITION, &maneuver.epoch_ignition)?;
        out.number("MAN_DURATION", maneuver.duration_s)?;
        out.number("MAN_DELTA_MASS", maneuver.delta_mass_kg)?;
        out.required("MAN_REF_FRAME", &maneuver.ref_frame)?;
        for (key, value) in ["MAN_DV_1", "MAN_DV_2", "MAN_DV_3"]
            .into_iter()
            .zip(maneuver.dv_km_s)
        {
            out.number(key, value)?;
        }
    }
    if opm.user_defined.is_empty() {
        if let Some(comment) = opm.user_defined_comments.first() {
            return Err(unwritable(COMMENT, comment, TextIssue::DetachedComment));
        }
    }
    out.comments(&opm.user_defined_comments)?;
    for parameter in &opm.user_defined {
        let key = format!("{USER_DEFINED_PREFIX}{}", parameter.parameter);
        if let Some(issue) = ndm::text::kvn_parameter_issue(&parameter.parameter) {
            return Err(unwritable(&key, &parameter.parameter, issue));
        }
        out.optional(&key, Some(parameter.value.as_str()))?;
    }
    Ok(out.lines.join("\n"))
}

/// Line assembly for [`encode_kvn`], checking that every value reads back.
#[derive(Default)]
struct KvnWriter {
    lines: Vec<String>,
}

impl KvnWriter {
    fn line(&mut self, key: &str, value: &str) {
        self.lines.push(format!("{key} = {value}"));
    }

    fn required(&mut self, key: &str, value: &str) -> Result<(), OpmError> {
        if let Some(issue) = ndm::text::kvn_required_issue(value) {
            return Err(unwritable(key, value, issue));
        }
        self.line(key, value);
        Ok(())
    }

    /// A keyword written blank when absent, which reads back absent.
    fn optional(&mut self, key: &str, value: Option<&str>) -> Result<(), OpmError> {
        let value = value.unwrap_or_default();
        if let Some(issue) = ndm::text::kvn_value_issue(value) {
            return Err(unwritable(key, value, issue));
        }
        self.line(key, value);
        Ok(())
    }

    /// A keyword written only when present.
    fn present(&mut self, key: &str, value: Option<&str>) -> Result<(), OpmError> {
        match value {
            Some(value) => self.optional(key, Some(value)),
            None => Ok(()),
        }
    }

    fn number(&mut self, key: &'static str, value: f64) -> Result<(), OpmError> {
        check_finite(key, value)?;
        self.line(key, &fmt_num(value));
        Ok(())
    }

    fn comments(&mut self, comments: &[String]) -> Result<(), OpmError> {
        for comment in comments {
            if let Some(issue) = ndm::text::kvn_comment_issue(comment) {
                return Err(unwritable(COMMENT, comment, issue));
            }
            if comment.is_empty() {
                self.lines.push(COMMENT.to_string());
            } else {
                self.lines.push(format!("{COMMENT} {comment}"));
            }
        }
        Ok(())
    }
}

/// Refuse a `USER_DEFINED_*` parameter name given more than once. Both
/// readers keep one of two equal repeats and refuse two that differ, so
/// neither encoding carries such a list back unchanged.
fn check_user_defined_names(opm: &Opm) -> Result<(), OpmError> {
    for (index, parameter) in opm.user_defined.iter().enumerate() {
        if opm.user_defined[..index]
            .iter()
            .any(|earlier| earlier.parameter == parameter.parameter)
        {
            return Err(unwritable(
                &format!("{USER_DEFINED_PREFIX}{}", parameter.parameter),
                &parameter.parameter,
                TextIssue::RepeatedParameter,
            ));
        }
    }
    Ok(())
}

fn unwritable(field: &str, value: &str, issue: TextIssue) -> OpmError {
    OpmError::UnwritableText {
        field: field.to_string(),
        value: value.to_string(),
        issue,
    }
}

fn check_finite(field: &'static str, value: f64) -> Result<(), OpmError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(OpmError::InvalidField {
            field,
            kind: OpmInputErrorKind::NonFinite,
        })
    }
}

fn state_values(state: &OpmState) -> [(&'static str, f64); 6] {
    [
        ("X", state.position_km[0]),
        ("Y", state.position_km[1]),
        ("Z", state.position_km[2]),
        ("X_DOT", state.velocity_km_s[0]),
        ("Y_DOT", state.velocity_km_s[1]),
        ("Z_DOT", state.velocity_km_s[2]),
    ]
}

fn keplerian_values(keplerian: &OpmKeplerian) -> [(&'static str, f64); 7] {
    let anomaly = match keplerian.anomaly {
        OpmAnomaly::True(value) => ("TRUE_ANOMALY", value),
        OpmAnomaly::Mean(value) => ("MEAN_ANOMALY", value),
    };
    [
        ("SEMI_MAJOR_AXIS", keplerian.semi_major_axis_km),
        ("ECCENTRICITY", keplerian.eccentricity),
        ("INCLINATION", keplerian.inclination_deg),
        ("RA_OF_ASC_NODE", keplerian.ra_of_asc_node_deg),
        ("ARG_OF_PERICENTER", keplerian.arg_of_pericenter_deg),
        anomaly,
        ("GM", keplerian.gm_km3_s2),
    ]
}

fn spacecraft_values(spacecraft: &OpmSpacecraft) -> [(&'static str, Option<f64>); 5] {
    [
        ("MASS", spacecraft.mass_kg),
        ("SOLAR_RAD_AREA", spacecraft.solar_rad_area_m2),
        ("SOLAR_RAD_COEFF", spacecraft.solar_rad_coeff),
        ("DRAG_AREA", spacecraft.drag_area_m2),
        ("DRAG_COEFF", spacecraft.drag_coeff),
    ]
}

/// Parse a CCSDS OPM in XML encoding into an [`Opm`].
///
/// The message may be the root element or sit inside an `<ndm>` combined
/// instantiation (505.0-B-3 4.11); a document holding more than one OPM is
/// refused with [`OpmError::MultipleMessages`]. Each value is read from the
/// element that owns it: the header, the segment metadata, or its data block
/// (502.0-B-3 table 8-3). A `units` attribute must match the table 3-3 unit
/// (8.8.11), an element the tables do not define at its position is refused
/// by name when it holds a value, and a block element holding the same
/// keyword twice with different values is refused.
pub fn parse_xml(text: &str) -> Result<Opm, OpmError> {
    let doc = Document::parse(text).map_err(|e| OpmError::Field(format!("malformed XML: {e}")))?;
    let messages = ndm::message_elements(&doc, "opm");
    let opm_node = match messages.as_slice() {
        [] => return Err(OpmError::Field("missing opm element".to_string())),
        [message] => *message,
        _ => {
            return Err(OpmError::MultipleMessages {
                count: messages.len(),
            })
        }
    };

    let mut fields = OpmFields::default();
    if let Some(version) = opm_node
        .attribute("version")
        .map(str::trim)
        .filter(|version| !version.is_empty())
    {
        fields
            .pairs
            .push((OPM_VERSION_KEY.to_string(), version.to_string()));
    }
    let mut segments = Vec::new();
    for child in ndm::element_children(opm_node) {
        match child.tag_name().name() {
            "header" => read_xml_block(child, OpmBlock::Header, &mut fields)?,
            "body" => {
                for body_child in ndm::element_children(child) {
                    match body_child.tag_name().name() {
                        "segment" => segments.push(body_child),
                        other => unknown_element("body", other, body_child)?,
                    }
                }
            }
            other => unknown_element("opm", other, child)?,
        }
    }
    // 502.0-B-3 8.8.6: the OPM body holds a single segment.
    let segment = match segments.as_slice() {
        [segment] => *segment,
        [] => return Err(OpmError::Field("OPM contains no segment".to_string())),
        _ => {
            return Err(OpmError::Field(format!(
                "OPM holds {} segments where CCSDS 502.0-B-3 8.8.6 defines one",
                segments.len()
            )))
        }
    };
    let mut metadata_seen = false;
    let mut data_seen = false;
    for child in ndm::element_children(segment) {
        match child.tag_name().name() {
            "metadata" => {
                metadata_seen = true;
                read_xml_block(child, OpmBlock::Metadata, &mut fields)?;
            }
            "data" => {
                data_seen = true;
                read_xml_data(child, &mut fields)?;
            }
            other => unknown_element("segment", other, child)?,
        }
    }
    if !metadata_seen {
        return Err(OpmError::Field("segment missing metadata".to_string()));
    }
    if !data_seen {
        return Err(OpmError::Field("segment missing data".to_string()));
    }
    opm_from_fields(fields)
}

/// Read the logical blocks of an OPM `<data>` element (502.0-B-3 table 8-3).
fn read_xml_data(data: Node, fields: &mut OpmFields) -> Result<(), OpmError> {
    let mut state_seen = false;
    for child in ndm::element_children(data) {
        match child.tag_name().name() {
            COMMENT => fields.state_comments.push(xml_comment(child)?),
            "stateVector" => {
                if state_seen {
                    return Err(OpmError::Field(
                        "OPM data holds more than one stateVector".to_string(),
                    ));
                }
                state_seen = true;
                read_xml_block(child, OpmBlock::State, fields)?;
            }
            "keplerianElements" => {
                single_block(fields.keplerian_block, "keplerianElements")?;
                fields.keplerian_block = true;
                read_xml_block(child, OpmBlock::Keplerian, fields)?;
            }
            "spacecraftParameters" => {
                single_block(fields.spacecraft_block, "spacecraftParameters")?;
                fields.spacecraft_block = true;
                read_xml_block(child, OpmBlock::Spacecraft, fields)?;
            }
            "covarianceMatrix" => {
                single_block(fields.covariance_block, "covarianceMatrix")?;
                fields.covariance_block = true;
                read_xml_block(child, OpmBlock::Covariance, fields)?;
            }
            "maneuverParameters" => {
                fields.maneuvers.push(ManeuverFields::default());
                read_xml_block(child, OpmBlock::Maneuver, fields)?;
            }
            "userDefinedParameters" => read_xml_user_defined(child, fields)?,
            other => unknown_element("data", other, child)?,
        }
    }
    if !state_seen {
        return Err(OpmError::Field("data missing stateVector".to_string()));
    }
    Ok(())
}

fn single_block(seen: bool, tag: &str) -> Result<(), OpmError> {
    if seen {
        Err(OpmError::Field(format!(
            "OPM data holds more than one {tag}"
        )))
    } else {
        Ok(())
    }
}

/// Refuse an element the tables do not define at its position when it holds a
/// value; an empty one holds nothing to keep.
fn unknown_element(parent: &str, name: &str, node: Node) -> Result<(), OpmError> {
    if ndm::carries_data(node) {
        Err(OpmError::UnknownField(format!("{parent}/{name}")))
    } else {
        Ok(())
    }
}

/// Read the `COMMENT` and keyword elements of one block element.
fn read_xml_block(node: Node, block: OpmBlock, fields: &mut OpmFields) -> Result<(), OpmError> {
    const NO_UNIT: &[&str] = &[];
    for child in ndm::element_children(node) {
        let name = child.tag_name().name();
        if let Some(nested) = ndm::element_children(child).first() {
            return Err(OpmError::UnknownField(format!(
                "{name}/{}",
                nested.tag_name().name()
            )));
        }
        if name == COMMENT {
            fields.comments_mut(block).push(ndm::comment_text(child));
            continue;
        }
        let known = OpmBlock::of_keyword(name) == Some(block) && block != OpmBlock::UserDefined;
        // Within a maneuverParameters element every maneuver keyword belongs to
        // the maneuver it opened, including MAN_EPOCH_IGNITION.
        if !known {
            unknown_element(node.tag_name().name(), name, child)?;
            continue;
        }
        if let Some(unit) = ndm::units_attribute(child) {
            ndm::check_unit(Some(unit.trim()), opm_unit(name).unwrap_or(NO_UNIT))
                .map_err(|mismatch| unit_mismatch(name, mismatch))?;
        }
        let text = ndm::leaf_text(child);
        if block == OpmBlock::Maneuver {
            if let Some(maneuver) = fields.maneuvers.last_mut() {
                maneuver.pairs.push((name.to_string(), text));
            }
        } else {
            fields.pairs.push((name.to_string(), text));
        }
    }
    Ok(())
}

/// The text of a `COMMENT` element; one holding an element is refused by name.
fn xml_comment(node: Node) -> Result<String, OpmError> {
    match ndm::nested_element(node) {
        Some(path) => Err(OpmError::UnknownField(path)),
        None => Ok(ndm::comment_text(node)),
    }
}

/// Read `<userDefinedParameters>` (505.0-B-3 4.10).
fn read_xml_user_defined(node: Node, fields: &mut OpmFields) -> Result<(), OpmError> {
    for child in ndm::element_children(node) {
        match child.tag_name().name() {
            COMMENT => fields.user_defined_comments.push(xml_comment(child)?),
            "USER_DEFINED" => {
                if let Some(path) = ndm::nested_element(child) {
                    return Err(OpmError::UnknownField(path));
                }
                let parameter = child.attribute("parameter").ok_or_else(|| {
                    OpmError::Field(
                        "USER_DEFINED element without a parameter attribute".to_string(),
                    )
                })?;
                if let Some(unit) = ndm::units_attribute(child) {
                    // 505.0-B-3 4.10.1.6 carries user-defined units in the value.
                    return Err(OpmError::UnitMismatch {
                        field: format!("{USER_DEFINED_PREFIX}{parameter}"),
                        unit: unit.to_string(),
                        expected: None,
                    });
                }
                fields.push_user_defined(parameter, &ndm::leaf_text(child))?;
            }
            other => unknown_element("userDefinedParameters", other, child)?,
        }
    }
    Ok(())
}

/// Encode an [`Opm`] as CCSDS OPM XML.
///
/// Text the reader would not return unchanged is refused with
/// [`OpmError::UnwritableText`]: a value with surrounding whitespace (the
/// reader trims element text), an empty required value, a comment with
/// trailing whitespace, a `USER_DEFINED_*` parameter name given more than
/// once, and any character XML 1.0 cannot carry. A non-finite number is
/// refused with [`OpmError::InvalidField`].
pub fn encode_xml(opm: &Opm) -> Result<String, OpmError> {
    check_user_defined_names(opm)?;
    let mut out = XmlWriter::default();
    out.raw(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    if let Some(issue) = ndm::text::xml_required_issue(&opm.ccsds_opm_vers) {
        return Err(unwritable(OPM_VERSION_KEY, &opm.ccsds_opm_vers, issue));
    }
    out.raw(&format!(
        r#"<opm id="CCSDS_OPM_VERS" version="{}">"#,
        ndm::text::escape_attribute(&opm.ccsds_opm_vers)
    ));
    out.raw("  <header>");
    out.comments(4, &opm.comments)?;
    out.present(4, "CLASSIFICATION", opm.classification.as_deref())?;
    out.optional(4, "CREATION_DATE", opm.creation_date.as_deref())?;
    out.optional(4, "ORIGINATOR", opm.originator.as_deref())?;
    out.present(4, "MESSAGE_ID", opm.message_id.as_deref())?;
    out.raw("  </header>");
    out.raw("  <body>");
    out.raw("    <segment>");
    out.raw("      <metadata>");
    let metadata = &opm.metadata;
    out.comments(8, &metadata.comments)?;
    out.required(8, "OBJECT_NAME", &metadata.object_name)?;
    out.required(8, "OBJECT_ID", &metadata.object_id)?;
    out.required(8, "CENTER_NAME", &metadata.center_name)?;
    out.required(8, "REF_FRAME", &metadata.ref_frame)?;
    out.present(8, "REF_FRAME_EPOCH", metadata.ref_frame_epoch.as_deref())?;
    out.required(8, "TIME_SYSTEM", &metadata.time_system)?;
    out.raw("      </metadata>");
    out.raw("      <data>");

    out.raw("        <stateVector>");
    out.comments(10, &opm.state.comments)?;
    out.required(10, "EPOCH", &opm.state.epoch)?;
    for (key, value) in state_values(&opm.state) {
        out.number(10, key, value)?;
    }
    out.raw("        </stateVector>");

    if let Some(keplerian) = &opm.keplerian {
        out.raw("        <keplerianElements>");
        out.comments(10, &keplerian.comments)?;
        for (key, value) in keplerian_values(keplerian) {
            out.number(10, key, value)?;
        }
        out.raw("        </keplerianElements>");
    }
    if let Some(spacecraft) = &opm.spacecraft {
        out.raw("        <spacecraftParameters>");
        out.comments(10, &spacecraft.comments)?;
        for (key, value) in spacecraft_values(spacecraft) {
            if let Some(value) = value {
                out.number(10, key, value)?;
            }
        }
        out.raw("        </spacecraftParameters>");
    }
    if let Some(covariance) = &opm.covariance {
        out.raw("        <covarianceMatrix>");
        out.comments(10, &covariance.comments)?;
        out.present(10, COV_REF_FRAME, covariance.cov_ref_frame.as_deref())?;
        for (key, value) in COVARIANCE6_KEYS.into_iter().zip(covariance.lower_triangle) {
            out.number(10, key, value)?;
        }
        out.raw("        </covarianceMatrix>");
    }
    for maneuver in &opm.maneuvers {
        out.raw("        <maneuverParameters>");
        out.comments(10, &maneuver.comments)?;
        out.required(10, MAN_EPOCH_IGNITION, &maneuver.epoch_ignition)?;
        out.number(10, "MAN_DURATION", maneuver.duration_s)?;
        out.number(10, "MAN_DELTA_MASS", maneuver.delta_mass_kg)?;
        out.required(10, "MAN_REF_FRAME", &maneuver.ref_frame)?;
        for (key, value) in ["MAN_DV_1", "MAN_DV_2", "MAN_DV_3"]
            .into_iter()
            .zip(maneuver.dv_km_s)
        {
            out.number(10, key, value)?;
        }
        out.raw("        </maneuverParameters>");
    }
    if !opm.user_defined.is_empty() || !opm.user_defined_comments.is_empty() {
        out.raw("        <userDefinedParameters>");
        out.comments(10, &opm.user_defined_comments)?;
        for parameter in &opm.user_defined {
            let field = format!("{USER_DEFINED_PREFIX}{}", parameter.parameter);
            if let Some(issue) = ndm::text::xml_attribute_issue(&parameter.parameter) {
                return Err(unwritable(&field, &parameter.parameter, issue));
            }
            if let Some(issue) = ndm::text::xml_value_issue(&parameter.value) {
                return Err(unwritable(&field, &parameter.value, issue));
            }
            out.raw(&format!(
                r#"          <USER_DEFINED parameter="{}">{}</USER_DEFINED>"#,
                ndm::text::escape_attribute(&parameter.parameter),
                xml::escape(&parameter.value)
            ));
        }
        out.raw("        </userDefinedParameters>");
    }

    out.raw("      </data>");
    out.raw("    </segment>");
    out.raw("  </body>");
    out.raw("</opm>");
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

    fn required(&mut self, indent: usize, name: &str, value: &str) -> Result<(), OpmError> {
        if let Some(issue) = ndm::text::xml_required_issue(value) {
            return Err(unwritable(name, value, issue));
        }
        self.element(indent, name, &xml::escape(value));
        Ok(())
    }

    /// An element written empty when absent, which reads back absent.
    fn optional(&mut self, indent: usize, name: &str, value: Option<&str>) -> Result<(), OpmError> {
        let value = value.unwrap_or_default();
        if let Some(issue) = ndm::text::xml_value_issue(value) {
            return Err(unwritable(name, value, issue));
        }
        self.element(indent, name, &xml::escape(value));
        Ok(())
    }

    fn present(&mut self, indent: usize, name: &str, value: Option<&str>) -> Result<(), OpmError> {
        match value {
            Some(value) => self.optional(indent, name, Some(value)),
            None => Ok(()),
        }
    }

    fn number(&mut self, indent: usize, name: &'static str, value: f64) -> Result<(), OpmError> {
        check_finite(name, value)?;
        self.element(indent, name, &fmt_num(value));
        Ok(())
    }

    fn comments(&mut self, indent: usize, comments: &[String]) -> Result<(), OpmError> {
        for comment in comments {
            if let Some(issue) = ndm::text::xml_comment_issue(comment) {
                return Err(unwritable(COMMENT, comment, issue));
            }
            self.element(indent, COMMENT, &xml::escape(comment));
        }
        Ok(())
    }
}

fn parse_metadata(map: &FieldMap, comments: Vec<String>) -> Result<OpmMetadata, OpmError> {
    Ok(OpmMetadata {
        comments,
        object_name: req_text(map, "OBJECT_NAME")?,
        object_id: req_text(map, "OBJECT_ID")?,
        center_name: req_text(map, "CENTER_NAME")?,
        ref_frame: req_text(map, "REF_FRAME")?,
        ref_frame_epoch: opt_text(map, "REF_FRAME_EPOCH"),
        time_system: req_text(map, "TIME_SYSTEM")?,
    })
}

fn parse_state(map: &FieldMap, comments: Vec<String>) -> Result<OpmState, OpmError> {
    Ok(OpmState {
        comments,
        epoch: req_text(map, "EPOCH")?,
        position_km: [req_num(map, "X")?, req_num(map, "Y")?, req_num(map, "Z")?],
        velocity_km_s: [
            req_num(map, "X_DOT")?,
            req_num(map, "Y_DOT")?,
            req_num(map, "Z_DOT")?,
        ],
    })
}

/// Whether any of `keys` occurs, with any value including a blank one. A blank
/// optional keyword still opens its block.
fn keyword_occurs(map: &FieldMap, keys: &[&str]) -> bool {
    map.pairs()
        .iter()
        .any(|(key, _)| keys.contains(&key.as_str()))
}

fn parse_keplerian(map: &FieldMap, comments: Vec<String>) -> Result<OpmKeplerian, OpmError> {
    let true_anomaly = opt_num(map, "TRUE_ANOMALY")?;
    let mean_anomaly = opt_num(map, "MEAN_ANOMALY")?;
    let anomaly = match (true_anomaly, mean_anomaly) {
        (Some(value), None) => OpmAnomaly::True(value),
        (None, Some(value)) => OpmAnomaly::Mean(value),
        (None, None) => {
            return Err(OpmError::Field(
                "keplerianElements requires TRUE_ANOMALY or MEAN_ANOMALY".to_string(),
            ))
        }
        (Some(_), Some(_)) => {
            return Err(OpmError::Field(
                "keplerianElements cannot contain both TRUE_ANOMALY and MEAN_ANOMALY".to_string(),
            ))
        }
    };

    Ok(OpmKeplerian {
        comments,
        semi_major_axis_km: req_num(map, "SEMI_MAJOR_AXIS")?,
        eccentricity: req_num(map, "ECCENTRICITY")?,
        inclination_deg: req_num(map, "INCLINATION")?,
        ra_of_asc_node_deg: req_num(map, "RA_OF_ASC_NODE")?,
        arg_of_pericenter_deg: req_num(map, "ARG_OF_PERICENTER")?,
        anomaly,
        gm_km3_s2: req_num(map, "GM")?,
    })
}

fn parse_spacecraft(map: &FieldMap, comments: Vec<String>) -> Result<OpmSpacecraft, OpmError> {
    Ok(OpmSpacecraft {
        comments,
        mass_kg: opt_num(map, "MASS")?,
        solar_rad_area_m2: opt_num(map, "SOLAR_RAD_AREA")?,
        solar_rad_coeff: opt_num(map, "SOLAR_RAD_COEFF")?,
        drag_area_m2: opt_num(map, "DRAG_AREA")?,
        drag_coeff: opt_num(map, "DRAG_COEFF")?,
    })
}

fn parse_maneuver(map: &FieldMap, comments: Vec<String>) -> Result<OpmManeuver, OpmError> {
    Ok(OpmManeuver {
        comments,
        epoch_ignition: req_text(map, MAN_EPOCH_IGNITION)?,
        duration_s: req_num(map, "MAN_DURATION")?,
        delta_mass_kg: req_num(map, "MAN_DELTA_MASS")?,
        ref_frame: req_text(map, "MAN_REF_FRAME")?,
        dv_km_s: [
            req_num(map, "MAN_DV_1")?,
            req_num(map, "MAN_DV_2")?,
            req_num(map, "MAN_DV_3")?,
        ],
    })
}

/// The unit spellings CCSDS 502.0-B-3 table 3-3 gives a numeric keyword: an
/// empty list for a dimensionless one, `None` for a text keyword.
fn opm_unit(key: &str) -> Option<&'static [&'static str]> {
    const DIMENSIONLESS: &[&str] = &[];
    const KM: &[&str] = &["km"];
    const KM_PER_S: &[&str] = &["km/s"];
    const DEG: &[&str] = &["deg"];
    const GM: &[&str] = &["km**3/s**2"];
    const KG: &[&str] = &["kg"];
    const M2: &[&str] = &["m**2"];
    const S: &[&str] = &["s"];
    match key {
        "X" | "Y" | "Z" | "SEMI_MAJOR_AXIS" => Some(KM),
        "X_DOT" | "Y_DOT" | "Z_DOT" | "MAN_DV_1" | "MAN_DV_2" | "MAN_DV_3" => Some(KM_PER_S),
        "ECCENTRICITY" | "SOLAR_RAD_COEFF" | "DRAG_COEFF" => Some(DIMENSIONLESS),
        "INCLINATION" | "RA_OF_ASC_NODE" | "ARG_OF_PERICENTER" | "TRUE_ANOMALY"
        | "MEAN_ANOMALY" => Some(DEG),
        "GM" => Some(GM),
        "MASS" | "MAN_DELTA_MASS" => Some(KG),
        "SOLAR_RAD_AREA" | "DRAG_AREA" => Some(M2),
        "MAN_DURATION" => Some(S),
        other => covariance6_unit(other),
    }
}

fn unit_mismatch(key: &str, mismatch: UnitMismatch) -> OpmError {
    OpmError::UnitMismatch {
        field: key.to_string(),
        unit: mismatch.unit,
        expected: mismatch.expected,
    }
}

fn reject_conflict(map: &FieldMap) -> Result<(), OpmError> {
    match map.first_conflict(|key| key != COMMENT) {
        Some(conflict) => Err(OpmError::DuplicateField {
            field: conflict.key,
            first: conflict.first,
            second: conflict.second,
        }),
        None => Ok(()),
    }
}

fn req_text(map: &FieldMap, field: &'static str) -> Result<String, OpmError> {
    map.get(field)
        .map(str::to_string)
        .ok_or(OpmError::MissingField(field))
}

fn opt_text(map: &FieldMap, field: &'static str) -> Option<String> {
    map.get(field).map(str::to_string)
}

fn req_num(map: &FieldMap, field: &'static str) -> Result<f64, OpmError> {
    let value = map.get(field).ok_or(OpmError::MissingField(field))?;
    parse_num(value, field)
}

fn opt_num(map: &FieldMap, field: &'static str) -> Result<Option<f64>, OpmError> {
    map.get(field)
        .map(|value| parse_num(value, field))
        .transpose()
}

fn parse_num(value: &str, field: &'static str) -> Result<f64, OpmError> {
    validate::strict_f64(value, field).map_err(map_opm_field_error)
}

fn map_opm_field_error(error: validate::FieldError) -> OpmError {
    OpmError::InvalidField {
        field: error.field(),
        kind: OpmInputErrorKind::from(&error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_kvn() -> String {
        "\
CCSDS_OPM_VERS = 2.0
CREATION_DATE = 2026-06-28T00:00:00
ORIGINATOR = SIDEREON
OBJECT_NAME = OSPREY
OBJECT_ID = 2026-001A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
EPOCH = 2026-06-28T00:00:00
X = 7000
Y = 0
Z = 0
X_DOT = 0
Y_DOT = 7.5
Z_DOT = 1
"
        .to_string()
    }

    #[test]
    fn malformed_xml_is_an_error() {
        assert!(parse_xml("<opm></opm><opm></opm>").is_err());
    }

    #[test]
    fn missing_required_field_is_an_error() {
        let kvn = minimal_kvn().replace("OBJECT_ID = 2026-001A\n", "");
        assert_eq!(parse_kvn(&kvn), Err(OpmError::MissingField("OBJECT_ID")));
    }

    #[test]
    fn parses_two_maneuvers_from_kvn() {
        let kvn = format!(
            "{}{}",
            minimal_kvn(),
            "\
MAN_EPOCH_IGNITION = 2026-06-28T00:10:00
MAN_DURATION = 10
MAN_DELTA_MASS = -0.5
MAN_REF_FRAME = TNW
MAN_DV_1 = 0.001
MAN_DV_2 = 0
MAN_DV_3 = 0
MAN_EPOCH_IGNITION = 2026-06-28T00:20:00
MAN_DURATION = 20
MAN_DELTA_MASS = -0.7
MAN_REF_FRAME = TNW
MAN_DV_1 = 0
MAN_DV_2 = 0.002
MAN_DV_3 = 0
"
        );
        let opm = parse_kvn(&kvn).unwrap();
        assert_eq!(opm.maneuvers.len(), 2);
        assert_eq!(opm.maneuvers[0].ref_frame, "TNW");
        assert_eq!(opm.maneuvers[1].dv_km_s, [0.0, 0.002, 0.0]);
    }

    /// CCSDS 502.0-B-3 annex G figure G-2: units on the numeric values of every
    /// block, comments, Keplerian elements and two maneuvers.
    const STANDARD_UNITS_KVN: &str = "\
CCSDS_OPM_VERS     =   3.0

COMMENT   Generated by GSOC, R. Kiehling
COMMENT   Current intermediate orbit IO2 and maneuver planning data

CREATION_DATE      =   2021-06-03T05:33:00.000
ORIGINATOR         =   GSOC

OBJECT_NAME        =   EUTELSAT W4
OBJECT_ID          =   2021-028A
CENTER_NAME        =   EARTH
REF_FRAME          =   TOD
TIME_SYSTEM        =   UTC

COMMENT   State Vector
EPOCH              = 2021-06-03T00:00:00.000
X                  =   6655.9942        [km]
Y                  = -40218.5751        [km]
Z                  =    -82.9177        [km]
X_DOT              =      3.11548208    [km/s]
Y_DOT              =      0.47042605    [km/s]
Z_DOT              =     -0.00101495    [km/s]

COMMENT Keplerian elements
SEMI_MAJOR_AXIS   =  41399.5123            [km]
ECCENTRICITY      =      0.020842611
INCLINATION       =      0.117746          [deg]
RA_OF_ASC_NODE    =     17.604721          [deg]
ARG_OF_PERICENTER =    218.242943          [deg]
TRUE_ANOMALY      =     41.922339          [deg]
GM                = 398600.4415            [km**3/s**2]

COMMENT Spacecraft parameters
MASS             =    1913.000             [kg]
SOLAR_RAD_AREA   =      10.000             [m**2]
SOLAR_RAD_COEFF  =       1.300
DRAG_AREA        =      10.000             [m**2]
DRAG_COEFF       =       2.300

COMMENT   2 planned maneuvers

COMMENT First maneuver: AMF-3
COMMENT Non-impulsive, thrust direction fixed in inertial frame
MAN_EPOCH_IGNITION =     2021-06-03T09:00:34.1
MAN_DURATION      =    132.60          [s]
MAN_DELTA_MASS    =    -18.418         [kg]
MAN_REF_FRAME     =      EME2000
MAN_DV_1          =     -0.02325700    [km/s]
MAN_DV_2          =      0.01683160    [km/s]
MAN_DV_3          =     -0.00893444    [km/s]

COMMENT Second maneuver: first station acquisition maneuver
COMMENT impulsive, thrust direction fixed in RTN frame
MAN_EPOCH_IGNITION =     2021-06-05T18:59:21.0
MAN_DURATION      =      0.00          [s]
MAN_DELTA_MASS    =     -1.469         [kg]
MAN_REF_FRAME     =      RTN
MAN_DV_1          =      0.00101500    [km/s]
MAN_DV_2          =     -0.00187300    [km/s]
MAN_DV_3          =      0.00000000    [km/s]
";

    #[test]
    fn reads_the_standard_units_example() {
        let opm = parse_kvn(STANDARD_UNITS_KVN).expect("502.0-B-3 figure G-2 parses");
        assert_eq!(opm.metadata.object_name, "EUTELSAT W4");
        assert_eq!(opm.state.position_km, [6655.9942, -40218.5751, -82.9177]);
        let keplerian = opm.keplerian.as_ref().expect("Keplerian block");
        assert_eq!(keplerian.gm_km3_s2, 398600.4415);
        assert_eq!(keplerian.anomaly, OpmAnomaly::True(41.922339));
        assert_eq!(
            opm.spacecraft.as_ref().and_then(|s| s.mass_kg),
            Some(1913.0)
        );
        assert_eq!(opm.maneuvers.len(), 2);
        assert_eq!(opm.maneuvers[0].duration_s, 132.60);
        assert_eq!(opm.maneuvers[1].delta_mass_kg, -1.469);
        assert_eq!(parse_kvn(&encode_kvn(&opm).unwrap()).unwrap(), opm);
    }

    #[test]
    fn a_contradicting_unit_is_refused_by_name() {
        let kvn = minimal_kvn().replace("X = 7000\n", "X = 7000 [m]\n");
        assert_eq!(
            parse_kvn(&kvn),
            Err(OpmError::UnitMismatch {
                field: "X".to_string(),
                unit: "m".to_string(),
                expected: Some("km"),
            })
        );
        let kvn = format!(
            "{}SEMI_MAJOR_AXIS = 7000\nECCENTRICITY = 0.001 [n/a]\nINCLINATION = 51\n\
RA_OF_ASC_NODE = 1\nARG_OF_PERICENTER = 2\nMEAN_ANOMALY = 3\nGM = 398600.4415\n",
            minimal_kvn()
        );
        assert_eq!(
            parse_kvn(&kvn),
            Err(OpmError::UnitMismatch {
                field: "ECCENTRICITY".to_string(),
                unit: "n/a".to_string(),
                expected: None,
            })
        );
    }

    #[test]
    fn text_values_ending_in_brackets_are_kept_verbatim() {
        let kvn =
            minimal_kvn().replace("OBJECT_NAME = OSPREY\n", "OBJECT_NAME = OSPREY [BLOCK 2]\n");
        let opm = parse_kvn(&kvn).unwrap();
        assert_eq!(opm.metadata.object_name, "OSPREY [BLOCK 2]");
        assert_eq!(parse_kvn(&encode_kvn(&opm).unwrap()).unwrap(), opm);
        assert_eq!(parse_xml(&encode_xml(&opm).unwrap()).unwrap(), opm);
    }

    #[test]
    fn a_keyword_repeated_with_a_different_value_is_refused() {
        let kvn = format!("{}X = 7001\n", minimal_kvn());
        assert_eq!(
            parse_kvn(&kvn),
            Err(OpmError::DuplicateField {
                field: "X".to_string(),
                first: "7000".to_string(),
                second: "7001".to_string(),
            })
        );
        let same = format!("{}X = 7000\n", minimal_kvn());
        assert_eq!(
            parse_kvn(&same).unwrap(),
            parse_kvn(&minimal_kvn()).unwrap()
        );

        let maneuver = "\
MAN_EPOCH_IGNITION = 2026-06-28T00:10:00
MAN_DURATION = 10
MAN_DURATION = 11
MAN_DELTA_MASS = -0.5
MAN_REF_FRAME = TNW
MAN_DV_1 = 0.001
MAN_DV_2 = 0
MAN_DV_3 = 0
";
        assert_eq!(
            parse_kvn(&format!("{}{maneuver}", minimal_kvn())),
            Err(OpmError::DuplicateField {
                field: "MAN_DURATION".to_string(),
                first: "10".to_string(),
                second: "11".to_string(),
            })
        );
    }

    #[test]
    fn xml_units_duplicates_and_message_count_are_checked() {
        let opm = parse_kvn(&minimal_kvn()).unwrap();
        let xml = encode_xml(&opm).unwrap();

        let with_unit = xml.replace("<X>7000</X>", "<X units=\"km\">7000</X>");
        assert_eq!(parse_xml(&with_unit).unwrap(), opm);
        let wrong_unit = xml.replace("<X>7000</X>", "<X units=\"m\">7000</X>");
        assert_eq!(
            parse_xml(&wrong_unit),
            Err(OpmError::UnitMismatch {
                field: "X".to_string(),
                unit: "m".to_string(),
                expected: Some("km"),
            })
        );
        let repeated = xml.replace("<X>7000</X>", "<X>7000</X><X>7001</X>");
        assert_eq!(
            parse_xml(&repeated),
            Err(OpmError::DuplicateField {
                field: "X".to_string(),
                first: "7000".to_string(),
                second: "7001".to_string(),
            })
        );

        let message = xml.trim_start_matches(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        let combined = format!("<ndm>{message}{message}</ndm>");
        assert_eq!(
            parse_xml(&combined),
            Err(OpmError::MultipleMessages { count: 2 })
        );
        let single = format!("<ndm>{message}</ndm>");
        assert_eq!(parse_xml(&single).unwrap(), opm);
    }

    #[test]
    fn standard_example_comments_are_kept_with_their_blocks() {
        let opm = parse_kvn(STANDARD_UNITS_KVN).unwrap();
        assert_eq!(
            opm.comments,
            vec![
                "  Generated by GSOC, R. Kiehling",
                "  Current intermediate orbit IO2 and maneuver planning data",
            ]
        );
        assert_eq!(opm.state.comments, vec!["  State Vector"]);
        assert_eq!(
            opm.keplerian.as_ref().unwrap().comments,
            vec!["Keplerian elements"]
        );
        assert_eq!(
            opm.spacecraft.as_ref().unwrap().comments,
            vec!["Spacecraft parameters"]
        );
        assert_eq!(
            opm.maneuvers[0].comments,
            vec![
                "  2 planned maneuvers",
                "First maneuver: AMF-3",
                "Non-impulsive, thrust direction fixed in inertial frame",
            ]
        );
        assert_eq!(opm.maneuvers[1].comments.len(), 2);
        assert_eq!(parse_kvn(&encode_kvn(&opm).unwrap()).unwrap(), opm);
        assert_eq!(parse_xml(&encode_xml(&opm).unwrap()).unwrap(), opm);
    }

    #[test]
    fn header_items_ref_frame_epoch_and_user_defined_parameters_are_retained() {
        let kvn = minimal_kvn()
            .replace(
                "CREATION_DATE = 2026-06-28T00:00:00\n",
                "CLASSIFICATION = SBU\nCREATION_DATE = 2026-06-28T00:00:00\n",
            )
            .replace(
                "ORIGINATOR = SIDEREON\n",
                "ORIGINATOR = SIDEREON\nMESSAGE_ID = OPM 201113719185\n",
            )
            .replace(
                "REF_FRAME = EME2000\n",
                "REF_FRAME = EME2000\nREF_FRAME_EPOCH = 2000-01-01T12:00:00\n",
            )
            + "COMMENT user-defined block\nUSER_DEFINED_EARTH_MODEL = WGS-84\nUSER_DEFINED_C3 = 29.376 [km**2/s**2]\n";
        let opm = parse_kvn(&kvn).unwrap();
        assert_eq!(opm.classification.as_deref(), Some("SBU"));
        assert_eq!(opm.message_id.as_deref(), Some("OPM 201113719185"));
        assert_eq!(
            opm.metadata.ref_frame_epoch.as_deref(),
            Some("2000-01-01T12:00:00")
        );
        assert_eq!(opm.user_defined_comments, vec!["user-defined block"]);
        assert_eq!(
            opm.user_defined,
            vec![
                OpmUserDefined {
                    parameter: "EARTH_MODEL".to_string(),
                    value: "WGS-84".to_string(),
                },
                OpmUserDefined {
                    parameter: "C3".to_string(),
                    value: "29.376 [km**2/s**2]".to_string(),
                },
            ]
        );
        assert_eq!(parse_kvn(&encode_kvn(&opm).unwrap()).unwrap(), opm);
        assert_eq!(parse_xml(&encode_xml(&opm).unwrap()).unwrap(), opm);
    }

    #[test]
    fn unknown_keywords_malformed_lines_and_orphan_maneuver_keys_are_refused() {
        assert_eq!(
            parse_kvn(&format!("{}FOO = 1\n", minimal_kvn())),
            Err(OpmError::UnknownField("FOO".to_string()))
        );
        // A keyword that carries no value holds nothing to keep.
        assert_eq!(
            parse_kvn(&format!("{}FOO = \n", minimal_kvn())).unwrap(),
            parse_kvn(&minimal_kvn()).unwrap()
        );
        let with_garbage = format!("{}not an assignment\n", minimal_kvn());
        assert_eq!(
            parse_kvn(&with_garbage),
            Err(OpmError::MalformedLine {
                line: with_garbage.lines().count(),
                text: "not an assignment".to_string(),
            })
        );
        assert!(matches!(
            parse_kvn(&format!("{}MAN_DURATION = 10\n", minimal_kvn())),
            Err(OpmError::Field(message)) if message.contains("MAN_DURATION precedes")
        ));
        let opm = parse_kvn(&minimal_kvn()).unwrap();
        let xml = encode_xml(&opm)
            .unwrap()
            .replace("<Z_DOT>1</Z_DOT>", "<Z_DOT>1</Z_DOT><W_DOT>2</W_DOT>");
        assert_eq!(
            parse_xml(&xml),
            Err(OpmError::UnknownField("stateVector/W_DOT".to_string()))
        );
        // An element inside a COMMENT is refused rather than dropped with
        // only the comment's text kept.
        let xml = encode_xml(&opm).unwrap().replacen(
            "<stateVector>",
            "<COMMENT>note <b>1</b></COMMENT><stateVector>",
            1,
        );
        assert_eq!(
            parse_xml(&xml),
            Err(OpmError::UnknownField("COMMENT/b".to_string()))
        );
    }

    #[test]
    fn writers_refuse_text_that_would_read_back_differently() {
        let base = parse_kvn(&minimal_kvn()).unwrap();

        let mut opm = base.clone();
        opm.metadata.object_name = "OSPREY\nX = 1".to_string();
        assert_eq!(
            encode_kvn(&opm),
            Err(OpmError::UnwritableText {
                field: "OBJECT_NAME".to_string(),
                value: "OSPREY\nX = 1".to_string(),
                issue: TextIssue::LineBreak,
            })
        );

        let mut opm = base.clone();
        opm.metadata.object_id = String::new();
        assert!(matches!(
            encode_kvn(&opm),
            Err(OpmError::UnwritableText {
                issue: TextIssue::Empty,
                ..
            })
        ));
        assert!(matches!(
            encode_xml(&opm),
            Err(OpmError::UnwritableText {
                issue: TextIssue::Empty,
                ..
            })
        ));

        let mut opm = base.clone();
        opm.metadata.center_name = "EARTH\u{1}".to_string();
        assert!(matches!(
            encode_xml(&opm),
            Err(OpmError::UnwritableText {
                issue: TextIssue::XmlIllegalCharacter,
                ..
            })
        ));

        let mut opm = base.clone();
        opm.state.velocity_km_s[1] = f64::NAN;
        assert_eq!(
            encode_kvn(&opm),
            Err(OpmError::InvalidField {
                field: "Y_DOT",
                kind: OpmInputErrorKind::NonFinite,
            })
        );

        // Both readers keep one of two equal repeats and refuse two that
        // differ, so a parameter named twice reads back as something else.
        for second_value in ["WGS-84", "EGM-96"] {
            let mut opm = base.clone();
            opm.user_defined = vec![
                OpmUserDefined {
                    parameter: "EARTH_MODEL".to_string(),
                    value: "WGS-84".to_string(),
                },
                OpmUserDefined {
                    parameter: "EARTH_MODEL".to_string(),
                    value: second_value.to_string(),
                },
            ];
            let expected: Result<String, OpmError> = Err(OpmError::UnwritableText {
                field: "USER_DEFINED_EARTH_MODEL".to_string(),
                value: "EARTH_MODEL".to_string(),
                issue: TextIssue::RepeatedParameter,
            });
            assert_eq!(encode_kvn(&opm), expected, "KVN, {second_value}");
            assert_eq!(encode_xml(&opm), expected, "XML, {second_value}");
        }
    }

    #[cfg(all(test, sidereon_repo_tests))]
    mod fixtures {
        use super::*;

        const OSPREY_KVN: &str = include_str!("../../tests/fixtures/opm/osprey.kvn");
        const OSPREY_XML: &str = include_str!("../../tests/fixtures/opm/osprey.xml");

        #[test]
        fn parses_osprey_kvn_fixture() {
            let opm = parse_kvn(OSPREY_KVN).unwrap();
            assert_eq!(opm.ccsds_opm_vers, "2.0");
            assert_eq!(opm.metadata.object_name, "OSPREY-1");
            assert_eq!(opm.state.position_km[0], 6878.137);
            assert_eq!(opm.maneuvers.len(), 2);
            assert!(matches!(
                opm.keplerian.as_ref().unwrap().anomaly,
                OpmAnomaly::True(42.0)
            ));
        }

        #[test]
        fn parses_osprey_xml_fixture() {
            let opm = parse_xml(OSPREY_XML).unwrap();
            assert_eq!(opm.metadata.object_id, "2026-045A");
            assert_eq!(opm.spacecraft.as_ref().unwrap().mass_kg, Some(425.0));
            assert_eq!(
                opm.covariance.as_ref().unwrap().cov_ref_frame.as_deref(),
                Some("EME2000")
            );
        }

        #[test]
        fn fixture_kvn_round_trips() {
            let opm = parse_kvn(OSPREY_KVN).unwrap();
            assert_eq!(parse_kvn(&encode_kvn(&opm).unwrap()).unwrap(), opm);
        }

        #[test]
        fn fixture_xml_round_trips() {
            let opm = parse_xml(OSPREY_XML).unwrap();
            assert_eq!(parse_xml(&encode_xml(&opm).unwrap()).unwrap(), opm);
        }
    }
}
