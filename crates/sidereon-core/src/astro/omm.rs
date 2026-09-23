#![warn(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

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
//! ## Keywords, blocks and comments
//!
//! The KVN and XML readers accept the keywords of CCSDS 502.0-B-3 tables 4-1,
//! 4-2 and 4-3 and retain every one: the optional header `CLASSIFICATION` and
//! `MESSAGE_ID`, `REF_FRAME_EPOCH`, `SEMI_MAJOR_AXIS` as well as
//! `MEAN_MOTION`, `GM`, the spacecraft-parameters block, the TLE related
//! parameters including the SGP4-XP `BTERM` and `AGOM`, the covariance block,
//! `USER_DEFINED_*` parameters, and the comments of every block. An OMM for
//! any mean-element theory the tables allow therefore reads and writes back;
//! the table 4-3 items that depend on the theory are held as options. A
//! keyword outside those tables that carries a value is refused by name
//! (4.2.2.2, 4.2.3.2, 4.2.4.2) rather than dropped. A keyword repeated with a
//! different value is refused by name; an exact repeat carries nothing new and
//! is read once. In KVN a comment belongs to the block of the keyword that
//! follows it (7.8.8), and comments after the last keyword belong to that
//! keyword's block.
//!
//! GP JSON and CSV are flat product encodings rather than CCSDS encodings: they
//! carry no block structure and one comment, the `COMMENT` member or column
//! Space-Track writes, read as the header comment. Their readers ignore member
//! names that are not OMM keywords, because Space-Track GP records carry
//! catalog extras such as `OBJECT_TYPE`, `RCS_SIZE`, `DECAY_DATE` and
//! `TLE_LINE1` alongside the OMM keywords. Their writers refuse a record
//! holding any other comment; [`encode_json_discarding_comments`],
//! [`encode_json_array_discarding_comments`] and
//! [`encode_csv_discarding_comments`] write it without those comments.
//!
//! ## SGP4 bridge
//!
//! [`Omm::to_element_set`] refuses an OMM whose explicitly stated
//! `MEAN_ELEMENT_THEORY`, `CENTER_NAME`, `REF_FRAME` or `TIME_SYSTEM` is not
//! the SGP4/Earth/TEME/UTC convention of 502.0-B-3 4.2.4.6. A field the message
//! does not state, as CelesTrak GP JSON and CSV omit them, does not block
//! propagation. Parsing an OMM with any other center, frame, time system or
//! theory is unaffected; only the bridge refuses it. The bridge requires the
//! elements SGP4 propagates with, `MEAN_MOTION` and `BSTAR`, and carries
//! `NORAD_CAT_ID` and the mean-motion derivatives when stated.
//!
//! ## TLE-derived field quantization
//!
//! An OMM carries a full UTC calendar `EPOCH`, which is converted directly to
//! SGP4's split Julian date. B\* and the second mean-motion derivative are
//! still TLE-derived GP parameters:
//!
//! - **B\* and the second mean-motion derivative.** A TLE stores these in its
//!   "assumed decimal" field (five significant mantissa digits and a
//!   single-digit power-of-ten exponent), and that quantized value is what SGP4
//!   actually receives. OMM prints the same quantities as plain decimals, so
//!   the bridge re-quantizes them with the rounding the TLE writer in
//!   [`crate::astro::tle`] uses.
//!
//! The quantized values are ones a TLE carries: writing them as a TLE and
//! reading that back gives the same values. The first mean-motion derivative
//! is carried as stated; SGP4 does not propagate with it, and a catalog OMM
//! states the same value as its TLE.

use crate::astro::ndm::{
    self, check_unit, covariance6_unit, split_unit, FieldMap, KvnLine, UnitMismatch,
    COVARIANCE6_KEYS,
};
use crate::astro::sgp4::{
    self, ElementSet, Error as Sgp4Error, JulianDate, Satellite, Sgp4InputErrorKind,
};
use crate::astro::tle;
use crate::astro::xml;
use crate::validate;
use roxmltree::{Document, Node};
use std::fmt::{self, Write as _};

/// Header keywords, CCSDS 502.0-B-3 table 4-1 (`COMMENT` is handled apart).
const HEADER_KEYS: &[&str] = &[
    "CCSDS_OMM_VERS",
    "CLASSIFICATION",
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
];
/// Metadata keywords, CCSDS 502.0-B-3 table 4-2.
const METADATA_KEYS: &[&str] = &[
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
    "MEAN_ELEMENT_THEORY",
];
/// Mean Keplerian elements keywords, CCSDS 502.0-B-3 table 4-3.
const MEAN_ELEMENT_KEYS: &[&str] = &[
    "EPOCH",
    "SEMI_MAJOR_AXIS",
    "MEAN_MOTION",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "MEAN_ANOMALY",
    "GM",
];
/// Spacecraft parameters keywords, CCSDS 502.0-B-3 table 4-3.
const SPACECRAFT_KEYS: &[&str] = &[
    "MASS",
    "SOLAR_RAD_AREA",
    "SOLAR_RAD_COEFF",
    "DRAG_AREA",
    "DRAG_COEFF",
];
/// TLE related parameters keywords, CCSDS 502.0-B-3 table 4-3.
const TLE_PARAMETER_KEYS: &[&str] = &[
    "EPHEMERIS_TYPE",
    "CLASSIFICATION_TYPE",
    "NORAD_CAT_ID",
    "ELEMENT_SET_NO",
    "REV_AT_EPOCH",
    "BSTAR",
    "BTERM",
    "MEAN_MOTION_DOT",
    "MEAN_MOTION_DDOT",
    "AGOM",
];
/// Covariance reference-frame keyword; the 21 matrix keywords are
/// [`COVARIANCE6_KEYS`].
const COV_REF_FRAME: &str = "COV_REF_FRAME";
/// Prefix of a user-defined keyword, CCSDS 502.0-B-3 table 4-3.
const USER_DEFINED_PREFIX: &str = "USER_DEFINED_";
/// Comment keyword and XML element name.
const COMMENT: &str = "COMMENT";

/// Labels of `MEAN_ELEMENT_THEORY` that name the SGP4 model this crate
/// propagates. `SGP4` is the table 4-2 value and the CelesTrak XML label;
/// `SGP/SGP4` is the label 502.0-B-3 table 4-3 and annex G use for TLE-based
/// messages and the CelesTrak KVN label; `SDP4` names the deep-space branch of
/// the same NORAD element set, which this crate's SGP4 includes. `SGP`,
/// `SGP4-XP`, `PPT3`, `DSST` and `USM` are other theories.
const SGP4_THEORY_LABELS: &[&str] = &["SGP4", "SGP/SGP4", "SDP4"];

/// CSV columns emitted by the compact GP CSV writer.
const GP_CSV_FIELDS: &[&str] = &[
    "OBJECT_NAME",
    "OBJECT_ID",
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
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OmmEpoch {
    /// Civil year copied from the parsed `EPOCH`, or derived from an SGP4 split
    /// Julian date without applying the TLE two-digit-year pivot.
    pub year: i32,
    /// Civil month copied from the parsed `EPOCH`, or derived from an SGP4 split
    /// Julian date when a fit produces an [`OmmEpoch`].
    pub month: u32,
    /// Civil day copied from the parsed `EPOCH`, or derived from an SGP4 split
    /// Julian date when a fit produces an [`OmmEpoch`].
    pub day: u32,
    /// Civil hour passed to the calendar-to-Julian-date conversion; parsed
    /// year-end OMMs retain hour 23 when bridged to SGP4.
    pub hour: u32,
    /// Civil minute passed to the calendar-to-Julian-date conversion; the ISS
    /// fixture's `04:32` epoch decodes this field as 32.
    pub minute: u32,
    /// Civil second copied from the parsed `EPOCH`; UTC-like input may retain
    /// leap-second value 60, while continuous-time input rejects it.
    pub second: u32,
    /// Fractional second expressed in whole microseconds (0..=999_999).
    pub microsecond: u32,
    /// Fractional remainder within `microsecond`, in whole femtoseconds
    /// (0..=999_999_999). Ordinary catalog messages usually leave this at zero;
    /// fitted OMMs use it to avoid losing sub-microsecond split-JD precision.
    #[serde(default, skip_serializing_if = "is_zero_u32")]
    pub femtosecond: u32,
}

/// Canonical, format-agnostic OMM container.
///
/// Pure data: it knows nothing about KVN/XML/JSON serialization. The numeric
/// element values use standard astrodynamic units (angles in degrees, mean
/// motion in revolutions/day, its derivatives in rev/day^2 and rev/day^3, B\*
/// in inverse earth-radii) and are stored as directly parsed `f64`s, so every
/// encoding decodes to the same value.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Omm {
    // -- Header / metadata --
    /// `CCSDS_OMM_VERS` as the message states it: the KVN keyword, the XML
    /// `version` attribute, or the GP JSON member or CSV column. `None` when
    /// the message states no version, as CelesTrak GP JSON and CSV do; each
    /// writer states it only when present.
    pub ccsds_omm_vers: Option<String>,
    /// Optional header `CLASSIFICATION` text (502.0-B-3 table 4-1), the
    /// free-text message classification or caveat. It is distinct from the TLE
    /// parameter `CLASSIFICATION_TYPE`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub classification: Option<String>,
    /// Optional validated `CREATION_DATE` header text, emitted as empty/`null`
    /// when absent according to the selected encoding.
    pub creation_date: Option<String>,
    /// Optional validated `ORIGINATOR` header text, emitted as empty/`null`
    /// when absent according to the selected encoding.
    pub originator: Option<String>,
    /// Optional header `MESSAGE_ID` text (502.0-B-3 table 4-1), emitted only
    /// when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub message_id: Option<String>,
    /// Optional `OBJECT_NAME` text used by constellation conversion to resolve
    /// a PRN and retained in CelesTrak source or skipped-entry identity.
    pub object_name: Option<String>,
    /// International designator, CCSDS form (e.g. `"1998-067A"`).
    pub object_id: Option<String>,
    /// Optional `CENTER_NAME` metadata; CelesTrak JSON may omit it, yielding
    /// `None` in cross-encoding comparisons. A stated value other than `EARTH`
    /// is refused by [`Omm::to_element_set`].
    pub center_name: Option<String>,
    /// Optional `REF_FRAME` metadata; CelesTrak JSON may omit it, yielding
    /// `None` in cross-encoding comparisons. A stated value other than `TEME`
    /// is refused by [`Omm::to_element_set`].
    pub ref_frame: Option<String>,
    /// Optional `REF_FRAME_EPOCH` text (502.0-B-3 table 4-2), retained without
    /// date conversion and emitted only when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ref_frame_epoch: Option<String>,
    /// Optional `TIME_SYSTEM` metadata that selects UTC-like parsing for absent,
    /// `UTC`, `GLO`, and `GLONASS` labels and continuous parsing otherwise. A
    /// stated value other than `UTC` is refused by [`Omm::to_element_set`].
    pub time_system: Option<String>,
    /// Optional validated `MEAN_ELEMENT_THEORY` text; XML encoding preserves
    /// legal carriage returns through character-reference escaping. A stated
    /// value other than `SGP4`, `SGP/SGP4` or `SDP4` is refused by
    /// [`Omm::to_element_set`].
    pub mean_element_theory: Option<String>,

    // -- Mean elements --
    /// Parsed `EPOCH` calendar value emitted as ISO-8601 and used by
    /// [`Omm::to_element_set`] when no exact in-memory split Julian date exists.
    pub epoch: OmmEpoch,
    /// `MEAN_MOTION`, revolutions per day. CCSDS 502.0-B-3 table 4-3 requires
    /// it or `SEMI_MAJOR_AXIS`; SGP4 propagation requires it (4.2.4.6).
    pub mean_motion: Option<f64>,
    /// `SEMI_MAJOR_AXIS`, km, the table 4-3 alternative to `MEAN_MOTION` for
    /// theories other than SGP/SGP4. The SGP4 bridge does not use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub semi_major_axis_km: Option<f64>,
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
    /// Optional gravitational coefficient `GM`, km^3/s^2 (502.0-B-3 table
    /// 4-3). The SGP4 bridge does not use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gm_km3_s2: Option<f64>,
    /// Optional spacecraft-parameters block. Present when any of its keywords
    /// occurs in KVN, JSON or CSV, even with a blank value, or when the XML
    /// `spacecraftParameters` element does.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spacecraft: Option<OmmSpacecraft>,

    // -- TLE related parameters (502.0-B-3 table 4-3) --
    /// `EPHEMERIS_TYPE`, `None` when the message does not state it; the table
    /// gives 0 as its default. Fit-generated OMMs set 0.
    pub ephemeris_type: Option<i32>,
    /// `CLASSIFICATION_TYPE`, `None` when the message does not state it; the
    /// table gives `U` as its default. Fit-generated OMMs copy the
    /// classification from their TLE metadata.
    pub classification_type: Option<String>,
    /// `NORAD_CAT_ID` parsed as `u32`, required by table 4-3 only for SGP/SGP4
    /// messages; it feeds the SGP4 catalog number and constellation record
    /// identity, and the SGP4 bridge requires it.
    pub norad_cat_id: Option<u32>,
    /// `ELEMENT_SET_NO`, `None` when the message does not state it; fit-
    /// generated OMMs copy `TleMetadata::element_set_number`.
    pub element_set_no: Option<i32>,
    /// `REV_AT_EPOCH`, `None` when the message does not state it; TLE
    /// conversion requires a fit-generated value to fit in `i32`.
    pub rev_at_epoch: Option<i64>,
    /// SGP4 drag term B\*, inverse earth-radii; the SGP4 bridge requires it.
    pub bstar: Option<f64>,
    /// `BTERM`, the SGP4-XP ballistic coefficient CD*A/m, m^2/kg (table 4-3).
    /// The SGP4 bridge does not use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bterm_m2_kg: Option<f64>,
    /// First derivative of mean motion, rev/day^2; the SGP4 bridge requires it.
    pub mean_motion_dot: Option<f64>,
    /// Second derivative of mean motion, rev/day^3; the SGP4 bridge requires it.
    pub mean_motion_ddot: Option<f64>,
    /// `AGOM`, the SGP4-XP solar radiation pressure coefficient gamma*A/m,
    /// m^2/kg (table 4-3). The SGP4 bridge does not use it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agom_m2_kg: Option<f64>,
    /// Optional position/velocity covariance block. Present when
    /// `COV_REF_FRAME` or any matrix keyword occurs, or when the XML
    /// `covarianceMatrix` element does; a present block must give all 21
    /// matrix values (502.0-B-3 table 4-3).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub covariance: Option<OmmCovariance>,
    /// `USER_DEFINED_*` parameters in source order (502.0-B-3 table 4-3,
    /// 505.0-B-3 4.10), with values kept as verbatim text.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub user_defined: Vec<OmmUserDefined>,
    /// Comments of the header, metadata, mean-elements, TLE-parameters and
    /// user-defined blocks. Spacecraft and covariance comments live in
    /// [`OmmSpacecraft`] and [`OmmCovariance`].
    #[serde(default, skip_serializing_if = "OmmComments::is_empty")]
    pub comments: OmmComments,
    /// Exact split SGP4 epoch for in-memory producers that already have the
    /// split JD. CCSDS encodings do not carry this side channel; parsed catalog
    /// messages leave it `None` and rebuild the split from `epoch`.
    ///
    /// Scope: this preserves the producer's exact `(whole, fraction)` split
    /// *in memory only*. Through encode -> reparse the epoch travels as the
    /// femtosecond-rounded calendar text and is rebuilt as a canonical
    /// midnight-anchored split: the same instant to femtosecond precision (and
    /// bit-identical when the producer's split was already midnight-anchored),
    /// but not necessarily the same split representation, which SGP4's split
    /// tsince subtraction is sensitive to at the last ULP.
    #[serde(default, skip)]
    pub exact_sgp4_epoch: Option<JulianDate>,
    /// Whether TLE-derived GP fields (B\*, the second mean-motion derivative)
    /// should be snapped to the TLE assumed-decimal grid when bridging into
    /// SGP4. Parsed catalog OMMs default to `true`: their GP values
    /// originate in the TLE field format, so the historical compatibility path
    /// reproduces the value a TLE consumer would see. Fitted OMMs set this
    /// `false`: their elements were estimated directly and never lived on the
    /// TLE grid, so snapping would discard converged precision for no
    /// compatibility gain; `to_element_set` then passes them through losslessly.
    #[serde(default = "default_quantize_tle_derived_fields", skip_serializing)]
    pub quantize_tle_derived_fields: bool,
}

/// Comments of the OMM blocks whose presence does not depend on optional
/// keywords (CCSDS 502.0-B-3 7.8.8), each in source order.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OmmComments {
    /// Header comments, written after `CCSDS_OMM_VERS`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub header: Vec<String>,
    /// Metadata comments, written before `OBJECT_NAME`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub metadata: Vec<String>,
    /// Mean-elements comments, written before `EPOCH`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mean_elements: Vec<String>,
    /// TLE-parameters comments, written before `EPHEMERIS_TYPE`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tle_parameters: Vec<String>,
    /// User-defined-parameters comments, written before the first
    /// `USER_DEFINED_*` keyword.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub user_defined: Vec<String>,
}

impl OmmComments {
    /// Whether no block carries a comment.
    pub fn is_empty(&self) -> bool {
        self.header.is_empty()
            && self.metadata.is_empty()
            && self.mean_elements.is_empty()
            && self.tle_parameters.is_empty()
            && self.user_defined.is_empty()
    }
}

/// Optional OMM spacecraft parameters (CCSDS 502.0-B-3 table 4-3).
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OmmSpacecraft {
    /// Comments at the start of the block, in source order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<String>,
    /// `MASS`, kg.
    pub mass_kg: Option<f64>,
    /// `SOLAR_RAD_AREA`, m^2.
    pub solar_rad_area_m2: Option<f64>,
    /// `SOLAR_RAD_COEFF`, dimensionless.
    pub solar_rad_coeff: Option<f64>,
    /// `DRAG_AREA`, m^2.
    pub drag_area_m2: Option<f64>,
    /// `DRAG_COEFF`, dimensionless.
    pub drag_coeff: Option<f64>,
}

/// Optional OMM position/velocity covariance (CCSDS 502.0-B-3 table 4-3).
///
/// The values are kept exactly as read; no symmetry or definiteness check is
/// applied, since 502.0-B-3 section 4 places none on the OMM matrix.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct OmmCovariance {
    /// Comments at the start of the block, in source order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub comments: Vec<String>,
    /// Optional `COV_REF_FRAME`; when absent the matrix is in `REF_FRAME`.
    pub cov_ref_frame: Option<String>,
    /// The 21 lower-triangle values in keyword order `CX_X`, `CY_X`, `CY_Y`,
    /// `CZ_X`, `CZ_Y`, `CZ_Z`, `CX_DOT_X` ... `CZ_DOT_Z_DOT`: km^2 for two
    /// position components, km^2/s for one position and one velocity
    /// component, km^2/s^2 for two velocity components.
    pub lower_triangle: [f64; 21],
}

/// One `USER_DEFINED_*` parameter (CCSDS 502.0-B-3 table 4-3, 505.0-B-3 4.10).
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OmmUserDefined {
    /// The text after `USER_DEFINED_` in the KVN keyword, which is the XML
    /// `parameter` attribute.
    pub parameter: String,
    /// The value text, verbatim, including any units it states.
    pub value: String,
}

/// Failure modes of the OMM readers, writers and SGP4 bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OmmError {
    /// A required field was absent from the message.
    MissingField(&'static str),
    /// A decoded scalar field failed boundary validation.
    InvalidField {
        /// Static field label returned by the shared validator.
        field: &'static str,
        /// OMM validation category derived from the shared validator.
        kind: OmmInputErrorKind,
    },
    /// A numeric/integer field could not be parsed.
    Field(String),
    /// The `EPOCH` value was malformed.
    Epoch(String),
    /// A single-valued keyword occurred more than once with different values.
    DuplicateField {
        /// The repeated keyword.
        field: String,
        /// The value of its first occurrence.
        first: String,
        /// The later value that differs.
        second: String,
    },
    /// A KVN keyword or XML element that CCSDS 502.0-B-3 tables 4-1 to 4-3 do
    /// not define at that position (4.2.2.2, 4.2.3.2, 4.2.4.2) and that
    /// carries a value. XML elements are named with their parent element.
    UnknownField(String),
    /// A GP CSV data row whose field count differs from the header's.
    CsvColumnCount {
        /// The number of fields in the row.
        found: usize,
        /// The number of header columns.
        expected: usize,
    },
    /// GP CSV has no form for a block that is present but holds no value, such
    /// as a spacecraft-parameters block read from a blank `MASS`: every
    /// record fills every column, and an empty cell reads back as an absent
    /// value. The payload names the block.
    CsvEmptyBlock(&'static str),
    /// A KVN line that is not blank, a comment, or a `keyword = value`
    /// assignment (502.0-B-3 7.3.1, 7.4.1).
    MalformedLine {
        /// One-based line number.
        line: usize,
        /// The trimmed line text.
        text: String,
    },
    /// A stated unit contradicts the unit 502.0-B-3 table 4-3 defines for the
    /// keyword (7.7.1.1, 8.9.11).
    UnitMismatch {
        /// The keyword whose value carried the unit.
        field: String,
        /// The stated unit.
        unit: String,
        /// The table unit, or `None` for a dimensionless or text keyword.
        expected: Option<&'static str>,
    },
    /// A document handed to a single-record reader ([`parse_xml`],
    /// [`parse_json`], [`parse_csv`]) holds several OMMs; [`parse_xml_all`],
    /// [`parse_json_array`] and [`parse_csv_array`] read all of them.
    MultipleMessages {
        /// The number of OMM records in the document.
        count: usize,
    },
    /// A record handed to [`encode_json_array`] or [`encode_csv`] could not be
    /// written.
    InRecord {
        /// Zero-based position of the record in the slice.
        index: usize,
        /// The failure of that record.
        source: Box<OmmError>,
    },
    /// Two GP CSV records order two `USER_DEFINED_*` parameters oppositely.
    /// GP CSV gives every record one column order, and the reader lists a
    /// record's parameters in that order, so one of the two would read back
    /// reordered.
    CsvColumnOrder {
        /// The parameter the refused record gives first.
        first: String,
        /// The parameter it gives after `first`, which an earlier record gives
        /// before `first`.
        second: String,
    },
    /// An explicitly stated metadata value is not the Earth/TEME/UTC/SGP4
    /// convention SGP4 propagation requires (502.0-B-3 4.2.4.6).
    IncompatibleMetadata {
        /// The metadata keyword: `CENTER_NAME`, `REF_FRAME`, `TIME_SYSTEM` or
        /// `MEAN_ELEMENT_THEORY`.
        field: &'static str,
        /// The stated value.
        value: String,
    },
    /// A writer cannot write a text value so that its reader returns it
    /// unchanged.
    UnwritableText {
        /// The keyword, or `COMMENT`.
        field: String,
        /// The text that cannot be written.
        value: String,
        /// Why the text would not read back unchanged.
        issue: TextIssue,
    },
}

/// Why an OMM writer refuses a text value.
pub use crate::astro::ndm::TextIssue;

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
            OmmError::DuplicateField {
                field,
                first,
                second,
            } => write!(
                f,
                "OMM keyword {field} occurs with different values {first:?} and {second:?}"
            ),
            OmmError::UnknownField(name) => write!(f, "OMM has no keyword {name}"),
            OmmError::CsvColumnCount { found, expected } => write!(
                f,
                "GP CSV row holds {found} fields; the header names {expected}"
            ),
            OmmError::CsvEmptyBlock(block) => write!(
                f,
                "GP CSV cannot state an empty {block} block; an empty cell reads back as absent"
            ),
            OmmError::MalformedLine { line, text } => write!(
                f,
                "OMM line {line} is not a comment or keyword assignment: {text:?}"
            ),
            OmmError::UnitMismatch {
                field,
                unit,
                expected,
            } => write!(
                f,
                "OMM keyword {field} states unit [{unit}], expected {}",
                ndm::expected_unit_label(*expected)
            ),
            OmmError::MultipleMessages { count } => write!(
                f,
                "document holds {count} OMM records; parse_xml_all, parse_json_array and parse_csv_array read all of them"
            ),
            OmmError::InRecord { index, source } => write!(f, "OMM record {index}: {source}"),
            OmmError::CsvColumnOrder { first, second } => write!(
                f,
                "GP CSV record gives USER_DEFINED_{first} before USER_DEFINED_{second}, which an earlier record orders the other way"
            ),
            OmmError::IncompatibleMetadata { field, value } => write!(
                f,
                "OMM {field} = {value:?} is not the {} that SGP4 propagation requires (CCSDS 502.0-B-3 4.2.4.6)",
                bridge_requirement(field)
            ),
            OmmError::UnwritableText {
                field,
                value,
                issue,
            } => write!(f, "OMM {field} value {value:?} {issue}"),
        }
    }
}

impl std::error::Error for OmmError {}

/// The accepted value named in an [`OmmError::IncompatibleMetadata`] message.
fn bridge_requirement(field: &str) -> &'static str {
    match field {
        "CENTER_NAME" => "EARTH",
        "REF_FRAME" => "TEME",
        "TIME_SYSTEM" => "UTC",
        _ => "SGP4, SGP/SGP4 or SDP4 theory",
    }
}

// ── KVN ──────────────────────────────────────────────────────────────

/// Parse a CCSDS OMM in KVN (`KEY = VALUE`) encoding into an [`Omm`].
///
/// Lines end at CR, LF, CR LF or LF CR (502.0-B-3 7.3.7). Blank lines are
/// ignored and keys and values are trimmed; any other line must be a comment
/// or an assignment. Numeric values accept the CelesTrak forms, including a
/// leading decimal point (`.0004737`) and scientific notation (`.17172E-3`),
/// and may carry the table 4-3 unit in brackets (`15.5 [rev/day]`). Text
/// values are kept verbatim, including any trailing bracketed text.
pub fn parse_kvn(text: &str) -> Result<Omm, OmmError> {
    let mut fields = OmmFields::default();
    let mut pending_comments: Vec<String> = Vec::new();
    let mut last_block = OmmBlock::Header;
    for (index, line) in ndm::kvn_lines(text).into_iter().enumerate() {
        match ndm::classify(line) {
            KvnLine::Blank => {}
            KvnLine::Comment(comment) => pending_comments.push(comment.to_string()),
            KvnLine::Assignment { key, value } => {
                let Some(block) = OmmBlock::of_keyword(key) else {
                    // A keyword the tables do not define is refused when it
                    // carries a value; a blank one holds nothing to keep.
                    if value.is_empty() {
                        continue;
                    }
                    return Err(OmmError::UnknownField(key.to_string()));
                };
                for comment in pending_comments.drain(..) {
                    fields.push_comment(block, comment);
                }
                fields.push_kvn_value(key, value)?;
                last_block = block;
            }
            KvnLine::Other(other) => {
                return Err(OmmError::MalformedLine {
                    line: index + 1,
                    text: other.to_string(),
                })
            }
        }
    }
    for comment in pending_comments {
        fields.push_comment(last_block, comment);
    }
    Omm::from_fields(fields)
}

/// Encode an [`Omm`] as a CCSDS OMM KVN message.
///
/// Numeric values use their shortest round-tripping decimal form, so parsing the
/// output reproduces the same `f64`s. The epoch is emitted to microseconds,
/// extended to femtoseconds only when a sub-microsecond remainder is present.
/// Each block's comments are written at the start of the block, and optional
/// keywords only when present. An absent or empty optional text value is
/// written blank, which reads back as absent.
///
/// Text that would not read back unchanged is refused with
/// [`OmmError::UnwritableText`]: a line break, whitespace the reader trims, an
/// XML-illegal character, an empty `CCSDS_OMM_VERS` (which reads back as
/// absent), a `USER_DEFINED_*` parameter containing `=` or given more than
/// once, or user-defined comments with no user-defined parameter to precede.
/// `CCSDS_OMM_VERS` is written only when the message holds one. A non-finite number, and an
/// [`OmmEpoch`] that names no instant the reader accepts under the message's
/// `TIME_SYSTEM`, are refused with [`OmmError::InvalidField`]. Comments of an
/// empty spacecraft or TLE-parameter block are written before a blank `MASS`
/// or `EPHEMERIS_TYPE`, which reads back as the same empty block.
pub fn encode_kvn(omm: &Omm) -> Result<String, OmmError> {
    check_user_defined_names(omm)?;
    let epoch = epoch_text(omm)?;
    let mut out = KvnWriter::default();
    if let Some(version) = &omm.ccsds_omm_vers {
        out.required_text("CCSDS_OMM_VERS", version)?;
    }
    out.comments(&omm.comments.header)?;
    out.present_text("CLASSIFICATION", omm.classification.as_deref())?;
    out.text("CREATION_DATE", omm.creation_date.as_deref())?;
    out.text("ORIGINATOR", omm.originator.as_deref())?;
    out.present_text("MESSAGE_ID", omm.message_id.as_deref())?;

    out.comments(&omm.comments.metadata)?;
    out.text("OBJECT_NAME", omm.object_name.as_deref())?;
    out.text("OBJECT_ID", omm.object_id.as_deref())?;
    out.text("CENTER_NAME", omm.center_name.as_deref())?;
    out.text("REF_FRAME", omm.ref_frame.as_deref())?;
    out.present_text("REF_FRAME_EPOCH", omm.ref_frame_epoch.as_deref())?;
    out.text("TIME_SYSTEM", omm.time_system.as_deref())?;
    out.text("MEAN_ELEMENT_THEORY", omm.mean_element_theory.as_deref())?;

    out.comments(&omm.comments.mean_elements)?;
    out.line("EPOCH", &epoch);
    out.present_number("SEMI_MAJOR_AXIS", omm.semi_major_axis_km)?;
    out.present_number("MEAN_MOTION", omm.mean_motion)?;
    out.number("ECCENTRICITY", omm.eccentricity)?;
    out.number("INCLINATION", omm.inclination_deg)?;
    out.number("RA_OF_ASC_NODE", omm.ra_of_asc_node_deg)?;
    out.number("ARG_OF_PERICENTER", omm.arg_of_pericenter_deg)?;
    out.number("MEAN_ANOMALY", omm.mean_anomaly_deg)?;
    if let Some(gm) = omm.gm_km3_s2 {
        out.number("GM", gm)?;
    }

    if let Some(spacecraft) = &omm.spacecraft {
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

    out.comments(&omm.comments.tle_parameters)?;
    if !has_tle_parameters(omm) && !omm.comments.tle_parameters.is_empty() {
        // A blank EPHEMERIS_TYPE keeps the comments in their block on read.
        out.line("EPHEMERIS_TYPE", "");
    }
    out.present_line("EPHEMERIS_TYPE", omm.ephemeris_type.map(|v| v.to_string()));
    out.present_text("CLASSIFICATION_TYPE", omm.classification_type.as_deref())?;
    out.present_line("NORAD_CAT_ID", omm.norad_cat_id.map(|v| v.to_string()));
    out.present_line("ELEMENT_SET_NO", omm.element_set_no.map(|v| v.to_string()));
    out.present_line("REV_AT_EPOCH", omm.rev_at_epoch.map(|v| v.to_string()));
    out.present_number("BSTAR", omm.bstar)?;
    out.present_number("BTERM", omm.bterm_m2_kg)?;
    out.present_number("MEAN_MOTION_DOT", omm.mean_motion_dot)?;
    out.present_number("MEAN_MOTION_DDOT", omm.mean_motion_ddot)?;
    out.present_number("AGOM", omm.agom_m2_kg)?;

    if let Some(covariance) = &omm.covariance {
        out.comments(&covariance.comments)?;
        out.present_text(COV_REF_FRAME, covariance.cov_ref_frame.as_deref())?;
        for (key, value) in COVARIANCE6_KEYS.into_iter().zip(covariance.lower_triangle) {
            out.number(key, value)?;
        }
    }

    if omm.user_defined.is_empty() {
        if let Some(comment) = omm.comments.user_defined.first() {
            return Err(OmmError::UnwritableText {
                field: COMMENT.to_string(),
                value: comment.clone(),
                issue: TextIssue::DetachedComment,
            });
        }
    }
    out.comments(&omm.comments.user_defined)?;
    for parameter in &omm.user_defined {
        out.user_defined(parameter)?;
    }
    Ok(out.text)
}

/// Line assembly for [`encode_kvn`], checking that every value reads back.
#[derive(Default)]
struct KvnWriter {
    text: String,
}

impl KvnWriter {
    fn line(&mut self, key: &str, value: &str) {
        self.text.push_str(key);
        self.text.push_str(" = ");
        self.text.push_str(value);
        self.text.push('\n');
    }

    /// A text keyword that is always written; `None` and empty write a blank
    /// value, which reads back as absent.
    fn text(&mut self, key: &str, value: Option<&str>) -> Result<(), OmmError> {
        let value = value.unwrap_or_default();
        if !value.is_empty() {
            check_kvn_text(key, value)?;
        }
        self.line(key, value);
        Ok(())
    }

    /// A text keyword written only when present.
    fn present_text(&mut self, key: &str, value: Option<&str>) -> Result<(), OmmError> {
        match value {
            Some(value) => self.text(key, Some(value)),
            None => Ok(()),
        }
    }

    /// A text keyword whose empty value would read back as a default.
    fn required_text(&mut self, key: &str, value: &str) -> Result<(), OmmError> {
        if value.is_empty() {
            return Err(OmmError::UnwritableText {
                field: key.to_string(),
                value: String::new(),
                issue: TextIssue::Empty,
            });
        }
        self.text(key, Some(value))
    }

    fn number(&mut self, key: &'static str, value: f64) -> Result<(), OmmError> {
        check_finite(key, value)?;
        self.line(key, &fmt_num(value));
        Ok(())
    }

    fn present_number(&mut self, key: &'static str, value: Option<f64>) -> Result<(), OmmError> {
        match value {
            Some(value) => self.number(key, value),
            None => Ok(()),
        }
    }

    fn present_line(&mut self, key: &str, value: Option<String>) {
        if let Some(value) = value {
            self.line(key, &value);
        }
    }

    fn comments(&mut self, comments: &[String]) -> Result<(), OmmError> {
        for comment in comments {
            let issue = if comment.contains(['\n', '\r']) {
                Some(TextIssue::LineBreak)
            } else if comment.trim_end() != comment.as_str() {
                Some(TextIssue::SurroundingWhitespace)
            } else if xml::first_illegal_xml_1_0_char(comment).is_some() {
                Some(TextIssue::XmlIllegalCharacter)
            } else {
                None
            };
            if let Some(issue) = issue {
                return Err(OmmError::UnwritableText {
                    field: COMMENT.to_string(),
                    value: comment.clone(),
                    issue,
                });
            }
            if comment.is_empty() {
                self.text.push_str(COMMENT);
            } else {
                self.text.push_str(COMMENT);
                self.text.push(' ');
                self.text.push_str(comment);
            }
            self.text.push('\n');
        }
        Ok(())
    }

    fn user_defined(&mut self, parameter: &OmmUserDefined) -> Result<(), OmmError> {
        let key = format!("{USER_DEFINED_PREFIX}{}", parameter.parameter);
        let name_issue = if parameter.parameter.contains(['\n', '\r']) {
            Some(TextIssue::LineBreak)
        } else if parameter.parameter.contains('=') {
            Some(TextIssue::KeywordSeparator)
        } else if parameter.parameter.trim_end() != parameter.parameter.as_str() {
            Some(TextIssue::SurroundingWhitespace)
        } else if xml::first_illegal_xml_1_0_char(&parameter.parameter).is_some() {
            Some(TextIssue::XmlIllegalCharacter)
        } else {
            None
        };
        if let Some(issue) = name_issue {
            return Err(OmmError::UnwritableText {
                field: key,
                value: parameter.parameter.clone(),
                issue,
            });
        }
        if !parameter.value.is_empty() {
            check_kvn_text(&key, &parameter.value)?;
        }
        self.line(&key, &parameter.value);
        Ok(())
    }
}

/// Refuse a non-finite number, which no OMM reader accepts.
fn check_finite(key: &'static str, value: f64) -> Result<(), OmmError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(OmmError::InvalidField {
            field: key,
            kind: OmmInputErrorKind::NonFinite,
        })
    }
}

/// The `EPOCH` text of an OMM, refusing an [`OmmEpoch`] the readers would not
/// return unchanged: a calendar date or time outside the civil range, a `60`
/// second that is not a leap second under a UTC-like `TIME_SYSTEM` or any
/// `60` second under a continuous one, or a sub-second field beyond its range.
fn epoch_text(omm: &Omm) -> Result<String, OmmError> {
    validate_epoch(
        &omm.epoch,
        omm_civil_second_policy(omm.time_system.as_deref()),
    )?;
    Ok(omm.epoch.to_iso8601())
}

/// Refuse a `USER_DEFINED_*` parameter name given more than once. Every OMM
/// reader keeps one of two equal repeats and refuses two that differ, so no
/// encoding carries such a list back unchanged.
fn check_user_defined_names(omm: &Omm) -> Result<(), OmmError> {
    for (index, parameter) in omm.user_defined.iter().enumerate() {
        if omm.user_defined[..index]
            .iter()
            .any(|earlier| earlier.parameter == parameter.parameter)
        {
            return Err(OmmError::UnwritableText {
                field: format!("{USER_DEFINED_PREFIX}{}", parameter.parameter),
                value: parameter.parameter.clone(),
                issue: TextIssue::RepeatedParameter,
            });
        }
    }
    Ok(())
}

/// Whether any TLE related parameter is present.
fn has_tle_parameters(omm: &Omm) -> bool {
    omm.ephemeris_type.is_some()
        || omm.classification_type.is_some()
        || omm.norad_cat_id.is_some()
        || omm.element_set_no.is_some()
        || omm.rev_at_epoch.is_some()
        || omm.bstar.is_some()
        || omm.bterm_m2_kg.is_some()
        || omm.mean_motion_dot.is_some()
        || omm.mean_motion_ddot.is_some()
        || omm.agom_m2_kg.is_some()
}

/// Refuse a non-empty text value that would not read back unchanged.
fn check_kvn_text(key: &str, value: &str) -> Result<(), OmmError> {
    let issue = if value.contains(['\n', '\r']) {
        Some(TextIssue::LineBreak)
    } else if value.trim() != value {
        Some(TextIssue::SurroundingWhitespace)
    } else if xml::first_illegal_xml_1_0_char(value).is_some() {
        Some(TextIssue::XmlIllegalCharacter)
    } else {
        None
    };
    match issue {
        Some(issue) => Err(OmmError::UnwritableText {
            field: key.to_string(),
            value: value.to_string(),
            issue,
        }),
        None => Ok(()),
    }
}

/// The spacecraft keywords with their values, in table 4-3 order.
fn spacecraft_values(spacecraft: &OmmSpacecraft) -> [(&'static str, Option<f64>); 5] {
    [
        ("MASS", spacecraft.mass_kg),
        ("SOLAR_RAD_AREA", spacecraft.solar_rad_area_m2),
        ("SOLAR_RAD_COEFF", spacecraft.solar_rad_coeff),
        ("DRAG_AREA", spacecraft.drag_area_m2),
        ("DRAG_COEFF", spacecraft.drag_coeff),
    ]
}

// ── XML ──────────────────────────────────────────────────────────────

/// Parse a CCSDS OMM in XML encoding into an [`Omm`].
///
/// Uses the `roxmltree` DOM reader (which handles the `<?xml?>` declaration,
/// namespaces, comments, and entity decoding) and reads each value from the
/// element that owns it: header keywords from `<header>`, metadata from the
/// segment's `<metadata>`, and each data keyword from its logical-block element
/// (502.0-B-3 table 8-5). The format-set version is taken from the `version`
/// attribute on `<omm>`. The message may be the root element or sit inside an
/// `<ndm>` combined instantiation (505.0-B-3 4.11); a document holding more
/// than one OMM is refused with [`OmmError::MultipleMessages`], and
/// [`parse_xml_all`] reads such a document. An element the tables do not define
/// at its position is refused by name, and a `units` attribute must match the
/// table 4-3 unit (8.9.11).
pub fn parse_xml(text: &str) -> Result<Omm, OmmError> {
    let doc = parse_xml_document(text)?;
    let messages = ndm::message_elements(&doc, "omm");
    match messages.as_slice() {
        [] => Err(OmmError::Field("XML contains no omm element".to_string())),
        [message] => omm_from_xml(*message),
        _ => Err(OmmError::MultipleMessages {
            count: messages.len(),
        }),
    }
}

/// Parse every OMM in an XML document, in document order.
///
/// Reads a single `<omm>` message or every `<omm>` of an `<ndm>` combined
/// instantiation (505.0-B-3 4.11) through the same scoped reader as
/// [`parse_xml`]. A message that cannot be read is skipped and reported in
/// [`OmmArray::skipped`] with its position and reason, as [`parse_json_array`]
/// and [`parse_csv_array`] report a bad record; a document that is not
/// well-formed XML is still an error. A document without an OMM yields no
/// records.
pub fn parse_xml_all(text: &str) -> Result<OmmArray, OmmError> {
    let doc = parse_xml_document(text)?;
    let mut omms = Vec::new();
    let mut skipped = Vec::new();
    for (index, message) in ndm::message_elements(&doc, "omm").into_iter().enumerate() {
        match omm_from_xml(message) {
            Ok(omm) => omms.push(omm),
            Err(reason) => skipped.push(OmmSkippedRecord { index, reason }),
        }
    }
    Ok(OmmArray { omms, skipped })
}

fn parse_xml_document(text: &str) -> Result<Document<'_>, OmmError> {
    Document::parse(text).map_err(|e| OmmError::Field(format!("malformed XML: {e}")))
}

/// Read one `<omm>` element.
fn omm_from_xml(message: Node) -> Result<Omm, OmmError> {
    let mut fields = OmmFields::default();
    if let Some(version) = message.attribute("version") {
        fields
            .pairs
            .push(("CCSDS_OMM_VERS".to_string(), version.trim().to_string()));
    }
    for child in ndm::element_children(message) {
        match child.tag_name().name() {
            "header" => read_xml_block(child, OmmBlock::Header, &mut fields)?,
            "body" => read_xml_body(child, &mut fields)?,
            other => unknown_element("omm", other, child)?,
        }
    }
    Omm::from_fields(fields)
}

/// Read the single segment of an OMM `<body>` (502.0-B-3 8.9.6).
fn read_xml_body(body: Node, fields: &mut OmmFields) -> Result<(), OmmError> {
    let mut segments = Vec::new();
    for child in ndm::element_children(body) {
        match child.tag_name().name() {
            "segment" => segments.push(child),
            other => unknown_element("body", other, child)?,
        }
    }
    let segment = match segments.as_slice() {
        [segment] => *segment,
        _ => {
            return Err(OmmError::Field(format!(
                "OMM body holds {} segments; CCSDS 502.0-B-3 8.9.6 requires one",
                segments.len()
            )))
        }
    };
    for child in ndm::element_children(segment) {
        match child.tag_name().name() {
            "metadata" => read_xml_block(child, OmmBlock::Metadata, fields)?,
            "data" => read_xml_data(child, fields)?,
            other => unknown_element("segment", other, child)?,
        }
    }
    Ok(())
}

/// Read the logical blocks of an OMM `<data>` element (502.0-B-3 table 8-5).
fn read_xml_data(data: Node, fields: &mut OmmFields) -> Result<(), OmmError> {
    for child in ndm::element_children(data) {
        match child.tag_name().name() {
            COMMENT => {
                reject_nested_element(child)?;
                fields.push_comment(OmmBlock::MeanElements, ndm::comment_text(child));
            }
            "meanElements" => read_xml_block(child, OmmBlock::MeanElements, fields)?,
            "spacecraftParameters" => {
                fields.spacecraft_block = true;
                read_xml_block(child, OmmBlock::Spacecraft, fields)?;
            }
            "tleParameters" => read_xml_block(child, OmmBlock::TleParameters, fields)?,
            "covarianceMatrix" => {
                fields.covariance_block = true;
                read_xml_block(child, OmmBlock::Covariance, fields)?;
            }
            "userDefinedParameters" => read_xml_user_defined(child, fields)?,
            other => unknown_element("data", other, child)?,
        }
    }
    Ok(())
}

/// Refuse an element the tables do not define at its position when it holds a
/// value; an empty one holds nothing to keep.
fn unknown_element(parent: &str, name: &str, node: Node) -> Result<(), OmmError> {
    if ndm::carries_data(node) {
        Err(OmmError::UnknownField(format!("{parent}/{name}")))
    } else {
        Ok(())
    }
}

/// Refuse an element inside a keyword or comment element, whose content the
/// reader would otherwise not read.
fn reject_nested_element(node: Node) -> Result<(), OmmError> {
    match ndm::element_children(node).first() {
        Some(nested) => Err(OmmError::UnknownField(format!(
            "{}/{}",
            node.tag_name().name(),
            nested.tag_name().name()
        ))),
        None => Ok(()),
    }
}

/// Read the `COMMENT` and keyword elements of one block element.
fn read_xml_block(node: Node, block: OmmBlock, fields: &mut OmmFields) -> Result<(), OmmError> {
    for child in ndm::element_children(node) {
        reject_nested_element(child)?;
        let name = child.tag_name().name();
        if name == COMMENT {
            fields.push_comment(block, ndm::comment_text(child));
            continue;
        }
        if block == OmmBlock::UserDefined || OmmBlock::of_keyword(name) != Some(block) {
            unknown_element(node.tag_name().name(), name, child)?;
            continue;
        }
        fields.push_xml_value(name, &ndm::leaf_text(child), ndm::units_attribute(child))?;
    }
    Ok(())
}

/// Read `<userDefinedParameters>` (505.0-B-3 4.10).
fn read_xml_user_defined(node: Node, fields: &mut OmmFields) -> Result<(), OmmError> {
    for child in ndm::element_children(node) {
        reject_nested_element(child)?;
        match child.tag_name().name() {
            COMMENT => fields.push_comment(OmmBlock::UserDefined, ndm::comment_text(child)),
            "USER_DEFINED" => {
                let parameter = child.attribute("parameter").ok_or_else(|| {
                    OmmError::Field(
                        "USER_DEFINED element without a parameter attribute".to_string(),
                    )
                })?;
                if let Some(unit) = ndm::units_attribute(child) {
                    // 505.0-B-3 4.10.1.6 carries user-defined units in the value.
                    return Err(OmmError::UnitMismatch {
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

/// Encode an [`Omm`] as a CCSDS OMM XML message, following the CelesTrak/`ndm`
/// document layout. Numeric values use their shortest round-tripping form and
/// text values are XML-escaped, so parsing the output reproduces the same [`Omm`].
/// Optional keywords and blocks are written only when present, and each
/// block's comments as `COMMENT` elements at its start.
///
/// Text the reader would not return unchanged is refused with
/// [`OmmError::UnwritableText`]: a keyword value or `CCSDS_OMM_VERS` with
/// surrounding whitespace (the reader trims element text and the `version`
/// attribute), an empty `CCSDS_OMM_VERS` (which reads back as `2.0`), a
/// comment with trailing whitespace, a `USER_DEFINED_*` parameter given more
/// than once, and any character XML 1.0 cannot carry. A line break inside a
/// value or comment is written and reads back unchanged, a carriage return as
/// a character reference. A non-finite number, and an [`OmmEpoch`] that names
/// no instant the reader accepts under the message's `TIME_SYSTEM`, are
/// refused with [`OmmError::InvalidField`].
pub fn encode_xml(omm: &Omm) -> Result<String, OmmError> {
    check_user_defined_names(omm)?;
    let epoch = epoch_text(omm)?;
    let mut out = XmlWriter::default();
    out.raw("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<ndm>\n");
    match &omm.ccsds_omm_vers {
        Some(version) => {
            if version.is_empty() {
                return Err(OmmError::UnwritableText {
                    field: "CCSDS_OMM_VERS".to_string(),
                    value: String::new(),
                    issue: TextIssue::Empty,
                });
            }
            check_xml_text("CCSDS_OMM_VERS", version)?;
            out.raw(&format!(
                "<omm id=\"CCSDS_OMM_VERS\" version=\"{}\">\n",
                ndm::text::escape_attribute(version)
            ));
        }
        None => out.raw("<omm>\n"),
    }

    out.raw("<header>");
    out.comments(&omm.comments.header)?;
    out.present_text("CLASSIFICATION", omm.classification.as_deref())?;
    out.text("CREATION_DATE", omm.creation_date.as_deref())?;
    out.text("ORIGINATOR", omm.originator.as_deref())?;
    out.present_text("MESSAGE_ID", omm.message_id.as_deref())?;
    out.raw("</header>\n");

    out.raw("<body><segment>\n<metadata>");
    out.comments(&omm.comments.metadata)?;
    out.text("OBJECT_NAME", omm.object_name.as_deref())?;
    out.text("OBJECT_ID", omm.object_id.as_deref())?;
    out.text("CENTER_NAME", omm.center_name.as_deref())?;
    out.text("REF_FRAME", omm.ref_frame.as_deref())?;
    out.present_text("REF_FRAME_EPOCH", omm.ref_frame_epoch.as_deref())?;
    out.text("TIME_SYSTEM", omm.time_system.as_deref())?;
    out.text("MEAN_ELEMENT_THEORY", omm.mean_element_theory.as_deref())?;
    out.raw("</metadata>\n<data>\n<meanElements>");

    out.comments(&omm.comments.mean_elements)?;
    out.element("EPOCH", &epoch);
    out.present_number("SEMI_MAJOR_AXIS", omm.semi_major_axis_km)?;
    out.present_number("MEAN_MOTION", omm.mean_motion)?;
    out.number("ECCENTRICITY", omm.eccentricity)?;
    out.number("INCLINATION", omm.inclination_deg)?;
    out.number("RA_OF_ASC_NODE", omm.ra_of_asc_node_deg)?;
    out.number("ARG_OF_PERICENTER", omm.arg_of_pericenter_deg)?;
    out.number("MEAN_ANOMALY", omm.mean_anomaly_deg)?;
    out.present_number("GM", omm.gm_km3_s2)?;
    out.raw("</meanElements>\n");

    if let Some(spacecraft) = &omm.spacecraft {
        out.raw("<spacecraftParameters>");
        out.comments(&spacecraft.comments)?;
        for (key, value) in spacecraft_values(spacecraft) {
            out.present_number(key, value)?;
        }
        out.raw("</spacecraftParameters>\n");
    }

    if has_tle_parameters(omm) || !omm.comments.tle_parameters.is_empty() {
        out.raw("<tleParameters>");
        out.comments(&omm.comments.tle_parameters)?;
        if let Some(value) = omm.ephemeris_type {
            out.element("EPHEMERIS_TYPE", &value.to_string());
        }
        out.present_text("CLASSIFICATION_TYPE", omm.classification_type.as_deref())?;
        if let Some(value) = omm.norad_cat_id {
            out.element("NORAD_CAT_ID", &value.to_string());
        }
        if let Some(value) = omm.element_set_no {
            out.element("ELEMENT_SET_NO", &value.to_string());
        }
        if let Some(value) = omm.rev_at_epoch {
            out.element("REV_AT_EPOCH", &value.to_string());
        }
        out.present_number("BSTAR", omm.bstar)?;
        out.present_number("BTERM", omm.bterm_m2_kg)?;
        out.present_number("MEAN_MOTION_DOT", omm.mean_motion_dot)?;
        out.present_number("MEAN_MOTION_DDOT", omm.mean_motion_ddot)?;
        out.present_number("AGOM", omm.agom_m2_kg)?;
        out.raw("</tleParameters>\n");
    }

    if let Some(covariance) = &omm.covariance {
        out.raw("<covarianceMatrix>");
        out.comments(&covariance.comments)?;
        out.present_text(COV_REF_FRAME, covariance.cov_ref_frame.as_deref())?;
        for (key, value) in COVARIANCE6_KEYS.into_iter().zip(covariance.lower_triangle) {
            out.number(key, value)?;
        }
        out.raw("</covarianceMatrix>\n");
    }

    if !omm.user_defined.is_empty() || !omm.comments.user_defined.is_empty() {
        out.raw("<userDefinedParameters>");
        out.comments(&omm.comments.user_defined)?;
        for parameter in &omm.user_defined {
            let field = format!("{USER_DEFINED_PREFIX}{}", parameter.parameter);
            if xml::first_illegal_xml_1_0_char(&parameter.parameter).is_some() {
                return Err(OmmError::UnwritableText {
                    field,
                    value: parameter.parameter.clone(),
                    issue: TextIssue::XmlIllegalCharacter,
                });
            }
            check_xml_text(&field, &parameter.value)?;
            out.raw(&format!(
                "<USER_DEFINED parameter=\"{}\">{}</USER_DEFINED>",
                ndm::text::escape_attribute(&parameter.parameter),
                xml::escape(&parameter.value)
            ));
        }
        out.raw("</userDefinedParameters>\n");
    }

    out.raw("</data>\n</segment></body>\n</omm>\n</ndm>\n");
    Ok(out.text)
}

/// Element assembly for [`encode_xml`], checking that every value reads back.
#[derive(Default)]
struct XmlWriter {
    text: String,
}

impl XmlWriter {
    fn raw(&mut self, text: &str) {
        self.text.push_str(text);
    }

    /// An element whose content needs no escaping (numbers, epochs).
    fn element(&mut self, name: &str, content: &str) {
        let _ = write!(self.text, "<{name}>{content}</{name}>");
    }

    /// A text element that is always written; `None` writes an empty element,
    /// which reads back as absent.
    fn text(&mut self, name: &str, value: Option<&str>) -> Result<(), OmmError> {
        let value = value.unwrap_or_default();
        check_xml_text(name, value)?;
        self.element(name, &xml::escape(value));
        Ok(())
    }

    fn present_text(&mut self, name: &str, value: Option<&str>) -> Result<(), OmmError> {
        match value {
            Some(value) => self.text(name, Some(value)),
            None => Ok(()),
        }
    }

    fn number(&mut self, name: &'static str, value: f64) -> Result<(), OmmError> {
        check_finite(name, value)?;
        self.element(name, &fmt_num(value));
        Ok(())
    }

    fn present_number(&mut self, name: &'static str, value: Option<f64>) -> Result<(), OmmError> {
        match value {
            Some(value) => self.number(name, value),
            None => Ok(()),
        }
    }

    fn comments(&mut self, comments: &[String]) -> Result<(), OmmError> {
        for comment in comments {
            check_xml_comment(comment)?;
            self.element(COMMENT, &xml::escape(comment));
        }
        Ok(())
    }
}

/// Refuse element text the XML reader would not return unchanged: surrounding
/// whitespace, which it trims, or a character XML 1.0 cannot carry.
fn check_xml_text(field: &str, value: &str) -> Result<(), OmmError> {
    let issue = if value.trim() != value {
        Some(TextIssue::SurroundingWhitespace)
    } else if xml::first_illegal_xml_1_0_char(value).is_some() {
        Some(TextIssue::XmlIllegalCharacter)
    } else {
        None
    };
    match issue {
        Some(issue) => Err(OmmError::UnwritableText {
            field: field.to_string(),
            value: value.to_string(),
            issue,
        }),
        None => Ok(()),
    }
}

/// Refuse comment text the XML reader would not return unchanged: trailing
/// whitespace, which it removes, or a character XML 1.0 cannot carry.
fn check_xml_comment(comment: &str) -> Result<(), OmmError> {
    let issue = if comment.trim_end() != comment {
        Some(TextIssue::SurroundingWhitespace)
    } else if xml::first_illegal_xml_1_0_char(comment).is_some() {
        Some(TextIssue::XmlIllegalCharacter)
    } else {
        None
    };
    match issue {
        Some(issue) => Err(OmmError::UnwritableText {
            field: COMMENT.to_string(),
            value: comment.to_string(),
            issue,
        }),
        None => Ok(()),
    }
}

// ── JSON ─────────────────────────────────────────────────────────────

/// Parse a CCSDS/CelesTrak OMM in JSON encoding into an [`Omm`].
///
/// Accepts a single object, or an array holding one object. An array of
/// several objects, as CelesTrak GP queries return, is refused with
/// [`OmmError::MultipleMessages`] rather than read for its first record;
/// [`parse_json_array`] reads all of them. Each member is mapped onto the
/// shared `(key, value)` field set - numbers stringified, strings taken
/// verbatim (so the Space-Track quirk of quoting numeric values is handled) -
/// then flows through the single field mapping. A member name repeated with a
/// different value is refused by name rather than resolved to one of them. A
/// `COMMENT` member, which Space-Track writes, is read as a header comment.
pub fn parse_json(text: &str) -> Result<Omm, OmmError> {
    let value: JsonNode =
        serde_json::from_str(text).map_err(|e| OmmError::Field(format!("malformed JSON: {e}")))?;
    let object = match &value {
        JsonNode::Array(items) => match items.as_slice() {
            [] => return Err(OmmError::Field("empty JSON array".to_string())),
            [item] => item,
            _ => {
                return Err(OmmError::MultipleMessages { count: items.len() });
            }
        },
        JsonNode::Object(_) => &value,
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
fn omm_from_json_value(object: &JsonNode) -> Result<Omm, OmmError> {
    let JsonNode::Object(members) = object else {
        return Err(OmmError::Field("expected a JSON object".to_string()));
    };
    let pairs = members
        .iter()
        .map(|(key, value)| (key.clone(), value.scalar_text()))
        .collect();
    Omm::from_fields(OmmFields::from_flat_pairs(pairs)?)
}

/// A JSON value that keeps every member of an object in source order,
/// including a repeated name, which `serde_json::Value` would collapse to its
/// last occurrence.
#[derive(Debug, Clone, PartialEq)]
enum JsonNode {
    Null,
    Bool(bool),
    Number(serde_json::Number),
    String(String),
    Array(Vec<JsonNode>),
    Object(Vec<(String, JsonNode)>),
}

impl JsonNode {
    /// The text the shared field mapping consumes. Numbers use their canonical
    /// decimal form, strings pass through, null becomes empty, and a nested
    /// array or object is its JSON text.
    fn scalar_text(&self) -> String {
        match self {
            Self::Null => String::new(),
            Self::Bool(value) => value.to_string(),
            Self::Number(value) => value.to_string(),
            Self::String(value) => value.clone(),
            Self::Array(_) | Self::Object(_) => self.to_value().to_string(),
        }
    }

    fn to_value(&self) -> serde_json::Value {
        use serde_json::Value;
        match self {
            Self::Null => Value::Null,
            Self::Bool(value) => Value::Bool(*value),
            Self::Number(value) => Value::Number(value.clone()),
            Self::String(value) => Value::String(value.clone()),
            Self::Array(items) => Value::Array(items.iter().map(Self::to_value).collect()),
            Self::Object(members) => Value::Object(
                members
                    .iter()
                    .map(|(key, value)| (key.clone(), value.to_value()))
                    .collect(),
            ),
        }
    }
}

impl<'de> serde::Deserialize<'de> for JsonNode {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(JsonNodeVisitor)
    }
}

struct JsonNodeVisitor;

impl<'de> serde::de::Visitor<'de> for JsonNodeVisitor {
    type Value = JsonNode;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_unit<E>(self) -> Result<JsonNode, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonNode::Null)
    }

    fn visit_none<E>(self) -> Result<JsonNode, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonNode::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<JsonNode, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        serde::Deserialize::deserialize(deserializer)
    }

    fn visit_bool<E>(self, value: bool) -> Result<JsonNode, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonNode::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<JsonNode, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonNode::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<JsonNode, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonNode::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<JsonNode, E>
    where
        E: serde::de::Error,
    {
        Ok(serde_json::Number::from_f64(value).map_or(JsonNode::Null, JsonNode::Number))
    }

    fn visit_str<E>(self, value: &str) -> Result<JsonNode, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonNode::String(value.to_string()))
    }

    fn visit_string<E>(self, value: String) -> Result<JsonNode, E>
    where
        E: serde::de::Error,
    {
        Ok(JsonNode::String(value))
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<JsonNode, A::Error>
    where
        A: serde::de::SeqAccess<'de>,
    {
        let mut items = Vec::new();
        while let Some(item) = seq.next_element::<JsonNode>()? {
            items.push(item);
        }
        Ok(JsonNode::Array(items))
    }

    fn visit_map<A>(self, mut map: A) -> Result<JsonNode, A::Error>
    where
        A: serde::de::MapAccess<'de>,
    {
        let mut members = Vec::new();
        while let Some((key, value)) = map.next_entry::<String, JsonNode>()? {
            members.push((key, value));
        }
        Ok(JsonNode::Object(members))
    }
}

/// The result of parsing several OMM records, an XML document, a GP JSON
/// array or a GP CSV table: every OMM that parsed, plus each record that was
/// skipped and why.
#[derive(Debug, Clone, PartialEq)]
pub struct OmmArray {
    /// The successfully parsed OMMs, in input order.
    pub omms: Vec<Omm>,
    /// The records that were skipped, in input order: an XML message that
    /// failed to read, a JSON array element that is not an object or failed
    /// field validation, or a CSV data row whose field count differs from the
    /// header's or that failed field validation. Lets callers tell an empty input (both lists empty) apart
    /// from one whose every record was malformed, without aborting the whole
    /// parse on one bad record. No fabricated OMM is emitted in their place.
    pub skipped: Vec<OmmSkippedRecord>,
}

/// One XML message, GP JSON array element or GP CSV data row that was not
/// read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OmmSkippedRecord {
    /// Zero-based position of the record: the position of the `<omm>` message
    /// in the XML document, the JSON array index, or the position of the CSV
    /// data record among the records after the header.
    /// A blank CSV line is not a record, and a quoted CSV field may span
    /// lines, so a CSV index is not a line number.
    pub index: usize,
    /// Why the record was not read.
    pub reason: OmmError,
}

/// Parse a CelesTrak OMM JSON array into every contained [`Omm`].
///
/// CelesTrak GP queries return a JSON array of OMM objects; this reads all of
/// them (a lone object is accepted as a one-element array) through the same
/// field mapping [`parse_json`] uses. An individual array element that is not a
/// valid OMM object is reported in [`OmmArray::skipped`] with its index and
/// error rather than aborting the whole array. A malformed top-level document
/// (not valid JSON, or neither an object nor an array) is still an error.
pub fn parse_json_array(text: &str) -> Result<OmmArray, OmmError> {
    let value: JsonNode =
        serde_json::from_str(text).map_err(|e| OmmError::Field(format!("malformed JSON: {e}")))?;
    let items: &[JsonNode] = match &value {
        JsonNode::Array(items) => items.as_slice(),
        JsonNode::Object(_) => std::slice::from_ref(&value),
        _ => {
            return Err(OmmError::Field(
                "expected a JSON object or array".to_string(),
            ))
        }
    };

    let mut omms = Vec::with_capacity(items.len());
    let mut skipped = Vec::new();
    for (index, object) in items.iter().enumerate() {
        match omm_from_json_value(object) {
            Ok(omm) => omms.push(omm),
            Err(reason) => skipped.push(OmmSkippedRecord { index, reason }),
        }
    }
    Ok(OmmArray { omms, skipped })
}

/// Encode an [`Omm`] as a CCSDS/CelesTrak OMM JSON object.
///
/// Numeric element values are emitted as JSON numbers (round-tripping the exact
/// `f64`), strings as JSON strings, and the epoch as an ISO-8601 string, so
/// parsing the output reproduces the same [`Omm`]. Optional keywords are
/// emitted only when present, under their CCSDS keyword names; an empty
/// spacecraft block is emitted as `"MASS": null`.
///
/// A record whose only comment is a single header comment states it as a
/// `COMMENT` member, as Space-Track does. `CCSDS_OMM_VERS` is written only when
/// the record holds one.
///
/// A JSON string carries any text, including line breaks and surrounding
/// whitespace, and the JSON reader keeps it verbatim. What the reader would not
/// return unchanged is refused: any other comment, which GP JSON has no member
/// for, with [`OmmError::UnwritableText`] and [`TextIssue::CommentNotCarried`]
/// ([`encode_json_discarding_comments`] writes the record without it
/// instead); a character XML 1.0 cannot carry (every OMM reader refuses one),
/// an empty `CCSDS_OMM_VERS` or header comment (which reads back as absent)
/// and a `USER_DEFINED_*` parameter given more than once, with
/// [`OmmError::UnwritableText`]; a non-finite number, which JSON cannot hold
/// and was previously written as `null`, and an [`OmmEpoch`] that names no
/// instant the reader accepts under the message's `TIME_SYSTEM`, with
/// [`OmmError::InvalidField`].
pub fn encode_json(omm: &Omm) -> Result<String, OmmError> {
    json_object(omm).map(|object| serde_json::Value::Object(object).to_string())
}

/// Encode an [`Omm`] as a GP JSON object, discarding the comments GP JSON
/// cannot carry.
///
/// GP JSON carries one comment, a single header comment, as the `COMMENT`
/// member. This writes what [`encode_json`] writes for the record with every
/// other comment removed: further header comments and those of the metadata,
/// mean-elements, spacecraft, TLE-parameter, covariance and user-defined
/// blocks. A spacecraft-parameters block that held only comments is still
/// written, as `"MASS": null`. Everything else is checked and refused as
/// [`encode_json`] does.
pub fn encode_json_discarding_comments(omm: &Omm) -> Result<String, OmmError> {
    encode_json(&uncarried_comments_cleared(omm))
}

/// Build the GP JSON object of one [`Omm`] for [`encode_json`] and
/// [`encode_json_array`].
fn json_object(omm: &Omm) -> Result<serde_json::Map<String, serde_json::Value>, OmmError> {
    use serde_json::{Map, Number, Value};

    let num = |key: &'static str, value: f64| {
        Number::from_f64(value)
            .map(Value::Number)
            .ok_or(OmmError::InvalidField {
                field: key,
                kind: OmmInputErrorKind::NonFinite,
            })
    };
    let text = |key: &str, value: &str| {
        check_json_text(key, value).map(|()| Value::String(value.to_string()))
    };
    let opt = |key: &str, value: &Option<String>| match value {
        Some(value) => text(key, value),
        None => Ok(Value::Null),
    };

    let comment = carried_comment(omm)?;
    check_user_defined_names(omm)?;
    let mut map = Map::new();
    for (key, value) in [
        ("CCSDS_OMM_VERS", omm.ccsds_omm_vers.as_deref()),
        (COMMENT, comment),
    ] {
        let Some(value) = value else {
            continue;
        };
        // An empty member reads back as absent.
        if value.is_empty() {
            return Err(OmmError::UnwritableText {
                field: key.to_string(),
                value: String::new(),
                issue: TextIssue::Empty,
            });
        }
        map.insert(key.into(), text(key, value)?);
    }
    if let Some(classification) = &omm.classification {
        map.insert(
            "CLASSIFICATION".into(),
            text("CLASSIFICATION", classification)?,
        );
    }
    map.insert(
        "CREATION_DATE".into(),
        opt("CREATION_DATE", &omm.creation_date)?,
    );
    map.insert("ORIGINATOR".into(), opt("ORIGINATOR", &omm.originator)?);
    if let Some(message_id) = &omm.message_id {
        map.insert("MESSAGE_ID".into(), text("MESSAGE_ID", message_id)?);
    }
    map.insert("OBJECT_NAME".into(), opt("OBJECT_NAME", &omm.object_name)?);
    map.insert("OBJECT_ID".into(), opt("OBJECT_ID", &omm.object_id)?);
    map.insert("CENTER_NAME".into(), opt("CENTER_NAME", &omm.center_name)?);
    map.insert("REF_FRAME".into(), opt("REF_FRAME", &omm.ref_frame)?);
    if let Some(ref_frame_epoch) = &omm.ref_frame_epoch {
        map.insert(
            "REF_FRAME_EPOCH".into(),
            text("REF_FRAME_EPOCH", ref_frame_epoch)?,
        );
    }
    map.insert("TIME_SYSTEM".into(), opt("TIME_SYSTEM", &omm.time_system)?);
    map.insert(
        "MEAN_ELEMENT_THEORY".into(),
        opt("MEAN_ELEMENT_THEORY", &omm.mean_element_theory)?,
    );
    map.insert("EPOCH".into(), Value::String(epoch_text(omm)?));
    if let Some(value) = omm.semi_major_axis_km {
        map.insert("SEMI_MAJOR_AXIS".into(), num("SEMI_MAJOR_AXIS", value)?);
    }
    if let Some(value) = omm.mean_motion {
        map.insert("MEAN_MOTION".into(), num("MEAN_MOTION", value)?);
    }
    for (key, value) in [
        ("ECCENTRICITY", omm.eccentricity),
        ("INCLINATION", omm.inclination_deg),
        ("RA_OF_ASC_NODE", omm.ra_of_asc_node_deg),
        ("ARG_OF_PERICENTER", omm.arg_of_pericenter_deg),
        ("MEAN_ANOMALY", omm.mean_anomaly_deg),
    ] {
        map.insert(key.into(), num(key, value)?);
    }
    if let Some(gm) = omm.gm_km3_s2 {
        map.insert("GM".into(), num("GM", gm)?);
    }
    if let Some(spacecraft) = &omm.spacecraft {
        let values = spacecraft_values(spacecraft);
        for (key, value) in values {
            if let Some(value) = value {
                map.insert(key.into(), num(key, value)?);
            }
        }
        if values.iter().all(|(_, value)| value.is_none()) {
            map.insert("MASS".into(), Value::Null);
        }
    }
    if let Some(value) = omm.ephemeris_type {
        map.insert("EPHEMERIS_TYPE".into(), Value::Number(value.into()));
    }
    if let Some(value) = &omm.classification_type {
        map.insert(
            "CLASSIFICATION_TYPE".into(),
            text("CLASSIFICATION_TYPE", value)?,
        );
    }
    if let Some(value) = omm.norad_cat_id {
        map.insert("NORAD_CAT_ID".into(), Value::Number(value.into()));
    }
    if let Some(value) = omm.element_set_no {
        map.insert("ELEMENT_SET_NO".into(), Value::Number(value.into()));
    }
    if let Some(value) = omm.rev_at_epoch {
        map.insert("REV_AT_EPOCH".into(), Value::Number(value.into()));
    }
    for (key, value) in [
        ("BSTAR", omm.bstar),
        ("BTERM", omm.bterm_m2_kg),
        ("MEAN_MOTION_DOT", omm.mean_motion_dot),
        ("MEAN_MOTION_DDOT", omm.mean_motion_ddot),
        ("AGOM", omm.agom_m2_kg),
    ] {
        if let Some(value) = value {
            map.insert(key.into(), num(key, value)?);
        }
    }
    if let Some(covariance) = &omm.covariance {
        if let Some(frame) = &covariance.cov_ref_frame {
            map.insert(COV_REF_FRAME.into(), text(COV_REF_FRAME, frame)?);
        }
        for (key, value) in COVARIANCE6_KEYS.into_iter().zip(covariance.lower_triangle) {
            map.insert(key.into(), num(key, value)?);
        }
    }
    for parameter in &omm.user_defined {
        let key = format!("{USER_DEFINED_PREFIX}{}", parameter.parameter);
        check_json_text(&key, &parameter.parameter)?;
        let value = text(&key, &parameter.value)?;
        map.insert(key, value);
    }
    Ok(map)
}

/// Refuse JSON text that every OMM reader refuses: a character XML 1.0 cannot
/// carry.
fn check_json_text(field: &str, value: &str) -> Result<(), OmmError> {
    match xml::first_illegal_xml_1_0_char(value) {
        Some(_) => Err(OmmError::UnwritableText {
            field: field.to_string(),
            value: value.to_string(),
            issue: TextIssue::XmlIllegalCharacter,
        }),
        None => Ok(()),
    }
}

/// Encode a slice of [`Omm`] records as a GP JSON array.
///
/// Each array member is the same object produced by [`encode_json`], preserving
/// all scalar fields and optional metadata carried by [`Omm`]. A record
/// [`encode_json`] refuses, including one holding a comment, fails the whole
/// array with [`OmmError::InRecord`], naming its position.
pub fn encode_json_array(omms: &[Omm]) -> Result<String, OmmError> {
    let values = omms
        .iter()
        .enumerate()
        .map(|(index, omm)| {
            json_object(omm)
                .map(serde_json::Value::Object)
                .map_err(|error| OmmError::InRecord {
                    index,
                    source: Box::new(error),
                })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(serde_json::Value::Array(values).to_string())
}

/// Encode a slice of [`Omm`] records as a GP JSON array, discarding the
/// comments GP JSON cannot carry as [`encode_json_discarding_comments`] does
/// for one record.
pub fn encode_json_array_discarding_comments(omms: &[Omm]) -> Result<String, OmmError> {
    let cleared: Vec<Omm> = omms.iter().map(uncarried_comments_cleared).collect();
    encode_json_array(&cleared)
}

/// The one comment GP JSON and GP CSV carry: a single header comment, which
/// Space-Track writes as a `COMMENT` member or column. Any further header
/// comment, or a comment of another block, is refused, since the encodings
/// have no place for it.
fn carried_comment(omm: &Omm) -> Result<Option<&str>, OmmError> {
    let spacecraft = omm
        .spacecraft
        .as_ref()
        .map_or(&[] as &[String], |spacecraft| {
            spacecraft.comments.as_slice()
        });
    let covariance = omm
        .covariance
        .as_ref()
        .map_or(&[] as &[String], |covariance| {
            covariance.comments.as_slice()
        });
    let uncarried: [&[String]; 7] = [
        omm.comments.header.get(1..).unwrap_or_default(),
        &omm.comments.metadata,
        &omm.comments.mean_elements,
        spacecraft,
        &omm.comments.tle_parameters,
        covariance,
        &omm.comments.user_defined,
    ];
    match uncarried.into_iter().flat_map(|block| block.iter()).next() {
        Some(comment) => Err(OmmError::UnwritableText {
            field: COMMENT.to_string(),
            value: comment.clone(),
            issue: TextIssue::CommentNotCarried,
        }),
        None => Ok(omm.comments.header.first().map(String::as_str)),
    }
}

/// A copy of `omm` holding only the comment GP JSON and GP CSV carry, its
/// first header comment, with every block kept.
fn uncarried_comments_cleared(omm: &Omm) -> Omm {
    let mut omm = omm.clone();
    let header = omm.comments.header.first().cloned();
    omm.comments = OmmComments::default();
    omm.comments.header.extend(header);
    if let Some(spacecraft) = omm.spacecraft.as_mut() {
        spacecraft.comments.clear();
    }
    if let Some(covariance) = omm.covariance.as_mut() {
        covariance.comments.clear();
    }
    omm
}

// ── CSV ──────────────────────────────────────────────────────────────

/// Parse a GP CSV holding one record into an [`Omm`].
///
/// The input must begin with a header row of OMM keyword columns followed by
/// one data record, which is read as [`parse_csv_array`] reads each record; a
/// record it would skip is refused with the reason. Several records are
/// refused with [`OmmError::MultipleMessages`] rather than read for the first
/// valid one; [`parse_csv_array`] reads all of them.
pub fn parse_csv(text: &str) -> Result<Omm, OmmError> {
    let mut parsed = parse_csv_array(text)?;
    match parsed.omms.len() + parsed.skipped.len() {
        0 => Err(OmmError::Field("empty GP CSV".to_string())),
        1 => match (parsed.omms.pop(), parsed.skipped.pop()) {
            (Some(omm), _) => Ok(omm),
            (None, Some(skipped)) => Err(skipped.reason),
            (None, None) => Err(OmmError::Field("empty GP CSV".to_string())),
        },
        count => Err(OmmError::MultipleMessages { count }),
    }
}

/// Parse all valid GP CSV records into [`Omm`] values.
///
/// Header names are the OMM keyword names used by GP CSV and JSON. Each record
/// is mapped onto the same `(key, value)` field set used by KVN, XML, and JSON,
/// so all encodings share one [`Omm`] intermediate representation. An empty
/// cell is an absent value, since every row of a table carries every column. A
/// bad data row, including one whose repeated column names carry different
/// values, is reported in [`OmmArray::skipped`] rather than aborting the whole
/// file.
pub fn parse_csv_array(text: &str) -> Result<OmmArray, OmmError> {
    let records = parse_csv_records(text)?;
    let Some((header, rows)) = records.split_first() else {
        return Err(OmmError::Field("missing GP CSV header".to_string()));
    };
    let header: Vec<String> = header.iter().map(|key| key.trim().to_string()).collect();
    if header.is_empty() || header.iter().all(String::is_empty) {
        return Err(OmmError::Field("missing GP CSV header".to_string()));
    }

    let mut omms = Vec::with_capacity(rows.len());
    let mut skipped = Vec::new();
    for (index, row) in rows.iter().enumerate() {
        if row.len() != header.len() {
            skipped.push(OmmSkippedRecord {
                index,
                reason: OmmError::CsvColumnCount {
                    found: row.len(),
                    expected: header.len(),
                },
            });
            continue;
        }
        let pairs = header
            .iter()
            .zip(row.iter())
            .map(|(key, value)| (key.clone(), value.trim().to_string()))
            .filter(|(_, value)| !value.is_empty())
            .collect();
        match OmmFields::from_flat_pairs(pairs).and_then(Omm::from_fields) {
            Ok(omm) => omms.push(omm),
            Err(reason) => skipped.push(OmmSkippedRecord { index, reason }),
        }
    }

    Ok(OmmArray { omms, skipped })
}

/// Encode [`Omm`] records as GP CSV.
///
/// The header starts with the common GP CSV/JSON keyword set, followed by a
/// column for every other keyword any record holds (the version, header items,
/// metadata labels, `SEMI_MAJOR_AXIS`, the SGP4-XP terms, the optional blocks
/// and each `USER_DEFINED_*` parameter), so a record reads back as written; a
/// record without a value leaves its cell empty. A record whose only comment is
/// a single header comment states it in a `COMMENT` column, as Space-Track
/// does. The `USER_DEFINED_*` columns follow an order that keeps every
/// record's own parameter order, which the reader restores. Numeric values use
/// their shortest round-tripping decimal form, and fields containing CSV
/// delimiters are quoted with standard double-quote escaping.
///
/// A record [`parse_csv_array`] would not return unchanged is refused, naming
/// its position with [`OmmError::InRecord`]:
///
/// - any other comment, which GP CSV has no column for, with
///   [`OmmError::UnwritableText`] and [`TextIssue::CommentNotCarried`];
///   [`encode_csv_discarding_comments`] writes the record without it
///   instead;
/// - a spacecraft-parameters block that holds no value, which an empty cell
///   cannot state, with [`OmmError::CsvEmptyBlock`];
/// - text with surrounding whitespace, which the reader trims from every cell
///   and header name, a character XML 1.0 cannot carry, which every OMM reader
///   refuses, an empty `CCSDS_OMM_VERS`, header comment or `USER_DEFINED_*`
///   value, which reads back as absent, and a `USER_DEFINED_*` parameter given
///   more than once, with [`OmmError::UnwritableText`];
/// - a record giving two `USER_DEFINED_*` parameters in the order opposite to
///   an earlier record, which one column order cannot state for both, with
///   [`OmmError::CsvColumnOrder`];
/// - a non-finite number, which was written as `NaN` or `inf` and read back as
///   a skipped record, and an [`OmmEpoch`] that names no instant the reader
///   accepts under the record's `TIME_SYSTEM`, with [`OmmError::InvalidField`].
///
/// A blank optional text value is written as an empty cell and reads back
/// absent, as in the other encodings.
pub fn encode_csv(omms: &[Omm]) -> Result<String, OmmError> {
    for (index, omm) in omms.iter().enumerate() {
        check_csv_record(omm).map_err(|error| OmmError::InRecord {
            index,
            source: Box::new(error),
        })?;
    }
    let parameters = user_defined_columns(omms)?;
    Ok(write_csv_records(omms, &parameters))
}

/// Encode [`Omm`] records as GP CSV, discarding the comments GP CSV cannot
/// carry.
///
/// GP CSV carries one comment, a single header comment, in the `COMMENT`
/// column. This writes what [`encode_csv`] writes for each record with every
/// other comment removed: further header comments and those of the metadata,
/// mean-elements, spacecraft, TLE-parameter, covariance and user-defined
/// blocks. A spacecraft-parameters block that holds comments and no value is
/// removed with them. Everything else is checked and refused as
/// [`encode_csv`] does, including a spacecraft-parameters block that holds
/// neither a value nor a comment.
pub fn encode_csv_discarding_comments(omms: &[Omm]) -> Result<String, OmmError> {
    let stripped: Vec<Omm> = omms.iter().map(without_comments).collect();
    encode_csv(&stripped)
}

/// A copy of `omm` holding only the comment GP CSV carries. A spacecraft block
/// that held comments and no value goes with its comments, since GP CSV
/// cannot state it empty.
fn without_comments(omm: &Omm) -> Omm {
    let held_only_comments = omm.spacecraft.as_ref().is_some_and(|spacecraft| {
        !spacecraft.comments.is_empty()
            && spacecraft_values(spacecraft)
                .iter()
                .all(|(_, value)| value.is_none())
    });
    let mut omm = uncarried_comments_cleared(omm);
    if held_only_comments {
        omm.spacecraft = None;
    }
    omm
}

/// Refuse a record that [`parse_csv_array`] would not return unchanged.
fn check_csv_record(omm: &Omm) -> Result<(), OmmError> {
    let comment = carried_comment(omm)?;
    if let Some(spacecraft) = &omm.spacecraft {
        if spacecraft_values(spacecraft)
            .iter()
            .all(|(_, value)| value.is_none())
        {
            return Err(OmmError::CsvEmptyBlock("spacecraft parameters"));
        }
    }

    check_user_defined_names(omm)?;
    epoch_text(omm)?;
    for (key, value) in [
        ("CCSDS_OMM_VERS", omm.ccsds_omm_vers.as_deref()),
        (COMMENT, comment),
    ] {
        let Some(value) = value else {
            continue;
        };
        // An empty cell reads back as absent.
        if value.is_empty() {
            return Err(OmmError::UnwritableText {
                field: key.to_string(),
                value: String::new(),
                issue: TextIssue::Empty,
            });
        }
        check_csv_text(key, value)?;
    }
    for (key, value) in csv_text_values(omm) {
        if let Some(value) = value {
            check_csv_text(key, value)?;
        }
    }
    for (key, value) in csv_number_values(omm) {
        check_finite(key, value)?;
    }
    for parameter in &omm.user_defined {
        let key = format!("{USER_DEFINED_PREFIX}{}", parameter.parameter);
        // The name ends a header cell, whose trailing whitespace the reader
        // trims; whitespace after the prefix is inside the cell and kept.
        let name_issue = if parameter.parameter.trim_end() != parameter.parameter.as_str() {
            Some(TextIssue::SurroundingWhitespace)
        } else if xml::first_illegal_xml_1_0_char(&parameter.parameter).is_some() {
            Some(TextIssue::XmlIllegalCharacter)
        } else {
            None
        };
        if let Some(issue) = name_issue {
            return Err(OmmError::UnwritableText {
                field: key,
                value: parameter.parameter.clone(),
                issue,
            });
        }
        if parameter.value.is_empty() {
            // An empty cell reads back as no parameter at all.
            return Err(OmmError::UnwritableText {
                field: key,
                value: String::new(),
                issue: TextIssue::Empty,
            });
        }
        check_csv_text(&key, &parameter.value)?;
    }
    Ok(())
}

/// Refuse CSV cell text the reader would not return unchanged: surrounding
/// whitespace, which it trims, and a character XML 1.0 cannot carry, which
/// every OMM reader refuses. Delimiters, quotes and line breaks are quoted.
fn check_csv_text(field: &str, value: &str) -> Result<(), OmmError> {
    let issue = if value.trim() != value {
        Some(TextIssue::SurroundingWhitespace)
    } else if xml::first_illegal_xml_1_0_char(value).is_some() {
        Some(TextIssue::XmlIllegalCharacter)
    } else {
        None
    };
    match issue {
        Some(issue) => Err(OmmError::UnwritableText {
            field: field.to_string(),
            value: value.to_string(),
            issue,
        }),
        None => Ok(()),
    }
}

/// The text values GP CSV writes for a record, other than the version, the
/// epoch and the user-defined parameters.
fn csv_text_values(omm: &Omm) -> [(&'static str, Option<&str>); 13] {
    [
        ("CLASSIFICATION", omm.classification.as_deref()),
        ("CREATION_DATE", omm.creation_date.as_deref()),
        ("ORIGINATOR", omm.originator.as_deref()),
        ("MESSAGE_ID", omm.message_id.as_deref()),
        ("OBJECT_NAME", omm.object_name.as_deref()),
        ("OBJECT_ID", omm.object_id.as_deref()),
        ("CENTER_NAME", omm.center_name.as_deref()),
        ("REF_FRAME", omm.ref_frame.as_deref()),
        ("REF_FRAME_EPOCH", omm.ref_frame_epoch.as_deref()),
        ("TIME_SYSTEM", omm.time_system.as_deref()),
        ("MEAN_ELEMENT_THEORY", omm.mean_element_theory.as_deref()),
        ("CLASSIFICATION_TYPE", omm.classification_type.as_deref()),
        (
            COV_REF_FRAME,
            omm.covariance
                .as_ref()
                .and_then(|covariance| covariance.cov_ref_frame.as_deref()),
        ),
    ]
}

/// The floating-point values GP CSV writes for a record, with their keywords.
fn csv_number_values(omm: &Omm) -> Vec<(&'static str, f64)> {
    let mut values = vec![
        ("ECCENTRICITY", omm.eccentricity),
        ("INCLINATION", omm.inclination_deg),
        ("RA_OF_ASC_NODE", omm.ra_of_asc_node_deg),
        ("ARG_OF_PERICENTER", omm.arg_of_pericenter_deg),
        ("MEAN_ANOMALY", omm.mean_anomaly_deg),
    ];
    let optional = [
        ("SEMI_MAJOR_AXIS", omm.semi_major_axis_km),
        ("MEAN_MOTION", omm.mean_motion),
        ("GM", omm.gm_km3_s2),
        ("BSTAR", omm.bstar),
        ("BTERM", omm.bterm_m2_kg),
        ("MEAN_MOTION_DOT", omm.mean_motion_dot),
        ("MEAN_MOTION_DDOT", omm.mean_motion_ddot),
        ("AGOM", omm.agom_m2_kg),
    ];
    let spacecraft = omm
        .spacecraft
        .as_ref()
        .map(spacecraft_values)
        .unwrap_or([("MASS", None); 5]);
    for (key, value) in optional.into_iter().chain(spacecraft) {
        if let Some(value) = value {
            values.push((key, value));
        }
    }
    if let Some(covariance) = &omm.covariance {
        values.extend(COVARIANCE6_KEYS.into_iter().zip(covariance.lower_triangle));
    }
    values
}

/// The `USER_DEFINED_*` parameter names in a column order that keeps every
/// record's own parameter order, since the reader lists a record's parameters
/// in column order. Among orders that do, a parameter comes as early as its
/// first appearance allows. A record giving two parameters in the order
/// opposite to earlier records is refused, naming it.
fn user_defined_columns(omms: &[Omm]) -> Result<Vec<String>, OmmError> {
    let mut names: Vec<&str> = Vec::new();
    // `after[i]` lists the parameters some record gives directly after `i`.
    let mut after: Vec<Vec<usize>> = Vec::new();
    for (index, omm) in omms.iter().enumerate() {
        let mut previous: Option<usize> = None;
        for parameter in &omm.user_defined {
            let name = parameter.parameter.as_str();
            let node = match names.iter().position(|known| *known == name) {
                Some(node) => node,
                None => {
                    names.push(name);
                    after.push(Vec::new());
                    names.len() - 1
                }
            };
            if let Some(before) = previous {
                if reaches(&after, node, before) {
                    return Err(OmmError::InRecord {
                        index,
                        source: Box::new(OmmError::CsvColumnOrder {
                            first: names[before].to_string(),
                            second: name.to_string(),
                        }),
                    });
                }
                if let Some(successors) = after.get_mut(before) {
                    if !successors.contains(&node) {
                        successors.push(node);
                    }
                }
            }
            previous = Some(node);
        }
    }
    // Take the earliest-seen parameter none still waiting precedes.
    let mut waiting = vec![0usize; names.len()];
    for successors in &after {
        for &node in successors {
            waiting[node] += 1;
        }
    }
    let mut placed = vec![false; names.len()];
    let mut order = Vec::with_capacity(names.len());
    while order.len() < names.len() {
        let Some(next) = (0..names.len()).find(|&node| !placed[node] && waiting[node] == 0) else {
            break;
        };
        placed[next] = true;
        order.push(format!("{USER_DEFINED_PREFIX}{}", names[next]));
        for &node in &after[next] {
            waiting[node] -= 1;
        }
    }
    Ok(order)
}

/// Whether `to` follows `from` in the order the records give so far.
fn reaches(after: &[Vec<usize>], from: usize, to: usize) -> bool {
    let mut stack = vec![from];
    let mut seen = vec![false; after.len()];
    while let Some(node) = stack.pop() {
        if node == to {
            return true;
        }
        if std::mem::replace(&mut seen[node], true) {
            continue;
        }
        stack.extend(after[node].iter().copied());
    }
    false
}

/// Write records [`check_csv_record`] accepted, with the `USER_DEFINED_*`
/// columns in the order [`user_defined_columns`] gives.
fn write_csv_records(omms: &[Omm], parameters: &[String]) -> String {
    let mut columns: Vec<String> = GP_CSV_FIELDS.iter().map(|key| key.to_string()).collect();
    for key in ["CCSDS_OMM_VERS", COMMENT] {
        if omms
            .iter()
            .any(|omm| !omm_csv_field_value(omm, key).is_empty())
        {
            columns.push(key.to_string());
        }
    }
    for key in CSV_EXTRA_FIELDS.iter().chain(COVARIANCE6_KEYS.iter()) {
        if omms
            .iter()
            .any(|omm| !omm_csv_field_value(omm, key).is_empty())
        {
            columns.push((*key).to_string());
        }
    }
    columns.extend(parameters.iter().cloned());

    let mut out = String::new();
    write_csv_record(&mut out, columns.iter());
    for omm in omms {
        out.push('\n');
        write_csv_record(
            &mut out,
            columns
                .iter()
                .map(|key| match key.strip_prefix(USER_DEFINED_PREFIX) {
                    Some(parameter) => omm
                        .user_defined
                        .iter()
                        .find(|entry| entry.parameter == parameter)
                        .map(|entry| entry.value.clone())
                        .unwrap_or_default(),
                    None => omm_csv_field_value(omm, key),
                }),
        );
    }
    out
}

fn parse_csv_records(text: &str) -> Result<Vec<Vec<String>>, OmmError> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut chars = text.chars().peekable();
    let mut in_quotes = false;
    let mut quoted_field = false;

    while let Some(ch) = chars.next() {
        if in_quotes {
            match ch {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => in_quotes = false,
                _ => field.push(ch),
            }
            continue;
        }

        match ch {
            '"' if field.is_empty() && !quoted_field => {
                in_quotes = true;
                quoted_field = true;
            }
            ',' => {
                record.push(std::mem::take(&mut field));
                quoted_field = false;
            }
            '\n' => {
                record.push(std::mem::take(&mut field));
                push_csv_record(&mut records, &mut record);
                quoted_field = false;
            }
            '\r' if chars.peek() == Some(&'\n') => {}
            '\r' => {
                record.push(std::mem::take(&mut field));
                push_csv_record(&mut records, &mut record);
                quoted_field = false;
            }
            _ => field.push(ch),
        }
    }

    if in_quotes {
        return Err(OmmError::Field(
            "malformed GP CSV: unclosed quoted field".to_string(),
        ));
    }
    if !field.is_empty() || !record.is_empty() || quoted_field {
        record.push(field);
        push_csv_record(&mut records, &mut record);
    }

    Ok(records)
}

fn push_csv_record(records: &mut Vec<Vec<String>>, record: &mut Vec<String>) {
    if record.len() == 1 && record[0].is_empty() {
        record.clear();
        return;
    }
    records.push(std::mem::take(record));
}

fn write_csv_record<I, S>(out: &mut String, fields: I)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    for (index, field) in fields.into_iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        write_csv_field(out, field.as_ref());
    }
}

fn write_csv_field(out: &mut String, field: &str) {
    if field.contains([',', '"', '\n', '\r']) {
        out.push('"');
        for ch in field.chars() {
            if ch == '"' {
                out.push('"');
            }
            out.push(ch);
        }
        out.push('"');
    } else {
        out.push_str(field);
    }
}

/// Keywords outside [`GP_CSV_FIELDS`] that [`encode_csv`] writes when a record
/// holds them, in table 4-1 to 4-3 order; the covariance keywords follow.
const CSV_EXTRA_FIELDS: &[&str] = &[
    "CLASSIFICATION",
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "REF_FRAME_EPOCH",
    "TIME_SYSTEM",
    "MEAN_ELEMENT_THEORY",
    "SEMI_MAJOR_AXIS",
    "GM",
    "MASS",
    "SOLAR_RAD_AREA",
    "SOLAR_RAD_COEFF",
    "DRAG_AREA",
    "DRAG_COEFF",
    "BTERM",
    "AGOM",
    "COV_REF_FRAME",
];

fn omm_csv_field_value(omm: &Omm, key: &str) -> String {
    let text = |value: &Option<String>| value.clone().unwrap_or_default();
    let num = |value: Option<f64>| value.map(fmt_num).unwrap_or_default();
    match key {
        "CCSDS_OMM_VERS" => text(&omm.ccsds_omm_vers),
        COMMENT => omm.comments.header.first().cloned().unwrap_or_default(),
        "CLASSIFICATION" => text(&omm.classification),
        "CREATION_DATE" => text(&omm.creation_date),
        "ORIGINATOR" => text(&omm.originator),
        "MESSAGE_ID" => text(&omm.message_id),
        "OBJECT_NAME" => text(&omm.object_name),
        "OBJECT_ID" => text(&omm.object_id),
        "CENTER_NAME" => text(&omm.center_name),
        "REF_FRAME" => text(&omm.ref_frame),
        "REF_FRAME_EPOCH" => text(&omm.ref_frame_epoch),
        "TIME_SYSTEM" => text(&omm.time_system),
        "MEAN_ELEMENT_THEORY" => text(&omm.mean_element_theory),
        "EPOCH" => omm.epoch.to_iso8601(),
        "SEMI_MAJOR_AXIS" => num(omm.semi_major_axis_km),
        "MEAN_MOTION" => num(omm.mean_motion),
        "ECCENTRICITY" => fmt_num(omm.eccentricity),
        "INCLINATION" => fmt_num(omm.inclination_deg),
        "RA_OF_ASC_NODE" => fmt_num(omm.ra_of_asc_node_deg),
        "ARG_OF_PERICENTER" => fmt_num(omm.arg_of_pericenter_deg),
        "MEAN_ANOMALY" => fmt_num(omm.mean_anomaly_deg),
        "GM" => num(omm.gm_km3_s2),
        "MASS" => num(omm.spacecraft.as_ref().and_then(|s| s.mass_kg)),
        "SOLAR_RAD_AREA" => num(omm.spacecraft.as_ref().and_then(|s| s.solar_rad_area_m2)),
        "SOLAR_RAD_COEFF" => num(omm.spacecraft.as_ref().and_then(|s| s.solar_rad_coeff)),
        "DRAG_AREA" => num(omm.spacecraft.as_ref().and_then(|s| s.drag_area_m2)),
        "DRAG_COEFF" => num(omm.spacecraft.as_ref().and_then(|s| s.drag_coeff)),
        "EPHEMERIS_TYPE" => omm
            .ephemeris_type
            .map(|v| v.to_string())
            .unwrap_or_default(),
        "CLASSIFICATION_TYPE" => text(&omm.classification_type),
        "NORAD_CAT_ID" => omm.norad_cat_id.map(|v| v.to_string()).unwrap_or_default(),
        "ELEMENT_SET_NO" => omm
            .element_set_no
            .map(|v| v.to_string())
            .unwrap_or_default(),
        "REV_AT_EPOCH" => omm.rev_at_epoch.map(|v| v.to_string()).unwrap_or_default(),
        "BSTAR" => num(omm.bstar),
        "BTERM" => num(omm.bterm_m2_kg),
        "MEAN_MOTION_DOT" => num(omm.mean_motion_dot),
        "MEAN_MOTION_DDOT" => num(omm.mean_motion_ddot),
        "AGOM" => num(omm.agom_m2_kg),
        "COV_REF_FRAME" => omm
            .covariance
            .as_ref()
            .map(|covariance| text(&covariance.cov_ref_frame))
            .unwrap_or_default(),
        other => match COVARIANCE6_KEYS
            .iter()
            .position(|candidate| *candidate == other)
        {
            Some(index) => omm
                .covariance
                .as_ref()
                .map(|covariance| fmt_num(covariance.lower_triangle[index]))
                .unwrap_or_default(),
            None => String::new(),
        },
    }
}

// ── Encoding auto-detect ─────────────────────────────────────────────

/// Parse an OMM in any supported encoding, detecting it from the leading
/// non-whitespace character and first content line: `<` is XML, `{` or `[` is
/// JSON, a comma-separated header is CSV, and anything else is KVN. A document
/// holding several records is refused with [`OmmError::MultipleMessages`], as
/// each single-record reader refuses it.
pub fn parse(text: &str) -> Result<Omm, OmmError> {
    match text.trim_start().chars().next() {
        Some('<') => parse_xml(text),
        Some('{') | Some('[') => parse_json_detected(text),
        _ if looks_like_csv(text) => parse_csv(text),
        _ => parse_kvn(text),
    }
}

/// Parse a single CCSDS `EPOCH` string field to the canonical [`OmmEpoch`].
///
/// The accepted form is `YYYY-MM-DDThh:mm:ss[.f...][Z]` with up to 15
/// fractional-second digits (whole microseconds plus a femtosecond remainder),
/// interpreted under the UTC-like civil-second policy (the OMM default when no
/// `TIME_SYSTEM` is declared, matching how a CelesTrak GP `EPOCH` is read).
/// This is the single public entry point a thin binding (for example the
/// Elixir constellation NIF) delegates to instead of hand-rolling the split;
/// it wraps the same shared `NdmEpoch` parser the full OMM decode uses, so it
/// produces byte-identical components.
pub fn parse_epoch(text: &str) -> Result<OmmEpoch, OmmError> {
    OmmEpoch::parse(text, validate::CivilSecondPolicy::UtcLike)
}

fn parse_json_detected(text: &str) -> Result<Omm, OmmError> {
    parse_json(text)
}

fn looks_like_csv(text: &str) -> bool {
    text.lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .is_some_and(|line| line.contains(',') && !line.contains('='))
}

// ── Field mapping (shared by every encoding) ─────────────────────────

/// The logical groups of OMM keywords: the header (table 4-1), the metadata
/// (table 4-2), and the five data blocks of table 4-3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OmmBlock {
    Header,
    Metadata,
    MeanElements,
    Spacecraft,
    TleParameters,
    Covariance,
    UserDefined,
}

impl OmmBlock {
    /// The block a keyword belongs to, or `None` for a keyword the tables do
    /// not define. `COMMENT` belongs to every block and is handled apart.
    fn of_keyword(key: &str) -> Option<Self> {
        if HEADER_KEYS.contains(&key) {
            Some(Self::Header)
        } else if METADATA_KEYS.contains(&key) {
            Some(Self::Metadata)
        } else if MEAN_ELEMENT_KEYS.contains(&key) {
            Some(Self::MeanElements)
        } else if SPACECRAFT_KEYS.contains(&key) {
            Some(Self::Spacecraft)
        } else if TLE_PARAMETER_KEYS.contains(&key) {
            Some(Self::TleParameters)
        } else if key == COV_REF_FRAME || COVARIANCE6_KEYS.contains(&key) {
            Some(Self::Covariance)
        } else if key.starts_with(USER_DEFINED_PREFIX) {
            Some(Self::UserDefined)
        } else {
            None
        }
    }
}

/// The unit spellings CCSDS 502.0-B-3 tables 4-3 and 8-4 give a numeric
/// keyword: an empty list for a dimensionless one, `None` for a text keyword.
/// `BSTAR` carries two documented spellings, `1/[Earth radii]` in table 4-3 and
/// `1/ER` in table 8-4. Table 4-3 gives `BTERM` and `AGOM` in m^2/kg, written
/// `m**2/kg` in the unit notation of 1.5.
fn omm_unit(key: &str) -> Option<&'static [&'static str]> {
    const DIMENSIONLESS: &[&str] = &[];
    const DEG: &[&str] = &["deg"];
    const REV_PER_DAY: &[&str] = &["rev/day"];
    const REV_PER_DAY2: &[&str] = &["rev/day**2"];
    const REV_PER_DAY3: &[&str] = &["rev/day**3"];
    const GM: &[&str] = &["km**3/s**2"];
    const KG: &[&str] = &["kg"];
    const M2: &[&str] = &["m**2"];
    const INVERSE_EARTH_RADII: &[&str] = &["1/[Earth radii]", "1/ER"];
    const KM: &[&str] = &["km"];
    const M2_PER_KG: &[&str] = &["m**2/kg"];
    match key {
        "MEAN_MOTION" => Some(REV_PER_DAY),
        "SEMI_MAJOR_AXIS" => Some(KM),
        "BTERM" | "AGOM" => Some(M2_PER_KG),
        "ECCENTRICITY" | "SOLAR_RAD_COEFF" | "DRAG_COEFF" | "EPHEMERIS_TYPE" | "NORAD_CAT_ID"
        | "ELEMENT_SET_NO" | "REV_AT_EPOCH" => Some(DIMENSIONLESS),
        "INCLINATION" | "RA_OF_ASC_NODE" | "ARG_OF_PERICENTER" | "MEAN_ANOMALY" => Some(DEG),
        "GM" => Some(GM),
        "MASS" => Some(KG),
        "SOLAR_RAD_AREA" | "DRAG_AREA" => Some(M2),
        "BSTAR" => Some(INVERSE_EARTH_RADII),
        "MEAN_MOTION_DOT" => Some(REV_PER_DAY2),
        "MEAN_MOTION_DDOT" => Some(REV_PER_DAY3),
        other => covariance6_unit(other),
    }
}

fn map_unit_mismatch(key: &str, mismatch: UnitMismatch) -> OmmError {
    OmmError::UnitMismatch {
        field: key.to_string(),
        unit: mismatch.unit,
        expected: mismatch.expected,
    }
}

/// Decoded OMM content before the typed mapping: single-valued keywords with
/// any unit already checked and removed, block comments, and user-defined
/// parameters, all in source order.
#[derive(Debug, Default)]
struct OmmFields {
    pairs: Vec<(String, String)>,
    comments: OmmComments,
    spacecraft_comments: Vec<String>,
    covariance_comments: Vec<String>,
    /// The XML `spacecraftParameters` element is present.
    spacecraft_block: bool,
    /// The XML `covarianceMatrix` element is present.
    covariance_block: bool,
    user_defined: Vec<OmmUserDefined>,
}

impl OmmFields {
    /// Map flat GP JSON or CSV members. Names that are not OMM keywords, such
    /// as product-specific extras and `COMMENT`, are not read.
    fn from_flat_pairs(pairs: Vec<(String, String)>) -> Result<Self, OmmError> {
        let mut fields = Self::default();
        for (key, value) in pairs {
            if let Some(parameter) = key.strip_prefix(USER_DEFINED_PREFIX) {
                fields.push_user_defined(parameter, &value)?;
            } else if key == COMMENT {
                fields.push_flat_comment(value)?;
            } else if OmmBlock::of_keyword(&key).is_some() {
                fields.pairs.push((key, value));
            }
        }
        Ok(fields)
    }

    /// Record a GP JSON `COMMENT` member or GP CSV `COMMENT` column, which
    /// Space-Track writes (`GENERATED VIA SPACE-TRACK.ORG API`, the text its
    /// KVN gives as a header comment), as the header comment. An empty value
    /// is absent, as for every other member; a repeat is read once when equal
    /// and refused by name when it differs, as a repeated keyword is.
    fn push_flat_comment(&mut self, value: String) -> Result<(), OmmError> {
        if value.is_empty() {
            return Ok(());
        }
        match self.comments.header.first() {
            None => self.comments.header.push(value),
            Some(first) if *first == value => {}
            Some(first) => {
                return Err(OmmError::DuplicateField {
                    field: COMMENT.to_string(),
                    first: first.clone(),
                    second: value,
                })
            }
        }
        Ok(())
    }

    fn push_comment(&mut self, block: OmmBlock, comment: String) {
        match block {
            OmmBlock::Header => self.comments.header.push(comment),
            OmmBlock::Metadata => self.comments.metadata.push(comment),
            OmmBlock::MeanElements => self.comments.mean_elements.push(comment),
            OmmBlock::Spacecraft => self.spacecraft_comments.push(comment),
            OmmBlock::TleParameters => self.comments.tle_parameters.push(comment),
            OmmBlock::Covariance => self.covariance_comments.push(comment),
            OmmBlock::UserDefined => self.comments.user_defined.push(comment),
        }
    }

    /// Record a KVN assignment. A numeric keyword's trailing `[unit]` is checked
    /// against table 4-3 and removed; a text keyword's value is kept verbatim.
    fn push_kvn_value(&mut self, key: &str, value: &str) -> Result<(), OmmError> {
        if let Some(parameter) = key.strip_prefix(USER_DEFINED_PREFIX) {
            return self.push_user_defined(parameter, value);
        }
        let value = match omm_unit(key) {
            Some(allowed) => {
                let (number, unit) = split_unit(value);
                check_unit(unit, allowed).map_err(|mismatch| map_unit_mismatch(key, mismatch))?;
                number
            }
            None => value,
        };
        self.pairs.push((key.to_string(), value.to_string()));
        Ok(())
    }

    /// Record an XML keyword element. A `units` attribute must match the table
    /// 4-3 unit (502.0-B-3 8.9.11); a text keyword takes none.
    fn push_xml_value(
        &mut self,
        key: &str,
        text: &str,
        units: Option<&str>,
    ) -> Result<(), OmmError> {
        if let Some(unit) = units {
            const NO_UNIT: &[&str] = &[];
            check_unit(Some(unit.trim()), omm_unit(key).unwrap_or(NO_UNIT))
                .map_err(|mismatch| map_unit_mismatch(key, mismatch))?;
        }
        self.pairs.push((key.to_string(), text.to_string()));
        Ok(())
    }

    /// Record a user-defined parameter. An exact repeat is read once; a repeat
    /// with a different value is refused by name.
    fn push_user_defined(&mut self, parameter: &str, value: &str) -> Result<(), OmmError> {
        if let Some(existing) = self
            .user_defined
            .iter()
            .find(|existing| existing.parameter == parameter)
        {
            if existing.value == value {
                return Ok(());
            }
            return Err(OmmError::DuplicateField {
                field: format!("{USER_DEFINED_PREFIX}{parameter}"),
                first: existing.value.clone(),
                second: value.to_string(),
            });
        }
        self.user_defined.push(OmmUserDefined {
            parameter: parameter.to_string(),
            value: value.to_string(),
        });
        Ok(())
    }
}

impl Omm {
    /// Build an [`Omm`] from decoded fields. This is the single place the CCSDS
    /// keyword names map onto the canonical container for every encoding.
    ///
    /// Only the table 4-3 items every theory needs are required: `EPOCH`, one
    /// of `SEMI_MAJOR_AXIS` and `MEAN_MOTION`, and the four angles with the
    /// eccentricity. The items that depend on the mean-element theory are
    /// kept as present; [`Omm::to_element_set`] requires those SGP4 reads.
    fn from_fields(fields: OmmFields) -> Result<Omm, OmmError> {
        let OmmFields {
            pairs,
            comments,
            spacecraft_comments,
            covariance_comments,
            spacecraft_block,
            covariance_block,
            user_defined,
        } = fields;
        let map = FieldMap::from_pairs(pairs);
        if let Some(conflict) = map.first_conflict(|_| true) {
            return Err(OmmError::DuplicateField {
                field: conflict.key,
                first: conflict.first,
                second: conflict.second,
            });
        }
        validate_texts(&comments.header, COMMENT)?;
        validate_texts(&comments.metadata, COMMENT)?;
        validate_texts(&comments.mean_elements, COMMENT)?;
        validate_texts(&comments.tle_parameters, COMMENT)?;
        validate_texts(&comments.user_defined, COMMENT)?;
        validate_texts(&spacecraft_comments, COMMENT)?;
        validate_texts(&covariance_comments, COMMENT)?;
        for parameter in &user_defined {
            xml_text_value(&parameter.parameter, "USER_DEFINED")?;
            xml_text_value(&parameter.value, "USER_DEFINED")?;
        }

        let get = |key: &str| map.get(key);

        let time_system = xml_text(get("TIME_SYSTEM"), "TIME_SYSTEM")?;
        let epoch = OmmEpoch::parse(
            get("EPOCH").ok_or(OmmError::MissingField("EPOCH"))?,
            omm_civil_second_policy(time_system.as_deref()),
        )?;
        let header = parse_header(&map)?;
        let metadata = parse_metadata(&map, time_system)?;
        let mean_elements = parse_mean_elements(&map, epoch)?;
        let tle_parameters = parse_tle_parameters(&map)?;

        let spacecraft = if spacecraft_block || keyword_occurs(&map, SPACECRAFT_KEYS) {
            Some(OmmSpacecraft {
                comments: spacecraft_comments,
                mass_kg: opt_num(get("MASS"), "MASS")?,
                solar_rad_area_m2: opt_num(get("SOLAR_RAD_AREA"), "SOLAR_RAD_AREA")?,
                solar_rad_coeff: opt_num(get("SOLAR_RAD_COEFF"), "SOLAR_RAD_COEFF")?,
                drag_area_m2: opt_num(get("DRAG_AREA"), "DRAG_AREA")?,
                drag_coeff: opt_num(get("DRAG_COEFF"), "DRAG_COEFF")?,
            })
        } else {
            None
        };

        let covariance = if covariance_block
            || keyword_occurs(&map, &[COV_REF_FRAME])
            || keyword_occurs(&map, &COVARIANCE6_KEYS)
        {
            // Table 4-3: none or all of the matrix values must be given.
            let mut lower_triangle = [0.0_f64; 21];
            for (slot, key) in lower_triangle.iter_mut().zip(COVARIANCE6_KEYS) {
                *slot = req_num(get(key), key)?;
            }
            Some(OmmCovariance {
                comments: covariance_comments,
                cov_ref_frame: xml_text(get(COV_REF_FRAME), COV_REF_FRAME)?,
                lower_triangle,
            })
        } else {
            None
        };

        Ok(Omm {
            ccsds_omm_vers: header.ccsds_omm_vers,
            classification: header.classification,
            creation_date: header.creation_date,
            originator: header.originator,
            message_id: header.message_id,
            object_name: metadata.object_name,
            object_id: metadata.object_id,
            center_name: metadata.center_name,
            ref_frame: metadata.ref_frame,
            ref_frame_epoch: metadata.ref_frame_epoch,
            time_system: metadata.time_system,
            mean_element_theory: metadata.mean_element_theory,
            epoch: mean_elements.epoch,
            mean_motion: mean_elements.mean_motion,
            semi_major_axis_km: mean_elements.semi_major_axis_km,
            eccentricity: mean_elements.eccentricity,
            inclination_deg: mean_elements.inclination_deg,
            ra_of_asc_node_deg: mean_elements.ra_of_asc_node_deg,
            arg_of_pericenter_deg: mean_elements.arg_of_pericenter_deg,
            mean_anomaly_deg: mean_elements.mean_anomaly_deg,
            gm_km3_s2: mean_elements.gm_km3_s2,
            spacecraft,
            ephemeris_type: tle_parameters.ephemeris_type,
            classification_type: tle_parameters.classification_type,
            norad_cat_id: tle_parameters.norad_cat_id,
            element_set_no: tle_parameters.element_set_no,
            rev_at_epoch: tle_parameters.rev_at_epoch,
            bstar: tle_parameters.bstar,
            bterm_m2_kg: tle_parameters.bterm_m2_kg,
            mean_motion_dot: tle_parameters.mean_motion_dot,
            mean_motion_ddot: tle_parameters.mean_motion_ddot,
            agom_m2_kg: tle_parameters.agom_m2_kg,
            covariance,
            user_defined,
            comments,
            exact_sgp4_epoch: None,
            quantize_tle_derived_fields: true,
        })
    }
}

/// Whether any of `keys` occurs, with any value including a blank one. A
/// blank optional keyword still opens its block.
fn keyword_occurs(map: &FieldMap, keys: &[&str]) -> bool {
    map.pairs()
        .iter()
        .any(|(key, _)| keys.contains(&key.as_str()))
}

struct OmmHeader {
    ccsds_omm_vers: Option<String>,
    classification: Option<String>,
    creation_date: Option<String>,
    originator: Option<String>,
    message_id: Option<String>,
}

/// Consume the OMM header fields and produce their validated canonical values.
fn parse_header(map: &FieldMap) -> Result<OmmHeader, OmmError> {
    let get = |key: &str| map.get(key);
    Ok(OmmHeader {
        ccsds_omm_vers: xml_text(get("CCSDS_OMM_VERS"), "CCSDS_OMM_VERS")?,
        classification: xml_text(get("CLASSIFICATION"), "CLASSIFICATION")?,
        creation_date: xml_text(get("CREATION_DATE"), "CREATION_DATE")?,
        originator: xml_text(get("ORIGINATOR"), "ORIGINATOR")?,
        message_id: xml_text(get("MESSAGE_ID"), "MESSAGE_ID")?,
    })
}

struct OmmMetadata {
    object_name: Option<String>,
    object_id: Option<String>,
    center_name: Option<String>,
    ref_frame: Option<String>,
    ref_frame_epoch: Option<String>,
    time_system: Option<String>,
    mean_element_theory: Option<String>,
}

/// Consume OMM metadata fields and produce their validated canonical values.
fn parse_metadata(map: &FieldMap, time_system: Option<String>) -> Result<OmmMetadata, OmmError> {
    let get = |key: &str| map.get(key);
    Ok(OmmMetadata {
        object_name: xml_text(get("OBJECT_NAME"), "OBJECT_NAME")?,
        object_id: xml_text(get("OBJECT_ID"), "OBJECT_ID")?,
        center_name: xml_text(get("CENTER_NAME"), "CENTER_NAME")?,
        ref_frame: xml_text(get("REF_FRAME"), "REF_FRAME")?,
        ref_frame_epoch: xml_text(get("REF_FRAME_EPOCH"), "REF_FRAME_EPOCH")?,
        time_system,
        mean_element_theory: xml_text(get("MEAN_ELEMENT_THEORY"), "MEAN_ELEMENT_THEORY")?,
    })
}

struct OmmMeanElements {
    epoch: OmmEpoch,
    mean_motion: Option<f64>,
    semi_major_axis_km: Option<f64>,
    eccentricity: f64,
    inclination_deg: f64,
    ra_of_asc_node_deg: f64,
    arg_of_pericenter_deg: f64,
    mean_anomaly_deg: f64,
    gm_km3_s2: Option<f64>,
}

/// Consume the OMM epoch and mean-element fields and produce canonical values.
/// Table 4-3 requires `SEMI_MAJOR_AXIS` or `MEAN_MOTION`; a message that gives
/// both keeps both.
fn parse_mean_elements(map: &FieldMap, epoch: OmmEpoch) -> Result<OmmMeanElements, OmmError> {
    let get = |key: &str| map.get(key);
    let mean_motion = opt_num(get("MEAN_MOTION"), "MEAN_MOTION")?;
    let semi_major_axis_km = opt_num(get("SEMI_MAJOR_AXIS"), "SEMI_MAJOR_AXIS")?;
    if mean_motion.is_none() && semi_major_axis_km.is_none() {
        return Err(OmmError::MissingField("SEMI_MAJOR_AXIS or MEAN_MOTION"));
    }
    Ok(OmmMeanElements {
        epoch,
        mean_motion,
        semi_major_axis_km,
        eccentricity: req_num(get("ECCENTRICITY"), "ECCENTRICITY")?,
        inclination_deg: req_num(get("INCLINATION"), "INCLINATION")?,
        ra_of_asc_node_deg: req_num(get("RA_OF_ASC_NODE"), "RA_OF_ASC_NODE")?,
        arg_of_pericenter_deg: req_num(get("ARG_OF_PERICENTER"), "ARG_OF_PERICENTER")?,
        mean_anomaly_deg: req_num(get("MEAN_ANOMALY"), "MEAN_ANOMALY")?,
        gm_km3_s2: opt_num(get("GM"), "GM")?,
    })
}

struct OmmTleParameters {
    ephemeris_type: Option<i32>,
    classification_type: Option<String>,
    norad_cat_id: Option<u32>,
    element_set_no: Option<i32>,
    rev_at_epoch: Option<i64>,
    bstar: Option<f64>,
    bterm_m2_kg: Option<f64>,
    mean_motion_dot: Option<f64>,
    mean_motion_ddot: Option<f64>,
    agom_m2_kg: Option<f64>,
}

/// Consume the OMM TLE-parameter fields. Every one is conditional on the
/// mean-element theory in table 4-3, so each is kept as stated.
fn parse_tle_parameters(map: &FieldMap) -> Result<OmmTleParameters, OmmError> {
    let get = |key: &str| map.get(key);
    Ok(OmmTleParameters {
        ephemeris_type: opt_int(get("EPHEMERIS_TYPE"), "EPHEMERIS_TYPE")?,
        classification_type: xml_text(get("CLASSIFICATION_TYPE"), "CLASSIFICATION_TYPE")?,
        norad_cat_id: opt_int(get("NORAD_CAT_ID"), "NORAD_CAT_ID")?,
        element_set_no: opt_int(get("ELEMENT_SET_NO"), "ELEMENT_SET_NO")?,
        rev_at_epoch: opt_int(get("REV_AT_EPOCH"), "REV_AT_EPOCH")?,
        bstar: opt_num(get("BSTAR"), "BSTAR")?,
        bterm_m2_kg: opt_num(get("BTERM"), "BTERM")?,
        mean_motion_dot: opt_num(get("MEAN_MOTION_DOT"), "MEAN_MOTION_DOT")?,
        mean_motion_ddot: opt_num(get("MEAN_MOTION_DDOT"), "MEAN_MOTION_DDOT")?,
        agom_m2_kg: opt_num(get("AGOM"), "AGOM")?,
    })
}

fn xml_text(value: Option<&str>, field: &'static str) -> Result<Option<String>, OmmError> {
    value
        .map(|value| xml_text_value(value, field).map(str::to_string))
        .transpose()
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

fn validate_texts(texts: &[String], field: &'static str) -> Result<(), OmmError> {
    for text in texts {
        xml_text_value(text, field)?;
    }
    Ok(())
}

// ── SGP4 bridge ──────────────────────────────────────────────────────

impl Omm {
    /// Convert the canonical OMM elements into the SGP4 [`ElementSet`] consumed
    /// by [`Satellite::from_elements`].
    ///
    /// The epoch is converted directly from the OMM calendar timestamp into
    /// SGP4's split Julian date, preserving years outside the TLE pivot range.
    ///
    /// With [`Omm::quantize_tle_derived_fields`] set (the default for a parsed
    /// OMM), B\* and the second mean-motion derivative are rounded to the
    /// values a TLE carries for them, because those GP parameters originate in
    /// the TLE field format: the assumed-decimal field's five mantissa digits
    /// and single-digit exponent, rounded as python-sgp4's `export_tle` rounds
    /// them, and at exponent `-9` below `1e-10`. The rounding is the TLE
    /// writer's, so these values written as a TLE read back unchanged, and a
    /// catalog OMM gives exactly the values of its catalog TLE. A value no TLE
    /// field holds (a magnitude that rounds to `1e9` or more) has no TLE value
    /// to match and passes through unquantized: SGP4 propagates any finite B\*
    /// and does not propagate with the derivatives, so such an element set
    /// still propagates correctly. Only [`tle::encode`] refuses it. The first
    /// mean-motion derivative passes through as stated. With the flag clear, as
    /// for a fitted OMM, every value passes through unchanged.
    ///
    /// An explicitly stated `MEAN_ELEMENT_THEORY` other than `SGP4`,
    /// `SGP/SGP4` or `SDP4` (any letter case), `CENTER_NAME` other than
    /// `EARTH`, `REF_FRAME` other than `TEME`, or `TIME_SYSTEM` other than
    /// `UTC` is refused with [`OmmError::IncompatibleMetadata`], since the
    /// elements would then not be the Earth-centred TEME UTC SGP4 elements the
    /// propagator reads (CCSDS 502.0-B-3 4.2.4.6). The keywords are checked in
    /// that order, so an element set of another theory is refused naming the
    /// theory whatever its frame. An absent or blank value is not refused. The
    /// elements SGP4 propagates with must be present: `MEAN_MOTION` (4.2.4.6;
    /// an OMM that gives only `SEMI_MAJOR_AXIS` is refused naming
    /// `MEAN_MOTION`) and `BSTAR`, each refused with [`OmmError::MissingField`]
    /// when absent. `NORAD_CAT_ID`, of up to nine digits, `MEAN_MOTION_DOT`
    /// and `MEAN_MOTION_DDOT` are carried when stated, since SGP4 does not
    /// propagate with them; only a TLE writer needs the five-character catalog
    /// field and the derivatives. The epoch calendar
    /// is validated, so a mutated [`OmmEpoch`] naming no civil instant is
    /// refused with [`OmmError::InvalidField`] for `epoch`.
    pub fn to_element_set(&self) -> Result<ElementSet, OmmError> {
        let inputs = validate_omm_bridge(self)?;
        // `validate_omm_bridge` has refused a non-finite value, so a
        // quantizer refuses only a magnitude its TLE field cannot hold. That
        // value has no TLE to match and SGP4 propagates it as it is.
        let quantize = |value: f64, round: fn(f64) -> Result<f64, tle::TleError>| {
            if self.quantize_tle_derived_fields {
                round(value).unwrap_or(value)
            } else {
                value
            }
        };
        let bstar = quantize(inputs.bstar, tle::quantize_bstar);
        let mean_motion_double_dot = self
            .mean_motion_ddot
            .map(|value| quantize(value, tle::quantize_mean_motion_double_dot));
        Ok(ElementSet {
            epoch: self
                .exact_sgp4_epoch
                .unwrap_or_else(|| self.epoch.sgp4_julian_date()),
            bstar,
            mean_motion_dot: self.mean_motion_dot,
            mean_motion_double_dot,
            eccentricity: self.eccentricity,
            argument_of_perigee_deg: self.arg_of_pericenter_deg,
            inclination_deg: self.inclination_deg,
            mean_anomaly_deg: self.mean_anomaly_deg,
            mean_motion_rev_per_day: inputs.mean_motion,
            right_ascension_deg: self.ra_of_asc_node_deg,
            catalog_number: self.norad_cat_id,
        })
    }
}

impl Satellite {
    /// Build a propagation-ready [`Satellite`] from an [`Omm`].
    ///
    /// Bridges the OMM mean elements into the validated SGP4 element path via
    /// [`Omm::to_element_set`]. Metadata that [`Omm::to_element_set`] refuses
    /// maps to [`Sgp4Error::InvalidInput`] naming the metadata keyword with
    /// [`Sgp4InputErrorKind::OutOfRange`]; the stated value is available from
    /// the [`OmmError::IncompatibleMetadata`] that [`Omm::to_element_set`]
    /// returns. A missing element maps to [`Sgp4Error::InvalidInput`] naming
    /// its keyword with [`Sgp4InputErrorKind::Missing`].
    pub fn from_omm(omm: &Omm) -> Result<Self, Sgp4Error> {
        let elements = omm.to_element_set().map_err(map_omm_bridge_to_sgp4)?;
        Self::from_elements(&elements)
    }
}

/// The OMM values SGP4 reads that table 4-3 makes conditional on the theory.
/// SGP4 propagates with neither the catalog number nor the mean-motion
/// derivatives, so those are carried when stated and not required.
struct Sgp4Inputs {
    mean_motion: f64,
    bstar: f64,
}

fn validate_omm_bridge(omm: &Omm) -> Result<Sgp4Inputs, OmmError> {
    // The theory is checked first: elements of another theory are not SGP4
    // input in any frame, so it is the incompatibility to name.
    check_bridge_label(
        "MEAN_ELEMENT_THEORY",
        omm.mean_element_theory.as_deref(),
        SGP4_THEORY_LABELS,
    )?;
    check_bridge_label("CENTER_NAME", omm.center_name.as_deref(), &["EARTH"])?;
    check_bridge_label("REF_FRAME", omm.ref_frame.as_deref(), &["TEME"])?;
    check_bridge_label("TIME_SYSTEM", omm.time_system.as_deref(), &["UTC"])?;
    let inputs = Sgp4Inputs {
        mean_motion: omm
            .mean_motion
            .ok_or(OmmError::MissingField("MEAN_MOTION"))?,
        bstar: omm.bstar.ok_or(OmmError::MissingField("BSTAR"))?,
    };
    validate_epoch(&omm.epoch, validate::CivilSecondPolicy::UtcLike)?;
    validate::finite_positive(inputs.mean_motion, "mean_motion").map_err(map_omm_field_error)?;
    validate::finite_in_range_exclusive_upper(omm.eccentricity, 0.0, 1.0, "eccentricity")
        .map_err(map_omm_field_error)?;
    validate::finite(omm.inclination_deg, "inclination_deg").map_err(map_omm_field_error)?;
    validate::finite(omm.ra_of_asc_node_deg, "ra_of_asc_node_deg").map_err(map_omm_field_error)?;
    validate::finite(omm.arg_of_pericenter_deg, "arg_of_pericenter_deg")
        .map_err(map_omm_field_error)?;
    validate::finite(omm.mean_anomaly_deg, "mean_anomaly_deg").map_err(map_omm_field_error)?;
    validate::finite(inputs.bstar, "bstar").map_err(map_omm_field_error)?;
    if let Some(value) = omm.mean_motion_dot {
        validate::finite(value, "mean_motion_dot").map_err(map_omm_field_error)?;
    }
    if let Some(value) = omm.mean_motion_ddot {
        validate::finite(value, "mean_motion_ddot").map_err(map_omm_field_error)?;
    }
    Ok(inputs)
}

/// Refuse an [`OmmEpoch`] that names no civil instant under `policy`, or whose
/// sub-second fields exceed their ranges.
fn validate_epoch(epoch: &OmmEpoch, policy: validate::CivilSecondPolicy) -> Result<(), OmmError> {
    if epoch.microsecond >= 1_000_000 {
        return Err(OmmError::InvalidField {
            field: "epoch.microsecond",
            kind: OmmInputErrorKind::OutOfRange,
        });
    }
    if epoch.femtosecond >= 1_000_000_000 {
        return Err(OmmError::InvalidField {
            field: "epoch.femtosecond",
            kind: OmmInputErrorKind::OutOfRange,
        });
    }
    validate::civil_datetime_with_second_policy(
        i64::from(epoch.year),
        i64::from(epoch.month),
        i64::from(epoch.day),
        i64::from(epoch.hour),
        i64::from(epoch.minute),
        f64::from(epoch.second),
        policy,
    )
    .map_err(|error| OmmError::InvalidField {
        field: "epoch",
        kind: OmmInputErrorKind::from(&error),
    })?;
    Ok(())
}

/// Refuse a stated metadata value outside `accepted`. Absent and blank values
/// pass: CelesTrak GP JSON and CSV omit these fields because the GP convention
/// fixes them, and CelesTrak KVN and XML may leave unavailable values blank.
/// Comparison ignores surrounding whitespace and ASCII letter case (502.0-B-3
/// 7.5.3 permits all-uppercase or all-lowercase normative values).
fn check_bridge_label(
    field: &'static str,
    value: Option<&str>,
    accepted: &[&str],
) -> Result<(), OmmError> {
    let Some(stated) = value else {
        return Ok(());
    };
    let label = stated.trim();
    if label.is_empty()
        || accepted
            .iter()
            .any(|candidate| label.eq_ignore_ascii_case(candidate))
    {
        return Ok(());
    }
    Err(OmmError::IncompatibleMetadata {
        field,
        value: stated.to_string(),
    })
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
        OmmError::IncompatibleMetadata { field, .. } => Sgp4Error::InvalidInput {
            field,
            kind: Sgp4InputErrorKind::OutOfRange,
        },
        OmmError::MissingField(field) => Sgp4Error::InvalidInput {
            field,
            kind: Sgp4InputErrorKind::Missing,
        },
        other => Sgp4Error::InvalidTle(other.to_string()),
    }
}
// ── Epoch ────────────────────────────────────────────────────────────

impl OmmEpoch {
    /// Parse a CCSDS `EPOCH` value (`YYYY-MM-DDThh:mm:ss[.f...][Z]`, UTC) by
    /// delegating to the shared NDM epoch parser.
    fn parse(text: &str, second_policy: validate::CivilSecondPolicy) -> Result<OmmEpoch, OmmError> {
        let e = crate::astro::ndm::NdmEpoch::parse(text, second_policy)
            .map_err(|err| map_omm_epoch_field_error(err, text.trim()))?;
        Ok(OmmEpoch {
            year: e.year,
            month: e.month,
            day: e.day,
            hour: e.hour,
            minute: e.minute,
            second: e.second,
            microsecond: e.microsecond,
            femtosecond: e.femtosecond,
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
            self.second as f64
                + self.microsecond as f64 / 1_000_000.0
                + self.femtosecond as f64 / 1_000_000_000_000_000.0,
        )
    }

    pub(crate) fn from_sgp4_julian_date(epoch: JulianDate) -> Self {
        let (mut jd_midnight, mut day_fraction) = if (epoch.0.fract().abs() - 0.5).abs() < 1.0e-9 {
            (epoch.0, epoch.1)
        } else if epoch.1 >= 0.5 {
            (epoch.0 + 0.5, epoch.1 - 0.5)
        } else {
            (epoch.0 - 0.5, epoch.1 + 0.5)
        };
        let day_carry = day_fraction.floor();
        jd_midnight += day_carry;
        day_fraction -= day_carry;
        let (year, month, day, hour, minute, second) =
            crate::astro::time::civil::civil_from_split_julian_date(jd_midnight, day_fraction);
        let whole_second = second.floor();
        let subsecond = second - whole_second;
        let mut femtoseconds = (subsecond * FEMTOSECONDS_PER_SECOND as f64).round() as i128;
        let mut second = whole_second as u32;
        if femtoseconds == FEMTOSECONDS_PER_SECOND {
            second += 1;
            femtoseconds = 0;
        }
        OmmEpoch {
            year: year as i32,
            month: month as u32,
            day: day as u32,
            hour: hour as u32,
            minute: minute as u32,
            second,
            microsecond: (femtoseconds / FEMTOSECONDS_PER_MICROSECOND) as u32,
            femtosecond: (femtoseconds % FEMTOSECONDS_PER_MICROSECOND) as u32,
        }
    }

    /// Format as a CCSDS `EPOCH` string via the shared NDM epoch encoder:
    /// six fractional digits, extended to 15 only when a sub-microsecond
    /// remainder is present.
    fn to_iso8601(&self) -> String {
        crate::astro::ndm::NdmEpoch {
            year: self.year,
            month: self.month,
            day: self.day,
            hour: self.hour,
            minute: self.minute,
            second: self.second,
            microsecond: self.microsecond,
            femtosecond: self.femtosecond,
        }
        .to_iso8601()
    }
}

const FEMTOSECONDS_PER_SECOND: i128 = 1_000_000_000_000_000;
const FEMTOSECONDS_PER_MICROSECOND: i128 = 1_000_000_000;

fn is_zero_u32(value: &u32) -> bool {
    *value == 0
}

fn default_quantize_tle_derived_fields() -> bool {
    true
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

fn opt_num(value: Option<&str>, field: &'static str) -> Result<Option<f64>, OmmError> {
    value.map(|value| parse_num(value, field)).transpose()
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
#[allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;

    const ISS_KVN: &str = include_str!("../../tests/fixtures/omm/25544.kvn");
    const ISS_XML: &str = include_str!("../../tests/fixtures/omm/25544.xml");
    const ISS_CSV: &str = "OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT\n\
ISS (ZARYA),1998-067A,2026-06-17T04:32:52.099296,15.49273435,0.0004737,51.6332,300.0813,195.1146,164.9702,0,U,25544,999,57175,0.00017172,9.113e-5,0";

    /// Reduce an OMM to its canonical orbital + catalog content (the fields the
    /// Elixir `Sidereon.Elements` struct carries), blanking the free-text header
    /// metadata that CelesTrak emits inconsistently across encodings: it labels
    /// the element theory `SGP/SGP4` in KVN but `SGP4` in XML/JSON, and its JSON
    /// omits `CENTER_NAME`/`REF_FRAME`/`TIME_SYSTEM` entirely. Cross-encoding
    /// identity is asserted on this canonical content, which must match exactly.
    fn canonical(omm: &Omm) -> Omm {
        Omm {
            ccsds_omm_vers: None,
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
        assert_eq!(omm.ccsds_omm_vers.as_deref(), Some("2.0"));
        assert_eq!(omm.object_name.as_deref(), Some("ISS (ZARYA)"));
        assert_eq!(omm.object_id.as_deref(), Some("1998-067A"));
        assert_eq!(omm.norad_cat_id, Some(25544));
        assert_eq!(omm.mean_motion, Some(15.49273435));
        assert_eq!(omm.eccentricity, 0.0004737);
        assert_eq!(omm.inclination_deg, 51.6332);
        assert_eq!(omm.bstar, Some(0.00017172));
        assert_eq!(omm.mean_motion_dot, Some(9.113e-5));
        assert_eq!(omm.mean_motion_ddot, Some(0.0));
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
                femtosecond: 0,
            }
        );
    }

    #[test]
    fn missing_drag_terms_are_kept_absent_and_only_bstar_is_required() {
        // Table 4-3 makes BSTAR, MEAN_MOTION_DOT and MEAN_MOTION_DDOT conditional
        // on the mean-element theory, so the reader keeps the OMM without them.
        // SGP4 propagates with BSTAR, so the bridge refuses its absence.
        let omm = parse_kvn(&kvn_without_field("BSTAR")).expect("OMM without BSTAR");
        assert_eq!(omm.to_element_set(), Err(OmmError::MissingField("BSTAR")));
        assert_eq!(
            Satellite::from_omm(&omm).expect_err("missing BSTAR"),
            Sgp4Error::InvalidInput {
                field: "BSTAR",
                kind: Sgp4InputErrorKind::Missing,
            }
        );

        // SGP4 does not propagate with the catalog number or the mean-motion
        // derivatives: an OMM without them propagates as one stating them.
        let full = Satellite::from_omm(&parse_kvn(ISS_KVN).unwrap())
            .unwrap()
            .propagate(sgp4::MinutesSinceEpoch(90.0))
            .unwrap();
        for field in ["NORAD_CAT_ID", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT"] {
            let omm = parse_kvn(&kvn_without_field(field)).expect("OMM without the field");
            let state = Satellite::from_omm(&omm)
                .unwrap_or_else(|error| panic!("{field} absent: {error}"))
                .propagate(sgp4::MinutesSinceEpoch(90.0))
                .unwrap();
            assert_eq!(state, full, "{field}");
        }
        let omm = parse_kvn(&kvn_with_field("NORAD_CAT_ID", "123456789")).unwrap();
        assert_eq!(
            omm.to_element_set().unwrap().catalog_number,
            Some(123_456_789)
        );
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
                femtosecond: 0,
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
    fn parse_kvn_preserves_sub_microsecond_epoch_seconds() {
        let omm = parse_kvn(&kvn_with_field("EPOCH", "2026-06-17T04:32:52.9999995"))
            .expect("fractional epoch");
        assert_eq!(
            omm.epoch,
            OmmEpoch {
                year: 2026,
                month: 6,
                day: 17,
                hour: 4,
                minute: 32,
                second: 52,
                microsecond: 999_999,
                femtosecond: 500_000_000,
            }
        );
        assert!(
            encode_kvn(&omm)
                .unwrap()
                .contains("EPOCH = 2026-06-17T04:32:52.999999500000000"),
            "sub-microsecond epoch must encode with high fractional precision"
        );
    }

    #[test]
    fn parse_kvn_preserves_continuous_time_sub_microsecond_epoch_near_day() {
        let omm = parse_kvn(&kvn_with_fields(&[
            ("TIME_SYSTEM", "GPS"),
            ("EPOCH", "2026-06-17T23:59:59.9999995"),
        ]))
        .expect("continuous-time fractional epoch");
        assert_eq!(
            omm.epoch,
            OmmEpoch {
                year: 2026,
                month: 6,
                day: 17,
                hour: 23,
                minute: 59,
                second: 59,
                microsecond: 999_999,
                femtosecond: 500_000_000,
            }
        );
    }

    #[test]
    fn parse_kvn_preserves_continuous_time_sub_microsecond_epoch_near_year() {
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
                femtosecond: 0,
            }
        );
        assert!(
            encode_kvn(&ordinary)
                .unwrap()
                .contains("EPOCH = 2026-12-31T23:59:58.123456"),
            "ordinary epoch must encode unchanged"
        );

        let carried = parse_kvn(&kvn_with_fields(&[
            ("TIME_SYSTEM", "GPS"),
            ("EPOCH", "2026-12-31T23:59:59.9999995"),
        ]))
        .expect("continuous-time fractional epoch near year boundary");
        assert_eq!(
            carried.epoch,
            OmmEpoch {
                year: 2026,
                month: 12,
                day: 31,
                hour: 23,
                minute: 59,
                second: 59,
                microsecond: 999_999,
                femtosecond: 500_000_000,
            }
        );
        assert!(
            encode_kvn(&carried)
                .unwrap()
                .contains("EPOCH = 2026-12-31T23:59:59.999999500000000"),
            "sub-microsecond year-end epoch must encode unchanged"
        );
    }

    #[test]
    fn kvn_round_trips_through_struct() {
        let omm = parse_kvn(ISS_KVN).unwrap();
        let reparsed = parse_kvn(&encode_kvn(&omm).unwrap()).unwrap();
        assert_eq!(omm, reparsed);
    }

    #[test]
    fn kvn_re_encodes_catalog_epoch_byte_faithfully() {
        // Parse -> encode must not balloon precision: a real microsecond-form
        // catalog epoch re-encodes as the exact source text, and the 15-digit
        // form appears only when sub-microsecond information is present.
        let source_epoch = ISS_KVN
            .lines()
            .find_map(|line| match line.split_once('=') {
                Some((key, value)) if key.trim() == "EPOCH" => Some(value.trim()),
                _ => None,
            })
            .expect("fixture EPOCH");
        let encoded = encode_kvn(&parse_kvn(ISS_KVN).unwrap()).unwrap();
        assert!(
            encoded.contains(&format!("EPOCH = {source_epoch}\n")),
            "catalog epoch {source_epoch} must re-encode byte-faithfully"
        );
        assert_eq!(source_epoch.len(), "2026-06-17T04:32:52.099296".len());
    }

    #[test]
    fn kvn_round_trips_femtosecond_epoch_through_struct() {
        // IR-level parse(encode(ir)) == ir must include the femtosecond field.
        let omm = parse_kvn(&kvn_with_field("EPOCH", "2026-06-17T04:32:52.9999995")).unwrap();
        assert_eq!(omm.epoch.femtosecond, 500_000_000);
        let reparsed = parse_kvn(&encode_kvn(&omm).unwrap()).unwrap();
        assert_eq!(omm, reparsed);
        assert_eq!(reparsed.epoch.femtosecond, 500_000_000);
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
        let reparsed = parse_xml(&encode_xml(&omm).unwrap()).unwrap();
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
        let reparsed =
            parse_xml(&encode_xml(&omm).unwrap()).expect("encoded OMM must remain valid XML");
        assert_eq!(omm, reparsed);
    }

    #[test]
    fn xml_round_trip_preserves_carriage_returns_in_text_values() {
        for value in ["SGP\rSGP4", "SGP\r\nSGP4"] {
            let mut omm = parse_kvn(ISS_KVN).expect("base OMM must parse");
            omm.mean_element_theory = Some(value.to_string());
            let encoded = encode_xml(&omm).unwrap();
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
        let reparsed = parse_json(&encode_json(&omm).unwrap()).unwrap();
        assert_eq!(omm, reparsed);
    }

    #[test]
    fn json_array_round_trips_through_struct() {
        const ISS_JSON: &str = include_str!("../../tests/fixtures/omm/25544.json");
        let omm = parse_json(ISS_JSON).unwrap();
        let encoded = encode_json_array(std::slice::from_ref(&omm)).unwrap();
        let reparsed = parse_json_array(&encoded).unwrap();
        assert!(reparsed.skipped.is_empty());
        assert_eq!(reparsed.omms, vec![omm]);
    }

    #[test]
    fn csv_matches_json_orbital_content() {
        const ISS_JSON: &str = include_str!("../../tests/fixtures/omm/25544.json");
        let csv = parse_csv(ISS_CSV).unwrap();
        let json = parse_json(ISS_JSON).unwrap();
        assert_eq!(canonical(&csv), canonical(&json));
    }

    #[test]
    fn csv_round_trips_through_struct() {
        let omm = parse_csv(ISS_CSV).unwrap();
        let reparsed = parse_csv(&encode_csv(std::slice::from_ref(&omm)).unwrap()).unwrap();
        assert_eq!(omm, reparsed);
    }

    #[test]
    fn csv_preserves_sub_microsecond_epoch() {
        let text = ISS_CSV.replace(
            "2026-06-17T04:32:52.099296",
            "2026-06-17T04:32:52.099296123456789",
        );
        let omm = parse_csv(&text).expect("high-precision CSV epoch");
        assert_eq!(omm.epoch.microsecond, 99_296);
        assert_eq!(omm.epoch.femtosecond, 123_456_789);
    }

    #[test]
    fn parse_csv_array_skips_malformed_rows_and_counts_them() {
        let mut text = String::from(ISS_CSV);
        text.push('\n');
        text.push_str("BROKEN,ROW\n");
        text.push_str(ISS_CSV.lines().nth(1).expect("CSV data row"));
        let parsed = parse_csv_array(&text).expect("CSV with bad row still parses");
        assert_eq!(
            parsed.skipped,
            vec![OmmSkippedRecord {
                index: 1,
                reason: OmmError::CsvColumnCount {
                    found: 2,
                    expected: 17,
                },
            }]
        );
        assert_eq!(parsed.omms.len(), 2);
        assert_eq!(
            parsed
                .omms
                .iter()
                .map(|omm| omm.norad_cat_id)
                .collect::<Vec<_>>(),
            vec![Some(25544), Some(25544)]
        );
    }

    #[test]
    fn csv_quotes_delimiters() {
        let mut omm = parse_csv(ISS_CSV).unwrap();
        omm.object_name = Some("SAT, \"A\"".to_string());
        let encoded = encode_csv(std::slice::from_ref(&omm)).unwrap();
        assert!(encoded.contains("\"SAT, \"\"A\"\"\""));
        let reparsed = parse_csv(&encoded).unwrap();
        assert_eq!(reparsed.object_name.as_deref(), Some("SAT, \"A\""));
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
    fn parse_auto_detects_csv() {
        assert_eq!(parse(ISS_CSV).unwrap(), parse_csv(ISS_CSV).unwrap());
    }

    #[test]
    fn parse_json_array_skips_malformed_objects_and_counts_them() {
        // A CelesTrak-shaped array with two good OMMs interleaved with a
        // non-object element and a malformed object (no EPOCH or mean
        // elements). One bad object must not reject the whole array: the good
        // records survive and each skip is reported in `skipped` with its
        // index and reason.
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
        assert_eq!(
            result.skipped,
            vec![
                OmmSkippedRecord {
                    index: 1,
                    reason: OmmError::Field("expected a JSON object".to_string()),
                },
                OmmSkippedRecord {
                    index: 2,
                    reason: OmmError::MissingField("EPOCH"),
                },
            ],
            "the string and the malformed object"
        );
        let norads: Vec<Option<u32>> = result.omms.iter().map(|o| o.norad_cat_id).collect();
        assert_eq!(
            norads,
            vec![Some(25544), Some(25545)],
            "both good OMMs must survive"
        );
    }

    #[test]
    fn bstar_quantizes_onto_assumed_decimal_grid() {
        // OMM B* is the plain-decimal 0.00017172; the SGP4 element set must carry
        // the assumed-decimal value 0.17172e-3 the TLE actually feeds SGP4.
        let omm = parse_kvn(ISS_KVN).unwrap();
        let es = omm.to_element_set().expect("valid OMM bridge");
        assert_eq!(es.bstar, 0.17172 * 10.0_f64.powi(-3));
        assert_ne!(Some(es.bstar), omm.bstar);
    }

    const ISS_TLE: &str = include_str!("../../tests/fixtures/omm/25544.tle");

    /// The ISS catalog TLE, the same element set as `ISS_KVN`.
    fn iss_tle_elements() -> tle::TleElements {
        let mut lines = ISS_TLE.lines().filter(|line| !line.trim().is_empty());
        let _name = lines.next();
        let (line1, line2) = (lines.next().unwrap(), lines.next().unwrap());
        tle::parse(line1, line2).unwrap().elements
    }

    /// Bridge an ISS OMM carrying the given B\* and second derivative, write
    /// the resulting element set as a TLE, and read that back.
    fn omm_tle_round_trip(bstar: f64, nddot: f64) -> (ElementSet, ElementSet, String) {
        let mut omm = parse_kvn(ISS_KVN).unwrap();
        omm.bstar = Some(bstar);
        omm.mean_motion_ddot = Some(nddot);
        let from_omm = omm.to_element_set().expect("TLE-holdable terms bridge");

        let mut el = iss_tle_elements();
        el.bstar = from_omm.bstar;
        el.bstar_text = None;
        el.mean_motion_double_dot = from_omm.mean_motion_double_dot.unwrap();
        el.mean_motion_double_dot_text = None;
        let (line1, line2) = tle::encode(&el).expect("a quantized element set is writable");
        let from_tle = tle::parse_with_policy(&line1, &line2, tle::TlePolicy::Strict)
            .unwrap()
            .elements
            .to_element_set()
            .unwrap();
        (from_omm, from_tle, line1)
    }

    #[test]
    fn quantized_element_set_is_a_fixed_point_of_tle_text() {
        let cases = [
            (-2.3456789e-12, -2.3456789e-12, "-00235-9", "-00235-9"),
            (1.0e-10, 1.0e-10, " 10000-9", " 10000-9"),
            (0.999996e-9, 9.999996e-11, " 10000-8", " 10000-9"),
            (4.0e-15, -4.0e-15, " 00000+0", "-00000-0"),
            (-0.0, 0.0, "-00000+0", " 00000-0"),
            (1.009e-5, 0.0, " 10090-4", " 00000-0"),
            (3.21675e-9, 0.5, " 32168-8", " 50000-0"),
            (0.999994e9, -0.999994e9, " 99999+9", "-99999+9"),
        ];
        let check = |bstar: f64, nddot: f64| {
            let (from_omm, from_tle, line1) = omm_tle_round_trip(bstar, nddot);
            let label = format!("{bstar:e} {nddot:e}: {line1}");
            assert_eq!(
                from_omm.bstar.to_bits(),
                from_tle.bstar.to_bits(),
                "{label}"
            );
            assert_eq!(
                from_omm.mean_motion_double_dot.map(f64::to_bits),
                from_tle.mean_motion_double_dot.map(f64::to_bits),
                "{label}"
            );
            (line1, label)
        };
        for (bstar, nddot, bstar_text, nddot_text) in cases {
            let (line1, label) = check(bstar, nddot);
            assert_eq!(&line1[53..61], bstar_text, "{label}");
            assert_eq!(&line1[44..52], nddot_text, "{label}");
        }

        // Sampled values across magnitudes 1e-20 to 1e9, from a fixed
        // SplitMix64 stream.
        let mut state = 0x0A11_CE5E_ED00_0001_u64;
        let mut unit = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
        };
        let mut sample = || {
            let magnitude = libm::pow(10.0, -20.0 + 28.9 * unit());
            if unit() < 0.5 {
                magnitude
            } else {
                -magnitude
            }
        };
        for _ in 0..5_000 {
            let (bstar, nddot) = (sample(), sample());
            check(bstar, nddot);
        }
    }

    #[test]
    fn quantized_omm_propagates_as_the_tle_carrying_its_terms() {
        for (bstar, nddot) in [
            (-2.3456789e-12, 0.0),
            (1.009e-5, 1.0e-10),
            (4.0e-15, -4.0e-15),
        ] {
            let mut omm = parse_kvn(ISS_KVN).unwrap();
            omm.bstar = Some(bstar);
            omm.mean_motion_ddot = Some(nddot);
            let from_omm = Satellite::from_omm(&omm).unwrap();

            let elements = omm.to_element_set().unwrap();
            let mut el = iss_tle_elements();
            el.bstar = elements.bstar;
            el.bstar_text = None;
            el.mean_motion_double_dot = elements.mean_motion_double_dot.unwrap();
            el.mean_motion_double_dot_text = None;
            let (line1, line2) = tle::encode(&el).unwrap();
            let from_tle = Satellite::from_tle(&line1, &line2).unwrap();
            for minutes in [0.0, 90.0, 1440.0] {
                let a = from_omm
                    .propagate(sgp4::MinutesSinceEpoch(minutes))
                    .unwrap();
                let b = from_tle
                    .propagate(sgp4::MinutesSinceEpoch(minutes))
                    .unwrap();
                for axis in 0..3 {
                    assert_eq!(
                        a.position[axis].to_bits(),
                        b.position[axis].to_bits(),
                        "{bstar:e} at {minutes} min"
                    );
                    assert_eq!(
                        a.velocity[axis].to_bits(),
                        b.velocity[axis].to_bits(),
                        "{bstar:e} at {minutes} min"
                    );
                }
            }
        }
    }

    #[test]
    fn terms_no_tle_field_holds_pass_through_unquantized() {
        let iss = parse_kvn(ISS_KVN).unwrap().to_element_set().unwrap();
        for (bstar, nddot) in [
            (1.0e9, None),
            (-0.999996e9, None),
            (1.7172e-4, Some(2.0e12)),
        ] {
            let mut omm = parse_kvn(ISS_KVN).unwrap();
            omm.bstar = Some(bstar);
            if nddot.is_some() {
                omm.mean_motion_ddot = nddot;
            }
            for quantize in [true, false] {
                omm.quantize_tle_derived_fields = quantize;
                let label = format!("{bstar:e} {nddot:?}, quantize {quantize}");
                let elements = omm
                    .to_element_set()
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                if bstar.abs() >= 0.999996e9 || !quantize {
                    assert_eq!(elements.bstar.to_bits(), bstar.to_bits(), "{label}");
                } else {
                    assert_eq!(elements.bstar.to_bits(), iss.bstar.to_bits(), "{label}");
                }
                match nddot {
                    Some(value) => assert_eq!(elements.mean_motion_double_dot, Some(value)),
                    None => assert_eq!(elements.mean_motion_double_dot, omm.mean_motion_ddot),
                }
                assert_eq!(elements.mean_motion_dot, omm.mean_motion_dot, "{label}");
                Satellite::from_omm(&omm).unwrap_or_else(|error| panic!("{label}: {error}"));

                // The TLE writer refuses the element set: no TLE field holds
                // the term.
                let mut el = iss_tle_elements();
                el.bstar = elements.bstar;
                el.bstar_text = None;
                el.mean_motion_double_dot = elements.mean_motion_double_dot.unwrap();
                el.mean_motion_double_dot_text = None;
                assert!(
                    matches!(tle::encode(&el), Err(tle::TleError::InvalidField { .. })),
                    "{label}"
                );
            }
        }

        // The first derivative is carried as stated, with more decimals than
        // its TLE field or a magnitude the field cannot hold.
        for ndot in [1.23456789e-5, 1.0, -0.999999996] {
            let mut omm = parse_kvn(ISS_KVN).unwrap();
            omm.mean_motion_dot = Some(ndot);
            let elements = omm.to_element_set().unwrap();
            assert_eq!(elements.mean_motion_dot, Some(ndot));
        }
    }

    #[test]
    fn to_element_set_rejects_invalid_bridge_fields() {
        let mut omm = parse_kvn(ISS_KVN).unwrap();
        omm.mean_motion = Some(f64::NAN);
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
    fn from_omm_preserves_sub_microsecond_year_end_epoch_directly() {
        for (epoch, expected_year) in [
            ("2021-12-31T23:59:59.9999995", 2021),
            ("2020-12-31T23:59:59.9999995", 2020),
        ] {
            let omm = parse_kvn(&kvn_with_field("EPOCH", epoch)).expect("year-end OMM epoch");
            assert_eq!(omm.epoch.year, expected_year);
            assert_eq!(omm.epoch.month, 12);
            assert_eq!(omm.epoch.day, 31);
            assert_eq!(omm.epoch.hour, 23);
            assert_eq!(omm.epoch.minute, 59);
            assert_eq!(omm.epoch.second, 59);
            assert_eq!(omm.epoch.microsecond, 999_999);
            assert_eq!(omm.epoch.femtosecond, 500_000_000);

            let sat =
                Satellite::from_omm(&omm).expect("sub-microsecond year-end OMM must initialize");
            let epoch_jd = sat.epoch_jd();
            let actual_jd = epoch_jd.0 + epoch_jd.1;
            let expected_jd =
                crate::astro::time::scales::julian_day_number(expected_year, 12, 31) as f64 - 0.5
                    + (86_399.999_999_5 / 86_400.0);

            assert!(
                (actual_jd - expected_jd).abs() < 1.0e-9,
                "{epoch} converted to JD {actual_jd}, expected {expected_jd}",
            );
        }
    }

    #[test]
    fn from_sgp4_julian_date_normalizes_split_fraction_carry() {
        let (jd_midnight, _) =
            crate::astro::time::civil::split_julian_date(2026, 12, 31, 0, 0, 0.0);
        let epoch = OmmEpoch::from_sgp4_julian_date(JulianDate(jd_midnight, 1.0));

        assert_eq!(
            epoch,
            OmmEpoch {
                year: 2027,
                month: 1,
                day: 1,
                hour: 0,
                minute: 0,
                second: 0,
                microsecond: 0,
                femtosecond: 0,
            }
        );
    }

    #[test]
    fn from_omm_rejects_invalid_sgp4_element_fields() {
        let mut omm = parse_kvn(ISS_KVN).unwrap();
        omm.mean_motion = Some(f64::NAN);
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

    const ISS_JSON_FIXTURE: &str = include_str!("../../tests/fixtures/omm/25544.json");

    /// CCSDS 502.0-B-3 annex G figure G-9: units on numeric values, day-of-year
    /// epochs and a user-defined parameter.
    const STANDARD_UNITS_KVN: &str = "\
CCSDS_OMM_VERS = 3.0
CREATION_DATE = 2020-065T16:00:00
ORIGINATOR     = NOAA

OBJECT_NAME    = GOES 9
OBJECT_ID      = 1995-025A
CENTER_NAME    = EARTH
REF_FRAME      = TEME
TIME_SYSTEM    = UTC
MEAN_ELEMENT_THEORY = SGP/SGP4


EPOCH              = 2020-064T10:34:41.4264
MEAN_MOTION        = 1.00273272    [rev/day]
ECCENTRICITY       = 0.0005013
INCLINATION        =    3.0539     [deg]
RA_OF_ASC_NODE     = 81.7939       [deg]
ARG_OF_PERICENTER = 249.2363       [deg]
MEAN_ANOMALY       = 150.1602      [deg]
GM                 = 398600.8      [km**3/s**2]
EPHEMERIS_TYPE     = 0
CLASSIFICATION_TYPE = U
NORAD_CAT_ID    = 23581
ELEMENT_SET_NO = 0925
REV_AT_EPOCH       = 4316
BSTAR              = 0.0001        [1/ER]
MEAN_MOTION_DOT    = -0.00000113   [rev/day**2]
MEAN_MOTION_DDOT = 0.0             [rev/day**3]

USER_DEFINED_EARTH_MODEL = WGS-84
";

    /// CCSDS 502.0-B-3 annex G figure G-10, the OMM example in XML, with the
    /// schema location on one line.
    const STANDARD_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<omm xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
      xsi:noNamespaceSchemaLocation="https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-3.0.0-master-3.0.xsd"
      id="CCSDS_OMM_VERS" version="3.0">
  <header>
    <COMMENT> THIS IS AN XML VERSION OF THE OMM </COMMENT>
    <CLASSIFICATION>CUI</CLASSIFICATION>
    <CREATION_DATE>2020-065T16:00:00</CREATION_DATE>
    <ORIGINATOR>NOAA</ORIGINATOR>
    <MESSAGE_ID> OMM 202013719185</MESSAGE_ID>
  </header>
  <body>
    <segment>
      <metadata>
        <OBJECT_NAME>GOES-9</OBJECT_NAME>
        <OBJECT_ID>1995-025A</OBJECT_ID>
        <CENTER_NAME>EARTH</CENTER_NAME>
        <REF_FRAME>TEME</REF_FRAME>
        <TIME_SYSTEM>UTC</TIME_SYSTEM>
        <MEAN_ELEMENT_THEORY>SGP4</MEAN_ELEMENT_THEORY>
      </metadata>
       <data>
         <meanElements>
           <EPOCH>2020-064T10:34:41.4264</EPOCH>
           <MEAN_MOTION>1.00273272</MEAN_MOTION>
           <ECCENTRICITY>0.0005013</ECCENTRICITY>
           <INCLINATION>3.0539</INCLINATION>
           <RA_OF_ASC_NODE>81.7939</RA_OF_ASC_NODE>
           <ARG_OF_PERICENTER>249.2363</ARG_OF_PERICENTER>
           <MEAN_ANOMALY>150.1602</MEAN_ANOMALY>
           <GM>398600.8</GM>
         </meanElements>
         <tleParameters>
           <NORAD_CAT_ID>23581</NORAD_CAT_ID>
           <ELEMENT_SET_NO>0925</ELEMENT_SET_NO>
           <REV_AT_EPOCH>4316</REV_AT_EPOCH>
           <BSTAR>0.0001</BSTAR>
           <MEAN_MOTION_DOT>-0.00000113</MEAN_MOTION_DOT>
           <MEAN_MOTION_DDOT>0.0</MEAN_MOTION_DDOT>
         </tleParameters>
         <covarianceMatrix>
           <COV_REF_FRAME>TEME</COV_REF_FRAME>
           <CX_X>3.331349476038534e-04</CX_X>
           <CY_X>4.618927349220216e-04</CY_X>
           <CY_Y>6.782421679971363e-04</CY_Y>
           <CZ_X>-3.070007847730449e-04</CZ_X>
           <CZ_Y>-4.221234189514228e-04</CZ_Y>
           <CZ_Z>3.231931992380369e-04</CZ_Z>
           <CX_DOT_X>-3.349365033922630e-07</CX_DOT_X>
           <CX_DOT_Y>-4.686084221046758e-07</CX_DOT_Y>
           <CX_DOT_Z>2.484949578400095e-07</CX_DOT_Z>
           <CX_DOT_X_DOT>4.296022805587290e-10</CX_DOT_X_DOT>
           <CY_DOT_X>-2.211832501084875e-07</CY_DOT_X>
           <CY_DOT_Y>-2.864186892102733e-07</CY_DOT_Y>
           <CY_DOT_Z>1.798098699846038e-07</CY_DOT_Z>
           <CY_DOT_X_DOT>2.608899201686016e-10</CY_DOT_X_DOT>
           <CY_DOT_Y_DOT>1.767514756338532e-10</CY_DOT_Y_DOT>
           <CZ_DOT_X>-3.041346050686871e-07</CZ_DOT_X>
           <CZ_DOT_Y>-4.989496988610662e-07</CZ_DOT_Y>
           <CZ_DOT_Z>3.540310904497689e-07</CZ_DOT_Z>
           <CZ_DOT_X_DOT>1.869263192954590e-10</CZ_DOT_X_DOT>
           <CZ_DOT_Y_DOT>1.008862586240695e-10</CZ_DOT_Y_DOT>
           <CZ_DOT_Z_DOT>6.224444338635500e-10</CZ_DOT_Z_DOT>
         </covarianceMatrix>
       </data>
    </segment>
  </body>
</omm>"#;

    /// Every optional table 4-1 to 4-3 item the container retains, with a
    /// comment at the start of each block (502.0-B-3 7.8.8).
    const RETAINED_KVN: &str = "\
CCSDS_OMM_VERS = 3.0
COMMENT header comment
CLASSIFICATION = SBU
CREATION_DATE = 2020-065T16:00:00
ORIGINATOR = NOAA
MESSAGE_ID = OMM 202013719185
COMMENT metadata comment
OBJECT_NAME = GOES 9
OBJECT_ID = 1995-025A
CENTER_NAME = EARTH
REF_FRAME = TEME
REF_FRAME_EPOCH = 2020-064T00:00:00
TIME_SYSTEM = UTC
MEAN_ELEMENT_THEORY = SGP/SGP4
COMMENT mean elements comment
EPOCH = 2020-064T10:34:41.4264
MEAN_MOTION = 1.00273272
ECCENTRICITY = 0.0005013
INCLINATION = 3.0539
RA_OF_ASC_NODE = 81.7939
ARG_OF_PERICENTER = 249.2363
MEAN_ANOMALY = 150.1602
GM = 398600.8
COMMENT spacecraft comment
MASS = 2500 [kg]
DRAG_COEFF = 2.2
COMMENT  indented TLE comment
EPHEMERIS_TYPE = 0
CLASSIFICATION_TYPE = U
NORAD_CAT_ID = 23581
ELEMENT_SET_NO = 925
REV_AT_EPOCH = 4316
BSTAR = 0.0001
MEAN_MOTION_DOT = -0.00000113
MEAN_MOTION_DDOT = 0.0
COMMENT covariance comment
COV_REF_FRAME = TEME
CX_X = 3.331349476038534e-04
CY_X = 4.618927349220216e-04
CY_Y = 6.782421679971363e-04
CZ_X = -3.070007847730449e-04
CZ_Y = -4.221234189514228e-04
CZ_Z = 3.231931992380369e-04
CX_DOT_X = -3.349365033922630e-07
CX_DOT_Y = -4.686084221046758e-07
CX_DOT_Z = 2.484949578400095e-07
CX_DOT_X_DOT = 4.296022805587290e-10
CY_DOT_X = -2.211832501084875e-07
CY_DOT_Y = -2.864186892102733e-07
CY_DOT_Z = 1.798098699846038e-07
CY_DOT_X_DOT = 2.608899201686016e-10
CY_DOT_Y_DOT = 1.767514756338532e-10
CZ_DOT_X = -3.041346050686871e-07
CZ_DOT_Y = -4.989496988610662e-07
CZ_DOT_Z = 3.540310904497689e-07
CZ_DOT_X_DOT = 1.869263192954590e-10
CZ_DOT_Y_DOT = 1.008862586240695e-10
CZ_DOT_Z_DOT = 6.224444338635500e-10
COMMENT user-defined comment
USER_DEFINED_EARTH_MODEL = WGS-84
USER_DEFINED_C3 = 29.376 [km**2/s**2]
";

    fn omm_element(xml: &str) -> &str {
        let start = xml.find("<omm ").expect("omm start tag");
        let end = xml.find("</omm>").expect("omm end tag") + "</omm>".len();
        &xml[start..end]
    }

    #[test]
    fn reads_the_standard_units_example_and_writes_it_back() {
        let omm = parse_kvn(STANDARD_UNITS_KVN).expect("502.0-B-3 figure G-9 parses");
        assert_eq!(omm.ccsds_omm_vers.as_deref(), Some("3.0"));
        assert_eq!(omm.creation_date.as_deref(), Some("2020-065T16:00:00"));
        assert_eq!(
            (omm.epoch.year, omm.epoch.month, omm.epoch.day),
            (2020, 3, 4)
        );
        assert_eq!(omm.epoch.microsecond, 426_400);
        assert_eq!(omm.mean_motion, Some(1.00273272));
        assert_eq!(omm.inclination_deg, 3.0539);
        assert_eq!(omm.gm_km3_s2, Some(398600.8));
        assert_eq!(omm.bstar, Some(0.0001));
        assert_eq!(omm.mean_motion_dot, Some(-0.00000113));
        assert_eq!(omm.element_set_no, Some(925));
        assert_eq!(
            omm.user_defined,
            vec![OmmUserDefined {
                parameter: "EARTH_MODEL".to_string(),
                value: "WGS-84".to_string(),
            }]
        );

        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_json(&encode_json(&omm).unwrap()).unwrap(), omm);
        assert_eq!(
            parse_csv(&encode_csv(std::slice::from_ref(&omm)).unwrap()).unwrap(),
            omm
        );
        let _satellite = Satellite::from_omm(&omm).expect("GOES 9 elements propagate");
    }

    #[test]
    fn reads_the_standard_xml_example_with_header_items_and_covariance() {
        let omm = parse_xml(STANDARD_XML).expect("502.0-B-3 figure G-10 parses");
        assert_eq!(
            omm.comments.header,
            // Leading whitespace belongs to a comment (7.8.5); trailing does not.
            vec![" THIS IS AN XML VERSION OF THE OMM"]
        );
        assert_eq!(omm.classification.as_deref(), Some("CUI"));
        assert_eq!(omm.message_id.as_deref(), Some("OMM 202013719185"));
        assert_eq!(omm.object_name.as_deref(), Some("GOES-9"));
        assert_eq!(omm.gm_km3_s2, Some(398600.8));
        // Figure G-10 states neither; the table 4-3 defaults are not invented.
        assert_eq!(omm.ephemeris_type, None);
        assert_eq!(omm.classification_type, None);
        let covariance = omm.covariance.as_ref().expect("covariance retained");
        assert_eq!(covariance.cov_ref_frame.as_deref(), Some("TEME"));
        assert_eq!(covariance.lower_triangle[0], 3.331349476038534e-04);
        // Figure G-10 states 6.224444338635500e-10; the trailing zeros name
        // the same double.
        assert_eq!(covariance.lower_triangle[20], 6.2244443386355e-10);

        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
    }

    #[test]
    fn retains_optional_items_and_block_comments_through_kvn_and_xml() {
        let omm = parse_kvn(RETAINED_KVN).expect("retention message parses");
        assert_eq!(omm.comments.header, vec!["header comment"]);
        assert_eq!(omm.comments.metadata, vec!["metadata comment"]);
        assert_eq!(omm.comments.mean_elements, vec!["mean elements comment"]);
        assert_eq!(omm.comments.tle_parameters, vec![" indented TLE comment"]);
        assert_eq!(omm.comments.user_defined, vec!["user-defined comment"]);
        assert_eq!(omm.classification.as_deref(), Some("SBU"));
        assert_eq!(omm.message_id.as_deref(), Some("OMM 202013719185"));
        assert_eq!(omm.ref_frame_epoch.as_deref(), Some("2020-064T00:00:00"));
        assert_eq!(
            omm.spacecraft,
            Some(OmmSpacecraft {
                comments: vec!["spacecraft comment".to_string()],
                mass_kg: Some(2500.0),
                solar_rad_area_m2: None,
                solar_rad_coeff: None,
                drag_area_m2: None,
                drag_coeff: Some(2.2),
            })
        );
        let covariance = omm.covariance.as_ref().expect("covariance retained");
        assert_eq!(covariance.comments, vec!["covariance comment"]);
        assert_eq!(covariance.lower_triangle[3], -3.070007847730449e-04);
        assert_eq!(
            omm.user_defined[1],
            OmmUserDefined {
                parameter: "C3".to_string(),
                value: "29.376 [km**2/s**2]".to_string(),
            }
        );

        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);

        // GP JSON carries a single header comment as the COMMENT member and
        // no other comment. The lossless writer refuses the first comment it
        // cannot carry rather than drop it; asked to discard those, it writes
        // every other item.
        let mut without_comments = omm.clone();
        without_comments.comments = OmmComments {
            header: vec!["header comment".to_string()],
            ..OmmComments::default()
        };
        if let Some(spacecraft) = without_comments.spacecraft.as_mut() {
            spacecraft.comments.clear();
        }
        if let Some(covariance) = without_comments.covariance.as_mut() {
            covariance.comments.clear();
        }
        let refused: Result<String, OmmError> = Err(OmmError::UnwritableText {
            field: "COMMENT".to_string(),
            value: "metadata comment".to_string(),
            issue: TextIssue::CommentNotCarried,
        });
        assert_eq!(encode_json(&omm), refused);
        assert_eq!(
            encode_json_array(std::slice::from_ref(&omm)),
            Err(OmmError::InRecord {
                index: 0,
                source: Box::new(OmmError::UnwritableText {
                    field: "COMMENT".to_string(),
                    value: "metadata comment".to_string(),
                    issue: TextIssue::CommentNotCarried,
                }),
            })
        );
        assert_eq!(
            parse_json(&encode_json_discarding_comments(&omm).unwrap()).unwrap(),
            without_comments
        );
        assert_eq!(
            parse_json_array(
                &encode_json_array_discarding_comments(std::slice::from_ref(&omm)).unwrap()
            )
            .unwrap()
            .omms,
            vec![without_comments.clone()]
        );
        assert_eq!(
            encode_json_discarding_comments(&omm).unwrap(),
            encode_json(&without_comments).unwrap()
        );
    }

    #[test]
    fn trailing_comments_belong_to_the_last_block() {
        let omm = parse_kvn(&format!("{ISS_KVN}COMMENT trailing\n")).unwrap();
        assert_eq!(omm.comments.tle_parameters, vec!["trailing"]);
        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
    }

    #[test]
    fn empty_spacecraft_block_with_comment_round_trips() {
        let xml = ISS_XML.replace(
            "</meanElements>",
            "</meanElements><spacecraftParameters><COMMENT>only a comment</COMMENT></spacecraftParameters>",
        );
        let omm = parse_xml(&xml).unwrap();
        assert_eq!(
            omm.spacecraft,
            Some(OmmSpacecraft {
                comments: vec!["only a comment".to_string()],
                ..OmmSpacecraft::default()
            })
        );
        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);
        // GP JSON drops the comment only when asked to, and still states the
        // empty block as "MASS": null.
        assert_eq!(
            encode_json(&omm),
            Err(OmmError::UnwritableText {
                field: "COMMENT".to_string(),
                value: "only a comment".to_string(),
                issue: TextIssue::CommentNotCarried,
            })
        );
        assert_eq!(
            parse_json(&encode_json_discarding_comments(&omm).unwrap())
                .unwrap()
                .spacecraft,
            Some(OmmSpacecraft::default())
        );
    }

    #[test]
    fn units_are_checked_against_table_4_3_and_text_stays_verbatim() {
        assert_eq!(
            parse_kvn(&kvn_with_field("MEAN_MOTION", "15.49273435 [rad/s]")),
            Err(OmmError::UnitMismatch {
                field: "MEAN_MOTION".to_string(),
                unit: "rad/s".to_string(),
                expected: Some("rev/day"),
            })
        );
        assert_eq!(
            parse_kvn(&kvn_with_field("ECCENTRICITY", ".0004737 [n/a]")),
            Err(OmmError::UnitMismatch {
                field: "ECCENTRICITY".to_string(),
                unit: "n/a".to_string(),
                expected: None,
            })
        );
        let bstar = parse_kvn(&kvn_with_field("BSTAR", ".17172E-3 [1/[Earth radii]]"))
            .expect("table 4-3 BSTAR unit");
        assert_eq!(bstar.bstar, Some(0.00017172));
        let named = parse_kvn(&kvn_with_field("OBJECT_NAME", "ISS [ZARYA]")).unwrap();
        assert_eq!(named.object_name.as_deref(), Some("ISS [ZARYA]"));

        let xml = ISS_XML.replace("<MEAN_MOTION>", "<MEAN_MOTION units=\"rev/day\">");
        assert_eq!(parse_xml(&xml).unwrap(), parse_xml(ISS_XML).unwrap());
        let xml = ISS_XML.replace("<MEAN_MOTION>", "<MEAN_MOTION units=\"deg\">");
        assert_eq!(
            parse_xml(&xml),
            Err(OmmError::UnitMismatch {
                field: "MEAN_MOTION".to_string(),
                unit: "deg".to_string(),
                expected: Some("rev/day"),
            })
        );
    }

    #[test]
    fn conflicting_repeated_keywords_are_refused_by_name() {
        assert_eq!(
            parse_kvn(&format!("{ISS_KVN}MEAN_MOTION = 15.5\n")),
            Err(OmmError::DuplicateField {
                field: "MEAN_MOTION".to_string(),
                first: "15.49273435".to_string(),
                second: "15.5".to_string(),
            })
        );
        assert_eq!(
            parse_kvn(&format!("{ISS_KVN}CREATION_DATE = 2026-06-17T00:00:00\n")),
            Err(OmmError::DuplicateField {
                field: "CREATION_DATE".to_string(),
                first: String::new(),
                second: "2026-06-17T00:00:00".to_string(),
            })
        );
        assert_eq!(
            parse_kvn(&format!("{ISS_KVN}MEAN_MOTION = 15.49273435\n")).unwrap(),
            parse_kvn(ISS_KVN).unwrap()
        );
        assert_eq!(
            parse_kvn(&format!(
                "{ISS_KVN}USER_DEFINED_X = 1\nUSER_DEFINED_X = 2\n"
            )),
            Err(OmmError::DuplicateField {
                field: "USER_DEFINED_X".to_string(),
                first: "1".to_string(),
                second: "2".to_string(),
            })
        );

        let json = ISS_JSON_FIXTURE.replacen(
            "\"NORAD_CAT_ID\":25544",
            "\"NORAD_CAT_ID\":25544,\"NORAD_CAT_ID\":25545",
            1,
        );
        assert_eq!(
            parse_json(&json),
            Err(OmmError::DuplicateField {
                field: "NORAD_CAT_ID".to_string(),
                first: "25544".to_string(),
                second: "25545".to_string(),
            })
        );

        let xml = ISS_XML.replace(
            "<MEAN_MOTION>15.49273435</MEAN_MOTION>",
            "<MEAN_MOTION>15.49273435</MEAN_MOTION><MEAN_MOTION>15.5</MEAN_MOTION>",
        );
        assert_eq!(
            parse_xml(&xml),
            Err(OmmError::DuplicateField {
                field: "MEAN_MOTION".to_string(),
                first: "15.49273435".to_string(),
                second: "15.5".to_string(),
            })
        );
    }

    #[test]
    fn unknown_and_malformed_kvn_content_is_refused() {
        assert_eq!(
            parse_kvn(&format!("{ISS_KVN}FOO = 1\n")),
            Err(OmmError::UnknownField("FOO".to_string()))
        );
        // A keyword the tables do not define but that carries no value holds
        // nothing to keep.
        assert_eq!(
            parse_kvn(&format!("{ISS_KVN}FOO = \n")).unwrap(),
            parse_kvn(ISS_KVN).unwrap()
        );
        let with_garbage = format!("{ISS_KVN}not an assignment\n");
        let line = with_garbage.lines().count();
        assert_eq!(
            parse_kvn(&with_garbage),
            Err(OmmError::MalformedLine {
                line,
                text: "not an assignment".to_string(),
            })
        );
        assert_eq!(
            parse_kvn(&format!("{ISS_KVN}COV_REF_FRAME = TEME\nCX_X = 1\n")),
            Err(OmmError::MissingField("CY_X"))
        );
        // GP JSON extras that are not OMM keywords are not read.
        let json = ISS_JSON_FIXTURE.replacen("{", "{\"GP_ID\":1,", 1);
        assert_eq!(
            parse_json(&json).unwrap(),
            parse_json(ISS_JSON_FIXTURE).unwrap()
        );
    }

    #[test]
    fn xml_elements_are_scoped_to_their_message_and_block() {
        let first = omm_element(ISS_XML);
        let second = first
            .replace(
                "<NORAD_CAT_ID>25544</NORAD_CAT_ID>",
                "<NORAD_CAT_ID>25545</NORAD_CAT_ID>",
            )
            .replace(
                "<MEAN_ANOMALY>164.9702</MEAN_ANOMALY>",
                "<MEAN_ANOMALY>164.9702</MEAN_ANOMALY><GM>398600.4418</GM>",
            );
        let doc = format!("<?xml version=\"1.0\"?><ndm>{first}{second}</ndm>");
        assert_eq!(
            parse_xml(&doc),
            Err(OmmError::MultipleMessages { count: 2 })
        );
        let all = parse_xml_all(&doc).unwrap().omms;
        assert_eq!(
            all.iter().map(|omm| omm.norad_cat_id).collect::<Vec<_>>(),
            vec![Some(25544), Some(25545)]
        );
        assert_eq!(all[0], parse_xml(ISS_XML).unwrap());
        assert_eq!(all[1].gm_km3_s2, Some(398600.4418));

        // A field missing from the first message is not taken from the second.
        let without_id = first.replace("<OBJECT_ID>1998-067A</OBJECT_ID>", "");
        let doc = format!("<ndm>{without_id}{second}</ndm>");
        let all = parse_xml_all(&doc).unwrap().omms;
        assert_eq!(all[0].object_id, None);
        assert_eq!(all[1].object_id.as_deref(), Some("1998-067A"));

        // Nor is one missing from the second taken from the first. Table 4-3
        // requires NORAD_CAT_ID of an SGP/SGP4 element set; the reader keeps
        // what the message states and holds it absent, and SGP4, which does
        // not propagate with the catalog number, takes the elements without it.
        let without_catalog = first.replace("<NORAD_CAT_ID>25544</NORAD_CAT_ID>", "");
        let doc = format!("<ndm>{first}{without_catalog}</ndm>");
        let all = parse_xml_all(&doc).unwrap().omms;
        assert_eq!(all[0].norad_cat_id, Some(25544));
        assert_eq!(all[1].norad_cat_id, None);
        assert_eq!(all[1].to_element_set().unwrap().catalog_number, None);

        // A message the reader cannot read is skipped and reported with its
        // position and reason, as a bad GP JSON or CSV record is: EPOCH is
        // mandatory for every theory (table 4-3).
        let broken = first.replace("<EPOCH>2026-06-17T04:32:52.099296</EPOCH>", "");
        let doc = format!("<ndm>{first}{broken}</ndm>");
        assert_eq!(
            parse_xml_all(&doc),
            Ok(OmmArray {
                omms: vec![parse_xml(ISS_XML).unwrap()],
                skipped: vec![OmmSkippedRecord {
                    index: 1,
                    reason: OmmError::MissingField("EPOCH"),
                }],
            })
        );

        let misplaced = ISS_XML.replace("<EPOCH>", "<MASS>1</MASS><EPOCH>");
        assert_eq!(
            parse_xml(&misplaced),
            Err(OmmError::UnknownField("meanElements/MASS".to_string()))
        );
    }

    #[test]
    fn bridge_refuses_explicit_non_sgp4_metadata() {
        for (field, value) in [
            ("CENTER_NAME", "MARS"),
            ("REF_FRAME", "ICRF"),
            ("TIME_SYSTEM", "TAI"),
            ("MEAN_ELEMENT_THEORY", "DSST"),
            ("MEAN_ELEMENT_THEORY", "SGP4-XP"),
            ("MEAN_ELEMENT_THEORY", "SGP"),
        ] {
            let omm = parse_kvn(&kvn_with_field(field, value))
                .unwrap_or_else(|error| panic!("{field} = {value} must parse: {error}"));
            assert_eq!(
                omm.to_element_set(),
                Err(OmmError::IncompatibleMetadata {
                    field,
                    value: value.to_string(),
                }),
                "{field} = {value}"
            );
            let err =
                Satellite::from_omm(&omm).expect_err("incompatible metadata must not propagate");
            assert_eq!(
                err,
                Sgp4Error::InvalidInput {
                    field,
                    kind: Sgp4InputErrorKind::OutOfRange,
                }
            );
        }
    }

    #[test]
    fn bridge_accepts_sgp4_labels_and_unstated_metadata() {
        for theory in ["SGP4", "SGP/SGP4", "sgp4", "SDP4"] {
            let omm = parse_kvn(&kvn_with_field("MEAN_ELEMENT_THEORY", theory)).unwrap();
            omm.to_element_set()
                .unwrap_or_else(|error| panic!("{theory} must bridge: {error}"));
        }
        for field in [
            "CENTER_NAME",
            "REF_FRAME",
            "TIME_SYSTEM",
            "MEAN_ELEMENT_THEORY",
        ] {
            let absent = parse_kvn(&kvn_without_field(field)).unwrap();
            assert!(absent.to_element_set().is_ok(), "{field} absent");
            let blank = parse_kvn(&kvn_with_field(field, "")).unwrap();
            assert!(blank.to_element_set().is_ok(), "{field} blank");
        }
    }

    #[test]
    fn bridge_validates_a_mutated_epoch_calendar() {
        let base = parse_kvn(ISS_KVN).unwrap();
        type EpochMutation = fn(&mut OmmEpoch);
        let cases: [(EpochMutation, OmmInputErrorKind); 4] = [
            (
                |epoch| epoch.month = 13,
                OmmInputErrorKind::InvalidCivilDate,
            ),
            (|epoch| epoch.day = 31, OmmInputErrorKind::InvalidCivilDate),
            (|epoch| epoch.hour = 24, OmmInputErrorKind::InvalidCivilTime),
            (
                |epoch| epoch.second = 60,
                OmmInputErrorKind::InvalidCivilTime,
            ),
        ];
        for (mutate, kind) in cases {
            let mut omm = base.clone();
            mutate(&mut omm.epoch);
            assert_eq!(
                omm.to_element_set(),
                Err(OmmError::InvalidField {
                    field: "epoch",
                    kind,
                })
            );
        }
    }

    #[test]
    fn kvn_writer_refuses_text_that_would_read_back_differently() {
        let base = parse_kvn(ISS_KVN).unwrap();

        let mut omm = base.clone();
        omm.object_name = Some("ISS\nMEAN_MOTION = 1".to_string());
        assert_eq!(
            encode_kvn(&omm),
            Err(OmmError::UnwritableText {
                field: "OBJECT_NAME".to_string(),
                value: "ISS\nMEAN_MOTION = 1".to_string(),
                issue: TextIssue::LineBreak,
            })
        );

        let mut omm = base.clone();
        omm.object_name = Some(" ISS".to_string());
        assert!(matches!(
            encode_kvn(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::SurroundingWhitespace,
                ..
            })
        ));

        let mut omm = base.clone();
        omm.ccsds_omm_vers = Some(String::new());
        assert!(matches!(
            encode_kvn(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::Empty,
                ..
            })
        ));

        let mut omm = base.clone();
        omm.comments.header = vec!["trailing ".to_string()];
        assert!(matches!(
            encode_kvn(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::SurroundingWhitespace,
                ..
            })
        ));

        let mut omm = base.clone();
        omm.user_defined = vec![OmmUserDefined {
            parameter: "A=B".to_string(),
            value: "1".to_string(),
        }];
        assert!(matches!(
            encode_kvn(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::KeywordSeparator,
                ..
            })
        ));

        let mut omm = base.clone();
        omm.comments.user_defined = vec!["no parameter follows".to_string()];
        assert!(matches!(
            encode_kvn(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::DetachedComment,
                ..
            })
        ));

        let mut omm = base.clone();
        omm.mean_motion = Some(f64::NAN);
        assert_eq!(
            encode_kvn(&omm),
            Err(OmmError::InvalidField {
                field: "MEAN_MOTION",
                kind: OmmInputErrorKind::NonFinite,
            })
        );

        // Blank optional text is written blank and reads back as absent.
        let mut omm = base;
        omm.object_id = Some(String::new());
        let reparsed = parse_kvn(&encode_kvn(&omm).unwrap()).unwrap();
        assert_eq!(reparsed.object_id, None);
    }

    /// A DSST mean-element OMM (502.0-B-3 table 4-2 theory `DSST`): the
    /// semi-major axis in place of the mean motion and no TLE related
    /// parameters, which table 4-3 requires only for SGP/SGP4.
    const DSST_KVN: &str = "\
CCSDS_OMM_VERS = 3.0
CREATION_DATE = 2020-065T16:00:00
ORIGINATOR = NOAA
OBJECT_NAME = GOES 9
OBJECT_ID = 1995-025A
CENTER_NAME = EARTH
REF_FRAME = EME2000
TIME_SYSTEM = UTC
MEAN_ELEMENT_THEORY = DSST
EPOCH = 2020-064T10:34:41.4264
SEMI_MAJOR_AXIS = 42164.1 [km]
ECCENTRICITY = 0.0005013
INCLINATION = 3.0539 [deg]
RA_OF_ASC_NODE = 81.7939 [deg]
ARG_OF_PERICENTER = 249.2363 [deg]
MEAN_ANOMALY = 150.1602 [deg]
GM = 398600.4415 [km**3/s**2]
";

    /// An SGP4-XP OMM: table 4-3 replaces BSTAR with BTERM and
    /// MEAN_MOTION_DDOT with AGOM, both in m**2/kg.
    const SGP4_XP_KVN: &str = "\
CCSDS_OMM_VERS = 3.0
CREATION_DATE = 2020-065T16:00:00
ORIGINATOR = 18 SDS
OBJECT_NAME = GOES 9
OBJECT_ID = 1995-025A
CENTER_NAME = EARTH
REF_FRAME = TEME
TIME_SYSTEM = UTC
MEAN_ELEMENT_THEORY = SGP4-XP
EPOCH = 2020-064T10:34:41.4264
MEAN_MOTION = 1.00273272 [rev/day]
ECCENTRICITY = 0.0005013
INCLINATION = 3.0539 [deg]
RA_OF_ASC_NODE = 81.7939 [deg]
ARG_OF_PERICENTER = 249.2363 [deg]
MEAN_ANOMALY = 150.1602 [deg]
EPHEMERIS_TYPE = 4
CLASSIFICATION_TYPE = U
NORAD_CAT_ID = 23581
ELEMENT_SET_NO = 925
REV_AT_EPOCH = 4316
BTERM = 0.0015 [m**2/kg]
AGOM = 0.001 [m**2/kg]
";

    fn assert_round_trips_in_every_encoding(omm: &Omm) {
        let kvn = encode_kvn(omm).expect("KVN encode");
        assert_eq!(&parse_kvn(&kvn).unwrap(), omm, "KVN");
        let xml = encode_xml(omm).expect("XML encode");
        assert_eq!(&parse_xml(&xml).unwrap(), omm, "XML");
        assert_eq!(
            &parse_json(&encode_json(omm).unwrap()).unwrap(),
            omm,
            "JSON"
        );
        assert_eq!(
            &parse_csv(&encode_csv(std::slice::from_ref(omm)).unwrap()).unwrap(),
            omm,
            "CSV"
        );
    }

    #[test]
    fn dsst_omm_reads_writes_back_and_is_refused_by_the_bridge() {
        let omm = parse_kvn(DSST_KVN).expect("a DSST OMM parses");
        assert_eq!(omm.semi_major_axis_km, Some(42164.1));
        assert_eq!(omm.mean_motion, None);
        assert_eq!(omm.gm_km3_s2, Some(398600.4415));
        assert_eq!(omm.norad_cat_id, None);
        assert_eq!(omm.ephemeris_type, None);
        assert_eq!(omm.classification_type, None);
        assert_eq!(omm.bstar, None);

        // Nothing the message does not state is written back.
        let kvn = encode_kvn(&omm).unwrap();
        for absent in [
            "MEAN_MOTION ",
            "EPHEMERIS_TYPE",
            "CLASSIFICATION_TYPE",
            "NORAD_CAT_ID",
            "ELEMENT_SET_NO",
            "REV_AT_EPOCH",
            "BSTAR",
        ] {
            assert!(!kvn.contains(absent), "{absent} was not in the message");
        }
        assert!(kvn.contains("SEMI_MAJOR_AXIS = 42164.1\n"));
        assert!(!encode_xml(&omm).unwrap().contains("tleParameters"));
        assert_round_trips_in_every_encoding(&omm);

        assert_eq!(
            omm.to_element_set(),
            Err(OmmError::IncompatibleMetadata {
                field: "MEAN_ELEMENT_THEORY",
                value: "DSST".to_string(),
            })
        );
        // Without the theory label, the bridge names the element SGP4 lacks.
        let mut unlabelled = omm.clone();
        unlabelled.mean_element_theory = None;
        unlabelled.ref_frame = Some("TEME".to_string());
        assert_eq!(
            unlabelled.to_element_set(),
            Err(OmmError::MissingField("MEAN_MOTION"))
        );
    }

    #[test]
    fn sgp4_xp_omm_reads_writes_back_and_is_refused_by_the_bridge() {
        let omm = parse_kvn(SGP4_XP_KVN).expect("an SGP4-XP OMM parses");
        assert_eq!(omm.bterm_m2_kg, Some(0.0015));
        assert_eq!(omm.agom_m2_kg, Some(0.001));
        assert_eq!(omm.bstar, None);
        assert_eq!(omm.mean_motion_ddot, None);
        assert_eq!(omm.ephemeris_type, Some(4));
        assert_round_trips_in_every_encoding(&omm);
        let kvn = encode_kvn(&omm).unwrap();
        assert!(kvn.contains("BTERM = 0.0015\n"));
        assert!(kvn.contains("AGOM = 0.001\n"));

        assert_eq!(
            omm.to_element_set(),
            Err(OmmError::IncompatibleMetadata {
                field: "MEAN_ELEMENT_THEORY",
                value: "SGP4-XP".to_string(),
            })
        );
        let mut unlabelled = omm.clone();
        unlabelled.mean_element_theory = None;
        assert_eq!(
            unlabelled.to_element_set(),
            Err(OmmError::MissingField("BSTAR"))
        );

        assert_eq!(
            parse_kvn(&SGP4_XP_KVN.replace("[m**2/kg]\nAGOM", "[m**2/s]\nAGOM")),
            Err(OmmError::UnitMismatch {
                field: "BTERM".to_string(),
                unit: "m**2/s".to_string(),
                expected: Some("m**2/kg"),
            })
        );
    }

    #[test]
    fn an_omm_needs_a_semi_major_axis_or_a_mean_motion() {
        let without = DSST_KVN.replace("SEMI_MAJOR_AXIS = 42164.1 [km]\n", "");
        assert_eq!(
            parse_kvn(&without),
            Err(OmmError::MissingField("SEMI_MAJOR_AXIS or MEAN_MOTION"))
        );
    }

    #[test]
    fn gp_csv_writes_every_held_keyword_and_reads_it_back() {
        let omm = parse_kvn(RETAINED_KVN).unwrap();
        let mut without_comments = omm.clone();
        without_comments.comments = OmmComments {
            header: vec!["header comment".to_string()],
            ..OmmComments::default()
        };
        if let Some(spacecraft) = without_comments.spacecraft.as_mut() {
            spacecraft.comments.clear();
        }
        if let Some(covariance) = without_comments.covariance.as_mut() {
            covariance.comments.clear();
        }

        // GP CSV carries a single header comment in a COMMENT column and no
        // other comment, so the lossless writer refuses the first comment it
        // cannot carry rather than drop it.
        assert_eq!(
            encode_csv(std::slice::from_ref(&omm)),
            Err(OmmError::InRecord {
                index: 0,
                source: Box::new(OmmError::UnwritableText {
                    field: "COMMENT".to_string(),
                    value: "metadata comment".to_string(),
                    issue: TextIssue::CommentNotCarried,
                }),
            })
        );

        // Asked to discard those, it writes every keyword and nothing else is
        // lost.
        let csv = encode_csv_discarding_comments(std::slice::from_ref(&omm)).unwrap();
        assert!(csv.lines().next().unwrap().contains("USER_DEFINED_C3"));
        assert!(csv.lines().next().unwrap().contains(",COMMENT,"));
        assert_eq!(parse_csv(&csv).unwrap(), without_comments);
        assert_eq!(
            encode_csv(std::slice::from_ref(&without_comments)).unwrap(),
            csv
        );

        // The compact GP column set is unchanged for a CelesTrak record.
        let celestrak = parse_csv(ISS_CSV).unwrap();
        assert_eq!(
            encode_csv(std::slice::from_ref(&celestrak))
                .unwrap()
                .lines()
                .next()
                .unwrap(),
            ISS_CSV.lines().next().unwrap()
        );
    }

    #[test]
    fn gp_csv_keeps_each_record_parameter_order() {
        let base = parse_csv(ISS_CSV).unwrap();
        let parameter = |name: &str, value: &str| OmmUserDefined {
            parameter: name.to_string(),
            value: value.to_string(),
        };
        let mut first = base.clone();
        first.user_defined = vec![parameter("B", "2")];
        let mut second = base.clone();
        second.user_defined = vec![parameter("A", "1"), parameter("B", "2")];
        // Columns in first-appearance order, B then A, read the second record
        // back as B, A.
        let csv = encode_csv(&[first.clone(), second.clone()]).unwrap();
        assert!(csv
            .lines()
            .next()
            .unwrap()
            .ends_with(",USER_DEFINED_A,USER_DEFINED_B"));
        assert_eq!(
            parse_csv_array(&csv).unwrap().omms,
            vec![first.clone(), second.clone()]
        );

        // No column order keeps both A, B and B, A.
        let mut third = base;
        third.user_defined = vec![parameter("B", "2"), parameter("A", "1")];
        assert_eq!(
            encode_csv(&[first, second, third]),
            Err(OmmError::InRecord {
                index: 2,
                source: Box::new(OmmError::CsvColumnOrder {
                    first: "B".to_string(),
                    second: "A".to_string(),
                }),
            })
        );
    }

    #[test]
    fn single_record_readers_refuse_several_records() {
        let record = ISS_CSV.lines().nth(1).unwrap();
        let two_rows = format!("{ISS_CSV}\n{record}");
        assert_eq!(
            parse_csv(&two_rows),
            Err(OmmError::MultipleMessages { count: 2 })
        );
        assert_eq!(parse_csv_array(&two_rows).unwrap().omms.len(), 2);
        // A lone record the array reader would skip is refused with its reason.
        let short = format!("{}\nISS,1998-067A", ISS_CSV.lines().next().unwrap());
        assert_eq!(
            parse_csv(&short),
            Err(OmmError::CsvColumnCount {
                found: 2,
                expected: 17,
            })
        );

        const ISS_JSON: &str = include_str!("../../tests/fixtures/omm/25544.json");
        let object = ISS_JSON
            .trim()
            .trim_start_matches('[')
            .trim_end_matches(']');
        let two = format!("[{object},{object}]");
        assert_eq!(
            parse_json(&two),
            Err(OmmError::MultipleMessages { count: 2 })
        );
        assert_eq!(parse_json_array(&two).unwrap().omms.len(), 2);
        assert_eq!(parse_json(ISS_JSON), parse_json(object));
    }

    /// A Space-Track GP record: string-quoted numbers, catalog extras that are
    /// not OMM keywords, and the `COMMENT` member Space-Track writes.
    const SPACE_TRACK_JSON: &str = r#"{"CCSDS_OMM_VERS":"2.0","COMMENT":"GENERATED VIA SPACE-TRACK.ORG API","CREATION_DATE":"2026-06-17T06:26:03","ORIGINATOR":"18 SPCS","OBJECT_NAME":"ISS (ZARYA)","OBJECT_ID":"1998-067A","CENTER_NAME":"EARTH","REF_FRAME":"TEME","TIME_SYSTEM":"UTC","MEAN_ELEMENT_THEORY":"SGP4","EPOCH":"2026-06-17T04:32:52.099296","MEAN_MOTION":"15.49273435","ECCENTRICITY":"0.00047370","INCLINATION":"51.6332","RA_OF_ASC_NODE":"300.0813","ARG_OF_PERICENTER":"195.1146","MEAN_ANOMALY":"164.9702","EPHEMERIS_TYPE":"0","CLASSIFICATION_TYPE":"U","NORAD_CAT_ID":"25544","ELEMENT_SET_NO":"999","REV_AT_EPOCH":"57175","BSTAR":"0.00017172000000","MEAN_MOTION_DOT":"0.00009113","MEAN_MOTION_DDOT":"0.0000000000000","SEMIMAJOR_AXIS":"6795.123","PERIOD":"92.946","GP_ID":"123456789"}"#;

    #[test]
    fn space_track_comment_is_the_header_comment() {
        let omm = parse_json(SPACE_TRACK_JSON).unwrap();
        assert_eq!(
            omm.comments.header,
            vec!["GENERATED VIA SPACE-TRACK.ORG API"]
        );
        assert_eq!(omm.ccsds_omm_vers.as_deref(), Some("2.0"));

        // Every encoding carries it back: GP JSON as the member, GP CSV as a
        // column, KVN and XML as a header comment.
        let json = encode_json(&omm).unwrap();
        assert!(json.starts_with(
            r#"{"CCSDS_OMM_VERS":"2.0","COMMENT":"GENERATED VIA SPACE-TRACK.ORG API","#
        ));
        assert_eq!(parse_json(&json).unwrap(), omm);
        let csv = encode_csv(std::slice::from_ref(&omm)).unwrap();
        assert!(csv
            .lines()
            .next()
            .unwrap()
            .contains(",CCSDS_OMM_VERS,COMMENT,"));
        assert_eq!(parse_csv(&csv).unwrap(), omm);
        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);

        // A second header comment has no member to carry it.
        let mut two = omm.clone();
        two.comments.header.push("second".to_string());
        assert_eq!(
            encode_json(&two),
            Err(OmmError::UnwritableText {
                field: "COMMENT".to_string(),
                value: "second".to_string(),
                issue: TextIssue::CommentNotCarried,
            })
        );
        assert_eq!(
            parse_json(&encode_json_discarding_comments(&two).unwrap()).unwrap(),
            omm
        );

        // A repeated member is read once when equal and refused when it differs.
        let repeated = SPACE_TRACK_JSON.replacen(
            r#""CREATION_DATE""#,
            r#""COMMENT":"GENERATED VIA SPACE-TRACK.ORG API","CREATION_DATE""#,
            1,
        );
        assert_eq!(parse_json(&repeated).unwrap(), omm);
        let differing = SPACE_TRACK_JSON.replacen(
            r#""CREATION_DATE""#,
            r#""COMMENT":"other","CREATION_DATE""#,
            1,
        );
        assert_eq!(
            parse_json(&differing),
            Err(OmmError::DuplicateField {
                field: "COMMENT".to_string(),
                first: "GENERATED VIA SPACE-TRACK.ORG API".to_string(),
                second: "other".to_string(),
            })
        );
    }

    #[test]
    fn an_unstated_version_is_not_written() {
        let omm = parse_csv(ISS_CSV).unwrap();
        assert_eq!(omm.ccsds_omm_vers, None);
        assert!(!encode_kvn(&omm).unwrap().contains("CCSDS_OMM_VERS"));
        assert!(!encode_xml(&omm).unwrap().contains("CCSDS_OMM_VERS"));
        assert!(!encode_json(&omm).unwrap().contains("CCSDS_OMM_VERS"));
        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_json(&encode_json(&omm).unwrap()).unwrap(), omm);
    }

    #[test]
    fn csv_writer_refuses_what_the_csv_reader_would_not_return() {
        let base = parse_csv(ISS_CSV).unwrap();
        let in_record = |index: usize, source: OmmError| -> Result<String, OmmError> {
            Err(OmmError::InRecord {
                index,
                source: Box::new(source),
            })
        };

        // The reader trims every cell, so surrounding whitespace would be lost.
        let mut omm = base.clone();
        omm.object_name = Some(" ISS (ZARYA)".to_string());
        assert_eq!(
            encode_csv(&[base.clone(), omm]),
            in_record(
                1,
                OmmError::UnwritableText {
                    field: "OBJECT_NAME".to_string(),
                    value: " ISS (ZARYA)".to_string(),
                    issue: TextIssue::SurroundingWhitespace,
                }
            )
        );

        // A non-finite number was written as NaN, which the reader refuses, so
        // the record read back as skipped.
        let mut omm = base.clone();
        omm.bstar = Some(f64::NAN);
        assert_eq!(
            encode_csv(std::slice::from_ref(&omm)),
            in_record(
                0,
                OmmError::InvalidField {
                    field: "BSTAR",
                    kind: OmmInputErrorKind::NonFinite,
                }
            )
        );

        // The writer kept the first of two parameters with one name and dropped
        // the second.
        let mut omm = base.clone();
        omm.user_defined = vec![
            OmmUserDefined {
                parameter: "EARTH_MODEL".to_string(),
                value: "WGS-84".to_string(),
            },
            OmmUserDefined {
                parameter: "EARTH_MODEL".to_string(),
                value: "EGM-96".to_string(),
            },
        ];
        assert_eq!(
            encode_csv(std::slice::from_ref(&omm)),
            in_record(
                0,
                OmmError::UnwritableText {
                    field: "USER_DEFINED_EARTH_MODEL".to_string(),
                    value: "EARTH_MODEL".to_string(),
                    issue: TextIssue::RepeatedParameter,
                }
            )
        );

        // An empty user-defined value is an empty cell, which reads back as no
        // parameter.
        let mut omm = base.clone();
        omm.user_defined = vec![OmmUserDefined {
            parameter: "EARTH_MODEL".to_string(),
            value: String::new(),
        }];
        assert_eq!(
            encode_csv(std::slice::from_ref(&omm)),
            in_record(
                0,
                OmmError::UnwritableText {
                    field: "USER_DEFINED_EARTH_MODEL".to_string(),
                    value: String::new(),
                    issue: TextIssue::Empty,
                }
            )
        );

        // A month 13 names no instant the reader accepts.
        let mut omm = base.clone();
        omm.epoch.month = 13;
        assert_eq!(
            encode_csv(std::slice::from_ref(&omm)),
            in_record(
                0,
                OmmError::InvalidField {
                    field: "epoch",
                    kind: OmmInputErrorKind::InvalidCivilDate,
                }
            )
        );

        // An empty spacecraft block has no cell to state it; with comments
        // discarded, a block that held only comments goes with them.
        let mut omm = base.clone();
        omm.spacecraft = Some(OmmSpacecraft::default());
        let refused = in_record(0, OmmError::CsvEmptyBlock("spacecraft parameters"));
        assert_eq!(encode_csv(std::slice::from_ref(&omm)), refused);
        assert_eq!(
            encode_csv_discarding_comments(std::slice::from_ref(&omm)),
            refused
        );
        omm.spacecraft = Some(OmmSpacecraft {
            comments: vec!["spacecraft comment".to_string()],
            ..OmmSpacecraft::default()
        });
        assert_eq!(
            parse_csv(&encode_csv_discarding_comments(std::slice::from_ref(&omm)).unwrap())
                .unwrap(),
            base
        );

        // A line break inside a value is quoted and reads back unchanged.
        let mut omm = base;
        omm.object_name = Some("ISS\r\n(ZARYA)".to_string());
        assert_eq!(
            parse_csv(&encode_csv(std::slice::from_ref(&omm)).unwrap()).unwrap(),
            omm
        );
    }

    #[test]
    fn xml_writer_refuses_text_that_would_read_back_differently() {
        let base = parse_kvn(ISS_KVN).unwrap();

        let mut omm = base.clone();
        omm.object_name = Some("ISS ".to_string());
        assert!(matches!(
            encode_xml(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::SurroundingWhitespace,
                ..
            })
        ));

        let mut omm = base.clone();
        omm.object_id = Some("1998\u{1}067A".to_string());
        assert!(matches!(
            encode_xml(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::XmlIllegalCharacter,
                ..
            })
        ));

        let mut omm = base.clone();
        omm.gm_km3_s2 = Some(f64::INFINITY);
        assert_eq!(
            encode_xml(&omm),
            Err(OmmError::InvalidField {
                field: "GM",
                kind: OmmInputErrorKind::NonFinite,
            })
        );

        // An empty version would read back as absent.
        let mut omm = base.clone();
        omm.ccsds_omm_vers = Some(String::new());
        assert_eq!(
            encode_xml(&omm),
            Err(OmmError::UnwritableText {
                field: "CCSDS_OMM_VERS".to_string(),
                value: String::new(),
                issue: TextIssue::Empty,
            })
        );

        // A line break inside a value or comment reads back unchanged from XML,
        // so the XML writer writes it; a KVN line cannot hold one.
        let mut omm = base.clone();
        omm.object_name = Some("ISS\n(ZARYA)".to_string());
        omm.comments.header = vec!["first\nsecond".to_string()];
        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);
        assert!(matches!(
            encode_kvn(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::LineBreak,
                ..
            })
        ));

        // A comment keeps its leading whitespace in XML as in KVN.
        let mut omm = base;
        omm.comments.metadata = vec!["  indented".to_string()];
        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
    }

    #[test]
    fn json_writer_refuses_what_the_json_reader_would_not_return() {
        let base = parse_kvn(ISS_KVN).unwrap();

        // A JSON string carries line breaks and surrounding whitespace, and the
        // JSON reader keeps them.
        let mut omm = base.clone();
        omm.object_name = Some(" ISS\n(ZARYA) ".to_string());
        assert_eq!(parse_json(&encode_json(&omm).unwrap()).unwrap(), omm);

        // JSON has no non-finite number; it was previously written as null.
        let mut omm = base.clone();
        omm.mean_motion = Some(f64::NAN);
        assert_eq!(
            encode_json(&omm),
            Err(OmmError::InvalidField {
                field: "MEAN_MOTION",
                kind: OmmInputErrorKind::NonFinite,
            })
        );

        let mut omm = base.clone();
        omm.object_id = Some("1998\u{1}067A".to_string());
        assert_eq!(
            encode_json(&omm),
            Err(OmmError::UnwritableText {
                field: "OBJECT_ID".to_string(),
                value: "1998\u{1}067A".to_string(),
                issue: TextIssue::XmlIllegalCharacter,
            })
        );

        let mut omm = base.clone();
        omm.ccsds_omm_vers = Some(String::new());
        assert!(matches!(
            encode_json(&omm),
            Err(OmmError::UnwritableText {
                issue: TextIssue::Empty,
                ..
            })
        ));

        // A record the JSON writer refuses fails the array, naming its position.
        let mut bad = base.clone();
        bad.bstar = Some(f64::INFINITY);
        assert_eq!(
            encode_json_array(&[base, bad]),
            Err(OmmError::InRecord {
                index: 1,
                source: Box::new(OmmError::InvalidField {
                    field: "BSTAR",
                    kind: OmmInputErrorKind::NonFinite,
                }),
            })
        );
    }

    #[test]
    fn json_numbers_read_as_the_nearest_double() {
        // The shortest round-tripping form of a covariance value of figure
        // G-10, which the JSON reader read one unit in the last place away
        // before it rounded decimal numbers exactly.
        let text = "2.608899201686016e-10";
        let expected: f64 = text.parse().unwrap();
        let json = format!(
            r#"{{"OBJECT_NAME":"SAT","OBJECT_ID":"2026-001A","EPOCH":"2026-06-17T04:32:52.099296","MEAN_MOTION":15.49273435,"ECCENTRICITY":0.0004737,"INCLINATION":51.6332,"RA_OF_ASC_NODE":300.0813,"ARG_OF_PERICENTER":195.1146,"MEAN_ANOMALY":164.9702,"BSTAR":{text}}}"#
        );
        assert_eq!(
            parse_json(&json).unwrap().bstar.map(f64::to_bits),
            Some(expected.to_bits())
        );
    }

    #[test]
    fn writers_refuse_a_user_defined_parameter_given_twice() {
        let mut omm = parse_kvn(ISS_KVN).unwrap();
        // The readers keep one of two equal repeats and refuse two that differ.
        for second_value in ["WGS-84", "EGM-96"] {
            omm.user_defined = vec![
                OmmUserDefined {
                    parameter: "EARTH_MODEL".to_string(),
                    value: "WGS-84".to_string(),
                },
                OmmUserDefined {
                    parameter: "EARTH_MODEL".to_string(),
                    value: second_value.to_string(),
                },
            ];
            let expected: Result<String, OmmError> = Err(OmmError::UnwritableText {
                field: "USER_DEFINED_EARTH_MODEL".to_string(),
                value: "EARTH_MODEL".to_string(),
                issue: TextIssue::RepeatedParameter,
            });
            assert_eq!(encode_kvn(&omm), expected, "KVN, {second_value}");
            assert_eq!(encode_xml(&omm), expected, "XML, {second_value}");
            assert_eq!(encode_json(&omm), expected, "JSON, {second_value}");
        }
    }

    #[test]
    fn writers_refuse_an_epoch_the_readers_would_not_accept() {
        let base = parse_kvn(ISS_KVN).unwrap();

        let mut omm = base.clone();
        omm.epoch.month = 13;
        let expected: Result<String, OmmError> = Err(OmmError::InvalidField {
            field: "epoch",
            kind: OmmInputErrorKind::InvalidCivilDate,
        });
        assert_eq!(encode_kvn(&omm), expected, "KVN");
        assert_eq!(encode_xml(&omm), expected, "XML");
        assert_eq!(encode_json(&omm), expected, "JSON");

        let mut omm = base.clone();
        omm.epoch.microsecond = 1_000_000;
        assert_eq!(
            encode_kvn(&omm),
            Err(OmmError::InvalidField {
                field: "epoch.microsecond",
                kind: OmmInputErrorKind::OutOfRange,
            })
        );

        // A leap second is written under UTC, which labels it, and refused
        // under TAI, which has none.
        let mut omm = base;
        omm.epoch = OmmEpoch {
            year: 2016,
            month: 12,
            day: 31,
            hour: 23,
            minute: 59,
            second: 60,
            microsecond: 0,
            femtosecond: 0,
        };
        assert_eq!(omm.time_system.as_deref(), Some("UTC"));
        assert_eq!(parse_kvn(&encode_kvn(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_xml(&encode_xml(&omm).unwrap()).unwrap(), omm);
        assert_eq!(parse_json(&encode_json(&omm).unwrap()).unwrap(), omm);
        omm.time_system = Some("TAI".to_string());
        assert_eq!(
            encode_xml(&omm),
            Err(OmmError::InvalidField {
                field: "epoch",
                kind: OmmInputErrorKind::InvalidCivilTime,
            })
        );
    }

    #[test]
    fn xml_reads_the_semi_major_axis_and_sgp4_xp_terms_with_their_units() {
        let dsst = parse_kvn(DSST_KVN).unwrap();
        let xml = encode_xml(&dsst)
            .unwrap()
            .replace("<SEMI_MAJOR_AXIS>", "<SEMI_MAJOR_AXIS units=\"km\">");
        assert!(xml.contains("<SEMI_MAJOR_AXIS units=\"km\">42164.1<"));
        assert_eq!(parse_xml(&xml).unwrap(), dsst);

        let xp = parse_kvn(SGP4_XP_KVN).unwrap();
        let xml = encode_xml(&xp)
            .unwrap()
            .replace("<BTERM>", "<BTERM units=\"m**2/kg\">")
            .replace("<AGOM>", "<AGOM units=\"m**2/kg\">");
        assert!(xml.contains("<BTERM units=\"m**2/kg\">0.0015<"));
        assert!(xml.contains("<AGOM units=\"m**2/kg\">0.001<"));
        assert_eq!(parse_xml(&xml).unwrap(), xp);
        assert_eq!(
            parse_xml(&xml.replace("<AGOM units=\"m**2/kg\">", "<AGOM units=\"m**2\">")),
            Err(OmmError::UnitMismatch {
                field: "AGOM".to_string(),
                unit: "m**2".to_string(),
                expected: Some("m**2/kg"),
            })
        );
    }
}
