//! CCSDS Conjunction Data Message (CDM) KVN and XML format reader and writer.
//!
//! CDM (CCSDS 508.0-B-1) describes a predicted close approach between two space
//! objects: a header, the relative metadata/data (time of closest approach,
//! miss geometry, relative state vector, screening volume and collision
//! probability), and for each object its metadata, OD parameters, additional
//! parameters, state vector and RTN covariance of up to 9x9. The readers
//! retain every keyword of tables 3-1 to 3-4 and every comment, and the
//! writers write them back in both serializations:
//!
//! - KVN: lines end at CR, LF, CR LF or LF CR (6.2.2.4), and each line is
//!   blank, a comment, or a `keyword = value` assignment; any other line is
//!   refused by number. A comment belongs to the block of the keyword that
//!   follows it and is written at the start of that block (6.2.5.2); comments
//!   after the last keyword belong to that keyword's block. A numeric
//!   keyword's bracketed unit must be the unit tables 3-2 and 3-4 give it
//!   (6.2.4.1) and is refused by name otherwise; text values, such as object
//!   names, are kept verbatim.
//! - XML: a DOM parse via the `roxmltree` crate, reading each value from the
//!   element that owns it: the header, `relativeMetadataData` and its
//!   `relativeStateVector`, and each `segment`'s `metadata` and the
//!   logical-block elements of its `data` (4.3, tables 4-1 and 4-2). A `units`
//!   attribute must match the same tables (4.3.10). A `COMMENT` directly
//!   inside `<data>` belongs to the logical block that follows it, as it does
//!   in KVN. Encoding emits the CCSDS document layout through a controlled
//!   serializer (see [`encode_xml`]).
//!
//! A keyword or element the tables do not define at its position is refused
//! by name when it carries a value, and so is a keyword repeated in its scope
//! with a different value: the message for header and relative
//! metadata/data keywords, the object block otherwise. Header and relative
//! keywords are recognized anywhere in a KVN message, since none shares a name
//! with an object keyword. An object block is assigned by its `OBJECT` value
//! (table 3-3), or by its position when it states none.
//!
//! The hard-body radius, which 508.0-B-1 does not define, is read from the
//! NASA CARA `HBR = <value>` comment convention and from an `HBR` keyword or
//! element of the relative metadata/data; see [`CdmKvn::hard_body_radius_m`].
//!
//! Date/time fields cross this boundary as raw strings: resolving `TCA` /
//! `CREATION_DATE` to a concrete instant (and formatting one back) is the host's
//! job using its native date/time type, exactly as the TLE epoch is handled. This
//! module deliberately does not depend on the time-scale machinery, and applies
//! no calendar validation; it only carries the textual value through.

use crate::astro::ndm::{self, FieldMap, KvnLine, UnitMismatch};
use crate::astro::xml;
use crate::format::kvn::ConflictingField;
use crate::validate;
use roxmltree::{Document, Node};
use std::fmt;

/// Why a CDM writer refuses a text value.
pub use crate::astro::ndm::TextIssue;

const COMMENT: &str = "COMMENT";
/// Marker keyword whose value (`OBJECT1` / `OBJECT2`) opens an object block.
const OBJECT_MARKER: &str = "OBJECT";
const VERSION_KEY: &str = "CCSDS_CDM_VERS";
/// The hard-body radius keyword some producers add to the relative
/// metadata/data; 508.0-B-1 does not define it.
const HBR_KEY: &str = "HBR";
/// Element name wrapping one object's metadata/data block in the CDM XML schema.
const SEGMENT_TAG: &str = "segment";

/// Header keywords, CCSDS 508.0-B-1 table 3-1 (`COMMENT` is handled apart).
const HEADER_KEYS: &[&str] = &[
    VERSION_KEY,
    "CREATION_DATE",
    "ORIGINATOR",
    "MESSAGE_FOR",
    "MESSAGE_ID",
];
/// Relative metadata/data keywords, table 3-2.
const RELATIVE_KEYS: &[&str] = &[
    "TCA",
    "MISS_DISTANCE",
    "RELATIVE_SPEED",
    "RELATIVE_POSITION_R",
    "RELATIVE_POSITION_T",
    "RELATIVE_POSITION_N",
    "RELATIVE_VELOCITY_R",
    "RELATIVE_VELOCITY_T",
    "RELATIVE_VELOCITY_N",
    "START_SCREEN_PERIOD",
    "STOP_SCREEN_PERIOD",
    "SCREEN_VOLUME_FRAME",
    "SCREEN_VOLUME_SHAPE",
    "SCREEN_VOLUME_X",
    "SCREEN_VOLUME_Y",
    "SCREEN_VOLUME_Z",
    "SCREEN_ENTRY_TIME",
    "SCREEN_EXIT_TIME",
    "COLLISION_PROBABILITY",
    "COLLISION_PROBABILITY_METHOD",
];
/// Relative position keywords in R, T, N order; XML groups them with the
/// relative velocity in `relativeStateVector` (table 4-2).
const RELATIVE_POSITION_KEYS: [&str; 3] = [
    "RELATIVE_POSITION_R",
    "RELATIVE_POSITION_T",
    "RELATIVE_POSITION_N",
];
/// Relative velocity keywords in R, T, N order.
const RELATIVE_VELOCITY_KEYS: [&str; 3] = [
    "RELATIVE_VELOCITY_R",
    "RELATIVE_VELOCITY_T",
    "RELATIVE_VELOCITY_N",
];
/// Screening volume size keywords in X, Y, Z order.
const SCREEN_VOLUME_KEYS: [&str; 3] = ["SCREEN_VOLUME_X", "SCREEN_VOLUME_Y", "SCREEN_VOLUME_Z"];
/// Object metadata keywords after `OBJECT`, table 3-3 order.
const METADATA_KEYS: [&str; 20] = [
    "OBJECT_DESIGNATOR",
    "CATALOG_NAME",
    "OBJECT_NAME",
    "INTERNATIONAL_DESIGNATOR",
    "OBJECT_TYPE",
    "OPERATOR_CONTACT_POSITION",
    "OPERATOR_ORGANIZATION",
    "OPERATOR_PHONE",
    "OPERATOR_EMAIL",
    "EPHEMERIS_NAME",
    "COVARIANCE_METHOD",
    "MANEUVERABLE",
    "ORBIT_CENTER",
    "REF_FRAME",
    "GRAVITY_MODEL",
    "ATMOSPHERIC_MODEL",
    "N_BODY_PERTURBATIONS",
    "SOLAR_RAD_PRESSURE",
    "EARTH_TIDES",
    "INTRACK_THRUST",
];
/// OD parameter keywords, table 3-4.
const OD_KEYS: &[&str] = &[
    "TIME_LASTOB_START",
    "TIME_LASTOB_END",
    "RECOMMENDED_OD_SPAN",
    "ACTUAL_OD_SPAN",
    "OBS_AVAILABLE",
    "OBS_USED",
    "TRACKS_AVAILABLE",
    "TRACKS_USED",
    "RESIDUALS_ACCEPTED",
    "WEIGHTED_RMS",
];
/// Additional parameter keywords, table 3-4.
const ADDITIONAL_KEYS: [&str; 8] = [
    "AREA_PC",
    "AREA_DRG",
    "AREA_SRP",
    "MASS",
    "CD_AREA_OVER_MASS",
    "CR_AREA_OVER_MASS",
    "THRUST_ACCELERATION",
    "SEDR",
];
/// Keys of the six-component state vector, in CCSDS order (position then
/// velocity).
const STATE_KEYS: [&str; 6] = ["X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"];
/// Keys of the RTN position covariance lower triangle, in CCSDS order.
const COVARIANCE_KEYS: [&str; 6] = ["CR_R", "CT_R", "CT_T", "CN_R", "CN_T", "CN_N"];
/// Rows 4 to 6 of the RTN covariance lower triangle, completing the 6x6
/// position/velocity matrix (the six [`COVARIANCE_KEYS`] are its position 3x3
/// block). 508.0-B-1 table 3-4 makes all of them obligatory; a message that
/// gives some but not all is refused naming the first missing term, and a
/// message that gives none reads as a position-only covariance.
const VELOCITY_COVARIANCE_KEYS: [&str; 15] = [
    "CRDOT_R",
    "CRDOT_T",
    "CRDOT_N",
    "CRDOT_RDOT",
    "CTDOT_R",
    "CTDOT_T",
    "CTDOT_N",
    "CTDOT_RDOT",
    "CTDOT_TDOT",
    "CNDOT_R",
    "CNDOT_T",
    "CNDOT_N",
    "CNDOT_RDOT",
    "CNDOT_TDOT",
    "CNDOT_NDOT",
];
/// Row 7 of the 9x9 RTN covariance lower triangle, the drag (`DRG`) term.
const DRAG_COVARIANCE_KEYS: [&str; 7] = [
    "CDRG_R",
    "CDRG_T",
    "CDRG_N",
    "CDRG_RDOT",
    "CDRG_TDOT",
    "CDRG_NDOT",
    "CDRG_DRG",
];
/// Row 8, the solar radiation pressure (`SRP`) term.
const SRP_COVARIANCE_KEYS: [&str; 8] = [
    "CSRP_R",
    "CSRP_T",
    "CSRP_N",
    "CSRP_RDOT",
    "CSRP_TDOT",
    "CSRP_NDOT",
    "CSRP_DRG",
    "CSRP_SRP",
];
/// Row 9, the in-track thrust (`THR`) term.
const THRUST_COVARIANCE_KEYS: [&str; 9] = [
    "CTHR_R",
    "CTHR_T",
    "CTHR_N",
    "CTHR_RDOT",
    "CTHR_TDOT",
    "CTHR_NDOT",
    "CTHR_DRG",
    "CTHR_SRP",
    "CTHR_THR",
];

/// A two-object conjunction parsed from a CDM message (KVN or XML). Date/time
/// fields are the raw textual values; the host resolves them to its own instant
/// type. An absent keyword is `None` and is not written; a blank KVN text value
/// reads as empty text and is written back blank.
#[derive(Debug, Clone, PartialEq)]
pub struct CdmKvn {
    /// `CCSDS_CDM_VERS` text (table 3-1): the KVN keyword, or the XML `<cdm>`
    /// element's `version` attribute or a `CCSDS_CDM_VERS` element of its
    /// header. `None` when the message states no version, and then neither
    /// writer states one.
    pub ccsds_cdm_vers: Option<String>,
    /// Header comments (table 3-1), written after `CCSDS_CDM_VERS`.
    pub comments: Vec<String>,
    /// Raw CREATION_DATE text from the message. The readers do not validate it as a calendar value, and the host is responsible for interpreting and formatting the instant.
    pub creation_date: Option<String>,
    /// ORIGINATOR text copied from the message and emitted when present.
    pub originator: Option<String>,
    /// `MESSAGE_FOR` text (table 3-1): the spacecraft the message is provided
    /// for.
    pub message_for: Option<String>,
    /// MESSAGE_ID text copied without format validation and emitted in the corresponding header field when present.
    pub message_id: Option<String>,
    /// Relative metadata/data comments (table 3-2), written before `TCA`.
    pub relative_comments: Vec<String>,
    /// Raw TCA text; this module leaves date/time resolution and formatting to the host.
    pub tca: Option<String>,
    /// Finite MISS_DISTANCE value in meters. A stated KVN or XML unit must be `m`; encoders label the value in meters.
    pub miss_distance_m: Option<f64>,
    /// Finite RELATIVE_SPEED value in meters per second. A stated KVN or XML unit must be `m/s`; encoders label the value in meters per second.
    pub relative_speed_m_s: Option<f64>,
    /// `RELATIVE_POSITION_R`, `RELATIVE_POSITION_T` and `RELATIVE_POSITION_N`
    /// in metres: Object2's position relative to Object1's in Object1's RTN
    /// frame (table 3-2). Each component is optional on its own.
    pub relative_position_rtn_m: [Option<f64>; 3],
    /// `RELATIVE_VELOCITY_R`, `RELATIVE_VELOCITY_T` and `RELATIVE_VELOCITY_N`
    /// in metres per second, in the same frame.
    pub relative_velocity_rtn_m_s: [Option<f64>; 3],
    /// Raw `START_SCREEN_PERIOD` text.
    pub start_screen_period: Option<String>,
    /// Raw `STOP_SCREEN_PERIOD` text.
    pub stop_screen_period: Option<String>,
    /// `SCREEN_VOLUME_FRAME` text, such as `RTN` or `TVN`.
    pub screen_volume_frame: Option<String>,
    /// `SCREEN_VOLUME_SHAPE` text, such as `ELLIPSOID` or `BOX`.
    pub screen_volume_shape: Option<String>,
    /// `SCREEN_VOLUME_X`, `SCREEN_VOLUME_Y` and `SCREEN_VOLUME_Z` in metres,
    /// in `SCREEN_VOLUME_FRAME`. Each component is optional on its own.
    pub screen_volume_m: [Option<f64>; 3],
    /// Raw `SCREEN_ENTRY_TIME` text.
    pub screen_entry_time: Option<String>,
    /// Raw `SCREEN_EXIT_TIME` text.
    pub screen_exit_time: Option<String>,
    /// Optional finite COLLISION_PROBABILITY value, emitted without a units attribute.
    pub collision_probability: Option<f64>,
    /// COLLISION_PROBABILITY_METHOD text, emitted when present and XML-escaped in XML output.
    pub collision_probability_method: Option<String>,
    /// Optional hard-body radius in meters, which 508.0-B-1 does not define.
    /// It is read from an `HBR` keyword or element of the relative
    /// metadata/data, or from the first comment of the message, in writing
    /// order, that follows the NASA CARA `HBR = <value>` convention. Such a
    /// comment stays among the comments unless it reads exactly as the
    /// writers state a radius (`HBR = ` and the shortest round-tripping
    /// decimal), which the readers take as the radius alone. The writers state
    /// a radius the retained comments do not by adding that comment before the
    /// relative metadata/data comments, or before the header comments when one
    /// of them follows the convention, and refuse a retained comment that
    /// would read back as a radius when this is `None` with
    /// [`CdmError::HardBodyRadiusComment`].
    pub hard_body_radius_m: Option<f64>,
    /// The object of the KVN block or XML segment stating `OBJECT1`, or the
    /// first when it states none.
    pub object1: CdmObject,
    /// The object of the KVN block or XML segment stating `OBJECT2`, or the
    /// second when it states none.
    pub object2: CdmObject,
}

/// One object's CCSDS metadata block, OD and additional parameters, state
/// vector, and RTN covariance. Every metadata field is the verbatim textual
/// value (CCSDS enum fields such as `OBJECT_TYPE` and `MANEUVERABLE` are
/// carried as strings); absent fields are `None` and are not emitted on
/// encode.
#[derive(Debug, Clone, PartialEq)]
pub struct CdmObject {
    /// Metadata comments (table 3-3), written before `OBJECT`.
    pub metadata_comments: Vec<String>,
    /// Optional OBJECT_DESIGNATOR text copied from either serialization and emitted under the same key when present.
    pub object_designator: Option<String>,
    /// Optional CATALOG_NAME text copied from either serialization and emitted under the same key when present.
    pub catalog_name: Option<String>,
    /// Optional OBJECT_NAME text copied from either serialization; XML output escapes it.
    pub object_name: Option<String>,
    /// Optional INTERNATIONAL_DESIGNATOR text copied from either serialization and emitted under the same key when present.
    pub international_designator: Option<String>,
    /// OBJECT_TYPE is carried as text rather than parsed into an enum, then emitted when present.
    pub object_type: Option<String>,
    /// Optional OPERATOR_CONTACT_POSITION text emitted in the canonical metadata order when present.
    pub operator_contact_position: Option<String>,
    /// Optional OPERATOR_ORGANIZATION text copied by both readers and emitted when present.
    pub operator_organization: Option<String>,
    /// Optional OPERATOR_PHONE text emitted in the canonical metadata order when present.
    pub operator_phone: Option<String>,
    /// Optional OPERATOR_EMAIL text emitted in the canonical metadata order when present.
    pub operator_email: Option<String>,
    /// Optional EPHEMERIS_NAME text copied by both readers and emitted when present.
    pub ephemeris_name: Option<String>,
    /// Optional COVARIANCE_METHOD text copied by both readers and emitted when present.
    pub covariance_method: Option<String>,
    /// MANEUVERABLE is carried as text, preserving values such as YES or NO, and emitted when present.
    pub maneuverable: Option<String>,
    /// Optional ORBIT_CENTER text emitted in the canonical metadata order when present.
    pub orbit_center: Option<String>,
    /// Optional REF_FRAME text copied without frame conversion and emitted when present.
    pub ref_frame: Option<String>,
    /// Optional GRAVITY_MODEL text, including any model detail supplied by the message, emitted under the same key.
    pub gravity_model: Option<String>,
    /// Optional ATMOSPHERIC_MODEL text emitted in the canonical metadata order when present.
    pub atmospheric_model: Option<String>,
    /// Optional N_BODY_PERTURBATIONS text preserving the listed perturbing bodies and emitted when present.
    pub n_body_perturbations: Option<String>,
    /// SOLAR_RAD_PRESSURE is carried as text rather than interpreted as a boolean, then emitted when present.
    pub solar_rad_pressure: Option<String>,
    /// EARTH_TIDES is carried as text rather than interpreted as a boolean, then emitted when present.
    pub earth_tides: Option<String>,
    /// INTRACK_THRUST is carried as text, preserving values such as NO, and emitted when present.
    pub intrack_thrust: Option<String>,
    /// The OD parameters logical block (table 3-4).
    pub od_parameters: CdmOdParameters,
    /// The additional parameters logical block (table 3-4).
    pub additional_parameters: CdmAdditionalParameters,
    /// State vector comments, written before `X`.
    pub state_comments: Vec<String>,
    /// Position `(x, y, z)` then velocity `(x_dot, y_dot, z_dot)`.
    pub state: ((f64, f64, f64), (f64, f64, f64)),
    /// Covariance comments, written before `CR_R`.
    pub covariance_comments: Vec<String>,
    /// RTN position covariance lower triangle as read: CR_R, CT_R, CT_T, CN_R,
    /// CN_T, CN_N. Each value is finite; the covariance rows are held and
    /// written as stated, and [`CdmObject::to_covariance_rtn`] validates the
    /// matrix for a consumer that needs a covariance.
    pub covariance_rtn: [f64; 6],
    /// Rows 4 to 6 of the RTN covariance lower triangle, completing the 6x6
    /// matrix, in table 3-4 order `CRDOT_R` to `CNDOT_NDOT`, or `None` when the
    /// producer carried only the position block. A block with some but not
    /// all of its 15 terms is refused rather than read.
    pub velocity_covariance_rtn: Option<[f64; 15]>,
    /// Row 7 of the 9x9 RTN covariance lower triangle, `CDRG_R` to `CDRG_DRG`
    /// (table 3-4, 5.2.8), or `None` when not given. A row with some but not
    /// all of its terms is refused naming the first missing term.
    pub drag_covariance_rtn: Option<[f64; 7]>,
    /// Row 8, `CSRP_R` to `CSRP_SRP`, under the same rule.
    pub srp_covariance_rtn: Option<[f64; 8]>,
    /// Row 9, `CTHR_R` to `CTHR_THR`, under the same rule.
    pub thrust_covariance_rtn: Option<[f64; 9]>,
}

impl CdmObject {
    /// The RTN covariance as the symmetric matrix of the rows the object
    /// holds, row-major: 3x3 for the position block alone, 6x6 with the
    /// velocity rows, and 7x7 to 9x9 with rows 7 to 9 (508.0-B-1 table 3-4),
    /// validated positive semidefinite within the covariance tolerance.
    ///
    /// The readers and writers hold the values as stated: producers print them
    /// to a few significant digits, as the standard's own example does, so a
    /// nearly singular matrix can fall short of positive semidefinite only
    /// through that rounding, although every value reads correctly. A
    /// consumer that needs a covariance asks here. A matrix that is not
    /// positive semidefinite is refused with [`CdmError::InvalidField`] for
    /// `covariance_rtn` with `NotPositive`, and a row given while a row before
    /// it is absent, which leaves its values no place in the matrix, with
    /// `Missing` naming the absent row.
    pub fn to_covariance_rtn(&self) -> Result<Vec<Vec<f64>>, CdmError> {
        let rows: [(Option<&[f64]>, &'static str); 4] = [
            (
                self.velocity_covariance_rtn
                    .as_ref()
                    .map(|row| row.as_slice()),
                "velocity_covariance_rtn",
            ),
            (
                self.drag_covariance_rtn.as_ref().map(|row| row.as_slice()),
                "drag_covariance_rtn",
            ),
            (
                self.srp_covariance_rtn.as_ref().map(|row| row.as_slice()),
                "srp_covariance_rtn",
            ),
            (
                self.thrust_covariance_rtn
                    .as_ref()
                    .map(|row| row.as_slice()),
                "thrust_covariance_rtn",
            ),
        ];
        let mut lower = self.covariance_rtn.to_vec();
        let mut dimension = 3;
        let mut absent: Option<&'static str> = None;
        for (row, field) in rows {
            match (row, absent) {
                (Some(_), Some(missing)) => {
                    return Err(CdmError::InvalidField {
                        field: missing,
                        kind: CdmInputErrorKind::Missing,
                    })
                }
                (Some(values), None) => {
                    lower.extend_from_slice(values);
                    dimension = if dimension == 3 { 6 } else { dimension + 1 };
                }
                (None, None) => absent = Some(field),
                (None, Some(_)) => {}
            }
        }
        match dimension {
            3 => validated_symmetric::<3>(&lower),
            6 => validated_symmetric::<6>(&lower),
            7 => validated_symmetric::<7>(&lower),
            8 => validated_symmetric::<8>(&lower),
            _ => validated_symmetric::<9>(&lower),
        }
    }
}

/// The symmetric `N`x`N` matrix whose lower triangle, row by row, is `lower`,
/// validated positive semidefinite.
fn validated_symmetric<const N: usize>(lower: &[f64]) -> Result<Vec<Vec<f64>>, CdmError> {
    let mut matrix = [[0.0_f64; N]; N];
    let mut values = lower.iter().copied();
    for row in 0..N {
        for col in 0..=row {
            let value = values.next().unwrap_or(f64::NAN);
            matrix[row][col] = value;
            matrix[col][row] = value;
        }
    }
    validate::validate_covariance_psd(&matrix, "covariance_rtn").map_err(map_cdm_field_error)?;
    Ok(matrix.iter().map(|row| row.to_vec()).collect())
}

/// The OD parameters of one CDM object (508.0-B-1 table 3-4). Every item is
/// optional.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CdmOdParameters {
    /// Comments at the start of the block.
    pub comments: Vec<String>,
    /// Raw `TIME_LASTOB_START` text.
    pub time_lastob_start: Option<String>,
    /// Raw `TIME_LASTOB_END` text.
    pub time_lastob_end: Option<String>,
    /// `RECOMMENDED_OD_SPAN`, days.
    pub recommended_od_span_d: Option<f64>,
    /// `ACTUAL_OD_SPAN`, days.
    pub actual_od_span_d: Option<f64>,
    /// `OBS_AVAILABLE`, a count.
    pub obs_available: Option<u64>,
    /// `OBS_USED`, a count.
    pub obs_used: Option<u64>,
    /// `TRACKS_AVAILABLE`, a count.
    pub tracks_available: Option<u64>,
    /// `TRACKS_USED`, a count.
    pub tracks_used: Option<u64>,
    /// `RESIDUALS_ACCEPTED`, percent.
    pub residuals_accepted_pct: Option<f64>,
    /// `WEIGHTED_RMS`, dimensionless.
    pub weighted_rms: Option<f64>,
}

/// The additional parameters of one CDM object (508.0-B-1 table 3-4). Every
/// item is optional.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CdmAdditionalParameters {
    /// Comments at the start of the block.
    pub comments: Vec<String>,
    /// `AREA_PC`, m^2.
    pub area_pc_m2: Option<f64>,
    /// `AREA_DRG`, m^2.
    pub area_drg_m2: Option<f64>,
    /// `AREA_SRP`, m^2.
    pub area_srp_m2: Option<f64>,
    /// `MASS`, kg.
    pub mass_kg: Option<f64>,
    /// `CD_AREA_OVER_MASS`, m^2/kg.
    pub cd_area_over_mass_m2_kg: Option<f64>,
    /// `CR_AREA_OVER_MASS`, m^2/kg.
    pub cr_area_over_mass_m2_kg: Option<f64>,
    /// `THRUST_ACCELERATION`, m/s^2.
    pub thrust_acceleration_m_s2: Option<f64>,
    /// `SEDR`, W/kg.
    pub sedr_w_kg: Option<f64>,
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

/// One keyword value of a block, typed as its table defines it.
#[derive(Debug, Clone, Copy)]
enum CdmValue<'a> {
    Text(Option<&'a str>),
    Number(Option<f64>),
    Count(Option<u64>),
}

impl CdmValue<'_> {
    fn is_present(&self) -> bool {
        match self {
            Self::Text(value) => value.is_some(),
            Self::Number(value) => value.is_some(),
            Self::Count(value) => value.is_some(),
        }
    }
}

/// The relative metadata/data values of `cdm` in table 3-2 order.
fn relative_items(cdm: &CdmKvn) -> [(&'static str, CdmValue<'_>); 20] {
    let [position_r, position_t, position_n] = cdm.relative_position_rtn_m;
    let [velocity_r, velocity_t, velocity_n] = cdm.relative_velocity_rtn_m_s;
    let [volume_x, volume_y, volume_z] = cdm.screen_volume_m;
    [
        ("TCA", CdmValue::Text(cdm.tca.as_deref())),
        ("MISS_DISTANCE", CdmValue::Number(cdm.miss_distance_m)),
        ("RELATIVE_SPEED", CdmValue::Number(cdm.relative_speed_m_s)),
        ("RELATIVE_POSITION_R", CdmValue::Number(position_r)),
        ("RELATIVE_POSITION_T", CdmValue::Number(position_t)),
        ("RELATIVE_POSITION_N", CdmValue::Number(position_n)),
        ("RELATIVE_VELOCITY_R", CdmValue::Number(velocity_r)),
        ("RELATIVE_VELOCITY_T", CdmValue::Number(velocity_t)),
        ("RELATIVE_VELOCITY_N", CdmValue::Number(velocity_n)),
        (
            "START_SCREEN_PERIOD",
            CdmValue::Text(cdm.start_screen_period.as_deref()),
        ),
        (
            "STOP_SCREEN_PERIOD",
            CdmValue::Text(cdm.stop_screen_period.as_deref()),
        ),
        (
            "SCREEN_VOLUME_FRAME",
            CdmValue::Text(cdm.screen_volume_frame.as_deref()),
        ),
        (
            "SCREEN_VOLUME_SHAPE",
            CdmValue::Text(cdm.screen_volume_shape.as_deref()),
        ),
        ("SCREEN_VOLUME_X", CdmValue::Number(volume_x)),
        ("SCREEN_VOLUME_Y", CdmValue::Number(volume_y)),
        ("SCREEN_VOLUME_Z", CdmValue::Number(volume_z)),
        (
            "SCREEN_ENTRY_TIME",
            CdmValue::Text(cdm.screen_entry_time.as_deref()),
        ),
        (
            "SCREEN_EXIT_TIME",
            CdmValue::Text(cdm.screen_exit_time.as_deref()),
        ),
        (
            "COLLISION_PROBABILITY",
            CdmValue::Number(cdm.collision_probability),
        ),
        (
            "COLLISION_PROBABILITY_METHOD",
            CdmValue::Text(cdm.collision_probability_method.as_deref()),
        ),
    ]
}

/// The OD parameter values in table 3-4 order.
fn od_items(od: &CdmOdParameters) -> [(&'static str, CdmValue<'_>); 10] {
    [
        (
            "TIME_LASTOB_START",
            CdmValue::Text(od.time_lastob_start.as_deref()),
        ),
        (
            "TIME_LASTOB_END",
            CdmValue::Text(od.time_lastob_end.as_deref()),
        ),
        (
            "RECOMMENDED_OD_SPAN",
            CdmValue::Number(od.recommended_od_span_d),
        ),
        ("ACTUAL_OD_SPAN", CdmValue::Number(od.actual_od_span_d)),
        ("OBS_AVAILABLE", CdmValue::Count(od.obs_available)),
        ("OBS_USED", CdmValue::Count(od.obs_used)),
        ("TRACKS_AVAILABLE", CdmValue::Count(od.tracks_available)),
        ("TRACKS_USED", CdmValue::Count(od.tracks_used)),
        (
            "RESIDUALS_ACCEPTED",
            CdmValue::Number(od.residuals_accepted_pct),
        ),
        ("WEIGHTED_RMS", CdmValue::Number(od.weighted_rms)),
    ]
}

/// The additional parameter values in table 3-4 order.
fn additional_items(additional: &CdmAdditionalParameters) -> [(&'static str, CdmValue<'_>); 8] {
    let values = [
        additional.area_pc_m2,
        additional.area_drg_m2,
        additional.area_srp_m2,
        additional.mass_kg,
        additional.cd_area_over_mass_m2_kg,
        additional.cr_area_over_mass_m2_kg,
        additional.thrust_acceleration_m_s2,
        additional.sedr_w_kg,
    ];
    let mut items = [("", CdmValue::Number(None)); 8];
    for ((item, key), value) in items.iter_mut().zip(ADDITIONAL_KEYS).zip(values) {
        *item = (key, CdmValue::Number(value));
    }
    items
}

/// The covariance terms `object` holds, in table 3-4 order.
fn covariance_values(object: &CdmObject) -> Vec<(&'static str, f64)> {
    let mut values: Vec<(&'static str, f64)> = COVARIANCE_KEYS
        .into_iter()
        .zip(object.covariance_rtn)
        .collect();
    if let Some(row) = object.velocity_covariance_rtn {
        values.extend(VELOCITY_COVARIANCE_KEYS.into_iter().zip(row));
    }
    if let Some(row) = object.drag_covariance_rtn {
        values.extend(DRAG_COVARIANCE_KEYS.into_iter().zip(row));
    }
    if let Some(row) = object.srp_covariance_rtn {
        values.extend(SRP_COVARIANCE_KEYS.into_iter().zip(row));
    }
    if let Some(row) = object.thrust_covariance_rtn {
        values.extend(THRUST_COVARIANCE_KEYS.into_iter().zip(row));
    }
    values
}

/// Failure modes of the CDM readers and writers. The message strings are the
/// historical public contract surfaced by the Elixir binding.
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
    /// A keyword occurred more than once in its scope with different values: a
    /// header or relative-metadata keyword in the message, or an object keyword
    /// in its object block.
    DuplicateField {
        /// The repeated keyword.
        field: String,
        /// The value of its first occurrence.
        first: String,
        /// The later value that differs.
        second: String,
    },
    /// A stated unit contradicts the unit CCSDS 508.0-B-1 tables 3-2 and 3-4
    /// give the keyword (6.2.4.1 for KVN, 4.3.10 for XML).
    UnitMismatch {
        /// The keyword whose value carried the unit.
        field: String,
        /// The stated unit.
        unit: String,
        /// The table unit, or `None` for a dimensionless or text keyword.
        expected: Option<&'static str>,
    },
    /// The message holds more than the two objects a CDM describes (508.0-B-1
    /// 3.1.2, 4.2.2).
    UnexpectedObjectCount(usize),
    /// An XML document holds more than one CDM.
    MultipleMessages {
        /// The number of CDM messages in the document.
        count: usize,
    },
    /// A KVN keyword or XML element that tables 3-1 to 3-4 and 4-1 to 4-2 do
    /// not define at that position and that carries a value. XML elements are
    /// named with their parent element.
    UnknownField(String),
    /// A KVN line that is not blank, a comment, or a `keyword = value`
    /// assignment (508.0-B-1 6.3.1.1).
    MalformedLine {
        /// One-based line number.
        line: usize,
        /// The trimmed line text.
        text: String,
    },
    /// An object block's `OBJECT` value is neither `OBJECT1` nor `OBJECT2`
    /// (table 3-3).
    UnknownObject(String),
    /// Two object blocks describe the same object: both state it, or one
    /// states the object the other's position assigns it.
    RepeatedObject(String),
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
    /// A comment following the `HBR = <value>` convention would read back as a
    /// hard-body radius, and [`CdmKvn::hard_body_radius_m`] holds none.
    HardBodyRadiusComment {
        /// The comment text.
        comment: String,
    },
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
            CdmError::DuplicateField {
                field,
                first,
                second,
            } => write!(
                f,
                "CDM keyword {field} occurs with different values {first:?} and {second:?}"
            ),
            CdmError::UnitMismatch {
                field,
                unit,
                expected,
            } => write!(
                f,
                "CDM keyword {field} states unit [{unit}], expected {}",
                ndm::expected_unit_label(*expected)
            ),
            CdmError::UnexpectedObjectCount(count) => {
                write!(f, "CDM holds {count} objects; a CDM describes two")
            }
            CdmError::MultipleMessages { count } => {
                write!(f, "XML holds {count} CDM messages; the reader reads one")
            }
            CdmError::UnknownField(name) => write!(f, "CDM has no keyword {name}"),
            CdmError::MalformedLine { line, text } => write!(
                f,
                "CDM line {line} is not a comment or keyword assignment: {text:?}"
            ),
            CdmError::UnknownObject(object) => {
                write!(f, "CDM OBJECT {object:?} is neither OBJECT1 nor OBJECT2")
            }
            CdmError::RepeatedObject(object) => {
                write!(f, "CDM describes {object} in more than one block")
            }
            CdmError::UnwritableText {
                field,
                value,
                issue,
            } => write!(f, "CDM {field} value {value:?} {issue}"),
            CdmError::HardBodyRadiusComment { comment } => write!(
                f,
                "CDM comment {comment:?} would read back as a hard-body radius the message does not hold"
            ),
        }
    }
}

impl std::error::Error for CdmError {}

/// The logical groups of CDM keywords: the header (table 3-1), the relative
/// metadata/data (table 3-2), and each object's metadata (table 3-3) and
/// data blocks (table 3-4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CdmBlock {
    Header,
    Relative,
    Metadata,
    OdParameters,
    AdditionalParameters,
    StateVector,
    Covariance,
}

impl CdmBlock {
    fn of_keyword(key: &str) -> Option<Self> {
        if HEADER_KEYS.contains(&key) {
            Some(Self::Header)
        } else if RELATIVE_KEYS.contains(&key) || key == HBR_KEY {
            Some(Self::Relative)
        } else if key == OBJECT_MARKER || METADATA_KEYS.contains(&key) {
            Some(Self::Metadata)
        } else if OD_KEYS.contains(&key) {
            Some(Self::OdParameters)
        } else if ADDITIONAL_KEYS.contains(&key) {
            Some(Self::AdditionalParameters)
        } else if STATE_KEYS.contains(&key) {
            Some(Self::StateVector)
        } else if COVARIANCE_KEYS.contains(&key)
            || VELOCITY_COVARIANCE_KEYS.contains(&key)
            || DRAG_COVARIANCE_KEYS.contains(&key)
            || SRP_COVARIANCE_KEYS.contains(&key)
            || THRUST_COVARIANCE_KEYS.contains(&key)
        {
            Some(Self::Covariance)
        } else {
            None
        }
    }

    /// The data-section block an XML logical-block element holds (table 4-1).
    fn of_data_element(name: &str) -> Option<Self> {
        match name {
            "odParameters" => Some(Self::OdParameters),
            "additionalParameters" => Some(Self::AdditionalParameters),
            "stateVector" => Some(Self::StateVector),
            "covarianceMatrix" => Some(Self::Covariance),
            _ => None,
        }
    }

    fn is_message(self) -> bool {
        matches!(self, Self::Header | Self::Relative)
    }
}

/// Header and relative metadata/data values and comments before the typed
/// mapping.
#[derive(Default)]
struct MessageFields {
    pairs: Vec<(String, String)>,
    header_comments: Vec<String>,
    relative_comments: Vec<String>,
}

/// One object's values and comments before the typed mapping.
#[derive(Default)]
struct ObjectFields {
    pairs: Vec<(String, String)>,
    metadata_comments: Vec<String>,
    od_comments: Vec<String>,
    additional_comments: Vec<String>,
    state_comments: Vec<String>,
    covariance_comments: Vec<String>,
}

impl ObjectFields {
    fn comments_mut(&mut self, block: CdmBlock) -> &mut Vec<String> {
        match block {
            CdmBlock::OdParameters => &mut self.od_comments,
            CdmBlock::AdditionalParameters => &mut self.additional_comments,
            CdmBlock::StateVector => &mut self.state_comments,
            CdmBlock::Covariance => &mut self.covariance_comments,
            // Header and relative keywords never belong to an object.
            CdmBlock::Metadata | CdmBlock::Header | CdmBlock::Relative => {
                &mut self.metadata_comments
            }
        }
    }

    /// The object the block states with `OBJECT`; a blank or absent value
    /// states none.
    fn label(&self) -> Result<Option<String>, CdmError> {
        let labels: Vec<&str> = self
            .pairs
            .iter()
            .filter(|(key, _)| key == OBJECT_MARKER)
            .map(|(_, value)| value.as_str())
            .collect();
        match labels.split_first() {
            None => Ok(None),
            Some((first, rest)) => match rest.iter().find(|value| *value != first) {
                Some(second) => Err(CdmError::DuplicateField {
                    field: OBJECT_MARKER.to_string(),
                    first: (*first).to_string(),
                    second: (*second).to_string(),
                }),
                None => Ok(Some((*first).to_string()).filter(|label| !label.is_empty())),
            },
        }
    }
}

/// Where a KVN keyword's value and the comments before it belong: a message
/// block, or a block of the object at an index.
#[derive(Debug, Clone, Copy)]
struct Target {
    block: CdmBlock,
    object: Option<usize>,
}

/// The comment list of `target`.
fn comments_at<'a>(
    message: &'a mut MessageFields,
    objects: &'a mut [ObjectFields],
    target: Target,
) -> &'a mut Vec<String> {
    match (
        target.block,
        target.object.and_then(|index| objects.get_mut(index)),
    ) {
        (CdmBlock::Header, _) => &mut message.header_comments,
        (CdmBlock::Relative, _) => &mut message.relative_comments,
        (block, Some(object)) => object.comments_mut(block),
        // An object block always names its object.
        (_, None) => &mut message.relative_comments,
    }
}

/// Parse a CDM in KVN format.
///
/// Lines end at CR, LF, CR LF or LF CR (508.0-B-1 6.2.2.4). Blank lines are
/// ignored; any other line must be a comment or an assignment
/// ([`CdmError::MalformedLine`]), and an assignment must name a keyword of
/// tables 3-1 to 3-4 unless its value is blank ([`CdmError::UnknownField`]).
/// An `OBJECT` line opens an object block, and object keywords before the
/// first one are refused. A comment belongs to the block of the keyword after
/// it. Date/time fields are returned verbatim for the host to resolve;
/// presence/format checks on them (and on `MESSAGE_ID`) are the host's
/// concern. An object block missing any state component is rejected with
/// [`CdmError::IncompleteStateVector`], and a message with more than two
/// object blocks with [`CdmError::UnexpectedObjectCount`]. Every position
/// covariance component is required, each further covariance row group must
/// be complete or absent, and every accepted numeric value must be finite. A
/// bracketed unit must match 508.0-B-1 tables 3-2 and 3-4.
pub fn parse_kvn(text: &str) -> Result<CdmKvn, CdmError> {
    let lines = ndm::kvn_lines(text);
    let markers = lines
        .iter()
        .filter(|line| {
            matches!(
                ndm::classify(line),
                KvnLine::Assignment {
                    key: OBJECT_MARKER,
                    ..
                }
            )
        })
        .count();
    if markers > 2 {
        return Err(CdmError::UnexpectedObjectCount(markers));
    }

    let mut message = MessageFields::default();
    let mut objects: Vec<ObjectFields> = Vec::new();
    let mut pending: Vec<String> = Vec::new();
    let mut last = Target {
        block: CdmBlock::Header,
        object: None,
    };
    for (index, line) in lines.iter().enumerate() {
        match ndm::classify(line) {
            KvnLine::Blank => {}
            KvnLine::Comment(comment) => pending.push(comment.to_string()),
            KvnLine::Other(other) => {
                return Err(CdmError::MalformedLine {
                    line: index + 1,
                    text: other.to_string(),
                })
            }
            KvnLine::Assignment { key, value } => {
                let Some(block) = CdmBlock::of_keyword(key) else {
                    // A keyword the tables do not define is refused when it
                    // carries a value; a blank one holds nothing to keep.
                    if value.is_empty() {
                        continue;
                    }
                    return Err(CdmError::UnknownField(key.to_string()));
                };
                let value = kvn_value(key, value)?;
                let target = if block.is_message() {
                    message.pairs.push((key.to_string(), value.to_string()));
                    Target {
                        block,
                        object: None,
                    }
                } else {
                    if key == OBJECT_MARKER {
                        objects.push(ObjectFields::default());
                    }
                    let Some(object) = objects.len().checked_sub(1) else {
                        // An object keyword before the first OBJECT line
                        // belongs to neither object.
                        if value.is_empty() {
                            continue;
                        }
                        return Err(CdmError::UnknownField(key.to_string()));
                    };
                    if let Some(fields) = objects.get_mut(object) {
                        fields.pairs.push((key.to_string(), value.to_string()));
                    }
                    Target {
                        block,
                        object: Some(object),
                    }
                };
                comments_at(&mut message, &mut objects, target).append(&mut pending);
                last = target;
            }
        }
    }
    comments_at(&mut message, &mut objects, last).append(&mut pending);
    assemble(message, objects, true)
}

/// A KVN value with any unit tables 3-2 and 3-4 give its keyword checked and
/// removed; a text value is kept verbatim.
fn kvn_value<'a>(key: &str, value: &'a str) -> Result<&'a str, CdmError> {
    match cdm_unit(key) {
        Some(allowed) => {
            let (number, unit) = ndm::split_unit(value);
            ndm::check_unit(unit, allowed).map_err(|mismatch| unit_mismatch(key, mismatch))?;
            Ok(number)
        }
        None => Ok(value),
    }
}

/// Encode a [`CdmKvn`] back to KVN text.
///
/// Each block's comments are written at its start and each present keyword in
/// table order; an absent value is not written. The date/time fields are taken
/// as already-formatted strings (the host owns the instant-to-string
/// conversion). Numeric values are written with their shortest round-tripping
/// decimal form and their table unit, so a re-parse recovers the exact same
/// bits; the output is therefore round-trip faithful rather than
/// byte-identical to any one producer.
///
/// Comments of the relative metadata/data, OD parameters or additional
/// parameters with no value of their block after them are written before a
/// blank `MISS_DISTANCE`, `RECOMMENDED_OD_SPAN` or `AREA_PC`, which the reader
/// takes as absent, so they read back in their block. Header comments are
/// written before `CCSDS_CDM_VERS` when no other header keyword follows it,
/// where they still read back as header comments.
///
/// What would not read back unchanged is refused: a non-finite number with
/// [`CdmError::InvalidField`]; with [`CdmError::UnwritableText`], text with a
/// line break or surrounding whitespace, a comment with a line break or
/// trailing whitespace, and header comments with no header keyword to precede
/// ([`TextIssue::DetachedComment`]), since a blank header keyword reads back as
/// empty text; and a comment that would read back as a hard-body radius when
/// the message holds none with [`CdmError::HardBodyRadiusComment`].
pub fn encode_kvn(cdm: &CdmKvn) -> Result<String, CdmError> {
    validate_cdm(cdm)?;
    let statement = hard_body_radius_statement(cdm)?;
    let mut out = KvnWriter::default();

    let header = header_items(cdm);
    let header_follows = header.iter().any(|(_, value)| value.is_some());
    let comments = header_comments(cdm, &statement);
    match cdm.ccsds_cdm_vers.as_deref() {
        Some(version) if header_follows => {
            out.text(VERSION_KEY, version)?;
            out.comments(&comments)?;
        }
        Some(version) => {
            out.comments(&comments)?;
            out.text(VERSION_KEY, version)?;
        }
        None if header_follows => out.comments(&comments)?,
        None => refuse_detached(&comments)?,
    }
    for (key, value) in header {
        out.item(key, CdmValue::Text(value))?;
    }

    let relative = relative_items(cdm);
    if let HbrStatement::Relative(comment) = &statement {
        out.comments(std::slice::from_ref(comment))?;
    }
    out.comments(&cdm.relative_comments)?;
    out.anchor(&cdm.relative_comments, &relative, "MISS_DISTANCE");
    for (key, value) in relative {
        out.item(key, value)?;
    }

    for (object, name) in [(&cdm.object1, "OBJECT1"), (&cdm.object2, "OBJECT2")] {
        encode_object_kvn(&mut out, object, name)?;
    }
    Ok(out.lines.join("\n"))
}

/// The header keywords after `CCSDS_CDM_VERS`, in table 3-1 order.
fn header_items(cdm: &CdmKvn) -> [(&'static str, Option<&str>); 4] {
    [
        ("CREATION_DATE", cdm.creation_date.as_deref()),
        ("ORIGINATOR", cdm.originator.as_deref()),
        ("MESSAGE_FOR", cdm.message_for.as_deref()),
        ("MESSAGE_ID", cdm.message_id.as_deref()),
    ]
}

fn encode_object_kvn(out: &mut KvnWriter, object: &CdmObject, name: &str) -> Result<(), CdmError> {
    out.comments(&object.metadata_comments)?;
    out.lines.push(format!("{OBJECT_MARKER} = {name}"));
    for (key, value) in object_metadata_pairs(object) {
        out.item(key, CdmValue::Text(value.as_deref()))?;
    }
    for (comments, items, anchor) in [
        (
            &object.od_parameters.comments,
            od_items(&object.od_parameters).to_vec(),
            "RECOMMENDED_OD_SPAN",
        ),
        (
            &object.additional_parameters.comments,
            additional_items(&object.additional_parameters).to_vec(),
            "AREA_PC",
        ),
    ] {
        out.comments(comments)?;
        out.anchor(comments, &items, anchor);
        for (key, value) in items {
            out.item(key, value)?;
        }
    }
    out.comments(&object.state_comments)?;
    for (key, value) in STATE_KEYS.into_iter().zip(state_values(object)) {
        out.item(key, CdmValue::Number(Some(value)))?;
    }
    out.comments(&object.covariance_comments)?;
    for (key, value) in covariance_values(object) {
        out.item(key, CdmValue::Number(Some(value)))?;
    }
    Ok(())
}

/// Line assembly for [`encode_kvn`], checking that every value reads back.
#[derive(Default)]
struct KvnWriter {
    lines: Vec<String>,
}

impl KvnWriter {
    /// Write a present value; an absent one is not written.
    fn item(&mut self, key: &str, value: CdmValue<'_>) -> Result<(), CdmError> {
        match value {
            CdmValue::Text(Some(text)) => self.text(key, text)?,
            CdmValue::Number(Some(number)) => self.lines.push(numeric_line(key, number)),
            CdmValue::Count(Some(count)) => self.lines.push(format!("{key} = {count}")),
            CdmValue::Text(None) | CdmValue::Number(None) | CdmValue::Count(None) => {}
        }
        Ok(())
    }

    /// A text value, which reads back verbatim apart from its surrounding
    /// whitespace; an empty one is written blank and reads back empty.
    fn text(&mut self, key: &str, value: &str) -> Result<(), CdmError> {
        if let Some(issue) = ndm::text::kvn_value_issue(value) {
            return Err(unwritable(key, value, issue));
        }
        self.lines.push(format!("{key} = {value}"));
        Ok(())
    }

    fn comments(&mut self, comments: &[String]) -> Result<(), CdmError> {
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

    /// Give comments of a block that holds no value a keyword of that block to
    /// precede: `numeric_key` written blank, which the reader takes as absent,
    /// so the comments read back in their block rather than the next one.
    fn anchor(&mut self, comments: &[String], items: &[(&str, CdmValue<'_>)], numeric_key: &str) {
        if !comments.is_empty() && !items.iter().any(|(_, value)| value.is_present()) {
            self.lines.push(format!("{numeric_key} ="));
        }
    }
}

/// A numeric KVN line labelled with the table unit, if the keyword has one.
fn numeric_line(key: &str, value: f64) -> String {
    match unit_label(key) {
        Some(unit) => format!("{key} = {} [{unit}]", fmt_num(value)),
        None => format!("{key} = {}", fmt_num(value)),
    }
}

/// The unit the tables give `key`, or `None` for a dimensionless or text
/// keyword.
fn unit_label(key: &str) -> Option<&'static str> {
    cdm_unit(key).and_then(|units| units.first().copied())
}

/// Refuse comments of a block the writer writes no keyword of, which would
/// read back as comments of the next block.
fn refuse_detached(comments: &[String]) -> Result<(), CdmError> {
    match comments.first() {
        Some(comment) => Err(unwritable(COMMENT, comment, TextIssue::DetachedComment)),
        None => Ok(()),
    }
}

fn unwritable(field: &str, value: &str, issue: TextIssue) -> CdmError {
    CdmError::UnwritableText {
        field: field.to_string(),
        value: value.to_string(),
        issue,
    }
}

fn state_values(object: &CdmObject) -> [f64; 6] {
    let ((x, y, z), (x_dot, y_dot, z_dot)) = object.state;
    [x, y, z, x_dot, y_dot, z_dot]
}

/// Parse a CDM in XML format.
///
/// Parses the document with `roxmltree` (a real XML DOM reader: the `<?xml?>`
/// declaration, comments, namespaces, entity escaping, and encoding are handled
/// by the library, not by string scanning), then reads each value from the
/// element that owns it: the `<header>`, the `<relativeMetadataData>` with its
/// `<relativeStateVector>`, and each `<segment>`'s `<metadata>` and the
/// logical-block elements of its `<data>`. The message may be the root element
/// or sit inside an `<ndm>` combined instantiation (505.0-B-3 4.11); a
/// document holding more than one CDM is refused with
/// [`CdmError::MultipleMessages`], and more than two segments with
/// [`CdmError::UnexpectedObjectCount`]. An element the tables do not define at
/// its position is refused by name when it holds a value. A `units` attribute
/// must match 508.0-B-1 tables 3-2 and 3-4 (4.3.10). The covariance row groups
/// beyond the position block are recovered when complete; a partial group
/// (e.g. a lone `CRDOT_R`) is refused naming the first missing term.
/// Date/time fields are returned verbatim for the host to resolve, matching
/// [`parse_kvn`]; text that is not well-formed XML is rejected with
/// [`CdmError::MalformedXml`] and an object block missing any state component
/// with [`CdmError::IncompleteStateVector`]. Every covariance component is
/// required and every accepted numeric value must be finite.
pub fn parse_xml(text: &str) -> Result<CdmKvn, CdmError> {
    let doc = Document::parse(text).map_err(|e| CdmError::MalformedXml(e.to_string()))?;
    let messages = ndm::message_elements(&doc, "cdm");
    let root = match messages.as_slice() {
        [] => doc.root_element(),
        [message] => *message,
        _ => {
            return Err(CdmError::MultipleMessages {
                count: messages.len(),
            })
        }
    };

    let mut message = MessageFields::default();
    if let Some(version) = root
        .attribute("version")
        .map(str::trim)
        .filter(|version| !version.is_empty())
    {
        message
            .pairs
            .push((VERSION_KEY.to_string(), version.to_string()));
    }
    let mut segments = Vec::new();
    for child in ndm::element_children(root) {
        match child.tag_name().name() {
            "header" => read_xml_header(child, &mut message)?,
            "body" => {
                for part in ndm::element_children(child) {
                    read_xml_body_part(part, "body", &mut message, &mut segments)?;
                }
            }
            _ => read_xml_body_part(child, root.tag_name().name(), &mut message, &mut segments)?,
        }
    }
    if segments.len() > 2 {
        return Err(CdmError::UnexpectedObjectCount(segments.len()));
    }
    let objects = segments
        .into_iter()
        .map(read_xml_segment)
        .collect::<Result<Vec<_>, _>>()?;
    assemble(message, objects, false)
}

/// Read a child of `<body>`: the relative metadata/data or an object segment.
/// Both are also taken directly under the message element.
fn read_xml_body_part<'a, 'input>(
    node: Node<'a, 'input>,
    parent: &str,
    message: &mut MessageFields,
    segments: &mut Vec<Node<'a, 'input>>,
) -> Result<(), CdmError> {
    match node.tag_name().name() {
        "relativeMetadataData" => read_xml_relative(node, message),
        SEGMENT_TAG => {
            segments.push(node);
            Ok(())
        }
        other => unknown_element(parent, other, node),
    }
}

/// Read `<header>` (508.0-B-1 4.3.4).
fn read_xml_header(node: Node, message: &mut MessageFields) -> Result<(), CdmError> {
    for child in ndm::element_children(node) {
        let name = child.tag_name().name();
        if name == COMMENT {
            message.header_comments.push(xml_comment(child)?);
        } else if CdmBlock::of_keyword(name) == Some(CdmBlock::Header) {
            push_xml_leaf(child, &mut message.pairs)?;
        } else {
            unknown_element("header", name, child)?;
        }
    }
    Ok(())
}

/// Read `<relativeMetadataData>` and its `<relativeStateVector>` (4.3.6,
/// table 4-2). The relative state keywords are also taken directly in
/// `<relativeMetadataData>`.
fn read_xml_relative(node: Node, message: &mut MessageFields) -> Result<(), CdmError> {
    for child in ndm::element_children(node) {
        let name = child.tag_name().name();
        if name == COMMENT {
            message.relative_comments.push(xml_comment(child)?);
        } else if name == "relativeStateVector" {
            for leaf in ndm::element_children(child) {
                let leaf_name = leaf.tag_name().name();
                if leaf_name == COMMENT {
                    message.relative_comments.push(xml_comment(leaf)?);
                } else if RELATIVE_POSITION_KEYS.contains(&leaf_name)
                    || RELATIVE_VELOCITY_KEYS.contains(&leaf_name)
                {
                    push_xml_leaf(leaf, &mut message.pairs)?;
                } else {
                    unknown_element(name, leaf_name, leaf)?;
                }
            }
        } else if CdmBlock::of_keyword(name) == Some(CdmBlock::Relative) {
            push_xml_leaf(child, &mut message.pairs)?;
        } else {
            unknown_element("relativeMetadataData", name, child)?;
        }
    }
    Ok(())
}

/// Read one `<segment>`: its `<metadata>` and `<data>` (4.3.7, 4.3.8).
fn read_xml_segment(segment: Node) -> Result<ObjectFields, CdmError> {
    let mut object = ObjectFields::default();
    for child in ndm::element_children(segment) {
        match child.tag_name().name() {
            "metadata" => {
                for leaf in ndm::element_children(child) {
                    let name = leaf.tag_name().name();
                    if name == COMMENT {
                        object.metadata_comments.push(xml_comment(leaf)?);
                    } else if CdmBlock::of_keyword(name) == Some(CdmBlock::Metadata) {
                        push_xml_leaf(leaf, &mut object.pairs)?;
                    } else {
                        unknown_element("metadata", name, leaf)?;
                    }
                }
            }
            "data" => read_xml_data(child, &mut object)?,
            other => unknown_element(SEGMENT_TAG, other, child)?,
        }
    }
    Ok(object)
}

/// Read one `<data>`: its logical-block elements (table 4-1) and the comments
/// before them, which belong to the block that follows, as in KVN. A data
/// keyword outside its block element still names its block. Comments after
/// the last block belong to it.
fn read_xml_data(data: Node, object: &mut ObjectFields) -> Result<(), CdmError> {
    let mut pending: Vec<String> = Vec::new();
    let mut last = CdmBlock::StateVector;
    for child in ndm::element_children(data) {
        let name = child.tag_name().name();
        if name == COMMENT {
            pending.push(xml_comment(child)?);
            continue;
        }
        if let Some(block) = CdmBlock::of_data_element(name) {
            object.comments_mut(block).append(&mut pending);
            for leaf in ndm::element_children(child) {
                let leaf_name = leaf.tag_name().name();
                if leaf_name == COMMENT {
                    object.comments_mut(block).push(xml_comment(leaf)?);
                } else if CdmBlock::of_keyword(leaf_name) == Some(block) {
                    push_xml_leaf(leaf, &mut object.pairs)?;
                } else {
                    unknown_element(name, leaf_name, leaf)?;
                }
            }
            last = block;
            continue;
        }
        match CdmBlock::of_keyword(name) {
            Some(
                block @ (CdmBlock::OdParameters
                | CdmBlock::AdditionalParameters
                | CdmBlock::StateVector
                | CdmBlock::Covariance),
            ) => {
                object.comments_mut(block).append(&mut pending);
                push_xml_leaf(child, &mut object.pairs)?;
                last = block;
            }
            _ => unknown_element("data", name, child)?,
        }
    }
    object.comments_mut(last).append(&mut pending);
    Ok(())
}

/// The text of a `COMMENT` element; one holding an element is refused by name.
fn xml_comment(node: Node) -> Result<String, CdmError> {
    match ndm::nested_element(node) {
        Some(path) => Err(CdmError::UnknownField(path)),
        None => Ok(ndm::comment_text(node)),
    }
}

/// Record a keyword element. A `units` attribute must match the table unit
/// (4.3.10); a text or dimensionless keyword takes none. A keyword element
/// holding elements is refused, naming the first.
fn push_xml_leaf(node: Node, pairs: &mut Vec<(String, String)>) -> Result<(), CdmError> {
    let name = node.tag_name().name();
    if let Some(nested) = ndm::element_children(node).first() {
        return Err(CdmError::UnknownField(format!(
            "{name}/{}",
            nested.tag_name().name()
        )));
    }
    if let Some(unit) = ndm::units_attribute(node) {
        const NO_UNIT: &[&str] = &[];
        ndm::check_unit(Some(unit.trim()), cdm_unit(name).unwrap_or(NO_UNIT))
            .map_err(|mismatch| unit_mismatch(name, mismatch))?;
    }
    pairs.push((name.to_string(), ndm::leaf_text(node)));
    Ok(())
}

/// Refuse an element the tables do not define at its position when it holds a
/// value; an empty one holds nothing to keep.
fn unknown_element(parent: &str, name: &str, node: Node) -> Result<(), CdmError> {
    if ndm::carries_data(node) {
        Err(CdmError::UnknownField(format!("{parent}/{name}")))
    } else {
        Ok(())
    }
}

/// Encode a [`CdmKvn`] to a CCSDS 508.0-B-1 CDM XML document.
///
/// The date/time fields are taken as already-formatted strings (the host owns
/// the instant-to-string conversion), and numeric values use the shortest
/// round-tripping decimal form, so the output is round-trip faithful rather than
/// byte-identical to any one producer. String values are XML-escaped; each
/// block's comments open its element, and an absent value is not written.
///
/// This is a controlled straight-line serializer (no parsing, no tag scanning)
/// rather than a generic streaming-writer dependency: the output is a fixed,
/// documented CCSDS layout (`cdm > header/body > segment > metadata/data`) whose
/// element nesting and `units` attributes are the inter-system exchange contract,
/// and every interpolated value is escaped via `xml::escape`. The matching
/// reader is the vetted `roxmltree` DOM parser in [`parse_xml`].
///
/// What would not read back unchanged is refused: a non-finite number with
/// [`CdmError::InvalidField`]; with [`CdmError::UnwritableText`], an empty
/// text value or version, which reads back absent, text with surrounding
/// whitespace, which the reader trims, a comment with trailing whitespace,
/// and a character XML 1.0 cannot carry; and a comment that would read back as
/// a hard-body radius when the message holds none with
/// [`CdmError::HardBodyRadiusComment`]. A line break inside a value or comment
/// reads back unchanged and is written.
pub fn encode_xml(cdm: &CdmKvn) -> Result<String, CdmError> {
    validate_cdm(cdm)?;
    let statement = hard_body_radius_statement(cdm)?;
    let mut out = XmlWriter::default();
    out.raw(0, r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    match cdm.ccsds_cdm_vers.as_deref() {
        Some(version) => {
            if let Some(issue) = ndm::text::xml_required_issue(version) {
                return Err(unwritable(VERSION_KEY, version, issue));
            }
            out.raw(
                0,
                &format!(
                    r#"<cdm id="CCSDS_CDM_VERS" version="{}">"#,
                    ndm::text::escape_attribute(version)
                ),
            );
        }
        None => out.raw(0, "<cdm>"),
    }

    out.raw(2, "<header>");
    out.comments(4, &header_comments(cdm, &statement))?;
    for (key, value) in header_items(cdm) {
        out.item(4, key, CdmValue::Text(value))?;
    }
    out.raw(2, "</header>");
    out.raw(2, "<body>");

    out.raw(4, "<relativeMetadataData>");
    if let HbrStatement::Relative(comment) = &statement {
        out.comments(6, std::slice::from_ref(comment))?;
    }
    out.comments(6, &cdm.relative_comments)?;
    let relative = relative_items(cdm);
    let in_state_vector =
        |key: &str| RELATIVE_POSITION_KEYS.contains(&key) || RELATIVE_VELOCITY_KEYS.contains(&key);
    let state_vector_present = relative
        .iter()
        .any(|(key, value)| in_state_vector(key) && value.is_present());
    let mut state_vector_open = false;
    for (key, value) in relative {
        if in_state_vector(key) {
            if state_vector_present && !state_vector_open {
                out.raw(6, "<relativeStateVector>");
                state_vector_open = true;
            }
            out.item(8, key, value)?;
            continue;
        }
        if state_vector_open {
            out.raw(6, "</relativeStateVector>");
            state_vector_open = false;
        }
        out.item(6, key, value)?;
    }
    out.raw(4, "</relativeMetadataData>");

    for (object, name) in [(&cdm.object1, "OBJECT1"), (&cdm.object2, "OBJECT2")] {
        encode_object_xml(&mut out, object, name)?;
    }
    out.raw(2, "</body>");
    out.raw(0, "</cdm>");
    Ok(out.lines.join("\n"))
}

fn encode_object_xml(out: &mut XmlWriter, object: &CdmObject, name: &str) -> Result<(), CdmError> {
    out.raw(4, "<segment>");
    out.raw(6, "<metadata>");
    out.comments(8, &object.metadata_comments)?;
    out.raw(8, &format!("<{OBJECT_MARKER}>{name}</{OBJECT_MARKER}>"));
    for (key, value) in object_metadata_pairs(object) {
        out.item(8, key, CdmValue::Text(value.as_deref()))?;
    }
    out.raw(6, "</metadata>");
    out.raw(6, "<data>");
    for (tag, comments, items) in [
        (
            "odParameters",
            &object.od_parameters.comments,
            od_items(&object.od_parameters).to_vec(),
        ),
        (
            "additionalParameters",
            &object.additional_parameters.comments,
            additional_items(&object.additional_parameters).to_vec(),
        ),
    ] {
        if comments.is_empty() && !items.iter().any(|(_, value)| value.is_present()) {
            continue;
        }
        out.raw(8, &format!("<{tag}>"));
        out.comments(10, comments)?;
        for (key, value) in items {
            out.item(10, key, value)?;
        }
        out.raw(8, &format!("</{tag}>"));
    }
    out.raw(8, "<stateVector>");
    out.comments(10, &object.state_comments)?;
    for (key, value) in STATE_KEYS.into_iter().zip(state_values(object)) {
        out.item(10, key, CdmValue::Number(Some(value)))?;
    }
    out.raw(8, "</stateVector>");
    out.raw(8, "<covarianceMatrix>");
    out.comments(10, &object.covariance_comments)?;
    for (key, value) in covariance_values(object) {
        out.item(10, key, CdmValue::Number(Some(value)))?;
    }
    out.raw(8, "</covarianceMatrix>");
    out.raw(6, "</data>");
    out.raw(4, "</segment>");
    Ok(())
}

/// Element assembly for [`encode_xml`], checking that every value reads back.
#[derive(Default)]
struct XmlWriter {
    lines: Vec<String>,
}

impl XmlWriter {
    fn raw(&mut self, indent: usize, line: &str) {
        self.lines.push(format!("{:indent$}{line}", ""));
    }

    /// Write a present value; an absent one is not written.
    fn item(&mut self, indent: usize, key: &str, value: CdmValue<'_>) -> Result<(), CdmError> {
        match value {
            CdmValue::Text(Some(text)) => {
                if let Some(issue) = ndm::text::xml_required_issue(text) {
                    return Err(unwritable(key, text, issue));
                }
                self.raw(indent, &format!("<{key}>{}</{key}>", xml::escape(text)));
            }
            CdmValue::Number(Some(number)) => {
                let line = match unit_label(key) {
                    Some(unit) => format!(r#"<{key} units="{unit}">{}</{key}>"#, fmt_num(number)),
                    None => format!("<{key}>{}</{key}>", fmt_num(number)),
                };
                self.raw(indent, &line);
            }
            CdmValue::Count(Some(count)) => self.raw(indent, &format!("<{key}>{count}</{key}>")),
            CdmValue::Text(None) | CdmValue::Number(None) | CdmValue::Count(None) => {}
        }
        Ok(())
    }

    fn comments(&mut self, indent: usize, comments: &[String]) -> Result<(), CdmError> {
        for comment in comments {
            if let Some(issue) = ndm::text::xml_comment_issue(comment) {
                return Err(unwritable(COMMENT, comment, issue));
            }
            self.raw(
                indent,
                &format!("<{COMMENT}>{}</{COMMENT}>", xml::escape(comment)),
            );
        }
        Ok(())
    }
}

// -- Typed mapping shared by both readers --

/// Keyword values of one scope after the repeat check. `blank_is_text` keeps a
/// blank KVN value as empty text, as the KVN reader always has; the XML reader
/// reads an empty element as absent.
struct Fields {
    map: FieldMap,
    blank_is_text: bool,
}

impl Fields {
    fn raw(&self, key: &str) -> Option<&str> {
        if self.blank_is_text {
            self.map.get_last(key)
        } else {
            self.map.get(key)
        }
    }

    fn text(&self, key: &str) -> Option<String> {
        self.raw(key).map(str::to_string)
    }

    /// A numeric value; a blank one is absent, as the ODM readers take it.
    fn numeric(&self, key: &str) -> Option<&str> {
        self.raw(key).filter(|value| !value.is_empty())
    }

    fn num(&self, key: &'static str) -> Result<Option<f64>, CdmError> {
        self.numeric(key)
            .map(|value| validate::strict_f64(value, key).map_err(map_cdm_field_error))
            .transpose()
    }

    fn nums<const N: usize>(&self, keys: [&'static str; N]) -> Result<[Option<f64>; N], CdmError> {
        let mut values = [None; N];
        for (slot, key) in values.iter_mut().zip(keys) {
            *slot = self.num(key)?;
        }
        Ok(values)
    }

    /// A count of table 3-4 ("Data type = integer"): decimal digits with an
    /// optional `+` (6.3.2.1). A count cannot be negative.
    fn count(&self, key: &'static str) -> Result<Option<u64>, CdmError> {
        self.numeric(key)
            .map(|value| validate::strict_int::<u64>(value, key).map_err(map_cdm_field_error))
            .transpose()
    }

    fn required(&self, key: &'static str) -> Result<f64, CdmError> {
        let value = self.numeric(key).ok_or(CdmError::InvalidField {
            field: key,
            kind: CdmInputErrorKind::Missing,
        })?;
        validate::strict_f64(value, key).map_err(map_cdm_field_error)
    }

    fn state(&self, key: &'static str) -> Result<f64, CdmError> {
        let value = self.numeric(key).ok_or(CdmError::IncompleteStateVector)?;
        validate::strict_f64(value, key).map_err(map_cdm_field_error)
    }

    /// Read one covariance row group: every term present yields the group,
    /// none yields `None`, and some but not all is refused naming the first
    /// missing term. 508.0-B-1 table 3-4 makes every term of the 6x6
    /// position/velocity submatrix obligatory, and 5.2.8 allows no subset of
    /// row 7, 8 or 9.
    fn covariance_row<const N: usize>(
        &self,
        keys: [&'static str; N],
    ) -> Result<Option<[f64; N]>, CdmError> {
        let mut values = [0.0_f64; N];
        let mut missing: Option<&'static str> = None;
        let mut present = 0_usize;
        for (slot, key) in values.iter_mut().zip(keys) {
            match self.num(key)? {
                Some(value) => {
                    *slot = value;
                    present += 1;
                }
                None => {
                    missing.get_or_insert(key);
                }
            }
        }
        match (present, missing) {
            (0, _) => Ok(None),
            (_, None) => Ok(Some(values)),
            (_, Some(field)) => Err(CdmError::InvalidField {
                field,
                kind: CdmInputErrorKind::Missing,
            }),
        }
    }
}

/// Map the values and comments either reader collected onto a [`CdmKvn`].
fn assemble(
    message: MessageFields,
    objects: Vec<ObjectFields>,
    blank_is_text: bool,
) -> Result<CdmKvn, CdmError> {
    let MessageFields {
        pairs,
        header_comments,
        relative_comments,
    } = message;
    let map = FieldMap::from_pairs(pairs);
    reject_conflict(&map)?;

    let [first, second] = assign_objects(objects)?;
    let object1 = object_from_fields(first.ok_or(CdmError::IncompleteStateVector)?, blank_is_text)?;
    let object2 = object_from_fields(
        second.ok_or(CdmError::IncompleteStateVector)?,
        blank_is_text,
    )?;

    let fields = Fields { map, blank_is_text };
    let mut cdm = CdmKvn {
        ccsds_cdm_vers: fields.text(VERSION_KEY),
        comments: header_comments,
        creation_date: fields.text("CREATION_DATE"),
        originator: fields.text("ORIGINATOR"),
        message_for: fields.text("MESSAGE_FOR"),
        message_id: fields.text("MESSAGE_ID"),
        relative_comments,
        tca: fields.text("TCA"),
        miss_distance_m: fields.num("MISS_DISTANCE")?,
        relative_speed_m_s: fields.num("RELATIVE_SPEED")?,
        relative_position_rtn_m: fields.nums(RELATIVE_POSITION_KEYS)?,
        relative_velocity_rtn_m_s: fields.nums(RELATIVE_VELOCITY_KEYS)?,
        start_screen_period: fields.text("START_SCREEN_PERIOD"),
        stop_screen_period: fields.text("STOP_SCREEN_PERIOD"),
        screen_volume_frame: fields.text("SCREEN_VOLUME_FRAME"),
        screen_volume_shape: fields.text("SCREEN_VOLUME_SHAPE"),
        screen_volume_m: fields.nums(SCREEN_VOLUME_KEYS)?,
        screen_entry_time: fields.text("SCREEN_ENTRY_TIME"),
        screen_exit_time: fields.text("SCREEN_EXIT_TIME"),
        collision_probability: fields.num("COLLISION_PROBABILITY")?,
        collision_probability_method: fields.text("COLLISION_PROBABILITY_METHOD"),
        hard_body_radius_m: None,
        object1,
        object2,
    };
    let stated = fields
        .num(HBR_KEY)?
        .map(|value| (value, fields.text(HBR_KEY).unwrap_or_default()));
    resolve_hard_body_radius(&mut cdm, stated)?;
    Ok(cdm)
}

/// Assign each object block to `OBJECT1` or `OBJECT2` by the object it
/// states, or by its position when it states none.
fn assign_objects(objects: Vec<ObjectFields>) -> Result<[Option<ObjectFields>; 2], CdmError> {
    let mut slots: [Option<ObjectFields>; 2] = [None, None];
    for (position, object) in objects.into_iter().enumerate() {
        let label = object.label()?;
        let slot = match label.as_deref() {
            None => position,
            Some("OBJECT1") => 0,
            Some("OBJECT2") => 1,
            Some(other) => return Err(CdmError::UnknownObject(other.to_string())),
        };
        match slots.get_mut(slot) {
            Some(entry) if entry.is_none() => *entry = Some(object),
            _ => {
                return Err(CdmError::RepeatedObject(
                    label.unwrap_or_else(|| format!("OBJECT{}", slot + 1)),
                ))
            }
        }
    }
    Ok(slots)
}

fn object_from_fields(fields: ObjectFields, blank_is_text: bool) -> Result<CdmObject, CdmError> {
    let ObjectFields {
        pairs,
        metadata_comments,
        od_comments,
        additional_comments,
        state_comments,
        covariance_comments,
    } = fields;
    let map = FieldMap::from_pairs(pairs);
    reject_conflict(&map)?;
    let fields = Fields { map, blank_is_text };

    let mut state = [0.0_f64; 6];
    for (slot, key) in state.iter_mut().zip(STATE_KEYS) {
        *slot = fields.state(key)?;
    }
    validate::finite_slice(&state, "state").map_err(map_cdm_field_error)?;

    let mut covariance_rtn = [0.0_f64; 6];
    for (slot, key) in covariance_rtn.iter_mut().zip(COVARIANCE_KEYS) {
        *slot = fields.required(key)?;
    }
    validate::finite_slice(&covariance_rtn, "covariance_rtn").map_err(map_cdm_field_error)?;

    let velocity_covariance_rtn = fields.covariance_row(VELOCITY_COVARIANCE_KEYS)?;
    let drag_covariance_rtn = fields.covariance_row(DRAG_COVARIANCE_KEYS)?;
    let srp_covariance_rtn = fields.covariance_row(SRP_COVARIANCE_KEYS)?;
    let thrust_covariance_rtn = fields.covariance_row(THRUST_COVARIANCE_KEYS)?;

    let od_parameters = CdmOdParameters {
        comments: od_comments,
        time_lastob_start: fields.text("TIME_LASTOB_START"),
        time_lastob_end: fields.text("TIME_LASTOB_END"),
        recommended_od_span_d: fields.num("RECOMMENDED_OD_SPAN")?,
        actual_od_span_d: fields.num("ACTUAL_OD_SPAN")?,
        obs_available: fields.count("OBS_AVAILABLE")?,
        obs_used: fields.count("OBS_USED")?,
        tracks_available: fields.count("TRACKS_AVAILABLE")?,
        tracks_used: fields.count("TRACKS_USED")?,
        residuals_accepted_pct: fields.num("RESIDUALS_ACCEPTED")?,
        weighted_rms: fields.num("WEIGHTED_RMS")?,
    };
    // In ADDITIONAL_KEYS order.
    let additional = fields.nums(ADDITIONAL_KEYS)?;
    let additional_parameters = CdmAdditionalParameters {
        comments: additional_comments,
        area_pc_m2: additional[0],
        area_drg_m2: additional[1],
        area_srp_m2: additional[2],
        mass_kg: additional[3],
        cd_area_over_mass_m2_kg: additional[4],
        cr_area_over_mass_m2_kg: additional[5],
        thrust_acceleration_m_s2: additional[6],
        sedr_w_kg: additional[7],
    };

    Ok(CdmObject {
        metadata_comments,
        object_designator: fields.text("OBJECT_DESIGNATOR"),
        catalog_name: fields.text("CATALOG_NAME"),
        object_name: fields.text("OBJECT_NAME"),
        international_designator: fields.text("INTERNATIONAL_DESIGNATOR"),
        object_type: fields.text("OBJECT_TYPE"),
        operator_contact_position: fields.text("OPERATOR_CONTACT_POSITION"),
        operator_organization: fields.text("OPERATOR_ORGANIZATION"),
        operator_phone: fields.text("OPERATOR_PHONE"),
        operator_email: fields.text("OPERATOR_EMAIL"),
        ephemeris_name: fields.text("EPHEMERIS_NAME"),
        covariance_method: fields.text("COVARIANCE_METHOD"),
        maneuverable: fields.text("MANEUVERABLE"),
        orbit_center: fields.text("ORBIT_CENTER"),
        ref_frame: fields.text("REF_FRAME"),
        gravity_model: fields.text("GRAVITY_MODEL"),
        atmospheric_model: fields.text("ATMOSPHERIC_MODEL"),
        n_body_perturbations: fields.text("N_BODY_PERTURBATIONS"),
        solar_rad_pressure: fields.text("SOLAR_RAD_PRESSURE"),
        earth_tides: fields.text("EARTH_TIDES"),
        intrack_thrust: fields.text("INTRACK_THRUST"),
        od_parameters,
        additional_parameters,
        state_comments,
        state: (
            (state[0], state[1], state[2]),
            (state[3], state[4], state[5]),
        ),
        covariance_comments,
        covariance_rtn,
        velocity_covariance_rtn,
        drag_covariance_rtn,
        srp_covariance_rtn,
        thrust_covariance_rtn,
    })
}

// -- Hard-body radius comment convention --

/// The first comment of a message, in writing order, that follows the
/// `HBR = <value>` convention.
struct HbrComment {
    /// Index of its list in [`comment_lists`] order.
    list: usize,
    /// Index within that list.
    index: usize,
    /// The comment text.
    text: String,
    /// The radius it states; `None` for `HBR =` with no value.
    value: Option<f64>,
    /// Whether it reads exactly as the writers state a radius, so the readers
    /// take it as the radius alone.
    canonical: bool,
}

/// The comment the writers add to state a radius.
fn canonical_hbr_comment(value: f64) -> String {
    format!("{HBR_KEY} = {}", fmt_num(value))
}

/// Every comment list of `cdm` in writing order: header, relative
/// metadata/data, then for each object its metadata, OD parameters,
/// additional parameters, state vector and covariance.
fn comment_lists(cdm: &CdmKvn) -> Vec<&[String]> {
    let mut lists: Vec<&[String]> = vec![cdm.comments.as_slice(), cdm.relative_comments.as_slice()];
    for object in [&cdm.object1, &cdm.object2] {
        lists.extend([
            object.metadata_comments.as_slice(),
            object.od_parameters.comments.as_slice(),
            object.additional_parameters.comments.as_slice(),
            object.state_comments.as_slice(),
            object.covariance_comments.as_slice(),
        ]);
    }
    lists
}

/// The comment list at `list` in [`comment_lists`] order.
fn comment_list_mut(cdm: &mut CdmKvn, list: usize) -> &mut Vec<String> {
    fn object_list(object: &mut CdmObject, list: usize) -> &mut Vec<String> {
        match list {
            0 => &mut object.metadata_comments,
            1 => &mut object.od_parameters.comments,
            2 => &mut object.additional_parameters.comments,
            3 => &mut object.state_comments,
            _ => &mut object.covariance_comments,
        }
    }
    match list {
        0 => &mut cdm.comments,
        1 => &mut cdm.relative_comments,
        2..=6 => object_list(&mut cdm.object1, list - 2),
        _ => object_list(&mut cdm.object2, list.saturating_sub(7)),
    }
}

fn first_hbr_comment(cdm: &CdmKvn) -> Option<HbrComment> {
    for (list, comments) in comment_lists(cdm).into_iter().enumerate() {
        for (index, comment) in comments.iter().enumerate() {
            if let Some(value) = hbr_from_comment(comment) {
                return Some(HbrComment {
                    list,
                    index,
                    text: comment.clone(),
                    value,
                    canonical: value.is_some_and(|value| *comment == canonical_hbr_comment(value)),
                });
            }
        }
    }
    None
}

/// Set the hard-body radius a reader found, from an `HBR` keyword or element
/// (`stated`, with its text) or the first convention comment. The two must
/// agree. A comment in the form the writers produce is taken as the radius
/// alone and leaves the comments; any other stays where it was.
fn resolve_hard_body_radius(
    cdm: &mut CdmKvn,
    stated: Option<(f64, String)>,
) -> Result<(), CdmError> {
    let found = first_hbr_comment(cdm);
    let radius = match (stated, &found) {
        (None, None) => None,
        (Some((value, _)), None) => Some(value),
        (None, Some(comment)) => comment.value,
        (Some((value, text)), Some(comment)) => {
            if comment.value.map(f64::to_bits) != Some(value.to_bits()) {
                return Err(CdmError::DuplicateField {
                    field: HBR_KEY.to_string(),
                    first: text,
                    second: comment.text.clone(),
                });
            }
            Some(value)
        }
    };
    if let Some(comment) = found.filter(|comment| comment.canonical) {
        let list = comment_list_mut(cdm, comment.list);
        if comment.index < list.len() {
            list.remove(comment.index);
        }
    }
    cdm.hard_body_radius_m = radius;
    Ok(())
}

/// Where a writer adds the comment that states [`CdmKvn::hard_body_radius_m`].
#[derive(Debug, Clone, PartialEq, Eq)]
enum HbrStatement {
    /// The retained comments already state the radius, or state none and the
    /// message holds none.
    Retained,
    /// Before the relative metadata/data comments.
    Relative(String),
    /// Before the header comments, because a header comment follows the
    /// convention and the reader would take it first.
    Header(String),
}

/// How a writer states [`CdmKvn::hard_body_radius_m`] so it reads back. The
/// added comment is read first, so it states the radius whatever the retained
/// comments say. A retained comment that would read back as a radius when the
/// message holds none is refused.
fn hard_body_radius_statement(cdm: &CdmKvn) -> Result<HbrStatement, CdmError> {
    match (cdm.hard_body_radius_m, first_hbr_comment(cdm)) {
        (None, None) => Ok(HbrStatement::Retained),
        (None, Some(comment)) if comment.value.is_none() => Ok(HbrStatement::Retained),
        (None, Some(comment)) => Err(CdmError::HardBodyRadiusComment {
            comment: comment.text,
        }),
        (Some(radius), Some(comment))
            if !comment.canonical && comment.value.map(f64::to_bits) == Some(radius.to_bits()) =>
        {
            Ok(HbrStatement::Retained)
        }
        (Some(radius), Some(comment)) if comment.list == 0 => {
            Ok(HbrStatement::Header(canonical_hbr_comment(radius)))
        }
        (Some(radius), _) => Ok(HbrStatement::Relative(canonical_hbr_comment(radius))),
    }
}

/// The header comments a writer writes: the retained ones, opened by the
/// radius comment when it goes there.
fn header_comments(cdm: &CdmKvn, statement: &HbrStatement) -> Vec<String> {
    let mut comments = Vec::with_capacity(cdm.comments.len() + 1);
    if let HbrStatement::Header(comment) = statement {
        comments.push(comment.clone());
    }
    comments.extend(cdm.comments.iter().cloned());
    comments
}

/// Read the `HBR = <value>` convention from a comment's text: `HBR`, `=`,
/// then nothing, or one finite number alone or followed by `m` or `[m]`.
/// Returns `Some(Some(radius))` for a radius, `Some(None)` for `HBR =` with
/// nothing after it, and `None` for any other text, which stays an ordinary
/// comment: a comment is free text, so one that states the radius in another
/// unit, or no number, is neither read as a radius nor refused.
fn hbr_from_comment(text: &str) -> Option<Option<f64>> {
    let rest = strip_prefix_ci(text.trim_start(), HBR_KEY)?;
    let rest = rest.trim_start().strip_prefix('=')?.trim();
    if rest.is_empty() {
        return Some(None);
    }
    let (body, bracketed_unit) = ndm::split_unit(rest);
    let mut tokens = body.split_whitespace();
    let number = tokens.next()?;
    let trailing: Vec<&str> = tokens.collect();
    let in_metres = matches!(
        (bracketed_unit, trailing.as_slice()),
        (None, []) | (None, ["m"]) | (Some("m"), [])
    );
    if !in_metres {
        return None;
    }
    validate::strict_f64(number, "HBR").ok().map(Some)
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

// -- Units, repeats and value helpers --

/// The unit CCSDS 508.0-B-1 tables 3-2 and 3-4 give a numeric keyword: an empty
/// list for a dimensionless one (`n/a` in the tables), `None` for a text
/// keyword. `HBR` follows the meters of the NASA CARA comment convention.
fn cdm_unit(key: &str) -> Option<&'static [&'static str]> {
    const DIMENSIONLESS: &[&str] = &[];
    const M: &[&str] = &["m"];
    const M_PER_S: &[&str] = &["m/s"];
    const M_PER_S2: &[&str] = &["m/s**2"];
    const KM: &[&str] = &["km"];
    const KM_PER_S: &[&str] = &["km/s"];
    const DAYS: &[&str] = &["d"];
    const PERCENT: &[&str] = &["%"];
    const M2: &[&str] = &["m**2"];
    const KG: &[&str] = &["kg"];
    const M2_PER_KG: &[&str] = &["m**2/kg"];
    const W_PER_KG: &[&str] = &["W/kg"];
    const M2_PER_S: &[&str] = &["m**2/s"];
    const M2_PER_S2: &[&str] = &["m**2/s**2"];
    const M2_PER_S3: &[&str] = &["m**2/s**3"];
    const M2_PER_S4: &[&str] = &["m**2/s**4"];
    const M3_PER_KG: &[&str] = &["m**3/kg"];
    const M3_PER_KG_S: &[&str] = &["m**3/(kg*s)"];
    const M3_PER_KG_S2: &[&str] = &["m**3/(kg*s**2)"];
    const M4_PER_KG2: &[&str] = &["m**4/kg**2"];
    Some(match key {
        "MISS_DISTANCE"
        | "RELATIVE_POSITION_R"
        | "RELATIVE_POSITION_T"
        | "RELATIVE_POSITION_N"
        | "SCREEN_VOLUME_X"
        | "SCREEN_VOLUME_Y"
        | "SCREEN_VOLUME_Z"
        | "HBR" => M,
        "RELATIVE_SPEED"
        | "RELATIVE_VELOCITY_R"
        | "RELATIVE_VELOCITY_T"
        | "RELATIVE_VELOCITY_N" => M_PER_S,
        "COLLISION_PROBABILITY"
        | "OBS_AVAILABLE"
        | "OBS_USED"
        | "TRACKS_AVAILABLE"
        | "TRACKS_USED"
        | "WEIGHTED_RMS" => DIMENSIONLESS,
        "RECOMMENDED_OD_SPAN" | "ACTUAL_OD_SPAN" => DAYS,
        "RESIDUALS_ACCEPTED" => PERCENT,
        "AREA_PC" | "AREA_DRG" | "AREA_SRP" => M2,
        "MASS" => KG,
        "CD_AREA_OVER_MASS" | "CR_AREA_OVER_MASS" => M2_PER_KG,
        "THRUST_ACCELERATION" => M_PER_S2,
        "SEDR" => W_PER_KG,
        "X" | "Y" | "Z" => KM,
        "X_DOT" | "Y_DOT" | "Z_DOT" => KM_PER_S,
        "CR_R" | "CT_R" | "CT_T" | "CN_R" | "CN_T" | "CN_N" => M2,
        "CRDOT_R" | "CRDOT_T" | "CRDOT_N" | "CTDOT_R" | "CTDOT_T" | "CTDOT_N" | "CNDOT_R"
        | "CNDOT_T" | "CNDOT_N" => M2_PER_S,
        "CRDOT_RDOT" | "CTDOT_RDOT" | "CTDOT_TDOT" | "CNDOT_RDOT" | "CNDOT_TDOT" | "CNDOT_NDOT" => {
            M2_PER_S2
        }
        "CDRG_R" | "CDRG_T" | "CDRG_N" | "CSRP_R" | "CSRP_T" | "CSRP_N" => M3_PER_KG,
        "CDRG_RDOT" | "CDRG_TDOT" | "CDRG_NDOT" | "CSRP_RDOT" | "CSRP_TDOT" | "CSRP_NDOT" => {
            M3_PER_KG_S
        }
        "CDRG_DRG" | "CSRP_DRG" | "CSRP_SRP" => M4_PER_KG2,
        "CTHR_R" | "CTHR_T" | "CTHR_N" => M2_PER_S2,
        "CTHR_RDOT" | "CTHR_TDOT" | "CTHR_NDOT" => M2_PER_S3,
        "CTHR_DRG" | "CTHR_SRP" => M3_PER_KG_S2,
        "CTHR_THR" => M2_PER_S4,
        _ => return None,
    })
}

fn unit_mismatch(key: &str, mismatch: UnitMismatch) -> CdmError {
    CdmError::UnitMismatch {
        field: key.to_string(),
        unit: mismatch.unit,
        expected: mismatch.expected,
    }
}

/// Refuse a keyword that repeats in its scope with a different value.
fn reject_conflict(map: &FieldMap) -> Result<(), CdmError> {
    match map.first_conflict(|_| true) {
        Some(ConflictingField { key, first, second }) => Err(CdmError::DuplicateField {
            field: key,
            first,
            second,
        }),
        None => Ok(()),
    }
}

fn map_cdm_field_error(error: validate::FieldError) -> CdmError {
    CdmError::InvalidField {
        field: error.field(),
        kind: CdmInputErrorKind::from(&error),
    }
}

/// Shortest round-tripping decimal for a finite value.
fn fmt_num(value: f64) -> String {
    format!("{value}")
}

/// Refuse a non-finite number, which neither reader accepts.
fn validate_cdm(cdm: &CdmKvn) -> Result<(), CdmError> {
    for (key, value) in relative_items(cdm) {
        if let CdmValue::Number(value) = value {
            validate_optional_num(value, key)?;
        }
    }
    validate_optional_num(cdm.hard_body_radius_m, HBR_KEY)?;
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
    validate::finite_slice(&state_values(object), "state").map_err(map_cdm_field_error)?;
    validate::finite_slice(&object.covariance_rtn, "covariance_rtn")
        .map_err(map_cdm_field_error)?;
    let rows: [(Option<&[f64]>, &'static str); 4] = [
        (
            object
                .velocity_covariance_rtn
                .as_ref()
                .map(|row| row.as_slice()),
            "velocity_covariance_rtn",
        ),
        (
            object
                .drag_covariance_rtn
                .as_ref()
                .map(|row| row.as_slice()),
            "drag_covariance_rtn",
        ),
        (
            object.srp_covariance_rtn.as_ref().map(|row| row.as_slice()),
            "srp_covariance_rtn",
        ),
        (
            object
                .thrust_covariance_rtn
                .as_ref()
                .map(|row| row.as_slice()),
            "thrust_covariance_rtn",
        ),
    ];
    for (row, field) in rows {
        if let Some(row) = row {
            validate::finite_slice(row, field).map_err(map_cdm_field_error)?;
        }
    }
    for (key, value) in od_items(&object.od_parameters)
        .into_iter()
        .chain(additional_items(&object.additional_parameters))
    {
        if let CdmValue::Number(value) = value {
            validate_optional_num(value, key)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_split_removes_trailing_bracket() {
        assert_eq!(ndm::split_unit("7000.0 [km]").0, "7000.0");
        assert_eq!(ndm::split_unit("4.835E-05").0, "4.835E-05");
        assert_eq!(ndm::split_unit("0.045663 [m**2/kg]").0, "0.045663");
        assert_eq!(ndm::split_unit("97.8 [%]").0, "97.8");
    }

    /// The object `bare_object` builds with this position covariance.
    fn with_position_covariance(covariance_rtn: [f64; 6]) -> CdmObject {
        bare_object(((1.0, 2.0, 3.0), (0.1, 0.2, 0.3)), covariance_rtn)
    }

    #[test]
    fn cdm_covariance_rtn_validation_accepts_psd_lower_triangle() {
        assert_eq!(
            with_position_covariance([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).to_covariance_rtn(),
            Ok(vec![
                vec![1.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
            ])
        );
    }

    #[test]
    fn cdm_covariance_rtn_validation_rejects_non_psd_lower_triangle() {
        let expected = Err(CdmError::InvalidField {
            field: "covariance_rtn",
            kind: CdmInputErrorKind::NotPositive,
        });

        assert_eq!(
            with_position_covariance([-1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).to_covariance_rtn(),
            expected
        );
        assert_eq!(
            with_position_covariance([1.0, 2.0, 1.0, 0.0, 0.0, 1.0]).to_covariance_rtn(),
            expected
        );
    }

    #[test]
    fn covariance_rtn_spans_the_rows_the_object_holds() {
        let mut object = with_position_covariance([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]);
        let mut velocity = [0.0; 15];
        for index in [3, 8, 14] {
            velocity[index] = 1.0;
        }
        object.velocity_covariance_rtn = Some(velocity);
        let mut drag = [0.0; 7];
        drag[6] = 1.0;
        object.drag_covariance_rtn = Some(drag);
        let matrix = object.to_covariance_rtn().unwrap();
        assert_eq!(matrix.len(), 7);
        for (row, values) in matrix.iter().enumerate() {
            for (col, value) in values.iter().enumerate() {
                assert_eq!(*value, if row == col { 1.0 } else { 0.0 });
            }
        }

        // Row 8 has no place without row 7.
        object.srp_covariance_rtn = Some([0.0; 8]);
        object.drag_covariance_rtn = None;
        assert_eq!(
            object.to_covariance_rtn(),
            Err(CdmError::InvalidField {
                field: "drag_covariance_rtn",
                kind: CdmInputErrorKind::Missing,
            })
        );
    }

    #[test]
    fn incomplete_state_vector_is_rejected() {
        let kvn = "OBJECT = OBJECT1\nX = 7000.0 [km]\nOBJECT = OBJECT2\nX = 1.0 [km]\n";
        assert_eq!(parse_kvn(kvn), Err(CdmError::IncompleteStateVector));
    }

    #[test]
    fn hbr_is_recovered_from_comment_only() {
        assert_eq!(hbr_from_comment("HBR = 15.5"), Some(Some(15.5)));
        assert_eq!(hbr_from_comment("hbr=15.5 m"), Some(Some(15.5)));
        assert_eq!(hbr_from_comment("HBR = 15.5 [m]"), Some(Some(15.5)));
        assert_eq!(hbr_from_comment("HBR ="), Some(None));
        assert_eq!(hbr_from_comment("Relative Metadata/Data"), None);
        // Another unit, trailing text or no number leaves an ordinary comment.
        for text in [
            "HBR = 0.02 km",
            "HBR = 20 [km]",
            "HBR = TBD",
            "HBR = N/A",
            "HBR = 15.5m",
            "HBR = 15.5 m, assumed",
            "HBR = NaN",
        ] {
            assert_eq!(hbr_from_comment(text), None, "{text}");
        }
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

    /// The keyword values of every leaf element of `xml`, as the XML reader
    /// records them.
    fn xml_leaf_fields(xml: &str) -> Result<Fields, CdmError> {
        let doc = Document::parse(xml).unwrap();
        let mut pairs = Vec::new();
        for leaf in ndm::leaf_descendants(doc.root_element()) {
            push_xml_leaf(leaf, &mut pairs)?;
        }
        Ok(Fields {
            map: FieldMap::from_pairs(pairs),
            blank_is_text: false,
        })
    }

    #[test]
    fn xml_fields_read_leaf_values_and_accept_table_units() {
        let fields = xml_leaf_fields(
            r#"<r><MESSAGE_ID>abc123</MESSAGE_ID><X units="km">2570.097065</X><ORIGINATOR></ORIGINATOR></r>"#,
        )
        .unwrap();
        assert_eq!(fields.text("MESSAGE_ID").as_deref(), Some("abc123"));
        // A `units` attribute matching table 3-4 is accepted; the element text is returned.
        assert_eq!(fields.text("X").as_deref(), Some("2570.097065"));
        // An empty leaf element yields None.
        assert_eq!(fields.text("ORIGINATOR"), None);

        // A distinct element name sharing a prefix must not match.
        let fields = xml_leaf_fields(r#"<r><X_DOT units="km/s">4.4</X_DOT></r>"#).unwrap();
        assert_eq!(fields.text("X"), None);

        // A unit on a text keyword contradicts the tables.
        assert_eq!(
            xml_leaf_fields(r#"<r><OBJECT_NAME units="m">SAT</OBJECT_NAME></r>"#).err(),
            Some(CdmError::UnitMismatch {
                field: "OBJECT_NAME".to_string(),
                unit: "m".to_string(),
                expected: None,
            })
        );
    }

    /// Two segments carrying the position covariance only, for the XML tests
    /// below; `{velocity}` is inserted after `CN_N` of the first segment.
    fn two_segment_xml(object_name: &str, velocity: &str) -> String {
        format!(
            r#"<cdm><body>
<segment><metadata><OBJECT_NAME>{object_name}</OBJECT_NAME></metadata>
<data><stateVector>
<X units="km">1.0</X><Y units="km">2.0</Y><Z units="km">3.0</Z>
<X_DOT units="km/s">0.1</X_DOT><Y_DOT units="km/s">0.2</Y_DOT><Z_DOT units="km/s">0.3</Z_DOT>
</stateVector><covarianceMatrix>
<CR_R units="m**2">41.42</CR_R><CT_R units="m**2">-8.579</CT_R><CT_T units="m**2">2533.0</CT_T>
<CN_R units="m**2">-23.13</CN_R><CN_T units="m**2">13.36</CN_T><CN_N units="m**2">70.98</CN_N>
{velocity}
</covarianceMatrix></data></segment>
<segment><data><stateVector>
<X units="km">4.0</X><Y units="km">5.0</Y><Z units="km">6.0</Z>
<X_DOT units="km/s">0.4</X_DOT><Y_DOT units="km/s">0.5</Y_DOT><Z_DOT units="km/s">0.6</Z_DOT>
</stateVector><covarianceMatrix>
<CR_R units="m**2">1.0</CR_R><CT_R units="m**2">0.0</CT_R><CT_T units="m**2">1.0</CT_T>
<CN_R units="m**2">0.0</CN_R><CN_T units="m**2">0.0</CN_T><CN_N units="m**2">1.0</CN_N>
</covarianceMatrix></data></segment>
</body></cdm>"#
        )
    }

    #[test]
    fn xml_parse_decodes_entities() {
        let cdm = parse_xml(&two_segment_xml("SAT A &amp; B", "")).unwrap();
        // The DOM reader decodes the `&amp;` entity.
        assert_eq!(cdm.object1.object_name.as_deref(), Some("SAT A & B"));
        assert_eq!(
            cdm.object1.covariance_rtn,
            [41.42, -8.579, 2533.0, -23.13, 13.36, 70.98]
        );
        assert_eq!(cdm.object1.velocity_covariance_rtn, None);
    }

    #[test]
    fn partial_velocity_covariance_is_refused_naming_the_first_missing_term() {
        // 508.0-B-1 table 3-4 makes every term of the 6x6 position/velocity
        // submatrix obligatory, so a lone CRDOT_R is an incomplete block.
        let xml = two_segment_xml("SAT", r#"<CRDOT_R units="m**2/s">2.52e-3</CRDOT_R>"#);
        assert_eq!(
            parse_xml(&xml),
            Err(CdmError::InvalidField {
                field: "CRDOT_T",
                kind: CdmInputErrorKind::Missing,
            })
        );
        let kvn = FULL_KVN.replacen("CTDOT_N = -1.359E-03 [m**2/s]\n", "", 1);
        assert_eq!(
            parse_kvn(&kvn),
            Err(CdmError::InvalidField {
                field: "CTDOT_N",
                kind: CdmInputErrorKind::Missing,
            })
        );
    }

    #[test]
    fn units_are_checked_against_tables_3_2_and_3_4() {
        let kvn = FULL_KVN.replacen("MISS_DISTANCE = 715 [m]", "MISS_DISTANCE = 0.715 [km]", 1);
        assert_eq!(
            parse_kvn(&kvn),
            Err(CdmError::UnitMismatch {
                field: "MISS_DISTANCE".to_string(),
                unit: "km".to_string(),
                expected: Some("m"),
            })
        );
        let kvn = FULL_KVN.replacen(
            "COLLISION_PROBABILITY = 4.835E-05",
            "COLLISION_PROBABILITY = 4.835E-05 [n/a]",
            1,
        );
        assert_eq!(
            parse_kvn(&kvn),
            Err(CdmError::UnitMismatch {
                field: "COLLISION_PROBABILITY".to_string(),
                unit: "n/a".to_string(),
                expected: None,
            })
        );
        let named = FULL_KVN.replacen(
            "OBJECT_NAME = SATELLITE A",
            "OBJECT_NAME = SATELLITE A [BLOCK 2]",
            1,
        );
        assert_eq!(
            parse_kvn(&named).unwrap().object1.object_name.as_deref(),
            Some("SATELLITE A [BLOCK 2]")
        );

        // 4.3.10: XML units are those of the KVN tables.
        let xml = FULL_XML.replacen(
            r#"<MISS_DISTANCE units="m">715</MISS_DISTANCE>"#,
            r#"<MISS_DISTANCE units="km">0.715</MISS_DISTANCE>"#,
            1,
        );
        assert_eq!(
            parse_xml(&xml),
            Err(CdmError::UnitMismatch {
                field: "MISS_DISTANCE".to_string(),
                unit: "km".to_string(),
                expected: Some("m"),
            })
        );
    }

    #[test]
    fn repeated_keywords_with_different_values_are_refused() {
        let kvn = FULL_KVN.replacen(
            "TCA = 2010-03-13T22:37:52.618\n",
            "TCA = 2010-03-13T22:37:52.618\nTCA = 2010-03-13T22:37:53.000\n",
            1,
        );
        assert_eq!(
            parse_kvn(&kvn),
            Err(CdmError::DuplicateField {
                field: "TCA".to_string(),
                first: "2010-03-13T22:37:52.618".to_string(),
                second: "2010-03-13T22:37:53.000".to_string(),
            })
        );
        let kvn = FULL_KVN.replacen(
            "X = 2570.097065 [km]\n",
            "X = 2570.097065 [km]\nX = 2570.1 [km]\n",
            1,
        );
        assert_eq!(
            parse_kvn(&kvn),
            Err(CdmError::DuplicateField {
                field: "X".to_string(),
                first: "2570.097065".to_string(),
                second: "2570.1".to_string(),
            })
        );
        // The same keyword in the two object blocks is not a repeat.
        assert!(parse_kvn(FULL_KVN).is_ok());

        let xml = FULL_XML.replacen(
            "<TCA>2010-03-13T22:37:52.618</TCA>",
            "<TCA>2010-03-13T22:37:52.618</TCA><TCA>2010-03-13T22:37:53.000</TCA>",
            1,
        );
        assert!(matches!(
            parse_xml(&xml),
            Err(CdmError::DuplicateField { field, .. }) if field == "TCA"
        ));
    }

    #[test]
    fn more_than_two_objects_are_refused() {
        let third = &FULL_KVN[FULL_KVN.find("OBJECT = OBJECT2").unwrap()..];
        let kvn = format!("{FULL_KVN}{}", third.replacen("OBJECT2", "OBJECT3", 1));
        assert_eq!(parse_kvn(&kvn), Err(CdmError::UnexpectedObjectCount(3)));

        let message = FULL_XML.trim_start_matches(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
        assert_eq!(
            parse_xml(&format!("<ndm>{message}{message}</ndm>")),
            Err(CdmError::MultipleMessages { count: 2 })
        );
    }

    #[test]
    fn absent_header_values_are_not_written_and_read_back_absent() {
        let mut cdm = parse_kvn(FULL_KVN).unwrap();
        cdm.message_id = None;
        cdm.miss_distance_m = None;
        let encoded = encode_kvn(&cdm).unwrap();
        assert!(!encoded.contains("MESSAGE_ID"));
        assert!(!encoded.contains("MISS_DISTANCE"));
        assert_eq!(parse_kvn(&encoded).unwrap(), cdm);
    }

    #[test]
    fn xml_hard_body_radius_round_trips() {
        let mut cdm = parse_xml(FULL_XML).unwrap();
        cdm.hard_body_radius_m = Some(15.5);
        let encoded = encode_xml(&cdm).unwrap();
        assert!(encoded.contains("<COMMENT>HBR = 15.5</COMMENT>"));
        assert_eq!(parse_xml(&encoded).unwrap(), cdm);
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

    /// An object holding only the given state and position covariance.
    fn bare_object(
        state: ((f64, f64, f64), (f64, f64, f64)),
        covariance_rtn: [f64; 6],
    ) -> CdmObject {
        CdmObject {
            metadata_comments: Vec::new(),
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
            od_parameters: CdmOdParameters::default(),
            additional_parameters: CdmAdditionalParameters::default(),
            state_comments: Vec::new(),
            state,
            covariance_comments: Vec::new(),
            covariance_rtn,
            velocity_covariance_rtn: None,
            drag_covariance_rtn: None,
            srp_covariance_rtn: None,
            thrust_covariance_rtn: None,
        }
    }

    /// A message holding only the two objects.
    fn bare_message(object1: CdmObject, object2: CdmObject) -> CdmKvn {
        CdmKvn {
            ccsds_cdm_vers: None,
            comments: Vec::new(),
            creation_date: None,
            originator: None,
            message_for: None,
            message_id: None,
            relative_comments: Vec::new(),
            tca: None,
            miss_distance_m: None,
            relative_speed_m_s: None,
            relative_position_rtn_m: [None; 3],
            relative_velocity_rtn_m_s: [None; 3],
            start_screen_period: None,
            stop_screen_period: None,
            screen_volume_frame: None,
            screen_volume_shape: None,
            screen_volume_m: [None; 3],
            screen_entry_time: None,
            screen_exit_time: None,
            collision_probability: None,
            collision_probability_method: None,
            hard_body_radius_m: None,
            object1,
            object2,
        }
    }

    #[test]
    fn xml_round_trips_through_encode_and_parse() {
        let mut object = bare_object(
            ((1.5, 2.5, 3.5), (0.1, 0.2, 0.3)),
            [41.42, -8.579, 2533.0, -23.13, 13.36, 70.98],
        );
        object.object_designator = Some("12345".to_string());
        object.object_name = Some("SAT A & B".to_string());
        object.ref_frame = Some("EME2000".to_string());
        let mut original = bare_message(object.clone(), object);
        original.ccsds_cdm_vers = Some("1.0".to_string());
        original.creation_date = Some("2024-01-01T00:00:00.000".to_string());
        original.originator = Some("TEST".to_string());
        original.message_id = Some("ID-1".to_string());
        original.tca = Some("2024-01-01T12:00:00.000".to_string());
        original.miss_distance_m = Some(715.0);
        original.relative_speed_m_s = Some(14762.0);
        original.collision_probability = Some(4.835e-5);
        original.collision_probability_method = Some("FOSTER-1992".to_string());

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
        let object = bare_object(
            ((1.0, 2.0, 3.0), (0.1, 0.2, 0.3)),
            [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
        );
        let mut cdm = bare_message(object.clone(), object);
        cdm.miss_distance_m = Some(f64::NAN);

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
    /// full metadata block and the complete 6x6 RTN covariance for both objects,
    /// with a relative metadata/data comment.
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
        // The relative metadata/data comment is written at the start of its
        // block.
        assert!(encoded.contains("COMMENT Relative Metadata/Data\nTCA = "));
        assert_eq!(parsed.relative_comments, vec!["Relative Metadata/Data"]);

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
        // The relativeMetadataData COMMENT element is written back.
        assert!(encoded.contains("<COMMENT>Relative Metadata/Data</COMMENT>"));

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

    /// CCSDS 508.0-B-1 3.6.3, the KVN example with optional keywords, one
    /// assignment per line. The standard prints Object1's `TRACKS_USED` as
    /// `TRACKS USED`; it is spelled here as table 3-4 defines it, and the
    /// misprint is refused by name in
    /// `unknown_keywords_and_malformed_lines_are_refused`.
    const STANDARD_OPTIONAL_KVN: &str = "\
CCSDS_CDM_VERS = 1.0
CREATION_DATE = 2010-03-12T22:31:12.000
ORIGINATOR = JSPOC
MESSAGE_FOR = SATELLITE A
MESSAGE_ID = 201113719185
COMMENT Relative Metadata/Data
TCA = 2010-03-13T22:37:52.618
MISS_DISTANCE = 715 [m]
RELATIVE_SPEED = 14762 [m/s]
RELATIVE_POSITION_R = 27.4 [m]
RELATIVE_POSITION_T = -70.2 [m]
RELATIVE_POSITION_N = 711.8 [m]
RELATIVE_VELOCITY_R = -7.2 [m/s]
RELATIVE_VELOCITY_T = -14692.0 [m/s]
RELATIVE_VELOCITY_N = -1437.2 [m/s]
START_SCREEN_PERIOD = 2010-03-12T18:29:32:212
STOP_SCREEN_PERIOD = 2010-03-15T18:29:32:212
SCREEN_VOLUME_FRAME = RTN
SCREEN_VOLUME_SHAPE = ELLIPSOID
SCREEN_VOLUME_X = 200 [m]
SCREEN_VOLUME_Y = 1000 [m]
SCREEN_VOLUME_Z = 1000 [m]
SCREEN_ENTRY_TIME = 2010-03-13T22:37:52.222
SCREEN_EXIT_TIME = 2010-03-13T22:37:52.824
COLLISION_PROBABILITY = 4.835E-05
COLLISION_PROBABILITY_METHOD = FOSTER-1992
COMMENT Object1 Metadata
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 12345
CATALOG_NAME = SATCAT
OBJECT_NAME = SATELLITE A
INTERNATIONAL_DESIGNATOR = 1997-030E
OBJECT_TYPE = PAYLOAD
OPERATOR_CONTACT_POSITION = OSA
OPERATOR_ORGANIZATION = EUMETSAT
OPERATOR_PHONE = +49615130312
OPERATOR_EMAIL = JOHN.DOE@SOMEWHERE.NET

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
COMMENT Object1 Data
COMMENT Object1 OD Parameters
TIME_LASTOB_START = 2010-03-12T02:14:12.746
TIME_LASTOB_END = 2010-03-12T02:14:12.746
RECOMMENDED_OD_SPAN = 7.88 [d]
ACTUAL_OD_SPAN = 5.50 [d]
OBS_AVAILABLE = 592
OBS_USED = 579
TRACKS_AVAILABLE = 123
TRACKS_USED = 119
RESIDUALS_ACCEPTED = 97.8 [%]
WEIGHTED_RMS = 0.864
COMMENT Object1 Additional Parameters
COMMENT Apogee Altitude=779 km
COMMENT Perigee Altitude=765 km
COMMENT Inclination=86.4 deg
AREA_PC = 5.2 [m**2]
MASS = 251.6 [kg]
CD_AREA_OVER_MASS = 0.045663 [m**2/kg]
CR_AREA_OVER_MASS = 0.000000 [m**2/kg]
THRUST_ACCELERATION = 0.0 [m/s**2]
SEDR = 4.54570E-05 [W/kg]
COMMENT Object1 State Vector
X = 2570.097065 [km]
Y = 2244.654904 [km]
Z = 6281.497978 [km]
X_DOT = 4.418769571 [km/s]
Y_DOT = 4.833547743 [km/s]
Z_DOT = -3.526774282 [km/s]
COMMENT Object1 Covariance in the RTN Coordinate Frame
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
CDRG_R = -1.862E+00 [m**3/kg]
CDRG_T = 3.530E+00 [m**3/kg]
CDRG_N = -3.100E-01 [m**3/kg]
CDRG_RDOT = -1.214E-04 [m**3/(kg*s)]
CDRG_TDOT = 2.580E-04 [m**3/(kg*s)]
CDRG_NDOT = -6.467E-05 [m**3/(kg*s)]
CDRG_DRG = 3.483E-06 [m**4/kg**2]
CSRP_R = -1.492E+02 [m**3/kg]
CSRP_T = 2.044E+02 [m**3/kg]
CSRP_N = -2.331E+01 [m**3/kg]
CSRP_RDOT = -1.254E-03 [m**3/(kg*s)]
CSRP_TDOT = 2.013E-02 [m**3/(kg*s)]
CSRP_NDOT = -4.700E-03 [m**3/(kg*s)]
CSRP_DRG = 2.210E-04 [m**4/kg**2]
CSRP_SRP = 1.593E-02 [m**4/kg**2]
COMMENT Object2 Metadata
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
COMMENT Object2 Data
COMMENT Object2 OD Parameters
TIME_LASTOB_START = 2010-03-12T01:14:12.746
TIME_LASTOB_END = 2010-03-12T03:14:12.746
RECOMMENDED_OD_SPAN = 2.63 [d]
ACTUAL_OD_SPAN = 2.63 [d]
OBS_AVAILABLE = 59
OBS_USED = 58
TRACKS_AVAILABLE = 15
TRACKS_USED = 15
RESIDUALS_ACCEPTED = 97.8 [%]
WEIGHTED_RMS = 0.864
COMMENT Object2 Additional Parameters
COMMENT Apogee Altitude=786 km
COMMENT Perigee Altitude=414 km
COMMENT Inclination=98.9 deg
AREA_PC = 0.9 [m**2]
CD_AREA_OVER_MASS = 0.118668 [m**2/kg]
CR_AREA_OVER_MASS = 0.075204 [m**2/kg]
THRUST_ACCELERATION = 0.0 [m/s**2]
SEDR = 5.40900E-03 [W/kg]
COMMENT Object2 State Vector
X = 2569.540800 [km]
Y = 2245.093614 [km]
Z = 6281.599946 [km]
X_DOT = -2.888612500 [km/s]
Y_DOT = -6.007247516 [km/s]
Z_DOT = 3.328770172 [km/s]
COMMENT Object2 Covariance in the RTN Coordinate Frame
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
CDRG_R = -5.117E-01 [m**3/kg]
CDRG_T = 1.319E+00 [m**3/kg]
CDRG_N = -9.034E-02 [m**3/kg]
CDRG_RDOT = -7.708E-05 [m**3/(kg*s)]
CDRG_TDOT = 7.402E-05 [m**3/(kg*s)]
CDRG_NDOT = -1.903E-05 [m**3/(kg*s)]
CDRG_DRG = 1.053E-06 [m**4/kg**2]
CSRP_R = -3.297E+01 [m**3/kg]
CSRP_T = 8.164E+01 [m**3/kg]
CSRP_N = -5.651E+00 [m**3/kg]
CSRP_RDOT = -4.636E-03 [m**3/(kg*s)]
CSRP_TDOT = 4.738E-03 [m**3/(kg*s)]
CSRP_NDOT = -1.198E-03 [m**3/(kg*s)]
CSRP_DRG = 6.407E-05 [m**4/kg**2]
CSRP_SRP = 4.108E-03 [m**4/kg**2]
";

    #[test]
    fn reads_the_standard_optional_keyword_example_and_writes_it_back() {
        let cdm = parse_kvn(STANDARD_OPTIONAL_KVN).expect("508.0-B-1 3.6.3 parses");
        assert_eq!(cdm.ccsds_cdm_vers.as_deref(), Some("1.0"));
        assert_eq!(cdm.message_for.as_deref(), Some("SATELLITE A"));
        assert_eq!(cdm.relative_comments, vec!["Relative Metadata/Data"]);
        assert_eq!(
            cdm.relative_position_rtn_m,
            [Some(27.4), Some(-70.2), Some(711.8)]
        );
        assert_eq!(
            cdm.relative_velocity_rtn_m_s,
            [Some(-7.2), Some(-14692.0), Some(-1437.2)]
        );
        assert_eq!(
            cdm.start_screen_period.as_deref(),
            Some("2010-03-12T18:29:32:212")
        );
        assert_eq!(
            cdm.stop_screen_period.as_deref(),
            Some("2010-03-15T18:29:32:212")
        );
        assert_eq!(cdm.screen_volume_frame.as_deref(), Some("RTN"));
        assert_eq!(cdm.screen_volume_shape.as_deref(), Some("ELLIPSOID"));
        assert_eq!(
            cdm.screen_volume_m,
            [Some(200.0), Some(1000.0), Some(1000.0)]
        );
        assert_eq!(
            cdm.screen_entry_time.as_deref(),
            Some("2010-03-13T22:37:52.222")
        );
        assert_eq!(
            cdm.screen_exit_time.as_deref(),
            Some("2010-03-13T22:37:52.824")
        );

        let object1 = &cdm.object1;
        assert_eq!(object1.metadata_comments, vec!["Object1 Metadata"]);
        assert_eq!(object1.operator_contact_position.as_deref(), Some("OSA"));
        assert_eq!(object1.operator_phone.as_deref(), Some("+49615130312"));
        let od = &object1.od_parameters;
        // Both comments precede TIME_LASTOB_START, the first keyword of the
        // OD parameters block.
        assert_eq!(od.comments, vec!["Object1 Data", "Object1 OD Parameters"]);
        assert_eq!(
            od.time_lastob_start.as_deref(),
            Some("2010-03-12T02:14:12.746")
        );
        assert_eq!(
            od.time_lastob_end.as_deref(),
            Some("2010-03-12T02:14:12.746")
        );
        assert_eq!(od.recommended_od_span_d, Some(7.88));
        assert_eq!(od.actual_od_span_d, Some(5.5));
        assert_eq!(
            (
                od.obs_available,
                od.obs_used,
                od.tracks_available,
                od.tracks_used
            ),
            (Some(592), Some(579), Some(123), Some(119))
        );
        assert_eq!(od.residuals_accepted_pct, Some(97.8));
        assert_eq!(od.weighted_rms, Some(0.864));
        let additional = &object1.additional_parameters;
        assert_eq!(
            additional.comments,
            vec![
                "Object1 Additional Parameters",
                "Apogee Altitude=779 km",
                "Perigee Altitude=765 km",
                "Inclination=86.4 deg",
            ]
        );
        assert_eq!(additional.area_pc_m2, Some(5.2));
        assert_eq!(additional.area_drg_m2, None);
        assert_eq!(additional.mass_kg, Some(251.6));
        assert_eq!(additional.cd_area_over_mass_m2_kg, Some(0.045663));
        assert_eq!(additional.cr_area_over_mass_m2_kg, Some(0.0));
        assert_eq!(additional.thrust_acceleration_m_s2, Some(0.0));
        assert_eq!(additional.sedr_w_kg, Some(4.5457e-5));
        assert_eq!(object1.state_comments, vec!["Object1 State Vector"]);
        assert_eq!(
            object1.covariance_comments,
            vec!["Object1 Covariance in the RTN Coordinate Frame"]
        );
        assert_eq!(
            object1.drag_covariance_rtn,
            Some([-1.862, 3.53, -0.31, -1.214e-4, 2.58e-4, -6.467e-5, 3.483e-6])
        );
        assert_eq!(
            object1.srp_covariance_rtn,
            Some([-149.2, 204.4, -23.31, -1.254e-3, 2.013e-2, -4.7e-3, 2.21e-4, 1.593e-2])
        );
        assert_eq!(object1.thrust_covariance_rtn, None);
        assert_eq!(cdm.object2.additional_parameters.mass_kg, None);
        assert_eq!(
            cdm.object2.srp_covariance_rtn.map(|row| row[7]),
            Some(4.108e-3)
        );

        let kvn = encode_kvn(&cdm).unwrap();
        assert!(kvn
            .contains("COMMENT Object1 Data\nCOMMENT Object1 OD Parameters\nTIME_LASTOB_START = "));
        assert_eq!(parse_kvn(&kvn).unwrap(), cdm);
        let xml = encode_xml(&cdm).unwrap();
        assert!(xml.contains("<relativeStateVector>"));
        assert!(xml.contains("<odParameters>"));
        assert!(xml.contains(r#"<CSRP_SRP units="m**4/kg**2">0.01593</CSRP_SRP>"#));
        assert_eq!(parse_xml(&xml).unwrap(), cdm);
    }

    #[test]
    fn unknown_keywords_and_malformed_lines_are_refused() {
        // 508.0-B-1 3.6.3 prints TRACKS_USED with a blank. A keyword holds no
        // blank (6.3.1.5), and table 3-4 defines no `TRACKS USED`.
        let misprint = STANDARD_OPTIONAL_KVN.replacen("TRACKS_USED = 119", "TRACKS USED = 119", 1);
        assert_eq!(
            parse_kvn(&misprint),
            Err(CdmError::UnknownField("TRACKS USED".to_string()))
        );
        // A keyword the tables do not define is refused when it carries a
        // value; a blank one holds nothing to keep.
        let unknown = FULL_KVN.replacen(
            "MANEUVERABLE = YES\n",
            "MANEUVERABLE = YES\nMANEUVER_COUNT = 3\n",
            1,
        );
        assert_eq!(
            parse_kvn(&unknown),
            Err(CdmError::UnknownField("MANEUVER_COUNT".to_string()))
        );
        let blank = FULL_KVN.replacen(
            "MANEUVERABLE = YES\n",
            "MANEUVERABLE = YES\nMANEUVER_COUNT =\n",
            1,
        );
        assert_eq!(parse_kvn(&blank).unwrap(), parse_kvn(FULL_KVN).unwrap());
        // A line that is neither blank, a comment, nor an assignment was
        // skipped; it is refused with its number.
        let garbage = FULL_KVN.replacen("TCA = ", "not an assignment\nTCA = ", 1);
        let line = garbage
            .lines()
            .position(|text| text == "not an assignment")
            .unwrap()
            + 1;
        assert_eq!(
            parse_kvn(&garbage),
            Err(CdmError::MalformedLine {
                line,
                text: "not an assignment".to_string(),
            })
        );
        // An object keyword before the first OBJECT line belongs to neither
        // object.
        let early = FULL_KVN.replacen("TCA = ", "X = 1.0 [km]\nTCA = ", 1);
        assert_eq!(
            parse_kvn(&early),
            Err(CdmError::UnknownField("X".to_string()))
        );

        let xml = FULL_XML.replacen("<TCA>", "<MANEUVER_COUNT>3</MANEUVER_COUNT><TCA>", 1);
        assert_eq!(
            parse_xml(&xml),
            Err(CdmError::UnknownField(
                "relativeMetadataData/MANEUVER_COUNT".to_string()
            ))
        );
        let xml = FULL_XML.replacen(
            r#"<X units="km">2570.097065</X>"#,
            r#"<X units="km">2570.097065</X><MASS units="kg">251.6</MASS>"#,
            1,
        );
        assert_eq!(
            parse_xml(&xml),
            Err(CdmError::UnknownField("stateVector/MASS".to_string()))
        );
        // An empty unknown element holds nothing to keep.
        let xml = FULL_XML.replacen("<TCA>", "<MANEUVER_COUNT/><TCA>", 1);
        assert_eq!(parse_xml(&xml).unwrap(), parse_xml(FULL_XML).unwrap());
        // An element inside a COMMENT is refused rather than dropped with only
        // the comment's text kept.
        let xml = FULL_XML.replacen(
            "<COMMENT>Relative Metadata/Data</COMMENT>",
            "<COMMENT>Relative <b>Metadata</b></COMMENT>",
            1,
        );
        assert_eq!(
            parse_xml(&xml),
            Err(CdmError::UnknownField("COMMENT/b".to_string()))
        );
    }

    #[test]
    fn covariance_rows_seven_to_nine_are_read_whole_and_written_back() {
        // Row 9 completes the 9x9 matrix of table 3-4.
        let thrust = "\
CTHR_R = 1.0E-06 [m**2/s**2]
CTHR_T = 2.0E-06 [m**2/s**2]
CTHR_N = 3.0E-06 [m**2/s**2]
CTHR_RDOT = 4.0E-09 [m**2/s**3]
CTHR_TDOT = 5.0E-09 [m**2/s**3]
CTHR_NDOT = 6.0E-09 [m**2/s**3]
CTHR_DRG = 7.0E-08 [m**3/(kg*s**2)]
CTHR_SRP = 8.0E-08 [m**3/(kg*s**2)]
CTHR_THR = 9.0E-12 [m**2/s**4]
";
        let kvn = STANDARD_OPTIONAL_KVN.replacen(
            "COMMENT Object2 Metadata\n",
            &format!("{thrust}COMMENT Object2 Metadata\n"),
            1,
        );
        let cdm = parse_kvn(&kvn).unwrap();
        assert_eq!(
            cdm.object1.thrust_covariance_rtn,
            Some([1.0e-6, 2.0e-6, 3.0e-6, 4.0e-9, 5.0e-9, 6.0e-9, 7.0e-8, 8.0e-8, 9.0e-12])
        );
        assert_eq!(parse_kvn(&encode_kvn(&cdm).unwrap()).unwrap(), cdm);
        assert_eq!(parse_xml(&encode_xml(&cdm).unwrap()).unwrap(), cdm);

        // 5.2.8 allows no subset of a row; the first missing term is named.
        let partial = STANDARD_OPTIONAL_KVN.replacen("CDRG_N = -3.100E-01 [m**3/kg]\n", "", 1);
        assert_eq!(
            parse_kvn(&partial),
            Err(CdmError::InvalidField {
                field: "CDRG_N",
                kind: CdmInputErrorKind::Missing,
            })
        );
        let wrong_unit = STANDARD_OPTIONAL_KVN.replacen(
            "CDRG_DRG = 3.483E-06 [m**4/kg**2]",
            "CDRG_DRG = 3.483E-06 [m**2]",
            1,
        );
        assert_eq!(
            parse_kvn(&wrong_unit),
            Err(CdmError::UnitMismatch {
                field: "CDRG_DRG".to_string(),
                unit: "m**2".to_string(),
                expected: Some("m**4/kg**2"),
            })
        );
    }

    #[test]
    fn object_blocks_are_assigned_by_the_object_they_state() {
        let start = FULL_KVN.find("OBJECT = OBJECT1").unwrap();
        let middle = FULL_KVN.find("OBJECT = OBJECT2").unwrap();
        let swapped = format!(
            "{}{}{}",
            &FULL_KVN[..start],
            &FULL_KVN[middle..],
            &FULL_KVN[start..middle]
        );
        assert_eq!(parse_kvn(&swapped).unwrap(), parse_kvn(FULL_KVN).unwrap());

        let unknown = FULL_KVN.replacen("OBJECT = OBJECT2", "OBJECT = OBJECT3", 1);
        assert_eq!(
            parse_kvn(&unknown),
            Err(CdmError::UnknownObject("OBJECT3".to_string()))
        );
        let repeated = FULL_KVN.replacen("OBJECT = OBJECT2", "OBJECT = OBJECT1", 1);
        assert_eq!(
            parse_kvn(&repeated),
            Err(CdmError::RepeatedObject("OBJECT1".to_string()))
        );
        // Segments that state no object are taken in order.
        let cdm = parse_xml(&two_segment_xml("SAT", "")).unwrap();
        assert_eq!(cdm.object1.object_name.as_deref(), Some("SAT"));
        assert_eq!(cdm.object2.object_name, None);
    }

    #[test]
    fn writers_refuse_what_the_readers_would_not_return() {
        let base = parse_kvn(STANDARD_OPTIONAL_KVN).unwrap();

        let mut cdm = base.clone();
        cdm.object1.object_name = Some("SATELLITE A\nX = 1".to_string());
        assert_eq!(
            encode_kvn(&cdm),
            Err(CdmError::UnwritableText {
                field: "OBJECT_NAME".to_string(),
                value: "SATELLITE A\nX = 1".to_string(),
                issue: TextIssue::LineBreak,
            })
        );
        // XML element text carries a line break unchanged.
        assert_eq!(parse_xml(&encode_xml(&cdm).unwrap()).unwrap(), cdm);

        let mut cdm = base.clone();
        cdm.screen_volume_frame = Some(" RTN".to_string());
        let expected: Result<String, CdmError> = Err(CdmError::UnwritableText {
            field: "SCREEN_VOLUME_FRAME".to_string(),
            value: " RTN".to_string(),
            issue: TextIssue::SurroundingWhitespace,
        });
        assert_eq!(encode_kvn(&cdm), expected);
        assert_eq!(encode_xml(&cdm), expected);

        // A blank KVN value reads back as empty text; an empty XML element
        // reads back as absent.
        let mut cdm = base.clone();
        cdm.message_for = Some(String::new());
        assert_eq!(parse_kvn(&encode_kvn(&cdm).unwrap()).unwrap(), cdm);
        assert_eq!(
            encode_xml(&cdm),
            Err(CdmError::UnwritableText {
                field: "MESSAGE_FOR".to_string(),
                value: String::new(),
                issue: TextIssue::Empty,
            })
        );

        let mut cdm = base.clone();
        cdm.object2.operator_email = Some("JOHN.DOE@\u{1}SOMEWHERE.NET".to_string());
        assert!(matches!(
            encode_xml(&cdm),
            Err(CdmError::UnwritableText {
                issue: TextIssue::XmlIllegalCharacter,
                ..
            })
        ));

        let mut cdm = base.clone();
        cdm.relative_comments = vec!["Relative Metadata/Data ".to_string()];
        let expected: Result<String, CdmError> = Err(CdmError::UnwritableText {
            field: "COMMENT".to_string(),
            value: "Relative Metadata/Data ".to_string(),
            issue: TextIssue::SurroundingWhitespace,
        });
        assert_eq!(encode_kvn(&cdm), expected);
        assert_eq!(encode_xml(&cdm), expected);

        let mut cdm = base.clone();
        cdm.screen_volume_m[0] = Some(f64::INFINITY);
        let expected: Result<String, CdmError> = Err(CdmError::InvalidField {
            field: "SCREEN_VOLUME_X",
            kind: CdmInputErrorKind::NonFinite,
        });
        assert_eq!(encode_kvn(&cdm), expected);
        assert_eq!(encode_xml(&cdm), expected);

        let mut cdm = base.clone();
        cdm.object1.srp_covariance_rtn = Some([f64::NAN; 8]);
        assert_eq!(
            encode_kvn(&cdm),
            Err(CdmError::InvalidField {
                field: "srp_covariance_rtn",
                kind: CdmInputErrorKind::NonFinite,
            })
        );

        // Comments of an OD parameters block without a value precede a blank
        // RECOMMENDED_OD_SPAN in KVN, which reads back as absent, so they
        // stay in their block; XML keeps the empty block.
        let mut cdm = base;
        cdm.object2.od_parameters = CdmOdParameters {
            comments: vec!["Object2 OD Parameters".to_string()],
            ..CdmOdParameters::default()
        };
        let encoded = encode_kvn(&cdm).unwrap();
        assert!(encoded.contains("COMMENT Object2 OD Parameters\nRECOMMENDED_OD_SPAN =\n"));
        assert_eq!(parse_kvn(&encoded).unwrap(), cdm);
        assert_eq!(parse_xml(&encode_xml(&cdm).unwrap()).unwrap(), cdm);

        // Header comments have no such keyword: a blank header value reads
        // back as empty text.
        let mut cdm = bare_message(cdm.object1.clone(), cdm.object2.clone());
        cdm.comments = vec!["Sample CDM".to_string()];
        assert_eq!(
            encode_kvn(&cdm),
            Err(CdmError::UnwritableText {
                field: "COMMENT".to_string(),
                value: "Sample CDM".to_string(),
                issue: TextIssue::DetachedComment,
            })
        );
    }

    #[test]
    fn hard_body_radius_comments_are_kept_and_checked() {
        // A comment in the form the writers produce is the radius alone.
        let kvn = FULL_KVN.replacen(
            "COMMENT Relative Metadata/Data\n",
            "COMMENT HBR = 20\nCOMMENT Relative Metadata/Data\n",
            1,
        );
        let cdm = parse_kvn(&kvn).unwrap();
        assert_eq!(cdm.hard_body_radius_m, Some(20.0));
        assert_eq!(cdm.relative_comments, vec!["Relative Metadata/Data"]);
        let encoded = encode_kvn(&cdm).unwrap();
        assert!(encoded.contains("COMMENT HBR = 20\nCOMMENT Relative Metadata/Data\n"));
        assert_eq!(parse_kvn(&encoded).unwrap(), cdm);

        // Any other spelling stays among the comments where it was and is
        // written back once, as it was.
        let kvn = FULL_KVN.replacen(
            "COMMENT Relative Metadata/Data\n",
            "COMMENT Relative Metadata/Data\nCOMMENT HBR = 20.0 [m]\n",
            1,
        );
        let cdm = parse_kvn(&kvn).unwrap();
        assert_eq!(cdm.hard_body_radius_m, Some(20.0));
        assert_eq!(
            cdm.relative_comments,
            vec!["Relative Metadata/Data", "HBR = 20.0 [m]"]
        );
        let encoded = encode_kvn(&cdm).unwrap();
        assert_eq!(encoded.matches("HBR").count(), 1);
        assert_eq!(parse_kvn(&encoded).unwrap(), cdm);
        assert_eq!(parse_xml(&encode_xml(&cdm).unwrap()).unwrap(), cdm);

        // A new radius is stated ahead of the retained comment, which then
        // reads back as a comment.
        let mut changed = cdm.clone();
        changed.hard_body_radius_m = Some(25.0);
        assert_eq!(parse_kvn(&encode_kvn(&changed).unwrap()).unwrap(), changed);

        // A header comment is read before the relative metadata/data, so a
        // new radius is stated ahead of a header comment in the convention.
        let mut header = cdm;
        header.relative_comments = vec!["Relative Metadata/Data".to_string()];
        header.comments = vec!["HBR = 20.0 [m]".to_string()];
        header.hard_body_radius_m = Some(25.0);
        let encoded = encode_kvn(&header).unwrap();
        assert!(encoded.contains("COMMENT HBR = 25\nCOMMENT HBR = 20.0 [m]\nCREATION_DATE = "));
        assert_eq!(parse_kvn(&encoded).unwrap(), header);
        assert_eq!(parse_xml(&encode_xml(&header).unwrap()).unwrap(), header);
        header.hard_body_radius_m = Some(20.0);
        assert_eq!(parse_kvn(&encode_kvn(&header).unwrap()).unwrap(), header);
        // A comment that would read back as a radius the message does not hold
        // is refused.
        header.hard_body_radius_m = None;
        let expected: Result<String, CdmError> = Err(CdmError::HardBodyRadiusComment {
            comment: "HBR = 20.0 [m]".to_string(),
        });
        assert_eq!(encode_kvn(&header), expected);
        assert_eq!(encode_xml(&header), expected);

        // An HBR element and a comment stating another radius disagree.
        let xml = FULL_XML.replacen(
            "<TCA>",
            r#"<COMMENT>HBR = 6</COMMENT><HBR units="m">5</HBR><TCA>"#,
            1,
        );
        assert_eq!(
            parse_xml(&xml),
            Err(CdmError::DuplicateField {
                field: "HBR".to_string(),
                first: "5".to_string(),
                second: "HBR = 6".to_string(),
            })
        );
        // An HBR element or keyword is written as the comment and reads back.
        let xml = FULL_XML.replacen("<TCA>", r#"<HBR units="m">5</HBR><TCA>"#, 1);
        let cdm = parse_xml(&xml).unwrap();
        assert_eq!(cdm.hard_body_radius_m, Some(5.0));
        assert_eq!(parse_xml(&encode_xml(&cdm).unwrap()).unwrap(), cdm);
        let kvn = FULL_KVN.replacen("TCA = ", "HBR = 5 [m]\nTCA = ", 1);
        let cdm = parse_kvn(&kvn).unwrap();
        assert_eq!(cdm.hard_body_radius_m, Some(5.0));
        assert_eq!(parse_kvn(&encode_kvn(&cdm).unwrap()).unwrap(), cdm);
    }

    #[test]
    fn relative_comments_before_only_a_radius_read_back_in_place() {
        // Relative metadata/data holding a comment and a hard-body radius but
        // no keyword of table 3-2.
        let kvn = FULL_KVN.replacen(
            "COMMENT Relative Metadata/Data\n\
TCA = 2010-03-13T22:37:52.618\n\
MISS_DISTANCE = 715 [m]\n\
RELATIVE_SPEED = 14762 [m/s]\n\
COLLISION_PROBABILITY = 4.835E-05\n\
COLLISION_PROBABILITY_METHOD = FOSTER-1992\n",
            "COMMENT Relative Metadata/Data\nHBR = 5 [m]\n",
            1,
        );
        let cdm = parse_kvn(&kvn).unwrap();
        assert_eq!(cdm.relative_comments, vec!["Relative Metadata/Data"]);
        assert_eq!(cdm.hard_body_radius_m, Some(5.0));
        assert_eq!(cdm.tca, None);

        // The comment is written before a blank MISS_DISTANCE, which reads
        // back as absent, so the comment reads back in its block.
        let encoded = encode_kvn(&cdm).unwrap();
        assert!(
            encoded.contains("COMMENT Relative Metadata/Data\nMISS_DISTANCE =\nOBJECT = OBJECT1")
        );
        assert_eq!(parse_kvn(&encoded).unwrap(), cdm);
        assert_eq!(parse_xml(&encode_xml(&cdm).unwrap()).unwrap(), cdm);

        // A retained comment stating another radius still reads back as a
        // comment, behind the one that states this radius.
        let mut other = cdm.clone();
        other.relative_comments.push("HBR = 7.0 [m]".to_string());
        assert_eq!(parse_kvn(&encode_kvn(&other).unwrap()).unwrap(), other);

        // Without a radius the blank MISS_DISTANCE still keeps the comment in
        // its block; a blank numeric value reads as absent.
        let mut without = cdm;
        without.hard_body_radius_m = None;
        assert_eq!(parse_kvn(&encode_kvn(&without).unwrap()).unwrap(), without);
        assert_eq!(parse_xml(&encode_xml(&without).unwrap()).unwrap(), without);
    }

    #[test]
    fn the_stated_version_is_kept() {
        // The writers wrote 1.0 whatever the message stated.
        let kvn = FULL_KVN.replacen("CCSDS_CDM_VERS = 1.0", "CCSDS_CDM_VERS = 2.0", 1);
        let cdm = parse_kvn(&kvn).unwrap();
        assert_eq!(cdm.ccsds_cdm_vers.as_deref(), Some("2.0"));
        assert!(encode_kvn(&cdm)
            .unwrap()
            .starts_with("CCSDS_CDM_VERS = 2.0\n"));
        assert!(encode_xml(&cdm)
            .unwrap()
            .contains(r#"<cdm id="CCSDS_CDM_VERS" version="2.0">"#));

        // The version attribute and a header element must agree.
        let xml = FULL_XML.replacen(
            "<CCSDS_CDM_VERS>1.0</CCSDS_CDM_VERS>",
            "<CCSDS_CDM_VERS>2.0</CCSDS_CDM_VERS>",
            1,
        );
        assert_eq!(
            parse_xml(&xml),
            Err(CdmError::DuplicateField {
                field: "CCSDS_CDM_VERS".to_string(),
                first: "1.0".to_string(),
                second: "2.0".to_string(),
            })
        );

        // A message that states none is written without one.
        let unstated = parse_kvn(&FULL_KVN.replacen("CCSDS_CDM_VERS = 1.0\n", "", 1)).unwrap();
        assert_eq!(unstated.ccsds_cdm_vers, None);
        assert!(!encode_kvn(&unstated).unwrap().contains("CCSDS_CDM_VERS"));
        assert_eq!(
            parse_xml(&encode_xml(&unstated).unwrap()).unwrap(),
            unstated
        );
    }

    /// CCSDS 508.0-B-1 4.4, the CDM/XML example, without the page breaks.
    const STANDARD_XML: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<cdm xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xmlns:ndm="urn:ccsds:schema:ndmxml"
xsi:noNamespaceSchemaLocation="https://sanaregistry.org/r/ndmxml_unqualified/ndmxml-2.0.0-master-2.0.xsd"
id="CCSDS_CDM_VERS" version="1.0">
 <header>
  <COMMENT>Sample CDM - XML version</COMMENT>
  <CREATION_DATE>2010-03-12T22:31:12.000</CREATION_DATE>
  <ORIGINATOR>JSPOC</ORIGINATOR>
  <MESSAGE_FOR>SATELLITE A</MESSAGE_FOR>
  <MESSAGE_ID>20111371985</MESSAGE_ID>
 </header>
 <body>
  <relativeMetadataData>
   <COMMENT>Relative Metadata/Data</COMMENT>
   <TCA>2010-03-13T22:37:52.618</TCA>
   <MISS_DISTANCE units="m">715</MISS_DISTANCE>
   <RELATIVE_SPEED units="m/s">14762</RELATIVE_SPEED>
   <relativeStateVector>
     <RELATIVE_POSITION_R units="m">27.4</RELATIVE_POSITION_R>
     <RELATIVE_POSITION_T units="m">-70.2</RELATIVE_POSITION_T>
     <RELATIVE_POSITION_N units="m">711.8</RELATIVE_POSITION_N>
     <RELATIVE_VELOCITY_R units="m/s">-7.2</RELATIVE_VELOCITY_R>
     <RELATIVE_VELOCITY_T units="m/s">-14692.0</RELATIVE_VELOCITY_T>
     <RELATIVE_VELOCITY_N units="m/s">-1437.2</RELATIVE_VELOCITY_N>
   </relativeStateVector>
   <START_SCREEN_PERIOD>2010-03-12T18:29:32.212</START_SCREEN_PERIOD>
   <STOP_SCREEN_PERIOD>2010-03-15T18:29:32.212</STOP_SCREEN_PERIOD>
   <SCREEN_VOLUME_FRAME>RTN</SCREEN_VOLUME_FRAME>
   <SCREEN_VOLUME_SHAPE>ELLIPSOID</SCREEN_VOLUME_SHAPE>
   <SCREEN_VOLUME_X units="m">200</SCREEN_VOLUME_X>
   <SCREEN_VOLUME_Y units="m">1000</SCREEN_VOLUME_Y>
   <SCREEN_VOLUME_Z units="m">1000</SCREEN_VOLUME_Z>
   <SCREEN_ENTRY_TIME>2010-03-13T20:25:43.222</SCREEN_ENTRY_TIME>
   <SCREEN_EXIT_TIME>2010-03-13T23:44:29.324</SCREEN_EXIT_TIME>
   <COLLISION_PROBABILITY>4.835E-05</COLLISION_PROBABILITY>
   <COLLISION_PROBABILITY_METHOD>FOSTER-1992</COLLISION_PROBABILITY_METHOD>
  </relativeMetadataData>
  <segment>
   <metadata>
    <COMMENT>Object1 Metadata</COMMENT>
    <OBJECT>OBJECT1</OBJECT>
    <OBJECT_DESIGNATOR>12345</OBJECT_DESIGNATOR>
    <CATALOG_NAME>SATCAT</CATALOG_NAME>
    <OBJECT_NAME>SATELLITE A</OBJECT_NAME>
    <INTERNATIONAL_DESIGNATOR>1997-030E</INTERNATIONAL_DESIGNATOR>
    <OBJECT_TYPE>PAYLOAD</OBJECT_TYPE>
    <OPERATOR_CONTACT_POSITION>OSA</OPERATOR_CONTACT_POSITION>
    <OPERATOR_ORGANIZATION>EUMETSAT</OPERATOR_ORGANIZATION>
    <OPERATOR_PHONE>+49615130312</OPERATOR_PHONE>
    <OPERATOR_EMAIL>JOHN.DOE@SOMEWHERE>NET</OPERATOR_EMAIL>
    <EPHEMERIS_NAME>EPHEMERIS SATELLITE A</EPHEMERIS_NAME>
    <COVARIANCE_METHOD>CALCULATED</COVARIANCE_METHOD>
    <MANEUVERABLE>YES</MANEUVERABLE>
    <REF_FRAME>EME2000</REF_FRAME>
    <GRAVITY_MODEL>EGM-96: 36D 36O</GRAVITY_MODEL>
    <ATMOSPHERIC_MODEL>JACCHIA 70 DCA</ATMOSPHERIC_MODEL>
    <N_BODY_PERTURBATIONS>MOON,SUN</N_BODY_PERTURBATIONS>
    <SOLAR_RAD_PRESSURE>NO</SOLAR_RAD_PRESSURE>
    <EARTH_TIDES>NO</EARTH_TIDES>
    <INTRACK_THRUST>NO</INTRACK_THRUST>
   </metadata>
   <data>
    <COMMENT>Object1 Data</COMMENT>
    <odParameters>
     <COMMENT>Object1 OD Parameters</COMMENT>
     <TIME_LASTOB_START>2010-03-12T02:14:12.746</TIME_LASTOB_START>
     <TIME_LASTOB_END>2010-03-12T02:14:12.746</TIME_LASTOB_END>
     <RECOMMENDED_OD_SPAN units="d">7.88</RECOMMENDED_OD_SPAN>
     <ACTUAL_OD_SPAN units="d">5.50</ACTUAL_OD_SPAN>
     <OBS_AVAILABLE>592</OBS_AVAILABLE>
     <OBS_USED>59</OBS_USED>
     <TRACKS_AVAILABLE>123</TRACKS_AVAILABLE>
     <TRACKS_USED>119</TRACKS_USED>
     <RESIDUALS_ACCEPTED units="%" >97.8</RESIDUALS_ACCEPTED>
     <WEIGHTED_RMS>0.864</WEIGHTED_RMS>
    </odParameters>
    <additionalParameters>
     <COMMENT>Object 1 Additional Parameters</COMMENT>
     <AREA_PC units="m**2">5.2</AREA_PC>
     <MASS units="kg">2516</MASS>
     <CD_AREA_OVER_MASS units="m**2/kg">0.045663</CD_AREA_OVER_MASS>
     <CR_AREA_OVER_MASS units="m**2/kg">0.000000</CR_AREA_OVER_MASS>
     <THRUST_ACCELERATION units="m/s**2">0.0</THRUST_ACCELERATION>
     <SEDR units="W/kg">4.54570E-05</SEDR>
    </additionalParameters>
    <stateVector>
     <COMMENT>Object1 State Vector</COMMENT>
     <X units="km">2570.097065</X>
     <Y units="km">2244.654904</Y>
     <Z units="km">6281.497978</Z>
     <X_DOT units="km/s">4.418769571</X_DOT>
     <Y_DOT units="km/s">4.833547743</Y_DOT>
     <Z_DOT units="km/s">-3.526774282</Z_DOT>
    </stateVector>
    <covarianceMatrix>
     <COMMENT>Object1 Covariance in the RTN Coordinate Frame </COMMENT>
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
    <COMMENT>Object2 Metadata</COMMENT>
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
    <N_BODY_PERTURBATIONS>MOON,SUN</N_BODY_PERTURBATIONS>
    <SOLAR_RAD_PRESSURE>YES</SOLAR_RAD_PRESSURE>
    <EARTH_TIDES>NO</EARTH_TIDES>
    <INTRACK_THRUST>NO</INTRACK_THRUST>
   </metadata>
   <data>
    <COMMENT>Object2 Data</COMMENT>
    <odParameters>
     <COMMENT>Object2 OD Parameters</COMMENT>
     <TIME_LASTOB_START>2010-03-12T01:14:12.746</TIME_LASTOB_START>
     <TIME_LASTOB_END>2010-03-12T03:14:12.746</TIME_LASTOB_END>
     <RECOMMENDED_OD_SPAN units="d">2.63</RECOMMENDED_OD_SPAN>
     <ACTUAL_OD_SPAN units="d">2.63</ACTUAL_OD_SPAN>
     <OBS_AVAILABLE>59</OBS_AVAILABLE>
     <OBS_USED>58</OBS_USED>
     <TRACKS_AVAILABLE>15</TRACKS_AVAILABLE>
     <TRACKS_USED>15</TRACKS_USED>
     <RESIDUALS_ACCEPTED units="%" >97.8</RESIDUALS_ACCEPTED>
     <WEIGHTED_RMS>0.864</WEIGHTED_RMS>
    </odParameters>
    <additionalParameters>
     <COMMENT>Object2 Additional Parameters</COMMENT>
     <COMMENT>Apogee Altitude=768 km</COMMENT>
     <COMMENT>Perigee Altitude=414 km</COMMENT>
     <COMMENT>Inclination=98.8 deg</COMMENT>
     <AREA_PC units="m**2">0.9</AREA_PC>
     <CD_AREA_OVER_MASS units="m**2/kg">0.118668</CD_AREA_OVER_MASS>
     <CR_AREA_OVER_MASS units="m**2/kg">0.075204</CR_AREA_OVER_MASS>
     <THRUST_ACCELERATION units="m/s**2">0.0</THRUST_ACCELERATION>
     <SEDR units="W/kg">5.40900E-03</SEDR>
    </additionalParameters>
    <stateVector>
     <COMMENT>Object2 State Vector</COMMENT>
     <X units="km">2569.540800</X>
     <Y units="km">2245.093614</Y>
     <Z units="km">6281.599946</Z>
     <X_DOT units="km/s">-2.888612500</X_DOT>
     <Y_DOT units="km/s">-6.007247516</Y_DOT>
     <Z_DOT units="km/s">3.328770172</Z_DOT>
    </stateVector>
    <covarianceMatrix>
     <COMMENT>Object2 Covariance in the RTN Coordinate Frame</COMMENT>
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
    fn reads_the_standard_xml_example_and_writes_it_back() {
        let cdm = parse_xml(STANDARD_XML).expect("508.0-B-1 4.4 parses");
        assert_eq!(cdm.ccsds_cdm_vers.as_deref(), Some("1.0"));
        assert_eq!(cdm.comments, vec!["Sample CDM - XML version"]);
        assert_eq!(cdm.message_for.as_deref(), Some("SATELLITE A"));
        assert_eq!(cdm.relative_comments, vec!["Relative Metadata/Data"]);
        assert_eq!(
            cdm.relative_velocity_rtn_m_s,
            [Some(-7.2), Some(-14692.0), Some(-1437.2)]
        );
        assert_eq!(
            cdm.screen_entry_time.as_deref(),
            Some("2010-03-13T20:25:43.222")
        );
        let object1 = &cdm.object1;
        assert_eq!(object1.metadata_comments, vec!["Object1 Metadata"]);
        assert_eq!(
            object1.operator_email.as_deref(),
            Some("JOHN.DOE@SOMEWHERE>NET")
        );
        // A comment directly in <data> belongs to the block after it, as the
        // same comment does in KVN.
        assert_eq!(
            object1.od_parameters.comments,
            vec!["Object1 Data", "Object1 OD Parameters"]
        );
        assert_eq!(object1.od_parameters.obs_used, Some(59));
        assert_eq!(object1.additional_parameters.mass_kg, Some(2516.0));
        assert_eq!(object1.state_comments, vec!["Object1 State Vector"]);
        // Trailing whitespace of a comment is not significant.
        assert_eq!(
            object1.covariance_comments,
            vec!["Object1 Covariance in the RTN Coordinate Frame"]
        );
        assert_eq!(
            cdm.object2.additional_parameters.comments,
            vec![
                "Object2 Additional Parameters",
                "Apogee Altitude=768 km",
                "Perigee Altitude=414 km",
                "Inclination=98.8 deg",
            ]
        );

        assert_eq!(parse_xml(&encode_xml(&cdm).unwrap()).unwrap(), cdm);
        // KVN carries the same content, comments included.
        assert_eq!(parse_kvn(&encode_kvn(&cdm).unwrap()).unwrap(), cdm);
    }
}
