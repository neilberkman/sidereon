//! CCSDS Orbit Parameter Message (OPM) KVN and XML reader/writer.
//!
//! OPM date/time values are preserved as text. The parser validates required
//! scalar presence and numeric fields, but it does not normalize epochs.

use crate::astro::covariance::Covariance6;
use crate::astro::ndm::{
    read_covariance6, write_covariance6, FieldMap, NdmHeader, COVARIANCE6_KEYS,
};
use crate::astro::xml;
use crate::format::fmtnum::fmt_num;
use crate::validate;
use roxmltree::{Document, Node};
use std::fmt;

const COMMENT_PREFIX: &str = "COMMENT";
const OPM_VERSION_KEY: &str = "CCSDS_OPM_VERS";

const METADATA_KEYS: [&str; 5] = [
    "OBJECT_NAME",
    "OBJECT_ID",
    "CENTER_NAME",
    "REF_FRAME",
    "TIME_SYSTEM",
];
const STATE_KEYS: [&str; 7] = ["EPOCH", "X", "Y", "Z", "X_DOT", "Y_DOT", "Z_DOT"];
const KEPLERIAN_KEYS: [&str; 8] = [
    "SEMI_MAJOR_AXIS",
    "ECCENTRICITY",
    "INCLINATION",
    "RA_OF_ASC_NODE",
    "ARG_OF_PERICENTER",
    "TRUE_ANOMALY",
    "MEAN_ANOMALY",
    "GM",
];
const SPACECRAFT_KEYS: [&str; 5] = [
    "MASS",
    "SOLAR_RAD_AREA",
    "SOLAR_RAD_COEFF",
    "DRAG_AREA",
    "DRAG_COEFF",
];
const MANEUVER_KEYS: [&str; 7] = [
    "MAN_EPOCH_IGNITION",
    "MAN_DURATION",
    "MAN_DELTA_MASS",
    "MAN_REF_FRAME",
    "MAN_DV_1",
    "MAN_DV_2",
    "MAN_DV_3",
];

/// Canonical, format-agnostic OPM container.
#[derive(Debug, Clone, PartialEq)]
pub struct Opm {
    pub ccsds_opm_vers: String,
    pub creation_date: Option<String>,
    pub originator: Option<String>,
    pub metadata: OpmMetadata,
    pub state: OpmState,
    pub keplerian: Option<OpmKeplerian>,
    pub spacecraft: Option<OpmSpacecraft>,
    pub covariance: Option<OpmCovariance>,
    pub maneuvers: Vec<OpmManeuver>,
}

/// OPM metadata block.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmMetadata {
    pub object_name: String,
    pub object_id: String,
    pub center_name: String,
    pub ref_frame: String,
    pub time_system: String,
}

/// OPM Cartesian state vector.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmState {
    pub epoch: String,
    pub position_km: [f64; 3],
    pub velocity_km_s: [f64; 3],
}

/// Optional OPM Keplerian elements.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmKeplerian {
    pub semi_major_axis_km: f64,
    pub eccentricity: f64,
    pub inclination_deg: f64,
    pub ra_of_asc_node_deg: f64,
    pub arg_of_pericenter_deg: f64,
    pub anomaly: OpmAnomaly,
    pub gm_km3_s2: f64,
}

/// OPM true or mean anomaly.
#[derive(Debug, Clone, PartialEq)]
pub enum OpmAnomaly {
    True(f64),
    Mean(f64),
}

/// Optional OPM spacecraft parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmSpacecraft {
    pub mass_kg: Option<f64>,
    pub solar_rad_area_m2: Option<f64>,
    pub solar_rad_coeff: Option<f64>,
    pub drag_area_m2: Option<f64>,
    pub drag_coeff: Option<f64>,
}

/// Optional OPM 6x6 covariance.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmCovariance {
    pub cov_ref_frame: Option<String>,
    pub matrix: Covariance6,
}

/// One OPM maneuver block. Every field is mandatory in CCSDS 502.0-B when a
/// maneuver is present, including `MAN_REF_FRAME`.
#[derive(Debug, Clone, PartialEq)]
pub struct OpmManeuver {
    pub epoch_ignition: String,
    pub duration_s: f64,
    pub delta_mass_kg: f64,
    pub ref_frame: String,
    pub dv_km_s: [f64; 3],
}

/// Failure modes of the OPM readers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpmError {
    /// A required field was absent from the message.
    MissingField(&'static str),
    /// A decoded scalar field failed validation.
    InvalidField {
        field: &'static str,
        kind: OpmInputErrorKind,
    },
    /// A structural or XML-level error.
    Field(String),
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
        }
    }
}

impl std::error::Error for OpmError {}

/// Parse a CCSDS OPM in KVN encoding into an [`Opm`].
pub fn parse_kvn(text: &str) -> Result<Opm, OpmError> {
    let lines = significant_lines(text);
    let (base_lines, maneuver_blocks) = split_maneuver_blocks(&lines);
    let map = FieldMap::from_pairs(parse_kv_lines(&base_lines));
    let header = NdmHeader::read(&map, OPM_VERSION_KEY);
    if header.vers.is_empty() {
        return Err(OpmError::MissingField(OPM_VERSION_KEY));
    }

    Ok(Opm {
        ccsds_opm_vers: header.vers,
        creation_date: header.creation_date,
        originator: header.originator,
        metadata: parse_metadata(&map)?,
        state: parse_state(&map)?,
        keplerian: keplerian_present(&map)
            .then(|| parse_keplerian(&map))
            .transpose()?,
        spacecraft: spacecraft_present(&map)
            .then(|| parse_spacecraft(&map))
            .transpose()?,
        covariance: covariance_present(&map)
            .then(|| parse_covariance(&map))
            .transpose()?,
        maneuvers: maneuver_blocks
            .iter()
            .map(|block| parse_maneuver(&FieldMap::from_pairs(parse_kv_lines(block))))
            .collect::<Result<_, _>>()?,
    })
}

/// Encode an [`Opm`] as CCSDS OPM KVN.
pub fn encode_kvn(opm: &Opm) -> String {
    let mut lines = NdmHeader {
        vers: opm.ccsds_opm_vers.clone(),
        creation_date: opm.creation_date.clone(),
        originator: opm.originator.clone(),
    }
    .write_kvn(OPM_VERSION_KEY);

    lines.extend(encode_metadata_kvn(&opm.metadata));
    lines.extend(encode_state_kvn(&opm.state));
    if let Some(keplerian) = &opm.keplerian {
        lines.extend(encode_keplerian_kvn(keplerian));
    }
    if let Some(spacecraft) = &opm.spacecraft {
        lines.extend(encode_spacecraft_kvn(spacecraft));
    }
    if let Some(covariance) = &opm.covariance {
        if let Some(cov_ref_frame) = &covariance.cov_ref_frame {
            lines.push(format!("COV_REF_FRAME = {cov_ref_frame}"));
        }
        lines.extend(write_covariance6(&covariance.matrix));
    }
    for maneuver in &opm.maneuvers {
        lines.extend(encode_maneuver_kvn(maneuver));
    }

    lines.join("\n")
}

/// Parse a CCSDS OPM in XML encoding into an [`Opm`].
pub fn parse_xml(text: &str) -> Result<Opm, OpmError> {
    let doc = Document::parse(text).map_err(|e| OpmError::Field(format!("malformed XML: {e}")))?;
    let opm_node = doc
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "opm")
        .ok_or_else(|| OpmError::Field("missing opm element".to_string()))?;

    let version = opm_node
        .attribute("version")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .or_else(|| node_text(opm_node, OPM_VERSION_KEY))
        .ok_or(OpmError::MissingField(OPM_VERSION_KEY))?;

    let segment = opm_node
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "segment")
        .ok_or_else(|| OpmError::Field("OPM contains no segment".to_string()))?;
    let metadata_node = child_element(segment, "metadata")
        .ok_or_else(|| OpmError::Field("segment missing metadata".to_string()))?;
    let data_node = child_element(segment, "data")
        .ok_or_else(|| OpmError::Field("segment missing data".to_string()))?;
    let state_node = data_node
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "stateVector")
        .ok_or_else(|| OpmError::Field("data missing stateVector".to_string()))?;

    let keplerian_node = data_node
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "keplerianElements");
    let spacecraft_node = data_node
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "spacecraftParameters");
    let covariance_node = data_node
        .descendants()
        .find(|n| n.is_element() && n.tag_name().name() == "covarianceMatrix");

    Ok(Opm {
        ccsds_opm_vers: version,
        creation_date: node_text(opm_node, "CREATION_DATE"),
        originator: node_text(opm_node, "ORIGINATOR"),
        metadata: parse_metadata(&FieldMap::from_pairs(xml_fields(
            metadata_node,
            &METADATA_KEYS,
        )))?,
        state: parse_state(&FieldMap::from_pairs(xml_fields(state_node, &STATE_KEYS)))?,
        keplerian: keplerian_node
            .map(|node| parse_keplerian(&FieldMap::from_pairs(xml_fields(node, &KEPLERIAN_KEYS))))
            .transpose()?,
        spacecraft: spacecraft_node
            .map(|node| parse_spacecraft(&FieldMap::from_pairs(xml_fields(node, &SPACECRAFT_KEYS))))
            .transpose()?,
        covariance: covariance_node
            .map(|node| parse_covariance(&FieldMap::from_pairs(xml_all_fields(node))))
            .transpose()?,
        maneuvers: data_node
            .descendants()
            .filter(|n| n.is_element() && n.tag_name().name() == "maneuverParameters")
            .map(|node| parse_maneuver(&FieldMap::from_pairs(xml_fields(node, &MANEUVER_KEYS))))
            .collect::<Result<_, _>>()?,
    })
}

/// Encode an [`Opm`] as CCSDS OPM XML.
pub fn encode_xml(opm: &Opm) -> String {
    let mut lines = vec![
        r#"<?xml version="1.0" encoding="UTF-8"?>"#.to_string(),
        format!(
            r#"<opm id="CCSDS_OPM_VERS" version="{}">"#,
            xml::escape(&opm.ccsds_opm_vers)
        ),
        "  <header>".to_string(),
        format!(
            "    <CREATION_DATE>{}</CREATION_DATE>",
            xml::escape_opt(&opm.creation_date)
        ),
        format!(
            "    <ORIGINATOR>{}</ORIGINATOR>",
            xml::escape_opt(&opm.originator)
        ),
        "  </header>".to_string(),
        "  <body>".to_string(),
        "    <segment>".to_string(),
        "      <metadata>".to_string(),
        elem_line(8, "OBJECT_NAME", &opm.metadata.object_name),
        elem_line(8, "OBJECT_ID", &opm.metadata.object_id),
        elem_line(8, "CENTER_NAME", &opm.metadata.center_name),
        elem_line(8, "REF_FRAME", &opm.metadata.ref_frame),
        elem_line(8, "TIME_SYSTEM", &opm.metadata.time_system),
        "      </metadata>".to_string(),
        "      <data>".to_string(),
    ];

    lines.extend(encode_xml_state(&opm.state));
    if let Some(keplerian) = &opm.keplerian {
        lines.extend(encode_xml_keplerian(keplerian));
    }
    if let Some(spacecraft) = &opm.spacecraft {
        lines.extend(encode_xml_spacecraft(spacecraft));
    }
    if let Some(covariance) = &opm.covariance {
        lines.extend(encode_xml_covariance(covariance));
    }
    for maneuver in &opm.maneuvers {
        lines.extend(encode_xml_maneuver(maneuver));
    }

    lines.push("      </data>".to_string());
    lines.push("    </segment>".to_string());
    lines.push("  </body>".to_string());
    lines.push("</opm>".to_string());
    lines.join("\n")
}

fn parse_metadata(map: &FieldMap) -> Result<OpmMetadata, OpmError> {
    Ok(OpmMetadata {
        object_name: req_text(map, "OBJECT_NAME")?,
        object_id: req_text(map, "OBJECT_ID")?,
        center_name: req_text(map, "CENTER_NAME")?,
        ref_frame: req_text(map, "REF_FRAME")?,
        time_system: req_text(map, "TIME_SYSTEM")?,
    })
}

fn parse_state(map: &FieldMap) -> Result<OpmState, OpmError> {
    Ok(OpmState {
        epoch: req_text(map, "EPOCH")?,
        position_km: [req_num(map, "X")?, req_num(map, "Y")?, req_num(map, "Z")?],
        velocity_km_s: [
            req_num(map, "X_DOT")?,
            req_num(map, "Y_DOT")?,
            req_num(map, "Z_DOT")?,
        ],
    })
}

/// A Keplerian-elements block is present when any of its keys (KVN) or its XML
/// element is supplied. Once present, every CCSDS-mandatory field is required, so
/// a malformed block errors instead of vanishing.
fn keplerian_present(map: &FieldMap) -> bool {
    KEPLERIAN_KEYS.iter().any(|key| map.get(key).is_some())
}

/// A spacecraft-parameters block is present when any of its keys (KVN) or its XML
/// element is supplied. Its sub-fields are individually optional.
fn spacecraft_present(map: &FieldMap) -> bool {
    SPACECRAFT_KEYS.iter().any(|key| map.get(key).is_some())
}

/// A covariance block is present when its reference-frame label or any of the 21
/// lower-triangle keys (KVN) or its XML element is supplied. Once present, every
/// matrix component is required.
fn covariance_present(map: &FieldMap) -> bool {
    map.get("COV_REF_FRAME").is_some() || COVARIANCE6_KEYS.iter().any(|key| map.get(key).is_some())
}

fn parse_keplerian(map: &FieldMap) -> Result<OpmKeplerian, OpmError> {
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
        semi_major_axis_km: req_num(map, "SEMI_MAJOR_AXIS")?,
        eccentricity: req_num(map, "ECCENTRICITY")?,
        inclination_deg: req_num(map, "INCLINATION")?,
        ra_of_asc_node_deg: req_num(map, "RA_OF_ASC_NODE")?,
        arg_of_pericenter_deg: req_num(map, "ARG_OF_PERICENTER")?,
        anomaly,
        gm_km3_s2: req_num(map, "GM")?,
    })
}

fn parse_spacecraft(map: &FieldMap) -> Result<OpmSpacecraft, OpmError> {
    Ok(OpmSpacecraft {
        mass_kg: opt_num(map, "MASS")?,
        solar_rad_area_m2: opt_num(map, "SOLAR_RAD_AREA")?,
        solar_rad_coeff: opt_num(map, "SOLAR_RAD_COEFF")?,
        drag_area_m2: opt_num(map, "DRAG_AREA")?,
        drag_coeff: opt_num(map, "DRAG_COEFF")?,
    })
}

fn parse_covariance(map: &FieldMap) -> Result<OpmCovariance, OpmError> {
    Ok(OpmCovariance {
        cov_ref_frame: opt_text(map, "COV_REF_FRAME"),
        matrix: read_covariance6(map).map_err(map_opm_field_error)?,
    })
}

fn parse_maneuver(map: &FieldMap) -> Result<OpmManeuver, OpmError> {
    Ok(OpmManeuver {
        epoch_ignition: req_text(map, "MAN_EPOCH_IGNITION")?,
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

fn encode_metadata_kvn(metadata: &OpmMetadata) -> Vec<String> {
    vec![
        format!("OBJECT_NAME = {}", metadata.object_name),
        format!("OBJECT_ID = {}", metadata.object_id),
        format!("CENTER_NAME = {}", metadata.center_name),
        format!("REF_FRAME = {}", metadata.ref_frame),
        format!("TIME_SYSTEM = {}", metadata.time_system),
    ]
}

fn encode_state_kvn(state: &OpmState) -> Vec<String> {
    vec![
        format!("EPOCH = {}", state.epoch),
        format!("X = {}", fmt_num(state.position_km[0])),
        format!("Y = {}", fmt_num(state.position_km[1])),
        format!("Z = {}", fmt_num(state.position_km[2])),
        format!("X_DOT = {}", fmt_num(state.velocity_km_s[0])),
        format!("Y_DOT = {}", fmt_num(state.velocity_km_s[1])),
        format!("Z_DOT = {}", fmt_num(state.velocity_km_s[2])),
    ]
}

fn encode_keplerian_kvn(keplerian: &OpmKeplerian) -> Vec<String> {
    let mut lines = vec![
        format!(
            "SEMI_MAJOR_AXIS = {}",
            fmt_num(keplerian.semi_major_axis_km)
        ),
        format!("ECCENTRICITY = {}", fmt_num(keplerian.eccentricity)),
        format!("INCLINATION = {}", fmt_num(keplerian.inclination_deg)),
        format!("RA_OF_ASC_NODE = {}", fmt_num(keplerian.ra_of_asc_node_deg)),
        format!(
            "ARG_OF_PERICENTER = {}",
            fmt_num(keplerian.arg_of_pericenter_deg)
        ),
    ];
    match keplerian.anomaly {
        OpmAnomaly::True(value) => lines.push(format!("TRUE_ANOMALY = {}", fmt_num(value))),
        OpmAnomaly::Mean(value) => lines.push(format!("MEAN_ANOMALY = {}", fmt_num(value))),
    }
    lines.push(format!("GM = {}", fmt_num(keplerian.gm_km3_s2)));
    lines
}

fn encode_spacecraft_kvn(spacecraft: &OpmSpacecraft) -> Vec<String> {
    let mut lines = Vec::new();
    push_opt_num(&mut lines, "MASS", spacecraft.mass_kg);
    push_opt_num(&mut lines, "SOLAR_RAD_AREA", spacecraft.solar_rad_area_m2);
    push_opt_num(&mut lines, "SOLAR_RAD_COEFF", spacecraft.solar_rad_coeff);
    push_opt_num(&mut lines, "DRAG_AREA", spacecraft.drag_area_m2);
    push_opt_num(&mut lines, "DRAG_COEFF", spacecraft.drag_coeff);
    lines
}

fn encode_maneuver_kvn(maneuver: &OpmManeuver) -> Vec<String> {
    let mut lines = vec![
        format!("MAN_EPOCH_IGNITION = {}", maneuver.epoch_ignition),
        format!("MAN_DURATION = {}", fmt_num(maneuver.duration_s)),
        format!("MAN_DELTA_MASS = {}", fmt_num(maneuver.delta_mass_kg)),
        format!("MAN_REF_FRAME = {}", maneuver.ref_frame),
    ];
    lines.push(format!("MAN_DV_1 = {}", fmt_num(maneuver.dv_km_s[0])));
    lines.push(format!("MAN_DV_2 = {}", fmt_num(maneuver.dv_km_s[1])));
    lines.push(format!("MAN_DV_3 = {}", fmt_num(maneuver.dv_km_s[2])));
    lines
}

fn encode_xml_state(state: &OpmState) -> Vec<String> {
    vec![
        "        <stateVector>".to_string(),
        elem_line(10, "EPOCH", &state.epoch),
        elem_line_raw(10, "X", &fmt_num(state.position_km[0])),
        elem_line_raw(10, "Y", &fmt_num(state.position_km[1])),
        elem_line_raw(10, "Z", &fmt_num(state.position_km[2])),
        elem_line_raw(10, "X_DOT", &fmt_num(state.velocity_km_s[0])),
        elem_line_raw(10, "Y_DOT", &fmt_num(state.velocity_km_s[1])),
        elem_line_raw(10, "Z_DOT", &fmt_num(state.velocity_km_s[2])),
        "        </stateVector>".to_string(),
    ]
}

fn encode_xml_keplerian(keplerian: &OpmKeplerian) -> Vec<String> {
    let mut lines = vec![
        "        <keplerianElements>".to_string(),
        elem_line_raw(
            10,
            "SEMI_MAJOR_AXIS",
            &fmt_num(keplerian.semi_major_axis_km),
        ),
        elem_line_raw(10, "ECCENTRICITY", &fmt_num(keplerian.eccentricity)),
        elem_line_raw(10, "INCLINATION", &fmt_num(keplerian.inclination_deg)),
        elem_line_raw(10, "RA_OF_ASC_NODE", &fmt_num(keplerian.ra_of_asc_node_deg)),
        elem_line_raw(
            10,
            "ARG_OF_PERICENTER",
            &fmt_num(keplerian.arg_of_pericenter_deg),
        ),
    ];
    match keplerian.anomaly {
        OpmAnomaly::True(value) => lines.push(elem_line_raw(10, "TRUE_ANOMALY", &fmt_num(value))),
        OpmAnomaly::Mean(value) => lines.push(elem_line_raw(10, "MEAN_ANOMALY", &fmt_num(value))),
    }
    lines.push(elem_line_raw(10, "GM", &fmt_num(keplerian.gm_km3_s2)));
    lines.push("        </keplerianElements>".to_string());
    lines
}

fn encode_xml_spacecraft(spacecraft: &OpmSpacecraft) -> Vec<String> {
    let mut lines = vec!["        <spacecraftParameters>".to_string()];
    push_opt_xml_num(&mut lines, "MASS", spacecraft.mass_kg);
    push_opt_xml_num(&mut lines, "SOLAR_RAD_AREA", spacecraft.solar_rad_area_m2);
    push_opt_xml_num(&mut lines, "SOLAR_RAD_COEFF", spacecraft.solar_rad_coeff);
    push_opt_xml_num(&mut lines, "DRAG_AREA", spacecraft.drag_area_m2);
    push_opt_xml_num(&mut lines, "DRAG_COEFF", spacecraft.drag_coeff);
    lines.push("        </spacecraftParameters>".to_string());
    lines
}

fn encode_xml_covariance(covariance: &OpmCovariance) -> Vec<String> {
    let mut lines = vec!["        <covarianceMatrix>".to_string()];
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

fn encode_xml_maneuver(maneuver: &OpmManeuver) -> Vec<String> {
    let mut lines = vec![
        "        <maneuverParameters>".to_string(),
        elem_line(10, "MAN_EPOCH_IGNITION", &maneuver.epoch_ignition),
        elem_line_raw(10, "MAN_DURATION", &fmt_num(maneuver.duration_s)),
        elem_line_raw(10, "MAN_DELTA_MASS", &fmt_num(maneuver.delta_mass_kg)),
        elem_line(10, "MAN_REF_FRAME", &maneuver.ref_frame),
    ];
    lines.push(elem_line_raw(10, "MAN_DV_1", &fmt_num(maneuver.dv_km_s[0])));
    lines.push(elem_line_raw(10, "MAN_DV_2", &fmt_num(maneuver.dv_km_s[1])));
    lines.push(elem_line_raw(10, "MAN_DV_3", &fmt_num(maneuver.dv_km_s[2])));
    lines.push("        </maneuverParameters>".to_string());
    lines
}

fn significant_lines(text: &str) -> Vec<String> {
    text.lines()
        .map(|line| line.trim().to_string())
        .filter(|line| !line.is_empty() && !line.starts_with(COMMENT_PREFIX))
        .collect()
}

fn split_maneuver_blocks(lines: &[String]) -> (Vec<String>, Vec<Vec<String>>) {
    let markers: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| {
            line.split_once('=')
                .is_some_and(|(key, _)| key.trim() == "MAN_EPOCH_IGNITION")
        })
        .map(|(idx, _)| idx)
        .collect();

    let Some(first_marker) = markers.first().copied() else {
        return (lines.to_vec(), Vec::new());
    };

    let base = lines[..first_marker].to_vec();
    let mut blocks = Vec::new();
    for (pos, marker) in markers.iter().copied().enumerate() {
        let end = markers.get(pos + 1).copied().unwrap_or(lines.len());
        blocks.push(lines[marker..end].to_vec());
    }
    (base, blocks)
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

fn xml_all_fields(node: Node) -> Vec<(String, String)> {
    node.descendants()
        .filter(Node::is_element)
        .filter_map(|n| {
            let text = n.text()?.trim();
            if text.is_empty() {
                None
            } else {
                Some((n.tag_name().name().to_string(), text.to_string()))
            }
        })
        .collect()
}

fn push_opt_num(lines: &mut Vec<String>, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        lines.push(format!("{key} = {}", fmt_num(value)));
    }
}

fn push_opt_xml_num(lines: &mut Vec<String>, key: &str, value: Option<f64>) {
    if let Some(value) = value {
        lines.push(elem_line_raw(10, key, &fmt_num(value)));
    }
}

fn elem_line(indent: usize, name: &str, value: &str) -> String {
    elem_line_raw(indent, name, &xml::escape(value))
}

fn elem_line_raw(indent: usize, name: &str, value: &str) -> String {
    format!("{:indent$}<{name}>{value}</{name}>", "")
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
            assert_eq!(parse_kvn(&encode_kvn(&opm)).unwrap(), opm);
        }

        #[test]
        fn fixture_xml_round_trips() {
            let opm = parse_xml(OSPREY_XML).unwrap();
            assert_eq!(parse_xml(&encode_xml(&opm)).unwrap(), opm);
        }
    }
}
