#![cfg(sidereon_repo_tests)]
//! Authoritative CDM KVN reader/writer gate, ported from the sidereon Elixir suite.
//!
//! The primary fixture is the CCSDS 508.0-B-1 CDMExample2 (JSPOC, SATELLITE A vs
//! FENGYUN 1C DEB). Reference Pc = 4.835e-05, miss distance = 715 m, relative
//! speed = 14762 m/s. Every parsed field is checked, and a KVN re-encode must
//! parse back to the identical bits, proving the Rust codec reproduces the
//! historical Elixir behavior on its own.

use sidereon_core::astro::cdm::{self, CdmError, CdmInputErrorKind};

const EXAMPLE2: &str = include_str!("fixtures/cdm/ccsds_example2.kvn");
const EXAMPLE2_XML: &str = include_str!("fixtures/cdm/ccsds_example2.xml");

#[test]
fn parses_header_and_relative_metadata() {
    let cdm = cdm::parse_kvn(EXAMPLE2).unwrap();

    assert_eq!(
        cdm.creation_date.as_deref(),
        Some("2010-03-12T22:31:12.000")
    );
    assert_eq!(cdm.originator.as_deref(), Some("JSPOC"));
    assert_eq!(cdm.message_id.as_deref(), Some("201113719185"));
    assert_eq!(cdm.tca.as_deref(), Some("2010-03-13T22:37:52.618"));

    // f64 equality is bit equality for these finite values: the decimal literals
    // compile to the nearest f64, so this pins the value parse to 0 ULP.
    assert_eq!(cdm.miss_distance_m, Some(715.0));
    assert_eq!(cdm.relative_speed_m_s, Some(14762.0));
    assert_eq!(cdm.collision_probability, Some(4.835e-5));
    assert_eq!(
        cdm.collision_probability_method.as_deref(),
        Some("FOSTER-1992")
    );
    // HBR is only set from a COMMENT line; this fixture has none.
    assert_eq!(cdm.hard_body_radius_m, None);
}

#[test]
fn parses_object1_metadata_state_and_covariance() {
    let object = cdm::parse_kvn(EXAMPLE2).unwrap().object1;

    assert_eq!(object.object_designator.as_deref(), Some("12345"));
    assert_eq!(object.catalog_name.as_deref(), Some("SATCAT"));
    assert_eq!(object.object_name.as_deref(), Some("SATELLITE A"));
    assert_eq!(
        object.international_designator.as_deref(),
        Some("1997-030E")
    );
    assert_eq!(object.object_type.as_deref(), Some("PAYLOAD"));
    assert_eq!(object.ref_frame.as_deref(), Some("EME2000"));

    assert_eq!(
        object.state,
        (
            (2570.097065, 2244.654904, 6281.497978),
            (4.418769571, 4.833547743, -3.526774282)
        )
    );
    assert_eq!(
        object.covariance_rtn,
        [41.42, -8.579, 2533.0, -23.13, 13.36, 70.98]
    );
}

#[test]
fn parses_object2_metadata_state_and_covariance() {
    let object = cdm::parse_kvn(EXAMPLE2).unwrap().object2;

    assert_eq!(object.object_designator.as_deref(), Some("30337"));
    assert_eq!(object.object_name.as_deref(), Some("FENGYUN 1C DEB"));
    assert_eq!(
        object.international_designator.as_deref(),
        Some("1999-025AA")
    );
    assert_eq!(object.object_type.as_deref(), Some("DEBRIS"));
    assert_eq!(object.ref_frame.as_deref(), Some("EME2000"));

    assert_eq!(
        object.state,
        (
            (2569.540800, 2245.093614, 6281.599946),
            (-2.888612500, -6.007247516, 3.328770172)
        )
    );
    assert_eq!(
        object.covariance_rtn,
        [1337.0, -48060.0, 2492000.0, -32.98, -758.88, 71.05]
    );
}

#[test]
fn missing_state_component_is_rejected() {
    let incomplete = "\
CREATION_DATE = 2024-01-01T00:00:00.000
MESSAGE_ID = INCOMPLETE
TCA = 2024-01-01T12:00:00.000
OBJECT = OBJECT1
OBJECT_DESIGNATOR = 00001
X = 7000.0 [km]
OBJECT = OBJECT2
OBJECT_DESIGNATOR = 00002
";
    assert_eq!(
        cdm::parse_kvn(incomplete),
        Err(CdmError::IncompleteStateVector)
    );
}

#[test]
fn hbr_is_recovered_from_comment() {
    let kvn = "\
CREATION_DATE = 2024-01-01T00:00:00.000
MESSAGE_ID = HBR_TEST
COMMENT HBR = 15.5
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
    assert_eq!(cdm::parse_kvn(kvn).unwrap().hard_body_radius_m, Some(15.5));
}

#[test]
fn missing_kvn_covariance_component_is_rejected() {
    let missing = EXAMPLE2.replace("CN_N                          = 7.098E+01 [m**2]\n", "");

    assert_eq!(
        cdm::parse_kvn(&missing),
        Err(CdmError::InvalidField {
            field: "CN_N",
            kind: CdmInputErrorKind::Missing,
        })
    );
}

#[test]
fn non_finite_kvn_covariance_component_is_rejected() {
    let non_finite = EXAMPLE2.replace(
        "CN_N                          = 7.098E+01 [m**2]",
        "CN_N                          = NaN [m**2]",
    );

    assert_eq!(
        cdm::parse_kvn(&non_finite),
        Err(CdmError::InvalidField {
            field: "CN_N",
            kind: CdmInputErrorKind::NonFinite,
        })
    );
}

#[test]
fn negative_kvn_covariance_variance_is_rejected() {
    let negative = EXAMPLE2.replace(
        "CR_R                          = 4.142E+01 [m**2]",
        "CR_R                          = -1.0 [m**2]",
    );

    assert_eq!(
        cdm::parse_kvn(&negative),
        Err(CdmError::InvalidField {
            field: "covariance_rtn",
            kind: CdmInputErrorKind::NotPositive,
        })
    );
}

#[test]
fn indefinite_kvn_covariance_is_rejected() {
    let indefinite = EXAMPLE2.replace(
        "CT_R                          = -8.579E+00 [m**2]",
        "CT_R                          = 1.0E+04 [m**2]",
    );

    assert_eq!(
        cdm::parse_kvn(&indefinite),
        Err(CdmError::InvalidField {
            field: "covariance_rtn",
            kind: CdmInputErrorKind::NotPositive,
        })
    );
}

#[test]
fn xml_parses_header_relative_metadata_and_objects() {
    let cdm = cdm::parse_xml(EXAMPLE2_XML).unwrap();

    // Header / relative-metadata fields survive comment and prologue stripping.
    assert_eq!(
        cdm.creation_date.as_deref(),
        Some("2010-03-12T22:31:12.000")
    );
    assert_eq!(cdm.originator.as_deref(), Some("JSPOC"));
    assert_eq!(cdm.message_id.as_deref(), Some("201113719185"));
    assert_eq!(cdm.tca.as_deref(), Some("2010-03-13T22:37:52.618"));
    assert_eq!(cdm.miss_distance_m, Some(715.0));
    assert_eq!(cdm.relative_speed_m_s, Some(14762.0));
    assert_eq!(cdm.collision_probability, Some(4.835e-5));
    assert_eq!(
        cdm.collision_probability_method.as_deref(),
        Some("FOSTER-1992")
    );
    assert_eq!(cdm.hard_body_radius_m, None);

    // The two segments yield the same state and covariance as the KVN fixture,
    // proving the flat leaf-element extraction (the extra CRDOT_R element in the
    // first covariance block is correctly ignored).
    assert_eq!(cdm.object1.object_designator.as_deref(), Some("12345"));
    assert_eq!(cdm.object1.object_name.as_deref(), Some("SATELLITE A"));
    assert_eq!(cdm.object1.ref_frame.as_deref(), Some("EME2000"));
    assert_eq!(
        cdm.object1.state,
        (
            (2570.097065, 2244.654904, 6281.497978),
            (4.418769571, 4.833547743, -3.526774282)
        )
    );
    assert_eq!(
        cdm.object1.covariance_rtn,
        [41.42, -8.579, 2533.0, -23.13, 13.36, 70.98]
    );

    assert_eq!(cdm.object2.object_name.as_deref(), Some("FENGYUN 1C DEB"));
    assert_eq!(cdm.object2.object_type.as_deref(), Some("DEBRIS"));
    assert_eq!(
        cdm.object2.state,
        (
            (2569.540800, 2245.093614, 6281.599946),
            (-2.888612500, -6.007247516, 3.328770172)
        )
    );
    assert_eq!(
        cdm.object2.covariance_rtn,
        [1337.0, -48060.0, 2492000.0, -32.98, -758.88, 71.05]
    );
}

#[test]
fn xml_matches_kvn_parse_of_the_same_message() {
    let from_kvn = cdm::parse_kvn(EXAMPLE2).unwrap();
    let from_xml = cdm::parse_xml(EXAMPLE2_XML).unwrap();

    // The same physical message in either serialization parses to identical bits.
    assert_eq!(from_xml.object1.state, from_kvn.object1.state);
    assert_eq!(from_xml.object2.state, from_kvn.object2.state);
    assert_eq!(
        from_xml.object1.covariance_rtn,
        from_kvn.object1.covariance_rtn
    );
    assert_eq!(
        from_xml.object2.covariance_rtn,
        from_kvn.object2.covariance_rtn
    );
    assert_eq!(from_xml.miss_distance_m, from_kvn.miss_distance_m);
    assert_eq!(from_xml.relative_speed_m_s, from_kvn.relative_speed_m_s);
    assert_eq!(
        from_xml.collision_probability,
        from_kvn.collision_probability
    );
    assert_eq!(from_xml.message_id, from_kvn.message_id);
    assert_eq!(from_xml.tca, from_kvn.tca);
}

#[test]
fn xml_round_trips_to_identical_bits() {
    let original = cdm::parse_xml(EXAMPLE2_XML).unwrap();
    let encoded = cdm::encode_xml(&original).expect("valid CDM XML encode");

    // Shape: a well-formed CCSDS XML document.
    assert!(encoded.trim_start().starts_with("<?xml"));
    assert!(encoded.contains("<cdm "));
    assert!(encoded.contains("<relativeMetadataData>"));
    assert!(encoded.contains("<segment>"));

    let reparsed = cdm::parse_xml(&encoded).unwrap();
    assert_eq!(reparsed.object1.state, original.object1.state);
    assert_eq!(reparsed.object2.state, original.object2.state);
    assert_eq!(
        reparsed.object1.covariance_rtn,
        original.object1.covariance_rtn
    );
    assert_eq!(
        reparsed.object2.covariance_rtn,
        original.object2.covariance_rtn
    );
    assert_eq!(reparsed.miss_distance_m, original.miss_distance_m);
    assert_eq!(reparsed.relative_speed_m_s, original.relative_speed_m_s);
    assert_eq!(
        reparsed.collision_probability,
        original.collision_probability
    );
    assert_eq!(reparsed.message_id, original.message_id);
    assert_eq!(reparsed.tca, original.tca);
    assert_eq!(
        reparsed.object1.object_designator,
        original.object1.object_designator
    );
}

#[test]
fn xml_missing_state_component_is_rejected() {
    let incomplete = "\
<cdm><body>
<segment><data><stateVector><X units=\"km\">7000.0</X></stateVector></data></segment>
<segment><data><stateVector></stateVector></data></segment>
</body></cdm>";
    assert_eq!(
        cdm::parse_xml(incomplete),
        Err(CdmError::IncompleteStateVector)
    );
}

#[test]
fn missing_xml_covariance_component_is_rejected() {
    let missing = EXAMPLE2_XML.replace("          <CN_N units=\"m**2\">7.098E+01</CN_N>\n", "");

    assert_eq!(
        cdm::parse_xml(&missing),
        Err(CdmError::InvalidField {
            field: "CN_N",
            kind: CdmInputErrorKind::Missing,
        })
    );
}

#[test]
fn non_finite_xml_covariance_component_is_rejected() {
    let non_finite = EXAMPLE2_XML.replace(
        "          <CN_N units=\"m**2\">7.098E+01</CN_N>",
        "          <CN_N units=\"m**2\">inf</CN_N>",
    );

    assert_eq!(
        cdm::parse_xml(&non_finite),
        Err(CdmError::InvalidField {
            field: "CN_N",
            kind: CdmInputErrorKind::NonFinite,
        })
    );
}

#[test]
fn indefinite_xml_covariance_is_rejected() {
    let indefinite = EXAMPLE2_XML.replace(
        "          <CT_R units=\"m**2\">-8.579E+00</CT_R>",
        "          <CT_R units=\"m**2\">1.0E+04</CT_R>",
    );

    assert_eq!(
        cdm::parse_xml(&indefinite),
        Err(CdmError::InvalidField {
            field: "covariance_rtn",
            kind: CdmInputErrorKind::NotPositive,
        })
    );
}

#[test]
fn kvn_round_trips_to_identical_bits() {
    let original = cdm::parse_kvn(EXAMPLE2).unwrap();
    let encoded = cdm::encode_kvn(&original).expect("valid CDM KVN encode");
    let reparsed = cdm::parse_kvn(&encoded).unwrap();

    // The re-encode is shortest round-tripping, so every numeric field must come
    // back bit-identical; metadata and the carried date/time strings are exact.
    assert_eq!(reparsed.object1.state, original.object1.state);
    assert_eq!(reparsed.object2.state, original.object2.state);
    assert_eq!(
        reparsed.object1.covariance_rtn,
        original.object1.covariance_rtn
    );
    assert_eq!(
        reparsed.object2.covariance_rtn,
        original.object2.covariance_rtn
    );
    assert_eq!(reparsed.miss_distance_m, original.miss_distance_m);
    assert_eq!(reparsed.relative_speed_m_s, original.relative_speed_m_s);
    assert_eq!(
        reparsed.collision_probability,
        original.collision_probability
    );
    assert_eq!(reparsed.message_id, original.message_id);
    assert_eq!(reparsed.tca, original.tca);
    assert_eq!(
        reparsed.object1.object_designator,
        original.object1.object_designator
    );
}
