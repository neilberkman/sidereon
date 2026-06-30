#![cfg(sidereon_repo_tests)]
//! Env-gated emitter that dumps Area 7 CDM parse/encode reference data as a JSON
//! fixture for the Python binding's pytest.
//!
//! The fixture uses the committed CCSDS CDM example files and computes every
//! expected field plus encoded KVN/XML text through `sidereon-core::astro::cdm`.
//! The Python binding must reproduce these engine values exactly.

use std::path::PathBuf;

use sidereon_core::astro::cdm::{self, CdmKvn, CdmObject};

const KVN_FIXTURE: &str = "tests/fixtures/cdm/ccsds_example2.kvn";
const XML_FIXTURE: &str = "tests/fixtures/cdm/ccsds_example2.xml";
const EXAMPLE2: &str = include_str!("fixtures/cdm/ccsds_example2.kvn");
const EXAMPLE2_XML: &str = include_str!("fixtures/cdm/ccsds_example2.xml");

fn hex(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn opt_hex(value: Option<f64>) -> Option<String> {
    value.map(hex)
}

fn hex3(values: (f64, f64, f64)) -> Vec<String> {
    [values.0, values.1, values.2]
        .iter()
        .map(|&v| hex(v))
        .collect()
}

fn hex6(values: [f64; 6]) -> Vec<String> {
    values.iter().map(|&v| hex(v)).collect()
}

fn object_json(object: &CdmObject) -> serde_json::Value {
    use serde_json::json;

    json!({
        "object_designator": object.object_designator,
        "catalog_name": object.catalog_name,
        "object_name": object.object_name,
        "international_designator": object.international_designator,
        "object_type": object.object_type,
        "ref_frame": object.ref_frame,
        "position_km_hex": hex3(object.state.0),
        "velocity_km_s_hex": hex3(object.state.1),
        "covariance_rtn_hex": hex6(object.covariance_rtn),
    })
}

fn cdm_json(cdm: &CdmKvn) -> serde_json::Value {
    use serde_json::json;

    json!({
        "creation_date": cdm.creation_date,
        "originator": cdm.originator,
        "message_id": cdm.message_id,
        "tca": cdm.tca,
        "miss_distance_m_hex": opt_hex(cdm.miss_distance_m),
        "relative_speed_m_s_hex": opt_hex(cdm.relative_speed_m_s),
        "collision_probability_hex": opt_hex(cdm.collision_probability),
        "collision_probability_method": cdm.collision_probability_method,
        "hard_body_radius_m_hex": opt_hex(cdm.hard_body_radius_m),
        "object1": object_json(&cdm.object1),
        "object2": object_json(&cdm.object2),
    })
}

#[test]
fn cdm_python_reference_self_validates() {
    let from_kvn = cdm::parse_kvn(EXAMPLE2).expect("parse KVN fixture");
    let from_xml = cdm::parse_xml(EXAMPLE2_XML).expect("parse XML fixture");
    assert_eq!(from_kvn.object1.state, from_xml.object1.state);
    assert_eq!(from_kvn.object2.state, from_xml.object2.state);
    assert_eq!(from_kvn.miss_distance_m, from_xml.miss_distance_m);

    let encoded_kvn = cdm::encode_kvn(&from_kvn).expect("valid CDM KVN encode");
    let encoded_xml = cdm::encode_xml(&from_xml).expect("valid CDM XML encode");
    let reparsed_kvn = cdm::parse_kvn(&encoded_kvn).expect("reparse KVN encode");
    let reparsed_xml = cdm::parse_xml(&encoded_xml).expect("reparse XML encode");
    assert_eq!(
        reparsed_kvn.object1.covariance_rtn,
        from_kvn.object1.covariance_rtn
    );
    assert_eq!(
        reparsed_xml.object2.covariance_rtn,
        from_xml.object2.covariance_rtn
    );

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture();
    }
}

fn dump_fixture() {
    use serde_json::json;

    let from_kvn = cdm::parse_kvn(EXAMPLE2).expect("parse KVN fixture");
    let from_xml = cdm::parse_xml(EXAMPLE2_XML).expect("parse XML fixture");

    let doc = json!({
        "source": "cdm_python_reference_self_validates",
        "kvn_fixture": KVN_FIXTURE,
        "xml_fixture": XML_FIXTURE,
        "from_kvn": cdm_json(&from_kvn),
        "from_xml": cdm_json(&from_xml),
        "encoded_kvn": cdm::encode_kvn(&from_kvn).expect("valid CDM KVN encode"),
        "encoded_xml": cdm::encode_xml(&from_xml).expect("valid CDM XML encode"),
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/cdm.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped CDM fixture to {out:?}");
}
