#![cfg(sidereon_repo_tests)]
//! Env-gated emitter that dumps OMM KVN/XML/JSON parse and encode reference data
//! as a JSON fixture for the Python binding's pytest.
//!
//! Every value is computed through `sidereon-core::astro::omm` from the
//! committed CelesTrak fixtures. The Python wrapper is checked against these
//! engine outputs for each supported serialization.

use std::path::PathBuf;

use sidereon_core::astro::omm::{self, Omm, OmmEpoch};

struct Fixture {
    name: &'static str,
    kvn_path: &'static str,
    xml_path: &'static str,
    json_path: &'static str,
    kvn: &'static str,
    xml: &'static str,
    json: &'static str,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "ISS (ZARYA)",
        kvn_path: "tests/fixtures/omm/25544.kvn",
        xml_path: "tests/fixtures/omm/25544.xml",
        json_path: "tests/fixtures/omm/25544.json",
        kvn: include_str!("fixtures/omm/25544.kvn"),
        xml: include_str!("fixtures/omm/25544.xml"),
        json: include_str!("fixtures/omm/25544.json"),
    },
    Fixture {
        name: "NAVSTAR 43",
        kvn_path: "tests/fixtures/omm/24876.kvn",
        xml_path: "tests/fixtures/omm/24876.xml",
        json_path: "tests/fixtures/omm/24876.json",
        kvn: include_str!("fixtures/omm/24876.kvn"),
        xml: include_str!("fixtures/omm/24876.xml"),
        json: include_str!("fixtures/omm/24876.json"),
    },
    Fixture {
        name: "GALAXY 15",
        kvn_path: "tests/fixtures/omm/28884.kvn",
        xml_path: "tests/fixtures/omm/28884.xml",
        json_path: "tests/fixtures/omm/28884.json",
        kvn: include_str!("fixtures/omm/28884.kvn"),
        xml: include_str!("fixtures/omm/28884.xml"),
        json: include_str!("fixtures/omm/28884.json"),
    },
];

fn hex(value: f64) -> String {
    format!("0x{:016x}", value.to_bits())
}

fn epoch_json(epoch: &OmmEpoch) -> serde_json::Value {
    use serde_json::json;

    json!({
        "year": epoch.year,
        "month": epoch.month,
        "day": epoch.day,
        "hour": epoch.hour,
        "minute": epoch.minute,
        "second": epoch.second,
        "microsecond": epoch.microsecond,
        "iso8601": format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:06}",
            epoch.year,
            epoch.month,
            epoch.day,
            epoch.hour,
            epoch.minute,
            epoch.second,
            epoch.microsecond
        ),
    })
}

fn omm_json(omm: &Omm) -> serde_json::Value {
    use serde_json::json;

    json!({
        "ccsds_omm_vers": omm.ccsds_omm_vers,
        "creation_date": omm.creation_date,
        "originator": omm.originator,
        "object_name": omm.object_name,
        "object_id": omm.object_id,
        "center_name": omm.center_name,
        "ref_frame": omm.ref_frame,
        "time_system": omm.time_system,
        "mean_element_theory": omm.mean_element_theory,
        "epoch": epoch_json(&omm.epoch),
        "mean_motion_hex": hex(omm.mean_motion),
        "eccentricity_hex": hex(omm.eccentricity),
        "inclination_deg_hex": hex(omm.inclination_deg),
        "ra_of_asc_node_deg_hex": hex(omm.ra_of_asc_node_deg),
        "arg_of_pericenter_deg_hex": hex(omm.arg_of_pericenter_deg),
        "mean_anomaly_deg_hex": hex(omm.mean_anomaly_deg),
        "ephemeris_type": omm.ephemeris_type,
        "classification_type": omm.classification_type,
        "norad_cat_id": omm.norad_cat_id,
        "element_set_no": omm.element_set_no,
        "rev_at_epoch": omm.rev_at_epoch,
        "bstar_hex": hex(omm.bstar),
        "mean_motion_dot_hex": hex(omm.mean_motion_dot),
        "mean_motion_ddot_hex": hex(omm.mean_motion_ddot),
    })
}

#[test]
fn omm_python_reference_self_validates() {
    for fixture in FIXTURES {
        let kvn =
            omm::parse_kvn(fixture.kvn).unwrap_or_else(|e| panic!("{} KVN: {e}", fixture.name));
        let xml =
            omm::parse_xml(fixture.xml).unwrap_or_else(|e| panic!("{} XML: {e}", fixture.name));
        let json =
            omm::parse_json(fixture.json).unwrap_or_else(|e| panic!("{} JSON: {e}", fixture.name));

        assert_eq!(
            omm::parse_kvn(&omm::encode_kvn(&kvn)).expect("KVN reparse"),
            kvn
        );
        assert_eq!(
            omm::parse_xml(&omm::encode_xml(&xml)).expect("XML reparse"),
            xml
        );
        assert_eq!(
            omm::parse_json(&omm::encode_json(&json)).expect("JSON reparse"),
            json
        );
    }

    if std::env::var("SIDEREON_DUMP_FIXTURES").is_ok() {
        dump_fixture();
    }
}

fn fixture_json(fixture: &Fixture) -> serde_json::Value {
    use serde_json::json;

    let kvn = omm::parse_kvn(fixture.kvn).expect("parse KVN fixture");
    let xml = omm::parse_xml(fixture.xml).expect("parse XML fixture");
    let json_omm = omm::parse_json(fixture.json).expect("parse JSON fixture");

    json!({
        "name": fixture.name,
        "kvn_fixture": fixture.kvn_path,
        "xml_fixture": fixture.xml_path,
        "json_fixture": fixture.json_path,
        "from_kvn": omm_json(&kvn),
        "from_xml": omm_json(&xml),
        "from_json": omm_json(&json_omm),
        "encoded_kvn": omm::encode_kvn(&kvn),
        "encoded_xml": omm::encode_xml(&xml),
        "encoded_json": omm::encode_json(&json_omm),
    })
}

fn dump_fixture() {
    use serde_json::json;

    let doc = json!({
        "source": "omm_python_reference_self_validates",
        "fixtures": FIXTURES.iter().map(fixture_json).collect::<Vec<_>>(),
    });

    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../bindings/python/tests/fixtures/omm.json");
    std::fs::create_dir_all(out.parent().unwrap()).expect("dump: create fixture dir");
    std::fs::write(&out, serde_json::to_string_pretty(&doc).unwrap()).expect("dump: write fixture");
    eprintln!("dumped OMM fixture to {out:?}");
}
