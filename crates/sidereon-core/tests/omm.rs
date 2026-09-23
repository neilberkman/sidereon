#![cfg(sidereon_repo_tests)]
//! Authoritative OMM gate: every encoding of an OMM parses to the same
//! orbital content and drives SGP4 identically, bridged as python-sgp4 2.22's
//! `sgp4.omm.initialize` bridges it (the path Skyfield's
//! `EarthSatellite.from_omm` takes).
//!
//! The initialisation itself is checked against python-sgp4 in the unit test
//! `omms_initialise_sgp4_as_python_sgp4_does`, over the three CelesTrak
//! fixtures and 200 generated OMMs, and the SGP4 kernel against the reference
//! build in `sgp4_verification.json`. The TLE of each fixture, captured in the
//! same query, states the same elements; its epoch has eight decimals of a day
//! and its B* is quantized, so a TLE-built `Satellite` is not expected to match
//! the OMM bit for bit (python-sgp4 gives NAVSTAR 43 an OMM epoch 0.36
//! microseconds before its TLE's and GALAXY 15 one 0.23 microseconds after).

use sha2::{Digest, Sha256};
use sidereon_core::astro::omm::{self, Omm};
use sidereon_core::astro::sgp4::{MinutesSinceEpoch, Satellite};

struct Fixture {
    name: &'static str,
    kvn: &'static str,
    xml: &'static str,
    json: &'static str,
    tle: &'static str,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "ISS (ZARYA) - near-Earth SGP4",
        kvn: include_str!("fixtures/omm/25544.kvn"),
        xml: include_str!("fixtures/omm/25544.xml"),
        json: include_str!("fixtures/omm/25544.json"),
        tle: include_str!("fixtures/omm/25544.tle"),
    },
    Fixture {
        name: "NAVSTAR 43 - deep-space SDP4 (12 h)",
        kvn: include_str!("fixtures/omm/24876.kvn"),
        xml: include_str!("fixtures/omm/24876.xml"),
        json: include_str!("fixtures/omm/24876.json"),
        tle: include_str!("fixtures/omm/24876.tle"),
    },
    Fixture {
        name: "GALAXY 15 - deep-space SDP4 (geosynchronous)",
        kvn: include_str!("fixtures/omm/28884.kvn"),
        xml: include_str!("fixtures/omm/28884.xml"),
        json: include_str!("fixtures/omm/28884.json"),
        tle: include_str!("fixtures/omm/28884.tle"),
    },
];

type FrozenOmmCase = (&'static str, &'static str, fn(&Omm) -> String);

/// `encode_kvn` refuses text that would not read back unchanged; the committed
/// fixtures hold none, so their encoding is expected to succeed.
fn encode_kvn_fixture(record: &Omm) -> String {
    omm::encode_kvn(record).expect("fixture OMM encodes as KVN")
}

/// `encode_xml` refuses text the XML reader would not return unchanged; the
/// committed fixtures hold none.
fn encode_xml_fixture(record: &Omm) -> String {
    omm::encode_xml(record).expect("fixture OMM encodes as XML")
}

/// `encode_json` refuses what the JSON reader would not return unchanged, such
/// as a non-finite number; the committed fixtures hold none.
fn encode_json_fixture(record: &Omm) -> String {
    omm::encode_json(record).expect("fixture OMM encodes as JSON")
}

const ISS_CSV: &str = "OBJECT_NAME,OBJECT_ID,EPOCH,MEAN_MOTION,ECCENTRICITY,INCLINATION,RA_OF_ASC_NODE,ARG_OF_PERICENTER,MEAN_ANOMALY,EPHEMERIS_TYPE,CLASSIFICATION_TYPE,NORAD_CAT_ID,ELEMENT_SET_NO,REV_AT_EPOCH,BSTAR,MEAN_MOTION_DOT,MEAN_MOTION_DDOT\n\
ISS (ZARYA),1998-067A,2026-06-17T04:32:52.099296,15.49273435,0.0004737,51.6332,300.0813,195.1146,164.9702,0,U,25544,999,57175,0.00017172,9.113e-5,0";

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Pull the two element lines out of a CelesTrak `.tle` file (a name line plus
/// the two element lines, with CRLF endings).
fn tle_lines(body: &str) -> (String, String) {
    let mut l1 = None;
    let mut l2 = None;
    for line in body.lines() {
        let line = line.trim();
        if line.starts_with("1 ") && l1.is_none() {
            l1 = Some(line.to_string());
        } else if line.starts_with("2 ") && l2.is_none() {
            l2 = Some(line.to_string());
        }
    }
    (
        l1.expect("line 1 in TLE fixture"),
        l2.expect("line 2 in TLE fixture"),
    )
}

/// Reduce an OMM to its canonical orbital + catalog content, blanking the
/// free-text header metadata CelesTrak emits inconsistently across encodings (it
/// labels the element theory `SGP/SGP4` in KVN but `SGP4` in XML/JSON, and its
/// JSON omits `CENTER_NAME`/`REF_FRAME`/`TIME_SYSTEM`). Cross-encoding identity
/// is asserted on this canonical content, which must match exactly.
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

/// Propagation offsets (minutes since epoch), including 0 and a deep-space-sized
/// span so both the SGP4 and SDP4 branches are exercised away from epoch.
const TSINCE_MINUTES: &[f64] = &[-1440.0, 0.0, 10.0, 100.0, 720.0, 1440.0, 4320.0];

/// Assert that `satellite` has the epoch split and, at every offset, the
/// position and velocity of `reference` bit for bit.
fn assert_same_propagation(label: &str, satellite: &Satellite, reference: &Satellite) {
    let (a, b) = (satellite.epoch_jd(), reference.epoch_jd());
    assert_eq!(
        (a.0.to_bits(), a.1.to_bits()),
        (b.0.to_bits(), b.1.to_bits()),
        "{label}: epoch split"
    );
    for &t in TSINCE_MINUTES {
        let got = satellite.propagate(MinutesSinceEpoch(t)).unwrap();
        let want = reference.propagate(MinutesSinceEpoch(t)).unwrap();
        for axis in 0..3 {
            assert_eq!(
                got.position[axis].to_bits(),
                want.position[axis].to_bits(),
                "{label}: position[{axis}] at t={t} min"
            );
            assert_eq!(
                got.velocity[axis].to_bits(),
                want.velocity[axis].to_bits(),
                "{label}: velocity[{axis}] at t={t} min"
            );
        }
    }
}

/// The `Satellite` the fixture's KVN encoding bridges to, as python-sgp4
/// bridges it.
fn kvn_satellite(fix: &Fixture) -> Satellite {
    let kvn = omm::parse_kvn(fix.kvn).unwrap_or_else(|e| panic!("{}: {e}", fix.name));
    let elements = kvn.to_element_set().unwrap();
    assert!(elements.omm_epoch_days.is_some(), "{}", fix.name);
    Satellite::from_omm(&kvn).unwrap_or_else(|e| panic!("{}: {e}", fix.name))
}

#[test]
fn omm_encodings_drive_sgp4_bit_identically() {
    for fix in FIXTURES {
        let kvn = omm::parse_kvn(fix.kvn).unwrap_or_else(|e| panic!("{}: {e}", fix.name));
        let xml = omm::parse_xml(fix.xml).unwrap_or_else(|e| panic!("{}: {e}", fix.name));

        // Cross-encoding identity: every encoding decodes to the same orbital
        // content (only CelesTrak's free-text theory label differs).
        assert_eq!(
            canonical(&kvn),
            canonical(&xml),
            "{}: KVN and XML disagree on orbital content",
            fix.name,
        );

        let reference = kvn_satellite(fix);
        let from_xml =
            Satellite::from_omm(&xml).unwrap_or_else(|e| panic!("{} XML: {e}", fix.name));
        assert_same_propagation(&format!("{} [XML]", fix.name), &from_xml, &reference);
    }
}

#[test]
fn omm_json_matches_other_encodings_and_drives_sgp4_to_0_ulp() {
    for fix in FIXTURES {
        let kvn = omm::parse_kvn(fix.kvn).unwrap_or_else(|e| panic!("{}: {e}", fix.name));
        let json = omm::parse_json(fix.json).unwrap_or_else(|e| panic!("{}: {e}", fix.name));

        assert_eq!(
            canonical(&kvn),
            canonical(&json),
            "{}: KVN and JSON disagree on orbital content",
            fix.name,
        );

        let from_omm =
            Satellite::from_omm(&json).unwrap_or_else(|e| panic!("{} JSON: {e}", fix.name));
        assert_same_propagation(
            &format!("{} [JSON]", fix.name),
            &from_omm,
            &kvn_satellite(fix),
        );
    }
}

#[test]
fn frozen_auto_parse_encode_output_hashes() {
    let cases: [FrozenOmmCase; 9] = [
        ("25544.kvn", FIXTURES[0].kvn, encode_kvn_fixture),
        ("25544.xml", FIXTURES[0].xml, encode_xml_fixture),
        ("25544.json", FIXTURES[0].json, encode_json_fixture),
        ("24876.kvn", FIXTURES[1].kvn, encode_kvn_fixture),
        ("24876.xml", FIXTURES[1].xml, encode_xml_fixture),
        ("24876.json", FIXTURES[1].json, encode_json_fixture),
        ("28884.kvn", FIXTURES[2].kvn, encode_kvn_fixture),
        ("28884.xml", FIXTURES[2].xml, encode_xml_fixture),
        ("28884.json", FIXTURES[2].json, encode_json_fixture),
    ];
    // The GP JSON records state no CCSDS_OMM_VERS, and the JSON writer states
    // none for them.
    let expected = [
        (
            "25544.kvn",
            "42a52d09a9a61ee5326f85e18255743f1711e1df3402c766d3133c36ecd5ebeb",
        ),
        (
            "25544.xml",
            "732a730d444213b3894b1c40893da31a7f707e578a3ff01fbaf241d91d59fe77",
        ),
        (
            "25544.json",
            "09d0fc2201d8939ca31d9a96009ebc56ecd0ab89c78d671c7fd912b216a928fa",
        ),
        (
            "24876.kvn",
            "e20c3848984d9df2556d78f43040092a352699cce5b9397da71a0615d271e51c",
        ),
        (
            "24876.xml",
            "943fadef4fe75501f8593e00756b3806fa31183e543ffd87a4f422b2ba02149f",
        ),
        (
            "24876.json",
            "a8f955c726d990dcf5df3be0e216dbac6aa63f01b8d5ee29d9c57682a2ab57df",
        ),
        (
            "28884.kvn",
            "f5a83dab177b5333b223fc3b3c2e439b2da6769258c7d978b43fcdf8cfe699e5",
        ),
        (
            "28884.xml",
            "90287ce38685ce36d6562ebc575c53ecc70efc7038dbce7db8aa276ed77a0ca9",
        ),
        (
            "28884.json",
            "34261b71d9c5183b37ef0b7a924018b9ecc58198477ed8924bc0622ee3996953",
        ),
    ];

    for (index, (name, input, encode)) in cases.into_iter().enumerate() {
        let parsed = omm::parse(input).unwrap_or_else(|error| panic!("{name}: {error}"));
        let output = encode(&parsed);
        let actual = sha256_hex(output.as_bytes());
        assert_eq!(actual, expected[index].1, "{name}");
    }
}

#[test]
fn gp_csv_matches_json_and_drives_sgp4_to_0_ulp() {
    let fix = &FIXTURES[0];
    let csv = omm::parse_csv(ISS_CSV).expect("ISS GP CSV parses");
    let json = omm::parse_json(fix.json).expect("ISS GP JSON parses");
    assert_eq!(
        canonical(&csv),
        canonical(&json),
        "CSV and JSON disagree on orbital content",
    );

    let from_csv = Satellite::from_omm(&csv).expect("ISS GP CSV initializes");
    assert_same_propagation("ISS (ZARYA) [CSV]", &from_csv, &kvn_satellite(fix));

    // The element set carries the TLE's elements, with B* as the OMM states
    // it (.17172E-3), as python-sgp4 takes it; the TLE's assumed-decimal
    // field, a mantissa scaled by a power of ten, decodes to another double.
    let (l1, l2) = tle_lines(fix.tle);

    let elements = csv.to_element_set().expect("CSV converts to element set");
    let tle_elements = sidereon_core::astro::tle::parse(&l1, &l2)
        .expect("ISS TLE parses")
        .elements
        .to_element_set()
        .expect("ISS TLE converts to element set");
    assert_eq!(elements.catalog_number, tle_elements.catalog_number);
    assert_eq!(
        elements.mean_motion_rev_per_day,
        tle_elements.mean_motion_rev_per_day
    );
    assert_eq!(elements.eccentricity, tle_elements.eccentricity);
    assert_eq!(elements.inclination_deg, tle_elements.inclination_deg);
    assert_eq!(
        elements.right_ascension_deg,
        tle_elements.right_ascension_deg
    );
    assert_eq!(
        elements.argument_of_perigee_deg,
        tle_elements.argument_of_perigee_deg
    );
    assert_eq!(elements.mean_anomaly_deg, tle_elements.mean_anomaly_deg);
    assert_eq!(elements.bstar.to_bits(), 0.17172e-3_f64.to_bits());
    assert_ne!(tle_elements.bstar.to_bits(), elements.bstar.to_bits());
    assert_eq!(
        elements.mean_motion_double_dot.map(f64::to_bits),
        tle_elements.mean_motion_double_dot.map(f64::to_bits)
    );
}
