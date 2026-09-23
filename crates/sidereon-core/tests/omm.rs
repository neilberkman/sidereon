#![cfg(sidereon_repo_tests)]
//! Authoritative OMM gate: an OMM's SGP4 epoch is python-sgp4's for the same
//! EPOCH, the encodings agree, and where python-sgp4 gives the OMM and the
//! matching TLE one epoch, the OMM drives SGP4 bit-identically to the TLE.
//!
//! OMM and TLE encode the same SGP4 mean elements. The TLE states its epoch as
//! a day of year with eight decimals; the OMM states it to the microsecond, and
//! python-sgp4 (`sgp4.omm.initialize`, which Skyfield uses) keeps the OMM's
//! own day fraction unless its day count has at most eight decimals. For the
//! ISS fixture the two agree and propagation from the OMM (`Satellite::from_omm`)
//! must match the TLE (`Satellite::from_tle`) to 0 ULP on every
//! position/velocity component; for NAVSTAR 43 and GALAXY 15 python-sgp4's OMM
//! epoch is 0.36 microseconds before and 0.23 microseconds after the TLE's.
//! Each encoding (KVN, XML, JSON) must parse to the same orbital content. The
//! committed fixtures are real CelesTrak GP data: each object's OMM in every
//! encoding plus its TLE, captured in one query so they share an epoch.

use sha2::{Digest, Sha256};
use sidereon_core::astro::omm::{self, Omm};
use sidereon_core::astro::sgp4::{MinutesSinceEpoch, Satellite};

struct Fixture {
    name: &'static str,
    kvn: &'static str,
    xml: &'static str,
    json: &'static str,
    tle: &'static str,
    /// python-sgp4 2.22 `jdsatepoch` and `jdsatepochF` for the OMM's EPOCH
    /// (`fixtures/omm/python_sgp4_epochs.json`).
    python_epoch: (u64, u64),
    /// Whether python-sgp4 gives the TLE the same epoch.
    tle_epoch_matches: bool,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "ISS (ZARYA) - near-Earth SGP4",
        kvn: include_str!("fixtures/omm/25544.kvn"),
        xml: include_str!("fixtures/omm/25544.xml"),
        json: include_str!("fixtures/omm/25544.json"),
        tle: include_str!("fixtures/omm/25544.tle"),
        python_epoch: (0x4142_c70c_4000_0000, 0x3fc8_4145_2f34_2018),
        tle_epoch_matches: true,
    },
    Fixture {
        name: "NAVSTAR 43 - deep-space SDP4 (12 h)",
        kvn: include_str!("fixtures/omm/24876.kvn"),
        xml: include_str!("fixtures/omm/24876.xml"),
        json: include_str!("fixtures/omm/24876.json"),
        tle: include_str!("fixtures/omm/24876.tle"),
        python_epoch: (0x4142_c70b_c000_0000, 0x3fca_2b0c_32bc_0000),
        tle_epoch_matches: false,
    },
    Fixture {
        name: "GALAXY 15 - deep-space SDP4 (geosynchronous)",
        kvn: include_str!("fixtures/omm/28884.kvn"),
        xml: include_str!("fixtures/omm/28884.xml"),
        json: include_str!("fixtures/omm/28884.json"),
        tle: include_str!("fixtures/omm/28884.tle"),
        python_epoch: (0x4142_c70b_c000_0000, 0x3fe6_ea19_fa27_8000),
        tle_epoch_matches: false,
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
const TSINCE_MINUTES: &[f64] = &[0.0, 10.0, 100.0, 720.0, 1440.0, 4320.0];

/// Assert that a `Satellite` built from an OMM has python-sgp4's epoch for it,
/// and, where python-sgp4 gives the matching TLE the same epoch, propagates
/// bit-identically to the `Satellite` built from that TLE.
fn assert_bit_identical(label: &str, fix: &Fixture, from_omm: &Satellite, from_tle: &Satellite) {
    let e_omm = from_omm.epoch_jd();
    assert_eq!(
        (e_omm.0.to_bits(), e_omm.1.to_bits()),
        fix.python_epoch,
        "{label}: epoch JD differs from python-sgp4 ({:?})",
        (e_omm.0, e_omm.1),
    );
    if !fix.tle_epoch_matches {
        let e_tle = from_tle.epoch_jd();
        assert_ne!(e_omm.1.to_bits(), e_tle.1.to_bits(), "{label}");
        return;
    }
    let e_tle = from_tle.epoch_jd();
    assert_eq!(
        (e_omm.0.to_bits(), e_omm.1.to_bits()),
        (e_tle.0.to_bits(), e_tle.1.to_bits()),
        "{label}: epoch JD differs (OMM {:?} vs TLE {:?})",
        (e_omm.0, e_omm.1),
        (e_tle.0, e_tle.1),
    );

    for &t in TSINCE_MINUTES {
        let p_omm = from_omm.propagate(MinutesSinceEpoch(t)).unwrap();
        let p_tle = from_tle.propagate(MinutesSinceEpoch(t)).unwrap();
        for axis in 0..3 {
            assert_eq!(
                p_omm.position[axis].to_bits(),
                p_tle.position[axis].to_bits(),
                "{label}: position[{axis}] differs at t={t} min (OMM {} vs TLE {})",
                p_omm.position[axis],
                p_tle.position[axis],
            );
            assert_eq!(
                p_omm.velocity[axis].to_bits(),
                p_tle.velocity[axis].to_bits(),
                "{label}: velocity[{axis}] differs at t={t} min (OMM {} vs TLE {})",
                p_omm.velocity[axis],
                p_tle.velocity[axis],
            );
        }
    }
}

#[test]
fn omm_drives_sgp4_bit_identically_to_matching_tle() {
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

        let (l1, l2) = tle_lines(fix.tle);
        let from_tle =
            Satellite::from_tle(&l1, &l2).unwrap_or_else(|e| panic!("{}: {e}", fix.name));

        for (enc, parsed) in [("KVN", &kvn), ("XML", &xml)] {
            let from_omm =
                Satellite::from_omm(parsed).unwrap_or_else(|e| panic!("{} {enc}: {e}", fix.name));
            assert_bit_identical(&format!("{} [{enc}]", fix.name), fix, &from_omm, &from_tle);
        }
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

        let (l1, l2) = tle_lines(fix.tle);
        let from_tle =
            Satellite::from_tle(&l1, &l2).unwrap_or_else(|e| panic!("{}: {e}", fix.name));
        let from_omm =
            Satellite::from_omm(&json).unwrap_or_else(|e| panic!("{} JSON: {e}", fix.name));
        assert_bit_identical(&format!("{} [JSON]", fix.name), fix, &from_omm, &from_tle);
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

    let (l1, l2) = tle_lines(fix.tle);
    let from_tle = Satellite::from_tle(&l1, &l2).expect("ISS TLE initializes");
    let from_csv = Satellite::from_omm(&csv).expect("ISS GP CSV initializes");
    assert_bit_identical("ISS (ZARYA) [CSV]", fix, &from_csv, &from_tle);

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
    assert_eq!(elements.bstar.to_bits(), tle_elements.bstar.to_bits());
    assert_eq!(
        elements.mean_motion_double_dot.map(f64::to_bits),
        tle_elements.mean_motion_double_dot.map(f64::to_bits)
    );
}
