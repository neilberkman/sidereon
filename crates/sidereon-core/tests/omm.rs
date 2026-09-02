#![cfg(sidereon_repo_tests)]
//! Authoritative OMM gate: an OMM must drive SGP4 bit-identically to the
//! matching TLE for the same object/epoch, and the encodings must agree.
//!
//! OMM and TLE encode the same SGP4 mean elements, so propagating from an OMM
//! (parsed and bridged through `Satellite::from_omm`) must agree with the TLE
//! (`Satellite::from_tle`) to 0 ULP on every position/velocity component, across
//! near-Earth (SGP4) and deep-space (SDP4) objects. Each encoding (KVN, XML)
//! must parse to the same orbital content (cross-encoding identity). The
//! committed fixtures are real CelesTrak GP data: each object's OMM in every
//! encoding plus its TLE, captured in one query so they share an epoch. The TLE
//! is the correctness anchor; no invented truth.

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

/// Propagation offsets (minutes since epoch), including 0 and a deep-space-sized
/// span so both the SGP4 and SDP4 branches are exercised away from epoch.
const TSINCE_MINUTES: &[f64] = &[0.0, 10.0, 100.0, 720.0, 1440.0, 4320.0];

/// Assert that a `Satellite` built from an OMM propagates bit-identically to one
/// built from the matching TLE, including a bit-equal cached epoch.
fn assert_bit_identical(label: &str, from_omm: &Satellite, from_tle: &Satellite) {
    let e_omm = from_omm.epoch_jd();
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
            assert_bit_identical(&format!("{} [{enc}]", fix.name), &from_omm, &from_tle);
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
        assert_bit_identical(&format!("{} [JSON]", fix.name), &from_omm, &from_tle);
    }
}

#[test]
fn frozen_auto_parse_encode_output_hashes() {
    let cases: [FrozenOmmCase; 9] = [
        ("25544.kvn", FIXTURES[0].kvn, omm::encode_kvn),
        ("25544.xml", FIXTURES[0].xml, omm::encode_xml),
        ("25544.json", FIXTURES[0].json, omm::encode_json),
        ("24876.kvn", FIXTURES[1].kvn, omm::encode_kvn),
        ("24876.xml", FIXTURES[1].xml, omm::encode_xml),
        ("24876.json", FIXTURES[1].json, omm::encode_json),
        ("28884.kvn", FIXTURES[2].kvn, omm::encode_kvn),
        ("28884.xml", FIXTURES[2].xml, omm::encode_xml),
        ("28884.json", FIXTURES[2].json, omm::encode_json),
    ];
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
            "74d7f55c517507f16f1e06d1193582afeaeda9a1e11fa78436e64567860944bc",
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
            "3ee9b047cf83f0b358d5a3ad550d6c905293fa7e4b8858d096cd33388cb92f80",
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
            "a41879ecb0ffdd4bc0aa673e708a9126d9d1671e4cf6a45325943a7136d311a7",
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
    assert_bit_identical("ISS (ZARYA) [CSV]", &from_csv, &from_tle);

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
        elements.mean_motion_double_dot.to_bits(),
        tle_elements.mean_motion_double_dot.to_bits()
    );
}
