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
