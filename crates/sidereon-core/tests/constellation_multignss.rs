//! Multi-system GNSS constellation catalog tests (campaign item A2).
//!
//! These exercise the per-system identity adapters that [`from_celestrak_omm`]
//! dispatches on for Galileo, GLONASS, BeiDou, and QZSS, using real CelesTrak
//! OMM snapshots committed under `tests/fixtures/constellation/`. Each test
//! asserts the per-system identity recovered from the OMM (PRN/slot, SP3 id,
//! NORAD), and for GLONASS the FDMA frequency channel cross-checked against the
//! published IGS/MCC slot-channel table.
//!
//! Fixture provenance (point-in-time CelesTrak GP/OMM snapshots, open published
//! JSON, fetched 2026-06-27 from
//! `https://celestrak.org/NORAD/elements/gp.php?GROUP=<group>&FORMAT=json`; each
//! non-GPS sample is a real unmodified subset of full OMM records):
//!   * `gps_ops_sample.json` (group `gps-ops`): pre-existing GPS sample.
//!   * `galileo_sample.json` (`galileo`): 6 satellites with assigned SVIDs (IOV+FOC).
//!   * `glonass_ops_sample.json` (`glo-ops`): the 18 operational satellites present
//!     in both the feed and the slot table.
//!   * `beidou_sample.json` (`beidou`): 6 satellites spanning GEO/IGSO/MEO, BDS-2/3.
//!   * `qzss_sample.json` (QZSS members of `gnss`): the 5 `QZSS/PRN` satellites.
//!   * `navcen_gps_sample.html`: NAVCEN GPS constellation status (GPS-only overlay).
//!
//! Derived identity tables in `src/constellation.rs`: Galileo GSAT->SVID
//! (`galileo_prn_for_gsat`) from EU GNSS Service Centre Galileo metadata,
//! cross-checked against the 2026-06-24 IGS broadcast nav E-PRN set; GLONASS
//! number->slot (`glonass_slot_for_number`) from the IAC operational constellation
//! (~2025-2026), regenerate with `glonass_ops_sample.json`; GLONASS slot->FDMA
//! channel (`glonass_fdma_channel`) is a bit-exact dual-sourced golden, identical
//! between the UNB GLONASS Constellation Status table
//! (https://gge.ext.unb.ca/Resources/GLONASSConstellationStatus.txt) and the
//! 2026-06-24 IGS daily merged broadcast nav frequency-number field.
//!
//! [`from_celestrak_omm`]: sidereon_core::constellation::from_celestrak_omm

use sidereon_core::astro::omm::{self, Omm};
use sidereon_core::constellation::{self, ConstellationError, Record};
use sidereon_core::GnssSystem;

const GALILEO_JSON: &str = include_str!("fixtures/constellation/galileo_sample.json");
const GLONASS_JSON: &str = include_str!("fixtures/constellation/glonass_ops_sample.json");
const BEIDOU_JSON: &str = include_str!("fixtures/constellation/beidou_sample.json");
const QZSS_JSON: &str = include_str!("fixtures/constellation/qzss_sample.json");

fn omms(json: &str) -> Vec<Omm> {
    let parsed = omm::parse_json_array(json).expect("parse OMM array");
    assert_eq!(parsed.skipped, 0, "committed fixture must parse cleanly");
    parsed.omms
}

fn records(system: GnssSystem, json: &str) -> Vec<Record> {
    constellation::from_celestrak_omm(system, &omms(json)).expect("records")
}

fn identity(records: &[Record]) -> Vec<(u16, &str, u32)> {
    records
        .iter()
        .map(|r| (r.prn, r.sp3_id.as_str(), r.norad_id))
        .collect()
}

#[test]
fn galileo_resolves_svid_from_gsat_build_id() {
    // Galileo OMM names carry the GSATdddd build id, not the SVID/PRN; the PRN
    // is resolved from the published GSAT->SVID table. Note GSAT0210 is
    // "GALILEO 13" yet SVID E01 - the name's nickname is not the PRN.
    let recs = records(GnssSystem::Galileo, GALILEO_JSON);
    assert_eq!(
        identity(&recs),
        vec![
            (2, "E02", 41549),  // GSAT0211 (GALILEO 14)
            (7, "E07", 41859),  // GSAT0207 (GALILEO 15)
            (10, "E10", 49810), // GSAT0224 (GALILEO 28)
            (11, "E11", 37846), // GSAT0101 (GALILEO-PFM, IOV)
            (26, "E26", 40544), // GSAT0203 (GALILEO 7)
            (34, "E34", 49809), // GSAT0223 (GALILEO 27)
        ]
    );
    // Each record is GAL, no FDMA channel, and IOV/FOC generation is captured.
    let pfm = recs.iter().find(|r| r.prn == 11).unwrap();
    assert_eq!(pfm.system, GnssSystem::Galileo);
    assert_eq!(pfm.fdma_channel, None);
    assert_eq!(
        pfm.source.celestrak.as_ref().unwrap().block_type.as_deref(),
        Some("IOV")
    );
    assert_eq!(
        recs.iter()
            .find(|r| r.prn == 2)
            .unwrap()
            .source
            .celestrak
            .as_ref()
            .unwrap()
            .block_type
            .as_deref(),
        Some("FOC")
    );
}

#[test]
fn galileo_rejects_unknown_gsat() {
    // A GSAT with no SVID assigned in the table is unresolvable.
    let mut bad = omms(GALILEO_JSON);
    bad[0].object_name = Some("GSAT9999 (GALILEO 99)".to_string());
    assert_eq!(
        constellation::from_celestrak_omm(GnssSystem::Galileo, &bad).unwrap_err(),
        ConstellationError::MissingPrn(Some("GSAT9999 (GALILEO 99)".to_string()))
    );
}

#[test]
fn beidou_resolves_prn_from_parenthesized_group() {
    let recs = records(GnssSystem::BeiDou, BEIDOU_JSON);
    assert_eq!(
        identity(&recs),
        vec![
            (1, "C01", 44231),  // BEIDOU-2 G8
            (6, "C06", 36828),  // BEIDOU-2 IGSO-1
            (19, "C19", 43001), // BEIDOU-3 M1
            (20, "C20", 43002), // BEIDOU-3 M2
            (21, "C21", 43208), // BEIDOU-3 M3
            (30, "C30", 43246), // BEIDOU-3 M10
        ]
    );
    let bds2 = recs.iter().find(|r| r.prn == 1).unwrap();
    assert_eq!(
        bds2.source
            .celestrak
            .as_ref()
            .unwrap()
            .block_type
            .as_deref(),
        Some("BDS-2")
    );
    let bds3 = recs.iter().find(|r| r.prn == 19).unwrap();
    assert_eq!(
        bds3.source
            .celestrak
            .as_ref()
            .unwrap()
            .block_type
            .as_deref(),
        Some("BDS-3")
    );
    assert!(recs.iter().all(|r| r.fdma_channel.is_none()));
}

#[test]
fn qzss_slot_is_broadcast_prn_minus_192() {
    // Cross-checked against the J-slots present in the 2026-06-24 IGS broadcast
    // navigation file: {194->J02, 199->J07, 195->J03, 196->J04, 200->J08}.
    let recs = records(GnssSystem::Qzss, QZSS_JSON);
    assert_eq!(
        identity(&recs),
        vec![
            (2, "J02", 42738), // QZS-2  (broadcast PRN 194)
            (3, "J03", 42965), // QZS-4  (broadcast PRN 195)
            (4, "J04", 49336), // QZS-1R (broadcast PRN 196)
            (7, "J07", 42917), // QZS-3  (broadcast PRN 199)
            (8, "J08", 62876), // QZS-6  (broadcast PRN 200)
        ]
    );
    assert!(recs.iter().all(|r| r.system == GnssSystem::Qzss));
}

#[test]
fn glonass_resolves_slot_and_fdma_channel() {
    let recs = records(GnssSystem::Glonass, GLONASS_JSON);

    // All 18 operational satellites in the snapshot resolve to a slot 1..=24.
    assert_eq!(recs.len(), 18);
    assert!(recs.iter().all(|r| (1..=24).contains(&r.prn)));

    // Spot-check the GLONASS-number -> slot -> SP3 id -> NORAD chain.
    let by_norad = |norad: u32| recs.iter().find(|r| r.norad_id == norad).unwrap();
    assert_eq!(
        (by_norad(36111).prn, by_norad(36111).sp3_id.as_str()),
        (1, "R01")
    ); // num 730
    assert_eq!(
        (by_norad(32393).prn, by_norad(32393).sp3_id.as_str()),
        (13, "R13")
    ); // num 721
    assert_eq!(
        (by_norad(54377).prn, by_norad(54377).sp3_id.as_str()),
        (16, "R16")
    ); // num 761

    // FDMA channel is filled for every GLONASS record, and matches the
    // published IGS/MCC slot-channel golden via glonass_fdma_channel(slot).
    for r in &recs {
        assert_eq!(
            r.fdma_channel,
            constellation::glonass_fdma_channel(r.prn),
            "slot R{:02} channel",
            r.prn
        );
        assert!(r.fdma_channel.is_some());
    }
    assert_eq!(by_norad(36111).fdma_channel, Some(1)); // R01 -> +1
    assert_eq!(by_norad(32393).fdma_channel, Some(-2)); // R13 -> -2
    assert_eq!(by_norad(54377).fdma_channel, Some(-1)); // R16 -> -1
}

#[test]
fn glonass_fdma_channels_match_published_table() {
    // The bit-exact golden: the published IGS/MCC slot<->channel assignment,
    // verified identical between the UNB/IAC table and the 2026-06-24 IGS merged
    // broadcast navigation file. Antipodal slots (k, k+4 within a plane group)
    // share a channel.
    let golden: [(u16, i8); 24] = [
        (1, 1),
        (2, -4),
        (3, 5),
        (4, 6),
        (5, 1),
        (6, -4),
        (7, 5),
        (8, 6),
        (9, -2),
        (10, -7),
        (11, 0),
        (12, -1),
        (13, -2),
        (14, -7),
        (15, 0),
        (16, -1),
        (17, 4),
        (18, -3),
        (19, 3),
        (20, 2),
        (21, 4),
        (22, -3),
        (23, 3),
        (24, 2),
    ];
    for (slot, channel) in golden {
        assert_eq!(
            constellation::glonass_fdma_channel(slot),
            Some(channel),
            "R{slot:02}"
        );
    }
    // Channels stay within the GLONASS FDMA range and only 14 distinct values
    // are used across 24 slots (antipodal reuse).
    assert!(golden.iter().all(|(_, k)| (-7..=6).contains(k)));
    let mut distinct: Vec<i8> = golden.iter().map(|(_, k)| *k).collect();
    distinct.sort_unstable();
    distinct.dedup();
    assert_eq!(distinct.len(), 12);
    // Slots outside 1..=24 have no assignment.
    assert_eq!(constellation::glonass_fdma_channel(0), None);
    assert_eq!(constellation::glonass_fdma_channel(25), None);
}

#[test]
fn multignss_catalog_validates_against_multi_system_sp3_product() {
    // A combined catalog (GPS + Galileo) compared to a multi-GNSS product only
    // flags the systems it actually covers: the extra GLONASS/BeiDou ids are not
    // reported, but a missing GPS id and an extra Galileo id are.
    let mut catalog = records(GnssSystem::Galileo, GALILEO_JSON);
    catalog.extend(records(GnssSystem::BeiDou, BEIDOU_JSON));

    let product = [
        "E02", "E07", "E10", "E11", "E26", // all catalog Galileo ids present
        "C01", "C06", "C19", "C20", "C21", // BeiDou: C30 missing, no extras
        "R01", "G05", "J02", // other systems - must be ignored
    ];
    let report = constellation::validate_against_sp3_ids(&catalog, &product);
    assert_eq!(report.missing_sp3_ids, vec!["C30", "E34"]);
    assert!(
        report.extra_sp3_ids.is_empty(),
        "{:?}",
        report.extra_sp3_ids
    );
}

#[test]
fn validate_keys_duplicates_by_system_not_bare_prn() {
    // The same bare PRN across two systems (GPS PRN 2 and Galileo PRN 2) is a
    // legitimate multi-system catalog, not a duplicate.
    let mut catalog = records(GnssSystem::Galileo, GALILEO_JSON); // includes E02
    catalog.push(Record {
        system: GnssSystem::Gps,
        prn: 2,
        svn: None,
        norad_id: 99_999,
        sp3_id: constellation::gnss_sp3_id(GnssSystem::Gps, 2),
        fdma_channel: None,
        active: true,
        usable: true,
        source: Default::default(),
    });

    let report = constellation::validate(&catalog);
    assert!(
        report.duplicate_prns.is_empty(),
        "cross-system PRN must not be a duplicate: {:?}",
        report.duplicate_prns
    );

    // A genuine same-system collision is still caught, keyed by (system, prn).
    catalog.push(Record {
        system: GnssSystem::Gps,
        prn: 2,
        svn: None,
        norad_id: 88_888,
        sp3_id: constellation::gnss_sp3_id(GnssSystem::Gps, 2),
        fdma_channel: None,
        active: true,
        usable: true,
        source: Default::default(),
    });
    let report = constellation::validate(&catalog);
    assert_eq!(report.duplicate_prns, vec![(GnssSystem::Gps, 2)]);
}

#[test]
fn diff_reports_fdma_channel_correction() {
    // A GLONASS channel correction on a held slot must surface in the diff.
    let previous = records(GnssSystem::Glonass, GLONASS_JSON);
    let mut current = previous.clone();
    let slot = current[0].prn;
    let corrected = current[0].fdma_channel.map(|k| k + 1);
    current[0].fdma_channel = corrected;

    let diff = constellation::diff(&previous, &current);
    assert!(constellation::changed(&diff));
    assert_eq!(diff.fdma_channel_changed.len(), 1);
    let change = &diff.fdma_channel_changed[0];
    assert_eq!(change.system, GnssSystem::Glonass);
    assert_eq!(change.prn, slot);
    assert_eq!(change.to, corrected);

    // No spurious change when nothing moved.
    assert!(constellation::diff(&previous, &previous)
        .fdma_channel_changed
        .is_empty());
}
