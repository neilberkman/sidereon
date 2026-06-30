//! GNSS constellation catalog parity tests.
//!
//! These mirror the reference Elixir module
//! (`Sidereon.GNSS.Constellation`) and assert the same parsed records and CSV
//! from the same cached CelesTrak JSON + NAVCEN HTML fixtures. The CelesTrak
//! fixture is a valid full `gps-ops` OMM array (the real feed shape, which the
//! core OMM parser validates in full); the identity fields it shares with the
//! reference fixture (OBJECT_NAME, OBJECT_ID, EPOCH, NORAD_CAT_ID) are
//! unchanged, so every record and CSV assertion matches the reference exactly.

use sidereon_core::astro::omm::{self, Omm};
use sidereon_core::constellation::{
    self, BoolStyle, ConstellationError, NavcenStatus, Record, RecordSource,
};
use sidereon_core::ephemeris::Sp3;
use sidereon_core::GnssSystem;

const GPS_OPS_JSON: &str = include_str!("fixtures/constellation/gps_ops_sample.json");
const NAVCEN_HTML: &[u8] = include_bytes!("fixtures/constellation/navcen_gps_sample.html");

const SP3: &str = "\
#cP2020  6 24  0  0  0.00000000       1 ORBIT IGS14 FIT  TST
## 2111 432000.00000000   900.00000000 59024 0.0000000000000
+    2   G03G32  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
++         0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
%c G  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%c cc cc ccc ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
%f  1.2500000  1.025000000  0.00000000000  0.000000000000000
%f  0.0000000  0.000000000  0.00000000000  0.000000000000000
%i    0    0    0    0      0      0      0      0         0
%i    0    0    0    0      0      0      0      0         0
/* TEST SP3-c FIXTURE
*  2020  6 24  0  0  0.00000000
PG03  15000.000000 -20000.000000   5000.000000    123.456789
PG32  -1234.567890   2345.678901  -3456.789012    100.000000
EOF
";

fn celestrak_omms() -> Vec<Omm> {
    let parsed = omm::parse_json_array(GPS_OPS_JSON).expect("parse gps-ops OMM array");
    assert_eq!(parsed.skipped, 0, "committed fixture must parse cleanly");
    parsed.omms
}

fn merged_records() -> Vec<Record> {
    let records = constellation::from_celestrak_omm(GnssSystem::Gps, &celestrak_omms())
        .expect("celestrak records");
    let statuses = constellation::parse_navcen(NAVCEN_HTML).expect("navcen statuses");
    constellation::merge_navcen(&records, &statuses)
}

fn record(prn: u16) -> Record {
    Record {
        system: GnssSystem::Gps,
        prn,
        svn: Some(prn + 60),
        norad_id: 40_000 + u32::from(prn),
        sp3_id: constellation::gnss_sp3_id(GnssSystem::Gps, prn),
        fdma_channel: None,
        active: true,
        usable: true,
        source: RecordSource::default(),
    }
}

#[test]
fn from_celestrak_omm_normalizes_gps_records() {
    let records =
        constellation::from_celestrak_omm(GnssSystem::Gps, &celestrak_omms()).expect("records");
    assert_eq!(
        records.iter().map(|r| r.prn).collect::<Vec<_>>(),
        vec![3, 5, 13, 19]
    );

    let prn3 = records.iter().find(|r| r.prn == 3).unwrap();
    assert_eq!(prn3.system, GnssSystem::Gps);
    assert_eq!(prn3.svn, None);
    assert_eq!(prn3.norad_id, 40294);
    assert_eq!(prn3.sp3_id, "G03");
    assert!(prn3.active);
    assert!(prn3.usable);
    let celestrak = prn3.source.celestrak.as_ref().unwrap();
    assert_eq!(celestrak.group, "gps-ops");
    assert_eq!(celestrak.block_type.as_deref(), Some("IIF"));
    // Identity/provenance fields shared with the reference fixture are preserved
    // verbatim (the rest of the OMM is enrichment the catalog never reads).
    assert_eq!(
        celestrak.object_name.as_deref(),
        Some("GPS BIIF-8  (PRN 03)")
    );
    assert_eq!(celestrak.object_id.as_deref(), Some("2014-068A"));
    assert_eq!(
        celestrak.epoch.as_deref(),
        Some("2026-06-01T23:12:11.987424")
    );
}

#[test]
fn from_celestrak_omm_rejects_record_without_prn() {
    let mut bad = celestrak_omms();
    bad[0].object_name = Some("GPS WITHOUT PRN".to_string());

    let err = constellation::from_celestrak_omm(GnssSystem::Gps, &bad).unwrap_err();
    assert_eq!(
        err,
        ConstellationError::MissingPrn(Some("GPS WITHOUT PRN".to_string()))
    );
}

#[test]
fn parse_navcen_extracts_svn_and_active_nanu_status() {
    let statuses = constellation::parse_navcen(NAVCEN_HTML).expect("statuses");
    assert_eq!(
        statuses.iter().map(|s| (s.prn, s.svn)).collect::<Vec<_>>(),
        vec![(3, Some(69)), (5, Some(50)), (13, Some(43)), (19, Some(59))]
    );

    let prn19: &NavcenStatus = statuses.iter().find(|s| s.prn == 19).unwrap();
    assert!(prn19.active_nanu);
    assert!(!prn19.usable);
    assert_eq!(prn19.nanu_type.as_deref(), Some("UNUSABLE"));
}

#[test]
fn merge_navcen_overlays_svn_and_usability() {
    let records = merged_records();
    assert_eq!(
        records
            .iter()
            .map(|r| (r.prn, r.svn, r.usable))
            .collect::<Vec<_>>(),
        vec![
            (3, Some(69), true),
            (5, Some(50), true),
            (13, None, true),
            (19, Some(59), false),
        ]
    );

    let prn3 = records.iter().find(|r| r.prn == 3).unwrap();
    assert_eq!(prn3.norad_id, 40294);
    assert_eq!(
        prn3.source.navcen.as_ref().unwrap().nanu_type.as_deref(),
        Some("FCSTSUMM")
    );

    let prn13 = records.iter().find(|r| r.prn == 13).unwrap();
    assert_eq!(prn13.norad_id, 68791);
    assert_eq!(
        prn13
            .source
            .celestrak
            .as_ref()
            .unwrap()
            .block_type
            .as_deref(),
        Some("III")
    );
    let conflict = prn13.source.navcen_conflict.as_ref().unwrap();
    assert_eq!(conflict.svn, Some(43));
    assert_eq!(conflict.block_type.as_deref(), Some("IIR"));
    assert!(prn13.source.navcen.is_none());
}

#[test]
fn to_csv_matches_reference_format() {
    assert_eq!(
        constellation::to_csv(&merged_records(), BoolStyle::Lower),
        "prn,norad_cat_id,active,sp3_id\n3,40294,true,G03\n5,35752,true,G05\n13,68791,true,G13\n19,28190,false,G19\n"
    );
}

#[test]
fn to_csv_title_booleans() {
    let titled = constellation::to_csv(&merged_records(), BoolStyle::Title);
    assert!(titled.contains("3,40294,True,G03"));
    assert!(titled.contains("19,28190,False,G19"));

    let lower = constellation::to_csv(&merged_records(), BoolStyle::Lower);
    assert!(lower.contains("3,40294,true,G03"));
    assert!(lower.contains("19,28190,false,G19"));
}

#[test]
fn validate_reports_duplicates_and_inactive() {
    let records = vec![
        Record {
            svn: Some(69),
            ..record(3)
        },
        Record {
            svn: Some(70),
            active: false,
            ..record(3)
        },
    ];
    let mut records = records;
    records[0].norad_id = 40294;
    records[1].norad_id = 40294;

    let report = constellation::validate(&records);
    assert_eq!(report.duplicate_prns, vec![(GnssSystem::Gps, 3)]);
    assert_eq!(report.duplicate_norad_ids, vec![40294]);
    assert_eq!(report.inactive_unusable_prns, vec![(GnssSystem::Gps, 3)]);
    assert!(!constellation::is_valid(&report));
}

#[test]
fn validate_against_sp3_product() {
    let sp3 = Sp3::parse(SP3.as_bytes()).expect("parse sp3");
    assert_eq!(
        sp3.header
            .satellites
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>(),
        vec!["G03", "G32"]
    );

    let report = constellation::validate_against_sp3(&merged_records(), &sp3);
    assert_eq!(report.missing_sp3_ids, vec!["G05", "G13"]);
    assert_eq!(report.extra_sp3_ids, vec!["G32"]);
    assert_eq!(report.inactive_unusable_prns, vec![(GnssSystem::Gps, 19)]);
    assert!(!constellation::is_valid(&report));
}

#[test]
fn validate_against_plain_sp3_id_list() {
    let report = constellation::validate_against_sp3_ids(&merged_records(), &["G03", "G05", "G13"]);
    assert!(report.missing_sp3_ids.is_empty());
    assert!(report.extra_sp3_ids.is_empty());
    assert_eq!(report.inactive_unusable_prns, vec![(GnssSystem::Gps, 19)]);
}

#[test]
fn validate_strict_gate() {
    let clean = vec![
        Record {
            svn: Some(69),
            norad_id: 40_294,
            ..record(3)
        },
        Record {
            svn: Some(50),
            norad_id: 35_752,
            ..record(5)
        },
    ];

    assert!(constellation::validate_against_sp3_ids_strict(&clean, &["G03", "G05"]).is_ok());

    let err = constellation::validate_against_sp3_ids_strict(&clean, &["G03"]).unwrap_err();
    let message = err.to_string();
    assert!(message.contains("missing_sp3_ids"), "{message}");
    assert!(message.contains("G05"), "{message}");
}

#[test]
fn diff_reports_no_change_for_identical_snapshots() {
    let previous = vec![record(3), record(5)];
    let reversed = vec![record(5), record(3)];
    let diff = constellation::diff(&previous, &reversed);
    assert!(!constellation::changed(&diff));
}

#[test]
fn diff_reports_added_and_removed_sorted() {
    let added = constellation::diff(&[record(3)], &[record(9), record(3), record(5)]);
    assert_eq!(
        added.added.iter().map(|r| r.prn).collect::<Vec<_>>(),
        vec![5, 9]
    );
    assert!(added.removed.is_empty());

    let removed = constellation::diff(&[record(9), record(3), record(5)], &[record(3)]);
    assert_eq!(
        removed.removed.iter().map(|r| r.prn).collect::<Vec<_>>(),
        vec![5, 9]
    );
    assert!(removed.added.is_empty());
}

#[test]
fn diff_reports_field_changes() {
    let mut prev = record(13);
    prev.norad_id = 28_190;
    let mut curr = record(13);
    curr.norad_id = 68_791;
    let diff = constellation::diff(&[prev], &[curr]);
    assert_eq!(diff.norad_reassigned.len(), 1);
    let change = &diff.norad_reassigned[0];
    assert_eq!(
        (change.system, change.prn, change.from, change.to),
        (GnssSystem::Gps, 13, 28_190, 68_791)
    );

    let mut prev = record(3);
    prev.svn = Some(69);
    prev.sp3_id = "G03".to_string();
    prev.usable = true;
    let mut curr = record(3);
    curr.svn = Some(70);
    curr.sp3_id = "G33".to_string();
    curr.usable = false;
    let diff = constellation::diff(&[prev], &[curr]);
    assert_eq!(diff.svn_changed[0].from, Some(69));
    assert_eq!(diff.svn_changed[0].to, Some(70));
    assert_eq!(diff.sp3_id_changed[0].to, "G33");
    assert!(diff.usability_changed[0].from);
    assert!(!diff.usability_changed[0].to);
}
