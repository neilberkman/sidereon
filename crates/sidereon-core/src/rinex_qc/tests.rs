//! Focused first-slice RINEX QC tests.
//!
//! Broken inputs are deterministic mutations of small in-test products or
//! existing public fixtures under `tests/fixtures`. The QC layer delegates text
//! decoding to the existing RINEX and CRINEX modules; these tests cover the
//! typed rule/action surface added by this module.

use super::*;
use std::collections::BTreeMap;

fn header_line(body: &str, label: &str) -> String {
    format!("{body:<60}{label}")
}

fn obs_text(headers: &[String], body: &str) -> String {
    let mut lines = vec![
        header_line(
            "     3.05           OBSERVATION DATA    M (MIXED)",
            "RINEX VERSION / TYPE",
        ),
        header_line(
            "test                test                20200101 000000 UTC",
            "PGM / RUN BY / DATE",
        ),
        header_line("QC01", "MARKER NAME"),
        header_line("observer            agency", "OBSERVER / AGENCY"),
        header_line(
            "receiver            type                version",
            "REC # / TYPE / VERS",
        ),
        header_line("antenna             type", "ANT # / TYPE"),
        header_line(
            "        0.0000        0.0000        0.0000",
            "APPROX POSITION XYZ",
        ),
        header_line(
            "        0.0000        0.0000        0.0000",
            "ANTENNA: DELTA H/E/N",
        ),
        header_line("G    1 C1C", "SYS / # / OBS TYPES"),
        header_line(
            "  2020    01    01    00    00    0.0000000     GPS",
            "TIME OF FIRST OBS",
        ),
    ];
    lines.extend(headers.iter().cloned());
    lines.push(header_line("", "END OF HEADER"));
    lines.extend(body.lines().map(str::to_string));
    lines.join("\n")
}

fn gps_epoch(minute: u8, second: f64, value: &str) -> String {
    format!("> 2020 01 01 00 {minute:02}{second:11.7}  0  1\nG01{value}")
}

fn finding_code_counts(report: &LintReport) -> BTreeMap<&'static str, usize> {
    let mut counts = BTreeMap::new();
    for finding in &report.findings {
        *counts.entry(finding.code()).or_default() += 1;
    }
    counts
}

fn expected_counts(expected: &[(&'static str, usize)]) -> BTreeMap<&'static str, usize> {
    expected.iter().copied().collect()
}

fn assert_finding_counts(report: &LintReport, expected: &[(&'static str, usize)]) {
    assert_eq!(
        finding_code_counts(report),
        expected_counts(expected),
        "{:?}",
        report.findings
    );
}

fn nav_fixture() -> String {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_MN.rnx"
    );
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read NAV fixture {path}: {e}"))
}

#[test]
fn committed_obs_fixtures_have_pinned_lint_findings() {
    let fixtures = [
        ("ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx", &[][..]),
        (
            // Trim fixture retains a full-day TIME OF LAST OBS header on a
            // two-epoch body, so OBS-H08 is expected.
            "ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx",
            &[("OBS-H08", 1)][..],
        ),
        (
            // Same trim artifact as the CRINEX twin above.
            "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx",
            &[("OBS-H08", 1)][..],
        ),
        (
            // RCV CLOCK OFFS APPL is parsed as an unretained disclosure only.
            "PASA00ESP_R_20261201000_02H_30S_MO.rnx",
            &[("OBS-H90", 1)][..],
        ),
        (
            // RCV CLOCK OFFS APPL is parsed as an unretained disclosure only.
            "SCOA00FRA_R_20261201000_02H_30S_MO.rnx",
            &[("OBS-H90", 1)][..],
        ),
        (
            // The 120-epoch trim keeps full-file satellite totals and an
            // unretained receiver-clock-offset header.
            "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
            &[("OBS-H10", 1), ("OBS-H90", 1)][..],
        ),
        (
            // RCV CLOCK OFFS APPL is parsed as an unretained disclosure only.
            "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
            &[("OBS-H90", 1)][..],
        ),
        (
            // The 120-epoch ZIM2 trim keeps a full-day TIME OF LAST OBS header
            // and one unretained receiver-clock-offset header.
            "ZIM200CHE_R_20261330000_01H_30S_MO_120epoch.rnx",
            &[("OBS-H08", 1), ("OBS-H90", 1)][..],
        ),
        (
            // RINEX 2.11 mixed OBS has no GLONASS slot table and carries one
            // retained-header disclosure.
            "algo0010_2015001_v1_trim.crx",
            &[("OBS-H12", 8), ("OBS-H90", 1)][..],
        ),
        (
            // Same decoded RINEX 2.11 text as the CRINEX twin above.
            "algo0010_2015001_v1_trim.rnx",
            &[("OBS-H12", 8), ("OBS-H90", 1)][..],
        ),
    ];
    for (fixture, expected) in fixtures {
        let path = format!(
            "{}/tests/fixtures/obs/{fixture}",
            env!("CARGO_MANIFEST_DIR")
        );
        let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        let report = lint_obs_text(&text);
        assert_finding_counts(&report, expected);
    }
}

#[test]
fn obs_code_tables_accept_fixture_deltas_and_reject_bad_bands() {
    let accepted = [
        (GnssSystem::Glonass, "C3Q"),
        (GnssSystem::Glonass, "D3Q"),
        (GnssSystem::Glonass, "L3Q"),
        (GnssSystem::Glonass, "S3Q"),
        (GnssSystem::Qzss, "C5Q"),
        (GnssSystem::Qzss, "D5Q"),
        (GnssSystem::Qzss, "L5Q"),
        (GnssSystem::Qzss, "S5Q"),
        (GnssSystem::Gps, "C1N"),
        (GnssSystem::Sbas, "C5Q"),
        (GnssSystem::BeiDou, "C7Z"),
    ];
    for (system, code) in accepted {
        assert!(is_valid_obs_code(system, code, 3.05), "{system:?} {code}");
    }
    assert!(!is_valid_obs_code(GnssSystem::Gps, "C9C", 3.05));
}

#[test]
fn obs_lint_reports_time_interval_order_and_repair_fixes_them() {
    let headers = [
        header_line(
            "  2020    01    01    00    00    1.0000000     GPS",
            "TIME OF FIRST OBS",
        ),
        header_line("    60.000", "INTERVAL"),
    ];
    let body = [
        gps_epoch(1, 0.0, "  20000000.000  "),
        gps_epoch(0, 0.0, "  21000000.000  "),
        gps_epoch(0, 30.0, "  20000000.000  "),
        gps_epoch(0, 0.0, "  22000000.000  "),
    ]
    .join("\n");
    let obs = RinexObs::parse(&obs_text(&headers, &body)).expect("parse OBS");

    let report = lint_obs(&obs);
    assert_finding_counts(
        &report,
        &[
            ("OBS-H07", 1),
            ("OBS-H09", 1),
            ("OBS-B01", 2),
            ("OBS-B02", 1),
        ],
    );
    assert!(!report.is_clean());

    let repair = repair_obs(
        &obs,
        &RepairOptions {
            set_interval: true,
            ..RepairOptions::default()
        },
    );
    assert_eq!(
        repair
            .actions
            .iter()
            .map(|action| action.id)
            .collect::<Vec<_>>(),
        vec!["A3", "A4", "A6"]
    );
    assert_finding_counts(&repair.remaining, &[]);

    let second = repair_obs(&repair.repaired, &RepairOptions::default());
    assert!(second.actions.is_empty(), "{:?}", second.actions);
}

#[test]
fn zero_interval_is_standards_compatible_unavailable_metadata() {
    let headers = [header_line("     0.000", "INTERVAL")];
    let body = [
        gps_epoch(0, 0.0, "  20000000.000  "),
        gps_epoch(0, 30.0, "  20000001.000  "),
    ]
    .join("\n");
    let obs = RinexObs::parse(&obs_text(&headers, &body)).expect("parse OBS");

    let lint = lint_obs(&obs);
    assert_finding_counts(&lint, &[("OBS-H19", 1)]);
    let finding = lint
        .findings
        .iter()
        .find(|finding| finding.code() == "OBS-H19")
        .expect("unavailable INTERVAL finding");
    assert_eq!(finding.severity(), Severity::Info);
    assert!(finding.is_repairable());
    assert!(lint.is_clean());

    let unchanged = repair_obs(&obs, &RepairOptions::default());
    assert!(unchanged.actions.is_empty());
    assert_eq!(unchanged.repaired.header.interval_s, Some(0.0));

    let repaired = repair_obs(
        &obs,
        &RepairOptions {
            set_interval: true,
            ..RepairOptions::default()
        },
    );
    assert_eq!(repaired.repaired.header.interval_s, Some(30.0));
    assert_eq!(
        repaired
            .actions
            .iter()
            .map(|action| action.id)
            .collect::<Vec<_>>(),
        vec!["A6"]
    );
    assert_finding_counts(&repaired.remaining, &[]);
}

#[test]
fn explicit_interval_repair_removes_unresolved_zero_metadata() {
    let headers = [header_line("    -0.000", "INTERVAL")];
    let obs = RinexObs::parse(&obs_text(&headers, "")).expect("parse OBS");

    let unchanged = repair_obs(&obs, &RepairOptions::default());
    assert_eq!(unchanged.repaired.header.interval_s, Some(-0.0));
    assert!(unchanged.actions.is_empty());

    let repaired = repair_obs(
        &obs,
        &RepairOptions {
            set_interval: true,
            ..RepairOptions::default()
        },
    );
    assert_eq!(repaired.repaired.header.interval_s, None);
    assert_eq!(repaired.actions.len(), 1);
    assert_eq!(repaired.actions[0].id, "A6");
    assert!(repaired.actions[0].message.contains("unusable INTERVAL"));
    assert_finding_counts(&repaired.remaining, &[]);

    let repeated = repair_obs(
        &repaired.repaired,
        &RepairOptions {
            set_interval: true,
            ..RepairOptions::default()
        },
    );
    assert!(repeated.actions.is_empty());
}

#[test]
fn negative_and_nonfinite_intervals_are_errors_and_never_used_as_cadence() {
    let headers = [header_line("   -30.000", "INTERVAL")];
    let body = [
        gps_epoch(0, 0.0, "  20000000.000  "),
        gps_epoch(0, 30.0, "  20000001.000  "),
    ]
    .join("\n");
    let original = RinexObs::parse(&obs_text(&headers, &body)).expect("parse OBS");

    for invalid in [-30.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut obs = original.clone();
        obs.header.interval_s = Some(invalid);

        let lint = lint_obs(&obs);
        let finding = lint
            .findings
            .iter()
            .find(|finding| finding.code() == "OBS-H20")
            .expect("invalid INTERVAL finding");
        assert_eq!(finding.severity(), Severity::Error, "{invalid:?}");
        assert!(finding.is_repairable());
        assert!(!lint.is_clean());

        let unchanged = repair_obs(&obs, &RepairOptions::default());
        let retained = unchanged.repaired.header.interval_s.unwrap();
        if invalid.is_nan() {
            assert!(retained.is_nan());
        } else {
            assert_eq!(retained, invalid);
        }

        let repaired = repair_obs(
            &obs,
            &RepairOptions {
                set_interval: true,
                ..RepairOptions::default()
            },
        );
        assert_eq!(repaired.repaired.header.interval_s, Some(30.0));
        assert!(!repaired
            .remaining
            .findings
            .iter()
            .any(|finding| finding.code() == "OBS-H20"));
    }
}

#[test]
fn obs_lint_reports_time_scale_mismatch_and_repair_fixes_it() {
    // RINEX 3.05: TIME OF FIRST OBS defines the file time system, so it is
    // authoritative. A TIME OF LAST OBS declaring a different system is the
    // finding (OBS-H08), and repair rewrites it to match TIME OF FIRST OBS.
    let headers = [
        header_line(
            "  2020    01    01    00    00    0.0000000     UTC",
            "TIME OF FIRST OBS",
        ),
        header_line(
            "  2020    01    01    00    00   30.0000000     GPS",
            "TIME OF LAST OBS",
        ),
    ];
    let body = [
        gps_epoch(0, 0.0, "  20000000.000  "),
        gps_epoch(0, 30.0, "  20000001.000  "),
    ]
    .join("\n");
    let obs = RinexObs::parse(&obs_text(&headers, &body)).expect("parse OBS");

    let report = lint_obs(&obs);
    assert_finding_counts(&report, &[("OBS-H08", 1)]);

    let repair = repair_obs(&obs, &RepairOptions::default());
    assert_finding_counts(&repair.remaining, &[]);
    assert_eq!(
        repair
            .repaired
            .header
            .time_of_first_obs
            .map(|(_, scale)| scale),
        Some(TimeScale::Utc)
    );
    assert_eq!(
        repair
            .repaired
            .header
            .time_of_last_obs
            .map(|(_, scale)| scale),
        Some(TimeScale::Utc)
    );
}

#[test]
fn obs_drop_empty_satellite_record_is_opt_in() {
    let headers = [header_line(
        "  2020    01    01    00    00    0.0000000     GPS",
        "TIME OF FIRST OBS",
    )];
    let body = "> 2020 01 01 00 00  0.0000000  0  1\nG01";
    let obs = RinexObs::parse(&obs_text(&headers, body)).expect("parse OBS");

    let report = lint_obs(&obs);
    assert_finding_counts(&report, &[("OBS-B08", 1)]);

    let repair = repair_obs(
        &obs,
        &RepairOptions {
            drop_empty_records: true,
            ..RepairOptions::default()
        },
    );
    assert_eq!(
        repair
            .actions
            .iter()
            .map(|action| action.id)
            .collect::<Vec<_>>(),
        vec!["A7"]
    );
    assert_finding_counts(&repair.remaining, &[]);
}

#[test]
fn obs_repair_duplicate_epoch_keeps_first_satellite_row() {
    let headers = [header_line(
        "  2020    01    01    00    00    0.0000000     GPS",
        "TIME OF FIRST OBS",
    )];
    let body = [
        gps_epoch(0, 0.0, "  11111111.000  "),
        gps_epoch(0, 0.0, "  22222222.000  "),
    ]
    .join("\n");
    let obs = RinexObs::parse(&obs_text(&headers, &body)).expect("parse OBS");
    let repair = repair_obs(&obs, &RepairOptions::default());
    assert_eq!(repair.repaired.epochs.len(), 1);
    let g01 = GnssSatelliteId::new(GnssSystem::Gps, 1).expect("G01");
    assert_eq!(
        repair.repaired.epochs[0].sats[&g01][0].value,
        Some(11_111_111.0)
    );
    assert!(repair.actions[0].message.contains("G01"));
}

#[test]
fn glonass_slot_findings_are_per_satellite_and_check_channel_range() {
    let headers = [
        header_line(
            "  2020    01    01    00    00    0.0000000     GPS",
            "TIME OF FIRST OBS",
        ),
        header_line("R    1 C1C", "SYS / # / OBS TYPES"),
    ];
    let body = [
        "> 2020 01 01 00 00  0.0000000  0  1\nR01  20000000.000  ",
        "> 2020 01 01 00 00 30.0000000  0  1\nR01  20000001.000  ",
    ]
    .join("\n");
    let mut obs = RinexObs::parse(&obs_text(&headers, &body)).expect("parse OBS");
    let report = lint_obs(&obs);
    assert_finding_counts(&report, &[("OBS-H12", 1)]);

    obs.header.glonass_slots.insert(1, 99);
    let report = lint_obs(&obs);
    assert_finding_counts(&report, &[("OBS-H12", 1)]);
    assert!(matches!(
        report.findings.as_slice(),
        [Finding::ObsGlonassSlotIssue {
            issue: "invalid channel",
            ..
        }]
    ));
}

#[test]
fn obs_text_repair_guards_event_special_records() {
    let headers = [header_line(
        "  2020    01    01    00    00    0.0000000     GPS",
        "TIME OF FIRST OBS",
    )];
    let body = "> 2020 01 01 00 00  0.0000000  2  1\nCOMMENT";
    let text = obs_text(&headers, body);
    assert!(repair_obs_text(&text, &RepairOptions::default()).is_err());

    let repair = repair_obs_text(
        &text,
        &RepairOptions {
            drop_unsupported: true,
            ..RepairOptions::default()
        },
    )
    .expect("drop unsupported event records");
    assert_eq!(
        repair
            .actions
            .iter()
            .map(|action| action.id)
            .collect::<Vec<_>>(),
        vec!["OBS-B11"]
    );
}

#[test]
fn obs_text_repair_guards_unretained_header_records() {
    let headers = [header_line("payload", "UNSUPPORTED LABEL")];
    let text = obs_text(&headers, &gps_epoch(0, 0.0, "  20000000.000  "));
    let report = lint_obs_text(&text);
    assert_finding_counts(&report, &[("OBS-H90", 1)]);
    assert!(repair_obs_text(&text, &RepairOptions::default()).is_err());

    let repair = repair_obs_text(
        &text,
        &RepairOptions {
            drop_unsupported: true,
            ..RepairOptions::default()
        },
    )
    .expect("drop unsupported header");
    assert_eq!(
        repair
            .actions
            .iter()
            .map(|action| action.id)
            .collect::<Vec<_>>(),
        vec!["OBS-H90"]
    );
    assert_finding_counts(&repair.remaining, &[]);
}

#[test]
fn obs_writer_skips_unretained_header_labels() {
    let headers = [header_line("payload", "UNSUPPORTED LABEL")];
    let text = obs_text(&headers, &gps_epoch(0, 0.0, "  20000000.000  "));
    let obs = RinexObs::parse(&text).expect("parse OBS");
    assert_eq!(
        obs.header.unretained_header_labels,
        vec!["UNSUPPORTED LABEL".to_string()]
    );

    let serialized = obs.to_rinex_string();
    assert!(!serialized.contains("UNSUPPORTED LABEL"));
    let reparsed = RinexObs::parse(&serialized).expect("parse serialized OBS");
    assert!(reparsed.header.unretained_header_labels.is_empty());
}

#[test]
fn rinex4_epoch_extension_and_clock_offset_round_trip() {
    let text = [
        header_line(
            "     4.02           OBSERVATION DATA    M (MIXED)",
            "RINEX VERSION / TYPE",
        ),
        header_line("G    1 C1C", "SYS / # / OBS TYPES"),
        header_line("", "END OF HEADER"),
        "> 2026 01 02 03 04  5.0000000 12345  0  1  0.123456789012".to_string(),
        "G01  20000000.000  ".to_string(),
    ]
    .join("\n");
    let obs = RinexObs::parse(&text).expect("parse RINEX 4 OBS");
    assert_eq!(obs.epochs[0].epoch_picoseconds, Some(12345));
    assert_eq!(obs.epochs[0].rcv_clock_offset_s, Some(0.123456789012));
    let reparsed = RinexObs::parse(&obs.to_rinex_string()).expect("reparse RINEX 4 OBS");
    assert_eq!(reparsed, obs);
}

#[test]
fn obs_text_lint_decodes_crinex_before_linting() {
    let rnx_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx"
    );
    let crx_path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/obs/ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx"
    );
    let rnx = std::fs::read_to_string(rnx_path).expect("read RINEX fixture");
    let crx = std::fs::read_to_string(crx_path).expect("read CRINEX fixture");

    let rnx_report = lint_obs_text(&rnx);
    let crx_report = lint_obs_text(&crx);
    assert!(!rnx_report.decoded_from_crinex);
    assert!(crx_report.decoded_from_crinex);
    assert_finding_counts(&crx_report, &[("OBS-H08", 1)]);
    assert_finding_counts(&rnx_report, &[("OBS-H08", 1)]);
}

#[test]
fn nav_lint_and_repair_identical_duplicates_and_order() {
    let records = parse_nav(&nav_fixture()).expect("parse NAV fixture");
    assert!(records.len() >= 2);
    let damaged = vec![records[1], records[0], records[0]];

    let report = LintReport {
        findings: nav_findings(&damaged),
        decoded_from_crinex: false,
    };
    assert_finding_counts(&report, &[("NAV-B02", 1), ("NAV-B03", 1)]);

    let repair = repair_nav(&damaged, &RepairOptions::default());
    assert_eq!(
        repair
            .actions
            .iter()
            .map(|action| action.id)
            .collect::<Vec<_>>(),
        vec!["A11", "A12"]
    );
    assert_eq!(repair.records.len(), 2);
    assert_finding_counts(&repair.remaining, &[]);
}

#[test]
fn nav_text_lint_reports_header_findings_without_file_io() {
    let text = crate::rinex_nav::encode_nav(&[]);
    let report = lint_nav_text(&text);
    assert_finding_counts(&report, &[("NAV-H02", 1)]);
}

#[test]
fn nav_text_lint_total_parse_failure_does_not_emit_block_drop() {
    let report = lint_nav_text("not a RINEX NAV file\n");
    assert_finding_counts(&report, &[("NAV-H01", 1), ("NAV-H02", 1)]);
}

#[test]
fn nav_text_lint_reports_lenient_supported_block_drop_only_for_that_block() {
    let records = parse_nav(&nav_fixture()).expect("parse NAV fixture");
    let text = crate::rinex_nav::encode_nav(&records[..2]);
    let mut lines = text.lines().map(str::to_string).collect::<Vec<_>>();
    let damaged = lines
        .iter_mut()
        .find(|line| line.starts_with("C05 2020 06 24 22 00 00"))
        .expect("first C05 record");
    damaged.replace_range(23..42, " not-a-number      ");
    let text = lines.join("\n");

    let report = lint_nav_text(&text);
    assert_finding_counts(&report, &[("NAV-B01", 1), ("NAV-H02", 1)]);
    assert!(matches!(
        report.findings.as_slice(),
        [
            Finding::NavDroppedBlock { satellite, .. },
            Finding::NavLeapSecondsAbsent { .. },
        ] if satellite == "C05"
    ));
}

#[test]
fn teqc_algo0010_oracle_fixture_pins_reported_metrics() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qc/teqc_algo0010_2015001_v1_trim.json"
    );
    let raw = std::fs::read_to_string(path).expect("read TEQC oracle");
    let oracle: serde_json::Value = serde_json::from_str(&raw).expect("parse TEQC oracle");
    assert_eq!(oracle["provenance"]["tool_version"], "teqc  2019Feb25");
    assert_eq!(
        oracle["source_fixture"]["sha256"],
        "f2eae58b37fa267b6f64549de8eb1504473057b46fba1d039b9d2f063b536f22"
    );
    assert_eq!(oracle["summary"]["possible_observation_epochs"], 2);
    assert_eq!(oracle["summary"]["epochs_with_observations"], 2);
    assert_eq!(oracle["summary"]["complete_observations"], 39);
    assert_eq!(oracle["summary"]["gaps"].as_array().unwrap().len(), 0);
    assert_eq!(oracle["summary"]["moving_average_mp12_m"], 0.208985);
    assert_eq!(oracle["summary"]["moving_average_mp21_m"], 0.138536);
    assert_eq!(
        oracle["satellite_completeness"]
            .as_array()
            .expect("satellite completeness")
            .len(),
        20
    );
}

#[test]
fn nav_text_repair_guards_out_of_scope_records() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/nav/ESBC00DNK_R_20201770000_01D_RN.rnx"
    );
    let text = std::fs::read_to_string(path).expect("read GLONASS NAV fixture");
    assert!(repair_nav_text(&text, &RepairOptions::default()).is_err());

    let repair = repair_nav_text(
        &text,
        &RepairOptions {
            drop_unsupported: true,
            ..RepairOptions::default()
        },
    )
    .expect("drop unsupported NAV records");
    assert_eq!(
        repair
            .actions
            .iter()
            .map(|action| action.id)
            .collect::<Vec<_>>(),
        vec!["NAV-B06"]
    );
}

#[test]
fn repair_is_idempotent_and_byte_stable_on_committed_obs_fixtures() {
    let fixtures = [
        "ESBC00DNK_R_20201770000_01D_30S_MO_120epoch.rnx",
        "ESBC00DNK_R_20201770000_01D_30S_MO_trim.crx",
        "ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx",
        "PASA00ESP_R_20261201000_02H_30S_MO.rnx",
        "SCOA00FRA_R_20261201000_02H_30S_MO.rnx",
        "WTZR00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
        "WTZZ00DEU_R_20201770000_01D_30S_MO_120epoch.rnx",
        "ZIM200CHE_R_20261330000_01H_30S_MO_120epoch.rnx",
    ];
    let options = repair_oracle_options();
    for fixture in fixtures {
        let input = obs_fixture_text(fixture);
        let first = repair_obs_fixture_output(&input, &options);
        let repeated = repair_obs_fixture_output(&input, &options);
        assert_eq!(first, repeated, "{fixture} repair output changed");

        let second = repair_obs_fixture_output(&first, &options);
        assert_eq!(first, second, "{fixture} repair was not idempotent");
    }
}

#[test]
fn repair_is_idempotent_for_scheduled_fuzz_non_ascii_header_regression() {
    let input = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qc/",
        "rinex_qc_repair_roundtrip-crash-37835df362fe94c02f1151f1d3126fb19723a76c"
    ));
    let text = String::from_utf8_lossy(input);
    let obs = RinexObs::parse(&text).expect("parse scheduled-fuzz regression");
    let options = repair_oracle_options();

    let first = repair_obs(&obs, &options).repaired.to_rinex_string();
    assert!(
        first
            .bytes()
            .all(|byte| byte == b'\n' || byte == b' ' || byte.is_ascii_graphic()),
        "repaired RINEX must contain printable ASCII records"
    );
    let reparsed = RinexObs::parse(&first).expect("reparse repaired scheduled-fuzz regression");
    let second = repair_obs(&reparsed, &options).repaired.to_rinex_string();
    assert_eq!(second.as_bytes(), first.as_bytes());
}

#[test]
fn repair_is_total_for_scheduled_fuzz_zero_interval_regression() {
    let input = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qc/",
        "rinex_qc_repair_roundtrip-zero-interval-unknown.rnx"
    ));
    assert_eq!(
        sha256_hex(input),
        "5fb8bee02018479d441792d13aa9acaf45a27e03ddaaf67773a376809ba6d748"
    );
    assert_repair_roundtrip_is_total(input);
}

#[test]
fn repair_is_total_for_exact_scheduled_fuzz_crash_artifact() {
    let encoded = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qc/",
        "rinex_qc_repair_roundtrip-crash-207bfdaa15c81f885d3cc9c0bee4f82daabc20b3.hex"
    ));
    let input = decode_hex(encoded);
    assert_eq!(input.len(), 583);
    assert_eq!(
        sha256_hex(&input),
        "8091e235ba62ac613623e4d3c1856e8d0b9e04685584e42bed993fe8f9c2838a"
    );
    assert_repair_roundtrip_is_total(&input);
}

#[test]
fn scheduled_fuzz_over_width_obs_type_code_is_rejected_before_repair() {
    let encoded = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/qc/",
        "rinex_qc_repair_roundtrip-crash-eb54697fed8ee6948066ac7f5b35025242feed9b.hex"
    ));
    let input = decode_hex(encoded);
    assert_eq!(input.len(), 632);
    assert_eq!(
        sha256_hex(&input),
        "b85dbaf667a9867e5650a69dca8ce1fb2d26ccfccf249566d7fe7b01d93d5e0c"
    );

    // The header declares one SYS / # / OBS TYPES code that spans the whole
    // content area, then a second record for the same system. Such a header
    // cannot be serialized into 1X,A3 code fields: the re-emitted record
    // overran its 60 columns and re-parsed one code short of its own count,
    // which made repair -> write -> parse fail instead of round-tripping.
    let text = String::from_utf8_lossy(&input);
    let err = RinexObs::parse(&text).expect_err("an unrepresentable header must be rejected");
    assert!(
        matches!(err, crate::Error::Parse(ref message)
            if message.contains("SYS / # / OBS TYPES code")
                && message.contains("exceeds the A3 field width")),
        "{err}"
    );
}

#[test]
fn repair_round_trips_repeated_obs_type_records_for_one_system() {
    // The accepted shape of the same header: two SYS / # / OBS TYPES records
    // for one system accumulate into a single re-emitted record whose count
    // covers both codes, so repair stays byte-idempotent.
    // `obs_text` already declares `G    1 C1C`; this adds a second record for
    // the same system.
    let text = obs_text(&[header_line("G    1 L1C", "SYS / # / OBS TYPES")], "");
    let obs = RinexObs::parse(&text).expect("parse repeated obs-type records");
    assert_eq!(
        obs.obs_codes(crate::id::GnssSystem::Gps)
            .expect("GPS codes"),
        ["C1C".to_string(), "L1C".to_string()]
    );

    let options = repair_oracle_options();
    let first = repair_obs(&obs, &options).repaired.to_rinex_string();
    let reparsed = RinexObs::parse(&first).expect("reparse repaired obs-type records");
    assert_eq!(
        reparsed
            .obs_codes(crate::id::GnssSystem::Gps)
            .expect("reparsed GPS codes"),
        ["C1C".to_string(), "L1C".to_string()]
    );
    let second = repair_obs(&reparsed, &options).repaired.to_rinex_string();
    assert_eq!(second.as_bytes(), first.as_bytes());
}

#[test]
fn repair_is_idempotent_and_byte_stable_on_committed_nav_fixtures() {
    let fixtures = [
        "BRDC00GOP_R_20210010000_01D_MN.rnx",
        "ESBC00DNK_R_20201770000_01D_MN.rnx",
        "ESBC00DNK_R_20201770000_01D_RN.rnx",
        "KMS300DNK_R_20221591000_01H_MN.rnx",
    ];
    let options = repair_oracle_options();
    for fixture in fixtures {
        let input = nav_fixture_text(fixture);
        let first = repair_nav_fixture_output(&input, &options);
        let repeated = repair_nav_fixture_output(&input, &options);
        assert_eq!(first, repeated, "{fixture} repair output changed");

        let second = repair_nav_fixture_output(&first, &options);
        assert_eq!(first, second, "{fixture} repair was not idempotent");
    }
}

#[test]
fn obs_repair_header_only_changes_are_spp_bit_equivalent() {
    let input = obs_fixture_text("ESBC00DNK_R_20201770000_01D_30S_MO_trim.rnx");
    let mut damaged = RinexObs::parse(&input).expect("parse ESBC OBS");
    damaged.header.time_of_last_obs = Some((
        ObsEpochTime {
            year: 2020,
            month: 6,
            day: 25,
            hour: 23,
            minute: 59,
            second: 30.0,
        },
        TimeScale::Gpst,
    ));
    damaged.header.n_satellites = Some(999);

    let options = repair_oracle_options();
    let repair = repair_obs(&damaged, &options);
    assert_eq!(damaged.epochs, repair.repaired.epochs);
    assert_finding_counts(&repair.remaining, &[]);

    let broadcast = crate::rinex_nav::BroadcastStore::from_nav(&nav_fixture_text(
        "ESBC00DNK_R_20201770000_01D_MN.rnx",
    ))
    .expect("parse ESBC NAV");
    let before = solve_esbc_first_epoch(&broadcast, &damaged);
    let after = solve_esbc_first_epoch(&broadcast, &repair.repaired);
    assert_solution_bits_eq(&before, &after);
}

fn repair_oracle_options() -> RepairOptions {
    RepairOptions {
        set_interval: true,
        set_time_of_last_obs: true,
        set_obs_counts: true,
        drop_empty_records: true,
        drop_unsupported: true,
        ..RepairOptions::default()
    }
}

fn assert_repair_roundtrip_is_total(input: &[u8]) {
    let text = String::from_utf8_lossy(input);
    let obs = RinexObs::parse(&text).expect("parse scheduled-fuzz regression");

    let report = crate::observation_qc::observation_qc(&obs);
    assert_ne!(
        report.interval_source,
        crate::observation_qc::IntervalSource::Header
    );
    assert!(report
        .lint_findings
        .iter()
        .any(|finding| finding.code == "OBS-H19"));

    let options = repair_oracle_options();
    let repaired_text = repair_obs(&obs, &options).repaired.to_rinex_string();
    let reparsed = RinexObs::parse(&repaired_text).expect("reparse scheduled-fuzz regression");
    let _ = crate::observation_qc::observation_qc(&reparsed);
    let repeated_text = repair_obs(&reparsed, &options).repaired.to_rinex_string();
    assert_eq!(repeated_text.as_bytes(), repaired_text.as_bytes());
}

fn decode_hex(encoded: &[u8]) -> Vec<u8> {
    let digits: Vec<u8> = encoded
        .iter()
        .copied()
        .filter(|byte| !byte.is_ascii_whitespace())
        .collect();
    assert_eq!(digits.len() % 2, 0, "hex fixture must have complete bytes");
    digits
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16).expect("hex high nibble");
            let low = (pair[1] as char).to_digit(16).expect("hex low nibble");
            ((high << 4) | low) as u8
        })
        .collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

fn obs_fixture_text(fixture: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/obs/{fixture}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn nav_fixture_text(fixture: &str) -> String {
    let path = format!(
        "{}/tests/fixtures/nav/{fixture}",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
}

fn repair_obs_fixture_output(input: &str, options: &RepairOptions) -> String {
    let repair = repair_obs_text(input, options).expect("repair OBS fixture");
    if repair.decoded_from_crinex {
        repair_obs_to_crinex_string(&repair).expect("encode repaired CRINEX")
    } else {
        repair.repaired.to_rinex_string()
    }
}

fn repair_nav_fixture_output(input: &str, options: &RepairOptions) -> String {
    let repair = repair_nav_text(input, options).expect("repair NAV fixture");
    crate::rinex_nav::encode_nav(&repair.records)
}

fn solve_esbc_first_epoch(
    broadcast: &crate::rinex_nav::BroadcastStore,
    obs: &RinexObs,
) -> crate::spp::ReceiverSolution {
    let policy = crate::rinex_obs::SignalPolicy {
        codes: [(GnssSystem::Gps, vec!["C1C".to_string()])]
            .into_iter()
            .collect(),
    };
    let observations = crate::rinex_obs::pseudoranges(obs, &obs.epochs()[0], &policy)
        .expect("extract pseudoranges")
        .into_iter()
        .map(|(satellite_id, pseudorange_m)| crate::spp::Observation {
            satellite_id,
            pseudorange_m,
        })
        .collect();
    let approx = obs
        .header()
        .approx_position_m
        .expect("ESBC OBS approximate position");
    let inputs = crate::spp::SolveInputs {
        observations,
        t_rx_j2000_s: 646_315_200.0,
        t_rx_second_of_day_s: 0.0,
        day_of_year: 177.0,
        initial_guess: [approx[0], approx[1], approx[2], 0.0],
        corrections: crate::spp::Corrections {
            ionosphere: false,
            troposphere: true,
        },
        klobuchar: crate::spp::KlobucharCoeffs {
            alpha: [0.0; 4],
            beta: [0.0; 4],
        },
        beidou_klobuchar: None,
        galileo_nequick: None,
        sbas_iono: None,
        glonass_channels: BTreeMap::new(),
        met: crate::spp::SurfaceMet {
            pressure_hpa: 1013.25,
            temperature_k: 288.15,
            relative_humidity: 0.5,
        },
        robust: None,
    };
    crate::spp::solve(broadcast, &inputs, false).expect("solve SPP")
}

fn assert_solution_bits_eq(
    left: &crate::spp::ReceiverSolution,
    right: &crate::spp::ReceiverSolution,
) {
    assert_eq!(left.position.x_m.to_bits(), right.position.x_m.to_bits());
    assert_eq!(left.position.y_m.to_bits(), right.position.y_m.to_bits());
    assert_eq!(left.position.z_m.to_bits(), right.position.z_m.to_bits());
    assert_eq!(left.geodetic, right.geodetic);
    assert_eq!(left.rx_clock_s.to_bits(), right.rx_clock_s.to_bits());
    assert_eq!(left.rx_clock_drift_s_s, right.rx_clock_drift_s_s);
    assert_eq!(left.system_clocks_s.len(), right.system_clocks_s.len());
    for ((left_system, left_clock), (right_system, right_clock)) in left
        .system_clocks_s
        .iter()
        .zip(right.system_clocks_s.iter())
    {
        assert_eq!(left_system, right_system);
        assert_eq!(left_clock.to_bits(), right_clock.to_bits());
    }
    assert_eq!(left.dop, right.dop);
    assert_eq!(
        left.position_covariance
            .ecef_m2
            .iter()
            .flatten()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        right
            .position_covariance
            .ecef_m2
            .iter()
            .flatten()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        left.position_covariance
            .enu_m2
            .iter()
            .flatten()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        right
            .position_covariance
            .enu_m2
            .iter()
            .flatten()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        left.residuals_m
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        right
            .residuals_m
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
    assert_eq!(left.used_sats, right.used_sats);
    assert_eq!(left.rejected_sats, right.rejected_sats);
    assert_eq!(left.metadata, right.metadata);
}
